/* Copyright (c) 2017 */

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

bool MkldnnLayer::init(
  const LayerMap& layerMap, const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  if (!Layer::init(layerMap, parameterMap)) return false;

  size_t sz = inputLayers_.size();
  // buffers
  botDatas_.resize(sz, nullptr);
  botDiffs_.resize(sz, nullptr);
  botDims_.resize(sz, {0});
  botFmt_.resize(sz, mkldnn::memory::format::nchw);
  wgtDims_.resize(sz, {0});
  wgtFmt_.resize(sz, mkldnn::memory::format::format_undef);
  biasDims_.resize(sz, {0});
  biasFmt_.resize(sz, mkldnn::memory::format::x);
  topDims_ = {0};
  topFmt_ = mkldnn::memory::format::nchw;

  // image sizes
  bs_ = 0;
  oc_ = 0;
  ih_.resize(sz, 0);
  iw_.resize(sz, 0);
  oh_.resize(sz, 0);
  ow_.resize(sz, 0);
  ic_.resize(sz, 0);

  return initDnn(layerMap, parameterMap);
}

void MkldnnLayer::clearAllDnnCvtFlags() {
  for (size_t i = 0; i < botDatas_.size(); ++i) {
    if (botDatas_[i]) botDatas_[i]->clearCvtFlag();
    if (botDiffs_[i]) botDiffs_[i]->clearCvtFlag();
  }
  if (topData_) topData_->clearCvtFlag();
  if (topDiff_) topDiff_->clearCvtFlag();
  if (topDiffBwdWgt_) topDiffBwdWgt_->clearCvtFlag();
}

void MkldnnLayer::prepareTopDiff() {
  sumTopDiffs_ = nullptr;
  if (nextLayers_.size() <= 1)
    return;
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  std::vector<mkldnn::memory::primitive_desc> srcPDs;
  std::vector<std::shared_ptr<mkldnn::memory::desc>> prvMDs;
  std::vector<mkldnn::primitive::at> srcMems;
  std::vector<double> scales;
  CHECK_EQ(nextLayers_.size(), topDiffBuffers_.size());
  for (size_t i = 0; i < topDiffBuffers_.size(); ++i) {
    // 1. create buffers and init user
    real* diff = dnnOutGrads_[i]->getData();
    topDiffBuffers_[i].reset(new MkldnnBuffer());
    topDiffBuffers_[i]->initUser(diff, topDims_, topFmt_, eg);
    // 2. use private MD when init user if has
    // TODO(TJ): any improvment if prvs are different format?
    const std::shared_ptr<mkldnn::memory::desc>& prvMD = getTopDiffMD(i);
    if (prvMD) {
      topDiffBuffers_[i]->resetUser(diff, *prvMD, eg);
      prvMDs.push_back(prvMD);
    }
    // 3. init Intl with empty cvt
    topDiffBuffers_[i]->initCvt();
    CHECK(i == 0 || (topDiffBuffers_[i-1]->getIntlSize()
      == topDiffBuffers_[i]->getIntlSize())) << "All size should be equal";
    // 4. save buffers
    scales.push_back(1.0);  // no scale
    srcPDs.push_back(topDiffBuffers_[i]->getIntlPD());
    srcMems.push_back(*(topDiffBuffers_[i]->getIntlMem()));
  }
  if (prvMDs.size() > 0 && prvMDs.size() != nextLayers_.size()) {
    LOG(INFO) << "prvMDs.size() != nextLayers_.size(): " << prvMDs.size()
      << " vs " << nextLayers_.size();
    LOG(INFO) << "Next layers: ";
    for (size_t i = 0; i < nextLayers_.size(); ++i) {
      LOG(INFO) << nextLayers_[i]->getName()
        << ", type: " << nextLayers_[i]->getType();
    }
    LOG(FATAL)  << "Do not support mixed layer type inside branch";
  }
  // 5. create sum PD
  std::shared_ptr<mkldnn::sum::primitive_desc> sumPD;
  sumPD.reset(new mkldnn::sum::primitive_desc(
    MkldnnBuffer::getMD(topDims_), scales, srcPDs));
  // 6. init the buffer of result
  tmpDiff_.reset(new MkldnnBuffer());
  real *topDiffData = getDnnOutputGrad()->getData();
  tmpDiff_->initUser(topDiffData, sumPD->dst_primitive_desc());
  tmpDiff_->initCvt();
  // change the first intl MD
  topDiffMDs_[0].reset(new mkldnn::memory::desc(tmpDiff_->getIntlMD()));
  // 7. create sum handle
  sumTopDiffs_.reset(
    new mkldnn::sum(*sumPD, srcMems, *(tmpDiff_->getIntlMem())));
}

void MkldnnLayer::forward(PassType passType) {
  Layer::forward(passType);

  // reshape if batch size changes
  if (bs_ == getInput(0).getBatchSize()) {
    // choose to clear top data or top diff
    clearDataDiff();
  } else {
    if (prepareOnce_) {
      topDiffBuffers_.resize(nextLayers_.size(), nullptr);
      dnnOutGrads_.resize(nextLayers_.size(), nullptr);
      for (size_t i = 0; i < nextLayers_.size(); ++i) {
        topDiffMDs_.push_back(nullptr);
        dnnOutIdxMap_[nextLayers_[i]->getName()] = i;
      //  LOG(INFO)<<"next name:" << nextLayers_[i]->getName();
      }
      if (nextLayers_.size() > 0 && topDiffMDs_.size() > nextLayers_.size()) {
        // in base layer init will add one nullptr for PASS_grad check
        // so remove the redundant one
        topDiffMDs_.pop_back();
        CHECK_EQ(topDiffMDs_.size(), nextLayers_.size());
      } else {
        CHECK_EQ(topDiffMDs_.size() - 1, nextLayers_.size());
      }
      // this function can work only after all layers init done
      // and should be called only once
      initDnnflags();
      prepareOnce_ = false;
    }

    bs_ = getInput(0).getBatchSize();
    VLOG(1) << "reset forward batch size to " << bs_
      << " of mkldnn layer: " << getName();

    // reshape the input and output size
    REGISTER_TIMER_INFO("mkldnn_ResetDnnTimer", getName().c_str());
    reshape();
    printInfo();

    // mkldnn init or reset forward
    resetOutput(bs_, getSize());
    resetDnnFwd(passType);

    // print the data flow
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
      // TODO(TJ): consider multi input
      if (botDatas_[i] && topData_)
        VLOG(1) << "data format flow --- "
          << DNN_FMTS[botDatas_[i]->getUserFmt()] << " >>> ("
          << DNN_FMTS[botDatas_[i]->getIntlFmt()] << " >>> "
          << DNN_FMTS[topData_->getIntlFmt()] << ") >>> "
          << DNN_FMTS[topData_->getUserFmt()];
    }
    
    if (passType != PASS_TEST && nextLayers_.size() > 1) {  // Training
      sumTopDiffs_ = nullptr;
      for (size_t i = 0; i < nextLayers_.size(); ++i) {
        topDiffMDs_[i] = nullptr;
        dnnOutGrads_[i] = Matrix::create(bs_, getSize(), false, false);
      }
    }
    needResetBwd_ = true;
  }

  // all sumbit cvt should be clear
  clearAllDnnCvtFlags();
  // then submit dnn forward
  REGISTER_TIMER_INFO("mkldnn_FwdTimer", getName().c_str());
  submitDnnFwd(passType);
}

void MkldnnLayer::backward(const UpdateCallback& callback) {
  if (needResetBwd_) {
    needResetBwd_ = false;
    // mkldnn init or reset backward
    VLOG(1) << "reset backward batch size to " << bs_
      << " of mkldnn layer: " << getName();

    prepareTopDiff();
    resetDnnBwd();

    // print the diff flow
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
      // TODO(TJ): consider multi input
      if (botDiffs_[i] && topDiff_)
        VLOG(1) << "diff format flow --- "
          << DNN_FMTS[botDiffs_[i]->getUserFmt()] << " <<< ("
          << DNN_FMTS[botDiffs_[i]->getIntlFmt()] << " <<< "
          << DNN_FMTS[topDiff_->getIntlFmt()] << ") <<< "
          << DNN_FMTS[topDiff_->getUserFmt()];
    }
  }

  // submit dnn backward
  REGISTER_TIMER_INFO("mkldnn_BwdTimer", getName().c_str());
  if (nullptr != sumTopDiffs_) {
    std::vector<mkldnn::primitive> sum;
    sum.push_back(*sumTopDiffs_);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(sum).wait();
  }
  submitDnnBwd(callback);
}

void MkldnnLayer::printInfo() {
  for (size_t i = 0; i < ih_.size(); ++i) {
    VLOG(2)
      << "ih: " << ih_[i] << ", iw: " << iw_[i]
      << ", ic: " << ic_[i]
      << ", oh: " << oh_[i] << ", ow: " << ow_[i]
      << ", oc: " << oc_;
  }
}


}  // namespace paddle
