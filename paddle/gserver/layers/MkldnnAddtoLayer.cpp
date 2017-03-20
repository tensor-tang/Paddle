/* Copyright (c) 2017 */

#include "MkldnnAddtoLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_addto, MkldnnAddtoLayer);

bool MkldnnAddtoLayer::initDnn(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  if (config_.has_add_size()) {
    addSize_ = config_.add_size();
  }
  bs_ = 0;
  oc_ = 0;
  layerSize_ = getSize();
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    CHECK(layerSize_ == inputLayers_[i]->getSize());
  }
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(
      new Weight(1, layerSize_, biasParameter_));
  }
  return true;
}

void MkldnnAddtoLayer::clearDataDiff() {
//  reserveOutput(bs_, getSize());
}

void MkldnnAddtoLayer::reshape() {
  // reshape input and output size
  bs_ = getInput(0).getBatchSize();
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = 0;
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    int height = inputLayers_[i]->getOutput().getFrameHeight();
    int width = inputLayers_[i]->getOutput().getFrameWidth();
    if (height > 0 && width > 0) {
      has_spatial_ = true;
      ih_[i] = height;
      iw_[i] = width;
    } else {
      has_spatial_ = false;
      ih_[i] = 1;
      iw_[i] = 1;
    }
    ic_[i] = inputLayers_[i]->getSize() / (iw_[i] * ih_[i]);
    oh_[i] = ih_[i];
    ow_[i] = iw_[i];
    oc_ = ic_[i];
    CHECK(ih_[i] * iw_[i]);
    CHECK(layerSize == 0 || size_t(oh_[i] * ow_[i] * oc_) == layerSize);
    layerSize = oh_[i] * ow_[i] * oc_;
  }
  CHECK(layerSize == layerSize_);

  // reset output image size
  if (has_spatial_) {
    getOutput().setFrameHeight(oh_[0]);
    getOutput().setFrameWidth(ow_[0]);
  }
}

void MkldnnAddtoLayer::resetDnnFwd(PassType passType) {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  if (!has_spatial_) {
    topDims_ = {bs_, oc_};
    topFmt_ = memory::format::nc;
  } else {
    topDims_ = {bs_, oc_, oh_[0], ow_[0]};
    topFmt_ = memory::format::nchw;
  }
  // create top buffer and init user, only have one output
  topData_.reset(new MkldnnBuffer());
  real *topDataData = getOutputValue()->getData();
  topData_->initUser(topDataData, topDims_, topFmt_, eg);

  // prepare bottoms
  std::vector<memory::primitive_desc> botPDs;
  std::vector<std::shared_ptr<memory::desc>> prvMDs;
  std::vector<primitive::at> botMems;
  CHECK_EQ(botDatas_.size(), inputLayers_.size());
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize())
      << "Assert batchsize of input layers are equal";
    CHECK(oc_ == ic_[i] && iw_[i] == ow_[i] && ih_[i] == oh_[i]);
    // init dim structure that describes user data.
    if (!has_spatial_) {
      botDims_[i] = {bs_, ic_[i]};
      botFmt_[i] = memory::format::nc;
    } else {
      botDims_[i] = {bs_, ic_[i], ih_[i], iw_[i]};
      botFmt_[i] = memory::format::nchw;
    }
    // 1. create bottom buffer and init user
    botDatas_[i].reset(new MkldnnBuffer());
    real *botDataData = getInputValue(i)->getData();
    botDatas_[i]->initUser(botDataData, botDims_[i], botFmt_[i], eg);
    const std::shared_ptr<memory::desc>& prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      // this layer support both nc and internal format
      botDatas_[i]->resetUser(botDataData, *prvMD, eg);
      VLOG(4) << "use prev data fmt: "
        << DNN_FMTS[botDatas_[i]->getUserFmt()];
      prvMDs.push_back(prvMD);
    }
    botPDs.push_back(botDatas_[i]->getUserPD());
    scales_.push_back(1.0);  // no scale here

    // 2. init bot internals
    botDatas_[i]->initCvt(botDatas_[i]->getUserPD(), dnnCvtNoNeed);
    botMems.push_back(*(botDatas_[i]->getIntlMem()));
  }
  // check all inputs format should be the same
  CHECK(prvMDs.size() == 0 || prvMDs.size() == inputLayers_.size())
    << "intl input size does not match: "
    << prvMDs.size() << " vs " << inputLayers_.size();
  for (size_t i = 1; i < prvMDs.size(); ++i) {
    CHECK(compareMD(*(prvMDs[i-1]), *(prvMDs[i])))
      << "all input formats should be the same";
  }

  // 3. create fwd PD
  std::shared_ptr<sum::primitive_desc> fwdPD;
  fwdPD.reset(new sum::primitive_desc(
    prvMDs.size() > 0 ? *(prvMDs[0]) : MkldnnBuffer::getMD(topDims_),
    scales_, botPDs));
  // reset top user using internal fmt if next is dnn
  if (nextIsDnn_) {
    // fwdPD should be init with any type before, if in here.
    topData_->resetUser(topDataData, fwdPD->dst_primitive_desc());
    setTopDataMD(topData_->getUserMD());
    VLOG(4) << "set next data fmt: " << DNN_FMTS[topData_->getUserFmt()];
  }

  // 4. init top cvt
  topData_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);

  // 5. create fwd handle
  fwd_.reset(new sum(*fwdPD, botMems, *(topData_->getIntlMem())));
}

void MkldnnAddtoLayer::resetDnnBwd() {
  const std::shared_ptr<mkldnn::memory::desc>& prvMD = getTopDiffMD();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize()) << "batchsize should equal";
    if (prevIsDnn_[i]) {
      getPrev(i)->setTopDiffMD(this->getName(), *prvMD);
      VLOG(4) << "set bot diff fmt: "
        << DNN_FMTS[MkldnnBuffer::getMDFmt(*prvMD)];
    }
  }
}

void MkldnnAddtoLayer::submitDnnFwd(PassType passType) {
  real *topDataData = getOutputValue()->getData();
  std::vector<primitive> pipeline;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    real *botDataData = getPrev(i)->getOutputValue()->getData();
    botDatas_[i]->submitCvt(pipeline, botDataData);
  }
  pipeline.push_back(*fwd_);
  topData_->submitCvt(pipeline, topDataData);
  stream(stream::kind::eager).submit(pipeline).wait();

  /* add the bias-vector */
  if (biases_.get() != NULL) {
    // TODO(TJ): try to use mkldnn
    if (topData_->getIntlFmt() != topFmt_) {
      // not nc or nchw
      LOG(FATAL) << "not implemented with internal format";
    }
    LOG(WARNING) << "not speedup with MKLDNN yet";
    getOutputValue()->addBias(*(biases_->getW()), 1);
  }

  forwardActivation();
}

void MkldnnAddtoLayer::submitDnnBwd(const UpdateCallback& callback) {
  backwardActivation();
  if (biases_ && biases_->getWGrad()) {
    // TODO(TJ): try to use mkldnn
    const std::shared_ptr<mkldnn::memory::desc>& prvMD = getTopDiffMD();
    if (prvMD && MkldnnBuffer::getMDFmt(*prvMD) != topFmt_) {
      LOG(FATAL) << "not implemented with internal format";
    }
    LOG(WARNING) << "not speedup with MKLDNN yet";
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);
    biases_->getParameterPtr()->incUpdate(callback);
  }
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    MatrixPtr preGrad = getInputGrad(i);
    if (NULL == preGrad)
      continue;
    if (addSize_ == 0) {
      // directly set the diff, do not copy
      getDnnInputGrad_mutable(i) = getDnnOutputGrad();
    } else {
      const std::shared_ptr<mkldnn::memory::desc>& prvMD = getTopDiffMD();
      if (prvMD && MkldnnBuffer::getMDFmt(*prvMD) != topFmt_) {
        LOG(FATAL) << "not implemented when addsize > 0 with internal format";
      }
      LOG(WARNING) << "not speedup with MKLDNN yet";
      // TODO(TJ): try to use mkldnn
      preGrad->add(*getOutputGrad());
    }
  }
}

}  // namespace paddle
