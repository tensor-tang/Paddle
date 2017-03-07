/* Copyright (c) 2016 */

#include "MkldnnAddtoLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_addto, MkldnnAddtoLayer);

bool MkldnnAddtoLayer::initDnn(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  bs_ = 0;
  oc_ = 0;
  layerSize_ = getSize();
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    ih_.push_back(0);
    iw_.push_back(0);
    oh_.push_back(0);
    ow_.push_back(0);
    ic_.push_back(0);
    dataBottoms_.push_back(nullptr);
    CHECK(layerSize_ == inputLayers_[i]->getSize());
  }
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(
      new Weight(1, layerSize_, biasParameter_));
  }
  return true;
}

void MkldnnAddtoLayer::clearDataDiff() {
  reserveOutput(bs_, getSize());
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
  dataTop_.reset(new MkldnnBuffer());
  real *topData = getOutputValue()->getData();
  dataTop_->initUser(topData, topDims_, topFmt_, eg);

  // prepare bottoms
  std::vector<memory::primitive_desc> botPDs;
  std::vector<std::shared_ptr<memory::desc>> prvMDs;
  std::vector<primitive::at> botMems;
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
    dataBottoms_[i].reset(new MkldnnBuffer());
    real *botData = getInputValue(i)->getData();
    dataBottoms_[i]->initUser(botData, botDims_[i], botFmt_[i], eg);
    const std::shared_ptr<memory::desc> prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      dataBottoms_[i]->resetUser(botData, *prvMD, eg);
      VLOG(4) << "use prev format: " << DNN_FMTS[dataBottoms_[i]->getUserFmt()];
      prvMDs.push_back(prvMD);
    }
    botPDs.push_back(dataBottoms_[i]->getUserPD());
    scales_.push_back(1.0);  // no scale here

    // 2. init bot internals
    dataBottoms_[i]->initCvt(dataBottoms_[i]->getUserPD(), dnnCvtNoNeed);
    botMems.push_back(*(dataBottoms_[i]->getIntlMem()));
  }
  // check all inputs format should be the same
  CHECK(prvMDs.size() == 0 || prvMDs.size() == inputLayers_.size())
    << "input format size does not match: "
    << prvMDs.size() << " vs " << inputLayers_.size();
  if (prvMDs.size() > 1) {
    for (size_t i = 1; i < prvMDs.size(); ++i) {
      CHECK(compareMD(*(prvMDs[i-1]), *(prvMDs[i])))
        << "all input formats should be the same";
    }
  }

  // 3. create fwd PD
  std::shared_ptr<sum::primitive_desc> fwdPD;
  fwdPD.reset(new sum::primitive_desc(
    prvMDs.size() > 0 ? *(prvMDs[0]) : getAnyMD(topDims_), scales_, botPDs));
  if (setDnnTopDataFmt_) {
    // fwdPD should be init with any type before, if in here.
    dataTop_->resetUser(topData, fwdPD->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
    VLOG(4) << "set next format: " << DNN_FMTS[dataTop_->getUserFmt()];
  }

  // 4. init top cvt
  dataTop_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);

  // 5. create fwd handle
  fwd_.reset(new sum(*fwdPD, botMems, *(dataTop_->getIntlMem())));

  // TODO(TJ): remove when dataBot vector done
  VLOG(1) << "data format flow --- "
    << DNN_FMTS[dataBottoms_[0]->getUserFmt()] << " >>> ("
    << DNN_FMTS[dataBottoms_[0]->getIntlFmt()] << " >>> "
    << DNN_FMTS[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FMTS[dataTop_->getUserFmt()];
}

void MkldnnAddtoLayer::exFwd(PassType passType) {
  MatrixPtr outV = Matrix::create(bs_, oc_, false, false);
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    MatrixPtr input = getInputValue(i);
    i == 0 ? outV->assign(*input) : outV->add(*input);
  }
//  real *topData = outV->getData();
//  LOG(INFO) << "ex-" << topData[0] << "," << topData[1] << "," << topData[2];
}

void MkldnnAddtoLayer::resetDnnBwd() {
}

void MkldnnAddtoLayer::submitDnnFwd(PassType passType) {
  real *topData = getOutputValue()->getData();
  std::vector<primitive> pipeline;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    real *botData = getPrev(i)->getOutputValue()->getData();
    dataBottoms_[i]->submitCvt(pipeline, botData);
  }
  pipeline.push_back(*fwd_);
  dataTop_->submitCvt(pipeline, topData);
  stream(stream::kind::eager).submit(pipeline).wait();

  /* add the bias-vector */
  if (biases_.get() != NULL) {
    // TODO(TJ): try to use mkldnn speedup this
    getOutputValue()->addBias(*(biases_->getW()), 1);
  }

  forwardActivation();
}

void MkldnnAddtoLayer::exBwd(const UpdateCallback& callback) {
  /* Do derivation */ { backwardActivation(); }

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    /* Calculate the input layers error */
    MatrixPtr preGrad = getInputGrad(i);
    if (NULL != preGrad) {
      preGrad->add(*getOutputGrad());
    }
  }
}

void MkldnnAddtoLayer::submitDnnBwd(const UpdateCallback& callback) {
  // there is no backward for eltwise in mkldnn
  // so use the backward in paddle
  exBwd(callback);
}

}  // namespace paddle
