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

size_t MkldnnAddtoLayer::getOneBatchSize() {
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
  return layerSize;
}

bool MkldnnAddtoLayer::reshapeOutput() {
  if (bs_ == getInput(0).getBatchSize()) {
    // can remove reserveOutput when confirm how multi inputs work
    // and whether to clear diff
    reserveOutput(bs_, getOneBatchSize());
    return false;
  }
  // reset data
  bs_ = getInput(0).getBatchSize();
  LOG(INFO) << "reshape batch size: " << bs_;
  reserveOutput(bs_, getOneBatchSize());
  if(has_spatial_) {
    getOutput().setFrameHeight(oh_[0]);
    getOutput().setFrameWidth(ow_[0]);
  }
  printInfo();
  return true;
}

void MkldnnAddtoLayer::resetDnnFwd(PassType passType) {
  LOG(INFO) << "reset mkldnn forward of addto layer: " << config_.name();
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  memory::dims botDims, topDims;
  memory::format botFmt, topFmt;
  if (!has_spatial_) {
    botDims = {bs_, ic_[0]};
    topDims = {bs_, oc_};
    botFmt = memory::format::nc;
    topFmt = memory::format::nc;
  } else {
    botDims = {bs_, ic_[0], ih_[0], iw_[0]};
    topDims = {bs_, oc_, oh_[0], ow_[0]};
    botFmt = memory::format::nchw;
    topFmt = memory::format::nchw;
  }

  std::vector<memory::primitive_desc> botPDs;
  std::vector<std::shared_ptr<memory::desc>> prvs;
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize())
      << "Assert batchsize of input layers are equal";
    CHECK(oc_ == ic_[i] && iw_[i] == ow_[i] && ih_[i] == oh_[i]);
    dataBottoms_[i].reset(new MkldnnBuffer(botDims));
    MatrixPtr input = getInputValue(i);
    real *botData = input->getData();
    const std::shared_ptr<memory::desc> prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      dataBottoms_[i]->initUser(botData, *prvMD, eg);
      LOG(INFO) << "use prev format: " << DNN_FORMAT[dataBottoms_[i]->getUserFmt()];
      prvs.push_back(prvMD);
    } else {
      dataBottoms_[i]->initUser(botData, botDims, botFmt, eg);
    }

    botPDs.push_back(dataBottoms_[i]->getUserPD());
    scales_.push_back(1.0);  // no scale here

    // init bot cvt
    if (dataBottoms_[i]->initCvt(
      dataBottoms_[i]->getUserPD(), dnnCvtUser2Internal)) {
      LOG(INFO) << "need reorder --- bottom data: "
        << DNN_FORMAT[dataBot_->getUserFmt()]
        << " >>>>> "
        << DNN_FORMAT[dataBot_->getIntlFmt()];
    }
  }
  // inputs format should be all the same
  CHECK(prvs.size() == 0 || prvs.size() == inputLayers_.size())
    << "input format size does not match: "
    << prvs.size() << " vs " << inputLayers_.size();
  if (prvs.size() > 1) {
    for (size_t i = 1; i < prvs.size(); ++i) {
      CHECK(compareMD(*(prvs[i-1]), *(prvs[i])))
        << "all input formats should be the same";
    }
  }

  // top data
  dataTop_.reset(new MkldnnBuffer(topDims));
  real *topData = getOutputValue()->getData();
  fwdPD_.reset(new sum::primitive_desc(
    prvs.size() > 0 ? *(prvs[0]) : dataTop_->getMDAny(),
    scales_, botPDs));
  if (setDnnTopDataFmt_) {
    // fwdPD_ should be init with any type before, if in here.
    dataTop_->initUser(topData, fwdPD_->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
    LOG(INFO) << "set next format: " << DNN_FORMAT[dataTop_->getUserFmt()];
  } else {
    dataTop_->initUser(topData, topDims, topFmt, eg);
  }

  // init top cvt
  if (dataTop_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtInternal2User)) {
    LOG(INFO) << "need reorder --- top data: "
      << DNN_FORMAT[dataTop_->getIntlFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataTop_->getUserFmt()];
  }
  LOG(INFO) << "data format flow --- "
    << DNN_FORMAT[dataBottoms_[0]->getUserFmt()] << " >>> ("
    << DNN_FORMAT[dataBottoms_[0]->getIntlFmt()] << " >>> "
    << DNN_FORMAT[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FORMAT[dataTop_->getUserFmt()];
}

void MkldnnAddtoLayer::myFwd(PassType passType) {
  /// all sumbit cvt should be clear
  clearAllCvtFlags();

  std::vector<primitive> fwd;
  std::vector<primitive::at> srcs;

  for (size_t i = 0; i < inputLayers_.size(); i++) {
    real *botdata = getPrev(i)->getOutputValue()->getData();
    dataBottoms_[i]->submitCvt(fwd, botdata);
    srcs.push_back(*(dataBottoms_[i]->getIntlMem()));
  }
  fwd.push_back(mkldnn::sum(*fwdPD_, srcs, *(dataTop_->getIntlMem())));
  real *topdata = getOutputValue()->getData();
  dataTop_->submitCvt(fwd, topdata);

  // start forward
  REGISTER_TIMER_INFO("mkldnn_AddtoFwd", getName().c_str());
  stream(stream::kind::eager).submit(fwd).wait();
//  LOG(INFO) << "my-" << topdata[0] << "," << topdata[1] << "," << topdata[2];
}


void MkldnnAddtoLayer::exFwd(PassType passType) {
  MatrixPtr outV = Matrix::create(bs_, oc_, false, false);
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    MatrixPtr input = getInputValue(i);
    i == 0 ? outV->assign(*input) : outV->add(*input);
  }
//  real *topdata = outV->getData();
//  LOG(INFO) << "ex-" << topdata[0] << "," << topdata[1] << "," << topdata[2];
}

void MkldnnAddtoLayer::submitDnnFwd(PassType passType) {
  myFwd(passType);

//  exFwd(passType);

  /* add the bias-vector */
  if (biases_.get() != NULL) {
    getOutputValue()->addBias(*(biases_->getW()), 1);
  }

  REGISTER_TIMER_INFO("mkldnn_addto_FwAtvTimer", getName().c_str());
  forwardActivation();
}

void MkldnnAddtoLayer::resetDnnBwd() {
  // there is no backward for addto in mkldnn
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
