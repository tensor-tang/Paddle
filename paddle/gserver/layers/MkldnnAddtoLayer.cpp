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
  oc_ = getSize();
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    ic_.push_back(oc_);
    MkldnnBufferPtr bot;
    bot.reset(new MkldnnBuffer({bs_, oc_}));
    dataBottoms_.push_back(bot);
    // iw==ih==ow==oh==1, actually iw, ih, ow, oh not been used
    ih_.push_back(1);
    iw_.push_back(1);
    oh_.push_back(1);
    ow_.push_back(1);
  }
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
  }
  return true;
}

size_t MkldnnAddtoLayer::getOneBatchSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    ih_[i] = iw_[i] = 1;
    oh_[i] = ow_[i] = 1;
    ic_[i] = getSize();
  }
  oc_ = ic_[0];
  return oc_;
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
  return true;
}

void MkldnnAddtoLayer::resetDnnFwd(PassType passType) {
  LOG(INFO) << "reset mkldnn forward of addto layer: " << config_.name();
  memory::dims botDims, topDims;
  memory::format botFmt, topFmt;
  botDims = {bs_, ic_[0]};
  topDims = {bs_, oc_};
  botFmt = memory::format::nc;
  topFmt = memory::format::nc;

  std::vector<memory::primitive_desc> botPDs;
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize())
      << "Assert batchsize of input layers are equal";
    CHECK(oc_ == ic_[i]);
    dataBottoms_[i].reset(new MkldnnBuffer(botDims));
    MatrixPtr input = getInputValue(i);
    real *botData = input->getData();
    const std::shared_ptr<memory::desc> prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      dataBottoms_[i]->initUser(botData, *prvMD, *engine_);
      LOG(FATAL) << "should not be here!";
    } else {
      dataBottoms_[i]->initUser(botData, botDims, botFmt, *engine_);
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

  // top data
  dataTop_.reset(new MkldnnBuffer(topDims));
  real *topData = getOutputValue()->getData();
  if (!setDnnTopDataFmt_) {
    dataTop_->initUser(topData, topDims, topFmt, *engine_);
  } else {
    LOG(FATAL) << "should not be here so far!";
    // fwdPD_ should be init with any type before, if in here.
    dataTop_->initUser(topData, fwdPD_->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
  }
  
  fwdPD_.reset(new sum::primitive_desc(dataTop_->getUserMD(), scales_, botPDs));

  // init top cvt
  if (dataTop_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtInternal2User)) {
    LOG(INFO) << "need reorder --- top data: "
      << DNN_FORMAT[dataTop_->getIntlFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataTop_->getUserFmt()];
  }

  printInfo();
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
