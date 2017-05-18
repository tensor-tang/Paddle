/* Copyright (c) 2017 */

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnFcLayer.h"

using namespace mkldnn;  // NOLINT
typedef mkldnn::inner_product_forward fc_fwd;
//typedef mkldnn::inner_product_backward_data fc_bwdData;

namespace paddle {

REGISTER_LAYER(mkldnn_fc, MkldnnFcLayer);

// load the settings from proto
void MkldnnFcLayer::loadConfig() {
  CHECK_EQ(config_.inputs_size(), 1) << "should have only one input config!";
  if (config_.has_test_with_paddle_wgt()) {
    testWithPaddleWgt = config_.test_with_paddle_wgt();
  }

  // get dim of input and output
  const FCConfig &conf = config_.inputs(0).fc_conf();
  inputLayerSize_ = conf.dim_in();
  oc_ = conf.dim_out();
}

bool MkldnnFcLayer::initDnnWgt(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  // only support 1 input layer by now
  CHECK_EQ(inputLayers_.size(), 1) << "Only support one input layer yet!";
  CHECK_EQ(inputLayers_.size(), parameters_.size());
  CHECK(!parameters_[0]->isSparse()) << "Do not support sparse yet";
  CHECK_EQ(oc_, getSize());
  CHECK_EQ(parameters_[0]->getSize(), oc_ * inputLayerSize_);

  // create a mkldnn weight
  weight_ = std::unique_ptr<Weight>(new Weight(oc_, inputLayerSize_, parameters_[0], 0));
  // paddle wgt for initial or test with paddle wg
  paddleWgt_ = Matrix::create(inputLayerSize_, oc_, false, false);
  paddleWgt_->zeroMem();

  // create biases
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
    hasBias_ = true;
  }

  return true;
}

// FC layer do not care about the seqlen changing
// the bs would not be actually used,
// use outputMatH_ instead as the bs in MKLDNN
void MkldnnFcLayer::reshapeOutputInfo() {
  CHECK_EQ(inputLayers_.size(), 1UL);

  reshapeOutMatSize();

  reshapeBatchSize();

  reshapeImgSize();
  
  keepOutputSize();
}

void MkldnnFcLayer::resetDnnFwd(PassType passType) {
  CHECK_EQ(inputLayers_.size(), 1);
  CHECK_EQ(hasBias_, biases_ && biases_->getW());
  passType_ = passType;
  usePrevLayout_ = false;

  engine_ = CpuEngine::Instance().getEngine();

  resetDnnConfigs();

  resetDnnFwdBuffers();

  resetDnnUserLayout();

  usePrevDnnLayout();

  std::shared_ptr<fc_fwd::primitive_desc> fwdPD;
  resetDnnFwdPD(fwdPD);

  resetDnnIntlLayout(fwdPD);

  keepDnnLayoutToNext(fwdPD);

  resetDnnFwdHandle(fwdPD);

  if (!hasInitedWgt_) {
    hasInitedWgt_ = true;
    getInitialWgtFromPaddle();
  }
}

void MkldnnFcLayer::resetDnnBwd() {
  mkldnn::engine engine_ = CpuEngine::Instance().getEngine();
  prop_kind pk = prop_kind::forward;

  CHECK_EQ(hasBias_, biases_ && biases_->getWGrad());

  bool hasCvtTopDiffBwdData = false;
  bool hasCvtTopDiffBwdWgt = false;
  bool hasCvtBiasDiff = false;

  // 1. create mkldnn buffer, only have one output and bias buffer
  topDiff_.reset(new MkldnnBuffer());
  topDiffBwdWgt_.reset(new MkldnnBuffer());
  if (hasBias_) {
    biasDiff_.reset(new MkldnnBuffer());
  }
  // 2. init user top and bias
  real *topDiffData = getDnnOutputGrad()->getData();
  topDiff_->initUser(topDiffData, topDims_, topFmt_, engine_);
  topDiffBwdWgt_->initUser(topDiffData, topDims_, topFmt_, engine_);
  if (hasBias_) {
    real* biasDiffData = biases_->getWGrad()->getData();
    biasDiff_->initUser(biasDiffData, biasDims_, biasFmt_, engine_);
  } else {
//    biasDiff_->initUser(NULL, biasDims_, biasFmt_, engine_);
  }
  // use internal top diff if use dnn input
  const std::shared_ptr<mkldnn::memory::desc>& prv = getTopDiffMD();
  if (prv) {
    topDiff_->resetUser(topDiffData, *prv, engine_);
    topDiffBwdWgt_->resetUser(topDiffData, *prv, engine_);
    bool isNCHW = topDiff_->getUserFmt() == memory::format::nchw;
    if (isNCHW && oh_ == ow_ && oh_ == 1) {
      // if prv is nchw and h==w==1, use nc instead
      topDiff_->resetUser(topDiffData, topDims_, memory::format::nc, engine_);
      topDiffBwdWgt_->resetUser(topDiffData, topDims_, memory::format::nc, engine_);
      VLOG(4) << "use nc diff fmt";
    } else {
      VLOG(4) << "use prev diff fmt: " << DNN_FMTS[topDiff_->getUserFmt()];
    }
  }
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    // 1. create mkldnn buffer and init user
    CHECK(weight_->getWGrad()) << "should have weight anyway";
    wgtDiff_.reset(new MkldnnBuffer());
    real *wgtDiffData = weight_->getWGrad()->getData();
    wgtDiff_->initUser(wgtDiffData, wgtDims_, wgtFmt_, engine_);
    // 2. prepare backward weight and bias
    std::shared_ptr<fc_fwd::desc> bwdFwdDesc;
    std::shared_ptr<fc_fwd::primitive_desc> bwdFwdPD;
    std::shared_ptr<inner_product_backward_weights::desc> bwdWgtDesc;
    std::shared_ptr<inner_product_backward_weights::primitive_desc> bwdWgtPD;
    bwdFwdDesc.reset(new fc_fwd::desc(pk,
      botData_->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_),
      MkldnnBuffer::getMD(topDims_)));
    bwdFwdPD.reset(new fc_fwd::primitive_desc(
      *bwdFwdDesc, engine_));
    if (hasBias_) {
      bwdWgtDesc.reset(new inner_product_backward_weights::desc(
        botData_->getIntlMD(),
        MkldnnBuffer::getMD(wgtDims_),
        biasData_->getIntlMD(),  // bias do not use any
        MkldnnBuffer::getMD(topDims_)));
    } else {
      bwdWgtDesc.reset(new inner_product_backward_weights::desc(
        botData_->getIntlMD(),
        MkldnnBuffer::getMD(wgtDims_),
        MkldnnBuffer::getMD(topDims_)));
    }
    bwdWgtPD.reset(new inner_product_backward_weights::primitive_desc(
      *bwdWgtDesc, engine_, *bwdFwdPD));
    CHECK(botData_->getIntlPD() == bwdWgtPD->src_primitive_desc());
    if (hasBias_) {
      CHECK(biasData_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc());
    }
    // 3. init conversion
    if (!testWithPaddleWgt) {
      wgtDiffData = weight_->getWGrad()->getData();
      wgtDiff_->resetUser(wgtDiffData, bwdWgtPD->diff_weights_primitive_desc());
      wgtDiff_->initCvt(bwdWgtPD->diff_weights_primitive_desc(), dnnCvtNoNeed);
      CHECK_EQ(wgtDiff_->getIntlSize(), wgtData_->getIntlSize())
        << "can not use mkldnn wgt since memory size does not equal";
    } else {
      wgtDiff_->initCvt(
        bwdWgtPD->diff_weights_primitive_desc(), dnnCvtIntl2User);
    }
    if (hasBias_) {
      if (!hasCvtBiasDiff) {
        hasCvtBiasDiff = true;
        CHECK(biasDiff_->getUserPD() == bwdWgtPD->diff_bias_primitive_desc())
          << "should always be format::x, or changed in new mkldnn version";
        biasDiff_->initCvt(biasDiff_->getUserPD(), dnnCvtNoNeed);
      } else {
        CHECK(biasDiff_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc())
          << "all bias formats should equal";
      }
    }
    if (!hasCvtTopDiffBwdWgt) {
      hasCvtTopDiffBwdWgt = true;
      topDiffBwdWgt_->initCvt(bwdWgtPD->diff_dst_primitive_desc(),
        dnnCvtUser2Intl);
    } else {
      CHECK(topDiffBwdWgt_->getIntlPD() == bwdWgtPD->diff_dst_primitive_desc())
        << "all topDiffData formats should equal";
    }
    // 4. bias backward can only be executed in weight backward with MKL-DNN
    if (hasBias_) {
      bwdWgt_.reset(new inner_product_backward_weights(*bwdWgtPD,
        *(botData_->getIntlMem()), *(topDiffBwdWgt_->getIntlMem()),
        *(wgtDiff_->getIntlMem()), *(biasDiff_->getIntlMem())));
    } else {
      bwdWgt_.reset(new inner_product_backward_weights(*bwdWgtPD,
        *(botData_->getIntlMem()), *(topDiffBwdWgt_->getIntlMem()),
        *(wgtDiff_->getIntlMem()),
        memory(memory::primitive_desc(memory::desc({}, memory::data_type::f32, biasFmt_), engine_))));
    }
    if (wgtDiff_) {
      VLOG(3) << "weight diff flow --- "
        << DNN_FMTS[wgtDiff_->getUserFmt()]
        << " <<< "
        << DNN_FMTS[wgtDiff_->getIntlFmt()];
    }
    // then prepare backward data ----------------------------------------------
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      continue;  // data layer has not diff
    }
    // 1. create buffer and init user
    real* botDiffData = getDnnInputGrad(i)->getData();
    botDiff_.reset(new MkldnnBuffer());
    botDiff_->initUser(botDiffData, botDims_, botFmt_, engine_);
    // 2. init backward data primitive desc
    std::shared_ptr<inner_product_backward_data::desc> bwdDataDesc;
    std::shared_ptr<inner_product_backward_data::primitive_desc> bwdDataPD;
    bwdDataDesc.reset(new inner_product_backward_data::desc(
      MkldnnBuffer::getMD(botDims_),
      wgtData_->getIntlMD(),
      MkldnnBuffer::getMD(topDims_)));
    bwdDataPD.reset(new inner_product_backward_data::primitive_desc(
      *bwdDataDesc, engine_, *bwdFwdPD));
// CHECK(botData_->getIntlPD() == bwdDataPD->diff_src_primitive_desc());
    CHECK(wgtData_->getIntlPD() == bwdDataPD->weights_primitive_desc());
// CHECK(topDiffBwdWgt_->getIntlPD() == bwdDataPD->diff_dst_primitive_desc());
    // 3. init conversion
    if (prevIsDnn_[i]) {
      botDiff_->resetUser(
        botDiffData, bwdDataPD->diff_src_primitive_desc());
      prevLayer->setTopDiffMD(this->getName(), botDiff_->getUserMD());
      VLOG(4) << "set next diff fmt: " << DNN_FMTS[botDiff_->getUserFmt()];
    }
    botDiff_->initCvt(
      bwdDataPD->diff_src_primitive_desc(), dnnCvtIntl2User);
    if (!hasCvtTopDiffBwdData) {
      hasCvtTopDiffBwdData = true;
      topDiff_->initCvt(bwdDataPD->diff_dst_primitive_desc(), dnnCvtUser2Intl);
    } else {
      CHECK(topDiff_->getIntlPD() == bwdDataPD->diff_dst_primitive_desc())
        << "all topDiffData formats should equal";
    }
    // 4. create bwd data handle
    bwdData_.reset(new inner_product_backward_data(
      *bwdDataPD, *(topDiff_->getIntlMem()),
      *(wgtData_->getIntlMem()), *(botDiff_->getIntlMem())));
  }
}

void MkldnnFcLayer::submitDnnFwd() {
  forwardDnn();

  forwardActivation();
}

void MkldnnFcLayer::submitBwdData(int idx) {
  const MatrixPtr& botGrad = getDnnInputGrad(idx);
  if (botGrad == NULL) {
    return;
  }
  real* botDiffData = botGrad->getData();
  real* topDiffData = getDnnOutputGrad()->getData();
  std::vector<primitive> pipeline;
  if (testWithPaddleWgt) {  // no need cvt wgt without testWithPaddleWgt
    CHECK(paddleWgt_);
    real* wgtValData = paddleWgt_->getData();
    wgtData_->submitCvt(pipeline, wgtValData);
  }
  topDiff_->submitCvt(pipeline, topDiffData);
  pipeline.push_back(*bwdData_);
  botDiff_->submitCvt(pipeline, botDiffData);
  stream(stream::kind::eager).submit(pipeline).wait();
}

void MkldnnFcLayer::submitBwdWgts(int idx) {
  real* botValData = getInputValue(idx)->getData();
  real* topDiffData = getDnnOutputGrad()->getData();
  real* wgtDiffData = weight_->getWGrad()->getData();

  std::vector<primitive> pipeline;
  topDiffBwdWgt_->submitCvt(pipeline, topDiffData);
  botData_->submitCvt(pipeline, botValData);
  pipeline.push_back(*bwdWgt_);
  wgtDiff_->submitCvt(pipeline, wgtDiffData);
  // no need to submit cvt bias since biasfmt would not be changed
  stream(stream::kind::eager).submit(pipeline).wait();
}

void MkldnnFcLayer::submitDnnBwd(const UpdateCallback &callback) {
  backwardActivation();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    submitBwdData(i);
    if (weight_->getWGrad()) {
      submitBwdWgts(i);
      weight_->getParameterPtr()->incUpdate(callback);
    }
  }
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}






/// protected method:

void MkldnnFcLayer::reshapeOutMatSize() {
  // input layer size can not be changed
  CHECK_EQ(inputLayerSize_, inputMatW_) << "should not change input layer size,"
    << "which means the weight size would need change too";
  outputMatW_ = oc_;

  // only can change height
  outputMatH_ = inputMatH_;
}

void MkldnnFcLayer::reshapeBatchSize() {
  int seqLen = getInput(0).getMklSeqLen();
  if (seqLen > 1) {
    bs_ = outputMatH_ / seqLen;
    CHECK_EQ(bs_ * seqLen, outputMatH_) << "not divisible";
  } else {
    bs_ = outputMatH_;
  }
}

void MkldnnFcLayer::reshapeImgSize() {
  // reshape input sizes
  const Argument& input = getInput(0);
  ih_ = input.getFrameHeight();
  iw_ = input.getFrameWidth();
  if (ih_ == 0) {
    ih_ = 1;
  }
  if (iw_ == 0) {
    iw_ = 1;
  }
  hasSpatial_ = true;
  if (ih_ == 1 && iw_ == 1) {
    hasSpatial_ = false;
  }
  ic_ = inputMatW_ / (ih_ * iw_);
  CHECK_EQ(ic_ * ih_ * iw_, inputMatW_) << "not divisible";

  // keep out size unchanged
  CHECK_EQ(oc_, outputMatW_) << "output layersize can not be changed";
  oh_ = 1;
  ow_ = 1;
}

void MkldnnFcLayer::keepOutputSize() {
  CHECK_EQ(oc_, outputMatW_) << "output layersize can not be changed";
  CHECK_EQ(oh_, 1);
  CHECK_EQ(ow_, 1);

  output_.setFrameHeight(oh_);
  output_.setFrameWidth(ow_);
}

void MkldnnFcLayer::resetDnnConfigs() {
  if (hasSpatial_) {
    botDims_ = {(int)(outputMatH_), ic_, ih_, iw_};
    botFmt_ = memory::format::nchw;
    wgtDims_ = {oc_, ic_, ih_, iw_};
    wgtFmt_ = memory::format::oihw;
  } else {
    botDims_ = {(int)outputMatH_, ic_};
    botFmt_ = memory::format::nc;
    wgtDims_ = {oc_, ic_};
    wgtFmt_ = memory::format::oi;
  }

  topDims_ = {(int)outputMatH_, oc_};
  topFmt_ = memory::format::nc;

  if (hasBias_) {
    biasDims_ = {oc_};
    biasFmt_ = memory::format::x;
  } else {
    biasDims_ = {};
    biasFmt_ = memory::format::format_undef;
  }
}

void MkldnnFcLayer::resetDnnFwdBuffers() {
  // reset buffers
  topData_.reset(new MkldnnBuffer());
  botData_.reset(new MkldnnBuffer());
  wgtData_.reset(new MkldnnBuffer());
  if (hasBias_) {
    biasData_.reset(new MkldnnBuffer());
  }
}
void MkldnnFcLayer::resetDnnUserLayout() {
  CHECK(getInput(0).value) << "The input of mkldnn fc layer must be matrix";

  const MatrixPtr& botVal = getInputValue(0);
  const MatrixPtr& topVal = getOutputValue();
  const MatrixPtr& wgtVal = weight_->getW();
  real *botValData = botVal->getData();
  real *topValData = topVal->getData();
  real *wgtValData = wgtVal->getData();

  topData_->initUser(topValData, topDims_, topFmt_, engine_);
  botData_->initUser(botValData, botDims_, botFmt_, engine_);
  wgtData_->initUser(wgtValData, wgtDims_, wgtFmt_, engine_);
  if (hasBias_) {
    real *biasDataData = biases_->getW()->getData();
    biasData_->initUser(biasDataData, biasDims_, biasFmt_, engine_);
  }
}

void MkldnnFcLayer::usePrevDnnLayout() {
  if (!prevIsDnn_[0]) {
    return;
  }

  const std::shared_ptr<memory::desc>& prvMD = getPrev(0)->getTopDataMD();
  if (!prvMD) {
    return;
  }

  const MatrixPtr& botVal = getInputValue(0);
  real *botValData = botVal->getData();
  botData_->resetUser(botValData, *prvMD, engine_);
  VLOG(4) << "use prev data fmt: " << DNN_FMTS[botData_->getUserFmt()];

// TODO: check change it???
  botDims_ = botData_->getUserDims();
  LOG(INFO) << "-----------"<<botDims_[0] << "," << botDims_[1];
  botFmt_ = (mkldnn::memory::format)botData_->getUserFmt();
  usePrevLayout_ = true;
}

void MkldnnFcLayer::resetDnnFwdPD(
  std::shared_ptr<fc_fwd::primitive_desc>& fwdPD) {
  prop_kind pk = prop_kind::forward;
  std::shared_ptr<fc_fwd::desc> fwdDesc;
  if (hasBias_) {
    fwdDesc.reset(new fc_fwd::desc(pk,
        MkldnnBuffer::getMD(botDims_),
        MkldnnBuffer::getMD(wgtDims_),
        MkldnnBuffer::getMD(biasDims_),
        MkldnnBuffer::getMD(topDims_)));
  } else {
    fwdDesc.reset(new fc_fwd::desc(pk,
        MkldnnBuffer::getMD(botDims_),
        MkldnnBuffer::getMD(wgtDims_),
        MkldnnBuffer::getMD(topDims_)));
  }
  fwdPD.reset(new fc_fwd::primitive_desc(*fwdDesc, engine_));
}

void MkldnnFcLayer::resetDnnIntlLayout(
  const std::shared_ptr<fc_fwd::primitive_desc>& fwdPD) {
  botData_->initCvt(fwdPD->src_primitive_desc(), dnnCvtUser2Intl);
  topData_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);

  // wgt
  const MatrixPtr& wgtVal = weight_->getW();
  real *wgtValData = wgtVal->getData();
  wgtData_->resetUser(wgtValData, fwdPD->weights_primitive_desc());
  wgtData_->initCvt(wgtData_->getUserPD(), dnnCvtNoNeed);
  CHECK_EQ(wgtData_->getIntlSize(), parameters_[0]->getSize())
    << "can not use mkldnn wgt since memory size does not equal";
  VLOG(3) << "weight data flow --- " << DNN_FMTS[wgtData_->getUserFmt()]
    << " >>> " << DNN_FMTS[wgtData_->getIntlFmt()];

  if (hasBias_) {
    CHECK(biasData_->getUserPD() == fwdPD->bias_primitive_desc())
      << "should always be format::x, or changed in new mkldnn version";
    biasData_->initCvt(biasData_->getUserPD(), dnnCvtNoNeed);
  }
}

void MkldnnFcLayer::keepDnnLayoutToNext(
  const std::shared_ptr<fc_fwd::primitive_desc>& fwdPD) {
  if (!nextIsDnn_) {
    return;
  }

  const MatrixPtr& topVal = getOutputValue();
  real *topValData = topVal->getData();
  topData_->resetUser(topValData, fwdPD->dst_primitive_desc());
  setTopDataMD(topData_->getUserMD());
  VLOG(4) << "set next data fmt: " << DNN_FMTS[topData_->getUserFmt()];
}

void MkldnnFcLayer::resetDnnFwdHandle(
  const std::shared_ptr<fc_fwd::primitive_desc>& fwdPD) {
  if (hasBias_) {
    fwd_.reset(new fc_fwd(*fwdPD,
      *(botData_->getIntlMem()), *(wgtData_->getIntlMem()),
      *(biasData_->getIntlMem()), *(topData_->getIntlMem())));
  } else {
    fwd_.reset(new fc_fwd(*fwdPD,
      *(botData_->getIntlMem()), *(wgtData_->getIntlMem()),
      *(topData_->getIntlMem())));
  }
}

// when training get initial wgt from paddle format
// when scoring from paddle also need get initial wgt from paddle format
// however when scoring with mkldnn wgt donot need get initial wgt
void MkldnnFcLayer::getInitialWgtFromPaddle() {
  if (passType_ == PASS_TEST && !testWithPaddleWgt) {
    return;
  }

  // paddle wgt is transposed
  const MatrixPtr& wgtVal = weight_->getW();
  paddleWgt_->copyFrom(*wgtVal);
  paddleWgt_ = paddleWgt_->getTranspose();
  
  MkldnnBufferPtr tmp(new MkldnnBuffer());
  tmp->initUser(paddleWgt_->getData(), wgtDims_, wgtFmt_, engine_);
  tmp->initCvt(wgtData_->getIntlPD(), dnnCvtUser2Intl);

  std::vector<primitive> cvtToDnnWgt;
  tmp->submitCvt(cvtToDnnWgt);
  stream(stream::kind::eager).submit(cvtToDnnWgt).wait();

  real* dst = weight_->getW()->getData();
  memcpy(dst, tmp->getIntlData(), tmp->getIntlSize() * sizeof(real));
}

void MkldnnFcLayer::forwardDnn() {
  real *topValData = getOutputValue()->getData();
  real *botValData = getPrev(0)->getOutputValue()->getData();
  std::vector<primitive> pipeline;

  botData_->submitCvt(pipeline, botValData);
  pipeline.push_back(*fwd_);
  topData_->submitCvt(pipeline, topValData);

  stream(stream::kind::eager).submit(pipeline).wait();
}

}  // namespace paddle
