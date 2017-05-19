/* Copyright (c) 2017 */

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnFcLayer.h"

using namespace mkldnn;  // NOLINT
typedef mkldnn::inner_product_forward fc_fwd;
typedef mkldnn::inner_product_backward_weights fc_bwdWgt;
typedef mkldnn::inner_product_backward_data fc_bwdData;

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

  engine_ = CpuEngine::Instance().getEngine();

  resetDnnConfigs();

  resetDnnFwdBuffers();

  resetDnnFwdUserLayout();

  if (prevIsDnn_[0]) {
    resetBotValUserWithPrevLayout();
  }

  std::shared_ptr<fc_fwd::primitive_desc> fwdPD;
  resetDnnFwdPD(fwdPD);

  resetDnnFwdIntlLayout(fwdPD);

  if (nextIsDnn_) {
    resetTopValUserLayout(fwdPD);
    keepLayoutToNextLayert();
  }

  resetDnnFwdHandle(fwdPD);

  if (!hasInitedWgt_) {
    hasInitedWgt_ = true;
    getInitialWgtFromPaddle();
  }
}

void MkldnnFcLayer::resetDnnBwd() {
  CHECK_EQ(hasBias_, biases_ && biases_->getWGrad());

  resetDnnBwdBuffers();

  resetDnnBwdUserLayout();

  if (nextIsDnn_) {
    resetTopDiffUserWithNextLayout();
  }

  std::shared_ptr<fc_bwdWgt::primitive_desc> bwdWgtPD;
  std::shared_ptr<fc_bwdData::primitive_desc> bwdDataPD;
 
  resetDnnBwdWgtPD(bwdWgtPD);

  resetDnnBwdDataPD(bwdDataPD);

  resetDnnBwdIntlLayout(bwdWgtPD, bwdDataPD);

  if (prevIsDnn_[0]) {
    resetBotGradUserLayout(bwdDataPD);
    keepLayoutToPrevLayer();
  }

  resetDnnBwdHandle(bwdWgtPD, bwdDataPD);
}

void MkldnnFcLayer::submitDnnFwd() {
  forwardDnnVal();

  forwardDnnAct();
}

void MkldnnFcLayer::submitDnnBwd(const UpdateCallback &callback) {
  BackwardDnnAct();

  backwardDnnData();

  backwardDnnWgt();

  updateParameter(callback);
}

/*************************** protected methods: *******************************/
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
void MkldnnFcLayer::resetDnnFwdUserLayout() {
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

void MkldnnFcLayer::resetDnnFwdIntlLayout(
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

void MkldnnFcLayer::forwardDnnVal() {
  real *topValData = getOutputValue()->getData();
  real *botValData = getPrev(0)->getOutputValue()->getData();
  std::vector<primitive> pipeline;

  botData_->submitCvt(pipeline, botValData);
  pipeline.push_back(*fwd_);
  topData_->submitCvt(pipeline, topValData);

  stream(stream::kind::eager).submit(pipeline).wait();
}

void MkldnnFcLayer::resetDnnBwdBuffers() {
  // topdiff buffer in bwdwgt may have differen format with bwddata
  // so have two different buffer
  topDiff_.reset(new MkldnnBuffer());
  topDiffBwdWgt_.reset(new MkldnnBuffer());

  CHECK(weight_->getWGrad()) << "should have weight grad anyway";
  wgtDiff_.reset(new MkldnnBuffer());

  // always create bias buffer even do not hasbias for empty bias buffer
  biasDiff_.reset(new MkldnnBuffer());

  if (!hasBotGrad()) {
    return;
  }

  botDiff_.reset(new MkldnnBuffer());
}

void MkldnnFcLayer::resetDnnBwdUserLayout() {
  const MatrixPtr& topGrad = getDnnOutputGrad();
  const MatrixPtr& wgtGrad = weight_->getWGrad();

  real *topGradData = topGrad->getData();
  real *wgtGradData = wgtGrad->getData();
  topDiff_->initUser(topGradData, topDims_, topFmt_, engine_);
  topDiffBwdWgt_->initUser(topGradData, topDims_, topFmt_, engine_);
  wgtDiff_->initUser(wgtGradData, wgtDims_, wgtFmt_, engine_);

  if (hasBias_) {
    real* biasGradData = biases_->getWGrad()->getData();
    biasDiff_->initUser(biasGradData, biasDims_, biasFmt_, engine_);
  } else {
    biasDiff_->initUser(NULL, biasDims_, biasFmt_, engine_);
  }

  if (!hasBotGrad()) {
    return;
  }

  const MatrixPtr& botGrad = getDnnInputGrad(0);
  real* botGradData = botGrad->getData();
  botDiff_->initUser(botGradData, botDims_, botFmt_, engine_);
}

void MkldnnFcLayer::resetDnnBwdWgtPD(
  std::shared_ptr<fc_bwdWgt::primitive_desc>& bwdWgtPD) {
  std::shared_ptr<fc_fwd::primitive_desc> bwdFwdPD;
  std::shared_ptr<fc_bwdWgt::desc> bwdWgtDesc;

  getBwdFwdPD(bwdFwdPD);

  if (hasBias_) {
    bwdWgtDesc.reset(new fc_bwdWgt::desc(
      botData_->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_),
      biasData_->getIntlMD(),
      MkldnnBuffer::getMD(topDims_)));
  } else {
    bwdWgtDesc.reset(new fc_bwdWgt::desc(
      botData_->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_),
      MkldnnBuffer::getMD(topDims_)));
  }
  bwdWgtPD.reset(new fc_bwdWgt::primitive_desc(
    *bwdWgtDesc, engine_, *bwdFwdPD));
  CHECK(botData_->getIntlPD() == bwdWgtPD->src_primitive_desc());
  if (hasBias_) {
    CHECK(biasData_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc());
  }
}

void MkldnnFcLayer::resetDnnBwdDataPD(
  std::shared_ptr<fc_bwdData::primitive_desc>& bwdDataPD) {
  if (!hasBotGrad()) {
    return;
  }

  std::shared_ptr<fc_fwd::primitive_desc> bwdFwdPD;
  std::shared_ptr<inner_product_backward_data::desc> bwdDataDesc;

  getBwdFwdPD(bwdFwdPD);

  bwdDataDesc.reset(new inner_product_backward_data::desc(
    MkldnnBuffer::getMD(botDims_),
    wgtData_->getIntlMD(),
    MkldnnBuffer::getMD(topDims_)));
  bwdDataPD.reset(new inner_product_backward_data::primitive_desc(
    *bwdDataDesc, engine_, *bwdFwdPD));

// CHECK(botData_->getIntlPD() == bwdDataPD->diff_src_primitive_desc());
  CHECK(wgtData_->getIntlPD() == bwdDataPD->weights_primitive_desc());
// CHECK(topDiffBwdWgt_->getIntlPD() == bwdDataPD->diff_dst_primitive_desc());
}

void MkldnnFcLayer::getBwdFwdPD(
  std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>& bwdFwdPD) {
  prop_kind pk = prop_kind::forward;
  std::shared_ptr<fc_fwd::desc> bwdFwdDesc;
  bwdFwdDesc.reset(new fc_fwd::desc(pk,
      botData_->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_),
      MkldnnBuffer::getMD(topDims_)));
  bwdFwdPD.reset(new fc_fwd::primitive_desc(*bwdFwdDesc, engine_));
}

void MkldnnFcLayer::resetDnnBwdIntlLayout(
  const std::shared_ptr<fc_bwdWgt::primitive_desc>& bwdWgtPD,
  const std::shared_ptr<fc_bwdData::primitive_desc>& bwdDataPD) {

  const MatrixPtr& wgtGrad = weight_->getWGrad();
  real *wgtGradData = wgtGrad->getData();
  wgtDiff_->resetUser(wgtGradData, bwdWgtPD->diff_weights_primitive_desc());
  wgtDiff_->initCvt(bwdWgtPD->diff_weights_primitive_desc(), dnnCvtNoNeed);
  CHECK_EQ(wgtDiff_->getIntlSize(), wgtData_->getIntlSize())
    << "can not use mkldnn wgt since memory size does not equal";
  CHECK(wgtDiff_->getUserPD() == wgtDiff_->getIntlPD());

  // topdiff for bwdwgt may differ for bwddata
  topDiffBwdWgt_->initCvt(bwdWgtPD->diff_dst_primitive_desc(), dnnCvtUser2Intl);
  VLOG(3) << "topdiff for bwdweight flow --- "
    << DNN_FMTS[topDiffBwdWgt_->getIntlFmt()]
    << " <<< "
    << DNN_FMTS[topDiffBwdWgt_->getUserFmt()];

  if (hasBias_) {
    CHECK(biasDiff_->getUserPD() == bwdWgtPD->diff_bias_primitive_desc())
      << "should always be format::x, or changed in new mkldnn version";
  }
  biasDiff_->initCvt(biasDiff_->getUserPD(), dnnCvtNoNeed);

  if (!hasBotGrad()) {
    return;
  }

  CHECK(bwdDataPD) << "should have bwdDataPD";
  botDiff_->initCvt(bwdDataPD->diff_src_primitive_desc(), dnnCvtIntl2User);
  topDiff_->initCvt(bwdDataPD->diff_dst_primitive_desc(), dnnCvtUser2Intl);
}

void MkldnnFcLayer::resetDnnBwdHandle(
  const std::shared_ptr<fc_bwdWgt::primitive_desc>& bwdWgtPD,
  const std::shared_ptr<fc_bwdData::primitive_desc>& bwdDataPD) {

  CHECK(bwdWgtPD);
  // bias buffer will automatic be empty memory if do not have bias
  bwdWgt_.reset(new inner_product_backward_weights(*bwdWgtPD,
    *(botData_->getIntlMem()), *(topDiffBwdWgt_->getIntlMem()),
    *(wgtDiff_->getIntlMem()), *(biasDiff_->getIntlMem())));
  //memory emptyBias = memory(memory::primitive_desc(memory::desc({}, memory::data_type::f32, biasFmt_), engine_))

  if (!hasBotGrad()) {
    return;
  }

  CHECK(bwdDataPD);
  bwdData_.reset(new inner_product_backward_data(*bwdDataPD, 
    *(topDiff_->getIntlMem()), *(wgtData_->getIntlMem()),
    *(botDiff_->getIntlMem())));

}

void MkldnnFcLayer::backwardDnnData() {
  if (!hasBotGrad()) {
    return;
  }

  real* botGradData = getDnnInputGrad(0)->getData();
  real* topGradData = getDnnOutputGrad()->getData();
  std::vector<primitive> pipeline;
  topDiff_->submitCvt(pipeline, topGradData);
  pipeline.push_back(*bwdData_);
  botDiff_->submitCvt(pipeline, botGradData);
  stream(stream::kind::eager).submit(pipeline).wait();
}

void MkldnnFcLayer::backwardDnnWgt() {
  real* botValData = getInputValue(0)->getData();
  real* topGradData = getDnnOutputGrad()->getData();

  std::vector<primitive> pipeline;
  topDiffBwdWgt_->submitCvt(pipeline, topGradData);
  botData_->submitCvt(pipeline, botValData);
  pipeline.push_back(*bwdWgt_);
  // no need to cvt wgt, user and intl are the same format
  // wgtDiff_->submitCvt(pipeline, wgtGradData);
  // no need to submit cvt bias since biasfmt would not be changed
  stream(stream::kind::eager).submit(pipeline).wait(); 
}

void MkldnnFcLayer::updateParameter(const UpdateCallback &callback) {
  if (weight_->getWGrad()) {
    weight_->getParameterPtr()->incUpdate(callback);
  }
  
  if (hasBias_) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

// others
void MkldnnFcLayer::resetBotValUserWithPrevLayout() {
  const std::shared_ptr<memory::desc>& prvMD = getPrev(0)->getTopDataMD();
  if (!prvMD) {
    LOG(WARNING) << "should have prev dnn layout!!! double check it!";
    return;
  }

  const MatrixPtr& botVal = getInputValue(0);
  real *botValData = botVal->getData();
  botData_->resetUser(botValData, *prvMD, engine_);
  VLOG(4) << "use prev data fmt: " << DNN_FMTS[botData_->getUserFmt()];

  // TODO: double check does it right?
  botDims_ = botData_->getUserDims();
  LOG(INFO) << "-----------"<<botDims_[0] << "," << botDims_[1];
  botFmt_ = (mkldnn::memory::format)botData_->getUserFmt();
}

void MkldnnFcLayer::resetTopDiffUserWithNextLayout() {
  const std::shared_ptr<memory::desc>& topMD = getTopDiffMD();
  if (!topMD) {
    LOG(WARNING) << "should have next dnn layout!!! double check it!";
    return;
  }

  const MatrixPtr& topGrad = getDnnOutputGrad();
  real *topGradData = topGrad->getData();
  topDiff_->resetUser(topGradData, *topMD, engine_);
  topDiffBwdWgt_->resetUser(topGradData, *topMD, engine_);

  bool isNCHW = topDiff_->getUserFmt() == memory::format::nchw;
  if (isNCHW && oh_ == ow_ && oh_ == 1) {
    // if prv is nchw and h==w==1, use nc instead
    CHECK_EQ(topFmt_, memory::format::nc);
    topDiff_->resetUser(topGradData, topDims_, topFmt_, engine_);
    topDiffBwdWgt_->resetUser(topGradData, topDims_, topFmt_, engine_);
    VLOG(4) << "use nc diff fmt";
  } else {
    VLOG(4) << "use prev diff fmt: " << DNN_FMTS[topDiff_->getUserFmt()];
  }
}

void MkldnnFcLayer::resetTopValUserLayout(
  const std::shared_ptr<fc_fwd::primitive_desc>& fwdPD) {
  CHECK(fwdPD);
  const MatrixPtr& topVal = getOutputValue();
  real *topValData = topVal->getData();
  topData_->resetUser(topValData, fwdPD->dst_primitive_desc());
  CHECK(topData_->getUserPD() == topData_->getIntlPD());
}

void MkldnnFcLayer::resetBotGradUserLayout(
  const std::shared_ptr<fc_bwdData::primitive_desc>& bwdDataPD) {
  if (!hasBotGrad()) {
    return;
  }

  CHECK(bwdDataPD);
  const MatrixPtr& botGrad = getDnnInputGrad(0);
  real* botGradData = botGrad->getData();
  botDiff_->resetUser(botGradData, bwdDataPD->diff_src_primitive_desc());
  CHECK(botDiff_->getUserPD() == botDiff_->getIntlPD());
}

void MkldnnFcLayer::keepLayoutToNextLayert() {
  setTopDataMD(topData_->getUserMD());
  VLOG(4) << "set next data fmt: " << DNN_FMTS[topData_->getUserFmt()];
}

void MkldnnFcLayer::keepLayoutToPrevLayer() {
  inputLayers_[0]->setTopDiffMD(this->getName(), botDiff_->getUserMD());
  VLOG(4) << "set prev diff fmt: " << DNN_FMTS[botDiff_->getUserFmt()];
}

inline bool MkldnnFcLayer::hasBotGrad() {
  return getDnnInputGrad(0) != nullptr ? true : false;
}



}  // namespace paddle
