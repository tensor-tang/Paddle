/* Copyright (c) 2017 */

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnFcLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_fc, MkldnnFcLayer);

bool MkldnnFcLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  // only support 1 input layer by now
  CHECK_EQ(config_.inputs_size(), 1) << "Only support one input layer yet!";
  CHECK_EQ(inputLayers_.size(), parameters_.size());

  CHECK_EQ(oc_, getSize());
  hasSpatial_ = false;

  // create a new weight
  size_t height, width;
  CHECK(!parameters_[0]->isSparse()) << "Do not support sparse yet";
  CHECK_EQ(parameters_[0]->getSize(), oc_ * inputLayerSize_);

  selfWgtData_.push_back(nullptr);
  selfWgtDiff_.push_back(nullptr);
  if (!useMkldnnWgt_) {
    height = inputLayerSize_;
    width = oc_;
    selfWgtData_[0] = Matrix::create(width, height, false, false);
    selfWgtDiff_[0] = Matrix::create(width, height, false, false);
    selfWgtData_[0]->zeroMem();
    selfWgtDiff_[0]->zeroMem();
  } else {
    height = oc_;
    width = inputLayerSize_;
  }
  Weight* w = new Weight(height, width, parameters_[0], 0);
  weights_.emplace_back(w);

  // initialize biases_
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
  }
  return true;
}

// load the settings from proto
void MkldnnFcLayer::loadConfig() {
  CHECK_EQ(config_.inputs_size(), 1) << "Only support one input config!";
  if (config_.has_use_mkldnn_wgt()) {
    useMkldnnWgt_ = config_.use_mkldnn_wgt();
  }
  const FCConfig &conf = config_.inputs(0).fc_conf();
  inputLayerSize_ = conf.dim_in();
  oc_ = conf.dim_out();
}

// keep for paddle parameter server
void MkldnnFcLayer::prefetch() {
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    auto* sparseParam =
        dynamic_cast<SparsePrefetchRowCpuMatrix*>(weights_[i]->getW().get());
    if (sparseParam) {
      MatrixPtr input = getInputValue(i);
      sparseParam->addRows(input);
    }
  }
}

void MkldnnFcLayer::reshapeOutput() {
  CHECK_EQ(inputLayers_.size(), 1UL);
  CHECK_EQ(inputLayerSize_, inputMatW_) << "should not change input size,"
    << "which means the weight size would be changed";
  size_t idx = 0;  // input index
  // reshape bs and mkl seqlen
  outputMatH_ = inputMatH_;
  seqLen_ = getInput(idx).getMklSeqLen();
  if (seqLen_ > 1) {
    bs_ = outputMatH_ / seqLen_;
    CHECK_EQ(bs_ * seqLen_, outputMatH_) << "maybe caused by un-divisible";
  } else {
    bs_ = outputMatH_;
  }

  // reshape image size and check should not change inputLayerSize
  ih_ = inputLayers_[idx]->getOutput().getFrameHeight();
  iw_ = inputLayers_[idx]->getOutput().getFrameWidth();
  if (ih_ == 0) ih_ = 1;
  if (iw_ == 0) iw_ = 1;
  hasSpatial_ = true;
  if (ih_ == 1 && iw_ == 1) 
    hasSpatial_ = false;
  ic_ = inputMatW_ / (ih_ * iw_);
  CHECK_EQ(ic_ * ih_ * iw_, inputMatW_) << "maybe caused by un-divisible";

  // cal out size
  oh_ = 1;
  ow_ = 1;
  outputMatW_ = oc_ * oh_ * ow_;
  config_.set_size(outputMatW_);

  // reset output image size
  resetOutput(outputMatH_, outputMatW_);
  getOutput().setFrameHeight(oh_);
  getOutput().setFrameWidth(ow_);
}

void MkldnnFcLayer::resetDnnFwd() {
  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  prop_kind pk = prop_kind::forward;
  bool hasBias = (biases_ && biases_->getW());
  // create dim structure that describes user data.
  if (!hasSpatial_) {
    botDims_ = {bs_, ic_};
    wgtDims_ = {oc_, ic_};  // transpose from paddle weight
    botFmt_ = memory::format::nc;
    wgtFmt_ = memory::format::oi;
  } else {
    botDims_ = {bs_, ic_, ih_, iw_};
    wgtDims_ = {oc_, ic_, ih_, iw_};
    botFmt_ = memory::format::nchw;
    wgtFmt_ = memory::format::oihw;
  }
  topDims_ = {bs_, oc_};
  topFmt_ = memory::format::nc;
  biasDims_ = {oc_};
  biasFmt_ = memory::format::x;

  bool hasCvtTopData = false;
  bool hasCvtBiasData = false;
  // 1. create mkldnn buffer, only have one output and bias buffer
  topData_.reset(new MkldnnBuffer());
  if (hasBias) {
    biasData_.reset(new MkldnnBuffer());
  }
  // 2. init user top and bias
  real *topDataData = getOutputValue()->getData();
  topData_->initUser(topDataData, topDims_, topFmt_, eg);
  if (hasBias) {
    real *biasDataData = biases_->getW()->getData();
    biasData_->initUser(biasDataData, biasDims_, biasFmt_, eg);
  }
  // TODO(TJ): only care about i==0 yet
  CHECK_EQ(inputLayers_.size(), 1);
//  CHECK_EQ(botDatas_.size(), inputLayers_.size());
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize()) << "batchsize should equal";
    /// 1. create buffer, could be vector later
    botData_.reset(new MkldnnBuffer());
    wgtData_.reset(new MkldnnBuffer());
    // 2. init user memory of bottom, weights and bias
    real *botDataData = getPrev(i)->getOutputValue()->getData();
    real *wgtDataData = useMkldnnWgt_ ? weights_[i]->getW()->getData()
        : selfWgtData_[i]->getData();
    botData_->initUser(botDataData, botDims_, botFmt_, eg);
    wgtData_->initUser(wgtDataData, wgtDims_, wgtFmt_, eg);
    // 3. create fc desc
    std::shared_ptr<inner_product_forward::desc> fwdDesc;
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> fwdPD;
    const std::shared_ptr<memory::desc>& prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      botData_->resetUser(botDataData, *prvMD, eg);
      VLOG(4) << "use prev data fmt: " << DNN_FMTS[botData_->getUserFmt()];
    }
    if (hasBias) {
      fwdDesc.reset(new inner_product_forward::desc(pk,
          MkldnnBuffer::getMD(botDims_),
          MkldnnBuffer::getMD(wgtDims_),
          MkldnnBuffer::getMD(biasDims_),
          MkldnnBuffer::getMD(topDims_)));
    } else {
      fwdDesc.reset(new inner_product_forward::desc(pk,
          MkldnnBuffer::getMD(botDims_),
          MkldnnBuffer::getMD(wgtDims_), MkldnnBuffer::getMD(topDims_)));
    }
    fwdPD.reset(new inner_product_forward::primitive_desc(*fwdDesc, eg));
    // 4. init cvt
    botData_->initCvt(fwdPD->src_primitive_desc(), dnnCvtUser2Intl);
    if (useMkldnnWgt_) {
      wgtDataData = weights_[i]->getW()->getData();
      wgtData_->resetUser(wgtDataData, fwdPD->weights_primitive_desc());
      wgtData_->initCvt(wgtData_->getUserPD(), dnnCvtNoNeed);
      // need check the memory size, should be strictly equal
      CHECK_EQ(wgtData_->getIntlSize(), parameters_[i]->getSize())
        << "can not use mkldnn wgt since memory size does not equal";
    } else {
      wgtData_->initCvt(fwdPD->weights_primitive_desc(), dnnCvtUser2Intl);
    }
    // only init wgt once
    if (!hasInited_) {
      hasInited_ = true;
      if (useMkldnnWgt_) {
        // cvt the initial paddle wgt to mkldnn wgt only once when training
        // in testing phase do not need cvt
        if (passType_ != PASS_TEST) {
          // paddle wgt is transposed
          size_t height = weights_[i]->getW()->getWidth();
          size_t width = weights_[i]->getW()->getHeight();
          MatrixPtr initWgt = Matrix::create(height, width, false, false);
          initWgt->copyFrom(*(weights_[i]->getW()));
          initWgt = initWgt->getTranspose();
          MkldnnBufferPtr tmp(new MkldnnBuffer());
          tmp->initUser(initWgt->getData(), wgtDims_, wgtFmt_, eg);
          tmp->initCvt(fwdPD->weights_primitive_desc(), dnnCvtUser2Intl);
          std::vector<primitive> cvtWgt;
          tmp->submitCvt(cvtWgt);
          stream(stream::kind::eager).submit(cvtWgt).wait();
          real* dst = weights_[i]->getW()->getData();
          memcpy(dst, tmp->getIntlData(), tmp->getIntlSize() * sizeof(real));
        }
      } else {
        // load the initial paddle wgt and cvt only once when scoring
        // in training phase will cvt in every forward
        if (passType_ == PASS_TEST) {
          weights_[i]->getW()->transpose(selfWgtData_[i], false);
          std::vector<primitive> cvtWgt;
          wgtDataData = selfWgtData_[i]->getData();
          wgtData_->submitCvt(cvtWgt, wgtDataData);
          stream(stream::kind::eager).submit(cvtWgt).wait();
        }
      }
    }
    if (hasBias) {
      // only cvt once
      if (!hasCvtBiasData) {
        hasCvtBiasData = true;
        CHECK(biasData_->getUserPD() == fwdPD->bias_primitive_desc())
          << "should always be format::x, or changed in new mkldnn version";
        biasData_->initCvt(biasData_->getUserPD(), dnnCvtNoNeed);
      } else {
        CHECK(biasData_->getIntlPD() == fwdPD->bias_primitive_desc())
          << "all bias formats should equal";
      }
    }
    // cvt topDataData buffer only once, set dnn MemDesc if next is also mkldnn
    if (!hasCvtTopData) {
      hasCvtTopData = true;
      if (nextIsDnn_) {
        topData_->resetUser(topDataData, fwdPD->dst_primitive_desc());
        setTopDataMD(topData_->getUserMD());
        VLOG(4) << "set next data fmt: " << DNN_FMTS[topData_->getUserFmt()];
      }
      topData_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);
    } else {
      CHECK(topData_->getIntlPD() == fwdPD->dst_primitive_desc())
        << "all output formats should equal";
    }
    // 5. create fwd handle
    if (hasBias) {
      fwd_.reset(new inner_product_forward(*fwdPD,
        *(botData_->getIntlMem()), *(wgtData_->getIntlMem()),
        *(biasData_->getIntlMem()), *(topData_->getIntlMem())));
    } else {
      fwd_.reset(new inner_product_forward(*fwdPD,
        *(botData_->getIntlMem()), *(wgtData_->getIntlMem()),
        *(topData_->getIntlMem())));
    }
    if (wgtData_) {
      VLOG(3) << "weight data flow --- "
        << DNN_FMTS[wgtData_->getUserFmt()]
        << " >>> "
        << DNN_FMTS[wgtData_->getIntlFmt()];
    }
  }
}

void MkldnnFcLayer::resetDnnBwd() {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  prop_kind pk = prop_kind::forward;
  bool hasBias = (biases_ && biases_->getWGrad());

  bool hasCvtTopDiffBwdData = false;
  bool hasCvtTopDiffBwdWgt = false;
  bool hasCvtBiasDiff = false;

  // 1. create mkldnn buffer, only have one output and bias buffer
  topDiff_.reset(new MkldnnBuffer());
  topDiffBwdWgt_.reset(new MkldnnBuffer());
  if (hasBias) {
    biasDiff_.reset(new MkldnnBuffer());
  }
  // 2. init user top and bias
  real *topDiffData = getDnnOutputGrad()->getData();
  topDiff_->initUser(topDiffData, topDims_, topFmt_, eg);
  topDiffBwdWgt_->initUser(topDiffData, topDims_, topFmt_, eg);
  if (hasBias) {
    real* biasDiffData = biases_->getWGrad()->getData();
    biasDiff_->initUser(biasDiffData, biasDims_, biasFmt_, eg);
  }
  // use internal top diff if use dnn input
  const std::shared_ptr<mkldnn::memory::desc>& prv = getTopDiffMD();
  if (prv) {
    topDiff_->resetUser(topDiffData, *prv, eg);
    topDiffBwdWgt_->resetUser(topDiffData, *prv, eg);
    bool isNCHW = topDiff_->getUserFmt() == memory::format::nchw;
    if (isNCHW && oh_ == ow_ && oh_ == 1) {
      // if prv is nchw and h==w==1, use nc instead
      topDiff_->resetUser(topDiffData, topDims_, memory::format::nc, eg);
      topDiffBwdWgt_->resetUser(topDiffData, topDims_, memory::format::nc, eg);
      VLOG(4) << "use nc diff fmt";
    } else {
      VLOG(4) << "use prev diff fmt: " << DNN_FMTS[topDiff_->getUserFmt()];
    }
  }
  // TODO(TJ): only care about i==0 yet
  CHECK_EQ(inputLayers_.size(), 1);
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    // 1. create mkldnn buffer and init user
    CHECK(weights_[i]->getWGrad()) << "should have weight anyway";
    wgtDiff_.reset(new MkldnnBuffer());
    real *wgtDiffData = useMkldnnWgt_ ? weights_[i]->getWGrad()->getData()
      : selfWgtDiff_[i]->getData();
    wgtDiff_->initUser(wgtDiffData, wgtDims_, wgtFmt_, eg);
    // 2. prepare backward weight and bias
    std::shared_ptr<inner_product_forward::desc> bwdFwdDesc;
    std::shared_ptr<inner_product_forward::primitive_desc> bwdFwdPD;
    std::shared_ptr<inner_product_backward_weights::desc> bwdWgtDesc;
    std::shared_ptr<inner_product_backward_weights::primitive_desc> bwdWgtPD;
    bwdFwdDesc.reset(new inner_product_forward::desc(pk,
      botData_->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_),
      MkldnnBuffer::getMD(topDims_)));
    bwdFwdPD.reset(new inner_product_forward::primitive_desc(
      *bwdFwdDesc, eg));
    CHECK(hasBias) << "only support with bias in mkldnn";
    bwdWgtDesc.reset(new inner_product_backward_weights::desc(
      botData_->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_),
      biasData_->getIntlMD(),  // bias do not use any
      MkldnnBuffer::getMD(topDims_)));
    bwdWgtPD.reset(new inner_product_backward_weights::primitive_desc(
      *bwdWgtDesc, eg, *bwdFwdPD));
    CHECK(botData_->getIntlPD() == bwdWgtPD->src_primitive_desc());
// CHECK(wgtData_->getIntlPD() == bwdWgtPD->diff_weights_primitive_desc());
    CHECK(biasData_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc());

    // 3. init conversion
    if (useMkldnnWgt_) {
      wgtDiffData = weights_[i]->getWGrad()->getData();
      wgtDiff_->resetUser(wgtDiffData, bwdWgtPD->diff_weights_primitive_desc());
      wgtDiff_->initCvt(bwdWgtPD->diff_weights_primitive_desc(), dnnCvtNoNeed);
      CHECK_EQ(wgtDiff_->getIntlSize(), wgtData_->getIntlSize())
        << "can not use mkldnn wgt since memory size does not equal";
    } else {
      wgtDiff_->initCvt(
        bwdWgtPD->diff_weights_primitive_desc(), dnnCvtIntl2User);
    }
    if (hasBias) {
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
    bwdWgt_.reset(new inner_product_backward_weights(*bwdWgtPD,
      *(botData_->getIntlMem()), *(topDiffBwdWgt_->getIntlMem()),
      *(wgtDiff_->getIntlMem()), *(biasDiff_->getIntlMem())));
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
    botDiff_->initUser(botDiffData, botDims_, botFmt_, eg);
    // 2. init backward data primitive desc
    std::shared_ptr<inner_product_backward_data::desc> bwdDataDesc;
    std::shared_ptr<inner_product_backward_data::primitive_desc> bwdDataPD;
    bwdDataDesc.reset(new inner_product_backward_data::desc(
      MkldnnBuffer::getMD(botDims_),
      wgtData_->getIntlMD(),
      MkldnnBuffer::getMD(topDims_)));
    bwdDataPD.reset(new inner_product_backward_data::primitive_desc(
      *bwdDataDesc, eg, *bwdFwdPD));
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
  real *topDataData = getOutputValue()->getData();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(getInput(i).value) << "The input of 'fc' layer must be matrix";
    real *botDataData = getPrev(0)->getOutputValue()->getData();
    std::vector<primitive> pipeline;
    botData_->submitCvt(pipeline, botDataData);
    if (!useMkldnnWgt_ && passType_ != PASS_TEST) {
      weights_[i]->getW()->transpose(selfWgtData_[i], false);
      real *wgtDataData = selfWgtData_[i]->getData();
      wgtData_->submitCvt(pipeline, wgtDataData);
    }  // else do not need cvt wgt
    pipeline.push_back(*fwd_);
    topData_->submitCvt(pipeline, topDataData);
    stream(stream::kind::eager).submit(pipeline).wait();
  }
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
  if (!useMkldnnWgt_) {  // no need cvt wgt without useMkldnnWgt_
    CHECK(selfWgtData_[idx]);
    real* wgtDataData = selfWgtData_[idx]->getData();
    wgtData_->submitCvt(pipeline, wgtDataData);
  }
  topDiff_->submitCvt(pipeline, topDiffData);
  pipeline.push_back(*bwdData_);
  botDiff_->submitCvt(pipeline, botDiffData);
  stream(stream::kind::eager).submit(pipeline).wait();
}

void MkldnnFcLayer::submitBwdWgts(int idx) {
  real* botDataData = getInputValue(idx)->getData();
  real* topDiffData = getDnnOutputGrad()->getData();
  real* wgtDiffData = weights_[idx]->getWGrad()->getData();

  std::vector<primitive> pipeline;
  topDiffBwdWgt_->submitCvt(pipeline, topDiffData);
  botData_->submitCvt(pipeline, botDataData);
  pipeline.push_back(*bwdWgt_);
  wgtDiff_->submitCvt(pipeline, wgtDiffData);
  // no need to submit cvt bias since biasfmt would not be changed
  stream(stream::kind::eager).submit(pipeline).wait();
}

void MkldnnFcLayer::submitDnnBwd(const UpdateCallback &callback) {
  backwardActivation();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    submitBwdData(i);
    if (weights_[i]->getWGrad()) {
      submitBwdWgts(i);
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
