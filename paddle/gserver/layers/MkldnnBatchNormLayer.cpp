/* Copyright (c) 2017 */


#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnBatchNormLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_bn, MkldnnBatchNormLayer);

const real MkldnnBatchNormLayer::EPS = 1E-5;

bool MkldnnBatchNormLayer::initDnnWgt(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* initialize the weightList */
  if (!useMkldnnWgt_) {
    selfScaleShiftData_ = Matrix::create(2, oc_, false, false);
    selfScaleShiftDiff_ = Matrix::create(2, oc_, false, false);
    selfScaleShiftData_->zeroMem();
    selfScaleShiftDiff_->zeroMem();
    weight_.reset(new Weight(1, oc_, parameters_[0]));
    if (biasParameter_.get() != NULL) {
      biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
    }
  } else {
    weight_.reset(new Weight(2, oc_, parameters_[0]));
  }

  localMean_ = Matrix::create(1, oc_, false, false);
  localVar_ = Matrix::create(1, oc_, false, false);
  localMean_->zeroMem();
  localVar_->zeroMem();

  movingAvgFraction_ = config_.moving_average_fraction();
  movingMean_.reset(new Weight(1, oc_, parameters_[1]));
  movingVar_.reset(new Weight(1, oc_, parameters_[2]));
  return true;
}

void MkldnnBatchNormLayer::loadConfig() {
  // first is Input
  // other two are created in config_parser.py for saving moving mean and var
  CHECK_EQ(inputLayers_.size(), 3U);
  CHECK_EQ(parameters_.size(), 3U);
  //const ImageConfig& conf = config_.inputs(0).image_conf();
  //oc_ = conf.channels();
  // only care about oc
  CHECK(config_.has_num_filters()) << "should have set filter(channel) number";
  oc_ = config_.num_filters();
  if (config_.has_use_mkldnn_wgt()) {
    useMkldnnWgt_ = config_.use_mkldnn_wgt();
  }
  if (config_.has_use_mkldnn_seq()) {
    useMkldnnSeq_ = config_.use_mkldnn_seq();
  }
  if (config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }

  VLOG(1) << "--- " << (useGlobalStats_ ? "use" : "do not use")
    << " --- global stats";
}

void MkldnnBatchNormLayer::reshapeOutputInfo() {
  size_t idx = 0;
  // reshape bs and mkl seqlen
  outputMatH_ = inputMatH_;
  seqLen_ = getInput(idx).getMklSeqLen();
  if (useMkldnnSeq_) {
    CHECK_GE(seqLen_, 1) << getName() << " seq length should larger than 1";
  }
  if (seqLen_ > 1) {
    bs_ = outputMatH_ / seqLen_;
    CHECK_EQ(bs_ * seqLen_, outputMatH_) << "maybe caused by un-divisible";
  } else {
    bs_ = outputMatH_;
  }
  // reshape image size
  ih_ = inputLayers_[idx]->getOutput().getFrameHeight();
  iw_ = inputLayers_[idx]->getOutput().getFrameWidth();
  if (ih_ == 0) ih_ = 1;
  if (iw_ == 0) iw_ = 1;
  ic_ = inputMatW_ / (ih_ * iw_);
  CHECK_EQ(ic_ * ih_ * iw_, inputMatW_) << "maybe caused by un-divisible";
  CHECK_EQ(ic_, oc_) << "Do not support change channel number,"
    << "which means the weight size would be changed";
  oh_ = ih_;
  ow_ = iw_;
  outputMatW_ = inputMatW_;
  config_.set_size(outputMatW_);

  // reset output image size
  resetOutput(outputMatH_, outputMatW_);
  getOutput().setFrameHeight(oh_);
  getOutput().setFrameWidth(ow_);
}

void MkldnnBatchNormLayer::resetDnnFwd() {
//  CHECK(bs_ == getInput(0).getBatchSize()) << "batchsize should equal";
  if (useMkldnnWgt_) {
    useScaleShift_ = weight_ && weight_->getW();
  } else {
    bool hasShift = (biases_ && biases_->getW());
    bool hasScale = (weight_ && weight_->getW());
    CHECK_EQ(hasScale, hasShift)
      << "only support both weigt and bias at same time, or neither";
    useScaleShift_ = (hasShift && hasScale);
  }

  // in train always calculate mean and var, so GlobalStats must be false
  // in test depends on manual choice
  useGlobalStats_ = (passType_ == PASS_TEST);
  if (passType_ == PASS_TEST && config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }
  if (passType_ != PASS_TEST && useGlobalStats_ == true) {
    LOG(WARNING) << "use_global_stats is invalid setting in training phase";
    useGlobalStats_ = false;
  }

  // start dnn
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  prop_kind fwdpk = (passType_ == PASS_TEST) ? prop_kind::forward_scoring
    : prop_kind::forward_training;
  flags_ = 0u;
  if (useGlobalStats_)
    flags_ = (flags_ | batch_normalization_flag::use_global_stats);
  if (useScaleShift_)
    flags_ = (flags_ | batch_normalization_flag::use_scale_shift);
  if (!hasInited_) {
    hasInited_ = true;
    // only do once
    if (useGlobalStats_) {
      localMean_->copyFrom(*(movingMean_->getW()));
      localVar_->copyFrom(*(movingVar_->getW()));
    }
    if (useScaleShift_ && useMkldnnWgt_ && passType_ != PASS_TEST) {
      // re-randomize scale and zero shift(bias in paddle) just as paddle did
      const ParameterConfig& wgtConfig = parameters_[0]->getConfig();
      VectorPtr wgt = parameters_[0]->getBuf(PARAMETER_VALUE);
      VectorPtr scale(new CpuVector(oc_, wgt->getMemoryHandle(), 0));
      VectorPtr shift(new CpuVector(oc_, wgt->getMemoryHandle(), oc_));
      Parameter::randomize(scale, wgtConfig);
      shift->zeroMem();
    }
  }
  /// init mkldnn forward ******************************************************
  // 1. create mkldnn data buffer
  botData_.reset(new MkldnnBuffer());
  topData_.reset(new MkldnnBuffer());
  if (useScaleShift_) {
    dataScaleShift_.reset(new MkldnnBuffer());
  }
  if (useGlobalStats_ || passType_ != PASS_TEST) {
    mean_.reset(new MkldnnBuffer());
    var_.reset(new MkldnnBuffer());
  }
  // init dim structure that describes user data.
  botDims_ = {bs_, ic_, ih_, iw_};
  topDims_ = {bs_, oc_, oh_, ow_};  // == botDims_
  wgtDims_ = {2, oc_};  // scale and shift
  wgtFmt_ = memory::format::nc;
  biasDims_ = {oc_};
  biasFmt_ = memory::format::x;

  real *botDataData = getPrev(0)->getOutputValue()->getData();
  real *topDataData = getOutputValue()->getData();
  // 2. init user
  topData_->initUser(topDataData, topDims_, topFmt_, eg);
  const std::shared_ptr<memory::desc>& prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    botData_->resetUser(botDataData, *prvMD, eg);
    bool isNC = botData_->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(ih_ == iw_ && ih_ == 1)
        << "iw, ih must be 1 with nc input";
      // do not support nc input, so change to nchw
      botData_->resetUser(botDataData, botDims_, botFmt_, eg);
      VLOG(4) << "use nchw data fmt";
    } else {
      VLOG(4) << "use prev data fmt: " << DNN_FMTS[botData_->getUserFmt()];
    }
  } else {
    botData_->initUser(botDataData, botDims_, botFmt_, eg);
  }
  // 3. create fwd desc
  std::shared_ptr<batch_normalization_forward::desc> fwdDesc;
  fwdDesc.reset(new batch_normalization_forward::desc(fwdpk,
    // TODO(TJ): use any if MKLDNN ready
    botData_->getUserMD(),  // MkldnnBuffer::getMD(botDims_),
    EPS, flags_));
  fwdPD_.reset(new batch_normalization_forward::primitive_desc(*fwdDesc, eg));
  // 4. init  conversion
  // src_primitive_desc == dst_primitive_desc in batch_normalization_forward
  botData_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtUser2Intl);
  if (nextIsDnn_) {
    topData_->resetUser(topDataData, fwdPD_->dst_primitive_desc());
    setTopDataMD(topData_->getUserMD());
    VLOG(4) << "set next data fmt: " << DNN_FMTS[topData_->getUserFmt()];
  }
  topData_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtIntl2User);
  // weight(scale) and bias(shift)
  if (useScaleShift_) {
    real *ssData = useMkldnnWgt_ ? weight_->getW()->getData()
      : selfScaleShiftData_->getData();
    dataScaleShift_->initUser(ssData, wgtDims_, wgtFmt_, eg);
    CHECK(dataScaleShift_->getUserPD() == fwdPD_->weights_primitive_desc())
      << "scaleshiftPD should be {2, oc} with nc format, check mkldnn version.";
    dataScaleShift_->initCvt(
      fwdPD_->weights_primitive_desc(), dnnCvtUser2Intl);
  } else {
    LOG(WARNING) << "Are you sure do not need scale and shift???";
  }
  // mean and var
  if (useGlobalStats_ || passType_ != PASS_TEST) {
    MkldnnBufferPtr tmppd(new MkldnnBuffer());
    tmppd->initUser(localMean_->getData(), biasDims_, biasFmt_, eg);
    CHECK(tmppd->getUserPD() == fwdPD_->mean_primitive_desc()
      && fwdPD_->mean_primitive_desc() == fwdPD_->variance_primitive_desc())
      << "assert both mean and var size are {oc_}";
    // use exactly the same size and format of mean and var in mkldnn
    mean_->initUser(localMean_->getData(), fwdPD_->mean_primitive_desc());
    mean_->initCvt(fwdPD_->mean_primitive_desc(), dnnCvtNoNeed);
    var_->initUser(localVar_->getData(), fwdPD_->variance_primitive_desc());
    var_->initCvt(fwdPD_->variance_primitive_desc(), dnnCvtNoNeed);
  }
  // 5. create fwd handle
  if (passType_ == PASS_TEST) {
    if (useGlobalStats_) {
      fwd_.reset(useScaleShift_
        ? new batch_normalization_forward(*fwdPD_, *botData_->getIntlMem(),
            (const primitive::at)(*mean_->getIntlMem()),
            (const primitive::at)(*var_->getIntlMem()),
            *dataScaleShift_->getIntlMem(),
            *topData_->getIntlMem())
        : new batch_normalization_forward(*fwdPD_, *botData_->getIntlMem(),
            (const primitive::at)(*mean_->getIntlMem()),
            (const primitive::at)(*var_->getIntlMem()),
            *topData_->getIntlMem()));
    } else {
      fwd_.reset(useScaleShift_
        ? new batch_normalization_forward(*fwdPD_, *botData_->getIntlMem(),
            *dataScaleShift_->getIntlMem(), *topData_->getIntlMem())
        : new batch_normalization_forward(*fwdPD_, *botData_->getIntlMem(),
            *topData_->getIntlMem()));
    }
  } else {
    CHECK(useGlobalStats_ == false)
      << "useGlobalStats should be false in training";
    fwd_.reset(useScaleShift_
      ? new batch_normalization_forward(*fwdPD_, *botData_->getIntlMem(),
          *dataScaleShift_->getIntlMem(), *topData_->getIntlMem(),
          *mean_->getIntlMem(), *var_->getIntlMem())
      : new batch_normalization_forward(*fwdPD_, *botData_->getIntlMem(),
          *topData_->getIntlMem(),
          *mean_->getIntlMem(), *var_->getIntlMem()));
  }
}

void MkldnnBatchNormLayer::resetDnnBwd() {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  LayerPtr prevLayer = getPrev(0);
  if (NULL == prevLayer->getOutputGrad()) {
    LOG(FATAL) << "maybe do not set batchnorm after data layer!!!";
  }
  // 1. create mkldnn diff buffer
  botDiff_.reset(new MkldnnBuffer());
  topDiff_.reset(new MkldnnBuffer());
  if (useScaleShift_) {
    diffScaleShift_.reset(new MkldnnBuffer());
  }
  // 2. init user, prepare top diff if use dnn input
  real* botDiffData = getDnnInputGrad(0)->getData();
  real *topDiffData = getOutputGrad()->getData();
  botDiff_->initUser(botDiffData, botDims_, botFmt_, eg);
  topDiff_->initUser(topDiffData, topDims_, topFmt_, eg);
  const std::shared_ptr<mkldnn::memory::desc>& inputDiffMD = getTopDiffMD();
  if (inputDiffMD) {
    topDiff_->resetUser(topDiffData, *inputDiffMD, eg);
    bool isNC = topDiff_->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(oh_ == ow_ && oh_ == 1)
        << "ow, oh must be 1 with nc input";
      // do not support nc input, so change to nchw
      topDiff_->resetUser(topDiffData, topDims_, topFmt_, eg);
      VLOG(4) << "use nchw diff fmt";
    } else {
      VLOG(4) << "use prev diff fmt: " << DNN_FMTS[topDiff_->getUserFmt()];
    }
  }
  // 3. create backward desc ---------------------------------------------------
  std::shared_ptr<batch_normalization_backward::desc> bwdDesc;
  std::shared_ptr<batch_normalization_backward::primitive_desc> bwdPD;
  bwdDesc.reset(new batch_normalization_backward::desc(prop_kind::backward,
    // TODO(TJ): use any if MKLDNN ready
    botData_->getIntlMD(), botData_->getIntlMD(), EPS, flags_));
  bwdPD.reset(new batch_normalization_backward::primitive_desc(
    *bwdDesc, eg, *fwdPD_));
  CHECK(bwdPD->diff_weights_primitive_desc() == dataScaleShift_->getIntlPD());
  CHECK(bwdPD->weights_primitive_desc() == fwdPD_->weights_primitive_desc());
  CHECK(bwdPD->mean_primitive_desc() == fwdPD_->mean_primitive_desc());
  CHECK(bwdPD->variance_primitive_desc() == fwdPD_->variance_primitive_desc());
  // 4. init conversion
  topDiff_->initCvt(botData_->getIntlPD(), dnnCvtUser2Intl);
  if (prevIsDnn_[0]) {
    botDiff_->resetUser(botDiffData, botData_->getIntlPD());
    prevLayer->setTopDiffMD(this->getName(), botDiff_->getUserMD());
    VLOG(4) << "set next diff format: " << DNN_FMTS[botDiff_->getUserFmt()];
  }
  botDiff_->initCvt(botData_->getIntlPD(), dnnCvtIntl2User);
  // weight(scale) and bias(shift)
  if (useScaleShift_) {
    real *ssDiff = useMkldnnWgt_ ? weight_->getWGrad()->getData()
      : selfScaleShiftDiff_->getData();
    diffScaleShift_->initUser(ssDiff, wgtDims_, wgtFmt_, eg);
    CHECK(diffScaleShift_->getUserPD() == bwdPD->diff_weights_primitive_desc())
      << "scaleshiftPD should be {2,oc} with nc format, check mkldnn version.";
    diffScaleShift_->initCvt(
      bwdPD->diff_weights_primitive_desc(), dnnCvtIntl2User);
  }
  // 5. create bwd data and weight handle
  bwd_.reset(useScaleShift_
    ? new batch_normalization_backward(*bwdPD,
        *botData_->getIntlMem(),
        *mean_->getIntlMem(), *var_->getIntlMem(),
        *topDiff_->getIntlMem(), *dataScaleShift_->getIntlMem(),
        *botDiff_->getIntlMem(), *diffScaleShift_->getIntlMem())
    : new batch_normalization_backward(*bwdPD,
        *botData_->getIntlMem(),
        *mean_->getIntlMem(), *var_->getIntlMem(),
        *topDiff_->getIntlMem(), *botDiff_->getIntlMem()));
}

void MkldnnBatchNormLayer::calMovingMeanAndVar() {
  // calculating and saving moving mean and variance
  CHECK_EQ(useGlobalStats_, false);
  MatrixPtr movingMean = movingMean_->getW();
  MatrixPtr movingVar = movingVar_->getW();
  if (!useGpu_ && FLAGS_trainer_count > 1) {
    auto mvMean = std::dynamic_pointer_cast<SharedCpuMatrix>(movingMean);
    auto mvVar = std::dynamic_pointer_cast<SharedCpuMatrix>(movingVar);
    CHECK(mvMean && mvVar);
    mvMean->add(*localMean_, movingAvgFraction_, 1.0 - movingAvgFraction_);
    mvVar->add(*localVar_, movingAvgFraction_, 1.0 - movingAvgFraction_);
  } else {
    movingMean->add(*localMean_, movingAvgFraction_, 1.0 - movingAvgFraction_);
    // here var is v^2
    movingVar->add(*localVar_, movingAvgFraction_, 1.0 - movingAvgFraction_);
  }
}

void MkldnnBatchNormLayer::submitDnnFwd() {
  real *botDataData = getPrev(0)->getOutputValue()->getData();
  real *topDataData = getOutputValue()->getData();

  std::vector<primitive> pipeline;
  // data bottom
  botData_->submitCvt(pipeline, botDataData);

  // prepare weight data of scale and shift
  if (useScaleShift_) {
    real *wgtDataData = useMkldnnWgt_ ? weight_->getW()->getData()
      : selfScaleShiftData_->getData();
    if (!useMkldnnWgt_) {
      // copy data from paddle weight and bias
      memcpy(selfScaleShiftData_->getData(), weight_->getW()->getData(),
        sizeof(real) * oc_);
      memcpy(selfScaleShiftData_->getData() + oc_, biases_->getW()->getData(),
        sizeof(real) * oc_);
    }
    dataScaleShift_->submitCvt(pipeline, wgtDataData);
  }
  pipeline.push_back(*fwd_);
  topData_->submitCvt(pipeline, topDataData);
  stream(stream::kind::eager).submit(pipeline).wait();

  // calculating and saving moving mean and variance
  if (passType_ != PASS_TEST) {
    calMovingMeanAndVar();
  }

  // activation
  forwardActivation();
}

void MkldnnBatchNormLayer::submitDnnBwd(const UpdateCallback &callback) {
  backwardActivation();

  real* botDataData = getPrev(0)->getOutputValue()->getData();
  real* topDiffData = getDnnOutputGrad()->getData();
  real* botDiffData = getDnnInputGrad(0)->getData();

  std::vector<primitive> pipeline;
  topDiff_->submitCvt(pipeline, topDiffData);
  botData_->submitCvt(pipeline, botDataData);
  if (useScaleShift_) {
    real *wgtDataData = useMkldnnWgt_ ? weight_->getW()->getData()
      : selfScaleShiftData_->getData();
    dataScaleShift_->submitCvt(pipeline, wgtDataData);
  }
  pipeline.push_back(*bwd_);
  botDiff_->submitCvt(pipeline, botDiffData);
  // output scaleshift diff
  if (useScaleShift_) {
    real *wgtDiffData = useMkldnnWgt_ ? weight_->getWGrad()->getData()
      : selfScaleShiftDiff_->getData();
    diffScaleShift_->submitCvt(pipeline, wgtDiffData);
  }
  stream(stream::kind::eager).submit(pipeline).wait();

  // update diff
  if (useScaleShift_) {
    weight_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
