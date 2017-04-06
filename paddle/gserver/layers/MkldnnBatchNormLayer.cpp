/* Copyright (c) 2017 */


#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnBatchNormLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_batch_norm, MkldnnBatchNormLayer);

const real MkldnnBatchNormLayer::EPS = 1E-5;

bool MkldnnBatchNormLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* initialize the weightList */
  // first is Input in configure
  // other two are created in config_parser.py saving moving mean and var
  CHECK_EQ(inputLayers_.size(), 3U);
  CHECK_EQ(inputLayers_.size(), parameters_.size());
  CHECK_EQ(inputLayers_.size(), size_t(config_.inputs_size()));
  const ImageConfig& conf = config_.inputs(0).image_conf();
  bs_ = 0;
  ic_[0] = conf.channels();
  oc_ = ic_[0];
  if (inputLayers_[0]->getOutput().getFrameHeight() == 0 &&
      inputLayers_[0]->getOutput().getFrameWidth() == 0) {
    iw_[0] = conf.img_size();
    ih_[0] = conf.img_size();
    ow_[0] = conf.img_size();
    oh_[0] = conf.img_size();
  } else {
    iw_[0] = inputLayers_[0]->getOutput().getFrameWidth();
    ih_[0] = inputLayers_[0]->getOutput().getFrameHeight();
    ow_[0] = inputLayers_[0]->getOutput().getFrameWidth();
    oh_[0] = inputLayers_[0]->getOutput().getFrameHeight();
  }

  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);

  if (config_.has_add_size()) {
    addSize_ = config_.add_size();
  }
  if (config_.has_use_mkldnn_wgt()) {
    useMkldnnWgt_ = config_.use_mkldnn_wgt();
  }
  if (config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }
  VLOG(1) << "--- " << (useGlobalStats_ ? "use" : "do not use")
    << " --- global stats";

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

void MkldnnBatchNormLayer::clearDataDiff() {
//  resetOutput(bs_, getSize());
}

void MkldnnBatchNormLayer::reshape() {
  // reshape input and output size
  CHECK_NE(inputLayers_.size(), 0UL);
  const ImageConfig& conf = config_.inputs(0).image_conf();
  int height = inputLayers_[0]->getOutput().getFrameHeight();
  int width = inputLayers_[0]->getOutput().getFrameWidth();
  if (height == 0 && width == 0) {
    ih_[0] = conf.img_size();
    iw_[0] = conf.img_size();
  } else {
    ih_[0] = height;
    iw_[0] = width;
  }
  oh_[0] = ih_[0];
  ow_[0] = iw_[0];

  // reset output image size
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);
}

void MkldnnBatchNormLayer::resetDnnFwd() {
  CHECK(bs_ == getInput(0).getBatchSize()) << "batchsize should equal";
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
  botDatas_[0].reset(new MkldnnBuffer());
  topData_.reset(new MkldnnBuffer());
  if (useScaleShift_) {
    dataScaleShift_.reset(new MkldnnBuffer());
  }
  if (useGlobalStats_ || passType_ != PASS_TEST) {
    mean_.reset(new MkldnnBuffer());
    var_.reset(new MkldnnBuffer());
  }
  // init dim structure that describes user data.
  botDims_[0] = {bs_, ic_[0], ih_[0], iw_[0]};
  topDims_ = {bs_, oc_, oh_[0], ow_[0]};  // == botDims_[0]
  wgtDims_[0] = {2, oc_};  // scale and shift
  wgtFmt_[0] = memory::format::nc;
  biasDims_[0] = {oc_};
  biasFmt_[0] = memory::format::x;

  real *botDataData = getPrev(0)->getOutputValue()->getData();
  real *topDataData = getOutputValue()->getData();
  // 2. init user
  topData_->initUser(topDataData, topDims_, topFmt_, eg);
  const std::shared_ptr<memory::desc>& prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    botDatas_[0]->resetUser(botDataData, *prvMD, eg);
    bool isNC = botDatas_[0]->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(ih_[0] == iw_[0] && ih_[0] == 1)
        << "iw, ih must be 1 with nc input";
      // do not support nc input, so change to nchw
      botDatas_[0]->resetUser(botDataData, botDims_[0], botFmt_[0], eg);
      VLOG(4) << "use nchw data fmt";
    } else {
      VLOG(4) << "use prev data fmt: " << DNN_FMTS[botDatas_[0]->getUserFmt()];
    }
  } else {
    botDatas_[0]->initUser(botDataData, botDims_[0], botFmt_[0], eg);
  }
  // 3. create fwd desc
  std::shared_ptr<batch_normalization_forward::desc> fwdDesc;
  fwdDesc.reset(new batch_normalization_forward::desc(fwdpk,
    // TODO(TJ): use any if MKLDNN ready
    botDatas_[0]->getUserMD(),  // MkldnnBuffer::getMD(botDims_[0]),
    EPS, flags_));
  fwdPD_.reset(new batch_normalization_forward::primitive_desc(*fwdDesc, eg));
  // 4. init  conversion
  // src_primitive_desc == dst_primitive_desc in batch_normalization_forward
  botDatas_[0]->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtUser2Intl);
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
    dataScaleShift_->initUser(ssData, wgtDims_[0], wgtFmt_[0], eg);
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
    tmppd->initUser(localMean_->getData(), biasDims_[0], biasFmt_[0], eg);
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
        ? new batch_normalization_forward(*fwdPD_, *botDatas_[0]->getIntlMem(),
            (const primitive::at)(*mean_->getIntlMem()),
            (const primitive::at)(*var_->getIntlMem()),
            *dataScaleShift_->getIntlMem(),
            *topData_->getIntlMem())
        : new batch_normalization_forward(*fwdPD_, *botDatas_[0]->getIntlMem(),
            (const primitive::at)(*mean_->getIntlMem()),
            (const primitive::at)(*var_->getIntlMem()),
            *topData_->getIntlMem()));
    } else {
      fwd_.reset(useScaleShift_
        ? new batch_normalization_forward(*fwdPD_, *botDatas_[0]->getIntlMem(),
            *dataScaleShift_->getIntlMem(), *topData_->getIntlMem())
        : new batch_normalization_forward(*fwdPD_, *botDatas_[0]->getIntlMem(),
            *topData_->getIntlMem()));
    }
  } else {
    CHECK(useGlobalStats_ == false)
      << "useGlobalStats should be false in training";
    fwd_.reset(useScaleShift_
      ? new batch_normalization_forward(*fwdPD_, *botDatas_[0]->getIntlMem(),
          *dataScaleShift_->getIntlMem(), *topData_->getIntlMem(),
          *mean_->getIntlMem(), *var_->getIntlMem())
      : new batch_normalization_forward(*fwdPD_, *botDatas_[0]->getIntlMem(),
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
  botDiffs_[0].reset(new MkldnnBuffer());
  topDiff_.reset(new MkldnnBuffer());
  if (useScaleShift_) {
    diffScaleShift_.reset(new MkldnnBuffer());
  }
  // 2. init user, prepare top diff if use dnn input
  real* botDiffData = getDnnInputGrad(0)->getData();
  real *topDiffData = getOutputGrad()->getData();
  botDiffs_[0]->initUser(botDiffData, botDims_[0], botFmt_[0], eg);
  topDiff_->initUser(topDiffData, topDims_, topFmt_, eg);
  const std::shared_ptr<mkldnn::memory::desc>& inputDiffMD = getTopDiffMD();
  if (inputDiffMD) {
    topDiff_->resetUser(topDiffData, *inputDiffMD, eg);
    bool isNC = topDiff_->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(oh_[0] == ow_[0] && oh_[0] == 1)
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
    botDatas_[0]->getIntlMD(), botDatas_[0]->getIntlMD(), EPS, flags_));
  bwdPD.reset(new batch_normalization_backward::primitive_desc(
    *bwdDesc, eg, *fwdPD_));
  CHECK(bwdPD->diff_weights_primitive_desc() == dataScaleShift_->getIntlPD());
  CHECK(bwdPD->weights_primitive_desc() == fwdPD_->weights_primitive_desc());
  CHECK(bwdPD->mean_primitive_desc() == fwdPD_->mean_primitive_desc());
  CHECK(bwdPD->variance_primitive_desc() == fwdPD_->variance_primitive_desc());
  // 4. init conversion
  topDiff_->initCvt(botDatas_[0]->getIntlPD(), dnnCvtUser2Intl);
  if (prevIsDnn_[0]) {
    botDiffs_[0]->resetUser(botDiffData, botDatas_[0]->getIntlPD());
    prevLayer->setTopDiffMD(this->getName(), botDiffs_[0]->getUserMD());
    VLOG(4) << "set next diff format: " << DNN_FMTS[botDiffs_[0]->getUserFmt()];
  }
  botDiffs_[0]->initCvt(botDatas_[0]->getIntlPD(), dnnCvtIntl2User);
  // weight(scale) and bias(shift)
  if (useScaleShift_) {
    real *ssDiff = useMkldnnWgt_ ? weight_->getWGrad()->getData()
      : selfScaleShiftDiff_->getData();
    diffScaleShift_->initUser(ssDiff, wgtDims_[0], wgtFmt_[0], eg);
    CHECK(diffScaleShift_->getUserPD() == bwdPD->diff_weights_primitive_desc())
      << "scaleshiftPD should be {2,oc} with nc format, check mkldnn version.";
    diffScaleShift_->initCvt(
      bwdPD->diff_weights_primitive_desc(), dnnCvtIntl2User);
  }
  // 5. create bwd data and weight handle
  bwd_.reset(useScaleShift_
    ? new batch_normalization_backward(*bwdPD,
        *botDatas_[0]->getIntlMem(),
        *mean_->getIntlMem(), *var_->getIntlMem(),
        *topDiff_->getIntlMem(), *dataScaleShift_->getIntlMem(),
        *botDiffs_[0]->getIntlMem(), *diffScaleShift_->getIntlMem())
    : new batch_normalization_backward(*bwdPD,
        *botDatas_[0]->getIntlMem(),
        *mean_->getIntlMem(), *var_->getIntlMem(),
        *topDiff_->getIntlMem(), *botDiffs_[0]->getIntlMem()));
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
  botDatas_[0]->submitCvt(pipeline, botDataData);

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
  botDatas_[0]->submitCvt(pipeline, botDataData);
  if (useScaleShift_) {
    real *wgtDataData = useMkldnnWgt_ ? weight_->getW()->getData()
      : selfScaleShiftData_->getData();
    dataScaleShift_->submitCvt(pipeline, wgtDataData);
  }
  pipeline.push_back(*bwd_);
  botDiffs_[0]->submitCvt(pipeline, botDiffData);
  // output scaleshift diff
  if (useScaleShift_) {
    real *wgtDiffData = useMkldnnWgt_ ? weight_->getWGrad()->getData()
      : selfScaleShiftDiff_->getData();
    diffScaleShift_->submitCvt(pipeline, wgtDiffData);
  }
  stream(stream::kind::eager).submit(pipeline).wait();

  // update diff
  if (useScaleShift_) {
    if (!useMkldnnWgt_) {
      // copy scale and shift diff to paddle weight and bias
      memcpy(weight_->getWGrad_mutable()->getData(),
        selfScaleShiftDiff_->getData(), sizeof(real) * oc_);
      memcpy(biases_->getWGrad_mutable()->getData(),
        selfScaleShiftDiff_->getData() + oc_, sizeof(real) * oc_);
      weight_->getParameterPtr()->incUpdate(callback);
      biases_->getParameterPtr()->incUpdate(callback);
    } else {
      // only update weight
      weight_->getParameterPtr()->incUpdate(callback);
    }
  }
}

}  // namespace paddle
