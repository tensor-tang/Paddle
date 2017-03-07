/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


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
  ic_.push_back(conf.channels());
  oc_ = ic_[0];
  if (inputLayers_[0]->getOutput().getFrameHeight() == 0 &&
      inputLayers_[0]->getOutput().getFrameWidth() == 0) {
    iw_.push_back(conf.img_size());
    ih_.push_back(conf.img_size());
    ow_.push_back(conf.img_size());
    oh_.push_back(conf.img_size());
  } else {
    iw_.push_back(inputLayers_[0]->getOutput().getFrameWidth());
    ih_.push_back(inputLayers_[0]->getOutput().getFrameHeight());
    ow_.push_back(inputLayers_[0]->getOutput().getFrameWidth());
    oh_.push_back(inputLayers_[0]->getOutput().getFrameHeight());
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
  LOG(INFO) << "--- " << (useGlobalStats_ ? "use" : "do not use")
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

void MkldnnBatchNormLayer::resetDnnFwd(PassType passType) {
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
  useGlobalStats_ = (passType == PASS_TEST);
  if (passType == PASS_TEST && config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }
  if (passType != PASS_TEST && useGlobalStats_ == true) {
    LOG(WARNING) << "use_global_stats is invalid setting in training phase";
    useGlobalStats_ = false;
  }

  // start dnn
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  prop_kind fwdpk = (passType == PASS_TEST) ? prop_kind::forward_scoring
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
    if (useScaleShift_ && useMkldnnWgt_ && passType != PASS_TEST) {
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
  dataBot_.reset(new MkldnnBuffer());
  dataTop_.reset(new MkldnnBuffer());
  if (useScaleShift_) {
    dataScaleShift_.reset(new MkldnnBuffer());
  }
  if (useGlobalStats_ || passType != PASS_TEST) {
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

  real *botData = getPrev(0)->getOutputValue()->getData();
  real *topData = getOutputValue()->getData();
  // 2. init user
  dataTop_->initUser(topData, topDims_, topFmt_, eg);
  const std::shared_ptr<memory::desc> prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    dataBot_->resetUser(botData, *prvMD, eg);
    bool isNC = dataBot_->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(ih_[0] == iw_[0] && ih_[0] == 1)
        << "iw, ih must be 1 with nc input";
      // do not support nc input, so change to nchw
      dataBot_->resetUser(botData, botDims_[0], botFmt_[0], eg);
      VLOG(4) << "use nchw data fmt";
    } else {
      VLOG(4) << "use prev data fmt: " << DNN_FMTS[dataBot_->getUserFmt()];
    }
  } else {
    dataBot_->initUser(botData, botDims_[0], botFmt_[0], eg);
  }
  // 3. create fwd desc
  std::shared_ptr<batch_normalization_forward::desc> fwdDesc;
  fwdDesc.reset(new batch_normalization_forward::desc(fwdpk,
    // pool policy in MKLDNN for BN, any MD would not work yet
    dataBot_->getUserMD(),  // getAnyMD(botDims_[0]),
    EPS, flags_));
  fwdPD_.reset(new batch_normalization_forward::primitive_desc(*fwdDesc, eg));
  // 4. init  conversion
  // src_primitive_desc == dst_primitive_desc in batch_normalization_forward
  dataBot_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtUser2Intl);
  if (setDnnTopDataFmt_) {
    dataTop_->resetUser(topData, fwdPD_->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
    VLOG(4) << "set next data fmt: " << DNN_FMTS[dataTop_->getUserFmt()];
  }
  dataTop_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtIntl2User);
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
  if (useGlobalStats_ || passType != PASS_TEST) {
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
  if (passType == PASS_TEST) {
    if (useGlobalStats_) {
      fwd_.reset(useScaleShift_
        ? new batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
            (const primitive::at)(*mean_->getIntlMem()),
            (const primitive::at)(*var_->getIntlMem()),
            *dataScaleShift_->getIntlMem(),
            *dataTop_->getIntlMem())
        : new batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
            (const primitive::at)(*mean_->getIntlMem()),
            (const primitive::at)(*var_->getIntlMem()),
            *dataTop_->getIntlMem()));
    } else {
      fwd_.reset(useScaleShift_
        ? new batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
            *dataScaleShift_->getIntlMem(), *dataTop_->getIntlMem())
        : new batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
            *dataTop_->getIntlMem()));
    }
  } else {
    CHECK(useGlobalStats_ == false)
      << "useGlobalStats should be false in training";
    fwd_.reset(useScaleShift_
      ? new batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
          *dataScaleShift_->getIntlMem(), *dataTop_->getIntlMem(),
          *mean_->getIntlMem(), *var_->getIntlMem())
      : new batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
          *dataTop_->getIntlMem(),
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
  diffBot_.reset(new MkldnnBuffer());
  diffTop_.reset(new MkldnnBuffer());
  if (useScaleShift_) {
    diffScaleShift_.reset(new MkldnnBuffer());
  }
  // 2. init user, prepare top diff if use dnn input
  real* botDiff = prevLayer->getOutputGrad()->getData();
  real *topDiff = getOutputGrad()->getData();
  diffBot_->initUser(botDiff, botDims_[0], botFmt_[0], eg);
  diffTop_->initUser(topDiff, topDims_, topFmt_, eg);
  const std::shared_ptr<mkldnn::memory::desc> inputDiffMD = getTopDiffMD();
  if (inputDiffMD) {
    diffTop_->resetUser(topDiff, *inputDiffMD, eg);
    bool isNC = diffTop_->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(oh_[0] == ow_[0] && oh_[0] == 1)
        << "ow, oh must be 1 with nc input";
      // do not support nc input, so change to nchw
      diffTop_->resetUser(topDiff, topDims_, topFmt_, eg);
      VLOG(4) << "use nchw diff fmt";
    } else {
      VLOG(4) << "use prev diff fmt: " << DNN_FMTS[diffTop_->getUserFmt()];
    }
  }
  // 3. create backward desc ---------------------------------------------------
  std::shared_ptr<batch_normalization_backward::desc> bwdDesc;
  std::shared_ptr<batch_normalization_backward::primitive_desc> bwdPD;
  bwdDesc.reset(new batch_normalization_backward::desc(prop_kind::backward,
    dataBot_->getIntlMD(), dataBot_->getIntlMD(), EPS, flags_));
  bwdPD.reset(new batch_normalization_backward::primitive_desc(
    *bwdDesc, eg, *fwdPD_));
  CHECK(bwdPD->diff_weights_primitive_desc() == dataScaleShift_->getIntlPD());
  CHECK(bwdPD->weights_primitive_desc() == fwdPD_->weights_primitive_desc());
  CHECK(bwdPD->mean_primitive_desc() == fwdPD_->mean_primitive_desc());
  CHECK(bwdPD->variance_primitive_desc() == fwdPD_->variance_primitive_desc());
  // 4. init conversion
  diffTop_->initCvt(dataBot_->getIntlPD(), dnnCvtUser2Intl);
  if (setDnnBotDiffFmt_[0]) {
    diffBot_->resetUser(botDiff, dataBot_->getIntlPD());
    prevLayer->setTopDiffMD(diffBot_->getUserMD());
    VLOG(4) << "set next diff format: " << DNN_FMTS[diffBot_->getUserFmt()];
  }
  diffBot_->initCvt(dataBot_->getIntlPD(), dnnCvtIntl2User);
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
        *dataBot_->getIntlMem(),
        *mean_->getIntlMem(), *var_->getIntlMem(),
        *diffTop_->getIntlMem(), *dataScaleShift_->getIntlMem(),
        *diffBot_->getIntlMem(), *diffScaleShift_->getIntlMem())
    : new batch_normalization_backward(*bwdPD,
        *dataBot_->getIntlMem(),
        *mean_->getIntlMem(), *var_->getIntlMem(),
        *diffTop_->getIntlMem(), *diffBot_->getIntlMem()));
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

void MkldnnBatchNormLayer::submitDnnFwd(PassType passType) {
  real *botData = getPrev(0)->getOutputValue()->getData();
  real *topData = getOutputValue()->getData();

  std::vector<primitive> pipeline;
  // data bottom
  dataBot_->submitCvt(pipeline, botData);

  // prepare weight data of scale and shift
  if (useScaleShift_) {
    real *wgtData = useMkldnnWgt_ ? weight_->getW()->getData()
      : selfScaleShiftData_->getData();
    if (!useMkldnnWgt_) {
      // copy data from paddle weight and bias
      memcpy(selfScaleShiftData_->getData(), weight_->getW()->getData(),
        sizeof(real) * oc_);
      memcpy(selfScaleShiftData_->getData() + oc_, biases_->getW()->getData(),
        sizeof(real) * oc_);
    }
    dataScaleShift_->submitCvt(pipeline, wgtData);
  }
  pipeline.push_back(*fwd_);
  dataTop_->submitCvt(pipeline, topData);
  stream(stream::kind::eager).submit(pipeline).wait();

  // calculating and saving moving mean and variance
  if (passType != PASS_TEST) {
    calMovingMeanAndVar();
  }

  // activation
  forwardActivation();
}

void MkldnnBatchNormLayer::submitDnnBwd(const UpdateCallback &callback) {
  backwardActivation();

  real* botData = getPrev(0)->getOutputValue()->getData();
  real* topDiff = getOutputGrad()->getData();
  real* botDiff = getPrev(0)->getOutputGrad()->getData();

  std::vector<primitive> pipeline;
  diffTop_->submitCvt(pipeline, topDiff);
  dataBot_->submitCvt(pipeline, botData);
  if (useScaleShift_) {
    real *wgtData = useMkldnnWgt_ ? weight_->getW()->getData()
      : selfScaleShiftData_->getData();
    dataScaleShift_->submitCvt(pipeline, wgtData);
  }
  pipeline.push_back(*bwd_);
  diffBot_->submitCvt(pipeline, botDiff);
  // output scaleshift diff
  if (useScaleShift_) {
    real *wgtDiff = useMkldnnWgt_ ? weight_->getWGrad()->getData()
      : selfScaleShiftDiff_->getData();
    diffScaleShift_->submitCvt(pipeline, wgtDiff);
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
