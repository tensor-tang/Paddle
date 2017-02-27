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
#include <string.h>

// ex fc
#include "paddle/math/SparseMatrix.h"
#include <vector>
#include <algorithm>

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_batch_norm, MkldnnBatchNormLayer);

const real MkldnnBatchNormLayer::EPS = 1E-5;

bool MkldnnBatchNormLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* initialize the weightList */
  // first is Input in configure
  // other two is created in config_parser.py
  // TODO(TJ): why other two is created in config_parser.py???
  // actually only use 1 input
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
  imgPixels_ = ih_[0] * iw_[0];  // TODO(TJ):  remove when all dnn done
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);

  // should add this flag to layer proto and get from it
  usePaddleFmt_ = true;
  if (!usePaddleFmt_)
    LOG(FATAL) << "have not considerated do not use paddle fmt here yet";
  if (config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }
  movingAvgFraction_ = config_.moving_average_fraction();
  weight_.reset(new Weight(1, oc_, parameters_[0]));
  movingMean_.reset(new Weight(1, oc_, parameters_[1]));
  movingVar_.reset(new Weight(1, oc_, parameters_[2]));

  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
  }

  if (usePaddleFmt_) {
    selfScaleShiftData_ = Matrix::create(2, oc_, false, false);
    selfScaleShiftDiff_ = Matrix::create(2, oc_, false, false);
    selfScaleShiftData_->zeroMem();
    selfScaleShiftDiff_->zeroMem();
  }
  localMean_ = Matrix::create(1, oc_, false, false);
  localVar_ = Matrix::create(1, oc_, false, false);
  localMean_->zeroMem();
  localVar_->zeroMem();

  savedInvVar_ = Matrix::create(1, oc_, false, useGpu_);
  savedInvVar_->zeroMem();

  return true;
}

void MkldnnBatchNormLayer::calMeanAndStd(const MatrixPtr& mat) {
  int numSamples = mat->getHeight();
  Matrix::resizeOrCreate(tmpMat_, numSamples, oc_, false, useGpu_);
  localMean_->zeroMem();
  localMean_->accumulateColSum(*mat);
  localMean_->mulScalar(1.0 / numSamples);  // E[x]

  tmpMat_->assign(*mat);
  tmpMat_->square();
  savedInvVar_->zeroMem();
  savedInvVar_->accumulateColSum(*tmpMat_);
  savedInvVar_->mulScalar(1.0 / numSamples);  // E[x^2]
  savedInvVar_->addSquare(*localMean_, -1.0);      // E[x^2] - E^2[x]

  // Variance may be small negative value
  // because of the subtraction operation.
  // Here using clipping.
  savedInvVar_->downClip(real(0.0));

// LOG(INFO) << "ex mean var" << localMean_->getData()[1]
// << "," << savedInvVar_->getData()[1];

  calMovingMeanAndVar();

  savedInvVar_->subScalar(-EPS);
  savedInvVar_->sqrt(*savedInvVar_);
}

void MkldnnBatchNormLayer::calMovingMeanAndVar() {
  // calculating and saving moving mean and variance
  MatrixPtr movingMean = movingMean_->getW();
  MatrixPtr movingVar = movingVar_->getW();

  if (!useGpu_ && FLAGS_trainer_count > 1) {
    auto mvMean = std::dynamic_pointer_cast<SharedCpuMatrix>(movingMean);
    auto mvVar = std::dynamic_pointer_cast<SharedCpuMatrix>(movingVar);
    CHECK(mvMean && mvVar);

    mvMean->add(*localMean_, movingAvgFraction_, 1.0 - movingAvgFraction_);
    mvVar->add(*savedInvVar_, movingAvgFraction_, 1.0 - movingAvgFraction_);
  } else {
    // movingMean =  movingMean * movingAvgFraction_
    //            + savedMean_ * (1 - movingAvgFraction_)
    movingMean->add(*localMean_, movingAvgFraction_, 1.0 - movingAvgFraction_);
    // movingVar =  movingVar * movingAvgFraction_
    //           + savedInvVar_ * (1 - movingAvgFraction_)
    movingVar->add(*savedInvVar_, movingAvgFraction_, 1.0 - movingAvgFraction_);
  }
}

void MkldnnBatchNormLayer::setMeanAndStd() {
  localMean_->copyFrom(*(movingMean_->getW()));
  savedInvVar_->copyFrom(*(movingVar_->getW()));

  savedInvVar_->downClip(real(0.0));

  savedInvVar_->subScalar(-EPS);
  savedInvVar_->sqrt(*savedInvVar_);
}

void MkldnnBatchNormLayer::expandMat(const MatrixPtr& in, MatrixPtr& out) {
  CHECK_EQ(in->getWidth(), static_cast<size_t>(oc_ * imgPixels_));
  CHECK_EQ(out->getWidth(), static_cast<size_t>(oc_));
  CHECK(!in->isTransposed());
  CHECK(!out->isTransposed());
  if (imgPixels_ == 1) {
    out->assign(*in);
    return;
  }
  size_t batchSize = in->getHeight();
  CHECK_EQ(out->getHeight(), batchSize * imgPixels_);
  if (useGpu_) {
#ifdef PADDLE_ONLY_CPU
    LOG(FATAL) << "paddle is compiled only for cpu";
#else
    batchTranspose(in->getData(), out->getData(), imgPixels_,
                   channels_, batchSize);
#endif
  } else {
    for (size_t i = 0; i < batchSize; i++) {
      const MatrixPtr inTmp =
          Matrix::create(in->getData() + i * imgPixels_ * oc_, oc_,
                         imgPixels_, false, useGpu_);
      MatrixPtr outTmp =
          Matrix::create(out->getData() + i * imgPixels_ * oc_,
                         imgPixels_, oc_, false, useGpu_);
      inTmp->transpose(outTmp, false);
    }
  }
}

void MkldnnBatchNormLayer::shrinkMat(const MatrixPtr& in, MatrixPtr& out) {
  CHECK_EQ(in->getWidth(), static_cast<size_t>(oc_));
  CHECK_EQ(out->getWidth(), static_cast<size_t>(oc_ * imgPixels_));
  size_t batchSize = out->getHeight();
  CHECK(!in->isTransposed());
  CHECK(!out->isTransposed());
  if (imgPixels_ == 1) {
    out->assign(*in);
    return;
  }
  CHECK_EQ(in->getHeight(), static_cast<size_t>(batchSize * imgPixels_));
  if (useGpu_) {
#ifdef PADDLE_ONLY_CPU
    LOG(FATAL) << "paddle is compiled only for cpu";
#else
    batchTranspose(in->getData(), out->getData(), oc_,
                   imgPixels_, batchSize);
#endif
  } else {
    for (size_t i = 0; i < batchSize; i++) {
      const MatrixPtr inTmp =
          Matrix::create(in->getData() + i * oc_ * imgPixels_, imgPixels_,
                         oc_, false, useGpu_);
      MatrixPtr outTmp =
          Matrix::create(out->getData() + i * imgPixels_ * oc_, oc_,
                         imgPixels_, useGpu_);
      inTmp->transpose(outTmp, false);
    }
  }
}

void MkldnnBatchNormLayer::clearDataDiff() {
  resetOutput(bs_, getSize());
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
  printInfo();
}

void MkldnnBatchNormLayer::resetDnnFwd(PassType passType) {
  CHECK(bs_ == getInput(0).getBatchSize()) << "batchsize should equal";
  bool hasShift = (biases_ && biases_->getW());
  bool hasScale = (weight_ && weight_->getW());
  CHECK_EQ(hasScale, hasShift)
    << "only support both weigt and bias at same time, or neither";
  useScaleShift_ = (hasShift && hasScale);

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
  unsigned flags = 0u;
  if (useGlobalStats_)
    flags = (flags | batch_normalization_flag::use_global_stats);
  if (useScaleShift_)
    flags = (flags | batch_normalization_flag::use_scale_shift);

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
  if (useGlobalStats_) {
    localMean_->copyFrom(*(movingMean_->getW()));
    localVar_->copyFrom(*(movingVar_->getW()));
  }
  // create dim structure that describes user data.
  memory::dims botDims, ssDims, topDims;
  memory::format fmtnchw, fmtx, fmtss;
  botDims = {bs_, ic_[0], ih_[0], iw_[0]};
  topDims = {bs_, oc_, oh_[0], ow_[0]};  // == botDims
  ssDims = {2, oc_};  // scale and shift
  fmtnchw = memory::format::nchw;
  fmtx = memory::format::x;
  fmtss = memory::format::nc;
  real *botData = getPrev(0)->getOutputValue()->getData();
  real *topData = getOutputValue()->getData();
  // 2. init user
  dataTop_->initUser(topData, topDims, fmtnchw, eg);
  const std::shared_ptr<memory::desc> prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    dataBot_->resetUser(botData, *prvMD, eg);
    bool isNC = dataBot_->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(ih_[0] == iw_[0] && ih_[0] == 1)
        << "iw, ih must be 1 with nc input";
      // do not support nc input, so change to nchw
      dataBot_->resetUser(botData, botDims, fmtnchw, eg);
      LOG(INFO) << "use nchw format";
    } else {
      LOG(INFO) << "use prev format: " << DNN_FMTS[dataBot_->getUserFmt()];
    }
  } else {
    dataBot_->initUser(botData, botDims, fmtnchw, eg);
  }
  // 3. create fwd desc
  std::shared_ptr<batch_normalization_forward::desc> fwdDesc;
  std::shared_ptr<batch_normalization_forward::primitive_desc> fwdPD;
  fwdDesc.reset(new batch_normalization_forward::desc(fwdpk, 
    dataBot_->getUserMD(), EPS, flags));
  fwdPD.reset(new batch_normalization_forward::primitive_desc(*fwdDesc, eg));
  // 4. init  conversion
  // src_primitive_desc == dst_primitive_desc in batch_normalization_forward
  dataBot_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtUser2Intl);
  if (setDnnTopDataFmt_) {
    dataTop_->resetUser(topData, fwdPD->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
    LOG(INFO) << "set next format: " << DNN_FMTS[dataTop_->getUserFmt()];
  }
  dataTop_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);
  // weight(scale) and bias(shift)
  if (useScaleShift_) {
    real *ssData = usePaddleFmt_ ? selfScaleShiftData_->getData()
      : weight_->getW()->getData();
    dataScaleShift_->initUser(ssData, ssDims, fmtss, eg);
    CHECK(dataScaleShift_->getUserPD() == fwdPD->weights_primitive_desc())
      << "scaleshiftPD should be {2,oc} with nc format, check mkldnn version.";
    dataScaleShift_->initCvt(
      fwdPD->weights_primitive_desc(), dnnCvtUser2Intl);
  } else {
    LOG(WARNING) << "Are you sure do not need scale and shift???";
  }
  // mean and var
  if (useGlobalStats_ || passType != PASS_TEST) {
    MkldnnBufferPtr tmppd(new MkldnnBuffer());
    tmppd->initUser(localMean_->getData(), {oc_}, fmtx, eg);
    CHECK(tmppd->getUserPD() == fwdPD->mean_primitive_desc()
      && fwdPD->mean_primitive_desc() == fwdPD->variance_primitive_desc())
      << "assert both mean and var size are {oc_}";
    // use exactly the same size and format of mean and var in mkldnn
    mean_->initUser(localMean_->getData(), fwdPD->mean_primitive_desc());
    mean_->initCvt(fwdPD->mean_primitive_desc(), dnnCvtNoNeed);
    var_->initUser(localVar_->getData(), fwdPD->variance_primitive_desc());
    var_->initCvt(fwdPD->variance_primitive_desc(), dnnCvtNoNeed);
  }
  // 5. create fwd handle
  if (passType == PASS_TEST) {
    if (useGlobalStats_) {
      fwd_.reset(useScaleShift_
        ? new batch_normalization_forward(*fwdPD, *dataBot_->getIntlMem(),
            (const primitive::at)(*mean_->getIntlMem()),
            (const primitive::at)(*var_->getIntlMem()),
            *dataScaleShift_->getIntlMem(),
            *dataTop_->getIntlMem())
        : new batch_normalization_forward(*fwdPD, *dataBot_->getIntlMem(),
            (const primitive::at)(*mean_->getIntlMem()),
            (const primitive::at)(*var_->getIntlMem()),
            *dataTop_->getIntlMem()));
    } else {
      fwd_.reset(useScaleShift_
        ? new batch_normalization_forward(*fwdPD, *dataBot_->getIntlMem(),
            *dataScaleShift_->getIntlMem(), *dataTop_->getIntlMem())
        : new batch_normalization_forward(*fwdPD, *dataBot_->getIntlMem(),
            *dataTop_->getIntlMem()));
    }
  } else {
    CHECK(useGlobalStats_ == false)
      << "useGlobalStats should be false in training";
    fwd_.reset(useScaleShift_
      ? new batch_normalization_forward(*fwdPD, *dataBot_->getIntlMem(),
          *dataScaleShift_->getIntlMem(), *dataTop_->getIntlMem(),
          *mean_->getIntlMem(), *var_->getIntlMem())
      : new batch_normalization_forward(*fwdPD, *dataBot_->getIntlMem(),
          *dataTop_->getIntlMem(),
          *mean_->getIntlMem(), *var_->getIntlMem()));
  }
  LOG(INFO) << "data format flow --- "
    << DNN_FMTS[dataBot_->getUserFmt()] << " >>> ("
    << DNN_FMTS[dataBot_->getIntlFmt()] << " >>> "
    << DNN_FMTS[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FMTS[dataTop_->getUserFmt()];

  /// init mkldnn backward *****************************************************
  if (passType == PASS_TEST)
    return;
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
  diffBot_->initUser(botDiff, botDims, fmtnchw, eg);
  diffTop_->initUser(topDiff, topDims, fmtnchw, eg);
  const std::shared_ptr<mkldnn::memory::desc> inputDiffMD = getTopDiffMD();
  if (inputDiffMD) {
    diffTop_->resetUser(topDiff, *inputDiffMD, eg);
    bool isNC = diffTop_->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(oh_[0] == ow_[0] && oh_[0] == 1)
        << "ow, oh must be 1 with nc input";
      // do not support nc input, so change to nchw
      diffTop_->resetUser(topDiff, topDims, fmtnchw, eg);
      LOG(INFO) << "use nchw format";
    } else {
      LOG(INFO) << "use prev format: " << DNN_FMTS[dataBot_->getUserFmt()];
    }
  }
  // 3. create backward desc ----------------------------------------
  std::shared_ptr<batch_normalization_backward::desc> bwdDesc;
  std::shared_ptr<batch_normalization_backward::primitive_desc> bwdPD;
  bwdDesc.reset(new batch_normalization_backward::desc(prop_kind::backward,
    dataBot_->getIntlMD(), dataBot_->getIntlMD(), EPS, flags));
  bwdPD.reset(new batch_normalization_backward::primitive_desc(
    *bwdDesc, eg, *fwdPD));
  CHECK(bwdPD->diff_weights_primitive_desc() == dataScaleShift_->getIntlPD());
  CHECK(bwdPD->weights_primitive_desc() == fwdPD->weights_primitive_desc());
  CHECK(bwdPD->mean_primitive_desc() == fwdPD->mean_primitive_desc());
  CHECK(bwdPD->variance_primitive_desc() == fwdPD->variance_primitive_desc());
  // 4. init conversion
  diffTop_->initCvt(dataBot_->getIntlPD(), dnnCvtUser2Intl);
  if (setDnnBotDiffFmt_[0]) {
    diffBot_->resetUser(botDiff, dataBot_->getIntlPD());
    prevLayer->setTopDiffMD(diffBot_->getUserMD());
    LOG(INFO) << "set prev diff format: " << DNN_FMTS[diffBot_->getUserFmt()];
  }
  diffBot_->initCvt(dataBot_->getIntlPD(), dnnCvtIntl2User);
  // weight(scale) and bias(shift)
  if (useScaleShift_) {
    real *ssDiff = usePaddleFmt_ ? selfScaleShiftDiff_->getData()
      : weight_->getWGrad()->getData();
    diffScaleShift_->initUser(ssDiff, ssDims, fmtss, eg);
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
  LOG(INFO) << "diff format flow --- "
  << DNN_FMTS[diffBot_->getUserFmt()] << " <<< ("
  << DNN_FMTS[diffBot_->getIntlFmt()] << " <<< "
  << DNN_FMTS[diffTop_->getIntlFmt()] << ") <<< "
  << DNN_FMTS[diffTop_->getUserFmt()];
}

void MkldnnBatchNormLayer::resetDnnBwd() {

}

void MkldnnBatchNormLayer::myFwd(PassType passType) {
  real *botdata = getPrev(0)->getOutputValue()->getData();
  real *topdata = getOutputValue()->getData();

  std::vector<primitive> pipeline;
  // data bottom
  dataBot_->submitCvt(pipeline, botdata);

  // prepare weight data of scale and shift 
  if (useScaleShift_) {
    real *wgtdata;
    if (usePaddleFmt_) {
      // copy data from paddle weight and bias
      memcpy(selfScaleShiftData_->getData(), weight_->getW()->getData(),
        sizeof(real) * oc_);
      memcpy(selfScaleShiftData_->getData() + oc_, biases_->getW()->getData(),
        sizeof(real) * oc_);
      wgtdata = selfScaleShiftData_->getData();
    } else {
      wgtdata = weight_->getW()->getData();
    }
    dataScaleShift_->submitCvt(pipeline, wgtdata);
  }
  
  pipeline.push_back(*fwd_);

  dataTop_->submitCvt(pipeline, topdata);
// LOG(INFO) << botdata[1] << "," << localMean_->getData()[1] << "," << localVar_->getData()[1] << "," << selfScaleShiftData_->getData()[1] << topdata[1];

  stream(stream::kind::eager).submit(pipeline).wait();

  // calculating and saving moving mean and variance
  if (passType != PASS_TEST) {
    // LOG(INFO) << "cal moving .............. should not in testing";
    CHECK(useGlobalStats_ == false);
    MatrixPtr movingMean = movingMean_->getW();
    MatrixPtr movingVar = movingVar_->getW();
    if (!useGpu_ && FLAGS_trainer_count > 1) {
      auto mvMean = std::dynamic_pointer_cast<SharedCpuMatrix>(movingMean);
      auto mvVar = std::dynamic_pointer_cast<SharedCpuMatrix>(movingVar);
      CHECK(mvMean && mvVar);
      mvMean->add(*localMean_, movingAvgFraction_, 1.0 - movingAvgFraction_);
      mvVar->add(*localVar_, movingAvgFraction_, 1.0 - movingAvgFraction_);
    } else {
      movingMean->add(
        *localMean_, movingAvgFraction_, 1.0 - movingAvgFraction_);
      // here var is v^2
      movingVar->add(*localVar_, movingAvgFraction_, 1.0 - movingAvgFraction_);
    }
  }
/*  if (passType != PASS_TEST)
//    LOG(INFO) << "my mean var" << localMean_->getData()[1] << "," << localVar_->getData()[1];
//  LOG(INFO) << "my ------------" << topdata[0] << "," << topdata[1] << "," << topdata[2] << "," << topdata[oc_*oh_[0]*ow_[0]-1];
*/
}

void MkldnnBatchNormLayer::exFwd(PassType passType) {
  int batchSize = getInputValue(0)->getHeight();
  // for testing in training peroid
  useGlobalStats_ = (passType == PASS_TEST);
  if (passType == PASS_TEST && config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }

  Matrix::resizeOrCreate(expandedIn_, batchSize * imgPixels_, oc_, false,
                         useGpu_);
  Matrix::resizeOrCreate(normIn_, batchSize * imgPixels_, oc_, false,
                         useGpu_);
  Matrix::resizeOrCreate(expandedOut_, batchSize * imgPixels_, oc_, false,
                         useGpu_);
  expandMat(getInputValue(0), expandedIn_);
/*
  if (useGlobalStats_) {
    if (firstTest_) {
      setMeanAndStd();
      firstTest_ = false;
    }
  } else {
    calMeanAndStd(expandedIn_);
    firstTest_ = true;
  }
  */
  normIn_->assign(*expandedIn_);
  normIn_->addBias(*localMean_, -1);  // subtract mean.
  normIn_->divRowVector(*savedInvVar_);  // divide std.
  expandedOut_->assign(*normIn_);

/*  if (!useEx_){
    LOG(INFO) << localMean_->getData()[1] << "," << savedInvVar_->getData()[1] << "," << normIn_->getData()[1];
  }*/
  expandedOut_->mulRowVector(*weight_->getW());  // multiple gamma.
  if (biases_) {
    expandedOut_->addBias(*(biases_->getW()), 1);  // add beta.
  }
/*
  if (useEx_) {
    // acutal in use
    MatrixPtr out = getOutputValue();
    shrinkMat(expandedOut_, out);
  } else {
    // just for my test to compare with my result
    // can be remove when finish this layer
    MatrixPtr out = Matrix::create(bs_, oc_*oh_[0]*ow_[0], false, false);
    // MatrixPtr out = getOutputValue();
    shrinkMat(expandedOut_, out);
//  real *topdata = out->getData();
//  LOG(INFO) << "ex ------------" << topdata[0] << "," << topdata[1] << "," << topdata[2] << "," << topdata[oc_*oh_[0]*ow_[0]-1];
  }*/
}

void MkldnnBatchNormLayer::submitDnnFwd(PassType passType) {
  //  exFwd(passType);
  myFwd(passType);
   
/*
{ // check mean and var
  localVar_->sqrt(*localVar_);  // mkldnn var is v^2
  LOG(INFO) << "my mean var:" << localMean_->getData()[1] << "," << localVar_->getData()[1] << "," << selfScaleShiftData_->getData()[1];
  
  localMean_->zeroMem();
    Matrix::resizeOrCreate(expandedIn_, bs_ * imgPixels_, oc_, false,
                         useGpu_);
  expandMat(getInputValue(0), expandedIn_);
  calMeanAndStd(expandedIn_);
  
 LOG(INFO) << "ex mean var:" << localMean_->getData()[1] << "," << savedInvVar_->getData()[1] << "," << selfScaleShiftData_->getData()[1];
  }
*/
  // activation
  forwardActivation();
}

void MkldnnBatchNormLayer::exBwd(const UpdateCallback &callback) {

  int batchSize = getInputValue(0)->getHeight();

  Matrix::resizeOrCreate(meanGrad_, 1, oc_, false, useGpu_);
  Matrix::resizeOrCreate(stdGrad_, 1, oc_, false, useGpu_);

  Matrix::resizeOrCreate(expandedInGrad_, batchSize * imgPixels_, oc_,
                         false, useGpu_);
  Matrix::resizeOrCreate(inGrad_, batchSize, imgPixels_ * oc_, false,
                         useGpu_);
  Matrix::resizeOrCreate(normInGrad_, batchSize * imgPixels_, oc_, false,
                         useGpu_);
  Matrix::resizeOrCreate(expandedOutGrad_, batchSize * imgPixels_, oc_,
                         false, useGpu_);
  Matrix::resizeOrCreate(tmpMat_, batchSize * imgPixels_, oc_, false,
                         useGpu_);
  Matrix::resizeOrCreate(tmpGrad_, batchSize * imgPixels_, oc_, false,
                         useGpu_);
  if (true) {
    // when use mkldnn should  prepare some matrix for ex Bwd
    Matrix::resizeOrCreate(normIn_, bs_ * imgPixels_, oc_, false, useGpu_);
    Matrix::resizeOrCreate(expandedIn_, bs_ * imgPixels_, oc_, false, useGpu_);
    savedInvVar_->copyFrom(*localVar_);
    savedInvVar_->sqrt(*savedInvVar_);
    expandMat(getInputValue(0), expandedIn_);
    normIn_->assign(*expandedIn_);
    normIn_->addBias(*localMean_, -1);  // subtract mean.
    normIn_->divRowVector(*savedInvVar_);  // divide std.
// LOG(INFO) << localMean_->getData()[1] << ","
// << savedInvVar_->getData()[1] << "," << normIn_->getData()[1];
  }
  expandMat(getOutputGrad(), expandedOutGrad_);
  // compute derivatives.
  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*expandedOutGrad_, 1);
//    biases_->getParameterPtr()->incUpdate(callback);
  }
  if (weight_->getWGrad()) {
    tmpMat_->dotMul(*expandedOutGrad_, *normIn_);
    weight_->getWGrad()->collectBias(*tmpMat_, 1);
  }

  // compute input gradients.
  normInGrad_->assign(*expandedOutGrad_);
  normInGrad_->mulRowVector(*(weight_->getW()));  // multiple gamma.
  // normInGrad * (x - \mu)/ \sqrt(\delta^2)
  tmpMat_->dotMul(*normInGrad_, *normIn_);
  stdGrad_->zeroMem();
  stdGrad_->collectBias(*tmpMat_, -1.0 / (batchSize * imgPixels_));
  tmpGrad_->assign(*normIn_);
  tmpGrad_->mulRowVector(*stdGrad_);

  meanGrad_->zeroMem();
  meanGrad_->collectBias(*normInGrad_, -1.0 / (batchSize * imgPixels_));

  expandedInGrad_->zeroMem();
  expandedInGrad_->add(*normInGrad_, *tmpGrad_);
  expandedInGrad_->addRowVector(*meanGrad_);
  expandedInGrad_->divRowVector(*savedInvVar_);

  shrinkMat(expandedInGrad_, inGrad_);
  if (getInputGrad(0)) {
    getInputGrad(0)->add(*getInputGrad(0), *inGrad_);
  }
  {
//    weight_->getParameterPtr()->incUpdate(callback);
  }
}

void MkldnnBatchNormLayer::submitDnnBwd(const UpdateCallback &callback) {
  backwardActivation();

//  exBwd(nullptr);

  real* botdata = getPrev(0)->getOutputValue()->getData();
  real* topdiff = getOutputGrad()->getData();
  real* botdiff = getPrev(0)->getOutputGrad()->getData();
//  LOG(INFO) << "-------ex botdiff wgtdiff:" << botdiff[0] << "," << botdiff[1] << "," << weight_->getWGrad()->getData()[0] << "," << weight_->getWGrad()->getData()[1] << "," << biases_->getWGrad()->getData()[0] << "," << biases_->getWGrad()->getData()[1];
  std::vector<primitive> pipeline;
  // push inputs
  diffTop_->submitCvt(pipeline, topdiff);
  dataBot_->submitCvt(pipeline, botdata);
  if (useScaleShift_) {
    real *wgtdata = usePaddleFmt_
      ? selfScaleShiftData_->getData() : weight_->getW()->getData();
    dataScaleShift_->submitCvt(pipeline, wgtdata);
  }
  // then add bwd
  pipeline.push_back(*bwd_);
  // output bot diff
  diffBot_->submitCvt(pipeline, botdiff);
  // output scaleshift diff
  if (useScaleShift_) {
    real *wgtdiff = usePaddleFmt_
      ? selfScaleShiftDiff_->getData() : weight_->getWGrad()->getData();
    diffScaleShift_->submitCvt(pipeline, wgtdiff);
  }
//  LOG(INFO) << "size:" << pipeline.size() << "== 1 or 3";
  // execute pipeline
  stream(stream::kind::eager).submit(pipeline).wait();

  // update diff
  if (useScaleShift_) {
    if (usePaddleFmt_) {
      // copy scale and shift diff to paddle weight and bias
      memcpy(weight_->getWGrad_mutable()->getData(),
        selfScaleShiftDiff_->getData(), sizeof(real) * oc_);
      memcpy(biases_->getWGrad_mutable()->getData(),
        selfScaleShiftDiff_->getData() + oc_, sizeof(real) * oc_);
//  LOG(INFO) << "-------my botdiff wgtdiff:" << botdiff[0] << "," << botdiff[1] << "," <<weight_->getWGrad()->getData()[0] << "," << weight_->getWGrad()->getData()[1] << "," << biases_->getWGrad()->getData()[0] << "," << biases_->getWGrad()->getData()[1];
      weight_->getParameterPtr()->incUpdate(callback);
      biases_->getParameterPtr()->incUpdate(callback);
    } else {
      // only update weight
      weight_->getParameterPtr()->incUpdate(callback);
    }
  }
}

}  // namespace paddle
