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


namespace paddle {

REGISTER_LAYER(mkldnn_batch_norm, MkldnnBatchNormLayer);

const real MkldnnBatchNormLayer::EPS = 1E-5;

bool MkldnnBatchNormLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {

  /* initialize the weightList */
  // first is Input in configure
  // other two is created in config_parser.py
  // TODO: why other two is created in config_parser.py???
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
  imgPixels_ = ih_[0] * iw_[0];
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);

  // should add this flag to layer proto and get from it
  usePaddleFmt_ = true;

  if (ih_[0] == iw_[0] && ih_[0] == 1) {
    // mkldnn has some issue with this case, so use paddle code instead
    useEx_ = true;
    usePaddleFmt_ = true; // force to true
  }
  if (!usePaddleFmt_)
    LOG(FATAL) << "have not considerated do not use paddle fmt in this layer yet";
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

  localMean_ = Matrix::create(1, oc_, false, false);
  localVar_ = Matrix::create(1, oc_, false, false);
  localMean_->zeroMem();
  localVar_->zeroMem();
  
  savedInvVar_ = Matrix::create(1, oc_, false, useGpu_);
  savedInvVar_->zeroMem();

  firstTest_ = true;

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

  //LOG(INFO) << "ex mean var" << localMean_->getData()[1] << "," << savedInvVar_->getData()[1];

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

size_t MkldnnBatchNormLayer::getOneBatchSize() {
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

  return oc_ * oh_[0] * ow_[0];
}

// whether reset batchsize and image size of input and output 
bool MkldnnBatchNormLayer::reshapeOutput() {
  if (bs_ == getInput(0).getBatchSize()) {
    // can remove reserveOutput when confirm how multi inputs work
    // and whether to clear diff
    resetOutput(bs_, getOneBatchSize()); 
    return false;
  }
  // reset data
  bs_ = getInput(0).getBatchSize();

  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);

  LOG(INFO) << "reshape batch size: " << bs_;
  resetOutput(bs_, getOneBatchSize());
  return true;
}

void MkldnnBatchNormLayer::resetDnnFwd(PassType passType) {
  LOG(INFO) << "reset mkldnn forward of batch_norm layer: " << config_.name();
  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";
  printInfo();
  if (ih_[0] == iw_[0] && ih_[0] == 1) {
    // mkldnn has some issue with this case, so use paddle code instead
    useEx_ = true;
    usePaddleFmt_ = true; // force to true
    LOG(INFO) << "Skip MKLDNN prepare when iw==ih==1";
    return;
  }
  useEx_ = false;
  bool hasBias = (biases_ && biases_->getW()) ? true : false;
  bool hasWgts = (weight_ && weight_->getW()) ? true : false;
  useScaleShift_ = (hasBias && hasWgts);
  // only wgt and bias at same time,
  if (!useScaleShift_ && hasWgts) {
    LOG(FATAL) << "only support both weigt and bias at same time, or neither";
  }

  // in train always calculate it, do not use GlobalStats
  // in test depends on choice
  useGlobalStats_ = (passType == PASS_TEST);
  if (passType == PASS_TEST && config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }
  if (passType == PASS_TRAIN && useGlobalStats_ == true) {
    LOG(FATAL) << "this should not happen!!! use gloal stat in training";
  }
  prop_kind pk = (passType == PASS_TEST) ? prop_kind::forward_scoring :
    prop_kind::forward_training;
  unsigned flags = 0u;
  if (useGlobalStats_) 
    flags = (flags | batch_normalization_flag::use_global_stats);
  if (useScaleShift_)
    flags = (flags | batch_normalization_flag::use_scale_shift);

  //create dim structure that describes user data.
  memory::dims botDims, wgtDims, topDims;
  botDims = {bs_, ic_[0], ih_[0], iw_[0]};
  topDims = {bs_, oc_, oh_[0], ow_[0]};  // == botDims
  dataBot_.reset(new MkldnnBuffer(botDims));
  dataTop_.reset(new MkldnnBuffer(topDims));
  real *botData = getPrev(0)->getOutputValue()->getData();
  real *topData = getOutputValue()->getData();
  const std::shared_ptr<mkldnn::memory::desc> prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    LOG(FATAL) << "should not be here so far...........";
    dataBot_->initUser(botData, *prvMD, *engine_);
  } else {
    dataBot_->initUser(botData, botDims, memory::format::nchw, *engine_);
  }
  
  std::shared_ptr<batch_normalization_forward::desc> fwdDesc;
  fwdDesc.reset(new batch_normalization_forward::desc(pk, 
    dataBot_->getUserMD(), EPS, flags));
  fwdPD_.reset(new batch_normalization_forward::primitive_desc(*fwdDesc, *engine_));

  // init bottom cvt
  if (dataBot_->initCvt(dataBot_->getUserPD(), dnnCvtUser2Internal)) {
    LOG(INFO) << "need reorder --- bottom data: "
      << DNN_FORMAT[dataBot_->getUserFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataBot_->getIntlFmt()];
  }
  // init top user memory and cvt
  if (setDnnTopDataFmt_) {
    dataTop_->initUser(topData, fwdPD_->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
    LOG(FATAL) << "should not be here so far...........";
  } else {
    dataTop_->initUser(topData, topDims, memory::format::nchw, *engine_);
  }
  if (dataTop_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtInternal2User)) {
    LOG(INFO) << "need reorder --- top data: " 
      << DNN_FORMAT[dataTop_->getIntlFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataTop_->getUserFmt()];
  }

  // weight
  if (useScaleShift_) {
    if (usePaddleFmt_) {
      myScaleShift_ = Matrix::create(2, oc_, false, false);
    } else {
      myScaleShift_ = weight_->getW();
      LOG(FATAL) << "should not come here so far!!!";
    }
    real *wgtData = myScaleShift_->getData();
    wgtDims = {2, oc_};
    wgtScaleShift_.reset(new MkldnnBuffer(wgtDims));
    wgtScaleShift_->initUser(wgtData, wgtDims, memory::format::nc, *engine_);
    if (wgtScaleShift_->initCvt(fwdPD_->weights_primitive_desc(), dnnCvtUser2Internal)) {
      LOG(FATAL) << "should donot need cvt!!! user vs intl format:"
      << DNN_FORMAT[wgtScaleShift_->getUserFmt()] << " vs " 
      << DNN_FORMAT[wgtScaleShift_->getIntlFmt()];
    }
  } else {
    LOG(WARNING) << "sure do not need scale and shift???";
  }
  
  if (passType == PASS_TRAIN || useGlobalStats_) {
    mean_.reset(new MkldnnBuffer({oc_}));
    var_.reset(new MkldnnBuffer({oc_}));
    // TODO: if input is userPD, should accept dnnCvtNoNeed, because do not care
    mean_->initUser(localMean_->getData(), {oc_}, memory::format::x, *engine_);
    var_->initUser(localVar_->getData(), {oc_}, memory::format::x, *engine_);
    if (useGlobalStats_) {
      if (mean_->initCvt(fwdPD_->mean_primitive_desc(), dnnCvtUser2Internal)) {
        LOG(FATAL) << "should donot need cvt!!! format-- user vs intl:"
          << DNN_FORMAT[mean_->getUserFmt()] << " vs " 
          << DNN_FORMAT[mean_->getIntlFmt()];
      }
      if (var_->initCvt(fwdPD_->variance_primitive_desc(), dnnCvtUser2Internal)) {
        LOG(FATAL) << "should donot need cvt!!! format-- user vs intl:"
          << DNN_FORMAT[var_->getUserFmt()] << " vs " 
          << DNN_FORMAT[var_->getIntlFmt()];
      }
    } else {
      if (mean_->initCvt(fwdPD_->mean_primitive_desc(), dnnCvtInternal2User)) {
        LOG(FATAL) << "should donot need cvt!!! format-- user vs intl:"
          << DNN_FORMAT[mean_->getUserFmt()] << " vs " 
          << DNN_FORMAT[mean_->getIntlFmt()];
      }
      if (var_->initCvt(fwdPD_->variance_primitive_desc(), dnnCvtInternal2User)) {
        LOG(FATAL) << "should donot need cvt!!! format-- user vs intl:"
          << DNN_FORMAT[var_->getUserFmt()] << " vs " 
          << DNN_FORMAT[var_->getIntlFmt()];
      }
    }
  }
  
  LOG(INFO) << "data format flow --- "
    << DNN_FORMAT[dataBot_->getUserFmt()] << " >>> ("
    << DNN_FORMAT[dataBot_->getIntlFmt()] << " >>> "
    << DNN_FORMAT[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FORMAT[dataTop_->getUserFmt()];

}

void MkldnnBatchNormLayer::resetDnnBwd() {
  /*LOG(INFO) << "init or reset fc backward of layer: " << config_.name();
  */
}

void MkldnnBatchNormLayer::myFwd(PassType passType) {
  if (passType == PASS_TRAIN && useGlobalStats_ == true) {
    LOG(FATAL) << "this should not happen!!! use gloal stat in training";
  }
  /// all sumbit cvt should be clear
  clearAllCvtFlags();
  std::vector<primitive> fwd;
  
  // data bottom
  real *botdata = getPrev(0)->getOutputValue()->getData();
  dataBot_->submitCvt(fwd, botdata);
  
  // data wgt
  if (useScaleShift_ ) {
    if (usePaddleFmt_) {
      memcpy(myScaleShift_->getData(), weight_->getW()->getData(),
        sizeof(real) * oc_);
      memcpy(myScaleShift_->getData() + oc_, biases_->getW()->getData(),
        sizeof(real) * oc_);
      /*for (int i = 0; i < oc_; ++i) {
        myScaleShift_->getData()[i] = weight_->getW()->getData()[i];
        myScaleShift_->getData()[i+oc_] = biases_->getW()->getData()[i];
      }*/
    } else {
      myScaleShift_ = weight_->getW();
      LOG(FATAL) << "should not come here so far!!!";
    }
    real *wgtdata = myScaleShift_->getData();
    wgtScaleShift_->submitCvt(fwd, wgtdata);
  }
  if (useGlobalStats_) {
    if (firstTest_) {
      LOG(INFO) << "should be here only once.................in Testing";
      localMean_->copyFrom(*(movingMean_->getW()));
      localVar_->copyFrom(*(movingVar_->getW()));
      firstTest_ = false;
    }
  } else {
    firstTest_ = true;
  }
  if (passType == PASS_TEST) {
    if (useGlobalStats_) {
      // mean and var are inputs, submit them before BN fwd
      mean_->submitCvt(fwd, localMean_->getData());
      var_->submitCvt(fwd, localVar_->getData());
      fwd.push_back(useScaleShift_
          ? batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
              (const primitive::at)(*mean_->getIntlMem()),
              (const primitive::at)(*var_->getIntlMem()),
              *wgtScaleShift_->getIntlMem(),
              *dataTop_->getIntlMem())
          : batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
              *mean_->getIntlMem(), *var_->getIntlMem(),
              *dataTop_->getIntlMem()));
    } else {
      //LOG(INFO) << "testing and use local mean and var";
      fwd.push_back(useScaleShift_
        ? batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
            *wgtScaleShift_->getIntlMem(),
            *dataTop_->getIntlMem())
        : batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
            *dataTop_->getIntlMem()));
    }
  } else {
    CHECK(useGlobalStats_ == false) << "useGlobalStats shoud not happed in train";
    fwd.push_back(useScaleShift_
          ? batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
              *wgtScaleShift_->getIntlMem(),
              *dataTop_->getIntlMem(),
              *mean_->getIntlMem(), *var_->getIntlMem())
          : batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
              *dataTop_->getIntlMem(),
              *mean_->getIntlMem(), *var_->getIntlMem()));
    // mean and var are outputs, submit them after BN fwd
    mean_->submitCvt(fwd, localMean_->getData());
    var_->submitCvt(fwd, localVar_->getData());
  }
  /*
  if (passType == PASS_TEST && !useGlobalStats_) {
    LOG(INFO) << "testing and use local mean and var, maybe should not be here...";
    fwd.push_back(useScaleShift_
      ? batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
          *wgtScaleShift_->getIntlMem(),
          *dataTop_->getIntlMem())
      : batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
          *dataTop_->getIntlMem()));
  } else {
    if (useGlobalStats_) {
      // mean and var are inputs, submit them before BN fwd
      mean_->submitCvt(fwd, localMean_->getData());
      var_->submitCvt(fwd, localVar_->getData());
      fwd.push_back(useScaleShift_
          ? batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
              *mean_->getIntlMem(), *var_->getIntlMem(),
              *wgtScaleShift_->getIntlMem(),
              *dataTop_->getIntlMem())
          : batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
              *mean_->getIntlMem(), *var_->getIntlMem(),
              *dataTop_->getIntlMem()));
    } else {
    //  LOG(INFO) << "should be here usually";
      fwd.push_back(useScaleShift_
          ? batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
              *wgtScaleShift_->getIntlMem(),
              *dataTop_->getIntlMem(),
              *mean_->getIntlMem(), *var_->getIntlMem())
          : batch_normalization_forward(*fwdPD_, *dataBot_->getIntlMem(),
              *dataTop_->getIntlMem(),
              *mean_->getIntlMem(), *var_->getIntlMem()));
      // mean and var are outputs, submit them after BN fwd
      mean_->submitCvt(fwd, localMean_->getData());
      var_->submitCvt(fwd, localVar_->getData());
    }
  }
*/
  // submit top after BN fwd
  real *topdata = getOutputValue()->getData();
  dataTop_->submitCvt(fwd, topdata);
//LOG(INFO) << botdata[1] << "," << localMean_->getData()[1] << ","
//  << localVar_->getData()[1] << "," << myScaleShift_->getData()[1] << topdata[1];
  // start forward
  REGISTER_TIMER_INFO("mkldnn_BN_Fwd", getName().c_str());
  stream(stream::kind::eager).submit(fwd).wait();

  if (passType == PASS_TRAIN && !useGlobalStats_) {
    //LOG(INFO) << "cal moving .............. should not in testing";
    // calculating and saving moving mean and variance
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
      movingVar->add(*localVar_, movingAvgFraction_, 1.0 - movingAvgFraction_);
    }
  }
//  if (passType != PASS_TEST)
//    LOG(INFO) << "my mean var" << localMean_->getData()[1] << "," << localVar_->getData()[1];
//  LOG(INFO) << "my ------------" << topdata[0] << "," << topdata[1] << "," << topdata[2] << "," << topdata[oc_*oh_[0]*ow_[0]-1];
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

  if (useGlobalStats_) {
    if (firstTest_) {
      setMeanAndStd();
      firstTest_ = false;
    }
  } else {
    calMeanAndStd(expandedIn_);
    firstTest_ = true;
  }
  normIn_->assign(*expandedIn_);
  normIn_->addBias(*localMean_, -1);  // subtract mean.
  normIn_->divRowVector(*savedInvVar_);  // divide std.
  expandedOut_->assign(*normIn_);

//  if (!useEx_){
//    LOG(INFO) << localMean_->getData()[1] << "," << savedInvVar_->getData()[1] << "," << normIn_->getData()[1];
//  }
  expandedOut_->mulRowVector(*weight_->getW());  // multiple gamma.
  if (biases_) {
    expandedOut_->addBias(*(biases_->getW()), 1);  // add beta.
  }

  if (useEx_) {
    // acutal in use
    MatrixPtr out = getOutputValue();
    shrinkMat(expandedOut_, out);
  } else {
    // just for my test to compare with my result
    // can be remove when finish this layer
    MatrixPtr out = Matrix::create(bs_, oc_*oh_[0]*ow_[0], false, false);//getOutputValue();
    shrinkMat(expandedOut_, out);
    //  real *topdata = out->getData();
    //  LOG(INFO) << "ex ------------" << topdata[0] << "," << topdata[1] << "," << topdata[2] << "," << topdata[oc_*oh_[0]*ow_[0]-1];
  }
}

void MkldnnBatchNormLayer::submitDnnFwd(PassType passType) {
  if (!useEx_) {
  //  exFwd(passType);
    myFwd(passType);
  } else {
    // only when ih==iw==1
    exFwd(passType);
  }
  // activation
  REGISTER_TIMER_INFO("mkldnn_BN_FwAtvTimer", getName().c_str());
  forwardActivation();
}

void MkldnnBatchNormLayer::exBwd(const UpdateCallback &callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }
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
  if(!useEx_) {
    // when use mkldnn should  prepare some matrix for ex Bwd
    Matrix::resizeOrCreate(normIn_, bs_ * imgPixels_, oc_, false, useGpu_);
    Matrix::resizeOrCreate(expandedIn_, bs_ * imgPixels_, oc_, false, useGpu_);
    savedInvVar_->copyFrom(*localVar_);
    savedInvVar_->sqrt(*savedInvVar_);
    expandMat(getInputValue(0), expandedIn_);
    normIn_->assign(*expandedIn_);
    normIn_->addBias(*localMean_, -1);  // subtract mean.
    normIn_->divRowVector(*savedInvVar_);  // divide std.
//    LOG(INFO) << localMean_->getData()[1] << "," << savedInvVar_->getData()[1] << "," << normIn_->getData()[1];
  }
  expandMat(getOutputGrad(), expandedOutGrad_);
  // compute derivatives.
  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("BpBiasTimer", getName().c_str());
    biases_->getWGrad()->collectBias(*expandedOutGrad_, 1);
    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
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
    REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
    weight_->getParameterPtr()->incUpdate(callback);
  }

}

void MkldnnBatchNormLayer::submitDnnBwd(const UpdateCallback &callback) {

  exBwd(callback);

  // dnn backward
}

}  // namespace paddle
