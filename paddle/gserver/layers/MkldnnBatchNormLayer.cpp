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


// ex fc
#include "paddle/math/SparseMatrix.h"
#include <vector>
#include <algorithm>


namespace paddle {

REGISTER_LAYER(mkldnn_batch_norm, MkldnnBatchNormLayer);

const real MkldnnBatchNormLayer::EPS = 1E-5;

void MkldnnBatchNormLayer::calFeatureMapSize() {
  const ImageConfig& conf = config_.inputs(0).image_conf();
  if (inputLayers_[0]->getOutput().getFrameHeight() == 0 &&
      inputLayers_[0]->getOutput().getFrameWidth() == 0) {
    ih_[0] = conf.img_size();
    iw_[0] = conf.img_size();
  } else {
    ih_[0] = inputLayers_[0]->getOutput().getFrameHeight();
    iw_[0] = inputLayers_[0]->getOutput().getFrameWidth();
  }
  oh_[0] = ih_[0];
  ow_[0] = iw_[0];
  imgPixels_ = oh_[0] * ow_[0];
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);
}

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

  usePaddleFmt_ = true;
  if (config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }
  movingAvgFraction_ = config_.moving_average_fraction();

  weight_.reset(new Weight(1, oc_, parameters_[0]));
  movingMean_.reset(new Weight(1, oc_, parameters_[1]));
  movingVar_.reset(new Weight(1, oc_, parameters_[2]));

  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
    hasBias_ = true;
  }

  savedMean_ = Matrix::create(1, oc_, false, useGpu_);
  savedInvVar_ = Matrix::create(1, oc_, false, useGpu_);
  savedMean_->zeroMem();
  savedInvVar_->zeroMem();

  firstTest_ = true;

  return true;
}

void MkldnnBatchNormLayer::calMeanAndStd(const MatrixPtr& mat) {
  int numSamples = mat->getHeight();
  Matrix::resizeOrCreate(tmpMat_, numSamples, oc_, false, useGpu_);
  savedMean_->zeroMem();
  savedMean_->accumulateColSum(*mat);
  savedMean_->mulScalar(1.0 / numSamples);  // E[x]

  tmpMat_->assign(*mat);
  tmpMat_->square();
  savedInvVar_->zeroMem();
  savedInvVar_->accumulateColSum(*tmpMat_);
  savedInvVar_->mulScalar(1.0 / numSamples);  // E[x^2]
  savedInvVar_->addSquare(*savedMean_, -1.0);      // E[x^2] - E^2[x]

  // Variance may be small negative value
  // because of the subtraction operation.
  // Here using clipping.
  savedInvVar_->downClip(real(0.0));

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

    mvMean->add(*savedMean_, movingAvgFraction_, 1.0 - movingAvgFraction_);
    mvVar->add(*savedInvVar_, movingAvgFraction_, 1.0 - movingAvgFraction_);
  } else {
    // movingMean =  movingMean * movingAvgFraction_
    //            + savedMean_ * (1 - movingAvgFraction_)
    movingMean->add(*savedMean_, movingAvgFraction_, 1.0 - movingAvgFraction_);
    // movingVar =  movingVar * movingAvgFraction_
    //           + savedInvVar_ * (1 - movingAvgFraction_)
    movingVar->add(*savedInvVar_, movingAvgFraction_, 1.0 - movingAvgFraction_);
  }
}

void MkldnnBatchNormLayer::setMeanAndStd() {
  savedMean_->copyFrom(*(movingMean_->getW()));
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

void MkldnnBatchNormLayer::resetDnnFwd() {
  /*
  LOG(INFO) << "reset mkldnn forward of fc layer: " << config_.name();

  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";
  hasBias_ = (biases_ && biases_->getW()) ? true : false;
  //create dim structure that describes user data.
  memory::dims botDims, wgtDims, biasDims, topDims;
  memory::format botFmt, wgtFmt, biasFmt, topFmt;
  if (!has_spatial_) {
    botDims = {bs_, ic_[0]};
    wgtDims = {oc_, ic_[0]}; // transpose from paddle weight
    // TODO: when backward done maybe we donot need reley on weight format of paddle
    botFmt = memory::format::nc;
    wgtFmt = memory::format::oi;
  } else {
    botDims = {bs_, ic_[0], ih_[0], iw_[0]};
    wgtDims = {oc_, ic_[0], ih_[0], iw_[0]};
    botFmt = memory::format::nchw;
    wgtFmt = memory::format::oihw;
  }
  // no matter what inputs
  topDims = {bs_, oc_};
  topFmt = memory::format::nc;
  biasDims = {oc_};
  biasFmt = memory::format::x;

  dataBot_.reset(new MkldnnBuffer(botDims));
  dataTop_.reset(new MkldnnBuffer(topDims));
  dataWgt_.reset(new MkldnnBuffer(wgtDims));
  if (hasBias_) {
    dataBias_.reset(new MkldnnBuffer(biasDims));
  }

  // init user memory of bottom, weights and bias
  real *botData = getPrev(0)->getOutputValue()->getData();
  real *topData = getOutputValue()->getData();
  real *wgtData = weights_[0]->getW()->getData();
  const std::shared_ptr<mkldnn::memory::desc> prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    dataBot_->initUser(botData, *prvMD, *engine_);
  } else {
    
    dataBot_->initUser(botData, botDims, botFmt, *engine_);
  }
  dataWgt_->initUser(wgtData, wgtDims, wgtFmt, *engine_);

  // create fc desc from internal desc 
  std::shared_ptr<inner_product_forward::desc> fwdDesc;
  if (hasBias_) {
    real *biasData = biases_->getW()->getData();
    dataBias_->initUser(biasData, biasDims, biasFmt, *engine_);
    fwdDesc.reset(new inner_product_forward::desc(
        prop_kind::forward_training, dataBot_->getMDAny(),
        dataWgt_->getMDAny(), dataBias_->getMDAny(), dataTop_->getMDAny()));
  } else {
    fwdDesc.reset(new inner_product_forward::desc(
        prop_kind::forward_training, dataBot_->getMDAny(),
        dataWgt_->getMDAny(), dataTop_->getMDAny()));
  }  
  fwdPD_.reset(new inner_product_forward::primitive_desc(*fwdDesc, *engine_));
  // init cvt
  if (dataBot_->initCvt(fwdPD_->src_primitive_desc(), dnnCvtUser2Internal)) {
    LOG(INFO) << "need reorder --- bottom data: "
      << DNN_FORMAT[dataBot_->getUserFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataBot_->getIntlFmt()];
  }
  if (dataWgt_->initCvt(fwdPD_->weights_primitive_desc(), dnnCvtUser2Internal)) {
    LOG(INFO) << "need reorder --- weight data: "
      << DNN_FORMAT[dataWgt_->getUserFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataWgt_->getIntlFmt()];
  }
  if (hasBias_) {
    if (dataBias_->initCvt(fwdPD_->bias_primitive_desc(), dnnCvtUser2Internal)) {
      LOG(INFO) << "need reorder --- bias data: "
        << DNN_FORMAT[dataBias_->getUserFmt()]
        << " >>>>> "
        << DNN_FORMAT[dataBias_->getIntlFmt()];
    }
  }
  // init top user memory and cvt
  if (setDnnTopDataFmt_) {
    dataTop_->initUser(topData, fwdPD_->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
  } else {
    dataTop_->initUser(topData, topDims, topFmt, *engine_);
  }
  if (dataTop_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtInternal2User)) {
    LOG(INFO) << "need reorder --- top data: " 
      << DNN_FORMAT[dataTop_->getIntlFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataTop_->getUserFmt()];
  }
  LOG(INFO) << "data format flow --- "
    << DNN_FORMAT[dataBot_->getUserFmt()] << " >>> ("
    << DNN_FORMAT[dataBot_->getIntlFmt()] << " >>> "
    << DNN_FORMAT[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FORMAT[dataTop_->getUserFmt()];
  
  printInfo();
  */
}

void MkldnnBatchNormLayer::resetDnnBwd() {
  /*LOG(INFO) << "init or reset fc backward of layer: " << config_.name();
  */
}

void MkldnnBatchNormLayer::myFwd(PassType passType) {
/*  /// all sumbit cvt should be clear
  clearAllCvtFlags();
  CHECK(getInput(0).value) << "The input of 'fc' layer must be matrix";

  real *botdata = getPrev(0)->getOutputValue()->getData();
  real *topdata = getOutputValue()->getData();

  // so far used paddle's format, transpose. TODO: use dnn format to save time in future!
  MatrixPtr myWgt = Matrix::create(oc_, ic_[0], false, false);
  weights_[0]->getW()->transpose(myWgt, false);
  real *wgtdata = myWgt->getData();//weights_[0]->getW()->getData();

  std::vector<primitive> fwd;
  dataBot_->submitCvt(fwd, botdata);
  dataWgt_->submitCvt(fwd, wgtdata);
  if(hasBias_) {
    real *biasdata = biases_->getW()->getData();
    dataBias_->submitCvt(fwd, biasdata);
    fwd.push_back(inner_product_forward(*fwdPD_,
      *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
      *(dataBias_->getIntlMem()),*(dataTop_->getIntlMem())));
  } else {
    fwd.push_back(inner_product_forward(*fwdPD_,
      *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
      *(dataTop_->getIntlMem())));
  }
  dataTop_->submitCvt(fwd, topdata);

  // start forward
  REGISTER_TIMER_INFO("mkldnn_FcFwd", getName().c_str());
  stream(stream::kind::eager).submit(fwd).wait();
  
//  LOG(INFO) << "my ------------" << topdata[0] << "," << topdata[1] << "," << topdata[2];

  // activation
  REGISTER_TIMER_INFO("mkldnn_FcFwAtvTimer", getName().c_str());
  forwardActivation();
  */
}

void MkldnnBatchNormLayer::exFwd(PassType passType) {

  int batchSize = getInputValue(0)->getHeight();
  calFeatureMapSize();

  // for testing in training peroid.
  // so in train always calculate it
  // in test depends on choice
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
  normIn_->addBias(*savedMean_, -1);  // subtract mean.
  normIn_->divRowVector(*savedInvVar_);  // divide std.

  expandedOut_->assign(*normIn_);
  expandedOut_->mulRowVector(*weight_->getW());  // multiple gamma.
  if (biases_) {
    expandedOut_->addBias(*(biases_->getW()), 1);  // add beta.
  }
  MatrixPtr out = getOutputValue();
  shrinkMat(expandedOut_, out);


  REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
  forwardActivation();


//  real *topdata = outV->getData();
//  LOG(INFO) << "ex ------------" << topdata[0] << "," << topdata[1] << "," << topdata[2];


  
}

void MkldnnBatchNormLayer::submitDnnFwd(PassType passType) {
//  myFwd(passType);

  exFwd(passType);
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
