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
#include "MkldnnFcLayer.h"


// ex fc
#include "paddle/math/SparseMatrix.h"
#include <vector>
#include <algorithm>


namespace paddle {

REGISTER_LAYER(mkldnn_fc, MkldnnFcLayer);

bool MkldnnFcLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  // only support 1 input layer by now
  CHECK_EQ(config_.inputs_size(), 1);
  CHECK(inputLayers_.size() == parameters_.size());
  
  bs_ = 0;
  oc_ = getSize();
  has_spatial_ = false;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    // Option the parameters
    ic_.push_back(inputLayers_[i]->getSize());
    iw_.push_back(0);
    ih_.push_back(0);
    ow_.push_back(0);
    oh_.push_back(0);
    // create a new weight
    if (parameters_[i]->isSparse()) {
      CHECK_LE(parameters_[i]->getSize(), oc_ * ic_[i]);
    } else {
      CHECK_EQ(parameters_[i]->getSize(), oc_ * ic_[i]);
    }
    // TODO: when backward done maybe we donot need reley on weight format of paddle
    Weight* w = new Weight(ic_[i], oc_, parameters_[i]);

    // append the new weight to the list
    weights_.emplace_back(w);
  }

  // initialize biases_
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
    hasBias_ = true;
  }
  return true;
}

// keep for paddle
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

size_t MkldnnFcLayer::getOneBatchSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = 0;
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    int height = inputLayers_[i]->getOutput().getFrameHeight();
    int width = inputLayers_[i]->getOutput().getFrameWidth();
    if (height > 1 || width > 1) {
      has_spatial_ = true;
      ih_[i] = height;
      iw_[i] = width;
    } else {
      has_spatial_ = false;
      ih_[i] = 1;
      iw_[i] = 1;
    }
    oh_[i] = 1;
    ow_[i] = 1;
    CHECK(ih_[i] * iw_[i]);
    CHECK(layerSize == 0 || size_t(oh_[i] * ow_[i] * oc_) == layerSize);
    layerSize = oh_[i] * ow_[i] * oc_;
  }
  return layerSize;
}

// whether reset batchsize and image size of input and output 
bool MkldnnFcLayer::reshapeOutput() {
  REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
  if (bs_ == getInput(0).getBatchSize()) {
    // can remove reserveOutput when confirm how multi inputs work
    // and whether to clear diff
    reserveOutput(bs_, getOneBatchSize()); 
    return false;
  }
  // reserve data
  bs_ = getInput(0).getBatchSize();
  LOG(INFO) << "reshape batch size: " << bs_;
  reserveOutput(bs_, getOneBatchSize());
  return true;
}

void MkldnnFcLayer::resetDnnFwd() {
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
  
}

void MkldnnFcLayer::resetDnnBwd() {
  /*LOG(INFO) << "init or reset fc backward of layer: " << config_.name();
  */
}

void MkldnnFcLayer::myFwd(PassType passType) {
  /// all sumbit cvt should be clear
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
}

void MkldnnFcLayer::exFwd(PassType passType) {
  /* malloc memory for the output_ if necessary */
  //int batchSize = getInput(0).getBatchSize();
  //int size = getSize();
  //reserveOutput(batchSize, size);
  //MatrixPtr outV = getOutputValue();

  MatrixPtr outV = Matrix::create(bs_, oc_, false, false);
  outV->zeroMem();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    auto input = getInput(i);
    CHECK(input.value) << "The input of 'fc' layer must be matrix";
    i == 0 ? outV->mul(input.value, weights_[i]->getW(), 1, 0)
           : outV->mul(input.value, weights_[i]->getW(), 1, 1);
  }

  /* add the bias-vector */
  if (biases_.get() != NULL) {
    outV->addBias(*(biases_->getW()), 1);
  }

  real *topdata = outV->getData();
  LOG(INFO) << "ex ------------" << topdata[0] << "," << topdata[1] << "," << topdata[2];

  //forwardActivation();
  
}

void MkldnnFcLayer::submitDnnFwd(PassType passType) {
  myFwd(passType);

//  exFwd(passType);
}

void MkldnnFcLayer::exBwd(const UpdateCallback &callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("BpBiasTimer", getName().c_str());
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  bool syncFlag = hl_get_sync_flag();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    /* Calculate the W-gradient for the current layer */
    if (weights_[i]->getWGrad()) {
      MatrixPtr input_T = getInputValue(i)->getTranspose();
      MatrixPtr oGrad = getOutputGrad();
      {
        REGISTER_TIMER_INFO("GradMulTimer", getName().c_str());
        weights_[i]->getWGrad()->mul(input_T, oGrad, 1, 1);
      }
    }

    // If callback does not change value, backprop error asynchronously so that
    // we can do the callback concurrently.
    hl_set_sync_flag(false);

    /* Calculate the input layers error */
    MatrixPtr preGrad = getInputGrad(i);
    if (NULL != preGrad) {
      MatrixPtr weights_T = weights_[i]->getW()->getTranspose();
      REGISTER_TIMER_INFO("BpMulTimer", getName().c_str());
      preGrad->mul(getOutputGrad(), weights_T, 1, 1);
    }

    hl_set_sync_flag(syncFlag);
    {
      REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }

}

void MkldnnFcLayer::submitDnnBwd(const UpdateCallback &callback) {

  exBwd(callback);

  // dnn backward
  /*
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    // backward weights before data, since may have not botdiff in some layer
    if (weights_[i]->getWGrad()) {
      submitBwdWgts(i, getPrev(i)->getOutputValue(), getOutputGrad());
      // Increasing the number of gradient 
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
    submitBwdData(i, getOutputGrad(), getPrev(i)->getOutputGrad());
  }
  if (biases_ && biases_->getWGrad()) {
    // Increasing the number of gradient 
    biases_->getParameterPtr()->incUpdate(callback);
  }
  */
}

}  // namespace paddle
