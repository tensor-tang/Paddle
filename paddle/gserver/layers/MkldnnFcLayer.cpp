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



#include "paddle/math/SparseMatrix.h"
#include <vector>
#include <algorithm>


#include "MkldnnFcLayer.h"


namespace paddle {

REGISTER_LAYER(mkldnn_fc, MkldnnFcLayer);

bool MkldnnFcLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {

  CHECK(inputLayers_.size() == parameters_.size());
  LOG(INFO) << "input size " << inputLayers_.size();
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    // Option the parameters
    size_t height = inputLayers_[i]->getSize();
    size_t width = getSize();

    // create a new weight
    if (parameters_[i]->isSparse()) {
      CHECK_LE(parameters_[i]->getSize(), width * height);
    } else {
      CHECK_EQ(parameters_[i]->getSize(), width * height);
    }
    Weight* w = new Weight(height, width, parameters_[i]);

    // append the new weight to the list
    weights_.emplace_back(w);
  }

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }




/*
  CHECK_EQ(config_.inputs_size(), 1);

  const PoolConfig& conf = config_.inputs(0).pool_conf();
//  if (!conf.caffe_mode()) {
//    LOG(FATAL) << "Only support caffe mode with MKL-DNN by now!";
//  }
  const std::string& poolType_ = conf.pool_type();
  if (poolType_ == "max-projection") {
    poolAlgo_= algorithm::pooling_max;
  } else if (poolType_ == "avg-projection") {
    poolAlgo_ = algorithm::pooling_avg;
    LOG(FATAL) << "Only support max pooling by now!";
  } else {
    LOG(FATAL) << "unknow pooling type!";
  }
  
  ic_.push_back(conf.channels());
  iw_.push_back(conf.img_size());
  ow_.push_back(conf.output_x());
  ih_.push_back(conf.has_img_size_y() ? conf.img_size_y() : conf.img_size());
  oh_.push_back(conf.has_output_y() ? conf.output_y() : conf.output_x());

  fw_ = conf.size_x();
  sw_ = conf.stride();
  pw_ = conf.padding();

  fh_ = conf.has_size_y() ? conf.size_y() : conf.size_x();
  sh_ = conf.has_stride_y() ? conf.stride_y() : conf.stride();
  ph_ = conf.has_padding_y() ? conf.padding_y() : conf.padding();
  bs_ = 0;
  oc_ = ic_[0];
  */
  return true;
}

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
  return 1;
  /*CHECK_NE(inputLayers_.size(), 0UL);

  int height = inputLayers_[0]->getOutput().getFrameHeight();
  int width = inputLayers_[0]->getOutput().getFrameWidth();
  
  if (height != 0) ih_[0] = height;
  if (width != 0) iw_[0] = width;
  // output image dimensions
  // TODO: why default use false caffe mode??
  oh_[0] = outputSize(ih_[0], fh_, ph_, sh_);
  ow_[0] = outputSize(iw_[0], fw_, pw_, sw_);
  
  return oh_[0] * ow_[0] * oc_;

  */
}

// whether reset batchsize and image size of input and output 
bool MkldnnFcLayer::reshapeOutput() {
  return false;
/*
  if (bs_ == getInput(0).getBatchSize()) {
    // can remove resetoutput when confirm how multi inputs work and whether to clear diff
    resetOutput(bs_, getOneBatchSize()); 
    return false;
  }
  
  // reset image size
  size_t layersize = getOneBatchSize();
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);
  
  // reset data
  bs_ = getInput(0).getBatchSize();
  LOG(INFO) << "layer name: " << getName();
  LOG(INFO) << "reshape batch size: " << bs_;
  resetOutput(bs_, layersize);
  return true;
  */
}

void MkldnnFcLayer::resetDnnFwd() {
  LOG(INFO) << "reset mkldnn forward of pool layer: " << config_.name();
/*
  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";
  
  // create dim structure that describes user data.
  memory::dims botDims = {bs_, ic_[0], ih_[0], iw_[0]};
  memory::dims kernel = {fh_, fw_};
  memory::dims strides = {sh_, sw_};
  memory::dims padding = {ph_, pw_};
  memory::dims topDims = {bs_, oc_, oh_[0], ow_[0]};

  dataBot_.reset(new MkldnnBuffer(botDims));
  dataTop_.reset(new MkldnnBuffer(topDims));
  
  // init user memory of bottom, weights and bias
  real *botData = getPrev(0)->getOutputValue()->getData();
  real *topData = getOutputValue()->getData();
  const std::shared_ptr<mkldnn::memory::desc> prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    dataBot_->initUser(botData, *prvMD, *engine_);
  } else {
    dataBot_->initUser(botData, botDims, memory::format::nchw, *engine_);
  }
  // create pool desc from internal desc 
  std::shared_ptr<pooling_forward::desc> fwdDesc;
  fwdDesc.reset(new pooling_forward::desc(prop_kind::forward_training, poolAlgo_,
                      dataBot_->getMDAny(), dataTop_->getMDAny(),
                      strides, kernel, padding, padding,
                      padding_kind::zero));
  // init cvt
  if (dataBot_->initCvt(dataBot_->getUserPD(), dnnCvtUser2Internal)) {
    LOG(INFO) << "need reorder --- bottom data: "
      << DNN_FORMAT[dataBot_->getUserFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataBot_->getIntlFmt()];
  }
  fwdPD_.reset(new pooling_forward::primitive_desc(*fwdDesc, *engine_));

  // init top user memory and cvt
  if (setDnnTopDataFmt_) {
    dataTop_->initUser(topData, fwdPD_->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
  } else {
    dataTop_->initUser(topData, topDims, memory::format::nchw, *engine_);
  }
  if (dataTop_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtInternal2User)) {
    LOG(INFO) << "need reorder --- top data: " 
      << DNN_FORMAT[dataTop_->getIntlFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataTop_->getUserFmt()];
  }
  
  bool isMax = poolAlgo_ != algorithm::pooling_avg;
  if (isMax) {
    workspace_.reset(new memory(fwdPD_->workspace_primitive_desc()));
  } else {
    auto p_workspace_desc = memory::primitive_desc(
      {{}, memory::data_type::f32, memory::format(dataTop_->getIntlFmt())}, *engine_);
    workspace_.reset(new memory(p_workspace_desc));
  }
  LOG(INFO) << "data format flow --- "
    << DNN_FORMAT[dataBot_->getUserFmt()] << " >>> ("
    << DNN_FORMAT[dataBot_->getIntlFmt()] << " >>> "
    << DNN_FORMAT[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FORMAT[dataTop_->getUserFmt()];
  
  printInfo();
  */
}

void MkldnnFcLayer::resetDnnBwd() {
  /*LOG(INFO) << "init or reset conv backward of layer: " << config_.name();
  */
}

void MkldnnFcLayer::myFwd(PassType passType) {
  /*
  /// all sumbit cvt should be clear
  clearAllCvtFlags();

  real *botdata = getPrev(0)->getOutputValue()->getData();
  real *topdata = getOutputValue()->getData();

  std::vector<primitive> poolFwd;
  dataBot_->submitCvt(poolFwd, botdata);
  
  if(poolAlgo_ == algorithm::pooling_max) {
    poolFwd.push_back(pooling_forward(*fwdPD_,
      *(dataBot_->getIntlMem()), *(dataTop_->getIntlMem()),
      *workspace_));
  } else {
    poolFwd.push_back(pooling_forward(*fwdPD_,
      *(dataBot_->getIntlMem()), *(dataTop_->getIntlMem())));
  }
  dataTop_->submitCvt(poolFwd, topdata);

  // start forward
  REGISTER_TIMER_INFO("mkldnnPoolFwd", getName().c_str());
  stream(stream::kind::eager).submit(poolFwd).wait();
//  LOG(INFO) << "------------" << topdata[0];// << "," << topdata[1] << "," << topdata[2];
*/
}

void MkldnnFcLayer::exFwd(PassType passType) {
  /* malloc memory for the output_ if necessary */
  int batchSize = getInput(0).getBatchSize();
  int size = getSize();

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    reserveOutput(batchSize, size);
  }

  MatrixPtr outV = getOutputValue();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    auto input = getInput(i);
    CHECK(input.value) << "The input of 'fc' layer must be matrix";
    REGISTER_TIMER_INFO("FwMulTimer", getName().c_str());
    i == 0 ? outV->mul(input.value, weights_[i]->getW(), 1, 0)
           : outV->mul(input.value, weights_[i]->getW(), 1, 1);
  }

  /* add the bias-vector */
  if (biases_.get() != NULL) {
    REGISTER_TIMER_INFO("FwBiasTimer", getName().c_str());
    outV->addBias(*(biases_->getW()), 1);
  }

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }

//  real *topdata = getOutputValue()->getData();
//  LOG(INFO) << "------------" << topdata[0];// << "," << topdata[1] << "," << topdata[2];
}

void MkldnnFcLayer::submitDnnFwd(PassType passType) {
  exFwd(passType);
//  myFwd(passType);
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
