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
#include "DnnConvLayer.h"

//using namespace mkldnn;

#pragma GCC diagnostic ignored "-Wunused-but-set-variable"


namespace paddle {


REGISTER_LAYER(dnnconv, DnnConvLayer);


bool DnnConvLayer::init(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ConvBaseLayer::init(layerMap, parameterMap);

  // for dnn init

// only support float by now
  /*
  convFwd_ = NULL;
  convBwdBias_ = NULL;
  convBwdData_ = NULL;
  convBwdFilter_ = NULL;
*/
  ih_.clear();
  iw_.clear();
  oh_.clear();
  ow_.clear();
  fh_.clear();
  fw_.clear();
  ph_.clear();
  pw_.clear();
  sh_.clear();
  sw_.clear();
  ic_.clear();
  gp_.clear();
  oc_ = numFilters_;
  /* Initialize the projection */
  for (auto &inputConfig : config_.inputs()) {
    const ConvConfig &conf = inputConfig.conv_conf();
    ih_.push_back(conf.img_size());
    iw_.push_back(conf.img_size());
    oh_.push_back(conf.output_x());
    ow_.push_back(conf.output_x());
    fw_.push_back(conf.filter_size());
    fh_.push_back(conf.filter_size_y());
    pw_.push_back(conf.padding());
    ph_.push_back(conf.padding_y());
    sw_.push_back(conf.stride());
    sh_.push_back(conf.stride_y());
    ic_.push_back(conf.channels());
    gp_.push_back(conf.groups());
    
    /* Consistent caffe mode for multiple input */
    caffeMode_ = conf.caffe_mode();
  }
  LOG(INFO) << "input layer size:" << inputLayers_.size();
  if (inputLayers_.size() != 1) {
     // TODO: only considerate One input
    LOG(FATAL) << "Only support one input layer with MKL-DNN by now!";
  }
  // if has bias
  if (biasParameter_.get() != NULL) {
    hasBias_ = true;
  } else {
    hasBias_ = false;
  }
  if (hasBias_ && !sharedBiases_) {
    LOG(FATAL) << "Do not support unshared bias with MKL DNN by now!";
  }
  if (!caffeMode_) {
    LOG(FATAL) << "Do not support un-caffe mode with MKL DNN by now!";
  }
  return true;
}

size_t DnnConvLayer::getSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = 0;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    // all output size should be the same
    // TODO: only care about i==0 by now
    // check out the output size
    CHECK(layerSize == 0 || oh_[i] * ow_[i] * size_t(numFilters_) == layerSize);
    layerSize = oh_[i] * ow_[i] * numFilters_;
  }
  return layerSize;
}

void DnnConvLayer::initDnnFwd() {
  // TODO: only care about i==0 by now
  memory::dims biasDims = {oc_};
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
  
    int bs = getInput(i).getBatchSize();
    LOG(INFO) << "init conv forward of layer: " << config_.name();
    LOG(INFO) << "batch size: " << bs;

    // init input and output image size
    ih_[i] = inputLayers_[i]->getOutput().getFrameHeight();
    iw_[i] = inputLayers_[i]->getOutput().getFrameWidth();
    if (ih_[i] == 0) ih_[i] = imgSize_[i];
    if (iw_[i] == 0) iw_[i] = imgSize_[i];
    // output image dimensions
    oh_[i] = outputSize(ih_[i], fh_[i], ph_[i], sh_[i]);
    ow_[i] = outputSize(iw_[i], fw_[i], pw_[i], sw_[i]);
    getOutput().setFrameHeight(oh_[0]);
    getOutput().setFrameWidth(ow_[0]);
    resetOutput(bs, getSize());
    // create dim structure that describes user data.
    memory::dims botDims = {bs, ic_[i], ih_[i], iw_[i]};
    memory::dims wgtDims = (gp_[i] == 1) ? 
        memory::dims{oc_, ic_[i], fh_[i], fw_[i]}
      : memory::dims{gp_[i], oc_/gp_[i], ic_[i]/gp_[i], fh_[i], fw_[i]};
    
    memory::dims strides = {sh_[i], sw_[i]};
    memory::dims padding = {ph_[i], pw_[i]};
    memory::dims topDims = {bs, oc_, oh_[i], ow_[i]};
    
    // Important: need confirm the user data memory is allocated!!!  
    real *botData = getPrev(i)->getOutputValue()->getData();
    real *wgtData = weights_[i]->getW()->getData(); // TODO: check index
    if (hasBias_) {
      real *biasData = biases_->getW()->getData();
      dataBias_->initUser(biasData, biasDims, memory::format::x, engineCpu_);
    }
    real *topData = getOutputValue()->getData();
  
    // init user memory
    dataBot_->initUser(botData, botDims, memory::format::nchw, engineCpu_);
    dataWgt_->initUser(wgtData, wgtDims, (gp_[i] == 1) ?
                 memory::format::oihw : memory::format::goihw, engineCpu_);
    dataTop_->initUser(topData, topDims, memory::format::nchw, engineCpu_);

    // create conv desc from internal desc 
    std::shared_ptr<convolution_forward::desc> fwdDesc;
    if (hasBias_) {
      fwdDesc.reset(new convolution_forward::desc(prop_kind::forward_training,
                          algorithm::convolution_direct,
                          dataBot_->getInitIntlMD(),
                          dataWgt_->getInitIntlMD(),
                          dataBias_->getInitIntlMD(),
                          dataTop_->getInitIntlMD(),
                          strides, padding, padding, // TODO: check here left and right
                          padding_kind::zero));
    } else {
      fwdDesc.reset(new convolution_forward::desc(prop_kind::forward_training,
                          algorithm::convolution_direct,
                          dataBot_->getInitIntlMD(),
                          dataWgt_->getInitIntlMD(),
                          dataTop_->getInitIntlMD(),
                          strides, padding, padding, // TODO: check here left and right
                          padding_kind::zero));
    }
    fwdPD_.reset(new convolution_forward::primitive_desc(*fwdDesc, engineCpu_));

    // init reorder
    dataBot_->initCvt(memory::primitive_desc(
                        fwdPD_->src_primitive_desc()), dnnCvtUser2Internal);
    dataWgt_->initCvt(memory::primitive_desc(
                        fwdPD_->weights_primitive_desc()), dnnCvtUser2Internal);
    if (hasBias_) {
      dataBias_->initCvt(memory::primitive_desc(
                          fwdPD_->bias_primitive_desc()), dnnCvtUser2Internal);
    }
    dataTop_->initCvt(memory::primitive_desc(
                        fwdPD_->dst_primitive_desc()), dnnCvtInternal2User);

  }
  printInfo();
}

void DnnConvLayer::initDnnBwd() {
  // TODO: only care about i==0 by now
  int bs = getInput(0).getBatchSize();
  memory::dims topDims = {bs, oc_, oh_[0], ow_[0]};
  real *topdiff = getOutputGrad()->getData();
  // init top diff user
  diffTop_.reset(new DnnBuffer());
  diffTop_->initUser(topdiff, topDims, memory::format::nchw, engineCpu_);
  
  /* backward bias*/
  memory::dims biasDims = {oc_};
  if (hasBias_&& biases_->getWGrad()) {
    real* biasdiff = biases_->getWGrad()->getData();
    // init bias diff user
    diffBias_.reset(new DnnBuffer());
    diffBias_->initUser(biasdiff, biasDims, memory::format::x, engineCpu_);
    // bias backward can not be execute seperately, 
    // can only execute with filter bakcward
  }
  
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      continue; // TODO: donot support missing bottom diff by now
    }
    CHECK(bs == getInput(i).getBatchSize());

    // create dim structure that describes user data.
    memory::dims botDims = {bs, ic_[i], ih_[i], iw_[i]};
    memory::dims wgtDims = (gp_[i] == 1) ? 
        memory::dims{oc_, ic_[i], fh_[i], fw_[i]}
      : memory::dims{gp_[i], oc_/gp_[i], ic_[i]/gp_[i], fh_[i], fw_[i]};
    memory::dims strides = {sh_[i], sw_[i]};
    memory::dims padding = {ph_[i], pw_[i]};
    
    /* backward data */
    // init bottom diff user
    // Important: need confirm the user data memory is allocated!!!  
    LOG(INFO) << "init backward data";
    real* botdiff = prevLayer->getOutputGrad()->getData();
    diffBot_.reset(new DnnBuffer());
    diffBot_->initUser(botdiff, botDims, memory::format::nchw, engineCpu_);
    // init backward data primitive desc
    std::shared_ptr<convolution_forward::desc> bwdDataFwdDesc;
    std::shared_ptr<convolution_backward_data::desc> bwdDataDesc;
    bwdDataFwdDesc.reset(new convolution_forward::desc(
      prop_kind::forward_training, algorithm::convolution_direct, 
      diffBot_->getInitIntlMD(), 
      dataWgt_->getIntlMem()->get_primitive_desc().desc(), 
      diffTop_->getInitIntlMD(), 
      strides, padding, padding, padding_kind::zero));
    bwdDataDesc.reset(new convolution_backward_data::desc(
      algorithm::convolution_direct,
      diffBot_->getInitIntlMD(),
      dataWgt_->getIntlMem()->get_primitive_desc().desc(),
      diffTop_->getInitIntlMD(),
      strides, padding, padding, padding_kind::zero));
    std::shared_ptr<convolution_forward::primitive_desc> bwdDataFwdPD;
    bwdDataFwdPD.reset(new convolution_forward::primitive_desc(
      *bwdDataFwdDesc, engineCpu_));
    bwdDataPD_.reset(new convolution_backward_data::primitive_desc(*bwdDataDesc,
                        engineCpu_, *bwdDataFwdPD));
    // init reorder
    diffTop_->initCvt(bwdDataPD_->diff_dst_primitive_desc(), dnnCvtUser2Internal);
    diffBot_->initCvt(bwdDataPD_->diff_src_primitive_desc(), dnnCvtInternal2User);
    CHECK(dataWgt_->getIntlMem()->get_primitive_desc() ==
      bwdDataPD_->weights_primitive_desc());
    
    /* backward weight and bias*/
    if (weights_[i]->getWGrad()) {
      real* wgtdiff = weights_[i]->getWGrad()->getData();
      // init weight diff user
      diffWgt_.reset(new DnnBuffer());
      diffWgt_->initUser(wgtdiff, wgtDims, (gp_[i] == 1) ?
                 memory::format::oihw : memory::format::goihw, engineCpu_);
    } else {
      continue;
    }
    LOG(INFO) << "init backward weights and bias";
    std::shared_ptr<convolution_forward::desc> bwdWgtFwdDesc;
    std::shared_ptr<convolution_backward_weights::desc> bwdWgtDesc;
    if (hasBias_ && diffBias_ != NULL) {
      bwdWgtFwdDesc.reset(new convolution_forward::desc(
        prop_kind::forward_training, algorithm::convolution_direct, 
        dataBot_->getIntlMem()->get_primitive_desc().desc(),
        diffWgt_->getInitIntlMD(), diffBias_->getInitIntlMD(), 
        diffTop_->getIntlMem()->get_primitive_desc().desc(),
        strides, padding, padding, padding_kind::zero));
      bwdWgtDesc.reset(new convolution_backward_weights::desc(
        algorithm::convolution_direct, 
        dataBot_->getIntlMem()->get_primitive_desc().desc(),
        diffWgt_->getInitIntlMD(), diffBias_->getInitIntlMD(), 
        diffTop_->getIntlMem()->get_primitive_desc().desc(),
        strides, padding, padding, padding_kind::zero));
    } else {
      bwdWgtFwdDesc.reset(new convolution_forward::desc(
        prop_kind::forward_training, algorithm::convolution_direct, 
        dataBot_->getIntlMem()->get_primitive_desc().desc(),
        diffWgt_->getInitIntlMD(),
        diffTop_->getIntlMem()->get_primitive_desc().desc(),
        strides, padding, padding, padding_kind::zero));
      bwdWgtDesc.reset(new convolution_backward_weights::desc(
        algorithm::convolution_direct,
        dataBot_->getIntlMem()->get_primitive_desc().desc(),
        diffWgt_->getInitIntlMD(),
        diffTop_->getIntlMem()->get_primitive_desc().desc(),
        strides, padding, padding, padding_kind::zero));
    }
    std::shared_ptr<convolution_forward::primitive_desc> bwdWgtFwdPD;
    bwdWgtFwdPD.reset(new convolution_forward::primitive_desc(
      *bwdWgtFwdDesc, engineCpu_));
    bwdWgtPD_.reset(new convolution_backward_weights::primitive_desc(
      *bwdWgtDesc, engineCpu_, *bwdWgtFwdPD));
    if (hasBias_ && diffBias_ != NULL) {
      diffBias_->initCvt(bwdWgtPD_->diff_bias_primitive_desc(),
                      dnnCvtInternal2User);
    }
    diffWgt_->initCvt(memory::primitive_desc(bwdWgtPD_->diff_weights_primitive_desc()),
                      dnnCvtInternal2User);
    CHECK(dataBot_->getIntlMem()->get_primitive_desc() ==
      bwdWgtPD_->src_primitive_desc());
    CHECK(diffTop_->getIntlMem()->get_primitive_desc() ==
      bwdWgtPD_->diff_dst_primitive_desc());
  }

}

void DnnConvLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* since can not get batchsize at layer.init(),
   * so init dnn at forward by now*/
  /// For dnn init
  if (!dnnFwdInited_) {
    initDnnFwd();
    dnnFwdInited_ = true;
  }

  int i=0;
  getPrev(i)->getOutputValue()->getData(); // TODO: if comment OK?
  // TODO: what if user's data address will change
  std::vector<primitive> fwd;
  dataBot_->submitCvt(fwd);
  dataWgt_->submitCvt(fwd);
  if(hasBias_) {
    dataBias_->submitCvt(fwd);
    fwd.push_back(convolution_forward(*fwdPD_,
                *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
                *(dataBias_->getIntlMem()), *(dataTop_->getIntlMem())));
  } else {
    fwd.push_back(convolution_forward(*fwdPD_, *(dataBot_->getIntlMem()),
                *(dataWgt_->getIntlMem()),
                *(dataTop_->getIntlMem())));
  }
  
  // output donot reorder untill last mkl-dnn layer
  dataBot_->submitCvt(fwd);

  // start forward
  stream(stream::kind::eager).submit(fwd).wait();
  LOG(INFO) << "!!!!!!!!Forward completed!!!!!!!";


  /* activation */
  forwardActivation();
}

void DnnConvLayer::backward(const UpdateCallback &callback) {
  backwardActivation();
  if (!dnnBwdInited_) {
    initDnnBwd();
    dnnBwdInited_ = true;
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    // First, calculate the input layers error 
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      return;
    }
    auto bwdData = convolution_backward_data(
      *bwdDataPD_, *(diffTop_->getIntlMem()),
      *(dataWgt_->getIntlMem()), *(diffBot_->getIntlMem()));
    std::vector<primitive> netData;

    diffTop_->submitCvt(netData);
    netData.push_back(bwdData);
    diffBot_->submitCvt(netData);
    stream(stream::kind::eager).submit(netData).wait();
    LOG(INFO) << "!!!!!!!!!backward data execute completed!!!!!!!!";

    if (weights_[i]->getWGrad()) {
      std::vector<primitive> netWgt;
      
      std::shared_ptr<convolution_backward_weights> bwdWgt;
      if (biases_ && biases_->getWGrad()) {
        // bias backward can only execute in filter backward with MKL-DNN
        bwdWgt.reset(new convolution_backward_weights(*bwdWgtPD_,
          *(dataBot_->getIntlMem()), *(diffTop_->getIntlMem()), 
          *(dataWgt_->getIntlMem()), *(dataBias_->getIntlMem())));
        netWgt.push_back(*bwdWgt);
        diffBias_->submitCvt(netWgt);
      } else {
        bwdWgt.reset(new convolution_backward_weights(*bwdWgtPD_,
          *(dataBot_->getIntlMem()), *(diffTop_->getIntlMem()), 
          *(dataWgt_->getIntlMem()), *(dataBias_->getIntlMem())));
        netWgt.push_back(*bwdWgt);
      }
      diffWgt_->submitCvt(netWgt);
      stream(stream::kind::eager).submit(netWgt).wait();
      LOG(INFO) << "!!!!!!!!!backward filter execute completed!!!!!!!!";
      // Increasing the number of gradient 
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
    
  }
  if (biases_ && biases_->getWGrad()) {
    // Increasing the number of gradient 
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
