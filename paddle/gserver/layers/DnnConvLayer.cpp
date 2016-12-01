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
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);
  
  // mkldnn only support float type by now
  bool sharedBiases = config_.shared_biases();
  if (biasParameter_.get() != NULL && !sharedBiases) {
    LOG(FATAL) << "Only support shared bias with MKL DNN by now!";
  }
  LOG(INFO) << "input layer size:" << inputLayers_.size();
  if (inputLayers_.size() != 1) {
     // TODO: considerate more than One input
    LOG(FATAL) << "Only support one input layer with MKL-DNN by now!";
  }
  
  oc_ = config_.num_filters();
  bs_ = 0;
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

    bool caffeMode = conf.caffe_mode();
    if (!caffeMode) {
      LOG(FATAL) << "Only support caffe mode with MKL-DNN by now!";
    }
  }

  /* initialize the weightList */
  CHECK(inputLayers_.size() == parameters_.size());
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    size_t height, width;
    height = oc_;
    width = ic_[i] * fh_[i] * fw_[i] / gp_[i];

    // create a new weight
    CHECK_EQ(parameters_[i]->getSize(), width * height);
    Weight* w = new Weight(height, width, parameters_[i]);
    weights_.emplace_back(w);
  }

  /* initialize the biases_ */
  if (biasParameter_.get() != NULL) {
    CHECK_EQ((size_t)oc_, biasParameter_->getSize());
    biases_ = std::unique_ptr<Weight>(new Weight(oc_, 1, biasParameter_));
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
    CHECK(layerSize == 0 || oh_[i] * ow_[i] * size_t(oc_) == layerSize);
    layerSize = oh_[i] * ow_[i] * oc_;
  }
  return layerSize;
}

void DnnConvLayer::initOrResetDnnFwd() {
  if (bs_ == getInput(0).getBatchSize()) {
    resetOutput(bs_, getSize());
    return;
  }
  bs_ = getInput(0).getBatchSize();
  resetOutput(bs_, getSize());
  
  LOG(INFO) << "init or reset conv forward of layer: " << config_.name();
  LOG(INFO) << " reshape batch size: " << bs_;

  // TODO: only care about i==0 by now
  memory::dims biasDims = {oc_};
  bool hasBias = (biases_ && biases_->getW()) ? true : false;
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize())
      << "Assert batchsize of input layers are equal";
    
    // reset image size of input and output 
    int height = inputLayers_[i]->getOutput().getFrameHeight();
    int width = inputLayers_[i]->getOutput().getFrameWidth();
    if (height != 0) ih_[i] = height;
    if (width != 0) iw_[i] = width;
    // output image dimensions
    oh_[i] = outputSize(ih_[i], fh_[i], ph_[i], sh_[i]);
    ow_[i] = outputSize(iw_[i], fw_[i], pw_[i], sw_[i]);
    getOutput().setFrameHeight(oh_[0]);
    getOutput().setFrameWidth(ow_[0]);

    // create dim structure that describes user data.
    memory::dims botDims = {bs_, ic_[i], ih_[i], iw_[i]};
    memory::dims wgtDims = (gp_[i] == 1) ? 
        memory::dims{oc_, ic_[i], fh_[i], fw_[i]}
      : memory::dims{gp_[i], oc_/gp_[i], ic_[i]/gp_[i], fh_[i], fw_[i]};
    memory::dims strides = {sh_[i], sw_[i]};
    memory::dims padding = {ph_[i], pw_[i]};
    memory::dims topDims = {bs_, oc_, oh_[i], ow_[i]};
       
    // init user memory
    real *botData = getPrev(i)->getOutputValue()->getData();
    real *topData = getOutputValue()->getData();
    real *wgtData = weights_[i]->getW()->getData();
    dataBot_.reset(new DnnBuffer());
    dataWgt_.reset(new DnnBuffer());
    dataTop_.reset(new DnnBuffer());
    dataBot_->initUser(botData, botDims, memory::format::nchw, engineCpu_);
    dataTop_->initUser(topData, topDims, memory::format::nchw, engineCpu_);
    dataWgt_->initUser(wgtData, wgtDims, (gp_[i] == 1) ?
                 memory::format::oihw : memory::format::goihw, engineCpu_);
    if (hasBias) {
      real *biasData = biases_->getW()->getData();
      dataBias_.reset(new DnnBuffer());
      dataBias_->initUser(biasData, biasDims, memory::format::x, engineCpu_);
    }
    
    // create conv desc from internal desc 
    std::shared_ptr<convolution_forward::desc> fwdDesc;
    if (hasBias) {
      fwdDesc.reset(new convolution_forward::desc(prop_kind::forward_training,
                          algorithm::convolution_direct,
                          dataBot_->getInitIntlMD(),
                          dataWgt_->getInitIntlMD(),
                          dataBias_->getInitIntlMD(),
                          dataTop_->getInitIntlMD(),
                          strides, padding, padding,
                          padding_kind::zero));
    } else {
      fwdDesc.reset(new convolution_forward::desc(prop_kind::forward_training,
                          algorithm::convolution_direct,
                          dataBot_->getInitIntlMD(),
                          dataWgt_->getInitIntlMD(),
                          dataTop_->getInitIntlMD(),
                          strides, padding, padding,
                          padding_kind::zero));
    }
    fwdPD_.reset(new convolution_forward::primitive_desc(*fwdDesc, engineCpu_));

    // init reorder
    if (dataBot_->initCvt(memory::primitive_desc(fwdPD_->src_primitive_desc()),
      dnnCvtUser2Internal)) {
      LOG(INFO) << "bottom data need reorder";
    }
    if (dataWgt_->initCvt(memory::primitive_desc(fwdPD_->weights_primitive_desc()),
      dnnCvtUser2Internal)) {
      LOG(INFO) << "weight data need reorder";
    }
    if (hasBias) {
      if (dataBias_->initCvt(memory::primitive_desc(fwdPD_->bias_primitive_desc()),
        dnnCvtUser2Internal)){
        LOG(INFO) << "bias data need reorder";
      }
    }
    if (dataTop_->initCvt(memory::primitive_desc(fwdPD_->dst_primitive_desc()),
      dnnCvtInternal2User)) {
      LOG(INFO) << "top data need reorder";
    }
  }
  printInfo();
  needBwdReset_ = true;
}

void DnnConvLayer::initOrResetDnnBwd() {
  if (!needBwdReset_) {
    return;
  }
  needBwdReset_ = false;
  bool hasBias = (biases_ && biases_->getWGrad()) ? true : false;
  // TODO: only care about i==0 by now
  memory::dims topDims = {bs_, oc_, oh_[0], ow_[0]};
  real *topdiff = getOutputGrad()->getData();
  // init top diff user
  diffTop_.reset(new DnnBuffer());
  diffTop_->initUser(topdiff, topDims, memory::format::nchw, engineCpu_);
  
  if (hasBias) {
    // bias backward can not be execute seperately, 
    //only can execute with filter bakcward
    memory::dims biasDims = {oc_};
    real* biasdiff = biases_->getWGrad()->getData();
    diffBias_.reset(new DnnBuffer());
    diffBias_->initUser(biasdiff, biasDims, memory::format::x, engineCpu_);
  }
  
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      continue; // TODO: donot support missing bottom diff by now
    }
    CHECK(bs_ == getInput(i).getBatchSize())
      << "Assert batchsize of input layers are equal";

    // create dim structure that describes user data.
    memory::dims botDims = {bs_, ic_[i], ih_[i], iw_[i]};
    memory::dims wgtDims = (gp_[i] == 1) ? 
        memory::dims{oc_, ic_[i], fh_[i], fw_[i]}
      : memory::dims{gp_[i], oc_/gp_[i], ic_[i]/gp_[i], fh_[i], fw_[i]};
    memory::dims strides = {sh_[i], sw_[i]};
    memory::dims padding = {ph_[i], pw_[i]};
    
    /* backward data *************************************/
    // init bottom diff user
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
    bwdDataFwdPD.reset(new convolution_forward::primitive_desc(*bwdDataFwdDesc, engineCpu_));
    bwdDataPD_.reset(new convolution_backward_data::primitive_desc(*bwdDataDesc,
                        engineCpu_, *bwdDataFwdPD));
    // init reorder
    if (diffTop_->initCvt(bwdDataPD_->diff_dst_primitive_desc(), 
      dnnCvtUser2Internal)) {
      LOG(INFO) << "top diff need reorder";
    }
    if (diffBot_->initCvt(bwdDataPD_->diff_src_primitive_desc(), 
      dnnCvtInternal2User)) {
      LOG(INFO) << "bottom diff need reorder";
    }
    CHECK(dataWgt_->getIntlMem()->get_primitive_desc() ==
      bwdDataPD_->weights_primitive_desc());
    
    /* backward weight and bias *************************************/
    if (weights_[i]->getWGrad()) {
      real* wgtdiff = weights_[i]->getWGrad()->getData();
      // init weight diff user
      diffWgt_.reset(new DnnBuffer());
      diffWgt_->initUser(wgtdiff, wgtDims, (gp_[i] == 1) ?
                 memory::format::oihw : memory::format::goihw, engineCpu_);
    } else {
      LOG(FATAL) << "should have weight";
    //  continue;
    }
    std::shared_ptr<convolution_forward::desc> bwdWgtFwdDesc;
    std::shared_ptr<convolution_backward_weights::desc> bwdWgtDesc;
    if (hasBias && diffBias_ != NULL) {
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
    if (hasBias && diffBias_ != NULL) {
      if (diffBias_->initCvt(bwdWgtPD_->diff_bias_primitive_desc(),
        dnnCvtInternal2User)) {
        LOG(INFO) << "bias diff need reorder";
      }
    }
    if (diffWgt_->initCvt(bwdWgtPD_->diff_weights_primitive_desc(),
      dnnCvtInternal2User)) {
      LOG(INFO) << "weight diff need reorder";
    }
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
  initOrResetDnnFwd();

  int i=0;
  std::vector<primitive> convFwd;
  dataBot_->submitCvt(convFwd, getPrev(i)->getOutputValue()->getData());
  dataWgt_->submitCvt(convFwd);
  if(biases_ && biases_->getW()) {
    dataBias_->submitCvt(convFwd);
    convFwd.push_back(convolution_forward(*fwdPD_,
                *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
                *(dataBias_->getIntlMem()), *(dataTop_->getIntlMem())));
  } else {
    convFwd.push_back(convolution_forward(*fwdPD_, *(dataBot_->getIntlMem()),
                *(dataWgt_->getIntlMem()),
                *(dataTop_->getIntlMem())));
  }
  
  // output donot reorder untill last mkl-dnn layer
  dataTop_->submitCvt(convFwd, getOutputValue()->getData());

  // start forward
  stream(stream::kind::eager).submit(convFwd).wait();
//  LOG(INFO) << "!!!!!!!!Forward completed!!!!!!!";


  /* activation */
  forwardActivation();
}

void DnnConvLayer::backward(const UpdateCallback &callback) {
  backwardActivation();
  
  initOrResetDnnBwd();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    // First, calculate the input layers error 
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      return;
    }
    auto bwdData = convolution_backward_data(
      *bwdDataPD_, *(diffTop_->getIntlMem()),
      *(dataWgt_->getIntlMem()), *(diffBot_->getIntlMem()));
    std::vector<primitive> convBwdData;

    diffTop_->submitCvt(convBwdData, getOutputGrad()->getData());
    convBwdData.push_back(bwdData);
    diffBot_->submitCvt(convBwdData, prevLayer->getOutputGrad()->getData());
    stream(stream::kind::eager).submit(convBwdData).wait();
 //   LOG(INFO) << "!!!!!!!!!backward data execute completed!!!!!!!!";

    if (weights_[i]->getWGrad()) {
      std::vector<primitive> convBwdWgt;
      std::shared_ptr<convolution_backward_weights> bwdWgt;
      if (biases_ && biases_->getWGrad()) {
        // bias backward can only execute in filter backward with MKL-DNN
        bwdWgt.reset(new convolution_backward_weights(*bwdWgtPD_,
          *(dataBot_->getIntlMem()), *(diffTop_->getIntlMem()), 
          *(diffWgt_->getIntlMem()), *(diffBias_->getIntlMem())));
        convBwdWgt.push_back(*bwdWgt);
        diffBias_->submitCvt(convBwdWgt);
      } else {
        bwdWgt.reset(new convolution_backward_weights(*bwdWgtPD_,
          *(dataBot_->getIntlMem()), *(diffTop_->getIntlMem()), 
          *(diffWgt_->getIntlMem())));
        convBwdWgt.push_back(*bwdWgt);
      }
      diffWgt_->submitCvt(convBwdWgt);
      stream(stream::kind::eager).submit(convBwdWgt).wait();
  //    LOG(INFO) << "!!!!!!!!!backward filter execute completed!!!!!!!!";
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
