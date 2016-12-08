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
  // Initialize the basic parent class
  DnnLayer::init(layerMap, parameterMap);

  // init for mkldnn, image shapes and some weights
  return initShapeAndDnn(layerMap, parameterMap);

}

bool DnnConvLayer::initShapeAndDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
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
  bs_ = 0;
  oc_ = config_.num_filters();
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

size_t DnnConvLayer::getOneBatchSize() {
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


// reset batchsize and image size of input and output 
void DnnConvLayer::reshapeOutput() {
  // reset image size
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    int height = inputLayers_[i]->getOutput().getFrameHeight();
    int width = inputLayers_[i]->getOutput().getFrameWidth();
    if (height != 0) ih_[i] = height;
    if (width != 0) iw_[i] = width;
    // output image dimensions
    oh_[i] = outputSize(ih_[i], fh_[i], ph_[i], sh_[i]);
    ow_[i] = outputSize(iw_[i], fw_[i], pw_[i], sw_[i]);
    
  }
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);
  
  // reset data
  bs_ = getInput(0).getBatchSize();
  resetOutput(bs_, getOneBatchSize());
  LOG(INFO) << "reshape batch size: " << bs_;
}

void DnnConvLayer::initOrResetDnnFwd() {
  if (bs_ == getInput(0).getBatchSize()) {
    // can remove resetoutput when confirm how multi inputs work and whether to clear diff
    resetOutput(bs_, getOneBatchSize()); 
    return;
  }
  LOG(INFO) << "init or reset conv forward of layer: " << config_.name();
  // need reshape output!
  reshapeOutput();

  // TODO: only care about i==0 by now
  memory::dims biasDims = {oc_};
  bool hasBias = (biases_ && biases_->getW()) ? true : false;
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize())
      << "Assert batchsize of input layers are equal";
    
    // create dim structure that describes user data.
    memory::dims botDims = {bs_, ic_[i], ih_[i], iw_[i]};
    memory::dims wgtDims = (gp_[i] == 1) ? 
        memory::dims{oc_, ic_[i], fh_[i], fw_[i]}
      : memory::dims{gp_[i], oc_/gp_[i], ic_[i]/gp_[i], fh_[i], fw_[i]};
    memory::dims strides = {sh_[i], sw_[i]};
    memory::dims padding = {ph_[i], pw_[i]};
    memory::dims topDims = {bs_, oc_, oh_[i], ow_[i]};
    
    dataBot_.reset(new DnnBuffer(botDims));
    dataWgt_.reset(new DnnBuffer(wgtDims));
    dataTop_.reset(new DnnBuffer(topDims));
    if (hasBias) {
      dataBias_.reset(new DnnBuffer(biasDims));
    }
    // create conv desc from internal desc 
    std::shared_ptr<convolution_forward::desc> fwdDesc;
    if (hasBias) {
      fwdDesc.reset(new convolution_forward::desc(prop_kind::forward_training,
                          algorithm::convolution_direct,
                          dataBot_->getMDAny(),
                          dataWgt_->getMDAny(),
                          dataBias_->getMDAny(),
                          dataTop_->getMDAny(),
                          strides, padding, padding,
                          padding_kind::zero));
    } else {
      fwdDesc.reset(new convolution_forward::desc(prop_kind::forward_training,
                          algorithm::convolution_direct,
                          dataBot_->getMDAny(),
                          dataWgt_->getMDAny(),
                          dataTop_->getMDAny(),
                          strides, padding, padding,
                          padding_kind::zero));
    }
    fwdPD_.reset(new convolution_forward::primitive_desc(*fwdDesc, engineCpu_));
    
    // init user memory and cvt
    real *botData = getPrev(i)->getOutputValue()->getData();
    real *wgtData = weights_[i]->getW()->getData();
    const std::shared_ptr<mkldnn::memory::desc> prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      dataBot_->initUser(botData, *prvMD, engineCpu_);
    } else {
      dataBot_->initUser(botData, botDims, memory::format::nchw, engineCpu_);
    }
    dataWgt_->initUser(wgtData, wgtDims, (gp_[i] == 1) ?
                 memory::format::oihw : memory::format::goihw, engineCpu_);
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
    if (hasBias) {
      real *biasData = biases_->getW()->getData();
      dataBias_->initUser(biasData, dataBias_->getDefaultDims(), memory::format::x, engineCpu_);
      if (dataBias_->initCvt(fwdPD_->bias_primitive_desc(),
        dnnCvtUser2Internal)) {
        LOG(INFO) << "need reorder --- bias data: "
          << DNN_FORMAT[dataBias_->getIntlFmt()]
          << " >>>>> "
          << DNN_FORMAT[dataBias_->getUserFmt()];
      }
    }
    real *topData = getOutputValue()->getData();
    if (setDnnTopDataFmt_) {
      dataTop_->initUser(topData, fwdPD_->dst_primitive_desc());
      setTopDataMD(dataTop_->getUserMD());
    } else {
      dataTop_->initUser(topData, topDims, memory::format::nchw, engineCpu_);
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
    
  }
  printInfo();
  needBwdReset_ = true;
}

void DnnConvLayer::initOrResetDnnBwd() {
  if (!needBwdReset_) {
    return;
  }
  needBwdReset_ = false;
  LOG(INFO) << "init or reset conv backward of layer: " << config_.name();
  bool hasBias = (biases_ && biases_->getWGrad()) ? true : false;
  // TODO: only care about i==0 by now
  real *topdiff = getOutputGrad()->getData();
  // init top diff user
  diffTop_.reset(new DnnBuffer(dataTop_->getDefaultDims()));
  
  const std::shared_ptr<mkldnn::memory::desc> inputDiffMD = getTopDiffMD();
  if (inputDiffMD) {
    diffTop_->initUser(topdiff, *inputDiffMD, engineCpu_);
  } else {
    diffTop_->initUser(topdiff, diffTop_->getDefaultDims(),
      memory::format::nchw, engineCpu_);
  }
  
  if (hasBias) {
    // bias backward can not be execute seperately, 
    //only can execute with filter bakcward
    real* biasdiff = biases_->getWGrad()->getData();
    diffBias_.reset(new DnnBuffer(dataBias_->getDefaultDims()));
    diffBias_->initUser(biasdiff, diffBias_->getDefaultDims(), memory::format::x, engineCpu_);
  }
  
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize())
      << "Assert batchsize of input layers are equal";

    // create dim structure that describes user data.
    memory::dims strides = {sh_[i], sw_[i]};
    memory::dims padding = {ph_[i], pw_[i]};

    /* backward weight and bias before data*****************************/
    if (weights_[i]->getWGrad()) {
      real* wgtdiff = weights_[i]->getWGrad()->getData();
      // init weight diff user
      diffWgt_.reset(new DnnBuffer(dataWgt_->getDefaultDims()));
      diffWgt_->initUser(wgtdiff, diffWgt_->getDefaultDims(), 
        memory::format(dataWgt_->getUserFmt()), engineCpu_);
    } else {
      LOG(FATAL) << "should have weight";
    //  continue;
    }
    std::shared_ptr<convolution_forward::desc> bwdWgtFwdDesc;
    std::shared_ptr<convolution_backward_weights::desc> bwdWgtDesc;
    if (hasBias && diffBias_ != NULL) {
      bwdWgtFwdDesc.reset(new convolution_forward::desc(
        prop_kind::forward_training, algorithm::convolution_direct, 
        dataBot_->getIntlMD(),
        diffWgt_->getMDAny(),
        diffBias_->getMDAny(),
        diffTop_->getMDAny(),
        strides, padding, padding, padding_kind::zero));
      bwdWgtDesc.reset(new convolution_backward_weights::desc(
        algorithm::convolution_direct, 
        dataBot_->getIntlMD(),
        diffWgt_->getMDAny(),
        diffBias_->getMDAny(),
        diffTop_->getMDAny(),
        strides, padding, padding, padding_kind::zero));
    } else {
      bwdWgtFwdDesc.reset(new convolution_forward::desc(
        prop_kind::forward_training, algorithm::convolution_direct, 
        dataBot_->getIntlMD(),
        diffWgt_->getMDAny(),
        diffTop_->getMDAny(),
        strides, padding, padding, padding_kind::zero));
      bwdWgtDesc.reset(new convolution_backward_weights::desc(
        algorithm::convolution_direct,
        dataBot_->getIntlMD(),
        diffWgt_->getMDAny(),
        diffTop_->getMDAny(), 
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
        LOG(INFO) << "need reorder --- bias diff: "
          << DNN_FORMAT[diffBias_->getIntlFmt()]
          << " >>>>> "
          << DNN_FORMAT[diffBias_->getUserFmt()];
      }
    }
    if (diffWgt_->initCvt(bwdWgtPD_->diff_weights_primitive_desc(),
      dnnCvtInternal2User)) {
      LOG(INFO) << "need reorder --- weight diff: "
        << DNN_FORMAT[diffWgt_->getIntlFmt()]
        << " >>>>> "
        << DNN_FORMAT[diffWgt_->getUserFmt()];
    }
    if (diffTop_->initCvt(bwdWgtPD_->diff_dst_primitive_desc(), 
      dnnCvtUser2Internal)) {
      LOG(INFO) << "need reorder --- top diff: "
        << DNN_FORMAT[diffTop_->getUserFmt()]
        << " >>>>> "
        << DNN_FORMAT[diffTop_->getIntlFmt()];
    }
    CHECK(dataBot_->getIntlPD() == bwdWgtPD_->src_primitive_desc());
    
    /* then backward data *************************************/
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      continue; // data layer has not diff
    }
    diffBot_.reset(new DnnBuffer(dataBot_->getDefaultDims()));
    // init backward data primitive desc
    std::shared_ptr<convolution_forward::desc> bwdDataFwdDesc;
    std::shared_ptr<convolution_backward_data::desc> bwdDataDesc;
    bwdDataFwdDesc.reset(new convolution_forward::desc(
      prop_kind::forward_training, algorithm::convolution_direct, 
      diffBot_->getMDAny(),
      dataWgt_->getIntlMD(), 
      diffTop_->getIntlMD(),
      strides, padding, padding, padding_kind::zero));
    bwdDataDesc.reset(new convolution_backward_data::desc(
      algorithm::convolution_direct,
      diffBot_->getMDAny(),
      dataWgt_->getIntlMD(), 
      diffTop_->getIntlMD(),
      strides, padding, padding, padding_kind::zero));
    std::shared_ptr<convolution_forward::primitive_desc> bwdDataFwdPD;
    bwdDataFwdPD.reset(new convolution_forward::primitive_desc(*bwdDataFwdDesc, engineCpu_));
    bwdDataPD_.reset(new convolution_backward_data::primitive_desc(*bwdDataDesc,
                        engineCpu_, *bwdDataFwdPD));
    // init user memory and cvt
    real* botdiff = prevLayer->getOutputGrad()->getData();
    if (setDnnBotDiffFmt_[i]) {
      diffBot_->initUser(botdiff, bwdDataPD_->diff_src_primitive_desc());
      getPrev(i)->setTopDiffMD(diffBot_->getUserMD());
    } else {
      diffBot_->initUser(botdiff, diffBot_->getDefaultDims(), memory::format::nchw, engineCpu_);
    }
    if (diffBot_->initCvt(bwdDataPD_->diff_src_primitive_desc(), 
      dnnCvtInternal2User)) {
      LOG(INFO) << "need reorder --- bottom diff: "
        << DNN_FORMAT[diffBot_->getIntlFmt()]
        << " >>>>> "
        << DNN_FORMAT[diffBot_->getUserFmt()];
    }
    CHECK(dataWgt_->getIntlPD() == bwdDataPD_->weights_primitive_desc());
    CHECK(diffTop_->getIntlPD() == bwdDataPD_->diff_dst_primitive_desc());
    LOG(INFO) << "diff format flow --- "
      << DNN_FORMAT[diffBot_->getUserFmt()] << " <<< ("
      << DNN_FORMAT[diffBot_->getIntlFmt()] << " <<< "
      << DNN_FORMAT[diffTop_->getIntlFmt()] << ") <<< "
      << DNN_FORMAT[diffTop_->getUserFmt()];

  }

}

void DnnConvLayer::submitFwd(
  int inputIdx, const MatrixPtr& botVal, const MatrixPtr& topVal) {
  real* botdata = botVal->getData();
  real* topdata = topVal->getData();
  real* wgtdata = weights_[inputIdx]->getW()->getData();
  std::vector<primitive> convFwd;
  dataBot_->submitCvt(convFwd, botdata);
  dataWgt_->submitCvt(convFwd, wgtdata);
  if(biases_ && biases_->getW()) {
    // only for shared bias
    // TODO: enable unshared bias
    real* biasdata = biases_->getW()->getData();
    dataBias_->submitCvt(convFwd, biasdata);
    convFwd.push_back(convolution_forward(*fwdPD_,
                *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
                *(dataBias_->getIntlMem()), *(dataTop_->getIntlMem())));
  } else {
    convFwd.push_back(convolution_forward(*fwdPD_, *(dataBot_->getIntlMem()),
                *(dataWgt_->getIntlMem()),
                *(dataTop_->getIntlMem())));
  }
  
  dataTop_->submitCvt(convFwd, topdata);

  // start forward
  REGISTER_TIMER_INFO("dnnFwd", getName().c_str());
  stream(stream::kind::eager).submit(convFwd).wait();
}

void DnnConvLayer::submitBwdData(
  int inputIdx, const MatrixPtr& topGrad, const MatrixPtr& botGrad) {
  if (botGrad == NULL) {
    return;
  }
  real* topdiff = topGrad->getData();
  real* wgtdata = weights_[inputIdx]->getW()->getData();
  real* botdiff = botGrad->getData();
  std::vector<primitive> convBwdData;
  dataWgt_->submitCvt(convBwdData, wgtdata);
  diffTop_->submitCvt(convBwdData, topdiff);
  auto bwdData = convolution_backward_data(
    *bwdDataPD_, *(diffTop_->getIntlMem()),
    *(dataWgt_->getIntlMem()), *(diffBot_->getIntlMem()));
  convBwdData.push_back(bwdData);
  diffBot_->submitCvt(convBwdData, botdiff);
  stream(stream::kind::eager).submit(convBwdData).wait();
//   LOG(INFO) << "!!!!!!!!!backward data execute completed!!!!!!!!";
}

void DnnConvLayer::submitBwdWgts(
  int inputIdx, const MatrixPtr& botVal, const MatrixPtr& topGrad) {
  real* botdata = botVal->getData();
  real* topdiff = topGrad->getData();
  real* wgtdiff = weights_[inputIdx]->getWGrad()->getData();
  std::vector<primitive> convBwdWgt;
  std::shared_ptr<convolution_backward_weights> bwdWgt;
  dataBot_->submitCvt(convBwdWgt, botdata);
  diffTop_->submitCvt(convBwdWgt, topdiff);
  if (biases_ && biases_->getWGrad()) {
    // bias backward can only execute in filter backward with MKL-DNN
    real* biasdiff = biases_->getWGrad()->getData();
    bwdWgt.reset(new convolution_backward_weights(*bwdWgtPD_,
      *(dataBot_->getIntlMem()), *(diffTop_->getIntlMem()), 
      *(diffWgt_->getIntlMem()), *(diffBias_->getIntlMem())));
    convBwdWgt.push_back(*bwdWgt);
    diffBias_->submitCvt(convBwdWgt, biasdiff);
  } else {
    bwdWgt.reset(new convolution_backward_weights(*bwdWgtPD_,
      *(dataBot_->getIntlMem()), *(diffTop_->getIntlMem()), 
      *(diffWgt_->getIntlMem())));
    convBwdWgt.push_back(*bwdWgt);
  }
  diffWgt_->submitCvt(convBwdWgt, wgtdiff);
  stream(stream::kind::eager).submit(convBwdWgt).wait();
//    LOG(INFO) << "!!!!!!!!!backward filter execute completed!!!!!!!!";
}

void DnnConvLayer::forward(PassType passType) {
  Layer::forward(passType);

  /// For dnn fwd init or reset
  initOrResetDnnFwd();

  clearAllCvtFlags();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    submitFwd(i, getPrev(i)->getOutputValue(), getOutputValue());
  }
//  LOG(INFO) << "!!!!!!!!Forward completed!!!!!!!";
  /* activation */
  forwardActivation();
}

void DnnConvLayer::backward(const UpdateCallback &callback) {
  backwardActivation();
  
  initOrResetDnnBwd();

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

}


}  // namespace paddle
