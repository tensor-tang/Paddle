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

void DnnConvLayer::initDnn() {

  // TODO: only care about i==0 by now
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
    memory::dims biasDims = {oc_};
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
                          dataBot_->getIntlMD(),
                          dataWgt_->getIntlMD(),
                          dataBias_->getIntlMD(),
                          dataTop_->getIntlMD(),
                          strides, padding, padding, // TODO: check here left and right
                          padding_kind::zero));
    } else {
      fwdDesc.reset(new convolution_forward::desc(prop_kind::forward_training,
                          algorithm::convolution_direct,
                          dataBot_->getIntlMD(),
                          dataWgt_->getIntlMD(),
                          dataTop_->getIntlMD(),
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
/*
  /// backward
  real *topdiff = getOutputGrad()->getData(); // here is the top diff
  if (hasBias_&& biases_->getWGrad()) {
    real* biasdiff = biases_->getWGrad()->getData();
  }

  // init user memory
  diffBot_->initUser(botData, botDims, memory::format::nchw, engineCpu_);
  diffWgt_->initUser(wgtData, wgtDims, 
  (gp == 1) ? memory::format::oihw : memory::format::goihw, engineCpu_);
  diffBias_->initUser(biasData, biasDims, memory::format::x, engineCpu_);
  diffTop_->initUser(topdiff, topDims, memory::format::nchw, engineCpu_);
  st = dnnConvolutionCreateBackwardData(
    &convBwdData_, NULL, dnnAlgorithmConvolutionDirect,
    dimension,  iSize,  oSize,  fSize,  convStride,
    inputOffset, dnnBorderZeros);

  CHECK_EQ(st, E_SUCCESS)
          << "Failed dnnCreateConvolution Backward with status "
          << st;
  st = dnnConvolutionCreateBackwardFilter(
    &convBwdFilter_, NULL, dnnAlgorithmConvolutionDirect,
    dimension,  iSize,  oSize,  fSize,  convStride,
    inputOffset, dnnBorderZeros);

  CHECK_EQ(st, E_SUCCESS)
          << "Failed dnnCreateConvolution<Dtype>(dnnfilter) with status "
          << st;
  
  /// if has bias
  st = dnnConvolutionCreateBackwardBias(
    &convBwdBias_, NULL, dnnAlgorithmConvolutionDirect,
    dimension,  oSize);

  CHECK_EQ(st, E_SUCCESS)
          << "Failed dnnCreateConvolution<Dtype>(dnnbias) with status "
          << st;
  CHECK_ST(diffTop_->createUser(dimension, oSize, oStrides), st);
  CHECK_ST(diffTop_->createIntl(convBwdData_, dnnResourceDiffDst), st); //careful: top==dst
  CHECK_ST(diffTop_->initConversion(dnnCvtUser2Internal), st);

  CHECK_ST(diffFilter_->createUser(dimension, fSize, fStrides), st);
  CHECK_ST(diffFilter_->createIntl(convBwdFilter_, dnnResourceDiffFilter), st);
  CHECK_ST(diffFilter_->initConversion(dnnCvtInternal2User), st);

  CHECK_ST(diffBias_->createUser(1, bSize, bStrides), st);
  CHECK_ST(diffBias_->createIntl(convBwdBias_, dnnResourceDiffBias), st);
  CHECK_ST(diffBias_->initConversion(dnnCvtInternal2User), st);

  CHECK_ST(diffBottom_->createUser(dimension, iSize, iStrides), st);
  CHECK_ST(diffBottom_->createIntl(convBwdData_, dnnResourceDiffSrc), st);
  CHECK_ST(diffBottom_->initConversion(dnnCvtInternal2User), st);
  */
  
  }
  printInfo();
}

void DnnConvLayer::forward(PassType passType) {
  Layer::forward(passType);

  /// For dnn init
  /* since can not get batchsize at layer.init(),
   * so init dnn at forward by now*/

  if (!dnnFwdInited_) {
    initDnn();
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
/*
int st;
  MatrixPtr topGrad = getOutputGrad(); // here is the top diff
  real *topdiff = topGrad->getData();

  // 
  if (diffTop_->needConversion()) {
  //  LOG(INFO) << "convert user top to internal";
    CHECK_ST(diffTop_->executeConversion(topdiff), st);
    resConv_[dnnResourceDiffDst] = diffTop_->getData();
  } else {
    resConv_[dnnResourceDiffDst] = (void *)topdiff;
  }
  
  if (biases_ && biases_->getWGrad()) {   
    // TODO: check data for bias backward
    real* biasdiff = biases_->getWGrad()->getData();
    if (diffBias_->needConversion()) {
      resConv_[dnnResourceDiffBias] = diffBias_->getData();
      CHECK_ST(dnnExecute(convBwdBias_, resConv_), st);
      CHECK_ST(diffBias_->executeConversion(biasdiff), st);
    } else {
      resConv_[dnnResourceDiffBias] = (void *)biasdiff;
      CHECK_ST(dnnExecute(convBwdBias_, resConv_), st);
    }
    LOG(INFO) << "!!!!!!!!!backward bias execute completed!!!!!!!!";

  //  bpropBiases(topGrad);

    // Increasing the number of gradient 
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    // First, calculate the input layers error 
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      return;
    }
    real *filterData = weights_[i]->getW()->getData(); // TODO: check index
    if (dataFilter_->needConversion()) {
      resConv_[dnnResourceFilter] = dataFilter_->getData();
    } else {
      resConv_[dnnResourceFilter] = (void *)filterData;
    }
    real* bottomdiff = prevLayer->getOutputGrad()->getData();
    // TODO: check top and bottom for data backward
    if (diffBottom_->needConversion()) {
      resConv_[dnnResourceDiffSrc] = diffBottom_->getData();
      CHECK_ST(dnnExecute(convBwdData_, resConv_), st);
      CHECK_ST(diffBottom_->executeConversion(bottomdiff), st);
    } else {
      resConv_[dnnResourceDiffDst] = (void *)bottomdiff;
      CHECK_ST(dnnExecute(convBwdData_, resConv_), st);
    }

    LOG(INFO) << "!!!!!!!!!backward data execute completed!!!!!!!!";

    //bpropActs(topGrad, i);
    
    if (weights_[i]->getWGrad()) {
      real *imgData = getPrev(i)->getOutputValue()->getData();
      if (dataBottom_->needConversion()) {
        resConv_[dnnResourceSrc] = dataBottom_->getData();
      } else {
        resConv_[dnnResourceSrc] = (void *)imgData;
      }
      // TODO: check data for filter backward
      real* filterdiff = weights_[i]->getWGrad()->getData();
      if (diffFilter_->needConversion()) {
        resConv_[dnnResourceDiffFilter] = diffFilter_->getData();
        CHECK_ST(dnnExecute(convBwdFilter_, resConv_), st);
        CHECK_ST(diffFilter_->executeConversion(filterdiff), st);
      } else {
        resConv_[dnnResourceDiffFilter] = (void *)filterdiff;
        CHECK_ST(dnnExecute(convBwdFilter_, resConv_), st);
      }
      LOG(INFO) << "!!!!!!!!!backward filter execute completed!!!!!!!!";

      
      // Then, calculate the W-gradient for the current layer 
      //bpropWeights(topGrad, i);
      // Increasing the number of gradient 
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
  */
}

}  // namespace paddle
