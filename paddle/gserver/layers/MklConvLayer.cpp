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
#include "MklConvLayer.h"

#ifndef CHECK_ST
#define CHECK_ST(f, st) do { \
    (st) = (f); \
    if ((st) != E_SUCCESS) { \
        LOG(FATAL) <<"[" << __FILE__ \
          << "," << __LINE__ << "] " \
          << "err code: " << st; \
    } \
  }while (0)
#endif

namespace paddle {

REGISTER_LAYER(mklconv, MklConvLayer);

bool MklConvLayer::init(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ConvBaseLayer::init(layerMap, parameterMap);

  // for dnn init
  convFwd_ = NULL;
  convBwdBias_ = NULL;
  convBwdData_ = NULL;
  convBwdFilter_ = NULL;

  
  /* Initialize the projection */
  for (auto &inputConfig : config_.inputs()) {
    const ConvConfig &conf = inputConfig.conv_conf();
    subM_.push_back(numFilters_ / conf.groups());
    subN_.push_back(conf.output_x() * conf.output_x());
    subK_.push_back(conf.channels() * conf.filter_size() * conf.filter_size() /
                    conf.groups());

    /* Consistent caffe mode for multiple input */
    caffeMode_ = conf.caffe_mode();
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

size_t MklConvLayer::getSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  imgSizeH_.clear();
  imgSizeW_.clear();
  outputH_.clear();
  outputW_.clear();
  subN_.clear();
  size_t layerSize = 0;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    imgSizeH_.push_back(inputLayers_[i]->getOutput().getFrameHeight());
    imgSizeW_.push_back(inputLayers_[i]->getOutput().getFrameWidth());
    if (imgSizeH_[i] == 0) imgSizeH_[i] = imgSize_[i];
    if (imgSizeW_[i] == 0) imgSizeW_[i] = imgSize_[i];
    outputH_.push_back(
        outputSize(imgSizeH_[i], filterSize_[i], padding_[i], stride_[i]));
    outputW_.push_back(
        outputSize(imgSizeW_[i], filterSize_[i], padding_[i], stride_[i]));
    subN_.push_back(outputH_[i] * outputW_[i]);
    CHECK(layerSize == 0 || subN_[i] * size_t(numFilters_) == layerSize);
    layerSize = subN_[i] * numFilters_;
  }
  getOutput().setFrameHeight(outputH_[0]);
  getOutput().setFrameWidth(outputW_[0]);
  return layerSize;
}

void MklConvLayer::initDnn() {
  int st;
  int i = 0;
  // TODO: should init all layers, here only i==0
  LOG(INFO) << "init conv forward of layer: " << config_.name();
  size_t bs, gp;
  size_t iw, ih, ic;
  size_t ow, oh, oc;
  size_t fw, fh;  // filter size
  size_t dimension = 4;
  // TODO: only consider group:1 and only one input layer by now

  gp  = std::max(groups_[i], 1);
  bs = getInput(i).getBatchSize();
  // input image dimensions
  imgSizeH_.clear();
  imgSizeW_.clear();
  outputH_.clear();
  outputW_.clear();
  // TODO: this maybe can move to init
  imgSizeH_.push_back(inputLayers_[i]->getOutput().getFrameHeight());
  imgSizeW_.push_back(inputLayers_[i]->getOutput().getFrameWidth());
//  LOG(INFO) << "wd: " << imgSizeW_[0] << ",ht: " << imgSizeH_[0];
  if (imgSizeH_[i] == 0) imgSizeH_[i] = imgSize_[i];
  if (imgSizeW_[i] == 0) imgSizeW_[i] = imgSize_[i];
  iw = imgSizeW_[i];
  ih = imgSizeH_[i];
  ic = channels_[i];
  // output image dimensions
  outputH_.push_back(
    outputSize(imgSizeH_[i], filterSize_[i], padding_[i], stride_[i]));
  outputW_.push_back(
      outputSize(imgSizeW_[i], filterSize_[i], padding_[i], stride_[i]));
  ow = outputW_[i];
  oh = outputH_[i];
  oc = numFilters_;  // TODO: need to check about with group

  // init input image layout
  size_t iSize[4] = {iw, ih, ic, bs};
  size_t iStrides[4] = {1, iw, iw*ih, iw*ih*ic};
  // init output image layout
  size_t oSize[4] = {ow, oh, oc, bs};
  size_t oStrides[4] = {1, ow, ow*oh, ow*oh*oc};

  // filter size
  // filter size
  fw = filterSize_[i];
  fh = filterSizeY_[i];
  size_t fSize[4] = {fw, fh, ic/gp, oc/gp};  // TODO: need check with intelcaffe group
  size_t fStrides[4] = {1, fw, fw*fh, fw*fh*ic/gp};
  // bias
  size_t bSize[1] = {oSize[2]};
  size_t bStrides[1] = {1};  // {oStrides[2]};
  // stride and padding
  size_t sw = stride_[i];
  size_t sh = strideY_[i];
  size_t convStride[2] = {sw, sh};
  int inputOffset[2] = {-padding_[i], -paddingY_[i]};
  // TODO:  and with group
  // TODO: enable without bias,  if (hasBias_)
  LOG(INFO) << "has bias: " << hasBias_;
  st = dnnConvolutionCreateForwardBias(
    &convFwd_, NULL, dnnAlgorithmConvolutionDirect,
    dimension, iSize, oSize, fSize, convStride,
    inputOffset, dnnBorderZeros);
  CHECK_EQ(st, E_SUCCESS)
        << "Failed dnnCreateConvolution<Dtype>(dnnForward) with status "
        << st;
  CHECK_ST(dataBottom_->createUser(dimension, iSize, iStrides), st);
  CHECK_ST(dataBottom_->createIntl(convFwd_, dnnResourceSrc), st);
  CHECK_ST(dataBottom_->initConversion(dnnCvtUser2Internal), st);

  CHECK_ST(dataFilter_->createUser(dimension, fSize, fStrides), st);
  CHECK_ST(dataFilter_->createIntl(convFwd_, dnnResourceFilter), st);
  CHECK_ST(dataFilter_->initConversion(dnnCvtUser2Internal), st);
  
  CHECK_ST(dataBias_->createUser(1, bSize, bStrides), st);
  CHECK_ST(dataBias_->createIntl(convFwd_, dnnResourceBias), st);
  CHECK_ST(dataBias_->initConversion(dnnCvtUser2Internal), st);
  
  CHECK_ST(dataTop_->createUser(dimension, oSize, oStrides), st);
  CHECK_ST(dataTop_->createIntl(convFwd_, dnnResourceDst), st);
  CHECK_ST(dataTop_->initConversion(dnnCvtInternal2User), st);

  /// backward
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
}

void MklConvLayer::forward(PassType passType) {
  Layer::forward(passType);

//  LOG(INFO) << "test...............dnn";
//  LOG(INFO) << "input layer size:" << inputLayers_.size();

  int st;
  int i = 0;
  
  /// For dnn init
  /* since can not get batchsize at layer.init(),
   * so init dnn at forward by now*/
  if (!convFwd_) {
    initDnn();
  }
  // get prv image data
  real *imgData = getPrev(i)->getOutputValue()->getData();
  real *filterData = weights_[i]->getW()->getData(); // TODO: check index
  real *biasData = biases_->getW()->getData();

  // 
  if (dataBottom_->needConversion()) {
    CHECK_ST(dataBottom_->executeConversion(imgData), st);
    resConv_[dnnResourceSrc] = dataBottom_->getData();
  } else {
    resConv_[dnnResourceSrc] = (void *)imgData;
  }
  // 
  if (dataFilter_->needConversion()) {
    CHECK_ST(dataFilter_->executeConversion(filterData), st);
    resConv_[dnnResourceFilter] = dataFilter_->getData();
  } else {
    resConv_[dnnResourceFilter] = (void *)filterData;
  }
  // 
  if (dataBias_->needConversion()) {
    CHECK_ST(dataBias_->executeConversion(filterData), st);
    resConv_[dnnResourceBias] = dataBias_->getData();
  } else {
    resConv_[dnnResourceBias] = (void *)biasData;
  }
  // execute
  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  resetOutput(batchSize, getSize());
  real *outData = getOutputValue()->getData();
  if (dataTop_->needConversion()) {
    resConv_[dnnResourceDst] = dataTop_->getData();
    CHECK_ST(dnnExecute(convFwd_, resConv_), st);
    CHECK_ST(dataTop_->executeConversion(outData), st);
  } else {
    resConv_[dnnResourceDst] = (void *)outData;
    CHECK_ST(dnnExecute(convFwd_, resConv_), st);
  }
  LOG(INFO) << "!!!!!!!!Forward completed!!!!!!!";

  /* activation */
  forwardActivation();
}

void MklConvLayer::backward(const UpdateCallback &callback) {
  backwardActivation();
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

    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    /* First, calculate the input layers error */
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

      
      /* Then, calculate the W-gradient for the current layer */
      //bpropWeights(topGrad, i);
      /* Increasing the number of gradient */
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

}  // namespace paddle
