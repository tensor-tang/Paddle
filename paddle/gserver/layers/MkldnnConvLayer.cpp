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
#include "MkldnnConvLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_conv, MkldnnConvLayer);

bool MkldnnConvLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  // mkldnn only support float type by now
  bool sharedBiases = config_.shared_biases();
  if (biasParameter_.get() != NULL && !sharedBiases) {
    LOG(FATAL) << "Only support shared bias with MKL DNN yet!";
  }
  LOG(INFO) << "input layer size:" << inputLayers_.size();
  if (inputLayers_.size() != 1) {
     // TODO(TJ): considerate more than One input
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

  // TODO(TJ): should get this flag from layer proto , default true
  usePaddleFmt_ = true;

  /* initialize the weightList */
  CHECK(inputLayers_.size() == parameters_.size());
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    size_t height, width;
    selfWgtData_.push_back(nullptr);
    selfWgtDiff_.push_back(nullptr);
    if (usePaddleFmt_) {
      height = ic_[i] * fh_[i] * fw_[i] / gp_[i];
      width = oc_;
      selfWgtData_[i] = Matrix::create(width, height, false, false);
      selfWgtDiff_[i] = Matrix::create(width, height, false, false);
      selfWgtData_[i]->zeroMem();
      selfWgtDiff_[i]->zeroMem();
    } else {  // TODO(TJ): never tested this case
      height = oc_;
      width = ic_[i] * fh_[i] * fw_[i] / gp_[i];
    }
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
  hasRelu_ = hasMkldnnRelu();
  if (hasRelu_) {
    // TODO(TJ): get from proto setting
    negativeSlope_ = -0.0;
  }
  return true;
}

void MkldnnConvLayer::clearDataDiff() {
  reserveOutput(bs_, getSize());
}

void MkldnnConvLayer::reshape() {
  // reshape input and output size
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = 0;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    // all output size should be the same
    CHECK(layerSize == 0 || oh_[i] * ow_[i] * size_t(oc_) == layerSize);
    layerSize = oh_[i] * ow_[i] * oc_;
    int height = inputLayers_[i]->getOutput().getFrameHeight();
    int width = inputLayers_[i]->getOutput().getFrameWidth();
    if (height != 0) ih_[i] = height;
    if (width != 0) iw_[i] = width;
    // output image dimensions
    oh_[i] = outputSize(ih_[i], fh_[i], ph_[i], sh_[i]);
    ow_[i] = outputSize(iw_[i], fw_[i], pw_[i], sw_[i]);
  }
  // reset output image size
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);
  printInfo();
}

void MkldnnConvLayer::resetDnn(PassType passType) {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  algorithm algo = algorithm::convolution_direct;
  memory::dims topDims = {bs_, oc_, oh_[0], ow_[0]};
  memory::dims biasDims = {oc_};
  memory::format fmt = memory::format::nchw;
  memory::format fmtx = memory::format::x;
  bool hasBias = (biases_ && biases_->getW());
  prop_kind pk = passType == PASS_TEST ?
    prop_kind::forward_scoring : prop_kind::forward_training;
  // conv_relu only support scoring yet
  useConvRelu_ = (hasRelu_ && passType == PASS_TEST);

  // create buffer, only have one output and bias buffer
  dataTop_.reset(new MkldnnBuffer());
  if (hasBias) {
    dataBias_.reset(new MkldnnBuffer());
    real *biasData = biases_->getW()->getData();
    dataBias_->initUser(biasData, biasDims, fmtx, eg);
  }
  if (passType == PASS_TRAIN) {
    diffTop_.reset(new MkldnnBuffer());
    if (hasBias && biases_->getWGrad()) {
      diffBias_.reset(new MkldnnBuffer());
      real* biasDiff = biases_->getWGrad()->getData();
      diffBias_->initUser(biasDiff, biasDims, fmtx, eg);
    }
    // prepare top diff if use dnn input
    real *topDiff = getOutputGrad()->getData();
    const std::shared_ptr<mkldnn::memory::desc> inputDiffMD = getTopDiffMD();
    if (inputDiffMD) {
      diffTop_->initUser(topDiff, *inputDiffMD, eg);
    } else {
      diffTop_->initUser(topDiff, topDims, fmt, eg);
    }
  }
  // TODO(TJ): only care about i==0 yet
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize()) << "batchsize should equal";
    // create buffer, could be vector later
    dataBot_.reset(new MkldnnBuffer());
    dataWgt_.reset(new MkldnnBuffer());
    
    // init dim structure that describes user data.
    memory::dims botDims = {bs_, ic_[i], ih_[i], iw_[i]};
    memory::dims wgtDims = (gp_[i] == 1) ?
      memory::dims{oc_, ic_[i], fh_[i], fw_[i]}
      : memory::dims{gp_[i], oc_/gp_[i], ic_[i]/gp_[i], fh_[i], fw_[i]};
    memory::dims strides = {sh_[i], sw_[i]};
    memory::dims padding = {ph_[i], pw_[i]};
    memory::format wgtFmt= (gp_[i] == 1) ?
      memory::format::oihw : memory::format::goihw;
    std::vector<int> padR = {ph_[i], pw_[i]};
    for (int k = 0; k < 2; ++k) {
      if ((ih_[i] + ph_[i] + padR[0] - fh_[i])/sh_[i] + 1 != oh_[i]) ++padR[0];
      if ((iw_[i] + pw_[i] + padR[1] - fw_[i])/sw_[i] + 1 != ow_[i]) ++padR[1];
    }
    real *botData = getPrev(i)->getOutputValue()->getData();
    real *topData = getOutputValue()->getData();
    
    // prepare bottom data if use prv input
    const std::shared_ptr<memory::desc> prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      dataBot_->initUser(botData, *prvMD, eg);
      LOG(INFO) << "use prev format: " << DNN_FORMAT[dataBot_->getUserFmt()];
    } else {
      dataBot_->initUser(botData, botDims, fmt, eg);
    }
    
    /// init mkldnn forward ****************************************************
    // create conv fwd
    std::shared_ptr<convolution_forward::desc> fwdDesc;
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> fwdPD;
    if (hasBias) {
      fwdDesc.reset(new convolution_forward::desc(pk,
                    algorithm::convolution_direct,
                    prvMD ? dataBot_->getUserMD() : getAnyMD(botDims),
                    getAnyMD(wgtDims),
                    getAnyMD(biasDims),
                    getAnyMD(topDims),
                    strides, padding, padR, padding_kind::zero));
    } else {
      fwdDesc.reset(new convolution_forward::desc(pk,
                    algorithm::convolution_direct,
                    prvMD ? dataBot_->getUserMD() : getAnyMD(botDims),
                    getAnyMD(wgtDims),
                    getAnyMD(topDims),
                    strides, padding, padR, padding_kind::zero));
    }
    fwdPD.reset(new convolution_forward::primitive_desc(*fwdDesc, eg));

    // create conv_relu fwd only in scoring
    std::shared_ptr<convolution_relu_forward::primitive_desc> convReluPD;
    if (useConvRelu_) {
      std::shared_ptr<convolution_relu_forward::desc> convReluDesc;
      convReluDesc.reset(
        new convolution_relu_forward::desc(*fwdDesc, negativeSlope_));
      convReluPD.reset(
        new convolution_relu_forward::primitive_desc(*convReluDesc, eg));
    }

    // init mkldnn buffer and conversion
    dataBot_->initIntlCvt(fwdPD->src_primitive_desc(), dnnCvtUser2Intl);
    // set output if next is mkldnn
    // TODO(TJ): need check only init once when have multi inputs layers
    if (setDnnTopDataFmt_) {
      dataTop_->initUser(topData, fwdPD->dst_primitive_desc());
      setTopDataMD(dataTop_->getUserMD());
      LOG(INFO) << "set next format: " << DNN_FORMAT[dataTop_->getUserFmt()];
    } else {
      dataTop_->initUser(topData, topDims, fmt, eg);
    }
    dataTop_->initIntlCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);
    if (usePaddleFmt_) {
      // TODO(TJ): never tested g!=1
      weights_[i]->getW()->transpose(selfWgtData_[i], false);
      real *wgtData = selfWgtData_[i]->getData();
      dataWgt_->initUser(wgtData, wgtDims, wgtFmt, eg);
      if (dataWgt_->initIntlCvt(
        fwdPD->weights_primitive_desc(), dnnCvtUser2Intl)) {
        LOG(INFO) << "need reorder --- weight data: "
          << DNN_FORMAT[dataWgt_->getUserFmt()]
          << " >>> "
          << DNN_FORMAT[dataWgt_->getIntlFmt()];
      }
      if (passType == PASS_TEST) {
        std::vector<primitive> cvtWgt;
        dataWgt_->submitCvt(cvtWgt, wgtData);
        stream(stream::kind::eager).submit(cvtWgt).wait();
      }
    } else {
      // TODO(TJ): initial wgt data with input format
      real *wgtData = weights_[i]->getW()->getData();
      dataWgt_->initUser(wgtData, fwdPD->weights_primitive_desc());
      dataWgt_->initIntlCvt(dataWgt_->getUserPD(), dnnCvtNoNeed);
    }
    // create fwd handle
    if (hasBias) {
      if (dataBias_->initIntlCvt(
        fwdPD->bias_primitive_desc(), dnnCvtUser2Intl)) {
        LOG(INFO) << "need reorder --- bias data: "
          << DNN_FORMAT[dataBias_->getUserFmt()]
          << " >>>>> "
          << DNN_FORMAT[dataBias_->getIntlFmt()];
      }
      if (useConvRelu_) {
        fwd_.reset(new convolution_relu_forward(*convReluPD,
              *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
              *(dataBias_->getIntlMem()), *(dataTop_->getIntlMem())));
      } else {
        fwd_.reset(new convolution_forward(*fwdPD,
              *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
              *(dataBias_->getIntlMem()), *(dataTop_->getIntlMem())));
      }
    } else {
      if (useConvRelu_) {
        fwd_.reset(new convolution_relu_forward(*convReluPD,
              *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
              *(dataTop_->getIntlMem())));
      } else {
        fwd_.reset(new convolution_forward(*fwdPD,
              *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
              *(dataTop_->getIntlMem())));
      }
    }
    LOG(INFO) << "data format flow --- "
      << DNN_FORMAT[dataBot_->getUserFmt()] << " >>> ("
      << DNN_FORMAT[dataBot_->getIntlFmt()] << " >>> "
      << DNN_FORMAT[dataTop_->getIntlFmt()] << ") >>> "
      << DNN_FORMAT[dataTop_->getUserFmt()];

    /// init mkldnn backward ***************************************************
    if (passType != PASS_TRAIN)
      continue;
    hasBias = (hasBias && biases_->getWGrad());
    // create buffer, could be vector later
    diffBot_.reset(new MkldnnBuffer());
    diffWgt_.reset(new MkldnnBuffer());
    // prepare backward weight and bias ----------------------------------------
    // bias backward can only execute with weight bakcward
    // unable execute seperately
    std::shared_ptr<convolution_backward_weights::desc> bwdWgtDesc;
    std::shared_ptr<convolution_backward_weights::primitive_desc> bwdWgtPD;
    if (hasBias) {
      // TODO(TJ): only do once bwd bias with multi inputs layers
      bwdWgtDesc.reset(new convolution_backward_weights::desc(
        algo,
        dataBot_->getIntlMD(),
        dataWgt_->getIntlMD(),
        dataBias_->getIntlMD(),
        dataTop_->getIntlMD(),
        strides, padding, padR, padding_kind::zero));
    } else {
      bwdWgtDesc.reset(new convolution_backward_weights::desc(
        algo,
        dataBot_->getIntlMD(),
        dataWgt_->getIntlMD(),
        dataTop_->getIntlMD(),
        strides, padding, padR, padding_kind::zero));
    }
    bwdWgtPD.reset(new convolution_backward_weights::primitive_desc(
      *bwdWgtDesc, eg, *fwdPD));
    CHECK(dataBot_->getIntlPD() == bwdWgtPD->src_primitive_desc());
    CHECK(dataWgt_->getIntlPD() == bwdWgtPD->diff_weights_primitive_desc());
    CHECK(dataBias_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc());
    CHECK(dataTop_->getIntlPD() == bwdWgtPD->diff_dst_primitive_desc());
    CHECK(weights_[i]->getWGrad()) << "should have weight";
    // init mkldnn buffer and conversion
    // TODO(TJ): need check only initIntlCvt once when multi inputs layers
    diffTop_->initIntlCvt(bwdWgtPD->diff_dst_primitive_desc(), dnnCvtUser2Intl);
    if (usePaddleFmt_) {
      real *wgtDiff = selfWgtDiff_[i]->getData();
      diffWgt_->initUser(wgtDiff, wgtDims, wgtFmt, eg);
      if (diffWgt_->initIntlCvt(dataWgt_->getIntlPD(), dnnCvtIntl2User)) {
        LOG(INFO) << "need reorder --- weight diff: "
          << DNN_FORMAT[diffWgt_->getUserFmt()]
          << " <<< "
          << DNN_FORMAT[diffWgt_->getIntlFmt()];
      }
    } else {
      real *wgtDiff = weights_[i]->getWGrad()->getData();
      diffWgt_->initUser(wgtDiff, dataWgt_->getIntlPD());
      diffWgt_->initIntlCvt(diffWgt_->getUserPD(), dnnCvtNoNeed);
    }
    if (hasBias) {
      // bias backward can only execute in weight backward within MKL-DNN
      if (diffBias_->initIntlCvt(dataBias_->getIntlPD(), dnnCvtIntl2User)) {
        LOG(INFO) << "need reorder --- bias diff: "
          << DNN_FORMAT[diffBias_->getUserFmt()]
          << " <<< "
          << DNN_FORMAT[diffBias_->getIntlFmt()];
      }
      bwdWgt_.reset(new convolution_backward_weights(*bwdWgtPD,
        *(dataBot_->getIntlMem()), *(diffTop_->getIntlMem()),
        *(diffWgt_->getIntlMem()), *(diffBias_->getIntlMem())));
    } else {
      bwdWgt_.reset(new convolution_backward_weights(*bwdWgtPD,
        *(dataBot_->getIntlMem()), *(diffTop_->getIntlMem()),
        *(diffWgt_->getIntlMem())));
    }
    // then prepare backward data ----------------------------------------------
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      continue; // data layer has not diff
    }
    // init backward data primitive desc
    std::shared_ptr<convolution_forward::desc> bwdDataFwdDesc;
    std::shared_ptr<convolution_forward::primitive_desc> bwdDataFwdPD;
    std::shared_ptr<convolution_backward_data::desc> bwdDataDesc;
    std::shared_ptr<convolution_backward_data::primitive_desc> bwdDataPD;
    bwdDataFwdDesc.reset(new convolution_forward::desc(
      pk, algo,
      dataBot_->getIntlMD(),
      dataWgt_->getIntlMD(),
      diffTop_->getIntlMD(),
      strides, padding, padR, padding_kind::zero));
    bwdDataDesc.reset(new convolution_backward_data::desc(
      algo,
      dataBot_->getIntlMD(),
      dataWgt_->getIntlMD(), 
      diffTop_->getIntlMD(),
      strides, padding, padR, padding_kind::zero));
    bwdDataFwdPD.reset(new convolution_forward::primitive_desc(
      *bwdDataFwdDesc, eg));
    bwdDataPD.reset(new convolution_backward_data::primitive_desc(
      *bwdDataDesc, eg, *bwdDataFwdPD));
    CHECK(dataBot_->getIntlPD() == bwdDataPD->diff_src_primitive_desc());
    CHECK(dataWgt_->getIntlPD() == bwdDataPD->weights_primitive_desc());
    CHECK(diffTop_->getIntlPD() == bwdDataPD->diff_dst_primitive_desc());
    // init mkldnn buffer and conversion
    real* botDiff = prevLayer->getOutputGrad()->getData();    
    if (setDnnBotDiffFmt_[i]) {
      diffBot_->initUser(botDiff, bwdDataPD->diff_src_primitive_desc());
      getPrev(i)->setTopDiffMD(diffBot_->getUserMD());
    } else {
      diffBot_->initUser(botDiff, botDims, fmt, eg);
    }
    diffBot_->initIntlCvt(dataBot_->getIntlPD(), dnnCvtIntl2User);
    bwdData_.reset(new convolution_backward_data(
      *bwdDataPD, *(diffTop_->getIntlMem()),
      *(dataWgt_->getIntlMem()), *(diffBot_->getIntlMem())));
    LOG(INFO) << "diff format flow --- "
      << DNN_FORMAT[diffBot_->getUserFmt()] << " <<< ("
      << DNN_FORMAT[diffBot_->getIntlFmt()] << " <<< "
      << DNN_FORMAT[diffTop_->getIntlFmt()] << ") <<< "
      << DNN_FORMAT[diffTop_->getUserFmt()];
  }
}

void MkldnnConvLayer::submitFwdOnce(PassType passType, int inputIdx,
  const MatrixPtr& botVal, const MatrixPtr& topVal) {
  real* botdata = botVal->getData();
  real* topdata = topVal->getData();
  std::vector<primitive> pipeline;

  dataBot_->submitCvt(pipeline, botdata);

  if (usePaddleFmt_ && passType == PASS_TRAIN) {
    weights_[inputIdx]->getW()->transpose(selfWgtData_[inputIdx], false);
    real* wgtdata = selfWgtData_[inputIdx]->getData();
    dataWgt_->submitCvt(pipeline, wgtdata);
  }
  pipeline.push_back(*fwd_);

  dataTop_->submitCvt(pipeline, topdata);
  // start forward
  stream(stream::kind::eager).submit(pipeline).wait();
}

void MkldnnConvLayer::submitDnnFwd(PassType passType) {
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    submitFwdOnce(passType, i, getPrev(i)->getOutputValue(), getOutputValue());
  }

  // forward activation
  if (useConvRelu_) {
    /* dropout */
    if (config_.drop_rate() > 0) {
      // TODO(TJ): check if other dnn format feasible for dropout
      // if not, add if when set datatop user format when has dropout
      CHECK(dataTop_->getUserFmt() == memory::format::nchw)
        << "format should only be nchw when dropout";
      forwardDropOut();
      CHECK_NE(activation_->getName(), "mkldnn_softmax")
          << "Softmax activation cannot be used with Dropout";
    }
    if (FLAGS_show_layer_stat) {
      showOutputStats();
    }
  } else {
    forwardActivation();
  }
}

void MkldnnConvLayer::exBwdBias(MatrixPtr topDiff) {
  MatrixPtr biases = Matrix::create(biases_->getWGrad()->getData(), 1,
    biases_->getWGrad()->getElementCnt(), false, useGpu_);

//  MatrixPtr biases =Matrix::create(1,biases_->getWGrad()->getElementCnt(), false, useGpu_);

  size_t mapW = getSize() / oc_;  // oh*ow
  size_t mapH = topDiff->getElementCnt() / mapW;  // oc*bs
  MatrixPtr vTmp = Matrix::create(topDiff->getData(), mapH, mapW, false, false);
  MatrixPtr transOutValue_;
  Matrix::resizeOrCreate(transOutValue_, mapW, mapH, false, false);
  vTmp->transpose(transOutValue_, false);  // false means no memory allocation
  vTmp->reshape(transOutValue_->getElementCnt() / oc_, oc_);
  biases->collectBias(*vTmp, 1.0f);

  biases->clear();

}

void MkldnnConvLayer::exBwdData(MatrixPtr topDiff, int i) {
  LayerPtr prevLayer = getPrev(i);
  if (NULL == prevLayer->getOutputGrad()) {
    return;
  }
  
  int subM = oc_ / gp_[i];
  int subN = oh_[i] * ow_[i];
  int subK = ic_[i] * fw_[i] * fh_[i] / gp_[i];
//  MatrixPtr tgtGrad = prevLayer->getOutputGrad();

  MatrixPtr tgtGrad = Matrix::create(prevLayer->getOutputGrad()->getHeight(),
    prevLayer->getOutputGrad()->getWidth(), false, false);//prevLayer->getOutputGrad();
  tgtGrad->zeroMem();

  /* reset the expand-grad memory */
  MatrixPtr expandInput_;
  Matrix::resizeOrCreate(expandInput_, subK*gp_[i], subN, false, false);
  real *localGradData = topDiff->getData();
  real *tgtGradData = tgtGrad->getData();
  MatrixPtr exWgt = weights_[i]->getW();
  if (!usePaddleFmt_) {
    MatrixPtr exWgt = Matrix::create(
      ic_[i] * fh_[i] * fw_[i] / gp_[i], oc_, false, false);
    weights_[i]->getW()->transpose(exWgt, false);
//    LOG(INFO) << "should not be here";
  }

  for (size_t n = 0; n < size_t(bs_); n++) {
    real *wgtData = exWgt->getData();
    real *expandInData = expandInput_->getData();
    for (int g = 0; g < gp_[i]; g++) {
      // create temporary matrix
      MatrixPtr C = Matrix::create(expandInData, subK, subN, false, useGpu_);
      MatrixPtr B = Matrix::create(localGradData, subM, subN, false, useGpu_);
      MatrixPtr A = Matrix::create(wgtData, subK, subM, false, useGpu_);
      C->mul(A, B);  // mul
      // clear the temporary matrix
      A->clear();
      B->clear();
      C->clear();
      expandInData += subK * subN;
      localGradData += subM * subN;
      wgtData += subK * subM;
    }
    // shrink one frame outGrad
    MatrixPtr oneGradTmp = Matrix::create(
      expandInput_->getData(), subK * gp_[i], subN, false, useGpu_);
    MatrixPtr vTmp = Matrix::create(
      tgtGradData, 1, ih_[i] * iw_[i] * ic_[i], false, false);
    vTmp->convShrink(*oneGradTmp, ih_[i], iw_[i], ic_[i], fh_[i], fw_[i],
      sh_[i], sw_[i], ph_[i], pw_[i], oh_[i], ow_[i], 1.0f, 1.0f);
    vTmp->clear();
    oneGradTmp->clear();
    // move the data-pointer
    tgtGradData += ih_[i] * iw_[i] * ic_[i];
  }

  LOG(INFO) << "--------------------ex data diff: "<< tgtGrad->getData()[0] << "," << tgtGrad->getData()[10];
  
}

void MkldnnConvLayer::exBwdWgts(MatrixPtr topDiff, int i) {
  MatrixPtr exWgt = weights_[i]->getWGrad();
  if (!usePaddleFmt_) {
    exWgt = Matrix::create(ic_[i]*fh_[i]*fw_[i]/gp_[i], oc_, false, false);
    weights_[i]->getWGrad()->transpose(exWgt, false);
  }
  MatrixPtr weightGrad = exWgt;
  MatrixPtr inputV = getPrev(i)->getOutputValue();
  int subM = oc_ / gp_[i];
  int subN = oh_[i] * ow_[i];
  int subK = ic_[i] * fw_[i] * fh_[i] / gp_[i];
  MatrixPtr expandInput_;
  Matrix::resizeOrCreate(expandInput_, subK*gp_[i], subN, false, false);
  real *gradData = topDiff->getData();
  for (size_t n = 0; n < size_t(bs_); n++) {  // frame by frame
    // expand
    Matrix::resizeOrCreate(expandInput_, subK*gp_[i], subN, false, false);
    real *imgData = inputV->getData() + n * inputV->getWidth();
    MatrixPtr imageTmp = Matrix::create(
      imgData, 1, ih_[i]*iw_[i]*ic_[i], false, false);
    expandInput_->convExpand(*imageTmp, ih_[i], iw_[i], ic_[i], fh_[i], fw_[i],
      sh_[i], sw_[i], ph_[i], pw_[i], oh_[i], ow_[i]);
    imageTmp->clear();
    real *wGradData = weightGrad->getData();
    real *expandInData = expandInput_->getData();
    // expand-mul one-group by one
    for (int g = 0; g < gp_[i]; g++) {
      MatrixPtr A = Matrix::create(expandInData, subK, subN, false, useGpu_);
      MatrixPtr B = Matrix::create(gradData, subM, subN, true, useGpu_);
      MatrixPtr C = Matrix::create(wGradData, subK, subM, false, useGpu_);
      C->mul(A, B, 1, 1);
      A->clear();
      B->clear();
      C->clear();
      gradData += subM * subN;
      wGradData += subK * subM;
      expandInData += subK * subN;
      }
    }
  // transpose to my wgts
  if (!usePaddleFmt_)
    exWgt->transpose(weights_[i]->getWGrad_mutable(), false);
}

void MkldnnConvLayer::exBackward(const UpdateCallback &callback) {
  MatrixPtr topGrad = getOutputGrad();
  if (biases_ && biases_->getWGrad()) {
    exBwdBias(topGrad);
//    biases_->getParameterPtr()->incUpdate(callback);
  }
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    exBwdData(topGrad, i);
    if (weights_[i]->getWGrad()) {
      exBwdWgts(topGrad, i);
//      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

void MkldnnConvLayer::submitBwdData(int inputIdx) {
  const MatrixPtr& botGrad = getPrev(inputIdx)->getOutputGrad();
  if (botGrad == NULL) {
    return;
  }
  real* botdiff = botGrad->getData();
  real* topdiff = getOutputGrad()->getData();
  std::vector<primitive> pipeline;
  if (usePaddleFmt_) {  // no need cvt wgt without usePaddleFmt_
    CHECK(selfWgtData_[inputIdx]);
//    weights_[inputIdx]->getW()->transpose(selfWgtData_[inputIdx], false);
    real* wgtdata = selfWgtData_[inputIdx]->getData();
//    dataWgt_->clearCvtFlag();
    dataWgt_->submitCvt(pipeline, wgtdata);
  }
  diffTop_->submitCvt(pipeline, topdiff);
  pipeline.push_back(*bwdData_);
//  diffBot_->clearCvtFlag();
  diffBot_->submitCvt(pipeline, botdiff);
  stream(stream::kind::eager).submit(pipeline).wait();
  LOG(INFO) << "--------------------my data diff: "<< botdiff[0] << "," << botdiff[10];
}

void MkldnnConvLayer::submitBwdWgts(int inputIdx) {
  real* botdata = getInputValue(inputIdx)->getData();  
  real* topdiff = getOutputGrad()->getData();
  real* wgtdiff = weights_[inputIdx]->getWGrad()->getData();
  if (usePaddleFmt_) {
    CHECK(selfWgtDiff_[inputIdx]);
    wgtdiff = selfWgtDiff_[inputIdx]->getData();
  }
  std::vector<primitive> pipeline;
  diffTop_->submitCvt(pipeline, topdiff);
  dataBot_->submitCvt(pipeline, botdata);
  pipeline.push_back(*bwdWgt_);
  diffWgt_->submitCvt(pipeline, wgtdiff);
  if (biases_ && biases_->getWGrad()) {
    // bias backward can only execute in filter backward with MKL-DNN
    real* biasdiff = biases_->getWGrad()->getData();
    diffBias_->submitCvt(pipeline, biasdiff);
  }
//  LOG(INFO) << "size:" << pipeline.size();
  stream(stream::kind::eager).submit(pipeline).wait();
  
  if (usePaddleFmt_) {
    // save to actual weight param
    selfWgtDiff_[inputIdx]->transpose(
      weights_[inputIdx]->getWGrad_mutable(), false);
  }
}

void MkldnnConvLayer::submitDnnBwd(const UpdateCallback &callback) {
  // backward activation
  backwardActivation();
  
  exBackward(nullptr);

  //real* wgtdiff = weights_[0]->getWGrad()->getData();
  real* biasdiff = biases_->getWGrad()->getData();
  //real* topdiff = getOutputGrad()->getData();
  
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    submitBwdData(i);
    if (weights_[i]->getWGrad()) {
    //  LOG(INFO) << "--------------------ex wgt, bias diff: "<< wgtdiff[0] << "," << wgtdiff[2] << ","<<biasdiff[0]<< ","<<biasdiff[2];
    LOG(INFO) << "--ex bias diff: "<< biasdiff[0]<< ","<<biasdiff[1]<< ","<<biasdiff[2]<< ","<<biasdiff[3]<< ","<<biasdiff[4]<< ","<<biasdiff[5]<< ","<<biasdiff[6]<< ","<<biasdiff[7];

      submitBwdWgts(i);
   //   LOG(INFO) << "--------------------my wgt, bias diff: "<< wgtdiff[0] << "," << wgtdiff[2] << ","<<biasdiff[0]<< ","<<biasdiff[2];
   LOG(INFO) << "--my bias diff: "<< biasdiff[0]<< ","<<biasdiff[1]<< ","<<biasdiff[2]<< ","<<biasdiff[3]<< ","<<biasdiff[4]<< ","<<biasdiff[5]<< ","<<biasdiff[6]<< ","<<biasdiff[7];

      weights_[i]->getParameterPtr()->incUpdate(callback);
     
    }
  }
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }

}

}  // namespace paddle
