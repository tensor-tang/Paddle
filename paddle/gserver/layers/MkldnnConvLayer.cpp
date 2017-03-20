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

  if (config_.has_add_size()) {
    addSize_ = config_.add_size();
  }
  if (config_.has_use_mkldnn_wgt()) {
    useMkldnnWgt_ = config_.use_mkldnn_wgt();
  }

  /* initialize the weightList */
  CHECK(inputLayers_.size() == parameters_.size());
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    size_t height, width;
    selfWgtData_.push_back(nullptr);
    selfWgtDiff_.push_back(nullptr);
    if (!useMkldnnWgt_) {
      height = ic_[i] * fh_[i] * fw_[i] / gp_[i];
      width = oc_;
      selfWgtData_[i] = Matrix::create(width, height, false, false);
      selfWgtDiff_[i] = Matrix::create(width, height, false, false);
      selfWgtData_[i]->zeroMem();
      selfWgtDiff_[i]->zeroMem();
    } else {
      height = oc_;
      width = ic_[i] * fh_[i] * fw_[i] / gp_[i];
    }
    // create a new weight
    CHECK_EQ(parameters_[i]->getSize(), width * height);
    Weight* w = new Weight(height, width, parameters_[i], 0);
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
//  reserveOutput(bs_, getSize());
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
}

void MkldnnConvLayer::resetDnnFwd(PassType passType) {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  algorithm algo = algorithm::convolution_direct;
  prop_kind fwdpk = passType == PASS_TEST ? prop_kind::forward_scoring
    : prop_kind::forward_training;
  padding_kind padKind = padding_kind::zero;
  topDims_ = {bs_, oc_, oh_[0], ow_[0]};
  biasDims_[0] = {oc_};

  bool hasBias = (biases_ && biases_->getW());

  // conv_relu only support scoring yet
  useConvRelu_ = (hasRelu_ && passType == PASS_TEST);

  bool hasCvtTopData = false;
  bool hasCvtBiasData = false;

  // 1. create buffer, only have one output and bias buffer
  dataTop_.reset(new MkldnnBuffer());
  if (hasBias) {
    dataBias_.reset(new MkldnnBuffer());
  }
  // 2. init user
  real *topData = getOutputValue()->getData();
  dataTop_->initUser(topData, topDims_, topFmt_, eg);
  if (hasBias) {
    real *biasData = biases_->getW()->getData();
    dataBias_->initUser(biasData, biasDims_[0], biasFmt_[0], eg);
  }
  // TODO(TJ): only care about i==0 yet, and never tested g!=1
  CHECK_EQ(inputLayers_.size(), 1);
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize()) << "batchsize should equal";
    // init dim structure that describes user data.
    botDims_[i] = {bs_, ic_[i], ih_[i], iw_[i]};
    botFmt_[i] = memory::format::nchw;
    wgtDims_[i] = (gp_[i] == 1) ? memory::dims{oc_, ic_[i], fh_[i], fw_[i]}
      : memory::dims{gp_[i], oc_/gp_[i], ic_[i]/gp_[i], fh_[i], fw_[i]};
    wgtFmt_[i] = (gp_[i] == 1) ? memory::format::oihw : memory::format::goihw;
    memory::dims strides = {sh_[i], sw_[i]};
    memory::dims padding = {ph_[i], pw_[i]};
    std::vector<int> padR = {ph_[i], pw_[i]};
    for (int k = 0; k < 2; ++k) {
      if ((ih_[i] + ph_[i] + padR[0] - fh_[i])/sh_[i] + 1 != oh_[i]) ++padR[0];
      if ((iw_[i] + pw_[i] + padR[1] - fw_[i])/sw_[i] + 1 != ow_[i]) ++padR[1];
    }
    /// 1. create buffer, could be vector later ********************************
    dataBot_.reset(new MkldnnBuffer());
    dataWgt_.reset(new MkldnnBuffer());
    /// 2. init user ***********************************************************
    real *botData = getPrev(i)->getOutputValue()->getData();
    // if use mkldnn wgt directly save into weight parameter
    real *wgtData = useMkldnnWgt_ ? weights_[i]->getW()->getData()
      : selfWgtData_[i]->getData();
    dataBot_->initUser(botData, botDims_[i], botFmt_[i], eg);
    dataWgt_->initUser(wgtData, wgtDims_[i], wgtFmt_[i], eg);
    // use internal bottom data if use prv input
    const std::shared_ptr<memory::desc>& prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      dataBot_->resetUser(botData, *prvMD, eg);
      bool isNC = dataBot_->getUserFmt() == memory::format::nc;
      if (isNC) {
        CHECK(ih_[i] == iw_[i] && ih_[i] == 1)
          << "iw, ih must be 1 with nc input";
        // do not support nc as input, so change to nchw
        memory::format fmt = memory::format::nchw;
        dataBot_->resetUser(botData, botDims_[i], fmt, eg);
        VLOG(4) << "use nchw data fmt";
      } else {
        VLOG(4) << "use prev data fmt: " << DNN_FMTS[dataBot_->getUserFmt()];
      }
    }
    /// 3. create mkldnn forward PD ********************************************
    std::shared_ptr<convolution_forward::desc> fwdDesc;
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> fwdPD;
    if (hasBias) {
      fwdDesc.reset(new convolution_forward::desc(fwdpk, algo,
      // since conv have very solid policy to choose best format, so use any
        MkldnnBuffer::getMD(botDims_[i]),
        MkldnnBuffer::getMD(wgtDims_[i]),
        MkldnnBuffer::getMD(biasDims_[i]),
        MkldnnBuffer::getMD(topDims_),
        strides, padding, padR, padKind));
    } else {
      fwdDesc.reset(new convolution_forward::desc(fwdpk, algo,
        MkldnnBuffer::getMD(botDims_[i]),
        MkldnnBuffer::getMD(wgtDims_[i]),
        MkldnnBuffer::getMD(topDims_),
        strides, padding, padR, padKind));
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
    /// 4. init conversion *****************************************************
    dataBot_->initCvt(fwdPD->src_primitive_desc(), dnnCvtUser2Intl);
    if (useMkldnnWgt_) {
      // directly use internal format and save in paddle weight parameter
      wgtData = weights_[i]->getW()->getData();
      dataWgt_->resetUser(wgtData, fwdPD->weights_primitive_desc());
      dataWgt_->initCvt(dataWgt_->getUserPD(), dnnCvtNoNeed);
      // need check the memory size, should be strictly equal
      CHECK_EQ(dataWgt_->getIntlSize(), parameters_[i]->getSize())
        << "can not use mkldnn wgt since memory size does not equal";
    } else {
      dataWgt_->initCvt(fwdPD->weights_primitive_desc(), dnnCvtUser2Intl);
    }
    // only init wgt once
    if (!hasInited_) {
      hasInited_ = true;
      if (useMkldnnWgt_) {
        // cvt the initial paddle wgt to mkldnn wgt only once when training
        // in testing phase do not need cvt
        if (passType != PASS_TEST) {
          // paddle wgt is transposed
          size_t height = weights_[i]->getW()->getWidth();
          size_t width = weights_[i]->getW()->getHeight();
          MatrixPtr initWgt = Matrix::create(height, width, false, false);
          initWgt->copyFrom(*(weights_[i]->getW()));
          initWgt = initWgt->getTranspose();
          MkldnnBufferPtr tmp(new MkldnnBuffer());
          tmp->initUser(initWgt->getData(), wgtDims_[i], wgtFmt_[i], eg);
          tmp->initCvt(fwdPD->weights_primitive_desc(), dnnCvtUser2Intl);
          std::vector<primitive> cvtWgt;
          tmp->submitCvt(cvtWgt);
          stream(stream::kind::eager).submit(cvtWgt).wait();
          real* dst = weights_[i]->getW()->getData();
          memcpy(dst, tmp->getIntlData(), tmp->getIntlSize() * sizeof(real));
        }
      } else {
        // load the initial paddle wgt and cvt only once when scoring
        // in training phase will cvt in every forward
        if (passType == PASS_TEST) {
          weights_[i]->getW()->transpose(selfWgtData_[i], false);
          std::vector<primitive> cvtWgt;
          wgtData = selfWgtData_[i]->getData();
          dataWgt_->submitCvt(cvtWgt, wgtData);
          stream(stream::kind::eager).submit(cvtWgt).wait();
        }
      }
    }
    if (hasBias) {
      // only cvt once
      if (!hasCvtBiasData) {
        hasCvtBiasData = true;
        CHECK(dataBias_->getUserPD() == fwdPD->bias_primitive_desc())
          << "should always be format::x, or changed in new mkldnn version";
        dataBias_->initCvt(dataBias_->getUserPD(), dnnCvtNoNeed);
      } else {
        CHECK(dataBias_->getIntlPD() == fwdPD->bias_primitive_desc())
          << "all bias formats should equal";
      }
    }
    // cvt topData buffer only once, set dnn MemDesc if next is also mkldnn
    if (!hasCvtTopData) {
      hasCvtTopData = true;
      if (nextIsDnn_) {
        dataTop_->resetUser(topData, fwdPD->dst_primitive_desc());
        setTopDataMD(dataTop_->getUserMD());
        VLOG(4) << "set next data fmt: " << DNN_FMTS[dataTop_->getUserFmt()];
      }
      dataTop_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);
    } else {
      CHECK(dataTop_->getIntlPD() == fwdPD->dst_primitive_desc())
        << "all output formats should equal";
    }
    /// 5. create fwd handle ***************************************************
    if (hasBias) {
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
    if (dataWgt_) {
      VLOG(3) << "weight data flow --- "
        << DNN_FMTS[dataWgt_->getUserFmt()]
        << " >>> "
        << DNN_FMTS[dataWgt_->getIntlFmt()];
    }
  }
}

void MkldnnConvLayer::resetDnnBwd() {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  algorithm algo = algorithm::convolution_direct;
  padding_kind padKind = padding_kind::zero;
  prop_kind fwdpk = prop_kind::forward_training;
  bool hasBias = (biases_ && biases_->getWGrad());

  bool hasCvtTopDiffBwdWgt = false;
  bool hasCvtTopDiffBwdData = false;
  bool hasCvtBiasDiff = false;

  // 1. create buffer, only have one output and bias buffer
  topDiffBwdWgt_.reset(new MkldnnBuffer());
  diffTop_.reset(new MkldnnBuffer());
  if (hasBias) {
    diffBias_.reset(new MkldnnBuffer());
  }
  // 2. init user
  real *topDiff = getDnnOutputGrad()->getData();
  topDiffBwdWgt_->initUser(topDiff, topDims_, topFmt_, eg);
  diffTop_->initUser(topDiff, topDims_, topFmt_, eg);
  if (hasBias) {
    real* biasDiff = biases_->getWGrad()->getData();
    diffBias_->initUser(biasDiff, biasDims_[0], biasFmt_[0], eg);
  }
  // use internal top diff if use dnn input
  const std::shared_ptr<mkldnn::memory::desc>& prvMD = getTopDiffMD();
  if (prvMD) {
    topDiffBwdWgt_->resetUser(topDiff, *prvMD, eg);
    diffTop_->resetUser(topDiff, *prvMD, eg);
    bool isNC = topDiffBwdWgt_->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(oh_[0] == ow_[0] && oh_[0] == 1)
        << "ow, oh must be 1 with nc input";
      // do not support nc as input, so change to nchw
      memory::format fmt = memory::format::nchw;
      topDiffBwdWgt_->resetUser(topDiff, topDims_, fmt, eg);
      diffTop_->resetUser(topDiff, topDims_, fmt, eg);
      VLOG(4) << "use nchw diff fmt";
    } else {
      VLOG(4) << "use prev diff fmt: " << DNN_FMTS[topDiffBwdWgt_->getUserFmt()];
    }
  }
  // TODO(TJ): only care about i==0 yet, and never tested g!=1
  CHECK_EQ(inputLayers_.size(), 1);
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize()) << "batchsize should equal";
    memory::dims strides = {sh_[i], sw_[i]};
    memory::dims padding = {ph_[i], pw_[i]};
    std::vector<int> padR = {ph_[i], pw_[i]};
    for (int k = 0; k < 2; ++k) {
      if ((ih_[i] + ph_[i] + padR[0] - fh_[i])/sh_[i] + 1 != oh_[i]) ++padR[0];
      if ((iw_[i] + pw_[i] + padR[1] - fw_[i])/sw_[i] + 1 != ow_[i]) ++padR[1];
    }
    // 1. create wgt buffer and init, could be vector later
    diffWgt_.reset(new MkldnnBuffer());
    real *wgtDiff = useMkldnnWgt_? weights_[i]->getWGrad()->getData()
      : selfWgtDiff_[i]->getData();
    diffWgt_->initUser(wgtDiff, wgtDims_[i], wgtFmt_[i], eg);
    // 2. prepare backward weight and bias PD-----------------------------------
    // bias backward can only execute with weight bakcward
    // unable execute seperately
    std::shared_ptr<convolution_forward::desc> fwdDesc;
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> fwdPD;
    std::shared_ptr<convolution_backward_weights::desc> bwdWgtDesc;
    std::shared_ptr<convolution_backward_weights::primitive_desc> bwdWgtPD;
    // conv has solid policy to choose best format, so use any
    if (hasBias) {
      fwdDesc.reset(new convolution_forward::desc(
        fwdpk, algo,
        MkldnnBuffer::getMD(botDims_[i]),
        MkldnnBuffer::getMD(wgtDims_[i]),
        dataBias_->getIntlMD(),
        MkldnnBuffer::getMD(topDims_),
        strides, padding, padR, padKind));
      // TODO(TJ): only bwd bias once with multi inputs layers, or sum them?
      bwdWgtDesc.reset(new convolution_backward_weights::desc(
        algo,
        MkldnnBuffer::getMD(botDims_[i]),
        MkldnnBuffer::getMD(wgtDims_[i]),
        dataBias_->getIntlMD(),
        MkldnnBuffer::getMD(topDims_),
        strides, padding, padR, padKind));
    } else {
      fwdDesc.reset(new convolution_forward::desc(
        fwdpk, algo,
        MkldnnBuffer::getMD(botDims_[i]),
        MkldnnBuffer::getMD(wgtDims_[i]),
        MkldnnBuffer::getMD(topDims_),
        strides, padding, padR, padKind));
      bwdWgtDesc.reset(new convolution_backward_weights::desc(
        algo,
        MkldnnBuffer::getMD(botDims_[i]),
        MkldnnBuffer::getMD(wgtDims_[i]),
        MkldnnBuffer::getMD(topDims_),
        strides, padding, padR, padKind));
    }
    fwdPD.reset(new convolution_forward::primitive_desc(*fwdDesc, eg));
    bwdWgtPD.reset(new convolution_backward_weights::primitive_desc(
      *bwdWgtDesc, eg, *fwdPD));
    CHECK(dataBot_->getIntlPD() == bwdWgtPD->src_primitive_desc());
//    CHECK(dataWgt_->getIntlPD() == bwdWgtPD->diff_weights_primitive_desc());
//    CHECK(dataBias_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc());
//    CHECK(dataTop_->getIntlPD() == bwdWgtPD->diff_dst_primitive_desc());
    CHECK(weights_[i]->getWGrad()) << "should have weight";
    // 3. init conversion
    if (useMkldnnWgt_) {
      diffWgt_->resetUser(wgtDiff, bwdWgtPD->diff_weights_primitive_desc());
      diffWgt_->initCvt(diffWgt_->getUserPD(), dnnCvtNoNeed);
      CHECK_EQ(diffWgt_->getIntlSize(), dataWgt_->getIntlSize())
        << "can not use mkldnn wgt since memory size does not equal";
    } else {
      diffWgt_->initCvt(
        bwdWgtPD->diff_weights_primitive_desc(), dnnCvtIntl2User);
    }
    if (hasBias) {
      if (!hasCvtBiasDiff) {
        hasCvtBiasDiff = true;
        CHECK(diffBias_->getUserPD() == bwdWgtPD->diff_bias_primitive_desc())
          << "should always be format::x, or changed in new mkldnn version";
        diffBias_->initCvt(diffBias_->getUserPD(), dnnCvtNoNeed);
      } else {
        CHECK(diffBias_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc())
          << "all bias formats should equal";
      }
    }
    if (!hasCvtTopDiffBwdWgt) {
      hasCvtTopDiffBwdWgt = true;
      topDiffBwdWgt_->initCvt(bwdWgtPD->diff_dst_primitive_desc(),
        dnnCvtUser2Intl);
      VLOG(3) << "bwd wgt top diff flow --- "
        << DNN_FMTS[topDiffBwdWgt_->getUserFmt()]
        << " >>> "
        << DNN_FMTS[topDiffBwdWgt_->getIntlFmt()];
    } else {
      CHECK(topDiffBwdWgt_->getIntlPD() == bwdWgtPD->diff_dst_primitive_desc())
        << "all topDiff formats should equal";
    }
    // 4. create bwdwgt handle
    if (hasBias) {
      // bias backward can only execute with weight backward in MKL-DNN
      bwdWgt_.reset(new convolution_backward_weights(*bwdWgtPD,
        *(dataBot_->getIntlMem()), *(topDiffBwdWgt_->getIntlMem()),
        *(diffWgt_->getIntlMem()), *(diffBias_->getIntlMem())));
    } else {
      bwdWgt_.reset(new convolution_backward_weights(*bwdWgtPD,
        *(dataBot_->getIntlMem()), *(topDiffBwdWgt_->getIntlMem()),
        *(diffWgt_->getIntlMem())));
    }
    if (diffWgt_) {
      VLOG(3) << "bwd data weight diff flow --- "
        << DNN_FMTS[diffWgt_->getUserFmt()]
        << " <<< "
        << DNN_FMTS[diffWgt_->getIntlFmt()];
    }
    // then prepare backward data ----------------------------------------------
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      continue;  // data layer has not diff
    }
    // 1. create buffer and init user
    real* botDiff = getDnnInputGrad(i)->getData();
    diffBot_.reset(new MkldnnBuffer());
    dataWgtBwd_.reset(new MkldnnBuffer());
    diffBot_->initUser(botDiff, botDims_[i], botFmt_[i], eg);
    dataWgtBwd_->initUser(dataWgt_->getIntlData(), dataWgt_->getIntlPD());
    // 2. init backward data primitive desc
    std::shared_ptr<convolution_forward::desc> bwdDataFwdDesc;
    std::shared_ptr<convolution_forward::primitive_desc> bwdDataFwdPD;
    std::shared_ptr<convolution_backward_data::desc> bwdDataDesc;
    std::shared_ptr<convolution_backward_data::primitive_desc> bwdDataPD;
    bwdDataFwdDesc.reset(new convolution_forward::desc(
      fwdpk, algo,
      MkldnnBuffer::getMD(botDims_[i]),  // dataBot_->getIntlMD(), 
      MkldnnBuffer::getMD(wgtDims_[i]),  //, bwdWgtFmt),
      MkldnnBuffer::getMD(topDims_),  // topDiffBwdWgt_->getIntlMD(),
      strides, padding, padR, padKind));
    bwdDataDesc.reset(new convolution_backward_data::desc(
      algo,
      MkldnnBuffer::getMD(botDims_[i]),  // dataBot_->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_[i]),  //, bwdWgtFmt),
      MkldnnBuffer::getMD(topDims_),  //topDiffBwdWgt_->getIntlMD(),
      strides, padding, padR, padKind));
    bwdDataFwdPD.reset(new convolution_forward::primitive_desc(
      *bwdDataFwdDesc, eg));
    bwdDataPD.reset(new convolution_backward_data::primitive_desc(
      *bwdDataDesc, eg, *bwdDataFwdPD));
//    CHECK(dataBot_->getIntlPD() == bwdDataPD->diff_src_primitive_desc());
//    CHECK(diffTop_->getIntlPD() == bwdDataPD->diff_dst_primitive_desc());
    // 3. init conversion
    if (!hasCvtTopDiffBwdData) {
      hasCvtTopDiffBwdData = true;
      diffTop_->initCvt(bwdDataPD->diff_dst_primitive_desc(), dnnCvtUser2Intl);
    } else {
      CHECK(diffTop_->getIntlPD() == bwdDataPD->diff_dst_primitive_desc())
        << "all topDiff formats should equal";
    }
    if (dataWgtBwd_->initCvt(
      bwdDataPD->weights_primitive_desc(), dnnCvtUser2Intl)) {
      VLOG(3) << "bwd data weight data flow --- "
          << DNN_FMTS[dataWgtBwd_->getUserFmt()]
          << " >>> "
          << DNN_FMTS[dataWgtBwd_->getIntlFmt()];
    }
    if (prevIsDnn_[i]) {
      diffBot_->resetUser(botDiff, bwdDataPD->diff_src_primitive_desc());
      prevLayer->setTopDiffMD(this->getName(), diffBot_->getUserMD());
      VLOG(4) << "set next diff fmt: " << DNN_FMTS[diffBot_->getUserFmt()];
    }
    diffBot_->initCvt(bwdDataPD->diff_src_primitive_desc(), dnnCvtIntl2User);
    // 4. create bwd data handle
    bwdData_.reset(new convolution_backward_data(
      *bwdDataPD, *(diffTop_->getIntlMem()),
      *(dataWgtBwd_->getIntlMem()), *(diffBot_->getIntlMem())));
  }
}

void MkldnnConvLayer::submitDnnFwd(PassType passType) {
  real* topData = getOutputValue()->getData();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    real* botData = getPrev(i)->getOutputValue()->getData();
    std::vector<primitive> pipeline;
    dataBot_->submitCvt(pipeline, botData);
    if (!useMkldnnWgt_ && passType != PASS_TEST) {
      // transpose and cvt every time in training if do not use mkldnn wgt
      weights_[i]->getW()->transpose(selfWgtData_[i], false);
      real* wgtData = selfWgtData_[i]->getData();
      dataWgt_->submitCvt(pipeline, wgtData);
    }
    // no need to cvt bias
  //  if (biases_ && biases_->getW()) {
  //    real* biasdata = biases_->getW()->getData();
  //    dataBias_->submitCvt(pipeline, biasdata);
  //  }
    pipeline.push_back(*fwd_);
    dataTop_->submitCvt(pipeline, topData);
    stream(stream::kind::eager).submit(pipeline).wait();
  }

  // if use convrelu, skip activation
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

void MkldnnConvLayer::submitBwdData(int idx) {
  const MatrixPtr& botGrad = getDnnInputGrad(idx);
  if (botGrad == NULL) {
    return;
  }
  real* botDiff = botGrad->getData();
  real* topDiff = getOutputGrad()->getData();
  std::vector<primitive> pipeline;
  dataWgtBwd_->submitCvt(pipeline, dataWgt_->getIntlData());
  diffTop_->submitCvt(pipeline, topDiff);  pipeline.push_back(*bwdData_);
  diffBot_->submitCvt(pipeline, botDiff);
  stream(stream::kind::eager).submit(pipeline).wait();
}

void MkldnnConvLayer::submitBwdWgts(int idx) {
  real* botData = getInputValue(idx)->getData();
  real* topDiff = getOutputGrad()->getData();
  real* wgtDiff = weights_[idx]->getWGrad()->getData();
  if (!useMkldnnWgt_) {
    CHECK(selfWgtDiff_[idx]);
    wgtDiff = selfWgtDiff_[idx]->getData();
  }
  std::vector<primitive> pipeline;
  topDiffBwdWgt_->submitCvt(pipeline, topDiff);
  dataBot_->submitCvt(pipeline, botData);
  pipeline.push_back(*bwdWgt_);
  diffWgt_->submitCvt(pipeline, wgtDiff);
  // no need to submit cvt bias since biasfmt would not be changed
  stream(stream::kind::eager).submit(pipeline).wait();

  if (!useMkldnnWgt_) {
    // save to actual weight param
    selfWgtDiff_[idx]->transpose(weights_[idx]->getWGrad_mutable(), false);
  }
}

void MkldnnConvLayer::submitDnnBwd(const UpdateCallback &callback) {
  // backward activation
  backwardActivation();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    submitBwdData(i);
    if (weights_[i]->getWGrad()) {
      submitBwdWgts(i);
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
