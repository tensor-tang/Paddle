/* Copyright (c) 2017 */

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
  int i = 0;
  for (auto &inputConfig : config_.inputs()) {
    const ConvConfig &conf = inputConfig.conv_conf();
    ih_[i] = conf.img_size();
    iw_[i] = conf.img_size();
    oh_[i] = conf.output_x();
    ow_[i] = conf.output_x();
    ic_[i] = conf.channels();
    fw_.push_back(conf.filter_size());
    fh_.push_back(conf.filter_size_y());
    pw_.push_back(conf.padding());
    ph_.push_back(conf.padding_y());
    sw_.push_back(conf.stride());
    sh_.push_back(conf.stride_y());
    gp_.push_back(conf.groups());

    bool caffeMode = conf.caffe_mode();
    if (!caffeMode) {
      LOG(FATAL) << "Only support caffe mode with MKL-DNN by now!";
    }
    i++;
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
  topData_.reset(new MkldnnBuffer());
  if (hasBias) {
    biasData_.reset(new MkldnnBuffer());
  }
  // 2. init user
  real *topDataData = getOutputValue()->getData();
  topData_->initUser(topDataData, topDims_, topFmt_, eg);
  if (hasBias) {
    real *biasDataData = biases_->getW()->getData();
    biasData_->initUser(biasDataData, biasDims_[0], biasFmt_[0], eg);
  }
  // TODO(TJ): only care about i==0 yet, and never tested g!=1
  CHECK_EQ(inputLayers_.size(), 1);
  CHECK_EQ(botDatas_.size(), inputLayers_.size());
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
    botDatas_[i].reset(new MkldnnBuffer());
    wgtData_.reset(new MkldnnBuffer());
    /// 2. init user ***********************************************************
    real *botDataData = getPrev(i)->getOutputValue()->getData();
    // if use mkldnn wgt directly save into weight parameter
    real *wgtDataData = useMkldnnWgt_ ? weights_[i]->getW()->getData()
      : selfWgtData_[i]->getData();
    botDatas_[i]->initUser(botDataData, botDims_[i], botFmt_[i], eg);
    wgtData_->initUser(wgtDataData, wgtDims_[i], wgtFmt_[i], eg);
    // use internal bottom data if use prv input
    const std::shared_ptr<memory::desc>& prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      botDatas_[i]->resetUser(botDataData, *prvMD, eg);
      bool isNC = botDatas_[i]->getUserFmt() == memory::format::nc;
      if (isNC) {
        CHECK(ih_[i] == iw_[i] && ih_[i] == 1)
          << "iw, ih must be 1 with nc input";
        // do not support nc as input, so change to nchw
        memory::format fmt = memory::format::nchw;
        botDatas_[i]->resetUser(botDataData, botDims_[i], fmt, eg);
        VLOG(4) << "use nchw data fmt";
      } else {
        VLOG(4) << "use prev data fmt:" << DNN_FMTS[botDatas_[i]->getUserFmt()];
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
    botDatas_[i]->initCvt(fwdPD->src_primitive_desc(), dnnCvtUser2Intl);
    if (useMkldnnWgt_) {
      // directly use internal format and save in paddle weight parameter
      wgtDataData = weights_[i]->getW()->getData();
      wgtData_->resetUser(wgtDataData, fwdPD->weights_primitive_desc());
      wgtData_->initCvt(wgtData_->getUserPD(), dnnCvtNoNeed);
      // need check the memory size, should be strictly equal
      CHECK_EQ(wgtData_->getIntlSize(), parameters_[i]->getSize())
        << "can not use mkldnn wgt since memory size does not equal";
    } else {
      wgtData_->initCvt(fwdPD->weights_primitive_desc(), dnnCvtUser2Intl);
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
          wgtDataData = selfWgtData_[i]->getData();
          wgtData_->submitCvt(cvtWgt, wgtDataData);
          stream(stream::kind::eager).submit(cvtWgt).wait();
        }
      }
    }
    if (hasBias) {
      // only cvt once
      if (!hasCvtBiasData) {
        hasCvtBiasData = true;
        CHECK(biasData_->getUserPD() == fwdPD->bias_primitive_desc())
          << "should always be format::x, or changed in new mkldnn version";
        biasData_->initCvt(biasData_->getUserPD(), dnnCvtNoNeed);
      } else {
        CHECK(biasData_->getIntlPD() == fwdPD->bias_primitive_desc())
          << "all bias formats should equal";
      }
    }
    // cvt topDataData buffer only once, set dnn MemDesc if next is also mkldnn
    if (!hasCvtTopData) {
      hasCvtTopData = true;
      if (nextIsDnn_) {
        topData_->resetUser(topDataData, fwdPD->dst_primitive_desc());
        setTopDataMD(topData_->getUserMD());
        VLOG(4) << "set next data fmt: " << DNN_FMTS[topData_->getUserFmt()];
      }
      topData_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);
    } else {
      CHECK(topData_->getIntlPD() == fwdPD->dst_primitive_desc())
        << "all output formats should equal";
    }
    /// 5. create fwd handle ***************************************************
    if (hasBias) {
      if (useConvRelu_) {
        fwd_.reset(new convolution_relu_forward(*convReluPD,
              *(botDatas_[i]->getIntlMem()), *(wgtData_->getIntlMem()),
              *(biasData_->getIntlMem()), *(topData_->getIntlMem())));
      } else {
        fwd_.reset(new convolution_forward(*fwdPD,
              *(botDatas_[i]->getIntlMem()), *(wgtData_->getIntlMem()),
              *(biasData_->getIntlMem()), *(topData_->getIntlMem())));
      }
    } else {
      if (useConvRelu_) {
        fwd_.reset(new convolution_relu_forward(*convReluPD,
              *(botDatas_[i]->getIntlMem()), *(wgtData_->getIntlMem()),
              *(topData_->getIntlMem())));
      } else {
        fwd_.reset(new convolution_forward(*fwdPD,
              *(botDatas_[i]->getIntlMem()), *(wgtData_->getIntlMem()),
              *(topData_->getIntlMem())));
      }
    }
    if (wgtData_) {
      VLOG(3) << "weight data flow --- "
        << DNN_FMTS[wgtData_->getUserFmt()]
        << " >>> "
        << DNN_FMTS[wgtData_->getIntlFmt()];
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
  topDiff_.reset(new MkldnnBuffer());
  if (hasBias) {
    biasDiff_.reset(new MkldnnBuffer());
  }
  // 2. init user
  real *topDiffData = getDnnOutputGrad()->getData();
  topDiffBwdWgt_->initUser(topDiffData, topDims_, topFmt_, eg);
  topDiff_->initUser(topDiffData, topDims_, topFmt_, eg);
  if (hasBias) {
    real* biasDiffData = biases_->getWGrad()->getData();
    biasDiff_->initUser(biasDiffData, biasDims_[0], biasFmt_[0], eg);
  }
  // use internal top diff if use dnn input
  const std::shared_ptr<mkldnn::memory::desc>& prvMD = getTopDiffMD();
  if (prvMD) {
    topDiffBwdWgt_->resetUser(topDiffData, *prvMD, eg);
    topDiff_->resetUser(topDiffData, *prvMD, eg);
    bool isNC = topDiffBwdWgt_->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(oh_[0] == ow_[0] && oh_[0] == 1)
        << "ow, oh must be 1 with nc input";
      // do not support nc as input, so change to nchw
      memory::format fmt = memory::format::nchw;
      topDiffBwdWgt_->resetUser(topDiffData, topDims_, fmt, eg);
      topDiff_->resetUser(topDiffData, topDims_, fmt, eg);
      VLOG(4) << "use nchw diff fmt";
    } else {
      VLOG(4) << "use prev diff fmt:" << DNN_FMTS[topDiffBwdWgt_->getUserFmt()];
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
    wgtDiff_.reset(new MkldnnBuffer());
    real *wgtDiffData = useMkldnnWgt_? weights_[i]->getWGrad()->getData()
      : selfWgtDiff_[i]->getData();
    wgtDiff_->initUser(wgtDiffData, wgtDims_[i], wgtFmt_[i], eg);
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
        biasData_->getIntlMD(),
        MkldnnBuffer::getMD(topDims_),
        strides, padding, padR, padKind));
      // TODO(TJ): only bwd bias once with multi inputs layers, or sum them?
      bwdWgtDesc.reset(new convolution_backward_weights::desc(
        algo,
        MkldnnBuffer::getMD(botDims_[i]),
        MkldnnBuffer::getMD(wgtDims_[i]),
        biasData_->getIntlMD(),
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
    CHECK(botDatas_[i]->getIntlPD() == bwdWgtPD->src_primitive_desc());
//    CHECK(wgtData_->getIntlPD() == bwdWgtPD->diff_weights_primitive_desc());
//    CHECK(biasData_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc());
//    CHECK(topData_->getIntlPD() == bwdWgtPD->diff_dst_primitive_desc());
    CHECK(weights_[i]->getWGrad()) << "should have weight";
    // 3. init conversion
    if (useMkldnnWgt_) {
      wgtDiff_->resetUser(wgtDiffData, bwdWgtPD->diff_weights_primitive_desc());
      wgtDiff_->initCvt(wgtDiff_->getUserPD(), dnnCvtNoNeed);
      CHECK_EQ(wgtDiff_->getIntlSize(), wgtData_->getIntlSize())
        << "can not use mkldnn wgt since memory size does not equal";
    } else {
      wgtDiff_->initCvt(
        bwdWgtPD->diff_weights_primitive_desc(), dnnCvtIntl2User);
    }
    if (hasBias) {
      if (!hasCvtBiasDiff) {
        hasCvtBiasDiff = true;
        CHECK(biasDiff_->getUserPD() == bwdWgtPD->diff_bias_primitive_desc())
          << "should always be format::x, or changed in new mkldnn version";
        biasDiff_->initCvt(biasDiff_->getUserPD(), dnnCvtNoNeed);
      } else {
        CHECK(biasDiff_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc())
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
        << "all topDiffData formats should equal";
    }
    // 4. create bwdwgt handle
    if (hasBias) {
      // bias backward can only execute with weight backward in MKL-DNN
      bwdWgt_.reset(new convolution_backward_weights(*bwdWgtPD,
        *(botDatas_[i]->getIntlMem()), *(topDiffBwdWgt_->getIntlMem()),
        *(wgtDiff_->getIntlMem()), *(biasDiff_->getIntlMem())));
    } else {
      bwdWgt_.reset(new convolution_backward_weights(*bwdWgtPD,
        *(botDatas_[i]->getIntlMem()), *(topDiffBwdWgt_->getIntlMem()),
        *(wgtDiff_->getIntlMem())));
    }
    if (wgtDiff_) {
      VLOG(3) << "bwd data weight diff flow --- "
        << DNN_FMTS[wgtDiff_->getUserFmt()]
        << " <<< "
        << DNN_FMTS[wgtDiff_->getIntlFmt()];
    }
    // then prepare backward data ----------------------------------------------
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      continue;  // data layer has not diff
    }
    // 1. create buffer and init user
    real* botDiffData = getDnnInputGrad(i)->getData();
    botDiffs_[i].reset(new MkldnnBuffer());
    wgtDataBwd_.reset(new MkldnnBuffer());
    botDiffs_[i]->initUser(botDiffData, botDims_[i], botFmt_[i], eg);
    wgtDataBwd_->initUser(wgtData_->getIntlData(), wgtData_->getIntlPD());
    // 2. init backward data primitive desc
    std::shared_ptr<convolution_forward::desc> bwdDataFwdDesc;
    std::shared_ptr<convolution_forward::primitive_desc> bwdDataFwdPD;
    std::shared_ptr<convolution_backward_data::desc> bwdDataDesc;
    std::shared_ptr<convolution_backward_data::primitive_desc> bwdDataPD;
    bwdDataFwdDesc.reset(new convolution_forward::desc(
      fwdpk, algo,
      MkldnnBuffer::getMD(botDims_[i]),  // botDatas_[i]->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_[i]),  //, bwdWgtFmt),
      MkldnnBuffer::getMD(topDims_),  // topDiffBwdWgt_->getIntlMD(),
      strides, padding, padR, padKind));
    bwdDataDesc.reset(new convolution_backward_data::desc(
      algo,
      MkldnnBuffer::getMD(botDims_[i]),  // botDatas_[i]->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_[i]),  //, bwdWgtFmt),
      MkldnnBuffer::getMD(topDims_),  // topDiffBwdWgt_->getIntlMD(),
      strides, padding, padR, padKind));
    bwdDataFwdPD.reset(new convolution_forward::primitive_desc(
      *bwdDataFwdDesc, eg));
    bwdDataPD.reset(new convolution_backward_data::primitive_desc(
      *bwdDataDesc, eg, *bwdDataFwdPD));
//    CHECK(botDatas_[i]->getIntlPD() == bwdDataPD->diff_src_primitive_desc());
//    CHECK(topDiff_->getIntlPD() == bwdDataPD->diff_dst_primitive_desc());
    // 3. init conversion
    if (!hasCvtTopDiffBwdData) {
      hasCvtTopDiffBwdData = true;
      topDiff_->initCvt(bwdDataPD->diff_dst_primitive_desc(), dnnCvtUser2Intl);
    } else {
      CHECK(topDiff_->getIntlPD() == bwdDataPD->diff_dst_primitive_desc())
        << "all topDiffData formats should equal";
    }
    if (wgtDataBwd_->initCvt(
      bwdDataPD->weights_primitive_desc(), dnnCvtUser2Intl)) {
      VLOG(3) << "bwd data weight data flow --- "
          << DNN_FMTS[wgtDataBwd_->getUserFmt()]
          << " >>> "
          << DNN_FMTS[wgtDataBwd_->getIntlFmt()];
    }
    if (prevIsDnn_[i]) {
      botDiffs_[i]->resetUser(botDiffData,
        bwdDataPD->diff_src_primitive_desc());
      prevLayer->setTopDiffMD(this->getName(), botDiffs_[i]->getUserMD());
      VLOG(4) << "set next diff fmt: " << DNN_FMTS[botDiffs_[i]->getUserFmt()];
    }
    botDiffs_[i]->initCvt(
      bwdDataPD->diff_src_primitive_desc(), dnnCvtIntl2User);
    // 4. create bwd data handle
    bwdData_.reset(new convolution_backward_data(
      *bwdDataPD, *(topDiff_->getIntlMem()),
      *(wgtDataBwd_->getIntlMem()), *(botDiffs_[i]->getIntlMem())));
  }
}

void MkldnnConvLayer::submitDnnFwd(PassType passType) {
  real* topDataData = getOutputValue()->getData();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    real* botDataData = getPrev(i)->getOutputValue()->getData();
    std::vector<primitive> pipeline;
    botDatas_[i]->submitCvt(pipeline, botDataData);
    if (!useMkldnnWgt_ && passType != PASS_TEST) {
      // transpose and cvt every time in training if do not use mkldnn wgt
      weights_[i]->getW()->transpose(selfWgtData_[i], false);
      real* wgtDataData = selfWgtData_[i]->getData();
      wgtData_->submitCvt(pipeline, wgtDataData);
    }
    // no need to cvt bias
  //  if (biases_ && biases_->getW()) {
  //    real* biasDataData = biases_->getW()->getData();
  //    biasData_->submitCvt(pipeline, biasDataData);
  //  }
    pipeline.push_back(*fwd_);
    topData_->submitCvt(pipeline, topDataData);
    stream(stream::kind::eager).submit(pipeline).wait();
  }

  // if use convrelu, skip activation
  if (useConvRelu_) {
    /* dropout */
    if (config_.drop_rate() > 0) {
      // TODO(TJ): check if other dnn format feasible for dropout
      // if not, add if when set datatop user format when has dropout
      CHECK(topData_->getUserFmt() == memory::format::nchw)
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
  real* botDiffData = botGrad->getData();
  real* topDiffData = getOutputGrad()->getData();
  std::vector<primitive> pipeline;
  wgtDataBwd_->submitCvt(pipeline, wgtData_->getIntlData());
  topDiff_->submitCvt(pipeline, topDiffData);  pipeline.push_back(*bwdData_);
  botDiffs_[idx]->submitCvt(pipeline, botDiffData);
  stream(stream::kind::eager).submit(pipeline).wait();
}

void MkldnnConvLayer::submitBwdWgts(int idx) {
  real* botDataData = getInputValue(idx)->getData();
  real* topDiffData = getOutputGrad()->getData();
  real* wgtDiffData = weights_[idx]->getWGrad()->getData();
  if (!useMkldnnWgt_) {
    CHECK(selfWgtDiff_[idx]);
    wgtDiffData = selfWgtDiff_[idx]->getData();
  }
  std::vector<primitive> pipeline;
  topDiffBwdWgt_->submitCvt(pipeline, topDiffData);
  botDatas_[idx]->submitCvt(pipeline, botDataData);
  pipeline.push_back(*bwdWgt_);
  wgtDiff_->submitCvt(pipeline, wgtDiffData);
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
