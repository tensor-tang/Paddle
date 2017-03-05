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

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_fc, MkldnnFcLayer);

bool MkldnnFcLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  // only support 1 input layer by now
  CHECK_EQ(config_.inputs_size(), 1);
  CHECK(inputLayers_.size() == parameters_.size());

  bs_ = 0;
  oc_ = getSize();
  hasSpatial_ = false;
  if (config_.has_add_size()) {
    addSize_ = config_.add_size();
  }
  if (config_.has_use_mkldnn_wgt()) {
    useMkldnnWgt_ = config_.use_mkldnn_wgt();
  }

  for (size_t i = 0; i < inputLayers_.size(); i++) {
    // Option the parameters
    ic_.push_back(0);
    iw_.push_back(0);
    ih_.push_back(0);
    ow_.push_back(0);
    oh_.push_back(0);
    inputSizeByBS_.push_back(inputLayers_[i]->getSize());  // == ic*ih*iw
    // create a new weight
    size_t height, width;
    if (parameters_[i]->isSparse()) {
      CHECK_LE(parameters_[i]->getSize(), oc_ * inputSizeByBS_[i]);
    } else {
      CHECK_EQ(parameters_[i]->getSize(), oc_ * inputSizeByBS_[i]);
    }
    selfWgtData_.push_back(nullptr);
    selfWgtDiff_.push_back(nullptr);
    if (!useMkldnnWgt_) {
      height = inputSizeByBS_[i];
      width = oc_;
      selfWgtData_[i] = Matrix::create(width, height, false, false);
      selfWgtDiff_[i] = Matrix::create(width, height, false, false);
      selfWgtData_[i]->zeroMem();
      selfWgtDiff_[i]->zeroMem();
    } else {
      height = oc_;
      width = inputSizeByBS_[i];
    }
    Weight* w = new Weight(height, width, parameters_[i], 0);
    weights_.emplace_back(w);
  }

  // initialize biases_
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
  }
  return true;
}

// keep for paddle parameter server
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

void MkldnnFcLayer::clearDataDiff() {
//  reserveOutput(bs_, getSize());
}

void MkldnnFcLayer::reshape() {
  // reshape input and output size
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = 0;
  oc_ = getSize();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    int height = inputLayers_[i]->getOutput().getFrameHeight();
    int width = inputLayers_[i]->getOutput().getFrameWidth();
    if (height > 0 && width > 0) {
      hasSpatial_ = true;
      ih_[i] = height;
      iw_[i] = width;
    } else {
      hasSpatial_ = false;
      ih_[i] = 1;
      iw_[i] = 1;
    }
    ic_[i] = inputSizeByBS_[i] / (iw_[i] * ih_[i]);
    oh_[i] = 1;
    ow_[i] = 1;
    CHECK(ih_[i] * iw_[i]);
    CHECK(layerSize == 0 || size_t(oh_[i] * ow_[i] * oc_) == layerSize);
    layerSize = oh_[i] * ow_[i] * oc_;
  }
}

void MkldnnFcLayer::resetDnnFwd(PassType passType) {
  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  prop_kind pk = prop_kind::forward;
  bool hasBias = (biases_ && biases_->getW());
  // create dim structure that describes user data.
  if (!hasSpatial_) {
    botDims_[0] = {bs_, ic_[0]};
    wgtDims_[0] = {oc_, ic_[0]};  // transpose from paddle weight
    botFmt_[0] = memory::format::nc;
    wgtFmt_[0] = memory::format::oi;
  } else {
    botDims_[0] = {bs_, ic_[0], ih_[0], iw_[0]};
    wgtDims_[0] = {oc_, ic_[0], ih_[0], iw_[0]};
    botFmt_[0] = memory::format::nchw;
    wgtFmt_[0] = memory::format::oihw;
  }
  topDims_ = {bs_, oc_};
  topFmt_ = memory::format::nc;
  biasDims_[0] = {oc_};
  biasFmt_[0] = memory::format::x;

  hasCvtTopData_ = false;
  hasCvtBiasData_ = false;
  // 1. create mkldnn buffer, only have one output and bias buffer
  dataTop_.reset(new MkldnnBuffer());
  if (hasBias) {
    dataBias_.reset(new MkldnnBuffer());
  }
  // 2. init user top and bias
  real *topData = getOutputValue()->getData();
  dataTop_->initUser(topData, topDims_, topFmt_, eg);
  if (hasBias) {
    real *biasData = biases_->getW()->getData();
    dataBias_->initUser(biasData, biasDims_[0], biasFmt_[0], eg);
  }
  // TODO(TJ): only care about i==0 yet
  CHECK_EQ(inputLayers_.size(), 1);
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize()) << "batchsize should equal";
    /// 1. create buffer, could be vector later
    dataBot_.reset(new MkldnnBuffer());
    dataWgt_.reset(new MkldnnBuffer());
    // 2. init user memory of bottom, weights and bias
    real *botData = getPrev(i)->getOutputValue()->getData();
    real *wgtData = useMkldnnWgt_ ? weights_[i]->getW()->getData()
        : selfWgtData_[i]->getData();
    dataBot_->initUser(botData, botDims_[i], botFmt_[i], eg);
    dataWgt_->initUser(wgtData, wgtDims_[i], wgtFmt_[i], eg);
    // 3. create fc desc
    std::shared_ptr<inner_product_forward::desc> fwdDesc;
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> fwdPD;
    const std::shared_ptr<memory::desc> prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      dataBot_->resetUser(botData, *prvMD, eg);
      VLOG(4) << "use prev data fmt: " << DNN_FMTS[dataBot_->getUserFmt()];
    }
    if (hasBias) {
      fwdDesc.reset(new inner_product_forward::desc(pk,
          prvMD ? dataBot_->getUserMD() : getAnyMD(botDims_[i]),
          getAnyMD(wgtDims_[i]), getAnyMD(biasDims_[i]), getAnyMD(topDims_)));
    } else {
      fwdDesc.reset(new inner_product_forward::desc(pk,
          prvMD ? dataBot_->getUserMD() : getAnyMD(botDims_[i]),
          getAnyMD(wgtDims_[i]), getAnyMD(topDims_)));
    }
    fwdPD.reset(new inner_product_forward::primitive_desc(*fwdDesc, eg));
    // 4. init cvt
    dataBot_->initCvt(fwdPD->src_primitive_desc(), dnnCvtUser2Intl);
    if (useMkldnnWgt_) {
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
          memcpy(dst, tmp->getIntlData(), tmp->getIntlSize());
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
      if (!hasCvtBiasData_) {
        hasCvtBiasData_ = true;
        CHECK(dataBias_->getUserPD() == fwdPD->bias_primitive_desc())
          << "should always be format::x, or changed in new mkldnn version";
        dataBias_->initCvt(dataBias_->getUserPD(), dnnCvtNoNeed);
      } else {
        CHECK(dataBias_->getIntlPD() == fwdPD->bias_primitive_desc())
          << "all bias formats should equal";
      }
    }
    // cvt topdata buffer only once, set dnn MemDesc if next is also mkldnn
    if (!hasCvtTopData_) {
      hasCvtTopData_ = true;
      if (setDnnTopDataFmt_) {
        dataTop_->resetUser(topData, fwdPD->dst_primitive_desc());
        setTopDataMD(dataTop_->getUserMD());
        VLOG(4) << "set next data fmt: " << DNN_FMTS[dataTop_->getUserFmt()];
      }
      dataTop_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);
    } else {
      CHECK(dataTop_->getIntlPD() == fwdPD->dst_primitive_desc())
        << "all output formats should equal";
    }
    // 5. create fwd handle
    if (hasBias) {
      fwd_.reset(new inner_product_forward(*fwdPD,
        *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
        *(dataBias_->getIntlMem()), *(dataTop_->getIntlMem())));
    } else {
      fwd_.reset(new inner_product_forward(*fwdPD,
        *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
        *(dataTop_->getIntlMem())));
    }
    if (dataWgt_) {
      VLOG(3) << "weight data flow --- "
        << DNN_FMTS[dataWgt_->getUserFmt()]
        << " >>> "
        << DNN_FMTS[dataWgt_->getIntlFmt()];
    }
  }
}

void MkldnnFcLayer::resetDnnBwd() {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  prop_kind pk = prop_kind::forward;
  bool hasBias = (biases_ && biases_->getWGrad());

  hasCvtTopDiff_ = false;
  hasCvtBiasDiff_ = false;

  // 1. create mkldnn buffer, only have one output and bias buffer
  diffTop_.reset(new MkldnnBuffer());
  if (hasBias) {
    diffBias_.reset(new MkldnnBuffer());
  }
  // 2. init user top and bias
  real *topDiff = getOutputGrad()->getData();
  diffTop_->initUser(topDiff, topDims_, topFmt_, eg);
  if (hasBias) {
    real* biasDiff = biases_->getWGrad()->getData();
    diffBias_->initUser(biasDiff, biasDims_[0], biasFmt_[0], eg);
  }
  // use internal top diff if use dnn input
  const std::shared_ptr<mkldnn::memory::desc> prv = getTopDiffMD();
  if (prv) {
    diffTop_->resetUser(topDiff, *prv, eg);
    bool isNCHW = diffTop_->getUserFmt() == memory::format::nchw;
    if (isNCHW && oh_[0] == ow_[0] && oh_[0] == 1) {
      // if prv is nchw and h==w==1, use nc instead
      diffTop_->resetUser(topDiff, topDims_, memory::format::nc, eg);
      VLOG(4) << "use nc diff fmt";
    } else {
      VLOG(4) << "use prev diff fmt: " << DNN_FMTS[diffTop_->getUserFmt()];
    }
  }
  // TODO(TJ): only care about i==0 yet
  CHECK_EQ(inputLayers_.size(), 1);
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    // 1. create mkldnn buffer and init user
    CHECK(weights_[i]->getWGrad()) << "should have weight anyway";
    diffWgt_.reset(new MkldnnBuffer());
    real *wgtDiff = useMkldnnWgt_ ? weights_[i]->getWGrad()->getData()
      : selfWgtDiff_[i]->getData();
    diffWgt_->initUser(wgtDiff, wgtDims_[i], wgtFmt_[i], eg);
    // 2. prepare backward weight and bias
    std::shared_ptr<inner_product_forward::desc> bwdFwdDesc;
    std::shared_ptr<inner_product_forward::primitive_desc> bwdFwdPD;
    std::shared_ptr<inner_product_backward_weights::desc> bwdWgtDesc;
    std::shared_ptr<inner_product_backward_weights::primitive_desc> bwdWgtPD;
    bwdFwdDesc.reset(new inner_product_forward::desc(pk,
      // poor any policy for FC bwd, so use wgt data internal format
      dataBot_->getIntlMD(), dataWgt_->getIntlMD(),
      prv ? diffTop_->getUserMD() : dataTop_->getIntlMD()));
    bwdFwdPD.reset(new inner_product_forward::primitive_desc(
      *bwdFwdDesc, eg));
    CHECK(hasBias) << "only support with bias in mkldnn";
    bwdWgtDesc.reset(new inner_product_backward_weights::desc(
      dataBot_->getIntlMD(), dataWgt_->getIntlMD(),
      dataBias_->getIntlMD(),
      prv ? diffTop_->getUserMD() : dataTop_->getIntlMD()));
    bwdWgtPD.reset(new inner_product_backward_weights::primitive_desc(
      *bwdWgtDesc, eg, *bwdFwdPD));
    CHECK(dataBot_->getIntlPD() == bwdWgtPD->src_primitive_desc());
    CHECK(dataWgt_->getIntlPD() == bwdWgtPD->diff_weights_primitive_desc());
    CHECK(dataBias_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc());

    // 3. init conversion
    if (useMkldnnWgt_) {
      wgtDiff = weights_[i]->getWGrad()->getData();
      diffWgt_->resetUser(wgtDiff, dataWgt_->getIntlPD());
      diffWgt_->initCvt(diffWgt_->getUserPD(), dnnCvtNoNeed);
      CHECK_EQ(diffWgt_->getIntlSize(), dataWgt_->getIntlSize())
        << "can not use mkldnn wgt since memory size does not equal";
    } else {
      diffWgt_->initCvt(
        bwdWgtPD->diff_weights_primitive_desc(), dnnCvtIntl2User);
    }
    if (hasBias) {
      if (!hasCvtBiasDiff_) {
        hasCvtBiasDiff_ = true;
        CHECK(diffBias_->getUserPD() == bwdWgtPD->diff_bias_primitive_desc())
          << "should always be format::x, or changed in new mkldnn version";
        diffBias_->initCvt(diffBias_->getUserPD(), dnnCvtNoNeed);
      } else {
        CHECK(diffBias_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc())
          << "all bias formats should equal";
      }
    }
    if (!hasCvtTopDiff_) {
      hasCvtTopDiff_ = true;
      diffTop_->initCvt(bwdWgtPD->diff_dst_primitive_desc(), dnnCvtUser2Intl);
    } else {
      CHECK(diffTop_->getIntlPD() == bwdWgtPD->diff_dst_primitive_desc())
        << "all topdiff formats should equal";
    }
    // 4. bias backward can only be executed in weight backward with MKL-DNN
    bwdWgt_.reset(new inner_product_backward_weights(*bwdWgtPD,
      *(dataBot_->getIntlMem()), *(diffTop_->getIntlMem()),
      *(diffWgt_->getIntlMem()), *(diffBias_->getIntlMem())));
    if (diffWgt_) {
      VLOG(3) << "weight diff flow --- "
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
    real* botDiff = prevLayer->getOutputGrad()->getData();
    diffBot_.reset(new MkldnnBuffer());
    diffBot_->initUser(botDiff, botDims_[i], botFmt_[i], eg);
    // 2. init backward data primitive desc
    std::shared_ptr<inner_product_backward_data::desc> bwdDataDesc;
    std::shared_ptr<inner_product_backward_data::primitive_desc> bwdDataPD;
    bwdDataDesc.reset(new inner_product_backward_data::desc(
      // since fc have pool policy to choose best format, so data intlMD
      dataBot_->getIntlMD(),
      dataWgt_->getIntlMD(),
      diffTop_->getIntlMD()));
    bwdDataPD.reset(new inner_product_backward_data::primitive_desc(
      *bwdDataDesc, eg, *bwdFwdPD));
    CHECK(dataBot_->getIntlPD() == bwdDataPD->diff_src_primitive_desc());
    CHECK(dataWgt_->getIntlPD() == bwdDataPD->weights_primitive_desc());
    CHECK(diffTop_->getIntlPD() == bwdDataPD->diff_dst_primitive_desc());
    // 3. init conversion
    if (setDnnBotDiffFmt_[i]) {
      diffBot_->resetUser(botDiff, bwdDataPD->diff_src_primitive_desc());
      prevLayer->setTopDiffMD(diffBot_->getUserMD());
      VLOG(4) << "set next diff fmt: " << DNN_FMTS[diffBot_->getUserFmt()];
    }
    diffBot_->initCvt(bwdDataPD->diff_src_primitive_desc(), dnnCvtIntl2User);
    // 4. create bwd data handle
    bwdData_.reset(new inner_product_backward_data(
      *bwdDataPD, *(diffTop_->getIntlMem()),
      *(dataWgt_->getIntlMem()), *(diffBot_->getIntlMem())));
  }
}

void MkldnnFcLayer::submitDnnFwd(PassType passType) {
  real *topdata = getOutputValue()->getData();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(getInput(i).value) << "The input of 'fc' layer must be matrix";
    real *botdata = getPrev(0)->getOutputValue()->getData();
    std::vector<primitive> pipeline;
    dataBot_->submitCvt(pipeline, botdata);
    if (!useMkldnnWgt_ && passType != PASS_TEST) {
      weights_[i]->getW()->transpose(selfWgtData_[i], false);
      real *wgtdata = selfWgtData_[i]->getData();
      dataWgt_->submitCvt(pipeline, wgtdata);
    }  // else do not need cvt wgt
    pipeline.push_back(*fwd_);
    dataTop_->submitCvt(pipeline, topdata);
    stream(stream::kind::eager).submit(pipeline).wait();
  }
  forwardActivation();
}

void MkldnnFcLayer::submitBwdData(int idx, const MatrixPtr& botGrad) {
  if (botGrad == NULL) {
    return;
  }
  real* botdiff = botGrad->getData();
  real* topdiff = getOutputGrad()->getData();
  std::vector<primitive> pipeline;
  if (!useMkldnnWgt_) {  // no need cvt wgt without useMkldnnWgt_
    CHECK(selfWgtData_[idx]);
    real* wgtdata = selfWgtData_[idx]->getData();
    dataWgt_->submitCvt(pipeline, wgtdata);
  }
  diffTop_->submitCvt(pipeline, topdiff);
  pipeline.push_back(*bwdData_);
  diffBot_->submitCvt(pipeline, botdiff);
  stream(stream::kind::eager).submit(pipeline).wait();
}

void MkldnnFcLayer::submitBwdWgts(int idx, const MatrixPtr& botVal) {
  real* botdata = botVal->getData();
  real* topdiff = getOutputGrad()->getData();
  real* wgtdiff = weights_[idx]->getWGrad()->getData();
  if (!useMkldnnWgt_) {
    CHECK(selfWgtDiff_[idx]);
    wgtdiff = selfWgtDiff_[idx]->getData();
  }
  std::vector<primitive> pipeline;
  diffTop_->submitCvt(pipeline, topdiff);
  dataBot_->submitCvt(pipeline, botdata);
  pipeline.push_back(*bwdWgt_);
  diffWgt_->submitCvt(pipeline, wgtdiff);
  // no need to submit cvt bias since biasfmt would not be changed
  stream(stream::kind::eager).submit(pipeline).wait();

  if (!useMkldnnWgt_) {
    // save to actual weight param
    selfWgtDiff_[idx]->transpose(weights_[idx]->getWGrad_mutable(), false);
  }
}

void MkldnnFcLayer::submitDnnBwd(const UpdateCallback &callback) {
  backwardActivation();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    submitBwdData(i, getPrev(i)->getOutputGrad());
    if (weights_[i]->getWGrad()) {
      submitBwdWgts(i, getPrev(i)->getOutputValue());
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
