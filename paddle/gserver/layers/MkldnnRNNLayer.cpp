/* Copyright (c) 2017 */


#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnRNNLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_rnn, MkldnnRNNLayer);

bool MkldnnRNNLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  // TODO(TJ): initialize the weight

  return true;
}

void MkldnnRNNLayer::loadConfig() {
  CHECK_EQ(inputLayers_.size(), 1U);
  if (config_.has_use_mkldnn_wgt()) {
    useMkldnnWgt_ = config_.use_mkldnn_wgt();
  }
  const RNNConfig &conf = config_.inputs(0).rnn_conf();
  inputMode_ = conf.input_mode();
  algKind_ = conf.alg_kind();
  useBiDir_ = conf.use_bi_direction();
  layerNum_ = conf.layer_num();
  if (useBiDir_) {
    CHECK(conf.has_sum_output())
      << "should have set output mode: sum or concat";
    sumOutput_ = conf.sum_output();
  }
}

void MkldnnRNNLayer::reshapeOutput() {
  size_t idx = 0;
  // reshape bs and mkl seqlen
  outputMatH_ = inputMatH_;
  seqLen_ = getInput(idx).getMklSeqLen();
  CHECK_GE(seqLen_, 1) << "seq length should larger than 1";
  
  if (seqLen_ > 1) {
    bs_ = outputMatH_ / seqLen_;
    CHECK_EQ(bs_ * seqLen_, outputMatH_) << "maybe caused by un-divisible";
  } else {
    bs_ = outputMatH_;
  }
  // reshape image size
  ih_ = inputLayers_[idx]->getOutput().getFrameHeight();
  iw_ = inputLayers_[idx]->getOutput().getFrameWidth();
  if (ih_ == 0) ih_ = 1;
  if (iw_ == 0) iw_ = 1;
  ic_ = inputMatW_ / (ih_ * iw_);
  CHECK_EQ(ic_ * ih_ * iw_, inputMatW_) << "maybe caused by un-divisible";
  oc_ = ic_;
  oh_ = ih_;
  ow_ = iw_;
  outputMatW_ = inputMatW_;
  config_.set_size(outputMatW_);

  // reset output image size
  resetOutput(outputMatH_, outputMatW_);
  getOutput().setFrameHeight(oh_);
  getOutput().setFrameWidth(ow_);
}

void MkldnnRNNLayer::resetDnnFwd() {
}

void MkldnnRNNLayer::resetDnnBwd() {
}

void MkldnnRNNLayer::submitDnnFwd() {
  real *topDataData = getOutputValue()->getData();
  MatrixPtr tmp = Matrix::create(topDataData, outputMatH_, outputMatW_,
    false, false);
  // TODO(TJ): replace it with rnn fwd
  tmp->assign(*getInputValue(0));
}

void MkldnnRNNLayer::submitDnnBwd(const UpdateCallback &callback) {
  const MatrixPtr& botGrad = getDnnInputGrad(0);
  if (!botGrad) {
    return;
  }

  MatrixPtr tmp = Matrix::create(
    botGrad->getData(), inputMatH_, inputMatW_, false, false);
  // TODO(TJ): replace it with rnn bwd
  tmp->add(*getOutputGrad());
}

}  // namespace paddle
