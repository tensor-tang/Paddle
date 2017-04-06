/* Copyright (c) 2017 */

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnLRNLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_lrn, MkldnnLRNLayer);

bool MkldnnLRNLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* the size of inputs for norm-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  k_ = 1.0;  // TODO: what meaning?

  const NormConfig& conf = config_.inputs(0).norm_conf();
  const std::string& normType = conf.norm_type();
  if (normType == "across_channel") {
    algo_ = algorithm::lrn_across_channels;
  } else if (normType == "within_channel") {
    algo_ = algorithm::lrn_within_channel;
  } else {
    LOG(FATAL) << "unknow LRN type!";
  }
  localSize_= conf.size();
  alpha_ = conf.scale();
  beta_ = conf.pow();
  
  ic_[0] = conf.channels();
  iw_[0] = conf.img_size();
  ow_[0] = conf.output_x();
  CHECK_EQ(iw_[0], ow_[0]);
  ih_[0] = iw_[0];
  ih_[0] = oh_[0];

  bs_ = 0;
  oc_ = ic_[0];
 
  return true;
}

void MkldnnLRNLayer::reshape() {
  CHECK_EQ(inputLayers_.size(), 1UL);

  int height = inputLayers_[0]->getOutput().getFrameHeight();
  int width = inputLayers_[0]->getOutput().getFrameWidth();
  if (height != 0) ih_[0] = height;
  if (width != 0) iw_[0] = width;
  oh_[0] = ih_[0];
  ow_[0] = iw_[0];

  // reset output image size
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);
}

void MkldnnLRNLayer::resetDnnFwd(PassType passType) {
  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  prop_kind pk = (passType == PASS_TEST) ? prop_kind::forward_scoring :
    prop_kind::forward_training;
  // create dim structure that describes user data.
  botDims_[0] = {bs_, ic_[0], ih_[0], iw_[0]};
  botFmt_[0] = memory::format::nchw;
  topDims_ = {bs_, oc_, oh_[0], ow_[0]};
  topFmt_ = memory::format::nchw;

  // 1. create buffer
  botDatas_[0].reset(new MkldnnBuffer());
  topData_.reset(new MkldnnBuffer());
  // 2. init user
  real *botDataData = getPrev(0)->getOutputValue()->getData();
  real *topDataData = getOutputValue()->getData();
  botDatas_[0]->initUser(botDataData, botDims_[0], botFmt_[0], eg);
  topData_->initUser(topDataData, topDims_, topFmt_, eg);
  const std::shared_ptr<memory::desc>& prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    botDatas_[0]->resetUser(botDataData, *prvMD, eg);
    bool isNC = botDatas_[0]->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(ih_[0] == iw_[0] && ih_[0] == 1)
        << "iw, ih must be 1 with nc input";
      // do not support nc input, so change to nchw or nChw8c
      memory::format fmt = memory::format::nchw;
      botDatas_[0]->resetUser(botDataData, botDims_[0], fmt, eg);
      VLOG(4) << "use nchw data fmt";
    } else {
      VLOG(4) << "use prev data fmt: " << DNN_FMTS[botDatas_[0]->getUserFmt()];
    }
  }
  // 3. create forward PD
  std::shared_ptr<lrn_forward::desc> fwdDesc;
  fwdDesc.reset(new lrn_forward::desc(pk, algo_,
    // TODO(TJ): use any if MKLDNN ready
    // the src format do not accept any format yet
    prvMD ? botDatas_[0]->getUserMD()
          : MkldnnBuffer::getMD(botDims_[0], botFmt_[0]),
    localSize_, alpha_, beta_, k_));
  fwdPD_.reset(new lrn_forward::primitive_desc(*fwdDesc, eg));
  // 4. init cvt
  botDatas_[0]->initCvt();  // dnnCvtNoNeed
  // set topDataData dnn MemDesc if next is also mkldnn
  if (nextIsDnn_) {
    topData_->resetUser(topDataData, fwdPD_->dst_primitive_desc());
    setTopDataMD(topData_->getUserMD());
    VLOG(4) << "set next data format: " << DNN_FMTS[topData_->getUserFmt()];
  }
  topData_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtIntl2User);
  // 5. create handle
  if (passType != PASS_TEST) {  // training and grad check
    workspace_.reset(new memory(fwdPD_->workspace_primitive_desc()));
    fwd_.reset(new lrn_forward(*fwdPD_, *(botDatas_[0]->getIntlMem()),
      *workspace_, *(topData_->getIntlMem())));
  } else {
    fwd_.reset(new lrn_forward(*fwdPD_, *(botDatas_[0]->getIntlMem()),
      *(topData_->getIntlMem())));
  }
}

void MkldnnLRNLayer::resetDnnBwd() {
  if (algo_ == algorithm::lrn_within_channel)
    LOG(FATAL) << "not support backward within channel yet";
  mkldnn::engine eg = CpuEngine::Instance().getEngine();

  // 1. create buffer
  botDiffs_[0].reset(new MkldnnBuffer());
  topDiff_.reset(new MkldnnBuffer());
  // 2. init user
  real *topDiffData = getDnnOutputGrad()->getData();
  real* botDiffData = getDnnInputGrad(0)->getData();
  botDiffs_[0]->initUser(botDiffData, botDatas_[0]->getUserMD(), eg);
  topDiff_->initUser(topDiffData, topData_->getUserMD(), eg);
  const std::shared_ptr<mkldnn::memory::desc>& prvMD = getTopDiffMD();
  if (prvMD) {
    topDiff_->resetUser(topDiffData, *prvMD, eg);
    bool isNC = topDiff_->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(oh_[0] == ow_[0] && oh_[0] == 1)
        << "ow, oh must be 1 with nc input";
      // do not support nc input, so change to nchw
      memory::format fmt = memory::format::nchw;
      topDiff_->resetUser(topDiffData, topDims_, fmt, eg);
      VLOG(4) << "use nchw diff fmt";
    } else {
      VLOG(4) << "use prev diff fmt: " << DNN_FMTS[topDiff_->getUserFmt()];
    }
  }
  // 3. create bwd PD
  std::shared_ptr<lrn_backward::desc> bwdDesc;
  std::shared_ptr<lrn_backward::primitive_desc> bwdPD;
  bwdDesc.reset(new lrn_backward::desc(algo_,
    botDatas_[0]->getIntlMD(),
    topDiff_->getUserMD(), // TODO(TJ): use any MD if MKLDNN ready
    localSize_, alpha_, beta_, k_));
  bwdPD.reset(new lrn_backward::primitive_desc(
    *bwdDesc, eg, *fwdPD_));
  // 4. init cvt
  if (prevIsDnn_[0]) {
    botDiffs_[0]->resetUser(botDiffData, bwdPD->diff_src_primitive_desc());
    getPrev(0)->setTopDiffMD(this->getName(), botDiffs_[0]->getUserMD());
    VLOG(4) << "set next diff format: " << DNN_FMTS[botDiffs_[0]->getUserFmt()];
  }
  botDiffs_[0]->initCvt(bwdPD->diff_src_primitive_desc(), dnnCvtIntl2User);
  topDiff_->initCvt();
  // 5. create bwd handle
  CHECK(workspace_);
  bwd_.reset(new lrn_backward(*bwdPD, *(botDatas_[0]->getIntlMem()),
    *(topDiff_->getIntlMem()), *workspace_, *(botDiffs_[0]->getIntlMem())));

}

void MkldnnLRNLayer::submitDnnFwd(PassType passType) {
  real *botDataData = getPrev(0)->getOutputValue()->getData();
  real *topDataData = getOutputValue()->getData();

  std::vector<primitive> pipeline;
  botDatas_[0]->submitCvt(pipeline, botDataData);
  pipeline.push_back(*fwd_);
  topData_->submitCvt(pipeline, topDataData);
  stream(stream::kind::eager).submit(pipeline).wait();

//  not activation for LRN
}

void MkldnnLRNLayer::submitDnnBwd(const UpdateCallback &callback) {
  (void)callback;
  if (nullptr == getInputGrad(0)) {
    return;
  }

  real* botDiffData = getDnnInputGrad(0)->getData();
  real* topDiffData = getOutputGrad()->getData();
  std::vector<primitive> pipeline;
  topDiff_->submitCvt(pipeline, topDiffData);
  pipeline.push_back(*bwd_);
  botDiffs_[0]->submitCvt(pipeline, botDiffData);
  stream(stream::kind::eager).submit(pipeline).wait();
}

}  // namespace paddle
