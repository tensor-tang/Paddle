/* Copyright (c) 2017 */

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnPoolLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_pool, MkldnnPoolLayer);

bool MkldnnPoolLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* the size of inputs for pool-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  const PoolConfig& conf = config_.inputs(0).pool_conf();
//  if (!conf.caffe_mode()) {
//    LOG(FATAL) << "Only support caffe mode with MKL-DNN by now!";
//  }
  const std::string& poolType_ = conf.pool_type();
  if (poolType_ == "max-projection") {
    poolAlgo_ = algorithm::pooling_max;
  } else if (poolType_ == "avg-projection") {
    poolAlgo_ = algorithm::pooling_avg;
  } else {
    LOG(FATAL) << "unknow pooling type!";
  }
  if (config_.has_add_size()) {
    addSize_ = config_.add_size();
  }
  if (config_.has_use_mkldnn_wgt()) {
    useMkldnnWgt_ = config_.use_mkldnn_wgt();
  }

  bs_ = 0;
  ic_[0] = conf.channels();
  oc_ = ic_[0];
  iw_[0] = conf.img_size();
  ow_[0] = conf.output_x();
  ih_[0] = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  oh_[0] = conf.has_output_y() ? conf.output_y() : conf.output_x();

  fw_ = conf.size_x();
  sw_ = conf.stride();
  pw_ = conf.padding();

  fh_ = conf.has_size_y() ? conf.size_y() : conf.size_x();
  sh_ = conf.has_stride_y() ? conf.stride_y() : conf.stride();
  ph_ = conf.has_padding_y() ? conf.padding_y() : conf.padding();

  return true;
}

void MkldnnPoolLayer::clearDataDiff() {
//  reserveOutput(bs_, getSize());
}

void MkldnnPoolLayer::reshape() {
  // reshape input and output size
  CHECK_NE(inputLayers_.size(), 0UL);
  int height = inputLayers_[0]->getOutput().getFrameHeight();
  int width = inputLayers_[0]->getOutput().getFrameWidth();
  if (height != 0) ih_[0] = height;
  if (width != 0) iw_[0] = width;
  oh_[0] = outputSize(ih_[0], fh_, ph_, sh_, false);
  ow_[0] = outputSize(iw_[0], fw_, pw_, sw_, false);

  // reset output image size
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);
}

void MkldnnPoolLayer::resetDnnFwd(PassType passType) {
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
  memory::dims kernel = {fh_, fw_};
  memory::dims strides = {sh_, sw_};
  memory::dims padding = {ph_, pw_};
  padding_kind padKind = padding_kind::zero;
  std::vector<int> padR = {ph_, pw_};
  for (int k = 0; k < 2; ++k) {
    if ((ih_[0] + ph_ + padR[0] - fh_)/sh_ + 1 < oh_[0]) ++padR[0];
    if ((iw_[0] + pw_ + padR[1] - fw_)/sw_ + 1 < ow_[0]) ++padR[1];
  }
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
  std::shared_ptr<pooling_forward::desc> fwdDesc;
  fwdDesc.reset(new pooling_forward::desc(pk, poolAlgo_,
  // TODO(TJ): use any if MKLDNN ready
    // the src format do not accept any format yet
    prvMD ? botDatas_[0]->getUserMD()
    : MkldnnBuffer::getMD(botDims_[0], botFmt_[0]),
    MkldnnBuffer::getMD(topDims_),
    strides, kernel, padding, padR, padKind));
  fwdPD_.reset(new pooling_forward::primitive_desc(*fwdDesc, eg));
  // 4. init cvt
  botDatas_[0]->initCvt();  // dnnCvtNoNeed
  // set topDataData dnn MemDesc if next is also mkldnn
  if (nextIsDnn_) {
    topData_->resetUser(topDataData, fwdPD_->dst_primitive_desc());
    setTopDataMD(topData_->getUserMD());
    VLOG(4) << "set next data format: " << DNN_FMTS[topData_->getUserFmt()];
  }
  topData_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtIntl2User);
  withWorkspace_ = (passType != PASS_TEST
      && poolAlgo_ != algorithm::pooling_avg);
  if (withWorkspace_) {
    workspace_.reset(new memory(fwdPD_->workspace_primitive_desc()));
  } else {
    auto p_workspace_desc = memory::primitive_desc(
      {{}, memory::data_type::f32, memory::format(topData_->getIntlFmt())},
      eg);
    workspace_.reset(new memory(p_workspace_desc));
  }
  // 5. create handle
  if (withWorkspace_) {
    fwd_.reset(new pooling_forward(*fwdPD_,
      *(botDatas_[0]->getIntlMem()), *(topData_->getIntlMem()),
      *workspace_));
  } else {
    fwd_.reset(new pooling_forward(*fwdPD_,
      *(botDatas_[0]->getIntlMem()), *(topData_->getIntlMem())));
  }
}

void MkldnnPoolLayer::resetDnnBwd() {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  // create dim structure that describes user data.
  memory::dims kernel = {fh_, fw_};
  memory::dims strides = {sh_, sw_};
  memory::dims padding = {ph_, pw_};
  padding_kind padKind = padding_kind::zero;
  std::vector<int> padR = {ph_, pw_};
  for (int k = 0; k < 2; ++k) {
    if ((ih_[0] + ph_ + padR[0] - fh_)/sh_ + 1 < oh_[0]) ++padR[0];
    if ((iw_[0] + pw_ + padR[1] - fw_)/sw_ + 1 < ow_[0]) ++padR[1];
  }
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
  std::shared_ptr<pooling_backward::desc> bwdDesc;
  std::shared_ptr<pooling_backward::primitive_desc> bwdPD;
  bwdDesc.reset(new pooling_backward::desc(poolAlgo_,
    MkldnnBuffer::getMD(botDims_[0]),
    // TODO(TJ): use any if MKLDNN ready
    // use intl topdata format, since bwd prev is from fc's output: always nchw
    // which is not best format for pooling
    MkldnnBuffer::getMD(topDims_, memory::format(topData_->getIntlFmt())),
    strides, kernel, padding, padR, padKind));
  bwdPD.reset(new pooling_backward::primitive_desc(
    *bwdDesc, eg, *fwdPD_));
  // 4. init cvt
  if (prevIsDnn_[0]) {
    botDiffs_[0]->resetUser(botDiffData, bwdPD->diff_src_primitive_desc());
    getPrev(0)->setTopDiffMD(this->getName(), botDiffs_[0]->getUserMD());
    VLOG(4) << "set next diff format: " << DNN_FMTS[botDiffs_[0]->getUserFmt()];
  }
  botDiffs_[0]->initCvt(bwdPD->diff_src_primitive_desc(), dnnCvtIntl2User);
  topDiff_->initCvt(topData_->getIntlPD(), dnnCvtUser2Intl);
  // 5. create bwd handle
  if (withWorkspace_) {
    bwd_.reset(new pooling_backward(*bwdPD,
      *(topDiff_->getIntlMem()), *workspace_, *(botDiffs_[0]->getIntlMem())));
  } else {
    bwd_.reset(new pooling_backward(*bwdPD,
      *(topDiff_->getIntlMem()), *(botDiffs_[0]->getIntlMem())));
  }
}

void MkldnnPoolLayer::submitDnnFwd(PassType passType) {
  real *botDataData = getPrev(0)->getOutputValue()->getData();
  real *topDataData = getOutputValue()->getData();

  std::vector<primitive> pipeline;
  botDatas_[0]->submitCvt(pipeline, botDataData);
  pipeline.push_back(*fwd_);
  topData_->submitCvt(pipeline, topDataData);
  stream(stream::kind::eager).submit(pipeline).wait();

//  kepp as paddle do not forward activation
//  forwardActivation();
}

void MkldnnPoolLayer::submitDnnBwd(const UpdateCallback &callback) {
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
