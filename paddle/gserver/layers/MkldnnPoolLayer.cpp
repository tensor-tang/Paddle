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
  if (config_.has_use_mkldnn_fmt()) {
    useMkldnnFmt_ = config_.use_mkldnn_fmt();
  }
  ic_.push_back(conf.channels());
  iw_.push_back(conf.img_size());
  ow_.push_back(conf.output_x());
  ih_.push_back(conf.has_img_size_y() ? conf.img_size_y() : conf.img_size());
  oh_.push_back(conf.has_output_y() ? conf.output_y() : conf.output_x());

  fw_ = conf.size_x();
  sw_ = conf.stride();
  pw_ = conf.padding();

  fh_ = conf.has_size_y() ? conf.size_y() : conf.size_x();
  sh_ = conf.has_stride_y() ? conf.stride_y() : conf.stride();
  ph_ = conf.has_padding_y() ? conf.padding_y() : conf.padding();
  bs_ = 0;
  oc_ = ic_[0];
  return true;
}

void MkldnnPoolLayer::clearDataDiff() {
  reserveOutput(bs_, getSize());
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

  printInfo();
}

void MkldnnPoolLayer::resetDnnFwd(PassType passType) {
  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  prop_kind pk = (passType == PASS_TEST) ? prop_kind::forward_scoring :
    prop_kind::forward_training;
  // create dim structure that describes user data.
  memory::dims botDims = {bs_, ic_[0], ih_[0], iw_[0]};
  memory::dims kernel = {fh_, fw_};
  memory::dims strides = {sh_, sw_};
  memory::dims padding = {ph_, pw_};
  memory::dims topDims = {bs_, oc_, oh_[0], ow_[0]};
  padding_kind padKind = padding_kind::zero;
  memory::format fmt = memory::format::nchw;
  std::vector<int> padR = {ph_, pw_};
  // TODO(TJ): uncomment it, wait for mkldnn update
  // temporary skip it for googlenet, for better performance
//  for (int k = 0; k < 2; ++k) {
//    if ((ih_[0] + ph_ + padR[0] - fh_)/sh_ + 1 < oh_[0]) ++padR[0];
//    if ((iw_[0] + pw_ + padR[1] - fw_)/sw_ + 1 < ow_[0]) ++padR[1];
//  }
  // 1. create buffer
  dataBot_.reset(new MkldnnBuffer());
  dataTop_.reset(new MkldnnBuffer());
  // 2. init user
  real *botData = getPrev(0)->getOutputValue()->getData();
  real *topData = getOutputValue()->getData();
  dataBot_->initUser(botData, botDims, fmt, eg);
  dataTop_->initUser(topData, topDims, fmt, eg);
  const std::shared_ptr<memory::desc> prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    dataBot_->resetUser(botData, *prvMD, eg);
    LOG(INFO) << "use prev data format: " << DNN_FMTS[dataBot_->getUserFmt()];
  }
  // 3. create forward PD
  std::shared_ptr<pooling_forward::desc> fwdDesc;
  fwdDesc.reset(new pooling_forward::desc(pk, poolAlgo_,
    // since pool have pool policy to choose best format, so depends on prv
    prvMD ? dataBot_->getUserMD() : getAnyMD(botDims),
    getAnyMD(topDims),
    strides, kernel, padding, padR, padKind));
  fwdPD_.reset(new pooling_forward::primitive_desc(*fwdDesc, eg));
  // 4. init cvt
  dataBot_->initCvt();  // dnnCvtNoNeed
  // set topdata dnn MemDesc if next is also mkldnn
  if (setDnnTopDataFmt_) {
    dataTop_->resetUser(topData, fwdPD_->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
    LOG(INFO) << "set next data format: " << DNN_FMTS[dataTop_->getUserFmt()];
  }
  dataTop_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtIntl2User);
  withWorkspace_ = (passType != PASS_TEST
      && poolAlgo_ != algorithm::pooling_avg);
  if (withWorkspace_) {
    workspace_.reset(new memory(fwdPD_->workspace_primitive_desc()));
  } else {
    auto p_workspace_desc = memory::primitive_desc(
      {{}, memory::data_type::f32, memory::format(dataTop_->getIntlFmt())},
      eg);
    workspace_.reset(new memory(p_workspace_desc));
  }
  // 5. create handle
  if (withWorkspace_) {
    fwd_.reset(new pooling_forward(*fwdPD_,
      *(dataBot_->getIntlMem()), *(dataTop_->getIntlMem()),
      *workspace_));
  } else {
    fwd_.reset(new pooling_forward(*fwdPD_,
      *(dataBot_->getIntlMem()), *(dataTop_->getIntlMem())));
  }
}

void MkldnnPoolLayer::exFwd(PassType passType) {
  const Argument& in = getInput(0);
  const Argument& out = output_;
  CHECK_EQ(getSize(), out.value->getWidth());
  MatrixPtr inputV = in.value;
  MatrixPtr outV = out.value;
  outV->maxPoolForward(*inputV, ih_[0], iw_[0], ic_[0], fw_, fh_,
                       sh_, sw_, oh_[0], ow_[0], ph_, pw_);
  
}

void MkldnnPoolLayer::resetDnnBwd() {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  // create dim structure that describes user data.
  memory::dims botDims = {bs_, ic_[0], ih_[0], iw_[0]};
  memory::dims kernel = {fh_, fw_};
  memory::dims strides = {sh_, sw_};
  memory::dims padding = {ph_, pw_};
  memory::dims topDims = {bs_, oc_, oh_[0], ow_[0]};
  padding_kind padKind = padding_kind::zero;
  std::vector<int> padR = {ph_, pw_};
  // TODO(TJ): uncomment it, wait for mkldnn update
  // temporary skip it for googlenet, for better performance
//  for (int k = 0; k < 2; ++k) {
//    if ((ih_[0] + ph_ + padR[0] - fh_)/sh_ + 1 < oh_[0]) ++padR[0];
//    if ((iw_[0] + pw_ + padR[1] - fw_)/sw_ + 1 < ow_[0]) ++padR[1];
//  }
  // 1. create buffer
  diffBot_.reset(new MkldnnBuffer());
  diffTop_.reset(new MkldnnBuffer());
  // 2. init user
  real *topDiff = getOutputGrad()->getData();
  real* botDiff = getPrev(0)->getOutputGrad()->getData();
  diffBot_->initUser(botDiff, dataBot_->getUserMD(), eg);
  diffTop_->initUser(topDiff, dataTop_->getUserMD(), eg);
  const std::shared_ptr<mkldnn::memory::desc> inputDiffMD = getTopDiffMD();
  if (inputDiffMD) {
    diffTop_->resetUser(topDiff, *inputDiffMD, eg);
    LOG(INFO) << "use prev diff format: " << DNN_FMTS[diffTop_->getUserFmt()];
  }
  if (setDnnBotDiffFmt_[0]) {
    diffBot_->resetUser(botDiff, dataBot_->getIntlPD());
    getPrev(0)->setTopDiffMD(diffBot_->getUserMD());
    LOG(INFO) << "set next diff format: " << DNN_FMTS[diffBot_->getUserFmt()];
  }
  // 3. create bwd PD
  std::shared_ptr<pooling_backward::desc> bwdDesc;
  std::shared_ptr<pooling_backward::primitive_desc> bwdPD;
  bwdDesc.reset(new pooling_backward::desc(
    poolAlgo_, diffBot_->getUserMD(), diffTop_->getUserMD(),
    strides, kernel, padding, padR, padKind));
  bwdPD.reset(new pooling_backward::primitive_desc(
    *bwdDesc, eg, *fwdPD_));
  // 4. init cvt
  diffTop_->initCvt();
  diffBot_->initCvt();
  // 5. create bwd handle
  if (withWorkspace_) {
    bwd_.reset(new pooling_backward(*bwdPD,
      *(diffTop_->getIntlMem()), *workspace_, *(diffBot_->getIntlMem())));
  } else {
    bwd_.reset(new pooling_backward(*bwdPD,
      *(diffTop_->getIntlMem()), *(diffBot_->getIntlMem())));
  }
}

void MkldnnPoolLayer::submitDnnFwd(PassType passType) {
//  exFwd(passType);

  real *botdata = getPrev(0)->getOutputValue()->getData();
  real *topdata = getOutputValue()->getData();

  std::vector<primitive> pipeline;
  dataBot_->submitCvt(pipeline, botdata);
  pipeline.push_back(*fwd_);
  dataTop_->submitCvt(pipeline, topdata);

//  LOG(INFO) << "------------ ex top data:" << topdata[0] << "," << topdata[1];

  stream(stream::kind::eager).submit(pipeline).wait();
//  LOG(INFO) << "------------ my top data:" << topdata[0]<< "," << topdata[1];

//  as paddle no forward activation
//  forwardActivation();
}

void MkldnnPoolLayer::exBwd(const UpdateCallback &callback) {
  
  const Argument& in = getInput(0);
  MatrixPtr outGrad = getOutputGrad();
  MatrixPtr inputV = in.value;
  MatrixPtr outV = getOutputValue();
  MatrixPtr inputGrad = in.grad;

  if (nullptr == getInputGrad(0)) {
    return;
  }

  inputGrad->maxPoolBackward(*inputV, ih_[0], iw_[0], *outGrad, *outV,
                             fw_, fh_, sh_, sw_, oh_[0], ow_[0],
                             1, 1, ph_, pw_);
}

void MkldnnPoolLayer::submitDnnBwd(const UpdateCallback &callback) {
  (void)callback;

//  exBwd(nullptr);

  if (nullptr == getInputGrad(0)) {
    return;
  }

  real* botdiff = getInputGrad(0)->getData();
  real* topdiff = getOutputGrad()->getData();

//  MatrixPtr ex = Matrix::create(getInputGrad(0)->getHeight(), getInputGrad(0)->getWidth(),false);
//  ex->copyFrom(*getInputGrad(0));
//  real* exdiff = ex->getData();
//  LOG(INFO) << "--------------------ex bot diff: "<< exdiff[10] << "," << exdiff[11];
  std::vector<primitive> pipeline;
  diffTop_->submitCvt(pipeline, topdiff);
  pipeline.push_back(*bwd_);
  diffBot_->submitCvt(pipeline, botdiff);
  stream(stream::kind::eager).submit(pipeline).wait();
//  LOG(INFO) << "--------------------my bot diff: "<< botdiff[10] << "," << botdiff[11];
/*  real sum=0;
  real mx=0;
  size_t cnt = ex->getElementCnt();
  for (size_t i=0; i<cnt; ++i) {
    real tmp = fabs(exdiff[i]-botdiff[i]);
    sum += tmp;
    mx = std::max(tmp, mx);
  }
  LOG(INFO) << "cnt:" << cnt << "max:"<<mx;
  LOG(INFO) <<"--------------absf diff sum:"<<sum/cnt;
*/
}

}  // namespace paddle
