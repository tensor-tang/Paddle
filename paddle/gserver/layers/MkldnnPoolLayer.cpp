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
  // create dim structure that describes user data.
  memory::dims botDims = {bs_, ic_[0], ih_[0], iw_[0]};
  memory::dims kernel = {fh_, fw_};
  memory::dims strides = {sh_, sw_};
  memory::dims padding = {ph_, pw_};
  memory::dims topDims = {bs_, oc_, oh_[0], ow_[0]};

  dataBot_.reset(new MkldnnBuffer());
  dataTop_.reset(new MkldnnBuffer());

  // init user memory of bottom, weights and bias
  real *botData = getPrev(0)->getOutputValue()->getData();
  real *topData = getOutputValue()->getData();
  const std::shared_ptr<memory::desc> prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    dataBot_->initUser(botData, *prvMD, eg);
    LOG(INFO) << "use prev format: " << DNN_FORMAT[dataBot_->getUserFmt()];
  } else {
    dataBot_->initUser(botData, botDims, memory::format::nchw, eg);
  }

  // create pool desc from internal desc
  std::shared_ptr<pooling_forward::desc> fwdDesc;
  prop_kind pk = (passType == PASS_TEST) ? prop_kind::forward_scoring :
    prop_kind::forward_training;

  fwdDesc.reset(new pooling_forward::desc(pk, poolAlgo_,
                    prvMD ? dataBot_->getUserMD() : getAnyMD(botDims),
                    getAnyMD(topDims),
                    strides, kernel, padding, padding,
                    padding_kind::zero));
  // init cvt
  dataBot_->initIntlCvt(dataBot_->getUserPD(), dnnCvtNoNeed);
  
  fwdPD_.reset(new pooling_forward::primitive_desc(*fwdDesc, eg));

  // init top user memory and cvt
  if (setDnnTopDataFmt_) {
    dataTop_->initUser(topData, fwdPD_->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
    LOG(INFO) << "set next format: " << DNN_FORMAT[dataTop_->getUserFmt()];
  } else {
    dataTop_->initUser(topData, topDims, memory::format::nchw, eg);
  }
  if (dataTop_->initIntlCvt(fwdPD_->dst_primitive_desc(), dnnCvtIntl2User)) {
    LOG(INFO) << "need reorder --- top data: "
      << DNN_FORMAT[dataTop_->getIntlFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataTop_->getUserFmt()];
  }

  withWorkspace_ = passType != PASS_TEST && poolAlgo_ != algorithm::pooling_avg;
  if (withWorkspace_) {
    workspace_.reset(new memory(fwdPD_->workspace_primitive_desc()));
  } else {
    auto p_workspace_desc = memory::primitive_desc(
      {{}, memory::data_type::f32, memory::format(dataTop_->getIntlFmt())},
      eg);
    workspace_.reset(new memory(p_workspace_desc));
  }
  if (withWorkspace_) {
    fwd_.reset(new pooling_forward(*fwdPD_,
      *(dataBot_->getIntlMem()), *(dataTop_->getIntlMem()),
      *workspace_));
  } else {
    fwd_.reset(new pooling_forward(*fwdPD_,
      *(dataBot_->getIntlMem()), *(dataTop_->getIntlMem())));
  }
  LOG(INFO) << "data format flow --- "
    << DNN_FORMAT[dataBot_->getUserFmt()] << " >>> ("
    << DNN_FORMAT[dataBot_->getIntlFmt()] << " >>> "
    << DNN_FORMAT[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FORMAT[dataTop_->getUserFmt()];
}

void MkldnnPoolLayer::resetDnnBwd() {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  memory::dims kernel = {fh_, fw_};
  memory::dims strides = {sh_, sw_};
  memory::dims padding = {ph_, pw_};
  diffBot_.reset(new MkldnnBuffer());
  diffTop_.reset(new MkldnnBuffer());

  // init top diff user
  real *topDiff = getOutputGrad()->getData();
  real* botDiff = getPrev(0)->getOutputGrad()->getData();
  const std::shared_ptr<mkldnn::memory::desc> inputDiffMD = getTopDiffMD();
  if (inputDiffMD) {
    diffTop_->initUser(topDiff, *inputDiffMD, eg);
  } else {
    memory::dims topDims = {bs_, oc_, oh_[0], ow_[0]};
    diffTop_->initUser(topDiff, topDims, memory::format::nchw, eg);
  }
  if (setDnnBotDiffFmt_[0]) {
    diffBot_->initUser(botDiff, dataBot_->getIntlPD());
    getPrev(0)->setTopDiffMD(diffBot_->getUserMD());
  } else {
    memory::dims botDims = {bs_, ic_[0], ih_[0], iw_[0]};
    diffBot_->initUser(botDiff, botDims, memory::format::nchw, eg);
  }
  diffTop_->initIntlCvt(dataTop_->getIntlPD(), dnnCvtUser2Intl);
  diffBot_->initIntlCvt(dataBot_->getIntlPD(), dnnCvtIntl2User);

  std::shared_ptr<pooling_backward::desc> bwdDesc;
  std::shared_ptr<pooling_backward::primitive_desc> bwdPD;
  bwdDesc.reset(new pooling_backward::desc(
    poolAlgo_, dataBot_->getIntlMD(), dataTop_->getIntlMD(),
    strides, kernel, padding, padding, padding_kind::zero));
  bwdPD.reset(new pooling_backward::primitive_desc(
    *bwdDesc, eg, *fwdPD_));
  if (withWorkspace_) {
    bwd_.reset(new pooling_backward(*bwdPD,
      *(diffTop_->getIntlMem()), *workspace_, *(diffBot_->getIntlMem())));
  } else {
    bwd_.reset(new pooling_backward(*bwdPD,
      *(diffTop_->getIntlMem()), *(diffBot_->getIntlMem())));
  }
  LOG(INFO) << "diff format flow --- "
    << DNN_FORMAT[diffBot_->getUserFmt()] << " <<< ("
    << DNN_FORMAT[diffBot_->getIntlFmt()] << " <<< "
    << DNN_FORMAT[diffTop_->getIntlFmt()] << ") <<< "
    << DNN_FORMAT[diffTop_->getUserFmt()];
}

void MkldnnPoolLayer::myFwd(PassType passType) {
  real *botdata = getPrev(0)->getOutputValue()->getData();
  real *topdata = getOutputValue()->getData();

  std::vector<primitive> pipeline;
  dataBot_->submitCvt(pipeline, botdata);

  pipeline.push_back(*fwd_);
  dataTop_->submitCvt(pipeline, topdata);

  // start forward
  REGISTER_TIMER_INFO("mkldnnPoolFwd", getName().c_str());
  stream(stream::kind::eager).submit(pipeline).wait();
//  LOG(INFO) << "------------" << topdata[0];
// << "," << topdata[1] << "," << topdata[2];

// TODO(TJ): no activation???
//  forwardActivation();
}

void MkldnnPoolLayer::exFwd(PassType passType) {
  const Argument& in = getInput(0);
  const Argument& out = output_;
  CHECK_EQ(getSize(), out.value->getWidth());
  MatrixPtr inputV = in.value;
  MatrixPtr outV = out.value;
  outV->maxPoolForward(*inputV, ih_[0], iw_[0], ic_[0], fw_, fh_,
                       sh_, sw_, oh_[0], ow_[0], ph_, pw_);

//  real *topdata = getOutputValue()->getData();
//  LOG(INFO) << "------------" << topdata[0];
// << "," << topdata[1] << "," << topdata[2];
}

void MkldnnPoolLayer::submitDnnFwd(PassType passType) {
//  exFwd(passType);
  myFwd(passType);
}

void MkldnnPoolLayer::exBwd(const UpdateCallback &callback) {
  
  const Argument& in = getInput(0);
  MatrixPtr outGrad = getOutputGrad();
  MatrixPtr inputV = in.value;
  MatrixPtr outV = getOutputValue();
  MatrixPtr inputGrad = in.grad;

  if (NULL == getInputGrad(0)) {
    return;
  }

  inputGrad->maxPoolBackward(*inputV, ih_[0], iw_[0], *outGrad, *outV,
                             fw_, fh_, sh_, sw_, oh_[0], ow_[0],
                             1, 1, ph_, pw_);
}

void MkldnnPoolLayer::submitDnnBwd(const UpdateCallback &callback) {
  (void)callback;

//  exBwd(nullptr);

  if (NULL == getInputGrad(0)) {
    return;
  }

  real* botdiff = getInputGrad(0)->getData();
  real* topdiff = getOutputGrad()->getData();
  
//  LOG(INFO) << "--------------------ex data diff: "<< botdiff[0] << "," << botdiff[10];
  std::vector<primitive> pipeline;
  diffTop_->submitCvt(pipeline, topdiff);
  pipeline.push_back(*bwd_);
  diffBot_->submitCvt(pipeline, botdiff);
  stream(stream::kind::eager).submit(pipeline).wait();
//  LOG(INFO) << "--------------------my data diff: "<< botdiff[0] << "," << botdiff[10];
  
}

}  // namespace paddle
