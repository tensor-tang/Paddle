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
#include "MkldnnConcatLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_concat, MkldnnConcatLayer);

bool MkldnnConcatLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  CHECK(!biasParameter_);
  
  bs_ = 0;
  oc_ = getSize();
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    ic_.push_back(0);
    iw_.push_back(0);
    ih_.push_back(0);
    ow_.push_back(0);
    oh_.push_back(0);
  }

  return true;
}

size_t MkldnnConcatLayer::getOneBatchSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  int height = inputLayers_[0]->getOutput().getFrameHeight();
  int width = inputLayers_[0]->getOutput().getFrameWidth();
  if (height != 0) ih_[0] = height;
  if (width != 0) iw_[0] = width;
  oh_[0] = outputSize(ih_[0], fh_, ph_, sh_, false);
  ow_[0] = outputSize(iw_[0], fw_, pw_, sw_, false);



  for (size_t i = 0; i < inputLayers_.size(); i++) {
    ic_.push_back(0);
    iw_.push_back(0);
    ih_.push_back(0);
    ow_.push_back(0);
    oh_.push_back(0);
  }
  return oh_[0] * ow_[0] * oc_;
}

// whether reset batchsize and image size of input and output
bool MkldnnConcatLayer::reshapeOutput() {
  return false;
/*
  if (bs_ == getInput(0).getBatchSize()) {
    // TODO(TJ): can remove
    // when confirm how multi inputs work and whether to clear diff
    reserveOutput(bs_, getOneBatchSize());
    return false;
  }

  // reset image size
  size_t layersize = getOneBatchSize();
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);

  // reset data
  bs_ = getInput(0).getBatchSize();
  LOG(INFO) << "layer name: " << getName();
  LOG(INFO) << "reshape batch size: " << bs_;
  resetOutput(bs_, layersize);
  printInfo();
  return true;*/
}

void MkldnnConcatLayer::resetDnnFwd(PassType passType) {
  /*

  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  // create dim structure that describes user data.
  memory::dims botDims = {bs_, ic_[0], ih_[0], iw_[0]};
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
  std::shared_ptr<pooling_forward::primitive_desc> fwdPD;
  fwdPD.reset(new pooling_forward::primitive_desc(*fwdDesc, eg));

  // init top user memory and cvt
  if (setDnnTopDataFmt_) {
    dataTop_->initUser(topData, fwdPD->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
    LOG(INFO) << "set next format: " << DNN_FORMAT[dataTop_->getUserFmt()];
  } else {
    dataTop_->initUser(topData, topDims, memory::format::nchw, eg);
  }
  if (dataTop_->initIntlCvt(fwdPD->dst_primitive_desc(), dnnCvtInternal2User)) {
    LOG(INFO) << "need reorder --- top data: "
      << DNN_FORMAT[dataTop_->getIntlFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataTop_->getUserFmt()];
  }

  withWorkspace_ = passType != PASS_TEST && poolAlgo_ != algorithm::pooling_avg;
  if (withWorkspace_) {
    workspace_.reset(new memory(fwdPD->workspace_primitive_desc()));
  } else {
    auto p_workspace_desc = memory::primitive_desc(
      {{}, memory::data_type::f32, memory::format(dataTop_->getIntlFmt())},
      eg);
    workspace_.reset(new memory(p_workspace_desc));
  }
  if (withWorkspace_) {
    fwd_.reset(new pooling_forward(*fwdPD,
      *(dataBot_->getIntlMem()), *(dataTop_->getIntlMem()),
      *workspace_));
  } else {
    fwd_.reset(new pooling_forward(*fwdPD,
      *(dataBot_->getIntlMem()), *(dataTop_->getIntlMem())));
  }
  LOG(INFO) << "data format flow --- "
    << DNN_FORMAT[dataBot_->getUserFmt()] << " >>> ("
    << DNN_FORMAT[dataBot_->getIntlFmt()] << " >>> "
    << DNN_FORMAT[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FORMAT[dataTop_->getUserFmt()];
  */
}

void MkldnnConcatLayer::resetDnnBwd() {

}

void MkldnnConcatLayer::myFwd(PassType passType) {
  /// all sumbit cvt should be clear
  clearAllCvtFlags();

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
}

void MkldnnConcatLayer::exFwd(PassType passType) {

  int batchSize = getInput(0).getBatchSize();
  int size = getSize();
  reserveOutput(batchSize, size);

  const MatrixPtr& out = getOutputValue();
  int offset = 0;

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    const MatrixPtr& in = getInputValue(i);
    size_t inSize = in->getWidth();
    out->assignAtOffset(*in, offset);
    offset += inSize;
  }
  CHECK_EQ(size, offset);


//  real *topdata = getOutputValue()->getData();
//  LOG(INFO) << "------------" << topdata[0];
// << "," << topdata[1] << "," << topdata[2];
}

void MkldnnConcatLayer::submitDnnFwd(PassType passType) {
  exFwd(passType);
//  myFwd(passType);


  REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
  forwardActivation();

}

void MkldnnConcatLayer::exBwd(const UpdateCallback &callback) {
    (void)callback;

  /* Do activation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  const MatrixPtr& out = getOutputGrad();
  int offset = 0;

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    const MatrixPtr& in = getInputGrad(i);
    size_t inSize = getInputValue(i)->getWidth();
    if (in) {
      in->addAtOffset(*out, offset);
    }
    offset += inSize;
  }

}

void MkldnnConcatLayer::submitDnnBwd(const UpdateCallback &callback) {
  exBwd(callback);

  // dnn backward
  /*
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    // backward weights before data, since may have not botdiff in some layer
    if (weights_[i]->getWGrad()) {
      submitBwdWgts(i, getPrev(i)->getOutputValue(), getOutputGrad());
      // Increasing the number of gradient 
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
    submitBwdData(i, getOutputGrad(), getPrev(i)->getOutputGrad());
  }
  if (biases_ && biases_->getWGrad()) {
    // Increasing the number of gradient 
    biases_->getParameterPtr()->incUpdate(callback);
  }
  */
}

}  // namespace paddle
