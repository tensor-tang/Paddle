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
  num_concats_ = inputLayers_.size();
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    ic_.push_back(0);
    iw_.push_back(0);
    ih_.push_back(0);
    ow_.push_back(0);
    oh_.push_back(0);
    dataBottoms_.push_back(nullptr);
  }
  return true;
}

void MkldnnConcatLayer::clearDataDiff() {
  reserveOutput(bs_, getSize());
}

void MkldnnConcatLayer::reshape() {
  // reshape input and output size
  CHECK(inputLayers_.size() == size_t(num_concats_));
  int chls = 0;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    int height = inputLayers_[i]->getOutput().getFrameHeight();
    int width = inputLayers_[i]->getOutput().getFrameWidth();
    CHECK_NE(height * width, 0);
    ih_[i] = height;
    iw_[i] = width;
    oh_[i] = ih_[i];
    ow_[i] = iw_[i];
    CHECK(i == 0 || (ih_[i-1] == ih_[i] && iw_[i-1] == iw_[i]));
    ic_[i] = inputLayers_[i]->getSize() / ih_[i] / iw_[i];
    chls += ic_[i];
  }
  oc_ = getSize() / oh_[0] / ow_[0];
  CHECK(oc_ == chls);

  // reset output image size
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);
  printInfo();
}

void MkldnnConcatLayer::resetDnnFwd(PassType passType) {
  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";
  mkldnn::engine eg = CpuEngine::Instance().getEngine();

  std::vector<memory::primitive_desc> botPDs;
  std::vector<std::shared_ptr<memory::desc>> prvs;
  std::vector<primitive::at> srcs;
  memory::format botFmt = memory::format::nchw;
  for (int i = 0; i < num_concats_; ++i) {
    CHECK(bs_ == getInput(i).getBatchSize())
      << "Assert batchsize of input layers are equal";
    CHECK(iw_[i] == ow_[i] && ih_[i] == oh_[i]);
    memory::dims botDims = {bs_, ic_[i], ih_[i], iw_[i]};
    dataBottoms_[i].reset(new MkldnnBuffer());
    real *botData = getInputValue(i)->getData();
    const std::shared_ptr<memory::desc> prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      dataBottoms_[i]->initUser(botData, *prvMD, eg);
      LOG(INFO) << "use prev format: "
        << DNN_FMTS[dataBottoms_[i]->getUserFmt()];
      prvs.push_back(prvMD);
    } else {
      dataBottoms_[i]->initUser(botData, botDims, botFmt, eg);
    }
    botPDs.push_back(dataBottoms_[i]->getUserPD());

    // init bot cvt
    dataBottoms_[i]->initCvt(dataBottoms_[i]->getUserPD(), dnnCvtNoNeed);
    srcs.push_back(*(dataBottoms_[i]->getIntlMem()));
  }

  // inputs format should be all the same
  CHECK(prvs.size() == 0 || prvs.size() == inputLayers_.size())
    << "intl input format size does not match: "
    << prvs.size() << " vs " << inputLayers_.size();

  // top data
  dataTop_.reset(new MkldnnBuffer());
  memory::format topFmt = memory::format::nchw;
  memory::dims topDims = {bs_, oc_, oh_[0], ow_[0]};
  real *topData = getOutputValue()->getData();
  std::shared_ptr<concat::primitive_desc> fwdPD;

  if (setDnnTopDataFmt_) {
    // fwdPD_ should be init with any type before, if in here.
    dataTop_->initUser(topData, topDims,
      memory::format(dataBottoms_[0]->getUserFmt()), eg);
    setTopDataMD(dataTop_->getUserMD());
    LOG(INFO) << "set next format: " << DNN_FMTS[dataTop_->getUserFmt()];
  } else {
    dataTop_->initUser(topData, topDims, topFmt, eg);
  }
  auto concat_dimension = 1;  // TODO(TJ): FIXME: concat dimension
  fwdPD.reset(new concat::primitive_desc(
    dataTop_->getUserMD(), concat_dimension, botPDs));

  // init top cvt
  if (dataTop_->initCvt(
    fwdPD->dst_primitive_desc(), dnnCvtIntl2User)) {
    LOG(INFO) << "need reorder --- top data: "
      << DNN_FMTS[dataTop_->getIntlFmt()]
      << " >>>>> "
      << DNN_FMTS[dataTop_->getUserFmt()];
  }
  fwd_.reset(new concat(*fwdPD, srcs, *(dataTop_->getIntlMem())));
  LOG(INFO) << "data format flow --- "
    << DNN_FMTS[dataBottoms_[0]->getUserFmt()] << " >>> ("
    << DNN_FMTS[dataBottoms_[0]->getIntlFmt()] << " >>> "
    << DNN_FMTS[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FMTS[dataTop_->getUserFmt()];
}

void MkldnnConcatLayer::resetDnnBwd() {

}

void MkldnnConcatLayer::myFwd(PassType passType) {
  real *topdata = getOutputValue()->getData();

  std::vector<primitive> pipeline;
  for (int i = 0; i < num_concats_; ++i) {
    real *botdata = getPrev(i)->getOutputValue()->getData();
    dataBottoms_[i]->submitCvt(pipeline, botdata);
  }

  pipeline.push_back(*fwd_);
  dataTop_->submitCvt(pipeline, topdata);

  // start forward
  REGISTER_TIMER_INFO("mkldnnPoolFwd", getName().c_str());
  stream(stream::kind::eager).submit(pipeline).wait();
//  LOG(INFO) << "------------my" << topdata[0]
//    << "," << topdata[1] << "," << topdata[2];
}

void MkldnnConcatLayer::exFwd(PassType passType) {
  int batchSize = getInput(0).getBatchSize();
  int size = getSize();
  resetOutput(batchSize, size);

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
//  LOG(INFO) << "------------ex" << topdata[0]
//    << "," << topdata[1] << "," << topdata[2];
}

void MkldnnConcatLayer::submitDnnFwd(PassType passType) {
  myFwd(passType);
//  exFwd(passType);

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
