/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "Layer.h"
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"

//#include "ViewLayer.h"

namespace paddle {

/**
 * @brief A layer for reshaping
 * @note
 * origin matrix 
 * view matrix: 
 */
class ViewLayer : public Layer {
public:
  explicit ViewLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;

  void backward(const UpdateCallback& callback) override;

protected:
  std::string viewType_;
  size_t elementCnt_;
  size_t layerSizeIn_, layerSizeOut_;
  int seqLenIn_, seqLenOut_;  // sequence length for input and output 

  // used for none to seq
  ICpuGpuVectorPtr seqIdx_;
  // for further use
  ICpuGpuVectorPtr subSeqIdx_;
  IVectorPtr seqDims_;

  int bsIn_, bsOut_;
  int ic_, ih_, iw_;
  int oc_, oh_, ow_;
  
  // get the sequence length, and the length among batchsize should be the same
  int getSeqLength(const Argument& arg);

  // generate sequence index, used in NoneToSeq
  void generateSeqInfo();

  // set sequence info to arg
  void setSeqInfo(Argument& arg);

  void configSeqToNone(const Argument& arg);

  void configNoneToSeq(const Argument& arg);

  void configKeepType(const Argument& arg);

  // check size of channel, height and width
  // should have only one uncertain size at most.
  // sz == channel * height * width
  void checkOutputSize(size_t sz);

  // reset configure
  // reload the settings from proto
  void reloadConfig();
};


REGISTER_LAYER(view, ViewLayer);

bool ViewLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) return false;
  CHECK_EQ(1U, inputLayers_.size());

  reloadConfig();

  if (!(viewType_ == "NoChange"
    || viewType_ == "SeqToNone"
    || viewType_ == "NoneToSeq")) {
    LOG(FATAL) << "Unknown view type: " << viewType_;
    return false;
  }

  elementCnt_ = 0;
  layerSizeIn_ = 0;
  layerSizeOut_ = 0;
  bsOut_ = -1;
  bsIn_ = -1;

  return true;
}

void ViewLayer::reloadConfig() {
  const ViewConfig& conf = config_.inputs(0).view_conf();
  viewType_ = conf.view_type();
  oh_ = conf.height();
  ow_ = conf.width();
  oc_ = conf.has_channel() ? conf.channel() : 1;
  seqLenOut_ = conf.has_seq_len() ? conf.seq_len() : 1;
}

int ViewLayer::getSeqLength(const Argument& arg) {
  CHECK(arg.sequenceStartPositions);
  int sampleSize = arg.getBatchSize();  // bs*seqlen
  size_t numSequences = arg.getNumSequences();
  const int* starts = arg.sequenceStartPositions->getData(false);
  CHECK_EQ(starts[numSequences], sampleSize);
  int len = 0;
  for (size_t i = 0; i < numSequences; ++i) {
    int tmp = starts[i + 1] - starts[i];
    CHECK(len == 0 || len == tmp)
      << "all seq length should be equal," << len << " vs " << tmp;
    len = tmp;
  }
  return len;
}

void ViewLayer::generateSeqInfo() {
  size_t numSequences = bsOut_ / seqLenOut_;
  CHECK_EQ(numSequences * seqLenOut_, bsOut_) << "maybe caused by un-divisible";
  ICpuGpuVector::resizeOrCreate(seqIdx_, numSequences + 1, /* useGpu= */ false);
  int* buf = seqIdx_->getMutableData(false);
  buf[0] = 0;
  for (size_t i = 1; i <= numSequences; ++i) {
    buf[i] = buf[i - 1] + seqLenOut_;
  }
  CHECK_EQ(buf[numSequences], bsOut_);
  subSeqIdx_ = nullptr;  // not use sub yet
  seqDims_ = nullptr;  // not figure out usage yet
}

void ViewLayer::setSeqInfo(Argument& arg) {
  arg.sequenceStartPositions = seqIdx_;
  arg.subSequenceStartPositions = subSeqIdx_;
  arg.cpuSequenceDims = seqDims_;
}

void ViewLayer::checkOutputSize(size_t size) {
  if (oc_ <= 0 || oh_ <= 0 || ow_ <= 0) {
    if (ow_ <= 0) {
      CHECK(oc_ > 0 && oh_ > 0) << "only should have one uncertain";
      ow_ = size / (oc_ * oh_);
    } else if (oh_ <= 0) {
      CHECK(oc_ > 0 && ow_ > 0) << "only should have one uncertain";
      oh_ = size / (oc_ * ow_);
    } else {
      CHECK(oh_ > 0 && ow_ > 0) << "only should have one uncertain";
      oc_ = size / (oh_ * ow_);
    }
  }
  CHECK_EQ(oc_ * oh_ * ow_, size) << "maybe caused by un-divisible";
}

void ViewLayer::configSeqToNone(const Argument& arg) {
  CHECK(!arg.hasSubseq()) << "Do not support sub seqence yet";
  seqLenIn_ = getSeqLength(arg);
  bsOut_ = bsIn_ / seqLenIn_;
  CHECK_EQ(bsOut_ * seqLenIn_, bsIn_) << "maybe caused by un-divisible";
  layerSizeOut_ = seqLenIn_ * layerSizeIn_;
  setNeedSequenceInfo(false);
}

void ViewLayer::configNoneToSeq(const Argument& arg) {
  CHECK(!arg.sequenceStartPositions) << "already sequence, maybe change type";
  if (seqLenOut_ > 0) {
    layerSizeOut_ = layerSizeIn_ / seqLenOut_;
  } else {
    CHECK(oc_ > 0 && oh_ > 0 && ow_ > 0)
      << "all of them should be larger than 0, when uncertain seq_len";
    layerSizeOut_ = oc_ * oh_ * ow_;
    seqLenOut_ = layerSizeIn_ / layerSizeOut_;
  }
  CHECK_EQ(seqLenOut_ * layerSizeOut_, layerSizeIn_)
    << "maybe caused by un-divisible";;
  bsOut_ = bsIn_ * seqLenOut_;
  setNeedSequenceInfo(true);
  generateSeqInfo();
}

void ViewLayer::configKeepType(const Argument& arg) {
  if (arg.sequenceStartPositions) {  // if have sequence
    LOG(WARNING) << "only change image size, will not change seqence";
  }
  CHECK(!arg.hasSubseq()) << "Do not support sub seqence yet";
  bsOut_ = bsIn_;
  layerSizeOut_ = layerSizeIn_;
}

void ViewLayer::forward(PassType passType) {
  const Argument& input = getInput(0);
  CHECK(getInputValue(0)) << "Should have one input value and only one!";
  if (getInputValue(0)->getElementCnt() != elementCnt_) {
    // if input cnt (input batchsize * layer_size) change, reset configure
    reloadConfig();
    elementCnt_ = getInputValue(0)->getElementCnt();
//    LOG(INFO) << "change input cnt: " << elementCnt_ << ", view type: " << viewType_;
    bsIn_ = input.getBatchSize();
    CHECK_EQ(bsIn_, input.value->getHeight());
    layerSizeIn_ = input.value->getWidth();
    if (viewType_ == "NoChange") {
      configKeepType(input);
    } else if (viewType_ == "SeqToNone") {
      configSeqToNone(input);
    } else if (viewType_ == "NoneToSeq") {
      configNoneToSeq(input);
    } else {
      LOG(FATAL) << "Unknown view type: " << viewType_;
      return;
    }
    checkOutputSize(layerSizeOut_);
    CHECK_EQ(bsOut_ * layerSizeOut_, elementCnt_);
    // update new layer size, in case it would be used in other layers
    config_.set_size(layerSizeOut_);
    getOutput().setFrameHeight(oh_);
    getOutput().setFrameWidth(ow_);
//    LOG(INFO) << "outsize:"<<getSize();
//    LOG(INFO) << "oh:"<<oh_<<",ow:"<<ow_;
  }
  Layer::forward(passType);
  if (viewType_ == "NoneToSeq") {
    setSeqInfo(getOutput());
  }

  reserveOutput(bsOut_, getSize());  // set the output buffer and output size
  MatrixPtr tmp = Matrix::create(output_.value->getData(),
    bsIn_, layerSizeIn_, // should use input size when use assgin function
    false, useGpu_);
  tmp->assign(*input.value);
  //output_.value->assign(*input.value);
//  LOG(INFO) << "outheight outbs:" << output_.value->getHeight() <<",width:"<<output_.value->getWidth();
}

void ViewLayer::backward(const UpdateCallback& callback) {
  const Argument& input = getInput(0);

  if (!input.grad) {
    return;
  }
  MatrixPtr tmp = Matrix::create(input.grad->getData(),
                                 bsOut_,
                                 getSize(),
                                 false,
                                 useGpu_);
  tmp->add(*output_.grad);
}

}  // namespace paddle

