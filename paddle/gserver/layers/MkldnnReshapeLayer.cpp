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

#include "MkldnnLayer.h"

namespace paddle {

/**
 * @brief A layer for view
 * @note
 */
class MkldnnReshapeLayer : public MkldnnLayer {
protected:
  std::string viewType_;


  // sequence length for input and output, set 1 if do not have seq
  int seqLenIn_;

  // used for none to seq
  ICpuGpuVectorPtr seqIdx_;
  // for further use
  ICpuGpuVectorPtr subSeqIdx_;
  IVectorPtr seqDims_;

public:
  explicit MkldnnReshapeLayer(const LayerConfig& config)
    : MkldnnLayer(config) {}

  virtual bool initDnn(const LayerMap& layerMap,
                           const ParameterMap& parameterMap);

  // reload the settings from proto
  virtual void loadConfig();

  // reshape 
  // output matrix height and width
  // and the output buffer
  virtual void reshapeOutput();

  /** 
   * each dnn layer should have function
   * to init or reset forward mkldnn
   */
  virtual void resetDnnFwd() {};

  /** 
   * each dnn layer should have function
   * to init or reset backward mkldnn
   */
  virtual void resetDnnBwd() {};

  virtual void submitDnnFwd();
  virtual void submitDnnBwd(const UpdateCallback& callback);

protected:

  // get the sequence length from paddle seq info
  // and the length among batchsize should be the same
  int getAlignedSeqLength(const Argument& arg);

  // check size of channel, height and width
  // should have only one uncertain size at most.
  // sz == channel * height * width
  void checkOutputImgSize(size_t sz);


  // generate sequence index, used in NoneToSeq
  void generatePaddleSeqInfo();

  // set sequence info to arg
  void setPaddleSeqInfo(Argument& arg);

  // prepare output matrix height, width, and seqlen
  void prepareSeqInfo();
  

  void configToNonSeq();
  
  void configToMklSeq();

  void configToSeq();


};


REGISTER_LAYER(mkldnn_reshape, MkldnnReshapeLayer);

bool MkldnnReshapeLayer::initDnn(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  CHECK_EQ(1U, inputLayers_.size());

  return true;
}

void MkldnnReshapeLayer::loadConfig() {
  const ViewConfig& conf = config_.inputs(0).view_conf();
  viewType_ = conf.view_type();
  if (!(viewType_ == "ToNonSeq"
    || viewType_ == "ToMklSeq"
    || viewType_ == "ToSeq")) {
    LOG(FATAL) << "Unknown view type: " << viewType_;
  }
  oh_ = conf.height();
  ow_ = conf.width();
  oc_ = conf.has_channel() ? conf.channel() : -1;
  seqLen_ = conf.has_seq_len() ? conf.seq_len() : -1;
}

int MkldnnReshapeLayer::getAlignedSeqLength(const Argument& arg) {
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

void MkldnnReshapeLayer::configToNonSeq() {
  outputMatH_ = inputMatH_ / seqLenIn_;
  //bs_ = outputMatH_;
  CHECK_EQ(outputMatH_ * seqLenIn_, inputMatH_) << "maybe caused by un-divisible";
  outputMatW_ = seqLenIn_ * inputMatW_;
  setNeedSequenceInfo(false);
  setNeedMklSeqInfo(false);
}

// confirm seqlen, output matrix height and width
void MkldnnReshapeLayer::prepareSeqInfo() {  
  if (seqLen_ > 0) {
    // if have set from proto
    outputMatW_ = inputMatW_ / seqLen_;
  } else {
    CHECK(oc_ > 0 && oh_ > 0 && ow_ > 0)
      << "all of them should be larger than 0, when uncertain seq_len";
    outputMatW_ = oc_ * oh_ * ow_;
    seqLen_ = inputMatW_ / outputMatW_;
  }
  CHECK_EQ(seqLen_ * outputMatW_, inputMatW_)
    << "maybe caused by un-divisible";
  outputMatH_ = inputMatH_ * seqLen_;
}

void MkldnnReshapeLayer::configToMklSeq() {
  CHECK_EQ(seqLenIn_, 1) << "only support reshape from nonseq yet";
  prepareSeqInfo();
  setNeedMklSeqInfo(true);
}

void MkldnnReshapeLayer::generatePaddleSeqInfo() {
  size_t bs = outputMatH_ / seqLen_;
  CHECK_EQ(bs * seqLen_, outputMatH_) << "maybe caused by un-divisible";
  ICpuGpuVector::resizeOrCreate(seqIdx_, bs + 1, /* useGpu= */ false);
  int* buf = seqIdx_->getMutableData(false);
  buf[0] = 0;
  for (size_t i = 1; i <= bs; ++i) {
    buf[i] = buf[i - 1] + seqLen_;
  }
  CHECK_EQ(buf[bs], outputMatH_);
  subSeqIdx_ = nullptr;  // not use sub yet
  seqDims_ = nullptr;  // not figure out usage yet
}

void MkldnnReshapeLayer::configToSeq() {
  CHECK_EQ(seqLenIn_, 1) << "only support reshape from nonseq yet";
  prepareSeqInfo();
  setNeedSequenceInfo(true);
  generatePaddleSeqInfo();
}

void MkldnnReshapeLayer::checkOutputImgSize(size_t size) {
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
    CHECK_EQ(oc_ * oh_ * ow_, size) << "maybe caused by un-divisible";
  } else {
    CHECK_EQ(oc_ * oh_ * ow_, size) << "size does not match";
  }
}

void MkldnnReshapeLayer::reshapeOutput() {
  // get input seqlen
  const Argument& input = getInput(0);
  if (input.hasMklSeq()) {
    CHECK(nullptr == input.sequenceStartPositions)
      << "should only have one: mklseq or paddleseq";
    seqLenIn_ = input.getMklSeqLen();
  } else if (input.sequenceStartPositions) {
    // if has seq, get aligned seq length
    CHECK(!input.hasSubseq()) << "Do not support sub seqence yet";
    seqLenIn_ = getAlignedSeqLength(input);
  } else {
    seqLenIn_ = 1;
  }
 
  // set output matrix height and width
  if (viewType_ == "ToNonSeq") {
    configToNonSeq();
  } else if (viewType_ == "ToMklSeq") {
    configToMklSeq();
  } else if (viewType_ == "ToSeq") {
    configToSeq();
  } else {
    LOG(FATAL) << "Unknown view type: " << viewType_;
    return;
  }

  checkOutputImgSize(outputMatW_);
  // update new layer size, in case it would be used in other layers
  config_.set_size(outputMatW_);
  
  resetOutput(outputMatH_, outputMatW_);
  getOutput().setFrameHeight(oh_);
  getOutput().setFrameWidth(ow_);
}

void MkldnnReshapeLayer::setPaddleSeqInfo(Argument& arg) {
  arg.sequenceStartPositions = seqIdx_;
  arg.subSequenceStartPositions = subSeqIdx_;
  arg.cpuSequenceDims = seqDims_;
}

void MkldnnReshapeLayer::submitDnnFwd() {
  const Argument& input = getInput(0);
  if (viewType_ == "ToSeq") {
    setPaddleSeqInfo(getOutput());
  }
  if (viewType_ == "ToMklSeq") {
    getOutput().setMklSeqLen(seqLen_);
  }
  output_.value = Matrix::create(input.value->getData(),
    outputMatH_, getSize(), false, false);
}

void MkldnnReshapeLayer::submitDnnBwd(const UpdateCallback& callback) {
  const Argument& input = getInput(0);
  if (!input.grad) {
    return;
  }
  inputLayers_[0]->getOutput().grad = Matrix::create(output_.grad->getData(),
    inputMatH_, inputMatW_, false, false);
}

}  // namespace paddle

