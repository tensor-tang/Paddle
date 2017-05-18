/* Copyright (c) 2017*/

#include "MkldnnReshapeLayer.h"

namespace paddle {

REGISTER_LAYER(mkldnn_reshape, MkldnnReshapeLayer);

bool MkldnnReshapeLayer::initDnn(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  CHECK_EQ(1U, inputLayers_.size());

  return true;
}

void MkldnnReshapeLayer::loadConfig() {
  const ReshapeConfig& conf = config_.inputs(0).reshape_conf();
  reshapeType_ = conf.reshape_type();
  if (!(reshapeType_ == "ToNonSeq"
    || reshapeType_ == "ToMklSeq"
    || reshapeType_ == "ToPaddleSeq")) {
    LOG(FATAL) << "Unknown view type: " << reshapeType_;
  }
  CHECK_EQ(conf.img_dims_size(), 3);
  confChannel_ = conf.img_dims(0);
  confHeight_ = conf.img_dims(1);
  confWidth_ = conf.img_dims(2);  
  confSeqLen_ = conf.has_seq_len() ? conf.seq_len() : -1;

  if (reshapeType_ == "ToNonSeq" && confSeqLen_ > 0) {
    LOG(WARNING) << "conf seqlen is invalid when reshape to nonseq";
  }
}

void MkldnnReshapeLayer::reshapeOutputInfo() {
  int seqLen = getOutputSeqLen();

  reshapeOutMatSize(seqLen);

  reshapeImgSize(outputMatW_);

  resetSeqInfo(seqLen);

  resetOutputSize();
}

void MkldnnReshapeLayer::submitDnnFwd() {
  setSeqInfo();

  shareValue();
}

void MkldnnReshapeLayer::submitDnnBwd(const UpdateCallback& callback) {
  const MatrixPtr& input = getInputGrad(0);
  if (!input) {
    return;
  }
  shareGrad();
}

/// protected methods

int MkldnnReshapeLayer::getOutputSeqLen() {
  if (reshapeType_ == "ToNonSeq") {
    return 1;
  }

  int inputSeqLen = getInputSeqLen();
  if (inputSeqLen > 1) {
    if (confSeqLen_ > 0) {
      CHECK_EQ(inputSeqLen, confSeqLen_) << "can't change seqlen if input is seq";
    }
    return inputSeqLen;
  }

  if (confSeqLen_ > 0) {
    return confSeqLen_;
  }

  CHECK(confChannel_ > 0 && confHeight_ > 0 && confWidth_ > 0)
    << "all of them should be larger than 0, when uncertain seq_len";
  int layerSize = confChannel_ * confHeight_ * confWidth_;
  int seqLen = inputMatW_ / layerSize;
  CHECK_EQ(inputMatW_, seqLen * layerSize) << "not divisible";
  return seqLen;
}

// set the seqinfo
void MkldnnReshapeLayer::resetSeqInfo(const int seqLen) {
  setNeedSequenceInfo(false);
  setNeedMklSeqInfo(false);
  if (reshapeType_ == "ToNonSeq") {
    return;
  }

  if (reshapeType_ == "ToPaddleSeq") {
    resetPaddleSeqInfo(seqLen);
    return;
  }

  resetMklSeqInfo(seqLen);
}

// reshape the output matrix height and width
void MkldnnReshapeLayer::reshapeOutMatSize(const int seqLen) {
  if (reshapeType_ == "ToNonSeq") {
    int inputSeqLen = getInputSeqLen();
    // do not need to separate from seq or not
    outputMatH_ = inputMatH_ / inputSeqLen;
    CHECK_EQ(outputMatH_ * inputSeqLen, inputMatH_) << "not divisible";
    outputMatW_ = inputSeqLen * inputMatW_;
    return;
  }

  // to seq: mklseq or paddle seq
  if (inputIsSequential()) {
    outputMatH_ = inputMatH_;
    outputMatW_ = inputMatW_;
    return;
  }

  outputMatW_ = inputMatW_ / seqLen;
  CHECK_EQ(outputMatW_ * seqLen, inputMatW_) << "not divisible";
  outputMatH_ = inputMatH_ * seqLen;
}

void MkldnnReshapeLayer::reshapeImgSize(const size_t layerSize) {
  if (confChannel_ <= 0 || confWidth_ <= 0 || confHeight_ <= 0) {
    if (confWidth_ <= 0) {
      CHECK(confChannel_ > 0 && confHeight_ > 0) << "allow only one uncertain";
      oc_ = confChannel_;
      oh_ = confHeight_;
      ow_ = layerSize / (oc_ * oh_);
    } else if (confHeight_ <= 0) {
      CHECK(confChannel_ > 0 && confWidth_ > 0) << "allow only one uncertain";
      oc_ = confChannel_;
      ow_ = confWidth_;
      oh_ = layerSize / (oc_ * ow_);
    } else {
      CHECK(confHeight_ > 0 && confWidth_ > 0) << "allow only one uncertain";
      oh_ = confHeight_;
      ow_ = confWidth_;
      oc_ = layerSize / (oh_ * ow_);
    }
    CHECK_EQ(oc_ * oh_ * ow_, layerSize) << getName() << "not divisible";
  } else {
    oc_ = confChannel_;
    oh_ = confHeight_;
    ow_ = confWidth_;
    CHECK_EQ(oc_ * oh_ * ow_, layerSize)
      << getName() << " layerSize does not match";
  }
}

void MkldnnReshapeLayer::resetOutputSize() {
// update new layer size, in case it would be used in other layers
  config_.set_size(outputMatW_);
  getOutput().setFrameHeight(oh_);
  getOutput().setFrameWidth(ow_);
}

int MkldnnReshapeLayer::getInputSeqLen() {
  const Argument& input = getInput(0);
  if (input.hasMklSeq()) {
    CHECK(nullptr == input.sequenceStartPositions)
      << "should only have one: mklseq or paddleseq";
    return input.getMklSeqLen();
  } else if (input.sequenceStartPositions) {
    // if has seq, get aligned seq length
    CHECK(!input.hasSubseq()) << "Do not support sub seqence yet";
    return getPaddleAlignedSeqLen(input);
  }
  return -1;
}

bool MkldnnReshapeLayer::inputIsSequential() {
  const Argument& input = getInput(0);
  bool hasMklSeq = input.hasMklSeq();
  bool hasPaddleSeq = input.sequenceStartPositions != nullptr;
  return hasMklSeq || hasPaddleSeq ? true : false;
}

void MkldnnReshapeLayer::resetMklSeqInfo(int seqLen) {
  mklSeqLen_ = seqLen;
}

void MkldnnReshapeLayer::resetPaddleSeqInfo(int seqLen) {
  size_t bs = outputMatH_ / seqLen;
  CHECK_EQ(bs * seqLen, outputMatH_) << "not divisible";

  ICpuGpuVector::resizeOrCreate(seqIdx_, bs + 1, /* useGpu= */ false);
  int* buf = seqIdx_->getMutableData(false);
  buf[0] = 0;
  for (size_t i = 1; i <= bs; ++i) {
    buf[i] = buf[i - 1] + seqLen;
  }
  CHECK_EQ(buf[bs], outputMatH_);

  subSeqIdx_ = nullptr;  // not use sub yet
  seqDims_ = nullptr;  // not figure out usage yet
}

inline void MkldnnReshapeLayer::setSeqInfo() {
  if (reshapeType_ == "ToNonSeq") {
    return;
  }

  Argument& output = getOutput();
  if (reshapeType_ == "ToMklSeq") {
    setMklSeqInfo(output);
  } else if (reshapeType_ == "ToPaddleSeq") {
    setPaddleSeqInfo(output);
  }
}

inline void MkldnnReshapeLayer::shareValue() {
  const MatrixPtr& input = getInputValue(0);
  real* value = input->getData();
  output_.value->setData(value);
}

inline void MkldnnReshapeLayer::shareGrad() {
  const MatrixPtr& output = getOutputGrad();
  real* grad = output->getData();
  inputLayers_[0]->getOutput().grad->setData(grad);
}

void MkldnnReshapeLayer::setPaddleSeqInfo(Argument& arg) {
  arg.sequenceStartPositions = seqIdx_;
  arg.subSequenceStartPositions = subSeqIdx_;
  arg.cpuSequenceDims = seqDims_;
}

void MkldnnReshapeLayer::setMklSeqInfo(Argument& arg) {
  arg.setMklSeqLen(mklSeqLen_);
}


}  // namespace paddle

