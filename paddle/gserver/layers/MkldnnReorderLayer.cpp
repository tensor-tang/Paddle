/* Copyright (c) 2017*/

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"


using namespace mkldnn;  // NOLINT

namespace paddle {

/**
 * @brief A layer for reorder
 * API: mkldnn_reorder
 * @note
 */
class MkldnnReorderLayer : public MkldnnLayer {
protected:
  std::shared_ptr<memory> botVal_, botGrd_;
  std::shared_ptr<memory> topVal_, topGrd_;
  std::shared_ptr<primitive> fwd_, bwd_;
  int bsIdx_;
  std::vector<int> fromDims_;


public:
  explicit MkldnnReorderLayer(const LayerConfig& config)
    : MkldnnLayer(config) {}

  virtual bool initDnn(const LayerMap& layerMap,
    const ParameterMap& parameterMap) {return true;}

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
  virtual void resetDnnFwd();

  /** 
   * each dnn layer should have function
   * to init or reset backward mkldnn
   */
  virtual void resetDnnBwd();

  virtual void submitDnnFwd();
  virtual void submitDnnBwd(const UpdateCallback& callback);

protected:
  // confirm the bot dims
  // return the total size
  size_t confirmBotDims(memory::dims& bot);

  // confirm the top dims from bot dims
  void confirmTopDims(const memory::dims& bot, memory::dims& top);

};


REGISTER_LAYER(mkldnn_reorder, MkldnnReorderLayer);

void MkldnnReorderLayer::loadConfig() {
  CHECK_EQ(1U, inputLayers_.size());
  const ReorderConfig& conf = config_.inputs(0).reorder_conf();
  const std::map<std::string, memory::format> mp = {
    {"nchw", memory::format::nchw},
    {"nhwc", memory::format::nhwc},
    {"chwn", memory::format::chwn}};
  CHECK(mapGet(conf.format_from(), mp, &botFmt_))
    << "Do not have format:" << conf.format_from();
  CHECK(mapGet(conf.format_to(), mp, &topFmt_))
    << "Do not have format:" << conf.format_to();
  CHECK(int(botFmt_) != int(topFmt_))
    << "the format should not equal, use mkldnn_reshape instead";
  bsIdx_ = conf.bs_index();
  CHECK_LT(bsIdx_, 4);
  CHECK_EQ(conf.dims_from_size(), 4);
  fromDims_.resize(4, 0);
  int cnt = 0;
  for (int i = 0; i < 4; ++i) {
    fromDims_[i] = conf.dims_from(i);
    if (fromDims_[i] < 1) {
      // which means not fixed size, will auto set it
      ++cnt;
    }
  }
  CHECK_LE(cnt, 2) << "allow 2 uncertain size at most";
}

size_t MkldnnReorderLayer::confirmBotDims(memory::dims& bot) {
  botDims_ = {fromDims_[0], fromDims_[1], fromDims_[2], fromDims_[3]};
  int uncertainIdx = -1;
  size_t sz = 1;
  for (int i = 0; i < 4; ++i) {
    if (fromDims_[i] > 0) {
      sz *= fromDims_[i];
    } else {
      if (i != bsIdx_)
        uncertainIdx = i;
    }
  }
  // firstly confirm bs
  if (fromDims_[bsIdx_] < 1) {
    CHECK_GT(bs_, 0) << "batchsize must lager than 0";
    // auto change the size of bs
    botDims_[bsIdx_] = bs_;
    sz *= bs_;
  }
  if (uncertainIdx != -1) {
    // have another uncertain size
    botDims_[uncertainIdx] = inputElmCnt_ / sz;
    CHECK_EQ(botDims_[uncertainIdx] * sz, inputElmCnt_) << "un-divisible";
    sz *= botDims_[uncertainIdx];
  }
  LOG(INFO) << getName() << ": bot size: " << botDims_[0] << ","<< botDims_[1]<<","
    <<botDims_[2]<<","<<botDims_[3];
  return sz;
}

void MkldnnReorderLayer::confirmTopDims(
  const memory::dims& bot, memory::dims& top) {
  if (botFmt_ == memory::format::nchw) {
    if (topFmt_ == memory::format::nhwc) {
      topDims_ = {botDims_[0], botDims_[2], botDims_[3], botDims_[1]};
    } else {
      topDims_ = {botDims_[1], botDims_[2], botDims_[3], botDims_[0]};
    }
  } else if (botFmt_ == memory::format::nhwc) {
    if (topFmt_ == memory::format::nchw) {
      topDims_ = {botDims_[0], botDims_[3], botDims_[1], botDims_[2]};
    } else {
      topDims_ = {botDims_[3], botDims_[1], botDims_[2], botDims_[0]};
    }
  } else if (botFmt_ == memory::format::chwn) {
    if (topFmt_ == memory::format::nchw) {
      topDims_ = {botDims_[3], botDims_[0], botDims_[1], botDims_[2]};
    } else {
      topDims_ = {botDims_[3], botDims_[1], botDims_[2], botDims_[0]};
    }
  } else {
    LOG(FATAL) << "error input format";
  }
}


void MkldnnReorderLayer::reshapeOutput() {
  // get input seqlen
  const Argument& input = getInput(0);
  if (input.hasMklSeq()) {
    CHECK(nullptr == input.sequenceStartPositions)
      << "should only have one: mklseq or paddleseq";
    seqLen_ = input.getMklSeqLen();
  } else if (input.sequenceStartPositions) {
    // if has seq, get aligned seq length
    CHECK(!input.hasSubseq()) << "Do not support sub seqence yet";
    seqLen_ = getPaddleAlignedSeqLen(input);
  } else {
    seqLen_ = 1;
  }

  // confirm the batch size firstly
  bs_ = inputMatH_ / seqLen_;
  outputMatH_ = inputMatH_;
  outputMatW_ = inputMatW_;
  // reshape the dims
  botDims_ = {0, 0, 0, 0};
  topDims_ = {0, 0, 0, 0};
  CHECK_EQ(confirmBotDims(botDims_), inputElmCnt_) << "bot size does not match";
  confirmTopDims(botDims_, topDims_);

  // reset the output
  config_.set_size(outputMatW_);
  resetOutput(outputMatH_, outputMatW_);
  ih_ = input.getFrameHeight();
  iw_ = input.getFrameWidth();
  // keep image setting if have
  if (ih_ > 0) {
    oh_ = ih_;
    getOutput().setFrameHeight(oh_);
  }
  if (iw_ > 0) {
    ow_ = iw_;
    getOutput().setFrameWidth(ow_);
  }
}

void MkldnnReorderLayer::resetDnnFwd() {
  engine eg = CpuEngine::Instance().getEngine();
  memory::data_type tp = memory::data_type::f32;
//  CHECK_EQ(real, float) << "only support f32";

  memory::desc botMD = memory::desc({botDims_}, tp, botFmt_);
  // do not use topdim as mkldnn test did
  memory::desc topMD = memory::desc({botDims_}, tp, topFmt_);

  real *botValueData = getInputValue(0)->getData();
  real *topValueData = getOutputValue()->getData();

  botVal_.reset(new memory(memory::primitive_desc(botMD, eg), botValueData));
  topVal_.reset(new memory(memory::primitive_desc(topMD, eg), topValueData));

  fwd_.reset(new reorder(*botVal_, *topVal_));

}

void MkldnnReorderLayer::resetDnnBwd() {
  engine eg = CpuEngine::Instance().getEngine();
  memory::data_type tp = memory::data_type::f32;

  memory::desc topMD = memory::desc({topDims_}, tp, topFmt_);
  // use top dim
  memory::desc botMD = memory::desc({topDims_}, tp, botFmt_);

  real *botDiffData = getDnnInputGrad(0)->getData();
  real *topDiffData = getDnnOutputGrad()->getData();

  botGrd_.reset(new memory(memory::primitive_desc(botMD, eg), botDiffData));
  topGrd_.reset(new memory(memory::primitive_desc(topMD, eg), topDiffData));

  bwd_.reset(new reorder(*topGrd_, *botGrd_));

}

void MkldnnReorderLayer::submitDnnFwd() { 
  real* botValueData = getInputValue(0)->getData();
  real* topValueData = getOutputValue()->getData();
  if (botValueData != botVal_->get_data_handle()) {
    botVal_->set_data_handle(botValueData);
  }
  if (topValueData != topVal_->get_data_handle()) {
    topVal_->set_data_handle(topValueData);
  }
  std::vector<primitive> pipeline;
  pipeline.push_back(*fwd_);
  stream(stream::kind::eager).submit(pipeline).wait();
}

void MkldnnReorderLayer::submitDnnBwd(const UpdateCallback& callback) {
  const MatrixPtr& botGrad = getDnnInputGrad(0);
  if (nullptr == botGrad) {
    return;
  }
  real* botDiffData = botGrad->getData();
  real* topDiffData = getOutputGrad()->getData();
  if (botDiffData != botGrd_->get_data_handle()) {
    botGrd_->set_data_handle(botDiffData);
  }
  if (topDiffData != topGrd_->get_data_handle()) {
    topGrd_->set_data_handle(topDiffData);
  }
  std::vector<primitive> pipeline;
  pipeline.push_back(*bwd_);
  stream(stream::kind::eager).submit(pipeline).wait();
}

}  // namespace paddle

