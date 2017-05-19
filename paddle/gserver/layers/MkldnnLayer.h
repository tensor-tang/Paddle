/* Copyright (c) 2017 */

#pragma once

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnBase.h"
#include "MkldnnMemory.h"

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

class MkldnnLayer;
typedef std::shared_ptr<MkldnnLayer> MkldnnLayerPtr;

/**
 * @brief Base class of Dnnlayer.
 *
 */
class MkldnnLayer : public Layer {
public:
  /// bottom data and diff buffers
  MkldnnBufferPtr botData_, botDiff_;
  /// top data and diff buffers
  MkldnnBufferPtr topData_, topDiff_;

  /// dims and format for user buffer
  mkldnn::memory::dims botDims_, wgtDims_, biasDims_, topDims_;
  mkldnn::memory::format botFmt_, wgtFmt_, biasFmt_, topFmt_;
  mkldnn::engine engine_;

  // input element cnt
  size_t inputElmCnt_;
  // the input matrix height and width
  size_t inputMatH_, inputMatW_;
  // the output matrix height and width
  // height: mklSeqLen * bs
  // width : layer size == oc * oh * ow
  size_t outputMatH_, outputMatW_;

  // MKLDNN aligned seqLen
  int seqLen_;
  // batchsize
  int bs_;
  // input image channel, height and width
  int ic_, ih_, iw_;
  // output image channel, height and width
  int oc_, oh_, ow_;

  bool needResetBwd_;
  bool hasInitedWgt_;

  /******
   * for support mixed with cpu layers */
  // flags whether to set memory format of top data or bots diff
  // only one top data but may have several bot diff
  bool nextIsDnn_;
  std::vector<bool> prevIsDnn_;

  // some operations should not be called at init function
  // and should only do once : like initflags and prepare topdiffMD etc.
  bool prepareOnce_;

  /******
   * for support cpu weights */
  // layers with weight have an option to choose
  // whether use mkldnn foramt weight to get a better performance
  // sacrificing the compatibility with original CPU layers
  bool testWithPaddleWgt;


public:
  explicit MkldnnLayer(const LayerConfig& config)
    : Layer(config),
      botData_(nullptr),
      botDiff_(nullptr),
      topData_(nullptr),
      topDiff_(nullptr),
      engine_(mkldnn::engine::cpu, 0),
      needResetBwd_(true),
      hasInitedWgt_(false),
      nextIsDnn_(false),
      prepareOnce_(true),
      testWithPaddleWgt(false)
    {}

  ~MkldnnLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap) {
    /* Initialize the basic parent class */
    if (!Layer::init(layerMap, parameterMap)) return false;

    // buffers
    botDims_ = {};
    wgtDims_ = {};
    biasDims_ = {};
    topDims_ = {};
    botFmt_ = mkldnn::memory::format::nchw;
    wgtFmt_ = mkldnn::memory::format::format_undef;
    biasFmt_ = mkldnn::memory::format::x;
    topFmt_ = mkldnn::memory::format::nchw;

    inputElmCnt_ = 0;
    bs_ = 0; seqLen_ = 0;
    oc_ = 0; ih_ = 0; iw_ = 0;
    ic_ = 0; oh_ = 0; ow_ = 0;

    // load from proto setting
    loadConfig();

    return initDnnWgt(layerMap, parameterMap);
  }


  // reload the settings from proto
  virtual void loadConfig() = 0;

  /**
   * each dnn layer should have function 
   * to init weight
   */
  virtual bool initDnnWgt(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) = 0;

  /** 
   * each dnn layer should have function
   * to clear all the MkldnnBuffer cvt flags
   */
  virtual void clearAllDnnCvtFlags() {
    if (botData_) botData_->clearCvtFlag();
    if (botDiff_) botDiff_->clearCvtFlag();
    if (topData_) topData_->clearCvtFlag();
    if (topDiff_) topDiff_->clearCvtFlag();
  }

  // reset activation fwd
  virtual void resetDnnFwdAct() {
    if (!hasMkldnnAct()) {
      return;
    }
    std::shared_ptr<mkldnn::memory::desc> md(
      new mkldnn::memory::desc(topData_->getUserMD()));
    CHECK(md);
    activation_->resetDnnFwd(output_, std::static_pointer_cast<void>(md));
  }

  // reset activation bwd
  virtual void resetDnnBwdAct() {
    if (!hasMkldnnAct()) {
      return;
    }
    std::shared_ptr<mkldnn::memory::desc> md(
      new mkldnn::memory::desc(topDiff_->getUserMD()));
    CHECK(md);
    activation_->resetDnnBwd(output_, std::static_pointer_cast<void>(md));
  }

  /** 
   * each dnn layer should have function
   * to init or reset forward mkldnn
   */
  virtual void resetDnnFwd(PassType passType) = 0;

  /** 
   * each dnn layer should have function
   * to init or reset backward mkldnn
   */
  virtual void resetDnnBwd() = 0;

  virtual void submitDnnFwd() = 0;

  virtual void submitDnnBwd(const UpdateCallback& callback) = 0;

  // reshape output info:
  // output matrix height and width 
  // bs and sometimes seqlen
  virtual void reshapeOutputInfo() = 0;

  // the activation will auto call dnn act if have
  virtual void forwardDnnAct() {
    forwardActivation();
  }

  // the activation will auto call dnn act if have
  virtual void BackwardDnnAct() {
    backwardActivation();
  }

  // reshape the buffer of output
  virtual void reshapeOutputBuffer() {
    CHECK_EQ(outputMatW_, getSize())
      << "maybe forget to set new layersize when reshape output info";
    resetOutput(outputMatH_, outputMatW_);
  }

  void forward(PassType passType) {
    if (inputElmCnt_ != getInputValue(0)->getElementCnt()) {
      VLOG(1) << "reshape mkldnn fwd of layer: " << getName();

      if (testWithPaddleWgt && passType != PASS_TEST) {
        LOG(WARNING) << "testWithPaddleWgt is invalid when training";
      }

      if (prepareOnce_) {
        prepareOnce_ = false;
        prepare();
      }

      updateInputInfo();

      reshapeOutputInfo();

      reshapeOutputBuffer();

      resetDnnFwd(passType);

      resetDnnFwdAct();

      printDataFlow();

      needResetBwd_ = true;

      printInfo();
    }

    {
      //REGISTER_TIMER_DYNAMIC("Fwd_" + getName());
      REGISTER_TIMER_INFO("mkldnn_FwdTimer", getName().c_str());
      Layer::forward(passType);
      // all sumbit cvt should be clear
      clearAllDnnCvtFlags();
      // then submit dnn forward
      submitDnnFwd();
    }
  }

  void backward(const UpdateCallback& callback) {
    if (needResetBwd_) {
      needResetBwd_ = false;
      VLOG(1) << "reshape mkldnn bwd of layer: " << getName();

      gatherTopDiffs();

      resetDnnBwd();

      resetDnnBwdAct();

      printDiffFlow();      
    }
    {
      //REGISTER_TIMER_DYNAMIC("Bwd_" + getName());
      REGISTER_TIMER_INFO("mkldnn_BwdTimer", getName().c_str());
      submitDnnBwd(callback);
    }
  }

protected:
  // TODO: add comment or rename
  void prepare() {
    dnnOutGrads_.resize(nextLayers_.size(), nullptr);
    for (size_t i = 0; i < nextLayers_.size(); ++i) {
      topDiffMDs_.push_back(nullptr);
      dnnOutIdxMap_[nextLayers_[i]->getName()] = i;
    //  LOG(INFO)<<"next name:" << nextLayers_[i]->getName();
    }
    if (nextLayers_.size() > 0 && topDiffMDs_.size() > nextLayers_.size()) {
      // in base layer init will add one nullptr for PASS_grad check
      // so remove the redundant one
      topDiffMDs_.pop_back();
      CHECK_EQ(topDiffMDs_.size(), nextLayers_.size());
    } else {
      CHECK_EQ(topDiffMDs_.size() - 1, nextLayers_.size());
    }
    initDnnflags();
  }

  /**
   * init the flags whether to set memory desc
   * of top data or bot diff.
   * each layer can have its own implements.
   * Caution: Do not call it at init function
   *          this function can work only after all layers init have done
   */
  void initDnnflags() {
    // set topdata internal only if all next layers are MKLDNN layers
    nextIsDnn_ = areNextAllDnn();
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
      prevIsDnn_.push_back(isPrevDnn(i));
    }
  }

  void updateInputInfo() {
    inputElmCnt_ = getInputValue(0)->getElementCnt();
    inputMatW_ = getInputValue(0)->getWidth();
    inputMatH_ = getInputValue(0)->getHeight();
    CHECK_EQ(inputElmCnt_, inputMatW_ * inputMatH_);
  }

  void printDataFlow() {
    if (botData_ && topData_) {
      VLOG(1) << "data format flow --- "
        << DNN_FMTS[botData_->getUserFmt()] << " >>> ("
        << DNN_FMTS[botData_->getIntlFmt()] << " >>> "
        << DNN_FMTS[topData_->getIntlFmt()] << ") >>> "
        << DNN_FMTS[topData_->getUserFmt()];
    }
  }

  void printDiffFlow() {
    // print the diff flow
    if (botDiff_ && topDiff_) {
      VLOG(1) << "diff format flow --- "
        << DNN_FMTS[botDiff_->getUserFmt()] << " <<< ("
        << DNN_FMTS[botDiff_->getIntlFmt()] << " <<< "
        << DNN_FMTS[topDiff_->getIntlFmt()] << ") <<< "
        << DNN_FMTS[topDiff_->getUserFmt()];
    }
  }

  /**
   * if have several input topdiffs
   * then create handle to sum them
   */
  virtual void gatherTopDiffs() {
    // TODO(TJ): enable it if want to support models with branches
    // like resnet and googlenet
  };

  /**
   * print some info like input or output size
   */
  virtual void printInfo() {
    VLOG(2) << "bs: " << bs_
      << ", ic: " << ic_ << ", ih: " << ih_ << ", iw: " << iw_
      << ", oc: " << oc_ << ", oh: " << oh_ << ", ow: " << ow_;
  }

  // get the aligned seq length from paddle sequence info
  // and the length among batchsize should be the same
  int getPaddleAlignedSeqLen(const Argument& arg) {
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

  /**
   * Calculate output size based on caffeMode_.
   * - input(+padding): 0123456789
   * - imageSize(+padding) = 10;
   * - filterSize = 3;
   * - stride = 2;
   * - caffeMode_ is true:
       - output: (012), (234), (456), (678)
       - outputSize = 4;
   * - caffeMode_ is false:
   *   - output: (012), (234), (456), (678), (9)
   *   - outputSize = 5;
   *** for conv only support caffe mode by now
   */
  int getOutputSize(int imageSize, int filterSize, int padding, int stride,
                       bool caffeMode = true) {
    int outputSize;
    if (!caffeMode) {
      outputSize =
          (imageSize - filterSize + 2 * padding + stride - 1) / stride + 1;
    } else {
      outputSize = (imageSize - filterSize + 2 * padding) / stride + 1;
    }
    CHECK_GE(outputSize, 1);
    return outputSize;
  }


};

}  // namespace paddle
