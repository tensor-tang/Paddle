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
  bool useMkldnnWgt_;


public:
  explicit MkldnnLayer(const LayerConfig& config)
    : Layer(config),
      botData_(nullptr),
      botDiff_(nullptr),
      topData_(nullptr),
      topDiff_(nullptr),
      needResetBwd_(true),
      nextIsDnn_(false),
      prepareOnce_(true),
      useMkldnnWgt_(true)
    {}

  ~MkldnnLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap) {
    /* Initialize the basic parent class */
    if (!Layer::init(layerMap, parameterMap)) return false;

    // buffers
    botDims_ = {0};
    wgtDims_ = {0};
    biasDims_ = {0};
    topDims_ = {0};
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

    return initDnn(layerMap, parameterMap);
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

  /**
   * each dnn layer should have function 
   * to init weight
   */
  virtual bool initDnn(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) = 0;

  /** 
   * each dnn layer should have function
   * to init or reset forward mkldnn
   */
  virtual void resetDnnFwd() = 0;

  /** 
   * each dnn layer should have function
   * to init or reset backward mkldnn
   */
  virtual void resetDnnBwd() = 0;

  virtual void submitDnnFwd() = 0;
  virtual void submitDnnBwd(const UpdateCallback& callback) = 0;


  // reload the settings from proto
  virtual void loadConfig() = 0;

  // reshape 
  // output matrix height and width 
  // and the bs
  // and the output buffer
  virtual void reshapeOutput() = 0;

  void forward(PassType passType) {
    if (inputElmCnt_ != getInputValue(0)->getElementCnt()) {
      VLOG(1) << "reshape mkldnn layout of layer: " << getName();
      if (prepareOnce_) {
        initDnnflags();
        prepareOnce_ = false;
      }
      // get new input size info
      inputElmCnt_ = getInputValue(0)->getElementCnt();
      inputMatW_ = getInputValue(0)->getWidth();
      inputMatH_ = getInputValue(0)->getHeight();
      CHECK_EQ(inputElmCnt_, inputMatW_ * inputMatH_);

      // reshape output buffer and need re-calculate bs and image size info
      reshapeOutput();
      resetOutput(outputMatH_, outputMatW_);
      CHECK_EQ(outputMatW_, getSize())
        << "maybe forget to set new layersize when changed it";

      // resest MKLDNN layout
      resetDnnFwd();

      // print the data flow
      if (botData_ && topData_) {
        VLOG(1) << "data format flow --- "
          << DNN_FMTS[botData_->getUserFmt()] << " >>> ("
          << DNN_FMTS[botData_->getIntlFmt()] << " >>> "
          << DNN_FMTS[topData_->getIntlFmt()] << ") >>> "
          << DNN_FMTS[topData_->getUserFmt()];
      }
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
      CHECK(useMkldnnWgt_) << "use mkldnn wgt for better perf";
      needResetBwd_ = false;
      // mkldnn init or reset backward
      VLOG(1) << "reset backward batch size to " << bs_
        << " of mkldnn layer: " << getName();

      gatherTopDiffs();
      resetDnnBwd();

      // print the diff flow
      if (botDiff_ && topDiff_) {
        VLOG(1) << "diff format flow --- "
          << DNN_FMTS[botDiff_->getUserFmt()] << " <<< ("
          << DNN_FMTS[botDiff_->getIntlFmt()] << " <<< "
          << DNN_FMTS[topDiff_->getIntlFmt()] << ") <<< "
          << DNN_FMTS[topDiff_->getUserFmt()];
      }
    }

    {
      //REGISTER_TIMER_DYNAMIC("Bwd_" + getName());
      REGISTER_TIMER_INFO("mkldnn_BwdTimer", getName().c_str());
      submitDnnBwd(callback);
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
