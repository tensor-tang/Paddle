/* Copyright (c) 2016 */

#pragma once

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Stat.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnBase.h"
#include "MkldnnMemory.h"

namespace paddle {

class MkldnnLayer;
typedef std::shared_ptr<MkldnnLayer> MkldnnLayerPtr;

/**
 * @brief Base class of Dnnlayer.
 *
 */
class MkldnnLayer : public Layer {
public:
  /// data buffers
  // TODO(TJ): need vector when know how RNN works
  MkldnnBufferPtr dataBot_;
  MkldnnBufferPtr dataTop_;
  /// diff buffer
  MkldnnBufferPtr diffBot_;
  MkldnnBufferPtr diffTop_;

  // dims and format for user buffer
  std::vector<mkldnn::memory::dims> botDims_, wgtDims_, biasDims_;
  std::vector<mkldnn::memory::format> botFmt_, wgtFmt_, biasFmt_;
  mkldnn::memory::dims topDims_;
  mkldnn::memory::format topFmt_;

  // The spatial dimensions of height and width of input feature map.
  std::vector<int> ih_, iw_;
  // The spatial dimensions of height and width of output feature map.
  std::vector<int> oh_, ow_;  // TODO(TJ): no need vector??
  // input channel number
  std::vector<int> ic_;
  // output channels
  int oc_;
  // batchsize
  int bs_;

  // flags whether to set memory format of top data or bots diff
  // only one top data but may have several bot diff
  bool setDnnTopDataFmt_;
  std::vector<bool> setDnnBotDiffFmt_;

  // each MKLDNN layers has WriteToMode and AddToMode
  // use WriteToMode if addSize_ == 0, otherwise use AddToMode
  int addSize_;

  // layers with weight have an option to choose
  // whether use mkldnn foramt weight to get a better performance
  // sacrificing the compatibility with original CPU layers
  bool useMkldnnWgt_;

  bool needResetBwd_;
  // the initflags functions should only be called once
  bool hasInitFlags_;

public:
  explicit MkldnnLayer(const LayerConfig& config)
    : Layer(config),
      dataBot_(nullptr),
      dataTop_(nullptr),
      diffBot_(nullptr),
      diffTop_(nullptr),
      setDnnTopDataFmt_(false),
      addSize_(0),
      useMkldnnWgt_(true),
      needResetBwd_(true),
      hasInitFlags_(false)
    {}

  ~MkldnnLayer() {}

  mkldnn::memory::desc getAnyMD(mkldnn::memory::dims & dm,
    mkldnn::memory::data_type tp = mkldnn::memory::data_type::f32) {
    return mkldnn::memory::desc({dm}, tp, mkldnn::memory::format::any);
  }

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap) {
    /* Initialize the basic parent class */
    if (!Layer::init(layerMap, parameterMap)) return false;

    bs_ = 0;
    oc_ = 0;
    topDims_ = {0};
    topFmt_ = mkldnn::memory::format::nchw;
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      botDims_.push_back({0});
      wgtDims_.push_back({0});
      biasDims_.push_back({0});
      botFmt_.push_back(mkldnn::memory::format::nchw);
      wgtFmt_.push_back(mkldnn::memory::format::format_undef);
      biasFmt_.push_back(mkldnn::memory::format::x);
    }

    return initDnn(layerMap, parameterMap);
  }

  void forward(PassType passType) {
    Layer::forward(passType);

    // reshape if batch size changes
    if (bs_ == getInput(0).getBatchSize()) {
      // choose to clear top data or top diff
      clearDataDiff();
    } else {
      if (!hasInitFlags_) {
        // should not be called at init function
        // this function can work only after all layers init done
        // and should be called only once
        initDnnflags();
        hasInitFlags_ = true;
      }

      bs_ = getInput(0).getBatchSize();
      VLOG(1) << "reset forward batch size to " << bs_
        << " of mkldnn layer: " << getName();

      // reshape the input and output size
      REGISTER_TIMER_INFO("mkldnn_ResetDnnTimer", getName().c_str());
      reshape();
      printInfo();

      // mkldnn init or reset forward
      resetOutput(bs_, getSize());
      resetDnnFwd(passType);

      // print the data flow
      for (size_t i = 0; i != inputLayers_.size(); ++i) {
        // TODO(TJ): consider multi input
        if (dataBot_ && dataTop_)
          VLOG(1) << "data format flow --- "
            << DNN_FMTS[dataBot_->getUserFmt()] << " >>> ("
            << DNN_FMTS[dataBot_->getIntlFmt()] << " >>> "
            << DNN_FMTS[dataTop_->getIntlFmt()] << ") >>> "
            << DNN_FMTS[dataTop_->getUserFmt()];
        // in batch norm layer, the other two will be moving mean and var
        if (getType() == "mkldnn_batch_norm")
          break;
      }
      needResetBwd_ = true;
    }

    // all sumbit cvt should be clear
    clearAllDnnCvtFlags();
    // then submit dnn forward
    REGISTER_TIMER_INFO("mkldnn_FwdTimer", getName().c_str());
    submitDnnFwd(passType);
  }

  void backward(const UpdateCallback& callback) {
    if (needResetBwd_) {
      needResetBwd_ = false;
      // mkldnn init or reset backward
      VLOG(1) << "reset backward batch size to " << bs_
        << " of mkldnn layer: " << getName();
      resetDnnBwd();

      // print the diff flow
      for (size_t i = 0; i != inputLayers_.size(); ++i) {
        // TODO(TJ): consider multi input
        if (diffBot_ && diffTop_)
          VLOG(1) << "diff format flow --- "
            << DNN_FMTS[diffBot_->getUserFmt()] << " <<< ("
            << DNN_FMTS[diffBot_->getIntlFmt()] << " <<< "
            << DNN_FMTS[diffTop_->getIntlFmt()] << ") <<< "
            << DNN_FMTS[diffTop_->getUserFmt()];
        // in batch norm layer, the other two will be moving mean and var
        if (getType() == "mkldnn_batch_norm")
          break;
      }
    }

    // submit dnn backward
    REGISTER_TIMER_INFO("mkldnn_BwdTimer", getName().c_str());
    submitDnnBwd(callback);
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
    setDnnTopDataFmt_ = areNextAllDnn();
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
      setDnnBotDiffFmt_.push_back(isPrevDnn(i));
    }
  }

  // for conv only support caffe mode by now
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
   */
  int outputSize(int imageSize, int filterSize, int padding, int stride,
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

  /**
   * each dnn layer should have function 
   * to init the image size of input and output
   * (bs, ic, ih, iw, oc, oh, ow)
   * and other init for specific layers
   * before exit init()
   */
  virtual bool initDnn(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) = 0;

  /**
   * each dnn layer should have function
   * to reshape the input and output size, when batch size changes
   */
  virtual void reshape() = 0;

  /**
   * print some info like input or output size
   */
  virtual void printInfo() {
    for (size_t i = 0; i < ih_.size(); ++i) {
      VLOG(2)
        << "ih: " << ih_[i] << ", iw: " << iw_[i]
        << ", ic: " << ic_[i]
        << ", oh: " << oh_[i] << ", ow: " << ow_[i]
        << ", oc: " << oc_;
    }
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

  /** 
   * each dnn layer should have function
   * to clear the top data and diff, or choose to reseve.
   * Choose to use reserveOutput or resetOutput
   */
  // TODO(TJ): maybe can remove it
  // when confirm whether need to clear topdiff and how multi inputs work
  virtual void clearDataDiff() = 0;

  /** 
   * each dnn layer should have function
   * to clear all the MkldnnBuffer cvt flags
   */
  virtual void clearAllDnnCvtFlags() {
    if (dataBot_) dataBot_->clearCvtFlag();
    if (dataTop_) dataTop_->clearCvtFlag();
    if (diffBot_) diffBot_->clearCvtFlag();
    if (diffTop_) diffTop_->clearCvtFlag();
  }

  virtual void submitDnnFwd(PassType passType) = 0;
  virtual void submitDnnBwd(const UpdateCallback& callback) = 0;
};

}  // namespace paddle
