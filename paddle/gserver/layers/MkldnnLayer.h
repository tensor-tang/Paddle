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

  // flags whether to set memory format of top data or bot diff
  bool setDnnTopDataFmt_;
  std::vector<bool> setDnnBotDiffFmt_;

  // each MKLDNN layers has WriteToMode and AddToMode
  // use WriteToMode if addSize_ == 0, otherwise use AddToMode
  int addSize_;

  // layers with weight have an option to choose
  // whether use mkldnn format to get a better performance
  // sacrificing the compatibility with original CPU layers
  bool useMkldnnFmt_;

  bool needResetBwd_;

public:
  explicit MkldnnLayer(const LayerConfig& config)
    : Layer(config),
      dataBot_(nullptr),
      dataTop_(nullptr),
      diffBot_(nullptr),
      diffTop_(nullptr),
      setDnnTopDataFmt_(false),
      addSize_(0),
      useMkldnnFmt_(false),
      needResetBwd_(true)
    {}

  ~MkldnnLayer() {}

  mkldnn::memory::desc getAnyMD(mkldnn::memory::dims & dm,
    mkldnn::memory::data_type tp = mkldnn::memory::data_type::f32) {
    return mkldnn::memory::desc({dm}, tp, mkldnn::memory::format::any);
  }

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap) {
    /* Initialize the basic parent class */
    if (!Layer::init(layerMap, parameterMap)) return false;

    initDnnflags();

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
      bs_ = getInput(0).getBatchSize();
      LOG(INFO) << "reset forward batch size to " << bs_ 
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
      LOG(INFO) << "reset backward batch size to " << bs_ 
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
   */
  virtual void initDnnflags() {
    setDnnTopDataFmt_ = isNextLayerDnn();
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
      setDnnBotDiffFmt_.push_back(isPrevLayerDnn(i));
    }
  }

  bool isNextLayerDnn() {
    bool useMkldnnAct = true;
    if (hasActivation()) {
      // so far has mkldnn relu and softmax activations
      // activation mkldnn format: input == output
      // relu: support nchw, nc, nChw8c and so on
      // softmax: support nchw and nc
      // since activaion do not change format, so alos depends on next layer
      useMkldnnAct = hasMkldnnAct();
    }
    const std::string dnn("mkldnn");
    return (!isNextLayerTypeEmpty()
      // and type started with "mkldnn"
      && getNextLayerType().compare(0, dnn.length(), dnn) == 0) ?
      useMkldnnAct : false;
  }

  bool isPrevLayerDnn(size_t idx) {
    if (getPrev(idx) == NULL || getPrev(idx)->getType().empty())
      return false;
    bool useMkldnnAct = true;
    if (getPrev(idx)->hasActivation()) {
      useMkldnnAct = getPrev(idx)->hasMkldnnAct();
    }
    const std::string dnn("mkldnn");
    // type started with "mkldnn"
    return getPrev(idx)->getType().compare(0, dnn.length(), dnn) == 0 ?
      useMkldnnAct : false;
  }

  // for conv only support caffe mode by now
  // TODO(TJ): figure out why pool use false caffe mode
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
  virtual void clearAllDnnCvtFlags() = 0;

  virtual void submitDnnFwd(PassType passType) = 0;
  virtual void submitDnnBwd(const UpdateCallback& callback) = 0;
};

}  // namespace paddle
