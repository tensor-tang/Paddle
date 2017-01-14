/* Copyright (c) 2016 */

#pragma once

#include "Layer.h"
#include "paddle/math/Matrix.h"
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
  MkldnnBufferPtr dataBot_;
  MkldnnBufferPtr dataTop_;
  /// diff buffer
  MkldnnBufferPtr diffBot_;
  MkldnnBufferPtr diffTop_;

  // The spatial dimensions of height and width of input feature map.
  std::vector<int> ih_, iw_;
  // The spatial dimensions of height and width of output feature map.
  std::vector<int> oh_, ow_;
  // input channel number
  std::vector<int> ic_;
  // output channels
  int oc_;
  // batchsize
  int bs_;

  bool needResetBwd_;

  // flags whether to set memory format of top data or bot diff
  bool setDnnTopDataFmt_;
  std::vector<bool> setDnnBotDiffFmt_;

public:
  explicit MkldnnLayer(const LayerConfig& config)
    : Layer(config),
      dataBot_(NULL),
      dataTop_(NULL),
      diffBot_(NULL),
      diffTop_(NULL),
      needResetBwd_(true),
      setDnnTopDataFmt_(false)
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

    return initDnn(layerMap, parameterMap);
  }

  void forward(PassType passType) {
    Layer::forward(passType);

    // reshape if batchsize changes
    if (reshapeOutput()) {
      // dnn fwd init or reset
      resetDnnFwd(passType);
      needResetBwd_ = true;
    }

    // submit dnn forward
    submitDnnFwd(passType);
  }

  void backward(const UpdateCallback& callback) {
    if (needResetBwd_) {
      // dnn fwd init or reset
      resetDnnBwd();
      needResetBwd_ = false;
    }

    // submit dnn backward
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
    if (hasActivation()) {
      // so far has mkldnn relu and softmax activations
      // activation mkldnn format: input == output
      // relu: support nchw, nc, nChw8c and so on
      // softmax: support nchw and nc
      return hasMkldnnAct();
    }
    const std::string dnn("mkldnn");
    if (!isNextLayerTypeEmpty()  // not empty
      && getNextLayerType().compare(0, dnn.length(), dnn) == 0 ) {
      // type started with "mkldnn"
      return true;
    } else {
      return false;
    }
  }

  bool isPrevLayerDnn(size_t idx) {
    if (getPrev(idx) == NULL)
      return false;
    if (getPrev(idx)->hasActivation()) {
      return getPrev(idx)->hasMkldnnAct();
    }
    const std::string dnn("mkldnn");
    // type started with "mkldnn"
    return getPrev(idx)->getType().compare(0, dnn.length(), dnn) == 0 ?
      true : false;
  }

  // for conv only support caffe mode by now
  // TODO(TJ): figure out why paddle pool use false caffe mode
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
   * to reshape the size and data of output if batchsize changes
   * return false if donot need reshape 
   */
  virtual bool reshapeOutput() = 0;

  /** 
   * each dnn layer should have function
   * to init or reset dnn forward
   */
  virtual void resetDnnFwd(PassType passType) = 0;

  /** 
   * each dnn layer should have function
   * to init or reset dnn backward
   */
  virtual void resetDnnBwd() = 0;

  virtual void submitDnnFwd(PassType passType) = 0;
  virtual void submitDnnBwd(const UpdateCallback& callback) = 0;
};

}  // namespace paddle
