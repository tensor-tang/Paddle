/* Copyright (c) 2017 */

#pragma once

#include "Layer.h"
#include "paddle/math/Matrix.h"
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
  /// bottom data and diff buffers, can be array
  std::vector<MkldnnBufferPtr> botDatas_;
  std::vector<MkldnnBufferPtr> botDiffs_;

  /// top data and diff buffers
  MkldnnBufferPtr topData_;
  MkldnnBufferPtr topDiff_;  // top diff for backward data
  // in some conditions it may have different format with topdiff backward data
  MkldnnBufferPtr topDiffBwdWgt_;  // top diff for backward weight, if needed

  // dims and format for user buffer
  std::vector<mkldnn::memory::dims> botDims_, wgtDims_, biasDims_;
  std::vector<mkldnn::memory::format> botFmt_, wgtFmt_, biasFmt_;
  mkldnn::memory::dims topDims_;
  mkldnn::memory::format topFmt_;

  /// for summing topdiffs
  std::vector<MkldnnBufferPtr> topDiffBuffers_;
  std::shared_ptr<mkldnn::sum> sumTopDiffs_;
  // tmp result of sum
  MkldnnBufferPtr tmpDiff_;

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
  bool nextIsDnn_;
  std::vector<bool> prevIsDnn_;

  // each MKLDNN layers has WriteToMode and AddToMode
  // use WriteToMode if addSize_ == 0, otherwise use AddToMode
  // TODO(TJ): maybe can remove this
  int addSize_;

  // layers with weight have an option to choose
  // whether use mkldnn foramt weight to get a better performance
  // sacrificing the compatibility with original CPU layers
  bool useMkldnnWgt_;

  bool needResetBwd_;

  // some operations should not be called at init function
  // and should only do once : like initflags and prepare topdiffMD etc.
  bool prepareOnce_;

public:
  explicit MkldnnLayer(const LayerConfig& config)
    : Layer(config),
      topData_(nullptr),
      topDiff_(nullptr),
      topDiffBwdWgt_(nullptr),
      nextIsDnn_(false),
      addSize_(0),
      useMkldnnWgt_(true),
      needResetBwd_(true),
      prepareOnce_(true)
    {}

  ~MkldnnLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

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
  virtual void clearAllDnnCvtFlags();

  /**
   * if have several input topdiffs
   * then create handle to sum them
   */
  virtual void gatherTopDiff();

  void forward(PassType passType);

  void backward(const UpdateCallback& callback);

  /**
   * print some info like input or output size
   */
  virtual void printInfo();

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
   * each dnn layer should have function
   * to init or reset forward mkldnn
   */
  virtual void resetDnnFwd() = 0;

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

  virtual void submitDnnFwd() = 0;
  virtual void submitDnnBwd(const UpdateCallback& callback) = 0;
};

}  // namespace paddle
