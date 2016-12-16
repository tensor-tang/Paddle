/* Copyright (c) 2016 */

#pragma once

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnMemory.h"

using namespace mkldnn;

namespace paddle {

static const std::string DNN_FORMAT[] = {
  "undef", "any", "blocked", "x", "nc", "nchw", "nhwc", "chwn", "nChw8c", //oIhw8i",
  "oi", "oihw", "ihwo", "OIhw8i8o", "OIhw8o8i", "Ohwi8o", "goihw", "gOIhw8i8o",
  "gOIhw8o8i"};

/**
 * @brief Base class of Dnnlayer.
 *
 */
class MkldnnLayer : public Layer {
public:
  /// For dnn engine
  engine engineCpu_;
  
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

  // flags whether to set memory format of top data or bot diff
  bool setDnnTopDataFmt_;
  std::vector<bool> setDnnBotDiffFmt_;

public:
  explicit MkldnnLayer(const LayerConfig& config)
    : Layer(config),
      engineCpu_(engine::cpu, 0),
      dataBot_(NULL),
      dataTop_(NULL),
      diffBot_(NULL),
      diffTop_(NULL),
      setDnnTopDataFmt_(false)
    {}

  ~MkldnnLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap) {
    /* Initialize the basic parent class */
    if (!Layer::init(layerMap, parameterMap)) return false;

    initDnnflags();
    return true;
  }

  /**
   * each dnn layer should have function 
   * to init the image size of input and output
   * (bs, ic, ih, iw, oc, oh, ow)
   * and other init for specific layers
   * before exit init()
   */
  virtual bool initShapeAndDnn(const LayerMap& layerMap, const ParameterMap& parameterMap) = 0;
  
  /** 
   * each dnn layer should have function
   * to reshape the size and data of output
   */
  virtual void reshapeOutput() = 0;
  
  /** 
   * each dnn layer should have function
   * to init or reset dnn forward
   */
  virtual void initOrResetDnnFwd() = 0;
  
  /** 
   * each dnn layer should have function
   * to init or reset dnn backward
   */
  virtual void initOrResetDnnBwd() = 0;
  
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
    const std::string dnn("mkldnn");
    if (!isNextLayerTypeEmpty()  // not empty
      && getNextLayerType().compare(0, dnn.length(), dnn) == 0 ) {
      // type started with "mkldnn"
      return true;
    }
    else {
      return false;
    }
  }

  bool isPrevLayerDnn(size_t idx) {
    const std::string dnn("mkldnn");
    if (getPrev(idx) != NULL
      && getPrev(idx)->getType().compare(0, dnn.length(), dnn) == 0 ) {
      // type started with "mkldnn"
      return true;
    }
    else {
      return false;
    }
  }

};

}  // namespace paddle
