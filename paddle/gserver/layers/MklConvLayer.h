/* Copyright (c) 2016 */

#pragma once

#include "ConvBaseLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "MklWrapper.h"
#include "MklMemory.h"

namespace paddle {

/**
 * @brief A subclass of convolution layer.
 * This layer expands input and use matrix multiplication to
 * calculate convolution operation.
 *
 * The config file api is img_conv_layer.
 */
class MklConvLayer : public ConvBaseLayer {
protected:
  /// For dnn convolution.
  dnnPrimitive_t convFwd_;
  dnnPrimitive_t convBwdData_;
  dnnPrimitive_t convBwdFilter_;
  dnnPrimitive_t convBwdBias_;
  /// data buffers
  MklBufferPtr dataBottom_;
  MklBufferPtr dataFilter_;
  MklBufferPtr dataBias_;
  MklBufferPtr dataTop_;
  /// diff buffer
  MklBufferPtr diffBottom_;
  MklBufferPtr diffFilter_;
  MklBufferPtr diffBias_;
  MklBufferPtr diffTop_;

  /// resource
  void *resConv_[dnnResourceNumber];
  /// has bias
  bool hasBias_;

  /// The spatial dimensions of height of input feature map.
  IntV imgSizeH_;
  /// The spatial dimensions of width of input feature map.
  IntV imgSizeW_;
  /// The spatial dimensions of height of output feature map.
  IntV outputH_;
  /// The spatial dimensions of width of output feature map.
  IntV outputW_;

  /// subM_ = numFilters_ / groups_.
  IntV subM_;
  /// subN_ = outputH_ * outputW_.
  IntV subN_;
  /// subK_ = channels_ * filterPixels_ * groups_.
  IntV subK_;

public:
  explicit MklConvLayer(const LayerConfig& config)
    : ConvBaseLayer(config),
      dataBottom_(new MklBuffer()),
      dataFilter_(new MklBuffer()),
      dataBias_(new MklBuffer()),
      dataTop_(new MklBuffer()),
      diffBottom_(new MklBuffer()),
      diffFilter_(new MklBuffer()),
      diffBias_(new MklBuffer()),
      diffTop_(new MklBuffer())
    {}

  ~MklConvLayer() {
    // release all dnn
    if (convFwd_) {
      dnnDelete(convFwd_);
    }
    if (convBwdBias_) {
      dnnDelete(convBwdBias_);
    }
    if (convBwdData_) {
      dnnDelete(convBwdData_);
    }
    if (convBwdFilter_) {
      dnnDelete(convBwdFilter_);
    }
  }
  
  /// for dnn
  void initDnn();

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  size_t getSize();
  void forward(PassType passType);
  void backward(const UpdateCallback& callback);
};

}  // namespace paddle
