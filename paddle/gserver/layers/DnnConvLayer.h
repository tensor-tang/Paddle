/* Copyright (c) 2016 */

#pragma once

#include "ConvBaseLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>

#include <numeric>
#include "mkldnn.hpp"

#include "DnnMemory.h"

using namespace mkldnn;

namespace paddle {

/**
 * @brief A subclass of convolution layer.
 *
 * The config file api is img_conv_layer.
 */
class DnnConvLayer : public ConvBaseLayer {

protected:
  /// 
  bool dnnInited;
  /// For dnn engine
  engine engineCpu_;
  /// For dnn convolution.
  std::shared_ptr<convolution_forward::primitive_desc> fwdPD_; //Primitive Desc

//  dnnPrimitive_t convFwd_;
//  dnnPrimitive_t convBwdData_;
//  dnnPrimitive_t convBwdFilter_;
//  dnnPrimitive_t convBwdBias_;
  
  /// data buffers
  DnnBufferPtr dataBot_;
  DnnBufferPtr dataWgt_;
  DnnBufferPtr dataBias_;
  DnnBufferPtr dataTop_;
  /// diff buffer
  DnnBufferPtr diffBot_;
  DnnBufferPtr diffWgt_;
  DnnBufferPtr diffBias_;
  DnnBufferPtr diffTop_;

  
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
  explicit DnnConvLayer(const LayerConfig& config)
    : ConvBaseLayer(config),
      dnnInited(false),
      engineCpu_(engine::cpu, 0),
      fwdPD_(NULL),
      dataBot_(new DnnBuffer()),
      dataWgt_(new DnnBuffer()),
      dataBias_(new DnnBuffer()),
      dataTop_(new DnnBuffer()),
      diffBot_(new DnnBuffer()),
      diffWgt_(new DnnBuffer()),
      diffBias_(new DnnBuffer()),
      diffTop_(new DnnBuffer())
    {}

  ~DnnConvLayer() {
    /*
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
    */
  }
  
  /// for dnn
  void initDnn();
  int dimSize(const memory::dims &t) {
    int sz = 1;
    for (size_t i = 0; i < t.size(); ++i) 
      sz *= t[i];
    return sz;
  }

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  size_t getSize();
  void forward(PassType passType);
  void backward(const UpdateCallback& callback);
};

}  // namespace paddle
