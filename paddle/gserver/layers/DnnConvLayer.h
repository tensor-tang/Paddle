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
  bool dnnFwdInited_;
  bool dnnBwdInited_;
  /// For dnn engine
  engine engineCpu_;
  /// For dnn convolution. Primitive Desc
  std::shared_ptr<convolution_forward::primitive_desc> fwdPD_;
  std::shared_ptr<convolution_backward_data::primitive_desc> bwdDataPD_;
  std::shared_ptr<convolution_backward_weights::primitive_desc> bwdWgtPD_;
  
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

  // The spatial dimensions of height of input feature map.
  IntV ih_;
  // The spatial dimensions of width of input feature map.
  IntV iw_;
  // The spatial dimensions of height of output feature map.
  IntV oh_;
  // The spatial dimensions of width of output feature map.
  IntV ow_;
  // padding, stride and filter size
  IntV ph_, pw_;
  IntV sh_, sw_;
  IntV fh_, fw_;
  // input channel and group
  IntV ic_, gp_;
  // output channel == filter number
  int oc_;
public:
  explicit DnnConvLayer(const LayerConfig& config)
    : ConvBaseLayer(config),
      dnnFwdInited_(false),
      dnnBwdInited_(false),
      engineCpu_(engine::cpu, 0),
      fwdPD_(NULL),
      bwdDataPD_(NULL),
      bwdWgtPD_(NULL),
      dataBot_(new DnnBuffer()),
      dataWgt_(new DnnBuffer()),
      dataBias_(new DnnBuffer()),
      dataTop_(new DnnBuffer()),
      diffBot_(NULL),
      diffWgt_(NULL),
      diffBias_(NULL),
      diffTop_(NULL)
    {}

  ~DnnConvLayer() {}
  
  /// for dnn
  void initDnnFwd();
  void initDnnBwd();
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
private:
  void printInfo() {
    for(size_t i = 0; i < iw_.size(); ++i) {
      LOG(INFO)
        << "ih: " << ih_[i] << ", iw: " << iw_[i]
        << ", ic: " << ic_[i] << ", gp: " << gp_[i]
        << ", oh: " << oh_[i] << ", ow: " << ow_[i]
        << ", fh: " << fh_[i] << ", fw: " << fw_[i]
        << ", ph: " << ph_[i] << ", pw: " << pw_[i]
        << ", sh: " << sh_[i] << ", sw: " << sw_[i]
        << ", oc: " << oc_;
    }
  }
};

}  // namespace paddle
