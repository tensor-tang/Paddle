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
class DnnConvLayer : public Layer {

protected:
  typedef std::vector<int> IntV;
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

  bool needBwdReset_;
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
  // batchsize
  int bs_;

  /// shape of weight: (oc, ic*fh*fw/gp)
  WeightList weights_;
  /// If shared_biases is false shape of bias: (oc, 1)
  /// If shared_biases is ture shape of bias:
  /// (oc * outputX * outputY, 1)
  std::unique_ptr<Weight> biases_;

public:
  explicit DnnConvLayer(const LayerConfig& config)
    : Layer(config),
      engineCpu_(engine::cpu, 0),
      fwdPD_(NULL),
      bwdDataPD_(NULL),
      bwdWgtPD_(NULL),
      dataBot_(NULL),
      dataWgt_(NULL),
      dataBias_(NULL),
      dataTop_(NULL),
      diffBot_(NULL),
      diffWgt_(NULL),
      diffBias_(NULL),
      diffTop_(NULL),
      needBwdReset_(true)
    {}

  ~DnnConvLayer() {}

  /// for dnn
  void initOrResetDnnFwd();
  void initOrResetDnnBwd();
  int dimSize(const memory::dims &t) {
    int sz = 1;
    for (size_t i = 0; i < t.size(); ++i) 
      sz *= t[i];
    return sz;
  }

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  size_t getOneBatchSize();
  int outputSize(int imageSize, int filterSize, int padding, int stride) {
    int outputSize;
    bool caffeMode = true;
    if (!caffeMode) {
      outputSize =
          (imageSize - filterSize + 2 * padding + stride - 1) / stride + 1;
    } else {
      outputSize = (imageSize - filterSize + 2 * padding) / stride + 1;
    }
    CHECK_GE(outputSize, 1);
    return outputSize;
  }
  void clearAllCvtFlags() {
    if (dataBot_) dataBot_->clearCvtFlag();
    if (dataTop_) dataTop_->clearCvtFlag();
    if (dataBias_) dataBias_->clearCvtFlag();
    if (dataWgt_) dataWgt_->clearCvtFlag();
    if (diffBot_) diffBot_->clearCvtFlag();
    if (diffTop_) diffTop_->clearCvtFlag();
    if (diffBias_) diffBias_->clearCvtFlag();
    if (diffWgt_) diffWgt_->clearCvtFlag();
  }
  
  /* init the memory format of input and output data or diff
   */
  void initMemoryFormat();
  
  /* forward data
   * input: botdata, wgtdata, biasdata
   * output topdata
   */
  void submitFwd(int inputIdx, const MatrixPtr& botVal, const MatrixPtr& topVal);
  
  /* backward data
   * input: topdiff, wgtdata
   * output botdiff
   */
  void submitBwdData(int inputIdx, const MatrixPtr& topGrad, const MatrixPtr& botGrad);

  /* backward wgt and bias
   * input: topdiff, botdata
   * output wgtdiff, biasdiff
   */
  void submitBwdWgts(int inputIdx, const MatrixPtr& botVal, const MatrixPtr& topGrad);
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
