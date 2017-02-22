/* Copyright (c) 2016 */

#pragma once

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnMemory.h"

namespace paddle {

/**
 * @brief A subclass of convolution layer.
 *
 * The config file api is img_conv_layer.
 */
class MkldnnConvLayer : public MkldnnLayer {
protected:
  std::shared_ptr<mkldnn::convolution_forward::primitive> fwd_;
  std::shared_ptr<mkldnn::convolution_backward_weights> bwdWgt_;
  std::shared_ptr<mkldnn::convolution_backward_data> bwdData_;
  
  /// data buffers
  MkldnnBufferPtr dataWgt_;
  MkldnnBufferPtr dataBias_;
  /// diff buffer
  MkldnnBufferPtr diffWgt_;
  MkldnnBufferPtr diffBias_;

  // padding, stride and filter size
  std::vector<int> ph_, pw_;
  std::vector<int> sh_, sw_;
  std::vector<int> fh_, fw_;
  // group
  std::vector<int> gp_;

  // dnn self wgt, only create if use paddle fmt
  std::vector<MatrixPtr> selfWgtData_;
  std::vector<MatrixPtr> selfWgtDiff_;

  // use paddle weight format
  bool usePaddleFmt_;

  /// shape of weight: (oc, ic*fh*fw/gp)
  WeightList weights_;
  /// If shared_biases is false shape of bias: (oc * outputX * outputY, 1)
  /// If shared_biases is ture shape of bias: (oc, 1)
  std::unique_ptr<Weight> biases_;
  bool hasRelu_;
  bool hasCvtTopData_;
  bool hasCvtTopDiff_;
  bool hasCvtBiasData_;
  bool hasCvtBiasDiff_;
  bool useConvRelu_;
  double negativeSlope_;

public:
  explicit MkldnnConvLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwd_(nullptr),
      bwdWgt_(nullptr),
      bwdData_(nullptr),
      dataWgt_(nullptr),
      dataBias_(nullptr),
      diffWgt_(nullptr),
      diffBias_(nullptr),
      hasRelu_(false),
      hasCvtTopData_(false),
      hasCvtTopDiff_(false),
      hasCvtBiasData_(false),
      hasCvtBiasDiff_(false),
      useConvRelu_(false),
      negativeSlope_(-0.0)
    {}

  ~MkldnnConvLayer() {}

  bool initDnn(const LayerMap& layerMap, const ParameterMap& parameterMap);

  bool hasMkldnnRelu() {
    if (!hasActivation()) {
      return false;
    }
    const std::string dnn("mkldnn_relu");
    const std::string& type = activation_->getName();
    return type.compare(0, dnn.length(), dnn) == 0 ? true : false;
  }

  void clearAllDnnCvtFlags() {
    if (dataBot_) dataBot_->clearCvtFlag();
    if (dataTop_) dataTop_->clearCvtFlag();
    if (dataBias_) dataBias_->clearCvtFlag();
    if (dataWgt_) dataWgt_->clearCvtFlag();
    if (diffBot_) diffBot_->clearCvtFlag();
    if (diffTop_) diffTop_->clearCvtFlag();
    if (diffBias_) diffBias_->clearCvtFlag();
    if (diffWgt_) diffWgt_->clearCvtFlag();
  }

  void reshape();

  void clearDataDiff();

  // return false if donot need reshape
  bool reshapeOutput();

  void resetDnn(PassType passType);

  void submitDnnFwd(PassType passType);

  void submitBwdData(int idx);

  void submitBwdWgts(int idx);

  void submitDnnBwd(const UpdateCallback& callback);

private:
  void exBackward(const UpdateCallback &callback);
  void exBwdBias(MatrixPtr topDiff);
  void exBwdData(MatrixPtr topDiff, int i);
  void exBwdWgts(MatrixPtr topDiff, int i);

  void printInfo() {
    for (size_t i = 0; i < iw_.size(); ++i) {
      LOG(INFO)
        << "gp: " << gp_[i]
        << ", ic: " << ic_[i] << ", ih: " << ih_[i] << ", iw: " << iw_[i]
        << ", oc: " << oc_    << ", oh: " << oh_[i] << ", ow: " << ow_[i]
        << ", fh: " << fh_[i] << ", fw: " << fw_[i]
        << ", ph: " << ph_[i] << ", pw: " << pw_[i]
        << ", sh: " << sh_[i] << ", sw: " << sw_[i];
    }
  }
};

}  // namespace paddle
