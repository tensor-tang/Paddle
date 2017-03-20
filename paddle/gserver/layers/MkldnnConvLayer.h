/* Copyright (c) 2017 */

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

  /// weight data and diff buffers
  MkldnnBufferPtr wgtData_;
  MkldnnBufferPtr wgtDataBwd_;  // in backward the wgt data's format may differ
  MkldnnBufferPtr wgtDiff_;

  /// bias data and diff buffers
  MkldnnBufferPtr biasData_;
  MkldnnBufferPtr biasDiff_;

  // padding, stride and filter size
  std::vector<int> ph_, pw_;
  std::vector<int> sh_, sw_;
  std::vector<int> fh_, fw_;
  // group
  std::vector<int> gp_;

  // dnn self wgt, only create if use paddle fmt
  std::vector<MatrixPtr> selfWgtData_;
  std::vector<MatrixPtr> selfWgtDiff_;

  /// shape of weight: (oc, ic*fh*fw/gp)
  WeightList weights_;
  /// If shared_biases is false shape of bias: (oc * outputX * outputY, 1)
  /// If shared_biases is ture shape of bias: (oc, 1)
  std::unique_ptr<Weight> biases_;
  bool hasRelu_;
  bool useConvRelu_;
  bool hasInited_;
  double negativeSlope_;

public:
  explicit MkldnnConvLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwd_(nullptr),
      bwdWgt_(nullptr),
      bwdData_(nullptr),
      wgtData_(nullptr),
      wgtDataBwd_(nullptr),
      wgtDiff_(nullptr),
      biasData_(nullptr),
      biasDiff_(nullptr),
      hasRelu_(false),
      useConvRelu_(false),
      hasInited_(false),
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

  virtual void clearAllDnnCvtFlags() {
    MkldnnLayer::clearAllDnnCvtFlags();
    if (biasData_) biasData_->clearCvtFlag();
    if (wgtData_) wgtData_->clearCvtFlag();
    if (wgtDataBwd_) wgtDataBwd_->clearCvtFlag();
    if (biasDiff_) biasDiff_->clearCvtFlag();
    if (wgtDiff_) wgtDiff_->clearCvtFlag();
  }

  void reshape();

  void clearDataDiff();

  // return false if donot need reshape
  bool reshapeOutput();

  void resetDnnFwd(PassType passType);

  void resetDnnBwd();

  void submitDnnFwd(PassType passType);

  void submitBwdData(int idx);

  void submitBwdWgts(int idx);

  void submitDnnBwd(const UpdateCallback& callback);

  void printInfo() {
    for (size_t i = 0; i < iw_.size(); ++i) {
      VLOG(2)
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
