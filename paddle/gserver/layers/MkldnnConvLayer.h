/* Copyright (c) 2017 */

#pragma once

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnMemory.h"

namespace paddle {

/**
 * @brief A subclass of Mkldnn convolution layer.
 *
 * The config file api is mkldnn_conv.
 */
class MkldnnConvLayer : public MkldnnLayer {
protected:
  std::shared_ptr<mkldnn::convolution_forward::primitive> fwd_;
  std::shared_ptr<mkldnn::convolution_backward_weights> bwdWgt_;
  std::shared_ptr<mkldnn::convolution_backward_data> bwdData_;

  // top diff for backward weight
  // it may have different format with topdiff backward data
  MkldnnBufferPtr topDiffBwdWgt_;

  /// weight data and diff buffers
  MkldnnBufferPtr wgtData_;
  MkldnnBufferPtr wgtDataBwd_;  // in backward the wgt data's format may differ
  MkldnnBufferPtr wgtDiff_;

  /// bias data and diff buffers
  MkldnnBufferPtr biasData_;
  MkldnnBufferPtr biasDiff_;

  // padding size
  int ph_, pw_;
  // stride size
  int sh_, sw_;
  // filter size
  int fh_, fw_;
  // group
  int gp_;

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

  // load the settings from proto
  virtual void loadConfig();

  bool initDnnWgt(const LayerMap& layerMap, const ParameterMap& parameterMap);

  // reshape 
  // output matrix height and width 
  // and the bs
  void reshapeOutputInfo();

  void clearAllDnnCvtFlags() {
    MkldnnLayer::clearAllDnnCvtFlags();
    if (topDiffBwdWgt_) topDiffBwdWgt_->clearCvtFlag();
    if (biasData_) biasData_->clearCvtFlag();
    if (wgtData_) wgtData_->clearCvtFlag();
    if (wgtDataBwd_) wgtDataBwd_->clearCvtFlag();
    if (biasDiff_) biasDiff_->clearCvtFlag();
    if (wgtDiff_) wgtDiff_->clearCvtFlag();
  }

  void resetDnnFwd(PassType passType);

  void resetDnnBwd();

  void submitDnnFwd();

  void submitDnnBwd(const UpdateCallback& callback);

protected:
  bool hasMkldnnRelu() {
    if (!hasActivation()) {
      return false;
    }
    const std::string dnn("mkldnn_relu");
    const std::string& type = activation_->getName();
    return type.compare(0, dnn.length(), dnn) == 0 ? true : false;
  }

  void submitBwdData(int idx);

  void submitBwdWgts(int idx);

  void printInfo() {
    VLOG(2) << "bs: " << bs_
      << "gp: " << gp_
      << ", ic: " << ic_ << ", ih: " << ih_ << ", iw: " << iw_
      << ", oc: " << oc_  << ", oh: " << oh_ << ", ow: " << ow_
      << ", fh: " << fh_ << ", fw: " << fw_
      << ", ph: " << ph_ << ", pw: " << pw_
      << ", sh: " << sh_ << ", sw: " << sw_;
  }
};

}  // namespace paddle
