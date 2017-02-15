/* Copyright (c) 2016 */

#pragma once

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnMemory.h"

namespace paddle {

/**
 * @brief A subclass of MkldnnLayer fc layer.
 *
 * The config file api is 
 */
class MkldnnBatchNormLayer : public MkldnnLayer {
protected:
  std::shared_ptr<mkldnn::batch_normalization_forward> fwd_;
  std::shared_ptr<mkldnn::batch_normalization_backward> bwd_;

  // use paddle weight format
  bool usePaddleFmt_;

  /// data buffers
  MkldnnBufferPtr dataScaleShift_;  // wgt includes scale and shift in mkldnn
  MkldnnBufferPtr mean_;
  MkldnnBufferPtr var_;
  /// diff buffer 
  MkldnnBufferPtr diffScaleShift_;

  MatrixPtr selfScaleShiftData_;  // scale and shift value, 2*oc
  MatrixPtr selfScaleShiftDiff_;  // scale and shift diff, 2*oc
  MatrixPtr localMean_;  // m
  MatrixPtr localVar_;  // v^2

  bool useScaleShift_;

  /// Epsilon value used in the batch normalization formula.
  static const real EPS;

  /// Feature dimension. If the input layer is conv layer, it is the channels
    /// of feature map of the conv layer. If the input layer is fully-connected
    /// layer, it is the dimension of fc layer.

  /// here weight_ in paddle is scale in mkldnn
  std::unique_ptr<Weight> weight_;
  /// here bias in paddle is shift in mkldnn
  std::unique_ptr<Weight> biases_;
  /// Moving average of mean.
  std::unique_ptr<Weight> movingMean_;
  /// Moving average of variance.
  std::unique_ptr<Weight> movingVar_;

  /// Save intermediate results computed during the forward pass,
  /// these can then be reused to speed up the backward pass.
  MatrixPtr savedInvVar_;

  /// Height or width of input image feature, now height is equal to width.
  /// imgSize is 1 if the input is fully-connected layer.

  /// Height * Width.
  int imgPixels_;

  // if useGlobalStats_ is true, will use the loaded mean and variance.
  // otherwise, calculate mean and variance in this mini-batch.
  bool useGlobalStats_;
  // use to compute moving mean and variance.
  real movingAvgFraction_;

  /// Load pre-calculated mean and std.
  void setMeanAndStd();

  /// Calculate mean and std.
  void calMeanAndStd(const MatrixPtr& mat);

  /// Calculate moving mean and variance.
  void calMovingMeanAndVar();

  /// expand a Matrix from batch, channels* imagePixels to
  /// batch * ImagePixels * channels.
  void expandMat(const MatrixPtr& in, MatrixPtr& out);

  /// Shrink a Matrix from  from batch * ImagePixels * channels
  /// to batch, channels* imagePixels.
  void shrinkMat(const MatrixPtr& in, MatrixPtr& out);

  /// Load mean and variance only once flag.
  MatrixPtr tmpMat_, tmpGrad_;
  MatrixPtr expandedIn_, expandedOut_;
  MatrixPtr expandedInGrad_, expandedOutGrad_, inGrad_;
  MatrixPtr normIn_, normInGrad_, meanGrad_, stdGrad_;

public:
  explicit MkldnnBatchNormLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwd_(nullptr),
      bwd_(nullptr),
      usePaddleFmt_(true),
      dataScaleShift_(nullptr),
      mean_(nullptr),
      var_(nullptr),
      diffScaleShift_(nullptr),
//      diffBias_(nullptr),
//      bwdWgtPD_(nullptr)
      useScaleShift_(true)
    {}

  ~MkldnnBatchNormLayer() {}

  bool initDnn(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void clearAllDnnCvtFlags() {
    if (dataBot_) dataBot_->clearCvtFlag();
    if (dataTop_) dataTop_->clearCvtFlag();
    if (diffBot_) diffBot_->clearCvtFlag();
    if (diffTop_) diffTop_->clearCvtFlag();
    if (dataScaleShift_) dataScaleShift_->clearCvtFlag();
    if (diffScaleShift_) diffScaleShift_->clearCvtFlag();
    if (mean_) mean_->clearCvtFlag();
    if (var_) var_->clearCvtFlag();
  //  if (diffBias_) diffBias_->clearCvtFlag();
  }

  void reshape();

  void clearDataDiff();

  void resetDnn(PassType passType);

  void submitDnnFwd(PassType passType);
  void submitDnnBwd(const UpdateCallback& callback);

  /**
   * @brief Create BatchNorm layer by norm_type, including batch_norm and
   * cudnn_batch_norm. If do not set norm_type, it will automatically select
   * cudnn_batch_norm for GPU and batch_norm for CPU.
   */
  // keep for paddle
  static Layer* create(const LayerConfig& config);

private:
  void myFwd(PassType passType);
  void exFwd(PassType passType);
  void exBwd(const UpdateCallback &callback);

  void printInfo() {
    for (size_t i = 0; i < iw_.size(); ++i) {
      LOG(INFO)
        << "ic: " << ic_[i]
        << ", ih: " << ih_[i] << ", iw: " << iw_[i]
        << ", oc: " << oc_
        << ", oh: " << oh_[i] << ", ow: " << ow_[i];
    }
  }
};

}  // namespace paddle
