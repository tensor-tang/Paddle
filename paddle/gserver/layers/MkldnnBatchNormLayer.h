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
  std::shared_ptr<mkldnn::batch_normalization_forward::primitive_desc> fwdPD_;

  /// data buffers
  MkldnnBufferPtr dataScaleShift_;  // wgt includes scale and shift in mkldnn
  MkldnnBufferPtr mean_;
  MkldnnBufferPtr var_;
  /// diff buffer
  MkldnnBufferPtr diffScaleShift_;

  MatrixPtr selfScaleShiftData_;  // scale and shift value, 2*oc
  MatrixPtr selfScaleShiftDiff_;  // scale and shift diff, 2*oc
  MatrixPtr localMean_;  // output of mkldnn: m
  MatrixPtr localVar_;  // output of mkldnn: v^2

  bool useScaleShift_;
  unsigned flags_;

  /// Epsilon value used in the batch normalization formula.
  static const real EPS;

  /// here weight_ in paddle is scale in mkldnn
  std::unique_ptr<Weight> weight_;
  /// here bias in paddle is shift in mkldnn
  std::unique_ptr<Weight> biases_;
  /// Moving average of mean.
  std::unique_ptr<Weight> movingMean_;
  /// Moving average of variance.
  std::unique_ptr<Weight> movingVar_;

  // if useGlobalStats_ is true, will use the loaded mean and variance.
  // otherwise, calculate mean and variance in this mini-batch.
  bool useGlobalStats_;
  // use to compute moving mean and variance.
  real movingAvgFraction_;

  bool hasInited_;

  // whether use mkldnn seq batchnorm
  bool useMkldnnSeq_;

public:
  explicit MkldnnBatchNormLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwd_(nullptr),
      bwd_(nullptr),
      fwdPD_(nullptr),
      dataScaleShift_(nullptr),
      mean_(nullptr),
      var_(nullptr),
      diffScaleShift_(nullptr),
      useScaleShift_(true),
      useGlobalStats_(false),
      hasInited_(false),
      useMkldnnSeq_(false)
    {}

  ~MkldnnBatchNormLayer() {}

  bool initDnn(const LayerMap& layerMap, const ParameterMap& parameterMap);

  virtual void clearAllDnnCvtFlags() {
    MkldnnLayer::clearAllDnnCvtFlags();
    if (dataScaleShift_) dataScaleShift_->clearCvtFlag();
    if (diffScaleShift_) diffScaleShift_->clearCvtFlag();
    if (mean_) mean_->clearCvtFlag();
    if (var_) var_->clearCvtFlag();
  //  if (diffBias_) diffBias_->clearCvtFlag();
  }

  /// Calculate moving mean and variance.
  void calMovingMeanAndVar();


  // load the settings from proto
  virtual void loadConfig();


  // reshape 
  // output matrix height and width 
  // and the bs
  // and the output buffer
  virtual void reshapeOutput();

  void resetDnnFwd();

  void resetDnnBwd();

  void submitDnnFwd();
  void submitDnnBwd(const UpdateCallback& callback);

  /**
   * @brief Create BatchNorm layer by norm_type, including batch_norm and
   * cudnn_batch_norm. If do not set norm_type, it will automatically select
   * cudnn_batch_norm for GPU and batch_norm for CPU.
   */
  /// keep for paddle
  static Layer* create(const LayerConfig& config);

};

}  // namespace paddle
