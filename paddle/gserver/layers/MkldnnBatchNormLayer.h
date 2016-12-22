/* Copyright (c) 2016 */

#pragma once

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnMemory.h"

using namespace mkldnn;

namespace paddle {

/**
 * @brief A subclass of MkldnnLayer fc layer.
 *
 * The config file api is 
 */
class MkldnnBatchNormLayer : public MkldnnLayer {
protected:
  /// For dnn fc. Primitive Desc
  std::shared_ptr<batch_normalization_forward::primitive_desc> fwdPD_;
  //std::shared_ptr<convolution_backward_data::primitive_desc> bwdDataPD_;
  //std::shared_ptr<convolution_backward_weights::primitive_desc> bwdWgtPD_;

  // use paddle weight format 
  bool usePaddleFmt_;

  /// data buffers
  MkldnnBufferPtr dataWgt_;
  MkldnnBufferPtr dataBias_;
  /// diff buffer
//  MkldnnBufferPtr diffWgt_;
//  MkldnnBufferPtr diffBias_;
  bool hasBias_;
  /// Epsilon value used in the batch normalization formula.
  static const real EPS;
protected:
  /// Feature dimension. If the input layer is conv layer, it is the channels
    /// of feature map of the conv layer. If the input layer is fully-connected
    /// layer, it is the dimension of fc layer.

  // ex bn
  /// Batch normalization scale parameter, which is referred to as gamma in
  /// in original paper.
  std::unique_ptr<Weight> weight_;
  /// Moving average of mean.
  std::unique_ptr<Weight> movingMean_;
  /// Moving average of variance.
  std::unique_ptr<Weight> movingVar_;
  /// Batch normalization bias parameter, which is referred to as beta in
  /// in original paper.
  std::unique_ptr<Weight> biases_;

  /// Save intermediate results computed during the forward pass,
  /// these can then be reused to speed up the backward pass.
  MatrixPtr savedMean_;
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
  bool firstTest_;
  MatrixPtr tmpMat_, tmpGrad_;
  MatrixPtr expandedIn_, expandedOut_;
  MatrixPtr expandedInGrad_, expandedOutGrad_, inGrad_;
  MatrixPtr normIn_, normInGrad_, meanGrad_, stdGrad_;

  
public:
  explicit MkldnnBatchNormLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwdPD_(NULL),
      usePaddleFmt_(true),
      dataWgt_(NULL),
      dataBias_(NULL),/*
      diffWgt_(NULL),
      diffBias_(NULL),
      bwdWgtPD_(NULL)*/
      hasBias_(false)
    {}

  ~MkldnnBatchNormLayer() {}
  
  bool initDnn(const LayerMap& layerMap, const ParameterMap& parameterMap);

  size_t getOneBatchSize();

  void clearAllCvtFlags() {
    if (dataBot_) dataBot_->clearCvtFlag();
    if (dataTop_) dataTop_->clearCvtFlag();
    if (diffBot_) diffBot_->clearCvtFlag();
    if (diffTop_) diffTop_->clearCvtFlag();
    if (dataBias_) dataBias_->clearCvtFlag();
    if (dataWgt_) dataWgt_->clearCvtFlag();
  //  if (diffBias_) diffBias_->clearCvtFlag();
  //  if (diffWgt_) diffWgt_->clearCvtFlag();
  }

  // return false if donot need reshape 
  bool reshapeOutput();

  void resetDnnFwd();
  
  void resetDnnBwd();

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
  void calFeatureMapSize();
  void myFwd(PassType passType);
  void exFwd(PassType passType);
  void exBwd(const UpdateCallback &callback);
  
  void printInfo() {
    for(size_t i = 0; i < iw_.size(); ++i) {
      LOG(INFO)
        << "ih: " << ih_[i] << ", iw: " << iw_[i]
        << ", ic: " << ic_[i] 
        << ", oh: " << oh_[i] << ", ow: " << ow_[i]
        << ", oc: " << oc_;
    }
  }
};

}  // namespace paddle
