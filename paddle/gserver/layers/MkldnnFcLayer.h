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
class MkldnnFcLayer : public MkldnnLayer {
protected:
  /// For dnn fc. Primitive Desc
  std::shared_ptr<inner_product_forward::primitive_desc> fwdPD_;
  //std::shared_ptr<convolution_backward_data::primitive_desc> bwdDataPD_;
  //std::shared_ptr<convolution_backward_weights::primitive_desc> bwdWgtPD_;

  // if image width and height !=0 
  bool has_spatial_;
  /// data buffers
  MkldnnBufferPtr dataWgt_;
  MkldnnBufferPtr dataBias_;
  /// diff buffer
//  MkldnnBufferPtr diffWgt_;
//  MkldnnBufferPtr diffBias_;
  bool hasBias_;

  // fc
  WeightList weights_;
  std::unique_ptr<Weight> biases_;
public:
  explicit MkldnnFcLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwdPD_(NULL),
      has_spatial_(false),
      dataWgt_(NULL),
      dataBias_(NULL),/*
      diffWgt_(NULL),
      diffBias_(NULL),
      bwdWgtPD_(NULL)*/
      hasBias_(false)
    {}

  ~MkldnnFcLayer() {}
  

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

  void resetDnnFwd(PassType passType);
  
  void resetDnnBwd();

  void submitDnnFwd(PassType passType);
  void submitDnnBwd(const UpdateCallback& callback);
  // keep for paddle
  void prefetch();

private:
  Weight& getWeight(int idx) { return *weights_[idx]; }
  
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
