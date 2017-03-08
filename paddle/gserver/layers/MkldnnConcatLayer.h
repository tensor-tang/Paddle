/* Copyright (c) 2016 */

#pragma once

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnMemory.h"

namespace paddle {

/**
 * @brief A subclass of pool layer.
 *
 * The config file api is 
 */
class MkldnnConcatLayer : public MkldnnLayer {
protected:
  std::shared_ptr<mkldnn::concat::primitive> fwd_;
  // std::shared_ptr<convolution_backward_data::primitive_desc> bwdDataPD_;
  // std::shared_ptr<convolution_backward_weights::primitive_desc> bwdWgtPD_;
  std::vector<MkldnnBufferPtr> dataBottoms_;
  
  /*** concat_dimension in MKLDNN
   * if axis_ == 0, concat batchsize
   * if axis_ == 1, concat channel (default)
   */
  size_t axis_;


public:
  explicit MkldnnConcatLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwd_(nullptr),
      axis_(1)
//      bwdWgtPD_(nullptr)
    {}

  ~MkldnnConcatLayer() {}

  bool initDnn(const LayerMap& layerMap, const ParameterMap& parameterMap);

  virtual void clearAllDnnCvtFlags() {
    MkldnnLayer::clearAllDnnCvtFlags();
    for (size_t i = 0; i < dataBottoms_.size(); ++i) {
      if (dataBottoms_[i])
        dataBottoms_[i]->clearCvtFlag();
    }
  }

  void reshape();

  void clearDataDiff();

  void resetDnnFwd(PassType passType);

  void resetDnnBwd();

  void submitDnnFwd(PassType passType);

  void submitDnnBwd(const UpdateCallback& callback);

private:
  void myFwd(PassType passType);
  void exFwd(PassType passType);
  void exBwd(const UpdateCallback &callback);

  void printInfo() {
    VLOG(1) << "concats number: " << dataBottoms_.size();
    for (size_t i = 0; i < iw_.size(); ++i) {
      VLOG(2)
        << "ih: " << ih_[i] << ", iw: " << iw_[i]
        << ", ic: " << ic_[i]
        << ", oh: " << oh_[i] << ", ow: " << ow_[i]
        << ", oc: " << oc_;
    }
  }
};

}  // namespace paddle
