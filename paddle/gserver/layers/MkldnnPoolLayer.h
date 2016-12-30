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
class MkldnnPoolLayer : public MkldnnLayer {
protected:
  /// For dnn pooling. Primitive Desc
  std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwdPD_;
  // std::shared_ptr<convolution_backward_data::primitive_desc> bwdDataPD_;
  // std::shared_ptr<convolution_backward_weights::primitive_desc> bwdWgtPD_;

  std::shared_ptr<mkldnn::memory> workspace_;
  bool withWorkspace_;
  // padding, stride and filter size
  int ph_, pw_;
  int sh_, sw_;
  int fh_, fw_;

  mkldnn::algorithm poolAlgo_;

public:
  explicit MkldnnPoolLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwdPD_(NULL),
      workspace_(NULL)
//      bwdWgtPD_(NULL)
    {}

  ~MkldnnPoolLayer() {}

  bool initDnn(const LayerMap& layerMap, const ParameterMap& parameterMap);

  size_t getOneBatchSize();

  void clearAllCvtFlags() {
    if (dataBot_) dataBot_->clearCvtFlag();
    if (dataTop_) dataTop_->clearCvtFlag();
    if (diffBot_) diffBot_->clearCvtFlag();
    if (diffTop_) diffTop_->clearCvtFlag();
  }

  // return false if donot need reshape
  bool reshapeOutput();

  void resetDnnFwd(PassType passType);

  void resetDnnBwd();

  void submitDnnFwd(PassType passType);

  void submitDnnBwd(const UpdateCallback& callback);

private:
  void myFwd(PassType passType);
  void exFwd(PassType passType);
  void exBwd(const UpdateCallback &callback);

  void printInfo() {
    for (size_t i = 0; i < iw_.size(); ++i) {
      LOG(INFO)
        << "ih: " << ih_[i] << ", iw: " << iw_[i]
        << ", ic: " << ic_[i]
        << ", oh: " << oh_[i] << ", ow: " << ow_[i]
        << ", fh: " << fh_ << ", fw: " << fw_
        << ", ph: " << ph_ << ", pw: " << pw_
        << ", sh: " << sh_ << ", sw: " << sw_
        << ", oc: " << oc_;
    }
  }
};

}  // namespace paddle
