/* Copyright (c) 2016 */

#pragma once

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnMemory.h"

namespace paddle {

/**
 * @brief A subclass of LRN layer.
 *
 * The config file api is 
 */
class MkldnnLRNLayer : public MkldnnLayer {
protected:
  std::shared_ptr<mkldnn::lrn_forward> fwd_;
  std::shared_ptr<mkldnn::lrn_forward::primitive_desc> fwdPD_;
  std::shared_ptr<mkldnn::lrn_backward> bwd_;
  std::shared_ptr<mkldnn::memory> workspace_;

  mkldnn::algorithm algo_;
  int localSize_;
  double alpha_, beta_, k_;  // scale, pow, 
  // TODO: what k meaning???? block??

public:
  explicit MkldnnLRNLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwd_(nullptr),
      fwdPD_(nullptr),
      bwd_(nullptr),
      workspace_(nullptr)
    {}

  ~MkldnnLRNLayer() {}

  bool initDnn(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void reshape();

  void clearDataDiff() {};

  void resetDnnFwd(PassType passType);

  void resetDnnBwd();

  void submitDnnFwd(PassType passType);

  void submitDnnBwd(const UpdateCallback& callback);

  void printInfo() {
    for (size_t i = 0; i < iw_.size(); ++i) {
      VLOG(2)
        << "ih: " << ih_[i] << ", iw: " << iw_[i] << ", ic: " << ic_[i]
        << ", oh: " << oh_[i] << ", ow: " << ow_[i] << ", oc: " << oc_
        << ", localSize: " << localSize_
        << ", alpha/scale: " << alpha_
        << ", beta/pow: " << beta_;
    }
  }
};

}  // namespace paddle
