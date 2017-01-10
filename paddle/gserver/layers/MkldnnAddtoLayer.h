/* Copyright (c) 2016 */

#pragma once

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/ThreadLocal.h"
#include "mkldnn.hpp"
#include "MkldnnMemory.h"
#include <vector>

namespace paddle {

/** 
 * This layer just simply add all input layers together, then activate 
 * the sum inputs. Each input of this layer should be the same size, 
 * which is also the output size of this layer.
 * \f[
 *   y=f(\sum_{i}x_i + b)
 * \f]
 * where \f$y\f$ is output, \f$x\f$ is input, \f$b\f$ is bias, and \f$f\f$ is activation function.
 * 
 * The config file api is addto_layer.
 */
class MkldnnAddtoLayer : public MkldnnLayer {
protected:
  std::shared_ptr<mkldnn::sum::primitive_desc> fwdPD_;
  std::vector<MkldnnBufferPtr> dataBottoms_;
  std::vector<double> scales_;

  std::unique_ptr<Weight> biases_;

  bool has_spatial_;
  size_t layerSize_;  // layer size by batch == oh*ow*oc should also == ih*iw*ic 

public:
  explicit MkldnnAddtoLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwdPD_(NULL)
    {}

  ~MkldnnAddtoLayer() {}

  /** 
   * Intialization of MkldnnAddtoLayer. 
   */
  bool initDnn(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void clearAllCvtFlags() {
    for (size_t i = 0; i < dataBottoms_.size(); ++i) {
      if (dataBottoms_[i])
        dataBottoms_[i]->clearCvtFlag();
    }
    if (dataBot_) dataBot_->clearCvtFlag();
    if (dataTop_) dataTop_->clearCvtFlag();
    if (diffBot_) diffBot_->clearCvtFlag();
    if (diffTop_) diffTop_->clearCvtFlag();
  }

  size_t getOneBatchSize();

  bool reshapeOutput();

  void resetDnnFwd(PassType passType);
  void resetDnnBwd();

  /** 
   * Forward propagation.
   * @note There is no weight matrix for each input, 
   *       because it just a simple add operation.
   */
  void submitDnnFwd(PassType passType);

  /** 
   * Backward propagation. 
   */
  void submitDnnBwd(const UpdateCallback& callback);

private:
  void myFwd(PassType passType);
  void exFwd(PassType passType);
  void exBwd(const UpdateCallback &callback);

  void printInfo() {
    for (size_t i = 0; i < ic_.size(); ++i) {
      LOG(INFO)
        << "ic: " << ic_[i]
        << ", ih: " << ih_[i] << ", iw: " << iw_[i]
        << ", oc: " << oc_
        << ", oh: " << oh_[i] << ", ow: " << ow_[i];
    }
  }
};

}  // namespace paddle
