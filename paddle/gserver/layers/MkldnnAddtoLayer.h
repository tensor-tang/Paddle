/* Copyright (c) 2017 */

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
  std::shared_ptr<mkldnn::sum> fwd_;
  std::vector<double> scales_;

  std::unique_ptr<Weight> biases_;

  bool has_spatial_;
  size_t layerSize_;  // layer size by batch == oh*ow*oc should also == ih*iw*ic

public:
  explicit MkldnnAddtoLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwd_(nullptr)
    {}

  ~MkldnnAddtoLayer() {}

  bool initDnn(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void reshape();

  void clearDataDiff();

  void resetDnnFwd(PassType passType);

  void resetDnnBwd();

  void submitDnnFwd(PassType passType);

  void submitDnnBwd(const UpdateCallback& callback);

  void printInfo() {
    VLOG(1) << "sum number: " << botDatas_.size();
    for (size_t i = 0; i < ic_.size(); ++i) {
      VLOG(2)
        << "ic: " << ic_[i]
        << ", ih: " << ih_[i] << ", iw: " << iw_[i]
        << ", oc: " << oc_
        << ", oh: " << oh_[i] << ", ow: " << ow_[i];
    }
  }

private:
  int getMDDimSize(const mkldnn::memory::desc & md) {
    return md.data.ndims;
  }

  // return true if equal
  bool compareMD(const mkldnn::memory::desc & md1,
    const mkldnn::memory::desc & md2) {
    // skip mkldnn_primitive_kind_t and mkldnn_blocking_desc_t comparasion
    // TODO(TJ): MKLDNN sum seems support different format inputs
    // TODO(TJ): maybe no need check format, double check necessary
    if (MkldnnBuffer::getMDFmt(md1) != MkldnnBuffer::getMDFmt(md2))
      return false;
    int ndims = getMDDimSize(md1);
    if (ndims != getMDDimSize(md2)) return false;
    bool res = true;
    for (int i = 0; i < ndims; ++i) {
      res = res && (md1.data.dims[i] == md2.data.dims[i]);
    }
    return res && (md1.data.data_type == md2.data.data_type);
  }
};

}  // namespace paddle
