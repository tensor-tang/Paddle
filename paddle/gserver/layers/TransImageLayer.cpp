/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Logging.h"

namespace paddle {

/**
 * A layer for transposing image matrix.
 *
 * The config file api is img_trans_layer.
 */
class TransImageLayer : public Layer {
public:
  explicit TransImageLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

protected:
  size_t imgH_, imgW_;
  size_t batchSize_, layerSize_;
};


REGISTER_LAYER(trans_image, TransImageLayer);

bool TransImageLayer::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* the size of inputs for img_trans_layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);
  CHECK_EQ(1U, inputLayers_.size());

  return true;
}

void TransImageLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& input = getInput(0);
  batchSize_ = input.getBatchSize();
  layerSize_ = inputLayers_[0]->getSize();
//  LOG(INFO) << "in batchSize_"<<batchSize_;
  imgH_ = input.getFrameHeight();
  imgW_ = input.getFrameWidth();
  CHECK_EQ(imgH_ * imgW_, layerSize_)
    << "input height: " << imgH_ << ", width: " << imgW_;
  reserveOutput(batchSize_, layerSize_);

  real* in = getInputValue(0)->getData();
  real* out = getOutputValue()->getData();
  for (size_t i = 0; i < batchSize_; ++i) {
    real* pin = in + i * layerSize_;
    real* pout = out + i * layerSize_;
    MatrixPtr from = Matrix::create(pin, imgH_, imgW_, false, useGpu_);
    MatrixPtr to = Matrix::create(pout, imgW_, imgH_, false, useGpu_);
    from->transpose(to, false);
  }
  getOutput().setFrameHeight(imgW_);
  getOutput().setFrameWidth(imgH_);
}

void TransImageLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  MatrixPtr outputGrad = getOutputGrad();
  if (outputGrad == NULL) {
    return;
  }

  real* in = getInputGrad(0)->getData();
  real* out = outputGrad->getData();
  for (size_t i = 0; i < batchSize_; ++i) {
    real* pin = in + i * layerSize_;
    real* pout = out + i * layerSize_;
    MatrixPtr from = Matrix::create(pout, imgW_, imgH_, false, useGpu_);
    MatrixPtr to = Matrix::create(pin, imgH_, imgW_, false, useGpu_);
    from->transpose(to, false);
  }
}

}  // namespace paddle
