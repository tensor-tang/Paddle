/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "ActivationFunction.h"

#include <algorithm>
#include <memory>
#include <iostream>
#include <type_traits>
#include <string>
#include <thread>
#include "paddle/utils/ClassRegistrar.h"
#include "paddle/parameter/Argument.h"

#include "paddle/utils/Logging.h"

#ifdef PADDLE_USE_MKLDNN
#include "MkldnnActivation.h"
#endif

namespace paddle {  // NOLINT

static ClassRegistrar<ActivationFunction> gActivationRegistrar;
/**
 * @def ACTIVATION_CLASS_NAME
 * @brief Macro for getting derived activation class name
 * @note ACTIVATION_CLASS_NAME(softmax) softmax_;
 * means softmaxActivation softmax_;
 */
#define ACTIVATION_CLASS_NAME(ACTIVATION_NAME) ACTIVATION_NAME##Activation
/**
 * @def BEGIN_DEFINE_ACTIVATION
 * @brief Macro for defining a devried activation class
 */
#define BEGIN_DEFINE_ACTIVATION(ACTIVATION_NAME)                             \
  class ACTIVATION_CLASS_NAME(ACTIVATION_NAME) : public ActivationFunction { \
  private:                                                                   \
    static const std::string name;                                           \
                                                                             \
  public:                                                                    \
    const std::string& getName() const { return name; }
/**
 * @def END_DEFINE_ACTIVATION
 * @brief Macro for registering a derived activation class
 */
#define END_DEFINE_ACTIVATION(ACTIVATION_NAME)                     \
  };                                                               \
  const std::string ACTIVATION_CLASS_NAME(ACTIVATION_NAME)::name = \
      #ACTIVATION_NAME;                                            \
  static InitFunction __reg_activation__##ACTIVATION_NAME([] {     \
    gActivationRegistrar.registerClass<                            \
        ACTIVATION_CLASS_NAME(ACTIVATION_NAME)>(#ACTIVATION_NAME); \
  });

/**
 * @brief The IdentityActivation class
 *
 * Do nothing when forward/backward.
 */
class IdentityActivation : public ActivationFunction {
public:
  static const std::string name;
  void forward(Argument& act) { (void)act; }
  void backward(Argument& act) { (void)act; }
  const std::string& getName() const { return name; }
};
const std::string IdentityActivation::name = "";
static InitFunction __reg_activation__identity([] {
  gActivationRegistrar.registerClass<IdentityActivation>("");
  gActivationRegistrar.registerClass<IdentityActivation>("linear");
});

/**
 * @brief Sigmoid Activation
 * \f[
 * f(z) = \frac{1}{1+exp(-z)}
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(sigmoid)
void forward(Argument& act) { act.value->sigmoid(*act.value); }
void backward(Argument& act) { act.grad->sigmoidDerivative(*act.value); }
END_DEFINE_ACTIVATION(sigmoid)

/**
 * @brief Softmax Activation
 * \f[
 * P(y=j|x) = \frac{e^{x^Tw_j}}{\sum^K_{k=1}e^{x^Tw_k}}
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(softmax)
private:
MatrixPtr sftMaxSum_;
MatrixPtr sftMaxDot_;
MatrixPtr one_;

public:
void forward(Argument& act) { act.value->softmax(*act.value); }

void backward(Argument& act) {
  MatrixPtr outputV = act.value;
  MatrixPtr outputG = act.grad;

  if (outputG->useGpu()) {
    outputG->softmaxBackward(*outputV);
  } else {
    SetDevice device(act.deviceId);
    Matrix::resizeOrCreate(sftMaxDot_, outputG->getHeight(),
                           outputG->getWidth(),
                           /* trans */ false, useGpu(act.deviceId));
    Matrix::resizeOrCreate(sftMaxSum_, outputG->getHeight(), 1,
                           /* trans */ false, useGpu(act.deviceId));
    if (!one_ || one_->getWidth() != outputG->getWidth()) {
      Matrix::resizeOrCreate(one_, 1, outputG->getWidth(),
                             /* trans */ false, useGpu(act.deviceId));
      one_->one();
    }

    sftMaxDot_->dotMul(*outputG, *outputV);
    sftMaxSum_->colMerge(*sftMaxDot_);

    act.grad->softmaxDerivative(*act.value, *sftMaxSum_);
  }
}
END_DEFINE_ACTIVATION(softmax)


/**
 * @brief Sequence_softmax Activation
 * @note Softmax on all frames of one sequence.
 * Width of frame must be one.
 */
BEGIN_DEFINE_ACTIVATION(sequence_softmax)
private:
ACTIVATION_CLASS_NAME(softmax) softmax_;
Argument argument_;

public:
void forward(Argument& act) {
  CHECK_EQ(act.value->getWidth(), 1UL);

  if (!argument_.value) {
    argument_.value = Matrix::create(nullptr, /* height= */ 1, 1,
                                     /* trans= */ false, useGpu(act.deviceId));
    argument_.grad = Matrix::create(nullptr, /* height= */ 1, 1,
                                    /* trans= */ false, useGpu(act.deviceId));
  }

  auto starts = act.sequenceStartPositions->getVector(useGpu(act.deviceId));
  act.value->sequenceSoftmax(*act.value, *starts);
}

void backward(Argument& act) {
  CHECK_EQ(act.grad->getWidth(), 1UL);

  size_t numSequences = act.getNumSequences();
  const int* starts = act.sequenceStartPositions->getData(false);

  for (size_t i = 0; i < numSequences; ++i) {
    // TODO(Dangqingqing) optimization for GPU
    size_t offset = starts[i];
    size_t size = starts[i + 1] - starts[i];
    argument_.value->setData(act.value->getData() + offset, 1UL, size);
    argument_.grad->setData(act.grad->getData() + offset, 1UL, size);

    softmax_.backward(argument_);
  }
}
END_DEFINE_ACTIVATION(sequence_softmax)

/**
 * @brief Relu Activation.
 * forward. y = max(0, z)
 *
 * derivative of relu is:
 *
 *    1 if z > 0
 *
 *    0 otherwise.
 */
BEGIN_DEFINE_ACTIVATION(relu)
void forward(Argument& act) { act.value->relu(*act.value); }

void backward(Argument& act) { act.grad->reluDerivative(*act.value); }
END_DEFINE_ACTIVATION(relu)

#ifdef PADDLE_USE_MKLDNN

/**
 * @brief MKLDNN Relu Activation.
 * forward  
 *  f(x) = negative_slope * x  (x <  0)
 *  f(x) = x                   (x >= 0) 
 */
class ACTIVATION_CLASS_NAME(mkldnn_relu)
                        : public ActivationFunction, public MkldnnActivation {
private:
  static const std::string name;
  std::shared_ptr<mkldnn::relu_forward> fwd_;
  std::shared_ptr<mkldnn::relu_forward::primitive_desc> fwdPD_;
  std::shared_ptr<mkldnn::relu_backward> bwd_;

public:
  const std::string& getName() const { return name; }

  float negative_slope;

  void resetDnnFwd(const Argument& arg, std::shared_ptr<void> topDataMD) {
    if (bs_ == arg.getBatchSize()) {
      return;
    }
    MkldnnActivation::reshapeDnnFwd(arg, topDataMD);
    LOG(INFO) << this->getName() << " reshape batchsize: "
          << bs_ << ", " << oc_ << ", " << oh_ << ", " << ow_;

    mkldnn::engine eg = CpuEngine::Instance().getEngine();

    real* pdata = arg.value->getData();
    dataBot_.reset(new mkldnn::memory({*md_, eg}, pdata));
    dataTop_.reset(new mkldnn::memory({*md_, eg}, pdata));

    // careful: should be -0, not 0
    negative_slope = -0.f;

    auto fwdMD = mkldnn::relu_forward::desc(
      mkldnn::prop_kind::forward_training, *md_, negative_slope);
    fwdPD_.reset(new mkldnn::relu_forward::primitive_desc(fwdMD, eg));
    fwd_.reset(new mkldnn::relu_forward(*fwdPD_, *dataBot_, *dataTop_));

    needResetBwd_ = true;
  }

  void resetDnnBwd(const Argument& arg, std::shared_ptr<void> topDiffMD) {
    if (!needResetBwd_)
      return;

    mkldnn::engine eg = CpuEngine::Instance().getEngine();
    real* pdiff = arg.grad->getData();
    diffBot_.reset(new mkldnn::memory({*md_, eg}, pdiff));
    diffTop_.reset(new mkldnn::memory({*md_, eg}, pdiff));

    auto bwdMD = mkldnn::relu_backward::desc(*md_, *md_, negative_slope);
    auto bwdPD = mkldnn::relu_backward::primitive_desc(bwdMD, eg, *fwdPD_);
    bwd_.reset(new mkldnn::relu_backward(
      bwdPD, *dataBot_, *diffTop_, *diffBot_));
    needResetBwd_ = false;
  }

  void forward(Argument& act) {
    std::vector<mkldnn::primitive> pipeline;
    pipeline.push_back(*fwd_);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }
  void backward(Argument& act) {
    std::vector<mkldnn::primitive> pipeline;
    pipeline.push_back(*bwd_);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }
END_DEFINE_ACTIVATION(mkldnn_relu)

/**
 * @brief MKLDNN softmax Activation.
 * \f[
 * P(y=j|x) = \frac{e^{x^Tw_j}}{\sum^K_{k=1}e^{x^Tw_k}}
 * \f]
 */
class ACTIVATION_CLASS_NAME(mkldnn_softmax)
                        : public ActivationFunction, public MkldnnActivation {
private:
  static const std::string name;
  std::shared_ptr<mkldnn::softmax_forward> fwd_;
  int axis;  // 1 means bs*nc, 0 means nc*bs

  // backward
  MatrixPtr sftMaxSum_;
  MatrixPtr sftMaxDot_;
  MatrixPtr one_;

public:
  const std::string& getName() const { return name; }

  void resetDnnFwd(const Argument& arg, std::shared_ptr<void> topDataMD) {
    if (bs_ == arg.getBatchSize()) {
      return;
    }

    MkldnnActivation::reshapeDnnFwd(arg, topDataMD);
    LOG(INFO) << this->getName() << " reshape batchsize: "
          << bs_ << ", " << oc_ << ", " << oh_ << ", " << ow_;
    mkldnn::engine eg = CpuEngine::Instance().getEngine();

    real* pdata = arg.value->getData();
    dataBot_.reset(new mkldnn::memory({*md_, eg}, pdata));
    dataTop_.reset(new mkldnn::memory({*md_, eg}, pdata));
    // TODO(TJ): check if OK use same pdata?
    // in forward src and dst memory can be the same,
    // but in backward not sure it's OK if they are the same, need double check
    // maybe need define a temporary mkldnn:memory to handle the dst
    // and then copy it to the output

    axis = 1;  // 1 means bs*nc, 0 means nc*bs
    // only support forward_scoring by now
    auto fwdMD = mkldnn::softmax_forward::desc(
      mkldnn::prop_kind::forward_scoring, *md_, axis);
    auto fwdPD = mkldnn::softmax_forward::primitive_desc(
      fwdMD, eg);
    fwd_.reset(new mkldnn::softmax_forward(fwdPD, *dataBot_, *dataTop_));

    needResetBwd_ = true;
  }

  /** 
   * each dnn layer should have function
   * to init or reset dnn backward
   */
  void resetDnnBwd(const Argument& arg, std::shared_ptr<void> topDiffFmt) {
    if (!needResetBwd_)
      return;

    // not implement
    // in forward src and dst memory can be the same,
    // but in backward not sure it's OK if they are the same, need double check
    // maybe need define a temporary mkldnn:memory to handle the dst
    // and then copy it to the output
    needResetBwd_ = false;
  }

// mkldnn format only support nchw
  void forward(Argument& act) {
    // for comparing with defalut result
    // MatrixPtr tmp = Matrix::create(act.value->getHeight(),
    // act.value->getWidth(), false, false);
    // MatrixPtr in = Matrix::create(act.value->getHeight(),
    // act.value->getWidth(), false, false);
    // in->copyFrom(*act.value);
    // act.value->softmax(*tmp);

    std::vector<mkldnn::primitive> pipeline;
    pipeline.push_back(*fwd_);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    // for comparing with defalut result
    // for (size_t i = 0; i < std::max(act.value->getElementCnt()/10,
    // (size_t)1); ++i)
    //  LOG(INFO) << "----------src: " << in->getData()[i] << "; "
    //    << tmp->getData()[i] << " --- " << act.value->getData()[i];
  }

  void backward(Argument& act) {
    MatrixPtr outputV = act.value;
    MatrixPtr outputG = act.grad;

    if (outputG->useGpu()) {
      outputG->softmaxBackward(*outputV);
    } else {
      SetDevice device(act.deviceId);
      Matrix::resizeOrCreate(sftMaxDot_, outputG->getHeight(),
                             outputG->getWidth(),
                             /* trans */ false, useGpu(act.deviceId));
      Matrix::resizeOrCreate(sftMaxSum_, outputG->getHeight(), 1,
                             /* trans */ false, useGpu(act.deviceId));
      if (!one_ || one_->getWidth() != outputG->getWidth()) {
        Matrix::resizeOrCreate(one_, 1, outputG->getWidth(),
                               /* trans */ false, useGpu(act.deviceId));
        one_->one();
      }

      sftMaxDot_->dotMul(*outputG, *outputV);
      sftMaxSum_->colMerge(*sftMaxDot_);

      act.grad->softmaxDerivative(*act.value, *sftMaxSum_);
    }
  }
END_DEFINE_ACTIVATION(mkldnn_softmax)

#endif

/**
 * @brief BRelu Activation.
 *
 * forward. y = min(24, max(0, z))
 *
 * derivative of brelu is:
 *
 *    1 if 0 < z < 24
 *
 *    0 otherwise.
 *
 * TODO(yuyang18): Remove magic number 24 or make it configuable.
 */
BEGIN_DEFINE_ACTIVATION(brelu)
void forward(Argument& act) { act.value->brelu(*act.value); }

void backward(Argument& act) { act.grad->breluDerivative(*act.value); }
END_DEFINE_ACTIVATION(brelu)

/**
 * @brief Tanh Activation.
 * \f[
 * f(z) = tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(tanh)
void forward(Argument& act) { act.value->tanh(*act.value); }

void backward(Argument& act) { act.grad->tanhDerivative(*act.value); }
END_DEFINE_ACTIVATION(tanh)

/**
 * @brief Scaled Tanh Activation
 * \f[
 * f(z) = 1.7159 * tanh(2/3*z)
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(stanh)
private:
real a, b;

public:
ACTIVATION_CLASS_NAME(stanh)() : a(1.7159), b(2. / 3.) {}
void forward(Argument& act) { act.value->scaledTanh(*act.value, a, b); }

void backward(Argument& act) {
  act.grad->scaledTanhDerivative(*act.value, a, b);
}
END_DEFINE_ACTIVATION(stanh)

/**
 * @brief Soft Relu Activation.
 * \f[
 * f(z) = ln(1+e^z)
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(softrelu)
void forward(Argument& act) { act.value->softrelu(*act.value); }

void backward(Argument& act) { act.grad->softreluDerivative(*act.value); }
END_DEFINE_ACTIVATION(softrelu)

/**
 * @brief Abs Activation.
 * Forward: f(z) = abs(z)
 *
 * Derivative:
 *
 *     1   if z>0
 *
 *    -1   if z<0
 *
 *     0   if z=0
 */
BEGIN_DEFINE_ACTIVATION(abs)
void forward(Argument& act) {
  SetDevice device(act.deviceId);
  Matrix::resizeOrCreate(act.in, act.value->getHeight(), act.value->getWidth(),
                         /* trans */ false, useGpu(act.deviceId));

  act.in->copyFrom(*act.value);
  act.value->abs(*act.value);
}

void backward(Argument& act) { act.grad->absDerivative(*act.in); }
END_DEFINE_ACTIVATION(abs)

/**
 * @brief Square Activation.
 * \f[
 * f(z) = z^2.
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(square)
void forward(Argument& act) {
  SetDevice device(act.deviceId);
  Matrix::resizeOrCreate(act.in, act.value->getHeight(), act.value->getWidth(),
                         /* trans */ false, useGpu(act.deviceId));

  act.in->copyFrom(*act.value);
  act.value->square(*act.value);
}

void backward(Argument& act) { act.grad->squareDerivative(*act.in); }
END_DEFINE_ACTIVATION(square)

/**
 * @brief Exponential Activation.
 * \f[
 * f(z) = e^z
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(exponential)
void forward(Argument& act) { act.value->exp(*act.value); }

void backward(Argument& act) { act.grad->expDerivative(*act.value); }
END_DEFINE_ACTIVATION(exponential)

/**
 * @brief Logarithm Activation.
 * \f[
 * f(z) = log(z)
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(log)
void forward(Argument& act) {
  SetDevice device(act.deviceId);
  Matrix::resizeOrCreate(act.in, act.value->getHeight(), act.value->getWidth(),
                         /* trans */ false, useGpu(act.deviceId));

  act.in->copyFrom(*act.value);
  act.value->log(*act.value);
}

void backward(Argument& act) { act.grad->dotDiv(*act.grad, *act.in); }
END_DEFINE_ACTIVATION(log)

ActivationFunction* ActivationFunction::create(const std::string& type) {
  return gActivationRegistrar.createByType(type);
}

std::vector<std::string> ActivationFunction::getAllRegisteredTypes() {
  std::vector<std::string> types;
  gActivationRegistrar.forEachType([&](const std::string& type) {
      types.push_back(type);
    });
  return types;
}
}  // namespace paddle
