/* Copyright (c) 2017 */

#pragma once

#include "ActivationFunction.h"
#include "paddle/parameter/Argument.h"

#include "paddle/gserver/layers/MkldnnBase.h"
// #include "paddle/gserver/layers/MkldnnMemory.h"

namespace paddle {

/**
 * @brief Base class of MKLDNN Activation.
 *
 */
class MkldnnActivation {
public:
  // mkldnn
  std::shared_ptr<mkldnn::memory> botData_;
  std::shared_ptr<mkldnn::memory> botDiff_;
  std::shared_ptr<mkldnn::memory> topData_;
  std::shared_ptr<mkldnn::memory> topDiff_;

public:
  MkldnnActivation()
    : botData_(nullptr),
      botDiff_(nullptr),
      topData_(nullptr),
      topDiff_(nullptr)
    {}

  virtual ~MkldnnActivation() {}

  /** 
   * each dnn layer should have function
   * to reset dnn forward
   */
  virtual void resetDnnFwd(const Argument& arg,
    std::shared_ptr<void> topDataMD) = 0;

  /** 
   * each dnn layer should have function
   * to reset dnn backward
   */
  virtual void resetDnnBwd(const Argument& arg,
    std::shared_ptr<void> topDiffMD) = 0;
};

}  // namespace paddle
