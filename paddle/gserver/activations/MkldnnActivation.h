/* Copyright (c) 2016 */

#pragma once

#include "ActivationFunction.h"
#include "paddle/parameter/Argument.h"

#include "mkldnn.hpp"
// #include "paddle/gserver/layers/MkldnnMemory.h"

using namespace mkldnn;

namespace paddle {

/**
 * @brief Base class of MKLDNN Activation.
 *
 */
class MkldnnActivation {
public:
  /// For dnn engine
  std::shared_ptr<engine> cpuEngine_;
  
  // dims
  int bs_, oc_, oh_, ow_;
  
  // mkldnn
  std::shared_ptr<memory::desc> srcMD_;
  std::shared_ptr<memory::desc> dstMD_;
  std::shared_ptr<memory> dataBot_;
  std::shared_ptr<memory> dataTop_;
  bool needResetBwd_;

public:
  explicit MkldnnActivation()
    : cpuEngine_(NULL),
      bs_(0),
      oc_(0),
      oh_(0),
      ow_(0),
      srcMD_(NULL),
      dstMD_(NULL),
      dataBot_(NULL),
      dataTop_(NULL),
    //  diffBot_(NULL),
    //  diffTop_(NULL),
      needResetBwd_(true)
    {}

  virtual ~MkldnnActivation() {}
  /** 
   * each dnn layer should have function
   * to init or reset dnn forward
   */
  virtual void resetDnnFwd(const Argument& arg) = 0;
  /** 
   * each dnn layer should have function
   * to init or reset dnn backward
   */
  virtual void resetDnnBwd(const Argument& arg) = 0;

};

}  // namespace paddle
