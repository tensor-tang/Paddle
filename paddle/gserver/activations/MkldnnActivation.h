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
  std::shared_ptr<engine> engine_;
  
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
    : engine_(new engine(engine::cpu, 0)),
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

  /**
   * call in resetDnnFwd
   */
  void reshapeDnnFwd(const Argument& arg) {
    int batchsize = arg.getBatchSize();
    
    if (bs_ == batchsize) {
      return;
    }
    bs_ = batchsize;
    oh_ = arg.getFrameHeight(); 
    ow_ = arg.getFrameWidth();
    if (oh_ == 0 && ow_ == 0) {
      // in softmax get width and height return 0
      oh_ = 1;
      ow_ = 1;
      oc_ = arg.value->getElementCnt()/(bs_*oh_*ow_);
      memory::dims dm = {bs_, oc_};
      srcMD_.reset(new memory::desc(dm, memory::data_type::f32,
        memory::format::nc));
      dstMD_.reset(new memory::desc(dm, memory::data_type::f32,
        memory::format::nc));
    } else {
      CHECK(oh_ != 0 && ow_ != 0) << "neither should be zero";
      oc_ = arg.value->getElementCnt()/(bs_*oh_*ow_);
      memory::dims dm = {bs_, oc_, oh_, ow_};
      srcMD_.reset(new memory::desc(dm, memory::data_type::f32,
        memory::format::nchw));
      dstMD_.reset(new memory::desc(dm, memory::data_type::f32,
        memory::format::nchw));
    }
  }
};

}  // namespace paddle
