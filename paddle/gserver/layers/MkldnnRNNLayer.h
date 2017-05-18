/* Copyright (c) 2017 */

#pragma once

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnMemory.h"

namespace paddle {

/**
 * @brief A subclass of MkldnnLayer rnn layer.
 *
 * The config file api is mkldnn_rnn
 */
class MkldnnRNNLayer : public MkldnnLayer {
protected:
//  std::shared_ptr<mkldnn::rnn_forward> fwd_;
//  std::shared_ptr<mkldnn::rnn_backward> bwd_;

  // input mode:
  // "skip_input" or "linear_input"
  std::string inputMode_;  // TODO: use type:  mkldnn::input_mode
  // algorithm kind: rnn_relu, rnn_tanh, rnn_lstm, rnn_gru
  std::string algKind_;  // TODO: use type: mkldnn::algorithm
  // if use bi-diretion
  bool useBiDir_;
  // layer num
  int layerNum_;
  // only used in bi-direction:
  // sum or concat(defualt)
  // sum should be impelenmted 
  bool sumOutput_;


public:
  explicit MkldnnRNNLayer(const LayerConfig& config)
    : MkldnnLayer(config),
//      fwd_(nullptr),
//      bwd_(nullptr),
      useBiDir_(false),
      layerNum_(1),
      sumOutput_(false)
    {}

  ~MkldnnRNNLayer() {}

  bool initDnn(const LayerMap& layerMap, const ParameterMap& parameterMap);

  virtual void clearAllDnnCvtFlags() {
    MkldnnLayer::clearAllDnnCvtFlags();
  }


  // load the settings from proto
  virtual void loadConfig();


  // reshape 
  // output matrix height and width 
  // and the bs
  virtual void reshapeOutputInfo();

  void resetDnnFwd();

  void resetDnnBwd();

  void submitDnnFwd();
  void submitDnnBwd(const UpdateCallback& callback);

};

}  // namespace paddle
