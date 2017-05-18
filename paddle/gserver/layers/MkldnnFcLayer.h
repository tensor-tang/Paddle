/* Copyright (c) 2017 */

#pragma once

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnMemory.h"

namespace paddle {

/**
 * @brief A subclass of MkldnnLayer fc layer.
 *
 * The config file api is mkldnn_fc
 */
class MkldnnFcLayer : public MkldnnLayer {
protected:
  std::shared_ptr<mkldnn::inner_product_forward> fwd_;
  std::shared_ptr<mkldnn::inner_product_backward_data> bwdData_;
  std::shared_ptr<mkldnn::inner_product_backward_weights> bwdWgt_;

  // top diff for backward weight
  // it may have different format with topdiff backward data
  MkldnnBufferPtr topDiffBwdWgt_;

  // if image width and height !=0
  bool hasSpatial_;

  bool hasBias_;

  bool usePrevLayout_;

  /// weight data and diff buffers
  MkldnnBufferPtr wgtData_;
  MkldnnBufferPtr wgtDiff_;
  /// weight data and diff buffers
  MkldnnBufferPtr biasData_;
  MkldnnBufferPtr biasDiff_;

  // fc weight
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> biases_;

  // support inference with paddle format wgt if do not use dnn wgt
  MatrixPtr paddleWgt_;

  // use paddle weight format
  bool usePaddleFmt_;
  bool hasInited_;

  // input size (== ic*ih*iw) by batch size
  std::vector<size_t> inputSizeByBS_;

  size_t inputLayerSize_;

public:
  explicit MkldnnFcLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwd_(nullptr),
      hasSpatial_(false),
      hasBias_(false),
      usePrevLayout_(false),
      wgtData_(nullptr),
      wgtDiff_(nullptr),
      biasData_(nullptr),
      biasDiff_(nullptr),
      paddleWgt_(nullptr),
      hasInited_(false)
    {}

  ~MkldnnFcLayer() {}

  // load the settings from proto
  void loadConfig();

  bool initDnnWgt(const LayerMap& layerMap, const ParameterMap& parameterMap);

  // reshape 
  // output matrix height and width 
  // and the bs
  void reshapeOutputInfo();

  void clearAllDnnCvtFlags() {
    MkldnnLayer::clearAllDnnCvtFlags();
    if (topDiffBwdWgt_) topDiffBwdWgt_->clearCvtFlag();
    if (biasData_) biasData_->clearCvtFlag();
    if (wgtData_) wgtData_->clearCvtFlag();
    if (biasDiff_) biasDiff_->clearCvtFlag();
    if (wgtDiff_) wgtDiff_->clearCvtFlag();
  }

  void resetDnnFwd(PassType passType);

  void resetDnnBwd();

  void submitDnnFwd();

  void submitDnnBwd(const UpdateCallback& callback);

protected:
  // fc can only change the out mat height but not width(layersize)
  // since when input width changed, the wgt would need be changed too
  void reshapeOutMatSize();

  void reshapeBatchSize();

  // reshape input and output channel, height and width
  // layerSize == channel * height * width
  void reshapeImgSize();

  // FC do not change output size
  void keepOutputSize();

  void resetDnnConfigs();

  void resetDnnFwdBuffers();

  void resetDnnUserLayout();

  void usePrevDnnLayout();

  void resetDnnFwdPD(
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>& fwdPD);

  void resetDnnIntlLayout(
    const std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>& fwdPD);

  void keepDnnLayoutToNext(
    const std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>& fwdPD);

  void resetDnnFwdHandle(
    const std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>& fwdPD);

  // when training get initial wgt from paddle format
  // when scoring from paddle also need get initial wgt from paddle format
  // however when scoring with mkldnn wgt donot need get initial wgt
  void getInitialWgtFromPaddle();

  void forwardDnn();

  void submitBwdData(int idx);

  void submitBwdWgts(int idx);

};

}  // namespace paddle
