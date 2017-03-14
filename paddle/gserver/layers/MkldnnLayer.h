/* Copyright (c) 2016 */

#pragma once

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Stat.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnBase.h"
#include "MkldnnMemory.h"

namespace paddle {

class MkldnnLayer;
typedef std::shared_ptr<MkldnnLayer> MkldnnLayerPtr;

/**
 * @brief Base class of Dnnlayer.
 *
 */
class MkldnnLayer : public Layer {
public:
  /// data buffers
  // TODO(TJ): need vector when know how RNN works
  MkldnnBufferPtr dataBot_;
  MkldnnBufferPtr dataTop_;
  /// diff buffer
  MkldnnBufferPtr diffBot_;
  MkldnnBufferPtr diffTop_;

  // dims and format for user buffer
  std::vector<mkldnn::memory::dims> botDims_, wgtDims_, biasDims_;
  std::vector<mkldnn::memory::format> botFmt_, wgtFmt_, biasFmt_;
  mkldnn::memory::dims topDims_;
  mkldnn::memory::format topFmt_;

  /// for summing topdiffs
  std::vector<MkldnnBufferPtr> topDiffBuffers_;
  std::shared_ptr<mkldnn::sum> sumTopDiffs_;
  // tmp result of sum
  MkldnnBufferPtr tmpDiff_;

  // The spatial dimensions of height and width of input feature map.
  std::vector<int> ih_, iw_;
  // The spatial dimensions of height and width of output feature map.
  std::vector<int> oh_, ow_;  // TODO(TJ): no need vector??
  // input channel number
  std::vector<int> ic_;
  // output channels
  int oc_;
  // batchsize
  int bs_;

  // flags whether to set memory format of top data or bots diff
  // only one top data but may have several bot diff
  bool nextIsDnn_;
  std::vector<bool> prevIsDnn_;

  // each MKLDNN layers has WriteToMode and AddToMode
  // use WriteToMode if addSize_ == 0, otherwise use AddToMode
  int addSize_;

  // layers with weight have an option to choose
  // whether use mkldnn foramt weight to get a better performance
  // sacrificing the compatibility with original CPU layers
  bool useMkldnnWgt_;

  bool needResetBwd_;

  // some operations should not be called at init function
  // and should only do once : like initflags and prepare topdiffMD etc.
  bool prepareOnce_;

public:
  explicit MkldnnLayer(const LayerConfig& config)
    : Layer(config),
      dataBot_(nullptr),
      dataTop_(nullptr),
      diffBot_(nullptr),
      diffTop_(nullptr),
      nextIsDnn_(false),
      addSize_(0),
      useMkldnnWgt_(true),
      needResetBwd_(true),
      prepareOnce_(true)
    {}

  ~MkldnnLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap) {
    /* Initialize the basic parent class */
    if (!Layer::init(layerMap, parameterMap)) return false;

    bs_ = 0;
    oc_ = 0;
    topDims_ = {0};
    topFmt_ = mkldnn::memory::format::nchw;
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      botDims_.push_back({0});
      wgtDims_.push_back({0});
      biasDims_.push_back({0});
      botFmt_.push_back(mkldnn::memory::format::nchw);
      wgtFmt_.push_back(mkldnn::memory::format::format_undef);
      biasFmt_.push_back(mkldnn::memory::format::x);
    }

    return initDnn(layerMap, parameterMap);
  }

  void forward(PassType passType) {
    Layer::forward(passType);

    // reshape if batch size changes
    if (bs_ == getInput(0).getBatchSize()) {
      // choose to clear top data or top diff
      clearDataDiff();
    } else {
      if (prepareOnce_) {
        topDiffBuffers_.resize(nextLayers_.size(), nullptr);
        dnnOutGrads_.resize(nextLayers_.size(), nullptr);
        for (size_t i = 0; i < nextLayers_.size(); ++i) {
          topDiffMDs_.push_back(nullptr);
          dnnOutIdxMap_[nextLayers_[i]->getName()] = i;
        //  LOG(INFO)<<"next name:" << nextLayers_[i]->getName();
        }
        if (nextLayers_.size() > 0 && topDiffMDs_.size() > nextLayers_.size()) {
          // in base layer init will add one nullptr for PASS_grad check
          // so remove the redundant one
          topDiffMDs_.pop_back();
          CHECK_EQ(topDiffMDs_.size(), nextLayers_.size());
        } else {
          CHECK_EQ(topDiffMDs_.size() - 1, nextLayers_.size());
        }
        // this function can work only after all layers init done
        // and should be called only once
        initDnnflags();
        prepareOnce_ = false;
      }

      bs_ = getInput(0).getBatchSize();
      VLOG(1) << "reset forward batch size to " << bs_
        << " of mkldnn layer: " << getName();

      // reshape the input and output size
      REGISTER_TIMER_INFO("mkldnn_ResetDnnTimer", getName().c_str());
      reshape();
      printInfo();

      // mkldnn init or reset forward
      resetOutput(bs_, getSize());
      resetDnnFwd(passType);

      // print the data flow
      for (size_t i = 0; i != inputLayers_.size(); ++i) {
        // TODO(TJ): consider multi input
        if (dataBot_ && dataTop_)
          VLOG(1) << "data format flow --- "
            << DNN_FMTS[dataBot_->getUserFmt()] << " >>> ("
            << DNN_FMTS[dataBot_->getIntlFmt()] << " >>> "
            << DNN_FMTS[dataTop_->getIntlFmt()] << ") >>> "
            << DNN_FMTS[dataTop_->getUserFmt()];
        // in batch norm layer, the other two will be moving mean and var
        if (getType() == "mkldnn_batch_norm")
          break;
      }
      
      if (passType != PASS_TEST && nextLayers_.size() > 1) {  // Training
        sumTopDiffs_ = nullptr;
        for (size_t i = 0; i < nextLayers_.size(); ++i) {
          topDiffMDs_[i] = nullptr;
          dnnOutGrads_[i] = Matrix::create(bs_, getSize(), false, false);
        }
      }
      needResetBwd_ = true;
    }

    // all sumbit cvt should be clear
    clearAllDnnCvtFlags();
    // then submit dnn forward
    REGISTER_TIMER_INFO("mkldnn_FwdTimer", getName().c_str());
    submitDnnFwd(passType);
  }

  void backward(const UpdateCallback& callback) {
    if (needResetBwd_) {
      needResetBwd_ = false;
      // mkldnn init or reset backward
      VLOG(1) << "reset backward batch size to " << bs_
        << " of mkldnn layer: " << getName();

      prepareTopDiff();
      resetDnnBwd();

      // print the diff flow
      for (size_t i = 0; i != inputLayers_.size(); ++i) {
        // TODO(TJ): consider multi input
        if (diffBot_ && diffTop_)
          VLOG(1) << "diff format flow --- "
            << DNN_FMTS[diffBot_->getUserFmt()] << " <<< ("
            << DNN_FMTS[diffBot_->getIntlFmt()] << " <<< "
            << DNN_FMTS[diffTop_->getIntlFmt()] << ") <<< "
            << DNN_FMTS[diffTop_->getUserFmt()];
        // in batch norm layer, the other two will be moving mean and var
        if (getType() == "mkldnn_batch_norm")
          break;
      }
    }

    // submit dnn backward
    REGISTER_TIMER_INFO("mkldnn_BwdTimer", getName().c_str());
    if (nullptr != sumTopDiffs_) {
      LOG(INFO)<<"---------------------------------------" << getName();
      std::vector<mkldnn::primitive> sum;
      sum.push_back(*sumTopDiffs_);
      mkldnn::stream(mkldnn::stream::kind::eager).submit(sum).wait();
    }
    submitDnnBwd(callback);
  }

  /**
   * if have several input topdiffs
   * then create handle to sum them
   */
  virtual void prepareTopDiff() {
    sumTopDiffs_ = nullptr;
    if (nextLayers_.size() <= 1)
      return;
    mkldnn::engine eg = CpuEngine::Instance().getEngine();
    std::vector<mkldnn::memory::primitive_desc> srcPDs;
    std::vector<std::shared_ptr<mkldnn::memory::desc>> prvMDs;
    std::vector<mkldnn::primitive::at> srcMems;
    std::vector<double> scales;
    CHECK_EQ(nextLayers_.size(), topDiffBuffers_.size());
    for (size_t i = 0; i < topDiffBuffers_.size(); ++i) {
      // 1. create buffers and init user
      real* diff = dnnOutGrads_[i]->getData();
      topDiffBuffers_[i].reset(new MkldnnBuffer());
      topDiffBuffers_[i]->initUser(diff, topDims_, topFmt_, eg);
      // 2. use private MD when init user if has
      const std::shared_ptr<mkldnn::memory::desc>& prvMD = getTopDiffMD(i);
      if (prvMD) {
        topDiffBuffers_[i]->resetUser(diff, *prvMD, eg);
        prvMDs.push_back(prvMD);
      }
      // 3. init Intl with empty cvt
      topDiffBuffers_[i]->initCvt();
      CHECK(i == 0 || (topDiffBuffers_[i-1]->getIntlSize()
        == topDiffBuffers_[i]->getIntlSize())) << "All size should be equal";
      // 4. save buffers
      scales.push_back(1.0);  // no scale
      srcPDs.push_back(topDiffBuffers_[i]->getIntlPD());
      srcMems.push_back(*(topDiffBuffers_[i]->getIntlMem()));
    }
    if (prvMDs.size() > 0 && prvMDs.size() != nextLayers_.size()) {
      LOG(INFO) << "prvMDs.size() != nextLayers_.size(): " << prvMDs.size()
        << " vs " << nextLayers_.size();
      LOG(INFO) << "Next layers: ";
      for (size_t i = 0; i < nextLayers_.size(); ++i) {
        LOG(INFO) << nextLayers_[i]->getName()
          << ", type: " << nextLayers_[i]->getType();
      }
      LOG(FATAL)  << "Do not support mixed layer type inside branch";
    }
    // 5. create sum PD
    std::shared_ptr<mkldnn::sum::primitive_desc> sumPD;
    sumPD.reset(new mkldnn::sum::primitive_desc(
      MkldnnBuffer::getMD(topDims_), scales, srcPDs));
    // 6. init the buffer of result
    tmpDiff_.reset(new MkldnnBuffer());
    real *topDiff = getDnnOutputGrad()->getData();
    tmpDiff_->initUser(topDiff, sumPD->dst_primitive_desc());
    tmpDiff_->initCvt();
    // change the first intl MD
    topDiffMDs_[0].reset(new mkldnn::memory::desc(tmpDiff_->getIntlMD()));
    // 7. create sum handle
    sumTopDiffs_.reset(
      new mkldnn::sum(*sumPD, srcMems, *(tmpDiff_->getIntlMem())));
  }

  /**
   * init the flags whether to set memory desc
   * of top data or bot diff.
   * each layer can have its own implements.
   * Caution: Do not call it at init function
   *          this function can work only after all layers init have done
   */
  void initDnnflags() {
    // set topdata internal only if all next layers are MKLDNN layers
    nextIsDnn_ = areNextAllDnn();
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
      prevIsDnn_.push_back(isPrevDnn(i));
    }
  }

  /**
   * Calculate output size based on caffeMode_.
   * - input(+padding): 0123456789
   * - imageSize(+padding) = 10;
   * - filterSize = 3;
   * - stride = 2;
   * - caffeMode_ is true:
       - output: (012), (234), (456), (678)
       - outputSize = 4;
   * - caffeMode_ is false:
   *   - output: (012), (234), (456), (678), (9)
   *   - outputSize = 5;
   *** for conv only support caffe mode by now
   */
  int outputSize(int imageSize, int filterSize, int padding, int stride,
                       bool caffeMode = true) {
    int outputSize;
    if (!caffeMode) {
      outputSize =
          (imageSize - filterSize + 2 * padding + stride - 1) / stride + 1;
    } else {
      outputSize = (imageSize - filterSize + 2 * padding) / stride + 1;
    }
    CHECK_GE(outputSize, 1);
    return outputSize;
  }

  /**
   * each dnn layer should have function 
   * to init the image size of input and output
   * (bs, ic, ih, iw, oc, oh, ow)
   * and other init for specific layers
   * before exit init()
   */
  virtual bool initDnn(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) = 0;

  /**
   * each dnn layer should have function
   * to reshape the input and output size, when batch size changes
   */
  virtual void reshape() = 0;

  /**
   * print some info like input or output size
   */
  virtual void printInfo() {
    for (size_t i = 0; i < ih_.size(); ++i) {
      VLOG(2)
        << "ih: " << ih_[i] << ", iw: " << iw_[i]
        << ", ic: " << ic_[i]
        << ", oh: " << oh_[i] << ", ow: " << ow_[i]
        << ", oc: " << oc_;
    }
  }

  /** 
   * each dnn layer should have function
   * to init or reset forward mkldnn
   */
  virtual void resetDnnFwd(PassType passType) = 0;

  /** 
   * each dnn layer should have function
   * to init or reset backward mkldnn
   */
  virtual void resetDnnBwd() = 0;

  /** 
   * each dnn layer should have function
   * to clear the top data and diff, or choose to reseve.
   * Choose to use reserveOutput or resetOutput
   */
  // TODO(TJ): maybe can remove it
  // when confirm whether need to clear topdiff and how multi inputs work
  virtual void clearDataDiff() = 0;

  /** 
   * each dnn layer should have function
   * to clear all the MkldnnBuffer cvt flags
   */
  virtual void clearAllDnnCvtFlags() {
    if (dataBot_) dataBot_->clearCvtFlag();
    if (dataTop_) dataTop_->clearCvtFlag();
    if (diffBot_) diffBot_->clearCvtFlag();
    if (diffTop_) diffTop_->clearCvtFlag();
  }

  virtual void submitDnnFwd(PassType passType) = 0;
  virtual void submitDnnBwd(const UpdateCallback& callback) = 0;
};

}  // namespace paddle
