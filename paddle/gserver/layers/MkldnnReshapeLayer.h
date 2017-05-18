/* Copyright (c) 2017*/

#include "Layer.h"
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"

#include "MkldnnLayer.h"

namespace paddle {

/**
 * @brief A layer for view
 * @note
 */
class MkldnnReshapeLayer : public MkldnnLayer {
protected:

  std::string reshapeType_;
  // seq lenght from config
  int confSeqLen_;
  // output channel, width and height from config
  int confChannel_, confWidth_, confHeight_;

  /// seq info for mkl
  int mklSeqLen_;

  /// seq info for paddle
  // used for none to seq
  ICpuGpuVectorPtr seqIdx_;
  // for further use
  ICpuGpuVectorPtr subSeqIdx_;
  IVectorPtr seqDims_;

public:
  explicit MkldnnReshapeLayer(const LayerConfig& config)
    : MkldnnLayer(config) {}

  // reload the settings from proto
  void loadConfig();

  bool initDnnWgt(const LayerMap& layerMap,
                           const ParameterMap& parameterMap);

  // reshape 
  // output matrix height and width
  void reshapeOutputInfo();

  /** 
   * each dnn layer should have function
   * to init or reset forward mkldnn
   */
  void resetDnnFwd(PassType passType) {};

  /** 
   * each dnn layer should have function
   * to init or reset backward mkldnn
   */
  void resetDnnBwd() {};

  void submitDnnFwd();

  void submitDnnBwd(const UpdateCallback& callback);

protected:
  int getOutputSeqLen();

  void resetSeqInfo(const int seqLen);

  // reshape output matrix width and height
  void reshapeOutMatSize(const int seqLen);

  // reshape channel, height and width
  // should have only one uncertain size at most.
  // layerSize == channel * height * width
  void reshapeImgSize(const size_t layerSize);

  // upadte output layersize, width and height
  void resetOutputSize();

  // get input seq len, if input is not seq return 1
  int getInputSeqLen();

  bool inputIsSequential();

  void resetMklSeqInfo(int seqLen);

  void resetPaddleSeqInfo(int seqLen);

  void setPaddleSeqInfo(Argument& arg);

  void setMklSeqInfo(Argument& arg);

  void setSeqInfo();

  // only share pointer of value
  void shareValue();

  // only share pointer of grad
  void shareGrad();
};


}  // namespace paddle

