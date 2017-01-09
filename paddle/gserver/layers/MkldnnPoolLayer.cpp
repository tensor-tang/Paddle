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

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnPoolLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_pool, MkldnnPoolLayer);

bool MkldnnPoolLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* the size of inputs for pool-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  const PoolConfig& conf = config_.inputs(0).pool_conf();
//  if (!conf.caffe_mode()) {
//    LOG(FATAL) << "Only support caffe mode with MKL-DNN by now!";
//  }
  const std::string& poolType_ = conf.pool_type();
  if (poolType_ == "max-projection") {
    poolAlgo_ = algorithm::pooling_max;
  } else if (poolType_ == "avg-projection") {
    poolAlgo_ = algorithm::pooling_avg;
  } else {
    LOG(FATAL) << "unknow pooling type!";
  }

  ic_.push_back(conf.channels());
  iw_.push_back(conf.img_size());
  ow_.push_back(conf.output_x());
  ih_.push_back(conf.has_img_size_y() ? conf.img_size_y() : conf.img_size());
  oh_.push_back(conf.has_output_y() ? conf.output_y() : conf.output_x());

  fw_ = conf.size_x();
  sw_ = conf.stride();
  pw_ = conf.padding();

  fh_ = conf.has_size_y() ? conf.size_y() : conf.size_x();
  sh_ = conf.has_stride_y() ? conf.stride_y() : conf.stride();
  ph_ = conf.has_padding_y() ? conf.padding_y() : conf.padding();
  bs_ = 0;
  oc_ = ic_[0];
  return true;
}

size_t MkldnnPoolLayer::getOneBatchSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  int height = inputLayers_[0]->getOutput().getFrameHeight();
  int width = inputLayers_[0]->getOutput().getFrameWidth();
  if (height != 0) ih_[0] = height;
  if (width != 0) iw_[0] = width;
  // TODO(TJ): why use false caffe mode??
  oh_[0] = outputSize(ih_[0], fh_, ph_, sh_, false);
  ow_[0] = outputSize(iw_[0], fw_, pw_, sw_, false);
  return oh_[0] * ow_[0] * oc_;
}

// whether reset batchsize and image size of input and output
bool MkldnnPoolLayer::reshapeOutput() {
  if (bs_ == getInput(0).getBatchSize()) {
    // can remove resetoutput
    // when confirm how multi inputs work and whether to clear diff
    resetOutput(bs_, getOneBatchSize());
    return false;
  }

  // reset image size
  size_t layersize = getOneBatchSize();
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);

  // reset data
  bs_ = getInput(0).getBatchSize();
  LOG(INFO) << "layer name: " << getName();
  LOG(INFO) << "reshape batch size: " << bs_;
  resetOutput(bs_, layersize);
  printInfo();
  return true;
}

void MkldnnPoolLayer::resetDnnFwd(PassType passType) {
  LOG(INFO) << "reset mkldnn forward of pool layer: " << config_.name();

  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";

  // create dim structure that describes user data.
  memory::dims botDims = {bs_, ic_[0], ih_[0], iw_[0]};
  memory::dims kernel = {fh_, fw_};
  memory::dims strides = {sh_, sw_};
  memory::dims padding = {ph_, pw_};
  memory::dims topDims = {bs_, oc_, oh_[0], ow_[0]};

  dataBot_.reset(new MkldnnBuffer(botDims));
  dataTop_.reset(new MkldnnBuffer(topDims));

  // init user memory of bottom, weights and bias
  real *botData = getPrev(0)->getOutputValue()->getData();
  real *topData = getOutputValue()->getData();
  const std::shared_ptr<memory::desc> prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    dataBot_->initUser(botData, *prvMD, *engine_);
    LOG(INFO) << "use prev format: " << DNN_FORMAT[dataBot_->getUserFmt()];
  } else {
    dataBot_->initUser(botData, botDims, memory::format::nchw, *engine_);
  }

  // create pool desc from internal desc
  std::shared_ptr<pooling_forward::desc> fwdDesc;
  prop_kind pk = (passType == PASS_TEST) ? prop_kind::forward_scoring :
    prop_kind::forward_training;

  fwdDesc.reset(new pooling_forward::desc(pk, poolAlgo_,
                    dataBot_->getMDAny(), dataTop_->getMDAny(),
                    strides, kernel, padding, padding,
                    padding_kind::zero));
  // init cvt
  if (dataBot_->initCvt(dataBot_->getUserPD(), dnnCvtUser2Internal)) {
    LOG(INFO) << "need reorder --- bottom data: "
      << DNN_FORMAT[dataBot_->getUserFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataBot_->getIntlFmt()];
  }
  fwdPD_.reset(new pooling_forward::primitive_desc(*fwdDesc, *engine_));

  // init top user memory and cvt
  if (setDnnTopDataFmt_) {
    dataTop_->initUser(topData, fwdPD_->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
    LOG(INFO) << "set next format: " << DNN_FORMAT[dataTop_->getUserFmt()];
  } else {
    dataTop_->initUser(topData, topDims, memory::format::nchw, *engine_);
  }
  if (dataTop_->initCvt(fwdPD_->dst_primitive_desc(), dnnCvtInternal2User)) {
    LOG(INFO) << "need reorder --- top data: "
      << DNN_FORMAT[dataTop_->getIntlFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataTop_->getUserFmt()];
  }

  withWorkspace_ = passType != PASS_TEST && poolAlgo_ != algorithm::pooling_avg;
  if (withWorkspace_) {
    workspace_.reset(new memory(fwdPD_->workspace_primitive_desc()));
  } else {
    auto p_workspace_desc = memory::primitive_desc(
      {{}, memory::data_type::f32, memory::format(dataTop_->getIntlFmt())},
      *engine_);
    workspace_.reset(new memory(p_workspace_desc));
  }
  LOG(INFO) << "data format flow --- "
    << DNN_FORMAT[dataBot_->getUserFmt()] << " >>> ("
    << DNN_FORMAT[dataBot_->getIntlFmt()] << " >>> "
    << DNN_FORMAT[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FORMAT[dataTop_->getUserFmt()];
}

void MkldnnPoolLayer::resetDnnBwd() {
  /*LOG(INFO) << "init or reset conv backward of layer: " << config_.name();
  bool hasBias = (biases_ && biases_->getWGrad()) ? true : false;
  // TODO: only care about i==0 by now
  real *topdiff = getOutputGrad()->getData();
  // init top diff user
  diffTop_.reset(new MkldnnBuffer(dataTop_->getDefaultDims()));
  
  const std::shared_ptr<mkldnn::memory::desc> inputDiffMD = getTopDiffMD();
  if (inputDiffMD) {
    diffTop_->initUser(topdiff, *inputDiffMD, *engine_);
  } else {
    diffTop_->initUser(topdiff, diffTop_->getDefaultDims(),
      memory::format::nchw, *engine_);
  }
  
  if (hasBias) {
    // bias backward can not be execute seperately, 
    //only can execute with filter bakcward
    real* biasdiff = biases_->getWGrad()->getData();
    diffBias_.reset(new MkldnnBuffer(dataBias_->getDefaultDims()));
    diffBias_->initUser(biasdiff, diffBias_->getDefaultDims(), memory::format::x, *engine_);
  }
  
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize())
      << "Assert batchsize of input layers are equal";

    // create dim structure that describes user data.
    memory::dims strides = {sh_[i], sw_[i]};
    memory::dims padding = {ph_[i], pw_[i]};

    // backward weight and bias before data*****************************
    if (weights_[i]->getWGrad()) {
      real* wgtdiff = weights_[i]->getWGrad()->getData();
      // init weight diff user
      diffWgt_.reset(new MkldnnBuffer(dataWgt_->getDefaultDims()));
      diffWgt_->initUser(wgtdiff, diffWgt_->getDefaultDims(), 
        memory::format(dataWgt_->getUserFmt()), *engine_);
    } else {
      LOG(FATAL) << "should have weight";
    //  continue;
    }
    std::shared_ptr<convolution_forward::desc> bwdWgtFwdDesc;
    std::shared_ptr<convolution_backward_weights::desc> bwdWgtDesc;
    if (hasBias && diffBias_ != NULL) {
      bwdWgtFwdDesc.reset(new convolution_forward::desc(
        prop_kind::forward_training, algorithm::convolution_direct, 
        dataBot_->getIntlMD(),
        diffWgt_->getMDAny(),
        diffBias_->getMDAny(),
        diffTop_->getMDAny(),
        strides, padding, padding, padding_kind::zero));
      bwdWgtDesc.reset(new convolution_backward_weights::desc(
        algorithm::convolution_direct, 
        dataBot_->getIntlMD(),
        diffWgt_->getMDAny(),
        diffBias_->getMDAny(),
        diffTop_->getMDAny(),
        strides, padding, padding, padding_kind::zero));
    } else {
      bwdWgtFwdDesc.reset(new convolution_forward::desc(
        prop_kind::forward_training, algorithm::convolution_direct, 
        dataBot_->getIntlMD(),
        diffWgt_->getMDAny(),
        diffTop_->getMDAny(),
        strides, padding, padding, padding_kind::zero));
      bwdWgtDesc.reset(new convolution_backward_weights::desc(
        algorithm::convolution_direct,
        dataBot_->getIntlMD(),
        diffWgt_->getMDAny(),
        diffTop_->getMDAny(), 
        strides, padding, padding, padding_kind::zero));
    }
    std::shared_ptr<convolution_forward::primitive_desc> bwdWgtFwdPD;
    bwdWgtFwdPD.reset(new convolution_forward::primitive_desc(
      *bwdWgtFwdDesc, *engine_));
    bwdWgtPD_.reset(new convolution_backward_weights::primitive_desc(
      *bwdWgtDesc, *engine_, *bwdWgtFwdPD));
    if (hasBias && diffBias_ != NULL) {
      if (diffBias_->initCvt(bwdWgtPD_->diff_bias_primitive_desc(),
        dnnCvtInternal2User)) {
        LOG(INFO) << "need reorder --- bias diff: "
          << DNN_FORMAT[diffBias_->getIntlFmt()]
          << " >>>>> "
          << DNN_FORMAT[diffBias_->getUserFmt()];
      }
    }
    if (diffWgt_->initCvt(bwdWgtPD_->diff_weights_primitive_desc(),
      dnnCvtInternal2User)) {
      LOG(INFO) << "need reorder --- weight diff: "
        << DNN_FORMAT[diffWgt_->getIntlFmt()]
        << " >>>>> "
        << DNN_FORMAT[diffWgt_->getUserFmt()];
    }
    if (diffTop_->initCvt(bwdWgtPD_->diff_dst_primitive_desc(), 
      dnnCvtUser2Internal)) {
      LOG(INFO) << "need reorder --- top diff: "
        << DNN_FORMAT[diffTop_->getUserFmt()]
        << " >>>>> "
        << DNN_FORMAT[diffTop_->getIntlFmt()];
    }
    CHECK(dataBot_->getIntlPD() == bwdWgtPD_->src_primitive_desc());
    
    // then backward data *************************************
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      continue; // data layer has not diff
    }
    diffBot_.reset(new MkldnnBuffer(dataBot_->getDefaultDims()));
    // init backward data primitive desc
    std::shared_ptr<convolution_forward::desc> bwdDataFwdDesc;
    std::shared_ptr<convolution_backward_data::desc> bwdDataDesc;
    bwdDataFwdDesc.reset(new convolution_forward::desc(
      prop_kind::forward_training, algorithm::convolution_direct, 
      diffBot_->getMDAny(),
      dataWgt_->getIntlMD(), 
      diffTop_->getIntlMD(),
      strides, padding, padding, padding_kind::zero));
    bwdDataDesc.reset(new convolution_backward_data::desc(
      algorithm::convolution_direct,
      diffBot_->getMDAny(),
      dataWgt_->getIntlMD(), 
      diffTop_->getIntlMD(),
      strides, padding, padding, padding_kind::zero));
    std::shared_ptr<convolution_forward::primitive_desc> bwdDataFwdPD;
    bwdDataFwdPD.reset(new convolution_forward::primitive_desc(*bwdDataFwdDesc, *engine_));
    bwdDataPD_.reset(new convolution_backward_data::primitive_desc(*bwdDataDesc,
                        *engine_, *bwdDataFwdPD));
    // init user memory and cvt
    real* botdiff = prevLayer->getOutputGrad()->getData();
    if (setDnnBotDiffFmt_[i]) {
      diffBot_->initUser(botdiff, bwdDataPD_->diff_src_primitive_desc());
      getPrev(i)->setTopDiffMD(diffBot_->getUserMD());
    } else {
      diffBot_->initUser(botdiff, diffBot_->getDefaultDims(), memory::format::nchw, *engine_);
    }
    if (diffBot_->initCvt(bwdDataPD_->diff_src_primitive_desc(), 
      dnnCvtInternal2User)) {
      LOG(INFO) << "need reorder --- bottom diff: "
        << DNN_FORMAT[diffBot_->getIntlFmt()]
        << " >>>>> "
        << DNN_FORMAT[diffBot_->getUserFmt()];
    }
    CHECK(dataWgt_->getIntlPD() == bwdDataPD_->weights_primitive_desc());
    CHECK(diffTop_->getIntlPD() == bwdDataPD_->diff_dst_primitive_desc());
    LOG(INFO) << "diff format flow --- "
      << DNN_FORMAT[diffBot_->getUserFmt()] << " <<< ("
      << DNN_FORMAT[diffBot_->getIntlFmt()] << " <<< "
      << DNN_FORMAT[diffTop_->getIntlFmt()] << ") <<< "
      << DNN_FORMAT[diffTop_->getUserFmt()];

  }
*/
}

void MkldnnPoolLayer::myFwd(PassType passType) {
  /// all sumbit cvt should be clear
  clearAllCvtFlags();

  real *botdata = getPrev(0)->getOutputValue()->getData();
  real *topdata = getOutputValue()->getData();

  std::vector<primitive> poolFwd;
  dataBot_->submitCvt(poolFwd, botdata);

  if (withWorkspace_) {
    poolFwd.push_back(pooling_forward(*fwdPD_,
      *(dataBot_->getIntlMem()), *(dataTop_->getIntlMem()),
      *workspace_));
  } else {
    poolFwd.push_back(pooling_forward(*fwdPD_,
      *(dataBot_->getIntlMem()), *(dataTop_->getIntlMem())));
  }
  dataTop_->submitCvt(poolFwd, topdata);

  // start forward
  REGISTER_TIMER_INFO("mkldnnPoolFwd", getName().c_str());
  stream(stream::kind::eager).submit(poolFwd).wait();
//  LOG(INFO) << "------------" << topdata[0];
// << "," << topdata[1] << "," << topdata[2];
}

void MkldnnPoolLayer::exFwd(PassType passType) {
  const Argument& in = getInput(0);
  const Argument& out = output_;
  CHECK_EQ(getOneBatchSize(), out.value->getWidth());
  MatrixPtr inputV = in.value;
  MatrixPtr outV = out.value;
  outV->maxPoolForward(*inputV, ih_[0], iw_[0], ic_[0], fw_, fh_,
                       sh_, sw_, oh_[0], ow_[0], ph_, pw_);

//  real *topdata = getOutputValue()->getData();
//  LOG(INFO) << "------------" << topdata[0];
// << "," << topdata[1] << "," << topdata[2];
}

void MkldnnPoolLayer::submitDnnFwd(PassType passType) {
//  exFwd(passType);
  myFwd(passType);
}

void MkldnnPoolLayer::exBwd(const UpdateCallback &callback) {
  (void)callback;
  const Argument& in = getInput(0);
  MatrixPtr outGrad = getOutputGrad();
  MatrixPtr inputV = in.value;
  MatrixPtr outV = getOutputValue();
  MatrixPtr inputGrad = in.grad;

  if (NULL == getInputGrad(0)) {
    return;
  }

  inputGrad->maxPoolBackward(*inputV, ih_[0], iw_[0], *outGrad, *outV,
                             fw_, fh_, sh_, sw_, oh_[0], ow_[0],
                             1, 1, ph_, pw_);
}

void MkldnnPoolLayer::submitDnnBwd(const UpdateCallback &callback) {
  exBwd(callback);

  // dnn backward
  /*
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    // backward weights before data, since may have not botdiff in some layer
    if (weights_[i]->getWGrad()) {
      submitBwdWgts(i, getPrev(i)->getOutputValue(), getOutputGrad());
      // Increasing the number of gradient 
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
    submitBwdData(i, getOutputGrad(), getPrev(i)->getOutputGrad());
  }
  if (biases_ && biases_->getWGrad()) {
    // Increasing the number of gradient 
    biases_->getParameterPtr()->incUpdate(callback);
  }
  */
}

}  // namespace paddle
