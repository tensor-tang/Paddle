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
#include "MkldnnConcatLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_concat, MkldnnConcatLayer);

bool MkldnnConcatLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  CHECK(!biasParameter_);
  if (config_.has_add_size()) {
    addSize_ = config_.add_size();
  }
  // TODO(TJ): axis should load from proto, change proto
  // and layersize should change depends on axis, change config.py
  axis_ = 1;
  CHECK(axis_ == 1 || axis_ == 0) << "unknow axis:" << axis_;
  // TODO(TJ): remove me when done
  CHECK_EQ(axis_, 1) << "only support concat channel yet";

  bs_ = 0;
  oc_ = 0;
  for (auto &inputConfig : config_.inputs()) {
    if (inputConfig.has_image_conf()) {
      const ImageConfig &conf = inputConfig.image_conf();
      iw_.push_back(conf.img_size());
      ih_.push_back(conf.img_size());
    } else {
      iw_.push_back(0);
      ih_.push_back(0);
    }
    ic_.push_back(0);
    ow_.push_back(0);
    oh_.push_back(0);
    dataBottoms_.push_back(nullptr);
  }
  return true;
}

void MkldnnConcatLayer::clearDataDiff() {
//  reserveOutput(bs_, getSize());
}

void MkldnnConcatLayer::reshape() {
  int sum_ch = 0;
  int sum_bs = 0;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    int height = inputLayers_[i]->getOutput().getFrameHeight();
    int width = inputLayers_[i]->getOutput().getFrameWidth();
    CHECK(height * width != 0 || ih_[i] * iw_[i] != 0);
    if (height != 0) ih_[i] = height;
    if (width != 0) iw_[i] = width;
    oh_[i] = ih_[i];
    ow_[i] = iw_[i];
    // check all image size equal
    CHECK(i == 0 || (ih_[i-1] == ih_[i] && iw_[i-1] == iw_[i]));
    ic_[i] = inputLayers_[i]->getSize() / ih_[i] / iw_[i];
    sum_ch += ic_[i];
    sum_bs += getInput(i).getBatchSize();
    if (axis_ == 0) {
      CHECK_EQ(inputLayers_[i]->getSize(), getSize());
    }
  }
  // reset output image size
  getOutput().setFrameHeight(oh_[0]);
  getOutput().setFrameWidth(ow_[0]);
  if (axis_ == 1) {
    oc_ = getSize() / oh_[0] / ow_[0];
    CHECK_EQ(oc_, sum_ch);
  } else {  // axis == 0
    oc_ = ic_[0];
    // TODO(TJ): beloew not work yet, since bs_ = getInput(0).getBatchSize()
    CHECK_EQ(bs_, sum_bs);
  }
}

void MkldnnConcatLayer::resetDnnFwd(PassType passType) {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();

  // create top buffer and init user, only have one output
  dataTop_.reset(new MkldnnBuffer());
  real *topData = getOutputValue()->getData();
  topFmt_ = memory::format::nchw;
  topDims_ = {bs_, oc_, oh_[0], ow_[0]};
  dataTop_->initUser(topData, topDims_, topFmt_, eg);

  // prepare bottoms
  std::vector<memory::primitive_desc> botPDs;
  std::vector<std::shared_ptr<memory::desc>> prvMDs;
  std::vector<primitive::at> botMems;
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize())
      << "Assert batchsize of input layers are equal";
    CHECK(iw_[i] == ow_[i] && ih_[i] == oh_[i]);
    botDims_[i] = {bs_, ic_[i], ih_[i], iw_[i]};
    botFmt_[i] = memory::format::nchw;
    // 1. create bottom buffer and init user
    dataBottoms_[i].reset(new MkldnnBuffer());
    real *botData = getInputValue(i)->getData();
    const std::shared_ptr<memory::desc>& prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      dataBottoms_[i]->resetUser(botData, *prvMD, eg);
      bool isNC = dataBottoms_[i]->getUserFmt() == memory::format::nc;
      if (isNC) {
        CHECK(ih_[i] == iw_[i] && ih_[i] == 1)
          << "iw, ih must be 1 with nc input";
        // do not support nc input, so change to nchw
        dataBottoms_[i]->resetUser(botData, botDims_[i], botFmt_[i], eg);
        VLOG(4) << "use nchw data fmt";
      } else {
        VLOG(4) << "use prev data fmt: "
          << DNN_FMTS[dataBottoms_[i]->getUserFmt()];
      }
      prvMDs.push_back(prvMD);
    } else {
      dataBottoms_[i]->initUser(botData, botDims_[i], botFmt_[i], eg);
    }
    botPDs.push_back(dataBottoms_[i]->getUserPD());

    // 2. init bot internals
    dataBottoms_[i]->initCvt(dataBottoms_[i]->getUserPD(), dnnCvtNoNeed);
    botMems.push_back(*(dataBottoms_[i]->getIntlMem()));
  }
  // inputs size should equal and all format should be the same
  CHECK(prvMDs.size() == 0 || prvMDs.size() == inputLayers_.size())
    << "intl input size does not match: "
    << prvMDs.size() << " vs " << inputLayers_.size();
  for (size_t i = 1; i < prvMDs.size(); ++i) {
    CHECK_EQ(MkldnnBuffer::getMDFmt(*prvMDs[i-1]),
      MkldnnBuffer::getMDFmt(*prvMDs[i]))
      << "all input formats should be the same";
  }
  
  // 3. create fwd PD  
  std::shared_ptr<concat::primitive_desc> fwdPD;
  fwdPD.reset(new concat::primitive_desc(
    MkldnnBuffer::getMD(topDims_), axis_, botPDs));
  // reset top user using best internal fmt if next is also dnn
  if (nextIsDnn_) {
    dataTop_->resetUser(topData, fwdPD->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
    VLOG(4) << "set next data fmt: " << DNN_FMTS[dataTop_->getUserFmt()];
  }

  // 4. init top cvt
  dataTop_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);

  // 5. create fwd handle
  fwd_.reset(new concat(*fwdPD, botMems, *(dataTop_->getIntlMem())));

  // TODO(TJ): remove when dataBot vector done
  VLOG(1) << "data format flow --- "
    << DNN_FMTS[dataBottoms_[0]->getUserFmt()] << " >>> ("
    << DNN_FMTS[dataBottoms_[0]->getIntlFmt()] << " >>> "
    << DNN_FMTS[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FMTS[dataTop_->getUserFmt()];
}

void MkldnnConcatLayer::resetDnnBwd() {
}

void MkldnnConcatLayer::submitDnnFwd(PassType passType) {
  real *topData = getOutputValue()->getData();
  std::vector<primitive> pipeline;
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    real *botData = getPrev(i)->getOutputValue()->getData();
    dataBottoms_[i]->submitCvt(pipeline, botData);
  }
  pipeline.push_back(*fwd_);
  dataTop_->submitCvt(pipeline, topData);

  stream(stream::kind::eager).submit(pipeline).wait();

  forwardActivation();
}

void MkldnnConcatLayer::submitDnnBwd(const UpdateCallback &callback) {
  (void)callback;
  backwardActivation();

  const MatrixPtr& out = getOutputGrad();
  const std::shared_ptr<mkldnn::memory::desc>& prvMD = getTopDiffMD();
  int offset = 0;
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    const MatrixPtr& in = getInputGrad(i);
    if (NULL == in)
      continue;
    // TODO(TJ): use mkldnn with both addsize ==0 and >0
    if (prvMD && MkldnnBuffer::getMDFmt(*prvMD) != topFmt_) {
      LOG(FATAL) << "not implemented with internal format";
    } else {
      LOG(WARNING) << "not speedup with MKLDNN yet";
      size_t inSize = getInputValue(i)->getWidth();
      if (in) {
        in->addAtOffset(*out, offset);
      }
      offset += inSize;
    }
    
  }
}

}  // namespace paddle
