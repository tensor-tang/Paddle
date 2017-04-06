/* Copyright (c) 2017 */

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
  int i = 0;
  for (auto &inputConfig : config_.inputs()) {
    if (inputConfig.has_image_conf()) {
      const ImageConfig &conf = inputConfig.image_conf();
      iw_[i] = conf.img_size();
      ih_[i] = conf.img_size();
    } else {
      iw_[i] = 0;
      ih_[i] = 0;
    }
    ic_[i] = 0;
    ow_[i] = 0;
    oh_[i] = 0;
    i++;
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

void MkldnnConcatLayer::resetDnnFwd() {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();

  // create top buffer and init user, only have one output
  topData_.reset(new MkldnnBuffer());
  real *topDataData = getOutputValue()->getData();
  topFmt_ = memory::format::nchw;
  topDims_ = {bs_, oc_, oh_[0], ow_[0]};
  topData_->initUser(topDataData, topDims_, topFmt_, eg);

  // prepare bottoms
  std::vector<memory::primitive_desc> botPDs;
  std::vector<std::shared_ptr<memory::desc>> prvMDs;
  std::vector<primitive::at> botMems;
  CHECK_EQ(botDatas_.size(), inputLayers_.size());
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize())
      << "Assert batchsize of input layers are equal";
    CHECK(iw_[i] == ow_[i] && ih_[i] == oh_[i]);
    botDims_[i] = {bs_, ic_[i], ih_[i], iw_[i]};
    botFmt_[i] = memory::format::nchw;
    // 1. create bottom buffer and init user
    botDatas_[i].reset(new MkldnnBuffer());
    real *botDataData = getInputValue(i)->getData();
    const std::shared_ptr<memory::desc>& prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      botDatas_[i]->resetUser(botDataData, *prvMD, eg);
      bool isNC = botDatas_[i]->getUserFmt() == memory::format::nc;
      if (isNC) {
        CHECK(ih_[i] == iw_[i] && ih_[i] == 1)
          << "iw, ih must be 1 with nc input";
        // do not support nc input, so change to nchw
        botDatas_[i]->resetUser(botDataData, botDims_[i], botFmt_[i], eg);
        VLOG(4) << "use nchw data fmt";
      } else {
        VLOG(4) << "use prev data fmt: "
          << DNN_FMTS[botDatas_[i]->getUserFmt()];
      }
      prvMDs.push_back(prvMD);
    } else {
      botDatas_[i]->initUser(botDataData, botDims_[i], botFmt_[i], eg);
    }
    botPDs.push_back(botDatas_[i]->getUserPD());

    // 2. init bot internals
    botDatas_[i]->initCvt(botDatas_[i]->getUserPD(), dnnCvtNoNeed);
    botMems.push_back(*(botDatas_[i]->getIntlMem()));
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
    topData_->resetUser(topDataData, fwdPD->dst_primitive_desc());
    setTopDataMD(topData_->getUserMD());
    VLOG(4) << "set next data fmt: " << DNN_FMTS[topData_->getUserFmt()];
  }

  // 4. init top cvt
  topData_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);

  // 5. create fwd handle
  fwd_.reset(new concat(*fwdPD, botMems, *(topData_->getIntlMem())));
}

void MkldnnConcatLayer::resetDnnBwd() {
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  // 1. create top buffer and init, only have one output buffer
  topDiff_.reset(new MkldnnBuffer());
  real *topDiffData = getDnnOutputGrad()->getData();
  topDiff_->initUser(topDiffData, topDims_, topFmt_, eg);
  // 2. use internal top diff if use dnn input
  const std::shared_ptr<mkldnn::memory::desc>& prvMD = getTopDiffMD();
  if (prvMD) {
    topDiff_->resetUser(topDiffData, *prvMD, eg);
    bool isNC = topDiff_->getUserFmt() == memory::format::nc;
    if (isNC) {
      CHECK(oh_[0] == ow_[0] && oh_[0] == 1)
        << "ow, oh must be 1 with nc input";
      // do not support nc as input, so change to nchw
      memory::format fmt = memory::format::nchw;
      topDiff_->resetUser(topDiffData, topDims_, fmt, eg);
      VLOG(4) << "use nchw diff fmt";
    } else {
      VLOG(4) << "use prev diff fmt:" << DNN_FMTS[topDiff_->getUserFmt()];
    }
  }
  topDiff_->initCvt(topDiff_->getUserPD(), dnnCvtNoNeed);
  // 3. prepare bottom diffs
  size_t sz = 0;
  memory::dims offsets = {0, 0, 0, 0};
  bwds_.resize(inputLayers_.size(), nullptr);
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize()) << "batchsize should equal";
    if (nullptr == getPrev(i)->getOutputGrad()) {
      continue;  // data layer has not diff
    }
    // 1. create bottom buffer and init
    real* botDiffData = getDnnInputGrad(i)->getData();
    botDiffs_[i].reset(new MkldnnBuffer());
    // directly keep the format as botdata
    botDiffs_[i]->initUser(botDiffData, botDatas_[i]->getUserPD());
    botDiffs_[i]->initCvt(botDiffs_[i]->getUserPD(), dnnCvtNoNeed);
    if (prevIsDnn_[i]) {
      getPrev(i)->setTopDiffMD(this->getName(), botDiffs_[i]->getIntlMD());
      VLOG(4) << "set bot diff fmt: "
        << DNN_FMTS[botDiffs_[i]->getIntlFmt()];
    }
    // 2. create bwd handle, call reorder function to backward
    auto topPD = view::primitive_desc(
      topDiff_->getIntlPD(), botDims_[i], offsets);
    auto bwdPD = reorder::primitive_desc(
      topPD.dst_primitive_desc(), botDiffs_[i]->getIntlPD());
    bwds_[i].reset(new reorder(
      bwdPD, *(topDiff_->getIntlMem()), *(botDiffs_[i]->getIntlMem())));
    offsets[axis_] += botDims_[i][axis_];
    sz += botDiffs_[i]->getUserSize();
  }
  CHECK_EQ(sz, getDnnOutputGrad()->getElementCnt());
}

void MkldnnConcatLayer::submitDnnFwd() {
  real *topDataData = getOutputValue()->getData();
  std::vector<primitive> pipeline;
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    real *botDataData = getPrev(i)->getOutputValue()->getData();
    botDatas_[i]->submitCvt(pipeline, botDataData);
  }
  pipeline.push_back(*fwd_);
  topData_->submitCvt(pipeline, topDataData);

  stream(stream::kind::eager).submit(pipeline).wait();

  forwardActivation();
}

void MkldnnConcatLayer::submitDnnBwd(const UpdateCallback &callback) {
  (void)callback;
  backwardActivation();

  CHECK_EQ(getDnnOutputGrad()->getData(), topDiff_->getIntlData());

  // actually topdiff and botdiffs no need submit cvt
  // since the format are the same
  // so only need do bwd
  std::vector<primitive> pipeline;
  for (size_t i = 0; i < bwds_.size(); ++i) {
    pipeline.push_back(*bwds_[i]);
  }
  stream(stream::kind::eager).submit(pipeline).wait();
}

}  // namespace paddle
