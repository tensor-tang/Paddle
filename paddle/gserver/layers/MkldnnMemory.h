/* Copyright (c) 2016 */

#pragma once

#include <vector>
#include "paddle/utils/Logging.h"
#include "mkldnn.hpp"


namespace paddle {
/**
 * @brief A MKLDNN memory buffer class
 *
 * 
 */
 typedef enum {
    dnnCvtNone             = 0,
    dnnCvtUser2Intl        = 1,
    dnnCvtIntl2User        = 2,
    dnnCvtNoNeed           = 3,
    dnnCvtNumber           = 4
} dnnCvtType_t;

class MkldnnBuffer {
protected:
  /// user and internal layout
  std::shared_ptr<mkldnn::memory> pUser_;
  std::shared_ptr<mkldnn::memory> pIntl_;

  /// data type: mkldnn only support f32 and s32
  /// here only support f32 yet
  mkldnn::memory::data_type tp_;

  /// conversion handle and type
  std::shared_ptr<mkldnn::primitive> pCvt_;

  int  cvtType_;
  bool hasCvted_;  // to avoid re-cvt

public:
  explicit MkldnnBuffer(
    mkldnn::memory::data_type type = mkldnn::memory::data_type::f32) :
    pUser_(nullptr),
    pIntl_(nullptr),
    pCvt_(nullptr),
    cvtType_(dnnCvtNone),
    hasCvted_(false) {
    tp_ = type;
    if (tp_ != mkldnn::memory::data_type::f32)
      LOG(FATAL) << "only support float 32 so far";
  }

  ~MkldnnBuffer() {}

  void initUser(void *pd,
    mkldnn::memory::dims dm, mkldnn::memory::format fmt, mkldnn::engine eg) {
    initUser(pd, mkldnn::memory::desc({dm}, tp_, fmt), eg);
  }

  void initUser(void *pd, mkldnn::memory::desc md, mkldnn::engine eg) {
    CHECK_EQ(md.data.data_type, tp_) << "input data type does not match: "
      << md.data.data_type << " vs " << tp_;
    pUser_.reset(
      new mkldnn::memory(mkldnn::memory::primitive_desc(md, eg), pd));
  }

  void initUser(void *pdata, mkldnn::memory::primitive_desc pd) {
    CHECK_EQ(pd.desc().data.data_type, tp_)
      << "input data type does not match: "
      << pd.desc().data.data_type << " vs " << tp_;
    pUser_.reset(new mkldnn::memory(pd, pdata));
  }

  void resetUser(void *pd,
    mkldnn::memory::dims dm, mkldnn::memory::format fmt, mkldnn::engine eg) {
    initUser(pd, mkldnn::memory::desc({dm}, tp_, fmt), eg);
  }

  void resetUser(void *pd, mkldnn::memory::desc md, mkldnn::engine eg) {
    CHECK_EQ(md.data.data_type, tp_) << "input data type does not match: "
      << md.data.data_type << " vs " << tp_;
    pUser_.reset(
      new mkldnn::memory(mkldnn::memory::primitive_desc(md, eg), pd));
  }

  void resetUser(void *pdata, mkldnn::memory::primitive_desc pd) {
    CHECK_EQ(pd.desc().data.data_type, tp_)
      << "input data type does not match: "
      << pd.desc().data.data_type << " vs " << tp_;
    pUser_.reset(new mkldnn::memory(pd, pdata));
  }

  /// functions for getting infos
  size_t getSize(size_t sz) {
    size_t unit;
    switch (tp_) {
      case mkldnn::memory::data_type::f32:
        unit = sizeof(float);
        break;
      case mkldnn::memory::data_type::s32:
        unit = sizeof(signed int);
        break;
      default:
        LOG(ERROR) << "Error data type";
        return 0;
    }
    return sz / unit;
  }

  /// it's the element size not memory size
  size_t getIntlSize() {
    CHECK(pIntl_) << "haven't init internal layout, call initUser then initCvt";
    return getSize(pIntl_->get_primitive_desc().get_size());
  }

  /// it's the element size not memory size
  size_t getUserSize() {
    CHECK(pUser_) << "haven't init user layout";
    return getSize(pUser_->get_primitive_desc().get_size());
  }

  const std::shared_ptr<mkldnn::memory>& getIntlMem() {
     return this->pIntl_;
  }

//  std::shared_ptr<memory> getUserMem() {
//     return this->pUser_;
//  }

  // user primitive desc
  mkldnn::memory::primitive_desc getUserPD() {
    CHECK(pUser_) << "haven't init user layout";
    return pUser_->get_primitive_desc();
  }

  // internal primitive desc
  mkldnn::memory::primitive_desc getIntlPD() {
    CHECK(pIntl_) << "haven't init internal layout, call initUser then initCvt";
    return pIntl_->get_primitive_desc();
  }

  // get memory desc
  mkldnn::memory::desc getUserMD() {
    CHECK(pUser_) << "haven't init user layout";
    return pUser_->get_primitive_desc().desc();
  }

  static mkldnn::memory::desc getMD(mkldnn::memory::dims & dm,
    mkldnn::memory::format fmt = mkldnn::memory::format::any,
    mkldnn::memory::data_type tp = mkldnn::memory::data_type::f32) {
    return mkldnn::memory::desc({dm}, tp, fmt);
  }

  // get format from MD
  static int getMDFmt(const mkldnn::memory::desc& md) {
    return md.data.format;
  }

  // get format from PD
  static int getPDFmt(mkldnn::memory::primitive_desc pd) {
    return pd.desc().data.format;
  }

  // get user memory format
  int getUserFmt() {
    CHECK(pUser_) << "haven't init user layout";
    return getMDFmt(getUserMD());
  }

  // get user memory format
  int getIntlFmt() {
    CHECK(pIntl_) << "haven't init internal layout, call initUser then initCvt";
    return getMDFmt(getIntlMD());
  }

  // get internal data handle
  void* getIntlData() {
    return pIntl_->get_data_handle();
  }

  mkldnn::memory::desc getIntlMD() {
    CHECK(pIntl_) << "haven't init internal layout, call initUser then initCvt";
    return pIntl_->get_primitive_desc().desc();
  }

  void clearCvtFlag() {
    hasCvted_ = false;
  }

  // init conversion(reorder), will create internal buffer if needed
  // return true if need cvt.
  bool initCvt(mkldnn::memory::primitive_desc intlPD, int cvtType) {
    CHECK(cvtType == dnnCvtUser2Intl || cvtType == dnnCvtIntl2User
      || cvtType == dnnCvtNoNeed) << "please specify one type of conversion";
    CHECK(pUser_)
      << "call initUser before init internal layout and conversion";
    CHECK(nullptr == pIntl_)
      << "internal memory should be empty before initCvt";
    pIntl_ = pUser_;
    cvtType_ = cvtType;
    clearCvtFlag();
    if (cvtType == dnnCvtNoNeed || intlPD == getUserPD()) {
      cvtType_ = dnnCvtNoNeed;
      return false;
    } else {
      // allocate internal src memory from user
      this->pIntl_.reset(new mkldnn::memory(intlPD));
      // create a reorder
      if (cvtType == dnnCvtUser2Intl) {
        this->pCvt_.reset(new mkldnn::reorder(*pUser_, *pIntl_));
      } else {
        this->pCvt_.reset(new mkldnn::reorder(*pIntl_, *pUser_));
      }
      return true;
    }
  }

  // init with dnnCvtNoNeed
  bool initCvt() {
    CHECK(pUser_)
      << "call initUser before init internal layout and conversion";
    CHECK(nullptr == pIntl_)
      << "internal memory should be empty before initCvt";
    pIntl_ = pUser_;
    cvtType_ = dnnCvtNoNeed;
    clearCvtFlag();
    return false;
  }

  bool needCvt() {
    CHECK(cvtType_) << "init conversion firstly";
    if (cvtType_ == dnnCvtNoNeed) {
      return false;
    } else {
      return nullptr == pCvt_ ? false : true;
    }
  }

  /**
   * submit reorder conversion.
   */
  void submitCvt(std::vector<mkldnn::primitive> &net,
                      void* userData = NULL) {
    CHECK(cvtType_) << "init conversion firstly";
    // set user data handle, whether if need reorder or not
    if (userData) {
      if (userData != pUser_->get_data_handle()) {
        pUser_->set_data_handle(userData);
        // data changed, so donot care hasCvted_
      } else {  // user data do not change
        if (hasCvted_)  return;
      }
    } else {  // user data do not change
      if (hasCvted_)  return;
    }
    if (cvtType_ == dnnCvtNoNeed)
      return;
    CHECK(pCvt_) << "init conversion firstly";
    net.push_back(*pCvt_);
    hasCvted_ = true;
  }
};

typedef std::shared_ptr<MkldnnBuffer> MkldnnBufferPtr;

}  // namespace paddle
