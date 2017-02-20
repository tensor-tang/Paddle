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

  /// conversion handle and type
  std::shared_ptr<mkldnn::primitive> pCvt_;

  int  type_;
  bool hasCvted_;  // to avoid re-cvt

public:
  explicit MkldnnBuffer(
    mkldnn::memory::data_type tp = mkldnn::memory::data_type::f32) :
    pUser_(nullptr),
    pIntl_(nullptr),
    pCvt_(nullptr),
    type_(dnnCvtNone),
    hasCvted_(false) {
    if (tp != mkldnn::memory::data_type::f32)
      LOG(FATAL) << "only support float 32 so far";
  }

  ~MkldnnBuffer() {}

  void initUser(void *pd,
    mkldnn::memory::dims dm, mkldnn::memory::format fmt, mkldnn::engine eg,
    mkldnn::memory::data_type tp = mkldnn::memory::data_type::f32) {
    initUser(pd, mkldnn::memory::desc({dm}, tp, fmt), eg);
  }

  void initUser(void *pd, mkldnn::memory::desc md, mkldnn::engine eg) {
    pUser_.reset(
      new mkldnn::memory(mkldnn::memory::primitive_desc(md, eg), pd));
  }

  void initUser(void *pdata, mkldnn::memory::primitive_desc pd) {
    pUser_.reset(new mkldnn::memory(pd, pdata));
  }

  void resetUser(void *pd,
    mkldnn::memory::dims dm, mkldnn::memory::format fmt, mkldnn::engine eg,
    mkldnn::memory::data_type tp = mkldnn::memory::data_type::f32) {
    initUser(pd, mkldnn::memory::desc({dm}, tp, fmt), eg);
  }

  void resetUser(void *pd, mkldnn::memory::desc md, mkldnn::engine eg) {
    pUser_.reset(
      new mkldnn::memory(mkldnn::memory::primitive_desc(md, eg), pd));
  }

  void resetUser(void *pdata, mkldnn::memory::primitive_desc pd) {
    pUser_.reset(new mkldnn::memory(pd, pdata));
  }

  std::shared_ptr<mkldnn::memory> getIntlMem() {
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
    CHECK(pIntl_) << "haven't init internal layout, call initCvt firstly";
    return pIntl_->get_primitive_desc();
  }

  // get memory desc
  mkldnn::memory::desc getUserMD() {
    CHECK(pUser_) << "haven't init user layout";
    return pUser_->get_primitive_desc().desc();
  }

  // get user memory format
  int getUserFmt() {
    CHECK(pUser_) << "haven't init user layout";
    return getUserMD().data.format;
  }

  // get format from PD
  int getFmt(mkldnn::memory::primitive_desc &pd) {
    return pd.desc().data.format;
  }

  // get user memory format
  int getIntlFmt() {
    CHECK(pIntl_) << "haven't init user layout";
    return getIntlMD().data.format;
  }

  mkldnn::memory::desc getIntlMD() {
    CHECK(pIntl_) << "haven't init internal layout, call initCvt firstly";
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
    CHECK(pIntl_ == NULL)
      << "internal memory should be empty before initCvt";
    pIntl_ = pUser_;
    type_ = cvtType;
    clearCvtFlag();
    if (cvtType == dnnCvtNoNeed || intlPD == getUserPD()) {
      type_ = dnnCvtNoNeed;
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
    CHECK(pIntl_ == NULL)
      << "internal memory should be empty before initCvt";
    pIntl_ = pUser_;
    type_ = dnnCvtNoNeed;
    clearCvtFlag();
    return false;
  }

  bool needCvt() {
    CHECK(type_) << "init conversion firstly";
    if (type_ == dnnCvtNoNeed) {
      return false;
    } else {
      return pCvt_ == NULL ? false : true;
    }
  }

  /**
   * submit reorder conversion.
   */
  void submitCvt(std::vector<mkldnn::primitive> &net,
                      void* userData = NULL) {
    CHECK(type_) << "init conversion firstly";
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
    if (type_ == dnnCvtNoNeed)
      return;
    CHECK(pCvt_) << "init conversion firstly";
    net.push_back(*pCvt_);
    hasCvted_ = true;
  }
};

typedef std::shared_ptr<MkldnnBuffer> MkldnnBufferPtr;

}  // namespace paddle
