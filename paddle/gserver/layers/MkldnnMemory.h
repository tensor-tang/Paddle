/* Copyright (c) 2017 */

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
using mem = mkldnn::memory;

protected:
  /// user and internal layout
  std::shared_ptr<mem> pUser_;
  std::shared_ptr<mem> pIntl_;

  /// data type: mkldnn only support f32 and s32
  /// here only support f32 yet
  mem::data_type tp_;

  /// conversion handle and type
  std::shared_ptr<mkldnn::primitive> pCvt_;

  int  cvtType_;
  bool hasCvted_;  // to avoid re-cvt

public:
  explicit MkldnnBuffer(
    mem::data_type type = mem::data_type::f32) :
    pUser_(nullptr),
    pIntl_(nullptr),
    pCvt_(nullptr),
    cvtType_(dnnCvtNone),
    hasCvted_(false) {
    tp_ = type;
    if (tp_ != mem::data_type::f32)
      LOG(FATAL) << "only support float 32 so far";
  }

  ~MkldnnBuffer() {}

  void initUser(void *pd,
    const mem::dims& dm, const mem::format& fmt, const mkldnn::engine& eg) {
    initUser(pd, mem::desc(dm, tp_, fmt), eg);
  }

  void initUser(void *pd, const mem::desc& md, const mkldnn::engine& eg) {
    if (pd == NULL) {
      pUser_.reset(new mem(mem::primitive_desc(md, eg)));
    } else {
      CHECK_EQ(int(md.data.data_type), int(tp_))
        << "input data type does not match: "
        << md.data.data_type << " vs " << tp_;
      pUser_.reset(new mem(mem::primitive_desc(md, eg), pd));
    }
  }

  void initUser(void *pdata, mem::primitive_desc pd) {
    if (pdata == NULL) {
      pUser_.reset(new mem(pd));
    } else {
      CHECK_EQ(int(pd.desc().data.data_type), int(tp_))
        << "input data type does not match: "
        << pd.desc().data.data_type << " vs " << tp_;
      pUser_.reset(new mem(pd, pdata));
    }
  }

  void resetUser(void *pd,
    const mem::dims& dm, const mem::format& fmt, const mkldnn::engine& eg) {
    initUser(pd, mem::desc({dm}, tp_, fmt), eg);
  }

  void resetUser(
    void *pd, const mem::desc& md, const mkldnn::engine& eg) {
    CHECK_EQ(int(md.data.data_type), int(tp_))
      << "input data type does not match: "
      << md.data.data_type << " vs " << tp_;
    pUser_.reset(new mem(mem::primitive_desc(md, eg), pd));
  }

  void resetUser(void *pdata, mem::primitive_desc pd) {
    CHECK_EQ(int(pd.desc().data.data_type), int(tp_))
      << "input data type does not match: "
      << pd.desc().data.data_type << " vs " << tp_;
    pUser_.reset(new mem(pd, pdata));
  }

  /// functions for getting infos
  size_t getSize(size_t sz) {
    size_t unit;
    switch (tp_) {
      case mem::data_type::f32:
        unit = sizeof(float);
        break;
      case mem::data_type::s32:
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

  const std::shared_ptr<mem>& getIntlMem() {
     return this->pIntl_;
  }

//  std::shared_ptr<memory> getUserMem() {
//     return this->pUser_;
//  }

  // user primitive desc
  mem::primitive_desc getUserPD() {
    CHECK(pUser_) << "haven't init user layout";
    return pUser_->get_primitive_desc();
  }

  // internal primitive desc
  mem::primitive_desc getIntlPD() {
    CHECK(pIntl_) << "haven't init internal layout, call initUser then initCvt";
    return pIntl_->get_primitive_desc();
  }

  // get memory desc
  mem::desc getUserMD() {
    CHECK(pUser_) << "haven't init user layout";
    return pUser_->get_primitive_desc().desc();
  }

  static mem::desc getMD(const mem::dims& dm,
    const mem::format& fmt = mem::format::any,
    const mem::data_type &tp = mem::data_type::f32) {
    return mem::desc({dm}, tp, fmt);
  }

  // get format from MD
  static int getMDFmt(const mem::desc& md) {
    return md.data.format;
  }

  // get format from PD
  static int getPDFmt(mem::primitive_desc pd) {
    return pd.desc().data.format;
  }

  // get user memory format
  int getUserFmt() {
    CHECK(pUser_) << "haven't init user layout";
    return getMDFmt(getUserMD());
  }

  static mem::dims getMDDims(const mem::desc& md) {
    const int* dm = md.data.dims;
    int ndims = md.data.ndims;
    std::vector<int> v(dm, dm + ndims);
    return v;
  }

  mem::dims getUserDims() {
    return getMDDims(getUserMD());
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

  mem::desc getIntlMD() {
    CHECK(pIntl_) << "haven't init internal layout, call initUser then initCvt";
    return pIntl_->get_primitive_desc().desc();
  }

  void clearCvtFlag() {
    hasCvted_ = false;
  }

  // init conversion(reorder), will create internal buffer if needed
  // return true if need cvt.
  bool initCvt(mem::primitive_desc intlPD, int cvtType) {
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
      this->pIntl_.reset(new mem(intlPD));
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
