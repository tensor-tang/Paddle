/* 
 */

#pragma once

#include <vector>
#include "paddle/utils/Logging.h"
#include "mkldnn.hpp"

using namespace mkldnn;

namespace paddle {

/**
 * @brief A DNN memory buffer class
 *
 * 
 */
 typedef enum {
    dnnCvtNone             = 0,
    dnnCvtUser2Internal    = 1,
    dnnCvtInternal2User    = 2,
    dnnCvtNoNeed           = 3,
    dnnCvtNumber           = 4
} dnnCvtType_t;

class DnnBuffer {

protected:
  /// dims of user memory
  memory::dims dims_;
  
  /// user and internal layout
  std::shared_ptr<memory> pUser_;
  std::shared_ptr<memory> pIntl_;
  
  /// conversion handle and type
  std::shared_ptr<primitive> pCvt_;

  int  type_;
  bool hasCvted_; // to avoid re-cvt
  
public:
  explicit DnnBuffer(memory::dims dm) :
    dims_(dm),
    pUser_(NULL),
    pIntl_(NULL),
    pCvt_(NULL),
    type_(dnnCvtNone),
    hasCvted_(false) {}

  ~DnnBuffer() {}

  const memory::dims& getDefaultDims() {
    return dims_;
  }
  
  void initUser(void *pd, memory::dims dm, memory::format fmt, engine eg,
                   memory::data_type tp = memory::data_type::f32) {
    initUser(pd, memory::desc({dm}, tp, fmt), eg);
  }
  
  void initUser(void *pd, memory::desc md, engine eg) {
    pUser_.reset(new memory(memory::primitive_desc(md, eg), pd));
  }

  void initUser(void *pdata, memory::primitive_desc pd) {
    pUser_.reset(new memory(pd, pdata));
  }

  memory::desc getMDAny(memory::data_type tp = memory::data_type::f32) {
    return memory::desc({dims_}, tp, memory::format::any);
  }

  std::shared_ptr<memory> getIntlMem() {
     return this->pIntl_;
  }

//  std::shared_ptr<memory> getUserMem() {
//     return this->pUser_;
//  }

  // user primitive desc
  memory::primitive_desc getUserPD() {
    CHECK(pUser_) << "haven't init user layout";
    return pUser_->get_primitive_desc();
  }

  // internal primitive desc
  memory::primitive_desc getIntlPD() {
    CHECK(pIntl_) << "haven't init internal layout, call initCvt firstly";
    return pIntl_->get_primitive_desc();
  }

  // get memory desc
  memory::desc getUserMD() {
    CHECK(pUser_) << "haven't init user layout";
    return pUser_->get_primitive_desc().desc();
  }

  // get user memory format
  int getUserFmt() {
    CHECK(pUser_) << "haven't init user layout";
    return getUserMD().data.format;
  }

  // get user memory format
  int getIntlFmt() {
    CHECK(pIntl_) << "haven't init user layout";
    return getIntlMD().data.format;
  }

  memory::desc getIntlMD() {
    CHECK(pIntl_) << "haven't init internal layout, call initCvt firstly";
    return pIntl_->get_primitive_desc().desc();
  }
  
  void clearCvtFlag() {
    hasCvted_ = false;
  }
  
  // init conversion(reorder) return true if need cvt.
  bool initCvt(memory::primitive_desc intlPD, int cvtType) {
    CHECK(cvtType == dnnCvtUser2Internal || cvtType == dnnCvtInternal2User) <<
      "please specify one type of conversion";

    CHECK(pUser_) << "need create user layout before init conversion, call initUser";
    CHECK(pIntl_ == NULL) << "internal memory should be empty before initCvt";
    pIntl_ = pUser_;
    type_ = cvtType;
    clearCvtFlag();
    if (intlPD != getUserPD()) {
      // allocate internal src memory from user
      this->pIntl_.reset(new mkldnn::memory(intlPD));

      // create a reorder 
      if (cvtType == dnnCvtUser2Internal) {
        this->pCvt_.reset(new reorder(*pUser_, *pIntl_));
      } else {
        this->pCvt_.reset(new reorder(*pIntl_, *pUser_));
      }
      return true;
    } else {
      type_ = dnnCvtNoNeed;
      return false;
    }
  }
  
  bool needCvt() {
    CHECK(type_) << "init conversion firstly";
    if (type_ == dnnCvtNoNeed) {
      return false;
    } else {
      return pCvt_==NULL ? false : true;
    }
  }

  /**
   * submit reorder conversion.
   */
  void submitCvt(std::vector<primitive> &net, void* userData = NULL) {
    CHECK(type_) << "init conversion firstly";
    // set user data handle, whether if need reorder or not
    if (userData) {
      if (userData != pUser_->get_data_handle()) {
        pUser_->set_data_handle(userData);
        //data changed, so donot care hasCvted_
      } else { // user data do not change
        if (hasCvted_)  return;
      }
    } else { // user data do not change
      if (hasCvted_)  return;
    }
    if (type_ == dnnCvtNoNeed)
      return;
    CHECK(pCvt_) << "init conversion firstly";
    net.push_back(*pCvt_);
    hasCvted_ = true;
  }
};

typedef std::shared_ptr<DnnBuffer> DnnBufferPtr;

}  // namespace paddle
