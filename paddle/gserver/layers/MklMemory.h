/* 
 */

#pragma once

#include <vector>
#include "mkl_dnn_types.h"
#include "paddle/utils/Logging.h"

namespace paddle {

#ifndef CHECK_ST
#define CHECK_ST(f, st) do { \
      (st) = (f); \
      if ((st) != E_SUCCESS) { \
          LOG(FATAL) <<"[" << __FILE__ \
            << "," << __LINE__ << "] " \
            << "err code: " << st; \
      } \
    }while (0)
#endif

/**
 * @brief A MKL DNN memory buffer class
 *
 * 
 */
 typedef enum {
    dnnCvtNone             = 0,
    dnnCvtUser2Internal    = 1,
    dnnCvtInternal2User    = 2,
    dnnCvtNumber           = 3
} dnnCvtType_t;

class MklBuffer {

protected:
  /// user and internal layout
  dnnLayout_t user_;
  dnnLayout_t intl_;
  /// conversion handle and type
  dnnPrimitive_t cvt_;
  int   type_;
  /// internal buffer data
  void *data_;
  
public:
  explicit MklBuffer() :
    user_(NULL),
    intl_(NULL),
    cvt_(NULL),
    type_(dnnCvtNone),
    data_(NULL) {}

  ~MklBuffer() {
    // release all dnn
    if (user_) {
      dnnLayoutDelete(user_);
    }
    if (intl_) {
      dnnLayoutDelete(intl_);
    }
    if (data_) {
      dnnReleaseBuffer(data_);
    }
    if (cvt_) {
      dnnDelete(cvt_);
    }
  }
  
  dnnError_t createUser(
    const size_t dimension, const size_t size[], const size_t strides[]) {
    return dnnLayoutCreate(&user_, dimension, size, strides);;
  }
  
  dnnError_t createIntl(
    const dnnPrimitive_t primitive, dnnResourceType_t type) {
    return dnnLayoutCreateFromPrimitive(&intl_, primitive, type);
  }

  /**
   * .
   */
  dnnError_t initConversion(int cvtType = dnnCvtUser2Internal) {
    CHECK(cvtType == dnnCvtUser2Internal || cvtType == dnnCvtInternal2User) <<
      "please specify one type of conversion";
    CHECK(user_) << "need create user layout before init conversion";
    CHECK(intl_) << "need create internal layout before init conversion";
    type_ = cvtType;
    int st;
    if (dnnLayoutCompare(user_, intl_)) {
      LOG(INFO) << "user and internal layout equals";
      type_ = dnnCvtNone;
      return E_SUCCESS;
    }
    /// then need conversion now
    if (data_) {
      CHECK_ST(dnnReleaseBuffer(data_), st);
      data_ = NULL;
    }
    if (cvt_) {
      CHECK_ST(dnnDelete(cvt_), st);
      cvt_ = NULL;
    }
    if (cvtType == dnnCvtUser2Internal) {
      CHECK_ST(dnnConversionCreate(&cvt_, user_, intl_), st);
    } else {
      CHECK_ST(dnnConversionCreate(&cvt_, intl_, user_), st);
    }
    // internal buffer
    return dnnAllocateBuffer(&data_, intl_);
  }
  
  /**
   * .
   */
  bool needConversion() {
    if (cvt_ != NULL
        && (type_ == dnnCvtUser2Internal || type_ == dnnCvtInternal2User)) {
      return true;
    } else {
      return false;
    }
  }
  
  /**
   * execute conversion type.
   */
  dnnError_t executeConversion(void *userData) {
    CHECK(cvt_) << "init conversion firstly";
    switch(type_) {
      case dnnCvtUser2Internal:
        return dnnConversionExecute(cvt_, userData, data_);
      case dnnCvtInternal2User:
        return dnnConversionExecute(cvt_, data_, userData);
      default:
        LOG(INFO) << "no supported conversion type!";
        return E_UNIMPLEMENTED;
    }
  }
  
  /**
   * get internal data.
   */
  void* getData() {
    return data_;
  }
};

typedef std::shared_ptr<MklBuffer> MklBufferPtr;

}  // namespace paddle
