/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <stdlib.h>  // for malloc and free
#include <string>
#include "glog/logging.h"
#include "paddle/fluid/operators/jit/gen/jitcode.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

class MatMulJitCode : public JitCode {
 public:
  explicit MatMulJitCode(const matmul_attr_t& attr,
                         size_t code_size = 256 * 1024,
                         void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr), m_(attr.m), n_(attr.n), k_(attr.k) {
    PADDLE_ENFORCE_EQ(m_, 1, "Only support m==1 yet");
    if (platform::MayIUse(platform::avx512f)) {
      block_ = ZMM_FLOAT_BLOCK;
    } else {
      block_ = YMM_FLOAT_BLOCK;
    }
    aligned_n_ = n_ % block_ == 0 ? n_ : (n_ / block_ + 1) * block_;
    size_t sz = aligned_n_ * k_ * sizeof(float);
    PADDLE_ENFORCE_EQ(posix_memalign(&wgt_, 64, sz), 0, "Alloc %ld error!", sz);
    this->genCode();
  }

  virtual const char* name() const {
    std::string base = "MatMulJitCode";
    base = base + "_M" + std::to_string(m_) + "_N" + std::to_string(n_) + "_K" +
           std::to_string(k_);
    return base.c_str();
  }
  void genCode() override;

  virtual ~MatMulJitCode() { free(wgt_); }

 protected:
  template <typename JMM>
  void pool_height(int w_offset, int block, int max_num_regs) {}

 private:
  int m_, n_, k_;
  int aligned_n_;
  int block_;

  void* wgt_;

  reg64_t param_x{abi_param1};
  reg64_t param_y{abi_param2};
  reg64_t param_z{abi_param3};
  reg64_t param_attr{abi_param4};
  reg64_t reg_tmp{rax};

  reg64_t reg_ptr_wgt{r10};
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle
