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

#include "paddle/fluid/operators/jit/gen/matmul.h"
#include <vector>
#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

void MatMulJitCode::genCode() {
  preCode();

  constexpr int max_num_regs = 16 - 2;  // one for x, one for y, others for z
  const int num_block = aligned_n_ / block_;
  const int num_groups = num_block / max_num_regs;
  std::vector<int> groups(num_groups, max_num_regs);
  int rest_num_regs = num_block % max_num_regs;
  if (rest_num_regs != 0) {
    groups.push_back(rest_num_regs);
  }
  PADDLE_ENFORCE_GT(groups.front(), 0);

  //  mov(reg_ptr_wgt, reinterpret_cast<size_t>(wgt_));
  const int block_len = sizeof(float) * block_;
  bool careful_save = aligned_n_ != n_;

  // prepare_wgt(groups);

  size_t z_offset = 0;
  int n_offset = 0;
  for (size_t g = 0; g < groups.size(); ++g) {
    size_t x_offset = 0;
    for (int k = 0; k < k_; ++k) {
      size_t y_offset = sizeof(float) * (k * n_ + n_offset * block_);
      vbroadcastss(zmm_t(15), ptr[param_x + x_offset]);
      // clean
      if (k == 0) {
        for (int i = 0; i < groups[g]; ++i) {
          vxorps(zmm_t(i), zmm_t(i), zmm_t(i));
        }
      }
      for (int i = 0; i < groups[g]; ++i) {
        if (careful_save && g == groups.size() - 1 && i == groups[g] - 1) {
          continue;
        }
        vmovups(zmm_t(14), ptr[param_y + y_offset + i * block_len]);
        vfmadd231ps(zmm_t(i), zmm_t(14), zmm_t(15));
        // wgt_offset += block_len;
      }
      // last one, save
      if (k == k_ - 1) {
        for (int i = 0; i < groups[g]; ++i) {
          // only rest save should be careful
          if (careful_save && g == groups.size() - 1 && i == groups[g] - 1) {
            break;
          }
          vmovups(ptr[param_z + z_offset + i * block_len], zmm_t(i));
        }
      }
      x_offset += sizeof(float);
    }
    z_offset += block_len * groups[g];
    n_offset += groups[g];
  }

  // if (careful_save) {
  //   int reg_idx = groups.back() -1;
  //   int rest = n_ % block_;
  //   int shift_num = 0;
  //   z_offset = (n_ - (n_%block_)) * sizeof(float);
  //   while (rest >0){
  //     int inner_block = block_/2;
  //     if (rest >= 8) {
  //         vmovups(ptr[param_z + z_offset], ymm_t(reg_idx));

  //         rest -= 8;

  //     } else if (rest >= 4) {

  //     } else if (rest >=2) {

  //     } else {

  //     }

  //   }
  // }

  postCode();
}

class MatMulCreator : public JitCodeCreator<matmul_attr_t> {
 public:
  bool UseMe(const matmul_attr_t& attr) const override {
    return attr.m == 1 && platform::MayIUse(platform::avx512f);
  }
  size_t CodeSize(const matmul_attr_t& attr) const override {
    // change me
    return 96 +
           ((attr.k / YMM_FLOAT_BLOCK + 4 /* for rest */) *
                4 /* load, mul and save */ +
            256) *
               attr.n * 8;
  }
  std::unique_ptr<GenBase> CreateJitCode(
      const matmul_attr_t& attr) const override {
    PADDLE_ENFORCE_GT(attr.m, 0);
    PADDLE_ENFORCE_GT(attr.n, 0);
    PADDLE_ENFORCE_GT(attr.k, 0);
    return make_unique<MatMulJitCode>(attr, CodeSize(attr));
  }
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace gen = paddle::operators::jit::gen;

REGISTER_JITKERNEL_GEN(kMatMul, gen::MatMulCreator);
