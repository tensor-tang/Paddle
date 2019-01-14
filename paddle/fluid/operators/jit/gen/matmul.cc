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

  mov(reg_ptr_wgt, reinterpret_cast<size_t>(wgt_));
  const int block_len = sizeof(float) * block_;
  bool careful_save = aligned_n_ != n_;
  size_t wgt_offset = 0;
  size_t z_offset = 0;
  for (size_t g = 0; g < groups.size(); ++g) {
    size_t x_offset = 0;
    for (int k = 0; k < k_; ++k) {
      vbroadcastss(zmm_t(15), ptr[param_x + x_offset]);
      // clean
      if (k == 0) {
        for (int i = 0; i < groups[g]; ++i) {
          vxorps(zmm_t(i), zmm_t(i), zmm_t(i));
        }
      }
      for (int i = 0; i < groups[g]; ++i) {
        vmovups(zmm_t(14), ptr[reg_ptr_wgt + wgt_offset]);
        vfmadd231ps(zmm_t(i), zmm_t(14), zmm_t(15));
        wgt_offset += block_len;
      }
      // last one, save
      if (k == k_ - 1) {
        for (int i = 0; i < groups[g]; ++i) {
          // rest only save should be careful or use xmm
          if (careful_save && g == groups.size() - 1 && i == groups[g] - 1) {
            break;
          }
          vmovups(ptr[param_z + z_offset + i * block_len], zmm_t(i));
        }
      }
      x_offset += sizeof(float);
    }
    z_offset += block_len * groups[g];
  }

  // if (careful_save) {
  //   int i = groups.back();
  //   int rest = n_ %block_;
  //   if ()
  // }

  // const int group_len = max_num_regs * block_ * sizeof(float);
  // for (int g = 0; g < num_groups; ++g) {
  //   pool_height<zmm_t>(g * group_len, block_, max_num_regs);
  // }
  // if (rest_num_regs > 0) {
  //   pool_height<zmm_t>(num_groups * group_len, block_, rest_num_regs);
  // }
  // // part of rest_w * height
  // const int rest = w_ % block_;
  // pool_height_of_rest_width(rest, (w_ - rest) * sizeof(float), max_num_regs);
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
