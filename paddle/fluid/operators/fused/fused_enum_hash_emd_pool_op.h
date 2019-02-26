/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

template <typename T>
struct EmbeddingVSumFunctor {
  void operator()(const framework::ExecutionContext &context,
                  const LoDTensor *table_t, const LoDTensor *ids_t,
                  LoDTensor *output_t) {
    auto *table = table_t->data<T>();
    int64_t table_height = table_t->dims()[0];
    int64_t table_width = table_t->dims()[1];
    int64_t out_width = output_t->dims()[1];
    const int64_t *ids = ids_t->data<int64_t>();
    auto ids_lod = ids_t->lod()[0];
    int64_t idx_width = ids_t->numel() / ids_lod.back();
    auto *output = output_t->mutable_data<T>(context.GetPlace());

    PADDLE_ENFORCE_LE(table_width * idx_width, out_width);
    PADDLE_ENFORCE_GT(ids_lod.size(), 1UL);

    jit::emb_seq_pool_attr_t attr(table_height, table_width, 0, idx_width,
                                  out_width, jit::SeqPoolType::kSum);
    for (size_t i = 0; i != ids_lod.size() - 1; ++i) {
      attr.index_height = ids_lod[i + 1] - ids_lod[i];
      auto emb_seqpool = jit::Get<jit::kEmbSeqPool, jit::EmbSeqPoolTuples<T>,
                                  platform::CPUPlace>(attr);
      emb_seqpool(table, ids + ids_lod[i] * idx_width, output + i * out_width,
                  &attr);
    }
  }
};

template <typename DeviceContext, typename T>
class FusedEnumHashEmdPoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<LoDTensor>("X");
    // auto* out = context.Output<LoDTensor>("Out");
    auto win_size = context.Attr<std::vector<int>>("win_size");
    int pad_value = context.Attr<int>("pad_value");

    auto in_dims = in->dims();
    auto in_lod = in->lod();

    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(in_dims[0]), in_lod[0].back(),
        "The actual input data's size mismatched with LoD information.");

    // enumerate
    int max_win_size = *std::max_element(win_size.begin(), win_size.end());
    LoDTensor enum_out;
    enum_out.Resize({in_dims[0], max_win_size});

    // Generate enumerate sequence set
    auto lod0 = in_lod[0];
    auto in_data = in->data<T>();
    auto enum_out_data = enum_out.mutable_data<T>(context.GetPlace());

    for (size_t i = 0; i < lod0.size() - 1; ++i) {
      for (size_t idx = lod0[i]; idx < lod0[i + 1]; ++idx) {
        for (int word_idx = 0; word_idx < max_win_size; ++word_idx) {
          size_t word_pos = idx + word_idx;
          enum_out_data[max_win_size * idx + word_idx] =
              word_pos < lod0[i + 1] ? in_data[word_pos] : pad_value;
        }
      }
    }

    // hash
    std::vector<int> num_hash = context.Attr<std::vector<int>>("num_hash");
    std::vector<int> mod_by = context.Attr<std::vector<int>>("mod_by");
    PADDLE_ENFORCE(
        num_hash.size() == mod_by.size() && num_hash.size() == win_size.size(),
        "All attributes's count should be equal!");
    std::vector<LoDTensor> hash_out;
    auto seq_length = in_dims[0];
    for (auto hash_len : num_hash) {
      LoDTensor lod_tensor;
      lod_tensor.set_lod(in_lod);
      lod_tensor.Resize({seq_length, hash_len, 1});
      hash_out.push_back(lod_tensor);
    }
    for (int win_idx = 0; win_idx < win_size.size(); win_idx++) {
      auto hash_out_data =
          hash_out[win_idx].mutable_data<T>(context.GetPlace());
      auto *input = enum_out_data;
      auto last_dim = win_size[win_idx];
      for (int idx = 0; idx < seq_length; ++idx) {
        for (int ihash = 0; ihash != num_hash[win_idx]; ++ihash) {
          hash_out_data[idx * num_hash[win_idx] + ihash] =
              XXH64(input, sizeof(int) * last_dim, ihash) % mod_by[win_idx];
        }
        input += max_win_size;
      }
    }

    //
    EmbeddingVSumFunctor<float> functor;
    std::vector<std::string> outputs = {"Out1", "Out2", "Out3"};
    for (int idx = 0; idx < outputs.size(); idx++) {
      const LoDTensor *table_var = context.Input<LoDTensor>("W1");
      LoDTensor *output_t = context.Output<LoDTensor>(outputs[idx]);
      functor(context, table_var, &hash_out[idx], output_t);
    }

    const LoDTensor *table_var = context.Input<LoDTensor>("W0");
    LoDTensor *output_t = context.Output<LoDTensor>("Out0");
    functor(context, table_var, in, output_t);
  }
};

}  // namespace operators
}  // namespace paddle
