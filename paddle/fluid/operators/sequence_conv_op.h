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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/context_project.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class SequenceConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    auto filter = *context.Input<Tensor>("Filter");

    out->mutable_data<T>(context.GetPlace());

    int context_start = context.Attr<int>("contextStart");    // -1, -2
    int context_length = context.Attr<int>("contextLength");  // 4
    int context_stride = context.Attr<int>("contextStride");  // 1
    bool padding_trainable = context.Attr<bool>("paddingTrainable");

    PADDLE_ENFORCE_EQ(in->lod().size(), 1UL,
                      "Only support one level sequence now.");

    PADDLE_ENFORCE(!padding_trainable, "no.");
    PADDLE_ENFORCE_EQ(context_stride, 1);
    // const Tensor* padding_data = nullptr;
    // if (padding_trainable) {
    //   padding_data = context.Input<Tensor>("PaddingData");
    // }

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    int sequence_width = static_cast<int>(in->dims()[1]);
    framework::DDim col_shape = {in->dims()[0],
                                 context_length * sequence_width};
    Tensor col;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

    auto in_lod = in->lod();
    PADDLE_ENFORCE_EQ(in_lod.size(), 1UL);
    T* col_data = col.mutable_data<T>(col_shape, context.GetPlace());
    const T* in_data = in->data<T>();
    for (int i = 0; i < static_cast<int>(in_lod[0].size()) - 1; ++i) {
      int st = in_lod[0][i];
      int ed = in_lod[0][i + 1];
      size_t src_stride = sequence_width;
      size_t src_stride_sz = sequence_width * sizeof(T);
      size_t dst_stride = context_length * sequence_width;
      size_t dst_stride_sz = dst_stride * sizeof(T);

      const T* src_data = in_data + st * src_stride;
      T* dst_data = col_data + st * dst_stride;
      size_t zero_sz = up_pad * src_stride * sizeof(T);
      int seq_len = ed - st;
      if (seq_len > up_pad + down_pad) {
        // fill up pad
        for (int j = 0; j < up_pad; ++j) {
          std::memset(dst_data, 0, zero_sz);
          // blas.VCOPY((dst_stride_sz-zero_sz)/sizeof(T), src_data,
          // dst_data+zero_sz/sizeof(T));
          std::memcpy(dst_data + zero_sz / sizeof(T), src_data,
                      dst_stride_sz - zero_sz);
          dst_data += dst_stride;
          zero_sz -= src_stride_sz;
        }
        // fill data
        for (int j = st + up_pad; j < ed - down_pad; ++j) {
          // blas.VCOPY(dst_stride_sz/ sizeof(T), src_data, dst_data);
          std::memcpy(dst_data, src_data, dst_stride_sz);
          dst_data += dst_stride;
          src_data += src_stride;
        }
        // fill down pad
        zero_sz = src_stride_sz;
        src_data -= src_stride;
        for (int j = 0; j < down_pad; ++j) {
          std::memcpy(dst_data, src_data, dst_stride_sz - zero_sz);
          dst_data += dst_stride;
          std::memset(dst_data - zero_sz / sizeof(T), 0, zero_sz);
          zero_sz += src_stride_sz;
          src_data -= src_stride;
        }
      } else {
        PADDLE_ENFORCE_GE(context_length, up_pad + down_pad + 1);
        std::memset(dst_data, 0, seq_len * dst_stride_sz);
        size_t seq_len_size = seq_len * src_stride_sz;
        for (int j = 0; j < std::min(up_pad, seq_len); ++j) {
          size_t copy_size = std::min(seq_len_size, dst_stride_sz - zero_sz);
          // vcopy?
          std::memcpy(dst_data + zero_sz / sizeof(T), src_data, copy_size);
          dst_data += dst_stride;
          zero_sz -= src_stride_sz;
        }
        zero_sz = down_pad * sequence_width * sizeof(T);
        dst_data = col_data + (ed - 1) * dst_stride;
        src_data = in_data + (ed - up_pad - 1) * src_stride;
        for (int j = 0; j < std::min(0, seq_len - up_pad); ++j) {
          size_t copy_size = std::min(seq_len_size, dst_stride_sz - zero_sz);
          // vcopy?
          std::memcpy(dst_data, src_data, copy_size);
          dst_data -= dst_stride;
          src_data += src_stride;
          zero_sz -= src_stride_sz;
        }
      }
    }
    blas.MatMul(col, filter, out);
  }
};

template <typename DeviceContext, typename T>
class SequenceConvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in_g = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto* out_g = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* filter_g = context.Output<Tensor>(framework::GradVarName("Filter"));
    auto* padding_data_g =
        context.Output<Tensor>(framework::GradVarName("PaddingData"));
    auto* in = context.Input<LoDTensor>("X");
    auto* filter = context.Input<Tensor>("Filter");

    int context_start = context.Attr<int>("contextStart");
    int context_length = context.Attr<int>("contextLength");
    int context_stride = context.Attr<int>("contextStride");
    bool padding_trainable = context.Attr<bool>("paddingTrainable");

    PADDLE_ENFORCE_EQ(in->lod().size(), 1UL,
                      "Only support one level sequence now.");
    auto lod_g_level_0 = in->lod()[0];

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    int sequence_width = static_cast<int>(in->dims()[1]);

    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    // use col_shape in the im2col calculation
    framework::DDim col_shape = {in->dims()[0],
                                 sequence_width * context_length};
    Tensor col;

    if (in_g || filter_g || (padding_trainable && padding_data_g)) {
      col.mutable_data<T>(col_shape, context.GetPlace());
      // Because if padding_trainable is false, padding data should be zeros.
      set_zero(dev_ctx, &col, static_cast<T>(0));
      blas.MatMul(*out_g, false, *filter, true, &col);
    }
    math::ContextProjectFunctor<DeviceContext, T> seq_project_functor;
    math::ContextProjectGradFunctor<DeviceContext, T> seq_project_grad_functor;

    if (in_g) {
      in_g->mutable_data<T>(context.GetPlace());
      in_g->set_lod(in->lod());
      set_zero(dev_ctx, in_g, static_cast<T>(0));

      seq_project_grad_functor(dev_ctx, *in_g, padding_trainable, context_start,
                               context_length, context_stride, up_pad, down_pad,
                               false, true, padding_data_g, &col);
    }

    if (padding_trainable && padding_data_g) {
      padding_data_g->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, padding_data_g, static_cast<T>(0));

      LoDTensor* input = const_cast<LoDTensor*>(in);
      seq_project_grad_functor(
          dev_ctx, *input, padding_trainable, context_start, context_length,
          context_stride, up_pad, down_pad, true, false, padding_data_g, &col);
    }

    if (filter_g) {
      filter_g->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, filter_g, static_cast<T>(0));

      Tensor filter_grad = *filter_g;
      LoDTensor out_grad = *out_g;

      const Tensor* padding_data = nullptr;
      if (padding_trainable) {
        padding_data = context.Input<Tensor>("PaddingData");
      }

      seq_project_functor(dev_ctx, *in, *padding_data, padding_trainable,
                          context_start, context_length, context_stride, up_pad,
                          down_pad, &col);

      blas.MatMul(col, true, out_grad, false, &filter_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle
