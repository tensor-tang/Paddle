/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <cstdlib>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
namespace paddle {
namespace operators {

static void StridedSliceFunctor(int* starts, int* ends, int* strides, int* axes,
                                int* reverse_axis, const framework::DDim dims,
                                const size_t size) {
  for (size_t axis = 0; axis < size; axis++) {
    int axis_size = dims[axes[axis]];
    int axis_index = axis;
    if (axis_size < 0) {
      starts[axis_index] = 0;
      ends[axis_index] = 1;
      strides[axis_index] = 1;
    }
    // stride must not be zero
    if (starts[axis_index] < 0) {
      starts[axis_index] = starts[axis_index] + axis_size;
    }

    if (ends[axis_index] < 0) {
      ends[axis_index] = ends[axis_index] + axis_size;
    }
    if (strides[axis_index] < 0) {
      reverse_axis[axis_index] = 1;
      strides[axis_index] = -strides[axis_index];
      if (starts[axis_index] > ends[axis_index]) {
        // swap the reverse
        starts[axis_index] = starts[axis_index] + 1;
        ends[axis_index] = ends[axis_index] + 1;
      }
      std::swap(starts[axis_index], ends[axis_index]);
    } else {
      reverse_axis[axis_index] = 0;
      strides[axis_index] = strides[axis_index];
    }
  }
}

template <typename DeviceContext, typename T>
class StridedSliceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int rank = ctx.Input<framework::Tensor>("Input")->dims().size();
    switch (rank) {
      case 1:
        StridedSliceCompute<1>(ctx);
        break;
      case 2:
        StridedSliceCompute<2>(ctx);
        break;
      case 3:
        StridedSliceCompute<3>(ctx);
        break;
      case 4:
        StridedSliceCompute<4>(ctx);
        break;
      case 5:
        StridedSliceCompute<5>(ctx);
        break;
      case 6:
        StridedSliceCompute<6>(ctx);
        break;
    }
  }

 private:
  template <size_t D>
  void StridedSliceCompute(const framework::ExecutionContext& context) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto in = context.Input<framework::Tensor>("Input");
    auto out = context.Output<framework::Tensor>("Out");
    auto out_dims = out->dims();
    auto in_dims = in->dims();

    auto starts = context.Attr<std::vector<int>>("starts");
    auto ends = context.Attr<std::vector<int>>("ends");
    auto strides = context.Attr<std::vector<int>>("strides");
    auto axes = context.Attr<std::vector<int>>("axes");

    auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto reverse_axis = Eigen::array<bool, D>();

    std::vector<int> reverse_vector(starts.size(), 0);
    StridedSliceFunctor(starts.data(), ends.data(), strides.data(), axes.data(),
                        reverse_vector.data(), in_dims, starts.size());

    for (size_t axis = 0; axis < D; axis++) {
      starts_indices[axis] = 0;
      ends_indices[axis] = out_dims[axis];
      strides_indices[axis] = 1;
    }
    for (size_t axis = 0; axis < axes.size(); axis++) {
      int axis_index = axes[axis];
      starts_indices[axis_index] = starts[axis];
      ends_indices[axis_index] = ends[axis];
      strides_indices[axis_index] = strides[axis];
      reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
    }

    framework::Tensor tmp;
    tmp.mutable_data<T>(out_dims, context.GetPlace());

    out->mutable_data<T>(context.GetPlace());
    auto in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *in);
    auto tmp_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            tmp);
    auto out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *out, out_dims);
    tmp_t.device(place) =
        in_t.stridedSlice(starts_indices, ends_indices, strides_indices);
    out_t.device(place) = tmp_t.reverse(reverse_axis);
  }
};

template <typename DeviceContext, typename T>
class StridedSliceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    size_t rank = ctx.Input<framework::Tensor>("Input")->dims().size();
    switch (rank) {
      case 1:
        StridedSliceGradCompute<1>(ctx);
        break;
      case 2:
        StridedSliceGradCompute<2>(ctx);
        break;
      case 3:
        StridedSliceGradCompute<3>(ctx);
        break;
      case 4:
        StridedSliceGradCompute<4>(ctx);
        break;
      case 5:
        StridedSliceGradCompute<5>(ctx);
        break;
      case 6:
        StridedSliceGradCompute<6>(ctx);
        break;
    }
  }

 private:
  template <size_t D>
  void StridedSliceGradCompute(
      const framework::ExecutionContext& context) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto* d_input =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_out =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    d_out->mutable_data<T>(context.GetPlace());

    auto& dev_ctx = context.template device_context<DeviceContext>();
    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, d_out, static_cast<T>(0));
    auto out_dims = d_out->dims();
    auto in_dims = d_input->dims();
    auto starts = context.Attr<std::vector<int>>("starts");
    auto ends = context.Attr<std::vector<int>>("ends");
    auto strides = context.Attr<std::vector<int>>("strides");
    auto axes = context.Attr<std::vector<int>>("axes");

    auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();

    auto reverse_axis = Eigen::array<bool, D>();
    std::vector<int> reverse_vector(starts.size(), 0);

    StridedSliceFunctor(starts.data(), ends.data(), strides.data(), axes.data(),
                        reverse_vector.data(), out_dims, starts.size());

    for (size_t axis = 0; axis < D; axis++) {
      starts_indices[axis] = 0;
      ends_indices[axis] = out_dims[axis];
      strides_indices[axis] = 1;
    }
    for (size_t axis = 0; axis < axes.size(); axis++) {
      int axis_index = axes[axis];
      starts_indices[axis_index] = starts[axis];
      ends_indices[axis_index] = ends[axis];
      strides_indices[axis_index] = strides[axis];
      reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
    }

    framework::Tensor reverse_input;
    reverse_input.mutable_data<T>(in_dims, context.GetPlace());

    auto in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_input);
    auto reverse_in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            reverse_input);
    auto out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_out, out_dims);

    reverse_in_t.device(place) = in_t.reverse(reverse_axis);
    out_t.stridedSlice(starts_indices, ends_indices, strides_indices)
        .device(place) = reverse_in_t;
  }
};
}  // namespace operators
}  // namespace paddle
