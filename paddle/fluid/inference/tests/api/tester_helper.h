// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"

#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/tests/api/config_printer.h"
#include "paddle/fluid/inference/tests/test_helper.h"
#include "paddle/fluid/inference/utils/benchmark.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_string(model_name, "", "model name");
DEFINE_string(infer_model, "", "model path");
DEFINE_string(infer_data, "", "data file");
DEFINE_string(refer_result, "", "reference result for comparison");
DEFINE_int32(batch_size, 1, "batch size.");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");
DEFINE_bool(test_all_data, false, "Test the all dataset in data file.");
DEFINE_int32(num_threads, 1, "Running the inference program in multi-threads.");
DEFINE_bool(use_analysis, true,
            "Running the inference program in analysis mode.");
DEFINE_bool(record_benchmark, false,
            "Record benchmark after profiling the model");
DEFINE_double(accuracy, 1e-3, "Result Accuracy.");

DECLARE_bool(profile);
DECLARE_int32(paddle_num_threads);

namespace paddle {
namespace inference {

void PrintConfig(const PaddlePredictor::Config *config, bool use_analysis) {
  if (use_analysis) {
    LOG(INFO) << *reinterpret_cast<const contrib::AnalysisConfig *>(config);
    return;
  }
  LOG(INFO) << *reinterpret_cast<const NativeConfig *>(config);
}

void CompareResult(const std::vector<PaddleTensor> &outputs,
                   const std::vector<PaddleTensor> &ref_outputs) {
  EXPECT_GT(outputs.size(), 0UL);
  EXPECT_EQ(outputs.size(), ref_outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &out = outputs[i];
    auto &ref_out = ref_outputs[i];
    size_t size = VecReduceToInt(out.shape);
    size_t ref_size = VecReduceToInt(ref_out.shape);
    EXPECT_GT(size, 0UL);
    EXPECT_EQ(size, ref_size);
    EXPECT_EQ(out.dtype, ref_out.dtype);
    switch (out.dtype) {
      case PaddleDType::INT64: {
        int64_t *pdata = static_cast<int64_t *>(out.data.data());
        int64_t *pdata_ref = static_cast<int64_t *>(ref_out.data.data());
        for (size_t j = 0; j < size; ++j) {
          EXPECT_EQ(pdata_ref[j], pdata[j]);
        }
        break;
      }
      case PaddleDType::FLOAT32: {
        float *pdata = static_cast<float *>(out.data.data());
        float *pdata_ref = static_cast<float *>(ref_out.data.data());
        for (size_t j = 0; j < size; ++j) {
          EXPECT_NEAR(pdata_ref[j], pdata[j], FLAGS_accuracy);
        }
        break;
      }
    }
  }
}

std::unique_ptr<PaddlePredictor> CreateTestPredictor(
    const PaddlePredictor::Config *config, bool use_analysis = true) {
  if (use_analysis) {
    return CreatePaddlePredictor<contrib::AnalysisConfig>(
        *(reinterpret_cast<const contrib::AnalysisConfig *>(config)));
  }
  return CreatePaddlePredictor<NativeConfig>(
      *(reinterpret_cast<const NativeConfig *>(config)));
}

size_t GetSize(const PaddleTensor &out) { return VecReduceToInt(out.shape); }

std::unordered_map<std::string, int> GetFuseStatis(PaddlePredictor *predictor,
                                                   int *num_ops) {
  std::unordered_map<std::string, int> res;
  auto *analysis_predictor = static_cast<AnalysisPredictor *>(predictor);
  auto *fusion_status =
      analysis_predictor->analysis_argument().fusion_statis_ptr();
  if (!fusion_status) {
    return res;
  }
  for (auto &item : *fusion_status) {
    LOG(INFO) << "fused " << item.first << " " << item.second;
  }
  int num = 0;
  for (auto &node :
       analysis_predictor->analysis_argument().main_graph().Nodes()) {
    if (node->IsOp()) {
      ++num;
    }
  }
  *num_ops = num;
  return *fusion_status;
}

void SetFakeImageInput(std::vector<std::vector<PaddleTensor>> *inputs,
                       const std::string &dirname, bool is_combined = true,
                       std::string model_filename = "model",
                       std::string params_filename = "params") {
  // Set fake_image_data
  PADDLE_ENFORCE_EQ(FLAGS_test_all_data, 0, "Only have single batch of data.");
  std::vector<std::vector<int64_t>> feed_target_shapes = GetFeedTargetShapes(
      dirname, is_combined, model_filename, params_filename);
  std::ostringstream os;
  for (size_t i = 0; i < feed_target_shapes.size(); ++i) {
    os << "feed target " << i << ": {" << feed_target_shapes[i][0];
    for (size_t j = 1; j < feed_target_shapes[i].size(); ++j) {
      os << ", " << feed_target_shapes[i][j];
    }
    os << "}\n";
  }
  LOG(INFO) << os.str();
  std::vector<std::string> names = {
      "slot10000_embed", "slot10001_embed", "slot10004_embed",
      "slot10005_embed", "slot10008_embed", "slot10009_embed",
      "slot10012_embed", "slot10013_embed", "slot10108_embed",
      "slot13324_embed", "slot13325_embed", "slot13326_embed",
      "slot13327_embed", "slot13328_embed", "slot13329_embed",
      "slot13330_embed", "slot13331_embed", "slot15501_embed",
      "slot15502_embed", "slot15503_embed", "slot15504_embed",
      "slot15505_embed", "slot15506_embed", "slot15507_embed",
      "slot15508_embed", "slot15516_embed", "slot15519_embed",
      "slot15523_embed", "slot15531_embed", "slot15533_embed",
      "slot15548_embed", "slot15564_embed", "slot15565_embed",
      "slot15566_embed", "slot15570_embed", "slot15571_embed",
      "slot15572_embed", "slot15573_embed", "slot15574_embed",
      "slot15575_embed", "slot15576_embed", "slot15577_embed",
      "slot15579_embed", "slot15581_embed", "slot15582_embed",
      "slot15583_embed", "slot15584_embed", "slot5016_embed",
      "slot5021_embed",  "slot6002_embed",  "slot6003_embed",
      "slot6004_embed",  "slot6005_embed",  "slot6006_embed",
      "slot6007_embed",  "slot6008_embed",  "slot6009_embed",
      "slot6011_embed",  "slot6014_embed",  "slot6015_embed",
      "slot6023_embed",  "slot6024_embed",  "slot6025_embed",
      "slot6027_embed",  "slot6029_embed",  "slot6031_embed",
      "slot6034_embed",  "slot6035_embed",  "slot6036_embed",
      "slot6037_embed",  "slot6039_embed",  "slot6048_embed",
      "slot6050_embed",  "slot6058_embed",  "slot6059_embed",
      "slot6060_embed",  "slot6066_embed",  "slot6067_embed",
      "slot6068_embed",  "slot6069_embed",  "slot6070_embed",
      "slot6071_embed",  "slot6072_embed",  "slot6073_embed",
      "slot6182_embed",  "slot6183_embed",  "slot6184_embed",
      "slot6185_embed",  "slot6186_embed",  "slot6188_embed",
      "slot6189_embed",  "slot6190_embed",  "slot6201_embed",
      "slot6202_embed",  "slot6203_embed",  "slot6247_embed",
      "slot6248_embed",  "slot6250_embed",  "slot6251_embed",
      "slot6807_embed",  "slot6808_embed",  "slot6809_embed",
      "slot6810_embed",  "slot6811_embed",  "slot6812_embed",
      "slot6813_embed",  "slot6814_embed",  "slot6815_embed",
      "slot6816_embed",  "slot6817_embed",  "slot6818_embed",
      "slot6819_embed",  "slot6820_embed",  "slot6822_embed",
      "slot6823_embed",  "slot6826_embed",  "slot7002_embed",
      "slot7003_embed",  "slot7004_embed",  "slot7005_embed",
      "slot7006_embed",  "slot7008_embed",  "slot7009_embed",
      "slot7010_embed",  "slot7011_embed",  "slot7013_embed",
      "slot7014_embed",  "slot7015_embed",  "slot7016_embed",
      "slot7017_embed",  "slot7019_embed",  "slot7100_embed",
      "slot7506_embed",  "slot7507_embed",  "slot7514_embed",
      "slot7515_embed",  "slot7516_embed"};
  PADDLE_ENFORCE_EQ(names.size(), feed_target_shapes.size());
  std::vector<PaddleTensor> input_slots(feed_target_shapes.size());

  for (size_t i = 0; i < feed_target_shapes.size(); ++i) {
    const auto &feed_shape = feed_target_shapes[i];
    // PaddleTensor input ;
    auto &input = input_slots[i];
    std::vector<int> shape({FLAGS_batch_size});
    for (size_t s = 1; s < feed_shape.size(); ++s) {
      shape.push_back(static_cast<int>(feed_shape[s]));
    }
    input.name = names[i];
    input.shape = shape;
    input.dtype = PaddleDType::FLOAT32;
    size_t len = std::accumulate(shape.begin(), shape.end(), 1,
                                 [](int a, int b) { return a * b; });
    input.data.Resize(len * sizeof(float));
    input.lod.assign({{0, 1}});
    PADDLE_ENFORCE_EQ(len, 11);
    float *input_data = static_cast<float *>(input.data.data());
    // fill input data, for profile easily, do not use random data here.
    for (size_t j = 0; j < len; ++j) {
      *(input_data + j) = static_cast<float>(j) / len;
    }

    // std::vector<PaddleTensor> input_slots;
    // input_slots.assign({input});
    //  (*inputs).emplace_back(input_slots);
  }
  (*inputs).emplace_back(input_slots);
}

void TestOneThreadPrediction(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    std::vector<PaddleTensor> *outputs, bool use_analysis = true) {
  int batch_size = FLAGS_batch_size;
  int num_times = FLAGS_repeat;
  auto predictor = CreateTestPredictor(config, use_analysis);

  // warmup run
  LOG(INFO) << "Warm up run...";
  {
    Timer warmup_timer;
    warmup_timer.tic();
    predictor->Run(inputs[0], outputs, batch_size);
    PrintTime(batch_size, 1, 1, 0, warmup_timer.toc(), 1);
    if (FLAGS_profile) {
      paddle::platform::ResetProfiler();
    }
  }

  LOG(INFO) << "Run " << num_times << " times...";
  {
    Timer run_timer;
    run_timer.tic();
    for (int i = 0; i < num_times; i++) {
      for (size_t j = 0; j < inputs.size(); j++) {
        predictor->Run(inputs[j], outputs, batch_size);
      }
    }

    double latency = run_timer.toc() / num_times;
    PrintTime(batch_size, num_times, 1, 0, latency, inputs.size());
    if (FLAGS_record_benchmark) {
      Benchmark benchmark;
      benchmark.SetName(FLAGS_model_name);
      benchmark.SetBatchSize(batch_size);
      benchmark.SetLatency(latency);
      benchmark.PersistToFile("benchmark_record.txt");
    }
  }
}

void TestMultiThreadPrediction(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    std::vector<PaddleTensor> *outputs, int num_threads,
    bool use_analysis = true) {
  int batch_size = FLAGS_batch_size;
  int num_times = FLAGS_repeat;
  std::vector<std::thread> threads;
  auto main_predictor = CreateTestPredictor(config, use_analysis);

  size_t total_time{0};
  for (int tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      // Each thread should have local inputs and outputs.
      // The inputs of each thread are all the same.
      std::vector<PaddleTensor> outputs_tid;
      // To ensure the thread binding correctly,
      // please clone inside the threadpool.
      auto predictor = main_predictor->Clone();
#ifdef PADDLE_WITH_MKLDNN
      if (use_analysis) {
        static_cast<AnalysisPredictor *>(predictor.get())
            ->SetMkldnnThreadID(static_cast<int>(tid) + 1);
      }
#endif

      // warmup run
      LOG(INFO) << "Running thread " << tid << ", warm up run...";
      {
        Timer warmup_timer;
        warmup_timer.tic();
        predictor->Run(inputs[0], outputs, batch_size);
        PrintTime(batch_size, 1, num_threads, tid, warmup_timer.toc(), 1);
        if (FLAGS_profile) {
          paddle::platform::ResetProfiler();
        }
      }

      LOG(INFO) << "Thread " << tid << " run " << num_times << " times...";
      {
        Timer timer;
        timer.tic();
        for (int i = 0; i < num_times; i++) {
          for (const auto &input : inputs) {
            ASSERT_TRUE(predictor->Run(input, &outputs_tid));
          }
        }

        auto time = timer.toc();
        total_time += time;
        PrintTime(batch_size, num_times, num_threads, tid, time / num_times,
                  inputs.size());
      }
    });
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }
}

void TestPrediction(const PaddlePredictor::Config *config,
                    const std::vector<std::vector<PaddleTensor>> &inputs,
                    std::vector<PaddleTensor> *outputs, int num_threads,
                    bool use_analysis = FLAGS_use_analysis) {
  PrintConfig(config, use_analysis);
  if (num_threads == 1) {
    TestOneThreadPrediction(config, inputs, outputs, use_analysis);
  } else {
    TestMultiThreadPrediction(config, inputs, outputs, num_threads,
                              use_analysis);
  }
}

void CompareDeterministic(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  int batch_size = FLAGS_batch_size;
  int num_times = FLAGS_repeat;
  auto predictor = CreateTestPredictor(config, FLAGS_use_analysis);

  // warmup run
  std::vector<PaddleTensor> warmup_outputs, outputs;
  predictor->Run(inputs[0], &warmup_outputs, batch_size);

  // run num_times to Compare Deterministic Result.
  for (int i = 0; i < num_times; i++) {
    for (size_t j = 0; j < inputs.size(); j++) {
      predictor->Run(inputs[j], &outputs, batch_size);
      CompareResult(outputs, warmup_outputs);
    }
  }
}

void CompareNativeAndAnalysis(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  PrintConfig(config, true);
  std::vector<PaddleTensor> native_outputs, analysis_outputs;
  TestOneThreadPrediction(config, inputs, &native_outputs, false);
  TestOneThreadPrediction(config, inputs, &analysis_outputs, true);
  CompareResult(analysis_outputs, native_outputs);
}

template <typename T>
std::string LoDTensorSummary(const framework::LoDTensor &tensor) {
  std::stringstream ss;
  ss << "\n---- tensor ---" << '\n';
  ss << "lod: [";
  for (const auto &level : tensor.lod()) {
    ss << "[ ";
    for (auto i : level) {
      ss << i << ", ";
    }
    ss << "]";
  }
  ss << "]\n";

  ss << "shape: [";
  int size = 1;
  for (int i = 0; i < tensor.dims().size(); i++) {
    int dim = tensor.dims()[i];
    ss << dim << ", ";
    size *= dim;
  }
  ss << "]\n";

  ss << "data: ";
  for (int i = 0; i < std::min(20, size); i++) {
    ss << tensor.data<T>()[i] << " ";
  }
  ss << "\n";

  return ss.str();
}

static bool CompareLoD(const framework::LoD &a, const framework::LoD &b) {
  if (a.size() != b.size()) {
    LOG(ERROR) << string::Sprintf("lod size not match %d != %d", a.size(),
                                  b.size());
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    auto &al = a[i];
    auto &bl = b[i];
    if (al.size() != bl.size()) {
      LOG(ERROR) << string::Sprintf("level size %d != %d", al.size(),
                                    bl.size());
      return false;
    }
  }
  return true;
}

static bool CompareShape(const std::vector<int64_t> &a,
                         const std::vector<int64_t> &b) {
  if (a.size() != b.size()) {
    LOG(ERROR) << string::Sprintf("shape size not match %d != %d", a.size(),
                                  b.size());
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) {
      LOG(ERROR) << string::Sprintf("shape %d-th element not match %d != %d", i,
                                    a[i], b[i]);
      return false;
    }
  }
  return true;
}

static bool CompareTensorData(const framework::LoDTensor &a,
                              const framework::LoDTensor &b) {
  auto a_shape = framework::vectorize(a.dims());
  auto b_shape = framework::vectorize(b.dims());
  size_t a_size = std::accumulate(a_shape.begin(), a_shape.end(), 1,
                                  [](int a, int b) { return a * b; });
  size_t b_size = std::accumulate(b_shape.begin(), b_shape.end(), 1,
                                  [](int a, int b) { return a * b; });
  if (a_size != b_size) {
    LOG(ERROR) << string::Sprintf("tensor data size not match, %d != %d",
                                  a_size, b_size);
  }

  for (size_t i = 0; i < a_size; i++) {
    if (a.type() == framework::proto::VarType::FP32) {
      const auto *a_data = a.data<float>();
      const auto *b_data = b.data<float>();
      if (std::abs(a_data[i] - b_data[i]) > 1e-3) {
        LOG(ERROR) << string::Sprintf(
            "tensor data %d-th element not match, %f != %f", i, a_data[i],
            b_data[i]);
        return false;
      }
    } else if (a.type() == framework::proto::VarType::INT64) {
      const auto *a_data = a.data<int64_t>();
      const auto *b_data = b.data<int64_t>();
      if (std::abs(a_data[i] - b_data[i]) > 1e-3) {
        LOG(ERROR) << string::Sprintf(
            "tensor data %d-th element not match, %f != %f", i, a_data[i],
            b_data[i]);
        return false;
      }
    }
  }

  return true;
}

static bool CompareTensor(const framework::LoDTensor &a,
                          const framework::LoDTensor &b) {
  if (!CompareLoD(a.lod(), b.lod())) {
    return false;
  }
  if (!CompareShape(framework::vectorize(a.dims()),
                    framework::vectorize(b.dims()))) {
    return false;
  }

  if (!CompareTensorData(a, b)) {
    return false;
  }

  return true;
}

}  // namespace inference
}  // namespace paddle
