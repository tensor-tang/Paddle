/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/contrib/inference/demo/utils.h"
#include "paddle/contrib/inference/paddle_inference_api.h"

DEFINE_string(dirname, "", "Directory of the inference model.");
DEFINE_int32(batchsize, 1, "Batch size of input data");
DEFINE_int32(repeat, 100, "Running the inference program repeat times");

template <typename T>
void RandomData(T* data, size_t sz, T lower, T upper) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  for (size_t i = 0; i < sz; ++i) {
    data[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
  }
}

TEST(inference, ocr) {
  if (FLAGS_dirname.empty() || FLAGS_batchsize < 1 || FLAGS_repeat < 1) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model "
                  "--batchsize=1 --repeat=1";
  }

  LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname;
  LOG(INFO) << "FLAGS_batchsize: " << FLAGS_batchsize;

  using namespace paddle;
  NativeConfig config;
  config.param_file = FLAGS_dirname + "/params";
  config.prog_file = FLAGS_dirname + "/model";
  config.use_gpu = false;
  config.device = 0;

  LOG(INFO) << "init predictor";
  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

  float input_len = sizeof(float) * FLAGS_batchsize * 48 * 512;
  float* input_data = (float*)malloc(input_len);
  PaddleTensor input{.name = "xx",
                     .shape = {FLAGS_batchsize, 1, 48, 512},
                     .data = PaddleBuf(input_data, input_len),
                     .dtype = PaddleDType::FLOAT32};
  RandomData<float>(static_cast<float*>(input.data.data()),
                    FLAGS_batchsize * 48 * 512,
                    0.f,
                    1.f);

  LOG(INFO) << "run executor";
  std::vector<PaddleTensor> output;
  predictor->Run({input}, &output);

  free(input_data);
}
