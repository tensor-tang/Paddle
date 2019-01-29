// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"
#include "time.h"

using namespace paddle;

DEFINE_string(model, "/home/chunwei/project2/models/fc/fluid_checkpoint", "");

TEST(test, main) {
  AnalysisConfig config;
  config.SetModel(FLAGS_model);
  config.SwitchIrOptim(true);
  config.SwitchIrDebug(true);
  // config.EnableMKLDNN();
  config.pass_builder()->TurnOnDebug();

  auto predictor = CreatePaddlePredictor(config);

  std::vector<PaddleTensor> inputs;
  inputs.resize(1);

  auto& input = inputs.front();
  input.shape.assign({FLAGS_batch_size, 210});
  input.data.Resize(FLAGS_batch_size * 210 * sizeof(float));
  auto* data = static_cast<float*>(input.data.data());
  for (int i = 0; i < 210 * FLAGS_batch_size; i++) {
    data[i] = rand() / RAND_MAX;
  }

  std::vector<PaddleTensor> outputs;

  inference::Timer timer;
  timer.tic();
  for (int i = 0; i < 1000; i++) {
    ASSERT_TRUE(predictor->Run(inputs, &outputs));
  }
  LOG(INFO) << "latency " << timer.toc() / 1000;
}
