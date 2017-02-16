/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/math/MathUtils.h"

#include "LayerGradUtil.h"
#include "TestUtil.h"


using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

//P_DECLARE_bool(use_gpu);
//P_DECLARE_int32(gpu_id);
P_DECLARE_double(checkgrad_eps);
P_DECLARE_bool(thread_local_rand_use_global_seed);
P_DECLARE_bool(prev_batch_state);


void testConvLayer(bool trans) {
  TestConfig config;
  config.biasSize = 16;
  config.layerConfig.set_type("mkldnn_conv");
  config.layerConfig.set_num_filters(16);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 768, 288});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_filter_size(2);
  conv->set_filter_size_y(3);
  conv->set_channels(3);
  conv->set_padding(0);
  conv->set_padding_y(1);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_groups(1);
  conv->set_filter_channels(conv->channels() / conv->groups());
  conv->set_img_size(16);
  conv->set_output_x(outputSize(conv->img_size(), conv->filter_size(),
                                conv->padding(), conv->stride(),
                                /* caffeMode */ true));
  config.layerConfig.set_size(conv->output_x() * conv->output_x() *
                              config.layerConfig.num_filters());

  testLayerGrad(config, "mkldnn_conv", 100, trans, false);
  // Use small batch_size and useWeight=true to test biasGrad
  testLayerGrad(config, "mkldnn_conv", 2, trans, false, true, 0.02);
}

TEST(Layer, convLayer) {
  testConvLayer(/* trans= */ false);
}

void testFcLayer(string format, size_t nnz) {
  TestConfig config;
  config.biasSize = 4096;
  config.layerConfig.set_type("mkldnn_fc");
  config.layerConfig.set_size(4096);
  config.layerConfig.set_active_type("sigmoid");
  config.layerConfig.set_drop_rate(0.1);

  config.inputDefs.push_back(
      {INPUT_DATA, "layer_0", 8192, nnz, ParaSparse(format)});
  config.layerConfig.add_inputs();

  LOG(INFO) << config.inputDefs[0].sparse.sparse << " "
            << config.inputDefs[0].sparse.format;

  testLayerGrad(config, "mkldnn_fc", 100, false, false, true);
}

TEST(Layer, fcLayer) {
  testFcLayer("", 4096 * 4096 * 2);
  testFcLayer("csc", 4096 * 40);
  testFcLayer("csr", 4096 * 40);
}

void setPoolConfig(TestConfig* config, PoolConfig* pool,
                   const string& poolType) {
  (*config).biasSize = 0;
  (*config).layerConfig.set_type("mkldnn_pool");
  (*config).layerConfig.set_num_filters(16);

  int kw = 3, kh = 3;
  int pw = 0, ph = 0;
  int sw = 2, sh = 2;
  pool->set_pool_type(poolType);
  pool->set_channels(16);
  pool->set_size_x(kw);
  pool->set_size_y(kh);
  pool->set_start(0);
  pool->set_padding(pw);
  pool->set_padding_y(ph);
  pool->set_stride(sw);
  pool->set_stride_y(sh);

  int ow = outputSize(pool->img_size(), kw, pw, sw, /* caffeMode */ false);
  int oh = outputSize(pool->img_size_y(), kh, ph, sh, /* caffeMode */ false);
  pool->set_output_x(ow);
  pool->set_output_y(oh);
}

void testPoolLayer(const string& poolType, bool trans) {
  TestConfig config;
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 3136, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  PoolConfig* pool = input->mutable_pool_conf();

  pool->set_img_size(14);
  pool->set_img_size_y(14);
  setPoolConfig(&config, pool, poolType);
  config.layerConfig.set_size(pool->output_x() * pool->output_y() *
                              pool->channels());

  testLayerGrad(config, "mkldnn_pool", 100, trans, false);
}

TEST(Layer, PoolLayer) {
  testPoolLayer("max-projection", /* trans= */ false);
//  testPoolLayer("avg-projection", /* trans= */ false);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}


