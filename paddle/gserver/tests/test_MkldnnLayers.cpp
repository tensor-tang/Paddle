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


void testConvLayer() {
  bool trans = false;
  bool useGpu = false;
  TestConfig config;
  config.biasSize = 64;
  config.layerConfig.set_type("mkldnn_conv");
  config.layerConfig.set_num_filters(64);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 3072, 1728});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_filter_size(3);
  conv->set_filter_size_y(3);
  conv->set_channels(3);
  conv->set_padding(1);
  conv->set_padding_y(1);
  conv->set_stride(1);
  conv->set_stride_y(1);
  conv->set_groups(1);
  conv->set_filter_channels(conv->channels() / conv->groups());
  conv->set_img_size(32);
  conv->set_output_x(outputSize(conv->img_size(), conv->filter_size(),
                                conv->padding(), conv->stride(),
                                /* caffeMode */ true));
  config.layerConfig.set_size(conv->output_x() * conv->output_x() *
                              config.layerConfig.num_filters());

  testLayerGrad(config, "mkldnn_conv", 128, trans, useGpu);
  // Use small batch_size and useWeight=true to test biasGrad
  LOG(INFO) <<"------------------------------------------------13";
  testLayerGrad(config, "mkldnn_conv", 2, trans, useGpu, true, 0.02);

  
}

TEST(Layer, convLayer) {
//  testConvLayer();
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
//  testFcLayer("csc", 4096 * 40);
//  testFcLayer("csr", 4096 * 40);
}


struct test_pool_desc_t {
    int bs, cl;
    int ih, iw;
    int oh, ow;
    int kh, kw;
    int ph, pw;
    int sh, sw;
};

void testPoolLayer(const string& poolType, const test_pool_desc_t pm) {
  bool trans = false;
  TestConfig config;
  config.inputDefs.push_back({INPUT_DATA, "layer_0",
    size_t(pm.ih * pm.iw * pm.cl), 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  PoolConfig* pool = input->mutable_pool_conf();

  config.biasSize = 0;
  config.layerConfig.set_type("mkldnn_pool");
  config.layerConfig.set_num_filters(pm.cl);
  pool->set_pool_type(poolType);
  pool->set_img_size(pm.iw);
  pool->set_img_size_y(pm.ih);
  pool->set_channels(pm.cl);
  pool->set_size_x(pm.kw);
  pool->set_size_y(pm.kh);
  pool->set_start(0);
  pool->set_padding(pm.pw);
  pool->set_padding_y(pm.ph);
  pool->set_stride(pm.sw);
  pool->set_stride_y(pm.sh);

  bool caffeMode = false;
  int oh = outputSize(pm.ih, pm.kh, pm.ph, pm.sh, caffeMode);
  int ow = outputSize(pm.iw, pm.kw, pm.pw, pm.sw, caffeMode);
  CHECK(oh == pm.oh && ow == pm.ow)
    << "double check output size, " << oh << " vs " << pm.oh;
  pool->set_output_x(ow);
  pool->set_output_y(oh);

  config.layerConfig.set_size(pool->output_x() * pool->output_y() *
                              pool->channels());
  // TODO(TJ): use {0, 1} if AddToMode ready 
  for (auto addSize : {0}) {
    config.layerConfig.set_add_size(addSize);
    testLayerGrad(config, "mkldnn_pool", pm.bs, trans, false);
  }
  // TODO(TJ): add comparation with cpu pooling
}

TEST(Layer, PoolLayer) {
  testPoolLayer("max-projection", {10, 64, 32, 32, 16, 16, 2, 2, 0, 0, 2, 2});
  testPoolLayer("max-projection", {100, 16, 14, 14, 7, 7, 3, 3, 0, 0, 2, 2});
//  testPoolLayer("avg-projection");

}

void testBatchNormLayer() {
  bool trans = false;
  bool useGpu = false;
  TestConfig config;
  const int CHANNELS = 10;
  const int IMG_SIZE = 16;
  config.layerConfig.set_type("mkldnn_batch_norm");
  config.layerConfig.set_size(CHANNELS * IMG_SIZE * IMG_SIZE);
  config.layerConfig.set_active_type("mkldnn_relu");
  config.biasSize = CHANNELS;
  config.inputDefs.push_back({INPUT_DATA, "layer_0",
                              /* dim= */ IMG_SIZE * IMG_SIZE * CHANNELS,
                              /* paraSize= */ CHANNELS});

  config.inputDefs.push_back({INPUT_DATA, "layer_1_running_mean", 1, CHANNELS});
  config.inputDefs.back().isStatic = true;
  config.inputDefs.push_back({INPUT_DATA, "layer_2_running_var", 1, CHANNELS});
  config.inputDefs.back().isStatic = true;

  LayerInputConfig* input = config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  ImageConfig* img_conf = input->mutable_image_conf();
  img_conf->set_channels(CHANNELS);
  img_conf->set_img_size(IMG_SIZE);

  testLayerGrad(config, "mkldnn_batch_norm", 64, /* trans= */ trans, useGpu,
                /* useWeight */ true);
}

TEST(Layer, BatchNormalizationLayer) {
//  testBatchNormLayer();
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}


