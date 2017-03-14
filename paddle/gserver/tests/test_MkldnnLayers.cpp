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

P_DECLARE_double(checkgrad_eps);
P_DECLARE_bool(thread_local_rand_use_global_seed);
P_DECLARE_bool(prev_batch_state);

struct testConvDesc {
  int bs, gp;
  int ic, ih, iw;
  int oc, oh, ow;
  int kh, kw;
  int ph, pw;
  int sh, sw;
};

void testConvLayer(const testConvDesc& pm) {
  bool trans = false;
  bool useGpu = false;
  TestConfig config;
  config.biasSize = pm.oc;
  config.layerConfig.set_type("mkldnn_conv");
  config.layerConfig.set_num_filters(pm.oc);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);
  config.inputDefs.push_back({INPUT_DATA, "layer_0",
    size_t(pm.ih * pm.iw * pm.ic),  // size of input layer
    size_t(pm.kw * pm.kh * pm.oc * pm.ic / pm.gp)});  // param size
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_filter_size(pm.kw);
  conv->set_filter_size_y(pm.kh);
  conv->set_channels(pm.ic);
  conv->set_padding(pm.pw);
  conv->set_padding_y(pm.ph);
  conv->set_stride(pm.sw);
  conv->set_stride_y(pm.sh);
  conv->set_groups(pm.gp);
  conv->set_filter_channels(conv->channels() / conv->groups());
  conv->set_img_size(pm.iw);
  CHECK(conv->filter_channels() * pm.gp == conv->channels())
    << "it is indivisible";
  bool caffeMode = true;
  int ow = outputSize(pm.iw, pm.kw, pm.pw, pm.sw, caffeMode);
  CHECK(ow == pm.ow)
    << "double check output size, " << ow << " vs " << pm.ow;
  conv->set_output_x(ow);
  config.layerConfig.set_size(conv->output_x() * conv->output_x() *
                              config.layerConfig.num_filters());

  // TODO(TJ): test both true and false
  config.layerConfig.set_use_mkldnn_wgt(false);

  // TODO(TJ): use {0, 1} if AddToMode ready
  for (auto addSize : {0}) {
    config.layerConfig.set_add_size(addSize);
    testLayerGrad(config, "mkldnn_conv", pm.bs, trans, useGpu);
    // Use small batch_size and useWeight=true to test biasGrad
    testLayerGrad(config, "mkldnn_conv", 2, trans, useGpu, true, 0.02);
  }

  // test comparison with exconv
  TestConfig ref = config;
  ref.layerConfig.set_type("exconv");
  std::vector<TestConfig> cfg = {config, ref};
  // TODO(TJ): use {0, 1} if AddToMode ready
  for (auto addSize : {0}) {
    config.layerConfig.set_add_size(addSize);
    for (auto bs : {1, pm.bs}) {
      testLayerFunc(cfg, bs);
    }
  }
}

TEST(MkldnnLayer, convLayer) {
  testConvLayer({64, 1, 3, 32, 32, 64, 32, 32, 3, 3, 1, 1, 1, 1});
  testConvLayer({100, 1, 8, 32, 32, 64, 32, 32, 3, 3, 1, 1, 1, 1});
  testConvLayer({128, 1, 64, 14, 14, 32, 14, 14, 3, 3, 1, 1, 1, 1});
  testConvLayer({2, 1, 64, 14, 14, 32, 7, 7, 1, 1, 0, 0, 2, 2});
  // TODO(TJ): enable and test group != 1
//  testConvLayer({1, 2, 4, 32, 32, 4, 32, 32, 3, 3, 1, 1, 1, 1});
//  testConvLayer({128, 2, 64, 14, 14, 32, 14, 14, 3, 3, 1, 1, 1, 1});
}

struct testPoolDesc {
  int bs, cl;
  int ih, iw;
  int oh, ow;
  int kh, kw;
  int ph, pw;
  int sh, sw;
};

void testPoolLayer(const string& poolType, const testPoolDesc& pm) {
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
  config.layerConfig.set_use_mkldnn_wgt(false);
  // TODO(TJ): use {0, 1} if AddToMode ready
  for (auto addSize : {0}) {
    config.layerConfig.set_add_size(addSize);
    testLayerGrad(config, "mkldnn_pool", pm.bs, trans, false);
  }

  // test comparison with pool
  TestConfig ref = config;
  ref.layerConfig.set_type("pool");
  std::vector<TestConfig> cfg = {config, ref};
  // TODO(TJ): use {0, 1} if AddToMode ready
  for (auto addSize : {0}) {
    config.layerConfig.set_add_size(addSize);
    for (auto bs : {1, pm.bs}) {
      testLayerFunc(cfg, bs);
    }
  }
}

TEST(MkldnnLayer, PoolLayer) {
  testPoolLayer("max-projection", {10, 64, 32, 32, 16, 16, 2, 2, 0, 0, 2, 2});
  testPoolLayer("max-projection", {100, 16, 14, 14, 7, 7, 3, 3, 0, 0, 2, 2});
  testPoolLayer("max-projection", {8, 192, 56, 56, 28, 28, 3, 3, 0, 0, 2, 2});
  testPoolLayer("max-projection", {16, 64, 56, 56, 29, 29, 3, 3, 1, 1, 2, 2});
/**
 * Do not compare stride > 1 in avg-pool (avg = sum/poolSize)
 * In MKLDNN: poolSize if always fw * fh, no matter if the kernel is out range
 * of the image size, so avg = sum/(fw*fh)
 * But in Paddle: poolSize is the element of valid element, do not included the
 * out range of padding element (avg = sum/valid_num).
 * So when kernel may scan out range of the input image size, the result BTW
 * MKLDNN and Paddle are not the same
 * However this will not impact max-pool, as the padding are always zeros.
 */
  testPoolLayer("avg-projection", {2, 64, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1});
}

struct testFCDesc {
  int bs;
  int ic;
  int oc;
  int ih, iw;  // oh == ow == 1
};

void testFcLayer(const testFCDesc& pm) {
  TestConfig config;
  config.biasSize = pm.oc;
  config.layerConfig.set_type("mkldnn_fc");
  config.layerConfig.set_size(pm.oc);
  config.inputDefs.push_back({INPUT_DATA, "layer_0",
    /* size of input layer= */ size_t(pm.ic * pm.ih * pm.iw),
    /* size of weight= */  size_t(pm.ic * pm.oc)});
  config.layerConfig.add_inputs();

  // TODO(TJ): test both true and false
  config.layerConfig.set_use_mkldnn_wgt(false);

  // test functionality as fc
  TestConfig ref = config;
  ref.layerConfig.set_type("fc");
  std::vector<TestConfig> cfg = {config, ref};
  // TODO(TJ): use {0, 1} if AddToMode ready
  for (auto addSize : {0}) {
    config.layerConfig.set_add_size(addSize);
    for (auto bs : {1, pm.bs}) {
      testLayerFunc(cfg, bs);
    }
  }

  // test layer grad
  config.layerConfig.set_active_type("sigmoid");
  config.layerConfig.set_drop_rate(0.1);
  // TODO(TJ): use {0, 1} if AddToMode ready
  for (auto addSize : {0}) {
    config.layerConfig.set_add_size(addSize);
    testLayerGrad(config, "mkldnn_fc", pm.bs, false, false, true);
  }
}

TEST(MkldnnLayer, fcLayer) {
  testFcLayer({100, 512, 128, 1, 1});
// TODO(TJ): test iw and ih both > 1
// do not support sparse yet
}

struct testBNDesc {
  int bs;
  int ic;
  int ih, iw;
};

void testBatchNormLayer(const testBNDesc& pm) {
  TestConfig config;
  config.layerConfig.set_type("mkldnn_batch_norm");
  config.layerConfig.set_size(pm.ic * pm.ih * pm.iw);
  config.biasSize = pm.ic;
  
  config.inputDefs.push_back({INPUT_DATA, "layer_0",
    /* size of input layer= */ size_t(pm.ic * pm.ih * pm.iw),
    /* size of weight= */ size_t(pm.ic)});
  // each input will has its own weight as param vector
  config.inputDefs.push_back({INPUT_DATA,
    "layer_1_running_mean", 1, size_t(pm.ic)});
  config.inputDefs.back().isStatic = true;
  config.inputDefs.push_back({INPUT_DATA,
    "layer_2_running_var", 1, size_t(pm.ic)});
  config.inputDefs.back().isStatic = true;

  LayerInputConfig* input = config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  ImageConfig* img_conf = input->mutable_image_conf();
  img_conf->set_channels(pm.ic);
  img_conf->set_img_size(pm.ih);

  // TODO(TJ): test both true and false??
  config.layerConfig.set_use_mkldnn_wgt(false);

  // test functionality as batch norm
  TestConfig ref = config;
  ref.layerConfig.set_type("batch_norm");
  std::vector<TestConfig> cfg = {config, ref};
  for (auto useGS: {false, true}) {
    cfg[0].layerConfig.set_use_global_stats(useGS);
    cfg[1].layerConfig.set_use_global_stats(useGS);
    // TODO(TJ): use {0, 1} if AddToMode ready
    for (auto addSize : {0}) {
      config.layerConfig.set_add_size(addSize);
      for (auto bs : {1, pm.bs}) {
        testLayerFunc(cfg, bs);
      }
    }
  }

  // test layer grad
  config.layerConfig.set_active_type("relu");
  // only need to check false when training
  config.layerConfig.set_use_global_stats(false);
  // TODO(TJ): use {0, 1} if AddToMode ready
  for (auto addSize : {0}) {
    config.layerConfig.set_add_size(addSize);
    testLayerGrad(config, "mkldnn_batch_norm", pm.bs, false, false,
                /* useWeight */ true);
  }
}

TEST(MkldnnLayer, BatchNormLayer) {
  testBatchNormLayer({64, 10, 16, 16});
}

struct testAddtoDesc {
  int nInputs;
  int bs;
  int ic;
  int ih, iw;
};

void testAddtoLayer(const testAddtoDesc& pm) {
  CHECK_GE(pm.nInputs, 1);
  TestConfig config;
  size_t layerSize = pm.ic * pm.ih * pm.iw;
  config.layerConfig.set_type("mkldnn_addto");
  config.layerConfig.set_use_mkldnn_wgt(false);  // has no wgt
  config.layerConfig.set_size(layerSize);
  for (int idx = 0; idx < pm.nInputs; ++idx) {
    std::stringstream ss;
    ss << "layer_" << idx;
    config.inputDefs.push_back({INPUT_DATA, ss.str(), layerSize, 0});
    LayerInputConfig* input = config.layerConfig.add_inputs();
    ImageConfig* img_conf = input->mutable_image_conf();
    img_conf->set_channels(pm.ic);
    // if image size == 1, do not treat as an image
    if (pm.iw > 1) {
      CHECK_EQ(pm.iw, pm.ih);
      img_conf->set_img_size(pm.iw);
    }
  }

  // TODO(TJ): check bias both false and true
  for (auto biasSize : {0, int(layerSize)}) {
    config.biasSize = biasSize;
    // only test functionality, no need to check grad
    TestConfig ref = config;
    ref.layerConfig.set_type("addto");
    std::vector<TestConfig> cfg = {config, ref};

    // TODO(TJ): use {0, 1} if AddToMode ready
    for (auto addSize : {0}) {
      config.layerConfig.set_add_size(addSize);
      for (auto bs : {1, pm.bs}) {
        testLayerFunc(cfg, bs);
      }
    }
  }
}


TEST(MkldnnLayer, AddtoLayer) {
  testAddtoLayer({3, 64, 128, 1, 1});
  testAddtoLayer({2, 64, 100, 8, 8});
  testAddtoLayer({1, 64, 50, 14, 14});
}

struct testConcatDesc {
  size_t axis;
  std::vector<std::vector<int>> inputs;
};

void testConcatLayer(const testConcatDesc& pm) {
  const int N =0, C = 1, H = 2, W =3;
  CHECK_GE(pm.inputs.size(), 2) << "at least two inputs";
  CHECK(pm.axis == 0 || pm.axis == 1);
  CHECK_EQ(pm.axis, 1) << "only support concat channel yet";
  const std::vector<std::vector<int>>& in = pm.inputs;
  std::vector<int> out(in[0]);  // nchw

  out[C] = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    out[C] += in[i][C];
  }

  TestConfig config;
  size_t layerSize = out[C] * out[H] * out[W];
  config.layerConfig.set_type("mkldnn_concat");
  config.layerConfig.set_use_mkldnn_wgt(false);  // has no wgt
  config.layerConfig.set_size(layerSize);
  for (size_t idx = 0; idx < in.size(); ++idx) {
    size_t in_size = in[idx][C] * in[idx][H] * in[idx][W];
    std::stringstream ss;
    ss << "layer_" << idx;
    config.inputDefs.push_back({INPUT_DATA, ss.str(), in_size, 0});
    LayerInputConfig* input = config.layerConfig.add_inputs();
    ImageConfig* img_conf = input->mutable_image_conf();
    img_conf->set_channels(in[idx][C]);
    CHECK_EQ(in[idx][W], in[idx][H]);
    img_conf->set_img_size(in[idx][W]);
  }

  // only test functionality, no need to check grad
  TestConfig ref = config;
  ref.layerConfig.set_type("concat");
  std::vector<TestConfig> cfg = {config, ref};

  // TODO(TJ): use {0, 1} if AddToMode ready
  for (auto addSize : {0}) {
    config.layerConfig.set_add_size(addSize);
    for (auto bs : {1, out[N]}) {
      testLayerFunc(cfg, bs);
    }
  }
}

TEST(MkldnnLayer, ConcatLayer) {
  testConcatLayer({1, {{64, 128, 1, 1}, {64, 32, 1, 1}, {64, 64, 1, 1}}});
  testConcatLayer({1, {{32, 100, 8, 8}, {32, 10, 8, 8}}});
}

struct testActDesc {
  int bs;
  int sz;
};

void testActivation(std::string act, const testActDesc& pm) {
  const std::string dnn("mkldnn_");
  CHECK_EQ(act.compare(0, dnn.length(), dnn), 0);
  LOG(INFO) << "test activation: " << act;

  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("addto");
  config.layerConfig.set_size(pm.sz);
  config.layerConfig.set_active_type(act);
  config.inputDefs.push_back({INPUT_DATA, "layer_0", size_t(pm.sz), 0});
  config.layerConfig.add_inputs();
  // test gradient
  testLayerGrad(config, act + "_activation", pm.bs, false, false, true);

  // test functionality
  TestConfig ref = config;
  act.erase(0, 7);
  ref.layerConfig.set_active_type(act);
  std::vector<TestConfig> cfg = {config, ref};
  // TODO(TJ): use {0, 1} if AddToMode ready
  for (auto addSize : {0}) {
    config.layerConfig.set_add_size(addSize);
    for (auto bs : {1, pm.bs}) {
      testLayerFunc(cfg, bs);
    }
  }
}

TEST(MkldnnLayer, activations) {
  testActivation("mkldnn_softmax", {100, 1000});
  testActivation("mkldnn_relu", {100, 1000});
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}


