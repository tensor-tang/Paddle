//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/fused/fused_enum_hash_emd_pool_op.h"
#include <map>

namespace paddle {
namespace operators {

class FusedEnumHashEmdPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("X"),
        "Input(X) of FusedEnumHashEmdPool operator should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("W0"),
        "Input(W1) of FusedEnumHashEmdPool operator should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("W1"),
        "Input(W2) of FusedEnumHashEmdPool operator should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out0"),
        "Output(Out1) of FusedEnumHashEmdPool operator should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out1"),
        "Output(Out2) of FusedEnumHashEmdPool operator should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out2"),
        "Output(Out3) of FusedEnumHashEmdPool operator should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out3"),
        "Output(Out4) of FusedEnumHashEmdPool operator should not be null.");

    const auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(
        x_dims.size(), 2,
        "Input(X) of FusedEnumHashEmdPool operator's rank should be 2.");
    PADDLE_ENFORCE_EQ(x_dims[1], 1,
                      "Input(X) of FusedEnumHashEmdPool operator's 2nd "
                      "dimension should be 1.");

    std::map<std::string, std::string> map;
    map.insert(std::make_pair("Out0", "W0"));
    map.insert(std::make_pair("Out1", "W1"));
    map.insert(std::make_pair("Out2", "W1"));
    map.insert(std::make_pair("Out3", "W1"));
    for (auto it = map.begin(); it != map.end(); it++) {
      std::string table = it->second;
      std::string output = it->first;
      auto table_dims = ctx->GetInputDim(table);
      PADDLE_ENFORCE_EQ(table_dims.size(), 2);

      int64_t last_dim = table_dims[1];
      std::string input_name = "X";
      // auto input_dims = ctx->GetInputDim(input_name);
      // for (int i = 1; i != input_dims.size(); ++i) {
      //  last_dim *= input_dims[i];
      if (table != "W0") {
        const auto num_hash = ctx->Attrs().Get<std::vector<int>>("num_hash");
        auto idx = std::distance(map.begin(), it);
        idx = std::min(static_cast<int>(num_hash.size() - 1),
                       std::max(0, static_cast<int>(idx - 1)));
        last_dim *= num_hash[idx];
      }

      if (ctx->IsRuntime()) {
        framework::Variable* ids_var = boost::get<framework::Variable*>(
            ctx->GetInputVarPtrs(input_name)[0]);
        const auto& ids_lod = ids_var->Get<LoDTensor>().lod();

        // in run time, the LoD of ids must be 1
        PADDLE_ENFORCE(ids_lod.size(), 1u,
                       "The LoD level of Input(Ids) must be 1");
        PADDLE_ENFORCE_GE(ids_lod[0].size(), 1u, "The LoD could NOT be empty");

        int64_t batch_size = ids_lod[0].size() - 1;

        // in run time, the shape from Ids -> output
        // should be [seq_length, 1] -> [batch_size, embedding_size]
        ctx->SetOutputDim(output, framework::make_ddim({batch_size, last_dim}));
      } else {
        // in compile time, the lod level of ids must be 1
        framework::VarDesc* ids_desc = boost::get<framework::VarDesc*>(
            ctx->GetInputVarPtrs(input_name)[0]);
        PADDLE_ENFORCE_EQ(ids_desc->GetLoDLevel(), 1);

        // in compile time, the shape from Ids -> output
        // should be [-1, 1] -> [-1, embedding_size]
        ctx->SetOutputDim(output, framework::make_ddim({-1, last_dim}));
      }
    }
  }
};

class FusedEnumHashEmdPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(2-D LoDTensor with the 2nd dimension equal to 1) "
             "Input LoDTensor of SequenceEnumerate operator.");
    AddInput("W0",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddInput("W1",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddOutput("Out0", "The lookup results, which have the same type as W1.");
    AddOutput("Out1", "The lookup results, which have the same type as W2.");
    AddOutput("Out2", "The lookup results, which have the same type as W2.");
    AddOutput("Out3", "The lookup results, which have the same type as W2.");
    AddAttr<std::vector<int>>("win_size",
                              "(vector<int>) "
                              "the length of each output along the "
                              "specified axis.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("pad_value", "(int) The enumerate sequence padding value.")
        .SetDefault(0);
    AddAttr<std::vector<int>>("num_hash", "")
        .SetDefault(std::vector<int>{
            1,
        });
    AddAttr<std::vector<int>>("mod_by", "")
        .SetDefault(std::vector<int>{
            100000,
        });
    AddComment(R"DOC(
Fused_enum_hash_emd_pool Operator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fused_enum_hash_emd_pool,
                             ops::FusedEnumHashEmdPoolOp,
                             ops::FusedEnumHashEmdPoolOpMaker);
REGISTER_OP_CPU_KERNEL(
    fused_enum_hash_emd_pool,
    ops::FusedEnumHashEmdPoolKernel<paddle::platform::CPUDeviceContext,
                                    int32_t>,
    ops::FusedEnumHashEmdPoolKernel<paddle::platform::CPUDeviceContext,
                                    int64_t>);
