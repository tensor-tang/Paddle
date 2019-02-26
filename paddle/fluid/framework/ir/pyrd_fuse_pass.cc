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

#include "paddle/fluid/framework/ir/pyrd_fuse_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> FusePyrdPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("fused_pyrd", graph.get());

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("fused_pyrd/x")
                ->AsInput()
                ->assert_is_op_output("feed", "Out");
  patterns::FusedPyrd fuse_pyrd_pattern(gpd.mutable_pattern(), "fused_pyrd");
  fuse_pyrd_pattern(x);

  int found_fp_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle Fuse Pyramid fuse";
    //GET_IR_NODE_FROM_SUBGRAPH(emb1, embwithvsum, fuse_pyrd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(emb2, emb2, fuse_pyrd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out1, Out1, fuse_pyrd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out2, Out2, fuse_pyrd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out3, Out3, fuse_pyrd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out4, Out4, fuse_pyrd_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(emb1, emb1, fuse_pyrd_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(fused_hash1, fused_hash1, fuse_pyrd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fused_hash2, fused_hash2, fuse_pyrd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fused_hash3, fused_hash3, fuse_pyrd_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(fuse_emb_seq_pool, fused_emb_seq_pool, fuse_pyrd_pattern);
    


    // Create an FC Node.
    OpDesc desc;
    std::string fp_x_in = subgraph.at(x)->Name();
    std::string fp_emb1_in = emb1->Name();
    std::string fp_emb2_in = emb2->Name();
    std::string fp_out1 = out1->Name();
    std::string fp_out2 = out2->Name();
    std::string fp_out3 = out3->Name();
    std::string fp_out4 = out4->Name();

    //mod_by.push_back(boost::get<int>(hash3->Op()->GetAttr("mod_by")));


    desc.SetInput("Input", std::vector<std::string>({fp_x_in}));
    desc.SetInput("W1", std::vector<std::string>({fp_emb1_in}));
    desc.SetInput("W2", std::vector<std::string>({fp_emb2_in}));
    desc.SetOutput("Out1", std::vector<std::string>({fp_out1}));
    desc.SetOutput("Out2", std::vector<std::string>({fp_out2}));
    desc.SetOutput("Out3", std::vector<std::string>({fp_out3}));
    desc.SetOutput("Out4", std::vector<std::string>({fp_out4}));

    std::vector<int> win_size;
    std::vector<int> mod_by;
    std::vector<int> num_hash;

    win_size.push_back(boost::get<int>(fused_hash1->Op()->GetAttr("win_size")));
    win_size.push_back(boost::get<int>(fused_hash2->Op()->GetAttr("win_size")));
    win_size.push_back(boost::get<int>(fused_hash3->Op()->GetAttr("win_size")));

    mod_by.push_back(boost::get<int>(fused_hash1->Op()->GetAttr("mod_by")));
    mod_by.push_back(boost::get<int>(fused_hash2->Op()->GetAttr("mod_by")));
    mod_by.push_back(boost::get<int>(fused_hash3->Op()->GetAttr("mod_by")));

    num_hash.push_back(boost::get<int>(fused_hash1->Op()->GetAttr("num_hash")));
    num_hash.push_back(boost::get<int>(fused_hash2->Op()->GetAttr("num_hash")));
    num_hash.push_back(boost::get<int>(fused_hash3->Op()->GetAttr("num_hash")));

    desc.SetAttr("num_hash", num_hash);
    desc.SetAttr("mod_by", mod_by);
    desc.SetAttr("win_size", win_size);
    desc.SetAttr("pad_value", fused_hash1->Op()->GetAttr("pad_value"));
    desc.SetAttr("combiner", "SUM");
    
    desc.SetType("fuse_pyrd");
    auto fp_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    GraphSafeRemoveNodes(graph.get(), {fused_hash1, fused_hash2, fused_hash3, fuse_emb_seq_pool});

    PADDLE_ENFORCE(subgraph.count(x));
    IR_NODE_LINK_TO(subgraph.at(x), fp_node);
    IR_NODE_LINK_TO(emb1, fp_node);
    IR_NODE_LINK_TO(emb2, fp_node);
    IR_NODE_LINK_TO(fp_node, out1);
    IR_NODE_LINK_TO(fp_node, out2);
    IR_NODE_LINK_TO(fp_node, out3);
    IR_NODE_LINK_TO(fp_node, out4);


    found_fp_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_fp_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(pyrd_fuse_pass, paddle::framework::ir::FusePyrdPass);
