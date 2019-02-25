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

#include "paddle/fluid/framework/ir/fh_fuse_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> FusedHashPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("fused_hash", graph.get());

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("fused_hash/x")
                ->AsInput()
                ->assert_is_op_output("feed", "Out");
  patterns::FusedHash fused_hash_pattern(gpd.mutable_pattern(), "fused_hash");
  fused_hash_pattern(x);

  int found_fh_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle FC fuse";
    GET_IR_NODE_FROM_SUBGRAPH(emb1, embwithvsum, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(emb2, pyramidhash_emb, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out1, Out1, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out2, Out2, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out3, Out3, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out4, Out4, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(seq_ent1, seq_ent1, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(seq_ent2, seq_ent2, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(seq_ent3, seq_ent3, fused_hash_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(hash1, hash1, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(hash2, hash2, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(hash3, hash3, fused_hash_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(fuse_emb_seq_pool1, fused_emb_seq_pool, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_emb_seq_pool2, fused_emb_seq_pool2, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_emb_seq_pool3, fused_emb_seq_pool3, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_emb_seq_pool4, fused_emb_seq_pool4, fused_hash_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(seq_ent1_out, seq_ent1_out, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(seq_ent2_out, seq_ent2_out, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(seq_ent3_out, seq_ent3_out, fused_hash_pattern);
    
    GET_IR_NODE_FROM_SUBGRAPH(hash1_out, hash1_out, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(hash2_out, hash1_out, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(hash3_out, hash1_out, fused_hash_pattern);



    // Create an FC Node.
    OpDesc desc;
    std::string fh_x_in = subgraph.at(x)->Name();
    std::string fh_emb1_in = emb1->Name();
    std::string fh_emb2_in = emb2->Name();
    std::string fh_out1 = out1->Name();
    std::string fh_out2 = out2->Name();
    std::string fh_out3 = out3->Name();
    std::string fh_out4 = out4->Name();
    
    std::vector<int> win_size;
    std::vector<int> num_hash;
    std::vector<int> mod_by;

    win_size.push_back(boost::get<int>(seq_ent1->Op()->GetAttr("win_size")));
    win_size.push_back(boost::get<int>(seq_ent2->Op()->GetAttr("win_size")));
    win_size.push_back(boost::get<int>(seq_ent3->Op()->GetAttr("win_size")));

    num_hash.push_back(boost::get<int>(hash1->Op()->GetAttr("num_hash")));
    num_hash.push_back(boost::get<int>(hash2->Op()->GetAttr("num_hash")));
    num_hash.push_back(boost::get<int>(hash3->Op()->GetAttr("num_hash")));

    mod_by.push_back(boost::get<int>(hash1->Op()->GetAttr("mod_by")));
    mod_by.push_back(boost::get<int>(hash2->Op()->GetAttr("mod_by")));
    mod_by.push_back(boost::get<int>(hash3->Op()->GetAttr("mod_by")));


    desc.SetInput("Input", std::vector<std::string>({fh_x_in}));
    desc.SetInput("W1", std::vector<std::string>({fh_emb1_in}));
    desc.SetInput("W2", std::vector<std::string>({fh_emb2_in}));
    desc.SetOutput("Out1", std::vector<std::string>({fh_out1}));
    desc.SetOutput("Out2", std::vector<std::string>({fh_out2}));
    desc.SetOutput("Out3", std::vector<std::string>({fh_out3}));
    desc.SetOutput("Out4", std::vector<std::string>({fh_out4}));
    desc.SetAttr("num_hash", num_hash);
    desc.SetAttr("mod_by", mod_by);
    desc.SetAttr("win_size", win_size);
    desc.SetAttr("pad_value", seq_ent1->Op()->GetAttr("pad_value"));
    desc.SetAttr("pooltype", "SUM");

    desc.SetType("fused_hash");
    auto fh_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    GraphSafeRemoveNodes(graph.get(), {seq_ent1, seq_ent2,seq_ent3, hash1,hash2,hash3, fuse_emb_seq_pool1,
                                       fuse_emb_seq_pool2,fuse_emb_seq_pool3,fuse_emb_seq_pool4});

    PADDLE_ENFORCE(subgraph.count(x));
    IR_NODE_LINK_TO(subgraph.at(x), fh_node);
    IR_NODE_LINK_TO(emb1, fh_node);
    IR_NODE_LINK_TO(emb2, fh_node);
    IR_NODE_LINK_TO(fh_node, out1);
    IR_NODE_LINK_TO(fh_node, out2);
    IR_NODE_LINK_TO(fh_node, out3);
    IR_NODE_LINK_TO(fh_node, out4);


    found_fh_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_fh_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fh_fuse_pass, paddle::framework::ir::FusedHashPass);
