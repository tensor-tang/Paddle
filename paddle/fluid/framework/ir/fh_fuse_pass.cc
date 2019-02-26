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
    //GET_IR_NODE_FROM_SUBGRAPH(emb1, embwithvsum, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(emb2, pyramidhash_emb, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out, Out, fused_hash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(seq_ent, seq_ent, fused_hash_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(hash, hash, fused_hash_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(fuse_emb_seq_pool, fused_emb_seq_pool, fused_hash_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(seq_ent_out, seq_ent_out, fused_hash_pattern);
    
    GET_IR_NODE_FROM_SUBGRAPH(hash_out, hash_out, fused_hash_pattern);



    // Create an FC Node.
    OpDesc desc;
    std::string fh_x_in = subgraph.at(x)->Name();
    //std::string fh_emb1_in = emb1->Name();
    std::string fh_emb2_in = emb2->Name();
    std::string fh_out = out->Name();
    
    //mod_by.push_back(boost::get<int>(hash3->Op()->GetAttr("mod_by")));


    desc.SetInput("Input", std::vector<std::string>({fh_x_in}));
    desc.SetInput("W", std::vector<std::string>({fh_emb2_in}));
    desc.SetOutput("Out", std::vector<std::string>({fh_out}));
    desc.SetAttr("num_hash", hash->Op()->GetAttr("num_hash"));
    desc.SetAttr("mod_by", hash->Op()->GetAttr("mod_by"));
    desc.SetAttr("win_size", seq_ent->Op()->GetAttr("win_size"));
    desc.SetAttr("pad_value", seq_ent->Op()->GetAttr("pad_value"));
    desc.SetAttr("combiner", "SUM");
    
    desc.SetType("fused_hash");
    auto fh_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    GraphSafeRemoveNodes(graph.get(), {seq_ent,  hash, fuse_emb_seq_pool,
                                       seq_ent_out,hash_out});

    PADDLE_ENFORCE(subgraph.count(x));
    IR_NODE_LINK_TO(subgraph.at(x), fh_node);
    // IR_NODE_LINK_TO(emb1, fh_node);
    IR_NODE_LINK_TO(emb2, fh_node);
    IR_NODE_LINK_TO(fh_node, out);


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
