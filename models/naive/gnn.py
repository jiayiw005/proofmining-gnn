import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

class ProofGNN_NextNode(nn.Module):
    """
    Node-type prediction
    Predicts node_type for each node.
    """
    def __init__(self, num_entities, num_node_types,
                 embed_dim=128, hidden_dim=128):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, embed_dim, padding_idx=0)

        self.conv1 = SAGEConv(embed_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_node_types)

    def forward(self, data):
        x = self.entity_emb(data.entity)
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index)
        return self.classifier(x)


class ProofGNN_NextTactic(nn.Module):
    """
    tactic-only GNN:
      - node_type ∈ {0,1,2}
      - node_tactic_id ∈ {-1, 0..num_tactics-1}
      - predict graph.target_tactic ∈ [0..num_tactics-1]
    """

    def __init__(
        self,
        num_node_types,
        num_tactics,
        type_embed_dim=32,
        tactic_embed_dim=64,
        hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        self.num_tactics = num_tactics

        # node type embedding
        self.type_emb = nn.Embedding(num_node_types, type_embed_dim)

        # tactic embedding
        self.tactic_emb = nn.Embedding(num_tactics + 1, tactic_embed_dim)

        # GNN layers
        input_dim = type_embed_dim + tactic_embed_dim
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_tactics),
        )
    # fwd
    def forward(self, data, remove_tactic_feature=False, remove_node_type=False):

        node_type = data.node_type 
        node_tactic_id = data.node_tactic_id 

        # ablation
        if remove_node_type:
            node_type = torch.zeros_like(node_type)

        t_type = self.type_emb(node_type)
        shifted = torch.clamp(node_tactic_id + 1, min=0, max=self.num_tactics)

        if remove_tactic_feature:
            t_tactic = torch.zeros_like(self.tactic_emb(shifted))
        else:
            t_tactic = self.tactic_emb(shifted)



        x = torch.cat([t_type, t_tactic], dim=-1)

        x = self.conv1(x, data.edge_index).relu()
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index).relu()
        x = self.dropout(x)

        graph_repr = global_mean_pool(x, data.batch) 
        logits = self.classifier(graph_repr)
        return logits
