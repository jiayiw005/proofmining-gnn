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
    Clean tactic-only GNN:
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
        dropout=0.2
    ):
        super().__init__()

        self.num_tactics = num_tactics

        # Node type embedding (state/tactic/premise)
        self.type_emb = nn.Embedding(
            num_node_types, 
            type_embed_dim
        )

        # Tactic embedding:
        #   node_tactic_id = -1 for non tactic nodes
        #   shift by +1 to map:
        #       -1 → 0
        #       0 → 1
        #       ...
        #       num_tactics-1 → num_tactics
        self.tactic_emb = nn.Embedding(
            num_tactics + 1, # extra slot for "no tactic"
            tactic_embed_dim
        )



        input_dim = type_embed_dim + tactic_embed_dim
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_tactics),
        )

    def forward(self, data):
        node_type = data.node_type
        node_tactic_id = data.node_tactic_id

        # type embedding
        t_type = self.type_emb(node_type)

        # tactic embedding with shift
        shifted = torch.clamp(node_tactic_id + 1, min=0, max=self.num_tactics)
        t_tactic = self.tactic_emb(shifted)

        x = torch.cat([t_type, t_tactic], dim=-1)

        x = self.conv1(x, data.edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, data.edge_index)
        x = self.dropout(x)

        graph_repr = global_mean_pool(x, data.batch)

        # predict next tactic
        logits = self.classifier(graph_repr)
        return logits