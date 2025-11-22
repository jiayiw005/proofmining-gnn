import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class ProofGNN(nn.Module):
    def __init__(self, num_entities, num_node_types=3, embed_dim=128, hidden_dim=128):
        super().__init__()

        # learned embeddings for both symbolic identity + type
        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        self.type_emb   = nn.Embedding(num_node_types, embed_dim)

        # core graph layers
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # classification head (predict node type)
        self.classifier = nn.Linear(hidden_dim, num_node_types)

    def forward(self, data):
        # Base representation: entity embedding + type embedding
        x = self.entity_emb(data.entity) + self.type_emb(data.node_type)

        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index)

        logits = self.classifier(x)  # shape: [num_nodes, num_node_types]
        return logits
