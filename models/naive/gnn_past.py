import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_softmax, scatter_add
from semantic_sage import SemanticSAGEConv



class SemanticAwareConv(nn.Module):
    """
    message passing weighted by semantic similarity.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_src = nn.Linear(in_dim, out_dim)
        self.lin_dst = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, semantic_vec):
        """
        x:             [N, in_dim]
        edge_index:    [2, E]
        semantic_vec:  [N, sem_dim]
        """
        src, dst = edge_index 

        x_j = self.lin_src(x[src])
        x_i = self.lin_dst(x[dst]) 

        # semantic similarity weighting
        sim = F.cosine_similarity(
            semantic_vec[src],
            semantic_vec[dst],
            dim=-1,
            eps=1e-8
        )
        
        attn = scatter_softmax(sim, dst)

        msg = attn.unsqueeze(-1) * x_j

        out = scatter_add(msg, dst, dim=0, dim_size=x.size(0))

        return out


class ProofGNN_NextTactic(nn.Module):
    """
    GNN with semantic-aware message passing
    """

    def __init__(
        self,
        num_node_types,
        num_tactics,
        state_lm_dim,
        type_embed_dim=32,
        tactic_embed_dim=64,
        state_embed_dim=64,
        hidden_dim=512,
        dropout=0.2,
        state_lm_bank=None,
    ):
        super().__init__()

        self.num_tactics = num_tactics

        # embeddings
        self.type_emb = nn.Embedding(num_node_types, type_embed_dim)
        self.tactic_emb = nn.Embedding(num_tactics + 1, tactic_embed_dim)

        if state_lm_bank is None:
            raise ValueError("must provide state_lm_bank")

        self.register_buffer("state_lm_bank", state_lm_bank)
        self.state_proj = nn.Linear(state_lm_dim, state_embed_dim)

        input_dim = type_embed_dim + tactic_embed_dim + state_embed_dim

        # semantic-aware convolutions
        self.conv1 = SemanticSAGEConv(input_dim, hidden_dim)
        self.conv2 = SemanticSAGEConv(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tactics),
        )

    def forward(self, data, remove_tactic_feature=False, remove_node_type=False):

        node_type = data.node_type
        node_tactic_id = data.node_tactic_id
        state_lm_id = data.state_lm_id

        if remove_node_type:
            node_type = torch.zeros_like(node_type)

        t_type = self.type_emb(node_type)

        shifted = torch.clamp(node_tactic_id + 1, 0, self.num_tactics)
        if remove_tactic_feature:
            t_tactic = torch.zeros_like(self.tactic_emb(shifted))
        else:
            t_tactic = self.tactic_emb(shifted)

        # semantic state embeddings
        N = node_type.size(0)
        device = node_type.device
        state_sem = torch.zeros(N, self.state_proj.out_features, device=device)

        mask = state_lm_id >= 0
        if mask.any():
            lm_vecs = self.state_lm_bank[state_lm_id[mask]]
            state_sem[mask] = self.state_proj(lm_vecs)

        # concatenate full feature
        x = torch.cat([t_type, t_tactic, state_sem], dim=-1)

        # semantic message passing
        x = self.conv1(x, data.edge_index, state_sem).relu()
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index, state_sem).relu()
        x = self.dropout(x)

        # graph pooling
        graph_repr = global_mean_pool(x, data.batch)

        return self.classifier(graph_repr)
