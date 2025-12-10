import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GlobalAttention


class ProofGNN_NextTactic(nn.Module):
    """
    Stable Semantic Late-Fusion Proof GNN

      - node_type      ∈ {0,1,2}
      - node_tactic_id ∈ {-1, 0..num_tactics-1}
      - state_lm_id    ∈ {-1, 0..N_state_strings-1}
      - state_lm_bank  ∈ R^{N_state_strings × lm_dim}
      - target_tactic ∈ [0..num_tactics-1]
    """

    def __init__(
        self,
        num_node_types,
        num_tactics,
        state_lm_dim,
        type_embed_dim=32,
        tactic_embed_dim=64,
        state_embed_dim=128, 
        hidden_dim=512,
        dropout=0.2,
        state_lm_bank=None,
    ):
        super().__init__()

        if state_lm_bank is None:
            raise ValueError("state_lm_bank must be provided")

        self.num_tactics = num_tactics

        # structural embeddings
        self.type_emb = nn.Embedding(num_node_types, type_embed_dim)
        self.tactic_emb = nn.Embedding(num_tactics + 1, tactic_embed_dim)

        # semantic projections
        self.state_proj = nn.Sequential(
            nn.Linear(state_lm_dim, state_embed_dim),
            nn.ReLU(),
            nn.LayerNorm(state_embed_dim)
        )
        self.register_buffer("state_lm_bank", state_lm_bank)

        # gnn over structure only
        gnn_input_dim = type_embed_dim + tactic_embed_dim

        self.conv1 = SAGEConv(gnn_input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # attention pooling over structural embeddings
        self.att_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )

        # final classifier (structure + semantics)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + state_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tactics),
        )

    # forward pass
    def forward(
        self, 
        data, 
        remove_tactic_feature=False, 
        remove_node_type=False,
        tactic_dropout_p=0.0,
        training=True,
    ):
        node_type = data.node_type
        node_tactic_id = data.node_tactic_id
        state_lm_id = data.state_lm_id
        

        # ablation: remove node type
        if remove_node_type:
            node_type = torch.zeros_like(node_type)

        # structural embeddings
        t_type = self.type_emb(node_type)
        shifted = torch.clamp(node_tactic_id + 1, min=0, max=self.num_tactics)
        t_tactic = self.tactic_emb(shifted)

        # tactic id dropout (training only)
        if training and tactic_dropout_p > 0:
            drop_mask = torch.rand_like(shifted.float()) < tactic_dropout_p
            t_tactic = torch.where(
                drop_mask.unsqueeze(-1),
                torch.zeros_like(t_tactic),
                t_tactic
            )

        # ablation: remove tactic feature
        if remove_tactic_feature:
            t_tactic = torch.zeros_like(t_tactic)


        x_struct = torch.cat([t_type, t_tactic], dim=-1)

        # gnn over structure only
        x = self.conv1(x_struct, data.edge_index).relu()
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index).relu()
        x = self.dropout(x)

        # attention pooling over structural embeddings
        graph_struct = self.att_pool(x, data.batch)

        # graph-level semantic aggregation (state only)
        device = node_type.device
        B = data.batch.max().item() + 1
        state_sem_graph = torch.zeros(B, self.state_proj[0].out_features, device=device)

        mask = state_lm_id >= 0
        if mask.any():
            lm_vecs = self.state_lm_bank[state_lm_id[mask]]
            projected = self.state_proj(lm_vecs)

            # aggregate per graph
            batch_ids = data.batch[mask]
            state_sem_graph.index_add_(0, batch_ids, projected)

            # normalize by count per graph
            counts = torch.bincount(batch_ids, minlength=B).unsqueeze(1)
            state_sem_graph = state_sem_graph / (counts + 1e-6)

        # final fusion & prediction
        graph_repr = torch.cat([graph_struct, state_sem_graph], dim=-1)
        logits = self.classifier(graph_repr)
        return logits