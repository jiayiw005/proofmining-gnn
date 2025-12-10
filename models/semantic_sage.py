import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class SemanticSAGEConv(MessagePassing):
    """
    GraphSAGE-style convolution with semantic similarity weighting.
    Neighbor messages are weighted by cosine similarity between
    state semantic embeddings.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")  
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, state_sem):
        """
        x           : [N, in_dim]
        state_sem   : [N, sem_dim]
        edge_index  : [2, E]
        """
        return self.propagate(edge_index, x=x, state_sem=state_sem)

    def message(self, x_j, state_sem_i, state_sem_j):
        """
        x_j         : neighbor features
        state_sem_i : target node semantic
        state_sem_j : neighbor semantic
        """

        # cosine similarity in semantic space
        sim = F.cosine_similarity(state_sem_i, state_sem_j, dim=-1)

        # positive weighting via exp
        weight = torch.exp(sim).unsqueeze(-1)

        # weighted message
        return weight * self.lin(x_j)
