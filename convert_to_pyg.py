import os
import pickle
from pathlib import Path
from collections import defaultdict

import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx


# ==========
# GLOBAL MAPPINGS (persistent vocabularies)
# For real experiments, save these as files so IDs are consistent across runs.
# ==========

entity2id = defaultdict(lambda: len(entity2id))
type2id   = {"state": 0, "tactic": 1, "premise": 2}
relation2id = {"applies": 0, "yields": 1, "used_in": 2}


# ==========
# NODE FEATURE EXTRACTION
# ==========

def extract_node_features(G):
    """
    Return tensor of entity IDs, tensor of node types, and list of node names in order.
    Node ordering follows NetworkX node order.
    """
    nodes = list(G.nodes())
    entity_ids = []
    type_ids   = []

    for n in nodes:
        node_type = G.nodes[n].get("type", "unknown")
        type_ids.append(type2id.get(node_type, 0))
        
        # use raw string as unique symbolic entity
        entity_ids.append(entity2id[n])

    return (
        torch.tensor(entity_ids, dtype=torch.long),
        torch.tensor(type_ids, dtype=torch.long),
        nodes,
    )


# ==========
# EDGE FEATURE EXTRACTION
# ==========

def extract_edge_features(G, nodes):
    """
    Convert edge relations to integer relation IDs and return tensor aligned with edge_index.
    """
    edge_type_ids = []

    for u, v, data in G.edges(data=True):
        rel = data.get("relation", None)
        if rel not in relation2id:
            relation2id[rel] = len(relation2id)
        edge_type_ids.append(relation2id[rel])

    return torch.tensor(edge_type_ids, dtype=torch.long)


# ==========
# MAIN CONVERSION FUNCTION
# ==========

def convert_graph(path):
    """Load .gpickle → PyG Data object."""
    with open(path, "rb") as f:
        G = pickle.load(f)

    # Convert edges + structure
    data = from_networkx(G)

    # Add node features
    entity_ids, type_ids, nodes = extract_node_features(G)
    data.entity = entity_ids
    data.node_type = type_ids

    # Add edge types
    data.edge_type = extract_edge_features(G, nodes)

    return data


# ==========
# BATCH CONVERSION PIPELINE
# ==========

def convert_all(input_dir="data/processed/val", output_dir="data/pyg_graphs/val"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for fp in input_dir.glob("*.gpickle"):
        try:
            data = convert_graph(fp)
            out = output_dir / (fp.stem + ".pt")
            torch.save(data, out)
            print(f"[OK] Converted {fp.name} → {out}")
        except Exception as e:
            print(f"[ERROR] {fp.name}: {e}")


if __name__ == "__main__":
    convert_all()
