import torch
from torch_geometric.loader import DataLoader
from pathlib import Path

def load_graphs(path):
    path = Path(path)
    return [torch.load(fp, weights_only=False) for fp in path.glob("*.pt")]


def load_train_data(path="data/pyg/train", batch_size=4):
    graphs = load_graphs(path)
    return DataLoader(graphs, batch_size=batch_size, shuffle=True)


def load_val_data(path="data/pyg/val", batch_size=4):
    graphs = load_graphs(path)
    return DataLoader(graphs, batch_size=batch_size, shuffle=False)


def load_test_data(path="data/pyg/test", batch_size=4):
    graphs = load_graphs(path)
    return DataLoader(graphs, batch_size=batch_size, shuffle=False)
