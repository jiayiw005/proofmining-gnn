import torch
from data_loader import load_test_data
from gnn import ProofGNN


def test():
    test_loader = load_test_data(batch_size=1, path="data/pyg_graphs/test")

    # Vocab must match training, so infer again from train data OR load saved vocab
    num_entities = max(g.entity.max().item() for g in test_loader.dataset) + 1

    model = ProofGNN(num_entities=num_entities, num_node_types=3)
    model.load_state_dict(torch.load("proof_gnn.pt"))
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch)
            pred = logits.argmax(dim=-1)
            correct += (pred == batch.node_type).sum().item()
            total += batch.node_type.numel()

    print(f"Test Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    test()
