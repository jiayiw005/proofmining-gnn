# test_next_tactic.py
import json
import torch
from tqdm import tqdm
import argparse
import os

from data_loader import load_test_data
from gnn import ProofGNN_NextTactic


def test(model_path, run_dir, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    vocab_path = os.path.join(run_dir, "tactic_vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Could not find {vocab_path}. "
            "The test script must load the SAME vocab used in training."
        )

    with open(vocab_path) as f:
        TACTICS = json.load(f)

    num_tactics = len(TACTICS)
    num_node_types = 3  # (state, tactic, premise)

    print(f"Loaded {num_tactics} tactics from {vocab_path}.")

    test_loader = load_test_data(batch_size=batch_size, path="data/pyg/test")
    print(f"Loaded {len(test_loader.dataset)} test graphs.")

    model = ProofGNN_NextTactic(
        num_node_types=num_node_types,
        num_tactics=num_tactics,
        type_embed_dim=32,
        tactic_embed_dim=64,
        hidden_dim=512,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from: {model_path}")

    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)

            logits = model(batch) 
            preds = logits.argmax(dim=-1)
            targets = batch.target_tactic

            all_preds.append(preds.item())
            all_targets.append(targets.item())

            correct += (preds == targets).sum().item()
            total += targets.numel()

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nTest Accuracy: {accuracy:.4f}")

    return accuracy, all_preds, all_targets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="Path to best_model.pt or final_model.pt")
    p.add_argument("--run_dir", type=str, required=True,
                   help="Run directory containing tactic_vocab.json")
    p.add_argument("--batch_size", default=1, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(
        model_path=args.model,
        run_dir=args.run_dir,
        batch_size=args.batch_size,
    )
