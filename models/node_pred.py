import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import os
from datetime import datetime

from data_loader import load_train_data, load_val_data
from gnn import ProofGNN_NextTactic


def make_run_dir(base="runs"):
    os.makedirs(base, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"tactic_run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def train_next_tactic(batch_size, epochs, lr, save_every_epoch=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    run_dir = make_run_dir()
    print(f"Saving outputs to: {run_dir}")

    with open("data/cache/tactic_vocab.json") as f:
        TACTICS = json.load(f)
    num_tactics = len(TACTICS)
    num_node_types = 3  # (state, tactic, premise)

    print(f"Loaded {num_tactics} tactics.")

    # Save a snapshot of vocab for reproducibility
    with open(os.path.join(run_dir, "tactic_vocab.json"), "w") as f:
        json.dump(TACTICS, f, indent=2)

    train_loader = load_train_data(batch_size=batch_size, path="data/pyg/train")
    val_loader   = load_val_data(batch_size=batch_size, path="data/pyg/val")

    model = ProofGNN_NextTactic(
        num_node_types=num_node_types,
        num_tactics=num_tactics,
        type_embed_dim=32,
        tactic_embed_dim=64,
        hidden_dim=512,
    ).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    log = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "num_tactics": num_tactics,
            "num_node_types": num_node_types,
        }
    }

    best_val_acc = -1.0
    best_model_path = os.path.join(run_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            targets = batch.target_tactic

            loss = criterion(logits, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        log["train_loss"].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                batch = batch.to(device)

                logits = model(batch)
                targets = batch.target_tactic

                loss = criterion(logits, targets)
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.numel()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = correct / total if total > 0 else 0.0

        log["val_loss"].append(avg_val_loss)
        log["val_acc"].append(val_acc)

        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}  |  "
              f"Val Loss: {avg_val_loss:.4f}  |  Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  → New best model saved at epoch {epoch} (val_acc={val_acc:.4f})")

        if save_every_epoch:
            epoch_path = os.path.join(run_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save(model.state_dict(), epoch_path)

    final_model_path = os.path.join(run_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)

    with open(os.path.join(run_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print("\nTraining complete.")
    print(f"Best model saved to:  {best_model_path}")
    print(f"Final model saved to: {final_model_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_every_epoch", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_next_tactic(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        save_every_epoch=args.save_every_epoch,
    )
