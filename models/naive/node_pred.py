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

def train_next_tactic(batch_size, epochs, lr, save_every_epoch=False, model_hparams={}, run_dir_override=None):
    
    # MLP baseline helper
    def make_mlp(num_tactics):
        return nn.Sequential(
            nn.Linear(num_tactics, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_tactics)
        )

    # preps
    remove_tactic_feature = model_hparams.get("remove_tactic_feature", False)
    remove_node_type = model_hparams.get("remove_node_type", False)
    shuffle_edges = model_hparams.get("shuffle_edges", False)
    mlp_bag = model_hparams.get("mlp_bag_of_tactics", False)
    shuffle_tactic_ids = model_hparams.get("shuffle_tactic_ids", False)
    shuffle_targets = model_hparams.get("shuffle_targets", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    run_dir = make_run_dir()
    print(f"Saving outputs to: {run_dir}")

    with open("data/cache/tactic_vocab.json") as f:
        TACTICS = json.load(f)
    num_tactics = len(TACTICS)
    num_node_types = 3

    print(f"Loaded {num_tactics} tactics.")

    with open(os.path.join(run_dir, "tactic_vocab.json"), "w") as f:
        json.dump(TACTICS, f, indent=2)

    train_loader = load_train_data(batch_size=batch_size, path="data/pyg/train")
    val_loader   = load_val_data(batch_size=batch_size, path="data/pyg/val")

    # models (by ablation)
    if mlp_bag:
        model = make_mlp(num_tactics).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        print("Running MLP baseline (no GNN).")
    else:
        model = ProofGNN_NextTactic(
            num_node_types=num_node_types,
            num_tactics=num_tactics,
            type_embed_dim=32,
            tactic_embed_dim=64,
            hidden_dim=512,
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        print(model)

    criterion = nn.CrossEntropyLoss()

    # logging
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
            "hparams": model_hparams
        }
    }

    best_val_acc = -1.0
    best_model_path = os.path.join(run_dir, "best_model.pt")

    # training
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            batch = batch.to(device)
            optimizer.zero_grad()

            # ablations
            if shuffle_edges:
                E = batch.edge_index
                perm = torch.randperm(E.size(1))
                batch.edge_index = E[:, perm]

            if shuffle_targets:
                batch.target_tactic = torch.randint(0, num_tactics, batch.target_tactic.shape).to(device)

            if shuffle_tactic_ids:
                perm = torch.randperm(num_tactics).to(device)
                batch.target_tactic = perm[batch.target_tactic]
                if (batch.node_tactic_id >= 0).any():
                    batch.node_tactic_id = torch.where(
                        batch.node_tactic_id >= 0,
                        perm[batch.node_tactic_id],
                        -1
                    )
                    
            if mlp_bag:
                node_ids = batch.node_tactic_id
                graph_ids = batch.batch
                B = graph_ids.max().item() + 1

                hists = []
                for g in range(B):
                    mask = (graph_ids == g) & (node_ids >= 0)
                    ids = node_ids[mask]
                    hist = torch.bincount(ids, minlength=num_tactics).float()
                    hists.append(hist)

                hists = torch.stack(hists, dim=0).to(device) 
                logits = model(hists)

            # normal GNN
            else:
                logits = model(
                    batch,
                    remove_tactic_feature=remove_tactic_feature,
                    remove_node_type=remove_node_type
                )

            # loss
            targets = batch.target_tactic
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        log["train_loss"].append(avg_train_loss)



        # validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                batch = batch.to(device)

                # same ablations as in training
                if shuffle_edges:
                    E = batch.edge_index
                    perm = torch.randperm(E.size(1))
                    batch.edge_index = E[:, perm]

                if shuffle_targets:
                    batch.target_tactic = torch.randint(0, num_tactics, batch.target_tactic.shape).to(device)

                if shuffle_tactic_ids:
                    perm = torch.randperm(num_tactics).to(device)
                    batch.target_tactic = perm[batch.target_tactic]
                    if (batch.node_tactic_id >= 0).any():
                        batch.node_tactic_id = torch.where(
                            batch.node_tactic_id >= 0,
                            perm[batch.node_tactic_id],
                            -1
                        )

                if mlp_bag:
                    node_ids = batch.node_tactic_id
                    graph_ids = batch.batch
                    B = graph_ids.max().item() + 1

                    hists = []
                    for g in range(B):
                        mask = (graph_ids == g) & (node_ids >= 0)
                        ids = node_ids[mask]
                        hist = torch.bincount(ids, minlength=num_tactics).float()
                        hists.append(hist)

                    hists = torch.stack(hists, dim=0).to(device) 
                    logits = model(hists)
                else:
                    logits = model(
                        batch,
                        remove_tactic_feature=remove_tactic_feature,
                        remove_node_type=remove_node_type
                    )

                targets = batch.target_tactic
                loss = criterion(logits, targets)
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        log["val_loss"].append(avg_val_loss)
        log["val_acc"].append(val_acc)

        print(f"[Epoch {epoch}] Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    # save final
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pt"))
    with open(os.path.join(run_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print("Training complete.")


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
