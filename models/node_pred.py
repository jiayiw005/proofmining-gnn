import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import load_train_data, load_val_data
from gnn import ProofGNN

# ===== CONFIG ==========
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
NUM_NODE_TYPES = 3  # {state, tactic, premise}
# =======================


def train():
    train_loader = load_train_data(batch_size=BATCH_SIZE, path="data/pyg_graphs/train")
    val_loader   = load_val_data(batch_size=BATCH_SIZE, path="data/pyg_graphs/val")

    # Infer vocabulary size *from train only* to avoid leakage
    num_entities = max(g.entity.max().item() for g in train_loader.dataset) + 1

    model = ProofGNN(num_entities=num_entities, num_node_types=NUM_NODE_TYPES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.node_type)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Train Loss = {total_loss:.4f}")

        # -------- VALIDATION --------
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch)
                val_loss += criterion(logits, batch.node_type).item()
                pred = logits.argmax(dim=-1)
                correct += (pred == batch.node_type).sum().item()
                total += batch.node_type.numel()

        print(f"[Epoch {epoch+1}] Val Loss = {val_loss:.4f} | Val Acc = {correct/total:.4f}")

    torch.save(model.state_dict(), "proof_gnn.pt")
    print("✓ Saved model to proof_gnn.pt")


if __name__ == "__main__":
    train()
