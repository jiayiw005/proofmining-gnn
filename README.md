# **ProofMining-GNN: Capturing Graph Structures in Lean4 Theorem Proving**

This repository contains the full code for training, evaluating, and running ablation studies on a Graph Neural Network (GNN) model for *next-tactic prediction* in Lean4 theorem proving.
The model operates over proof-state graphs extracted from [LeanDojo](https://zenodo.org/records/12740403) and uses both symbolic features (node types, tactic IDs) and optional semantic features (sentence-embedding representations of proof states).

---

## **Setup**

### **Clone the repository**

```bash
git clone https://github.com/jiayiw005/proofmining-gnn.git
cd proofmining-gnn
```

---

## **Prepare Data**

### **Dataset preprocessing**

Download LeanDojo proof traces (Mathlib proofs) inside `data/` and run `preprocessing/preprocessing.py`, which would place `.gpickle` proof graphs into:

```
data/processed/train/
data/processed/val/
data/processed/test/
```

### **Build tactic vocabulary**

```bash
python scripts/build_tactic_vocab.py
```

Produces:

```
data/cache/tactic_vocab.json
```

### **Build semantic embedding bank**

Encodes state-node strings into MiniLM embeddings:

```bash
python scripts/build_state_lm_bank.py
```

Produces:

```
data/cache/state_lm_bank.pt
data/cache/state_lm_strings.json
```

---

## **Convert proofs into PyG format**

Run semantic-aware conversion:

```bash
python preprocessing/convert_to_pyg_semantic.py
```

This generates:

```
data/pyg_semantic/train/*.pt
data/pyg_semantic/val/*.pt
data/pyg_semantic/test/*.pt
```

Each `.pt` file is a PyTorch Geometric `Data` object with:

* `node_type`
* `node_tactic_id`
* `state_lm_id`
* `edge_index`
* `target_tactic`

---

## **Train the model**

### **Basic training**

```bash
python models/node_pred.py --epochs 10 --batch_size 8 --lr 1e-4
```

A new run directory will be created, for example:

```
runs/tactic_run_20251208_222405/
```

which contains:

* `best_model.pt`
* `final_model.pt`
* `tactic_vocab.json`
* `training_log.json`

### **Run ablations**

Examples:

```bash
python models/node_pred.py --epochs 10 --batch_size 8 --lr 1e-4 \
    --remove_tactic_feature
```

```bash
python models/node_pred.py --epochs 10 --batch_size 8 --lr 1e-4 \
    --shuffle_edges
```

---

## **Evaluate on test set**

You can choose to run either the best model or the final model trained in one run. 
```bash
python models/node_eval.py \
    --model runs/tactic_run_20251208_222405/best_model.pt \
    --run_dir runs/tactic_run_20251208_222405/
```
Every test run produces model's next-tactic prediction accuracy on the LeanDojo test set. 

---

## **Expected results (full model + ablations)**

| Experiment               | Val Acc   |
| ------------------------ | --------- |
| Full model               | **0.875** |
| Remove tactic embedding  | 0.423     |
| Remove node type         | 0.874     |
| Shuffle edges            | 0.873     |
| Shuffle tactic IDs       | 0.777     |
| Shuffle targets (sanity) | 0.000     |
| MLP baseline             | 0.836     |
| Add semantic embeddings  | **0.888** |
