import pickle
import json
import re
from pathlib import Path
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# load canonical tactic vocabulary
with open("data/cache/tactic_vocab.json") as f:
    TACTICS = json.load(f)

tactic2id = {t: i for i, t in enumerate(TACTICS)}


# node type vocabulary
type2id = {
    "state": 0,
    "tactic": 1,
    "premise": 2,
}
NUM_NODE_TYPES = len(type2id)


# boundary-aware tactic matcher
NON_ID_CHARS = set(" \t\n\r\f\v[]{}().,:;!?+-*/=<>|")

def match_tactic(node_str, tactic_list=TACTICS):
    """
    Return the canonical tactic name if node_str invokes it.
    Matching rules:
      1. Exact match
      2. Prefix match with boundary: startswith(t) and next char is non-identifier
      3. "tactic?" automatically treated as distinct (if in vocab)
      4. Ignore state/premise/proof strings unless they invoke a known tactic
    """
    s = str(node_str).strip()

    # case 1 - exact match
    if s in tactic_list:
        return s

    # case 2 — prefix match with boundary check
    for t in tactic_list:
        if s.startswith(t):
            if len(s) == len(t):
                return t
            nxt = s[len(t)]
            if not nxt.isalnum() and nxt != "_":
                return t

    return None


# node feature extraction
# node_type = categorical (state / tactic / premise)
# node_tactic_id = [0..num_tactics-1] if tactic node
#                = -1 otherwise
def extract_node_features(G):
    node_type_ids = []
    node_tactic_ids = []

    for n in G.nodes():
        # node type
        ntype = G.nodes[n].get("type", "state")
        node_type_ids.append(type2id.get(ntype, type2id["state"]))

        # tactic ID
        matched = match_tactic(n)
        if matched is not None:
            node_tactic_ids.append(tactic2id[matched])
        else:
            node_tactic_ids.append(-1)

    return (
        torch.tensor(node_type_ids, dtype=torch.long),
        torch.tensor(node_tactic_ids, dtype=torch.long),
    )

# graph conversion
def convert_graph(fp):
    with open(fp, "rb") as f:
        G = pickle.load(f)

    data = from_networkx(G)

    # node-level features
    node_type_ids, node_tactic_ids = extract_node_features(G)

    data.node_type = node_type_ids
    data.node_tactic_id = node_tactic_ids

    # target tactic label (graph-level)
    tstr = G.graph.get("target_tactic", None)
    if tstr is None:
        raise ValueError(f"{fp}: graph has no 'target_tactic' attribute.")

    matched = match_tactic(tstr)
    if matched is None:
        raise ValueError(
            f"{fp}: unrecognized target tactic '{tstr}'.\n"
            f"Add its canonical form to tactic_vocab.json."
        )

    data.target_tactic = torch.tensor(tactic2id[matched], dtype=torch.long)

    return data


# convert an entire split
def convert_split(split):
    in_dir = Path(f"data/processed/{split}")
    out_dir = Path(f"data/pyg/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = list(in_dir.glob("*.gpickle"))
    print(f"[{split}] Converting {len(fps)} graphs...")

    for fp in tqdm(fps, desc=f"Convert {split}"):
        try:
            data = convert_graph(fp)
        except ValueError as e:
            print(f"Skipping {fp.name}: {e}")
            continue
        torch.save(data, out_dir / f"{fp.stem}.pt")


def main():
    for split in ["train", "val", "test"]:
        convert_split(split)

if __name__ == "__main__":
    main()

