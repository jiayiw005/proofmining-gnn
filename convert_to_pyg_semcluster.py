import pickle
import json
from pathlib import Path
from tqdm import tqdm

import torch
from torch_geometric.utils import from_networkx

# canonical tactic vocabulary
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

# load state-node LM index + clusters
STATE_LM_BANK = torch.load("data/cache/state_lm_bank.pt") 
with open("data/cache/state_lm_strings.json", "r") as f:
    STATE_LM_STRINGS = json.load(f)
STATE_LM_INDEX = {s: i for i, s in enumerate(STATE_LM_STRINGS)}

STATE_LM_CLUSTERS = torch.load("data/cache/state_lm_clusters.pt") 
NUM_STATE_CLUSTERS = int(STATE_LM_CLUSTERS.max().item() + 1)

# boundary-aware tactic matcher
NON_ID_CHARS = set(" \t\n\r\f\v[]{}().,:;!?+-*/=<>|")

def match_tactic(node_str, tactic_list=TACTICS):
    """
    Return canonical tactic name if node_str invokes it.
    """
    s = str(node_str).strip()

    # exact match
    if s in tactic_list:
        return s

    # prefix match with boundary
    for t in tactic_list:
        if s.startswith(t):
            if len(s) == len(t):
                return t
            nxt = s[len(t)]
            if not nxt.isalnum() and nxt != "_":
                return t

    return None


# node feature extraction
def extract_node_features(G):
    """
    Returns:
      node_type_ids      : LongTensor [N]
      node_tactic_ids    : LongTensor [N]
      state_cluster_ids  : LongTensor [N]
    """
    node_type_ids = []
    node_tactic_ids = []
    state_cluster_ids = []

    for n in G.nodes():
        ntype = G.nodes[n].get("type", "state")
        node_type_ids.append(type2id.get(ntype, type2id["state"]))

        # tactic ID
        matched = match_tactic(n)
        if matched is not None:
            node_tactic_ids.append(tactic2id[matched])
        else:
            node_tactic_ids.append(-1)

        # state semantic cluster ID
        if ntype == "state":
            s = str(n)
            lm_idx = STATE_LM_INDEX.get(s, -1)
            if lm_idx >= 0:
                c_id = int(STATE_LM_CLUSTERS[lm_idx].item())
            else:
                c_id = -1
        else:
            c_id = -1

        state_cluster_ids.append(c_id)

    return (
        torch.tensor(node_type_ids, dtype=torch.long),
        torch.tensor(node_tactic_ids, dtype=torch.long),
        torch.tensor(state_cluster_ids, dtype=torch.long),
    )


# single graph conversion
def convert_graph(fp):
    with open(fp, "rb") as f:
        G = pickle.load(f)

    data = from_networkx(G)

    node_type_ids, node_tactic_ids, state_cluster_ids = extract_node_features(G)

    data.node_type = node_type_ids
    data.node_tactic_id = node_tactic_ids
    data.state_cluster_id = state_cluster_ids

    # target tactic label
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


# split conversion
def convert_split(split):
    in_dir = Path(f"data/processed/{split}")
    out_dir = Path(f"data/pyg_semcluster/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = list(in_dir.glob("*.gpickle"))
    print(f"[{split}] Converting {len(fps)} graphs with semantic clusters...")

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
