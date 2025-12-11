import pickle, json
from pathlib import Path
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

def collect_state_nodes(dirs):
    states = set()
    for d in dirs:
        for fp in tqdm(list(Path(d).glob("*.gpickle")), desc=f"Scan {d}"):
            with open(fp, "rb") as f:
                G = pickle.load(f)
            for n in G.nodes():
                ntype = G.nodes[n].get("type", "state")
                if ntype == "state":
                    states.add(str(n))
    return sorted(states)

def main():
    gdirs = ["data/processed/train", "data/processed/val", "data/processed/test"]
    all_states = collect_state_nodes(gdirs)
    print(f"Found {len(all_states)} unique state-node strings.")

    lm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = lm.encode(all_states, convert_to_numpy=True, show_progress_bar=True)

    Path("data/cache").mkdir(exist_ok=True)
    torch.save(torch.tensor(emb, dtype=torch.float), "data/cache/state_lm_bank.pt")
    with open("data/cache/state_lm_strings.json", "w") as f:
        json.dump(all_states, f, indent=2)

if __name__ == "__main__":
    main()
