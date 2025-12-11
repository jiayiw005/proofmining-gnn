import os
import json, networkx as nx
import re
from pathlib import Path
from tqdm import tqdm
import pickle

def extract_premises(a):
    """ extract premises from an annotated tactic string """
    if isinstance(a, list):
        parts = []
        for x in a:
            if isinstance(x, list):
                parts.append("".join(str(xx) for xx in x))
            else:
                parts.append(str(x))
        a = " ".join(parts)
    if not isinstance(a, str):
        a = str(a)
    return re.findall(r"<a>(.*?)</a>", a)

def safe_str(x):
    if x is None:
        return ""
    return str(x).replace("\n", " ").strip()

def build_ptg(theorem):
    G = nx.DiGraph()
    tactics = theorem.get("traced_tactics") or []

    last_tactic = None

    for t in tactics:
        s_before = safe_str(t.get("state_before"))
        s_after  = safe_str(t.get("state_after"))
        tactic_str = safe_str(t.get("tactic"))

        if not tactic_str:
            continue

        last_tactic = tactic_str   # store latest

        G.add_node(s_before, type="state")
        G.add_node(s_after, type="state")
        G.add_node(tactic_str, type="tactic")

        G.add_edge(s_before, tactic_str, relation="applies")
        G.add_edge(tactic_str, s_after, relation="yields")

        for p in extract_premises(t.get("annotated_tactic")):
            G.add_node(p, type="premise")
            G.add_edge(p, tactic_str, relation="used_in")

    return G, last_tactic


def sanitize_name(name):
    if not name:
        return "unknown"
    return re.sub(r'[^A-Za-z0-9_]+', '_', name)

def preprocess_dataset():
    """ preprocess the dataset and save PTGs """
    for split in ["train", "val", "test"]:
        json_path = Path(f"data/raw/{split}.json")
        if not json_path.exists():
            continue

        out_split = Path(f"data/processed/{split}")
        out_split.mkdir(parents=True, exist_ok=True)

        with open(json_path, "r") as f:
            data = json.load(f)

        ok = 0
        for th in tqdm(data):
            if not th.get("traced_tactics"):
                continue
            try:
                G, target = build_ptg(th)
                if target is None: 
                    continue
                if len(G) == 0:
                    continue
                name = sanitize_name(th.get("full_name", "unknown"))
                G.graph["target_tactic"] = target 

                with open(out_split / f"{name}.gpickle", "wb") as f:
                    pickle.dump(G, f)
                ok += 1
            except:
                pass

        print(f"{ok} graphs written")

def main():
    preprocess_dataset()

if __name__ == "__main__":
    main()