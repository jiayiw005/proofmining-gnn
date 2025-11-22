import os
import json, networkx as nx
import re
from pathlib import Path
from tqdm import tqdm

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
    """ build a proof term graph (PTG) for a given theorem """
    G = nx.DiGraph()
    print(theorem.get("full_name", "unknown theorem"))
    tactics = theorem.get("traced_tactics") or []

    for t in tactics:
        print("==========Traced Tactics==========")
        print(t)
        print("==========Traced Tactics Done==========")
        s_before = safe_str(t.get("state_before"))
        s_after  = safe_str(t.get("state_after"))
        tactic_str = safe_str(t.get("tactic"))
        
        print("==========States==========")
        print(f"Processing tactic: {tactic_str}")
        print(f"State before: {s_before}")
        print(f"State after: {s_after}")    
        print("==========States Done==========")

        if not tactic_str:
            continue
        
        # add nodes
        G.add_node(s_before, type="state")
        G.add_node(s_after, type="state")
        G.add_node(tactic_str, type="tactic")
        
        # add edges (state_before -> tactic -> state_after)
        G.add_edge(s_before, tactic_str, relation="applies")
        G.add_edge(tactic_str, s_after, relation="yields")

        # add premises used in the tactic
        for p in extract_premises(t.get("annotated_tactic")):
            print("==========Premises==========")
            print(f"  Found premise: {p}")
            G.add_node(p, type="premise")
            G.add_edge(p, tactic_str, relation="used_in")
            print("==========Premises Done==========")
    
    return G

def sanitize_name(name):
    if not name:
        return "unknown"
    return re.sub(r'[^A-Za-z0-9_]+', '_', name)

def preprocess_dataset():
    """ preprocess the dataset and save PTGs """
    for split in ["train", "val", "test"]:
        json_path = Path(f"data/test_parsing.json")
        if not json_path.exists():
            continue

        out_split = Path(f"data")
        out_split.mkdir(parents=True, exist_ok=True)

        with open(json_path, "r") as f:
            data = json.load(f)

        ok = 0
        for th in tqdm(data):
            if not th.get("traced_tactics"):
                continue
            try:
                G = build_ptg(th)
                if len(G) == 0:
                    continue
                name = sanitize_name(th.get("full_name", "unknown"))
                import pickle
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