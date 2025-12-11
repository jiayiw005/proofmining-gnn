import torch
import json
from pathlib import Path
from sklearn.cluster import KMeans


def main(
    bank_path="data/cache/state_lm_bank.pt",
    strings_path="data/cache/state_lm_strings.json",
    out_path="data/cache/state_lm_clusters.pt",
    n_clusters=128,
    random_state=42,
):
    print(f"Loading LM bank from: {bank_path}")
    state_bank = torch.load(bank_path)
    print(f"Bank shape: {state_bank.shape}")

    X = state_bank.cpu().numpy()

    print(f"Clustering into {n_clusters} clusters with KMeans...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        verbose=1,
    )
    labels = kmeans.fit_predict(X)  # [N_state_strings]

    labels_t = torch.tensor(labels, dtype=torch.long)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(labels_t, out_path)

    print(f"Saved cluster assignments to: {out_path}")


if __name__ == "__main__":
    main()
