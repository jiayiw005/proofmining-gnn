import os
import json
import torch
from datetime import datetime
from copy import deepcopy

from node_pred import train_next_tactic
from data_loader import load_train_data, load_val_data, load_test_data
from gnn import ProofGNN_NextTactic
import argparse



def run_experiment(config_overrides, name_suffix):
    """run a training experiment with specific ablation settings"""

    base_config = {
        "batch_size":   8,
        "epochs":       10,
        "lr":           1e-3,
        "dropout":      0.2,
        "hidden_dim":   128,
        "type_embed":   32,
        "tactic_embed": 64,
    }

    # merge overrides
    cfg = {**base_config, **config_overrides}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ablation_{name_suffix}_{stamp}"
    run_dir  = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print(f" running {run_name}")
    print("=" * 80)
    print(json.dumps(cfg, indent=2))

    # save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    train_next_tactic(
        batch_size = cfg["batch_size"],
        epochs     = cfg["epochs"],
        lr         = cfg["lr"],
        save_every_epoch=False,
        run_dir_override=run_dir,
        model_hparams=cfg,
    )


# ablation defs

def ablation_no_tactic_embedding():
    """
    remove node_tactic_id as a feature, replace with zeros
    """
    run_experiment(
        config_overrides={
            "remove_tactic_feature": True
        },
        name_suffix="no_tactic_embedding"
    )


def ablation_no_node_type():
    """
    remove node_type from inputs (constant type)
    """
    run_experiment(
        config_overrides={
            "remove_node_type": True
        },
        name_suffix="no_node_type"
    )


def ablation_shuffle_edges():
    """
    randomize graph edges for every graph
    """
    run_experiment(
        config_overrides={
            "shuffle_edges": True
        },
        name_suffix="shuffle_edges"
    )


def ablation_mlp_bag_of_tactics():
    """
    no graph used; convert graph to histogram(features), predict next tactic
    """
    run_experiment(
        config_overrides={
            "mlp_bag_of_tactics": True,
            "hidden_dim": 512,
        },
        name_suffix="mlp_bag_of_tactics"
    )


def ablation_shuffle_tactic_ids():
    """
    randomly permute tactic classes each epoch
    """
    run_experiment(
        config_overrides={
            "shuffle_tactic_ids": True
        },
        name_suffix="shuffle_tactic_ids"
    )


def ablation_shuffle_targets():
    """
    destroy labels by permuting target_tactic
    should produce near-chance accuracy
    """
    run_experiment(
        config_overrides={
            "shuffle_targets": True
        },
        name_suffix="shuffle_targets"
    )

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", type=str, required=True, nargs='+',
                   choices=[
                       "no_tactic_embedding",
                       "no_node_type",
                       "shuffle_edges",
                       "mlp_bag_of_tactics",
                       "shuffle_tactic_ids",
                       "shuffle_targets",
                   ],
                   help="Which ablation(s) to run")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    for ablation in args.ablation:
        if ablation == "no_tactic_embedding":
            ablation_no_tactic_embedding()
        elif ablation == "no_node_type":
            ablation_no_node_type()
        elif ablation == "shuffle_edges":
            ablation_shuffle_edges()
        elif ablation == "mlp_bag_of_tactics":
            ablation_mlp_bag_of_tactics()
        elif ablation == "shuffle_tactic_ids":
            ablation_shuffle_tactic_ids()
        elif ablation == "shuffle_targets":
            ablation_shuffle_targets()
