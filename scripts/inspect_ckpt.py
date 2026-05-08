"""Inspect a Lightning checkpoint and recover backbone architecture + head config.

Usage:
    python scripts/inspect_ckpt.py data/ft_ckpts/my_new.ckpt
    python scripts/inspect_ckpt.py data/ft_ckpts/my_new.ckpt --suggest-config
"""
import argparse
import sys
from pathlib import Path


def inspect(ckpt_path: str, suggest_config: bool = False) -> dict:
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    hp = ck.get("hyper_parameters", {})

    info = {}

    # ---- Backbone architecture (from weight shapes) ----------------------
    # invariant_node_feat_dim (d) from qkv_proj
    for k, v in sd.items():
        if "qkv_proj.weight" in k and "thermo" not in k and "rdkit" not in k and "combined" not in k:
            info["d"] = int(v.shape[1])
            break

    # num_layers from dit_layers
    layers = {int(k.split("dit_layers.")[1].split(".")[0])
              for k in sd if "dit_layers." in k}
    if layers:
        info["num_layers"] = max(layers) + 1

    # n_vector_features from coord_emb
    for k, v in sd.items():
        if "coord_emb.weight" in k:
            info["n_vector_features"] = int(v.shape[0])
            break

    # num_heads from qkv_proj shape (3*d / d = 3 → head_dim can be read from DiTeBlock)
    for k, v in sd.items():
        if "qkv_proj.weight" in k and "thermo" not in k and "rdkit" not in k and "combined" not in k:
            qkv_out = v.shape[0]
            d = info.get("d", 1)
            # num_heads can be inferred from a norm weight if it exists
            break
    # Try to get num_heads from stored hyperparams
    try:
        from omegaconf import OmegaConf
        if hp:
            d_args = OmegaConf.select(hp, "dynamics_params.model_args", default=None)
            if d_args is not None:
                for key in ("num_heads", "n_heads"):
                    v = OmegaConf.select(d_args, key, default=None)
                    if v is not None:
                        info["num_heads"] = int(v)
                        break
    except Exception:
        pass

    # ---- Heads present ---------------------------------------------------
    heads = []
    head_configs = {}
    for head in ("thermo_heads", "rdkit_heads", "combined_heads"):
        keys = [k for k in sd if head in k]
        if keys:
            heads.append(head)
            # Count mp layers from q_proj ModuleList
            mp_layers = {int(k.split(f"{head}.mp.q_proj.")[1].split(".")[0])
                         for k in keys if f"{head}.mp.q_proj." in k}
            # hidden from final layer
            hidden = None
            for k, v in sd.items():
                if f"{head}.mp.final.1.weight" in k:
                    hidden = int(v.shape[0])
                    break
            n_targets = None
            for k, v in sd.items():
                if f"{head}.mp.final.3.weight" in k:
                    n_targets = int(v.shape[0])
                    break
            head_configs[head] = {
                "n_mp_layers": max(mp_layers) + 1 if mp_layers else None,
                "hidden": hidden,
                "n_targets": n_targets,
            }
    info["heads"] = heads
    info["head_configs"] = head_configs

    # ---- Training metadata -----------------------------------------------
    info["epoch"]       = ck.get("epoch", None)
    info["global_step"] = ck.get("global_step", None)

    # ---- Suggest config ---------------------------------------------------
    d = info.get("d")
    nl = info.get("num_layers")
    nv = info.get("n_vector_features")
    nh = info.get("num_heads", 4)

    config_hint = None
    if d and nl:
        if d == 256 and nl == 10:
            config_hint = "loqi_thermo_flow_warm.yaml"
        elif d == 384 and nl <= 12:
            config_hint = "loqi_thermo_flow_cold.yaml"
        elif d == 384 and nl > 12:
            config_hint = "loqi_thermo_flow_cold.yaml  (possibly larger variant)"
        elif d == 256 and "thermo_heads" not in heads and "combined_heads" not in heads:
            config_hint = "loqi_flow.yaml  (no thermo head)"
        else:
            config_hint = f"custom — d={d}, layers={nl}, n_vec={nv}"

    info["suggested_config"] = config_hint

    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="Path to .ckpt file")
    ap.add_argument("--suggest-config", action="store_true",
                    help="Print suggested YAML config name")
    args = ap.parse_args()

    info = inspect(args.ckpt)

    print(f"\n{'='*60}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"{'='*60}")

    print(f"\n--- Backbone ---")
    print(f"  invariant_node_feat_dim : {info.get('d')}")
    print(f"  num_layers              : {info.get('num_layers')}")
    print(f"  n_vector_features       : {info.get('n_vector_features')}")
    print(f"  num_heads               : {info.get('num_heads', '(try saved hparams)')}")

    print(f"\n--- Heads ---")
    for h in info.get("heads", []):
        cfg = info["head_configs"][h]
        print(f"  {h}:")
        print(f"    n_mp_layers = {cfg['n_mp_layers']}")
        print(f"    hidden      = {cfg['hidden']}")
        print(f"    n_targets   = {cfg['n_targets']}")
    if not info.get("heads"):
        print("  (no thermo/rdkit/combined heads detected — pure flow ckpt)")

    print(f"\n--- Training state ---")
    print(f"  epoch       : {info.get('epoch')}")
    print(f"  global_step : {info.get('global_step')}")

    print(f"\n--- Suggested config ---")
    print(f"  {info.get('suggested_config', '(could not determine)')}")

    if args.suggest_config:
        print(f"\n  YAML to use: scripts/conf/loqi/{info.get('suggested_config', '')}")

    print()


if __name__ == "__main__":
    main()
