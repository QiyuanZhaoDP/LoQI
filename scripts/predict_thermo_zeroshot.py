"""Zero-shot thermo prediction using the pretrained thermo head.

No fine-tuning, no head training — just run model.thermo_heads's prediction
on K=5 conformers per SMILES (from sample_conformers.py output) and compare
to ground-truth labels in a downstream CSV.

Two questions this answers in one shot:
  1. How well does the foundation model's thermo head generalize zero-shot
     to a new dataset (e.g., gas_Hf.csv vs ChEMBL3D's TCIT labels)?
  2. How conformer-sensitive is the prediction? Per-input-id σ across K
     predictions tells us if the head is robust or memorized conformer-
     specific features during pretraining.

Usage (Hf_298 on gas_Hf.csv with the warm flow ckpt):
  python scripts/predict_thermo_zeroshot.py \\
      --ckpt data/thermo_flow_warm.ckpt \\
      --config scripts/conf/loqi/loqi_thermo_flow_warm.yaml \\
      --conformer-pkl data/downstream_k5/gas_Hf.pkl \\
      --target-csv  downstream_ft/gas_Hf.csv \\
      --smiles-col smiles --target-col mean \\
      --target enthalpy_298 \\
      --output outputs/zeroshot/gas_Hf_H298.json
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from rdkit import Chem, RDLogger
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.models.module import Graph3DInterpolantModel

# Reuse the same RDKit Mol → PyG Data helper as prepare_downstream_K_pt.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_downstream_dataset import _fallback_mol_to_data, HAVE_UTIL  # noqa
if HAVE_UTIL:
    from utils_data import mol_to_pyg_data  # noqa

RDLogger.DisableLog("rdApp.*")


# Thermo target name → output column index of out["thermo_mp"] (matches
# loqi_thermo_flow_warm.yaml's thermo_loss target_mean/target_std order).
THERMO_TARGETS = {
    "enthalpy_298": 0,
    "gibbs_298":    1,
    "cv_gas":       2,
    "entropy_gas":  3,
    "enthalpy_0":   4,
}


def _mol_to_data(mol):
    return mol_to_pyg_data(mol) if HAVE_UTIL else _fallback_mol_to_data(mol)


def _canon(mol_or_smi):
    if isinstance(mol_or_smi, str):
        m = Chem.MolFromSmiles(mol_or_smi)
        if m is None:
            return None
    else:
        m = mol_or_smi
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--conformer-pkl", required=True,
                   help="sample_conformers.py output (list of K conformers per SMILES)")
    p.add_argument("--target-csv", required=True)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--target-col", required=True)
    p.add_argument("--target", default="enthalpy_298",
                   choices=list(THERMO_TARGETS.keys()),
                   help="which thermo target to predict (default: enthalpy_298)")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    target_idx = THERMO_TARGETS[args.target]

    # ---- Load config + normalization stats ------------------------------
    cfg = OmegaConf.load(args.config)
    tl = OmegaConf.select(cfg, "thermo_loss", default=None)
    if tl is None:
        raise SystemExit("Config has no thermo_loss block — can't denormalize.")
    target_mean = float(tl.target_mean[target_idx])
    target_std  = float(tl.target_std[target_idx])
    print(f"[config] target={args.target} idx={target_idx}  "
          f"mean={target_mean:.3f} std={target_std:.3f}")

    # ---- Load model -----------------------------------------------------
    pre = BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    print(f"[model] loading {args.ckpt}")
    model = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_fn=None,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=pre,
        strict=False,
    ).to(args.device).eval()

    t_type = str(cfg.interpolant.time_type)
    t_max = cfg.interpolant.timesteps - 1 if t_type == "discrete" else 1.0

    # ---- Load conformer pickle + group by canonical SMILES --------------
    print(f"[data] loading {args.conformer_pkl}")
    with open(args.conformer_pkl, "rb") as f:
        d = pickle.load(f)
    raw_mols = d["generated"]
    print(f"  pickle has {len(raw_mols):,} mols")

    by_canon: dict[str, list] = defaultdict(list)
    for m in raw_mols:
        if m is None:
            continue
        c = _canon(m)
        if c is not None:
            by_canon[c].append(m)
    print(f"  unique canonical SMILES: {len(by_canon):,}")

    # ---- Load CSV target + canonicalize keys ----------------------------
    df = pd.read_csv(args.target_csv)
    target_by_canon: dict[str, float] = {}
    n_target_canon_fail = 0
    for smi, val in zip(df[args.smiles_col].astype(str), df[args.target_col]):
        if pd.isna(val):
            continue
        c = _canon(smi)
        if c is None:
            n_target_canon_fail += 1
            continue
        target_by_canon[c] = float(val)
    print(f"  CSV: {len(df):,} rows  →  {len(target_by_canon):,} canonicalized targets")
    if n_target_canon_fail:
        print(f"  {n_target_canon_fail} SMILES failed to canonicalize")

    # ---- Build PyG Data list (only mols with a target) ------------------
    data_list = []
    canons = []
    n_skip_no_target = 0
    n_skip_convert = 0
    for canon, group_mols in by_canon.items():
        if canon not in target_by_canon:
            n_skip_no_target += len(group_mols)
            continue
        for m in group_mols:
            try:
                data = _mol_to_data(m)
            except Exception:
                n_skip_convert += 1
                continue
            data_list.append(data)
            canons.append(canon)
    print(f"  built {len(data_list):,} PyG Data "
          f"(skipped: no_target={n_skip_no_target}, convert_err={n_skip_convert})")
    if not data_list:
        raise SystemExit("Nothing to predict — empty data list after filtering.")

    # ---- Forward through model, collect z-score predictions -------------
    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=False)
    preds_z: list[float] = []
    print(f"[inference] forward through model on {args.device}")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            bs = int(batch.batch.max().item()) + 1
            batch = pre(batch)
            if t_type == "discrete":
                t = torch.full((bs,), t_max, dtype=torch.long, device=args.device)
            else:
                t = torch.full((bs,), t_max, dtype=torch.float32, device=args.device)
            out, _, _ = model(batch, t)
            if "thermo_mp" not in out:
                raise SystemExit(
                    "Model has no `thermo_mp` output — ckpt was not trained "
                    "with thermo_head_args. Use a thermo-pretrained ckpt."
                )
            pred_z = out["thermo_mp"][:, target_idx].detach().cpu().numpy()
            preds_z.extend(pred_z.tolist())

    # ---- De-normalize + group by SMILES ---------------------------------
    preds_phys = np.asarray(preds_z) * target_std + target_mean
    by_canon_pred: dict[str, list] = defaultdict(list)
    for canon, pred in zip(canons, preds_phys):
        by_canon_pred[canon].append(float(pred))

    # ---- Aggregate metrics ---------------------------------------------
    rows = []
    for canon, preds in by_canon_pred.items():
        if canon not in target_by_canon:
            continue
        target = target_by_canon[canon]
        pred_mean = float(np.mean(preds))
        pred_std = float(np.std(preds)) if len(preds) > 1 else 0.0
        rows.append({
            "smiles": canon,
            "target": target,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "K": len(preds),
            "abs_error": abs(pred_mean - target),
        })
    if not rows:
        raise SystemExit("No (mol, target) overlaps after canonical join.")

    df_res = pd.DataFrame(rows)
    mae = mean_absolute_error(df_res["target"], df_res["pred_mean"])
    rmse = float(np.sqrt(((df_res["pred_mean"] - df_res["target"]) ** 2).mean()))
    r2 = float(r2_score(df_res["target"], df_res["pred_mean"]))
    pred_std_mean = float(df_res["pred_std"].mean())
    pred_std_p95 = float(df_res["pred_std"].quantile(0.95))
    target_overall_std = float(df_res["target"].std())
    sensitivity_pct = pred_std_mean / max(target_overall_std, 1e-12) * 100

    # ---- Print + save report -------------------------------------------
    print("\n" + "=" * 64)
    print(f"  ZERO-SHOT {args.target.upper()}  ({args.conformer_pkl})")
    print("=" * 64)
    print(f"  N molecules:     {len(df_res):,}")
    print(f"  MAE:             {mae:.2f}  (target unit)")
    print(f"  RMSE:            {rmse:.2f}")
    print(f"  R²:              {r2:.3f}")
    print(f"  --- conformer sensitivity ---")
    print(f"  pred σ across K conformers (mean over mols): {pred_std_mean:.2f}")
    print(f"  pred σ 95th-percentile:                       {pred_std_p95:.2f}")
    print(f"  target distribution σ:                        {target_overall_std:.2f}")
    print(f"  sensitivity % (pred σ / target σ):            {sensitivity_pct:.1f}%")
    print("=" * 64)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "target": args.target,
        "target_idx": target_idx,
        "target_mean": target_mean, "target_std": target_std,
        "n_mols": len(df_res),
        "mae": float(mae),
        "rmse": rmse,
        "r2": r2,
        "conformer_pred_std_mean": pred_std_mean,
        "conformer_pred_std_p95":  pred_std_p95,
        "target_distribution_std": target_overall_std,
        "conformer_sensitivity_pct": sensitivity_pct,
        "per_mol": rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nReport -> {out_path}")


if __name__ == "__main__":
    main()
