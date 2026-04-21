"""Quick diagnostic: evaluate a LoQI checkpoint on the val set.

Loads the checkpoint, iterates the val split with the same preprocessing /
time-sampling / interpolation / forward path used during training, and
reports the denoising loss (and a raw MSE on the predicted coords as a
second reference).

This matches what `train.py` logs as `val/x_loss` on an epoch boundary,
so it's the right "where did we converge to" number to quote when
comparing a fresh pre-train vs. a fine-tune.

Usage:
    python scripts/eval_loqi_loss.py \\
        --ckpt data/loqi.ckpt \\
        --config scripts/conf/loqi/loqi.yaml \\
        --max-batches 200 \\
        --device cuda
"""
import argparse
import time
from statistics import mean, median, stdev

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.data.molecule_datamodule import MoleculeDataModule
from megalodon.models.module import Graph3DInterpolantModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True,
                   help="LoQI backbone config (same one the ckpt was trained with).")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--max-batches", type=int, default=None,
                   help="Cap batches to iterate. Default None = whole split.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-suffix", default=None,
                   help="Override cfg.data.data_suffix (e.g. '_h' or '_h_thermo').")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)

    # --- Load model ---
    print(f"Loading {args.ckpt}")
    pre = BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    model = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=pre,
        map_location=device,
    )
    model.eval().to(device)
    for p_ in model.parameters():
        p_.requires_grad = False

    # --- Data ---
    suffix = args.data_suffix or OmegaConf.select(cfg, "data.data_suffix", default="_h")
    dm = MoleculeDataModule(
        cfg.data.dataset_root,
        cfg.data.processed_folder,
        cfg.data.batch_size,
        cfg.data.data_loader_type,
        cfg.data.inference_batch_size,
        data_suffix=suffix,
    )
    loader = {
        "train": dm.train_dataloader,
        "val":   dm.val_dataloader,
        "test":  dm.test_dataloader,
    }[args.split]()

    # --- Eval ---
    x_losses = []
    x_mse    = []
    t0 = time.time()
    n_seen = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"eval {args.split}")):
            if args.max_batches and i >= args.max_batches:
                break
            batch = batch.to(device)
            batch = model.batch_preprocessor(batch)
            t = model.sample_time(batch)
            out, batch_out, t = model(batch, t)

            # Full InterpolantLossFunction loss for variable 'x' — matches
            # what train.py logs. Pull the x-variable's loss-fn out of the
            # model's initialized loss_functions dict.
            x_loss_fn = model.loss_functions["x"]
            # InterpolantLossFunction's call signature varies slightly by
            # version — call the x-loss path used in training_step.
            try:
                # Try the high-level interface that calculate_loss uses
                x_loss, _ = x_loss_fn.continuous_loss(
                    batch_out, out, t, stage="val",
                )
            except Exception:
                # Fall back to raw MSE on predicted coords vs clean coords
                x_loss = F.mse_loss(out["x_hat"], batch_out["x"])
            x_losses.append(float(x_loss))

            # Second reference: direct MSE on coords
            x_mse.append(float(F.mse_loss(out["x_hat"], batch_out["x"])))

            bs = int(batch_out.batch.max().item()) + 1
            n_seen += bs

    # --- Report ---
    def summarize(name, xs):
        if not xs:
            return f"{name}: (no data)"
        xs_np = np.array(xs)
        return (f"{name:<20s}  mean={xs_np.mean():.4f}  std={xs_np.std():.4f}  "
                f"median={np.median(xs_np):.4f}  min={xs_np.min():.4f}  "
                f"max={xs_np.max():.4f}")

    print("\n" + "=" * 78)
    print(f"{args.split} split  |  {len(x_losses):,} batches  |  "
          f"{n_seen:,} molecules  |  {time.time()-t0:.1f}s")
    print("-" * 78)
    print(summarize("x_loss (training)", x_losses))
    print(summarize("x_mse  (raw MSE)",  x_mse))
    print("=" * 78)


if __name__ == "__main__":
    main()
