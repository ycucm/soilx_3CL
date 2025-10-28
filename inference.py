#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SoilX inference script (with reference-centered, piecewise linear scaling)

This script loads a pretrained SoilX3CL model (artifact.pth) and performs inference
on a CSV dataset containing the same feature columns as used in training.

Uses reference-centered, two-sided scaling:
    For each component g:
        if s >= 0:  y_pred = y0 + (s / s_pos_max) * (y_max - y0)
        if s <  0:  y_pred = y0 + (s / |s_neg_min|) * (y0 - y_min)
  This ensures:
    • the reference sample (#0) aligns exactly to y0 (not y_min)
    • the mapping is monotonic on both sides
    • values remain bounded and interpretable

Outputs:
- Prints model summary, data info, intermediate tensor shapes
- Displays predicted vs ground-truth (if available)
- Saves a CSV with predicted labels appended
"""

import argparse
import torch
import pandas as pd
import torch.nn.functional as F

# import the same encoder model definition as training
from train import SoilX3CL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file for inference")
    parser.add_argument("--model", type=str, required=True, help="Trained artifact .pth file")
    parser.add_argument("--save_csv", type=str, default="predictions.csv", help="Output CSV file with predictions")
    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1. Load trained artifact and model
    # ------------------------------------------------------------
    print("\n[1] Loading model artifact:", args.model)
    art = torch.load(args.model, map_location="cpu")

    feat_cols = art["feat_cols"]
    label_cols = art["label_cols"]
    latent_dim = art["latent_dim"]

    model = SoilX3CL(input_dim=len(feat_cols), latent_dim=latent_dim)
    model.load_state_dict(art["model_state"])
    model.eval()

    print(f"  ✓ Feature dims: {len(feat_cols)}")
    print(f"  ✓ Label dims: {len(label_cols)}")
    print(f"  ✓ Latent dim: {latent_dim}")

    # ------------------------------------------------------------
    # 2. Load input data
    # ------------------------------------------------------------
    print("\n[2] Loading data:", args.csv)
    df = pd.read_csv(args.csv)
    X = torch.tensor(df[feat_cols].values, dtype=torch.float32)

    print(f"  ✓ Loaded {len(df)} samples with {X.shape[1]} features.")
    if all(col in df.columns for col in label_cols):
        has_gt = True
        Y_gt = torch.tensor(df[label_cols].values, dtype=torch.float32)
        print("  ✓ Ground-truth labels found in CSV.")
    else:
        has_gt = False
        print("  ⚠️ No ground-truth labels found in CSV.")

    # ------------------------------------------------------------
    # 3. Prepare model components and scaling parameters
    # ------------------------------------------------------------
    print("\n[3] Preparing model parameters and scaling references...")

    # encoder reference embedding and orthonormal axes
    z0 = art["z0"]
    Z = torch.stack([torch.tensor(row) for row in art["z_avg"]], dim=0)  # [6, D]

    # reference label and global bounds
    y0 = torch.tensor(art["y0"])      # reference label (y_ref)
    y_min = torch.tensor(art["y_min"])
    y_max = torch.tensor(art["y_max"])

    # per-axis scaling limits measured during training
    s_pos_max = torch.tensor(art["s_pos_max"])  # ≥ 0
    s_neg_min = torch.tensor(art["s_neg_min"])  # ≤ 0

    print(f"  ✓ Loaded reference y0: {y0.tolist()}")
    print(f"  ✓ s_pos_max: {s_pos_max.tolist()}")
    print(f"  ✓ s_neg_min: {s_neg_min.tolist()}")

    # ------------------------------------------------------------
    # 4. Forward pass through encoder → embedding → projection
    # ------------------------------------------------------------
    print("\n[4] Running encoder forward pass...")
    with torch.no_grad():
        z = model(X)                   # [N, D]
        v = z - z0.unsqueeze(0)        # difference from reference
        S = (Z @ v.T).T                # [N, 6] scalar projections along axes

    print(f"  ✓ Embedding shape: {z.shape}")
    print(f"  ✓ Projection shape: {S.shape}")

    # ------------------------------------------------------------
    # 5. Apply two-sided piecewise scaling around y0
    # ------------------------------------------------------------
    print("\n[5] Applying piecewise scaling around y0 (reference-centered)...")

    # clamp projections within observed training range
    S_clamped = torch.minimum(S, s_pos_max.unsqueeze(0))
    S_clamped = torch.maximum(S_clamped, s_neg_min.unsqueeze(0))

    # initialize predicted labels
    Y_pred = torch.zeros_like(S_clamped)

    for j, lbl in enumerate(label_cols):
        sp = s_pos_max[j].clamp(min=1e-8)
        sn = s_neg_min[j].clamp(max=-1e-8)

        pos_mask = S_clamped[:, j] >= 0
        neg_mask = ~pos_mask

        # reference-centered piecewise mapping
        Y_pred[pos_mask, j] = y0[j] + (S_clamped[pos_mask, j] / sp) * (y_max[j] - y0[j])
        Y_pred[neg_mask, j] = y0[j] + (S_clamped[neg_mask, j] / sn.abs()) * (y0[j] - y_min[j])

        print(f"  - [{lbl}] range → "
              f"y_min={y_min[j]:.4f}, y0={y0[j]:.4f}, y_max={y_max[j]:.4f}, "
              f"s_neg_min={s_neg_min[j]:.4f}, s_pos_max={s_pos_max[j]:.4f}")

    print("  ✓ Scaling applied successfully.")
    print(f"  ✓ Predicted label tensor shape: {Y_pred.shape}")

    # ------------------------------------------------------------
    # 6. Convert to DataFrame and print sample results
    # ------------------------------------------------------------
    df_pred = df.copy()
    for j, lbl in enumerate(label_cols):
        df_pred[f"pred_{lbl}"] = Y_pred[:, j].numpy()

    print("\n[6] Sample prediction results:")
    display_cols = [c for c in df_pred.columns if c.startswith("pred_")]
    print(df_pred[display_cols].head(10).to_string(index=False))

        # Report MAE ± SE (Standard Error)
    if has_gt:
        print("\n[6] Evaluation metrics (MAE ± SE):")

        # Absolute error tensor [N, 6]
        AE = torch.abs(Y_pred - Y_gt)
        N = AE.shape[0]

        # Per-label statistics
        mae_mean = torch.mean(AE, dim=0)
        mae_std = torch.std(AE, dim=0)
        mae_se = mae_std / (N ** 0.5)  # Standard Error = std / sqrt(N)

        for j, lbl in enumerate(label_cols):
            print(f"    - {lbl}: MAE = {mae_mean[j]:.6f} ± {mae_se[j]:.6f} (SE)")
        
    # ------------------------------------------------------------
    # 7. Save predicted CSV
    # ------------------------------------------------------------
    df_pred.to_csv(args.save_csv, index=False)
    print(f"\n[7] Saved predictions to: {args.save_csv}")
    print("✅ Inference complete.\n")


if __name__ == "__main__":
    main()