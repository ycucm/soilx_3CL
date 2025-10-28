#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SoilX 3CL training script (full dataset, group-based orthogonality regularization)
Implements:
  - PCA (principal axis + orientation) for each group direction
  - "Reference point as zero" scaling: save y0, s_pos_max, s_neg_min for inference
  - L_ort with EMA + warmup, QR orthonormalization
"""

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from datetime import datetime

LABEL_COLS = ["M", "N", "P", "K", "C", "Al"]

# ------------------ utilities ------------------
def seed_everything(seed=2025):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def infer_group_from_id(id_str: str) -> str:
    if isinstance(id_str, str) and len(id_str) > 0:
        s = id_str.lstrip("#")
        if s == "0":
            return "REF"
        s_up = s.upper()
        for key in ["M","N","P","K","C","AL"]:
            if s_up.startswith(key):
                return key if key!="AL" else "Al"
    return "UNK"

def separation_loss(z, y_norm):
    diff_z = z.unsqueeze(0) - z.unsqueeze(1)
    dist_z = torch.norm(diff_z, dim=-1)
    diff_y = y_norm.unsqueeze(0) - y_norm.unsqueeze(1)
    dist_y = torch.norm(diff_y, dim=-1)
    return ((dist_z - dist_y) ** 2).mean()

def principal_axis_with_orientation(d, delta_y):
    """Return PCA principal axis (unit vector) oriented s.t. correlation(projection, delta_y) >= 0."""
    device = d.device
    if d.shape[0] == 0:
        return torch.zeros(d.shape[1], device=device)
    d_center = d - d.mean(dim=0, keepdim=True)
    if torch.allclose(d_center, torch.zeros_like(d_center)):
        v = d.mean(dim=0)
        v_norm = torch.norm(v)
        return v / (v_norm + 1e-12) if v_norm > 0 else torch.zeros(d.shape[1], device=device)
    U, S, Vh = torch.linalg.svd(d_center, full_matrices=False)
    v = Vh[0]
    proj = d @ v
    corr = torch.sum(proj * delta_y)
    if corr.item() < 0:
        v = -v
    return v

def orthogonality_regularizer(z_axis_list, ema_Z, ema_alpha=0.8):
    device = z_axis_list[0].device
    Z_cur = F.normalize(torch.stack(z_axis_list, dim=0), p=2, dim=1)
    with torch.no_grad():
        if ema_Z is None:
            ema_Z = Z_cur.detach()
        else:
            ema_Z = ema_alpha * ema_Z + (1 - ema_alpha) * Z_cur.detach()
    G = ema_Z @ ema_Z.T
    I = torch.eye(G.size(0), device=device)
    L_ort = ((G - I)**2).mean()
    return L_ort, ema_Z

class SoilX3CL(nn.Module):
    def __init__(self, input_dim, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        return F.normalize(z, dim=-1)

# ------------------ main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="runs/soilx")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--lambda_sep", type=float, default=0.82)
    parser.add_argument("--lambda_ort", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed)

    df = pd.read_csv(args.csv)
    feat_cols = [c for c in df.columns if c=="dielectric_permittivity" or c.startswith("vnir_")]
    for lc in LABEL_COLS:
        if lc not in df.columns:
            raise ValueError(f"Missing label col: {lc}")
    if "id" not in df.columns:
        raise ValueError("Need 'id' column")
    df["group"] = df["id"].apply(infer_group_from_id)
    if not (df["group"]=="REF").any():
        med = df[LABEL_COLS].median()
        ref_idx = (df[LABEL_COLS].sub(med).abs().sum(axis=1)).idxmin()
        df.loc[ref_idx,"group"]="REF"

    X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    Y = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
    y_min = Y.min(dim=0).values
    y_max = Y.max(dim=0).values
    y_rng = torch.clamp(y_max - y_min, min=1e-8)
    Yn = (Y - y_min) / y_rng

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoilX3CL(X.shape[1], args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    X,Y,Yn = X.to(device),Y.to(device),Yn.to(device)
    ref_idx = int(df.index[df["group"]=="REF"][0])
    groups = {k: df.index[df["group"]==k].tolist() for k in LABEL_COLS}

    ort_warmup_epochs = max(1, args.epochs//3)
    ema_Z = None; ema_alpha=0.8
    best_loss=float("inf"); best_state=None

    for epoch in range(1, args.epochs+1):
        model.train(); opt.zero_grad()
        z=model(X); z0=z[ref_idx].detach()
        z_axes=[]
        for j,g in enumerate(LABEL_COLS):
            idxs=groups[g]
            if len(idxs)==0:
                z_axes.append(torch.zeros(z.shape[1],device=device))
                continue
            d=z[idxs]-z0.unsqueeze(0)
            delta_y=(Y[idxs,j]-Y[ref_idx,j])
            v=principal_axis_with_orientation(d,delta_y)
            z_axes.append(v)
        L_ort,ema_Z=orthogonality_regularizer(z_axes,ema_Z,ema_alpha)
        L_sep=separation_loss(z,Yn)
        # warmup_scale=min(1.0,epoch/ort_warmup_epochs)
        warmup_scale=1.0
        loss=args.lambda_sep*L_sep+args.lambda_ort*warmup_scale*L_ort
        loss.backward(); opt.step()
        if loss.item()<best_loss:
            best_loss=loss.item()
            best_state={
                "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "feat_cols":feat_cols,"label_cols":LABEL_COLS,
                "y_min":y_min.cpu().tolist(),"y_max":y_max.cpu().tolist(),
                "ref_idx":ref_idx,"groups":groups,
                "latent_dim":args.latent_dim,
                "lambda_sep":args.lambda_sep,"lambda_ort":args.lambda_ort}
        if epoch%20==0 or epoch==1:
            print(f"Epoch {epoch:04d} Loss={loss.item():.6f} L_sep={L_sep.item():.6f} L_ort={L_ort.item():.6f}")

    # rebuild & compute reference-based scaling
    model.load_state_dict(best_state["model_state"]); model.eval()
    with torch.no_grad():
        z=model(X); z0=z[best_state["ref_idx"]]
        z_axes=[]
        for j,g in enumerate(LABEL_COLS):
            idxs=best_state["groups"][g]
            if len(idxs)==0:
                z_axes.append(torch.zeros(z.shape[1],device=device))
                continue
            d=z[idxs]-z0.unsqueeze(0)
            delta_y=(Y[idxs,j]-Y[best_state["ref_idx"],j])
            v=principal_axis_with_orientation(d,delta_y)
            z_axes.append(v)
        Z=F.normalize(torch.stack(z_axes,dim=0),dim=1)
        Q,_=torch.linalg.qr(Z.t(),mode="reduced")
        Z_ortho=Q.t().contiguous()
        # align positive correlation
        for j in range(6):
            proj=( (z - z0) @ Z_ortho[j] )
            corr=torch.sum(proj*(Y[:,j]-Y[best_state["ref_idx"],j]))
            if corr.item()<0: Z_ortho[j]=-Z_ortho[j]
        # compute scaling for inference
        S=(z - z0) @ Z_ortho.T  # [N,6]
        y0=Y[best_state["ref_idx"]]
        s_pos_max=torch.zeros(6,device=device); s_neg_min=torch.zeros(6,device=device)
        for j in range(6):
            mask_pos=Y[:,j]>=y0[j]; mask_neg=Y[:,j]<=y0[j]
            s_pos_max[j]=torch.max(S[mask_pos,j]) if mask_pos.any() else torch.tensor(1.0,device=device)
            s_neg_min[j]=torch.min(S[mask_neg,j]) if mask_neg.any() else torch.tensor(-1.0,device=device)

        artifacts={
            "model_state":{k:v.cpu() for k,v in model.state_dict().items()},
            "feat_cols":best_state["feat_cols"],
            "label_cols":best_state["label_cols"],
            "y_min":best_state["y_min"],"y_max":best_state["y_max"],
            "y0":y0.cpu().tolist(),
            "z0":z0.cpu(),
            "z_avg":[row.cpu() for row in Z_ortho],
            "s_pos_max":s_pos_max.cpu().tolist(),
            "s_neg_min":s_neg_min.cpu().tolist(),
            "latent_dim":best_state["latent_dim"],}

    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path=os.path.join(args.save_dir,f"soilx_pretrained_{timestamp}.pth")
    torch.save(artifacts,save_path)
    print("Saved:",save_path)

if __name__=="__main__":
    main()