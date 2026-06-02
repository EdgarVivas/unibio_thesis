"""
train_nh4_gaussian.py – Probabilistic Gaussian forecast for T1_NH4,
with T1_O2 treated as exogenous (same setup as train_nh4_exo_o2.py).

Changes from the deterministic version
---------------------------------------
  Head   : Linear(d_model → horizon*2)  — outputs mean + log_var per step
  Loss   : Gaussian NLL instead of MSE
              L = 0.5 * (log_var + (y - mean)^2 / exp(log_var))
  Output : mean (point forecast, residual-anchored) + log_var (uncertainty)

Everything else is identical: same architecture, same data split, same
residual connection, same hyperparameter defaults.

Point forecast comparison
--------------------------
The mean is a valid point forecast. MSE/MAE on the mean is reported alongside
NLL so you can directly compare accuracy against the deterministic model.

Coverage check
--------------
After test evaluation the script reports what % of actual values fall inside
the 68%, 90% and 95% prediction intervals. If 95% interval coverage is below
~90%, the model is overconfident (uncertainty underestimated).

Usage
-----
    python train_nh4_gaussian.py
    python train_nh4_gaussian.py --epochs 50 --d_model 128 --n_heads 8 --n_layers 3 --d_ff 512
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t as scipy_t
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Column definitions  (same as train_nh4_exo_o2.py)
# ─────────────────────────────────────────────────────────────────────────────

ENDO_COLS = ['T1_NH4']
EXO_COLS  = [
    'IN_METAL_Q', 'METAL_Q', 'TEMPERATURE', 'IN_Q',
    'MAX_CF', 'PROCESSPHASE_INLET', 'PROCESSPHASE_OUTLET', 'T1_PO4',
    'T1_O2',
]


# ─────────────────────────────────────────────────────────────────────────────
# Data  (identical to train_nh4_exo_o2.py)
# ─────────────────────────────────────────────────────────────────────────────

class WWTPDataset(Dataset):
    def __init__(self, endo, exo, context_len, horizon, stride=1):
        self.endo    = torch.from_numpy(endo.astype(np.float32))
        self.exo     = torch.from_numpy(exo .astype(np.float32))
        self.ctx     = context_len
        self.hor     = horizon
        self.indices = list(range(0, len(endo) - context_len - horizon + 1, stride))

    def __len__(self):  return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (self.endo[i        : i + self.ctx          ].T,
                self.exo [i        : i + self.ctx          ].T,
                self.endo[i + self.ctx : i + self.ctx + self.hor].T)


def build_dataloaders(data_path, context_len, horizon, batch_size,
                      stride=1, val_frac=0.1, test_frac=0.1,
                      save_scalers_to=None):
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date').sort_index()
    df = df[ENDO_COLS + EXO_COLS].dropna()
    print(f"  Dataset size: {len(df):,} rows")

    T       = len(df)
    n_test  = int(T * test_frac)
    n_val   = int(T * val_frac)
    n_train = T - n_val - n_test
    print(f"  Split: train={n_train:,}  val={n_val:,}  test={n_test:,}")

    endo = df[ENDO_COLS].values
    exo  = df[EXO_COLS ].values

    sc_endo = StandardScaler().fit(endo[:n_train])
    sc_exo  = StandardScaler().fit(exo [:n_train])

    endo_tr = sc_endo.transform(endo[:n_train]);              exo_tr = sc_exo.transform(exo[:n_train])
    endo_va = sc_endo.transform(endo[n_train:n_train+n_val]); exo_va = sc_exo.transform(exo[n_train:n_train+n_val])
    endo_te = sc_endo.transform(endo[n_train+n_val:]);        exo_te = sc_exo.transform(exo[n_train+n_val:])

    kw = dict(context_len=context_len, horizon=horizon, stride=stride)
    nw = min(4, __import__('os').cpu_count() or 1)
    train_dl = DataLoader(WWTPDataset(endo_tr, exo_tr, **kw), batch_size=batch_size,
                          shuffle=True,  num_workers=nw, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(WWTPDataset(endo_va, exo_va, **kw), batch_size=batch_size,
                          shuffle=False, num_workers=nw, pin_memory=True)
    test_dl  = DataLoader(WWTPDataset(endo_te, exo_te, **kw), batch_size=batch_size,
                          shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Samples: train={len(train_dl.dataset):,}  "
          f"val={len(val_dl.dataset):,}  test={len(test_dl.dataset):,}")

    if save_scalers_to:
        p = Path(save_scalers_to); p.mkdir(parents=True, exist_ok=True)
        joblib.dump(sc_endo, p / 'scaler_endo.joblib')
        joblib.dump(sc_exo,  p / 'scaler_exo.joblib')

    return train_dl, val_dl, test_dl, sc_endo, sc_exo


# ─────────────────────────────────────────────────────────────────────────────
# Model building blocks  (identical to train_nh4_exo_o2.py)
# ─────────────────────────────────────────────────────────────────────────────

class _Chomp1d(nn.Module):
    def __init__(self, size): super().__init__(); self.size = size
    def forward(self, x): return x[..., :-self.size].contiguous() if self.size > 0 else x

class _TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                           dilation=dilation, padding=pad)),
            _Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                           dilation=dilation, padding=pad)),
            _Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)
    def forward(self, x):
        return self.norm((self.net(x) + x).transpose(1, 2)).transpose(1, 2)

class SharedTCN(nn.Module):
    def __init__(self, d_model, n_levels=3, kernel_size=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            _TCNBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_levels)
        ])
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, d_model, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T = x.shape; n = T // self.patch_size
        return self.drop(self.norm(self.proj(
            x[:, :n * self.patch_size].reshape(B, n, self.patch_size))))

class _Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model); self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v):
        out, _ = self.attn(q, k, v)
        return self.norm(q + self.drop(out))

class _FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x): return self.norm(x + self.net(x))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn  = _Attention(d_model, n_heads, dropout)
        self.cross_attn = _Attention(d_model, n_heads, dropout)
        self.ffn_glob   = _FFN(d_model, d_ff, dropout)
        self.ffn_patch  = _FFN(d_model, d_ff, dropout)
    def forward(self, patches, global_tok, exo_kv):
        seq        = torch.cat([global_tok, patches], dim=1)
        seq        = self.self_attn(seq, seq, seq)
        global_tok = seq[:, :1]; patches = seq[:, 1:]
        global_tok = self.cross_attn(global_tok, exo_kv, exo_kv)
        return self.ffn_patch(patches), self.ffn_glob(global_tok)


# ─────────────────────────────────────────────────────────────────────────────
# Probabilistic model
# ─────────────────────────────────────────────────────────────────────────────

class NH4GaussianModel(nn.Module):
    """
    Same architecture as NH4Model in train_nh4_exo_o2.py.

    Gaussian:   head outputs horizon*2 → (mean, log_var)
    Student-t:  head outputs horizon*3 → (mean, log_scale, raw_df)
                df = softplus(raw_df) + 2.1  — learned per timestep per sample, df > 2

    Forward returns (mean, log_aux, df_tensor):
      Gaussian:  df_tensor is None
      Student-t: df_tensor shape [B, 1, horizon], values > 2
    """
    def __init__(self, c_exo, context_len, horizon, dist='gaussian',
                 patch_size=12, d_model=64, n_heads=4, n_layers=2,
                 d_ff=256, tcn_levels=3, tcn_kernel=3, dropout=0.1):
        super().__init__()
        assert context_len % patch_size == 0
        self.horizon   = horizon
        self.dist      = dist
        self.n_patches = context_len // patch_size
        self.d_model   = d_model
        self.c_exo     = c_exo

        self.endo_patch   = PatchEmbed(patch_size, d_model, dropout)
        self.global_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.pos_emb      = nn.Parameter(torch.empty(1, self.n_patches + 1, d_model))
        nn.init.trunc_normal_(self.global_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb,      std=0.02)

        self.exo_patch  = PatchEmbed(patch_size, d_model, dropout)
        self.shared_tcn = SharedTCN(d_model, tcn_levels, tcn_kernel, dropout)
        self.var_emb    = nn.Parameter(torch.empty(c_exo, 1, d_model))
        nn.init.trunc_normal_(self.var_emb, std=0.02)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        n_out = horizon * 3 if dist == 'student_t' else horizon * 2
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_out),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _encode_exo(self, x_exo):
        B, C_exo, T = x_exo.shape; N = self.n_patches
        patches = self.exo_patch(x_exo.reshape(B * C_exo, T))
        tcn_out = self.shared_tcn(patches.transpose(1, 2)).transpose(1, 2)
        var_id  = self.var_emb.expand(C_exo, N, -1).unsqueeze(0).expand(B,-1,-1,-1)
        tcn_out = tcn_out + var_id.reshape(B * C_exo, N, self.d_model)
        return tcn_out.reshape(B, C_exo * N, self.d_model)

    def forward(self, x_endo, x_exo):
        B = x_endo.shape[0]
        exo_kv  = self._encode_exo(x_exo)
        patches = self.endo_patch(x_endo.squeeze(1))
        glob    = self.global_token.expand(B, -1, -1)
        seq     = torch.cat([glob, patches], dim=1) + self.pos_emb
        glob    = seq[:, :1]; patches = seq[:, 1:]
        for layer in self.layers:
            patches, glob = layer(patches, glob, exo_kv)

        out        = self.head(glob)                       # [B, 1, n_out]
        H          = self.horizon
        mean       = out[:, :, :H] + x_endo[:, :, -1:]   # residual anchor
        log_aux    = out[:, :, H:2*H]

        if self.dist == 'student_t':
            df = F.softplus(out[:, :, 2*H:]) + 2.1        # learned df > 2, per step
            return mean, log_aux, df
        return mean, log_aux, None


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

LOG_VAR_MIN, LOG_VAR_MAX = -6.0, 6.0   # clamp to keep training stable

def gaussian_nll(mean, log_var, df, target):
    """log_var is log(sigma^2). df unused."""
    log_var = log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    return 0.5 * (log_var + (target - mean).pow(2) / log_var.exp()).mean()


def student_t_nll(mean, log_scale, df, target):
    """log_scale is log(sigma). df is a tensor [B, 1, H] — learned per step."""
    log_scale = log_scale.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    return -torch.distributions.StudentT(
        df=df, loc=mean, scale=log_scale.exp()
    ).log_prob(target).mean()


def make_loss_fn(dist):
    return gaussian_nll if dist == 'gaussian' else student_t_nll


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total, n = 0.0, 0
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        optimizer.zero_grad()
        mean, log_aux, df = model(x_endo, x_exo)
        loss = loss_fn(mean, log_aux, df, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * x_endo.size(0); n += x_endo.size(0)
    return total / n


@torch.no_grad()
def eval_epoch(model, loader, device, loss_fn):
    model.eval()
    total, n = 0.0, 0
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        mean, log_aux, df = model(x_endo, x_exo)
        total += loss_fn(mean, log_aux, df, y).item() * x_endo.size(0)
        n += x_endo.size(0)
    return total / n


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    means, log_auxs, dfs, targets = [], [], [], []
    for x_endo, x_exo, y in loader:
        m, la, df = model(x_endo.to(device), x_exo.to(device))
        means   .append(m .cpu().numpy())
        log_auxs.append(la.cpu().numpy())
        dfs     .append(df.cpu().numpy() if df is not None else None)
        targets .append(y .numpy())
    return (np.concatenate(means),
            np.concatenate(log_auxs),
            np.concatenate(dfs) if dfs[0] is not None else None,
            np.concatenate(targets))


# ─────────────────────────────────────────────────────────────────────────────
# Coverage check
# ─────────────────────────────────────────────────────────────────────────────

def coverage_report(means, log_aux, dfs, targets, dist='gaussian'):
    """
    Check what fraction of actual values fall inside prediction intervals.
    Gaussian:   stds from log_var (log sigma^2), normal quantiles.
    Student-t:  scale from log_scale (log sigma), per-sample t quantiles using learned df.
    """
    if dist == 'gaussian':
        stds = np.exp(0.5 * np.clip(log_aux, LOG_VAR_MIN, LOG_VAR_MAX))
        z68  = scipy_t.ppf(0.84,  1e9)   # ~1.000
        z90  = scipy_t.ppf(0.95,  1e9)   # ~1.645
        z95  = scipy_t.ppf(0.975, 1e9)   # ~1.960
    else:
        stds = np.exp(np.clip(log_aux, LOG_VAR_MIN, LOG_VAR_MAX))
        # dfs is per-sample per-step — scipy.stats.t.ppf accepts arrays
        z68  = scipy_t.ppf(0.84,  dfs)
        z90  = scipy_t.ppf(0.95,  dfs)
        z95  = scipy_t.ppf(0.975, dfs)
    diff = np.abs(targets - means)
    return {
        '68%': float((diff <= z68 * stds).mean()),
        '90%': float((diff <= z90 * stds).mean()),
        '95%': float((diff <= z95 * stds).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_horizon(model, sc_endo, sc_exo, data_path, context_len, horizon,
                 out_path, dist='gaussian',
                 n_windows=6, context_show=60, seed=42,
                 val_frac=0.1, test_frac=0.1):
    device = next(model.parameters()).device
    raw_df = pd.read_csv(data_path)
    raw_df['date'] = pd.to_datetime(raw_df['date'], utc=True)
    raw_df = raw_df.set_index('date').sort_index()
    raw_df = raw_df[ENDO_COLS + EXO_COLS].dropna()

    T = len(raw_df); n_test = int(T * test_frac); n_val = int(T * val_frac)
    n_train = T - n_val - n_test

    endo_raw = raw_df[ENDO_COLS].values
    endo_te  = sc_endo.transform(endo_raw[n_train + n_val:])
    exo_te   = sc_exo .transform(raw_df[EXO_COLS].values[n_train + n_val:])
    endo_te_raw = endo_raw[n_train + n_val:]

    rng    = np.random.default_rng(seed)
    starts = sorted(rng.choice(len(endo_te) - context_len - horizon,
                               size=n_windows, replace=False))
    model.eval()
    fig, axes = plt.subplots(n_windows, 1, figsize=(11, 4 * n_windows), squeeze=False)
    ctx_x = np.arange(-context_show, 0)
    fore_x = np.arange(0, horizon)

    for row, s in enumerate(starts):
        x_e = torch.tensor(endo_te[s:s+context_len].T, dtype=torch.float32).unsqueeze(0).to(device)
        x_x = torch.tensor(exo_te [s:s+context_len].T, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mean_sc, lv_sc, df_sc = model(x_e, x_x)   # [1, 1, H]

        mean_sc = mean_sc.squeeze().cpu().numpy()    # [H]
        lv_sc   = lv_sc  .squeeze().cpu().numpy()
        if dist == 'gaussian':
            std_sc = np.exp(0.5 * np.clip(lv_sc, LOG_VAR_MIN, LOG_VAR_MAX))
            z68 = np.full(horizon, scipy_t.ppf(0.84,  1e9))
            z95 = np.full(horizon, scipy_t.ppf(0.975, 1e9))
        else:
            std_sc = np.exp(np.clip(lv_sc, LOG_VAR_MIN, LOG_VAR_MAX))
            df_np  = df_sc.squeeze().cpu().numpy()     # [H] — learned per step
            z68    = scipy_t.ppf(0.84,  df_np)
            z95    = scipy_t.ppf(0.975, df_np)

        # Inverse-transform mean and ±1σ / ±1.96σ bounds
        def inv(arr):
            return sc_endo.inverse_transform(arr.reshape(-1, 1)).ravel()

        mean_orig   = inv(mean_sc)
        upper_1s    = inv(mean_sc + z68 * std_sc)
        lower_1s    = inv(mean_sc - z68 * std_sc)
        upper_95    = inv(mean_sc + z95 * std_sc)
        lower_95    = inv(mean_sc - z95 * std_sc)
        ctx_raw     = endo_te_raw[s + context_len - context_show : s + context_len, 0]
        actual_raw  = endo_te_raw[s + context_len : s + context_len + horizon, 0]

        ax = axes[row][0]
        ax.plot(ctx_x,  ctx_raw,    color='#888888', linewidth=1.2, alpha=0.7, label='Context')
        ax.fill_between(fore_x, lower_95, upper_95,  color='#ff7f0e', alpha=0.15, label='95% PI')
        ax.fill_between(fore_x, lower_1s, upper_1s,  color='#ff7f0e', alpha=0.30, label='68% PI')
        ax.plot(fore_x, mean_orig,  color='#ff7f0e', linewidth=2.0, label='Mean forecast')
        ax.plot(fore_x, actual_raw, color='#1f77b4', linewidth=2.0, label='Actual NH4')
        ax.axvline(0, color='black', linewidth=0.8, linestyle=':')

        mae  = float(np.mean(np.abs(mean_orig - actual_raw)))
        rmse = float(np.sqrt(np.mean((mean_orig - actual_raw)**2)))
        cov  = float(np.mean(np.abs(actual_raw - mean_orig) <= z95 *
                             sc_endo.scale_[0] * std_sc))
        ax.set_title(f'Window {s}  |  T1_NH4  |  MAE={mae:.3f}  RMSE={rmse:.3f}  '
                     f'95%-coverage={cov:.0%}', fontsize=9)
        ax.set_xlabel('Steps from forecast origin  (1 step ≈ 2 min)', fontsize=8)
        tick_pos = np.concatenate([np.arange(-context_show, 0, context_show//4),
                                   np.arange(0, horizon+1, max(1, horizon//6))])
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f'{int(t*2)}m' for t in tick_pos], fontsize=7)
        if row == 0: ax.legend(fontsize=8, loc='upper left')

    dist_label = 'Gaussian' if dist == 'gaussian' else 'Student-t (learned df)'
    fig.suptitle(f'NH4 {dist_label} probabilistic forecast (O2 as exogenous)\n'
                 f'Full {horizon}-step horizon  ({horizon*2} min ahead)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = make_loss_fn(args.dist)
    print(f"Device: {device}")
    print(f"Distribution: {args.dist}" + (" (df learned per timestep)" if args.dist == 'student_t' else ""))

    print("\n── Loading data ──────────────────────────────────────────────")
    train_dl, val_dl, test_dl, sc_endo, sc_exo = build_dataloaders(
        args.data_path, args.context_len, args.horizon,
        args.batch_size, stride=args.stride, save_scalers_to=str(out_dir))

    print("\n── Building model ────────────────────────────────────────────")
    model = NH4GaussianModel(
        c_exo=len(EXO_COLS), context_len=args.context_len, horizon=args.horizon,
        dist=args.dist, patch_size=args.patch_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff,
        tcn_levels=args.tcn_levels, tcn_kernel=args.tcn_kernel, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_out = args.horizon * 3 if args.dist == 'student_t' else args.horizon * 2
    print(f"  Parameters: {n_params:,}  (head outputs {n_out})")

    # Memory probe
    try:
        xe = torch.randn(args.batch_size, 1,            args.context_len, device=device)
        xx = torch.randn(args.batch_size, len(EXO_COLS), args.context_len, device=device)
        m, lv, df = model(xe, xx)
        loss = loss_fn(m, lv, df, torch.zeros_like(m))
        loss.backward(); model.zero_grad()
        mem = torch.cuda.max_memory_allocated(device)/1e6 if device.type=='cuda' else 0
        df_info = f'  df={list(df.shape)}' if df is not None else ''
        print(f"  Memory probe: {mem:.1f} MB  mean={list(m.shape)}  log_aux={list(lv.shape)}{df_info}  ✓")
    except RuntimeError as e:
        print(f"  Memory probe FAILED: {e}"); return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    dist_label = 'Gaussian NLL' if args.dist == 'gaussian' else 'Student-t NLL (learned df)'
    print(f"\n── Training for up to {args.epochs} epochs "
          f"(patience={args.patience}, loss={dist_label}) ───")
    history      = {'train': [], 'val': []}
    best_val, patience_cnt = float('inf'), 0
    best_path = out_dir / 'best_model.pt'

    for epoch in range(1, args.epochs + 1):
        t0      = time.time()
        tr_loss = train_epoch(model, train_dl, optimizer, device, loss_fn)
        va_loss = eval_epoch (model, val_dl,   device, loss_fn)
        scheduler.step()
        history['train'].append(float(tr_loss))
        history['val'  ].append(float(va_loss))

        if va_loss < best_val:
            best_val, patience_cnt = va_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"  ep {epoch:4d}/{args.epochs} │ "
                  f"NLL train={tr_loss:.5f}  val={va_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"patience={patience_cnt}/{args.patience}  │ {time.time()-t0:.1f}s")

        if patience_cnt >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}"); break

    print("\n── Test evaluation ───────────────────────────────────────────")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_nll = eval_epoch(model, test_dl, device, loss_fn)
    print(f"  Test NLL (scaled): {test_nll:.5f}")

    means, log_auxs, dfs, targets = collect_predictions(model, test_dl, device)
    # means / log_vars / targets: [N, 1, H]

    # Point forecast accuracy (mean vs actual, original scale)
    N, _, H = means.shape
    m_orig = sc_endo.inverse_transform(means[:,0,:].reshape(-1,1)).reshape(N, H)
    t_orig = sc_endo.inverse_transform(targets[:,0,:].reshape(-1,1)).reshape(N, H)
    mae  = float(np.mean(np.abs(m_orig - t_orig)))
    rmse = float(np.sqrt(np.mean((m_orig - t_orig)**2)))
    mse  = float(np.mean((means[:,0,:] - targets[:,0,:])**2))   # scaled, for comparison
    print(f"\n  Point forecast (mean):")
    print(f"    MSE (scaled) = {mse:.5f}   ← compare directly to deterministic model")
    print(f"    MAE (orig)   = {mae:.4f} mg/L")
    print(f"    RMSE (orig)  = {rmse:.4f} mg/L")

    # Coverage
    cov = coverage_report(means[:,0,:], log_auxs[:,0,:],
                          dfs[:,0,:] if dfs is not None else None,
                          targets[:,0,:], dist=args.dist)
    print(f"\n  Calibration (scaled space):")
    for label, frac in cov.items():
        flag = '✓' if abs(frac - float(label[:2])/100) < 0.08 else '✗ overconfident' if frac < float(label[:2])/100 else '~ conservative'
        print(f"    {label}: {frac:.1%}  {flag}")

    results = {
        'test_nll_scaled': float(test_nll),
        'test_mse_scaled': float(mse),
        'metrics': {'T1_NH4': {'mae': mae, 'rmse': rmse}},
        'calibration': cov,
        'history': history, 'n_params': n_params, 'args': vars(args),
        'distribution': args.dist,
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history['train'], label='Train NLL'); ax.plot(history['val'], label='Val NLL')
    ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Gaussian NLL')
    ax.set_title(f'NH4 {dist_label} — training history'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / 'training_loss.png', dpi=150); plt.close(fig)

    plot_horizon(model, sc_endo, sc_exo, args.data_path,
                 args.context_len, args.horizon,
                 out_dir / 'horizon_structure_seed42.png',
                 dist=args.dist)
    print("\nDone.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_path',   default='../dataset/agtrup.csv')
    p.add_argument('--context_len', type=int,   default=360)
    p.add_argument('--horizon',     type=int,   default=18)
    p.add_argument('--stride',      type=int,   default=1)
    p.add_argument('--patch_size',  type=int,   default=12)
    p.add_argument('--d_model',     type=int,   default=64)
    p.add_argument('--n_heads',     type=int,   default=4)
    p.add_argument('--n_layers',    type=int,   default=2)
    p.add_argument('--d_ff',        type=int,   default=256)
    p.add_argument('--tcn_levels',  type=int,   default=3)
    p.add_argument('--tcn_kernel',  type=int,   default=3)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--patience',    type=int,   default=10)
    p.add_argument('--dist',        choices=['gaussian', 'student_t'], default='gaussian')
    p.add_argument('--out_dir',     default='results4/nh4_student_t_18')
    main(p.parse_args())
