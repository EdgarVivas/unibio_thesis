"""
vanilla_nh4_exo_o2.py – Vanilla TimeXer forecasting only T1_NH4,
with T1_O2 treated as one of 9 exogenous variables.

Exogenous encoding: each variable → single linear projection of whole series
→ 1 token (no patching, no TCN, no temporal info in exogenous).

Compare against train_nh4_exo_o2.py (same setup but TCN-enriched exogenous).

Run
---
    python vanilla_nh4_exo_o2.py
    python vanilla_nh4_exo_o2.py --epochs 50 --d_model 128 --n_heads 8 --n_layers 3 --d_ff 512
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Column definitions
# ─────────────────────────────────────────────────────────────────────────────

ENDO_COLS = ['T1_NH4']

EXO_COLS  = [
    'IN_METAL_Q', 'METAL_Q', 'TEMPERATURE', 'IN_Q',
    'MAX_CF', 'PROCESSPHASE_INLET', 'PROCESSPHASE_OUTLET', 'T1_PO4',
    'T1_O2',
]


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

class WWTPDataset(Dataset):
    def __init__(self, endo, exo, context_len, horizon, stride=1):
        self.endo    = torch.from_numpy(endo.astype(np.float32))
        self.exo     = torch.from_numpy(exo .astype(np.float32))
        self.ctx     = context_len
        self.hor     = horizon
        self.indices = list(range(0, len(endo) - context_len - horizon + 1, stride))

    def __len__(self): return len(self.indices)

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

    endo_tr = sc_endo.transform(endo[:n_train]);                exo_tr = sc_exo.transform(exo[:n_train])
    endo_va = sc_endo.transform(endo[n_train:n_train+n_val]);   exo_va = sc_exo.transform(exo[n_train:n_train+n_val])
    endo_te = sc_endo.transform(endo[n_train+n_val:]);          exo_te = sc_exo.transform(exo[n_train+n_val:])

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
# Model
# ─────────────────────────────────────────────────────────────────────────────

class _Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v):
        out, _ = self.attn(q, k, v)
        return self.norm(q + self.drop(out))


class _FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        return self.norm(x + self.net(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, d_model, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T = x.shape
        n = T // self.patch_size
        return self.drop(self.norm(self.proj(
            x[:, :n * self.patch_size].reshape(B, n, self.patch_size))))


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
        global_tok = seq[:, :1]
        patches    = seq[:, 1:]
        global_tok = self.cross_attn(global_tok, exo_kv, exo_kv)
        global_tok = self.ffn_glob(global_tok)
        patches    = self.ffn_patch(patches)
        return patches, global_tok


class VanillaNH4Model(nn.Module):
    """
    Vanilla TimeXer, NH4 only, O2 as exogenous.
    Exogenous: each variable → Linear(context_len → d_model) → 1 token.
    No TCN, no temporal structure in exogenous tokens.
    9 K/V tokens total for cross-attention (vs 270 in train_nh4_exo_o2.py).
    """
    def __init__(self, c_exo, context_len, horizon,
                 patch_size=12, d_model=64, n_heads=4, n_layers=2,
                 d_ff=256, dropout=0.1):
        super().__init__()
        assert context_len % patch_size == 0
        self.n_patches   = context_len // patch_size
        self.d_model     = d_model
        self.c_exo       = c_exo

        self.endo_patch   = PatchEmbed(patch_size, d_model, dropout)
        self.global_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.pos_emb      = nn.Parameter(torch.empty(1, self.n_patches + 1, d_model))
        nn.init.trunc_normal_(self.global_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb,      std=0.02)

        # One shared linear projection: full series → single token per variable
        self.exo_embed = nn.Sequential(
            nn.Linear(context_len, d_model),
            nn.LayerNorm(d_model),
        )

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x_endo, x_exo):
        # x_endo: [B, 1, T],  x_exo: [B, 9, T]
        B, _, T = x_endo.shape
        exo_kv  = self.exo_embed(x_exo.reshape(B * self.c_exo, T))\
                      .reshape(B, self.c_exo, self.d_model)         # [B, 9, d]
        patches = self.endo_patch(x_endo.squeeze(1))                # [B, N_p, d]
        glob    = self.global_token.expand(B, -1, -1)
        seq     = torch.cat([glob, patches], dim=1) + self.pos_emb
        glob    = seq[:, :1]
        patches = seq[:, 1:]
        for layer in self.layers:
            patches, glob = layer(patches, glob, exo_kv)
        last_value = x_endo[:, :, -1:]                              # [B, 1, 1] – last observed
        return self.head(glob) + last_value                         # residual anchor


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_endo, x_exo), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * x_endo.size(0); n += x_endo.size(0)
    return total / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        total += criterion(model(x_endo, x_exo), y).item() * x_endo.size(0)
        n += x_endo.size(0)
    return total / n


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x_endo, x_exo, y in loader:
        preds  .append(model(x_endo.to(device), x_exo.to(device)).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


# ─────────────────────────────────────────────────────────────────────────────
# Horizon plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_horizon(model, sc_endo, sc_exo, data_path, context_len, horizon,
                 out_path, n_windows=6, context_show=60, seed=42,
                 val_frac=0.1, test_frac=0.1):
    device = next(model.parameters()).device
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date').sort_index()
    df = df[ENDO_COLS + EXO_COLS].dropna()

    T       = len(df)
    n_test  = int(T * test_frac)
    n_val   = int(T * val_frac)
    n_train = T - n_val - n_test

    endo_raw = df[ENDO_COLS].values
    endo_te  = sc_endo.transform(endo_raw[n_train + n_val:])
    exo_te   = sc_exo .transform(df[EXO_COLS].values[n_train + n_val:])
    endo_te_raw = endo_raw[n_train + n_val:]

    rng    = np.random.default_rng(seed)
    starts = sorted(rng.choice(len(endo_te) - context_len - horizon,
                               size=n_windows, replace=False))
    model.eval()
    windows = []
    for s in starts:
        x_e = torch.tensor(endo_te[s:s+context_len].T, dtype=torch.float32).unsqueeze(0).to(device)
        x_x = torch.tensor(exo_te [s:s+context_len].T, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_e, x_x).squeeze(0).cpu().numpy()
        pred_orig = sc_endo.inverse_transform(pred.T).T
        windows.append({
            'window_idx':  int(s),
            'context_raw': endo_te_raw[s + context_len - context_show : s + context_len],
            'actual_raw':  endo_te_raw[s + context_len : s + context_len + horizon],
            'pred_orig':   pred_orig,
        })

    fig, axes = plt.subplots(n_windows, 1, figsize=(10, 3.5 * n_windows), squeeze=False)
    ctx_x  = np.arange(-context_show, 0)
    fore_x = np.arange(0, horizon)

    for row, wd in enumerate(windows):
        ax = axes[row][0]
        ax.plot(ctx_x,  wd['context_raw'][:, 0], color='#888888',
                linewidth=1.2, alpha=0.7, label='Context')
        ax.plot(fore_x, wd['actual_raw'][:, 0],  color='#1f77b4',
                linewidth=2.0, label='Actual NH4')
        ax.plot(fore_x, wd['pred_orig'][0],       color='#ff7f0e',
                linewidth=2.0, linestyle='--', label='Predicted NH4')
        ax.axvline(0, color='black', linewidth=0.8, linestyle=':')
        mae  = float(np.mean(np.abs(wd['pred_orig'][0] - wd['actual_raw'][:, 0])))
        rmse = float(np.sqrt(np.mean((wd['pred_orig'][0] - wd['actual_raw'][:, 0])**2)))
        ax.set_title(f'Window {wd["window_idx"]}  |  T1_NH4\n'
                     f'MAE={mae:.3f}  RMSE={rmse:.3f}', fontsize=9)
        ax.set_xlabel('Steps from forecast origin  (1 step ≈ 2 min)', fontsize=8)
        tick_pos = np.concatenate([np.arange(-context_show, 0, context_show//4),
                                   np.arange(0, horizon+1, max(1, horizon//6))])
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f'{int(t*2)}m' for t in tick_pos], fontsize=7)
        if row == 0: ax.legend(fontsize=8)

    fig.suptitle(f'Vanilla TimeXer — NH4 only, O2 as exogenous (1 token per variable)\n'
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Endogenous : {ENDO_COLS}")
    print(f"Exogenous  : {EXO_COLS}  ({len(EXO_COLS)} vars → {len(EXO_COLS)} K/V tokens)")

    print("\n── Loading data ──────────────────────────────────────────────")
    train_dl, val_dl, test_dl, sc_endo, sc_exo = build_dataloaders(
        args.data_path, args.context_len, args.horizon,
        args.batch_size, stride=args.stride,
        save_scalers_to=str(out_dir),
    )

    print("\n── Building model ────────────────────────────────────────────")
    model = VanillaNH4Model(
        c_exo=len(EXO_COLS), context_len=args.context_len, horizon=args.horizon,
        patch_size=args.patch_size, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    try:
        d = torch.randn(args.batch_size, 1,           args.context_len, device=device)
        x = torch.randn(args.batch_size, len(EXO_COLS), args.context_len, device=device)
        out  = model(d, x)
        loss = nn.MSELoss()(out, torch.zeros_like(out))
        loss.backward(); model.zero_grad()
        mem = torch.cuda.max_memory_allocated(device)/1e6 if device.type=='cuda' else 0
        print(f"  Memory probe: {mem:.1f} MB  output {list(out.shape)}  ✓")
    except RuntimeError as e:
        print(f"  Memory probe FAILED: {e}"); return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.MSELoss()

    print(f"\n── Training for up to {args.epochs} epochs "
          f"(patience={args.patience}) ───")
    history, best_val, patience_cnt = {'train':[], 'val':[]}, float('inf'), 0
    best_path = out_dir / 'best_model.pt'

    for epoch in range(1, args.epochs + 1):
        t0      = time.time()
        tr_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        va_loss = eval_epoch (model, val_dl,   criterion, device)
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
                  f"train={tr_loss:.5f}  val={va_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"patience={patience_cnt}/{args.patience}  │ {time.time()-t0:.1f}s")

        if patience_cnt >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}"); break

    print("\n── Test evaluation ───────────────────────────────────────────")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss = eval_epoch(model, test_dl, criterion, device)
    preds, targets = collect_predictions(model, test_dl, device)
    p_orig = sc_endo.inverse_transform(preds  [:,0,:])
    t_orig = sc_endo.inverse_transform(targets[:,0,:])
    mae  = float(np.mean(np.abs(p_orig - t_orig)))
    rmse = float(np.sqrt(np.mean((p_orig - t_orig)**2)))
    print(f"  Test MSE (scaled): {test_loss:.5f}")
    print(f"  T1_NH4  MAE={mae:.4f}  RMSE={rmse:.4f}")

    results = {'test_mse_scaled': float(test_loss),
               'metrics': {'T1_NH4': {'mae': mae, 'rmse': rmse}},
               'history': history, 'n_params': n_params, 'args': vars(args)}
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history['train'], label='Train'); ax.plot(history['val'], label='Val')
    ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.set_title('Vanilla NH4-only (O2 as exo, 1 token) — training history')
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / 'training_loss.png', dpi=150); plt.close(fig)

    plot_horizon(model, sc_endo, sc_exo, args.data_path,
                 args.context_len, args.horizon,
                 out_dir / 'horizon_structure_seed42.png')
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
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--patience',    type=int,   default=10)
    p.add_argument('--out_dir',     default='results3/vanilla_residual_best_config')
    main(p.parse_args())
