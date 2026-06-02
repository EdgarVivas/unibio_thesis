"""
vanilla_timexer.py – Vanilla TimeXer baseline for WWTP O2/NH4 forecasting.

Architecture (standard TimeXer, no modifications):
  Endogenous  : patch → embed → [global token | N patches] → self-attention
  Exogenous   : each variable → linear projection of whole series → 1 token each
  Bridge      : cross-attention: global token (Q) ← C_exo tokens (K, V)
  Output      : global token → Linear → horizon
  Note        : channel-independent (O2 and NH4 never interact)

Run
---
    python vanilla_timexer.py                          # default settings
    python vanilla_timexer.py --epochs 100 --patience 15 --stride 3
    python vanilla_timexer.py --help
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
import torch
import torch.nn as nn

from data import build_dataloaders, ENDO_COLS, REAL_EXO_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class _Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v):
        out, _ = self.attn(q, k, v)
        return self.norm(q + self.drop(out))


class _FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
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
    def __init__(self, patch_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):           # x: [B, T]
        B, T = x.shape
        n = T // self.patch_size
        x = x[:, :n * self.patch_size].reshape(B, n, self.patch_size)
        return self.drop(self.norm(self.proj(x)))   # [B, N_patches, d_model]


class VanillaTimeXerLayer(nn.Module):
    """
    Standard TimeXer encoder layer.
      1. Self-attention : [global | patches] attend within each variable
      2. Cross-attention: global token (Q) ← exogenous single tokens (K, V)
      3. FFN
    No cross-variate coupling between endogenous variables.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = _Attention(d_model, n_heads, dropout)
        self.cross_attn = _Attention(d_model, n_heads, dropout)
        self.ffn_glob   = _FFN(d_model, d_ff, dropout)
        self.ffn_patch  = _FFN(d_model, d_ff, dropout)

    def forward(self, patches, global_tok, exo_kv):
        # [B*C_endo, N_p, d]  [B*C_endo, 1, d]  [B*C_endo, C_exo, d]
        seq        = torch.cat([global_tok, patches], dim=1)
        seq        = self.self_attn(seq, seq, seq)
        global_tok = seq[:, :1]
        patches    = seq[:, 1:]

        global_tok = self.cross_attn(global_tok, exo_kv, exo_kv)

        global_tok = self.ffn_glob(global_tok)
        patches    = self.ffn_patch(patches)
        return patches, global_tok


class VanillaTimeXer(nn.Module):
    """
    Vanilla TimeXer.

    Key difference from ModifiedTimeXer:
      - Exogenous variables are each summarised into ONE token via a linear
        projection of the full series (no patching, no TCN, no temporal info).
      - Only C_exo=8 tokens for cross-attention (vs 240 in the modified version).
      - No cross-variate coupling between O2 and NH4.
    """
    def __init__(self,
                 c_endo:      int,
                 c_exo:       int,
                 context_len: int,
                 horizon:     int,
                 patch_size:  int   = 12,
                 d_model:     int   = 64,
                 n_heads:     int   = 4,
                 n_layers:    int   = 2,
                 d_ff:        int   = 256,
                 dropout:     float = 0.1):
        super().__init__()

        assert context_len % patch_size == 0
        self.c_endo    = c_endo
        self.c_exo     = c_exo
        self.n_patches = context_len // patch_size
        self.d_model   = d_model

        # ── Endogenous
        self.endo_patch   = PatchEmbed(patch_size, d_model, dropout)
        self.global_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.pos_emb      = nn.Parameter(
            torch.empty(1, self.n_patches + 1, d_model))
        nn.init.trunc_normal_(self.global_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb,      std=0.02)

        # ── Exogenous: one linear per variable  (shared weights across variables)
        # Projects the full context series [T] → one d_model token
        self.exo_embed = nn.Sequential(
            nn.Linear(context_len, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Encoder
        self.layers = nn.ModuleList([
            VanillaTimeXerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # ── Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_exogenous(self, x_exo):
        # x_exo: [B, C_exo, T]
        B, C_exo, T = x_exo.shape
        x_flat  = x_exo.reshape(B * C_exo, T)          # [B*C_exo, T]
        tokens  = self.exo_embed(x_flat)                # [B*C_exo, d]
        return tokens.reshape(B, C_exo, self.d_model)   # [B, C_exo, d]

    def forward(self, x_endo, x_exo):
        B, C_endo, T = x_endo.shape

        # Exogenous: [B, C_exo, d]
        exo_kv = self._encode_exogenous(x_exo)

        # Expand exo to [B*C_endo, C_exo, d]
        exo_kv_exp = (exo_kv
                      .unsqueeze(1)
                      .expand(-1, C_endo, -1, -1)
                      .reshape(B * C_endo, self.c_exo, self.d_model))

        # Endogenous patches
        patches = self.endo_patch(x_endo.reshape(B * C_endo, T))  # [B*C, N_p, d]
        glob    = self.global_token.expand(B * C_endo, -1, -1)

        seq     = torch.cat([glob, patches], dim=1) + self.pos_emb
        glob    = seq[:, :1]
        patches = seq[:, 1:]

        for layer in self.layers:
            patches, glob = layer(patches, glob, exo_kv_exp)

        out = self.head(glob.squeeze(1).reshape(B, C_endo, self.d_model))
        return out   # [B, C_endo, horizon]


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total, n = 0.0, 0
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_endo, x_exo), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * x_endo.size(0)
        n     += x_endo.size(0)
    return total / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        total += criterion(model(x_endo, x_exo), y).item() * x_endo.size(0)
        n     += x_endo.size(0)
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
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history['train'], label='Train MSE', linewidth=2)
    ax.plot(history['val'],   label='Val MSE',   linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (scaled)')
    ax.set_title('Vanilla TimeXer – training history')
    ax.legend(); ax.grid(alpha=0.3); ax.set_yscale('log')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def plot_last_step(preds, targets, sc_endo, out_dir, n_show=300):
    """Quick overview: last horizon step across n_show test samples."""
    N, C, H   = preds.shape
    p2 = sc_endo.inverse_transform(preds[:, :, -1])
    t2 = sc_endo.inverse_transform(targets[:, :, -1])

    fig, axes = plt.subplots(C, 1, figsize=(14, 4 * C), sharex=True)
    if C == 1: axes = [axes]
    for i, (name, ax) in enumerate(zip(ENDO_COLS, axes)):
        ax.plot(t2[:n_show, i], label='Actual',    linewidth=1.5, alpha=0.8)
        ax.plot(p2[:n_show, i], label='Predicted', linewidth=1.5, alpha=0.8)
        mae  = np.mean(np.abs(p2[:, i] - t2[:, i]))
        rmse = np.sqrt(np.mean((p2[:, i] - t2[:, i])**2))
        ax.set_title(f'{name}  |  MAE={mae:.4f}  RMSE={rmse:.4f}  '
                     f'(last horizon step, original scale)')
        ax.legend(); ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Test sample index')
    fig.tight_layout()
    path = out_dir / 'test_predictions.png'
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def plot_horizon_windows(model, endo_sc, exo_sc, endo_raw,
                         sc_endo, context_len, horizon,
                         out_dir, device,
                         n_windows=6, context_show=60, seed=42):
    """
    For n_windows random test windows: show full 18-step forecast horizon
    vs actual, with the last context_show steps of history on the left.
    """
    rng   = np.random.default_rng(seed)
    max_s = len(endo_sc) - context_len - horizon
    starts = sorted(rng.choice(max_s, size=n_windows, replace=False))

    n_vars = len(ENDO_COLS)
    fig, axes = plt.subplots(n_windows, n_vars,
                              figsize=(7 * n_vars, 3.5 * n_windows),
                              squeeze=False)

    model.eval()
    for row, s in enumerate(starts):
        x_endo = torch.tensor(
            endo_sc[s: s + context_len].T, dtype=torch.float32
        ).unsqueeze(0).to(device)
        x_exo  = torch.tensor(
            exo_sc[s: s + context_len].T, dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_sc = model(x_endo, x_exo).squeeze(0).cpu().numpy()  # [C, H]

        pred_orig = sc_endo.inverse_transform(pred_sc.T).T           # [C, H]
        ctx_raw   = endo_raw[s + context_len - context_show
                              : s + context_len]                      # [ctx_show, C]
        act_raw   = endo_raw[s + context_len
                              : s + context_len + horizon]            # [H, C]

        ctx_x  = np.arange(-context_show, 0)
        fore_x = np.arange(0, horizon)

        for col, name in enumerate(ENDO_COLS):
            ax = axes[row][col]
            ax.plot(ctx_x,  ctx_raw[:, col],
                    color='#888888', linewidth=1.2, alpha=0.7, label='Context')
            ax.plot(fore_x, act_raw[:, col],
                    color='#1f77b4', linewidth=2.0, label='Actual')
            ax.plot(fore_x, pred_orig[col],
                    color='#ff7f0e', linewidth=2.0, linestyle='--', label='Predicted')
            ax.axvline(0, color='black', linewidth=0.8, linestyle=':')

            mae  = np.mean(np.abs(pred_orig[col] - act_raw[:, col]))
            rmse = np.sqrt(np.mean((pred_orig[col] - act_raw[:, col])**2))
            ax.set_title(f'Window {s}  |  {name}\nMAE={mae:.3f}  RMSE={rmse:.3f}',
                         fontsize=9)
            ax.set_xlabel('Steps from forecast origin  (1 step ≈ 2 min)', fontsize=8)
            ax.set_ylabel(name, fontsize=8)

            ticks = np.concatenate([
                np.arange(-context_show, 0, context_show // 4),
                np.arange(0, horizon + 1, max(1, horizon // 6)),
            ])
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{int(t*2)}m' for t in ticks], fontsize=7)
            if row == 0 and col == n_vars - 1:
                ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(
        f'Vanilla TimeXer – full {horizon}-step forecast horizon\n'
        f'Context shown: last {context_show} steps ({context_show*2} min)',
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    path = out_dir / 'horizon_structure.png'
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device       : {device}")
    print(f"Context len  : {args.context_len}  ({args.context_len*2} min  ≈  "
          f"{args.context_len*2/60:.1f}h)")
    print(f"Horizon      : {args.horizon}  ({args.horizon*2} min)")
    print(f"Exo tokens   : {len(REAL_EXO_COLS)} (one per variable, no patching)")

    # ── Data
    print("\n── Loading data ──────────────────────────────────────────────")
    train_dl, val_dl, test_dl, sc_endo, sc_exo, exo_cols = build_dataloaders(
        data_path       = args.data_path,
        context_len     = args.context_len,
        horizon         = args.horizon,
        batch_size      = args.batch_size,
        stride          = args.stride,
        save_scalers_to = str(out_dir),
    )
    print(f"  Batches/epoch: {len(train_dl):,}")

    # ── Model
    print("\n── Building model ────────────────────────────────────────────")
    model = VanillaTimeXer(
        c_endo      = len(ENDO_COLS),
        c_exo       = len(REAL_EXO_COLS),
        context_len = args.context_len,
        horizon     = args.horizon,
        patch_size  = args.patch_size,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        d_ff        = args.d_ff,
        dropout     = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Memory probe
    print("\n── Memory probe ──────────────────────────────────────────────")
    try:
        d_endo = torch.randn(args.batch_size, len(ENDO_COLS),     args.context_len, device=device)
        d_exo  = torch.randn(args.batch_size, len(REAL_EXO_COLS), args.context_len, device=device)
        d_y    = torch.randn(args.batch_size, len(ENDO_COLS), args.horizon, device=device)
        out    = model(d_endo, d_exo)
        nn.MSELoss()(out, d_y).backward()
        model.zero_grad()
        if device.type == 'cuda':
            print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated(device)/1e6:.1f} MB")
        print(f"  Output shape   : {list(out.shape)}  ✓")
        del d_endo, d_exo, d_y, out
    except RuntimeError as e:
        print(f"  FAILED: {e}")
        return

    # ── Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.MSELoss()

    # ── Training loop
    print(f"\n── Training for up to {args.epochs} epochs "
          f"(patience={args.patience}) ───")
    history      = {'train': [], 'val': []}
    best_val     = float('inf')
    best_path    = out_dir / 'best_model.pt'
    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        t0      = time.time()
        tr_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        va_loss = eval_epoch(model, val_dl,   criterion, device)
        scheduler.step()

        history['train'].append(float(tr_loss))
        history['val'  ].append(float(va_loss))

        if va_loss < best_val:
            best_val     = va_loss
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"  ep {epoch:4d}/{args.epochs} │ "
                  f"train={tr_loss:.5f}  val={va_loss:.5f}  "
                  f"patience={patience_cnt}/{args.patience}"
                  f"  │ {time.time()-t0:.1f}s")

        if patience_cnt >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── Test
    print("\n── Test evaluation ───────────────────────────────────────────")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss = eval_epoch(model, test_dl, criterion, device)
    print(f"  Test MSE (scaled): {test_loss:.5f}")

    preds, targets = collect_predictions(model, test_dl, device)
    N, C, H = preds.shape
    p2 = preds  .transpose(0, 2, 1).reshape(N * H, C)
    t2 = targets.transpose(0, 2, 1).reshape(N * H, C)
    p_orig = sc_endo.inverse_transform(p2).reshape(N, H, C).transpose(0, 2, 1)
    t_orig = sc_endo.inverse_transform(t2).reshape(N, H, C).transpose(0, 2, 1)

    metrics = {}
    print(f"\n  {'Variable':<12}  {'MAE':>8}  {'RMSE':>8}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*8}")
    for i, name in enumerate(ENDO_COLS):
        mae  = float(np.mean(np.abs(p_orig[:, i] - t_orig[:, i])))
        rmse = float(np.sqrt(np.mean((p_orig[:, i] - t_orig[:, i])**2)))
        metrics[name] = {'mae': mae, 'rmse': rmse}
        print(f"  {name:<12}  {mae:>8.4f}  {rmse:>8.4f}")

    results = {
        'model'          : 'VanillaTimeXer',
        'test_mse_scaled': float(test_loss),
        'metrics'        : metrics,
        'history'        : history,
        'n_params'       : n_params,
        'args'           : vars(args),
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_dir}/results.json")

    # ── Plots
    print("\n── Plotting ──────────────────────────────────────────────────")
    plot_loss(history, out_dir / 'training_loss.png')
    plot_last_step(preds, targets, sc_endo, out_dir)

    # Full horizon plot: load raw test data for inverse-transform display
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(args.data_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date').sort_index()
    df = df[ENDO_COLS + REAL_EXO_COLS].dropna()
    T_total = len(df)
    n_test  = int(T_total * 0.1)
    n_val   = int(T_total * 0.1)
    n_train = T_total - n_val - n_test
    endo_raw_full = df[ENDO_COLS].values
    exo_full      = df[REAL_EXO_COLS].values
    sc_e2 = joblib.load(out_dir / 'scaler_endo.joblib')
    sc_x2 = joblib.load(out_dir / 'scaler_exo.joblib')
    endo_te_sc = sc_e2.transform(endo_raw_full[n_train + n_val:])
    exo_te_sc  = sc_x2.transform(exo_full[n_train + n_val:])
    endo_te_raw = endo_raw_full[n_train + n_val:]

    plot_horizon_windows(
        model, endo_te_sc, exo_te_sc, endo_te_raw,
        sc_endo, args.context_len, args.horizon,
        out_dir, device,
        n_windows=6, context_show=60,
    )
    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Train Vanilla TimeXer baseline on WWTP dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_path',   default='../dataset/agtrup.csv')
    p.add_argument('--context_len', type=int,   default=360)
    p.add_argument('--horizon',     type=int,   default=18)
    p.add_argument('--stride',      type=int,   default=1)
    p.add_argument('--patch_size',  type=int,   default=12)
    p.add_argument('--d_model',     type=int,   default=128)
    p.add_argument('--n_heads',     type=int,   default=8)
    p.add_argument('--n_layers',    type=int,   default=3)
    p.add_argument('--d_ff',        type=int,   default=512)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--patience',    type=int,   default=10)
    p.add_argument('--out_dir',     default='results/vanilla_timexer')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
