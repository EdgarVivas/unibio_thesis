"""
plot_horizon.py – Visualise the full 18-step forecast horizon for individual windows.

For N randomly chosen test windows this script shows:
  • the last `context_show` steps of the input context (grey)
  • the actual future values for all 18 horizon steps (blue)
  • the model's predicted values for all 18 horizon steps (orange)

Usage
-----
    python plot_horizon.py                          # uses default paths
    python plot_horizon.py --results_dir results/run1  --n_windows 8
    python plot_horizon.py --seed 99                # different random windows
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from model import ModifiedTimeXer
from data  import ENDO_COLS, REAL_EXO_COLS


# ─────────────────────────────────────────────────────────────────────────────

def load_model(results_dir: Path, device: torch.device) -> tuple[ModifiedTimeXer, dict]:
    with open(results_dir / 'results.json') as f:
        results = json.load(f)
    args = results['args']

    model = ModifiedTimeXer(
        c_endo      = len(ENDO_COLS),
        c_exo       = len(REAL_EXO_COLS),
        context_len = args['context_len'],
        horizon     = args['horizon'],
        patch_size  = args['patch_size'],
        d_model     = args['d_model'],
        n_heads     = args['n_heads'],
        n_layers    = args['n_layers'],
        d_ff        = args['d_ff'],
        tcn_levels  = args['tcn_levels'],
        tcn_kernel  = args['tcn_kernel'],
        dropout     = 0.0,            # no dropout at inference
    ).to(device)

    state = torch.load(results_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, args


def load_test_data(data_path: str,
                   sc_endo, sc_exo,
                   val_frac: float = 0.1,
                   test_frac: float = 0.1):
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date').sort_index()
    df = df[ENDO_COLS + REAL_EXO_COLS].dropna()

    T       = len(df)
    n_test  = int(T * test_frac)
    n_val   = int(T * val_frac)
    n_train = T - n_val - n_test

    endo_raw = df[ENDO_COLS].values
    exo_raw  = df[REAL_EXO_COLS].values

    endo_te_raw = endo_raw[n_train + n_val:]
    exo_te      = sc_exo.transform(exo_raw[n_train + n_val:])
    endo_te     = sc_endo.transform(endo_te_raw)

    return endo_te, exo_te, endo_te_raw   # scaled + raw for plotting


def predict_window(model, endo_scaled, exo_scaled,
                   start: int, context_len: int, device: torch.device):
    """Run model on a single window starting at `start`."""
    x_endo = torch.tensor(
        endo_scaled[start : start + context_len].T, dtype=torch.float32
    ).unsqueeze(0).to(device)                          # [1, C_endo, ctx]
    x_exo = torch.tensor(
        exo_scaled[start : start + context_len].T, dtype=torch.float32
    ).unsqueeze(0).to(device)                          # [1, C_exo,  ctx]

    with torch.no_grad():
        pred = model(x_endo, x_exo)                   # [1, C_endo, H]
    return pred.squeeze(0).cpu().numpy()               # [C_endo, H]


# ─────────────────────────────────────────────────────────────────────────────

def plot_windows(windows_data: list, horizon: int,
                 context_show: int, out_path: Path) -> None:
    """
    windows_data : list of dicts with keys
        context_raw  [context_show, C_endo]  – raw (unscaled) context tail
        actual_raw   [H, C_endo]             – raw actual future
        pred_orig    [C_endo, H]             – inverse-transformed predictions
        window_idx   int
    """
    n_windows = len(windows_data)
    n_vars    = len(ENDO_COLS)

    fig, axes = plt.subplots(
        n_windows, n_vars,
        figsize=(7 * n_vars, 3.5 * n_windows),
        squeeze=False,
    )

    for row, wd in enumerate(windows_data):
        ctx   = wd['context_raw']    # [context_show, C]
        act   = wd['actual_raw']     # [H, C]
        pred  = wd['pred_orig']      # [C, H]

        ctx_x  = np.arange(-context_show, 0)       # negative = past
        fore_x = np.arange(0, horizon)              # positive = future

        for col, name in enumerate(ENDO_COLS):
            ax = axes[row][col]

            # Context
            ax.plot(ctx_x, ctx[:, col],
                    color='#888888', linewidth=1.2, alpha=0.7, label='Context')

            # Actual future
            ax.plot(fore_x, act[:, col],
                    color='#1f77b4', linewidth=2.0, label='Actual')

            # Predicted future
            ax.plot(fore_x, pred[col],
                    color='#ff7f0e', linewidth=2.0, linestyle='--', label='Predicted')

            # Dividing line at t=0
            ax.axvline(0, color='black', linewidth=0.8, linestyle=':')

            mae  = np.mean(np.abs(pred[col] - act[:, col]))
            rmse = np.sqrt(np.mean((pred[col] - act[:, col])**2))

            ax.set_title(
                f'Window {wd["window_idx"]}  |  {name}\n'
                f'MAE={mae:.3f}  RMSE={rmse:.3f}',
                fontsize=9,
            )
            ax.set_xlabel('Steps from forecast origin  (1 step ≈ 2 min)', fontsize=8)
            ax.set_ylabel(name, fontsize=8)

            # x-axis ticks: label in minutes
            tick_pos = np.concatenate([
                np.arange(-context_show, 0, context_show // 4),
                np.arange(0, horizon + 1, max(1, horizon // 6)),
            ])
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([f'{int(t*2)}m' for t in tick_pos], fontsize=7)

            if row == 0 and col == n_vars - 1:
                ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(
        f'Full {horizon}-step forecast horizon  ({horizon * 2} min ahead)\n'
        f'Context shown: last {context_show} steps ({context_show * 2} min)',
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model + scalers
    print(f"Loading model from {results_dir} ...")
    model, model_args = load_model(results_dir, device)
    sc_endo = joblib.load(results_dir / 'scaler_endo.joblib')
    sc_exo  = joblib.load(results_dir / 'scaler_exo.joblib')

    context_len = model_args['context_len']
    horizon     = model_args['horizon']

    # ── Load test split
    print("Loading test data ...")
    endo_sc, exo_sc, endo_raw = load_test_data(
        args.data_path, sc_endo, sc_exo)

    # ── Pick random windows
    rng      = np.random.default_rng(args.seed)
    max_start = len(endo_sc) - context_len - horizon
    starts   = sorted(rng.choice(max_start, size=args.n_windows, replace=False))

    context_show = min(args.context_show, context_len)

    # ── Build predictions
    windows_data = []
    for s in starts:
        pred_sc = predict_window(model, endo_sc, exo_sc, s, context_len, device)
        # [C_endo, H] → inverse transform each step
        pred_orig = sc_endo.inverse_transform(pred_sc.T).T      # [C_endo, H]

        ctx_raw = endo_raw[s + context_len - context_show
                           : s + context_len]                   # [context_show, C]
        act_raw = endo_raw[s + context_len
                           : s + context_len + horizon]         # [H, C]

        windows_data.append({
            'window_idx':  int(s),
            'context_raw': ctx_raw,
            'actual_raw':  act_raw,
            'pred_orig':   pred_orig,
        })

    # ── Plot
    out_path = results_dir / f'horizon_structure_seed{args.seed}.png'
    plot_windows(windows_data, horizon, context_show, out_path)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Plot full forecast horizon for individual test windows.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--results_dir',   default='results/timexer_tcn',
                   help='Folder containing best_model.pt, results.json, scalers.')
    p.add_argument('--data_path',     default='../dataset/agtrup.csv')
    p.add_argument('--n_windows',     type=int, default=6,
                   help='How many random test windows to plot.')
    p.add_argument('--context_show',  type=int, default=60,
                   help='How many context steps to show left of the forecast origin.')
    p.add_argument('--seed',          type=int, default=42)
    main(p.parse_args())
