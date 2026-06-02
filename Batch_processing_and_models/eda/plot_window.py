"""
plot_window.py
──────────────
Plot a specific dataset window showing context, forecast horizon, and the
actual continuation of the time series beyond the horizon.

Example:
  python plot_window.py \
      --experiment KAU079 \
      --window 739 \
      --extra 120 \
      --data_paths f1.csv f2.csv \
      --cheatsheet stamps.txt \
      --endogenous nh4 --log_cols nh4 \
      --context_len 180 --horizon 18 \
      --out results/window_739.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from anchor_issue_sequential_train_reactor_probabilistic import (
    resolve_columns,
    load_experiment,
    compute_time_since_valid,
    fit_scalers,
    apply_scalers,
    inverse_target,
    ReactorDataset,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment',          nargs='+', required=True)
    p.add_argument('--window',              type=int,  required=True,
                   help='Window index to plot (0-based within training split).')
    p.add_argument('--extra',               type=int,  default=100,
                   help='Extra steps to show beyond the forecast horizon.')
    p.add_argument('--data_paths',          nargs='+', required=True)
    p.add_argument('--cheatsheet',          required=True)
    p.add_argument('--endogenous',          nargs='+', required=True)
    p.add_argument('--exogenous',           nargs='*', default=[])
    p.add_argument('--log_cols',            nargs='*', default=None)
    p.add_argument('--target',              default=None)
    p.add_argument('--context_len',         type=int,  default=180)
    p.add_argument('--horizon',             type=int,  default=18)
    p.add_argument('--stride',              type=int,  default=1)
    p.add_argument('--val_frac',            type=float, default=0.1)
    p.add_argument('--test_frac',           type=float, default=0.1)
    p.add_argument('--scaling',             default='standard',
                   choices=['standard', 'robust'])
    p.add_argument('--scale_mode',          default='global',
                   choices=['global', 'per_var'])
    p.add_argument('--segment_isolated',    action='store_true')
    p.add_argument('--segment_gap_minutes', type=float, default=60.0)
    p.add_argument('--gap_threshold',       type=float, default=4.0)
    p.add_argument('--min_segment_rows',    type=int,   default=0)
    p.add_argument('--out',                 default='results/window.png')
    args = p.parse_args()

    endo_cols, exo_cols = resolve_columns(args.endogenous, args.exogenous)
    log_cols = set(args.log_cols) if args.log_cols else set(args.endogenous)
    target   = args.target or args.endogenous[0]
    tgt_idx  = endo_cols.index(target)

    # ── Load data ─────────────────────────────────────────────────────────
    df = load_experiment(args.data_paths, args.cheatsheet, args.experiment)
    df['time_since_valid'] = compute_time_since_valid(df)
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].values)

    T       = len(df)
    n_test  = int(T * args.test_frac)
    n_val   = int(T * args.val_frac)
    n_train = T - n_val - n_test

    endo_arr = df[endo_cols].values.astype(np.float64)
    exo_arr  = df[exo_cols ].values.astype(np.float64)
    flags    = df['flag'].values.astype(int)
    tsv      = df['time_since_valid'].values.astype(np.float32)
    times    = df['time'].values if 'time' in df.columns else None

    train_mask = flags[:n_train] == 0
    sc_endo = fit_scalers(endo_arr[:n_train][train_mask],
                          endo_cols, args.scaling, args.scale_mode)
    sc_exo  = fit_scalers(exo_arr [:n_train][train_mask],
                          exo_cols,  args.scaling, args.scale_mode)

    endo_sc = apply_scalers(endo_arr[:n_train], sc_endo, endo_cols, args.scale_mode)
    exo_sc  = apply_scalers(exo_arr [:n_train], sc_exo,  exo_cols,  args.scale_mode)

    ds = ReactorDataset(
        endo_sc, exo_sc, flags[:n_train],
        context_len=args.context_len, horizon=args.horizon,
        stride=args.stride,
        segment_isolated=args.segment_isolated,
        gap_threshold=args.gap_threshold,
        time_since_valid=tsv[:n_train],
        times=times[:n_train] if times is not None else None,
        segment_gap_minutes=args.segment_gap_minutes,
        min_segment_rows=args.min_segment_rows,
        target_indices=[tgt_idx],
    )

    idx = args.window
    if idx >= len(ds):
        print(f"Window {idx} out of range — dataset has {len(ds)} windows.")
        return

    # ── Extract data ──────────────────────────────────────────────────────
    start    = ds.indices[idx]
    ctx      = args.context_len
    hor      = args.horizon
    extra    = args.extra

    # Full slice in scaled space: context + horizon + extra
    end_full = min(start + ctx + hor + extra, len(endo_sc))
    raw_sc   = endo_sc[start:end_full, tgt_idx]   # scaled

    # Inverse-transform back to original scale
    raw = inverse_target(raw_sc, sc_endo, target, endo_cols,
                         args.scale_mode, log_cols)

    ctx_vals  = raw[:ctx]
    hor_vals  = raw[ctx : ctx + hor]
    cont_vals = raw[ctx + hor :]

    tx   = np.arange(-ctx, 0)
    th   = np.arange(0, len(hor_vals))
    tc   = np.arange(len(hor_vals), len(hor_vals) + len(cont_vals))

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(tx, ctx_vals,  color='steelblue', lw=1.4, label='context')
    ax.plot(th, hor_vals,  color='tomato',    lw=1.6, label='horizon (forecast target)')
    if len(cont_vals):
        ax.plot(tc, cont_vals, color='seagreen', lw=1.4,
                linestyle='--', label=f'continuation (+{len(cont_vals)} steps)')
    ax.axvline(0,   color='k', lw=0.9, linestyle='--', alpha=0.6)
    ax.axvline(hor, color='k', lw=0.9, linestyle=':',  alpha=0.4)
    ax.set_xlabel('steps relative to forecast start')
    ax.set_ylabel(target)
    ax.set_title(f'Window {idx}  |  experiment: {"+".join(args.experiment)}  '
                 f'|  start row: {start}')
    ax.legend()
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
