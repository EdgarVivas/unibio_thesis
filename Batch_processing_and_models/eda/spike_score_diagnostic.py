"""
spike_score_diagnostic.py
─────────────────────────
Visualises the spike-score distribution for every training window so you can
tune spike_abs_weight and spike_oversample without blind guessing.

Run with the same data/column/experiment flags as the training script:

  python spike_score_diagnostic.py \\
      --experiment KAU073 KAU075 \\
      --data_paths f1.csv f2.csv \\
      --cheatsheet stamps.txt \\
      --endogenous nh4 --log_cols nh4 \\
      --context_len 120 --horizon 30 \\
      --spike_abs_weight 3.0 --spike_oversample 1.0 \\
      --out_dir results/spike_diag

Outputs (all in --out_dir):
  rel_score_histogram.png     – rel_score distribution per experiment
  delta_histogram.png         – raw delta distribution per experiment
  combined_score_histogram.png– combined score with current abs_weight
  score_vs_weight.png         – combined score tail for different abs_weight values
  oversample_concentration.png– how oversample power concentrates weight on spikes
  top_windows_{exp}.png       – top 12 highest-scoring windows (visual sanity check)
  stats.txt                   – numeric summary + recommended abs_weight & oversample
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
    ALL_DATA_COLS,
    resolve_columns,
    load_experiment,
    compute_time_since_valid,
    fit_scalers,
    apply_scalers,
    ReactorDataset,
)


# ───────────────────────────────────────────────────────────────────────────

def score_dataset(ds: ReactorDataset, tgt_idx: list[int],
                  spike_abs_weight: float) -> np.ndarray:
    """Return (N, 4) array: [rel_score, delta, ctx_std, abs_score] per window."""
    rows = []
    for i in range(len(ds)):
        x_endo, _, y = ds[i]
        ctx     = x_endo[tgt_idx]
        ctx_std = float(ctx.std().clamp(min=1e-3))
        delta   = float((y - ctx[:, -1:]).abs().max())
        rel     = delta / ctx_std
        abs_s   = delta * spike_abs_weight
        rows.append([rel, delta, ctx_std, abs_s])
    return np.array(rows, dtype=np.float32)


def combined_scores(arr: np.ndarray, abs_weight: float) -> np.ndarray:
    return arr[:, 1] * abs_weight + arr[:, 0]   # abs_score recomputed + rel_score


def top_frac_weight(scores: np.ndarray, power: float, top_pct: float) -> float:
    """Fraction of total sampling weight held by top top_pct% of windows."""
    w = np.clip(scores, 1e-6, None) ** power
    w = w / w.sum()
    k = max(1, int(top_pct * len(w)))
    return float(np.sort(w)[::-1][:k].sum())


def recommend_oversample(scores: np.ndarray,
                         target_top_pct: float = 0.05,
                         target_weight_frac: float = 0.50) -> float:
    """
    Find the smallest power such that the top target_top_pct% of windows
    hold at least target_weight_frac of total sampling weight.
    Returns the power, or np.nan if unreachable.
    """
    for power in np.arange(0.5, 8.1, 0.1):
        if top_frac_weight(scores, power, target_top_pct) >= target_weight_frac:
            return round(float(power), 1)
    return float('nan')


def recommend_abs_weight(arr: np.ndarray) -> float:
    """
    Suggest abs_weight such that the combined score at p95(delta) is
    2× the combined score at mean(rel_score) when abs_weight=0.
    i.e. the abs component alone at p95 = 2 * mean(rel).
    """
    p95d = float(np.percentile(arr[:, 1], 95))
    mean_rel = float(arr[:, 0].mean())
    if p95d < 1e-9:
        return 1.0
    return round(2.0 * mean_rel / p95d, 3)


# ── Plot helpers ─────────────────────────────────────────────────────────────

def plot_histogram(scores_dict: dict, values_fn, title: str, xlabel: str,
                   save_path: Path, bins: int = 80) -> None:
    fig, axes = plt.subplots(len(scores_dict), 1,
                             figsize=(10, 3.5 * len(scores_dict)), squeeze=False)
    for ax, (exp_name, arr) in zip(axes[:, 0], scores_dict.items()):
        vals = values_fn(arr)
        p50, p90, p95, p99 = np.percentile(vals, [50, 90, 95, 99])
        ax.hist(vals, bins=bins, color='steelblue', edgecolor='none', alpha=0.8)
        for pv, lbl, c in [(p50, 'p50', 'green'), (p90, 'p90', 'orange'),
                           (p95, 'p95', 'red'),   (p99, 'p99', 'darkred')]:
            ax.axvline(pv, color=c, lw=1.5, linestyle='--', label=f'{lbl}={pv:.3f}')
        ax.set_title(f'{exp_name}  ({len(vals):,} windows)')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('count')
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


def plot_score_vs_weight(scores_dict: dict, weights_to_try: list[float],
                         oversample: float, save_path: Path) -> None:
    """Cumulative sampling weight curves for different abs_weight values."""
    n_exp = len(scores_dict)
    fig, axes = plt.subplots(n_exp, 1, figsize=(10, 3.5 * n_exp), squeeze=False)
    for ax, (exp_name, arr) in zip(axes[:, 0], scores_dict.items()):
        for w in weights_to_try:
            sc = combined_scores(arr, w)
            pw = np.clip(sc, 1e-6, None) ** oversample
            pw = pw / pw.sum()
            sorted_w = np.sort(pw)[::-1]
            cum = np.cumsum(sorted_w)
            ax.plot(np.arange(1, len(cum) + 1), cum, label=f'abs_w={w:.2f}')
        ax.axhline(0.5, color='grey', lw=0.8, linestyle=':',
                   label='50% of weight')
        ax.set_xlabel('top-k windows (log scale)')
        ax.set_ylabel('cumulative sampling weight')
        ax.set_title(f'{exp_name}  —  oversample power={oversample:.1f}\n'
                     'Steeper = weight concentrated on fewer spike windows')
        ax.legend(fontsize=8)
        ax.set_xscale('log')
    fig.suptitle('Effect of abs_weight on sampling concentration\n'
                 '(want: 50% line crossed at small k)', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


def plot_oversample_concentration(scores_dict: dict, abs_weight: float,
                                  powers_to_try: list[float],
                                  save_path: Path) -> None:
    """
    For the current abs_weight, show how different oversample power values
    concentrate weight onto the top windows. This is the plot for tuning
    spike_oversample once abs_weight is fixed.
    """
    n_exp = len(scores_dict)
    fig, axes = plt.subplots(n_exp, 1, figsize=(10, 3.5 * n_exp), squeeze=False)
    for ax, (exp_name, arr) in zip(axes[:, 0], scores_dict.items()):
        sc = combined_scores(arr, abs_weight)
        for power in powers_to_try:
            pw = np.clip(sc, 1e-6, None) ** power
            pw = pw / pw.sum()
            sorted_w = np.sort(pw)[::-1]
            cum = np.cumsum(sorted_w)
            frac5 = top_frac_weight(sc, power, 0.05)
            ax.plot(np.arange(1, len(cum) + 1), cum,
                    label=f'power={power:.1f}  (top 5%={frac5*100:.0f}% of weight)')
        ax.axhline(0.50, color='grey', lw=0.8, linestyle=':', label='50% weight')
        ax.axvline(max(1, int(0.05 * len(arr))), color='grey', lw=0.8,
                   linestyle='--', label='5% of windows')
        ax.set_xlabel('top-k windows (log scale)')
        ax.set_ylabel('cumulative sampling weight')
        ax.set_title(f'{exp_name}  —  abs_weight={abs_weight:.2f}')
        ax.legend(fontsize=7)
        ax.set_xscale('log')
    fig.suptitle('Effect of oversample power on weight concentration\n'
                 '(want: top-5% windows ≥ 50% of weight)', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


def plot_top_windows(ds: ReactorDataset, arr: np.ndarray,
                     tgt_idx: list[int], target_col: str,
                     context_len: int, horizon: int,
                     abs_weight: float, save_path: Path, n: int = 12) -> None:
    sc   = combined_scores(arr, abs_weight)
    top  = np.argsort(sc)[::-1][:n]
    ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for k, idx in enumerate(top):
        ax = axes[k // ncols][k % ncols]
        x_endo, _, y = ds[idx]
        ctx_vals = x_endo[tgt_idx[0]].numpy()
        hor_vals = y[0].numpy()
        tx = np.arange(-context_len, 0)
        th = np.arange(0, horizon)
        ax.plot(tx, ctx_vals, color='steelblue', lw=1.2, label='context')
        ax.plot(th, hor_vals, color='tomato',    lw=1.5, label='future')
        ax.axvline(0, color='k', lw=0.8, linestyle='--')
        ax.set_title(f'rank {k+1}  combined={sc[idx]:.2f}  '
                     f'rel={arr[idx,0]:.2f}  δ={arr[idx,1]:.3f}',
                     fontsize=8)
        ax.legend(fontsize=7)
    for k in range(len(top), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle(f'Top-{n} windows by combined score  (target: {target_col})\n'
                 'Check: do these look like real spikes/rises?', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


# ── Text report ───────────────────────────────────────────────────────────────

def write_stats(scores_dict: dict, out_path: Path,
                weights_to_try: list[float], current_abs_weight: float,
                current_oversample: float) -> None:
    powers_to_try = [0.5, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    lines = ['Spike Score Diagnostic\n' + '=' * 70 + '\n']

    for exp_name, arr in scores_dict.items():
        rel   = arr[:, 0]
        delta = arr[:, 1]
        std   = arr[:, 2]

        lines.append(f'\n[{exp_name}]  {len(arr):,} windows\n')
        lines.append(f'  rel_score : mean={rel.mean():.3f}  '
                     f'p50={np.percentile(rel,50):.3f}  '
                     f'p90={np.percentile(rel,90):.3f}  '
                     f'p95={np.percentile(rel,95):.3f}  '
                     f'max={rel.max():.3f}\n')
        lines.append(f'  delta     : mean={delta.mean():.4f}  '
                     f'p50={np.percentile(delta,50):.4f}  '
                     f'p95={np.percentile(delta,95):.4f}  '
                     f'max={delta.max():.4f}\n')
        lines.append(f'  ctx_std   : mean={std.mean():.4f}  '
                     f'p5={np.percentile(std,5):.4f}  '
                     f'p95={np.percentile(std,95):.4f}\n')

        # abs_weight recommendation
        rec_abs = recommend_abs_weight(arr)
        lines.append(f'\n  ── abs_weight recommendation ──\n')
        lines.append(f'  Current:     {current_abs_weight:.2f}\n')
        lines.append(f'  Recommended: {rec_abs:.2f}  '
                     f'(so p95(delta)*w = 2 × mean(rel_score))\n')
        lines.append(f'  Sanity: p95(delta) × {rec_abs:.2f} = '
                     f'{np.percentile(delta,95)*rec_abs:.3f}  '
                     f'vs mean(rel) = {rel.mean():.3f}\n')

        # abs_weight comparison table
        lines.append(f'\n  Weight concentration (top 5% windows) vs abs_weight '
                     f'[power={current_oversample:.1f}]:\n')
        lines.append(f'  {"abs_w":>8}  {"max/min":>10}  '
                     f'{"top 1%":>8}  {"top 5%":>8}  {"top 10%":>9}\n')
        for w in weights_to_try:
            sc = combined_scores(arr, w)
            ratio = np.clip(sc, 1e-6, None).max() / np.clip(sc, 1e-6, None).min()
            t1  = top_frac_weight(sc, current_oversample, 0.01)
            t5  = top_frac_weight(sc, current_oversample, 0.05)
            t10 = top_frac_weight(sc, current_oversample, 0.10)
            lines.append(f'  {w:>8.2f}  {ratio:>9.1f}x  '
                         f'{t1*100:>7.1f}%  {t5*100:>7.1f}%  {t10*100:>8.1f}%\n')

        # oversample recommendation
        sc_current = combined_scores(arr, current_abs_weight)
        lines.append(f'\n  ── spike_oversample recommendation ──\n')
        lines.append(f'  (using current abs_weight={current_abs_weight:.2f})\n')
        lines.append(f'  Goal: top 5% of windows hold ≥ 50% of sampling weight.\n')
        rec_pow = recommend_oversample(sc_current, target_top_pct=0.05,
                                       target_weight_frac=0.50)
        if np.isnan(rec_pow):
            lines.append(f'  Recommended: unreachable — abs_weight is too low.\n'
                         f'  Try increasing abs_weight first, then re-run.\n')
        else:
            lines.append(f'  Current:     {current_oversample:.1f}\n')
            lines.append(f'  Recommended: {rec_pow:.1f}\n')
            if rec_pow > 3.0:
                lines.append(f'  WARNING: power > 3.0 risks loss explosion (sampler '
                              f'over-concentrates on a handful of windows).\n'
                              f'  Consider raising abs_weight to {rec_abs:.2f} first, '
                              f'then re-running to get a lower power recommendation.\n')
            elif rec_pow <= 1.5:
                lines.append(f'  Safe range (1.0–1.5). Training should be stable.\n')
            else:
                lines.append(f'  Moderate (1.5–3.0). Monitor val loss for instability.\n')

        # oversample power table
        lines.append(f'\n  Weight concentration vs oversample power '
                     f'[abs_weight={current_abs_weight:.2f}]:\n')
        lines.append(f'  {"power":>7}  {"max/min":>10}  '
                     f'{"top 1%":>8}  {"top 5%":>8}  {"top 10%":>9}\n')
        for power in powers_to_try:
            sc = combined_scores(arr, current_abs_weight)
            ratio = np.clip(sc, 1e-6, None).max() / np.clip(sc, 1e-6, None).min()
            ratio_pw = ratio ** power
            t1  = top_frac_weight(sc, power, 0.01)
            t5  = top_frac_weight(sc, power, 0.05)
            t10 = top_frac_weight(sc, power, 0.10)
            lines.append(f'  {power:>7.1f}  {ratio_pw:>9.0f}x  '
                         f'{t1*100:>7.1f}%  {t5*100:>7.1f}%  {t10*100:>8.1f}%\n')

        lines.append('\n')

    out_path.write_text(''.join(lines))
    print(f'  Saved {out_path}')


# ───────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Spike score diagnostic')
    p.add_argument('--experiment',       nargs='+', required=True)
    p.add_argument('--data_paths',       nargs='+', required=True)
    p.add_argument('--cheatsheet',       required=True)
    p.add_argument('--endogenous',       nargs='+', required=True)
    p.add_argument('--exogenous',        nargs='*', default=[])
    p.add_argument('--log_cols',         nargs='*', default=None)
    p.add_argument('--target',           default=None)
    p.add_argument('--context_len',      type=int,   default=120)
    p.add_argument('--horizon',          type=int,   default=30)
    p.add_argument('--stride',           type=int,   default=1)
    p.add_argument('--val_frac',         type=float, default=0.1)
    p.add_argument('--test_frac',        type=float, default=0.1)
    p.add_argument('--scaling',          default='standard',
                   choices=['standard', 'robust'])
    p.add_argument('--scale_mode',       default='global',
                   choices=['global', 'per_var'])
    p.add_argument('--segment_isolated', action='store_true')
    p.add_argument('--segment_gap_minutes', type=float, default=60.0)
    p.add_argument('--gap_threshold',    type=float, default=4.0)
    p.add_argument('--min_segment_rows', type=int,   default=0)
    p.add_argument('--spike_abs_weight', type=float, default=1.0)
    p.add_argument('--spike_oversample', type=float, default=1.0)
    p.add_argument('--out_dir',          default='results/spike_diag')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    endo_cols, exo_cols = resolve_columns(args.endogenous, args.exogenous)
    log_cols = set(args.log_cols) if args.log_cols else set(args.endogenous)
    target   = args.target or args.endogenous[0]
    tgt_idx  = [endo_cols.index(target)]

    weights_to_try = sorted({0.0, 1.0, args.spike_abs_weight,
                              args.spike_abs_weight * 2,
                              args.spike_abs_weight * 5})
    powers_to_try  = [0.5, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    scores_dict: dict[str, np.ndarray] = {}

    all_key = '+'.join(args.experiment)
    print(f'\n── {all_key} (concatenated) ──────────────────────────────────')
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
    times    = df['time'].values

    train_mask = flags[:n_train] == 0
    sc_endo = fit_scalers(endo_arr[:n_train][train_mask],
                          endo_cols, args.scaling, args.scale_mode)
    sc_exo  = fit_scalers(exo_arr [:n_train][train_mask],
                          exo_cols,  args.scaling, args.scale_mode)

    tr_ds = ReactorDataset(
        apply_scalers(endo_arr[:n_train], sc_endo, endo_cols, args.scale_mode),
        apply_scalers(exo_arr [:n_train], sc_exo,  exo_cols,  args.scale_mode),
        flags[:n_train],
        context_len=args.context_len, horizon=args.horizon,
        stride=args.stride,
        segment_isolated=args.segment_isolated,
        gap_threshold=args.gap_threshold,
        time_since_valid=tsv[:n_train],
        times=times[:n_train],
        segment_gap_minutes=args.segment_gap_minutes,
        min_segment_rows=args.min_segment_rows,
        target_indices=tgt_idx,
    )

    print(f'  Scoring {len(tr_ds):,} windows ...')
    arr = score_dataset(tr_ds, tgt_idx, args.spike_abs_weight)
    scores_dict[all_key] = arr

    sc_combined = combined_scores(arr, args.spike_abs_weight)
    rec_pow = recommend_oversample(sc_combined)
    rec_abs = recommend_abs_weight(arr)
    print(f'  rel_score  p95={np.percentile(arr[:,0],95):.3f}  '
          f'max={arr[:,0].max():.3f}')
    print(f'  delta      p95={np.percentile(arr[:,1],95):.4f}  '
          f'max={arr[:,1].max():.4f}')
    print(f'  → recommended abs_weight : {rec_abs:.2f}  '
          f'(current: {args.spike_abs_weight:.2f})')
    print(f'  → recommended oversample : {rec_pow:.1f}  '
          f'(current: {args.spike_oversample:.1f})')

    plot_top_windows(tr_ds, arr, tgt_idx, target,
                     args.context_len, args.horizon,
                     args.spike_abs_weight,
                     out_dir / 'top_windows_all.png')

    # ── combined plots across all experiments ──────────────────────────────
    plot_histogram(scores_dict,
                   values_fn=lambda a: a[:, 0],
                   title='rel_score distribution  (delta / ctx_std)',
                   xlabel='rel_score',
                   save_path=out_dir / 'rel_score_histogram.png')

    plot_histogram(scores_dict,
                   values_fn=lambda a: a[:, 1],
                   title='delta distribution  (max |future − last_ctx|, scaled space)',
                   xlabel='delta',
                   save_path=out_dir / 'delta_histogram.png')

    plot_histogram(scores_dict,
                   values_fn=lambda a: combined_scores(a, args.spike_abs_weight),
                   title=f'combined score distribution  '
                         f'(rel + δ×{args.spike_abs_weight:.2f})',
                   xlabel='combined score',
                   save_path=out_dir / 'combined_score_histogram.png')

    plot_score_vs_weight(scores_dict, weights_to_try,
                         oversample=args.spike_oversample,
                         save_path=out_dir / 'score_vs_weight.png')

    plot_oversample_concentration(scores_dict,
                                  abs_weight=args.spike_abs_weight,
                                  powers_to_try=powers_to_try,
                                  save_path=out_dir / 'oversample_concentration.png')

    write_stats(scores_dict, out_dir / 'stats.txt',
                weights_to_try, args.spike_abs_weight, args.spike_oversample)

    print(f'\nDone. Results in {out_dir}/')


if __name__ == '__main__':
    main()
