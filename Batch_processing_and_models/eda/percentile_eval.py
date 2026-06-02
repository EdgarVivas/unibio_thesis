"""
percentile_eval.py – Evaluate a model's forecast error broken down by
percentile of the target variable, rather than spike vs flat.

For every test window the script computes MAE and RMSE, then bins the
windows into percentile groups based on a chosen signal statistic
(mean/max of the actual horizon, or the last context value).

Outputs
-------
  <out_dir>/<exp>_<target>_percentile_metrics.json   – per-bin metrics
  <out_dir>/<exp>_<target>_percentile_plot.png        – bar chart

Usage
-----
    python percentile_eval.py \\
        --run_dir results/KAU084 \\
        --experiment KAU084 KAU081 \\
        --data_paths dataset/final_dataset.csv dataset/final_dataset_2.csv \\
        --cheatsheet filtered_data_stamps.txt \\
        --n_percentiles 10 \\
        --percentile_on horizon_mean \\
        --out_dir results/percentile_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_selected_reactor_windows import (
    _as_list,
    _fit_training_scalers,
    _forecast_color_for_family,
    _load_json,
    _load_model,
    _merge_run_config,
    _normalise_targets,
    _prepare_experiment_frame,
    _resolve_path,
)
from timexer_train_reactor_probabilistic import (
    LOG_SIG_MAX,
    LOG_SIG_MIN,
    LOG_VAR_MAX,
    LOG_VAR_MIN,
    apply_scalers,
    inverse_target,
)


# ── Single window inference ────────────────────────────────────────────────

def _infer(model, x_endo_t, x_exo_t, target_channel_idx, head):
    if getattr(model, "_forecast_family", None) == "csdi":
        with torch.no_grad():
            samples = model.sample(x_endo_t, x_exo_t, n_samples=50)
        mean_sc = samples.mean(dim=1)[0, target_channel_idx].cpu().numpy()
        return mean_sc

    with torch.no_grad():
        out = model(x_endo_t, x_exo_t)
    return out[0][0, target_channel_idx].cpu().numpy()


# ── Per-experiment evaluation ──────────────────────────────────────────────

def evaluate_experiment(
    model,
    sc_endo,
    sc_exo,
    df,
    endo_cols: Sequence[str],
    exo_cols: Sequence[str],
    scale_mode: str,
    context_len: int,
    horizon: int,
    target_col: str,
    log_cols: Sequence[str],
    head: str,
    target_channel_idx: int,
    val_frac: float,
    test_frac: float,
    n_percentiles: int,
    percentile_on: str,
    out_path_stem: Path,
    forecast_color: str,
) -> dict:
    device = next(model.parameters()).device
    T       = len(df)
    n_test  = int(T * test_frac)
    n_val   = int(T * val_frac)
    n_train = T - n_val - n_test

    flags_te   = df["flag"].values.astype(int)[n_train + n_val:]
    raw_te     = df[target_col].values[n_train + n_val:]
    endo_te_sc = apply_scalers(df[endo_cols].values[n_train + n_val:], sc_endo, endo_cols, scale_mode)
    exo_te_sc  = apply_scalers(df[exo_cols ].values[n_train + n_val:], sc_exo,  exo_cols,  scale_mode)

    log_cols_s = set(log_cols)
    inv = lambda a: inverse_target(a, sc_endo, target_col, endo_cols, scale_mode, log_cols_s)

    valid = np.where(flags_te == 0)[0]
    candidates = [i for i in valid if i + context_len + horizon < len(endo_te_sc)]
    if not candidates:
        print(f"  No valid test windows — skipping.")
        return {}

    print(f"  Running inference on {len(candidates):,} test windows...")

    maes, rmses, bin_vals = [], [], []

    for s in candidates:
        x_e = torch.tensor(endo_te_sc[s : s + context_len].T,
                            dtype=torch.float32, device=device).unsqueeze(0)
        x_x = torch.tensor(exo_te_sc [s : s + context_len].T,
                            dtype=torch.float32, device=device).unsqueeze(0)

        mean_sc  = _infer(model, x_e, x_x, target_channel_idx, head)
        mean_out = inv(mean_sc)

        actual_sc  = raw_te[s + context_len : s + context_len + horizon]
        actual_out = np.expm1(actual_sc) if target_col in log_cols_s else actual_sc

        maes.append(float(np.mean(np.abs(mean_out - actual_out))))
        rmses.append(float(np.sqrt(np.mean((mean_out - actual_out) ** 2))))

        ctx_last = raw_te[s + context_len - 1]
        if target_col in log_cols_s:
            ctx_last = float(np.expm1(ctx_last))

        if percentile_on == "horizon_mean":
            bin_vals.append(float(np.mean(actual_out)))
        elif percentile_on == "horizon_max":
            bin_vals.append(float(np.max(actual_out)))
        elif percentile_on == "context_last":
            bin_vals.append(float(ctx_last))
        else:  # context_mean
            ctx_sl = raw_te[s : s + context_len]
            if target_col in log_cols_s:
                ctx_sl = np.expm1(ctx_sl)
            bin_vals.append(float(np.mean(ctx_sl)))

    maes      = np.array(maes)
    rmses     = np.array(rmses)
    bin_vals  = np.array(bin_vals)

    # ── Percentile bins ───────────────────────────────────────────────────
    pct_edges = np.percentile(bin_vals, np.linspace(0, 100, n_percentiles + 1))
    pct_edges[-1] += 1e-9   # include max

    bins_result = []
    for k in range(n_percentiles):
        lo, hi = pct_edges[k], pct_edges[k + 1]
        mask = (bin_vals >= lo) & (bin_vals < hi)
        n = int(mask.sum())
        label = f"p{int(100*k/n_percentiles)}-{int(100*(k+1)/n_percentiles)}"
        if n == 0:
            bins_result.append({
                "label": label, "n": 0,
                "value_range": [round(float(lo), 4), round(float(hi), 4)],
                "mae": None, "rmse": None,
            })
            continue
        bins_result.append({
            "label": label,
            "n": n,
            "value_range": [round(float(lo), 4), round(float(hi), 4)],
            "mae":  round(float(np.mean(maes[mask])),  6),
            "rmse": round(float(np.mean(rmses[mask])), 6),
        })

    # ── Print table ───────────────────────────────────────────────────────
    print(f"\n  Percentile breakdown  (binned on: {percentile_on})")
    print(f"  {'Bin':12s}  {'n':>6s}  {'value range':>20s}  {'MAE':>10s}  {'RMSE':>10s}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*20}  {'─'*10}  {'─'*10}")
    for b in bins_result:
        lo, hi = b["value_range"]
        mae_s  = f"{b['mae']:.4f}"  if b["mae"]  is not None else "—"
        rmse_s = f"{b['rmse']:.4f}" if b["rmse"] is not None else "—"
        print(f"  {b['label']:12s}  {b['n']:6d}  {lo:9.3f} – {hi:9.3f}  {mae_s:>10s}  {rmse_s:>10s}")

    overall = {
        "n":    len(maes),
        "mae":  round(float(np.mean(maes)),  6),
        "rmse": round(float(np.mean(rmses)), 6),
    }
    print(f"\n  Overall  n={overall['n']}  MAE={overall['mae']:.4f}  RMSE={overall['rmse']:.4f}")

    results = {
        "target_col":    target_col,
        "percentile_on": percentile_on,
        "n_percentiles": n_percentiles,
        "overall":       overall,
        "bins":          bins_result,
    }

    json_path = out_path_stem.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    labels     = [b["label"] for b in bins_result]
    mae_vals   = [b["mae"]  if b["mae"]  is not None else 0.0 for b in bins_result]
    rmse_vals  = [b["rmse"] if b["rmse"] is not None else 0.0 for b in bins_result]
    ns         = [b["n"] for b in bins_result]
    midpoints  = [(b["value_range"][0] + b["value_range"][1]) / 2 for b in bins_result]

    fig, axes = plt.subplots(2, 1, figsize=(max(8, n_percentiles * 0.9), 9))

    # Bar chart: MAE and RMSE per bin
    x = np.arange(len(labels))
    w = 0.38
    ax = axes[0]
    ax.bar(x - w/2, mae_vals,  width=w, color=forecast_color, alpha=0.8, label="MAE")
    ax.bar(x + w/2, rmse_vals, width=w, color=forecast_color, alpha=0.45, label="RMSE")
    ax.axhline(overall["mae"],  color="black", ls="--", lw=1.0, label=f"Overall MAE={overall['mae']:.3f}")
    ax.axhline(overall["rmse"], color="gray",  ls=":",  lw=1.0, label=f"Overall RMSE={overall['rmse']:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Error (mg/L)", fontsize=10)
    ax.set_title(f"{target_col.upper()} forecast error by {percentile_on} percentile", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")

    # Window count per bin
    ax2 = axes[1]
    ax2.bar(x, ns, color="#888888", alpha=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Number of windows", fontsize=10)
    ax2.set_title(f"Window count per bin  (binned on: {percentile_on})", fontsize=10)
    ax2.grid(True, alpha=0.2, axis="y")

    # Annotate with bin midpoint values
    for xi, (mid, n) in enumerate(zip(midpoints, ns)):
        if n > 0:
            ax2.text(xi, n + max(ns)*0.01, f"{mid:.2f}", ha="center", fontsize=7, color="#444")

    fig.tight_layout()
    plot_path = out_path_stem.with_suffix(".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_path}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate forecast error broken down by target percentile.",
    )
    p.add_argument("--run_dir",      type=Path, required=True)
    p.add_argument("--results_json", type=Path, default=None)
    p.add_argument("--checkpoint",   type=Path, default=None)
    p.add_argument("--out_dir",      type=Path, default=None)

    p.add_argument("--experiment",   nargs="+", required=True)
    p.add_argument("--n_percentiles",type=int, default=10,
                   help="Number of percentile bins (10 = deciles, 4 = quartiles)")
    p.add_argument("--percentile_on",default="horizon_mean",
                   choices=["horizon_mean", "horizon_max", "context_last", "context_mean"],
                   help="Which signal statistic to use for binning windows into percentiles")

    p.add_argument("--data_paths",   nargs="+", default=None)
    p.add_argument("--cheatsheet",   default=None)
    p.add_argument("--endogenous",   nargs="+", default=None)
    p.add_argument("--exogenous",    nargs="+", default=None)
    p.add_argument("--target",       nargs="+", default=None)
    p.add_argument("--log_cols",     nargs="+", default=None)
    p.add_argument("--val_frac",     type=float, default=None)
    p.add_argument("--test_frac",    type=float, default=None)
    p.add_argument("--head",         choices=("gaussian", "student_t"), default=None)
    p.add_argument("--model_family", default="auto",
                   choices=("auto","timexer","itransformer","hybrid","patchtst",
                             "mamba","tft","csdi","ncde","timellm"))
    p.add_argument("--no_revin",         action="store_true")
    p.add_argument("--residual_anchor",  action="store_true")
    p.add_argument("--batch_token",      action="store_true")
    p.add_argument("--batch_token_cols", nargs="+", default=None)

    args = p.parse_args()

    run_dir      = args.run_dir.expanduser().resolve()
    results_json = (args.results_json or run_dir / "results.json").expanduser().resolve()
    checkpoint   = (args.checkpoint  or run_dir / "best_model.pt").expanduser().resolve()
    out_dir      = (args.out_dir     or run_dir / "percentile_eval").expanduser().resolve()

    if not results_json.exists():
        raise FileNotFoundError(f"results.json not found: {results_json}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    results  = _load_json(results_json)
    run_cfg  = _merge_run_config(results)

    for attr, key in [("head","head"),("val_frac","val_frac"),("test_frac","test_frac")]:
        v = getattr(args, attr)
        if v is not None: run_cfg[key] = v
    for attr, key in [("endogenous","endogenous"),("exogenous","exogenous"),("log_cols","log_cols")]:
        v = getattr(args, attr)
        if v is not None: run_cfg[key] = v
    if args.target is not None:
        run_cfg["target"] = args.target if len(args.target) > 1 else args.target[0]

    data_paths = args.data_paths or run_cfg.get("data_paths")
    cheatsheet = args.cheatsheet or run_cfg.get("cheatsheet")
    if not data_paths: raise ValueError("No data_paths provided.")
    if not cheatsheet: raise ValueError("No cheatsheet provided.")

    data_paths     = [str(_resolve_path(pp, run_dir)) for pp in data_paths]
    cheatsheet_str = str(_resolve_path(cheatsheet, run_dir))

    endo_cols   = _as_list(run_cfg.get("endogenous"))
    exo_cols    = _as_list(run_cfg.get("exogenous"))
    target_cols = _normalise_targets(run_cfg.get("target"), endo_cols)
    log_cols    = _as_list(run_cfg.get("log_cols")) or list(endo_cols)
    scale_mode  = str(run_cfg.get("scale_mode", "per_var"))
    scaling     = str(run_cfg.get("scaling", "standard"))
    val_frac    = float(run_cfg.get("val_frac", 0.1))
    test_frac   = float(run_cfg.get("test_frac", 0.1))
    head        = str(run_cfg.get("head", "student_t"))
    context_len = int(run_cfg.get("context_len", 180))
    horizon     = int(run_cfg.get("horizon", 18))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  context={context_len}  horizon={horizon}")
    print(f"n_percentiles={args.n_percentiles}  percentile_on={args.percentile_on}")

    sc_endo, sc_exo = _fit_training_scalers(
        run_cfg=run_cfg, data_paths=data_paths, cheatsheet=cheatsheet_str,
        endo_cols=endo_cols, exo_cols=exo_cols,
        scale_mode=scale_mode, scaling=scaling, log_cols=log_cols,
    )
    model = _load_model(
        run_cfg, endo_cols, exo_cols, target_cols, checkpoint, device,
        use_revin=False if args.no_revin else None,
        residual_anchor=True if args.residual_anchor else None,
        batch_token=True if args.batch_token else None,
        batch_token_cols=args.batch_token_cols,
        model_family=args.model_family,
    )
    forecast_color = _forecast_color_for_family(getattr(model, "_forecast_family", "timexer"))

    all_results = {}
    for exp_tag in args.experiment:
        print(f"\n── {exp_tag} ─────────────────────────────────────────────────")
        df = _prepare_experiment_frame(exp_tag, data_paths, cheatsheet_str, log_cols)
        for ti, target_col in enumerate(target_cols):
            stem = out_dir / f"{exp_tag}_{target_col}_percentile_metrics"
            res  = evaluate_experiment(
                model=model, sc_endo=sc_endo, sc_exo=sc_exo, df=df,
                endo_cols=endo_cols, exo_cols=exo_cols,
                scale_mode=scale_mode, context_len=context_len, horizon=horizon,
                target_col=target_col, log_cols=log_cols, head=head,
                target_channel_idx=ti,
                val_frac=val_frac, test_frac=test_frac,
                n_percentiles=args.n_percentiles,
                percentile_on=args.percentile_on,
                out_path_stem=stem,
                forecast_color=forecast_color,
            )
            all_results[f"{exp_tag}_{target_col}"] = res

    # ── Summary across all experiments ────────────────────────────────────
    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary → {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
