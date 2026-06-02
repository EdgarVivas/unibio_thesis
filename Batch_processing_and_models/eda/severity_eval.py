"""
severity_eval.py – Cross-model evaluation bucketed by spike severity.

Instead of binary spike/flat, each test window gets a continuous
severity score:
    severity = actual_horizon.max() - context_tail.mean()
(peak above baseline — isolates the jump, not the absolute level)

Three analyses are produced:
  1. MAE per model × severity bucket  (the decisive table)
  2. Dilution check — are "spike" windows concentrated in mild buckets?
  3. Threshold sweep — does the model ranking change as the spike
     threshold rises from 1.0 → 1.5 → 2.0 → 3.0?

Usage
-----
    python severity_eval.py \\
        --run_dirs results/timexer/KAU084 results/timellm/KAU084 \\
        --model_names timexer timellm \\
        --experiment KAU084 KAU081 KAU071 \\
        --data_paths dataset/final_dataset.csv dataset/final_dataset_2.csv \\
        --cheatsheet filtered_data_stamps.txt \\
        --out_dir results/severity_eval
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
import pandas as pd
import torch

from plot_selected_reactor_windows import (
    _as_list,
    _fit_training_scalers,
    _load_json,
    _load_model,
    _merge_run_config,
    _normalise_targets,
    _prepare_experiment_frame,
    _resolve_path,
)
from timexer_train_reactor_probabilistic import (
    apply_scalers,
    inverse_target,
)


SEVERITY_QUANTILES = [0.0, 0.5, 0.8, 0.9, 1.0]
SEVERITY_LABELS    = ["mild", "moderate", "high", "severe"]
SPIKE_THRESHOLDS   = [1.0, 1.5, 2.0, 3.0]
CONTEXT_TAIL       = 30   # steps used for baseline (context tail mean)


# ── Per-window inference ───────────────────────────────────────────────────

def _forward(model, x_endo_t, x_exo_t, target_channel_idx):
    if getattr(model, "_forecast_family", None) == "csdi":
        with torch.no_grad():
            samples = model.sample(x_endo_t, x_exo_t, n_samples=50)
        return samples.mean(dim=1)[0, target_channel_idx].cpu().numpy()
    with torch.no_grad():
        out = model(x_endo_t, x_exo_t)
    return out[0][0, target_channel_idx].cpu().numpy()


# ── Collect rows for one model × experiment ────────────────────────────────

def collect_windows(
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
    target_channel_idx: int,
    val_frac: float,
    test_frac: float,
    model_name: str,
    exp_tag: str,
) -> List[dict]:
    device   = next(model.parameters()).device
    T        = len(df)
    n_test   = int(T * test_frac)
    n_val    = int(T * val_frac)
    n_train  = T - n_val - n_test

    flags_te   = df["flag"].values.astype(int)[n_train + n_val:]
    raw_te     = df[target_col].values[n_train + n_val:]
    endo_te_sc = apply_scalers(df[endo_cols].values[n_train + n_val:], sc_endo, endo_cols, scale_mode)
    exo_te_sc  = apply_scalers(df[exo_cols ].values[n_train + n_val:], sc_exo,  exo_cols,  scale_mode)

    log_cols_s = set(log_cols)
    inv = lambda a: inverse_target(a, sc_endo, target_col, endo_cols, scale_mode, log_cols_s)

    valid      = np.where(flags_te == 0)[0]
    candidates = [i for i in valid if i + context_len + horizon < len(endo_te_sc)]
    if not candidates:
        return []

    rows = []
    for s in candidates:
        x_e = torch.tensor(endo_te_sc[s : s + context_len].T,
                            dtype=torch.float32, device=device).unsqueeze(0)
        x_x = torch.tensor(exo_te_sc [s : s + context_len].T,
                            dtype=torch.float32, device=device).unsqueeze(0)

        mean_sc  = _forward(model, x_e, x_x, target_channel_idx)
        mean_out = inv(mean_sc)

        actual_sc  = raw_te[s + context_len : s + context_len + horizon]
        actual_out = np.expm1(actual_sc) if target_col in log_cols_s else actual_sc.copy()

        ctx_raw = raw_te[s : s + context_len]
        if target_col in log_cols_s:
            ctx_raw = np.expm1(ctx_raw)

        # peak-above-baseline severity
        tail_mean = float(np.mean(ctx_raw[-CONTEXT_TAIL:]))
        severity  = float(np.max(actual_out) - tail_mean)

        # spike score (same formula as anchor training)
        ctx_std    = float(np.std(ctx_raw)) or 1e-3
        delta      = float(np.max(np.abs(actual_out - ctx_raw[-1])))
        spike_score = delta / ctx_std

        mae  = float(np.mean(np.abs(mean_out - actual_out)))
        rmse = float(np.sqrt(np.mean((mean_out - actual_out) ** 2)))

        rows.append({
            "model":       model_name,
            "experiment":  exp_tag,
            "window":      s,
            "severity":    severity,
            "spike_score": spike_score,
            "is_spike":    spike_score >= 1.0,
            "mae":         mae,
            "rmse":        rmse,
        })

    return rows


# ── Analysis 1 – MAE by model × severity bucket ───────────────────────────

def analysis_mae_by_bucket(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["sev_bucket"] = pd.qcut(
        df["severity"],
        q=SEVERITY_QUANTILES,
        labels=SEVERITY_LABELS,
        duplicates="drop",
    )
    table = df.groupby(["model", "sev_bucket"], observed=True)["mae"].mean().unstack()
    table = table.round(4)

    print("\n── Analysis 1: MAE by model × severity bucket ───────────────────")
    print(table.to_string())

    csv_path = out_dir / "analysis1_mae_by_bucket.csv"
    table.to_csv(csv_path)
    print(f"\n  Saved: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x     = np.arange(len(table.columns))
    width = 0.8 / max(len(table), 1)
    for i, (model_name, row) in enumerate(table.iterrows()):
        ax.bar(x + i * width, row.values, width=width, label=model_name, alpha=0.85)
    ax.set_xticks(x + width * (len(table) - 1) / 2)
    ax.set_xticklabels(table.columns, fontsize=10)
    ax.set_ylabel("Mean MAE (mg/L)", fontsize=11)
    ax.set_title("MAE by model and severity bucket", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    plot_path = out_dir / "analysis1_mae_by_bucket.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {plot_path}")

    return table


# ── Analysis 2 – Dilution check ───────────────────────────────────────────

def analysis_dilution(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["sev_bucket"] = pd.qcut(
        df["severity"],
        q=SEVERITY_QUANTILES,
        labels=SEVERITY_LABELS,
        duplicates="drop",
    )

    spike_df = df[df["is_spike"]]
    counts   = spike_df.groupby("sev_bucket", observed=True).size().rename("n_spike_windows")
    total    = df.groupby("sev_bucket", observed=True).size().rename("n_total")
    dilution = pd.concat([counts, total], axis=1).fillna(0).astype(int)
    dilution["pct_of_spikes"] = (dilution["n_spike_windows"] / dilution["n_spike_windows"].sum() * 100).round(1)
    dilution["pct_spike_in_bucket"] = (dilution["n_spike_windows"] / dilution["n_total"] * 100).round(1)

    print("\n── Analysis 2: Dilution check (spike windows by severity bucket) ─")
    print(dilution.to_string())
    if dilution.loc["mild", "pct_of_spikes"] > 40:
        print("\n  ⚠  >40% of spike-labelled windows fall in the mild bucket.")
        print("     The is_spike threshold is sweeping in flat-ish windows — spike MAE is diluted.")

    csv_path = out_dir / "analysis2_dilution.csv"
    dilution.to_csv(csv_path)
    print(f"\n  Saved: {csv_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    buckets = dilution.index.tolist()
    x = np.arange(len(buckets))

    axes[0].bar(x, dilution["pct_of_spikes"], color="#d62728", alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(buckets)
    axes[0].set_ylabel("% of all spike windows"); axes[0].set_title("Where are spike windows? (dilution check)")
    axes[0].grid(True, alpha=0.2, axis="y")

    axes[1].bar(x, dilution["pct_spike_in_bucket"], color="#ff7f0e", alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(buckets)
    axes[1].set_ylabel("% of bucket windows labelled spike"); axes[1].set_title("Spike density per bucket")
    axes[1].grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    plot_path = out_dir / "analysis2_dilution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {plot_path}")

    return dilution


# ── Analysis 3 – Threshold sweep ──────────────────────────────────────────

def analysis_threshold_sweep(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    records = []
    for thresh in SPIKE_THRESHOLDS:
        spike_df = df[df["spike_score"] >= thresh]
        for model_name, grp in spike_df.groupby("model"):
            records.append({
                "threshold": thresh,
                "model":     model_name,
                "n_windows": len(grp),
                "mae":       grp["mae"].mean(),
                "rmse":      grp["rmse"].mean(),
            })

    sweep = pd.DataFrame(records)

    print("\n── Analysis 3: Threshold sweep ──────────────────────────────────")
    pivot = sweep.pivot_table(index="threshold", columns="model", values="mae").round(4)
    print(pivot.to_string())
    print("\n  (Does the model ranking flip as threshold rises?)")

    csv_path = out_dir / "analysis3_threshold_sweep.csv"
    sweep.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Plot MAE vs threshold, one line per model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = sweep["model"].unique()
    colors = plt.cm.get_cmap("tab10", len(models))

    for i, model_name in enumerate(models):
        sub = sweep[sweep["model"] == model_name].sort_values("threshold")
        axes[0].plot(sub["threshold"], sub["mae"],  marker="o", label=model_name,
                     color=colors(i), lw=2)
        axes[1].plot(sub["threshold"], sub["n_windows"], marker="s", label=model_name,
                     color=colors(i), lw=2, linestyle="--")

    axes[0].set_xlabel("Spike score threshold"); axes[0].set_ylabel("Mean MAE (mg/L)")
    axes[0].set_title("MAE vs threshold (does ranking flip?)"); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)
    axes[1].set_xlabel("Spike score threshold"); axes[1].set_ylabel("Number of windows")
    axes[1].set_title("Windows selected at each threshold"); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    plot_path = out_dir / "analysis3_threshold_sweep.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {plot_path}")

    return sweep


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cross-model severity-bucketed evaluation.",
    )
    p.add_argument("--run_dirs",    nargs="+", required=True, type=Path,
                   help="One run dir per model (must contain best_model.pt + results.json)")
    p.add_argument("--model_names", nargs="+", default=None,
                   help="Display names for each model (defaults to run dir name)")
    p.add_argument("--experiment",  nargs="+", required=True)
    p.add_argument("--out_dir",     type=Path, default=Path("results/severity_eval"))

    p.add_argument("--data_paths",  nargs="+", default=None)
    p.add_argument("--cheatsheet",  default=None)
    p.add_argument("--val_frac",    type=float, default=None)
    p.add_argument("--test_frac",   type=float, default=None)
    p.add_argument("--model_family",default="auto",
                   choices=("auto","timexer","itransformer","hybrid","patchtst",
                             "mamba","tft","csdi","ncde","timellm"))
    p.add_argument("--no_revin",        action="store_true")
    p.add_argument("--residual_anchor", action="store_true")
    p.add_argument("--batch_token",     action="store_true")
    p.add_argument("--batch_token_cols",nargs="+", default=None)

    args = p.parse_args()

    run_dirs = [d.expanduser().resolve() for d in args.run_dirs]
    model_names = args.model_names or [d.name for d in run_dirs]
    if len(model_names) != len(run_dirs):
        raise ValueError("--model_names must have the same length as --run_dirs")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  models: {model_names}")

    all_rows = []

    for run_dir, model_name in zip(run_dirs, model_names):
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}  ({run_dir})")
        print(f"{'='*60}")

        results_json = run_dir / "results.json"
        checkpoint   = run_dir / "best_model.pt"
        if not results_json.exists():
            print(f"  SKIP: results.json not found"); continue
        if not checkpoint.exists():
            print(f"  SKIP: best_model.pt not found"); continue

        results = _load_json(results_json)
        run_cfg = _merge_run_config(results)

        if args.val_frac  is not None: run_cfg["val_frac"]  = args.val_frac
        if args.test_frac is not None: run_cfg["test_frac"] = args.test_frac

        data_paths = args.data_paths or run_cfg.get("data_paths")
        cheatsheet = args.cheatsheet or run_cfg.get("cheatsheet")
        if not data_paths or not cheatsheet:
            print("  SKIP: data_paths or cheatsheet missing"); continue

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
        model.eval()

        for exp_tag in args.experiment:
            print(f"\n  ── {exp_tag}")
            df = _prepare_experiment_frame(exp_tag, data_paths, cheatsheet_str, log_cols)
            for ti, target_col in enumerate(target_cols):
                rows = collect_windows(
                    model=model, sc_endo=sc_endo, sc_exo=sc_exo, df=df,
                    endo_cols=endo_cols, exo_cols=exo_cols,
                    scale_mode=scale_mode, context_len=context_len, horizon=horizon,
                    target_col=target_col, log_cols=log_cols,
                    target_channel_idx=ti,
                    val_frac=val_frac, test_frac=test_frac,
                    model_name=model_name, exp_tag=exp_tag,
                )
                print(f"    {len(rows):,} windows collected")
                all_rows.extend(rows)

    if not all_rows:
        print("\nNo windows collected — check your run dirs and experiments."); return

    df_all = pd.DataFrame(all_rows)
    csv_path = out_dir / "all_windows.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nFull window table → {csv_path}  ({len(df_all):,} rows)")

    print(f"\nSeverity summary:")
    print(df_all.groupby("model")["severity"].describe().round(3).to_string())

    analysis_mae_by_bucket(df_all, out_dir)
    analysis_dilution(df_all, out_dir)
    analysis_threshold_sweep(df_all, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
