"""
rolling_forecast_plot.py – Rolling non-overlapping forecast visualisation.

For each experiment, picks N starting positions in the test split and
produces one subplot per position.  At each position it tiles `n_forecasts`
consecutive horizon-length forecasts so you can see how the model tracks the
signal over a longer stretch.

Layout of each subplot
----------------------

  ←context_show→ | ←──────── n_forecasts × horizon ────────→|
   gray  context    [fc0]  [fc1]  ...  [fc9]    (mean + PI)
   blue  actual     continuous ground-truth signal

Usage
-----
    python rolling_forecast_plot.py \\
        --run_dir results/KAU084 \\
        --experiment KAU084 KAU081 \\
        --data_paths dataset/final_dataset.csv dataset/final_dataset_2.csv \\
        --cheatsheet filtered_data_stamps.txt \\
        --n_forecasts 10 \\
        --n_blocks 3 \\
        --out_dir results/rolling
"""
from __future__ import annotations

import argparse
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


# ── Core rolling plot ──────────────────────────────────────────────────────

def _run_one_forecast(model, x_endo_t, x_exo_t, target_channel_idx, head, device):
    """Run inference and return (mean_sc, std_sc, z_1s, z_95)."""
    if getattr(model, "_forecast_family", None) == "csdi":
        with torch.no_grad():
            samples = model.sample(x_endo_t, x_exo_t, n_samples=100)
        mean_sc = samples.mean(dim=1)[0, target_channel_idx].cpu().numpy()
        std_sc  = samples.std(dim=1)[0, target_channel_idx].cpu().numpy()
        return mean_sc, std_sc, 1.0, 1.96

    with torch.no_grad():
        out = model(x_endo_t, x_exo_t)

    mean_sc  = out[0][0, target_channel_idx].cpu().numpy()
    scale_sc = out[1][0, target_channel_idx].cpu().numpy()

    if head == "gaussian":
        std_sc = np.exp(0.5 * np.clip(scale_sc, LOG_VAR_MIN, LOG_VAR_MAX))
        return mean_sc, std_sc, 1.0, 1.96

    from scipy.stats import t as _st
    std_sc  = np.exp(np.clip(scale_sc, LOG_SIG_MIN, LOG_SIG_MAX))
    log_nu  = out[2][0, target_channel_idx].cpu().numpy()
    nu_med  = float(np.median(np.log1p(np.exp(np.clip(log_nu, -20.0, 20.0))) + 2.0))
    return mean_sc, std_sc, float(_st.ppf(0.84, df=nu_med)), float(_st.ppf(0.975, df=nu_med))


def render_rolling(
    model,
    sc_endo,
    sc_exo,
    df,
    endo_cols: Sequence[str],
    exo_cols: Sequence[str],
    scale_mode: str,
    context_len: int,
    horizon: int,
    out_path: Path,
    target_col: str,
    log_cols: Sequence[str],
    head: str,
    target_channel_idx: int,
    val_frac: float,
    test_frac: float,
    n_forecasts: int = 10,
    n_blocks: int = 3,
    context_show: int = 60,
    seed: int = 42,
    block_starts: Optional[List[int]] = None,
) -> None:
    device = next(model.parameters()).device
    T      = len(df)
    n_test = int(T * test_frac)
    n_val  = int(T * val_frac)
    n_train = T - n_val - n_test

    flags_te    = df["flag"].values.astype(int)[n_train + n_val:]
    raw_te      = df[target_col].values[n_train + n_val:]
    endo_te_sc  = apply_scalers(df[endo_cols].values[n_train + n_val:], sc_endo, endo_cols, scale_mode)
    exo_te_sc   = apply_scalers(df[exo_cols ].values[n_train + n_val:], sc_exo,  exo_cols,  scale_mode)

    span        = n_forecasts * horizon
    log_cols_s  = set(log_cols)
    inv = lambda a: inverse_target(a, sc_endo, target_col, endo_cols, scale_mode, log_cols_s)

    # Valid block starts: need context_len before + span after, all flag=0
    candidates = []
    for i in range(context_len, len(flags_te) - span):
        if (flags_te[i - context_len : i + span] == 0).all():
            candidates.append(i)

    if not candidates:
        print(f"  [{target_col}] No valid rolling blocks found in test split — skipping.")
        return

    if block_starts is not None:
        candidate_set = set(candidates)
        invalid = [s for s in block_starts if s not in candidate_set]
        if invalid:
            print(f"  WARNING: these block_starts are not valid in the test split: {invalid}")
        starts = sorted(s for s in block_starts if s in candidate_set)
        if not starts:
            print(f"  No valid block_starts remain — skipping."); return
    else:
        rng    = np.random.default_rng(seed)
        n_pick = min(n_blocks, len(candidates))
        starts = sorted(rng.choice(candidates, size=n_pick, replace=False))

    forecast_color = _forecast_color_for_family(getattr(model, "_forecast_family", "timexer"))

    fig, axes = plt.subplots(len(starts), 1,
                             figsize=(14, 4.5 * len(starts)), squeeze=False)

    for row, block_start in enumerate(starts):
        ax = axes[row][0]

        # ── Context tail ───────────────────────────────────────────────
        ctx_show = min(context_show, context_len)
        ctx_raw  = raw_te[block_start - ctx_show : block_start]
        if target_col in log_cols_s:
            ctx_raw = np.expm1(ctx_raw)
        ctx_x = np.arange(-ctx_show, 0)
        ax.plot(ctx_x, ctx_raw, color="#888888", lw=1.2, alpha=0.7, label="Context")

        # ── Actual signal across full span ─────────────────────────────
        actual_raw = raw_te[block_start : block_start + span]
        if target_col in log_cols_s:
            actual_raw = np.expm1(actual_raw)
        ax.plot(np.arange(span), actual_raw, color="black", lw=1.8, label="Actual", zorder=3)

        # ── Tiled forecasts ────────────────────────────────────────────
        all_mae = []
        for fi in range(n_forecasts):
            fc_origin = block_start + fi * horizon   # forecast origin
            ctx_start = fc_origin - context_len      # context window start
            x_e = torch.tensor(
                endo_te_sc[ctx_start : fc_origin].T,
                dtype=torch.float32, device=device).unsqueeze(0)
            x_x = torch.tensor(
                exo_te_sc [ctx_start : fc_origin].T,
                dtype=torch.float32, device=device).unsqueeze(0)

            mean_sc, std_sc, z1, z95 = _run_one_forecast(
                model, x_e, x_x, target_channel_idx, head, device)

            mean_orig  = inv(mean_sc)
            upper_1s   = inv(mean_sc + z1  * std_sc)
            lower_1s   = inv(mean_sc - z1  * std_sc)
            upper_95   = inv(mean_sc + z95 * std_sc)
            lower_95   = inv(mean_sc - z95 * std_sc)

            fore_x     = np.arange(fi * horizon, (fi + 1) * horizon)
            actual_seg = actual_raw[fi * horizon : (fi + 1) * horizon]

            ax.fill_between(fore_x, lower_95, upper_95,
                            color=forecast_color, alpha=0.12,
                            label="95% PI" if fi == 0 else None)
            ax.fill_between(fore_x, lower_1s, upper_1s,
                            color=forecast_color, alpha=0.28,
                            label="68% PI" if fi == 0 else None)
            ax.plot(fore_x, mean_orig, color=forecast_color, lw=1.8,
                    label="Forecast" if fi == 0 else None)
            ax.axvline(fi * horizon, color="black", lw=0.5, ls=":", alpha=0.5)

            all_mae.append(float(np.mean(np.abs(mean_orig - actual_seg))))

        mean_mae = float(np.mean(all_mae))
        ax.axvline(0, color="black", lw=0.9, ls=":")
        ax.set_title(
            f"Test block starting at step {block_start}  |  "
            f"mean MAE over {n_forecasts} forecasts = {mean_mae:.3f} mg/L",
            fontsize=9)
        ax.set_xlabel("Steps from block origin", fontsize=8)
        ax.set_ylabel(f"{target_col.upper()} (mg/L)", fontsize=8)
        ax.grid(True, alpha=0.2)

        # x-ticks at every horizon boundary
        xticks = np.arange(0, span + 1, horizon)
        ax.set_xticks(np.concatenate([np.arange(-ctx_show, 0, max(1, ctx_show // 4)), xticks]))
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"{target_col.upper()} rolling forecast  |  "
        f"{n_forecasts} × {horizon}-step tiles  |  {out_path.stem}",
        fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Rolling non-overlapping forecast plot from an existing checkpoint.",
    )
    p.add_argument("--run_dir",     type=Path, required=True)
    p.add_argument("--results_json",type=Path, default=None)
    p.add_argument("--checkpoint",  type=Path, default=None)
    p.add_argument("--out_dir",     type=Path, default=None)

    p.add_argument("--experiment",  nargs="+", required=True,
                   help="Experiment tags to plot (e.g. KAU084 KAU081)")
    p.add_argument("--n_forecasts",  type=int, default=10,
                   help="Number of tiled horizon-length forecasts per block")
    p.add_argument("--n_blocks",     type=int, default=3,
                   help="Number of independent time blocks per experiment (ignored if --block_starts is set)")
    p.add_argument("--block_starts", type=int, nargs="+", default=None,
                   help="Explicit block start positions in the test split (overrides --n_blocks)")
    p.add_argument("--context_show",type=int, default=60,
                   help="Context steps shown before each block")
    p.add_argument("--seed",        type=int, default=42)

    p.add_argument("--data_paths",  nargs="+", default=None)
    p.add_argument("--cheatsheet",  default=None)
    p.add_argument("--endogenous",  nargs="+", default=None)
    p.add_argument("--exogenous",   nargs="+", default=None)
    p.add_argument("--target",      nargs="+", default=None)
    p.add_argument("--log_cols",    nargs="+", default=None)
    p.add_argument("--val_frac",    type=float, default=None)
    p.add_argument("--test_frac",   type=float, default=None)
    p.add_argument("--head",        choices=("gaussian", "student_t"), default=None)
    p.add_argument("--model_family",default="auto",
                   choices=("auto","timexer","itransformer","hybrid","patchtst",
                             "mamba","tft","csdi","ncde","timellm"))
    p.add_argument("--no_revin",        action="store_true")
    p.add_argument("--residual_anchor", action="store_true")
    p.add_argument("--batch_token",     action="store_true")
    p.add_argument("--batch_token_cols",nargs="+", default=None)

    args = p.parse_args()

    run_dir     = args.run_dir.expanduser().resolve()
    results_json = (args.results_json or run_dir / "results.json").expanduser().resolve()
    checkpoint   = (args.checkpoint  or run_dir / "best_model.pt").expanduser().resolve()
    out_dir      = (args.out_dir     or run_dir / "rolling_forecast").expanduser().resolve()

    if not results_json.exists():
        raise FileNotFoundError(f"results.json not found: {results_json}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    results = _load_json(results_json)
    run_cfg  = _merge_run_config(results)

    for attr, key in [("head","head"), ("val_frac","val_frac"), ("test_frac","test_frac")]:
        val = getattr(args, attr)
        if val is not None:
            run_cfg[key] = val
    for attr, key in [("endogenous","endogenous"), ("exogenous","exogenous"),
                      ("log_cols","log_cols")]:
        val = getattr(args, attr)
        if val is not None:
            run_cfg[key] = val
    if args.target is not None:
        run_cfg["target"] = args.target if len(args.target) > 1 else args.target[0]

    data_paths = args.data_paths or run_cfg.get("data_paths")
    cheatsheet = args.cheatsheet or run_cfg.get("cheatsheet")
    if not data_paths:
        raise ValueError("No data_paths provided.")
    if not cheatsheet:
        raise ValueError("No cheatsheet provided.")

    data_paths     = [str(_resolve_path(p, run_dir)) for p in data_paths]
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
    context_len = int(run_cfg.get("context_len", 360))
    horizon     = int(run_cfg.get("horizon", 72))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  context={context_len}  horizon={horizon}")
    print(f"n_forecasts={args.n_forecasts}  n_blocks={args.n_blocks}  seed={args.seed}")

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

    for exp_tag in args.experiment:
        print(f"\n── {exp_tag} ─────────────────────────────────────────────────")
        df = _prepare_experiment_frame(exp_tag, data_paths, cheatsheet_str, log_cols)
        for ti, target_col in enumerate(target_cols):
            out_path = out_dir / f"{exp_tag}_{target_col}_rolling.png"
            render_rolling(
                model=model, sc_endo=sc_endo, sc_exo=sc_exo, df=df,
                endo_cols=endo_cols, exo_cols=exo_cols,
                scale_mode=scale_mode, context_len=context_len, horizon=horizon,
                out_path=out_path, target_col=target_col, log_cols=log_cols,
                head=head, target_channel_idx=ti,
                val_frac=val_frac, test_frac=test_frac,
                n_forecasts=args.n_forecasts, n_blocks=args.n_blocks,
                block_starts=args.block_starts,
                context_show=args.context_show, seed=args.seed,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
