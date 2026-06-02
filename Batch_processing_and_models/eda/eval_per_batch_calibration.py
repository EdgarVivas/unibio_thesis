"""
eval_per_batch_calibration.py

Re-runs inference on each experiment batch from a saved model folder and
computes per-batch calibration coverage (68%, 90%, 95%).  The original
training run only records aggregate calibration in results.json; this script
fills in the per-experiment breakdown that per_experiment_metrics lacks.

Usage:
    python eval_per_batch_calibration.py <model_folder>

    <model_folder> must contain:
        best_model.pt       — saved weights
        results.json        — training args and experiment list
        scaler_endo.joblib  — fitted endogenous scaler
        scaler_exo.joblib   — fitted exogenous scaler
        col_config.json     — endo_cols / exo_cols / scale_mode

Outputs:
    Prints a per-batch calibration table to stdout.
    Writes  per_batch_calibration.json  next to results.json.

Supports all four model types: timexer (own), patchtst, itransformer, timellm.
Model type is auto-detected from the folder path and the stored out_dir.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# All training scripts share the same data-pipeline code; import from timexer.
from timexer_train_reactor_probabilistic import (
    load_experiment,
    compute_time_since_valid,
    apply_scalers,
    ReactorDataset,
    collect_predictions,
    coverage_report,
    spike_window_mask,
)


# ---------------------------------------------------------------------------
# Model detection and construction
# ---------------------------------------------------------------------------

def detect_model_type(folder: Path, out_dir: str) -> str:
    """Return one of: timexer | patchtst | itransformer | timellm."""
    tag = (str(folder) + " " + out_dir).lower()
    if "timellm" in tag:
        return "timellm"
    if "itransformer" in tag:
        return "itransformer"
    if "patchtst" in tag:
        return "patchtst"
    return "timexer"


def build_model(model_type: str, a, endo_cols, exo_cols, target_cols, device):
    """Reconstruct the model from stored args (SimpleNamespace a)."""
    n_forecast = len(target_cols)
    fti = [endo_cols.index(t) for t in target_cols]

    bt_idx = None
    if getattr(a, "batch_token", False):
        bt_cols = getattr(a, "batch_token_cols", ["inoc_amount", "inoc_conc"])
        bt_idx = [exo_cols.index(c) for c in bt_cols if c in exo_cols]

    if model_type == "timexer":
        from timexer_train_reactor_probabilistic import ReactorTimeXerModel
        m = ReactorTimeXerModel(
            c_endo=len(endo_cols), c_exo=len(exo_cols),
            context_len=a.context_len, horizon=a.horizon,
            n_forecast=n_forecast, forecast_target_indices=fti,
            head=a.head, patch_size=a.patch_size,
            d_model=a.d_model, n_heads=a.n_heads, n_layers=a.n_layers,
            d_ff=a.d_ff, dropout=a.dropout,
            use_revin=not getattr(a, "no_revin", False),
            residual_anchor=getattr(a, "residual_anchor", False),
            use_batch_token=getattr(a, "batch_token", False),
            batch_token_idx=bt_idx,
        )

    elif model_type == "patchtst":
        from patchtst_train_reactor_probabilistic import ReactorPatchTSTModel
        m = ReactorPatchTSTModel(
            c_endo=len(endo_cols), c_exo=len(exo_cols),
            context_len=a.context_len, horizon=a.horizon,
            n_forecast=n_forecast, forecast_target_indices=fti,
            d_model=a.d_model, n_heads=a.n_heads, n_layers=a.n_layers,
            patch_len=getattr(a, "patch_len", 16),
            patch_stride=getattr(a, "patch_stride", 8),
            dropout=a.dropout, head=a.head,
            residual_anchor=getattr(a, "residual_anchor", False),
            use_revin=not getattr(a, "no_revin", False),
            use_batch_token=getattr(a, "batch_token", False),
            batch_token_idx=bt_idx,
        )

    elif model_type == "itransformer":
        from itransformer_train_reactor_probabilistic import ReactorITransformerModel
        m = ReactorITransformerModel(
            c_endo=len(endo_cols), c_exo=len(exo_cols),
            context_len=a.context_len, horizon=a.horizon,
            n_forecast=n_forecast, forecast_target_indices=fti,
            d_model=a.d_model, n_heads=a.n_heads, n_layers=a.n_layers,
            dropout=a.dropout, head=a.head,
            residual_anchor=getattr(a, "residual_anchor", False),
            use_revin=not getattr(a, "no_revin", False),
            use_batch_token=getattr(a, "batch_token", False),
            batch_token_idx=bt_idx,
            raw_target_as_exo=getattr(a, "raw_target_as_exo", False),
        )

    elif model_type == "timellm":
        from timellm_train_reactor_probabilistic import ReactorTimeLLMModel
        bt_names = None
        if bt_idx:
            bt_cols = getattr(a, "batch_token_cols", ["inoc_amount", "inoc_conc"])
            bt_names = [c for c in bt_cols if c in exo_cols]
        m = ReactorTimeLLMModel(
            c_endo=len(endo_cols), c_exo=len(exo_cols),
            context_len=a.context_len, horizon=a.horizon,
            n_forecast=n_forecast, forecast_target_indices=fti,
            patch_size=getattr(a, "patch_size", 16),
            n_llm_layers=getattr(a, "n_llm_layers", 6),
            n_heads=getattr(a, "n_heads", 8),
            n_source=getattr(a, "n_source", 512),
            dropout=a.dropout, head=a.head,
            use_revin=not getattr(a, "no_revin", False),
            use_batch_token=getattr(a, "batch_token", False),
            batch_token_idx=bt_idx,
            batch_token_names=bt_names,
            llm_model=getattr(a, "llm_model", "gpt2"),
            prompt_text=getattr(a, "prompt_text", ""),
            use_stats_prompt=getattr(a, "stats_prompt", False),
            residual_anchor=getattr(a, "residual_anchor", False),
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return m.to(device), fti


# ---------------------------------------------------------------------------
# Per-batch evaluation
# ---------------------------------------------------------------------------

def eval_batch(model, sc_endo, sc_exo, a, endo_cols, exo_cols, target_cols,
               log_cols, fti, exp_tag, device):
    """Run inference on the test split of one experiment; return calibration dict."""
    df_e = load_experiment(a.data_paths, a.cheatsheet, [exp_tag])
    if df_e is None or df_e.empty:
        print(f"  [{exp_tag}]  no data — skipping")
        return None

    df_e["time_since_valid"] = compute_time_since_valid(df_e)
    for col in log_cols:
        if col in df_e.columns:
            import numpy as _np
            df_e[col] = _np.log1p(df_e[col].values)

    T_e    = len(df_e)
    n_te_e = int(T_e * a.test_frac)
    te_sl  = slice(T_e - n_te_e, None)

    endo_e  = df_e[endo_cols].values.astype(float)
    exo_e   = df_e[exo_cols ].values.astype(float)
    flags_e = df_e["flag"].values.astype(int)
    tsv_e   = df_e["time_since_valid"].values.astype("float32")
    times_e = df_e["time"].values

    ds_e = ReactorDataset(
        apply_scalers(endo_e[te_sl], sc_endo, endo_cols, a.scale_mode),
        apply_scalers(exo_e [te_sl], sc_exo,  exo_cols,  a.scale_mode),
        flags_e[te_sl], a.context_len, a.horizon,
        stride=1,
        segment_isolated=a.segment_isolated,
        gap_threshold=a.gap_threshold,
        time_since_valid=tsv_e[te_sl],
        times=times_e[te_sl],
        segment_gap_minutes=a.segment_gap_minutes,
        min_segment_rows=a.min_segment_rows,
        target_indices=fti,
    )

    if len(ds_e) == 0:
        print(f"  [{exp_tag}]  empty test split — skipping")
        return None

    dl_e = DataLoader(ds_e, batch_size=a.batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)
    means, scales, tgts, nus = collect_predictions(model, dl_e, device, a.head)
    N = len(means)

    smask    = spike_window_mask(ds_e, fti, a.spike_metric_threshold)
    n_spike  = int(smask.sum())

    cov       = coverage_report(means,         scales,         tgts,         a.head, nus)
    cov_spike = coverage_report(means[smask],  scales[smask],  tgts[smask],  a.head,
                                nus[smask] if nus is not None else None) if n_spike else {}
    cov_flat  = coverage_report(means[~smask], scales[~smask], tgts[~smask], a.head,
                                nus[~smask] if nus is not None else None) if (N - n_spike) else {}

    return {
        "n_windows":      N,
        "n_spike":        n_spike,
        "calibration":    cov,
        "calib_spike":    cov_spike,
        "calib_flat":     cov_flat,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _fmt(cov: dict) -> str:
    if not cov:
        return "—"
    return "  ".join(f"{k}: {v:.1%}" for k, v in cov.items())


def print_table(per_batch: dict):
    levels = ["68%", "90%", "95%"]
    header = f"{'Batch':<10}  {'N':>5}  {'Spk':>4}  "
    header += "  ".join(f"{'Overall ' + lv:<12}" for lv in levels)
    header += "  |  " + "  ".join(f"{'Spike ' + lv:<12}" for lv in levels)
    print("\n" + header)
    print("-" * len(header))
    for batch, res in per_batch.items():
        if res is None:
            print(f"{batch:<10}  (no data)")
            continue
        cov      = res["calibration"]
        cov_sp   = res["calib_spike"]
        overall  = "  ".join(f"{cov.get(lv, float('nan')):.1%}{'':<5}" for lv in levels)
        spike    = "  ".join(f"{cov_sp.get(lv, float('nan')):.1%}{'':<5}"
                              if cov_sp else f"{'—':<8}" for lv in levels)
        print(f"{batch:<10}  {res['n_windows']:>5}  {res['n_spike']:>4}  {overall}  |  {spike}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute per-batch calibration from saved weights.")
    parser.add_argument("folder", type=Path,
                        help="Model folder containing best_model.pt and results.json")
    parser.add_argument("--device", default="auto",
                        help="'cpu', 'cuda', or 'auto' (default: auto)")
    cli = parser.parse_args()

    folder = cli.folder.resolve()
    if not folder.exists():
        sys.exit(f"ERROR: folder not found: {folder}")

    # ── Load saved artefacts ────────────────────────────────────────────────
    rj_path = folder / "results.json"
    if not rj_path.exists():
        sys.exit(f"ERROR: results.json not found in {folder}")
    with open(rj_path) as f:
        saved = json.load(f)

    raw_args = saved.get("args", {})
    a = SimpleNamespace(**raw_args)

    cc_path = folder / "col_config.json"
    if not cc_path.exists():
        sys.exit(f"ERROR: col_config.json not found in {folder}")
    with open(cc_path) as f:
        cc = json.load(f)
    endo_cols  = cc["endo_cols"]
    exo_cols   = cc["exo_cols"]
    a.scale_mode = cc.get("scale_mode", getattr(a, "scale_mode", "per_var"))

    sc_endo = joblib.load(folder / "scaler_endo.joblib")
    sc_exo  = joblib.load(folder / "scaler_exo.joblib")

    target_cols = saved.get("target", [endo_cols[0]])
    log_cols    = set(saved.get("log_cols", target_cols))
    experiments = saved.get("experiments", getattr(a, "experiment", []))

    # ── Device ──────────────────────────────────────────────────────────────
    if cli.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cli.device)
    print(f"\nDevice : {device}")
    print(f"Folder : {folder}")
    print(f"Batches: {experiments}")

    # ── Build and load model ─────────────────────────────────────────────────
    model_type = detect_model_type(folder, getattr(a, "out_dir", ""))
    print(f"Model  : {model_type}")

    model, fti = build_model(model_type, a, endo_cols, exo_cols, target_cols, device)

    ckpt = folder / "best_model.pt"
    state = torch.load(ckpt, map_location=device, weights_only=True)

    # Adapt checkpoint shapes to current model shapes where element count matches.
    # This handles architecture revisions (e.g. pos_emb [1,S,D] → [1,1,S,D]).
    model_state = model.state_dict()
    adapted, reshaped, skipped = {}, [], []
    for key, val in state.items():
        if key in model_state and val.shape != model_state[key].shape:
            tgt = model_state[key].shape
            if val.numel() == model_state[key].numel():
                adapted[key] = val.reshape(tgt)
                reshaped.append(f"{key}: {list(val.shape)}→{list(tgt)}")
            else:
                skipped.append(f"{key}: {list(val.shape)} vs {list(tgt)}")
        else:
            adapted[key] = val

    missing, unexpected = model.load_state_dict(adapted, strict=False)
    if reshaped:
        print(f"  Reshaped : {reshaped}")
    if skipped:
        print(f"  WARNING skipped (size mismatch): {skipped}")
    if missing:
        print(f"  WARNING missing keys: {missing[:5]}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected[:5]}")
    model.eval()
    print(f"  Loaded weights from {ckpt.name}"
          f"  ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Per-batch evaluation ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Per-batch calibration  (test_frac={a.test_frac})")
    print(f"{'─'*60}")

    per_batch = {}
    for exp_tag in experiments:
        print(f"\n  Evaluating {exp_tag} ...")
        res = eval_batch(model, sc_endo, sc_exo, a, endo_cols, exo_cols,
                         target_cols, log_cols, fti, exp_tag, device)
        per_batch[exp_tag] = res
        if res:
            print(f"    Overall:  {_fmt(res['calibration'])}")
            if res["calib_spike"]:
                print(f"    Spike:    {_fmt(res['calib_spike'])}")
            if res["calib_flat"]:
                print(f"    Non-spike:{_fmt(res['calib_flat'])}")

    # ── Summary table ────────────────────────────────────────────────────────
    print_table(per_batch)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = folder / "per_batch_calibration.json"
    with open(out_path, "w") as f:
        json.dump(per_batch, f, indent=2)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
