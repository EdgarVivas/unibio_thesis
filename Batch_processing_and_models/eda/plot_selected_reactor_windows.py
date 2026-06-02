"""Render selected reactor forecast windows from a trained TimeXer run.

This utility mirrors the reactor forecasting data treatment from the
training scripts but skips training and random window selection. It loads
an existing run directory containing ``results.json`` and ``best_model.pt``,
refits the training scalers from the same concatenated training split, and
then renders only the requested test windows.

By default it renders the two window groups requested in the task:

  - KAU075: 1189, 1250, 1271, 1316
  - KAU081: 179, 555, 695, 732

The output is one figure per experiment, each figure containing only the
selected windows stacked vertically.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from timexer_train_reactor_probabilistic import (
    LOG_SIG_MAX,
    LOG_SIG_MIN,
    LOG_VAR_MAX,
    LOG_VAR_MIN,
    ReactorTimeXerModel,
    apply_scalers,
    compute_time_since_valid,
    fit_scalers,
    inverse_target,
    load_experiment,
)
from patchtst_train_reactor_probabilistic import ReactorPatchTSTModel
from itransformer_train_reactor_probabilistic import ReactorITransformerModel
try:
    from timellm_train_reactor_probabilistic import ReactorTimeLLMModel
except Exception:
    ReactorTimeLLMModel = None
try:
    from mambats_train_reactor_probabilistic import ReactorMambaTSModel
except Exception:
    ReactorMambaTSModel = None
try:
    from tft_train_reactor_probabilistic import ReactorTFTModel
except Exception:
    ReactorTFTModel = None
try:
    from csdi_train_reactor_probabilistic import ReactorCSDIModel
except Exception:
    ReactorCSDIModel = None
try:
    from ncde_train_reactor_probabilistic import ReactorCDEModel
except Exception:
    ReactorCDEModel = None


def _import_hybrid_model_classes():
    classes = []
    for module_name in (
        "anchor_issue_sequential_train_reactor_probabilistic",
        "absolute_train_reactor_probabilistic_absolute_values",
        "delta_train_reactor_probabilistic",
    ):
        try:
            module = importlib.import_module(module_name)
            model_cls = getattr(module, "ReactorProbModel", None)
            if model_cls is not None:
                classes.append(model_cls)
        except Exception:
            continue
    return classes


HYBRID_MODEL_CLASSES = _import_hybrid_model_classes()


DEFAULT_WINDOW_SPECS = (
    "KAU075:1189,1250,1271,1316",
    "KAU081:179,555,695,732",
)

FAMILY_FORECAST_COLORS = {
    "itransformer": "#d62728",
    "timexer": "#ff7f0e",
    "hybrid": "#ff7f0e",
    "patchtst": "#2ca02c",
    "timellm": "#1f77b4",
    "csdi": "#9467bd",
    "mamba": "#e377c2",
    "tft": "#8c564b",
    "ncde": "#17becf",
}


def _forecast_color_for_family(family: str) -> str:
    return FAMILY_FORECAST_COLORS.get(str(family).lower(), "#ff7f0e")


def _load_json(path: Path) -> Dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(value: str | Path, start_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()

    for base in (start_dir, *start_dir.parents):
        probe = (base / candidate).resolve()
        if probe.exists():
            return probe

    return (start_dir / candidate).resolve()


def _merge_run_config(results: Dict) -> Dict:
    merged: Dict = {}
    args_blob = results.get("args")
    if isinstance(args_blob, dict):
        merged.update(args_blob)

    for key, value in results.items():
        if key == "args":
            continue
        merged[key] = value

    return merged


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def _normalise_targets(raw_target, endogenous: Sequence[str]) -> List[str]:
    targets = _as_list(raw_target)
    if not targets:
        if not endogenous:
            raise ValueError("No target specified and endogenous list is empty.")
        return [str(endogenous[0])]

    if len(targets) == 1 and targets[0].lower() == "none":
        if not endogenous:
            raise ValueError("Target is None and endogenous list is empty.")
        return [str(endogenous[0])]

    return targets


def _parse_window_specs(specs: Sequence[str]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for spec in specs:
        if ":" in spec:
            tag, window_text = spec.split(":", 1)
        elif "=" in spec:
            tag, window_text = spec.split("=", 1)
        else:
            raise ValueError(
                f"Invalid window spec {spec!r}. Use EXPERIMENT:idx1,idx2,..."
            )

        tag = tag.strip()
        if not tag:
            raise ValueError(f"Invalid empty experiment tag in spec {spec!r}.")

        raw_items = [item for item in window_text.replace(";", ",").split(",")]
        windows: List[int] = []
        for raw_item in raw_items:
            piece = raw_item.strip()
            if not piece:
                continue
            for token in piece.split():
                if token:
                    windows.append(int(token))

        if not windows:
            raise ValueError(f"No window indices found in spec {spec!r}.")

        grouped.setdefault(tag, []).extend(windows)

    return grouped


def _candidate_starts(flags: np.ndarray, context_len: int, horizon: int) -> List[int]:
    valid = np.where(flags == 0)[0]
    return [i for i in valid if i + context_len + horizon < len(flags)]


def _fit_training_scalers(
    run_cfg: Dict,
    data_paths: Sequence[str],
    cheatsheet: str,
    endo_cols: Sequence[str],
    exo_cols: Sequence[str],
    scale_mode: str,
    scaling: str,
    log_cols: Sequence[str],
):
    training_tags = _as_list(run_cfg.get("experiments"))
    if not training_tags:
        training_tags = _as_list(run_cfg.get("experiment"))
    if not training_tags:
        raise ValueError("Run configuration does not define the training experiments.")

    df = load_experiment(list(data_paths), cheatsheet, training_tags)
    if df.empty:
        raise ValueError("Combined training dataframe is empty; cannot fit scalers.")

    df = df.copy()
    df["time_since_valid"] = compute_time_since_valid(df)
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].values)

    total_len = len(df)
    n_test = int(total_len * float(run_cfg.get("test_frac", 0.1)))
    n_val = int(total_len * float(run_cfg.get("val_frac", 0.1)))
    n_train = total_len - n_val - n_test
    if n_train <= 0:
        raise ValueError("Training split is empty; check val_frac/test_frac in the run config.")

    train_slice = slice(0, n_train)
    sc_endo = fit_scalers(
        df[list(endo_cols)].values[train_slice].astype(np.float64),
        list(endo_cols),
        scaling,
        scale_mode,
    )
    sc_exo = fit_scalers(
        df[list(exo_cols)].values[train_slice].astype(np.float64),
        list(exo_cols),
        scaling,
        scale_mode,
    )
    return sc_endo, sc_exo


def _infer_model_family(state_dict: Dict[str, torch.Tensor], run_cfg: Dict) -> str:
    keys = set(state_dict.keys())
    if "patch_proj.weight" in keys and "reprogramming.source" in keys and any(key.startswith("llm.") for key in keys):
        return "timellm"
    if any(key.startswith("blocks.") and ".ssm." in key for key in keys):
        return "mamba"
    if "patch_embed.weight" in keys and "pos_embed" in keys and any(key.startswith("head_net.") for key in keys):
        return "patchtst"
    if (
        "global_tokens" in keys
        or "last_patch_emb" in keys
        or "var_emb" in keys
        or any(key.startswith(("shared_tcn.", "endo_patch.", "layers.0.intra_attn.")) for key in keys)
    ):
        return "hybrid"
    if any(key.startswith("score_layers.") for key in keys) or any(key.startswith("diff_emb.") for key in keys):
        return "csdi"
    if any(key.startswith("cde_func.") for key in keys) or any(key.startswith("initial_net.") for key in keys):
        return "ncde"
    if any(key.startswith("vsn.") for key in keys) or any(key.startswith("grn_pre_lstm.") for key in keys):
        return "tft"
    if (
        "global_token" in keys
        or "pos_emb" in keys
        or "exo_var_emb" in keys
        or "endo_var_emb" in keys
        or any(key.startswith(("exo_embed.", "endo_patch_embed.", "layers.0.self_attn.", "layers.0.cross_attn.")) for key in keys)
    ):
        return "timexer"

    if "var_embed.weight" in keys and any(key.startswith("transformer.") for key in keys):
        return "itransformer"

    declared = str(run_cfg.get("model_family", run_cfg.get("model", "auto"))).lower()
    if declared in {"timellm", "timexer", "itransformer", "hybrid", "patchtst", "mamba", "tft", "csdi", "ncde"}:
        return declared

    raise ValueError(
        "Could not infer checkpoint family from state_dict keys. "
        "Pass --model_family timellm, timexer, itransformer, hybrid, patchtst, mamba, tft, csdi, or ncde."
    )


def _infer_common_state_flags(state_dict: Dict[str, torch.Tensor]):
    keys = set(state_dict.keys())
    use_revin = any(key.startswith("revin.") for key in keys)
    use_batch_token = any(key.startswith("batch_proj.") for key in keys)
    return use_revin, use_batch_token


def _infer_hybrid_head_mode(state_dict: Dict[str, torch.Tensor], run_cfg: Dict) -> bool:
    keys = set(state_dict.keys())
    if any(key.startswith(("head.h_init.", "head.gru.", "head.out_proj.")) for key in keys):
        return True
    if any(key == "sigma_step_bias" or key.startswith("head.0.") for key in keys):
        return False
    return not bool(run_cfg.get("no_ar_head", False))


def _load_checkpoint_state(checkpoint: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location=device)


def _build_timexer_model(run_cfg: Dict, endo_cols: Sequence[str], exo_cols: Sequence[str],
                         target_cols: Sequence[str], device: torch.device,
                         use_revin: Optional[bool] = None,
                         residual_anchor: Optional[bool] = None,
                         batch_token: Optional[bool] = None,
                         batch_token_cols: Optional[Sequence[str]] = None) -> ReactorTimeXerModel:
    context_len = int(run_cfg.get("context_len", 360))
    horizon = int(run_cfg.get("horizon", 72))
    forecast_target_indices = [endo_cols.index(col) for col in target_cols]

    no_revin = bool(run_cfg.get("no_revin", False))
    if use_revin is not None:
        no_revin = not use_revin

    residual = bool(run_cfg.get("residual_anchor", False))
    if residual_anchor is not None:
        residual = bool(residual_anchor)

    use_batch_token = bool(run_cfg.get("batch_token", False))
    if batch_token is not None:
        use_batch_token = bool(batch_token)

    if batch_token_cols is None:
        batch_token_cols = run_cfg.get("batch_token_cols")
    batch_token_idx = None
    if use_batch_token and batch_token_cols:
        batch_token_idx = [exo_cols.index(col) for col in batch_token_cols if col in exo_cols]
        if not batch_token_idx:
            batch_token_idx = None

    model = ReactorTimeXerModel(
        c_endo=len(endo_cols),
        c_exo=len(exo_cols),
        context_len=context_len,
        horizon=horizon,
        n_forecast=len(target_cols),
        forecast_target_indices=forecast_target_indices,
        head=str(run_cfg.get("head", "student_t")),
        patch_size=int(run_cfg.get("patch_size", 12)),
        d_model=int(run_cfg.get("d_model", 128)),
        n_heads=int(run_cfg.get("n_heads", 4)),
        n_layers=int(run_cfg.get("n_layers", 2)),
        d_ff=int(run_cfg.get("d_ff", 512)),
        dropout=float(run_cfg.get("dropout", 0.1)),
        use_revin=not no_revin,
        residual_anchor=residual,
        use_batch_token=use_batch_token,
        batch_token_idx=batch_token_idx,
    ).to(device)

    return model


def _build_hybrid_model(model_cls, run_cfg: Dict, endo_cols: Sequence[str], exo_cols: Sequence[str],
                        target_cols: Sequence[str], device: torch.device,
                        use_ar_head: bool,
                        use_revin: Optional[bool] = None,
                        residual_anchor: Optional[bool] = None,
                        batch_token: Optional[bool] = None,
                        batch_token_cols: Optional[Sequence[str]] = None):
    context_len = int(run_cfg.get("context_len", 360))
    horizon = int(run_cfg.get("horizon", 72))
    forecast_target_indices = [endo_cols.index(col) for col in target_cols]

    no_revin = bool(run_cfg.get("no_revin", False))
    if use_revin is not None:
        no_revin = not use_revin

    residual = bool(run_cfg.get("residual_anchor", False))
    if residual_anchor is not None:
        residual = bool(residual_anchor)

    use_batch_token = bool(run_cfg.get("batch_token", False))
    if batch_token is not None:
        use_batch_token = bool(batch_token)

    if batch_token_cols is None:
        batch_token_cols = run_cfg.get("batch_token_cols")
    batch_token_idx = None
    if use_batch_token and batch_token_cols:
        batch_token_idx = [exo_cols.index(col) for col in batch_token_cols if col in exo_cols]
        if not batch_token_idx:
            batch_token_idx = None

    model = model_cls(
        c_endo=len(endo_cols),
        c_exo=len(exo_cols),
        context_len=context_len,
        horizon=horizon,
        n_forecast=len(target_cols),
        forecast_target_indices=forecast_target_indices,
        head=str(run_cfg.get("head", "student_t")),
        patch_size=int(run_cfg.get("patch_size", 12)),
        d_model=int(run_cfg.get("d_model", 128)),
        n_heads=int(run_cfg.get("n_heads", 4)),
        n_layers=int(run_cfg.get("n_layers", 2)),
        d_ff=int(run_cfg.get("d_ff", 512)),
        tcn_levels=int(run_cfg.get("tcn_levels", 3)),
        tcn_kernel=int(run_cfg.get("tcn_kernel", 3)),
        dropout=float(run_cfg.get("dropout", 0.1)),
        use_ar_head=use_ar_head,
        use_revin=not no_revin,
        residual_anchor=residual,
        use_batch_token=use_batch_token,
        batch_token_idx=batch_token_idx,
    ).to(device)

    return model


def _build_patchtst_model(run_cfg: Dict, endo_cols: Sequence[str], exo_cols: Sequence[str],
                          target_cols: Sequence[str], device: torch.device,
                          use_revin: Optional[bool] = None,
                          residual_anchor: Optional[bool] = None,
                          batch_token: Optional[bool] = None,
                          batch_token_cols: Optional[Sequence[str]] = None) -> ReactorPatchTSTModel:
    context_len = int(run_cfg.get("context_len", 180))
    horizon = int(run_cfg.get("horizon", 72))
    forecast_target_indices = [endo_cols.index(col) for col in target_cols]

    no_revin = bool(run_cfg.get("no_revin", False))
    if use_revin is not None:
        no_revin = not use_revin

    residual = bool(run_cfg.get("residual_anchor", False))
    if residual_anchor is not None:
        residual = bool(residual_anchor)

    use_batch_token = bool(run_cfg.get("batch_token", False))
    if batch_token is not None:
        use_batch_token = bool(batch_token)

    if batch_token_cols is None:
        batch_token_cols = run_cfg.get("batch_token_cols")
    batch_token_idx = None
    if use_batch_token and batch_token_cols:
        batch_token_idx = [exo_cols.index(col) for col in batch_token_cols if col in exo_cols]
        if not batch_token_idx:
            batch_token_idx = None

    patch_len = int(run_cfg.get("patch_len", run_cfg.get("patch_size", 16)))
    patch_stride = int(run_cfg.get("patch_stride", 8))

    model = ReactorPatchTSTModel(
        c_endo=len(endo_cols),
        c_exo=len(exo_cols),
        context_len=context_len,
        horizon=horizon,
        n_forecast=len(target_cols),
        forecast_target_indices=forecast_target_indices,
        d_model=int(run_cfg.get("d_model", 128)),
        n_heads=int(run_cfg.get("n_heads", 4)),
        n_layers=int(run_cfg.get("n_layers", 3)),
        patch_len=patch_len,
        patch_stride=patch_stride,
        dropout=float(run_cfg.get("dropout", 0.1)),
        head=str(run_cfg.get("head", "gaussian")),
        residual_anchor=residual,
        use_revin=not no_revin,
        use_batch_token=use_batch_token,
        batch_token_idx=batch_token_idx,
    ).to(device)

    return model


def _build_mambats_model(run_cfg: Dict, endo_cols: Sequence[str], exo_cols: Sequence[str],
                         target_cols: Sequence[str], device: torch.device,
                         use_revin: Optional[bool] = None,
                         residual_anchor: Optional[bool] = None,
                         batch_token: Optional[bool] = None,
                         batch_token_cols: Optional[Sequence[str]] = None):
    if ReactorMambaTSModel is None:
        raise RuntimeError(
            "MambaTS support is unavailable in this environment. "
            "Importing mambats_train_reactor_probabilistic.py failed."
        )

    context_len = int(run_cfg.get("context_len", 180))
    horizon = int(run_cfg.get("horizon", 18))
    forecast_target_indices = [endo_cols.index(col) for col in target_cols]

    no_revin = bool(run_cfg.get("no_revin", False))
    if use_revin is not None:
        no_revin = not use_revin

    residual = bool(run_cfg.get("residual_anchor", False))
    if residual_anchor is not None:
        residual = bool(residual_anchor)

    use_batch_token = bool(run_cfg.get("batch_token", False))
    if batch_token is not None:
        use_batch_token = bool(batch_token)

    if batch_token_cols is None:
        batch_token_cols = run_cfg.get("batch_token_cols")
    batch_token_idx = None
    if use_batch_token and batch_token_cols:
        batch_token_idx = [exo_cols.index(col) for col in batch_token_cols if col in exo_cols]
        if not batch_token_idx:
            batch_token_idx = None

    patch_len = int(run_cfg.get("patch_len", run_cfg.get("patch_size", 16)))
    patch_stride = int(run_cfg.get("patch_stride", 8))

    model = ReactorMambaTSModel(
        c_endo=len(endo_cols),
        c_exo=len(exo_cols),
        context_len=context_len,
        horizon=horizon,
        n_forecast=len(target_cols),
        forecast_target_indices=forecast_target_indices,
        d_model=int(run_cfg.get("d_model", 256)),
        n_layers=int(run_cfg.get("n_layers", 3)),
        n_heads=int(run_cfg.get("n_heads", 4)),
        patch_len=patch_len,
        patch_stride=patch_stride,
        d_state=int(run_cfg.get("d_state", 16)),
        d_conv=int(run_cfg.get("d_conv", 4)),
        expand=int(run_cfg.get("expand", 2)),
        dropout=float(run_cfg.get("dropout", 0.1)),
        head=str(run_cfg.get("head", "student_t")),
        residual_anchor=residual,
        use_revin=not no_revin,
        use_batch_token=use_batch_token,
        batch_token_idx=batch_token_idx,
    ).to(device)

    return model


def _build_tft_model(run_cfg: Dict, endo_cols: Sequence[str], exo_cols: Sequence[str],
                     target_cols: Sequence[str], device: torch.device,
                     use_revin: Optional[bool] = None,
                     residual_anchor: Optional[bool] = None,
                     batch_token: Optional[bool] = None,
                     batch_token_cols: Optional[Sequence[str]] = None):
    if ReactorTFTModel is None:
        raise RuntimeError(
            "TFT support is unavailable in this environment. "
            "Importing tft_train_reactor_probabilistic.py failed."
        )

    context_len = int(run_cfg.get("context_len", 180))
    horizon = int(run_cfg.get("horizon", 18))
    forecast_target_indices = [endo_cols.index(col) for col in target_cols]

    no_revin = bool(run_cfg.get("no_revin", False))
    if use_revin is not None:
        no_revin = not use_revin

    residual = bool(run_cfg.get("residual_anchor", False))
    if residual_anchor is not None:
        residual = bool(residual_anchor)

    use_batch_token = bool(run_cfg.get("batch_token", False))
    if batch_token is not None:
        use_batch_token = bool(batch_token)

    if batch_token_cols is None:
        batch_token_cols = run_cfg.get("batch_token_cols")
    batch_token_idx = None
    if use_batch_token and batch_token_cols:
        batch_token_idx = [exo_cols.index(col) for col in batch_token_cols if col in exo_cols]
        if not batch_token_idx:
            batch_token_idx = None

    model = ReactorTFTModel(
        c_endo=len(endo_cols),
        c_exo=len(exo_cols),
        context_len=context_len,
        horizon=horizon,
        n_forecast=len(target_cols),
        forecast_target_indices=forecast_target_indices,
        d_model=int(run_cfg.get("d_model", 128)),
        n_heads=int(run_cfg.get("n_heads", 4)),
        n_layers=int(run_cfg.get("n_layers", 2)),
        lstm_layers=int(run_cfg.get("lstm_layers", 1)),
        dropout=float(run_cfg.get("dropout", 0.1)),
        head=str(run_cfg.get("head", "student_t")),
        residual_anchor=residual,
        use_revin=not no_revin,
        use_batch_token=use_batch_token,
        batch_token_idx=batch_token_idx,
    ).to(device)

    return model


def _build_csdi_model(run_cfg: Dict, endo_cols: Sequence[str], exo_cols: Sequence[str],
                      target_cols: Sequence[str], device: torch.device,
                      use_revin: Optional[bool] = None,
                      residual_anchor: Optional[bool] = None,
                      batch_token: Optional[bool] = None,
                      batch_token_cols: Optional[Sequence[str]] = None):
    if ReactorCSDIModel is None:
        raise RuntimeError(
            "CSDI support is unavailable in this environment. "
            "Importing csdi_train_reactor_probabilistic.py failed."
        )

    context_len = int(run_cfg.get("context_len", 180))
    horizon = int(run_cfg.get("horizon", 18))
    forecast_target_indices = [endo_cols.index(col) for col in target_cols]

    no_revin = bool(run_cfg.get("no_revin", False))
    if use_revin is not None:
        no_revin = not use_revin

    residual = bool(run_cfg.get("residual_anchor", False))
    if residual_anchor is not None:
        residual = bool(residual_anchor)

    use_batch_token = bool(run_cfg.get("batch_token", False))
    if batch_token is not None:
        use_batch_token = bool(batch_token)

    if batch_token_cols is None:
        batch_token_cols = run_cfg.get("batch_token_cols")
    batch_token_idx = None
    if use_batch_token and batch_token_cols:
        batch_token_idx = [exo_cols.index(col) for col in batch_token_cols if col in exo_cols]
        if not batch_token_idx:
            batch_token_idx = None

    model = ReactorCSDIModel(
        c_endo=len(endo_cols),
        c_exo=len(exo_cols),
        context_len=context_len,
        horizon=horizon,
        n_forecast=len(target_cols),
        forecast_target_indices=forecast_target_indices,
        n_diffusion_steps=int(run_cfg.get("n_diffusion_steps", 100)),
        beta_schedule=str(run_cfg.get("beta_schedule", "cosine")),
        d_model=int(run_cfg.get("d_model", 64)),
        n_heads=int(run_cfg.get("n_heads", 4)),
        d_ff=int(run_cfg.get("d_ff", 256)),
        n_score_layers=int(run_cfg.get("n_score_layers", 4)),
        n_ctx_layers=int(run_cfg.get("n_ctx_layers", 2)),
        dropout=float(run_cfg.get("dropout", 0.1)),
        use_revin=not no_revin,
        revin_sigma_floor=float(run_cfg.get("revin_sigma_floor", 0.1)),
        residual_anchor=residual,
        use_batch_token=use_batch_token,
        batch_token_idx=batch_token_idx,
    ).to(device)

    return model


def _build_ncde_model(run_cfg: Dict, endo_cols: Sequence[str], exo_cols: Sequence[str],
                      target_cols: Sequence[str], device: torch.device,
                      use_revin: Optional[bool] = None,
                      residual_anchor: Optional[bool] = None,
                      batch_token: Optional[bool] = None,
                      batch_token_cols: Optional[Sequence[str]] = None):
    if ReactorCDEModel is None:
        raise RuntimeError(
            "NCDE support is unavailable in this environment. "
            "Importing ncde_train_reactor_probabilistic.py failed."
        )

    context_len = int(run_cfg.get("context_len", 180))
    horizon = int(run_cfg.get("horizon", 18))
    forecast_target_indices = [endo_cols.index(col) for col in target_cols]

    no_revin = bool(run_cfg.get("no_revin", False))
    if use_revin is not None:
        no_revin = not use_revin

    residual = bool(run_cfg.get("residual_anchor", False))
    if residual_anchor is not None:
        residual = bool(residual_anchor)

    use_batch_token = bool(run_cfg.get("batch_token", False))
    if batch_token is not None:
        use_batch_token = bool(batch_token)

    if batch_token_cols is None:
        batch_token_cols = run_cfg.get("batch_token_cols")
    batch_token_idx = None
    if use_batch_token and batch_token_cols:
        batch_token_idx = [exo_cols.index(col) for col in batch_token_cols if col in exo_cols]
        if not batch_token_idx:
            batch_token_idx = None

    model = ReactorCDEModel(
        c_endo=len(endo_cols),
        c_exo=len(exo_cols),
        context_len=context_len,
        horizon=horizon,
        n_forecast=len(target_cols),
        forecast_target_indices=forecast_target_indices,
        hidden_dim=int(run_cfg.get("hidden_dim", 64)),
        width=int(run_cfg.get("cde_width", 128)),
        head=str(run_cfg.get("head", "student_t")),
        dropout=float(run_cfg.get("dropout", 0.1)),
        residual_anchor=residual,
        solver=str(run_cfg.get("solver", "rk4")),
        adjoint=bool(run_cfg.get("adjoint", False)),
        use_revin=not no_revin,
        use_batch_token=use_batch_token,
        batch_token_idx=batch_token_idx,
    ).to(device)

    return model


def _build_timellm_model(run_cfg: Dict, endo_cols: Sequence[str], exo_cols: Sequence[str],
                         target_cols: Sequence[str], device: torch.device,
                         use_revin: Optional[bool] = None,
                         residual_anchor: Optional[bool] = None,
                         batch_token: Optional[bool] = None,
                         batch_token_cols: Optional[Sequence[str]] = None):
    if ReactorTimeLLMModel is None:
        raise RuntimeError(
            "TimeLLM support is unavailable in this environment. "
            "Importing timellm_train_reactor_probabilistic.py failed."
        )

    context_len = int(run_cfg.get("context_len", 180))
    horizon = int(run_cfg.get("horizon", 72))
    forecast_target_indices = [endo_cols.index(col) for col in target_cols]

    no_revin = bool(run_cfg.get("no_revin", False))
    if use_revin is not None:
        no_revin = not use_revin

    residual = bool(run_cfg.get("residual_anchor", False))
    if residual_anchor is not None:
        residual = bool(residual_anchor)

    use_batch_token = bool(run_cfg.get("batch_token", False))
    if batch_token is not None:
        use_batch_token = bool(batch_token)

    if batch_token_cols is None:
        batch_token_cols = run_cfg.get("batch_token_names") or run_cfg.get("batch_token_cols")
    batch_token_idx = None
    batch_token_names = None
    if use_batch_token and batch_token_cols:
        batch_token_idx = [exo_cols.index(col) for col in batch_token_cols if col in exo_cols]
        batch_token_names = [str(col) for col in batch_token_cols if col in exo_cols]
        if not batch_token_idx:
            batch_token_idx = None
            batch_token_names = None

    model = ReactorTimeLLMModel(
        c_endo=len(endo_cols),
        c_exo=len(exo_cols),
        context_len=context_len,
        horizon=horizon,
        n_forecast=len(target_cols),
        forecast_target_indices=forecast_target_indices,
        patch_size=int(run_cfg.get("patch_size", 16)),
        n_llm_layers=int(run_cfg.get("n_llm_layers", 6)),
        n_heads=int(run_cfg.get("n_heads", 8)),
        n_source=int(run_cfg.get("n_source", 512)),
        dropout=float(run_cfg.get("dropout", 0.1)),
        head=str(run_cfg.get("head", "student_t")),
        use_revin=not no_revin,
        use_batch_token=use_batch_token,
        batch_token_idx=batch_token_idx,
        batch_token_names=batch_token_names,
        llm_model=str(run_cfg.get("llm_model", "gpt2")),
        prompt_text=str(run_cfg.get("prompt_text", "")),
        use_stats_prompt=bool(run_cfg.get("stats_prompt", False) or run_cfg.get("use_stats_prompt", False)),
        residual_anchor=residual,
    ).to(device)

    return model


def _build_itransformer_model(run_cfg: Dict, endo_cols: Sequence[str], exo_cols: Sequence[str],
                              target_cols: Sequence[str], device: torch.device,
                              use_revin: Optional[bool] = None,
                              residual_anchor: Optional[bool] = None,
                              batch_token: Optional[bool] = None,
                              batch_token_cols: Optional[Sequence[str]] = None) -> ReactorITransformerModel:
    context_len = int(run_cfg.get("context_len", 360))
    horizon = int(run_cfg.get("horizon", 72))
    forecast_target_indices = [endo_cols.index(col) for col in target_cols]

    no_revin = bool(run_cfg.get("no_revin", False))
    if use_revin is not None:
        no_revin = not use_revin

    residual = bool(run_cfg.get("residual_anchor", False))
    if residual_anchor is not None:
        residual = bool(residual_anchor)

    use_batch_token = bool(run_cfg.get("batch_token", False))
    if batch_token is not None:
        use_batch_token = bool(batch_token)

    raw_target_as_exo = bool(run_cfg.get("raw_target_as_exo", False))
    if batch_token_cols is None:
        batch_token_cols = run_cfg.get("batch_token_cols")
    batch_token_idx = None
    if use_batch_token and batch_token_cols:
        batch_token_idx = [exo_cols.index(col) for col in batch_token_cols if col in exo_cols]
        if not batch_token_idx:
            batch_token_idx = None

    model = ReactorITransformerModel(
        c_endo=len(endo_cols),
        c_exo=len(exo_cols),
        context_len=context_len,
        horizon=horizon,
        n_forecast=len(target_cols),
        forecast_target_indices=forecast_target_indices,
        d_model=int(run_cfg.get("d_model", 128)),
        n_heads=int(run_cfg.get("n_heads", 4)),
        n_layers=int(run_cfg.get("n_layers", 3)),
        dropout=float(run_cfg.get("dropout", 0.1)),
        head=str(run_cfg.get("head", "student_t")),
        residual_anchor=residual,
        use_revin=not no_revin,
        use_batch_token=use_batch_token,
        batch_token_idx=batch_token_idx,
        raw_target_as_exo=raw_target_as_exo,
    ).to(device)

    return model


def _load_model(run_cfg: Dict, endo_cols: Sequence[str], exo_cols: Sequence[str],
                target_cols: Sequence[str], checkpoint: Path, device: torch.device,
                use_revin: Optional[bool] = None,
                residual_anchor: Optional[bool] = None,
                batch_token: Optional[bool] = None,
                batch_token_cols: Optional[Sequence[str]] = None,
                model_family: str = "auto"):
    state = _load_checkpoint_state(checkpoint, device)
    family = _infer_model_family(state, run_cfg) if model_family == "auto" else model_family.lower()
    inferred_use_revin, inferred_use_batch_token = _infer_common_state_flags(state)
    use_revin = inferred_use_revin if use_revin is None else use_revin
    batch_token = inferred_use_batch_token if batch_token is None else batch_token

    if family == "timexer":
        model = _build_timexer_model(
            run_cfg,
            endo_cols,
            exo_cols,
            target_cols,
            device,
            use_revin=use_revin,
            residual_anchor=residual_anchor,
            batch_token=batch_token,
            batch_token_cols=batch_token_cols,
        )
    elif family == "timellm":
        model = _build_timellm_model(
            run_cfg,
            endo_cols,
            exo_cols,
            target_cols,
            device,
            use_revin=use_revin,
            residual_anchor=residual_anchor,
            batch_token=batch_token,
            batch_token_cols=batch_token_cols,
        )
    elif family == "patchtst":
        model = _build_patchtst_model(
            run_cfg,
            endo_cols,
            exo_cols,
            target_cols,
            device,
            use_revin=use_revin,
            residual_anchor=residual_anchor,
            batch_token=batch_token,
            batch_token_cols=batch_token_cols,
        )
    elif family == "itransformer":
        model = _build_itransformer_model(
            run_cfg,
            endo_cols,
            exo_cols,
            target_cols,
            device,
            use_revin=use_revin,
            residual_anchor=residual_anchor,
            batch_token=batch_token,
            batch_token_cols=batch_token_cols,
        )
    elif family == "hybrid":
        use_ar_head = _infer_hybrid_head_mode(state, run_cfg)
        last_error = None
        for model_cls in HYBRID_MODEL_CLASSES:
            try:
                model = _build_hybrid_model(
                    model_cls,
                    run_cfg,
                    endo_cols,
                    exo_cols,
                    target_cols,
                    device,
                    use_ar_head=use_ar_head,
                    use_revin=use_revin,
                    residual_anchor=residual_anchor,
                    batch_token=batch_token,
                    batch_token_cols=batch_token_cols,
                )
                model.load_state_dict(state)
                model.eval()
                model._forecast_family = "hybrid"
                print(f"Detected checkpoint family: hybrid ({model_cls.__module__}.{model_cls.__name__})")
                return model
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(
            "Could not load the hybrid checkpoint with any known reactor model class. "
            f"Last error: {last_error}"
        )
    elif family == "mamba":
        model = _build_mambats_model(
            run_cfg,
            endo_cols,
            exo_cols,
            target_cols,
            device,
            use_revin=use_revin,
            residual_anchor=residual_anchor,
            batch_token=batch_token,
            batch_token_cols=batch_token_cols,
        )
    elif family == "tft":
        model = _build_tft_model(
            run_cfg,
            endo_cols,
            exo_cols,
            target_cols,
            device,
            use_revin=use_revin,
            residual_anchor=residual_anchor,
            batch_token=batch_token,
            batch_token_cols=batch_token_cols,
        )
    elif family == "csdi":
        model = _build_csdi_model(
            run_cfg,
            endo_cols,
            exo_cols,
            target_cols,
            device,
            use_revin=use_revin,
            residual_anchor=residual_anchor,
            batch_token=batch_token,
            batch_token_cols=batch_token_cols,
        )
    elif family == "ncde":
        model = _build_ncde_model(
            run_cfg,
            endo_cols,
            exo_cols,
            target_cols,
            device,
            use_revin=use_revin,
            residual_anchor=residual_anchor,
            batch_token=batch_token,
            batch_token_cols=batch_token_cols,
        )
    else:
        raise ValueError(f"Unknown model_family {model_family!r}. Use auto, timellm, timexer, patchtst, itransformer, hybrid, mamba, tft, csdi, or ncde.")

    model.load_state_dict(state)
    model.eval()
    model._forecast_family = family
    print(f"Detected checkpoint family: {family}")
    return model


def _prepare_experiment_frame(exp_tag: str, data_paths: Sequence[str], cheatsheet: str,
                              log_cols: Sequence[str]):
    df = load_experiment(list(data_paths), cheatsheet, [exp_tag])
    if df.empty:
        raise ValueError(f"No rows found for experiment {exp_tag!r}.")

    df = df.copy()
    df["time_since_valid"] = compute_time_since_valid(df)

    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].values)

    return df


def _render_windows(
    model: ReactorTimeXerModel,
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
    window_starts: Sequence[int],
    head: str,
    target_channel_idx: int,
    val_frac: float,
    test_frac: float,
    context_show: int = 60,
) -> None:
    device = next(model.parameters()).device
    total_len = len(df)
    n_test = int(total_len * test_frac)
    n_val = int(total_len * val_frac)
    n_train = total_len - n_val - n_test

    flags = df["flag"].values.astype(int)
    endo_te_raw = df[target_col].values[n_train + n_val :]
    endo_te_sc = apply_scalers(
        df[endo_cols].values[n_train + n_val :], sc_endo, endo_cols, scale_mode
    )
    exo_te_sc = apply_scalers(
        df[exo_cols].values[n_train + n_val :], sc_exo, exo_cols, scale_mode
    )
    flags_te = flags[n_train + n_val :]

    candidates = _candidate_starts(flags_te, context_len, horizon)
    candidate_set = set(candidates)
    invalid = [s for s in window_starts if s not in candidate_set]
    if invalid:
        raise ValueError(
            f"Selected window(s) for {target_col!r} are not valid in the test slice: {invalid}"
        )

    context_show = max(1, min(int(context_show), context_len))
    ctx_x = np.arange(-context_show, 0)
    fore_x = np.arange(0, horizon)

    fig, axes = plt.subplots(
        len(window_starts), 1,
        figsize=(11, 4 * len(window_starts)),
        squeeze=False,
    )

    log_cols_set = set(log_cols)
    forecast_color = _forecast_color_for_family(getattr(model, "_forecast_family", "timexer"))

    for row, start in enumerate(window_starts):
        x_endo = torch.tensor(
            endo_te_sc[start : start + context_len].T,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        x_exo = torch.tensor(
            exo_te_sc[start : start + context_len].T,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        if getattr(model, "_forecast_family", None) == "csdi":
            with torch.no_grad():
                samples = model.sample(x_endo, x_exo, n_samples=100)  # [B, S, F, H]
            mean_sc = samples.mean(dim=1)[0, target_channel_idx].detach().cpu().numpy()
            std_sc  = samples.std(dim=1)[0, target_channel_idx].detach().cpu().numpy()
            z_1s, z_95 = 1.0, 1.96
        else:
            with torch.no_grad():
                out = model(x_endo, x_exo)

            mean_sc = out[0][0, target_channel_idx].detach().cpu().numpy()
            scale_sc = out[1][0, target_channel_idx].detach().cpu().numpy()

            if head == "gaussian":
                std_sc = np.exp(0.5 * np.clip(scale_sc, LOG_VAR_MIN, LOG_VAR_MAX))
                z_1s, z_95 = 1.0, 1.96
            else:
                from scipy.stats import t as _student_t

                std_sc = np.exp(np.clip(scale_sc, LOG_SIG_MIN, LOG_SIG_MAX))
                log_nu_sc = out[2][0, target_channel_idx].detach().cpu().numpy()
                nu_med = float(
                    np.median(np.log1p(np.exp(np.clip(log_nu_sc, -20.0, 20.0))) + 2.0)
                )
                z_1s = float(_student_t.ppf(0.84, df=nu_med))
            z_95 = float(_student_t.ppf(0.975, df=nu_med))

        inv = lambda arr: inverse_target(arr, sc_endo, target_col, endo_cols, scale_mode, log_cols_set)
        mean_orig = inv(mean_sc)
        upper_1s = inv(mean_sc + z_1s * std_sc)
        lower_1s = inv(mean_sc - z_1s * std_sc)
        upper_95 = inv(mean_sc + z_95 * std_sc)
        lower_95 = inv(mean_sc - z_95 * std_sc)

        ctx_sl = endo_te_raw[start + context_len - context_show : start + context_len]
        actual_sl = endo_te_raw[start + context_len : start + context_len + horizon]

        if target_col in log_cols_set:
            ctx_raw = np.expm1(ctx_sl)
            actual_raw = np.expm1(actual_sl)
        else:
            ctx_raw = ctx_sl
            actual_raw = actual_sl

        ax = axes[row][0]
        ax.plot(ctx_x, ctx_raw, color="#888888", lw=1.2, alpha=0.7, label="Context")
        ax.fill_between(fore_x, lower_95, upper_95, color=forecast_color, alpha=0.15, label="95% PI")
        ax.fill_between(fore_x, lower_1s, upper_1s, color=forecast_color, alpha=0.30, label="68% PI")
        ax.plot(fore_x, mean_orig, color=forecast_color, lw=2.0, label="Mean forecast")
        ax.plot(fore_x, actual_raw, color="#1f77b4", lw=2.0, label=f"Actual {target_col.upper()}")
        ax.axvline(0, color="black", lw=0.8, ls=":")

        mae = float(np.mean(np.abs(mean_orig - actual_raw)))
        rmse = float(np.sqrt(np.mean((mean_orig - actual_raw) ** 2)))
        ax.set_title(
            f"Window {start} | {target_col.upper()} | MAE={mae:.3f} | RMSE={rmse:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("Steps from forecast origin", fontsize=8)

        ticks = np.concatenate(
            [
                np.arange(-context_show, 0, max(1, context_show // 4)),
                np.arange(0, horizon + 1, max(1, horizon // 6)),
            ]
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=7)
        if row == 0:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"{target_col.upper()} selected reactor windows | exp={out_path.stem}",
        fontsize=11,
        y=1.01,
    )
    fig.supylabel("Concentration (mg/L)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Render selected reactor forecast windows from an existing TimeXer checkpoint.",
    )

    parser.add_argument("--run_dir", type=Path, required=True,
                        help="Directory containing results.json, best_model.pt and scalers.")
    parser.add_argument("--results_json", type=Path, default=None,
                        help="Optional explicit results.json path. Defaults to run_dir/results.json.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional explicit checkpoint path. Defaults to run_dir/best_model.pt.")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Directory where the selected-window figures will be saved.")

    parser.add_argument("--window", action="append", default=None,
                        help="Window selection in the form EXPERIMENT:idx1,idx2,... .")
    parser.add_argument("--context_show", type=int, default=60,
                        help="Number of context steps to show before the forecast origin.")

    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--test_frac", type=float, default=None)
    parser.add_argument("--head", choices=("gaussian", "student_t"), default=None)
    parser.add_argument("--model_family", choices=("auto", "timellm", "timexer", "itransformer", "hybrid", "patchtst"), default="auto",
                        help="Force a model family or let the script infer it from the checkpoint.")
    parser.add_argument("--no_revin", action="store_true", help="Disable RevIN if the run used it off.")
    parser.add_argument("--residual_anchor", action="store_true",
                        help="Force residual anchoring on when rebuilding the model.")
    parser.add_argument("--batch_token", action="store_true",
                        help="Force batch-token support on when rebuilding the model.")
    parser.add_argument("--batch_token_cols", nargs="+", default=None,
                        help="Batch token columns to map from the exogenous inputs.")

    parser.add_argument("--data_paths", nargs="+", default=None,
                        help="Optional override for the raw CSV data paths.")
    parser.add_argument("--cheatsheet", default=None,
                        help="Optional override for the cheatsheet path.")
    parser.add_argument("--endogenous", nargs="+", default=None)
    parser.add_argument("--exogenous", nargs="+", default=None)
    parser.add_argument("--target", nargs="+", default=None)
    parser.add_argument("--log_cols", nargs="+", default=None)

    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if args.results_json is None:
        results_json = run_dir / "results.json"
    else:
        results_json = args.results_json.expanduser().resolve()

    if args.checkpoint is None:
        checkpoint = run_dir / "best_model.pt"
    else:
        checkpoint = args.checkpoint.expanduser().resolve()

    if args.output_dir is None:
        output_dir = run_dir / "selected_windows"
    else:
        output_dir = args.output_dir.expanduser().resolve()

    if not results_json.exists():
        raise FileNotFoundError(f"results.json not found: {results_json}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    results = _load_json(results_json)
    run_cfg = _merge_run_config(results)

    if args.head is not None:
        run_cfg["head"] = args.head
    if args.val_frac is not None:
        run_cfg["val_frac"] = float(args.val_frac)
    if args.test_frac is not None:
        run_cfg["test_frac"] = float(args.test_frac)
    if args.endogenous is not None:
        run_cfg["endogenous"] = list(args.endogenous)
    if args.exogenous is not None:
        run_cfg["exogenous"] = list(args.exogenous)
    if args.target is not None:
        run_cfg["target"] = list(args.target) if len(args.target) > 1 else args.target[0]
    if args.log_cols is not None:
        run_cfg["log_cols"] = list(args.log_cols)

    data_paths = args.data_paths or run_cfg.get("data_paths")
    cheatsheet = args.cheatsheet or run_cfg.get("cheatsheet")
    if not data_paths:
        raise ValueError("No data_paths found in results.json and none were passed on the command line.")
    if not cheatsheet:
        raise ValueError("No cheatsheet found in results.json and none were passed on the command line.")

    data_paths = [str(_resolve_path(path, run_dir)) for path in data_paths]
    cheatsheet_path = _resolve_path(cheatsheet, run_dir)

    endo_cols = _as_list(run_cfg.get("endogenous"))
    exo_cols = _as_list(run_cfg.get("exogenous"))
    if not endo_cols:
        raise ValueError("The run configuration does not contain endogenous columns.")
    if not exo_cols:
        raise ValueError("The run configuration does not contain exogenous columns.")

    target_cols = _normalise_targets(run_cfg.get("target"), endo_cols)
    for target_col in target_cols:
        if target_col not in endo_cols:
            raise ValueError(f"Target {target_col!r} is not in endogenous columns {endo_cols}.")

    log_cols = _as_list(run_cfg.get("log_cols"))
    if not log_cols:
        log_cols = list(endo_cols)

    window_specs = args.window if args.window else list(DEFAULT_WINDOW_SPECS)
    grouped_windows = _parse_window_specs(window_specs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Results: {results_json}")
    print(f"Output dir: {output_dir}")
    print(f"Targets: {target_cols}")

    sc_endo, sc_exo = _fit_training_scalers(
        run_cfg=run_cfg,
        data_paths=data_paths,
        cheatsheet=str(cheatsheet_path),
        endo_cols=endo_cols,
        exo_cols=exo_cols,
        scale_mode=str(run_cfg.get("scale_mode", "per_var")),
        scaling=str(run_cfg.get("scaling", "standard")),
        log_cols=log_cols,
    )
    model = _load_model(
        run_cfg,
        endo_cols,
        exo_cols,
        target_cols,
        checkpoint,
        device,
        use_revin=False if args.no_revin else None,
        residual_anchor=True if args.residual_anchor else None,
        batch_token=True if args.batch_token else None,
        batch_token_cols=args.batch_token_cols,
        model_family=args.model_family,
    )

    scale_mode = str(run_cfg.get("scale_mode", "per_var"))
    val_frac = float(run_cfg.get("val_frac", 0.1))
    test_frac = float(run_cfg.get("test_frac", 0.1))
    head = str(run_cfg.get("head", "student_t"))
    context_len = int(run_cfg.get("context_len", 360))
    horizon = int(run_cfg.get("horizon", 72))

    saved_paths: List[Path] = []
    for exp_tag, window_starts in grouped_windows.items():
        print(f"\nExperiment: {exp_tag}  windows={window_starts}")
        df = _prepare_experiment_frame(exp_tag, data_paths, str(cheatsheet_path), log_cols)

        for target_idx, target_col in enumerate(target_cols):
            suffix = f"{exp_tag}_{target_col}"
            out_path = output_dir / f"{suffix}_selected_windows.png"
            _render_windows(
                model=model,
                sc_endo=sc_endo,
                sc_exo=sc_exo,
                df=df,
                endo_cols=endo_cols,
                exo_cols=exo_cols,
                scale_mode=scale_mode,
                context_len=context_len,
                horizon=horizon,
                out_path=out_path,
                target_col=target_col,
                log_cols=log_cols,
                window_starts=window_starts,
                head=head,
                target_channel_idx=target_idx,
                val_frac=val_frac,
                test_frac=test_frac,
                context_show=args.context_show,
            )
            saved_paths.append(out_path)

    print("\nSaved figures:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()