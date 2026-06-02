"""
csdi_train_reactor_probabilistic.py

Conditional Score-based Diffusion model for Imputation (CSDI) adapted for
reactor batch time-series forecasting.

Architecture
------------
  Context encoder
    x_endo [B, C_e, T] + x_exo [B, C_x, T]
      → concat → project to d_model → positional embedding
      → transformer layers → condition tokens [B, T, d_model]

  Score network  ε_θ(y_k, k, condition) → predicted noise
    y_noisy [B, n_forecast, H]
      → project to [B, n_forecast, H, d_model]
      → + diffusion step embedding (sinusoidal)
      → n_layers of ScoreBlock:
            temporal self-attention  (over H steps, per forecast variable)
            feature  self-attention  (over n_forecast vars, per time step)
            condition cross-attention (queries from target, keys/values from context)
            FFN
      → project back to [B, n_forecast, H]  (predicted noise)

  Diffusion
    Forward:  y_k = √ᾱ_k · y_0 + √(1−ᾱ_k) · ε
    Reverse:  DDPM update, conditioned on context
    Training: MSE(ε, ε_θ)
    Inference: full reverse chain → N samples → mean ± std

Same data pipeline, evaluation, and CLI as the transformer / Neural CDE scripts.
No extra dependencies beyond the standard ML stack.

Reference: Tashiro et al. "CSDI: Conditional Score-based Diffusion Models
           for Probabilistic Time Series Imputation." NeurIPS 2021.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ── Column layout ──────────────────────────────────────────────────────────
ALL_DATA_COLS = [
 #fill with data column titles
]


def resolve_columns(endogenous: List[str],
                    exogenous:  List[str]) -> Tuple[List[str], List[str]]:
    if not endogenous:
        raise ValueError("--endogenous must list at least one variable.")
    bad = [c for c in endogenous if c not in ALL_DATA_COLS]
    if bad:
        raise ValueError(f"Unknown --endogenous: {bad}")
    bad = [c for c in exogenous if c not in ALL_DATA_COLS]
    if bad:
        raise ValueError(f"Unknown --exogenous: {bad}")
    endo_set  = set(endogenous)
    exo_set   = set(exogenous)
    leftovers = [c for c in ALL_DATA_COLS if c not in endo_set and c not in exo_set]
    return list(endogenous), list(exogenous) + leftovers + ['time_since_valid']


# ── Cheatsheet parsing ─────────────────────────────────────────────────────
MONTHS = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'okt': 10, 'maj': 5,
}
_DT_PAT   = re.compile(r'^\s*(?:c\.)?\s*(\d{1,2})\s*([A-Za-z]{3})[.\-]\s*(\d{1,2}):(\d{2})\s*$', re.I)
_DATE_PAT = re.compile(r'^\s*(?:c\.)?\s*(\d{1,2})\s*([A-Za-z]{3})\s*$', re.I)
_TIME_PAT = re.compile(r'^\s*(\d{1,2}):(\d{2})\s*$')
_EXP_PAT  = re.compile(r'^\s*(KAU\d+)\s*:?\s*$', re.I)
_YEAR_PAT = re.compile(r'^\s*(\d{4})\s*$')


@dataclass
class Experiment:
    name: str
    start: Optional[datetime] = None
    end:   Optional[datetime] = None
    inoc_amount: float = 0.0
    inoc_conc:   float = 0.0
    excludes: List[Tuple[datetime, datetime]] = field(default_factory=list)


def _parse_dt(tok: str, year: int) -> datetime:
    m = _DT_PAT.match(tok)
    if not m: raise ValueError(f"Bad datetime token: {tok!r}")
    d, mon, h, mi = m.groups()
    mo = MONTHS.get(mon.lower())
    if mo is None: raise ValueError(f"Unknown month: {mon}")
    return datetime(year, mo, int(d), int(h), int(mi))


def _adj_year(dt: datetime, ref: datetime) -> datetime:
    if dt < ref and (ref - dt).days > 200:
        return dt.replace(year=dt.year + 1)
    return dt


def _parse_exclude(text: str, year: int, ref: Optional[datetime]) -> Tuple[datetime, datetime]:
    parts = [p.strip() for p in text.split('-')]
    if (len(parts) == 3 and _DATE_PAT.match(parts[0])
            and _TIME_PAT.match(parts[1]) and _TIME_PAT.match(parts[2])):
        dm = _DATE_PAT.match(parts[0]); d, mon = dm.groups()
        base = datetime(year, MONTHS[mon.lower()], int(d))
        if ref: base = _adj_year(base, ref)
        h1, m1 = _TIME_PAT.match(parts[1]).groups()
        h2, m2 = _TIME_PAT.match(parts[2]).groups()
        return (base.replace(hour=int(h1), minute=int(m1)),
                base.replace(hour=int(h2), minute=int(m2)))
    left, right = text.split('-', 1)
    left, right = left.strip(), right.strip()
    if _DT_PAT.match(left):
        s = _parse_dt(left, year)
    elif _DATE_PAT.match(left):
        dm = _DATE_PAT.match(left); d, mon = dm.groups()
        s = datetime(year, MONTHS[mon.lower()], int(d))
    else:
        raise ValueError(f"Bad exclude start: {left!r}")
    if ref: s = _adj_year(s, ref)
    if _DT_PAT.match(right):
        e = _parse_dt(right, s.year); e = _adj_year(e, s)
    elif _TIME_PAT.match(right):
        tm = _TIME_PAT.match(right)
        e = s.replace(hour=int(tm.group(1)), minute=int(tm.group(2)))
    else:
        raise ValueError(f"Bad exclude end: {right!r}")
    return s, e


def parse_cheatsheet(path: str) -> Dict[str, Experiment]:
    exps: Dict[str, Experiment] = {}
    cur: Optional[Experiment] = None
    cur_year: Optional[int] = None
    in_excl = False
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            if _YEAR_PAT.match(line):
                cur_year = int(_YEAR_PAT.match(line).group(1)); in_excl = False; continue
            if _EXP_PAT.match(line):
                cur = Experiment(name=_EXP_PAT.match(line).group(1).upper())
                exps[cur.name] = cur; in_excl = False; continue
            if cur is None: continue
            low = line.lower()
            if low.startswith('start:'):
                cur.start = _parse_dt(line.split(':', 1)[1].strip(), cur_year); in_excl = False
            elif low.startswith('end:'):
                e = _parse_dt(line.split(':', 1)[1].strip(), cur.start.year)
                cur.end = _adj_year(e, cur.start); in_excl = False
            elif low.startswith('exclude:'):
                cur.excludes.append(
                    _parse_exclude(line.split(':', 1)[1].strip(), cur.start.year, cur.start))
                in_excl = True
            elif in_excl and not re.match(r'^inocc\.', low):
                try:
                    cur.excludes.append(_parse_exclude(line, cur.start.year, cur.start))
                except ValueError:
                    in_excl = False
            elif re.match(r'^inocc\.amount\s*:', low):
                cur.inoc_amount = float(re.split(r':', low, 1)[1].strip()); in_excl = False
            elif re.match(r'^inocc\.conc\s*:', low):
                cur.inoc_conc = float(re.split(r':', low, 1)[1].strip()); in_excl = False
    return exps


# ── Data loading ───────────────────────────────────────────────────────────

def load_experiment(data_paths: List[str], cheatsheet: str,
                    exp_tags: List[str]) -> pd.DataFrame:
    exps = parse_cheatsheet(cheatsheet)
    parts = [pd.read_csv(p, sep=';', decimal=',', low_memory=False) for p in data_paths]
    raw_df = pd.concat(parts, ignore_index=True)
    raw_df['time'] = pd.to_datetime(raw_df['time'], dayfirst=True, errors='coerce')
    raw_df = raw_df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)

    exp_frames = []
    for tag_raw in exp_tags:
        tag = tag_raw.upper()
        if tag not in exps:
            raise ValueError(f"Experiment {tag} not in cheatsheet. Available: {list(exps.keys())}")
        exp  = exps[tag]
        mask = (raw_df['time'] >= exp.start) & (raw_df['time'] <= exp.end)
        df_e = raw_df[mask].copy().reset_index(drop=True)
        if len(df_e) == 0:
            raise ValueError(f"No data for {tag} between {exp.start} and {exp.end}")
        print(f"  Experiment {tag}: {len(df_e):,} rows  [{exp.start} → {exp.end}]")
        for excl_s, excl_e in exp.excludes:
            df_e.loc[(df_e['time'] >= excl_s) & (df_e['time'] <= excl_e), 'flag'] = 1
        exp_frames.append(df_e)

    df = pd.concat(exp_frames, ignore_index=True)
    for col in ALL_DATA_COLS:
        if col not in df.columns:
            print(f"  WARNING: column '{col}' missing; filling with 0")
            df[col] = 0.0
    for col in ALL_DATA_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).clip(lower=0)
    return df


def compute_time_since_valid(df: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(df), dtype=np.float32)
    last_valid_time = None
    for i, (ts, flag) in enumerate(zip(df['time'], df['flag'])):
        if flag == 0:
            if last_valid_time is not None:
                result[i] = float((ts - last_valid_time).total_seconds() / 60.0)
            last_valid_time = ts
    return result


def get_valid_segments(flags: np.ndarray, times=None,
                       segment_gap_minutes: float = 60.0) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    in_seg = False; s = 0
    for i, f in enumerate(flags):
        if f == 0 and not in_seg:
            s = i; in_seg = True
        elif f != 0 and in_seg:
            segs.append((s, i - 1)); in_seg = False
    if in_seg:
        segs.append((s, len(flags) - 1))
    if times is None or segment_gap_minutes <= 0:
        return segs
    result: List[Tuple[int, int]] = []
    for seg_s, seg_e in segs:
        sub_s = seg_s
        for i in range(seg_s + 1, seg_e + 1):
            try:
                gap_min = (pd.Timestamp(times[i]) - pd.Timestamp(times[i-1])).total_seconds() / 60.0
            except Exception:
                gap_min = 0.0
            if gap_min > segment_gap_minutes:
                if i - 1 >= sub_s: result.append((sub_s, i - 1))
                sub_s = i
        result.append((sub_s, seg_e))
    return result


# ── Scaling ────────────────────────────────────────────────────────────────
Scalers = Union[StandardScaler, RobustScaler, Dict[str, Union[StandardScaler, RobustScaler]]]


def make_scaler(scaling: str):
    return RobustScaler() if scaling == 'robust' else StandardScaler()


def fit_scalers(data: np.ndarray, cols: List[str], scaling: str, scale_mode: str) -> Scalers:
    if scale_mode == 'per_var':
        d: Dict[str, any] = {}
        for i, col in enumerate(cols):
            sc = make_scaler(scaling); sc.fit(data[:, i:i+1]); d[col] = sc
        return d
    sc = make_scaler(scaling); sc.fit(data); return sc


def apply_scalers(data: np.ndarray, scalers: Scalers, cols: List[str], scale_mode: str) -> np.ndarray:
    if scale_mode == 'per_var':
        out = np.empty_like(data, dtype=np.float32)
        for i, col in enumerate(cols):
            out[:, i] = scalers[col].transform(data[:, i:i+1]).ravel()
        return out
    return scalers.transform(data).astype(np.float32)


def inverse_target(arr: np.ndarray, scalers: Scalers, target_col: str,
                   cols: List[str], scale_mode: str,
                   log_cols: Optional[Set[str]] = None) -> np.ndarray:
    if scale_mode == 'per_var':
        res = scalers[target_col].inverse_transform(arr.reshape(-1, 1)).ravel()
    else:
        idx   = cols.index(target_col)
        dummy = np.zeros((len(arr), len(cols)), dtype=np.float64)
        dummy[:, idx] = arr
        res   = scalers.inverse_transform(dummy)[:, idx]
    if log_cols and target_col in log_cols:
        return np.expm1(res)
    return res


# ── Dataset ────────────────────────────────────────────────────────────────

class ReactorDataset(Dataset):
    def __init__(self, endo: np.ndarray, exo: np.ndarray,
                 flags: np.ndarray, context_len: int, horizon: int,
                 stride: int = 1, segment_isolated: bool = False,
                 gap_threshold: float = 4.0,
                 time_since_valid: Optional[np.ndarray] = None,
                 times=None,
                 segment_gap_minutes: float = 60.0,
                 min_segment_rows: int = 0,
                 target_indices: Optional[List[int]] = None):
        self.endo           = torch.from_numpy(endo.astype(np.float32))
        self.exo            = torch.from_numpy(exo .astype(np.float32))
        self.ctx            = context_len
        self.hor            = horizon
        self.target_indices = list(range(endo.shape[1])) if target_indices is None else list(target_indices)
        total_win = context_len + horizon

        if segment_isolated:
            segs = get_valid_segments(flags, times=times,
                                      segment_gap_minutes=segment_gap_minutes)
            self.indices = []; n_used = 0
            for seg_s, seg_e in segs:
                seg_len = seg_e - seg_s + 1
                if seg_len < max(total_win, min_segment_rows if min_segment_rows > 0 else total_win):
                    continue
                n_used += 1
                for i in range(0, seg_len - total_win + 1, stride):
                    self.indices.append(seg_s + i)
            print(f"  Dataset: {len(self.indices):,} windows "
                  f"from {n_used}/{len(segs)} segments "
                  f"(ctx={context_len}, hor={horizon}, stride={stride})")
        else:
            valid_idx    = np.where(flags == 0)[0]
            self.indices = [i for i in valid_idx if i + total_win <= len(endo)]
            if time_since_valid is not None and gap_threshold > 0:
                self.indices = [i for i in self.indices
                                if not (time_since_valid[i: i + total_win] > gap_threshold).any()]
            print(f"  Dataset: {len(self.indices):,} windows "
                  f"(ctx={context_len}, hor={horizon}, stride={stride})")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (self.endo[i         : i + self.ctx          ].T,
                self.exo [i         : i + self.ctx          ].T,
                self.endo[i + self.ctx : i + self.ctx + self.hor][:, self.target_indices].T)


def build_dataloaders(df: pd.DataFrame,
                      endo_cols: List[str], exo_cols: List[str],
                      context_len: int, horizon: int, batch_size: int,
                      stride: int, val_frac: float, test_frac: float,
                      scaling: str, scale_mode: str,
                      segment_isolated: bool, gap_threshold: float,
                      segment_gap_minutes: float = 60.0,
                      min_segment_rows: int = 0,
                      forecast_target_indices: Optional[List[int]] = None,
                      spike_oversample: float = 0.0,
                      spike_abs_weight: float = 1.0,
                      save_scalers_to: Optional[str] = None,
                      per_exp_sizes: Optional[List[int]] = None):

    flags = df['flag'].values.astype(int)
    tsv   = df['time_since_valid'].values.astype(np.float32)
    times = df['time'].values

    endo_arr = df[endo_cols].values.astype(np.float64)
    exo_arr  = df[exo_cols ].values.astype(np.float64)

    def _make_ds(endo_s, exo_s, flags_s, tsv_s, times_s, split_name):
        return ReactorDataset(
            apply_scalers(endo_s, sc_endo, endo_cols, scale_mode),
            apply_scalers(exo_s,  sc_exo,  exo_cols,  scale_mode),
            flags_s, context_len, horizon,
            stride=stride if split_name == 'train' else 1,
            segment_isolated=segment_isolated,
            gap_threshold=gap_threshold,
            time_since_valid=tsv_s,
            times=times_s,
            segment_gap_minutes=segment_gap_minutes,
            min_segment_rows=min_segment_rows,
            target_indices=forecast_target_indices,
        )

    if per_exp_sizes is not None:
        tr_idx, va_idx, te_idx = [], [], []
        pos = 0
        for sz in per_exp_sizes:
            n_te = int(sz * test_frac)
            n_va = int(sz * val_frac)
            n_tr = sz - n_va - n_te
            tr_idx.extend(range(pos,               pos + n_tr))
            va_idx.extend(range(pos + n_tr,        pos + n_tr + n_va))
            te_idx.extend(range(pos + n_tr + n_va, pos + sz))
            pos += sz
        tr_idx = np.array(tr_idx); va_idx = np.array(va_idx); te_idx = np.array(te_idx)
        print(f"  Per-exp split: train={len(tr_idx):,}  val={len(va_idx):,}  test={len(te_idx):,}")
        train_valid_mask = (flags[tr_idx] == 0)
        sc_endo = fit_scalers(endo_arr[tr_idx][train_valid_mask], endo_cols, scaling, scale_mode)
        sc_exo  = fit_scalers(exo_arr [tr_idx][train_valid_mask], exo_cols,  scaling, scale_mode)
        tr_ds = _make_ds(endo_arr[tr_idx], exo_arr[tr_idx], flags[tr_idx], tsv[tr_idx], times[tr_idx], 'train')
        va_ds = _make_ds(endo_arr[va_idx], exo_arr[va_idx], flags[va_idx], tsv[va_idx], times[va_idx], 'val')
        te_ds = _make_ds(endo_arr[te_idx], exo_arr[te_idx], flags[te_idx], tsv[te_idx], times[te_idx], 'test')
    else:
        T       = len(df)
        n_test  = int(T * test_frac)
        n_val   = int(T * val_frac)
        n_train = T - n_val - n_test
        print(f"  Time-split: train={n_train:,}  val={n_val:,}  test={n_test:,}")
        train_valid_mask = (flags[:n_train] == 0)
        sc_endo = fit_scalers(endo_arr[:n_train][train_valid_mask], endo_cols, scaling, scale_mode)
        sc_exo  = fit_scalers(exo_arr [:n_train][train_valid_mask], exo_cols,  scaling, scale_mode)
        tr_ds = _make_ds(endo_arr[:n_train],              exo_arr[:n_train],              flags[:n_train],              tsv[:n_train],              times[:n_train],              'train')
        va_ds = _make_ds(endo_arr[n_train:n_train+n_val], exo_arr[n_train:n_train+n_val], flags[n_train:n_train+n_val], tsv[n_train:n_train+n_val], times[n_train:n_train+n_val], 'val')
        te_ds = _make_ds(endo_arr[n_train+n_val:],        exo_arr[n_train+n_val:],        flags[n_train+n_val:],        tsv[n_train+n_val:],        times[n_train+n_val:],        'test')

    nw = min(0, __import__('os').cpu_count() or 1)
    kw = dict(num_workers=nw, pin_memory=False)
    if spike_oversample > 0.0:
        tgt_idx = forecast_target_indices if forecast_target_indices is not None else list(range(len(endo_cols)))
        raw_scores = []
        for i in range(len(tr_ds)):
            x_endo, _, y = tr_ds[i]
            ctx     = x_endo[tgt_idx]
            ctx_std = ctx.std().clamp(min=1e-3)
            delta   = (y - ctx[:, -1:]).abs().max()
            raw_scores.append(float(delta / ctx_std) + float(delta) * spike_abs_weight)
        raw_scores = np.array(raw_scores, dtype=np.float32)
        if per_exp_sizes is not None and len(per_exp_sizes) > 1:
            n_tr_list  = [sz - int(sz * val_frac) - int(sz * test_frac) for sz in per_exp_sizes]
            cum_n_tr   = np.cumsum([0] + n_tr_list)
            win_starts = np.array(tr_ds.indices)
            exp_ids    = np.clip(np.searchsorted(cum_n_tr, win_starts, side='right') - 1,
                                 0, len(per_exp_sizes) - 1)
            norm_scores = raw_scores.copy()
            for eid in range(len(per_exp_sizes)):
                mask = (exp_ids == eid)
                if mask.sum() > 0:
                    med = np.median(raw_scores[mask])
                    if med > 1e-9:
                        norm_scores[mask] /= med
            scores = np.clip(norm_scores, 1e-6, None) ** spike_oversample
        else:
            scores = np.clip(raw_scores, 1e-6, None) ** spike_oversample
        sampler  = WeightedRandomSampler(scores, num_samples=len(scores), replacement=True)
        train_dl = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler, drop_last=True, **kw)
        print(f"  Spike sampler: power={spike_oversample}  abs_weight={spike_abs_weight}  "
              f"max_w/min_w={scores.max()/scores.min():.1f}x")
    else:
        train_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    val_dl  = DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw)
    test_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False, **kw)

    if save_scalers_to:
        sp = Path(save_scalers_to); sp.mkdir(parents=True, exist_ok=True)
        joblib.dump(sc_endo, sp / 'scaler_endo.joblib')
        joblib.dump(sc_exo,  sp / 'scaler_exo.joblib')
        with open(sp / 'col_config.json', 'w') as f:
            json.dump({'endo_cols': endo_cols, 'exo_cols': exo_cols,
                       'scale_mode': scale_mode}, f, indent=2)

    return train_dl, val_dl, test_dl, sc_endo, sc_exo


# ── RevIN ──────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    def __init__(self, num_vars: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_vars, 1))
            self.beta  = nn.Parameter(torch.zeros(1, num_vars, 1))
        else:
            self.gamma = self.beta = None

    def normalise(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True).clamp(min=self.eps)
        x_n  = (x - mean) / std
        if self.gamma is not None:
            x_n = x_n * self.gamma + self.beta
        return x_n, (mean, std)

    def denormalise(self, x, mean, std):
        F = x.shape[1]
        if self.gamma is not None:
            g = self.gamma[:, :F, :]
            b = self.beta[:, :F, :]
            x = (x - b) / g.clamp(min=self.eps)
        return x * std + mean


# ── Diffusion schedule ─────────────────────────────────────────────────────

def make_beta_schedule(schedule: str, n_steps: int,
                       beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    if schedule == 'linear':
        return torch.linspace(beta_start, beta_end, n_steps)
    # cosine schedule (Nichol & Dhariwal 2021)
    steps = n_steps + 1
    x     = torch.linspace(0, n_steps, steps)
    ac    = torch.cos(((x / n_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
    ac    = ac / ac[0]
    betas = 1 - ac[1:] / ac[:-1]
    return betas.clamp(0, 0.999)


class DiffusionSchedule:
    """Pre-computed diffusion constants; lives on the same device as the model."""
    def __init__(self, n_steps: int, schedule: str, device: torch.device):
        betas           = make_beta_schedule(schedule, n_steps).to(device)
        alphas          = 1.0 - betas
        alpha_bars      = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self.n_steps         = n_steps
        self.betas           = betas
        self.alphas          = alphas
        self.alpha_bars      = alpha_bars
        self.alpha_bars_prev = alpha_bars_prev
        # Pre-computed for reverse step
        self.sqrt_recip_alpha    = (1.0 / alphas.sqrt())
        self.sqrt_alpha_bar      = alpha_bars.sqrt()
        self.sqrt_one_minus_ab   = (1.0 - alpha_bars).sqrt()
        self.posterior_variance  = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    def q_sample(self, y0: torch.Tensor, k: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: y_k = √ᾱ_k · y0 + √(1−ᾱ_k) · noise"""
        s  = self.sqrt_alpha_bar[k - 1]
        sm = self.sqrt_one_minus_ab[k - 1]
        # broadcast over [B, n_forecast, H]
        while s.dim() < y0.dim(): s = s.unsqueeze(-1); sm = sm.unsqueeze(-1)
        return s * y0 + sm * noise

    @torch.no_grad()
    def p_sample_step(self, model_out: torch.Tensor, y_k: torch.Tensor,
                      k: int) -> torch.Tensor:
        """One DDPM reverse step: y_{k-1} from y_k and predicted noise."""
        eps  = model_out
        coef = self.betas[k - 1] / self.sqrt_one_minus_ab[k - 1]
        mean = self.sqrt_recip_alpha[k - 1] * (y_k - coef * eps)
        if k == 1:
            return mean
        var  = self.posterior_variance[k - 1]
        return mean + var.sqrt() * torch.randn_like(y_k)

    @torch.no_grad()
    def p_sample_ddim(self, model_out: torch.Tensor, y_k: torch.Tensor,
                      k: int, clip_normalized: float = 5.0) -> torch.Tensor:
        """
        One DDIM reverse step (η=0, deterministic).
        Eliminates stochastic noise accumulation → no spike artefacts.
        x0_hat is clipped to ±clip_normalized in RevIN-normalised space.
        """
        eps = model_out
        s   = self.sqrt_alpha_bar[k - 1]       # sqrt(ᾱ_k)
        sm  = self.sqrt_one_minus_ab[k - 1]    # sqrt(1-ᾱ_k)
        while s.dim() < y_k.dim():
            s = s.unsqueeze(-1); sm = sm.unsqueeze(-1)
        x0_hat = (y_k - sm * eps) / s.clamp(min=1e-6)
        if clip_normalized > 0:
            x0_hat = x0_hat.clamp(-clip_normalized, clip_normalized)
        if k == 1:
            return x0_hat
        sp  = self.sqrt_alpha_bar[k - 2]       # sqrt(ᾱ_{k-1})
        smp = self.sqrt_one_minus_ab[k - 2]    # sqrt(1-ᾱ_{k-1})
        while sp.dim() < y_k.dim():
            sp = sp.unsqueeze(-1); smp = smp.unsqueeze(-1)
        return sp * x0_hat + smp * eps


# ── Model building blocks ──────────────────────────────────────────────────

class DiffusionEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion step k → [B, d_model]."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=k.device) / (half - 1))
        args = k.float().unsqueeze(1) * freq.unsqueeze(0)       # [B, half]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)       # [B, d_model]
        return self.proj(emb)


class _Attn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q, k, v)
        return self.norm(q + self.drop(out))


class _FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_ff, d_model),
                                  nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ScoreBlock(nn.Module):
    """
    One residual block of the CSDI score network.

    Processes noisy target tokens [B, n_forecast, H, D]:
      1. Temporal self-attention  — each feature's H steps attend to themselves
      2. Feature  self-attention  — each time step's n_forecast vars attend to each other
      3. Condition cross-attention — target tokens query condition tokens from context
      4. FFN
    Diffusion step embedding is injected as a bias before step 1.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.temporal_attn = _Attn(d_model, n_heads, dropout)
        self.feature_attn  = _Attn(d_model, n_heads, dropout)
        self.cond_attn     = _Attn(d_model, n_heads, dropout)
        self.ffn           = _FFN(d_model, d_ff, dropout)
        self.diff_proj     = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, diff_emb: torch.Tensor,
                cond_kv: torch.Tensor) -> torch.Tensor:
        # x:        [B, n_forecast, H, D]
        # diff_emb: [B, D]
        # cond_kv:  [B, T_ctx, D]
        B, F, H, D = x.shape

        # Inject diffusion step embedding
        x = x + self.diff_proj(diff_emb)[:, None, None, :]

        # 1. Temporal self-attention (shared weights across forecast variables)
        xt = x.reshape(B * F, H, D)
        xt = self.temporal_attn(xt, xt, xt)
        x  = xt.reshape(B, F, H, D)

        # 2. Feature self-attention (shared weights across time steps)
        xf = x.permute(0, 2, 1, 3).reshape(B * H, F, D)
        xf = self.feature_attn(xf, xf, xf)
        x  = xf.reshape(B, H, F, D).permute(0, 2, 1, 3)

        # 3. Condition cross-attention (Q from flattened target, KV from context)
        xc  = x.reshape(B, F * H, D)
        kv  = cond_kv
        xc  = self.cond_attn(xc, kv, kv)
        x   = xc.reshape(B, F, H, D)

        # 4. FFN (shared across B, F, H)
        x = self.ffn(x.reshape(B * F * H, D)).reshape(B, F, H, D)
        return x


class ContextEncoder(nn.Module):
    """
    Encodes the context window into condition tokens for cross-attention.

    Projects (endo + exo) at each time step to d_model, adds positional
    embedding, then refines with transformer layers.
    Output: [B, context_len, d_model]
    """
    def __init__(self, c_endo: int, c_exo: int, context_len: int,
                 d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.proj    = nn.Linear(c_endo + c_exo, d_model)
        self.pos_emb = nn.Parameter(torch.empty(1, context_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.layers  = nn.ModuleList([
            nn.ModuleDict({
                'attn': _Attn(d_model, n_heads, dropout),
                'ffn':  _FFN(d_model, d_ff, dropout),
            })
            for _ in range(n_layers)
        ])

    def forward(self, x_endo: torch.Tensor, x_exo: torch.Tensor) -> torch.Tensor:
        # x_endo: [B, C_e, T], x_exo: [B, C_x, T]
        x = torch.cat([x_endo, x_exo], dim=1).permute(0, 2, 1)  # [B, T, C_e+C_x]
        x = self.proj(x) + self.pos_emb                          # [B, T, D]
        for layer in self.layers:
            x = layer['attn'](x, x, x)
            x = layer['ffn'](x)
        return x


class ReactorCSDIModel(nn.Module):
    """
    CSDI model for reactor batch forecasting.

    Training:  forward(x_endo, x_exo, y0) → noise prediction MSE loss
    Inference: sample(x_endo, x_exo, n_samples) → [B, n_samples, n_forecast, H]
    """
    def __init__(self,
                 c_endo: int, c_exo: int, context_len: int, horizon: int,
                 n_forecast: int = 1,
                 forecast_target_indices: Optional[List[int]] = None,
                 n_diffusion_steps: int = 100,
                 beta_schedule: str = 'cosine',
                 d_model: int = 64,
                 n_heads: int = 4,
                 d_ff: int = 256,
                 n_score_layers: int = 4,
                 n_ctx_layers: int = 2,
                 dropout: float = 0.1,
                 use_revin: bool = True,
                 revin_sigma_floor: float = 0.0,
                 residual_anchor: bool = False,
                 use_batch_token: bool = False,
                 batch_token_idx: Optional[List[int]] = None):
        super().__init__()
        self.n_diffusion_steps        = n_diffusion_steps
        self.beta_schedule            = beta_schedule
        self.n_forecast               = n_forecast
        self.horizon                  = horizon
        self.use_revin                = use_revin
        self.revin_sigma_floor        = revin_sigma_floor
        self.residual_anchor          = residual_anchor
        self.use_batch_token          = use_batch_token
        self.batch_token_idx          = batch_token_idx
        self.forecast_target_indices  = (list(range(n_forecast))
                                         if forecast_target_indices is None
                                         else list(forecast_target_indices))
        if use_revin:
            self.revin = RevIN(c_endo)

        self.context_encoder = ContextEncoder(
            c_endo, c_exo, context_len, d_model, n_heads, d_ff, n_ctx_layers, dropout)
        self.diff_emb    = DiffusionEmbedding(d_model)
        self.input_proj  = nn.Linear(1, d_model)
        self.pos_emb     = nn.Parameter(torch.empty(1, 1, horizon, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.score_layers = nn.ModuleList([
            ScoreBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_score_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        if use_batch_token and batch_token_idx:
            self.batch_proj = nn.Linear(len(batch_token_idx), d_model)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

        # Diffusion schedule — built lazily on first use so device is known
        self._schedule: Optional[DiffusionSchedule] = None

    def _get_schedule(self, device: torch.device) -> DiffusionSchedule:
        if self._schedule is None or self._schedule.betas.device != device:
            self._schedule = DiffusionSchedule(
                self.n_diffusion_steps, self.beta_schedule, device)
        return self._schedule

    def _run_score_network(self, y_noisy: torch.Tensor, k: torch.Tensor,
                           cond: torch.Tensor) -> torch.Tensor:
        """ε_θ(y_noisy, k, cond) → predicted noise, same shape as y_noisy."""
        B, F, H = y_noisy.shape
        # Project each scalar to d_model
        x = self.input_proj(y_noisy.unsqueeze(-1))   # [B, F, H, D]
        x = x + self.pos_emb                          # broadcast over B, F
        de = self.diff_emb(k)                         # [B, D]
        for layer in self.score_layers:
            x = layer(x, de, cond)
        return self.output_proj(x).squeeze(-1)        # [B, F, H]

    def forward(self, x_endo: torch.Tensor, x_exo: torch.Tensor,
                y0: torch.Tensor, peak_weight: float = 0.0) -> torch.Tensor:
        """Training: returns noise-prediction MSE loss (+ optional peak auxiliary loss)."""
        B = y0.shape[0]
        if self.use_revin:
            x_endo, (rv_mean, rv_std) = self.revin.normalise(x_endo)
            rv_mean_tgt = rv_mean[:, self.forecast_target_indices, :]   # [B, F, 1]
            rv_std_tgt  = rv_std[:, self.forecast_target_indices, :]
            if self.revin_sigma_floor > 0:
                rv_std_tgt = rv_std_tgt.clamp(min=self.revin_sigma_floor)
            y0 = (y0 - rv_mean_tgt) / rv_std_tgt
        # Capture last observed in normalised space (after RevIN if used)
        last_sc = x_endo[:, self.forecast_target_indices, -1:]          # [B, F, 1]
        if self.residual_anchor:
            y0 = y0 - last_sc
        schedule = self._get_schedule(y0.device)
        cond  = self.context_encoder(x_endo, x_exo)                     # [B, T, D]

        if self.use_batch_token and self.batch_token_idx is not None:
            scalars = x_exo[:, self.batch_token_idx, 0]                  # [B, 2]
            cond = cond + self.batch_proj(scalars).unsqueeze(1)          # broadcast [B,1,D]

        k     = torch.randint(1, self.n_diffusion_steps + 1, (B,), device=y0.device)
        noise = torch.randn_like(y0)
        y_k   = schedule.q_sample(y0, k, noise)
        pred  = self._run_score_network(y_k, k, cond)
        loss  = F.mse_loss(pred, noise)
        if peak_weight > 0.0:
            # Derive x0_hat via DDPM posterior: x0 = (y_k - sqrt(1-abar)*eps) / sqrt(abar)
            # Only apply for low-to-medium noise steps (k ≤ T//2, i.e. s > ~0.7 for cosine)
            # to avoid huge x0_hat values when s ≈ 0 at high noise levels.
            s  = schedule.sqrt_alpha_bar[k - 1]   # [B]
            sm = schedule.sqrt_one_minus_ab[k - 1]
            snr_ok = s > 0.3                       # only when sqrt(ᾱ_k) is large enough
            if snr_ok.any():
                s_ok  = s[snr_ok];  sm_ok = sm[snr_ok]
                yk_ok = y_k[snr_ok]; pr_ok = pred[snr_ok]; y0_ok = y0[snr_ok]
                while s_ok.dim() < yk_ok.dim():
                    s_ok = s_ok.unsqueeze(-1); sm_ok = sm_ok.unsqueeze(-1)
                x0_hat   = (yk_ok - sm_ok * pr_ok) / s_ok.clamp(min=1e-6)
                peak_mse = (x0_hat.max(dim=-1).values - y0_ok.max(dim=-1).values).pow(2).mean()
                loss     = loss + peak_weight * peak_mse
        return loss

    @torch.no_grad()
    def sample(self, x_endo: torch.Tensor, x_exo: torch.Tensor,
               n_samples: int = 50,
               sampler: str = 'ddpm',
               clip_normalized: float = 5.0) -> torch.Tensor:
        """
        Full reverse diffusion.
        sampler='ddpm' (default): stochastic DDPM with per-step trajectory clipping.
            Each of n_samples follows a different path → genuine uncertainty estimates.
        sampler='ddim': deterministic (η=0). All n_samples collapse to the same result.
            Only useful for point-forecast debugging.
        Returns samples of shape [B, n_samples, n_forecast, H].
        """
        B            = x_endo.shape[0]
        rv_mean_tgt  = rv_std_tgt = None
        if self.use_revin:
            x_endo, (rv_mean, rv_std) = self.revin.normalise(x_endo)
            rv_mean_tgt = rv_mean[:, self.forecast_target_indices, :]   # [B, F, 1]
            rv_std_tgt  = rv_std[:, self.forecast_target_indices, :]
            if self.revin_sigma_floor > 0:
                rv_std_tgt = rv_std_tgt.clamp(min=self.revin_sigma_floor)
        # Capture last observed in normalised space (after RevIN if used)
        last_sc  = x_endo[:, self.forecast_target_indices, -1:]         # [B, F, 1]
        schedule = self._get_schedule(x_endo.device)
        cond     = self.context_encoder(x_endo, x_exo)                  # [B, T, D]

        if self.use_batch_token and self.batch_token_idx is not None:
            scalars = x_exo[:, self.batch_token_idx, 0]                  # [B, 2]
            cond = cond + self.batch_proj(scalars).unsqueeze(1)          # broadcast [B,1,D]

        all_samples = []
        for _ in range(n_samples):
            y = torch.randn(B, self.n_forecast, self.horizon, device=x_endo.device)
            for step in reversed(range(1, self.n_diffusion_steps + 1)):
                k_t = torch.full((B,), step, device=x_endo.device, dtype=torch.long)
                eps = self._run_score_network(y, k_t, cond)
                if sampler == 'ddim':
                    # Deterministic: all n_samples collapse to the same trajectory.
                    # Gives zero variance — only useful for point-forecast debugging.
                    y = schedule.p_sample_ddim(eps, y, step, clip_normalized)
                else:
                    # Stochastic DDPM: each sample follows a different trajectory,
                    # giving genuine uncertainty estimates.
                    y = schedule.p_sample_step(eps, y, step)
                    # Clip in RevIN-normalised space to prevent runaway trajectories.
                    # Does NOT collapse diversity — samples still diverge via different
                    # starting noise and per-step stochastic kicks.
                    if clip_normalized > 0 and step > 1:
                        y = y.clamp(-clip_normalized, clip_normalized)
            if self.residual_anchor:
                y = y + last_sc
            if self.use_revin:
                y = self.revin.denormalise(y, rv_mean_tgt, rv_std_tgt)
            all_samples.append(y)

        return torch.stack(all_samples, dim=1)             # [B, n_samples, F, H]


# ── Training helpers ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, peak_weight: float = 0.0):
    model.train(); total, n = 0.0, 0
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        optimizer.zero_grad()
        loss = model(x_endo, x_exo, y, peak_weight)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * x_endo.size(0); n += x_endo.size(0)
    return total / n if n > 0 else float('nan')


@torch.no_grad()
def eval_epoch(model, loader, device, peak_weight: float = 0.0):
    model.eval(); total, n = 0.0, 0
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        loss = model(x_endo, x_exo, y, peak_weight)
        total += loss.item() * x_endo.size(0); n += x_endo.size(0)
    return total / n if n > 0 else float('nan')


@torch.no_grad()
def collect_predictions(model, loader, device, n_samples: int = 50,
                        sampler: str = 'ddim', clip_normalized: float = 5.0):
    """
    Returns (means, stds, targets) each of shape [N, n_forecast, H].
    Draws n_samples from the reverse diffusion and summarises.
    """
    model.eval(); means, stds, targets = [], [], []
    for x_endo, x_exo, y in loader:
        x_endo = x_endo.to(device); x_exo = x_exo.to(device)
        samps  = model.sample(x_endo, x_exo, n_samples=n_samples,
                              sampler=sampler, clip_normalized=clip_normalized)
        means.append(samps.mean(dim=1).cpu().numpy())
        stds .append(samps.std (dim=1).cpu().numpy())
        targets.append(y.numpy())
    return (np.concatenate(means),
            np.concatenate(stds),
            np.concatenate(targets))


def spike_window_mask(dataset, forecast_target_indices, threshold: float) -> np.ndarray:
    mask = np.zeros(len(dataset), dtype=bool)
    for i in range(len(dataset)):
        x_endo, _, y = dataset[i]
        ctx     = x_endo[forecast_target_indices]
        ctx_std = float(ctx.std().clamp(min=1e-3))
        delta   = float((y - ctx[:, -1:]).abs().max())
        mask[i] = (delta / ctx_std) >= threshold
    return mask


def _print_point_metrics(label, means_o, tgts_o, means_sc, tgts_sc, target_cols):
    N = means_o.shape[0]
    per_ch = {}
    print(f"\n  Point forecast — {label}  (n={N}):")
    for ci, col in enumerate(target_cols):
        if N == 0:
            print(f"    [{col}]  (no windows)"); per_ch[col] = {}; continue
        mae  = float(np.mean(np.abs(means_o[:, ci, :] - tgts_o[:, ci, :])))
        rmse = float(np.sqrt(np.mean((means_o[:, ci, :] - tgts_o[:, ci, :]) ** 2)))
        mse  = float(np.mean((means_sc[:, ci, :] - tgts_sc[:, ci, :]) ** 2))
        print(f"    [{col}]  MSE(scaled)={mse:.5f}  MAE={mae:.4f}  RMSE={rmse:.4f} mg/L")
        per_ch[col] = {'mae': mae, 'rmse': rmse, 'mse_scaled': mse}
    return per_ch


def _spike_scores_for_candidates(candidates, endo_te_sc, target_channel_idx,
                                  context_len, horizon):
    scores = np.zeros(len(candidates), dtype=np.float32)
    for j, s in enumerate(candidates):
        ctx   = endo_te_sc[s:s + context_len, target_channel_idx]
        hor   = endo_te_sc[s + context_len:s + context_len + horizon, target_channel_idx]
        std   = float(np.std(ctx)) or 1e-3
        delta = float(np.max(np.abs(hor - ctx[-1])))
        scores[j] = delta / std
    return scores


# ── Calibration ────────────────────────────────────────────────────────────

def coverage_report(means, stds, targets):
    """Empirical coverage using sample std as Gaussian approximation."""
    results = {}
    for label, z in [('68% (1σ)', 1.0), ('90% (1.645σ)', 1.645), ('95% (1.96σ)', 1.96)]:
        inside = np.abs(targets - means) <= z * stds
        results[label] = float(inside.mean())
    return results


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_horizon(model, sc_endo, sc_exo, df, endo_cols, exo_cols,
                 scale_mode, context_len, horizon, out_path, target_col,
                 log_cols=None, n_windows=6, context_show=60, seed=42,
                 val_frac=0.1, test_frac=0.1, target_channel_idx: int = 0,
                 n_samples: int = 50, n_spike_windows: int = 6,
                 spike_threshold: float = 1.0):
    device = next(model.parameters()).device
    T = len(df); n_test = int(T * test_frac); n_val = int(T * val_frac)
    n_train = T - n_val - n_test
    flags   = df['flag'].values.astype(int)

    endo_te_raw = df[target_col].values[n_train + n_val:]
    endo_te_sc  = apply_scalers(df[endo_cols].values[n_train + n_val:], sc_endo, endo_cols, scale_mode)
    exo_te_sc   = apply_scalers(df[exo_cols ].values[n_train + n_val:], sc_exo,  exo_cols,  scale_mode)
    flags_te    = flags[n_train + n_val:]

    valid_te   = np.where(flags_te == 0)[0]
    candidates = [i for i in valid_te if i + context_len + horizon < len(endo_te_sc)]
    if not candidates:
        print("  No valid windows for plotting."); return
    n_windows = min(n_windows, len(candidates))

    rng    = np.random.default_rng(seed)
    starts = sorted(rng.choice(candidates, size=n_windows, replace=False))

    spike_starts = []
    if n_spike_windows > 0:
        scores = _spike_scores_for_candidates(candidates, endo_te_sc,
                                              target_channel_idx, context_len, horizon)
        spike_cands = [(candidates[i], float(scores[i]))
                       for i in np.argsort(scores)[::-1]
                       if scores[i] >= spike_threshold]
        dedup = []
        for s, _ in spike_cands:
            if all(abs(s - sel) >= horizon for sel in dedup):
                dedup.append(s)
        rng_sp = np.random.default_rng(seed + 1)
        n_sel = min(n_spike_windows, len(dedup))
        if n_sel > 0:
            idx_sel = rng_sp.choice(len(dedup), size=n_sel, replace=False)
            spike_starts = sorted([dedup[i] for i in idx_sel])

    def _render_windows(window_starts, save_path, title_suffix=''):
        fig, axes = plt.subplots(len(window_starts), 1,
                                 figsize=(11, 4 * len(window_starts)), squeeze=False)
        ctx_x = np.arange(-context_show, 0); fore_x = np.arange(0, horizon)
        for row, s in enumerate(window_starts):
            x_e = torch.tensor(endo_te_sc[s:s+context_len].T, dtype=torch.float32).unsqueeze(0).to(device)
            x_x = torch.tensor(exo_te_sc [s:s+context_len].T, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                samps = model.sample(x_e, x_x, n_samples=n_samples)
            mean_sc = samps[0, :, target_channel_idx, :].mean(0).cpu().numpy()
            std_sc  = samps[0, :, target_channel_idx, :].std (0).cpu().numpy()
            inv      = lambda a: inverse_target(a, sc_endo, target_col, endo_cols, scale_mode, log_cols)
            mean_orig = inv(mean_sc)
            upper_1s  = inv(mean_sc + std_sc);        lower_1s = inv(mean_sc - std_sc)
            upper_95  = inv(mean_sc + 1.96 * std_sc); lower_95 = inv(mean_sc - 1.96 * std_sc)
            ctx_sl    = endo_te_raw[s + context_len - context_show : s + context_len]
            actual_sl = endo_te_raw[s + context_len : s + context_len + horizon]
            if log_cols and target_col in log_cols:
                ctx_raw = np.expm1(ctx_sl); actual_raw = np.expm1(actual_sl)
            else:
                ctx_raw = ctx_sl; actual_raw = actual_sl
            ax = axes[row][0]
            ax.plot(ctx_x,  ctx_raw,   color='#888888', lw=1.2, alpha=0.7, label='Context')
            ax.fill_between(fore_x, lower_95, upper_95, color='#9467bd', alpha=0.15, label='95% PI')
            ax.fill_between(fore_x, lower_1s, upper_1s, color='#9467bd', alpha=0.30, label='68% PI')
            ax.plot(fore_x, mean_orig,  color='#9467bd', lw=2.0, label='Mean forecast (CSDI)')
            ax.plot(fore_x, actual_raw, color='#1f77b4', lw=2.0, label=f'Actual {target_col.upper()}')
            ax.axvline(0, color='black', lw=0.8, ls=':')
            mae  = float(np.mean(np.abs(mean_orig - actual_raw)))
            rmse = float(np.sqrt(np.mean((mean_orig - actual_raw) ** 2)))
            ax.set_title(f'Window {s}  |  {target_col.upper()}  |  MAE={mae:.3f}  RMSE={rmse:.3f}', fontsize=9)
            ax.set_xlabel('Steps from forecast origin', fontsize=8)
            ticks = np.concatenate([np.arange(-context_show, 0, max(1, context_show // 4)),
                                    np.arange(0, horizon + 1, max(1, horizon // 6))])
            ax.set_xticks(ticks); ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=7)
            if row == 0: ax.legend(fontsize=8, loc='upper left')
        fig.suptitle(f'{target_col.upper()} CSDI forecast  endo={endo_cols}  {title_suffix}\n'
                     f'{horizon}-step horizon  {n_samples} samples  seed={seed}', fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved: {save_path}")

    model.eval()
    _render_windows(starts, out_path)
    if n_spike_windows > 0 and spike_starts:
        spike_path = out_path.parent / (out_path.stem + '_spikes' + out_path.suffix)
        _render_windows(spike_starts, spike_path, title_suffix='[spike windows]')


# ── Core training function ─────────────────────────────────────────────────

def _freeze_for_finetune(model: nn.Module, n_layers: int = 1) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for attr in ('head', 'head_net', 'output_proj'):
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters():
                p.requires_grad_(True)
    for attr in ('layers', 'score_layers', 'encoder_layers', 'blocks'):
        if hasattr(model, attr):
            enc = getattr(model, attr)
            for layer in list(enc)[-n_layers:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            break
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Fine-tune: {trainable:,} / {total:,} params trainable  "
          f"(head + last {n_layers} encoder layer(s))")


def _print_loo_summary(all_results: dict) -> None:
    import math
    rmses, nlls = [], []
    for res in all_results.values():
        m = res.get('metrics', {})
        if 'rmse' in m:
            rmses.append(m['rmse'])
        for key in ('test_nll_scaled', 'test_mse_scaled'):
            if key in res:
                nlls.append(res[key]); break
    print(f"\n{'='*60}")
    print(f"  LOO Summary  ({len(all_results)} folds)")
    if rmses:
        print(f"  RMSE  mean={float(np.mean(rmses)):.4f}  std={float(np.std(rmses)):.4f}")
    valid = [v for v in nlls if not math.isnan(v)]
    if valid:
        print(f"  Score mean={float(np.mean(valid)):.4f}  std={float(np.std(valid)):.4f}")
    print(f"{'='*60}")


def train_experiment(args, exp_tags, device, endo_cols, exo_cols,
                     target_cols, log_cols, out_dir,
                     transfer_from=None, lr_override=None,
                     eval_exp_tags=None, freeze_layers=0):
    forecast_target_indices = [endo_cols.index(t) for t in target_cols]
    n_forecast = len(target_cols)
    lr = lr_override if lr_override is not None else args.lr
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Loading data ─────────────────────────────────────────────────")
    df = load_experiment(args.data_paths, args.cheatsheet, exp_tags)
    df['time_since_valid'] = compute_time_since_valid(df)
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].values)

    n_valid = int((df['flag'] == 0).sum())
    n_excl  = int((df['flag'] == 1).sum())
    print(f"  Total: {len(df):,}  Valid: {n_valid:,}  Excluded: {n_excl:,}")

    per_exp_sizes = None
    if len(exp_tags) > 1:
        per_exp_sizes = [len(load_experiment(args.data_paths, args.cheatsheet, [t])) for t in exp_tags]

    print("\n── Building dataloaders ─────────────────────────────────────────")
    train_dl, val_dl, test_dl, sc_endo, sc_exo = build_dataloaders(
        df, endo_cols, exo_cols,
        context_len=args.context_len, horizon=args.horizon,
        batch_size=args.batch_size, stride=args.stride,
        val_frac=args.val_frac, test_frac=args.test_frac,
        scaling=args.scaling, scale_mode=args.scale_mode,
        segment_isolated=args.segment_isolated,
        gap_threshold=args.gap_threshold,
        segment_gap_minutes=args.segment_gap_minutes,
        min_segment_rows=args.min_segment_rows,
        forecast_target_indices=forecast_target_indices,
        spike_oversample=args.spike_oversample,
        spike_abs_weight=args.spike_abs_weight,
        save_scalers_to=str(out_dir),
        per_exp_sizes=per_exp_sizes,
    )

    if len(train_dl.dataset) == 0:
        print("  ERROR: training dataset empty — skipping.")
        return None, {}

    batch_token_idx = (
        [exo_cols.index(c) for c in args.batch_token_cols if c in exo_cols]
        if args.batch_token else None
    )
    print("\n── Building model ───────────────────────────────────────────────")
    model = ReactorCSDIModel(
        c_endo=len(endo_cols), c_exo=len(exo_cols),
        context_len=args.context_len, horizon=args.horizon,
        n_forecast=n_forecast,
        forecast_target_indices=forecast_target_indices,
        n_diffusion_steps=args.n_diffusion_steps,
        beta_schedule=args.beta_schedule,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_score_layers=args.n_score_layers,
        n_ctx_layers=args.n_ctx_layers,
        dropout=args.dropout,
        use_revin=not args.no_revin,
        revin_sigma_floor=args.revin_sigma_floor,
        residual_anchor=args.residual_anchor,
        use_batch_token=args.batch_token,
        batch_token_idx=batch_token_idx,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  |  LR: {lr:.2e}"
          + (f"  [from {Path(transfer_from).name}]" if transfer_from else "  [fresh]"))

    if transfer_from:
        ckpt = Path(transfer_from)
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"  Loaded  |  missing={len(missing)}  unexpected={len(unexpected)}")
        else:
            print(f"  WARNING: checkpoint not found: {ckpt}")

    if freeze_layers > 0:
        _freeze_for_finetune(model, n_layers=freeze_layers)

    # Forward probe (training loss, no sampling needed)
    try:
        xe = torch.randn(min(2, args.batch_size), len(endo_cols), args.context_len, device=device)
        xx = torch.randn(min(2, args.batch_size), len(exo_cols),  args.context_len, device=device)
        y0 = torch.randn(min(2, args.batch_size), n_forecast, args.horizon, device=device)
        loss_p = model(xe, xx, y0)
        loss_p.backward(); model.zero_grad()
        mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else 0.0
        print(f"  Forward probe: {mem:.1f} MB  loss={loss_p.item():.4f}  ✓")
    except RuntimeError as e:
        print(f"  Forward probe FAILED: {e}"); return None, {}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr * 0.01)

    print(f"\n── Training up to {args.epochs} epochs  (patience={args.patience}) ─")
    print(f"   Diffusion steps={args.n_diffusion_steps}  schedule={args.beta_schedule}")
    history = {'train': [], 'val': []}
    best_val, patience_cnt = float('inf'), 0
    best_path = out_dir / 'best_model.pt'

    for epoch in range(1, args.epochs + 1):
        t0      = time.time()
        tr_loss = train_epoch(model, train_dl, optimizer, device, args.peak_weight)
        va_loss = eval_epoch (model, val_dl,   device, args.peak_weight)
        scheduler.step()
        history['train'].append(float(tr_loss))
        history['val'  ].append(float(va_loss))
        if va_loss < best_val:
            best_val, patience_cnt = va_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"  ep {epoch:4d}/{args.epochs}  "
                  f"MSE train={tr_loss:.5f}  val={va_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"patience={patience_cnt}/{args.patience}  "
                  f"⏱ {time.time()-t0:.1f}s")
        if patience_cnt >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}"); break

    print("\n── Test evaluation ──────────────────────────────────────────────")
    model.load_state_dict(torch.load(best_path, map_location=device))

    plot_tags = eval_exp_tags if eval_exp_tags else exp_tags
    if eval_exp_tags:
        print(f"  LOO eval on held-out: {eval_exp_tags}")
        df_eval = load_experiment(args.data_paths, args.cheatsheet, eval_exp_tags)
        df_eval['time_since_valid'] = compute_time_since_valid(df_eval)
        for col in log_cols:
            if col in df_eval.columns:
                df_eval[col] = np.log1p(df_eval[col].values)
        # LOO eval: training scalers, last test_frac rows — matches concatenated/finetune split
        _endo_eval  = df_eval[endo_cols].values.astype(np.float64)
        _exo_eval   = df_eval[exo_cols].values.astype(np.float64)
        _flags_eval = df_eval['flag'].values.astype(int)
        _tsv_eval   = df_eval['time_since_valid'].values.astype(np.float32)
        _times_eval = df_eval['time'].values
        _T_eval     = len(df_eval)
        _n_te_eval  = int(_T_eval * args.test_frac)
        _te_sl_eval = slice(_T_eval - _n_te_eval, None)
        eval_ds = ReactorDataset(
            apply_scalers(_endo_eval[_te_sl_eval], sc_endo, endo_cols, args.scale_mode),
            apply_scalers(_exo_eval[_te_sl_eval],  sc_exo,  exo_cols,  args.scale_mode),
            _flags_eval[_te_sl_eval], args.context_len, args.horizon,
            stride=1, segment_isolated=args.segment_isolated,
            gap_threshold=args.gap_threshold, time_since_valid=_tsv_eval[_te_sl_eval],
            times=_times_eval[_te_sl_eval], segment_gap_minutes=args.segment_gap_minutes,
            min_segment_rows=args.min_segment_rows, target_indices=forecast_target_indices,
        )
        test_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    test_mse = eval_epoch(model, test_dl, device, args.peak_weight)
    print(f"  Test MSE (noise, scaled): {test_mse:.5f}")
    print(f"  Drawing {args.n_samples} samples per window for metrics …")

    results = {
        'experiments': exp_tags, 'target': target_cols,
        'endogenous': endo_cols, 'exogenous': exo_cols,
        'log_cols': list(log_cols), 'lr_used': lr,
        'test_mse_scaled': float(test_mse),
        'history': history, 'n_params': n_params, 'args': vars(args),
    }

    if len(test_dl.dataset) > 0:
        means, sample_stds, tgts = collect_predictions(
            model, test_dl, device, n_samples=args.n_samples,
            sampler=args.sampler, clip_normalized=args.clip_normalized)
        N, C_out, H = means.shape

        m_orig = np.stack([
            inverse_target(means[:,ci,:].reshape(-1), sc_endo, col,
                           endo_cols, args.scale_mode, log_cols).reshape(N, H)
            for ci, col in enumerate(target_cols)], axis=1)
        t_orig = np.stack([
            inverse_target(tgts[:,ci,:].reshape(-1), sc_endo, col,
                           endo_cols, args.scale_mode, log_cols).reshape(N, H)
            for ci, col in enumerate(target_cols)], axis=1)

        per_channel_metrics = _print_point_metrics(
            f"all windows ({C_out} channel(s), {args.n_samples} samples)",
            m_orig, t_orig, means, tgts, target_cols)

        smask   = spike_window_mask(test_dl.dataset, forecast_target_indices,
                                    args.spike_metric_threshold)
        n_spike = int(smask.sum()); n_flat = N - n_spike
        print(f"\n  Spike isolation — threshold={args.spike_metric_threshold}  "
              f"spike_windows={n_spike}/{N}  ({100*n_spike/max(N,1):.1f}%)")
        spike_metrics = _print_point_metrics(
            "spike windows", m_orig[smask], t_orig[smask],
            means[smask], tgts[smask], target_cols) if n_spike else {}
        flat_metrics  = _print_point_metrics(
            "non-spike windows", m_orig[~smask], t_orig[~smask],
            means[~smask], tgts[~smask], target_cols) if n_flat else {}

        cov = coverage_report(means, sample_stds, tgts)
        print(f"\n  Calibration — all windows (empirical from {args.n_samples} samples):")
        for label, frac in cov.items():
            nom  = float(label[:2]) / 100
            flag = '✓' if abs(frac - nom) < 0.08 else ('↓ overconfident' if frac < nom else '↑ conservative')
            print(f"    {label}: {frac:.1%}  {flag}")
        if n_spike:
            cov_spike = coverage_report(means[smask], sample_stds[smask], tgts[smask])
            print(f"\n  Calibration — spike windows:")
            for label, frac in cov_spike.items():
                nom  = float(label[:2]) / 100
                flag = '✓' if abs(frac - nom) < 0.08 else ('↓ overconfident' if frac < nom else '↑ conservative')
                print(f"    {label}: {frac:.1%}  {flag}")
        else:
            cov_spike = {}

        results['metrics']              = per_channel_metrics
        results['metrics_spike']        = spike_metrics
        results['metrics_flat']         = flat_metrics
        results['calibration']          = cov
        results['calibration_spike']    = cov_spike
        results['spike_window_count']   = n_spike
        results['spike_threshold_used'] = args.spike_metric_threshold

        # ── Per-experiment metrics ────────────────────────────────────────
        per_exp_metrics = {}
        breakdown_tags = eval_exp_tags if eval_exp_tags else exp_tags
        _bd_val_frac  = 0.0 if eval_exp_tags else args.val_frac
        _bd_test_frac = args.test_frac
        if breakdown_tags:
            print(f"\n  {'─'*56}")
            print(f"  Per-experiment breakdown:")
            for exp_tag in breakdown_tags:
                df_e = load_experiment(args.data_paths, args.cheatsheet, [exp_tag])
                if df_e.empty:
                    continue
                df_e['time_since_valid'] = compute_time_since_valid(df_e)
                for col in log_cols:
                    if col in df_e.columns:
                        df_e[col] = np.log1p(df_e[col].values)
                _T_e    = len(df_e)
                _n_te_e = int(_T_e * _bd_test_frac)
                _te_sl  = slice(_T_e - _n_te_e, None)
                _endo_e  = df_e[endo_cols].values.astype(np.float64)
                _exo_e   = df_e[exo_cols].values.astype(np.float64)
                _flags_e = df_e['flag'].values.astype(int)
                _tsv_e   = df_e['time_since_valid'].values.astype(np.float32)
                _times_e = df_e['time'].values
                _te_ds_e = ReactorDataset(
                    apply_scalers(_endo_e[_te_sl], sc_endo, endo_cols, args.scale_mode),
                    apply_scalers(_exo_e[_te_sl],  sc_exo,  exo_cols,  args.scale_mode),
                    _flags_e[_te_sl], args.context_len, args.horizon,
                    stride=1, segment_isolated=args.segment_isolated,
                    gap_threshold=args.gap_threshold, time_since_valid=_tsv_e[_te_sl],
                    times=_times_e[_te_sl], segment_gap_minutes=args.segment_gap_minutes,
                    min_segment_rows=args.min_segment_rows, target_indices=forecast_target_indices,
                )
                te_e     = DataLoader(_te_ds_e, batch_size=args.batch_size, shuffle=False,
                                      num_workers=0, pin_memory=False)
                sc_endo_e = sc_endo
                if len(te_e.dataset) == 0:
                    continue
                m_e, sp_e, t_e = collect_predictions(model, te_e, device)
                N_e = len(m_e)
                m_e_orig = np.stack([
                    inverse_target(m_e[:, ci, :].reshape(-1), sc_endo_e, col,
                                   endo_cols, args.scale_mode, log_cols).reshape(N_e, H)
                    for ci, col in enumerate(target_cols)], axis=1)
                t_e_orig = np.stack([
                    inverse_target(t_e[:, ci, :].reshape(-1), sc_endo_e, col,
                                   endo_cols, args.scale_mode, log_cols).reshape(N_e, H)
                    for ci, col in enumerate(target_cols)], axis=1)
                smask_e = spike_window_mask(te_e.dataset, forecast_target_indices,
                                            args.spike_metric_threshold)
                n_spike_e = int(smask_e.sum())
                n_flat_e  = N_e - n_spike_e
                print(f"\n  [{exp_tag}]  {N_e} test windows  {n_spike_e} spike  {n_flat_e} non-spike")
                exp_m = _print_point_metrics(exp_tag, m_e_orig, t_e_orig, m_e, t_e, target_cols)
                spike_m_e = {}
                flat_m_e  = {}
                if n_spike_e:
                    spike_m_e = _print_point_metrics(f"{exp_tag} spikes", m_e_orig[smask_e],
                                         t_e_orig[smask_e], m_e[smask_e], t_e[smask_e],
                                         target_cols)
                if n_flat_e:
                    flat_m_e = _print_point_metrics(f"{exp_tag} non-spike", m_e_orig[~smask_e],
                                         t_e_orig[~smask_e], m_e[~smask_e], t_e[~smask_e],
                                         target_cols)
                per_exp_metrics[exp_tag] = {'overall': exp_m, 'spike': spike_m_e, 'non_spike': flat_m_e}
        results['per_experiment_metrics'] = per_exp_metrics

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history['train'], label='Train MSE'); ax.plot(history['val'], label='Val MSE')
        ax.set_yscale('symlog'); ax.set_xlabel('Epoch'); ax.set_ylabel('Noise MSE')
        ax.set_title(f"{'+'.join(exp_tags)} – CSDI training history")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(out_dir / 'training_loss.png', dpi=150); plt.close(fig)

        for exp_tag in plot_tags:
            df_plot = load_experiment(args.data_paths, args.cheatsheet, [exp_tag])
            if df_plot.empty:
                continue
            df_plot['time_since_valid'] = compute_time_since_valid(df_plot)
            for col in log_cols:
                if col in df_plot.columns:
                    df_plot[col] = np.log1p(df_plot[col].values)
            plot_prefix = f'{exp_tag}_' if len(plot_tags) > 1 else ''
            for ti, tcol in enumerate(target_cols):
                plot_horizon(model, sc_endo, sc_exo, df_plot, endo_cols, exo_cols,
                             args.scale_mode, args.context_len, args.horizon,
                             out_dir / f'{plot_prefix}horizon_forecast_{tcol}.png',
                             target_col=tcol, log_cols=log_cols, seed=args.seed,
                             val_frac=args.val_frac, test_frac=args.test_frac,
                             target_channel_idx=ti, n_samples=args.n_samples,
                             n_spike_windows=args.n_spike_windows,
                             spike_threshold=args.spike_metric_threshold)
    else:
        print("  Test dataset empty — skipping metrics and plots.")

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results → {out_dir}")
    return best_path, results


# ── Main ───────────────────────────────────────────────────────────────────

def main(args):
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic      = False
    torch.backends.cudnn.benchmark          = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  mode={'sequential' if args.sequential else 'joint'}  "
          f"diffusion_steps={args.n_diffusion_steps}  schedule={args.beta_schedule}  seed={args.seed}")

    endo_cols, exo_cols = resolve_columns(args.endogenous, args.exogenous)
    target_cols: List[str] = args.target if args.target else [endo_cols[0]]
    for t in target_cols:
        if t not in endo_cols:
            raise ValueError(f"--target {t!r} is not in --endogenous {endo_cols}")
    log_cols: Set[str] = set(args.log_cols) if args.log_cols else set(endo_cols)

    print(f"\n── Column routing ───────────────────────────────────────────────")
    print(f"  Endogenous ({len(endo_cols)}): {endo_cols}")
    print(f"  Exogenous  ({len(exo_cols)}): {exo_cols[:6]} ...")
    print(f"  Target(s): {target_cols}   Log-transformed: {sorted(log_cols)}")

    base_out = Path(args.out_dir)

    if args.leave_one_out:
        print(f"\n{'='*60}")
        print(f"  Leave-one-out: {args.experiment}")
        print(f"{'='*60}")
        all_results = {}
        for held_out in args.experiment:
            train_tags = [e for e in args.experiment if e != held_out]
            if not train_tags:
                print(f"  Skipping {held_out} — only one experiment provided."); continue
            out_dir = base_out / f'LOO_{held_out.upper()}'
            print(f"\n{'='*60}")
            print(f"  Hold out: {held_out}  |  Train on: {train_tags}")
            print(f"{'='*60}")
            best_path, results = train_experiment(
                args, train_tags, device,
                endo_cols, exo_cols, target_cols, log_cols, out_dir,
                transfer_from=args.transfer_from,
                eval_exp_tags=[held_out],
            )
            if results:
                all_results[held_out] = results
        _print_loo_summary(all_results)
        summary = base_out / 'loo_summary.json'
        with open(summary, 'w') as f:
            json.dump({'mode': 'leave_one_out', 'experiments': args.experiment,
                       'endo_cols': endo_cols, 'log_cols': list(log_cols),
                       'target': target_cols, 'results': all_results}, f, indent=2)
        print(f"LOO summary → {summary}")

    elif args.finetune:
        print(f"\n{'='*60}")
        print(f"  Fine-tune: pre-train on {args.experiment[:-1]}  →  fine-tune on {args.experiment[-1]}")
        print(f"  Freeze all except head + last {args.finetune_layers} encoder layer(s)  |  LR={args.finetune_lr:.1e}")
        print(f"{'='*60}")
        pretrain_tags = args.experiment[:-1]
        finetune_tag  = args.experiment[-1]
        if pretrain_tags:
            pre_slug = '_'.join(e.upper() for e in pretrain_tags)
            pre_out  = base_out / f'pretrain_{pre_slug}'
            print(f"\n── Pre-training on {pretrain_tags} ──")
            best_path, _ = train_experiment(
                args, pretrain_tags, device,
                endo_cols, exo_cols, target_cols, log_cols, pre_out,
                transfer_from=args.transfer_from,
            )
            ckpt_for_ft = str(best_path) if best_path else args.transfer_from
        else:
            ckpt_for_ft = args.transfer_from
        ft_out = base_out / f'finetune_{finetune_tag.upper()}'
        print(f"\n── Fine-tuning on {finetune_tag} ──")
        ft_args = argparse.Namespace(**vars(args))
        ft_args.spike_oversample = args.finetune_spike_oversample if args.finetune_spike_oversample is not None else args.spike_oversample
        ft_args.spike_abs_weight = args.finetune_spike_abs_weight if args.finetune_spike_abs_weight is not None else args.spike_abs_weight
        train_experiment(
            ft_args, [finetune_tag], device,
            endo_cols, exo_cols, target_cols, log_cols, ft_out,
            transfer_from=ckpt_for_ft,
            freeze_layers=args.finetune_layers,
            lr_override=args.finetune_lr,
        )

    elif args.sequential:
        print(f"\n{'='*60}")
        print(f"  Sequential: {args.experiment}")
        print(f"  LR schedule: {args.lr:.1e} × {args.sequential_lr_scale}^i")
        print(f"{'='*60}")
        checkpoint  = args.transfer_from
        all_results = {}

        for i, exp_tag in enumerate(args.experiment):
            print(f"\n{'='*60}")
            print(f"  [{i+1}/{len(args.experiment)}]  {exp_tag}"
                  + (f"  ← {Path(checkpoint).name}" if checkpoint else "  [fresh]"))
            print(f"{'='*60}")
            lr_i    = args.lr * (args.sequential_lr_scale ** i)
            out_dir = base_out / exp_tag.upper()
            best_path, results = train_experiment(
                args, [exp_tag], device,
                endo_cols, exo_cols, target_cols, log_cols, out_dir,
                transfer_from=checkpoint, lr_override=lr_i,
            )
            if best_path is not None:
                checkpoint = str(best_path); all_results[exp_tag] = results
            else:
                print(f"  [{exp_tag}] skipped — keeping previous checkpoint")

        summary = base_out / 'sequential_summary.json'
        with open(summary, 'w') as f:
            json.dump({'experiments': args.experiment, 'endo_cols': endo_cols,
                       'log_cols': list(log_cols), 'target': target_cols,
                       'results': all_results}, f, indent=2)
        print(f"\nSequential training complete.  Summary: {summary}")

    else:
        exp_slug = '_'.join(e.upper() for e in args.experiment)
        if len(args.experiment) > 1:
            print(f"\nNOTE: {len(args.experiment)} experiments will be CONCATENATED.")
        train_experiment(
            args, args.experiment, device,
            endo_cols, exo_cols, target_cols, log_cols,
            base_out / exp_slug, transfer_from=args.transfer_from,
        )

    print("\nDone.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description='Probabilistic CSDI on reactor batch data.')

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument('--experiment',  nargs='+', required=True)
    p.add_argument('--data_paths',  nargs='+', required=True)
    p.add_argument('--cheatsheet',  default='filtered_data_stamps.txt')

    # ── Column routing ─────────────────────────────────────────────────────
    p.add_argument('--endogenous', nargs='+', default=['nh4'])
    p.add_argument('--exogenous',  nargs='*', default=[])
    p.add_argument('--target',     nargs='+', default=None,
                   help='Variable(s) to forecast. Must be in --endogenous.')
    p.add_argument('--log_cols',   nargs='*', default=None)

    # ── Training mode ──────────────────────────────────────────────────────
    p.add_argument('--sequential',          action='store_true')
    p.add_argument('--sequential_lr_scale', type=float, default=0.2)
    p.add_argument('--leave_one_out', action='store_true',
                   help='LOO evaluation: train on N-1 experiments, test on held-out.')
    p.add_argument('--finetune', action='store_true',
                   help='Pre-train on all but last experiment, then fine-tune on last.')
    p.add_argument('--finetune_layers', type=int, default=1,
                   help='Number of last encoder layers to keep trainable during fine-tune.')
    p.add_argument('--finetune_lr', type=float, default=1e-5,
                   help='Learning rate for fine-tune phase.')
    p.add_argument('--finetune_spike_oversample', type=float, default=None,
                   help='spike_oversample for fine-tune phase. Defaults to --spike_oversample.')
    p.add_argument('--finetune_spike_abs_weight', type=float, default=None,
                   help='spike_abs_weight for fine-tune phase. Defaults to --spike_abs_weight.')
    p.add_argument('--batch_token', action='store_true',
                   help='Project inoc_amount+inoc_conc via Linear(2,d_model) and add to context encoding.')
    p.add_argument('--batch_token_cols', nargs='+',
                   default=['inoc_amount', 'inoc_conc'],
                   help='Exo column names to use as batch token scalars (default: inoc_amount inoc_conc).')
    p.add_argument('--revin_sigma_floor', type=float, default=0.1,
                   help='Floor for RevIN context std when normalizing y0. '
                        'Prevents explosion for flat-context spike windows. 0=disabled.')
    p.add_argument('--transfer_from',       default=None)

    # ── Scaling ───────────────────────────────────────────────────────────
    p.add_argument('--scaling',    default='standard', choices=['standard', 'robust'])
    p.add_argument('--scale_mode', default='global',   choices=['global', 'per_var'])

    # ── Gap / segmentation ────────────────────────────────────────────────
    p.add_argument('--segment_isolated',    action='store_true')
    p.add_argument('--gap_threshold',       type=float, default=4.0)
    p.add_argument('--segment_gap_minutes', type=float, default=60.0)
    p.add_argument('--min_segment_rows',    type=int,   default=0)

    # ── Spike oversampling ────────────────────────────────────────────────
    p.add_argument('--spike_oversample', type=float, default=0.0,
                   help='0=off. Power applied to spike score. Start with 1.5.')
    p.add_argument('--spike_abs_weight', type=float, default=1.0,
                   help='Weight for the absolute-delta term of the spike score.')
    p.add_argument('--no_revin', action='store_true',
                   help='Disable Reversible Instance Normalisation (on by default).')
    p.add_argument('--residual_anchor', action='store_true',
                   help='Model predicts residuals from the last observed context value. '
                        'Subtracted in normalised space before diffusion, added back after '
                        'reverse diffusion before denormalisation. Default: off.')
    p.add_argument('--peak_weight', type=float, default=0.0,
                   help='Weight for auxiliary peak-MSE loss added to the noise-prediction '
                        'MSE. Peak supervision uses the DDPM x0-hat estimate derived from '
                        'the predicted noise; only applied for low-noise diffusion steps '
                        '(sqrt(alpha_bar) > 0.3) to avoid numerical instability. 0=off. Try 0.1.')
    p.add_argument('--sampler', default='ddpm', choices=['ddim', 'ddpm'],
                   help='Reverse diffusion sampler. ddpm (default): stochastic DDPM with '
                        'trajectory clipping — preserves genuine uncertainty across samples. '
                        'ddim: deterministic (η=0), all samples are identical, use only for '
                        'point-forecast debugging.')
    p.add_argument('--clip_normalized', type=float, default=5.0,
                   help='Clip x0_hat to ±N in RevIN-normalised space during DDIM sampling. '
                        'Prevents runaway trajectories. 0 to disable.')
    p.add_argument('--spike_metric_threshold', type=float, default=1.0,
                   help='Relative-delta threshold for spike window classification in metrics.')
    p.add_argument('--n_spike_windows', type=int, default=6,
                   help='Number of top-spike windows to plot separately. 0 to disable.')

    # ── Reproducibility ───────────────────────────────────────────────────
    p.add_argument('--seed', type=int, default=42)

    # ── Diffusion ─────────────────────────────────────────────────────────
    p.add_argument('--n_diffusion_steps', type=int,   default=100,
                   help='Number of forward/reverse diffusion steps. '
                        '50 is faster; 200 gives sharper samples.')
    p.add_argument('--beta_schedule',     default='cosine', choices=['linear', 'cosine'],
                   help='Noise schedule. Cosine is recommended.')
    p.add_argument('--n_samples',         type=int,   default=50,
                   help='Samples drawn per window at test time for metrics and plots.')

    # ── Architecture ──────────────────────────────────────────────────────
    p.add_argument('--d_model',       type=int,   default=64)
    p.add_argument('--n_heads',       type=int,   default=4)
    p.add_argument('--d_ff',          type=int,   default=256)
    p.add_argument('--n_score_layers',type=int,   default=4,
                   help='Residual blocks in the score network.')
    p.add_argument('--n_ctx_layers',  type=int,   default=2,
                   help='Transformer layers in the context encoder.')
    p.add_argument('--dropout',       type=float, default=0.1)

    # ── Sequence lengths ──────────────────────────────────────────────────
    p.add_argument('--context_len', type=int,   default=180)
    p.add_argument('--horizon',     type=int,   default=18)
    p.add_argument('--stride',      type=int,   default=1)

    # ── Optimisation ──────────────────────────────────────────────────────
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--patience',    type=int,   default=15)
    p.add_argument('--val_frac',    type=float, default=0.1)
    p.add_argument('--test_frac',   type=float, default=0.1)

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument('--out_dir', default='results/reactor_csdi')

    main(p.parse_args())
