"""
train_reactor_probabilistic.py – Probabilistic forecast on real reactor data.

Key options
-----------
  --experiment      KAU073 KAU075       Experiment tags (see --sequential for multi-exp strategy)
  --data_paths      f1.csv f2.csv       CSV data files
  --cheatsheet      stamps.txt          Cheatsheet with experiment date windows

  --endogenous      nh4 no3             Variables into the endogenous (endo) pathway.
                                        Multiple = cross-variate (coupled). First = primary target
                                        unless --target overrides.

  --exogenous       no3 do              Variables explicitly routed to the exogenous path.
                                        Can overlap with --endogenous (used in both pathways).
                                        Any ALL_DATA_COLS variable not listed in either arg
                                        defaults to exogenous automatically.

  --log_cols        nh4 no3             Which columns to apply log1p before scaling.
                                        Defaults to the same as --endogenous.
                                        Set explicitly to also log-transform exo vars
                                        (e.g. --log_cols nh4 no3 when no3 is exo-only).

  --target          nh4                 Primary variable for plots/metrics. Must be in
                                        --endogenous. Defaults to first --endogenous variable.

  --sequential                          Train one experiment at a time, in the order given,
                                        passing the checkpoint forward as a warm start.
                                        Each experiment gets its own scaler and output dir.
                                        Recommended for batch reactor data where experiments
                                        are independent chemical processes.

  --sequential_lr_scale  0.2            LR multiplier per subsequent experiment in sequential
                                        mode (e.g. 1e-4 → 2e-5 for exp 2+).

  --head            gaussian|student_t  Probabilistic head (default: gaussian)
  --scaling         standard|robust     Scaler type        (default: standard)
  --scale_mode      global|per_var      One scaler or one per column (default: global)

  --segment_isolated            Only create windows within contiguous valid segments.
  --gap_threshold   4           (non-isolated) time_since_valid > this filters windows.
  --segment_gap_minutes  60     Timestamp jump > this forces a new segment even within
                                flag=0. Catches unlogged calibration events and data gaps.
  --min_segment_rows  0         Discard segments shorter than this many rows.

  --transfer_from   path.pt     Load pretrained weights before training (manual transfer).
  --seed            42          Global random seed for reproducible plot windows.

Training strategy for batch reactor data
-----------------------------------------
Each batch experiment is an independent chemical process. Concatenating them creates
false temporal continuity at the boundaries.

Recommended workflow:

  1. SEQUENTIAL (preferred):
     --sequential --experiment KAU073 KAU074 KAU075 KAU084

     Trains KAU073 from scratch → checkpoint.
     Fine-tunes KAU074 from KAU073 checkpoint at lr * sequential_lr_scale.
     Fine-tunes KAU075 from KAU074 checkpoint at lr * sequential_lr_scale^2.
     Each experiment has its own scaler and output directory.

  2. SINGLE experiment:
     --experiment KAU084

  3. MANUAL transfer:
     --experiment KAU084 --transfer_from results/KAU073/best_model.pt --lr 2e-5

Gap handling
------------
Two kinds of gaps within an experiment:
  - Flagged gaps (flag=1): maintenance, cleaning. Already handled by segment_isolated.
  - Unflagged gaps: calibration jumps, short data outages, sensor resets.
    Use --segment_gap_minutes (e.g. 60) to split segments on large timestamp jumps.

With --segment_isolated --segment_gap_minutes 60:
  Windows only within contiguous, gap-free, flag=0 segments.
  Any gap > 60 min (flagged or not) forces a new segment boundary.

Usage examples
--------------
    # Sequential training (recommended for production)
    python train_reactor_probabilistic.py \\
        --sequential \\
        --experiment KAU073 KAU074 KAU075 KAU084 \\
        --endogenous nh4 --log_cols nh4 \\
        --data_paths dataset/final_dataset.csv dataset/final_dataset_2.csv \\
        --cheatsheet filtered_data_stamps.txt \\
        --head student_t --scaling standard --scale_mode per_var \\
        --segment_isolated --segment_gap_minutes 60 \\
        --d_model 128 --d_ff 512

    # Single experiment
    python train_reactor_probabilistic.py \\
        --experiment KAU084 \\
        --endogenous nh4 --log_cols nh4 no3 --exogenous no3 \\
        --data_paths dataset/final_dataset.csv dataset/final_dataset_2.csv \\
        --cheatsheet filtered_data_stamps.txt \\
        --head student_t --scaling standard --scale_mode per_var \\
        --segment_isolated --segment_gap_minutes 60 \\
        --d_model 128 --d_ff 512
"""
from __future__ import annotations

import argparse
import json
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
from torch.utils.data import Dataset, DataLoader


# ── Column layout ──────────────────────────────────────────────────────────
ALL_DATA_COLS = [
 #fill with data column titles
]


def resolve_columns(endogenous: List[str],
                    exogenous:  List[str]) -> Tuple[List[str], List[str]]:
    """
    Build (endo_cols, exo_cols) from user lists.

    endo_cols = exactly --endogenous
    exo_cols  = explicit --exogenous + leftovers (ALL_DATA_COLS not in either)
                + time_since_valid (always last)
    A variable can legally appear in both lists (used in both pathways).
    """
    if not endogenous:
        raise ValueError("--endogenous must list at least one variable.")
    bad = [c for c in endogenous if c not in ALL_DATA_COLS]
    if bad:
        raise ValueError(f"Unknown --endogenous: {bad}")
    bad = [c for c in exogenous if c not in ALL_DATA_COLS]
    if bad:
        raise ValueError(f"Unknown --exogenous: {bad}")
    endo_set = set(endogenous)
    exo_set  = set(exogenous)
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
    if not m:
        raise ValueError(f"Bad datetime token: {tok!r}")
    d, mon, h, mi = m.groups()
    mo = MONTHS.get(mon.lower())
    if mo is None:
        raise ValueError(f"Unknown month: {mon}")
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
        if ref:
            base = _adj_year(base, ref)
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
    if ref:
        s = _adj_year(s, ref)
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
            if not line:
                continue
            if _YEAR_PAT.match(line):
                cur_year = int(_YEAR_PAT.match(line).group(1)); in_excl = False; continue
            if _EXP_PAT.match(line):
                cur = Experiment(name=_EXP_PAT.match(line).group(1).upper())
                exps[cur.name] = cur; in_excl = False; continue
            if cur is None:
                continue
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
    """
    Load CSVs, filter to one or more experiment windows, apply exclusions.
    Multiple experiments are filtered independently then concatenated.
    In sequential mode this is called with a single tag per call.
    """
    exps = parse_cheatsheet(cheatsheet)
    parts = []
    for p in data_paths:
        parts.append(pd.read_csv(p, sep=';', decimal=',', low_memory=False))
    raw_df = pd.concat(parts, ignore_index=True)
    raw_df['time'] = pd.to_datetime(raw_df['time'], dayfirst=True, errors='coerce')
    raw_df = raw_df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)

    exp_frames = []
    for tag_raw in exp_tags:
        tag = tag_raw.upper()
        if tag not in exps:
            raise ValueError(f"Experiment {tag} not in cheatsheet. "
                             f"Available: {list(exps.keys())}")
        exp  = exps[tag]
        mask = (raw_df['time'] >= exp.start) & (raw_df['time'] <= exp.end)
        df_e = raw_df[mask].copy().reset_index(drop=True)
        if len(df_e) == 0:
            raise ValueError(f"No data for {tag} between {exp.start} and {exp.end}")
        print(f"  Experiment {tag}: {len(df_e):,} rows  [{exp.start} → {exp.end}]")
        for excl_s, excl_e in exp.excludes:
            em = (df_e['time'] >= excl_s) & (df_e['time'] <= excl_e)
            df_e.loc[em, 'flag'] = 1
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
    """Minutes since the last valid (flag=0) reading."""
    result = np.zeros(len(df), dtype=np.float32)
    last_valid_time = None
    for i, (ts, flag) in enumerate(zip(df['time'], df['flag'])):
        if flag == 0:
            if last_valid_time is not None:
                result[i] = float((ts - last_valid_time).total_seconds() / 60.0)
            last_valid_time = ts
    return result


def get_valid_segments(flags: np.ndarray,
                       times=None,
                       segment_gap_minutes: float = 60.0) -> List[Tuple[int, int]]:
    """
    Return (start_idx, end_idx) for each contiguous run of flag=0.

    When `times` is provided, also splits segments at any consecutive pair of
    flag=0 rows whose timestamp difference exceeds segment_gap_minutes.

    This handles:
    - Long data outages not reflected in the flag column
    - Sensor calibration events that reset the signal but aren't flagged
    - The boundary between concatenated experiments in joint mode
    """
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

    # Split on large timestamp jumps within flag=0 runs
    result: List[Tuple[int, int]] = []
    for seg_s, seg_e in segs:
        sub_s = seg_s
        for i in range(seg_s + 1, seg_e + 1):
            try:
                t_prev = pd.Timestamp(times[i - 1])
                t_curr = pd.Timestamp(times[i])
                gap_min = (t_curr - t_prev).total_seconds() / 60.0
            except Exception:
                gap_min = 0.0
            if gap_min > segment_gap_minutes:
                if i - 1 >= sub_s:
                    result.append((sub_s, i - 1))
                sub_s = i
        result.append((sub_s, seg_e))
    return result


# ── Scaling ────────────────────────────────────────────────────────────────
Scalers = Union[StandardScaler, RobustScaler, Dict[str, Union[StandardScaler, RobustScaler]]]


def make_scaler(scaling: str):
    return RobustScaler() if scaling == 'robust' else StandardScaler()


def fit_scalers(data: np.ndarray, cols: List[str],
                scaling: str, scale_mode: str) -> Scalers:
    if scale_mode == 'per_var':
        d: Dict[str, any] = {}
        for i, col in enumerate(cols):
            sc = make_scaler(scaling); sc.fit(data[:, i:i+1]); d[col] = sc
        return d
    sc = make_scaler(scaling); sc.fit(data); return sc


def apply_scalers(data: np.ndarray, scalers: Scalers,
                  cols: List[str], scale_mode: str) -> np.ndarray:
    if scale_mode == 'per_var':
        out = np.empty_like(data, dtype=np.float32)
        for i, col in enumerate(cols):
            out[:, i] = scalers[col].transform(data[:, i:i+1]).ravel()
        return out
    return scalers.transform(data).astype(np.float32)


def inverse_target(arr: np.ndarray, scalers: Scalers, target_col: str,
                   cols: List[str], scale_mode: str,
                   log_cols: Optional[Set[str]] = None) -> np.ndarray:
    """
    Inverse-transform predictions for the target column.
    Applies expm1 only if target_col is in log_cols (i.e. was log1p-transformed).
    """
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
    """
    Windowed dataset for reactor time series.

    segment_isolated=True (recommended for batch experiments):
      Windows only within contiguous, temporally-continuous flag=0 segments.
      Timestamp gaps > segment_gap_minutes force new segment boundaries even
      within flag=0 runs (catches unlogged calibration events / outages).

    segment_isolated=False:
      Windows across all flag=0 rows; filtered by time_since_valid > gap_threshold.
    """
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
            self.indices = []
            n_used = 0
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
                self.indices = [
                    i for i in self.indices
                    if not (time_since_valid[i: i + total_win] > gap_threshold).any()
                ]
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
    times = df['time'].values   # numpy datetime64 — passed to get_valid_segments

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
            tr_idx.extend(range(pos,              pos + n_tr))
            va_idx.extend(range(pos + n_tr,       pos + n_tr + n_va))
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
        tr_ds = _make_ds(endo_arr[:n_train],          exo_arr[:n_train],          flags[:n_train],          tsv[:n_train],          times[:n_train],          'train')
        va_ds = _make_ds(endo_arr[n_train:n_train+n_val], exo_arr[n_train:n_train+n_val], flags[n_train:n_train+n_val], tsv[n_train:n_train+n_val], times[n_train:n_train+n_val], 'val')
        te_ds = _make_ds(endo_arr[n_train+n_val:],    exo_arr[n_train+n_val:],    flags[n_train+n_val:],    tsv[n_train+n_val:],    times[n_train+n_val:],    'test')

    nw = min(0, __import__('os').cpu_count() or 1)
    kw = dict(num_workers=nw, pin_memory=False)
    if spike_oversample > 0.0:
        from torch.utils.data import WeightedRandomSampler
        tgt_idx = forecast_target_indices if forecast_target_indices is not None else list(range(len(endo_cols)))
        raw_scores = []
        for i in range(len(tr_ds)):
            x_endo, _, y = tr_ds[i]                           # [C,T], _, [n_forecast,H]
            ctx     = x_endo[tgt_idx]                         # [n_forecast, T]
            ctx_std = ctx.std().clamp(min=1e-3)               # variability of this window's context
            delta   = (y - ctx[:, -1:]).abs().max()           # max deviation from last context step
            rel_score = float(delta / ctx_std)                # catches small spikes on flat context
            abs_score = float(delta) * spike_abs_weight       # catches large trends/bumps
            raw_scores.append(rel_score + abs_score)
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
    """
    Reversible Instance Normalisation (Kim et al., 2022).

    Normalises each variable using its own context-window statistics so the
    downstream model is invariant to level shifts and slow distribution drift.

    Usage:
        revin = RevIN(c_endo)
        x_norm, (mean, std) = revin.normalise(x_endo)   # before encoder
        mean_denorm = revin.denormalise(mean_pred, mean, std)  # after head
    """
    def __init__(self, num_vars: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_vars, 1))
            self.beta  = nn.Parameter(torch.zeros(1, num_vars, 1))
        else:
            self.gamma = self.beta = None

    def normalise(self, x):
        # x: [B, C, T]
        mean = x.mean(dim=-1, keepdim=True)          # [B, C, 1]
        std  = x.std(dim=-1, keepdim=True).clamp(min=self.eps)
        x_n  = (x - mean) / std
        if self.gamma is not None:
            x_n = x_n * self.gamma + self.beta
        return x_n, (mean, std)

    def denormalise(self, x, mean, std):
        # x: [B, F, H]   mean/std: [B, F, 1]  (F = forecast targets subset of C)
        if self.gamma is not None:
            # Undo affine using the target-variable slice of gamma/beta
            F = x.shape[1]
            g = self.gamma[:, :F, :]
            b = self.beta[:, :F, :]
            x = (x - b) / g.clamp(min=self.eps)
        return x * std + mean


# ── Model building blocks ──────────────────────────────────────────────────

class _Chomp1d(nn.Module):
    def __init__(self, size): super().__init__(); self.size = size
    def forward(self, x): return x[..., :-self.size].contiguous() if self.size > 0 else x


class _TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                           dilation=dilation, padding=pad)),
            _Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                           dilation=dilation, padding=pad)),
            _Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)
    def forward(self, x):
        return self.norm((self.net(x) + x).transpose(1, 2)).transpose(1, 2)


class SharedTCN(nn.Module):
    def __init__(self, d_model, n_levels=3, kernel_size=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            _TCNBlock(d_model, kernel_size, 2 ** i, dropout) for i in range(n_levels)
        ])
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, d_model, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T = x.shape; n = T // self.patch_size
        return self.drop(self.norm(self.proj(
            x[:, :n * self.patch_size].reshape(B, n, self.patch_size))))


class _Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model); self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v):
        out, _ = self.attn(q, k, v)
        return self.norm(q + self.drop(out))


class _FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x): return self.norm(x + self.net(x))


class PerVarEncoderLayer(nn.Module):
    """
    Per-variable attention encoder layer.

    1. Intra-variable self-attention  – each variable's N patches attend to themselves
       (shared weights; variables processed in parallel via batch dimension).
    2. Global tokens cross-attend to pure endo patches, then to exo KV.
       Done before inter-variable mixing so each forecast target queries exogenous
       variables from its own uncontaminated perspective (e.g. NH4 asks "what
       temperature pattern matters for me?" before sharing information with NO3).
    3. Inter-variable cross-attention – each variable's patches attend to all other
       variables' patches (skipped when C_endo == 1).
    """
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.intra_attn = _Attention(d_model, n_heads, dropout)
        self.inter_attn = _Attention(d_model, n_heads, dropout)
        self.glob_endo  = _Attention(d_model, n_heads, dropout)
        self.glob_exo   = _Attention(d_model, n_heads, dropout)
        self.ffn_patch  = _FFN(d_model, d_ff, dropout)
        self.ffn_glob   = _FFN(d_model, d_ff, dropout)

    def forward(self, patches, global_toks, exo_kv):
        # patches:      [B, C, N, D]
        # global_toks:  [B, n_forecast, D]
        # exo_kv:       [B, C_exo * N, D]
        B, C, N, D = patches.shape

        # 1. Intra-variable self-attention (shared weights, each var independently)
        flat    = patches.reshape(B * C, N, D)
        flat    = self.intra_attn(flat, flat, flat)
        patches = flat.reshape(B, C, N, D)

        # 2. Global tokens attend to pure endo patches, then to exo.
        #    Patches are not yet cross-variate mixed, so each global token's
        #    exogenous query is uncontaminated by other variables.
        all_endo    = patches.reshape(B, C * N, D)
        global_toks = self.glob_endo(global_toks, all_endo, all_endo)
        global_toks = self.glob_exo(global_toks, exo_kv, exo_kv)

        # 3. Inter-variable cross-attention (each var queries from all others)
        if C > 1:
            new_patches = []
            for v in range(C):
                others_idx = [u for u in range(C) if u != v]
                kv = patches[:, others_idx, :, :].reshape(B, (C - 1) * N, D)
                q  = patches[:, v, :, :]                         # [B, N, D]
                new_patches.append(self.inter_attn(q, kv, kv))
            patches = torch.stack(new_patches, dim=1)            # [B, C, N, D]

        # FFN (shared weights across variables)
        patches     = self.ffn_patch(patches.reshape(B * C, N, D)).reshape(B, C, N, D)
        global_toks = self.ffn_glob(global_toks)
        return patches, global_toks


class ARHead(nn.Module):
    """
    Autoregressive GRU head.

    Generates H forecast steps one at a time.  Each step is conditioned on
    the previous predicted mean and a GRU hidden state initialised from the
    encoder's global token.  This lets the head produce varied, context-
    sensitive trajectories instead of a fixed MLP shape.

    start_val  [B, F]  – last observed context value for each forecast target.
    global_toks [B, F, D] – encoder global tokens (one per forecast target).

    Returns same format as the MLP head so the rest of the model is unchanged:
        gaussian   → (mean [B,F,H], log_var [B,F,H])
        student_t  → (mu,  log_sigma, log_nu)  all [B,F,H]
    """
    def __init__(self, d_model: int, horizon: int, head_type: str, dropout: float):
        super().__init__()
        self.horizon   = horizon
        self.head_type = head_type
        out_size = 2 if head_type == 'gaussian' else 3
        self.h_init   = nn.Sequential(nn.LayerNorm(d_model),
                                      nn.Linear(d_model, d_model),
                                      nn.Tanh())
        self.gru      = nn.GRUCell(1, d_model)
        self.out_proj = nn.Sequential(nn.LayerNorm(d_model),
                                      nn.Dropout(dropout),
                                      nn.Linear(d_model, out_size))
        self.sigma_step_bias = nn.Parameter(torch.zeros(horizon))
        nn.init.xavier_uniform_(self.h_init[1].weight)
        nn.init.zeros_(self.h_init[1].bias)
        nn.init.xavier_uniform_(self.out_proj[2].weight)
        nn.init.zeros_(self.out_proj[2].bias)

    def forward(self, global_toks: torch.Tensor, start_val: torch.Tensor):
        # global_toks: [B, F, D]   start_val: [B, F]
        B, F, D = global_toks.shape
        h    = self.h_init(global_toks).reshape(B * F, D)   # [B*F, D]
        anchor = start_val.reshape(B * F, 1)                # fixed — never updated
        prev   = torch.zeros_like(anchor)                   # GRU input: delta from anchor
        steps  = []
        for i in range(self.horizon):
            h      = self.gru(prev, h)                       # [B*F, D]
            out    = self.out_proj(h)                        # [B*F, out_size]
            # out[:,0] is a delta from the fixed anchor (last observed value)
            mean_t = anchor + out[:, :1]                     # [B*F, 1]
            scale  = out[:, 1:2] + self.sigma_step_bias[i]  # learnable per-step uncertainty offset
            steps.append(torch.cat([mean_t, scale, out[:, 2:]], dim=-1))
            prev   = out[:, :1]                              # feed delta back, not absolute
        steps = torch.stack(steps, dim=1).reshape(B, F, self.horizon, -1)  # [B,F,H,out_size]
        means = steps[..., 0]                                               # [B, F, H]
        if self.head_type == 'gaussian':
            return means, steps[..., 1]
        return means, steps[..., 1], steps[..., 2]


class ReactorProbModel(nn.Module):
    """
    Dual-pathway probabilistic encoder.

    Endogenous:  PatchEmbed → intra-attn → global queries pure patches + exo → inter-variable cross-attn (each layer)
    Exogenous:   PatchEmbed → SharedTCN → var_emb → KV for cross-attention
    Global toks: one per forecast target; cross-attend to endo patches + exo each layer
    Head:        gaussian → (mean, log_var)  |  student_t → (mu, log_sigma, log_nu)

    AR head (default):  ARHead generates steps autoregressively, seeded by the last
                        observed context value.  Produces varied trajectories and
                        preserves the residual anchor implicitly.
    MLP head:           simple Linear(d_model, H*params) + explicit residual anchor.
                        Enable with --no_ar_head.
    """
    def __init__(self, c_endo: int, c_exo: int, context_len: int, horizon: int,
                 n_forecast: int = 1,
                 forecast_target_indices: Optional[List[int]] = None,
                 head: str = 'gaussian',
                 patch_size: int = 12, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 512,
                 tcn_levels: int = 3, tcn_kernel: int = 3, dropout: float = 0.1,
                 use_ar_head: bool = True, use_revin: bool = True,
                 residual_anchor: bool = False,
                 use_batch_token: bool = False,
                 batch_token_idx: Optional[List[int]] = None,
                 raw_target_as_exo: bool = False):
        super().__init__()
        assert context_len % patch_size == 0, "context_len must be divisible by patch_size"
        self.horizon           = horizon
        self.d_model           = d_model
        self.n_patches         = context_len // patch_size
        self.head_type         = head
        self.c_endo            = c_endo
        self.n_forecast        = n_forecast
        self.use_ar_head       = use_ar_head
        self.use_revin         = use_revin
        self.residual_anchor   = residual_anchor
        self.use_batch_token   = use_batch_token
        self.batch_token_idx   = batch_token_idx
        self.raw_target_as_exo = raw_target_as_exo
        if use_revin:
            self.revin = RevIN(c_endo)
        self.forecast_target_indices = (list(range(n_forecast))
                                        if forecast_target_indices is None
                                        else list(forecast_target_indices))

        self.endo_patch    = PatchEmbed(patch_size, d_model, dropout)
        # One global token per forecast target
        self.global_tokens = nn.Parameter(torch.empty(1, n_forecast, d_model))
        # Positional embedding covers [global_tokens | all endo patches flattened]
        self.pos_emb       = nn.Parameter(torch.empty(1, n_forecast + self.n_patches * c_endo, d_model))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_emb,       std=0.02)
        # Marks the last context patch as the boundary adjacent to the forecast horizon
        self.last_patch_emb = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        if c_endo > 1:
            self.endo_var_emb = nn.Parameter(torch.empty(c_endo, 1, d_model))
            nn.init.trunc_normal_(self.endo_var_emb, std=0.02)

        self.exo_patch  = PatchEmbed(patch_size, d_model, dropout)
        self.shared_tcn = SharedTCN(d_model, tcn_levels, tcn_kernel, dropout)
        self.var_emb    = nn.Parameter(torch.empty(c_exo, 1, d_model))
        nn.init.trunc_normal_(self.var_emb, std=0.02)

        self.layers = nn.ModuleList([
            PerVarEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        if head not in ('gaussian', 'student_t'):
            raise ValueError(f"Unknown head: {head!r}")
        if use_ar_head:
            self.head = ARHead(d_model, horizon, head, dropout)
        else:
            head_out = horizon * 2 if head == 'gaussian' else horizon * 3
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, head_out))
            self.sigma_step_bias = nn.Parameter(torch.zeros(horizon))

        if use_batch_token and batch_token_idx:
            self.batch_proj = nn.Linear(len(batch_token_idx), d_model)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _encode_exo(self, x_exo):
        B, C, T = x_exo.shape; N = self.n_patches
        patches = self.exo_patch(x_exo.reshape(B * C, T))
        tcn_out = self.shared_tcn(patches.transpose(1, 2)).transpose(1, 2)
        var_id  = self.var_emb.expand(C, N, -1).unsqueeze(0).expand(B, -1, -1, -1)
        return (tcn_out + var_id.reshape(B * C, N, self.d_model)).reshape(B, C * N, self.d_model)

    def _encode_endo(self, x_endo):
        B, C, T = x_endo.shape; N = self.n_patches
        patches = self.endo_patch(x_endo.reshape(B * C, T))        # [B*C, N, D]
        if C > 1:
            var_id  = self.endo_var_emb.expand(C, N, -1).unsqueeze(0).expand(B, -1, -1, -1)
            patches = patches + var_id.reshape(B * C, N, self.d_model)
        patches = patches.reshape(B, C, N, self.d_model)           # [B, C, N, D]
        patches[:, :, -1, :] = patches[:, :, -1, :] + self.last_patch_emb
        return patches

    def forward(self, x_endo, x_exo):
        B = x_endo.shape[0]
        C = self.c_endo
        N = self.n_patches

        x_endo_raw = x_endo  # save pre-RevIN global-scaler values for raw_target_as_exo

        # RevIN: normalise each variable by its own context statistics
        if self.use_revin:
            x_endo, (rv_mean, rv_std) = self.revin.normalise(x_endo)
            rv_mean_tgt = rv_mean[:, self.forecast_target_indices, :]        # [B, F, 1]
            rv_std_tgt  = rv_std[:, self.forecast_target_indices, :]

        if self.raw_target_as_exo:
            raw_tgt = x_endo_raw[:, self.forecast_target_indices, :]        # [B, n_f, T]
            x_exo = torch.cat([x_exo, raw_tgt], dim=1)

        exo_kv  = self._encode_exo(x_exo)                                   # [B, C_exo*N, D]
        patches = self._encode_endo(x_endo)                                  # [B, C, N, D]

        # Apply positional embedding to [global_tokens | flattened patches]
        glob = self.global_tokens.expand(B, -1, -1)                          # [B, n_forecast, D]
        flat = patches.reshape(B, C * N, self.d_model)                       # [B, C*N, D]
        seq  = torch.cat([glob, flat], dim=1) + self.pos_emb                 # [B, n_forecast+C*N, D]
        glob    = seq[:, :self.n_forecast]                                   # [B, n_forecast, D]
        patches = seq[:, self.n_forecast:].reshape(B, C, N, self.d_model)   # [B, C, N, D]

        if self.use_batch_token and self.batch_token_idx is not None:
            scalars = x_exo[:, self.batch_token_idx, 0]                      # [B, 2]
            glob = glob + self.batch_proj(scalars).unsqueeze(1)              # broadcast [B,1,D]

        for layer in self.layers:
            patches, glob = layer(patches, glob, exo_kv)

        last_target = x_endo[:, self.forecast_target_indices, -1]            # [B, F] in normalised space
        if self.use_ar_head:
            out = self.head(glob, last_target)
        else:
            raw  = self.head(glob)
            H    = self.horizon
            mean = raw[:, :, :H]
            if self.residual_anchor:
                mean = mean + last_target.unsqueeze(-1)
            if self.head_type == 'gaussian':
                out = (mean, raw[:, :, H:] + self.sigma_step_bias)
            else:
                out = (mean, raw[:, :, H:2*H] + self.sigma_step_bias, raw[:, :, 2*H:])

        # RevIN: denormalise mean/mu back to original scale
        if self.use_revin:
            out = (self.revin.denormalise(out[0], rv_mean_tgt, rv_std_tgt),) + out[1:]

        return out


# ── Loss functions ─────────────────────────────────────────────────────────
LOG_VAR_MIN, LOG_VAR_MAX = -4.0, 4.0
LOG_SIG_MIN, LOG_SIG_MAX = -4.0, 2.0


def gaussian_nll(mean, log_var, target):
    lv = log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    return 0.5 * (lv + (target - mean).pow(2) / lv.exp()).mean()


def student_t_nll(mu, log_sigma, log_nu, target):
    ls    = log_sigma.clamp(LOG_SIG_MIN, LOG_SIG_MAX)
    nu    = F.softplus(log_nu) + 2.0    # nu > 2 guarantees finite variance
    sigma = torch.exp(ls)               # use clamped ls consistently
    z_sq  = ((target - mu) / sigma) ** 2
    nll   = (ls
             + 0.5 * (nu * float(torch.pi)).log()
             + torch.lgamma(nu / 2)
             - torch.lgamma((nu + 1) / 2)
             + (nu + 1) / 2 * torch.log(1 + z_sq / nu))
    return nll.mean()


def compute_loss(model_out, target, head: str,
                 peak_weight: float = 0.0) -> torch.Tensor:
    if head == 'gaussian':
        nll = gaussian_nll(model_out[0], model_out[1], target)
    else:
        nll = student_t_nll(model_out[0], model_out[1], model_out[2], target)
    if peak_weight > 0.0:
        mean     = model_out[0]
        peak_mse = (mean.max(dim=-1).values - target.max(dim=-1).values).pow(2).mean()
        return nll + peak_weight * peak_mse
    return nll


# ── Training helpers ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, head, peak_weight=0.0):
    model.train(); total, n = 0.0, 0
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        optimizer.zero_grad()
        loss = compute_loss(model(x_endo, x_exo), y, head, peak_weight)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * x_endo.size(0); n += x_endo.size(0)
    return total / n if n > 0 else float('nan')


@torch.no_grad()
def eval_epoch(model, loader, device, head, peak_weight=0.0):
    model.eval(); total, n = 0.0, 0
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        loss = compute_loss(model(x_endo, x_exo), y, head, peak_weight)
        total += loss.item() * x_endo.size(0); n += x_endo.size(0)
    return total / n if n > 0 else float('nan')


@torch.no_grad()
def collect_predictions(model, loader, device, head):
    model.eval(); means, scales, targets, nus = [], [], [], []
    for x_endo, x_exo, y in loader:
        out = model(x_endo.to(device), x_exo.to(device))
        means.append(out[0].cpu().numpy())
        scales.append(out[1].cpu().numpy())
        targets.append(y.numpy())
        if head == 'student_t':
            nus.append(out[2].cpu().numpy())
    nu_preds = np.concatenate(nus) if nus else None
    return (np.concatenate(means), np.concatenate(scales),
            np.concatenate(targets), nu_preds)


def spike_window_mask(dataset, forecast_target_indices: List[int],
                      threshold: float) -> np.ndarray:
    """
    Returns a boolean array of length len(dataset) marking windows whose
    forecast horizon contains a spike relative to the last context value.

    A window is a spike window when:
        max_over_horizon( |y - last_ctx_val| ) / ctx_std  >=  threshold

    This is the same relative-delta score used for spike oversampling, so
    the threshold has the same meaning: ~1.0 catches anything that moves
    more than one std-dev relative to the context variability.
    """
    mask = np.zeros(len(dataset), dtype=bool)
    for i in range(len(dataset)):
        x_endo, _, y = dataset[i]
        ctx      = x_endo[forecast_target_indices]          # [F, T]
        ctx_std  = float(ctx.std().clamp(min=1e-3))
        last_val = ctx[:, -1:]                              # [F, 1]
        delta    = float((y - last_val).abs().max())
        mask[i]  = (delta / ctx_std) >= threshold
    return mask


def _print_point_metrics(label: str, means_o, tgts_o, means_sc, tgts_sc,
                         target_cols: List[str]) -> dict:
    """Compute and print MAE/RMSE/MSE for one split. Returns metrics dict."""
    N = means_o.shape[0]
    per_ch = {}
    print(f"\n  Point forecast — {label}  (n={N}):")
    for ci, col in enumerate(target_cols):
        if N == 0:
            print(f"    [{col}]  (no windows)")
            per_ch[col] = {}
            continue
        mae  = float(np.mean(np.abs(means_o[:, ci, :] - tgts_o[:, ci, :])))
        rmse = float(np.sqrt(np.mean((means_o[:, ci, :] - tgts_o[:, ci, :]) ** 2)))
        mse  = float(np.mean((means_sc[:, ci, :] - tgts_sc[:, ci, :]) ** 2))
        print(f"    [{col}]  MSE(scaled)={mse:.5f}  MAE={mae:.4f}  RMSE={rmse:.4f} mg/L")
        per_ch[col] = {'mae': mae, 'rmse': rmse, 'mse_scaled': mse}
    return per_ch


# ── Calibration ────────────────────────────────────────────────────────────

def coverage_report(means, scales, targets, head, nu_preds=None):
    from scipy.stats import t as _st, norm as _norm
    if head == 'gaussian':
        stds = np.exp(0.5 * np.clip(scales, LOG_VAR_MIN, LOG_VAR_MAX))
    else:
        stds = np.exp(np.clip(scales, LOG_SIG_MIN, LOG_SIG_MAX))
    results = {}
    levels = [('68%', 0.84), ('90%', 0.95), ('95%', 0.975)]
    if head == 'student_t' and nu_preds is not None:
        nu = np.log1p(np.exp(np.clip(nu_preds, -20.0, 20.0))) + 2.0
        for label, p in levels:
            q = _st.ppf(p, df=nu)
            results[label] = float((np.abs(targets - means) <= q * stds).mean())
    else:
        for label, p in levels:
            results[label] = float((np.abs(targets - means) <= _norm.ppf(p) * stds).mean())
    return results


# ── Plotting ───────────────────────────────────────────────────────────────

def _spike_scores_for_candidates(candidates, endo_te_sc, target_channel_idx,
                                  context_len, horizon):
    """Return array of relative-delta spike scores, one per candidate start index."""
    scores = np.zeros(len(candidates), dtype=np.float32)
    for j, s in enumerate(candidates):
        ctx   = endo_te_sc[s:s + context_len, target_channel_idx]
        hor   = endo_te_sc[s + context_len:s + context_len + horizon, target_channel_idx]
        std   = float(np.std(ctx)) or 1e-3
        delta = float(np.max(np.abs(hor - ctx[-1])))
        scores[j] = delta / std
    return scores


def plot_horizon(model, sc_endo, sc_exo, df, endo_cols, exo_cols,
                 scale_mode, context_len, horizon, out_path, target_col,
                 log_cols=None, n_windows=6, context_show=60, seed=42,
                 val_frac=0.1, test_frac=0.1, head='gaussian',
                 target_channel_idx: int = 0,
                 n_spike_windows: int = 6, spike_threshold: float = 1.0):
    device = next(model.parameters()).device
    T = len(df); n_test = int(T * test_frac); n_val = int(T * val_frac)
    n_train = T - n_val - n_test
    flags   = df['flag'].values.astype(int)

    endo_te_raw = df[target_col].values[n_train + n_val:]   # in log space if applicable
    endo_te_sc  = apply_scalers(df[endo_cols].values[n_train + n_val:], sc_endo, endo_cols, scale_mode)
    exo_te_sc   = apply_scalers(df[exo_cols ].values[n_train + n_val:], sc_exo,  exo_cols,  scale_mode)
    flags_te    = flags[n_train + n_val:]

    valid_te   = np.where(flags_te == 0)[0]
    candidates = [i for i in valid_te if i + context_len + horizon < len(endo_te_sc)]
    if not candidates:
        print("  No valid windows for plotting."); return
    n_windows = min(n_windows, len(candidates))

    rng    = np.random.default_rng(seed)   # fixed seed = reproducible plot windows
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
                out = model(x_e, x_x)
            mean_sc  = out[0][0, target_channel_idx].cpu().numpy()
            scale_sc = out[1][0, target_channel_idx].cpu().numpy()
            if head == 'gaussian':
                std_sc = np.exp(0.5 * np.clip(scale_sc, LOG_VAR_MIN, LOG_VAR_MAX))
                z_1s, z_95 = 1.0, 1.96
            else:
                from scipy.stats import t as _st
                std_sc = np.exp(np.clip(scale_sc, LOG_SIG_MIN, LOG_SIG_MAX))
                log_nu_sc = out[2][0, target_channel_idx].cpu().numpy()
                nu_med = float(np.median(np.log1p(np.exp(np.clip(log_nu_sc, -20.0, 20.0))) + 2.0))
                z_1s = float(_st.ppf(0.84, df=nu_med))
                z_95  = float(_st.ppf(0.975, df=nu_med))
            inv = lambda a: inverse_target(a, sc_endo, target_col, endo_cols, scale_mode, log_cols)
            mean_orig = inv(mean_sc)
            upper_1s  = inv(mean_sc + z_1s * std_sc); lower_1s = inv(mean_sc - z_1s * std_sc)
            upper_95  = inv(mean_sc + z_95 * std_sc); lower_95 = inv(mean_sc - z_95 * std_sc)
            ctx_sl    = endo_te_raw[s + context_len - context_show : s + context_len]
            actual_sl = endo_te_raw[s + context_len : s + context_len + horizon]
            if log_cols and target_col in log_cols:
                ctx_raw = np.expm1(ctx_sl); actual_raw = np.expm1(actual_sl)
            else:
                ctx_raw = ctx_sl; actual_raw = actual_sl
            ax = axes[row][0]
            ax.plot(ctx_x,  ctx_raw,   color='#888888', lw=1.2, alpha=0.7, label='Context')
            ax.fill_between(fore_x, lower_95, upper_95, color='#ff7f0e', alpha=0.15, label='95% PI')
            ax.fill_between(fore_x, lower_1s, upper_1s, color='#ff7f0e', alpha=0.30, label='68% PI')
            ax.plot(fore_x, mean_orig,  color='#ff7f0e', lw=2.0, label='Mean forecast')
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
        fig.suptitle(f'{target_col.upper()} {head} forecast  endo={endo_cols}  {title_suffix}\n'
                     f'{horizon}-step horizon  seed={seed}', fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved: {save_path}")

    model.eval()
    _render_windows(starts, out_path)

    if n_spike_windows > 0 and spike_starts:
        spike_path = out_path.parent / (out_path.stem + '_spikes' + out_path.suffix)
        _render_windows(spike_starts, spike_path, title_suffix='[spike windows]')


# ── Core training function (called once per experiment or once jointly) ────

def _freeze_for_finetune(model: nn.Module, n_layers: int = 1) -> None:
    """Freeze all parameters, then unfreeze head + last n_layers encoder layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    # Unfreeze head
    for attr in ('head', 'head_net', 'output_proj'):
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters():
                p.requires_grad_(True)
    # Unfreeze last n_layers of encoder
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
    """Print aggregate LOO metrics across held-out experiments."""
    rmses, nlls = [], []
    for tag, res in all_results.items():
        m = res.get('metrics', {})
        if 'rmse' in m:
            rmses.append(m['rmse'])
        nlls.append(res.get('test_nll_scaled', float('nan')))
    print(f"\n{'='*60}")
    print(f"  LOO Summary  ({len(all_results)} folds)")
    if rmses:
        print(f"  RMSE  mean={float(np.mean(rmses)):.4f}  std={float(np.std(rmses)):.4f}")
    import math
    valid_nlls = [v for v in nlls if not math.isnan(v)]
    if valid_nlls:
        print(f"  NLL   mean={float(np.mean(valid_nlls)):.4f}  std={float(np.std(valid_nlls)):.4f}")
    print(f"{'='*60}")


def train_experiment(args, exp_tags, device, endo_cols, exo_cols,
                     target_cols, log_cols, out_dir,
                     transfer_from=None, lr_override=None,
                     eval_exp_tags=None, freeze_layers=0):
    """
    Load data, build dataloaders, train and evaluate one experiment set.
    Returns (best_model_path, results_dict). Returns (None, {}) on empty data.

    target_cols: list of variable names to forecast (subset of endo_cols).
    In sequential mode, called once per experiment with transfer_from pointing
    to the previous experiment's checkpoint and a reduced lr_override.
    """
    forecast_target_indices = [endo_cols.index(t) for t in target_cols]
    n_forecast = len(target_cols)
    lr = lr_override if lr_override is not None else args.lr
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Loading data ─────────────────────────────────────────────────")
    df = load_experiment(args.data_paths, args.cheatsheet, exp_tags)
    df['time_since_valid'] = compute_time_since_valid(df)

    # Apply log1p to specified columns (in-place on df, so both endo_arr and
    # exo_arr will pick up the transform if the column appears in both paths)
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
    _c_exo = len(exo_cols) + (n_forecast if args.raw_target_as_exo else 0)
    model = ReactorProbModel(
        c_endo=len(endo_cols), c_exo=_c_exo,
        context_len=args.context_len, horizon=args.horizon,
        n_forecast=n_forecast,
        forecast_target_indices=forecast_target_indices,
        head=args.head,
        patch_size=args.patch_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff,
        tcn_levels=args.tcn_levels, tcn_kernel=args.tcn_kernel, dropout=args.dropout,
        use_ar_head=not args.no_ar_head,
        use_revin=not args.no_revin,
        residual_anchor=args.residual_anchor,
        use_batch_token=args.batch_token,
        batch_token_idx=batch_token_idx,
        raw_target_as_exo=args.raw_target_as_exo,
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

    # Memory probe
    try:
        xe = torch.randn(min(4, args.batch_size), len(endo_cols), args.context_len, device=device)
        xx = torch.randn(min(4, args.batch_size), len(exo_cols),  args.context_len, device=device)
        out_p = model(xe, xx)
        probe_tgt = torch.zeros(min(4, args.batch_size), n_forecast, args.horizon, device=device)
        compute_loss(out_p, probe_tgt, args.head, args.peak_weight).backward()
        model.zero_grad()
        mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else 0.0
        print(f"  Memory probe: {mem:.1f} MB  shape={list(out_p[0].shape)}  ✓")
    except RuntimeError as e:
        print(f"  Memory probe FAILED: {e}"); return None, {}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr * 0.01)

    print(f"\n── Training up to {args.epochs} epochs  (patience={args.patience}) ─")
    history = {'train': [], 'val': []}
    best_val, patience_cnt = float('inf'), 0
    best_path = out_dir / 'best_model.pt'

    for epoch in range(1, args.epochs + 1):
        t0      = time.time()
        tr_loss = train_epoch(model, train_dl, optimizer, device, args.head, args.peak_weight)
        va_loss = eval_epoch (model, val_dl,   device,    args.head, args.peak_weight)
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
                  f"NLL train={tr_loss:.5f}  val={va_loss:.5f}  "
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

    test_nll = eval_epoch(model, test_dl, device, args.head, args.peak_weight)
    print(f"  Test NLL (scaled): {test_nll:.5f}")

    results = {
        'experiments': exp_tags, 'target': target_cols,
        'endogenous': endo_cols, 'exogenous': exo_cols,
        'log_cols': list(log_cols), 'lr_used': lr,
        'test_nll_scaled': float(test_nll),
        'history': history, 'n_params': n_params, 'args': vars(args),
    }

    if len(test_dl.dataset) > 0:
        means, scale_preds, tgts, nu_preds = collect_predictions(model, test_dl, device, args.head)
        N, C_out, H = means.shape

        # Inverse-transform all predictions and targets to original units once
        m_orig = np.stack([
            inverse_target(means[:,ci,:].reshape(-1), sc_endo, col,
                           endo_cols, args.scale_mode, log_cols).reshape(N, H)
            for ci, col in enumerate(target_cols)], axis=1)   # [N, C_out, H]
        t_orig = np.stack([
            inverse_target(tgts[:,ci,:].reshape(-1), sc_endo, col,
                           endo_cols, args.scale_mode, log_cols).reshape(N, H)
            for ci, col in enumerate(target_cols)], axis=1)   # [N, C_out, H]

        # ── Overall metrics ───────────────────────────────────────────────
        per_channel_metrics = _print_point_metrics(
            f"all windows ({C_out} channel(s))",
            m_orig, t_orig, means, tgts, target_cols)

        # ── Spike-isolated metrics ────────────────────────────────────────
        smask = spike_window_mask(test_dl.dataset, forecast_target_indices,
                                  args.spike_metric_threshold)
        n_spike = int(smask.sum()); n_flat = N - n_spike
        print(f"\n  Spike isolation — threshold={args.spike_metric_threshold}  "
              f"spike_windows={n_spike}/{N}  ({100*n_spike/max(N,1):.1f}%)")

        spike_metrics = _print_point_metrics(
            "spike windows", m_orig[smask], t_orig[smask],
            means[smask], tgts[smask], target_cols) if n_spike else {}

        flat_metrics = _print_point_metrics(
            "non-spike windows", m_orig[~smask], t_orig[~smask],
            means[~smask], tgts[~smask], target_cols) if n_flat else {}

        # ── Calibration (overall) ─────────────────────────────────────────
        cov = coverage_report(means, scale_preds, tgts, args.head, nu_preds)
        print(f"\n  Calibration (all windows):")
        for label, frac in cov.items():
            nom  = float(label[:2]) / 100
            flag = '✓' if abs(frac - nom) < 0.08 else ('↓ overconfident' if frac < nom else '↑ conservative')
            print(f"    {label}: {frac:.1%}  {flag}")

        if n_spike:
            cov_spike = coverage_report(means[smask], scale_preds[smask],
                                        tgts[smask], args.head,
                                        nu_preds[smask] if nu_preds is not None else None)
            print(f"\n  Calibration (spike windows):")
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

        # ── Per-experiment metrics ─────────────────────────────────────────
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
                m_e, sp_e, t_e, nu_e = collect_predictions(model, te_e, device, args.head)
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

        # Loss curve
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history['train'], label='Train NLL')
        ax.plot(history['val'],   label='Val NLL')
        ax.set_yscale('symlog'); ax.set_xlabel('Epoch'); ax.set_ylabel('NLL')
        ax.set_title(f"{'+'.join(exp_tags)} – {args.head} history")
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
                             val_frac=args.val_frac, test_frac=args.test_frac, head=args.head,
                             target_channel_idx=ti,
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
    torch.backends.cuda.matmul.allow_tf32   = True
    torch.backends.cudnn.allow_tf32         = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  mode={'sequential' if args.sequential else 'joint'}  "
          f"head={args.head}  scaling={args.scaling}/{args.scale_mode}  seed={args.seed}")

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
        # ── Leave-one-out mode ─────────────────────────────────────────────
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
        # ── Pre-train + fine-tune mode ─────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Fine-tune: pre-train on {args.experiment[:-1]}  →  fine-tune on {args.experiment[-1]}")
        print(f"  Freeze all except head + last {args.finetune_layers} encoder layer(s)  |  LR={args.finetune_lr:.1e}")
        print(f"{'='*60}")
        pretrain_tags = args.experiment[:-1]
        finetune_tag  = args.experiment[-1]
        # Pre-train on all but last experiment
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
        # Fine-tune on last experiment
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
        # ── Sequential mode ────────────────────────────────────────────────
        # Each experiment trained independently with its own scaler.
        # Checkpoint passes forward; LR decays each step.
        print(f"\n{'='*60}")
        print(f"  Sequential: {args.experiment}")
        print(f"  LR schedule: {args.lr:.1e} × {args.sequential_lr_scale}^i")
        print(f"{'='*60}")

        checkpoint   = args.transfer_from
        all_results  = {}

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
                checkpoint = str(best_path)
                all_results[exp_tag] = results
            else:
                print(f"  [{exp_tag}] skipped — keeping previous checkpoint")

        summary = base_out / 'sequential_summary.json'
        with open(summary, 'w') as f:
            json.dump({'experiments': args.experiment, 'endo_cols': endo_cols,
                       'log_cols': list(log_cols), 'target': target_cols,
                       'results': all_results}, f, indent=2)
        print(f"\nSequential training complete.")
        print(f"Summary:          {summary}")
        print(f"Final checkpoint: {checkpoint}")

    else:
        # ── Joint mode ─────────────────────────────────────────────────────
        exp_slug = '_'.join(e.upper() for e in args.experiment)
        if len(args.experiment) > 1:
            print(f"\nNOTE: {len(args.experiment)} experiments will be CONCATENATED.")
            print("Use --sequential for independent batch experiments.")
        train_experiment(
            args, args.experiment, device,
            endo_cols, exo_cols, target_cols, log_cols,
            base_out / exp_slug,
            transfer_from=args.transfer_from,
        )

    print("\nDone.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description='Probabilistic forecast on reactor batch data.')

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument('--experiment',  nargs='+', required=True)
    p.add_argument('--data_paths',  nargs='+', required=True)
    p.add_argument('--cheatsheet',  default='filtered_data_stamps.txt')

    # ── Column routing ─────────────────────────────────────────────────────
    p.add_argument('--endogenous', nargs='+', default=['nh4'])
    p.add_argument('--exogenous',  nargs='*', default=[])
    p.add_argument('--target',     nargs='+', default=None,
                   help='Variable(s) to forecast. Must be in --endogenous. '
                        'Defaults to the first --endogenous variable. '
                        'Example: --target nh4 no3  (coupled two-signal forecast).')
    p.add_argument('--log_cols',   nargs='*', default=None,
                   help='Columns to log1p-transform before scaling. '
                        'Defaults to --endogenous. '
                        'Pass explicitly to include exo vars, e.g. --log_cols nh4 no3.')

    # ── Training mode ──────────────────────────────────────────────────────
    p.add_argument('--sequential',  action='store_true',
                   help='Train experiments one at a time; pass checkpoint forward.')
    p.add_argument('--sequential_lr_scale', type=float, default=0.2,
                   help='LR multiplier per experiment in sequential mode. '
                        'Exp i uses lr * scale^i  (default: 1e-4, 2e-5, 4e-6, …)')
    p.add_argument('--leave_one_out', action='store_true',
                   help='LOO evaluation: train on N-1 experiments, test on held-out. '
                        'Rotates through all --experiment tags.')
    p.add_argument('--finetune', action='store_true',
                   help='Pre-train on all but last experiment, then fine-tune on last. '
                        'Freezes all layers except head + last --finetune_layers encoder layers.')
    p.add_argument('--finetune_layers', type=int, default=1,
                   help='Number of last encoder layers to keep trainable during fine-tune.')
    p.add_argument('--finetune_lr', type=float, default=1e-5,
                   help='Learning rate for fine-tune phase.')
    p.add_argument('--finetune_spike_oversample', type=float, default=None,
                   help='spike_oversample for fine-tune phase. Defaults to --spike_oversample.')
    p.add_argument('--finetune_spike_abs_weight', type=float, default=None,
                   help='spike_abs_weight for fine-tune phase. Defaults to --spike_abs_weight.')
    p.add_argument('--batch_token', action='store_true',
                   help='Project inoc_amount+inoc_conc via Linear(2,d_model) and add to global token.')
    p.add_argument('--batch_token_cols', nargs='+',
                   default=['inoc_amount', 'inoc_conc'],
                   help='Exo column names to use as batch token scalars (default: inoc_amount inoc_conc).')

    # ── Head ──────────────────────────────────────────────────────────────
    p.add_argument('--head', default='gaussian', choices=['gaussian', 'student_t'])

    # ── Scaling ───────────────────────────────────────────────────────────
    p.add_argument('--scaling',    default='standard', choices=['standard', 'robust'])
    p.add_argument('--scale_mode', default='global',   choices=['global', 'per_var'])

    # ── Gap / segmentation ────────────────────────────────────────────────
    p.add_argument('--segment_isolated',  action='store_true',
                   help='Only create windows within contiguous valid segments. '
                        'Recommended for short batch experiments.')
    p.add_argument('--gap_threshold',     type=float, default=4.0,
                   help='(Non-isolated) time_since_valid > this filters windows.')
    p.add_argument('--segment_gap_minutes', type=float, default=60.0,
                   help='(Isolated) Timestamp jump > this forces a new segment boundary '
                        'even within flag=0. Catches unlogged calibration events. '
                        'Typical: 30–120 min.')
    p.add_argument('--min_segment_rows',  type=int, default=0,
                   help='Discard segments shorter than this many rows.')
    p.add_argument('--spike_oversample', type=float, default=0.0,
                   help='Oversample spike windows during training. '
                        '0=off. Controls the power applied to the combined score. '
                        'Start with 1.5.')
    p.add_argument('--spike_abs_weight', type=float, default=1.0,
                   help='Weight for the absolute-delta component of the spike score. '
                        'Higher values upsample large trends/bumps more strongly '
                        'relative to small spikes on flat regions.')
    p.add_argument('--n_spike_windows', type=int, default=6,
                   help='Number of top-spike windows to plot separately after inference, '
                        'saved as horizon_forecast_<col>_spikes.png. 0 to disable.')
    p.add_argument('--spike_metric_threshold', type=float, default=1.0,
                   help='Relative-delta threshold for classifying a test window as a '
                        'spike window during evaluation. A window is a spike window when '
                        'max(|horizon - last_ctx|) / ctx_std >= this value. '
                        'Same scale as --spike_oversample scoring. Default: 1.0.')
    p.add_argument('--peak_weight', type=float, default=0.0,
                   help='Weight for the peak-matching auxiliary loss. '
                        'Penalises the gap between max(predicted mean) and max(target) '
                        'across the horizon. 0=off. Start with 0.1–0.5.')
    p.add_argument('--no_ar_head', action='store_true',
                   help='Disable the autoregressive GRU head and fall back to a simple '
                        'MLP head that decodes all horizon steps at once. '
                        'AR head is on by default; use this flag to compare.')
    p.add_argument('--residual_anchor', action='store_true',
                   help='(MLP head only) Add the last observed context value to the mean '
                        'output, so the model predicts residuals. Default: off (absolute '
                        'prediction, same as iTransformer / PatchTST).')
    p.add_argument('--no_revin', action='store_true',
                   help='Disable Reversible Instance Normalisation. RevIN is on by default '
                        'and removes per-window level/scale from the input before encoding, '
                        'then restores it on the output. Helps with distribution shift '
                        'across experiments or slow baseline drift within a run.')
    p.add_argument('--raw_target_as_exo', action='store_true',
                   help='Append raw (global-scaler) target channel(s) as extra exo inputs, '
                        'giving the model both RevIN-normalised shape and absolute level.')

    # ── Transfer learning ─────────────────────────────────────────────────
    p.add_argument('--transfer_from', default=None,
                   help='Checkpoint for manual warm-start. In --sequential mode '
                        'only applies to the first experiment.')

    # ── Reproducibility ───────────────────────────────────────────────────
    p.add_argument('--seed', type=int, default=42)

    # ── Architecture ──────────────────────────────────────────────────────
    p.add_argument('--context_len', type=int,   default=360)
    p.add_argument('--horizon',     type=int,   default=72)
    p.add_argument('--stride',      type=int,   default=1)
    p.add_argument('--patch_size',  type=int,   default=12)
    p.add_argument('--d_model',     type=int,   default=128)
    p.add_argument('--n_heads',     type=int,   default=4)
    p.add_argument('--n_layers',    type=int,   default=2)
    p.add_argument('--d_ff',        type=int,   default=512)
    p.add_argument('--tcn_levels',  type=int,   default=3)
    p.add_argument('--tcn_kernel',  type=int,   default=3)
    p.add_argument('--dropout',     type=float, default=0.1)

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--patience',    type=int,   default=10)
    p.add_argument('--val_frac',    type=float, default=0.1)
    p.add_argument('--test_frac',   type=float, default=0.1)

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument('--out_dir', default='results/reactor_probabilistic')

    main(p.parse_args())
