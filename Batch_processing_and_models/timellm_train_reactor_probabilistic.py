"""
timellm_train_reactor_probabilistic.py

Probabilistic TimeLLM for reactor batch time-series forecasting.

Architecture
------------
  x_endo [B, C_e, T]  ┐
  x_exo  [B, C_x, T]  ┘ → concat all channels → [B, C_total, T]
                           ↓  patching: [B, C_total, n_patches, patch_size]
                           ↓  flatten channels: [B*C_total, n_patches, patch_size]
                           ↓  patch projection → [B*C_total, n_patches, d_llm]
                           ↓  reprogramming (cross-attn to word tokens)
                           ↓  frozen GPT-2 backbone
                           ↓  reshape [B, C_total, n_patches, d_llm]
                           ↓  select forecast target channels
                       output_proj → (mu, log_sigma, log_nu) per target × horizon

Same data pipeline, evaluation, and CLI as all companion model scripts.

Reference: Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming
           Large Language Models" (ICLR 2024).
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

try:
    from transformers import GPT2Model, GPT2Tokenizer
except ImportError:
    raise ImportError("transformers library required: pip install transformers")


# ── Column layout ──────────────────────────────────────────────────────────
ALL_DATA_COLS = [
    'nh4', 'no3', 'flow_ch4', 'norm_ch4', 'temp', 'K', 'nh3',
    'subs1', 'subs2', 'supern', 'ph', 'ttpres', 'o2_sp', 'do',
    'inoc_am', 'inoc_conc', 'hours_elapsed', 'air_sp',
    'n2_norm', 'eth_norm', 'co2_norm', 'harvest',
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
            time_since_valid=tsv_s, times=times_s,
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

    kw = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    if spike_oversample > 0.0:
        from torch.utils.data import WeightedRandomSampler
        tgt_idx = forecast_target_indices if forecast_target_indices is not None else list(range(len(endo_cols)))
        raw_scores = []
        for i in range(len(tr_ds)):
            x_endo, _, y = tr_ds[i]
            ctx     = x_endo[tgt_idx]
            ctx_std = ctx.std().clamp(min=1e-3)
            delta   = (y - ctx[:, -1:]).abs().max()
            rel_score = float(delta / ctx_std)
            abs_score = float(delta) * spike_abs_weight
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
    val_dl   = DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw)
    test_dl  = DataLoader(te_ds, batch_size=batch_size, shuffle=False, **kw)

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


# ── TimeLLM model ──────────────────────────────────────────────────────────

class ReprogrammingLayer(nn.Module):
    """
    Cross-attention reprogramming: patches (queries) attend to learned source
    word tokens (keys/values), translating time-series patches into the LLM's
    representation space.
    """
    def __init__(self, d_model: int, n_heads: int, n_source: int = 512,
                 wte: Optional[torch.Tensor] = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        if wte is not None:
            idx = torch.randperm(wte.shape[0])[:n_source]
            self.source = nn.Parameter(wte[idx].float().clone())
        else:
            self.source = nn.Parameter(torch.randn(n_source, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        src = self.source.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(x, src, src)
        return self.norm(x + out)


class ReactorTimeLLMModel(nn.Module):
    """
    Probabilistic TimeLLM for reactor forecasting.

    All channels (endo + exo) are patched and independently fed through a
    frozen GPT-2 backbone via a reprogramming layer.  Only forecast-target
    endo channel outputs are projected to the probabilistic head.
    """
    def __init__(self,
                 c_endo: int,
                 c_exo: int,
                 context_len: int,
                 horizon: int,
                 n_forecast: int = 1,
                 forecast_target_indices: Optional[List[int]] = None,
                 patch_size: int = 16,
                 n_llm_layers: int = 6,
                 n_heads: int = 8,
                 n_source: int = 512,
                 dropout: float = 0.1,
                 head: str = 'student_t',
                 use_revin: bool = True,
                 use_batch_token: bool = False,
                 batch_token_idx: Optional[List[int]] = None,
                 batch_token_names: Optional[List[str]] = None,
                 llm_model: str = 'gpt2',
                 prompt_text: str = '',
                 use_stats_prompt: bool = False,
                 residual_anchor: bool = False):
        super().__init__()
        self.residual_anchor = residual_anchor
        self.c_endo       = c_endo
        self.c_total      = c_endo + c_exo
        self.context_len  = context_len
        self.horizon      = horizon
        self.n_forecast   = n_forecast
        self.head_type    = head
        self.use_revin    = use_revin
        self.use_batch_token  = use_batch_token
        self.batch_token_idx  = batch_token_idx
        self.forecast_target_indices = (list(range(n_forecast))
                                        if forecast_target_indices is None
                                        else list(forecast_target_indices))
        self.patch_size = patch_size
        pad_len = (patch_size - context_len % patch_size) % patch_size
        self.pad_len  = pad_len
        self.n_patches = (context_len + pad_len) // patch_size

        if use_revin:
            self.revin = RevIN(c_endo)

        try:
            llm = GPT2Model.from_pretrained(
                llm_model, attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16)
            print("  GPT-2 backbone: flash_attention_2 (bfloat16)")
        except Exception:
            llm = GPT2Model.from_pretrained(llm_model)
            print("  GPT-2 backbone: standard attention  (pip install flash-attn for faster training)")
        d_llm = llm.config.hidden_size
        self.d_llm = d_llm
        llm.h = nn.ModuleList(list(llm.h)[:n_llm_layers])
        self.llm = llm
        for p in self.llm.parameters():
            p.requires_grad_(False)

        self.patch_proj    = nn.Linear(patch_size, d_llm)
        self.drop          = nn.Dropout(dropout)
        with torch.no_grad():
            wte = llm.wte.weight.detach()
        self.reprogramming = ReprogrammingLayer(d_llm, n_heads, n_source, wte=wte)

        head_out = horizon * (2 if head == 'gaussian' else 3)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.n_patches * d_llm),
            nn.Linear(self.n_patches * d_llm, d_llm),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_llm, head_out),
        )

        self.batch_token_names = list(batch_token_names or [])
        if use_batch_token and batch_token_idx:
            self._bt_tokenizer = GPT2Tokenizer.from_pretrained(llm_model, use_fast=False)

        # ── Text prompt prefix ─────────────────────────────────────────────
        self.prefix_len = 0
        if prompt_text:
            _tok = GPT2Tokenizer.from_pretrained(llm_model, use_fast=False)
            ids  = _tok.encode(prompt_text, return_tensors='pt')[0]
            with torch.no_grad():
                embs = self.llm.wte(ids)           # [prefix_len, d_llm]
            self.register_buffer('prefix_embeds', embs.unsqueeze(0))
            self.prefix_len = len(ids)
            print(f"  Prompt prefix: {len(ids)} tokens  →  "
                  f"{prompt_text[:70]}{'…' if len(prompt_text) > 70 else ''}")
        else:
            self.register_buffer('prefix_embeds', None)

        # ── Per-sample statistics token ────────────────────────────────────
        self.use_stats_prompt = use_stats_prompt
        if use_stats_prompt:
            self.stats_proj = nn.Linear(4 * n_forecast, d_llm)
            self.stats_norm = nn.LayerNorm(d_llm)

        # Cross-channel attention: lets target channel attend to all channels after
        # the GPT-2 temporal encoder, so exogenous variables are genuinely used
        self.channel_attn = nn.MultiheadAttention(d_llm, n_heads, dropout=dropout, batch_first=True)
        self.channel_norm = nn.LayerNorm(d_llm)

        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)

    def forward(self, x_endo: torch.Tensor, x_exo: torch.Tensor):
        B, C_e, T = x_endo.shape
        x_endo_orig = x_endo  # save pre-RevIN for stats token

        if self.use_revin:
            x_endo, (rv_mean, rv_std) = self.revin.normalise(x_endo)
            rv_mean_tgt = rv_mean[:, self.forecast_target_indices, :]
            rv_std_tgt  = rv_std [:, self.forecast_target_indices, :]

        last_sc = x_endo[:, self.forecast_target_indices, -1:]  # [B, n_f, 1] RevIN space

        x = torch.cat([x_endo, x_exo], dim=1)          # [B, C_total, T]
        B, C, T = x.shape

        if self.pad_len > 0:
            x = F.pad(x, (self.pad_len, 0))             # pad left in time

        x = x.unfold(-1, self.patch_size, self.patch_size)   # [B, C, n_patches, patch_size]
        x = x.reshape(B * C, self.n_patches, self.patch_size)

        x = self.drop(self.patch_proj(x))               # [B*C, n_patches, d_llm]
        x = self.reprogramming(x)                       # [B*C, n_patches, d_llm]

        # ── Prompt prefix: batch metadata text + static text + stats token ──
        prefix_parts = []
        if self.use_batch_token and self.batch_token_idx is not None:
            scalars  = x_exo[:, self.batch_token_idx, 0]          # [B, n_bt]
            bt_embs  = []
            for b in range(B):
                text = " ".join(f"{name}={scalars[b, i].item():.2f}"
                                for i, name in enumerate(self.batch_token_names))
                ids  = self._bt_tokenizer.encode(text, return_tensors='pt')[0].to(x.device)
                bt_embs.append(self.llm.wte(ids))                  # [n_toks, d_llm]
            max_toks = max(e.shape[0] for e in bt_embs)
            padded   = torch.zeros(B, max_toks, self.d_llm, device=x.device, dtype=x.dtype)
            for b, e in enumerate(bt_embs):
                padded[b, :e.shape[0]] = e
            bt_tok = (padded.unsqueeze(1)
                      .expand(B, C, max_toks, self.d_llm)
                      .reshape(B * C, max_toks, self.d_llm))
            prefix_parts.append(bt_tok)

        if self.prefix_embeds is not None:
            prefix_parts.append(self.prefix_embeds.expand(B * C, -1, -1))

        if self.use_stats_prompt:
            x_tgt = x_endo_orig[:, self.forecast_target_indices, :]   # [B, n_f, T]
            svec  = torch.cat([x_tgt.mean(-1), x_tgt.std(-1),
                               x_tgt.min(-1).values, x_tgt.max(-1).values], dim=-1)  # [B, 4*n_f]
            stok  = self.stats_norm(self.stats_proj(svec))             # [B, d_llm]
            stok  = (stok.unsqueeze(1).unsqueeze(1)
                        .expand(B, C, 1, self.d_llm)
                        .reshape(B * C, 1, self.d_llm))
            prefix_parts.append(stok)

        if prefix_parts:
            prefix = torch.cat(prefix_parts, dim=1)                    # [B*C, n_pre, d_llm]
            x_in   = torch.cat([prefix, x], dim=1)
            n_pre  = prefix.shape[1]
        else:
            x_in  = x
            n_pre = 0

        llm_out = self.llm(inputs_embeds=x_in.to(self.llm.dtype)).last_hidden_state.float()
        llm_out = llm_out[:, n_pre:, :]                                # drop prefix positions
        llm_out = llm_out.reshape(B, C, self.n_patches, self.d_llm)

        # Cross-channel attention at each patch position.
        # [B, C, P, D] → [B, P, C, D] → [B*P, C, D]
        enc_ch = llm_out.permute(0, 2, 1, 3).reshape(B * self.n_patches, C, self.d_llm)
        mixed, _ = self.channel_attn(enc_ch, enc_ch, enc_ch)
        enc_ch = self.channel_norm(enc_ch + mixed)
        llm_out = enc_ch.reshape(B, self.n_patches, C, self.d_llm).permute(0, 2, 1, 3)

        tgt_out  = llm_out[:, self.forecast_target_indices, :, :]  # [B, n_f, n_patches, d_llm]
        n_f      = self.n_forecast
        tgt_flat = tgt_out.reshape(B * n_f, self.n_patches * self.d_llm)
        raw      = self.output_proj(tgt_flat).reshape(B, n_f, -1)

        H = self.horizon
        if self.head_type == 'gaussian':
            mean = raw[:, :, :H]
            lv   = raw[:, :, H:]
            if self.residual_anchor:
                mean = mean + last_sc
            if self.use_revin:
                mean = self.revin.denormalise(mean, rv_mean_tgt, rv_std_tgt)
            return mean, lv
        else:
            mu   = raw[:, :, :H]
            lsig = raw[:, :, H:2*H]
            lnu  = raw[:, :, 2*H:]
            if self.residual_anchor:
                mu = mu + last_sc
            if self.use_revin:
                mu = self.revin.denormalise(mu, rv_mean_tgt, rv_std_tgt)
            return mu, lsig, lnu


# ── Loss functions ─────────────────────────────────────────────────────────
LOG_VAR_MIN, LOG_VAR_MAX = -4.0, 4.0
LOG_SIG_MIN, LOG_SIG_MAX = -4.0, 2.0


def gaussian_nll(mean, log_var, target):
    lv = log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    return 0.5 * (lv + (target - mean).pow(2) / lv.exp()).mean()


def student_t_nll(mu, log_sigma, log_nu, target):
    ls    = log_sigma.clamp(LOG_SIG_MIN, LOG_SIG_MAX)
    nu    = F.softplus(log_nu) + 2.0
    sigma = torch.exp(ls)
    z_sq  = ((target - mu) / sigma) ** 2
    nll   = (ls
             + 0.5 * (nu * float(torch.pi)).log()
             + torch.lgamma(nu / 2)
             - torch.lgamma((nu + 1) / 2)
             + (nu + 1) / 2 * torch.log(1 + z_sq / nu))
    return nll.mean()


def compute_loss(model_out, target, head: str, peak_weight: float = 0.0) -> torch.Tensor:
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

def train_epoch(model, loader, optimizer, scaler, device, head, peak_weight: float = 0.0):
    model.train(); total, n = 0.0, 0
    amp_enabled = device.type == 'cuda'
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            loss = compute_loss(model(x_endo, x_exo), y, head, peak_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * x_endo.size(0); n += x_endo.size(0)
    return total / n if n > 0 else float('nan')


@torch.no_grad()
def eval_epoch(model, loader, device, head, peak_weight: float = 0.0):
    model.eval(); total, n = 0.0, 0
    amp_enabled = device.type == 'cuda'
    for x_endo, x_exo, y in loader:
        x_endo, x_exo, y = x_endo.to(device), x_exo.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            loss = compute_loss(model(x_endo, x_exo), y, head, peak_weight)
        total += loss.item() * x_endo.size(0); n += x_endo.size(0)
    return total / n if n > 0 else float('nan')


@torch.no_grad()
def collect_predictions(model, loader, device, head):
    model.eval(); means, scales, targets, nus = [], [], [], []
    amp_enabled = device.type == 'cuda'
    for x_endo, x_exo, y in loader:
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(x_endo.to(device), x_exo.to(device))
        means.append(out[0].float().cpu().numpy())
        scales.append(out[1].float().cpu().numpy())
        targets.append(y.numpy())
        if head == 'student_t':
            nus.append(out[2].float().cpu().numpy())
    nu_preds = np.concatenate(nus) if nus else None
    return np.concatenate(means), np.concatenate(scales), np.concatenate(targets), nu_preds


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

def plot_horizon(model, sc_endo, sc_exo, df, endo_cols, exo_cols,
                 scale_mode, context_len, horizon, out_path, target_col,
                 log_cols=None, n_windows=6, context_show=60, seed=42,
                 val_frac=0.1, test_frac=0.1, head='student_t',
                 target_channel_idx: int = 0, n_spike_windows: int = 6,
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
            ax.fill_between(fore_x, lower_95, upper_95, color='#2ca02c', alpha=0.15, label='95% PI')
            ax.fill_between(fore_x, lower_1s, upper_1s, color='#2ca02c', alpha=0.30, label='68% PI')
            ax.plot(fore_x, mean_orig,  color='#2ca02c', lw=2.0, label='Mean forecast (TimeLLM)')
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
        fig.suptitle(f'{target_col.upper()} TimeLLM {head} forecast  endo={endo_cols}  {title_suffix}\n'
                     f'{horizon}-step horizon  seed={seed}', fontsize=11, y=1.01)
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
    for attr in ('output_proj', 'stats_proj', 'stats_norm'):
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters():
                p.requires_grad_(True)
    # Unfreeze last n_layers of the GPT-2 blocks
    if hasattr(model, 'llm') and hasattr(model.llm, 'h'):
        for layer in list(model.llm.h)[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Fine-tune: {trainable:,} / {total:,} params trainable  "
          f"(output_proj + last {n_layers} LLM block(s))")


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

    _bt_cols_used = [c for c in args.batch_token_cols if c in exo_cols] if args.batch_token else []
    batch_token_idx = [exo_cols.index(c) for c in _bt_cols_used] if _bt_cols_used else None

    print("\n── Building model ───────────────────────────────────────────────")
    model = ReactorTimeLLMModel(
        c_endo=len(endo_cols), c_exo=len(exo_cols),
        context_len=args.context_len, horizon=args.horizon,
        n_forecast=n_forecast,
        forecast_target_indices=forecast_target_indices,
        patch_size=args.patch_size,
        n_llm_layers=args.n_llm_layers,
        n_heads=args.n_heads,
        n_source=args.n_source,
        dropout=args.dropout,
        head=args.head,
        use_revin=not args.no_revin,
        use_batch_token=args.batch_token,
        batch_token_idx=batch_token_idx,
        batch_token_names=_bt_cols_used,
        llm_model=args.llm_model,
        prompt_text=args.prompt_text,
        use_stats_prompt=args.stats_prompt,
        residual_anchor=args.residual_anchor,
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

    try:
        xe = torch.randn(min(2, args.batch_size), len(endo_cols), args.context_len, device=device)
        xx = torch.randn(min(2, args.batch_size), len(exo_cols),  args.context_len, device=device)
        out_p = model(xe, xx)
        probe_tgt = torch.zeros(min(2, args.batch_size), n_forecast, args.horizon, device=device)
        compute_loss(out_p, probe_tgt, args.head).backward()
        model.zero_grad()
        mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else 0.0
        print(f"  Forward probe: {mem:.1f} MB  shape={list(out_p[0].shape)}  ✓")
    except RuntimeError as e:
        print(f"  Forward probe FAILED: {e}"); return None, {}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr * 0.01)

    print(f"\n── Training up to {args.epochs} epochs  (patience={args.patience}) ─")
    history = {'train': [], 'val': []}
    best_val, patience_cnt = float('inf'), 0
    best_path = out_dir / 'best_model.pt'
    amp_scaler = torch.amp.GradScaler(device.type, enabled=False)  # bfloat16 needs no scaling

    for epoch in range(1, args.epochs + 1):
        t0      = time.time()
        tr_loss = train_epoch(model, train_dl, optimizer, amp_scaler, device, args.head, args.peak_weight)
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
            print(f"  Early stop at epoch {epoch}"); break

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
        test_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=torch.cuda.is_available())

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

        m_orig = np.stack([
            inverse_target(means[:,ci,:].reshape(-1), sc_endo, col,
                           endo_cols, args.scale_mode, log_cols).reshape(N, H)
            for ci, col in enumerate(target_cols)], axis=1)
        t_orig = np.stack([
            inverse_target(tgts[:,ci,:].reshape(-1), sc_endo, col,
                           endo_cols, args.scale_mode, log_cols).reshape(N, H)
            for ci, col in enumerate(target_cols)], axis=1)

        per_channel_metrics = _print_point_metrics(
            f"all windows ({C_out} channel(s))",
            m_orig, t_orig, means, tgts, target_cols)

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
                                      num_workers=0, pin_memory=torch.cuda.is_available())
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
        print(f"  Freeze all except output_proj + last {args.finetune_layers} LLM block(s)  |  LR={args.finetune_lr:.1e}")
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
                checkpoint = str(best_path)
            if results:
                all_results[exp_tag] = results
        summary = base_out / 'sequential_summary.json'
        with open(summary, 'w') as f:
            json.dump({'mode': 'sequential', 'experiments': args.experiment,
                       'endo_cols': endo_cols, 'log_cols': list(log_cols),
                       'target': target_cols, 'results': all_results}, f, indent=2)
        print(f"Sequential summary → {summary}")

    else:
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
                                description='Probabilistic TimeLLM on reactor batch data.')

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument('--experiment',  nargs='+', required=True)
    p.add_argument('--data_paths',  nargs='+', required=True)
    p.add_argument('--cheatsheet',  default='filtered_data_stamps.txt')

    # ── Column routing ─────────────────────────────────────────────────────
    p.add_argument('--endogenous', nargs='+', default=['nh4'])
    p.add_argument('--exogenous',  nargs='*', default=[])
    p.add_argument('--target',     nargs='+', default=None)
    p.add_argument('--log_cols',   nargs='*', default=None)

    # ── Training mode ──────────────────────────────────────────────────────
    p.add_argument('--sequential',  action='store_true')
    p.add_argument('--sequential_lr_scale', type=float, default=0.2)
    p.add_argument('--leave_one_out', action='store_true')
    p.add_argument('--finetune',      action='store_true')
    p.add_argument('--finetune_layers',          type=int,   default=1)
    p.add_argument('--finetune_lr',              type=float, default=1e-5)
    p.add_argument('--finetune_spike_oversample', type=float, default=None)
    p.add_argument('--finetune_spike_abs_weight', type=float, default=None)
    p.add_argument('--batch_token', action='store_true')
    p.add_argument('--batch_token_cols', nargs='+', default=['inoc_amount', 'inoc_conc'])
    p.add_argument('--transfer_from', default=None)

    # ── Head ──────────────────────────────────────────────────────────────
    p.add_argument('--head', default='student_t', choices=['gaussian', 'student_t'])

    # ── Scaling ───────────────────────────────────────────────────────────
    p.add_argument('--scaling',    default='standard', choices=['standard', 'robust'])
    p.add_argument('--scale_mode', default='global',   choices=['global', 'per_var'])

    # ── Gap / segmentation ────────────────────────────────────────────────
    p.add_argument('--segment_isolated',    action='store_true')
    p.add_argument('--gap_threshold',       type=float, default=4.0)
    p.add_argument('--segment_gap_minutes', type=float, default=60.0)
    p.add_argument('--min_segment_rows',    type=int,   default=0)

    # ── Reproducibility ───────────────────────────────────────────────────
    p.add_argument('--seed', type=int, default=42)

    # ── TimeLLM architecture ──────────────────────────────────────────────
    p.add_argument('--patch_size',   type=int,   default=16,
                   help='Time steps per patch.')
    p.add_argument('--n_llm_layers', type=int,   default=6,
                   help='Number of GPT-2 transformer blocks to use (max 12 for gpt2).')
    p.add_argument('--n_heads',      type=int,   default=8,
                   help='Attention heads in the reprogramming layer.')
    p.add_argument('--n_source',     type=int,   default=512,
                   help='Number of learnable source word tokens for reprogramming.')
    p.add_argument('--llm_model',    default='gpt2',
                   help='HuggingFace model name for the LLM backbone (e.g. gpt2, gpt2-medium).')
    p.add_argument('--prompt_text',  default='',
                   help='Static text prefix prepended before patches (e.g. dataset/task description). '
                        'Tokenized with the LLM tokenizer; embeddings are frozen.')
    p.add_argument('--stats_prompt', action='store_true',
                   help='Prepend a learned per-sample statistics token (mean/std/min/max of '
                        'target channels over the context window).')
    p.add_argument('--dropout',         type=float, default=0.1)
    p.add_argument('--no_revin',        action='store_true')
    p.add_argument('--residual_anchor', action='store_true',
                   help='Add last observed RevIN-normalised value to forecast mean (residual prediction).')
    p.add_argument('--spike_metric_threshold', type=float, default=1.0)
    p.add_argument('--n_spike_windows',        type=int,   default=6)
    p.add_argument('--spike_oversample',       type=float, default=0.0)
    p.add_argument('--spike_abs_weight',       type=float, default=1.0)
    p.add_argument('--peak_weight',            type=float, default=0.0)

    # ── Sequence lengths ──────────────────────────────────────────────────
    p.add_argument('--context_len', type=int,   default=180)
    p.add_argument('--horizon',     type=int,   default=72)
    p.add_argument('--stride',      type=int,   default=1)

    # ── Optimisation ──────────────────────────────────────────────────────
    p.add_argument('--batch_size',  type=int,   default=16,
                   help='Smaller default than other models due to GPT-2 memory.')
    p.add_argument('--lr',          type=float, default=1e-4,
                   help='Lower default LR than scratch models — LLM backbone is pretrained.')
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--patience',    type=int,   default=15)
    p.add_argument('--val_frac',    type=float, default=0.1)
    p.add_argument('--test_frac',   type=float, default=0.1)

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument('--out_dir', default='results/reactor_timellm')

    main(p.parse_args())
