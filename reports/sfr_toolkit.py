from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict

def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    from math import sqrt
    z = 1.959964 if abs(alpha - 0.05) < 1e-9 else 1.959964
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    halfwidth = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return (max(0.0, center - halfwidth), min(1.0, center + halfwidth))

def _decide_winner(x1: float, x2: float, *, lower_is_better: bool, tie_tol: float = 0.0) -> int:
    if np.isnan(x1) or np.isnan(x2):
        return 0
    diff = x1 - x2
    cmp = diff if lower_is_better else -diff
    if cmp < -tie_tol:
        return +1
    elif cmp > tie_tol:
        return -1
    else:
        return 0

def _pairwise_winners(df: pd.DataFrame, col1: str, col2: str, *, lower_is_better: bool, tie_tol: float = 0.0) -> np.ndarray:
    x1 = df[col1].to_numpy(dtype=float)
    x2 = df[col2].to_numpy(dtype=float)
    out = np.empty(len(df), dtype=int)
    for i in range(len(df)):
        out[i] = _decide_winner(x1[i], x2[i], lower_is_better=lower_is_better, tie_tol=tie_tol)
    return out

def _summarize_position_bias(w_a: np.ndarray, w_b: np.ndarray) -> Dict[str, float]:
    n = len(w_a)
    assert len(w_b) == n
    mask_strict_both = (w_a != 0) & (w_b != 0)
    n_strict = int(mask_strict_both.sum())
    flips = (w_a[mask_strict_both] != w_b[mask_strict_both]).sum() if n_strict > 0 else 0
    sfr = flips / n_strict if n_strict > 0 else np.nan
    pos1_win_a = (w_a == +1).mean()
    pos1_win_b = (w_b == -1).mean()
    pos1_win_avg = 0.5 * (pos1_win_a + pos1_win_b)
    pai = pos1_win_avg - (1.0 - pos1_win_avg)
    tie_rate_any = ((w_a == 0) | (w_b == 0)).mean()
    return {
        "swap_flip_rate": float(sfr),
        "n_pairs_strict": int(n_strict),
        "pos1_win_rate_orderA": float(pos1_win_a),
        "pos1_win_rate_orderB": float(pos1_win_b),
        "pos1_win_rate_avg": float(pos1_win_avg),
        "position_advantage_index": float(pai),
        "tie_rate_any": float(tie_rate_any),
        "n_pairs_total": int(n),
    }

def compute_sfr_from_frames(
    df_ab: pd.DataFrame,
    df_ba: pd.DataFrame,
    col_item1: str,
    col_item2: str,
    *,
    lower_is_better: bool = True,
    tie_tol: float = 0.0,
    on: Optional[str] = None,
    groupby: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if on is not None:
        merged = pd.merge(
            df_ab[[on, col_item1, col_item2] + ([g for g in (groupby or []) if g in df_ab.columns])],
            df_ba[[on, col_item1, col_item2] + ([g for g in (groupby or []) if g in df_ba.columns])],
            on=on,
            suffixes=("_A", "_B"),
            how="inner",
        )
    else:
        merged = pd.DataFrame({
            f"{col_item1}_A": df_ab[col_item1].to_numpy(),
            f"{col_item2}_A": df_ab[col_item2].to_numpy(),
            f"{col_item1}_B": df_ba[col_item1].to_numpy(),
            f"{col_item2}_B": df_ba[col_item2].to_numpy(),
        })
        if groupby:
            for g in groupby:
                if g in df_ab.columns:
                    merged[g] = df_ab[g].to_numpy()
    w_a = _pairwise_winners(
        merged.rename(columns={f"{col_item1}_A": "c1", f"{col_item2}_A": "c2"}),
        "c1", "c2", lower_is_better=lower_is_better, tie_tol=tie_tol
    )
    w_b = _pairwise_winners(
        merged.rename(columns={f"{col_item1}_B": "c1", f"{col_item2}_B": "c2"}),
        "c1", "c2", lower_is_better=lower_is_better, tie_tol=tie_tol
    )
    if not groupby:
        summary = _summarize_position_bias(w_a, w_b)
        ci_lo, ci_hi = _wilson_ci(int(summary["swap_flip_rate"] * summary["n_pairs_strict"]) if summary["n_pairs_strict"]>0 else 0,
                                  summary["n_pairs_strict"])
        summary.update({"sfr_ci_low": ci_lo, "sfr_ci_high": ci_hi})
        return pd.DataFrame([summary])
    res = []
    gcols = [g for g in groupby if g in merged.columns]
    grouped = merged.groupby(gcols, dropna=False)
    for keys, chunk in grouped:
        idx = chunk.index.to_numpy()
        w_a_g = w_a[idx]
        w_b_g = w_b[idx]
        s = _summarize_position_bias(w_a_g, w_b_g)
        ci_lo, ci_hi = _wilson_ci(int(s["swap_flip_rate"] * s["n_pairs_strict"]) if s["n_pairs_strict"]>0 else 0,
                                  s["n_pairs_strict"])
        s.update({"sfr_ci_low": ci_lo, "sfr_ci_high": ci_hi})
        if isinstance(keys, tuple):
            for k, gname in zip(keys, gcols):
                s[gname] = k
        else:
            s[gcols[0]] = keys
        res.append(s)
    return pd.DataFrame(res)[gcols + [c for c in res[0].keys() if c not in gcols]]

def compute_sfr_from_two_files(
    file_ab: str | Path,
    file_ba: str | Path,
    col_item1: str,
    col_item2: str,
    *,
    lower_is_better: bool = True,
    tie_tol: float = 0.0,
    on: Optional[str] = None,
    groupby: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    df_ab = pd.read_csv(file_ab)
    df_ba = pd.read_csv(file_ba)
    return compute_sfr_from_frames(
        df_ab, df_ba, col_item1, col_item2,
        lower_is_better=lower_is_better, tie_tol=tie_tol, on=on, groupby=groupby
    )
