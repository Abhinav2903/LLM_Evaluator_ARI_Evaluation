
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sfr_toolkit import compute_sfr_from_two_files

def _coalesce_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return subset of cols that exist in df."""
    return [c for c in cols if c in df.columns]

def build_sfr_summary(configs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    configs: list of dicts with keys
      - judge (str)
      - prompt (str)          e.g., "baseline", "cot"
      - ari (int)             e.g., 5, 9, 14
      - file_ab (str)
      - file_ba (str)
      - item1_col (str)       e.g., f"Rank_ARI_{ari}" or a score column
      - item2_col (str)       e.g., f"Rank_Wrong_Answer_ARI_{ari}"
      - lower_is_better (bool, default True)
      - on (str|None, default None)  key column shared by both files; if None align by row
    Returns a tidy DataFrame with one row per config containing:
      SFR, CI, PAI, pos1 win rates, tie rate, counts, and metadata.
    """
    rows = []
    for cfg in configs:
        judge = cfg["judge"]
        prompt = cfg["prompt"]
        ari = cfg["ari"]
        lower_is_better = cfg.get("lower_is_better", True)
        on = cfg.get("on", None)

        res = compute_sfr_from_two_files(
            file_ab=cfg["file_ab"],
            file_ba=cfg["file_ba"],
            col_item1=cfg["item1_col"],
            col_item2=cfg["item2_col"],
            lower_is_better=lower_is_better,
            on=on,
            groupby=None,        # we want one row per (judge, prompt, ari)
        )

        # There will be exactly one row
        r = res.iloc[0].to_dict()
        # Add useful derived values
        flips = int(round(r["swap_flip_rate"] * r["n_pairs_strict"])) if pd.notna(r["swap_flip_rate"]) else np.nan
        r.update({
            "judge": judge,
            "prompt": prompt,
            "ari": ari,
            "metric_kind": "rank" if lower_is_better else "score",
            "flips_count": flips,
        })
        rows.append(r)

    out = pd.DataFrame(rows)

    # Reorder columns for concise thesis tables
    preferred = [
        "judge","prompt","ari","metric_kind",
        "swap_flip_rate","sfr_ci_low","sfr_ci_high","flips_count",
        "position_advantage_index","pos1_win_rate_avg",
        "pos1_win_rate_orderA","pos1_win_rate_orderB",
        "tie_rate_any","n_pairs_strict","n_pairs_total"
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols].sort_values(["judge","prompt","ari"], ignore_index=True)
    return out

def save_sfr_summary(out_df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    # --- EDIT THIS CONFIG to your actual file paths ---
    # Example for GPT-4, ARI 5/9/14, baseline vs CoT.
    base = "../resultFiles/GPT4"
    cfgs = []
    for ari in (5, 9, 14):
        cfgs += [
            dict(
                judge="gpt-4",
                prompt="baseline",
                ari=ari,
                file_ab=f"{base}/NB/Rank/gpt4_ari{ari}_b1.csv",
                file_ba=f"{base}/PB/Rank/gpt4_ari{ari}_b1_pb.csv",
                item1_col=f"Rank_ARI_{ari}",
                item2_col=f"Rank_Wrong_Answer_ARI_{ari}",
                lower_is_better=True,
                on=None,   # set to "pair_id" if both CSVs contain it
            ),
            dict(
                judge="gpt-4",
                prompt="cot",
                ari=ari,
                file_ab=f"{base}/NB/Rank/gpt4_ari{ari}_cot.csv",
                file_ba=f"{base}/PB/Rank/gpt4_ari{ari}_cot_pb.csv",
                item1_col=f"Rank_ARI_{ari}",
                item2_col=f"Rank_Wrong_Answer_ARI_{ari}",
                lower_is_better=True,
                on=None,
            ),
        ]

    df = build_sfr_summary(cfgs)
    print(df)
    save_sfr_summary(df, "./sfr_summary_gpt4.csv")
    print("Saved to ./sfr_summary_gpt4.csv")
