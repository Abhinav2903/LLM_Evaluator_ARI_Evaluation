from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
DataFrameLike = Union[str, pd.DataFrame]


# def _load_df(source: DataFrameLike, *, name: str) -> pd.DataFrame:
#     if isinstance(source, str):
#         if not os.path.exists(source):
#             raise FileNotFoundError(f"{name} path not found: {source}")
#         return pd.read_csv(source)
#     if isinstance(source, pd.DataFrame):
#         return source.copy()
#     raise TypeError(f"{name} must be a CSV path or a pandas.DataFrame")


# --- Loader with normalization ---
def _load_df(source: DataFrameLike, *, name: str) -> pd.DataFrame:
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"{name} path not found: {source}")
        return pd.read_csv(
            source,
            na_values=["", " ", "no rank response required"]
        )
    if isinstance(source, pd.DataFrame):
        return source.copy()
    raise TypeError(f"{name} must be a CSV path or a pandas.DataFrame")

@dataclass
class LLMEvalVisualizer:
    rank_source: Optional[DataFrameLike] = None
    score_source: Optional[DataFrameLike] = None
    rank_df: Optional[pd.DataFrame] = field(default=None, init=False)
    score_df: Optional[pd.DataFrame] = field(default=None, init=False)

    def load(self) -> "LLMEvalVisualizer":
        if self.rank_source is not None:
            self.rank_df = _load_df(self.rank_source, name="rank_source")
        if self.score_source is not None:
            self.score_df = _load_df(self.score_source, name="score_source")
        return self

    def pairwise_rank_counts(
        self,
        score_columns: Sequence[str],
        *,
        rank_method: str = "min",
        ascending: bool = False,
        rename_map: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        if self.score_df is None:
            raise ValueError("score_df is not loaded.")
        df = self.score_df[list(score_columns)].copy()

        rows = []
        for _, row in df.iterrows():
            temp = pd.DataFrame({"score_name": df.columns, "score_value": row.values})
            temp["rank"] = temp["score_value"].rank(method=rank_method, ascending=ascending)
            rows.append(temp[["score_name", "rank"]])
        long = pd.concat(rows, ignore_index=True)

        if rename_map:
            long["score_name"] = long["score_name"].map(rename_map).fillna(long["score_name"])

        counts = (
            long.groupby(["score_name", "rank"])
                .size()
                .reset_index(name="count")
                .pivot(index="score_name", columns="rank", values="count")
                .fillna(0)
        )
        counts = counts.reindex(sorted(counts.columns), axis=1)
        return counts

    def listwise_rank_distribution(
        self,
        *,
        desired_order: Optional[Sequence[str]] = None,
        col_filter: Optional[str] = "Rank",
        rename_map: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        if self.rank_df is None:
            raise ValueError("rank_df is not loaded.")
        cols = [c for c in self.rank_df.columns if (col_filter in c if col_filter else True)]
        if not cols:
            raise ValueError("No rank columns found.")
        melted = self.rank_df[cols].melt(var_name="Rank_Type", value_name="Rank_Value")
        dist = (
            melted.groupby(["Rank_Type", "Rank_Value"])
                  .size()
                  .unstack(fill_value=0)
        )
        dist = dist.reindex(sorted(dist.columns), axis=1)
        if desired_order:
            dist = dist.reindex(desired_order)
        if rename_map:
            dist.index = [rename_map.get(x, x) for x in dist.index]
        return dist

    def _sample_colors(
        self,
        n: int,
        *,
        cmap_name: str = "Blues",
        lo: float = 0.25,
        hi: float = 0.9,
        reverse: bool = True
    ):
        cmap = get_cmap(cmap_name)
        colors = cmap(np.linspace(lo, hi, n))
        return colors[::-1] if reverse else colors

    def plot_stacked_distribution(
        self,
        distribution: pd.DataFrame,
        *,
        title: str = "Stacked Distribution of Ranks",
        xlabel: str = "Category",
        ylabel: str = "Count",
        cmap_name: str = "Blues",
        normalize: bool = False,
        legend_title: str = "Rank",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        data = distribution.copy()
        if normalize:
            data = data.div(data.sum(axis=1), axis=0).fillna(0) * 100
            ylabel = "Percentage"
        rank_cols = list(data.columns)
        colors = self._sample_colors(len(rank_cols), cmap_name=cmap_name)
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        bottom = np.zeros(len(data))
        for idx, col in enumerate(rank_cols):
            ax.bar(
                data.index,
                data[col].values,
                bottom=bottom,
                label=f"Rank {int(col) if str(col).isdigit() else col}",
                color=colors[idx],
                edgecolor="none",
            )
            bottom += data[col].values
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return ax

    def plot_violin_scores(
        self,
        score_columns: Sequence[str],
        *,
        rename_map: Optional[Mapping[str, str]] = None,
        title: str = "Violin Plot of Score Distributions",
        xlabel: str = "Score Type",
        ylabel: str = "Score",
        palette: Optional[Union[str, list]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if self.score_df is None:
            raise ValueError("score_df is not loaded.")
        df = self.score_df[list(score_columns)].copy()
        melted = df.melt(var_name="score_name", value_name="score")
        if rename_map:
            melted["score_name"] = melted["score_name"].map(rename_map).fillna(melted["score_name"])
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        if palette is None:
            palette = self._sample_colors(len(score_columns), cmap_name="Blues")
        sns.violinplot(x="score_name", y="score", data=melted, palette=palette, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return ax


    def plot_density(
        self,
        score_columns: Sequence[str],
        *,
        rename_map: Optional[Mapping[str, str]] = None,
        title: str = "Score Density (KDE)",
        xlabel: str = "Score",
        ylabel: str = "Density",
        cmap_name: str = "Blues",
        alpha: float = 0.35,
        bw_adjust: float = 0.6,                          # NEW
        xlim: Optional[tuple[float, float]] = None,      # NEW
        ylim: Optional[tuple[float, float]] = None,      # NEW
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if self.score_df is None:
            raise ValueError("score_df is not loaded.")

        df = self.score_df[list(score_columns)].copy()
        labels = [rename_map.get(c, c) for c in score_columns] if rename_map else list(score_columns)

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        colors = self._sample_colors(len(score_columns), cmap_name=cmap_name, reverse=False)

        for col, label, color in zip(score_columns, labels, colors):
            sns.kdeplot(
                df[col].dropna(),
                fill=True,
                alpha=alpha,
                linewidth=1.8,
                label=label,
                ax=ax,
                color=color,
                bw_adjust=bw_adjust,          # keep bandwidth same across figures
                # cut=0,                        # don't extend beyond data range
                # clip=xlim                     # respect your chosen x-limits
            )

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(title="Score Type", bbox_to_anchor=(.05, 1), loc="upper left")
        plt.grid(True, alpha=0.25)
        sns.despine()
        plt.tight_layout()
        return ax

    # # --- Ridgeline (joy) plot (wide input). Requires joypy ---
    # def plot_ridgeline_scores(
    #     self,
    #     score_columns: Sequence[str],
    #     *,
    #     rename_map: Optional[Mapping[str, str]] = None,
    #     cmap_name: str = "Blues",
    #     overlap: float = 0.6,
    #     linewidth: float = 1.2,
    #     figsize: tuple = (12, 8),
    #     Model
    # ):
    #     """
    #     Ridgeline (KDE) of score distributions for the given columns.
    #     Uses wide data (each column is one series).
    #     Requires: pip install joypy
    #     """
    #     if self.score_df is None:
    #         raise ValueError("score_df is not loaded.")
    #     try:
    #         import joypy
    #     except ImportError as e:
    #         raise ImportError("plot_ridgeline_scores requires joypy. Install with: pip install joypy") from e

    #     df = self.score_df[list(score_columns)].copy()
    #     if rename_map:
    #         df = df.rename(columns=rename_map)

    #     fig, axes = joypy.joyplot(
    #         df,
    #         colormap=plt.cm.get_cmap(cmap_name),
    #         overlap=overlap,
    #         linewidth=linewidth,
    #         figsize=figsize,
    #     )
    #     plt.title(f"Ridgeline (KDE) of Score Distributions-{Model}")
    #     plt.xlabel("Score")
    #     plt.tight_layout()
    #     return axes
    def plot_ridgeline_scores(
        self,
        score_columns: Sequence[str],
        *,
        rename_map: Optional[Mapping[str, str]] = None,
        cmap_name: str = "Blues",
        overlap: float = 0.6,
        linewidth: float = 1.2,
        figsize: tuple = (12, 8),
        Model: str = ""   # <-- fixed
    ):
        import joypy
        df = self.score_df[list(score_columns)].copy()
        if rename_map:
            df = df.rename(columns=rename_map)
        fig, axes = joypy.joyplot(
            df,
            colormap=plt.cm.get_cmap(cmap_name),
            overlap=overlap,
            linewidth=linewidth,
            figsize=figsize,
        )
        plt.title(f"Ridgeline (KDE) of Score Distributions-{Model}")
        plt.xlabel("Score")
        plt.tight_layout()
        return axes


def default_score_rename_map() -> Mapping[str, str]:
    return {
        "Score_ARI_5": "Correct ARI 5",
        "Score_ARI_9": "Correct ARI 9",
        "Score_ARI_14": "Correct ARI 14",
        "Score_Ref_Answer": "Reference",
        "Score_Wrong_Answer_ARI_5": "Wrong ARI 5",
        "Score_Wrong_Answer_ARI_9": "Wrong ARI 9",
        "Score_Wrong_Answer_ARI_14": "Wrong ARI 14",
    }


def default_rank_order() -> List[str]:
    return [
        "Rank_ARI_14",
        "Rank_ARI_9",
        "Rank_ARI_5",
        "Rank_Ref",
        "Rank_Wrong_Answer_ARI_5",
        "Rank_Wrong_Answer_ARI_9",
        "Rank_Wrong_Answer_ARI_14",
    ]


