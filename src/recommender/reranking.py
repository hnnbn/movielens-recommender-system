from __future__ import annotations

import numpy as np
import pandas as pd


def build_longtail_bonus(
    ratings: pd.DataFrame,
    item_col: str = "movie_id",
    smoothing: float = 1.0,
) -> dict[int, float]:
    """Create an inverse log-popularity bonus for long-tail item reranking."""

    counts = ratings[item_col].value_counts()
    max_count = float(counts.max())
    bonus = np.log1p(max_count + smoothing) / np.log1p(counts + smoothing)
    return bonus.astype(float).to_dict()


def rerank_with_longtail_bonus(
    recommendations: dict[int, list[tuple[int, float]]],
    longtail_bonus: dict[int, float],
    lambda_: float,
) -> dict[int, list[tuple[int, float]]]:
    """Rerank scored recommendations with a small long-tail bonus."""

    reranked: dict[int, list[tuple[int, float]]] = {}
    for user_id, rec_list in recommendations.items():
        rescored = [
            (item_id, score + lambda_ * longtail_bonus.get(item_id, 1.0))
            for item_id, score in rec_list
        ]
        reranked[user_id] = sorted(rescored, key=lambda row: row[1], reverse=True)

    return reranked
