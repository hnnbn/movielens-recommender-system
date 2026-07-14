from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class ItemCFArtifacts:
    """Objects needed for mean-centered top-K item collaborative filtering."""

    train_matrix: pd.DataFrame
    item_similarity: pd.DataFrame
    user_means: pd.Series
    item_means: pd.Series
    global_mean: float
    k_neighbors: int = 30


def build_item_cf(
    ratings: pd.DataFrame,
    k_neighbors: int = 30,
    user_col: str = "user_id",
    item_col: str = "movie_id",
    rating_col: str = "rating",
) -> ItemCFArtifacts:
    """Build mean-centered item-item similarity artifacts from explicit ratings."""

    train_matrix = ratings.pivot_table(
        index=user_col,
        columns=item_col,
        values=rating_col,
    )
    user_means = train_matrix.mean(axis=1)
    item_means = train_matrix.mean(axis=0)
    global_mean = float(train_matrix.stack().mean())

    centered = train_matrix.sub(user_means, axis=0).fillna(0)
    similarity = cosine_similarity(centered.T)
    item_similarity = pd.DataFrame(
        similarity,
        index=train_matrix.columns,
        columns=train_matrix.columns,
    )

    return ItemCFArtifacts(
        train_matrix=train_matrix,
        item_similarity=item_similarity,
        user_means=user_means,
        item_means=item_means,
        global_mean=global_mean,
        k_neighbors=k_neighbors,
    )


def predict_item_cf(
    user_id: int,
    item_id: int,
    artifacts: ItemCFArtifacts,
    clip_min: float = 1.0,
    clip_max: float = 5.0,
) -> float:
    """Predict a rating with user-mean centered top-K item CF."""

    if user_id not in artifacts.train_matrix.index:
        return float(np.clip(artifacts.global_mean, clip_min, clip_max))

    user_mean = float(artifacts.user_means.loc[user_id])
    if item_id not in artifacts.item_similarity.index:
        return float(np.clip(user_mean, clip_min, clip_max))

    user_ratings = artifacts.train_matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return float(np.clip(user_mean, clip_min, clip_max))

    similarities = artifacts.item_similarity.loc[item_id, user_ratings.index]
    similarities = similarities.drop(labels=[item_id], errors="ignore")
    user_ratings = user_ratings.loc[similarities.index]

    neighbors = similarities.abs().sort_values(ascending=False).head(
        artifacts.k_neighbors
    )
    if neighbors.empty or np.isclose(neighbors.abs().sum(), 0.0):
        fallback = artifacts.item_means.get(item_id, user_mean)
        return float(np.clip(fallback, clip_min, clip_max))

    neighbor_ratings = user_ratings.loc[neighbors.index]
    centered_ratings = neighbor_ratings - user_mean
    pred = user_mean + float(np.dot(neighbors, centered_ratings) / neighbors.abs().sum())
    return float(np.clip(pred, clip_min, clip_max))


def predict_many(
    ratings: pd.DataFrame,
    artifacts: ItemCFArtifacts,
    user_col: str = "user_id",
    item_col: str = "movie_id",
) -> np.ndarray:
    """Predict ratings for a dataframe of user-item pairs."""

    return np.array(
        [
            predict_item_cf(row[user_col], row[item_col], artifacts)
            for _, row in ratings.iterrows()
        ]
    )
