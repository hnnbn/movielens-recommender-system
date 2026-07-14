from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    """Root mean squared error for rating prediction."""

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error for rating prediction."""

    return float(mean_absolute_error(y_true, y_pred))


def catalog_coverage(recommendations: dict[int, list[int]], all_items, k: int) -> float:
    """Share of catalog items that appear in at least one user's top-K list."""

    catalog = set(all_items)
    if not catalog:
        return 0.0

    recommended = set()
    for rec_list in recommendations.values():
        recommended.update(rec_list[:k])

    return len(recommended) / len(catalog)
