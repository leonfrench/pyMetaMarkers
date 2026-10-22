"""Pareto front of markers in the (fold change of detection, AUROC) space."""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_pareto_front(x, y, tolerance: float = 0) -> np.ndarray:
    """Boolean mask of points on the Pareto front with respect to x and y (both maximized)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.lexsort((-y, -x))  # decreasing x, then decreasing y (stable)
    sorted_y = y[order]
    max_y = np.maximum.accumulate(sorted_y) if len(sorted_y) else sorted_y
    result = np.empty(len(x), dtype=bool)
    result[order] = sorted_y >= max_y - tolerance
    return result


def get_pareto_markers(meta_markers: pd.DataFrame, cell_type_name: str, min_recurrence: int = 1) -> list[str]:
    """Pareto front markers of a cell type, ordered by increasing fold change of detection."""
    selected = meta_markers[(meta_markers["recurrence"] >= min_recurrence)]
    selected = selected[selected["cell_type"] == cell_type_name]
    selected = selected[is_pareto_front(selected["auroc"], selected["fold_change_detection"])]
    selected = selected.sort_values("fold_change_detection", kind="stable")
    return list(selected["gene"])


def pareto_markers_data(
    meta_markers: pd.DataFrame,
    cell_type_name: str,
    min_recurrence: int = 1,
    min_auroc: float = 0.5,
    min_fc: float = 0,
) -> pd.DataFrame:
    """Markers shown by ``plot_pareto_markers``, with an ``is_pareto`` column."""
    columns = ["cell_type", "gene", "recurrence", "auroc", "fold_change_detection"]
    data = meta_markers[columns]
    data = data[(data["cell_type"] == cell_type_name) & (data["recurrence"] >= min_recurrence)]
    data = data[(data["auroc"] > min_auroc) & (data["fold_change_detection"] > min_fc)]
    data = data.reset_index(drop=True)
    data["is_pareto"] = _pareto_by_cell_type(data)
    return data


def pareto_summary_data(meta_markers: pd.DataFrame, min_recurrence: int = 1) -> pd.DataFrame:
    """Pareto front markers of every cell type (what ``plot_pareto_summary`` shows)."""
    data = meta_markers[meta_markers["recurrence"] >= min_recurrence].reset_index(drop=True)
    return data[_pareto_by_cell_type(data)].reset_index(drop=True)


def _pareto_by_cell_type(data: pd.DataFrame) -> np.ndarray:
    """Pareto front mask computed separately for each cell type."""
    keep = np.zeros(len(data), dtype=bool)
    cell_types = data["cell_type"].to_numpy()
    for cell_type in pd.unique(cell_types):
        rows = np.flatnonzero(cell_types == cell_type)
        keep[rows] = is_pareto_front(data["auroc"].to_numpy()[rows], data["fold_change_detection"].to_numpy()[rows])
    return keep
