"""AUROC computation, ROC-test p-values and FDR correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special, stats


def _sorted_levels(values) -> list[str]:
    """Unique values in C-locale order (R's sort() under LC_COLLATE=C, dplyr::arrange)."""
    return sorted(set(values))


def design_matrix(cell_type, scale_columns: bool = False) -> pd.DataFrame:
    """One-hot encode a vector of labels, with one column per unique label.

    Columns are sorted labels (C locale). If ``scale_columns`` is True, columns
    are normalized to sum to 1.
    """
    values = np.asarray(cell_type, dtype=object).astype(str)
    levels = _sorted_levels(values)
    codes = pd.Categorical(values, categories=levels).codes
    result = np.zeros((len(values), len(levels)))
    result[np.arange(len(values)), codes] = 1.0
    if scale_columns:
        result = result / result.sum(axis=0)
    return pd.DataFrame(result, columns=levels)


def _as_label_matrix(true_labels) -> tuple[np.ndarray, list[str]]:
    if isinstance(true_labels, pd.DataFrame):
        return true_labels.to_numpy(dtype=np.float64), [str(c) for c in true_labels.columns]
    labels = np.asarray(true_labels)
    if labels.ndim == 2:
        return labels.astype(np.float64), [str(i) for i in range(labels.shape[1])]
    design = design_matrix(labels)
    return design.to_numpy(), list(design.columns)


def compute_aurocs(predictors, label_matrix, compute_tie_correction: bool = False):
    """AUROCs of every (predictor, category) pair, for dense predictors.

    Parameters
    ----------
    predictors
        Samples x predictors matrix.
    label_matrix
        Samples x categories one-hot matrix (or a vector of labels).

    Returns
    -------
    (aurocs, tie_correction): a categories x predictors array and the tie
    correction per predictor (0 if not requested).
    """
    labels, _ = _as_label_matrix(label_matrix)
    predictors = np.asarray(predictors, dtype=np.float64)
    if predictors.ndim == 1:
        predictors = predictors[:, None]
    n_positives = labels.sum(axis=0)
    n_negatives = labels.shape[0] - n_positives
    ranks = stats.rankdata(predictors, axis=0, method="average")
    sum_of_positive_ranks = labels.T @ ranks
    result = (sum_of_positive_ranks / n_positives[:, None] - (n_positives[:, None] + 1) / 2) / n_negatives[:, None]
    if compute_tie_correction:
        n = ranks.shape[0]
        observed_var = ranks.var(axis=0, ddof=1)
        tie_correction = (_max_rank_variance(n) - observed_var) * 12 / n
    else:
        tie_correction = 0
    return result, tie_correction


def _max_rank_variance(n: int) -> float:
    """Sample variance of 1..n (ranks without ties)."""
    return n * (n + 1) / 12 if n > 1 else float("nan")


def auroc_p_value(aurocs, n_positives, n_negatives, two_tailed: bool = True, tie_correction=0, log_p: bool = False):
    """ROC test p-values (normal approximation of the AUROC).

    ``aurocs`` is a categories x genes array, ``n_positives`` and
    ``n_negatives`` have one value per category and ``tie_correction`` is a
    scalar or one value per gene.
    """
    aurocs = np.asarray(aurocs, dtype=np.float64)
    p = np.asarray(n_positives, dtype=np.float64)[:, None]
    n = np.asarray(n_negatives, dtype=np.float64)[:, None]
    n_samples = p[0, 0] + n[0, 0]
    tie_correction = np.asarray(tie_correction, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if tie_correction.size > 1:
            Z = (aurocs - 0.5) * np.sqrt(12 * n * p)
            Z = Z / np.sqrt(n_samples + 1 - tie_correction)[None, :]
        else:
            Z = (aurocs - 0.5) / np.sqrt((n_samples + 1 - tie_correction.reshape(())) / (12 * n * p))
    if two_tailed:
        if log_p:
            result = np.where(Z <= 0, special.log_ndtr(Z), special.log_ndtr(-Z)) + np.log(2)
        else:
            result = np.where(Z <= 0, special.ndtr(Z), special.ndtr(-Z)) * 2
        result[np.isnan(Z)] = np.nan
    else:
        result = special.log_ndtr(-Z) if log_p else special.ndtr(-Z)
    return result


def fdr(p_values, log_p: bool = False) -> np.ndarray:
    """Benjamini-Hochberg FDR (as R's p.adjust(method="fdr")), optionally on log p-values."""
    p_values = np.asarray(p_values, dtype=np.float64)
    shape = p_values.shape
    p_values = p_values.ravel()
    n = p_values.size
    if n == 0:
        return p_values.reshape(shape)
    i = np.arange(n, 0, -1, dtype=np.float64)
    order = np.argsort(-p_values, kind="stable")  # decreasing, NaN last
    with np.errstate(invalid="ignore"):
        if log_p:
            adjusted = np.minimum(0, np.minimum.accumulate(np.log(n) - np.log(i) + p_values[order]))
        else:
            adjusted = np.minimum(1, np.minimum.accumulate(n / i * p_values[order]))
    result = np.empty(n)
    result[order] = adjusted
    return result.reshape(shape)
