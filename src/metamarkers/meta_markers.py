"""Meta-analytic markers: aggregation of marker statistics across datasets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import _io
from .markers import remove_duplicated_genes

META_COLUMNS = [
    "group", "cell_type", "rank", "gene", "recurrence", "auroc", "fold_change",
    "fold_change_detection", "expression", "precision", "recall", "population_size", "n_datasets",
]


def make_meta_markers(
    marker_lists: Mapping[str, pd.DataFrame],
    order_by="auroc",
    fdr_threshold: float = 0.05,
    fc_threshold: float = 4,
    detection_threshold: float = 0,
    detailed_stats: bool = False,
    common_genes_only: bool = True,
    check_duplicates: bool = False,
) -> pd.DataFrame:
    """Extract meta-analytic markers from multiple datasets.

    Parameters
    ----------
    marker_lists
        Mapping from dataset name to the output of ``compute_markers``.
    order_by
        Secondary statistic(s) by which meta-markers are ranked (after recurrence).
    fdr_threshold, fc_threshold, detection_threshold
        A gene is DE in a dataset if FDR <= fdr_threshold, fold change >=
        fc_threshold and detection rate >= detection_threshold. (The R package
        ignores detection_threshold; results only differ when it is above 0.)
    detailed_stats
        If True, return average statistics and per-dataset DE status, otherwise
        only (group, cell_type, gene, rank).
    common_genes_only
        Keep only genes present in all datasets.
    check_duplicates
        Remove duplicated genes from each marker list first.

    Returns
    -------
    DataFrame of ranked meta-markers. Within each (group, cell type), genes are
    ranked by recurrence (number of datasets where the gene is DE), then by
    ``order_by``. Statistics are averaged across datasets (geometric mean for
    fold_change, fold_change_detection and expression).
    """
    if not isinstance(marker_lists, Mapping):
        raise TypeError("marker_lists must be a mapping from dataset name to marker table.")
    if check_duplicates:
        marker_lists = {name: remove_duplicated_genes(markers) for name, markers in marker_lists.items()}
    frames = [markers.assign(dataset=str(name)) for name, markers in marker_lists.items()]
    all_markers = pd.concat(frames, ignore_index=True)
    if common_genes_only:
        common = set.intersection(*(set(markers["gene"]) for markers in marker_lists.values()))
        all_markers = all_markers[all_markers["gene"].isin(common)].reset_index(drop=True)

    is_de = _and3(
        _and3(_compare(all_markers["log_fdr"], np.log(fdr_threshold), "le"),
              _compare(all_markers["fold_change"], fc_threshold, "ge")),
        _compare(all_markers["detection_rate"], detection_threshold, "ge"),
    )

    keys = all_markers[["group", "cell_type", "gene"]]
    codes = keys.groupby(["group", "cell_type", "gene"], sort=False, dropna=False).ngroup().to_numpy()
    first = np.unique(codes, return_index=True)[1]
    n = np.bincount(codes).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = pd.DataFrame({
            "group": keys["group"].to_numpy()[first],
            "cell_type": keys["cell_type"].to_numpy()[first],
            "gene": keys["gene"].to_numpy()[first],
            "recurrence": np.bincount(codes, weights=is_de),
            "auroc": _mean(codes, all_markers["auroc"], n),
            "fold_change": _gmean(codes, all_markers["fold_change"], n),
            "fold_change_detection": _gmean(codes, all_markers["fold_change_detection"], n),
            "expression": _gmean(codes, all_markers["average_expression"], n),
            "precision": _mean(codes, all_markers["precision"], n),
            "recall": _mean(codes, all_markers["detection_rate"], n),
            "population_size": _mean(codes, all_markers["population_size"], n),
            "n_datasets": n.astype(np.int64),
        })
    if not result["recurrence"].isna().any():
        result["recurrence"] = result["recurrence"].astype(np.int64)

    order_by = [order_by] if isinstance(order_by, str) else list(order_by)
    sort_keys = [-result[column].to_numpy(dtype=np.float64) for column in reversed(order_by)]
    sort_keys += [-result["recurrence"].to_numpy(dtype=np.float64), _string_codes(result["cell_type"]),
                  _string_codes(result["group"])]
    order = np.lexsort(sort_keys)
    result = result.iloc[order].reset_index(drop=True)
    segment = result.groupby(["group", "cell_type"], sort=False).cumcount().to_numpy()
    result.insert(2, "rank", segment + 1)
    result = result[META_COLUMNS]

    if not detailed_stats:
        return result[["group", "cell_type", "gene", "rank"]]

    datasets = pd.unique(all_markers["dataset"])
    position = np.empty(len(order), dtype=np.int64)  # position of each key after sorting
    position[order] = np.arange(len(order))
    for dataset in datasets:
        rows = np.flatnonzero(all_markers["dataset"].to_numpy() == dataset)
        values = np.full(len(result), np.nan)
        values[position[codes[rows]]] = is_de[rows]
        if np.isnan(values).any():
            result[dataset] = pd.array([pd.NA if np.isnan(v) else bool(v) for v in values], dtype="boolean")
        else:
            result[dataset] = values.astype(bool)
    return result


def _compare(values: pd.Series, threshold, operator: str) -> np.ndarray:
    """Comparison with R semantics: NaN where the value is missing."""
    values = values.to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        result = (values <= threshold) if operator == "le" else (values >= threshold)
    result = result.astype(np.float64)
    result[np.isnan(values)] = np.nan
    return result


def _and3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Three-valued logical AND (R's & with NA)."""
    result = np.where((a == 0) | (b == 0), 0.0, np.where(np.isnan(a) | np.isnan(b), np.nan, 1.0))
    return result


def _mean(codes: np.ndarray, values, n: np.ndarray) -> np.ndarray:
    return np.bincount(codes, weights=np.asarray(values, dtype=np.float64), minlength=len(n)) / n


def _gmean(codes: np.ndarray, values, n: np.ndarray) -> np.ndarray:
    return np.exp(_mean(codes, np.log(np.asarray(values, dtype=np.float64)), n))


def _string_codes(values: pd.Series) -> np.ndarray:
    """Integer codes following C-locale string order."""
    levels, codes = np.unique(values.to_numpy(dtype=object).astype(str), return_inverse=True)
    return codes


# ---- import / export ----------------------------------------------------------------

def export_meta_markers(
    meta_markers: pd.DataFrame,
    filename,
    dataset_names,
    order_by="auroc",
    fdr_threshold: float = 0.05,
    fc_threshold: float = 4,
    detection_threshold: float = 0,
    gzip: bool = True,
) -> str:
    """Export meta-markers to CSV with a header (time stamp, datasets, DE criteria)."""
    order_by = order_by if isinstance(order_by, str) else ", ".join(order_by)
    header = (
        f"# Meta marker list generated on {_io.today()}. "
        f"Ordered by recurrence among DE genes (FC >= {_io.format_number(fc_threshold)}, "
        f"FDR <= {_io.format_number(fdr_threshold)}, detection_rate >= {_io.format_number(detection_threshold)}), "
        f"then {order_by}. Based on: {', '.join(map(str, dataset_names))}. "
        "All stats are averaged across datasets, last columns indicate in which dataset the gene is DE."
    )
    return _io.write_table(meta_markers, filename, header, gzip)


def export_meta_markers_by_cell_type(
    meta_markers: pd.DataFrame,
    output_prefix,
    dataset_names,
    order_by="auroc",
    fdr_threshold: float = 0.05,
    fc_threshold: float = 4,
    detection_threshold: float = 0,
    gzip: bool = True,
) -> list[str]:
    """Export meta-markers, one file per cell type (prefix + make.names(cell_type) + .csv)."""
    paths = []
    for cell_type in pd.unique(meta_markers["cell_type"]):
        filename = f"{output_prefix}{_io.make_names(cell_type)}.csv"
        paths.append(export_meta_markers(
            meta_markers[meta_markers["cell_type"] == cell_type], filename, dataset_names,
            order_by, fdr_threshold, fc_threshold, detection_threshold, gzip,
        ))
    return paths


def read_meta_markers(filename, header_size: int = 1) -> pd.DataFrame:
    """Import meta-markers written by ``export_meta_markers`` (Python or R)."""
    return _io.read_table(filename, header_size)
