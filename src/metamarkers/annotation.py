"""Cell type annotation with marker sets.

Score matrices are cells x marker sets (one row per cell, like AnnData), which
is the transpose of the R package's marker sets x cells matrices.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._data import DEFAULT_MAX_NNZ, InMemorySource, resolve_expression
from .aurocs import _as_label_matrix, _sorted_levels, compute_aurocs
from .normalization import _normalize_block

PSEUDOCOUNT = 0.0001


def score_cells(
    expr,
    marker_set,
    *,
    gene_names=None,
    layer=None,
    genes_are_rows: bool = False,
    normalize=None,
    max_nnz_per_block: int = DEFAULT_MAX_NNZ,
) -> pd.DataFrame:
    """Compute marker scores (average marker expression) in a target dataset.

    Parameters
    ----------
    expr
        AnnData (in memory or backed), h5ad path, or cells x genes matrix. Log-normalized
        values are recommended (or use ``normalize="log1p_cpm"`` on counts).
    marker_set
        A marker table from MetaMarkers (columns group, cell_type, gene), a
        mapping {cell type: genes}, or a genes x marker sets weight matrix.
    gene_names
        ``var`` column or list of gene names (defaults to ``var_names``).
    normalize
        None, "cpm" or "log1p_cpm" (log1p of CPM, as in the R vignettes),
        applied on the fly.

    Returns
    -------
    DataFrame of scores, cells x marker sets (marker sets are named "group|cell_type").
    """
    if normalize not in (None, "cpm", "log1p_cpm"):
        raise ValueError("normalize must be None, 'cpm' or 'log1p_cpm'.")
    expression = resolve_expression(expr, layer=layer, genes_are_rows=genes_are_rows)
    try:
        genes = expression.gene_names(gene_names)
        matrix, set_names = _marker_matrix(marker_set, genes)
        source = expression.source
        factors = source.row_sums() / 1_000_000 if normalize is not None else None
        used = np.flatnonzero(np.diff(matrix.indptr))  # genes with a weight
        weights = matrix[used].toarray()
        scores = np.zeros((source.n_obs, len(set_names)))
        if isinstance(source, InMemorySource):
            scores = _product(source.X[:, used], weights, factors, normalize, 0)
        elif source.column_major:
            for start, stop, block in source.column_blocks(max_nnz_per_block):
                in_block = (used >= start) & (used < stop)
                if in_block.any():
                    columns = block[:, used[in_block] - start]
                    scores += _product(columns, weights[in_block], factors, normalize, 0)
        else:
            for start, stop, chunk in source.row_chunks(max_nnz_per_block):
                scores[start:stop] = _product(chunk[:, used], weights, factors, normalize, start)
        index = expression.obs_names if expression.obs_names is not None else None
    finally:
        expression.close()
    return pd.DataFrame(scores, index=index, columns=set_names)


def _product(X, weights: np.ndarray, factors, normalize, row_offset: int) -> np.ndarray:
    """(Optionally normalized) cells x genes matrix times genes x marker sets weights."""
    if sp.issparse(X):
        X = sp.csr_matrix(X, dtype=np.float64)
        if factors is not None:
            rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr)) + row_offset
            _normalize_block(X, factors, rows, normalize)
        return np.asarray(X @ weights)
    X = np.asarray(X, dtype=np.float64)
    if factors is not None:
        X = X / factors[row_offset:row_offset + X.shape[0], None]
        if normalize == "log1p_cpm":
            X = np.log1p(X)
    return X @ weights


def _marker_matrix(marker_set, genes) -> tuple[sp.csr_matrix, list[str]]:
    if isinstance(marker_set, pd.DataFrame) and {"cell_type", "gene"} <= set(marker_set.columns):
        return _table_to_sparse(marker_set, genes)
    if isinstance(marker_set, Mapping):
        return _table_to_sparse(_list_to_table(marker_set), genes)
    if isinstance(marker_set, pd.DataFrame):
        names = [str(c) for c in marker_set.columns]
        if all(isinstance(dtype, pd.SparseDtype) for dtype in marker_set.dtypes):
            matrix = sp.csr_matrix(marker_set.sparse.to_coo(), dtype=np.float64)
        else:
            matrix = sp.csr_matrix(marker_set.to_numpy(dtype=np.float64))
    else:
        matrix = sp.csr_matrix(marker_set, dtype=np.float64)
        names = [str(i) for i in range(matrix.shape[1])]
    if matrix.shape[0] != len(genes):
        raise ValueError(f"Weight matrix has {matrix.shape[0]} rows for {len(genes)} genes.")
    return matrix, names


def _list_to_table(marker_list: Mapping) -> pd.DataFrame:
    rows = [(str(cell_type), str(gene)) for cell_type, gene_list in marker_list.items()
            for gene in ([gene_list] if isinstance(gene_list, str) else gene_list)]
    table = pd.DataFrame(rows, columns=["cell_type", "gene"])
    table["group"] = "all"
    return table


def _table_to_sparse(marker_table: pd.DataFrame, known_genes, weighted: bool = True):
    known_genes = [str(g) for g in known_genes]
    gene_index = {}
    for i, gene in enumerate(known_genes):
        gene_index.setdefault(gene, i)
    group = marker_table["group"].astype(str) if "group" in marker_table.columns else "all"
    labels = (group + "|" + marker_table["cell_type"].astype(str)).to_numpy(dtype=object)
    genes = marker_table["gene"].astype(str).to_numpy(dtype=object)
    keep = np.array([gene in gene_index for gene in genes], dtype=bool)
    labels, genes = labels[keep], genes[keep]
    levels = _sorted_levels(labels)
    columns = pd.Categorical(labels, categories=levels).codes
    if weighted:
        counts = pd.Series(labels).value_counts()
        weights = 1 / counts.loc[labels].to_numpy(dtype=np.float64)
    else:
        weights = np.ones(len(labels))
    rows = np.array([gene_index[gene] for gene in genes], dtype=np.int64)
    matrix = sp.coo_matrix((weights, (rows, columns)), shape=(len(known_genes), len(levels))).tocsr()
    return matrix, levels


def marker_table_to_matrix(marker_table: pd.DataFrame, known_genes, weighted: bool = True) -> pd.DataFrame:
    """Convert a marker table to a (sparse) genes x marker sets weight matrix.

    If ``weighted``, coefficients are 1 / marker set size, otherwise 1.
    """
    matrix, levels = _table_to_sparse(marker_table, known_genes, weighted)
    matrix = sp.csc_matrix(matrix)
    columns = {
        name: pd.arrays.SparseArray(matrix[:, j].toarray().ravel(), fill_value=0.0)
        for j, name in enumerate(levels)
    }
    return pd.DataFrame(columns, index=list(map(str, known_genes)), columns=levels)


def marker_list_to_matrix(marker_list: Mapping, known_genes, weighted: bool = True) -> pd.DataFrame:
    """Convert a {cell type: genes} mapping to a genes x marker sets weight matrix."""
    return marker_table_to_matrix(_list_to_table(marker_list), known_genes, weighted)


def get_group(cell_type_label):
    """Group part of "group|cell_type" labels ("" for all labels if any lacks a "|")."""
    if isinstance(cell_type_label, str):
        return get_group([cell_type_label])[0]
    labels = [str(label) for label in cell_type_label]
    if all("|" in label for label in labels):
        return np.array([label.split("|", 1)[0] for label in labels], dtype=object)
    return np.array([""] * len(labels), dtype=object)


def get_cell_type(cell_type_label):
    """Cell type part of "group|cell_type" labels (labels unchanged if any lacks a "|")."""
    if isinstance(cell_type_label, str):
        return get_cell_type([cell_type_label])[0]
    labels = [str(label) for label in cell_type_label]
    if all("|" in label for label in labels):
        return np.array([label.split("|", 1)[1] for label in labels], dtype=object)
    return np.array(labels, dtype=object)


def compute_marker_enrichment(scores: pd.DataFrame, by_group: bool = True) -> pd.DataFrame:
    """Marker enrichment: score over the expected score if all marker sets were expressed equally.

    With ``by_group``, the expected score is the average over marker sets of the
    same group (requires "group|cell_type" marker set names).
    """
    scores = _as_scores(scores)
    names = list(scores.columns)
    values = scores.to_numpy(dtype=np.float64) + PSEUDOCOUNT
    group = get_group(names)
    if by_group and any(g == "" for g in group):
        warnings.warn("No group information: assuming that all cell types are at the same level.", stacklevel=2)
        by_group = False
    if not by_group:
        return pd.DataFrame(_relative_to_mean(values), index=scores.index, columns=names)
    parts, columns = [], []
    for g in pd.unique(group):
        selected = np.flatnonzero(group == g)
        subset = values[:, selected]
        parts.append(_relative_to_mean(subset))
        columns += [names[i] for i in selected]
    return pd.DataFrame(np.hstack(parts), index=scores.index, columns=columns)


def assign_cells(scores: pd.DataFrame, group_assignment=None) -> pd.DataFrame:
    """Assign each cell to the marker set with the highest score.

    Parameters
    ----------
    scores
        Cells x marker sets scores (``score_cells`` output).
    group_assignment
        Group assigned to each cell (hierarchical annotation): cells can only be
        assigned to marker sets of their group.

    Returns
    -------
    DataFrame with one row per cell: group, predicted cell type ("unassigned"
    for ties), score and enrichment of the predicted cell type.
    """
    scores = _as_scores(scores)
    names = np.asarray(scores.columns, dtype=object).astype(str)
    group = get_group(names)
    unique_groups = list(pd.unique(group))
    if group_assignment is None and len(unique_groups) > 1:
        warnings.warn(
            f"Detected group information in the score matrix ({', '.join(unique_groups)}) but no group "
            "assignments were provided. Usually this means that marker sets were intended to use "
            "hierarchically and groups needed to be assigned first. Proceeding without group "
            "information (any cell can be any cell type).", stacklevel=2,
        )
    values = scores.to_numpy(dtype=np.float64)
    if group_assignment is None:
        result = _assign(values, names)
    else:
        assignment = np.asarray(pd.Series(group_assignment, dtype=object).astype(object))
        if len(assignment) != len(values):
            raise ValueError(f"group_assignment has {len(assignment)} values for {len(values)} cells.")
        unique_assignments = list(pd.unique(assignment))
        if any(g == "" for g in unique_groups):
            raise ValueError("Group assignments provided but no group information in the score matrix. "
                             "Did you provide group information to score_cells?")
        unknown = [a for a in unique_assignments if a not in unique_groups]
        if unknown:
            warnings.warn(
                f"Some group assignments ({', '.join(map(str, unknown))}) do not match groups in the score "
                f"matrix ({', '.join(unique_groups)}) and will result in NA predictions.", stacklevel=2,
            )
        result = _assign(values[:0], names[:0])
        result = {column: np.empty(len(values), dtype=result[column].dtype) for column in result.columns}
        missing = pd.isna(assignment)
        for name in unique_assignments:
            if pd.isna(name):
                keep_cell, keep_set = np.flatnonzero(missing), np.array([], dtype=np.int64)
            else:
                keep_cell, keep_set = np.flatnonzero(~missing & (assignment == name)), np.flatnonzero(group == name)
            part = _assign(values[np.ix_(keep_cell, keep_set)], names[keep_set])
            for column in result:
                result[column][keep_cell] = part[column].to_numpy()
        result = pd.DataFrame(result)
    result.index = scores.index
    return result


def _assign(values: np.ndarray, names: np.ndarray) -> pd.DataFrame:
    n_cells, n_sets = values.shape
    if n_sets == 0 or n_cells == 0:
        return pd.DataFrame({
            "group": np.full(n_cells, None, dtype=object), "predicted": np.full(n_cells, None, dtype=object),
            "score": np.full(n_cells, np.nan), "enrichment": np.full(n_cells, np.nan),
        })
    first_index = values.argmax(axis=1)
    first_value = values[np.arange(n_cells), first_index]
    # a tie is when the two highest scores are equal (with one marker set, never)
    second_value = np.partition(values, -2, axis=1)[:, -2] if n_sets > 1 else np.full(n_cells, -np.inf)
    label = names[first_index]
    group_label = get_group(label)
    label = get_cell_type(label).astype(object)
    shifted = values + PSEUDOCOUNT
    enrichment = np.where(_constant_rows(shifted), 1.0, (first_value + PSEUDOCOUNT) / shifted.mean(axis=1))
    label[first_value == second_value] = "unassigned"
    return pd.DataFrame({"group": group_label, "predicted": label, "score": first_value, "enrichment": enrichment})


def summarize_auroc(scores: pd.DataFrame, true_labels) -> pd.DataFrame:
    """AUROC of each marker set for each true label (true labels x marker sets, as in R)."""
    scores = _as_scores(scores)
    labels, levels = _as_label_matrix(true_labels)
    aurocs, _ = compute_aurocs(scores.to_numpy(dtype=np.float64), labels)
    return pd.DataFrame(aurocs, index=levels, columns=scores.columns)


def summarize_fold_change(scores: pd.DataFrame, true_labels) -> pd.DataFrame:
    """Fold change of each marker set for each true label (marker sets x true labels, as in R)."""
    scores = _as_scores(scores)
    labels, levels = _as_label_matrix(true_labels)
    values = scores.to_numpy(dtype=np.float64).T
    positives = values @ (labels / labels.sum(axis=0))
    negatives = values @ ((1 - labels) / (1 - labels).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame(positives / negatives, index=scores.columns, columns=levels)


def summarize_precision_recall(scores: pd.DataFrame, true_labels, score_threshold) -> pd.DataFrame:
    """Annotation performance when cells are assigned to every marker set scoring above a threshold.

    Returns one row per (threshold, marker set, true label) with precision,
    recall, false positive rate, F1 and balanced accuracy.
    """
    scores = _as_scores(scores)
    labels, levels = _as_label_matrix(true_labels)
    thresholds = np.atleast_1d(np.asarray(score_threshold, dtype=np.float64))
    values = scores.to_numpy(dtype=np.float64)
    sets = np.asarray(scores.columns, dtype=object)
    p = labels.sum(axis=0)
    n = labels.shape[0] - p
    tables = []
    for threshold in thresholds:
        binary = (values > threshold).astype(np.float64)
        pp = binary.sum(axis=0)
        tp = binary.T @ labels
        fp = pp[:, None] - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = tp / pp[:, None]
            recall = tp / p[None, :]
            fpr = fp / n[None, :]
            f1 = 2 / (1 / precision + 1 / recall)
        tables.append(pd.DataFrame({
            "score_threshold": threshold,
            "marker_set": np.tile(sets, len(levels)),
            "true_label": np.repeat(np.asarray(levels, dtype=object), len(sets)),
            "precision": precision.ravel(order="F"),
            "recall": recall.ravel(order="F"),
            "fpr": fpr.ravel(order="F"),
            "f1": f1.ravel(order="F"),
            "balanced_accuracy": ((1 + recall - fpr) / 2).ravel(order="F"),
        }))
    return pd.concat(tables, ignore_index=True)


def _constant_rows(values: np.ndarray) -> np.ndarray:
    return values.max(axis=1, initial=-np.inf) == values.min(axis=1, initial=np.inf)


def _relative_to_mean(values: np.ndarray) -> np.ndarray:
    """values / row mean, exactly 1 when all values of a row are equal.

    With many marker sets, the floating-point mean of identical values can be
    one unit in the last place off, so cells without any marker expression would
    get an enrichment slightly above 1 and pass a threshold of 1.
    """
    result = values / values.mean(axis=1, keepdims=True)
    result[_constant_rows(values)] = 1.0
    return result


def _as_scores(scores) -> pd.DataFrame:
    if isinstance(scores, pd.DataFrame):
        return scores
    return pd.DataFrame(np.asarray(scores, dtype=np.float64))
