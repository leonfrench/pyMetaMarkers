"""Differential expression statistics for individual datasets."""

from __future__ import annotations

import itertools
import os
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _io
from ._data import DEFAULT_MAX_NNZ, resolve_expression
from .aurocs import _sorted_levels, auroc_p_value, fdr
from .normalization import _normalize_block

MARKER_COLUMNS = [
    "group", "cell_type", "gene", "fold_change", "auroc", "log_fdr", "population_size",
    "population_fraction", "average_expression", "se_expression", "detection_rate",
    "fold_change_detection", "precision", "recall",
]


@dataclass
class _Group:
    name: str
    rows: np.ndarray            # cell indices in the full matrix
    cell_types: list[str]       # sorted
    codes: np.ndarray           # cell type index of each cell in rows
    n_positives: np.ndarray     # cells per cell type
    all_rows: bool = False
    stats: dict = field(default_factory=dict)

    @property
    def n_cells(self):
        return len(self.rows)

    @property
    def n_negatives(self):
        return self.n_cells - self.n_positives


def compute_markers(
    data,
    cell_type_labels,
    group_labels=None,
    two_tailed: bool = False,
    tie_correction: bool = False,
    genes_are_rows: bool = False,
    *,
    gene_names=None,
    layer=None,
    normalize=None,
    max_nnz_per_block: int | None = None,
    n_jobs: int | None = None,
    tmp_dir=None,
) -> pd.DataFrame:
    """Compute differential expression statistics for all cell types, stratified by groups.

    All statistics are 1-vs-rest within groups, as in R's ``compute_markers``.
    Genes with duplicated names are removed from the output.

    Parameters
    ----------
    data
        AnnData object (in memory or backed), path to an h5ad file, or a
        cells x genes matrix (dense or scipy sparse). h5ad files are read in
        blocks, so datasets larger than memory are supported.
    cell_type_labels
        Cell type of each cell, or the name of an ``obs`` column. Cells with a
        missing label (None/NaN) are ignored.
    group_labels
        Group of each cell (or an ``obs`` column name). Cell types are compared
        to other cell types of the same group only. By default all cells belong
        to the group "all".
    two_tailed
        If False, only upregulated genes are considered significant.
    tie_correction
        Apply tie correction in the ROC test (slower, slightly less conservative).
    genes_are_rows
        Set to True for genes x cells matrices (the R convention). Ignored for
        AnnData input, where cells are rows.
    gene_names
        Gene names: a ``var`` column name or a list. Defaults to ``var_names``.
    layer
        Use this AnnData layer instead of X.
    normalize
        None (use values as provided) or "cpm" to apply ``convert_to_cpm`` on the
        fly (useful for h5ad files that only contain counts).
    max_nnz_per_block
        Maximum number of stored values in a block of genes. By default about
        20 million values are processed at once in total (2.5 to 10 million per
        block, depending on n_jobs), i.e. about 2 GB of working memory.
    n_jobs
        Number of gene blocks processed in parallel (threads); None (default)
        uses all CPU cores. Peak memory grows with n_jobs x max_nnz_per_block.
    tmp_dir
        Directory for temporary files when transposing row-major h5ad matrices.

    Returns
    -------
    DataFrame with one row per (group, cell type, gene), sorted by group (order
    of appearance), cell type and decreasing AUROC.
    """
    if normalize not in (None, "cpm"):
        raise ValueError("normalize must be None or 'cpm'.")
    expression = resolve_expression(data, layer=layer, genes_are_rows=genes_are_rows)
    try:
        genes = expression.gene_names(gene_names)
        cell_types = expression.labels(cell_type_labels, "cell_type_labels")
        if group_labels is None:
            groups = np.full(len(cell_types), "all", dtype=object)
        else:
            groups = expression.labels(group_labels, "group_labels")
        plans = _plan_groups(cell_types, groups)
        if not plans:
            return pd.DataFrame({column: [] for column in MARKER_COLUMNS})
        n_genes = len(genes)
        n_threads = (os.cpu_count() or 1) if n_jobs is None else max(int(n_jobs), 1)
        if max_nnz_per_block is None:
            max_nnz_per_block = int(np.clip(TOTAL_NNZ_IN_FLIGHT // n_threads, 2_500_000, DEFAULT_MAX_NNZ))
        for plan in plans:
            _allocate(plan, n_genes, tie_correction)
        # start reading blocks first: row-major files compute library sizes while being split
        blocks = expression.source.column_blocks(max_nnz_per_block, tmp_dir)
        first = next(blocks, None)
        factors = expression.source.row_sums() / 1_000_000 if normalize == "cpm" else None

        def process(start, stop, block):
            if factors is not None:
                _normalize_block(block, factors, block.indices, "cpm")
            for plan in plans:
                sub = block if plan.all_rows else block[plan.rows, :]
                # fills columns start:stop; with threads, use the GIL-friendly ranking
                _block_statistics(sub, plan, start, stop, tie_correction, vectorized_ranks=n_threads > 1)

        if first is not None:
            _run_blocks(process, itertools.chain([first], blocks), n_threads)
    finally:
        expression.close()
    tables = [_marker_table(plan, genes, two_tailed, tie_correction) for plan in plans]
    result = pd.concat(tables, ignore_index=True)
    return _remove_duplicated_names(result, genes)


def _run_blocks(process, blocks, n_jobs):
    """Apply process(start, stop, block) to every block, with up to n_jobs blocks in flight.

    Blocks are read in the calling thread; the heavy numpy work (sorting,
    bincounts) runs in worker threads.
    """
    if n_jobs == 1:
        for start, stop, block in blocks:
            process(start, stop, block)
        return
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        pending = deque()
        for start, stop, block in blocks:
            pending.append(pool.submit(process, start, stop, block))
            while len(pending) >= n_jobs:
                pending.popleft().result()
        for future in pending:
            future.result()


def _plan_groups(cell_types, groups) -> list[_Group]:
    valid = np.array([c is not None and g is not None for c, g in zip(cell_types, groups)], dtype=bool)
    if any(c == "" for c in cell_types[valid]):
        raise ValueError(
            "All cells must have a label, please remove cells without labels or consider renaming (e.g., unknown)."
        )
    plans = []
    for group in pd.unique(groups[valid]):
        rows = np.flatnonzero(valid & (groups == group))
        labels = cell_types[rows]
        levels = _sorted_levels(labels)
        if len(levels) < 2:
            warnings.warn(f"Skipping group {group}: only 1 cluster ({levels[0]}).", stacklevel=3)
            continue
        codes = pd.Categorical(labels, categories=levels).codes.astype(np.int64)  # int64: k * width may be large
        plans.append(_Group(
            name=str(group), rows=rows, cell_types=levels, codes=codes,
            n_positives=np.bincount(codes, minlength=len(levels)).astype(np.float64),
            all_rows=len(rows) == len(cell_types),
        ))
    return plans


#: Default number of stored values processed at once across all threads.
TOTAL_NNZ_IN_FLIGHT = 20_000_000

#: With threads, blocks whose genes have fewer stored values than this on average
#: are ranked with whole-block operations (see _ranks_vectorized).
SMALL_COLUMN_NNZ = 5000


def _allocate(plan: _Group, n_genes: int, tie_correction: bool):
    k = len(plan.cell_types)
    for name in ("auroc", "sums", "sums_of_squares"):
        plan.stats[name] = np.zeros((k, n_genes))
    plan.stats["n_expressing"] = np.zeros((k, n_genes), dtype=np.int64)
    if tie_correction:
        plan.stats["tie_correction"] = np.zeros(n_genes)


def _block_statistics(block: sp.csc_matrix, plan: _Group, start: int, stop: int, tie_correction: bool,
                      vectorized_ranks: bool = False):
    """Per cell type AUROCs, sums, sums of squares and detection counts for genes start:stop."""
    n_cells, width = block.shape
    k = len(plan.cell_types)
    x, indptr = block.data, block.indptr
    nnz_per_gene = np.diff(indptr)
    n_zeros = n_cells - nnz_per_gene
    index = plan.codes[block.indices] * width + np.repeat(np.arange(width), nnz_per_gene)

    def per_cell_type(weights=None):
        return np.bincount(index, weights=weights, minlength=k * width).reshape(k, width)

    # ranks among all cells (zeros included) of the stored values
    # the per-column loop is fastest unless threads compete for the GIL on small columns
    small_columns = block.nnz < SMALL_COLUMN_NNZ * max(width, 1)
    rank_function = _ranks_vectorized if vectorized_ranks and small_columns else _ranks_within_columns
    ranks, n_negative, tie_sums = rank_function(x, indptr, n_zeros)
    zero_rank = n_negative + (n_zeros + 1) / 2
    n_expressing = per_cell_type()
    rank_sum = per_cell_type(ranks) + (plan.n_positives[:, None] - n_expressing) * zero_rank
    n_pos, n_neg = plan.n_positives[:, None], plan.n_negatives[:, None]
    stats = plan.stats
    stats["auroc"][:, start:stop] = (rank_sum / n_pos - (n_pos + 1) / 2) / n_neg
    stats["n_expressing"][:, start:stop] = n_expressing
    stats["sums"][:, start:stop] = per_cell_type(x)
    stats["sums_of_squares"][:, start:stop] = per_cell_type(x * x)
    if tie_correction:
        # sum of (t^3 - t) / (N (N - 1)) over groups of t tied values (zeros included)
        zeros = n_zeros.astype(np.float64)
        stats["tie_correction"][start:stop] = (tie_sums + zeros ** 3 - zeros) / (n_cells * (n_cells - 1.0))


def _ranks_vectorized(x: np.ndarray, indptr: np.ndarray, n_zeros: np.ndarray):
    """Same as _ranks_within_columns with whole-block numpy operations.

    About twice as slow on one thread, but its large numpy calls release the GIL,
    so it scales with threads (the per-column loop does not).
    """
    n_columns = len(indptr) - 1
    nnz_per_column = np.diff(indptr)
    column = np.repeat(np.arange(n_columns, dtype=np.uint16 if n_columns < 2**16 else np.int64), nnz_per_column)
    order = np.argsort(x)  # tied values get the same (average) rank, so stability is not needed
    order = order[np.argsort(column[order], kind="stable")]  # radix sort on 16-bit keys
    sorted_values, sorted_column = x[order], column[order]
    new_run = np.empty(len(x), dtype=bool)
    new_run[:1] = True
    new_run[1:] = (sorted_column[1:] != sorted_column[:-1]) | (sorted_values[1:] != sorted_values[:-1])
    run_start = np.flatnonzero(new_run)
    run_end = np.append(run_start[1:], len(x)) - 1
    run_column = sorted_column[run_start]
    rank = ((run_start + run_end) / 2 + 1 - indptr[run_column])[np.cumsum(new_run) - 1]
    negative = sorted_values < 0
    rank[~negative] += n_zeros[sorted_column[~negative]]
    ranks = np.empty(len(x))
    ranks[order] = rank
    n_negative = np.bincount(sorted_column[negative], minlength=n_columns).astype(np.float64)
    run_length = (run_end - run_start + 1).astype(np.float64)
    tie_sums = np.bincount(run_column, weights=run_length ** 3 - run_length, minlength=n_columns)
    return ranks, n_negative, tie_sums


def _ranks_within_columns(x: np.ndarray, indptr: np.ndarray, n_zeros: np.ndarray):
    """Average ranks (among all cells, zeros included) of the stored values of each CSC column.

    Returns the ranks, the number of negative values per column (zeros rank
    after negative values and before positive values), and the sum of
    t^3 - t over groups of t tied stored values per column (for tie correction).
    Sorting each column separately is about twice as fast as one global sort.
    """
    ranks = np.empty(len(x))
    n_columns = len(indptr) - 1
    n_negative = np.zeros(n_columns)
    tie_sums = np.zeros(n_columns)
    for j in range(n_columns):
        start, stop = indptr[j], indptr[j + 1]
        if stop == start:
            continue
        values = x[start:stop]
        order = np.argsort(values)  # tied values get the same (average) rank
        sorted_values = values[order]
        new_run = np.empty(len(values), dtype=bool)
        new_run[0] = True
        np.not_equal(sorted_values[1:], sorted_values[:-1], out=new_run[1:])
        run_start = np.flatnonzero(new_run)
        run_end = np.append(run_start[1:], len(values)) - 1
        rank = ((run_start + run_end) / 2 + 1)[np.cumsum(new_run) - 1]
        negative = sorted_values < 0
        n_negative[j] = np.count_nonzero(negative)
        rank[~negative] += n_zeros[j]
        ranks[start + order] = rank
        run_length = (run_end - run_start + 1).astype(np.float64)
        tie_sums[j] = np.sum(run_length ** 3 - run_length)
    return ranks, n_negative, tie_sums


def _others(values: np.ndarray) -> np.ndarray:
    """For each row k, the sum of all other rows (without cancellation errors)."""
    before = np.vstack([np.zeros_like(values[:1]), np.cumsum(values, axis=0)[:-1]])
    after = np.vstack([np.cumsum(values[::-1], axis=0)[::-1][1:], np.zeros_like(values[:1])])
    return before + after


def _marker_table(plan: _Group, genes: np.ndarray, two_tailed: bool, tie_correction: bool) -> pd.DataFrame:
    stats = plan.stats
    k, n_genes = stats["auroc"].shape
    n_pos = plan.n_positives[:, None]
    n_neg = plan.n_negatives[:, None]
    auroc = stats["auroc"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = auroc_p_value(
            auroc, plan.n_positives, plan.n_negatives, two_tailed,
            tie_correction=stats["tie_correction"] if tie_correction else 0, log_p=True,
        )
        log_fdr = fdr(log_p, log_p=True)
        positives = stats["sums"] / n_pos
        negatives = _others(stats["sums"]) / n_neg
        fold_change = (positives + 1) / (negatives + 1)
        standard_error = np.sqrt((stats["sums_of_squares"] / n_pos - positives ** 2) / n_pos)
        n_expressing = stats["n_expressing"]
        total_expressing = n_expressing.sum(axis=0)
        detection = n_expressing / n_pos
        detection_background = (total_expressing[None, :] - n_expressing) / n_neg
        fold_change_detection = (detection + 0.001) / (detection_background + 0.001)
        precision = n_expressing / total_expressing[None, :]
        recall = n_expressing / n_pos

    cell_type_index = np.repeat(np.arange(k), n_genes)
    order = np.lexsort((-auroc.ravel(), cell_type_index))
    population_size = np.repeat(plan.n_positives, n_genes)
    columns = {
        "group": np.full(k * n_genes, plan.name, dtype=object),
        "cell_type": np.asarray(plan.cell_types, dtype=object)[cell_type_index],
        "gene": np.tile(genes, k),
        "fold_change": fold_change.ravel(),
        "auroc": auroc.ravel(),
        "log_fdr": log_fdr.ravel(),
        "population_size": population_size,
        "population_fraction": population_size / plan.n_positives.sum(),
        "average_expression": positives.ravel(),
        "se_expression": standard_error.ravel(),
        "detection_rate": detection.ravel(),
        "fold_change_detection": fold_change_detection.ravel(),
        "precision": precision.ravel(),
        "recall": recall.ravel(),
    }
    return pd.DataFrame({name: values[order] for name, values in columns.items()})


def _remove_duplicated_names(result: pd.DataFrame, genes: np.ndarray) -> pd.DataFrame:
    names = pd.Series(genes)
    duplicated = set(names[names.duplicated()])
    if not duplicated:
        return result
    return result[~result["gene"].isin(duplicated)].reset_index(drop=True)


def remove_duplicated_genes(marker_stats: pd.DataFrame) -> pd.DataFrame:
    """Remove genes that appear more than once for a (group, cell type) pair."""
    counts = marker_stats.groupby(["group", "cell_type", "gene"], sort=False).size()
    duplicated = set(counts[counts > 1].index.get_level_values("gene"))
    return marker_stats[~marker_stats["gene"].isin(duplicated)]


# ---- import / export ----------------------------------------------------------------

def export_markers(marker_stats: pd.DataFrame, filename, gzip: bool = True) -> str:
    """Export marker statistics to CSV, with a one-line header (date, cell types).

    With ``gzip=True`` (default) the file is compressed and ".gz" is appended
    to the file name, as in R. Returns the path of the written file.
    """
    cell_types = pd.unique(marker_stats["cell_type"])
    header = (
        f"# Differential expression statistics generated on {_io.today()}. "
        f"Cell types: {', '.join(map(str, cell_types))}."
    )
    return _io.write_table(marker_stats, filename, header, gzip)


def export_markers_by_group(de_stats: pd.DataFrame, output_prefix, gzip: bool = True) -> list[str]:
    """Export marker statistics, one file per group (prefix + make.names(group) + .csv).

    Unlike the R function (which writes all groups to every file), each file
    only contains the statistics of its group.
    """
    paths = []
    for group in pd.unique(de_stats["group"]):
        filename = f"{output_prefix}{_io.make_names(group)}.csv"
        paths.append(export_markers(de_stats[de_stats["group"] == group], filename, gzip))
    return paths


def export_markers_by_cell_type(de_stats: pd.DataFrame, output_prefix, gzip: bool = True) -> list[str]:
    """Export marker statistics, one file per cell type (prefix + make.names(cell_type) + .csv)."""
    paths = []
    for cell_type in pd.unique(de_stats["cell_type"]):
        filename = f"{output_prefix}{_io.make_names(cell_type)}.csv"
        paths.append(export_markers_for_cell_type(de_stats, cell_type, filename, gzip))
    return paths


def export_markers_for_cell_type(de_stats: pd.DataFrame, cell_type, filename, gzip: bool = True) -> str:
    """Export the marker statistics of one cell type, with a header listing outgroups.

    Unlike the R function (which writes all cell types), only rows of the
    requested cell type are written.
    """
    all_cell_types = pd.unique(de_stats["cell_type"])
    outgroups = [ct for ct in all_cell_types if ct != cell_type]
    header = (
        f"# Differential expression statistics generated on {_io.today()}. "
        f"Reference: {cell_type}, outgroups: {', '.join(map(str, outgroups))}."
    )
    return _io.write_table(de_stats[de_stats["cell_type"] == cell_type], filename, header, gzip)


def read_markers(filename, header_size: int = 1) -> pd.DataFrame:
    """Import marker statistics written by ``export_markers`` (Python or R)."""
    return _io.read_table(filename, header_size)
