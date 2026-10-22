"""Library size normalization."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._data import resolve_expression


def convert_to_cpm(M, scale_factor: float = 1_000_000, genes_are_rows: bool = False, layer=None):
    """Convert an expression matrix to CPM (library size normalization).

    Equivalent to R's ``convert_to_cpm``: every cell is divided by its library
    size and multiplied by ``scale_factor``.

    Parameters
    ----------
    M
        Expression matrix (sparse or dense) or AnnData object. Cells are rows
        by default (``genes_are_rows=False``), unlike the R package.
    scale_factor
        Cell counts sum to this value after normalization (1e6 by default).
    genes_are_rows
        Set to True for genes x cells matrices (the R convention).
    layer
        For AnnData input, normalize this layer instead of X.

    Returns
    -------
    Matrix in the same format (and orientation) as the input. For AnnData
    input, the normalized matrix (cells x genes) is returned.
    """
    if not sp.issparse(M) and not isinstance(M, np.ndarray):
        expression = resolve_expression(M, layer=layer)
        try:
            source = expression.source
            X = source.X if hasattr(source, "X") else None
            if X is None:
                raise TypeError("convert_to_cpm needs an in-memory matrix; use normalize='cpm' for h5ad files.")
            return convert_to_cpm(X, scale_factor)
        finally:
            expression.close()
    if genes_are_rows:
        return convert_to_cpm(M.T, scale_factor).T
    normalization_factor = _library_sizes(M) / scale_factor
    if sp.issparse(M):
        M = M.copy().astype(np.float64) if M.format in ("csr", "csc") else sp.csr_matrix(M, dtype=np.float64)
        if M.format == "csr":
            rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
        else:
            rows = M.indices
        M.data = M.data / normalization_factor[rows]
        return M
    return np.asarray(M, dtype=np.float64) / normalization_factor[:, None]


def _library_sizes(M) -> np.ndarray:
    """Sum of each row (cell)."""
    return np.asarray(M.sum(axis=1, dtype=np.float64)).ravel()


def _normalize_block(block, factors, rows, transform):
    """Apply CPM (and optionally log1p) to the stored values of a block in place."""
    if transform is None:
        return block
    block.data = block.data / factors[rows]
    if transform == "log1p_cpm":
        block.data = np.log1p(block.data)
    return block
