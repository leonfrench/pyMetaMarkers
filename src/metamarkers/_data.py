"""Access to expression matrices, in memory or on disk (h5ad), in bounded-memory blocks.

All matrices are handled as cells x genes (the AnnData convention). Marker
statistics need, for every gene, the values of all cells (to rank them), so the
central operation is iterating over blocks of genes as CSC matrices. Sources that
store genes contiguously (CSC in memory or on disk) are sliced directly. Row-major
sources (CSR or dense on disk) are transposed through temporary files, so that
memory stays bounded by ``max_nnz`` stored values whatever the dataset size.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._progress import QUIET

DEFAULT_MAX_NNZ = 10_000_000


def _plan_blocks(nnz_per_column: np.ndarray, max_nnz: int, max_width: int | None = None) -> list[tuple[int, int]]:
    """Split columns into contiguous blocks holding at most ``max_nnz`` values.

    A single column larger than ``max_nnz`` gets a block of its own.
    """
    max_nnz = max(int(max_nnz), 1)
    edges = []
    start, total = 0, 0
    for j, count in enumerate(np.asarray(nnz_per_column, dtype=np.int64)):
        too_wide = max_width is not None and j - start >= max_width
        if j > start and (total + count > max_nnz or too_wide):
            edges.append((start, j))
            start, total = j, 0
        total += count
    n = len(nnz_per_column)
    if n > start or n == 0:
        edges.append((start, n))
    return edges


def _canonical_csc(block) -> sp.csc_matrix:
    block = sp.csc_matrix(block, dtype=np.float64)
    block.sum_duplicates()
    block.eliminate_zeros()
    return block


class MatrixSource:
    """Cells x genes matrix that can be read in blocks of genes or chunks of cells."""

    shape: tuple[int, int]
    #: True if column blocks are cheap to extract (gene-major storage).
    column_major: bool = False

    @property
    def n_obs(self) -> int:
        return self.shape[0]

    @property
    def n_vars(self) -> int:
        return self.shape[1]

    def gene_nnz(self, log=QUIET) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        """Short description of the matrix and its storage (for verbose messages)."""
        return type(self).__name__

    def row_chunks(self, max_nnz: int = DEFAULT_MAX_NNZ) -> Iterator[tuple[int, int, object]]:
        """Yield (start, stop, chunk) where chunk holds rows start:stop (CSR or dense)."""
        raise NotImplementedError

    def column_blocks(self, max_nnz: int = DEFAULT_MAX_NNZ, tmp_dir=None,
                      log=QUIET) -> Iterator[tuple[int, int, sp.csc_matrix]]:
        """Yield (start, stop, block) where block is a float64 CSC matrix of genes start:stop."""
        edges = _plan_blocks(self.gene_nnz(log), max_nnz)
        if len(edges) == 1:
            start, stop = edges[0]
            chunks = [chunk for _, _, chunk in self.row_chunks(max_nnz)]
            if chunks and all(sp.issparse(c) for c in chunks):
                full = sp.vstack(chunks, format="csr")
            else:
                full = np.vstack([c.toarray() if sp.issparse(c) else np.asarray(c) for c in chunks])
            yield start, stop, _canonical_csc(full)
            return
        yield from _transpose_via_disk(self, edges, max_nnz, tmp_dir, log)

    #: library sizes computed as a by-product of a full pass (set by _transpose_via_disk)
    _cached_row_sums = None

    def row_sums(self, log=QUIET) -> np.ndarray:
        if self._cached_row_sums is not None:
            return self._cached_row_sums
        stage = log.stage("Computing library sizes (reading all cells)", self.n_obs, "cells")
        result = np.zeros(self.n_obs)
        for start, stop, chunk in self.row_chunks():
            result[start:stop] = _row_sums(chunk)
            stage.update(stop)
        stage.done()
        return result

    def close(self):
        pass


def _row_sums(chunk) -> np.ndarray:
    from .normalization import _library_sizes

    return _library_sizes(chunk)


class InMemorySource(MatrixSource):
    def __init__(self, X):
        if sp.issparse(X):
            if X.format not in ("csr", "csc"):
                X = sp.csr_matrix(X)
            self.column_major = X.format == "csc"
        else:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("Expression matrix must be 2-dimensional.")
        self.X = X
        self.shape = X.shape

    def describe(self) -> str:
        X = self.X
        if sp.issparse(X):
            return f"{X.format.upper()} matrix in memory ({X.dtype}, {X.nnz:,} stored values)"
        return f"dense matrix in memory ({X.dtype})"

    def gene_nnz(self, log=QUIET) -> np.ndarray:
        X = self.X
        if sp.issparse(X):
            if X.format == "csc":
                return np.diff(X.indptr)
            return np.bincount(X.indices, minlength=X.shape[1])
        return np.count_nonzero(X, axis=0)

    def row_chunks(self, max_nnz: int = DEFAULT_MAX_NNZ):
        X = self.X
        if sp.issparse(X) and X.format == "csr":
            nnz_rows = np.diff(X.indptr)
        elif sp.issparse(X):
            nnz_rows = np.bincount(X.indices, minlength=X.shape[0])
        else:
            nnz_rows = np.full(X.shape[0], X.shape[1])
        for start, stop in _plan_blocks(nnz_rows, max_nnz):
            yield start, stop, X[start:stop]

    def column_blocks(self, max_nnz: int = DEFAULT_MAX_NNZ, tmp_dir=None, log=QUIET):
        X = self.X
        edges = _plan_blocks(self.gene_nnz(), max_nnz)
        if sp.issparse(X) and X.format == "csr" and len(edges) > 1:
            log("Converting the CSR matrix to CSC (one copy)")
            X = X.tocsc()  # one copy, instead of scanning all rows for every block
        for start, stop in edges:
            yield start, stop, _canonical_csc(X[:, start:stop])

    def row_sums(self, log=QUIET) -> np.ndarray:
        return _row_sums(self.X)


class H5Source(MatrixSource):
    """Matrix stored in an h5ad file (X or a layer), read lazily with h5py."""

    def __init__(self, path, key="X"):
        import h5py

        self.path = str(path)
        self.key = key
        self.file = h5py.File(self.path, "r")
        if key not in self.file:
            self.file.close()
            raise KeyError(f"'{key}' not found in {self.path}.")
        element = self.file[key]
        if isinstance(element, h5py.Group):
            encoding = element.attrs.get("encoding-type", element.attrs.get("h5sparse_format", ""))
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            encoding = {"csr": "csr_matrix", "csc": "csc_matrix"}.get(encoding, encoding)
            if encoding not in ("csr_matrix", "csc_matrix"):
                raise ValueError(f"Unsupported encoding '{encoding}' for '{key}' in {self.path}.")
            self.format = encoding[:3]
            shape = element.attrs.get("shape", element.attrs.get("h5sparse_shape"))
            self.shape = tuple(int(s) for s in shape)
            self.data = element["data"]
            self.indices = element["indices"]
            self.indptr = element["indptr"][...].astype(np.int64)
        else:
            self.format = "dense"
            self.shape = tuple(int(s) for s in element.shape)
            self.dense = element
        self.column_major = self.format == "csc"

    def describe(self) -> str:
        dataset = self.dense if self.format == "dense" else self.data
        layout = "dense" if self.format == "dense" else self.format.upper()
        stored = "" if self.format == "dense" else f", {dataset.shape[0]:,} stored values"
        chunks = "contiguous" if dataset.chunks is None else "chunks of " + " x ".join(f"{c:,}" for c in dataset.chunks)
        return (f"{layout} matrix '{self.key}' in {self.path} ({dataset.dtype}{stored}; "
                f"{dataset.compression or 'no'} compression, {chunks})")

    def gene_nnz(self, log=QUIET) -> np.ndarray:
        if self.format == "csc":
            return np.diff(self.indptr)
        stage = log.stage("Counting stored values per gene (reading all cells)", self.n_obs, "cells")
        result = np.zeros(self.n_vars, dtype=np.int64)
        for _, stop, chunk in self.row_chunks():
            if sp.issparse(chunk):
                result += np.bincount(chunk.indices, minlength=self.n_vars)
            else:
                result += np.count_nonzero(chunk, axis=0)
            stage.update(stop)
        stage.done()
        return result

    def row_chunks(self, max_nnz: int = DEFAULT_MAX_NNZ):
        if self.format == "csr":
            for start, stop in _plan_blocks(np.diff(self.indptr), max_nnz):
                lo, hi = self.indptr[start], self.indptr[stop]
                chunk = sp.csr_matrix(
                    (self.data[lo:hi], self.indices[lo:hi], self.indptr[start:stop + 1] - lo),
                    shape=(stop - start, self.n_vars),
                )
                yield start, stop, chunk
        elif self.format == "dense":
            rows_per_chunk = max(1, int(max_nnz) // max(self.n_vars, 1))
            for start in range(0, self.n_obs, rows_per_chunk):
                stop = min(start + rows_per_chunk, self.n_obs)
                yield start, stop, self.dense[start:stop]
        else:
            # gene-major storage: assemble cell chunks from gene blocks (rarely needed)
            full = sp.hstack([b for _, _, b in self.column_blocks(max_nnz)], format="csr")
            for start, stop in _plan_blocks(np.diff(full.indptr), max_nnz):
                yield start, stop, full[start:stop]

    def column_blocks(self, max_nnz: int = DEFAULT_MAX_NNZ, tmp_dir=None, log=QUIET):
        if self.format != "csc":
            yield from super().column_blocks(max_nnz, tmp_dir, log)
            return
        for start, stop in _plan_blocks(self.gene_nnz(), max_nnz):
            lo, hi = self.indptr[start], self.indptr[stop]
            block = sp.csc_matrix(
                (self.data[lo:hi], self.indices[lo:hi], self.indptr[start:stop + 1] - lo),
                shape=(self.n_obs, stop - start),
            )
            yield start, stop, _canonical_csc(block)

    def row_sums(self, log=QUIET) -> np.ndarray:
        if self.format == "csc":
            stage = log.stage("Computing library sizes (reading all genes)", self.n_vars, "genes")
            result = np.zeros(self.n_obs)
            for _, stop, block in self.column_blocks():
                result += np.bincount(block.indices, weights=block.data, minlength=self.n_obs)
                stage.update(stop)
            stage.done()
            return result
        return super().row_sums(log)

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def _transpose_via_disk(source: MatrixSource, edges, max_nnz, tmp_dir,
                        log=QUIET) -> Iterator[tuple[int, int, sp.csc_matrix]]:
    """Split a row-major source into gene blocks through temporary files.

    One pass converts each chunk of cells to CSC (so every gene block is a
    contiguous slice) and appends it to one file per block: per-gene counts, cell
    indices and values. Each block is then read back and assembled as a CSC
    matrix. Library sizes are computed on the way (see MatrixSource.row_sums).
    """
    directory = tempfile.mkdtemp(prefix="metamarkers_", dir=tmp_dir)
    stage = log.stage(f"Splitting the matrix into {len(edges):,} blocks of genes (temporary files in {directory})",
                      source.n_obs, "cells")
    row_type = np.int32 if source.n_obs < 2**31 else np.int64
    row_sums = np.zeros(source.n_obs)
    value_type = None
    try:
        paths = [os.path.join(directory, f"block_{i}.bin") for i in range(len(edges))]
        for row_start, row_stop, chunk in source.row_chunks(max(int(max_nnz), 5_000_000)):
            chunk = sp.csc_matrix(chunk)
            value_type = value_type or chunk.dtype
            row_sums[row_start:row_stop] = np.bincount(chunk.indices, weights=chunk.data, minlength=row_stop - row_start)
            rows = (chunk.indices + row_start).astype(row_type)
            values = chunk.data.astype(value_type, copy=False)
            for path, (start, stop) in zip(paths, edges):
                lo, hi = chunk.indptr[start], chunk.indptr[stop]
                with open(path, "ab") as handle:
                    np.diff(chunk.indptr[start:stop + 1]).astype(np.int64).tofile(handle)
                    rows[lo:hi].tofile(handle)
                    values[lo:hi].tofile(handle)
            stage.update(row_stop)
        stage.done()
        source._cached_row_sums = row_sums
        for path, (start, stop) in zip(paths, edges):
            yield start, stop, _read_block(path, stop - start, source.n_obs, row_type, value_type)
            os.remove(path)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _read_block(path, width, n_obs, row_type, value_type) -> sp.csc_matrix:
    """Assemble the pieces of a gene block (one per chunk of cells) as a CSC matrix."""
    raw = np.fromfile(path, dtype=np.uint8)
    counts, rows, values = [], [], []
    position = 0
    while position < len(raw):
        piece_counts = raw[position:position + 8 * width].view(np.int64)
        position += 8 * width
        n = int(piece_counts.sum())
        size = np.dtype(row_type).itemsize * n
        rows.append(raw[position:position + size].view(row_type))
        position += size
        size = np.dtype(value_type).itemsize * n
        values.append(raw[position:position + size].view(value_type))
        position += size
        counts.append(piece_counts)
    columns = np.concatenate([np.repeat(np.arange(width), c) for c in counts]) if counts else np.zeros(0, int)
    data = np.concatenate(values).astype(np.float64) if values else np.zeros(0)
    cells = np.concatenate(rows) if rows else np.zeros(0, row_type)
    block = sp.coo_matrix((data, (cells, columns)), shape=(n_obs, width)).tocsc()
    block.eliminate_zeros()
    return block


# ---- resolving user inputs ---------------------------------------------------------

class Expression:
    """Resolved expression input: matrix source plus optional AnnData metadata."""

    def __init__(self, source: MatrixSource, obs=None, var=None, obs_names=None, var_names=None, owns_source=False):
        self.source = source
        self.obs = obs
        self.var = var
        self.obs_names = obs_names
        self.var_names = var_names
        self.owns_source = owns_source

    @property
    def shape(self):
        return self.source.shape

    def close(self):
        if self.owns_source:
            self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def gene_names(self, gene_names=None) -> np.ndarray:
        n_genes = self.shape[1]
        if gene_names is None:
            names = self.var_names if self.var_names is not None else np.arange(n_genes).astype(str)
        elif isinstance(gene_names, str):
            if self.var is None or gene_names not in self.var.columns:
                raise KeyError(f"gene_names='{gene_names}' is not a column of var.")
            names = self.var[gene_names]
        else:
            names = gene_names
        names = np.asarray(pd.Series(names).astype(str), dtype=object)
        if len(names) != n_genes:
            raise ValueError(f"Expected {n_genes} gene names, got {len(names)}.")
        return names

    def labels(self, labels, argument="labels") -> np.ndarray:
        """Labels as an object array of str, with None for missing values."""
        n_cells = self.shape[0]
        if isinstance(labels, str):
            if self.obs is None or labels not in self.obs.columns:
                raise KeyError(f"{argument}='{labels}' is not a column of obs.")
            labels = self.obs[labels]
        values = pd.Series(np.asarray(labels, dtype=object))
        if len(values) != n_cells:
            raise ValueError(f"{argument} has {len(values)} values for {n_cells} cells.")
        missing = values.isna().to_numpy()
        result = values.astype(str).to_numpy(dtype=object)
        result[missing] = None
        return result


def _is_anndata(data) -> bool:
    try:
        import anndata
    except ImportError:  # pragma: no cover
        return False
    return isinstance(data, anndata.AnnData)


def resolve_expression(data, layer=None, genes_are_rows=False) -> Expression:
    """Turn an AnnData, h5ad path or matrix into an :class:`Expression`."""
    if isinstance(data, (str, os.PathLike)):
        import anndata

        adata = anndata.read_h5ad(data, backed="r")
        obs, var = adata.obs.copy(), adata.var.copy()
        obs_names, var_names = np.asarray(adata.obs_names), np.asarray(adata.var_names)
        adata.file.close()
        source = H5Source(data, "X" if layer is None else f"layers/{layer}")
        return Expression(source, obs, var, obs_names, var_names, owns_source=True)
    if _is_anndata(data):
        if genes_are_rows:
            raise ValueError("genes_are_rows=True is not supported for AnnData objects (cells are rows).")
        obs_names, var_names = np.asarray(data.obs_names), np.asarray(data.var_names)
        if data.isbacked and not data.is_view:
            source = H5Source(data.filename, "X" if layer is None else f"layers/{layer}")
            return Expression(source, data.obs, data.var, obs_names, var_names, owns_source=True)
        X = data.X if layer is None else data.layers[layer]
        if X is None:
            raise ValueError("AnnData object has no expression matrix (X is None).")
        if not sp.issparse(X) and not isinstance(X, np.ndarray):
            X = X[...] if hasattr(X, "__getitem__") else np.asarray(X)
            if hasattr(X, "to_memory"):
                X = X.to_memory()
        return Expression(InMemorySource(X), data.obs, data.var, obs_names, var_names)
    if isinstance(data, MatrixSource):
        return Expression(data)
    X = data
    if isinstance(X, pd.DataFrame):
        obs_names, var_names = np.asarray(X.index), np.asarray(X.columns)
        X = X.to_numpy()
        if genes_are_rows:
            obs_names, var_names = var_names, obs_names
            X = X.T
        return Expression(InMemorySource(X), obs_names=obs_names.astype(str), var_names=var_names.astype(str))
    if genes_are_rows:
        X = X.T
    return Expression(InMemorySource(X))
