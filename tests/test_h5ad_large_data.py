"""h5ad files and out-of-memory processing give the same results as R / in-memory computation."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import metamarkers as mm
from conftest import (H5AD_DIR, assert_markers_equal, assert_matrix_close, read_ref, read_ref_matrix,
                      read_ref_vector, requires_data, vignette_labels)

pytestmark = requires_data

SMALL_BLOCKS = 400_000  # stored values per block: forces many blocks (and disk transposition for CSR)


@pytest.fixture(scope="module")
def muraro_files(tmp_path_factory):
    """Full Muraro dataset (all cells) as h5ad in several layouts, with vignette labels in obs."""
    directory = tmp_path_factory.mktemp("h5ad")
    adata = ad.read_h5ad(H5AD_DIR / "muraro.h5ad")
    labels = vignette_labels("muraro", adata.obs)
    common = read_ref_vector("v1_common_labels")
    adata.obs["new_label"] = labels.where(labels.isin(common)).astype("category")
    adata.obs["class"] = np.where(adata.obs["new_label"].isin(["alpha", "beta", "gamma", "delta", "epsilon"]),
                                  "endocrine", "non-endocrine")
    adata.obs.loc[adata.obs["new_label"].isna(), "class"] = None
    files = {}
    for layout in ("csr", "csc"):
        copy = adata.copy()
        copy.X = sp.csr_matrix(copy.X) if layout == "csr" else sp.csc_matrix(copy.X)
        files[layout] = directory / f"muraro_{layout}.h5ad"
        copy.write_h5ad(files[layout])
    layered = adata.copy()
    layered.layers["counts"] = sp.csr_matrix(layered.X)
    layered.X = sp.csr_matrix(layered.X.shape)
    files["layer"] = directory / "muraro_layer.h5ad"
    layered.write_h5ad(files["layer"])
    dense = adata[:, :1500].copy()
    dense.X = dense.X.toarray()
    files["dense"] = directory / "muraro_dense.h5ad"
    dense.write_h5ad(files["dense"])
    return files


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_markers_from_h5ad_path(muraro_files, layout):
    """Counts on disk + normalize='cpm' + small blocks reproduce R's vignette markers exactly."""
    markers = mm.compute_markers(muraro_files[layout], "new_label", gene_names="symbol", normalize="cpm",
                                 max_nnz_per_block=SMALL_BLOCKS)
    assert_markers_equal(markers, read_ref("v1_markers_muraro"))


def test_grouped_markers_from_layer(muraro_files):
    markers = mm.compute_markers(muraro_files["layer"], "new_label", "class", layer="counts", gene_names="symbol",
                                 normalize="cpm", max_nnz_per_block=SMALL_BLOCKS)
    assert_markers_equal(markers, read_ref("v1_within_markers_muraro"))


def test_backed_anndata(muraro_files):
    adata = ad.read_h5ad(muraro_files["csr"], backed="r")
    try:
        markers = mm.compute_markers(adata, "new_label", gene_names="symbol", normalize="cpm",
                                     max_nnz_per_block=2_000_000)
    finally:
        adata.file.close()
    assert_markers_equal(markers, read_ref("v1_markers_muraro"))


def test_dense_h5ad(muraro_files, v1_datasets):
    adata = ad.read_h5ad(muraro_files["dense"])
    keep = adata.obs["new_label"].notna().to_numpy()
    expected = mm.compute_markers(mm.convert_to_cpm(sp.csr_matrix(adata.X[keep])),
                                  adata.obs["new_label"].to_numpy()[keep], gene_names=adata.var["symbol"])
    markers = mm.compute_markers(muraro_files["dense"], "new_label", gene_names="symbol", normalize="cpm",
                                 max_nnz_per_block=SMALL_BLOCKS)
    pd.testing.assert_frame_equal(markers, expected)


def test_original_h5ad_file():
    """The converted vignette file can be used directly (cells outside the common types left unlabeled)."""
    adata = ad.read_h5ad(H5AD_DIR / "baron.h5ad", backed="r")
    labels = vignette_labels("baron", adata.obs)
    labels = labels.where(labels.isin(read_ref_vector("v1_common_labels")))
    adata.file.close()
    markers = mm.compute_markers(H5AD_DIR / "baron.h5ad", labels.to_numpy(), gene_names="symbol", normalize="cpm",
                                 max_nnz_per_block=4_000_000)
    assert_markers_equal(markers, read_ref("v1_markers_baron"))


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_score_cells_from_h5ad(muraro_files, v2_datasets, v2_meta_markers, layout):
    top_markers = v2_meta_markers["muraro"].query("rank <= 100")
    scores = mm.score_cells(muraro_files[layout], top_markers, gene_names="symbol", normalize="log1p_cpm",
                            max_nnz_per_block=SMALL_BLOCKS)
    adata = ad.read_h5ad(muraro_files[layout], backed="r")
    keep = adata.obs["new_label"].notna().to_numpy()
    adata.file.close()
    expected = read_ref_matrix("v2_scores_muraro").T
    assert_matrix_close(scores.to_numpy()[keep], expected)
    assert list(scores.columns) == list(expected.columns)
    assert list(scores.index[keep]) == list(expected.index)


def test_out_of_core_matches_in_memory(tmp_path):
    """Synthetic dataset processed in many blocks through disk transposition."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 6000, 800
    counts = sp.random(n_cells, n_genes, density=0.08, format="csr", random_state=rng,
                       data_rvs=lambda n: rng.poisson(3, n) + 1).astype(np.float32)
    labels = rng.choice(["a", "b", "c", "d"], n_cells)
    groups = np.where(np.isin(labels, ["a", "b"]), "g1", "g2")
    adata = ad.AnnData(counts, obs=pd.DataFrame({"label": labels, "group": groups}, index=[f"c{i}" for i in range(n_cells)]))
    path = tmp_path / "synthetic.h5ad"
    adata.write_h5ad(path)
    expected = mm.compute_markers(mm.convert_to_cpm(counts.astype(np.float64)), labels, groups,
                                  gene_names=adata.var_names, tie_correction=True)
    for n_jobs in (1, 4):
        markers = mm.compute_markers(path, "label", "group", normalize="cpm", tie_correction=True,
                                     max_nnz_per_block=20_000, n_jobs=n_jobs, tmp_dir=tmp_path)
        pd.testing.assert_frame_equal(markers, expected)
    assert not list(tmp_path.glob("metamarkers_*"))  # temporary files removed
    marker_table = mm.make_meta_markers({"x": markers, "y": expected}).query("rank <= 20")
    logcpm = mm.convert_to_cpm(counts.astype(np.float64))
    logcpm.data = np.log1p(logcpm.data)
    expected_scores = mm.score_cells(logcpm, marker_table, gene_names=adata.var_names)
    scores = mm.score_cells(path, marker_table, normalize="log1p_cpm", max_nnz_per_block=20_000)
    assert_matrix_close(scores, expected_scores)
