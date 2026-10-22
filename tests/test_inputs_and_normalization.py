"""The h5ad conversion of the vignette datasets, and CPM normalization."""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import metamarkers as mm
from conftest import FILES, H5AD_DIR, assert_matrix_close, read_ref, read_ref_matrix, requires_data

pytestmark = requires_data


@pytest.mark.parametrize("name", ["baron", "muraro", "seger"])
def test_h5ad_contents(name):
    adata = ad.read_h5ad(H5AD_DIR / f"{FILES[name]}.h5ad")
    assert adata.var_names.is_unique
    assert "symbol" in adata.var.columns
    assert sp.issparse(adata.X)
    library_size = read_ref(f"library_size_{name}")["value"].to_numpy()
    assert adata.n_obs == len(library_size)
    expected_columns = {"baron": {"donor", "label"}, "muraro": {"label", "donor", "plate"},
                        "seger": {"cell type", "individual", "disease"}}[name]
    assert expected_columns <= set(adata.obs.columns)
    assert read_ref("h5ad_roundtrip_check")["passed"].all()


def test_vignette_gene_names():
    muraro = ad.read_h5ad(H5AD_DIR / "muraro.h5ad")
    assert muraro.var_names[0] == "A1BG-AS1__chr19" and muraro.var["symbol"].iloc[0] == "A1BG-AS1"
    assert muraro.var["symbol"].duplicated().sum() == 13
    seger = ad.read_h5ad(H5AD_DIR / "segerstolpe.h5ad")
    assert seger.var["symbol"].duplicated().sum() == 725


@pytest.mark.parametrize("name", ["baron", "muraro", "seger"])
def test_library_sizes_match_r(name):
    counts = ad.read_h5ad(H5AD_DIR / f"{FILES[name]}.h5ad").X
    expected = read_ref(f"library_size_{name}")["value"].to_numpy()
    assert_matrix_close(mm.normalization._library_sizes(counts), expected)
    assert_matrix_close(mm.normalization._library_sizes(sp.csr_matrix(counts)), expected)


@pytest.mark.parametrize("name", ["baron", "muraro", "seger"])
def test_cpm_matches_r(name):
    counts = ad.read_h5ad(H5AD_DIR / f"{FILES[name]}.h5ad").X
    sample = read_ref(f"cpm_sample_{name}", strings=set())
    cpm = mm.convert_to_cpm(counts)
    values = np.asarray(cpm[sample["cell"].to_numpy(), sample["gene"].to_numpy()]).ravel()
    assert_matrix_close(values, sample["cpm"])
    # all formats agree
    dense = mm.convert_to_cpm(counts[:200].toarray())
    assert_matrix_close(dense, cpm[:200].toarray())
    csr = mm.convert_to_cpm(sp.csr_matrix(counts))
    assert csr.format == "csr"
    assert_matrix_close(csr.toarray()[:50], cpm[:50].toarray())
    np.testing.assert_allclose(np.asarray(cpm.sum(axis=1)).ravel(), 1e6, rtol=1e-12)


def test_cpm_dense_genes_as_rows():
    example = np.array([1, 0, 3, 2, 2, 0, 0, 5, 1, 1, 1, 1], dtype=float).reshape(3, 4, order="F")
    expected = read_ref_matrix("cpm_dense_example").to_numpy()
    assert_matrix_close(mm.convert_to_cpm(example, scale_factor=100, genes_are_rows=True), expected)


def test_cpm_anndata(v1_datasets):
    d = v1_datasets["muraro"]
    adata = ad.AnnData(d.counts)
    assert_matrix_close(mm.convert_to_cpm(adata).toarray()[:20], d.cpm.toarray()[:20])
