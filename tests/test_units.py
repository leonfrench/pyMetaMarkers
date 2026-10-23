"""Unit tests that need no external data (independent checks of the statistics)."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats

import metamarkers as mm


@pytest.fixture
def toy():
    rng = np.random.default_rng(1)
    n_cells, n_genes = 300, 40
    X = rng.poisson(0.7, size=(n_cells, n_genes)).astype(float)
    X[:, 0] = 0  # gene without expression
    X[:, 1] = 5  # constant gene
    labels = rng.choice(["t1", "t2", "t3"], n_cells)
    X[labels == "t1", 2] += 4  # marker of t1
    return X, labels


def test_auroc_matches_mann_whitney(toy):
    X, labels = toy
    markers = mm.compute_markers(sp.csr_matrix(X), labels)
    for cell_type in ("t1", "t2"):
        for gene in (2, 5, 11):
            positive = labels == cell_type
            u = stats.mannwhitneyu(X[positive, gene], X[~positive, gene]).statistic
            expected = u / (positive.sum() * (~positive).sum())
            gene_name = str(gene)
            row = markers.query("cell_type == @cell_type and gene == @gene_name")
            assert row["auroc"].item() == pytest.approx(expected, abs=1e-14)
    assert (markers.query("gene in ['0', '1']")["auroc"] == 0.5).all()


def test_p_values_match_normal_approximation(toy):
    X, labels = toy
    markers = mm.compute_markers(X, labels, two_tailed=True, tie_correction=True)
    positive = labels == "t1"
    result = stats.mannwhitneyu(X[positive, 2], X[~positive, 2], use_continuity=False, method="asymptotic")
    n = positive.sum() * (~positive).sum()
    auroc = markers.query("cell_type == 't1' and gene == '2'")["auroc"].item()
    assert auroc == pytest.approx(result.statistic / n, abs=1e-14)
    # FDR >= p-value, and the BH adjustment of this single test is bounded by n_tests * p
    log_fdr = markers.query("cell_type == 't1' and gene == '2'")["log_fdr"].item()
    assert np.log(result.pvalue) <= log_fdr + 1e-9 <= np.log(result.pvalue * len(markers)) + 1e-9


def test_sparse_and_dense_inputs_agree(toy):
    X, labels = toy
    dense = mm.compute_markers(X, labels)
    for matrix in (sp.csr_matrix(X), sp.csc_matrix(X), sp.coo_matrix(X)):
        pd.testing.assert_frame_equal(mm.compute_markers(matrix, labels), dense)
    blocks = mm.compute_markers(sp.csc_matrix(X), labels, max_nnz_per_block=50)
    pd.testing.assert_frame_equal(blocks, dense)


def test_negative_values_are_ranked_correctly(toy):
    X, labels = toy
    shifted = X - 1.5  # dense ranks include zeros between negative and positive values
    shifted[shifted == -1.5] = 0
    expected, _ = mm.compute_aurocs(shifted, labels)
    markers = mm.compute_markers(sp.csr_matrix(shifted), labels)
    for k, cell_type in enumerate(["t1", "t2", "t3"]):
        values = markers.query("cell_type == @cell_type").set_index("gene").loc[[str(g) for g in range(X.shape[1])]]
        np.testing.assert_allclose(values["auroc"].to_numpy(), expected[k], atol=1e-15)


def test_fdr_matches_benjamini_hochberg():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.5, 0.04, 1e-5])
    n = len(p)
    order = np.argsort(p)
    expected = np.empty(n)
    expected[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    np.testing.assert_allclose(mm.fdr(p), np.minimum(expected, 1))
    np.testing.assert_allclose(np.exp(mm.fdr(np.log(p), log_p=True)), np.minimum(expected, 1))



def test_meta_markers_ranking():
    def table(dataset_shift):
        return pd.DataFrame({
            "group": "all", "cell_type": ["x"] * 3 + ["y"] * 3, "gene": ["g1", "g2", "g3"] * 2,
            "fold_change": [10, 5, 1, 1, 8, 1], "auroc": [0.9, 0.95, 0.5, 0.5, 0.8, 0.6 + dataset_shift],
            "log_fdr": [-10, -10, 0, 0, -10, 0], "population_size": 10, "population_fraction": 0.5,
            "average_expression": [5, 2, 0, 0, 3, 1], "se_expression": 0.1, "detection_rate": 0.5,
            "fold_change_detection": [3, 2, 1, 1, 4, 1], "precision": 0.5, "recall": 0.5,
        })
    meta = mm.make_meta_markers({"a": table(0), "b": table(0.1)}, detailed_stats=True)
    x = meta.query("cell_type == 'x'")
    assert list(x["gene"]) == ["g2", "g1", "g3"]  # recurrence 2 for g1/g2, ordered by AUROC
    assert list(x["recurrence"]) == [2, 2, 0] and list(x["rank"]) == [1, 2, 3]
    assert meta.query("gene == 'g3' and cell_type == 'y'")["auroc"].item() == pytest.approx(0.65)
    assert meta.query("gene == 'g1' and cell_type == 'x'")["fold_change"].item() == pytest.approx(10)


def test_assign_cells_ties():
    scores = pd.DataFrame([[1.0, 0.0, 0.0], [0.0, 2.0, 2.0], [0.0, 0.0, 0.0], [0.5, 3.0, 1.0], [0.0, 0.0, 1.0]],
                          columns=["all|a", "all|b", "all|c"])
    result = mm.assign_cells(scores)
    # only equal top scores are ties (R also reported the first row as a tie, because the
    # maximum was in the first column and all other scores were 0)
    assert list(result["predicted"]) == ["a", "unassigned", "unassigned", "b", "c"]
    assert result["enrichment"].iloc[3] == pytest.approx((3.0001) / np.mean(np.array([0.5, 3, 1]) + 0.0001))
    negative = mm.assign_cells(pd.DataFrame([[-1.0, -2.0], [-3.0, -3.0]], columns=["all|a", "all|b"]))
    assert list(negative["predicted"]) == ["a", "unassigned"]
    single = mm.assign_cells(pd.DataFrame([[0.0], [2.0]], columns=["all|a"]))
    assert list(single["predicted"]) == ["a", "a"]


def test_enrichment_of_cells_without_marker_expression():
    """Identical scores give an enrichment of exactly 1, whatever the number of marker sets."""
    for n_sets in (3, 9, 85, 300):
        scores = pd.DataFrame(np.zeros((2, n_sets)), columns=[f"all|t{i}" for i in range(n_sets)])
        scores.iloc[1, 0] = 2.0
        enrichment = mm.compute_marker_enrichment(scores)
        assert (enrichment.iloc[0] == 1.0).all()
        assert enrichment.iloc[1, 0] > 1
        assignments = mm.assign_cells(scores)
        assert assignments["enrichment"].iloc[0] == 1.0


def test_ranking_methods_agree(toy):
    X, labels = toy
    shifted = X - 1.5
    shifted[shifted == -1.5] = 0  # negative, zero and positive values, many ties
    block = sp.csc_matrix(shifted)
    n_zeros = block.shape[0] - np.diff(block.indptr)
    loop = mm.markers._ranks_within_columns(block.data, block.indptr, n_zeros)
    vectorized = mm.markers._ranks_vectorized(block.data, block.indptr, n_zeros)
    for a, b in zip(loop, vectorized):
        np.testing.assert_array_equal(a, b)


def test_verbose_prints_stages(tmp_path, capsys, monkeypatch):
    """verbose=True prints each stage (and progress) without changing the results."""
    import anndata as ad

    rng = np.random.default_rng(2)
    counts = sp.random(2000, 300, density=0.1, format="csr", random_state=rng,
                       data_rvs=lambda n: rng.poisson(3, n) + 1).astype(np.float32)
    labels = rng.choice(["a", "b", "c"], 2000)
    for layout in ("csr", "csc"):
        ad.AnnData(counts.asformat(layout)).write_h5ad(tmp_path / f"{layout}.h5ad")
    cpm = mm.convert_to_cpm(counts.astype(np.float64))
    options = dict(max_nnz_per_block=5_000, n_jobs=2)
    expected = mm.compute_markers(cpm, labels, **options)
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(mm._progress, "PROGRESS_INTERVAL", 0)  # progress after every block
    cases = [
        (tmp_path / "csr.h5ad", "cpm", ["CSR matrix 'X'", "Counting stored values per gene", "Splitting the matrix"]),
        (tmp_path / "csc.h5ad", "cpm", ["CSC matrix 'X'", "Computing library sizes (reading all genes)"]),
        (cpm, None, ["CSR matrix in memory", "Converting the CSR matrix to CSC"]),
    ]
    for data, normalize, stages in cases:
        markers = mm.compute_markers(data, labels, normalize=normalize, verbose=True, **options)
        pd.testing.assert_frame_equal(markers, expected)
        out = capsys.readouterr().out
        lines = ["compute_markers: 2,000 cells x 300 genes", stages[0], "Group all: 2,000 cells, 3 cell types",
                 *stages[1:], "Computing statistics for 300 genes", "of 300 genes (", "reading blocks",
                 "Assembling the marker table", "Done: 900 rows"]
        positions = [out.find(line) for line in lines]
        assert -1 not in positions and positions == sorted(positions), out


@pytest.mark.parametrize("options", [{}, {"tie_correction": True, "two_tailed": True}])
def test_threads_match_serial(toy, options):
    X, labels = toy
    groups = np.where(labels == "t1", "g1", "g2")
    labels = np.where(labels == "t1", np.where(np.arange(len(labels)) % 2, "t1a", "t1b"), labels)
    serial = mm.compute_markers(sp.csr_matrix(X), labels, groups, n_jobs=1, max_nnz_per_block=500, **options)
    threaded = mm.compute_markers(sp.csr_matrix(X), labels, groups, n_jobs=4, max_nnz_per_block=500, **options)
    pd.testing.assert_frame_equal(threaded, serial, check_exact=True)
