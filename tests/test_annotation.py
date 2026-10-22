"""Cell type annotation (vignettes "Annotation" and "HierarchicalAnnotation") against R.

Score matrices are cells x marker sets in Python and marker sets x cells in R,
so R matrices are transposed before comparison.
"""

import numpy as np
import pandas as pd
import pytest

import metamarkers as mm
from conftest import assert_markers_equal, assert_matrix_close, read_ref, read_ref_matrix, read_ref_vector, requires_data

pytestmark = requires_data

TEST_DATASETS = ["baron", "muraro", "seger"]


def assert_scores_equal(python: pd.DataFrame, r_matrix: pd.DataFrame):
    expected = r_matrix.T
    assert list(python.columns) == list(expected.columns)
    assert list(python.index) == list(expected.index)
    assert_matrix_close(python, expected)


def assert_assignments_equal(python: pd.DataFrame, r: pd.DataFrame):
    assert_markers_equal(python.reset_index(drop=True), r)


@pytest.fixture(scope="module")
def flat_results(v2_datasets, v2_meta_markers):
    results = {}
    for test_dataset in TEST_DATASETS:
        d = v2_datasets[test_dataset]
        top_markers = v2_meta_markers[test_dataset].query("rank <= 100")
        scores = mm.score_cells(d.logcpm, top_markers, gene_names=d.genes)
        scores.index = d.cells
        results[test_dataset] = scores
    return results


@pytest.mark.parametrize("test_dataset", TEST_DATASETS)
def test_scores(flat_results, test_dataset):
    assert_scores_equal(flat_results[test_dataset], read_ref_matrix(f"v2_scores_{test_dataset}"))


@pytest.mark.parametrize("test_dataset", TEST_DATASETS)
def test_enrichment_and_assignment(flat_results, test_dataset):
    scores = flat_results[test_dataset]
    with pytest.warns(UserWarning, match="No group information"):
        pass_through = mm.compute_marker_enrichment(scores.rename(columns=mm.get_cell_type))
    assert pass_through.shape == scores.shape
    enrichment = mm.compute_marker_enrichment(scores)
    assert_scores_equal(enrichment, read_ref_matrix(f"v2_enrichment_{test_dataset}"))
    assignments = mm.assign_cells(scores)
    assert list(assignments.columns) == ["group", "predicted", "score", "enrichment"]
    assert_assignments_equal(assignments, read_ref(f"v2_pred_{test_dataset}"))


@pytest.mark.parametrize("test_dataset", TEST_DATASETS)
def test_performance_summaries(flat_results, v2_datasets, test_dataset):
    enrichment = mm.compute_marker_enrichment(flat_results[test_dataset])
    labels = v2_datasets[test_dataset].label
    auroc = mm.summarize_auroc(enrichment, labels)
    expected = read_ref_matrix(f"v2_auroc_{test_dataset}")
    assert list(auroc.index) == list(expected.index) and list(auroc.columns) == list(expected.columns)
    assert_matrix_close(auroc, expected)
    fold_change = mm.summarize_fold_change(enrichment, labels)
    expected = read_ref_matrix(f"v2_fold_change_{test_dataset}")
    assert list(fold_change.index) == list(expected.index)
    assert_matrix_close(fold_change, expected)
    thresholds = read_ref_vector("v2_thresholds", numeric=True)
    np.testing.assert_array_equal(thresholds, 1 + np.arange(41) * 0.1)  # R's seq(1, 5, 0.1)
    pr = mm.summarize_precision_recall(enrichment, labels, thresholds)
    assert_markers_equal(pr, read_ref(f"v2_precision_recall_{test_dataset}"))


def test_vignette_accuracy(flat_results, v2_datasets):
    """Annotation vignette: top 100 Baron+Muraro meta-markers give 95.6% accuracy on Segerstolpe."""
    assignments = mm.assign_cells(flat_results["seger"])
    accuracy = np.mean(assignments["predicted"].to_numpy() == v2_datasets["seger"].label)
    assert round(accuracy, 3) == 0.956
    epsilon = assignments[(assignments["predicted"] == "epsilon")]
    assert len(epsilon) == 11 and (v2_datasets["seger"].label == "epsilon").sum() == 7


@pytest.mark.parametrize("n", [1, 5, 1000])
def test_scores_with_marker_set_sizes(v2_datasets, v2_meta_markers, n):
    d = v2_datasets["seger"]
    scores = mm.score_cells(d.logcpm, v2_meta_markers["seger"].query("rank <= @n"), gene_names=d.genes)
    scores.index = d.cells
    assert_scores_equal(scores, read_ref_matrix(f"v2_scores_seger_top{n}"))


def test_scores_from_list_and_weight_matrix(v2_datasets, v2_meta_markers):
    d = v2_datasets["seger"]
    top_markers = v2_meta_markers["seger"].query("rank <= 20")
    marker_list = {ct: list(group["gene"]) for ct, group in top_markers.groupby("cell_type", sort=True)}
    scores = mm.score_cells(d.logcpm, marker_list, gene_names=d.genes)
    scores.index = d.cells
    assert_scores_equal(scores, read_ref_matrix("v2_scores_seger_from_list"))
    weights = mm.marker_table_to_matrix(top_markers, d.genes, weighted=False)
    scores = mm.score_cells(d.logcpm, weights, gene_names=d.genes)
    scores.index = d.cells
    assert_scores_equal(scores, read_ref_matrix("v2_scores_seger_unweighted"))


def test_marker_list_to_matrix():
    matrix = mm.marker_list_to_matrix({"beta": ["INS", "IAPP", "NOT_A_GENE"], "alpha": ["GCG", "TTR"]},
                                      ["GCG", "INS", "IAPP", "TTR", "OTHER"])
    expected = read_ref_matrix("helper_marker_list_to_matrix")
    assert list(matrix.columns) == list(expected.columns) == ["all|alpha", "all|beta"]
    np.testing.assert_array_equal(matrix.sparse.to_dense().to_numpy(), expected.to_numpy())


# ---- hierarchical annotation (vignette 3) ---------------------------------------

@pytest.fixture(scope="module")
def hierarchical(v2_datasets, v3_meta_markers):
    d = v2_datasets["seger"]
    class_scores = mm.score_cells(d.logcpm, v3_meta_markers["class"]["seger"].query("rank <= 100"), gene_names=d.genes)
    ct_scores = mm.score_cells(d.logcpm, v3_meta_markers["cell_type"]["seger"].query("rank <= 100"), gene_names=d.genes)
    class_scores.index = ct_scores.index = d.cells
    return class_scores, ct_scores


def test_class_level(hierarchical):
    class_scores, _ = hierarchical
    assert_scores_equal(class_scores, read_ref_matrix("v3_class_scores_seger"))
    assert_scores_equal(mm.compute_marker_enrichment(class_scores), read_ref_matrix("v3_class_enrichment_seger"))
    assert_assignments_equal(mm.assign_cells(class_scores), read_ref("v3_class_pred_seger"))


def test_cell_type_level(hierarchical, v2_datasets):
    class_scores, ct_scores = hierarchical
    assert_scores_equal(ct_scores, read_ref_matrix("v3_ct_scores_seger"))
    assert list(ct_scores.columns[:2]) == ["endocrine|alpha", "endocrine|beta"]
    enrichment = mm.compute_marker_enrichment(ct_scores, by_group=True)
    assert_scores_equal(enrichment, read_ref_matrix("v3_ct_enrichment_seger"))
    flat = mm.compute_marker_enrichment(ct_scores, by_group=False)
    assert_scores_equal(flat, read_ref_matrix("v3_ct_enrichment_flat_seger"))
    class_pred = mm.assign_cells(class_scores)
    ct_pred = mm.assign_cells(ct_scores, group_assignment=class_pred["predicted"])
    assert_assignments_equal(ct_pred, read_ref("v3_ct_pred_seger"))
    # vignette: hierarchical annotation reaches 98.3% accuracy (vs 95.6% flat)
    accuracy = np.mean(ct_pred["predicted"].to_numpy() == v2_datasets["seger"].label)
    assert round(accuracy, 3) == 0.983
    assert (ct_pred["predicted"] == "epsilon").sum() == 31


def test_assignment_warnings(hierarchical):
    _, ct_scores = hierarchical
    with pytest.warns(UserWarning, match="Detected group information"):
        no_groups = mm.assign_cells(ct_scores)
    assert_assignments_equal(no_groups, read_ref("v3_ct_pred_no_groups_seger"))
    odd = read_ref_vector("v3_odd_assignment_seger")
    with pytest.warns(UserWarning, match="unknown_group"):
        result = mm.assign_cells(ct_scores, group_assignment=odd)
    expected = read_ref("v3_ct_pred_unknown_group_seger")
    assert_assignments_equal(result, expected)
    assert result["predicted"].isna().sum() == len(range(0, len(odd), 50))
    with pytest.raises(ValueError, match="no group information"):
        mm.assign_cells(ct_scores.rename(columns=mm.get_cell_type), group_assignment=odd)


@pytest.mark.parametrize("test_dataset", TEST_DATASETS)
def test_endocrine_precision_recall(v2_datasets, v3_meta_markers, test_dataset):
    d = v2_datasets[test_dataset]
    scores = mm.score_cells(d.logcpm, v3_meta_markers["cell_type"][test_dataset].query("rank <= 100"), gene_names=d.genes)
    enrichment = mm.compute_marker_enrichment(scores, by_group=True)
    keep_cell = d.cell_class == "endocrine"
    keep_set = mm.get_group(enrichment.columns) == "endocrine"
    thresholds = 1 + np.arange(21) * 0.1  # R's seq(1, 3, 0.1)
    pr = mm.summarize_precision_recall(enrichment.loc[keep_cell, keep_set], d.label[keep_cell], thresholds)
    assert_markers_equal(pr, read_ref(f"v3_precision_recall_endocrine_{test_dataset}"))


def test_group_and_cell_type_helpers():
    expected = read_ref("helper_group_cell_type")
    assert list(mm.get_group(expected["label"])) == list(expected["group"])
    assert list(mm.get_cell_type(expected["label"])) == list(expected["cell_type"])
    mixed = read_ref("helper_group_cell_type_mixed")
    assert list(mm.get_group(mixed["label"])) == ["", ""]
    assert list(mm.get_cell_type(mixed["label"])) == list(mixed["cell_type"])
    assert mm.get_group("all|beta") == "all" and mm.get_cell_type("all|beta") == "beta"


def test_design_matrix():
    for scaled, name in [(False, "helper_design_matrix"), (True, "helper_design_matrix_scaled")]:
        expected = read_ref_matrix(name)
        result = mm.design_matrix(["b", "a", "c", "a"], scale_columns=scaled)
        assert list(result.columns) == list(expected.columns) == ["a", "b", "c"]
        np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())
