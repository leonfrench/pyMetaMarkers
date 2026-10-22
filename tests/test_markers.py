"""compute_markers against the R package (vignette 1 + option variants)."""

import warnings

import numpy as np
import pandas as pd
import pytest

import metamarkers as mm
from conftest import assert_markers_equal, read_ref, requires_data

pytestmark = requires_data


@pytest.mark.parametrize("name", ["baron", "muraro"])
def test_vignette_markers(v1_markers, name):
    assert_markers_equal(v1_markers[name], read_ref(f"v1_markers_{name}"))


def test_vignette_beta_markers(v1_markers):
    beta = v1_markers["baron"].query("cell_type == 'beta'").head(6)
    assert list(beta["gene"]) == ["INS", "IAPP", "HADH", "ADCYAP1", "PCSK1", "UCHL1"]
    np.testing.assert_allclose(beta["fold_change"].iloc[0], 63.2, atol=0.05)
    np.testing.assert_allclose(beta["auroc"].iloc[0], 0.993, atol=5e-4)


@pytest.mark.parametrize("name", ["baron", "muraro"])
def test_class_markers(v1_datasets, name):
    d = v1_datasets[name]
    markers = mm.compute_markers(d.cpm, d.cell_class, gene_names=d.genes)
    assert_markers_equal(markers, read_ref(f"v1_class_markers_{name}"))


@pytest.mark.parametrize("name", ["baron", "muraro"])
def test_within_class_markers(v1_datasets, name):
    d = v1_datasets[name]
    markers = mm.compute_markers(d.cpm, d.label, d.cell_class, gene_names=d.genes)
    assert_markers_equal(markers, read_ref(f"v1_within_markers_{name}"))


def test_segerstolpe_markers(v2_markers):
    # Segerstolpe has 725 duplicated gene names, which are removed
    assert_markers_equal(v2_markers["seger"], read_ref("v2_markers_seger"))


def test_segerstolpe_grouped_markers(v3_meta_markers):
    assert_markers_equal(v3_meta_markers["ct_markers"]["seger"], read_ref("v3_ct_markers_seger"))


@pytest.mark.parametrize("options, reference", [
    ({"two_tailed": True}, "opt_markers_muraro_two_tailed"),
    ({"tie_correction": True}, "opt_markers_muraro_tie_correction"),
])
def test_test_options(v1_datasets, options, reference):
    d = v1_datasets["muraro"]
    markers = mm.compute_markers(d.cpm, d.label, gene_names=d.genes, **options)
    assert_markers_equal(markers, read_ref(reference))


def test_grouped_two_tailed_tie_correction(v1_datasets):
    d = v1_datasets["muraro"]
    markers = mm.compute_markers(d.cpm, d.label, d.cell_class, two_tailed=True, tie_correction=True, gene_names=d.genes)
    assert_markers_equal(markers, read_ref("opt_markers_muraro_grouped_two_tailed_tie_correction"))


def test_dense_input(v1_datasets):
    d = v1_datasets["muraro"]
    dense = d.cpm[:, :300].toarray()
    markers = mm.compute_markers(dense, d.label, gene_names=d.genes[:300])
    assert_markers_equal(markers, read_ref("opt_markers_muraro_dense_cells_as_rows"))


def test_genes_as_rows(v1_datasets):
    d = v1_datasets["muraro"]
    genes_by_cells = d.cpm[:, :300].T.tocsc()
    markers = mm.compute_markers(genes_by_cells, d.label, gene_names=d.genes[:300], genes_are_rows=True)
    expected = mm.compute_markers(d.cpm[:, :300], d.label, gene_names=d.genes[:300])
    pd.testing.assert_frame_equal(markers, expected)


def test_group_with_single_cell_type_is_skipped(v1_datasets):
    d = v1_datasets["muraro"]
    groups = np.where(d.label == "acinar", "solo", "rest")
    with pytest.warns(UserWarning, match="Skipping group solo: only 1 cluster \\(acinar\\)"):
        markers = mm.compute_markers(d.cpm[:, :300], d.label, groups, gene_names=d.genes[:300])
    assert_markers_equal(markers, read_ref("opt_markers_muraro_single_cell_type_group"))


def test_missing_labels_are_excluded(v1_datasets):
    d = v1_datasets["muraro"]
    labels = d.label.copy()
    labels[d.label == "epsilon"] = None
    keep = d.label != "epsilon"
    markers = mm.compute_markers(d.cpm[:, :200], labels, gene_names=d.genes[:200])
    expected = mm.compute_markers(d.cpm[keep][:, :200], d.label[keep], gene_names=d.genes[:200])
    pd.testing.assert_frame_equal(markers, expected)


def test_empty_label_raises(v1_datasets):
    d = v1_datasets["muraro"]
    labels = d.label.copy()
    labels[0] = ""
    with pytest.raises(ValueError, match="All cells must have a label"):
        mm.compute_markers(d.cpm[:, :10], labels)



def test_remove_duplicated_genes():
    original = read_ref("helper_duplicated_input")
    expected = read_ref("helper_duplicated_output")
    result = mm.remove_duplicated_genes(original).reset_index(drop=True)
    assert_markers_equal(result, expected)
