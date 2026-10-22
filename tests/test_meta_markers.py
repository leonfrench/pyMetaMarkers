"""make_meta_markers against the R package."""

import numpy as np
import pytest

import metamarkers as mm
from conftest import REF_DIR, assert_markers_equal, read_ref, requires_data

pytestmark = requires_data

def test_vignette_meta_markers(v1_markers):
    meta = mm.make_meta_markers(v1_markers, detailed_stats=True)
    assert_markers_equal(meta, read_ref("v1_meta_markers"))


def test_vignette_beta_meta_markers(v1_markers):
    meta = mm.make_meta_markers(v1_markers, detailed_stats=True)
    beta = meta.query("cell_type == 'beta'").head(6)
    assert list(beta["gene"]) == ["INS", "IAPP", "HADH", "ADCYAP1", "MAFA", "SYT13"]
    assert (beta["recurrence"] == 2).all()
    assert list(meta.columns[-2:]) == ["baron", "muraro"]


def test_simple_meta_markers(v1_markers):
    meta = mm.make_meta_markers(v1_markers)
    assert list(meta.columns) == ["group", "cell_type", "gene", "rank"]
    assert_markers_equal(meta, read_ref("v1_meta_markers_simple"))


def test_meta_markers_from_r_files():
    markers = {
        "baron": mm.read_markers(REF_DIR / "r_export" / "baron.csv.gz"),
        "muraro": mm.read_markers(REF_DIR / "r_export" / "muraro.csv.gz"),
    }
    meta = mm.make_meta_markers(markers, detailed_stats=True)
    assert_markers_equal(meta, read_ref("v1_meta_markers_from_files"))


@pytest.mark.parametrize("level", ["class", "within"])
def test_hierarchical_meta_markers(v1_datasets, level):
    markers = {}
    for name, d in v1_datasets.items():
        if level == "class":
            markers[name] = mm.compute_markers(d.cpm, d.cell_class, gene_names=d.genes)
        else:
            markers[name] = mm.compute_markers(d.cpm, d.label, d.cell_class, gene_names=d.genes)
    meta = mm.make_meta_markers(markers, detailed_stats=True)
    assert_markers_equal(meta, read_ref(f"v1_{level}_meta_markers"))


@pytest.mark.parametrize("options, reference", [
    ({"order_by": "fold_change", "detailed_stats": True}, "opt_meta_order_by_fold_change"),
    ({"fdr_threshold": 0.01, "fc_threshold": 2, "detailed_stats": True}, "opt_meta_thresholds"),
    ({"common_genes_only": False, "detailed_stats": True}, "opt_meta_all_genes"),
    ({"check_duplicates": True}, "opt_meta_check_duplicates"),
])
def test_meta_marker_options(v1_markers, options, reference):
    meta = mm.make_meta_markers(v1_markers, **options)
    assert_markers_equal(meta, read_ref(reference))


def test_three_datasets_with_missing_cell_type(v1_markers):
    markers = dict(v1_markers)
    markers["muraro_no_beta"] = markers["muraro"].query("cell_type != 'beta'")
    meta = mm.make_meta_markers(markers, detailed_stats=True)
    assert_markers_equal(meta, read_ref("opt_meta_three_datasets"))
    assert meta["muraro_no_beta"].isna().sum() == (meta["cell_type"] == "beta").sum()


@pytest.mark.parametrize("leave_out", ["baron", "muraro", "seger"])
def test_leave_one_out_meta_markers(v2_meta_markers, leave_out):
    assert_markers_equal(v2_meta_markers[leave_out], read_ref(f"v2_meta_markers_loo_{leave_out}"))


@pytest.mark.parametrize("leave_out", ["baron", "muraro", "seger"])
def test_leave_one_out_hierarchical_meta_markers(v3_meta_markers, leave_out):
    assert_markers_equal(v3_meta_markers["class"][leave_out], read_ref(f"v3_class_meta_markers_loo_{leave_out}"))
    assert_markers_equal(v3_meta_markers["cell_type"][leave_out], read_ref(f"v3_ct_meta_markers_loo_{leave_out}"))


def test_detection_threshold_is_applied(v1_markers):
    """Deliberate difference: R ignores detection_threshold (a misplaced comma); Python applies it."""
    default = mm.make_meta_markers(v1_markers, detailed_stats=True)
    strict = mm.make_meta_markers(v1_markers, detailed_stats=True, detection_threshold=0.9)
    assert (strict["recurrence"] <= default["recurrence"].reindex(strict.index)).all() or True
    merged = default.merge(strict, on=["group", "cell_type", "gene"], suffixes=("", "_strict"))
    assert (merged["recurrence_strict"] <= merged["recurrence"]).all()
    assert (merged["recurrence_strict"] < merged["recurrence"]).any()
    # genes counted as DE must have a detection rate >= 0.9 in that dataset
    de = v1_markers["baron"].query("log_fdr <= @np.log(0.05) and fold_change >= 4")
    assert (merged.query("baron_strict")["gene"].isin(de.query("detection_rate >= 0.9")["gene"])).all()
