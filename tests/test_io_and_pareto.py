"""Import/export (interoperability with R files) and Pareto front markers."""

import gzip
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import metamarkers as mm
from conftest import REF_DIR, assert_markers_equal, assert_same_ranking, read_ref, requires_data

pytestmark = requires_data

@pytest.mark.parametrize("name", ["baron", "muraro"])
def test_read_r_marker_files(name):
    markers = mm.read_markers(REF_DIR / "r_export" / f"{name}.csv.gz")
    expected = read_ref(f"v1_markers_{name}")
    assert_markers_equal(markers, expected)


def test_read_r_meta_marker_file():
    meta = mm.read_meta_markers(REF_DIR / "r_export" / "meta_markers.csv.gz")
    expected = read_ref("v1_meta_markers")
    assert meta["baron"].dtype == bool
    assert_markers_equal(meta, expected)


def test_export_round_trip(tmp_path, v1_markers):
    path = mm.export_markers(v1_markers["muraro"], tmp_path / "muraro.csv")
    assert path.endswith("muraro.csv.gz")
    with gzip.open(path, "rt") as handle:
        header = handle.readline().strip()
    assert re.fullmatch(r"# Differential expression statistics generated on \d{4}-\d{2}-\d{2}\. "
                        r"Cell types: acinar, alpha, .*, mesenchymal\.", header)
    back = mm.read_markers(path)
    assert_markers_equal(back, v1_markers["muraro"])
    plain = mm.export_markers(v1_markers["muraro"].head(20), tmp_path / "plain.csv", gzip=False)
    assert plain.endswith("plain.csv") and open(plain).readline().startswith("# Differential")


def test_export_matches_r_format(tmp_path, v1_markers):
    """Python files use data.table::fwrite's format: re-exporting an R file reproduces it exactly."""
    r_file = REF_DIR / "r_export" / "baron.csv.gz"
    path = mm.export_markers(mm.read_markers(r_file), tmp_path / "baron.csv")
    with gzip.open(path, "rt") as handle:
        python_lines = handle.read().splitlines()
    with gzip.open(r_file, "rt") as handle:
        r_lines = handle.read().splitlines()
    strip_date = lambda line: re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", line)
    assert strip_date(python_lines[0]) == strip_date(r_lines[0])
    assert python_lines[1:] == r_lines[1:]
    # Exporting full-precision values: Python rounds to 15 significant digits correctly,
    # fwrite's fast formatter sometimes differs in the 15th digit.
    path = mm.export_markers(v1_markers["baron"], tmp_path / "baron_full.csv")
    back, from_r = mm.read_markers(path), mm.read_markers(r_file)
    assert_markers_equal(back, from_r)


def test_meta_marker_export(tmp_path, v1_markers):
    meta = mm.make_meta_markers(v1_markers, detailed_stats=True)
    path = mm.export_meta_markers(meta, tmp_path / "meta.csv", list(v1_markers))
    with gzip.open(path, "rt") as handle:
        header = handle.readline().strip()
    with gzip.open(REF_DIR / "r_export" / "meta_markers.csv.gz", "rt") as handle:
        r_header = handle.readline().strip()
    strip_date = lambda line: re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", line)
    assert strip_date(header) == strip_date(r_header)
    back = mm.read_meta_markers(path)
    assert_markers_equal(back, meta)


def test_export_by_group_and_cell_type(tmp_path, v1_datasets):
    d = v1_datasets["muraro"]
    markers = mm.compute_markers(d.cpm[:, :100], d.label, d.cell_class, gene_names=d.genes[:100])
    paths = mm.export_markers_by_group(markers, str(tmp_path / "group_"))
    assert sorted(p.split("/")[-1] for p in paths) == ["group_endocrine.csv.gz", "group_non.endocrine.csv.gz"]
    endocrine = mm.read_markers(tmp_path / "group_endocrine.csv.gz")
    assert set(endocrine["group"]) == {"endocrine"}
    paths = mm.export_markers_by_cell_type(markers, str(tmp_path / "ct_"))
    assert len(paths) == 9
    with gzip.open(tmp_path / "ct_beta.csv.gz", "rt") as handle:
        assert "Reference: beta, outgroups: " in handle.readline()
    assert set(mm.read_markers(tmp_path / "ct_beta.csv.gz")["cell_type"]) == {"beta"}
    meta = mm.make_meta_markers({"a": markers, "b": markers})
    paths = mm.export_meta_markers_by_cell_type(meta, str(tmp_path / "meta_"), ["a", "b"])
    assert set(mm.read_meta_markers(tmp_path / "meta_gamma.csv.gz")["cell_type"]) == {"gamma"}


@pytest.mark.requires_r
def test_r_reads_python_files(tmp_path, v1_markers):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript not available")
    check = subprocess.run([rscript, "-e", "library(MetaMarkers)"], capture_output=True)
    if check.returncode != 0:
        pytest.skip("R package MetaMarkers not installed")
    meta = mm.make_meta_markers(v1_markers, detailed_stats=True)
    markers_path = mm.export_markers(v1_markers["muraro"], tmp_path / "muraro.csv")
    meta_path = mm.export_meta_markers(meta, tmp_path / "meta.csv", list(v1_markers))
    out = tmp_path / "from_r.csv"
    script = f"""
        suppressPackageStartupMessages(library(MetaMarkers))
        markers = read_markers("{markers_path}")
        meta = read_meta_markers("{meta_path}")
        stopifnot(is.logical(meta$baron), nrow(markers) == {len(v1_markers["muraro"])})
        remade = make_meta_markers(list(baron = read_markers("{REF_DIR / 'r_export' / 'baron.csv.gz'}"),
                                        muraro = markers), detailed_stats = TRUE)
        data.table::fwrite(remade, "{out}")
    """
    result = subprocess.run([rscript, "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    remade = pd.read_csv(out, float_precision="round_trip", keep_default_na=False, na_values=[""])
    python = mm.make_meta_markers({"baron": mm.read_markers(REF_DIR / "r_export" / "baron.csv.gz"),
                                   "muraro": mm.read_markers(markers_path)}, detailed_stats=True)
    assert_same_ranking(python, remade)


# ---- Pareto front ------------------------------------------------------------------

@pytest.fixture(scope="module")
def v1_meta(v1_markers):
    return mm.make_meta_markers(v1_markers, detailed_stats=True)


def _reference_lists(name):
    table = read_ref(name)
    return {key: list(group.sort_values("position")["gene"]) for key, group in table.groupby("id")}


def test_pareto_markers(v1_meta):
    calls = {"beta_r1": ("beta", 1), "beta_r0": ("beta", 0), "gamma_r0": ("gamma", 0),
             "mesenchymal_r1": ("mesenchymal", 1), "mesenchymal_r0": ("mesenchymal", 0)}
    expected = _reference_lists("v1_pareto_markers")
    for key, (cell_type, min_recurrence) in calls.items():
        assert mm.get_pareto_markers(v1_meta, cell_type, min_recurrence) == expected[key], key
    assert expected["beta_r1"][:1] == ["INS"]


def test_class_and_within_class_pareto_markers(v1_datasets):
    class_markers = {n: mm.compute_markers(d.cpm, d.cell_class, gene_names=d.genes) for n, d in v1_datasets.items()}
    class_meta = mm.make_meta_markers(class_markers, detailed_stats=True)
    expected = _reference_lists("v1_class_pareto_markers")
    assert mm.get_pareto_markers(class_meta, "endocrine", 1)[:10] == expected["endocrine_r1_top10"]
    assert mm.get_pareto_markers(class_meta, "non-endocrine", 1) == expected["non-endocrine_r1"]
    within = {n: mm.compute_markers(d.cpm, d.label, d.cell_class, gene_names=d.genes) for n, d in v1_datasets.items()}
    within_meta = mm.make_meta_markers(within, detailed_stats=True)
    assert mm.get_pareto_markers(within_meta, "gamma", 0) == _reference_lists("v1_within_pareto_markers")["gamma_r0"]


def test_pareto_plot_data(v1_meta):
    expected = read_ref("v1_pareto_plot_data").drop(columns=".group")  # ggplot2 bookkeeping column
    data = pd.concat([mm.pareto_markers_data(v1_meta, ct, min_recurrence=0) for ct in ("beta", "gamma", "mesenchymal")],
                     ignore_index=True)
    assert_markers_equal(data, expected)
    summary = mm.pareto_summary_data(v1_meta, min_recurrence=0)
    expected = read_ref("v1_pareto_summary_data").drop(columns=".group")
    assert_markers_equal(summary, expected)


def test_is_pareto_front():
    x = [0.9, 0.8, 0.7, 0.95, 0.6]
    y = [1.0, 3.0, 2.0, 0.5, 4.0]
    assert list(mm.is_pareto_front(x, y)) == [True, True, False, True, True]
    assert list(mm.is_pareto_front([1, 1], [2, 2])) == [True, True]
