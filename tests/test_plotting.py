"""Smoke tests for the optional matplotlib plots."""

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import metamarkers as mm  # noqa: E402
from metamarkers import plotting  # noqa: E402
from conftest import requires_data  # noqa: E402

pytestmark = requires_data


def test_pareto_plots(v1_markers):
    meta = mm.make_meta_markers(v1_markers, detailed_stats=True)
    ax = plotting.plot_pareto_markers(meta, "beta", min_recurrence=0)
    assert "INS" in [t.get_text() for t in ax.texts]
    ax = plotting.plot_pareto_summary(meta, min_recurrence=0)
    assert len(ax.get_lines()) >= 9


def test_marker_expression_and_umap_plots(v1_datasets):
    d = v1_datasets["muraro"]
    fig = plotting.plot_marker_expression(d.cpm, ["INS", "GCG"], d.label, gene_names=d.genes)
    assert len(fig.axes) == 2
    rng = np.random.default_rng(0)
    coordinates = rng.normal(size=(d.cpm.shape[0], 2))
    scores = pd.DataFrame(rng.random((d.cpm.shape[0], 3)), columns=["all|a", "all|b", "all|c"])
    fig = plotting.plot_marker_scores(scores, coordinates, normalize_scores=True)
    assert sum(ax.get_visible() for ax in fig.axes) == 3
    assignments = mm.assign_cells(scores)
    ax = plotting.plot_assignments(assignments, coordinates, enrichment_threshold=1)
    assert ax.get_legend() is not None
