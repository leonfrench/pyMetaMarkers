"""Optional matplotlib versions of the R package's plots (requires matplotlib)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._data import resolve_expression
from .pareto import pareto_markers_data, pareto_summary_data


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib: pip install matplotlib") from error
    return plt


def plot_pareto_markers(meta_markers, cell_type_name, min_recurrence=1, min_auroc=0.5, min_fc=0,
                        fc_threshold=3, auroc_threshold=0.8, ax=None):
    """Markers in the (log2 fold change of detection, AUROC) space, Pareto front labeled."""
    plt = _pyplot()
    data = pareto_markers_data(meta_markers, cell_type_name, min_recurrence, min_auroc, min_fc)
    ax = ax or plt.figure(figsize=(6, 4.5)).gca()
    max_recurrence = int(meta_markers["recurrence"].max())
    levels = list(range(int(min_recurrence), max_recurrence + 1))
    colors = plt.get_cmap("Blues")(np.linspace(0.35, 0.9, max(len(levels) - 1, 1)))
    for i, level in enumerate(levels):
        subset = data[data["recurrence"] == level]
        color = "black" if level == max_recurrence else colors[min(i, len(colors) - 1)]
        ax.scatter(np.log2(subset["fold_change_detection"]), subset["auroc"], s=8, color=color, label=str(level))
    front = data[data["is_pareto"]].sort_values("fold_change_detection")
    ax.plot(np.log2(front["fold_change_detection"]), front["auroc"], color="black", linewidth=1)
    for _, row in front.iterrows():
        ax.annotate(row["gene"], (np.log2(row["fold_change_detection"]), row["auroc"]), fontsize=8,
                    bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})
    if auroc_threshold is not None:
        ax.axhline(auroc_threshold, linestyle="--", color="grey")
    if fc_threshold is not None:
        ax.axvline(fc_threshold, linestyle="--", color="grey")
    ax.set_xlabel("log2(Fold change) of detection rate")
    ax.set_ylabel("AUROC")
    ax.legend(title="# Datasets", fontsize=8)
    return ax


def plot_pareto_summary(meta_markers, min_recurrence=1, ax=None):
    """Pareto fronts of all cell types in the (log2 fold change of detection, AUROC) space."""
    plt = _pyplot()
    data = pareto_summary_data(meta_markers, min_recurrence)
    ax = ax or plt.figure(figsize=(6, 4.5)).gca()
    for cell_type, subset in data.groupby("cell_type", sort=True):
        subset = subset.sort_values("fold_change_detection")
        line, = ax.plot(np.log2(subset["fold_change_detection"]), subset["auroc"], label=cell_type)
        for _, row in subset.iterrows():
            ax.annotate(row["gene"], (np.log2(row["fold_change_detection"]), row["auroc"]), fontsize=7,
                        color=line.get_color())
    ax.axhline(0.8, linestyle="--", color="grey")
    ax.axvline(3, linestyle="--", color="grey")
    ax.set_xlabel("log2(Fold change) of detection rate")
    ax.set_ylabel("AUROC")
    ax.legend(fontsize=7)
    return ax


def plot_marker_expression(expression, genes, cell_type_label, *, gene_names=None, layer=None):
    """Violin plots of marker expression (log10(x + 1)) by cell type, one row per gene."""
    plt = _pyplot()
    resolved = resolve_expression(expression, layer=layer)
    try:
        names = resolved.gene_names(gene_names)
        labels = resolved.labels(cell_type_label, "cell_type_label")
        position = {}
        for i, name in enumerate(names):
            position.setdefault(name, i)
        columns = [position[g] for g in genes]
        values = {}
        for start, stop, block in resolved.source.column_blocks():
            for gene, column in zip(genes, columns):
                if start <= column < stop:
                    values[gene] = np.asarray(block[:, column - start].todense()).ravel()
    finally:
        resolved.close()
    cell_types = sorted(set(label for label in labels if label is not None))
    fig, axes = plt.subplots(len(genes), 1, figsize=(6, 1.2 * len(genes) + 1), sharex=True, squeeze=False)
    for ax, gene in zip(axes[:, 0], genes):
        data = [np.log10(values[gene][labels == ct] + 1) for ct in cell_types]
        ax.violinplot(data, showextrema=False, widths=0.9)
        ax.set_ylabel(gene, rotation=0, ha="right")
    axes[-1, 0].set_xticks(range(1, len(cell_types) + 1), cell_types, rotation=45, ha="right")
    fig.supylabel("Gene expression (log10)")
    return fig


def plot_marker_scores(scores, umap_coordinates, normalize_scores=False, point_size=1):
    """Marker scores in UMAP space, one panel per marker set (scores: cells x marker sets)."""
    plt = _pyplot()
    coordinates = np.asarray(umap_coordinates, dtype=np.float64)[:, :2]
    n_sets = scores.shape[1]
    n_columns = int(np.ceil(np.sqrt(n_sets)))
    n_rows = int(np.ceil(n_sets / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(3 * n_columns, 3 * n_rows), squeeze=False)
    for ax, name in zip(axes.ravel(), scores.columns):
        values = scores[name].to_numpy(dtype=np.float64)
        if normalize_scores:
            values = (values - values.min()) / (values.max() - values.min())
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c=values, s=point_size, cmap="Greens")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n_sets:]:
        ax.set_visible(False)
    return fig


def plot_assignments(assignments, umap_coordinates, enrichment_threshold=1, point_size=1, ax=None):
    """Cell type assignments in UMAP space (cells below the enrichment threshold in grey)."""
    plt = _pyplot()
    coordinates = np.asarray(umap_coordinates, dtype=np.float64)[:, :2]
    predicted = assignments["predicted"].astype(object).to_numpy()
    unassigned = (assignments["enrichment"].to_numpy() < enrichment_threshold) | (predicted == "unassigned")
    unassigned |= pd.isna(predicted)
    ax = ax or plt.figure(figsize=(6, 5)).gca()
    ax.scatter(coordinates[unassigned, 0], coordinates[unassigned, 1], s=point_size, color="0.8", label="NA")
    for cell_type in sorted(set(predicted[~unassigned])):
        selected = ~unassigned & (predicted == cell_type)
        ax.scatter(coordinates[selected, 0], coordinates[selected, 1], s=point_size, label=cell_type)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(title="Cell type", markerscale=4, fontsize=8)
    return ax


def compute_umap(expr, **kwargs) -> pd.DataFrame:
    """UMAP coordinates of a cells x genes matrix (requires umap-learn)."""
    try:
        import umap
    except ImportError as error:  # pragma: no cover
        raise ImportError("compute_umap requires umap-learn: pip install umap-learn") from error
    resolved = resolve_expression(expr)
    try:
        X = resolved.source.X if hasattr(resolved.source, "X") else None
        if X is None:
            raise TypeError("compute_umap needs an in-memory matrix.")
        coordinates = umap.UMAP(**kwargs).fit_transform(X)
        samples = resolved.obs_names if resolved.obs_names is not None else np.arange(len(coordinates))
    finally:
        resolved.close()
    return pd.DataFrame({"sample": samples, "umap_1": coordinates[:, 0], "umap_2": coordinates[:, 1]})
