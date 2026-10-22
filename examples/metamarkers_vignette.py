"""The R package's main vignette ("MetaMarkers: robust marker signatures"), in Python.

Usage (after scripts/convert_to_h5ad.R):
    python examples/metamarkers_vignette.py [data/h5ad]
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np

import metamarkers as mm

h5ad_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/h5ad")

# Load datasets and harmonize cell type names
dataset1 = ad.read_h5ad(h5ad_dir / "baron.h5ad")
dataset2 = ad.read_h5ad(h5ad_dir / "muraro.h5ad")
dataset1.obs["new_label"] = dataset1.obs["label"].astype(str).replace(
    {"activated_stellate": "mesenchymal", "quiescent_stellate": "mesenchymal"})
dataset2.obs["new_label"] = dataset2.obs["label"].astype(str).replace({"duct": "ductal", "pp": "gamma"})
common_labels = [l for l in dataset1.obs["new_label"].unique() if l in set(dataset2.obs["new_label"])]
print("Common cell types:", common_labels)

dataset1 = dataset1[dataset1.obs["new_label"].isin(common_labels)].copy()
dataset2 = dataset2[dataset2.obs["new_label"].isin(common_labels)].copy()
dataset1.layers["cpm"] = mm.convert_to_cpm(dataset1.X)
dataset2.layers["cpm"] = mm.convert_to_cpm(dataset2.X)

# Markers for individual datasets (gene symbols are in var["symbol"])
markers_dataset1 = mm.compute_markers(dataset1, "new_label", layer="cpm", gene_names="symbol")
markers_dataset2 = mm.compute_markers(dataset2, "new_label", layer="cpm", gene_names="symbol")
print(markers_dataset1.query("cell_type == 'beta'").head(6).to_string())

mm.export_markers(markers_dataset1, "baron.csv")
mm.export_markers(markers_dataset2, "muraro.csv")

# Meta-markers
markers = {"baron": mm.read_markers("baron.csv.gz"), "muraro": mm.read_markers("muraro.csv.gz")}
meta_markers = mm.make_meta_markers(markers, detailed_stats=True)
print(meta_markers.query("cell_type == 'beta'").head(6).to_string())
for cell_type, min_recurrence in [("beta", 1), ("gamma", 0), ("mesenchymal", 1)]:
    print(f"Pareto markers ({cell_type}):", mm.get_pareto_markers(meta_markers, cell_type, min_recurrence))
mm.export_meta_markers(meta_markers, "meta_markers.csv", list(markers))

# Hierarchical marker lists
endocrine = {"alpha", "beta", "delta", "epsilon", "gamma"}
for dataset in (dataset1, dataset2):
    dataset.obs["class"] = np.where(dataset.obs["new_label"].isin(endocrine), "endocrine", "non-endocrine")
class_markers = {
    "baron": mm.compute_markers(dataset1, "class", layer="cpm", gene_names="symbol"),
    "muraro": mm.compute_markers(dataset2, "class", layer="cpm", gene_names="symbol"),
}
class_meta_markers = mm.make_meta_markers(class_markers, detailed_stats=True)
print("Pareto markers (endocrine):", mm.get_pareto_markers(class_meta_markers, "endocrine", 1)[:10])

within_markers = {
    "baron": mm.compute_markers(dataset1, "new_label", "class", layer="cpm", gene_names="symbol"),
    "muraro": mm.compute_markers(dataset2, "new_label", "class", layer="cpm", gene_names="symbol"),
}
within_meta_markers = mm.make_meta_markers(within_markers, detailed_stats=True)
print("Pareto markers (gamma, within endocrine):", mm.get_pareto_markers(within_meta_markers, "gamma", 0))
