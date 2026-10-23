# pymetamarkers

A Python port of the R package [MetaMarkers](https://github.com/gillislab/MetaMarkers)
(Fischer & Gillis, *iScience* 2021) that works with AnnData objects and h5ad files,
including datasets that do not fit in memory.

It reproduces the R package's outputs: on the three vignette datasets, marker
statistics, meta-marker rankings, marker scores, cell assignments and
performance summaries agree with R to floating-point precision (see
[Agreement with R](#agreement-with-r)).

## Installation

```bash
pip install -e .            # numpy, scipy, pandas, anndata, h5py
pip install -e ".[plot]"    # + matplotlib for the optional plots
```

## Quick start

```python
import anndata as ad
import metamarkers as mm

# 1. markers for each dataset (h5ad paths are read in blocks, X holds raw counts)
markers = {
    "baron": mm.compute_markers("baron.h5ad", "cell_type", gene_names="symbol", normalize="cpm"),
    "muraro": mm.compute_markers("muraro.h5ad", "cell_type", gene_names="symbol", normalize="cpm"),
}
mm.export_markers(markers["baron"], "baron.csv")          # writes baron.csv.gz

# 2. meta-markers
meta_markers = mm.make_meta_markers(markers, detailed_stats=True)
mm.get_pareto_markers(meta_markers, "beta", min_recurrence=1)

# 3. annotation of a new dataset
top_markers = meta_markers.query("rank <= 100")
scores = mm.score_cells("target.h5ad", top_markers, gene_names="symbol", normalize="log1p_cpm")
enrichment = mm.compute_marker_enrichment(scores)
predictions = mm.assign_cells(scores)
```

In-memory data work the same way: pass an `AnnData` object (`layer=` selects a
layer) or a cells x genes matrix (dense or scipy sparse) with labels as arrays.

The R vignettes are reproduced step by step in the test suite
(`tests/test_markers.py`, `tests/test_meta_markers.py`, `tests/test_annotation.py`).

## Large datasets

* **h5ad paths and backed AnnData** (`ad.read_h5ad(path, backed="r")`) are never
  loaded whole. Genes are processed in blocks of at most `max_nnz_per_block`
  stored values, and peak memory depends on the block size, not on the dataset.
* **Threads:** `compute_markers` processes blocks in parallel (`n_jobs`, all CPU
  cores by default). By default about 20 million stored values are in flight
  in total (2.5–10 million per block), about 2 GB of memory. On 8 cores, a
  400,000-cell CSR file with 100 million stored values takes 6 s and peaks at
  1.7 GB (15 s with `n_jobs=1`); the Baron dataset in memory takes 1.1 s.
* **CSC on disk** is read block by block directly. **CSR on disk** (the scanpy
  default) is split into gene blocks once through temporary files (`tmp_dir=`,
  default system temp), so memory stays bounded; this costs one extra pass over
  the data. An **in-memory CSR** matrix is converted to CSC once (one copy) when
  it does not fit in a single block.
* **`verbose=True`** (`compute_markers`) prints each stage with the elapsed
  time, and the progress of long stages about once a minute. The first line
  describes the storage (layout, compression, chunks); for CSR on disk, the two
  passes over the file (counting values per gene, splitting it into gene
  blocks) run on a single thread.
* **`normalize="cpm"`** (`compute_markers`) and **`normalize="log1p_cpm"`**
  (`score_cells`) normalize on the fly, so files only need raw counts.
* **Cells without a label** (`None`/`NaN`) are ignored, so subsets can be analysed
  without copying the matrix.

## R to Python

| R | Python | Notes |
|---|---|---|
| `compute_markers(expr, labels, groups)` | `compute_markers(data, labels, groups)` | cells are rows by default (`genes_are_rows=False`) |
| `make_meta_markers(list(...))` | `make_meta_markers({...})` | dict of marker tables |
| `convert_to_cpm(M)` | `convert_to_cpm(M)` | or `normalize="cpm"` on the fly |
| `score_cells(expr, markers)` | `score_cells(data, markers)` | returns **cells x marker sets** (R: marker sets x cells) |
| `compute_marker_enrichment(scores)` | same | cells x marker sets |
| `assign_cells(scores, group_assignment)` | same | indexed by cell |
| `summarize_auroc` / `summarize_fold_change` / `summarize_precision_recall` | same | same orientation as R |
| `get_pareto_markers`, `design_matrix`, `get_group`, `get_cell_type`, `marker_list_to_matrix`, `marker_table_to_matrix`, `remove_duplicated_genes` | same | |
| `export_markers`, `read_markers`, `export_meta_markers`, `read_meta_markers`, `export_*_by_*` | same | files are interchangeable with R |
| `plot_pareto_markers`, `plot_pareto_summary`, `plot_marker_expression`, `plot_marker_scores`, `plot_assignments`, `compute_umap` | `metamarkers.plotting` | matplotlib (`compute_umap` needs `umap-learn`); `pareto_markers_data` / `pareto_summary_data` return the plotted data |

## Agreement with R

The test suite compares every function with outputs of the R package
(`scripts/make_r_reference.R`) on the vignette datasets. Statistics use plain
vectorized floating-point arithmetic, so values can differ from R in the last
digits (summation order); the tests allow a relative difference of 1e-9.
Everything discrete matches exactly: row order of marker tables, meta-marker
ranks and recurrence, Pareto markers, cell assignments, and the vignettes'
accuracies (95.6% flat, 98.3% hierarchical on Segerstolpe). AUROCs are exact,
because sums of ranks are exact in floating point.

Cells whose marker scores are all equal (e.g. no marker expression) get an
enrichment of exactly 1, so they never pass a threshold of 1 by rounding.

## Deliberate differences from R

* `make_meta_markers(detection_threshold=...)` is applied. In R a misplaced
  comma makes it a no-op. Results are identical at the default (0).
* `export_markers_by_group` and `export_markers_for_cell_type` write only the
  rows of the group/cell type. R's filters (`group == group`) keep every row.
* `assign_cells` reports a tie ("unassigned") only when the two highest scores
  are equal. R also reports a tie when the highest score is in the first marker
  set and all other scores are 0 (or negative), and when there is a single
  marker set.
* Cells with missing labels are ignored (R stops with an error).
* Sorting of cell type names always uses C-locale order (as `dplyr::arrange`).
* CSV numbers are correctly rounded to 15 significant digits. `fwrite`
  occasionally differs in the 15th digit; re-exporting an R file reproduces
  it byte for byte.
* The paper mentions a continuity correction in the ROC test; neither the R
  code nor this port applies one (tie correction is optional, as in R).

## Tests

The tests use the vignette datasets converted to h5ad and reference outputs
from the R package, both generated by the scripts in `scripts/` (R with
MetaMarkers, scRNAseq and zellkonverter installed):

```bash
Rscript scripts/convert_to_h5ad.R      # data/h5ad/{baron,muraro,segerstolpe}.h5ad
Rscript scripts/make_r_reference.R     # data/r_reference/*.csv.gz
pytest
```

`convert_to_h5ad.R` writes raw counts (X), the original cell metadata (obs) and
the gene names used by the vignettes (`var["symbol"]`; `var_names` are made
unique). `make_r_reference.R` reads the h5ad files back, checks them against the
original scRNAseq objects, and runs the three vignettes plus option variants.
Tests without the data (`tests/test_units.py`) always run.
