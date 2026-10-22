"""MetaMarkers: robust marker signatures from single-cell data (Python port).

A Python implementation of the R package MetaMarkers
(https://github.com/gillislab/MetaMarkers) that works with AnnData objects and
h5ad files, including datasets larger than memory.
"""

from importlib.metadata import PackageNotFoundError, version

from .annotation import (
    assign_cells,
    compute_marker_enrichment,
    get_cell_type,
    get_group,
    marker_list_to_matrix,
    marker_table_to_matrix,
    score_cells,
    summarize_auroc,
    summarize_fold_change,
    summarize_precision_recall,
)
from .aurocs import auroc_p_value, compute_aurocs, design_matrix, fdr
from .markers import (
    compute_markers,
    export_markers,
    export_markers_by_cell_type,
    export_markers_by_group,
    export_markers_for_cell_type,
    read_markers,
    remove_duplicated_genes,
)
from .meta_markers import (
    export_meta_markers,
    export_meta_markers_by_cell_type,
    make_meta_markers,
    read_meta_markers,
)
from .normalization import convert_to_cpm
from .pareto import get_pareto_markers, is_pareto_front, pareto_markers_data, pareto_summary_data

try:
    __version__ = version("pymetamarkers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "assign_cells", "auroc_p_value", "compute_aurocs", "compute_marker_enrichment", "compute_markers",
    "convert_to_cpm", "design_matrix", "export_markers", "export_markers_by_cell_type",
    "export_markers_by_group", "export_markers_for_cell_type", "export_meta_markers",
    "export_meta_markers_by_cell_type", "fdr", "get_cell_type", "get_group", "get_pareto_markers",
    "is_pareto_front", "make_meta_markers", "marker_list_to_matrix", "marker_table_to_matrix",
    "pareto_markers_data", "pareto_summary_data", "read_markers", "read_meta_markers",
    "remove_duplicated_genes", "score_cells", "summarize_auroc", "summarize_fold_change",
    "summarize_precision_recall",
]
