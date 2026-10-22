# Generate reference outputs with the R MetaMarkers package.
#
# Usage (from the pymetamarkers directory, after scripts/convert_to_h5ad.R):
#   Rscript scripts/make_r_reference.R [h5ad_dir] [output_dir]
#
# The datasets are read back from the h5ad files so that R and Python work from
# exactly the same input. The script follows the three vignettes of the R
# package (MetaMarkers, Annotation, HierarchicalAnnotation) and adds a few
# option variants that the vignettes do not exercise. Doubles are written with
# 17 significant digits so that Python can check results to the last bit.

suppressPackageStartupMessages({
    library(MetaMarkers)
    library(SingleCellExperiment)
    library(zellkonverter)
    library(dplyr)
})

args = commandArgs(trailingOnly = TRUE)
h5ad_dir = if (length(args) > 0) args[1] else file.path("data", "h5ad")
ref_dir = if (length(args) > 1) args[2] else file.path("data", "r_reference")
dir.create(ref_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(ref_dir, "r_export"), showWarnings = FALSE)
set.seed(1)

# ---- helpers ---------------------------------------------------------------

format_double = function(x) {
    result = sprintf("%.17g", x)
    result[is.na(x)] = "NA"
    result[is.nan(x)] = "NaN"
    result[!is.na(x) & x == Inf] = "inf"
    result[!is.na(x) & x == -Inf] = "-inf"
    result
}

write_ref = function(df, name) {
    df = as.data.frame(df, stringsAsFactors = FALSE)
    for (column in names(df)) {
        if (is.double(df[[column]])) df[[column]] = format_double(df[[column]])
        if (is.factor(df[[column]])) df[[column]] = as.character(df[[column]])
    }
    data.table::fwrite(df, file.path(ref_dir, paste0(name, ".csv.gz")), na = "NA")
}

write_matrix = function(m, name) {
    m = as.matrix(m)
    row_names = if (is.null(rownames(m))) as.character(seq_len(nrow(m))) else rownames(m)
    df = data.frame(row = row_names, stringsAsFactors = FALSE)
    df = cbind(df, as.data.frame(m, stringsAsFactors = FALSE, optional = TRUE))
    write_ref(df, name)
}

write_vector = function(x, name) {
    write_ref(data.frame(value = x), name)
}

read_dataset = function(name) {
    filename = file.path(h5ad_dir, paste0(name, ".h5ad"))
    sce = suppressWarnings(readH5AD(filename, X_name = "counts"))
    counts = as(assay(sce, "counts"), "dgCMatrix")
    rownames(counts) = rowData(sce)$symbol
    colnames(counts) = colnames(sce)
    obs = as.data.frame(colData(sce), optional = TRUE)
    # readH5AD applies make.names() to obs columns, restore the original names
    original_names = rhdf5::h5readAttributes(filename, "obs")[["column-order"]]
    stopifnot(identical(make.names(original_names), names(obs)))
    names(obs) = original_names
    for (column in names(obs)) {
        if (is.factor(obs[[column]])) obs[[column]] = as.character(obs[[column]])
    }
    list(counts = counts, obs = obs)
}

# ---- load data and check the h5ad round trip ---------------------------------

message("Reading h5ad files")
baron = read_dataset("baron")
muraro = read_dataset("muraro")
seger = read_dataset("segerstolpe")

message("Checking h5ad files against scRNAseq")
suppressPackageStartupMessages(library(scRNAseq))
check_dataset = function(h5ad, sce, gene_names, label_column) {
    original = as(counts(sce), "dgCMatrix")
    rownames(original) = gene_names
    stopifnot(isTRUE(all.equal(original, h5ad$counts, tolerance = 0)))
    stopifnot(identical(as.character(colData(sce)[[label_column]]), h5ad$obs[[label_column]]))
}
sce = BaronPancreasData()
check_dataset(baron, sce, rownames(sce), "label")
sce = MuraroPancreasData()
check_dataset(muraro, sce, rowData(sce)$symbol, "label")
sce = SegerstolpePancreasData()
check_dataset(seger, sce, rownames(sce), "cell type")
rm(sce)
write_ref(data.frame(check = "h5ad counts and labels identical to scRNAseq", passed = TRUE),
          "h5ad_roundtrip_check")

# ---- normalization ------------------------------------------------------------

message("Normalization")
for (name in c("baron", "muraro", "seger")) {
    counts = get(name)$counts
    write_vector(Matrix::colSums(counts), paste0("library_size_", name))
    cpm = convert_to_cpm(counts)
    # random sample of non-zero CPM values (0-based gene/cell indices)
    nz = Matrix::summary(cpm)
    keep = sort(sample(nrow(nz), 5000))
    write_ref(data.frame(gene = nz$i[keep] - 1L, cell = nz$j[keep] - 1L, cpm = nz$x[keep]),
              paste0("cpm_sample_", name))
}
dense_example = matrix(c(1, 0, 3, 2, 2, 0, 0, 5, 1, 1, 1, 1), nrow = 3)
write_matrix(convert_to_cpm(dense_example, scale_factor = 100), "cpm_dense_example")

# ---- vignette 1: MetaMarkers ---------------------------------------------------

message("Vignette: MetaMarkers")
label1 = baron$obs$label
label1[label1 == "activated_stellate"] = "mesenchymal"
label1[label1 == "quiescent_stellate"] = "mesenchymal"
label2 = muraro$obs$label
label2[label2 == "duct"] = "ductal"
label2[label2 == "pp"] = "gamma"
common_labels = intersect(unique(label1), unique(label2))
write_vector(common_labels, "v1_common_labels")

keep1 = label1 %in% common_labels
keep2 = label2 %in% common_labels
cpm1 = convert_to_cpm(baron$counts[, keep1])
cpm2 = convert_to_cpm(muraro$counts[, keep2])
label1 = label1[keep1]
label2 = label2[keep2]

markers1 = compute_markers(cpm1, label1)
markers2 = compute_markers(cpm2, label2)
write_ref(markers1, "v1_markers_baron")
write_ref(markers2, "v1_markers_muraro")

export_markers(markers1, file.path(ref_dir, "r_export", "baron.csv"))
export_markers(markers2, file.path(ref_dir, "r_export", "muraro.csv"))
markers = list(
    baron = read_markers(file.path(ref_dir, "r_export", "baron.csv.gz")),
    muraro = read_markers(file.path(ref_dir, "r_export", "muraro.csv.gz"))
)
meta_markers = make_meta_markers(markers, detailed_stats = TRUE)
write_ref(meta_markers, "v1_meta_markers_from_files")

markers = list(baron = markers1, muraro = markers2)
meta_markers = make_meta_markers(markers, detailed_stats = TRUE)
write_ref(meta_markers, "v1_meta_markers")
write_ref(make_meta_markers(markers), "v1_meta_markers_simple")
export_meta_markers(meta_markers, file.path(ref_dir, "r_export", "meta_markers.csv"),
                    names(markers))

pareto_calls = list(
    list(id = "beta_r1", cell_type = "beta", min_recurrence = 1),
    list(id = "beta_r0", cell_type = "beta", min_recurrence = 0),
    list(id = "gamma_r0", cell_type = "gamma", min_recurrence = 0),
    list(id = "mesenchymal_r1", cell_type = "mesenchymal", min_recurrence = 1),
    list(id = "mesenchymal_r0", cell_type = "mesenchymal", min_recurrence = 0)
)
pareto = bind_rows(lapply(pareto_calls, function(call) {
    genes = get_pareto_markers(meta_markers, call$cell_type, call$min_recurrence)
    tibble::tibble(id = call$id, position = seq_along(genes), gene = genes)
}))
write_ref(pareto, "v1_pareto_markers")

# data behind the Pareto plots (what plot_pareto_markers / plot_pareto_summary draw)
plot_data = bind_rows(lapply(c("beta", "gamma", "mesenchymal"), function(ct) {
    plot_pareto_markers(meta_markers, ct, min_recurrence = 0)$data %>%
        ungroup() %>%
        mutate(recurrence = as.integer(as.character(recurrence)))
}))
write_ref(plot_data, "v1_pareto_plot_data")
summary_data = plot_pareto_summary(meta_markers, min_recurrence = 0)$data %>% ungroup()
write_ref(summary_data, "v1_pareto_summary_data")

# hierarchical marker lists (class level)
to_class = c("acinar" = "non-endocrine", "alpha" = "endocrine", "beta" = "endocrine",
             "delta" = "endocrine", "ductal" = "non-endocrine",
             "endothelial" = "non-endocrine", "epsilon" = "endocrine",
             "gamma" = "endocrine", "mesenchymal" = "non-endocrine")
class1 = unname(to_class[label1])
class2 = unname(to_class[label2])
class_markers = list(
    baron = compute_markers(cpm1, class1),
    muraro = compute_markers(cpm2, class2)
)
write_ref(class_markers$baron, "v1_class_markers_baron")
write_ref(class_markers$muraro, "v1_class_markers_muraro")
class_meta_markers = make_meta_markers(class_markers, detailed_stats = TRUE)
write_ref(class_meta_markers, "v1_class_meta_markers")
pareto = bind_rows(
    tibble::tibble(id = "endocrine_r1_top10",
                   gene = get_pareto_markers(class_meta_markers, "endocrine", 1)[1:10]),
    tibble::tibble(id = "non-endocrine_r1",
                   gene = get_pareto_markers(class_meta_markers, "non-endocrine", 1))
) %>% group_by(id) %>% mutate(position = seq_len(n())) %>% ungroup()
write_ref(pareto, "v1_class_pareto_markers")

# within-class markers
within_markers = list(
    baron = compute_markers(cpm1, label1, class1),
    muraro = compute_markers(cpm2, label2, class2)
)
write_ref(within_markers$baron, "v1_within_markers_baron")
write_ref(within_markers$muraro, "v1_within_markers_muraro")
within_meta_markers = make_meta_markers(within_markers, detailed_stats = TRUE)
write_ref(within_meta_markers, "v1_within_meta_markers")
genes = get_pareto_markers(within_meta_markers, "gamma", min_recurrence = 0)
write_ref(tibble::tibble(id = "gamma_r0", position = seq_along(genes), gene = genes),
          "v1_within_pareto_markers")

# ---- option variants --------------------------------------------------------

message("Option variants")
write_ref(compute_markers(cpm2, label2, two_tailed = TRUE), "opt_markers_muraro_two_tailed")
write_ref(compute_markers(cpm2, label2, tie_correction = TRUE), "opt_markers_muraro_tie_correction")
write_ref(compute_markers(cpm2, label2, class2, two_tailed = TRUE, tie_correction = TRUE),
          "opt_markers_muraro_grouped_two_tailed_tie_correction")
# genes as columns, dense input
dense_subset = as.matrix(Matrix::t(cpm2[1:300, ]))
write_ref(compute_markers(dense_subset, label2, genes_are_rows = FALSE),
          "opt_markers_muraro_dense_cells_as_rows")
# group with a single cell type is skipped
lonely_group = ifelse(label2 == "acinar", "solo", "rest")
write_ref(suppressWarnings(compute_markers(cpm2[1:300, ], label2, lonely_group)),
          "opt_markers_muraro_single_cell_type_group")

markers = list(baron = markers1, muraro = markers2)
write_ref(make_meta_markers(markers, order_by = "fold_change", detailed_stats = TRUE),
          "opt_meta_order_by_fold_change")
write_ref(make_meta_markers(markers, fdr_threshold = 0.01, fc_threshold = 2, detailed_stats = TRUE),
          "opt_meta_thresholds")
write_ref(make_meta_markers(markers, common_genes_only = FALSE, detailed_stats = TRUE),
          "opt_meta_all_genes")
write_ref(make_meta_markers(markers, check_duplicates = TRUE), "opt_meta_check_duplicates")
# three datasets (mean over 3 values) with a cell type missing from one dataset
markers3 = list(baron = markers1, muraro = markers2,
                muraro_no_beta = filter(markers2, cell_type != "beta"))
write_ref(make_meta_markers(markers3, detailed_stats = TRUE), "opt_meta_three_datasets")

# small helpers
write_matrix(design_matrix(c("b", "a", "c", "a")), "helper_design_matrix")
write_matrix(design_matrix(c("b", "a", "c", "a"), scale_columns = TRUE), "helper_design_matrix_scaled")
labels = c("all|beta", "endocrine|alpha", "non-endocrine|acinar|sub")
write_ref(data.frame(label = labels, group = get_group(labels), cell_type = get_cell_type(labels)),
          "helper_group_cell_type")
write_ref(data.frame(label = c("beta", "all|alpha"), group = get_group(c("beta", "all|alpha")),
                     cell_type = get_cell_type(c("beta", "all|alpha"))),
          "helper_group_cell_type_mixed")
dup_markers = bind_rows(head(markers2, 5), head(markers2, 2), markers2[10:12, ])
write_ref(dup_markers, "helper_duplicated_input")
write_ref(remove_duplicated_genes(dup_markers), "helper_duplicated_output")

# ---- vignette 2: Annotation ---------------------------------------------------

message("Vignette: Annotation")
label3 = seger$obs$`cell type`
label3 = gsub(" cell", "", label3)
label3[label3 == "PSC"] = "mesenchymal"
all_labels = list(
    baron = baron$obs$label, muraro = muraro$obs$label, seger = label3
)
all_labels$baron[all_labels$baron %in% c("activated_stellate", "quiescent_stellate")] = "mesenchymal"
all_labels$muraro[all_labels$muraro == "duct"] = "ductal"
all_labels$muraro[all_labels$muraro == "pp"] = "gamma"
common_labels = Reduce(intersect, lapply(all_labels, unique))
write_vector(common_labels, "v2_common_labels")

datasets = list()
for (name in names(all_labels)) {
    counts = list(baron = baron, muraro = muraro, seger = seger)[[name]]$counts
    keep = all_labels[[name]] %in% common_labels
    datasets[[name]] = list(
        cpm = convert_to_cpm(counts[, keep]),
        label = all_labels[[name]][keep]
    )
    datasets[[name]]$class = ifelse(
        datasets[[name]]$label %in% c("alpha", "beta", "gamma", "delta", "epsilon"),
        "endocrine", "non-endocrine"
    )
    datasets[[name]]$logcpm = log1p(datasets[[name]]$cpm)
}

markers = lapply(datasets, function(d) compute_markers(d$cpm, d$label))
write_ref(markers$seger, "v2_markers_seger")
meta_markers = lapply(names(markers), function(leave_out) {
    make_meta_markers(markers[names(markers) != leave_out])
})
names(meta_markers) = names(markers)
for (name in names(meta_markers)) write_ref(meta_markers[[name]], paste0("v2_meta_markers_loo_", name))

thresholds = seq(1, 5, 0.1)
write_vector(thresholds, "v2_thresholds")
for (test_dataset in names(datasets)) {
    top_markers = filter(meta_markers[[test_dataset]], rank <= 100)
    ct_scores = score_cells(datasets[[test_dataset]]$logcpm, top_markers)
    ct_enrichment = compute_marker_enrichment(ct_scores)
    ct_pred = suppressWarnings(assign_cells(ct_scores))
    true_labels = datasets[[test_dataset]]$label
    write_matrix(ct_scores, paste0("v2_scores_", test_dataset))
    write_matrix(ct_enrichment, paste0("v2_enrichment_", test_dataset))
    write_ref(ct_pred, paste0("v2_pred_", test_dataset))
    write_matrix(summarize_auroc(ct_enrichment, true_labels), paste0("v2_auroc_", test_dataset))
    write_matrix(summarize_fold_change(ct_enrichment, true_labels), paste0("v2_fold_change_", test_dataset))
    write_ref(summarize_precision_recall(ct_enrichment, true_labels, thresholds),
              paste0("v2_precision_recall_", test_dataset))
}
for (n in c(1, 5, 1000)) {
    top_markers = filter(meta_markers$seger, rank <= n)
    ct_scores = score_cells(datasets$seger$logcpm, top_markers)
    write_matrix(ct_scores, paste0("v2_scores_seger_top", n))
}

# other marker set formats
top_markers = filter(meta_markers$seger, rank <= 20)
marker_list = split(top_markers$gene, top_markers$cell_type)
write_matrix(score_cells(datasets$seger$logcpm, marker_list), "v2_scores_seger_from_list")
weights = marker_table_to_matrix(top_markers, rownames(datasets$seger$logcpm), weighted = FALSE)
write_matrix(score_cells(datasets$seger$logcpm, as.matrix(weights)), "v2_scores_seger_unweighted")
small_list = list(beta = c("INS", "IAPP", "NOT_A_GENE"), alpha = c("GCG", "TTR"))
m = marker_list_to_matrix(small_list, c("GCG", "INS", "IAPP", "TTR", "OTHER"))
write_matrix(as.matrix(m), "helper_marker_list_to_matrix")

# ---- vignette 3: HierarchicalAnnotation -----------------------------------------

message("Vignette: HierarchicalAnnotation")
class_markers = lapply(datasets, function(d) compute_markers(d$cpm, d$class))
class_meta_markers = lapply(names(class_markers), function(leave_out) {
    make_meta_markers(class_markers[names(class_markers) != leave_out])
})
names(class_meta_markers) = names(class_markers)
ct_markers = lapply(datasets, function(d) compute_markers(d$cpm, d$label, d$class))
write_ref(ct_markers$seger, "v3_ct_markers_seger")
ct_meta_markers = lapply(names(ct_markers), function(leave_out) {
    make_meta_markers(ct_markers[names(ct_markers) != leave_out])
})
names(ct_meta_markers) = names(ct_markers)
for (name in names(ct_meta_markers)) {
    write_ref(class_meta_markers[[name]], paste0("v3_class_meta_markers_loo_", name))
    write_ref(ct_meta_markers[[name]], paste0("v3_ct_meta_markers_loo_", name))
}

test_dataset = "seger"
top_markers = filter(class_meta_markers[[test_dataset]], rank <= 100)
class_scores = score_cells(datasets[[test_dataset]]$logcpm, top_markers)
class_enrichment = compute_marker_enrichment(class_scores)
class_pred = assign_cells(class_scores)
write_matrix(class_scores, "v3_class_scores_seger")
write_matrix(class_enrichment, "v3_class_enrichment_seger")
write_ref(class_pred, "v3_class_pred_seger")

top_markers = filter(ct_meta_markers[[test_dataset]], rank <= 100)
ct_scores = score_cells(datasets[[test_dataset]]$logcpm, top_markers)
ct_enrichment = compute_marker_enrichment(ct_scores, by_group = TRUE)
ct_pred = assign_cells(ct_scores, group_assignment = class_pred$predicted)
write_matrix(ct_scores, "v3_ct_scores_seger")
write_matrix(ct_enrichment, "v3_ct_enrichment_seger")
write_matrix(suppressWarnings(compute_marker_enrichment(ct_scores, by_group = FALSE)),
             "v3_ct_enrichment_flat_seger")
write_ref(ct_pred, "v3_ct_pred_seger")
write_ref(suppressWarnings(assign_cells(ct_scores)), "v3_ct_pred_no_groups_seger")
odd_assignment = class_pred$predicted
odd_assignment[seq(1, length(odd_assignment), 50)] = "unknown_group"
write_ref(suppressWarnings(assign_cells(ct_scores, group_assignment = odd_assignment)),
          "v3_ct_pred_unknown_group_seger")
write_vector(odd_assignment, "v3_odd_assignment_seger")

for (test_dataset in names(datasets)) {
    markers_top = filter(ct_meta_markers[[test_dataset]], rank <= 100)
    scores = score_cells(datasets[[test_dataset]]$logcpm, markers_top)
    enrichment = compute_marker_enrichment(scores, by_group = TRUE)
    true_labels = datasets[[test_dataset]]$label
    keep_cell = datasets[[test_dataset]]$class == "endocrine"
    keep_marker_set = get_group(rownames(enrichment)) == "endocrine"
    pr = summarize_precision_recall(
        enrichment[keep_marker_set, keep_cell], true_labels[keep_cell], seq(1, 3, 0.1)
    )
    write_ref(pr, paste0("v3_precision_recall_endocrine_", test_dataset))
}

message("Done: ", length(list.files(ref_dir, pattern = "csv.gz$")), " reference files in ", ref_dir)
