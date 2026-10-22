# Convert the pancreas datasets used by the MetaMarkers vignettes to h5ad.
#
# Usage (from the pymetamarkers directory):
#   Rscript scripts/convert_to_h5ad.R [output_dir]
#
# Each h5ad file contains:
#   X                raw counts (cells x genes, sparse CSR)
#   obs              the original colData of the SingleCellExperiment
#   var["symbol"]    the gene name used by the vignettes (Baron: rownames,
#                    Muraro: rowData()$symbol, Segerstolpe: rownames)
#   var_names        unique gene identifiers (Segerstolpe has 725 duplicated
#                    rownames, which are made unique with make.unique())

suppressPackageStartupMessages({
    library(scRNAseq)
    library(SingleCellExperiment)
    library(zellkonverter)
})

args = commandArgs(trailingOnly = TRUE)
output_dir = if (length(args) > 0) args[1] else file.path("data", "h5ad")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

to_h5ad_input = function(sce, symbol) {
    counts = as(counts(sce), "dgCMatrix")
    result = SingleCellExperiment(
        assays = list(counts = counts),
        colData = colData(sce)
    )
    rowData(result)$symbol = symbol
    if (!is.null(rowData(sce)$chr)) rowData(result)$chr = rowData(sce)$chr
    if (!is.null(rowData(sce)$refseq)) rowData(result)$refseq = rowData(sce)$refseq
    rownames(result) = make.unique(rownames(sce))
    # factors are written as categoricals, characters as strings
    for (column in colnames(colData(result))) {
        if (is.factor(colData(result)[[column]])) {
            colData(result)[[column]] = as.character(colData(result)[[column]])
        }
    }
    result
}

datasets = list(
    baron = function() {
        sce = BaronPancreasData()
        to_h5ad_input(sce, rownames(sce))
    },
    muraro = function() {
        sce = MuraroPancreasData()
        to_h5ad_input(sce, rowData(sce)$symbol)
    },
    segerstolpe = function() {
        sce = SegerstolpePancreasData()
        to_h5ad_input(sce, rownames(sce))
    }
)

for (name in names(datasets)) {
    sce = datasets[[name]]()
    filename = file.path(output_dir, paste0(name, ".h5ad"))
    message("Writing ", filename, " (", nrow(sce), " genes x ", ncol(sce), " cells)")
    writeH5AD(sce, filename, X_name = "counts", compression = "gzip")
}
