#!/usr/bin/env Rscript
#
# Conversion script for Seurat objects (.rds) to Phoenix input format.
#
# Exports expression matrix, cell-type annotations, pseudotime values,
# and dimensionality reduction coordinates as CSV files compatible with Phoenix.
#
# Usage:
#   Rscript convert.R \
#       --input my_data.rds \
#       --output phoenix_input/ \
#       --cell_type_key cell_type \
#       --reduction_key umap
#
# Requirements: Seurat, optparse
#   install.packages(c("Seurat", "optparse"))

suppressPackageStartupMessages({
  library(Seurat)
  library(optparse)
  library(Matrix)
})

option_list <- list(
  make_option("--input", type = "character",
              help = "Path to the Seurat .rds file"),
  make_option("--output", type = "character",
              help = "Output directory for the CSV files"),
  make_option("--assay", type = "character", default = NULL,
              help = "Seurat assay to use (e.g. 'RNA','SCT'). Default: DefaultAssay"),
  make_option("--slot", type = "character", default = "counts",
              help = "Assay slot for expression: 'counts' (raw, default) or 'data' (normalized)"),
  make_option("--cell_type_key", type = "character", default = NULL,
              help = "Column name in metadata for cell-type annotations (e.g. 'cell_type', 'seurat_clusters')"),
  make_option("--pseudotime_key", type = "character", default = NULL,
              help = "Comma-separated column name(s) in metadata for pseudotime trajectories"),
  make_option("--reduction_key", type = "character", default = NULL,
              help = "Reduction name (e.g. 'umap', 'tsne', 'pca'). Default: auto-detect")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$input) || is.null(opt$output)) {
  stop("Both --input and --output are required. Run with --help for usage.")
}

# Create output directory
dir.create(opt$output, showWarnings = FALSE, recursive = TRUE)

# Load Seurat object
cat(sprintf("Loading %s...\n", opt$input))
obj <- readRDS(opt$input)
cat(sprintf("Loaded Seurat object: %d cells x %d features\n", ncol(obj), nrow(obj)))

# Set assay
if (!is.null(opt$assay)) {
  DefaultAssay(obj) <- opt$assay
}
cat(sprintf("Using assay: %s\n", DefaultAssay(obj)))

# --- Expression ---
cat("Extracting expression matrix...\n")
if (opt$slot == "counts") {
  expr_matrix <- GetAssayData(obj, slot = "counts")
} else {
  expr_matrix <- GetAssayData(obj, slot = "data")
}

if (inherits(expr_matrix, "sparseMatrix")) {
  expr_matrix <- as.matrix(expr_matrix)
}

# Transpose to cells x genes (Phoenix expects rows = cells, columns = genes)
expr_df <- as.data.frame(t(expr_matrix))
write.csv(expr_df, file.path(opt$output, "expression.csv"), row.names = TRUE)
cat(sprintf("  -> Saved expression.csv (%d cells x %d genes)\n", nrow(expr_df), ncol(expr_df)))

# --- Cell types ---
if (!is.null(opt$cell_type_key)) {
  if (!(opt$cell_type_key %in% colnames(obj@meta.data))) {
    stop(sprintf(
      "Cell-type key '%s' not found in metadata. Available columns: %s",
      opt$cell_type_key,
      paste(colnames(obj@meta.data), collapse = ", ")
    ))
  }
  ct <- data.frame(
    cell_type = obj@meta.data[[opt$cell_type_key]],
    row.names = colnames(obj)
  )
  write.csv(ct, file.path(opt$output, "cell_types.csv"), row.names = TRUE)
  cat(sprintf("  -> Saved cell_types.csv (%d unique types)\n", length(unique(ct$cell_type))))
}

# --- Pseudotime ---
if (!is.null(opt$pseudotime_key)) {
  pt_keys <- trimws(unlist(strsplit(opt$pseudotime_key, ",")))
  missing <- pt_keys[!(pt_keys %in% colnames(obj@meta.data))]
  if (length(missing) > 0) {
    stop(sprintf(
      "Pseudotime key(s) not found in metadata: %s. Available columns: %s",
      paste(missing, collapse = ", "),
      paste(colnames(obj@meta.data), collapse = ", ")
    ))
  }
  pt <- obj@meta.data[, pt_keys, drop = FALSE]
  write.csv(pt, file.path(opt$output, "pseudotime.csv"), row.names = TRUE)
  cat(sprintf("  -> Saved pseudotime.csv (%d trajectories)\n", length(pt_keys)))
}

# --- Reduction ---
reduction_name <- opt$reduction_key
if (is.null(reduction_name)) {
  # Auto-detect: prefer UMAP > t-SNE > PCA
  available <- Reductions(obj)
  for (candidate in c("umap", "tsne", "pca")) {
    if (candidate %in% available) {
      reduction_name <- candidate
      break
    }
  }
}

if (!is.null(reduction_name)) {
  if (!(reduction_name %in% Reductions(obj))) {
    stop(sprintf(
      "Reduction '%s' not found. Available reductions: %s",
      reduction_name,
      paste(Reductions(obj), collapse = ", ")
    ))
  }
  coords <- Embeddings(obj, reduction = reduction_name)[, 1:2]
  colnames(coords) <- paste0(reduction_name, "_", 1:2)
  write.csv(coords, file.path(opt$output, "reduction.csv"), row.names = TRUE)
  cat(sprintf("  -> Saved reduction.csv (from '%s')\n", reduction_name))
} else {
  cat("No dimensionality reduction found — skipping.\n")
  cat("Phoenix will compute one automatically using --reduction.\n")
}

# --- Summary ---
cat(sprintf("\nConversion complete. Output files in: %s/\n", opt$output))
cat("You can now run Phoenix:\n")
cmd <- sprintf("  python run.py \\\n    --expression %s/expression.csv", opt$output)
if (!is.null(opt$cell_type_key)) {
  cmd <- paste0(cmd, sprintf(" \\\n    --cell_types %s/cell_types.csv", opt$output))
}
if (!is.null(opt$pseudotime_key)) {
  cmd <- paste0(cmd, sprintf(" \\\n    --pseudotime %s/pseudotime.csv", opt$output))
}
if (!is.null(reduction_name)) {
  cmd <- paste0(cmd, sprintf(" \\\n    --reduction %s/reduction.csv", opt$output))
}
cmd <- paste0(cmd, " \\\n    --output <output_dir>")
cat(cmd, "\n")
