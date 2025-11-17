#!/usr/bin/env Rscript

## -------- USER PATHS (edit these two only) --------
PROJ_DIR <- "/Users/farhadsadat/thesis-benchmark/mendes16s"           # has export-table/ and export-seqs/
REF_DIR  <- "/Users/farhadsadat/tax4fun2_ref/Tax4Fun2_ReferenceData_v2"  # Tax4Fun2_ReferenceData_v2

## -------- Optional paths (these should already exist) --------
shotgun_file <- file.path(dirname(PROJ_DIR), "shotgun", "rhizo_wgs_p.txt")
sample_map   <- file.path(dirname(PROJ_DIR), "analysis", "sample_map.csv")

## -------- Libraries --------
suppressPackageStartupMessages({
  library(biomformat)
  library(data.table)
  library(ggplot2)
  library(Matrix)
})

if (!requireNamespace("Tax4Fun2", quietly = TRUE)) {
  stop(
    "The 'Tax4Fun2' package is not installed in this R.\n",
    "Please install it first and then re-run this script."
  )
}

## -------- Make sure BLAST from Tax4Fun2 bundle is usable --------
blast_bin <- file.path(REF_DIR, "blast_bin")
if (dir.exists(blast_bin)) {
  Sys.setenv(PATH = paste(blast_bin, Sys.getenv("PATH"), sep = .Platform$path.sep))
}

## -------- Derived paths --------
biom_file  <- file.path(PROJ_DIR, "export-table", "feature-table.biom")
fasta_file <- file.path(PROJ_DIR, "export-seqs",  "dna-sequences.fasta")
out_dir    <- file.path(PROJ_DIR, "tax4fun2_out")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

t4f2_tmp   <- file.path(out_dir, "Ref100NR")   # temp folder for Tax4Fun2
dir.create(t4f2_tmp, showWarnings = FALSE, recursive = TRUE)

## -------- 0) Load ASV table from BIOM and write Tax4Fun2 OTU table --------
cat("Reading BIOM: ", biom_file, "\n")
bm <- read_biom(biom_file)
otu_mat <- as(biom_data(bm), "dgCMatrix")   # sparse (features x samples)
otu <- as.matrix(otu_mat)
rm(otu_mat); gc()

# Ensure ASV IDs are rownames
if (is.null(rownames(otu)) && !is.null(dimnames(otu)[[1]])) {
  rownames(otu) <- dimnames(otu)[[1]]
}
cat("ASV table:", nrow(otu), "ASVs x", ncol(otu), "samples\n")

# Write Tax4Fun2-style OTU table (first column 'OTU', then samples)
otu_table_txt <- file.path(out_dir, "otu_table_tax4fun2.txt")
otu_df <- data.frame(OTU = rownames(otu), otu, check.names = FALSE)
data.table::fwrite(otu_df, otu_table_txt, sep = "\t", quote = FALSE)
cat("Wrote OTU table for Tax4Fun2:", otu_table_txt, "\n")

## -------- 1) Run Tax4Fun2: reference BLAST --------
cat("Running Tax4Fun2::runRefBlast() ...\n")
Tax4Fun2::runRefBlast(
  path_to_otus           = fasta_file,
  path_to_reference_data = REF_DIR,
  path_to_temp_folder    = t4f2_tmp,
  database_mode          = "Ref100NR",  # use Ref100NR bundle
  use_force              = TRUE,
  num_threads            = 4
)

## -------- 2) Run Tax4Fun2: functional prediction --------
cat("Running Tax4Fun2::makeFunctionalPrediction() ...\n")
Tax4Fun2::makeFunctionalPrediction(
  path_to_otu_table         = otu_table_txt,
  path_to_reference_data    = REF_DIR,
  path_to_temp_folder       = t4f2_tmp,
  database_mode             = "Ref100NR",
  normalize_by_copy_number  = TRUE,
  min_identity_to_reference = 0.97,
  normalize_pathways        = FALSE
)

## Tax4Fun2 writes 'functional_prediction.txt' in the temp folder (KOs x samples, relative)
fp_candidates <- c(
  file.path(t4f2_tmp, "functional_prediction.txt"),
  "functional_prediction.txt"
)
fp_file <- fp_candidates[file.exists(fp_candidates)][1]

if (is.na(fp_file)) {
  stop(
    "Could not find 'functional_prediction.txt' after makeFunctionalPrediction().\n",
    "Look inside ", t4f2_tmp, " to see which file was created and adjust the script."
  )
}

cat("Reading Tax4Fun2 KO predictions from:", fp_file, "\n")
fp <- data.table::fread(fp_file)

# Rename first column to 'KO' for consistency
data.table::setnames(fp, 1, "KO")

# Save as our benchmark input
out_pred <- file.path(out_dir, "tax4fun2_KO_rel.tsv")
data.table::fwrite(fp, out_pred, sep = "\t", quote = FALSE)
cat("Tax4Fun2 KO predictions saved to:", out_pred, "\n")

## -------- 3) Benchmark vs shotgun KO table (if available) --------
if (file.exists(shotgun_file) && file.exists(sample_map)) {
  cat("Benchmarking against shotgun:", shotgun_file, "\n")
  
  map <- data.table::fread(sample_map)
  data.table::setnames(map, tolower(names(map)))
  if (!all(c("err","shotgun_id") %in% names(map))) {
    stop("sample_map must have columns: ERR, shotgun_id (any case).")
  }
  
  ## ----- Load shotgun KO table -----
  wgs <- data.table::fread(shotgun_file)
  # First column is KO ID (or index+KO); force its name to "KO"
  data.table::setnames(wgs, 1, "KO")
  wgs[, KO := sub("^(ko:|KO:)", "", KO)]
  data.table::setkey(wgs, KO)
  
  # Force all non-KO columns to numeric
  wgs_num_cols <- setdiff(names(wgs), "KO")
  wgs[, (wgs_num_cols) := lapply(.SD, as.numeric), .SDcols = wgs_num_cols]
  
  ## ----- Load Tax4Fun2 predictions -----
  tax <- data.table::fread(out_pred)
  data.table::setkey(tax, KO)
  
  # Force all non-KO columns to numeric as well
  tax_num_cols <- setdiff(names(tax), "KO")
  tax[, (tax_num_cols) := lapply(.SD, as.numeric), .SDcols = tax_num_cols]
  
  ## ----- Match KOs -----
  common_KO <- intersect(tax$KO, wgs$KO)
  cat("Common KO between Tax4Fun2 and shotgun:", length(common_KO), "\n")
  
  tax2 <- as.matrix(tax[data.table::J(common_KO)][ , -1])
  rownames(tax2) <- common_KO
  wgs2 <- as.matrix(wgs[data.table::J(common_KO)][ , -1])
  rownames(wgs2) <- common_KO
  
  ## ----- Safety: convert to relative abundance (ignore NAs) -----
  tax2 <- sweep(tax2, 2, colSums(tax2, na.rm = TRUE), "/"); tax2[!is.finite(tax2)] <- 0
  wgs2 <- sweep(wgs2, 2, colSums(wgs2, na.rm = TRUE), "/"); wgs2[!is.finite(wgs2)] <- 0
  
  ## ----- Pair samples based on sample_map (ERR = 16S ID, shotgun_id = WGS column) -----
  pair_list <- list()
  for (i in seq_len(nrow(map))) {
    ERR <- as.character(map$err[i])
    SG  <- as.character(map$shotgun_id[i])
    if (ERR %in% colnames(tax2) && SG %in% colnames(wgs2)) {
      pair_list[[length(pair_list) + 1]] <- list(err = ERR, sg = SG)
    }
  }
  if (!length(pair_list)) stop("No matched samples present in Tax4Fun2 vs shotgun matrices.")
  
  ## ----- Metrics -----
  spearman <- function(x, y) suppressWarnings(cor(x, y, method = "spearman"))
  jaccard  <- function(x, y, t = 0) {
    xb <- x > t; yb <- y > t
    u <- sum(xb | yb); if (u == 0) return(NA_real_)
    sum(xb & yb) / u
  }
  
  rows <- lapply(pair_list, function(p) {
    x <- wgs2[, p$sg]
    y <- tax2[, p$err]
    list(
      ERR         = p$err,
      shotgun_id  = p$sg,
      spearman    = unname(spearman(x, y)),
      jaccard     = unname(jaccard(x, y))
    )
  })
  res <- data.table::rbindlist(rows)
  metrics_csv <- file.path(out_dir, "tax4fun2_vs_shotgun_metrics.csv")
  data.table::fwrite(res, metrics_csv)
  cat("Benchmark CSV:", metrics_csv, "\n")
  
  ## ----- Plots -----
  p1 <- ggplot2::ggplot(res, ggplot2::aes(x = as.numeric(spearman))) +
    ggplot2::geom_histogram(bins = 20) +
    ggplot2::labs(
      title = "Tax4Fun2 vs Shotgun — Spearman",
      x     = "Spearman",
      y     = "Count"
    )
  ggplot2::ggsave(file.path(out_dir, "tax4fun2_spearman_hist.png"), p1,
                  width = 6, height = 4, dpi = 200)
  
  p2 <- ggplot2::ggplot(res, ggplot2::aes(x = as.numeric(jaccard))) +
    ggplot2::geom_histogram(bins = 20) +
    ggplot2::labs(
      title = "Tax4Fun2 vs Shotgun — Jaccard",
      x     = "Jaccard",
      y     = "Count"
    )
  ggplot2::ggsave(file.path(out_dir, "tax4fun2_jaccard_hist.png"), p2,
                  width = 6, height = 4, dpi = 200)
  
  cat("Plots saved to tax4fun2_spearman_hist.png and tax4fun2_jaccard_hist.png\n")
} else {
  cat("Skip benchmark: shotgun_file or sample_map not found.\n")
}
