#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

## ---- paths (edit if needed) ----
proj_dir <- "/Users/farhadsadat/thesis-benchmark"

# where you are now organizing things
out_dir  <- file.path(proj_dir, "analysis", "results_colwise_New")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# per-pair metrics produced by your new scatter scripts
# PICRUSt2: analysis/results_colwise_New/figures_scatter_PICRUSt2/figures_scatter_metrics_picrust2.csv
# Tax4Fun2: analysis/results_colwise_New/figures_scatter_Taxfun2/figures_scatter_metrics_tax4fun2.csv
p2_metrics  <- file.path(out_dir, "figures_scatter_PICRUSt2", "figures_scatter_metrics_picrust2.csv")
tax_metrics <- file.path(out_dir, "figures_scatter_Taxfun2",  "figures_scatter_metrics_tax4fun2.csv")

# use professor’s corrected mapping (run_accession + sample_title)
sample_map  <- file.path(proj_dir, "analysis", "correct_map_samples.tsv")

## ---- helpers ----
load_map <- function(path) {
  M <- fread(path)
  setnames(M, tolower(names(M)))
  
  # case 1: professor's file: run_accession + sample_title
  if (all(c("run_accession", "sample_title") %in% names(M))) {
    setnames(M,
             old = c("run_accession", "sample_title"),
             new = c("err", "shotgun_id"))
  }
  
  if (!all(c("err","shotgun_id") %in% names(M))) {
    stop("sample map must have columns (run_accession, sample_title) or (ERR, shotgun_id).")
  }
  
  M[, err        := as.character(err)]
  M[, shotgun_id := as.character(shotgun_id)]
  
  grp <- function(s) {
    s <- tolower(s)
    if (startsWith(s, "bulk")) "Bulk"
    else if (startsWith(s, "rag")) "RAG"
    else if (startsWith(s, "rtp")) "RTP"
    else "Other"
  }
  M[, group := vapply(shotgun_id, grp, character(1))]
  unique(M[, .(err, shotgun_id, group)])
}

safe_read <- function(p) {
  if (file.exists(p)) fread(p) else NULL
}

## ---- load inputs ----
MAP <- load_map(sample_map)

TAX <- safe_read(tax_metrics)
if (!is.null(TAX)) {
  setnames(TAX, tolower(names(TAX)))
  # expect columns: err, shotgun_id, spearman, jaccard, ...
  # join group via shotgun_id
  TAX <- merge(TAX, MAP[, .(shotgun_id, group)], by = "shotgun_id", all.x = TRUE)
  TAX[, method := "Tax4Fun2"]
}

P2 <- safe_read(p2_metrics)
if (!is.null(P2)) {
  setnames(P2, tolower(names(P2)))
  # if group missing, add via map
  if (!"group" %in% names(P2)) {
    P2 <- merge(P2, MAP[, .(shotgun_id, group)], by = "shotgun_id", all.x = TRUE)
  }
  P2[, method := "PICRUSt2"]
}

if (is.null(TAX) && is.null(P2)) {
  stop("No metrics CSVs found. Run the scatter scripts first.")
}

## ---- combine & clean ----
ALL <- rbindlist(list(TAX, P2), use.names = TRUE, fill = TRUE)

req <- c("method","group","spearman","jaccard")
if (!all(req %in% names(ALL))) {
  stop("Missing required columns in metrics files: ",
       paste(setdiff(req, names(ALL)), collapse = ", "))
}

# ensure numeric / factors
ALL[, spearman := as.numeric(spearman)]
ALL[, jaccard  := as.numeric(jaccard)]
ALL[, group    := factor(group, levels = c("Bulk","RAG","RTP","Other"))]
ALL[, method   := factor(method, levels = c("Tax4Fun2","PICRUSt2"))]

# drop rows with missing info
ALL <- ALL[!is.na(group) & !is.na(method) &
             is.finite(spearman) & is.finite(jaccard)]

# save merged table
merged_csv <- file.path(out_dir, "merged_metrics_with_groups.csv")
fwrite(ALL, merged_csv)

## ---- plotting ----
# 1) Spearman boxplot by group & method
p_spear <- ggplot(ALL, aes(x = group, y = spearman, fill = method)) +
  geom_boxplot(outlier.alpha = 0.25, width = 0.7,
               position = position_dodge(width = 0.8)) +
  geom_jitter(aes(color = method),
              position = position_jitterdodge(jitter.width = 0.15,
                                              dodge.width = 0.8),
              size = 1.2, alpha = 0.35, show.legend = FALSE) +
  labs(title = "Spearman by Group (Benchmarking)",
       x = "Group", y = "Spearman") +
  theme_bw(base_size = 12) +
  theme(panel.grid.minor = element_blank())

ggsave(file.path(out_dir, "box_spearman_by_group.png"),
       p_spear, width = 7.5, height = 4.8, dpi = 300)

# 2) Jaccard boxplot by group & method
p_jacc <- ggplot(ALL, aes(x = group, y = jaccard, fill = method)) +
  geom_boxplot(outlier.alpha = 0.25, width = 0.7,
               position = position_dodge(width = 0.8)) +
  geom_jitter(aes(color = method),
              position = position_jitterdodge(jitter.width = 0.15,
                                              dodge.width = 0.8),
              size = 1.2, alpha = 0.35, show.legend = FALSE) +
  labs(title = "Jaccard by Group (Benchmarking)",
       x = "Group", y = "Jaccard") +
  theme_bw(base_size = 12) +
  theme(panel.grid.minor = element_blank())

ggsave(file.path(out_dir, "box_jaccard_by_group.png"),
       p_jacc, width = 7.5, height = 4.8, dpi = 300)

# 3) Facet: metric x method
ALL_long <- melt(ALL,
                 id.vars = c("method","group"),
                 measure.vars = c("spearman","jaccard"),
                 variable.name = "metric", value.name = "value")

p_facet <- ggplot(ALL_long, aes(x = group, y = value, fill = group)) +
  geom_boxplot(outlier.alpha = 0.25, width = 0.7) +
  geom_jitter(width = 0.12, size = 0.9, alpha = 0.25, show.legend = FALSE) +
  facet_grid(metric ~ method, scales = "free_y") +
  labs(title = "Benchmarking by Group: Spearman & Jaccard",
       x = "Group", y = "Value") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none",
        panel.grid.minor = element_blank())

ggsave(file.path(out_dir, "box_facet_method_metric.png"),
       p_facet, width = 9, height = 6, dpi = 300)

cat("Done.\n",
    "Merged table: ", merged_csv, "\n",
    "Spearman box: ", file.path(out_dir, "box_spearman_by_group.png"), "\n",
    "Jaccard box:  ", file.path(out_dir, "box_jaccard_by_group.png"), "\n",
    "Facet plot:   ", file.path(out_dir, "box_facet_method_metric.png"), "\n",
    sep = "")
