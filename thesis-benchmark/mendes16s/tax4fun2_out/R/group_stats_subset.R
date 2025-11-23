#!/usr/bin/env Rscript

# --- Paths -------------------------------------------------------------------
PROJ_DIR  <- "/Users/farhadsadat/thesis-benchmark"
S16_DIR   <- file.path(PROJ_DIR, "mendes16s")
OUT_DIR   <- file.path(S16_DIR, "tax4fun2_out")

# use professor's corrected mapping file
MAP_CSV   <- file.path(PROJ_DIR, "analysis", "correct_map_samples.tsv")

# --- Packages ----------------------------------------------------------------
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

# --- Load results ------------------------------------------------------------
tax_csv  <- file.path(OUT_DIR, "colwise_tax4fun2_vs_shotgun.csv")
pt_csv   <- file.path(OUT_DIR, "colwise_picrust2_vs_tax4fun2.csv")
ps_csv   <- file.path(OUT_DIR, "colwise_picrust2_vs_shotgun.csv")

stopifnot(file.exists(tax_csv), file.exists(pt_csv), file.exists(ps_csv), file.exists(MAP_CSV))

col_tax_wgs     <- fread(tax_csv)
col_pt          <- fread(pt_csv)
col_pic_vs_wgs  <- fread(ps_csv)
smap            <- fread(MAP_CSV)

# normalise mapping columns
setnames(smap, tolower(names(smap)))

# handle professor's file: run_accession + sample_title
if (all(c("run_accession", "sample_title") %in% names(smap))) {
  setnames(smap,
           old = c("run_accession", "sample_title"),
           new = c("err", "shotgun_id"))
}

if (!all(c("err", "shotgun_id") %in% names(smap))) {
  stop("Mapping file must have columns (run_accession, sample_title) or (ERR, shotgun_id).")
}

smap[, err        := as.character(err)]
smap[, shotgun_id := as.character(shotgun_id)]

# define group from shotgun_id
grp_fun <- function(s) {
  s <- tolower(s)
  if (startsWith(s, "bulk")) "Bulk"
  else if (startsWith(s, "rag")) "RAG"
  else if (startsWith(s, "rtp")) "RTP"
  else "Other"
}
smap[, group := vapply(shotgun_id, grp_fun, character(1))]
smap <- unique(smap[, .(err, shotgun_id, group)])

# --- standardise column names in results -------------------------------------

# lower-case all names
setnames(col_tax_wgs,     tolower(names(col_tax_wgs)))
setnames(col_pt,          tolower(names(col_pt)))
setnames(col_pic_vs_wgs,  tolower(names(col_pic_vs_wgs)))

# for tax4fun vs shotgun: expect 'shotgun', 'spearman', 'jaccard'
if ("shotgun_id" %in% names(col_tax_wgs) && !"shotgun" %in% names(col_tax_wgs)) {
  setnames(col_tax_wgs, "shotgun_id", "shotgun")
}
if (!"shotgun" %in% names(col_tax_wgs)) {
  stop("colwise_tax4fun2_vs_shotgun.csv must have a 'shotgun' or 'shotgun_id' column.")
}

# for picrust2 vs shotgun
if ("shotgun_id" %in% names(col_pic_vs_wgs) && !"shotgun" %in% names(col_pic_vs_wgs)) {
  setnames(col_pic_vs_wgs, "shotgun_id", "shotgun")
}
if (!"shotgun" %in% names(col_pic_vs_wgs)) {
  stop("colwise_picrust2_vs_shotgun.csv must have a 'shotgun' or 'shotgun_id' column.")
}

# for picrust2 vs tax4fun2: expect 'sample' (ERR)
if ("err" %in% names(col_pt) && !"sample" %in% names(col_pt)) {
  setnames(col_pt, "err", "sample")
}
if (!"sample" %in% names(col_pt)) {
  stop("colwise_picrust2_vs_tax4fun2.csv must have a 'sample' or 'err' column.")
}

# --- attach group to each table ----------------------------------------------

# Tax4Fun2 vs Shotgun: add group via shotgun
if (!"group" %in% names(col_tax_wgs)) {
  col_tax_wgs <- merge(
    col_tax_wgs,
    smap[, .(shotgun = shotgun_id, group)],
    by = "shotgun",
    all.x = TRUE
  )
}

# PICRUSt2 vs Shotgun: add group via shotgun
if (!"group" %in% names(col_pic_vs_wgs)) {
  col_pic_vs_wgs <- merge(
    col_pic_vs_wgs,
    smap[, .(shotgun = shotgun_id, group)],
    by = "shotgun",
    all.x = TRUE
  )
}

# PICRUSt2 vs Tax4Fun2: add group via sample (ERR)
if (!"group" %in% names(col_pt)) {
  col_pt <- merge(
    col_pt,
    smap[, .(sample = err, group)],
    by = "sample",
    all.x = TRUE
  )
}

# --- Define your 27 target shotgun IDs ---------------------------------------
TARGET_IDS <- c(
  "BulkRAG1","BulkRAG2","BulkRAG3",
  "BulkTP1","BulkTP2","BulkTP3",
  "RAG1","RAG10","RAG12","RAG2","RAG3","RAG4","RAG5","RAG6","RAG7","RAG8",
  "RTP1","RTP10","RTP11","RTP12","RTP2","RTP3","RTP4","RTP5","RTP6","RTP7","RTP9"
)

# helper for CI + summary
summarize_metrics <- function(dt, method_label, id_col = c("shotgun","sample")) {
  id_col <- match.arg(id_col)
  dts <- copy(dt)
  
  # restrict to target shotgun IDs
  if (id_col == "shotgun") {
    if (!"shotgun" %in% names(dts)) stop("Expected 'shotgun' column in dt.")
    dts <- dts[shotgun %in% TARGET_IDS]
  } else {
    # sample is ERR; map to those whose shotgun_id in TARGET_IDS
    keep_err <- smap[shotgun_id %in% TARGET_IDS, unique(err)]
    if (!"sample" %in% names(dts)) stop("Expected 'sample' column in dt.")
    dts <- dts[sample %in% keep_err]
  }
  
  if (!nrow(dts)) return(NULL)
  
  # make sure we have group, spearman, jaccard
  if (!all(c("group", "spearman", "jaccard") %in% names(dts))) {
    stop("dt must contain 'group', 'spearman', 'jaccard' columns.")
  }
  
  mean_ci <- function(x){
    x <- x[is.finite(x)]
    n <- length(x); m <- mean(x); s <- sd(x)
    if (n < 2 || is.na(s)) return(c(mean=m, lwr=NA, upr=NA, n=n))
    se <- s/sqrt(n); ci <- qt(0.975, df = n-1) * se
    c(mean=m, lwr=m-ci, upr=m+ci, n=n)
  }
  
  agg_s <- dts[, as.list(unlist(mean_ci(spearman))), by = group]
  agg_s[, `:=`(metric="Spearman", method=method_label)]
  setnames(agg_s, c("mean","lwr","upr","n"), c("mean","ci_lwr","ci_upr","n"))
  
  agg_j <- dts[, as.list(unlist(mean_ci(jaccard))), by = group]
  agg_j[, `:=`(metric="Jaccard", method=method_label)]
  setnames(agg_j, c("mean","lwr","upr","n"), c("mean","ci_lwr","ci_upr","n"))
  
  rbind(agg_s, agg_j, fill = TRUE)
}

# --- Summaries for all three comparisons -------------------------------------
gs_tax  <- summarize_metrics(col_tax_wgs,    "Tax4Fun2 vs Shotgun",    id_col = "shotgun")
gs_pt   <- summarize_metrics(col_pt,         "PICRUSt2 vs Tax4Fun2",   id_col = "sample")
gs_ps   <- summarize_metrics(col_pic_vs_wgs, "PICRUSt2 vs Shotgun",    id_col = "shotgun")

group_stats_subset <- rbindlist(list(gs_tax, gs_ps, gs_pt), fill = TRUE)

out_csv <- file.path(OUT_DIR, "group_stats_subset_27ids.csv")
fwrite(group_stats_subset, out_csv)
message("Saved → ", out_csv)

# --- Quick grouped barplots (with 95% CI) ------------------------------------
for (met in c("Spearman","Jaccard")) {
  p <- ggplot(group_stats_subset[metric == met],
              aes(x = group, y = mean, fill = method)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.65) +
    geom_errorbar(aes(ymin = ci_lwr, ymax = ci_upr),
                  position = position_dodge(width = 0.7), width = 0.15) +
    labs(title = paste(met, "— group means (subset of 27 IDs)"),
         x = NULL, y = met) +
    theme_minimal(base_size = 12) +
    scale_fill_brewer(palette = "Greys") +
    coord_cartesian(ylim = c(0, 1))
  
  out_png <- file.path(OUT_DIR, sprintf("subset27_%s_group_means.png", tolower(met)))
  ggsave(out_png, p, width = 7.5, height = 4.6, dpi = 220)
}

message(
  "Done.\n",
  "- ", out_csv, "\n",
  "- ", file.path(OUT_DIR, "subset27_spearman_group_means.png"), "\n",
  "- ", file.path(OUT_DIR, "subset27_jaccard_group_means.png"), "\n"
)
