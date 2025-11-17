# --- Paths -------------------------------------------------------------------
PROJ_DIR  <- "/Users/farhadsadat/thesis-benchmark"
S16_DIR   <- file.path(PROJ_DIR, "mendes16s")
OUT_DIR   <- file.path(S16_DIR, "tax4fun2_out")
MAP_CSV   <- file.path(PROJ_DIR, "analysis", "sample_map.csv")

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
setnames(smap, tolower(names(smap)))

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
  if (id_col == "shotgun") {
    dts <- dts[shotgun %in% TARGET_IDS]
  } else {
    keep_err <- smap[shotgun_id %in% TARGET_IDS, unique(err)]
    dts <- dts[sample %in% keep_err]
  }
  if (!nrow(dts)) return(NULL)
  
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
fwrite(group_stats_subset, file.path(OUT_DIR, "group_stats_subset_27ids.csv"))
message("Saved → group_stats_subset_27ids.csv")

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
  ggsave(file.path(OUT_DIR, sprintf("subset27_%s_group_means.png", tolower(met))),
         p, width = 7.5, height = 4.6, dpi = 220)
}

message("Done.
- group_stats_subset_27ids.csv
- subset27_spearman_group_means.png
- subset27_jaccard_group_means.png
saved under mendes16s/tax4fun2_out/")
