#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(tools)
})

## =========================
## USER SETTINGS
## =========================
proj_dir     <- "/Users/farhadsadat/thesis-benchmark"
p2_file      <- file.path(proj_dir, "picrust2", "picrust2_out", "KO_metagenome_out", "pred_metagenome_unstrat.tsv.gz")
shotgun_file <- file.path(proj_dir, "shotgun", "rhizo_wgs_p.txt")
sample_map   <- file.path(proj_dir, "analysis", "sample_map.csv")
out_dir      <- file.path(proj_dir, "analysis", "results_colwise_R")
normalize_to_rel <- TRUE
drop_err <- "ERR1456820"

fig_dir      <- file.path(out_dir, "figures_scatter")
metrics_csv  <- file.path(out_dir, "colwise_metrics_generated.csv")
group_csv    <- file.path(out_dir, "group_stats.csv")
latex_tex    <- file.path(out_dir, "chapter5_tables.tex")

dir.create(out_dir,  showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir,  showWarnings = FALSE, recursive = TRUE)

## =========================
## HELPERS
## =========================
clean_colnames <- function(nms) gsub('^"|"$', '', trimws(nms))
fread_smart <- function(path, ...) {
  if (grepl("\\.gz$", path, ignore.case = TRUE)) data.table::fread(cmd = paste("gzip -dc", shQuote(path)), ...)
  else data.table::fread(path, ...)
}
normalize_ko <- function(x) sub("^(ko:|KO:)", "", as.character(x))
is_ko_col <- function(v) { v <- as.character(v); v <- v[is.na(suppressWarnings(as.numeric(v)))]; any(grepl("^(ko:)?K\\d{5}$", head(v[!is.na(v) & nzchar(v)], 300))) }
coerce_numeric_cols <- function(DT, skip = "KO") {
  cols <- setdiff(names(DT), skip)
  if (!length(cols)) return(invisible(DT))
  DT[, (cols) := lapply(.SD, function(z) {
    z <- trimws(as.character(z)); z[z %in% c("", "NA", "NaN", "nan")] <- NA_character_; as.numeric(gsub(",", ".", z))
  }), .SDcols = cols]
  invisible(DT)
}
dedupe_rows_cols <- function(DT, keycol = "KO", label = "Table") {
  setkeyv(DT, keycol)
  if (any(duplicated(DT[[keycol]]))) {
    num <- setdiff(names(DT), keycol)
    DT <- DT[, lapply(.SD, sum, na.rm = TRUE), by = keycol, .SDcols = num]
  }
  if (any(duplicated(names(DT)))) DT <- DT[, !duplicated(names(DT)), with = FALSE]
  DT
}
rel_abund <- function(M) {
  num <- setdiff(names(M), "KO")
  cs  <- colSums(M[, ..num], na.rm = TRUE); cs[cs == 0] <- NA_real_
  M[, (num) := lapply(num, function(cn) M[[cn]] / cs[cn])]
  M[, (num) := lapply(.SD, function(z) { z[!is.finite(z)] <- 0; z }), .SDcols = num]
  M
}
jaccard_fun <- function(x, y, th = 0) { xb <- x > th; yb <- y > th; u <- sum(xb | yb); if (u == 0) return(NA_real_); sum(xb & yb) / u }
safe_spearman <- function(x, y) { out <- suppressWarnings(cor(x, y, method = "spearman")); if (!is.finite(out)) NA_real_ else out }
pick_ko_column <- function(DT) {
  cand <- c("function","KO","ko","kegg_orthology","KEGG_Orthology")
  have <- cand[cand %in% names(DT)]
  if (length(have)) return(have[1])
  if (is_ko_col(DT[[1]])) return(names(DT)[1])
  if (ncol(DT) >= 2 && is_ko_col(DT[[2]])) return(names(DT)[2])
  NA_character_
}

## Paper-style scatter helper (matches your Tax4Fun2 look)
make_scatter_like_paper <- function(x_rel, y_rel, sg, err, method_label, out_png) {
  spearman <- safe_spearman(x_rel, y_rel)
  jaccard  <- jaccard_fun(x_rel, y_rel, th = 0)
  eps <- 1e-9
  df <- data.frame(x = log10(pmax(x_rel, 0) + eps), y = log10(pmax(y_rel, 0) + eps))
  p <- ggplot(df, aes(x = x, y = y)) +
    geom_point(alpha = 0.55, size = 1.6) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, alpha = 0.5) +
    labs(
      title    = sprintf("%s vs shotgun (%s / %s)", method_label, sg, err),
      subtitle = sprintf("Spearman = %.3f   |   Jaccard = %.3f", spearman, jaccard),
      x = "Shotgun KO rel. abundance (log10)",
      y = sprintf("%s KO rel. abundance (log10)", method_label)
    ) +
    theme_bw(base_size = 14) +
    theme(
      plot.title   = element_text(face = "bold", size = 20),
      plot.subtitle= element_text(size = 14, margin = margin(b = 8)),
      panel.grid.minor = element_blank()
    )
  ggsave(out_png, p, width = 7, height = 4.8, dpi = 300)
  list(spearman = spearman, jaccard = jaccard)
}

## Loaders
load_picrust <- function(path) {
  cat("Reading PICRUSt2 KO table:", path, "\n")
  X <- fread_smart(path, sep = "\t", header = TRUE, quote = "", fill = TRUE,
                   na.strings = c("NA","NaN","","nan"), check.names = FALSE)
  setnames(X, clean_colnames(names(X)))
  ko_col <- pick_ko_column(X)
  if (is.na(ko_col)) { cat("\n--- PICRUSt2 column names ---\n"); print(names(X)); stop("Couldn't identify KO column in PICRUSt2 table.") }
  setnames(X, ko_col, "KO")
  X[, KO := normalize_ko(KO)]
  coerce_numeric_cols(X, skip = "KO")
  X <- dedupe_rows_cols(X, keycol = "KO", label = "PICRUSt2")
  setkey(X, KO); X
}
load_shotgun <- function(path) {
  cat("Reading shotgun KO table:", path, "\n")
  W <- fread_smart(path, sep = "\t", header = TRUE, quote = "", fill = TRUE,
                   na.strings = c("NA","NaN","","nan"), check.names = FALSE)
  setnames(W, clean_colnames(names(W)))
  setnames(W, 1, "KO")  # first column is KO IDs in your file
  W[, KO := normalize_ko(KO)]
  coerce_numeric_cols(W, skip = "KO")
  W <- dedupe_rows_cols(W, keycol = "KO", label = "Shotgun")
  setkey(W, KO); W
}
load_map <- function(path) {
  M <- fread(path)
  setnames(M, tolower(names(M)))
  if (!all(c("err","shotgun_id") %in% names(M))) stop("sample_map.csv must have columns: ERR, shotgun_id (any case)")
  M[, ERR := as.character(err)]; M[, shotgun_id := as.character(shotgun_id)]
  grp <- function(s) { s <- tolower(s); if (startsWith(s, "bulk")) "Bulk" else if (startsWith(s, "rag")) "RAG" else if (startsWith(s, "rtp")) "RTP" else "Other" }
  M[, group := vapply(shotgun_id, grp, character(1))]
  unique(M[, .(ERR, shotgun_id, group)])
}

## =========================
## MAIN
## =========================
PIC <- load_picrust(p2_file)
WGS <- load_shotgun(shotgun_file)
MAP <- load_map(sample_map)[ERR != drop_err]

pairs <- MAP[ERR %in% names(PIC) & shotgun_id %in% names(WGS)]
if (nrow(pairs) == 0) stop("No matched samples between PICRUSt2 and WGS using sample_map.")

if (normalize_to_rel) { PIC <- rel_abund(PIC); WGS <- rel_abund(WGS) }

kos_union <- union(PIC$KO, WGS$KO)
setkey(PIC, KO); setkey(WGS, KO)
fetch_vec <- function(DT, col, kos) { v <- DT[J(kos)][[col]]; v[is.na(v)] <- 0; v }

rows <- vector("list", nrow(pairs))
for (i in seq_len(nrow(pairs))) {
  err <- pairs$ERR[i]; sg <- pairs$shotgun_id[i]; grp <- pairs$group[i]
  x <- fetch_vec(WGS, sg, kos_union)  # shotgun
  y <- fetch_vec(PIC, err, kos_union) # PICRUSt2
  out_png <- file.path(fig_dir, sprintf("scatter_%s__%s.png", err, sg))
  m <- make_scatter_like_paper(x_rel = x, y_rel = y, sg = sg, err = err, method_label = "PICRUSt2", out_png = out_png)
  rows[[i]] <- data.table(
    ERR = err, shotgun_id = sg, group = grp,
    spearman = m$spearman, jaccard = m$jaccard,
    nonzero_picrust = sum(y > 0), nonzero_shotgun = sum(x > 0),
    common_kos = length(intersect(PIC$KO, WGS$KO)),
    scatter_path = out_png
  )
  cat("Saved scatter:", out_png, "\n")
}
RES <- rbindlist(rows)[order(group, ERR)]
fwrite(RES, metrics_csv)
cat("Per-sample metrics:", metrics_csv, "\n")

## SAFE agg_stats (avoid name clashes with functions/objects)
agg_stats <- function(dt) {
  s <- dt[["spearman"]]
  j <- dt[["jaccard"]]
  data.table(
    n_samples       = nrow(dt),
    spearman_min    = suppressWarnings(min(s, na.rm = TRUE)),
    spearman_median = suppressWarnings(median(s, na.rm = TRUE)),
    spearman_mean   = suppressWarnings(mean(s, na.rm = TRUE)),
    spearman_max    = suppressWarnings(max(s, na.rm = TRUE)),
    jaccard_min     = suppressWarnings(min(j, na.rm = TRUE)),
    jaccard_median  = suppressWarnings(median(j, na.rm = TRUE)),
    jaccard_mean    = suppressWarnings(mean(j, na.rm = TRUE)),
    jaccard_max     = suppressWarnings(max(j, na.rm = TRUE))
  )
}
GRP  <- RES[, agg_stats(.SD), by = group]
ALL  <- agg_stats(RES)[, group := "ALL"]
SUMM <- rbindlist(list(GRP, ALL), use.names = TRUE)
fwrite(SUMM, group_csv)
cat("Group stats:", group_csv, "\n")

## LaTeX tables (verbatim blocks)
to_tex <- function(RES, SUMM) {
  cap <- function(x) paste0("\\begin{table}[ht]\n\\centering\n", x, "\n\\end{table}\n")
  T1 <- copy(RES); T1[, `:=`(spearman = round(spearman, 3), jaccard = round(jaccard, 3))]
  body1 <- capture.output(print(as.data.frame(T1[, .(ERR, shotgun_id, group, spearman, jaccard, nonzero_picrust, nonzero_shotgun, common_kos)]), row.names = FALSE))
  txt1  <- paste0("\\caption{Per-sample similarity between PICRUSt2-inferred and shotgun KO profiles (column-wise).}\n",
                  "\\label{tab:per-sample-metrics}\n\\begin{verbatim}\n", paste(body1, collapse = "\n"), "\n\\end{verbatim}")
  G <- copy(SUMM); for (cc in setdiff(names(G), "group")) set(G, j = cc, value = round(G[[cc]], 3))
  body2 <- capture.output(print(as.data.frame(G), row.names = FALSE))
  txt2  <- paste0("\\caption{Summary statistics of similarity metrics by sample family (Bulk, RAG, RTP).}\n",
                  "\\label{tab:group-stats}\n\\begin{verbatim}\n", paste(body2, collapse = "\n"), "\n\\end{verbatim}")
  paste0(cap(txt1), "\n\n", cap(txt2))
}
writeLines(to_tex(RES, SUMM), con = latex_tex)
cat("LaTeX tables:", latex_tex, "\n")

cat("Done. Scatter plots dir:", fig_dir, "\n")
