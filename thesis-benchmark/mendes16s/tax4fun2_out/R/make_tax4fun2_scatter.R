#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

## -------- paths (adapt if needed) --------
proj_dir     <- "/Users/farhadsadat/thesis-benchmark"
p2_file      <- file.path(proj_dir, "picrust2", "picrust2_out", "KO_metagenome_out", "pred_metagenome_unstrat.tsv.gz")
shotgun_file <- file.path(proj_dir, "shotgun", "rhizo_wgs_p.txt")
sample_map   <- file.path(proj_dir, "analysis", "sample_map.csv")
fig_dir      <- file.path(proj_dir, "picrust2", "picrust2_out", "figures_scatter")
metrics_csv  <- file.path(fig_dir, "figures_scatter_metrics_picrust2.csv")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

## -------- helpers --------
peek_dt <- function(dt, n = 5) {
  cat("\n--- COLUMN NAMES ---\n"); print(names(dt))
  cat("\n--- FIRST ROWS ---\n"); print(utils::head(dt, n)); cat("\n")
}

is_ko_vector <- function(v) {
  v <- as.character(v)
  v <- gsub('^"|\'|"$|\'$', "", v)           # strip quotes inside cells
  v <- v[!is.na(v) & nzchar(v)]
  if (!length(v)) return(FALSE)
  m <- grepl("^(?i:ko:)?K\\d{5}$", v, perl = TRUE)
  sum(m) >= max(20, ceiling(0.05 * length(v)))  # “looks like KO” if many match
}

numericize_cols <- function(DT, cols) {
  if (!length(cols)) return(invisible(DT))
  DT[, (cols) := lapply(.SD, function(z) {
    z <- trimws(as.character(z))
    z <- gsub('^"|\'|"$|\'$', "", z)        # strip quotes
    z[z %in% c("", "NA", "NaN", "nan")] <- NA_character_
    as.numeric(gsub(",", ".", z))
  }), .SDcols = cols]
}

aggregate_dupe_KO <- function(DT) {
  if (any(duplicated(DT$KO))) {
    num <- setdiff(names(DT), "KO")
    DT <- DT[, lapply(.SD, sum, na.rm = TRUE), by = KO, .SDcols = num]
  }
  setkey(DT, KO)
  DT
}

fread_smart <- function(path, ...) {
  if (grepl("\\.gz$", path, ignore.case = TRUE)) {
    data.table::fread(cmd = paste("gzip -dc", shQuote(path)), ...)
  } else {
    data.table::fread(path, ...)
  }
}

## -------- loaders --------
load_picrust2 <- function(path) {
  x <- fread_smart(path, sep = "\t", header = TRUE, quote = "", fill = TRUE,
                   na.strings = c("NA","NaN","","nan"), check.names = FALSE)
  ## 1) KO column name varies; detect it
  if (!"KO" %in% names(x)) {
    # common PICRUSt2 headers
    cand <- c("function","feature","KEGG_Orthology","ko","KO")
    have <- cand[cand %in% names(x)]
    if (length(have)) setnames(x, have[1], "KO")
  }
  # fallback: detect by content
  if (!"KO" %in% names(x)) {
    if (is_ko_vector(x[[1]])) setnames(x, 1, "KO")
    else if (ncol(x) >= 2 && is_ko_vector(x[[2]])) {
      x[, (names(x)[1]) := NULL]       # drop index col
      setnames(x, 1, "KO")
    } else {
      peek_dt(x)
      stop("Couldn't find a KO column in PICRUSt2 table. See peek above and adjust.")
    }
  }
  ## 2) normalize KO ids
  x[, KO := gsub('^"|\'|"$|\'$', "", as.character(KO))]
  x[, KO := sub("^(?i:ko:)", "", KO, perl = TRUE)]
  x <- x[nzchar(KO)]
  ## 3) numeric-ize all non-KO columns
  num <- setdiff(names(x), "KO")
  numericize_cols(x, num)
  aggregate_dupe_KO(x)
}

load_shotgun <- function(path) {
  w <- fread_smart(path, sep = "\t", header = TRUE, quote = "", fill = TRUE,
                   na.strings = c("NA","NaN","","nan"), strip.white = TRUE,
                   check.names = FALSE)
  ## 1) clean header quotes/spaces
  clean_names <- function(nm) {
    nm <- gsub('^"|\'|"$|\'$', "", nm)
    trimws(nm)
  }
  setnames(w, clean_names(names(w)))
  ## 2) KO column detection (same logic as Tax4Fun2 script)
  if (!"KO" %in% names(w) && is_ko_vector(w[[1]])) setnames(w, 1, "KO")
  if (!"KO" %in% names(w) && ncol(w) >= 2 && is_ko_vector(w[[2]])) {
    w[, (names(w)[1]) := NULL]
    setnames(w, clean_names(names(w)))
    setnames(w, 1, "KO")
  }
  if (!"KO" %in% names(w)) {
    cand <- vapply(w, is_ko_vector, logical(1))
    if (any(cand)) setnames(w, names(w)[which.max(cand)], "KO")
  }
  if (!"KO" %in% names(w)) { peek_dt(w); stop("Couldn't find a KO column in shotgun file.") }
  ## 3) normalize KO, drop empties & stray V## cols
  w[, KO := gsub('^"|\'|"$|\'$', "", as.character(KO))]
  w[, KO := sub("^(?i:ko:)", "", KO, perl = TRUE)]
  w <- w[nzchar(KO)]
  vcols <- grep("^V\\d+$", names(w), value = TRUE)
  if (length(vcols)) w[, (vcols) := NULL]
  ## 4) numericize
  num <- setdiff(names(w), "KO")
  numericize_cols(w, num)
  aggregate_dupe_KO(w)
}

## -------- metrics --------
spearman_fun <- function(x, y) suppressWarnings(cor(x, y, method = "spearman"))
jaccard_fun  <- function(x, y, th = 0) {
  xb <- x > th; yb <- y > th
  u  <- sum(xb | yb)
  if (u == 0) return(NA_real_)
  sum(xb & yb) / u
}

## -------- load data --------
cat("Reading PICRUSt2 KO table:", p2_file, "\n")
p2 <- load_picrust2(p2_file)

cat("Reading shotgun KO table:", shotgun_file, "\n")
wgs <- load_shotgun(shotgun_file)

cat("Reading sample map:", sample_map, "\n")
map <- fread(sample_map)
setnames(map, tolower(names(map)))     # expect columns 'err', 'shotgun_id'
map[, err := trimws(as.character(err))]
map[, shotgun_id := trimws(as.character(shotgun_id))]

## -------- loop & plot --------
rows_metrics <- list()
n_saved <- 0L

for (i in seq_len(nrow(map))) {
  ERR <- map$err[i]
  SG  <- map$shotgun_id[i]
  
  if (!(ERR %in% names(p2)))  { cat("Skip:", ERR, "not in PICRUSt2 table\n"); next }
  if (!(SG %in% names(wgs)))  { cat("Skip:", SG,  "not in shotgun table\n");   next }
  
  common_KO <- intersect(p2$KO, wgs$KO)
  if (!length(common_KO)) { cat("Skip pair", ERR, "/", SG, "no common KO\n"); next }
  
  px <- p2[J(common_KO)][[ERR]]
  wx <- wgs[J(common_KO)][[SG]]
  
  px[!is.finite(px)] <- 0
  wx[!is.finite(wx)] <- 0
  if (sum(px) > 0) px <- px / sum(px)
  if (sum(wx) > 0) wx <- wx / sum(wx)
  
  sp  <- spearman_fun(wx, px)
  jac <- jaccard_fun(wx, px, th = 0)
  
  rows_metrics[[length(rows_metrics)+1]] <- data.table(
    ERR = ERR, shotgun_id = SG, nKO = length(common_KO),
    spearman = sp, jaccard = jac
  )
  
  df <- data.frame(KO = common_KO,
                   shotgun_rel = wx,
                   picrust2_rel = px)
  
  eps <- 1e-6
  df$shotgun_log  <- log10(df$shotgun_rel + eps)
  df$picrust2_log <- log10(df$picrust2_rel + eps)
  
  p <- ggplot(df, aes(x = shotgun_log, y = picrust2_log)) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, alpha = 0.35) +
    geom_point(alpha = 0.45) +
    labs(
      title = sprintf("PICRUSt2 vs Shotgun (%s / %s)", SG, ERR),
      subtitle = sprintf("Spearman = %.3f   |   Jaccard = %.3f   |   nKO = %d", sp, jac, nrow(df)),
      x = "Shotgun KO rel. abundance (log10)",
      y = "PICRUSt2 KO rel. abundance (log10)"
    ) +
    theme_bw(base_size = 11)
  
  out_png <- file.path(fig_dir, sprintf("picrust2_scatter_%s_%s.png", ERR, SG))
  ggsave(out_png, p, width = 6, height = 4, dpi = 300)
  cat("Saved scatter:", out_png, "\n")
  n_saved <- n_saved + 1L
}

## -------- save per-pair metrics --------
if (length(rows_metrics)) {
  M <- rbindlist(rows_metrics)
  fwrite(M, metrics_csv)
  cat("Metrics CSV:", metrics_csv, "\n")
}

cat("Done. Scatterplots saved:", n_saved, "in", fig_dir, "\n")
