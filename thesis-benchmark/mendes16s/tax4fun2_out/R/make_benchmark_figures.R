# --- EDIT THESE THREE PATHS ONLY ---------------------------------------------
PROJ_DIR    <- "/Users/farhadsadat/thesis-benchmark"                 # repo root
S16_DIR     <- file.path(PROJ_DIR, "mendes16s")                      # 16S work dir
SHOTGUN_TSV <- file.path(PROJ_DIR, "shotgun", "rhizo_wgs_p.txt")     # KO table
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(stringr); library(patchwork)
})

# ---------- helpers ----------
normalize_ko_firstcol <- function(dt) {
  # Ensure first column is literally "KO"
  if (is.null(names(dt)[1]) || names(dt)[1] == "" || names(dt)[1] == "V1" ||
      tolower(names(dt)[1]) %in% c("ko", "function", "id")) {
    setnames(dt, 1, "KO")
  } else {
    setnames(dt, 1, "KO")
  }
  dt[, KO := sub("^(ko:|KO:)", "", KO)]
  setkey(dt, KO)
  dt
}

read_ko_matrix <- function(path){
  # fread + .gz handling (no need to install R.utils)
  if (grepl("\\.gz$", path)) {
    dt <- data.table::fread(cmd = paste("gunzip -c", shQuote(path)),
                            check.names = FALSE)
  } else {
    dt <- data.table::fread(path, check.names = FALSE)
  }
  normalize_ko_firstcol(dt)
}

to_numeric_matrix <- function(dt_no_ko) {
  # Coerce all columns to numeric (robust to stray non-numerics)
  m <- as.matrix(data.frame(lapply(dt_no_ko, function(x) as.numeric(x)),
                            check.names = FALSE))
  # Ensure unique colnames (some inputs may duplicate)
  colnames(m) <- make.unique(colnames(m), sep = "_")
  m
}

rel_cols <- function(m){
  if (!is.matrix(m) || any(dim(m) == 0)) return(m)
  cs <- colSums(m, na.rm = TRUE)
  m <- sweep(m, 2, cs, "/")
  m[!is.finite(m)] <- 0
  m
}

spearman <- function(x,y) suppressWarnings(cor(x,y, method="spearman"))
jaccard  <- function(x,y,t=0){
  xb <- x > t; yb <- y > t
  u <- sum(xb | yb); if (u == 0) return(NA_real_)
  sum(xb & yb) / u
}

# ---------- inputs ----------
tax_tsv   <- file.path(S16_DIR, "tax4fun2_out", "tax4fun2_KO_rel.tsv")
picru_tsv <- "/Users/farhadsadat/thesis-benchmark/picrust2/picrust2_out/KO_metagenome_out/pred_metagenome_unstrat.tsv.gz"
smap_csv  <- file.path(PROJ_DIR, "analysis", "sample_map.csv")

stopifnot(file.exists(tax_tsv), file.exists(picru_tsv),
          file.exists(SHOTGUN_TSV), file.exists(smap_csv))

message("Reading tables …")
wgs  <- read_ko_matrix(SHOTGUN_TSV)
tax  <- read_ko_matrix(tax_tsv)
picr <- read_ko_matrix(picru_tsv)

# Intersect KOs and build numeric matrices
K <- Reduce(intersect, list(wgs$KO, tax$KO, picr$KO))
if (length(K) < 50) warning("Very small KO intersection: ", length(K))

wgs_dt <- wgs[J(K)]
tax_dt <- tax[J(K)]
pic_dt <- picr[J(K)]

wgsM  <- to_numeric_matrix(wgs_dt[, -1])
taxM  <- to_numeric_matrix(tax_dt[, -1])
picM  <- to_numeric_matrix(pic_dt[, -1])

rownames(wgsM) <- rownames(taxM) <- rownames(picM) <- K

wgsM <- rel_cols(wgsM); taxM <- rel_cols(taxM); picM <- rel_cols(picM)

# ---------- sample map + groups ----------
smap <- fread(smap_csv)
setnames(smap, tolower(names(smap)))

# Create/infer "group" if missing
if (!("group" %in% names(smap))) {
  if ("shotgun_id" %in% names(smap)) {
    smap[, group := fifelse(grepl("RTP", shotgun_id, ignore.case=TRUE), "RTP",
                            fifelse(grepl("RAG", shotgun_id, ignore.case=TRUE), "RAG",
                                    fifelse(grepl("BULK", shotgun_id, ignore.case=TRUE), "BULK", "OTHER")))]
  } else {
    smap[, group := "OTHER"]
  }
}

req_cols <- c("err","shotgun_id","group")
missing  <- setdiff(req_cols, names(smap))
if (length(missing)) stop("sample_map is missing columns: ", paste(missing, collapse=", "))

# Retain only matched pairs present in both matrices (Tax4Fun2 vs Shotgun)
pairs_tax <- smap[err %in% colnames(taxM) & shotgun_id %in% colnames(wgsM),
                  .(err, shotgun_id, group)]

# For PICRUSt2 vs Tax4Fun2 we use ERR on both sides
pairs_pt <- smap[err %in% colnames(picM) & err %in% colnames(taxM),
                 .(err, group)]

# ---------- 1) Column-wise Tax4Fun2 vs Shotgun ----------
message("Computing Tax4Fun2 vs Shotgun metrics …")
col_tax_wgs <- rbindlist(lapply(seq_len(nrow(pairs_tax)), function(i){
  e <- pairs_tax$err[i]; s <- pairs_tax$shotgun_id[i]; g <- pairs_tax$group[i]
  x <- wgsM[, s, drop=TRUE]; y <- taxM[, e, drop=TRUE]
  data.table(sample=e, shotgun=s, group=g,
             spearman=spearman(x,y), jaccard=jaccard(x,y))
}))
fwrite(col_tax_wgs, file.path(S16_DIR, "tax4fun2_out", "colwise_tax4fun2_vs_shotgun.csv"))

p_tax_spear <- ggplot(col_tax_wgs, aes(spearman)) +
  geom_histogram(bins=18) +
  labs(title="Tax4Fun2 vs Shotgun — Spearman (per sample)", x="Spearman", y="Count")

p_tax_jacc  <- ggplot(col_tax_wgs, aes(jaccard)) +
  geom_histogram(bins=18) +
  labs(title="Tax4Fun2 vs Shotgun — Jaccard (per sample)", x="Jaccard", y="Count")

ggsave(file.path(S16_DIR,"tax4fun2_out","colwise_tax4fun2_spearman_hist.png"),
       p_tax_spear, width=6.5, height=4.2, dpi=220)
ggsave(file.path(S16_DIR,"tax4fun2_out","colwise_tax4fun2_jaccard_hist.png"),
       p_tax_jacc, width=6.5, height=4.2, dpi=220)

# ---------- 2) Column-wise PICRUSt2 vs Tax4Fun2 ----------
message("Computing PICRUSt2 vs Tax4Fun2 metrics …")
grp_lookup <- pairs_pt[, .(err, group)]
setkey(grp_lookup, err)

col_pt <- rbindlist(lapply(pairs_pt$err, function(e){
  x <- picM[, e, drop=TRUE]; y <- taxM[, e, drop=TRUE]
  data.table(sample=e,
             group=grp_lookup[e, group],
             spearman=spearman(x,y),
             jaccard=jaccard(x,y))
}), fill=TRUE)

fwrite(col_pt, file.path(S16_DIR, "tax4fun2_out", "colwise_picrust2_vs_tax4fun2.csv"))

p_pt_spear <- ggplot(col_pt, aes(spearman)) +
  geom_histogram(bins=18) +
  labs(title="PICRUSt2 vs Tax4Fun2 — Spearman (per sample)", x="Spearman", y="Count")

p_pt_jacc  <- ggplot(col_pt, aes(jaccard)) +
  geom_histogram(bins=18) +
  labs(title="PICRUSt2 vs Tax4Fun2 — Jaccard (per sample)", x="Jaccard", y="Count")

ggsave(file.path(S16_DIR,"tax4fun2_out","colwise_picrust2_vs_tax4fun2_spearman_hist.png"),
       p_pt_spear, width=6.5, height=4.2, dpi=220)
ggsave(file.path(S16_DIR,"tax4fun2_out","colwise_picrust2_vs_tax4fun2_jaccard_hist.png"),
       p_pt_jacc, width=6.5, height=4.2, dpi=220)

# ---------- 3) Group-wise violin plots ----------
v_tax_spear <- ggplot(col_tax_wgs, aes(group, spearman)) +
  geom_violin(trim=FALSE) + geom_boxplot(width=.12, outlier.size=0.7) +
  labs(title="Tax4Fun2 vs Shotgun — Spearman by group", x=NULL, y="Spearman")

v_tax_jacc  <- ggplot(col_tax_wgs, aes(group, jaccard)) +
  geom_violin(trim=FALSE) + geom_boxplot(width=.12, outlier.size=0.7) +
  labs(title="Tax4Fun2 vs Shotgun — Jaccard by group", x=NULL, y="Jaccard")

ggsave(file.path(S16_DIR,"tax4fun2_out","tax4fun2_vs_shotgun_spearman_by_group_violin.png"),
       v_tax_spear, width=6.5, height=4.2, dpi=220)
ggsave(file.path(S16_DIR,"tax4fun2_out","tax4fun2_vs_shotgun_jaccard_by_group_violin.png"),
       v_tax_jacc, width=6.5, height=4.2, dpi=220)

# ---------- 4) Final 4-panel figure ----------
final_panel <- (p_tax_spear | p_tax_jacc) / (p_pt_spear | p_pt_jacc)
ggsave(file.path(S16_DIR,"tax4fun2_out","final_benchmark_panel.png"),
       final_panel, width=11.5, height=8.2, dpi=220)

message("Done.
- CSVs in mendes16s/tax4fun2_out/
- Individual histograms + violin plots saved.
- Final figure: mendes16s/tax4fun2_out/final_benchmark_panel.png")
