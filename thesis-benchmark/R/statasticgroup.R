library(data.table)

# Tax4Fun2 metrics
library(data.table)

tax <- fread("/Users/farhadsadat/thesis-benchmark/mendes16s/tax4fun2_out/tax4fun2_vs_shotgun_metrics.csv")

tax[, group := fifelse(grepl("^BulkRAG", shotgun_id), "BulkRAG",
                       fifelse(grepl("^BulkTP", shotgun_id),  "BulkTP",
                               fifelse(grepl("^RAG", shotgun_id),     "RAG",
                                       fifelse(grepl("^RTP", shotgun_id),     "RTP", "Other"))))]

tax_group <- tax[, .(
  n              = .N,
  spearman_mean  = mean(spearman),
  spearman_median= median(spearman),
  jaccard_mean   = mean(jaccard),
  jaccard_median = median(jaccard)
), by = group]

tax_group
