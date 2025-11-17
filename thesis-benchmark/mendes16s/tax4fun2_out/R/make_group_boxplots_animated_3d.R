#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

## --- paths ---
proj_dir <- "/Users/farhadsadat/thesis-benchmark"
in_csv   <- file.path(proj_dir, "analysis", "results_colwise_R", "merged_metrics_with_groups.csv")
out_dir  <- file.path(proj_dir, "analysis", "results_colwise_R")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## --- load data ---
if (!file.exists(in_csv)) stop("Not found: ", in_csv, "\nRun the group boxplot script first.")
ALL <- fread(in_csv)

## clean types / order
ALL[, group  := factor(group, levels = c("Bulk","RAG","RTP","Other"))]
ALL[, method := factor(method, levels = c("Tax4Fun2","PICRUSt2"))]
ALL[, spearman := as.numeric(spearman)]
ALL[, jaccard  := as.numeric(jaccard)]
ALL <- ALL[!is.na(group) & !is.na(method) & is.finite(spearman) & is.finite(jaccard)]

## ---------------------------
## 1) Animated facet boxplot
## ---------------------------
need_pkgs <- c("gganimate","gifski")
for (p in need_pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop("Package '", p, "' is required for animation. Install with install.packages('", p, "').")
  }
}
library(gganimate)

ALL_long <- melt(
  ALL,
  id.vars = c("method","group"),
  measure.vars = c("spearman","jaccard"),
  variable.name = "metric",
  value.name = "value"
)

# Animation over metric (frames), columns = method
p_anim <- ggplot(ALL_long, aes(x = group, y = value, fill = group)) +
  geom_boxplot(outlier.alpha = 0.25, width = 0.7) +
  geom_jitter(width = 0.12, size = 0.9, alpha = 0.25, show.legend = FALSE) +
  facet_wrap(~ method, nrow = 1) +
  labs(title = "Benchmarking by Group — Metric: {closest_state}",
       x = "Group", y = "Value") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none",
        panel.grid.minor = element_blank()) +
  transition_states(metric, transition_length = 2, state_length = 1)

anim_gif <- file.path(out_dir, "box_facet_method_metric.gif")
a <- animate(
  p_anim,
  renderer = gifski_renderer(anim_gif),
  width = 900, height = 500, fps = 20, duration = 6
)
invisible(a)
cat("Saved animation:", anim_gif, "\n")

# Optional: animate over method instead (two states), columns = metric
p_anim2 <- ggplot(ALL_long, aes(x = group, y = value, fill = group)) +
  geom_boxplot(outlier.alpha = 0.25, width = 0.7) +
  geom_jitter(width = 0.12, size = 0.9, alpha = 0.25, show.legend = FALSE) +
  facet_wrap(~ metric, nrow = 1, scales = "free_y") +
  labs(title = "Benchmarking by Group — Method: {closest_state}",
       x = "Group", y = "Value") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none",
        panel.grid.minor = element_blank()) +
  transition_states(method, transition_length = 2, state_length = 1)

anim_gif2 <- file.path(out_dir, "box_facet_method_cycle_method.gif")
a2 <- animate(
  p_anim2,
  renderer = gifski_renderer(anim_gif2),
  width = 900, height = 500, fps = 20, duration = 5
)
invisible(a2)
cat("Saved animation:", anim_gif2, "\n")

## ---------------------------
## 2) Interactive 3D scatter “box-style”
## ---------------------------
## Note: Plotly doesn’t have true 3D boxplots; we show a 3D point cloud
## z = metric value, x = group, y = method, with semi-transparent median planes.

if (!requireNamespace("plotly", quietly = TRUE) ||
    !requireNamespace("htmlwidgets", quietly = TRUE)) {
  cat("Skipping 3D HTML: install.packages(c('plotly','htmlwidgets')) to enable.\n")
} else {
  library(plotly)
  library(htmlwidgets)
  
  # map factors to numeric axes for 3D
  gx <- levels(ALL$group); mx <- levels(ALL$method)
  gx_map <- setNames(seq_along(gx), gx)
  mx_map <- setNames(seq_along(mx), mx)
  
  make_3d <- function(metric_name, outfile) {
    DT <- copy(ALL)
    DT[, x_group := gx_map[as.character(group)]]
    DT[, y_method := mx_map[as.character(method)]]
    DT[, z_value := get(metric_name)]
    
    # jitter helpers to spread points
    set.seed(42)
    DT[, x_j := x_group + runif(.N, -0.15, 0.15)]
    DT[, y_j := y_method + runif(.N, -0.15, 0.15)]
    
    # median “planes” per (group, method)
    meds <- DT[, .(z_med = median(z_value, na.rm = TRUE)), by = .(group, method)]
    meds[, x0 := gx_map[as.character(group)] - 0.28]
    meds[, x1 := gx_map[as.character(group)] + 0.28]
    meds[, y0 := mx_map[as.character(method)] - 0.28]
    meds[, y1 := mx_map[as.character(method)] + 0.28]
    
    plt <- plot_ly()
    
    # add 3D points
    plt <- add_markers(
      plt,
      data = DT,
      x = ~x_j, y = ~y_j, z = ~z_value,
      color = ~group, colors = "Set2",
      marker = list(size = 3, opacity = 0.6),
      hoverinfo = "text",
      text = ~paste("Group:", group, "<br>Method:", method,
                    "<br>", metric_name, ":", round(z_value, 3))
    )
    
    # add median “tiles” per (group, method)
    for (i in seq_len(nrow(meds))) {
      r <- meds[i]
      xs <- c(r$x0, r$x1, r$x1, r$x0)
      ys <- c(r$y0, r$y0, r$y1, r$y1)
      zs <- rep(r$z_med, 4)
      plt <- add_trace(
        plt,
        x = xs, y = ys, z = zs,
        type = "mesh3d",
        i = c(0), j = c(1), k = c(2),  # minimal mesh, plotly will fill
        opacity = 0.25,
        name = paste("Median:", r$group, r$method),
        showscale = FALSE
      )
    }
    
    plt <- layout(
      plt,
      scene = list(
        xaxis = list(title = "Group", tickmode = "array",
                     tickvals = seq_along(gx), ticktext = gx),
        yaxis = list(title = "Method", tickmode = "array",
                     tickvals = seq_along(mx), ticktext = mx),
        zaxis = list(title = metric_name)
      ),
      title = paste("3D view:", metric_name)
    )
    
    htmlwidgets::saveWidget(as_widget(plt), outfile, selfcontained = TRUE)
    cat("Saved 3D HTML:", outfile, "\n")
  }
  
  make_3d("spearman", file.path(out_dir, "box_3d_scatter_spearman.html"))
  make_3d("jaccard",  file.path(out_dir, "box_3d_scatter_jaccard.html"))
}

cat("All done.\n")
