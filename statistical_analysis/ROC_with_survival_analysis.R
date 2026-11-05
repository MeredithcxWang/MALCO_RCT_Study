# ==============================================================================
# ROC and Kaplan-Meier Survival Analysis for Model Evaluation
#
# Description:
#   This script generates ROC curves (internal + external validation) and
#   Kaplan-Meier overall survival curves (external validation) for the
#   specified prediction model.
# ==============================================================================

rm(list = ls())

if (.Platform$OS.type == "windows") suppressWarnings(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1])
source(file.path(dirname(script_file), "common.R"), encoding="UTF-8")
source(file.path(dirname(script_file), "roc_inputs.R"), encoding="UTF-8")


required_packages <- c("readxl", "dplyr", "pROC", "survival", "survminer",
                       "ggplot2", "grid")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0("Package '", pkg, "' is required but not installed."))
  }
  library(pkg, character.only = TRUE)
}

# User-defined parameters
target_model <- option("task", "5yrM")                # Model identifier

# Paths
base_path_int <- absolute_path(option("internal-dir", file.path(demo_root,"internal")))      # Folder containing internal validation results
base_path_ext <- absolute_path(option("external-dir", file.path(demo_root,"external")))      # Folder containing external validation results
output_path <- absolute_path(option("output-dir", file.path(demo_root,"statistics","roc")))     # Folder for saving output figures

# KM plot settings
max_days_full <- as.numeric(option("max-days", if (grepl("^[2-5]yrM$", target_model)) as.character(as.integer(substr(target_model,1,1))*365) else "730"))
if (!is.finite(max_days_full) || max_days_full <= 0) stop("max-days must be positive")
tick_interval <- 182

if (!dir.exists(output_path)) {
  dir.create(output_path, recursive = TRUE)
}

# 1. Data loading and preprocessing
load_and_clean_data <- function(model_name, dataset_type,
                                path_int = base_path_int,
                                path_ext = base_path_ext) {

  folder_path <- if (dataset_type == "internal") {
    file.path(path_int, model_name)
  } else {
    file.path(path_ext, model_name)
  }

  file_name <- paste0(model_name, "_prediction_results.xlsx")
  full_path <- file.path(folder_path, file_name)
  if (!file.exists(full_path)) {
    stop(paste("File not found:", full_path))
  }

  df <- read_excel(full_path)

  # ROC uses all valid binary outcomes/probabilities, independently of KM fields.
  return(roc_inputs(df))
}

data_int <- load_and_clean_data(target_model, "internal")
data_ext <- load_and_clean_data(target_model, "external")

# 2. ROC curves
datasets_list <- list(Internal = data_int, External = data_ext)
roc_data <- data.frame()
palette_roc <- c("#E7B800", "#2E9FDF")

set.seed(42)  # For reproducibility of bootstrap CI

for (i in seq_along(datasets_list)) {
  ds_name <- names(datasets_list)[i]
  ds_data <- datasets_list[[i]]

  roc_obj <- fixed_probability_roc(ds_data$standard_status, ds_data$standard_prob)
  auc_val <- auc(roc_obj)
  ci_val  <- ci.auc(roc_obj, method = option("ci-method", "delong"), conf.level = 0.95, boot.n = 2000, quiet = TRUE)

  auc_ci_text <- sprintf("(AUC = %.2f [95%% CI, %.2f\u2013%.2f])",
                         auc_val, ci_val[1], ci_val[3])
  model_label <- paste0(ds_name, " ", auc_ci_text)

  tmp_df <- data.frame(
    FalsePositiveRate = 1 - roc_obj$specificities,
    TruePositiveRate  = roc_obj$sensitivities,
    Model = model_label
  )
  roc_data <- rbind(roc_data, tmp_df)
}

plot_roc <- ggplot(roc_data,
                   aes(x = FalsePositiveRate, y = TruePositiveRate, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  theme_minimal() +
  labs(x = "1 \u2013 Specificity", y = "Sensitivity",
       title = paste("ROC Curves:", target_model)) +
  scale_color_manual(values = palette_roc) +
  theme(
    plot.title       = element_text(hjust = 0, size = 16, face = "bold"),
    axis.title       = element_text(size = 14, face = "bold.italic"),
    axis.text        = element_text(face = "bold"),
    axis.line        = element_line(linewidth = 1.2),
    legend.title     = element_blank(),
    legend.text      = element_text(size = 8),
    legend.position  = c(0.95, 0.05),
    legend.justification = c("right", "bottom"),
    legend.spacing.y = unit(0.1, "cm"),
    legend.key.height = unit(0.35, "cm"),
    legend.background = element_rect(fill = "white", color = "black",
                                     linewidth = 0.5, linetype = "solid")
  )

ggsave(
  filename = file.path(output_path, paste0(target_model, "_ROC.png")),
  plot = plot_roc, width = 7, height = 6, dpi = 600
)

# 3. Optional endpoint-specific Kaplan-Meier analysis of external data only.
km_enabled <- option("km", "true")
if (!km_enabled %in% c("true", "false")) stop("km must be true or false")
if (km_enabled == "true") {
  # The default event and time refer to the same selected endpoint/horizon.
  # Overall mortality requires explicitly matched --event-column/--time-column.
  data_ext <- km_inputs(data_ext, option("time-column", paste0("Days_to_", target_model)),
                        option("event-column", "True_Label"))
if (any(is.na(data_ext$death_all))) {
  stop("Missing event indicators; cannot generate KM curve.")
}

# Determine x-axis limit
if (is.null(max_days_full)) {
  max_days_full <- max(data_ext$standard_time, na.rm = TRUE)
}

fit_ext_full <- survfit(Surv(standard_time, death_all) ~ standard_class,
                        data = data_ext, conf.type = "plain")

plot_km_full <- ggsurvplot(
  fit_ext_full,
  data          = data_ext,
  conf.int      = TRUE,
  conf.int.style = "ribbon",
  conf.int.level = 0.95,
  conf.int.alpha = 0.45,
  censor        = FALSE,
  pval          = TRUE,
  pval.size     = 6,
  risk.table    = TRUE,
  risk.table.title = "Number at risk",

  xlim          = c(0, max_days_full),
  break.time.by = tick_interval,

  legend.title  = "Prediction",
  legend.labs   = c("Predicted = 0", "Predicted = 1"),
  xlab          = "Days",
  ylab          = "Survival probability",
  title         = paste("Event-free Survival (Supplied Follow-up) \u2014 External:", target_model),
  palette       = c("#1f78b4", "#e31a1c"),

  font.title    = c(17, "bold", "black"),
  font.x        = c(14, "bold.italic", "black"),
  font.y        = c(14, "bold.italic", "black"),
  font.tickslab = c(12, "italic", "black"),

  ggtheme = theme_classic() +
    theme(
      panel.grid.major.y = element_line(color = "gray", linetype = "dashed"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      legend.position       = c(0.95, 0.95),
      legend.justification  = c(1, 1),
      legend.background     = element_rect(fill = "transparent")
    ),

  risk.table.col = "black",
  tables.theme = theme_classic() +
    theme(
      axis.line    = element_line(color = "black"),
      panel.grid   = element_blank(),
      axis.title.y = element_blank(),
      plot.title   = element_text(face = "bold.italic", size = 14),
      axis.title.x = element_text(face = "bold.italic", size = 14),
      axis.text.x  = element_text(face = "italic", size = 12),
      axis.text.y  = element_text(face = "bold.italic", size = 12)
    ),
  risk.table.y.text = FALSE
)

# Remove border lines from confidence interval ribbons
g <- plot_km_full$plot
for (i in seq_along(g$layers)) {
  if (inherits(g$layers[[i]]$geom, "GeomRibbon")) {
    g$layers[[i]]$aes_params$colour    <- NA
    g$layers[[i]]$aes_params$linewidth <- 0
  }
}
plot_km_full$plot <- g

# Save KM plot
png(
  filename = file.path(output_path, paste0(target_model, "_KM_FullFollowup.png")),
  width = 10, height = 7, units = "in", res = 600
)
print(plot_km_full, newpage = FALSE)
dev.off()
}

# 4. Summary
cat("\n--- Analysis complete ---\n")
cat("Output directory:", output_path, "\n")
cat("  ROC figure:", paste0(target_model, "_ROC.png"), "\n")
if (km_enabled == "true") cat("  KM  figure:", paste0(target_model, "_KM_FullFollowup.png"), "\n")

# ==============================================================================
# Session information (for reproducibility)
# ==============================================================================
sessionInfo()
