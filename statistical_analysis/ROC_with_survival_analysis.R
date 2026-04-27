# ==============================================================================
# ROC and Kaplan-Meier Survival Analysis for Model Evaluation
#
# Description:
#   This script generates ROC curves (internal + external validation) and
#   Kaplan-Meier overall survival curves (external validation) for the
#   specified prediction model.
# ==============================================================================

rm(list = ls())

required_packages <- c("readxl", "dplyr", "pROC", "survival", "survminer",
                       "ggplot2", "grid")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0("Package '", pkg, "' is required but not installed."))
  }
  library(pkg, character.only = TRUE)
}

# User-defined parameters 
target_model <- "5yrM"                # Model identifier

# Paths 
base_path_int <- "data/internal"      # Folder containing internal validation results
base_path_ext <- "data/external"      # Folder containing external validation results
output_path   <- "output/figures"     # Folder for saving output figures

# KM plot settings
max_days_full <- 1825                
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
  
  # Identify the time-to-event column
  time_col <- paste0("Days_to_", model_name)
  if (!(time_col %in% colnames(df))) {
    stop(paste("Time column not found:", time_col))
  }
  
  # Standardize predicted probability column name
  if ("Predicted_Probabilities" %in% colnames(df)) {
    df$standard_prob <- df$Predicted_Probabilities
  } else if ("Predicted_Probability" %in% colnames(df)) {
    df$standard_prob <- df$Predicted_Probability
  } else {
    stop("Predicted probability column not found.")
  }
  
  # Standardize true label column name
  if ("True_Values" %in% colnames(df)) {
    df$standard_status <- df$True_Values
  } else if ("True_Label" %in% colnames(df)) {
    df$standard_status <- df$True_Label
  } else {
    stop("True label column (True_Values / True_Label) not found.")
  }
  
  # Predicted class and follow-up time
  df$standard_class <- df$Predicted_Class
  df$standard_time  <- as.numeric(df[[time_col]])
  
  # Overall mortality indicator (available only in external dataset)
  if ("Death.24" %in% colnames(df)) {
    df$death_all <- as.integer(df$Death.24)
  } else {
    df$death_all <- NA_integer_
  }
  
  # Remove rows with missing or invalid time/status
  df_clean <- df %>%
    filter(!is.na(standard_time), standard_time > 0) %>%
    filter(!is.na(standard_status), standard_status %in% c(0, 1))
  
  return(df_clean)
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
  
  roc_obj <- roc(ds_data$standard_status, ds_data$standard_prob, quiet = TRUE)
  auc_val <- auc(roc_obj)
  ci_val  <- ci.auc(roc_obj, conf.level = 0.95, boot.n = 2000, quiet = TRUE)
  
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

# 3. Kaplan-Meier overall survival 
if (!("Death.24" %in% colnames(data_ext))) {
  stop("Column 'Death.24' not found in external dataset; cannot generate KM curve.")
}
if (any(is.na(data_ext$death_all))) {
  stop("Missing values detected in 'Death.24'; please clean the data before running.")
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
  title         = paste("Overall Survival (Full Follow-up) \u2014 External:", target_model),
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

# 4. Summary
cat("\n--- Analysis complete ---\n")
cat("Output directory:", output_path, "\n")
cat("  ROC figure:", paste0(target_model, "_ROC.png"), "\n")
cat("  KM  figure:", paste0(target_model, "_KM_FullFollowup.png"), "\n")

# ==============================================================================
# Session information (for reproducibility)
# ==============================================================================
sessionInfo()
