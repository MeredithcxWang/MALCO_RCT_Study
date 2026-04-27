# ==============================================================================
# Inter-Rater Reliability Analysis (ICC) for Expert Evaluation Metrics
#
# Description:
#   This script calculates two-way random, absolute agreement, average-measures
#   intraclass correlation coefficients (ICC(2,k)) for each evaluation metric
#   across multiple raters. Results are exported as an Excel table and
#   visualized as a forest-style dot plot with 95% confidence intervals.
# ==============================================================================

rm(list = ls())

required_packages <- c("tidyverse", "readxl", "openxlsx", "irr", "ggsci", "scales")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0("Package '", pkg, "' is required but not installed."))
  }
  library(pkg, character.only = TRUE)
}

# Parameters

# Paths 
data_dir   <- "data/grades"       # Folder containing FINAL_*.xlsx rater files
output_dir <- "output/icc"        # Folder for saving results

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. Data loading and preprocessing

files <- list.files(data_dir, pattern = "^FINAL_.*\\.xlsx$", full.names = TRUE)
if (length(files) == 0) stop("No FINAL_*.xlsx files found. Please check `data_dir`.")

# Read and combine all rater files
raw_data <- map_dfr(files, function(f) {
  df <- read_excel(f)
  df$`Patient ID` <- as.character(df$`Patient ID`)
  return(df)
})

# Column name mapping (long form -> short label)
col_mapping <- c(
  "Accuracy Score (Likert 1~5)"          = "Accuracy",
  "Safety (Likert 1~5)"                  = "Safety",
  "Relevance (Likert 1~5)"               = "Relevance",
  "Empathy (Likert 1~5)"                 = "Empathy",
  "Completeness (Likert 1~5)"            = "Completeness",
  "Usefulness (Likert 1~5)"              = "Usefulness",
  "Adherence to Guidelines (Likert 1~5)" = "Adherence"
)

df_clean <- raw_data %>% rename(any_of(col_mapping))

# Retain only metrics that exist in the data
analysis_metrics <- names(col_mapping)
analysis_metrics <- analysis_metrics[analysis_metrics %in% names(df_clean)]

# 2. Compute ICC(2,k) for each metric

icc_results <- data.frame()

for (metric in analysis_metrics) {

  # Reshape to wide format: rows = patients, columns = raters
  wide_df <- df_clean %>%
    select(`Patient ID`, Rater, all_of(metric)) %>%
    pivot_wider(names_from = Rater, values_from = all_of(metric)) %>%
    column_to_rownames("Patient ID") %>%
    na.omit()

  k <- ncol(wide_df)
  n <- nrow(wide_df)

  if (k < 2) next

  # ICC(2,k): two-way random, absolute agreement, average measures
  icc_obj <- icc(wide_df, model = "twoway", type = "agreement", unit = "average")

  res_row <- data.frame(
    Metric  = metric,
    ICC     = icc_obj$value,
    LowerCI = icc_obj$lbound,
    UpperCI = icc_obj$ubound,
    P_value = icc_obj$p.value
  )

  icc_results <- rbind(icc_results, res_row)

  # Print result to console
  message(sprintf("  %s: ICC = %.3f [95%% CI: %.3f-%.3f], p = %s",
                  metric,
                  icc_obj$value,
                  icc_obj$lbound,
                  icc_obj$ubound,
                  formatC(icc_obj$p.value, format = "g", digits = 4)))
}

# 3. Export results to Excel

export_table <- icc_results %>%
  mutate(across(where(is.numeric), ~ round(., 3)))

output_excel_path <- file.path(output_dir, "ICC_Results.xlsx")
write.xlsx(export_table, output_excel_path)

# 4. Visualization 

plot_data <- icc_results %>%
  mutate(Metric = fct_reorder(Metric, ICC))

p <- ggplot(plot_data, aes(x = Metric, y = ICC, color = Metric)) +
  # Error bars (95% CI)
  geom_errorbar(aes(ymin = LowerCI, ymax = UpperCI),
                width = 0.4, linewidth = 0.8) +
  # Point estimate
  geom_point(size = 5) +
  # Numeric label above each point
  geom_text(aes(label = sprintf("%.2f", ICC)),
            vjust = -0.8, fontface = "bold", show.legend = FALSE) +

  # Reference line for "good reliability"
  geom_hline(yintercept = 0.75, linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.6, y = 0.755, label = "Good reliability (0.75)",
           color = "gray60", hjust = 0, size = 3.5) +

  # Flip coordinates
 coord_flip(ylim = c(0.4, 1.0)) +

  # Lancet color palette
  scale_color_lancet() +

  # Theme
  theme_minimal(base_size = 15) +
  labs(
    title    = "Inter-Rater Reliability (ICC 2,k)",
    subtitle = "Analysis across 7 evaluation metrics",
    x = "",
    y = "Intraclass Correlation Coefficient (95% CI)"
  ) +
  theme(
    legend.position  = "none",
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", fill = NA, linewidth = 0.5),
    axis.text.y      = element_text(face = "bold", color = "black"),
    plot.title       = element_text(face = "bold", hjust = 0.5),
    plot.subtitle    = element_text(hjust = 0.5, color = "gray30")
  )

# Save figure
output_plot_path <- file.path(output_dir, "ICC_Forest_Plot.png")
ggsave(output_plot_path, plot = p, width = 8, height = 6, dpi = 300)

# 5. Summary
cat("\n--- Analysis complete ---\n")
cat("Excel output:", output_excel_path, "\n")
cat("Figure output:", output_plot_path, "\n")

# Session information (for reproducibility)
sessionInfo()
