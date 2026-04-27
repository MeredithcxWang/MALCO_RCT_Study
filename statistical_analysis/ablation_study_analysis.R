# ==============================================================================
# Ablation Study: Score Distribution and Mean Comparison
#
# Description:
#   This script compares Likert-scale evaluation scores (1-5) between two
#   model variants ("My Model" vs "Base Model") across five quality metrics.
#   It performs paired Wilcoxon signed-rank tests with Bonferroni correction
#   and generates a 100% stacked bar chart with mean scores and significance
#   annotations overlaid.
# ==============================================================================

rm(list = ls())

required_packages <- c("tidyverse", "readxl", "ggsci", "rstatix")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0("Package '", pkg, "' is required but not installed."))
  }
  library(pkg, character.only = TRUE)
}

# Parameters

# Paths
data_dir   <- "data/ablation"    # Folder containing reviewer Excel files
output_dir <- "output/figures"   # Folder for saving output figures

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Reviewer file names
reviewer_files <- c("Review_doc_GST.xlsx", "Review_doc_CGH.xlsx")

# Evaluation metrics (must match the order in Excel columns)
metrics <- c("Accuracy", "Safety", "Completeness", "Personalization", "Adherence")

# 1. Data loading and preprocessing

read_scores_raw <- function(file_path, metric_names) {
  df <- read_excel(file_path, skip = 1)
  
  # "My Model" scores: columns 3-7; "Base Model" scores: columns 9-13
  df_my <- df[, c(1, 3:7)]
  colnames(df_my) <- c("PatientID", metric_names)
  df_my$Model <- "My Model"
  
  df_base <- df[, c(1, 9:13)]
  colnames(df_base) <- c("PatientID", metric_names)
  df_base$Model <- "Base Model"
  
  rbind(df_my, df_base)
}

data_raw <- map_dfr(reviewer_files, function(f) {
  read_scores_raw(file.path(data_dir, f), metrics)
})

# 2. Summary statistics and paired Wilcoxon signed-rank tests

# 2.1 Patient-level mean scores 
data_patient_means <- data_raw %>%
  pivot_longer(cols = all_of(metrics),
               names_to = "Metric", values_to = "Score") %>%
  group_by(PatientID, Model, Metric) %>%
  summarise(Score = mean(Score), .groups = "drop")

# 2.2 Group-level mean scores 
summary_stats <- data_patient_means %>%
  group_by(Model, Metric) %>%
  summarise(Mean = mean(Score), .groups = "drop")

# 2.3 Paired Wilcoxon signed-rank test with Bonferroni correction
stat_res <- data_patient_means %>%
  group_by(Metric) %>%
  wilcox_test(Score ~ Model, paired = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance("p.adj") %>%
  mutate(
    Significance = case_when(
      p < 0.001 ~ "***",
      p < 0.01  ~ "**",
      p < 0.05  ~ "*",
      TRUE      ~ ""
    )
  )

# Print statistical results to console
message("\n--- Paired Wilcoxon signed-rank test results (Bonferroni-corrected) ---")
for (i in seq_len(nrow(stat_res))) {
  row <- stat_res[i, ]
  message(sprintf("  %s: p = %s, p.adj = %s %s",
                  row$Metric,
                  formatC(row$p, format = "g", digits = 4),
                  formatC(row$p.adj, format = "g", digits = 4),
                  row$Significance))
}
message("----------------------------------------------------------------------\n")

# 3. Prepare label data for plot annotation

labels_df <- summary_stats %>%
  left_join(stat_res %>% select(Metric, Significance), by = "Metric") %>%
  mutate(
    Label_Text = paste0("Mean: ", sprintf("%.2f", Mean)),
    Label_Text = ifelse(Model == "My Model",
                        paste0(Label_Text, " ", Significance),
                        Label_Text),
    Y_Position = 0.5
  )

# 4. Prepare stacked bar data

plot_data_stack <- data_raw %>%
  pivot_longer(cols = all_of(metrics),
               names_to = "Metric", values_to = "Score") %>%
  mutate(Score = factor(Score, levels = c(1, 2, 3, 4, 5))) %>%
  group_by(Model, Metric, Score) %>%
  summarise(Count = n(), .groups = "drop_last") %>%
  mutate(Percentage = Count / sum(Count)) %>%
  ungroup()

plot_data_stack$Metric <- factor(
  plot_data_stack$Metric,
  levels = c("Adherence", "Personalization", "Completeness", "Safety", "Accuracy")
)

# 5. Visualization

p <- ggplot(plot_data_stack, aes(x = Metric, y = Percentage, fill = Score)) +
  geom_bar(stat = "identity", position = "fill", width = 0.7) +
  facet_wrap(~Model) +
  coord_flip() +

  scale_fill_manual(
    values = c("3" = "#D3D3D3", "4" = "#87CEFA", "5" = "#00008B"),
    name = "Likert Score"
  ) +

  # Overlay mean and significance text on bars
  geom_text(data = labels_df,
            aes(x = Metric, y = Y_Position, label = Label_Text),
            inherit.aes = FALSE,
            color = "white",
            fontface = "bold",
            size = 4.5,
            check_overlap = TRUE) +

  scale_y_continuous(labels = scales::percent) +

  theme_minimal(base_size = 14) +
  labs(
    title    = "Score Distribution & Mean Comparison",
    subtitle = "Mean scores and statistical significance (paired Wilcoxon signed-rank test, Bonferroni-corrected)",
    x = "",
    y = "Percentage of responses"
  ) +
  theme(
    legend.position    = "bottom",
    strip.text         = element_text(size = 14, face = "bold"),
    axis.text.y        = element_text(face = "bold", color = "black"),
    panel.grid.major.y = element_blank()
  )

# Save figure
output_plot_path <- file.path(output_dir, "Stacked_Bar_Comparison.png")
ggsave(output_plot_path, plot = p, width = 12, height = 6, dpi = 300)
print(p)

# 6. Summary
cat("\n--- Analysis complete ---\n")
cat("Figure saved to:", output_plot_path, "\n")

# Session information (for reproducibility)
sessionInfo()
