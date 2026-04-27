# ==============================================================================
# Pre- vs Post-Intervention Survey Score Analysis
#
# Description:
#   This script reads survey data, extracts pre- and post-intervention scores
#   for a specified item, performs a Wilcoxon rank-sum test, and generates a
#   violin-boxplot comparison figure.
# ==============================================================================

rm(list = ls())

required_packages <- c("readxl", "ggplot2", "dplyr", "ggpubr")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0("Package '", pkg, "' is required but not installed."))
  }
  library(pkg, character.only = TRUE)
}

options(scipen = 999)

# Parameters 
Col_Pre  <- "S24"       # Column name for the pre-intervention survey item
Col_Post <- "S25"       # Column name for the post-intervention survey item
Keyword  <- "AI cred"   # Label used in plot title and output file name

# Paths
file_path  <- "data/survey_data.xlsx"   # Input survey data file
output_dir <- "output/figures"          # Output directory for figures

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. Data loading and preprocessing
data <- read_excel(file_path)

# Ensure target columns are numeric
data[[Col_Pre]]  <- as.numeric(data[[Col_Pre]])
data[[Col_Post]] <- as.numeric(data[[Col_Post]])

# Filter to pre/post groups and assign the appropriate score column
plot_data <- data %>%
  filter(Group_Match %in% c("pre", "post")) %>%
  mutate(
    Score = case_when(
      Group_Match == "pre"  ~ .data[[Col_Pre]],
      Group_Match == "post" ~ .data[[Col_Post]]
    )
  ) %>%
  filter(!is.na(Score))

# Set factor levels to ensure Pre appears before Post
plot_data$Group_Match <- factor(plot_data$Group_Match,
                                levels = c("pre", "post"),
                                labels = c("Pre", "Post"))

# 2. Statistical test 
message(paste0("--- [", Keyword, "] Wilcoxon rank-sum test (Pre vs Post) ---"))

stat_result <- compare_means(Score ~ Group_Match,
                             data = plot_data,
                             method = "wilcox.test")
print(stat_result)

# 3. Visualization 
my_colors <- c("Pre" = "#B24745", "Post" = "#2E5F87")

p <- ggplot(plot_data, aes(x = Group_Match, y = Score, fill = Group_Match)) +

  # Violin layer
  geom_violin(trim = FALSE, alpha = 0.85, scale = "width",
              color = "black", size = 0.5, adjust = 1.3) +

  # Boxplot overlay
  geom_boxplot(width = 0.1, fill = "white", color = "black",
               outlier.shape = NA, alpha = 0.8, size = 0.5) +

  # Mean indicator (diamond)
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3,
               fill = "white", color = "black") +

  # Color mapping
  scale_fill_manual(values = my_colors) +

  # Y-axis range
  scale_y_continuous(limits = c(0, 7), breaks = seq(0, 7, 1)) +

  # Theme
 theme_bw() +
  theme(
    panel.grid.major  = element_blank(),
    panel.grid.minor  = element_blank(),
    panel.border      = element_rect(colour = "black", fill = NA, linewidth = 1.2),
    axis.ticks        = element_line(color = "black", linewidth = 1.2),
    axis.text.x       = element_text(size = 14, color = "black", face = "bold",
                                     margin = margin(t = 5)),
    axis.text.y       = element_text(size = 14, color = "black", face = "bold"),
    axis.title.y      = element_text(size = 16, face = "bold", color = "black",
                                     margin = margin(r = 10)),
    axis.title.x      = element_blank(),
    legend.position   = "none",
    plot.title        = element_text(hjust = 0.5, size = 18, face = "bold")
  ) +

  # Labels
  labs(y = "Score (1\u20135)",
       title = paste(Keyword, "Comparison"))

# 4. Save output
print(p)

file_name_pdf <- paste0("Violin_PrePost_", Keyword, ".pdf")
file_name_png <- paste0("Violin_PrePost_", Keyword, ".png")

ggsave(file.path(output_dir, file_name_pdf), plot = p, width = 5, height = 6, dpi = 300)
ggsave(file.path(output_dir, file_name_png), plot = p, width = 5, height = 6, dpi = 300)

message(paste("Figures saved to:", output_dir))
message(paste("Files:", file_name_pdf, "/", file_name_png))

# Session information (for reproducibility)
sessionInfo()
