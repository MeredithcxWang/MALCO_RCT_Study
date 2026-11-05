# Two-condition public example or four-stage V1-V4 analysis.
# Both paths average the two raters by patient, then use ID-paired tests.
if (.Platform$OS.type == "windows") suppressWarnings(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1])
source(file.path(dirname(script_file), "common.R"), encoding="UTF-8")
source(file.path(dirname(script_file), "paired_scores.R"), encoding="UTF-8")
for (pkg in c("dplyr", "tidyr", "readxl", "ggplot2")) {
  if (!requireNamespace(pkg, quietly=TRUE)) stop(paste("Missing package:", pkg))
  library(pkg, character.only=TRUE)
}

metrics <- c("Accuracy", "Safety", "Completeness", "Personalization", "Adherence")
design <- option("design", "two-condition")
if (!design %in% c("two-condition", "four-stage")) stop("design must be two-condition or four-stage")
models <- if (design == "four-stage") paste0("V", 1:4) else c("Base Model", "My Model")
output_dir <- absolute_path(option("output-dir", file.path(demo_root, "statistics", if (design == "two-condition") "ablation" else "ablation-four-stage")))

if (design == "two-condition") {
  data_dir <- absolute_path(option("data-dir", file.path(input_root, "ablation")))
  data_raw <- bind_rows(lapply(c("GST", "CGH"), function(rater) {
    df <- read_excel(file.path(data_dir, paste0("Review_doc_", rater, ".xlsx")), skip=1)
    if (ncol(df) < 13L || anyDuplicated(df[[1]]) || anyNA(df[[1]])) stop("Invalid rater table or patient IDs")
    bind_rows(lapply(seq_along(models), function(i) {
      scores <- df[, c(1, if (i == 1L) 9:13 else 3:7)]
      names(scores) <- c("PatientID", metrics)
      scores$Model <- models[i]
      scores$Rater <- rater
      pivot_longer(scores, all_of(metrics), names_to="Metric", values_to="Score")
    }))
  }))
} else {
  input <- absolute_path(option("input", file.path(input_root, "ablation_four_stage.csv")))
  data_raw <- read.csv(input, stringsAsFactors=FALSE, colClasses=c(PatientID="character", Rater="character"))
}

means <- patient_means(data_raw, models, metrics, expected_raters=2L)
stat_res <- paired_family(means, models, metrics)
summary_stats <- means %>% group_by(Model, Metric) %>% summarise(Mean=mean(Score), .groups="drop")
adjacent <- stat_res[stat_res$incremental, ]
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)
write.csv(stat_res, file.path(output_dir, "paired_tests.csv"), row.names=FALSE)
write.csv(adjacent, file.path(output_dir, "adjacent_tests.csv"), row.names=FALSE)
write.csv(summary_stats, file.path(output_dir, "mean_scores.csv"), row.names=FALSE)

# The figure and CSV use a single corrected significance field.
labels <- summary_stats %>% left_join(adjacent %>% select(Metric, Model=group2, p.adj.signif), by=c("Metric", "Model")) %>%
  mutate(stars=ifelse(is.na(p.adj.signif) | p.adj.signif == "ns", "", p.adj.signif),
         Label=paste0("Mean: ", sprintf("%.2f", Mean), " ", stars))
distribution <- data_raw %>% mutate(Score=factor(Score, levels=1:5)) %>% count(Model, Metric, Score)
distribution$Model <- factor(distribution$Model, levels=models)
labels$Model <- factor(labels$Model, levels=models)
distribution$Metric <- factor(distribution$Metric, levels=rev(metrics))
labels$Metric <- factor(labels$Metric, levels=rev(metrics))
p <- ggplot(distribution, aes(Metric, n, fill=Score)) +
  geom_col(position="fill", width=.7) + facet_wrap(~Model, nrow=1) + coord_flip() +
  geom_text(data=labels, aes(Metric, .5, label=Label), inherit.aes=FALSE, color="white", fontface="bold", size=3.5) +
  scale_fill_manual(values=c("#B22222", "#CD5C5C", "#D3D3D3", "#87CEFA", "#00008B"), drop=FALSE) +
  scale_y_continuous(labels=scales::percent) + theme_minimal(base_size=12) +
  labs(title=paste("Ablation:", design),
       subtitle=paste0("Two-sided paired Wilcoxon; Bonferroni across ", nrow(stat_res), " comparisons; labels compare adjacent stages"),
       caption="Adjusted P: * <= .05; ** <= .01; *** <= .001; **** <= .0001. No star: P > .05.",
       x=NULL, y="Percentage of ratings", fill="Likert score") + theme(legend.position="bottom")
ggsave(file.path(output_dir, "Stacked_Bar_Comparison.png"), p, width=if (length(models)==4L) 16 else 10, height=6, dpi=300)
cat("Saved", nrow(stat_res), "tests and", nrow(adjacent), "adjacent comparisons to", output_dir, "\n")
print(adjacent[, c("Metric", "group1", "group2", "p", "p.adj", "p.adj.signif")], row.names=FALSE)
sessionInfo()
