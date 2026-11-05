# Complete, explicitly ID-matched ratings; no rounding before correction.
significance_stars <- function(p) {
  if (anyNA(p) || any(!is.finite(p)) || any(p < 0 | p > 1)) stop("Invalid p-values")
  as.character(cut(p, c(-Inf, .0001, .001, .01, .05, Inf),
                   labels=c("****", "***", "**", "*", "ns")))
}

patient_means <- function(data, models, metrics, expected_raters=2L) {
  columns <- c("PatientID", "Rater", "Model", "Metric", "Score")
  if (!all(columns %in% names(data)) || !nrow(data)) stop("Missing rating columns or empty input")
  if (anyNA(data[, columns])) stop("Ratings must be complete")
  for (column in columns[1:4]) {
    data[[column]] <- as.character(data[[column]])
    if (any(!nzchar(trimws(data[[column]])))) stop("Blank rating identifiers")
  }
  if (!is.numeric(data$Score) || any(!is.finite(data$Score)) || any(!data$Score %in% 1:5))
    stop("Scores must be complete integers from 1 to 5")
  if (!setequal(unique(data$Model), models) || !setequal(unique(data$Metric), metrics))
    stop("Unexpected or missing models/metrics")
  if (length(unique(data$Rater)) != expected_raters) stop("Unexpected number of raters")
  if (anyDuplicated(data[, columns[1:4]])) stop("Duplicate patient/rater/model/metric rating")
  expected <- length(unique(data$PatientID)) * expected_raters * length(models) * length(metrics)
  if (nrow(data) != expected) stop("Incomplete panel: all raters must score the same patients, models and metrics")
  aggregate(Score ~ PatientID + Model + Metric, data, mean)
}

paired_family <- function(means, models, metrics) {
  results <- list()
  for (metric in metrics) for (pair in combn(models, 2, simplify=FALSE)) {
    a <- means[means$Metric == metric & means$Model == pair[1], c("PatientID", "Score")]
    b <- means[means$Metric == metric & means$Model == pair[2], c("PatientID", "Score")]
    if (anyDuplicated(a$PatientID) || anyDuplicated(b$PatientID) ||
        !setequal(a$PatientID, b$PatientID) || nrow(a) < 2L) stop("Invalid patient-level pairs")
    b <- b[match(a$PatientID, b$PatientID), ]
    delta <- a$Score - b$Score
    if (anyNA(delta) || any(!is.finite(delta))) stop("Invalid paired scores")
    # Likert ratings contain ties/zero differences: explicit asymptotic test,
    # continuity correction, two-sided alternative. Identical pairs give p=1.
    p <- if (all(delta == 0)) 1 else stats::wilcox.test(a$Score, b$Score,
      paired=TRUE, alternative="two.sided", exact=FALSE, correct=TRUE)$p.value
    results[[length(results)+1L]] <- data.frame(Metric=metric, group1=pair[1], group2=pair[2],
      n=nrow(a), nonzero_differences=sum(delta != 0), mean_difference=mean(b$Score-a$Score), p=p,
      incremental=match(pair[2], models)-match(pair[1], models) == 1L)
  }
  result <- do.call(rbind, results)
  result$family_size <- nrow(result)
  result$p.adj <- stats::p.adjust(result$p, method="bonferroni", n=nrow(result))
  result$p.adj.signif <- significance_stars(result$p.adj)
  result
}
