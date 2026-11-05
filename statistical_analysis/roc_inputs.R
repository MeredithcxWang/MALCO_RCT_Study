numeric_column <- function(df, name) {
  if (!name %in% names(df)) stop(paste("Missing column:", name))
  value <- df[[name]]
  if (!is.numeric(value) || anyNA(value) || any(!is.finite(value))) stop(paste("Invalid numeric column:", name))
  value
}
roc_inputs <- function(df) {
  label <- if ("True_Label" %in% names(df)) "True_Label" else "True_Values"
  probability <- if ("Predicted_Probability" %in% names(df)) "Predicted_Probability" else "Predicted_Probabilities"
  df$standard_status <- numeric_column(df, label)
  df$standard_prob <- numeric_column(df, probability)
  if (!setequal(unique(df$standard_status), c(0,1))) stop("ROC requires both binary outcome classes")
  if (any(df$standard_prob < 0 | df$standard_prob > 1)) stop("Probabilities must lie in [0,1]")
  df
}
km_inputs <- function(df, time_column, event_column) {
  df$standard_time <- numeric_column(df, time_column)
  df$death_all <- numeric_column(df, event_column)
  df$standard_class <- numeric_column(df, "Predicted_Class")
  if (any(df$standard_time <= 0)) stop("KM follow-up times must be positive")
  if (any(!df$death_all %in% c(0,1))) stop("KM event indicators must be binary")
  if (!setequal(unique(df$standard_class), c(0,1))) stop("KM comparison requires both predicted classes")
  df
}
fixed_probability_roc <- function(status, probability) {
  pROC::roc(status, probability, levels=c(0,1), direction="<", quiet=TRUE)
}
