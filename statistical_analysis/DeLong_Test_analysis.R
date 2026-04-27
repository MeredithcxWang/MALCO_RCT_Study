# ==============================================================================
# AUC Comparison Between Internal and External Validation Cohorts
#
# Description:
#   This script iterates over all prediction task folders, computes AUC with
#   95% DeLong confidence intervals for both internal and external validation
#   cohorts, and performs unpaired DeLong tests to compare discriminative
#   performance. Results are exported as a formatted Excel table.
#
# ==============================================================================

rm(list = ls())

required_packages <- c("pROC", "readxl", "openxlsx")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0("Package '", pkg, "' is required but not installed."))
  }
  library(pkg, character.only = TRUE)
}

# Parameters

# Paths 
internal_base <- "data/internal_train"        # Folder with internal validation subfolders
external_base <- "data/external_validation"   # Folder with external validation subfolders
output_path   <- "output/AUC_Comparison.xlsx" # Output Excel file path

# Ensure output directory exists
output_dir <- dirname(output_path)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. Identify prediction task folders

folders <- list.dirs(internal_base, full.names = FALSE, recursive = FALSE)
if (length(folders) == 0) {
  stop("No subfolders found in `internal_base`. Please check the path.")
}

# Initialize results table
results_df <- data.frame(
  Task_Name          = character(),
  Internal_AUC_95CI  = character(),
  External_AUC_95CI  = character(),
  DeLong_Z_Statistic = numeric(),
  P_Value            = numeric(),
  Significance       = character(),
  stringsAsFactors   = FALSE
)

# 2. Compute AUC and DeLong test for each task

for (folder in folders) {

  int_file <- file.path(internal_base, folder,
                        paste0(folder, "_prediction_results.xlsx"))
  ext_file <- file.path(external_base, folder,
                        paste0(folder, "_prediction_results.xlsx"))

  # Check file existence
  if (!file.exists(int_file) || !file.exists(ext_file)) {
    message(sprintf("Skipped '%s': internal or external result file not found.", folder))
    next
  }

  tryCatch({

    # Read data
    int_data <- read_excel(int_file)
    ext_data <- read_excel(ext_file)

    # Extract true labels and predicted probabilities
    int_true <- int_data$True_Values
    int_prob <- int_data$Predicted_Probabilities

    ext_true <- ext_data$True_Label
    ext_prob <- ext_data$Predicted_Probability

    # Build ROC objects
    roc_int <- roc(response = int_true, predictor = int_prob,
                   direction = "<", quiet = TRUE)
    roc_ext <- roc(response = ext_true, predictor = ext_prob,
                   direction = "<", quiet = TRUE)

    # AUC with 95% CI (DeLong method)
    ci_int <- ci.auc(roc_int, method = "delong")
    ci_ext <- ci.auc(roc_ext, method = "delong")

    int_auc_str <- sprintf("%.3f (%.3f\u2013%.3f)", ci_int[2], ci_int[1], ci_int[3])
    ext_auc_str <- sprintf("%.3f (%.3f\u2013%.3f)", ci_ext[2], ci_ext[1], ci_ext[3])

    # Unpaired DeLong test (different cohorts)
    test_res <- roc.test(roc_int, roc_ext, method = "delong", paired = FALSE)
    pval <- test_res$p.value

    sig <- case_when(
      pval < 0.001 ~ "***",
      pval < 0.01  ~ "**",
      pval < 0.05  ~ "*",
      TRUE         ~ "ns"
    )

    # Append to results
    results_df <- rbind(results_df, data.frame(
      Task_Name          = folder,
      Internal_AUC_95CI  = int_auc_str,
      External_AUC_95CI  = ext_auc_str,
      DeLong_Z_Statistic = round(test_res$statistic, 3),
      P_Value            = signif(pval, 3),
      Significance       = sig
    ))

    # Print to console
    message(sprintf("  %s: Internal AUC = %s | External AUC = %s | p = %s %s",
                    folder, int_auc_str, ext_auc_str,
                    formatC(pval, format = "g", digits = 3), sig))

  }, error = function(e) {
    message(sprintf("Error processing '%s': %s", folder, e$message))
  })
}

# 3. Export results to Excel

wb <- createWorkbook()
addWorksheet(wb, "AUC_Comparison")
writeData(wb, "AUC_Comparison", results_df)

# Header styling
headerStyle <- createStyle(textDecoration = "bold", border = "Bottom")
addStyle(wb, "AUC_Comparison", style = headerStyle,
         rows = 1, cols = 1:ncol(results_df), gridExpand = TRUE)
setColWidths(wb, "AUC_Comparison", cols = 1:ncol(results_df), widths = "auto")

saveWorkbook(wb, output_path, overwrite = TRUE)

# 4. Summary
cat("\n--- Analysis complete ---\n")
cat("Results saved to:", output_path, "\n")
cat("Tasks processed:", nrow(results_df), "/", length(folders), "\n")

# Session information (for reproducibility)
sessionInfo()
