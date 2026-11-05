# Shared command-line paths. All defaults are synthetic demonstration paths.
if (.Platform$OS.type == "windows") suppressWarnings(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))
script_arg <- grep("^--file=", commandArgs(FALSE), value=TRUE)
if (length(script_arg) != 1) stop("Run this script with Rscript, not source() without a file context.")
project_root <- dirname(dirname(normalizePath(sub("^--file=", "", script_arg), winslash="/", mustWork=TRUE)))
activate <- file.path(project_root,"renv","activate.R")
if (file.exists(activate)) source(activate)
cli_args <- commandArgs(trailingOnly=TRUE)
option <- function(name, default) {
  at <- which(cli_args == paste0("--", name))
  if (length(at) > 1 || (length(at) == 1 && at == length(cli_args))) stop(paste("Invalid option:", name))
  if (length(at)) cli_args[at+1] else default
}
absolute_path <- function(value) {
  if (grepl("^([A-Za-z]:|/|\\\\)", value)) value else file.path(project_root, value)
}
demo_root <- absolute_path(option("demo-dir", "output/demo"))
input_root <- file.path(demo_root, "input")
