# Restore exact package versions recorded in renv.lock into a project library.
# This may require R build tools if a pinned binary is unavailable for your platform.
options(repos=c(CRAN="https://cloud.r-project.org"))
if (!requireNamespace("renv",quietly=TRUE)) install.packages("renv")
renv::restore(project=".",lockfile="renv.lock",prompt=FALSE)
renv::activate(project=".")
cat("Use renv::load() or the generated project environment before running analyses.\n")
