packages <- c("tidyverse","readxl","openxlsx","ggsci","rstatix","pROC","ggplot2","dplyr","ggpubr","irr","scales","survival","survminer")
db <- installed.packages(noCache=TRUE)
deps <- unique(c(packages,unlist(tools::package_dependencies(packages,db=db,which=c("Depends","Imports","LinkingTo"),recursive=TRUE))))
deps <- deps[deps %in% rownames(db)]
deps <- deps[is.na(db[deps,"Priority"])]
if(length(deps)==0) stop("No dependency metadata available; cannot create a valid lockfile")
records <- lapply(sort(deps),function(p) list(Package=p,Version=unname(db[p,"Version"]),Source="Repository",Repository="CRAN"))
names(records)<-sort(deps)
lock<-list(R=list(Version=paste(R.version$major,R.version$minor,sep="."),Repositories=list(list(Name="CRAN",URL="https://cloud.r-project.org"))),Packages=records)
jsonlite::write_json(lock,"renv.lock",pretty=TRUE,auto_unbox=TRUE)
write.csv(db[sort(deps),c("Package","Version"),drop=FALSE],"statistical_analysis/tested_packages.csv",row.names=FALSE)
cat("Recorded",length(records),"package versions\n")
