# File Description: Merging FOIA employers with Revelio Employer RCIDs
# Author: Amy Kim
# Date Created: Mon Feb 24 16:16:53 2025

# IMPORTING PACKAGES ----
library(tidyverse)
library(glue)

# SETTING WORKING DIRECTORIES ----
root <- "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code <- "/Users/amykim/Documents/GitHub/h1bworkers/code"

# READING IN REVELIO DATA ----
all_data_list <- list()
for (yr in 2021:2023){
  all_data_list[[yr - 2020]] <- read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY{yr}.csv"))
}
all_data_list[[4]] <- bind_rows(list(read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_single_reg.csv")),
                                     read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_multi_reg.csv"))))
raw <- bind_rows(all_data_list)

emp_unique <- raw %>% group_by(employer_name, FEIN) %>%
  summarize(n = n())

# READING IN REVELIO EMPLOYERS ----
revelio_emp <- read_csv(glue("{root}/data/int/revelio/us_company_mapping.csv"))

# CLEANING EMPLOYER NAMES ----

  
