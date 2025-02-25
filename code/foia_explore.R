# File Description: FOIA Data Explore
# Author: Amy Kim
# Date Created: Mon Feb 24 16:16:53 2025

# IMPORTING PACKAGES ----
library(tidyverse)
library(glue)

# SETTING WORKING DIRECTORIES ----
root <- "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code <- "/Users/amykim/Documents/GitHub/h1bworkers/code"

# READING IN DATA ----
all_data_list <- list()
for (yr in 2021:2023){
  all_data_list[[yr - 2020]] <- read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY{yr}.csv"))
}
all_data_list[[4]] <- bind_rows(list(read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_single_reg.csv")),
                                     read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_multi_reg.csv"))))
raw <- bind_rows(all_data_list)

# TESTING FOR MERGE ----
raw_samp <- sample_n(raw, 100)

test1 <- left_join(raw_samp,
                   us_company_mapping,
                   by = c("employer_name" = "company"))

test2 <- left_join(raw_samp %>% mutate(employer_name = str_to_title(employer_name)), 
                   us_company_mapping,
                   by = c("employer_name" = "company"))

test3 <- left_join(raw_samp %>% mutate(match_name = str_to_title(employer_name)), 
                   us_company_mapping %>% mutate(match_name = str_to_title(company)),
                   by = c("match_name"))

test4 <- left_join(raw_samp %>% mutate(match_name = str_to_title(employer_name)), 
                   us_company_mapping %>% mutate(match_name = str_to_title(child_company)),
                   by = c("match_name"))
