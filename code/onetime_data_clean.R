# File Description: One-time Data Manipulation Stuff for Merging
# Author: Amy Kim
# Date Created: Thu Feb 27 14:11:24 2025

# IMPORTING PACKAGES ----
library(tidyverse)
library(glue)
library(httr)
library(readxl)

# SETTING WORKING DIRECTORIES ----
root <- "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code <- "/Users/amykim/Documents/GitHub/h1bworkers/code"

## READING IN FOIA DATA AND MERGING INTO ONE EMPLOYER FILE ----
all_data_list <- list()
for (yr in 2021:2023){
  all_data_list[[yr - 2020]] <- read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY{yr}.csv"))
}
all_data_list[[4]] <- bind_rows(list(read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_single_reg.csv")),
                                     read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_multi_reg.csv"))))
raw <- bind_rows(all_data_list) %>%
  mutate(across(everything(), ~str_remove_all(., '"')))
raw$state_name <- c(state.name, "Washington, D.C.")[match(raw$state, c(state.abb, "DC"))]
raw$worksite_state_name <- state.name[match(raw$WORKSITE_STATE, state.abb)]

write_csv(raw, glue("{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv"))
