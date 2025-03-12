# File Description: FOIA Data Explore
# Author: Amy Kim
# Date Created: Mon Feb 24 16:16:53 2025

# IMPORTING PACKAGES ----
library(tidyverse)
library(glue)
library(readxl)

# SETTING WORKING DIRECTORIES ----
root <- "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code <- "/Users/amykim/Documents/GitHub/h1bworkers/code"

# READING IN DATA ----
## Bloomberg FOIA data
all_data_list <- list()
for (yr in 2021:2023){
  all_data_list[[yr - 2020]] <- read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY{yr}.csv"))
}
all_data_list[[4]] <- bind_rows(list(read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_single_reg.csv")),
                                     read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_multi_reg.csv"))))

raw <- bind_rows(all_data_list) %>%
  mutate(rec_date = as.Date(rec_date, "%m/%d/%Y"),
         rec_fy = as.numeric(str_sub(quarter(rec_date, with_year = TRUE, fiscal_start = 10), 1, 4)))

## I-129 FOIA data
all_i129_data_list <- list()
for (yr in 2017:2022){
  print(yr)
  all_i129_data_list[[yr - 2016]] <- read_xlsx(glue("{root}/data/raw/foia_i129/i129_fy{yr}.xlsx"),
                                               na = "(b)(3) (b)(6) (b)(7)(c)")
}

raw_i129 <- bind_rows(all_i129_data_list)
#   lapply(all_i129_data_list, function(x) mutate(x, across(c(REC_FY, NUMBER_OF_BENEFICIARIES), ~as.character(.x)),
#                                                 across(c(REC_DATE, ACT_DATE, VALID_FROM, VALID_TO), ~ifelse(is.Date(.x), format(.x, "%Y-%m-%d"), as.character(.x)))))
# )

## USCIS data
all_uscis_data_list <- list(read_xlsx(glue("{root}/data/raw/uscis/emp_info_fy2009_2012.xlsx")),
                            read_xlsx(glue("{root}/data/raw/uscis/emp_info_fy2013_2016.xlsx")),
                            read_xlsx(glue("{root}/data/raw/uscis/emp_info_fy2017_2020.xlsx")),
                            read_xlsx(glue("{root}/data/raw/uscis/emp_info_fy2021_2024.xlsx")))
raw_uscis <- bind_rows(all_uscis_data_list)

# COMPARING DIFF DATA SOURCES ----
## merging all
aclean <- raw %>% filter(!is.na(rec_date) & S3Q1 %in% c("B", "M")) %>%
  filter(rec_date < as.Date("2023-01-01") & rec_date >= as.Date("2020-01-01")) %>%
  mutate(empclean = str_to_lower(i129_employer_name)) %>%
  select(c(rec_date, empclean, REQUESTED_CLASS, REQUESTED_ACTION, NUMBER_OF_BENEFICIARIES,
           BEN_COUNTRY_OF_BIRTH, JOB_TITLE, S3Q1))

bclean <- raw_i129 %>% filter(S3Q1 %in% c("B", "M")) %>%
  filter(REC_DATE < as.Date("2023-01-01") & REC_DATE >= as.Date("2020-01-01")) %>%
  mutate(empclean = str_to_lower(EMPLOYER_NAME)) %>%
  rename(rec_date = REC_DATE) %>%
  select(c(rec_date, empclean, REQUESTED_CLASS, REQUESTED_ACTION, NUMBER_OF_BENEFICIARIES,
           BEN_COUNTRY_OF_BIRTH, JOB_TITLE, S3Q1))
  
acoll <- aclean %>% group_by(rec_date, empclean, REQUESTED_CLASS, REQUESTED_ACTION, 
                             BEN_COUNTRY_OF_BIRTH, JOB_TITLE, S3Q1) %>%
  summarize(n_bb = n()) %>% ungroup() %>% mutate(id_bb = row_number())

bcoll <- bclean %>% group_by(rec_date, empclean, REQUESTED_CLASS, REQUESTED_ACTION, 
                             BEN_COUNTRY_OF_BIRTH, JOB_TITLE, S3Q1) %>%
  summarize(n_i129 = n()) %>% ungroup() %>% mutate(id_foia = row_number())

coll_merge <- full_join(acoll %>% mutate(data = "bb"), bcoll %>% mutate(data = "i129"),
                        by = c("rec_date", "empclean", "REQUESTED_CLASS", "REQUESTED_ACTION",
                               "BEN_COUNTRY_OF_BIRTH", "JOB_TITLE", "S3Q1")) %>%
  mutate(mergegroup = case_when(is.na(data.x) ~ "in I129 not BB",
                                is.na(data.y) ~ "in BB not I129",
                                !is.na(data.x) & !is.na(data.y) ~ "in Both",
                                TRUE ~ "Other"))

## fy 21
fy21 <- raw %>% filter(rec_fy == 2021 & REQUESTED_CLASS != "HSC" & S3Q1 != "E")
fy21i129 <- raw_i129 %>% filter(REC_FY == 2021 & REQUESTED_CLASS != "HSC" & S3Q1 != "E")

empa <- fy21 %>% 
  mutate(emp = str_to_lower(i129_employer_name)) %>%
  group_by(emp) %>%
  summarize(n_all=n())

empb <- fy21i129 %>% 
  mutate(emp = str_to_lower(EMPLOYER_NAME)) %>%
  group_by(emp) %>%
  summarize(n_i129=n())
  
y <- full_join(empa, empb, by = "emp") %>%
  mutate(ndiff = n_all - n_i129)

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
