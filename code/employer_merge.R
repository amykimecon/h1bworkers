# File Description: Merging FOIA employers with Revelio Employer RCIDs
# Author: Amy Kim
# Date Created: Mon Feb 24 16:16:53 2025

# IMPORTING PACKAGES ----
library(tidyverse)
library(glue)
library(httr)
library(readxl)

# SETTING WORKING DIRECTORIES ----
root <- "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code <- "/Users/amykim/Documents/GitHub/h1bworkers/code"

# HELPER ----
source(glue("{code}/merge_helper.R"))

# READING IN DATA ----
## FOIA DATA ----
all_data_list <- list()
for (yr in 2021:2023){
  all_data_list[[yr - 2020]] <- read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY{yr}.csv"))
}
all_data_list[[4]] <- bind_rows(list(read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_single_reg.csv")),
                                     read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_multi_reg.csv"))))
raw <- bind_rows(all_data_list)

## REVELIO EMPLOYERS ----
revelio_raw <- read_csv(glue("{root}/data/int/revelio/companies_by_positions_locations.csv"))
#print(glue("Failed Matches: {nrow(filter(revelio_emp, rcid...4 != rcid...5))}"))

## HUD API ZIP-CBSA CROSSWALK ----
key <- "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI2IiwianRpIjoiYjY4MTVlOTg0N2YwZTk4NWYzZmMyMmU5NDVhMjJiZmQzYmYwODk4N2QzNzRhZjIyMDE3NmI5Njk3NzI4NTJmZmQwOTJmOGQ1YTVhM2QyNjciLCJpYXQiOjE3NDA1MDgxNjkuNDI1Mzk3LCJuYmYiOjE3NDA1MDgxNjkuNDI1Mzk5LCJleHAiOjIwNTYwNDA5NjkuNDIxMjM1LCJzdWIiOiI5MTgxNiIsInNjb3BlcyI6W119.YPOncB67ZRwswmysM2jbN_koDb8-MUzqZc_JkfoLIBTJUeKO1vDctPWxGuq4sy-mtWApNql5kpBpHQ2sIy2d-g"
url <- "https://www.huduser.gov/hudapi/public/usps"
response <- httr::GET(url, query = list(type = 3, query = "All", year = 2023), 
                      add_headers(Authorization = paste("Bearer", key)))
#access the output
output <- httr::content(response)

# get list of metro areas from revelio data
met_areas_list <- read_csv(glue("{root}/data/int/revelio/metro_areas.csv"))

# get cbsa delineation file
raw_msa_crosswalk <- read_xlsx(glue("{root}/data/crosswalks/cbsa_msa_list.xlsx"))
colnames(raw_msa_crosswalk) <- raw_msa_crosswalk[2,]
msa_crosswalk <- raw_msa_crosswalk[3:(nrow(raw_msa_crosswalk)-3),] %>%
  distinct(`CBSA Code`, `CBSA Title`, `State Name`, `Metropolitan/Micropolitan Statistical Area`) #%>%
## TODO: MERGE WITH met_areas_list to get better mapping (@github issue 1)
  # mutate(city= ifelse(`Metropolitan/Micropolitan Statistical Area` == "Metropolitan Statistical Area", 
  #                            str_to_lower(str_remove(`CBSA Title`, ", [A-Z]{2}(-[A-Z]{2}){0,2}$")), NA)) %>%
  # left_join(met_areas_list %>% mutate(city = ifelse(str_detect(metro_area, "nonmetropolitan area"),
  #                                                   NA, str_remove(metro_area, " metropolitan area")),
  #                                     revelio = 1),
  #           by = c("city" = "city", "State Name" = "state")) %>%
  # mutate(metro_area = ifelse(revelio == 1, metro_area, ))

## could potentially use nber mappings to msa (broader than cbsa?)
# msa_crosswalk_new <- read_csv(glue("{root}/data/crosswalks/nber_cbsa_crosswalk.csv"))

#convert to df
zip_to_cbsa <- bind_rows(lapply(output$data$results, as.data.frame)) #%>%
  # arrange(desc(tot_ratio)) %>%
  # group_by(zip) %>% mutate(rank = row_number()) %>%
  # filter(rank == 1)

# merging all to get zip to msa
zip_to_msa <- zip_to_cbsa %>% 
  left_join(msa_crosswalk %>% distinct(`CBSA Code`, `CBSA Title`), 
            by = c("geoid" = "CBSA Code")) %>%
  filter(!is.na(`CBSA Title`)) %>%
  arrange(desc(tot_ratio)) %>%
  group_by(zip) %>% mutate(rank = row_number()) %>%
  filter(rank == 1)

# getting state abb to state name cw
state_cw <- msa_crosswalk %>% mutate(stateabb = str_match(`CBSA Title`, " ([A-Z]{2})$")[,2]) %>%
  filter(!is.na(stateabb)) %>% distinct(stateabb, `State Name`) %>%
  rename(statename = `State Name`)
state_cw[nrow(state_cw) + 1, ] <- t(c("RI", "Rhode Island"))

# CLEANING H1B EMPLOYER DATA ----
foia_emp <- raw %>% group_by(FEIN, lottery_year) %>% 
  mutate(n_tot_fein_yr = n()) %>% ungroup() %>%
  # filtering duplicate applications
  filter(ben_multi_reg_ind == 0) %>%
  # indicator for likely fraud (high ratio of applications to current employees in us)
  mutate(fraud_ratio = ifelse(is.na(NUM_OF_EMP_IN_US) | NUM_OF_EMP_IN_US == 0, -1,
         n_tot_fein_yr/NUM_OF_EMP_IN_US)) %>%
  # joining with crosswalk to get msa of employer and msa of worksite
  mutate(zip = str_extract(zip, "^[0-9]{5}")) %>%
  left_join(zip_to_msa %>% select(c(zip, city, `CBSA Title`)) %>% rename(msa = `CBSA Title`), by = "zip") %>%
  left_join(zip_to_msa %>% select(c(zip, city, `CBSA Title`)) %>%
              rename(worksite_msa = `CBSA Title`), by = c("WORKSITE_ZIP" = "zip")) %>%
  # grouping by employer location, then employer (getting modal employer msa)
  group_by(employer_name, FEIN, msa, state) %>%
  mutate(n_by_msa = n()) %>% arrange(desc(n_by_msa)) %>% ungroup() %>%
  group_by(employer_name, FEIN) %>%
  mutate(modal_msaname = first(msa),
         modal_msastate = first(state),
         n_by_msa = first(n_by_msa)) %>% ungroup() %>%
  # grouping by worksite, then employer (getting modal worksite msa)
  group_by(employer_name, FEIN, worksite_msa, WORKSITE_STATE) %>%
  mutate(n_by_msa_worksite = sum(ifelse(!is.na(worksite_msa), 1, 0))) %>% arrange(desc(n_by_msa_worksite)) %>% ungroup() %>%
  group_by(employer_name, FEIN) %>%
  mutate(modal_msaworksitename = first(worksite_msa),
         modal_msaworksitestate = first(WORKSITE_STATE),
         n_by_msa_worksite = first(n_by_msa_worksite)) %>% 
  summarize(n_apps = n(),
            n_success = sum(ifelse(status_type == "SELECTED", 1, 0)),
            across(c(modal_msaname, modal_msastate, n_by_msa, modal_msaworksitename,
                     modal_msaworksitestate, n_by_msa_worksite), first),
            fraud_ratio = max(fraud_ratio, na.rm=TRUE),
            last_lot_year = max(lottery_year)
            ) %>%
  mutate(share_msa = n_by_msa/n_apps,
         share_msa_worksite = n_by_msa_worksite/n_success) %>% ungroup() %>%
  # merging to get state names from abbreviations
  left_join(state_cw %>% rename(modal_msastatename = statename), 
            by = c("modal_msastate" = "stateabb")) %>%
  left_join(state_cw %>% rename(modal_msaworksitestatename = statename), 
            by = c("modal_msaworksitestate" = "stateabb"))
# TODO: filter out 'fake' applications/duplicates/high app/emp ratio 

# TESTING MERGE ----
revelio_emp <- revelio_raw %>%
  select(c(company, rcid, n, n_users, recent_start, recent_end, top_metro_area, top_state, lei,
           ultimate_parent_company_name))

foia_samp_ids <- sample_n(foia_emp, 100) %>% select(c(employer_name, FEIN))
revelio_samp <- sample_n(revelio_emp, 100000)

extra_words <- c("inc", "ltd", "plc", "pllc", "mso", "svc", "ggmbh", "gmbh", "limited", "pvt", "md", "private",
                 "llp", "pc", "corporation", "corp", "intl", "international", "group", "co", "usa", "na", "management", 
                 "global", "america", )
clean_emp_name <- function(df, empnamecol){
  # step zero: convert to lowercase and remove symbols
  df[["empnameraw"]] <- str_remove(str_to_lower(df[[empnamecol]]), "™|®|©")
  suffix_regex <- "\\s*,?\\s+(l\\.?\\s?l\\.?\\s?c\\.?|corp|inc|ltd|plc|pllc|mso|svc|gmbh|&?\\s?co|limited|pvt|llp)\\.?\\s*$"
  df_out <- df %>%
    mutate(
      # step one: convert characters to latin ascii (replace accented characters), remove parentheticals or dividers at end of name
      empnameclean = stringi::stri_trans_general(str_remove(empnameraw, "\\(.*\\)|\\|.*$"), "Latin-ASCII"),
      # step two: remove any common suffixes, replace '&' or '+' with and
      empnamestub = str_replace_all(str_remove(str_remove(empnameclean, suffix_regex), suffix_regex),
                                      "\\s?&\\s?|\\s\\+\\s", " and "),
      # step three: remove all parentheticals, removing all symbols and spaces
      empnamebase = str_remove(empnamestub, "\\(.*\\)") %>%
             str_remove_all("[^A-z0-9]"),
      # step four: extracting website if exists from raw string
      empnamesite = str_match(empnameraw, "([A-z0-9\\-]+\\.[a-z]{3})(?:^A-z0-9|$)")[,2]
    )
  return(df_out)
}

foia_samp <- foia_emp %>% filter(employer_name %in% foia_samp_ids$employer_name & FEIN %in% foia_samp_ids$FEIN)
foia_emp_clean <- clean_emp_name(foia_samp, "employer_name")
revelio_emp_clean <- clean_emp_name(revelio_emp , "company") %>% arrange(n_users)

exact_match <- matchfunc("empnameraw")
stub_match <- matchfunc()
base_match <- matchfunc("empnamebase")


exact_match <- left_join(foia_emp_clean,
                   revelio_emp_clean,
                   by = c("empnameraw")) %>%
  mutate(statematch = case_when(is.na(top_state) ~ -1,
                                modal_msastatename == top_state ~ 1,
                                is.na(modal_msaworksitestatename) | modal_msaworksitestatename != top_state ~ 0,
                                modal_msaworksitestatename == top_state ~ 1,
                                TRUE ~ 0)) %>%
  group_by(employer_name) %>% mutate(statematchind = max(statematch, na.rm=TRUE))

# counting number of matches with and without matching state
print(glue("Total number of companies matched: {length(unique(filter(exact_match, !is.na(company))$employer_name))}"))
print(glue("Total number of companies matched with state mismatch: {length(unique(filter(exact_match, statematchind == 0)$employer_name))}"))

stub_match <- left_join(foia_emp_clean,
                   revelio_emp_clean,
                   by = c("empnamestub")) %>%
  mutate(statematch = case_when(is.na(top_state) ~ -1,
                                modal_msastatename == top_state ~ 1,
                                is.na(modal_msaworksitestatename) | modal_msaworksitestatename != top_state ~ 0,
                                modal_msaworksitestatename == top_state ~ 1,
                                TRUE ~ 0)) %>%
  group_by(employer_name) %>% mutate(statematchind = max(statematch, na.rm=TRUE))

# View(test2 %>% select(employer_name, empnamestub, company, n_apps, n, n_users, lei, modal_msaname, modal_msastate,
#                       top_metro_area, top_state,
#                       fraud_ratio, last_lot_year, recent_start, recent_end))

base_match <- left_join(foia_emp_clean,
                   revelio_emp_clean,
                   by = c("empnamebase")) %>%
  mutate(statematch = case_when(is.na(top_state) ~ -1,
                                modal_msastatename == top_state ~ 1,
                                is.na(modal_msaworksitestatename) | modal_msaworksitestatename != top_state ~ 0,
                                modal_msaworksitestatename == top_state ~ 1,
                                TRUE ~ 0)) %>%
  group_by(employer_name) %>% mutate(statematchind = max(statematch, na.rm=TRUE))

View(base_match %>% select(employer_name, company, n_apps, n, n_users, modal_msaname, modal_msastate,
                           top_metro_area, top_state,
                      fraud_ratio, last_lot_year, recent_start, recent_end,
                      modal_msaworksitename, modal_msaworksitestate, lei))



write_csv(base_match, glue("{root}/data/int/base_match_test_26feb2025.csv"))





