# File Description: Merging FOIA employers with Revelio Employer RCIDs
# Author: Amy Kim
# Date Created: Mon Feb 24 16:16:53 2025

# IMPORTING PACKAGES ----
library(tidyverse)
library(glue)
library(httr)
library(readxl)
library(lubridate)

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
raw$state_name <- c(state.name, "Washington, D.C.")[match(raw$state, c(state.abb, "DC"))]
raw$worksite_state_name <- state.name[match(raw$WORKSITE_STATE, state.abb)]

## REVELIO EMPLOYERS ----
revelio_raw <- read_csv(glue("{root}/data/int/revelio/companies_by_positions_locations.csv"))
#print(glue("Failed Matches: {nrow(filter(revelio_emp, rcid...4 != rcid...5))}"))

# CLEANING H1B EMPLOYER DATA ----
foia_emp <- raw %>% group_by(FEIN, lottery_year) %>% 
  # getting total number of applications and cleaning name (lowercase + remove commas and periods)
  mutate(n_tot_fein_yr = n(),
         company_FOIA = str_to_lower(str_remove_all(employer_name, ",|\\.")),
  ) %>% ungroup() %>%
  # filtering duplicate applications
  filter(ben_multi_reg_ind == 0) %>%
  # indicator for likely fraud (high ratio of applications to current employees in us)
  mutate(fraud_ratio = ifelse(is.na(NUM_OF_EMP_IN_US) | NUM_OF_EMP_IN_US == 0, -1,
                              n_tot_fein_yr/NUM_OF_EMP_IN_US)) %>%
  # grouping by employer location, then employer (getting modal employer state)
  group_by(FEIN, state_name) %>%
  mutate(n_by_state = n()) %>% arrange(desc(n_by_state)) %>% ungroup() %>%
  group_by(FEIN) %>%
  mutate(top_emp_state = first(state_name),
         n_by_state = first(n_by_state)) %>% ungroup() %>%
  # grouping by worksite, then employer (getting modal worksite state)
  group_by(FEIN, worksite_state_name) %>%
  mutate(n_by_worksite_state = sum(ifelse(!is.na(worksite_state_name), 1, 0))) %>% arrange(desc(n_by_worksite_state)) %>% ungroup() %>%
  group_by(FEIN) %>%
  mutate(top_worksite_state = first(worksite_state_name),
         n_by_worksite_state = first(n_by_worksite_state)) %>% ungroup() %>%
  group_by(FEIN, NAICS_CODE) %>%
  mutate(n_by_naics = sum(ifelse(!is.na(NAICS_CODE), 1, 0))) %>% arrange(desc(n_by_naics)) %>% ungroup() %>%
  group_by(FEIN) %>%
  mutate(top_naics = first(NAICS_CODE),
         n_by_naics = first(n_by_naics)) %>% ungroup() %>%
  group_by(FEIN, company_FOIA) %>%
  summarize(n_apps = n(),
            n_success = sum(ifelse(status_type == "SELECTED", 1, 0)),
            across(c(top_emp_state, top_worksite_state, n_by_state, n_by_worksite_state, top_naics, n_by_naics), first),
            fraud_ratio = max(fraud_ratio, na.rm=TRUE),
            last_lot_year = max(lottery_year)
  ) %>%
  mutate(share_state = n_by_state/n_apps,
         share_worksite_state = n_by_worksite_state/n_success,
         share_naics = n_by_naics/n_success) %>% ungroup()%>%
  mutate(id=row_number())

# TESTING MERGE ----
revelio_emp <- revelio_raw %>%
  filter(!is.na(rcid)) %>%
  select(c(company, rcid, n, n_users, recent_start, recent_end, top_metro_area, top_state, lei,
           ultimate_parent_company_name, naics_code)) %>%
  mutate(id = row_number())

foia_samp_ids <- sample_n(foia_emp, 100) %>% select(c(company_FOIA, FEIN))
revelio_samp <- sample_n(revelio_emp, 100000)

# extra_words <- c("inc", "ltd", "plc", "pllc", "mso", "svc", "ggmbh", "gmbh", "limited", "pvt", "md", "private",
#                  "llp", "pc", "corporation", "corp", "intl", "international", "group", "co", "usa", "na", "management", 
#                  "global", "america", )
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
      empnamesite = str_match(empnameraw, "([A-z0-9\\-]+\\.[a-z]{3})(?:^A-z0-9|$)")[,2],
      empfortoken = trimws(str_replace_all(str_remove_all(str_replace_all(empnameclean, "\\s?&\\s?|\\s\\+\\s", " and "), 
                                                          "[^A-z0-9\\s]"), "\\s+", " "))
    )
  return(df_out)
}

#foia_samp <- foia_emp %>% filter(company_FOIA %in% foia_samp_ids$company_FOIA & FEIN %in% foia_samp_ids$FEIN)
foia_emp_clean <- clean_emp_name(foia_emp, "company_FOIA")
revelio_emp_clean <- clean_emp_name(revelio_emp , "company") #%>% arrange(n_users)

# tokenizing
tokens_long <- bind_rows(list(foia_emp_clean %>% select(id, empfortoken) %>% mutate(data = "foia"),
                              revelio_emp_clean %>% select(id, empfortoken) %>% mutate(data = "rev"))) %>%
  mutate(fullname = empfortoken) %>% #sample_n(10000) %>%
  separate_longer_delim(empfortoken, " ") %>%
  group_by(1) %>% mutate(tot = n(), token = empfortoken) %>% ungroup() %>%
  group_by(token) %>% mutate(freq = n()/tot) %>% ungroup() 

# getting frequency cutoff for top 200 most frequent tokens
token_freq_cutoff <- arrange(tokens_long %>% group_by(token) %>% summarize(freq = mean(freq)), desc(freq))$freq[200]

tokens_wide <- tokens_long %>% filter(freq < token_freq_cutoff) %>%
  group_by(id, data) %>% arrange(freq) %>% mutate(rank = row_number()) %>% filter(rank <= 5) %>%
  pivot_wider(id_cols = c(id, data), names_prefix = "token", names_from = "rank", values_from = "token")

foia_emp_tokens <- foia_emp_clean %>% left_join(filter(tokens_wide, data == "foia") %>% select(-c(data)))
revelio_emp_tokens <- revelio_emp_clean %>% left_join(filter(tokens_wide, data == "rev") %>% select(-c(data)))


## matching on rarest and second rarest tokens
token_match <- matchfunc(matchtype = c("token1", "token2"), 
                         dbfoia = foia_emp_tokens %>% filter(!is.na(token1)), 
                         dbrev = revelio_emp_tokens %>% filter(!is.na(token1)))
write_csv(token_match, glue("{root}/data/int/token_match_r_mar10.csv"))

## matching directly on strings               
exact_match <- matchfunc(c("empnameraw"))
write_csv(exact_match, glue("{root}/data/int/exact_match_r_mar6.csv"))
stub_match <- matchfunc()
write_csv(exact_match, glue("{root}/data/int/stub_match_r_mar6.csv"))
base_match <- matchfunc(c("empnamebase"))
write_csv(exact_match, glue("{root}/data/int/base_match_r_mar6.csv"))

## good matches
### for 1-1 matches, keep all for now
single_match <- token_match %>% group_by(id.x) %>% mutate(nmatch = sum(ifelse(!is.na(company), 1, 0))) %>%
  filter(nmatch == 1) %>%
  rowwise() %>% mutate(recent_act = max(recent_start, recent_end, as.Date("2000-01-01"), na.rm=TRUE)) %>% ungroup() #%>%
#filter(n > 5 & recent_act >= as.Date("2020-01-01"))

### for many-1 matches
mult_match_all <- token_match %>% group_by(id.x) %>% mutate(nmatch = sum(ifelse(!is.na(company), 1, 0))) %>%
  filter(nmatch > 1) %>%
  mutate(rawstringdist = stringdist(empnameraw.x, empnameraw.y),
         stubstringdist = stringdist(empnamestub.x, empnamestub.y)) 

mult_match_certain <- mult_match_all %>%
  filter(empnamestub.x == empnamestub.y & n >= max(n)*0.5 & statematch == 1) %>%
  mutate(nmatch = n()) %>% filter(nmatch == 1)

mult_match_certain2 <- mult_match_all %>% left_join(mult_match_certain %>% select(id.x) %>% mutate(certainmatch = 1)) %>%
  filter(is.na(certainmatch)) %>%
  filter(stubstringdist <= 2) %>%
  mutate(nmatch = n()) %>% 
  filter(nmatch <= 5) %>% filter(top_naics2 == naics_code2 | n > 100 | (statematch == 1 & n >= 0.5*max(n))) %>%
  mutate(nmatch = n()) %>% filter(nmatch == 1)

mult_match_filt <- mult_match_all %>% left_join(bind_rows(list(mult_match_certain, mult_match_certain2)) %>% 
                                                  select(id.x) %>% mutate(certainmatch = 1)) %>%
  filter(is.na(certainmatch)) %>% mutate(nmatch = n()) %>%
  filter(nmatch < 5 | rawstringdist <= median(rawstringdist)) %>%
  # filter((!is.na(top_naics) & !is.na(naics_code) & top_naics != 999999 & top_naics == naics_code) |
  #          (max(ifelse(!is.na(top_naics) & !is.na(naics_code) & top_naics != 999999 & top_naics == naics_code, 1, 0)) == 0)) %>%
  filter((!is.na(top_naics2) & !is.na(naics_code2) & top_naics2 != "99" & top_naics2 == naics_code2) |
           (max(ifelse(!is.na(top_naics2) & !is.na(naics_code2) & top_naics2 != "99" & top_naics2 == naics_code2, 1, 0)) == 0)) %>%
  filter(
    (!is.na(top_worksite_state) & !is.na(top_state) & top_worksite_state == top_state) | 
      (!is.na(top_emp_state) & !is.na(top_state) & top_emp_state == top_state) |
      (max(ifelse(!is.na(top_worksite_state) & !is.na(top_state) & top_worksite_state == top_state, 1, 0)) == 0 &
         max(ifelse(!is.na(top_emp_state) & !is.na(top_state) & top_emp_state == top_state, 1, 0)) == 0)
  ) %>%
  mutate(nmatch = n())

mult_match3 <- mult_match_filt %>%
  filter(nmatch <= 5 & stubstringdist <= min(stubstringdist) + 2 & n >= max(n)*0.5) %>%
  mutate(nmatch = n()) %>% filter(nmatch == 1)

# FINAL SET OF DECENT ONE-ONE MATCHES
good_matches <- bind_rows(list(
  single_match %>% filter(stringdist(empnamestub.x, empnamestub.y) <= 2 | statematch == 1 | naics2match == 1),
  mult_match_certain,
  mult_match_certain2,
  mult_match3
))
print(glue("Total FEINs: {length(unique(foia_emp$FEIN))}"))
print(glue("Matched FEINs: {length(unique(good_matches$FEIN))} ({round(length(unique(good_matches$FEIN))*100/length(unique(foia_emp$FEIN)), 1)}%)"))

unmatched <- filter(foia_emp, !(id %in% good_matches$id.x))
unmatched_fein <- filter(foia_emp, !(FEIN %in% good_matches$FEIN))

# OLD CODE ----
# ## HUD API ZIP-CBSA CROSSWALK ----
# key <- "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI2IiwianRpIjoiYjY4MTVlOTg0N2YwZTk4NWYzZmMyMmU5NDVhMjJiZmQzYmYwODk4N2QzNzRhZjIyMDE3NmI5Njk3NzI4NTJmZmQwOTJmOGQ1YTVhM2QyNjciLCJpYXQiOjE3NDA1MDgxNjkuNDI1Mzk3LCJuYmYiOjE3NDA1MDgxNjkuNDI1Mzk5LCJleHAiOjIwNTYwNDA5NjkuNDIxMjM1LCJzdWIiOiI5MTgxNiIsInNjb3BlcyI6W119.YPOncB67ZRwswmysM2jbN_koDb8-MUzqZc_JkfoLIBTJUeKO1vDctPWxGuq4sy-mtWApNql5kpBpHQ2sIy2d-g"
# url <- "https://www.huduser.gov/hudapi/public/usps"
# response <- httr::GET(url, query = list(type = 3, query = "All", year = 2023), 
#                       add_headers(Authorization = paste("Bearer", key)))
# #access the output
# output <- httr::content(response)
# 
# # get list of metro areas from revelio data
# met_areas_list <- read_csv(glue("{root}/data/int/revelio/metro_areas.csv"))
# 
# # get cbsa delineation file
# raw_msa_crosswalk <- read_xlsx(glue("{root}/data/crosswalks/cbsa_msa_list.xlsx"))
# colnames(raw_msa_crosswalk) <- raw_msa_crosswalk[2,]
# msa_crosswalk <- raw_msa_crosswalk[3:(nrow(raw_msa_crosswalk)-3),] %>%
#   distinct(`CBSA Code`, `CBSA Title`, `State Name`, `Metropolitan/Micropolitan Statistical Area`) #%>%
# ## TODO: MERGE WITH met_areas_list to get better mapping (@github issue 1)
#   # mutate(city= ifelse(`Metropolitan/Micropolitan Statistical Area` == "Metropolitan Statistical Area", 
#   #                            str_to_lower(str_remove(`CBSA Title`, ", [A-Z]{2}(-[A-Z]{2}){0,2}$")), NA)) %>%
#   # left_join(met_areas_list %>% mutate(city = ifelse(str_detect(metro_area, "nonmetropolitan area"),
#   #                                                   NA, str_remove(metro_area, " metropolitan area")),
#   #                                     revelio = 1),
#   #           by = c("city" = "city", "State Name" = "state")) %>%
#   # mutate(metro_area = ifelse(revelio == 1, metro_area, ))
# 
# ## could potentially use nber mappings to msa (broader than cbsa?)
# # msa_crosswalk_new <- read_csv(glue("{root}/data/crosswalks/nber_cbsa_crosswalk.csv"))
# 
# #convert to df
# zip_to_cbsa <- bind_rows(lapply(output$data$results, as.data.frame)) #%>%
#   # arrange(desc(tot_ratio)) %>%
#   # group_by(zip) %>% mutate(rank = row_number()) %>%
#   # filter(rank == 1)
# 
# # merging all to get zip to msa
# zip_to_msa <- zip_to_cbsa %>% 
#   left_join(msa_crosswalk %>% distinct(`CBSA Code`, `CBSA Title`), 
#             by = c("geoid" = "CBSA Code")) %>%
#   filter(!is.na(`CBSA Title`)) %>%
#   arrange(desc(tot_ratio)) %>%
#   group_by(zip) %>% mutate(rank = row_number()) %>%
#   filter(rank == 1)
# 
# # getting state abb to state name cw
# state_cw <- msa_crosswalk %>% mutate(stateabb = str_match(`CBSA Title`, " ([A-Z]{2})$")[,2]) %>%
#   filter(!is.na(stateabb)) %>% distinct(stateabb, `State Name`) %>%
#   rename(statename = `State Name`)
# state_cw[nrow(state_cw) + 1, ] <- t(c("RI", "Rhode Island"))
# 
# # CLEANING H1B EMPLOYER DATA ----
# foia_emp <- raw %>% group_by(FEIN, lottery_year) %>% 
#   # getting total number of applications and cleaning name (lowercase + remove commas and periods)
#   mutate(n_tot_fein_yr = n(),
#          company_FOIA = str_to_lower(str_remove_all(employer_name, ",|\\."))) %>% ungroup() %>%
#   # filtering duplicate applications
#   filter(ben_multi_reg_ind == 0) %>%
#   # indicator for likely fraud (high ratio of applications to current employees in us)
#   mutate(fraud_ratio = ifelse(is.na(NUM_OF_EMP_IN_US) | NUM_OF_EMP_IN_US == 0, -1,
#          n_tot_fein_yr/NUM_OF_EMP_IN_US)) %>%
#   # joining with crosswalk to get msa of employer and msa of worksite
#   mutate(zip = str_extract(zip, "^[0-9]{5}")) %>%
#   left_join(zip_to_msa %>% select(c(zip, city, `CBSA Title`)) %>% rename(msa = `CBSA Title`), by = "zip") %>%
#   left_join(zip_to_msa %>% select(c(zip, city, `CBSA Title`)) %>%
#               rename(worksite_msa = `CBSA Title`), by = c("WORKSITE_ZIP" = "zip")) %>%
#   # grouping by employer location, then employer (getting modal employer msa)
#   group_by(FEIN, msa, state) %>%
#   mutate(n_by_msa = n()) %>% arrange(desc(n_by_msa)) %>% ungroup() %>%
#   group_by(FEIN) %>%
#   mutate(modal_msaname = first(msa),
#          modal_msastate = first(state),
#          n_by_msa = first(n_by_msa)) %>% ungroup() %>%
#   # grouping by worksite, then employer (getting modal worksite msa)
#   group_by(FEIN, worksite_msa, WORKSITE_STATE) %>%
#   mutate(n_by_msa_worksite = sum(ifelse(!is.na(worksite_msa), 1, 0))) %>% arrange(desc(n_by_msa_worksite)) %>% ungroup() %>%
#   group_by(FEIN) %>%
#   mutate(modal_msaworksitename = first(worksite_msa),
#          modal_msaworksitestate = first(WORKSITE_STATE),
#          n_by_msa_worksite = first(n_by_msa_worksite)) %>% ungroup() %>%
#   group_by(FEIN, company_FOIA) %>%
#   summarize(n_apps = n(),
#             n_success = sum(ifelse(status_type == "SELECTED", 1, 0)),
#             across(c(modal_msaname, modal_msastate, n_by_msa, modal_msaworksitename,
#                      modal_msaworksitestate, n_by_msa_worksite), first),
#             fraud_ratio = max(fraud_ratio, na.rm=TRUE),
#             last_lot_year = max(lottery_year)
#             ) %>%
#   mutate(share_msa = n_by_msa/n_apps,
#          share_msa_worksite = n_by_msa_worksite/n_success) %>% ungroup() %>%
#   # merging to get state names from abbreviations
#   left_join(state_cw %>% rename(modal_msastatename = statename), 
#             by = c("modal_msastate" = "stateabb")) %>%
#   left_join(state_cw %>% rename(modal_msaworksitestatename = statename), 
#             by = c("modal_msaworksitestate" = "stateabb"))
# # TODO: filter out 'fake' applications/duplicates/high app/emp ratio 
# 
# # TESTING MERGE ----
# revelio_emp <- revelio_raw %>%
#   select(c(company, rcid, n, n_users, recent_start, recent_end, top_metro_area, top_state, lei,
#            ultimate_parent_company_name, naics_code))
# 
# foia_samp_ids <- sample_n(foia_emp, 100) %>% select(c(company_FOIA, FEIN))
# revelio_samp <- sample_n(revelio_emp, 100000)
# 
# extra_words <- c("inc", "ltd", "plc", "pllc", "mso", "svc", "ggmbh", "gmbh", "limited", "pvt", "md", "private",
#                  "llp", "pc", "corporation", "corp", "intl", "international", "group", "co", "usa", "na", "management", 
#                  "global", "america", )
# clean_emp_name <- function(df, empnamecol){
#   # step zero: convert to lowercase and remove symbols
#   df[["empnameraw"]] <- str_remove(str_to_lower(df[[empnamecol]]), "™|®|©")
#   suffix_regex <- "\\s*,?\\s+(l\\.?\\s?l\\.?\\s?c\\.?|corp|inc|ltd|plc|pllc|mso|svc|gmbh|&?\\s?co|limited|pvt|llp)\\.?\\s*$"
#   df_out <- df %>%
#     mutate(
#       # step one: convert characters to latin ascii (replace accented characters), remove parentheticals or dividers at end of name
#       empnameclean = stringi::stri_trans_general(str_remove(empnameraw, "\\(.*\\)|\\|.*$"), "Latin-ASCII"),
#       # step two: remove any common suffixes, replace '&' or '+' with and
#       empnamestub = str_replace_all(str_remove(str_remove(empnameclean, suffix_regex), suffix_regex),
#                                       "\\s?&\\s?|\\s\\+\\s", " and "),
#       # step three: remove all parentheticals, removing all symbols and spaces
#       empnamebase = str_remove(empnamestub, "\\(.*\\)") %>%
#              str_remove_all("[^A-z0-9]"),
#       # step four: extracting website if exists from raw string
#       empnamesite = str_match(empnameraw, "([A-z0-9\\-]+\\.[a-z]{3})(?:^A-z0-9|$)")[,2]
#     )
#   return(df_out)
# }
# 
# foia_samp <- foia_emp %>% filter(company_FOIA %in% foia_samp_ids$company_FOIA & FEIN %in% foia_samp_ids$FEIN)
# foia_emp_clean <- clean_emp_name(foia_samp, "company_FOIA")
# revelio_emp_clean <- clean_emp_name(revelio_emp , "company") %>% arrange(n_users)
# 
# exact_match <- matchfunc("empnameraw")
# stub_match <- matchfunc()
# base_match <- matchfunc("empnamebase")
# 
# 
# exact_match <- left_join(foia_emp_clean,
#                    revelio_emp_clean,
#                    by = c("empnameraw")) %>%
#   mutate(statematch = case_when(is.na(top_state) ~ -1,
#                                 modal_msastatename == top_state ~ 1,
#                                 is.na(modal_msaworksitestatename) | modal_msaworksitestatename != top_state ~ 0,
#                                 modal_msaworksitestatename == top_state ~ 1,
#                                 TRUE ~ 0)) %>%
#   group_by(employer_name) %>% mutate(statematchind = max(statematch, na.rm=TRUE))
# 
# # counting number of matches with and without matching state
# print(glue("Total number of companies matched: {length(unique(filter(exact_match, !is.na(company))$employer_name))}"))
# print(glue("Total number of companies matched with state mismatch: {length(unique(filter(exact_match, statematchind == 0)$employer_name))}"))
# 
# stub_match <- left_join(foia_emp_clean,
#                    revelio_emp_clean,
#                    by = c("empnamestub")) %>%
#   mutate(statematch = case_when(is.na(top_state) ~ -1,
#                                 modal_msastatename == top_state ~ 1,
#                                 is.na(modal_msaworksitestatename) | modal_msaworksitestatename != top_state ~ 0,
#                                 modal_msaworksitestatename == top_state ~ 1,
#                                 TRUE ~ 0)) %>%
#   group_by(employer_name) %>% mutate(statematchind = max(statematch, na.rm=TRUE))
# 
# # View(test2 %>% select(employer_name, empnamestub, company, n_apps, n, n_users, lei, modal_msaname, modal_msastate,
# #                       top_metro_area, top_state,
# #                       fraud_ratio, last_lot_year, recent_start, recent_end))
# 
# base_match <- left_join(foia_emp_clean,
#                    revelio_emp_clean,
#                    by = c("empnamebase")) %>%
#   mutate(statematch = case_when(is.na(top_state) ~ -1,
#                                 modal_msastatename == top_state ~ 1,
#                                 is.na(modal_msaworksitestatename) | modal_msaworksitestatename != top_state ~ 0,
#                                 modal_msaworksitestatename == top_state ~ 1,
#                                 TRUE ~ 0)) %>%
#   group_by(employer_name) %>% mutate(statematchind = max(statematch, na.rm=TRUE))
# 
# View(base_match %>% select(employer_name, company, n_apps, n, n_users, modal_msaname, modal_msastate,
#                            top_metro_area, top_state,
#                       fraud_ratio, last_lot_year, recent_start, recent_end,
#                       modal_msaworksitename, modal_msaworksitestate, lei))
# 
# 
# 
# write_csv(base_match, glue("{root}/data/int/base_match_test_26feb2025.csv"))





