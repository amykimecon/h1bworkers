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
  filter(FEIN != "(b)(3) (b)(6) (b)(7)(c)") %>%
  # getting total number of applications and cleaning name (lowercase + remove commas and periods)
  mutate(n_tot_fein_yr = n(),
         company_FOIA = str_to_lower(str_remove_all(employer_name, ",|\\.|'")),
         valid_app = ifelse(ben_multi_reg_ind == 0, 1, 0)
  ) %>% ungroup() %>%
  # filtering duplicate applications
  #filter(ben_multi_reg_ind == 0) %>%
  # indicator for likely fraud (high ratio of applications to current employees in us)
  mutate(fraud_ratio = ifelse(is.na(NUM_OF_EMP_IN_US) | NUM_OF_EMP_IN_US == 0, -1,
                              n_tot_fein_yr/NUM_OF_EMP_IN_US)) %>%
  # grouping by employer location, then employer (getting modal employer state)
  group_by(FEIN, state_name) %>%
  mutate(n_by_state = sum(valid_app)) %>% arrange(desc(n_by_state)) %>% ungroup() %>%
  group_by(FEIN) %>%
  mutate(top_emp_state = first(state_name),
         n_by_state = first(n_by_state)) %>% ungroup() %>%
  # grouping by worksite, then employer (getting modal worksite state)
  group_by(FEIN, worksite_state_name) %>%
  mutate(n_by_worksite_state = sum(ifelse(!is.na(worksite_state_name), valid_app, 0))) %>% arrange(desc(n_by_worksite_state)) %>% ungroup() %>%
  group_by(FEIN) %>%
  mutate(top_worksite_state = first(worksite_state_name),
         n_by_worksite_state = first(n_by_worksite_state)) %>% ungroup() %>%
  group_by(FEIN, NAICS_CODE) %>%
  mutate(n_by_naics = sum(ifelse(!is.na(NAICS_CODE), valid_app, 0))) %>% arrange(desc(n_by_naics)) %>% ungroup() %>%
  group_by(FEIN) %>%
  mutate(top_naics = first(NAICS_CODE),
         n_by_naics = first(n_by_naics)) %>% ungroup() %>%
  group_by(FEIN, company_FOIA, lottery_year) %>%
  summarize(n_apps = sum(valid_app),
            n_success = sum(ifelse(status_type == "SELECTED", valid_app, 0)),
            across(c(top_emp_state, top_worksite_state, n_by_state, n_by_worksite_state, top_naics, n_by_naics), first),
            fraud_ratio = max(fraud_ratio, na.rm=TRUE)
  ) %>%
  mutate(share_state = n_by_state/n_apps,
         share_worksite_state = n_by_worksite_state/n_success,
         share_naics = n_by_naics/n_success) %>% ungroup() %>%
  mutate(id=row_number())

# TESTING MERGE ----
revelio_emp <- revelio_raw %>%
  filter(!is.na(rcid)) %>%
  select(c(company, rcid, n, n_users, recent_start, recent_end, top_metro_area, top_state, lei,
           ultimate_parent_company_name, naics_code)) %>%
  mutate(id = row_number(),
         company_rev = str_to_lower(str_remove_all(company, ",|\\.|'")))

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

foia_names <- foia_emp %>% select(company_FOIA, FEIN) %>% distinct()
foia_names_clean <- clean_emp_name(foia_names, "company_FOIA")

revelio_names <- revelio_emp %>% select(company_rev, rcid)
revelio_names_clean <- clean_emp_name(revelio_names, "company_rev")

# #foia_samp <- foia_emp %>% filter(company_FOIA %in% foia_samp_ids$company_FOIA & FEIN %in% foia_samp_ids$FEIN)
# foia_emp_clean <- clean_emp_name(foia_emp, "company_FOIA")
# revelio_emp_clean <- clean_emp_name(revelio_emp , "company") #%>% arrange(n_users)

# tokenizing
tokens_long <- data.frame(empfortoken = c(foia_names_clean$empfortoken, revelio_names_clean$empfortoken)) %>%
  mutate(fullname = empfortoken) %>% #sample_n(10000) %>%
  separate_longer_delim(empfortoken, " ") %>%
  group_by(1) %>% mutate(tot = n(), token = empfortoken) %>% ungroup() %>%
  group_by(token) %>% mutate(freq = n()/tot) %>% ungroup() 

# getting frequency cutoff for top 100 most frequent tokens
token_freq_cutoff <- arrange(tokens_long %>% group_by(token) %>% summarize(freq = mean(freq)), desc(freq))$freq[100]

tokens_wide <- tokens_long %>% distinct(fullname, token, freq) %>% arrange(freq) %>% 
  filter(freq < token_freq_cutoff) %>% group_by(fullname) %>% mutate(rank = row_number()) %>% filter(rank <= 5) %>%
  pivot_wider(id_cols = c(fullname), names_prefix = "token", names_from = "rank", values_from = c("token", "freq"))

foia_emp_tokens <- foia_emp %>% left_join(foia_names_clean, by = c("company_FOIA", "FEIN")) %>%
  left_join(tokens_wide, by = c("empfortoken" = "fullname")) %>% filter(!is.na(token_token1))

revelio_emp_tokens <- revelio_emp %>% left_join(revelio_names_clean, by = c("rcid", "company_rev")) %>%
  left_join(tokens_wide, by = c("empfortoken" = "fullname")) %>% filter(!is.na(token_token1))


## matching on rarest and second rarest tokens
token_match <- matchfunc(matchtype = c("token_token1", "token_token2", "token_token3"), 
                         dbfoia = foia_emp_tokens, 
                         dbrev = revelio_emp_tokens)
write_csv(token_match, glue("{root}/data/int/token_match_r_mar20.csv"))

token_match_clean <- token_match %>% 
  filter(!is.na(rcid)) %>%
  rowwise() %>%
  mutate(f1 = log10(freq_token1.x),
         f2 = log10(freq_token2.x),
         f3 = log10(freq_token3.x),
         ftot = sum(f1, f2, f3, na.rm=TRUE)) %>%
  mutate(rawstringdist = stringdist(empnameraw.x, empnameraw.y),
         stubstringdist = stringdist(empnamestub.x, empnamestub.y),
         basestringdist = stringdist(empnamebase.x, empnamebase.y))

## looking for duplicate linkedin companies
token_dup <- token_match_clean %>% filter((basestringdist <= 2 & ftot < -6.9) |
                                            (f1 < -6.9) | (ftot < -15) | (f1 < -5.5 & ftot < -9)) %>%
  group_by(FEIN, lottery_year) %>% mutate(unique_rcid = n_distinct(rcid))


token_dup2 <- token_match_clean %>% filter((basestringdist <= 2 & ftot < -6.9) |
                                            (f1 < -6.9) | (ftot < -15) | (f1 < -5.5 & ftot < -9))

# this is a dataset that maps rcids to 'duplicate' rcids -- when outputting final matched
#   dataset, make sure to link back here 
dup_rcids <- token_dup2 %>%
  group_by(empnamebase.x) %>% mutate(unique_rcid = n_distinct(rcid)) %>% 
  filter(unique_rcid > 1 & unique_rcid < 4) %>%
  group_by(empnamebase.x, rcid, empnamebase.y, n) %>% summarize() %>%
  group_by(empnamebase.x) %>%
  arrange(desc(n)) %>% mutate(main_rcid = ifelse(row_number() == 1, rcid, NA)) %>%
  fill(main_rcid)

write_csv(dup_rcids, glue("{root}/data/int/dup_rcids_mar20.csv"))

## looking for duplicate FEINs
dup_feins <- token_dup2 %>% 
  group_by(rcid) %>% 
  mutate(unique_fein = n_distinct(FEIN)) %>%
  filter(unique_fein > 1) %>%
  group_by(rcid, company_rev, FEIN) %>% summarize(n_success = sum(n_success)) %>%
  group_by(rcid) %>% arrange(desc(n_success)) %>% mutate(main_fein = ifelse(row_number() == 1, FEIN, NA)) %>%
  fill(main_fein)

## good matches
## iding and removing dups
token_match_nodups <- token_match_clean %>% 
  left_join(dup_rcids %>% group_by(rcid, main_rcid) %>% summarize(), by = c("rcid")) %>%
  mutate(main_rcid = ifelse(is.na(main_rcid), rcid, main_rcid)) %>%
  group_by(id.x) %>%
  mutate(main_match = ifelse(rcid == main_rcid, 1, 0)) %>%
  filter(main_match | max(main_match) == 0)

### for 1-1 matches, keep all for now
single_match <- token_match_nodups %>% group_by(id.x) %>% mutate(nmatch = sum(ifelse(!is.na(company), 1, 0))) %>%
  filter(nmatch == 1) %>%
  rowwise() %>% mutate(recent_act = max(recent_start, recent_end, as.Date("2000-01-01"), na.rm=TRUE)) %>% ungroup() #%>%
#filter(n > 5 & recent_act >= as.Date("2020-01-01"))

### for many-1 matches
mult_match_all <- token_match_nodups %>% group_by(id.x) %>% mutate(nmatch = sum(ifelse(!is.na(company), 1, 0))) %>%
  filter(nmatch > 1) 
  
# best matches: exact match of stub name and state match and high n AND unique (after all those filters)
mult_match_certain <- mult_match_all %>%
  filter(empnamestub.x == empnamestub.y & n >= max(n)*0.5 & statematch == 1) %>%
  mutate(nmatch = n()) %>% filter(nmatch == 1)

# second best matches: among matches with stub name lev dist of 2 or less, keep companies with 5 or fewer matches
#   remaining (indicator of stub name lev dist being a good filter), then keep if naics code match OR 
#   n > 100 OR state match with sufficient n AND unique (after all those filters)
mult_match_certain2 <- mult_match_all %>% left_join(mult_match_certain %>% select(id.x) %>% mutate(certainmatch = 1)) %>%
  filter(is.na(certainmatch)) %>%
  filter(stubstringdist <= 2) %>%
  mutate(nmatch = n()) %>% 
  # filter(nmatch <= 5) %>% 
  filter(top_naics2 == naics_code2 | n > 100 | 
           (statematch == 1 & n >= 0.5*max(n))) %>%
  mutate(nmatch = n()) %>% filter(nmatch == 1)

# FINAL SET OF DECENT ONE-ONE MATCHES ----
good_matches <- bind_rows(list(
  single_match %>% filter(stubstringdist <= 2 | statematch == 1 | naics2match == 1) %>%
    mutate(matchtype = "single"),
  mult_match_certain %>% mutate(matchtype = "mult_high"),
  mult_match_certain2 %>% mutate(matchtype = "mult_med")#, mult_match3
)) %>% 
  group_by(main_rcid, lottery_year) %>%
  mutate(foia_id = cur_group_id())
# 
# dup_feins2 <- good_matches_all %>% group_by(rcid) %>% mutate(n_feins = n_distinct(FEIN)) %>%
#   filter(n_feins > 1) %>% group_by(rcid, company_rev, FEIN) %>% summarize(n_success = sum(n_success)) %>%
#   group_by(rcid) %>% arrange(desc(n_success)) %>% mutate(main_fein = ifelse(row_number() == 1, FEIN, NA)) %>%
#   fill(main_fein)
# 
# dup_feins_all <- bind_rows(list(dup_feins, dup_feins2)) %>%
#   group_by(FEIN, main_fein) %>% summarize()
# 
# write_csv(dup_feins_all, glue("{root}/data/int/dup_feins_mar19.csv"))

print(glue("Total FEINs: {length(unique(foia_emp$FEIN))}"))
print(glue("Matched FEINs: {length(unique(good_matches$FEIN))} ({round(length(unique(good_matches$FEIN))*100/length(unique(foia_emp$FEIN)), 1)}%)"))

unmatched <- filter(foia_emp, !(id %in% good_matches$id.x))
unmatched_fein <- filter(foia_emp, !(FEIN %in% good_matches$FEIN))

write_csv(good_matches %>% select(foia_id, FEIN, lottery_year, rcid, main_rcid, matchtype), 
          glue("{root}/data/int/good_match_ids_mar20.csv"))

# FILTERING TO LOTTERY YEAR 2021, REMOVING DUPLICATES
# goodmatch21 <- filter(raw, lottery_year == 2021) %>%
#   mutate(company_FOIA_original = str_to_lower(str_remove_all(employer_name, ",|\\."))) %>%
#   group_by(FEIN, company_FOIA_original) %>% summarize(n_foia=n()) %>%
#   left_join(good_matches %>% filter(lottery_year == 2021), by = c("FEIN")) %>%
#   left_join(dup_rcids %>% )
# 
# write_csv(good_matches, glue("{root}/data/int/good_matches_mar14.csv"))

# unique_goodmatch21 <- goodmatch21 %>% group_by(FEIN, rcid) %>%
#   summarize(n=n()) %>% ungroup() %>% group_by(FEIN) %>% mutate(nmatch = n()) %>%
#   filter(nmatch == 1)
# 
# filtered_goodmatch21 <- goodmatch21 %>% filter(!(FEIN %in% unique_goodmatch21$FEIN))
# 
# # drop duplicates with very different strings (if good string match exists)
# #   then drop duplicates with way fewer n (if high n match exists)
# x <- filtered_goodmatch21 %>% group_by(FEIN) %>%
#   mutate(strmatch = ifelse(!is.na(company_FOIA) & !is.na(company_FOIA_original) &
#                              stringdist(company_FOIA_original, company_FOIA) <= 2, 1, 0)) %>%
#   filter(strmatch == 1) %>%
#   filter(n >= 0.5*max(n) | max(n) < 50) %>%
#   group_by(FEIN) %>% mutate(unique_rcid = n_distinct(rcid))

# 
# %>%
#   mutate(strmatch = ifelse(!is.na(company_FOIA) & !is.na(company_FOIA_original) &
#                              stringdist(company_FOIA_original, company_FOIA) <= 1, 1, 0)) %>%
#   group_by(FEIN) %>%
#   mutate(strmatchind = max(strmatch)) %>%
#   filter(strmatch == 1 | strmatchind == 0) %>%
#   mutate(nmatch = n())
# 
# x <- filter(goodmatch21, is.na(rcid)) %>%
#   select(c(FEIN, company_FOIA)) %>% left_join(good_matches, by = "FEIN")

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





