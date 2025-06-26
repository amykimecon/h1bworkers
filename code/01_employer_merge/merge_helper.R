# File Description: Helper functions for Revelio/FOIA data merge
# Author: Amy Kim
# Date Created: Wed Feb 26 11:56:48 2025

# IMPORTING PACKAGES ----
library(tidyverse)
library(glue)
library(httr)
library(readxl)
library(RPostgres)
library(stringdist)

# SETTING WORKING DIRECTORIES ----
root <- "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code <- "/Users/amykim/Documents/GitHub/h1bworkers/code"

# REVELIO WRDS QUERIES ----
## get colnames of table in revelio data
get_names <- function(tablename){
  res <- dbSendQuery(wrds, glue("select column_name from 
                            information_schema.columns
                            where table_schema = 'revelio' 
                            and table_name = '{tablename}' "))
  names_out <- dbFetch(res, n = -1)
  dbClearResult(res)
  return(names_out$column_name)
}

## get number of rows of table in revelio data with optional filter (must start with 'where')
get_nrow <- function(tablename, filter = ""){
  names <- get_names(tablename)
  res <- dbSendQuery(wrds, glue("select count({names[1]}) from 
                                revelio.{tablename} 
                                {filter}"))
  n_out <- dbFetch(res, n = 5)
  dbClearResult(res)
  return(formatC(as.numeric(n_out$count), format = 'f', big.mark = ',', digits = 0))
}

## get first n rows of a table in revelio data with optional filter (must start with 'where')
get_head <- function(tablename, n = 10, filter = ""){
  res <- dbSendQuery(wrds, glue("select * from 
                     revelio.{tablename}
                     {filter}"))
  head_out <- dbFetch(res, n = n)
  dbClearResult(res)
  return(head_out)
}

# FOIA MERGE ----
matchfunc <- function(matchtype = c("empnamestub"), dbfoia = foia_emp_clean, dbrev = revelio_emp_clean){
  matchdf <- left_join(dbfoia, dbrev, by = matchtype, relationship = "many-to-many") %>%
    mutate(statematch = case_when(is.na(top_state) ~ -1,
                                  top_emp_state == top_state ~ 1,
                                  is.na(top_worksite_state) | top_worksite_state != top_state ~ 0,
                                  top_worksite_state == top_state ~ 1,
                                  TRUE ~ 0),
           naicsmatch = case_when(is.na(top_naics) | is.na(naics_code) | top_naics == 999999 ~ -1,
                                  top_naics == naics_code ~ 1,
                                  TRUE ~ 0),
           top_naics2raw = substr(top_naics, 1, 2),
           top_naics2 = case_when(top_naics2raw %in% c("31", "32", "33") ~ "31",
                                  top_naics2raw %in% c("44", "45") ~ "44",
                                  top_naics2raw %in% c("48", "49") ~ "48",
                                  TRUE ~ top_naics2raw),
           naics_code2raw = substr(naics_code, 1, 2),
           naics_code2 = case_when(naics_code2raw %in% c("31", "32", "33") ~ "31",
                                   naics_code2raw %in% c("44", "45") ~ "44",
                                   naics_code2raw %in% c("48", "49") ~ "48",
                                   TRUE ~ naics_code2raw),
           naics2match = case_when(is.na(top_naics2) | is.na(naics_code2) | top_naics2 == "99" ~ -1,
                                  top_naics2 == naics_code2 ~ 1,
                                  TRUE ~ 0)) %>%
    group_by(company_FOIA) %>% mutate(statematchind = max(statematch, na.rm=TRUE),
                                       naicsmatchind = max(naicsmatch, na.rm=TRUE),
                                      naics2matchind = max(naics2match, na.rm=TRUE))
  print(glue("Total number of companies matched: {length(unique(filter(matchdf, !is.na(company))$id.x))}"))
  print(glue("Total number of companies matched with state or naics match: {length(unique(filter(matchdf, statematchind == 1 | naics2matchind == 1)$id.x))}"))
  #print(glue("Total number of companies matched with state and naics mismatch: {length(unique(filter(matchdf, statematchind == 0 & naicsmatchind == 0)$company_FOIA))}"))
  
  return(matchdf)
}

# VIEWERS/UTILITIES ----
## look at relevant columns of matched df
matchedview <- function(matchdf){
  View(matchdf %>% select(id.x, ftot, lottery_year, company_FOIA, company_rev, statematch, naics2match, n_apps, n_success, n, n_users, 
                          top_naics, naics_code, top_naics2, naics_code2, top_emp_state, top_worksite_state,
                             top_state,
                             fraud_ratio, recent_start, recent_end, lei))
}

## look for keyword match (up to 4) in company name in revelio data
rev_match <- function(keyword, keyword2 = "", keyword3 = "", keyword4 = "", df = revelio_emp, view = TRUE){
  df_out <- kw_match("company", df=df, keyword=keyword, keyword2=keyword2, 
                     keyword3=keyword3, keyword4=keyword4, view=view)
}

## look for keyword match (up to 4) in company name in foia data
foia_match <- function(keyword, keyword2 = "", keyword3 = "", keyword4 = "", df = foia_emp, view = TRUE){
  df_out <- kw_match("employer_name", df=df, keyword=keyword, keyword2=keyword2, 
                     keyword3=keyword3, keyword4=keyword4, view=view)
}

## keyword match base function: looks for match in specified column name
kw_match <- function(colname, df, keyword, keyword2 = "", keyword3 = "", keyword4 = "", view = TRUE){
  df[["searchcol"]] <- df[[colname]]
  df_out <- df %>% filter(str_detect(str_to_lower(searchcol), keyword))
  for (kw in c(keyword2, keyword3, keyword4)){
    if (kw != ""){
      df_out <- df_out %>% filter(str_detect(str_to_lower(searchcol), kw))
    }
  }
  if (view){
    View(df_out)
  }
  else{
    return(df_out %>% select(-c(searchcol)))
  }
}

