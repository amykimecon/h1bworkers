# File Description: Helper functions for Revelio/FOIA data merge
# Author: Amy Kim
# Date Created: Wed Feb 26 11:56:48 2025

# IMPORTING PACKAGES ----
library(tidyverse)
library(glue)
library(httr)
library(readxl)
library(RPostgres)

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
matchfunc <- function(matchtype = "empnamestub", dbfoia = foia_emp_clean, dbrev = revelio_emp_clean){
  matchdf <- left_join(dbfoia, dbrev, by = c(matchtype)) %>%
    mutate(statematch = case_when(is.na(top_state) ~ -1,
                                  modal_msastatename == top_state ~ 1,
                                  is.na(modal_msaworksitestatename) | modal_msaworksitestatename != top_state ~ 0,
                                  modal_msaworksitestatename == top_state ~ 1,
                                  TRUE ~ 0)) %>%
    group_by(employer_name) %>% mutate(statematchind = max(statematch, na.rm=TRUE))
  print(glue("Total number of companies matched: {length(unique(filter(matchdf, !is.na(company))$employer_name))}"))
  print(glue("Total number of companies matched with state mismatch: {length(unique(filter(matchdf, statematchind == 0)$employer_name))}"))
  return(matchdf)
}

# VIEWERS/UTILITIES ----
## look at relevant columns of matched df
matchedview <- function(matchdf){
  View(base_match %>% select(employer_name, company, statematchind, n_apps, n_success, n, n_users, modal_msaname, modal_msastate,
                             top_metro_area, top_state,
                             fraud_ratio, last_lot_year, recent_start, recent_end,
                             modal_msaworksitename, modal_msaworksitestate, lei))
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

