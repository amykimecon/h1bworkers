# File Description: Exploring Revelio Data with WRDS API
# Author: Amy Kim
# Date Created: Mon Feb 24 09:37:53 2025

# IMPORTING PACKAGES ----
library(tidyverse)
library(glue)
library(RPostgres)

# SETTING WORKING DIRECTORIES ----
root <- "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code <- "/Users/amykim/Documents/GitHub/h1bworkers/code"

wrds <- dbConnect(Postgres(),
                  host='wrds-pgdata.wharton.upenn.edu',
                  port=9737,
                  dbname='wrds',
                  sslmode='require',
                  user='amykimecon')

# List all tables in revelio data
tablereq <- dbSendQuery(wrds, "select distinct table_name from 
                            information_schema.columns
                            where table_schema = 'revelio' 
                            order by table_name")
table_names <- dbFetch(tablereq, 
                n = -1)
dbClearResult(tablereq)

# List columns + nrow in relevant tables
for (name in table_names$table_name){
  print(glue("Table Name: {name}"))
  print(get_names(name))
  nrow = get_nrow(name)
  print(glue("n = {nrow}"))
}

# get_head("individual_positions")
# get_nrow("company_mapping")

company_mapping <- get_head("company_mapping", n = -1)
us_company_mapping <- get_head("company_mapping", n = -1,
                               filter = " where rcid in (select rcid from revelio.individual_positions 
                               where country = 'United States')")
write_csv(us_company_mapping, glue("{root}/data/int/revelio/us_company_mapping.csv"))

get_names <- function(tablename){
  res <- dbSendQuery(wrds, glue("select column_name from 
                            information_schema.columns
                            where table_schema = 'revelio' 
                            and table_name = '{tablename}' "))
  names_out <- dbFetch(res, n = -1)
  dbClearResult(res)
  return(names_out$column_name)
}

get_nrow <- function(tablename, filter = ""){
  names <- get_names(tablename)
  res <- dbSendQuery(wrds, glue("select count({names[1]}) from 
                                revelio.{tablename} 
                                {filter}"))
  n_out <- dbFetch(res, n = 5)
  dbClearResult(res)
  return(formatC(as.numeric(n_out$count), format = 'f', big.mark = ',', digits = 0))
}

get_head <- function(tablename, n = 10, filter = ""){
  res <- dbSendQuery(wrds, glue("select * from 
                     revelio.{tablename}
                     {filter}"))
  head_out <- dbFetch(res, n = n)
  dbClearResult(res)
  return(head_out)
}
