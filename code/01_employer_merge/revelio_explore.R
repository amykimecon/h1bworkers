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

source(glue("{code}/merge_helper.R"))

wrds <- dbConnect(Postgres(),
                  host='wrds-pgdata.wharton.upenn.edu',
                  port=9737,
                  dbname='wrds',
                  sslmode='require',
                  user='amykimecon')

# EXPLORE ----
# # List all tables in revelio data
# tablereq <- dbSendQuery(wrds, "select distinct table_name from
#                             information_schema.columns
#                             where table_schema = 'revelio'
#                             order by table_name")
# table_names <- dbFetch(tablereq,
#                        n = -1)
# dbClearResult(tablereq)
# 
# # List columns + nrow in relevant tables
# for (name in table_names$table_name){
#   print(glue("Table Name: {name}"))
#   print(get_names(name))
#   nrow = get_nrow(name)
#   print(glue("n = {nrow}"))
# }

# get_head("individual_positions")
# get_nrow("company_mapping")

# GET US COMPANIES LIST
#company_mapping <- get_head("company_mapping", n = -1)
# us_company_mapping <- get_head("company_mapping", n = -1,
#                                filter = " where rcid in (select rcid from revelio.individual_positions 
#                                where country = 'United States')")
# write_csv(us_company_mapping, glue("{root}/data/int/revelio/us_company_mapping.csv"))

# GET US COMPANIES BY NUMBER OF POSITIONS
## sql: 
##      STEP ONE: group by rcid
##          get position and user counts and latest start/end dates by rcid
##      STEP TWO: join with top metro area by rcid
##          innermost select groups positions to metro area x rcid level, gets counts
##          then compute rank of each city (within rcid) by number of positions (n)
##          then select only top rank (for each rcid, city with max number of positions)
##      STEP THREE: join with company info

# res <- dbSendQuery(wrds, glue("select * from ((select count(*) as n, count(distinct user_id) as n_users,
#                                         max(startdate) as recent_start,
#                                         max(enddate) as recent_end, rcid as rcid_positions
#                                         from revelio.individual_positions
#                                         where country = 'United States'
#                                         group by rcid) as company_counts
#                                     left join (select rcid as rcid_metros, top_metro_area, top_state from (
#                                         select n, metro_area as top_metro_area, 
#                                         state as top_state, rcid,
#                                         row_number() over (partition by rcid order by n desc) as r
#                                         from (
#                                           select count(*) as n, metro_area, state, rcid 
#                                           from revelio.individual_positions
#                                           where country = 'United States'
#                                           group by rcid, state, metro_area
#                                         ) 
#                                       )
#                                       where r = 1) as company_metros 
#                                     on company_counts.rcid_positions = company_metros.rcid_metros) as positions
#                                 left join (select * from revelio.company_mapping) as companies
#                                 on positions.rcid_positions = companies.rcid")
#                     )
# n_pos_locs <- dbFetch(res, n = -1)
# dbClearResult(res)
# write_csv(n_pos_locs, glue("{root}/data/int/revelio/companies_by_positions_locations.csv"))

# SAME AS ABOVE BUT ONLY RECENT (post-2020) POSITIONS
res <- dbSendQuery(wrds, glue("select * from ((select count(*) as n, count(distinct user_id) as n_users,
                                        max(startdate) as recent_start,
                                        max(enddate) as recent_end, rcid as rcid_positions
                                        from revelio.individual_positions
                                        where country = 'United States' and startdate > '2020-01-01'
                                        group by rcid) as company_counts
                                    left join (select rcid as rcid_metros, top_metro_area, top_state from (
                                        select n, metro_area as top_metro_area,
                                        state as top_state, rcid,
                                        row_number() over (partition by rcid order by n desc) as r
                                        from (
                                          select count(*) as n, metro_area, state, rcid
                                          from revelio.individual_positions
                                          where country = 'United States' and startdate > '2020-01-01'
                                          group by rcid, state, metro_area
                                        )
                                      )
                                      where r = 1) as company_metros
                                    on company_counts.rcid_positions = company_metros.rcid_metros) as positions
                                left join (select * from revelio.company_mapping) as companies
                                on positions.rcid_positions = companies.rcid")
                    )
n_pos_locs_recent <- dbFetch(res, n = -1)
dbClearResult(res)
write_csv(n_pos_locs_recent, glue("{root}/data/int/revelio/companies_by_positions_locations_recent.csv"))


# # GET FULL LIST OF METRO AREAS IN REVELIO DATA
# met_areas_res <- dbSendQuery(wrds, glue("select metro_area, state from revelio.individual_positions
#                                         where country = 'United States'"))
# met_areas <- dbFetch(met_areas_res, n = 10000000) 
# dbClearResult(met_areas_res)
# 
# met_areas_list <- met_areas %>% distinct()          
# write_csv(met_areas_list, glue("{root}/data/int/revelio/metro_areas.csv"))
# 
# # LOOKING AT RAW LOCATIONS
# res <- dbSendQuery(wrds, glue("select * from (select position_id, metro_area, state from revelio.individual_positions
#                                                 where country = 'United States' limit 1000) as positions 
#                                   left join (select position_id as position_id_raw, location_raw 
#                                                 from revelio.individual_positions_raw) as positions_raw 
#                                 on positions.position_id = positions_raw.position_id_raw"))
# out <- dbFetch(res, n = -1)
# dbClearResult(res)
                       
# )
#                               select count(*) as n, count(distinct user_id) as n_users,
#                                     max(startdate) as recent_start,
#                                     max(enddate) as recent_end, metro_area, state, rcid 
#                                     from revelio.individual_positions
#                                     where country = 'United States'
#                                     group by rcid, state, metro_area
#                               from (select count(*) as n, count(distinct user_id) as n_users,
#                                     max(startdate) as recent_start,
#                                     max(enddate) as recent_end, metro_area, state, rcid 
#                                     from revelio.individual_positions
#                                     where country = 'United States'
#                                     group by rcid, state, metro_area) 
#                               )
#                                   as positions 
#                               left join (select * from revelio.company_mapping) as companies
#                               on positions.rcid = companies.rcid"))
# n_pos <- dbFetch(res, n = -1)
# dbClearResult(res)
# write_csv(n_pos, glue("{root}/data/int/revelio/companies_by_positions.csv"))
# 
# 
# lucid <- get_head("individual_positions", n = -1, filter="where rcid in (637889, 937009, 4751812, 4751816, 14961712, 93153212)")
# biohelix <- get_head("individual_positions", n = -1, filter="where rcid in (637889, 937009, 4751812, 4751816, 14961712, 93153212)")
# 
# # JOIN COMPANIES LIST AND POSITIONS
# # 
# # 
# # test with roles
# res <- dbSendQuery(wrds, glue("select * from (select count(*) as n, count(distinct role_k300) as n_roles,
#                               max(role_k1500) as role_k1500, job_category from
#                               revelio.individual_role_lookup
#                               group by job_category) as a
#                               left join (select * from revelio.individual_positions limit 1000) as b
#                               on a.role_k1500 = b.role_k1500"))
# 
# res <- dbSendQuery(wrds, glue("select job_category, n from (select job_category, role_k50, n_roles, n,
#                               row_number() over (partition by job_category order by n desc) as r
#                               from (select count(*) as n, count(distinct role_k300) as n_roles,
#                               max(role_k1500) as role_k1500, job_category, role_k50 from
#                               revelio.individual_role_lookup
#                               group by job_category, role_k50))
#                               where r = 1"))
# dbFetch(res, n = -1)
# dbClearResult(res)








