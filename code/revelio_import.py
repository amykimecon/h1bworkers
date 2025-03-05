# File Description: Importing Revelio Data for Easier Manipulation Using WRDS API
# Author: Amy Kim
# Date Created: Thu Feb 27

# Imports and Paths
import wrds
import pandas as pd
import duckdb
import time
root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

# Connecting to WRDS
db = wrds.Connection(wrds_username='amykimecon')
## creating pgpass file (only need to do first time)
#db.create_pgpass_file()

# Listing tables in WRDS data
db.list_tables(library='revelio')

# Listing columns and nrow of company and positions data
# db.describe_table(library='revelio', table='company_mapping')
# db.describe_table(library='revelio', table='individual_positions')

# Querying data
## Grouping individual positions data by rcid to get number of positions globally, in US, post-2020; latest start and end dates, joining to company mapping data
testlim = -1
if (testlim == -1):
    limstring = ""
else:
    limstring = f"LIMIT {testlim}"

## Query to group all individual positions by employer, count number of positions total, in US, post-2020; latest start and end dates
query_byemp = f"""SELECT rcid, COUNT(*) AS n_positions_global, COUNT(DISTINCT user_id) AS n_users_global,
                        COUNT(CASE WHEN country = 'United States' THEN 1 END) AS n_positions_us,
                        COUNT(DISTINCT (CASE WHEN country = 'United States' THEN user_id END)) AS n_users_us,
                        COUNT(CASE WHEN startdate > '2020-01-01' THEN 1 END) AS n_positions_recent,
                        COUNT(DISTINCT (CASE WHEN startdate > '2020-01-01' THEN user_id END)) AS n_users_recent,
                        MAX(startdate) AS recent_startdate_global, MAX(enddate) AS recent_enddate_global
                        FROM revelio.individual_positions GROUP BY rcid {limstring}"""

## Query to group all individual positions in US by employer and location, count number of positions; latest start and end dates
query_byemp_byloc = f"""SELECT rcid, state, metro_area, COUNT(*) AS n_positions, COUNT(DISTINCT user_id) AS n_users, 
                            MAX(startdate) AS recent_startdate, MAX(enddate) AS recent_enddate
                        FROM revelio.individual_positions  
                        WHERE Country = 'United States' 
                        GROUP BY rcid, state, metro_area {limstring}"""

## Agg Query: all companies in positions data (one row for companies with no US postions, row per location with > 20% of US positions or highest number of positions for companies with US positions)
query_agg = f"""SELECT * FROM (SELECT a.rcid AS rcid, company, year_founded, factset_entity_id, cusip, lei, naics_code, ultimate_parent_rcid,
                                ultimate_parent_company_name, state, metro_area, n_positions, n_users, n_positions_global, 
                                n_users_global, n_positions_us, n_users_us, 
                                n_positions_recent, n_users_recent, 
                                ROW_NUMBER() OVER (PARTITION BY a.rcid ORDER BY n_positions DESC) AS rank,
                                n_positions/n_positions_us AS share_positions_us,
                                recent_startdate, recent_enddate, recent_startdate_global, recent_enddate_global FROM 
                                (({query_byemp}) AS a
                                    LEFT JOIN ({query_byemp_byloc}) AS b
                                    ON a.rcid = b.rcid
                                    LEFT JOIN revelio.company_mapping AS c
                                    ON a.rcid = c.rcid)
                                ) """ #WHERE n_positions >= 0.2*n_positions_us OR rank = 1"""

start = time.time()
data = db.raw_sql(query_agg)
print(data.shape)
data.to_parquet(f"{root}/data/int/revelio/revelio_agg.parquet")
end = time.time()
print(f"Time to query and save: {end-start}")

db.close()
