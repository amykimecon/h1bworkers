# File Description: Getting Revelio Data for Merged Companies
# Author: Amy Kim
# Date Created: Thurs Mar 20

# Imports and Paths
import wrds
import duckdb as ddb
import time
import pandas as pd
import random
import fiscalyear
import employer_merge_helpers as emh

root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

con = ddb.connect()

#####################
# IMPORTING DATA
#####################
## duplicate rcids
dup_rcids = con.read_csv(f"{root}/data/int/dup_rcids_mar20.csv")
con.sql("CREATE OR REPLACE TABLE dup_rcids AS SELECT main_rcid, rcid FROM dup_rcids, GROUP BY main_rcid, rcid")

## Importing R Matched data
rmerge = con.read_csv(f"{root}/data/int/good_match_ids_mar20.csv")
# merging with dup rcids to get full list of matched rcids
con.sql("CREATE OR REPLACE TABLE rmerge_w_dups AS SELECT a.main_rcid AS main_rcid, CASE WHEN b.rcid IS NULL THEN a.rcid ELSE b.rcid END AS rcid, FEIN FROM (rmerge AS a LEFT JOIN dup_rcids AS b ON a.main_rcid = b.main_rcid)")

## WRDS
db = wrds.Connection(wrds_username='amykimecon')

#####################
# QUERYING WRDS
#####################
rcids = list(con.sql("SELECT rcid FROM rmerge_w_dups GROUP BY rcid").df()['rcid'])

# # GETTING INDIVIDUAL POSITIONS OF MATCHED RCIDS
# merged_pos = db.raw_sql(f"SELECT * FROM (SELECT * FROM revelio.individual_positions AS a JOIN (SELECT * FROM revelio.company_mapping WHERE rcid IN ({','.join([str(i) for i in rcids])})) AS b ON a.rcid = b.rcid)")
mergedb = db.raw_sql(f"""SELECT a.user_id AS user_id, a.position_id AS a, country, state, metro_area, a.startdate AS pos_startdate, a.enddate AS pos_enddate, role_k1500, weight, start_salary, end_salary, seniority, salary, position_number, rcid, total_compensation, fullname, highest_degree, sex_predicted, ethnicity_predicted, user_location, user_country, updated_dt, university_name, c.education_number, c.startdate AS ed_startdate, c.enddate AS ed_enddate, degree, field, university_country, title_raw, university_raw, degree_raw, field_raw FROM (SELECT * FROM revelio.individual_positions WHERE country = 'United States' AND rcid IN ({','.join([str(i) for i in rcids])})) AS a LEFT JOIN (SELECT * FROM revelio.individual_user) AS b ON a.user_id = b.user_id LEFT JOIN (SELECT * FROM revelio.individual_user_education) AS c ON a.user_id = c.user_id LEFT JOIN (SELECT position_id, company_raw, location_raw, title_raw, description FROM revelio.individual_positions_raw) AS d ON a.position_id = d.position_id LEFT JOIN (SELECT * FROM revelio.individual_user_education_raw) AS e ON c.user_id = e.user_id AND c.education_number=e.education_number""")

mergedb.to_parquet("rev_merge.parquet")
