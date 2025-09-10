# File Description: Getting Revelio Data for Merged Companies
# Author: Amy Kim
# Date Created: Sep 2 2025

# Imports and Paths
import wrds
import duckdb as ddb
import pandas as pd
import time
import numpy as np
import datetime
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

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

# toggle for testing
test = False

#####################
# DEFINING USER SAMPLE
#####################
rcids = list(con.sql("SELECT rcid FROM rmerge_w_dups, GROUP BY rcid").df()['rcid'])

if test:
    rcidsamp = rcids[:20]
else:
    rcidsamp = rcids 

userids = db.raw_sql(
f"""
    SELECT user_id FROM revelio.individual_positions WHERE country = 'United States' AND rcid IN ({','.join([str(i) for i in rcidsamp])}) GROUP BY user_id ORDER BY user_id
""")

#####################
# HELPERS
# #####################
# function to get relevant position data given list of usernames
def get_merge_query(userids, db = db):
    userid_subset = ','.join(userids['user_id'].astype('str'))

    user_info = db.raw_sql(
        f"""SELECT CASE WHEN a.user_id IS NOT NULL THEN a.user_id ELSE b.user_id END AS user_id, 
            fullname, profile_linkedin_url, user_location, user_country, f_prob, updated_dt, university_name, rsid, education_number, ed_startdate, ed_enddate, degree, field, university_country, university_location, university_raw, degree_raw, field_raw, description 
        FROM (
            (SELECT user_id, fullname, profile_linkedin_url, user_location, user_country, f_prob, updated_dt FROM revelio.individual_user WHERE user_id IN ({userid_subset})) AS a
            FULL JOIN (
                SELECT b_educ.user_id, b_educ.education_number, university_name, rsid, ed_startdate, ed_enddate, degree, field, university_country, university_location, university_raw, degree_raw, field_raw, description FROM (
                        (SELECT user_id, university_name, rsid, education_number, startdate AS ed_startdate, enddate AS ed_enddate, degree, field, university_country, university_location FROM revelio.individual_user_education WHERE user_id IN ({userid_subset})) AS b_educ
                    LEFT JOIN 
                        (SELECT user_id, university_raw, education_number, degree_raw, field_raw, description FROM revelio.individual_user_education_raw) AS b_educ_raw 
                    ON b_educ.user_id = b_educ_raw.user_id AND b_educ.education_number = b_educ_raw.education_number
                )
            ) AS b
            ON a.user_id = b.user_id)
        """)

    return user_info


#####################
# QUERYING WRDS
#####################
saveloc = f"{root}/data/int/wrds_users/wrds_users"
outfileloc = f"{root}/data/int/wrds_users_sep2.parquet"

if test:
    saveloc = None 
    outfileloc = ""

j = 20

t0_0 = time.time()
print(f"Current Time: {datetime.datetime.now()}")
print(f"Running wrds_positions on {userids.shape[0]} userids")
print("---------------------")

# running chunks and saving
print("Querying and saving individual chunks...")
help.chunk_query(userids, j = j, fun = get_merge_query, d = 10000, verbose = True, extraverbose=test, outpath = saveloc)

t1_1 = time.time()
print(f"Done! Time Elapsed: {round((t1_1-t0_0)/3600, 2)} hours")

# getting merged chunks
out = help.chunk_merge(saveloc, j = j, outfile = outfileloc, verbose = True)

print(f"Script Ended: {datetime.datetime.now()}")

# x=userids.merge(temp_out.groupby('user_id').size().reset_index().assign(out='yes'), how = 'outer', on = 'user_id')
# x.loc[pd.isnull(x['out'])==True]