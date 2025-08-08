# File Description: Getting Revelio Data for Merged Companies
# Author: Amy Kim
# Date Created: Thurs Mar 20

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
    rcidsamp = rcids[:10]
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
    user_pos = db.raw_sql(f"SELECT a.user_id, a.position_id, a.position_number, rcid, country, startdate, enddate, role_k1500, salary, total_compensation, company_raw, title_raw FROM revelio.individual_positions AS a LEFT JOIN revelio.individual_positions_raw AS b ON a.position_id = b.position_id WHERE a.user_id IN ({userid_subset})")

    return user_pos 


#####################
# QUERYING WRDS
#####################
saveloc = f"{root}/data/int/wrds_positions/wrds_positions"
j = 20

t0_0 = time.time()
print(f"Current Time: {datetime.datetime.now()}")
print(f"Running wrds_positions on {userids.shape[0]} userids")
print("---------------------")

# running chunks and saving
print("Querying and saving individual chunks...")
help.chunk_query(userids, j = j, fun = get_merge_query, d = 10000, verbose = True, extraverbose=False, outpath = saveloc)

t1_1 = time.time()
print(f"Done! Time Elapsed: {round((t1_1-t0_0)/3600, 2)} hours")

# getting merged chunks
out = help.chunk_merge(saveloc, j = j, outfile = f"{root}/data/int/wrds_positions_aug1.parquet", verbose = True)

print(f"Script Ended: {datetime.datetime.now()}")
