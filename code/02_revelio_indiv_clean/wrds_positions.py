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
sys.path.append(os.path.dirname(__file__))
from config import *
import rev_indiv_config as rcfg

con = ddb.connect()
t_script0 = time.time()
print(f"Using config: {rcfg.ACTIVE_CONFIG_PATH}")

#####################
# IMPORTING DATA
#####################
## duplicate rcids
dup_rcids = con.read_csv(rcfg.DUP_RCIDS_CSV)
con.sql("CREATE OR REPLACE TABLE dup_rcids AS SELECT main_rcid, rcid FROM dup_rcids, GROUP BY main_rcid, rcid")

## Importing R Matched data
rmerge = con.read_csv(rcfg.GOOD_MATCH_IDS_CSV)
# merging with dup rcids to get full list of matched rcids
con.sql("CREATE OR REPLACE TABLE rmerge_w_dups AS SELECT a.main_rcid AS main_rcid, CASE WHEN b.rcid IS NULL THEN a.rcid ELSE b.rcid END AS rcid, FEIN FROM (rmerge AS a LEFT JOIN dup_rcids AS b ON a.main_rcid = b.main_rcid)")

## Importing valid LLM-reviewed crosswalk matches
foia_rcid_crosswalk = con.read_csv(
    rcfg.LLM_CROSSWALK_CSV,
    strict_mode=False,
    ignore_errors=True,
    all_varchar=True,
)
con.sql(
    """
    CREATE OR REPLACE TABLE all_target_rcids AS
    SELECT DISTINCT CAST(rcid AS BIGINT) AS rcid
    FROM rmerge_w_dups
    WHERE TRY_CAST(rcid AS BIGINT) IS NOT NULL
    UNION
    SELECT DISTINCT TRY_CAST(rcid AS BIGINT) AS rcid
    FROM foia_rcid_crosswalk
    WHERE LOWER(TRIM(crosswalk_validity_label)) = 'valid_match'
      AND TRY_CAST(rcid AS BIGINT) IS NOT NULL
    """
)

## WRDS
db = wrds.Connection(wrds_username='amykimecon')

# toggle for testing
test = rcfg.WRDS_POSITIONS_TEST

#####################
# DEFINING USER SAMPLE
#####################
rcids = list(con.sql("SELECT rcid FROM all_target_rcids ORDER BY rcid").df()['rcid'])

if test:
    rcidsamp = rcids[: rcfg.WRDS_POSITIONS_TEST_RCID_LIMIT]
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
    user_pos = db.raw_sql(f"SELECT a.user_id, a.position_id, a.position_number, rcid, country, startdate, enddate, role_k17000_v3, salary, total_compensation, company_raw, title_raw FROM revelio.individual_positions AS a LEFT JOIN revelio.individual_positions_raw AS b ON a.position_id = b.position_id WHERE a.user_id IN ({userid_subset})")

    return user_pos 


#####################
# QUERYING WRDS
#####################
saveloc = rcfg.WRDS_POSITIONS_CHUNK_STUB
j = rcfg.WRDS_POSITIONS_CHUNKS
j_eff = max(1, min(j, userids.shape[0]))

t0_0 = time.time()
print(f"Current Time: {datetime.datetime.now()}")
print(f"Running wrds_positions on {userids.shape[0]} userids from {len(rcidsamp)} target rcids")
print(f"Using {j_eff} chunks")
print("---------------------")

if userids.shape[0] == 0:
    print("No userids found for the current configuration; skipping query and merge.")
    out = pd.DataFrame()
    print(f"Script Ended: {datetime.datetime.now()}")
    print(f"Total script runtime: {round((time.time()-t_script0)/3600, 2)} hours")
    raise SystemExit(0)

# running chunks and saving
print("Querying and saving individual chunks...")
help.chunk_query(
    userids,
    j=j_eff,
    fun=get_merge_query,
    d=rcfg.WRDS_POSITIONS_CHUNK_SIZE,
    verbose=True,
    extraverbose=False,
    outpath=saveloc,
)

t1_1 = time.time()
print(f"Done! Time Elapsed: {round((t1_1-t0_0)/3600, 2)} hours")

# getting merged chunks
out = help.chunk_merge(saveloc, j=j_eff, outfile=rcfg.WRDS_POSITIONS_PARQUET, verbose=True)

print(f"Script Ended: {datetime.datetime.now()}")
print(f"Total script runtime: {round((time.time()-t_script0)/3600, 2)} hours")
