# File Description: Getting Revelio Data for Merged Companies
# Author: Amy Kim
# Date Created: Thurs Mar 20

# Imports and Paths
import wrds
import duckdb as ddb
import pandas as pd
import time
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

#####################
# QUERYING WRDS
#####################
rcids = list(con.sql("SELECT rcid FROM rmerge_w_dups, GROUP BY rcid").df()['rcid'])

t0_0 = time.time()

merged = []
i = 0
d = 1000

while d*(i+1) < len(rcids):
    t0 = time.time()
    merged = merged + [db.raw_sql(f"SELECT * FROM revelio.individual_positions WHERE rcid IN ({','.join([str(i) for i in rcids[d*i:d*(i+1)]])})")]
    t1 = time.time()
    print(f"iteration #{i*d}: {round((t1-t0)/60, 2)} min")
    i += 1    
merged = merged + [db.raw_sql(f"SELECT * FROM revelio.individual_positions WHERE rcid IN ({','.join([str(i) for i in rcids[d*i:]])})")]

merged_all = pd.concat(merged)
merged_all.to_parquet(f"{root}/data/int/rev_merge_jul9.parquet")

t1_0 = time.time()
print(f"Done! Total time: {round((t1_0-t0_0)/60, 2)} min")