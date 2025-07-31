# File Description: Getting Revelio Data for Merged Companies
# Author: Amy Kim
# Date Created: Thurs Mar 20

# Imports and Paths
import wrds
import duckdb as ddb
import pandas as pd
import time
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
# QUERYING WRDS
#####################
rcids = list(con.sql("SELECT rcid FROM rmerge_w_dups, GROUP BY rcid").df()['rcid'])

if test:
    rcidsamp = rcids[:10]
else:
    rcidsamp = rcids 

userids = db.raw_sql(
f"""
    SELECT user_id FROM revelio.individual_positions WHERE country = 'United States' AND rcid IN ({','.join([str(i) for i in rcidsamp])}) GROUP BY user_id
""")

t0_0 = time.time()
print(f"Current Time: {datetime.datetime.now()}")

merged = []
i = 0
d = 1000

while d*i < len(userids):
    t0 = time.time()
    userid_subset = ','.join(userids.iloc[d*i:d*(i+1),]['user_id'].astype('str'))

    user_pos = db.raw_sql(f"SELECT a.user_id, rcid, country, startdate, enddate, role_k1500, salary, total_compensation, company_raw, title_raw FROM revelio.individual_positions AS a LEFT JOIN revelio.individual_positions_raw AS b ON a.position_id = b.position_id WHERE a.user_id IN ({userid_subset})")

    merged = merged + [user_pos]
    t1 = time.time()
    print(f"iteration #{i+1} of {int((len(userids) -1)/d) + 1}: {round((t1-t0)/60, 2)} min")
    i += 1    

# merged = merged + [db.raw_sql(f"SELECT * FROM revelio.individual_positions WHERE rcid IN ({','.join([str(i) for i in rcids[d*i:]])})")]

t1_1 = time.time()
print(f"Iterations Completed! Time Elapsed: {round((t1_1-t0_0)/3600, 2)} hours")

print("Merging Iterations and saving to file...")
merged_all = pd.concat(merged)
merged_all.to_parquet(f"{root}/data/int/rev_positions_jul31.parquet")

t2_2 = time.time()
print(f"Done! Total time: {round((t2_2-t0_0)/60, 2)} min")
print(f"Current Time: {datetime.datetime.now()}")