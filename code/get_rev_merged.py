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
rcids = list(con.sql("SELECT rcid FROM rmerge_w_dups, GROUP BY rcid").df()['rcid'])

# # GETTING INDIVIDUAL POSITIONS OF MATCHED RCIDS
# merged_pos = db.raw_sql(f"SELECT * FROM (SELECT * FROM revelio.individual_positions AS a JOIN (SELECT * FROM revelio.company_mapping WHERE rcid IN ({','.join([str(i) for i in rcids])})) AS b ON a.rcid = b.rcid)")

merged = []
i = 0
d = 1000

while d*(i+1) < len(rcids):
    print(i*d)
    merged = merged + [db.raw_sql(f"SELECT * FROM revelio.individual_positions WHERE rcid IN ({','.join([str(i) for i in rcids[d*i:d*(i+1)]])})")]
    i += 1    
merged = merged + [db.raw_sql(f"SELECT * FROM revelio.individual_positions WHERE rcid IN ({','.join([str(i) for i in rcids[d*i:]])})")]

merged_all = pd.concat(merged)
merged_all.to_parquet(f"{root}/data/int/rev_merge_mar20.parquet")
