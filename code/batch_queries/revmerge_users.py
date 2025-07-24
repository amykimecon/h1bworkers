# File Description: Getting Revelio Data for Merged Companies -- WRDS Cloud
# Author: Amy Kim
# Date Created: Thurs Mar 20 2025 (Updated Jun 26 2025)

# Imports and Paths
import wrds
import duckdb as ddb
import pandas as pd
import os 
import time
import sys

con = ddb.connect()

# If on WRDS cloud:
if os.environ.get('USER') == 'amykimecon':
    in_path = "/home/princeton/amykimecon/data"
    out_path = "/scratch/princeton/amykimecon"

else:
    sys.path.append('../')
    from config import * 
    in_path = f'{root}/data/wrds/wrds_in'
    out_path = f'{root}/data/int'

# Toggle for testing
test = True

#####################
# IMPORTING DATA
#####################
## duplicate rcids
dup_rcids = con.read_csv(f"{in_path}/dup_rcids_mar20.csv")
con.sql("CREATE OR REPLACE TABLE dup_rcids AS SELECT main_rcid, rcid FROM dup_rcids, GROUP BY main_rcid, rcid")

## Importing R Matched data
rmerge = con.read_csv(f"{in_path}/good_match_ids_mar20.csv")
# merging with dup rcids to get full list of matched rcids
con.sql("CREATE OR REPLACE TABLE rmerge_w_dups AS SELECT a.main_rcid AS main_rcid, CASE WHEN b.rcid IS NULL THEN a.rcid ELSE b.rcid END AS rcid, FEIN FROM (rmerge AS a LEFT JOIN dup_rcids AS b ON a.main_rcid = b.main_rcid)")

## WRDS
db = wrds.Connection(wrds_username='amykimecon')

#####################
# QUERYING WRDS
#####################
t00 = time.time()

jtot = 10
for j in range(jtot):
    t0 = time.time()
    print(f"Chunking for storage limit: Chunk {j+1} of {jtot}")
    rcids = list(con.sql("SELECT rcid FROM rmerge_w_dups GROUP BY rcid").df()['rcid'])

    rcidsubset = [r for r in rcids if r % jtot == j] #subsetting by last digit of rcid

    if test:
        rcidsubset = rcidsubset[:10]

    # # GETTING INDIVIDUAL POSITIONS OF MATCHED RCIDS
    def getmergequery(rcidlist):
        return f"""
            SELECT a.user_id AS user_id, rcid, 
                fullname, f_prob, updated_dt, university_name, rsid, c.education_number, ed_startdate, ed_enddate, degree, field, university_country, university_location, university_raw, degree_raw, field_raw, description
            FROM (
                    SELECT user_id, rcid 
                    FROM revelio.individual_positions 
                    WHERE country = 'United States' AND rcid IN ({','.join([str(i) for i in rcidlist])}) 
                    GROUP BY user_id, rcid) AS a 
                LEFT JOIN (SELECT user_id, fullname, f_prob, updated_dt FROM revelio.individual_user) AS b 
                ON a.user_id = b.user_id 
                LEFT JOIN (SELECT user_id, university_name, rsid, education_number, startdate AS ed_startdate, enddate AS ed_enddate, degree, field, university_country, university_location FROM revelio.individual_user_education) AS c
                ON a.user_id = c.user_id 
                LEFT JOIN (SELECT user_id, university_raw, education_number, degree_raw, field_raw, description FROM revelio.individual_user_education_raw) AS e ON c.user_id = e.user_id AND c.education_number=e.education_number
            """
    merged = []
    i = 0
    d = 20

    print("Iterating...")

    while d*(i+1) < len(rcidsubset):
        #print(f"Iteration {i+1} of {int((len(rcidsubset)-1)/d) + 1}")
        merged = merged + [db.raw_sql(getmergequery(rcidsubset[d*i:d*(i+1)]))]
        i += 1

    print("Iteration done! Merging...")

    merged = merged + [db.raw_sql(getmergequery(rcidsubset[d*i:]))]
    merged_all = pd.concat(merged)

    print("Merging done! Saving...")

    merged_all.to_parquet(f"{out_path}/rev_user_merge{j}.parquet")

    t1 = time.time()
    print(f"Chunk {j+1} Completed! Time Elapsed: {round((t1-t0)/60,2)} minutes")

t11 = time.time()
print(f"All Chunks Completed! Total Time Elapsed: {round((t11-t00)/3600,2)} hours")
