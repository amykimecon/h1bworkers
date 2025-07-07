# File Description: Getting Position Info for Merged Companies -- WRDS Cloud 
# Author: Amy Kim
# Date Created: Jul 1 2025

# Imports and Paths
import wrds
import duckdb as ddb
import pandas as pd
import os 
import time

con = ddb.connect()

# If testing locally
if os.environ.get('USER') == 'amykim':
    in_path = '/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers/data/wrds/wrds_in'
    out_path = '/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers/data/int'

# If on WRDS cloud
else:
    in_path = "/home/princeton/amykimecon/data"
    out_path = "/scratch/princeton/amykimecon"

# Toggle for testing
test = False

# Importing company crosswalk
company_cw = con.read_parquet(f"{in_path}/company_merge_sample_jun30.parquet")

# WRDS
db = wrds.Connection(wrds_username='amykimecon')

#####################
# QUERYING WRDS
#####################
rcids = list(con.sql("SELECT rcid FROM company_cw WHERE sampgroup = 'insamp' GROUP BY rcid").df()['rcid'])

if test:
    rcidsamp = rcids[:10]
else:
    rcidsamp = rcids 

userids = db.raw_sql(
f"""
    SELECT user_id FROM revelio.individual_positions WHERE country = 'United States' AND rcid IN ({','.join([str(i) for i in rcidsamp])}) GROUP BY user_id
""")

t0 = time.time()
i = 0
d = 10000 
merged = []

while d*i < len(userids):
    print(f"Iteration {i+1} of {int((len(userids)-1)/d) + 1}")

    userid_subset = ','.join(userids.iloc[d*i:d*(i+1),]['user_id'].astype('str'))

    user_pos = db.raw_sql(
    f"""
    SELECT user_id, 
        ARRAY_AGG(title_raw ORDER BY position_number) AS positions,
        ARRAY_AGG(rcid ORDER BY position_number) AS rcids,
        MIN(CASE WHEN position_number > max_intern_position THEN startdate ELSE NULL END) AS min_startdate,
        MIN(CASE WHEN position_number > max_intern_position AND country = 'United States' THEN startdate ELSE NULL END) AS min_startdate_us
    FROM (
        SELECT a.user_id, title_raw, position_number, rcid, country, startdate, enddate,
            CASE WHEN 
                (lower(title_raw) ~ '(^|\\s)(intern)($|\\s)' AND 
                    EXTRACT(YEAR FROM AGE(enddate, startdate)) < 1) 
                OR (lower(title_raw) ~ '(^|\\s)(student)($|\\s)') 
            THEN 1 ELSE 0 END AS intern_ind,
            MAX(CASE WHEN 
                (lower(title_raw) ~ '(^|\\s)(intern)($|\\s)' AND 
                    EXTRACT(YEAR FROM AGE(enddate, startdate)) < 1) 
                OR (lower(title_raw) ~ '(^|\\s)(student)($|\\s)')
            THEN position_number ELSE 0 END) 
            OVER(PARTITION BY a.user_id) AS max_intern_position
        FROM revelio.individual_positions AS a LEFT JOIN revelio.individual_positions_raw AS c ON a.position_id = c.position_id WHERE a.user_id IN ({userid_subset})
    ) GROUP BY user_id
    """
    )

    merged = merged + [user_pos]
    i = i + 1

t1 = time.time()
print(f"Iterations Completed! Time Elapsed: {round((t1-t0)/3600, 2)} hours")

print("Merging Iterations...")
merged_all = pd.concat(merged)
t2 = time.time()
print(f"Merging Completed! Time Elapsed: {round((t2-t1)/3600, 2)} hours")

print("Saving to file:")
merged_all.to_parquet(f"{out_path}/rev_user_positionhist.parquet")
print("Done!")
