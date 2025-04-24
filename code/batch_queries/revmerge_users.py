# File Description: Getting Revelio Data for Merged Companies
# Author: Amy Kim
# Date Created: Thurs Mar 20

# Imports and Paths
import wrds
import duckdb as ddb
import pandas as pd

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
j = 4
rcids = list(con.sql("SELECT rcid FROM rmerge_w_dups GROUP BY rcid").df()['rcid'])
rcidsubset = [r for r in rcids if r % 10 == j] #subsetting by last digit of rcid
print(len(rcidsubset))

# # GETTING INDIVIDUAL POSITIONS OF MATCHED RCIDS
def getmergequery(rcidlist):
    return f"""
        SELECT a.user_id AS user_id, rcid, 
            fullname, f_prob, updated_dt, university_name, rsid, c.education_number, ed_startdate, ed_enddate, degree, field, university_country, university_location, university_raw, degree_raw, field_raw, descripti>
        FROM (
                SELECT user_id, rcid 
                FROM revelio.individual_positions 
                WHERE country = 'United States' AND rcid IN ({','.join([str(i) for i in rcidlist])}) 
                GROUP BY user_id, rcid) AS a 
            LEFT JOIN (SELECT user_id, fullname, f_prob, updated_dt FROM revelio.individual_user) AS b 
            ON a.user_id = b.user_id 
            LEFT JOIN (SELECT user_id, university_name, rsid, education_number, startdate AS ed_startdate, enddate AS ed_enddate, degree, field, university_country, university_location FROM revelio.individual_user_e>
            ON a.user_id = c.user_id 
            LEFT JOIN (SELECT user_id, university_raw, education_number, degree_raw, field_raw, description FROM revelio.individual_user_education_raw) AS e ON c.user_id = e.user_id AND c.education_number=e.educatio>
        """
merged = []
i = 0
d = 20

while d*(i+1) < len(rcidsubset):
    print(i*d)
    merged = merged + [db.raw_sql(getmergequery(rcidsubset[d*i:d*(i+1)]))]
    i += 1

merged = merged + [db.raw_sql(getmergequery(rcidsubset[d*i:]))]
merged_all = pd.concat(merged)

merged_all.to_parquet(f"rev_user_merge{j}.parquet")