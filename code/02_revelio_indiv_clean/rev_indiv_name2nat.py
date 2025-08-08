# File Description: Using Name2Nat Package to get nationalities from names
# Author: Amy Kim
# Date Created: June 26 2025

# Imports and Paths
import duckdb as ddb
import time
import datetime
import json
import sys 
import os
from name2nat import Name2nat

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

con = ddb.connect()
my_nanat = Name2nat()

# test toggle
test = False
testn = 1000

if test:
    print("TEST VERSION")

####################
## IMPORTING DATA ##
####################
# Importing Data (From WRDS Server)
rev_raw = con.read_parquet(f"{wrds_out}/rev_user_merge0.parquet")

for j in range(1,10):
    rev_raw = con.sql(f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{wrds_out}/rev_user_merge{j}.parquet'")

#title case function
con.create_function("title", lambda x: x.title(), ['VARCHAR'], 'VARCHAR')

####################
## GETTING NAMES ##
####################
rev_clean = con.sql(
f"""
    SELECT 
    fullname, university_country, university_location, degree, user_id,
    {help.degree_clean_regex_sql()} AS degree_clean,
    {help.inst_clean_regex_sql('university_raw')} AS univ_raw_clean,
    CASE WHEN fullname ~ '.*[A-z].*' THEN {help.fullname_clean_regex_sql('fullname')} ELSE '' END AS fullname_clean,
    degree_raw, field_raw, university_raw
    FROM rev_raw
"""
)

if test:
    rev_clean = con.sql(f"SELECT * FROM rev_clean LIMIT {testn}")

# collapsing to name level
rev_names = con.sql("SELECT *, ROW_NUMBER() OVER(ORDER BY fullname_clean) AS rownum FROM (SELECT fullname_clean FROM rev_clean WHERE fullname_clean != '' GROUP BY fullname_clean)")

n = rev_names.shape[0]

# helper function
def name2nat_run(df):
    df_out = df.copy()

    df_out['pred_nats_name'] = [dict(n[1]) for n in my_nanat([name for name in df['fullname_clean']], top_n = 20)]
    
    #df_out['pred_nats_name'] = df_out['fullname_clean'].apply(lambda x: json.dumps(my_nanat(x, top_n = 10)[0][1]))

    return(df_out)

## declaring constants
if test:
    saveloc = f"{root}/data/int/name2nat_revelio/test"

else:
    saveloc = f"{root}/data/int/name2nat_revelio/name2nat_revelio"

j = 20
d = 100000

## running code
t0 = time.time()
print(f"Current Time: {datetime.datetime.now()}")

rev_names_df = rev_names.df()
# rev_names_df_list = [name for name in rev_names_df['fullname_clean']]

print(f"Running rev_indiv_name2nat on {rev_names_df.shape[0]} userids")
print("-------------------------")

# running chunks and saving 
print("Querying and saving individual chunks...")
help.chunk_query(rev_names_df, j = j, fun = name2nat_run, d = d, verbose = True, extraverbose = test, outpath = saveloc)

t1 = time.time()
print(f"Done! Time Elapsed: {round((t1-t0)/3600,2)} hours")

# getting merged chunks
if test:
    out = help.chunk_merge(saveloc, j = j, verbose = True)
else:
    out = help.chunk_merge(saveloc, j = j, outfile = f"{root}/data/int/name2nat_aug1.parquet", verbose = True)

print(f"Script Ended: {datetime.datetime.now()}")