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
sys.path.append(os.path.dirname(__file__))
from config import *
import rev_indiv_config as rcfg

con = ddb.connect()
my_nanat = Name2nat()
print(f"Using config: {rcfg.ACTIVE_CONFIG_PATH}")

# test toggle
test = rcfg.NAME2NAT_TEST
testn = rcfg.NAME2NAT_TESTN

if test:
    print("TEST VERSION")

####################
## IMPORTING DATA ##
####################
# Importing Data (From WRDS Server)
if os.path.exists(rcfg.WRDS_USERS_PARQUET):
    print(f"Loading consolidated users file: {rcfg.WRDS_USERS_PARQUET}")
    rev_raw = con.read_parquet(rcfg.WRDS_USERS_PARQUET)
else:
    print("Consolidated users file not found. Falling back to legacy rev_user_merge shards.")
    rev_raw = con.read_parquet(rcfg.LEGACY_WRDS_USER_MERGE_SHARDS[0])
    for j in range(1, len(rcfg.LEGACY_WRDS_USER_MERGE_SHARDS)):
        rev_raw = con.sql(
            f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{rcfg.LEGACY_WRDS_USER_MERGE_SHARDS[j]}'"
        )

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
    saveloc = rcfg.NAME2NAT_CHUNK_STUB_TEST
else:
    saveloc = rcfg.NAME2NAT_CHUNK_STUB

j = rcfg.NAME2NAT_CHUNKS
d = rcfg.NAME2NAT_CHUNK_SIZE

## running code
t0 = time.time()
print(f"Current Time: {datetime.datetime.now()}")

rev_names_df = rev_names.df()
# rev_names_df_list = [name for name in rev_names_df['fullname_clean']]
j_eff = max(1, min(j, rev_names_df.shape[0]))

print(f"Running rev_indiv_name2nat on {rev_names_df.shape[0]} userids")
print(f"Using {j_eff} chunks")
print("-------------------------")

if rev_names_df.shape[0] == 0:
    print("No names found for the current configuration; skipping query and merge.")
    print(f"Script Ended: {datetime.datetime.now()}")
    raise SystemExit(0)

# running chunks and saving 
print("Querying and saving individual chunks...")
help.chunk_query(rev_names_df, j = j_eff, fun = name2nat_run, d = d, verbose = True, extraverbose = test, outpath = saveloc)

t1 = time.time()
print(f"Done! Time Elapsed: {round((t1-t0)/3600,2)} hours")

# getting merged chunks
outfile = rcfg.NAME2NAT_PARQUET_TEST if test else rcfg.NAME2NAT_PARQUET
out = help.chunk_merge(saveloc, j = j_eff, outfile = outfile, verbose = True)
print(f"Saved merged output: {outfile}")

print(f"Script Ended: {datetime.datetime.now()}")
