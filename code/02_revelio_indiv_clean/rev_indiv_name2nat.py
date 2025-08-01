# File Description: Using Name2Nat Package to get nationalities from names
# Author: Amy Kim
# Date Created: June 26 2025

# Imports and Paths
import duckdb as ddb
import time
import datetime
import pandas as pd
import sys 
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

# helper functions
sys.path.append('02_revelio_indiv_clean/')
import rev_indiv_clean_helpers as help

con = ddb.connect()

os.environ['LD_LIBRARY_PATH'] = '/home/yk0581/.conda/env/h1benv/lib/'

####################
## IMPORTING DATA ##
####################
# Importing Data (From WRDS Server)
rev_raw = con.read_parquet(f"{wrds_out}/rev_user_merge0.parquet")

for j in range(1,10):
    rev_raw = con.sql(f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{wrds_out}/rev_user_merge{j}.parquet'")
    print(rev_raw.shape)

# getting name2nat function and saving as duckdb function
con.create_function("name2nat", lambda x: help.name2nat_fun(x), ['VARCHAR'], 'VARCHAR')

#title case function
con.create_function("title", lambda x: x.title(), ['VARCHAR'], 'VARCHAR')

####################
## GETTING NAMES ##
####################
testn = 1000
int_save = False #toggle for intermediate save

rev_clean = con.sql(
f"""
    SELECT 
    fullname, university_country, university_location, degree, user_id,
    {help.degree_clean_regex_sql()} AS degree_clean,
    {help.inst_clean_regex_sql('university_raw')} AS univ_raw_clean,
    CASE WHEN fullname ~ '.*[A-z].*' THEN {help.fullname_clean_regex_sql('fullname')} ELSE '' END AS fullname_clean,
    degree_raw, field_raw, university_raw
    FROM rev_raw LIMIT {testn}
"""
)

# collapsing to name level
rev_names = con.sql("SELECT *, ROW_NUMBER() OVER(ORDER BY fullname_clean) AS rownum FROM (SELECT fullname_clean FROM rev_clean WHERE fullname_clean != '' GROUP BY fullname_clean)")
n = rev_names.shape[0]

# helper function
def name2nat_run(df):
    df_out = df.copy()
    df_out['pred_nats_name'] = df_out['fullname_clean'].apply(help.name2nat_fun)
    return(df_out)

## declaring constants
saveloc = f"{root}/data/int/name2nat_revelio/name2nat_revelio"
j = 10
d = 100

rev_names_df = rev_names.df()

## running code
t0 = time.time()
print(f"Current Time: {datetime.datetime.now()}")
print(f"Running rev_indiv_name2nat on {rev_names_df.shape[0]} userids")
print("-------------------------")

# running chunks and saving 
print("Querying and saving individual chunks...")
help.chunk_query(rev_names_df, j = j, fun = name2nat_run, d = d, verbose = True, extraverbose = False, outpath = saveloc)

t1 = time.time()
print(f"Done! Time Elapsed: {round((t1-t0)/3600,2)} hours")

# getting merged chunks
out = help.chunk_merge(saveloc, j = j, outfile = f"{root}/data/int/name2nat_aug1.parquet", verbose = True)

print(f"Script Ended: {datetime.datetime.now()}")