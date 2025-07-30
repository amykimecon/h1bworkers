# File Description: Using Name2Nat Package to get nationalities from names
# Author: Amy Kim
# Date Created: June 26 2025

# Imports and Paths
import duckdb as ddb
import time
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
testn = 20000
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

def name2nat_subdiv(i, n_tot, df, int_save = False):
    t0 = time.time()
    df_out = df.loc[df['rownum'] % n_tot == i].copy(deep=True)
    df_out['pred_nats_name'] = df_out['fullname_clean'].apply(help.name2nat_fun)

    # rev_names_withnat_temp = con.sql(f"SELECT fullname_clean, name2nat(fullname_clean) AS pred_nats_name FROM rev_names WHERE rownum % {n} = {i}")

    # con.sql(f"COPY rev_names_withnat_temp TO '{root}/data/int/name2nat_revelio/rev_names_withnat_jun26_{i}of{n}.parquet'")
    if int_save:
        df_out.to_parquet(f'{root}/data/int/name2nat_revelio/rev_names_withnat_jun26_{i}of{n_tot}.parquet')
    t1 = time.time()
    print(f"Time to complete iteration {i}: {round((t1-t0)/60,5)} min")

    return df_out


# subdividing into 1000-observation units (can start with )
n_subdiv = int(n/10000)
t0 = time.time()
print("TEST: SUBDIV, PANDAS")
print(f"Running name2nat on {n} names")

rev_names_df = rev_names.df()
df_out_all = []
for i in range(n_subdiv):
    temp = name2nat_subdiv(i, n_subdiv, rev_names_df, int_save)
    df_out_all = df_out_all + [temp]

t1 = time.time()
print(f"Done! Time to complete: {round((t1-t0)/60,5)} min")

# saving entire df concatenated
pd.concat(df_out_all).to_parquet(f'{root}/data/int/name2nat_revelio/rev_names_withnat_jun26.parquet')

# # no parallelization in duckdb:
# t0 = time.time()
# print("TEST: NO PARALLELIZATION, DUCKDB")
# print(f"Running name2nat on {n} names")
# rev_names_withnat = con.sql("SELECT fullname_clean, name2nat(fullname_clean) AS pred_nats_name FROM rev_names")
# con.sql(f"COPY rev_names_withnat TO '{root}/data/int/name2nat_revelio/rev_names_withnat_jun26.parquet'")
# t1 = time.time()
# print(f"Done! Time to complete: {round((t1-t0)/60,5)} min")

# # no parallelization in pandas
# t0 = time.time()
# print("TEST: NO PARALLELIZATION, PANDAS")
# print(f"Running name2nat on {n} names")

# rev_names_df = rev_names.df()
# rev_names_df['pred_nats_name'] = rev_names_df['fullname_clean'].apply(help.name2nat_fun)
# rev_names_df.to_parquet(f'{root}/data/int/name2nat_revelio/rev_names_withnat_jun26.parquet')
# t1 = time.time()
# print(f"Done! Time to complete: {round((t1-t0)/60,5)} min")


# # parallelizing
# num_cores = multiprocessing.cpu_count()

# t0_p = time.time()
# print("TEST: PARALLELIZATION")
# print(f"Running name2nat on {rev_names.shape[0]} names with {num_cores} processes")

# #p = multiprocessing.Pool(num_cores)
# outputs = Parallel(n_jobs = num_cores)(delayed(name2nat_multiprocessing)(i) for i in range(8))
# #p.map(name2nat_multiprocessing, range(8))

# t1_p = time.time()
# print(f"Done! Time to complete: {round((t1_p-t0_p)/60,5)} min")

