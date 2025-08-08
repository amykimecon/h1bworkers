# File Description: Using NameTrace Package to get region and gender from names
# Author: Amy Kim
# Date Created: July 8, 2025 (see rev_indiv_name2nat.py)

# Imports and Paths
import duckdb as ddb
from nametrace import NameTracer
import time
import sys
import pandas as pd 
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

# helper functions
sys.path.append('02_revelio_indiv_clean/')
import h1bworkers.code.helpers as help

con = ddb.connect()

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
testn = 200000
rev_clean = con.sql(
f"""
    SELECT 
    fullname, university_country, university_location, degree, user_id,
    {help.degree_clean_regex_sql()} AS degree_clean,
    {help.inst_clean_regex_sql('university_raw')} AS univ_raw_clean,
    CASE WHEN fullname ~ '.*[A-z].*' THEN {help.fullname_clean_regex_sql('fullname')} ELSE '' END AS fullname_clean,
    degree_raw, field_raw, university_raw
    FROM rev_raw -- LIMIT {testn}
"""
)

# collapsing to name level
rev_names = con.sql("SELECT *, ROW_NUMBER() OVER(ORDER BY fullname_clean) AS rownum FROM (SELECT fullname_clean FROM rev_clean WHERE fullname_clean != '' GROUP BY fullname_clean)")
n = rev_names.shape[0]

my_nt = NameTracer()

def get_female_score(d):
    gender = d.get('gender')
    if isinstance(gender, list):
        for label, score in gender:
            if label == 'female':
                return score
    return None 

def nametrace_subdiv(i, n_tot, df, nt = my_nt):
    t0 = time.time()
    df_out = df.loc[df['rownum'] % n_tot == i].copy(deep=True)
    df_out['nametrace_json'] = nt.predict(df_out['fullname_clean'].to_list(), batch_size = n, topk = 5)
    df_out['f_prob_nt'] = df_out['nametrace_json'].apply(get_female_score)
    df_out['region_probs'] = df_out['nametrace_json'].apply(lambda x: x['subregion'])

    # df_out.to_parquet(f'{root}/data/int/name2nat_revelio/rev_names_withnat_jun26_{i}of{n_tot}.parquet')
    t1 = time.time()
    print(f"Time to complete iteration {i}: {round((t1-t0)/60,5)} min")

    return df_out[['fullname_clean', 'f_prob_nt', 'region_probs']]

# subdividing into 100k-observation units
n_subdiv = int(n/100000)
t0 = time.time()
print("TEST: SUBDIV, PANDAS")
print(f"Running name2nat on {n} names")

rev_names_df = rev_names.df()
df_out_all = []
for i in range(n_subdiv):
    temp = nametrace_subdiv(i, n_subdiv, rev_names_df)
    df_out_all = df_out_all + [temp]

t1 = time.time()
print(f"Done! Time to complete: {round((t1-t0)/60,5)} min")

# concatenating df
df_out_all_concat = pd.concat(df_out_all)

# mutating region probs to individual indicators per region
df_exp = df_out_all_concat.explode('region_probs')
df_exp_notnull = df_exp.loc[df_exp['region_probs'].isna() == False]
df_exp_notnull['region'] = df_exp_notnull['region_probs'].apply(lambda x: x[0])
df_exp_notnull['prob'] = df_exp_notnull['region_probs'].apply(lambda x: x[1])

# merging back to main and saving (pivoting)
pd.merge(df_out_all_concat[['fullname_clean','f_prob_nt']], df_exp_notnull.pivot(columns = 'region', values = 'prob'),how = 'left', left_index = True, right_index = True).to_parquet(f'{root}/data/int/rev_names_nametrace_jul8.parquet')

# merging back to main and saving (unpivoted)
pd.merge(df_out_all_concat[['fullname_clean','f_prob_nt']], df_exp_notnull[['region','prob']], how = 'left', left_index = True, right_index = True).to_parquet(f'{root}/data/int/rev_names_nametrace_long_jul8.parquet')