# File Description: Using Name2Nat Package to get nationalities from names
# Author: Amy Kim
# Date Created: June 26 2025

# Imports and Paths
import duckdb as ddb
import time
import rev_indiv_clean_helpers as help

root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"
con = ddb.connect()

####################
## IMPORTING DATA ##
####################
# Importing Data (From WRDS Server)
rev_raw = con.read_parquet(f"{root}/data/wrds/wrds_out/rev_user_merge0.parquet")

for j in range(1,10):
    rev_raw = con.sql(f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{root}/data/wrds/wrds_out/rev_user_merge{j}.parquet'")
    print(rev_raw.shape)

# getting name2nat function and saving as duckdb function
con.create_function("name2nat", lambda x: help.name2nat_fun(x), ['VARCHAR'], 'VARCHAR')

#title case function
con.create_function("title", lambda x: x.title(), ['VARCHAR'], 'VARCHAR')

####################
## GETTING NAMES ##
####################
testn = 100000
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
rev_names = con.sql("SELECT fullname_clean FROM rev_clean WHERE fullname_clean != '' GROUP BY fullname_clean")

# running through name2nat 
t0 = time.time()
print(f"Running name2nat on {rev_names.shape[0]} names")
con.sql("CREATE OR REPLACE TABLE rev_names_withnat AS SELECT fullname_clean, name2nat(fullname_clean) AS pred_nats_name FROM rev_names")
t1 = time.time()
print(f"Done! Time to complete: {round((t1-t0)/60,5)} min")

con.sql(f"COPY rev_names_withnat TO '{root}/data/int/rev_names_withnat_jun26.parquet'")