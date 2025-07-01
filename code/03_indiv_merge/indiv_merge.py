# File Description: Merging H-1B and Revelio Individual Data
# Author: Amy Kim
# Date Created: Jun 30 2025

# Imports and Paths
import duckdb as ddb
import pandas as pd
import numpy as np
import fiscalyear
import json


root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

con = ddb.connect()

# Importing Country Codes Crosswalk
with open(f"{root}/data/crosswalks/country_dict.json", "r") as json_file:
    country_cw_dict = json.load(json_file)

# helper function to get standardized country name
def get_std_country(country, dict = country_cw_dict):
    if country is None:
        return None 
    
    if country in dict.keys():
        return dict[country]
    
    if country in dict.values():
        return country 
    
    return "No Country Match"

con.create_function("get_std_country", lambda x: get_std_country(x), ['VARCHAR'], 'VARCHAR')

#####################
# IMPORTING DATA
#####################
## raw FOIA bloomberg data
foia_raw_file = con.read_csv(f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all_withids.csv")

## company crosswalk
company_cw = con.read_parquet(f"{root}/data/int/company_merge_sample_jun30.parquet")

## revelio user x Position-level Data
merged_pos = con.read_parquet(f"{root}/data/int/rev_merge_mar20.parquet")

## revelio individual user x country cleaned data
rev_users = con.read_parquet(f'{root}/data/int/rev_users_clean_jun30.parquet')

#####################
# CLEANING INDIV DATA
#####################
# cleaning FOIA data (application level)
foia_indiv = con.sql("SELECT a.FEIN, a.lottery_year, country, female_ind, yob, status_type, ben_multi_reg_ind, employer_name, n_apps, foia_temp_id, main_rcid, rcid FROM (SELECT FEIN, lottery_year, get_std_country(country_of_birth) AS country, CASE WHEN gender = 'female' THEN 1 ELSE 0 END AS female_ind, ben_year_of_birth AS yob, status_type, ben_multi_reg_ind, employer_name, COUNT(*) OVER(PARTITION BY a.FEIN, a.lottery_year) AS n_apps, ROW_NUMBER() OVER() AS foia_temp_id FROM foia_raw_file) AS a JOIN company_cw AS b ON a.FEIN = b.FEIN AND a.lottery_year = b.lottery_year WHERE sampgroup = 'insamp'")

# cleaning revelio data (collapsing to user x company level)
rev_indiv = con.sql(
"""
SELECT * FROM (
    (SELECT user_id, 
        MIN(startdate)::DATETIME AS first_startdate, 
        MAX(CASE WHEN enddate IS NULL THEN '2025-03-01' ELSE enddate END)::DATETIME AS last_enddate, rcid 
    FROM merged_pos WHERE startdate >= '2015-01-01' GROUP BY user_id, rcid) AS a 
    JOIN 
    (SELECT rcid FROM company_cw WHERE sampgroup = 'insamp' GROUP BY rcid) AS b 
    ON a.rcid = b.rcid
) AS pos 
JOIN 
rev_users AS users 
ON pos.user_id = users.user_id
""")


#####################
# MERGING!
#####################
con.sql("CREATE OR REPLACE TABLE merge_raw AS SELECT * FROM foia_indiv AS a LEFT JOIN rev_indiv AS b ON a.rcid = b.rcid AND a.country = b.country")

merge_filt = con.sql("SELECT DATEDIFF('month', (lottery_year || '-03-01')::DATETIME, first_startdate) AS startdatediff, DATEDIFF('month', (lottery_year || '-03-01')::DATETIME, last_enddate) AS enddatediff, * FROM merge_raw WHERE ABS(female_ind - f_prob) < 0.7 AND (est_yob IS NULL OR ABS(yob::INTEGER - est_yob) <= 5) AND startdatediff <= 0 AND startdatediff >= -60 AND enddatediff > 0")