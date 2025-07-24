# File Description: Merging H-1B and Revelio Individual Data
# Author: Amy Kim
# Date Created: Jun 30 2025

# Imports and Paths
import duckdb as ddb
import pandas as pd
import numpy as np
import json
import sys

sys.path.append('../')
from config import * 

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
company_cw = con.read_parquet(f"{root}/data/int/company_merge_sample_jul10.parquet")

## revelio user x Position-level Data
merged_pos = con.read_parquet(f"{root}/data/int/rev_merge_jul9.parquet")

## revelio individual user x country cleaned data
rev_users = con.read_parquet(f'{root}/data/int/rev_users_clean_jul8.parquet')

## revelio user x position history data
rev_positionhist = con.read_parquet(f'{root}/data/wrds/wrds_out/rev_user_positionhist_jul2.parquet')

#####################
# CLEANING INDIV DATA
#####################
# cleaning FOIA data (application level)
foia_indiv = con.sql("SELECT a.FEIN, a.lottery_year, country, female_ind, yob, status_type, ben_multi_reg_ind, employer_name, n_apps, n_unique_country, foia_temp_id, main_rcid, rcid FROM (SELECT FEIN, lottery_year, get_std_country(country_of_nationality) AS country, CASE WHEN gender = 'female' THEN 1 ELSE 0 END AS female_ind, ben_year_of_birth AS yob, status_type, ben_multi_reg_ind, employer_name, COUNT(*) OVER(PARTITION BY FEIN, lottery_year) AS n_apps, COUNT(DISTINCT country_of_birth) OVER(PARTITION BY FEIN, lottery_year) AS n_unique_country, ROW_NUMBER() OVER() AS foia_temp_id FROM foia_raw_file) AS a JOIN company_cw AS b ON a.FEIN = b.FEIN AND a.lottery_year = b.lottery_year WHERE sampgroup = 'insamp'")

# cleaning revelio data (collapsing to user x company level)
rev_indiv = con.sql(
"""
SELECT * FROM (
    (SELECT user_id, 
        MIN(startdate)::DATETIME AS first_startdate, 
        MAX(CASE WHEN enddate IS NULL THEN '2025-03-01' ELSE enddate END)::DATETIME AS last_enddate, rcid 
    FROM merged_pos WHERE country = 'United States' AND startdate >= '2015-01-01' GROUP BY user_id, rcid) AS a 
    JOIN 
    (SELECT rcid FROM company_cw GROUP BY rcid) AS b 
    ON a.rcid = b.rcid
) AS pos 
JOIN 
(SELECT * FROM rev_users WHERE (us_hs_exact IS NULL OR us_hs_exact = 0) AND (us_educ IS NULL OR us_educ = 1) AND (stem_ind IS NULL OR stem_ind = 1)) AS users 
ON pos.user_id = users.user_id
LEFT JOIN
rev_positionhist AS poshist
ON pos.user_id = poshist.user_id
""")


#####################
# MERGING!
#####################
con.sql("""CREATE OR REPLACE TABLE merge_raw AS 
        SELECT *, COUNT(*) OVER(PARTITION BY foia_temp_id, a.rcid) AS n_match_raw,
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, first_startdate) AS startdatediff, 
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, last_enddate) AS enddatediff, 
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, min_startdate_us::DATETIME) AS months_work_in_us,
            SUBSTRING(min_startdate, 1, 4)::INTEGER - 16 AS max_yob,
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, updated_dt::DATETIME) AS updatediff
        FROM foia_indiv AS a LEFT JOIN rev_indiv AS b ON a.rcid = b.rcid AND a.country = b.country""")

con.sql("""CREATE OR REPLACE TABLE merge_raw_alt2 AS 
        SELECT *, COUNT(*) OVER(PARTITION BY foia_temp_id, a.rcid) AS n_match_raw,
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, first_startdate) AS startdatediff, 
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, last_enddate) AS enddatediff, 
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, min_startdate_us::DATETIME) AS months_work_in_us,
            SUBSTRING(min_startdate, 1, 4)::INTEGER - 16 AS max_yob,
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, updated_dt::DATETIME) AS updatediff
        FROM (SELECT * FROM foia_indiv WHERE country != 'China' AND country != 'India' AND country != 'Taiwan' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'Nepal' AND country != 'Pakistan') AS a LEFT JOIN rev_indiv AS b ON a.rcid = b.rcid AND a.country = b.country""")

# FILTERS
def merge_filt_func(merge_raw_tab, MONTH_BUFFER = 6, YOB_BUFFER = 5, F_PROB_BUFFER = 0.8):
    str_out = f"""
        SELECT *, COUNT(*) OVER(PARTITION BY foia_temp_id, rcid) AS n_match_filt,
        (COUNT(DISTINCT foia_temp_id) OVER(PARTITION BY FEIN, lottery_year))/n_apps AS share_apps_matched,
        COUNT(DISTINCT user_id) OVER(PARTITION BY FEIN, lottery_year) AS n_rev_users,
        (COUNT(*) OVER(PARTITION BY FEIN, lottery_year))/(COUNT(DISTINCT foia_temp_id) OVER(PARTITION BY FEIN, lottery_year)) AS match_mult,
        (COUNT(*) OVER(PARTITION BY FEIN, lottery_year))/(COUNT(DISTINCT user_id) OVER(PARTITION BY FEIN, lottery_year)) AS rev_mult,
        COUNT(DISTINCT foia_temp_id) OVER(PARTITION BY FEIN, lottery_year) AS n_apps_matched
        FROM {merge_raw_tab}
        WHERE  ABS(female_ind - (f_prob + f_prob_nt)/2) < {F_PROB_BUFFER} AND 
            (ABS(yob::INTEGER - est_yob) <= {YOB_BUFFER} OR (est_yob IS NULL AND yob::INTEGER <= max_yob))
            AND startdatediff <= {0+MONTH_BUFFER} AND startdatediff >= {-36 - MONTH_BUFFER} AND enddatediff >= {0-MONTH_BUFFER}
    """

    return str_out

merge_filt_base = con.sql(merge_filt_func('merge_raw'))

MATCH_MULT_CUTOFF = 3
REV_MULT_COEFF = 1
merge_filt_alt1 = con.sql(f'SELECT * FROM ({merge_filt_func('merge_raw')}) WHERE n_apps > 1 AND share_apps_matched = 1 AND match_mult <= {MATCH_MULT_CUTOFF} AND rev_mult < {REV_MULT_COEFF}*n_apps_matched ')

merge_filt_alt2 = con.sql(merge_filt_func('merge_raw_alt2'))

con.sql(f"COPY merge_filt_base TO '{root}/data/int/merge_filt_base_jul23.parquet'")
con.sql(f"COPY merge_filt_alt1 TO '{root}/data/int/merge_filt_postfilt_jul23.parquet'")
con.sql(f"COPY merge_filt_alt2 TO '{root}/data/int/merge_filt_prefilt_jul23.parquet'")






# con.sql("SELECT * FROM merge_filt WHERE n_apps < 5 AND n_apps > 1 AND n_unique_country > 1 AND country != 'India' AND country != 'China' ORDER BY RANDOM()").df()

# con.sql("SELECT * FROM foia_indiv WHERE FEIN = '824340234' ORDER BY lottery_year").df()

# con.sql("SELECT foia_temp_id, lottery_year, status_type, employer_name, female_ind, (f_prob + f_prob_nt)/2 AS f_prob_avg, yob, est_yob, country, fullname, total_score, inst_score, nanat_score, university_raw, first_startdate, last_enddate, user_id, hs_ind, valid_postsec, positions FROM merge_filt WHERE FEIN = '020622328' AND lottery_year = 2022 ORDER BY foia_temp_id").df()


# con.sql("SELECT lottery_year, status_type, employer_name, female_ind, f_prob, yob, est_yob, country, fullname, total_score, inst_score, nanat_score, university_raw, first_startdate, last_enddate FROM merge_filt WHERE foia_temp_id = 935115 ORDER BY total_score DESC").df()

# con.sql("SELECT lottery_year, employer_name, status_type, country_of_birth, gender, ben_year_of_birth FROM foia_raw_file WHERE FEIN = '411958972'").df()

# con.sql("SELECT lottery_year, status_type, employer_name, female_ind, f_prob, yob, est_yob, country, fullname, total_score, inst_score, nanat_score, university_raw, first_startdate, last_enddate FROM merge_filt WHERE FEIN = '411958972' ORDER BY total_score DESC").df()

# #foia_temp_id 364920
# #FEIN 201067637, lottery year 2023