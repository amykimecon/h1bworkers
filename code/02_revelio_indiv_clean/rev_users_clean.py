# File Description: Cleaning and Merging User and Position Data from Reveliio
# Author: Amy Kim
# Date Created: Wed Apr 9 (Updated June 26 2025)

# Imports and Paths
import duckdb as ddb
# import pandas as pd
# import numpy as np
# import seaborn as sns
import rev_indiv_clean_helpers as help
import os
import re
# import json

# local
if os.environ.get('USER') == 'amykim':
    root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
    code = "/Users/amykim/Documents/GitHub/h1bworkers/code"
# malloy
elif os.environ.get('USER') == 'yk0581':
    root = "/home/yk0581"
    code = "/Users/amykim/Documents/GitHub/h1bworkers/code"
    
wrds_out = f"{root}/data/wrds/wrds_out/jun26"

con = ddb.connect()

## Creating DuckDB functions from python helpers
#title case function
con.create_function("title", lambda x: x.title(), ['VARCHAR'], 'VARCHAR')

# country crosswalk function
con.create_function("get_std_country", lambda x: help.get_std_country(x), ['VARCHAR'], 'VARCHAR')

#####################################
## IMPORTING DATA
#####################################
## duplicate rcids (companies that appear more than once in linkedin data)
dup_rcids = con.read_csv(f"{root}/data/int/dup_rcids_mar20.csv")

## matched company data from R
rmerge = con.read_csv(f"{root}/data/int/good_match_ids_mar20.csv")

## raw FOIA bloomberg data
foia_raw_file = con.read_csv(f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv")

## joining raw FOIA data with merged data to get foia_ids in raw foia data
foia_with_ids = con.sql("SELECT *, CASE WHEN matched IS NULL THEN 0 ELSE matched END AS matchind FROM ((SELECT * FROM foia_raw_file WHERE NOT FEIN = '(b)(3) (b)(6) (b)(7)(c)') AS a LEFT JOIN (SELECT lottery_year, FEIN, foia_id, 1 AS matched FROM rmerge GROUP BY lottery_year, FEIN, foia_id) AS b ON a.lottery_year = b.lottery_year AND a.FEIN = b.FEIN)")

# Importing User x Education-level Data (From WRDS Server)
rev_raw = con.read_parquet(f"{wrds_out}/rev_user_merge0.parquet")

for j in range(1,10):
    rev_raw = con.sql(f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{wrds_out}/rev_user_merge{j}.parquet'")
    print(rev_raw.shape)

# Importing Institution x Country Matches
inst_country_cw = con.read_parquet(f"{root}/data/int/rev_inst_countries_jun30.parquet")

# Importing Name x Country Matches
nanats = con.read_parquet(f"{root}/data/int/name2nat_revelio/rev_names_withnat_jun26.parquet")

# Importing User x Position-level Data
merged_pos = con.read_parquet(f"{root}/data/int/rev_merge_mar20.parquet")
#occ_cw = con.read_csv(f"{root}/data/crosswalks/rev_occ_to_foia_freq.csv")

#####################################
# DEFINING MAIN SAMPLE OF USERS (RCID IN MAIN SAMP AND START DATE AFTER 2015)
#####################################
# id crosswalk between revelio and h1b companies
id_merge = con.sql("""
-- collapse matched IDs to be unique at the rcid x FEIN x lottery_year level
SELECT lottery_year, main_rcid, foia_id
FROM rmerge 
GROUP BY foia_id, lottery_year, main_rcid
""")

# IDing outsourcing/staffing companies (defining share of applications that are multiple registrations ['duplicates']; counting total number of apps)
foia_main_samp_unfilt = con.sql("SELECT FEIN, lottery_year, COUNT(CASE WHEN ben_multi_reg_ind = 1 THEN 1 END)/COUNT(*) AS share_multireg, COUNT(*) AS n_apps_tot, COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_success, COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END)/COUNT(*) AS win_rate FROM foia_with_ids GROUP BY FEIN, lottery_year")

n = con.sql('SELECT COUNT(*) FROM foia_main_samp_unfilt').df().iloc[0,0]
print(f"Total Employer x Years: {n}")
print(f"Employer x Years with Fewer than 50 Apps: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE n_apps_tot < 50").df().iloc[0,0]}")
print(f"Employer x Years with Fewer than 50% Duplicates: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE share_multireg < 0.5").df().iloc[0,0]}")
print(f"Employer x Years with No Duplicates: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE share_multireg = 0").df().iloc[0,0]}")

# main sample (conservative): companies with fewer than 50 applications and no duplicate registrations TODO: declare these as constants at top
foia_main_samp = con.sql("SELECT * FROM foia_main_samp_unfilt WHERE n_apps_tot < 50 AND share_multireg = 0")
print(f"Preferred Sample: {foia_main_samp.df().shape[0]} ({round(100*foia_main_samp.df().shape[0]/n)}%)")

# computing win rate by sample
foia_main_samp_def = con.sql("SELECT *, CASE WHEN n_apps_tot < 50 AND share_multireg = 0 THEN 'insamp' ELSE 'outsamp' END AS sampgroup FROM foia_main_samp_unfilt")
con.sql("SELECT sampgroup, SUM(n_success)/SUM(n_apps_tot) AS total_win_rate FROM foia_main_samp_def GROUP BY sampgroup")

# creating crosswalk between foia id and rcid (joining list of FEINs in and out of sample with foia data with foia ids and joining that to id crosswalk, then joining to dup_rcids)
samp_to_rcid = con.sql("SELECT a.FEIN, a.lottery_year, sampgroup, b.foia_id, c.main_rcid, CASE WHEN rcid IS NULL THEN c.main_rcid ELSE d.rcid END AS rcid FROM ((SELECT FEIN, lottery_year, sampgroup FROM foia_main_samp_def) AS a JOIN (SELECT FEIN, lottery_year, foia_id FROM foia_with_ids GROUP BY FEIN, lottery_year, foia_id) AS b ON a.FEIN = b.FEIN AND a.lottery_year = b.lottery_year JOIN (SELECT main_rcid, foia_id FROM id_merge) AS c ON b.foia_id = c.foia_id) LEFT JOIN (SELECT main_rcid, rcid FROM dup_rcids) AS d ON c.main_rcid = d.main_rcid")

# writing company sample crosswalk to file
con.sql(f"COPY samp_to_rcid TO '{root}/data/int/company_merge_sample_jun30.parquet'")

# selecting user ids from list of positions based on whether company in sample and start date is after 2015 (conservative bandwidth) -- TODO: declare cutoff date as constant; TODO: move startdate filter into merged_pos query, avoid pulling people who got promoted after 2015 but started working before 2015
user_samp = con.sql("SELECT user_id FROM ((SELECT rcid FROM samp_to_rcid WHERE sampgroup = 'insamp' GROUP BY rcid) AS a JOIN (SELECT user_id, startdate, rcid FROM merged_pos) AS b ON a.rcid = b.rcid) WHERE startdate >= '2015-01-01' GROUP BY user_id")

#####################################
### CLEANING AND MERGING REVELIO USERS TO COUNTRIES
#####################################
# Cleaning Revelio Data, removing duplicates
rev_clean = con.sql(
f"""
SELECT * FROM
    (SELECT 
    fullname, degree, user_id,
    {help.degree_clean_regex_sql()} AS degree_clean,
    {help.inst_clean_regex_sql('university_raw')} AS univ_raw_clean,
    CASE WHEN fullname ~ '.*[A-z].*' THEN {help.fullname_clean_regex_sql('fullname')} ELSE '' END AS fullname_clean,
    degree_raw, field_raw, university_raw, f_prob, education_number, ed_enddate, ed_startdate, ROW_NUMBER() OVER(PARTITION BY user_id, education_number) AS dup_num
    FROM rev_raw)
WHERE dup_num = 1
"""
)

# Filtering to only include users in the preferred sample
rev_users_filt = con.sql(f"SELECT * FROM rev_clean AS a JOIN user_samp AS b ON a.user_id = b.user_id")

# # temp for testing
# ids = ",".join(con.sql(help.random_ids_sql('user_id','rev_clean', n = 100)).df()['user_id'].astype(str))
# rev_users_filt = con.sql(f"SELECT * FROM rev_clean AS a JOIN user_samp AS b ON a.user_id = b.user_id WHERE a.user_id IN ({ids})")

# Cleaning name matches (long on country)
nanats_long = con.sql(f"SELECT fullname_clean, nanat_country, SUM(nanat_prob) AS nanat_prob FROM (SELECT fullname_clean, get_std_country({help.nanats_to_long('pred_nats_name')}[1]) AS nanat_country, {help.nanats_to_long('pred_nats_name')}[2]::FLOAT AS nanat_prob FROM nanats) WHERE nanat_prob > 0.02 GROUP BY fullname_clean, nanat_country")
# # temp for testing
# nanats_long = con.sql(f"SELECT fullname_clean, nanat_country, SUM(nanat_prob) AS nanat_prob FROM (SELECT fullname_clean, get_std_country({help.nanats_to_long('pred_nats_name')}[1]) AS nanat_country, {help.nanats_to_long('pred_nats_name')}[2]::FLOAT AS nanat_prob FROM (SELECT * FROM nanats AS a RIGHT JOIN (SELECT fullname_clean FROM rev_users_filt GROUP BY fullname_clean) AS b ON a.fullname_clean = b.fullname_clean)) WHERE nanat_prob > 0.02 GROUP BY fullname_clean, nanat_country")

# Cleaning institution matches 
inst_match_clean = con.sql(
"""
    SELECT *, 
        COUNT(*) OVER(PARTITION BY university_raw) AS n_country_match 
    FROM inst_country_cw
    WHERE lower(REGEXP_REPLACE(university_raw, '[^A-z]', '', 'g')) NOT IN ('highschool', 'ged', 'unknown', 'invalid')
""")

# Merging with institution matches (long) and collapsing to user x country level, filtering out 
inst_merge_long = con.sql(
"""
SELECT user_id, university_raw, match_country, 
    us_hs_exact, us_educ,
    CASE WHEN degree_clean = 'High School' OR hs_share > 0.9 THEN matchscore WHEN degree_clean = 'Bachelor' THEN matchscore*0.8 ELSE matchscore*0.5 END AS matchscore_corr, 
    matchscore, matchtype, education_number, 
    MAX(matchscore) OVER(PARTITION BY user_id, match_country) AS max_matchscore,
    COUNT(*) OVER(PARTITION BY user_id, match_country) AS n_match
FROM
    (SELECT user_id, education_number, degree_clean, 
        a.university_raw, match_country, matchscore, matchtype, hs_share,
    -- trying to get earliest education for each country
        ROW_NUMBER() OVER(PARTITION BY user_id, match_country ORDER BY education_number) AS educ_order,
    -- ID-ing if US high school (only exact matches)
        MAX(CASE WHEN 
            (degree_clean = 'High School' OR hs_share >= 0.5) AND 
            match_country = 'United States' AND 
            (matchtype = 'exact' OR (n_country_match = 1)) 
            THEN 1 ELSE 0 END) 
        OVER(PARTITION BY user_id) AS us_hs_exact,
    -- ID-ing if any US education
        MAX(CASE WHEN degree_clean != 'Non-Degree' AND match_country = 'United States' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS us_educ
    FROM 
        (SELECT *,
        -- Getting share of given institution labelled as high school
            (SUM(CASE WHEN degree_clean = 'High School' THEN 1 ELSE 0 END) OVER(PARTITION BY university_raw))/(SUM(CASE WHEN degree_clean != 'Missing' THEN 1 ELSE 0 END) OVER(PARTITION BY university_raw)) AS hs_share,
        FROM rev_users_filt) AS a 
    JOIN inst_match_clean AS b 
    ON a.university_raw = b.university_raw
    ) 
WHERE educ_order = 1 AND match_country != 'NA'
"""
)

# Merging with name matches (long)
name_merge_long = con.sql(
"""
SELECT user_id, a.fullname_clean, nanat_country, nanat_prob FROM (SELECT fullname_clean, user_id FROM rev_users_filt GROUP BY fullname_clean, user_id) AS a JOIN nanats_long AS b ON a.fullname_clean = b.fullname_clean
"""
)

# combining institution and name matches (user x country level)
all_merge_long = con.sql(
"""SELECT 
        CASE WHEN a.user_id IS NULL THEN b.user_id ELSE a.user_id END AS user_id, fullname_clean, university_raw,
        CASE WHEN match_country IS NULL THEN nanat_country ELSE match_country END AS country, 
        CASE WHEN match_country IS NULL THEN 0 ELSE matchscore_corr END AS inst_score, 
        CASE WHEN nanat_country IS NULL THEN 0 ELSE nanat_prob END AS nanat_score,
        us_hs_exact, us_educ
    FROM inst_merge_long AS a 
    FULL JOIN name_merge_long AS b 
    ON a.user_id = b.user_id AND a.match_country = b.nanat_country""")

#####################################
### GETTING AND EXPORTING FINAL USER FILE
#####################################
final_user_merge = con.sql(f"SELECT a.user_id, est_yob, f_prob, fullname, university_raw, country, inst_score, nanat_score, 0.5*inst_score + 0.5*nanat_score AS total_score, MAX(us_hs_exact) OVER(PARTITION BY a.user_id) AS us_hs_exact, MAX(us_educ) OVER(PARTITION BY a.user_id) AS us_educ FROM (SELECT * FROM (SELECT user_id, {help.get_est_yob()} AS est_yob, f_prob, fullname FROM rev_users_filt) GROUP BY user_id, est_yob, f_prob, fullname) AS a LEFT JOIN all_merge_long AS b ON a.user_id = b.user_id")

con.sql(f"COPY final_user_merge TO '{root}/data/int/rev_users_clean_jun30.parquet'")

final_user_merge = con.read_parquet(f'{root}/data/int/rev_users_clean_jun30.parquet')
