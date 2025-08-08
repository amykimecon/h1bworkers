# File Description: Cleaning and Merging User and Position Data from Reveliio
# Author: Amy Kim
# Date Created: Wed Apr 9 (Updated June 26 2025)

# Imports and Paths
import duckdb as ddb
import json
import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

# CONSTANTS
CUTOFF_SHARE_MULTIREG=0
CUTOFF_N_APPS_TOT=10

con = ddb.connect()

# Importing Country Codes Crosswalk
with open(f"{root}/data/crosswalks/country_dict.json", "r") as json_file:
    country_cw_dict = json.load(json_file)

# Importing Region Crosswalk
with open(f"{root}/data/crosswalks/subregion_dict.json", "r") as json_file:
    subregion_dict = json.load(json_file)

## Creating DuckDB functions from python helpers
#title case function
con.create_function("title", lambda x: x.title(), ['VARCHAR'], 'VARCHAR')

# country crosswalk function
con.create_function("get_std_country", lambda x: help.get_std_country(x, country_cw_dict), ['VARCHAR'], 'VARCHAR')

# region crosswalk function
con.create_function("get_country_subregion", lambda x: help.get_country_subregion(x, country_cw_dict, subregion_dict), ['VARCHAR'], 'VARCHAR')

#####################################
## IMPORTING DATA
#####################################
print('Loading all data sources...')
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

# Importing Institution x Country Matches
inst_country_cw = con.read_parquet(f"{root}/data/int/rev_inst_countries_jun30.parquet")

# Importing Name x Country Matches
nanats = con.read_parquet(f"{root}/data/int/name2nat_aug1.parquet")

nts_long = con.read_parquet(f'{root}/data/int/rev_names_nametrace_long_jul8.parquet')

# Importing User x Position-level Data (all positions)
merged_pos = con.read_parquet(f"{root}/data/int/wrds_positions_aug1.parquet")

# Occupation crosswalk
occ_cw = con.read_csv(f"{root}/data/crosswalks/rev_occ_to_foia_freq.csv")

print('Done!')

#####################################
# DEFINING MAIN SAMPLE OF USERS (RCID IN MAIN SAMP AND START DATE AFTER 2015)
#####################################
print('Defining Main Sample of H-1B Companies...')
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
print(f'Employer x Years with Fewer than 50 Apps: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE n_apps_tot < 50").df().iloc[0,0]}')
print(f'Employer x Years with Fewer than 50% Duplicates: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE share_multireg < 0.5").df().iloc[0,0]}')
print(f'Employer x Years with No Duplicates: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE share_multireg = 0").df().iloc[0,0]}')

# main sample (conservative): companies with fewer than 50 applications and no duplicate registrations TODO: declare these as constants at top
foia_main_samp = con.sql(f'SELECT * FROM foia_main_samp_unfilt WHERE n_apps_tot <= {CUTOFF_N_APPS_TOT} AND share_multireg <= {CUTOFF_SHARE_MULTIREG}')
print(f"Preferred Sample: {foia_main_samp.df().shape[0]} ({round(100*foia_main_samp.df().shape[0]/n)}%)")

# computing win rate by sample
foia_main_samp_def = con.sql(f"SELECT *, CASE WHEN n_apps_tot <= {CUTOFF_N_APPS_TOT} AND share_multireg <= {CUTOFF_SHARE_MULTIREG} THEN 'insamp' ELSE 'outsamp' END AS sampgroup FROM foia_main_samp_unfilt")
con.sql("SELECT sampgroup, SUM(n_success)/SUM(n_apps_tot) AS total_win_rate FROM foia_main_samp_def GROUP BY sampgroup")

# creating crosswalk between foia id and rcid (joining list of FEINs in and out of sample with foia data with foia ids and joining that to id crosswalk, then joining to dup_rcids)
samp_to_rcid = con.sql("SELECT a.FEIN, a.lottery_year, sampgroup, b.foia_id, c.main_rcid, CASE WHEN rcid IS NULL THEN c.main_rcid ELSE d.rcid END AS rcid FROM ((SELECT FEIN, lottery_year, sampgroup FROM foia_main_samp_def) AS a JOIN (SELECT FEIN, lottery_year, foia_id FROM foia_with_ids GROUP BY FEIN, lottery_year, foia_id) AS b ON a.FEIN = b.FEIN AND a.lottery_year = b.lottery_year JOIN (SELECT main_rcid, foia_id FROM id_merge) AS c ON b.foia_id = c.foia_id) LEFT JOIN (SELECT main_rcid, rcid FROM dup_rcids) AS d ON c.main_rcid = d.main_rcid")

# writing company sample crosswalk to file
con.sql(f"COPY samp_to_rcid TO '{root}/data/int/company_merge_sample_jul10.parquet'")

# selecting user ids from list of positions based on whether company in sample and start date is after 2015 (conservative bandwidth) -- TODO: declare cutoff date as constant; TODO: move startdate filter into merged_pos query, avoid pulling people who got promoted after 2015 but started working before 2015
user_samp = con.sql("SELECT user_id FROM ((SELECT rcid FROM samp_to_rcid WHERE sampgroup = 'insamp' GROUP BY rcid) AS a JOIN (SELECT user_id, rcid FROM merged_pos WHERE country = 'United States' AND startdate >= '2015-01-01') AS b ON a.rcid = b.rcid) GROUP BY user_id")
print('Done!')

#####################################
### CLEANING AND SAVING FOIA INDIVIDUAL DATA
#####################################
print('Cleaning and Saving Individual-level H-1B Data...')
foia_indiv = con.sql(
f"""SELECT a.FEIN, a.lottery_year, country, 
        get_country_subregion(country) AS subregion, 
        female_ind, yob, status_type, ben_multi_reg_ind, employer_name, 
        CASE WHEN BEN_EDUCATION_CODE = 'I' THEN 'Doctor'
            WHEN BEN_EDUCATION_CODE = 'G' OR BEN_EDUCATION_CODE = 'H' THEN 'Master'
            WHEN BEN_EDUCATION_CODE = 'F' THEN 'Bachelor'
            WHEN BEN_EDUCATION_CODE = 'NA' THEN NULL
            ELSE 'Other' END AS highest_ed_level,
        CASE WHEN BEN_CURRENT_CLASS IN ('F1','F2') THEN 'F Visa'
            WHEN BEN_CURRENT_CLASS IN ('UU', 'UN', 'B2', 'B1') THEN 'No Visa'
            WHEN BEN_CURRENT_CLASS = 'NA' THEN NULL
            ELSE 'Other' END AS prev_visa,
        CASE WHEN S3Q1 = 'M' THEN 1 WHEN S3Q1 = 'NA' THEN NULL ELSE 0 END AS ade_lottery, NAICS4,
        CASE WHEN NAICS4 = '5415' THEN 'Computer Systems' 
            WHEN NAICS4 = '5413' THEN 'Engineering/Architectural Services'
            WHEN NAICS4 = '5416' THEN 'Consulting Services'
            WHEN NAICS4 = '5417' THEN 'Scientific Research'
            WHEN NAICS2 = '52' THEN 'Finance and Insurance'
            WHEN NAICS4 = '5411' OR NAICS4 = '5412' THEN 'Legal and Accounting'
            WHEN NAICS2 = '54' THEN 'Other Professional Services'
            WHEN NAICS4 = '3254' THEN 'Pharmaceuticals'
            WHEN NAICS4 = '5112' OR NAICS4 = '5182' THEN 'Software and Data or Web Services'
            WHEN NAICS2 = '23' OR NAICS2 = '22' THEN 'Construction and Utilities'
            WHEN NAICS2 = '33' THEN 'Manufacturing'
            WHEN NAICS2 = '62' THEN 'Health Care'
            WHEN NAICS2 = '51' THEN 'Other Information'
            WHEN NAICS_CODE = 'NA' OR NAICS2 = '99' THEN NULL
            ELSE 'Other' END AS industry,
        {help.field_clean_regex_sql('BEN_PFIELD_OF_STUDY')} AS field_clean, DOT_CODE, {help.inst_clean_regex_sql('JOB_TITLE')} AS job_title, n_apps, n_unique_country, foia_indiv_id, main_rcid, rcid 
    FROM (
        SELECT FEIN, lottery_year, get_std_country(country_of_nationality) AS country, 
            CASE WHEN gender = 'female' THEN 1 ELSE 0 END AS female_ind, ben_year_of_birth AS yob, status_type, ben_multi_reg_ind, employer_name, BEN_PFIELD_OF_STUDY, BEN_EDUCATION_CODE, DOT_CODE, NAICS_CODE, SUBSTRING(NAICS_CODE, 1, 4) AS NAICS4, SUBSTRING(NAICS_CODE, 1, 2) AS NAICS2, JOB_TITLE, BEN_CURRENT_CLASS, S3Q1, COUNT(*) OVER(PARTITION BY FEIN, lottery_year) AS n_apps, COUNT(DISTINCT country_of_birth) OVER(PARTITION BY FEIN, lottery_year) AS n_unique_country, ROW_NUMBER() OVER() AS foia_indiv_id 
        FROM foia_raw_file WHERE FEIN != '(b)(3) (b)(6) (b)(7)(c)'
    ) AS a JOIN samp_to_rcid AS b ON a.FEIN = b.FEIN AND a.lottery_year = b.lottery_year WHERE sampgroup = 'insamp'""")

con.sql(f"COPY foia_indiv TO '{root}/data/clean/foia_indiv.parquet'")
print('Done!')

#####################################
### CLEANING AND MERGING REVELIO USERS TO COUNTRIES
#####################################
print('Cleaning and Saving Individual-level Revelio User Data...')

# Cleaning Revelio Data, removing duplicates
rev_clean = con.sql(
f"""
SELECT * FROM
    (SELECT 
    fullname, degree, user_id, rcid,
    {help.degree_clean_regex_sql()} AS degree_clean,
    {help.inst_clean_regex_sql('university_raw')} AS univ_raw_clean,
    {help.stem_ind_regex_sql()} AS stem_ind,
    {help.field_clean_regex_sql('field_raw')} AS field_clean,
    CASE WHEN fullname ~ '.*[A-z].*' THEN {help.fullname_clean_regex_sql('fullname')} ELSE '' END AS fullname_clean,
    university_raw, f_prob, education_number, ed_enddate, ed_startdate, ROW_NUMBER() OVER(PARTITION BY user_id, education_number) AS dup_num, updated_dt
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
nanats_long = con.sql(f"SELECT fullname_clean, nanat_country, SUM(nanat_prob) AS nanat_prob FROM (SELECT fullname_clean, get_std_country(raw_eth) AS nanat_country, nanat_prob FROM (UNPIVOT (SELECT fullname_clean, UNNEST(pred_nats_name) FROM nanats) ON COLUMNS(* EXCLUDE (fullname_clean)) INTO NAME raw_eth VALUE nanat_prob)) GROUP BY fullname_clean, nanat_country")

# # # temp for testing
# nanats_long = con.sql(f"SELECT fullname_clean, nanat_country, SUM(nanat_prob) AS nanat_prob FROM (SELECT fullname_clean, get_std_country({help.nanats_to_long('pred_nats_name')}[1]) AS nanat_country, {help.nanats_to_long('pred_nats_name')}[2]::FLOAT AS nanat_prob FROM (SELECT * FROM nanats AS a RIGHT JOIN (SELECT fullname_clean FROM rev_users_filt GROUP BY fullname_clean) AS b ON a.fullname_clean = b.fullname_clean)) GROUP BY fullname_clean, nanat_country")
# nts_long = con.sql("SELECT * FROM nts_long AS a RIGHT JOIN (SELECT fullname_clean FROM rev_users_filt GROUP BY fullname_clean) AS b ON a.fullname_clean = b.fullname_clean")

# Cleaning institution matches 
inst_match_clean = con.sql(
"""
    SELECT *, 
        COUNT(*) OVER(PARTITION BY university_raw) AS n_country_match 
    FROM inst_country_cw
    WHERE lower(REGEXP_REPLACE(university_raw, '[^A-z]', '', 'g')) NOT IN ('highschool', 'ged', 'unknown', 'invalid')
""")

# Merging users with institution matches (long) and collapsing to user x country level, filtering out 
inst_merge_long = con.sql(
"""
SELECT user_id, university_raw, get_std_country(match_country) AS match_country, 
    us_hs_exact, us_educ, ade_ind, ade_year, 
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
        MAX(CASE WHEN degree_clean != 'Non-Degree' AND match_country = 'United States' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS us_educ,
    -- ADE status 
        MAX(CASE WHEN degree_clean IN ('Master', 'Doctor', 'MBA') AND match_country = 'United States' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS ade_ind,
    -- ADE year
        MAX(CASE WHEN degree_clean IN ('Master', 'MBA') AND match_country = 'United States' 
                THEN (CASE WHEN ed_enddate IS NULL AND ed_startdate IS NOT NULL THEN SUBSTRING(ed_startdate, 1, 4)::INT + 1 ELSE SUBSTRING(ed_enddate, 1, 4)::INT END) 
            WHEN degree_clean = 'Doctor' AND match_country = 'United States' 
                THEN (CASE WHEN ed_enddate IS NULL AND ed_startdate IS NOT NULL THEN SUBSTRING(ed_startdate, 1, 4)::INT + 4 ELSE SUBSTRING(ed_enddate, 1, 4)::INT END) 
            END) OVER(PARTITION BY user_id) AS ade_year
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

# Merging users with name2nat matches (long on user x country)
name_merge_long = con.sql(
"""
SELECT user_id, a.fullname_clean, nanat_country, nanat_prob FROM 
    (SELECT fullname_clean, user_id FROM rev_users_filt GROUP BY fullname_clean, user_id) AS a 
    JOIN 
    nanats_long AS b 
    ON a.fullname_clean = b.fullname_clean 
"""
)

# Merging users with nametrace matches (long on user x subregion)
nt_merge_long = con.sql("""SELECT user_id, a.fullname_clean, region, prob, f_prob_nt FROM (SELECT fullname_clean, user_id FROM rev_users_filt GROUP BY fullname_clean, user_id) AS a 
    JOIN 
    nts_long AS b 
    ON a.fullname_clean = b.fullname_clean """)

country_merge_long = con.sql(
"""
SELECT user_id, MAX(fullname_clean) OVER(PARTITION BY user_id), country, subregion, nanat_score, inst_score, MAX(f_prob_nt) OVER(PARTITION BY user_id) AS f_prob_nt, SUM(nanat_score) OVER(PARTITION BY user_id, subregion) AS nanat_subregion_score, nt_subregion_score, university_raw, inst_score, us_hs_exact, us_educ, ade_ind, ade_year
FROM (
SELECT  
    CASE WHEN countries.user_id IS NULL THEN nt.user_id ELSE countries.user_id END AS user_id,
    CASE WHEN countries.fullname_clean IS NULL THEN nt.fullname_clean ELSE countries.fullname_clean END AS fullname_clean,
    country,
    CASE WHEN countries.subregion IS NULL THEN nt.region ELSE countries.subregion END AS subregion,
    CASE WHEN countries.nanat_score IS NULL THEN 0 ELSE nanat_score END AS nanat_score,
    f_prob_nt, 
    CASE WHEN prob IS NULL THEN 0 ELSE prob END AS nt_subregion_score, university_raw, 
    CASE WHEN countries.inst_score IS NULL THEN 0 ELSE inst_score END AS inst_score, us_hs_exact, us_educ, ade_ind, ade_year
FROM (
    SELECT 
        CASE WHEN a.user_id IS NULL THEN b.user_id ELSE a.user_id END AS user_id, fullname_clean, university_raw,
        CASE WHEN match_country IS NULL THEN nanat_country ELSE match_country END AS country,
        CASE WHEN match_country IS NULL THEN get_country_subregion(nanat_country) ELSE get_country_subregion(match_country) END AS subregion, 
        CASE WHEN match_country IS NULL THEN 0 ELSE matchscore_corr END AS inst_score, 
        CASE WHEN nanat_country IS NULL THEN 0 ELSE nanat_prob END AS nanat_score,
        us_hs_exact, us_educ, ade_ind, ade_year
    FROM inst_merge_long AS a 
    FULL JOIN name_merge_long AS b 
    ON a.user_id = b.user_id AND a.match_country = b.nanat_country
) AS countries 
FULL JOIN nt_merge_long AS nt
ON countries.user_id = nt.user_id AND countries.subregion = nt.region)
""")

#####################################
### CLEANING POSITION DATA
#####################################
## testing
# ids = ",".join(con.sql(help.random_ids_sql('user_id','rev_clean', n = 100)).df()['user_id'].astype(str))
# pos_filt = con.sql(f"SELECT * FROM merged_pos WHERE user_id IN ({ids})")

# # TODO:Cleaning position history -- (impute rcid if company name similar to below,) get rid of duplicates -- also need to clean duplicate users!

merged_pos_clean = con.sql("SELECT *, CASE WHEN max_share_foia > 0 THEN 1 ELSE 0 END AS foia_occ_ind FROM merged_pos LEFT JOIN occ_cw ON merged_pos.role_k1500 = occ_cw.role_k1500")

# x = con.sql("SELECT * FROM pos_filt AS a JOIN pos_filt AS b ON a.user_id = b.user_id AND (a.startdate = b.startdate AND a.enddate = b.enddate)")


# 8/6/25: what to do with null enddates? if last position, can impute as updated dt, but if not last position? if i take next start date, this will cut out freelance/part-time work or work preceding freelance/part-time work (e.g. if i have a FT job and start doing something on the side and hold both jobs concurrently)
# okay for now just leave as today if null (most conservative -- downside is that enddatediff filter will be less effective)


# User-level indicator for H-1B occupation
merged_pos_cw = con.sql("SELECT user_id, MAX(foia_occ_ind) AS foia_occ_ind, MIN(min_rank) AS min_h1b_occ_rank FROM merged_pos_clean GROUP BY user_id")

# Cleaning position history and aggregating to user level
merged_pos_user = con.sql(f"""
    SELECT user_id, 
        ARRAY_AGG(title_clean ORDER BY position_number) AS positions,
        ARRAY_AGG(rcid ORDER BY position_number) AS rcids,
        MIN(CASE WHEN position_number > max_intern_position THEN startdate ELSE NULL END) AS min_startdate,
        MIN(CASE WHEN position_number > max_intern_position AND country = 'United States' THEN startdate ELSE NULL END) AS min_startdate_us
    FROM (
        SELECT *, MAX(CASE WHEN intern_ind = 1 THEN position_number ELSE 0 END) OVER(PARTITION BY user_id) AS max_intern_position FROM (
            SELECT user_id, {help.inst_clean_regex_sql('title_raw')} AS title_clean, position_number, rcid, country, startdate, enddate, role_k1500,
                CASE WHEN 
                    (lower(title_raw) ~ '(^|\\s)(intern)($|\\s)' AND 
                        DATEDIFF('month', startdate::DATETIME, enddate::DATETIME) < 12) 
                    OR (lower(title_raw) ~ '(^|\\s)(student)($|\\s)') 
                THEN 1 ELSE 0 END AS intern_ind
            FROM merged_pos
        )
    ) GROUP BY user_id
    """
    )

#####################################
### GETTING AND EXPORTING FINAL USER FILE
#####################################
final_user_merge = con.sql(
f"""SELECT * FROM (SELECT a.user_id, est_yob, hs_ind, valid_postsec, updated_dt, f_prob, stem_ind, f_prob_nt, fullname, university_raw, country, subregion, inst_score, nanat_score, nanat_subregion_score, nt_subregion_score, 0.5*inst_score + 0.5*nanat_score AS total_score, 0.5*inst_score + 0.25*nanat_subregion_score + 0.25*nt_subregion_score AS total_subregion_score,
        MAX(us_hs_exact) OVER(PARTITION BY a.user_id) AS us_hs_exact, 
        MAX(us_educ) OVER(PARTITION BY a.user_id) AS us_educ,
        MAX(CASE WHEN ade_ind IS NULL THEN 0 ELSE ade_ind END) OVER(PARTITION BY a.user_id) AS ade_ind,
        MIN(ade_year) OVER(PARTITION BY a.user_id) AS ade_year,
        MAX(0.5*inst_score + 0.5*nanat_score) OVER(PARTITION BY a.user_id) AS max_total_score,
        MAX(CASE WHEN country = 'United States' THEN 0 ELSE 0.5*inst_score + 0.5*nanat_score END) OVER(PARTITION BY a.user_id) AS max_total_score_nonus,
        foia_occ_ind, min_h1b_occ_rank, fields, highest_ed_level
    FROM (
        SELECT user_id, est_yob, f_prob, fullname, hs_ind, valid_postsec, updated_dt, MAX(stem_ind_postsec) AS stem_ind, ARRAY_AGG(field_clean) FILTER (WHERE field_clean IS NOT NULL) AS fields, 
        CASE WHEN MAX(CASE WHEN degree_clean = 'Doctor' THEN 1 ELSE 0 END) = 1 THEN 'Doctor' 
            WHEN MAX(CASE WHEN degree_clean IN ('Master', 'MBA') THEN 1 ELSE 0 END) = 1 THEN 'Master'
            WHEN MAX(CASE WHEN degree_clean IN ('Bachelor') THEN 1 ELSE 0 END) = 1 THEN 'Bachelor'
            WHEN MAX(CASE WHEN degree_clean IS NOT NULL THEN 1 ELSE 0 END) = 1 THEN 'Other'
            ELSE NULL END AS highest_ed_level
        FROM (
            SELECT user_id, CASE WHEN degree_clean IN ('Non-Degree', 'High School', 'Associate') THEN 0 ELSE stem_ind END AS stem_ind_postsec,
                {help.get_est_yob()} AS est_yob, 
                MAX(CASE WHEN degree_clean = 'High School' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS hs_ind,
                MAX(CASE WHEN degree_clean NOT IN ('Non-Degree', 'Master', 'Doctor', 'MBA') AND (ed_enddate IS NOT NULL OR ed_startdate IS NOT NULL) THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS valid_postsec, updated_dt, field_clean, degree_clean,
                f_prob, fullname FROM rev_users_filt
        ) GROUP BY user_id, est_yob, f_prob, fullname, hs_ind, valid_postsec, updated_dt
    ) AS a 
LEFT JOIN country_merge_long AS b ON a.user_id = b.user_id
LEFT JOIN merged_pos_cw AS c ON a.user_id = c.user_id)
WHERE (max_total_score_nonus < 0.3 OR max_total_score_nonus = total_score) AND (total_score >= 0.01 OR nanat_subregion_score + nt_subregion_score >= 0.05)
""")

# # saving intermediate version
# con.sql(f"COPY final_user_merge TO '{root}/data/int/rev_users_clean_jul28.parquet'")

# final_user_merge = con.read_parquet(f'{root}/data/int/rev_users_clean_jul28.parquet')

## FURTHER COLLAPSING
# cleaning revelio data (collapsing to user x company x country level)
rev_indiv = con.sql(
"""
SELECT * FROM (
    (SELECT user_id, 
        MIN(startdate)::DATETIME AS first_startdate, 
        MAX(CASE WHEN enddate IS NULL THEN '2025-03-01' ELSE enddate END)::DATETIME AS last_enddate, rcid 
    FROM merged_pos WHERE country = 'United States' AND startdate >= '2015-01-01' GROUP BY user_id, rcid) AS a 
    JOIN 
    (SELECT rcid FROM samp_to_rcid GROUP BY rcid) AS b 
    ON a.rcid = b.rcid
) AS pos 
JOIN 
(SELECT * FROM final_user_merge WHERE (us_hs_exact IS NULL OR us_hs_exact = 0) AND (us_educ IS NULL OR us_educ = 1)) AS users 
ON pos.user_id = users.user_id
LEFT JOIN
merged_pos_user AS poshist
ON pos.user_id = poshist.user_id
""")

con.sql(f"COPY rev_indiv TO '{root}/data/clean/rev_indiv.parquet'")
print('Done!')

