# File Description: Individual Merge for Master's Subsample
# Author: Amy Kim
# Date Created: Tue Apr 1

# Imports and Paths
import duckdb as ddb
import pandas as pd
import numpy as np
import fiscalyear
import seaborn as sns
import employer_merge_helpers as emh
import analysis_helpers as ah
import wrds
import re

root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

con = ddb.connect()

# creating sql functions
con.create_function("get_fiscal_year", ah.get_fiscal_year_sql, ["VARCHAR"], "FLOAT")

con.create_function("get_fiscal_year_foia", ah.get_fiscal_year_foia_sql, ["VARCHAR"], "FLOAT")
    
con.create_function("get_quarter", ah.get_quarter_sql, ["VARCHAR"], "FLOAT")

con.create_function("get_quarter_foia", ah.get_quarter_foia_sql, ["VARCHAR"], "FLOAT")

#####################
# IMPORTING DATA
#####################
## duplicate rcids (companies that appear more than once in linkedin data)
dup_rcids = con.read_csv(f"{root}/data/int/dup_rcids_mar20.csv")
dup_rcids_unique = con.sql("SELECT rcid, main_rcid FROM dup_rcids GROUP BY rcid, main_rcid")

## matched company data from R
rmerge = con.read_csv(f"{root}/data/int/good_match_ids_mar20.csv")

## raw FOIA bloomberg data
foia_raw_file = con.read_csv(f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv")

## raw FOIA I129 data
foia_i129_file = con.read_csv(f"{root}/data/raw/foia_i129/i129_allfys.csv")

## joining raw FOIA data with merged data to get foia_ids in raw foia data
foia_with_ids = con.sql("SELECT ROW_NUMBER() OVER() AS bb_id, foia_id, BEN_COUNTRY_OF_BIRTH, ben_year_of_birth, BEN_SEX, JOB_TITLE, DOT_CODE, NAICS_CODE, S3Q1, PET_ZIP, WORKSITE_CITY, WORKSITE_STATE, BASIS_FOR_CLASSIFICATION,  state_name, worksite_state_name, REQUESTED_ACTION, BEN_PFIELD_OF_STUDY, BEN_EDUCATION_CODE, BEN_COMP_PAID, valid_from, valid_to, employer_name, i129_employer_name, NUM_OF_EMP_IN_US, a.FEIN AS FEIN, status_type, ben_multi_reg_ind, rec_date, first_decision_date, FIRST_DECISION, a.lottery_year AS lottery_year FROM ((SELECT * FROM foia_raw_file WHERE NOT FEIN = '(b)(3) (b)(6) (b)(7)(c)') AS a LEFT JOIN (SELECT lottery_year, FEIN, foia_id FROM rmerge GROUP BY lottery_year, FEIN, foia_id) AS b ON a.lottery_year = b.lottery_year AND a.FEIN = b.FEIN)")

id_merge = con.sql("""
-- collapse matched IDs to be unique at the rcid x FEIN x lottery_year level
SELECT lottery_year, main_rcid, foia_id
FROM rmerge 
GROUP BY foia_id, lottery_year, main_rcid
""")


#####################################
# 0. MERGING FOIA DATASETS
#####################################
i129_with_ids = con.sql("SELECT *, ROW_NUMBER() OVER() AS i129_id FROM foia_i129_file")
i129_for_merge = con.sql(
"""SELECT i129_id, DOT_CODE, BEN_EDUCATION_CODE, BASIS_FOR_CLASSIFICATION, REQUESTED_ACTION, BEN_COUNTRY_OF_BIRTH, FIRST_DECISION, S3Q1, JOB_TITLE, EMPLOYER_NAME, institution_txt,
    CASE WHEN PET_ZIP = 'NA' THEN NULL ELSE SUBSTRING(PET_ZIP, 1, 5)::INT END AS PET_ZIP,
    CASE WHEN BEN_COMP_PAID = 'NA' THEN 0 ELSE REPLACE(REPLACE(BEN_COMP_PAID, ',', '.'), ' ', '.')::FLOAT END AS BEN_COMP_PAID,
    CASE WHEN NAICS_CODE = 'NA' OR NAICS_CODE = 'I129' THEN NULL ELSE SUBSTRING(NAICS_CODE, 1, 6)::INT END AS NAICS_CODE,
    CASE WHEN REC_DATE = 'NA' THEN NULL ELSE SUBSTRING(REC_DATE, 1,4)::INT END AS rec_yr, 
    CASE WHEN REC_DATE = 'NA' THEN NULL ELSE SUBSTRING(REC_DATE, 6,2)::INT END AS rec_mo, 
    CASE WHEN REC_DATE = 'NA' THEN NULL ELSE SUBSTRING(REC_DATE, 9, 2)::INT END AS rec_day,
    CASE WHEN VALID_FROM = 'NA' THEN NULL ELSE SUBSTRING(VALID_FROM, 1,4)::INT END AS valid_from_yr, 
    CASE WHEN VALID_FROM = 'NA' THEN NULL ELSE SUBSTRING(VALID_FROM, 6,2)::INT END AS valid_from_mo, 
    CASE WHEN VALID_FROM = 'NA' THEN NULL ELSE SUBSTRING(VALID_FROM, 9, 2)::INT END AS valid_from_day,
    CASE WHEN VALID_TO = 'NA' THEN NULL ELSE SUBSTRING(VALID_TO, 1,4)::INT END AS valid_to_yr, 
    CASE WHEN VALID_TO = 'NA' THEN NULL ELSE SUBSTRING(VALID_TO, 6,2)::INT END AS valid_to_mo, 
    CASE WHEN VALID_TO = 'NA' THEN NULL ELSE SUBSTRING(VALID_TO, 9, 2)::INT END AS valid_to_day,
    CASE WHEN ACT_DATE = 'NA' THEN NULL ELSE SUBSTRING(ACT_DATE , 1,4)::INT END AS act_yr, 
    CASE WHEN ACT_DATE = 'NA' THEN NULL ELSE SUBSTRING(ACT_DATE , 6,2)::INT END AS act_mo, 
    CASE WHEN ACT_DATE = 'NA' THEN NULL ELSE SUBSTRING(ACT_DATE , 9, 2)::INT END AS act_day
FROM i129_with_ids
WHERE REC_DATE != 'NA'""")

bb_for_merge = con.sql(
f"""SELECT bb_id, lottery_year, foia_id, FEIN, DOT_CODE, BEN_EDUCATION_CODE, ben_year_of_birth, BASIS_FOR_CLASSIFICATION, BEN_SEX, BEN_PFIELD_OF_STUDY, REQUESTED_ACTION, WORKSITE_STATE, WORKSITE_CITY, BEN_COUNTRY_OF_BIRTH, FIRST_DECISION, S3Q1, JOB_TITLE, employer_name, i129_employer_name, state_name, worksite_state_name,
    CASE WHEN PET_ZIP = 'NA' THEN NULL ELSE SUBSTRING(PET_ZIP, 1, 5)::INT END AS PET_ZIP,
    CASE WHEN BEN_COMP_PAID = 'NA' THEN 0 ELSE BEN_COMP_PAID::FLOAT END AS BEN_COMP_PAID,
    CASE WHEN NAICS_CODE = 'NA' THEN NULL ELSE SUBSTRING(NAICS_CODE, 1, 6)::INT END AS NAICS_CODE,
    {ah.get_info_from_bb_date('rec_date','month')} AS rec_mo,
    {ah.get_info_from_bb_date('rec_date','day')} AS rec_day,
    {ah.get_info_from_bb_date('rec_date','year')} AS rec_yr,
    {ah.get_info_from_bb_date('valid_from','month')} AS valid_from_mo,
    {ah.get_info_from_bb_date('valid_from','day')} AS valid_from_day,
    {ah.get_info_from_bb_date('valid_from','year')} AS valid_from_yr,
    {ah.get_info_from_bb_date('valid_to','month')} AS valid_to_mo,
    {ah.get_info_from_bb_date('valid_to','day')} AS valid_to_day,
    {ah.get_info_from_bb_date('valid_to','year')} AS valid_to_yr,
    {ah.get_info_from_bb_date('first_decision_date','month')} AS act_mo,
    {ah.get_info_from_bb_date('first_decision_date','day')} AS act_day,
    {ah.get_info_from_bb_date('first_decision_date','year')} AS act_yr
FROM foia_with_ids 
WHERE rec_date != 'NA' AND lottery_year != '2024'
""")

con.sql(
"""SELECT COUNT(*), COUNT(DISTINCT bb_id)
FROM i129_for_merge AS a 
JOIN bb_for_merge AS b 
ON a.rec_yr = b.rec_yr 
AND a.rec_mo = b.rec_mo 
AND a.rec_day = b.rec_day 
AND a.DOT_CODE = b.DOT_CODE 
AND a.BEN_EDUCATION_CODE = b.BEN_EDUCATION_CODE
AND a.NAICS_CODE = b.NAICS_CODE 
AND a.BASIS_FOR_CLASSIFICATION = b.BASIS_FOR_CLASSIFICATION 
AND a.REQUESTED_ACTION = b.REQUESTED_ACTION 
AND a.BEN_COUNTRY_OF_BIRTH = b.BEN_COUNTRY_OF_BIRTH 
AND a.FIRST_DECISION = b.FIRST_DECISION 
AND a.EMPLOYER_NAME = b.i129_employer_name
AND a.S3Q1 = b.S3Q1 
AND a.JOB_TITLE = b.JOB_TITLE
AND a.BEN_COMP_PAID = b.BEN_COMP_PAID
AND a.act_yr = b.act_yr 
AND a.act_mo = b.act_mo
AND a.act_day = b.act_day
""")

indiv_matched = con.sql(
"""SELECT bb_id, i129_id, foia_id, FEIN, a.rec_yr, a.rec_mo, a.rec_day, a.DOT_CODE, b.BEN_SEX, a.BEN_EDUCATION_CODE, b.BEN_PFIELD_OF_STUDY, a.NAICS_CODE, a.BASIS_FOR_CLASSIFICATION, a.REQUESTED_ACTION, a.BEN_COUNTRY_OF_BIRTH, b.WORKSITE_CITY, b.WORKSITE_STATE, a.FIRST_DECISION, a.PET_ZIP, b.ben_year_of_birth, a.EMPLOYER_NAME, b.i129_employer_name, a.S3Q1, a.JOB_TITLE, b.employer_name, a.BEN_COMP_PAID AS i129_comp, b.BEN_COMP_PAID AS bb_comp, a.valid_from_yr AS i129vfy, b.valid_from_yr AS bbvfy, a.valid_from_mo AS i129vfm, b.valid_from_mo AS bbvfm, a.valid_from_day AS i129vfd, b.valid_from_day AS bbvfd, COUNT(i129_id) OVER(PARTITION BY bb_id) AS dupmatch_tobb, institution_txt, state_name, worksite_state_name
FROM i129_for_merge AS a 
JOIN bb_for_merge AS b 
ON a.rec_yr = b.rec_yr 
AND a.rec_mo = b.rec_mo 
AND a.rec_day = b.rec_day 
AND a.DOT_CODE = b.DOT_CODE 
AND a.BEN_EDUCATION_CODE = b.BEN_EDUCATION_CODE
AND a.NAICS_CODE = b.NAICS_CODE 
AND a.BASIS_FOR_CLASSIFICATION = b.BASIS_FOR_CLASSIFICATION 
AND a.REQUESTED_ACTION = b.REQUESTED_ACTION 
AND a.BEN_COUNTRY_OF_BIRTH = b.BEN_COUNTRY_OF_BIRTH 
AND a.FIRST_DECISION = b.FIRST_DECISION 
AND a.EMPLOYER_NAME = b.i129_employer_name
AND a.S3Q1 = b.S3Q1 
AND a.JOB_TITLE = b.JOB_TITLE
AND a.BEN_COMP_PAID = b.BEN_COMP_PAID
AND a.act_yr = b.act_yr 
AND a.act_mo = b.act_mo
AND a.act_day = b.act_day
""")

#con.sql(f"COPY indiv_matched TO '{root}/data/int/indiv_matched_mar31.parquet'")

## merging to get rcids
indiv_matched_with_rcids = con.sql("SELECT * FROM (SELECT * FROM indiv_matched WHERE dupmatch_tobb = 1 AND S3Q1 = 'M') AS a LEFT JOIN id_merge AS b ON a.foia_id = b.foia_id")

# ## merging with revelio data
# # con.sql("ATTACH 'host = wrds-pgdata.wharton.upenn.edu port = 9737 dbname = wrds user = amykimecon password = nojqiq-berDoz-4fobke' AS wrdsdb (TYPE postgres, READ_ONLY, SCHEMA 'revelio')")
# rcidlist_m = list(con.sql("SELECT main_rcid FROM indiv_matched_with_rcids WHERE main_rcid IS NOT NULL GROUP BY main_rcid").df()['main_rcid'])

# def getmergequery(rcidlist):
#     return f"""SELECT a.user_id AS user_id, a.position_id AS a, country, state, metro_area, a.startdate AS pos_startdate, a.enddate AS pos_enddate, role_k1500, weight, start_salary, end_salary, seniority, salary, position_number, rcid, total_compensation, fullname, highest_degree, sex_predicted, ethnicity_predicted, user_location, user_country, updated_dt, university_name, c.education_number, c.startdate AS ed_startdate, c.enddate AS ed_enddate, degree, field, university_country, title_raw, university_raw, degree_raw, field_raw FROM (SELECT * FROM revelio.individual_positions WHERE country = 'United States' AND rcid IN ({','.join([str(i) for i in rcidlist])})) AS a LEFT JOIN (SELECT * FROM revelio.individual_user) AS b ON a.user_id = b.user_id LEFT JOIN (SELECT * FROM revelio.individual_user_education) AS c ON a.user_id = c.user_id LEFT JOIN (SELECT position_id, company_raw, location_raw, title_raw, description FROM revelio.individual_positions_raw) AS d ON a.position_id = d.position_id LEFT JOIN (SELECT * FROM revelio.individual_user_education_raw) AS e ON c.user_id = e.user_id AND c.education_number=e.education_number WHERE university_raw IS NOT NULL """

# ## WRDS
# db = wrds.Connection(wrds_username='amykimecon')

# # merged = []
# # i = 0
# # d = 10

# # while d*(i+1) < len(rcidlist_m):
# #     print(i*d)
# #     merged = merged + [db.raw_sql(getmergequery(rcidlist_m[d*i:d*(i+1)]))]
# #     i += 1    
# # merged = merged + [db.raw_sql(getmergequery(rcidlist_m[d*i:]))]

# # merged_all = pd.concat(merged)
# # merged_all.to_parquet(f"{root}/data/int/rev_merge_masters_i129_mar31.parquet")

# ## TESTING
# dataout = db.raw_sql(getmergequery(rcidlist_m[0:10]))
# datamatch = con.sql(f"SELECT * FROM indiv_matched_with_rcids WHERE main_rcid IN ({','.join([str(i) for i in rcidlist_m[0:10]])})").df()

# ## cleaning and tokenizing institution name strings
# dataout['inst_clean_rev'] = dataout.apply(lambda x: re.sub('[^a-z\\s]', '', x['university_raw'].lower()) if pd.isnull(x['university_name']) else re.sub('[^a-z\\s]', '', x['university_name'].lower()), axis = 1)
# dataout['inst_clean_rev_tokens'] = dataout.apply(lambda x: [None] if x['inst_clean_rev'] == 'na' or x['inst_clean_rev'].strip() == '' else [item for item in x['inst_clean_rev'].split()], axis = 1)
                                      
# datamatch['inst_clean_foia'] = datamatch['institution_txt'].str.lower().str.replace('[^a-z\\s]','',regex=True)
# datamatch['inst_clean_foia_tokens'] = datamatch.apply(lambda x: [None] if x['inst_clean_foia'] == 'na' else [item for item in x['inst_clean_foia'].split()], axis = 1)

# ## imputing yob based on education start and end dates
# dataout['ed_startyr'] = dataout['ed_startdate'].apply(lambda x: None if pd.isnull(x) else int(x[0:4]))
# dataout['ed_endyr'] = dataout['ed_enddate'].apply(lambda x: None if pd.isnull(x) else int(x[0:4]))

# # helper function: getting est yob from each education entry
# def est_yob_int_func(df_row):
#     # if high school
#     if (not pd.isnull(df_row['degree']) and df_row['degree'] == "High School") or (not pd.isnull(re.search("high school", df_row['inst_clean_rev']))):
#         return df_row['ed_endyr'] - 18
#     elif not pd.isnull(df_row['degree']) and df_row['degree'] in ['Doctor', "MBA", "Master"]:
#         return None 
#     elif pd.isnull(df_row['degree']) and not pd.isnull(df_row['degree_raw']) and not pd.isnull(re.search("certificat|course|semester|exchange", df_row['degree_raw'].lower())):
#         return None 
#     else:
#         return df_row['ed_startyr'] - 18

# dataout['est_yob_int'] = dataout.apply(est_yob_int_func, axis = 1)

# dataout['est_yob'] = dataout.groupby('user_id')['est_yob_int'].transform('min')

# ## tokenizing names
# dataout_user_educ = dataout.drop_duplicates(subset = ['user_id','inst_clean_rev'])[['user_id','inst_clean_rev','inst_clean_rev_tokens']]
# dataout_user_educ['idnum'] = dataout_user_educ.reset_index().index
# dataout_user_educ['data'] = 'rev'
# dataout_user_educ['inst_clean'] = dataout_user_educ['inst_clean_rev_tokens']

# datamatch['inst_clean'] = datamatch['inst_clean_foia_tokens']
# datamatch['idnum'] = datamatch.reset_index().index
# datamatch['data'] = 'foia'

# univtokens = pd.concat([dataout_user_educ[['inst_clean','idnum', 'data']], datamatch[['inst_clean','idnum', 'data']]], ignore_index = True).explode('inst_clean')
# univtokens_freqs = (univtokens.groupby('inst_clean').count()/univtokens.shape[0]).reset_index()
# univtokens_freqs['freq'] = univtokens_freqs['idnum']

# ## merging 
# merge1 = dataout.loc[dataout['highest_degree'].isin(["Doctor", "MBA", "Master"])].merge(datamatch, left_on = ['rcid','sex_predicted'], right_on=['main_rcid','BEN_SEX'])

# # intersecting tokenized university names and getting product of frequency of overlaps
# def get_intersect(df_row):
#     if pd.isnull(df_row['inst_clean_rev_tokens']).any() or pd.isnull(df_row['inst_clean_foia_tokens']).any():
#         return []
#     else:
#         return list(set(df_row['inst_clean_rev_tokens']).intersection(set(df_row['inst_clean_foia_tokens'])))

# merge1['univ_intersect'] = merge1.apply(get_intersect, axis = 1)

# merge1_freqs_long = merge1.explode('univ_intersect').merge(univtokens_freqs, left_on = 'univ_intersect', right_on = 'inst_clean')
# merge1_freqs_long['rank'] = merge1_freqs_long.sort_values('freq').groupby(['a', 'education_number', 'bb_id']).cumcount() + 1
# merge1_freqs_long['top3freq'] = merge1_freqs_long.apply(lambda x: x['freq'] if x['rank'] <= 3 else None, axis = 1)
# merge1_freqs_long['rarefreq'] = merge1_freqs_long.apply(lambda x: x['freq'] if x['freq'] <= 0.025 else None, axis = 1)
                                 
# merge1_freqs = merge1.merge(
#     merge1_freqs_long.groupby(['a', 'education_number', 'bb_id'])[['freq','rarefreq', 'top3freq']].prod().reset_index(), 
#     how = 'left', 
#     on = ['a', 'education_number', 'bb_id'])
# merge1_freqs['simind'] = merge1_freqs.apply(lambda x: 0 if pd.isnull(x['freq']) else np.log10(x['freq']), axis = 1)
# merge1_freqs['simindrare'] = merge1_freqs.apply(lambda x: 0 if pd.isnull(x['rarefreq']) else np.log10(x['rarefreq']), axis = 1)
# merge1_freqs['simindtop3'] = merge1_freqs.apply(lambda x: 0 if pd.isnull(x['top3freq']) else np.log10(x['top3freq']), axis = 1)
# merge1_freqs[['univ_intersect','freq','simind','simindtop3','simindrare']]

# merge1_match1 = merge1_freqs.loc[(merge1_freqs['simindrare'] <= -5)|(merge1_freqs['inst_clean_rev'] == merge1_freqs['inst_clean_foia'])]

# merge1_match2 = merge1_freqs.loc[(merge1_freqs['inst_clean_rev'] == merge1_freqs['inst_clean_foia'])]


# merge1_match1.sort_values('bb_id')[['bb_id','fullname','BEN_COUNTRY_OF_BIRTH','sex_predicted','BEN_SEX','metro_area','WORKSITE_CITY','state','WORKSITE_STATE','title_raw', 'JOB_TITLE', 'university_name', 'institution_txt','field_raw', 'BEN_PFIELD_OF_STUDY','degree_raw','ed_startdate','ed_enddate','lottery_year']]

# # ## random sample of users
# # usamp = dataout.sample(n=10)['a']
# # usampdata = dataout.loc[dataout['a'].isin(usamp.tolist())]
# # usampdata.sort_values('a')[['a','university_name','university_raw','degree','degree_raw','ed_startdate','ed_enddate','est_yob_int','est_yob']]

# # merge1 = dataout.loc[dataout['highest_degree'].isin(["Doctor", "MBA", "Master"])].merge(datamatch, left_on = ['rcid','sex_predicted'], right_on=['main_rcid','BEN_SEX'])


# # def sim_score(df_row):
# #     if pd.isnull(df_row['inst_clean_rev_tokens']).any() or pd.isnull(df_row['inst_clean_foia_tokens']).any():
# #         return None
# #     else:
# #         int = len(list(set(df_row['inst_clean_rev_tokens']).intersection(set(df_row['inst_clean_foia_tokens']))))
# #         # un = (len(set(df_row['inst_clean_rev_tokens'])) + len(set(df_row['inst_clean_foia_tokens']))) - int
# #         foia_tot = len(set(df_row['inst_clean_foia_tokens']))
# #         return int/foia_tot

# # merge1['univ_jacc'] = merge1.apply(jacc, axis = 1)

# # merge1[['inst_clean_rev_tokens','inst_clean_foia_tokens', 'univ_intersect']]

# # # merge2 = merge1.loc[(merge1['university_name'].str.lower() == merge1['institution_txt'].str.lower())&]

# # merge2.sort_values('bb_id')[['bb_id','fullname','BEN_COUNTRY_OF_BIRTH','sex_predicted','BEN_SEX','metro_area','WORKSITE_CITY','WORKSITE_STATE','title_raw', 'JOB_TITLE', 'university_name', 'institution_txt','field_raw', 'BEN_PFIELD_OF_STUDY','degree_raw','ed_startdate','ed_enddate','lottery_year']]

# # dataout_user_educ = dataout[['user_id','degree_raw', 'degree', 'field_raw', 'field', 'university_name','university_raw']].drop_duplicates()
# # dataout_user_educ['institution_txt'] = dataout_user_educ['university_raw'].str.lower().str.replace('[^a-z\\s]','',regex=True)
# # dataout_user_educ['idnum'] = dataout_user_educ.reset_index().index 
# # dataout_user_educ['data'] = 'rev'

# # datamatch['institution_txt'] = datamatch['institution_txt'].str.lower().str.replace('[^a-z\\s]','',regex=True)
# # datamatch['idnum'] = datamatch.reset_index().index
# # datamatch['data'] = 'foia'
# # univnames = pd.concat([dataout_user_educ[['institution_txt','idnum', 'data']], datamatch[['institution_txt','idnum', 'data']]], ignore_index = True)

# #todo: finish this -- clean names
# # from name2nat import Name2nat 
# # my_nanat = Name2nat()

# #x = my_nanat(dataout.sample(n=10)['fullname'], top_n=3)

# # import requests
# # names = merge1['fullname'].unique()[0:10]
# # names_out = [requests.get(f"https://api.nationalize.io/?name={name}").json() for name in names]
# # names_merge = pd.DataFrame(data = {'names': names, 'json' : [x['country'] if 'country' in x.keys() else [] for x in names_out]})

# # dataout['name_pred1'] = dataout['fullname'].apply(lambda 

# # test1 = con.sql("SELECT * FROM (SELECT * FROM indiv_matched_with_rcids LIMIT 10) AS a LEFT JOIN (SELECT * FROM wrdsdb.company_mapping) AS b ON a.main_rcid = b.rcid")

#################
## FULL MATCHING 
#################
rev_indiv = con.read_parquet(f"{root}/data/int/rev_merge_masters_i129_mar31.parquet")
foia_indiv = indiv_matched_with_rcids

## STEP 0: CLEANING DATA
rev_indiv_for_merge_int = con.sql(
"""
SELECT * FROM 
(SELECT 
    ROW_NUMBER() OVER() AS unique_id,
    fullname,
    user_id,
    rcid,
    sex_predicted AS sex,
    a AS pos_id,
    state,
    metro_area,
    pos_startdate,
    pos_enddate,
    title_raw,
    degree_clean,
    degree,
    CASE WHEN field_raw IS NULL AND degree_clean IS NULL THEN TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(strip_accents(lower(degree_raw)), '\\s*\\(.*\\)\\s*', ' ', 'g'), '[^a-z0-9\\s]', ' ', 'g'), '\\s+', ' ', 'g')) ELSE TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(strip_accents(lower(field_raw)), '\\s*\\(.*\\)\\s*', ' ', 'g'), '[^a-z0-9\\s]', ' ', 'g'), '\\s+', ' ', 'g')) END AS field_clean,
    univ_raw_clean,
    univ_name_clean,
    education_number,
    CASE WHEN (MAX(CASE WHEN degree_clean = 'High School' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id)) = 1 
        THEN MAX(CASE WHEN degree_clean = 'High School' THEN SUBSTRING(ed_enddate, 1, 4)::INT - 18 ELSE NULL END) OVER(PARTITION BY user_id) 
        ELSE MIN(CASE WHEN degree_clean = 'Non-Degree' OR degree_clean = 'Master' OR degree_clean = 'Doctor' OR degree_clean = 'MBA' THEN NULL ELSE SUBSTRING(ed_startdate, 1, 4)::INT - 18 END) OVER(PARTITION BY user_id) 
        END AS est_yob
FROM (
    SELECT *,
        CASE WHEN lower(university_raw) ~ '.*(high\\s?school).*' OR (degree IS NULL AND university_raw ~ '.*(HS| High| HIGH| high|H\\.S\\.|S\\.?S\\.?C|H\\.?S\\.?C\\.?)$') THEN 'High School' 
            WHEN degree IS NULL AND (lower(degree_raw) ~ '.*(cert|course|semester|exchange|abroad|summer|internship|edx|cdl).*' OR lower(university_raw) ~ '.*(edx|course|semester|exchange|abroad|summer|internship|certificat).*') THEN 'Non-Degree'
            WHEN degree IS NULL AND ((lower(degree_raw) ~ '.*(undergrad).*') OR (degree_raw ~ '.*(B\\.?A\\.?|B\\.?S\\.?C\\.?E\\.?|B\\.?Sc\\.?|B\\.?A\\.?E\\.?|B\\.?Eng\\.?|A\\.?B\\.?|S\\.?B\\.?|B\\.?B\\.?M\\.?).*') OR degree_raw ~ '^B\\.?S\\.?.*' OR lower(field_raw) ~ '.*bachelor.*') THEN 'Bachelor'
            WHEN degree IS NULL AND (degree_raw ~ '.*(M\\.?S\\.?C\\.?E\\.?|M\\.?P\\.?A\\.?).*' OR lower(field_raw) ~ '.*master.*') THEN 'Master'
            WHEN degree IS NULL AND lower(field_raw) ~ '.*(associate).*' THEN 'Associate' 
            ELSE degree END AS degree_clean,
        TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(strip_accents(lower(university_raw)), '\\s*\\(.*\\)\\s*', ' ', 'g'), '[^a-z0-9\\s]', ' ', 'g'), '\\s+', ' ', 'g')) AS univ_raw_clean,
        TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(strip_accents(lower(university_name)), '\\s*\\(.*\\)\\s*', ' ', 'g'), '[^a-z0-9\\s]', ' ', 'g'), '\\s+', ' ', 'g')) AS univ_name_clean
    FROM rev_indiv
    )) WHERE degree_clean NOT IN ('Non-Degree', 'High School', 'Bachelor', 'Associate')
""")
con.sql("CREATE OR REPLACE TABLE rev_indiv_for_merge_out AS SELECT * FROM rev_indiv_for_merge_int")

foia_indiv_for_merge_int = con.sql(
"""
SELECT 
    bb_id AS unique_id,
    foia_id,
    rec_yr, rec_mo, rec_day,
    BEN_SEX AS sex,
    TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(strip_accents(lower(BEN_PFIELD_OF_STUDY)), '\\s*\\(.*\\)\\s*', ' ', 'g'), '[^a-z0-9\\s]', ' ', 'g'), '\\s+', ' ', 'g')) AS field_clean,
    BEN_COUNTRY_OF_BIRTH AS pob, 
    WORKSITE_CITY AS city,
    worksite_state_name AS state,
    ben_year_of_birth AS yob,
    CASE WHEN JOB_TITLE = 'NA' THEN NULL ELSE JOB_TITLE END AS title_raw,
    bbvfd, bbvfy, bbvfm,
    TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(strip_accents(lower(institution_txt)), '\\s*\\(.*\\)\\s*', ' ', 'g'), '[^a-z0-9\\s]', ' ', 'g'), '\\s+', ' ', 'g')) AS univ_raw_clean,
    lottery_year,
    main_rcid
FROM foia_indiv
""")
con.sql("CREATE OR REPLACE TABLE foia_indiv_for_merge_out AS SELECT * FROM foia_indiv_for_merge_int")

# # step one: tokenization (field and university names)
# emh.create_replace_table(con, "SELECT univ_raw_clean, ROW_NUMBER() OVER() AS unique_id FROM (SELECT univ_raw_clean FROM (SELECT unique_id, univ_raw_clean, 'rev' AS dataset FROM rev_indiv_for_merge_out UNION ALL SELECT unique_id, univ_raw_clean, 'foia' AS dataset FROM foia_indiv_for_merge_out) GROUP BY univ_raw_clean)", "univ_names_to_tokenize")

# emh.tokenize(con, "univ_names_to_tokenize", "univ_names_tokenized", "univ_raw", "\\s+", show = True)

# emh.create_replace_table(con, "SELECT field_clean, ROW_NUMBER() OVER() AS unique_id FROM (SELECT field_clean FROM (SELECT field_clean, 'rev' AS dataset FROM rev_indiv_for_merge_out UNION ALL SELECT field_clean, 'foia' AS dataset FROM foia_indiv_for_merge_out) GROUP BY field_clean)", "fields_to_tokenize")

# emh.tokenize(con, "fields_to_tokenize", "fields_tokenized", "field", "\\s+", show = True)

# need to collapse to user x educ level
rev_full_for_merge_with_tokens = con.sql(
"""
SELECT fullname, user_id, est_yob, rcid, sex AS sex_rev, pos_id, state AS state_rev, metro_area AS metro_area_rev, title_raw AS title_raw_rev, pos_startdate, pos_enddate, 
SUBSTRING(pos_startdate, 1, 4)::INT AS pos_startyr,
SUBSTRING(pos_enddate, 1, 4)::INT AS pos_endyr,
    education_number, 
    a.field_clean AS field_clean_rev, field_tokens AS field_tokens_rev, field_rarest_token AS field_rarest_token_rev, field_rarest_token2 AS field_rarest_token2_rev,
    degree_clean, 
    a.univ_raw_clean AS univ_raw_clean_rev, univ_tokens AS univ_tokens_rev, univ_rarest_token AS univ_rarest_token_rev, univ_rarest_token2 AS univ_rarest_token2_rev
FROM rev_indiv_for_merge_int AS a 
LEFT JOIN (
    SELECT rare_name_tokens_with_freq AS univ_tokens, rarest_token AS univ_rarest_token, second_rarest_token AS univ_rarest_token2, univ_raw_clean FROM univ_names_tokenized) AS b 
ON a.univ_raw_clean = b.univ_raw_clean 
LEFT JOIN (
    SELECT rare_name_tokens_with_freq AS field_tokens, rarest_token AS field_rarest_token, second_rarest_token AS field_rarest_token2, field_clean FROM fields_tokenized) AS c 
ON a.field_clean = c.field_clean 
""")

rev_indiv_for_merge_with_tokens = con.sql(
"""
SELECT *, ROW_NUMBER() OVER() AS unique_id FROM(
SELECT fullname, user_id, est_yob AS yob, rcid, sex,
    MIN(SUBSTRING(pos_startdate, 1, 4)::INT) AS min_start_yr, 
    MAX(CASE WHEN pos_enddate IS NULL THEN 2025 ELSE SUBSTRING(pos_enddate, 1, 4)::INT END) AS max_end_yr, 
    education_number, 
    a.field_clean, field_tokens, field_rarest_token, field_rarest_token2,
    degree_clean, 
    a.univ_raw_clean, univ_tokens, univ_rarest_token, univ_rarest_token2 
FROM rev_indiv_for_merge_int AS a 
LEFT JOIN (
    SELECT rare_name_tokens_with_freq AS univ_tokens, rarest_token AS univ_rarest_token, second_rarest_token AS univ_rarest_token2, univ_raw_clean FROM univ_names_tokenized) AS b 
ON a.univ_raw_clean = b.univ_raw_clean 
LEFT JOIN (
    SELECT rare_name_tokens_with_freq AS field_tokens, rarest_token AS field_rarest_token, second_rarest_token AS field_rarest_token2, field_clean FROM fields_tokenized) AS c 
ON a.field_clean = c.field_clean 
GROUP BY fullname, user_id, est_yob, rcid, sex, education_number, a.field_clean, field_tokens, field_rarest_token, field_rarest_token2, degree_clean, a.univ_raw_clean, univ_tokens, univ_rarest_token, univ_rarest_token2)
""")
rev_indiv_splink = con.sql("SELECT unique_id, yob, rcid, sex, min_start_yr, max_end_yr, field_tokens, field_rarest_token, field_rarest_token2, field_clean, univ_raw_clean, univ_tokens, univ_rarest_token, univ_rarest_token2 FROM rev_indiv_for_merge_with_tokens")


foia_full_for_merge_with_tokens = con.sql(
"""SELECT 
    state AS state_foia, city AS city_foia, pob, title_raw AS title_raw_foia, main_rcid AS rcid, sex AS sex_foia, unique_id, yob,
    bbvfy::INT AS start_yr,
    a.field_clean AS field_clean_foia, field_tokens AS field_tokens_foia, field_rarest_token AS field_rarest_token_foia, field_rarest_token2 AS field_rarest_token2_foia,
    a.univ_raw_clean AS univ_raw_clean_foia, univ_tokens AS univ_tokens_foia, univ_rarest_token AS univ_rarest_token_foia, univ_rarest_token2 AS univ_rarest_token2_foia
FROM foia_indiv_for_merge_int AS a 
LEFT JOIN (
    SELECT rare_name_tokens_with_freq AS univ_tokens, rarest_token AS univ_rarest_token, second_rarest_token AS univ_rarest_token2, univ_raw_clean  FROM univ_names_tokenized) AS b 
ON a.univ_raw_clean = b.univ_raw_clean
LEFT JOIN (
    SELECT rare_name_tokens_with_freq AS field_tokens, rarest_token AS field_rarest_token, second_rarest_token AS field_rarest_token2, field_clean FROM fields_tokenized) AS c
ON a.field_clean = c.field_clean
""")
foia_indiv_for_merge_with_tokens = con.sql(
"""SELECT 
    state, city, pob, title_raw, main_rcid AS rcid, sex, unique_id, yob, bbvfy::INT AS min_start_yr, bbvfy::INT AS max_end_yr,
    a.field_clean, field_tokens, field_rarest_token, field_rarest_token2,
    a.univ_raw_clean, univ_tokens, univ_rarest_token, univ_rarest_token2 
FROM foia_indiv_for_merge_int AS a 
LEFT JOIN (
    SELECT rare_name_tokens_with_freq AS univ_tokens, rarest_token AS univ_rarest_token, second_rarest_token AS univ_rarest_token2, univ_raw_clean  FROM univ_names_tokenized) AS b 
ON a.univ_raw_clean = b.univ_raw_clean
LEFT JOIN (
    SELECT rare_name_tokens_with_freq AS field_tokens, rarest_token AS field_rarest_token, second_rarest_token AS field_rarest_token2, field_clean FROM fields_tokenized) AS c
ON a.field_clean = c.field_clean
""")
foia_indiv_splink = con.sql("SELECT unique_id, yob, rcid, sex, min_start_yr, max_end_yr, field_tokens, field_rarest_token, field_rarest_token2, field_clean, univ_raw_clean, univ_tokens, univ_rarest_token, univ_rarest_token2 FROM foia_indiv_for_merge_with_tokens")

# step two: matching
initial_indiv_merge = con.sql(
"""
SELECT * FROM (
    foia_full_for_merge_with_tokens AS a 
    LEFT JOIN
    rev_full_for_merge_with_tokens AS b
    ON a.rcid = b.rcid AND (a.univ_rarest_token_foia = b.univ_rarest_token_rev OR POSITION(a.univ_raw_clean_foia IN b.univ_raw_clean_rev) > 0) AND a.start_yr between b.pos_startyr AND b.pos_endyr
)
""")

merged_df = initial_indiv_merge.df()
merged_df['n'] = merged_df.groupby('unique_id')['user_id'].transform('nunique')
merged_df['est_yob_diff'] = merged_df.apply(lambda x: -1 if pd.isnull(x['est_yob']) else np.abs(int(x['est_yob'])-int(x['yob'])), axis = 1)
merged_df['fieldmatch'] = merged_df.apply(lambda x: -1 if pd.isnull(x['field_clean_foia']) or pd.isnull(x['field_clean_rev']) else x['field_clean_foia'] in x['field_clean_rev'], axis = 1)

match_df = merged_df.loc[(merged_df['user_id'].isnull() == 0)&(merged_df['est_yob_diff'] <= 4) & (merged_df['sex_rev'] == merged_df['sex_foia'])] 
match_df.shape
match_df.loc[match_df['state_foia']==match_df['state_rev']].shape
match_df_exact = match_df.loc[(match_df['state_foia']==match_df['state_rev'])&(match_df['fieldmatch']==1)]
match_df_exact['n2'] = match_df_exact.groupby('unique_id')['user_id'].transform('nunique')
match_df_exact.sample(100).sort_values('unique_id')[['n2','unique_id','user_id','univ_raw_clean_rev','univ_raw_clean_foia','fullname','pob','state_rev','state_foia','title_raw_rev','title_raw_foia','sex_rev', 'sex_foia', 'est_yob', 'yob', 'field_clean_foia','field_clean_rev']]

# y=x.merge(con.sql("SELECT * FROM foia_indiv_for_merge_int").df(), on = 'unique_id')

# y.sort_values('unique_id')[['unique_id','univ_raw_clean_1','univ_raw_clean_y','fullname','pob','metro_area','city','state_x','state_y','title_raw_x','title_raw_y','field_clean_1','field_clean_y','est_yob','yob_x']]

# z = y.loc[y['pos_id'].isnull() == 0]
# z['n'] = z.groupby('unique_id')['user_id'].transform('nunique')
# z.sort_values('unique_id')[['n','unique_id','univ_raw_clean_1','univ_raw_clean_y','fullname','pob','metro_area','city','state_x','state_y','title_raw_x','title_raw_y','field_clean_1','field_clean_y','est_yob','yob_x']]

# step three: checking matches

# step four: post-processing
## est yob
## country of birth x university 
## worksite location
## position overlaps with h1b start date
## job title?


# # splink???
# from splink import DuckDBAPI, Linker, SettingsCreator, block_on
# from splink.blocking_analysis import count_comparisons_from_blocking_rule
# from splink.blocking_rule_library import CustomRule
# import splink.comparison_library as cl
# import splink.comparison_level_library as cll
# from splink.comparison_library import CustomComparison

# db_api = DuckDBAPI(con)

# ## comparisons
# def calculate_tf_product_array_sql(token_rel_freq_array_name):

#     return f"""
#     list_intersect({token_rel_freq_array_name}_l, {token_rel_freq_array_name}_r)
#         .list_transform(x -> x.freq::float)
#         .list_concat([1.0::FLOAT]) -- in case there are no matches
#         .list_reduce((p, q) -> p * q)
#     """

# univ_comp = {
# "output_column_name": "univ_tokens",
# "comparison_levels": [
#     {
#         "sql_condition": '"univ_tokens_l" IS NULL OR "univ_tokens_r" IS NULL',
#         "label_for_charts": "univ_tokens is NULL",
#         "is_null_level": True,
#     },
#     {
#         "sql_condition": f"""
#         {calculate_tf_product_array_sql('univ_tokens')} < 1e-16
#         """,
#         "label_for_charts": "Array product is less than 1e-16",
#     },
#     {
#         "sql_condition": f"""
#         {calculate_tf_product_array_sql('univ_tokens')} < 1e-12
#         """,
#         "label_for_charts": "Array product is less than 1e-12",
#     },
#     {
#         "sql_condition": f"""
#         {calculate_tf_product_array_sql('univ_tokens')} < 1e-8
#         """,
#         "label_for_charts": "Array product is less than 1e-8",
#     },
#     {
#         "sql_condition": f"""
#         {calculate_tf_product_array_sql('univ_tokens')} < 1e-4
#         """,
#         "label_for_charts": "Array product is less than 1e-4",
#     },
#     {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
#     ],
# "comparison_description": "Comparison of levels of product of frequencies of exactly matched univ name tokens",
# }


# field_comp = {
# "output_column_name": "field_tokens",
# "comparison_levels": [
#     {
#         "sql_condition": '"field_tokens_l" IS NULL OR "field_tokens_r" IS NULL',
#         "label_for_charts": "field_tokens is NULL",
#         "is_null_level": True,
#     },
#     {
#         "sql_condition": f"""
#         {calculate_tf_product_array_sql('field_tokens')} < 1e-16
#         """,
#         "label_for_charts": "Array product is less than 1e-16",
#     },
#     {
#         "sql_condition": f"""
#         {calculate_tf_product_array_sql('field_tokens')} < 1e-12
#         """,
#         "label_for_charts": "Array product is less than 1e-12",
#     },
#     {
#         "sql_condition": f"""
#         {calculate_tf_product_array_sql('field_tokens')} < 1e-8
#         """,
#         "label_for_charts": "Array product is less than 1e-8",
#     },
#     {
#         "sql_condition": f"""
#         {calculate_tf_product_array_sql('field_tokens')} < 1e-4
#         """,
#         "label_for_charts": "Array product is less than 1e-4",
#     },
#     {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
#     ],
# "comparison_description": "Comparison of levels of product of frequencies of exactly matched field name tokens",
# }

# yob_sim = {
#     "output_column_name": "yob",
#     "comparison_levels": [
#     {
#         "sql_condition": '"yob_l" IS NULL OR "yob_r" IS NULL',
#         "label_for_charts": "yob is NULL",
#         "is_null_level": True,
#     },
#     {
#         "sql_condition": 'yob_l = yob_r',
#         "label_for_charts": 'yob same',
#     },
#     {
#         "sql_condition": 'abs(yob_l::INT - yob_r::INT) < 3',
#         "label_for_charts": "yob diff less than 3",
#     },
#     {
#         "sql_condition": 'abs(yob_l::INT - yob_r::INT) < 6',
#          "label_for_charts": "yob diff less than 6",
#     },
#     {
#         "sql_condition": 'abs(yob_l::INT - yob_r::INT) < 10',
#          "label_for_charts": "yob diff less than 10",
#     },
#     {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
#     ],
# "comparison_description": "Comparison of levels of yob"
# }

# position_overlap = {
#     "output_column_name": "pos_overlap",
#     "comparison_levels": [
#     {
#         "sql_condition": 'min_start_yr_l IS NULL OR min_start_yr_r IS NULL OR max_end_yr_l IS NULL OR max_end_yr_r IS NULL',
#         "label_for_charts": "start/end yr is NULL",
#         "is_null_level": True,
#     },
#     {
#         "sql_condition": '(min_start_yr_l::INT <= min_start_yr_r::INT AND max_end_yr_l::INT >= min_start_yr_r::INT) OR (min_start_yr_r::INT <= min_start_yr_l::INT AND max_end_yr_r::INT >= min_start_yr_l::INT) OR (max_end_yr_l::INT >= max_end_yr_r::INT AND min_start_yr_l::INT <= max_end_yr_r::INT) OR (max_end_yr_r::INT >= max_end_yr_l::INT AND min_start_yr_r::INT <= max_end_yr_l::INT)',
#         "label_for_charts": 'overlap in position',
#     },
#     {
#         "sql_condition": "ELSE", "label_for_charts": "All other comparisons"
#     },
#     ],
#     "comparison_description": "Comparison of levels of position overlap"
# }

# ## initializing splink model
# settings = SettingsCreator(
#     link_type = 'link_only',
#     blocking_rules_to_generate_predictions = [block_on('rcid', 'univ_rarest_token')],
#     comparisons = [cl.ExactMatch('sex'),
#                    univ_comp, field_comp, yob_sim, position_overlap]
# )

# linker = Linker([rev_indiv_splink, foia_indiv_splink], settings, db_api)

# ## training model
# # estimating lambdas
# linker.training.estimate_probability_two_random_records_match([block_on("univ_raw_clean", "field_clean", "rcid", "sex")], recall = 0.1)

# # estimating u's
# linker.training.estimate_u_using_random_sampling()
# linker.visualisations.match_weights_chart()

# # # estimating m
# # ## will skip sex, name?
# # linker.training.estimate_parameters_using_expectation_maximization(block_on("rcid", "univ_tokens"))

# # ## will skip field?
# # linker.training.estimate_parameters_using_expectation_maximization(block_on("rcid", "field_tokens"))

# linker.visualisations.m_u_parameters_chart()
# linker.evaluation.unlinkables_chart()

# df_predictions_all = linker.inference.predict()

# sql = f"WITH ranked_matches AS (SELECT *, ROW_NUMBER() OVER(PARTITION BY unique_id_r ORDER BY match_weight DESC) AS rank FROM {df_predictions_all.physical_name}), best_match as (SELECT * FROM ranked_matches where rank = 1 ORDER BY match_weight desc), matched_bb_ids AS (SELECT DISTINCT unique_id_r FROM best_match) SELECT * FROM best_match ORDER BY match_probability DESC"
# ranked_matches = con.sql(sql)
# match_eval = con.sql("SELECT * FROM (ranked_matches AS a LEFT JOIN foia_indiv_for_merge_with_tokens AS b ON a.unique_id_r = b.unique_id LEFT JOIN rev_indiv_for_merge_with_tokens AS c ON a.unique_id_l = c.unique_id)")

# x= match_eval.df()
# x[['fullname','match_probability','gamma_sex','yob_l','yob_r','min_start_yr_l', 'max_end_yr_l','field_clean','field_clean_1','univ_raw_clean','univ_raw_clean_1']]
