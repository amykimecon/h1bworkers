# File Description: Exploring Data for MFin Programs
# Author: Amy Kim
# Date Created: Sep 15 2025

# Imports and Paths
import duckdb as ddb
import pandas as pd
import numpy as np
import sys
import os
import wrds
import seaborn as sns 

sys.path.append(os.path.dirname(__file__))
from config import * 


db = wrds.Connection(username = 'amykimecon')

con = ddb.connect()

# filtering MFin students
mfin1 = db.raw_sql("SELECT a.user_id, fullname, university_raw, rsid, a.education_number, startdate, enddate, degree, degree_raw, field, field_raw, university_country, user_location, profile_linkedin_url FROM revelio.individual_user_education AS a LEFT JOIN revelio.individual_user_education_raw AS b ON a.user_id = b.user_id AND a.education_number = b.education_number LEFT JOIN revelio.individual_user AS c ON a.user_id = c.user_id WHERE degree = 'Master' AND field = 'Finance'")

mba = db.raw_sql("SELECT COUNT(*) AS n_mba, gradyr FROM (SELECT SUBSTRING(enddate::VARCHAR, 1, 4)::INT AS gradyr FROM revelio.individual_user_education WHERE degree = 'MBA' AND university_country = 'United States') GROUP BY gradyr")

mfin = db.raw_sql("SELECT COUNT(*) AS n_mfin, gradyr FROM (SELECT SUBSTRING(enddate::VARCHAR, 1, 4)::INT AS gradyr FROM revelio.individual_user_education WHERE degree = 'Master' AND field = 'Finance' AND university_country = 'United States') GROUP BY gradyr")

all = db.raw_sql("SELECT COUNT(*) AS n_educ, gradyr FROM (SELECT SUBSTRING(enddate::VARCHAR, 1, 4)::INT AS gradyr FROM revelio.individual_user_education WHERE university_country = 'United States') GROUP BY gradyr")

all_pos = db.raw_sql("SELECT COUNT(*) AS n_pos, gradyr FROM (SELECT SUBSTRING(startdate::VARCHAR, 1, 4)::INT AS gradyr FROM revelio.individual_positions WHERE country = 'United States') GROUP BY gradyr")

mba_mfin = pd.merge(pd.merge(pd.merge(mba, mfin, how = 'inner', on = 'gradyr'), all, how = 'inner', on = 'gradyr'), all_pos, how = 'inner', on = 'gradyr')
mba_mfin['n_mba_norm'] = mba_mfin['n_mba']/56517
mba_mfin['n_mfin_norm'] = mba_mfin['n_mfin']/1124
mba_mfin['n_educ_norm'] = mba_mfin['n_educ']/1491278
mba_mfin['n_pos_norm'] = mba_mfin['n_pos']/2988739

deg_by_year = pd.melt(mba_mfin.loc[(mba_mfin['gradyr'] >= 1980)&(mba_mfin['gradyr']<=2025)], id_vars = 'gradyr', var_name = 'var', value_vars = ['n_mba_norm', 'n_mfin_norm','n_educ_norm','n_pos_norm'])

sns.scatterplot(deg_by_year, x='gradyr', y = 'value',hue = 'var')

mfin1 = db.raw_sql("SELECT a.user_id, fullname, university_raw, rsid, a.education_number, startdate, enddate, degree, degree_raw, field, field_raw, university_country, user_location, profile_linkedin_url FROM revelio.individual_user_education AS a LEFT JOIN revelio.individual_user_education_raw AS b ON a.user_id = b.user_id AND a.education_number = b.education_number LEFT JOIN revelio.individual_user AS c ON a.user_id = c.user_id WHERE degree IS NULL AND (degree_raw ~ '.*(master|mfin|msf).*)")

# PRINCETON CASE STUDY
x = db.raw_sql("SELECT a.user_id, fullname, university_raw, rsid, a.education_number, startdate, enddate, degree, degree_raw, field, field_raw, university_country, user_location, profile_linkedin_url, university_name FROM revelio.individual_user_education AS a LEFT JOIN revelio.individual_user_education_raw AS b ON a.user_id = b.user_id AND a.education_number = b.education_number LEFT JOIN revelio.individual_user AS c ON a.user_id = c.user_id WHERE rsid = 104934 OR rsid = 150503 OR rsid = 155977")
# ucla: 150503
# usc: 155977
# princeton: 104934

y = x.loc[((x['degree_raw'].str.lower().str.match('.*(mfin|financ|msf).*'))|((x['degree'] == 'Master') & (x['field_raw'].str.lower().str.match('.*financ.*'))))&((pd.isnull(x['degree']))|(x['degree'] != 'Bachelor'))]

y['gradyr'] = pd.to_numeric(y['enddate'].str[:4])

sns.scatterplot(y.loc[(y['gradyr'] > 1990) & (y['gradyr'] < 2025)].groupby(['gradyr','university_name'])['university_raw'].agg('count').reset_index(), x = 'gradyr', y = 'university_raw', hue = 'university_name')

# class of 2026
# PRAGNYA AKELLA x
# FOUAD ALLAHWERDI x
# PEDRO ALBIN LAZCANO x
# JAMYANG-DORJÉ BHUTIA NOT FOUND -- (on linkedin but not yet scraped?)
# GAUTAM CHAWLA x
# YIJING (KARRY) CHEN x
# MINHA (ALYSSA) CHOI NOT FOUND -- (on linkedin but not yet scraped?)
# ROMAIN DESBIOLLES x
# XIAOFU DING x
# MARVIN ERTL x
# WENDI FAN x
# ASHISH GUPTA x
# IAN GURLAND x
# JIAHAN (PETER) JIANG x
# LING JIN x (duplicate?)
# TAKAHIRO KOBAYASHI NOT FOUND -- ???
# PARTH SATISH LATURIA x
# YINGYAO LIU x
# YANG OU x
# RAJ GAURANGBHAI PATEL x
# FILIPPO RONZINO NOT FOUND -- (on linkedin but not yet scraped?)
# FAIZ SHOAIB NOT FOUND -- (on linkedin but not yet scraped?)
# YING (ANDREA) SUN x
# YUQIAN TIAN x
# MARK CHRISTOPHER UY x
# YANZHE WANG NOT FOUND --
# YIFEI WANG x
# YINCHEN (ALEX) WU NOT FOUND --
# ZHE YAN x
# YUCHEN (YOLANDA) YANG x
# YUXIAO (CERINA) YAO NOT FOUND --
# MINGYI (FRANK) YU x
# YUCHEN YU x
# YICHI ZHANG x
# ZHONG (KEVIN) ZHANG x
# JINGLE (VIOLA) ZHOU x
# (15/36 not found)

# class of 2023
# Alexander Aronovich x
# Poorva Arora x
# Laurent Benayoun x
# Luis Felipe Bento x
# Jai Krishna Chaparala x
# Yuyang (Eric) Chen x
# Haoting (David) Dai x
# Aditya Shivaji Divekar x
# Andrew Elzayn x
# Qingyang (Young) Gao x
# Arjun Goyal x 
# Taku Hasegawa x
# Lina Huang x
# Zixun (Connie) Huang x
# Gregoire Lachaise x
# Sixtine Le Roux x
# Jingyi (Joy) Li x
# Mengyuan (Molly) Li NOT FOUND --
# Xuechen Li x
# Zhuomin (Judy) Mao x
# Dimitriades Mathieu-Savvas x
# Sneha Mohan x
# Sinan Ozbay x
# Stephane Poznanski x
# Ryan Slattery x
# Jing Wen x
# Die (Catherine) Wu x
# Dorothy Zhang x
# Erhao Zhao x
# Hange (Cathy) Zhu x

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
rev_raw = con.read_parquet(f"{root}/data/int/wrds_users_sep2.parquet") #con.read_parquet(f"{wrds_out}/rev_user_merge0.parquet")

# for j in range(1,10):
#     rev_raw = con.sql(f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{wrds_out}/rev_user_merge{j}.parquet'")

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
