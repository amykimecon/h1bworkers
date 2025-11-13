# File Description: Creating crosswalk from Revelio rsids/university names (in education data) to CBSA/MSA names (in positions data)

# Imports and Paths
import pandas as pd
import json
import sys
import os
import wrds
import duckdb as ddb
import requests
import re
from rapidfuzz import distance

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

# DuckDB
con = ddb.connect()

## WRDS
db = wrds.Connection(wrds_username='amykimecon')

# fromscratch
fromscratch = False

#####################
# IMPORTING DATA
#####################
# WRDS query: list of MSAs from position data
# check if crosswalk exists, if not, create it
if os.path.exists(f"{root}/data/int/msa_list.parquet") and not fromscratch:
    print("MSA list already exists! Reading in...")
    msa_list = pd.read_parquet(f"{root}/data/int/msa_list.parquet")
else:
    msa_list = db.raw_sql(
    """
    SELECT msa, state, COUNT(*) AS n
    FROM (SELECT msa, state FROM revelio.individual_positions WHERE msa IS NOT NULL AND country = 'United States' LIMIT 10000000) GROUP BY msa, state
    """)

    # saving MSA list
    msa_list.to_parquet(f"{root}/data/int/msa_list.parquet", index = False)

# filtering: at least 10 MSAs
msa_list = msa_list[msa_list['n'] >= 10]

# WRDS query: crosswalk of all university names to rsids
# check if crosswalk exists, if not, create it
rev_raw = con.read_parquet(f"{root}/data/int/wrds_users_sep2.parquet") 
univ_rsid_cw = con.sql("SELECT rsid, university_raw, COUNT(*) AS n FROM rev_raw WHERE university_raw IS NOT NULL AND rsid IS NOT NULL AND university_country = 'United States' GROUP BY rsid, university_raw").df()
univ_rsid_cw = univ_rsid_cw[univ_rsid_cw['n'] >= 10]

# reading in FOIA F1 data and creating university name to ZIP crosswalk
# check if crosswalk exists, if not, create it
if os.path.exists(f"{root}/data/int/univ_zip_cw.parquet"):
    print("University Name to ZIP crosswalk already exists! Reading in...")
    univ_zip_cw = pd.read_parquet(f"{root}/data/int/univ_zip_cw.parquet")
else:
    f1_foia_raw = con.read_parquet(f"{root}/data/int/foia_sevp_combined_raw.parquet")
    univ_zip_cw = con.sql("SELECT school_name, campus_zip_code, campus_state,campus_city, MIN(year) AS min_year, MAX(year) AS max_year FROM f1_foia_raw GROUP BY school_name, campus_zip_code, campus_state, campus_city").df()
    # saving crosswalk
    univ_zip_cw.to_parquet(f"{root}/data/int/univ_zip_cw.parquet", index = False)

# reading in ZIP to CBSA crosswalk from HUD API from various years
hud_url = "https://www.huduser.gov/hudapi/public/usps?type=3&query=All&year=2015"
hud_headers = {"Authorization": "Bearer {0}".format(HUD_API_KEY)}
hud_response = requests.get(hud_url, headers = hud_headers)
zip_cbsa_cw = pd.DataFrame(hud_response.json()['data']['results'])
zip_cbsa_cw = zip_cbsa_cw[pd.isna(zip_cbsa_cw['geoid']) == False]

# reading in CBSA to MSA crosswalk
cbsa_msa_cw = pd.concat([pd.read_excel(f"{root}/data/raw/cbsa_metro_cw_jul15.xls", header = 2), pd.read_excel(f"{root}/data/raw/cbsa_metro_cw_jul23.xlsx", header = 2)])
cbsa_msa_cw = cbsa_msa_cw[pd.isnull(cbsa_msa_cw['CBSA Title']) == False].drop_duplicates()

# reading in IPEDS ID to name crosswalk
ipeds_univ_cw_raw = pd.read_excel(f"{root}/data/raw/ipeds_name_cw_2021.xlsx", sheet_name='Crosswalk', usecols = ['OPEID', 'IPEDSMatch', 'PEPSSchname', 'PEPSLocname', 'IPEDSInstnm', 'OPEIDMain', 'IPEDSMain'])
ipeds_univ_cw_raw['UNITID'] = ipeds_univ_cw_raw['IPEDSMatch'].str.replace('No match', '-1').astype(int)

ipeds_zip_cw_raw = pd.read_csv(f"{root}/data/raw/ipeds_cw_2021.csv", usecols = ['UNITID', 'OPEID', 'INSTNM', 'CITY', 'STABBR', 'ZIP', 'ALIAS'])

ipeds_cw = (
    ipeds_univ_cw_raw[ipeds_univ_cw_raw['UNITID']!=-1]
    .merge(ipeds_zip_cw_raw, on = ['OPEID','UNITID'], how = 'left')
    .melt(id_vars=['UNITID', 'CITY', 'STABBR', 'ZIP'], value_vars = ['PEPSSchname', 'PEPSLocname', 'IPEDSInstnm', 'INSTNM', 'ALIAS'], var_name = 'source', value_name = 'instname')
    .dropna(subset=['instname'])          # drop rows with missing names
    .drop_duplicates(subset=['UNITID', 'instname'])  # ensure uniqueness
    .reset_index(drop=True)
)

ipeds_cw['ZIP'] = (
    ipeds_cw.groupby('UNITID')['ZIP']
    .transform(lambda x: x.ffill().bfill())
    .str.replace(r'-[0-9]+$','',regex=True)
)

ipeds_cw['CITY'] = (
    ipeds_cw.groupby('UNITID')['CITY']
    .transform(lambda x: x.ffill().bfill())
)

#####################
# CLEANING DATA
#####################
# university names: cleaning by removing characters, lowercasing
univ_rsid_cw['university_raw'] = univ_rsid_cw['university_raw'].str.replace(r'[^a-zA-Z0-9 ]', ' ', regex=True).str.lower()
univ_zip_cw['school_name'] = univ_zip_cw['school_name'].str.replace(r'[^a-zA-Z0-9 ]', ' ', regex=True).str.lower()
ipeds_cw['instname'] = ipeds_cw['instname'].str.replace(r'[^a-zA-Z0-9 ]', ' ', regex=True).str.lower()

# indicator for msa
msa_list['metroarea'] = msa_list.apply(lambda row: True if "MSA" in row['msa'] else False, axis = 1)

# state/metro names: lowercasing and cleaning
univ_zip_cw['campus_state'] = univ_zip_cw['campus_state'].str.lower()
univ_zip_cw['campus_city'] = univ_zip_cw['campus_city'].str.lower()
msa_list['state'] = msa_list['state'].str.lower()
cbsa_msa_cw['State Name'] = cbsa_msa_cw['State Name'].str.lower()
msa_list['msa_clean'] = msa_list.apply(lambda row: re.sub(r'(MSA|msa|,)', '', row['msa']).strip().lower() if row['metroarea'] == True else None, axis = 1)
cbsa_msa_cw['CBSA Title'] = cbsa_msa_cw.apply(lambda row: re.sub(r'(MSA|msa|,)', '', row['CBSA Title']).strip().lower(), axis = 1)
ipeds_cw['CITY'] = ipeds_cw['CITY'].str.lower()

# zip codes as int
univ_zip_cw['campus_zip_code'] = univ_zip_cw['campus_zip_code'].astype(int)
zip_cbsa_cw['zip'] = zip_cbsa_cw['zip'].astype(int)

# renaming 
zip_cbsa_cw['cbsa'] = zip_cbsa_cw['geoid'].astype(int)
cbsa_msa_cw['metroarea'] = cbsa_msa_cw['Metropolitan/Micropolitan Statistical Area'] == "Metropolitan Statistical Area"
cbsa_msa_cw = cbsa_msa_cw.rename(columns = {'CBSA Code':'cbsa', 'CBSA Title':'cbsa_name', 'Metropolitan Division Code': 'met_code', 'State Name':'state'})

# create new column in cbsa_msa_cw equal to met_code if nonmissing, otherwise equal to cbsa
cbsa_msa_cw['cbsa_code'] = cbsa_msa_cw.apply(lambda row: row['met_code'] if not pd.isna(row['met_code']) else row['cbsa'], axis = 1)

# grouping by cbsa code
cbsa_msa_cw_bycbsa = cbsa_msa_cw[['cbsa', 'cbsa_code','cbsa_name','metroarea','state']].drop_duplicates()
cbsa_msa_cw_bycbsa['cbsa_code'] = cbsa_msa_cw_bycbsa['cbsa_code'].astype(int)
cbsa_msa_cw_bycbsa['cbsa'] = cbsa_msa_cw_bycbsa['cbsa'].astype(int)

######################################
# MERGING DATA: POSITIONS MSA TO CBSA
######################################
# split msa_clean into cbsa and state, then further split into list on hyphens
msa_list['msa_name'] = msa_list['msa_clean'].str.replace(r'([a-z]{2}-?)+$','',regex=True).str.strip().str.split('-')
msa_list['msa_state'] = msa_list['state']

cbsa_pos = cbsa_msa_cw_bycbsa[['cbsa','cbsa_name','state']].drop_duplicates()
cbsa_pos['msa_name_cbsa'] = cbsa_pos['cbsa_name'].str.replace(r'([a-z]{2}-?)+$','',regex=True).str.strip().str.split('-')

# initial merge
pos_merge_initial = msa_list[msa_list['metroarea']==True][['msa_clean', 'msa_name','msa_state']].merge(cbsa_pos, left_on = 'msa_clean', right_on = 'cbsa_name', how = 'left')

# for rows that did not merge, try merging on state and checking if msa name and cbsa name have any overlap
pos_merge_unmerged = pos_merge_initial[pd.isna(pos_merge_initial['cbsa_name'])]
pos_merge_merged = pos_merge_initial[pd.isna(pos_merge_initial['cbsa_name']) == False]

pos_second_merge = pos_merge_unmerged[['msa_clean','msa_name','msa_state']].copy().merge(cbsa_pos, left_on = 'msa_state', right_on = 'state', how = 'left', suffixes = ('_x','_y'))
pos_second_merge['msa_overlap'] = pos_second_merge.apply(lambda row: len(set(row['msa_name']).intersection(row['msa_name_cbsa'])), axis = 1)
pos_second_merge['max_msa_overlap'] = pos_second_merge.groupby('msa_clean')['msa_overlap'].transform('max')
pos_second_merge_match = pos_second_merge[(pos_second_merge['msa_overlap'] == pos_second_merge['max_msa_overlap']) & (pos_second_merge['msa_overlap'] > 0)].groupby(['msa_clean', 'cbsa']).size().reset_index().drop(columns = 0)

# manually add new CBSA codes for missing MSAs
manual_pos_merge = pd.DataFrame({
    'msa_clean': ['corvalis or', 'honolulu hi'],
    'cbsa': [18700, 46520]
})
pos_merge_final = pd.concat([pos_merge_merged.groupby(['msa_clean', 'cbsa']).size().reset_index().drop(columns = 0), pos_second_merge_match, manual_pos_merge], ignore_index = True).merge(msa_list[['msa','msa_clean']], on = 'msa_clean', how = 'left').drop_duplicates()

# merge back to msa_list to check coverage
msa_list_check = msa_list[msa_list['metroarea']==True][['msa_clean']].merge(pos_merge_final, left_on = 'msa_clean', right_on = 'msa_clean', how = 'left')
print("MSAs without CBSA match:")
print(msa_list_check[pd.isna(msa_list_check['cbsa'])])

# save
pos_merge_final.to_parquet(f"{root}/data/int/positions_msa_cw.parquet", index = False)

######################################
# MERGING DATA: RSID TO CBSA
######################################
# combine univ_zip_cw (FOIA) with ipeds_cw (IPEDS) to get master list of names x zip codes
ipeds_cw_formerge = ipeds_cw.rename(columns={'instname': 'school_name', 'ZIP': 'campus_zip_code'})[['school_name','campus_zip_code']].dropna()
ipeds_cw_formerge['campus_zip_code'] = ipeds_cw_formerge['campus_zip_code'].astype(int)
univ_zip_all = pd.concat([ipeds_cw_formerge, univ_zip_cw[['school_name','campus_zip_code']]]).dropna().drop_duplicates()

# merge univ_zip_cw with zip_cbsa_cw and cbsa_msa_cw via CBSA/ZIP code
univ_cbsa_cw_raw = univ_zip_all.merge(zip_cbsa_cw[['zip','cbsa']], left_on = 'campus_zip_code', right_on = 'zip', how = 'left').merge(cbsa_msa_cw_bycbsa, left_on = 'cbsa', right_on = 'cbsa_code', how = 'left')

univ_cbsa_cw = univ_cbsa_cw_raw.groupby(['school_name', 'cbsa_y', 'cbsa_name']).size().reset_index().rename(columns = {'cbsa_y':'cbsa', 0:'n'}) 

# merging rsid with zip codes via university names, first exactly
univ_rsid_cw['hs_cc'] = univ_rsid_cw['university_raw'].str.contains('high school|highschool|community college', regex=True)
univ_rsid_cw_nohs_cc = univ_rsid_cw[
    ~univ_rsid_cw.groupby("rsid")["hs_cc"].transform("any")
]
univ_rsid_zip_exact = univ_rsid_cw_nohs_cc.merge(univ_cbsa_cw, left_on = 'university_raw', right_on = 'school_name', how = 'left')
univ_rsid_zip_exact_matched = univ_rsid_zip_exact[pd.isna(univ_rsid_zip_exact['school_name']) == False]
print(f"Number of RSIDs matched exactly to CBSA via university name: {univ_rsid_zip_exact_matched['rsid'].nunique()} out of {univ_rsid_cw_nohs_cc['rsid'].nunique()} total RSIDs")

# if rsid not matched exactly, do fuzzy matching
unmatched_rsids = univ_rsid_cw_nohs_cc[univ_rsid_cw_nohs_cc['rsid'].isin(univ_rsid_zip_exact_matched['rsid']) == False].copy()
unmatched_rsids['alt_univ_name'] = unmatched_rsids['university_raw'].str.replace(r'\b(and|or|of|college|university|institute|state|school|for|the|academy|technology|tech|center|graduate|at|prep|preparatory)\b', '', regex = True).str.strip()
univ_cbsa_cw['alt_school_name'] = univ_cbsa_cw['school_name'].str.replace(r'\b(and|or|of|college|university|institute|state|school|for|the|academy|technology|tech|center|graduate|at|prep|preparatory)\b', '', regex = True).str.strip()

univ_rsid_zip_cw_fuzz = help.fuzzy_join_lev_jw(unmatched_rsids[['alt_univ_name','university_raw','rsid']], univ_cbsa_cw[['cbsa','alt_school_name','school_name','cbsa_name']], left_on = 'alt_univ_name', right_on = 'alt_school_name', top_n = 3, threshold = 0.8)
univ_rsid_zip_cw_fuzz['lev_sim'] = univ_rsid_zip_cw_fuzz.apply(lambda x: distance.Levenshtein.normalized_similarity(x['university_raw_left'],x['school_name_right']), axis = 1)

# count number of unique cbsas per rsid
univ_rsid_zip_cw_fuzz['num_cbsas'] = univ_rsid_zip_cw_fuzz.groupby('rsid_left')['cbsa_right'].transform('nunique')

# mark as correct if num_cbsas is one or lev_sim is over 80%
univ_rsid_zip_cw_fuzz['matched'] = univ_rsid_zip_cw_fuzz.apply(lambda x: True if x['num_cbsas'] == 1 or x['lev_sim'] >= 0.8 else False, axis = 1)

print(f"Number of RSIDs matched fuzzily to CBSA: {univ_rsid_zip_cw_fuzz[univ_rsid_zip_cw_fuzz['matched']]['rsid_left'].nunique()}")

# combining matches
univ_rsid_zip_cw_allmatches = pd.concat([univ_rsid_zip_exact_matched[['rsid','cbsa']], univ_rsid_zip_cw_fuzz[univ_rsid_zip_cw_fuzz['matched']==True].rename(columns = {'rsid_left': 'rsid', 'cbsa_right': 'cbsa'})[['rsid','cbsa']]]).drop_duplicates()

univ_rsid_zip_cw_allmatches['nmatch'] = univ_rsid_zip_cw_allmatches.groupby('rsid')['cbsa'].transform('nunique')

print(f"Total RSIDs matched to CBSA: {univ_rsid_zip_cw_allmatches['rsid'].nunique()}")
print(f"Total RSIDs matched uniquely to CBSA: {univ_rsid_zip_cw_allmatches[univ_rsid_zip_cw_allmatches['nmatch']==1]['rsid'].nunique()}")

# save
univ_rsid_zip_cw_allmatches.to_parquet(f"{root}/data/int/rsid_cbsa_cw.parquet", index = False)

######################################
# MERGING DATA: IPEDS ID TO CBSA
######################################
ipeds_cw_formerge2 = ipeds_cw.rename(columns={'instname': 'school_name', 'ZIP': 'campus_zip_code'})[['school_name','campus_zip_code','UNITID']].dropna().drop_duplicates()
ipeds_cw_formerge2['campus_zip_code'] = ipeds_cw_formerge['campus_zip_code'].astype(int)

# merge univ_zip_cw with zip_cbsa_cw and cbsa_msa_cw via CBSA/ZIP code
ipeds_cbsa_cw = ipeds_cw_formerge2.merge(zip_cbsa_cw[['zip','cbsa']], left_on = 'campus_zip_code', right_on = 'zip', how = 'left').merge(cbsa_msa_cw_bycbsa, left_on = 'cbsa', right_on = 'cbsa_code', how = 'left').groupby(['UNITID', 'cbsa_y']).size().reset_index().rename(columns = {'cbsa_y':'cbsa', 0:'n'}) 

ipeds_cbsa_cw['nmatch'] = ipeds_cbsa_cw.groupby('UNITID')['cbsa'].transform('nunique')

print("IPEDS to CBSA:")
print(f"Total UNITIDs Matched to CBSA: {ipeds_cbsa_cw['UNITID'].nunique()} out of {ipeds_cw['UNITID'].nunique()}")
print(f"Total UNITIDs matched uniquely to CBSA: {ipeds_cbsa_cw[ipeds_cbsa_cw['nmatch']==1]['UNITID'].nunique()}")

# save
ipeds_cbsa_cw.to_parquet(f"{root}/data/int/ipeds_cbsa_cw.parquet", index = False)

######################################
# TODO: MERGING DATA: rsid to IPEDS ID
######################################