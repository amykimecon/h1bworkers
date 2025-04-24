# File Description: Cleaning and Merging User and Position Data from Reveliio
# Author: Amy Kim
# Date Created: Wed Apr 9

# Imports and Paths
import duckdb as ddb
import pandas as pd
import numpy as np
import seaborn as sns
import employer_merge_helpers as emh
import analysis_helpers as ah
from name2nat import Name2nat
import os
import re
import json

root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

con = ddb.connect()

#title case function
con.create_function("title", lambda x: x.title(), ['VARCHAR'], 'VARCHAR')

# name2nat
my_nanat = Name2nat()
def name2nat_fun(name, nanat = my_nanat):
    return json.dumps(nanat(name, top_n = 5)[0][1])

con.create_function("name2nat", lambda x: name2nat_fun(x), ['VARCHAR'], 'VARCHAR')


# CLEANING COUNTRIES
import json 
with open(f"{root}/data/crosswalks/country_dict.json", "r") as json_file:
    country_cw_dict = json.load(json_file)

def get_gmaps_country(adr, dict = country_cw_dict):
    if adr is None:
        return None 
    
    #print(adr)
    stub = re.search(', ([A-z\\s]+)[^,]*$', adr)
    if stub is not None:
        if stub.group(1) in dict.keys():
            return dict[stub.group(1)]
        elif stub.group(1) in dict.values():
            return stub.group(1)

    country_search = '(' + '|'.join(dict.values()) + ')'
    country_match = re.search(country_search, adr)

    if country_match is not None:
        #print(f"indirect match: {country_match.group(1)}")
        return country_match.group(1)
    
    return "No valid country match found"

con.create_function("get_gmaps_country", lambda x: get_gmaps_country(x), ['VARCHAR'], 'VARCHAR')

def get_all_nats(str, dict = country_cw_dict):
    if str is None:
        return []
    items = re.sub('\\\\u00e9','e', re.sub('(\\[\\[|\\]\\])','',str)).split('], [')
    out = []
    for s in items:
        if re.search('^"([A-z\\s\\-]+)", ', s) is None or re.search('", ([0-9\\.e\\-]+)$', s) is None:
            print(s)
        else:
            nat = re.search('^"([A-z\\s\\-]+)", ', s).group(1)
            prob = float(re.search('", ([0-9\\.e\\-]+)$', s).group(1))
            if nat in dict.keys():
                out = out + [[dict[nat], prob]]
            else:
                out = out + [[nat, prob]]
    return out
    # return [[dict[re.search('^"([A-z]+)", ', s).group(1)], float(re.search('", ([0-9\\.e\\-]+)$', s).group(1))] for s in items]

def get_top_nat(str):
    allnats = get_all_nats(str)
    if allnats is None or len(allnats) == 0:
        return ""
    return allnats[0][0]

def get_main_nats(str):
    allnats = get_all_nats(str)
    if allnats is None:
        return []
    return [s[0] for s in allnats if s[1] > 0.01]

con.create_function("get_all_nats", lambda x: get_all_nats(x), ['VARCHAR'], 'VARCHAR')
con.create_function("top_nat", lambda x: get_top_nat(x), ['VARCHAR'], 'VARCHAR')
con.create_function("get_main_nats", get_main_nats, ['VARCHAR'], 'VARCHAR')


## duplicate rcids (companies that appear more than once in linkedin data)
dup_rcids = con.read_csv(f"{root}/data/int/dup_rcids_mar20.csv")

## matched company data from R
rmerge = con.read_csv(f"{root}/data/int/good_match_ids_mar20.csv")

## raw FOIA bloomberg data
foia_raw_file = con.read_csv(f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv")

## joining raw FOIA data with merged data to get foia_ids in raw foia data
foia_with_ids = con.sql("SELECT *, CASE WHEN matched IS NULL THEN 0 ELSE matched END AS matchind FROM ((SELECT * FROM foia_raw_file WHERE NOT FEIN = '(b)(3) (b)(6) (b)(7)(c)') AS a LEFT JOIN (SELECT lottery_year, FEIN, foia_id, 1 AS matched FROM rmerge GROUP BY lottery_year, FEIN, foia_id) AS b ON a.lottery_year = b.lottery_year AND a.FEIN = b.FEIN)")

## revelio data (pre-filtered to only companies in rmerge)
merged_pos = con.read_parquet(f"{root}/data/int/rev_merge_mar20.parquet")
#occ_cw = con.read_csv(f"{root}/data/crosswalks/rev_occ_to_foia_freq.csv")

# merged_pos_full = con.sql(
#     f"""SELECT ROW_NUMBER() OVER(PARTITION BY main_rcid, user_id ORDER BY get_quarter(startdate)) AS rank, weight,
#         get_quarter(startdate) AS startq,
#         get_fiscal_year(startdate) AS startfy,
#         CASE WHEN enddate IS NULL AND startdate IS NOT NULL THEN 2025.25 ELSE get_quarter(enddate) END AS endq,
#         CASE WHEN enddate IS NULL AND startdate IS NOT NULL THEN 2025 ELSE get_fiscal_year(enddate) END AS endfy,
#         country, main_rcid, user_id, 
#         role_k1500, top3occ, top10occ, mean_n_100
#     FROM (SELECT startdate, enddate, user_id, country, weight, pos.role_k1500 AS role_k1500, top3occ, top10occ, mean_n_100,
#             CASE WHEN main_rcid IS NULL THEN pos.rcid ELSE main_rcid END AS main_rcid
#         FROM (merged_pos AS pos
#         LEFT JOIN
#             (SELECT role_k1500, top3occ, top10occ, mean_n_100 FROM occ_cw) AS occ_cw 
#         ON pos.role_k1500 = occ_cw.role_k1500
#         LEFT JOIN 
#             dup_rcids_unique AS dup_cw
#         ON pos.rcid = dup_cw.rcid))""")


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

# IDing outsourcing/staffing companies
foia_main_samp_unfilt = con.sql("SELECT FEIN, lottery_year, COUNT(CASE WHEN ben_multi_reg_ind = 1 THEN 1 END)/COUNT(*) AS share_multireg, COUNT(*) AS n_apps_tot, COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_success, COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END)/COUNT(*) AS win_rate FROM foia_with_ids GROUP BY FEIN, lottery_year")

n = con.sql('SELECT COUNT(*) FROM foia_main_samp_unfilt').df().iloc[0,0]
print(f"Total Employer x Years: {n}")
print(f"Employer x Years with Fewer than 50 Apps: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE n_apps_tot < 50").df().iloc[0,0]}")
print(f"Employer x Years with Fewer than 50% Duplicates: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE share_multireg < 0.5").df().iloc[0,0]}")
print(f"Employer x Years with No Duplicates: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE share_multireg = 0").df().iloc[0,0]}")

foia_main_samp = con.sql("SELECT * FROM foia_main_samp_unfilt WHERE n_apps_tot < 50 AND share_multireg = 0")
print(f"Preferred Sample: {foia_main_samp.df().shape[0]} ({round(100*foia_main_samp.df().shape[0]/n)}%)")

foia_main_samp_def = con.sql("SELECT *, CASE WHEN n_apps_tot < 50 AND share_multireg = 0 THEN 'insamp' ELSE 'outsamp' END AS sampgroup FROM foia_main_samp_unfilt")
con.sql("SELECT sampgroup, SUM(n_success)/SUM(n_apps_tot) AS total_win_rate FROM foia_main_samp_def GROUP BY sampgroup")

samp_to_rcid = con.sql("SELECT * FROM ((SELECT FEIN, lottery_year, sampgroup FROM foia_main_samp_def) AS a JOIN (SELECT FEIN, lottery_year, foia_id FROM foia_with_ids GROUP BY FEIN, lottery_year, foia_id) AS b ON a.FEIN = b.FEIN AND a.lottery_year = b.lottery_year JOIN (SELECT main_rcid, foia_id FROM id_merge) AS c ON b.foia_id = c.foia_id)")

user_samp = con.sql("SELECT user_id FROM ((SELECT main_rcid FROM samp_to_rcid WHERE sampgroup = 'insamp' GROUP BY main_rcid) AS a JOIN (SELECT user_id, startdate, rcid FROM merged_pos) AS b ON a.main_rcid = b.rcid) WHERE startdate >= '2015-01-01' GROUP BY user_id")

######### USER-LEVEL DATA ###########
# IMPORT USER DATA (FROM WRDS SERVER)
rev_users = con.read_parquet(f"{root}/data/wrds/wrds_out/rev_user_merge0.parquet")

for j in range(1,10):
    rev_users = con.sql(f"SELECT * FROM rev_users UNION ALL SELECT * FROM '{root}/data/wrds/wrds_out/rev_user_merge{j}.parquet'")

rev_users_filt = con.sql("SELECT * FROM ((SELECT user_id, fullname, university_raw, education_number, ed_startdate, ed_enddate, degree, field,  university_country, degree_raw, field_raw FROM rev_users GROUP BY user_id, fullname, university_raw, education_number, ed_startdate, ed_enddate, degree, field,  university_country, degree_raw, field_raw) AS a JOIN user_samp AS b ON a.user_id = b.user_id)")

rev_users_test = con.sql("SELECT * FROM rev_users_filt LIMIT 10000")

# IMPORT GMAPS DATA
gmaps = con.sql(f"""SELECT * FROM read_parquet('{"')UNION ALL SELECT * FROM read_parquet('".join([f"{root}/data/int/gmaps_univ_locations/{f}" for f in os.listdir(f"{root}/data/int/gmaps_univ_locations/")])}')""")

gmaps_clean = con.sql("SELECT top_country, top_country_n_users, university_raw, CASE WHEN gmaps_json IS NULL THEN top_country ELSE get_gmaps_country(gmaps_json.candidates[1].formatted_address) END AS univ_gmaps_country, gmaps_json FROM gmaps")

# MERGE GMAPS DATA WITH USERS
rev_users_with_gmaps = con.sql(
"""SELECT user_id, top_country, top_country_n_users, lower(a.university_raw) AS university_raw, university_country AS univ_rev_country, fullname, education_number, univ_gmaps_country, DATEDIFF('year',ed_startdate::DATETIME,ed_enddate::DATETIME) AS ed_length, ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY ed_startdate) AS ed_num,
    CASE 
        WHEN lower(a.university_raw) ~ '.*(high\\s?school).*' OR (degree = '<NA>' AND a.university_raw ~ '.*(HS| High| HIGH| high|H\\.S\\.|S\\.?S\\.?C|H\\.?S\\.?C\\.?)$') THEN 'High School' 
        WHEN degree != '<NA>' THEN degree
        WHEN lower(degree_raw) ~ '.*(cert|credential|course|semester|exchange|abroad|summer|internship|edx|cdl|coursera).*' OR lower(a.university_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera).*' OR lower(field_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera).*' THEN 'Non-Degree'
        WHEN (lower(degree_raw) ~ '.*(undergrad).*') OR (degree_raw ~ '.*(B\\.?A\\.?|B\\.?S\\.?C\\.?E\\.?|B\\.?Sc\\.?|B\\.?A\\.?E\\.?|B\\.?Eng\\.?|A\\.?B\\.?|S\\.?B\\.?|B\\.?B\\.?M\\.?|B\\.?I\\.?S\\.?).*') OR degree_raw ~ '^B\\.?\\s?S\\.?.*' OR lower(field_raw) ~ '.*bachelor.*' OR lower(degree_raw) ~ '.*bachelor.*' THEN 'Bachelor'
        WHEN lower(degree_raw) ~ '.*(master).*' OR degree_raw ~ '^M\\.?(Eng|Sc|A)\\.?.*' THEN 'Master'
        WHEN degree_raw ~ '.*(M\\.?S\\.?C\\.?E\\.?|M\\.?P\\.?A\\.?|M\\.?Eng|M\\.?Sc|M\\.?A).*' OR lower(field_raw) ~ '.*master.*' OR lower(degree_raw) ~ '.*master.*' THEN 'Master'
        WHEN lower(field_raw) ~ '.*(associate).*' OR degree_raw ~ 'A\\.?\\s?A\\.?.*' THEN 'Associate' 
        WHEN degree_raw ~ '^B\\.?\\s?[A-Z].*' THEN 'Bachelor'
        WHEN degree_raw ~ '^M\\.?\\s?[A-Z].*' THEN 'Master'
        ELSE degree END AS degree_clean,
    FROM rev_users_filt AS a LEFT JOIN gmaps_clean AS b ON lower(a.university_raw) = b.university_raw""")

# GET IMPUTED NATS FROM CLEANED NAME
full_names_clean = con.sql(
"""
    SELECT user_id, univ_gmaps_country, univ_rev_country, university_raw, top_country, top_country_n_users, degree_clean, ed_num, ed_length, CASE WHEN univ_gmaps_country IS NULL THEN univ_rev_country ELSE univ_gmaps_country END AS univ_country,
    title(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE((CASE WHEN fullname ~ '.*[a-z].*' THEN 
        REGEXP_REPLACE(REGEXP_REPLACE(fullname, ',.*$', '', 'g'), '\\s([A-Z]\\.?){2,4}$', '', 'g')
        ELSE REGEXP_REPLACE(fullname, ',.*$', '', 'g') END), '\\s?\\(.*\\)$', '', 'g'), 'P\\.?h\\.?D\\.?', '', 'g'), ' +', ' ', 'g'))) AS fullname_clean
    FROM rev_users_with_gmaps
""")

con.sql(f"COPY full_names_clean TO '{root}/data/int/rev_user_gmaps_apr15.parquet' (FORMAT parquet)")

# all_names_with_nats = con.sql("SELECT fullname_clean, name2nat(fullname_clean) AS pred_nats FROM full_names_clean GROUP BY fullname_clean")# WHERE top_country != 'United States' OR top_country IS NULL OR top_country_n_users < 10 OR univ_gmaps_country != 'United States' GROUP BY fullname_clean")

# MERGING DATASETS AND CLEANING
# rev_users_merged = con.sql("SELECT * FROM full_names_clean as a LEFT JOIN all_names_with_nats AS b ON a.fullname_clean = b.fullname_clean")

# con.sql(f"COPY rev_users_merged TO '{root}/data/int/rev_user_merge_apr13.parquet' (FORMAT parquet)")

# nanat countries: collapsing to user level
rev_users_merged = con.read_parquet(f"{root}/data/int/rev_user_merge_apr13.parquet")
rev_users_unique = con.sql("SELECT user_id, fullname_clean, pred_nats FROM rev_users_merged GROUP BY user_id, fullname_clean, pred_nats")

# for gmaps countries: collapsing to user level
rev_users_gmaps = con.read_parquet(f"{root}/data/int/rev_user_gmaps_apr15.parquet")

rev_users_gmaps_unique = con.sql(
"""SELECT a.user_id AS user_id, gmaps_nats_all, univ_nats_all, gmaps_nats_bach, univ_nats_bach, gmaps_hs, univ_hs, gmaps_first_ed, univ_first_ed
    FROM (
        (SELECT user_id, ARRAY_AGG(univ_gmaps_country ORDER BY ed_num) AS gmaps_nats_all
        FROM (
            SELECT DISTINCT user_id, univ_gmaps_country, MIN(ed_num) AS ed_num FROM rev_users_gmaps WHERE univ_gmaps_country IS NOT NULL GROUP BY user_id, univ_gmaps_country
        ) GROUP BY user_id) AS a
        LEFT JOIN  
        (SELECT user_id, ARRAY_AGG(univ_country ORDER BY ed_num) AS univ_nats_all
        FROM (
            SELECT DISTINCT user_id, univ_country, MIN(ed_num) AS ed_num FROM rev_users_gmaps WHERE univ_country IS NOT NULL GROUP BY user_id, univ_country
        ) GROUP BY user_id) AS b
        ON a.user_id = b.user_id
        LEFT JOIN
        (SELECT user_id, ARRAY_AGG(univ_gmaps_country ORDER BY ed_num) AS gmaps_nats_bach 
        FROM (
            SELECT DISTINCT user_id, univ_gmaps_country, MIN(ed_num) AS ed_num FROM rev_users_gmaps WHERE univ_gmaps_country IS NOT NULL AND degree_clean != 'Master' AND degree_clean != 'Doctor' AND ed_length >= 2 GROUP BY user_id, univ_gmaps_country
        ) GROUP BY user_id) AS c
        ON a.user_id = c.user_id
        LEFT JOIN
        (SELECT user_id, ARRAY_AGG(univ_country ORDER BY ed_num) AS univ_nats_bach 
        FROM (
            SELECT DISTINCT user_id, univ_country, MIN(ed_num) AS ed_num FROM rev_users_gmaps WHERE univ_country IS NOT NULL AND degree_clean != 'Master' AND degree_clean != 'Doctor' AND ed_length >= 2 GROUP BY user_id, univ_country
        ) GROUP BY user_id) AS d
        ON a.user_id = d.user_id
        LEFT JOIN 
        (SELECT user_id, univ_gmaps_country AS gmaps_hs, univ_country AS univ_hs
        FROM (
            SELECT *, ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY ed_num) AS hs_rank 
            FROM rev_users_gmaps 
            WHERE degree_clean = 'High School' AND ed_length > 1 AND univ_country IS NOT NULL
        ) WHERE hs_rank = 1) AS e
        ON a.user_id = e.user_id
        LEFT JOIN 
        (SELECT user_id, univ_gmaps_country AS gmaps_first_ed, univ_country AS univ_first_ed
        FROM (
            SELECT *, ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY ed_num) AS rank 
            FROM rev_users_gmaps 
            WHERE univ_country IS NOT NULL AND ed_length > 0
        ) WHERE rank = 1) AS f
        ON a.user_id = f.user_id
    )""")

rev_users_merged_nats = con.sql(
"""
    SELECT a.user_id, get_all_nats(pred_nats) AS all_nats, get_main_nats(pred_nats) AS main_nats, top_nat(pred_nats) AS top_nat, univ_nats_all, univ_nats_bach, univ_hs, univ_first_ed FROM rev_users_unique AS a FULL OUTER JOIN rev_users_gmaps_unique AS b ON a.user_id = b.user_id
""")

con.sql(f"COPY rev_users_merged_nats TO '{root}/data/int/rev_user_nats_final_apr15.parquet' (FORMAT parquet)")

rev_users_merged_nats = con.read_parquet(f"{root}/data/int/rev_user_nats_final_apr15.parquet")

# test version stored in rev_users_merged.df()

# all_names['pred_nat'] = all_names['fullname_clean'].apply(lambda x: my_nanat(x, top_n = 5))

# rev_clean = con.sql(
# """
#     SELECT 
#     fullname, university_country, university_location, degree, user_id,
#     CASE 
#         WHEN lower(university_raw) ~ '.*(high\\s?school).*' OR (degree = '<NA>' AND university_raw ~ '.*(HS| High| HIGH| high|H\\.S\\.|S\\.?S\\.?C|H\\.?S\\.?C\\.?)$') THEN 'High School' 
#         WHEN degree != '<NA>' THEN degree
#         WHEN lower(degree_raw) ~ '.*(cert|credential|course|semester|exchange|abroad|summer|internship|edx|cdl|coursera).*' OR lower(university_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera).*' OR lower(field_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera).*' THEN 'Non-Degree'
#         WHEN (lower(degree_raw) ~ '.*(undergrad).*') OR (degree_raw ~ '.*(B\\.?A\\.?|B\\.?S\\.?C\\.?E\\.?|B\\.?Sc\\.?|B\\.?A\\.?E\\.?|B\\.?Eng\\.?|A\\.?B\\.?|S\\.?B\\.?|B\\.?B\\.?M\\.?|B\\.?I\\.?S\\.?).*') OR degree_raw ~ '^B\\.?\\s?S\\.?.*' OR lower(field_raw) ~ '.*bachelor.*' OR lower(degree_raw) ~ '.*bachelor.*' THEN 'Bachelor'
#         WHEN lower(degree_raw) ~ '.*(master).*' OR degree_raw ~ '^M\\.?(Eng|Sc|A)\\.?.*' THEN 'Master'
#         WHEN degree_raw ~ '.*(M\\.?S\\.?C\\.?E\\.?|M\\.?P\\.?A\\.?|M\\.?Eng|M\\.?Sc|M\\.?A).*' OR lower(field_raw) ~ '.*master.*' OR lower(degree_raw) ~ '.*master.*' THEN 'Master'
#         WHEN lower(field_raw) ~ '.*(associate).*' OR degree_raw ~ 'A\\.?\\s?A\\.?.*' THEN 'Associate' 
#         WHEN degree_raw ~ '^B\\.?\\s?[A-Z].*' THEN 'Bachelor'
#         WHEN degree_raw ~ '^M\\.?\\s?[A-Z].*' THEN 'Master'
#         ELSE degree END AS degree_clean,
#     TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(strip_accents(lower(university_raw)), '\\s*\\(.*\\)\\s*', ' ', 'g'), '[^a-z0-9\\s]', ' ', 'g'), '\\s+', ' ', 'g')) AS univ_raw_clean,
#     degree_raw, field_raw, university_raw
#     FROM rev_raw
# """
# )

# # grouping by user x university (filtering out non-degree) x country, then by university x country, then by university (taking top non-null country)
# univ_names = con.sql(
# """SELECT university_raw, 
#     (ARRAY_AGG(university_country ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country, 
#     (ARRAY_AGG(n_users_univ_ctry ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country_n_users, 
#     SUM(n_users_univ_ctry) AS n_users 
# FROM (
#     SELECT university_raw, university_country, COUNT(*) AS n_users_univ_ctry 
#     FROM (
#         SELECT lower(university_raw) AS university_raw, user_id, university_country 
#         FROM rev_clean 
#         WHERE degree_clean != 'Non-Degree'
#         GROUP BY university_raw, university_country, user_id
#     ) GROUP BY university_country, university_raw
# ) GROUP BY university_raw ORDER BY n_users DESC
# """).df()

# univ_names_for_lookup = univ_names.loc[(univ_names['top_country'].isnull() == 1)|(univ_names['top_country_n_users'] < 10)]
# print(f"Total universities to check: {univ_names_for_lookup.shape[0]}")

# ## HARMONIZING COUNTRIES
# from unidecode import unidecode
# iso_cw = pd.read_csv(f"{root}/data/crosswalks/iso_country_codes.csv")
# iso_cw['name_clean'] = iso_cw['name'].str.replace(',.*$','',regex =True).str.replace(' \\(.*\\)','',regex=True).apply(unidecode)
# # iso_names = pd.concat([iso_cw['name'],iso_cw['name_clean']]).unique() + ["United States", "USA"]
# country_cw_dict = iso_cw[['name','name_clean']].set_index('name').T.to_dict('records')[0]|iso_cw[['alpha-3','name_clean']].set_index('alpha-3').T.to_dict('records')[0]

# for key, val in country_cw_dict.items():
#     if val == "Brunei Darussalam":
#         country_cw_dict[key] = 'Brunei'

#     if val == 'Congo':
#         country_cw_dict[key] = "Democratic Republic of the Congo"

#     if val == "Lao People's Democratic Republic":
#         country_cw_dict[key] = 'Laos'
    
#     if val == "Puerto Rico":
#         country_cw_dict[key] = 'United States'

#     if val == "Russian Federation":
#         country_cw_dict[key] = 'Russia'
    
#     if val == "Syrian Arab Republic":
#         country_cw_dict[key] = 'Syria'

#     if val == "United Kingdom of Great Britain and Northern Ireland":
#         country_cw_dict[key] = "United Kingdom"
    
#     if val == "United States of America":
#         country_cw_dict[key] = 'United States'
    
#     if val == "Viet Nam":
#         country_cw_dict[key] = 'Vietnam'

# country_cw_dict['DRC'] = 'Democratic Republic of the Congo'
# country_cw_dict['Czech Republic'] = 'Czechia'
# country_cw_dict["Korea, Democratic People's Republic of"] = "North Korea"
# country_cw_dict['PRK'] = "North Korea"
# country_cw_dict["Korea, Republic of"] = "South Korea"
# country_cw_dict['KOR'] = "South Korea"
# country_cw_dict['Virgin Islands (British)'] = 'British Virgin Islands'
# country_cw_dict['VGB'] = 'British Virgin Islands'
# country_cw_dict['VIR'] = 'U.S. Virgin Islands'
# country_cw_dict['Virgin Islands (U.S.)'] = 'U.S. Virgin Islands'
# country_cw_dict['East Timor'] = 'Timor-Leste'
# country_cw_dict['UAE'] = "United Arab Emirates"
# country_cw_dict['UK'] = "United Kingdom"
# country_cw_dict['USA'] = 'United States'
# country_cw_dict['Kashmir'] = "India"
# country_cw_dict['Burkinabe'] = "Burkina Faso"
# country_cw_dict['Tibetan'] = 'Nepal'
# country_cw_dict['Tamil'] = 'Sri Lanka'

# def get_nanat_country(nat, dict = country_cw_dict):
#     print(f"NAT: {nat}")
#     if nat in dict.keys():
#         return dict[nat]
    
#     if nat in dict.items():
#         return nat
    
#     if re.sub("n$","",nat) in dict.items():
#         print(re.sub("n$","",nat))
#         return re.sub("n$","",nat)
    
#     stub = re.sub("(ian|ish|ese|e)$","",nat)
#     print(stub)

#     if len(stub) >= 3:
#         for (key, value) in dict.items():
#             if re.search(stub, key) is not None:
#                 valid = input(f"Is {key} the correct match? (Y/N) ")
#                 if valid == "Y":
#                     return key
#             if value != key and re.search(stub, value) is not None:
#                 valid = input(f"Is {value} the correct match? (Y/N) ")
#                 if valid == "Y":
#                     return value
        
#     return None 
            
# unmatched = []
# for nat in my_nanat.name2nats.keys():
#     country = get_nanat_country(nat)
#     if country is not None:
#         country_cw_dict[nat] = country
#     else:
#         unmatched = unmatched + [nat]

# for nat in unmatched:
#     ctry = input(f"Nationality of {nat}: ")
#     if ctry != "":
#         country_cw_dict[nat] = ctry
#         print(ctry)
#     else:
#         print("ERROR")

# import json 
# with open(f"{root}/data/crosswalks/country_dict.json", "w") as json_file:
#     json.dump(country_cw_dict, json_file)