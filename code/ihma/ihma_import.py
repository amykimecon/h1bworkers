# File Description: First pass at constructing shift-share design for IHMA project

# Imports and Paths
import wrds
import duckdb as ddb
import pandas as pd
import sys 
import os 
import time
import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import re 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

## WRDS
db = wrds.Connection(wrds_username='amykimecon')

# just importing everything directly lol
# print("importing data from WRDS")
# all_educ = db.raw_sql("SELECT * FROM revelio.individual_user_education")
# all_educ_raw = db.raw_sql("SELECT user_id, university_raw, education_number, degree_raw, field_raw FROM revelio.individual_user_education_raw")

# print("saving data to parquet")
# all_educ.to_parquet(f"{root}/data/raw/wrds_all_educ.parquet")
# all_educ_raw.to_parquet(f"{root}/data/raw/wrds_all_educ_raw.parquet")
test = False
saveloc_users = f"{root}/data/int/ihma_educ/ihma_educ_full_raw"
outfileloc_users = f"{root}/data/int/ihma_educ_full_raw_nov2025.parquet"

saveloc_positions = f"{root}/data/int/ihma_positions/ihma_positions_all"
outfileloc_positions = f"{root}/data/int/ihma_positions_all_nov2025.parquet"

users = pd.read_parquet(f"{root}/data/clean/ihma_main_user_samp_nov2025.parquet")
usersamp = users[['user_id']]

# def fetch_user_education_with_raw(userids: pd.DataFrame, *, db=db) -> pd.DataFrame:
#     """
#     Helper for chunked imports: returns education rows plus the matching
#     `individual_user_education_raw` fields for the supplied user ids.

#     Parameters
#     ----------
#     userids : DataFrame
#         DataFrame containing a `user_id` column. Matches helpers.chunk_query
#         expectations.
#     db : wrds.Connection
#         Active WRDS connection (defaults to the module-level connection).
#     """

#     if "user_id" not in userids.columns:
#         raise KeyError("Expected userids DataFrame to contain 'user_id' column.")

#     userid_subset = ",".join(userids["user_id"].astype(str))
#     if not userid_subset:
#         return pd.DataFrame(
#             columns=[
#                 "user_id",
#                 "university_name",
#                 "university_raw",
#                 "rsid",
#                 "degree",
#                 "education_number",
#                 "startdate",
#                 "enddate",
#                 "degree_raw",
#                 "field_raw"
#             ]
#         )

#     query = f"""
#         SELECT
#             educ.user_id,
#             educ.university_name,
#             raw.university_raw,
#             educ.rsid,
#             educ.degree,
#             educ.education_number,
#             educ.startdate,
#             educ.enddate,
#             raw.degree_raw,
#             raw.field_raw
#         FROM (
#             SELECT *
#             FROM revelio.individual_user_education
#             WHERE user_id IN ({userid_subset})
#         ) AS educ
#         LEFT JOIN (
#             SELECT user_id, education_number, university_raw, degree_raw, field_raw
#             FROM revelio.individual_user_education_raw
#         ) AS raw
#         ON educ.user_id = raw.user_id
#         AND educ.education_number = raw.education_number
#     """
#     return db.raw_sql(query)


# if test:
#     testextra = "LIMIT 100000"
# else:
#     testextra = ""
#     print(f"Current Time: {datetime.datetime.now()}")
#     print("Generating user sample...")
#     t1user = time.time()

# usersamp_raw = db.raw_sql(f"SELECT DISTINCT user_id FROM revelio.individual_user_education {testextra}")

# print(f"Total users before filtering: {usersamp_raw.shape[0]}")

# # filtering to relevant users
# j = 100
# usersamp = help.chunk_query(usersamp_raw, j = j, fun = fetch_user_education_with_raw, d = 10000, verbose = True, extraverbose=test, outpath = saveloc_users)

# if not test:
#     print("Merging user sample chunks...")
#     usersamp = help.chunk_merge(saveloc_users, j = j, outfile = outfileloc_users, verbose = True)

#     print(f"Total users after filtering: {usersamp.shape[0]}")
#     print(f"Time Elapsed: {round((time.time()-t1user)/60, 2)} minutes")


# ## TOGGLES
# test = False
# usersamp_fromscratch = True
# #country_name = 'France'
# country_tag = 'all'
# version_tag = 'nov2025'

# saveloc_users = f"{root}/data/int/ihma_users/ihma_users_{country_tag}"
# outfileloc_users = f"{root}/data/int/ihma_users_{country_tag}_{version_tag}.parquet"

# saveloc_educ = f"{root}/data/int/ihma_educ/ihma_educ_{country_tag}"
# outfileloc_educ = f"{root}/data/int/ihma_educ_{country_tag}_{version_tag}.parquet"

# saveloc_positions = f"{root}/data/int/ihma_positions/ihma_positions_{country_tag}"
# outfileloc_positions = f"{root}/data/int/ihma_positions_{country_tag}_{version_tag}.parquet"

# # creating directories
# os.makedirs(saveloc_users, exist_ok=True)
# os.makedirs(saveloc_educ, exist_ok=True)
# os.makedirs(saveloc_positions, exist_ok=True)

# if test:
#     saveloc_users = None 
#     outfileloc_users = ""

#     saveloc_educ = None 
#     outfileloc_educ = ""

#     saveloc_positions = None 
#     outfileloc_positions = ""

# j = 10
# #####################
# # DEFINING USER SAMPLE
# #####################
# def get_user_samp_filt(userids, db = db):
#     userid_subset = ','.join(userids['user_id'].astype('str'))

#     users_filt = db.raw_sql(
#     f"""
#     SELECT * 
#     FROM (
#         SELECT *, 
#             ROW_NUMBER() OVER(PARTITION BY user_id, degree_clean ORDER BY gradyr ASC) AS edu_rank,
#             MAX(CASE WHEN degree_clean = 'Bachelor' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS any_bach
#         FROM (
#             SELECT *
#             FROM (
#                 SELECT educ.user_id, 
#                     CASE WHEN enddate IS NOT NULL THEN SUBSTRING(enddate::VARCHAR,1,4)::INT 
#                         WHEN startdate IS NOT NULL THEN SUBSTRING(startdate::VARCHAR, 1, 4)::INT + 4 
#                         ELSE NULL END AS gradyr, 
#                     university_name, university_raw, rsid, university_country, educ.education_number, field, degree, degree_raw, field_raw,
#                     {help.degree_clean_regex_sql()} AS degree_clean
#                 FROM (
#                     SELECT * FROM revelio.individual_user_education WHERE user_id IN ({userid_subset})
#                 ) AS educ 
#                 LEFT JOIN (
#                     SELECT user_id, education_number, university_raw, field_raw, degree_raw FROM revelio.individual_user_education_raw
#                 ) AS educ_raw
#                 ON educ.user_id = educ_raw.user_id AND educ.education_number = educ_raw.education_number
#             ) WHERE degree_clean = 'Bachelor' OR degree_clean = 'Missing'
#         )
#     ) WHERE (any_bach = 1 AND degree_clean = 'Bachelor' AND edu_rank = 1) OR (any_bach = 0 AND edu_rank = 1)
#     """)

#     return users_filt

# if usersamp_fromscratch:
#     # sample of users: start with everyone
#     if test:
#         testextra = "LIMIT 100000"
#     else:
#         testextra = ""
#         print(f"Current Time: {datetime.datetime.now()}")
#         print("Generating user sample...")
#         t1user = time.time()

#     usersamp_raw = db.raw_sql(f"SELECT DISTINCT user_id FROM revelio.individual_user_education {testextra}")

#     print(f"Total users before filtering: {usersamp_raw.shape[0]}")

#     # filtering to relevant users
#     usersamp = help.chunk_query(usersamp_raw, j = j, fun = get_user_samp_filt, d = 1000, verbose = True, extraverbose=test, outpath = saveloc_users)

#     if not test:
#         print("Merging user sample chunks...")
#         usersamp = help.chunk_merge(saveloc_users, j = j, outfile = outfileloc_users, verbose = True)
    
#     print(f"Total users after filtering: {usersamp.shape[0]}")
#     print(f"Time Elapsed: {round((time.time()-t1user)/60, 2)} minutes")

# else:
#     usersamp = pd.read_parquet(f"{root}/data/int/ihma_users_{country_tag}_{version_tag}.parquet")

# #####################
# # HELPERS
# # #####################
# # function to get relevant education data given list of usernames
# def get_merge_query_educ(userids, db = db):
#     userid_subset = ','.join(userids['user_id'].astype('str'))

#     educ_us = db.raw_sql(
#     f"""
#     SELECT educ.user_id, university_name, university_raw, rsid, educ.education_number, 
#         CASE WHEN enddate IS NOT NULL THEN SUBSTRING(enddate::VARCHAR,1,4)::INT 
#             WHEN startdate IS NOT NULL THEN SUBSTRING(startdate::VARCHAR, 1, 4)::INT + 4 
#             ELSE NULL END AS gradyr,
#         degree, field, degree_raw, field_raw, university_country, university_location 
#     FROM 
#         (SELECT * FROM revelio.individual_user_education WHERE user_id IN ({userid_subset}) AND university_country = 'United States') AS educ 
#         LEFT JOIN
#         (SELECT user_id, university_raw, education_number, degree_raw, field_raw FROM revelio.individual_user_education_raw) AS educ_raw
#         ON educ.user_id = educ_raw.user_id AND educ.education_number = educ_raw.education_number   
#     """)

#     return educ_us

# function to get relevant position data given list of usernames
def get_merge_query_positions(userids, db = db):
    userid_subset = ','.join(userids['user_id'].astype('str'))
    # positions = db.raw_sql(
    # f"""
    # SELECT user_id, position_id, position_number, rcid, country, state, metro_area, msa, startdate, enddate, role_k1500, salary, total_compensation FROM revelio.individual_positions WHERE user_id IN ({userid_subset})
    # """)
    positions = db.raw_sql(
    f"""
    SELECT user_id, country, state, metro_area, startdate, enddate, salary, total_compensation FROM revelio.individual_positions WHERE user_id IN ({userid_subset})
    """)

    return positions

#####################
# QUERYING WRDS
#####################
t0_0 = time.time()
j = 10
print(f"Running ihma_import for All Countries on {usersamp.shape[0]} userids")
print("---------------------")

# running chunks and saving
# print("Education: Querying and saving individual chunks...")
# testdf = help.chunk_query(usersamp, j = j, fun = get_merge_query_educ, d = 20, verbose = True, extraverbose=test, outpath = saveloc_educ)

# t1_1 = time.time()
# print(f"Done! Time Elapsed: {round((t1_1-t0_0)/3600, 2)} hours")

print("Positions: Querying and saving individual chunks...")
help.chunk_query(usersamp, j = j, fun = get_merge_query_positions, d = 10000, verbose = True, extraverbose=test, outpath = saveloc_positions)

t2_2 = time.time()
print(f"Done! Time Elapsed: {round((t2_2-t0_0)/3600, 2)} hours")

# getting merged chunks
if not test:
    print("Merging chunks...")
    # out_educ = help.chunk_merge(saveloc_educ, j = j, outfile = outfileloc_educ, verbose = True)

    out_positions = help.chunk_merge(saveloc_positions, j = j, outfile = outfileloc_positions, verbose = True)

print(f"Script Ended: {datetime.datetime.now()}")
