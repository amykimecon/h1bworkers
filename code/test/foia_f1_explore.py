# File Description: Cleaning and Merging User and Position Data from Reveliio
# Author: Amy Kim
# Date Created: Wed Apr 9 (Updated June 26 2025)

# Imports and Paths
import duckdb as ddb
import pandas as pd
import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

con = ddb.connect()

#####################################
## IMPORTING DATA
#####################################
folders = [dirname for dirname in os.listdir(f"{root}/data/raw/foia_sevp") if os.path.isdir(f"{root}/data/raw/foia_sevp/{dirname}")]

# reading in raw data and concatenating
allyrs_raw = [pd.read_excel(f"{root}/data/raw/foia_sevp/{dirname}/{filename}").assign(dir = dirname, filename = filename) for dirname in folders for filename in os.listdir(f"{root}/data/raw/foia_sevp/{dirname}")]




first = True
for dir in os.listdir(f"{root}/data/raw/foia_sevp"):
    if os.path.isdir(f"{root}/data/raw/foia_sevp/{dir}") and dir == "2013":
        for filename in os.listdir(f"{root}/data/raw/foia_sevp/{dir}"):
            if first:
                con.sql(f"CREATE TABLE foia_f1_raw AS SELECT * FROM read_xlsx('{root}/data/raw/foia_sevp/{dir}/{filename}', all_varchar = true)")
                first = False
            else:
                con.sql(f"INSERT INTO foia_f1_raw SELECT * FROM read_xlsx('{root}/data/raw/foia_sevp/{dir}/{filename}', all_varchar = true)")

# for j in range(1,10):
#     rev_raw = con.sql(f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{wrds_out}/rev_user_merge{j}.parquet'")

# reading in raw data (read cipcode as string to preserve decimals)
allyrs_raw = [pd.read_excel(f"{root}/data/raw/ipeds/{filename}", sheet_name = 'TableLibrary', names = ['group', 'awardlvl', 'ciptitle', 'cipcode', 'total', 'native', 'asian', 'black', 'hispanic', 'pacificislander', 'white', 'twoormore', 'unknown', 'nonresident'], dtype = {'cipcode': str}, skiprows = 5).assign(year = int('20' + re.findall('ipeds_([0-9]{2}).xlsx', filename)[0])) for filename in os.listdir(f"{root}/data/raw/ipeds") if re.match('ipeds_([0-9]{2}).xlsx',filename) is not None]
