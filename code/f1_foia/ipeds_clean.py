# PROGRAM TO CLEAN IPEDS DATA
# Imports and Paths
import duckdb as ddb
import pandas as pd
import sys 
import os 
import time
import pyarrow as pa
import pyarrow.parquet as pq
import re 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 
INT_FOLDER = f"{root}/data/int/int_files_nov2025"

# READING IN RAW FILES
raw_labs = pd.read_stata(
    f"{root}/data/raw/ipeds_completions_all.dta",
    columns=["cipcode", "awlevel", "cip2dig", "year", "progid"]
).rename(columns={"cipcode": "cipcode_lab", "awlevel": "awlevel_lab", "cip2dig": "cip2dig_lab"})
# Ensure label columns are strings so pyarrow doesn't choke on categorical/int hybrids
raw_labs[["cipcode_lab", "awlevel_lab", "cip2dig_lab"]] = raw_labs[["cipcode_lab", "awlevel_lab", "cip2dig_lab"]].astype(str)

raw = pd.read_stata(f"{root}/data/raw/ipeds_completions_all.dta", convert_categoricals=False).merge(raw_labs, on = ["year", "progid"])

# list of STEM-OPT eligible CIP codes
stemopt_cip = pd.read_csv(f"{root}/data/crosswalks/stem_opt_cip_codes.csv")
stemopt_cip['cipcode'] = (stemopt_cip['2020 CIP Code']).astype(str).str.replace('.', '').astype(int)
stemopt_cip['STEMOPT'] = 1

clean = raw[(raw['majornum']==1)&(raw['ctotalt'] > 0)&(raw['year']>=2002)].merge(
    stemopt_cip[['cipcode', 'STEMOPT']], on='cipcode', how='left'
)
clean['STEMOPT'] = clean['STEMOPT'].fillna(0).astype(int)

# SAVING TO PARQUET
clean.to_parquet(f"{INT_FOLDER}/ipeds_completions_all.parquet", index = False)
