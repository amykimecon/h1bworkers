# PROGRAM TO CLEAN IPEDS DATA
# Imports and Paths
import duckdb as ddb
import pandas as pd
import sys 
import os 
import time
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 
INT_FOLDER = f"{root}/data/int/int_files_nov2025"

# READING IN RAW FILES
# completions data
raw_labs = pd.read_stata(
    f"{root}/data/raw/ipeds_completions_all.dta",
    columns=["cipcode", "awlevel", "cip2dig", "year", "progid"]
).rename(columns={"cipcode": "cipcode_lab", "awlevel": "awlevel_lab", "cip2dig": "cip2dig_lab"})
# Ensure label columns are strings so pyarrow doesn't choke on categorical/int hybrids
raw_labs[["cipcode_lab", "awlevel_lab", "cip2dig_lab"]] = raw_labs[["cipcode_lab", "awlevel_lab", "cip2dig_lab"]].astype(str)

raw = pd.read_stata(f"{root}/data/raw/ipeds_completions_all.dta", convert_categoricals=False).merge(raw_labs, on = ["year", "progid"])

# directory data
raw_hd_labs = pd.read_stata(f"{root}/data/raw/ipeds/directoryinfo2023.dta", convert_categoricals=True, columns = ['sector','instcat','c21basic','instsize','UNITID']).rename(columns={'sector':'sector_lab', 'instcat':'instcat_lab', 'c21basic':'c21basic_lab', 'instsize':'instsize_lab'})

raw_hd = pd.read_stata(f"{root}/data/raw/ipeds/directoryinfo2023.dta", convert_categoricals=False)[['sector','instcat','c21basic','instsize','UNITID']].merge(raw_hd_labs, on = ['UNITID']).rename(columns = {'UNITID':'unitid'})

# list of STEM-OPT eligible CIP codes
stemopt_cip = pd.read_csv(f"{root}/data/crosswalks/stem_opt_cip_codes.csv")
stemopt_cip['cipcode'] = (stemopt_cip['2020 CIP Code']).astype(str).str.replace('.', '').astype(int)
stemopt_cip['STEMOPT'] = 1

clean = raw[(raw['majornum']==1)&(raw['ctotalt'] > 0)&(raw['year']>=2002)&(raw['cipcode']>10000)].merge(
    stemopt_cip[['cipcode', 'STEMOPT']], on='cipcode', how='left'
).merge(
    raw_hd, on='unitid', how='left'
)
clean['STEMOPT'] = clean['STEMOPT'].fillna(0).astype(int)

# DEFINING IHMPs
clean['ihmp'] = (clean['share_intl'] >= 0.5)&(clean['awlevel'] == 7)&(clean['majornum'] == 1)&(clean['STEMOPT'] == 1)
clean['awlevel_group'] = np.where(clean['awlevel'].isin([5, 7, 17, 9]), clean['awlevel'].replace({
    5: 'Bachelor',
    7: 'Master',
    17: 'Doctor',
    9: 'Doctor'
}), 'Other')
clean['intl_ihmp'] = clean['ihmp']*clean['cnralt']

# SAVING TO PARQUET
clean.to_parquet(f"{INT_FOLDER}/ipeds_completions_all.parquet", index = False)
