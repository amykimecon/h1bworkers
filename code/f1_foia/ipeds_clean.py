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
IPEDS_RAW_PATH = f"{root}/data/raw/ipeds_completions_all.dta"
IPEDS_RAW_FALLBACK_PATH = f"{root}/data/raw/ipeds/completions_all.dta"


def _resolve_ipeds_raw_path() -> str:
    if os.path.exists(IPEDS_RAW_PATH):
        return IPEDS_RAW_PATH
    if os.path.exists(IPEDS_RAW_FALLBACK_PATH):
        return IPEDS_RAW_FALLBACK_PATH
    return IPEDS_RAW_PATH


def _read_completions_with_labels(path: str) -> pd.DataFrame:
    raw = pd.read_stata(path, convert_categoricals=False)
    label_cols = ["cipcode_lab", "awlevel_lab", "cip2dig_lab"]
    if all(col in raw.columns for col in label_cols):
        raw[label_cols] = raw[label_cols].astype(str)
        return raw

    raw_labs = pd.read_stata(
        path,
        columns=["cipcode", "awlevel", "cip2dig", "year", "progid"],
    ).rename(columns={"cipcode": "cipcode_lab", "awlevel": "awlevel_lab", "cip2dig": "cip2dig_lab"})
    raw_labs[["cipcode_lab", "awlevel_lab", "cip2dig_lab"]] = raw_labs[
        ["cipcode_lab", "awlevel_lab", "cip2dig_lab"]
    ].astype(str)
    return raw.merge(raw_labs, on=["year", "progid"])

# READING IN RAW FILES
# completions data
raw = _read_completions_with_labels(_resolve_ipeds_raw_path())

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
os.makedirs(INT_FOLDER, exist_ok=True)
clean.to_parquet(f"{INT_FOLDER}/ipeds_completions_all.parquet", index = False)
