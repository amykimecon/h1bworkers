# File Description: Creating role_k* to DOT code crosswalk
# Author: Amy Kim
# Date Created: Mon Mar 31


# Imports and Paths
import duckdb as ddb
import pandas as pd
import wrds
import re
import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

con = ddb.connect()
db = wrds.Connection(wrds_username='amykimecon')


def _parse_lookup_version(table_name):
    m = re.match(r"^individual_role_lookup(?:_?v?(\d+))?$", table_name)
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 0


def _find_lookup_table(dbconn):
    tables = dbconn.list_tables("revelio")
    candidates = []
    for table in tables:
        version = _parse_lookup_version(table)
        if version is not None:
            candidates.append((version, table))
    if not candidates:
        raise RuntimeError("No revelio table matched pattern individual_role_lookup[versionnum].")
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][1]


def _parse_role_col(col):
    m = re.match(r"^role_k(\d+)(?:_v(\d+))?$", col)
    if not m:
        return None
    occnum = int(m.group(1))
    version = int(m.group(2)) if m.group(2) else 0
    return (version, occnum, col)


def _find_role_col(cols):
    parsed = [p for p in (_parse_role_col(c) for c in cols) if p is not None]
    if not parsed:
        raise RuntimeError("No role_k[occnum][versionnum] column found in role lookup table.")
    parsed.sort()
    return parsed[-1][2]

#####################
# IMPORTING DATA
#####################
## raw FOIA bloomberg data (getting shares of reported dot_codes)
foia_raw_file = con.read_csv(f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv")
foia_dot_codes = con.sql("SELECT COUNT(*) AS n, DOT_CODE FROM foia_raw_file WHERE DOT_CODE != 'NA' GROUP BY DOT_CODE").df()
foia_dot_codes['share'] = foia_dot_codes['n']/foia_dot_codes.sum()['n']
foia_dot_codes['rank'] = foia_dot_codes['n'].rank(ascending = False)

## dot to onet code crosswalk
dot_onet_cw = pd.read_excel(f"{root}/data/crosswalks/DOT_to_ONET_SOC.xlsx", skiprows = 3, names = ['dot_code', 'dot_title', 'onet_code', 'onet_title'])

## revelio occ crosswalk
lookup_table = _find_lookup_table(db)
rev_occ_cw = db.raw_sql(f"SELECT * FROM revelio.{lookup_table}")
role_col = _find_role_col(rev_occ_cw.columns)
print(f"Using role lookup table: revelio.{lookup_table}")
print(f"Using role column: {role_col}")

#####################
# MERGING
#####################
# getting 3-digit dot code
dot_onet_cw['dot3_code'] = dot_onet_cw['dot_code'].apply(lambda x: x[0:3])

# merging revelio occupations with dot-onet crosswalk (on onet codes), then merging with foia dot codes
occ_cw = rev_occ_cw.merge(dot_onet_cw.groupby(['dot3_code','onet_code'])['onet_title'].agg('count').reset_index(), how = "left", on = "onet_code").merge(foia_dot_codes, how = 'left', left_on = 'dot3_code', right_on = 'DOT_CODE')

occ_cw[['n_foia','share_foia']] = occ_cw[['n','share']].fillna(0)
occ_cw['rank_foia'] = occ_cw['rank'].fillna(1000)

# grouping by revelio role key + occupation dimensions
group_cols = [role_col, 'onet_code']
if 'onet_title_x' in occ_cw.columns:
    group_cols.append('onet_title_x')
elif 'onet_title' in occ_cw.columns:
    group_cols.append('onet_title')

occ_cw_grouped = (
    occ_cw.groupby(group_cols)
    .agg(
        mean_n_foia=pd.NamedAgg(column='n_foia', aggfunc='mean'),
        max_share_foia=pd.NamedAgg(column='share_foia', aggfunc='max'),
        min_rank=pd.NamedAgg(column='rank_foia', aggfunc='min'),
    )
    .reset_index()
    .sort_values('min_rank')
)
occ_cw_grouped['top3occ'] = occ_cw_grouped['min_rank'] <= 3
occ_cw_grouped['top10occ'] = occ_cw_grouped['min_rank'] <= 10
occ_cw_grouped['mean_n_100'] = occ_cw_grouped['mean_n_foia'] >= 100

occ_cw_grouped.to_csv(f"{root}/data/crosswalks/rev_occ_to_foia_freq.csv", index=False)
