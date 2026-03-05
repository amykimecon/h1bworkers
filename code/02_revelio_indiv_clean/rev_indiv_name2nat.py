# File Description: Using Name2Nat Package to get nationalities from names
# Author: Amy Kim
# Date Created: June 26 2025

# Imports and Paths
import duckdb as ddb
import time
import datetime
import json
import sys 
import os
import pandas as pd
from name2nat import Name2nat

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from config import *
import rev_indiv_config as rcfg

con = ddb.connect()
my_nanat = Name2nat()
print(f"Using config: {rcfg.ACTIVE_CONFIG_PATH}")

# test toggle
test = rcfg.NAME2NAT_TEST
testn = rcfg.NAME2NAT_TESTN

if test:
    print("TEST VERSION")

####################
## IMPORTING DATA ##
####################
# Importing Data (From WRDS Server)
if os.path.exists(rcfg.WRDS_USERS_PARQUET):
    print(f"Loading consolidated users file: {rcfg.WRDS_USERS_PARQUET}")
    rev_raw = con.read_parquet(rcfg.WRDS_USERS_PARQUET)
else:
    print("Consolidated users file not found. Falling back to legacy rev_user_merge shards.")
    rev_raw = con.read_parquet(rcfg.LEGACY_WRDS_USER_MERGE_SHARDS[0])
    for j in range(1, len(rcfg.LEGACY_WRDS_USER_MERGE_SHARDS)):
        rev_raw = con.sql(
            f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{rcfg.LEGACY_WRDS_USER_MERGE_SHARDS[j]}'"
        )

#title case function
con.create_function("title", lambda x: x.title(), ['VARCHAR'], 'VARCHAR')

####################
## GETTING NAMES ##
####################
rev_clean = con.sql(
f"""
    SELECT 
    fullname, university_country, university_location, degree, user_id,
    {help.degree_clean_regex_sql()} AS degree_clean,
    {help.inst_clean_regex_sql('university_raw')} AS univ_raw_clean,
    CASE WHEN fullname ~ '.*[A-z].*' THEN {help.fullname_clean_regex_sql('fullname')} ELSE '' END AS fullname_clean,
    degree_raw, field_raw, university_raw
    FROM rev_raw
"""
)

if test:
    rev_clean = con.sql(f"SELECT * FROM rev_clean LIMIT {testn}")

# collapsing to name level
rev_names = con.sql("SELECT *, ROW_NUMBER() OVER(ORDER BY fullname_clean) AS rownum FROM (SELECT fullname_clean FROM rev_clean WHERE fullname_clean != '' GROUP BY fullname_clean)")

# helper functions
def _split_name_tokens(fullname):
    if not isinstance(fullname, str):
        return ("", "")
    parts = [p for p in fullname.strip().split(" ") if p]
    if not parts:
        return ("", "")
    if len(parts) == 1:
        return (parts[0], parts[0])
    return (parts[0], parts[-1])


def _name2nat_chunk_run(df, input_col, output_col):
    df_out = df.copy()
    names = df_out[input_col].fillna("").astype(str).tolist()
    df_out[output_col] = [dict(n[1]) for n in my_nanat(names, top_n = 20)]
    return df_out[[input_col, output_col]]


def _score_unique_names(name_df, input_col, output_col, chunk_stub, chunks, chunk_size):
    unique_df = name_df[[input_col]].dropna().drop_duplicates().reset_index(drop=True)
    unique_df = unique_df[unique_df[input_col].astype(str).str.strip() != ""]
    n_unique = unique_df.shape[0]
    if n_unique == 0:
        return pd.DataFrame(columns=[input_col, output_col])

    j_eff = max(1, min(chunks, n_unique))
    print(f"Scoring {n_unique} unique values for {input_col} using {j_eff} chunks")

    help.chunk_query(
        unique_df,
        j = j_eff,
        fun = lambda df: _name2nat_chunk_run(df, input_col, output_col),
        d = chunk_size,
        verbose = True,
        extraverbose = test,
        outpath = chunk_stub,
    )
    scored = help.chunk_merge(chunk_stub, j = j_eff, verbose = True)
    return scored[[input_col, output_col]].drop_duplicates(subset=[input_col])


def _existing_lookup(existing_df, key_col, value_col):
    if existing_df is None:
        return pd.DataFrame(columns=[key_col, value_col])
    if key_col not in existing_df.columns or value_col not in existing_df.columns:
        return pd.DataFrame(columns=[key_col, value_col])
    out = existing_df[[key_col, value_col]].dropna(subset=[key_col, value_col]).drop_duplicates(subset=[key_col])
    return out

## declaring constants
if test:
    saveloc = rcfg.NAME2NAT_CHUNK_STUB_TEST
else:
    saveloc = rcfg.NAME2NAT_CHUNK_STUB

j = rcfg.NAME2NAT_CHUNKS
d = rcfg.NAME2NAT_CHUNK_SIZE
outfile = rcfg.NAME2NAT_PARQUET_TEST if test else rcfg.NAME2NAT_PARQUET

## running code
t0 = time.time()
print(f"Current Time: {datetime.datetime.now()}")

rev_names_df = rev_names.df()
first_last = rev_names_df["fullname_clean"].apply(_split_name_tokens)
rev_names_df["first_name_clean"] = first_last.apply(lambda x: x[0])
rev_names_df["last_name_clean"] = first_last.apply(lambda x: x[1])

print(f"Running rev_indiv_name2nat on {rev_names_df.shape[0]} unique full names")
print("-------------------------")

if rev_names_df.shape[0] == 0:
    print("No names found for the current configuration; skipping query and merge.")
    print(f"Script Ended: {datetime.datetime.now()}")
    raise SystemExit(0)

existing_out = None
if os.path.exists(outfile):
    print(f"Loading existing output for reuse: {outfile}")
    existing_out = con.read_parquet(outfile).df()
else:
    print("No prior output found; computing all channels from scratch.")

# 1) FULL NAME: reuse prior predictions when available, score only missing full names.
existing_full_col = None
if existing_out is not None:
    if "pred_nats_full" in existing_out.columns:
        existing_full_col = "pred_nats_full"
    elif "pred_nats_name" in existing_out.columns:
        existing_full_col = "pred_nats_name"

full_lookup = pd.DataFrame(columns=["fullname_clean", "pred_nats_full"])
if existing_full_col is not None:
    full_lookup = _existing_lookup(existing_out, "fullname_clean", existing_full_col).rename(
        columns={existing_full_col: "pred_nats_full"}
    )
    print(f"Reused full-name predictions for {full_lookup.shape[0]} names")

full_base = rev_names_df[["fullname_clean"]].merge(full_lookup, on="fullname_clean", how="left")
full_missing = full_base[full_base["pred_nats_full"].isna()][["fullname_clean"]]
if full_missing.shape[0] > 0:
    print(f"Scoring missing full names: {full_missing.shape[0]}")
    full_scored = _score_unique_names(
        full_missing,
        input_col = "fullname_clean",
        output_col = "pred_nats_full",
        chunk_stub = f"{saveloc}_full_",
        chunks = j,
        chunk_size = d,
    )
    full_lookup = pd.concat([full_lookup, full_scored], ignore_index=True).drop_duplicates(
        subset=["fullname_clean"], keep="last"
    )
else:
    print("No missing full-name predictions.")

# 2) FIRST NAME: score unique first names (reuse prior cache when available).
first_lookup = _existing_lookup(existing_out, "first_name_clean", "pred_nats_first")
if first_lookup.shape[0] > 0:
    print(f"Reused first-name predictions for {first_lookup.shape[0]} names")

first_base = rev_names_df[["first_name_clean"]].drop_duplicates().merge(
    first_lookup, on="first_name_clean", how="left"
)
first_missing = first_base[first_base["pred_nats_first"].isna()][["first_name_clean"]]
if first_missing.shape[0] > 0:
    print(f"Scoring missing first names: {first_missing.shape[0]}")
    first_scored = _score_unique_names(
        first_missing,
        input_col = "first_name_clean",
        output_col = "pred_nats_first",
        chunk_stub = f"{saveloc}_first_",
        chunks = j,
        chunk_size = d,
    )
    first_lookup = pd.concat([first_lookup, first_scored], ignore_index=True).drop_duplicates(
        subset=["first_name_clean"], keep="last"
    )
else:
    print("No missing first-name predictions.")

# 3) LAST NAME: score unique last names (reuse prior cache when available).
last_lookup = _existing_lookup(existing_out, "last_name_clean", "pred_nats_last")
if last_lookup.shape[0] > 0:
    print(f"Reused last-name predictions for {last_lookup.shape[0]} names")

last_base = rev_names_df[["last_name_clean"]].drop_duplicates().merge(
    last_lookup, on="last_name_clean", how="left"
)
last_missing = last_base[last_base["pred_nats_last"].isna()][["last_name_clean"]]
if last_missing.shape[0] > 0:
    print(f"Scoring missing last names: {last_missing.shape[0]}")
    last_scored = _score_unique_names(
        last_missing,
        input_col = "last_name_clean",
        output_col = "pred_nats_last",
        chunk_stub = f"{saveloc}_last_",
        chunks = j,
        chunk_size = d,
    )
    last_lookup = pd.concat([last_lookup, last_scored], ignore_index=True).drop_duplicates(
        subset=["last_name_clean"], keep="last"
    )
else:
    print("No missing last-name predictions.")

# Final assembly
out = (
    rev_names_df
    .merge(full_lookup, on="fullname_clean", how="left")
    .merge(first_lookup, on="first_name_clean", how="left")
    .merge(last_lookup, on="last_name_clean", how="left")
)

# Fallback: if first/last missing, use full-name posterior.
out["pred_nats_first"] = out["pred_nats_first"].where(out["pred_nats_first"].notna(), out["pred_nats_full"])
out["pred_nats_last"] = out["pred_nats_last"].where(out["pred_nats_last"].notna(), out["pred_nats_full"])
# Keep legacy column for backward compatibility.
out["pred_nats_name"] = out["pred_nats_full"]
out = out.sort_values("rownum").reset_index(drop=True)
out.to_parquet(outfile)
t1 = time.time()
print(f"Saved merged output: {outfile}")
print(f"Done! Time Elapsed: {round((t1-t0)/3600,2)} hours")

print(f"Script Ended: {datetime.datetime.now()}")
