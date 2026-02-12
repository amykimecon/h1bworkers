"""Using NameTrace to infer gender probability and subregion from names."""

import datetime
import duckdb as ddb
from nametrace import NameTracer
import json
import os
import sys
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from config import *
import rev_indiv_config as rcfg

con = ddb.connect()
my_nt = NameTracer()
print(f"Using config: {rcfg.ACTIVE_CONFIG_PATH}")

# toggles
test = rcfg.NAMETRACE_TEST
testn = rcfg.NAMETRACE_TESTN

# data locations
wrds_users_file = rcfg.WRDS_USERS_PARQUET
legacy_chunk_files = rcfg.LEGACY_WRDS_USER_MERGE_SHARDS


def load_wrds_users():
    """Load the latest consolidated WRDS users file; fallback to old shards."""
    if os.path.exists(wrds_users_file):
        print(f"Loading consolidated users file: {wrds_users_file}")
        return con.read_parquet(wrds_users_file)

    print("Consolidated users file not found. Falling back to legacy rev_user_merge shards.")
    rev_raw = con.read_parquet(legacy_chunk_files[0])
    for j in range(1, 10):
        rev_raw = con.sql(
            f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{legacy_chunk_files[j]}'"
        )
    return rev_raw


def get_female_score(pred):
    gender = pred.get("gender")
    if isinstance(gender, list):
        for label, score in gender:
            if label == "female":
                return score
    return None


def get_region(pred):
    region_probs = pred.get("subregion")
    if isinstance(region_probs, list):
        return region_probs
    return []


def nametrace_run(df, nt=my_nt):
    if df.empty:
        return df

    df_out = df.copy()
    batch_size = max(1, min(100000, df_out.shape[0]))
    pred = nt.predict(df_out["fullname_clean"].to_list(), batch_size=batch_size, topk=5)
    df_out["nametrace_json"] = pred
    df_out["f_prob_nt"] = df_out["nametrace_json"].apply(get_female_score)
    # Serialize nested region output so parquet chunk writes are stable.
    df_out["region_probs_json"] = df_out["nametrace_json"].apply(
        lambda x: json.dumps(get_region(x))
    )
    return df_out[["fullname_clean", "f_prob_nt", "region_probs_json"]]


t0 = time.time()
print(f"Current Time: {datetime.datetime.now()}")

####################
## IMPORTING DATA ##
####################
rev_raw = load_wrds_users()

# title-case function used inside fullname cleaner
con.create_function("title", lambda x: x.title(), ["VARCHAR"], "VARCHAR")

####################
## GETTING NAMES ##
####################
rev_clean = con.sql(
    f"""
    SELECT
        fullname,
        university_country,
        university_location,
        degree,
        user_id,
        {help.degree_clean_regex_sql()} AS degree_clean,
        {help.inst_clean_regex_sql('university_raw')} AS univ_raw_clean,
        CASE WHEN fullname ~ '.*[A-z].*' THEN {help.fullname_clean_regex_sql('fullname')} ELSE '' END AS fullname_clean,
        degree_raw,
        field_raw,
        university_raw
    FROM rev_raw
"""
)

if test:
    rev_clean = con.sql(f"SELECT * FROM rev_clean LIMIT {testn}")

# collapse to name level
rev_names = con.sql(
    """
    SELECT *, ROW_NUMBER() OVER(ORDER BY fullname_clean) AS rownum
    FROM (
        SELECT fullname_clean
        FROM rev_clean
        WHERE fullname_clean != ''
        GROUP BY fullname_clean
    )
"""
)

rev_names_df = rev_names.df()
n = rev_names_df.shape[0]
print(f"Running rev_indiv_nametrace on {n} unique names")
print("-------------------------")

if n == 0:
    run_tag = "test" if test else rcfg.RUN_TAG
    wide_outfile = f"{root}/data/int/rev_names_nametrace_{run_tag}.parquet"
    long_outfile = f"{root}/data/int/rev_names_nametrace_long_{run_tag}.parquet"
    pd.DataFrame(columns=["fullname_clean", "f_prob_nt"]).to_parquet(wide_outfile)
    pd.DataFrame(columns=["fullname_clean", "f_prob_nt", "region", "prob"]).to_parquet(
        long_outfile
    )
    print("No names to score; wrote empty outputs.")
    print(f"Saved: {wide_outfile}")
    print(f"Saved: {long_outfile}")
    print(f"Script Ended: {datetime.datetime.now()}")
    raise SystemExit(0)

#############################
## QUERYING + SAVING CHUNKS ##
#############################
run_tag = "test" if test else rcfg.RUN_TAG
chunk_dir = rcfg.NAMETRACE_CHUNK_DIR
os.makedirs(chunk_dir, exist_ok=True)

saveloc = f"{chunk_dir}/nametrace_revelio_{run_tag}_"
j = rcfg.NAMETRACE_CHUNKS
d = rcfg.NAMETRACE_CHUNK_SIZE
j_eff = max(1, min(j, rev_names_df.shape[0]))

print("Querying and saving individual chunks...")
help.chunk_query(
    rev_names_df,
    j=j_eff,
    fun=nametrace_run,
    d=d,
    verbose=True,
    extraverbose=test,
    outpath=saveloc,
)

df_out_all_concat = help.chunk_merge(saveloc, j=j_eff, verbose=True)

if df_out_all_concat is None or df_out_all_concat.empty:
    raise ValueError("No NameTrace output was produced.")

#############################
## RESHAPING + FINAL SAVES ##
#############################
df_out_all_concat["region_probs"] = df_out_all_concat["region_probs_json"].apply(
    lambda x: json.loads(x) if isinstance(x, str) and x != "" else []
)
df_exp = df_out_all_concat.explode("region_probs")
df_exp_notnull = df_exp.loc[df_exp["region_probs"].notna()].copy()
df_exp_notnull["region"] = df_exp_notnull["region_probs"].apply(
    lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else None
)
df_exp_notnull["prob"] = df_exp_notnull["region_probs"].apply(
    lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else None
)
df_exp_notnull = df_exp_notnull.loc[df_exp_notnull["region"].notna()].copy()

base_names = df_out_all_concat[["fullname_clean", "f_prob_nt"]].drop_duplicates()
regions_wide = (
    df_exp_notnull.pivot_table(
        index="fullname_clean", columns="region", values="prob", aggfunc="max"
    )
    .reset_index()
)

out_wide = base_names.merge(regions_wide, how="left", on="fullname_clean")
out_long = base_names.merge(
    df_exp_notnull[["fullname_clean", "region", "prob"]],
    how="left",
    on="fullname_clean",
)

wide_outfile = f"{root}/data/int/rev_names_nametrace_{run_tag}.parquet"
long_outfile = f"{root}/data/int/rev_names_nametrace_long_{run_tag}.parquet"

out_wide.to_parquet(wide_outfile)
out_long.to_parquet(long_outfile)

t1 = time.time()
print(f"Saved: {wide_outfile}")
print(f"Saved: {long_outfile}")
print(f"Done! Time Elapsed: {round((t1 - t0) / 3600, 2)} hours")
print(f"Script Ended: {datetime.datetime.now()}")
