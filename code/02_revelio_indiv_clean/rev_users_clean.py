# File Description: Cleaning and Merging User and Position Data from Reveliio
# Author: Amy Kim
# Date Created: Wed Apr 9 (Updated Feb 12 2026)

# Imports and Paths
import duckdb as ddb
import json
import sys 
import os 
import re
import subprocess
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from config import *
import rev_indiv_config as rcfg

# CONSTANTS
CUTOFF_SHARE_MULTIREG=0
CUTOFF_N_APPS_TOT=10
ANGLO_COUNTRIES = (
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
    "New Zealand",
)
NANAT_ANGLO_PRESSURE_CUTOFF = 0.35
NANAT_W_DEFAULT_FULL = 0.55
NANAT_W_DEFAULT_LAST = 0.35
NANAT_W_CROWDED_FULL = 0.30
NANAT_W_CROWDED_LAST = 0.60
NANAT_W_FIRST = 0.10

# TOGGLE FOR TESTING
test = rcfg.REV_USERS_CLEAN_TEST
test_user = rcfg.REV_USERS_CLEAN_TEST_USER
run_tag = rcfg.RUN_TAG

con = ddb.connect()
t_script0 = time.time()
print(f"Using config: {rcfg.ACTIVE_CONFIG_PATH}")


def _configure_duckdb_runtime(connection):
    threads = max(1, os.cpu_count() or 1)
    connection.sql(f"PRAGMA threads={threads}")
    connection.sql("PRAGMA preserve_insertion_order=false")
    tmp_dir = os.path.join(root, ".tmp", "duckdb")
    os.makedirs(tmp_dir, exist_ok=True)
    escaped_tmp = tmp_dir.replace("'", "''")
    connection.sql(f"PRAGMA temp_directory='{escaped_tmp}'")


_configure_duckdb_runtime(con)


def _parse_role_col(col):
    m = re.match(r"^role_k(\d+)(?:_v(\d+))?$", col)
    if not m:
        return None
    occnum = int(m.group(1))
    version = int(m.group(2)) if m.group(2) else 0
    return (version, occnum, col)


def _detect_role_col(cols):
    parsed = [p for p in (_parse_role_col(c) for c in cols) if p is not None]
    if not parsed:
        return None
    parsed.sort()
    return parsed[-1][2]

# Importing Country Codes Crosswalk
with open(f"{root}/data/crosswalks/country_dict.json", "r") as json_file:
    country_cw_dict = json.load(json_file)

# Importing Region Crosswalk
with open(f"{root}/data/crosswalks/subregion_dict.json", "r") as json_file:
    subregion_dict = json.load(json_file)

## Creating DuckDB functions from python helpers
#title case function
con.create_function("title", lambda x: x.title(), ['VARCHAR'], 'VARCHAR')

# country crosswalk function (kept for any legacy callers; SQL code uses _country_cw table instead)
con.create_function("get_std_country", lambda x: help.get_std_country(x, country_cw_dict), ['VARCHAR'], 'VARCHAR')

# region crosswalk function (kept for any legacy callers; SQL code uses _subregion_cw table instead)
con.create_function("get_country_subregion", lambda x: help.get_country_subregion(x, country_cw_dict, subregion_dict), ['VARCHAR'], 'VARCHAR')

# Load crosswalk dicts into DuckDB tables for vectorized lookups (avoids row-by-row Python UDF calls)
con.execute("CREATE OR REPLACE TABLE _country_cw (raw_country VARCHAR, std_country VARCHAR)")
con.executemany("INSERT INTO _country_cw VALUES (?, ?)", list(country_cw_dict.items()))
con.execute("CREATE OR REPLACE TABLE _subregion_cw (std_country VARCHAR, subregion VARCHAR)")
con.executemany("INSERT INTO _subregion_cw VALUES (?, ?)", list(subregion_dict.items()))

#####################################
## IMPORTING DATA
#####################################
print('Loading all data sources...')
## duplicate rcids (companies that appear more than once in linkedin data)
dup_rcids = con.read_csv(rcfg.DUP_RCIDS_CSV)

## matched company data from LLM-reviewed crosswalk
llm_crosswalk = con.read_csv(
    rcfg.LLM_CROSSWALK_CSV,
    strict_mode=False,
    ignore_errors=True,
    all_varchar=True,
)
good_matches = con.read_csv(
    rcfg.GOOD_MATCH_IDS_CSV,
    strict_mode=False,
    ignore_errors=True,
    all_varchar=True,
)

## raw FOIA Bloomberg data
# foia_unique_id is required later to build deterministic foia_indiv_id values
# that align with downstream spotchecks and firm-outcomes code.
foia_raw_path = f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all_withids.csv"
if not os.path.exists(foia_raw_path):
    raise FileNotFoundError(
        f"Missing FOIA raw file with ids: {foia_raw_path}. "
        "rev_users_clean expects foia_unique_id in the Bloomberg source."
    )
foia_raw_file = con.read_csv(foia_raw_path)

## Normalize raw LLM crosswalk first (used for bridge/lookup tables).
con.sql(
"""
CREATE OR REPLACE TABLE llm_crosswalk_prepped AS
SELECT
    foia_firm_uid,
    CASE
        WHEN REGEXP_REPLACE(COALESCE(CAST(fein_clean AS VARCHAR), ''), '[^0-9]', '', 'g') = '' THEN NULL
        ELSE COALESCE(
            NULLIF(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(COALESCE(CAST(fein_clean AS VARCHAR), ''), '[^0-9]', '', 'g'),
                    '^0+',
                    ''
                ),
                ''
            ),
            '0'
        )
    END AS fein_norm,
    TRY_CAST(fein_year AS INTEGER) AS lottery_year,
    TRY_CAST(rcid AS DOUBLE) AS rcid_raw,
    LOWER(TRIM(crosswalk_validity_label)) AS crosswalk_validity_label,
    LOWER(TRIM(firm_status)) AS firm_status,
    TRY_CAST(score AS DOUBLE) AS score_num,
    TRY_CAST(confidence AS DOUBLE) AS confidence_num
FROM llm_crosswalk
"""
)

## LLM match subset mapped to foia_firm_uid x rcid (allowing multiple rcids per firm uid)
con.sql(
"""
CREATE OR REPLACE TABLE llm_crosswalk_unique AS
SELECT
    foia_firm_uid,
    fein_norm,
    lottery_year,
    rcid_raw AS rcid,
    CASE WHEN firm_status = 'exact match' THEN 1 ELSE 3 END AS category_priority,
    score_num,
    confidence_num
FROM llm_crosswalk_prepped
WHERE foia_firm_uid IS NOT NULL
  AND TRIM(foia_firm_uid) != ''
  AND rcid_raw IS NOT NULL
  AND lottery_year IS NOT NULL
  AND crosswalk_validity_label = 'valid_match'
"""
)

con.sql(
"""
CREATE OR REPLACE TABLE llm_uid_bridge AS
SELECT foia_firm_uid, fein_norm, lottery_year
FROM llm_crosswalk_prepped
WHERE foia_firm_uid IS NOT NULL
  AND TRIM(foia_firm_uid) != ''
  AND fein_norm IS NOT NULL
  AND lottery_year IS NOT NULL
GROUP BY foia_firm_uid, fein_norm, lottery_year
"""
)

con.sql(
"""
CREATE OR REPLACE TABLE good_matches_mapped AS
SELECT
    b.foia_firm_uid,
    CASE
        WHEN REGEXP_REPLACE(COALESCE(CAST(a.FEIN AS VARCHAR), ''), '[^0-9]', '', 'g') = '' THEN NULL
        ELSE COALESCE(
            NULLIF(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(COALESCE(CAST(a.FEIN AS VARCHAR), ''), '[^0-9]', '', 'g'),
                    '^0+',
                    ''
                ),
                ''
            ),
            '0'
        )
    END AS fein_norm,
    TRY_CAST(a.lottery_year AS INTEGER) AS lottery_year,
    TRY_CAST(a.rcid AS DOUBLE) AS rcid,
    TRIM(a.foia_id) AS legacy_foia_id
FROM good_matches AS a
LEFT JOIN llm_uid_bridge AS b
    ON (
        CASE
            WHEN REGEXP_REPLACE(COALESCE(CAST(a.FEIN AS VARCHAR), ''), '[^0-9]', '', 'g') = '' THEN NULL
            ELSE COALESCE(
                NULLIF(
                    REGEXP_REPLACE(
                        REGEXP_REPLACE(COALESCE(CAST(a.FEIN AS VARCHAR), ''), '[^0-9]', '', 'g'),
                        '^0+',
                        ''
                    ),
                    ''
                ),
                '0'
            )
        END
    ) = b.fein_norm
   AND TRY_CAST(a.lottery_year AS INTEGER) = b.lottery_year
WHERE TRY_CAST(a.rcid AS DOUBLE) IS NOT NULL
  AND TRY_CAST(a.lottery_year AS INTEGER) IS NOT NULL
"""
)
good_unmapped_n = con.sql(
    """
    SELECT COUNT(*) FROM (
        SELECT legacy_foia_id, fein_norm, lottery_year
        FROM good_matches_mapped
        WHERE foia_firm_uid IS NULL
        GROUP BY legacy_foia_id, fein_norm, lottery_year
    )
    """
).df().iloc[0, 0]
print(f"Legacy good matches with no mapped foia_firm_uid: {good_unmapped_n}")

con.sql(
"""
CREATE OR REPLACE TABLE good_matches_unique AS
SELECT
    foia_firm_uid,
    fein_norm,
    lottery_year,
    rcid,
    2 AS category_priority,
    NULL::DOUBLE AS score_num,
    NULL::DOUBLE AS confidence_num
FROM good_matches_mapped
WHERE foia_firm_uid IS NOT NULL
GROUP BY foia_firm_uid, fein_norm, lottery_year, rcid
"""
)

con.sql(
"""
CREATE OR REPLACE TABLE foia_rcid_union AS
SELECT
    foia_firm_uid,
    fein_norm,
    lottery_year,
    rcid,
    'llm' AS source,
    category_priority,
    score_num,
    confidence_num
FROM llm_crosswalk_unique
UNION
SELECT
    foia_firm_uid,
    fein_norm,
    lottery_year,
    rcid,
    'legacy_good' AS source,
    category_priority,
    score_num,
    confidence_num
FROM good_matches_unique
"""
)
con.sql(
"""
CREATE OR REPLACE TABLE foia_rcid_union_collapsed AS
SELECT
    x.foia_firm_uid,
    x.fein_norm,
    x.lottery_year,
    x.rcid,
    CASE
        WHEN s.has_llm = 1 AND s.has_legacy = 1 THEN 'both'
        WHEN s.has_llm = 1 THEN 'llm'
        ELSE 'legacy_good'
    END AS source,
    CASE
        WHEN x.category_priority = 1 THEN 'exact_llm'
        WHEN x.category_priority = 2 THEN 'legacy_good'
        ELSE 'llm'
    END AS match_priority_bucket,
    x.category_priority,
    x.score_num,
    x.confidence_num,
    ROW_NUMBER() OVER (
        PARTITION BY x.rcid, x.lottery_year
        ORDER BY
            x.category_priority ASC,
            x.score_num DESC NULLS LAST,
            x.confidence_num DESC NULLS LAST,
            x.foia_firm_uid ASC
    ) AS rcid_priority_rank
FROM (
    -- keep one best row per foia_firm_uid x lottery_year x rcid across FEIN variants
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY foia_firm_uid, lottery_year, rcid
            ORDER BY
                category_priority ASC,
                score_num DESC NULLS LAST,
                confidence_num DESC NULLS LAST,
                fein_norm ASC,
                lottery_year ASC,
                foia_firm_uid ASC
        ) AS foia_rcid_best_rank
    FROM foia_rcid_union
) AS x
JOIN (
    SELECT
        foia_firm_uid,
        lottery_year,
        rcid,
        MAX(CASE WHEN source = 'llm' THEN 1 ELSE 0 END) AS has_llm,
        MAX(CASE WHEN source = 'legacy_good' THEN 1 ELSE 0 END) AS has_legacy
    FROM foia_rcid_union
    GROUP BY foia_firm_uid, lottery_year, rcid
) AS s
    ON x.foia_firm_uid = s.foia_firm_uid
   AND x.lottery_year = s.lottery_year
   AND x.rcid = s.rcid
WHERE x.foia_rcid_best_rank = 1
"""
)
llm_uid_n = con.sql(
    "SELECT COUNT(DISTINCT foia_firm_uid) FROM llm_crosswalk_unique"
).df().iloc[0, 0]
llm_uid_rcid_n = con.sql("SELECT COUNT(*) FROM llm_crosswalk_unique").df().iloc[0, 0]
legacy_uid_n = con.sql(
    "SELECT COUNT(DISTINCT foia_firm_uid) FROM good_matches_unique"
).df().iloc[0, 0]
legacy_uid_rcid_n = con.sql("SELECT COUNT(*) FROM good_matches_unique").df().iloc[0, 0]
union_uid_n = con.sql(
    "SELECT COUNT(DISTINCT foia_firm_uid) FROM foia_rcid_union_collapsed"
).df().iloc[0, 0]
union_uid_rcid_n = con.sql("SELECT COUNT(*) FROM foia_rcid_union_collapsed").df().iloc[0, 0]
print(
    f"Loaded matches: llm={llm_uid_n} uids/{llm_uid_rcid_n} uid-rcid rows, "
    f"legacy={legacy_uid_n} uids/{legacy_uid_rcid_n} uid-rcid rows, "
    f"union={union_uid_n} uids/{union_uid_rcid_n} uid-rcid rows "
)

con.sql(
"""
CREATE OR REPLACE TABLE foia_uid_lookup AS
SELECT foia_firm_uid, fein_norm, lottery_year
FROM llm_uid_bridge
GROUP BY foia_firm_uid, fein_norm, lottery_year
"""
)

## joining raw FOIA data to LLM crosswalk with foia_firm_uid
con.sql(
"""
CREATE OR REPLACE TEMP TABLE foia_with_ids AS
SELECT
    a.*,
    b.foia_firm_uid
FROM (
    SELECT
        *,
        CASE
            WHEN REGEXP_REPLACE(COALESCE(CAST(FEIN AS VARCHAR), ''), '[^0-9]', '', 'g') = '' THEN NULL
            ELSE COALESCE(
                NULLIF(
                    REGEXP_REPLACE(
                        REGEXP_REPLACE(COALESCE(CAST(FEIN AS VARCHAR), ''), '[^0-9]', '', 'g'),
                        '^0+',
                        ''
                    ),
                    ''
                ),
                '0'
            )
        END AS fein_norm,
        TRY_CAST(lottery_year AS INTEGER) AS lottery_year_int
    FROM foia_raw_file
    WHERE FEIN != '(b)(3) (b)(6) (b)(7)(c)'
) AS a
LEFT JOIN foia_uid_lookup AS b
    ON a.fein_norm = b.fein_norm
   AND a.lottery_year_int = b.lottery_year
"""
)

# Importing User x Education-level Data (From WRDS Server)
wrds_users_file = rcfg.WRDS_USERS_PARQUET
wrds_users_legacy = rcfg.WRDS_USERS_PARQUET_LEGACY
if os.path.exists(wrds_users_file):
    print(f"Loading users file: {wrds_users_file}")
    rev_raw = con.read_parquet(wrds_users_file)
else:
    print(f"Missing {wrds_users_file}; using legacy {wrds_users_legacy}")
    rev_raw = con.read_parquet(wrds_users_legacy)

# Importing Institution x Country Matches
inst_country_file = rcfg.REV_INST_COUNTRIES_PARQUET
inst_country_legacy = rcfg.REV_INST_COUNTRIES_PARQUET_LEGACY
if os.path.exists(inst_country_file):
    print(f"Loading institution-country file: {inst_country_file}")
    inst_country_cw = con.read_parquet(inst_country_file)
else:
    print(f"Missing {inst_country_file}; using legacy {inst_country_legacy}")
    inst_country_cw = con.read_parquet(inst_country_legacy)

# Importing Name x Country Matches
nanats_file = rcfg.NAME2NAT_PARQUET
nanats = con.read_parquet(nanats_file)

nametrace_long_file = rcfg.NAMETRACE_LONG_PARQUET
nametrace_long_legacy = rcfg.NAMETRACE_LONG_PARQUET_LEGACY
if os.path.exists(nametrace_long_file):
    print(f"Loading NameTrace long file: {nametrace_long_file}")
    nts_long = con.read_parquet(nametrace_long_file)
else:
    print(f"Missing {nametrace_long_file}; using legacy {nametrace_long_legacy}")
    nts_long = con.read_parquet(nametrace_long_legacy)

# Importing User x Position-level Data (all positions)
wrds_positions_file = rcfg.WRDS_POSITIONS_PARQUET
wrds_positions_legacy = rcfg.WRDS_POSITIONS_PARQUET_LEGACY
if os.path.exists(wrds_positions_file):
    print(f"Loading positions file: {wrds_positions_file}")
    merged_pos = con.read_parquet(wrds_positions_file)
else:
    print(f"Missing {wrds_positions_file}; using legacy {wrds_positions_legacy}")
    merged_pos = con.read_parquet(wrds_positions_legacy)

# Occupation crosswalk
occ_cw = con.read_csv(f"{root}/data/crosswalks/rev_occ_to_foia_freq.csv")
role_col = _detect_role_col(merged_pos.columns)
if role_col is None:
    raise ValueError("No role_k[occnum][versionnum] column found in merged_pos.")

if role_col not in occ_cw.columns:
    print(
        f"Column {role_col} not found in rev_occ_to_foia_freq.csv; "
        "regenerating occupation crosswalk..."
    )
    create_occ_script = os.path.join(root, "h1bworkers", "code", "10_misc", "create_occ_cw.py")
    subprocess.run([sys.executable, create_occ_script], check=True)
    occ_cw = con.read_csv(f"{root}/data/crosswalks/rev_occ_to_foia_freq.csv")
    if role_col not in occ_cw.columns:
        raise ValueError(
            f"Occupation crosswalk regenerated but still missing column {role_col}."
        )
print(f"Using occupation role column: {role_col}")

print('Done!')

#####################################
# DEFINING MAIN SAMPLE OF USERS (RCID IN MAIN SAMP AND START DATE AFTER 2015)
#####################################
print('Defining Main Sample of H-1B Companies...')
# IDing outsourcing/staffing companies (defining share of applications that are multiple registrations ['duplicates']; counting total number of apps)
con.sql(
"""
CREATE OR REPLACE TEMP TABLE foia_main_samp_unfilt AS
SELECT
    foia_firm_uid,
    lottery_year,
    COUNT(CASE WHEN ben_multi_reg_ind = 1 THEN 1 END) / COUNT(*) AS share_multireg,
    COUNT(*) AS n_apps_tot,
    COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_success,
    COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) / COUNT(*) AS win_rate
FROM foia_with_ids
WHERE foia_firm_uid IS NOT NULL
GROUP BY foia_firm_uid, lottery_year
"""
)

# # counts (verbose)
# n = con.sql('SELECT COUNT(*) FROM foia_main_samp_unfilt').df().iloc[0,0]
# print(f"Total Employer x Years: {n}")
# print(f'Employer x Years with Fewer than 50 Apps: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE n_apps_tot < 50").df().iloc[0,0]}')
# print(f'Employer x Years with Fewer than 50% Duplicates: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE share_multireg < 0.5").df().iloc[0,0]}')
# print(f'Employer x Years with No Duplicates: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE share_multireg = 0").df().iloc[0,0]}')

# # main sample (conservative): companies with fewer than 50 applications and no duplicate registrations TODO: declare these as constants at top
# foia_main_samp = con.sql(f'SELECT * FROM foia_main_samp_unfilt WHERE n_apps_tot <= {CUTOFF_N_APPS_TOT} AND share_multireg <= {CUTOFF_SHARE_MULTIREG}')
# print(f"Preferred Sample: {foia_main_samp.df().shape[0]} ({round(100*foia_main_samp.df().shape[0]/n)}%)")

# computing win rate by sample
con.sql(
    f"""
    CREATE OR REPLACE TEMP TABLE foia_main_samp_def AS
    SELECT *,
        CASE
            WHEN n_apps_tot <= {CUTOFF_N_APPS_TOT} AND share_multireg <= {CUTOFF_SHARE_MULTIREG}
                THEN 'insamp'
            ELSE 'outsamp'
        END AS sampgroup
    FROM foia_main_samp_unfilt
    """
)
# con.sql("SELECT sampgroup, SUM(n_success)/SUM(n_apps_tot) AS total_win_rate FROM foia_main_samp_def GROUP BY sampgroup")

# Union source (LLM + legacy good matches)
# Keeps multiple rcids per foia_firm_uid x lottery_year.
con.sql(
"""
CREATE OR REPLACE TEMP TABLE samp_to_rcid AS
SELECT
    a.foia_firm_uid,
    a.lottery_year,
    a.sampgroup,
    b.rcid,
    b.source AS match_source,
    b.score_num AS llm_match_score,
    b.match_priority_bucket,
    b.rcid_priority_rank
FROM (
    SELECT foia_firm_uid, lottery_year, sampgroup
    FROM foia_main_samp_def
    WHERE foia_firm_uid IS NOT NULL
    GROUP BY foia_firm_uid, lottery_year, sampgroup
) AS a
JOIN foia_rcid_union_collapsed AS b
    ON a.foia_firm_uid = b.foia_firm_uid
   AND TRY_CAST(a.lottery_year AS INTEGER) = TRY_CAST(b.lottery_year AS INTEGER)
"""
)

# Legacy version (kept for quick revert): expand via dup_rcids/main_rcid.
# samp_to_rcid = con.sql(
# """
# SELECT
#     a.foia_firm_uid,
#     a.FEIN,
#     a.lottery_year,
#     a.sampgroup,
#     b.rcid AS main_rcid,
#     CASE WHEN d.rcid IS NULL THEN b.rcid ELSE d.rcid END AS rcid
# FROM (
#     SELECT foia_firm_uid, FEIN, lottery_year, sampgroup
#     FROM foia_main_samp_def
#     WHERE foia_firm_uid IS NOT NULL
#     GROUP BY foia_firm_uid, FEIN, lottery_year, sampgroup
# ) AS a
# JOIN llm_crosswalk_unique AS b
#     ON a.foia_firm_uid = b.foia_firm_uid
# LEFT JOIN (SELECT DISTINCT main_rcid, rcid FROM dup_rcids) AS d
#     ON b.rcid = d.main_rcid
# """
# )

# writing company sample crosswalk to file
if not test:
    con.sql(f"COPY samp_to_rcid TO '{rcfg.COMPANY_MERGE_SAMPLE_PARQUET}'")

# selecting user ids from list of positions based on whether company in sample and start date is after 2015 (conservative bandwidth) -- TODO: declare cutoff date as constant; TODO: move startdate filter into merged_pos query, avoid pulling people who got promoted after 2015 but started working before 2015
con.sql(
    """
    CREATE OR REPLACE TEMP TABLE user_samp AS
    SELECT user_id
    FROM (
        (SELECT rcid FROM samp_to_rcid WHERE sampgroup = 'insamp' GROUP BY rcid) AS a
        JOIN
        (SELECT user_id, rcid FROM merged_pos WHERE country = 'United States' AND startdate >= '2015-01-01') AS b
        ON a.rcid = b.rcid
    )
    GROUP BY user_id
    """
)
con.sql(
    """
    CREATE OR REPLACE TEMP TABLE merged_pos_scope AS
    SELECT p.*
    FROM merged_pos AS p
    JOIN user_samp AS u
        ON p.user_id = u.user_id
    """
)
print('Done!')
print(f"Total script runtime: {round((time.time() - t_script0)/3600, 2)} hours")

#####################################
### CLEANING AND SAVING FOIA INDIVIDUAL DATA
#####################################
print('Cleaning Individual-level H-1B Data...')

# Save lean raw match parquet (15 raw Bloomberg cols + foia_indiv_id + foia_firm_uid).
# Save lean raw match parquet (15 raw Bloomberg cols + foia_indiv_id + foia_firm_uid).
# foia_indiv_id uses ROW_NUMBER() OVER(ORDER BY foia_unique_id) — same ORDER BY used in
# foia_indiv's inner subquery — so IDs are deterministic and aligned across both parquets.
if not test:
    print("Writing foia_raw_match parquet (raw match cols for TRK exact matching)...")
    con.sql(f"""
        COPY (
            SELECT ROW_NUMBER() OVER(ORDER BY foia_unique_id) AS foia_indiv_id,
                   foia_firm_uid,
                   FIRST_DECISION, BASIS_FOR_CLASSIFICATION, BEN_SEX, BEN_COUNTRY_OF_BIRTH,
                   S3Q1, ED_LEVEL_DEFINITION, BEN_PFIELD_OF_STUDY, i129_employer_name,
                   PET_CITY, PET_STATE, JOB_TITLE, DOT_CODE, BEN_COMP_PAID,
                   valid_from, valid_to
            FROM foia_with_ids
            WHERE foia_firm_uid IS NOT NULL
        ) TO '{rcfg.FOIA_RAW_MATCH_PARQUET}'
    """)
    n_raw = con.execute(f"SELECT COUNT(*) FROM read_parquet('{rcfg.FOIA_RAW_MATCH_PARQUET}')").fetchone()[0]
    print(f"  foia_raw_match: {n_raw:,} rows saved to {rcfg.FOIA_RAW_MATCH_PARQUET}")

# identifying companies that submit 'repeat applications' for losers
# Uses grouped counts to avoid a large row-level self-join on FOIA rows.
con.sql(
"""
CREATE OR REPLACE TEMP TABLE match_rep AS
WITH foia_base AS (
    SELECT *
    FROM foia_with_ids
    WHERE foia_firm_uid IS NOT NULL
),
firm_years AS (
    SELECT
        foia_firm_uid,
        COUNT(DISTINCT TRY_CAST(lottery_year AS INTEGER)) AS n_ly
    FROM foia_base
    GROUP BY foia_firm_uid
),
losers_grp AS (
    SELECT
        foia_firm_uid,
        TRY_CAST(lottery_year AS INTEGER) AS base_ly,
        gender,
        ben_year_of_birth,
        country_of_nationality,
        COUNT(*) AS n_losers
    FROM foia_base
    WHERE TRY_CAST(lottery_year AS INTEGER) < 2024
      AND status_type != 'SELECTED'
    GROUP BY
        foia_firm_uid,
        TRY_CAST(lottery_year AS INTEGER),
        gender,
        ben_year_of_birth,
        country_of_nationality
),
next_grp AS (
    SELECT
        foia_firm_uid,
        TRY_CAST(lottery_year AS INTEGER) AS next_ly,
        gender,
        ben_year_of_birth,
        country_of_nationality,
        COUNT(*) AS n_next
    FROM foia_base
    GROUP BY
        foia_firm_uid,
        TRY_CAST(lottery_year AS INTEGER),
        gender,
        ben_year_of_birth,
        country_of_nationality
),
rep_totals AS (
    SELECT
        l.foia_firm_uid,
        SUM(l.n_losers * COALESCE(n.n_next, 0))::DOUBLE AS n_rep_rows,
        SUM(
            l.n_losers * CASE WHEN COALESCE(n.n_next, 0) > 0 THEN n.n_next ELSE 1 END
        )::DOUBLE AS n_total_rows
    FROM losers_grp AS l
    LEFT JOIN next_grp AS n
        ON l.foia_firm_uid = n.foia_firm_uid
       AND l.base_ly = n.next_ly - 1
       AND l.gender = n.gender
       AND l.ben_year_of_birth = n.ben_year_of_birth
       AND l.country_of_nationality = n.country_of_nationality
    GROUP BY l.foia_firm_uid
)
SELECT
    r.foia_firm_uid,
    CASE
        WHEN r.n_rep_rows = 0 AND f.n_ly > 1 AND r.n_total_rows > 1 THEN 1
        ELSE 0
    END AS no_rep_emp_ind,
    CASE
        WHEN (r.n_rep_rows / NULLIF(r.n_total_rows, 0)) >= 0.8 THEN 1
        ELSE 0
    END AS high_rep_emp_ind,
    r.n_rep_rows / NULLIF(r.n_total_rows, 0) AS share_rep,
    f.n_ly,
    r.n_total_rows AS n
FROM rep_totals AS r
LEFT JOIN firm_years AS f
    ON r.foia_firm_uid = f.foia_firm_uid
"""
)

con.sql(
f"""CREATE OR REPLACE TEMP TABLE foia_indiv AS
SELECT a.foia_firm_uid, a.FEIN, a.lottery_year, a.country,
        COALESCE(sub.subregion, a.country) AS subregion,
        female_ind, yob, status_type, ben_multi_reg_ind, employer_name,
        CASE WHEN BEN_EDUCATION_CODE = 'I' THEN 'Doctor'
            WHEN BEN_EDUCATION_CODE = 'G' OR BEN_EDUCATION_CODE = 'H' THEN 'Master'
            WHEN BEN_EDUCATION_CODE = 'F' THEN 'Bachelor'
            WHEN BEN_EDUCATION_CODE = 'NA' THEN NULL
            ELSE 'Other' END AS highest_ed_level,
        CASE WHEN BEN_CURRENT_CLASS IN ('F1','F2') THEN 'F Visa'
            WHEN BEN_CURRENT_CLASS IN ('UU', 'UN', 'B2', 'B1') THEN 'No Visa'
            WHEN BEN_CURRENT_CLASS = 'NA' THEN NULL
            ELSE 'Other' END AS prev_visa,
        CASE WHEN S3Q1 = 'M' THEN 1 WHEN S3Q1 = 'NA' THEN NULL ELSE 0 END AS ade_lottery, NAICS4,
        CASE WHEN NAICS4 = '5415' THEN 'Computer Systems'
            WHEN NAICS4 = '5413' THEN 'Engineering/Architectural Services'
            WHEN NAICS4 = '5416' THEN 'Consulting Services'
            WHEN NAICS4 = '5417' THEN 'Scientific Research'
            WHEN NAICS2 = '52' THEN 'Finance and Insurance'
            WHEN NAICS4 = '5411' OR NAICS4 = '5412' THEN 'Legal and Accounting'
            WHEN NAICS2 = '54' THEN 'Other Professional Services'
            WHEN NAICS4 = '3254' THEN 'Pharmaceuticals'
            WHEN NAICS4 = '5112' OR NAICS4 = '5182' THEN 'Software and Data or Web Services'
            WHEN NAICS2 = '23' OR NAICS2 = '22' THEN 'Construction and Utilities'
            WHEN NAICS2 = '33' THEN 'Manufacturing'
            WHEN NAICS2 = '62' THEN 'Health Care'
            WHEN NAICS2 = '51' THEN 'Other Information'
            WHEN NAICS_CODE = 'NA' OR NAICS2 = '99' THEN NULL
            ELSE 'Other' END AS industry,
        {help.field_clean_regex_sql('BEN_PFIELD_OF_STUDY')} AS field_clean, DOT_CODE, {help.inst_clean_regex_sql('JOB_TITLE')} AS job_title, n_apps, n_unique_country, no_rep_emp_ind, high_rep_emp_ind, foia_indiv_id, rcid AS main_rcid, rcid
    FROM (
        SELECT FEIN, lottery_year, COALESCE(cw.std_country, fi.country_of_nationality) AS country,
            foia_firm_uid,
            CASE WHEN gender = 'female' THEN 1 ELSE 0 END AS female_ind, ben_year_of_birth AS yob, status_type, ben_multi_reg_ind, employer_name, BEN_PFIELD_OF_STUDY, BEN_EDUCATION_CODE, DOT_CODE, NAICS_CODE, SUBSTRING(NAICS_CODE, 1, 4) AS NAICS4, SUBSTRING(NAICS_CODE, 1, 2) AS NAICS2, JOB_TITLE, BEN_CURRENT_CLASS, S3Q1, COUNT(*) OVER(PARTITION BY foia_firm_uid, lottery_year) AS n_apps, COUNT(DISTINCT country_of_nationality) OVER(PARTITION BY foia_firm_uid, lottery_year) AS n_unique_country, ROW_NUMBER() OVER(ORDER BY fi.foia_unique_id) AS foia_indiv_id
        FROM foia_with_ids AS fi
        LEFT JOIN _country_cw AS cw ON fi.country_of_nationality = cw.raw_country
        WHERE foia_firm_uid IS NOT NULL
    ) AS a JOIN samp_to_rcid AS b ON a.foia_firm_uid = b.foia_firm_uid AND TRY_CAST(a.lottery_year AS INTEGER) = TRY_CAST(b.lottery_year AS INTEGER)
    LEFT JOIN (
        SELECT foia_firm_uid, no_rep_emp_ind, high_rep_emp_ind, share_rep FROM match_rep
    ) AS c ON a.foia_firm_uid = c.foia_firm_uid
    LEFT JOIN _subregion_cw AS sub ON a.country = sub.std_country
    WHERE sampgroup = 'insamp'""")

if not test:
    print("Writing FOIA indiv to file...")
    con.sql(f"COPY foia_indiv TO '{rcfg.FOIA_INDIV_PARQUET}'")
print('Done!')

#####################################
### CLEANING AND MERGING REVELIO USERS TO COUNTRIES
#####################################
print('Cleaning Individual-level Revelio User Data...')

# # Test for specific firm
# rev_users_filt = con.sql(f"""
# SELECT * FROM
#     (SELECT 
#     fullname, degree, user_id, rcid,
#     {help.degree_clean_regex_sql()} AS degree_clean,
#     {help.inst_clean_regex_sql('university_raw')} AS univ_raw_clean,
#     {help.stem_ind_regex_sql()} AS stem_ind,
#     {help.field_clean_regex_sql('field_raw')} AS field_clean,
#     CASE WHEN fullname ~ '.*[A-z].*' THEN {help.fullname_clean_regex_sql('fullname')} ELSE '' END AS fullname_clean,
#     university_raw, f_prob, education_number, ed_enddate, ed_startdate, ROW_NUMBER() OVER(PARTITION BY user_id, education_number) AS dup_num, updated_dt
#     FROM rev_raw)
# WHERE rcid = 857329.0
# """)

con.sql(
    """
    CREATE OR REPLACE TEMP TABLE rev_raw_scope AS
    SELECT r.*
    FROM rev_raw AS r
    JOIN user_samp AS u
        ON r.user_id = u.user_id
    """
)

# Cleaning Revelio Data, removing duplicates
con.sql(
f"""
CREATE OR REPLACE TEMP TABLE rev_clean AS
SELECT * FROM
    (SELECT 
    fullname, degree, user_id, 
    {help.degree_clean_regex_sql()} AS degree_clean,
    {help.inst_clean_regex_sql('university_raw')} AS univ_raw_clean,
    {help.stem_ind_regex_sql()} AS stem_ind,
    {help.field_clean_regex_sql('field_raw')} AS field_clean,
    profile_linkedin_url, user_location, user_country,
    CASE WHEN fullname ~ '.*[A-z].*' THEN {help.fullname_clean_regex_sql('fullname')} ELSE '' END AS fullname_clean,
    university_raw, f_prob, education_number, ed_enddate, ed_startdate, ROW_NUMBER() OVER(PARTITION BY user_id, education_number) AS dup_num, updated_dt
    FROM rev_raw_scope)
WHERE dup_num = 1
"""
)

con.sql("CREATE OR REPLACE TEMP TABLE rev_users_filt AS SELECT * FROM rev_clean")

# testing
if test:
    ids = ",".join(
        con.sql(
            help.random_ids_sql(
                "user_id", "rev_clean", n=rcfg.REV_USERS_CLEAN_RANDOM_USER_SAMPLE_N
            )
        )
        .df()["user_id"]
        .astype(str)
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE rev_users_filt AS
        SELECT *
        FROM rev_clean
        WHERE user_id IN ({ids}) OR user_id = {test_user}
        """
    )

con.sql(
    """
    CREATE OR REPLACE TEMP TABLE rev_user_names AS
    SELECT fullname_clean
    FROM rev_users_filt
    GROUP BY fullname_clean
    """
)
con.sql(
    """
    CREATE OR REPLACE TEMP TABLE nanats_scope AS
    SELECT a.*
    FROM nanats AS a
    JOIN rev_user_names AS b
        ON a.fullname_clean = b.fullname_clean
    """
)
con.sql(
    """
    CREATE OR REPLACE TEMP TABLE nts_long_scope AS
    SELECT a.*
    FROM nts_long AS a
    JOIN rev_user_names AS b
        ON a.fullname_clean = b.fullname_clean
    """
)

# Cleaning name matches (long on country) and blending full/first/last channels.
nanat_cols = set(nanats.columns)
nanat_full_col = "pred_nats_full" if "pred_nats_full" in nanat_cols else "pred_nats_name"
nanat_first_col = "pred_nats_first" if "pred_nats_first" in nanat_cols else None
nanat_last_col = "pred_nats_last" if "pred_nats_last" in nanat_cols else None
nanat_has_last_channel = 1 if nanat_last_col is not None else 0
if nanat_full_col != "pred_nats_full":
    print("Name2Nat full-name channel not found; falling back to legacy pred_nats_name.")
if nanat_first_col is None or nanat_last_col is None:
    print("Name2Nat first/last channels missing; blended nationality uses available channels only.")


def _nanat_long_query(pred_col, prob_alias):
    if pred_col is None:
        return (
            f"SELECT NULL::VARCHAR AS fullname_clean, NULL::VARCHAR AS nanat_country, "
            f"NULL::DOUBLE AS {prob_alias} WHERE FALSE"
        )
    source_sql = f"SELECT fullname_clean, UNNEST({pred_col}) FROM nanats_scope"
    return f"""
    SELECT fullname_clean, nanat_country, SUM(nanat_prob) AS {prob_alias}
    FROM (
        SELECT u.fullname_clean, COALESCE(cw.std_country, u.raw_eth) AS nanat_country, u.nanat_prob
        FROM (
            UNPIVOT ({source_sql})
            ON COLUMNS(* EXCLUDE (fullname_clean)) INTO NAME raw_eth VALUE nanat_prob
        ) AS u
        LEFT JOIN _country_cw AS cw ON u.raw_eth = cw.raw_country
    )
    WHERE nanat_country IS NOT NULL AND nanat_country != 'NA'
    GROUP BY fullname_clean, nanat_country
    """


anglo_country_sql = ", ".join([f"'{c}'" for c in ANGLO_COUNTRIES])
con.sql(
    f"""
    CREATE OR REPLACE TEMP TABLE nanats_long AS
    WITH nanats_full AS ({_nanat_long_query(nanat_full_col, "nanat_prob_full")}),
    nanats_first AS ({_nanat_long_query(nanat_first_col, "nanat_prob_first")}),
    nanats_last AS ({_nanat_long_query(nanat_last_col, "nanat_prob_last")}),
    nanats_merged AS (
        -- UNION ALL + GROUP BY avoids two expensive FULL JOINs across large (name x country) sets
        SELECT fullname_clean, nanat_country,
            SUM(nanat_prob_full)  AS nanat_prob_full,
            SUM(nanat_prob_first) AS nanat_prob_first,
            SUM(nanat_prob_last)  AS nanat_prob_last
        FROM (
            SELECT fullname_clean, nanat_country, nanat_prob_full, 0::DOUBLE AS nanat_prob_first, 0::DOUBLE AS nanat_prob_last FROM nanats_full
            UNION ALL
            SELECT fullname_clean, nanat_country, 0::DOUBLE AS nanat_prob_full, nanat_prob_first, 0::DOUBLE AS nanat_prob_last FROM nanats_first
            UNION ALL
            SELECT fullname_clean, nanat_country, 0::DOUBLE AS nanat_prob_full, 0::DOUBLE AS nanat_prob_first, nanat_prob_last FROM nanats_last
        )
        GROUP BY fullname_clean, nanat_country
    ),
    nanats_scored AS (
        SELECT *,
            SUM(CASE WHEN nanat_country IN ({anglo_country_sql}) THEN nanat_prob_full ELSE 0 END)
                OVER(PARTITION BY fullname_clean) AS anglo_prob_full,
            SUM(CASE WHEN nanat_country IN ({anglo_country_sql}) THEN nanat_prob_last ELSE 0 END)
                OVER(PARTITION BY fullname_clean) AS anglo_prob_last
        FROM nanats_merged
    ),
    nanats_weighted AS (
        SELECT *,
            CASE
                WHEN {nanat_has_last_channel} = 1 THEN GREATEST(0, anglo_prob_full - anglo_prob_last)
                ELSE 0
            END AS anglo_pressure,
            CASE WHEN (
                CASE
                    WHEN {nanat_has_last_channel} = 1 THEN GREATEST(0, anglo_prob_full - anglo_prob_last)
                    ELSE 0
                END
            ) > {NANAT_ANGLO_PRESSURE_CUTOFF}
                THEN {NANAT_W_CROWDED_FULL} ELSE {NANAT_W_DEFAULT_FULL} END AS w_full,
            CASE WHEN (
                CASE
                    WHEN {nanat_has_last_channel} = 1 THEN GREATEST(0, anglo_prob_full - anglo_prob_last)
                    ELSE 0
                END
            ) > {NANAT_ANGLO_PRESSURE_CUTOFF}
                THEN {NANAT_W_CROWDED_LAST} ELSE {NANAT_W_DEFAULT_LAST} END AS w_last,
            {NANAT_W_FIRST} AS w_first
        FROM nanats_scored
    ),
    nanats_blended AS (
        SELECT *,
            w_full*nanat_prob_full + w_last*nanat_prob_last + w_first*nanat_prob_first AS nanat_prob_raw
        FROM nanats_weighted
    )
    SELECT
        fullname_clean,
        nanat_country,
        nanat_prob_full,
        nanat_prob_first,
        nanat_prob_last,
        anglo_pressure,
        CASE WHEN anglo_pressure > {NANAT_ANGLO_PRESSURE_CUTOFF} THEN 1 ELSE 0 END AS nanat_anglo_crowding_ind,
        CASE WHEN SUM(nanat_prob_raw) OVER(PARTITION BY fullname_clean) > 0
            THEN nanat_prob_raw / SUM(nanat_prob_raw) OVER(PARTITION BY fullname_clean)
            ELSE 0 END AS nanat_prob
    FROM nanats_blended
    WHERE nanat_country IS NOT NULL AND nanat_country != 'NA'
    """
)

# Cleaning institution matches — materialized so DuckDB has statistics for the JOIN into rev_users_with_inst
con.sql(
"""
CREATE OR REPLACE TEMP TABLE inst_match_clean AS
    SELECT *,
        COUNT(*) OVER(PARTITION BY university_raw) AS n_country_match
    FROM inst_country_cw
    WHERE lower(REGEXP_REPLACE(university_raw, '[^A-z]', '', 'g')) NOT IN ('highschool', 'ged', 'unknown', 'invalid')
""")

# Merging users with institution matches (materialized — referenced twice: parquet write + inst_merge_long)
con.sql(
"""
CREATE OR REPLACE TEMP TABLE rev_users_with_inst AS
SELECT user_id, education_number, degree_clean, ed_startdate, ed_enddate,
    a.university_raw, match_country, matchscore, matchtype, hs_share,
-- trying to get earliest education for each country
    ROW_NUMBER() OVER(PARTITION BY user_id, match_country ORDER BY education_number) AS educ_order,
-- ID-ing if US high school (only exact matches)
    MAX(CASE WHEN 
        (degree_clean = 'High School' OR hs_share >= 0.5) AND 
        match_country = 'United States' AND 
        (matchtype = 'exact' OR (n_country_match = 1)) 
        THEN 1 ELSE 0 END) 
    OVER(PARTITION BY user_id) AS us_hs_exact,
-- ID-ing if any US education
    MAX(CASE WHEN degree_clean != 'Non-Degree' AND match_country = 'United States' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS us_educ,
-- ADE status 
    MAX(CASE WHEN degree_clean IN ('Master', 'Doctor', 'MBA') AND match_country = 'United States' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS ade_ind,
-- ADE year
    MAX(CASE WHEN degree_clean IN ('Master', 'MBA') AND match_country = 'United States' 
            THEN (CASE WHEN ed_enddate IS NULL AND ed_startdate IS NOT NULL THEN SUBSTRING(ed_startdate, 1, 4)::INT + 2 ELSE SUBSTRING(ed_enddate, 1, 4)::INT END) 
        WHEN degree_clean = 'Doctor' AND match_country = 'United States' 
            THEN (CASE WHEN ed_enddate IS NULL AND ed_startdate IS NOT NULL THEN SUBSTRING(ed_startdate, 1, 4)::INT + 4 ELSE SUBSTRING(ed_enddate, 1, 4)::INT END) 
        END) OVER(PARTITION BY user_id) AS ade_year,
-- Latest Degree Year
    MAX(CASE WHEN match_country = 'United States' AND ed_enddate IS NULL AND ed_startdate IS NOT NULL AND degree_clean NOT IN ('Master','MBA','Associate') THEN SUBSTRING(ed_startdate, 1, 4)::INT + 4 WHEN match_country = 'United States' AND ed_enddate IS NULL AND ed_startdate IS NOT NULL THEN SUBSTRING(ed_startdate, 1, 4)::INT + 2 WHEN match_country = 'United States' THEN SUBSTRING(ed_enddate, 1, 4)::INT ELSE NULL END) OVER(PARTITION BY user_id) AS last_grad_year
FROM 
    (SELECT *,
    -- Getting share of given institution labelled as high school
        (SUM(CASE WHEN degree_clean = 'High School' THEN 1 ELSE 0 END) OVER(PARTITION BY university_raw))/(SUM(CASE WHEN degree_clean != 'Missing' THEN 1 ELSE 0 END) OVER(PARTITION BY university_raw)) AS hs_share,
    FROM rev_users_filt) AS a 
JOIN inst_match_clean AS b 
ON a.university_raw = b.university_raw 
""")

# Saving intermediate user x inst x year data
if not test:
    print("Saving intermediate copy of users with institutions to file...")
    con.sql(f"COPY (SELECT * EXCLUDE (us_hs_exact, us_educ, ade_ind, ade_year)  FROM rev_users_with_inst) TO '{rcfg.REV_EDUC_LONG_PARQUET}'")

# Merging users with institution matches (long) and collapsing to user x country level
# Uses JOIN to _country_cw instead of Python UDF get_std_country; materialized for country_merge_long
con.sql(
"""
CREATE OR REPLACE TEMP TABLE inst_merge_long AS
WITH std AS (
    SELECT
        r.user_id, r.university_raw, r.hs_share, r.degree_clean,
        r.matchscore, r.matchtype, r.education_number,
        r.us_hs_exact, r.us_educ, r.ade_ind, r.ade_year, r.last_grad_year,
        COALESCE(cw.std_country, r.match_country) AS match_country
    FROM rev_users_with_inst AS r
    LEFT JOIN _country_cw AS cw ON r.match_country = cw.raw_country
    WHERE r.educ_order = 1 AND r.match_country != 'NA'
)
SELECT
    user_id, university_raw, match_country,
    us_hs_exact, us_educ, ade_ind, ade_year, last_grad_year,
    CASE WHEN degree_clean = 'High School' OR hs_share > 0.9 THEN matchscore
         WHEN degree_clean = 'Bachelor' THEN matchscore*0.8
         ELSE matchscore*0.5 END AS matchscore_corr,
    matchscore, matchtype, education_number,
    MAX(matchscore) OVER(PARTITION BY user_id, match_country) AS max_matchscore,
    COUNT(*) OVER(PARTITION BY user_id, match_country) AS n_match
FROM std
"""
)

# Merging users with name2nat matches (long on user x country) — materialized for country_merge_long
con.sql(
"""
CREATE OR REPLACE TEMP TABLE name_merge_long AS
SELECT
    user_id,
    a.fullname_clean,
    nanat_country,
    nanat_prob,
    nanat_prob_full,
    nanat_prob_first,
    nanat_prob_last,
    anglo_pressure,
    nanat_anglo_crowding_ind
FROM
    (SELECT fullname_clean, user_id FROM rev_users_filt GROUP BY fullname_clean, user_id) AS a
    JOIN
    nanats_long AS b
    ON a.fullname_clean = b.fullname_clean
"""
)

# Merging users with nametrace matches (long on user x subregion) — materialized for country_merge_long
con.sql("""
CREATE OR REPLACE TEMP TABLE nt_merge_long AS
SELECT user_id, a.fullname_clean, region, prob, f_prob_nt
FROM (SELECT fullname_clean, user_id FROM rev_users_filt GROUP BY fullname_clean, user_id) AS a
    JOIN
    nts_long_scope AS b
    ON a.fullname_clean = b.fullname_clean
""")

# combining all country matches — materialized; get_country_subregion UDFs replaced with JOIN to _subregion_cw
con.sql(
"""
CREATE OR REPLACE TEMP TABLE country_merge_long AS
SELECT
    CASE WHEN countries.user_id IS NULL THEN nt.user_id ELSE countries.user_id END AS user_id,
    MAX(CASE WHEN countries.fullname_clean IS NULL THEN nt.fullname_clean ELSE countries.fullname_clean END)
        OVER(PARTITION BY CASE WHEN countries.user_id IS NULL THEN nt.user_id ELSE countries.user_id END) AS fullname_clean,
    country,
    CASE WHEN countries.subregion IS NULL THEN nt.region ELSE countries.subregion END AS subregion,
    CASE WHEN countries.nanat_score IS NULL THEN 0 ELSE countries.nanat_score END AS nanat_score,
    CASE WHEN countries.nanat_prob_full IS NULL THEN 0 ELSE countries.nanat_prob_full END AS nanat_prob_full,
    CASE WHEN countries.nanat_prob_first IS NULL THEN 0 ELSE countries.nanat_prob_first END AS nanat_prob_first,
    CASE WHEN countries.nanat_prob_last IS NULL THEN 0 ELSE countries.nanat_prob_last END AS nanat_prob_last,
    CASE WHEN countries.anglo_pressure IS NULL THEN 0 ELSE countries.anglo_pressure END AS anglo_pressure,
    CASE WHEN countries.nanat_anglo_crowding_ind IS NULL THEN 0 ELSE countries.nanat_anglo_crowding_ind END AS nanat_anglo_crowding_ind,
    CASE WHEN countries.inst_score IS NULL THEN 0 ELSE countries.inst_score END AS inst_score,
    MAX(f_prob_nt) OVER(PARTITION BY CASE WHEN countries.user_id IS NULL THEN nt.user_id ELSE countries.user_id END) AS f_prob_nt,
    SUM(CASE WHEN countries.nanat_score IS NULL THEN 0 ELSE countries.nanat_score END)
        OVER(PARTITION BY CASE WHEN countries.user_id IS NULL THEN nt.user_id ELSE countries.user_id END,
                          CASE WHEN countries.subregion IS NULL THEN nt.region ELSE countries.subregion END) AS nanat_subregion_score,
    CASE WHEN prob IS NULL THEN 0 ELSE prob END AS nt_subregion_score,
    university_raw,
    us_hs_exact,
    us_educ,
    ade_ind,
    ade_year,
    last_grad_year
FROM (
    SELECT
        CASE WHEN a.user_id IS NULL THEN b.user_id ELSE a.user_id END AS user_id,
        b.fullname_clean AS fullname_clean,
        university_raw,
        CASE WHEN a.match_country IS NULL THEN b.nanat_country ELSE a.match_country END AS country,
        CASE WHEN a.match_country IS NULL
             THEN COALESCE(sub_n.subregion, b.nanat_country)
             ELSE COALESCE(sub_m.subregion, a.match_country)
        END AS subregion,
        CASE WHEN a.match_country IS NULL THEN 0 ELSE matchscore_corr END AS inst_score,
        CASE WHEN b.nanat_country IS NULL THEN 0 ELSE nanat_prob END AS nanat_score,
        CASE WHEN b.nanat_country IS NULL THEN 0 ELSE nanat_prob_full END AS nanat_prob_full,
        CASE WHEN b.nanat_country IS NULL THEN 0 ELSE nanat_prob_first END AS nanat_prob_first,
        CASE WHEN b.nanat_country IS NULL THEN 0 ELSE nanat_prob_last END AS nanat_prob_last,
        CASE WHEN b.nanat_country IS NULL THEN 0 ELSE anglo_pressure END AS anglo_pressure,
        CASE WHEN b.nanat_country IS NULL THEN 0 ELSE nanat_anglo_crowding_ind END AS nanat_anglo_crowding_ind,
        us_hs_exact,
        us_educ,
        ade_ind,
        ade_year,
        last_grad_year
    FROM inst_merge_long AS a
    FULL JOIN name_merge_long AS b
        ON a.user_id = b.user_id AND a.match_country = b.nanat_country
    LEFT JOIN _subregion_cw AS sub_n ON b.nanat_country = sub_n.std_country
    LEFT JOIN _subregion_cw AS sub_m ON a.match_country = sub_m.std_country
) AS countries
FULL JOIN nt_merge_long AS nt
ON countries.user_id = nt.user_id AND countries.subregion = nt.region
""")

print("Done!")

#####################################
### CLEANING POSITION DATA
#####################################
print("Cleaning Revelio Position Data...")

# testing
if test:
    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE merged_pos_scope_test AS
        SELECT *
        FROM merged_pos_scope
        WHERE user_id IN ({ids}) OR user_id = {test_user}
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP TABLE merged_pos_scope AS
        SELECT *
        FROM merged_pos_scope_test
        """
    )

# # TODO:Cleaning position history -- (impute rcid if company name similar to below,) get rid of duplicates -- also need to clean duplicate users!

# Pre-materialize null-enddate positions so next_pos and dup_pos each get one scan (not two)
con.sql(
"""
CREATE OR REPLACE TEMP TABLE _pos_null AS
SELECT user_id, position_number, startdate, enddate, company_raw
FROM merged_pos_scope
WHERE enddate IS NULL
"""
)

# Scope join target to only users with null-enddate positions — avoids scanning all of merged_pos_scope
con.sql(
"""
CREATE OR REPLACE TEMP TABLE _pos_null_user_scope AS
SELECT user_id, position_number, startdate, company_raw
FROM merged_pos_scope
WHERE user_id IN (SELECT user_id FROM _pos_null)
"""
)

con.sql(
"""
    CREATE OR REPLACE TEMP TABLE pos_with_null_enddates AS
    WITH next_pos AS (
        SELECT
            a.user_id,
            a.position_number,
            arg_min(b.startdate, b.position_number) AS alt_enddate
        FROM _pos_null AS a
        LEFT JOIN _pos_null_user_scope AS b
            ON a.user_id = b.user_id
           AND a.position_number != b.position_number
           AND b.startdate::DATE >= a.startdate::DATE + INTERVAL '6' MONTH
        GROUP BY a.user_id, a.position_number
    ),
    dup_pos AS (
        SELECT
            a.user_id,
            a.position_number,
            1 AS pos_dup_ind
        FROM _pos_null AS a
        JOIN _pos_null_user_scope AS b
            ON a.user_id = b.user_id
           AND a.position_number != b.position_number
           AND a.startdate < b.startdate
           AND DATEDIFF('month', a.startdate::DATETIME, b.startdate::DATETIME) <= 3
           AND LEFT(UPPER(REGEXP_REPLACE(a.company_raw, '[^A-Za-z0-9]', '', 'g')), 1)
             = LEFT(UPPER(REGEXP_REPLACE(b.company_raw, '[^A-Za-z0-9]', '', 'g')), 1)
           AND jaro_winkler_similarity(a.company_raw, b.company_raw) >= 0.85
        GROUP BY a.user_id, a.position_number
    )
    SELECT
        a.user_id,
        a.position_number,
        a.startdate,
        a.enddate,
        n.alt_enddate,
        COALESCE(d.pos_dup_ind, 0) AS pos_dup_ind
    FROM _pos_null AS a
    LEFT JOIN next_pos AS n
        ON a.user_id = n.user_id AND a.position_number = n.position_number
    LEFT JOIN dup_pos AS d
        ON a.user_id = d.user_id AND a.position_number = d.position_number
"""
)


con.sql(
    f"""
    CREATE OR REPLACE TEMP TABLE merged_pos_clean AS
    SELECT *,
        CASE WHEN max_share_foia > 0 THEN 1 ELSE 0 END AS foia_occ_ind
    FROM merged_pos_scope AS a
    LEFT JOIN (
        SELECT user_id, position_number, alt_enddate, pos_dup_ind
        FROM pos_with_null_enddates
    ) AS b
        ON a.user_id = b.user_id AND a.position_number = b.position_number
    LEFT JOIN occ_cw AS c
        ON a.{role_col} = c.{role_col}
"""
)

if not test:
    print("Saving cleaned positions to file...")
    con.sql(f"COPY merged_pos_clean TO '{rcfg.MERGED_POS_CLEAN_PARQUET}'")

# 8/6/25: what to do with null enddates? if last position, can impute as updated dt, but if not last position? if i take next start date, this will cut out freelance/part-time work or work preceding freelance/part-time work (e.g. if i have a FT job and start doing something on the side and hold both jobs concurrently)
# okay for now just leave as today if null (most conservative -- downside is that enddatediff filter will be less effective)


# User-level indicator for H-1B occupation — materialized so DuckDB has statistics for the JOIN in user_merge
con.sql("CREATE OR REPLACE TEMP TABLE merged_pos_cw AS SELECT user_id, MAX(foia_occ_ind) AS foia_occ_ind, MIN(min_rank) AS min_h1b_occ_rank FROM merged_pos_clean GROUP BY user_id")

# Cleaning position history and aggregating to user level — materialized (regex-heavy; needed for rev_indiv LEFT JOIN)
con.sql(f"""
    CREATE OR REPLACE TEMP TABLE merged_pos_user AS
    SELECT user_id,
        ARRAY_AGG(title_clean ORDER BY position_number) AS positions,
        ARRAY_AGG(rcid ORDER BY position_number) AS rcids,
        MIN(CASE WHEN position_number > max_intern_position THEN startdate ELSE NULL END) AS min_startdate,
        MIN(CASE WHEN position_number > max_intern_position AND country = 'United States' THEN startdate ELSE NULL END) AS min_startdate_us
    FROM (
        SELECT *, MAX(CASE WHEN intern_ind = 1 THEN position_number ELSE 0 END) OVER(PARTITION BY user_id) AS max_intern_position FROM (
            SELECT user_id, {help.inst_clean_regex_sql('title_raw')} AS title_clean, position_number, rcid, country, startdate, enddate, {role_col} AS role_k_selected,
                CASE WHEN
                    (lower(title_raw) ~ '(^|\\s)(intern)($|\\s)' AND
                        (enddate IS NULL OR DATEDIFF('month', startdate::DATETIME, enddate::DATETIME) < 12))
                    OR (lower(title_raw) ~ '(^|\\s)(student)($|\\s)')
                THEN 1 ELSE 0 END AS intern_ind
            FROM merged_pos_scope
        )
    ) GROUP BY user_id
    """
    )

print("Done!")

#####################################
### GETTING AND EXPORTING FINAL USER FILE
#####################################
print("Final merge...")
con.sql(
f"""
CREATE OR REPLACE TEMP TABLE user_merge AS
SELECT * FROM (
    SELECT a.user_id, est_yob, hs_ind, valid_postsec, updated_dt, f_prob, stem_ind, f_prob_nt, fullname, university_raw, country, subregion, inst_score, nanat_score, nanat_subregion_score, nt_subregion_score, 0.5*inst_score + 0.5*nanat_score AS total_score, 0.5*inst_score + 0.25*nanat_subregion_score + 0.25*nt_subregion_score AS total_subregion_score,
        MAX(COALESCE(nanat_anglo_crowding_ind, 0)) OVER(PARTITION BY a.user_id) AS country_uncertain_ind,
        MAX(COALESCE(anglo_pressure, 0)) OVER(PARTITION BY a.user_id) AS max_anglo_pressure,
        MAX(us_hs_exact) OVER(PARTITION BY a.user_id) AS us_hs_exact, 
        MAX(us_educ) OVER(PARTITION BY a.user_id) AS us_educ,
        MAX(CASE WHEN ade_ind IS NULL THEN 0 ELSE ade_ind END) OVER(PARTITION BY a.user_id) AS ade_ind,
        MIN(ade_year) OVER(PARTITION BY a.user_id) AS ade_year,
        MIN(last_grad_year) OVER(PARTITION BY a.user_id) AS last_grad_year,
        MAX(0.5*inst_score + 0.5*nanat_score) OVER(PARTITION BY a.user_id) AS max_total_score,
        MAX(CASE WHEN country = 'United States' THEN 0 ELSE 0.5*inst_score + 0.5*nanat_score END) OVER(PARTITION BY a.user_id) AS max_total_score_nonus,
        foia_occ_ind, min_h1b_occ_rank, fields, highest_ed_level

    -- taking original revelio user data, collapsing to user level and creating necessary variables
    FROM (
        SELECT user_id, est_yob, f_prob, fullname, hs_ind, valid_postsec, updated_dt, MAX(stem_ind_postsec) AS stem_ind, ARRAY_AGG(field_clean) FILTER (WHERE field_clean IS NOT NULL) AS fields,
        CASE WHEN MAX(CASE WHEN degree_clean = 'Doctor' THEN 1 ELSE 0 END) = 1 THEN 'Doctor'
            WHEN MAX(CASE WHEN degree_clean IN ('Master', 'MBA') THEN 1 ELSE 0 END) = 1 THEN 'Master'
            WHEN MAX(CASE WHEN degree_clean IN ('Bachelor') THEN 1 ELSE 0 END) = 1 THEN 'Bachelor'
            WHEN MAX(CASE WHEN degree_clean IS NOT NULL THEN 1 ELSE 0 END) = 1 THEN 'Other'
            ELSE NULL END AS highest_ed_level,
        -- profile-level location fields (user-level, constant within user)
        ANY_VALUE(user_location) AS user_location,
        ANY_VALUE(user_country) AS user_country
        FROM (
            SELECT user_id, CASE WHEN degree_clean IN ('Non-Degree', 'High School', 'Associate') THEN 0 ELSE stem_ind END AS stem_ind_postsec,
                {help.get_est_yob()} AS est_yob,
                MAX(CASE WHEN degree_clean = 'High School' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS hs_ind,
                MAX(CASE WHEN degree_clean NOT IN ('Non-Degree', 'Master', 'Doctor', 'MBA') AND (ed_enddate IS NOT NULL OR ed_startdate IS NOT NULL) THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS valid_postsec, updated_dt, field_clean, degree_clean,
                f_prob, fullname, user_location, user_country FROM rev_users_filt
        ) GROUP BY user_id, est_yob, f_prob, fullname, hs_ind, valid_postsec, updated_dt
    ) AS a 

    -- left joining on user id with user x country level cleaned data from above
    LEFT JOIN country_merge_long AS b ON a.user_id = b.user_id

    -- left joining on user id with user-level position information from above
    LEFT JOIN merged_pos_cw AS c ON a.user_id = c.user_id
    )

-- filtering on country guess quality 
WHERE (max_total_score_nonus < 0.3 OR max_total_score_nonus = total_score) AND (total_score >= 0.01 OR nanat_subregion_score + nt_subregion_score >= 0.05)
""")

# # saving intermediate version
# con.sql(f"COPY final_user_merge TO '{root}/data/int/rev_users_clean_jul28.parquet'")

# final_user_merge = con.read_parquet(f'{root}/data/int/rev_users_clean_jul28.parquet')

## DUPLICATE PROFILE DEDUP
# Within (fullname_clean, rcid) groups that have multiple user_ids, keep the one with
# the most positions (proxy for most complete profile); tiebreak: latest updated_dt, then
# smallest user_id. A user is dropped only if it is non-canonical in some group and
# canonical in none (avoids incorrectly dropping users who share a name but differ elsewhere).
# Date-overlap guard: only treat two profiles as duplicates if their position date ranges
# at the shared rcid overlap (within a 1-year buffer for LinkedIn imprecision) OR their
# overall education date ranges overlap — this filters out two genuinely different people
# who happen to share a name and work at the same company at different times.
dup_profile_dedup = rcfg.BUILD_DUP_PROFILE_DEDUP
print(f"dup_profile_dedup: {dup_profile_dedup}")
if dup_profile_dedup:
    con.sql("""
    CREATE OR REPLACE TEMP TABLE dup_user_drop AS
    WITH in_scope AS (
        SELECT DISTINCT user_id FROM user_merge
    ),
    user_completeness AS (
        SELECT p.user_id,
            COUNT(*) AS n_positions,
            MAX(r.updated_dt) AS updated_dt
        FROM merged_pos_clean AS p
        JOIN (SELECT user_id, MAX(updated_dt) AS updated_dt FROM rev_users_filt GROUP BY user_id) AS r
            ON p.user_id = r.user_id
        JOIN in_scope AS s ON p.user_id = s.user_id
        GROUP BY p.user_id
    ),
    user_name AS (
        SELECT user_id,
               ANY_VALUE(fullname_clean) AS fullname_clean,
               ANY_VALUE(fullname)       AS fullname_raw   -- raw, for single-word gate
        FROM rev_users_filt
        WHERE fullname_clean IS NOT NULL AND LENGTH(TRIM(fullname_clean)) > 0
        GROUP BY user_id
    ),
    -- Position date range per (user_id, rcid)
    user_rcid_dates AS (
        SELECT user_id, rcid,
            MIN(TRY_CAST(startdate AS DATE)) AS pos_start,
            MAX(CASE
                WHEN alt_enddate IS NULL AND enddate IS NULL THEN CURRENT_DATE
                WHEN enddate IS NULL THEN TRY_CAST(alt_enddate AS DATE)
                ELSE TRY_CAST(enddate AS DATE)
            END) AS pos_end
        FROM merged_pos_clean
        WHERE rcid IS NOT NULL
        GROUP BY user_id, rcid
    ),
    -- Education date range per user (across all education records)
    user_educ_dates AS (
        SELECT user_id,
            MIN(TRY_CAST(ed_startdate AS DATE)) AS educ_start,
            MAX(COALESCE(TRY_CAST(ed_enddate AS DATE), TRY_CAST(ed_startdate AS DATE))) AS educ_end
        FROM rev_users_filt
        WHERE ed_startdate IS NOT NULL OR ed_enddate IS NOT NULL
        GROUP BY user_id
    ),
    -- Distinct normalized school names per user
    user_schools AS (
        SELECT DISTINCT user_id, LOWER(TRIM(university_raw)) AS school
        FROM rev_users_filt
        WHERE university_raw IS NOT NULL AND LENGTH(TRIM(university_raw)) > 0
    ),
    -- All (fullname_clean, rcid) groups with 2+ in-scope users — GROUP BY only, no self-join.
    -- Gate: single-word fullname_clean values (last initial stripped by regex) must also match
    -- on the raw fullname to avoid grouping all "Christopher"s at the same company as duplicates.
    dup_groups_raw AS (
        SELECT n.fullname_clean, da.rcid
        FROM user_name AS n
        JOIN in_scope AS s ON n.user_id = s.user_id
        JOIN user_rcid_dates AS da ON n.user_id = da.user_id
        GROUP BY n.fullname_clean, da.rcid
        HAVING COUNT(DISTINCT n.user_id) > 1
           AND (
               n.fullname_clean LIKE '% %'
               OR COUNT(DISTINCT LOWER(TRIM(n.fullname_raw))) = 1
           )
    ),
    -- Position date overlap: sort each group by pos_start and check if LAG(pos_end) >= pos_start.
    -- Detecting any pairwise overlap via a sorted sweep is O(n log n) with no self-join.
    -- (If any two intervals overlap, consecutive ones in sorted order must also overlap.)
    pos_overlap_groups AS (
        SELECT DISTINCT fullname_clean, rcid
        FROM (
            SELECT n.fullname_clean, da.rcid, da.pos_start,
                LAG(da.pos_end) OVER (
                    PARTITION BY n.fullname_clean, da.rcid ORDER BY da.pos_start
                ) AS prev_pos_end
            FROM user_name AS n
            JOIN in_scope AS s ON n.user_id = s.user_id
            JOIN user_rcid_dates AS da ON n.user_id = da.user_id
            JOIN dup_groups_raw AS dg
                ON n.fullname_clean = dg.fullname_clean AND da.rcid = dg.rcid
        )
        WHERE prev_pos_end IS NOT NULL
          AND pos_start <= prev_pos_end + INTERVAL '1 year'
    ),
    -- Education date overlap: same LAG sweep per group.
    educ_overlap_groups AS (
        SELECT DISTINCT fullname_clean, rcid
        FROM (
            SELECT n.fullname_clean, da.rcid, ea.educ_start,
                LAG(ea.educ_end) OVER (
                    PARTITION BY n.fullname_clean, da.rcid ORDER BY ea.educ_start
                ) AS prev_educ_end
            FROM user_name AS n
            JOIN in_scope AS s ON n.user_id = s.user_id
            JOIN user_rcid_dates AS da ON n.user_id = da.user_id
            JOIN user_educ_dates AS ea ON n.user_id = ea.user_id
            JOIN dup_groups_raw AS dg
                ON n.fullname_clean = dg.fullname_clean AND da.rcid = dg.rcid
            WHERE ea.educ_start IS NOT NULL AND ea.educ_end IS NOT NULL
        )
        WHERE prev_educ_end IS NOT NULL
          AND educ_start <= prev_educ_end + INTERVAL '1 year'
    ),
    -- School name overlap: groups where 2+ users share a school.
    -- COUNT OVER (PARTITION BY group, school) detects shared schools with no self-join.
    school_overlap_groups AS (
        SELECT DISTINCT fullname_clean, rcid
        FROM (
            SELECT n.fullname_clean, da.rcid,
                COUNT(DISTINCT n.user_id) OVER (
                    PARTITION BY n.fullname_clean, da.rcid, us.school
                ) AS n_users_with_school
            FROM user_name AS n
            JOIN in_scope AS s ON n.user_id = s.user_id
            JOIN user_rcid_dates AS da ON n.user_id = da.user_id
            JOIN user_schools AS us ON n.user_id = us.user_id
            JOIN dup_groups_raw AS dg
                ON n.fullname_clean = dg.fullname_clean AND da.rcid = dg.rcid
        )
        WHERE n_users_with_school > 1
    ),
    -- Confirmed duplicate groups: pass at least one overlap check.
    confirmed_groups AS (
        SELECT fullname_clean, rcid FROM pos_overlap_groups
        UNION
        SELECT fullname_clean, rcid FROM educ_overlap_groups
        UNION
        SELECT fullname_clean, rcid FROM school_overlap_groups
    ),
    -- Rank users within each confirmed group; keep the most complete profile.
    ranked AS (
        SELECT n.user_id, cg.fullname_clean, cg.rcid,
            ROW_NUMBER() OVER (
                PARTITION BY cg.fullname_clean, cg.rcid
                ORDER BY COALESCE(c.n_positions, 0) DESC, c.updated_dt DESC NULLS LAST, n.user_id ASC
            ) AS dup_rank
        FROM confirmed_groups AS cg
        JOIN user_name AS n ON cg.fullname_clean = n.fullname_clean
        JOIN in_scope AS s ON n.user_id = s.user_id
        JOIN user_rcid_dates AS da ON n.user_id = da.user_id AND cg.rcid = da.rcid
        LEFT JOIN user_completeness AS c ON n.user_id = c.user_id
    ),
    non_canonical AS (SELECT DISTINCT user_id FROM ranked WHERE dup_rank > 1),
    canonical     AS (SELECT DISTINCT user_id FROM ranked WHERE dup_rank = 1)
    -- only drop users that are never the canonical in any group
    SELECT user_id FROM non_canonical WHERE user_id NOT IN (SELECT user_id FROM canonical)
    """)
    n_dup = con.sql("SELECT COUNT(*) FROM dup_user_drop").fetchone()[0]
    print(f"  Duplicate profiles identified for removal: {n_dup}")
else:
    con.sql("CREATE OR REPLACE TEMP TABLE dup_user_drop AS SELECT NULL::DOUBLE AS user_id WHERE FALSE")

## FURTHER COLLAPSING
# cleaning revelio data (collapsing to user x company x country level)
rev_indiv = con.sql(
"""
SELECT * EXCLUDE (users.user_id, poshist.user_id) FROM
-- start with user x rcid data (get rcid-specific startdate)
(SELECT user_id, a.rcid, first_startdate, last_enddate, b.foia_firm_uid, b.llm_match_score FROM
    (SELECT user_id,
        MIN(startdate)::DATETIME AS first_startdate,
        MAX(CASE WHEN alt_enddate IS NULL AND enddate IS NULL THEN STRFTIME(CURRENT_DATE, '%Y-%m-%d') WHEN enddate IS NULL AND alt_enddate IS NOT NULL THEN alt_enddate ELSE enddate END)::DATETIME AS last_enddate, rcid
    FROM merged_pos_clean WHERE country = 'United States' AND startdate >= '2015-01-01' GROUP BY user_id, rcid) AS a
    JOIN
    (SELECT rcid, foia_firm_uid, MAX(llm_match_score) AS llm_match_score FROM samp_to_rcid GROUP BY rcid, foia_firm_uid) AS b
    ON a.rcid = b.rcid
) AS pos

-- joining on user id with final user merge data from above, filtered on OPT-likely
JOIN
(SELECT * FROM user_merge WHERE (us_hs_exact IS NULL OR us_hs_exact = 0) AND (us_educ IS NULL OR us_educ = 1)
    AND user_id NOT IN (SELECT user_id FROM dup_user_drop)) AS users
ON pos.user_id = users.user_id

-- left joining on user id with user-level position history data
LEFT JOIN
merged_pos_user AS poshist
ON pos.user_id = poshist.user_id
""")

if not test:
    print("Saving full rev indiv file...")
    con.sql(f"COPY rev_indiv TO '{rcfg.REV_INDIV_PARQUET}'")

print('Done!')
