# File Description: Reading and Cleaning FOIA F-1 Student Data

# Imports and Paths
import duckdb as ddb
import pandas as pd
import sys 
import os 
import time
import pyarrow as pa
import pyarrow.parquet as pq
import re 
# Ensure progress logs flush immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 
from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
from external_f1_employer_matching import (
    build_preferred_rcid_activity,
    stage_external_f1_crosswalk,
)

#########################
## DECLARING CONSTANTS ##
#########################
INT_FOLDER = f"{root}/data/int/int_files_nov2025"

DATE_COLS = {
    'authorization_end_date','authorization_start_date','date_of_birth',
    'first_entry_date','last_departure_date','last_entry_date',
    'opt_authorization_end_date','opt_authorization_start_date',
    'opt_employer_end_date','opt_employer_start_date',
    'program_end_date','program_start_date',
    'visa_expiration_date','visa_issue_date',
}

STATE_NAME_TO_ABBR = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "washington dc": "DC",
    "dc": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "puerto rico": "PR",
    "guam": "GU",
    "american samoa": "AS",
    "northern mariana islands": "MP",
    "us virgin islands": "VI",
}

US_COUNTRY_CODES = {
    "US",
    "USA",
    "UNITED STATES",
    "UNITED STATES OF AMERICA",
}

COMMON_CITY_NAMES = [
    "new york",
    "new york city",
    "san francisco",
    "los angeles",
    "chicago",
    "houston",
    "dallas",
    "atlanta",
    "seattle",
    "washington",
]
COMMON_CITY_LIST = ", ".join([f"'{city}'" for city in COMMON_CITY_NAMES])

REMATCH_SAMPLE_SIZE = None
REMATCH_JW_THRESHOLD = 0.95

ENTITY_NAME_BLOCK_PREFIX_LEN = 3
FUZZY_NAME_BLOCK_PREFIX_LEN = 3

F1_INST_CROSSWALK_PATH = f"{INT_FOLDER}/f1_inst_unitid_crosswalk.parquet"
F1_EMPLOYER_FINAL_CROSSWALK_PATH = f"{INT_FOLDER}/f1_employer_final_crosswalk.parquet"
F1_EMPLOYER_ENTITY_MAPPING_PATH = f"{INT_FOLDER}/f1_employer_entity_mapping.parquet"
F1_EMPLOYER_AUTH_COUNTS_PATH = f"{INT_FOLDER}/f1_employer_auth_counts.parquet"
F1_PREFERRED_RCID_LIST_PATH = f"{INT_FOLDER}/f1_preferred_rcids_multi_year.parquet"
F1_INST_LOAD_FROM_CACHE = True
F1_INST_SAVE_TO_CACHE = False 
F1_EMPLOYER_LOAD_FROM_CACHE = True 
F1_EMPLOYER_SAVE_TO_CACHE = False

INDIVIDUAL_ID_COLS = ["individual_key", "student_key", "student_id", "individual_id"]
AUTH_START_COLS = ["authorization_start_date", "opt_authorization_start_date", "opt_employer_start_date"]

##################################
## FUNCTIONS FOR IMPORTING DATA ##
##################################
def _import_all_data(con, test = False, verbose = False):
    """
    Import all relevant data files into DuckDB.
    """
    # FOIA SEVP Data
    foiatab = _read_foia_all(con, test=test, verbose=verbose)

    # Revelio Company Data
    revcompanytab = _read_revelio_companies(con, verbose=verbose)

    # IPEDS Crosswalk Data
    ipedsnamestab = _read_ipeds_crosswalk(con, verbose=verbose)

    return {
        'foia_sevp': foiatab,
        'revelio_companies': revcompanytab,
        'ipeds_crosswalk': ipedsnamestab,
    }

def _read_foia_all(con, test=False,verbose = False):
    """
    Read all FOIA F-1 data files from all year directories, concatenate and save into a DuckDB relation.
    """
    # Check if combined parquet already exists
    combined_parquet_path = f"{INT_FOLDER}/foia_sevp_combined_raw.parquet"
   
    # If path exists, read in combined parquet
    if os.path.exists(combined_parquet_path):
        con.sql(f"""
        CREATE OR REPLACE TABLE allyrs_raw AS
        SELECT *
        FROM read_parquet('{combined_parquet_path}')
        """)
        if verbose:
            print("✅ Loaded combined F-1 data as 'allyrs_raw' from Parquet.")
        return True 
    
    # If not, proceed to combine yearly data
    if verbose:
        print("🔄 Combining yearly FOIA SEVP data into one table...")
      
    # Iterate through each year's directory
    first_parquet = True
    for dirname in sorted(os.listdir(f"{root}/data/raw/foia_sevp"), reverse = True):
        if not os.path.isdir(f"{root}/data/raw/foia_sevp/{dirname}") or dirname == "J-1":
            continue
        if test and dirname != "2022":
            continue
        
        # Check if merged parquet for the year exists
        path = f'{root}/data/raw/foia_sevp/{dirname}/merged{dirname}.parquet'

        # If does not exist, create from scratch
        if not os.path.exists(path):  
            if verbose:
                print(f"🔄 Merged parquet for {dirname} not found. Creating from raw files...")
            _read_foia_raw(dirname, test=test, verbose=verbose)

        # Read in the year's merged parquet into duckdb safely
        if first_parquet:
            base_cols = None
        base_cols = _read_dir_parquet(con, path, first_parquet, base_cols, verbose)
        first_parquet = False 

    # Export combined data to Parquet
    con.sql(f"""COPY allyrs_raw TO '{combined_parquet_path}' (FORMAT PARQUET)""")
    if verbose:
        print("✅ Exported combined data to Parquet.")
    return "allyrs_raw"

def _read_foia_raw(dirname, test=False, verbose = False):
    # reading in raw data files and concatenating each year into one parquet
    t1 = time.time()
    yr_raw = []
    if not os.path.isdir(f"{root}/data/raw/foia_sevp/{dirname}") and dirname != "J-1":
        return None 
    if test and dirname != "2013":
        return None
    for filename in os.listdir(f"{root}/data/raw/foia_sevp/{dirname}"):
        if not filename[0].isalpha():
            continue
        print("Reading in: ", f"{dirname}/{filename}")
        df_temp = pd.read_excel(f"{root}/data/raw/foia_sevp/{dirname}/{filename}").assign(filename = filename)
        yr_raw.append(df_temp)

    # combining all files in the year into one dataframe
    df_yr = pd.concat(yr_raw, ignore_index=True).assign(year = dirname)
    
    # renaming columns to convert spaces to underscores
    df_yr.columns = [col.lower().replace(" ", "_") for col in df_yr.columns]

    # across all columns, convert "b(6)" to NaN
    df_yr = df_yr.replace("b(6),b(7)(c)", pd.NA)

    # convert all object columns to string (for mixed-type safety)
    df_yr = df_yr.astype({col: "string" for col in df_yr.select_dtypes(include="object").columns})

    # writing out combined year data (exclude birth date columns)
    df_yr[df_yr.columns.difference(['birth_date'])].to_parquet(f"{root}/data/raw/foia_sevp/{dirname}/merged{dirname}.parquet", index=False)
    
    t2 = time.time()
    if verbose:
        print(f"Finished writing {dirname} data ({(t2-t1)/60:.2f} minutes)")
    
    return None
        
def _read_dir_parquet(con, path: str, first_parquet: bool, base_cols, verbose: bool):
    if verbose:
        print(f"🔄 Reading in merged parquet: {path}")
    
    # Read schema for comparison
    pq_file = pq.ParquetFile(path)
    orig_cols = pq_file.schema.names
    safe_cols = [re.sub(r'[^0-9a-zA-Z_]', '', c) for c in orig_cols]

    # for first parquet, create table directly
    if first_parquet:
        col_rename_expr = ", ".join(
            [build_typed_expr(f'"{orig}"', safe) for orig, safe in zip(orig_cols, safe_cols)]
        )

        con.sql(f"""
            CREATE OR REPLACE TABLE allyrs_raw AS
            SELECT {col_rename_expr}
            FROM read_parquet('{path}')
        """)

        print(f"✅ Created table from {path} with {len(safe_cols)} columns.")
        return safe_cols 
    
    else:
        if base_cols is None:
            raise ValueError("base_cols must be provided for non-first parquet files.")
        # Compare columns to base schema
        missing_in_new = [c for c in base_cols if c not in safe_cols]
        extra_in_new = [c for c in safe_cols if c not in base_cols]

        if missing_in_new or extra_in_new:
            print(f"⚠️ Column mismatch in {path}:")
            if missing_in_new:
                print(f"   Missing in new file: {missing_in_new}")
            if extra_in_new:
                print(f"   Extra in new file: {extra_in_new}")

        # Build aligned select
        select_exprs = []
        for c in base_cols:
            if c in safe_cols:
                orig = orig_cols[safe_cols.index(c)]
                select_exprs.append(build_typed_expr(f'"{orig}"', c))
            elif f'{c}_x' in safe_cols:
                orig = orig_cols[safe_cols.index(f'{c}_x')]
                select_exprs.append(build_typed_expr(f'"{orig}"', c))
            else:
                select_exprs.append(build_typed_null(c))

        select_clause = ", ".join(select_exprs)

        con.sql(f"""
            INSERT INTO allyrs_raw
            SELECT {select_clause}
            FROM read_parquet('{path}')
        """)

        print(f"✅ Appended {path} ({len(safe_cols)} cols).")
        return base_cols

def _date_parse_sql(col_sql: str) -> str:
    fmt_try = f"""
        COALESCE(
          try_strptime(CAST({col_sql} AS VARCHAR), '%Y-%m-%d %H:%M:%S'),
          try_strptime(CAST({col_sql} AS VARCHAR), '%Y-%m-%d'),
          try_strptime(CAST({col_sql} AS VARCHAR), '%m/%d/%Y %H:%M:%S'),
          try_strptime(CAST({col_sql} AS VARCHAR), '%m/%d/%Y %H:%M'),
          try_strptime(CAST({col_sql} AS VARCHAR), '%m/%d/%Y')
        )
    """

    inner = f"""
        COALESCE(
          -- Already a TIMESTAMP_*? Keep it.
          CASE
            WHEN typeof({col_sql}) LIKE 'TIMESTAMP%' THEN CAST({col_sql} AS TIMESTAMP)
            ELSE NULL
          END,
          -- String parse
          CASE
            WHEN typeof({col_sql}) = 'VARCHAR' THEN {fmt_try}
            ELSE NULL
          END,
          -- Excel serial days (numeric) => DATE + INTEGER => TIMESTAMP
          CASE
            WHEN typeof({col_sql}) IN ('DOUBLE','DECIMAL','HUGEINT','BIGINT','INTEGER') THEN
              CAST(DATE '1899-12-30' + CAST({col_sql} AS INTEGER) AS TIMESTAMP)
              -- alternatively: CAST((DATE '1899-12-30') + MAKE_INTERVAL(days := CAST({col_sql} AS INTEGER)) AS TIMESTAMP)
            ELSE NULL
          END
        )
    """

    guarded = f"""
        CASE
          WHEN ({inner}) IS NOT NULL
               AND EXTRACT(YEAR FROM ({inner})) BETWEEN 1900 AND 2100
          THEN ({inner})
          ELSE NULL
        END
    """
    return f"({guarded})"

def _varchar_sql(col_sql: str) -> str:
    return f"CAST({col_sql} AS VARCHAR)"

def build_typed_expr(orig_col: str, alias: str) -> str:
    """
    Build a SELECT expression for an *existing* column.
    orig_col should be the ORIGINAL quoted name (e.g., '"Program End Date"').
    alias is the safe snake_case target column name.
    """
    col_sql = f'{orig_col}'  # already quoted upstream
    if alias in DATE_COLS:
        return f"{_date_parse_sql(col_sql)} AS {alias}"
    else:
        return f"{_varchar_sql(col_sql)} AS {alias}"

def build_typed_null(alias: str) -> str:
    """
    Build a correctly-typed NULL for a *missing* column.
    """
    if alias in DATE_COLS:
        return f"CAST(NULL AS TIMESTAMP) AS {alias}"
    else:
        return f"CAST(NULL AS VARCHAR) AS {alias}"

def _first_present(cols, candidates, label):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise ValueError(f"Could not find {label}. Available columns: {sorted(cols)}")

def _read_revelio_companies(con, verbose = False):
    """
    Read and clean Revelio company data.
    """
    # check if revelio companies parquet exists, if not, read and save
    revelio_parquet_path = f"{INT_FOLDER}/revelio_company_mapping_full_20251119.parquet"
    
    if not os.path.exists(revelio_parquet_path):
        if verbose:
            print("🔄 Reading and saving Revelio company data...")

        import wrds 
        db = wrds.Connection(wrds_username='amykimecon')
        data = db.raw_sql("SELECT rcid, company, year_founded, factset_entity_id, url, naics_code, hq_street_address, hq_zip_code, hq_city, hq_metro_area, hq_state, hq_country FROM revelio.company_mapping")
        data.to_parquet(f"{INT_FOLDER}/revelio_company_mapping_full_20251119.parquet")
        print("✅ Saved Revelio company mapping data.")
    
    con.sql(f"CREATE OR REPLACE TABLE revelio_companies AS SELECT * FROM read_parquet('{revelio_parquet_path}')")

    if verbose:
        print("✅ Loaded Revelio company data as 'revelio_companies' from Parquet.")
    return "revelio_companies"

def _read_ipeds_crosswalk(con, verbose = False):
    """
    Read and clean IPEDS crosswalk data.
    """
    # check if ipeds crosswalk parquet exists, if not, read and save
    ipeds_parquet_path = f"{INT_FOLDER}/ipeds_crosswalk_2021.parquet"

    if not os.path.exists(ipeds_parquet_path):
        if verbose:
            print("🔄 Reading and saving IPEDS crosswalk data...")
        ipeds_name_path = f"{root}/data/raw/ipeds_name_cw_2021.xlsx"
        ipeds_zip_path = f"{root}/data/raw/ipeds_cw_2021.csv"

        univ_cw = pd.read_excel(
            ipeds_name_path,
            sheet_name="Crosswalk",
            usecols=["OPEID", "IPEDSMatch", "PEPSSchname", "PEPSLocname", "IPEDSInstnm", "OPEIDMain", "IPEDSMain"],
        )
        univ_cw["UNITID"] = univ_cw["IPEDSMatch"].astype(str).str.replace("No match", "-1", regex=False).astype(int)

        zip_cw = pd.read_csv(
            ipeds_zip_path,
            usecols=["UNITID", "OPEID", "INSTNM", "CITY", "STABBR", "ZIP", "ALIAS"],
        )

        # merge sources on id numbers
        merged = univ_cw[univ_cw["UNITID"] != -1].merge(zip_cw, on=["UNITID", "OPEID"], how="right").melt(
                id_vars=["UNITID", "CITY", "STABBR", "ZIP"],
                value_vars=["PEPSSchname", "PEPSLocname", "IPEDSInstnm", "INSTNM", "ALIAS"],
                var_name="source",
                value_name="instname",
            ).dropna(subset=["instname"])
        
        # split name columns on | and explode
        merged['instname_raw'] = merged['instname']
        merged["instname"] = merged["instname_raw"].str.split("|")
        merged = merged.explode("instname").reset_index(drop=True).drop_duplicates(subset=["UNITID", "instname"])

        merged["ZIP"] = merged.groupby("UNITID")["ZIP"].transform(lambda s: s.ffill().bfill()).astype(str).str.replace(r"-[0-9]+$", "", regex=True)
        merged["CITY"] = merged.groupby("UNITID")["CITY"].transform(lambda s: s.ffill().bfill())
        merged['ALIAS'] = merged['source'] == 'ALIAS'
        
        merged.to_parquet(ipeds_parquet_path, index=False)

        if verbose:
          print("✅ Saved IPEDS crosswalk data.")

    con.sql(f"CREATE OR REPLACE TABLE ipeds_crosswalk AS SELECT * FROM read_parquet('{ipeds_parquet_path}')")

    if verbose:
        print("✅ Loaded IPEDS crosswalk data as 'ipeds_crosswalk' from Parquet.")
    return "ipeds_crosswalk"

##################################
## CREATING CROSSWALKS ##
##################################
def _create_f1_inst_crosswalk(con, verbose=False, cache_path=None, load_cache=False, save_output=False):
    """
    Create crosswalk tables for FOIA SEVP to UNITID from IPEDS data.
    """
    if load_cache and cache_path and os.path.exists(cache_path):
        con.sql(f"CREATE OR REPLACE TABLE f1_inst_unitid_crosswalk AS SELECT * FROM read_parquet('{cache_path}')")
        if verbose:
            print(f"✅ Loaded FOIA SEVP to IPEDS UNITID crosswalk from '{cache_path}'.")
        return "f1_inst_unitid_crosswalk"

    if verbose:
        print("🔄 Creating institution name crosswalk tables...")

    f1_inst = con.sql(
    f"""    
        SELECT
          school_name,
          ROW_NUMBER() OVER(ORDER BY school_name, campus_zip_code) AS f1_row_num,
          {_sql_clean_inst_name("school_name")} AS f1_instname_clean,
          {_sql_normalize("campus_city")} AS f1_city_clean,
          {_sql_state_name_to_abbr("campus_state")} AS f1_state_clean,
          {_sql_clean_zip("campus_zip_code")} AS f1_zip_clean
        FROM allyrs_raw
        WHERE school_name IS NOT NULL
        GROUP BY school_name, campus_city, campus_state, campus_zip_code
    """)
    f1_inst.create_view("f1_inst")

    ipeds_inst = con.sql(
    f"""
      SELECT * FROM (
        SELECT
          UNITID,
          {_sql_clean_inst_name("instname")} AS ipeds_instname_clean,
          {_sql_normalize("CITY")} AS ipeds_city_clean,
          {_sql_state_name_to_abbr("STABBR")} AS ipeds_state_clean,
          {_sql_clean_zip("ZIP")} AS ipeds_zip_clean,
          ALIAS AS ipeds_alias
        FROM ipeds_crosswalk
        WHERE instname IS NOT NULL
        GROUP BY UNITID, instname, CITY, STABBR, ZIP, ALIAS
      ) WHERE NOT ipeds_instname_clean IN ({",".join([f"'{statename}'" for statename in STATE_NAME_TO_ABBR.keys()])}) AND LENGTH(ipeds_instname_clean) > 1 GROUP BY UNITID, ipeds_instname_clean, ipeds_city_clean, ipeds_state_clean, ipeds_zip_clean, ipeds_alias
    """)
    ipeds_inst.create_view("ipeds_inst")

    con.sql(
    f"""
      CREATE OR REPLACE VIEW raw_matches AS 
      WITH
      raw_match AS (
        SELECT *,
          CASE WHEN f1_inst.f1_zip_clean = ipeds_inst.ipeds_zip_clean THEN 5
          WHEN f1_inst.f1_city_clean = ipeds_inst.ipeds_city_clean AND f1_inst.f1_state_clean = ipeds_inst.ipeds_state_clean THEN 4
          WHEN f1_inst.f1_state_clean = ipeds_inst.ipeds_state_clean THEN 3
          WHEN ipeds_inst.ipeds_alias = FALSE THEN 2
          WHEN ipeds_inst.UNITID IS NOT NULL THEN 1
          ELSE 0 END AS match_rank,
          MAX(CASE WHEN f1_inst.f1_zip_clean = ipeds_inst.ipeds_zip_clean THEN 5
          WHEN f1_inst.f1_city_clean = ipeds_inst.ipeds_city_clean AND f1_inst.f1_state_clean = ipeds_inst.ipeds_state_clean THEN 4
          WHEN f1_inst.f1_state_clean = ipeds_inst.ipeds_state_clean THEN 3
          WHEN ipeds_inst.ipeds_alias = FALSE THEN 2
          WHEN ipeds_inst.UNITID IS NOT NULL THEN 1
          ELSE 0 END) OVER(PARTITION BY f1_inst.f1_row_num) AS max_match_rank
        FROM f1_inst LEFT JOIN ipeds_inst ON 
          f1_inst.f1_instname_clean = ipeds_inst.ipeds_instname_clean
      )
      SELECT *, COUNT(UNITID) OVER(PARTITION BY f1_row_num) AS match_count FROM raw_match WHERE match_rank = max_match_rank
    """)

    print(f"Total F-1 institutions: {con.sql('SELECT COUNT(DISTINCT f1_row_num) AS cnt FROM f1_inst').df().iloc[0,0]}")

    good_matches = con.sql("SELECT * FROM raw_matches WHERE match_count = 1")
    good_matches.create_view("good_matches")
    print(f"Good matches (unique UNITID): {good_matches.df().shape[0]}")

    rematch_sample_filter = ""
    if REMATCH_SAMPLE_SIZE is not None:
        rematch_sample_filter = f"WHERE rn <= {REMATCH_SAMPLE_SIZE}"

    second_match = con.sql(
    f"""
      WITH unmatched AS (
        SELECT f1_row_num
        FROM raw_matches
        WHERE match_count = 0
      ),
      rematch_samp AS (
        SELECT f1_row_num
        FROM (
          SELECT
            f1_row_num,
            ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn
          FROM unmatched
        )
        {rematch_sample_filter}
      ),
      f1_rematch AS (
        SELECT f1_inst.*
        FROM rematch_samp
        JOIN f1_inst USING(f1_row_num)
      ),
      candidate_scores AS (
        SELECT
          f1_rematch.f1_row_num,
          f1_rematch.school_name,
          f1_rematch.f1_instname_clean,
          f1_rematch.f1_city_clean,
          f1_rematch.f1_state_clean,
          f1_rematch.f1_zip_clean,
          ipeds_inst.UNITID,
          ipeds_inst.ipeds_instname_clean,
          ipeds_inst.ipeds_city_clean,
          ipeds_inst.ipeds_state_clean,
          ipeds_inst.ipeds_zip_clean,
          ipeds_inst.ipeds_alias,
          CASE
            WHEN f1_rematch.f1_zip_clean IS NOT NULL
                AND ipeds_inst.ipeds_zip_clean IS NOT NULL
                AND f1_rematch.f1_zip_clean = ipeds_inst.ipeds_zip_clean
            THEN TRUE ELSE FALSE END AS is_zip_match,
          CASE
            WHEN f1_rematch.f1_city_clean IS NOT NULL
                AND ipeds_inst.ipeds_city_clean IS NOT NULL
                AND f1_rematch.f1_city_clean = ipeds_inst.ipeds_city_clean
            THEN TRUE ELSE FALSE END AS is_city_match,
          jaro_winkler_similarity(f1_rematch.f1_instname_clean, ipeds_inst.ipeds_instname_clean) AS name_jaro_winkler,
          CASE
            WHEN f1_rematch.f1_instname_clean IS NULL OR ipeds_inst.ipeds_instname_clean IS NULL OR LENGTH(ipeds_inst.ipeds_instname_clean) <= 5 THEN FALSE
            WHEN POSITION(ipeds_inst.ipeds_instname_clean IN f1_rematch.f1_instname_clean) > 0 THEN TRUE
            WHEN POSITION(f1_rematch.f1_instname_clean IN ipeds_inst.ipeds_instname_clean) > 0 THEN TRUE
            ELSE FALSE
          END AS is_subset_match
        FROM f1_rematch
        JOIN ipeds_inst
          ON f1_rematch.f1_state_clean = ipeds_inst.ipeds_state_clean
      ),
      ranked_candidates AS (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY f1_row_num
            ORDER BY
              is_subset_match DESC,
              is_zip_match DESC,
              is_city_match DESC,
              name_jaro_winkler DESC,
              ipeds_alias ASC,
              UNITID
          ) AS candidate_rank
        FROM candidate_scores
        WHERE name_jaro_winkler IS NOT NULL
      )
      SELECT *
      FROM ranked_candidates
      WHERE candidate_rank = 1
        AND ((is_subset_match AND ipeds_instname_clean ~ '.*\\s.*') OR ((is_zip_match OR is_city_match) AND name_jaro_winkler >= 0.8) OR name_jaro_winkler >= {REMATCH_JW_THRESHOLD})
    """)
    second_match.create_view("second_match")
    second_match_df = second_match.df()
    print(
        f"Second match candidates: {second_match_df.shape[0]} rows covering {second_match_df['f1_row_num'].nunique()} institutions "
        f"(threshold={REMATCH_JW_THRESHOLD})"
    )

    tie_matches = con.sql(
    """
      WITH tie_candidates AS (
        SELECT
          *,
          CASE
            WHEN f1_zip_clean IS NOT NULL
                AND ipeds_zip_clean IS NOT NULL
                AND f1_zip_clean = ipeds_zip_clean
            THEN TRUE ELSE FALSE END AS is_zip_match,
          CASE
            WHEN f1_city_clean IS NOT NULL
                AND ipeds_city_clean IS NOT NULL
                AND f1_city_clean = ipeds_city_clean
            THEN TRUE ELSE FALSE END AS is_city_match
        FROM raw_matches
        WHERE match_count > 1
      ),
      ranked_ties AS (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY f1_row_num
            ORDER BY
              is_zip_match DESC,
              is_city_match DESC,
              ipeds_alias ASC,
              UNITID
          ) AS tie_rank,
          COUNT(*) OVER (PARTITION BY f1_row_num) AS tie_candidate_count
        FROM tie_candidates
      ),
      aggregated_ties AS (
        SELECT
          f1_row_num,
          array_agg(DISTINCT UNITID) AS tie_unitids,
          array_agg(DISTINCT ipeds_instname_clean) AS tie_instnames
        FROM ranked_ties
        GROUP BY f1_row_num
      )
      SELECT
        ranked_ties.f1_row_num,
        ranked_ties.school_name,
        ranked_ties.f1_instname_clean,
        ranked_ties.UNITID,
        ranked_ties.ipeds_instname_clean,
        ranked_ties.tie_candidate_count,
        aggregated_ties.tie_unitids,
        aggregated_ties.tie_instnames
      FROM ranked_ties
      JOIN aggregated_ties USING (f1_row_num)
      WHERE tie_rank = 1
    """)
    tie_matches.create_view("tie_matches")
    tie_matches_df = tie_matches.df()
    if tie_matches_df.empty:
        print("Ambiguous direct matches: 0 institutions with equal top ranks")
    else:
        avg_candidates = tie_matches_df["tie_candidate_count"].mean()
        print(
            f"Ambiguous direct matches: {tie_matches_df.shape[0]} institutions with multiple best UNITIDs "
            f"(avg candidates {avg_candidates:.2f})"
        )

    con.sql(
    """
    CREATE OR REPLACE TABLE f1_inst_unitid_crosswalk AS
    SELECT
      f1_inst.*,
      CASE
        WHEN match1.UNITID IS NOT NULL THEN match1.UNITID
        WHEN match2.UNITID IS NOT NULL THEN match2.UNITID
        WHEN match3.UNITID IS NOT NULL THEN match3.UNITID
        ELSE NULL
      END AS UNITID,
      CASE 
        WHEN match1.ipeds_instname_clean IS NOT NULL THEN match1.ipeds_instname_clean
        WHEN match2.ipeds_instname_clean IS NOT NULL THEN match2.ipeds_instname_clean
        WHEN match3.ipeds_instname_clean IS NOT NULL THEN match3.ipeds_instname_clean
        ELSE NULL
      END AS ipeds_instname_clean,
      CASE
        WHEN match1.matchtype = 'direct' THEN 'direct'
        WHEN match2.matchtype = 'fuzzy' THEN 'fuzzy'
        WHEN match3.matchtype = 'tie' THEN 'tie'
        ELSE 'none'
      END AS matchtype
    FROM f1_inst
    LEFT JOIN (
      SELECT f1_row_num, UNITID, ipeds_instname_clean, 'direct' AS matchtype
      FROM good_matches
    ) AS match1 USING (f1_row_num)
    LEFT JOIN (
      SELECT f1_row_num, UNITID, ipeds_instname_clean, 'fuzzy' AS matchtype
      FROM second_match
    ) AS match2 USING (f1_row_num)
    LEFT JOIN (
      SELECT f1_row_num, UNITID, ipeds_instname_clean, tie_candidate_count, tie_unitids, tie_instnames, 'tie' AS matchtype
      FROM tie_matches
    ) AS match3 USING (f1_row_num)
    """)
    if save_output and cache_path:
        con.sql(f"COPY (SELECT * FROM f1_inst_unitid_crosswalk) TO '{cache_path}' (FORMAT PARQUET)")
        if verbose:
            print(f"💾 Saved FOIA SEVP to IPEDS UNITID crosswalk to '{cache_path}'.")
    if verbose:
        print("✅ Created FOIA SEVP to IPEDS UNITID crosswalk table 'f1_inst_unitid_crosswalk'.")
    return "f1_inst_unitid_crosswalk"

def _create_employer_crosswalk(
    con,
    verbose=False,
    cache_path=None,
    entity_cache_path=None,
    load_cache=False,
    save_output=False,
    test = False,
    testn = None
):
    """
    Stage the canonical external F1 employer crosswalk into the local workspace.
    """
    if load_cache and cache_path and os.path.exists(cache_path):
        con.sql(
            f"CREATE OR REPLACE TABLE f1_employer_final_crosswalk AS SELECT * FROM read_parquet('{cache_path}')"
        )
        if verbose:
            print(f"✅ Loaded staged employer crosswalk from '{cache_path}'")
        return "f1_employer_final_crosswalk"

    if verbose:
        print("🔄 Staging external employer crosswalk...")

    runtime_cfg = load_config(DEFAULT_CONFIG_PATH)
    external_cfg = get_cfg_section(runtime_cfg, "external_employer_matching")
    external_output_dir = str(
        external_cfg.get("revelio_cleaning_output_dir", f"{root}/revelio-cleaning/data/company_matching_f1")
    )
    match_source = str(external_cfg.get("match_source", "deterministic")).strip().lower()
    staged = stage_external_f1_crosswalk(
        external_output_dir=external_output_dir,
        match_source=match_source,
        out_path=str(cache_path) if save_output and cache_path else None,
        con=con,
        table_name="f1_employer_final_crosswalk",
        verbose=verbose,
    )
    if verbose:
        matched_total = int(staged["preferred_rcid"].notna().sum())
        print(
            "✅ Staged external employer crosswalk 'f1_employer_final_crosswalk' "
            f"with {len(staged):,} rows ({matched_total:,} unique preferred RCID rows)."
        )
    return "f1_employer_final_crosswalk"


def _compute_preferred_rcid_activity(
    con,
    verbose: bool = False,
    save_output: bool = False,
    auth_counts_path: str = None,
    rcid_list_path: str = None,
):
    """
    Recompute preferred RCID activity from the staged final crosswalk and the
    person-linked employment-corrected F1 file.
    """
    runtime_cfg = load_config(DEFAULT_CONFIG_PATH)
    paths = runtime_cfg.get("paths", {})
    external_cfg = get_cfg_section(runtime_cfg, "external_employer_matching")
    activity_path = str(paths.get("foia_sevp_with_person_id_employment_corrected"))
    min_unique_individuals = int(external_cfg.get("preferred_rcid_min_unique_individuals", 1))
    min_years = int(external_cfg.get("preferred_rcid_min_years", 3))
    min_max_year_gt = external_cfg.get("preferred_rcid_min_max_year_gt", 2012)
    if isinstance(min_max_year_gt, str) and min_max_year_gt.strip().lower() in {"", "none", "null"}:
        min_max_year_gt = None
    if min_max_year_gt is not None:
        min_max_year_gt = int(min_max_year_gt)

    return build_preferred_rcid_activity(
        con,
        activity_parquet_path=activity_path,
        sql_clean_company_name_expr=_sql_clean_company_name,
        sql_normalize_expr=_sql_normalize,
        sql_state_name_to_abbr_expr=_sql_state_name_to_abbr,
        sql_clean_zip_expr=_sql_clean_zip,
        date_parse_sql=_date_parse_sql,
        individual_id_cols=INDIVIDUAL_ID_COLS,
        auth_start_cols=AUTH_START_COLS,
        auth_counts_path=auth_counts_path,
        rcid_list_path=rcid_list_path,
        save_output=save_output,
        min_unique_individuals=min_unique_individuals,
        min_years=min_years,
        min_max_year_gt=min_max_year_gt,
        verbose=verbose,
    )

# SQL helper function for cleaning and normalizing names
def _sql_normalize(colname):
    return f"""
        TRIM(
          REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    LOWER({colname}),
                      '[0-9]+/[0-9]+/[0-9]+', ' ', 'g'),
                    '[^a-z0-9 ]', ' ', 'g'
                ), 
            '\\s+', ' ', 'g'
        ))
    """

# SQL helper function for cleaning institution names
def _sql_clean_inst_name(instnamecol):
    return _sql_normalize(f"REGEXP_REPLACE({instnamecol}, '\\b(at|campus|Campus|CAMPUS|inc|Inc|Inc.|inc.|INC|INC.|\\(.*\\))\\b', ' ', 'g')")

def _sql_clean_company_name(companycol):
    """
    Normalize employer names by stripping common corporate suffixes before applying the generic normalization.
    """
    suffix_regex = (
        "(?i)\\b("
        "inc|inc\\.|incorporated|llc|l\\.l\\.c|llp|l\\.l\\.p|lp|l\\.p|"
        "ltd|ltd\\.|limited|corp|corp\\.|corporation|company|co|co\\.|"
        "pllc|plc|pc|pc\\.|gmbh|ag|sa"
        ")\\b"
    )
    return _sql_normalize(f"REGEXP_REPLACE({companycol}, '{suffix_regex}', ' ', 'g')")

# SQL helper function for cleaning and normalizing ZIP codes (if four digits, add leading zero, if more than five digits, truncate to first five)
def _sql_clean_zip(zipcol):
    zipcolclean = f"TRIM(CAST(REGEXP_REPLACE({zipcol}, '[^0-9]', '', 'g') AS VARCHAR))"
    return f"""
        CASE
            WHEN LENGTH(TRIM(CAST({zipcolclean} AS VARCHAR))) = 4 THEN '0' || TRIM(CAST({zipcolclean} AS VARCHAR))
            WHEN LENGTH(TRIM(CAST({zipcolclean} AS VARCHAR))) >= 5 THEN SUBSTRING(TRIM(CAST({zipcolclean} AS VARCHAR)) FROM 1 FOR 5)
            ELSE TRIM(CAST({zipcolclean} AS VARCHAR))
        END
    """

# SQL helper function to convert state names to abbreviations
def _sql_state_name_to_abbr(statecol):
    cases = " \n".join([f"WHEN LOWER(TRIM({statecol})) = '{name}' THEN '{abbr}'" for name, abbr in STATE_NAME_TO_ABBR.items()])
    return f"""
        CASE
            {cases}
            ELSE UPPER(TRIM({statecol}))
        END
    """

#########
# MAIN  #
#########
con = ddb.connect()
raw_tabs = _import_all_data(con, test=False, verbose=True)
f1_inst_cw = _create_f1_inst_crosswalk(con, verbose=True, cache_path=F1_INST_CROSSWALK_PATH, load_cache=F1_INST_LOAD_FROM_CACHE, save_output=F1_INST_SAVE_TO_CACHE)

t1 = time.time()
employer_cw = _create_employer_crosswalk(
      con,
      verbose=True,
      cache_path=F1_EMPLOYER_FINAL_CROSSWALK_PATH,
      entity_cache_path=F1_EMPLOYER_ENTITY_MAPPING_PATH,
      load_cache=F1_EMPLOYER_LOAD_FROM_CACHE,
      save_output=F1_EMPLOYER_SAVE_TO_CACHE
)
t2 = time.time()
print(f"Employer crosswalk creation took {t2 - t1:.2f} seconds.")

# Employer activity filter: RCIDs with >=10 unique individuals in >=3 years where auth start year matches FOIA year
_ = _compute_preferred_rcid_activity(
      con,
      verbose=True,
      save_output=True,
      auth_counts_path=F1_EMPLOYER_AUTH_COUNTS_PATH,
      rcid_list_path=F1_PREFERRED_RCID_LIST_PATH
)
