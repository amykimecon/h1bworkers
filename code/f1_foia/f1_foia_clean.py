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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

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
F1_INST_LOAD_FROM_CACHE = False
F1_INST_SAVE_TO_CACHE = True
F1_EMPLOYER_LOAD_FROM_CACHE = True
F1_EMPLOYER_SAVE_TO_CACHE = False

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
    Create a crosswalk between FOIA employer names and Revelio company RCIDs.
    """
    if load_cache and cache_path and os.path.exists(cache_path):
        con.sql(
            f"CREATE OR REPLACE TABLE f1_employer_final_crosswalk AS SELECT * FROM read_parquet('{cache_path}')"
        )
        if verbose:
            loaded_msg = f"✅ Loaded employer final crosswalk from '{cache_path}'"
            print(loaded_msg)
        return "f1_employer_final_crosswalk"

    if verbose:
        print("🔄 Creating employer name crosswalk tables...")

    con.sql(
    f"""
        CREATE OR REPLACE TABLE f1_employers_raw AS
        WITH base AS (
          SELECT DISTINCT
            employer_name,
            {_sql_clean_company_name("employer_name")} AS f1_empname_clean,
            {_sql_normalize("employer_city")} AS f1_city_clean,
            {_sql_state_name_to_abbr("employer_state")} AS f1_state_clean,
            {_sql_clean_zip("employer_zip_code")} AS f1_zip_clean
          FROM allyrs_raw
          WHERE employer_name IS NOT NULL {f"LIMIT {testn}" if test else ""}
        )
        SELECT
          employer_name,
          f1_empname_clean,
          f1_city_clean,
          f1_state_clean,
          f1_zip_clean,
          ROW_NUMBER() OVER(ORDER BY f1_empname_clean, f1_city_clean, f1_state_clean, f1_zip_clean, employer_name) AS f1_emp_row_num
        FROM base
    """)

    base_employer_rows = con.sql("SELECT f1_emp_row_num FROM f1_employers_raw ORDER BY f1_emp_row_num").df()
    raw_employer_cnt = len(base_employer_rows)
    if verbose:
        print(f"Employer staging: {raw_employer_cnt:,} unique employer/location combinations.")
    base_row_nums = base_employer_rows["f1_emp_row_num"].astype(int).tolist()

    # Generate edges based on similarity criteria
    entity_block_join = ""
    entity_block_nonnull = ""
    if ENTITY_NAME_BLOCK_PREFIX_LEN and ENTITY_NAME_BLOCK_PREFIX_LEN > 0:
        entity_block_join = f"AND substr(a.f1_empname_clean, 1, {ENTITY_NAME_BLOCK_PREFIX_LEN}) = substr(b.f1_empname_clean, 1, {ENTITY_NAME_BLOCK_PREFIX_LEN})"
        entity_block_nonnull = (
            f"AND substr(a.f1_empname_clean, 1, {ENTITY_NAME_BLOCK_PREFIX_LEN}) <> '' "
            f"AND substr(b.f1_empname_clean, 1, {ENTITY_NAME_BLOCK_PREFIX_LEN}) <> ''"
        )
    edge_candidates = con.sql(
    f"""
        SELECT
          a.f1_emp_row_num AS row_a,
          b.f1_emp_row_num AS row_b
        FROM f1_employers_raw AS a
        JOIN f1_employers_raw AS b
          ON a.f1_emp_row_num < b.f1_emp_row_num
         {entity_block_join}
         AND a.f1_empname_clean IS NOT NULL
         AND b.f1_empname_clean IS NOT NULL
         {entity_block_nonnull}
    WHERE
      (
        a.f1_state_clean = b.f1_state_clean
        AND a.f1_city_clean = b.f1_city_clean
        AND a.f1_city_clean NOT IN ({COMMON_CITY_LIST})
        AND jaro_winkler_similarity(a.f1_empname_clean, b.f1_empname_clean) >= 0.95
      )
      OR jaro_winkler_similarity(a.f1_empname_clean, b.f1_empname_clean) >= 0.97
    """
    ).df()
    edge_candidates = edge_candidates.dropna(subset=["row_a", "row_b"])
    edge_candidates["row_a"] = edge_candidates["row_a"].astype(int)
    edge_candidates["row_b"] = edge_candidates["row_b"].astype(int)

    parent = {row: row for row in base_row_nums}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for _, edge in edge_candidates.iterrows():
        union(int(edge["row_a"]), int(edge["row_b"]))

    root_to_entity = {}
    entity_ids = []
    for row in base_row_nums:
        root = find(row)
        if root not in root_to_entity:
            root_to_entity[root] = len(root_to_entity) + 1
        entity_ids.append(root_to_entity[root])

    entity_mapping_df = pd.DataFrame(
        {
            "f1_emp_row_num": base_row_nums,
            "f1_emp_entity_id": entity_ids,
        }
    )
    con.register("f1_emp_entity_mapping_df", entity_mapping_df)
    con.sql(
    """
        CREATE OR REPLACE TABLE f1_emp_entity_mapping AS
        SELECT * FROM f1_emp_entity_mapping_df
    """)

    con.sql(
    """
        CREATE OR REPLACE VIEW f1_employers AS
        SELECT
          raw.*,
          mapping.f1_emp_entity_id
        FROM f1_employers_raw AS raw
        LEFT JOIN f1_emp_entity_mapping AS mapping USING (f1_emp_row_num)
    """)

    total_entities = len(set(entity_ids))
    multi_member_entities = entity_mapping_df.groupby("f1_emp_entity_id").size()
    multi_member_count = (multi_member_entities > 1).sum()
    if verbose:
        print(
            f"Employer entity clustering produced {total_entities:,} entities from {len(base_row_nums):,} employer names "
            f"({multi_member_count:,} with multiple location/name variants)."
        )

    con.sql(
    """
    CREATE OR REPLACE VIEW f1_employer_entities AS
    SELECT
      f1_emp_entity_id,
      MIN(f1_emp_row_num) AS anchor_f1_emp_row_num,
      MIN(f1_empname_clean) AS f1_empname_clean,
      COUNT(*) AS employer_variant_count,
      array_agg(DISTINCT employer_name) AS employer_aliases,
      array_agg(DISTINCT f1_city_clean) FILTER (WHERE f1_city_clean IS NOT NULL) AS city_list,
      array_agg(DISTINCT f1_state_clean) FILTER (WHERE f1_state_clean IS NOT NULL) AS state_list,
      array_agg(DISTINCT f1_zip_clean) FILTER (WHERE f1_zip_clean IS NOT NULL) AS zip_list
    FROM f1_employers
    GROUP BY f1_emp_entity_id
    """)

    rev_employers = con.sql(
    f"""
        SELECT
          rcid,
          {_sql_clean_company_name("company")} AS rev_company_clean,
          {_sql_normalize("hq_city")} AS rev_city_clean,
          {_sql_state_name_to_abbr("hq_state")} AS rev_state_clean,
          {_sql_clean_zip("hq_zip_code")} AS rev_zip_clean,
          UPPER(TRIM(hq_country)) AS rev_country_clean
        FROM revelio_companies
        WHERE company IS NOT NULL
        GROUP BY rcid, company, hq_city, hq_state, hq_zip_code, hq_country
    """)
    rev_employers.create_view("rev_employers")

    con.sql(
    """
      CREATE OR REPLACE VIEW employer_raw_matches AS
      WITH raw_match AS (
        SELECT
          f1_emp.*,
          rev_emp.rcid,
          rev_emp.rev_company_clean,
          rev_emp.rev_city_clean,
          rev_emp.rev_state_clean,
          rev_emp.rev_zip_clean,
          rev_emp.rev_country_clean,
          CASE
            WHEN f1_emp.f1_zip_clean = rev_emp.rev_zip_clean THEN 5
            WHEN f1_emp.f1_city_clean = rev_emp.rev_city_clean AND f1_emp.f1_state_clean = rev_emp.rev_state_clean THEN 4
            WHEN f1_emp.f1_state_clean = rev_emp.rev_state_clean THEN 3
            WHEN rev_emp.rcid IS NOT NULL THEN 1
            ELSE 0
          END AS match_rank,
          MAX(
            CASE
              WHEN f1_emp.f1_zip_clean = rev_emp.rev_zip_clean THEN 5
              WHEN f1_emp.f1_city_clean = rev_emp.rev_city_clean AND f1_emp.f1_state_clean = rev_emp.rev_state_clean THEN 4
              WHEN f1_emp.f1_state_clean = rev_emp.rev_state_clean THEN 3
              WHEN rev_emp.rcid IS NOT NULL THEN 1
              ELSE 0
            END
          ) OVER(PARTITION BY f1_emp.f1_emp_row_num) AS max_match_rank
        FROM f1_employers AS f1_emp
        LEFT JOIN rev_employers AS rev_emp
          ON f1_emp.f1_empname_clean = rev_emp.rev_company_clean
      )
      SELECT
        *,
        COUNT(rcid) OVER(PARTITION BY f1_emp_row_num) AS match_count
      FROM raw_match
      WHERE match_rank = max_match_rank
    """)

    total_employers = con.sql("SELECT COUNT(DISTINCT f1_emp_row_num) AS cnt FROM f1_employers").df().iloc[0, 0]
    if verbose:
        print(f"Total FOIA employers: {total_employers:,}")

    employer_good_matches = con.sql("SELECT * FROM employer_raw_matches WHERE match_count = 1")
    employer_good_matches.create_view("employer_good_matches")
    employer_direct_cnt = con.sql("SELECT COUNT(DISTINCT f1_emp_row_num) AS cnt FROM employer_good_matches").df().iloc[0, 0]
    if verbose:
        print(f"Direct employer matches (unique RCID): {employer_direct_cnt:,}")

    us_country_list = ",".join([f"'{c}'" for c in US_COUNTRY_CODES])
    employer_rematch_sample_filter = ""
    if REMATCH_SAMPLE_SIZE is not None:
        employer_rematch_sample_filter = f"WHERE rn <= {REMATCH_SAMPLE_SIZE}"
    unmatched_cnt = con.sql("SELECT COUNT(DISTINCT f1_emp_row_num) AS cnt FROM employer_raw_matches WHERE match_count = 0").df().iloc[0, 0]
    if verbose:
        sample_note = f" (capped at {REMATCH_SAMPLE_SIZE:,})" if REMATCH_SAMPLE_SIZE is not None else ""
        print(f"Employers requiring fuzzy rematch: {unmatched_cnt:,}{sample_note}")
    fuzzy_block_join = ""
    if FUZZY_NAME_BLOCK_PREFIX_LEN and FUZZY_NAME_BLOCK_PREFIX_LEN > 0:
        fuzzy_block_join = f"AND substr(f1_rematch.f1_empname_clean, 1, {FUZZY_NAME_BLOCK_PREFIX_LEN}) = substr(rev_employers.rev_company_clean, 1, {FUZZY_NAME_BLOCK_PREFIX_LEN})"
    con.sql(
    f"""
    CREATE OR REPLACE TABLE employer_second_match AS
      WITH unmatched AS (
        SELECT f1_emp_row_num
        FROM employer_raw_matches
        WHERE match_count = 0
      ),
      rematch_samp AS (
        SELECT f1_emp_row_num
        FROM (
          SELECT
            f1_emp_row_num,
            ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn
          FROM unmatched
        )
        {employer_rematch_sample_filter}
      ),
      f1_rematch AS (
        SELECT f1_employers.*
        FROM rematch_samp
        JOIN f1_employers USING(f1_emp_row_num)
      ),
      candidate_scores AS (
        SELECT
          f1_rematch.f1_emp_row_num,
          f1_rematch.employer_name,
          f1_rematch.f1_empname_clean,
          f1_rematch.f1_city_clean,
          f1_rematch.f1_state_clean,
      f1_rematch.f1_zip_clean,
      rev_employers.rcid,
      rev_employers.rev_company_clean,
      rev_employers.rev_city_clean,
      rev_employers.rev_state_clean,
      rev_employers.rev_zip_clean,
      rev_employers.rev_country_clean,
      substr(f1_rematch.f1_empname_clean, 1, 1) AS f1_first_char,
      substr(rev_employers.rev_company_clean, 1, 1) AS rev_first_char,
      CASE
        WHEN f1_rematch.f1_zip_clean IS NOT NULL
             AND rev_employers.rev_zip_clean IS NOT NULL
             AND f1_rematch.f1_zip_clean = rev_employers.rev_zip_clean
        THEN TRUE ELSE FALSE END AS is_zip_match,
      CASE
        WHEN f1_rematch.f1_city_clean IS NOT NULL
             AND rev_employers.rev_city_clean IS NOT NULL
             AND f1_rematch.f1_city_clean = rev_employers.rev_city_clean
             AND f1_rematch.f1_city_clean NOT IN ({COMMON_CITY_LIST})
        THEN TRUE ELSE FALSE END AS is_city_match,
      jaro_winkler_similarity(f1_rematch.f1_empname_clean, rev_employers.rev_company_clean) AS name_jaro_winkler,
      CASE
        WHEN f1_rematch.f1_empname_clean IS NULL OR rev_employers.rev_company_clean IS NULL OR LENGTH(rev_employers.rev_company_clean) <= 5 THEN FALSE
        WHEN POSITION(rev_employers.rev_company_clean IN f1_rematch.f1_empname_clean) > 0 THEN TRUE
        WHEN POSITION(f1_rematch.f1_empname_clean IN rev_employers.rev_company_clean) > 0 THEN TRUE
        ELSE FALSE
      END AS is_subset_match
    FROM f1_rematch
    JOIN rev_employers
      ON f1_rematch.f1_state_clean = rev_employers.rev_state_clean
      {fuzzy_block_join}
    WHERE rev_employers.rev_country_clean IN ({us_country_list})
  ),
      ranked_candidates AS (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY f1_emp_row_num
            ORDER BY
              is_subset_match DESC,
              is_zip_match DESC,
              is_city_match DESC,
              name_jaro_winkler DESC,
              rcid
          ) AS candidate_rank
        FROM candidate_scores
        WHERE name_jaro_winkler IS NOT NULL
      )
      SELECT *
      FROM ranked_candidates
      WHERE candidate_rank = 1
        AND (
          (is_subset_match AND rev_company_clean ~ '.*\\s.*' AND name_jaro_winkler >= 0.9)
          OR ((is_zip_match OR is_city_match) AND name_jaro_winkler >= 0.9) OR
          (is_subset_match AND (is_city_match OR is_zip_match) AND name_jaro_winkler >= 0.85)
          OR name_jaro_winkler >= 0.98
        )
    """)
    employer_second_total = con.sql("SELECT COUNT(*) AS cnt FROM employer_second_match").df().iloc[0, 0]
    employer_second_institutions = con.sql("SELECT COUNT(DISTINCT f1_emp_row_num) AS cnt FROM employer_second_match").df().iloc[0, 0]
    print(
        f"Employer fuzzy matches: {employer_second_total} rows covering "
        f"{employer_second_institutions} employers (threshold={REMATCH_JW_THRESHOLD})"
    )

    employer_tie_matches = con.sql(
    f"""
      WITH tie_candidates AS (
        SELECT
          *,
          CASE
            WHEN f1_zip_clean IS NOT NULL AND rev_zip_clean IS NOT NULL AND f1_zip_clean = rev_zip_clean THEN TRUE
            ELSE FALSE
          END AS is_zip_match,
          CASE
            WHEN f1_city_clean IS NOT NULL
                 AND rev_city_clean IS NOT NULL
                 AND f1_city_clean = rev_city_clean
                 AND f1_city_clean NOT IN ({COMMON_CITY_LIST})
            THEN TRUE
            ELSE FALSE
          END AS is_city_match
        FROM employer_raw_matches
        WHERE match_count > 1
      ),
      ranked_ties AS (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY f1_emp_row_num
            ORDER BY
              is_zip_match DESC,
              is_city_match DESC,
              rcid
          ) AS tie_rank,
          COUNT(*) OVER (PARTITION BY f1_emp_row_num) AS tie_candidate_count
        FROM tie_candidates
      ),
      aggregated_ties AS (
        SELECT
          f1_emp_row_num,
          array_agg(DISTINCT rcid) AS tie_rcids,
          array_agg(DISTINCT rev_company_clean) AS tie_company_names
        FROM ranked_ties
        GROUP BY f1_emp_row_num
      )
      SELECT
        ranked_ties.f1_emp_row_num,
        ranked_ties.employer_name,
        ranked_ties.f1_empname_clean,
        ranked_ties.rcid,
        ranked_ties.rev_company_clean,
        ranked_ties.tie_candidate_count,
        aggregated_ties.tie_rcids,
        aggregated_ties.tie_company_names
      FROM ranked_ties
      JOIN aggregated_ties USING (f1_emp_row_num)
      WHERE tie_rank = 1
    """)
    employer_tie_matches.create_view("employer_tie_matches")
    employer_tie_matches_df = employer_tie_matches.df()
    if verbose:
        if employer_tie_matches_df.empty:
            print("Ambiguous employer direct matches: 0 employers with equal top ranks")
        else:
            avg_ties = employer_tie_matches_df["tie_candidate_count"].mean()
            print(
                f"Ambiguous employer direct matches: {employer_tie_matches_df.shape[0]:,} employers "
                f"(avg candidates {avg_ties:.2f})"
            )

    con.sql(
    """
    CREATE OR REPLACE TABLE f1_employer_rcid_crosswalk AS
    SELECT
      f1_employers.*,
      match1.rcid AS direct_rcid,
      match1.rev_company_clean AS direct_company_name,
      match2.rcid AS fuzzy_rcid,
      match2.rev_company_clean AS fuzzy_company_name,
      match3.rcid AS tie_rcid,
      match3.rev_company_clean AS tie_company_name,
      match3.tie_candidate_count,
      match3.tie_rcids,
      match3.tie_company_names,
      CASE
        WHEN match1.matchtype = 'direct' THEN 'direct'
        WHEN match2.matchtype = 'fuzzy' THEN 'fuzzy'
        WHEN match3.matchtype = 'tie' THEN 'tie'
        ELSE 'none'
      END AS matchtype
    FROM f1_employers
    LEFT JOIN (
      SELECT f1_emp_row_num, rcid, rev_company_clean, 'direct' AS matchtype
      FROM employer_good_matches
    ) AS match1 USING (f1_emp_row_num)
    LEFT JOIN (
      SELECT f1_emp_row_num, rcid, rev_company_clean, 'fuzzy' AS matchtype
      FROM employer_second_match
    ) AS match2 USING (f1_emp_row_num)
    LEFT JOIN (
      SELECT f1_emp_row_num, rcid, rev_company_clean, tie_candidate_count, tie_rcids, tie_company_names, 'tie' AS matchtype
      FROM employer_tie_matches
    ) AS match3 USING (f1_emp_row_num)
    """)
    if verbose:
        print("✅ Created FOIA employer to Revelio RCID crosswalk table 'f1_employer_rcid_crosswalk'.")
    con.sql(
    """
    CREATE OR REPLACE TABLE f1_employer_entity_crosswalk AS
    WITH match_union AS (
      SELECT
        f1_emp_entity_id,
        direct_rcid AS rcid,
        direct_company_name AS company_name,
        'direct' AS match_source
      FROM f1_employer_rcid_crosswalk
      WHERE direct_rcid IS NOT NULL
      UNION ALL
      SELECT
        f1_emp_entity_id,
        fuzzy_rcid AS rcid,
        fuzzy_company_name AS company_name,
        'fuzzy' AS match_source
      FROM f1_employer_rcid_crosswalk
      WHERE fuzzy_rcid IS NOT NULL
      UNION ALL
      SELECT
        f1_emp_entity_id,
        tie_rcid AS rcid,
        tie_company_name AS company_name,
        'tie_primary' AS match_source
      FROM f1_employer_rcid_crosswalk
      WHERE tie_rcid IS NOT NULL
      UNION ALL
      SELECT
        cw.f1_emp_entity_id,
        list_element(cw.tie_rcids, idx) AS tie_rcid_candidate,
        list_element(cw.tie_company_names, idx) AS tie_company_candidate,
        'tie_pool' AS match_source
      FROM f1_employer_rcid_crosswalk AS cw,
      LATERAL UNNEST(range(1, array_length(cw.tie_rcids) + 1)) AS t(idx)
      WHERE cw.tie_rcids IS NOT NULL
        AND cw.tie_company_names IS NOT NULL
        AND array_length(cw.tie_company_names) >= idx
    ),
    aggregated AS (
      SELECT
        f1_emp_entity_id,
        array_agg(DISTINCT rcid) AS matched_rcids,
        array_agg(DISTINCT company_name) FILTER (WHERE company_name IS NOT NULL) AS matched_company_names
      FROM match_union
      GROUP BY f1_emp_entity_id
    ),
    prioritized AS (
      SELECT
        f1_emp_entity_id,
        rcid,
        company_name,
        match_source
      FROM (
        SELECT
          f1_emp_entity_id,
          rcid,
          company_name,
          match_source,
          CASE
            WHEN match_source = 'direct' THEN 1
            WHEN match_source = 'fuzzy' THEN 2
            ELSE 3
          END AS source_rank,
          ROW_NUMBER() OVER (
            PARTITION BY f1_emp_entity_id
            ORDER BY source_rank, rcid
          ) AS rn
        FROM match_union
      )
      WHERE rn = 1
    ),
    match_presence AS (
      SELECT
        f1_emp_entity_id,
        MAX(CASE WHEN match_source = 'direct' THEN 1 ELSE 0 END) AS has_direct_match,
        MAX(CASE WHEN match_source = 'fuzzy' THEN 1 ELSE 0 END) AS has_fuzzy_match,
        MAX(CASE WHEN match_source LIKE 'tie%' THEN 1 ELSE 0 END) AS has_tie_match
      FROM match_union
      GROUP BY f1_emp_entity_id
    )
    SELECT
      entities.*,
      aggregated.matched_rcids,
      aggregated.matched_company_names,
      prioritized.rcid AS preferred_rcid,
      prioritized.company_name AS preferred_company_name,
      prioritized.match_source AS preferred_match_source,
      COALESCE(match_presence.has_direct_match, 0) AS has_direct_match,
      COALESCE(match_presence.has_fuzzy_match, 0) AS has_fuzzy_match,
      COALESCE(match_presence.has_tie_match, 0) AS has_tie_match
    FROM f1_employer_entities AS entities
    LEFT JOIN aggregated USING (f1_emp_entity_id)
    LEFT JOIN prioritized USING (f1_emp_entity_id)
    LEFT JOIN match_presence USING (f1_emp_entity_id)
    """)

    con.sql(
    """
    CREATE OR REPLACE TABLE f1_employer_final_crosswalk AS
    SELECT
      base.employer_name,
      base.f1_emp_row_num,
      base.f1_emp_entity_id,
      base.f1_empname_clean,
      base.f1_city_clean,
      base.f1_state_clean,
      base.f1_zip_clean,
      entity.preferred_rcid,
      entity.preferred_company_name,
      entity.preferred_match_source,
      entity.matched_rcids,
      entity.matched_company_names,
      entity.has_direct_match,
      entity.has_fuzzy_match,
      entity.has_tie_match
    FROM f1_employers AS base
    LEFT JOIN f1_employer_entity_crosswalk AS entity USING (f1_emp_entity_id)
    """)
    if verbose:
        final_total = con.sql("SELECT COUNT(*) AS cnt FROM f1_employer_final_crosswalk").df().iloc[0, 0]
        matched_total = con.sql("SELECT COUNT(*) AS cnt FROM f1_employer_final_crosswalk WHERE preferred_rcid IS NOT NULL").df().iloc[0, 0]
        print(
            "✅ Created final employer crosswalk 'f1_employer_final_crosswalk' "
            f"with {final_total:,} rows ({matched_total:,} matched to an RCID)."
        )
    if save_output and cache_path:
        con.sql(f"COPY (SELECT * FROM f1_employer_final_crosswalk) TO '{cache_path}' (FORMAT PARQUET)")
        if verbose:
            print(f"💾 Saved final employer crosswalk to '{cache_path}'.")
    return "f1_employer_final_crosswalk"

# SQL helper function for cleaning and normalizing names
def _sql_normalize(colname):
    return f"""
        TRIM(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    LOWER({colname}),
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
