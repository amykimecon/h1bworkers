# File Description: Reading and Cleaning FOIA F-1 Student Data

# Imports and Paths
import argparse
import duckdb as ddb
import pandas as pd
import sys 
import os
import time
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import re 
from pathlib import Path
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
from external_us_school_matching import stage_external_us_school_matching_artifacts
import helpers as help

#########################
## DECLARING CONSTANTS ##
#########################

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
FUZZY_NAME_BLOCK_PREFIX_LEN = 5

INDIVIDUAL_ID_COLS = ["individual_key", "student_key", "student_id", "individual_id"]
AUTH_START_COLS = ["authorization_start_date", "opt_authorization_start_date", "opt_employer_start_date"]

GEONAMES_DIR_PATH: str | None = None
GEONAMES_VARIANTS_LOADED = False

#########################
## PRINTING UTILITIES  ##
#########################
def _print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

def _print_subsection(title: str) -> None:
    print("\n" + "-" * 72)
    print(title)
    print("-" * 72)


def _scalar(con, query: str):
    """Return the first scalar value from a DuckDB query."""
    return con.sql(query).fetchone()[0]


def _relation_exists(con, relation_name: str) -> bool:
    relation_escaped = relation_name.replace("'", "''")
    return bool(
        _scalar(
            con,
            f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name = '{relation_escaped}'
            """,
        )
    )


def _relation_columns(con, relation_name: str) -> set[str]:
    relation_escaped = relation_name.replace("'", "''")
    if not _relation_exists(con, relation_name):
        return set()
    return {row[1] for row in con.sql(f"PRAGMA table_info('{relation_escaped}')").fetchall()}


def _relation_has_column(con, relation_name: str, column_name: str) -> bool:
    return column_name in _relation_columns(con, relation_name)


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds / 60:.2f}m"


def _run_timed_stage(stage_name: str, fn, *, show_timing: bool = False, stage_times: list[tuple[str, float]] | None = None, **kwargs):
    t0 = time.perf_counter()
    result = fn(**kwargs)
    elapsed = time.perf_counter() - t0
    if stage_times is not None:
        stage_times.append((stage_name, elapsed))
    if show_timing:
        print(f"⏱️  {stage_name}: {_fmt_elapsed(elapsed)}")
    return result

#########################
## NAME SUBSTITUTIONS  ##
#########################
INST_NAME_SUBSTITUTIONS: dict[str, str] = {}

def _sql_apply_substitutions(colname: str, substitutions: dict[str, str]) -> str:
    expr = colname
    if not substitutions:
        return expr
    for key, value in substitutions.items():
        key_escaped = re.escape(str(key))
        value_escaped = str(value).replace("'", "''")
        expr = f"REGEXP_REPLACE({expr}, '(?i)\\\\b{key_escaped}\\\\b', '{value_escaped}', 'g')"
    return expr



##################################
## FUNCTIONS FOR IMPORTING DATA ##
##################################
def _import_all_data(
    con,
    *,
    combined_parquet_path: str,
    foia_raw_dir: str,
    foia_byyear_dir: str,
    revelio_parquet_path: str,
    ipeds_parquet_path: str,
    ipeds_name_path: str,
    ipeds_zip_path: str,
    ipeds_unitid_main_cw_path: str | None = None,
    wrds_users_path: str,
    revelio_inst_min_count: int = 10,
    revelio_inst_test_sample_size: int | None = 1000,
    ipeds_year_start: int | None = None,
    ipeds_year_end: int | None = None,
    wrds_username: str,
    revelio_test_limit: int | None = None,
    test: bool = False,
    verbose: bool = False,
):
    """
    Import all relevant data files into DuckDB.
    """
    stage_times: list[tuple[str, float]] = []
    total_t0 = time.perf_counter()

    # FOIA SEVP Data
    t0 = time.perf_counter()
    foiatab = _read_foia_all(
        con,
        combined_parquet_path=combined_parquet_path,
        foia_raw_dir=foia_raw_dir,
        foia_byyear_dir=foia_byyear_dir,
        save_output=(not test),
        verbose=verbose,
    )
    stage_times.append(("FOIA import", time.perf_counter() - t0))

    # Revelio Company Data
    t0 = time.perf_counter()
    revcompanytab = _read_revelio_companies(
        con,
        revelio_parquet_path=revelio_parquet_path,
        wrds_username=wrds_username,
        test_limit=revelio_test_limit,
        save_output=(not test),
        verbose=verbose,
    )
    stage_times.append(("Revelio companies", time.perf_counter() - t0))

    # Revelio Institution Data
    t0 = time.perf_counter()
    revinstitutiontab = _read_revelio_institutions(
        con,
        wrds_users_path=wrds_users_path,
        min_count=revelio_inst_min_count,
        test=test,
        test_sample_size=revelio_inst_test_sample_size,
        verbose=verbose,
    )
    stage_times.append(("Revelio institutions", time.perf_counter() - t0))

    # IPEDS Crosswalk Data
    t0 = time.perf_counter()
    ipedsnamestab = _read_ipeds_crosswalk(
        con,
        ipeds_parquet_path=ipeds_parquet_path,
        ipeds_name_path=ipeds_name_path,
        ipeds_zip_path=ipeds_zip_path,
        ipeds_unitid_main_cw_path=ipeds_unitid_main_cw_path,
        ipeds_year_start=ipeds_year_start,
        ipeds_year_end=ipeds_year_end,
        save_output=(not test),
        verbose=verbose,
    )
    stage_times.append(("IPEDS crosswalk", time.perf_counter() - t0))

    if verbose:
        _print_subsection("Import timing summary")
        for stage_name, elapsed in stage_times:
            print(f"{stage_name:<24} {_fmt_elapsed(elapsed)}")
        print(f"{'Total import':<24} {_fmt_elapsed(time.perf_counter() - total_t0)}")

    return {
        'foia_sevp': foiatab,
        'revelio_companies': revcompanytab,
        'revelio_institutions': revinstitutiontab,
        'ipeds_crosswalk': ipedsnamestab,
    }

def _read_foia_all(
    con,
    *,
    combined_parquet_path: str,
    foia_raw_dir: str,
    foia_byyear_dir: str,
    save_output: bool = True,
    verbose: bool = False,
):
    """
    Read all FOIA F-1 data files from all year directories, concatenate and save into a DuckDB relation.
    """
    # Check if combined parquet already exists
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
        _print_section("FOIA SEVP: Combine yearly data")
      
    # Iterate through each year's directory
    first_parquet = True
    for dirname in sorted(os.listdir(foia_raw_dir), reverse = True):
        if not os.path.isdir(f"{foia_raw_dir}/{dirname}") or dirname == "J-1":
            continue
        if dirname == "2009":
            if verbose:
                print("⚠️ Skipping 2009 raw directory (duplicate of 2010).")
            continue

        effective_year = dirname
        if dirname in {"2005", "2006", "2007", "2008"}:
            effective_year = str(int(dirname) + 1)
            if verbose:
                print(f"⚠️ Relabeling raw directory {dirname} -> {effective_year}.")
        
        # Check if merged parquet for the year exists
        path = f'{foia_byyear_dir}/merged{effective_year}.parquet'

        # If does not exist, create from scratch
        temp_parquet_path = None
        if not os.path.exists(path):  
            if verbose:
                print(f"🔄 Merged parquet for {effective_year} not found. Creating from raw files...")
            yr_df = _read_foia_raw(
                dirname,
                effective_year=effective_year,
                foia_raw_dir=foia_raw_dir,
                foia_byyear_dir=foia_byyear_dir,
                save_output=save_output,
                verbose=verbose,
            )
            if not save_output and yr_df is not None:
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, dir="/tmp") as tmp:
                    temp_parquet_path = tmp.name
                yr_df.to_parquet(temp_parquet_path, index=False)
                path = temp_parquet_path

        # Read in the year's merged parquet into duckdb safely
        if first_parquet:
            base_cols = None
        base_cols = _read_dir_parquet(con, path, first_parquet, base_cols, verbose)
        first_parquet = False 
        if temp_parquet_path and os.path.exists(temp_parquet_path):
            os.remove(temp_parquet_path)

    # Export combined data to Parquet
    if save_output:
        con.sql(f"""COPY allyrs_raw TO '{combined_parquet_path}' (FORMAT PARQUET)""")
    if verbose and save_output:
        print("✅ Exported combined data to Parquet.")
    return "allyrs_raw"

def _read_foia_raw(
    dirname,
    *,
    effective_year: str,
    foia_raw_dir: str,
    foia_byyear_dir: str,
    save_output: bool = True,
    verbose: bool = False,
):
    # reading in raw data files and concatenating each year into one parquet
    t1 = time.time()
    yr_raw = []
    if not os.path.isdir(f"{foia_raw_dir}/{dirname}") and dirname != "J-1":
        return None 
    for filename in os.listdir(f"{foia_raw_dir}/{dirname}"):
        if not filename[0].isalpha():
            continue
        if filename.lower().startswith("merged"):
            continue
        print("Reading in: ", f"{dirname}/{filename}")
        file_path = f"{foia_raw_dir}/{dirname}/{filename}"
        engine = None
        lower = filename.lower()
        if lower.endswith(".xlsx") or lower.endswith(".xlsm"):
            engine = "openpyxl"
        elif lower.endswith(".xls"):
            engine = "xlrd"
        df_temp = pd.read_excel(file_path, engine=engine).assign(filename=filename)
        yr_raw.append(df_temp)

    # combining all files in the year into one dataframe
    df_yr = pd.concat(yr_raw, ignore_index=True).assign(year = effective_year)
    
    # renaming columns to convert spaces to underscores
    df_yr.columns = [col.lower().replace(" ", "_") for col in df_yr.columns]

    # across all columns, convert "b(6)" to NaN
    df_yr = df_yr.replace("b(6),b(7)(c)", pd.NA)

    # convert all object columns to string (for mixed-type safety)
    df_yr = df_yr.astype({col: "string" for col in df_yr.select_dtypes(include="object").columns})

    df_out = df_yr[df_yr.columns.difference(['birth_date'])]
    if save_output:
        # writing out combined year data (exclude birth date columns)
        os.makedirs(foia_byyear_dir, exist_ok=True)
        df_out.to_parquet(
            f"{foia_byyear_dir}/merged{effective_year}.parquet",
            index=False,
        )
    
    t2 = time.time()
    if verbose:
        action = "writing" if save_output else "processing"
        print(f"Finished {action} {dirname} data ({(t2-t1)/60:.2f} minutes)")
    
    return df_out if not save_output else None
        
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

def _none_if_blankish(value):
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value

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

def _read_revelio_companies(
    con,
    *,
    revelio_parquet_path: str,
    wrds_username: str,
    test_limit: int | None = None,
    save_output: bool = True,
    verbose: bool = False,
):
    """
    Read and clean Revelio company data.
    """
    # check if revelio companies parquet exists, if not, read and save
    if not os.path.exists(revelio_parquet_path):
        if verbose:
            _print_section("Revelio: Load and save company data")

        import wrds 
        db = wrds.Connection(wrds_username=wrds_username)
        rcid_filter = ""
        if test_limit is not None:
            rcid_filter = f"AND rcid IN (SELECT rcid FROM revelio.company_mapping LIMIT {int(test_limit)})"
            if verbose:
                print(f"🔧 Test mode: limiting Revelio RCIDs to first {test_limit} entries.")

        data = db.raw_sql(
            f"""
            SELECT *
            FROM (
                (
                    SELECT
                        COUNT(*) AS n,
                        COUNT(DISTINCT user_id) AS n_users,
                        MAX(startdate) AS recent_start,
                        MAX(enddate) AS recent_end,
                        rcid AS rcid_positions
                    FROM revelio.individual_positions
                    WHERE country = 'United States'
                    {rcid_filter}
                    GROUP BY rcid
                ) AS company_counts
                LEFT JOIN (
                    SELECT rcid AS rcid_metros, top_metro_area, top_state
                    FROM (
                        SELECT
                            n,
                            metro_area AS top_metro_area,
                            state AS top_state,
                            rcid,
                            ROW_NUMBER() OVER (PARTITION BY rcid ORDER BY n DESC) AS r
                        FROM (
                            SELECT
                                COUNT(*) AS n,
                                metro_area,
                                state,
                                rcid
                            FROM revelio.individual_positions
                            WHERE country = 'United States'
                            {rcid_filter}
                            GROUP BY rcid, state, metro_area
                        )
                    )
                    WHERE r = 1
                ) AS company_metros
                    ON company_counts.rcid_positions = company_metros.rcid_metros
            ) AS positions
            LEFT JOIN (
                SELECT * FROM revelio.company_mapping
            ) AS companies
                ON positions.rcid_positions = companies.rcid
            """
        )
        if save_output:
            data.to_parquet(revelio_parquet_path, index=False)
            print("✅ Saved Revelio company mapping data.")
        else:
            con.register("revelio_companies_df", data)
            con.sql("CREATE OR REPLACE TABLE revelio_companies AS SELECT * FROM revelio_companies_df")
            if verbose:
                print("✅ Loaded Revelio company data from WRDS query (test mode, not saved).")
            return "revelio_companies"
    
    con.sql(f"CREATE OR REPLACE TABLE revelio_companies AS SELECT * FROM read_parquet('{revelio_parquet_path}')")

    if verbose:
        print("✅ Loaded Revelio company data as 'revelio_companies' from Parquet.")
    return "revelio_companies"

def _read_revelio_institutions(
    con,
    *,
    wrds_users_path: str,
    min_count: int = 10,
    test: bool = False,
    test_sample_size: int | None = 1000,
    verbose: bool = False,
):
    """
    Load Revelio institutions (RSIDs + university names) from WRDS users parquet.
    Mirrors load_univ_rsid in 10_misc/create_rsid_geoname_cw.py.
    """
    con.sql(f"CREATE OR REPLACE VIEW wrds_users_raw AS SELECT * FROM read_parquet('{wrds_users_path}')")
    cols = {row[1] for row in con.sql("PRAGMA table_info('wrds_users_raw')").fetchall()}

    def _col_or_null(name: str) -> str:
        return name if name in cols else f"NULL::VARCHAR AS {name}"

    select_cols = [
        "rsid",
        "university_raw",
        _col_or_null("university_country"),
        _col_or_null("degree_raw"),
        _col_or_null("field_raw"),
        _col_or_null("degree"),
    ]
    con.sql(
        f"""
        CREATE OR REPLACE VIEW wrds_users_clean AS
        SELECT {", ".join(select_cols)}
        FROM wrds_users_raw
        """
    )

    degree_expr = help.degree_clean_regex_sql()
    con.sql(
        f"""
        CREATE OR REPLACE VIEW wrds_users_filtered AS
        SELECT
          *,
          {degree_expr} AS degree_group
        FROM wrds_users_clean
        WHERE university_raw IS NULL
           OR NOT regexp_matches(university_raw, '^[* ]+$')
        """
    )
    users_view = "wrds_users_filtered"

    con.sql(
        f"""
        CREATE OR REPLACE TABLE revelio_institutions AS
        WITH uni_counts AS (
          SELECT
            university_raw,
            COUNT(*) AS total_n,
            SUM(CASE WHEN degree_group = 'High School' THEN 1 ELSE 0 END) AS hs_n,
            SUM(CASE WHEN degree_group = 'Non-Degree' THEN 1 ELSE 0 END) AS nd_n
          FROM {users_view}
          WHERE university_raw IS NOT NULL
          GROUP BY university_raw
        ),
        uni_mode AS (
          SELECT
            university_raw,
            degree_group AS modal_degree_group
          FROM (
            SELECT
              university_raw,
              degree_group,
              COUNT(*) AS cnt,
              ROW_NUMBER() OVER (
                PARTITION BY university_raw
                ORDER BY COUNT(*) DESC, degree_group
              ) AS rn
            FROM {users_view}
            WHERE university_raw IS NOT NULL
              AND degree_group IS NOT NULL
              AND degree_group <> 'Missing'
            GROUP BY university_raw, degree_group
          )
          WHERE rn = 1
        ),
        uni_stats AS (
          SELECT
            c.university_raw,
            c.total_n,
            m.modal_degree_group,
            CASE WHEN c.total_n > 0 THEN c.hs_n::DOUBLE / c.total_n ELSE 0 END AS hs_share,
            CASE WHEN c.total_n > 0 THEN c.nd_n::DOUBLE / c.total_n ELSE 0 END AS nd_share
          FROM uni_counts AS c
          LEFT JOIN uni_mode AS m USING (university_raw)
        )
        SELECT
          u.university_raw,
          MAX(CASE WHEN u.university_country = 'United States' THEN 1 ELSE 0 END) AS us_ind,
          MAX(CASE WHEN NOT (u.university_country = 'United States' OR u.university_country IS NULL) THEN 1 ELSE 0 END) AS nonus_ind,
          COUNT(*) AS n
        FROM {users_view} AS u
        LEFT JOIN uni_stats AS s USING (university_raw)
        WHERE u.university_raw IS NOT NULL
          AND NOT (
            (s.modal_degree_group = 'High School' AND s.hs_share > 0.01)
            OR (s.modal_degree_group = 'Non-Degree' AND s.nd_share > 0.01)
          )
        GROUP BY u.university_raw
        HAVING COUNT(*) >= {int(min_count)}
        """
    )
    if test:
        sample_n = 1000 if test_sample_size is None else int(test_sample_size)
        if sample_n <= 0:
            raise ValueError("test_sample_size must be positive when test mode is enabled.")
        con.sql(
            f"""
            CREATE OR REPLACE TABLE revelio_institutions AS
            SELECT *
            FROM revelio_institutions
            ORDER BY RANDOM()
            LIMIT {sample_n}
            """
        )
        if verbose:
            sampled_unis = int(_scalar(con, "SELECT COUNT(*) FROM revelio_institutions"))
            print(f"🔧 Test mode: sampled {sampled_unis:,} filtered Revelio institutions.")
    if verbose:
        _print_section("Revelio Institution Filtering")
        totals = con.sql(
            f"""
            WITH uni_counts AS (
              SELECT university_raw, COUNT(*) AS total_n
              FROM {users_view}
              WHERE university_raw IS NOT NULL
              GROUP BY university_raw
            ),
            uni_mode AS (
              SELECT
                university_raw,
                degree_group AS modal_degree_group
              FROM (
                SELECT
                  university_raw,
                  degree_group,
                  COUNT(*) AS cnt,
                  ROW_NUMBER() OVER (
                    PARTITION BY university_raw
                    ORDER BY COUNT(*) DESC, degree_group
                  ) AS rn
                FROM {users_view}
                WHERE university_raw IS NOT NULL
                  AND degree_group IS NOT NULL
                  AND degree_group <> 'Missing'
                GROUP BY university_raw, degree_group
              )
              WHERE rn = 1
            ),
            uni_shares AS (
              SELECT
                university_raw,
                SUM(CASE WHEN degree_group = 'High School' THEN 1 ELSE 0 END) AS hs_n,
                SUM(CASE WHEN degree_group = 'Non-Degree' THEN 1 ELSE 0 END) AS nd_n,
                COUNT(*) AS total_n
              FROM {users_view}
              WHERE university_raw IS NOT NULL
              GROUP BY university_raw
            ),
            uni_stats AS (
              SELECT
                c.university_raw,
                c.total_n,
                m.modal_degree_group,
                CASE WHEN c.total_n > 0 THEN s.hs_n::DOUBLE / c.total_n ELSE 0 END AS hs_share,
                CASE WHEN c.total_n > 0 THEN s.nd_n::DOUBLE / c.total_n ELSE 0 END AS nd_share
              FROM uni_counts AS c
              LEFT JOIN uni_mode AS m USING (university_raw)
              LEFT JOIN uni_shares AS s USING (university_raw)
            ),
            base AS (
              SELECT * FROM uni_stats
            ),
            after_min_count AS (
              SELECT * FROM base WHERE total_n >= {int(min_count)}
            ),
            after_hs AS (
              SELECT * FROM after_min_count
              WHERE NOT (modal_degree_group = 'High School' AND hs_share > 0.01)
            ),
            after_nd AS (
              SELECT * FROM after_hs
              WHERE NOT (modal_degree_group = 'Non-Degree' AND nd_share > 0.01)
            )
            SELECT
              (SELECT COUNT(*) FROM base) AS total_univs,
              (SELECT COUNT(*) FROM after_min_count) AS after_min_count,
              (SELECT COUNT(*) FROM after_hs) AS after_hs,
              (SELECT COUNT(*) FROM after_nd) AS after_nd,
              (SELECT SUM(total_n) FROM base) AS total_users,
              (SELECT SUM(total_n) FROM after_min_count) AS users_after_min_count,
              (SELECT SUM(total_n) FROM after_hs) AS users_after_hs,
              (SELECT SUM(total_n) FROM after_nd) AS users_after_nd
            """
        ).df().iloc[0]
        total = int(totals["total_univs"])
        after_min = int(totals["after_min_count"])
        after_hs = int(totals["after_hs"])
        after_nd = int(totals["after_nd"])
        total_users = float(totals["total_users"] or 0)
        users_after_min = float(totals["users_after_min_count"] or 0)
        users_after_hs = float(totals["users_after_hs"] or 0)
        users_after_nd = float(totals["users_after_nd"] or 0)
        def _pct(x: int) -> float:
            return (x / total_users * 100) if total_users else 0.0
        print(f"Total university_raw: {total:,}")
        print(f"After min_count >= {int(min_count)}: {after_min:,} ({_pct(users_after_min):.2f}% of users)")
        print(f"After High School filter: {after_hs:,} ({_pct(users_after_hs):.2f}% of users)")
        print(f"After Non-Degree filter: {after_nd:,} ({_pct(users_after_nd):.2f}% of users)")
        print("✅ Loaded Revelio institutions as 'revelio_institutions'.")
    return "revelio_institutions"

def _read_ipeds_crosswalk(
    con,
    *,
    ipeds_parquet_path: str,
    ipeds_name_path: str,
    ipeds_zip_path: str,
    ipeds_unitid_main_cw_path: str | None = None,
    ipeds_year_start: int | None = None,
    ipeds_year_end: int | None = None,
    save_output: bool = True,
    verbose: bool = False,
):
    """
    Read and clean IPEDS crosswalk data.
    """
    # check if ipeds crosswalk parquet exists, if not, read and save
    if not os.path.exists(ipeds_parquet_path):
        if verbose:
            _print_section("IPEDS: Build crosswalk")

        def _find_year_file(directory: str, year: int) -> str | None:
            if not os.path.isdir(directory):
                return None
            matches = [f for f in os.listdir(directory) if str(year) in f]
            if not matches:
                return None
            matches = sorted(matches)
            if len(matches) > 1 and verbose:
                print(f"⚠️ Multiple files for {year} in {directory}, using '{matches[0]}'")
            return os.path.join(directory, matches[0])

        def _infer_year_from_filename(path: str) -> int | None:
            m = re.search(r"(19|20)\d{2}", os.path.basename(path))
            return int(m.group(0)) if m else None

        def _merge_ipeds_year(name_file: str, zip_file: str, year: int | None) -> pd.DataFrame:
            def _select_cols_case_insensitive(
                df: pd.DataFrame,
                required_cols: list[str],
                label: str,
                optional_cols: list[str] | None = None,
            ) -> pd.DataFrame:
                optional_cols = optional_cols or []
                col_map = {c.upper(): c for c in df.columns}
                missing = [c for c in required_cols if c.upper() not in col_map]
                if missing:
                    raise ValueError(f"Missing columns in {label}: {missing}")
                cols = [col_map[c.upper()] for c in required_cols]
                for col in optional_cols:
                    if col.upper() in col_map:
                        cols.append(col_map[col.upper()])
                    else:
                        df[col] = pd.NA
                        cols.append(col)
                return df[cols].copy()

            def _normalize_id_col(series: pd.Series) -> pd.Series:
                return pd.to_numeric(series, errors="coerce").astype("Int64")

            univ_cw_raw = pd.read_excel(
                name_file,
                sheet_name="Crosswalk",
            )
            univ_cw = _select_cols_case_insensitive(
                univ_cw_raw,
                ["OPEID", "IPEDSMatch", "PEPSSchname", "PEPSLocname", "IPEDSInstnm", "OPEIDMain", "IPEDSMain"],
                f"IPEDS name file {name_file}",
            )
            univ_cw.columns = [c.upper() for c in univ_cw.columns]
            univ_cw["OPEID"] = _normalize_id_col(univ_cw["OPEID"])
            univ_cw["UNITID"] = (
                univ_cw["IPEDSMATCH"].astype(str).str.lower().str.replace("no match", "-1", regex=False).astype(int)
            )
            univ_cw["UNITID"] = _normalize_id_col(univ_cw["UNITID"])

            try:
                zip_cw_raw = pd.read_csv(zip_file)
            except UnicodeDecodeError:
                zip_cw_raw = pd.read_csv(zip_file, encoding="latin1")
            zip_cw = _select_cols_case_insensitive(
                zip_cw_raw,
                ["UNITID", "OPEID", "INSTNM", "CITY", "STABBR", "ZIP"],
                f"IPEDS zip file {zip_file}",
                optional_cols=["ALIAS", "INSTSIZE"],
            )
            zip_cw.columns = [c.upper() for c in zip_cw.columns]
            zip_cw["OPEID"] = _normalize_id_col(zip_cw["OPEID"])
            zip_cw["UNITID"] = _normalize_id_col(zip_cw["UNITID"])

            # merge sources on id numbers
            merged = univ_cw[univ_cw["UNITID"] != -1].merge(
                zip_cw, on=["UNITID", "OPEID"], how="right"
            ).melt(
                id_vars=["UNITID", "CITY", "STABBR", "ZIP", "INSTSIZE"],
                value_vars=["PEPSSCHNAME", "PEPSLOCNAME", "IPEDSINSTNM", "INSTNM", "ALIAS"],
                var_name="source",
                value_name="instname",
            ).dropna(subset=["instname"])

            # split name columns on | and explode
            merged["instname_raw"] = merged["instname"]
            merged["instname"] = merged["instname_raw"].str.split("|")
            merged = (
                merged.explode("instname")
                .reset_index(drop=True)
                .drop_duplicates(subset=["UNITID", "instname"])
            )

            merged["ZIP"] = (
                merged.groupby("UNITID")["ZIP"]
                .transform(lambda s: s.ffill().bfill())
                .astype(str)
                .str.replace(r"-[0-9]+$", "", regex=True)
            )
            merged["CITY"] = merged.groupby("UNITID")["CITY"].transform(lambda s: s.ffill().bfill())
            merged["INSTSIZE"] = (
                pd.to_numeric(merged["INSTSIZE"], errors="coerce")
                .groupby(merged["UNITID"])
                .transform(lambda s: s.ffill().bfill())
            )
            merged["ALIAS"] = merged["source"] == "ALIAS"
            merged["IPEDS_YEAR"] = year
            return merged

        merged = None
        merged_with_year = None
        name_is_dir = os.path.isdir(ipeds_name_path)
        zip_is_dir = os.path.isdir(ipeds_zip_path)

        if name_is_dir and zip_is_dir and ipeds_year_start is not None and ipeds_year_end is not None:
            all_years = []
            for year in range(int(ipeds_year_start), int(ipeds_year_end) + 1):
                name_file = _find_year_file(ipeds_name_path, year)
                zip_file = _find_year_file(ipeds_zip_path, year)
                if not name_file or not zip_file:
                    if verbose:
                        print(f"⚠️ Missing IPEDS files for {year}; name: {bool(name_file)} zip: {bool(zip_file)}")
                    continue
                if verbose:
                    print(f"🔄 Merging IPEDS files for {year}")
                all_years.append(_merge_ipeds_year(name_file, zip_file, year=year))
            if not all_years:
                raise FileNotFoundError(
                    "No IPEDS year pairs found; check directory paths and year range."
                )
            merged_with_year = pd.concat(all_years, ignore_index=True)
            merged = merged_with_year.drop(columns=["IPEDS_YEAR"]).drop_duplicates()
        else:
            if verbose:
                print("🔄 Reading and saving IPEDS crosswalk data (single file mode)")
            inferred_year = _infer_year_from_filename(ipeds_name_path) or _infer_year_from_filename(ipeds_zip_path)
            merged_with_year = _merge_ipeds_year(ipeds_name_path, ipeds_zip_path, year=inferred_year)
            merged = merged_with_year.drop(columns=["IPEDS_YEAR"]).drop_duplicates()

        con.register("ipeds_crosswalk_raw_with_year_df", merged_with_year)
        con.sql("CREATE OR REPLACE TABLE ipeds_crosswalk_raw_with_year AS SELECT * FROM ipeds_crosswalk_raw_with_year_df")
        con.sql(
            f"""
            CREATE OR REPLACE TABLE ipeds_unitid_main_crosswalk AS
            WITH base AS (
              SELECT
                TRY_CAST(UNITID AS BIGINT) AS original_unitid,
                TRY_CAST(IPEDS_YEAR AS INTEGER) AS ipeds_year,
                TRY_CAST(INSTSIZE AS DOUBLE) AS instsize,
                instname,
                {_sql_normalize("instname")} AS instname_clean
              FROM ipeds_crosswalk_raw_with_year
              WHERE UNITID IS NOT NULL
                AND TRY_CAST(UNITID AS BIGINT) IS NOT NULL
            ),
            unitid_name_stats AS (
              SELECT
                instname_clean,
                original_unitid,
                MAX(ipeds_year) AS latest_year,
                MAX(instsize) AS max_instsize,
                MIN(instname) AS alpha_instname
              FROM base
              GROUP BY instname_clean, original_unitid
            ),
            unitid_stats AS (
              SELECT
                original_unitid,
                MAX(ipeds_year) AS latest_year,
                MAX(instsize) AS max_instsize,
                MIN(instname) AS alpha_instname
              FROM base
              GROUP BY original_unitid
            ),
            dupe_names AS (
              SELECT instname_clean
              FROM unitid_name_stats
              GROUP BY instname_clean
              HAVING COUNT(DISTINCT original_unitid) > 1
            ),
            ranked AS (
              SELECT
                instname_clean,
                original_unitid,
                ROW_NUMBER() OVER (
                  PARTITION BY instname_clean
                  ORDER BY latest_year DESC NULLS LAST, max_instsize DESC NULLS LAST, alpha_instname ASC, original_unitid ASC
                ) AS rn
              FROM unitid_name_stats
              WHERE instname_clean IN (SELECT instname_clean FROM dupe_names)
            ),
            name_level_map AS (
              SELECT
                s.original_unitid,
                r.main_unitid AS candidate_main_unitid
              FROM (
                SELECT DISTINCT original_unitid, instname_clean
                FROM unitid_name_stats
              ) AS s
              LEFT JOIN (
                SELECT instname_clean, original_unitid AS main_unitid
                FROM ranked
                WHERE rn = 1
              ) AS r USING (instname_clean)
            ),
            unitid_votes AS (
              SELECT
                original_unitid,
                candidate_main_unitid,
                COUNT(*) AS name_vote_count
              FROM name_level_map WHERE candidate_main_unitid IS NOT NULL
              GROUP BY original_unitid, candidate_main_unitid
            ),
            unitid_to_main AS (
              SELECT original_unitid, candidate_main_unitid AS main_unitid
              FROM (
                SELECT
                  v.original_unitid,
                  v.candidate_main_unitid,
                  ROW_NUMBER() OVER (
                    PARTITION BY v.original_unitid
                    ORDER BY
                      v.name_vote_count DESC,
                      s.latest_year DESC NULLS LAST,
                      s.max_instsize DESC NULLS LAST,
                      s.alpha_instname ASC,
                      v.candidate_main_unitid ASC
                  ) AS rn
                FROM unitid_votes AS v
                LEFT JOIN unitid_stats AS s
                  ON v.candidate_main_unitid = s.original_unitid
              )
              WHERE rn = 1
            )
            SELECT DISTINCT
              b.original_unitid,
              b.ipeds_year,
              COALESCE(m.main_unitid, b.original_unitid) AS main_unitid
            FROM base AS b
            LEFT JOIN unitid_to_main AS m USING (original_unitid)
            """
        )

        con.register("ipeds_crosswalk_pre_df", merged)
        con.sql(
            """
            CREATE OR REPLACE TABLE ipeds_crosswalk AS
            WITH remapped AS (
              SELECT
                CAST(COALESCE(cw.main_unitid, TRY_CAST(p.UNITID AS BIGINT)) AS BIGINT) AS UNITID,
                p.CITY,
                p.STABBR,
                p.ZIP,
                p.source,
                p.instname,
                p.instname_raw,
                p.ALIAS
              FROM ipeds_crosswalk_pre_df AS p
              LEFT JOIN (
                SELECT DISTINCT original_unitid, main_unitid
                FROM ipeds_unitid_main_crosswalk
              ) AS cw
                ON TRY_CAST(p.UNITID AS BIGINT) = cw.original_unitid
            ),
            ranked AS (
              SELECT
                *,
                ROW_NUMBER() OVER (
                  PARTITION BY UNITID, CITY, STABBR, ZIP, instname
                  ORDER BY source, instname_raw, ALIAS
                ) AS rn
              FROM remapped
            )
            SELECT
              UNITID, CITY, STABBR, ZIP, source, instname, instname_raw, ALIAS
            FROM ranked
            WHERE rn = 1
            """
        )

        if save_output:
            con.sql(f"COPY (SELECT * FROM ipeds_crosswalk) TO '{ipeds_parquet_path}' (FORMAT PARQUET)")
            if ipeds_unitid_main_cw_path:
                con.sql(f"COPY (SELECT * FROM ipeds_unitid_main_crosswalk) TO '{ipeds_unitid_main_cw_path}' (FORMAT PARQUET)")
            if verbose:
                print("✅ Saved IPEDS crosswalk data (with UNITID remap).")
                if ipeds_unitid_main_cw_path:
                    print(f"✅ Saved IPEDS UNITID main crosswalk to '{ipeds_unitid_main_cw_path}'.")
        else:
            if verbose:
                print("✅ Loaded IPEDS crosswalk from raw files (test mode, not saved).")
            return "ipeds_crosswalk"

    if os.path.exists(ipeds_parquet_path):
        con.sql(f"CREATE OR REPLACE TABLE ipeds_crosswalk AS SELECT * FROM read_parquet('{ipeds_parquet_path}')")
    if ipeds_unitid_main_cw_path and os.path.exists(ipeds_unitid_main_cw_path):
        con.sql(f"CREATE OR REPLACE TABLE ipeds_unitid_main_crosswalk AS SELECT * FROM read_parquet('{ipeds_unitid_main_cw_path}')")
    elif _relation_exists(con, "ipeds_crosswalk") and not _relation_exists(con, "ipeds_unitid_main_crosswalk"):
        con.sql(
            """
            CREATE OR REPLACE TABLE ipeds_unitid_main_crosswalk AS
            SELECT DISTINCT
              TRY_CAST(UNITID AS BIGINT) AS original_unitid,
              NULL::INTEGER AS ipeds_year,
              TRY_CAST(UNITID AS BIGINT) AS main_unitid
            FROM ipeds_crosswalk
            WHERE UNITID IS NOT NULL
              AND TRY_CAST(UNITID AS BIGINT) IS NOT NULL
            """
        )

    if verbose:
        print("✅ Loaded IPEDS crosswalk data as 'ipeds_crosswalk' from Parquet.")
    return "ipeds_crosswalk"

##################################
## CREATING CROSSWALKS ##
##################################
def _sql_col_or_null(col: str | None) -> str:
    return col if col else "NULL"

CITY_STOPWORDS = [
    "ba",
    "to",
    "in",
    "of",
    "and",
    "university",
    "academy",
    "college",
    "mba",
    "bsc",
    "the",
    "st",
    "area",
    "center",
    "universidad",
    "central",
    "central high",
    "at",
    "valley",
    "jr",
    "sr",
    "columbia",
    "earth",
    "washington",
    "masters",
    "bachelors"
]

CITY_ABBREV_STOPWORDS = [
    "de",
    "in",
    "be",
    "at",
    "st",
    "en",
]


def _sql_city_in_name_expr(
    *,
    left_city: str | None,
    right_name: str,
    right_city: str | None,
    left_name: str,
    mode: str,
) -> str:
    def _expr(city_col: str | None, name_col: str) -> str:
        if not city_col:
            return "FALSE"
        stopwords = ", ".join([f"'{w}'" for w in CITY_STOPWORDS])
        return (
            f"(CASE WHEN {city_col} IS NULL OR {name_col} IS NULL THEN FALSE "
            f"WHEN {city_col} IN ({stopwords}) THEN FALSE "
            f"WHEN LENGTH({city_col}) <= 2 THEN FALSE "
            f"WHEN POSITION({city_col} IN {name_col}) > 0 THEN TRUE "
            f"ELSE FALSE END)"
        )

    if mode == "left_in_right":
        return _expr(left_city, right_name)
    if mode == "right_in_left":
        return _expr(right_city, left_name)
    if mode == "both":
        return f"({_expr(left_city, right_name)} OR {_expr(right_city, left_name)})"
    if mode == "auto":
        if left_city:
            return _expr(left_city, right_name)
        return _expr(right_city, left_name)
    raise ValueError(f"Unknown city_in_name mode: {mode}")

def _state_pattern_sql() -> str:
    names = list(STATE_NAME_TO_ABBR.keys())
    abbrs = list(STATE_NAME_TO_ABBR.values())
    parts = []
    for item in abbrs + names:
        s = re.escape(item.lower())
        s = s.replace("\\ ", "\\\\s+")
        parts.append(s)
    return "(?:" + "|".join(parts) + ")"

def _ensure_geonames_variants(con, geonames_dir: str) -> None:
    global GEONAMES_VARIANTS_LOADED
    geonames_ready = (
        _relation_exists(con, "geonames_variants")
        and _relation_exists(con, "geonames_state_tokens")
        and _relation_exists(con, "geonames_country_tokens")
    )
    if GEONAMES_VARIANTS_LOADED and geonames_ready:
        return

    geoname_path = Path(geonames_dir) / "cities500.txt"
    if not geoname_path.exists():
        raise FileNotFoundError(f"Could not find geonames file: {geoname_path}")

    geonames_cols = [
        "geoname_id",
        "name",
        "asciiname",
        "alternatenames",
        "latitude",
        "longitude",
        "feature_class",
        "feature_code",
        "country_code",
        "cc2",
        "admin1_code",
        "admin2_code",
        "admin3_code",
        "admin4_code",
        "population",
        "elevation",
        "dem",
        "timezone",
        "modification_date",
    ]

    df = pd.read_csv(
        geoname_path,
        sep="\t",
        header=None,
        names=geonames_cols,
        dtype={"geoname_id": str, "admin1_code": str, "alternatenames": str},
        keep_default_na=True,
    )
    df["admin1_code"] = df["admin1_code"].fillna("").str.upper()

    def _norm(series: pd.Series) -> pd.Series:
        return (
            series.fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    variants = []
    for col in ("name", "asciiname"):
        chunk = df[["geoname_id", "admin1_code", "country_code", col]].rename(columns={col: "variant_raw"})
        variants.append(chunk)
    alt = df.loc[df["alternatenames"].notna(), ["geoname_id", "admin1_code", "country_code", "alternatenames"]].copy()
    if not alt.empty:
        alt["variant_raw"] = alt["alternatenames"].str.split(",")
        alt = alt.explode("variant_raw")
        variants.append(alt[["geoname_id", "admin1_code", "country_code", "variant_raw"]])

    combined = pd.concat(variants, ignore_index=True)
    combined["city_clean"] = _norm(combined["variant_raw"])
    combined = combined[combined["city_clean"].ne("")]
    combined = combined.drop_duplicates(subset=["geoname_id", "admin1_code", "city_clean"])
    combined = combined.rename(columns={"admin1_code": "state_clean"})
    combined["state_clean"] = combined.apply(
        lambda r: r["state_clean"] if r["country_code"] == "US" else "",
        axis=1,
    )
    abbr_to_name = {abbr: name for name, abbr in STATE_NAME_TO_ABBR.items()}
    combined["state_name_clean"] = combined["state_clean"].map(abbr_to_name).fillna("").astype(str)
    combined["state_name_clean"] = (
        combined["state_name_clean"]
        .str.lower()
        .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    country_dict_path = Path(root) / "data" / "crosswalks" / "country_dict.json"
    if country_dict_path.exists():
        try:
            import json
            with country_dict_path.open("r") as fh:
                country_map = json.load(fh)
        except Exception:
            country_map = {}
    else:
        country_map = {}
    combined["country_name_clean"] = combined["country_code"].map(country_map).fillna("").astype(str)
    combined["country_name_clean"] = (
        combined["country_name_clean"]
        .str.lower()
        .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    combined["country_code_clean"] = (
        combined["country_code"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    pop_map = df.drop_duplicates("geoname_id").set_index("geoname_id")["population"]
    combined["pop"] = pd.to_numeric(combined["geoname_id"].map(pop_map), errors="coerce")
    state_tokens = []
    for name, abbr in STATE_NAME_TO_ABBR.items():
        state_tokens.append(
            {
                "geoname_id": f"STATE_{abbr}",
                "state_clean": abbr.lower(),
                "state_name_clean": name.lower(),
            }
        )
    state_tokens_df = pd.DataFrame(state_tokens)
    country_tokens = []
    for code, name in country_map.items():
        if not name:
            continue
        country_tokens.append(
            {
                "geoname_id": f"COUNTRY_{code}",
                "country_code_clean": str(code).lower(),
                "country_name_clean": str(name).lower(),
            }
        )
    country_tokens_df = pd.DataFrame(country_tokens)
    con.register(
        "geonames_variants_df",
        combined[[
            "geoname_id",
            "country_code",
            "country_code_clean",
            "country_name_clean",
            "state_clean",
            "state_name_clean",
            "city_clean",
            "pop",
        ]],
    )
    con.register("geonames_state_tokens_df", state_tokens_df)
    con.register("geonames_country_tokens_df", country_tokens_df)
    con.sql("CREATE OR REPLACE TABLE geonames_variants AS SELECT * FROM geonames_variants_df")
    con.sql("CREATE OR REPLACE TABLE geonames_state_tokens AS SELECT * FROM geonames_state_tokens_df")
    con.sql("CREATE OR REPLACE TABLE geonames_country_tokens AS SELECT * FROM geonames_country_tokens_df")
    GEONAMES_VARIANTS_LOADED = True

def _build_inst_match_views(
    con,
    *,
    left_view: str,
    right_view: str,
    left_id_col: str,
    right_id_col: str,
    left_name_col: str,
    right_name_col: str,
    left_city_col: str | None,
    left_state_col: str | None,
    left_zip_col: str | None,
    right_city_col: str | None,
    right_state_col: str | None,
    right_zip_col: str | None,
    right_alias_col: str | None = None,
    left_geo_flag_col: str | None = None,
    include_geo: bool = True,
    include_city_in_name: bool = True,
    city_in_name_mode: str = "auto",
    suppress_city_in_name_if_geo: bool = False,
    include_subset: bool = True,
    include_jw: bool = True,
    geo_match_priority: str = "high",
    token_fallback: bool = False,
    token_top_n: int = 100,
    token_min_len: int = 3,
    rematch_jw_threshold: float = REMATCH_JW_THRESHOLD,
    prefix: str = "inst",
):
    left_city = _sql_col_or_null(left_city_col)
    left_state = _sql_col_or_null(left_state_col)
    left_zip = _sql_col_or_null(left_zip_col)
    right_city = _sql_col_or_null(right_city_col)
    right_state = _sql_col_or_null(right_state_col)
    right_zip = _sql_col_or_null(right_zip_col)
    right_alias = _sql_col_or_null(right_alias_col)
    left_city_sel = f"l.{left_city_col}" if left_city_col else "NULL"
    left_state_sel = f"l.{left_state_col}" if left_state_col else "NULL"
    left_zip_sel = f"l.{left_zip_col}" if left_zip_col else "NULL"
    left_geo_flag_sel = f"l.{left_geo_flag_col}" if left_geo_flag_col else "FALSE"
    right_city_sel = f"r.{right_city_col}" if right_city_col else "NULL"
    right_state_sel = f"r.{right_state_col}" if right_state_col else "NULL"
    right_zip_sel = f"r.{right_zip_col}" if right_zip_col else "NULL"
    right_alias_sel = f"r.{right_alias_col}" if right_alias_col else "NULL"

    geo_gate = f" AND {left_geo_flag_sel}" if left_geo_flag_col else ""
    zip_match_expr = (
        f"({left_zip} IS NOT NULL AND {right_zip} IS NOT NULL AND {left_zip} = {right_zip}{geo_gate})"
        if include_geo and left_zip_col and right_zip_col
        else "FALSE"
    )
    city_match_expr = (
        f"({left_city} IS NOT NULL AND {right_city} IS NOT NULL AND {left_city} = {right_city}{geo_gate})"
        if include_geo and left_city_col and right_city_col
        else "FALSE"
    )
    state_match_expr = (
        f"({left_state} IS NOT NULL AND {right_state} IS NOT NULL AND {left_state} = {right_state}{geo_gate})"
        if include_geo and left_state_col and right_state_col
        else "FALSE"
    )
    state_pred = state_match_expr if include_geo else "FALSE"

    city_in_name_expr = (
        _sql_city_in_name_expr(
            left_city=left_city_col,
            right_name=right_name_col,
            right_city=right_city_col,
            left_name=left_name_col,
            mode=city_in_name_mode,
        )
        if include_city_in_name
        else "FALSE"
    )
    if suppress_city_in_name_if_geo and left_geo_flag_col:
        city_in_name_expr = f"(CASE WHEN {left_geo_flag_sel} THEN FALSE ELSE {city_in_name_expr} END)"

    subset_expr = (
        f"(CASE "
        f"WHEN {left_name_col} IS NULL OR {right_name_col} IS NULL OR LENGTH({right_name_col}) <= 5 THEN FALSE "
        f"WHEN {left_name_col} NOT LIKE '% %' THEN FALSE "
        f"WHEN POSITION({right_name_col} IN {left_name_col}) > 0 THEN TRUE "
        f"WHEN POSITION({left_name_col} IN {right_name_col}) > 0 THEN TRUE "
        f"ELSE FALSE END)"
        if include_subset
        else "FALSE"
    )

    jw_expr = (
        f"jaro_winkler_similarity({left_name_col}, {right_name_col})"
        if include_jw
        else "NULL"
    )
    jw_filter = "name_jaro_winkler IS NOT NULL" if include_jw else "TRUE"

    state_join = (
        f"AND {left_state} = {right_state}"
        if include_geo and left_state_col and right_state_col
        else ""
    )
    order_geo = "high" if geo_match_priority not in {"high", "low"} else geo_match_priority
    if order_geo == "low":
        order_clause = """
              is_subset_match DESC,
              is_city_in_name DESC,
              name_jaro_winkler DESC,
              is_zip_match DESC,
              is_city_match DESC,
              right_id
        """
    else:
        order_clause = """
              is_subset_match DESC,
              is_zip_match DESC,
              is_city_match DESC,
              is_city_in_name DESC,
              name_jaro_winkler DESC,
              right_id
        """

    token_cte = ""
    token_join = ""
    token_where = ""
    token_only_candidate_generation = bool(token_fallback and order_geo == "low")
    if token_fallback:
        stopwords = ", ".join([f"'{w}'" for w in CITY_STOPWORDS])
        left_no_geo_pred = f"{left_geo_flag_sel} = FALSE" if left_geo_flag_col else f"{left_city_sel} IS NULL AND {left_state_sel} IS NULL"
        token_cte = f"""
      , right_tokens AS (
        SELECT
          r.{right_id_col} AS right_id,
          token
        FROM {right_view} AS r,
        UNNEST(str_split(r.{right_name_col}, ' ')) AS t(token)
        WHERE LENGTH(token) >= {int(token_min_len)}
          AND token NOT IN ({stopwords})
      ),
      token_freq AS (
        SELECT token, COUNT(*) AS cnt
        FROM right_tokens
        GROUP BY token
        ORDER BY cnt DESC
        LIMIT {int(token_top_n)}
      ),
      right_tokens_filt AS (
        SELECT * FROM right_tokens
        WHERE token NOT IN (SELECT token FROM token_freq)
      ),
      left_tokens AS (
        SELECT
          l.{left_id_col} AS left_id,
          token
        FROM f1_rematch AS l,
        UNNEST(str_split(l.{left_name_col}, ' ')) AS t(token)
        WHERE LENGTH(token) >= {int(token_min_len)}
          AND token NOT IN ({stopwords})
      ),
      left_no_geo AS (
        SELECT {left_id_col} AS left_id
        FROM f1_rematch
        WHERE {left_no_geo_pred}
      ),
      token_overlap AS (
        SELECT
          lt.left_id,
          rt.right_id,
          COUNT(DISTINCT lt.token) AS overlap_count
        FROM left_tokens AS lt
        JOIN right_tokens_filt AS rt USING (token)
        JOIN left_no_geo AS ng USING (left_id)
        GROUP BY lt.left_id, rt.right_id
      )
        """
        token_join = f"LEFT JOIN token_overlap AS tovl ON tovl.left_id = l.{left_id_col} AND tovl.right_id = r.{right_id_col}"
        token_where = "OR (tovl.overlap_count IS NOT NULL AND tovl.overlap_count > 0)"

    candidate_where = (
        "(tovl.overlap_count IS NOT NULL AND tovl.overlap_count > 0)"
        if token_only_candidate_generation
        else f"({state_pred} {token_where})"
    )

    con.sql(
    f"""
      CREATE OR REPLACE TABLE {prefix}_raw_matches AS
      WITH raw_match AS (
        SELECT
          l.*,
          {left_city_sel} AS left_city_clean,
          {left_state_sel} AS left_state_clean,
          {left_zip_sel} AS left_zip_clean,
          {left_geo_flag_sel} AS left_geo_extracted,
          r.{right_id_col} AS right_id,
          r.{right_name_col} AS right_name_clean,
          {right_city_sel} AS right_city_clean,
          {right_state_sel} AS right_state_clean,
          {right_zip_sel} AS right_zip_clean,
          {right_alias_sel} AS right_alias,
          CASE
            WHEN {zip_match_expr} THEN 5
            WHEN {city_match_expr} AND {state_match_expr} THEN 4
            WHEN {state_match_expr} THEN 3
            WHEN {right_alias} IS NOT NULL AND {right_alias} = FALSE THEN 2
            WHEN r.{right_id_col} IS NOT NULL THEN 1
            ELSE 0
          END AS match_rank,
          MAX(
            CASE
              WHEN {zip_match_expr} THEN 5
              WHEN {city_match_expr} AND {state_match_expr} THEN 4
              WHEN {state_match_expr} THEN 3
              WHEN {right_alias} IS NOT NULL AND {right_alias} = FALSE THEN 2
              WHEN r.{right_id_col} IS NOT NULL THEN 1
              ELSE 0
            END
          ) OVER (PARTITION BY l.{left_id_col}) AS max_match_rank
        FROM {left_view} AS l
        LEFT JOIN {right_view} AS r
          ON l.{left_name_col} = r.{right_name_col}
      )
      SELECT *,
        COUNT(DISTINCT right_id) OVER (PARTITION BY {left_id_col}) AS match_count
      FROM raw_match
      WHERE match_rank = max_match_rank
    """)

    con.sql(
    f"""
      CREATE OR REPLACE TABLE {prefix}_good_matches AS
      SELECT *
      FROM (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY {left_id_col}
            ORDER BY
              match_rank DESC,
              right_id,
              right_zip_clean,
              right_city_clean,
              right_state_clean,
              right_name_clean
          ) AS good_rn
        FROM {prefix}_raw_matches
        WHERE match_count = 1
      )
      WHERE good_rn = 1
    """)

    con.sql(
    f"""
      CREATE OR REPLACE TABLE {prefix}_second_match_full AS
      WITH unmatched AS (
        SELECT {left_id_col}
        FROM {prefix}_raw_matches
        WHERE match_count = 0
      ),
      f1_rematch AS (
        SELECT l.*
        FROM unmatched
        JOIN {left_view} AS l USING({left_id_col})
      ){token_cte},
      candidate_scores AS (
        SELECT
          l.{left_id_col},
          l.{left_name_col} AS left_name_clean,
          {left_city_sel} AS left_city_clean,
          {left_state_sel} AS left_state_clean,
          {left_zip_sel} AS left_zip_clean,
          r.{right_id_col} AS right_id,
          r.{right_name_col} AS right_name_clean,
          {right_city_sel} AS right_city_clean,
          {right_state_sel} AS right_state_clean,
          {right_zip_sel} AS right_zip_clean,
          {right_alias_sel} AS right_alias,
          CASE WHEN {zip_match_expr} THEN TRUE ELSE FALSE END AS is_zip_match,
          CASE WHEN {city_match_expr} THEN TRUE ELSE FALSE END AS is_city_match,
          {city_in_name_expr} AS is_city_in_name,
          {jw_expr} AS name_jaro_winkler,
          {subset_expr} AS is_subset_match,
          CASE
            WHEN {left_geo_flag_sel} AND ({zip_match_expr} OR {city_match_expr} OR {state_match_expr}) THEN TRUE
            ELSE FALSE
          END AS rev_geo_match,
          (
            CASE
              WHEN {zip_match_expr} THEN 3
              WHEN {city_match_expr} THEN 2
              WHEN {state_match_expr} THEN 1
              ELSE 0
            END + CASE WHEN {subset_expr} THEN 1 ELSE 0 END
          ) AS geo_subset_score
        FROM f1_rematch AS l
        JOIN {right_view} AS r
          ON 1=1
        {token_join}
        WHERE {candidate_where}
      ),
      ranked_candidates AS (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY {left_id_col}
            ORDER BY
            {order_clause}
          ) AS candidate_rank
        FROM candidate_scores
        WHERE {jw_filter}
      )
      SELECT
        *,
        COUNT(DISTINCT right_id) OVER (PARTITION BY {left_id_col}) AS rematch_candidate_count
      FROM ranked_candidates
    """)
    
    con.sql(f"""CREATE OR REPLACE TABLE {prefix}_second_match AS SELECT * FROM {prefix}_second_match_full 
      WHERE (
        (is_subset_match AND right_name_clean ~ '.*\\s.*')
        OR ((is_zip_match OR is_city_match OR is_city_in_name OR left_city_clean IS NULL) AND name_jaro_winkler >= 0.8)
        OR name_jaro_winkler >= {rematch_jw_threshold}
      ) """)

    con.sql(
    f"""
      CREATE OR REPLACE TABLE {prefix}_good_second_matches AS
      SELECT *
      FROM {prefix}_second_match
      WHERE rematch_candidate_count = 1 AND candidate_rank < 10
    """)

    con.sql(
    f"""
      CREATE OR REPLACE TABLE {prefix}_tie_matches AS
      WITH tie_candidates AS (
        SELECT
          *,
          CASE
            WHEN left_zip_clean IS NOT NULL
                 AND right_zip_clean IS NOT NULL
                 AND left_zip_clean = right_zip_clean
            THEN TRUE ELSE FALSE
          END AS is_zip_match,
          CASE
            WHEN left_city_clean IS NOT NULL
                 AND right_city_clean IS NOT NULL
                 AND left_city_clean = right_city_clean
            THEN TRUE ELSE FALSE
          END AS is_city_match
        FROM {prefix}_raw_matches
        WHERE match_count > 1
      ),
      ranked_ties AS (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY {left_id_col}
            ORDER BY
              is_zip_match DESC,
              is_city_match DESC,
              right_id
          ) AS tie_rank,
          COUNT(*) OVER (PARTITION BY {left_id_col}) AS tie_candidate_count
        FROM tie_candidates
      ),
      aggregated_ties AS (
        SELECT
          {left_id_col},
          array_agg(DISTINCT right_id) AS tie_right_ids,
          array_agg(DISTINCT right_name_clean) AS tie_right_names
        FROM ranked_ties
        GROUP BY {left_id_col}
      ),
      fuzzy_ranked AS (
        SELECT
          *,
          ROW_NUMBER() OVER (PARTITION BY {left_id_col} ORDER BY candidate_rank, right_id) AS tie_rank
        FROM {prefix}_second_match
        WHERE rematch_candidate_count > 1
      ),
      fuzzy_aggregated AS (
        SELECT
          {left_id_col},
          array_agg(DISTINCT right_id) AS tie_right_ids,
          array_agg(DISTINCT right_name_clean) AS tie_right_names,
          MAX(rematch_candidate_count) AS tie_candidate_count
        FROM {prefix}_second_match
        WHERE rematch_candidate_count > 1
        GROUP BY {left_id_col}
      )
      SELECT
        ranked_ties.{left_id_col},
        ranked_ties.{left_name_col} AS left_name_clean,
        ranked_ties.right_id,
        ranked_ties.right_name_clean,
        'direct_tie' AS tie_matchtype,
        ranked_ties.tie_candidate_count,
        aggregated_ties.tie_right_ids,
        aggregated_ties.tie_right_names
      FROM ranked_ties
      JOIN aggregated_ties USING ({left_id_col})
      WHERE tie_rank = 1
      UNION ALL
      SELECT
        fuzzy_ranked.{left_id_col},
        fuzzy_ranked.left_name_clean,
        fuzzy_ranked.right_id,
        fuzzy_ranked.right_name_clean,
        'fuzzy_tie' AS tie_matchtype,
        fuzzy_aggregated.tie_candidate_count,
        fuzzy_aggregated.tie_right_ids,
        fuzzy_aggregated.tie_right_names
      FROM fuzzy_ranked
      JOIN fuzzy_aggregated USING ({left_id_col})
      WHERE tie_rank = 1
    """)

    return {
        "raw_matches": f"{prefix}_raw_matches",
        "good_matches": f"{prefix}_good_matches",
        "second_match": f"{prefix}_second_match",
        "good_second_matches": f"{prefix}_good_second_matches",
        "tie_matches": f"{prefix}_tie_matches",
    }

def _create_f1_inst_view(con):
    if _relation_exists(con, "f1_inst"):
        return
    f1_inst = con.sql(
    f"""    
        WITH cleaned AS (
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
        )
        SELECT
          *,
          {_sql_f1_inst_site_id_expr("school_name", "f1_instname_clean", "f1_city_clean", "f1_state_clean", "f1_zip_clean")} AS f1_inst_site_id,
          {_sql_f1_school_id_expr("school_name", "f1_instname_clean")} AS f1_school_id
        FROM cleaned
        WHERE f1_instname_clean <> ''
    """)
    f1_inst.create_view("f1_inst")

def _create_ipeds_inst_view(con):
    if _relation_exists(con, "ipeds_inst"):
        return
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
      ) WHERE ipeds_instname_clean <> '' AND NOT ipeds_instname_clean IN ({",".join([f"'{statename}'" for statename in STATE_NAME_TO_ABBR.keys()])}) AND LENGTH(ipeds_instname_clean) > 1 GROUP BY UNITID, ipeds_instname_clean, ipeds_city_clean, ipeds_state_clean, ipeds_zip_clean, ipeds_alias
    """)
    ipeds_inst.create_view("ipeds_inst")

def _create_f1_inst_crosswalk(con, verbose=False, cache_path=None, load_cache=False, save_output=False):
    """
    Create crosswalk tables for FOIA SEVP to UNITID from IPEDS data.
    """
    if load_cache and cache_path and os.path.exists(cache_path):
        con.sql(f"CREATE OR REPLACE TABLE f1_inst_unitid_crosswalk AS SELECT * FROM read_parquet('{cache_path}')")
        _create_f1_inst_view(con)
        _create_ipeds_inst_view(con)
        if verbose:
            print(f"✅ Loaded FOIA SEVP to IPEDS UNITID crosswalk from '{cache_path}'.")
        return "f1_inst_unitid_crosswalk"

    if verbose:
        _print_section("F-1 Institution Crosswalk")

    _create_f1_inst_view(con)
    _create_ipeds_inst_view(con)

    match_views = _build_inst_match_views(
        con,
        left_view="f1_inst",
        right_view="ipeds_inst",
        left_id_col="f1_row_num",
        right_id_col="UNITID",
        left_name_col="f1_instname_clean",
        right_name_col="ipeds_instname_clean",
        left_city_col="f1_city_clean",
        left_state_col="f1_state_clean",
        left_zip_col="f1_zip_clean",
        right_city_col="ipeds_city_clean",
        right_state_col="ipeds_state_clean",
        right_zip_col="ipeds_zip_clean",
        right_alias_col="ipeds_alias",
        include_geo=True,
        include_city_in_name=True,
        city_in_name_mode="both",
        include_subset=True,
        include_jw=True,
        rematch_jw_threshold=REMATCH_JW_THRESHOLD,
        prefix="f1_ipeds"
    )

    _print_subsection("Institution match summary")
    total_f1_rows = int(_scalar(con, "SELECT COUNT(DISTINCT f1_row_num) AS cnt FROM f1_inst"))
    total_f1_names = int(_scalar(con, "SELECT COUNT(DISTINCT f1_instname_clean) AS cnt FROM f1_inst"))
    print(f"Total F-1 institution x ZIPs: {total_f1_rows}")
    print(f"Total unique F-1 institution names: {total_f1_names}")

    con.sql(f"CREATE OR REPLACE VIEW good_matches AS SELECT * FROM {match_views['good_matches']}")
    good_match_rows = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM good_matches"))
    good_match_names = int(_scalar(con, "SELECT COUNT(DISTINCT f1_instname_clean) AS cnt FROM good_matches"))
    print(
        f"F-1 institution x ZIPs uniquely matched: {good_match_rows} "
        f"({(good_match_rows / total_f1_rows * 100) if total_f1_rows else 0:.2f}%)"
    )
    print(
        f"F-1 institution names uniquely matched: {good_match_names} "
        f"({(good_match_names / total_f1_names * 100) if total_f1_names else 0:.2f}%)"
    )
    match_quality_df = con.sql(
        """
        SELECT match_rank, match_quality, COUNT(*) AS count
        FROM (
          SELECT
            match_rank,
            CASE
              WHEN match_rank = 5 THEN 'ZIP'
              WHEN match_rank = 4 THEN 'city + state'
              WHEN match_rank = 3 THEN 'state only'
              WHEN match_rank = 2 THEN 'none, not matched on alias'
              WHEN match_rank = 1 THEN 'none'
              ELSE 'unknown'
            END AS match_quality
          FROM good_matches
        )
        GROUP BY match_rank, match_quality
        ORDER BY match_rank
        """
    ).df()
    print(f"Count by location match quality: \n {match_quality_df}")

    con.sql(f"CREATE OR REPLACE VIEW good_second_matches AS SELECT * FROM {match_views['good_second_matches']}")
    good_second_rows = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM good_second_matches"))
    good_second_names = int(_scalar(con, "SELECT COUNT(DISTINCT left_name_clean) AS cnt FROM good_second_matches"))
    print(
        f"Second match unique F-1 institution x ZIPs: {good_second_rows} "
        f"(+{(good_second_rows / total_f1_rows * 100) if total_f1_rows else 0:.2f}%)"
    )
    print(
        f"Second match unique F-1 institution names: {good_second_names} "
        f"(+{(good_second_names / total_f1_names * 100) if total_f1_names else 0:.2f}%)"
    )

    con.sql(f"CREATE OR REPLACE VIEW tie_matches AS SELECT * FROM {match_views['tie_matches']}")
    tie_count = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM tie_matches"))
    if tie_count == 0:
        print("Ambiguous matches (direct or fuzzy): 0 institutions with equal top ranks")
    else:
        avg_candidates = float(_scalar(con, "SELECT AVG(tie_candidate_count) AS avg_cnt FROM tie_matches"))
        print(
            f"Ambiguous matches (direct or fuzzy): {tie_count} institutions with multiple best UNITIDs "
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
        WHEN match1.UNITID IS NOT NULL THEN match1.UNITID
        WHEN match2.UNITID IS NOT NULL THEN match2.UNITID
        WHEN match3.UNITID IS NOT NULL THEN match3.UNITID
        ELSE NULL
      END AS main_unitid,
      CASE 
        WHEN match1.ipeds_instname_clean IS NOT NULL THEN match1.ipeds_instname_clean
        WHEN match2.ipeds_instname_clean IS NOT NULL THEN match2.ipeds_instname_clean
        WHEN match3.ipeds_instname_clean IS NOT NULL THEN match3.ipeds_instname_clean
        ELSE NULL
      END AS ipeds_instname_clean,
      CASE
        WHEN match1.matchtype = 'direct' THEN 'direct'
        WHEN match3.matchtype = 'direct_tie' THEN 'direct_tie'
        WHEN match2.matchtype = 'fuzzy' THEN 'fuzzy'
        WHEN match3.matchtype = 'fuzzy_tie' THEN 'fuzzy_tie'
        ELSE 'none'
      END AS matchtype,
      match3.tie_candidate_count,
      match3.tie_unitids,
      match3.tie_instnames
    FROM f1_inst
    LEFT JOIN (
      SELECT f1_row_num, right_id AS UNITID, right_name_clean AS ipeds_instname_clean, 'direct' AS matchtype
      FROM good_matches
    ) AS match1 USING (f1_row_num)
    LEFT JOIN (
      SELECT f1_row_num, right_id AS UNITID, right_name_clean AS ipeds_instname_clean, 'fuzzy' AS matchtype
      FROM good_second_matches
    ) AS match2 USING (f1_row_num)
    LEFT JOIN (
      SELECT
        f1_row_num,
        right_id AS UNITID,
        right_name_clean AS ipeds_instname_clean,
        tie_candidate_count,
        tie_right_ids AS tie_unitids,
        tie_right_names AS tie_instnames,
        tie_matchtype AS matchtype
      FROM tie_matches
    ) AS match3 USING (f1_row_num)
    """)
    
    matched_rows = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM f1_inst_unitid_crosswalk WHERE UNITID IS NOT NULL"))
    print(
        f"Final crosswalk: {matched_rows} F-1 institution x ZIPs matched to UNITID. "
        f"({(matched_rows / total_f1_rows * 100) if total_f1_rows else 0:.2f}%)"
    )
    match_type_df = con.sql(
        "SELECT matchtype, COUNT(*) AS count FROM f1_inst_unitid_crosswalk GROUP BY matchtype ORDER BY count DESC"
    ).df()
    print(f"Match Type Distribution: \n {match_type_df}")
    
    if save_output and cache_path:
        con.sql(f"COPY (SELECT * FROM f1_inst_unitid_crosswalk) TO '{cache_path}' (FORMAT PARQUET)")
        if verbose:
            print(f"💾 Saved FOIA SEVP to IPEDS UNITID crosswalk to '{cache_path}'.")
    if verbose:
        print("✅ Created FOIA SEVP to IPEDS UNITID crosswalk table 'f1_inst_unitid_crosswalk'.")
    return "f1_inst_unitid_crosswalk"

def _create_revelio_inst_view(con):
    if _relation_exists(con, "revelio_inst"):
        return
    if GEONAMES_DIR_PATH:
        _ensure_geonames_variants(con, GEONAMES_DIR_PATH)
    rev_loc_expr = _sql_normalize(
        "REGEXP_EXTRACT(lower(university_raw), '^(?:.*?)(?:-|,|\\\\bin\\\\b|\\\\bat\\\\b|;)(.*)$', 1)"
    )
    query = f"""
    CREATE OR REPLACE VIEW revelio_inst_base AS
    WITH cleaned AS (
    SELECT
      university_raw,
      md5(university_raw) AS rev_key,
      {_sql_revelio_inst_raw_id_expr("university_raw")} AS revelio_inst_raw_id,
      {_sql_clean_inst_name("university_raw")} AS rev_instname_clean,
      {_sql_revelio_inst_family_id_expr(_sql_clean_inst_name("university_raw"))} AS revelio_inst_family_id,
      {rev_loc_expr} AS rev_loc_clean
    FROM (
      SELECT DISTINCT university_raw
      FROM wrds_users_filtered
      WHERE university_raw IS NOT NULL
        AND university_raw IN (SELECT university_raw FROM revelio_institutions)
    )
    )
    SELECT *
    FROM cleaned
    WHERE rev_instname_clean <> ''
    """
    con.sql(query)

    if GEONAMES_DIR_PATH:
        con.sql(
        f"""
        CREATE OR REPLACE VIEW revelio_inst_geo AS
        WITH rev_tokens AS (
          SELECT
            rev_key,
            rev_loc_clean,
            token
          FROM revelio_inst_base,
          UNNEST(str_split(rev_loc_clean, ' ')) AS t(token)
          WHERE rev_loc_clean IS NOT NULL AND rev_loc_clean <> '' AND LENGTH(token) >= 3
            AND token NOT IN ({", ".join([f"'{w}'" for w in CITY_STOPWORDS])})
        ),
        geo_tokens AS (
          SELECT geoname_id, city_clean, state_clean, state_name_clean, country_name_clean, country_code_clean, pop, token
          FROM geonames_variants,
          UNNEST(str_split(city_clean, ' ')) AS t(token)
          WHERE city_clean IS NOT NULL AND LENGTH(token) >= 3
            AND token NOT IN ({", ".join([f"'{w}'" for w in CITY_STOPWORDS])})
          UNION ALL
          SELECT geoname_id, NULL AS city_clean, state_clean, state_name_clean, NULL AS country_name_clean, NULL AS country_code_clean, NULL AS pop, token
          FROM geonames_state_tokens,
          UNNEST(str_split(state_name_clean, ' ')) AS t(token)
          WHERE state_name_clean IS NOT NULL AND LENGTH(token) >= 3
          UNION ALL
          SELECT geoname_id, NULL AS city_clean, state_clean, state_name_clean, NULL AS country_name_clean, NULL AS country_code_clean, NULL AS pop, token
          FROM geonames_state_tokens,
          UNNEST(str_split(state_clean, ' ')) AS t(token)
          WHERE state_clean IS NOT NULL AND LENGTH(token) >= 2
          UNION ALL
          SELECT geoname_id, NULL AS city_clean, NULL AS state_clean, NULL AS state_name_clean, country_name_clean, country_code_clean, NULL AS pop, token
          FROM geonames_country_tokens,
          UNNEST(str_split(country_name_clean, ' ')) AS t(token)
          WHERE country_name_clean IS NOT NULL AND LENGTH(token) >= 3
            AND token NOT IN ({", ".join([f"'{w}'" for w in CITY_STOPWORDS])})
        ),
        token_matches AS (
          SELECT DISTINCT
            r.rev_key,
            r.rev_loc_clean,
            g.geoname_id,
            g.city_clean,
            g.state_clean,
            g.state_name_clean,
            g.country_name_clean,
            g.country_code_clean,
            g.pop
          FROM rev_tokens AS r
          JOIN geo_tokens AS g USING (token)
        ),
        ranked AS (
          SELECT
            *,
            CASE
              WHEN city_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(city_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)') THEN 1
              WHEN state_name_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(state_name_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)') THEN 2
              WHEN state_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(state_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)') THEN 3
              WHEN country_name_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(country_name_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)') THEN 4
              ELSE 99
            END AS match_priority,
            ROW_NUMBER() OVER (
              PARTITION BY rev_key
              ORDER BY
                CASE
                  WHEN city_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(city_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)') THEN 1
                  WHEN state_name_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(state_name_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)') THEN 2
                  WHEN state_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(state_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)') THEN 3
                  WHEN country_name_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(country_name_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)') THEN 4
                  ELSE 99
                END,
                pop DESC NULLS LAST,
                LENGTH(city_clean) DESC
            ) AS rn
          FROM token_matches
          WHERE
            (city_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(city_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)'))
            OR (state_name_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(state_name_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)'))
            OR (state_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(state_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)'))
            OR (country_name_clean <> '' AND regexp_matches(rev_loc_clean, '(?:^|\\s)' || regexp_replace(country_name_clean, ' ', '\\\\s+', 'g') || '(?:\\s|$)'))
        )
        SELECT
          r.university_raw,
          r.rev_key,
          r.revelio_inst_raw_id,
          r.rev_instname_clean,
          r.revelio_inst_family_id,
          m.geoname_id AS rev_geoname_id,
          m.city_clean AS rev_city_clean,
          m.state_clean AS rev_state_clean,
          CASE WHEN m.country_name_clean IS NOT NULL AND NOT m.country_name_clean = 'united states' THEN TRUE ELSE FALSE END AS rev_non_us_geo,
          CASE WHEN m.geoname_id IS NOT NULL THEN TRUE ELSE FALSE END AS rev_geo_matched
        FROM revelio_inst_base AS r
        LEFT JOIN ranked AS m
          ON r.rev_key = m.rev_key AND m.rn = 1
        """
        )
        con.sql("CREATE OR REPLACE VIEW revelio_inst AS SELECT * FROM revelio_inst_geo")
    else:
        con.sql(
            "CREATE OR REPLACE VIEW revelio_inst AS "
            "SELECT university_raw, rev_key, revelio_inst_raw_id, rev_instname_clean, revelio_inst_family_id, "
            "NULL AS rev_geoname_id, NULL AS rev_city_clean, NULL AS rev_state_clean, FALSE AS rev_geo_matched "
            "FROM revelio_inst_base"
        )

def _create_revelio_ipeds_inst_crosswalk(con, verbose=False, cache_path=None, save_output=False):
    if verbose:
        _print_section("Revelio ↔ IPEDS Institution Crosswalk")

    _create_revelio_inst_view(con)
    _create_ipeds_inst_view(con)

    match_views = _build_inst_match_views(
        con,
        left_view="revelio_inst",
        right_view="ipeds_inst",
        left_id_col="rev_key",
        right_id_col="UNITID",
        left_name_col="rev_instname_clean",
        right_name_col="ipeds_instname_clean",
        left_city_col="rev_city_clean",
        left_state_col="rev_state_clean",
        left_zip_col=None,
        right_city_col="ipeds_city_clean",
        right_state_col="ipeds_state_clean",
        right_zip_col="ipeds_zip_clean",
        right_alias_col="ipeds_alias",
        left_geo_flag_col="rev_geo_matched",
        include_geo=True,
        include_city_in_name=True,
        city_in_name_mode="right_in_left",
        suppress_city_in_name_if_geo=True,
        include_subset=True,
        include_jw=True,
        geo_match_priority="low",
        token_fallback=True,
        token_top_n=100,
        rematch_jw_threshold=REMATCH_JW_THRESHOLD,
        prefix="rev_ipeds",
    )

    if verbose:
        total_rows = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM revelio_inst"))
        direct_cnt = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM rev_ipeds_good_matches"))
        fuzzy_cnt = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM rev_ipeds_good_second_matches"))
        tie_cnt = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM rev_ipeds_tie_matches"))
        print(f"Revelio institutions: {total_rows:,}")
        print(f"Direct matches: {direct_cnt:,} | Fuzzy matches: {fuzzy_cnt:,} | Ties: {tie_cnt:,}")

    con.sql(
    """
    CREATE OR REPLACE TABLE revelio_ipeds_inst_crosswalk AS
    SELECT
      rev.university_raw,
      rev.rev_key,
      rev.revelio_inst_raw_id,
      rev.rev_instname_clean,
      rev.revelio_inst_family_id,
      rev.rev_city_clean,
      rev.rev_state_clean,
      rev.rev_geo_matched,
      CASE
        WHEN match1.UNITID IS NOT NULL THEN match1.UNITID
        WHEN match2.UNITID IS NOT NULL THEN match2.UNITID
        WHEN match3.UNITID IS NOT NULL THEN match3.UNITID
        ELSE NULL
      END AS UNITID,
      CASE
        WHEN match1.UNITID IS NOT NULL THEN match1.UNITID
        WHEN match2.UNITID IS NOT NULL THEN match2.UNITID
        WHEN match3.UNITID IS NOT NULL THEN match3.UNITID
        ELSE NULL
      END AS main_unitid,
      CASE 
        WHEN match1.ipeds_instname_clean IS NOT NULL THEN match1.ipeds_instname_clean
        WHEN match2.ipeds_instname_clean IS NOT NULL THEN match2.ipeds_instname_clean
        WHEN match3.ipeds_instname_clean IS NOT NULL THEN match3.ipeds_instname_clean
        ELSE NULL
      END AS ipeds_instname_clean,
      CASE
        WHEN match1.matchtype = 'direct' THEN 'direct'
        WHEN match3.matchtype = 'direct_tie' THEN 'direct_tie'
        WHEN match2.matchtype = 'fuzzy' THEN 'fuzzy'
        WHEN match3.matchtype = 'fuzzy_tie' THEN 'fuzzy_tie'
        ELSE 'none'
      END AS matchtype,
      match3.tie_candidate_count,
      match3.tie_unitids,
      match3.tie_instnames
    FROM revelio_inst AS rev
    LEFT JOIN (
      SELECT rev_key, right_id AS UNITID, right_name_clean AS ipeds_instname_clean, 'direct' AS matchtype
      FROM rev_ipeds_good_matches
    ) AS match1 USING (rev_key)
    LEFT JOIN (
      SELECT rev_key, right_id AS UNITID, right_name_clean AS ipeds_instname_clean, 'fuzzy' AS matchtype
      FROM rev_ipeds_good_second_matches
    ) AS match2 USING (rev_key)
    LEFT JOIN (
      SELECT
        rev_key,
        right_id AS UNITID,
        right_name_clean AS ipeds_instname_clean,
        tie_candidate_count,
        tie_right_ids AS tie_unitids,
        tie_right_names AS tie_instnames,
        tie_matchtype AS matchtype
      FROM rev_ipeds_tie_matches
    ) AS match3 USING (rev_key)
    """)

    if save_output and cache_path:
        con.sql(f"COPY (SELECT * FROM revelio_ipeds_inst_crosswalk) TO '{cache_path}' (FORMAT PARQUET)")
        if verbose:
            print(f"💾 Saved Revelio ↔ IPEDS crosswalk to '{cache_path}'.")
    if verbose:
        print("✅ Created Revelio ↔ IPEDS crosswalk table 'revelio_ipeds_inst_crosswalk'.")
    return "revelio_ipeds_inst_crosswalk"

def _create_revelio_foia_inst_crosswalk(con, verbose=False, cache_path=None, save_output=False):
    if verbose:
        _print_section("Revelio ↔ FOIA Institution Crosswalk")

    _create_revelio_inst_view(con)

    match_views = _build_inst_match_views(
        con,
        left_view="revelio_inst",
        right_view="f1_inst",
        left_id_col="rev_key",
        right_id_col="f1_row_num",
        left_name_col="rev_instname_clean",
        right_name_col="f1_instname_clean",
        left_city_col="rev_city_clean",
        left_state_col="rev_state_clean",
        left_zip_col=None,
        right_city_col="f1_city_clean",
        right_state_col="f1_state_clean",
        right_zip_col="f1_zip_clean",
        right_alias_col=None,
        left_geo_flag_col="rev_geo_matched",
        include_geo=True,
        include_city_in_name=True,
        city_in_name_mode="right_in_left",
        suppress_city_in_name_if_geo=True,
        include_subset=True,
        include_jw=True,
        geo_match_priority="low",
        token_fallback=True,
        token_top_n=100,
        rematch_jw_threshold=REMATCH_JW_THRESHOLD,
        prefix="rev_foia",
    )

    if verbose:
        total_rows = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM revelio_inst"))
        direct_cnt = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM rev_foia_good_matches"))
        fuzzy_cnt = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM rev_foia_good_second_matches"))
        tie_cnt = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM rev_foia_tie_matches"))
        print(f"Revelio institutions: {total_rows:,}")
        print(f"Direct matches: {direct_cnt:,} | Fuzzy matches: {fuzzy_cnt:,} | Ties: {tie_cnt:,}")

    con.sql(
    """
    CREATE OR REPLACE TABLE revelio_foia_inst_crosswalk AS
    SELECT
      rev.university_raw,
      rev.rev_key,
      rev.revelio_inst_raw_id,
      rev.rev_instname_clean,
      rev.revelio_inst_family_id,
      rev.rev_city_clean,
      rev.rev_state_clean,
      rev.rev_geo_matched,
      CASE
        WHEN match1.f1_row_num IS NOT NULL THEN match1.f1_row_num
        WHEN match2.f1_row_num IS NOT NULL THEN match2.f1_row_num
        WHEN match3.f1_row_num IS NOT NULL THEN match3.f1_row_num
        ELSE NULL
      END AS f1_row_num,
      CASE
        WHEN match1.f1_row_num IS NOT NULL THEN match1.f1_inst_site_id
        WHEN match2.f1_row_num IS NOT NULL THEN match2.f1_inst_site_id
        WHEN match3.f1_row_num IS NOT NULL THEN match3.f1_inst_site_id
        ELSE NULL
      END AS f1_inst_site_id,
      CASE
        WHEN match1.f1_row_num IS NOT NULL THEN match1.f1_school_id
        WHEN match2.f1_row_num IS NOT NULL THEN match2.f1_school_id
        WHEN match3.f1_row_num IS NOT NULL THEN match3.f1_school_id
        ELSE NULL
      END AS f1_school_id,
      CASE 
        WHEN match1.f1_instname_clean IS NOT NULL THEN match1.f1_instname_clean
        WHEN match2.f1_instname_clean IS NOT NULL THEN match2.f1_instname_clean
        WHEN match3.f1_instname_clean IS NOT NULL THEN match3.f1_instname_clean
        ELSE NULL
      END AS f1_instname_clean,
      CASE
        WHEN match1.matchtype = 'direct' THEN 'direct'
        WHEN match3.matchtype = 'direct_tie' THEN 'direct_tie'
        WHEN match2.matchtype = 'fuzzy' THEN 'fuzzy'
        WHEN match3.matchtype = 'fuzzy_tie' THEN 'fuzzy_tie'
        ELSE 'none'
      END AS matchtype,
      match3.tie_candidate_count,
      match3.tie_f1_row_nums,
      match3.tie_f1_instnames
    FROM revelio_inst AS rev
    LEFT JOIN (
      SELECT
        rev_key,
        right_id AS f1_row_num,
        r.f1_inst_site_id,
        r.f1_school_id,
        right_name_clean AS f1_instname_clean,
        'direct' AS matchtype
      FROM rev_foia_good_matches
      LEFT JOIN f1_inst AS r
        ON rev_foia_good_matches.right_id = r.f1_row_num
    ) AS match1 USING (rev_key)
    LEFT JOIN (
      SELECT
        rev_key,
        right_id AS f1_row_num,
        r.f1_inst_site_id,
        r.f1_school_id,
        right_name_clean AS f1_instname_clean,
        'fuzzy' AS matchtype
      FROM rev_foia_good_second_matches
      LEFT JOIN f1_inst AS r
        ON rev_foia_good_second_matches.right_id = r.f1_row_num
    ) AS match2 USING (rev_key)
    LEFT JOIN (
      SELECT
        rev_key,
        right_id AS f1_row_num,
        r.f1_inst_site_id,
        r.f1_school_id,
        right_name_clean AS f1_instname_clean,
        tie_candidate_count,
        tie_right_ids AS tie_f1_row_nums,
        tie_right_names AS tie_f1_instnames,
        tie_matchtype AS matchtype
      FROM rev_foia_tie_matches
      LEFT JOIN f1_inst AS r
        ON rev_foia_tie_matches.right_id = r.f1_row_num
    ) AS match3 USING (rev_key)
    """)

    if save_output and cache_path:
        con.sql(f"COPY (SELECT * FROM revelio_foia_inst_crosswalk) TO '{cache_path}' (FORMAT PARQUET)")
        if verbose:
            print(f"💾 Saved Revelio ↔ FOIA crosswalk to '{cache_path}'.")
    if verbose:
        print("✅ Created Revelio ↔ FOIA crosswalk table 'revelio_foia_inst_crosswalk'.")
    return "revelio_foia_inst_crosswalk"

def _create_three_way_inst_crosswalk(
    con,
    verbose=False,
    cache_path=None,
    matched_university_path=None,
    save_output=False,
):
    if verbose:
        _print_section("Three-Way Institution Crosswalk (Revelio ↔ IPEDS ↔ FOIA)")

    _create_revelio_inst_view(con)
    _create_f1_inst_view(con)
    _create_ipeds_inst_view(con)

    con.sql(
    """
    CREATE OR REPLACE TABLE revelio_ipeds_foia_inst_crosswalk AS
    WITH
    rev_ipeds AS (
      SELECT
        rev.rev_key,
        rev.university_raw,
        rev.rev_instname_clean,
        cw.UNITID,
        cw.matchtype AS rev_ipeds_matchtype,
        cw.ipeds_instname_clean AS ipeds_instname_clean
      FROM revelio_inst AS rev
      LEFT JOIN revelio_ipeds_inst_crosswalk AS cw USING (rev_key)
    ),
    rev_foia AS (
      SELECT
        rev.rev_key,
        rev.university_raw,
        rev.rev_instname_clean,
        cw.f1_row_num,
        cw.matchtype AS rev_foia_matchtype,
        cw.f1_instname_clean AS f1_instname_clean
      FROM revelio_inst AS rev
      LEFT JOIN revelio_foia_inst_crosswalk AS cw USING (rev_key)
    ),
    rev_ipeds_with_foia AS (
      SELECT
        r.university_raw,
        r.rev_instname_clean,
        r.UNITID,
        fi.f1_row_num,
        r.rev_ipeds_matchtype AS rev_matchtype,
        'rev_ipeds' AS rev_match_source,
        r.ipeds_instname_clean,
        fi.f1_instname_clean,
        fi.f1_city_clean,
        fi.f1_state_clean,
        fi.f1_zip_clean,
        ip.ipeds_city_clean,
        ip.ipeds_state_clean,
        ip.ipeds_zip_clean,
        COALESCE(sm.is_subset_match, FALSE) AS subset_match,
        COALESCE(sm.is_city_in_name, FALSE) AS city_in_name,
        sm.name_jaro_winkler AS jw_score
      FROM rev_ipeds AS r
      LEFT JOIN f1_inst_unitid_crosswalk AS fi
        ON r.UNITID = fi.UNITID
      LEFT JOIN ipeds_inst AS ip
        ON r.UNITID = ip.UNITID
      LEFT JOIN rev_ipeds_second_match AS sm
        ON sm.rev_key = r.rev_key
       AND sm.right_id = r.UNITID
    ),
    rev_foia_with_ipeds AS (
      SELECT
        r.university_raw,
        r.rev_instname_clean,
        fi.UNITID,
        r.f1_row_num,
        r.rev_foia_matchtype AS rev_matchtype,
        'rev_foia' AS rev_match_source,
        ip.ipeds_instname_clean,
        r.f1_instname_clean,
        fi.f1_city_clean,
        fi.f1_state_clean,
        fi.f1_zip_clean,
        ip.ipeds_city_clean,
        ip.ipeds_state_clean,
        ip.ipeds_zip_clean,
        COALESCE(sm.is_subset_match, FALSE) AS subset_match,
        COALESCE(sm.is_city_in_name, FALSE) AS city_in_name,
        sm.name_jaro_winkler AS jw_score
      FROM rev_foia AS r
      LEFT JOIN f1_inst_unitid_crosswalk AS fi
        ON r.f1_row_num = fi.f1_row_num
      LEFT JOIN ipeds_inst AS ip
        ON fi.UNITID = ip.UNITID
      LEFT JOIN rev_foia_second_match AS sm
        ON sm.rev_key = r.rev_key
       AND sm.right_id = r.f1_row_num
    ),
    candidates AS (
      SELECT * FROM rev_ipeds_with_foia
      UNION ALL
      SELECT * FROM rev_foia_with_ipeds
    ),
    scored AS (
      SELECT
        *,
        CASE
          WHEN rev_matchtype = 'direct' THEN 4
          WHEN rev_matchtype = 'direct_tie' THEN 3
          WHEN rev_matchtype = 'fuzzy' THEN 2
          WHEN rev_matchtype = 'fuzzy_tie' THEN 1
          ELSE 0
        END AS match_group,
        CASE
          WHEN f1_zip_clean IS NOT NULL AND ipeds_zip_clean IS NOT NULL AND f1_zip_clean = ipeds_zip_clean THEN 1
          ELSE 0
        END AS zip_match,
        CASE
          WHEN f1_city_clean IS NOT NULL AND ipeds_city_clean IS NOT NULL AND f1_city_clean = ipeds_city_clean THEN 1
          ELSE 0
        END AS city_match,
        CASE
          WHEN f1_state_clean IS NOT NULL AND ipeds_state_clean IS NOT NULL AND f1_state_clean = ipeds_state_clean THEN 1
          ELSE 0
        END AS state_match,
        COALESCE(jw_score, 0.0) AS jw_score_filled
      FROM candidates
    ),
    ranked AS (
      SELECT
        *,
        ROW_NUMBER() OVER (
          PARTITION BY rev_instname_clean
          ORDER BY
            match_group DESC,
            zip_match DESC,
            city_match DESC,
            subset_match DESC,
            city_in_name DESC,
            state_match DESC,
            jw_score_filled DESC
        ) AS rn
      FROM scored
    )
    SELECT
      university_raw,
      rev_instname_clean,
      UNITID,
      f1_row_num,
      ipeds_instname_clean,
      f1_instname_clean,
      rev_match_source,
      rev_matchtype,
      match_group,
      subset_match AS rev_subset_match,
      city_in_name AS rev_city_in_name,
      jw_score AS rev_jw_score
    FROM ranked
    WHERE rn = 1
    """)
    con.sql(
        """
        CREATE OR REPLACE TABLE revelio_matched_university_raws AS
        SELECT DISTINCT university_raw
        FROM revelio_ipeds_foia_inst_crosswalk
        WHERE university_raw IS NOT NULL
          AND TRIM(university_raw) <> ''
          AND (UNITID IS NOT NULL OR f1_row_num IS NOT NULL)
        """
    )

    if verbose:
        total_rev = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM revelio_inst"))
        matched_rev = int(_scalar(con, "SELECT COUNT(*) AS cnt FROM revelio_ipeds_foia_inst_crosswalk WHERE UNITID IS NOT NULL OR f1_row_num IS NOT NULL"))
        print(f"Revelio institutions: {total_rev:,}")
        print(f"Matched to any institution: {matched_rev:,} ({(matched_rev / total_rev * 100) if total_rev else 0:.2f}%)")

    if save_output and cache_path:
        con.sql(f"COPY (SELECT * FROM revelio_ipeds_foia_inst_crosswalk) TO '{cache_path}' (FORMAT PARQUET)")
        if verbose:
            print(f"💾 Saved three-way institution crosswalk to '{cache_path}'.")
    if save_output and matched_university_path:
        con.sql(f"COPY (SELECT * FROM revelio_matched_university_raws) TO '{matched_university_path}' (FORMAT PARQUET)")
        if verbose:
            print(f"💾 Saved matched university_raw list to '{matched_university_path}'.")
    if verbose:
        print("✅ Created three-way institution crosswalk table 'revelio_ipeds_foia_inst_crosswalk'.")
    return "revelio_ipeds_foia_inst_crosswalk"


def _save_relation_parquet(con, relation_name: str, cache_path: str | None, verbose: bool = False, label: str | None = None):
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    con.sql(f"COPY (SELECT * FROM {relation_name}) TO '{cache_path}' (FORMAT PARQUET)")
    if verbose:
        rel_label = label or relation_name
        print(f"💾 Saved {rel_label} to '{cache_path}'.")


def _create_f1_ipeds_source_view(con):
    site_id_expr = (
        "f1_inst_site_id"
        if _relation_has_column(con, "f1_inst_unitid_crosswalk", "f1_inst_site_id")
        else _sql_f1_inst_site_id_expr("school_name", "f1_instname_clean", "f1_city_clean", "f1_state_clean", "f1_zip_clean")
    )
    school_id_expr = (
        "f1_school_id"
        if _relation_has_column(con, "f1_inst_unitid_crosswalk", "f1_school_id")
        else _sql_f1_school_id_expr("school_name", "f1_instname_clean")
    )
    main_unitid_expr = "main_unitid" if _relation_has_column(con, "f1_inst_unitid_crosswalk", "main_unitid") else "UNITID"
    tie_candidate_count_expr = (
        "tie_candidate_count" if _relation_has_column(con, "f1_inst_unitid_crosswalk", "tie_candidate_count") else "NULL::BIGINT"
    )
    tie_unitids_expr = "tie_unitids" if _relation_has_column(con, "f1_inst_unitid_crosswalk", "tie_unitids") else "NULL"
    tie_instnames_expr = "tie_instnames" if _relation_has_column(con, "f1_inst_unitid_crosswalk", "tie_instnames") else "NULL"
    con.sql(
        f"""
        CREATE OR REPLACE VIEW f1_ipeds_source AS
        SELECT
          {site_id_expr} AS f1_inst_site_id,
          {school_id_expr} AS f1_school_id,
          school_name,
          f1_row_num,
          f1_instname_clean,
          f1_city_clean,
          f1_state_clean,
          f1_zip_clean,
          CAST({main_unitid_expr} AS BIGINT) AS main_unitid,
          ipeds_instname_clean,
          matchtype,
          {tie_candidate_count_expr} AS tie_candidate_count,
          {tie_unitids_expr} AS tie_unitids,
          {tie_instnames_expr} AS tie_instnames
        FROM f1_inst_unitid_crosswalk
        """
    )


def _create_revelio_ipeds_source_view(con):
    raw_id_expr = (
        "r.revelio_inst_raw_id"
        if _relation_has_column(con, "revelio_ipeds_inst_crosswalk", "revelio_inst_raw_id")
        else _sql_revelio_inst_raw_id_expr("r.university_raw")
    )
    family_id_expr = (
        "r.revelio_inst_family_id"
        if _relation_has_column(con, "revelio_ipeds_inst_crosswalk", "revelio_inst_family_id")
        else _sql_revelio_inst_family_id_expr("r.rev_instname_clean")
    )
    main_unitid_expr = "r.main_unitid" if _relation_has_column(con, "revelio_ipeds_inst_crosswalk", "main_unitid") else "r.UNITID"
    tie_candidate_count_expr = (
        "r.tie_candidate_count" if _relation_has_column(con, "revelio_ipeds_inst_crosswalk", "tie_candidate_count") else "NULL::BIGINT"
    )
    tie_unitids_expr = "r.tie_unitids" if _relation_has_column(con, "revelio_ipeds_inst_crosswalk", "tie_unitids") else "NULL"
    tie_instnames_expr = "r.tie_instnames" if _relation_has_column(con, "revelio_ipeds_inst_crosswalk", "tie_instnames") else "NULL"
    users_join = ""
    users_select = "NULL::BIGINT AS n_revelio_users"
    if _relation_exists(con, "revelio_institutions"):
        users_join = "LEFT JOIN revelio_institutions AS ri ON r.university_raw = ri.university_raw"
        users_select = "TRY_CAST(ri.n AS BIGINT) AS n_revelio_users"
    con.sql(
        f"""
        CREATE OR REPLACE VIEW revelio_ipeds_source AS
        SELECT
          {raw_id_expr} AS revelio_inst_raw_id,
          {family_id_expr} AS revelio_inst_family_id,
          r.university_raw,
          r.rev_key,
          r.rev_instname_clean,
          r.rev_city_clean,
          r.rev_state_clean,
          COALESCE(r.rev_geo_matched, FALSE) AS rev_geo_matched,
          CAST({main_unitid_expr} AS BIGINT) AS main_unitid,
          r.ipeds_instname_clean,
          r.matchtype,
          {tie_candidate_count_expr} AS tie_candidate_count,
          {tie_unitids_expr} AS tie_unitids,
          {tie_instnames_expr} AS tie_instnames,
          {users_select}
        FROM revelio_ipeds_inst_crosswalk AS r
        {users_join}
        """
    )


def _create_ipeds_main_institutions(con, verbose=False, cache_path=None, save_output=False):
    source_priority_expr = """
        CASE source
            WHEN 'IPEDSINSTNM' THEN 1
            WHEN 'INSTNM' THEN 2
            WHEN 'PEPSSCHNAME' THEN 3
            WHEN 'PEPSLOCNAME' THEN 4
            ELSE 5
        END
    """
    con.sql(
        f"""
        CREATE OR REPLACE TABLE ipeds_main_institutions AS
        WITH base AS (
          SELECT
            TRY_CAST(UNITID AS BIGINT) AS main_unitid,
            instname,
            instname_raw,
            source,
            COALESCE(ALIAS, FALSE) AS alias_ind,
            {_sql_clean_inst_name("instname")} AS ipeds_instname_clean,
            {_sql_normalize("CITY")} AS ipeds_city_clean,
            {_sql_state_name_to_abbr("STABBR")} AS ipeds_state_clean,
            {_sql_clean_zip("ZIP")} AS ipeds_zip_clean
          FROM ipeds_crosswalk
          WHERE UNITID IS NOT NULL
            AND TRY_CAST(UNITID AS BIGINT) IS NOT NULL
            AND instname IS NOT NULL
        ),
        ranked AS (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY main_unitid
              ORDER BY
                alias_ind ASC,
                {source_priority_expr},
                LENGTH(instname) DESC,
                instname ASC
            ) AS canonical_rank
          FROM base
        )
        SELECT
          main_unitid,
          MAX(CASE WHEN canonical_rank = 1 THEN instname END) AS ipeds_name,
          MAX(CASE WHEN canonical_rank = 1 THEN ipeds_instname_clean END) AS ipeds_instname_clean,
          MAX(CASE WHEN canonical_rank = 1 THEN ipeds_city_clean END) AS ipeds_city_clean,
          MAX(CASE WHEN canonical_rank = 1 THEN ipeds_state_clean END) AS ipeds_state_clean,
          MAX(CASE WHEN canonical_rank = 1 THEN ipeds_zip_clean END) AS ipeds_zip_clean,
          MAX(CASE WHEN canonical_rank = 1 THEN source END) AS ipeds_name_source,
          ARRAY_AGG(DISTINCT instname) FILTER (WHERE instname IS NOT NULL) AS ipeds_name_variants,
          ARRAY_AGG(DISTINCT instname_raw) FILTER (WHERE instname_raw IS NOT NULL) AS ipeds_name_raw_variants,
          ARRAY_AGG(DISTINCT instname) FILTER (WHERE alias_ind AND instname IS NOT NULL) AS ipeds_alias_names,
          ARRAY_AGG(DISTINCT ipeds_city_clean) FILTER (WHERE ipeds_city_clean IS NOT NULL AND ipeds_city_clean <> '') AS ipeds_city_values,
          ARRAY_AGG(DISTINCT ipeds_state_clean) FILTER (WHERE ipeds_state_clean IS NOT NULL AND ipeds_state_clean <> '') AS ipeds_state_values,
          ARRAY_AGG(DISTINCT ipeds_zip_clean) FILTER (WHERE ipeds_zip_clean IS NOT NULL AND ipeds_zip_clean <> '') AS ipeds_zip_values,
          COUNT(DISTINCT instname) AS n_ipeds_name_variants
        FROM ranked
        GROUP BY main_unitid
        """
    )
    if save_output:
        _save_relation_parquet(con, "ipeds_main_institutions", cache_path, verbose=verbose, label="IPEDS main institutions")
    if verbose:
        print("✅ Created IPEDS main-institution dimension 'ipeds_main_institutions'.")
    return "ipeds_main_institutions"


def _create_f1_ipeds_edges(con, verbose=False, cache_path=None, save_output=False):
    _create_f1_ipeds_source_view(con)
    con.sql(
        f"""
        CREATE OR REPLACE TABLE f1_ipeds_edges AS
        SELECT
          f1_inst_site_id,
          f1_school_id,
          school_name,
          f1_row_num,
          f1_instname_clean,
          f1_city_clean,
          f1_state_clean,
          f1_zip_clean,
          main_unitid,
          ipeds_instname_clean,
          matchtype,
          {_sql_match_group_expr("matchtype")} AS match_group,
          {_sql_auto_promote_expr("matchtype")} AS auto_promote_ind,
          {_sql_confidence_tier_expr("matchtype")} AS confidence_tier,
          {_sql_match_score_expr("matchtype")} AS source_match_score,
          tie_candidate_count,
          tie_unitids,
          tie_instnames
        FROM f1_ipeds_source
        """
    )
    con.sql(
        """
        CREATE OR REPLACE VIEW f1_ipeds_best AS
        SELECT *
        FROM f1_ipeds_edges
        WHERE auto_promote_ind
          AND main_unitid IS NOT NULL
        """
    )
    if save_output:
        _save_relation_parquet(con, "f1_ipeds_edges", cache_path, verbose=verbose, label="F1 ↔ IPEDS edge table")
    if verbose:
        print("✅ Created canonical F1 ↔ IPEDS edge table 'f1_ipeds_edges'.")
    return "f1_ipeds_edges"


def _create_revelio_ipeds_edges(con, verbose=False, cache_path=None, save_output=False):
    _create_revelio_ipeds_source_view(con)
    con.sql(
        f"""
        CREATE OR REPLACE TABLE revelio_ipeds_edges AS
        SELECT
          revelio_inst_raw_id,
          revelio_inst_family_id,
          university_raw,
          rev_key,
          rev_instname_clean,
          rev_city_clean,
          rev_state_clean,
          rev_geo_matched,
          main_unitid,
          ipeds_instname_clean,
          matchtype,
          {_sql_match_group_expr("matchtype")} AS match_group,
          {_sql_auto_promote_expr("matchtype")} AS auto_promote_ind,
          {_sql_confidence_tier_expr("matchtype")} AS confidence_tier,
          {_sql_match_score_expr("matchtype")} AS source_match_score,
          tie_candidate_count,
          tie_unitids,
          tie_instnames,
          n_revelio_users
        FROM revelio_ipeds_source
        """
    )
    if save_output:
        _save_relation_parquet(con, "revelio_ipeds_edges", cache_path, verbose=verbose, label="Revelio ↔ IPEDS edge table")
    if verbose:
        print("✅ Created canonical Revelio ↔ IPEDS edge table 'revelio_ipeds_edges'.")
    return "revelio_ipeds_edges"


def _create_revelio_ipeds_best(con, verbose=False, cache_path=None, save_output=False):
    con.sql(
        """
        CREATE OR REPLACE TABLE revelio_ipeds_best AS
        SELECT
          e.revelio_inst_raw_id,
          e.revelio_inst_family_id,
          e.university_raw,
          e.rev_key,
          e.rev_instname_clean,
          e.rev_city_clean,
          e.rev_state_clean,
          e.rev_geo_matched,
          e.main_unitid,
          i.ipeds_name,
          i.ipeds_instname_clean,
          e.matchtype,
          'revelio_ipeds_pairwise' AS matchsource,
          CASE
            WHEN e.matchtype = 'direct_tie' OR COALESCE(e.tie_candidate_count, 0) > 1 THEN TRUE
            ELSE FALSE
          END AS ambiguous_ind,
          e.match_group,
          e.confidence_tier,
          e.source_match_score,
          e.tie_candidate_count,
          e.tie_unitids,
          e.tie_instnames,
          e.n_revelio_users
        FROM revelio_ipeds_edges AS e
        LEFT JOIN ipeds_main_institutions AS i
          ON e.main_unitid = i.main_unitid
        WHERE e.auto_promote_ind
          AND e.main_unitid IS NOT NULL
        """
    )
    if save_output:
        _save_relation_parquet(con, "revelio_ipeds_best", cache_path, verbose=verbose, label="best Revelio ↔ IPEDS mappings")
    if verbose:
        print("✅ Created promoted Revelio ↔ IPEDS table 'revelio_ipeds_best'.")
    return "revelio_ipeds_best"


def _create_ipeds_to_revelio_families(con, verbose=False, cache_path=None, save_output=False):
    con.sql(
        """
        CREATE OR REPLACE TABLE ipeds_to_revelio_families AS
        WITH ranked AS (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY main_unitid, revelio_inst_family_id
              ORDER BY
                COALESCE(n_revelio_users, 0) DESC,
                match_group DESC,
                university_raw ASC
            ) AS family_raw_rank
          FROM revelio_ipeds_best
        ),
        families AS (
          SELECT
            main_unitid,
            revelio_inst_family_id,
            MIN(rev_instname_clean) AS rev_instname_clean,
            MAX(CASE WHEN family_raw_rank = 1 THEN revelio_inst_raw_id END) AS representative_revelio_inst_raw_id,
            MAX(CASE WHEN family_raw_rank = 1 THEN university_raw END) AS representative_university_raw,
            COUNT(DISTINCT revelio_inst_raw_id) AS n_revelio_inst_raw_variants,
            CASE
              WHEN COUNT(n_revelio_users) > 0 THEN SUM(COALESCE(n_revelio_users, 0))
              ELSE COUNT(DISTINCT revelio_inst_raw_id)
            END AS n_revelio_institution_records,
            MAX(source_match_score) AS revelio_family_match_score,
            MAX(match_group) AS revelio_family_match_group,
            CASE MAX(match_group)
              WHEN 4 THEN 'direct'
              WHEN 3 THEN 'direct_tie'
              WHEN 2 THEN 'fuzzy'
              ELSE 'none'
            END AS revelio_family_matchtype,
            CASE WHEN MAX(CASE WHEN ambiguous_ind THEN 1 ELSE 0 END) = 1 THEN TRUE ELSE FALSE END AS ambiguous_ind,
            ARRAY_AGG(DISTINCT university_raw) FILTER (WHERE university_raw IS NOT NULL) AS university_raw_variants
          FROM ranked
          GROUP BY main_unitid, revelio_inst_family_id
        )
        SELECT
          f.*,
          i.ipeds_name,
          i.ipeds_instname_clean,
          i.ipeds_city_clean,
          i.ipeds_state_clean,
          i.ipeds_zip_clean
        FROM families AS f
        LEFT JOIN ipeds_main_institutions AS i
          ON f.main_unitid = i.main_unitid
        """
    )
    if save_output:
        _save_relation_parquet(con, "ipeds_to_revelio_families", cache_path, verbose=verbose, label="IPEDS → Revelio families")
    if verbose:
        print("✅ Created IPEDS → Revelio family mapping 'ipeds_to_revelio_families'.")
    return "ipeds_to_revelio_families"


def _create_f1_revelio_ipeds_resolution(con, verbose=False, cache_path=None, save_output=False):
    con.sql(
        """
        CREATE OR REPLACE TABLE f1_revelio_ipeds_resolution AS
        WITH resolved AS (
          SELECT
            f.f1_inst_site_id,
            f.f1_school_id,
            f.school_name AS f1_school_name,
            f.f1_row_num,
            f.f1_instname_clean,
            f.f1_city_clean,
            f.f1_state_clean,
            f.f1_zip_clean,
            f.main_unitid,
            f.main_unitid AS UNITID,
            i.ipeds_name,
            r.revelio_inst_family_id,
            r.representative_revelio_inst_raw_id AS revelio_inst_raw_id,
            r.representative_university_raw AS rev_university_raw,
            r.rev_instname_clean,
            LEAST(f.source_match_score, r.revelio_family_match_score) AS school_match_score,
            r.revelio_family_matchtype AS rev_matchtype,
            r.revelio_family_match_group AS match_group,
            'shared_main_unitid' AS rev_match_source,
            f.matchtype AS f1_matchtype,
            f.confidence_tier AS f1_confidence_tier,
            r.n_revelio_inst_raw_variants,
            r.n_revelio_institution_records
          FROM f1_ipeds_best AS f
          JOIN ipeds_to_revelio_families AS r
            ON f.main_unitid = r.main_unitid
          LEFT JOIN ipeds_main_institutions AS i
            ON f.main_unitid = i.main_unitid
        )
        SELECT
          *,
          CASE WHEN COUNT(*) OVER (PARTITION BY f1_inst_site_id) > 1 THEN TRUE ELSE FALSE END AS site_match_ambiguous_ind
        FROM resolved
        """
    )
    if save_output:
        _save_relation_parquet(
            con,
            "f1_revelio_ipeds_resolution",
            cache_path,
            verbose=verbose,
            label="F1 ↔ Revelio shared-IPEDS resolution",
        )
    if verbose:
        print("✅ Created F1 ↔ Revelio resolution table 'f1_revelio_ipeds_resolution'.")
    return "f1_revelio_ipeds_resolution"


def _create_f1_revelio_school_crosswalk(con, verbose=False, cache_path=None, save_output=False):
    con.sql(
        """
        CREATE OR REPLACE TABLE f1_revelio_school_crosswalk AS
        WITH family_agg AS (
          SELECT
            f1_school_id,
            MAX(f1_school_name) AS f1_school_name,
            MAX(f1_instname_clean) AS f1_instname_clean,
            revelio_inst_family_id,
            MAX(rev_instname_clean) AS rev_instname_clean,
            COUNT(DISTINCT f1_inst_site_id) AS supporting_site_count,
            COUNT(DISTINCT main_unitid) AS supporting_main_unitid_count,
            MAX(n_revelio_inst_raw_variants) AS n_revelio_inst_raw_variants,
            MAX(n_revelio_institution_records) AS n_revelio_institution_records,
            MAX(school_match_score) AS match_score
          FROM f1_revelio_ipeds_resolution
          GROUP BY f1_school_id, revelio_inst_family_id
        ),
        raw_support AS (
          SELECT
            f1_school_id,
            revelio_inst_family_id,
            revelio_inst_raw_id,
            rev_university_raw,
            COUNT(DISTINCT f1_inst_site_id) AS raw_supporting_site_count,
            MAX(n_revelio_institution_records) AS raw_n_revelio_institution_records
          FROM f1_revelio_ipeds_resolution
          GROUP BY f1_school_id, revelio_inst_family_id, revelio_inst_raw_id, rev_university_raw
        ),
        representatives AS (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY f1_school_id, revelio_inst_family_id
              ORDER BY
                raw_supporting_site_count DESC,
                raw_n_revelio_institution_records DESC,
                rev_university_raw
            ) AS raw_rank
          FROM raw_support
        ),
        school_counts AS (
          SELECT
            f1_school_id,
            COUNT(DISTINCT f1_inst_site_id) AS n_f1_sites
          FROM f1_revelio_ipeds_resolution
          GROUP BY f1_school_id
        ),
        ranked AS (
          SELECT
            fa.*,
            sc.n_f1_sites,
            MAX(fa.match_score) OVER (PARTITION BY fa.f1_school_id) AS top_match_score,
            ROW_NUMBER() OVER (
              PARTITION BY fa.f1_school_id
              ORDER BY
                fa.match_score DESC,
                fa.supporting_site_count DESC,
                fa.n_revelio_institution_records DESC,
                fa.rev_instname_clean
            ) AS school_match_rank,
            COUNT(*) OVER (PARTITION BY fa.f1_school_id) AS family_count
          FROM family_agg AS fa
          LEFT JOIN school_counts AS sc USING (f1_school_id)
        )
        SELECT
          r.f1_school_id,
          r.f1_school_name,
          r.f1_instname_clean,
          rep.revelio_inst_raw_id,
          r.revelio_inst_family_id,
          rep.rev_university_raw,
          r.rev_instname_clean,
          r.n_f1_sites,
          r.supporting_site_count,
          r.supporting_main_unitid_count,
          r.n_revelio_inst_raw_variants,
          r.n_revelio_institution_records,
          r.match_score,
          r.match_score AS lev_sim,
          r.match_score AS jw_sim,
          CASE WHEN r.family_count > 1 THEN 1 ELSE 0 END AS match_ambiguous_ind,
          r.school_match_rank,
          r.top_match_score - r.match_score AS match_score_gap_from_top
        FROM ranked AS r
        LEFT JOIN representatives AS rep
          ON r.f1_school_id = rep.f1_school_id
         AND r.revelio_inst_family_id = rep.revelio_inst_family_id
         AND rep.raw_rank = 1
        ORDER BY r.f1_school_name, r.school_match_rank, r.rev_instname_clean
        """
    )
    if save_output:
        _save_relation_parquet(
            con,
            "f1_revelio_school_crosswalk",
            cache_path,
            verbose=verbose,
            label="F1 ↔ Revelio school-family crosswalk",
        )
    if verbose:
        print("✅ Created F1 ↔ Revelio school crosswalk 'f1_revelio_school_crosswalk'.")
    return "f1_revelio_school_crosswalk"


def _create_canonical_us_inst_mappings(
    con,
    *,
    verbose=False,
    ipeds_main_path=None,
    revelio_ipeds_edges_path=None,
    revelio_ipeds_best_path=None,
    ipeds_to_revelio_families_path=None,
    f1_ipeds_edges_path=None,
    f1_revelio_ipeds_resolution_path=None,
    f1_revelio_school_crosswalk_path=None,
    save_output=False,
):
    _create_ipeds_main_institutions(con, verbose=verbose, cache_path=ipeds_main_path, save_output=save_output)
    _create_f1_ipeds_edges(con, verbose=verbose, cache_path=f1_ipeds_edges_path, save_output=save_output)
    _create_revelio_ipeds_edges(con, verbose=verbose, cache_path=revelio_ipeds_edges_path, save_output=save_output)
    _create_revelio_ipeds_best(con, verbose=verbose, cache_path=revelio_ipeds_best_path, save_output=save_output)
    _create_ipeds_to_revelio_families(con, verbose=verbose, cache_path=ipeds_to_revelio_families_path, save_output=save_output)
    _create_f1_revelio_ipeds_resolution(
        con,
        verbose=verbose,
        cache_path=f1_revelio_ipeds_resolution_path,
        save_output=save_output,
    )
    _create_f1_revelio_school_crosswalk(
        con,
        verbose=verbose,
        cache_path=f1_revelio_school_crosswalk_path,
        save_output=save_output,
    )

    if verbose:
        n_f1_sites = int(_scalar(con, "SELECT COUNT(*) FROM f1_ipeds_edges WHERE auto_promote_ind AND main_unitid IS NOT NULL"))
        n_rev_raw = int(_scalar(con, "SELECT COUNT(*) FROM revelio_ipeds_best"))
        n_res = int(_scalar(con, "SELECT COUNT(*) FROM f1_revelio_ipeds_resolution"))
        n_school = int(_scalar(con, "SELECT COUNT(*) FROM f1_revelio_school_crosswalk"))
        print(f"Canonical promoted F1 sites: {n_f1_sites:,}")
        print(f"Canonical promoted Revelio raw schools: {n_rev_raw:,}")
        print(f"Canonical F1 ↔ Revelio site-family rows: {n_res:,}")
        print(f"Canonical F1 ↔ Revelio school-family rows: {n_school:,}")
    return "f1_revelio_school_crosswalk"


def _stage_external_us_school_mappings(
    con,
    *,
    verbose=False,
    revelio_cleaning_repo_root: str | None = None,
    ipeds_main_institutions_path: str,
    revelio_ipeds_edges_path: str,
    revelio_ipeds_best_path: str,
    ipeds_to_revelio_families_path: str,
    f1_ipeds_edges_path: str,
    f1_revelio_ipeds_resolution_path: str,
    f1_revelio_school_crosswalk_path: str,
):
    artifact_paths = stage_external_us_school_matching_artifacts(
        artifact_paths={
            "ipeds_main_institutions": ipeds_main_institutions_path,
            "revelio_ipeds_edges": revelio_ipeds_edges_path,
            "revelio_ipeds_best": revelio_ipeds_best_path,
            "ipeds_to_revelio_families": ipeds_to_revelio_families_path,
            "f1_ipeds_edges": f1_ipeds_edges_path,
            "f1_revelio_ipeds_resolution": f1_revelio_ipeds_resolution_path,
            "f1_revelio_school_crosswalk": f1_revelio_school_crosswalk_path,
        },
        revelio_cleaning_repo_root=revelio_cleaning_repo_root,
        verbose=verbose,
    )

    for table_name, parquet_path in artifact_paths.items():
        con.sql(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{str(parquet_path)}')")

    if verbose:
        _print_subsection("External US school matching summary")
        print(f"IPEDS main institutions: {int(_scalar(con, 'SELECT COUNT(*) FROM ipeds_main_institutions')):,}")
        print(f"F1/IPEDS edges: {int(_scalar(con, 'SELECT COUNT(*) FROM f1_ipeds_edges')):,}")
        print(f"Promoted Revelio/IPEDS rows: {int(_scalar(con, 'SELECT COUNT(*) FROM revelio_ipeds_best')):,}")
        print(
            f"F1/Revelio school crosswalk rows: "
            f"{int(_scalar(con, 'SELECT COUNT(*) FROM f1_revelio_school_crosswalk')):,}"
        )
    return "f1_revelio_school_crosswalk"


def _create_employer_crosswalk(
    con,
    verbose=False,
    cache_path=None,
    entity_cache_path=None,
    rcid_crosswalk_path=None,
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
        _print_section("Employer Crosswalk")
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

    if verbose:
        _print_section("Preferred RCID Activity")
    result = build_preferred_rcid_activity(
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
    return result

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


def _sql_raw_name_norm_v1(colname):
    return f"""
        CASE
            WHEN {colname} IS NULL THEN NULL
            ELSE TRIM(
                REGEXP_REPLACE(
                    strip_accents(lower(TRIM(CAST({colname} AS VARCHAR)))),
                    '\\s+',
                    ' ',
                    'g'
                )
            )
        END
    """


# SQL helper function for cleaning institution names
def _sql_clean_inst_name(instnamecol):
    substituted = _sql_apply_substitutions(instnamecol, INST_NAME_SUBSTITUTIONS)
    return _sql_normalize(
        f"REGEXP_REPLACE({substituted}, '(?i)\\b(at|campus|inc|the|\\(.*\\))\\b', ' ', 'g')"
    )


def _sql_revelio_inst_raw_id_expr(university_raw_col: str) -> str:
    raw_norm = _sql_raw_name_norm_v1(university_raw_col)
    return f"""
        CASE
            WHEN {university_raw_col} IS NULL THEN NULL
            ELSE 'revraw_v1_' || sha1(COALESCE({raw_norm}, ''))
        END
    """


def _sql_revelio_inst_family_id_expr(rev_instname_clean_col: str) -> str:
    return f"""
        CASE
            WHEN {rev_instname_clean_col} IS NULL THEN NULL
            ELSE 'revfam_v1_' || sha1(COALESCE({rev_instname_clean_col}, ''))
        END
    """


def _sql_f1_inst_site_id_expr(
    school_name_col: str,
    f1_instname_clean_col: str,
    f1_city_clean_col: str,
    f1_state_clean_col: str,
    f1_zip_clean_col: str,
) -> str:
    hash_input = (
        f"COALESCE({f1_instname_clean_col}, '') || '|' || "
        f"COALESCE({f1_city_clean_col}, '') || '|' || "
        f"COALESCE({f1_state_clean_col}, '') || '|' || "
        f"COALESCE({f1_zip_clean_col}, '')"
    )
    return f"""
        CASE
            WHEN {school_name_col} IS NULL THEN NULL
            ELSE 'f1site_v1_' || sha1({hash_input})
        END
    """


def _sql_f1_school_id_expr(school_name_col: str, f1_instname_clean_col: str) -> str:
    return f"""
        CASE
            WHEN {school_name_col} IS NULL THEN NULL
            ELSE 'f1school_v1_' || sha1(COALESCE({f1_instname_clean_col}, ''))
        END
    """


def _sql_match_group_expr(matchtype_col: str) -> str:
    return f"""
        CASE
            WHEN {matchtype_col} = 'direct' THEN 4
            WHEN {matchtype_col} = 'direct_tie' THEN 3
            WHEN {matchtype_col} = 'fuzzy' THEN 2
            WHEN {matchtype_col} = 'fuzzy_tie' THEN 1
            ELSE 0
        END
    """


def _sql_auto_promote_expr(matchtype_col: str) -> str:
    return f"CASE WHEN {matchtype_col} IN ('direct', 'direct_tie', 'fuzzy') THEN TRUE ELSE FALSE END"


def _sql_confidence_tier_expr(matchtype_col: str) -> str:
    return f"""
        CASE
            WHEN {matchtype_col} = 'direct' THEN 'high'
            WHEN {matchtype_col} IN ('direct_tie', 'fuzzy') THEN 'medium'
            WHEN {matchtype_col} = 'fuzzy_tie' THEN 'candidate'
            ELSE 'unresolved'
        END
    """


def _sql_match_score_expr(matchtype_col: str) -> str:
    return f"""
        CASE
            WHEN {matchtype_col} = 'direct' THEN 1.00
            WHEN {matchtype_col} = 'direct_tie' THEN 0.98
            WHEN {matchtype_col} = 'fuzzy' THEN 0.95
            WHEN {matchtype_col} = 'fuzzy_tie' THEN 0.90
            ELSE 0.00
        END
    """

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

def ipython_load_paths():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    return cfg
  
def run_all_fxns(cfg):
    
    paths = get_cfg_section(cfg, "paths")
    foia_cfg = get_cfg_section(cfg, "foia_clean")

    # PATHS
    # foia paths
    foia_raw_dir = paths.get("foia_sevp_raw_dir")
    foia_byyear_dir = paths.get("foia_sevp_byyear_dir")
    combined_parquet_path = paths.get("foia_sevp_combined")
    
    # revelio paths
    revelio_parquet_path = _none_if_blankish(paths.get("revelio_company_mapping"))
    wrds_users_path = paths.get("wrds_users")
    
    # ipeds paths
    ipeds_name_path = paths.get("ipeds_name_cw_raw")
    ipeds_zip_path = paths.get("ipeds_zip_cw_raw")
    ipeds_parquet_path = paths.get("ipeds_name_to_zip_crosswalk")
    ipeds_unitid_main_cw_path = paths.get("ipeds_unitid_main_crosswalk")
    
    # crosswalk paths (input)
    geonames_dir = paths.get("geonames_dir")
    
    # f1 output paths
    f1_inst_crosswalk_path = paths.get("f1_inst_unitid_crosswalk")
    revelio_ipeds_cw_path = paths.get("revelio_ipeds_inst_crosswalk")
    revelio_foia_cw_path = paths.get("revelio_foia_inst_crosswalk")
    revelio_three_way_cw_path = paths.get("revelio_ipeds_foia_inst_crosswalk")
    revelio_matched_university_path = paths.get("revelio_matched_university_raws")
    ipeds_main_institutions_path = paths.get("ipeds_main_institutions")
    revelio_ipeds_edges_path = paths.get("revelio_ipeds_edges")
    revelio_ipeds_best_path = paths.get("revelio_ipeds_best")
    ipeds_to_revelio_families_path = paths.get("ipeds_to_revelio_families")
    f1_ipeds_edges_path = paths.get("f1_ipeds_edges")
    f1_revelio_ipeds_resolution_path = paths.get("f1_revelio_ipeds_resolution")
    f1_revelio_school_crosswalk_path = paths.get("f1_revelio_school_crosswalk")
    
    # employer output paths
    employer_crosswalk_path = paths.get("employer_crosswalk")
    employer_entity_mapping_path = paths.get("employer_entity_mapping")
    employer_rcid_crosswalk_path = paths.get("f1_employer_revelio_rcid_crosswalk")
    employer_auth_counts_path = paths.get("employer_auth_counts")
    preferred_rcids_path = paths.get("preferred_rcids")
    external_employer_cfg = get_cfg_section(cfg, "external_employer_matching")
    external_employer_output_dir = external_employer_cfg.get("revelio_cleaning_output_dir")
    external_school_cfg = get_cfg_section(cfg, "external_school_matching")
    external_school_repo_root = external_school_cfg.get("revelio_cleaning_repo_root")

    missing = [
        name
        for name, value in [
            ("paths.foia_combined_raw", combined_parquet_path),
            ("paths.foia_sevp_raw_dir", foia_raw_dir),
            ("paths.foia_sevp_byyear_dir", foia_byyear_dir),
            ("paths.revelio_company_mapping", revelio_parquet_path),
            ("paths.ipeds_crosswalk", ipeds_parquet_path),
            ("paths.ipeds_name_cw", ipeds_name_path),
            ("paths.ipeds_cw", ipeds_zip_path),
            ("paths.employer_crosswalk", employer_crosswalk_path),
            ("paths.wrds_users", wrds_users_path),
            ("paths.employer_auth_counts", employer_auth_counts_path),
            ("paths.preferred_rcids", preferred_rcids_path),
            ("external_employer_matching.revelio_cleaning_output_dir", external_employer_output_dir),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required config paths: {', '.join(missing)}")

    verbose = bool(foia_cfg.get("verbose", True))
    test = bool(foia_cfg.get("test", False))
    revelio_test_limit = foia_cfg.get("revelio_test_limit")
    if isinstance(revelio_test_limit, str) and revelio_test_limit.strip().lower() in {"", "none", "null"}:
        revelio_test_limit = None
    if revelio_test_limit is not None:
        revelio_test_limit = int(revelio_test_limit)
        rp = Path(str(revelio_parquet_path))
        revelio_parquet_path = str(rp.with_name(f"{rp.stem}_test{rp.suffix}"))
        print(f"Using Revelio test limit of {revelio_test_limit}, loading from '{revelio_parquet_path}'")

    ipeds_year_start = foia_cfg.get("ipeds_year_start")
    ipeds_year_end = foia_cfg.get("ipeds_year_end")
    if isinstance(ipeds_year_start, str) and ipeds_year_start.strip():
        ipeds_year_start = int(ipeds_year_start)
    if isinstance(ipeds_year_end, str) and ipeds_year_end.strip():
        ipeds_year_end = int(ipeds_year_end)
    if os.path.isdir(str(ipeds_name_path)) and os.path.isdir(str(ipeds_zip_path)):
        if ipeds_year_start is None or ipeds_year_end is None:
            raise ValueError("ipeds_year_start and ipeds_year_end must be set when IPEDS paths are directories.")

    revelio_inst_min_count = foia_cfg.get("revelio_inst_min_count", 5)
    revelio_inst_test_sample_size = foia_cfg.get("revelio_inst_test_sample_size", 1000)
    if isinstance(revelio_inst_test_sample_size, str):
        sample_size_raw = revelio_inst_test_sample_size.strip().lower()
        if sample_size_raw in {"", "none", "null"}:
            revelio_inst_test_sample_size = None
        else:
            revelio_inst_test_sample_size = int(revelio_inst_test_sample_size)
    build_revelio_ipeds = bool(foia_cfg.get("build_revelio_ipeds_crosswalk", False))
    build_revelio_foia = bool(foia_cfg.get("build_revelio_foia_crosswalk", False))
    build_revelio_three_way = bool(foia_cfg.get("build_revelio_three_way_crosswalk", False))
    build_canonical_inst_mappings = bool(foia_cfg.get("build_canonical_institution_mappings", True))

    inst_subs = foia_cfg.get("f1_inst_substitutions", {}) or {}
    if not isinstance(inst_subs, dict):
        raise ValueError("foia_clean.f1_inst_substitutions must be a dict of {key: value}.")
    global INST_NAME_SUBSTITUTIONS
    INST_NAME_SUBSTITUTIONS = {str(k): str(v) for k, v in inst_subs.items()}

    global GEONAMES_DIR_PATH
    GEONAMES_DIR_PATH = str(geonames_dir) if geonames_dir else None
  
    con = ddb.connect()
    stage_times: list[tuple[str, float]] = []
    total_t0 = time.perf_counter()

    _run_timed_stage(
        "Import all data",
        _import_all_data,
        show_timing=verbose,
        stage_times=stage_times,
        con=con,
        combined_parquet_path=str(combined_parquet_path),
        foia_raw_dir=str(foia_raw_dir),
        foia_byyear_dir=str(foia_byyear_dir),
        revelio_parquet_path=str(revelio_parquet_path),
        ipeds_parquet_path=str(ipeds_parquet_path),
        ipeds_name_path=str(ipeds_name_path),
        ipeds_zip_path=str(ipeds_zip_path),
        ipeds_unitid_main_cw_path=str(ipeds_unitid_main_cw_path) if ipeds_unitid_main_cw_path else None,
        wrds_users_path=str(wrds_users_path),
        revelio_inst_min_count=int(revelio_inst_min_count),
        revelio_inst_test_sample_size=revelio_inst_test_sample_size,
        ipeds_year_start=ipeds_year_start,
        ipeds_year_end=ipeds_year_end,
        wrds_username=str(foia_cfg.get("wrds_username", "amykimecon")),
        revelio_test_limit=revelio_test_limit,
        test=test,
        verbose=verbose,
    )

    _run_timed_stage(
        "F1 ↔ IPEDS crosswalk",
        _create_f1_inst_crosswalk,
        show_timing=verbose,
        stage_times=stage_times,
        con=con,
        verbose=verbose,
        cache_path=str(f1_inst_crosswalk_path) if f1_inst_crosswalk_path else None,
        load_cache=bool(foia_cfg.get("load_inst_from_cache", False)),
        save_output=(not test) and bool(foia_cfg.get("save_inst_to_cache", False)),
    )

    if build_revelio_ipeds:
        _run_timed_stage(
            "Revelio ↔ IPEDS crosswalk",
            _create_revelio_ipeds_inst_crosswalk,
            show_timing=verbose,
            stage_times=stage_times,
            con=con,
            verbose=verbose,
            cache_path=str(revelio_ipeds_cw_path) if revelio_ipeds_cw_path else None,
            save_output=(not test) and bool(revelio_ipeds_cw_path),
        )
    if build_revelio_foia:
        _run_timed_stage(
            "Revelio ↔ FOIA crosswalk",
            _create_revelio_foia_inst_crosswalk,
            show_timing=verbose,
            stage_times=stage_times,
            con=con,
            verbose=verbose,
            cache_path=str(revelio_foia_cw_path) if revelio_foia_cw_path else None,
            save_output=(not test) and bool(revelio_foia_cw_path),
        )
    if build_revelio_three_way:
        if not build_revelio_ipeds:
            _run_timed_stage(
                "Revelio ↔ IPEDS crosswalk",
                _create_revelio_ipeds_inst_crosswalk,
                show_timing=verbose,
                stage_times=stage_times,
                con=con,
                verbose=verbose,
                cache_path=str(revelio_ipeds_cw_path) if revelio_ipeds_cw_path else None,
                save_output=(not test) and bool(revelio_ipeds_cw_path),
            )
        if not build_revelio_foia:
            _run_timed_stage(
                "Revelio ↔ FOIA crosswalk",
                _create_revelio_foia_inst_crosswalk,
                show_timing=verbose,
                stage_times=stage_times,
                con=con,
                verbose=verbose,
                cache_path=str(revelio_foia_cw_path) if revelio_foia_cw_path else None,
                save_output=(not test) and bool(revelio_foia_cw_path),
            )
        _run_timed_stage(
            "Three-way institution crosswalk",
            _create_three_way_inst_crosswalk,
            show_timing=verbose,
            stage_times=stage_times,
            con=con,
            verbose=verbose,
            cache_path=str(revelio_three_way_cw_path) if revelio_three_way_cw_path else None,
            matched_university_path=str(revelio_matched_university_path) if revelio_matched_university_path else None,
            save_output=(not test) and bool(revelio_three_way_cw_path or revelio_matched_university_path),
        )

    if build_canonical_inst_mappings:
        if not build_revelio_ipeds and not _relation_exists(con, "revelio_ipeds_inst_crosswalk"):
            _run_timed_stage(
                "Revelio ↔ IPEDS crosswalk",
                _create_revelio_ipeds_inst_crosswalk,
                show_timing=verbose,
                stage_times=stage_times,
                con=con,
                verbose=verbose,
                cache_path=str(revelio_ipeds_cw_path) if revelio_ipeds_cw_path else None,
                save_output=(not test) and bool(revelio_ipeds_cw_path),
            )
        _run_timed_stage(
            "Canonical US institution mappings",
            _stage_external_us_school_mappings,
            show_timing=verbose,
            stage_times=stage_times,
            con=con,
            verbose=verbose,
            revelio_cleaning_repo_root=str(external_school_repo_root) if external_school_repo_root else None,
            ipeds_main_institutions_path=str(ipeds_main_institutions_path) if ipeds_main_institutions_path else None,
            revelio_ipeds_edges_path=str(revelio_ipeds_edges_path) if revelio_ipeds_edges_path else None,
            revelio_ipeds_best_path=str(revelio_ipeds_best_path) if revelio_ipeds_best_path else None,
            ipeds_to_revelio_families_path=str(ipeds_to_revelio_families_path) if ipeds_to_revelio_families_path else None,
            f1_ipeds_edges_path=str(f1_ipeds_edges_path) if f1_ipeds_edges_path else None,
            f1_revelio_ipeds_resolution_path=str(f1_revelio_ipeds_resolution_path) if f1_revelio_ipeds_resolution_path else None,
            f1_revelio_school_crosswalk_path=str(f1_revelio_school_crosswalk_path) if f1_revelio_school_crosswalk_path else None,
        )

    
    _run_timed_stage( "Employer crosswalk",   
      _create_employer_crosswalk,
      show_timing=verbose,
      stage_times=stage_times,
        con=con,
        verbose=verbose,
        cache_path=str(employer_crosswalk_path),
        entity_cache_path=str(employer_entity_mapping_path),
        rcid_crosswalk_path=(
            str(employer_rcid_crosswalk_path) if employer_rcid_crosswalk_path else None
        ),
        load_cache=bool(foia_cfg.get("load_employer_from_cache", False)),
        save_output=bool(foia_cfg.get("save_employer_to_cache", False)),
    )
    
    _run_timed_stage(
        "Employer activity filter",
        _compute_preferred_rcid_activity,
        show_timing=verbose,
        stage_times=stage_times,
        con=con,
        verbose=verbose,
        save_output=True,
        auth_counts_path=str(employer_auth_counts_path),
        rcid_list_path=str(preferred_rcids_path),
    )
    
    
    if verbose:
        _print_section("Pipeline timing summary")
        for stage_name, elapsed in stage_times:
            print(f"{stage_name:<32} {_fmt_elapsed(elapsed)}")
        print(f"{'Total pipeline':<32} {_fmt_elapsed(time.perf_counter() - total_t0)}")
    
    
#########
# MAIN  #
#########
def run(config_path=None):
    """iPython-friendly entry point. Loads config and runs full pipeline.

    Usage (from iPython):
        import importlib
        import company_shift_share.deps_foia_clean as m
        importlib.reload(m)
        m.run()
    """
    cfg = load_config(config_path or DEFAULT_CONFIG_PATH)
    run_all_fxns(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FOIA clean pipeline (config-driven).")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--test", action="store_true", help="Use test mode for FOIA import.")
    args, _ = parser.parse_known_args()  # ignore iPython/Jupyter argv

    cfg = load_config(args.config)
    run_all_fxns(cfg)


if __name__ == "__main__":
    main()
