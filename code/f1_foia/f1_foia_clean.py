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

con = ddb.connect()

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

##################################
## FUNCTIONS FOR IMPORTING DATA ##
##################################
def _read_foia_dir(con: ddb.DuckDBPyConnection, dirname: str, verbose = False) -> ddb.DuckDBPyRelation:
    """
    Read all FOIA F-1 data files from a given directory path into a DuckDB relation.
    """
    t1 = time.time()
    yr_raw = []

    # checking if directory exists (and is not J-1)
    if not os.path.isdir(f"{root}/data/raw/foia_sevp/{dirname}") and dirname != "J-1":
        print(f"Directory {dirname} does not exist, skipping...")
        return None
    
    # reading in all files in the year directory matching pattern
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
    print(f"Finished writing {dirname} data ({(t2-t1)/60:.2f} minutes)")
    return rel

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