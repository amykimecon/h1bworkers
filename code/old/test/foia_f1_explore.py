# initial exploration of foia sevp data
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

# TOGGLES
test = False
from_scratch = True # whether to read raw data files and process from scratch
data_concat = True  # whether to read raw data files and concatenate into one parquet

#####################################
## IMPORTING DATA
#####################################
# columns to treat as dates
DATE_COLS = {
    'authorization_end_date','authorization_start_date','date_of_birth',
    'first_entry_date','last_departure_date','last_entry_date',
    'opt_authorization_end_date','opt_authorization_start_date',
    'opt_employer_end_date','opt_employer_start_date',
    'program_end_date','program_start_date',
    'visa_expiration_date','visa_issue_date',
}
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

if data_concat:
    if from_scratch:
        # reading in raw data files and concatenating each year into one parquet
        for dirname in os.listdir(f"{root}/data/raw/foia_sevp"):
            t1 = time.time()
            yr_raw = []
            if not os.path.isdir(f"{root}/data/raw/foia_sevp/{dirname}") and dirname != "J-1":
                continue
            if test and dirname != "2013":
                continue
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

    # reading in each year's merged parquet into duckdb
    first = True
    for dirname in sorted(os.listdir(f"{root}/data/raw/foia_sevp"), reverse = True):
        if not os.path.isdir(f"{root}/data/raw/foia_sevp/{dirname}") or dirname == "J-1":
            continue
        if test and dirname != "2022":
            continue

        # Read schema for comparison
        path = f'{root}/data/raw/foia_sevp/{dirname}/merged{dirname}.parquet'
        if not os.path.exists(path):
            print(f"⚠️ Skipping {dirname}: file not found")
            continue

        pq_file = pq.ParquetFile(path)
        orig_cols = pq_file.schema.names
        safe_cols = [re.sub(r'[^0-9a-zA-Z_]', '', c) for c in orig_cols]

        if first:
            # Read file and rename columns safely within DuckDB
            col_rename_expr = ", ".join(
                [build_typed_expr(f'"{orig}"', safe) for orig, safe in zip(orig_cols, safe_cols)]
            )

            con.sql(f"""
                CREATE OR REPLACE TABLE allyrs_raw AS
                SELECT {col_rename_expr}
                FROM read_parquet('{path}')
            """)

            base_cols = safe_cols
            first = False
            print(f"✅ Created table from {dirname} with {len(safe_cols)} columns.")
        else:
            # Compare columns to base schema
            missing_in_new = [c for c in base_cols if c not in safe_cols]
            extra_in_new = [c for c in safe_cols if c not in base_cols]

            if missing_in_new or extra_in_new:
                print(f"⚠️ Column mismatch in {dirname}:")
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

            print(f"✅ Appended {dirname} ({len(safe_cols)} cols).")

    # Export combined data to Parquet
    con.sql(f"""COPY allyrs_raw TO '{root}/data/int/foia_sevp_combined_raw.parquet' (FORMAT PARQUET)""")
    print("✅ Exported combined data to Parquet.")

else:
    # simply read in the combined parquet if it already exists
    con.sql(f"""
        CREATE OR REPLACE TABLE allyrs_raw AS
        SELECT *
        FROM read_parquet('{root}/data/int/foia_sevp_combined_raw.parquet')
    """)
    print("✅ Loaded combined data from Parquet.")

print("Total records:", con.sql("SELECT COUNT(*) AS total_records FROM allyrs_raw").fetchdf())

#####################################
## CLEANING DATA
#####################################
rawdf = con.sql("SELECT * FROM allyrs_raw").df()

rawdf['phd'] = rawdf['student_edu_level_desc'].str.lower().str.contains('doctorate', na=False)
rawdf['masters'] = rawdf['student_edu_level_desc'].str.lower().str.contains('master', na=False)
rawdf['bachelors'] = rawdf['student_edu_level_desc'].str.lower().str.contains('bachelor', na=False)

rawdf['year'] = rawdf['year'].astype("int")
rawdf['major_1_cip_code'] = rawdf['major_1_cip_code'].astype("float")

rawdf['study'] = (rawdf['program_start_date'].dt.year <= rawdf['year'])&(rawdf['program_end_date'].dt.year >= rawdf['year'])
rawdf['opt'] = (rawdf['authorization_start_date'].dt.year <= rawdf['year'])&(rawdf['authorization_end_date'] >= rawdf['year'])
rawdf['opt_emp'] = (rawdf['opt_employer_start_date'].dt.year <= rawdf['year'])&(rawdf['opt_employer_end_date'].dt.year >= rawdf['year'])

clean = rawdf[['year', 'individual_key', 'class_of_admission', 'school_name', 'major_1_cip_code', 'major_1_description', 'phd', 'masters', 'bachelors', 'country_of_birth', 'program_start_date', 'program_end_date', 'visa_issue_date', 'visa_expiration_date', 'authorization_start_date', 'authorization_end_date', 'employer_name', 'employment_description', 'employment_opt_type', 'first_entry_date', 'last_entry_date', 'last_departure_date', 'requested_status', 'status_code', 'tuition__fees', 'students_personal_funds', 'funds_from_this_school', 'funds_from_other_sources']].copy()

raw18 = rawdf[rawdf['year'] == 2018].copy()
raw18_stud = raw18[raw18['program_start_date'] >= '2017-12-12'].copy()

raw19 = rawdf[rawdf['year'] == 2019].copy()
#####################################
## ATTEMPT AT LINKING TO REVELIO
#####################################
# TESTING for 2022 and 2023
# subsetting to princeton econ phds
pecon = clean[(clean['school_name'].str.lower() == 'princeton university') & (clean['phd']) & (clean['major_1_cip_code'] == 45.0603)].copy()

# looking for kaan
kaan = clean[(clean['school_name'].str.lower().str.contains('yale')) & (clean['country_of_birth'] == "TURKEY") & (clean['employer_name'] == "Princeton University")].copy()
# indiv key 558258 in 2019, 398314 in 2020, 260816 in 2021

# looking for yuci 
yuci = clean[(clean['school_name'].str.lower().str.contains('michigan')) & (clean['bachelors']) & (clean['country_of_birth'] == "CHINA") & (clean['employer_name'] == "Massachusetts Institute of Technology")].copy()
# indiv key 817142 in 2019, 620096 in 2020, 434108 in 2021

# looking for manuel
manuel = clean[(clean['school_name'].str.lower().str.contains('columbia')) & (clean['bachelors']) & (clean['country_of_birth'] == "COLOMBIA")& (clean['major_1_cip_code'] == "45.0603")].copy()

manuel2 = clean[(clean['school_name'].str.lower().str.contains('princeton')) & (clean['phd']) & (clean['country_of_birth'] == "COLOMBIA") & (clean['major_1_cip_code'] == "45.0603")].copy()

# visa after ug: opt - h1b 
# ug major(s): economics and math
# ug program start and end: 2016-2020

