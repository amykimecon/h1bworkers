"""
Analyze FOIA master's program outcomes and link to IPEDS program-level international shares.

Steps:
1) Load FOIA combined raw data and institution crosswalk.
2) Filter to master's students where program_end_year = reported year; deduplicate to student level.
3) Aggregate by unitid x cipcode x year: counts, OPT shares/length, status-change share, tuition/funding averages.
4) Join to IPEDS program-level international share and counts.
5) Plot a binned regplot of IPEDS share_intl vs. a chosen outcome (default avg_tuition).
"""

import os
import sys
from pathlib import Path
from typing import Iterable

import duckdb as ddb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *  # noqa: F401,F403

INT_FOLDER = f"{root}/data/int/int_files_nov2025"
FOIA_PATH = f"{root}/data/int/foia_sevp_combined_raw.parquet"
IPEDSCLEAN_PATH = f"{INT_FOLDER}/ipeds_completions_all.parquet"
F1_INST_CW_PATH = f"{INT_FOLDER}/f1_inst_unitid_crosswalk.parquet"
FIG_DIR = Path(f"{root}/figures")

# Candidate columns for FOIA
FOIA_INST_COLS = ["school_name", "f1_inst_row_num", "inst_row_num", "school_id", "f1_inst_id"]
FOIA_CIP_COLS = ["major_1_cip_code", "program_cip_code", "cipcode", "cip_code", "cip"]
FOIA_PROG_END_COLS = ["program_end_date", "program_end_dt"]
FOIA_STUDENT_KEY_COLS = ["student_key", "individual_key", "student_unique_id"]
FOIA_YEAR_COLS = ["year", "reporting_year", "data_year"]
FOIA_TUITION_COLS = ["tuition__fees", "tuition_fees", "tuition", "tuition_fees_usd"]
FOIA_FUNDING_COLS = ["students_personal_funds", "funds_from_this_school", "funds_from_other_sources"]
FOIA_OPT_END_COLS = ["opt_authorization_end_date", "authorization_end_date", "opt_employer_end_date"]
FOIA_OPT_TYPE_COL = "employment_opt_type"
FOIA_STATUS_COLS = ["requested_status", "status_code", "status"]
FOIA_EDU_LEVEL_COLS = ["student_edu_level_desc", "edu_level", "education_level_desc"]
FOIA_OPT_START_COLS = ["opt_authorization_start_date", "authorization_start_date", "opt_employer_start_date"]
FOIA_PROG_START_COLS = ["program_start_date", "program_begin_date", "program_begin_dt", "program_start_dt"]
FOIA_PROG_LENGTH_MONTHS_COLS = ["program_length_months", "program_duration_months"]
FOIA_PROG_LENGTH_WEEKS_COLS = ["program_length_weeks", "program_duration_weeks"]
FOIA_PROG_LENGTH_GENERIC_COLS = ["program_length", "program_duration"]
FOIA_INST_CW_INST_COLS = ["school_name", "f1_inst_row_num", "inst_row_num", "school_id", "f1_inst_id"]
FOIA_INST_CW_UNITID_COLS = ["unitid", "UNITID", "ipeds_unitid"]

DEFAULT_OUTCOME = "opt_share"
DEFAULT_BINS = 20
IHMP_THRESHOLD = 0.5
FOIA_IPEDS_COMMON_COLS = None  # populated lazily
Y_LABELS = {
    "avg_tuition": "Avg. tuition",
    "avg_funding": "Avg. funding",
    "opt_share": "OPT share",
    "opt_stem_share": "STEM OPT share",
    "status_change_share": "Status-change share",
    "avg_opt_years": "Avg. OPT duration (years)",
    "program_length_years": "Avg. program length (years)",
    "avg_tuition": "Avg. tuition",
}
IHMP_COLORS = {
    "IHMP": "#2e8b57",      # medium sea green
    "Non-IHMP": "#e07a5f",  # terracotta
}


def first_present(cols: Iterable[str], candidates: Iterable[str], label: str) -> str:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(f"Could not find {label}. Available columns: {cols}")


def optional_present(cols: Iterable[str], candidates: Iterable[str]) -> str | None:
    try:
        return first_present(cols, candidates, "optional")
    except KeyError:
        return None


def normalize_cip_sql(colname: str) -> str:
    return f"TRY_CAST(REGEXP_REPLACE(CAST({colname} AS VARCHAR), '[^0-9]', '', 'g') AS INTEGER)"


def program_length_years_sql(c: dict) -> str:
    """
    Build a SQL expression (DuckDB) for program length in years using available FOIA columns.
    Preference order: start/end dates → months → weeks → generic numeric length.
    """
    end_col = c["foia_end_col"]
    start_col = c.get("foia_start_col")
    months_col = c.get("foia_prog_len_months_col")
    weeks_col = c.get("foia_prog_len_weeks_col")
    generic_col = c.get("foia_prog_len_generic_col")

    if start_col:
        return (
            f"CASE WHEN {start_col} IS NOT NULL AND {end_col} IS NOT NULL "
            f"THEN DATE_DIFF('day', TRY_CAST({start_col} AS DATE), TRY_CAST({end_col} AS DATE)) / 365.25 "
            f"ELSE NULL END"
        )
    if months_col:
        return f"TRY_CAST({months_col} AS DOUBLE) / 12.0"
    if weeks_col:
        return f"TRY_CAST({weeks_col} AS DOUBLE) / 52.0"
    if generic_col:
        return f"TRY_CAST({generic_col} AS DOUBLE)"
    return "NULL"


def load_and_aggregate(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    if not os.path.exists(FOIA_PATH):
        raise FileNotFoundError(f"Missing FOIA parquet at {FOIA_PATH}")
    if not os.path.exists(F1_INST_CW_PATH):
        raise FileNotFoundError(f"Missing institution crosswalk at {F1_INST_CW_PATH}")
    if not os.path.exists(IPEDSCLEAN_PATH):
        raise FileNotFoundError(f"Missing IPEDS cleaned parquet at {IPEDSCLEAN_PATH}")

    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{FOIA_PATH}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_cw AS SELECT * FROM read_parquet('{F1_INST_CW_PATH}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_clean AS SELECT * FROM read_parquet('{IPEDSCLEAN_PATH}')")

    foia_cols = [row[0] for row in con.sql("DESCRIBE foia_raw").fetchall()]
    cw_cols = [row[0] for row in con.sql("DESCRIBE f1_inst_cw").fetchall()]

    foia_inst_col = first_present(foia_cols, FOIA_INST_COLS, "FOIA institution column")
    foia_cip_col = first_present(foia_cols, FOIA_CIP_COLS, "FOIA CIP column")
    foia_end_col = first_present(foia_cols, FOIA_PROG_END_COLS, "program end date column")
    foia_student_col = first_present(foia_cols, FOIA_STUDENT_KEY_COLS, "student identifier column")
    foia_year_col = first_present(foia_cols, FOIA_YEAR_COLS, "FOIA reporting year column")
    foia_tuition_col = first_present(foia_cols, FOIA_TUITION_COLS, "tuition column")
    foia_opt_end_col = first_present(foia_cols, FOIA_OPT_END_COLS, "OPT end date column")
    foia_status_col = first_present(foia_cols, FOIA_STATUS_COLS, "status column")
    foia_edu_col = first_present(foia_cols, FOIA_EDU_LEVEL_COLS, "education level column")
    foia_opt_start_col = optional_present(foia_cols, FOIA_OPT_START_COLS)
    cw_inst_col = first_present(cw_cols, FOIA_INST_CW_INST_COLS, "crosswalk institution column")
    cw_unitid_col = first_present(cw_cols, FOIA_INST_CW_UNITID_COLS, "crosswalk unitid column")
    foia_start_col = optional_present(foia_cols, FOIA_PROG_START_COLS)
    foia_prog_len_months_col = optional_present(foia_cols, FOIA_PROG_LENGTH_MONTHS_COLS)
    foia_prog_len_weeks_col = optional_present(foia_cols, FOIA_PROG_LENGTH_WEEKS_COLS)
    foia_prog_len_generic_col = optional_present(foia_cols, FOIA_PROG_LENGTH_GENERIC_COLS)
    global FOIA_IPEDS_COMMON_COLS
    FOIA_IPEDS_COMMON_COLS = {
        "foia_inst_col": foia_inst_col,
        "foia_cip_col": foia_cip_col,
        "foia_end_col": foia_end_col,
        "foia_student_col": foia_student_col,
        "foia_year_col": foia_year_col,
        "foia_tuition_col": foia_tuition_col,
        "foia_opt_end_col": foia_opt_end_col,
        "foia_status_col": foia_status_col,
        "foia_edu_col": foia_edu_col,
        "foia_opt_start_col": foia_opt_start_col,
        "cw_inst_col": cw_inst_col,
        "cw_unitid_col": cw_unitid_col,
        "foia_start_col": foia_start_col,
        "foia_prog_len_months_col": foia_prog_len_months_col,
        "foia_prog_len_weeks_col": foia_prog_len_weeks_col,
        "foia_prog_len_generic_col": foia_prog_len_generic_col,
    }

    funding_sum_sql = " + ".join([f"TRY_CAST({col} AS DOUBLE)" for col in FOIA_FUNDING_COLS if col in foia_cols])
    funding_expr = f"({funding_sum_sql})" if funding_sum_sql else "NULL"
    prog_length_expr = program_length_years_sql(FOIA_IPEDS_COMMON_COLS)

    foia_agg = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                {normalize_cip_sql(foia_cip_col)} AS cipcode,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS program_end_year,
                CAST({foia_year_col} AS INTEGER) AS reported_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                {foia_tuition_col} AS tuition,
                {funding_expr} AS total_funding,
                {foia_end_col} AS program_end_date,
                {foia_opt_end_col} AS opt_end_date,
                {foia_status_col} AS requested_status,
                {FOIA_OPT_TYPE_COL} AS opt_type,
                {foia_edu_col} AS edu_level, cw.school_name AS instname,
                {prog_length_expr} AS program_length_years
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
        ),
        filtered AS (
            SELECT *
            FROM foia_base
            WHERE unitid IS NOT NULL
              AND cipcode IS NOT NULL
              AND program_end_year = reported_year
              AND edu_level = 'MASTER''S'
              AND program_end_year IS NOT NULL
        ),
        student_level AS (
            SELECT
                unitid,
                cipcode,
                FIRST(instname) AS instname,
                program_end_year AS year,
                student_id,
                MAX(CASE WHEN opt_type = 'POST-COMPLETION' THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN opt_type = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                AVG(TRY_CAST(total_funding AS DOUBLE)) AS avg_funding,
                AVG(program_length_years) AS program_length_years,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL AND MAX(program_end_date) IS NOT NULL
                    THEN DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25
                    ELSE NULL
                END AS opt_years
            FROM filtered
            GROUP BY unitid, cipcode, program_end_year, student_id
        ),
        program_level AS (
            SELECT
                unitid,
                cipcode,
                FIRST(instname) AS instname,
                year,
                COUNT(DISTINCT student_id) AS total_students,
                AVG(opt_ind) AS opt_share,
                AVG(opt_stem_ind) AS opt_stem_share,
                AVG(status_change_ind) AS status_change_share,
                AVG(opt_years) AS avg_opt_years,
                AVG(avg_tuition) AS avg_tuition,
                AVG(avg_funding) AS avg_funding,
                AVG(program_length_years) AS program_length_years
            FROM student_level
            GROUP BY unitid, cipcode, year
        ),
        ipeds_prog AS (
            SELECT
                unitid,
                TRY_CAST(cipcode AS INTEGER) AS cipcode,
                CAST(year AS INTEGER) AS year,
                SUM(cnralt) AS intl_students,
                SUM(ctotalt) AS total_students,
                AVG(share_intl) AS share_intl,
                ANY_VALUE(c21basic_lab) AS c21basic_lab,
                ANY_VALUE(cipcode_lab) AS cipcode_lab
            FROM ipeds_clean
            WHERE awlevel = 7 AND cnralt >= 0 AND ctotalt >= 10
            GROUP BY unitid, cipcode, year
        )
        SELECT
            p.*,
            i.share_intl,
            i.intl_students AS ipeds_intl,
            i.total_students AS ipeds_total,
            i.c21basic_lab,
            i.cipcode_lab
        FROM program_level p
        LEFT JOIN ipeds_prog i
          ON p.unitid = i.unitid AND p.cipcode = i.cipcode AND p.year = i.year
        WHERE p.year <= 2022 AND p.year >= 2010
        """
    ).df()

    return foia_agg


def load_student_level_with_ipeds(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    if FOIA_IPEDS_COMMON_COLS is None:
        raise RuntimeError("Call load_and_aggregate first to initialize column mappings.")

    if not os.path.exists(IPEDSCLEAN_PATH):
        raise FileNotFoundError(f"Missing IPEDS cleaned parquet at {IPEDSCLEAN_PATH}")

    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{FOIA_PATH}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_cw AS SELECT * FROM read_parquet('{F1_INST_CW_PATH}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_clean AS SELECT * FROM read_parquet('{IPEDSCLEAN_PATH}')")

    c = FOIA_IPEDS_COMMON_COLS
    funding_sum_sql = " + ".join([f"TRY_CAST({col} AS DOUBLE)" for col in FOIA_FUNDING_COLS if col in con.sql("DESCRIBE foia_raw").df()["column_name"].tolist()])
    funding_expr = f"({funding_sum_sql})" if funding_sum_sql else "NULL"
    prog_length_expr = program_length_years_sql(c)
    edu_filter = "MASTER''S"
    student_df = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{c['cw_unitid_col']} AS unitid,
                {normalize_cip_sql(c['foia_cip_col'])} AS cipcode,
                CAST(EXTRACT(YEAR FROM {c['foia_end_col']}) AS INTEGER) AS program_end_year,
                CAST({c['foia_year_col']} AS INTEGER) AS reported_year,
                CAST({c['foia_student_col']} AS VARCHAR) AS student_id,
                {c['foia_tuition_col']} AS tuition,
                {funding_expr} AS total_funding,
                {c['foia_end_col']} AS program_end_date,
                {c['foia_opt_end_col']} AS opt_end_date,
                {c['foia_status_col']} AS requested_status,
                {FOIA_OPT_TYPE_COL} AS opt_type,
                {c['foia_edu_col']} AS edu_level,
                COALESCE(cw.ipeds_instname_clean, cw.school_name) AS instname,
                {prog_length_expr} AS program_length_years
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{c['foia_inst_col']} = cw.{c['cw_inst_col']}
        ),
        filtered AS (
            SELECT *
            FROM foia_base
            WHERE unitid IS NOT NULL
              AND cipcode IS NOT NULL
              AND program_end_year = reported_year
              AND edu_level = '{edu_filter}'
              AND program_end_year IS NOT NULL
        ),
        student_level AS (
            SELECT
                unitid,
                cipcode,
                program_end_year AS year,
                student_id,
                ANY_VALUE(instname) AS instname,
                MAX(CASE WHEN opt_type = 'POST-COMPLETION' THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN opt_type = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                AVG(TRY_CAST(total_funding AS DOUBLE)) AS avg_funding,
                AVG(program_length_years) AS program_length_years,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL AND MAX(program_end_date) IS NOT NULL
                    THEN DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25
                    ELSE NULL
                END AS opt_years
            FROM filtered
            GROUP BY unitid, cipcode, program_end_year, student_id
        ),
        ipeds_prog AS (
            SELECT
                unitid,
                TRY_CAST(cipcode AS INTEGER) AS cipcode,
                CAST(year AS INTEGER) AS year,
                SUM(cnralt) AS intl_students,
                SUM(ctotalt) AS total_students,
                AVG(share_intl) AS share_intl,
                ANY_VALUE(c21basic_lab) AS c21basic_lab,
                ANY_VALUE(cipcode_lab) AS cipcode_lab
            FROM ipeds_clean
            WHERE awlevel = 7 AND cnralt >= 0 AND ctotalt >= 10
            GROUP BY unitid, cipcode, year
        )
        SELECT
            s.*,
            i.share_intl,
            i.intl_students AS ipeds_intl,
            i.total_students AS ipeds_total,
            i.c21basic_lab,
            i.cipcode_lab
        FROM student_level s
        LEFT JOIN ipeds_prog i
          ON s.unitid = i.unitid AND s.cipcode = i.cipcode AND s.year = i.year
        """
    ).df()

    # Align outcome naming with program-level aggregates
    student_df["opt_share"] = student_df["opt_ind"]
    student_df["opt_stem_share"] = student_df["opt_stem_ind"]
    student_df["status_change_share"] = student_df["status_change_ind"]
    student_df["avg_opt_years"] = student_df["opt_years"]

    return student_df


def load_student_level_with_ipeds_any_level(
    con: ddb.DuckDBPyConnection, edu_level_filter: str | None = None, ipeds_awlevel: int | None = None
) -> pd.DataFrame:
    """
    Student-level data joined to IPEDS with optional education-level and IPEDS award-level filters.
    Used for computing IHMP shares across all degrees.
    """
    if FOIA_IPEDS_COMMON_COLS is None:
        raise RuntimeError("Call load_and_aggregate first to initialize column mappings.")

    if not os.path.exists(IPEDSCLEAN_PATH):
        raise FileNotFoundError(f"Missing IPEDS cleaned parquet at {IPEDSCLEAN_PATH}")

    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{FOIA_PATH}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_cw AS SELECT * FROM read_parquet('{F1_INST_CW_PATH}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_clean AS SELECT * FROM read_parquet('{IPEDSCLEAN_PATH}')")

    c = FOIA_IPEDS_COMMON_COLS
    funding_sum_sql = " + ".join([f"TRY_CAST({col} AS DOUBLE)" for col in FOIA_FUNDING_COLS if col in con.sql("DESCRIBE foia_raw").df()["column_name"].tolist()])
    funding_expr = f"({funding_sum_sql})" if funding_sum_sql else "NULL"
    edu_filter_clause = ""
    if edu_level_filter:
        safe_level = edu_level_filter.replace("'", "''")
        edu_filter_clause = f"AND edu_level = '{safe_level}'"
    awlevel_clause = f"AND awlevel = {ipeds_awlevel}" if ipeds_awlevel is not None else ""
    prog_length_expr = program_length_years_sql(c)

    student_df = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{c['cw_unitid_col']} AS unitid,
                {normalize_cip_sql(c['foia_cip_col'])} AS cipcode,
                CAST(EXTRACT(YEAR FROM {c['foia_end_col']}) AS INTEGER) AS program_end_year,
                CAST({c['foia_year_col']} AS INTEGER) AS reported_year,
                CAST({c['foia_student_col']} AS VARCHAR) AS student_id,
                {c['foia_tuition_col']} AS tuition,
                {funding_expr} AS total_funding,
                {c['foia_end_col']} AS program_end_date,
                {c['foia_opt_end_col']} AS opt_end_date,
                {c['foia_status_col']} AS requested_status,
                {FOIA_OPT_TYPE_COL} AS opt_type,
                {c['foia_edu_col']} AS edu_level,
                COALESCE(cw.ipeds_instname_clean, cw.school_name) AS instname,
                {prog_length_expr} AS program_length_years
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{c['foia_inst_col']} = cw.{c['cw_inst_col']}
        ),
        filtered AS (
            SELECT *
            FROM foia_base
            WHERE unitid IS NOT NULL
              AND cipcode IS NOT NULL
              AND program_end_year = reported_year
              {edu_filter_clause}
              AND program_end_year IS NOT NULL
        ),
        student_level AS (
            SELECT
                unitid,
                cipcode,
                program_end_year AS year,
                student_id,
                ANY_VALUE(instname) AS instname,
                MAX(CASE WHEN opt_type = 'POST-COMPLETION' THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN opt_type = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                AVG(TRY_CAST(total_funding AS DOUBLE)) AS avg_funding,
                AVG(program_length_years) AS program_length_years,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL AND MAX(program_end_date) IS NOT NULL
                    THEN DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25
                    ELSE NULL
                END AS opt_years
            FROM filtered
            GROUP BY unitid, cipcode, program_end_year, student_id
        ),
        ipeds_prog AS (
            SELECT
                unitid,
                TRY_CAST(cipcode AS INTEGER) AS cipcode,
                CAST(year AS INTEGER) AS year,
                SUM(cnralt) AS intl_students,
                SUM(ctotalt) AS total_students,
                AVG(share_intl) AS share_intl,
                ANY_VALUE(c21basic_lab) AS c21basic_lab,
                ANY_VALUE(cipcode_lab) AS cipcode_lab
            FROM ipeds_clean
            WHERE cnralt >= 0 AND ctotalt >= 10 {awlevel_clause}
            GROUP BY unitid, cipcode, year
        )
        SELECT
            s.*,
            i.share_intl,
            i.intl_students AS ipeds_intl,
            i.total_students AS ipeds_total,
            i.c21basic_lab,
            i.cipcode_lab
        FROM student_level s
        LEFT JOIN ipeds_prog i
          ON s.unitid = i.unitid AND s.cipcode = i.cipcode AND s.year = i.year
        """
    ).df()

    student_df["opt_share"] = student_df["opt_ind"]
    student_df["opt_stem_share"] = student_df["opt_stem_ind"]
    student_df["status_change_share"] = student_df["status_change_ind"]
    student_df["avg_opt_years"] = student_df["opt_years"]

    return student_df


def summarize_ihmp_opt_holders(student_df: pd.DataFrame, label: str) -> dict[str, float | int | str | None]:
    """
    Compute share of OPT and STEM OPT holders coming from IHMPs (share_intl >= threshold).
    Returns summary metrics for logging.
    """
    missing_share = int(student_df["share_intl"].isna().sum())
    df = student_df.dropna(subset=["share_intl"]).copy()
    df["ihmp"] = df["share_intl"] >= IHMP_THRESHOLD

    total_opt = df["opt_ind"].sum()
    total_stem_opt = df["opt_stem_ind"].sum()
    ihmp_opt = df.loc[df["ihmp"], "opt_ind"].sum()
    ihmp_stem_opt = df.loc[df["ihmp"], "opt_stem_ind"].sum()

    share_opt = ihmp_opt / total_opt if total_opt else None
    share_stem_opt = ihmp_stem_opt / total_stem_opt if total_stem_opt else None

    return {
        "label": label,
        "observations": len(df),
        "missing_share_intl": missing_share,
        "share_opt_from_ihmp": share_opt,
        "share_stem_opt_from_ihmp": share_stem_opt,
    }


def plot_binned(df: pd.DataFrame, yvar: str = DEFAULT_OUTCOME, bins: int = DEFAULT_BINS, show: bool = True) -> Path:
    if yvar not in df.columns:
        raise ValueError(f"yvar {yvar} not in dataframe")
    plot_df = df.dropna(subset=["share_intl", yvar]).copy()
    if plot_df.empty:
        raise ValueError("No data available for plotting.")
    plot_df["bin"] = pd.qcut(plot_df["share_intl"], bins, duplicates="drop")
    binned = plot_df.groupby("bin", as_index=False).agg(
        share_intl_mean=("share_intl", "mean"),
        y_mean=(yvar, "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=binned, x="share_intl_mean", y="y_mean", ax=ax, scatter_kws={"s": 40})
    ax.set_xlabel("IPEDS share international (binned mean)")
    ax.set_ylabel(yvar)
    ax.set_title(f"Binned regplot: share_intl vs. {yvar}")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"foia_ipeds_binned_{yvar}.png"
    fig.set_size_inches(11,6)
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def compute_opt_active_counts(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    """
    For each year, count unique students with active OPT authorization windows (start/end inclusive),
    split by IHMP (share_intl >= threshold) using program-level IPEDS shares.
    Note: no longer restricted to master's; counts use all F-1 records with valid OPT dates.
    """
    if FOIA_IPEDS_COMMON_COLS is None:
        raise RuntimeError("Call load_and_aggregate first to initialize column mappings.")
    c = FOIA_IPEDS_COMMON_COLS
    if c.get("foia_opt_start_col") is None or c.get("foia_opt_end_col") is None:
        raise ValueError("OPT start or end date column missing; cannot compute active OPT counts.")

    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{FOIA_PATH}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_cw AS SELECT * FROM read_parquet('{F1_INST_CW_PATH}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_clean AS SELECT * FROM read_parquet('{IPEDSCLEAN_PATH}')")

    opt_df = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{c['cw_unitid_col']} AS unitid,
                {normalize_cip_sql(c['foia_cip_col'])} AS cipcode,
                CAST(EXTRACT(YEAR FROM {c['foia_end_col']}) AS INTEGER) AS program_end_year,
                CAST({c['foia_student_col']} AS VARCHAR) AS student_id,
                TRY_CAST({c['foia_opt_start_col']} AS DATE) AS opt_start_dt,
                TRY_CAST({c['foia_opt_end_col']} AS DATE) AS opt_end_dt,
                {c['foia_edu_col']} AS edu_level
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{c['foia_inst_col']} = cw.{c['cw_inst_col']}
        ),
        filtered AS (
            SELECT *
            FROM foia_base
            WHERE unitid IS NOT NULL
              AND cipcode IS NOT NULL
              AND opt_start_dt IS NOT NULL
              AND opt_end_dt IS NOT NULL
              AND opt_end_dt >= opt_start_dt
        ),
        active_years AS (
            SELECT
                unitid,
                cipcode,
                student_id,
                program_end_year,
                CAST(EXTRACT(YEAR FROM opt_start_dt) AS INTEGER) AS year
            FROM filtered
            WHERE opt_start_dt <= MAKE_DATE(CAST(EXTRACT(YEAR FROM opt_start_dt) AS INTEGER), 9, 1)
              AND opt_end_dt >= MAKE_DATE(CAST(EXTRACT(YEAR FROM opt_start_dt) AS INTEGER), 9, 1)
        ),
        ipeds_prog AS (
            SELECT
                unitid,
                TRY_CAST(cipcode AS INTEGER) AS cipcode,
                CAST(year AS INTEGER) AS year,
                AVG(share_intl) AS share_intl
            FROM ipeds_clean
            WHERE awlevel = 7 AND cnralt >= 0 AND ctotalt >= 10
            GROUP BY unitid, cipcode, year
        )
        SELECT
            a.year,
            COUNT(DISTINCT a.student_id) AS total_students,
            COUNT(DISTINCT CASE WHEN i.share_intl >= {IHMP_THRESHOLD} THEN a.student_id END) AS ihmp_students,
            COUNT(DISTINCT CASE WHEN i.share_intl < {IHMP_THRESHOLD} THEN a.student_id END) AS non_ihmp_students
        FROM active_years a
        LEFT JOIN ipeds_prog i
          ON a.unitid = i.unitid AND a.cipcode = i.cipcode AND a.year = i.year
        WHERE a.year BETWEEN 2010 AND 2025
        GROUP BY a.year
        ORDER BY a.year
        """
    ).df()

    return opt_df


def plot_opt_active_counts(counts_df: pd.DataFrame, show: bool = True) -> tuple[Path, Path]:
    """
    Plot active OPT student counts: total, and IHMP vs Non-IHMP by year.
    """
    if counts_df.empty:
        raise ValueError("No data for OPT active counts plotting.")
    counts_df = counts_df.copy()
    counts_df[["total_students", "ihmp_students", "non_ihmp_students"]] = (
        counts_df[["total_students", "ihmp_students", "non_ihmp_students"]] / 1000.0
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=counts_df[counts_df["year"].between(2010, 2022)],
        x="year",
        y="total_students",
        marker="o",
        ax=ax,
        color=IHMP_COLORS["IHMP"],
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Active OPT students (thousands, all F-1)")
    ax.set_title("")
    fig.tight_layout()
    out_total = FIG_DIR / "opt_active_students_total.png"
    fig.set_size_inches(11,6)
    fig.savefig(out_total, dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    plot_df = counts_df.melt(
        id_vars="year",
        value_vars=["ihmp_students", "non_ihmp_students"],
        var_name="group",
        value_name="students",
    )
    plot_df["group_label"] = plot_df["group"].map({
        "ihmp_students": "IHMP",
        "non_ihmp_students": "Non-IHMP",
    })
    sns.lineplot(
        data=plot_df[plot_df['year'].between(2010, 2022)],
        x="year",
        y="students",
        hue="group_label",
        hue_order=["Non-IHMP", "IHMP"],
        palette=[IHMP_COLORS["Non-IHMP"], IHMP_COLORS["IHMP"]],
        marker="o",
        ax=ax2,
    )
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Active OPT students (thousands)")
    ax2.set_title("")
    ax2.legend(title=None)
    fig2.tight_layout()
    out_split = FIG_DIR / "opt_active_students_ihmp.png"
    fig2.set_size_inches(11,6)
    fig2.savefig(out_split, dpi=300)
    if show:
        plt.show()
    plt.close(fig2)

    return out_total, out_split


def plot_intl_share_in_ihmp(df: pd.DataFrame, show: bool = True) -> Path:
    """
    Plot share of international students (IPEDS intl counts) that are in IHMP programs by graduation year.
    Uses program-level aggregates (masters only).
    """
    required_cols = ["share_intl", "ipeds_intl", "year"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{col} not available for intl share in IHMP plot.")
    plot_df = df.dropna(subset=required_cols).copy()
    if plot_df.empty:
        raise ValueError("No data for intl share in IHMP plot.")
    agg = (
        plot_df.groupby("year", as_index=False)
        .agg(
            total_intl=("ipeds_intl", "sum"),
            ihmp_intl=("ipeds_intl", lambda x: x[plot_df.loc[x.index, "share_intl"] >= IHMP_THRESHOLD].sum()),
        )
    )
    agg["ihmp_share"] = agg["ihmp_intl"] / agg["total_intl"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=agg, x="year", y="ihmp_share", marker="o", color=IHMP_COLORS["IHMP"], ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of intl students in IHMP programs")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("")
    fig.tight_layout()
    out_path = FIG_DIR / "intl_students_share_in_ihmp.png"
    fig.set_size_inches(11,6)
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_opt_share_from_ihmp(
    datasets: list[tuple[pd.DataFrame, str]], show: bool = True
) -> Path:
    """
    Plot share of OPT holders coming from IHMP programs (share_intl >= threshold) over time.
    Accepts multiple datasets to contrast (e.g., all degrees vs master's only).
    """
    records: list[pd.DataFrame] = []
    required_cols = {"share_intl", "opt_ind", "year"}

    for df, label in datasets:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for OPT IHMP share plot in {label}: {missing}")
        df_use = df.dropna(subset=["share_intl", "opt_ind", "year"]).copy()
        if df_use.empty:
            continue
        df_use["opt_ihmp"] = df_use["opt_ind"] * (df_use["share_intl"] >= IHMP_THRESHOLD)
        agg = (
            df_use.groupby("year", as_index=False)
            .agg(opt_total=("opt_ind", "sum"), opt_ihmp=("opt_ihmp", "sum"))
            .assign(series=label)
        )
        agg["ihmp_opt_share"] = agg.apply(
            lambda r: r["opt_ihmp"] / r["opt_total"] if r["opt_total"] else pd.NA, axis=1
        )
        records.append(agg)

    if not records:
        raise ValueError("No data available for OPT IHMP share plot.")

    plot_df = pd.concat(records, ignore_index=True)
    plot_df = plot_df.dropna(subset=["ihmp_opt_share"])

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=plot_df[plot_df["year"].between(2010, 2022)],
        x="year",
        y="ihmp_opt_share",
        hue="series",
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of OPT holders from IHMP programs")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("")
    ax.legend(title=None)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "opt_share_from_ihmp.png"
    fig.set_size_inches(11,6)
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_ihmp_vs_nonihmp(
    df: pd.DataFrame, yvar: str = DEFAULT_OUTCOME, show: bool = True, source: str = "student"
) -> Path:
    """
    Plot outcome over time by IHMP status (share_intl >= threshold).
    source: "student" uses student-level DF with share_intl joined per student;
            "program" uses program-level aggregates.
    """
    if yvar not in df.columns:
        raise ValueError(f"yvar {yvar} not in dataframe")
    plot_df = df.dropna(subset=["share_intl", yvar]).copy()
    if plot_df.empty:
        raise ValueError("No data available for plotting.")
    plot_df["ihmp"] = (plot_df["share_intl"] >= IHMP_THRESHOLD).astype(int)
    plot_df["ihmp_label"] = plot_df["ihmp"].map({1: "IHMP", 0: "Non-IHMP"})
    level = "student" if source == "student" else "program"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(
        data=plot_df[plot_df['year'].between(2010, 2022)],
        x="year",
        y=yvar,
        hue="ihmp_label",
        hue_order=["Non-IHMP", "IHMP"],
        palette=[IHMP_COLORS["Non-IHMP"], IHMP_COLORS["IHMP"]],
        estimator="mean",
        errorbar=("ci", 95),
        ax=ax,
        marker="o",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(Y_LABELS.get(yvar, yvar))
    ax.set_title("")
    ax.legend(title=None)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"foia_ipeds_{yvar}_ihmp_{level}.png"
    fig.set_size_inches(11,6)
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_outcome_bars(df: pd.DataFrame, outcomes: list[str] | None = None, show: bool = True, source: str = "student") -> Path:
    """
    Bar chart of average outcomes by IHMP vs non-IHMP. Uses share_intl to define IHMP.
    """
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid", {'axes.facecolor': 'white'})
    if "share_intl" not in df.columns:
        raise ValueError("share_intl not in dataframe for IHMP split")
    if outcomes is None:
        outcomes = [c for c in Y_LABELS.keys() if c in df.columns]
    missing = [o for o in outcomes if o not in df.columns]
    if missing:
        raise ValueError(f"Outcomes not found in dataframe: {missing}")

    plot_df = df.dropna(subset=["share_intl"] + outcomes).copy()
    plot_df["ihmp"] = (plot_df["share_intl"] >= IHMP_THRESHOLD).astype(int)
    melted = plot_df.melt(id_vars="ihmp", value_vars=outcomes, var_name="outcome", value_name="value")
    melted["ihmp_label"] = melted["ihmp"].map({1: "IHMP", 0: "Non-IHMP"})
    melted["outcome_label"] = melted["outcome"].map(Y_LABELS).fillna(melted["outcome"])
    # normalize to mean 0, std 1 for comparability across outcomes
    melted["value_norm"] = melted.groupby("outcome")["value"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) else 0
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=melted,
        x="outcome_label",
        y="value_norm",
        hue="ihmp_label",
        hue_order=["Non-IHMP", "IHMP"],
        palette=[IHMP_COLORS["Non-IHMP"], IHMP_COLORS["IHMP"]],
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Average value (normalized)")
    ax.set_title("")
    ax.legend(title=None)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"foia_ipeds_outcomes_by_ihmp_{source}.png"
    fig.set_size_inches(11,6)
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_program_length_distribution(df: pd.DataFrame, show: bool = True) -> Path:
    """
    Plot distribution of program length (years) for IHMP vs non-IHMP master's students.
    """
    required_cols = ["share_intl", "program_length_years"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{col} not found in dataframe for program length plot.")
    plot_df = df.dropna(subset=required_cols).copy()
    if plot_df.empty:
        raise ValueError("No data available for program length plotting.")
    plot_df["ihmp_label"] = (plot_df["share_intl"] >= IHMP_THRESHOLD).map({True: "IHMP", False: "Non-IHMP"})

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(
        data=plot_df,
        x="program_length_years",
        hue="ihmp_label",
        hue_order=["Non-IHMP", "IHMP"],
        palette=[IHMP_COLORS["Non-IHMP"], IHMP_COLORS["IHMP"]],
        stat="density",
        common_norm=False,
        bins=40,
        element="bars",
        multiple="dodge",
        ax=ax,
        alpha=0.7,
    )
    ax.set_xlabel("Program length (years)")
    ax.set_ylabel("Density")
    ax.set_title("")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "program_length_distribution_ihmp.png"
    fig.set_size_inches(11,6)
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_combined_outcome_bars(student_df: pd.DataFrame, program_df: pd.DataFrame, show: bool = True) -> Path:
    """
    Single bar plot with IHMP vs non-IHMP:
    - Student-level: opt_share, opt_stem_share, status_change_share, avg_opt_years
    - Program-level: avg_tuition, program_length_years
    """
    outcomes = [
        ("student", "opt_share"),
        ("student", "opt_stem_share"),
        ("student", "status_change_share"),
        ("student", "avg_opt_years"),
        ("program", "avg_tuition"),
        ("program", "program_length_years"),
    ]
    records = []
    for src, outcome in outcomes:
        df = student_df if src == "student" else program_df
        if outcome not in df.columns:
            raise ValueError(f"{outcome} not in {src} dataframe.")
        df_use = df.dropna(subset=["share_intl", outcome]).copy()
        if df_use.empty:
            continue
        df_use["ihmp_label"] = (df_use["share_intl"] >= IHMP_THRESHOLD).map({True: "IHMP", False: "Non-IHMP"})
        for _, row in df_use.iterrows():
            records.append(
                {
                    "source": src,
                    "outcome": outcome,
                    "outcome_label": Y_LABELS.get(outcome, outcome),
                    "ihmp_label": row["ihmp_label"],
                    "value": row[outcome],
                }
            )
    if not records:
        raise ValueError("No data available for combined outcome bars.")
    plot_df = pd.DataFrame(records)
    group_means = plot_df.groupby(["outcome", "ihmp_label"])["value"].mean().unstack()

    def fmt_val(val: float | None) -> str:
        return "n/a" if pd.isna(val) else f"{val:,.2f}"

    label_map = {}
    for outcome in plot_df["outcome"].unique():
        label = Y_LABELS.get(outcome, outcome)
        non_val = fmt_val(group_means.loc[outcome].get("Non-IHMP"))
        ihmp_val = fmt_val(group_means.loc[outcome].get("IHMP"))
        label_map[outcome] = rf"$\bf{{{label}}}$" + "\n" + f"Mean: Non-IHMP = {non_val}; IHMP = {ihmp_val}"

    plot_df["outcome_label_with_stats"] = plot_df["outcome"].map(label_map)
    plot_df["value_norm"] = plot_df.groupby("outcome")["value"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) else 0
    )
    # Preserve desired order
    outcome_order = [label_map[o] for _, o in outcomes if o in plot_df["outcome"].unique()]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=plot_df,
        y="outcome_label_with_stats",
        x="value_norm",
        hue="ihmp_label",
        hue_order=["Non-IHMP", "IHMP"],
        palette=[IHMP_COLORS["Non-IHMP"], IHMP_COLORS["IHMP"]],
        errorbar=("ci", 95),
        ax=ax,
        order=outcome_order,
    )
    ax.set_ylabel("")
    ax.set_xlabel("Average value (normalized)")
    ax.set_title("")
    ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    fig.subplots_adjust(left=0.35)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "foia_ipeds_outcomes_combined.png"
    fig.set_size_inches(11,6)
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def ihmp_share_by_carnegie(df: pd.DataFrame, year_min: int | None = None, year_max: int | None = None) -> pd.DataFrame:
    """
    Table: by Carnegie (c21basic_lab), share and count of institutions with at least one IHMP.
    """
    if "c21basic_lab" not in df.columns:
        raise ValueError("c21basic_lab not available in dataframe.")
    ihmp_df = df.dropna(subset=["share_intl"]).copy()
    if year_min is not None:
        ihmp_df = ihmp_df[ihmp_df["year"] >= year_min]
    if year_max is not None:
        ihmp_df = ihmp_df[ihmp_df["year"] <= year_max]
    ihmp_df["ihmp"] = ihmp_df["share_intl"] >= IHMP_THRESHOLD
    inst_level = ihmp_df.groupby(["unitid", "c21basic_lab"], as_index=False)["ihmp"].max()
    summary = (
        inst_level.groupby("c21basic_lab", as_index=False)
        .agg(total_inst=("unitid", "nunique"), ihmp_inst=("ihmp", "sum"))
        .assign(ihmp_share=lambda x: x["ihmp_inst"] / x["total_inst"])
        .sort_values("ihmp_share", ascending=False)
    )
    return summary


def top_ihmp_names(
    df: pd.DataFrame, top_n: int = 20, year_min: int | None = None, year_max: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return top school names and major names among IHMP programs.
    """
    ihmp_df = df.loc[df["share_intl"] >= IHMP_THRESHOLD].copy()
    if year_min is not None:
        ihmp_df = ihmp_df[ihmp_df["year"] >= year_min]
    if year_max is not None:
        ihmp_df = ihmp_df[ihmp_df["year"] <= year_max]
    school_counts = (
        ihmp_df["instname"]
        .dropna()
        .value_counts()
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "school", "instname": "count"})
    )
    major_counts = (
        ihmp_df["cipcode_lab"]
        .dropna()
        .value_counts()
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "major", "cipcode_lab": "count"})
    )
    return school_counts, major_counts


def main():
    con = ddb.connect()
    year_min_filter = 2018  # set to an int (e.g., 2015) to restrict summaries/leaderboards
    year_max_filter = 2018  # set to an int (e.g., 2022) to restrict summaries/leaderboards

    df = load_and_aggregate(con)
    out_path = plot_binned(df, yvar=DEFAULT_OUTCOME, show=True)
    print(f"Saved binned regplot to {out_path}")

    student_df = load_student_level_with_ipeds(con)
    all_degree_df = load_student_level_with_ipeds_any_level(con, edu_level_filter=None, ipeds_awlevel=None)
    ihmp_student_path = plot_ihmp_vs_nonihmp(student_df, yvar=DEFAULT_OUTCOME, show=True, source="student")
    print(f"Saved IHMP vs non-IHMP plot (student-level) to {ihmp_student_path}")

    ihmp_program_path = plot_ihmp_vs_nonihmp(df, yvar=DEFAULT_OUTCOME, show=True, source="program")
    print(f"Saved IHMP vs non-IHMP plot (program-level) to {ihmp_program_path}")

    # combined_bar_path = plot_combined_outcome_bars(student_df, df, show=True)
    # print(f"Saved combined outcome bars by IHMP to {combined_bar_path}")

    # try:
    #     prog_len_path = plot_program_length_distribution(student_df, show=True)
    #     print(f"Saved program length distribution plot to {prog_len_path}")
    # except ValueError as exc:
    #     print(f"Program length distribution not generated: {exc}")

    carnegie_table = ihmp_share_by_carnegie(df, year_min=year_min_filter, year_max=year_max_filter)
    print("IHMP share by Carnegie classification:")
    print(carnegie_table)

    school_top, major_top = top_ihmp_names(df, top_n=20, year_min=year_min_filter, year_max=year_max_filter)
    print("Top IHMP schools:\n", school_top)
    print("Top IHMP majors:\n", major_top)

    def fmt_share(val: float | None) -> str:
        return f"{val:.3f}" if val is not None else "n/a"

    opt_stats = [
        summarize_ihmp_opt_holders(all_degree_df, "All degrees"),
        summarize_ihmp_opt_holders(student_df, "Master's only"),
    ]
    print("Share of OPT holders coming from IHMPs (share_intl >= 0.5):")
    for stat in opt_stats:
        print(
            f"  {stat['label']}: OPT {fmt_share(stat['share_opt_from_ihmp'])}, "
            f"STEM OPT {fmt_share(stat['share_stem_opt_from_ihmp'])}, "
            f"observations used {stat['observations']}, missing share_intl rows {stat['missing_share_intl']}"
        )

    try:
        opt_counts = compute_opt_active_counts(con)
        total_path, split_path = plot_opt_active_counts(opt_counts, show=True)
        print(f"Saved OPT active counts plots to {total_path} and {split_path}")
    except Exception as exc:
        print(f"Skipping OPT active counts plots: {exc}")

    try:
        opt_share_path = plot_opt_share_from_ihmp(
            [
                (all_degree_df, "All degrees"),
                (student_df, "Master's only"),
            ],
            show=True,
        )
        print(f"Saved OPT IHMP share plot to {opt_share_path}")
    except Exception as exc:
        print(f"Skipping OPT IHMP share plot: {exc}")

    try:
        intl_share_path = plot_intl_share_in_ihmp(df, show=True)
        print(f"Saved intl students share in IHMP plot to {intl_share_path}")
    except Exception as exc:
        print(f"Skipping intl share in IHMP plot: {exc}")


if __name__ == "__main__":
    main()
