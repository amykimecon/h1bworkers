# Compute international student growth by year at the institution x CIP level
# using IPEDS completions and FOIA F-1 data, then compare in plots.

import os
import sys
from typing import Iterable

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *  # noqa: F401,F403

INT_FOLDER = f"{root}/data/int/int_files_nov2025"
IPEDS_PATH = f"{INT_FOLDER}/ipeds_completions_all.parquet"
FOIA_PATH = f"{INT_FOLDER}/foia_sevp_combined_raw.parquet"
F1_INST_CW_PATH = f"{INT_FOLDER}/f1_inst_unitid_crosswalk.parquet"
FIG_DIR = f"{root}/figures"

# Candidate columns for FOIA inputs and crosswalks; adjust if schema differs.
FOIA_INST_COLS = ["school_name","f1_inst_row_num", "inst_row_num", "school_id", "f1_inst_id"]
FOIA_CIP_COLS = ["major_1_cip_code","program_cip_code", "cipcode", "cip_code", "cip"]
FOIA_ENDDATE_COLS = ["program_end_date", "program_end_dt", "opt_authorization_end_date"]
CW_INST_COLS = ["school_name","f1_inst_row_num", "inst_row_num", "school_id", "f1_inst_id"]
CW_UNITID_COLS = ["unitid", "UNITID", "ipeds_unitid"]


def first_present(cols: Iterable[str], candidates: Iterable[str], label: str) -> str:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise ValueError(f"Could not find {label}. Available columns: {sorted(cols)}")


def normalize_cip_sql(colname: str) -> str:
    # Strip dots, keep digits, cast to integer for joins across sources.
    return f"TRY_CAST(REGEXP_REPLACE(CAST({colname} AS VARCHAR), '[^0-9]', '', 'g') AS INTEGER)"


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    con = ddb.connect()

    # Load data
    con.sql(f"CREATE OR REPLACE TABLE ipeds_raw AS SELECT * FROM read_parquet('{IPEDS_PATH}')")
    con.sql(f"CREATE OR REPLACE TABLE foia_raw AS SELECT * FROM read_parquet('{FOIA_PATH}')")
    con.sql(f"CREATE OR REPLACE TABLE f1_inst_cw AS SELECT * FROM read_parquet('{F1_INST_CW_PATH}')")

    # Detect column names
    foia_cols = [row[0] for row in con.sql("DESCRIBE foia_raw").fetchall()]
    cw_cols = [row[0] for row in con.sql("DESCRIBE f1_inst_cw").fetchall()]

    foia_inst_col = first_present(foia_cols, FOIA_INST_COLS, "FOIA institution column")
    foia_cip_col = first_present(foia_cols, FOIA_CIP_COLS, "FOIA CIP column")
    foia_end_col = first_present(foia_cols, FOIA_ENDDATE_COLS, "FOIA program end date column")
    cw_inst_col = first_present(cw_cols, CW_INST_COLS, "crosswalk institution column")
    cw_unitid_col = first_present(cw_cols, CW_UNITID_COLS, "crosswalk unitid column")

    print(f"Using FOIA inst: {foia_inst_col}, CIP: {foia_cip_col}, end date: {foia_end_col}")
    print(f"Using crosswalk inst: {cw_inst_col} -> unitid: {cw_unitid_col}")

    ipeds = con.sql(
        """
        SELECT
            unitid,
            TRY_CAST(cipcode AS INTEGER) AS cipcode,
            year::INT AS year,
            SUM(cnralt) AS intl_students,
            SUM(ctotalt) AS total_students,
            COUNT(*) AS num_programs
        FROM ipeds_raw
        WHERE unitid IS NOT NULL AND cipcode >= 10000 AND cnralt > 0
        GROUP BY unitid, cipcode, year
        """
    ).df()
    print(
        "IPEDS combos:",
        f"{ipeds['unitid'].nunique()} unique unitids,",
        f"{ipeds['cipcode'].nunique()} unique cips,",
        f"{ipeds['year'].nunique()} unique years,",
        "rows:",
        ipeds.shape[0],
    )

    foia = con.sql(
        f"""
        SELECT
            unitid,
            cipcode,
            year,
            COUNT(DISTINCT student_key) AS intl_students
        FROM (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                student_key,
                {normalize_cip_sql(foia_cip_col)} AS cipcode,
                EXTRACT(YEAR FROM fr.{foia_end_col})::INT AS year
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE fr.{foia_end_col} IS NOT NULL AND EXTRACT(YEAR FROM fr.{foia_end_col})::INT = year::INT AND year::INT <= 2022
        )
        WHERE unitid IS NOT NULL AND cipcode IS NOT NULL AND year IS NOT NULL
        GROUP BY unitid, cipcode, year
        """
    ).df()
    print(
        "FOIA combos:",
        f"{foia['unitid'].nunique()} unique unitids,",
        f"{foia['cipcode'].nunique()} unique cips,",
        f"{foia['year'].nunique()} unique years,",
        "rows:",
        foia.shape[0],
    )

    # def add_growth(df: pd.DataFrame, source: str) -> pd.DataFrame:
    #     df = df.sort_values(["unitid", "cipcode", "year"])
    #     df["intl_growth"] = df.groupby(["unitid", "cipcode"])["intl_students"].diff()
    #     df["intl_growth_pct"] = df.groupby(["unitid", "cipcode"])["intl_students"].pct_change()
    #     df["source"] = source
    #     return df

    # ipeds_growth = add_growth(ipeds, "ipeds")
    # foia_growth = add_growth(foia, "foia")

    combined_growth = ipeds.merge(
        foia,
        on=["unitid", "cipcode", "year"],
        suffixes=("_ipeds", "_foia"),
        how="right",
    )
    combined_growth['intl_students_ipeds'] = combined_growth['intl_students_ipeds'].fillna(0)

    print(
        "Merged combos:",
        f"{combined_growth['unitid'].nunique()} unique unitids,",
        f"{combined_growth['cipcode'].nunique()} unique cips,",
        f"{combined_growth['year'].nunique()} unique years,",
        "rows:",
        combined_growth.shape[0],
    )

    # Binned-by-x scatter with regression line
    # Bin on IPEDS growth, plot bin means of FOIA growth, overlay OLS fit.
    combined_growth = combined_growth.copy()
    combined_growth['log_intl_students_ipeds'] = np.log(combined_growth['intl_students_ipeds'])
    combined_growth['log_intl_students_foia'] = np.log(combined_growth['intl_students_foia'])
    sns.regplot(
        data=combined_growth[combined_growth['intl_students_foia'] < 200],
        x="intl_students_foia",
        y="intl_students_ipeds",
        line_kws={"color": "red", "lw": 2, "label": "OLS fit"},
        x_bins = 20
    )
    plt.xlabel("Intl students (FOIA)")
    plt.ylabel("Intl students (IPEDS)")
    plt.legend()
    plt.title("Institution x CIP-level intl student count: binned means + OLS line")
    plt.tight_layout()
    # regplot_path = os.path.join(FIG_DIR, "intl_growth_binned_regplot.png")
    # plt.savefig(regplot_path, dpi=300)
    # plt.close()
    # print(f"Saved binned regplot to {regplot_path}")

    # Time series of totals over all programs/CIPs
    ipeds_tot = ipeds.groupby("year", as_index=False)["intl_students"].sum()
    foia_tot = foia.groupby("year", as_index=False)["intl_students"].sum()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=ipeds_tot, x="year", y="intl_students", label="IPEDS")
    sns.lineplot(data=foia_tot, x="year", y="intl_students", label="FOIA")
    plt.ylabel("International students")
    plt.title("International students over time (all institutions/CIPs)")
    plt.tight_layout()
    # line_path = os.path.join(FIG_DIR, "intl_students_over_time.png")
    # plt.savefig(line_path, dpi=300)
    # plt.close()
    # print(f"Saved totals line plot to {line_path}")

    # Optional: save growth datasets
    out_growth = os.path.join(root, "data", "int", "intl_student_growth_ipeds_vs_foia.parquet")
    combined_growth.to_parquet(out_growth, index=False)
    print(f"Wrote merged growth data to {out_growth}")


# if __name__ == "__main__":
#     main()
