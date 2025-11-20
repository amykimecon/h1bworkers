# Compute international student growth by year at the institution x CIP level
# using IPEDS completions and FOIA F-1 data, then compare in plots.

import os
import sys
from typing import Iterable

import duckdb as ddb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *  # noqa: F401,F403

INT_FOLDER = f"{root}/data/int/int_files_nov2025"
IPEDS_PATH = f"{INT_FOLDER}/ipeds_completions_all.parquet"
FOIA_PATH = f"{root}/data/int/foia_sevp_combined_raw.parquet"
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
        WHERE unitid IS NOT NULL AND cipcode >= 10000
        GROUP BY unitid, cipcode, year
        """
    ).df()
    
    foia = con.sql(
        f"""
        SELECT
            unitid,
            cipcode,
            year,
            COUNT(*) AS intl_students
        FROM (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                {normalize_cip_sql(foia_cip_col)} AS cipcode,
                EXTRACT(YEAR FROM fr.{foia_end_col})::INT AS year
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE fr.{foia_end_col} IS NOT NULL
        )
        WHERE unitid IS NOT NULL AND cipcode IS NOT NULL AND year IS NOT NULL
        GROUP BY unitid, cipcode, year
        """
    ).df()

    def add_growth(df: pd.DataFrame, source: str) -> pd.DataFrame:
        df = df.sort_values(["unitid", "cipcode", "year"])
        df["intl_growth"] = df.groupby(["unitid", "cipcode"])["intl_students"].diff()
        df["intl_growth_pct"] = df.groupby(["unitid", "cipcode"])["intl_students"].pct_change()
        df["source"] = source
        return df

    ipeds_growth = add_growth(ipeds, "ipeds")
    foia_growth = add_growth(foia, "foia")

    combined_growth = (
        ipeds_growth.merge(
            foia_growth,
            on=["unitid", "cipcode", "year"],
            suffixes=("_ipeds", "_foia"),
            how="inner",
        )
        .dropna(subset=["intl_growth_ipeds", "intl_growth_foia"])
    )

    # Binned scatterplot of growth correlation
    plt.figure(figsize=(7, 6))
    hb = plt.hexbin(
        combined_growth["intl_growth_ipeds"],
        combined_growth["intl_growth_foia"],
        gridsize=40,
        cmap="viridis",
        mincnt=1,
    )
    plt.colorbar(hb, label="Count")
    plt.xlabel("Intl growth (IPEDS)")
    plt.ylabel("Intl growth (FOIA)")
    plt.title("Institution x CIP international student growth correlation")
    plt.tight_layout()
    hexbin_path = os.path.join(FIG_DIR, "intl_growth_binned_scatter.png")
    plt.savefig(hexbin_path, dpi=300)
    plt.close()
    print(f"Saved binned scatterplot to {hexbin_path}")

    # Time series of totals over all programs/CIPs
    ipeds_tot = ipeds_growth.groupby("year", as_index=False)["intl_students"].sum()
    foia_tot = foia_growth.groupby("year", as_index=False)["intl_students"].sum()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=ipeds_tot, x="year", y="intl_students", label="IPEDS")
    sns.lineplot(data=foia_tot, x="year", y="intl_students", label="FOIA")
    plt.ylabel("International students")
    plt.title("International students over time (all institutions/CIPs)")
    plt.tight_layout()
    line_path = os.path.join(FIG_DIR, "intl_students_over_time.png")
    plt.savefig(line_path, dpi=300)
    plt.close()
    print(f"Saved totals line plot to {line_path}")

    # Optional: save growth datasets
    out_growth = os.path.join(root, "data", "int", "intl_student_growth_ipeds_vs_foia.parquet")
    combined_growth.to_parquet(out_growth, index=False)
    print(f"Wrote merged growth data to {out_growth}")


if __name__ == "__main__":
    main()
