# Detect economics relabels in IPEDS completions data and plot OPT usage
# for those programs using F-1 FOIA records.
#
# An "economics relabel" occurs when a master's economics program (unitid × CIP)
# with at least 10 graduates in year t-1 that is coded to an economics CIP
# outside of Econometrics (45.0603) experiences a ≥50% drop in total degrees
# in year t, while the same institution sees a comparable rise in Econometrics
# degrees. The relabel indicator is set on the program-year where the drop
# occurs (year t).

import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import duckdb as ddb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *  # noqa: F401,F403

BASE_FONT_SIZE = 12 * 1.4
IHMP_COLORS = {
    "IHMP": "#2e8b57",      # medium sea green
    "Non-IHMP": "#e07a5f",  # terracotta
}
PALETTE_SEQ = [IHMP_COLORS["IHMP"], IHMP_COLORS["Non-IHMP"], "#4c78a8", "#a05195", "#ffb000"]

INT_FOLDER = f"{root}/data/int/int_files_nov2025"
IPEDS_PATH = f"{INT_FOLDER}/ipeds_completions_all.parquet"
FOIA_PATH = f"{root}/data/int/foia_sevp_combined_raw.parquet"
F1_INST_CW_PATH = f"{INT_FOLDER}/f1_inst_unitid_crosswalk.parquet"
FIG_DIR = Path(f"{root}/figures")
ECON_PREFIX = "4506"  # Economics 4-digit CIP prefix
ECONOMETRICS_CIP = "450603"
RELABEL_OUTPUT = Path(f"{INT_FOLDER}/econ_relabels.parquet")
RELABEL_YEAR_MIN = 2015
RELABEL_YEAR_MAX = 2019
PLOT_YEAR_MIN = 2010
PLOT_YEAR_MAX = 2022
CONTROL_CIP_PREFIXES = ["40"]  # control CIP group (e.g., CS 11xxxx)
SINGLE_YEAR_DROP_PCT = 0.5  # minimum drop pct in single year to count as relabel
MIN_PREV_TOTAL = 10  # minimum prior year total to consider relabel
AVG_DROP_PCT = 0.5  # minimum avg drop pct over 4 years to count as relabel
TOP_COMMON_RELABELS = 20
PICK_FIRST_RELABEL = True  # True: keep first relabel year; False: keep largest drop_pct
YVAR = "avg_tuition"
YVAR_LABELS = {
    "opt_share": "Share of F-1s using OPT",
    "opt_stem_share": "Share of F-1s using STEM OPT",
    "status_change_share": "Share with status change",
    "opt_years_avg": "Average OPT years",
    "f1_share_of_ctotalt": "FOIA share of IPEDS ctotalt",
    "f1_share_of_cnralt": "FOIA share of IPEDS cnralt",
    "avg_tuition": "Average tuition (USD)",
}
RELABEL_SPECS = [
    {
        "name": "econ_to_econometrics",
        "source_prefix": "4506",
        "source_exclude_exact": ["450603"],
        "target_prefixes": [],
        "target_exact": ["450603"],
    },
    # {
    #     "name": "finance_to_applied_math",
    #     "source_prefix": "5208",  # Finance and Financial Management Services
    #     "source_exclude_exact": [],
    #     "target_prefixes": ["2703", "2705"],  # Applied Math prefix
    #     "target_exact": ["270301"],  # Applied Mathematics, General
    # },
]

# Candidate column names to make FOIA ingestion resilient to schema changes
FOIA_INST_COLS = ["school_name", "f1_inst_row_num", "inst_row_num", "school_id", "f1_inst_id"]
FOIA_CIP_COLS = ["major_1_cip_code", "program_cip_code", "cipcode", "cip_code", "cip"]
FOIA_PROG_END_COLS = ["program_end_date", "program_end_dt"]
FOIA_OPT_START_COLS = [
    "opt_authorization_start_date",
    "authorization_start_date",
    "opt_employer_start_date",
]
FOIA_OPT_END_COLS = [
    "authorization_end_date",
    "opt_employer_end_date",
    "opt_authorization_end_date"
]
FOIA_STATUS_COLS = ["requested_status", "status_code", "status"]
FOIA_STUDENT_KEY_COLS = ["student_key", "individual_key", "student_unique_id"]
FOIA_YEAR_COLS = ["year", "reporting_year", "data_year"]
FOIA_EDU_LEVEL_COLS = ["student_edu_level_desc", "edu_level", "education_level_desc"]
FOIA_TUITION_COLS = ["tuition__fees", "tuition_fees", "tuition", "tuition_fees_usd"]
CW_INST_COLS = ["school_name", "f1_inst_row_num", "inst_row_num", "school_id", "f1_inst_id"]
CW_UNITID_COLS = ["unitid", "UNITID", "ipeds_unitid"]


def first_present(cols: Iterable[str], candidates: Iterable[str], label: str) -> str:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise ValueError(f"Could not find {label}. Available columns: {sorted(cols)}")


def yvar_label(yvar: str) -> str:
    return YVAR_LABELS.get(yvar, yvar)


def normalize_cip_sql(colname: str) -> str:
    """Strip non-digits and cast to integer for robust joins."""
    return f"TRY_CAST(REGEXP_REPLACE(CAST({colname} AS VARCHAR), '[^0-9]', '', 'g') AS INTEGER)"


def _cip_prefix_pred(col: str, prefix: str) -> str:
    return f"LPAD(CAST({col} AS VARCHAR), 6, '0') LIKE '{prefix}%'"


def _cip_exact_pred(col: str, code: str) -> str:
    return f"LPAD(CAST({col} AS VARCHAR), 6, '0') = '{code}'"


def _spec_cip_pred(spec: dict, col: str = "cipcode") -> str:
    parts = [_cip_prefix_pred(col, spec["source_prefix"])]
    for pref in spec["target_prefixes"]:
        parts.append(_cip_prefix_pred(col, pref))
    for exact in spec["target_exact"]:
        parts.append(_cip_exact_pred(col, exact))
    return " OR ".join(parts) if parts else "FALSE"


# CIP filters used for FOIA subset (any source or target code/prefix)
FOIA_CIP_FILTERS = []
for spec in RELABEL_SPECS:
    FOIA_CIP_FILTERS.append(_spec_cip_pred(spec))
FOIA_CIP_WHERE = " OR ".join(FOIA_CIP_FILTERS) if FOIA_CIP_FILTERS else "TRUE"
RELABEL_TYPE_PREDICATES = " OR ".join(
    [f"(r.relabel_type = '{spec['name']}' AND ({_spec_cip_pred(spec)}))" for spec in RELABEL_SPECS]
) if RELABEL_SPECS else "FALSE"
CONTROL_CIP_WHERE = " OR ".join([_cip_prefix_pred("cipcode", pref) for pref in CONTROL_CIP_PREFIXES]) if CONTROL_CIP_PREFIXES else "FALSE"


def detect_econ_relabels(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Identify economics relabel events at the program-year level.

    Returns a DataFrame with program-year counts and a relabel flag.
    """
    if not os.path.exists(IPEDS_PATH):
        raise FileNotFoundError(f"Missing IPEDS completions parquet at {IPEDS_PATH}")

    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw AS SELECT * FROM read_parquet('{IPEDS_PATH}')")

    all_results: list[pd.DataFrame] = []
    min_year = 2004
    max_year = 2024

    for spec in RELABEL_SPECS:
        src_pred = _cip_prefix_pred("cipcode", spec["source_prefix"])
        exclude_clause = (
            " AND cip6 NOT IN (" + ", ".join([f"'{c}'" for c in spec["source_exclude_exact"]]) + ")"
            if spec["source_exclude_exact"]
            else ""
        )

        target_parts = []
        for pref in spec["target_prefixes"]:
            target_parts.append(_cip_prefix_pred("cipcode", pref))
        for exact in spec["target_exact"]:
            target_parts.append(_cip_exact_pred("cipcode", exact))
        target_pred = " OR ".join(target_parts) if target_parts else "FALSE"

        result = con.sql(
            f"""
            WITH masters AS (
                SELECT
                    unitid,
                    CAST(year AS INTEGER) AS year,
                    CAST(cipcode AS INTEGER) AS cipcode,
                    CAST(ctotalt AS DOUBLE) AS ctotalt,
                    LPAD(CAST(cipcode AS VARCHAR), 6, '0') AS cip6,
                    (CAST(awlevel AS INTEGER) = 7) AS is_master
                FROM ipeds_raw
                WHERE unitid IS NOT NULL AND cipcode IS NOT NULL AND share_intl >= 0.2
            ),
        source AS (
            SELECT
                unitid,
                year,
                SUM(ctotalt) AS source_total
            FROM masters
            WHERE is_master AND {src_pred}{exclude_clause}
            GROUP BY unitid, year
        ),
        source_completed AS (
            SELECT
                b.unitid,
                y.year AS year,
                COALESCE(s.source_total, 0) AS source_total,
                LAG(COALESCE(s.source_total, 0)) OVER(PARTITION BY b.unitid ORDER BY y.year) AS source_prev,
                AVG(COALESCE(s.source_total, 0)) OVER(PARTITION BY b.unitid ORDER BY y.year ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS source_prev5_avg,
                AVG(COALESCE(s.source_total, 0)) OVER(PARTITION BY b.unitid ORDER BY y.year ROWS BETWEEN 0 FOLLOWING AND 4 FOLLOWING) AS source_next5_avg
            FROM (
                SELECT DISTINCT unitid FROM source
            ) b
            CROSS JOIN (
                SELECT * FROM generate_series(
                    {min_year},
                    {max_year}
                ) AS y(year)
            ) y
            LEFT JOIN source s
              ON s.unitid = b.unitid AND s.year = y.year
        ),
        target AS (
            SELECT
                unitid,
                year,
                SUM(ctotalt) AS target_total
            FROM masters
            WHERE is_master AND ({target_pred})
            GROUP BY unitid, year
        ),
        target_completed AS (
            SELECT
                tb.unitid,
                y.year AS year,
                COALESCE(t.target_total, 0) AS target_total,
                LAG(COALESCE(t.target_total, 0)) OVER(PARTITION BY tb.unitid ORDER BY y.year) AS target_prev,
                AVG(COALESCE(t.target_total, 0)) OVER(PARTITION BY tb.unitid ORDER BY y.year ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS target_prev5_avg,
                AVG(COALESCE(t.target_total, 0)) OVER(PARTITION BY tb.unitid ORDER BY y.year ROWS BETWEEN 0 FOLLOWING AND 4 FOLLOWING) AS target_next5_avg
            FROM (
                SELECT DISTINCT unitid FROM target
            ) tb
            CROSS JOIN (
                SELECT * FROM generate_series(
                    {min_year},
                    {max_year}
                ) AS y(year)
            ) y
            LEFT JOIN target t
              ON t.unitid = tb.unitid AND t.year = y.year
        ),
        relabels AS (
            SELECT s.unitid, s.year, source_total, target_total, 
                s.source_prev,
                t.target_prev,
                source_total - source_prev AS source_drop,
                (source_total - source_prev)::DECIMAL / (CASE WHEN source_prev = 0 THEN 0.01 ELSE source_prev END) AS source_drop_pct,
                target_total - target_prev AS target_increase,
                (target_total - target_prev)::DECIMAL / (CASE WHEN target_prev = 0 THEN 0.01 ELSE target_prev END) AS target_increase_pct,
                source_prev5_avg, source_next5_avg,
                target_prev5_avg, target_next5_avg,
                source_next5_avg - source_prev5_avg AS avg5_source_drop,
                (source_next5_avg - source_prev5_avg)::DECIMAL / (CASE WHEN source_prev5_avg = 0 THEN 0.01 ELSE source_prev5_avg END) AS avg5_source_drop_pct,
                target_next5_avg - target_prev5_avg AS avg5_target_increase,
                (target_next5_avg - target_prev5_avg)::DECIMAL / (CASE WHEN target_prev5_avg = 0 THEN 0.01 ELSE target_prev5_avg END) AS avg5_target_increase_pct
            FROM
                source_completed s JOIN target_completed t ON s.unitid = t.unitid AND s.year = t.year
        )
        SELECT
            unitid,
            year,
            source_total,
            source_prev AS source_total_prev,
            source_total AS source_total_intl,
            source_prev AS source_total_intl_prev,
            target_total,
            target_prev AS target_total_prev,
            target_total AS target_total_intl,
            target_prev AS target_total_intl_prev,
            source_drop,
            source_drop_pct,
            target_increase,
            target_increase_pct,
                source_prev5_avg AS source_prev5_avg,
                avg5_source_drop,
                avg5_source_drop_pct,
                avg5_target_increase,
                avg5_target_increase_pct,
                CASE
                    WHEN source_prev5_avg >= {MIN_PREV_TOTAL}
                     AND source_drop_pct <= -{SINGLE_YEAR_DROP_PCT}
                     AND source_drop < 0
                     AND target_increase >= -0.5 * source_drop
                     AND avg5_source_drop_pct <= -{AVG_DROP_PCT}
                     AND avg5_source_drop < 0
                     AND avg5_target_increase >= -0.5 * avg5_source_drop
                     THEN 1 ELSE 0
                END AS relabel_flag,
                '{spec["name"]}' AS relabel_type
            FROM relabels
            """
        ).df()

        result = result[
            (result["year"] >= RELABEL_YEAR_MIN) & (result["year"] <= RELABEL_YEAR_MAX)
        ].reset_index(drop=True)
        all_results.append(result)

    base_result = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    if base_result.empty:
        return base_result

    if PICK_FIRST_RELABEL:
        base_result["first_relabel_year"] = (
            base_result.loc[base_result["relabel_flag"] == 1]
            .groupby(["unitid", "relabel_type"])["year"]
            .transform("min")
        )
        base_result.loc[
            (base_result["relabel_flag"] == 1)
            & (base_result["year"] != base_result["first_relabel_year"]),
            "relabel_flag",
        ] = 0
        base_result = base_result.drop(columns=["first_relabel_year"])
    else:
        if "drop_pct" in base_result.columns:
            max_drop = (
                base_result.loc[base_result["relabel_flag"] == 1]
                .groupby(["unitid", "relabel_type"])["drop_pct"]
                .transform("max")
            )
            base_result["max_drop_pct"] = max_drop
            base_result.loc[
                (base_result["relabel_flag"] == 1)
                & (base_result["drop_pct"] < base_result["max_drop_pct"]),
                "relabel_flag",
            ] = 0
            base_result = base_result.drop(columns=["max_drop_pct"])

    events = (
        base_result.loc[base_result["relabel_flag"] == 1]
        .sort_values("year")
        .groupby(["unitid", "relabel_type"], as_index=False)
        .first()
        .rename(columns={"year": "relabel_year"})
    )

    ipeds_ann = con.sql(
        f"""
        SELECT
            unitid,
            CAST(year AS INTEGER) AS year,
            SUM(cnralt) AS cnralt,
            SUM(ctotalt) AS ctotalt
        FROM (SELECT *, LPAD(CAST(cipcode AS VARCHAR), 6, '0') AS cip6 FROM ipeds_raw)
        WHERE CAST(awlevel AS INTEGER) = 7
          AND ({src_pred}{exclude_clause} OR ({target_pred}))
        GROUP BY unitid, year
        """
    ).df()
    if ipeds_ann.empty:
        return base_result
    years = pd.DataFrame({"year": range(min_year, max_year)})

    events["key"] = 1
    years["key"] = 1
    panel = events.merge(years, on="key", how="left").drop(columns=["key"])
    panel = panel.merge(ipeds_ann, on=["unitid", "year"], how="left")

    # repeat event-level metrics across years
    event_metrics = (
        base_result.loc[base_result["relabel_flag"] == 1]
        .rename(columns={"year": "relabel_year"})
        .drop_duplicates(subset=["unitid", "relabel_type", "relabel_year"])
    )
    panel = panel.merge(
        event_metrics.drop(columns=["relabel_flag"]),
        on=["unitid", "relabel_type", "relabel_year"],
        how="left",
        suffixes=("", "_event"),
    )

    panel["relabel_flag"] = 1
    panel["event_flag"] = (panel["year"] == panel["relabel_year"]).astype(int)
    return panel


def find_common_relabels(con: ddb.DuckDBPyConnection, top_n: int = TOP_COMMON_RELABELS) -> pd.DataFrame:
    """
    Scan master's programs (all CIPs) for relabel-like events in 2016-2019 and report the
    most common source->target CIP pairs where a drop in one CIP coincides with an increase
    in another CIP at the same unit and year.
    """
    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw AS SELECT * FROM read_parquet('{IPEDS_PATH}')")

    df = con.sql(
        f"""
        WITH masters AS (
            SELECT
                unitid,
                CAST(year AS INTEGER) AS year,
                CAST(cipcode AS INTEGER) AS cipcode,
                LPAD(CAST(cipcode AS VARCHAR), 6, '0') AS cip6,
                CAST(ctotalt AS DOUBLE) AS ctotalt,
                CAST(cnralt AS DOUBLE) AS cnralt
            FROM ipeds_raw
            WHERE unitid IS NOT NULL
              AND cipcode IS NOT NULL
              AND CAST(awlevel AS INTEGER) = 7
        ),
        lagged AS (
            SELECT
                unitid,
                year,
                cip6,
                ctotalt,
                cnralt,
                cnralt/ctotalt AS share_intl,
                LAG(ctotalt) OVER (PARTITION BY unitid, cip6 ORDER BY year) AS prev_total,
                LAG(year) OVER (PARTITION BY unitid, cip6 ORDER BY year) AS prev_year,
                AVG(ctotalt) OVER (
                    PARTITION BY unitid, cip6
                    ORDER BY year
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) AS avg_prev5,
                AVG(ctotalt) OVER (
                    PARTITION BY unitid, cip6
                    ORDER BY year
                    ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING
                ) AS avg_future
            FROM masters
        ),
        drops AS (
            SELECT
                unitid,
                year,
                cip6 AS source_cip6,
                ctotalt AS curr_total,
                avg_prev5,
                CASE WHEN avg_prev5 IS NOT NULL AND avg_prev5 > 0 THEN (avg_prev5 - ctotalt) / avg_prev5 ELSE NULL END AS drop_pct,
                CASE WHEN avg_prev5 IS NOT NULL THEN avg_prev5 - ctotalt ELSE NULL END AS drop_count
            FROM lagged
            WHERE avg_prev5 >= {MIN_PREV_TOTAL}
              AND prev_total - ctotalt > 0
              AND (avg_prev5 - ctotalt) / avg_prev5 >= {SINGLE_YEAR_DROP_PCT} AND share_intl >= 0.5
              AND year BETWEEN 2016 AND 2019
        ),
        increases AS (
            SELECT
                unitid,
                year,
                cip6 AS target_cip6,
                ctotalt AS target_curr,
                avg_prev5 AS target_prev_avg
            FROM lagged
            WHERE year BETWEEN 2016 AND 2019
        ),
        paired AS (
            SELECT
                d.unitid,
                d.year,
                d.source_cip6,
                i.target_cip6,
                d.drop_count,
                d.drop_pct,
                CASE
                    WHEN COALESCE(i.target_prev_avg, 0) > 0
                    THEN (COALESCE(i.target_curr, 0) - COALESCE(i.target_prev_avg, 0)) / i.target_prev_avg
                    ELSE NULL
                END AS target_increase_pct,
                COALESCE(i.target_curr, 0) - COALESCE(i.target_prev_avg, 0) AS target_increase
            FROM drops d
            JOIN increases i
              ON d.unitid = i.unitid AND d.year = i.year
            WHERE COALESCE(i.target_curr, 0) - COALESCE(i.target_prev_avg, 0) >= 0.5 * d.drop_count
              AND d.source_cip6 <> i.target_cip6
        )
        SELECT
            source_cip6,
            target_cip6,
            COUNT(*) AS event_count,
            AVG(drop_pct) AS avg_drop_pct,
            AVG(target_increase_pct) AS avg_target_increase_pct
        FROM paired
        GROUP BY source_cip6, target_cip6
        ORDER BY event_count DESC, avg_drop_pct DESC, avg_target_increase_pct DESC
        LIMIT {top_n}
        """
    ).df()
    return df


def compute_opt_usage(
    con: ddb.DuckDBPyConnection, relabel_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute OPT usage shares by calendar year with hue = relabel year.
    """
    if relabel_df.empty:
        raise ValueError("No relabel events found; cannot compute OPT usage.")

    if not os.path.exists(FOIA_PATH):
        raise FileNotFoundError(f"Missing FOIA parquet at {FOIA_PATH}")
    if not os.path.exists(F1_INST_CW_PATH):
        raise FileNotFoundError(f"Missing F-1 institution crosswalk at {F1_INST_CW_PATH}")

    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{FOIA_PATH}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_cw AS SELECT * FROM read_parquet('{F1_INST_CW_PATH}')")

    relabel_events = (
        relabel_df[["unitid", "year", "relabel_type", "ctotalt", "cnralt", "relabel_year"]]
        .drop_duplicates()
    )
    print(f"Registering {relabel_df['event_flag'].sum()} relabel events in DuckDB temp table.")
    con.register("relabel_events_py", relabel_events)
    con.sql("CREATE OR REPLACE TEMP VIEW relabel_events AS SELECT * FROM relabel_events_py")

    foia_cols = [row[0] for row in con.sql("DESCRIBE foia_raw").fetchall()]
    cw_cols = [row[0] for row in con.sql("DESCRIBE f1_inst_cw").fetchall()]

    foia_inst_col = first_present(foia_cols, FOIA_INST_COLS, "FOIA institution column")
    foia_cip_col = first_present(foia_cols, FOIA_CIP_COLS, "FOIA CIP column")
    foia_end_col = first_present(foia_cols, FOIA_PROG_END_COLS, "program end date column")
    foia_student_col = first_present(foia_cols, FOIA_STUDENT_KEY_COLS, "student identifier column")
    foia_tuition_col = first_present(foia_cols, FOIA_TUITION_COLS, "tuition column")
    opt_start_col: Optional[str] = None
    for cand in FOIA_OPT_START_COLS:
        if cand.lower() in [c.lower() for c in foia_cols]:
            opt_start_col = next(c for c in foia_cols if c.lower() == cand.lower())
            break
    foia_year_col = None
    for cand in FOIA_YEAR_COLS:
        if cand.lower() in [c.lower() for c in foia_cols]:
            foia_year_col = next(c for c in foia_cols if c.lower() == cand.lower())
            break
    if opt_start_col is None:
        raise ValueError("Could not locate an OPT authorization start column in FOIA data.")
    opt_end_col = None
    for cand in FOIA_OPT_END_COLS:
        if cand.lower() in [c.lower() for c in foia_cols]:
            opt_end_col = next(c for c in foia_cols if c.lower() == cand.lower())
            break
    if opt_end_col is None:
        raise ValueError("Could not locate an OPT authorization end column in FOIA data.")
    status_col = first_present(foia_cols, FOIA_STATUS_COLS, "requested status column")
    foia_edu_col = first_present(foia_cols, FOIA_EDU_LEVEL_COLS, "education level column")
    status_col = first_present(foia_cols, FOIA_STATUS_COLS, "requested status column")

    cw_inst_col = first_present(cw_cols, CW_INST_COLS, "crosswalk institution column")
    cw_unitid_col = first_present(cw_cols, CW_UNITID_COLS, "crosswalk unitid column")

    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )

    opt_usage = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                {normalize_cip_sql(foia_cip_col)} AS cipcode,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                {'CAST(' + foia_year_col + ' AS INTEGER)' if foia_year_col else 'NULL'} AS reported_year,
                employer_name,
                employment_opt_type,
                {opt_end_col},
                {foia_tuition_col} AS tuition,
                program_end_date,
                {status_col} AS requested_status
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE {foia_end_col} IS NOT NULL
              AND fr.{foia_edu_col} = 'MASTER''S'
              AND ({FOIA_CIP_WHERE})
              {year_match_clause}
        ),
        relevant_foia AS (
            SELECT *
            FROM foia_base
            WHERE unitid IS NOT NULL
              AND cipcode IS NOT NULL
              AND grad_year IS NOT NULL
        ),
        flagged AS (
            SELECT
                f.*,
                r.ctotalt,
                r.cnralt,
                r.relabel_year,
                r.relabel_type
            FROM relevant_foia f
            JOIN relabel_events r
              ON f.unitid = r.unitid AND f.grad_year = r.year
             AND ({RELABEL_TYPE_PREDICATES})
        ),
        student_level AS (
            SELECT
                unitid,
                cipcode,
                grad_year,
                ctotalt, cnralt,
                MAX(CASE WHEN employer_name IS NOT NULL THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'POST-COMPLETION' THEN 1 ELSE 0 END) AS opt_ind_old,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                CASE WHEN MAX({opt_end_col}) IS NOT NULL THEN DATE_DIFF('day', MAX({foia_end_col}), MAX({opt_end_col})) / 365.25 ELSE 0 END AS opt_years,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                student_id,
                relabel_year,
                relabel_type
            FROM flagged
            GROUP BY unitid, cipcode, grad_year, student_id, relabel_year, relabel_type, ctotalt, cnralt
        ),
        dedup AS (
            SELECT * FROM student_level
        )
        SELECT
            grad_year AS calendar_year,
            relabel_year AS relabel_year,
            relabel_type,
            cnralt,
            ctotalt,
            AVG(avg_tuition) AS avg_tuition,
            COUNT(DISTINCT student_id) AS total_grads,
            COUNT(DISTINCT CASE WHEN opt_ind = 1 THEN student_id END) AS opt_users,
            COUNT(DISTINCT CASE WHEN opt_stem_ind = 1 THEN student_id END) AS opt_stem_users,
            COUNT(DISTINCT CASE WHEN status_change_ind = 1 THEN student_id END) AS status_change_users,
            SUM(opt_years) AS total_opt_years
        FROM dedup
        WHERE relabel_year IS NOT NULL
          AND grad_year IS NOT NULL
        GROUP BY calendar_year, relabel_year, relabel_type, cnralt, ctotalt
        ORDER BY calendar_year, relabel_year, relabel_type, cnralt, ctotalt
        """
    ).df()

    opt_usage["opt_share"] = opt_usage["opt_users"] / opt_usage["total_grads"]
    opt_usage['opt_stem_share'] = opt_usage["opt_stem_users"] / opt_usage["total_grads"]
    opt_usage["status_change_share"] = opt_usage["status_change_users"] / opt_usage["total_grads"]
    opt_usage["opt_years_avg"] = opt_usage["total_opt_years"] / opt_usage["total_grads"]
    opt_usage['f1_share_of_ctotalt'] = opt_usage['total_grads'] / opt_usage['ctotalt']
    opt_usage['f1_share_of_cnralt'] = opt_usage['total_grads'] / opt_usage['cnralt']
    opt_usage["tuition_total"] = opt_usage["avg_tuition"] * opt_usage["total_grads"]
    return opt_usage


def compute_opt_usage_event_time(opt_usage: pd.DataFrame) -> pd.DataFrame:
    """
    Stack relabel cohorts relative to their relabel year: event_t = calendar_year - relabel_year.
    Aggregates totals and shares at the grouping stage (by event_t x relabel_type).
    """
    df = opt_usage[opt_usage['calendar_year'].between(PLOT_YEAR_MIN, PLOT_YEAR_MAX)].copy()
    df["event_t"] = df["calendar_year"] - df["relabel_year"]
    grouped = (
        df.groupby(["event_t", "relabel_type"], as_index=False)
        .agg(
            total_grads=("total_grads", "sum"),
            opt_users=("opt_users", "sum"),
            opt_stem_users=("opt_stem_users", "sum"),
            total_opt_years=("total_opt_years", "sum"),
            total_status_change_users=("status_change_users", "sum"),
            tuition_total=("tuition_total", "sum"),
            ctotalt=("ctotalt", "sum"),
            cnralt=("cnralt", "sum")
        )
    )
    grouped["opt_share"] = grouped["opt_users"] / grouped["total_grads"]
    grouped["opt_stem_share"] = grouped["opt_stem_users"] / grouped["total_grads"]
    grouped["opt_years_avg"] = grouped["total_opt_years"] / grouped["total_grads"]
    grouped["status_change_share"] = grouped["total_status_change_users"] / grouped["total_grads"]
    grouped['f1_share_of_ctotalt'] = grouped['total_grads'] / grouped['ctotalt']
    grouped['f1_share_of_cnralt'] = grouped['total_grads'] / grouped['cnralt']
    grouped["avg_tuition"] = grouped["tuition_total"] / grouped["total_grads"]
    return grouped


def compute_control_opt_usage_event_time(
    con: ddb.DuckDBPyConnection, relabel_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute OPT usage for a control CIP group (e.g., 11xxxx) relative to the same relabel events.
    Aggregates at the event-time stage (event_t x relabel_type) to keep averaging at grouping.
    """
    if relabel_df.empty:
        raise ValueError("No relabel events found; cannot compute control OPT usage.")

    if not os.path.exists(FOIA_PATH):
        raise FileNotFoundError(f"Missing FOIA parquet at {FOIA_PATH}")
    if not os.path.exists(F1_INST_CW_PATH):
        raise FileNotFoundError(f"Missing F-1 institution crosswalk at {F1_INST_CW_PATH}")

    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{FOIA_PATH}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_cw AS SELECT * FROM read_parquet('{F1_INST_CW_PATH}')")

    relabel_events = (
        relabel_df[["unitid", "year", "relabel_type", "ctotalt", "cnralt", "relabel_year"]]
        .drop_duplicates()
    )
    con.register("relabel_events_control_py", relabel_events)
    con.sql("CREATE OR REPLACE TEMP VIEW relabel_events_control AS SELECT * FROM relabel_events_control_py")

    foia_cols = [row[0] for row in con.sql("DESCRIBE foia_raw").fetchall()]
    cw_cols = [row[0] for row in con.sql("DESCRIBE f1_inst_cw").fetchall()]

    foia_inst_col = first_present(foia_cols, FOIA_INST_COLS, "FOIA institution column")
    foia_cip_col = first_present(foia_cols, FOIA_CIP_COLS, "FOIA CIP column")
    foia_end_col = first_present(foia_cols, FOIA_PROG_END_COLS, "program end date column")
    foia_student_col = first_present(foia_cols, FOIA_STUDENT_KEY_COLS, "student identifier column")
    foia_tuition_col = first_present(foia_cols, FOIA_TUITION_COLS, "tuition column")
    opt_start_col: Optional[str] = None
    for cand in FOIA_OPT_START_COLS:
        if cand.lower() in [c.lower() for c in foia_cols]:
            opt_start_col = next(c for c in foia_cols if c.lower() == cand.lower())
            break
    foia_year_col = None
    for cand in FOIA_YEAR_COLS:
        if cand.lower() in [c.lower() for c in foia_cols]:
            foia_year_col = next(c for c in foia_cols if c.lower() == cand.lower())
            break
    if opt_start_col is None:
        raise ValueError("Could not locate an OPT authorization start column in FOIA data.")
    opt_end_col = None
    for cand in FOIA_OPT_END_COLS:
        if cand.lower() in [c.lower() for c in foia_cols]:
            opt_end_col = next(c for c in foia_cols if c.lower() == cand.lower())
            break
    if opt_end_col is None:
        raise ValueError("Could not locate an OPT authorization end column in FOIA data.")
    foia_edu_col = first_present(foia_cols, FOIA_EDU_LEVEL_COLS, "education level column")

    status_col = first_present(foia_cols, FOIA_STATUS_COLS, "requested status column")
    cw_inst_col = first_present(cw_cols, CW_INST_COLS, "crosswalk institution column")
    cw_unitid_col = first_present(cw_cols, CW_UNITID_COLS, "crosswalk unitid column")

    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )

    control_calendar = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                {normalize_cip_sql(foia_cip_col)} AS cipcode,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                {'CAST(' + foia_year_col + ' AS INTEGER)' if foia_year_col else 'NULL'} AS reported_year,
                employer_name,
                employment_opt_type,
                {opt_end_col},
                {foia_tuition_col} AS tuition,
                {foia_end_col} AS program_end_date,
                {status_col} AS requested_status
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE {foia_end_col} IS NOT NULL
              AND fr.{foia_edu_col} = 'MASTER''S'
              AND ({CONTROL_CIP_WHERE})
              {year_match_clause}
        ),
        relevant_foia AS (
            SELECT *
            FROM foia_base
            WHERE unitid IS NOT NULL
              AND cipcode IS NOT NULL
              AND grad_year IS NOT NULL
        ),
        flagged AS (
            SELECT
                f.*,
                r.relabel_year AS relabel_year,
                r.relabel_type,
                r.ctotalt,
                r.cnralt
            FROM relevant_foia f
            JOIN relabel_events_control r
              ON f.unitid = r.unitid AND f.grad_year = r.year
        ),
        student_level AS (
            SELECT
                unitid,
                cipcode,
                grad_year,
                ctotalt, cnralt,
                MAX(CASE WHEN employer_name IS NOT NULL THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'POST-COMPLETION' THEN 1 ELSE 0 END) AS opt_ind_old,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                CASE WHEN MAX({opt_end_col}) IS NOT NULL THEN DATE_DIFF('day', MAX(program_end_date), MAX({opt_end_col})) / 365.25 ELSE 0 END AS opt_years,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                student_id,
                relabel_year,
                relabel_type
            FROM flagged
            GROUP BY unitid, cipcode, grad_year, student_id, relabel_year, relabel_type, ctotalt, cnralt
        ),
        calendar_level AS (
            SELECT
                grad_year AS calendar_year,
                relabel_year,
                relabel_type,
                cnralt,
                ctotalt,
                AVG(avg_tuition) AS avg_tuition,
                COUNT(DISTINCT student_id) AS total_grads,
                COUNT(DISTINCT CASE WHEN opt_ind = 1 THEN student_id END) AS opt_users,
                COUNT(DISTINCT CASE WHEN opt_stem_ind = 1 THEN student_id END) AS opt_stem_users,
                COUNT(DISTINCT CASE WHEN status_change_ind = 1 THEN student_id END) AS status_change_users,
                SUM(opt_years) AS total_opt_years
            FROM student_level
            WHERE relabel_year IS NOT NULL
              AND grad_year IS NOT NULL
            GROUP BY calendar_year, relabel_year, relabel_type, cnralt, ctotalt
        )
        SELECT
            calendar_year,
            relabel_year AS relabel_year,
            relabel_type,
            avg_tuition,
            total_grads,
            opt_users,
            opt_stem_users,
            status_change_users,
            total_opt_years,
            ctotalt,
            cnralt
        FROM calendar_level
        """
    ).df()

    control_calendar["opt_share"] = control_calendar["opt_users"] / control_calendar["total_grads"]
    control_calendar["opt_stem_share"] = control_calendar["opt_stem_users"] / control_calendar["total_grads"]
    control_calendar["status_change_share"] = control_calendar["status_change_users"] / control_calendar["total_grads"]
    control_calendar["opt_years_avg"] = control_calendar["total_opt_years"] / control_calendar["total_grads"]
    control_calendar["f1_share_of_ctotalt"] = control_calendar["total_grads"] / control_calendar["ctotalt"]
    control_calendar["f1_share_of_cnralt"] = control_calendar["total_grads"] / control_calendar["cnralt"]
    control_calendar["tuition_total"] = control_calendar["avg_tuition"] * control_calendar["total_grads"]

    control_calendar["event_t"] = control_calendar["calendar_year"] - control_calendar["relabel_year"]
    control_event = (
        control_calendar.groupby(["event_t", "relabel_type"], as_index=False)
        .agg(
            total_grads=("total_grads", "sum"),
            opt_users=("opt_users", "sum"),
            opt_stem_users=("opt_stem_users", "sum"),
            status_change_users=("status_change_users", "sum"),
            total_opt_years=("total_opt_years", "sum"),
            tuition_total=("tuition_total", "sum"),
            ctotalt=("ctotalt", "sum"),
            cnralt=("cnralt", "sum")
        )
    )
    control_event["opt_share"] = control_event["opt_users"] / control_event["total_grads"]
    control_event["opt_stem_share"] = control_event["opt_stem_users"] / control_event["total_grads"]
    control_event["status_change_share"] = control_event["status_change_users"] / control_event["total_grads"]
    control_event["opt_years_avg"] = control_event["total_opt_years"] / control_event["total_grads"]
    control_event["f1_share_of_ctotalt"] = control_event["total_grads"] / control_event["ctotalt"]
    control_event["f1_share_of_cnralt"] = control_event["total_grads"] / control_event["cnralt"]
    control_event["avg_tuition"] = control_event["tuition_total"] / control_event["total_grads"]
    return control_event    


def _compute_unit_level_foia_counts(
    con: ddb.DuckDBPyConnection,
    relabel_df: pd.DataFrame,
    foia_path: str = FOIA_PATH,
    inst_cw_path: str = F1_INST_CW_PATH,
) -> pd.DataFrame:
    """
    Compute FOIA total grads at the unitid x relabel_year x relabel_type level for sanity checks.
    """
    if relabel_df.empty:
        return pd.DataFrame()
    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{foia_path}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_cw AS SELECT * FROM read_parquet('{inst_cw_path}')")

    foia_cols = [row[0] for row in con.sql("DESCRIBE foia_raw").fetchall()]
    cw_cols = [row[0] for row in con.sql("DESCRIBE f1_inst_cw").fetchall()]

    foia_inst_col = first_present(foia_cols, FOIA_INST_COLS, "FOIA institution column")
    foia_cip_col = first_present(foia_cols, FOIA_CIP_COLS, "FOIA CIP column")
    foia_end_col = first_present(foia_cols, FOIA_PROG_END_COLS, "program end date column")
    foia_student_col = first_present(foia_cols, FOIA_STUDENT_KEY_COLS, "student identifier column")
    foia_year_col = first_present(foia_cols, FOIA_YEAR_COLS, "FOIA reporting year column")
    foia_edu_col = first_present(foia_cols, FOIA_EDU_LEVEL_COLS, "education level column")
    cw_inst_col = first_present(cw_cols, CW_INST_COLS, "crosswalk institution column")
    cw_unitid_col = first_present(cw_cols, CW_UNITID_COLS, "crosswalk unitid column")

    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )

    relabel_events = (
        relabel_df.loc[relabel_df["event_flag"] == 1, ["unitid", "year", "relabel_type"]]
        .drop_duplicates()
        .rename(columns={"year": "relabel_year"})
    )
    con.register("relabel_events_unit_py", relabel_events)
    con.sql("CREATE OR REPLACE TEMP VIEW relabel_events_unit AS SELECT * FROM relabel_events_unit_py")

    foia_unit = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                {normalize_cip_sql(foia_cip_col)} AS cipcode,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE {foia_end_col} IS NOT NULL
              AND fr.{foia_edu_col} = 'MASTER''S'
              {year_match_clause}
        ),
        relevant AS (
            SELECT *
            FROM foia_base
            WHERE unitid IS NOT NULL AND cipcode IS NOT NULL AND grad_year IS NOT NULL
        ),
        flagged AS (
            SELECT
                f.unitid,
                f.grad_year,
                f.student_id,
                r.relabel_year,
                r.relabel_type
            FROM relevant f
            JOIN relabel_events_unit r
              ON f.unitid = r.unitid
        )
        SELECT
            unitid,
            relabel_year,
            relabel_type,
            grad_year AS calendar_year,
            COUNT(DISTINCT student_id) AS total_grads
        FROM flagged
        GROUP BY unitid, relabel_year, relabel_type, calendar_year
        """
    ).df()
    return foia_unit


def plot_opt_usage(opt_usage: pd.DataFrame, yvar = 'opt_share', show: bool = False, save = False) -> Path:
    """Plot OPT usage by calendar year with hue = relabel year."""
    plt.rcParams.update({"font.size": BASE_FONT_SIZE})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = opt_usage[
        (opt_usage["calendar_year"] >= PLOT_YEAR_MIN) & (opt_usage["calendar_year"] <= PLOT_YEAR_MAX)
    ].copy()
    if data.empty:
        raise ValueError("No OPT usage data in the requested plot window.")

    sns.set(style="whitegrid")
    sns.set_palette(PALETTE_SEQ)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x="calendar_year",
        y=yvar,
        hue="relabel_year",
        marker="o",
        ax=ax,
    )
    ax.set_ylabel(yvar_label(yvar))
    ax.set_xlabel("Calendar year (program end year)")
    ax.set_title("")
    fig.tight_layout()

    # print figure to see
    print(fig)
    if save:
        out_path = FIG_DIR / "opt_usage_by_relabel_year.png"
        fig.savefig(out_path, dpi=300)
    
        return out_path
    
    if show:
        plt.show()
        
    plt.close(fig)


def plot_opt_usage_event_time(
    opt_usage_event: pd.DataFrame,
    control_event: Optional[pd.DataFrame] = None,
    yvar: str = "opt_share",
    show: bool = False,
    save: bool = False,
) -> Optional[Path]:
    """
    Plot OPT usage by event time (years relative to relabel year), stacked by relabel type.
    If a control dataset is provided, it is plotted alongside, labeled as control.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if opt_usage_event.empty:
        raise ValueError("No event-time OPT usage data to plot.")

    treated = opt_usage_event.copy()
    treated["series_label"] = "Economics"

    sns.set(style="whitegrid")
    sns.set_palette(PALETTE_SEQ)
    plt.rcParams.update({"font.size": BASE_FONT_SIZE})

    paths: dict[str, Path | None] = {"treated_only": None, "treated_control": None}

    # Treated-only plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=treated[treated["event_t"].between(-5, 4)],
        x="event_t",
        y=yvar,
        hue="series_label",
        marker="o",
        ax=ax1,
    )
    ax1.set_ylabel(yvar_label(yvar))
    ax1.set_xlabel("Years relative to relabel (t=0)")
    ax1.set_title("")
    ax1.axvline(x=0, linestyle="--", color="gray", linewidth=1)
    ax1.legend(title=None)
    fig1.tight_layout()
    treated_path = FIG_DIR / f"{yvar}_event_time_treated.png"
    if save:
        fig1.savefig(treated_path, dpi=300)
        paths["treated_only"] = treated_path
    if show:
        plt.show()
    plt.close(fig1)

    # Treated + control plot
    if control_event is not None and not control_event.empty:
        ctrl = control_event.copy()
        ctrl["series_label"] = "Physical Sciences"
        plot_df = pd.concat([treated, ctrl], ignore_index=True)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=plot_df[plot_df["event_t"].between(-5, 4)],
            x="event_t",
            y=yvar,
            hue="series_label",
            marker="o",
            ax=ax2,
        )
        ax2.set_ylabel(yvar_label(yvar))
        ax2.set_xlabel("Years relative to relabel (t=0)")
        ax2.set_title("")
        ax2.axvline(x=0, linestyle="--", color="gray", linewidth=1)
        ax2.legend(title=None)
        fig2.tight_layout()
        combined_path = FIG_DIR / f"{yvar}_event_time_treated_control.png"
        if save:
            fig2.savefig(combined_path, dpi=300)
            paths["treated_control"] = combined_path
        if show:
            plt.show()
        plt.close(fig2)

    # Prefer returning combined path if available, else treated-only
    return paths["treated_control"] or paths["treated_only"]


def main():
    con = ddb.connect()

    # 1) Detect economics relabels in IPEDS
    relabel_df = detect_econ_relabels(con)
    yvar = YVAR
    #yvar = "opt_stem_share"  # alternative: "opt_share"

    # Histogram of relabel events by year
    if not relabel_df.empty:
        plt.rcParams.update({"font.size": BASE_FONT_SIZE})
        sns.set(style="whitegrid")
        sns.set_palette(PALETTE_SEQ)
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=relabel_df[relabel_df["event_flag"] == 1],
            x="relabel_year",
            bins=range(RELABEL_YEAR_MIN, RELABEL_YEAR_MAX + 2),
            discrete=True,
            hue="relabel_type",
            multiple="dodge",
        )
        plt.xlabel("Relabel year")
        plt.ylabel("Count of relabel events")
        plt.title("")
        plt.tight_layout()
        plt.show()

    # 2) Compute OPT usage and plot
    opt_usage = compute_opt_usage(con, relabel_df)
    fig_path = plot_opt_usage(opt_usage, yvar=yvar, show=True, save=True)
    print(f"Saved OPT usage plot to {fig_path}")

    # 3) Event-time aggregation and plot
    opt_usage_event = compute_opt_usage_event_time(opt_usage)
    control_event = compute_control_opt_usage_event_time(con, relabel_df)
    _ = plot_opt_usage_event_time(opt_usage_event, control_event=control_event, yvar=yvar, show=True, save=True)

    # 4) Sanity check: IPEDS intl totals vs FOIA grads (by relabel_year x relabel_type)
    relabel_events = relabel_df[relabel_df["relabel_flag"] == 1].copy()
    relabel_events["ipeds_current_intl"] = relabel_events["source_total_intl"].fillna(0) + relabel_events["target_total_intl"].fillna(0)
    relabel_events["ipeds_prev_intl"] = relabel_events["source_total_intl_prev"].fillna(0) + relabel_events["target_total_intl_prev"].fillna(0)

    # foia_unit = _compute_unit_level_foia_counts(con, relabel_df)
    # foia_curr = foia_unit[foia_unit["calendar_year"] == foia_unit["relabel_year"]][
    #     ["unitid", "relabel_year", "relabel_type", "total_grads"]
    # ].rename(columns={"relabel_year": "year", "total_grads": "foia_total_curr"})
    # foia_prev = foia_unit[foia_unit["calendar_year"] == foia_unit["relabel_year"] - 1][
    #     ["unitid", "relabel_year", "relabel_type", "total_grads"]
    # ].rename(columns={"relabel_year": "year", "total_grads": "foia_total_prev"})

    # merged_curr = relabel_events.merge(
    #     foia_curr, on=["unitid", "year", "relabel_type"], how="left", suffixes=("", "_foia")
    # )
    # merged_prev = relabel_events.merge(
    #     foia_prev, on=["unitid", "year", "relabel_type"], how="left", suffixes=("", "_foia")
    # )

    # plt.figure(figsize=(7, 5))
    # sns.scatterplot(
    #     data=merged_curr,
    #     x="ipeds_current_intl",
    #     y="foia_total_curr",
    #     hue="relabel_type",
    #     style="relabel_type",
    # )
    # plt.xlabel("IPEDS intl (current, source+target)")
    # plt.ylabel("FOIA total grads (relabel year)")
    # plt.title("Sanity check: current year (unitid x relabel_type)")
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7, 5))
    # sns.scatterplot(
    #     data=merged_prev,
    #     x="ipeds_prev_intl",
    #     y="foia_total_prev",
    #     hue="relabel_type",
    #     style="relabel_type",
    # )
    # plt.xlabel("IPEDS intl (prev year, source+target)")
    # plt.ylabel("FOIA total grads (relabel year - 1)")
    # plt.title("Sanity check: previous year (unitid x relabel_type)")
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
