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
FOIA_PATH = f"{root}/data/int/foia_sevp_combined_raw.parquet"
F1_INST_CW_PATH = f"{INT_FOLDER}/f1_inst_unitid_crosswalk.parquet"
FIG_DIR = f"{root}/figures"

# Candidate columns for FOIA inputs and crosswalks; adjust if schema differs.
FOIA_INST_COLS = ["school_name","f1_inst_row_num", "inst_row_num", "school_id", "f1_inst_id"]
FOIA_CIP_COLS = ["major_1_cip_code","program_cip_code", "cipcode", "cip_code", "cip"]
FOIA_ENDDATE_COLS = ["program_end_date", "program_end_dt", "opt_authorization_end_date"]
FOIA_COUNTRY_COLS = ["country", "citizenship_country", "citizenship_country_desc", "country_of_citizenship", "country_of_citizenship_desc"]
FOIA_YEAR_COLS = ["year", "reporting_year", "data_year"]
FOIA_TUITION_COLS = ["tuition__fees", "tuition_fees", "tuition", "tuition_fees_usd"]
CW_INST_COLS = ["school_name","f1_inst_row_num", "inst_row_num", "school_id", "f1_inst_id"]
CW_UNITID_COLS = ["unitid", "UNITID", "ipeds_unitid"]
DEFAULT_TOP_FOIA_EXCESS_N = 100
DEFAULT_FOIA_GROWTH_MODE = "flow"  # flow: program end year; stock: reported year
DEFAULT_FOIA_GROWTH_NORMALIZE = "absolute"  # or "relative"
DEFAULT_FOIA_GROWTH_BASE_YEAR = 2010
DEFAULT_FOIA_GROUP_COL = "c21basic_lab"


def first_present(cols: Iterable[str], candidates: Iterable[str], label: str) -> str:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise ValueError(f"Could not find {label}. Available columns: {sorted(cols)}")


def normalize_cip_sql(colname: str) -> str:
    # Strip dots, keep digits, cast to integer for joins across sources.
    return f"TRY_CAST(REGEXP_REPLACE(CAST({colname} AS VARCHAR), '[^0-9]', '', 'g') AS INTEGER)"


def prompt_user_options():
    """Prompt user for filtering/plotting options instead of relying on env vars."""
    def _ask_int(prompt: str, default: int) -> int:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"Invalid integer '{raw}', using default {default}")
            return default

    def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
        raw = input(prompt).strip().lower()
        if raw == "":
            return default
        if raw not in choices:
            print(f"Invalid choice '{raw}', using default {default}")
            return default
        return raw

    top_n = _ask_int(
        f"How many top FOIA-excess institution x awlevel combos to drop? (default {DEFAULT_TOP_FOIA_EXCESS_N}): ",
        DEFAULT_TOP_FOIA_EXCESS_N,
    )
    growth_mode = _ask_choice(
        f"FOIA growth mode? ['flow' (program end year) / 'stock' (reported year)] (default {DEFAULT_FOIA_GROWTH_MODE}): ",
        ["flow", "stock"],
        DEFAULT_FOIA_GROWTH_MODE,
    )
    normalize = _ask_choice(
        f"Plot scale? ['absolute' counts / 'relative' index to base year] (default {DEFAULT_FOIA_GROWTH_NORMALIZE}): ",
        ["absolute", "relative"],
        DEFAULT_FOIA_GROWTH_NORMALIZE,
    )
    base_year = _ask_int(
        f"Base year for relative index? (default {DEFAULT_FOIA_GROWTH_BASE_YEAR}): ",
        DEFAULT_FOIA_GROWTH_BASE_YEAR,
    )
    group_col_raw = input(
        f"IPEDS group column to use for FOIA-only grouping (default {DEFAULT_FOIA_GROUP_COL}, enter 'none' to skip): "
    ).strip()
    if group_col_raw.lower() == "none":
        group_col = None
    elif group_col_raw == "":
        group_col = DEFAULT_FOIA_GROUP_COL
    else:
        group_col = group_col_raw
    print(f"Using IPEDS group column: {group_col or 'None'}")

    use_flow = growth_mode != "stock"
    relative = normalize == "relative"
    print(
        f"Using top_n={top_n}, growth_mode={growth_mode}, relative={relative}, base_year={base_year}, group_col={group_col or 'None'}"
    )
    return top_n, use_flow, relative, base_year, group_col

def _merge_counts(con, cip2dig = False):
    print("Loading and processing IPEDS and FOIA data...")
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
    foia_tuition_col = first_present(foia_cols, FOIA_TUITION_COLS, "FOIA tuition/fees column")
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
            awlevel_group,
            c21basic_lab,
            SUM(cnralt) AS intl_students,
            SUM(ctotalt) AS total_students,
            COUNT(*) AS num_programs
        FROM ipeds_raw
        WHERE unitid IS NOT NULL AND cipcode >= 10000 AND cnralt > 0 AND awlevel_group IN ('Bachelor', 'Master', 'Doctor') AND year::INT <= 2022 AND year::INT >= 2010
        GROUP BY unitid, cipcode, awlevel_group, year, c21basic_lab
        """
    ).df()
    print("Loaded IPEDS data")
    ipeds["cip2"] = (ipeds["cipcode"] // 10000).astype("Int64")
    c21_map = ipeds[["unitid", "c21basic_lab"]].dropna().drop_duplicates("unitid")

    foia = con.sql(
        f"""
        SELECT
            unitid,
            ipeds_instname_clean,
            cipcode,
            awlevel_group,
            year,
            COUNT(DISTINCT student_key) AS intl_students,
            AVG(tuition) AS avg_tuition
        FROM (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                cw.ipeds_instname_clean,
                student_key,
                {normalize_cip_sql(foia_cip_col)} AS cipcode,
                CASE WHEN fr.student_edu_level_desc = 'DOCTORATE' THEN 'Doctor' WHEN fr.student_edu_level_desc = 'MASTER''S' THEN 'Master' WHEN fr.student_edu_level_desc = 'BACHELOR''S' THEN 'Bachelor' ELSE 'Other' END AS awlevel_group,
                EXTRACT(YEAR FROM fr.{foia_end_col})::INT AS year,
                TRY_CAST({foia_tuition_col} AS DOUBLE) AS tuition
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE fr.{foia_end_col} IS NOT NULL AND EXTRACT(YEAR FROM fr.{foia_end_col})::INT = year::INT AND year::INT <= 2022 AND year::INT >= 2010 AND fr.student_edu_level_desc IN ('BACHELOR''S', 'MASTER''S', 'DOCTORATE')
        )
        WHERE unitid IS NOT NULL AND cipcode IS NOT NULL AND year IS NOT NULL AND awlevel_group IN ('Bachelor', 'Master', 'Doctor')
        GROUP BY unitid, cipcode, awlevel_group, year, ipeds_instname_clean
        """
    ).df()
    print("Loaded FOIA data")
    foia["cip2"] = (foia["cipcode"] // 10000).astype("Int64")
    foia = foia.merge(c21_map, on="unitid", how="left")


    if cip2dig:
        # aggregate to 2-digit CIP
        ipeds = (
            ipeds.groupby(["unitid", "cip2", "awlevel_group", "year", "c21basic_lab"], as_index=False)
            .agg(
                intl_students=("intl_students", "sum"),
                total_students=("total_students", "sum"),
                num_programs=("num_programs", "sum"),
            )
        )

        foia = (
            foia.groupby(["unitid", "cip2", "awlevel_group", "year", "ipeds_instname_clean", "c21basic_lab"], as_index=False)
            .agg(intl_students=("intl_students", "sum"))
        )

        on = ["unitid", "cip2", "awlevel_group", "year"]
    else:
        on = ["unitid", "cipcode", "awlevel_group", "year"]

    combined = ipeds.merge(
        foia,
        on=on,
        suffixes=("_ipeds", "_foia"),
        how="outer",
    )
    print("Merged IPEDS and FOIA data")
    combined['intl_students_ipeds'] = combined['intl_students_ipeds'].fillna(0)
    combined['intl_students_foia'] = combined['intl_students_foia'].fillna(0)

    print(
        "IPEDS rows:",
        ipeds.shape[0],
    )
    print(
        "FOIA rows:",
        foia.shape[0],
    )
    print(
        "Merged rows:",
        combined.shape[0],
    )
    return ipeds, foia, combined


def exclude_top_excess(
    combined: pd.DataFrame,
    top_n: int = 0,
) -> pd.DataFrame:
    """
    Aggregate FOIA excess by institution x award level and optionally exclude top-N.
    """
    if top_n <= 0:
        return combined

    agg = (
        combined.groupby(["unitid", "awlevel_group"], as_index=False)
        .agg(
            intl_students_ipeds=("intl_students_ipeds", "sum"),
            intl_students_foia=("intl_students_foia", "sum"),
            ipeds_instname_clean=("ipeds_instname_clean", "first"),
        )
    )
    agg["foia_excess_total"] = (
        (agg["intl_students_foia"] - agg["intl_students_ipeds"])
        #/ agg["intl_students_ipeds"].replace({0: np.nan})
    )
    agg = agg.sort_values("foia_excess_total", ascending=False)

    top_list = agg.head(top_n)
    pd.set_option("display.max_rows", max(10, top_n))
    print(f"Top {top_n} institution x awlevel FOIA excess combos (printed, not saved):")
    print(top_list)

    top_keys = set(
        top_list[["unitid", "awlevel_group"]].itertuples(index=False, name=None)
    )
    filtered = combined[~combined[["unitid", "awlevel_group"]].apply(tuple, axis=1).isin(top_keys)]
    print(f"Excluded {len(top_keys)} high-excess institution x awlevel combos; {filtered.shape[0]} rows remain")
    return filtered


def _apply_relative_index(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str = "intl_students",
    base_year: int = 2010,
) -> pd.DataFrame:
    """
    Convert counts to an index relative to base_year within each group.
    """
    base = (
        df[df["year"] == base_year][group_cols + [value_col]].rename(columns={value_col: "_base"})
    )
    merged = df.merge(base, on=group_cols, how="left")
    merged[value_col] = merged[value_col] / merged["_base"]
    missing = merged["_base"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows missing base year {base_year} for relative index (values set to NaN)")
    return merged.drop(columns="_base")


def plot_ipeds_foia_scatter(combined: pd.DataFrame) -> None:
    combined_plot = combined.copy()[(combined['intl_students_foia'] < 200)&(combined['intl_students_ipeds'] < 200)&(combined['intl_students_foia'] >=0)&(combined['intl_students_ipeds'] >=0)]
    print(f"Dropped {combined.shape[0] - combined_plot.shape[0]} rows ({(combined.shape[0] - combined_plot.shape[0]) / combined.shape[0] * 100:.2f}%) with extreme counts for plotting")
    ax = sns.regplot(
        data=combined_plot,
        x="intl_students_ipeds",
        y="intl_students_foia",
        fit_reg = False,
        line_kws={"color": "red", "lw": 2, "label": "OLS fit"},
        x_bins = 100
    )
    max_lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_lim], [0, max_lim], color="gray", linestyle="--", lw=1, label="45° line")
    ax.set_xlim(0, max_lim)
    ax.set_ylim(0, max_lim)
    plt.xlabel("Intl students (IPEDS)")
    plt.ylabel("Intl students (FOIA)")
    plt.legend()
    plt.title("Institution x CIP-level intl student count: scatter w/ OLS line")
    plt.tight_layout()
    print(plt)


def plot_intl_totals_by_awlevel(ipeds: pd.DataFrame, foia: pd.DataFrame) -> None:
    ipeds_tot = (
        ipeds.groupby(["year", "awlevel_group"], as_index=False)["intl_students"].sum()
        .assign(source="IPEDS")
    )
    foia_tot = (
        foia.groupby(["year", "awlevel_group"], as_index=False)["intl_students"].sum()
        .assign(source="FOIA")
    )
    totals = pd.concat([ipeds_tot, foia_tot], ignore_index=True)
    g = sns.relplot(
        data=totals,
        kind="line",
        x="year",
        y="intl_students",
        hue="source",
        col="awlevel_group",
        col_wrap=2,
        facet_kws={"sharey": False},
        height=4,
        aspect=1.2,
    )
    g.set_axis_labels("Year", "International students")
    g.figure.suptitle("International students over time by degree level", y=1.02)
    plt.tight_layout()


def plot_intl_totals_by_c21basic(ipeds: pd.DataFrame, foia: pd.DataFrame) -> None:
    ipeds_tot = (
        ipeds.dropna(subset=["c21basic_lab"])
        .groupby(["year", "c21basic_lab"], as_index=False)["intl_students"]
        .sum()
        .assign(source="IPEDS")
    )
    foia_tot = (
        foia.dropna(subset=["c21basic_lab"])
        .groupby(["year", "c21basic_lab"], as_index=False)["intl_students"]
        .sum()
        .assign(source="FOIA")
    )
    totals = pd.concat([ipeds_tot, foia_tot], ignore_index=True)
    if totals.empty:
        print("No data available for c21basic_lab plot.")
        return
    top_cats = (
        totals.groupby("c21basic_lab")["intl_students"].sum().sort_values(ascending=False).head(8).index.tolist()
    )
    totals = totals[totals["c21basic_lab"].isin(top_cats)]
    g = sns.relplot(
        data=totals,
        kind="line",
        x="year",
        y="intl_students",
        hue="source",
        col="c21basic_lab",
        col_wrap=3,
        facet_kws={"sharey": False},
        height=4,
        aspect=1.1,
    )
    g.set_axis_labels("Year", "International students")
    g.figure.suptitle("International students over time by Carnegie (c21basic_lab)", y=1.02)
    plt.tight_layout()


def plot_foia_tuition_by_stemopt(foia: pd.DataFrame, con: ddb.DuckDBPyConnection) -> None:
    """Plot average FOIA tuition over time by degree type and STEM OPT eligibility."""

    # Build STEM OPT lookup from IPEDS
    stem_map = con.sql(
        """
        SELECT CAST(cipcode AS INTEGER) AS cipcode, MAX(CAST(STEMOPT AS INTEGER)) AS stemopt
        FROM ipeds_raw
        WHERE cipcode IS NOT NULL
        GROUP BY cipcode
        """
    ).df()
    stem_full = set(stem_map.loc[stem_map["stemopt"] == 1, "cipcode"].tolist())
    stem_two = set(int(x) // 10000 for x in stem_full if pd.notna(x))

    work = foia.copy()
    # Resolve CIP column (cipcode or cip2)
    cip_col = None
    for cand in ("cipcode", "cip2"):
        if cand in work.columns:
            cip_col = cand
            break
    if cip_col is None:
        print("Skipping tuition STEM plot: no CIP column found in FOIA data.")
        return

    if cip_col == "cip2":
        work["stemopt"] = work[cip_col].astype("Int64").isin(stem_two)
    else:
        work["stemopt"] = work[cip_col].astype("Int64").isin(stem_full)

    if "avg_tuition" not in work.columns:
        print("Skipping tuition STEM plot: avg_tuition column missing.")
        return

    work = work.dropna(subset=["avg_tuition"])
    if work.empty:
        print("No FOIA tuition data available for plotting.")
        return

    tuition = (
        work.groupby(["year", "awlevel_group", "stemopt"], as_index=False)["avg_tuition"]
        .mean()
        .rename(columns={"avg_tuition": "mean_tuition"})
    )
    g = sns.relplot(
        data=tuition,
        kind="line",
        x="year",
        y="mean_tuition",
        hue="stemopt",
        col="awlevel_group",
        col_wrap=2,
        facet_kws={"sharey": False},
        height=4,
        aspect=1.2,
    )
    g.set_axis_labels("Year", "Average tuition/fees")
    g.figure.suptitle("FOIA average tuition by degree level and STEM OPT eligibility", y=1.02)
    plt.tight_layout()


def _prepare_foia_growth(con: ddb.DuckDBPyConnection, use_flow: bool = True, group_col: str | None = DEFAULT_FOIA_GROUP_COL) -> pd.DataFrame:
    foia_cols = [row[0] for row in con.sql("DESCRIBE foia_raw").fetchall()]
    cw_cols = [row[0] for row in con.sql("DESCRIBE f1_inst_cw").fetchall()]
    ipeds_cols = [row[0] for row in con.sql("DESCRIBE ipeds_raw").fetchall()]
    foia_cip_col = first_present(foia_cols, FOIA_CIP_COLS, "FOIA CIP column")
    foia_end_col = first_present(foia_cols, FOIA_ENDDATE_COLS, "FOIA program end date column")
    foia_country_col = first_present(foia_cols, FOIA_COUNTRY_COLS, "FOIA country column")
    foia_inst_col = first_present(foia_cols, FOIA_INST_COLS, "FOIA institution column")
    cw_inst_col = first_present(cw_cols, CW_INST_COLS, "crosswalk institution column")
    cw_unitid_col = first_present(cw_cols, CW_UNITID_COLS, "crosswalk unitid column")
    foia_year_col = first_present(foia_cols, FOIA_YEAR_COLS, "FOIA year column") if not use_flow else None

    if group_col and group_col not in ipeds_cols:
        print(f"Group column {group_col} not found in IPEDS data; skipping group assignment.")
        group_col = None

    # Coverage check for group_col on FOIA-linked unitids
    join_map = ""
    group_select = ""
    group_groupby = ""
    select_group = ""
    coverage = None
    if group_col:
        join_map = f"""
        LEFT JOIN (
            SELECT DISTINCT unitid, {group_col} AS group_col
            FROM ipeds_raw
            WHERE {group_col} IS NOT NULL
        ) map
          ON cw.{cw_unitid_col} = map.unitid
        """
        group_select = ", map.group_col AS group_col"
        group_groupby = ", group_col"
        select_group = ", group_col"
        coverage = con.sql(
            f"""
            WITH foia_unitids AS (
                SELECT DISTINCT cw.{cw_unitid_col} AS unitid
                FROM foia_raw fr
                LEFT JOIN f1_inst_cw cw
                  ON fr.{foia_inst_col} = cw.{cw_inst_col}
                WHERE cw.{cw_unitid_col} IS NOT NULL
            )
            SELECT
                (SELECT COUNT(DISTINCT unitid) FROM foia_unitids) AS total_unitids,
                COUNT(DISTINCT m.unitid) AS unitids_with_group,
                (SELECT COUNT(DISTINCT unitid) FROM foia_unitids) - COUNT(DISTINCT m.unitid) AS unitids_missing_group
            FROM foia_unitids fu
            LEFT JOIN (
                SELECT DISTINCT unitid, {group_col} AS group_col
                FROM ipeds_raw
                WHERE {group_col} IS NOT NULL
            ) m
              ON fu.unitid = m.unitid
            """
        ).fetchone()
        if coverage:
            total_unitids, unitids_with_group, unitids_missing_group = coverage
            print(
                f"FOIA unitid {group_col} coverage: total={total_unitids}, with {group_col}={unitids_with_group}, missing={unitids_missing_group}"
            )

    if use_flow:
        growth_sql = f"""
            WITH foia_clean AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                CASE
                    WHEN fr.student_edu_level_desc = 'DOCTORATE' THEN 'Doctor'
                    WHEN fr.student_edu_level_desc = 'MASTER''S' THEN 'Master'
                    WHEN fr.student_edu_level_desc = 'BACHELOR''S' THEN 'Bachelor'
                    ELSE 'Other'
                END AS awlevel_group,
                EXTRACT(YEAR FROM fr.{foia_end_col})::INT AS year,
                {normalize_cip_sql(foia_cip_col)} AS cipcode,
                TRIM(UPPER(CAST(fr.{foia_country_col} AS VARCHAR))) AS country,
                student_key
                {group_select}
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            {join_map}
            WHERE fr.{foia_end_col} IS NOT NULL
              AND EXTRACT(YEAR FROM fr.{foia_end_col})::INT BETWEEN 2010 AND 2022
              AND fr.student_edu_level_desc IN ('BACHELOR''S', 'MASTER''S', 'DOCTORATE')
            )
            SELECT
                awlevel_group,
                year,
                cipcode,
                country{select_group},
                COUNT(DISTINCT student_key) AS intl_students
            FROM foia_clean
            GROUP BY awlevel_group, year, cipcode, country {group_groupby}
            HAVING cipcode IS NOT NULL AND country IS NOT NULL
        """
    else:
        growth_sql = f"""
            WITH foia_clean AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                CASE
                    WHEN fr.student_edu_level_desc = 'DOCTORATE' THEN 'Doctor'
                    WHEN fr.student_edu_level_desc = 'MASTER''S' THEN 'Master'
                    WHEN fr.student_edu_level_desc = 'BACHELOR''S' THEN 'Bachelor'
                    ELSE 'Other'
                END AS awlevel_group,
                {foia_year_col}::INT AS year,
                {normalize_cip_sql(foia_cip_col)} AS cipcode,
                TRIM(UPPER(CAST(fr.{foia_country_col} AS VARCHAR))) AS country,
                student_key
                {group_select}
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            {join_map}
            WHERE {foia_year_col} IS NOT NULL
              AND {foia_year_col}::INT BETWEEN 2010 AND 2022
              AND fr.student_edu_level_desc IN ('BACHELOR''S', 'MASTER''S', 'DOCTORATE')
            )
            SELECT
                awlevel_group,
                year,
                cipcode,
                country{select_group},
                COUNT(DISTINCT student_key) AS intl_students
            FROM foia_clean
            GROUP BY awlevel_group, year, cipcode, country {group_groupby}
            HAVING cipcode IS NOT NULL AND country IS NOT NULL
        """

    growth = con.sql(growth_sql).df()
    if group_col and "group_col" in growth.columns:
        growth = growth.rename(columns={"group_col": group_col})
    growth["cip2"] = (growth["cipcode"] // 10000).astype("Int64")
    return growth


def plot_foia_growth_by_country_and_cip(
    con: ddb.DuckDBPyConnection,
    use_flow: bool = True,
    relative: bool = False,
    base_year: int = 2010,
    group_col: str | None = DEFAULT_FOIA_GROUP_COL,
) -> None:
    growth = _prepare_foia_growth(con, use_flow=use_flow, group_col=group_col)
    mode_desc = "flow (program end year)" if use_flow else "stock (reported year)"
    print(f"FOIA growth mode: {mode_desc}; scale: {'relative' if relative else 'absolute'}")
    y_label = "International students (FOIA)" if not relative else f"Index vs {base_year} (base=1.0)"
    # 1) By award level (aggregate over country/CIP)
    by_awlevel = growth.groupby(["year", "awlevel_group"], as_index=False)["intl_students"].sum()
    if relative:
        by_awlevel = _apply_relative_index(by_awlevel, ["awlevel_group"], "intl_students", base_year)
    g1 = sns.relplot(
        data=by_awlevel,
        kind="line",
        x="year",
        y="intl_students",
        hue="awlevel_group",
        height=4,
        aspect=1.4,
    )
    g1.set_axis_labels("Year", y_label)
    g1.figure.suptitle(
        "FOIA intl students by award level" + (f" (relative to {base_year})" if relative else ""),
        y=1.02,
    )
    plt.tight_layout()

    # 2) Top 5 countries (+ Other), aggregated over CIP/award level
    country_order = (
        growth.groupby("country")["intl_students"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    growth["country_group"] = np.where(growth["country"].isin(country_order), growth["country"], "Other")
    country_order.append("Other")
    by_country = (
        growth.groupby(["year", "country_group"], as_index=False)["intl_students"].sum()
    )
    if relative:
        by_country = _apply_relative_index(by_country, ["country_group"], "intl_students", base_year)
    by_country["country_group"] = pd.Categorical(by_country["country_group"], categories=country_order, ordered=True)
    g2 = sns.relplot(
        data=by_country,
        kind="line",
        x="year",
        y="intl_students",
        hue="country_group",
        height=4,
        aspect=1.4,
    )
    g2.set_axis_labels("Year", y_label)
    g2.figure.suptitle(
        "FOIA intl students by country (top 5 + Other)" + (f" (relative to {base_year})" if relative else ""),
        y=1.02,
    )
    plt.tight_layout()

    # 3) Top 5 2-digit CIP (+ Other), aggregated over country/award level
    cip_order = (
        growth.groupby("cip2")["intl_students"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    growth["cip2_group"] = np.where(growth["cip2"].isin(cip_order), growth["cip2"].astype("Int64"), pd.NA)
    growth["cip2_group"] = growth["cip2_group"].fillna("Other")
    cip_order = [str(c) for c in cip_order] + ["Other"]

    by_cip = (
        growth.assign(cip2_group=growth["cip2_group"].astype(str))
        .groupby(["year", "cip2_group"], as_index=False)["intl_students"].sum()
    )
    if relative:
        by_cip = _apply_relative_index(by_cip, ["cip2_group"], "intl_students", base_year)
    by_cip["cip2_group"] = pd.Categorical(by_cip["cip2_group"], categories=cip_order, ordered=True)
    g3 = sns.relplot(
        data=by_cip,
        kind="line",
        x="year",
        y="intl_students",
        hue="cip2_group",
        height=4,
        aspect=1.4,
    )
    g3.set_axis_labels("Year", y_label)
    g3.figure.suptitle(
        "FOIA intl students by 2-digit CIP (top 5 + Other)" + (f" (relative to {base_year})" if relative else ""),
        y=1.02,
    )
    plt.tight_layout()

    # 4) Masters only: country and CIP breakdowns
    masters = growth[growth["awlevel_group"] == "Master"].copy()
    if not masters.empty:
        m_country_order = (
            masters.groupby("country")["intl_students"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        masters["country_group"] = np.where(masters["country"].isin(m_country_order), masters["country"], "Other")
        m_country_order.append("Other")
        masters_by_country = (
            masters.groupby(["year", "country_group"], as_index=False)["intl_students"].sum()
        )
        if relative:
            masters_by_country = _apply_relative_index(
                masters_by_country, ["country_group"], "intl_students", base_year
            )
        masters_by_country["country_group"] = pd.Categorical(
            masters_by_country["country_group"], categories=m_country_order, ordered=True
        )
        g4 = sns.relplot(
            data=masters_by_country,
            kind="line",
            x="year",
            y="intl_students",
            hue="country_group",
            height=4,
            aspect=1.4,
        )
        g4.set_axis_labels("Year", y_label)
        g4.figure.suptitle(
            "FOIA masters intl students by country (top 5 + Other)" + (f" (relative to {base_year})" if relative else ""),
            y=1.02,
        )
        plt.tight_layout()

    # 6/7) FOIA totals by user-selected IPEDS group column (overall and Masters-only)
    if group_col and group_col in growth.columns:
        group_totals = (
            growth.dropna(subset=[group_col])
            .groupby(["year", group_col], as_index=False)["intl_students"]
            .sum()
        )
        if relative:
            group_totals = _apply_relative_index(group_totals, [group_col], "intl_students", base_year)
        if group_totals.empty:
            print(f"No data available for {group_col} FOIA plot.")
        else:
            top_groups = (
                group_totals.groupby(group_col)["intl_students"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .index.tolist()
            )
            group_totals[group_col] = np.where(
                group_totals[group_col].isin(top_groups), group_totals[group_col], "Other"
            )
            top_groups = top_groups + ["Other"]
            group_totals[group_col] = pd.Categorical(group_totals[group_col], categories=top_groups, ordered=True)
            g6 = sns.relplot(
                data=group_totals,
                kind="line",
                x="year",
                y="intl_students",
                hue=group_col,
                height=4.5,
                aspect=1.35,
            )
            g6.set_axis_labels("Year", y_label)
            g6.figure.suptitle(
                f"FOIA intl students by {group_col}" + (f" (relative to {base_year})" if relative else ""),
                y=1.02,
            )
            plt.tight_layout()

        masters_group = (
            growth[(growth["awlevel_group"] == "Master") & growth[group_col].notna()]
            .groupby(["year", group_col], as_index=False)["intl_students"]
            .sum()
        )
        if relative:
            masters_group = _apply_relative_index(masters_group, [group_col], "intl_students", base_year)
        if masters_group.empty:
            print(f"No data available for masters-only {group_col} FOIA plot.")
        else:
            top_groups_m = (
                masters_group.groupby(group_col)["intl_students"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .index.tolist()
            )
            masters_group[group_col] = np.where(
                masters_group[group_col].isin(top_groups_m), masters_group[group_col], "Other"
            )
            top_groups_m = top_groups_m + ["Other"]
            masters_group[group_col] = pd.Categorical(masters_group[group_col], categories=top_groups_m, ordered=True)
            g7 = sns.relplot(
                data=masters_group,
                kind="line",
                x="year",
                y="intl_students",
                hue=group_col,
                height=4.5,
                aspect=1.35,
            )
            g7.set_axis_labels("Year", y_label)
            g7.figure.suptitle(
                f"FOIA masters intl students by {group_col}" + (f" (relative to {base_year})" if relative else ""),
                y=1.02,
            )
            plt.tight_layout()
    else:
        print(f"Group column {group_col} not available in FOIA growth data; skipping group plots.")

    m_cip_order = (
        masters.groupby("cip2")["intl_students"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    masters["cip2_group"] = np.where(masters["cip2"].isin(m_cip_order), masters["cip2"].astype("Int64"), pd.NA)
    masters["cip2_group"] = masters["cip2_group"].fillna("Other")
    m_cip_order = [str(c) for c in m_cip_order] + ["Other"]
    masters_by_cip = (
        masters.assign(cip2_group=masters["cip2_group"].astype(str))
        .groupby(["year", "cip2_group"], as_index=False)["intl_students"].sum()
    )
    if relative:
        masters_by_cip = _apply_relative_index(
            masters_by_cip, ["cip2_group"], "intl_students", base_year
        )
    masters_by_cip["cip2_group"] = pd.Categorical(masters_by_cip["cip2_group"], categories=m_cip_order, ordered=True)
    g5 = sns.relplot(
        data=masters_by_cip,
        kind="line",
        x="year",
        y="intl_students",
        hue="cip2_group",
        height=4,
        aspect=1.4,
    )
    g5.set_axis_labels("Year", y_label)
    g5.figure.suptitle(
        "FOIA masters intl students by 2-digit CIP (top 5 + Other)" + (f" (relative to {base_year})" if relative else ""),
        y=1.02,
    )
    plt.tight_layout()

def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    con = ddb.connect()
    #top_n, use_flow, relative, base_year, group_col = prompt_user_options()
    (ipeds, foia, combined) = _merge_counts(con, cip2dig=False)
    #combined = exclude_top_excess(combined, top_n=top_n)
    #plot_ipeds_foia_scatter(combined)
    #plot_intl_totals_by_awlevel(ipeds, foia)
    plot_foia_tuition_by_stemopt(foia, con)
    # plot_foia_growth_by_country_and_cip(
    #     con,
    #     use_flow=use_flow,
    #     relative=relative,
    #     base_year=base_year,
    #     group_col=group_col,
    # )

    # # Optional: save growth datasets
    # combined_growth = combined.copy()
    # out_growth = os.path.join(root, "data", "int", "intl_student_growth_ipeds_vs_foia.parquet")
    # combined_growth.to_parquet(out_growth, index=False)
    # print(f"Wrote merged growth data to {out_growth}")



if __name__ == "__main__":
    main()
