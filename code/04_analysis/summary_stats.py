"""summary_stats.py — Summary statistics tables and figures for the data section.

Produces two paired exhibits:
  Panel A: H-1B application summary stats (computed from FOIA data only, no Revelio contamination)
  Panel B: Revelio user summary stats

Each panel is output as:
  - A LaTeX table (.tex) using booktabs + threeparttable
  - A dot-plot figure (.pdf): one subplot per statistic, samples on x-axis
  - A grouped bar-chart figure (.pdf): percentage stats only

Output directory: code/output/summary_stats/
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import yaml

try:
    from IPython import get_ipython as _get_ipython
    _IN_IPYTHON = _get_ipython() is not None
except Exception:
    _IN_IPYTHON = False
if not _IN_IPYTHON:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
if "__file__" in globals():
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    _THIS_DIR = os.path.join(os.getcwd(), "04_analysis")

_CODE_DIR = os.path.dirname(_THIS_DIR)
sys.path.append(_CODE_DIR)

from config import root, code  # noqa: E402

_CFG_PATH = Path(_CODE_DIR) / "configs" / "summary_stats.yaml"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(_CFG_PATH) as f:
        raw = yaml.safe_load(f) or {}

    run_tag = raw.get("run_tag", "feb2026")
    data_root = Path(root)
    code_dir = Path(code)

    cfg: dict = {
        "run_tag": run_tag,
        "foia_raw_csv": str(
            data_root / "data" / "raw" / "foia_bloomberg" / "foia_bloomberg_all_withids.csv"
        ),
        "output_dir": code_dir / "output" / "summary_stats",
        "paths": {
            "foia_indiv": data_root / "data" / "clean" / f"foia_indiv_{run_tag}.parquet",
            "foia_raw_match": data_root / "data" / "int" / f"foia_raw_match_{run_tag}.parquet",
            "rev_indiv":  data_root / "data" / "clean" / f"rev_indiv_{run_tag}.parquet",
            "merge_baseline":       data_root / "data" / "int" / f"merge_filt_baseline_{run_tag}.parquet",
            "merge_mult4":          data_root / "data" / "int" / f"merge_filt_mult4_{run_tag}.parquet",
            "merge_prefilt":        data_root / "data" / "int" / f"merge_filt_prefilt_{run_tag}.parquet",
            "merge_strict":         data_root / "data" / "int" / f"merge_filt_strict_{run_tag}.parquet",
            "merge_us_educ":        data_root / "data" / "int" / f"merge_filt_us_educ_{run_tag}.parquet",
            "merge_us_educ_prefilt":data_root / "data" / "int" / f"merge_filt_us_educ_prefilt_{run_tag}.parquet",
            "merge_us_educ_opt":    data_root / "data" / "int" / f"merge_filt_us_educ_opt_{run_tag}.parquet",
            "merge_strict_med":     data_root / "data" / "int" / f"merge_filt_strict_med_{run_tag}.parquet",
        },
    }

    # Allow YAML overrides for paths
    def _resolve(s: str) -> str:
        return s.replace("{root}", str(data_root)).replace("{code}", str(code_dir))

    if "foia_raw_csv" in raw:
        cfg["foia_raw_csv"] = _resolve(raw["foia_raw_csv"])
    if "output_dir" in raw:
        cfg["output_dir"] = Path(_resolve(raw["output_dir"]))

    return cfg


# ---------------------------------------------------------------------------
# H1B stats
# ---------------------------------------------------------------------------

def _clean_firm_uid_sql(col: str = "foia_firm_uid") -> str:
    """Normalize foia_firm_uid for firm-year counting."""
    return (
        "CASE "
        f"WHEN TRIM(COALESCE(CAST({col} AS VARCHAR), '')) IN ('', '0', 'None', 'NULL', 'nan', 'NaN') "
        "THEN NULL "
        f"ELSE TRIM(CAST({col} AS VARCHAR)) "
        "END"
    )

def compute_h1b_stats_raw(con: duckdb.DuckDBPyConnection, csv_path: str) -> dict:
    """Compute H1B summary stats from the raw Bloomberg CSV (~1.8M rows)."""
    sql = f"""
    WITH raw AS (
        SELECT
            CASE WHEN LOWER(TRIM(gender)) = 'female' THEN 1.0 ELSE 0.0 END AS female_ind,
            TRY_CAST(ben_year_of_birth AS INTEGER)                          AS yob,
            TRY_CAST(lottery_year AS INTEGER)                               AS lottery_year,
            LOWER(TRIM(country_of_nationality))                             AS country,
            COALESCE(
                NULLIF(
                    REGEXP_REPLACE(
                        REGEXP_REPLACE(COALESCE(CAST(FEIN AS VARCHAR), ''), '[^0-9]', '', 'g'),
                        '^0+', '', 'g'
                    ), ''
                ), '0'
            ) AS fein_clean,
            CASE WHEN status_type = 'SELECTED' THEN 1.0 ELSE 0.0 END      AS winner_ind,
            CASE WHEN status_type = 'SELECTED' AND TRIM(COALESCE(S3Q1,'')) = 'M'  THEN 1.0
                 WHEN status_type = 'SELECTED' AND TRIM(COALESCE(S3Q1,'')) = 'NA' THEN NULL
                 WHEN status_type = 'SELECTED'                                    THEN 0.0
                 ELSE NULL END                                             AS ade_ind
        FROM read_csv_auto('{csv_path}', ignore_errors=true, header=true)
        WHERE TRY_CAST(lottery_year AS INTEGER) IS NOT NULL
    ),
    ind AS (
        SELECT
            female_ind,
            CASE WHEN country = 'ind' THEN 1.0 ELSE 0.0 END AS india_ind,
            CASE WHEN country = 'chn' THEN 1.0 ELSE 0.0 END AS china_ind,
            CASE WHEN yob > 1900 AND (lottery_year - 1 - yob) BETWEEN 15 AND 80
                 THEN (lottery_year - 1 - yob) END                AS age,
            fein_clean,
            lottery_year,
            winner_ind,
            ade_ind
        FROM raw
    )
    SELECT
        COUNT(*)                                                                       AS n,
        AVG(female_ind)                                                                AS share_female,
        SQRT(AVG(female_ind) * (1 - AVG(female_ind)) / NULLIF(COUNT(*), 0))           AS share_female_se,
        AVG(age)                                                                       AS avg_age_at_app,
        STDDEV_SAMP(age) / SQRT(NULLIF(COUNT(age), 0))                                AS avg_age_at_app_se,
        AVG(india_ind)                                                                 AS share_india,
        SQRT(AVG(india_ind) * (1 - AVG(india_ind)) / NULLIF(COUNT(*), 0))             AS share_india_se,
        AVG(china_ind)                                                                 AS share_china,
        SQRT(AVG(china_ind) * (1 - AVG(china_ind)) / NULLIF(COUNT(*), 0))             AS share_china_se,
        COUNT(*) * 1.0
            / COUNT(DISTINCT fein_clean || '_' || CAST(lottery_year AS VARCHAR))       AS avg_apps_per_firm_year,
        AVG(winner_ind)                                                                AS share_winner,
        SQRT(AVG(winner_ind) * (1 - AVG(winner_ind)) / NULLIF(COUNT(*), 0))           AS share_winner_se,
        AVG(ade_ind)                                                                   AS share_ade,
        SQRT(AVG(ade_ind) * (1 - AVG(ade_ind)) / NULLIF(COUNT(ade_ind), 0))           AS share_ade_se
    FROM ind
    """
    return con.execute(sql).fetchdf().iloc[0].to_dict()


def compute_h1b_stats_parquet(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
    is_merged: bool = False,
) -> dict | None:
    """Compute H1B summary stats from a cleaned parquet.

    For merged parquets (is_merged=True) there are multiple rows per foia_indiv_id
    (one per Revelio candidate). Dedup on foia_indiv_id before computing stats.
    """
    if not Path(parquet_path).exists():
        warnings.warn(f"Parquet not found, skipping: {parquet_path}")
        return None

    if is_merged:
        # Merged parquets use foia_country (prefixed to avoid collision with rev_country)
        base_sql = f"""
        WITH raw AS (
            SELECT
                foia_indiv_id,
                ANY_VALUE(TRY_CAST(female_ind AS DOUBLE))                             AS female_ind,
                ANY_VALUE(TRY_CAST(yob AS INTEGER))                                   AS yob,
                ANY_VALUE(TRY_CAST(lottery_year AS INTEGER))                          AS lottery_year,
                ANY_VALUE(foia_country)                                               AS country,
                ANY_VALUE(foia_firm_uid)                                              AS foia_firm_uid,
                ANY_VALUE(CASE WHEN status_type = 'SELECTED' THEN 1.0 ELSE 0.0 END)  AS winner_ind,
                ANY_VALUE(TRY_CAST(ade_ind AS DOUBLE))                                AS ade_ind
            FROM read_parquet('{parquet_path}')
            GROUP BY foia_indiv_id
        )
        """
    else:
        base_sql = f"""
        WITH raw AS (
            SELECT
                TRY_CAST(female_ind AS DOUBLE)                                        AS female_ind,
                TRY_CAST(yob AS INTEGER)                                              AS yob,
                TRY_CAST(lottery_year AS INTEGER)                                     AS lottery_year,
                country,
                foia_firm_uid,
                CASE WHEN status_type = 'SELECTED' THEN 1.0 ELSE 0.0 END             AS winner_ind,
                TRY_CAST(ade_lottery AS DOUBLE)                                       AS ade_ind
            FROM read_parquet('{parquet_path}')
        )
        """

    sql = base_sql + """
    , ind AS (
        SELECT
            female_ind,
            CASE WHEN LOWER(country) = 'india' THEN 1.0 ELSE 0.0 END AS india_ind,
            CASE WHEN LOWER(country) = 'china' THEN 1.0 ELSE 0.0 END AS china_ind,
            CASE WHEN yob > 1900 AND (lottery_year - 1 - yob) BETWEEN 15 AND 80
                 THEN (lottery_year - 1 - yob) END                    AS age,
            foia_firm_uid,
            lottery_year,
            winner_ind,
            ade_ind
        FROM raw
    )
    SELECT
        COUNT(*)                                                                       AS n,
        AVG(female_ind)                                                                AS share_female,
        SQRT(AVG(female_ind) * (1 - AVG(female_ind)) / NULLIF(COUNT(*), 0))           AS share_female_se,
        AVG(age)                                                                       AS avg_age_at_app,
        STDDEV_SAMP(age) / SQRT(NULLIF(COUNT(age), 0))                                AS avg_age_at_app_se,
        AVG(india_ind)                                                                 AS share_india,
        SQRT(AVG(india_ind) * (1 - AVG(india_ind)) / NULLIF(COUNT(*), 0))             AS share_india_se,
        AVG(china_ind)                                                                 AS share_china,
        SQRT(AVG(china_ind) * (1 - AVG(china_ind)) / NULLIF(COUNT(*), 0))             AS share_china_se,
        COUNT(*) * 1.0
            / COUNT(DISTINCT CAST(foia_firm_uid AS VARCHAR)
                             || '_' || CAST(lottery_year AS VARCHAR))                  AS avg_apps_per_firm_year,
        AVG(winner_ind)                                                                AS share_winner,
        SQRT(AVG(winner_ind) * (1 - AVG(winner_ind)) / NULLIF(COUNT(*), 0))           AS share_winner_se,
        AVG(ade_ind)                                                                   AS share_ade,
        SQRT(AVG(ade_ind) * (1 - AVG(ade_ind)) / NULLIF(COUNT(ade_ind), 0))           AS share_ade_se
    FROM ind
    """
    return con.execute(sql).fetchdf().iloc[0].to_dict()


def _empty_h1b_winlose_bucket() -> dict[str, float | int | None]:
    return {
        "share_ade": None,
        "share_female": None,
        "avg_age_at_app": None,
        "share_india": None,
        "share_china": None,
        "n_apps": None,
        "n_firm_years": None,
    }


def _compute_h1b_winlose_stats(
    con: duckdb.DuckDBPyConnection,
    base_sql: str,
) -> dict:
    """Aggregate H-1B stats by winner/loser, plus overall winner share."""
    sql = f"""
    WITH base AS (
        {base_sql}
    ),
    prep AS (
        SELECT
            winner,
            CASE WHEN winner = 1 THEN ade ELSE NULL END                            AS ade,
            female_ind,
            CASE WHEN LOWER(TRIM(country)) IN ('ind', 'india') THEN 1.0 ELSE 0.0 END AS india_ind,
            CASE WHEN LOWER(TRIM(country)) IN ('chn', 'china') THEN 1.0 ELSE 0.0 END AS china_ind,
            CASE WHEN yob > 1900 AND (lottery_year - 1 - yob) BETWEEN 15 AND 80
                 THEN (lottery_year - 1 - yob) END                                AS age,
            {_clean_firm_uid_sql('foia_firm_uid')}                                AS firm_uid_clean,
            lottery_year
        FROM base
    ),
    overall AS (
        SELECT
            AVG(winner) AS share_winner,
            COUNT(*) AS n_apps_total,
            COUNT(DISTINCT CASE
                WHEN firm_uid_clean IS NOT NULL AND lottery_year IS NOT NULL
                THEN firm_uid_clean || '_' || CAST(lottery_year AS VARCHAR)
            END) AS n_firm_years_total
        FROM prep
    ),
    grouped AS (
        SELECT
            winner,
            COUNT(*) AS n_apps,
            AVG(ade) AS share_ade,
            AVG(female_ind) AS share_female,
            AVG(age) AS avg_age_at_app,
            AVG(india_ind) AS share_india,
            AVG(china_ind) AS share_china,
            COUNT(DISTINCT CASE
                WHEN firm_uid_clean IS NOT NULL AND lottery_year IS NOT NULL
                THEN firm_uid_clean || '_' || CAST(lottery_year AS VARCHAR)
            END) AS n_firm_years
        FROM prep
        GROUP BY winner
    )
    SELECT g.*, o.share_winner, o.n_apps_total, o.n_firm_years_total
    FROM grouped AS g
    CROSS JOIN overall AS o
    """
    out = {
        "share_winner": None,
        "n_apps_total": None,
        "n_firm_years_total": None,
        "winner": _empty_h1b_winlose_bucket(),
        "loser": _empty_h1b_winlose_bucket(),
    }

    df = con.execute(sql).fetchdf()
    if df.empty:
        return out

    out["share_winner"] = df["share_winner"].iloc[0]
    out["n_apps_total"] = df["n_apps_total"].iloc[0]
    out["n_firm_years_total"] = df["n_firm_years_total"].iloc[0]
    for _, row in df.iterrows():
        bucket = "winner" if int(row["winner"]) == 1 else "loser"
        out[bucket] = {
            "share_ade": row["share_ade"],
            "share_female": row["share_female"],
            "avg_age_at_app": row["avg_age_at_app"],
            "share_india": row["share_india"],
            "share_china": row["share_china"],
            "n_apps": row["n_apps"],
            "n_firm_years": row["n_firm_years"],
        }
    return out


def compute_h1b_winlose_stats_raw(
    con: duckdb.DuckDBPyConnection,
    csv_path: str,
    raw_match_path: str,
) -> dict | None:
    """Winner/loser H-1B stats from the full Bloomberg CSV."""
    if not Path(csv_path).exists():
        warnings.warn(f"Raw FOIA CSV not found, skipping: {csv_path}")
        return None
    if not Path(raw_match_path).exists():
        warnings.warn(f"foia_raw_match parquet not found, skipping: {raw_match_path}")
        return None

    base_sql = f"""
    WITH raw AS (
        SELECT
            ROW_NUMBER() OVER (ORDER BY foia_unique_id) AS foia_indiv_id,
            CASE WHEN status_type = 'SELECTED' THEN 1 ELSE 0 END AS winner,
            CASE
                WHEN status_type = 'SELECTED' AND TRIM(S3Q1) = 'M' THEN 1.0
                WHEN status_type = 'SELECTED' AND TRIM(S3Q1) = 'NA' THEN NULL
                WHEN status_type = 'SELECTED' THEN 0.0
                ELSE NULL
            END AS ade,
            CASE WHEN LOWER(TRIM(gender)) = 'female' THEN 1.0 ELSE 0.0 END AS female_ind,
            TRY_CAST(ben_year_of_birth AS INTEGER) AS yob,
            TRY_CAST(lottery_year AS INTEGER) AS lottery_year,
            country_of_nationality AS country
        FROM read_csv_auto('{csv_path}', ignore_errors=true, header=true)
        WHERE TRY_CAST(lottery_year AS INTEGER) IS NOT NULL
    )
    SELECT
        r.winner,
        r.ade,
        r.female_ind,
        r.yob,
        r.lottery_year,
        r.country,
        m.foia_firm_uid
    FROM raw AS r
    LEFT JOIN (
        SELECT foia_indiv_id, foia_firm_uid
        FROM read_parquet('{raw_match_path}')
    ) AS m USING (foia_indiv_id)
    """
    return _compute_h1b_winlose_stats(con, base_sql)


def compute_h1b_winlose_stats_parquet(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
    is_merged: bool = False,
    foia_indiv_path: str | None = None,
) -> dict | None:
    """Winner/loser H-1B stats from cleaned FOIA / merged parquets."""
    if not Path(parquet_path).exists():
        warnings.warn(f"Parquet not found, skipping: {parquet_path}")
        return None

    if is_merged:
        if foia_indiv_path is None or not Path(foia_indiv_path).exists():
            warnings.warn(
                "foia_indiv parquet is required to compute S3Q1-based ADE for merged samples; "
                f"skipping: {parquet_path}"
            )
            return None
        base_sql = f"""
        WITH dedup AS (
            SELECT
                foia_indiv_id,
                ANY_VALUE(status_type) AS status_type,
                ANY_VALUE(TRY_CAST(female_ind AS DOUBLE)) AS female_ind,
                ANY_VALUE(TRY_CAST(yob AS INTEGER)) AS yob,
                ANY_VALUE(TRY_CAST(lottery_year AS INTEGER)) AS lottery_year,
                ANY_VALUE(foia_country) AS country,
                ANY_VALUE(foia_firm_uid) AS foia_firm_uid
            FROM read_parquet('{parquet_path}')
            GROUP BY foia_indiv_id
        )
        SELECT
            CASE WHEN d.status_type = 'SELECTED' THEN 1 ELSE 0 END AS winner,
            CASE WHEN d.status_type = 'SELECTED' THEN TRY_CAST(f.ade_lottery AS DOUBLE) ELSE NULL END AS ade,
            d.female_ind,
            d.yob,
            d.lottery_year,
            d.country,
            d.foia_firm_uid
        FROM dedup AS d
        LEFT JOIN (
            SELECT foia_indiv_id, ade_lottery
            FROM read_parquet('{foia_indiv_path}')
        ) AS f USING (foia_indiv_id)
        """
    else:
        base_sql = f"""
        SELECT
            CASE WHEN status_type = 'SELECTED' THEN 1 ELSE 0 END AS winner,
            CASE WHEN status_type = 'SELECTED' THEN TRY_CAST(ade_lottery AS DOUBLE) ELSE NULL END AS ade,
            TRY_CAST(female_ind AS DOUBLE) AS female_ind,
            TRY_CAST(yob AS INTEGER) AS yob,
            TRY_CAST(lottery_year AS INTEGER) AS lottery_year,
            country,
            foia_firm_uid
        FROM read_parquet('{parquet_path}')
        """

    return _compute_h1b_winlose_stats(con, base_sql)


# ---------------------------------------------------------------------------
# Revelio stats
# ---------------------------------------------------------------------------

def compute_rev_stats(
    con: duckdb.DuckDBPyConnection,
    rev_parquet: str,
    matched_parquet: str | None = None,
) -> dict | None:
    """Compute Revelio user summary stats.

    If matched_parquet is None: stats over all users in rev_parquet.
    If matched_parquet is provided: filter to unique user_ids present there.
    """
    if not Path(rev_parquet).exists():
        warnings.warn(f"Revelio parquet not found: {rev_parquet}")
        return None

    # rev_indiv has one row per user x firm — dedup on user_id before computing stats
    if matched_parquet is not None:
        if not Path(matched_parquet).exists():
            warnings.warn(f"Matched parquet not found, skipping: {matched_parquet}")
            return None
        # Filter to users present in the matched sample, then dedup
        filter_join = f"""
            INNER JOIN (
                SELECT DISTINCT user_id
                FROM read_parquet('{matched_parquet}')
                WHERE user_id IS NOT NULL
            ) m ON r.user_id = m.user_id
        """
    else:
        filter_join = ""

    sql = f"""
    WITH dedup AS (
        SELECT
            ANY_VALUE(f_prob)            AS f_prob,
            ANY_VALUE(est_yob)           AS est_yob,
            ANY_VALUE(country)           AS country,
            ANY_VALUE(us_educ)           AS us_educ,
            ANY_VALUE(highest_ed_level)  AS highest_ed_level
        FROM read_parquet('{rev_parquet}') r
        {filter_join}
        GROUP BY r.user_id
    ),
    ind AS (
        SELECT
            f_prob,
            est_yob,
            CASE WHEN LOWER(country) = 'india'  THEN 1.0 ELSE 0.0 END                   AS india_ind,
            CASE WHEN LOWER(country) = 'china'  THEN 1.0 ELSE 0.0 END                   AS china_ind,
            CASE WHEN us_educ = 1 THEN 1.0 ELSE 0.0 END                                  AS us_educ_ind,
            CASE WHEN LOWER(highest_ed_level) = 'doctor'   THEN 1.0 ELSE 0.0 END         AS doctor_ind,
            CASE WHEN LOWER(highest_ed_level) = 'master'   THEN 1.0 ELSE 0.0 END         AS master_ind,
            CASE WHEN LOWER(highest_ed_level) = 'bachelor' THEN 1.0 ELSE 0.0 END         AS bachelor_ind,
            CASE WHEN LOWER(highest_ed_level) NOT IN ('doctor', 'master', 'bachelor')
                   OR highest_ed_level IS NULL THEN 1.0 ELSE 0.0 END                     AS other_ind
        FROM dedup
    )
    SELECT
        COUNT(*)                                                                           AS n,
        AVG(f_prob)                                                                        AS share_female,
        SQRT(AVG(f_prob) * (1 - AVG(f_prob)) / NULLIF(COUNT(*), 0))                       AS share_female_se,
        AVG(est_yob)                                                                       AS avg_birth_year,
        STDDEV_SAMP(est_yob) / SQRT(NULLIF(COUNT(est_yob), 0))                            AS avg_birth_year_se,
        AVG(india_ind)                                                                     AS share_india,
        SQRT(AVG(india_ind) * (1 - AVG(india_ind)) / NULLIF(COUNT(*), 0))                 AS share_india_se,
        AVG(china_ind)                                                                     AS share_china,
        SQRT(AVG(china_ind) * (1 - AVG(china_ind)) / NULLIF(COUNT(*), 0))                 AS share_china_se,
        AVG(us_educ_ind)                                                                   AS share_us_educ,
        SQRT(AVG(us_educ_ind) * (1 - AVG(us_educ_ind)) / NULLIF(COUNT(*), 0))             AS share_us_educ_se,
        AVG(doctor_ind)                                                                    AS share_doctor,
        SQRT(AVG(doctor_ind) * (1 - AVG(doctor_ind)) / NULLIF(COUNT(*), 0))               AS share_doctor_se,
        AVG(master_ind)                                                                    AS share_master,
        SQRT(AVG(master_ind) * (1 - AVG(master_ind)) / NULLIF(COUNT(*), 0))               AS share_master_se,
        AVG(bachelor_ind)                                                                  AS share_bachelor,
        SQRT(AVG(bachelor_ind) * (1 - AVG(bachelor_ind)) / NULLIF(COUNT(*), 0))           AS share_bachelor_se,
        AVG(other_ind)                                                                     AS share_other_educ,
        SQRT(AVG(other_ind) * (1 - AVG(other_ind)) / NULLIF(COUNT(*), 0))                 AS share_other_educ_se
    FROM ind
    """
    return con.execute(sql).fetchdf().iloc[0].to_dict()


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

H1B_COLS: list[tuple[str, str]] = [
    ("n",                    r"$N$"),
    ("share_winner",         r"\% Winner"),
    ("share_ade",            r"\% ADE"),
    ("share_female",         r"\% Female"),
    ("avg_age_at_app",       r"Avg.\ Age"),
    ("share_india",          r"\% India"),
    ("share_china",          r"\% China"),
    ("avg_apps_per_firm_year", r"Apps/Firm-Yr"),
]

REV_COLS: list[tuple[str, str]] = [
    ("n",               r"$N$"),
    ("share_female",    r"\% Female"),
    ("avg_birth_year",  r"Avg.\ Birth Yr"),
    ("share_india",     r"\% India"),
    ("share_china",     r"\% China"),
    ("share_us_educ",   r"\% US Educ"),
    ("share_doctor",    r"\% Doctor"),
    ("share_master",    r"\% Master"),
    ("share_bachelor",  r"\% Bach."),
    ("share_other_educ", r"\% Other"),
]


def _fmt(v, col: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"---"
    if col == "n":
        return f"{int(v):,}"
    if col in ("avg_age_at_app", "avg_birth_year", "avg_apps_per_firm_year"):
        return f"{v:.1f}"
    # All share_* columns → percentage
    return f"{100 * v:.1f}"


_COL_ABBREV: dict[str, str] = {
    "All H-1B":          r"\shortstack{All\\H-1B}",
    "In-sample firms":   r"\shortstack{In-sample\\firms}",
    "Baseline":          "Baseline",
    r"Mult $\leq$4":     r"Mult $\leq$4",
    "Pre-filtered":      r"\shortstack{Pre-\\filtered}",
    "Strict":            "Strict",
    "U.S. Educ.":        r"\shortstack{U.S.\\Educ.}",
    r"U.S.\ Education":  r"\shortstack{U.S.\\Educ.}",
    "All Revelio users": r"\shortstack{All Rev.\\users}",
    "One-to-one":        r"\shortstack{One-\\to-one}",
    "Strict Q50":        "Strict Q50",
}


def build_latex_table(rows: list[tuple[str, dict | None]], panel: str = "h1b") -> str:
    """Transposed table: rows = statistics, columns = samples.
    Wrapped in \\resizebox{\\textwidth}{!} so it always fits the page width.
    """
    cols = H1B_COLS if panel == "h1b" else REV_COLS
    n_samples = len(rows)
    col_spec = "l" + "r" * n_samples

    # Column headers = abbreviated sample names
    col_headers = " & ".join(
        _COL_ABBREV.get(label, label) for label, _ in rows
    )

    # Body: one row per statistic
    body_lines: list[str] = []
    for stat_col, stat_label in cols:
        cells: list[str] = []
        for _, stats in rows:
            cells.append(_fmt(stats.get(stat_col) if stats else None, stat_col))
        body_lines.append(f"    {stat_label} & {' & '.join(cells)} \\\\")
    body = "\n".join(body_lines)

    if panel == "h1b":
        caption = "Summary Statistics: H-1B Applications"
        label   = "tab:summary_h1b"
        notes   = (
            r"\% Female, \% India, \% China are computed exclusively from "
            r"FOIA administrative records. "
            r"For matched samples, statistics are over unique applicants "
            r"(deduplicated on applicant ID). "
            r"Apps/Firm-Yr = total applications divided by distinct "
            r"employer $\times$ lottery-year cells. "
            r"Pre-filtered sample explicitly excludes India (Southern Asia) and China by design."
        )
    else:
        caption = "Summary Statistics: Revelio Users"
        label   = "tab:summary_rev"
        notes   = (
            r"\% Female is the mean predicted female probability from "
            r"name-based inference ($f\_prob$). "
            r"Avg.\ Birth Yr is mean estimated year of birth ($est\_yob$). "
            r"For matched samples, $N$ counts all unique Revelio candidate users "
            r"linked to H-1B applicants in that sample."
        )

    return rf"""\begin{{table}}[htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\begin{{threeparttable}}
\begin{{tabular}}{{{col_spec}}}
\toprule
Statistic & {col_headers} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\footnotesize
\item {notes}
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}""".strip()


_H1B_WINLOSE_SPLIT_ROWS: list[tuple[str, str]] = [
    ("share_ade", r"\% ADE"),
    ("share_female", r"\% Female"),
    ("avg_age_at_app", r"Avg.\ Age at App."),
    ("share_india", r"\% India"),
    ("share_china", r"\% China"),
]

_H1B_WINLOSE_COUNT_ROWS: list[tuple[str, str]] = [
    ("n_apps_total", r"N apps"),
    ("n_firm_years_total", r"N firm $\times$ years"),
]


def _fmt_h1b_winlose(v, col: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"---"
    if col in ("n_apps", "n_firm_years", "n_apps_total", "n_firm_years_total"):
        return f"{int(v):,}"
    if col == "avg_age_at_app":
        return f"{v:.1f}"
    return f"{100 * v:.1f}"


def build_h1b_winner_loser_table(groups: list[tuple[str, dict | None]]) -> str:
    """Grouped H-1B table with Winner/Loser subcolumns within each sample."""
    n_groups = len(groups)
    col_spec = "l" + "rr" * n_groups

    top_headers = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{_COL_ABBREV.get(label, label)}}}" for label, _ in groups
    )
    cmidrules = "".join(
        rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n_groups)
    )
    sub_headers = "Statistic" + "".join(" & Winner & Loser" for _ in groups)

    body_lines: list[str] = []

    winner_cells: list[str] = []
    for _, stats in groups:
        share_winner = stats.get("share_winner") if stats else None
        winner_cells.append(
            rf"\multicolumn{{2}}{{c}}{{{_fmt_h1b_winlose(share_winner, 'share_winner')}}}"
        )
    body_lines.append(f"    \\% Winner & {' & '.join(winner_cells)} \\\\")

    for stat_col, stat_label in _H1B_WINLOSE_SPLIT_ROWS:
        cells: list[str] = []
        for _, stats in groups:
            winner_stats = stats.get("winner", {}) if stats else {}
            loser_stats = stats.get("loser", {}) if stats else {}
            cells.append(_fmt_h1b_winlose(winner_stats.get(stat_col), stat_col))
            cells.append(_fmt_h1b_winlose(loser_stats.get(stat_col), stat_col))
        body_lines.append(f"    {stat_label} & {' & '.join(cells)} \\\\")

    body_lines.append(r"    \midrule")

    for stat_col, stat_label in _H1B_WINLOSE_COUNT_ROWS:
        cells = []
        for _, stats in groups:
            total_val = stats.get(stat_col) if stats else None
            cells.append(
                rf"\multicolumn{{2}}{{c}}{{{_fmt_h1b_winlose(total_val, stat_col)}}}"
            )
        body_lines.append(f"    {stat_label} & {' & '.join(cells)} \\\\")

    body = "\n".join(body_lines)
    notes = (
        r"\% ADE is based on the FOIA field $S3Q1 = M$ and is shown only for selected "
        r"applicants; loser cells are undefined by construction. "
        r"$N$ apps and $N$ firm $\times$ years are sample-wide totals reported once per group; "
        r"$N$ firm $\times$ years counts distinct $foia\_firm\_uid$ $\times$ lottery-year cells. "
        r"The U.S.\ Educ.\ sample is deduplicated to unique applicants using $foia\_indiv\_id$ "
        r"before computing statistics."
    )

    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Summary Statistics: H-1B Applications by Lottery Outcome}}
\label{{tab:summary_h1b}}
\begin{{threeparttable}}
\begin{{tabular}}{{{col_spec}}}
\toprule
 & {top_headers} \\
{cmidrules}
{sub_headers} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\footnotesize
\item {notes}
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}""".strip()


# ---------------------------------------------------------------------------
# Color palette — dark-to-light green; consistent across H1B and Rev panels
# ---------------------------------------------------------------------------

_SAMPLE_COLOR_IDX: dict[str, int] = {
    "All H-1B":          0,   # full H1B sample
    "All Revelio users": 0,   # full Rev sample — same color as "All H-1B"
    "In-sample firms":   1,
    r"U.S.\ Education":  2,
    "Pre-filtered":      3,
    "One-to-one":        4,
    "Strict Q50":        5,
    # kept for backwards compat
    "Baseline":          2,
    r"Mult $\leq$4":     3,
    "Strict":            4,
}
_N_SAMPLE_COLORS = 6
_GREEN_PALETTE = plt.cm.Greens(np.linspace(0.98, 0.18, _N_SAMPLE_COLORS))


def _sample_color(label: str) -> np.ndarray:
    idx = _SAMPLE_COLOR_IDX.get(label, _N_SAMPLE_COLORS - 1)
    return _GREEN_PALETTE[idx]


_MPL_LABEL: dict[str, str] = {
    r"Mult $\leq$4":    "Mult \u22644",
    r"U.S.\ Education": "U.S. Education",
}


def _mpl_label(label: str) -> str:
    """Matplotlib-safe (non-LaTeX) version of a sample label."""
    return _MPL_LABEL.get(label, label)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

_STAT_LABELS: dict[str, str] = {
    "share_winner":           "\\% Winner",
    "share_ade":              "\\% ADE",
    "share_female":           "\\% Female",
    "avg_age_at_app":         "Avg. Age at App.",
    "avg_birth_year":         "Avg. Birth Year",
    "share_india":            "\\% India",
    "share_china":            "\\% China",
    "avg_apps_per_firm_year": "Avg. Apps / Firm-Year",
    "share_us_educ":          "\\% U.S. Education",
    "share_doctor":           "\\% Doctor",
    "share_master":           "\\% Master",
    "share_bachelor":         "\\% Bachelor",
    "share_other_educ":       "\\% Other Ed.",
}


def _is_pct(col: str) -> bool:
    return col.startswith("share_")


def _get_value(stats: dict | None, col: str) -> float:
    if stats is None:
        return np.nan
    v = stats.get(col)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    return v * 100 if _is_pct(col) else v


def _get_se(stats: dict | None, col: str) -> float:
    """Return the SE of `col` in the same units as _get_value (pct for shares)."""
    if stats is None:
        return np.nan
    v = stats.get(col + "_se")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    return float(v) * 100 if _is_pct(col) else float(v)


def _full_individual_std(full_stats: dict | None, col: str) -> float:
    """Individual-level std of `col` in the full sample (same units as _get_value).

    For shares: sqrt(p*(1-p))*100.
    For continuous: se * sqrt(n)  (= sample std dev of the full distribution).
    """
    if full_stats is None:
        return 1.0
    full_val = _get_value(full_stats, col)
    if np.isnan(full_val):
        return 1.0
    if _is_pct(col):
        p = full_val / 100.0
        std = np.sqrt(max(p * (1.0 - p), 0.0)) * 100
    else:
        full_se = _get_se(full_stats, col)
        n = full_stats.get("n")
        if np.isnan(full_se) or n is None or float(n) <= 0:
            return 1.0
        std = full_se * np.sqrt(float(n))
    return std if (std > 0 and not np.isnan(std)) else 1.0


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if _IN_IPYTHON:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Normalized horizontal bar-chart figure
# ---------------------------------------------------------------------------

def build_normalized_barplot(
    rows: list[tuple[str, dict | None]],
    panel: str = "h1b",
    figsize: tuple | None = None,
) -> plt.Figure:
    """Horizontal grouped bar chart with cross-sample z-scores and SE bars.

    Normalization: z_i = (val_i - cross_mean) / cross_std, where cross_mean
    and cross_std are the mean and std of the sample-level values across all
    shown samples (ignoring NaN). This gives mean = 0 and std ≈ 1 across bars
    for each statistic. Error bars show ±1 SE of the sample mean divided by
    the same cross_std.
    Colors: dark-to-light green gradient, consistent across H1B and Rev panels.
    """
    cols = H1B_COLS if panel == "h1b" else REV_COLS
    plot_cols = [(c, h) for c, h in cols if c != "n"]

    n_stats   = len(plot_cols)
    n_samples = len(rows)

    # --- normalize each stat relative to cross-sample mean and std ---
    z_matrix:  list[np.ndarray] = []
    se_matrix: list[np.ndarray] = []
    for col, _ in plot_cols:
        vals = np.array([_get_value(stats, col) for _, stats in rows])
        ses  = np.array([_get_se(stats, col)    for _, stats in rows])
        valid = vals[~np.isnan(vals)]
        cross_mean = np.mean(valid) if len(valid) > 0 else 0.0
        cross_std  = np.std(valid, ddof=1) if len(valid) > 1 else 1.0
        if cross_std == 0 or np.isnan(cross_std):
            cross_std = 1.0
        z_matrix.append((vals - cross_mean) / cross_std)
        se_matrix.append(ses / cross_std)

    # --- y-positions ---
    bar_h         = 0.7 / n_samples
    gap           = 0.5
    group_h       = n_samples * bar_h
    group_centers = np.arange(n_stats) * (group_h + gap)

    if figsize is None:
        total_h = n_stats * (group_h + gap) + 1.2
        figsize = (7.0, max(4.0, total_h))

    fig, ax = plt.subplots(figsize=figsize)

    for si, (label, _) in enumerate(rows):
        color   = _sample_color(label)
        mpl_lbl = _mpl_label(label)
        offset  = (si - n_samples / 2.0 + 0.5) * bar_h
        y_pos   = group_centers + offset
        z_vals  = np.array([z_matrix[ci][si]  for ci in range(n_stats)])
        se_vals = np.array([se_matrix[ci][si] for ci in range(n_stats)])
        for yi, zv, sv in zip(y_pos, z_vals, se_vals):
            kw: dict = {}
            if not np.isnan(sv):
                kw["xerr"] = sv
                kw["error_kw"] = {
                    "elinewidth": 0.8, "capsize": 2, "capthick": 0.8,
                    "ecolor": "black", "alpha": 0.6,
                }
            ax.barh(
                yi, 0.0 if np.isnan(zv) else zv,
                height=bar_h * 0.88,
                color=color, alpha=0.85,
                label=mpl_lbl,
                **kw,
            )

    ax.set_yticks(group_centers)
    ax.set_yticklabels([_STAT_LABELS.get(c, c) for c, _ in plot_cols], fontsize=8)
    ax.invert_yaxis()

    ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.55)
    ax.set_xlabel("Standardized value (z-score; mean\u2009=\u20090 across samples)", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    # Deduplicate legend (one entry per sample)
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    h_dedup, l_dedup = [], []
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen.add(lbl)
            h_dedup.append(h)
            l_dedup.append(lbl)
    ax.legend(
        h_dedup, l_dedup,
        bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=7, title="Sample", title_fontsize=7, framealpha=0.7,
    )

    title = "H-1B Applications" if panel == "h1b" else "Revelio Users"
    ax.set_title(f"Summary Statistics: {title}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg     = load_config()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths   = cfg["paths"]

    con = duckdb.connect()

    # -----------------------------------------------------------------------
    # Panel A: H1B application stats
    # -----------------------------------------------------------------------
    print("Computing H1B stats...")
    h1b_rows: list[tuple[str, dict | None]] = []

    print("  Full H1B sample (raw CSV)...")
    h1b_rows.append(("All H-1B", compute_h1b_stats_raw(con, cfg["foia_raw_csv"])))

    print("  In-sample firms...")
    h1b_rows.append((
        "In-sample firms",
        compute_h1b_stats_parquet(con, str(paths["foia_indiv"]), is_merged=False),
    ))

    h1b_matched = [
        ("merge_us_educ",         r"U.S.\ Education"),
        ("merge_us_educ_prefilt", "Pre-filtered"),
        ("merge_us_educ_opt",     "One-to-one"),
        ("merge_strict_med",      "Strict Q50"),
    ]
    for key, label in h1b_matched:
        print(f"  {label}...")
        h1b_rows.append((
            label,
            compute_h1b_stats_parquet(con, str(paths[key]), is_merged=True),
        ))

    print("Computing H1B winner/loser table stats...")
    h1b_table_groups: list[tuple[str, dict | None]] = [
        (
            "All H-1B",
            compute_h1b_winlose_stats_raw(
                con,
                cfg["foia_raw_csv"],
                str(paths["foia_raw_match"]),
            ),
        ),
        ("In-sample firms", compute_h1b_winlose_stats_parquet(con, str(paths["foia_indiv"]))),
        (
            "U.S. Educ.",
            compute_h1b_winlose_stats_parquet(
                con,
                str(paths["merge_us_educ"]),
                is_merged=True,
                foia_indiv_path=str(paths["foia_indiv"]),
            ),
        ),
    ]

    # -----------------------------------------------------------------------
    # Panel B: Revelio user stats
    # -----------------------------------------------------------------------
    print("Computing Revelio stats...")
    rev_rows: list[tuple[str, dict | None]] = []

    print("  All Revelio users (in-sample firms)...")
    rev_rows.append((
        "All Revelio users",
        compute_rev_stats(con, str(paths["rev_indiv"])),
    ))

    rev_matched = [
        ("merge_us_educ",         r"U.S.\ Education"),
        ("merge_us_educ_prefilt", "Pre-filtered"),
        ("merge_us_educ_opt",     "One-to-one"),
        ("merge_strict_med",      "Strict Q50"),
    ]
    for key, label in rev_matched:
        print(f"  {label}...")
        rev_rows.append((
            label,
            compute_rev_stats(con, str(paths["rev_indiv"]), str(paths[key])),
        ))

    con.close()

    # -----------------------------------------------------------------------
    # Save raw CSV (all numbers in one place for inspection)
    # -----------------------------------------------------------------------
    records: list[dict] = []
    for label, stats in h1b_rows:
        if stats is not None:
            records.append({"panel": "h1b", "sample": label, **stats})
    for label, stats in rev_rows:
        if stats is not None:
            records.append({"panel": "rev", "sample": label, **stats})

    csv_path = out_dir / "summary_stats_raw.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"Saved raw stats → {csv_path}")

    h1b_winlose_records: list[dict] = []
    for label, stats in h1b_table_groups:
        if stats is None:
            continue
        for bucket in ("winner", "loser"):
            h1b_winlose_records.append({
                "sample": label,
                "bucket": bucket,
                "share_winner": stats.get("share_winner"),
                "n_apps_total": stats.get("n_apps_total"),
                "n_firm_years_total": stats.get("n_firm_years_total"),
                **stats.get(bucket, {}),
            })

    h1b_winlose_csv = out_dir / "h1b_summary_winner_loser_raw.csv"
    pd.DataFrame(h1b_winlose_records).to_csv(h1b_winlose_csv, index=False)
    print(f"Saved H1B winner/loser stats → {h1b_winlose_csv}")

    # -----------------------------------------------------------------------
    # LaTeX tables
    # -----------------------------------------------------------------------
    (out_dir / "h1b_summary_table.tex").write_text(build_h1b_winner_loser_table(h1b_table_groups))
    (out_dir / "rev_summary_table.tex").write_text(build_latex_table(rev_rows, "rev"))
    print(f"Saved LaTeX tables → {out_dir}/{{h1b,rev}}_summary_table.tex")

    # -----------------------------------------------------------------------
    # Figures — normalized horizontal bar charts
    # -----------------------------------------------------------------------
    _save(build_normalized_barplot(h1b_rows, "h1b"), out_dir / "h1b_summary_normplot.pdf")
    _save(build_normalized_barplot(rev_rows, "rev"), out_dir / "rev_summary_normplot.pdf")
    print(f"Saved figures → {out_dir}/{{h1b,rev}}_summary_normplot.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
