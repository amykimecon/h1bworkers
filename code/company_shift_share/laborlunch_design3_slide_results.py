"""Build slide-ready Design 3 result figures for slides_laborlunch_20260507.

The figures here are intentionally downstream of the comparison suite.  They
reuse the current strict transition-share cache, construct matched stacks for
loose and strict high-OPT-takeup-cell exposure families, and render the specific
firm-level result slides requested for the labor lunch.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import laborlunch_plot_style as llstyle
    from company_shift_share.config_loader import get_cfg_section, load_config
    from company_shift_share.design_comparison_suite import (
        build_comparison_stacked_panel,
        detect_largest_jump_events,
        ensure_shift_share_share_variants,
        estimate_stacked_event_coefficients,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import laborlunch_plot_style as llstyle
    from company_shift_share.config_loader import get_cfg_section, load_config
    from company_shift_share.design_comparison_suite import (
        build_comparison_stacked_panel,
        detect_largest_jump_events,
        ensure_shift_share_share_variants,
        estimate_stacked_event_coefficients,
    )


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/yk0581/data/out/company_shift_share_apr2026")
DEFAULT_CONFIG = CODE_ROOT / "configs" / "company_shift_share_design3_outcome_check.yaml"
DEFAULT_PANEL = (
    DATA_ROOT
    / "design3_strict_ihmp_share_2013_2016_pre3_post5_ihs_no_replacement_p35"
    / "prepared_panels"
    / "stacked_did.parquet"
)
DEFAULT_OUT_DIR = CODE_ROOT / "output" / "company_shift_share" / "slides_20260507_design3"
POSITIONS_PATH = DATA_ROOT / "wrds_workforce_selected_us_positions.parquet"
PROFILE_PATH = DATA_ROOT / "wrds_user_profile_origin_cache.parquet"
FOIA_PATH = Path("/home/yk0581/data/int/foia_sevp_with_person_id_employment_corrected.parquet")
IPEDS_MA_PATH = Path("/home/yk0581/data/int/ipeds_ma_only.parquet")
IPEDS_MA_BA_PATH = Path("/home/yk0581/data/int/ipeds_ma_ba_only.parquet")
TRANSITION_SHARES_PATH = Path("/home/yk0581/data/out/company_shift_share/apr2026/transition_shares.parquet")

HIGH_OPT_DEFINITIONS = {
    "loose": {
        "method": "all_years_cip4_foia_n_gt_100_ge_10pct_over_ipeds",
        "threshold": 0.10,
        "min_foia_n": 100,
        "label": "Loose high-OPT",
        "table_title": "Loose High-OPT-Takeup CIP4 x Degree Cells",
    },
    "strict": {
        "method": "all_years_cip4_foia_n_gt_1000_ge_50pct_over_ipeds",
        "threshold": 0.50,
        "min_foia_n": 1000,
        "label": "Strict high-OPT",
        "table_title": "Strict High-OPT-Takeup CIP4 x Degree Cells",
    },
}
POOLED_EXPOSURE = "high_opt_takeup_strict_share"
STRICT_EXPOSURE = "high_opt_takeup_strict_share"
LOOSE_EXPOSURE = "high_opt_takeup_loose_share"
MAIN_EXPOSURE_KEYS = [STRICT_EXPOSURE]
LOOSE_EXPOSURE_KEYS = [LOOSE_EXPOSURE]

EXPOSURES = {
    "high_opt_takeup_loose_share": {
        "label": "Loose high-OPT",
        "coef_label": "Loose high-OPT-cell exposure",
        "col": "z_ct_high_opt_takeup_loose_share",
        "color": "#9E2F45",
        "marker": "o",
        "high_opt_definition": "loose",
    },
    "high_opt_takeup_strict_share": {
        "label": "Strict high-OPT",
        "coef_label": "Strict high-OPT-cell exposure",
        "col": "z_ct_high_opt_takeup_strict_share",
        "color": "#005AB5",
        "marker": "o",
        "high_opt_definition": "strict",
    },
}

EXPOSURE_MATCH_WITH_REPLACEMENT = {
    "high_opt_takeup_loose_share": False,
    "high_opt_takeup_strict_share": False,
}

OUTCOME_LABELS = {
    "any_opt_hires_correction_aware": "OPT hires",
    "masters_opt_hires_correction_aware": "Master's OPT hires",
    "y_new_hires_foreign_lag0": "Foreign new-grad hires",
    "y_new_hires_native_lag0": "Native new-grad hires",
    "y_new_hires_foreign_lag0": "IHS(foreign new-grad hires)",
    "y_new_hires_native_lag0": "IHS(native new-grad hires)",
    "avg_tenure_opt_likely_foreign_active_lag0": "Avg Tenure: Foreign Employees",
    "avg_tenure_opt_likely_native_active_lag0": "Avg Tenure: Native Employees",
    "avg_tenure_foreign_active_lag0": "Avg Tenure: Foreign Employees",
    "avg_tenure_native_active_lag0": "Avg Tenure: Native Employees",
    "y_intern_users_opt_likely_foreign_lag0": "Foreign interns",
    "y_intern_users_opt_likely_native_lag0": "Native interns",
}

COEF_MARKER_SIZE = llstyle.MARKER_SIZE
TWO_PANEL_MARKER_SIZE = 8.5
TITLE_FONT_SIZE = 14
EVENT_TIME_LABEL_SHIFT = 1
SHOW_LEGENDS = False


def _read_cfg(config_path: Path) -> dict:
    cfg = load_config(config_path)
    cmp_cfg = dict(get_cfg_section(cfg, "design_comparison"))
    cmp_cfg.update(
        {
            "designs_to_run": ["stacked_did"],
            "stacked_exposures_to_run": ["ihmp_share"],
            "stacked_matching_styles": ["matched"],
            "stacked_match_with_replacement": False,
            "stacked_jump_min_year": 2013,
            "stacked_jump_max_year": 2017,
            "stacked_pre_years": 4,
            "stacked_post_years": 4,
            "stacked_ref_event_time": -2,
            "first_stage_type": "ols_continuous",
            "first_stage_col": "any_opt_hires_correction_aware",
            "count_outcome_transform": "none",
            "use_log_outcome": False,
            "verbose": True,
            "show_figures": False,
            "outcome_cols": [],
        }
    )
    return cmp_cfg


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def _fmt_signed(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):+.{digits}f}"


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _truncate_text(value: object, max_len: int = 44) -> str:
    text = "" if pd.isna(value) else str(value)
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _save_high_opt_cell_table(cells: pd.DataFrame, out_path: Path, *, definition: dict[str, object]) -> None:
    high = cells.loc[cells["high_opt_takeup"].eq(1)].copy()
    if high.empty:
        return
    high = high.sort_values(["foia_n", "opt_takeup"], ascending=[False, False]).head(30)
    table_df = pd.DataFrame(
        {
            "Degree": high["degree_group"],
            "CIP": high["cip_display"],
            "Field": high["cip_desc"].map(lambda value: _truncate_text(value, 48)),
            "FOIA/IPEDS": high["opt_takeup"].map(lambda value: f"{100 * float(value):.1f}%"),
            "N": high["foia_n"].map(lambda value: f"{int(value):,}"),
        }
    )
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.axis("off")
    high_count = int(cells["high_opt_takeup"].sum())
    min_foia_n = int(definition["min_foia_n"])
    threshold = float(definition["threshold"])
    subtitle = (
        f"Top 30 by FOIA cell size; full list has {high_count:,} high cells. "
        f"Rule: FOIA N > {min_foia_n:,}, FOIA/IPEDS >= {threshold:.0%}, "
        "all available years."
    )
    ax.text(
        0.0,
        1.02,
        str(definition["table_title"]),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="semibold",
    )
    ax.text(0.0, 0.985, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=9.5)
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 0.94],
        colWidths=[0.11, 0.10, 0.57, 0.10, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("0.85")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.94")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def _high_opt_takeup_shift_share(out_dir: Path, definition_key: str, *, force: bool = False) -> pd.DataFrame:
    """Build z_ct from school shares in FOIA/IPEDS high-takeup CIP4-degree cells."""
    definition = HIGH_OPT_DEFINITIONS[definition_key]
    method = str(definition["method"])
    threshold = float(definition["threshold"])
    min_foia_n = int(definition["min_foia_n"])
    metric_col = f"high_opt_takeup_{definition_key}_share"
    z_col = f"z_ct_high_opt_takeup_{definition_key}_share"
    n_univ_col = f"n_universities_high_opt_takeup_{definition_key}_share"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cells_path = table_dir / f"high_opt_takeup_cip_degree_cells_{method}.csv"
    metric_path = table_dir / f"school_high_opt_takeup_metric_{method}.parquet"
    zct_path = table_dir / f"z_ct_high_opt_takeup_{definition_key}_share_{method}.parquet"
    table_png = fig_dir / f"high_opt_takeup_cells_{definition_key}.png"
    if zct_path.exists() and cells_path.exists() and not force:
        cells = pd.read_csv(cells_path)
        _save_high_opt_cell_table(cells, table_png, definition=definition)
        return pd.read_parquet(zct_path)
    missing = [
        str(path)
        for path in [FOIA_PATH, IPEDS_MA_PATH, IPEDS_MA_BA_PATH, TRANSITION_SHARES_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing inputs for high-OPT-takeup exposure:\n" + "\n".join(missing))
    con = ddb.connect()
    try:
        cells = con.sql(
            f"""
            WITH
            foia_person_cell AS (
                SELECT
                    CAST(person_id AS VARCHAR) AS person_id,
                    CASE
                        WHEN LOWER(CAST(student_edu_level_desc AS VARCHAR)) LIKE '%master%' THEN 'Master'
                        WHEN LOWER(CAST(student_edu_level_desc AS VARCHAR)) LIKE '%bachelor%' THEN 'Bachelor'
                        ELSE NULL
                    END AS degree_group,
                    LPAD(
                        SUBSTRING(
                            REGEXP_REPLACE(TRIM(CAST(major_1_cip_code AS VARCHAR)), '[^0-9]', '', 'g')
                            FROM 1 FOR 4
                        ),
                        4,
                        '0'
                    ) AS cip4,
                    ANY_VALUE(NULLIF(TRIM(CAST(major_1_description AS VARCHAR)), '')) AS cip_desc
                FROM read_parquet('{_sql_path(FOIA_PATH)}')
                WHERE person_id IS NOT NULL
                  AND program_end_date IS NOT NULL
                  AND year_int = EXTRACT(YEAR FROM CAST(program_end_date AS DATE))
                  AND major_1_cip_code IS NOT NULL
                GROUP BY 1, 2, 3
            ),
            ma AS (
                SELECT
                    CAST(year AS INTEGER) AS t,
                    SUBSTRING(LPAD(CAST(CAST(cipcode AS BIGINT) AS VARCHAR), 6, '0') FROM 1 FOR 4) AS cip4,
                    SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0.0)) AS ma_students
                FROM read_parquet('{_sql_path(IPEDS_MA_PATH)}')
                WHERE year IS NOT NULL AND cipcode IS NOT NULL
                GROUP BY 1, 2
            ),
            bama AS (
                SELECT
                    CAST(year AS INTEGER) AS t,
                    SUBSTRING(LPAD(CAST(CAST(cipcode AS BIGINT) AS VARCHAR), 6, '0') FROM 1 FOR 4) AS cip4,
                    SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0.0)) AS bama_students
                FROM read_parquet('{_sql_path(IPEDS_MA_BA_PATH)}')
                WHERE year IS NOT NULL AND cipcode IS NOT NULL
                GROUP BY 1, 2
            ),
            joined AS (
                SELECT
                    COALESCE(b.t, m.t) AS t,
                    COALESCE(b.cip4, m.cip4) AS cip4,
                    COALESCE(b.bama_students, 0.0) AS bama_students,
                    COALESCE(m.ma_students, 0.0) AS ma_students
                FROM bama b
                FULL OUTER JOIN ma m
                  ON b.t = m.t AND b.cip4 = m.cip4
            ),
            ipeds_cells AS (
                SELECT cip4, 'Master' AS degree_group, SUM(ma_students) AS ipeds_total
                FROM joined
                GROUP BY 1
                HAVING SUM(ma_students) > 0
                UNION ALL
                SELECT cip4, 'Bachelor' AS degree_group, SUM(GREATEST(bama_students - ma_students, 0.0)) AS ipeds_total
                FROM joined
                GROUP BY 1
                HAVING SUM(GREATEST(bama_students - ma_students, 0.0)) > 0
            ),
            foia_cells AS (
                SELECT
                    degree_group,
                    cip4,
                    ANY_VALUE(cip_desc) AS cip_desc,
                    COUNT(DISTINCT person_id)::INTEGER AS foia_n
                FROM foia_person_cell
                WHERE degree_group IN ('Bachelor', 'Master')
                  AND cip4 IS NOT NULL
                  AND cip4 <> '0000'
                GROUP BY 1, 2
            )
            SELECT
                f.degree_group,
                f.cip4,
                SUBSTRING(f.cip4 FROM 1 FOR 2) || '.' || SUBSTRING(f.cip4 FROM 3 FOR 2) AS cip_display,
                f.cip_desc,
                f.foia_n,
                CAST(i.ipeds_total AS DOUBLE) AS ipeds_total,
                f.foia_n / NULLIF(CAST(i.ipeds_total AS DOUBLE), 0.0) AS opt_takeup,
                CASE
                    WHEN f.foia_n > {min_foia_n}
                     AND f.foia_n / NULLIF(CAST(i.ipeds_total AS DOUBLE), 0.0) >= {threshold}
                    THEN 1 ELSE 0
                END AS high_opt_takeup
            FROM foia_cells f
            JOIN ipeds_cells i
              ON f.degree_group = i.degree_group
             AND f.cip4 = i.cip4
            ORDER BY high_opt_takeup DESC, f.foia_n DESC, opt_takeup DESC
            """
        ).df()
        con.register("high_cells", cells[["degree_group", "cip4", "high_opt_takeup"]])
        metric = con.sql(
            f"""
            WITH
            ma AS (
                SELECT
                    CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k,
                    CAST(year AS INTEGER) AS t,
                    SUBSTRING(LPAD(CAST(CAST(cipcode AS BIGINT) AS VARCHAR), 6, '0') FROM 1 FOR 4) AS cip4,
                    SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0.0)) AS ma_students
                FROM read_parquet('{_sql_path(IPEDS_MA_PATH)}')
                WHERE unitid IS NOT NULL AND year IS NOT NULL AND cipcode IS NOT NULL
                GROUP BY 1, 2, 3
            ),
            bama AS (
                SELECT
                    CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k,
                    CAST(year AS INTEGER) AS t,
                    SUBSTRING(LPAD(CAST(CAST(cipcode AS BIGINT) AS VARCHAR), 6, '0') FROM 1 FOR 4) AS cip4,
                    SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0.0)) AS bama_students
                FROM read_parquet('{_sql_path(IPEDS_MA_BA_PATH)}')
                WHERE unitid IS NOT NULL AND year IS NOT NULL AND cipcode IS NOT NULL
                GROUP BY 1, 2, 3
            ),
            joined AS (
                SELECT
                    COALESCE(b.k, m.k) AS k,
                    COALESCE(b.t, m.t) AS t,
                    COALESCE(b.cip4, m.cip4) AS cip4,
                    COALESCE(b.bama_students, 0.0) AS bama_students,
                    COALESCE(m.ma_students, 0.0) AS ma_students
                FROM bama b
                FULL OUTER JOIN ma m
                  ON b.k = m.k AND b.t = m.t AND b.cip4 = m.cip4
            ),
            ipeds_cells AS (
                SELECT k, t, cip4, 'Master' AS degree_group, ma_students AS students
                FROM joined
                WHERE ma_students > 0
                UNION ALL
                SELECT k, t, cip4, 'Bachelor' AS degree_group, GREATEST(bama_students - ma_students, 0.0) AS students
                FROM joined
                WHERE GREATEST(bama_students - ma_students, 0.0) > 0
            )
            SELECT
                i.k,
                i.t,
                SUM(i.students) AS school_size,
                SUM(CASE WHEN COALESCE(h.high_opt_takeup, 0) = 1 THEN i.students ELSE 0.0 END)
                    AS high_opt_takeup_students,
                CASE
                    WHEN SUM(i.students) > 0
                    THEN SUM(CASE WHEN COALESCE(h.high_opt_takeup, 0) = 1 THEN i.students ELSE 0.0 END)
                         / SUM(i.students)
                    ELSE NULL
                END AS {metric_col}
            FROM ipeds_cells i
            LEFT JOIN high_cells h
              ON i.degree_group = h.degree_group
             AND i.cip4 = h.cip4
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).df()
        con.register("high_opt_metric", metric)
        zct = con.sql(
            f"""
            SELECT
                CAST(s.c AS VARCHAR) AS c,
                CAST(m.t AS INTEGER) AS t,
                SUM(
                    CASE WHEN TRY_CAST(s.share_ck AS DOUBLE) > 1 THEN NULL
                         ELSE TRY_CAST(s.share_ck AS DOUBLE)
                    END
                    * COALESCE(CAST(m.{metric_col} AS DOUBLE), 0.0)
                ) AS {z_col},
                COUNT(DISTINCT CASE
                    WHEN TRY_CAST(s.share_ck AS DOUBLE) > 0
                     AND COALESCE(CAST(m.{metric_col} AS DOUBLE), 0.0) != 0
                    THEN s.k END
                ) AS {n_univ_col}
            FROM read_parquet('{_sql_path(TRANSITION_SHARES_PATH)}') s
            JOIN high_opt_metric m
              ON CAST(s.k AS VARCHAR) = m.k
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).df()
    finally:
        con.close()
    cells.to_csv(cells_path, index=False)
    metric.to_parquet(metric_path, index=False)
    zct.to_parquet(zct_path, index=False)
    _save_high_opt_cell_table(cells, table_png, definition=definition)
    return zct


def _active_tenure_by_origin(out_dir: Path, *, force: bool = False) -> pd.DataFrame:
    cache = out_dir / "tables" / "active_tenure_by_origin.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    if not POSITIONS_PATH.exists() or not PROFILE_PATH.exists():
        raise FileNotFoundError("WRDS position/profile caches needed for origin-specific tenure are missing.")
    out_dir.joinpath("tables").mkdir(parents=True, exist_ok=True)
    soc2_values = "('15'), ('17'), ('19'), ('13'), ('11')"
    con = ddb.connect()
    try:
        out = con.sql(
            f"""
            WITH pos AS MATERIALIZED (
                SELECT
                    CAST(rcid AS BIGINT) AS c,
                    CAST(user_id AS BIGINT) AS user_id,
                    CAST(startdate AS DATE) AS startdate,
                    COALESCE(CAST(enddate AS DATE), DATE '2026-12-31') AS enddate,
                    CASE
                        WHEN onet_code IS NULL OR TRIM(CAST(onet_code AS VARCHAR)) = '' THEN NULL
                        ELSE SUBSTRING(CAST(onet_code AS VARCHAR), 1, 2)
                    END AS soc2
                FROM read_parquet('{_sql_path(POSITIONS_PATH)}')
                WHERE rcid IS NOT NULL
                  AND user_id IS NOT NULL
                  AND CAST(startdate AS DATE) IS NOT NULL
                  AND EXTRACT(YEAR FROM CAST(startdate AS DATE)) <= 2022
                  AND EXTRACT(YEAR FROM COALESCE(CAST(enddate AS DATE), DATE '2026-12-31')) >= 2010
            ),
            profile AS MATERIALIZED (
                SELECT
                    CAST(user_id AS BIGINT) AS user_id,
                    COALESCE(CAST(likely_foreign_hard AS INTEGER), 0) AS likely_foreign_hard
                FROM read_parquet('{_sql_path(PROFILE_PATH)}')
                WHERE user_id IS NOT NULL
            ),
            firm_user_bounds AS MATERIALIZED (
                SELECT c, user_id, MIN(startdate) AS first_start
                FROM pos
                GROUP BY 1, 2
            ),
            user_year AS MATERIALIZED (
                SELECT
                    p.c,
                    gs.year::INTEGER AS t,
                    p.user_id,
                    MAX(CASE WHEN p.soc2 IN (SELECT * FROM (VALUES {soc2_values}) AS v(soc2)) THEN 1 ELSE 0 END)
                        AS has_opt_likely_job,
                    MAX(COALESCE(pr.likely_foreign_hard, 0)) AS likely_foreign_hard,
                    MIN(b.first_start) AS first_start
                FROM pos p
                JOIN firm_user_bounds b
                  ON b.c = p.c
                 AND b.user_id = p.user_id
                LEFT JOIN profile pr
                  ON pr.user_id = p.user_id
                JOIN LATERAL generate_series(
                    GREATEST(EXTRACT(YEAR FROM p.startdate)::INTEGER, 2010),
                    LEAST(EXTRACT(YEAR FROM p.enddate)::INTEGER, 2022)
                ) AS gs(year) ON TRUE
                GROUP BY 1, 2, 3
            )
            SELECT
                c,
                t,
                AVG(CASE
                    WHEN has_opt_likely_job = 1 AND likely_foreign_hard = 1
                    THEN GREATEST(0.0, (make_date(t, 12, 31) - first_start)::DOUBLE / 365.25)
                    ELSE NULL
                END) AS avg_tenure_opt_likely_foreign_active_lag0,
                AVG(CASE
                    WHEN has_opt_likely_job = 1 AND likely_foreign_hard = 0
                    THEN GREATEST(0.0, (make_date(t, 12, 31) - first_start)::DOUBLE / 365.25)
                    ELSE NULL
                END) AS avg_tenure_opt_likely_native_active_lag0,
                AVG(CASE
                    WHEN likely_foreign_hard = 1
                    THEN GREATEST(0.0, (make_date(t, 12, 31) - first_start)::DOUBLE / 365.25)
                    ELSE NULL
                END) AS avg_tenure_foreign_active_lag0,
                AVG(CASE
                    WHEN likely_foreign_hard = 0
                    THEN GREATEST(0.0, (make_date(t, 12, 31) - first_start)::DOUBLE / 365.25)
                    ELSE NULL
                END) AS avg_tenure_native_active_lag0
            FROM user_year
            GROUP BY 1, 2
            """
        ).df()
    finally:
        con.close()
    out.to_parquet(cache, index=False)
    return out


def _load_panel(panel_path: Path, out_dir: Path, cmp_cfg: dict, *, force_origin_tenure: bool) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    panel["t"] = _safe_num(panel["t"]).astype(int)
    panel = ensure_shift_share_share_variants(panel, cmp_cfg)
    panel["c"] = panel["c"].astype(str)
    for definition_key in HIGH_OPT_DEFINITIONS:
        high_opt_zct = _high_opt_takeup_shift_share(out_dir, definition_key)
        high_opt_zct["c"] = high_opt_zct["c"].astype(str)
        high_opt_zct["t"] = _safe_num(high_opt_zct["t"]).astype(int)
        panel = panel.drop(columns=[col for col in high_opt_zct.columns if col not in {"c", "t"} and col in panel.columns])
        panel = panel.merge(high_opt_zct, on=["c", "t"], how="left")
    tenure = _active_tenure_by_origin(out_dir, force=force_origin_tenure)
    tenure["c"] = tenure["c"].astype(str)
    tenure["t"] = _safe_num(tenure["t"]).astype(int)
    panel = panel.merge(tenure, on=["c", "t"], how="left")
    panel["y_intern_users_opt_likely_foreign_lag0"] = _safe_num(
        panel["y_intern_positions_opt_likely_foreign_lag0"]
    ).fillna(0.0)
    panel["y_intern_users_opt_likely_native_lag0"] = (
        _safe_num(panel["y_intern_positions_opt_likely_lag0"]).fillna(0.0)
        - _safe_num(panel["y_intern_positions_opt_likely_foreign_lag0"]).fillna(0.0)
    ).clip(lower=0.0)
    return panel


def _stack_path(out_dir: Path, exposure: str, cmp_cfg: dict) -> Path:
    replacement = "replacement" if bool(cmp_cfg.get("stacked_match_with_replacement", False)) else "no_replacement"
    return out_dir / "prepared_panels" / f"stacked_matched_{exposure}_{replacement}.parquet"


def _cfg_for_exposure(cmp_cfg: dict, exposure: str) -> dict:
    cfg = dict(cmp_cfg)
    cfg["stacked_match_with_replacement"] = bool(EXPOSURE_MATCH_WITH_REPLACEMENT.get(exposure, False))
    return cfg


def _build_or_load_stacks(
    panel: pd.DataFrame,
    out_dir: Path,
    cmp_cfg: dict,
    *,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    out_dir.joinpath("prepared_panels").mkdir(parents=True, exist_ok=True)
    for exposure, meta in EXPOSURES.items():
        exp_cfg = _cfg_for_exposure(cmp_cfg, exposure)
        path = _stack_path(out_dir, exposure, exp_cfg)
        if path.exists() and not force:
            stacked = pd.read_parquet(path)
            out[exposure] = stacked
            continue
        exposure_col = str(meta["col"])
        events = detect_largest_jump_events(
            panel,
            exposure_col=exposure_col,
            cohort_min_year=int(cmp_cfg["stacked_jump_min_year"]),
            cohort_max_year=int(cmp_cfg["stacked_jump_max_year"]),
            min_jump=float(cmp_cfg["stacked_min_event_jump"]),
            treated_jump_percentile=float(cmp_cfg["stacked_treated_jump_percentile"]),
            control_min_jump_percentile=float(cmp_cfg["stacked_control_min_jump_percentile"]),
        )
        stacked = build_comparison_stacked_panel(
            panel,
            events,
            exp_cfg,
            matching_style="matched",
            exposure_col=exposure_col,
        )
        stacked.to_parquet(path, index=False)
        events.to_csv(out_dir / "tables" / f"event_detection_{exposure}.csv", index=False)
        out[exposure] = stacked
    return out


def _variant_cfg(cmp_cfg: dict, *, first_stage_type: str, first_stage_col: str, outcomes: list[str], ihs: bool) -> dict:
    cfg = dict(cmp_cfg)
    cfg["first_stage_type"] = first_stage_type
    cfg["first_stage_col"] = first_stage_col
    cfg["outcome_cols"] = outcomes
    cfg["count_outcome_transform"] = "ihs" if ihs else "none"
    cfg["use_log_outcome"] = False
    cfg["verbose"] = False
    return cfg


def _estimate_variant(
    stacks: dict[str, pd.DataFrame],
    cmp_cfg: dict,
    *,
    variant: str,
    first_stage_type: str = "ols_continuous",
    first_stage_col: str = "any_opt_hires_correction_aware",
    outcomes: list[str] | None = None,
    ihs: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cfg = _variant_cfg(
        cmp_cfg,
        first_stage_type=first_stage_type,
        first_stage_col=first_stage_col,
        outcomes=list(outcomes or []),
        ihs=ihs,
    )
    for exposure, stacked in stacks.items():
        spec = f"matched_{exposure}"
        for row in estimate_stacked_event_coefficients(stacked, "sun_abraham", spec, cfg):
            row["exposure"] = exposure
            row["variant"] = variant
            rows.append(row)
    return pd.DataFrame(rows)


def _estimate_zeroth_stage_did(stacks: dict[str, pd.DataFrame], cmp_cfg: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exposure, stacked in stacks.items():
        exposure_col = str(EXPOSURES[exposure]["col"])
        cfg = _variant_cfg(
            cmp_cfg,
            first_stage_type="ols_continuous",
            first_stage_col="any_opt_hires_correction_aware",
            outcomes=[exposure_col],
            ihs=False,
        )
        spec = f"matched_{exposure}"
        for row in estimate_stacked_event_coefficients(stacked, "sun_abraham", spec, cfg):
            row["exposure"] = exposure
            row["variant"] = "zeroth_stage_did"
            rows.append(row)
    return pd.DataFrame(rows)


def _transform_for_plot(values: pd.Series, *, ihs: bool = False) -> pd.Series:
    vals = _safe_num(values)
    return np.arcsinh(vals) if ihs else vals


def _plot_event_time(values: pd.Series) -> pd.Series:
    return _safe_num(values).astype(float) + EVENT_TIME_LABEL_SHIFT


def _baseline_mean(stacked: pd.DataFrame, outcome_col: str, *, ihs: bool = False) -> float:
    rel = _plot_event_time(stacked["rel_time"])
    treated = _safe_num(stacked["treated"]).fillna(0).eq(1)
    values = _transform_for_plot(stacked[outcome_col], ihs=ihs)
    keep = treated & rel.lt(0) & values.notna()
    if not keep.any():
        return float("nan")
    return float(values.loc[keep].mean())


def _pooled_effect(coefs: pd.DataFrame, *, exposure: str = POOLED_EXPOSURE, min_event: int = 0, max_event: int = 4) -> tuple[float, float]:
    sub = coefs.loc[
        coefs["exposure"].eq(exposure)
        & coefs["rel_time"].between(min_event, max_event)
        & coefs["coef"].notna()
    ].copy()
    if sub.empty:
        return float("nan"), float("nan")
    coef = float(_safe_num(sub["coef"]).mean())
    se_vals = _safe_num(sub["se"]).fillna(0.0).to_numpy(dtype=float)
    se = float(np.sqrt(np.square(se_vals).sum()) / len(se_vals))
    return coef, se


def _summary_box(
    *,
    stacked: pd.DataFrame,
    coefs: pd.DataFrame,
    outcome_col: str,
    ihs: bool,
    exposure: str = POOLED_EXPOSURE,
) -> str:
    baseline = _baseline_mean(stacked, outcome_col, ihs=ihs)
    pooled, pooled_se = _pooled_effect(coefs, exposure=exposure)
    effect_size = 100.0 * pooled / baseline if math.isfinite(baseline) and not np.isclose(baseline, 0.0) else float("nan")
    return (
        f"Baseline mean: {_fmt(baseline)}\n"
        f"Pooled effect: {_fmt_signed(pooled)} ({_fmt(pooled_se)})\n"
        f"Effect size: {_fmt_signed(effect_size, 1)}%"
    )


def _savefig(fig: plt.Figure, path: Path, *, legend_bottom: float = 0.16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, legend_bottom, 1, 1))
    fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def _bottom_legend(fig: plt.Figure, handles: list, labels: list[str], *, ncol: int) -> None:
    if not SHOW_LEGENDS:
        return
    pairs = [(h, l) for h, l in zip(handles, labels) if l and not str(l).startswith("_")]
    if not pairs:
        return
    seen: set[str] = set()
    deduped = []
    for handle, label in pairs:
        if str(label) in seen:
            continue
        seen.add(str(label))
        deduped.append((handle, str(label)))
    handles_out, labels_out = zip(*deduped)
    fig.legend(
        handles_out,
        labels_out,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        frameon=False,
        fontsize=llstyle.LEGEND_FONT_SIZE,
    )


def _draw_coef_panel(
    ax: plt.Axes,
    coefs: pd.DataFrame,
    *,
    stacks: dict[str, pd.DataFrame],
    outcome_col: str,
    y_label: str,
    ihs: bool = False,
    show_ylabel: bool = True,
    marker_size: float = COEF_MARKER_SIZE,
    exposure_keys: list[str] | None = None,
) -> None:
    exposure_keys = list(exposure_keys or EXPOSURES.keys())
    offsets = dict(zip(exposure_keys, llstyle.offsets(len(exposure_keys), span=0.22)))
    summary_exposure = exposure_keys[0] if exposure_keys else POOLED_EXPOSURE
    for exposure in exposure_keys:
        meta = EXPOSURES[exposure]
        sub = coefs.loc[coefs["exposure"].eq(exposure)].sort_values("rel_time")
        if sub.empty:
            continue
        ax.errorbar(
            _plot_event_time(sub["rel_time"]) + offsets.get(exposure, 0.0),
            _safe_num(sub["coef"]),
            yerr=1.96 * _safe_num(sub["se"]).fillna(0.0),
            marker=str(meta["marker"]),
            linewidth=1.6,
            capsize=0,
            markersize=marker_size,
            color=str(meta["color"]),
            ecolor=llstyle.rgba(str(meta["color"]), 0.35),
            elinewidth=marker_size,
            label=str(meta["label"]),
        )
    ax.axhline(0, color="0.25", linestyle=":", linewidth=1.0)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Event time")
    if show_ylabel:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")
    ax.text(
        0.03,
        0.97,
        _summary_box(stacked=stacks[summary_exposure], coefs=coefs, outcome_col=outcome_col, ihs=ihs, exposure=summary_exposure),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.94},
    )


def _coef_subset(df: pd.DataFrame, *, variant: str, family: str, outcome_col: str) -> pd.DataFrame:
    return df.loc[
        df["variant"].eq(variant)
        & df["family"].eq(family)
        & df["outcome_col"].eq(outcome_col)
    ].copy()


def _save_single_coef(
    df: pd.DataFrame,
    stacks: dict[str, pd.DataFrame],
    out_path: Path,
    *,
    variant: str,
    family: str,
    outcome_col: str,
    y_label: str,
    ihs: bool = False,
    exposure_keys: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    sub = _coef_subset(df, variant=variant, family=family, outcome_col=outcome_col)
    _draw_coef_panel(ax, sub, stacks=stacks, outcome_col=outcome_col, y_label=y_label, ihs=ihs, exposure_keys=exposure_keys)
    handles, labels = ax.get_legend_handles_labels()
    _bottom_legend(fig, handles, labels, ncol=max(1, len(exposure_keys or EXPOSURES)))
    _savefig(fig, out_path)


def _save_two_panel_coef(
    df: pd.DataFrame,
    stacks: dict[str, pd.DataFrame],
    out_path: Path,
    *,
    variant: str,
    outcomes: list[tuple[str, str]],
    y_label: str,
    ihs: bool = False,
    exposure_keys: list[str] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=llstyle.PANEL_FIGSIZE, sharey=False)
    handles = []
    labels = []
    for idx, (ax, (outcome_col, panel_label)) in enumerate(zip(axes, outcomes)):
        sub = _coef_subset(df, variant=variant, family="reduced_form", outcome_col=outcome_col)
        _draw_coef_panel(
            ax,
            sub,
            stacks=stacks,
            outcome_col=outcome_col,
            y_label=y_label,
            ihs=ihs,
            show_ylabel=idx == 0,
            marker_size=TWO_PANEL_MARKER_SIZE,
            exposure_keys=exposure_keys,
        )
        ax.text(
            0.5,
            1.02,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=TITLE_FONT_SIZE,
            fontweight="semibold",
        )
        h, l = ax.get_legend_handles_labels()
        handles, labels = h, l
    if handles:
        _bottom_legend(fig, handles, labels, ncol=max(1, len(exposure_keys or EXPOSURES)))
    _savefig(fig, out_path)


def _raw_means(
    stacked: pd.DataFrame,
    outcome_col: str,
    *,
    ihs: bool = False,
) -> pd.DataFrame:
    work = stacked[["rel_time", "treated", outcome_col]].copy()
    work["_y"] = _transform_for_plot(work[outcome_col], ihs=ihs)
    return (
        work.dropna(subset=["_y", "rel_time", "treated"])
        .groupby(["rel_time", "treated"], as_index=False)["_y"]
        .mean()
        .rename(columns={"_y": "mean"})
    )


def _draw_raw_panel(
    ax: plt.Axes,
    stacks: dict[str, pd.DataFrame],
    *,
    outcome_col: str,
    y_label: str,
    ihs: bool = False,
    show_ylabel: bool = True,
    marker_size: float = COEF_MARKER_SIZE,
    exposure_keys: list[str] | None = None,
) -> None:
    exposure_keys = list(exposure_keys or EXPOSURES.keys())
    offsets = dict(zip(exposure_keys, llstyle.offsets(len(exposure_keys), span=0.20)))
    for exposure in exposure_keys:
        meta = EXPOSURES[exposure]
        raw = _raw_means(stacks[exposure], outcome_col, ihs=ihs)
        if raw.empty:
            continue
        for treated_value, role_label, linestyle, marker, alpha in (
            (1, "Treated", "-", str(meta["marker"]), 1.0),
            (0, "Control", ":", str(meta["marker"]), 0.9),
        ):
            sub = raw.loc[_safe_num(raw["treated"]).eq(treated_value)].sort_values("rel_time")
            if sub.empty:
                continue
            ax.plot(
                _plot_event_time(sub["rel_time"]) + offsets.get(exposure, 0.0),
                _safe_num(sub["mean"]),
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=marker_size,
                color=str(meta["color"]),
                alpha=alpha,
                label=f"{role_label}: {meta['label']}",
            )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Event time")
    ax.set_ylabel(y_label if show_ylabel else "")


def _save_single_raw(
    stacks: dict[str, pd.DataFrame],
    out_path: Path,
    *,
    outcome_col: str,
    y_label: str,
    ihs: bool = False,
    exposure_keys: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    _draw_raw_panel(ax, stacks, outcome_col=outcome_col, y_label=y_label, ihs=ihs, exposure_keys=exposure_keys)
    handles, labels = ax.get_legend_handles_labels()
    _bottom_legend(fig, handles, labels, ncol=max(1, 2 * len(exposure_keys or EXPOSURES)))
    _savefig(fig, out_path, legend_bottom=0.20)


def _save_two_panel_raw(
    stacks: dict[str, pd.DataFrame],
    out_path: Path,
    *,
    outcomes: list[tuple[str, str]],
    y_label: str,
    ihs: bool = False,
    exposure_keys: list[str] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=llstyle.PANEL_FIGSIZE, sharey=False)
    handles = []
    labels = []
    for idx, (ax, (outcome_col, panel_label)) in enumerate(zip(axes, outcomes)):
        _draw_raw_panel(
            ax,
            stacks,
            outcome_col=outcome_col,
            y_label=y_label,
            ihs=ihs,
            show_ylabel=idx == 0,
            marker_size=TWO_PANEL_MARKER_SIZE,
            exposure_keys=exposure_keys,
        )
        ax.text(
            0.5,
            1.02,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=TITLE_FONT_SIZE,
            fontweight="semibold",
        )
        h, l = ax.get_legend_handles_labels()
        handles, labels = h, l
    if handles:
        _bottom_legend(fig, handles, labels, ncol=max(1, 2 * len(exposure_keys or EXPOSURES)))
    _savefig(fig, out_path, legend_bottom=0.20)


def _save_treated_event_histogram(out_dir: Path, out_path: Path, *, exposure_keys: list[str] | None = None) -> None:
    rows: list[pd.DataFrame] = []
    exposure_keys = list(exposure_keys or EXPOSURES.keys())
    for exposure in exposure_keys:
        meta = EXPOSURES[exposure]
        path = out_dir / "tables" / f"event_detection_{exposure}.csv"
        if not path.exists():
            continue
        events = pd.read_csv(path)
        if "treated" not in events.columns or "g" not in events.columns:
            continue
        treated = events.loc[_safe_num(events["treated"]).fillna(0).eq(1), ["g"]].copy()
        treated["g"] = _safe_num(treated["g"])
        treated = treated.dropna(subset=["g"])
        if treated.empty:
            continue
        treated["event_year"] = treated["g"].astype(int)
        treated["exposure"] = exposure
        treated["label"] = str(meta["label"])
        rows.append(treated[["event_year", "exposure", "label"]])
    if not rows:
        return

    events = pd.concat(rows, ignore_index=True)
    years = list(range(int(events["event_year"].min()), int(events["event_year"].max()) + 1))
    x = np.arange(len(years), dtype=float)
    width = 0.34
    offsets = dict(zip(exposure_keys, llstyle.offsets(len(exposure_keys), span=width)))

    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    handles = []
    labels = []
    for exposure in exposure_keys:
        meta = EXPOSURES[exposure]
        counts = (
            events.loc[events["exposure"].eq(exposure)]
            .groupby("event_year")
            .size()
            .reindex(years, fill_value=0)
        )
        bars = ax.bar(
            x + offsets.get(exposure, 0.0),
            counts.to_numpy(dtype=float),
            width=width,
            color=str(meta["color"]),
            alpha=0.88,
            label=str(meta["label"]),
        )
        handles.append(bars)
        labels.append(str(meta["label"]))
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in years])
    ax.set_xlabel("Largest-jump year")
    ax.set_ylabel("Treated firms")
    ax.set_title("Treated Events by Exposure Timing", fontsize=TITLE_FONT_SIZE, fontweight="semibold")
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    _bottom_legend(fig, handles, labels, ncol=2)
    _savefig(fig, out_path, legend_bottom=0.15)


def _save_zeroth_stage(stacks: dict[str, pd.DataFrame], out_path: Path, *, exposure_keys: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    exposure_keys = list(exposure_keys or EXPOSURES.keys())
    offsets = dict(zip(exposure_keys, llstyle.offsets(len(exposure_keys), span=0.20)))
    handles = []
    labels = []
    for exposure in exposure_keys:
        meta = EXPOSURES[exposure]
        stacked = stacks.get(exposure)
        exposure_col = str(meta["col"])
        if stacked is None or exposure_col not in stacked.columns:
            continue
        raw = _raw_means(stacked, exposure_col)
        if raw.empty:
            continue
        for treated_value, role_label, linestyle, alpha in (
            (1, "Treated", "-", 1.0),
            (0, "Control", ":", 0.9),
        ):
            sub = raw.loc[_safe_num(raw["treated"]).eq(treated_value)].sort_values("rel_time")
            if sub.empty:
                continue
            line = ax.plot(
                _plot_event_time(sub["rel_time"]) + offsets.get(exposure, 0.0),
                _safe_num(sub["mean"]),
                marker=str(meta["marker"]),
                linestyle=linestyle,
                linewidth=1.8,
                markersize=COEF_MARKER_SIZE,
                color=str(meta["color"]),
                alpha=alpha,
                label=f"{role_label}: {meta['label']}",
            )[0]
            handles.append(line)
            labels.append(f"{role_label}: {meta['label']}")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Event time")
    ax.set_ylabel("Raw mean exposure")
    ax.set_title("Zeroth Stage: Exposure in Event Time", fontsize=TITLE_FONT_SIZE, fontweight="semibold")
    _bottom_legend(fig, handles, labels, ncol=2)
    _savefig(fig, out_path, legend_bottom=0.20)


def build_assets(
    *,
    config_path: Path = DEFAULT_CONFIG,
    panel_path: Path = DEFAULT_PANEL,
    out_dir: Path = DEFAULT_OUT_DIR,
    force_stacks: bool = False,
    force_origin_tenure: bool = False,
) -> None:
    llstyle.apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("figures").mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("tables").mkdir(parents=True, exist_ok=True)
    cmp_cfg = _read_cfg(config_path)
    panel = _load_panel(panel_path, out_dir, cmp_cfg, force_origin_tenure=force_origin_tenure)
    stacks = _build_or_load_stacks(panel, out_dir, cmp_cfg, force=force_stacks)
    _save_treated_event_histogram(out_dir, out_dir / "figures" / "treated_events_by_year.png", exposure_keys=MAIN_EXPOSURE_KEYS)
    _save_treated_event_histogram(out_dir, out_dir / "figures" / "treated_events_by_year_loose_high_opt.png", exposure_keys=LOOSE_EXPOSURE_KEYS)
    _save_zeroth_stage(stacks, out_dir / "figures" / "zeroth_stage_exposure_raw_means.png", exposure_keys=MAIN_EXPOSURE_KEYS)
    _save_zeroth_stage(stacks, out_dir / "figures" / "zeroth_stage_exposure_raw_means_loose_high_opt.png", exposure_keys=LOOSE_EXPOSURE_KEYS)

    variants = [
        ("first_stage_levels", {"outcomes": [], "first_stage_type": "ols_continuous", "first_stage_col": "any_opt_hires_correction_aware", "ihs": False}),
        ("first_stage_ihs", {"outcomes": [], "first_stage_type": "ols_ihs", "first_stage_col": "any_opt_hires_correction_aware", "ihs": False}),
        ("first_stage_masters", {"outcomes": [], "first_stage_type": "ols_continuous", "first_stage_col": "masters_opt_hires_correction_aware", "ihs": False}),
        ("employment_ihs", {"outcomes": ["y_cst_lag0"], "ihs": True}),
        ("newgrad_levels", {"outcomes": ["y_new_hires_foreign_lag0", "y_new_hires_native_lag0"], "ihs": False}),
        ("newgrad_ihs", {"outcomes": ["y_new_hires_foreign_lag0", "y_new_hires_native_lag0"], "ihs": True}),
        (
            "tenure_opt_likely_origin",
            {
                "outcomes": [
                    "avg_tenure_opt_likely_foreign_active_lag0",
                    "avg_tenure_opt_likely_native_active_lag0",
                ],
                "ihs": False,
            },
        ),
        (
            "tenure_all_origin",
            {"outcomes": ["avg_tenure_foreign_active_lag0", "avg_tenure_native_active_lag0"], "ihs": False},
        ),
        (
            "intern_levels",
            {
                "outcomes": ["y_intern_users_opt_likely_foreign_lag0", "y_intern_users_opt_likely_native_lag0"],
                "ihs": False,
            },
        ),
        (
            "intern_ihs",
            {
                "outcomes": ["y_intern_users_opt_likely_foreign_lag0", "y_intern_users_opt_likely_native_lag0"],
                "ihs": True,
            },
        ),
    ]
    estimates = []
    estimates.append(_estimate_zeroth_stage_did(stacks, cmp_cfg))
    for variant, kwargs in variants:
        estimates.append(_estimate_variant(stacks, cmp_cfg, variant=variant, **kwargs))
    all_estimates = pd.concat(estimates, ignore_index=True, sort=False)
    all_estimates.to_csv(out_dir / "tables" / "design3_slide_coefficients.csv", index=False)

    fig_dir = out_dir / "figures"
    _save_single_coef(
        all_estimates,
        stacks,
        fig_dir / "zeroth_stage_did.png",
        variant="zeroth_stage_did",
        family="reduced_form",
        outcome_col=str(EXPOSURES[STRICT_EXPOSURE]["col"]),
        y_label="Coefficient, exposure",
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )
    _save_single_coef(
        all_estimates,
        stacks,
        fig_dir / "first_stage_levels.png",
        variant="first_stage_levels",
        family="first_stage",
        outcome_col="any_opt_hires_correction_aware",
        y_label="Coefficient, OPT hires",
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )
    _save_single_coef(
        all_estimates,
        stacks,
        fig_dir / "first_stage_ihs.png",
        variant="first_stage_ihs",
        family="first_stage",
        outcome_col="any_opt_hires_correction_aware",
        y_label="Coefficient, IHS(OPT hires)",
        ihs=True,
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )
    _save_single_coef(
        all_estimates,
        stacks,
        fig_dir / "first_stage_masters_levels.png",
        variant="first_stage_masters",
        family="first_stage",
        outcome_col="masters_opt_hires_correction_aware",
        y_label="Coefficient, master's OPT hires",
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )
    _save_single_coef(
        all_estimates,
        stacks,
        fig_dir / "employment_ihs.png",
        variant="employment_ihs",
        family="reduced_form",
        outcome_col="y_cst_lag0",
        y_label="Coefficient, IHS(total employment)",
        ihs=True,
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )
    _save_single_raw(stacks, fig_dir / "first_stage_raw_means.png", outcome_col="any_opt_hires_correction_aware", y_label="Raw mean", exposure_keys=MAIN_EXPOSURE_KEYS)

    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "newgrad_hires_levels.png",
        variant="newgrad_levels",
        outcomes=[
            ("y_new_hires_foreign_lag0", "Foreign new-grad hires"),
            ("y_new_hires_native_lag0", "Native new-grad hires"),
        ],
        y_label="Coefficient, level",
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )
    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "newgrad_hires_ihs.png",
        variant="newgrad_ihs",
        outcomes=[
            ("y_new_hires_foreign_lag0", "IHS(foreign new-grad hires)"),
            ("y_new_hires_native_lag0", "IHS(native new-grad hires)"),
        ],
        y_label="Coefficient, IHS outcome",
        ihs=True,
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )
    _save_two_panel_raw(
        stacks,
        fig_dir / "newgrad_hires_raw_means.png",
        outcomes=[
            ("y_new_hires_foreign_lag0", "Foreign new-grad hires"),
            ("y_new_hires_native_lag0", "Native new-grad hires"),
        ],
        y_label="Raw mean",
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )

    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "tenure_opt_likely_origin.png",
        variant="tenure_opt_likely_origin",
        outcomes=[
            ("avg_tenure_opt_likely_foreign_active_lag0", "Avg Tenure: Foreign Employees"),
            ("avg_tenure_opt_likely_native_active_lag0", "Avg Tenure: Native Employees"),
        ],
        y_label="Coefficient, years",
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )
    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "tenure_all_origin.png",
        variant="tenure_all_origin",
        outcomes=[
            ("avg_tenure_foreign_active_lag0", "Avg Tenure: Foreign Employees"),
            ("avg_tenure_native_active_lag0", "Avg Tenure: Native Employees"),
        ],
        y_label="Coefficient, years",
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )
    _save_two_panel_raw(
        stacks,
        fig_dir / "tenure_opt_likely_origin_raw_means.png",
        outcomes=[
            ("avg_tenure_opt_likely_foreign_active_lag0", "Avg Tenure: Foreign Employees"),
            ("avg_tenure_opt_likely_native_active_lag0", "Avg Tenure: Native Employees"),
        ],
        y_label="Raw mean",
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )

    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "interns_opt_likely_levels.png",
        variant="intern_levels",
        outcomes=[
            ("y_intern_users_opt_likely_foreign_lag0", "Foreign interns"),
            ("y_intern_users_opt_likely_native_lag0", "Native interns"),
        ],
        y_label="Coefficient, users",
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )
    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "interns_opt_likely_ihs.png",
        variant="intern_ihs",
        outcomes=[
            ("y_intern_users_opt_likely_foreign_lag0", "IHS(foreign interns)"),
            ("y_intern_users_opt_likely_native_lag0", "IHS(native interns)"),
        ],
        y_label="Coefficient, IHS outcome",
        ihs=True,
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )
    _save_two_panel_raw(
        stacks,
        fig_dir / "interns_opt_likely_raw_means.png",
        outcomes=[
            ("y_intern_users_opt_likely_foreign_lag0", "Foreign interns"),
            ("y_intern_users_opt_likely_native_lag0", "Native interns"),
        ],
        y_label="Raw mean",
        exposure_keys=MAIN_EXPOSURE_KEYS,
    )

    _save_single_coef(
        all_estimates,
        stacks,
        fig_dir / "first_stage_levels_loose_high_opt.png",
        variant="first_stage_levels",
        family="first_stage",
        outcome_col="any_opt_hires_correction_aware",
        y_label="Coefficient, OPT hires",
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_single_coef(
        all_estimates,
        stacks,
        fig_dir / "zeroth_stage_did_loose_high_opt.png",
        variant="zeroth_stage_did",
        family="reduced_form",
        outcome_col=str(EXPOSURES[LOOSE_EXPOSURE]["col"]),
        y_label="Coefficient, exposure",
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_single_coef(
        all_estimates,
        stacks,
        fig_dir / "first_stage_ihs_loose_high_opt.png",
        variant="first_stage_ihs",
        family="first_stage",
        outcome_col="any_opt_hires_correction_aware",
        y_label="Coefficient, IHS(OPT hires)",
        ihs=True,
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_single_coef(
        all_estimates,
        stacks,
        fig_dir / "first_stage_masters_levels_loose_high_opt.png",
        variant="first_stage_masters",
        family="first_stage",
        outcome_col="masters_opt_hires_correction_aware",
        y_label="Coefficient, master's OPT hires",
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_single_coef(
        all_estimates,
        stacks,
        fig_dir / "employment_ihs_loose_high_opt.png",
        variant="employment_ihs",
        family="reduced_form",
        outcome_col="y_cst_lag0",
        y_label="Coefficient, IHS(total employment)",
        ihs=True,
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_single_raw(
        stacks,
        fig_dir / "first_stage_raw_means_loose_high_opt.png",
        outcome_col="any_opt_hires_correction_aware",
        y_label="Raw mean",
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "newgrad_hires_levels_loose_high_opt.png",
        variant="newgrad_levels",
        outcomes=[
            ("y_new_hires_foreign_lag0", "Foreign new-grad hires"),
            ("y_new_hires_native_lag0", "Native new-grad hires"),
        ],
        y_label="Coefficient, level",
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "newgrad_hires_ihs_loose_high_opt.png",
        variant="newgrad_ihs",
        outcomes=[
            ("y_new_hires_foreign_lag0", "IHS(foreign new-grad hires)"),
            ("y_new_hires_native_lag0", "IHS(native new-grad hires)"),
        ],
        y_label="Coefficient, IHS outcome",
        ihs=True,
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_two_panel_raw(
        stacks,
        fig_dir / "newgrad_hires_raw_means_loose_high_opt.png",
        outcomes=[
            ("y_new_hires_foreign_lag0", "Foreign new-grad hires"),
            ("y_new_hires_native_lag0", "Native new-grad hires"),
        ],
        y_label="Raw mean",
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "tenure_opt_likely_origin_loose_high_opt.png",
        variant="tenure_opt_likely_origin",
        outcomes=[
            ("avg_tenure_opt_likely_foreign_active_lag0", "Avg Tenure: Foreign Employees"),
            ("avg_tenure_opt_likely_native_active_lag0", "Avg Tenure: Native Employees"),
        ],
        y_label="Coefficient, years",
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "tenure_all_origin_loose_high_opt.png",
        variant="tenure_all_origin",
        outcomes=[
            ("avg_tenure_foreign_active_lag0", "Avg Tenure: Foreign Employees"),
            ("avg_tenure_native_active_lag0", "Avg Tenure: Native Employees"),
        ],
        y_label="Coefficient, years",
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_two_panel_raw(
        stacks,
        fig_dir / "tenure_opt_likely_origin_raw_means_loose_high_opt.png",
        outcomes=[
            ("avg_tenure_opt_likely_foreign_active_lag0", "Avg Tenure: Foreign Employees"),
            ("avg_tenure_opt_likely_native_active_lag0", "Avg Tenure: Native Employees"),
        ],
        y_label="Raw mean",
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "interns_opt_likely_levels_loose_high_opt.png",
        variant="intern_levels",
        outcomes=[
            ("y_intern_users_opt_likely_foreign_lag0", "Foreign interns"),
            ("y_intern_users_opt_likely_native_lag0", "Native interns"),
        ],
        y_label="Coefficient, users",
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_two_panel_coef(
        all_estimates,
        stacks,
        fig_dir / "interns_opt_likely_ihs_loose_high_opt.png",
        variant="intern_ihs",
        outcomes=[
            ("y_intern_users_opt_likely_foreign_lag0", "IHS(foreign interns)"),
            ("y_intern_users_opt_likely_native_lag0", "IHS(native interns)"),
        ],
        y_label="Coefficient, IHS outcome",
        ihs=True,
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    _save_two_panel_raw(
        stacks,
        fig_dir / "interns_opt_likely_raw_means_loose_high_opt.png",
        outcomes=[
            ("y_intern_users_opt_likely_foreign_lag0", "Foreign interns"),
            ("y_intern_users_opt_likely_native_lag0", "Native interns"),
        ],
        y_label="Raw mean",
        exposure_keys=LOOSE_EXPOSURE_KEYS,
    )
    print(f"[laborlunch-design3] wrote slide assets to {out_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force-stacks", action="store_true")
    parser.add_argument("--force-origin-tenure", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    build_assets(
        config_path=args.config,
        panel_path=args.panel,
        out_dir=args.out_dir,
        force_stacks=args.force_stacks,
        force_origin_tenure=args.force_origin_tenure,
    )


if __name__ == "__main__":
    main()
