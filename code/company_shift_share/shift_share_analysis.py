"""
Company-level shift-share pipeline and regressions — April 2026.

Builds the analysis panel (z_ct, x_ct, y lags), runs diagnostics, and runs
simple first-stage / reduced-form regressions with and without year × company FEs.

All parameters are configured via configs/company_shift_share_apr2026.yaml.
Runnable interactively: %run company_shift_share/shift_share_analysis.py
"""
from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

try:
    from company_shift_share.config_loader import get_cfg_section, load_config
    from company_shift_share.institution_mapping import load_revelio_school_map, sql_normalize_school_key
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import get_cfg_section, load_config
    from company_shift_share.institution_mapping import load_revelio_school_map, sql_normalize_school_key

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "company_shift_share_apr2026.yaml"

# Exposed connection + cached outputs when the script is run interactively.
DUCKDB_CONN: Optional[ddb.DuckDBPyConnection] = None
ANALYSIS_PANEL_DF = None
ERRORBAR_INTERVAL_ALPHA = 0.25


def get_connection() -> Optional[ddb.DuckDBPyConnection]:
    """Return the active DuckDB connection from the last completed main() run."""
    return DUCKDB_CONN


def get_analysis_panel_df() -> Optional[pd.DataFrame]:
    """Return the analysis_panel DataFrame from the last completed main() run."""
    return ANALYSIS_PANEL_DF


def _soften_errorbar_interval(errorbar_container, alpha: float = ERRORBAR_INTERVAL_ALPHA) -> None:
    """Apply alpha to errorbar intervals/caps without fading the coefficient line."""
    if errorbar_container is None or not hasattr(errorbar_container, "lines"):
        return
    for artist_group in errorbar_container.lines[1:]:
        if artist_group is None:
            continue
        artists = artist_group if isinstance(artist_group, (tuple, list)) else [artist_group]
        for artist in artists:
            if hasattr(artist, "set_alpha"):
                artist.set_alpha(alpha)


# =============================================================================
# ── UTILITY HELPERS ───────────────────────────────────────────────────────────
# =============================================================================

def _escape(path) -> str:
    return str(path).replace("'", "''")

def _sql_in_list(values: list) -> str:
    if not values:
        return "()"
    return "(" + ",".join(f"'{str(v).replace(chr(39), chr(39)+chr(39))}'" for v in values) + ")"

def _ensure_out_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _write_view(con: ddb.DuckDBPyConnection, view: str, path: Path) -> None:
    _ensure_out_dir(path)
    con.sql(f"COPY (SELECT * FROM {view}) TO '{_escape(path)}' (FORMAT PARQUET)")
    print(f"  → wrote {path}")


def _load_saved_panel_views(
    con: ddb.DuckDBPyConnection,
    paths_cfg: dict,
    school_sample_mode: str,
    growth_window_start: int = 2014,
    growth_window_end: int = 2017,
) -> None:
    gk_start = int(growth_window_start)
    gk_end = int(growth_window_end)

    required = {
        "analysis_panel": paths_cfg.get("analysis_panel"),
        "instrument_components": paths_cfg.get("instrument_components"),
        "school_shift_metric_panel": paths_cfg.get("school_shift_metric_panel"),
        "school_shift_sample": paths_cfg.get("school_shift_sample"),
    }
    _check_paths({k: v for k, v in required.items() if v})

    con.sql(
        f"CREATE OR REPLACE TEMP VIEW analysis_panel AS "
        f"SELECT * FROM read_parquet('{_escape(required['analysis_panel'])}')"
    )
    con.sql(
        f"CREATE OR REPLACE TEMP VIEW instrument_components AS "
        f"SELECT * FROM read_parquet('{_escape(required['instrument_components'])}')"
    )

    # Optional saved views. If missing, rebuild from instrument components.
    path_transition_shares = paths_cfg.get("transition_shares")
    if path_transition_shares and Path(path_transition_shares).exists():
        con.sql(
            f"CREATE OR REPLACE TEMP VIEW transition_shares AS "
            f"SELECT * FROM read_parquet('{_escape(path_transition_shares)}')"
        )
        print("[reload] loaded transition_shares from disk")
    else:
        con.sql("""
            CREATE OR REPLACE TEMP VIEW transition_shares AS
            SELECT c, k, MAX(share_ck) AS share_ck
            FROM instrument_components
            GROUP BY c, k
        """)
        print("[reload] reconstructed transition_shares from instrument_components")

    path_ipeds_growth = paths_cfg.get("ipeds_unit_growth")
    if path_ipeds_growth and Path(path_ipeds_growth).exists():
        con.sql(
            f"CREATE OR REPLACE TEMP VIEW ipeds_unit_growth AS "
            f"SELECT * FROM read_parquet('{_escape(path_ipeds_growth)}')"
        )
        print("[reload] loaded ipeds_unit_growth from disk")
    else:
        con.sql(f"""
            CREATE OR REPLACE TEMP VIEW ipeds_unit_growth AS
            SELECT DISTINCT CAST(k AS VARCHAR) AS k, CAST(t AS INTEGER) AS t,
                CASE WHEN CAST(t AS INTEGER) BETWEEN {gk_start} AND {gk_end} THEN CAST(g_kt AS DOUBLE) ELSE 0 END AS g_kt
            FROM instrument_components
            WHERE g_kt IS NOT NULL
        """)
        print("[reload] reconstructed ipeds_unit_growth from instrument_components")

    path_instrument_panel = paths_cfg.get("instrument_panel")
    if path_instrument_panel and Path(path_instrument_panel).exists():
        con.sql(
            f"CREATE OR REPLACE TEMP VIEW instrument_panel AS "
            f"SELECT * FROM read_parquet('{_escape(path_instrument_panel)}')"
        )
        print("[reload] loaded instrument_panel from disk")
    else:
        con.sql("""
            CREATE OR REPLACE TEMP VIEW instrument_panel AS
            SELECT c, t,
                SUM(CASE WHEN share_ck > 1 THEN NULL ELSE z_ct_component END) AS z_ct,
                COUNT(DISTINCT CASE WHEN share_ck IS NOT NULL AND share_ck > 0
                                     AND g_kt IS NOT NULL AND g_kt != 0 THEN k END) AS n_universities
            FROM instrument_components
            GROUP BY c, t
        """)
        print("[reload] reconstructed instrument_panel from instrument_components")

    con.sql(
        f"CREATE OR REPLACE TEMP VIEW school_shift_metric_panel AS "
        f"SELECT * FROM read_parquet('{_escape(required['school_shift_metric_panel'])}')"
    )
    print("[reload] loaded school_shift_metric_panel from disk")
    con.sql(
        f"CREATE OR REPLACE TEMP VIEW school_shift_sample AS "
        f"SELECT * FROM read_parquet('{_escape(required['school_shift_sample'])}')"
    )
    print("[reload] loaded school_shift_sample from disk")

    n_rows = con.sql("SELECT COUNT(*) FROM analysis_panel").fetchone()[0]
    n_cos = con.sql("SELECT COUNT(DISTINCT c) FROM analysis_panel").fetchone()[0]
    n_pairs = con.sql("SELECT COUNT(*) FROM transition_shares").fetchone()[0]
    print(f"[reload] analysis_panel: {n_rows:,} rows | {n_cos:,} companies")
    print(f"[reload] transition_shares: {n_pairs:,} company-school pairs")

def _has_column(con: ddb.DuckDBPyConnection, view: str, column: str) -> bool:
    try:
        rows = con.execute(f"PRAGMA table_info('{view}')").fetchall()
    except Exception:
        return False
    return column.lower() in {r[1].lower() for r in rows}


def _diagnostic_school_name_lookup(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    """Collect a best-effort school-name lookup from available diagnostic views."""
    sources: list[str] = []
    for view in ("school_shift_metric_panel", "school_shift_sample", "ipeds_unit_growth"):
        if _has_column(con, view, "k") and _has_column(con, view, "school_name"):
            sources.append(
                f"SELECT CAST(k AS VARCHAR) AS k, "
                f"NULLIF(TRIM(CAST(school_name AS VARCHAR)), '') AS school_name "
                f"FROM {view}"
            )
    if not sources:
        return pd.DataFrame(columns=["k", "school_name"])
    union_sql = " UNION ALL ".join(sources)
    lookup = con.sql(f"""
        SELECT k, MAX(school_name) AS school_name
        FROM ({union_sql}) src
        WHERE k IS NOT NULL AND school_name IS NOT NULL
        GROUP BY 1
    """).df()
    if lookup.empty:
        return pd.DataFrame(columns=["k", "school_name"])
    lookup["k"] = lookup["k"].astype(str)
    lookup["school_name"] = lookup["school_name"].fillna("").astype(str)
    lookup = lookup.loc[lookup["school_name"].str.strip() != ""].drop_duplicates("k")
    return lookup[["k", "school_name"]].copy()

def _first_present_column(
    con: ddb.DuckDBPyConnection, view: str, candidates: list[str]
) -> Optional[str]:
    rows = con.execute(f"PRAGMA table_info('{view}')").fetchall()
    cols = {str(r[1]).lower(): str(r[1]) for r in rows}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _parse_unitids(raw) -> list[str]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _safe_path_component(value: str) -> str:
    """Return a filesystem-safe label for variant output directories."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("_") or "outcome"


def _safe_linear_slope(x: pd.Series, y: pd.Series) -> float:
    """Return the slope from a simple linear fit, or NaN if not estimable."""
    xy = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()
    if len(xy) < 2:
        return np.nan
    try:
        return float(np.polyfit(xy["x"], xy["y"], 1)[0])
    except Exception:
        return np.nan


def _groupwise_linear_slopes(
    df: pd.DataFrame,
    group_col: str,
    x_col: str,
    y_cols: list[str],
) -> pd.DataFrame:
    """Compute per-group linear slopes without relying on groupby.apply."""
    rows: list[dict[str, object]] = []
    for group_value, group_df in df.groupby(group_col, sort=True, dropna=False):
        row: dict[str, object] = {group_col: group_value}
        for y_col in y_cols:
            row[f"{y_col}_slope"] = _safe_linear_slope(group_df[x_col], group_df[y_col])
        rows.append(row)
    return pd.DataFrame(rows)


def _select_regression_instrument_cols(reg_cfg: dict, panel_columns: list[str]) -> list[str]:
    """Choose which instrument variants to run in regressions, preserving order."""
    baseline_col = str(reg_cfg.get("instrument_col", "z_ct")).strip() or "z_ct"
    raw_cols = reg_cfg.get("instrument_cols")
    if raw_cols is None:
        candidates = [
            baseline_col,
            "z_ct_raw_flow",
            "z_ct_ihmp_share",
            "z_ct_event_pulse",
            "z_ct_flow_diff",
            "z_ct_flow_ar_resid",
            "z_ct_flow_diff_cumulative",
            "z_ct_flow_ar_resid_cumulative",
            "z_ct_common_base_level",
            "z_ct_common_base_asinh",
            "z_ct_event_step_dose",
            "z_ct_v2_broad_cumulative",
            "z_ct_v3_broad_predicted_opt",
            "z_ct_v4_matched_step",
            "z_ct_v5_matched_pulse",
            "z_ct_v6_broad_composition",
            "z_ct_v7_matched_pulse_growth_rate",
            "z_ct_share_2008_2010",
            "z_ct_share_2011_2013",
            "z_ct_share_2008_2013",
            "z_ct_full",
        ]
    elif isinstance(raw_cols, str):
        candidates = [c.strip() for c in raw_cols.split(",") if c and c.strip()]
    else:
        candidates = [str(c).strip() for c in raw_cols if str(c).strip()]

    out: list[str] = []
    seen: set[str] = set()
    available = set(panel_columns)
    for col in candidates:
        if col in available and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _select_regression_outcome_cols(reg_cfg: dict, panel_columns: list[str]) -> list[str]:
    """Choose reduced-form outcomes to run, preserving config order."""
    baseline_col = str(reg_cfg.get("outcome_col", "y_cst_lag0")).strip() or "y_cst_lag0"
    raw_cols = reg_cfg.get("outcome_cols")
    if raw_cols is None:
        return [baseline_col]
    elif isinstance(raw_cols, str):
        candidates = [c.strip() for c in raw_cols.split(",") if c and c.strip()]
    else:
        candidates = [str(c).strip() for c in raw_cols if str(c).strip()]
    if baseline_col not in candidates:
        candidates.insert(0, baseline_col)

    out: list[str] = []
    seen: set[str] = set()
    available = set(panel_columns)
    for col in candidates:
        if col in seen:
            continue
        seen.add(col)
        if col in available:
            out.append(col)
        else:
            print(f"[regressions] requested outcome '{col}' not found in analysis panel; skipping.")
    return out


def _parse_config_string_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [v.strip() for v in raw.split(",") if v.strip()]
    return [str(v).strip() for v in raw if str(v).strip()]


def _instrument_transform_for_column(reg_cfg: dict, instrument_col: str) -> str:
    """Return the configured regression transform for this instrument column."""
    transform = str(reg_cfg.get("instrument_transform", "none") or "none").strip().lower()
    if transform in {"", "none", "level", "raw"}:
        return "none"
    apply_to_raw = reg_cfg.get("instrument_transform_apply_to")
    if apply_to_raw is not None:
        apply_to = set(_parse_config_string_list(apply_to_raw))
        if instrument_col not in apply_to:
            return "none"
    return transform


def _transformed_instrument_col(instrument_col: str, transform: str) -> str:
    """Name the regression regressor column after applying a transform."""
    if transform in {"", "none", "level", "raw"}:
        return instrument_col
    safe_transform = re.sub(r"[^A-Za-z0-9_]+", "_", transform).strip("_") or "transformed"
    return f"{safe_transform}_{instrument_col}"


def _add_transformed_instrument(
    panel: pd.DataFrame,
    instrument_col: str,
    transform: str,
) -> tuple[pd.DataFrame, str]:
    """Add transformed instrument column and return the DataFrame plus regressor name."""
    if transform in {"", "none", "level", "raw"}:
        return panel, instrument_col
    reg_col = _transformed_instrument_col(instrument_col, transform)
    work = panel.copy()
    values = pd.to_numeric(work[instrument_col], errors="coerce")
    if transform in {"log1p", "ln1p"}:
        work[reg_col] = np.where(values >= -1.0, np.log1p(values), np.nan)
    elif transform in {"log", "ln"}:
        work[reg_col] = np.where(values > 0.0, np.log(values), np.nan)
    elif transform == "asinh":
        work[reg_col] = np.arcsinh(values)
    else:
        raise ValueError(f"Unsupported instrument_transform: {transform!r}")
    return work, reg_col


def _build_absorb_fe_term(fe_cols: list[str]) -> str:
    """Build a pyfixest fixed-effect term from non-empty FE column names."""
    cols = [str(col).strip() for col in fe_cols if str(col).strip()]
    return "| " + " + ".join(cols) if cols else ""


def _instrument_variant_display_name(instrument_col: str) -> str:
    """Human-readable label for regression variant summaries."""
    labels = {
        "z_ct": "Event pre/post level-growth pulse",
        "z_ct_raw_flow": "Raw annual flow",
        "z_ct_ihmp_share": "Raw annual IHMP share",
        "z_ct_event_pulse": "Event pulse",
        "z_ct_flow_diff": "First-difference flow innovation",
        "z_ct_flow_ar_resid": "AR residual flow innovation",
        "z_ct_flow_diff_cumulative": "Cumulative first-difference innovation pool",
        "z_ct_flow_ar_resid_cumulative": "Cumulative AR-residual innovation pool",
        "z_ct_common_base_level": "Common-base level",
        "z_ct_common_base_asinh": "Common-base asinh",
        "z_ct_event_step_dose": "Event-step dose",
        "z_ct_v1_broad_step": "Event-step dose",
        "z_ct_v2_broad_cumulative": "V2 broad cumulative",
        "z_ct_v3_broad_predicted_opt": "V3 predicted OPT",
        "z_ct_v4_matched_step": "V4 matched step",
        "z_ct_v5_matched_pulse": "V5 matched pulse",
        "z_ct_v6_broad_composition": "V6 composition only",
        "z_ct_v7_matched_pulse_growth_rate": "V7 matched pulse growth rate",
        "z_ct_falsification_lead4_broad": "Falsification lead4 broad",
        "z_ct_falsification_lead4_matched": "Falsification lead4 matched",
        "z_ct_share_2008_2010": "Share window 2008-2010",
        "z_ct_share_2011_2013": "Share window 2011-2013",
        "z_ct_share_2008_2013": "Share window 2008-2013",
        "z_ct_full": "Full-sample shares",
    }
    return labels.get(instrument_col, instrument_col)


def _build_regression_variant_headline_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Construct a compact cross-variant summary table for console output."""
    if results_df.empty or "instrument_variant" not in results_df.columns or "label" not in results_df.columns:
        return pd.DataFrame()

    order_map = {
        instrument: idx
        for idx, instrument in enumerate(pd.unique(results_df["instrument_variant"]))
    }
    outcome_map = {
        outcome: idx
        for idx, outcome in enumerate(pd.unique(results_df["outcome_col"]))
    } if "outcome_col" in results_df.columns else {}
    id_cols = ["instrument_variant"]
    if "outcome_col" in results_df.columns:
        id_cols.insert(0, "outcome_col")
    wanted = id_cols + ["label", "coef_instrument", "se_instrument", "f_stat", "n_obs"]
    work = results_df.loc[:, [c for c in wanted if c in results_df.columns]].copy()
    if work.empty:
        return pd.DataFrame()

    first_stage = (
        work.loc[work["label"] == "first_stage_twfe"]
        .rename(columns={
            "coef_instrument": "fs_twfe_coef",
            "se_instrument": "fs_twfe_se",
            "f_stat": "fs_twfe_f",
            "n_obs": "fs_twfe_n",
        })
        .drop(columns=["label"], errors="ignore")
    )
    reduced_form = (
        work.loc[work["label"] == "reduced_form_twfe"]
        .rename(columns={
            "coef_instrument": "rf_twfe_coef",
            "se_instrument": "rf_twfe_se",
            "f_stat": "rf_twfe_f",
            "n_obs": "rf_twfe_n",
        })
        .drop(columns=["label"], errors="ignore")
    )
    summary = first_stage.merge(reduced_form, on=id_cols, how="outer")
    if summary.empty:
        return pd.DataFrame()
    insert_at = 2 if "outcome_col" in summary.columns else 1
    summary.insert(insert_at, "variant_label", summary["instrument_variant"].map(_instrument_variant_display_name))
    if "outcome_col" in summary.columns:
        summary["_outcome_order"] = summary["outcome_col"].map(outcome_map).fillna(len(outcome_map))
    else:
        summary["_outcome_order"] = 0
    summary["_variant_order"] = summary["instrument_variant"].map(order_map).fillna(len(order_map))
    summary = summary.sort_values(["_outcome_order", "_variant_order"]).drop(columns=["_outcome_order", "_variant_order"])
    return summary.reset_index(drop=True)

def _check_paths(paths: dict) -> None:
    missing = [f"  {k}: {v}" for k, v in paths.items() if not Path(v).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

# --- normalization helpers ---

def _normalize_school_sample_mode(raw) -> str:
    v = str(raw or "").strip().lower().replace("-", "_")
    if v in {"", "all", "legacy"}:
        return "all"
    if v in {"matched_shift_sample", "matched_sample", "matched", "sampled"}:
        return "matched_shift_sample"
    raise ValueError(f"Invalid school_sample_mode: {raw!r}")

def _normalize_school_shift_metric(raw) -> str:
    v = str(raw or "").strip().lower().replace("-", "_")
    if v in {"", "ihmp", "ihmp_share"}:
        return "ihmp_share"
    if v in {"international", "international_share", "intl", "intl_share"}:
        return "international_share"
    if v in {"opt_ihmp", "opt_ihmp_share"}:
        return "opt_ihmp_share"
    if v in {"opt", "opt_share"}:
        return "opt_share"
    raise ValueError(f"Invalid school_shift_metric: {raw!r}")

def _normalize_degree_scope(raw) -> str:
    v = str(raw or "").strip().lower().replace("-", "_")
    if v in {"", "both", "all", "ba_ma", "ma_ba", "bachelors_masters", "masters_bachelors"}:
        return "bachelors_masters"
    if v in {"bachelors", "bachelor", "ba"}:
        return "bachelors"
    if v in {"masters", "master", "ma", "ms"}:
        return "masters"
    raise ValueError(f"Invalid degree_scope: {raw!r}")

def _normalize_opt_shift_normalization(raw) -> str:
    v = str(raw or "").strip().lower().replace("-", "_")
    if v in {"", "ipeds", "ipeds_graduates", "graduates"}:
        return "ipeds_graduates"
    if v in {"foia", "foia_students"}:
        return "foia_students"
    if v in {"none", "raw", "count", "unnormalized"}:
        return "none"
    raise ValueError(f"Invalid opt_shifts_normalization: {raw!r}")

def _school_metric_uses_foia(metric: str) -> bool:
    return metric in {"opt_ihmp_share", "opt_share"}


def _normalize_shock_design(raw) -> str:
    v = str(raw or "").strip().lower().replace("-", "_")
    if v in {"", "event_quantity", "event_based", "event", "quantity"}:
        return "event_quantity"
    if v in {"legacy", "raw"}:
        return "legacy"
    raise ValueError(f"Invalid shock_design: {raw!r}")


def _normalize_share_period(raw) -> str:
    v = str(raw or "").strip().lower().replace("-", "_")
    if v in {"", "pre_window", "pre", "window"}:
        return "pre_window"
    if v in {"full", "all"}:
        return "full"
    if v in {"base_year", "base"}:
        return "base_year"
    raise ValueError(f"Invalid share_period: {raw!r}")


def _window_label(start: int, end: int) -> str:
    return f"{int(start)}_{int(end)}"


def _parse_share_robustness_windows(raw) -> list[tuple[int, int]]:
    default = [(2008, 2010), (2011, 2013), (2008, 2013)]
    if raw is None:
        return default
    out: list[tuple[int, int]] = []
    if isinstance(raw, str):
        chunks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
        for chunk in chunks:
            parts = [p.strip() for p in chunk.replace(":", "-").split("-") if p.strip()]
            if len(parts) != 2:
                continue
            try:
                out.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue
        return out or default
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    out.append((int(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
    return out or default


def _resolve_share_windows(
    share_period: str,
    share_base_year: int,
    share_year_min: int,
    share_year_max: int,
    robustness_windows: list[tuple[int, int]],
) -> tuple[str, Optional[tuple[int, int]], list[tuple[str, tuple[int, int] | None]]]:
    period = _normalize_share_period(share_period)
    baseline_window: tuple[int, int] | None
    baseline_label: str
    if period == "full":
        baseline_window = None
        baseline_label = "full"
    elif period == "base_year":
        baseline_window = (int(share_base_year) - 5, int(share_base_year))
        baseline_label = _window_label(*baseline_window)
    else:
        baseline_window = (int(share_year_min), int(share_year_max))
        baseline_label = _window_label(*baseline_window)

    labels_seen = {baseline_label}
    variants: list[tuple[str, tuple[int, int] | None]] = [(baseline_label, baseline_window)]
    for start, end in robustness_windows:
        label = _window_label(start, end)
        if label in labels_seen:
            continue
        labels_seen.add(label)
        variants.append((label, (int(start), int(end))))
    if "full" not in labels_seen:
        variants.append(("full", None))
    return baseline_label, baseline_window, variants


def _share_column_name(label: str, baseline_label: str) -> str:
    return "share_ck" if label == baseline_label else f"share_ck_{label}"

def _school_metric_is_opt_family(metric: str) -> bool:
    return metric in {"opt_ihmp_share", "opt_share"}

# --- SQL expression builders ---

def _sql_normalize(colname: str) -> str:
    return (
        f"TRIM(REGEXP_REPLACE(REGEXP_REPLACE(LOWER({colname}), "
        f"'[^a-z0-9 ]', ' ', 'g'), '\\\\s+', ' ', 'g'))"
    )

def _sql_clean_company_name(col: str) -> str:
    sfx = (
        "(?i)\\b(inc|inc\\.|incorporated|llc|l\\.l\\.c|llp|l\\.l\\.p|lp|l\\.p|"
        "ltd|ltd\\.|limited|corp|corp\\.|corporation|company|co|co\\.|"
        "pllc|plc|pc|pc\\.|gmbh|ag|sa)\\b"
    )
    return _sql_normalize(f"REGEXP_REPLACE({col}, '{sfx}', ' ', 'g')")

def _sql_state_name_to_abbr(statecol: str) -> str:
    mapping = {
        "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
        "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
        "district of columbia": "DC", "washington dc": "DC", "dc": "DC",
        "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
        "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
        "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
        "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
        "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
        "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
        "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
        "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
        "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
        "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
        "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR", "guam": "GU",
    }
    cases = "\n".join(
        f"WHEN LOWER(TRIM({statecol})) = '{n}' THEN '{a}'" for n, a in mapping.items()
    )
    return f"CASE {cases} ELSE UPPER(TRIM({statecol})) END"

def _sql_clean_zip(zipcol: str) -> str:
    zc = f"TRIM(CAST(REGEXP_REPLACE({zipcol}, '[^0-9]', '', 'g') AS VARCHAR))"
    return (
        f"CASE WHEN LENGTH({zc}) = 4 THEN '0' || {zc} "
        f"WHEN LENGTH({zc}) >= 5 THEN SUBSTRING({zc} FROM 1 FOR 5) "
        f"ELSE {zc} END"
    )

def _date_parse_sql(col: str) -> str:
    return (
        f"COALESCE("
        f"TRY_CAST({col} AS DATE),"
        f"TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%Y-%m-%d') AS DATE),"
        f"TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%m/%d/%Y') AS DATE),"
        f"TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%Y-%m-%d %H:%M:%S') AS DATE),"
        f"TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%m/%d/%Y %H:%M:%S') AS DATE)"
        f")"
    )

def _sql_normalize_cip6(colname: str) -> str:
    digits = f"REGEXP_REPLACE(TRIM(CAST({colname} AS VARCHAR)), '[^0-9]', '', 'g')"
    return (
        f"CASE WHEN {digits} IS NULL OR TRIM(CAST({digits} AS VARCHAR)) = '' THEN NULL "
        f"ELSE LPAD(SUBSTRING(TRIM(CAST({digits} AS VARCHAR)) FROM 1 FOR 6), 6, '0') END"
    )

def _degree_predicate_for_scope(
    con: ddb.DuckDBPyConnection, view: str, degree_scope: str
) -> Optional[str]:
    degree_scope = _normalize_degree_scope(degree_scope)
    if degree_scope == "bachelors_masters":
        word = (
            "LOWER(CAST({col} AS VARCHAR)) LIKE '%master%' "
            "OR LOWER(CAST({col} AS VARCHAR)) LIKE '%bachelor%'"
        )
    elif degree_scope == "bachelors":
        word = "LOWER(CAST({col} AS VARCHAR)) LIKE '%bachelor%'"
    else:
        word = "LOWER(CAST({col} AS VARCHAR)) LIKE '%master%'"

    if _has_column(con, view, "student_edu_level_desc"):
        col = "student_edu_level_desc"
        if degree_scope == "bachelors_masters":
            return (
                f"LOWER(CAST({col} AS VARCHAR)) IN ('master''s','masters','bachelor''s','bachelors') "
                f"OR ({word.format(col=col)})"
            )
        if degree_scope == "bachelors":
            return (
                f"LOWER(CAST({col} AS VARCHAR)) IN ('bachelor''s','bachelors') "
                f"OR {word.format(col=col)}"
            )
        return (
            f"LOWER(CAST({col} AS VARCHAR)) IN ('master''s','masters') "
            f"OR {word.format(col=col)}"
        )
    if _has_column(con, view, "awlevel_group"):
        col = "awlevel_group"
        if degree_scope == "bachelors_masters":
            return (
                f"LOWER(CAST({col} AS VARCHAR)) IN ('master','masters','bachelor','bachelors') "
                f"OR ({word.format(col=col)})"
            )
        if degree_scope == "bachelors":
            return (
                f"LOWER(CAST({col} AS VARCHAR)) IN ('bachelor','bachelors') "
                f"OR {word.format(col=col)}"
            )
        return (
            f"LOWER(CAST({col} AS VARCHAR)) IN ('master','masters') "
            f"OR {word.format(col=col)}"
        )
    if _has_column(con, view, "awlevel"):
        if degree_scope == "bachelors_masters":
            return "CAST(awlevel AS INTEGER) IN (5, 7)"
        if degree_scope == "bachelors":
            return "CAST(awlevel AS INTEGER) = 5"
        return "CAST(awlevel AS INTEGER) = 7"
    return None

def _degree_predicate(
    con: ddb.DuckDBPyConnection, view: str, include_bachelors: bool = False
) -> Optional[str]:
    scope = "bachelors_masters" if include_bachelors else "masters"
    return _degree_predicate_for_scope(con, view=view, degree_scope=scope)

def _ipeds_program_year_ctes_sql(exclude_unitids: Optional[list[str]] = None) -> str:
    ex = ""
    if exclude_unitids:
        ex = f"AND CAST(unitid AS VARCHAR) NOT IN {_sql_in_list(exclude_unitids)}"
    return f"""
        ipeds_program_year AS (
            SELECT
                CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k,
                CAST(year AS INTEGER) AS t,
                LPAD(CAST(CAST(cipcode AS BIGINT) AS VARCHAR), 6, '0') AS cip6,
                SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0)) AS program_students,
                SUM(COALESCE(CAST(cnralt AS DOUBLE), 0)) AS program_intl_students,
                CASE WHEN SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0)) > 0
                     THEN SUM(COALESCE(CAST(cnralt AS DOUBLE), 0))
                          / SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0))
                     ELSE NULL END AS program_share_intl
            FROM ipeds_raw
            WHERE unitid IS NOT NULL AND year IS NOT NULL AND cipcode IS NOT NULL {ex}
            GROUP BY 1, 2, 3
        ),
        ipeds_school_year AS (
            SELECT k, t,
                SUM(program_students) AS school_size,
                SUM(program_intl_students) AS total_intl_students
            FROM ipeds_program_year GROUP BY 1, 2
        )
    """

def _foia_school_program_person_ctes_sql(
    con: ddb.DuckDBPyConnection,
    degree_scope: str,
    exclude_unitids: Optional[list[str]] = None,
) -> str:
    end_col = _first_present_column(
        con, "foia_raw",
        ["program_end_date", "program_completion_date", "program_end_dt", "program_complete_date"],
    )
    if end_col is None:
        raise ValueError("FOIA school metric requires a program end-date column.")
    cip_col = _first_present_column(
        con, "foia_raw", ["major_1_cip_code", "program_cip_code", "cipcode", "cip"]
    )
    end_expr = _date_parse_sql(end_col)
    opt_date_cols = [
        c for c in ("opt_employer_start_date", "opt_authorization_start_date", "authorization_start_date")
        if _has_column(con, "foia_raw", c)
    ]
    opt_expr = (
        f"CASE WHEN COALESCE({', '.join(_date_parse_sql(c) for c in opt_date_cols)}) IS NOT NULL THEN 1 ELSE 0 END"
        if opt_date_cols else "0"
    )
    deg_pred = _degree_predicate_for_scope(con, view="foia_raw", degree_scope=degree_scope)
    deg_clause = f"AND ({deg_pred})" if deg_pred else ""
    ex_clause = f"AND cw.k NOT IN {_sql_in_list(exclude_unitids)}" if exclude_unitids else ""
    cip_expr = _sql_normalize_cip6(cip_col) if cip_col else "CAST(NULL AS VARCHAR)"
    return f"""
        cw AS (
            SELECT TRIM(CAST(school_name AS VARCHAR)) AS school_name_raw,
                COALESCE(TRIM(CAST(f1_city_clean AS VARCHAR)), '') AS f1_city_clean,
                COALESCE(TRIM(CAST(f1_state_clean AS VARCHAR)), '') AS f1_state_clean,
                COALESCE(TRIM(CAST(f1_zip_clean AS VARCHAR)), '') AS f1_zip_clean,
                CAST(CAST(MIN(UNITID) AS BIGINT) AS VARCHAR) AS k
            FROM f1_inst_unitid_cw WHERE UNITID IS NOT NULL AND school_name IS NOT NULL
            GROUP BY 1, 2, 3, 4
        ),
        foia_students AS (
            SELECT person_id,
                CAST(EXTRACT(YEAR FROM {end_expr}) AS INTEGER) AS t,
                TRIM(CAST(school_name AS VARCHAR)) AS school_name_raw,
                COALESCE({_sql_normalize('campus_city')}, '') AS f1_city_clean,
                COALESCE({_sql_state_name_to_abbr('campus_state')}, '') AS f1_state_clean,
                COALESCE({_sql_clean_zip('campus_zip_code')}, '') AS f1_zip_clean,
                {cip_expr} AS cip6,
                {opt_expr} AS has_opt
            FROM foia_raw
            WHERE person_id IS NOT NULL AND school_name IS NOT NULL AND {end_expr} IS NOT NULL
              {deg_clause}
        ),
        foia_school_program_person AS (
            SELECT cw.k, f.t, f.person_id, f.cip6, MAX(f.has_opt) AS ever_opt
            FROM foia_students f
            JOIN cw ON f.school_name_raw = cw.school_name_raw
              AND f.f1_city_clean = cw.f1_city_clean
              AND f.f1_state_clean = cw.f1_state_clean
              AND f.f1_zip_clean = cw.f1_zip_clean
            WHERE f.t IS NOT NULL {ex_clause}
            GROUP BY 1, 2, 3, 4
        ),
        foia_school_year AS (
            SELECT k, t,
                COUNT(DISTINCT person_id) AS foia_total_students,
                COUNT(DISTINCT CASE WHEN ever_opt = 1 THEN person_id END) AS foia_total_opt_students
            FROM foia_school_program_person GROUP BY 1, 2
        ),
        foia_program_year AS (
            SELECT k, t, cip6,
                COUNT(DISTINCT person_id) AS foia_program_students,
                COUNT(DISTINCT CASE WHEN ever_opt = 1 THEN person_id END) AS foia_program_opt_students,
                CASE WHEN COUNT(DISTINCT person_id) > 0
                     THEN COUNT(DISTINCT CASE WHEN ever_opt = 1 THEN person_id END)::DOUBLE
                          / COUNT(DISTINCT person_id)::DOUBLE
                    ELSE NULL END AS foia_program_opt_share
            FROM foia_school_program_person WHERE cip6 IS NOT NULL GROUP BY 1, 2, 3
        )
    """


def _foia_school_year_ctes_sql(
    con: ddb.DuckDBPyConnection,
    degree_scope: str,
    exclude_unitids: Optional[list[str]] = None,
) -> str:
    """Backward-compatible helper retained for opt-ihmp/opt-share paths."""
    return _foia_school_program_person_ctes_sql(
        con,
        degree_scope,
        exclude_unitids=exclude_unitids,
    )


# =============================================================================
# ── SCHOOL METRIC PANEL ───────────────────────────────────────────────────────
# =============================================================================

def _empty_school_metric_panel() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "k", "t", "metric", "school_size", "metric_level", "metric_share",
        "ipeds_total_students", "ipeds_total_intl_students",
        "foia_total_students", "foia_total_opt_students",
    ])

def _empty_school_shift_sample() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "k", "school_name", "metric", "sample_role", "selected_for_instrument",
        "matched_school_k", "matched_school_name", "matched_pair_id", "treated_rank",
        "treated_score", "control_eligible", "treated_candidate", "has_full_window_coverage",
        "has_event_window_coverage", "event_pre_share", "event_post_share", "event_pre_size",
        "event_pre_level", "event_post_level", "event_level_growth", "event_level_growth_rate",
        "event_pre_opt_rate",
        "meets_min_size", "fails_large_yoy_drop", "min_required_size", "control_positive_cap",
        "avg_size_window", "log_avg_size_window", "max_positive_annual_change",
        "max_positive_yoy_size_change", "max_negative_yoy_size_change", "fails_large_yoy_size_jump",
        "treated_event_year", "matched_treated_event_year",
    ])

def _finalize_school_metric_panel(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        out = _empty_school_metric_panel()
        out["metric"] = pd.Series(dtype="string")
        return out
    out = df.copy()
    out["k"] = out["k"].astype(str)
    out["t"] = pd.to_numeric(out["t"], errors="coerce").astype("Int64")
    for col in ["school_size", "metric_level", "metric_share",
                "ipeds_total_students", "ipeds_total_intl_students",
                "foia_total_students", "foia_total_opt_students"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["metric"] = metric
    return out[[
        "k", "t", "metric", "school_size", "metric_level", "metric_share",
        "ipeds_total_students", "ipeds_total_intl_students",
        "foia_total_students", "foia_total_opt_students",
    ]].sort_values(["k", "t"]).reset_index(drop=True)

def _build_ipeds_school_metric_panel(
    con: ddb.DuckDBPyConnection, metric: str, exclude_unitids: Optional[list[str]] = None
) -> pd.DataFrame:
    metric = _normalize_school_shift_metric(metric)
    if metric not in {"ihmp_share", "international_share"}:
        raise ValueError(f"Unsupported IPEDS school metric: {metric!r}")
    if metric == "ihmp_share":
        level_expr = "COALESCE(m.metric_level, 0)"
        metric_cte = """
            school_metric AS (
                SELECT k, t,
                    SUM(CASE WHEN program_share_intl >= 0.5 AND program_students >= 10
                             THEN program_students ELSE 0 END) AS metric_level
                FROM ipeds_program_year GROUP BY 1, 2
            )
        """
    else:
        level_expr = "COALESCE(sy.total_intl_students, 0)"
        metric_cte = """
            school_metric AS (
                SELECT k, t, total_intl_students AS metric_level FROM ipeds_school_year
            )
        """
    df = con.sql(f"""
        WITH {_ipeds_program_year_ctes_sql(exclude_unitids)}, {metric_cte}
        SELECT sy.k, sy.t, sy.school_size,
            {level_expr} AS metric_level,
            CASE WHEN sy.school_size > 0 THEN {level_expr} / sy.school_size ELSE NULL END AS metric_share,
            sy.school_size AS ipeds_total_students,
            sy.total_intl_students AS ipeds_total_intl_students,
            CAST(NULL AS DOUBLE) AS foia_total_students,
            CAST(NULL AS DOUBLE) AS foia_total_opt_students
        FROM ipeds_school_year sy
        LEFT JOIN school_metric m ON sy.k = m.k AND sy.t = m.t
        ORDER BY sy.k, sy.t
    """).df()
    return _finalize_school_metric_panel(df, metric)

def _build_opt_ihmp_school_metric_panel(
    con: ddb.DuckDBPyConnection,
    degree_scope: str,
    exclude_unitids: Optional[list[str]] = None,
    ipeds_share_intl_threshold: float = 0.30,
    foia_opt_share_threshold: float = 0.50,
    min_program_f1_count: int = 10,
) -> pd.DataFrame:
    df = con.sql(f"""
        WITH {_ipeds_program_year_ctes_sql(exclude_unitids)},
        {_foia_school_year_ctes_sql(con, degree_scope=degree_scope, exclude_unitids=exclude_unitids)},
        school_metric AS (
            SELECT
                k,
                t,
                foia_total_opt_students AS metric_level
            FROM foia_school_year
        )
        SELECT sy.k, sy.t, sy.school_size,
            COALESCE(m.metric_level, 0) AS metric_level,
            CASE WHEN sy.school_size > 0
                 THEN COALESCE(m.metric_level, 0) / sy.school_size ELSE NULL END AS metric_share,
            sy.school_size AS ipeds_total_students,
            sy.total_intl_students AS ipeds_total_intl_students,
            fy.foia_total_students, fy.foia_total_opt_students
        FROM ipeds_school_year sy
        LEFT JOIN school_metric m ON sy.k = m.k AND sy.t = m.t
        LEFT JOIN foia_school_year fy ON sy.k = fy.k AND sy.t = fy.t
        ORDER BY sy.k, sy.t
    """).df()
    return _finalize_school_metric_panel(df, "opt_ihmp_share")

def _build_opt_share_school_metric_panel(
    con: ddb.DuckDBPyConnection,
    degree_scope: str,
    exclude_unitids: Optional[list[str]] = None,
) -> pd.DataFrame:
    df = con.sql(f"""
        WITH {_foia_school_year_ctes_sql(con, degree_scope=degree_scope, exclude_unitids=exclude_unitids)}
        SELECT fy.k, fy.t, fy.foia_total_students AS school_size,
            fy.foia_total_opt_students AS metric_level,
            CASE WHEN fy.foia_total_students > 0
                 THEN fy.foia_total_opt_students::DOUBLE / fy.foia_total_students::DOUBLE
                 ELSE NULL END AS metric_share,
            CAST(NULL AS DOUBLE) AS ipeds_total_students,
            CAST(NULL AS DOUBLE) AS ipeds_total_intl_students,
            fy.foia_total_students, fy.foia_total_opt_students
        FROM foia_school_year fy
        ORDER BY fy.k, fy.t
    """).df()
    return _finalize_school_metric_panel(df, "opt_share")

def _build_school_metric_panel(
    con: ddb.DuckDBPyConnection,
    metric: str,
    degree_scope: str,
    exclude_unitids: Optional[list[str]] = None,
    opt_ihmp_ipeds_share_intl_threshold: float = 0.30,
    opt_ihmp_foia_opt_share_threshold: float = 0.50,
    opt_ihmp_min_program_f1_count: int = 10,
) -> pd.DataFrame:
    metric = _normalize_school_shift_metric(metric)
    if metric in {"ihmp_share", "international_share"}:
        return _build_ipeds_school_metric_panel(con, metric=metric, exclude_unitids=exclude_unitids)
    if metric == "opt_ihmp_share":
        return _build_opt_ihmp_school_metric_panel(
            con, degree_scope=degree_scope, exclude_unitids=exclude_unitids,
            ipeds_share_intl_threshold=opt_ihmp_ipeds_share_intl_threshold,
            foia_opt_share_threshold=opt_ihmp_foia_opt_share_threshold,
            min_program_f1_count=opt_ihmp_min_program_f1_count,
        )
    if metric == "opt_share":
        return _build_opt_share_school_metric_panel(
            con, degree_scope=degree_scope, exclude_unitids=exclude_unitids
        )
    raise ValueError(f"Unsupported school shift metric: {metric!r}")


def _build_school_event_summary(
    metric_panel: pd.DataFrame,
    window_start: int,
    window_end: int,
    event_pre_years: int = 2,
    event_post_years: int = 2,
    opt_share_panel: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if metric_panel.empty:
        return pd.DataFrame(columns=[
            "k", "treated_event_year", "treated_score", "event_pre_share", "event_post_share",
            "event_pre_size", "event_pre_level", "event_post_level", "event_level_growth",
            "event_pre_opt_rate", "has_event_window_coverage",
        ])

    needed_cols = ["k", "t", "school_size", "metric_share"]
    work = metric_panel.loc[:, needed_cols + [c for c in ["metric_level"] if c in metric_panel.columns]].copy()
    work["k"] = work["k"].astype(str)
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work = work.loc[work["t"].notna()].copy()
    work["t"] = work["t"].astype(int)
    if "metric_level" not in work.columns:
        work["metric_level"] = (
            pd.to_numeric(work["school_size"], errors="coerce")
            * pd.to_numeric(work["metric_share"], errors="coerce")
        )

    candidate_years = list(range(int(window_start), int(window_end) + 1))
    required_years = sorted({
        year
        for e in candidate_years
        for year in list(range(e - int(event_pre_years), e)) + list(range(e, e + int(event_post_years)))
    })
    share_wide = work.pivot(index="k", columns="t", values="metric_share").reindex(columns=required_years)
    size_wide = work.pivot(index="k", columns="t", values="school_size").reindex(columns=required_years)
    level_wide = work.pivot(index="k", columns="t", values="metric_level").reindex(columns=required_years)
    opt_wide = None
    if opt_share_panel is not None and not opt_share_panel.empty:
        opt_work = opt_share_panel.loc[:, ["k", "t", "metric_share"]].copy()
        opt_work["k"] = opt_work["k"].astype(str)
        opt_work["t"] = pd.to_numeric(opt_work["t"], errors="coerce")
        opt_work = opt_work.loc[opt_work["t"].notna()].copy()
        opt_work["t"] = opt_work["t"].astype(int)
        opt_wide = opt_work.pivot(index="k", columns="t", values="metric_share").reindex(
            index=share_wide.index, columns=required_years
        )

    summary = pd.DataFrame({"k": share_wide.index.astype(str)}).set_index("k", drop=False)
    delta_cols: list[str] = []
    coverage_cols: list[str] = []
    for event_year in candidate_years:
        pre_cols = list(range(event_year - int(event_pre_years), event_year))
        post_cols = list(range(event_year, event_year + int(event_post_years)))
        pre_share = share_wide[pre_cols].mean(axis=1, skipna=False)
        post_share = share_wide[post_cols].mean(axis=1, skipna=False)
        pre_size = size_wide[pre_cols].mean(axis=1, skipna=False)
        pre_level = level_wide[pre_cols].mean(axis=1, skipna=False)
        post_level = level_wide[post_cols].mean(axis=1, skipna=False)
        if opt_wide is not None:
            pre_opt = opt_wide[pre_cols].mean(axis=1, skipna=False)
        else:
            pre_opt = pd.Series(0.0, index=share_wide.index)
        coverage = (
            share_wide[pre_cols + post_cols].notna().all(axis=1)
            & level_wide[pre_cols + post_cols].notna().all(axis=1)
            & size_wide[pre_cols].notna().all(axis=1)
        )
        delta_col = f"event_delta_{event_year}"
        coverage_col = f"event_coverage_{event_year}"
        summary[delta_col] = (post_share - pre_share).where(coverage)
        summary[f"event_pre_share_{event_year}"] = pre_share.where(coverage)
        summary[f"event_post_share_{event_year}"] = post_share.where(coverage)
        summary[f"event_pre_size_{event_year}"] = pre_size.where(coverage)
        summary[f"event_pre_level_{event_year}"] = pre_level.where(coverage)
        summary[f"event_post_level_{event_year}"] = post_level.where(coverage)
        level_growth = (post_level - pre_level).where(coverage)
        summary[f"event_level_growth_{event_year}"] = level_growth
        summary[f"event_level_growth_rate_{event_year}"] = (
            level_growth / pre_level.where(pre_level > 0)
        ).where(coverage)
        summary[f"event_pre_opt_rate_{event_year}"] = pre_opt.where(coverage)
        summary[coverage_col] = coverage.astype("int8")
        delta_cols.append(delta_col)
        coverage_cols.append(coverage_col)

    positive_deltas = summary[delta_cols].clip(lower=0)
    summary["treated_score"] = positive_deltas.max(axis=1, skipna=True).fillna(0.0)
    summary["treated_event_year"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    summary["event_pre_share"] = np.nan
    summary["event_post_share"] = np.nan
    summary["event_pre_size"] = np.nan
    summary["event_pre_level"] = np.nan
    summary["event_post_level"] = np.nan
    summary["event_level_growth"] = np.nan
    summary["event_level_growth_rate"] = np.nan
    summary["event_pre_opt_rate"] = np.nan
    summary["has_event_window_coverage"] = 0
    for school_k in summary.index:
        best_year: Optional[int] = None
        best_val = float("-inf")
        for event_year in candidate_years:
            val = summary.loc[school_k, f"event_delta_{event_year}"]
            if pd.notna(val) and float(val) > best_val:
                best_val = float(val)
                best_year = int(event_year)
        if best_year is None or best_val <= 0:
            continue
        summary.loc[school_k, "treated_event_year"] = best_year
        summary.loc[school_k, "event_pre_share"] = summary.loc[school_k, f"event_pre_share_{best_year}"]
        summary.loc[school_k, "event_post_share"] = summary.loc[school_k, f"event_post_share_{best_year}"]
        summary.loc[school_k, "event_pre_size"] = summary.loc[school_k, f"event_pre_size_{best_year}"]
        summary.loc[school_k, "event_pre_level"] = summary.loc[school_k, f"event_pre_level_{best_year}"]
        summary.loc[school_k, "event_post_level"] = summary.loc[school_k, f"event_post_level_{best_year}"]
        summary.loc[school_k, "event_level_growth"] = summary.loc[school_k, f"event_level_growth_{best_year}"]
        summary.loc[school_k, "event_level_growth_rate"] = summary.loc[
            school_k, f"event_level_growth_rate_{best_year}"
        ]
        summary.loc[school_k, "event_pre_opt_rate"] = summary.loc[school_k, f"event_pre_opt_rate_{best_year}"]
        summary.loc[school_k, "has_event_window_coverage"] = int(summary.loc[school_k, f"event_coverage_{best_year}"] or 0)
    return summary.reset_index(drop=True)

def _build_school_shift_sample(
    metric_panel: pd.DataFrame,
    metric: str,
    school_name_lookup: Optional[pd.DataFrame] = None,
    school_classification: Optional[pd.DataFrame] = None,
    opt_share_panel: Optional[pd.DataFrame] = None,
    n_shifted: int = 25,
    window_start: int = 2014,
    window_end: int = 2017,
    event_pre_years: int = 2,
    event_post_years: int = 2,
    control_positive_cap: float = 0.02,
    match_on_carnegie_classification: bool = False,
    min_school_size: int = 100,
    opt_share_min_school_f1_count: int = 50,
    opt_share_max_yoy_drop: float = 0.50,
    restrict_treated_to_no_large_enrollment_jump: bool = False,
    max_yoy_size_jump: float = 0.50,
) -> pd.DataFrame:
    metric = _normalize_school_shift_metric(metric)
    if metric_panel.empty:
        return _empty_school_shift_sample()
    years = list(range(int(window_start), int(window_end) + 1))
    if len(years) < 2:
        raise ValueError("School sample window must span at least two years.")
    work = metric_panel.loc[
        metric_panel["t"].isin(years),
        ["k", "t", "school_size", "metric_level", "metric_share"],
    ].copy()
    if work.empty:
        return _empty_school_shift_sample()

    share_wide = work.pivot(index="k", columns="t", values="metric_share").reindex(columns=years)
    size_wide = work.pivot(index="k", columns="t", values="school_size").reindex(columns=years)
    level_wide = work.pivot(index="k", columns="t", values="metric_level").reindex(columns=years)

    summary = pd.DataFrame({"k": share_wide.index.astype(str)}).set_index("k", drop=False)
    summary.index.name = None
    summary["metric"] = metric
    name_map: dict = {}
    if school_name_lookup is not None and not school_name_lookup.empty:
        name_map = dict(zip(
            school_name_lookup["k"].astype(str),
            school_name_lookup["school_name"].fillna("").astype(str),
        ))
    summary["school_name"] = summary.index.map(name_map).fillna("")
    for year in years:
        summary[f"school_size_{year}"] = size_wide[year]
        summary[f"metric_level_{year}"] = level_wide[year]
        summary[f"metric_share_{year}"] = share_wide[year]

    event_summary = _build_school_event_summary(
        metric_panel=metric_panel,
        window_start=window_start,
        window_end=window_end,
        event_pre_years=event_pre_years,
        event_post_years=event_post_years,
        opt_share_panel=opt_share_panel,
    ).set_index("k", drop=False)
    for col in event_summary.columns:
        if col == "k":
            continue
        summary[col] = event_summary[col]

    share_change_cols: list[str] = []
    size_change_cols: list[str] = []
    for y0, y1 in zip(years, years[1:]):
        sc = f"metric_share_change_{y0}_{y1}"
        szc = f"school_size_pct_change_{y0}_{y1}"
        summary[sc] = summary[f"metric_share_{y1}"] - summary[f"metric_share_{y0}"]
        prev = summary[f"school_size_{y0}"]
        curr = summary[f"school_size_{y1}"]
        summary[szc] = curr / prev - 1.0
        share_change_cols.append(sc)
        size_change_cols.append(szc)

    summary["has_full_window_coverage"] = (
        size_wide.notna().all(axis=1) & share_wide.notna().all(axis=1)
    ).astype("int8")
    min_req = int(opt_share_min_school_f1_count) if metric == "opt_share" else int(min_school_size)
    summary["min_required_size"] = min_req
    summary["meets_min_size"] = size_wide.ge(float(min_req)).all(axis=1).astype("int8")
    event_delta_cols = [c for c in summary.columns if c.startswith("event_delta_")]
    positive_event_deltas = summary[event_delta_cols].clip(lower=0) if event_delta_cols else pd.DataFrame(index=summary.index)
    summary["max_positive_annual_change"] = positive_event_deltas.max(axis=1, skipna=True).fillna(0.0)
    summary["treated_score"] = positive_event_deltas.max(axis=1, skipna=True).fillna(0.0)
    summary["control_positive_cap"] = float(control_positive_cap)
    summary["max_positive_yoy_size_change"] = summary[size_change_cols].max(axis=1, skipna=True)
    if metric == "opt_share":
        summary["max_negative_yoy_size_change"] = summary[size_change_cols].min(axis=1, skipna=True)
        summary["fails_large_yoy_drop"] = (
            summary["max_negative_yoy_size_change"] < -float(opt_share_max_yoy_drop)
        ).astype("int8")
    else:
        summary["max_negative_yoy_size_change"] = pd.NA
        summary["fails_large_yoy_drop"] = 0
    if restrict_treated_to_no_large_enrollment_jump:
        summary["fails_large_yoy_size_jump"] = (
            summary["max_positive_yoy_size_change"] > float(max_yoy_size_jump)
        ).astype("int8")
    else:
        summary["fails_large_yoy_size_jump"] = 0

    base_eligible = (
        (summary["has_full_window_coverage"] == 1)
        & (summary["meets_min_size"] == 1)
        & (summary["fails_large_yoy_drop"] == 0)
    )
    treated_base_eligible = (
        base_eligible
        if not restrict_treated_to_no_large_enrollment_jump
        else (base_eligible & (summary["fails_large_yoy_size_jump"] == 0))
    )
    summary["control_eligible"] = (
        base_eligible
        & summary.get("has_event_window_coverage", 0).fillna(0).astype(int).eq(1)
        & (
            positive_event_deltas.le(float(control_positive_cap)).all(axis=1)
            if not positive_event_deltas.empty else False
        )
    ).astype("int8")
    summary["treated_candidate"] = (
        treated_base_eligible
        & summary.get("has_event_window_coverage", 0).fillna(0).astype(int).eq(1)
        & summary["treated_score"].gt(0)
    ).astype("int8")
    summary["avg_size_window"] = summary[[f"school_size_{y}" for y in years]].mean(axis=1, skipna=True)
    summary["log_avg_size_window"] = summary["avg_size_window"].apply(
        lambda v: math.log(v) if pd.notna(v) and v > 0 else float("nan")
    )
    summary["selected_for_instrument"] = 0
    summary["sample_role"] = pd.NA
    summary["matched_school_k"] = pd.NA
    summary["matched_school_name"] = pd.NA
    summary["matched_pair_id"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    summary["treated_rank"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    summary["matched_treated_event_year"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")

    class_lookup: dict[str, str] = {}
    if match_on_carnegie_classification and school_classification is not None and not school_classification.empty:
        if {"k", "carnegie_classification"}.issubset(set(school_classification.columns)):
            class_lookup = dict(
                zip(
                    school_classification["k"].astype(str).str.strip(),
                    school_classification["carnegie_classification"].astype(str).str.strip(),
                )
            )
        else:
            print("[match] school_classification provided without expected columns; skipping Carnegie-aware matching.")

    treated_candidates = (
        summary.loc[summary["treated_candidate"] == 1]
        .sort_values(["treated_score", "k"], ascending=[False, True])
        .copy()
    )
    if not treated_candidates.empty:
        summary.loc[treated_candidates.index, "treated_rank"] = pd.Series(
            range(1, len(treated_candidates) + 1),
            index=treated_candidates.index, dtype="Int64",
        )
    selected_treated = treated_candidates.head(int(n_shifted)).copy()
    control_pool = summary.loc[
        (summary["control_eligible"] == 1) & (~summary.index.isin(selected_treated.index))
    ].copy()
    used_controls: set = set()
    fallback_to_size_only_count = 0
    pair_id = 0
    assignments: list = []
    for treated_k, treated_row in selected_treated.iterrows():
        avail = control_pool.loc[~control_pool.index.isin(used_controls)].copy()
        if class_lookup:
            treated_class = class_lookup.get(str(treated_k), "")
            if treated_class:
                class_avail = avail.loc[
                    avail.index.to_series().map(lambda k: class_lookup.get(str(k), "")) == treated_class
                ].copy()
                if not class_avail.empty:
                    avail = class_avail
                else:
                    fallback_to_size_only_count += 1
        if avail.empty:
            break
        avail["match_gap"] = (avail["log_avg_size_window"] - treated_row["log_avg_size_window"]).abs()
        chosen = avail.sort_values(["match_gap", "k"]).iloc[0]
        pair_id += 1
        assignments.append({
            "treated_k": treated_k,
            "control_k": str(chosen["k"]),
            "pair": pair_id,
            "treated_event_year": treated_row["treated_event_year"],
        })
        used_controls.add(str(chosen["k"]))

    if pair_id == 0 and int(n_shifted) > 0:
        raise ValueError("Matched-school sampling found no treat-control pairs. Check metric thresholds.")
    if pair_id < int(n_shifted):
        print(f"[warn] Requested {n_shifted} shifted schools but only matched {pair_id} pair(s).")
    if match_on_carnegie_classification and class_lookup and fallback_to_size_only_count:
        print(
            f"[warn] Carnegie-aware matching fell back to size-only for {fallback_to_size_only_count} treated school(s)."
        )

    for a in assignments:
        tk, ck, pid = str(a["treated_k"]), str(a["control_k"]), int(a["pair"])
        event_year = a.get("treated_event_year", pd.NA)
        summary.loc[tk, "selected_for_instrument"] = 1
        summary.loc[tk, "sample_role"] = "treated"
        summary.loc[tk, "matched_school_k"] = ck
        summary.loc[tk, "matched_school_name"] = name_map.get(ck, "")
        summary.loc[tk, "matched_pair_id"] = pid
        summary.loc[tk, "matched_treated_event_year"] = event_year
        summary.loc[ck, "selected_for_instrument"] = 1
        summary.loc[ck, "sample_role"] = "control"
        summary.loc[ck, "matched_school_k"] = tk
        summary.loc[ck, "matched_school_name"] = name_map.get(tk, "")
        summary.loc[ck, "matched_pair_id"] = pid
        summary.loc[ck, "matched_treated_event_year"] = event_year

    return summary.reset_index(drop=True).sort_values(
        ["selected_for_instrument", "sample_role", "treated_rank", "k"],
        ascending=[False, True, True, True], na_position="last",
    ).reset_index(drop=True)

def _school_name_lookup(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    if not _has_column(con, "ipeds_main_institutions", "main_unitid"):
        return pd.DataFrame(columns=["k", "school_name"])
    lookup = con.sql("""
        SELECT CAST(main_unitid AS VARCHAR) AS k,
            COALESCE(NULLIF(TRIM(CAST(ipeds_name AS VARCHAR)), ''),
                     NULLIF(TRIM(CAST(ipeds_instname_clean AS VARCHAR)), '')) AS school_name
        FROM ipeds_main_institutions WHERE main_unitid IS NOT NULL
    """).df()
    if lookup.empty:
        return pd.DataFrame(columns=["k", "school_name"])
    lookup["k"] = lookup["k"].astype(str)
    lookup["school_name"] = lookup["school_name"].fillna("").astype(str)
    return lookup.drop_duplicates("k")


def _school_classification_lookup(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    class_candidates = [
        "carnegie_classification",
        "c21basic_lab",
        "c21basic",
        "instcat_lab",
        "instcat",
        "carnegie",
        "carnegie_class",
        "carnegie_class_cd",
        "carnegie_basic",
    ]
    sources = [
        ("ipeds_completions_all", ["unitid"]),
        ("ipeds_main_institutions", ["main_unitid", "unitid", "k"]),
        ("ipeds_raw", ["unitid"]),
        ("ipeds_raw_ma", ["unitid"]),
    ]
    for source, id_candidates in sources:
        try:
            id_col = _first_present_column(con, source, id_candidates)
            class_col = _first_present_column(con, source, class_candidates)
        except Exception:
            continue
        if not id_col or not class_col:
            continue
        lookup = con.sql(f"""
            SELECT
                CAST({id_col} AS VARCHAR) AS k,
                CAST({class_col} AS VARCHAR) AS carnegie_classification
            FROM {source}
            WHERE {id_col} IS NOT NULL AND {class_col} IS NOT NULL
        """).df()
        if lookup.empty:
            continue
        lookup["k"] = lookup["k"].astype(str).str.strip()
        lookup["carnegie_classification"] = lookup["carnegie_classification"].astype(str).str.strip()
        lookup = lookup[(lookup["k"] != "") & (lookup["carnegie_classification"] != "")]
        lookup = lookup.dropna(subset=["k", "carnegie_classification"]).drop_duplicates("k")
        if not lookup.empty:
            return lookup[["k", "carnegie_classification"]].copy()
    return pd.DataFrame(columns=["k", "carnegie_classification"])


def _register_school_metric_views(
    con: ddb.DuckDBPyConnection,
    metric_panel: pd.DataFrame,
    sample_summary: pd.DataFrame,
) -> None:
    panel = metric_panel.copy()
    sample_cols = ["k", "school_name", "selected_for_instrument", "sample_role",
                   "matched_school_k", "matched_pair_id"]
    panel = panel.merge(sample_summary[sample_cols], on="k", how="left")
    con.register("_school_shift_metric_panel_df", panel)
    con.sql("CREATE OR REPLACE TEMP VIEW school_shift_metric_panel AS SELECT * FROM _school_shift_metric_panel_df")
    con.register("_school_shift_sample_df", sample_summary)
    con.sql("CREATE OR REPLACE TEMP VIEW school_shift_sample AS SELECT * FROM _school_shift_sample_df")


def _build_event_quantity_growth_view(
    con: ddb.DuckDBPyConnection,
    metric_panel: pd.DataFrame,
    sample_summary: pd.DataFrame,
    opt_share_panel: Optional[pd.DataFrame],
    year_min: int,
    year_max: int,
    event_window_start: int,
    event_window_end: int,
    event_pre_years: int = 2,
    event_post_years: int = 2,
    falsification_lead_years: int = 4,
    common_baseline_start: int = 2011,
    common_baseline_end: int = 2013,
    require_common_baseline_full_coverage: bool = True,
) -> None:
    if int(falsification_lead_years) != 4:
        raise ValueError("Only falsification_lead_years=4 is currently implemented.")
    if metric_panel.empty:
        con.register("_event_quantity_growth_df", pd.DataFrame(columns=["k", "t", "g_kt"]))
        con.sql("CREATE OR REPLACE TEMP VIEW ipeds_unit_growth AS SELECT * FROM _event_quantity_growth_df")
        return

    work = metric_panel.copy()
    work["k"] = work["k"].astype(str)
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work = work.loc[work["t"].notna()].copy()
    work["t"] = work["t"].astype(int)
    work["metric_level"] = pd.to_numeric(work.get("metric_level"), errors="coerce")

    base_start = int(common_baseline_start)
    base_end = int(common_baseline_end)
    if base_end < base_start:
        raise ValueError("common_baseline_end must be >= common_baseline_start.")
    base_years = list(range(base_start, base_end + 1))
    baseline = (
        work.loc[work["t"].isin(base_years), ["k", "t", "metric_level"]]
        .dropna(subset=["metric_level"])
        .groupby("k", as_index=False)
        .agg(
            common_base_metric_level=("metric_level", "mean"),
            common_base_n_years=("t", "nunique"),
        )
    )
    if require_common_baseline_full_coverage:
        baseline.loc[
            baseline["common_base_n_years"] < len(base_years),
            "common_base_metric_level",
        ] = np.nan
    work = work.merge(baseline, on="k", how="left")
    work = work.loc[work["t"].between(int(year_min), int(year_max))].copy()

    sample_cols = [
        "k", "school_name", "selected_for_instrument", "sample_role",
        "treated_event_year", "treated_score", "event_pre_share",
        "event_post_share", "event_pre_size", "event_pre_level",
        "event_post_level", "event_level_growth", "event_level_growth_rate",
        "event_pre_opt_rate", "matched_treated_event_year", "matched_pair_id",
    ]
    sample_merge = sample_summary.loc[:, [c for c in sample_cols if c in sample_summary.columns]].copy()
    work = work.merge(sample_merge, on="k", how="left")

    if opt_share_panel is not None and not opt_share_panel.empty:
        opt_merge = opt_share_panel.loc[:, ["k", "t", "metric_share", "metric_level"]].copy()
        opt_merge["k"] = opt_merge["k"].astype(str)
        opt_merge["t"] = pd.to_numeric(opt_merge["t"], errors="coerce")
        opt_merge = opt_merge.loc[opt_merge["t"].notna()].copy()
        opt_merge["t"] = opt_merge["t"].astype(int)
        opt_merge = opt_merge.rename(columns={
            "metric_share": "opt_share_school",
            "metric_level": "opt_count_school",
        })
        work = work.merge(opt_merge, on=["k", "t"], how="left")
    else:
        work["opt_share_school"] = np.nan
        work["opt_count_school"] = np.nan

    event_year = pd.to_numeric(work["treated_event_year"], errors="coerce")
    t_vals = pd.to_numeric(work["t"], errors="coerce")
    delta_share = pd.to_numeric(work["treated_score"], errors="coerce").fillna(0.0)
    pre_size = pd.to_numeric(work["event_pre_size"], errors="coerce").fillna(0.0)
    pre_share = pd.to_numeric(work["event_pre_share"], errors="coerce").fillna(0.0)
    if "event_level_growth" in work.columns:
        event_level_growth = pd.to_numeric(work["event_level_growth"], errors="coerce").fillna(0.0)
    else:
        event_level_growth = pre_size * delta_share
    if "event_pre_level" in work.columns:
        event_pre_level = pd.to_numeric(work["event_pre_level"], errors="coerce")
    else:
        event_pre_level = pre_size * pre_share
    event_pre_level = event_pre_level.replace([float("inf"), float("-inf")], np.nan)
    if "event_level_growth_rate" in work.columns:
        event_level_growth_rate = pd.to_numeric(work["event_level_growth_rate"], errors="coerce")
    else:
        event_level_growth_rate = event_level_growth / event_pre_level.where(event_pre_level > 0)
    event_level_growth_rate = (
        event_level_growth_rate.replace([float("inf"), float("-inf")], np.nan).fillna(0.0)
    )
    pre_opt_rate = pd.to_numeric(work["event_pre_opt_rate"], errors="coerce").fillna(0.0)
    selected_mask = pd.to_numeric(work["selected_for_instrument"], errors="coerce").fillna(0).astype(int).eq(1)
    post_mask = event_year.notna() & t_vals.notna() & (t_vals >= event_year)
    pulse_mask = event_year.notna() & t_vals.notna() & (t_vals == event_year)
    metric_level = pd.to_numeric(work["metric_level"], errors="coerce")
    common_base_level = pd.to_numeric(work["common_base_metric_level"], errors="coerce")

    g_raw_flow = metric_level
    g_common_base_level = metric_level - common_base_level
    g_common_base_asinh = np.arcsinh(metric_level.clip(lower=0.0)) - np.arcsinh(
        common_base_level.clip(lower=0.0)
    )
    work["_metric_level_lag1"] = work.groupby("k", sort=False)["metric_level"].shift(1)
    g_flow_diff = (metric_level - pd.to_numeric(work["_metric_level_lag1"], errors="coerce")).fillna(0.0)
    g_flow_ar_resid = pd.Series(0.0, index=work.index, dtype="float64")
    ar_mask = metric_level.notna() & pd.to_numeric(work["_metric_level_lag1"], errors="coerce").notna()
    if ar_mask.sum() >= 3:
        try:
            y_ar = metric_level.loc[ar_mask].astype(float)
            lag_ar = pd.to_numeric(work.loc[ar_mask, "_metric_level_lag1"], errors="coerce").astype(float)
            y_resid = _residualize_fixed_effects(
                y_ar,
                [work.loc[ar_mask, "k"].astype(str), work.loc[ar_mask, "t"].astype(str)],
            )
            lag_resid = _residualize_fixed_effects(
                lag_ar,
                [work.loc[ar_mask, "k"].astype(str), work.loc[ar_mask, "t"].astype(str)],
            )
            denom = float(np.dot(lag_resid, lag_resid))
            beta_ar = float(np.dot(lag_resid, y_resid) / denom) if denom > 0 else 0.0
            g_flow_ar_resid.loc[ar_mask] = y_resid - beta_ar * lag_resid
        except Exception as e:
            print(f"[growth] AR-residual flow innovation failed; falling back to first differences: {e}")
            g_flow_ar_resid = g_flow_diff.copy()
    g_event_pulse = np.where(pulse_mask, event_level_growth_rate, 0.0)
    work["_g_kt_flow_diff_for_cumsum"] = g_flow_diff
    work["_g_kt_flow_ar_resid_for_cumsum"] = g_flow_ar_resid
    work = work.sort_values(["k", "t"]).copy()
    g_flow_diff_cumulative = work.groupby("k")["_g_kt_flow_diff_for_cumsum"].cumsum()
    g_flow_ar_resid_cumulative = work.groupby("k")["_g_kt_flow_ar_resid_for_cumsum"].cumsum()
    g_v1 = np.where(post_mask, pre_size * delta_share, 0.0)
    g_v2 = np.where(post_mask, pre_size * (pd.to_numeric(work["metric_share"], errors="coerce").fillna(0.0) - pre_share), 0.0)
    g_v3 = np.where(post_mask, pre_opt_rate * pre_size * delta_share, 0.0)
    g_v4 = np.where(selected_mask, g_v1, 0.0)
    g_v5 = np.where(selected_mask & pulse_mask, pre_size * delta_share, 0.0)
    g_v6 = np.where(post_mask, delta_share, 0.0)
    g_v7 = np.where(selected_mask & pulse_mask, event_level_growth_rate, 0.0)
    falsification_lead_year = event_year - int(falsification_lead_years)
    falsification_post_mask = falsification_lead_year.notna() & t_vals.notna() & (t_vals >= falsification_lead_year)
    g_falsification_lead4_broad = np.where(falsification_post_mask, pre_size * delta_share, 0.0)
    g_falsification_lead4_matched = np.where(
        selected_mask & falsification_post_mask,
        pre_size * delta_share,
        0.0,
    )

    work["g_kt"] = g_event_pulse
    work["g_kt_raw_flow"] = g_raw_flow
    work["g_kt_ihmp_share"] = pd.to_numeric(work["metric_share"], errors="coerce").fillna(0.0)
    work["g_kt_event_pulse"] = g_event_pulse
    work["g_kt_flow_diff"] = g_flow_diff
    work["g_kt_flow_ar_resid"] = g_flow_ar_resid
    work["g_kt_flow_diff_cumulative"] = g_flow_diff_cumulative
    work["g_kt_flow_ar_resid_cumulative"] = g_flow_ar_resid_cumulative
    work["g_kt_common_base_level"] = g_common_base_level
    work["g_kt_common_base_asinh"] = g_common_base_asinh
    work["g_kt_event_step_dose"] = g_v1
    work["g_kt_v1_broad_step"] = g_v1
    work["g_kt_v2_broad_cumulative"] = g_v2
    work["g_kt_v3_broad_predicted_opt"] = g_v3
    work["g_kt_v4_matched_step"] = g_v4
    work["g_kt_v5_matched_pulse"] = g_v5
    work["g_kt_v6_broad_composition"] = g_v6
    work["g_kt_v7_matched_pulse_growth_rate"] = g_v7
    work["falsification_lead_year"] = falsification_lead_year
    work["g_kt_falsification_lead4_broad"] = g_falsification_lead4_broad
    work["g_kt_falsification_lead4_matched"] = g_falsification_lead4_matched
    work["delta_share_event"] = delta_share
    work["pre_school_size_event"] = pre_size
    work["event_level_growth_rate"] = event_level_growth_rate
    work["pre_opt_rate_event"] = pre_opt_rate
    work["common_base_metric_level"] = common_base_level
    work["common_base_n_years"] = pd.to_numeric(work["common_base_n_years"], errors="coerce")
    work["common_baseline_start"] = base_start
    work["common_baseline_end"] = base_end
    work["common_baseline_requires_full_coverage"] = bool(require_common_baseline_full_coverage)
    work["event_window_start"] = int(event_window_start)
    work["event_window_end"] = int(event_window_end)
    work["event_pre_years"] = int(event_pre_years)
    work["event_post_years"] = int(event_post_years)

    cols = [
        "k", "t", "metric", "school_size", "metric_level", "metric_share",
        "ipeds_total_students", "ipeds_total_intl_students",
        "foia_total_students", "foia_total_opt_students",
        "school_name", "selected_for_instrument", "sample_role", "matched_pair_id",
        "treated_event_year", "treated_score", "event_pre_share", "event_post_share",
        "event_pre_size", "event_pre_level", "event_post_level", "event_level_growth",
        "event_level_growth_rate", "event_pre_opt_rate", "matched_treated_event_year",
        "opt_share_school", "opt_count_school", "delta_share_event",
        "pre_school_size_event", "pre_opt_rate_event",
        "common_base_metric_level", "common_base_n_years",
        "common_baseline_start", "common_baseline_end",
        "common_baseline_requires_full_coverage",
        "g_kt", "g_kt_raw_flow", "g_kt_common_base_level", "g_kt_common_base_asinh",
        "g_kt_ihmp_share",
        "g_kt_event_pulse", "g_kt_flow_diff", "g_kt_flow_ar_resid",
        "g_kt_flow_diff_cumulative", "g_kt_flow_ar_resid_cumulative",
        "g_kt_event_step_dose", "g_kt_v1_broad_step", "g_kt_v2_broad_cumulative",
        "g_kt_v3_broad_predicted_opt", "g_kt_v4_matched_step",
        "g_kt_v5_matched_pulse", "g_kt_v6_broad_composition",
        "g_kt_v7_matched_pulse_growth_rate",
        "falsification_lead_year",
        "g_kt_falsification_lead4_broad", "g_kt_falsification_lead4_matched",
        "event_window_start", "event_window_end", "event_pre_years", "event_post_years",
    ]
    out = work.loc[:, [c for c in cols if c in work.columns]].sort_values(["k", "t"]).reset_index(drop=True)
    con.register("_event_quantity_growth_df", out)
    con.sql("CREATE OR REPLACE TEMP VIEW ipeds_unit_growth AS SELECT * FROM _event_quantity_growth_df")


# =============================================================================
# ── SECTION 1: INPUT REGISTRATION ────────────────────────────────────────────
# =============================================================================

def _load_inputs(
    con: ddb.DuckDBPyConnection,
    paths_cfg: dict,
    pipeline_cfg: dict,
    testing_cfg: dict,
) -> None:
    """Register all input parquet files as DuckDB views and check required paths exist."""
    include_bachelors = bool(pipeline_cfg.get("include_bachelors", False))
    opt_shifts = bool(pipeline_cfg.get("opt_shifts", True))
    shock_design = _normalize_shock_design(pipeline_cfg.get("shock_design", "event_quantity"))
    school_sample_mode = _normalize_school_sample_mode(pipeline_cfg.get("school_sample_mode", "matched_shift_sample"))
    school_shift_metric = _normalize_school_shift_metric(
        pipeline_cfg.get("school_shift_metric", "ihmp_share")
    )

    required = {
        "transitions": paths_cfg["transitions"],
        "headcounts": paths_cfg["headcounts"],
        "revelio_ipeds_foia_inst_crosswalk": paths_cfg["revelio_ipeds_foia_inst_crosswalk"],
        "ipeds_ma_only": paths_cfg["ipeds_ma_only"],
        "foia_sevp_with_person_id": paths_cfg["foia_sevp_with_person_id"],
        "foia_sevp_with_person_id_employment_corrected": paths_cfg["foia_sevp_with_person_id_employment_corrected"],
        "employer_crosswalk": paths_cfg["employer_crosswalk"],
        "preferred_rcids": paths_cfg["preferred_rcids"],
    }
    if shock_design == "event_quantity" or opt_shifts or _school_metric_uses_foia(school_shift_metric):
        required["f1_inst_unitid_crosswalk"] = paths_cfg["f1_inst_unitid_crosswalk"]
    _check_paths(required)

    # --- Revelio transitions and headcounts ---
    con.sql(f"CREATE OR REPLACE TEMP VIEW revelio_transitions AS SELECT * FROM read_parquet('{_escape(paths_cfg['transitions'])}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW revelio_headcount AS SELECT * FROM read_parquet('{_escape(paths_cfg['headcounts'])}')")
    print(f"[inputs] transitions: {con.sql('SELECT COUNT(*) FROM revelio_transitions').fetchone()[0]:,} rows")
    print(f"[inputs] headcount: {con.sql('SELECT COUNT(*) FROM revelio_headcount').fetchone()[0]:,} rows")
    workforce_path = paths_cfg.get("wrds_company_year_workforce_out")
    if workforce_path and Path(workforce_path).exists():
        con.sql(
            "CREATE OR REPLACE TEMP VIEW wrds_company_year_workforce AS "
            f"SELECT * FROM read_parquet('{_escape(workforce_path)}')"
        )
        n_workforce = con.sql("SELECT COUNT(*) FROM wrds_company_year_workforce").fetchone()[0]
        print(f"[inputs] WRDS company-year workforce cache: {n_workforce:,} rows")
    else:
        print("[inputs] WRDS company-year workforce cache not found; split-hire and tenure outcomes will be missing/zero.")

    # --- Revelio school -> IPEDS mapping ---
    det_map = paths_cfg.get("revelio_inst_deterministic_map")
    ref_cat = paths_cfg.get("revelio_ref_inst_catalog")
    school_map, meta = load_revelio_school_map(
        legacy_crosswalk=Path(paths_cfg["revelio_ipeds_foia_inst_crosswalk"]),
        deterministic_triple_map=Path(det_map) if det_map and Path(det_map).exists() else None,
        ref_inst_catalog=Path(ref_cat) if ref_cat and Path(ref_cat).exists() else None,
    )
    con.register("_revelio_inst_cw_df", school_map[["university_raw_key", "unitid"]])
    con.sql("""
        CREATE OR REPLACE TEMP VIEW revelio_inst_cw AS
        SELECT CAST(university_raw_key AS VARCHAR) AS university_raw_norm,
               CAST(unitid AS VARCHAR) AS unitid
        FROM _revelio_inst_cw_df
        WHERE university_raw_key IS NOT NULL AND unitid IS NOT NULL
    """)
    print(f"[inputs] revelio school map: {meta['mapping_method']} ({len(school_map):,} schools)")

    # --- IPEDS ---
    ipeds_src = paths_cfg.get("ipeds_ma_ba_only") if include_bachelors else None
    if ipeds_src and Path(ipeds_src).exists():
        con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw AS SELECT * FROM read_parquet('{_escape(ipeds_src)}')")
    else:
        con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw AS SELECT * FROM read_parquet('{_escape(paths_cfg['ipeds_ma_only'])}')")
    # Also load BA+MA IPEDS for opt normalization if needed.
    ipeds_ma_ba_path = paths_cfg.get("ipeds_ma_ba_only")
    if ipeds_ma_ba_path and Path(ipeds_ma_ba_path).exists():
        con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw_ma AS SELECT * FROM read_parquet('{_escape(paths_cfg['ipeds_ma_only'])}')")
        con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw_ma_ba AS SELECT * FROM read_parquet('{_escape(ipeds_ma_ba_path)}')")
    else:
        con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw_ma AS SELECT * FROM ipeds_raw")
        con.sql("CREATE OR REPLACE TEMP VIEW ipeds_raw_ma_ba AS SELECT * FROM ipeds_raw WHERE 1=0")

    # IPEDS institution dimension (school names, optional).
    ipeds_main_path = paths_cfg.get("ipeds_main_institutions")
    if ipeds_main_path and Path(ipeds_main_path).exists():
        con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_main_institutions AS SELECT * FROM read_parquet('{_escape(ipeds_main_path)}')")
    else:
        con.sql("""CREATE OR REPLACE TEMP VIEW ipeds_main_institutions AS
                   SELECT CAST(NULL AS VARCHAR) AS main_unitid, CAST(NULL AS VARCHAR) AS ipeds_name,
                   CAST(NULL AS VARCHAR) AS ipeds_instname_clean WHERE 1=0""")
    
    # Optional full IPEDS completion export for Carnegie/instcat lookup.
    ipeds_class_path = (
        paths_cfg.get("ipeds_completions_all")
        or paths_cfg.get("ipeds_completions_parquet")
    )
    if ipeds_class_path:
        ipeds_class_path = Path(ipeds_class_path)
        if ipeds_class_path.exists() and ipeds_class_path.suffix.lower() == ".parquet":
            con.sql(
                f"CREATE OR REPLACE TEMP VIEW ipeds_completions_all AS "
                f"SELECT * FROM read_parquet('{_escape(ipeds_class_path)}')"
            )
            print(
                "[inputs] ipeds completions (classification source): "
                f"{con.sql('SELECT COUNT(*) FROM ipeds_completions_all').fetchone()[0]:,} rows"
            )
        elif ipeds_class_path.exists():
            print(
                f"[warn] Skipping ipeds_completions_all (not parquet): {ipeds_class_path}"
            )
        else:
            print(f"[warn] Missing ipeds_completions_all: {ipeds_class_path}")

    # --- FOIA ---
    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{_escape(paths_cfg['foia_sevp_with_person_id_employment_corrected'])}') WHERE year_int > 2005")
    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw_full AS SELECT * FROM read_parquet('{_escape(paths_cfg['foia_sevp_with_person_id'])}') WHERE year_int > 2005")

    # --- F1 institution -> IPEDS crosswalk (for FOIA-based metrics) ---
    f1_cw = paths_cfg.get("f1_inst_unitid_crosswalk")
    if f1_cw and Path(f1_cw).exists():
        con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_unitid_cw AS SELECT * FROM read_parquet('{_escape(f1_cw)}')")

    # --- Employer crosswalk and preferred RCIDs ---
    con.sql(f"CREATE OR REPLACE TEMP VIEW preferred_rcids AS SELECT DISTINCT preferred_rcid FROM read_parquet('{_escape(paths_cfg['preferred_rcids'])}')")
    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW employer_crosswalk AS
        SELECT ec.* FROM read_parquet('{_escape(paths_cfg['employer_crosswalk'])}') ec
        JOIN preferred_rcids pr ON ec.preferred_rcid = pr.preferred_rcid
    """)
    con.sql("""
        CREATE OR REPLACE TEMP VIEW matched_rcids AS
        SELECT DISTINCT preferred_rcid AS rcid FROM employer_crosswalk WHERE preferred_rcid IS NOT NULL
    """)
    print(f"[inputs] matched companies: {con.sql('SELECT COUNT(*) FROM matched_rcids').fetchone()[0]:,}")

    # --- Testing: subsample matched_rcids ---
    if testing_cfg.get("enabled", False):
        n = int(testing_cfg.get("sample_n_companies", 100))
        seed = int(testing_cfg.get("seed", 0))
        con.sql(f"""
            CREATE OR REPLACE TEMP VIEW matched_rcids AS
            SELECT rcid FROM (
                SELECT DISTINCT preferred_rcid AS rcid FROM employer_crosswalk WHERE preferred_rcid IS NOT NULL
                ORDER BY rcid
            ) USING SAMPLE {n} (system, {seed})
        """)
        actual_n = con.sql("SELECT COUNT(*) FROM matched_rcids").fetchone()[0]
        print(f"[testing] sampled {actual_n} companies (seed={seed})")


# =============================================================================
# ── SECTION 2a: GROWTH VIEW (g_kt) ───────────────────────────────────────────
# =============================================================================

def _build_growth_view_matched_sample(
    con: ddb.DuckDBPyConnection,
    growth_window_start: int = 2014,
    growth_window_end: int = 2017,
) -> None:
    """Build g_kt from the matched school sample metric panel (for matched_shift_sample mode)."""
    gk_start = int(growth_window_start)
    gk_end = int(growth_window_end)
    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW ipeds_unit_growth AS
        WITH selected AS (
            SELECT m.k, CAST(m.t AS INTEGER) AS t,
                   CAST(COALESCE(m.metric_level, 0) AS DOUBLE) AS g_kt
            FROM school_shift_metric_panel m
            JOIN school_shift_sample s ON m.k = s.k
            WHERE COALESCE(s.selected_for_instrument, 0) = 1 AND m.t IS NOT NULL
        ),
        bounds AS (SELECT k, MIN(t) AS min_t, MAX(t) AS max_t FROM selected GROUP BY k),
        expanded AS (
            SELECT b.k, gs.year AS t
            FROM bounds b, LATERAL generate_series(b.min_t, b.max_t) AS gs(year)
        ),
        filled AS (
            SELECT e.k, e.t, COALESCE(s.g_kt, 0) AS g_kt
            FROM expanded e LEFT JOIN selected s ON e.k = s.k AND e.t = s.t
        )
        SELECT k, t,
            CASE WHEN t BETWEEN {gk_start} AND {gk_end} THEN g_kt ELSE 0 END AS g_kt
        FROM filled
        ORDER BY k, t
    """)

def _build_ipeds_growth_view(
    con: ddb.DuckDBPyConnection,
    growth_population: str,
    use_changes: bool,
    demean_by_school: bool,
    exclude_unitids: Optional[list[str]] = None,
    growth_window_start: int = 2014,
    growth_window_end: int = 2017,
) -> None:
    """Build g_kt from IPEDS international enrollment. growth_population selects column."""
    ex_clause = f"AND CAST(unitid AS VARCHAR) NOT IN {_sql_in_list(exclude_unitids)}" if exclude_unitids else ""
    gp = growth_population.lower().strip()
    if gp == "main":
        g_src = "tot_seats_ihma"
    elif gp == "all":
        g_src = "tot_intl_students"
    elif gp == "intl":
        g_src = "tot_intl_seats_ihma"
    else:
        raise ValueError(f"Invalid growth_population: {growth_population!r}")

    g_expr = g_src if not use_changes else f"ASINH({g_src}) - ASINH({g_src}_lag)"
    gk_start = int(growth_window_start)
    gk_end = int(growth_window_end)
    demean_cte = ""
    final_view = "raw_out"
    if demean_by_school:
        demean_cte = f", demeaned AS (SELECT k, t, g_kt - AVG(g_kt) OVER (PARTITION BY k) AS g_kt FROM raw_out)"
        final_view = "demeaned"

    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW ipeds_unit_growth AS
        WITH base AS (
            SELECT CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k,
                CAST(cipcode AS VARCHAR) AS cipcode,
                CAST(year AS INTEGER) AS year,
                CAST(cnralt AS DOUBLE) AS cnralt,
                CAST(ctotalt AS DOUBLE) AS ctotalt,
                CAST(share_intl AS DOUBLE) AS share_intl
            FROM ipeds_raw WHERE unitid IS NOT NULL {ex_clause}
        ),
        program_flags AS (
            SELECT k, cipcode,
                MAX(CASE WHEN year > 2010 AND share_intl >= 0.5 AND ctotalt >= 10 THEN 1 ELSE 0 END) AS ihmp
            FROM base WHERE cipcode IS NOT NULL GROUP BY k, cipcode
        ),
        joined AS (
            SELECT b.k, b.year,
                SUM(b.cnralt) AS tot_intl_students,
                SUM(COALESCE(b.ctotalt, 0) * COALESCE(p.ihmp, 0)) AS tot_seats_ihma,
                SUM(COALESCE(b.cnralt, 0) * COALESCE(p.ihmp, 0)) AS tot_intl_seats_ihma
            FROM base b LEFT JOIN program_flags p ON b.k = p.k AND b.cipcode = p.cipcode
            GROUP BY b.k, b.year
        ),
        bounds AS (SELECT k, MIN(year) AS min_y, MAX(year) AS max_y FROM joined GROUP BY k),
        expanded AS (SELECT b.k, gs.year FROM bounds b, LATERAL generate_series(b.min_y, b.max_y) AS gs(year)),
        filled AS (
            SELECT e.k, e.year,
                COALESCE(j.tot_intl_students, 0) AS tot_intl_students,
                COALESCE(j.tot_seats_ihma, 0) AS tot_seats_ihma,
                COALESCE(j.tot_intl_seats_ihma, 0) AS tot_intl_seats_ihma
            FROM expanded e LEFT JOIN joined j ON e.k = j.k AND e.year = j.year
        ),
        with_lags AS (
            SELECT k, year,
                {g_src} AS {g_src},
                LAG({g_src}) OVER (PARTITION BY k ORDER BY year) AS {g_src}_lag
            FROM filled
        ),
        raw_out AS (
            SELECT k, year AS t, {g_expr} AS g_kt FROM with_lags WHERE {g_src} IS NOT NULL
        )
        {demean_cte}
        SELECT k, t,
            CASE WHEN t BETWEEN {gk_start} AND {gk_end} THEN g_kt ELSE 0 END AS g_kt
        FROM {final_view}
    """)

def _build_opt_shift_growth_view(
    con: ddb.DuckDBPyConnection,
    degree_scope: str,
    normalization: str,
    use_changes: bool,
    demean_by_school: bool,
    exclude_unitids: Optional[list[str]] = None,
    growth_window_start: int = 2014,
    growth_window_end: int = 2017,
) -> None:
    """Build g_kt from FOIA OPT student counts per school-year."""
    end_col = _first_present_column(
        con, "foia_raw_full",
        ["program_end_date", "program_completion_date", "program_end_dt", "program_complete_date"],
    )
    if end_col is None:
        raise ValueError("opt_shifts=true requires a FOIA program end-date column.")
    end_expr = _date_parse_sql(end_col)
    opt_date_cols = [
        c for c in ("opt_employer_start_date", "opt_authorization_start_date", "authorization_start_date")
        if _has_column(con, "foia_raw_full", c)
    ]
    opt_expr = (
        f"CASE WHEN COALESCE({', '.join(_date_parse_sql(c) for c in opt_date_cols)}) IS NOT NULL THEN 1 ELSE 0 END"
        if opt_date_cols else "0"
    )
    deg_pred = _degree_predicate_for_scope(con, view="foia_raw_full", degree_scope=_normalize_degree_scope(degree_scope))
    deg_clause = f"AND ({deg_pred})" if deg_pred else ""
    ex_foia = f"AND cw.k NOT IN {_sql_in_list(exclude_unitids)}" if exclude_unitids else ""
    ex_ipeds = f"AND CAST(unitid AS VARCHAR) NOT IN {_sql_in_list(exclude_unitids)}" if exclude_unitids else ""

    normalization = _normalize_opt_shift_normalization(normalization)
    print(f"[growth] OPT shifts: degree_scope={degree_scope}, normalization={normalization}")

    if normalization == "none":
        post_opt_ctes = "school_years AS (SELECT k, t FROM opt_counts),"
        filled_metric_select = "CAST(NULL AS DOUBLE) AS total_graduates, COALESCE(o.opt_students, 0) AS opt_metric_kt"
        filled_join = "LEFT JOIN opt_counts o ON e.k = o.k AND e.t = o.t"
    elif normalization == "ipeds_graduates":
        post_opt_ctes = f"""
        ipeds_ma AS (
            SELECT CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k, CAST(year AS INTEGER) AS t,
                SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0)) AS total_graduates_ma
            FROM ipeds_raw_ma WHERE unitid IS NOT NULL AND year IS NOT NULL {ex_ipeds} GROUP BY 1, 2
        ),
        ipeds_maba AS (
            SELECT CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k, CAST(year AS INTEGER) AS t,
                SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0)) AS total_graduates_maba
            FROM ipeds_raw_ma_ba WHERE unitid IS NOT NULL AND year IS NOT NULL {ex_ipeds} GROUP BY 1, 2
        ),
        denom AS (
            SELECT COALESCE(ma.k, mb.k) AS k, COALESCE(ma.t, mb.t) AS t,
                COALESCE(mb.total_graduates_maba, COALESCE(ma.total_graduates_ma, 0)) AS total_graduates
            FROM ipeds_ma ma FULL OUTER JOIN ipeds_maba mb ON ma.k = mb.k AND ma.t = mb.t
        ),
        school_years AS (SELECT k, t FROM denom UNION SELECT k, t FROM opt_counts),"""
        filled_metric_select = ("d.total_graduates AS total_graduates, "
            "CASE WHEN d.total_graduates > 0 THEN COALESCE(o.opt_students, 0) / d.total_graduates ELSE 0 END AS opt_metric_kt")
        filled_join = "LEFT JOIN opt_counts o ON e.k = o.k AND e.t = o.t JOIN denom d ON e.k = d.k AND e.t = d.t"
    else:  # foia_students
        post_opt_ctes = """
        denom AS (SELECT k, t, COUNT(DISTINCT person_id) AS total_foia_students FROM foia_school_year_person GROUP BY 1, 2),
        school_years AS (SELECT k, t FROM denom UNION SELECT k, t FROM opt_counts),"""
        filled_metric_select = ("d.total_foia_students AS total_graduates, "
            "CASE WHEN d.total_foia_students > 0 THEN COALESCE(o.opt_students, 0) / d.total_foia_students ELSE 0 END AS opt_metric_kt")
        filled_join = "LEFT JOIN opt_counts o ON e.k = o.k AND e.t = o.t JOIN denom d ON e.k = d.k AND e.t = d.t"

    g_expr = "opt_metric_kt" if not use_changes else "ASINH(opt_metric_kt) - ASINH(opt_metric_kt_lag)"
    gk_start = int(growth_window_start)
    gk_end = int(growth_window_end)
    demean_cte = ""
    final_view = "raw_out"
    if demean_by_school:
        demean_cte = ", demeaned AS (SELECT k, t, g_kt - AVG(g_kt) OVER (PARTITION BY k) AS g_kt FROM raw_out)"
        final_view = "demeaned"

    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW ipeds_unit_growth AS
        WITH cw AS (
            SELECT TRIM(CAST(school_name AS VARCHAR)) AS school_name_raw,
                COALESCE(TRIM(CAST(f1_city_clean AS VARCHAR)), '') AS f1_city_clean,
                COALESCE(TRIM(CAST(f1_state_clean AS VARCHAR)), '') AS f1_state_clean,
                COALESCE(TRIM(CAST(f1_zip_clean AS VARCHAR)), '') AS f1_zip_clean,
                CAST(CAST(MIN(UNITID) AS BIGINT) AS VARCHAR) AS k
            FROM f1_inst_unitid_cw WHERE UNITID IS NOT NULL AND school_name IS NOT NULL GROUP BY 1,2,3,4
        ),
        foia_students AS (
            SELECT person_id, CAST(EXTRACT(YEAR FROM {end_expr}) AS INTEGER) AS t,
                TRIM(CAST(school_name AS VARCHAR)) AS school_name_raw,
                COALESCE({_sql_normalize('campus_city')}, '') AS f1_city_clean,
                COALESCE({_sql_state_name_to_abbr('campus_state')}, '') AS f1_state_clean,
                COALESCE({_sql_clean_zip('campus_zip_code')}, '') AS f1_zip_clean,
                {opt_expr} AS has_opt
            FROM foia_raw_full
            WHERE person_id IS NOT NULL AND school_name IS NOT NULL AND {end_expr} IS NOT NULL {deg_clause}
        ),
        foia_school_year_person AS (
            SELECT cw.k, f.t, f.person_id, MAX(f.has_opt) AS ever_opt
            FROM foia_students f JOIN cw ON f.school_name_raw = cw.school_name_raw
              AND f.f1_city_clean = cw.f1_city_clean AND f.f1_state_clean = cw.f1_state_clean
              AND f.f1_zip_clean = cw.f1_zip_clean
            WHERE f.t IS NOT NULL {ex_foia} GROUP BY 1,2,3
        ),
        opt_counts AS (
            SELECT k, t, COUNT(DISTINCT CASE WHEN ever_opt = 1 THEN person_id END) AS opt_students
            FROM foia_school_year_person GROUP BY k, t
        ),
        {post_opt_ctes}
        bounds AS (SELECT k, MIN(t) AS min_t, MAX(t) AS max_t FROM school_years GROUP BY k),
        expanded AS (SELECT b.k, gs.year AS t FROM bounds b, LATERAL generate_series(b.min_t, b.max_t) AS gs(year)),
        filled AS (
            SELECT e.k, e.t, COALESCE(o.opt_students, 0) AS opt_students, {filled_metric_select}
            FROM expanded e {filled_join}
        ),
        with_lags AS (
            SELECT k, t, opt_metric_kt, LAG(opt_metric_kt) OVER (PARTITION BY k ORDER BY t) AS opt_metric_kt_lag
            FROM filled
        ),
        raw_out AS (SELECT k, t, {g_expr} AS g_kt FROM with_lags WHERE opt_metric_kt IS NOT NULL)
        {demean_cte}
        SELECT k, t,
            CASE WHEN t BETWEEN {gk_start} AND {gk_end} THEN g_kt ELSE 0 END AS g_kt
        FROM {final_view}
    """)


# =============================================================================
# ── SECTION 2c: TRANSITION SHARES (share_ck) ──────────────────────────────────
# =============================================================================

def _build_transition_shares(
    con: ddb.DuckDBPyConnection,
    share_period: str,
    share_base_year: int,
    share_year_min: int,
    share_year_max: int,
    robustness_windows: list[tuple[int, int]],
    exclude_unitids: Optional[list[str]] = None,
    min_universities_for_share: int = 1,
) -> None:
    """
    Build share_ck: fraction of company c's new hires from university k.

    Denominator is total hires to IPEDS-matched universities within each window.
    `share_ck` is the baseline pre-shock window share and additional windows are
    exposed as `share_ck_<start>_<end>` plus `share_ck_full`.
    """
    if min_universities_for_share <= 0:
        min_universities_for_share = 1
    ex_clause = f"AND CAST(cw.unitid AS VARCHAR) NOT IN {_sql_in_list(exclude_unitids)}" if exclude_unitids else ""
    baseline_label, baseline_window, share_variants = _resolve_share_windows(
        share_period=share_period,
        share_base_year=share_base_year,
        share_year_min=share_year_min,
        share_year_max=share_year_max,
        robustness_windows=robustness_windows,
    )

    pair_terms: list[str] = []
    total_terms: list[str] = []
    count_terms: list[str] = []
    final_terms: list[str] = ["p.c", "p.k"]
    for label, window in share_variants:
        if window is None:
            n_expr = "SUM(n_transitions)"
        else:
            start, end = window
            n_expr = f"SUM(CASE WHEN year BETWEEN {int(start)} AND {int(end)} THEN n_transitions ELSE 0 END)"
        n_col = f"n_trans_{label}"
        share_col = _share_column_name(label, baseline_label)
        pair_terms.append(f"{n_expr} AS {n_col}")
        total_terms.append(f"SUM({n_col}) AS total_{label}")
        count_terms.append(f"COUNT(DISTINCT CASE WHEN {n_col} > 0 THEN k END) AS n_universities_{label}")
        final_terms.extend([
            f"p.{n_col}",
            f"CASE WHEN ct.total_{label} > 0 THEN p.{n_col} / ct.total_{label} ELSE NULL END AS {share_col}",
        ])
        if label == baseline_label:
            final_terms.append(f"p.{n_col} AS n_trans")
    final_terms.append(f"pc.n_universities_{baseline_label} AS n_universities_share_window")

    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW transition_shares AS
        WITH transitions AS (
            SELECT CAST(t.rcid AS INTEGER) AS c,
                CAST(cw.unitid AS VARCHAR) AS k,
                CAST(t.year AS INTEGER) AS year,
                t.n_transitions
            FROM revelio_transitions t
            JOIN matched_rcids mr ON t.rcid = mr.rcid
            JOIN revelio_inst_cw cw ON {sql_normalize_school_key('t.university_raw')} = cw.university_raw_norm
            WHERE t.n_transitions IS NOT NULL AND cw.unitid IS NOT NULL
              {ex_clause}
        ),
        pairs AS (
            SELECT c, k,
                {", ".join(pair_terms)}
            FROM transitions GROUP BY c, k
        ),
        pair_counts AS (
            SELECT c,
                {", ".join(count_terms)}
            FROM pairs GROUP BY c
        ),
        eligible_companies AS (
            SELECT c FROM pair_counts WHERE n_universities_{baseline_label} >= {int(min_universities_for_share)}
        ),
        company_totals_windowed AS (
            SELECT p.c,
                {", ".join(total_terms)}
            FROM pairs p GROUP BY p.c
        )
        SELECT
            {", ".join(final_terms)}
        FROM pairs p
        JOIN eligible_companies ec ON p.c = ec.c
        JOIN pair_counts pc ON p.c = pc.c
        JOIN company_totals_windowed ct ON p.c = ct.c
    """)
    n_pairs = con.sql("SELECT COUNT(*) FROM transition_shares").fetchone()[0]
    n_cos = con.sql("SELECT COUNT(DISTINCT c) FROM transition_shares").fetchone()[0]
    n_ks = con.sql("SELECT COUNT(DISTINCT k) FROM transition_shares").fetchone()[0]
    print(
        f"[shares] min_universities_for_share={int(min_universities_for_share)} | "
        f"baseline_window={baseline_label} | {n_pairs:,} (c,k) pairs | {n_cos:,} companies | {n_ks:,} universities"
    )


# =============================================================================
# ── SECTION 2d: INSTRUMENT (z_ct) ─────────────────────────────────────────────
# =============================================================================

def _build_instrument(con: ddb.DuckDBPyConnection) -> None:
    """Build multiple z_ct variants from pre-period share windows and event-based school shocks."""
    share_cols = [col for col in [
        "share_ck",
        "share_ck_2008_2010",
        "share_ck_2011_2013",
        "share_ck_2008_2013",
        "share_ck_full",
    ] if _has_column(con, "transition_shares", col)]
    g_cols = [col for col in [
        "g_kt",
        "g_kt_raw_flow",
        "g_kt_ihmp_share",
        "g_kt_event_pulse",
        "g_kt_flow_diff",
        "g_kt_flow_ar_resid",
        "g_kt_flow_diff_cumulative",
        "g_kt_flow_ar_resid_cumulative",
        "g_kt_common_base_level",
        "g_kt_common_base_asinh",
        "g_kt_event_step_dose",
        "g_kt_v1_broad_step",
        "g_kt_v2_broad_cumulative",
        "g_kt_v3_broad_predicted_opt",
        "g_kt_v4_matched_step",
        "g_kt_v5_matched_pulse",
        "g_kt_v6_broad_composition",
        "g_kt_v7_matched_pulse_growth_rate",
        "g_kt_falsification_lead4_broad",
        "g_kt_falsification_lead4_matched",
    ] if _has_column(con, "ipeds_unit_growth", col)]
    select_terms = ["s.c", "s.k", "g.t"]
    for share_col in share_cols:
        select_terms.append(f"CASE WHEN s.{share_col} > 1 THEN NULL ELSE s.{share_col} END AS {share_col}")
    passthrough_g_cols = [
        "metric_share", "metric_level", "school_size",
        "treated_event_year", "treated_score", "event_pre_size", "event_pre_share",
        "event_post_share", "event_level_growth", "event_level_growth_rate",
        "event_pre_opt_rate", "sample_role", "selected_for_instrument",
        "common_base_metric_level", "common_base_n_years",
    ]
    for g_col in g_cols:
        select_terms.append(f"g.{g_col}")
    for col in passthrough_g_cols:
        if _has_column(con, "ipeds_unit_growth", col):
            select_terms.append(f"g.{col}")

    variant_map = [
        ("z_ct", "share_ck", "g_kt"),
        ("z_ct_raw_flow", "share_ck", "g_kt_raw_flow"),
        ("z_ct_ihmp_share", "share_ck", "g_kt_ihmp_share"),
        ("z_ct_event_pulse", "share_ck", "g_kt_event_pulse"),
        ("z_ct_flow_diff", "share_ck", "g_kt_flow_diff"),
        ("z_ct_flow_ar_resid", "share_ck", "g_kt_flow_ar_resid"),
        ("z_ct_flow_diff_cumulative", "share_ck", "g_kt_flow_diff_cumulative"),
        ("z_ct_flow_ar_resid_cumulative", "share_ck", "g_kt_flow_ar_resid_cumulative"),
        ("z_ct_common_base_level", "share_ck", "g_kt_common_base_level"),
        ("z_ct_common_base_asinh", "share_ck", "g_kt_common_base_asinh"),
        ("z_ct_event_step_dose", "share_ck", "g_kt_event_step_dose"),
        ("z_ct_v1_broad_step", "share_ck", "g_kt_v1_broad_step"),
        ("z_ct_v2_broad_cumulative", "share_ck", "g_kt_v2_broad_cumulative"),
        ("z_ct_v3_broad_predicted_opt", "share_ck", "g_kt_v3_broad_predicted_opt"),
        ("z_ct_v4_matched_step", "share_ck", "g_kt_v4_matched_step"),
        ("z_ct_v5_matched_pulse", "share_ck", "g_kt_v5_matched_pulse"),
        ("z_ct_v6_broad_composition", "share_ck", "g_kt_v6_broad_composition"),
        ("z_ct_v7_matched_pulse_growth_rate", "share_ck", "g_kt_v7_matched_pulse_growth_rate"),
        ("z_ct_falsification_lead4_broad", "share_ck", "g_kt_falsification_lead4_broad"),
        ("z_ct_falsification_lead4_matched", "share_ck", "g_kt_falsification_lead4_matched"),
        ("z_ct_share_2008_2010", "share_ck_2008_2010", "g_kt"),
        ("z_ct_share_2011_2013", "share_ck_2011_2013", "g_kt"),
        ("z_ct_share_2008_2013", "share_ck_2008_2013", "g_kt"),
        ("z_ct_full", "share_ck_full", "g_kt"),
    ]
    component_terms: list[str] = []
    component_names: list[str] = []
    for instrument_col, share_col, g_col in variant_map:
        if share_col in share_cols and g_col in g_cols:
            comp_col = instrument_col.replace("z_ct", "z_ct_component", 1)
            component_terms.append(
                f"CASE WHEN s.{share_col} > 1 THEN NULL ELSE s.{share_col} END * g.{g_col} AS {comp_col}"
            )
            component_names.append(comp_col)
    select_terms.extend(component_terms)

    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW instrument_components AS
        SELECT
            {", ".join(select_terms)}
        FROM transition_shares s
        JOIN ipeds_unit_growth g ON s.k = g.k
        WHERE g.g_kt IS NOT NULL
    """)

    panel_terms = ["c", "t"]
    count_terms = []
    for instrument_col, share_col, g_col in variant_map:
        comp_col = instrument_col.replace("z_ct", "z_ct_component", 1)
        if comp_col not in component_names:
            continue
        panel_terms.append(f"SUM({comp_col}) AS {instrument_col}")
        count_col = "n_universities" if instrument_col == "z_ct" else f"n_universities_{instrument_col.replace('z_ct_', '')}"
        count_terms.append(
            f"COUNT(DISTINCT CASE WHEN {share_col} IS NOT NULL AND {share_col} > 0 "
            f"AND {g_col} IS NOT NULL AND {g_col} != 0 THEN k END) AS {count_col}"
        )
    panel_terms.extend(count_terms)
    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW instrument_panel AS
        SELECT
            {", ".join(panel_terms)}
        FROM instrument_components
        GROUP BY c, t
    """)
    n = con.sql("SELECT COUNT(DISTINCT c) FROM instrument_panel").fetchone()[0]
    yr = con.sql("SELECT MIN(t), MAX(t) FROM instrument_panel").fetchone()
    avg_z = con.sql("SELECT AVG(z_ct) FROM instrument_panel WHERE z_ct IS NOT NULL").fetchone()[0]
    print(f"[instrument] {n:,} companies | years {yr[0]}-{yr[1]} | mean z_ct={avg_z:.4f}")


# =============================================================================
# ── SECTION 2e: TREATMENT (OPT hires) ─────────────────────────────────────────
# =============================================================================

def _build_treatment(
    con: ddb.DuckDBPyConnection,
    include_non_masters: bool,
    include_bachelors: bool,
) -> None:
    """
    Build treatment variable: count of Master's OPT hires per company-year.
    Computes all three variants internally; analysis panel exposes only the selected one.
    """
    degree_clause = ""
    if not include_non_masters:
        deg_pred = _degree_predicate(con, view="foia_raw", include_bachelors=include_bachelors)
        if deg_pred:
            degree_clause = f"AND {deg_pred}"

    # Raw and valid hires (from foia_raw_full, no employer correction).
    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW foia_opt_authorizations_old AS
        SELECT person_id,
            {_sql_clean_company_name('employer_name')} AS f1_empname_clean,
            {_sql_normalize('employer_city')} AS f1_city_clean,
            {_sql_state_name_to_abbr('employer_state')} AS f1_state_clean,
            {_sql_clean_zip('employer_zip_code')} AS f1_zip_clean,
            MIN(EXTRACT(YEAR FROM program_end_date)) AS gradyear,
            MAX(CASE WHEN opt_employer_start_date >= program_end_date THEN 1 ELSE 0 END) AS valid_opt_hire
        FROM foia_raw_full
        WHERE employer_name IS NOT NULL {degree_clause}
        GROUP BY person_id, employer_name, employer_city, employer_state, employer_zip_code
    """)

    # Correction-aware: use first-spell records for post-2014 persons, fallback for others.
    con.sql("""
        CREATE OR REPLACE TEMP VIEW person_post2014_correction_status AS
        SELECT f.person_id,
            MAX(CASE WHEN f.year_int >= 2015 AND c.original_row_num IS NULL THEN 1 ELSE 0 END)
                AS has_post2014_correction
        FROM foia_raw_full f
        LEFT JOIN foia_raw c ON f.original_row_num = c.original_row_num
        WHERE f.person_id IS NOT NULL GROUP BY f.person_id
    """)
    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW foia_opt_authorizations_first_spell AS
        WITH base AS (
            SELECT person_id,
                {_sql_clean_company_name('employer_name')} AS f1_empname_clean,
                {_sql_normalize('employer_city')} AS f1_city_clean,
                {_sql_state_name_to_abbr('employer_state')} AS f1_state_clean,
                {_sql_clean_zip('employer_zip_code')} AS f1_zip_clean,
                EXTRACT(YEAR FROM program_end_date) AS gradyear,
                CASE WHEN opt_employer_start_date >= program_end_date THEN 1 ELSE 0 END AS valid_opt_hire,
                COALESCE({_date_parse_sql('opt_employer_start_date')},
                         {_date_parse_sql('opt_authorization_start_date')},
                         {_date_parse_sql('authorization_start_date')}) AS spell_start_dt,
                original_row_num
            FROM foia_raw WHERE employer_name IS NOT NULL {degree_clause}
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY person_id, f1_empname_clean, f1_city_clean, f1_state_clean, f1_zip_clean
                ORDER BY spell_start_dt ASC NULLS LAST, original_row_num ASC
            ) AS spell_rank FROM base
        )
        SELECT person_id, f1_empname_clean, f1_city_clean, f1_state_clean, f1_zip_clean,
               gradyear, valid_opt_hire FROM ranked WHERE spell_rank = 1
    """)
    con.sql("""
        CREATE OR REPLACE TEMP VIEW foia_opt_authorizations_correction_aware AS
        SELECT o.*, 'old_fallback' AS correction_source
        FROM foia_opt_authorizations_old o
        LEFT JOIN person_post2014_correction_status pcs ON o.person_id = pcs.person_id
        WHERE COALESCE(pcs.has_post2014_correction, 0) = 0
        UNION ALL
        SELECT f.*, 'first_spell' AS correction_source
        FROM foia_opt_authorizations_first_spell f
        JOIN person_post2014_correction_status pcs ON f.person_id = pcs.person_id
        WHERE pcs.has_post2014_correction = 1
    """)
    con.sql("""
        CREATE OR REPLACE TEMP VIEW opt_new_hires_old AS
        SELECT cw.preferred_rcid AS c, gradyear::INT AS t,
            COUNT(DISTINCT person_id) AS masters_opt_hires,
            COUNT(DISTINCT CASE WHEN valid_opt_hire = 1 THEN person_id END) AS valid_masters_opt_hires
        FROM foia_opt_authorizations_old f
        JOIN employer_crosswalk cw ON f.f1_empname_clean = cw.f1_empname_clean
          AND f.f1_city_clean = cw.f1_city_clean AND f.f1_state_clean = cw.f1_state_clean
          AND f.f1_zip_clean = cw.f1_zip_clean
        WHERE gradyear IS NOT NULL AND cw.preferred_rcid IS NOT NULL
        GROUP BY cw.preferred_rcid, gradyear::INT
    """)
    con.sql("""
        CREATE OR REPLACE TEMP VIEW opt_new_hires_correction_aware AS
        SELECT cw.preferred_rcid AS c, gradyear::INT AS t,
            COUNT(DISTINCT CASE WHEN valid_opt_hire = 1 THEN person_id END)
                AS masters_opt_hires_correction_aware
        FROM foia_opt_authorizations_correction_aware f
        JOIN employer_crosswalk cw ON f.f1_empname_clean = cw.f1_empname_clean
          AND f.f1_city_clean = cw.f1_city_clean AND f.f1_state_clean = cw.f1_state_clean
          AND f.f1_zip_clean = cw.f1_zip_clean
        WHERE gradyear IS NOT NULL AND cw.preferred_rcid IS NOT NULL
        GROUP BY cw.preferred_rcid, gradyear::INT
    """)
    # Merge all three variants into a single view.
    con.sql("""
        CREATE OR REPLACE TEMP VIEW opt_new_hires AS
        WITH base AS (
            SELECT COALESCE(o.c, n.c) AS c, COALESCE(o.t, n.t) AS t,
                COALESCE(o.masters_opt_hires, 0) AS masters_opt_hires,
                COALESCE(o.valid_masters_opt_hires, 0) AS valid_masters_opt_hires,
                COALESCE(n.masters_opt_hires_correction_aware, 0) AS masters_opt_hires_correction_aware
            FROM opt_new_hires_old o
            FULL OUTER JOIN opt_new_hires_correction_aware n ON o.c = n.c AND o.t = n.t
        ),
        bounds AS (SELECT c, MIN(t) AS min_t, MAX(t) AS max_t FROM base GROUP BY c),
        expanded AS (SELECT b.c, gs.year AS t FROM bounds b, LATERAL generate_series(b.min_t, b.max_t) AS gs(year)),
        filled AS (
            SELECT e.c, e.t,
                COALESCE(b.masters_opt_hires, 0) AS masters_opt_hires,
                COALESCE(b.valid_masters_opt_hires, 0) AS valid_masters_opt_hires,
                COALESCE(b.masters_opt_hires_correction_aware, 0) AS masters_opt_hires_correction_aware
            FROM expanded e LEFT JOIN base b ON e.c = b.c AND e.t = b.t
        )
        SELECT c, t, masters_opt_hires, valid_masters_opt_hires, masters_opt_hires_correction_aware
        FROM filled WHERE c IS NOT NULL AND t IS NOT NULL
    """)
    n = con.sql("SELECT COUNT(DISTINCT c) FROM opt_new_hires").fetchone()[0]
    total = con.sql("SELECT SUM(masters_opt_hires_correction_aware) FROM opt_new_hires").fetchone()[0]
    print(f"[treatment] {n:,} companies | total correction-aware OPT hires across all years: {total:,.0f}")


# =============================================================================
# ── SECTION 2f: OUTCOMES (y lags) ─────────────────────────────────────────────
# =============================================================================

def _build_outcomes(
    con: ddb.DuckDBPyConnection, lag_start: int, lag_end: int, use_changes: bool
) -> None:
    """Build firm outcomes with lags relative to hire year t."""
    def _sfx(lag: int) -> str:
        return f"m{abs(lag)}" if lag < 0 else str(lag)

    count_metrics = [
        "y_cst",
        "y_new_hires",
        "y_new_hires_foreign",
        "y_new_hires_native",
        "y_new_hires_foreign_hard",
        "y_new_hires_native_hard",
    ]
    continuous_metrics = ["avg_tenure_years"]
    metric_cols = count_metrics + continuous_metrics

    def _metric_expr(metric: str) -> str:
        if not use_changes:
            return metric
        return f"ASINH({metric}) - ASINH({metric}_lag)"

    lag_select_cols = ",\n                ".join(
        f"LAG({metric}) OVER (PARTITION BY c ORDER BY outcome_year) AS {metric}_lag"
        for metric in metric_cols
    )
    long_select_cols = ",\n            ".join(
        f"{_metric_expr(metric)} AS {metric}" for metric in metric_cols
    )
    wide_count_cols = ",\n            ".join(
        f"COALESCE(MAX(CASE WHEN lag = {lag} THEN {metric} END), 0) AS {metric}_lag{_sfx(lag)}"
        for metric in count_metrics
        for lag in range(lag_start, lag_end + 1)
    )
    wide_continuous_cols = ",\n            ".join(
        f"MAX(CASE WHEN lag = {lag} THEN {metric} END) AS {metric}_lag{_sfx(lag)}"
        for metric in continuous_metrics
        for lag in range(lag_start, lag_end + 1)
    )
    wide_cols = ",\n            ".join([c for c in [wide_count_cols, wide_continuous_cols] if c])

    has_workforce = _has_column(con, "wrds_company_year_workforce", "c") and _has_column(
        con, "wrds_company_year_workforce", "t"
    )
    if has_workforce:
        con.sql("""
            CREATE OR REPLACE TEMP VIEW outcome_workforce_metrics AS
            SELECT
                CAST(c AS INTEGER) AS c,
                CAST(t AS INTEGER) AS outcome_year,
                COALESCE(CAST(n_new_hires_wrds_annual AS DOUBLE), 0) AS y_new_hires,
                COALESCE(CAST(n_new_hires_foreign_weighted_annual AS DOUBLE), 0) AS y_new_hires_foreign,
                COALESCE(CAST(n_new_hires_native_weighted_annual AS DOUBLE), 0) AS y_new_hires_native,
                COALESCE(CAST(n_new_hires_foreign_hard_annual AS DOUBLE), 0) AS y_new_hires_foreign_hard,
                COALESCE(CAST(n_new_hires_native_hard_annual AS DOUBLE), 0) AS y_new_hires_native_hard,
                CAST(avg_tenure_years_annual AS DOUBLE) AS avg_tenure_years
            FROM wrds_company_year_workforce
            WHERE c IN (SELECT rcid FROM matched_rcids) AND t IS NOT NULL
        """)
        print("[outcomes] using WRDS company-year workforce cache for new-hire splits and tenure.")
    else:
        con.sql("""
            CREATE OR REPLACE TEMP VIEW outcome_workforce_metrics AS
            SELECT
                CAST(rcid AS INTEGER) AS c,
                CAST(year AS INTEGER) AS outcome_year,
                COALESCE(MAX(CAST(total_new_hires AS DOUBLE)), 0) AS y_new_hires,
                0.0 AS y_new_hires_foreign,
                0.0 AS y_new_hires_native,
                0.0 AS y_new_hires_foreign_hard,
                0.0 AS y_new_hires_native_hard,
                NULL::DOUBLE AS avg_tenure_years
            FROM revelio_transitions
            WHERE rcid IN (SELECT rcid FROM matched_rcids)
              AND year IS NOT NULL AND total_new_hires IS NOT NULL
            GROUP BY c, outcome_year
        """)

    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW outcomes_long AS
        WITH hc AS (
            SELECT CAST(rcid AS INTEGER) AS c, CAST(year AS INTEGER) AS outcome_year,
                COALESCE(MAX(CAST(total_headcount AS DOUBLE)), 0) AS y_cst
            FROM revelio_headcount WHERE rcid IN (SELECT rcid FROM matched_rcids) AND year IS NOT NULL
            GROUP BY c, outcome_year
        ),
        joined AS (
            SELECT
                COALESCE(h.c, w.c) AS c,
                COALESCE(h.outcome_year, w.outcome_year) AS outcome_year,
                COALESCE(h.y_cst, 0) AS y_cst,
                COALESCE(w.y_new_hires, 0) AS y_new_hires,
                COALESCE(w.y_new_hires_foreign, 0) AS y_new_hires_foreign,
                COALESCE(w.y_new_hires_native, 0) AS y_new_hires_native,
                COALESCE(w.y_new_hires_foreign_hard, 0) AS y_new_hires_foreign_hard,
                COALESCE(w.y_new_hires_native_hard, 0) AS y_new_hires_native_hard,
                w.avg_tenure_years AS avg_tenure_years
            FROM hc h
            FULL OUTER JOIN outcome_workforce_metrics w
              ON h.c = w.c AND h.outcome_year = w.outcome_year
        ),
        bounds AS (SELECT c, MIN(outcome_year) AS min_y, MAX(outcome_year) AS max_y FROM joined GROUP BY c),
        expanded AS (SELECT b.c, gs.year AS outcome_year FROM bounds b, LATERAL generate_series(b.min_y, b.max_y) AS gs(year)),
        filled AS (
            SELECT e.c, e.outcome_year,
                COALESCE(b.y_cst, 0) AS y_cst,
                COALESCE(b.y_new_hires, 0) AS y_new_hires,
                COALESCE(b.y_new_hires_foreign, 0) AS y_new_hires_foreign,
                COALESCE(b.y_new_hires_native, 0) AS y_new_hires_native,
                COALESCE(b.y_new_hires_foreign_hard, 0) AS y_new_hires_foreign_hard,
                COALESCE(b.y_new_hires_native_hard, 0) AS y_new_hires_native_hard,
                b.avg_tenure_years AS avg_tenure_years
            FROM expanded e LEFT JOIN joined b ON e.c = b.c AND e.outcome_year = b.outcome_year
        ),
        with_lags AS (
            SELECT c, outcome_year, {", ".join(metric_cols)},
                {lag_select_cols}
            FROM filled
        )
        SELECT c, outcome_year AS s, outcome_year - lag.lag AS t,
            {long_select_cols}, lag.lag AS lag
        FROM with_lags
        CROSS JOIN LATERAL generate_series({lag_start}, {lag_end}) AS lag(lag)
    """)

    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW outcomes_wide AS
        SELECT c, t, {wide_cols}
        FROM outcomes_long WHERE t IS NOT NULL GROUP BY c, t
    """)


# =============================================================================
# ── SECTION 2g: ANALYSIS PANEL ASSEMBLY ───────────────────────────────────────
# =============================================================================

def _build_analysis_panel(
    con: ddb.DuckDBPyConnection,
    lag_start: int,
    lag_end: int,
    use_log_y: bool,
    panel_year_min: int = 2008,
    panel_year_max: int = 2022,
    conditioning_baseline_window_start: int = 2008,
    conditioning_baseline_window_end: int = 2013,
) -> None:
    """Merge outcomes, treatment, instrument, and company state into final panel."""
    def _sfx(lag: int) -> str:
        return f"m{abs(lag)}" if lag < 0 else str(lag)

    def _headcount_lag_expr(lag: int) -> str:
        if lag_start <= lag <= lag_end:
            return f"COALESCE(o.y_cst_lag{_sfx(lag)}, 0)"
        return "0.0"

    outcome_metrics = [
        "y_cst",
        "y_new_hires",
        "y_new_hires_foreign",
        "y_new_hires_native",
        "y_new_hires_foreign_hard",
        "y_new_hires_native_hard",
        "avg_tenure_years",
    ]
    count_outcome_metrics = {
        "y_cst",
        "y_new_hires",
        "y_new_hires_foreign",
        "y_new_hires_native",
        "y_new_hires_foreign_hard",
        "y_new_hires_native_hard",
    }

    def _outcome_select_expr(metric: str, lag: int) -> str:
        col = f"{metric}_lag{_sfx(lag)}"
        if _has_column(con, "outcomes_wide", col):
            return f"ASINH(o.{col}) AS {col}" if use_log_y else f"o.{col}"
        default = "0.0" if metric in count_outcome_metrics else "NULL::DOUBLE"
        return f"{default} AS {col}"

    y_cols = ",\n            ".join(
        _outcome_select_expr(metric, lag)
        for metric in outcome_metrics
        for lag in range(lag_start, lag_end + 1)
    )
    instrument_cols = [
        "z_ct",
        "z_ct_raw_flow",
        "z_ct_ihmp_share",
        "z_ct_event_pulse",
        "z_ct_flow_diff",
        "z_ct_flow_ar_resid",
        "z_ct_flow_diff_cumulative",
        "z_ct_flow_ar_resid_cumulative",
        "z_ct_common_base_level",
        "z_ct_common_base_asinh",
        "z_ct_event_step_dose",
        "z_ct_v1_broad_step",
        "z_ct_v2_broad_cumulative",
        "z_ct_v3_broad_predicted_opt",
        "z_ct_v4_matched_step",
        "z_ct_v5_matched_pulse",
        "z_ct_v6_broad_composition",
        "z_ct_v7_matched_pulse_growth_rate",
        "z_ct_falsification_lead4_broad",
        "z_ct_falsification_lead4_matched",
        "z_ct_share_2008_2010",
        "z_ct_share_2011_2013",
        "z_ct_share_2008_2013",
        "z_ct_full",
        "n_universities",
        "n_universities_raw_flow",
        "n_universities_ihmp_share",
        "n_universities_event_pulse",
        "n_universities_flow_diff",
        "n_universities_flow_ar_resid",
        "n_universities_flow_diff_cumulative",
        "n_universities_flow_ar_resid_cumulative",
        "n_universities_common_base_level",
        "n_universities_common_base_asinh",
        "n_universities_event_step_dose",
        "n_universities_v1_broad_step",
        "n_universities_v2_broad_cumulative",
        "n_universities_v3_broad_predicted_opt",
        "n_universities_v4_matched_step",
        "n_universities_v5_matched_pulse",
        "n_universities_v6_broad_composition",
        "n_universities_v7_matched_pulse_growth_rate",
        "n_universities_falsification_lead4_broad",
        "n_universities_falsification_lead4_matched",
        "n_universities_share_2008_2010",
        "n_universities_share_2011_2013",
        "n_universities_share_2008_2013",
        "n_universities_full",
    ]
    instr_select = ",\n            ".join(
        f"instr.{col}" for col in instrument_cols if _has_column(con, "instrument_panel", col)
    )
    year_min = int(panel_year_min)
    year_max = int(panel_year_max)
    cond_start = int(conditioning_baseline_window_start)
    cond_end = int(conditioning_baseline_window_end)
    headcount_lag0_expr = _headcount_lag_expr(0)
    headcount_lag1_expr = _headcount_lag_expr(1)
    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW analysis_panel AS
        SELECT o.c, o.t,
            {y_cols},
            {headcount_lag0_expr} AS headcount_lag0_raw,
            {headcount_lag1_expr} AS headcount_lag1_raw,
            AVG(
                CASE
                    WHEN o.t BETWEEN {cond_start} AND {cond_end} THEN {headcount_lag0_expr}
                    ELSE NULL
                END
            ) OVER (PARTITION BY o.c) AS headcount_size_baseline,
            ASINH({headcount_lag0_expr}) - ASINH({headcount_lag1_expr}) AS headcount_growth_asinh,
            COALESCE(x.masters_opt_hires, 0) AS masters_opt_hires,
            COALESCE(x.valid_masters_opt_hires, 0) AS valid_masters_opt_hires,
            COALESCE(x.masters_opt_hires_correction_aware, 0) AS masters_opt_hires_correction_aware,
            {instr_select},
            st.company_state
        FROM (SELECT * FROM outcomes_wide WHERE t BETWEEN {year_min} AND {year_max}) o
        LEFT JOIN (SELECT * FROM opt_new_hires WHERE t BETWEEN {year_min} AND {year_max}) x USING (c, t)
        LEFT JOIN (SELECT * FROM instrument_panel WHERE t BETWEEN {year_min} AND {year_max}) instr USING (c, t)
        LEFT JOIN (
            WITH sc AS (
                SELECT CAST(preferred_rcid AS INTEGER) AS c,
                    UPPER(TRIM(CAST(f1_state_clean AS VARCHAR))) AS company_state,
                    COUNT(*) AS n
                FROM employer_crosswalk
                WHERE preferred_rcid IS NOT NULL AND f1_state_clean IS NOT NULL
                  AND TRIM(CAST(f1_state_clean AS VARCHAR)) <> ''
                GROUP BY 1, 2
            ),
            ranked AS (SELECT c, company_state, ROW_NUMBER() OVER (PARTITION BY c ORDER BY n DESC, company_state) AS rn FROM sc)
            SELECT c, company_state FROM ranked WHERE rn = 1
        ) st USING (c)
        WHERE o.t IS NOT NULL
    """)
    n_rows = con.sql("SELECT COUNT(*) FROM analysis_panel").fetchone()[0]
    n_cos = con.sql("SELECT COUNT(DISTINCT c) FROM analysis_panel").fetchone()[0]
    yr = con.sql("SELECT MIN(t), MAX(t) FROM analysis_panel").fetchone()
    n_with_z = con.sql("SELECT COUNT(*) FROM analysis_panel WHERE z_ct IS NOT NULL").fetchone()[0]
    print(f"[panel] {n_rows:,} rows | {n_cos:,} companies | years {yr[0]}-{yr[1]} | {n_with_z:,} rows with z_ct")


def _apply_analysis_sample_restrictions(
    con: ddb.DuckDBPyConnection,
    sample_year_min: int,
    sample_year_max: int,
    min_active_shock_schools: int = 1,
    require_balanced_panel: bool = False,
) -> None:
    min_year = int(sample_year_min)
    max_year = int(sample_year_max)
    expected_years = max_year - min_year + 1
    if expected_years <= 0:
        raise ValueError("Balanced sample window must have nonnegative length.")
    min_active = max(1, int(min_active_shock_schools))

    active_filter_sql = ""
    if min_active > 1:
        active_filter_sql = f"""
        ,
        active_eligible AS (
            SELECT c
            FROM base_window
            GROUP BY c
            HAVING
                MIN(
                    CASE
                        WHEN COALESCE(ABS(z_ct), 0) > 0 THEN
                            CASE WHEN COALESCE(n_universities, 0) >= {min_active} THEN 1 ELSE 0 END
                        ELSE 1
                    END
                ) = 1
        )
        """
        eligible_join = "JOIN active_eligible ae USING (c)"
    else:
        eligible_join = ""

    balanced_cte = ""
    balanced_join = ""
    if require_balanced_panel:
        balanced_cte = f"""
        ,
        balanced_eligible AS (
            SELECT c
            FROM base_window
            {eligible_join}
            GROUP BY c
            HAVING COUNT(DISTINCT t) = {expected_years}
        )
        """
        balanced_join = "JOIN balanced_eligible be USING (c)"

    con.sql("""
        CREATE OR REPLACE TEMP TABLE analysis_panel_unrestricted_tbl AS
        SELECT * FROM analysis_panel
    """)
    con.sql(f"""
        CREATE OR REPLACE TEMP VIEW analysis_panel AS
        WITH base_window AS (
            SELECT *
            FROM analysis_panel_unrestricted_tbl
            WHERE CAST(t AS INTEGER) BETWEEN {min_year} AND {max_year}
        )
        {active_filter_sql}
        {balanced_cte}
        SELECT bw.*
        FROM base_window bw
        {eligible_join}
        {balanced_join}
    """)
    n_rows = con.sql("SELECT COUNT(*) FROM analysis_panel").fetchone()[0]
    n_cos = con.sql("SELECT COUNT(DISTINCT c) FROM analysis_panel").fetchone()[0]
    yr = con.sql("SELECT MIN(t), MAX(t) FROM analysis_panel").fetchone()

    if _has_column(con, "instrument_panel", "c") and _has_column(con, "instrument_panel", "t"):
        con.sql("""
            CREATE OR REPLACE TEMP TABLE instrument_panel_unrestricted_tbl AS
            SELECT * FROM instrument_panel
        """)
        con.sql("""
            CREATE OR REPLACE TEMP VIEW instrument_panel AS
            SELECT ip.*
            FROM instrument_panel_unrestricted_tbl ip
            JOIN (SELECT DISTINCT c, t FROM analysis_panel) keep USING (c, t)
        """)
    if _has_column(con, "instrument_components", "c") and _has_column(con, "instrument_components", "t"):
        con.sql("""
            CREATE OR REPLACE TEMP TABLE instrument_components_unrestricted_tbl AS
            SELECT * FROM instrument_components
        """)
        con.sql("""
            CREATE OR REPLACE TEMP VIEW instrument_components AS
            SELECT ic.*
            FROM instrument_components_unrestricted_tbl ic
            JOIN (SELECT DISTINCT c, t FROM analysis_panel) keep USING (c, t)
        """)
    if _has_column(con, "transition_shares", "c"):
        con.sql("""
            CREATE OR REPLACE TEMP TABLE transition_shares_unrestricted_tbl AS
            SELECT * FROM transition_shares
        """)
        con.sql("""
            CREATE OR REPLACE TEMP VIEW transition_shares AS
            SELECT ts.*
            FROM transition_shares_unrestricted_tbl ts
            JOIN (SELECT DISTINCT c FROM analysis_panel) keep USING (c)
        """)

    print(
        f"[panel_sample] rows={n_rows:,} | companies={n_cos:,} | years {yr[0]}-{yr[1]} | "
        f"min_active_shock_schools={min_active} | require_balanced_panel={require_balanced_panel}"
    )


# =============================================================================
# ── SECTION 4: DIAGNOSTICS ────────────────────────────────────────────────────
# =============================================================================

def run_diagnostics(
    con: ddb.DuckDBPyConnection,
    analysis_panel_df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
) -> None:
    """
    Generate and save diagnostic plots:
      4a. school_shift_metric by year for treated vs control schools.
      4b. Firm samples by treated-school event year, with top firms selected by treated/control share.
      4c. Distributions of key inputs: g_kt, share_ck, z_ct, x_ct, y.
    """
    school_sample_mode = _normalize_school_sample_mode(cfg.get("school_sample_mode", "matched_shift_sample"))
    treatment_measure = cfg.get("treatment_measure", "correction_aware")
    x_col_map = {
        "raw": "masters_opt_hires",
        "valid": "valid_masters_opt_hires",
        "correction_aware": "masters_opt_hires_correction_aware",
    }
    x_col = x_col_map.get(treatment_measure, "masters_opt_hires_correction_aware")
    firm_top_n_per_year = int(cfg.get("firm_share_event_year_top_n", cfg.get("firm_share_size_bin_top_n", 8)) or 8)
    firm_top_n_per_year = max(1, firm_top_n_per_year)
    firm_y_metric = str(cfg.get("firm_share_y_metric", "y_cst_lag0"))
    firm_bins_enabled = bool(cfg.get("firm_share_size_bin_diagnostic", True))
    diag_dir = out_dir
    diag_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    diag_school_names = _diagnostic_school_name_lookup(con)
    con.register("_diag_school_names_df", diag_school_names)
    con.sql("CREATE OR REPLACE TEMP VIEW diag_school_names AS SELECT * FROM _diag_school_names_df")

    # ── 4a: school event-time diagnostics ────────────────────────────────────
    try:
        primary_metric = _normalize_school_shift_metric(cfg.get("school_shift_metric", "ihmp_share"))
        metrics_to_plot = ["ihmp_share", "international_share", "opt_share"]
        selected_schools = con.sql("""
            SELECT CAST(k AS VARCHAR) AS k, sample_role, matched_treated_event_year
            FROM school_shift_sample
            WHERE selected_for_instrument = 1 AND sample_role IS NOT NULL
        """).df()
        if selected_schools.empty:
            print("[diag] school event-time plots skipped: no matched schools with sample_role")
        else:
            include_bachelors = bool(cfg.get("include_bachelors", False))
            degree_scope_for_diag = "bachelors_masters" if include_bachelors else "masters"
            exclude_unitids = _parse_unitids(cfg.get("exclude_unitids"))
            for metric_name in metrics_to_plot:
                panel = _build_school_metric_panel(
                    con,
                    metric=metric_name,
                    degree_scope=degree_scope_for_diag,
                    exclude_unitids=exclude_unitids,
                )
                metric_df = panel.merge(selected_schools, on="k", how="inner")
                metric_df["t"] = pd.to_numeric(metric_df["t"], errors="coerce")
                metric_df["matched_treated_event_year"] = pd.to_numeric(
                    metric_df["matched_treated_event_year"], errors="coerce"
                )
                metric_df["event_time"] = metric_df["t"] - metric_df["matched_treated_event_year"]
                metric_df = metric_df.loc[
                    metric_df["event_time"].notna()
                    & metric_df["event_time"].between(-4, 5)
                ].copy()
                if metric_df.empty:
                    continue

                agg = (
                    metric_df.groupby(["event_time", "sample_role"], as_index=False)["metric_share"]
                    .mean()
                )
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.lineplot(
                    data=agg,
                    x="event_time",
                    y="metric_share",
                    hue="sample_role",
                    marker="o",
                    ax=ax,
                )
                ax.axvline(0, color="black", linestyle="--", linewidth=1)
                ax.set_xlabel("Event time")
                ax.set_ylabel(f"Mean {metric_name}")
                ax.set_title(f"School event-time path: {metric_name}")
                fig.tight_layout()
                output_name = f"school_event_time__{metric_name}.png"
                fig.savefig(diag_dir / output_name, dpi=150)
                plt.show()
                print(f"[diag] saved {output_name}")

                if metric_name == primary_metric:
                    extra_specs = [
                        ("metric_level", "school_event_time__primary_metric_level.png", "IHMP seats"),
                        ("school_size", "school_event_time__school_size.png", "Total masters size"),
                    ]
                elif metric_name == "opt_share":
                    extra_specs = [
                        ("metric_level", "school_event_time__opt_count.png", "FOIA OPT count"),
                    ]
                else:
                    extra_specs = []

                for value_col, out_name, label in extra_specs:
                    if value_col not in metric_df.columns:
                        continue
                    extra_agg = (
                        metric_df.groupby(["event_time", "sample_role"], as_index=False)[value_col]
                        .mean()
                    )
                    fig2, ax2 = plt.subplots(figsize=(9, 5))
                    sns.lineplot(
                        data=extra_agg,
                        x="event_time",
                        y=value_col,
                        hue="sample_role",
                        marker="o",
                        ax=ax2,
                    )
                    ax2.axvline(0, color="black", linestyle="--", linewidth=1)
                    ax2.set_xlabel("Event time")
                    ax2.set_ylabel(label)
                    ax2.set_title(f"School event-time path: {label}")
                    fig2.tight_layout()
                    fig2.savefig(diag_dir / out_name, dpi=150)
                    plt.show()
                    print(f"[diag] saved {out_name}")
    except Exception as e:
        print(f"[diag] school event-time plots skipped: {e}")

    try:
        _run_raw_exposure_proof_plots(
            con=con,
            analysis_panel_df=analysis_panel_df,
            diag_dir=diag_dir,
            x_col=x_col,
        )
    except Exception as e:
        print(f"[diag] raw exposure proof plots skipped: {e}")

    # ── 4b: Firm samples by treated-school event year ────────────────────────
    if firm_bins_enabled:
        try:
            if x_col not in analysis_panel_df.columns:
                raise ValueError(f"x column '{x_col}' not found in panel.")
            if firm_y_metric not in analysis_panel_df.columns:
                raise ValueError(f"y metric '{firm_y_metric}' not found in panel.")

            uni_role_count = con.sql("""
                SELECT COUNT(*) AS n
                FROM school_shift_sample
                WHERE selected_for_instrument = 1 AND sample_role IN ('treated', 'control')
            """).fetchone()[0]
            if (uni_role_count or 0) == 0:
                raise ValueError("No treated/control school assignments found.")

            event_year_counts = con.sql("""
                SELECT CAST(matched_treated_event_year AS INTEGER) AS event_year,
                       COUNT(*) AS n_treated_events
                FROM school_shift_sample
                WHERE selected_for_instrument = 1
                  AND sample_role = 'treated'
                  AND matched_treated_event_year IS NOT NULL
                GROUP BY 1
                ORDER BY 1
            """).df()
            if event_year_counts.empty:
                print("[diag] No treated-school event years available for firm diagnostics.")
            else:
                event_year_count_msg = ", ".join(
                    [f"{int(r['event_year'])}: {int(r['n_treated_events'])}" for _, r in event_year_counts.iterrows()]
                )
                print(f"[diag] Firm diagnostics: treated-school events by year -> {event_year_count_msg}")

            firm_share_sql = f"""
                WITH uni_roles AS (
                    SELECT CAST(k AS VARCHAR) AS k_role, sample_role, matched_treated_event_year
                    FROM school_shift_sample
                    WHERE selected_for_instrument = 1
                      AND sample_role IN ('treated', 'control')
                      AND matched_treated_event_year IS NOT NULL
                )
                SELECT
                    CAST(s.c AS VARCHAR) AS c,
                    CAST(ur.matched_treated_event_year AS INTEGER) AS event_year,
                    SUM(CASE WHEN ur.sample_role = 'treated' THEN COALESCE(s.share_ck, 0) ELSE 0 END) AS treated_share,
                    SUM(CASE WHEN ur.sample_role = 'control' THEN COALESCE(s.share_ck, 0) ELSE 0 END) AS control_share
                FROM transition_shares s
                JOIN uni_roles ur ON CAST(s.k AS VARCHAR) = ur.k_role
                GROUP BY s.c, ur.matched_treated_event_year
            """
            firm_share_df = con.sql(firm_share_sql).df()
            if firm_share_df.empty:
                raise ValueError("No firms in transition shares to evaluate.")

            panel = analysis_panel_df.copy()
            panel["c"] = panel["c"].astype("string")
            panel["t"] = pd.to_numeric(panel["t"], errors="coerce")
            firm_pool = firm_share_df.copy()
            firm_pool["c"] = firm_pool["c"].astype("string")
            firm_pool["event_year"] = pd.to_numeric(firm_pool["event_year"], errors="coerce").astype("Int64")
            firm_pool = firm_pool.loc[firm_pool["c"].notna() & firm_pool["event_year"].notna()].copy()
            if firm_pool.empty:
                raise ValueError("No firms overlap between shares and sampled schools.")

            top_treated = (
                firm_pool.sort_values(["event_year", "treated_share", "c"], ascending=[True, False, True])
                .groupby("event_year", group_keys=False)
                .head(firm_top_n_per_year)
                .assign(selection_type="treated")[["c", "event_year", "selection_type"]]
            )
            top_control = (
                firm_pool.sort_values(["event_year", "control_share", "c"], ascending=[True, False, True])
                .groupby("event_year", group_keys=False)
                .head(firm_top_n_per_year)
                .assign(selection_type="control")[["c", "event_year", "selection_type"]]
            )
            selected_firms = pd.concat([top_treated, top_control], ignore_index=True)
            if selected_firms.empty:
                raise ValueError("No firms selected from event-year groups.")

            timeseries = panel.merge(selected_firms, on="c", how="inner")
            if timeseries.empty:
                raise ValueError("No firm-year observations for selected firms.")

            selected_event_counts = selected_firms.groupby(["event_year", "selection_type"], as_index=False).size()
            if not selected_event_counts.empty:
                print("[diag] Firm diagnostics: selected firm counts by event year")
                for ev, row in selected_event_counts.sort_values("event_year").groupby("event_year"):
                    treated_count = int(row.loc[row["selection_type"] == "treated", "size"].sum())
                    control_count = int(row.loc[row["selection_type"] == "control", "size"].sum())
                    total_count = treated_count + control_count
                    print(
                        f"[diag]   year={int(ev)} treated={treated_count} control={control_count} total={total_count}"
                    )

            firm_share_diagnostic_metrics = [
                ("z_ct", "diag_firms_by_event_year_z_ct"),
                (x_col, "diag_firms_by_event_year_x_ct"),
                (firm_y_metric, "diag_firms_by_event_year_y_ct"),
            ]
            for col, out_name in firm_share_diagnostic_metrics:
                if col not in timeseries.columns:
                    print(f"[diag] firm event-year plot skipped: missing '{col}'.")
                    continue
                series = (
                    timeseries.loc[
                        timeseries["t"].notna(),
                        ["t", "event_year", "selection_type", col],
                    ].groupby(["t", "event_year", "selection_type"], as_index=False)[col]
                    .mean()
                )
                if series.empty:
                    print(f"[diag] firm event-year plot skipped: no data for '{col}'.")
                    continue
                for event_year in sorted(series["event_year"].dropna().unique()):
                    event_df = series.loc[series["event_year"] == event_year]
                    if event_df.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.lineplot(
                        data=event_df,
                        x="t", y=col,
                        hue="selection_type",
                        marker="o", ax=ax
                    )
                    ax.set_xlabel("Year")
                    ax.set_ylabel(col)
                    ax.set_title(f"{col} over time by selected firms (event year {int(event_year)})")
                    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                    fig.tight_layout()
                    fig.savefig(diag_dir / f"{out_name}_event_year_{int(event_year)}.png", dpi=150)
                    plt.show()
                    print(f"[diag] saved {out_name}_event_year_{int(event_year)}.png")
        except Exception as e:
            print(f"[diag] firm event-year diagnostic skipped: {e}")

    # ── 4c: Key input distributions ──────────────────────────────────────────

    # g_kt distribution by year (sample of years)
    try:
        g_df = con.sql("""
            SELECT t, g_kt FROM ipeds_unit_growth WHERE t BETWEEN 2008 AND 2022 AND g_kt IS NOT NULL
        """).df()
        if not g_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sample_years = sorted(g_df["t"].unique())
            sample_years = sample_years[::max(1, len(sample_years) // 8)]  # at most 8 years
            sns.boxplot(
                data=g_df[g_df["t"].isin(sample_years)], x="t", y="g_kt",
                order=sample_years, showfliers=False, ax=ax
            )
            ax.set_xlabel("Year"); ax.set_ylabel("g_kt")
            ax.set_title("Growth metric (g_kt) by year — selected years")
            fig.tight_layout()
            fig.savefig(diag_dir / "dist_g_kt.png", dpi=150)
            plt.show()
    except Exception as e:
        print(f"[diag] g_kt plot skipped: {e}")

    # share_ck histogram
    try:
        sh_df = con.sql("""
            SELECT AVG(share_ck) AS avg_share FROM transition_shares GROUP BY c
        """).df()
        if not sh_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(sh_df["avg_share"].dropna(), bins=50, ax=ax)
            ax.set_xlabel("Mean share_ck per company"); ax.set_ylabel("Count")
            ax.set_title("Distribution of company-average share_ck")
            fig.tight_layout()
            fig.savefig(diag_dir / "dist_share_ck.png", dpi=150)
            plt.show()
    except Exception as e:
        print(f"[diag] share_ck plot skipped: {e}")

    # z_ct distribution by year
    try:
        z_df = con.sql("""
            SELECT t, z_ct FROM instrument_panel WHERE t BETWEEN 2008 AND 2022 AND z_ct IS NOT NULL
        """).df()
        if not z_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sample_years = sorted(z_df["t"].unique())
            sample_years = sample_years[::max(1, len(sample_years) // 8)]
            sns.boxplot(
                data=z_df[z_df["t"].isin(sample_years)], x="t", y="z_ct",
                order=sample_years, showfliers=False, ax=ax
            )
            ax.set_xlabel("Year"); ax.set_ylabel("z_ct")
            ax.set_title("Instrument (z_ct) distribution by year — selected years")
            fig.tight_layout()
            fig.savefig(diag_dir / "dist_z_ct.png", dpi=150)
            plt.show()
    except Exception as e:
        print(f"[diag] z_ct plot skipped: {e}")

    # Treatment x_ct histogram + x_bin share by year
    if x_col in analysis_panel_df.columns:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            nonzero = analysis_panel_df[analysis_panel_df[x_col] > 0][x_col]
            sns.histplot(nonzero, bins=40, ax=axes[0])
            axes[0].set_xlabel(x_col); axes[0].set_ylabel("Count")
            axes[0].set_title(f"Distribution of {x_col} (nonzero firm-years)")
            x_bin_by_year = (
                analysis_panel_df.assign(x_bin=(analysis_panel_df[x_col] > 0).astype(int))
                .groupby("t")["x_bin"].mean().reset_index()
            )
            axes[1].plot(x_bin_by_year["t"], x_bin_by_year["x_bin"], marker="o")
            axes[1].set_xlabel("Year"); axes[1].set_ylabel("Share with x_ct > 0")
            axes[1].set_title("Share of firms with OPT hires, by year")
            fig.tight_layout()
            fig.savefig(diag_dir / "dist_treatment.png", dpi=150)
            plt.show()
        except Exception as e:
            print(f"[diag] treatment plot skipped: {e}")

    # Outcome y_cst_lag0 histogram
    if "y_cst_lag0" in analysis_panel_df.columns:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(analysis_panel_df["y_cst_lag0"].clip(upper=analysis_panel_df["y_cst_lag0"].quantile(0.99)), bins=50, ax=ax)
            ax.set_xlabel("y_cst_lag0 (total headcount)"); ax.set_ylabel("Count")
            ax.set_title("Distribution of outcome y_cst_lag0 (clipped at 99th pct)")
            fig.tight_layout()
            fig.savefig(diag_dir / "dist_outcome.png", dpi=150)
            plt.show()
        except Exception as e:
            print(f"[diag] outcome plot skipped: {e}")

    # ── 4d: Shock decomposition and influence ───────────────────────────────
    try:
        school_shock_df = con.sql("""
            SELECT
                CAST(g.k AS VARCHAR) AS k,
                COALESCE(MAX(COALESCE(sn.school_name, '')), CAST(g.k AS VARCHAR)) AS school_name,
                MAX(COALESCE(ss.sample_role, '')) AS sample_role,
                MAX(COALESCE(ss.selected_for_instrument, 0)) AS selected_for_instrument,
                MAX(CAST(ss.treated_event_year AS INTEGER)) AS treated_event_year,
                MAX(COALESCE(g.delta_share_event, 0)) AS delta_share_event,
                MAX(COALESCE(g.pre_school_size_event, 0)) AS pre_school_size_event,
                MAX(COALESCE(g.pre_opt_rate_event, 0)) AS pre_opt_rate_event,
                MAX(COALESCE(g.g_kt, 0)) AS shock_quantity_step
            FROM ipeds_unit_growth g
            LEFT JOIN school_shift_sample ss ON CAST(g.k AS VARCHAR) = CAST(ss.k AS VARCHAR)
            LEFT JOIN diag_school_names sn ON CAST(g.k AS VARCHAR) = sn.k
            GROUP BY 1
        """).df()
        if not school_shock_df.empty:
            school_shock_df["shock_quantity_step"] = pd.to_numeric(
                school_shock_df["shock_quantity_step"], errors="coerce"
            ).fillna(0.0)
            school_shock_df["shock_quantity_predicted_opt"] = (
                pd.to_numeric(school_shock_df["pre_opt_rate_event"], errors="coerce").fillna(0.0)
                * pd.to_numeric(school_shock_df["pre_school_size_event"], errors="coerce").fillna(0.0)
                * pd.to_numeric(school_shock_df["delta_share_event"], errors="coerce").fillna(0.0)
            )
            school_shock_df.to_csv(diag_dir / "shock_decomposition_by_school.csv", index=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                data=school_shock_df,
                x="delta_share_event",
                y="pre_school_size_event",
                hue="sample_role",
                style="selected_for_instrument",
                ax=ax,
            )
            ax.set_xlabel("Delta share at event")
            ax.set_ylabel("Pre-event masters size")
            ax.set_title("Shock decomposition by school")
            fig.tight_layout()
            fig.savefig(diag_dir / "shock_decomposition_scatter.png", dpi=150)
            plt.show()
            print("[diag] saved shock_decomposition_by_school.csv and shock_decomposition_scatter.png")
    except Exception as e:
        print(f"[diag] shock decomposition skipped: {e}")

    try:
        firm_decomp = con.sql("""
            WITH ranked AS (
                SELECT
                    CAST(c AS VARCHAR) AS c,
                    CAST(t AS INTEGER) AS t,
                    CAST(k AS VARCHAR) AS k,
                    COALESCE(z_ct_component, 0) AS z_ct_component,
                    ABS(COALESCE(z_ct_component, 0)) AS abs_component,
                    ROW_NUMBER() OVER (
                        PARTITION BY c, t
                        ORDER BY ABS(COALESCE(z_ct_component, 0)) DESC, CAST(k AS VARCHAR)
                    ) AS rn
                FROM instrument_components
                WHERE z_ct_component IS NOT NULL
            )
            SELECT
                c,
                t,
                COUNT(*) AS n_contributing_schools,
                SUM(abs_component) AS total_abs_component,
                SUM(CASE WHEN rn = 1 THEN abs_component ELSE 0 END) AS top1_abs_component,
                SUM(CASE WHEN rn <= 3 THEN abs_component ELSE 0 END) AS top3_abs_component,
                SUM(CASE WHEN rn <= 5 THEN abs_component ELSE 0 END) AS top5_abs_component
            FROM ranked
            GROUP BY 1, 2
        """).df()
        if not firm_decomp.empty:
            firm_decomp["top1_share_abs"] = np.where(
                firm_decomp["total_abs_component"] > 0,
                firm_decomp["top1_abs_component"] / firm_decomp["total_abs_component"],
                np.nan,
            )
            firm_decomp["top3_share_abs"] = np.where(
                firm_decomp["total_abs_component"] > 0,
                firm_decomp["top3_abs_component"] / firm_decomp["total_abs_component"],
                np.nan,
            )
            firm_decomp["top5_share_abs"] = np.where(
                firm_decomp["total_abs_component"] > 0,
                firm_decomp["top5_abs_component"] / firm_decomp["total_abs_component"],
                np.nan,
            )
            firm_decomp.to_csv(diag_dir / "firm_instrument_decomposition.csv", index=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(firm_decomp["top1_share_abs"].dropna(), bins=40, ax=ax)
            ax.set_xlabel("Top-school share of |z_ct|")
            ax.set_ylabel("Count")
            ax.set_title("Firm-year instrument concentration")
            fig.tight_layout()
            fig.savefig(diag_dir / "firm_instrument_concentration.png", dpi=150)
            plt.show()
            print("[diag] saved firm_instrument_decomposition.csv and firm_instrument_concentration.png")
    except Exception as e:
        print(f"[diag] firm decomposition skipped: {e}")

    try:
        influence_df = con.sql(f"""
            SELECT
                CAST(ic.k AS VARCHAR) AS k,
                COALESCE(MAX(COALESCE(sn.school_name, '')), CAST(ic.k AS VARCHAR)) AS school_name,
                SUM(ABS(COALESCE(ic.z_ct_component, 0))) AS abs_component_mass,
                SUM(COALESCE(ic.z_ct_component, 0) * COALESCE(ap.{x_col}, 0)) AS covariance_proxy
            FROM instrument_components ic
            JOIN analysis_panel ap USING (c, t)
            LEFT JOIN diag_school_names sn ON CAST(ic.k AS VARCHAR) = sn.k
            GROUP BY 1
        """).df()
        if not influence_df.empty:
            influence_df["abs_component_share"] = (
                influence_df["abs_component_mass"] / influence_df["abs_component_mass"].sum()
            )
            denom = influence_df["covariance_proxy"].abs().sum()
            influence_df["rotemberg_proxy_abs_share"] = np.where(
                denom > 0,
                influence_df["covariance_proxy"].abs() / denom,
                np.nan,
            )
            influence_df = influence_df.sort_values("rotemberg_proxy_abs_share", ascending=False)
            influence_df.to_csv(diag_dir / "school_influence_diagnostics.csv", index=False)
            print("[diag] saved school_influence_diagnostics.csv")
    except Exception as e:
        print(f"[diag] influence diagnostics skipped: {e}")

    # ── 4e: Shock-level first stage and share-window robustness ─────────────
    try:
        diag_variants = [
            "z_ct",
            "z_ct_raw_flow",
            "z_ct_event_pulse",
            "z_ct_flow_diff",
            "z_ct_flow_ar_resid",
            "z_ct_common_base_level",
            "z_ct_common_base_asinh",
            "z_ct_event_step_dose",
            "z_ct_v1_broad_step",
            "z_ct_v2_broad_cumulative",
            "z_ct_v3_broad_predicted_opt",
            "z_ct_v4_matched_step",
            "z_ct_v5_matched_pulse",
            "z_ct_v6_broad_composition",
            "z_ct_share_2008_2010",
            "z_ct_share_2011_2013",
            "z_ct_full",
        ]
        panel_frames: list[pd.DataFrame] = []
        summary_rows: list[dict[str, object]] = []
        for variant in diag_variants:
            comp_col = _instrument_component_col(variant)
            g_col = _instrument_g_col(variant)
            if (
                comp_col is None
                or g_col is None
                or not _has_column(con, "instrument_components", comp_col)
                or not _has_column(con, "instrument_components", g_col)
            ):
                continue
            shock_level = con.sql(f"""
                SELECT
                    '{variant}' AS instrument_variant,
                    CAST(ic.k AS VARCHAR) AS k,
                    CAST(ic.t AS INTEGER) AS t,
                    MAX(ic.{g_col}) AS g_kt,
                    SUM(ABS(COALESCE(ic.{comp_col}, 0))) AS abs_component_mass,
                    SUM(COALESCE(ic.share_ck, 0)) AS exposure_sum,
                    CASE WHEN SUM(COALESCE(ic.share_ck, 0)) > 0
                         THEN SUM(COALESCE(ic.share_ck, 0) * COALESCE(ap.{x_col}, 0))
                              / SUM(COALESCE(ic.share_ck, 0))
                         ELSE NULL END AS x_weighted,
                    CASE WHEN SUM(COALESCE(ic.share_ck, 0)) > 0
                         THEN SUM(COALESCE(ic.share_ck, 0) * COALESCE(ap.y_cst_lag0, 0))
                              / SUM(COALESCE(ic.share_ck, 0))
                         ELSE NULL END AS y_weighted
                FROM instrument_components ic
                JOIN analysis_panel ap USING (c, t)
                WHERE ic.{comp_col} IS NOT NULL AND ic.{g_col} IS NOT NULL
                GROUP BY 1, 2, 3
            """).df()
            if shock_level.empty:
                continue
            panel_frames.append(shock_level)
            mass = pd.to_numeric(shock_level["abs_component_mass"], errors="coerce").fillna(0.0)
            active = shock_level.loc[pd.to_numeric(shock_level["g_kt"], errors="coerce").fillna(0.0).ne(0)]
            summary_rows.append({
                "instrument_variant": variant,
                "shock_year_rows": int(len(shock_level)),
                "active_shock_year_rows": int(len(active)),
                "schools": int(shock_level["k"].nunique()),
                "active_schools": int(active["k"].nunique()) if not active.empty else 0,
                "abs_component_mass": float(mass.sum()),
                "effective_schools_by_abs_component_mass": (
                    float((mass.groupby(shock_level["k"]).sum().sum() ** 2)
                          / np.square(mass.groupby(shock_level["k"]).sum()).sum())
                    if np.square(mass.groupby(shock_level["k"]).sum()).sum() > 0 else np.nan
                ),
                "top1_school_abs_component_share": (
                    float(mass.groupby(shock_level["k"]).sum().max() / mass.sum())
                    if mass.sum() > 0 else np.nan
                ),
                "top5_school_abs_component_share": (
                    float(mass.groupby(shock_level["k"]).sum().nlargest(5).sum() / mass.sum())
                    if mass.sum() > 0 else np.nan
                ),
            })

        if panel_frames:
            shock_panel_all = pd.concat(panel_frames, ignore_index=True)
            shock_panel_all.to_csv(diag_dir / "shock_level_panel.csv", index=False)
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(diag_dir / "shock_level_diagnostics_by_variant.csv", index=False)
            print("[diag] saved shock_level_panel.csv and shock_level_diagnostics_by_variant.csv")
    except Exception as e:
        print(f"[diag] shock-level panel skipped: {e}")

    try:
        autocorr_rows: list[dict[str, object]] = []
        g_autocorr_cols = [
            "g_kt",
            "g_kt_raw_flow",
            "g_kt_event_pulse",
            "g_kt_flow_diff",
            "g_kt_flow_ar_resid",
            "g_kt_flow_diff_cumulative",
            "g_kt_flow_ar_resid_cumulative",
            "g_kt_common_base_level",
            "g_kt_event_step_dose",
        ]
        for g_col in g_autocorr_cols:
            if not _has_column(con, "ipeds_unit_growth", g_col):
                continue
            g_panel = con.sql(f"""
                SELECT CAST(k AS VARCHAR) AS id, CAST(t AS INTEGER) AS t, CAST({g_col} AS DOUBLE) AS value
                FROM ipeds_unit_growth
                WHERE {g_col} IS NOT NULL
            """).df()
            if g_panel.empty:
                continue
            g_panel = g_panel.sort_values(["id", "t"])
            for lag in [1, 2, 3]:
                g_panel[f"lag{lag}"] = g_panel.groupby("id")["value"].shift(lag)
                use = g_panel[["value", f"lag{lag}"]].dropna()
                autocorr_rows.append({
                    "level": "school_shock",
                    "variant": g_col,
                    "lag": lag,
                    "autocorrelation": float(use["value"].corr(use[f"lag{lag}"])) if len(use) >= 2 else np.nan,
                    "n_pairs": int(len(use)),
                    "n_units": int(g_panel["id"].nunique()),
                })
        z_autocorr_cols = [
            "z_ct",
            "z_ct_raw_flow",
            "z_ct_event_pulse",
            "z_ct_flow_diff",
            "z_ct_flow_ar_resid",
            "z_ct_flow_diff_cumulative",
            "z_ct_flow_ar_resid_cumulative",
            "z_ct_common_base_level",
            "z_ct_event_step_dose",
        ]
        for z_col in z_autocorr_cols:
            if z_col not in analysis_panel_df.columns:
                continue
            z_panel = analysis_panel_df.loc[:, ["c", "t", z_col]].copy()
            z_panel["id"] = z_panel["c"].astype(str)
            z_panel["t"] = pd.to_numeric(z_panel["t"], errors="coerce")
            z_panel["value"] = pd.to_numeric(z_panel[z_col], errors="coerce")
            z_panel = z_panel.dropna(subset=["id", "t", "value"]).sort_values(["id", "t"])
            for lag in [1, 2, 3]:
                z_panel[f"lag{lag}"] = z_panel.groupby("id")["value"].shift(lag)
                use = z_panel[["value", f"lag{lag}"]].dropna()
                autocorr_rows.append({
                    "level": "firm_shift_share",
                    "variant": z_col,
                    "lag": lag,
                    "autocorrelation": float(use["value"].corr(use[f"lag{lag}"])) if len(use) >= 2 else np.nan,
                    "n_pairs": int(len(use)),
                    "n_units": int(z_panel["id"].nunique()),
                })
        if autocorr_rows:
            pd.DataFrame(autocorr_rows).to_csv(diag_dir / "shock_autocorrelation_diagnostics.csv", index=False)
            print("[diag] saved shock_autocorrelation_diagnostics.csv")
    except Exception as e:
        print(f"[diag] shock autocorrelation diagnostics skipped: {e}")

    try:
        variant_cols = [c for c in [
            "z_ct",
            "z_ct_raw_flow",
            "z_ct_event_pulse",
            "z_ct_flow_diff",
            "z_ct_flow_ar_resid",
            "z_ct_common_base_level",
            "z_ct_common_base_asinh",
            "z_ct_event_step_dose",
            "z_ct_v1_broad_step",
            "z_ct_v2_broad_cumulative",
            "z_ct_v3_broad_predicted_opt",
            "z_ct_v4_matched_step",
            "z_ct_v5_matched_pulse",
            "z_ct_v6_broad_composition",
            "z_ct_share_2008_2010",
            "z_ct_share_2011_2013",
            "z_ct_share_2008_2013",
            "z_ct_full",
        ] if c in analysis_panel_df.columns]
        if variant_cols:
            rows = []
            tmp = analysis_panel_df.copy()
            for col in variant_cols:
                use = tmp[[col, x_col]].dropna()
                rows.append({
                    "instrument_col": col,
                    "corr_with_x": use[col].corr(use[x_col]) if not use.empty else np.nan,
                    "mean": pd.to_numeric(tmp[col], errors="coerce").mean(),
                    "std": pd.to_numeric(tmp[col], errors="coerce").std(),
                })
            pd.DataFrame(rows).to_csv(diag_dir / "instrument_variant_summary.csv", index=False)
            print("[diag] saved instrument_variant_summary.csv")
    except Exception as e:
        print(f"[diag] instrument variant summary skipped: {e}")

    # ── 4f: Balance / pre-trends ────────────────────────────────────────────
    try:
        firm_exposure = con.sql("""
            SELECT
                CAST(s.c AS VARCHAR) AS c,
                SUM(COALESCE(s.share_ck, 0) * COALESCE(g.delta_share_event, 0) * COALESCE(g.pre_school_size_event, 0))
                    AS exposure_baseline
            FROM transition_shares s
            JOIN (
                SELECT DISTINCT CAST(k AS VARCHAR) AS k,
                    COALESCE(delta_share_event, 0) AS delta_share_event,
                    COALESCE(pre_school_size_event, 0) AS pre_school_size_event
                FROM ipeds_unit_growth
            ) g ON s.k = g.k
            GROUP BY 1
        """).df()
        pre_panel = analysis_panel_df.copy()
        pre_panel["c"] = pre_panel["c"].astype(str)
        pre_panel["t"] = pd.to_numeric(pre_panel["t"], errors="coerce")
        pre_panel = pre_panel.loc[pre_panel["t"].notna() & (pre_panel["t"] <= 2013)].copy()
        if (not pre_panel.empty) and (not firm_exposure.empty):
            firm_exposure["c"] = firm_exposure["c"].astype(str)
            pre_means = pre_panel.groupby("c", as_index=False).agg(
                x_pre_mean=(x_col, "mean"),
                y_pre_mean=("y_cst_lag0", "mean"),
            )
            pre_trend = _groupwise_linear_slopes(
                pre_panel,
                group_col="c",
                x_col="t",
                y_cols=["y_cst_lag0", x_col],
            ).rename(columns={
                "y_cst_lag0_slope": "y_pre_slope",
                f"{x_col}_slope": "x_pre_slope",
            })
            firm_balance = firm_exposure.merge(pre_means, on="c", how="left").merge(pre_trend, on="c", how="left")
            firm_balance.to_csv(diag_dir / "firm_pretrend_balance.csv", index=False)
            print("[diag] saved firm_pretrend_balance.csv")
    except Exception as e:
        print(f"[diag] firm pre-trends skipped: {e}")

    try:
        school_pre = con.sql("""
            SELECT
                CAST(k AS VARCHAR) AS k,
                MAX(COALESCE(school_name, '')) AS school_name,
                MAX(COALESCE(delta_share_event, 0)) AS delta_share_event,
                MAX(COALESCE(pre_school_size_event, 0)) AS pre_school_size_event,
                AVG(CASE WHEN CAST(t AS INTEGER) <= 2013 THEN metric_share END) AS metric_share_pre_mean,
                AVG(CASE WHEN CAST(t AS INTEGER) <= 2013 THEN school_size END) AS school_size_pre_mean
            FROM ipeds_unit_growth
            GROUP BY 1
        """).df()
        school_pre_panel = con.sql("""
            SELECT CAST(k AS VARCHAR) AS k, CAST(t AS INTEGER) AS t,
                metric_share, school_size
            FROM ipeds_unit_growth
            WHERE CAST(t AS INTEGER) <= 2013
        """).df()
        if not school_pre.empty:
            if not school_pre_panel.empty:
                school_slopes = _groupwise_linear_slopes(
                    school_pre_panel,
                    group_col="k",
                    x_col="t",
                    y_cols=["metric_share", "school_size"],
                ).rename(columns={
                    "metric_share_slope": "metric_share_pre_slope",
                    "school_size_slope": "school_size_pre_slope",
                })
                school_pre = school_pre.merge(school_slopes, on="k", how="left")
            school_pre.to_csv(diag_dir / "school_pretrend_balance.csv", index=False)
            print("[diag] saved school_pretrend_balance.csv")
    except Exception as e:
        print(f"[diag] school pre-trends skipped: {e}")

    print(f"[diag] diagnostic plots saved to {diag_dir}")


# =============================================================================
# ── SECTION 5: REGRESSIONS ────────────────────────────────────────────────────
# =============================================================================

def _make_x_bin(panel: pd.DataFrame, x_col: str, rule: str) -> pd.Series:
    """Derive binary treatment indicator from continuous OPT-hire count."""
    if rule == "any_nonzero":
        return (panel[x_col] > 0).astype(int)
    if rule == "year_median":
        medians = panel.groupby("t")[x_col].transform("median")
        return (panel[x_col] > medians).astype(int)
    if rule == "topbot_quartile":
        q25 = panel.groupby("t")[x_col].transform(lambda s: s.quantile(0.25))
        q75 = panel.groupby("t")[x_col].transform(lambda s: s.quantile(0.75))
        mask = (panel[x_col] <= q25) | (panel[x_col] >= q75)
        result = (panel.loc[mask, x_col] >= panel.loc[mask, "t"].map(
            panel.groupby("t")[x_col].quantile(0.75)
        )).astype(int)
        out = pd.Series(pd.NA, index=panel.index, dtype="Int64")
        out[mask] = result.values
        return out
    raise ValueError(f"Invalid treatment_binary_rule: {rule!r}")


def _build_interaction_plot_frame(
    coefs: pd.Series,
    ses: pd.Series,
    term_specs: list[tuple[int, str]],
    omitted_value: int | None,
    axis_name: str,
) -> pd.DataFrame:
    """Convert interaction coefficients into a plotting frame."""
    rows: list[dict] = []
    if omitted_value is not None:
        rows.append({
            axis_name: int(omitted_value),
            "term": None,
            "coef": 0.0,
            "se": np.nan,
            "omitted": True,
            "available": True,
        })
    for value, term in term_specs:
        coef = np.nan
        se = np.nan
        available = False
        if hasattr(coefs, "index") and term in coefs.index:
            try:
                coef = float(coefs.loc[term])
                se = float(ses.loc[term])
                available = True
            except Exception:
                available = False
        rows.append({
            axis_name: int(value),
            "term": term,
            "coef": coef,
            "se": se,
            "omitted": False,
            "available": available,
        })
    out = pd.DataFrame(rows).sort_values(axis_name).reset_index(drop=True)
    out["se_low"] = out["coef"] - out["se"]
    out["se_high"] = out["coef"] + out["se"]
    return out


def _plot_interaction_coefficients(
    plot_df: pd.DataFrame,
    axis_name: str,
    title: str,
    x_label: str,
    out_path: Path,
) -> None:
    """Plot interaction coefficients with +/-1 standard-error bars."""
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(axis_name).reset_index(drop=True)
    valid = plot_df.loc[plot_df["available"] | plot_df["omitted"]].copy()
    if valid.empty:
        return

    fig_w = max(7.0, 0.55 * len(valid) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.7)

    line_df = valid.loc[valid["coef"].notna()].copy()
    if not line_df.empty:
        ax.plot(
            line_df[axis_name],
            line_df["coef"],
            color="#4c72b0",
            lw=1.5,
            alpha=0.85,
            zorder=1,
        )

    coef_df = valid.loc[(~valid["omitted"]) & valid["available"]].copy()
    if not coef_df.empty:
        errorbar_container = ax.errorbar(
            coef_df[axis_name],
            coef_df["coef"],
            yerr=coef_df["se"],
            fmt="o",
            color="#4c72b0",
            ecolor="#4c72b0",
            elinewidth=1.4,
            capsize=3,
            markersize=5,
            zorder=3,
        )
        _soften_errorbar_interval(errorbar_container)

    omitted_df = valid.loc[valid["omitted"]].copy()
    if not omitted_df.empty:
        ax.scatter(
            omitted_df[axis_name],
            omitted_df["coef"],
            color="#c44e52",
            marker="s",
            s=45,
            zorder=4,
            label="Omitted group",
        )

    missing_df = plot_df.loc[(~plot_df["omitted"]) & (~plot_df["available"])].copy()
    if not missing_df.empty:
        ax.scatter(
            missing_df[axis_name],
            np.zeros(len(missing_df)),
            color="#999999",
            marker="x",
            s=35,
            zorder=4,
            label="Dropped / unavailable",
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Coefficient")
    ax.set_xticks(plot_df[axis_name].tolist())
    ax.set_xlim(plot_df[axis_name].min() - 0.4, plot_df[axis_name].max() + 0.4)
    if axis_name.lower() == "year":
        ax.tick_params(axis="x", rotation=45)
    if (not omitted_df.empty) or (not missing_df.empty):
        ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    try:
        plt.show(block=False)
        plt.pause(0.001)
    except TypeError:
        plt.show()
    plt.close(fig)


def _assign_quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Assign 1-indexed quantile bins, tolerating duplicate values."""
    out = pd.Series(pd.NA, index=values.index, dtype="Int64")
    clean = pd.to_numeric(values, errors="coerce").replace([float("inf"), float("-inf")], np.nan)
    mask = clean.notna()
    if not mask.any():
        return out
    q = max(1, int(n_bins))
    if q == 1:
        out.loc[mask] = 1
        return out
    try:
        bins = pd.qcut(clean.loc[mask], q=q, labels=False, duplicates="drop")
    except Exception:
        ranked = clean.loc[mask].rank(method="first")
        bins = pd.qcut(ranked, q=min(q, max(1, len(ranked))), labels=False, duplicates="drop")
    bins = pd.Series(bins, index=clean.loc[mask].index).astype("Int64") + 1
    out.loc[bins.index] = bins
    return out


def _assign_group_quantile_bins(values: pd.Series, groups: pd.Series, n_bins: int) -> pd.Series:
    """Assign within-group 1-indexed quantile bins."""
    out = pd.Series(pd.NA, index=values.index, dtype="Int64")
    tmp = pd.DataFrame({"value": values, "group": groups}, index=values.index)
    for _, idx in tmp.groupby("group", sort=True).groups.items():
        out.loc[idx] = _assign_quantile_bins(tmp.loc[idx, "value"], n_bins)
    return out


def _prepare_first_stage_state_panel(
    panel: pd.DataFrame,
    baseline_window_start: int,
    baseline_window_end: int,
    current_size_bins: int,
    current_growth_bins: int,
    joint_size_growth_bins: int,
    baseline_growth_bins: Optional[int] = None,
    use_log_y_panel: bool = False,
) -> pd.DataFrame:
    """Add firm-state variables used for conditional first-stage diagnostics."""
    work = panel.copy()
    if "t_num" not in work.columns:
        work["t_num"] = pd.to_numeric(work.get("t"), errors="coerce")

    def _raw_headcount(col_name: str, fallback_col: str) -> pd.Series:
        if col_name in work.columns and pd.to_numeric(work[col_name], errors="coerce").notna().any():
            raw = pd.to_numeric(work[col_name], errors="coerce")
        elif not use_log_y_panel and fallback_col in work.columns:
            raw = pd.to_numeric(work[fallback_col], errors="coerce")
        else:
            raw = pd.Series(np.nan, index=work.index, dtype="float64")
        return raw.replace([float("inf"), float("-inf")], np.nan).fillna(0.0)

    work["headcount_lag0_raw"] = _raw_headcount("headcount_lag0_raw", "y_cst_lag0")
    work["headcount_lag1_raw"] = _raw_headcount("headcount_lag1_raw", "y_cst_lag1")

    computed_growth = np.arcsinh(work["headcount_lag0_raw"].clip(lower=0.0)) - np.arcsinh(
        work["headcount_lag1_raw"].clip(lower=0.0)
    )
    if "headcount_growth_asinh" in work.columns and pd.to_numeric(work["headcount_growth_asinh"], errors="coerce").notna().any():
        growth = pd.to_numeric(work["headcount_growth_asinh"], errors="coerce")
        work["headcount_growth_asinh"] = growth.replace([float("inf"), float("-inf")], np.nan).fillna(computed_growth)
    else:
        work["headcount_growth_asinh"] = computed_growth

    baseline_mask = work["t_num"].between(int(baseline_window_start), int(baseline_window_end), inclusive="both")
    baseline_by_firm = (
        work.loc[baseline_mask, ["c", "headcount_lag0_raw"]]
        .groupby("c", dropna=False)["headcount_lag0_raw"]
        .mean()
    )
    if "headcount_size_baseline" in work.columns and pd.to_numeric(work["headcount_size_baseline"], errors="coerce").notna().any():
        baseline = pd.to_numeric(work["headcount_size_baseline"], errors="coerce")
        baseline_fill = work["c"].map(baseline_by_firm)
        work["headcount_size_baseline"] = baseline.replace([float("inf"), float("-inf")], np.nan).fillna(baseline_fill)
    else:
        work["headcount_size_baseline"] = work["c"].map(baseline_by_firm)

    baseline_growth_by_firm = (
        work.loc[baseline_mask, ["c", "headcount_growth_asinh"]]
        .groupby("c", dropna=False)["headcount_growth_asinh"]
        .mean()
    )
    if "headcount_growth_baseline" in work.columns and pd.to_numeric(work["headcount_growth_baseline"], errors="coerce").notna().any():
        baseline_growth = pd.to_numeric(work["headcount_growth_baseline"], errors="coerce")
        baseline_growth_fill = work["c"].map(baseline_growth_by_firm)
        work["headcount_growth_baseline"] = (
            baseline_growth.replace([float("inf"), float("-inf")], np.nan).fillna(baseline_growth_fill)
        )
    else:
        work["headcount_growth_baseline"] = work["c"].map(baseline_growth_by_firm)

    baseline_df = (
        work.loc[:, ["c", "headcount_size_baseline", "headcount_growth_baseline"]]
        .drop_duplicates(subset=["c"])
        .reset_index(drop=True)
    )
    baseline_df["baseline_size_decile"] = _assign_quantile_bins(
        baseline_df["headcount_size_baseline"],
        current_size_bins,
    )
    baseline_df["baseline_growth_quantile"] = _assign_quantile_bins(
        baseline_df["headcount_growth_baseline"],
        int(baseline_growth_bins or current_growth_bins),
    )
    work = work.drop(columns=["baseline_size_decile", "baseline_growth_quantile"], errors="ignore").merge(
        baseline_df[["c", "baseline_size_decile", "baseline_growth_quantile"]],
        on="c",
        how="left",
    )
    work["current_size_decile"] = _assign_group_quantile_bins(
        work["headcount_lag0_raw"],
        work["t_num"],
        current_size_bins,
    )
    work["current_growth_quintile"] = _assign_group_quantile_bins(
        work["headcount_growth_asinh"],
        work["t_num"],
        current_growth_bins,
    )
    work["current_size_tercile"] = _assign_group_quantile_bins(
        work["headcount_lag0_raw"],
        work["t_num"],
        joint_size_growth_bins,
    )
    work["current_growth_tercile"] = _assign_group_quantile_bins(
        work["headcount_growth_asinh"],
        work["t_num"],
        joint_size_growth_bins,
    )

    def _state_year_label(parts: list[pd.Series]) -> pd.Series:
        out = pd.Series(pd.NA, index=work.index, dtype="string")
        mask = pd.Series(True, index=work.index)
        for part in parts:
            mask &= part.notna()
        if mask.any():
            labels = parts[0].loc[mask].astype(str)
            for part in parts[1:]:
                labels = labels + "__" + part.loc[mask].astype(str)
            out.loc[mask] = labels
        return out

    year_str = work["t_num"].astype("Int64").astype("string")
    baseline_str = work["baseline_size_decile"].astype("Int64").astype("string")
    baseline_growth_str = work["baseline_growth_quantile"].astype("Int64").astype("string")
    current_size_str = work["current_size_decile"].astype("Int64").astype("string")
    current_growth_str = work["current_growth_quintile"].astype("Int64").astype("string")
    current_size_joint_str = work["current_size_tercile"].astype("Int64").astype("string")
    current_growth_joint_str = work["current_growth_tercile"].astype("Int64").astype("string")

    work["baseline_size_year_fe"] = _state_year_label([year_str, pd.Series("b" + baseline_str, index=work.index)])
    work["baseline_growth_year_fe"] = _state_year_label([year_str, pd.Series("bg" + baseline_growth_str, index=work.index)])
    work["baseline_size_growth_cell"] = _state_year_label(
        [pd.Series("b" + baseline_str, index=work.index), pd.Series("bg" + baseline_growth_str, index=work.index)]
    )
    work["baseline_size_growth_year_fe"] = _state_year_label([year_str, work["baseline_size_growth_cell"]])
    work["current_size_year_fe"] = _state_year_label([year_str, pd.Series("s" + current_size_str, index=work.index)])
    work["current_growth_year_fe"] = _state_year_label([year_str, pd.Series("g" + current_growth_str, index=work.index)])
    work["joint_size_growth_cell"] = _state_year_label(
        [pd.Series("s" + current_size_joint_str, index=work.index), pd.Series("g" + current_growth_joint_str, index=work.index)]
    )
    work["joint_size_growth_year_fe"] = _state_year_label([year_str, work["joint_size_growth_cell"]])

    work["headcount_lag0_asinh"] = np.arcsinh(work["headcount_lag0_raw"].clip(lower=0.0))
    work["headcount_lag0_asinh_sq"] = work["headcount_lag0_asinh"] ** 2
    work["headcount_growth_asinh_sq"] = work["headcount_growth_asinh"] ** 2
    return work


def _prepare_first_stage_outcomes(panel: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Add first-stage outcome variants used to diagnose threshold mechanics."""
    work = panel.copy()
    x_vals = pd.to_numeric(work.get(x_col), errors="coerce").replace([float("inf"), float("-inf")], np.nan).fillna(0.0)
    denom = pd.to_numeric(work.get("headcount_lag0_raw"), errors="coerce").replace([float("inf"), float("-inf")], np.nan).fillna(0.0)
    work["x_asinh"] = np.arcsinh(x_vals.clip(lower=0.0))
    work["x_rate_headcount"] = x_vals / denom.clip(lower=1.0)
    return work


def _lag_suffix(lag: int) -> str:
    return f"m{abs(int(lag))}" if int(lag) < 0 else str(int(lag))


def _dynamic_x_outcome(
    panel: pd.DataFrame,
    value_col: str,
    horizon: int,
    lookup_panel: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Return firm-year value_col at t+horizon aligned to the current firm-year row."""
    lookup_source = lookup_panel if lookup_panel is not None else panel
    lookup = lookup_source.loc[:, ["c", "t_num", value_col]].copy()
    lookup["target_t"] = pd.to_numeric(lookup["t_num"], errors="coerce").astype(float)
    lookup = lookup.dropna(subset=["c", "target_t", value_col])
    mapper = {
        (str(c), float(t)): v
        for c, t, v in lookup.loc[:, ["c", "target_t", value_col]].itertuples(index=False, name=None)
    }
    return pd.Series(
        [
            mapper.get((str(c), float(t) + int(horizon)), np.nan)
            if pd.notna(t) else np.nan
            for c, t in panel.loc[:, ["c", "t_num"]].itertuples(index=False, name=None)
        ],
        index=panel.index,
    )


def _run_dynamic_effect_plots(
    panel: pd.DataFrame,
    instrument_col: str,
    x_col: str,
    outcome_col: str,
    use_log_outcome: bool,
    first_stage_use_ppml: bool,
    horizon_start: int,
    horizon_end: int,
    out_dir: Path,
    fe_cols: Optional[list[str]] = None,
    x_lookup_panel: Optional[pd.DataFrame] = None,
) -> None:
    """Estimate and plot horizon-specific first-stage and reduced-form coefficients."""
    try:
        import pyfixest as pf
    except ImportError:
        print("[regressions] dynamic effect plots skipped: pyfixest not installed.")
        return
    if panel.empty or instrument_col not in panel.columns:
        return
    horizons = list(range(int(horizon_start), int(horizon_end) + 1))
    if not horizons:
        return

    work = panel.copy()
    work["c"] = work["c"].astype(str)
    work["t"] = pd.to_numeric(work["t_num"], errors="coerce").astype("Int64").astype(str)
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    absorbed_fe_cols = list(fe_cols or ["c", "t"])
    fe_term = _build_absorb_fe_term(absorbed_fe_cols)

    y_base = None
    if "_lag" in str(outcome_col):
        y_base = str(outcome_col).rsplit("_lag", 1)[0]

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        x_dyn_col = f"dyn_x_h{_lag_suffix(horizon)}"
        work[x_dyn_col] = _dynamic_x_outcome(work, x_col, horizon, lookup_panel=x_lookup_panel)
        if first_stage_use_ppml:
            fs_col = x_dyn_col
            fs_panel = work.loc[
                work[[instrument_col, fs_col, *absorbed_fe_cols]].notna().all(axis=1)
                & pd.to_numeric(work[fs_col], errors="coerce").ge(0)
            ].copy()
            fs_estimator = "ppml"
            fs_formula = f"{fs_col} ~ {instrument_col} {fe_term}"
        else:
            fs_col = f"dyn_x_bin_h{_lag_suffix(horizon)}"
            work[fs_col] = (pd.to_numeric(work[x_dyn_col], errors="coerce").fillna(0.0) > 0).astype(float)
            fs_panel = work.loc[work[[instrument_col, fs_col, *absorbed_fe_cols]].notna().all(axis=1)].copy()
            fs_estimator = "ols_lpm"
            fs_formula = f"{fs_col} ~ {instrument_col} {fe_term}"
        if not fs_panel.empty and fs_panel[fs_col].nunique(dropna=True) > 1:
            try:
                fit = (
                    pf.fepois(fs_formula, data=fs_panel, vcov={"CRV1": "c"})
                    if fs_estimator == "ppml"
                    else pf.feols(fs_formula, data=fs_panel, vcov={"CRV1": "c"})
                )
                coef, se = _coef_for_term(fit, instrument_col)
                rows.append({
                    "family": "first_stage",
                    "horizon": int(horizon),
                    "lhs": fs_col,
                    "estimator": fs_estimator,
                    "coef": coef,
                    "se": se,
                    "n_obs": _model_nobs(fit, len(fs_panel)),
                    "n_companies": int(fs_panel["c"].nunique()),
                })
            except Exception as e:
                rows.append({
                    "family": "first_stage",
                    "horizon": int(horizon),
                    "lhs": fs_col,
                    "estimator": fs_estimator,
                    "error": str(e),
                })

        if y_base is not None:
            y_col = f"{y_base}_lag{_lag_suffix(horizon)}"
            if y_col in work.columns:
                rf_col = f"dyn_y_h{_lag_suffix(horizon)}"
                y_vals = pd.to_numeric(work[y_col], errors="coerce")
                work[rf_col] = y_vals.apply(
                    lambda v: math.log1p(max(v, 0)) if use_log_outcome and pd.notna(v) else v
                )
                rf_panel = work.loc[work[[instrument_col, rf_col, *absorbed_fe_cols]].notna().all(axis=1)].copy()
                if not rf_panel.empty and rf_panel[rf_col].nunique(dropna=True) > 1:
                    try:
                        fit = pf.feols(f"{rf_col} ~ {instrument_col} {fe_term}", data=rf_panel, vcov={"CRV1": "c"})
                        coef, se = _coef_for_term(fit, instrument_col)
                        rows.append({
                            "family": "reduced_form",
                            "horizon": int(horizon),
                            "lhs": rf_col,
                            "source_outcome_col": y_col,
                            "estimator": "ols",
                            "coef": coef,
                            "se": se,
                            "n_obs": _model_nobs(fit, len(rf_panel)),
                            "n_companies": int(rf_panel["c"].nunique()),
                        })
                    except Exception as e:
                        rows.append({
                            "family": "reduced_form",
                            "horizon": int(horizon),
                            "lhs": rf_col,
                            "source_outcome_col": y_col,
                            "estimator": "ols",
                            "error": str(e),
                        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("[regressions] dynamic effect plots skipped: no estimable horizons.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dynamic_effect_coefficients.csv"
    out.to_csv(out_path, index=False)
    plot_df = out.loc[out["coef"].notna() & out["se"].notna()].copy()
    if plot_df.empty:
        print(f"[regressions] saved dynamic effect coefficient data → {out_path}")
        return
    plot_df["ci_low"] = plot_df["coef"] - 1.96 * plot_df["se"]
    plot_df["ci_high"] = plot_df["coef"] + 1.96 * plot_df["se"]
    for family, title, filename in [
        ("first_stage", "Dynamic First Stage", "dynamic_first_stage_coefficients.png"),
        ("reduced_form", "Dynamic Reduced Form", "dynamic_reduced_form_coefficients.png"),
    ]:
        sub = plot_df.loc[plot_df["family"] == family].sort_values("horizon")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 5))
        errorbar_container = ax.errorbar(
            sub["horizon"],
            sub["coef"],
            yerr=1.96 * sub["se"],
            fmt="o-",
            capsize=3,
            linewidth=1.5,
        )
        _soften_errorbar_interval(errorbar_container)
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Outcome horizon relative to instrument year")
        ax.set_ylabel("Coefficient on z_ct")
        ax.set_title(f"{title}: {instrument_col}")
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.show()
        plt.close(fig)
    print(f"[regressions] saved dynamic effect coefficient data → {out_path}")


def _run_distributed_lag_robustness(
    panel: pd.DataFrame,
    instrument_col: str,
    x_col: str,
    y_reg_col: str,
    first_stage_use_ppml: bool,
    max_lag: int,
    out_dir: Path,
    fe_cols: list[str],
    lag_lookup_panel: Optional[pd.DataFrame] = None,
) -> None:
    """Estimate static first-stage/reduced-form specs with contemporaneous and lagged z."""
    try:
        import pyfixest as pf
    except ImportError:
        print("[regressions] distributed-lag robustness skipped: pyfixest not installed.")
        return
    if panel.empty or instrument_col not in panel.columns:
        return
    work = panel.copy()
    lag_terms: list[str] = []
    for lag in range(0, int(max_lag) + 1):
        col = f"{instrument_col}_lag{lag}"
        work[col] = (
            _dynamic_x_outcome(work, instrument_col, -lag, lookup_panel=lag_lookup_panel)
            if lag > 0
            else pd.to_numeric(work[instrument_col], errors="coerce")
        )
        lag_terms.append(col)
    absorbed_fe_cols = list(fe_cols)
    fe_term = _build_absorb_fe_term(absorbed_fe_cols)
    required_cols = [*lag_terms, *absorbed_fe_cols]
    rows: list[dict[str, object]] = []
    specs = [
        ("first_stage", x_col if first_stage_use_ppml else "x_bin", "ppml" if first_stage_use_ppml else "ols_lpm"),
        ("reduced_form", y_reg_col, "ols"),
    ]
    for family, lhs, estimator in specs:
        if lhs not in work.columns:
            continue
        spec_panel = work.loc[work[[lhs, *required_cols]].notna().all(axis=1)].copy()
        if estimator == "ppml":
            spec_panel = spec_panel.loc[pd.to_numeric(spec_panel[lhs], errors="coerce").ge(0)].copy()
        if spec_panel.empty or spec_panel[lhs].nunique(dropna=True) <= 1:
            rows.append({
                "family": family,
                "lhs": lhs,
                "estimator": estimator,
                "error": "No estimable outcome variation after lag/FE filtering.",
            })
            continue
        formula = f"{lhs} ~ {' + '.join(lag_terms)} {fe_term}"
        try:
            fit = (
                pf.fepois(formula, data=spec_panel, vcov={"CRV1": "c"})
                if estimator == "ppml"
                else pf.feols(formula, data=spec_panel, vcov={"CRV1": "c"})
            )
            coefs = fit.coef()
            ses = fit.se()
            cumulative = 0.0
            cumulative_var_proxy = 0.0
            for lag, term in enumerate(lag_terms):
                coef = float(coefs.loc[term]) if hasattr(coefs, "index") and term in coefs.index else np.nan
                se = float(ses.loc[term]) if hasattr(ses, "index") and term in ses.index else np.nan
                cumulative += coef if math.isfinite(coef) else 0.0
                cumulative_var_proxy += se ** 2 if math.isfinite(se) else 0.0
                rows.append({
                    "family": family,
                    "lhs": lhs,
                    "estimator": estimator,
                    "instrument_col": instrument_col,
                    "lag": lag,
                    "term": term,
                    "coef": coef,
                    "se": se,
                    "cumulative_coef_through_lag": cumulative,
                    "cumulative_se_independence_proxy": math.sqrt(cumulative_var_proxy) if cumulative_var_proxy >= 0 else np.nan,
                    "n_obs": _model_nobs(fit, len(spec_panel)),
                    "n_companies": int(spec_panel["c"].nunique()) if "c" in spec_panel.columns else np.nan,
                    "fe": " + ".join(absorbed_fe_cols),
                })
        except Exception as e:
            rows.append({
                "family": family,
                "lhs": lhs,
                "estimator": estimator,
                "instrument_col": instrument_col,
                "error": str(e),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "distributed_lag_coefficients.csv"
    out.to_csv(out_path, index=False)
    plot_df = out.loc[out.get("coef").notna() & out.get("se").notna()].copy() if "coef" in out.columns else pd.DataFrame()
    if not plot_df.empty:
        for family, title, filename in [
            ("first_stage", "Distributed-Lag First Stage", "distributed_lag_first_stage_coefficients.png"),
            ("reduced_form", "Distributed-Lag Reduced Form", "distributed_lag_reduced_form_coefficients.png"),
        ]:
            sub = plot_df.loc[plot_df["family"] == family].sort_values("lag")
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.8))
            errorbar_container = ax.errorbar(
                sub["lag"],
                sub["coef"],
                yerr=1.96 * sub["se"],
                fmt="o-",
                capsize=3,
            )
            _soften_errorbar_interval(errorbar_container)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_xlabel("Lag of z")
            ax.set_ylabel("Coefficient")
            ax.set_title(f"{title}: {instrument_col}")
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=150)
            plt.show()
            plt.close(fig)
    print(f"[regressions] saved distributed-lag robustness → {out_path}")


def _run_residualized_binscatter(
    panel: pd.DataFrame,
    lhs: str,
    instrument_col: str,
    fe_cols: list[str],
    label: str,
    estimator: str,
    model_coef: float | None,
    model_se: float | None,
    out_dir: Path,
    n_bins: int = 50,
) -> tuple[Path, Path] | None:
    """Save a residualized binned scatterplot for a single-instrument regression."""
    required_cols = [lhs, instrument_col, *fe_cols]
    if any(col not in panel.columns for col in required_cols):
        return None
    work = panel.loc[:, required_cols].copy()
    work[lhs] = pd.to_numeric(work[lhs], errors="coerce")
    work[instrument_col] = pd.to_numeric(work[instrument_col], errors="coerce")
    work = work.replace([float("inf"), float("-inf")], np.nan).dropna()
    if len(work) < 10:
        return None

    fe_groups = [work[col].astype(str) for col in fe_cols]
    y_resid = _residualize_fixed_effects(work[lhs], fe_groups) if fe_groups else work[lhs] - work[lhs].mean()
    x_resid = (
        _residualize_fixed_effects(work[instrument_col], fe_groups)
        if fe_groups else work[instrument_col] - work[instrument_col].mean()
    )
    plot_work = pd.DataFrame({
        "x_resid": pd.to_numeric(x_resid, errors="coerce"),
        "y_resid": pd.to_numeric(y_resid, errors="coerce"),
    }).replace([float("inf"), float("-inf")], np.nan).dropna()
    if len(plot_work) < 10 or plot_work["x_resid"].nunique(dropna=True) < 2:
        return None

    denom = float(np.dot(plot_work["x_resid"], plot_work["x_resid"]))
    fwl_slope = (
        float(np.dot(plot_work["x_resid"], plot_work["y_resid"]) / denom)
        if denom > 0 else np.nan
    )
    bins = min(max(2, int(n_bins)), int(plot_work["x_resid"].nunique(dropna=True)), len(plot_work))
    rank = plot_work["x_resid"].rank(method="first")
    plot_work["bin"] = pd.qcut(rank, q=bins, labels=False, duplicates="drop") + 1
    bin_df = (
        plot_work.groupby("bin", as_index=False)
        .agg(
            x_resid_mean=("x_resid", "mean"),
            y_resid_mean=("y_resid", "mean"),
            n=("x_resid", "size"),
            x_resid_min=("x_resid", "min"),
            x_resid_max=("x_resid", "max"),
        )
        .sort_values("x_resid_mean")
    )
    if bin_df.empty:
        return None
    bin_df["spec_label"] = label
    bin_df["lhs"] = lhs
    bin_df["instrument_col"] = instrument_col
    bin_df["estimator"] = estimator
    bin_df["model_coef"] = model_coef
    bin_df["model_se"] = model_se
    bin_df["fwl_ols_slope"] = fwl_slope
    bin_df["n_obs"] = len(plot_work)
    bin_df["absorbed_fe"] = " + ".join(fe_cols) if fe_cols else "none"

    scatter_dir = out_dir / "residualized_binscatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    safe_label = _safe_path_component(label)
    data_path = scatter_dir / f"{safe_label}_binscatter.csv"
    plot_path = scatter_dir / f"{safe_label}_binscatter.png"
    bin_df.to_csv(data_path, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    sizes = 18 + 42 * (bin_df["n"] / max(float(bin_df["n"].max()), 1.0))
    ax.scatter(
        bin_df["x_resid_mean"],
        bin_df["y_resid_mean"],
        s=sizes,
        color="#4c72b0",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    x_min = float(bin_df["x_resid_mean"].min())
    x_max = float(bin_df["x_resid_mean"].max())
    x_line = np.linspace(x_min, x_max, 100)
    if math.isfinite(fwl_slope):
        ax.plot(x_line, fwl_slope * x_line, color="#c44e52", linewidth=1.5, label=f"FWL OLS slope={fwl_slope:.4g}")
    if estimator != "ppml" and model_coef is not None and math.isfinite(float(model_coef)):
        ax.plot(
            x_line,
            float(model_coef) * x_line,
            color="#55a868",
            linewidth=1.2,
            linestyle="--",
            label=f"model coef={float(model_coef):.4g}",
        )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(f"Residualized {instrument_col}")
    ax.set_ylabel(f"Residualized {lhs}")
    title = f"Residualized Binned Scatter: {label}"
    if estimator == "ppml":
        title += "\nlinear FWL visualization; reported first-stage estimate is PPML"
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return plot_path, data_path


def _build_first_stage_conditioning_specs(
    lhs: str,
    instrument_col: str,
    continuous_control_cols: list[str],
    headline_first_stage_spec: str,
) -> list[dict]:
    """Return the conditional first-stage ladder with a single z coefficient."""
    rhs_continuous = " + ".join([instrument_col] + continuous_control_cols)
    spec_key_map = {
        "FS0": "twfe",
        "FS1": "baseline_size_year_fe",
        "FS2": "current_size_year_fe",
        "FS3": "current_growth_year_fe",
        "FS4": "joint_size_growth_year_fe",
        "FS5": "continuous_year_controls",
    }
    specs = [
        {
            "spec_code": "FS0",
            "spec_key": spec_key_map["FS0"],
            "formula": f"{lhs} ~ {instrument_col} | c + t",
            "added_conditioning": "firm FE + year FE",
        },
        {
            "spec_code": "FS1",
            "spec_key": spec_key_map["FS1"],
            "formula": f"{lhs} ~ {instrument_col} | c + t + baseline_size_year_fe",
            "added_conditioning": "baseline size decile x year FE",
        },
        {
            "spec_code": "FS2",
            "spec_key": spec_key_map["FS2"],
            "formula": f"{lhs} ~ {instrument_col} | c + t + current_size_year_fe",
            "added_conditioning": "current size decile x year FE",
        },
        {
            "spec_code": "FS3",
            "spec_key": spec_key_map["FS3"],
            "formula": f"{lhs} ~ {instrument_col} | c + t + current_growth_year_fe",
            "added_conditioning": "current growth quintile x year FE",
        },
        {
            "spec_code": "FS4",
            "spec_key": spec_key_map["FS4"],
            "formula": f"{lhs} ~ {instrument_col} | c + t + joint_size_growth_year_fe",
            "added_conditioning": "current size tercile x current growth tercile x year FE",
        },
        {
            "spec_code": "FS5",
            "spec_key": spec_key_map["FS5"],
            "formula": f"{lhs} ~ {rhs_continuous} | c + t",
            "added_conditioning": "year-interacted continuous size and growth controls",
        },
    ]
    for spec in specs:
        spec["is_headline"] = spec["spec_key"] == str(headline_first_stage_spec).strip()
    return specs


def _create_instrument_category_interactions(
    panel: pd.DataFrame,
    instrument_col: str,
    category_col: str,
    prefix: str,
    category_values: list[int],
) -> tuple[pd.DataFrame, list[str], list[tuple[int, str]]]:
    """Create z × category interactions and return term specs for plotting."""
    work = panel.copy()
    cols: list[str] = []
    term_specs: list[tuple[int, str]] = []
    for value in category_values:
        col = f"{instrument_col}_x_{prefix}_{int(value)}"
        work[col] = work[instrument_col] * (work[category_col] == int(value)).astype(int)
        cols.append(col)
        term_specs.append((int(value), col))
    return work, cols, term_specs


def _create_instrument_joint_interactions(
    panel: pd.DataFrame,
    instrument_col: str,
    size_col: str,
    growth_col: str,
    prefix: str,
    size_values: list[int],
    growth_values: list[int],
) -> tuple[pd.DataFrame, list[str], list[tuple[tuple[int, int], str]]]:
    """Create z × (size cell x growth cell) interactions."""
    work = panel.copy()
    cols: list[str] = []
    term_specs: list[tuple[tuple[int, int], str]] = []
    for size_value in size_values:
        for growth_value in growth_values:
            col = f"{instrument_col}_x_{prefix}_s{int(size_value)}_g{int(growth_value)}"
            work[col] = work[instrument_col] * (
                (work[size_col] == int(size_value)) & (work[growth_col] == int(growth_value))
            ).astype(int)
            cols.append(col)
            term_specs.append(((int(size_value), int(growth_value)), col))
    return work, cols, term_specs


def _build_joint_interaction_plot_frame(
    coefs: pd.Series,
    ses: pd.Series,
    term_specs: list[tuple[tuple[int, int], str]],
) -> pd.DataFrame:
    """Convert joint interaction coefficients into a heatmap frame."""
    rows: list[dict] = []
    for (size_value, growth_value), term in term_specs:
        coef = np.nan
        se = np.nan
        available = False
        if hasattr(coefs, "index") and term in coefs.index:
            try:
                coef = float(coefs.loc[term])
                se = float(ses.loc[term])
                available = True
            except Exception:
                available = False
        rows.append(
            {
                "current_size_tercile": int(size_value),
                "current_growth_tercile": int(growth_value),
                "term": term,
                "coef": coef,
                "se": se,
                "available": available,
            }
        )
    out = pd.DataFrame(rows)
    out["se_low"] = out["coef"] - out["se"]
    out["se_high"] = out["coef"] + out["se"]
    return out


def _plot_joint_interaction_heatmap(plot_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Plot the joint size-growth interaction coefficients as a heatmap."""
    if plot_df.empty:
        return
    coef_mat = (
        plot_df.pivot(index="current_growth_tercile", columns="current_size_tercile", values="coef")
        .sort_index(ascending=False)
        .sort_index(axis=1)
    )
    avail_mat = (
        plot_df.pivot(index="current_growth_tercile", columns="current_size_tercile", values="available")
        .reindex(index=coef_mat.index, columns=coef_mat.columns)
    )
    annot_mat = coef_mat.copy().astype(object)
    for idx in coef_mat.index:
        for col in coef_mat.columns:
            row = plot_df[
                (plot_df["current_growth_tercile"] == idx) & (plot_df["current_size_tercile"] == col)
            ]
            if row.empty or not bool(row.iloc[0]["available"]):
                annot_mat.loc[idx, col] = ""
            else:
                annot_mat.loc[idx, col] = f"{float(row.iloc[0]['coef']):.4f}\n({float(row.iloc[0]['se']):.4f})"

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    sns.heatmap(
        coef_mat,
        mask=~avail_mat.fillna(False),
        annot=annot_mat,
        fmt="",
        cmap="RdBu_r",
        center=0.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Coefficient"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Current size tercile")
    ax.set_ylabel("Current growth tercile")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    try:
        plt.show(block=False)
        plt.pause(0.001)
    except TypeError:
        plt.show()
    plt.close(fig)


def _run_raw_exposure_proof_plots(
    con: ddb.DuckDBPyConnection,
    analysis_panel_df: pd.DataFrame,
    diag_dir: Path,
    x_col: str,
) -> None:
    """Raw mean OPT-hire plots for simple first-stage proof-of-concept diagnostics."""
    if x_col not in analysis_panel_df.columns or not {"c", "t"}.issubset(analysis_panel_df.columns):
        print(f"[diag] raw exposure proof plots skipped: required panel columns missing for {x_col}")
        return
    required_views = ["transition_shares", "ipeds_unit_growth"]
    if any(not _has_column(con, view, "k") for view in required_views):
        print("[diag] raw exposure proof plots skipped: missing transition/growth views")
        return
    required_growth_cols = ["treated_event_year", "metric_level", "common_base_metric_level", "pre_school_size_event", "delta_share_event"]
    if any(not _has_column(con, "ipeds_unit_growth", col) for col in required_growth_cols):
        print("[diag] raw exposure proof plots skipped: saved growth view lacks common-baseline event columns")
        return

    shocks = con.sql("""
        WITH school_events AS (
            SELECT
                CAST(k AS VARCHAR) AS k,
                CAST(MAX(treated_event_year) AS INTEGER) AS event_year,
                AVG(
                    CASE
                        WHEN CAST(t AS INTEGER) BETWEEN CAST(treated_event_year AS INTEGER)
                             AND CAST(treated_event_year AS INTEGER) + 1
                        THEN CAST(metric_level AS DOUBLE)
                        ELSE NULL
                    END
                ) AS post_metric_level,
                MAX(COALESCE(CAST(pre_school_size_event AS DOUBLE), 0)
                    * COALESCE(CAST(delta_share_event AS DOUBLE), 0)) AS event_step_dose
            FROM ipeds_unit_growth
            WHERE treated_event_year IS NOT NULL
            GROUP BY 1
        )
        SELECT
            k,
            event_year,
            CASE WHEN post_metric_level IS NOT NULL THEN post_metric_level ELSE event_step_dose END AS raw_event_flow,
            event_step_dose
        FROM school_events
        WHERE event_year IS NOT NULL
    """).df()
    if shocks.empty:
        print("[diag] raw exposure proof plots skipped: no school event shocks")
        return
    shocks = shocks.loc[pd.to_numeric(shocks["raw_event_flow"], errors="coerce").fillna(0).gt(0)].copy()
    if shocks.empty:
        print("[diag] raw exposure proof plots skipped: no positive raw event flows")
        return
    con.register("_raw_proof_shocks_df", shocks)
    try:
        exposure_event = con.sql("""
            SELECT
                CAST(s.c AS VARCHAR) AS c,
                CAST(sh.event_year AS INTEGER) AS event_year,
                SUM(COALESCE(CAST(s.share_ck AS DOUBLE), 0) * sh.raw_event_flow) AS exposure
            FROM transition_shares s
            JOIN _raw_proof_shocks_df sh ON CAST(s.k AS VARCHAR) = sh.k
            GROUP BY 1, 2
        """).df()
    finally:
        try:
            con.unregister("_raw_proof_shocks_df")
        except Exception:
            pass
    if exposure_event.empty:
        print("[diag] raw exposure proof plots skipped: no firm exposure to event shocks")
        return

    panel = analysis_panel_df.loc[:, ["c", "t", x_col]].copy()
    panel["c"] = panel["c"].astype(str)
    panel["t"] = pd.to_numeric(panel["t"], errors="coerce")
    panel[x_col] = pd.to_numeric(panel[x_col], errors="coerce")
    panel = panel.dropna(subset=["c", "t", x_col])
    if panel.empty:
        return
    panel["t"] = panel["t"].astype(int)
    base_start, base_end = 2011, 2013
    baseline_x = (
        panel.loc[panel["t"].between(base_start, base_end)]
        .groupby("c", as_index=False)[x_col]
        .mean()
        .rename(columns={x_col: "baseline_x"})
    )
    panel = panel.merge(baseline_x, on="c", how="left")
    panel["x_minus_baseline"] = panel[x_col] - panel["baseline_x"]

    firm_total = (
        exposure_event.assign(c=exposure_event["c"].astype(str))
        .groupby("c", as_index=False)["exposure"].sum()
        .rename(columns={"exposure": "total_event_exposure"})
    )
    calendar_panel = panel.merge(firm_total, on="c", how="left")
    calendar_panel["total_event_exposure"] = calendar_panel["total_event_exposure"].fillna(0.0)
    positive = calendar_panel.loc[calendar_panel["total_event_exposure"] > 0, "total_event_exposure"]
    if positive.nunique() >= 2:
        q25 = float(positive.quantile(0.25))
        q75 = float(positive.quantile(0.75))
        calendar_panel["exposure_group"] = np.where(
            calendar_panel["total_event_exposure"] >= q75,
            "High exposure",
            np.where(calendar_panel["total_event_exposure"] <= q25, "Low/zero exposure", "Middle exposure"),
        )
        calendar_plot = calendar_panel.loc[calendar_panel["exposure_group"].isin(["High exposure", "Low/zero exposure"])].copy()
        calendar_means = (
            calendar_plot.groupby(["t", "exposure_group"], as_index=False)
            .agg(
                mean_x=(x_col, "mean"),
                mean_x_minus_baseline=("x_minus_baseline", "mean"),
                n_firm_years=(x_col, "size"),
                n_firms=("c", "nunique"),
            )
        )
        calendar_means.to_csv(diag_dir / "raw_x_calendar_by_total_exposure.csv", index=False)
        if not calendar_means.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.lineplot(data=calendar_means, x="t", y="mean_x_minus_baseline", hue="exposure_group", marker="o", ax=ax)
            ax.set_xlabel("Year")
            ax.set_ylabel(f"Mean {x_col} minus firm 2011-2013 mean")
            ax.set_title("Baseline-normalized OPT hiring by raw-flow exposure")
            fig.tight_layout()
            fig.savefig(diag_dir / "raw_x_calendar_by_total_exposure.png", dpi=150)
            plt.show()

    firms = panel[["c"]].drop_duplicates()
    groups: list[pd.DataFrame] = []
    for event_year, sub in exposure_event.groupby("event_year", sort=True):
        exp = firms.merge(sub.loc[:, ["c", "exposure"]].assign(c=sub["c"].astype(str)), on="c", how="left")
        exp["exposure"] = pd.to_numeric(exp["exposure"], errors="coerce").fillna(0.0)
        pos = exp.loc[exp["exposure"] > 0, "exposure"]
        if pos.nunique() < 2:
            continue
        q25 = float(pos.quantile(0.25))
        q75 = float(pos.quantile(0.75))
        exp["exposure_group"] = np.where(
            exp["exposure"] >= q75,
            "High exposure",
            np.where(exp["exposure"] <= q25, "Low/zero exposure", "Middle exposure"),
        )
        exp = exp.loc[exp["exposure_group"].isin(["High exposure", "Low/zero exposure"])].copy()
        exp["event_year"] = int(event_year)
        groups.append(exp[["c", "event_year", "exposure", "exposure_group"]])
    if not groups:
        print("[diag] event-time raw exposure proof plot skipped: no event cohort with exposure dispersion")
        return
    group_df = pd.concat(groups, ignore_index=True)
    event_panel = panel.merge(group_df, on="c", how="inner")
    event_panel["event_time"] = event_panel["t"] - event_panel["event_year"]
    event_baseline = (
        event_panel.loc[event_panel["event_time"].between(-3, -1)]
        .groupby(["c", "event_year"], as_index=False)[x_col]
        .mean()
        .rename(columns={x_col: "event_baseline_x"})
    )
    event_panel = event_panel.merge(event_baseline, on=["c", "event_year"], how="left")
    event_panel["x_minus_event_baseline"] = event_panel[x_col] - event_panel["event_baseline_x"]
    event_panel = event_panel.loc[event_panel["event_time"].between(-5, 5)].copy()
    if event_panel.empty:
        return
    event_means = (
        event_panel.groupby(["event_time", "exposure_group"], as_index=False)
        .agg(
            mean_x=(x_col, "mean"),
            mean_x_minus_event_baseline=("x_minus_event_baseline", "mean"),
            n_firm_years=(x_col, "size"),
            n_firms=("c", "nunique"),
        )
    )
    event_means.to_csv(diag_dir / "raw_x_event_time_by_event_exposure.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=event_means, x="event_time", y="mean_x_minus_event_baseline", hue="exposure_group", marker="o", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Event time")
    ax.set_ylabel(f"Mean {x_col} minus firm-event pre mean")
    ax.set_title("Baseline-normalized OPT hiring around IHMP-flow expansion events")
    fig.tight_layout()
    fig.savefig(diag_dir / "raw_x_event_time_by_event_exposure.png", dpi=150)
    plt.show()
    print("[diag] saved raw exposure proof-of-concept plots")


def _interaction_slope(plot_df: pd.DataFrame, axis_name: str) -> float | None:
    """Compute a simple linear slope through available interaction coefficients."""
    valid = plot_df.loc[plot_df["available"] & plot_df["coef"].notna(), [axis_name, "coef"]].copy()
    if len(valid) < 2:
        return None
    try:
        return float(np.polyfit(valid[axis_name].astype(float), valid["coef"].astype(float), 1)[0])
    except Exception:
        return None


def _model_nobs(model, fallback_n: int) -> int:
    """Best-effort extraction of model observation count."""
    for attr in ("nobs", "n_obs", "N", "_N_rows", "nobs_", "n_obs_", "N_rows", "nobs1"):
        if hasattr(model, attr):
            n = getattr(model, attr)
            try:
                n_int = int(n)
            except (TypeError, ValueError):
                continue
            if n_int >= 0:
                return n_int
    if hasattr(model, "_data"):
        try:
            return int(len(model._data))
        except Exception:
            pass
    return int(fallback_n)


def _single_term_f_stat(coef: float | None, se: float | None) -> float | None:
    """Return the coefficient-specific F-statistic when available."""
    if coef is None or se is None or se <= 0:
        return None
    try:
        out = float(coef / se) ** 2
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return out if math.isfinite(out) else None


def _extract_f_stat(model) -> float | None:
    """Best-effort extraction of a model F / Wald statistic."""
    import re

    def _coerce_stat_value(val) -> float | None:
        if val is None:
            return None
        if isinstance(val, dict):
            for key in ("fstat", "f_stat", "statistic", "value", "stat"):
                if key in val:
                    try:
                        out = float(val[key])
                        if math.isfinite(out) and out >= 0:
                            return out
                    except (TypeError, ValueError):
                        pass
            return None
        if hasattr(val, "index") and hasattr(val, "get"):
            for key in ("fstat", "f_stat", "statistic", "value", "stat"):
                try:
                    item = val.get(key)
                except Exception:
                    item = None
                if item is not None:
                    try:
                        out = float(item)
                        if math.isfinite(out) and out >= 0:
                            return out
                    except (TypeError, ValueError):
                        pass
            return None
        try:
            out = float(val)
            if math.isfinite(out) and out >= 0:
                return out
        except (TypeError, ValueError):
            return None
        return None

    for name in ("fstat", "f_stat", "f_statistic", "wald", "wald_test", "Wald"):
        if not hasattr(model, name):
            continue
        val = getattr(model, name)
        if callable(val):
            try:
                val = val()
            except Exception:
                continue
        out = _coerce_stat_value(val)
        if out is not None:
            return out

    try:
        txt = str(model.summary())
        match = re.search(r"F-?stat(?:istic)?[^0-9]*([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)", txt)
        if match:
            out = float(match.group(1))
            if math.isfinite(out) and out >= 0:
                return out
    except Exception:
        pass

    try:
        tstats = model.tstat()
        if hasattr(tstats, "__len__") and len(tstats) == 1:
            t_val = float(pd.to_numeric(tstats, errors="coerce").iloc[0])
            if math.isfinite(t_val):
                return t_val ** 2
    except Exception:
        pass
    return None


def _coef_for_term(model, term: str) -> tuple[float | None, float | None]:
    """Return coefficient and standard error for a named regressor."""
    try:
        coefs = model.coef()
        ses = model.se()
    except Exception:
        return None, None
    if not hasattr(coefs, "index") or term not in coefs.index:
        return None, None
    try:
        return float(coefs.loc[term]), float(ses.loc[term])
    except Exception:
        return None, None


def _instrument_component_col(instrument_col: str) -> Optional[str]:
    """Return the instrument_components column corresponding to a z_ct variant."""
    if instrument_col == "z_ct":
        return "z_ct_component"
    if not instrument_col.startswith("z_ct_"):
        return None
    return instrument_col.replace("z_ct", "z_ct_component", 1)


def _instrument_g_col(instrument_col: str) -> Optional[str]:
    """Return the instrument_components school shock column corresponding to a z_ct variant."""
    mapping = {
        "z_ct": "g_kt",
        "z_ct_raw_flow": "g_kt_raw_flow",
        "z_ct_ihmp_share": "g_kt_ihmp_share",
        "z_ct_event_pulse": "g_kt_event_pulse",
        "z_ct_flow_diff": "g_kt_flow_diff",
        "z_ct_flow_ar_resid": "g_kt_flow_ar_resid",
        "z_ct_flow_diff_cumulative": "g_kt_flow_diff_cumulative",
        "z_ct_flow_ar_resid_cumulative": "g_kt_flow_ar_resid_cumulative",
        "z_ct_common_base_level": "g_kt_common_base_level",
        "z_ct_common_base_asinh": "g_kt_common_base_asinh",
        "z_ct_event_step_dose": "g_kt_event_step_dose",
        "z_ct_v1_broad_step": "g_kt_v1_broad_step",
        "z_ct_v2_broad_cumulative": "g_kt_v2_broad_cumulative",
        "z_ct_v3_broad_predicted_opt": "g_kt_v3_broad_predicted_opt",
        "z_ct_v4_matched_step": "g_kt_v4_matched_step",
        "z_ct_v5_matched_pulse": "g_kt_v5_matched_pulse",
        "z_ct_v6_broad_composition": "g_kt_v6_broad_composition",
        "z_ct_v7_matched_pulse_growth_rate": "g_kt_v7_matched_pulse_growth_rate",
        "z_ct_falsification_lead4_broad": "g_kt_falsification_lead4_broad",
        "z_ct_falsification_lead4_matched": "g_kt_falsification_lead4_matched",
        "z_ct_share_2008_2010": "g_kt",
        "z_ct_share_2011_2013": "g_kt",
        "z_ct_share_2008_2013": "g_kt",
        "z_ct_full": "g_kt",
    }
    return mapping.get(instrument_col)


def _residualize_fixed_effects(
    values: pd.Series,
    fe_groups: list[pd.Series],
    max_iter: int = 200,
    tol: float = 1e-10,
) -> pd.Series:
    """Residualize a vector against fixed-effect groups by alternating projections."""
    work = pd.to_numeric(values, errors="coerce").astype(float)
    resid = work - work.mean()
    groups = [group.astype(str) for group in fe_groups]
    for _ in range(max_iter):
        old = resid.copy()
        for group in groups:
            resid = resid - resid.groupby(group).transform("mean")
        resid = resid + resid.mean()
        diff = float(np.nanmax(np.abs(resid.to_numpy() - old.to_numpy()))) if len(resid) else 0.0
        if diff < tol:
            break
    return resid


def _twoway_residualize(
    values: pd.Series,
    firm_ids: pd.Series,
    year_ids: pd.Series,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> pd.Series:
    """Residualize a vector against firm and year fixed effects by alternating projections."""
    return _residualize_fixed_effects(values, [firm_ids, year_ids], max_iter=max_iter, tol=tol)


def _shock_level_score_inference(
    panel: pd.DataFrame,
    lhs: str,
    instrument_col: str,
    coef: float | None,
    fe_cols: Optional[list[str]] = None,
) -> dict[str, object]:
    """
    School-shock score inference for the single-instrument TWFE coefficient.

    The score for school k is sum_ct z_component_ckt * residual_ct. This treats
    schools as the independent shock dimension and is intended as a complement
    to firm-clustered SEs for shift-share designs.
    """
    if coef is None or DUCKDB_CONN is None:
        return {}
    comp_col = _instrument_component_col(instrument_col)
    if comp_col is None or not _has_column(DUCKDB_CONN, "instrument_components", comp_col):
        return {}
    absorbed_fe_cols = list(fe_cols or ["c", "t_num"])
    required = {lhs, instrument_col, *absorbed_fe_cols}
    if not required.issubset(panel.columns):
        return {}
    work = panel.loc[:, list(dict.fromkeys([*absorbed_fe_cols, lhs, instrument_col, "c", "t_num"]))].copy()
    work = work.dropna(subset=list(required))
    if work.empty:
        return {}
    fe_groups = [work[col] for col in absorbed_fe_cols]
    y_resid = _residualize_fixed_effects(work[lhs], fe_groups)
    z_resid = _residualize_fixed_effects(work[instrument_col], fe_groups)
    denom = float(np.dot(z_resid, z_resid))
    if not math.isfinite(denom) or denom <= 0:
        return {}
    u = y_resid - float(coef) * z_resid
    score_df = pd.DataFrame({
        "c": work["c"].astype(str),
        "t": pd.to_numeric(work["t_num"], errors="coerce").astype("Int64"),
        "u": u,
    }).dropna(subset=["c", "t", "u"])
    if score_df.empty:
        return {}
    DUCKDB_CONN.register("_shock_score_residuals_df", score_df)
    try:
        scores = DUCKDB_CONN.sql(f"""
            SELECT
                CAST(ic.k AS VARCHAR) AS k,
                SUM(COALESCE(ic.{comp_col}, 0) * r.u) AS score,
                SUM(ABS(COALESCE(ic.{comp_col}, 0))) AS abs_component_mass
            FROM instrument_components ic
            JOIN _shock_score_residuals_df r
              ON CAST(ic.c AS VARCHAR) = r.c
             AND CAST(ic.t AS INTEGER) = CAST(r.t AS INTEGER)
            WHERE ic.{comp_col} IS NOT NULL
            GROUP BY 1
            HAVING SUM(ABS(COALESCE(ic.{comp_col}, 0))) > 0
        """).df()
    finally:
        try:
            DUCKDB_CONN.unregister("_shock_score_residuals_df")
        except Exception:
            pass
    if scores.empty:
        return {}
    n_shocks = int(len(scores))
    score_sum_sq = float(np.square(pd.to_numeric(scores["score"], errors="coerce").fillna(0.0)).sum())
    correction = n_shocks / max(n_shocks - 1, 1)
    se = math.sqrt(max(correction * score_sum_sq, 0.0)) / denom
    mass = pd.to_numeric(scores["abs_component_mass"], errors="coerce").fillna(0.0)
    effective = float((mass.sum() ** 2) / np.square(mass).sum()) if np.square(mass).sum() > 0 else np.nan
    t_stat = float(coef) / se if se > 0 else np.nan
    return {
        "shock_level_se_instrument": se,
        "shock_level_t_stat": t_stat,
        "shock_level_f_stat": t_stat ** 2 if math.isfinite(t_stat) else np.nan,
        "shock_level_n_shocks": n_shocks,
        "shock_level_effective_shocks": effective,
    }


def _run_first_stage_conditional_suite(
    panel: pd.DataFrame,
    instrument_col: str,
    x_col: str,
    reg_cfg: dict,
    out_dir: Path,
    use_log_y_panel: bool,
) -> None:
    """Run first-stage-only conditional size/growth diagnostics."""
    try:
        import pyfixest as pf
    except ImportError:
        print("[regressions] pyfixest not installed. Skipping conditional first-stage suite.")
        return

    baseline_start = int(reg_cfg.get("conditioning_baseline_window_start", 2008))
    baseline_end = int(reg_cfg.get("conditioning_baseline_window_end", 2013))
    current_size_bins = int(reg_cfg.get("current_size_bins", 10))
    current_growth_bins = int(reg_cfg.get("current_growth_bins", 5))
    joint_bins = int(reg_cfg.get("joint_size_growth_bins", 3))
    requested_outcomes = list(
        reg_cfg.get("first_stage_conditioning_outcomes", ["x_bin", "x_asinh", "x_rate_headcount"])
    )
    headline_spec = str(reg_cfg.get("headline_first_stage_spec", "joint_size_growth_year_fe")).strip()

    cond_panel = _prepare_first_stage_state_panel(
        panel,
        baseline_window_start=baseline_start,
        baseline_window_end=baseline_end,
        current_size_bins=current_size_bins,
        current_growth_bins=current_growth_bins,
        joint_size_growth_bins=joint_bins,
        baseline_growth_bins=int(reg_cfg.get("baseline_growth_bins", current_growth_bins)),
        use_log_y_panel=use_log_y_panel,
    )
    cond_panel = _prepare_first_stage_outcomes(cond_panel, x_col=x_col)

    required_state_cols = [
        instrument_col,
        "c",
        "t",
        "t_num",
        "headcount_lag0_raw",
        "headcount_lag1_raw",
        "headcount_size_baseline",
        "headcount_growth_asinh",
        "baseline_size_decile",
        "current_size_decile",
        "current_growth_quintile",
        "current_size_tercile",
        "current_growth_tercile",
        "baseline_size_year_fe",
        "baseline_growth_year_fe",
        "baseline_size_growth_year_fe",
        "current_size_year_fe",
        "current_growth_year_fe",
        "joint_size_growth_year_fe",
        "headcount_lag0_asinh",
        "headcount_lag0_asinh_sq",
        "headcount_growth_asinh_sq",
    ]
    state_mask = pd.Series(True, index=cond_panel.index)
    for col in required_state_cols:
        if col not in cond_panel.columns:
            state_mask &= False
            continue
        state_mask &= cond_panel[col].notna()
    cond_panel = cond_panel.loc[state_mask].copy()
    if cond_panel.empty:
        print("[regressions] conditional first-stage suite skipped: no rows with complete size/growth state data.")
        return

    cond_panel["c"] = cond_panel["c"].astype(str)
    cond_panel["t"] = cond_panel["t_num"].astype("Int64").astype(str)
    for col in (
        "baseline_size_year_fe",
        "baseline_growth_year_fe",
        "baseline_size_growth_year_fe",
        "current_size_year_fe",
        "current_growth_year_fe",
        "joint_size_growth_year_fe",
    ):
        cond_panel[col] = cond_panel[col].astype(str)

    observed_years = sorted(int(v) for v in cond_panel["t_num"].dropna().unique())
    continuous_control_cols: list[str] = []
    for stem, base_col in [
        ("headcount_size", "headcount_lag0_asinh"),
        ("headcount_size_sq", "headcount_lag0_asinh_sq"),
        ("headcount_growth", "headcount_growth_asinh"),
        ("headcount_growth_sq", "headcount_growth_asinh_sq"),
    ]:
        for yr in observed_years:
            col = f"{stem}_x_year_{int(yr)}"
            cond_panel[col] = cond_panel[base_col] * (cond_panel["t_num"] == int(yr)).astype(int)
            continuous_control_cols.append(col)

    conditional_rows: list[dict] = []
    for outcome in requested_outcomes:
        if outcome not in cond_panel.columns:
            print(f"[regressions] conditional first-stage outcome '{outcome}' not available; skipping.")
            continue
        outcome_panel = cond_panel.loc[cond_panel[outcome].notna()].copy()
        if outcome_panel.empty:
            continue
        specs = _build_first_stage_conditioning_specs(
            lhs=outcome,
            instrument_col=instrument_col,
            continuous_control_cols=continuous_control_cols,
            headline_first_stage_spec=headline_spec,
        )
        for spec in specs:
            try:
                fit = pf.feols(spec["formula"], data=outcome_panel, vcov={"CRV1": "c"})
                coef, se = _coef_for_term(fit, instrument_col)
                f_stat = _single_term_f_stat(coef, se)
                if f_stat is None:
                    f_stat = _extract_f_stat(fit)
                conditional_rows.append(
                    {
                        "instrument_col": instrument_col,
                        "outcome": outcome,
                        "spec_code": spec["spec_code"],
                        "spec_key": spec["spec_key"],
                        "added_conditioning": spec["added_conditioning"],
                        "formula": spec["formula"],
                        "is_headline_spec": bool(spec["is_headline"]),
                        "coef_instrument": coef,
                        "se_instrument": se,
                        "f_stat": f_stat,
                        "n_obs": _model_nobs(fit, len(outcome_panel)),
                        "n_companies": int(outcome_panel["c"].nunique()),
                    }
                )
            except Exception as e:
                conditional_rows.append(
                    {
                        "instrument_col": instrument_col,
                        "outcome": outcome,
                        "spec_code": spec["spec_code"],
                        "spec_key": spec["spec_key"],
                        "added_conditioning": spec["added_conditioning"],
                        "formula": spec["formula"],
                        "is_headline_spec": bool(spec["is_headline"]),
                        "error": str(e),
                    }
                )

    conditional_df = pd.DataFrame(conditional_rows)
    if conditional_df.empty:
        print("[regressions] conditional first-stage suite produced no estimates.")
        return

    conditional_path = out_dir / "first_stage_conditional_summary.csv"
    conditional_df.to_csv(conditional_path, index=False)

    printable = conditional_df.loc[
        conditional_df["error"].isna() if "error" in conditional_df.columns else conditional_df.index == conditional_df.index,
        ["outcome", "spec_code", "added_conditioning", "coef_instrument", "se_instrument", "f_stat", "n_obs"],
    ].copy()
    if not printable.empty:
        for col in ("coef_instrument", "se_instrument", "f_stat"):
            printable[col] = pd.to_numeric(printable[col], errors="coerce").round(6)
        print("\n── first-stage conditional summary ──")
        print(printable.to_string(index=False))
    print(f"[regressions] saved first-stage conditional summary → {conditional_path}")

    comparison_rows: list[dict] = []
    successful = conditional_df.loc[conditional_df.get("error").isna()] if "error" in conditional_df.columns else conditional_df
    if not successful.empty:
        for outcome, sub in successful.groupby("outcome"):
            row = {"outcome": outcome}
            for _, item in sub.sort_values("spec_code").iterrows():
                code = str(item["spec_code"]).lower()
                row[f"{code}_coef"] = item.get("coef_instrument")
                row[f"{code}_se"] = item.get("se_instrument")
                row[f"{code}_f"] = item.get("f_stat")
                row[f"{code}_n"] = item.get("n_obs")
            comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = out_dir / "first_stage_outcome_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"[regressions] saved first-stage outcome comparison → {comparison_path}")

    diag_panel = cond_panel.loc[cond_panel["x_bin"].notna()].copy()
    if diag_panel.empty:
        return

    heterogeneity_rows: list[dict] = []
    diag_plot_frames: list[pd.DataFrame] = []
    for kind, category_col, prefix, category_values, axis_name, title, filename in [
        (
            "baseline_size",
            "baseline_size_decile",
            "baseline_size_decile",
            list(range(1, current_size_bins + 1)),
            "baseline_size_decile",
            f"First Stage: {instrument_col} x Baseline Size Decile (TWFE)",
            "first_stage_interaction_baseline_size.png",
        ),
        (
            "current_size",
            "current_size_decile",
            "current_size_decile",
            list(range(1, current_size_bins + 1)),
            "current_size_decile",
            f"First Stage: {instrument_col} x Current Size Decile (TWFE)",
            "first_stage_interaction_current_size.png",
        ),
        (
            "growth",
            "current_growth_quintile",
            "current_growth_quintile",
            list(range(1, current_growth_bins + 1)),
            "current_growth_quintile",
            f"First Stage: {instrument_col} x Current Growth Quintile (TWFE)",
            "first_stage_interaction_growth.png",
        ),
    ]:
        design_panel, interaction_cols, term_specs = _create_instrument_category_interactions(
            diag_panel,
            instrument_col=instrument_col,
            category_col=category_col,
            prefix=prefix,
            category_values=category_values,
        )
        try:
            fit = pf.feols(f"x_bin ~ {' + '.join(interaction_cols)} | c + t", data=design_panel, vcov={"CRV1": "c"})
            plot_df = _build_interaction_plot_frame(
                coefs=fit.coef(),
                ses=fit.se(),
                term_specs=term_specs,
                omitted_value=None,
                axis_name=axis_name,
            )
            plot_df["interaction_kind"] = kind
            diag_plot_frames.append(plot_df.copy())
            _plot_interaction_coefficients(
                plot_df=plot_df,
                axis_name=axis_name,
                title=title,
                x_label=axis_name.replace("_", " ").title(),
                out_path=out_dir / filename,
            )
            heterogeneity_rows.append(
                {
                    "instrument_col": instrument_col,
                    "interaction_kind": kind,
                    "interaction_slope": _interaction_slope(plot_df, axis_name),
                    "n_obs": _model_nobs(fit, len(design_panel)),
                }
            )
        except Exception as e:
            print(f"[regressions] first-stage {kind} interaction diagnostic skipped: {e}")

    joint_panel, joint_cols, joint_specs = _create_instrument_joint_interactions(
        diag_panel,
        instrument_col=instrument_col,
        size_col="current_size_tercile",
        growth_col="current_growth_tercile",
        prefix="joint_size_growth",
        size_values=list(range(1, joint_bins + 1)),
        growth_values=list(range(1, joint_bins + 1)),
    )
    try:
        fit = pf.feols(f"x_bin ~ {' + '.join(joint_cols)} | c + t", data=joint_panel, vcov={"CRV1": "c"})
        joint_df = _build_joint_interaction_plot_frame(fit.coef(), fit.se(), joint_specs)
        joint_plot_path = out_dir / "first_stage_interaction_size_growth_heatmap.png"
        _plot_joint_interaction_heatmap(
            joint_df,
            title=f"First Stage: {instrument_col} x Current Size/Growth Cells (TWFE)",
            out_path=joint_plot_path,
        )
        diag_plot_frames.append(joint_df.assign(interaction_kind="joint_size_growth"))
        top_left = joint_df.loc[
            (joint_df["current_size_tercile"] == 1) & (joint_df["current_growth_tercile"] == joint_bins)
        ]
        bottom_right = joint_df.loc[
            (joint_df["current_size_tercile"] == joint_bins) & (joint_df["current_growth_tercile"] == 1)
        ]
        heterogeneity_rows.append(
            {
                "instrument_col": instrument_col,
                "interaction_kind": "joint_size_growth",
                "top_left_coef": float(top_left["coef"].iloc[0]) if not top_left.empty else np.nan,
                "bottom_right_coef": float(bottom_right["coef"].iloc[0]) if not bottom_right.empty else np.nan,
                "top_left_minus_bottom_right": (
                    float(top_left["coef"].iloc[0]) - float(bottom_right["coef"].iloc[0])
                    if (not top_left.empty and not bottom_right.empty)
                    else np.nan
                ),
                "n_obs": _model_nobs(fit, len(joint_panel)),
            }
        )
    except Exception as e:
        print(f"[regressions] first-stage joint size/growth diagnostic skipped: {e}")

    if heterogeneity_rows:
        heterogeneity_path = out_dir / "first_stage_heterogeneity_diagnostics.csv"
        pd.DataFrame(heterogeneity_rows).to_csv(heterogeneity_path, index=False)
        print(f"[regressions] saved first-stage heterogeneity diagnostics → {heterogeneity_path}")
    if diag_plot_frames:
        diag_frame_path = out_dir / "first_stage_interaction_plot_data.csv"
        pd.concat(diag_plot_frames, ignore_index=True).to_csv(diag_frame_path, index=False)
        print(f"[regressions] saved first-stage interaction plot data → {diag_frame_path}")

def run_regressions(
    analysis_panel_df: pd.DataFrame,
    cfg: dict,
    reg_cfg: dict,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    Run the baseline first-stage/reduced-form suite and the conditional first-stage
    decomposition diagnostics for a given instrument column.
    """
    try:
        import pyfixest as pf
    except ImportError:
        print("[regressions] pyfixest not installed. Skipping regressions.")
        return

    treatment_measure = cfg.get("treatment_measure", "correction_aware")
    x_col_map = {
        "raw": "masters_opt_hires",
        "valid": "valid_masters_opt_hires",
        "correction_aware": "masters_opt_hires_correction_aware",
    }
    x_col = x_col_map.get(treatment_measure, "masters_opt_hires_correction_aware")
    binary_rule = cfg.get("treatment_binary_rule", "any_nonzero")
    outcome_col = reg_cfg.get("outcome_col", "y_cst_lag0")
    instrument_col = str(reg_cfg.get("instrument_col", "z_ct")).strip()
    raw_instrument_col = instrument_col
    instrument_transform = _instrument_transform_for_column(reg_cfg, raw_instrument_col)
    regressor_col = _transformed_instrument_col(raw_instrument_col, instrument_transform)
    run_conditional_suite = _as_bool(reg_cfg.get("run_conditional_suite", True), default=True)
    run_interaction_specs = _as_bool(reg_cfg.get("run_interaction_specs", True), default=True)
    run_interaction_plots = _as_bool(reg_cfg.get("run_interaction_plots", True), default=True)
    run_dynamic_effect_plots = _as_bool(
        reg_cfg.get("run_dynamic_effect_plots", run_interaction_plots),
        default=run_interaction_plots,
    )
    run_distributed_lag_specs = _as_bool(
        reg_cfg.get("run_distributed_lag_specs", run_dynamic_effect_plots),
        default=run_dynamic_effect_plots,
    )
    run_residualized_binscatter = _as_bool(
        reg_cfg.get("run_residualized_binscatter", True),
        default=True,
    )
    residualized_binscatter_bins = int(reg_cfg.get("residualized_binscatter_bins", 50))
    write_outputs = _as_bool(reg_cfg.get("write_outputs", True), default=True)
    print_etable = _as_bool(reg_cfg.get("print_etable", run_interaction_plots), default=run_interaction_plots)
    first_stage_use_ppml = _as_bool(reg_cfg.get("first_stage_use_ppml", True), default=True)
    use_log_outcome = bool(reg_cfg.get("use_log_outcome", True))
    if str(outcome_col).startswith("avg_tenure_years"):
        use_log_outcome = _as_bool(reg_cfg.get("use_log_tenure_outcome", False), default=False)
    use_log_y_panel = bool(cfg.get("use_log_y", False))
    absorb_baseline_size_year_fe = _as_bool(
        reg_cfg.get("absorb_baseline_size_year_fe", True),
        default=True,
    )
    absorb_baseline_size_growth_year_fe = _as_bool(
        reg_cfg.get("absorb_baseline_size_growth_year_fe", False),
        default=False,
    )
    start_t = reg_cfg.get("start_t", 2008)
    end_t = reg_cfg.get("end_t", 2022)
    enforce_balanced = bool(reg_cfg.get("enforce_balanced_panel", True))

    panel = analysis_panel_df.copy()
    if raw_instrument_col not in panel.columns:
        print(f"[regressions] Instrument column '{raw_instrument_col}' not found. Skipping.")
        return None
    panel = panel[panel[raw_instrument_col].notna()].copy()
    if outcome_col in panel.columns:
        panel = panel[panel[outcome_col].notna()].copy()

    panel["t_num"] = pd.to_numeric(panel["t"], errors="coerce")
    panel = panel[panel["t_num"].notna()].copy()
    panel, regressor_col = _add_transformed_instrument(panel, raw_instrument_col, instrument_transform)
    dynamic_lookup_panel = panel.copy()

    absorbed_fe_cols = ["c", "t"]
    if absorb_baseline_size_year_fe or absorb_baseline_size_growth_year_fe:
        panel = _prepare_first_stage_state_panel(
            panel,
            baseline_window_start=int(reg_cfg.get("conditioning_baseline_window_start", 2008)),
            baseline_window_end=int(reg_cfg.get("conditioning_baseline_window_end", 2013)),
            current_size_bins=int(reg_cfg.get("current_size_bins", 10)),
            current_growth_bins=int(reg_cfg.get("current_growth_bins", 5)),
            joint_size_growth_bins=int(reg_cfg.get("joint_size_growth_bins", 3)),
            baseline_growth_bins=int(reg_cfg.get("baseline_growth_bins", reg_cfg.get("current_growth_bins", 5))),
            use_log_y_panel=use_log_y_panel,
        )
        if (
            absorb_baseline_size_growth_year_fe
            and "baseline_size_growth_year_fe" in panel.columns
            and panel["baseline_size_growth_year_fe"].notna().any()
        ):
            absorbed_fe_cols.append("baseline_size_growth_year_fe")
        elif absorb_baseline_size_growth_year_fe:
            print("[regressions] baseline-size × baseline-growth × year FE requested but unavailable.")
        if (
            absorb_baseline_size_year_fe
            and not absorb_baseline_size_growth_year_fe
            and "baseline_size_year_fe" in panel.columns
            and panel["baseline_size_year_fe"].notna().any()
        ):
            absorbed_fe_cols.append("baseline_size_year_fe")
        elif absorb_baseline_size_year_fe and not absorb_baseline_size_growth_year_fe:
            print("[regressions] baseline-size × year FE requested but unavailable; using firm + year FE only.")

    if start_t is not None:
        panel = panel[panel["t_num"] >= float(start_t)].copy()
    if end_t is not None:
        panel = panel[panel["t_num"] <= float(end_t)].copy()
    panel = panel.loc[panel[regressor_col].notna()].copy()

    # Derive x_bin.
    if binary_rule == "topbot_quartile":
        t_key = panel["t_num"].astype("Int64")
        q25 = panel.groupby(t_key)[x_col].transform(lambda s: s.quantile(0.25))
        q75 = panel.groupby(t_key)[x_col].transform(lambda s: s.quantile(0.75))
        mask = (panel[x_col] <= q25) | (panel[x_col] >= q75)
        panel = panel[mask].copy()
        q75_by_year = panel.groupby(t_key)[x_col].quantile(0.75)
        panel["x_bin"] = (panel[x_col] >= panel["t_num"].astype("Int64").map(q75_by_year)).astype(int)
    else:
        panel["x_bin"] = _make_x_bin(panel, x_col, binary_rule).values
    panel[x_col] = pd.to_numeric(panel[x_col], errors="coerce")
    if first_stage_use_ppml:
        panel = panel.loc[panel[x_col].notna() & panel[x_col].ge(0)].copy()

    # Log1p outcome.
    y_reg_col = outcome_col
    if use_log_outcome and outcome_col in panel.columns:
        y_reg_col = f"log1p_{outcome_col}"
        panel[y_reg_col] = panel[outcome_col].apply(lambda v: math.log1p(max(v, 0)) if pd.notna(v) else float("nan"))

    # Enforce balanced panel.
    if enforce_balanced:
        years_in_panel = sorted(panel["t_num"].dropna().unique())
        n_years = len(years_in_panel)
        obs_per_company = panel.groupby("c")["t_num"].nunique()
        balanced_cos = obs_per_company[obs_per_company == n_years].index
        panel = panel[panel["c"].isin(balanced_cos)].copy()
        print(f"[regressions] balanced panel: {panel['c'].nunique():,} companies × {n_years} years = {len(panel):,} obs")

    if panel.empty or "x_bin" not in panel.columns or x_col not in panel.columns:
        print("[regressions] Empty panel after filtering. Skipping.")
        return None

    # String/category identifiers for pyfixest and interaction creation.
    panel["c"] = panel["c"].astype(str)
    panel["t"] = panel["t_num"].astype("Int64").astype(str)
    panel = panel.loc[panel[[col for col in absorbed_fe_cols if col in panel.columns]].notna().all(axis=1)].copy()
    if panel.empty:
        print("[regressions] Empty panel after fixed-effect filtering. Skipping.")
        return None
    for fe_col in absorbed_fe_cols:
        if fe_col in panel.columns:
            panel[fe_col] = panel[fe_col].astype(str)
    dynamic_lookup_panel = dynamic_lookup_panel.copy()
    dynamic_lookup_panel["c"] = dynamic_lookup_panel["c"].astype(str)
    dynamic_lookup_panel["t_num"] = pd.to_numeric(dynamic_lookup_panel["t_num"], errors="coerce")
    dynamic_lookup_panel = dynamic_lookup_panel.loc[dynamic_lookup_panel["c"].isin(panel["c"].unique())].copy()
    twfe_term = _build_absorb_fe_term(absorbed_fe_cols)
    twfe_desc = "year + company FE"
    if "baseline_size_growth_year_fe" in absorbed_fe_cols:
        twfe_desc += " + baseline-size-decile × baseline-growth-quantile × year FE"
    elif "baseline_size_year_fe" in absorbed_fe_cols:
        twfe_desc += " + baseline-size-decile × year FE"

    observed_years = sorted(int(v) for v in panel["t_num"].dropna().unique())
    if not observed_years:
        print("[regressions] No valid years remain after filtering. Skipping.")
        return None

    year_interaction_cols: list[str] = []
    year_term_specs: list[tuple[int, str]] = []
    for yr in observed_years:
        yr = int(yr)
        col = f"{regressor_col}_x_year_{yr}"
        panel[col] = panel[regressor_col] * (panel["t_num"] == yr).astype(int)
        year_interaction_cols.append(col)
        year_term_specs.append((yr, col))

    def _build_formula(lhs: str, rhs_terms: list[str], fe_term: str) -> str:
        rhs = " + ".join(rhs_terms)
        return f"{lhs} ~ {rhs} {fe_term}" if fe_term else f"{lhs} ~ {rhs}"

    first_stage_lhs = x_col if first_stage_use_ppml else "x_bin"
    first_stage_estimator = "ppml" if first_stage_use_ppml else "ols_lpm"

    specs = [
        ("first_stage_no_fe", first_stage_lhs, _build_formula(first_stage_lhs, [regressor_col], ""), "no FE", first_stage_estimator),
        (
            "first_stage_twfe",
            first_stage_lhs,
            _build_formula(first_stage_lhs, [regressor_col], twfe_term),
            twfe_desc,
            first_stage_estimator,
        ),
        ("reduced_form_twfe", y_reg_col, _build_formula(y_reg_col, [regressor_col], twfe_term), twfe_desc, "ols"),
    ]
    if run_interaction_specs:
        specs.extend([
            (
                "first_stage_no_fe_interact_year",
                first_stage_lhs,
                _build_formula(first_stage_lhs, year_interaction_cols, ""),
                f"no FE, interact {regressor_col} × year",
                first_stage_estimator,
            ),
            (
                "reduced_form_no_fe_interact_year",
                y_reg_col,
                _build_formula(y_reg_col, year_interaction_cols, ""),
                f"no FE, interact {regressor_col} × year",
                "ols",
            ),
            (
                "first_stage_twfe_interact_year",
                first_stage_lhs,
                _build_formula(first_stage_lhs, year_interaction_cols, twfe_term),
                f"{twfe_desc}, interact {regressor_col} × year",
                first_stage_estimator,
            ),
            (
                "reduced_form_twfe_interact_year",
                y_reg_col,
                _build_formula(y_reg_col, year_interaction_cols, twfe_term),
                f"{twfe_desc}, interact {regressor_col} × year",
                "ols",
            ),
        ])

    # Keep specs valid if interaction lists are somehow empty.
    specs = [s for s in specs if s[2] is not None]
    specs = [s for s in specs if " +  |" not in s[2] and " ~  " not in s[2] and "+ | c + t" not in s[2]]

    print("\n" + "=" * 64)
    print("REGRESSION RESULTS")
    print(
        f"  x (first stage): {x_col} "
        f"({'PPML continuous count' if first_stage_use_ppml else 'LPM binary ' + binary_rule})"
    )
    print(f"  y (reduced form): {y_reg_col}  (log_outcome={use_log_outcome})")
    print(f"  z (regressor): {regressor_col}  (raw={raw_instrument_col}, transform={instrument_transform})")
    print(f"  absorbed FE: {' + '.join(absorbed_fe_cols)}")
    print(f"  interaction specs enabled: {run_interaction_specs}")
    print(f"  dynamic effect plots enabled: {run_dynamic_effect_plots}")
    print(f"  distributed-lag robustness enabled: {run_distributed_lag_specs}")
    print(f"  residualized binned scatter enabled: {run_residualized_binscatter}")
    print(f"  n_obs={len(panel):,}, n_companies={panel['c'].nunique():,}")
    print("=" * 64 + "\n")

    fits = []
    fit_lookup: dict[str, object] = {}
    results: list[dict] = []
    interaction_plot_specs: dict[str, dict[str, object]] = {
        "first_stage_no_fe_interact_year": {
            "term_specs": year_term_specs,
            "axis_name": "year",
            "reference_value": None,
            "title": f"First Stage: {regressor_col} x Year (no FE)\nOne coefficient per year",
            "x_label": "Year",
        },
        "reduced_form_no_fe_interact_year": {
            "term_specs": year_term_specs,
            "axis_name": "year",
            "reference_value": None,
            "title": f"Reduced Form: {regressor_col} x Year (no FE)\nOne coefficient per year",
            "x_label": "Year",
        },
        "first_stage_twfe_interact_year": {
            "term_specs": year_term_specs,
            "axis_name": "year",
            "reference_value": None,
            "title": f"First Stage: {regressor_col} x Year (TWFE)\nOne coefficient per year",
            "x_label": "Year",
        },
        "reduced_form_twfe_interact_year": {
            "term_specs": year_term_specs,
            "axis_name": "year",
            "reference_value": None,
            "title": f"Reduced Form: {regressor_col} x Year (TWFE)\nOne coefficient per year",
            "x_label": "Year",
        },
    }
    for label, lhs, fml, fe_desc, estimator in specs:
        try:
            fit = (
                pf.fepois(fml, data=panel, vcov={"CRV1": "c"})
                if estimator == "ppml"
                else pf.feols(fml, data=panel, vcov={"CRV1": "c"})
            )
            fits.append(fit)
            fit_lookup[label] = fit
            coef, se = _coef_for_term(fit, regressor_col)
            f_stat = _single_term_f_stat(coef, se)
            if f_stat is None:
                f_stat = _extract_f_stat(fit)
            nobs = _model_nobs(fit, len(panel))
            t_stat = round(coef / se, 3) if coef is not None and se is not None and se > 0 else None
            result_row = {
                "label": label, "lhs": lhs, "fe": fe_desc,
                "estimator": estimator,
                "instrument_col": raw_instrument_col,
                "regressor_col": regressor_col,
                "instrument_transform": instrument_transform,
                "coef_instrument": round(coef, 6) if coef is not None else None,
                "se_instrument": round(se, 6) if se is not None else None,
                "t_stat": t_stat,
                "f_stat": round(f_stat, 6) if f_stat is not None else None,
                "n_obs": nobs,
            }
            if (
                label in {"first_stage_twfe", "reduced_form_twfe"}
                and coef is not None
                and estimator != "ppml"
                and regressor_col == raw_instrument_col
            ):
                shock_inf = _shock_level_score_inference(
                    panel=panel,
                    lhs=lhs,
                    instrument_col=raw_instrument_col,
                    coef=coef,
                    fe_cols=absorbed_fe_cols,
                )
                if shock_inf:
                    result_row.update({
                        key: round(val, 6) if isinstance(val, float) and math.isfinite(val) else val
                        for key, val in shock_inf.items()
                    })
            if (
                write_outputs
                and run_residualized_binscatter
                and label in {"first_stage_no_fe", "first_stage_twfe", "reduced_form_twfe"}
                and coef is not None
            ):
                scatter_fe_cols = [] if label.endswith("_no_fe") else absorbed_fe_cols
                scatter_paths = _run_residualized_binscatter(
                    panel=panel,
                    lhs=lhs,
                    instrument_col=regressor_col,
                    fe_cols=scatter_fe_cols,
                    label=label,
                    estimator=estimator,
                    model_coef=coef,
                    model_se=se,
                    out_dir=out_dir,
                    n_bins=residualized_binscatter_bins,
                )
                if scatter_paths is not None:
                    plot_path, data_path = scatter_paths
                    result_row["residualized_binscatter_plot"] = str(plot_path)
                    result_row["residualized_binscatter_data"] = str(data_path)
            results.append(result_row)
            if coef is not None and se is not None:
                if f_stat is not None:
                    shock_msg = ""
                    if "shock_level_se_instrument" in result_row:
                        shock_msg = (
                            f", shock-se={result_row['shock_level_se_instrument']}, "
                            f"shock-F={result_row.get('shock_level_f_stat')}"
                        )
                    print(
                        f"  {label} ({fe_desc}): coef={coef:.4f} (se={se:.4f}, t={t_stat}, "
                        f"F={f_stat:.4f}{shock_msg}, n={nobs:,})"
                    )
                else:
                    print(f"  {label} ({fe_desc}): coef={coef:.4f} (se={se:.4f}, t={t_stat}, n={nobs:,})")
            elif f_stat is not None:
                print(f"  {label} ({fe_desc}): F={f_stat:.4f}, n={nobs:,}")
            else:
                print(f"  {label} ({fe_desc}): coefficient unavailable, n={nobs:,}")
        except Exception as e:
            print(f"  {label}: ERROR — {e}")
            results.append({"label": label, "lhs": lhs, "fe": fe_desc, "estimator": estimator, "error": str(e)})

    if fits and print_etable:
        print("\n── Regression table ──")
        try:
            pf.etable(fits)
        except Exception as e:
            print(f"  etable failed: {e}")

    result_df = pd.DataFrame(results)
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        reg_table_path = out_dir / "reg_table.csv"
        result_df.to_csv(reg_table_path, index=False)
        print(f"\n[regressions] saved coefficient table → {reg_table_path}")

    interaction_plot_rows: list[pd.DataFrame] = []
    for label, meta in (interaction_plot_specs.items() if run_interaction_plots else []):
        fit = fit_lookup.get(label)
        if fit is None:
            continue
        try:
            plot_df = _build_interaction_plot_frame(
                coefs=fit.coef(),
                ses=fit.se(),
                term_specs=meta["term_specs"],
                omitted_value=(
                    int(meta["reference_value"])
                    if meta["reference_value"] is not None
                    else None
                ),
                axis_name=str(meta["axis_name"]),
            )
            plot_df["spec_label"] = label
            plot_df["reference_value"] = meta["reference_value"]
            interaction_plot_rows.append(plot_df.copy())
            plot_path = out_dir / f"{label}_coefplot.png"
            _plot_interaction_coefficients(
                plot_df=plot_df,
                axis_name=str(meta["axis_name"]),
                title=str(meta["title"]),
                x_label=str(meta["x_label"]),
                out_path=plot_path,
            )
            print(f"[regressions] saved interaction coefficient plot → {plot_path}")
        except Exception as e:
            print(f"[regressions] interaction coefficient plot skipped for {label}: {e}")

    if interaction_plot_rows and write_outputs:
        interaction_table_path = out_dir / "interaction_coefficient_plot_data.csv"
        pd.concat(interaction_plot_rows, ignore_index=True).to_csv(interaction_table_path, index=False)
        print(f"[regressions] saved interaction coefficient data → {interaction_table_path}")

    if run_dynamic_effect_plots and write_outputs:
        _run_dynamic_effect_plots(
            panel=panel,
            instrument_col=regressor_col,
            x_col=x_col,
            outcome_col=outcome_col,
            use_log_outcome=use_log_outcome,
            first_stage_use_ppml=first_stage_use_ppml,
            horizon_start=int(reg_cfg.get("dynamic_horizon_start", -4)),
            horizon_end=int(reg_cfg.get("dynamic_horizon_end", 5)),
            out_dir=out_dir,
            fe_cols=absorbed_fe_cols,
            x_lookup_panel=dynamic_lookup_panel,
        )

    if run_distributed_lag_specs and write_outputs:
        _run_distributed_lag_robustness(
            panel=panel,
            instrument_col=regressor_col,
            x_col=x_col,
            y_reg_col=y_reg_col,
            first_stage_use_ppml=first_stage_use_ppml,
            max_lag=int(reg_cfg.get("distributed_lag_max_lag", 3)),
            out_dir=out_dir,
            fe_cols=absorbed_fe_cols,
            lag_lookup_panel=dynamic_lookup_panel,
        )

    if run_conditional_suite:
        _run_first_stage_conditional_suite(
            panel=panel,
            instrument_col=regressor_col,
            x_col=x_col,
            reg_cfg=reg_cfg,
            out_dir=out_dir,
            use_log_y_panel=use_log_y_panel,
        )
    else:
        print(
            "[regressions] conditional first-stage size/growth diagnostics skipped "
            f"for {instrument_col}."
        )
    return result_df


def run_regression_variants(
    analysis_panel_df: pd.DataFrame,
    cfg: dict,
    reg_cfg: dict,
    out_dir: Path,
) -> None:
    """Run core regressions for variants, with full diagnostics only for z_ct/main."""
    instrument_cols = _select_regression_instrument_cols(reg_cfg, list(analysis_panel_df.columns))
    if not instrument_cols:
        print("[regressions] No requested instrument variants are available in the analysis panel.")
        return
    outcome_cols = _select_regression_outcome_cols(reg_cfg, list(analysis_panel_df.columns))
    if not outcome_cols:
        print("[regressions] No requested reduced-form outcomes are available in the analysis panel.")
        return

    main_col = str(reg_cfg.get("instrument_col", "z_ct")).strip() or "z_ct"
    if main_col not in instrument_cols:
        main_col = "z_ct" if "z_ct" in instrument_cols else instrument_cols[0]
    primary_outcome = str(reg_cfg.get("outcome_col", outcome_cols[0])).strip() or outcome_cols[0]
    if primary_outcome not in outcome_cols:
        primary_outcome = outcome_cols[0]
    run_main_conditional_suite = _as_bool(
        reg_cfg.get("run_conditional_suite", True),
        default=True,
    )
    run_main_interaction_specs = _as_bool(
        reg_cfg.get("run_interaction_specs", True),
        default=True,
    )
    run_main_interaction_plots = _as_bool(
        reg_cfg.get("run_interaction_plots", run_main_interaction_specs),
        default=run_main_interaction_specs,
    )
    configured_dynamic_cols = _parse_config_string_list(
        reg_cfg.get("dynamic_instrument_cols", [main_col])
    )
    dynamic_instrument_cols = {
        col for col in configured_dynamic_cols
        if col in instrument_cols
    }
    dynamic_instrument_cols.add(main_col)
    configured_distributed_lag_cols = _parse_config_string_list(
        reg_cfg.get("distributed_lag_instrument_cols", configured_dynamic_cols)
    )
    distributed_lag_instrument_cols = {
        col for col in configured_distributed_lag_cols
        if col in instrument_cols
    }
    print(
        "[regressions] variant policy: full tables/plots only for preferred instrument "
        f"'{main_col}'; alternate instruments are consolidated in one summary table."
    )
    print(f"[regressions] reduced-form outcomes: {outcome_cols}")
    print(f"[regressions] dynamic instrument outputs: {sorted(dynamic_instrument_cols)}")
    print(f"[regressions] distributed-lag instrument outputs: {sorted(distributed_lag_instrument_cols)}")
    all_results: list[pd.DataFrame] = []
    for outcome_col in outcome_cols:
        is_primary_outcome = outcome_col == primary_outcome
        for instrument_col in instrument_cols:
            is_main = instrument_col == main_col
            run_dynamic_for_variant = bool(is_primary_outcome and instrument_col in dynamic_instrument_cols)
            run_distributed_lag_for_variant = bool(
                is_primary_outcome and instrument_col in distributed_lag_instrument_cols
            )
            write_full_outputs = bool(is_main or run_dynamic_for_variant or run_distributed_lag_for_variant)
            variant_cfg = dict(reg_cfg)
            variant_cfg["instrument_col"] = instrument_col
            variant_cfg["outcome_col"] = outcome_col
            variant_cfg["run_conditional_suite"] = bool(
                run_main_conditional_suite and is_main and is_primary_outcome
            )
            variant_cfg["run_interaction_specs"] = bool(is_main and run_main_interaction_specs)
            variant_cfg["run_interaction_plots"] = bool(is_main and run_main_interaction_plots)
            variant_cfg["run_dynamic_effect_plots"] = run_dynamic_for_variant
            variant_cfg["run_distributed_lag_specs"] = run_distributed_lag_for_variant
            variant_cfg["print_etable"] = bool(is_main)
            variant_cfg["write_outputs"] = write_full_outputs
            if is_main and is_primary_outcome:
                variant_out_dir = out_dir
            elif is_main:
                variant_out_dir = out_dir / "outcome_variants" / _safe_path_component(outcome_col)
            else:
                variant_out_dir = out_dir / "regression_variants" / _safe_path_component(outcome_col) / instrument_col
            if is_main:
                mode = "full preferred-spec outputs"
            elif run_dynamic_for_variant and run_distributed_lag_for_variant:
                mode = "dynamic/distributed-lag outputs + consolidated table"
            elif run_dynamic_for_variant:
                mode = "dynamic outputs + consolidated table"
            elif run_distributed_lag_for_variant:
                mode = "distributed-lag outputs + consolidated table"
            else:
                mode = "consolidated-table only"
            print(f"\n[regressions] running outcome '{outcome_col}', variant '{instrument_col}' ({mode})")
            variant_table = run_regressions(analysis_panel_df, cfg, variant_cfg, variant_out_dir)
            if isinstance(variant_table, pd.DataFrame) and not variant_table.empty:
                variant_table = variant_table.copy()
                variant_table["instrument_variant"] = instrument_col
                variant_table["outcome_col"] = outcome_col
                variant_table["output_dir"] = str(variant_out_dir) if write_full_outputs else ""
                all_results.append(variant_table)
                continue

            # Backward-compatible fallback for tests/mocks and older callers.
            reg_table_path = variant_out_dir / "reg_table.csv"
            if reg_table_path.exists():
                try:
                    variant_table = pd.read_csv(reg_table_path)
                    variant_table["instrument_variant"] = instrument_col
                    variant_table["outcome_col"] = outcome_col
                    variant_table["output_dir"] = str(variant_out_dir) if write_full_outputs else ""
                    all_results.append(variant_table)
                except Exception as e:
                    print(f"[regressions] could not read {reg_table_path}: {e}")

    if all_results:
        summary_df = pd.concat(all_results, ignore_index=True)
        summary_path = out_dir / "regression_variant_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[regressions] saved combined variant summary → {summary_path}")
        headline_df = _build_regression_variant_headline_summary(summary_df)
        if not headline_df.empty:
            headline_path = out_dir / "regression_variant_headline_summary.csv"
            headline_df.to_csv(headline_path, index=False)
            print("[regressions] variant headline summary (TWFE headline specs)")
            print(headline_df.to_string(index=False))
            print(f"[regressions] saved headline summary → {headline_path}")


def run_falsification_tests(
    con: ddb.DuckDBPyConnection,
    analysis_panel_df: pd.DataFrame,
    cfg: dict,
    reg_cfg: dict,
    out_dir: Path,
) -> None:
    """Run true placebo / falsification tests on pre-period data."""
    fals_dir = out_dir / "falsification_tests"
    fals_dir.mkdir(parents=True, exist_ok=True)

    pre_period_end = int(reg_cfg.get("falsification_pre_period_end", int(cfg.get("school_sample_window_start", 2014)) - 1))
    main_col = str(reg_cfg.get("instrument_col", "z_ct")).strip() or "z_ct"
    fake_timing_cols = [
        col for col in ["z_ct_falsification_lead4_broad"]
        if main_col == "z_ct" and col in analysis_panel_df.columns
    ]
    if not fake_timing_cols:
        print(
            "[falsification] fake-timing columns not found in analysis_panel. "
            "If you are loading saved panels, rebuild them to generate the new falsification shocks."
        )
    future_exposure_df = pd.DataFrame()
    try:
        future_exposure_df = con.sql("""
            SELECT
                CAST(s.c AS VARCHAR) AS c,
                SUM(COALESCE(s.share_ck, 0) * COALESCE(g.shock_quantity_step, 0)) AS future_exposure_step,
                SUM(COALESCE(s.share_ck, 0) * COALESCE(g.shock_quantity_matched_step, 0)) AS future_exposure_matched_step
            FROM transition_shares s
            JOIN (
                SELECT
                    CAST(k AS VARCHAR) AS k,
                    MAX(COALESCE(g_kt_v1_broad_step, 0)) AS shock_quantity_step,
                    MAX(COALESCE(g_kt_v4_matched_step, 0)) AS shock_quantity_matched_step
                FROM ipeds_unit_growth
                GROUP BY 1
            ) g ON CAST(s.k AS VARCHAR) = g.k
            GROUP BY 1
        """).df()
    except Exception as e:
        print(f"[falsification] future exposure build skipped: {e}")

    pretrend_cols = [
        col for col in ["future_exposure_step"]
    ]
    all_results: list[pd.DataFrame] = []

    if fake_timing_cols:
        print(f"\n[falsification] fake-timing regressions on pre-period sample through {pre_period_end}")
    for instrument_col in fake_timing_cols:
        variant_cfg = dict(reg_cfg)
        variant_cfg["instrument_col"] = instrument_col
        variant_cfg["run_conditional_suite"] = False
        variant_cfg["start_t"] = reg_cfg.get("start_t", 2008)
        variant_cfg["end_t"] = pre_period_end
        variant_out_dir = fals_dir / instrument_col
        print(f"[falsification] running fake-timing variant '{instrument_col}' → {variant_out_dir}")
        run_regressions(analysis_panel_df, cfg, variant_cfg, variant_out_dir)
        reg_table_path = variant_out_dir / "reg_table.csv"
        if reg_table_path.exists():
            try:
                variant_table = pd.read_csv(reg_table_path)
                variant_table["instrument_variant"] = instrument_col
                variant_table["falsification_family"] = "fake_timing_preperiod"
                variant_table["output_dir"] = str(variant_out_dir)
                all_results.append(variant_table)
            except Exception as e:
                print(f"[falsification] could not read {reg_table_path}: {e}")

    if not future_exposure_df.empty:
        print(f"\n[falsification] future-exposure pretrend interaction tests through {pre_period_end}")
        future_panel = analysis_panel_df.copy()
        future_panel["c"] = future_panel["c"].astype(str)
        future_panel = future_panel.merge(future_exposure_df, on="c", how="left")
        for col in pretrend_cols:
            if col in future_panel.columns:
                future_panel[col] = pd.to_numeric(future_panel[col], errors="coerce").fillna(0.0)
        for instrument_col in [c for c in pretrend_cols if c in future_panel.columns]:
            variant_cfg = dict(reg_cfg)
            variant_cfg["instrument_col"] = instrument_col
            variant_cfg["run_conditional_suite"] = False
            variant_cfg["start_t"] = reg_cfg.get("start_t", 2008)
            variant_cfg["end_t"] = pre_period_end
            variant_out_dir = fals_dir / instrument_col
            print(
                f"[falsification] running future-exposure pretrend spec '{instrument_col}' → "
                f"{variant_out_dir}"
            )
            print("[falsification] interpret year-interaction plots here as pre-trend tests; ignore the plain TWFE main term.")
            run_regressions(future_panel, cfg, variant_cfg, variant_out_dir)
            reg_table_path = variant_out_dir / "reg_table.csv"
            if reg_table_path.exists():
                try:
                    variant_table = pd.read_csv(reg_table_path)
                    variant_table["instrument_variant"] = instrument_col
                    variant_table["falsification_family"] = "future_exposure_pretrend"
                    variant_table["output_dir"] = str(variant_out_dir)
                    all_results.append(variant_table)
                except Exception as e:
                    print(f"[falsification] could not read {reg_table_path}: {e}")

    if all_results:
        summary_df = pd.concat(all_results, ignore_index=True)
        summary_path = fals_dir / "falsification_regression_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[falsification] saved combined falsification summary → {summary_path}")

        headline_df = _build_regression_variant_headline_summary(
            summary_df.loc[summary_df["falsification_family"] == "fake_timing_preperiod"].copy()
        )
        if not headline_df.empty:
            headline_path = fals_dir / "falsification_headline_summary.csv"
            headline_df.to_csv(headline_path, index=False)
            print("[falsification] fake-timing headline summary (pre-period TWFE specs)")
            print(headline_df.to_string(index=False))
            print(f"[falsification] saved headline summary → {headline_path}")


# =============================================================================
# ── MAIN ──────────────────────────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    t_total = time.time()

    # ── 0. Config ─────────────────────────────────────────────────────────────
    print("=" * 64)
    print("SHIFT-SHARE ANALYSIS — APRIL 2026")
    print("=" * 64)
    cfg = load_config(_CONFIG_PATH)
    paths_cfg = get_cfg_section(cfg, "paths")
    pipeline_cfg = get_cfg_section(cfg, "pipeline")
    reg_cfg = get_cfg_section(cfg, "regressions")
    output_cfg = get_cfg_section(cfg, "output")
    testing_cfg = get_cfg_section(cfg, "testing")

    out_dir = Path(paths_cfg["out_dir"])
    diag_dir = Path(paths_cfg.get("diagnostics_dir", str(out_dir / "diagnostics")))
    save_panel = bool(output_cfg.get("save_panel", True))
    plot_diagnostics = bool(output_cfg.get("plot_diagnostics", True))
    skip_panel_rebuild = _as_bool(output_cfg.get("skip_panel_rebuild", False), default=False)

    # Pipeline params.
    opt_shifts = bool(pipeline_cfg.get("opt_shifts", True))
    shock_design = _normalize_shock_design(pipeline_cfg.get("shock_design", "event_quantity"))
    school_sample_mode = _normalize_school_sample_mode(pipeline_cfg.get("school_sample_mode", "matched_shift_sample"))
    school_shift_metric = _normalize_school_shift_metric(pipeline_cfg.get("school_shift_metric", "ihmp_share"))
    share_period = _normalize_share_period(pipeline_cfg.get("share_period", "pre_window"))
    share_base_year = int(pipeline_cfg.get("share_base_year", 2010))
    share_year_min = int(pipeline_cfg.get("share_year_min", 2008))
    share_year_max = int(pipeline_cfg.get("share_year_max", 2013))
    share_robustness_windows = _parse_share_robustness_windows(
        pipeline_cfg.get("share_robustness_windows")
    )
    share_min_universities_for_share = int(pipeline_cfg.get("share_min_universities_for_share", 1))
    school_sample_window_start = int(pipeline_cfg.get("school_sample_window_start", 2014))
    school_sample_window_end = int(pipeline_cfg.get("school_sample_window_end", 2017))
    event_shock_pre_years = int(pipeline_cfg.get("event_shock_pre_years", 2))
    event_shock_post_years = int(pipeline_cfg.get("event_shock_post_years", 2))
    common_baseline_start = int(pipeline_cfg.get("common_baseline_start", 2011))
    common_baseline_end = int(pipeline_cfg.get("common_baseline_end", 2013))
    require_common_baseline_full_coverage = bool(
        pipeline_cfg.get("require_common_baseline_full_coverage", True)
    )
    growth_population = str(pipeline_cfg.get("growth_population", "main")).strip().lower()
    opt_shifts_degree_scope = _normalize_degree_scope(pipeline_cfg.get("opt_shifts_degree_scope", "bachelors_masters"))
    opt_shifts_normalization = _normalize_opt_shift_normalization(pipeline_cfg.get("opt_shifts_normalization", "none"))
    use_changes = bool(pipeline_cfg.get("use_changes", False))
    use_log_y = bool(pipeline_cfg.get("use_log_y", False))
    include_non_masters = bool(pipeline_cfg.get("include_non_masters", False))
    include_bachelors = bool(pipeline_cfg.get("include_bachelors", False))
    lag_start = int(pipeline_cfg.get("outcome_lag_start", -5))
    lag_end = int(pipeline_cfg.get("outcome_lag_end", 5))
    sample_year_min = int(pipeline_cfg.get("sample_year_min", 2008))
    sample_year_max = int(pipeline_cfg.get("sample_year_max", 2022))
    conditioning_baseline_window_start = int(reg_cfg.get("conditioning_baseline_window_start", 2008))
    conditioning_baseline_window_end = int(reg_cfg.get("conditioning_baseline_window_end", 2013))
    falsification_lead_years = int(pipeline_cfg.get("falsification_lead_years", 4))
    min_active_shock_schools = int(pipeline_cfg.get("min_active_shock_schools", 2))
    require_balanced_panel = bool(pipeline_cfg.get("require_balanced_panel", True))
    exclude_unitids = _parse_unitids(pipeline_cfg.get("exclude_unitids"))
    confirm_sample = bool(pipeline_cfg.get("confirm_matched_school_sample", False))
    match_school_pairs_by_carnegie_classification = bool(
        pipeline_cfg.get("match_school_pairs_by_carnegie_classification", False)
    )
    restrict_treated_to_no_large_enrollment_jump = bool(
        pipeline_cfg.get("restrict_treated_to_no_large_enrollment_jump", False)
    )
    school_sample_max_yoy_size_jump = float(
        pipeline_cfg.get("school_sample_max_yoy_size_jump", 0.50)
    )
    degree_scope_for_matched = "bachelors_masters" if include_bachelors else "masters"

    if use_changes and use_log_y:
        raise ValueError("use_changes and use_log_y are mutually exclusive.")

    print(
        f"Config: shock_design={shock_design}, opt_shifts={opt_shifts}, "
        f"school_sample_mode={school_sample_mode}, school_shift_metric={school_shift_metric}"
    )
    print(f"        match_school_pairs_by_carnegie_classification={match_school_pairs_by_carnegie_classification}")
    print(
        "        restrict_treated_to_no_large_enrollment_jump="
        f"{restrict_treated_to_no_large_enrollment_jump}, "
        f"school_sample_max_yoy_size_jump={school_sample_max_yoy_size_jump}"
    )
    print(
        f"        share_period={share_period}, share_year_min={share_year_min}, "
        f"share_year_max={share_year_max}, "
        f"share_min_universities_for_share={share_min_universities_for_share}, "
        f"treatment_measure={pipeline_cfg.get('treatment_measure','correction_aware')}"
    )
    print(
        f"        event_shock_pre_years={event_shock_pre_years}, "
        f"event_shock_post_years={event_shock_post_years}, "
        f"common_baseline={common_baseline_start}-{common_baseline_end}, "
        f"require_common_baseline_full_coverage={require_common_baseline_full_coverage}, "
        f"share_robustness_windows={share_robustness_windows}"
    )
    print(
        f"        sample_year_min={sample_year_min}, sample_year_max={sample_year_max}, "
        f"min_active_shock_schools={min_active_shock_schools}, "
        f"require_balanced_panel={require_balanced_panel}, "
        f"falsification_lead_years={falsification_lead_years}, "
        f"conditioning_baseline_window={conditioning_baseline_window_start}-{conditioning_baseline_window_end}"
    )
    print(f"        skip_panel_rebuild={skip_panel_rebuild}")
    if testing_cfg.get("enabled", False):
        print(f"[testing] ENABLED — sample_n_companies={testing_cfg.get('sample_n_companies', 100)}")

    # ── 1. Load inputs ─────────────────────────────────────────────────────────
    print("\n── 1. Loading inputs ──")
    t0 = time.time()
    global DUCKDB_CONN, ANALYSIS_PANEL_DF
    con = ddb.connect()
    DUCKDB_CONN = con
    _load_inputs(con, paths_cfg, pipeline_cfg, testing_cfg)
    print(f"   done in {time.time()-t0:.1f}s")

    # ── 2. Panel construction ──────────────────────────────────────────────────

    if skip_panel_rebuild:
        print("\n── 2. Loading saved panels (skip rebuild) ──")
        t0 = time.time()
        _load_saved_panel_views(
            con, paths_cfg, school_sample_mode,
            growth_window_start=school_sample_window_start,
            growth_window_end=school_sample_window_end,
        )
        _apply_analysis_sample_restrictions(
            con,
            sample_year_min=sample_year_min,
            sample_year_max=sample_year_max,
            min_active_shock_schools=min_active_shock_schools,
            require_balanced_panel=require_balanced_panel,
        )
        print(f"   done in {time.time()-t0:.1f}s")
    else:
        # 2b. Primary school metric panel and matched transparency sample.
        metric_panel = _empty_school_metric_panel()
        sample_summary = _empty_school_shift_sample()
        opt_share_panel = _empty_school_metric_panel()

        print("\n── 2b. Building school metric panel ──")
        t0 = time.time()
        metric_panel = _build_school_metric_panel(
            con, metric=school_shift_metric, degree_scope=degree_scope_for_matched,
            exclude_unitids=exclude_unitids,
            opt_ihmp_ipeds_share_intl_threshold=float(pipeline_cfg.get("opt_ihmp_ipeds_share_intl_threshold", 0.30)),
            opt_ihmp_foia_opt_share_threshold=float(pipeline_cfg.get("opt_ihmp_foia_opt_share_threshold", 0.50)),
            opt_ihmp_min_program_f1_count=int(pipeline_cfg.get("opt_ihmp_min_program_f1_count", 10)),
        )
        print(f"   metric panel: {len(metric_panel):,} school-year rows, {metric_panel['k'].nunique():,} schools")
        opt_share_panel = _build_school_metric_panel(
            con,
            metric="opt_share",
            degree_scope=opt_shifts_degree_scope,
            exclude_unitids=exclude_unitids,
        )
        print(f"   opt-share panel: {len(opt_share_panel):,} school-year rows")
        print(f"   done in {time.time()-t0:.1f}s")

        print("\n── 2b. Building school shift sample ──")
        t0 = time.time()
        school_names = _school_name_lookup(con)
        school_classification = pd.DataFrame(columns=["k", "carnegie_classification"])
        if match_school_pairs_by_carnegie_classification:
            school_classification = _school_classification_lookup(con)
            if school_classification.empty:
                print("[warn] Requested Carnegie-aware matching but no matching school classification field was found.")
        sample_summary = _build_school_shift_sample(
            metric_panel, metric=school_shift_metric,
            school_name_lookup=school_names,
            school_classification=school_classification,
            opt_share_panel=opt_share_panel,
            n_shifted=int(pipeline_cfg.get("school_sample_n_shifted", 25)),
            window_start=school_sample_window_start,
            window_end=school_sample_window_end,
            event_pre_years=event_shock_pre_years,
            event_post_years=event_shock_post_years,
            control_positive_cap=float(pipeline_cfg.get("school_sample_control_positive_cap", 0.02)),
            min_school_size=int(pipeline_cfg.get("school_sample_min_size", 400)),
            opt_share_min_school_f1_count=int(pipeline_cfg.get("opt_share_min_school_f1_count", 50)),
            opt_share_max_yoy_drop=float(pipeline_cfg.get("opt_share_max_yoy_drop", 0.50)),
            match_on_carnegie_classification=match_school_pairs_by_carnegie_classification,
            restrict_treated_to_no_large_enrollment_jump=restrict_treated_to_no_large_enrollment_jump,
            max_yoy_size_jump=school_sample_max_yoy_size_jump,
        )
        n_sel = int(sample_summary.get("selected_for_instrument", pd.Series(dtype="int64")).sum())
        n_treated = (sample_summary["sample_role"] == "treated").sum()
        n_control = (sample_summary["sample_role"] == "control").sum()
        print(f"   selected: {n_sel} schools ({n_treated} treated, {n_control} control)")

        years_window = list(range(
            school_sample_window_start,
            school_sample_window_end + 1,
        ))
        preview_cols = ["matched_pair_id", "sample_role", "school_name", "k", "treated_event_year", "treated_score"] + [
            f"metric_share_{y}" for y in years_window if f"metric_share_{y}" in sample_summary.columns
        ]
        preview = sample_summary.loc[sample_summary["selected_for_instrument"] == 1, preview_cols].copy()
        if "matched_pair_id" in preview.columns:
            preview = preview.sort_values(["matched_pair_id", "sample_role"], na_position="last")
        print("\n[matched-school sample]")
        print(preview.to_string(index=False))
        print()
        if confirm_sample:
            resp = input("Continue with this matched-school sample? [y/N]: ")
            if resp.strip().lower() not in {"y", "yes"}:
                print("Exiting.")
                return

        _register_school_metric_views(con, metric_panel=metric_panel, sample_summary=sample_summary)
        print(f"   done in {time.time()-t0:.1f}s")

        # 2a. Growth view.
        print("\n── 2a. Building growth view (g_kt) ──")
        t0 = time.time()
        if shock_design == "event_quantity":
            _build_event_quantity_growth_view(
                con,
                metric_panel=metric_panel,
                sample_summary=sample_summary,
                opt_share_panel=opt_share_panel,
                year_min=2008,
                year_max=2022,
                event_window_start=school_sample_window_start,
                event_window_end=school_sample_window_end,
                event_pre_years=event_shock_pre_years,
                event_post_years=event_shock_post_years,
                falsification_lead_years=falsification_lead_years,
                common_baseline_start=common_baseline_start,
                common_baseline_end=common_baseline_end,
                require_common_baseline_full_coverage=require_common_baseline_full_coverage,
            )
        elif school_sample_mode == "matched_shift_sample":
            _build_growth_view_matched_sample(
                con,
                growth_window_start=school_sample_window_start,
                growth_window_end=school_sample_window_end,
            )
        elif opt_shifts:
            _build_opt_shift_growth_view(
                con, degree_scope=opt_shifts_degree_scope, normalization=opt_shifts_normalization,
                use_changes=use_changes, demean_by_school=use_log_y, exclude_unitids=exclude_unitids,
                growth_window_start=school_sample_window_start, growth_window_end=school_sample_window_end,
            )
        else:
            _build_ipeds_growth_view(
                con, growth_population=growth_population, use_changes=use_changes,
                demean_by_school=use_log_y, exclude_unitids=exclude_unitids,
                growth_window_start=school_sample_window_start,
                growth_window_end=school_sample_window_end,
            )
        n_growth = con.sql("SELECT COUNT(DISTINCT k) FROM ipeds_unit_growth").fetchone()[0]
        yr_growth = con.sql("SELECT MIN(t), MAX(t) FROM ipeds_unit_growth").fetchone()
        print(f"   {n_growth:,} schools | years {yr_growth[0]}-{yr_growth[1]}")
        print(f"   done in {time.time()-t0:.1f}s")
    
        # 2c. Transition shares.
        print("\n── 2c. Building transition shares (share_ck) ──")
        t0 = time.time()
        _build_transition_shares(
            con,
            share_period=share_period,
            share_base_year=share_base_year,
            share_year_min=share_year_min,
            share_year_max=share_year_max,
            robustness_windows=share_robustness_windows,
            exclude_unitids=exclude_unitids,
            min_universities_for_share=share_min_universities_for_share,
        )
        print(f"   done in {time.time()-t0:.1f}s")
    
        # 2d. Instrument.
        print("\n── 2d. Building instrument (z_ct) ──")
        t0 = time.time()
        _build_instrument(con)
        print(f"   done in {time.time()-t0:.1f}s")
    
        # 2e. Treatment.
        print("\n── 2e. Building treatment (OPT hires) ──")
        t0 = time.time()
        _build_treatment(con, include_non_masters=include_non_masters, include_bachelors=include_bachelors)
        print(f"   done in {time.time()-t0:.1f}s")
    
        # 2f. Outcomes.
        print("\n── 2f. Building outcomes ──")
        t0 = time.time()
        _build_outcomes(con, lag_start=lag_start, lag_end=lag_end, use_changes=use_changes)
        print(f"   done in {time.time()-t0:.1f}s")
    
        # 2g. Analysis panel.
        print("\n── 2g. Assembling analysis panel ──")
        t0 = time.time()
        _build_analysis_panel(
            con,
            lag_start=lag_start,
            lag_end=lag_end,
            use_log_y=use_log_y,
            panel_year_min=sample_year_min,
            panel_year_max=sample_year_max,
            conditioning_baseline_window_start=conditioning_baseline_window_start,
            conditioning_baseline_window_end=conditioning_baseline_window_end,
        )
        _apply_analysis_sample_restrictions(
            con,
            sample_year_min=sample_year_min,
            sample_year_max=sample_year_max,
            min_active_shock_schools=min_active_shock_schools,
            require_balanced_panel=require_balanced_panel,
        )
        print(f"   done in {time.time()-t0:.1f}s")
    
    # ── 3. Save outputs ────────────────────────────────────────────────────────
    if save_panel:
        print("\n── 3. Saving outputs ──")
        t0 = time.time()
        _write_view(con, "analysis_panel", Path(paths_cfg["analysis_panel"]))
        _write_view(con, "instrument_components", Path(paths_cfg["instrument_components"]))
        _write_view(con, "instrument_panel", Path(paths_cfg["instrument_panel"]))
        _write_view(con, "transition_shares", Path(paths_cfg["transition_shares"]))
        _write_view(con, "ipeds_unit_growth", Path(paths_cfg["ipeds_unit_growth"]))
        _write_view(con, "school_shift_metric_panel", Path(paths_cfg["school_shift_metric_panel"]))
        _write_view(con, "school_shift_sample", Path(paths_cfg["school_shift_sample"]))
        print(f"   done in {time.time()-t0:.1f}s")

    # Load analysis panel to DataFrame for diagnostics and regressions.
    print("\n── Loading analysis panel into memory ──")
    analysis_start_t = reg_cfg.get("start_t")
    analysis_end_t = reg_cfg.get("end_t")
    if analysis_start_t is None:
        analysis_start_t = 2008
    if analysis_end_t is None:
        analysis_end_t = 2022
    try:
        analysis_start_t = int(analysis_start_t)
    except (TypeError, ValueError):
        analysis_start_t = 2008
    try:
        analysis_end_t = int(analysis_end_t)
    except (TypeError, ValueError):
        analysis_end_t = 2022
    panel_df = con.sql(
        f"SELECT * FROM analysis_panel WHERE CAST(t AS INTEGER) BETWEEN {analysis_start_t} AND {analysis_end_t}"
    ).df()
    ANALYSIS_PANEL_DF = panel_df
    print(f"   {len(panel_df):,} rows, {panel_df['c'].nunique():,} companies")

    # ── 4. Diagnostics ────────────────────────────────────────────────────────
    if plot_diagnostics:
        print("\n── 4. Running diagnostics ──")
        t0 = time.time()
        run_diagnostics(con, panel_df, pipeline_cfg, diag_dir)
        print(f"   done in {time.time()-t0:.1f}s")

    # ── 5. Regressions ────────────────────────────────────────────────────────
    print("\n── 5. Running regressions ──")
    t0 = time.time()
    run_regression_variants(panel_df, pipeline_cfg, reg_cfg, out_dir)
    run_falsification_tests(con, panel_df, pipeline_cfg, reg_cfg, out_dir)
    print(f"   done in {time.time()-t0:.1f}s")

    print(f"\n{'='*64}")
    print(f"TOTAL TIME: {time.time()-t_total:.1f}s")
    print(f"{'='*64}")

if __name__ == "__main__":
    main()
