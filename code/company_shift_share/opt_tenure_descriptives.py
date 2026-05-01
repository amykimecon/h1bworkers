"""Descriptives for firm-level OPT hiring and tenure relationships."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import duckdb as ddb
import numpy as np
import pandas as pd

_REPO_SRC = str(Path(__file__).resolve().parents[1])
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from company_shift_share.config_loader import get_cfg_section, load_config
from company_shift_share.opt_tenure_binsreg import run_tenure_opt_binsreg
from company_shift_share.opt_tenure_binsreg import (
    TENURE_SCOPE_ALL,
    TENURE_SCOPE_CHOICES,
    TENURE_SCOPE_NEW_HIRES,
    TENURE_SCOPE_RECENT_GRADS,
)
from company_shift_share.source_exposure_data import (
    load_or_build_source_opt_counts,
    load_or_build_wrds_company_year_workforce_cache,
)


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "company_shift_share_exposure_event_study.yaml"
)

FEATURE_TENURE_PRE_COL = "avg_tenure_years_annual_pre_level"
FEATURE_TENURE_ANNUAL_COL = "avg_tenure_years_annual"
HIRE_TENURE_ANNUAL_COL = "avg_tenure_new_hires_annual"
NEW_POS_TENURE_ANNUAL_COL = "avg_tenure_new_positions_annual"
RECENT_GRADS_TENURE_ANNUAL_COL = "avg_tenure_recent_grads_annual"

HIRE_TENURE_PRE_COL = f"{HIRE_TENURE_ANNUAL_COL}_pre_level"
NEW_POS_TENURE_PRE_COL = f"{NEW_POS_TENURE_ANNUAL_COL}_pre_level"
RECENT_GRADS_TENURE_PRE_COL = f"{RECENT_GRADS_TENURE_ANNUAL_COL}_pre_level"

ANALYSIS_PANEL_FILE = "opt_tenure_analysis_panel.parquet"
ANALYSIS_PANEL_SCHEMA_VERSION = 1
ANALYSIS_PANEL_META_SUFFIX = ".meta.json"

DEFAULT_OPT_COLS = (
    "any_opt_hire_rate_annual_pre_level",
    "any_opt_hire_count_annual_pre_level",
    "masters_opt_hire_rate_annual_pre_level",
)
SOURCE_TENURE_MODE_RAW = "raw_wrds"
WRDS_SOURCE_OPT_COUNT_COL = "any_opt_hires_correction_aware"
WRDS_MASTERS_OPT_COUNT_COL = "masters_opt_hires_correction_aware"
WRDS_NEW_HIRES_COUNT_COL = "n_new_hires_wrds_annual"

SCRIPT_CFG_SECTION = "opt_tenure_descriptives"

EXPOSURE_EVENT_EXCLUDE = {
    "c",
    "in_analysis_universe",
    "preferred_rcid_source",
    "outside_negative_candidate",
    "post2016_any_opt",
    "target_source",
    "train_sample",
    "event_study_sample",
    "leaveout_training_firm",
    "predicted_prob",
    "predicted_class",
    "predicted_index",
    "exposure_value",
    "model_method",
    "index_entry_mode",
}


def _escape_sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    work = pd.DataFrame({"left": _safe_numeric(left), "right": _safe_numeric(right)}).dropna()
    if len(work) < 3:
        return None
    if work["left"].nunique() < 2 or work["right"].nunique() < 2:
        return None
    return float(work["left"].corr(work["right"]))


def _safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    work = pd.DataFrame({"left": _safe_numeric(left), "right": _safe_numeric(right)}).dropna()
    if len(work) < 3:
        return None
    if work["left"].nunique() < 2 or work["right"].nunique() < 2:
        return None
    return float(work["left"].corr(work["right"], method="spearman"))


def _analysis_panel_meta_path(panel_path: Path) -> Path:
    return panel_path.with_suffix(ANALYSIS_PANEL_META_SUFFIX)


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] Failed to read analysis panel metadata from {path}: {type(exc).__name__}")
        return None


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _file_mtime(path: Path | str | None) -> int | None:
    if path is None or not path:
        return None
    try:
        return int(Path(path).stat().st_mtime)
    except OSError:
        return None


def _safe_slope(left: pd.Series, right: pd.Series) -> tuple[float | None, float | None]:
    work = pd.DataFrame({"left": _safe_numeric(left), "right": _safe_numeric(right)}).dropna()
    if len(work) < 2:
        return None, None
    x = work["left"].to_numpy(dtype=float)
    y = work["right"].to_numpy(dtype=float)
    if np.std(x) == 0.0:
        return None, None
    slope, intercept = np.polyfit(x, y, 1).tolist()
    return float(slope), float(intercept)


def _annual_to_pre_level(
    annual_df: pd.DataFrame,
    value_col: str,
    *,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    if annual_df.empty or value_col not in annual_df.columns:
        return pd.DataFrame(
            columns=[
                "c",
                f"{value_col}_pre_level",
                f"{value_col}_pre_n_years",
            ],
        )

    work = annual_df.copy()
    work["c"] = _safe_numeric(work["c"]).astype("Int64")
    work["t"] = _safe_numeric(work["t"])
    work[value_col] = _safe_numeric(work[value_col])
    window = work.loc[work["t"].between(int(year_min), int(year_max)), ["c", value_col]].dropna(
        subset=["c", value_col]
    )
    if window.empty:
        return pd.DataFrame(
            columns=[
                "c",
                f"{value_col}_pre_level",
                f"{value_col}_pre_n_years",
            ],
        )

    return (
        window.groupby("c", as_index=False)
        .agg(
            **{
                f"{value_col}_pre_level": (value_col, "mean"),
                f"{value_col}_pre_n_years": (value_col, "count"),
            }
        )
        .astype({"c": "Int64"})
    )


def _build_position_tenure_metrics(
    positions_path: Path,
    *,
    year_min: int,
    year_max: int,
    users_path: Path | None = None,
    grad_lag_days: int = 365,
) -> pd.DataFrame:
    con = ddb.connect()
    try:
        positions_path = Path(positions_path)
        sql = f"""
        WITH pos AS (
            SELECT
                CAST(rcid AS BIGINT) AS c,
                CAST(user_id AS BIGINT) AS user_id,
                CAST(position_id AS BIGINT) AS position_id,
                CAST(startdate AS DATE) AS startdate,
                COALESCE(CAST(enddate AS DATE), DATE '2026-12-31') AS enddate
            FROM read_parquet('{_escape_sql_path(positions_path)}')
            WHERE c IS NOT NULL
              AND user_id IS NOT NULL
              AND startdate IS NOT NULL
        ),
        first_start_with_end AS (
            SELECT
                c,
                user_id,
                MIN(startdate) AS first_start,
                MAX(enddate) AS last_end
            FROM pos
            GROUP BY 1, 2
        ),
        new_hires AS (
            SELECT
                c,
                EXTRACT(YEAR FROM first_start)::INTEGER AS t,
                COUNT(DISTINCT user_id)::BIGINT AS n_new_hires_year,
                AVG(
                    GREATEST(
                        0.0,
                        (last_end - first_start)::DOUBLE / 365.25
                    )
                ) AS avg_tenure_new_hires_annual
            FROM first_start_with_end
            WHERE EXTRACT(YEAR FROM first_start) BETWEEN {int(year_min)} AND {int(year_max)}
            GROUP BY 1, 2
        ),
        new_positions AS (
            SELECT
                c,
                EXTRACT(YEAR FROM startdate)::INTEGER AS t,
                COUNT(*)::BIGINT AS n_new_positions_year,
                AVG(
                    GREATEST(
                        0.0,
                        (LEAST(enddate, MAKE_DATE(EXTRACT(YEAR FROM startdate)::INTEGER, 12, 31)) - startdate)::DOUBLE / 365.25
                    )
                ) AS avg_tenure_new_positions_annual
            FROM pos
            WHERE EXTRACT(YEAR FROM startdate) BETWEEN {int(year_min)} AND {int(year_max)}
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(nh.c, np.c) AS c,
            COALESCE(nh.t, np.t) AS t,
            nh.n_new_hires_year,
            nh.avg_tenure_new_hires_annual,
            np.n_new_positions_year,
            np.avg_tenure_new_positions_annual
        FROM new_hires nh
        FULL OUTER JOIN new_positions np
          ON nh.c = np.c
         AND nh.t = np.t
        """
        metrics = con.sql(sql).df()
        grad_metrics = pd.DataFrame(
            columns=[
                "c",
                "t",
                "n_recent_grads_year",
                RECENT_GRADS_TENURE_ANNUAL_COL,
            ],
        )
        if users_path is not None and Path(users_path).exists():
            users_path = Path(users_path)
            grad_sql = f"""
            WITH users AS (
                SELECT
                    CAST(user_id AS BIGINT) AS user_id,
                    MAX(TRY_CAST(ed_enddate AS DATE)) AS grad_date
                FROM read_parquet('{_escape_sql_path(users_path)}')
                WHERE user_id IS NOT NULL
                  AND ed_enddate IS NOT NULL
                GROUP BY 1
            ),
            pos AS (
                SELECT
                    CAST(rcid AS BIGINT) AS c,
                    CAST(user_id AS BIGINT) AS user_id,
                    CAST(startdate AS DATE) AS startdate
                FROM read_parquet('{_escape_sql_path(positions_path)}')
                WHERE c IS NOT NULL
                  AND user_id IS NOT NULL
                  AND startdate IS NOT NULL
            ),
            first_grad_hire AS (
                SELECT
                    p.user_id,
                    p.c,
                    p.startdate,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.user_id
                        ORDER BY p.startdate ASC
                    ) AS hire_order
                FROM pos p
                JOIN users u
                  ON u.user_id = p.user_id
                WHERE p.startdate >= u.grad_date
                  AND p.startdate <= u.grad_date + INTERVAL '{int(grad_lag_days)} days'
                  AND EXTRACT(YEAR FROM p.startdate)::INT BETWEEN {int(year_min)} AND {int(year_max)}
            )
            SELECT
                c,
                EXTRACT(YEAR FROM startdate)::INTEGER AS t,
                COUNT(*)::BIGINT AS n_recent_grads_year,
                AVG(
                    GREATEST(
                        0.0,
                        (MAKE_DATE(EXTRACT(YEAR FROM startdate)::INTEGER, 12, 31) - startdate)::DOUBLE / 365.25
                    )
                ) AS avg_tenure_recent_grads_annual
            FROM first_grad_hire
            WHERE hire_order = 1
            GROUP BY 1, 2
            """
            grad_metrics = con.sql(grad_sql).df()
    finally:
        con.close()

    metrics = (
        metrics
        .merge(grad_metrics, on=["c", "t"], how="outer")
        if not metrics.empty
        else grad_metrics
    )

    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "c",
                "t",
                "n_new_hires_year",
                HIRE_TENURE_ANNUAL_COL,
                "n_new_positions_year",
                NEW_POS_TENURE_ANNUAL_COL,
                "n_recent_grads_year",
                RECENT_GRADS_TENURE_ANNUAL_COL,
            ],
        )

    metrics["c"] = _safe_numeric(metrics["c"]).astype("Int64")
    metrics["t"] = _safe_numeric(metrics["t"]).astype("Int64")
    for col in [
        "n_new_hires_year",
        HIRE_TENURE_ANNUAL_COL,
        "n_new_positions_year",
        NEW_POS_TENURE_ANNUAL_COL,
        "n_recent_grads_year",
        RECENT_GRADS_TENURE_ANNUAL_COL,
    ]:
        metrics[col] = _safe_numeric(metrics[col])
    return metrics


def _build_binned_summary(
    frame: pd.DataFrame,
    tenure_col: str,
    opt_col: str,
    *,
    n_bins: int = 5,
) -> pd.DataFrame:
    work = frame.loc[:, ["c", tenure_col, opt_col]].copy()
    work[tenure_col] = _safe_numeric(work[tenure_col])
    work[opt_col] = _safe_numeric(work[opt_col])
    work = work.dropna()
    if work.empty:
        return pd.DataFrame()

    unique = int(work[tenure_col].nunique())
    bins = max(2, min(int(n_bins), unique))
    if bins < 2:
        return pd.DataFrame()

    work["tenure_bin"] = (
        pd.qcut(work[tenure_col].rank(method="first"), q=bins, labels=False, duplicates="drop") + 1
    )
    if work["tenure_bin"].isna().all():
        return pd.DataFrame()

    out = (
        work.groupby("tenure_bin", as_index=False)
        .agg(
            n_obs=(opt_col, "size"),
            mean_tenure=(tenure_col, "mean"),
            median_tenure=(tenure_col, "median"),
            mean_opt=(opt_col, "mean"),
            median_opt=(opt_col, "median"),
        )
        .sort_values("tenure_bin")
    )
    out["tenure_metric"] = tenure_col
    out["opt_metric"] = opt_col
    return out


def _correlation_rows(frame: pd.DataFrame, tenure_cols: list[str], opt_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for opt_col in opt_cols:
        for tenure_col in tenure_cols:
            if tenure_col not in frame.columns:
                continue

            work = frame[[tenure_col, opt_col]].copy()
            work[tenure_col] = _safe_numeric(work[tenure_col])
            work[opt_col] = _safe_numeric(work[opt_col])
            work = work.dropna()
            if work.empty:
                rows.append(
                    {
                        "opt_metric": opt_col,
                        "tenure_metric": tenure_col,
                        "n": 0,
                        "pearson_corr": None,
                        "spearman_corr": None,
                        "slope": None,
                        "intercept": None,
                        "mean_opt": None,
                        "mean_tenure": None,
                    }
                )
                continue

            slope, intercept = _safe_slope(work[tenure_col], work[opt_col])
            rows.append(
                {
                    "opt_metric": opt_col,
                    "tenure_metric": tenure_col,
                    "n": int(len(work)),
                    "pearson_corr": _safe_corr(work[tenure_col], work[opt_col]),
                    "spearman_corr": _safe_spearman(work[tenure_col], work[opt_col]),
                    "slope": slope,
                    "intercept": intercept,
                    "mean_opt": float(work[opt_col].mean()),
                    "mean_tenure": float(work[tenure_col].mean()),
                }
            )
    return rows


def _resolve_positions_path(
    cfg: dict[str, Any],
    workforce_meta: dict[str, Any] | None,
) -> Path | None:
    paths_cfg = get_cfg_section(cfg, "paths")
    configured = paths_cfg.get("wrds_workforce_selected_us_positions_out")
    if isinstance(configured, str) and configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return configured_path

    if workforce_meta:
        meta_path = workforce_meta.get("local_selected_us_positions_path")
        if isinstance(meta_path, str) and meta_path:
            path = Path(meta_path)
            if path.exists():
                return path

    fallback = Path(paths_cfg.get("company_shift_share_out_dir", ".")) / "wrds_workforce_selected_us_positions.parquet"
    if fallback.exists():
        return fallback
    return None


def _resolve_users_path(
    cfg: dict[str, Any],
    workforce_meta: dict[str, Any] | None,
) -> Path | None:
    paths_cfg = get_cfg_section(cfg, "paths")
    configured = paths_cfg.get("wrds_workforce_users_out")
    if isinstance(configured, str) and configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return configured_path

    if workforce_meta:
        meta_path = workforce_meta.get("local_wrds_users_path")
        if isinstance(meta_path, str) and meta_path:
            path = Path(meta_path)
            if path.exists():
                return path

    fallback = Path(paths_cfg.get("company_shift_share_out_dir", ".")) / "wrds_workforce_users.parquet"
    if fallback.exists():
        return fallback
    return None


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _resolve_analysis_panel_path(
    out_dir: Path,
    panel_path: Path | None,
) -> Path:
    if panel_path is not None:
        return panel_path
    return out_dir / ANALYSIS_PANEL_FILE


def _resolve_source_cache_paths(cfg: dict[str, Any]) -> tuple[Path | None, Path | None]:
    paths_cfg = get_cfg_section(cfg, "paths")
    workforce = paths_cfg.get("wrds_company_year_workforce_out")
    opt_counts = paths_cfg.get("source_opt_counts_out")
    workforce_path = Path(workforce) if isinstance(workforce, str) and workforce else None
    opt_counts_path = Path(opt_counts) if isinstance(opt_counts, str) and opt_counts else None
    return workforce_path, opt_counts_path


def _build_opt_pre_columns_from_raw(
    workforce_df: pd.DataFrame,
    opt_counts_df: pd.DataFrame,
    *,
    year_min: int,
    year_max: int,
    n_hires_col: str = WRDS_NEW_HIRES_COUNT_COL,
    any_opt_count_col: str = WRDS_SOURCE_OPT_COUNT_COL,
    masters_opt_count_col: str = WRDS_MASTERS_OPT_COUNT_COL,
) -> pd.DataFrame:
    frame = pd.DataFrame(columns=["c", "t"])
    if workforce_df is not None:
        frame = workforce_df.copy()
        if "t" in frame.columns and "c" in frame.columns:
            frame["t"] = _safe_numeric(frame["t"])
            frame["c"] = _safe_numeric(frame["c"]).astype("Int64")
            keep_cols = ["c", "t"]
            if n_hires_col in frame.columns:
                keep_cols.append(n_hires_col)
            frame = frame.loc[frame["t"].between(int(year_min), int(year_max)), keep_cols].copy()

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "c",
                "any_opt_hire_count_annual_pre_level",
                "any_opt_hire_count_annual_pre_n_years",
                "any_opt_hire_rate_annual_pre_level",
                "any_opt_hire_rate_annual_pre_n_years",
                "masters_opt_hire_count_annual_pre_level",
                "masters_opt_hire_count_annual_pre_n_years",
                "masters_opt_hire_rate_annual_pre_level",
                "masters_opt_hire_rate_annual_pre_n_years",
            ],
        )

    if n_hires_col not in frame.columns:
        frame[n_hires_col] = 0.0
    frame[n_hires_col] = _safe_numeric(frame[n_hires_col])
    merged = frame[["c", "t", n_hires_col]].copy()

    if opt_counts_df is not None and not opt_counts_df.empty:
        opt_panel = opt_counts_df.copy()
        opt_panel["t"] = _safe_numeric(opt_panel["t"])
        opt_panel["c"] = _safe_numeric(opt_panel["c"]).astype("Int64")
        keep_cols = ["c", "t"]
        if any_opt_count_col in opt_panel.columns:
            keep_cols.append(any_opt_count_col)
        if masters_opt_count_col in opt_panel.columns:
            keep_cols.append(masters_opt_count_col)
        opt_panel = opt_panel.loc[:, keep_cols]
        merged = merged.merge(opt_panel, on=["c", "t"], how="left")
    else:
        if any_opt_count_col in merged.columns:
            merged = merged.drop(columns=[any_opt_count_col])
        if masters_opt_count_col in merged.columns:
            merged = merged.drop(columns=[masters_opt_count_col])

    for col in (n_hires_col, any_opt_count_col, masters_opt_count_col):
        if col in merged.columns:
            merged[col] = _safe_numeric(merged[col]).fillna(0.0)

    work_parts = {}
    count_cols = {
        "any_opt_hire_count_annual": any_opt_count_col,
        "masters_opt_hire_count_annual": masters_opt_count_col,
    }
    for out_prefix, src_col in count_cols.items():
        if src_col in merged.columns:
            pre = _annual_to_pre_level(
                merged[["c", "t", src_col]].copy(),
                src_col,
                year_min=year_min,
                year_max=year_max,
            )
            if not pre.empty:
                pre = pre.rename(
                    columns={
                        f"{src_col}_pre_level": f"{out_prefix}_pre_level",
                        f"{src_col}_pre_n_years": f"{out_prefix}_pre_n_years",
                    },
                )
                work_parts[out_prefix] = pre

    if any_opt_count_col in merged.columns and n_hires_col in merged.columns:
        merged["any_opt_hire_rate_pre_rate"] = np.where(
            merged[n_hires_col] > 0,
            merged[any_opt_count_col] / merged[n_hires_col],
            np.nan,
        )
        pre = _annual_to_pre_level(
            merged[["c", "t", "any_opt_hire_rate_pre_rate"]].copy(),
            "any_opt_hire_rate_pre_rate",
            year_min=year_min,
            year_max=year_max,
        )
        if not pre.empty:
            work_parts["any_opt_rate"] = pre.rename(
                columns={
                    "any_opt_hire_rate_pre_rate_pre_level": "any_opt_hire_rate_annual_pre_level",
                    "any_opt_hire_rate_pre_rate_pre_n_years": "any_opt_hire_rate_annual_pre_n_years",
                },
            )

    if masters_opt_count_col in merged.columns and n_hires_col in merged.columns:
        merged["masters_opt_hire_rate_pre_rate"] = np.where(
            merged[n_hires_col] > 0,
            merged[masters_opt_count_col] / merged[n_hires_col],
            np.nan,
        )
        pre = _annual_to_pre_level(
            merged[["c", "t", "masters_opt_hire_rate_pre_rate"]].copy(),
            "masters_opt_hire_rate_pre_rate",
            year_min=year_min,
            year_max=year_max,
        )
        if not pre.empty:
            work_parts["masters_opt_rate"] = pre.rename(
                columns={
                    "masters_opt_hire_rate_pre_rate_pre_level": "masters_opt_hire_rate_annual_pre_level",
                    "masters_opt_hire_rate_pre_rate_pre_n_years": "masters_opt_hire_rate_annual_pre_n_years",
                },
            )

    if not work_parts:
        return pd.DataFrame(
            columns=[
                "c",
                "any_opt_hire_count_annual_pre_level",
                "any_opt_hire_count_annual_pre_n_years",
                "any_opt_hire_rate_annual_pre_level",
                "any_opt_hire_rate_annual_pre_n_years",
                "masters_opt_hire_count_annual_pre_level",
                "masters_opt_hire_count_annual_pre_n_years",
                "masters_opt_hire_rate_annual_pre_level",
                "masters_opt_hire_rate_annual_pre_n_years",
            ],
        )

    out = merged[["c"]].drop_duplicates().reset_index(drop=True).copy()
    out["c"] = out["c"].astype("Int64")
    for part in work_parts.values():
        out = out.merge(part, on="c", how="left")
    return out


def _analysis_panel_metadata_expected(
    *,
    feature_year_min: int,
    feature_year_max: int,
    source_mode: str,
    workforce_path: Path | None,
    opt_counts_path: Path | None,
    positions_path: Path | None,
    source_force_rebuild: bool,
    skip_position_metrics: bool,
) -> dict[str, Any]:
    metadata = {
        "schema_version": ANALYSIS_PANEL_SCHEMA_VERSION,
        "feature_year_min": int(feature_year_min),
        "feature_year_max": int(feature_year_max),
        "source_mode": source_mode,
        "source_force_rebuild": bool(source_force_rebuild),
        "workforce_path": str(workforce_path) if workforce_path is not None else None,
        "workforce_mtime": _file_mtime(workforce_path),
        "opt_counts_path": str(opt_counts_path) if opt_counts_path is not None else None,
        "opt_counts_mtime": _file_mtime(opt_counts_path),
        "positions_path": str(positions_path) if positions_path else None,
        "positions_path_mtime": _file_mtime(positions_path),
        "position_metrics_required": not bool(skip_position_metrics),
        "required_columns": [  # sanity-check list
            "c",
            "any_opt_hire_rate_annual_pre_level",
            "any_opt_hire_count_annual_pre_level",
        ],
    }
    return metadata


def _analysis_panel_is_compatible(
    panel: pd.DataFrame,
    meta: dict[str, Any] | None,
    expected: dict[str, Any],
) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("schema_version") != ANALYSIS_PANEL_SCHEMA_VERSION:
        return False
    for key in ("feature_year_min", "feature_year_max", "position_metrics_required", "source_mode"):
        if meta.get(key) != expected.get(key):
            return False

    source_mode = expected.get("source_mode")
    if source_mode == SOURCE_TENURE_MODE_RAW:
        expected_workforce_mtime = expected.get("workforce_mtime")
        if expected_workforce_mtime is not None and meta.get("workforce_mtime") is not None:
            if int(expected_workforce_mtime) != int(meta.get("workforce_mtime")):
                return False
        expected_workforce_path = expected.get("workforce_path")
        if expected_workforce_path is not None and meta.get("workforce_path") != expected_workforce_path:
            return False
        expected_opt_counts_mtime = expected.get("opt_counts_mtime")
        if expected_opt_counts_mtime is not None and meta.get("opt_counts_mtime") is not None:
            if int(expected_opt_counts_mtime) != int(meta.get("opt_counts_mtime")):
                return False
        expected_opt_counts_path = expected.get("opt_counts_path")
        if expected_opt_counts_path is not None and meta.get("opt_counts_path") != expected_opt_counts_path:
            return False
    else:
        return False

    expected_positions_mtime = expected.get("positions_path_mtime")
    if expected_positions_mtime is not None:
        if meta.get("positions_path") != expected.get("positions_path"):
            return False
        if meta.get("positions_path_mtime") != expected_positions_mtime:
            return False
    elif expected.get("positions_path") is not None:
        # required positions path changed or became unavailable.
        return False

    required_cols = expected.get("required_columns", [])
    if required_cols and not set(required_cols).issubset(set(panel.columns.tolist())):
        return False
    return True


def _save_analysis_panel(
    panel_path: Path,
    panel: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path = Path(panel_path)
    panel.to_parquet(panel_path, index=False)
    _safe_write_json(_analysis_panel_meta_path(panel_path), metadata | {"columns": list(panel.columns)})
    print(f"[done] Saved reusable analysis panel to {panel_path}")


def _resolve_tenure_scope_pre_col(scope: str) -> str:
    if scope == TENURE_SCOPE_NEW_HIRES:
        return HIRE_TENURE_PRE_COL
    if scope == TENURE_SCOPE_RECENT_GRADS:
        return RECENT_GRADS_TENURE_PRE_COL
    return FEATURE_TENURE_PRE_COL


def _is_interactive() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def _argv_has_option(argv: list[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    return None


def _coerce_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return Path(stripped) if stripped else None
    return None


def _apply_script_config_defaults(args: argparse.Namespace, cfg: dict[str, Any], argv: list[str]) -> argparse.Namespace:
    script_cfg = get_cfg_section(cfg, SCRIPT_CFG_SECTION)
    if not script_cfg:
        return args

    def _set_if_missing(flag: str, key: str, cast: callable | None = None) -> None:
        if _argv_has_option(argv, flag):
            return
        if key not in script_cfg:
            return
        raw = script_cfg.get(key)
        if raw is None:
            return
        if cast is None:
            setattr(args, key, raw)
            return
        value = cast(raw)
        if value is not None:
            setattr(args, key, value)

    _set_if_missing("--feature-year-min", "feature_year_min", int)
    _set_if_missing("--feature-year-max", "feature_year_max", int)
    _set_if_missing("--force-rebuild-company-features", "force_rebuild_company_features", _coerce_bool)
    _set_if_missing("--workforce-cache-only", "workforce_cache_only", _coerce_bool)
    _set_if_missing("--skip-position-metrics", "skip_position_metrics", _coerce_bool)
    _set_if_missing("--positions-path", "positions_path", _coerce_optional_path)
    _set_if_missing("--out-dir", "out_dir", _coerce_optional_path)
    _set_if_missing("--analysis-panel-path", "analysis_panel_path", _coerce_optional_path)
    _set_if_missing("--rebuild-analysis-panel", "rebuild_analysis_panel", _coerce_bool)
    _set_if_missing("--tenure-analysis-scope", "tenure_analysis_scope", str)
    _set_if_missing("--output-prefix", "output_prefix", str)
    _set_if_missing("--run-binsreg", "run_binsreg", _coerce_bool)
    _set_if_missing("--bins", "bins", int)
    _set_if_missing("--binsreg-tenure-col", "binsreg_tenure_col", str)
    _set_if_missing("--binsreg-tenure-scope", "binsreg_tenure_scope", str)
    _set_if_missing("--binsreg-opt-count-col", "binsreg_opt_count_col", str)
    _set_if_missing("--binsreg-n-hires-col", "binsreg_n_hires_col", str)
    _set_if_missing("--binsreg-firm-size-col", "binsreg_firm_size_col", str)
    _set_if_missing("--binsreg-metric", "binsreg_metric", str)
    _set_if_missing("--binsreg-data-min-t", "binsreg_data_min_t", int)
    _set_if_missing("--binsreg-data-max-t", "binsreg_data_max_t", int)
    _set_if_missing("--binsreg-force-rebuild", "binsreg_force_rebuild", _coerce_bool)
    _set_if_missing("--binsreg-by-year-bins", "binsreg_by_year_bins", _coerce_bool)
    _set_if_missing("--binsreg-log-opt-hires", "binsreg_log_opt_hires", _coerce_bool)
    _set_if_missing("--show-plots", "show_plots", _coerce_bool)
    return args


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate descriptive summaries linking firm OPT hiring and tenure.",
        allow_abbrev=False,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the exposure-event-study config (default: configs/company_shift_share_exposure_event_study.yaml).",
    )
    p.add_argument(
        "--feature-year-min",
        type=int,
        default=None,
        help="Feature year minimum for pre-period aggregates.",
    )
    p.add_argument(
        "--feature-year-max",
        type=int,
        default=None,
        help="Feature year maximum for pre-period aggregates.",
    )
    p.add_argument(
        "--force-rebuild-company-features",
        action="store_true",
        help="Recompute raw WRDS-derived caches before generating descriptives.",
    )
    p.add_argument(
        "--workforce-cache-only",
        action="store_true",
        default=None,
        help=(
            "Skip WRDS rebuild and reuse cached wrds_company_year_workforce.parquet only. "
            "Useful when you want to avoid any WRDS activity."
        ),
    )
    p.add_argument(
        "--skip-position-metrics",
        action="store_true",
        help="Skip tenure metrics derived from cached WRDS positions.",
    )
    p.add_argument(
        "--positions-path",
        type=Path,
        default=None,
        help="Optional override for wrds_workforce_selected_us_positions.parquet.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for CSV diagnostics output.",
    )
    p.add_argument(
        "--analysis-panel-path",
        type=Path,
        default=None,
        help=(
            "Optional path for reusable analysis panel. "
            "If not provided, defaults to <out_dir>/opt_tenure_analysis_panel.parquet."
        ),
    )
    p.add_argument(
        "--rebuild-analysis-panel",
        action="store_true",
        help="Recompute and overwrite the analysis panel even if a compatible cache exists.",
    )
    p.add_argument(
        "--tenure-analysis-scope",
        choices=TENURE_SCOPE_CHOICES,
        default=TENURE_SCOPE_ALL,
        help=(
            "Tenure metric to retain as the primary summary series in the final analysis panel: "
            "all (all employees), new_hires, or recent_grads."
        ),
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default="opt_tenure_descriptives",
        help="Filename prefix for output files.",
    )
    p.add_argument(
        "--run-binsreg",
        action="store_true",
        default=None,
        help="Also run FE binsreg outputs (tenure vs OPT hiring) within annual firm-year panel.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of bins for binsreg output.",
    )
    p.add_argument(
        "--binsreg-tenure-col",
        type=str,
        default="avg_tenure_years_annual",
        help="Tenure metric for binsreg (defaults to annual, not pre-level).",
    )
    p.add_argument(
        "--binsreg-tenure-scope",
        choices=TENURE_SCOPE_CHOICES,
        default=TENURE_SCOPE_ALL,
        help=(
            "Tenure scope for binsreg: all employees (default), new_hires, or recent_grads. "
            "If new_hires is not available, recent_grads is used for fallback."
        ),
    )
    p.add_argument(
        "--binsreg-opt-count-col",
        type=str,
        default="any_opt_hires_correction_aware",
        help="OPT hire count column from source_opt_counts for binsreg.",
    )
    p.add_argument(
        "--binsreg-n-hires-col",
        type=str,
        default="n_new_hires_wrds_annual",
        help="Denominator column for binsreg opt hiring rates.",
    )
    p.add_argument(
        "--binsreg-firm-size-col",
        type=str,
        default="total_headcount_wrds_annual",
        help="Firm-size column for binsreg firm-size quartiles.",
    )
    p.add_argument(
        "--binsreg-metric",
        choices=("rate", "count"),
        default="rate",
        help="Whether binsreg uses OPT hires count or hiring rate.",
    )
    p.add_argument(
        "--binsreg-data-min-t",
        type=int,
        default=None,
        help="Optional year min for binsreg window (defaults to exposure_event_study.data_min_t).",
    )
    p.add_argument(
        "--binsreg-data-max-t",
        type=int,
        default=None,
        help="Optional year max for binsreg window (defaults to exposure_event_study.data_max_t).",
    )
    p.add_argument(
        "--binsreg-force-rebuild",
        action="store_true",
        help="Rebuild annual workforce/OPT count cache for binsreg window (requires WRDS access).",
    )
    p.add_argument(
        "--binsreg-by-year-bins",
        action="store_true",
        default=None,
        help="Create additional binsreg curves separately by year.",
    )
    p.add_argument(
        "--binsreg-log-opt-hires",
        action="store_true",
        default=None,
        help="Use log(1 + OPT hires count) in binsreg when metric=count.",
    )
    p.add_argument(
        "--show-plots",
        action="store_true",
        default=None,
        help=(
            "Show binsreg figures in the current interactive output (in addition to saving). "
            "Defaults to true in IPython/Jupyter."
        ),
    )
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args, unknown = p.parse_known_args(args=raw_argv)
    if unknown:
        if _is_interactive():
            print(f"[warn] Ignoring extra CLI args from interactive launch: {unknown}")
        else:
            raise SystemExit(f"unrecognized arguments: {' '.join(unknown)}")
    _apply_script_config_defaults(args, load_config(args.config), raw_argv)
    if args.run_binsreg is None:
        args.run_binsreg = _is_interactive()
    if args.show_plots is None:
        args.show_plots = _is_interactive()
    if args.binsreg_by_year_bins is None:
        args.binsreg_by_year_bins = _is_interactive()
    if args.binsreg_log_opt_hires is None:
        args.binsreg_log_opt_hires = False
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv=argv)
    cfg = load_config(args.config)
    paths_cfg = get_cfg_section(cfg, "paths")
    feature_cfg = get_cfg_section(cfg, "revelio_company_features")
    exposure_cfg = get_cfg_section(cfg, "exposure_event_study")

    feature_year_min = int(
        args.feature_year_min
        if args.feature_year_min is not None
        else feature_cfg.get(
            "feature_year_min",
            exposure_cfg.get("data_min_t", 2010),
        )
    )
    feature_year_max = int(
        args.feature_year_max
        if args.feature_year_max is not None
        else feature_cfg.get(
            "feature_year_max",
            exposure_cfg.get("data_max_t", 2015),
        )
    )
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(paths_cfg.get("company_shift_share_out_dir", str(Path.cwd()))) / "descriptives"
    out_dir = Path(out_dir)

    analysis_panel_path = _resolve_analysis_panel_path(out_dir, args.analysis_panel_path)
    workforce_path, opt_counts_path = _resolve_source_cache_paths(cfg)
    requested_positions_path = (
        args.positions_path
        if args.positions_path is not None and args.positions_path.exists()
        else None
    )
    analysis_panel_meta = _analysis_panel_metadata_expected(
        feature_year_min=feature_year_min,
        feature_year_max=feature_year_max,
        source_mode=SOURCE_TENURE_MODE_RAW,
        workforce_path=workforce_path,
        opt_counts_path=opt_counts_path,
        positions_path=requested_positions_path,
        source_force_rebuild=args.force_rebuild_company_features,
        skip_position_metrics=args.skip_position_metrics,
    )

    analysis_df: pd.DataFrame | None = None
    workforce_meta: dict[str, Any] | None = None
    workforce_df: pd.DataFrame | None = None
    opt_counts_df: pd.DataFrame | None = None
    has_feature_tenure = False
    tenure_present_in_exposure = False
    used_position_metrics = False
    loaded_panel = False
    source_mode = SOURCE_TENURE_MODE_RAW
    use_position_metrics = not args.skip_position_metrics

    if not args.rebuild_analysis_panel and analysis_panel_path.exists():
        cached = pd.read_parquet(analysis_panel_path)
        cached_meta = _safe_read_json(_analysis_panel_meta_path(analysis_panel_path))
        if _analysis_panel_is_compatible(cached, cached_meta, analysis_panel_meta):
            analysis_df = cached
            loaded_panel = True
            has_feature_tenure = FEATURE_TENURE_PRE_COL in analysis_df.columns
            tenure_present_in_exposure = has_feature_tenure and FEATURE_TENURE_PRE_COL not in EXPOSURE_EVENT_EXCLUDE
            if cached_meta is not None:
                used_position_metrics = bool(cached_meta.get("position_metrics_included"))
                wf_positions_path = (
                    cached_meta.get("workforce_meta_positions_path")
                    or cached_meta.get("positions_path")
                    or cached_meta.get("positions_meta_path")
                )
                if wf_positions_path:
                    workforce_meta = {"local_selected_us_positions_path": str(wf_positions_path)}
            print(f"[reuse] Loaded analysis panel from {analysis_panel_path}")
        else:
            print("[warn] Existing analysis panel metadata is incompatible with requested settings; rebuilding panel.")

    if analysis_df is None:
        workforce_df, workforce_meta = load_or_build_wrds_company_year_workforce_cache(
            config_path=args.config,
            cfg=cfg,
            year_min=feature_year_min,
            year_max=feature_year_max,
            force_rebuild=args.force_rebuild_company_features,
            cache_only=bool(args.workforce_cache_only),
        )
        opt_counts_df, _opt_counts_meta = load_or_build_source_opt_counts(
            config_path=args.config,
            cfg=cfg,
            year_min=feature_year_min,
            year_max=feature_year_max,
            force_rebuild=args.force_rebuild_company_features,
        )
        analysis_panel_meta["workforce_rows"] = int(len(workforce_df)) if workforce_df is not None else 0
        analysis_panel_meta["opt_counts_rows"] = int(len(opt_counts_df)) if opt_counts_df is not None else 0
        analysis_panel_meta["source_mode"] = SOURCE_TENURE_MODE_RAW
        if workforce_df is not None and not workforce_df.empty and FEATURE_TENURE_ANNUAL_COL in workforce_df.columns:
            tenure_pre = _annual_to_pre_level(
                workforce_df,
                FEATURE_TENURE_ANNUAL_COL,
                year_min=feature_year_min,
                year_max=feature_year_max,
            )
            if not tenure_pre.empty:
                firms = tenure_pre[["c"]].copy()
                analysis_df = firms.copy()
            else:
                firms = pd.Series([], dtype="Int64")
                analysis_df = pd.DataFrame({"c": firms})
        elif opt_counts_df is not None and not opt_counts_df.empty:
            firms = (
                pd.Series(_safe_numeric(opt_counts_df["c"])).dropna().astype("Int64")
            ).drop_duplicates()
            analysis_df = pd.DataFrame({"c": firms.reset_index(drop=True)})
        else:
            print(
                "[tenure status] workforce and opt-count caches are empty; continuing with no firm rows."
            )
            analysis_df = pd.DataFrame(columns=["c"])
            firms = pd.Series([], dtype="Int64")

        if workforce_df is not None and not workforce_df.empty and FEATURE_TENURE_ANNUAL_COL in workforce_df.columns:
            tenure_pre = _annual_to_pre_level(
                workforce_df,
                FEATURE_TENURE_ANNUAL_COL,
                year_min=feature_year_min,
                year_max=feature_year_max,
            )
            if not tenure_pre.empty:
                analysis_df = analysis_df.merge(
                    tenure_pre[
                        [
                            "c",
                            f"{FEATURE_TENURE_ANNUAL_COL}_pre_level",
                            f"{FEATURE_TENURE_ANNUAL_COL}_pre_n_years",
                        ]
                    ].rename(
                        columns={
                            f"{FEATURE_TENURE_ANNUAL_COL}_pre_level": FEATURE_TENURE_PRE_COL,
                            f"{FEATURE_TENURE_ANNUAL_COL}_pre_n_years": "tenure_n_years",
                        }
                    ),
                    on="c",
                    how="left",
                )
                has_feature_tenure = True
                tenure_present_in_exposure = True
        else:
            print("[tenure status] workforce cache missing avg_tenure_years_annual; cannot derive all-employee tenure.")

        if opt_counts_df is not None and not opt_counts_df.empty:
            opt_pre = _build_opt_pre_columns_from_raw(
                workforce_df=workforce_df,
                opt_counts_df=opt_counts_df,
                year_min=feature_year_min,
                year_max=feature_year_max,
                n_hires_col=WRDS_NEW_HIRES_COUNT_COL,
                any_opt_count_col=WRDS_SOURCE_OPT_COUNT_COL,
                masters_opt_count_col=WRDS_MASTERS_OPT_COUNT_COL,
            )
            if not opt_pre.empty:
                analysis_df = analysis_df.merge(
                    opt_pre,
                    on="c",
                    how="left",
                )
                for col in ["any_opt_hire_count_annual_pre_level", "any_opt_hire_rate_annual_pre_level"]:
                    if col not in analysis_df.columns:
                        print(f"[warn] Raw OPT pre-level column missing from build: {col}")
            else:
                print("[warn] Raw OPT pre-level derivation returned no rows; no OPT pre-level columns added.")

        if use_position_metrics:
            positions_path = args.positions_path
            if positions_path is None:
                positions_path = _resolve_positions_path(cfg, workforce_meta)
            users_path = _resolve_users_path(cfg, workforce_meta)
            if positions_path is None:
                print(
                    "[tenure status] WRDS selected-US positions cache not found; "
                    "skipping position-derived tenure metrics."
                )
            else:
                analysis_panel_meta["positions_path"] = str(positions_path)
                analysis_panel_meta["positions_path_mtime"] = _file_mtime(positions_path)
                pos_metrics = _build_position_tenure_metrics(
                    positions_path,
                    year_min=feature_year_min,
                    year_max=feature_year_max,
                    users_path=users_path,
                )
                if pos_metrics.empty:
                    print("[tenure status] position-derived tenure metrics were empty.")
                else:
                    hires_pre = _annual_to_pre_level(
                        pos_metrics,
                        HIRE_TENURE_ANNUAL_COL,
                        year_min=feature_year_min,
                        year_max=feature_year_max,
                    )
                    new_positions_pre = _annual_to_pre_level(
                        pos_metrics,
                        NEW_POS_TENURE_ANNUAL_COL,
                        year_min=feature_year_min,
                        year_max=feature_year_max,
                    )
                    recent_grads_pre = _annual_to_pre_level(
                        pos_metrics,
                        RECENT_GRADS_TENURE_ANNUAL_COL,
                        year_min=feature_year_min,
                        year_max=feature_year_max,
                    )
                    if not hires_pre.empty:
                        analysis_df = analysis_df.merge(
                            hires_pre[
                                [
                                    "c",
                                    f"{HIRE_TENURE_ANNUAL_COL}_pre_level",
                                    f"{HIRE_TENURE_ANNUAL_COL}_pre_n_years",
                                ]
                            ],
                            on="c",
                            how="left",
                        )
                        analysis_df = analysis_df.rename(
                            columns={
                                f"{HIRE_TENURE_ANNUAL_COL}_pre_level": HIRE_TENURE_PRE_COL,
                                f"{HIRE_TENURE_ANNUAL_COL}_pre_n_years": "tenure_new_hires_n_years",
                            },
                        )
                    if not new_positions_pre.empty:
                        analysis_df = analysis_df.merge(
                            new_positions_pre[
                                [
                                    "c",
                                    f"{NEW_POS_TENURE_ANNUAL_COL}_pre_level",
                                    f"{NEW_POS_TENURE_ANNUAL_COL}_pre_n_years",
                                ]
                            ],
                            on="c",
                            how="left",
                        )
                        analysis_df = analysis_df.rename(
                            columns={
                                f"{NEW_POS_TENURE_ANNUAL_COL}_pre_level": NEW_POS_TENURE_PRE_COL,
                                f"{NEW_POS_TENURE_ANNUAL_COL}_pre_n_years": "tenure_new_positions_n_years",
                            },
                        )
                    if not recent_grads_pre.empty:
                        analysis_df = analysis_df.merge(
                            recent_grads_pre[
                                [
                                    "c",
                                    f"{RECENT_GRADS_TENURE_ANNUAL_COL}_pre_level",
                                    f"{RECENT_GRADS_TENURE_ANNUAL_COL}_pre_n_years",
                                ]
                            ],
                            on="c",
                            how="left",
                        )
                        analysis_df = analysis_df.rename(
                            columns={
                                f"{RECENT_GRADS_TENURE_ANNUAL_COL}_pre_level": RECENT_GRADS_TENURE_PRE_COL,
                                f"{RECENT_GRADS_TENURE_ANNUAL_COL}_pre_n_years": "tenure_recent_grads_n_years",
                            },
                        )
                    used_position_metrics = True

        analysis_panel_meta["position_metrics_included"] = used_position_metrics
        analysis_panel_meta["schema_version"] = ANALYSIS_PANEL_SCHEMA_VERSION
        analysis_panel_meta["position_metrics_required"] = not bool(args.skip_position_metrics)
        if workforce_meta is not None:
            analysis_panel_meta["workforce_meta_positions_path"] = workforce_meta.get("local_selected_us_positions_path")
        _save_analysis_panel(analysis_panel_path, analysis_df, analysis_panel_meta)

    requested_scope_col = _resolve_tenure_scope_pre_col(args.tenure_analysis_scope)
    if args.tenure_analysis_scope != TENURE_SCOPE_ALL and requested_scope_col not in analysis_df.columns:
        if args.tenure_analysis_scope == TENURE_SCOPE_NEW_HIRES and RECENT_GRADS_TENURE_PRE_COL in analysis_df.columns:
            print(
                "[tenure status] Requested new_hires scope unavailable in panel; "
                "falling back to recent_grads scope."
            )
            requested_scope_col = RECENT_GRADS_TENURE_PRE_COL
        else:
            print(
                f"[warn] Requested tenure analysis scope '{args.tenure_analysis_scope}' is unavailable; "
                "falling back to avg_tenure_years_annual_pre_level."
            )
            requested_scope_col = FEATURE_TENURE_PRE_COL

    tenure_cols = [
        col
        for col in [
            requested_scope_col,
            FEATURE_TENURE_PRE_COL,
            HIRE_TENURE_PRE_COL,
            NEW_POS_TENURE_PRE_COL,
            RECENT_GRADS_TENURE_PRE_COL,
        ]
        if col in analysis_df.columns
    ]
    opt_cols = [col for col in DEFAULT_OPT_COLS if col in analysis_df.columns]

    if not tenure_cols:
        print(
            "[warning] No tenure feature available for analysis. "
            "Either build features for an earlier step or provide WRDS selected-US positions data."
        )
    if not opt_cols:
        print("[warning] No OPT pre-level columns found in source panel. Nothing to correlate.")
        return

    corr_df = pd.DataFrame(_correlation_rows(analysis_df, tenure_cols, opt_cols))
    print("\n[correlation summary]")
    print(corr_df.to_string(index=False))

    binned_frames: list[pd.DataFrame] = []
    for opt_col in opt_cols:
        for tenure_col in tenure_cols:
            bin_df = _build_binned_summary(analysis_df, tenure_col, opt_col, n_bins=5)
            if not bin_df.empty:
                binned_frames.append(bin_df)
    bin_merged = pd.concat(binned_frames, ignore_index=True) if binned_frames else pd.DataFrame()

    status = pd.DataFrame(
        {
            "analysis_panel_reused": [bool(loaded_panel)],
            "analysis_panel_path": [str(analysis_panel_path)],
            "data_source_mode": [source_mode],
            "analysis_panel_rows": [int(len(analysis_df))],
            "workforce_rows": [analysis_panel_meta.get("workforce_rows", 0)],
            "opt_counts_rows": [analysis_panel_meta.get("opt_counts_rows", 0)],
            "feature_year_min": [feature_year_min],
            "feature_year_max": [feature_year_max],
            "tenure_in_feature_matrix": [has_feature_tenure],
            "tenure_in_exposure_event_study_selector": [tenure_present_in_exposure],
            "position_metrics_used": [bool(used_position_metrics)],
            "position_path": [str(args.positions_path) if args.positions_path else str(paths_cfg.get("wrds_workforce_selected_us_positions_out", ""))],
            "workforce_meta_positions_path": [
                workforce_meta.get("local_selected_us_positions_path") if workforce_meta else None
            ],
            "avg_tenure_tenure_columns": [",".join(tenure_cols)],
            "opt_columns": [",".join(opt_cols)],
            "run_binsreg": [bool(args.run_binsreg)],
            "binsreg_by_year": [bool(args.binsreg_by_year_bins)],
            "binsreg_log_opt_hires": [bool(args.binsreg_log_opt_hires)],
        }
    )

    _safe_write_csv(status, out_dir / f"{args.output_prefix}_status.csv")
    _safe_write_csv(corr_df, out_dir / f"{args.output_prefix}_correlations.csv")
    if not bin_merged.empty:
        _safe_write_csv(bin_merged, out_dir / f"{args.output_prefix}_binned.csv")

    if args.run_binsreg:
        binsreg_output_dir = Path(out_dir) / "binsreg"
        run_tenure_opt_binsreg(
            cfg_path=args.config,
            data_min_t=args.binsreg_data_min_t,
            data_max_t=args.binsreg_data_max_t,
            tenure_col=args.binsreg_tenure_col,
            tenure_scope=args.binsreg_tenure_scope,
            opt_count_col=args.binsreg_opt_count_col,
            n_hires_col=args.binsreg_n_hires_col,
            firm_size_col=args.binsreg_firm_size_col,
            metric=args.binsreg_metric,
            log_opt_hires=args.binsreg_log_opt_hires,
            bins=args.bins,
            out_dir=binsreg_output_dir,
            prefix=f"{args.output_prefix}_binsreg",
            force_rebuild=args.binsreg_force_rebuild,
            save_outputs=True,
            by_year_bins=args.binsreg_by_year_bins,
            show_plots=args.show_plots,
        )

    print(f"[done] Outputs written to {out_dir}")

if __name__ == "__main__":
    main()
