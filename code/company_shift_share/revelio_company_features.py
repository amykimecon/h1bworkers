"""Build firm-level pre-period features for OPT exposure modeling.

This module stays within the ``company_shift_share`` workflow. It combines
existing company-shift-share outputs with direct WRDS Revelio aggregates to
produce a firm-level feature frame over a configurable pre-period window.

Primary entry points:
    - ``load_or_build_company_features(...)``
    - ``build_company_features(...)``

The output is a firm-level frame keyed by ``c`` (RCID / preferred RCID) with
pre-period level and growth features, plus universe flags used by the exposure
model in ``exposure_event_study.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Iterable, Optional

import duckdb as ddb
import numpy as np
import pandas as pd

try:
    import wrds
except ImportError:  # pragma: no cover - WRDS is available in the real runtime.
    wrds = None  # type: ignore[assignment]

try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import (  # type: ignore[no-redef]
        DEFAULT_CONFIG_PATH,
        get_cfg_section,
        load_config,
    )


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


@dataclass(frozen=True)
class FeaturePaths:
    analysis_panel: Path
    instrument_components: Path
    transitions: Path
    headcounts: Path
    company_mapping: Path
    preferred_rcids: Path
    revelio_inst_crosswalk: Path
    company_features_out: Path
    outside_negative_sample_out: Path


_FEATURE_META_SUFFIX = ".meta.json"


def _escape(path: Path) -> str:
    return str(path).replace("'", "''")


def _resolve_path(paths_cfg: dict, key: str) -> Path:
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"Config paths.{key} must be set.")
    return Path(str(value))


def _resolve_optional_path(paths_cfg: dict, key: str) -> Optional[Path]:
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return Path(str(value))


def _resolve_feature_paths(cfg: dict) -> FeaturePaths:
    paths_cfg = get_cfg_section(cfg, "paths")
    company_features_out = _resolve_optional_path(paths_cfg, "company_features_out")
    outside_negative_sample_out = _resolve_optional_path(paths_cfg, "outside_negative_sample_out")
    if company_features_out is None:
        raise ValueError("Config paths.company_features_out must be set.")
    if outside_negative_sample_out is None:
        raise ValueError("Config paths.outside_negative_sample_out must be set.")
    return FeaturePaths(
        analysis_panel=_resolve_path(paths_cfg, "analysis_panel"),
        instrument_components=_resolve_path(paths_cfg, "instrument_components"),
        transitions=_resolve_path(paths_cfg, "transitions_out"),
        headcounts=_resolve_path(paths_cfg, "headcounts_out"),
        company_mapping=_resolve_path(paths_cfg, "revelio_company_mapping"),
        preferred_rcids=_resolve_path(paths_cfg, "preferred_rcids"),
        revelio_inst_crosswalk=_resolve_path(paths_cfg, "revelio_ipeds_foia_inst_crosswalk"),
        company_features_out=company_features_out,
        outside_negative_sample_out=outside_negative_sample_out,
    )


def validate_feature_window(feature_year_min: int, feature_year_max: int) -> None:
    if int(feature_year_min) > int(feature_year_max):
        raise ValueError(
            f"feature_year_min must be <= feature_year_max, got "
            f"{feature_year_min=} {feature_year_max=}."
        )
    if int(feature_year_max) >= 2016:
        raise ValueError(
            f"feature_year_max must be < 2016, got {feature_year_max}."
        )


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + _FEATURE_META_SUFFIX)


def _load_metadata(path: Path) -> dict:
    meta_path = _metadata_path(path)
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _write_metadata(path: Path, metadata: dict) -> None:
    meta_path = _metadata_path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _require_paths(paths: FeaturePaths) -> None:
    missing = [str(path) for path in paths.__dict__.values() if not Path(path).exists()]
    # Output paths may not exist yet; only enforce inputs.
    input_paths = [
        paths.analysis_panel,
        paths.instrument_components,
        paths.transitions,
        paths.headcounts,
        paths.company_mapping,
        paths.preferred_rcids,
        paths.revelio_inst_crosswalk,
    ]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def classify_opt_intensive_schools(
    components: pd.DataFrame,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    """Classify schools as OPT-intensive using only the configured feature window."""
    validate_feature_window(year_min, year_max)
    required = {"k", "t", "g_kt"}
    missing = sorted(required - set(components.columns))
    if missing:
        raise ValueError(f"components is missing required columns: {missing}")
    win = components.loc[components["t"].between(year_min, year_max), ["k", "g_kt"]].copy()
    if win.empty:
        return pd.DataFrame(columns=["k", "school_opt_rate", "opt_intensive"])
    school_rate = (
        win.groupby("k", as_index=False)["g_kt"]
        .mean()
        .rename(columns={"g_kt": "school_opt_rate"})
    )
    median_rate = school_rate["school_opt_rate"].median()
    school_rate["opt_intensive"] = school_rate["school_opt_rate"] > median_rate
    return school_rate


def _ols_slope(years: np.ndarray, values: np.ndarray) -> float:
    years_f = years.astype(float)
    values_f = values.astype(float)
    x = years_f - years_f.mean()
    denom = float(np.square(x).sum())
    if denom <= 0:
        return np.nan
    return float((x * (values_f - values_f.mean())).sum() / denom)


def summarize_pre_period_features(
    annual_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    year_min: int,
    year_max: int,
    firm_col: str = "c",
    year_col: str = "t",
    growth_min_obs: int = 3,
) -> pd.DataFrame:
    """Collapse annual firm-year features into configurable pre-period level/growth features."""
    validate_feature_window(year_min, year_max)
    work = annual_df.copy()
    if year_col not in work.columns or firm_col not in work.columns:
        raise ValueError(f"annual_df must contain '{firm_col}' and '{year_col}'.")
    work = work[work[year_col].between(year_min, year_max)].copy()
    firms = pd.Index(sorted(pd.Series(work[firm_col].dropna().unique()).astype(int).tolist()), name=firm_col)
    out = pd.DataFrame({firm_col: firms})

    for col in feature_cols:
        if col not in work.columns:
            continue
        sub = work[[firm_col, year_col, col]].copy()
        level = (
            sub.groupby(firm_col, as_index=False)[col]
            .mean()
            .rename(columns={col: f"{col}_pre_level"})
        )
        support = (
            sub.groupby(firm_col, as_index=False)[col]
            .count()
            .rename(columns={col: f"{col}_pre_n_years"})
        )

        valid = sub.dropna(subset=[col])
        growth_records: list[dict[str, float]] = []
        for firm_id, g in valid.groupby(firm_col):
            if len(g) < int(growth_min_obs):
                growth_records.append({firm_col: int(firm_id), f"{col}_pre_growth": np.nan})
                continue
            slope = _ols_slope(g[year_col].to_numpy(), g[col].to_numpy())
            growth_records.append({firm_col: int(firm_id), f"{col}_pre_growth": slope})
        growth = pd.DataFrame(growth_records) if growth_records else pd.DataFrame(columns=[firm_col, f"{col}_pre_growth"])

        out = out.merge(level, on=firm_col, how="left")
        out = out.merge(support, on=firm_col, how="left")
        out = out.merge(growth, on=firm_col, how="left")
        out[f"{col}_pre_n_years"] = out[f"{col}_pre_n_years"].fillna(0).astype(int)
        out[f"{col}_pre_level_missing_ind"] = out[f"{col}_pre_level"].isna().astype("int8")
        out[f"{col}_pre_growth_missing_ind"] = out[f"{col}_pre_growth"].isna().astype("int8")

    return out


def _naics_digits(series: pd.Series, n_digits: int) -> pd.Series:
    cleaned = series.astype("string").str.replace(r"[^0-9]", "", regex=True)
    return cleaned.str.slice(0, n_digits).fillna("__MISSING__")


def _state_coalesce(df: pd.DataFrame) -> pd.Series:
    state = df["top_state"].astype("string").str.strip()
    hq_state = df["hq_state"].astype("string").str.strip()
    state = state.where(state.notna() & state.ne(""), hq_state)
    return state.fillna("__MISSING__")


def _size_bucket(n_users: pd.Series) -> pd.Series:
    values = pd.to_numeric(n_users, errors="coerce").fillna(-1)
    conditions = [
        values < 10,
        values.between(10, 49),
        values.between(50, 249),
        values.between(250, 999),
        values >= 1000,
    ]
    labels = [
        "lt10",
        "10_49",
        "50_249",
        "250_999",
        "1000p",
    ]
    out = np.select(conditions, labels, default="unknown")
    return pd.Series(out, index=n_users.index, dtype="string")


def _load_company_meta(path: Path) -> pd.DataFrame:
    cols = [
        "rcid",
        "n_users",
        "top_state",
        "top_metro_area",
        "hq_state",
        "hq_region",
        "naics_code",
        "year_founded",
    ]
    meta = pd.read_parquet(path, columns=cols)
    meta = meta.rename(columns={"rcid": "c"}).copy()
    meta["c"] = pd.to_numeric(meta["c"], errors="coerce")
    meta = meta.dropna(subset=["c"]).copy()
    meta["c"] = meta["c"].astype(int)
    meta = meta.drop_duplicates(subset=["c"], keep="first")
    meta["naics2"] = _naics_digits(meta["naics_code"], 2)
    meta["naics4"] = _naics_digits(meta["naics_code"], 4)
    meta["company_state_feature"] = _state_coalesce(meta)
    meta["company_metro_feature"] = meta["top_metro_area"].astype("string").fillna("__MISSING__")
    meta["company_hq_region"] = meta["hq_region"].astype("string").fillna("__MISSING__")
    meta["size_bucket"] = _size_bucket(meta["n_users"])
    return meta


def _sample_stage(
    eligible: pd.DataFrame,
    target_counts: pd.DataFrame,
    group_cols: list[str],
    *,
    seed: int,
) -> pd.DataFrame:
    if eligible.empty or target_counts.empty:
        return eligible.iloc[0:0].copy()
    merged = eligible.merge(target_counts, on=group_cols, how="inner")
    if merged.empty:
        return merged
    parts: list[pd.DataFrame] = []
    for idx, (_, group) in enumerate(merged.groupby(group_cols, dropna=False), start=1):
        n_target = int(group["n_target"].iloc[0])
        if n_target <= 0:
            continue
        n_take = min(n_target, len(group))
        sampled = group.sample(n=n_take, random_state=int(seed) + idx)
        parts.append(sampled)
    if not parts:
        return merged.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def sample_outside_negative_firms(
    company_meta: pd.DataFrame,
    analysis_firms: pd.Series,
    preferred_rcids: pd.Series,
    *,
    ratio: float,
    seed: int,
    min_n_users: int,
) -> pd.DataFrame:
    """Sample outside-universe negative firms with deterministic fallback relaxation."""
    analysis_ids = pd.Series(pd.to_numeric(analysis_firms, errors="coerce")).dropna().astype(int).unique()
    preferred_ids = set(pd.Series(pd.to_numeric(preferred_rcids, errors="coerce")).dropna().astype(int).tolist())
    analysis_meta = company_meta[company_meta["c"].isin(analysis_ids)].copy()
    eligible = company_meta[
        (~company_meta["c"].isin(preferred_ids))
        & (~company_meta["c"].isin(analysis_ids))
        & (pd.to_numeric(company_meta["n_users"], errors="coerce").fillna(0) >= int(min_n_users))
        & company_meta["company_state_feature"].ne("__MISSING__")
        & company_meta["naics2"].ne("__MISSING__")
    ].copy()

    if analysis_meta.empty or eligible.empty:
        return eligible.iloc[0:0].copy()

    target_total = int(np.ceil(len(analysis_meta) * float(ratio)))
    chosen_frames: list[pd.DataFrame] = []

    exact_targets = (
        analysis_meta.groupby(["size_bucket", "naics2", "company_state_feature"], dropna=False)
        .size()
        .reset_index(name="n_analysis")
    )
    exact_targets["n_target"] = np.ceil(exact_targets["n_analysis"] * float(ratio)).astype(int)
    pick = _sample_stage(
        eligible,
        exact_targets[["size_bucket", "naics2", "company_state_feature", "n_target"]],
        ["size_bucket", "naics2", "company_state_feature"],
        seed=seed,
    )
    if not pick.empty:
        chosen_frames.append(pick)
    chosen_ids = set(pick["c"].tolist())

    remaining = eligible[~eligible["c"].isin(chosen_ids)].copy()
    if len(chosen_ids) < target_total and not remaining.empty:
        coarse_targets = (
            analysis_meta.groupby(["size_bucket", "company_state_feature"], dropna=False)
            .size()
            .reset_index(name="n_analysis")
        )
        already = (
            pick.groupby(["size_bucket", "company_state_feature"], dropna=False)
            .size()
            .reset_index(name="n_selected")
            if not pick.empty
            else pd.DataFrame(columns=["size_bucket", "company_state_feature", "n_selected"])
        )
        coarse_targets = coarse_targets.merge(already, on=["size_bucket", "company_state_feature"], how="left")
        coarse_targets["n_selected"] = coarse_targets["n_selected"].fillna(0)
        coarse_targets["n_target"] = (
            np.ceil(coarse_targets["n_analysis"] * float(ratio)) - coarse_targets["n_selected"]
        ).clip(lower=0).astype(int)
        pick2 = _sample_stage(
            remaining,
            coarse_targets[["size_bucket", "company_state_feature", "n_target"]],
            ["size_bucket", "company_state_feature"],
            seed=seed + 1_000,
        )
        if not pick2.empty:
            chosen_frames.append(pick2)
            chosen_ids.update(pick2["c"].tolist())
            remaining = remaining[~remaining["c"].isin(chosen_ids)].copy()

    if len(chosen_ids) < target_total and not remaining.empty:
        size_targets = (
            analysis_meta.groupby(["size_bucket"], dropna=False)
            .size()
            .reset_index(name="n_analysis")
        )
        already = (
            pd.concat(chosen_frames, ignore_index=True)
            .groupby(["size_bucket"], dropna=False)
            .size()
            .reset_index(name="n_selected")
            if chosen_frames
            else pd.DataFrame(columns=["size_bucket", "n_selected"])
        )
        size_targets = size_targets.merge(already, on=["size_bucket"], how="left")
        size_targets["n_selected"] = size_targets["n_selected"].fillna(0)
        size_targets["n_target"] = (
            np.ceil(size_targets["n_analysis"] * float(ratio)) - size_targets["n_selected"]
        ).clip(lower=0).astype(int)
        pick3 = _sample_stage(
            remaining,
            size_targets[["size_bucket", "n_target"]],
            ["size_bucket"],
            seed=seed + 2_000,
        )
        if not pick3.empty:
            chosen_frames.append(pick3)
            chosen_ids.update(pick3["c"].tolist())
            remaining = remaining[~remaining["c"].isin(chosen_ids)].copy()

    if len(chosen_ids) < target_total and not remaining.empty:
        need = target_total - len(chosen_ids)
        pick4 = remaining.sample(n=min(need, len(remaining)), random_state=seed + 3_000)
        chosen_frames.append(pick4)

    if not chosen_frames:
        return eligible.iloc[0:0].copy()
    sampled = pd.concat(chosen_frames, ignore_index=True)
    sampled = sampled.drop_duplicates(subset=["c"], keep="first").reset_index(drop=True)
    sampled["outside_negative_candidate"] = 1
    return sampled


def _duckdb_selected_filter(con: ddb.DuckDBPyConnection, selected_firms: Optional[pd.DataFrame]) -> str:
    if selected_firms is None:
        return ""
    con.register("selected_firms_df", selected_firms[["c"]].drop_duplicates())
    con.sql("CREATE OR REPLACE TEMP VIEW selected_firms AS SELECT CAST(c AS BIGINT) AS c FROM selected_firms_df")
    return "JOIN selected_firms sf ON base.c = sf.c"


def _build_local_company_year_features(
    paths: FeaturePaths,
    *,
    year_min: int,
    year_max: int,
    selected_firms: Optional[pd.DataFrame],
) -> pd.DataFrame:
    con = ddb.connect()
    try:
        con.sql(f"CREATE OR REPLACE TEMP VIEW analysis_panel AS SELECT * FROM read_parquet('{_escape(paths.analysis_panel)}')")
        con.sql(f"CREATE OR REPLACE TEMP VIEW instrument_components AS SELECT * FROM read_parquet('{_escape(paths.instrument_components)}')")
        con.sql(f"CREATE OR REPLACE TEMP VIEW transitions AS SELECT * FROM read_parquet('{_escape(paths.transitions)}')")
        con.sql(f"CREATE OR REPLACE TEMP VIEW headcounts AS SELECT * FROM read_parquet('{_escape(paths.headcounts)}')")
        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW inst_map AS
            WITH base AS (
                SELECT
                    LOWER(TRIM(CAST(university_raw AS VARCHAR))) AS university_raw_key,
                    CAST(CAST(UNITID AS BIGINT) AS VARCHAR) AS k
                FROM read_parquet('{_escape(paths.revelio_inst_crosswalk)}')
                WHERE university_raw IS NOT NULL
                  AND UNITID IS NOT NULL
            )
            SELECT university_raw_key, MIN(k) AS k
            FROM base
            GROUP BY 1
            """
        )

        selected_join = _duckdb_selected_filter(con, selected_firms)

        panel_sql = f"""
        WITH base AS (
            SELECT
                CAST(c AS BIGINT) AS c,
                CAST(t AS INTEGER) AS t,
                CAST(masters_opt_hires_correction_aware AS DOUBLE) AS opt_hire_count_annual,
                CASE
                    WHEN y_new_hires_lag0 IS NULL OR y_new_hires_lag0 <= 0 THEN NULL
                    ELSE CAST(masters_opt_hires_correction_aware AS DOUBLE) / CAST(y_new_hires_lag0 AS DOUBLE)
                END AS opt_hire_rate_annual,
                CAST(y_new_hires_lag0 AS DOUBLE) AS n_new_hires_panel_annual,
                CAST(y_cst_lag0 AS DOUBLE) AS firm_size_panel_annual
            FROM analysis_panel
            WHERE t BETWEEN {int(year_min)} AND {int(year_max)}
        )
        SELECT *
        FROM base
        {selected_join}
        """
        panel_features = con.sql(panel_sql).df()

        school_sql = f"""
        WITH school_rates AS (
            SELECT
                k,
                AVG(CAST(g_kt AS DOUBLE)) AS school_opt_rate
            FROM instrument_components
            WHERE t BETWEEN {int(year_min)} AND {int(year_max)}
            GROUP BY 1
        ),
        school_cut AS (
            SELECT MEDIAN(school_opt_rate) AS med
            FROM school_rates
        ),
        intensive AS (
            SELECT
                sr.k,
                CASE WHEN sr.school_opt_rate > sc.med THEN 1 ELSE 0 END AS opt_intensive
            FROM school_rates sr
            CROSS JOIN school_cut sc
        ),
        base AS (
            SELECT
                CAST(ic.c AS BIGINT) AS c,
                CAST(ic.t AS INTEGER) AS t,
                SUM(CASE WHEN COALESCE(i.opt_intensive, 0) = 1 THEN CAST(ic.n_transitions AS DOUBLE) ELSE 0 END)
                    AS intensive_transitions,
                MAX(CAST(ic.total_new_hires AS DOUBLE)) AS total_new_hires_components,
                COUNT(DISTINCT CASE WHEN COALESCE(ic.n_transitions, 0) > 0 THEN ic.k END) AS n_schools_new_hire_annual
            FROM instrument_components ic
            LEFT JOIN intensive i USING (k)
            WHERE ic.t BETWEEN {int(year_min)} AND {int(year_max)}
            GROUP BY 1, 2
        )
        SELECT
            c,
            t,
            CASE
                WHEN total_new_hires_components IS NULL OR total_new_hires_components <= 0 THEN NULL
                ELSE intensive_transitions / total_new_hires_components
            END AS school_opt_share_new_hire_annual,
            total_new_hires_components,
            n_schools_new_hire_annual
        FROM base
        {selected_join}
        """
        school_features = con.sql(school_sql).df()

        tenured_sql = f"""
        WITH school_rates AS (
            SELECT
                k,
                AVG(CAST(g_kt AS DOUBLE)) AS school_opt_rate
            FROM instrument_components
            WHERE t BETWEEN {int(year_min)} AND {int(year_max)}
            GROUP BY 1
        ),
        school_cut AS (
            SELECT MEDIAN(school_opt_rate) AS med
            FROM school_rates
        ),
        intensive AS (
            SELECT
                sr.k,
                CASE WHEN sr.school_opt_rate > sc.med THEN 1 ELSE 0 END AS opt_intensive
            FROM school_rates sr
            CROSS JOIN school_cut sc
        ),
        mapped AS (
            SELECT
                CAST(tr.rcid AS BIGINT) AS c,
                CAST(tr.year AS INTEGER) AS t,
                im.k,
                SUM(CAST(tr.n_emp AS DOUBLE)) AS n_emp
            FROM transitions tr
            JOIN inst_map im
              ON LOWER(TRIM(CAST(tr.university_raw AS VARCHAR))) = im.university_raw_key
            WHERE tr.year BETWEEN {int(year_min)} AND {int(year_max)}
            GROUP BY 1, 2, 3
        ),
        base AS (
            SELECT
                m.c,
                m.t,
                SUM(CASE WHEN COALESCE(i.opt_intensive, 0) = 1 THEN m.n_emp ELSE 0 END) AS intensive_emp,
                SUM(m.n_emp) AS total_emp_mapped,
                COUNT(DISTINCT CASE WHEN m.n_emp > 0 THEN m.k END) AS n_schools_tenured_annual
            FROM mapped m
            LEFT JOIN intensive i USING (k)
            GROUP BY 1, 2
        )
        SELECT
            c,
            t,
            CASE WHEN total_emp_mapped IS NULL OR total_emp_mapped <= 0 THEN NULL ELSE intensive_emp / total_emp_mapped END
                AS school_opt_share_tenured_annual,
            n_schools_tenured_annual
        FROM base
        {selected_join}
        """
        tenured_features = con.sql(tenured_sql).df()

        headcount_sql = f"""
        WITH base AS (
            SELECT
                CAST(rcid AS BIGINT) AS c,
                CAST(year AS INTEGER) AS t,
                CAST(total_headcount AS DOUBLE) AS total_headcount_annual,
                CAST(long_term_headcount AS DOUBLE) AS long_term_headcount_annual
            FROM headcounts
            WHERE year BETWEEN {int(year_min)} AND {int(year_max)}
        )
        SELECT *
        FROM base
        {selected_join}
        """
        headcount_features = con.sql(headcount_sql).df()
    finally:
        con.close()

    annual = panel_features.merge(school_features, on=["c", "t"], how="outer")
    annual = annual.merge(tenured_features, on=["c", "t"], how="outer")
    annual = annual.merge(headcount_features, on=["c", "t"], how="outer")
    return annual


def _set_statement_timeout(db: wrds.Connection, timeout_ms: Optional[int]) -> None:
    if timeout_ms is None or int(timeout_ms) <= 0:
        return
    timeout_ms_int = int(timeout_ms)
    try:
        db.raw_sql(f"SELECT set_config('statement_timeout', '{timeout_ms_int}', false)")
    except Exception:
        return


def _wrds_connect_args(query_timeout_ms: Optional[int]) -> dict:
    if query_timeout_ms is None or int(query_timeout_ms) <= 0:
        return {}
    timeout_ms_int = int(query_timeout_ms)
    return {
        "wrds_connect_args": {
            "options": f"-c statement_timeout={timeout_ms_int} -c lock_timeout={timeout_ms_int}",
        }
    }


def _open_wrds_connection(wrds_username: str, query_timeout_ms: Optional[int]) -> wrds.Connection:
    if wrds is None:  # pragma: no cover
        raise ImportError("wrds is not installed.")
    return wrds.Connection(wrds_username=wrds_username, **_wrds_connect_args(query_timeout_ms))


def _run_sql_with_retries(
    db: wrds.Connection,
    sql: str,
    *,
    wrds_username: str,
    query_timeout_ms: Optional[int],
    max_retries: int,
    label: str,
) -> tuple[pd.DataFrame, wrds.Connection]:
    attempts = max(1, int(max_retries) + 1)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            _set_statement_timeout(db, query_timeout_ms)
            return db.raw_sql(sql), db
        except Exception as exc:
            last_exc = exc
            print(f"[wrds] {label} failed on attempt {attempt}/{attempts}: {exc}")
            if attempt < attempts:
                try:
                    db.close()
                except Exception:
                    pass
                db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
    assert last_exc is not None
    raise last_exc


def _rcid_chunks(rcids: list[int], batch_size: int) -> list[list[int]]:
    if batch_size <= 0 or len(rcids) <= batch_size:
        return [rcids]
    return [rcids[i : i + batch_size] for i in range(0, len(rcids), batch_size)]


def _detect_user_race_column(db: wrds.Connection) -> Optional[str]:
    candidate_order = [
        "race_ethnicity",
        "race_ethnicity_raw",
        "race",
        "ethnicity",
        "demographic_race",
    ]
    try:
        cols = db.raw_sql(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'revelio'
              AND table_name IN ('individual_user', 'individual_user_raw')
            """
        )
    except Exception:
        return None
    available = {(str(r["table_name"]).lower(), str(r["column_name"]).lower()): str(r["column_name"]) for _, r in cols.iterrows()}
    for candidate in candidate_order:
        for table in ("individual_user", "individual_user_raw"):
            key = (table.lower(), candidate.lower())
            if key in available:
                return f"{table}.{available[key]}"
    return None


def _build_wrds_company_year_features_query(
    rcids: list[int],
    *,
    year_min: int,
    year_max: int,
    race_ref: Optional[str],
) -> str:
    rcid_sql = ", ".join(str(int(v)) for v in rcids)
    race_cte = ""
    race_join = ""
    race_cols = (
        "NULL::DOUBLE PRECISION AS race_share_asian_annual,"
        "NULL::DOUBLE PRECISION AS race_share_black_annual,"
        "NULL::DOUBLE PRECISION AS race_share_hispanic_annual,"
        "NULL::DOUBLE PRECISION AS race_share_white_annual,"
        "NULL::DOUBLE PRECISION AS race_share_other_annual"
    )
    if race_ref:
        table_name, col_name = race_ref.split(".", 1)
        race_cte = f"""
        ,
        user_race AS MATERIALIZED (
            SELECT
                user_id,
                CAST({col_name} AS VARCHAR) AS race_raw
            FROM revelio.{table_name}
            WHERE user_id IN (SELECT DISTINCT user_id FROM us_positions)
        )
        """
        race_join = "LEFT JOIN user_race ur ON ur.user_id = p.user_id"
        race_cols = """
        AVG(CASE
                WHEN race_raw IS NULL OR TRIM(race_raw) = '' THEN NULL
                WHEN race_raw ~* '(asian|aapi|pacific islander)' THEN 1
                ELSE 0
            END) AS race_share_asian_annual,
        AVG(CASE
                WHEN race_raw IS NULL OR TRIM(race_raw) = '' THEN NULL
                WHEN race_raw ~* '(black|african)' THEN 1
                ELSE 0
            END) AS race_share_black_annual,
        AVG(CASE
                WHEN race_raw IS NULL OR TRIM(race_raw) = '' THEN NULL
                WHEN race_raw ~* '(hispanic|latino)' THEN 1
                ELSE 0
            END) AS race_share_hispanic_annual,
        AVG(CASE
                WHEN race_raw IS NULL OR TRIM(race_raw) = '' THEN NULL
                WHEN race_raw ~* 'white' THEN 1
                ELSE 0
            END) AS race_share_white_annual,
        AVG(CASE
                WHEN race_raw IS NULL OR TRIM(race_raw) = '' THEN NULL
                WHEN race_raw ~* '(asian|aapi|pacific islander|black|african|hispanic|latino|white)' THEN 0
                ELSE 1
            END) AS race_share_other_annual
        """

    return f"""
    WITH us_positions AS MATERIALIZED (
        SELECT
            user_id,
            rcid,
            startdate::DATE AS startdate,
            COALESCE(enddate::DATE, DATE '2025-12-31') AS enddate,
            role_k17000_v3,
            salary,
            total_compensation
        FROM revelio.individual_positions
        WHERE country = 'United States'
          AND rcid IN ({rcid_sql})
          AND startdate IS NOT NULL
    ),
    user_educ AS MATERIALIZED (
        SELECT
            e.user_id,
            MAX(CASE
                    WHEN e.university_country IS NOT NULL AND e.university_country <> 'United States' THEN 1
                    ELSE 0
                END) AS has_nonus_educ,
            MIN(
                CASE
                    WHEN e.enddate IS NULL THEN NULL
                    ELSE EXTRACT(YEAR FROM e.enddate)::INT
                         - CASE
                               WHEN LOWER(COALESCE(e.degree, '')) LIKE '%doctor%' OR LOWER(COALESCE(e.degree, '')) LIKE '%phd%' THEN 30
                               WHEN LOWER(COALESCE(e.degree, '')) LIKE '%master%' OR LOWER(COALESCE(e.degree, '')) LIKE '%mba%' THEN 26
                               WHEN LOWER(COALESCE(e.degree, '')) LIKE '%bachelor%' THEN 22
                               WHEN LOWER(COALESCE(e.degree, '')) LIKE '%associate%' THEN 20
                               WHEN LOWER(COALESCE(e.degree, '')) LIKE '%high school%' THEN 18
                               ELSE 22
                           END
                END
            ) AS est_yob
        FROM revelio.individual_user_education e
        WHERE e.user_id IN (SELECT DISTINCT user_id FROM us_positions)
        GROUP BY 1
    ),
    company_user_first_start AS MATERIALIZED (
        SELECT
            user_id,
            rcid,
            MIN(startdate) AS first_start
        FROM us_positions
        GROUP BY 1, 2
    ),
    new_hires AS (
        SELECT
            rcid,
            EXTRACT(YEAR FROM first_start)::INT AS year,
            COUNT(DISTINCT user_id) AS n_new_hires_wrds_annual
        FROM company_user_first_start
        WHERE EXTRACT(YEAR FROM first_start)::INT BETWEEN {int(year_min)} AND {int(year_max)}
        GROUP BY 1, 2
    )
    {race_cte},
    active_user_year AS (
        SELECT
            p.rcid,
            gs.year::INT AS year,
            p.user_id,
            MAX(NULLIF(TRIM(CAST(p.salary AS VARCHAR)), '')::DOUBLE PRECISION) AS salary,
            MAX(NULLIF(TRIM(CAST(p.total_compensation AS VARCHAR)), '')::DOUBLE PRECISION) AS total_compensation,
            MAX(COALESCE(ue.has_nonus_educ, 0)) AS has_nonus_educ,
            MAX(ue.est_yob) AS est_yob,
            MAX(cufs.first_start) AS first_start,
            MAX(CASE WHEN p.role_k17000_v3 IS NULL OR TRIM(CAST(p.role_k17000_v3 AS VARCHAR)) = '' THEN 1 ELSE 0 END) AS role_unknown,
            MAX(CASE WHEN p.role_k17000_v3 ~* '(engineer|software|developer|programmer|data scientist|machine learning|scientist)' THEN 1 ELSE 0 END) AS role_engineering,
            MAX(CASE WHEN p.role_k17000_v3 ~* '(manager|director|vice president|vp|chief|head|lead)' THEN 1 ELSE 0 END) AS role_management,
            MAX(CASE WHEN p.role_k17000_v3 ~* '(finance|account|auditor|controller|treasury)' THEN 1 ELSE 0 END) AS role_finance,
            MAX(CASE WHEN p.role_k17000_v3 ~* '(operations|project|program|product|consultant|analyst)' THEN 1 ELSE 0 END) AS role_operations,
            MAX(COALESCE(ur.race_raw, NULL)) AS race_raw
        FROM us_positions p
        JOIN company_user_first_start cufs
          ON cufs.user_id = p.user_id
         AND cufs.rcid = p.rcid
        JOIN LATERAL generate_series(
            GREATEST(EXTRACT(YEAR FROM p.startdate)::INT, {int(year_min)}),
            LEAST(EXTRACT(YEAR FROM p.enddate)::INT, {int(year_max)})
        ) AS gs(year) ON TRUE
        LEFT JOIN user_educ ue
          ON ue.user_id = p.user_id
        {race_join}
        GROUP BY 1, 2, 3
    )
    SELECT
        CAST(ay.rcid AS BIGINT) AS c,
        CAST(ay.year AS INTEGER) AS t,
        COUNT(DISTINCT ay.user_id) AS total_headcount_wrds_annual,
        COUNT(DISTINCT CASE WHEN make_date(ay.year, 12, 31) >= ay.first_start + INTERVAL '365 days' THEN ay.user_id END)
            AS long_term_headcount_wrds_annual,
        AVG(ay.salary) AS salary_mean_annual,
        VAR_SAMP(ay.salary) AS salary_var_annual,
        AVG(ay.total_compensation) AS total_comp_mean_annual,
        VAR_SAMP(ay.total_compensation) AS total_comp_var_annual,
        AVG(CASE WHEN ay.salary IS NULL AND ay.total_compensation IS NULL THEN 1 ELSE 0 END)
            AS compensation_missing_share_annual,
        AVG(CASE WHEN ay.has_nonus_educ = 1 THEN 1 ELSE 0 END) AS nonus_educ_share_annual,
        AVG(CASE
                WHEN ay.est_yob IS NULL THEN NULL
                WHEN (ay.year - ay.est_yob) < 30 THEN 1
                ELSE 0
            END) AS age_share_lt30_annual,
        AVG(CASE
                WHEN ay.est_yob IS NULL THEN NULL
                WHEN (ay.year - ay.est_yob) BETWEEN 30 AND 39 THEN 1
                ELSE 0
            END) AS age_share_30_39_annual,
        AVG(CASE
                WHEN ay.est_yob IS NULL THEN NULL
                WHEN (ay.year - ay.est_yob) BETWEEN 40 AND 49 THEN 1
                ELSE 0
            END) AS age_share_40_49_annual,
        AVG(CASE
                WHEN ay.est_yob IS NULL THEN NULL
                WHEN (ay.year - ay.est_yob) BETWEEN 50 AND 59 THEN 1
                ELSE 0
            END) AS age_share_50_59_annual,
        AVG(CASE
                WHEN ay.est_yob IS NULL THEN NULL
                WHEN (ay.year - ay.est_yob) >= 60 THEN 1
                ELSE 0
            END) AS age_share_60p_annual,
        AVG(CASE WHEN ay.role_unknown = 1 THEN 1 ELSE 0 END) AS role_share_unknown_annual,
        AVG(CASE WHEN ay.role_engineering = 1 THEN 1 ELSE 0 END) AS role_share_engineering_annual,
        AVG(CASE WHEN ay.role_management = 1 THEN 1 ELSE 0 END) AS role_share_management_annual,
        AVG(CASE WHEN ay.role_finance = 1 THEN 1 ELSE 0 END) AS role_share_finance_annual,
        AVG(CASE WHEN ay.role_operations = 1 THEN 1 ELSE 0 END) AS role_share_operations_annual,
        {race_cols},
        MAX(nh.n_new_hires_wrds_annual)::DOUBLE PRECISION AS n_new_hires_wrds_annual
    FROM active_user_year ay
    LEFT JOIN new_hires nh
      ON nh.rcid = ay.rcid
     AND nh.year = ay.year
    GROUP BY 1, 2
    ORDER BY 1, 2
    """


def _build_wrds_company_year_features(
    rcids: list[int],
    *,
    wrds_username: str,
    year_min: int,
    year_max: int,
    rcid_batch_size: int,
    query_timeout_ms: Optional[int],
    query_max_retries: int,
) -> pd.DataFrame:
    if not rcids:
        return pd.DataFrame(columns=["c", "t"])
    if wrds is None:  # pragma: no cover
        raise ImportError("wrds is not installed.")

    db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
    race_ref = _detect_user_race_column(db)
    batches = _rcid_chunks(sorted({int(v) for v in rcids}), int(rcid_batch_size))
    frames: list[pd.DataFrame] = []
    try:
        for idx, batch in enumerate(batches, start=1):
            sql = _build_wrds_company_year_features_query(
                batch,
                year_min=year_min,
                year_max=year_max,
                race_ref=race_ref,
            )
            label = f"company-year WRDS enrichment batch {idx}/{len(batches)}"
            df, db = _run_sql_with_retries(
                db,
                sql,
                wrds_username=wrds_username,
                query_timeout_ms=query_timeout_ms,
                max_retries=query_max_retries,
                label=label,
            )
            if not df.empty:
                df["c"] = pd.to_numeric(df["c"], errors="coerce").astype("Int64")
                df["t"] = pd.to_numeric(df["t"], errors="coerce").astype("Int64")
                df = df.dropna(subset=["c", "t"]).copy()
                df["c"] = df["c"].astype(int)
                df["t"] = df["t"].astype(int)
            frames.append(df)
            print(f"[company_features] {label}: {len(df):,} rows")
    finally:
        try:
            db.close()
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["c", "t"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["c", "t"], keep="first")


def _merge_annual_features(local_annual: pd.DataFrame, wrds_annual: pd.DataFrame) -> pd.DataFrame:
    annual = local_annual.merge(wrds_annual, on=["c", "t"], how="outer")
    coalesce_pairs = [
        ("n_new_hires_wrds_annual", "n_new_hires_panel_annual", "n_new_hires_annual"),
        ("total_headcount_annual", "total_headcount_wrds_annual", "firm_size_annual"),
        ("long_term_headcount_annual", "long_term_headcount_wrds_annual", "long_term_headcount_combined_annual"),
    ]
    for left, right, out_col in coalesce_pairs:
        left_vals = annual[left] if left in annual.columns else pd.Series(np.nan, index=annual.index)
        right_vals = annual[right] if right in annual.columns else pd.Series(np.nan, index=annual.index)
        annual[out_col] = left_vals.where(left_vals.notna(), right_vals)
    return annual


def _build_static_feature_frame(company_meta: pd.DataFrame, firms: pd.DataFrame, *, feature_year_max: int) -> pd.DataFrame:
    static = firms.merge(company_meta, on="c", how="left")
    static["company_age_feature"] = np.where(
        pd.to_numeric(static["year_founded"], errors="coerce").notna(),
        np.maximum(0, int(feature_year_max) - pd.to_numeric(static["year_founded"], errors="coerce")),
        np.nan,
    )
    static["company_n_users_log1p"] = np.log1p(pd.to_numeric(static["n_users"], errors="coerce"))
    static_cols = [
        "c",
        "in_analysis_universe",
        "outside_negative_candidate",
        "naics2",
        "naics4",
        "company_state_feature",
        "company_metro_feature",
        "company_hq_region",
        "company_age_feature",
        "company_n_users_log1p",
    ]
    return static[static_cols].drop_duplicates(subset=["c"], keep="first")


def _prepare_feature_firms(
    company_meta: pd.DataFrame,
    analysis_panel_path: Path,
    preferred_rcids_path: Path,
    *,
    ratio: float,
    seed: int,
    min_n_users: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_firms = pd.read_parquet(analysis_panel_path, columns=["c"])
    analysis_firms["c"] = pd.to_numeric(analysis_firms["c"], errors="coerce")
    analysis_firms = analysis_firms.dropna(subset=["c"]).copy()
    analysis_firms["c"] = analysis_firms["c"].astype(int)
    analysis_firms = analysis_firms[["c"]].drop_duplicates().copy()
    analysis_firms["in_analysis_universe"] = 1

    preferred = pd.read_parquet(preferred_rcids_path, columns=["preferred_rcid"]).rename(columns={"preferred_rcid": "c"})
    outside = sample_outside_negative_firms(
        company_meta,
        analysis_firms["c"],
        preferred["c"],
        ratio=ratio,
        seed=seed,
        min_n_users=min_n_users,
    )
    analysis_firms["outside_negative_candidate"] = 0
    if outside.empty:
        outside = pd.DataFrame(columns=["c", "outside_negative_candidate"])
    outside["in_analysis_universe"] = 0
    firms = pd.concat(
        [
            analysis_firms[["c", "in_analysis_universe", "outside_negative_candidate"]],
            outside[["c", "in_analysis_universe", "outside_negative_candidate"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["c"], keep="first")
    return firms, outside


def build_company_features(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    feature_year_min: Optional[int] = None,
    feature_year_max: Optional[int] = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = _resolve_feature_paths(cfg_full)
    _require_paths(paths)

    feature_cfg = get_cfg_section(cfg_full, "revelio_company_features")
    year_min = int(feature_year_min if feature_year_min is not None else feature_cfg.get("feature_year_min", 2010))
    year_max = int(feature_year_max if feature_year_max is not None else feature_cfg.get("feature_year_max", 2015))
    validate_feature_window(year_min, year_max)

    out_path = paths.company_features_out
    metadata = {
        "feature_year_min": year_min,
        "feature_year_max": year_max,
        "outside_negative_ratio": float(feature_cfg.get("outside_negative_ratio", 2.0)),
        "outside_negative_seed": int(feature_cfg.get("outside_negative_seed", 42)),
    }
    if out_path.exists() and not force_rebuild:
        current_meta = _load_metadata(out_path)
        if (
            current_meta.get("feature_year_min") == year_min
            and current_meta.get("feature_year_max") == year_max
            and current_meta.get("outside_negative_ratio") == metadata["outside_negative_ratio"]
            and current_meta.get("outside_negative_seed") == metadata["outside_negative_seed"]
        ):
            print(f"[company_features] Reusing cached features from {out_path}")
            return pd.read_parquet(out_path)

    t0 = time.time()
    print(
        f"[company_features] Building firm features for "
        f"{year_min}–{year_max}"
    )

    company_meta = _load_company_meta(paths.company_mapping)
    firms, outside = _prepare_feature_firms(
        company_meta,
        paths.analysis_panel,
        paths.preferred_rcids,
        ratio=float(feature_cfg.get("outside_negative_ratio", 2.0)),
        seed=int(feature_cfg.get("outside_negative_seed", 42)),
        min_n_users=int(feature_cfg.get("outside_negative_min_n_users", 10)),
    )
    selected_firms = firms[["c"]].copy()

    local_annual = _build_local_company_year_features(
        paths,
        year_min=year_min,
        year_max=year_max,
        selected_firms=selected_firms,
    )

    wrds_annual = _build_wrds_company_year_features(
        firms["c"].tolist(),
        wrds_username=str(feature_cfg.get("wrds_username", "")).strip(),
        year_min=year_min,
        year_max=year_max,
        rcid_batch_size=int(feature_cfg.get("wrds_rcid_batch_size", 100)),
        query_timeout_ms=int(float(feature_cfg.get("query_timeout_minutes", 5)) * 60_000),
        query_max_retries=int(feature_cfg.get("query_max_retries", 1)),
    )

    annual = _merge_annual_features(local_annual, wrds_annual)
    annual = annual.merge(firms[["c", "in_analysis_universe", "outside_negative_candidate"]], on="c", how="left")

    feature_cols = [
        "opt_hire_count_annual",
        "opt_hire_rate_annual",
        "school_opt_share_new_hire_annual",
        "school_opt_share_tenured_annual",
        "n_schools_new_hire_annual",
        "n_schools_tenured_annual",
        "firm_size_annual",
        "total_headcount_annual",
        "long_term_headcount_annual",
        "long_term_headcount_combined_annual",
        "n_new_hires_annual",
        "salary_mean_annual",
        "salary_var_annual",
        "total_comp_mean_annual",
        "total_comp_var_annual",
        "compensation_missing_share_annual",
        "nonus_educ_share_annual",
        "age_share_lt30_annual",
        "age_share_30_39_annual",
        "age_share_40_49_annual",
        "age_share_50_59_annual",
        "age_share_60p_annual",
        "role_share_unknown_annual",
        "role_share_engineering_annual",
        "role_share_management_annual",
        "role_share_finance_annual",
        "role_share_operations_annual",
        "race_share_asian_annual",
        "race_share_black_annual",
        "race_share_hispanic_annual",
        "race_share_white_annual",
        "race_share_other_annual",
    ]
    summarized = summarize_pre_period_features(
        annual,
        feature_cols,
        year_min=year_min,
        year_max=year_max,
    )
    static = _build_static_feature_frame(company_meta, firms, feature_year_max=year_max)
    feature_df = static.merge(summarized, on="c", how="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(out_path, index=False)
    print(f"[company_features] Wrote {out_path}")

    paths.outside_negative_sample_out.parent.mkdir(parents=True, exist_ok=True)
    outside.to_parquet(paths.outside_negative_sample_out, index=False)
    print(f"[company_features] Wrote {paths.outside_negative_sample_out}")

    metadata.update(
        {
            "n_feature_firms": int(len(feature_df)),
            "n_analysis_firms": int(feature_df["in_analysis_universe"].fillna(0).sum()),
            "n_outside_negative_candidates": int(feature_df["outside_negative_candidate"].fillna(0).sum()),
            "build_seconds": round(time.time() - t0, 2),
        }
    )
    _write_metadata(out_path, metadata)
    return feature_df


def load_or_build_company_features(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    feature_year_min: Optional[int] = None,
    feature_year_max: Optional[int] = None,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = _resolve_feature_paths(cfg_full)
    feature_cfg = get_cfg_section(cfg_full, "revelio_company_features")
    year_min = int(feature_year_min if feature_year_min is not None else feature_cfg.get("feature_year_min", 2010))
    year_max = int(feature_year_max if feature_year_max is not None else feature_cfg.get("feature_year_max", 2015))
    validate_feature_window(year_min, year_max)

    if paths.company_features_out.exists() and not force_rebuild:
        meta = _load_metadata(paths.company_features_out)
        if meta.get("feature_year_min") == year_min and meta.get("feature_year_max") == year_max:
            return pd.read_parquet(paths.company_features_out), meta

    df = build_company_features(
        config_path=config_path,
        cfg=cfg_full,
        feature_year_min=year_min,
        feature_year_max=year_max,
        force_rebuild=force_rebuild,
    )
    return df, _load_metadata(paths.company_features_out)


def _parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build firm-level pre-period features for OPT exposure modeling.")
    parser.add_argument("--config", type=Path, default=None, help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).")
    parser.add_argument("--feature-year-min", type=int, default=None, help="Override revelio_company_features.feature_year_min.")
    parser.add_argument("--feature-year-max", type=int, default=None, help="Override revelio_company_features.feature_year_max.")
    parser.add_argument("--force-rebuild", action="store_true", default=False, help="Rebuild features even if cached outputs exist.")
    return parser.parse_args(args)


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(cli_args)
    build_company_features(
        config_path=args.config,
        feature_year_min=args.feature_year_min,
        feature_year_max=args.feature_year_max,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
