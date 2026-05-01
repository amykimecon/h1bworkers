"""Build firm-level pre-period features for OPT exposure modeling.

This module now builds the exposure-model feature frame directly from source
FOIA and WRDS inputs. It does not depend on downstream outputs from
build_company_shift_share.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import time
from typing import Iterable, Optional

import duckdb as ddb
import numpy as np
import pandas as pd

try:
    from company_shift_share.config_loader import (
        DEFAULT_CONFIG_PATH,
        apply_testing_output_suffix,
        get_cfg_section,
        load_config,
    )
    from company_shift_share.institution_mapping import (
        load_revelio_school_map,
        sql_normalize_school_key,
    )
    from company_shift_share.source_exposure_data import (
        ANALYSIS_UNIVERSE_METHOD,
        DEGREE_GROUPS,
        NEW_HIRE_ORIGIN_METHOD,
        OPT_COUNT_METHOD,
        SCHOOL_BENCHMARK_METHOD,
        load_or_build_source_opt_counts,
        load_or_build_source_school_opt_benchmark,
        load_or_build_source_firm_universe,
        load_or_build_wrds_company_year_workforce_cache,
        load_or_build_wrds_school_flows_cache,
        resolve_source_exposure_paths,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import (  # type: ignore[no-redef]
        DEFAULT_CONFIG_PATH,
        apply_testing_output_suffix,
        get_cfg_section,
        load_config,
    )
    from company_shift_share.institution_mapping import (  # type: ignore[no-redef]
        load_revelio_school_map,
        sql_normalize_school_key,
    )
    from company_shift_share.source_exposure_data import (  # type: ignore[no-redef]
        ANALYSIS_UNIVERSE_METHOD,
        DEGREE_GROUPS,
        NEW_HIRE_ORIGIN_METHOD,
        OPT_COUNT_METHOD,
        SCHOOL_BENCHMARK_METHOD,
        load_or_build_source_opt_counts,
        load_or_build_source_school_opt_benchmark,
        load_or_build_source_firm_universe,
        load_or_build_wrds_company_year_workforce_cache,
        load_or_build_wrds_school_flows_cache,
        resolve_source_exposure_paths,
    )


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


@dataclass(frozen=True)
class FeaturePaths:
    company_mapping: Path
    revelio_inst_crosswalk: Path
    company_features_out: Path
    outside_negative_sample_out: Path


_FEATURE_META_SUFFIX = ".meta.json"


def _escape(path: Path) -> str:
    return str(path).replace("'", "''")


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _validate_sql_identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQL identifier: {name}")
    return name


def _resolve_path(paths_cfg: dict, key: str, *, allow_missing: bool = False) -> Path:
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"Config paths.{key} must be set.")
    root = str(Path(__file__).resolve().parents[2])
    path = Path(str(value).replace("{root}", root))
    if not allow_missing and not path.exists():
        raise FileNotFoundError(f"Required path does not exist: {path}")
    return path


def _resolve_feature_paths(cfg: dict) -> FeaturePaths:
    paths_cfg = get_cfg_section(cfg, "paths")
    return FeaturePaths(
        company_mapping=_resolve_path(paths_cfg, "revelio_company_mapping"),
        revelio_inst_crosswalk=_resolve_path(paths_cfg, "revelio_ipeds_foia_inst_crosswalk"),
        company_features_out=apply_testing_output_suffix(
            _resolve_path(paths_cfg, "company_features_out", allow_missing=True),
            cfg,
        ),
        outside_negative_sample_out=apply_testing_output_suffix(
            _resolve_path(paths_cfg, "outside_negative_sample_out", allow_missing=True),
            cfg,
        ),
    )


def validate_feature_window(feature_year_min: int, feature_year_max: int) -> None:
    if int(feature_year_min) > int(feature_year_max):
        raise ValueError(
            f"feature_year_min must be <= feature_year_max, got "
            f"{feature_year_min=} {feature_year_max=}."
        )
    if int(feature_year_max) >= 2016:
        raise ValueError(f"feature_year_max must be < 2016, got {feature_year_max}.")


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + _FEATURE_META_SUFFIX)


def _legacy_cache_compat_mode() -> bool:
    return os.getenv("EXPOSURE_EVENT_STUDY_LEGACY_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}


def _legacy_cache_compat_ignore_keys() -> set[str]:
    env = os.getenv("EXPOSURE_EVENT_STUDY_LEGACY_CACHE_IGNORE_KEYS", "").strip()
    keys = {k.strip() for k in env.split(",") if k.strip()}
    if not keys:
        keys = {"wrds_workforce_include_education_features"}
    return keys


def _metadata_compatible(meta: dict, expected: dict) -> bool:
    if not _legacy_cache_compat_mode():
        return all(meta.get(k) == v for k, v in expected.items())
    for key, expected_value in expected.items():
        if key not in meta:
            continue
        if key in _legacy_cache_compat_ignore_keys():
            continue
        if meta.get(key) != expected_value:
            return False
    return True


def _load_metadata(path: Path) -> dict:
    meta_path = _metadata_path(path)
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _write_metadata(path: Path, metadata: dict) -> None:
    meta_path = _metadata_path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def classify_opt_intensive_schools(
    components: pd.DataFrame,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    """Legacy helper kept for testing: classify schools above-window median g_kt."""
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
    """Collapse annual firm-year features into pre-period level/growth features in DuckDB."""
    validate_feature_window(year_min, year_max)
    if firm_col not in annual_df.columns or year_col not in annual_df.columns:
        raise ValueError(f"annual_df must contain '{firm_col}' and '{year_col}'.")

    available_cols = [col for col in feature_cols if col in annual_df.columns]
    con = ddb.connect()
    try:
        con.register("annual_df", annual_df)
        firm_ident = _quote_ident(firm_col)
        year_ident = _quote_ident(year_col)
        numeric_selects = ", ".join(
            f"TRY_CAST({_quote_ident(col)} AS DOUBLE) AS {_quote_ident(col)}"
            for col in available_cols
        )
        work_select = (
            f"""
            SELECT
                CAST({firm_ident} AS BIGINT) AS c,
                CAST({year_ident} AS DOUBLE) AS t
                {', ' + numeric_selects if numeric_selects else ''}
            FROM annual_df
            WHERE {firm_ident} IS NOT NULL
              AND {year_ident} IS NOT NULL
              AND TRY_CAST({year_ident} AS INTEGER) BETWEEN {int(year_min)} AND {int(year_max)}
            """
        )
        con.sql(f"CREATE OR REPLACE TEMP VIEW annual_work AS {work_select}")
        if not available_cols:
            return con.sql("SELECT DISTINCT c FROM annual_work ORDER BY c").df()

        aggregate_exprs: list[str] = []
        final_exprs: list[str] = ["firms.c"]
        for col in available_cols:
            ident = _quote_ident(col)
            level_alias = f"{col}_pre_level"
            n_alias = f"{col}_pre_n_years"
            growth_alias = f"{col}_pre_growth"
            level_missing_alias = f"{col}_pre_level_missing_ind"
            growth_missing_alias = f"{col}_pre_growth_missing_ind"
            count_expr = f"COUNT({ident})"
            sum_t_expr = f"SUM(CASE WHEN {ident} IS NOT NULL THEN t END)"
            sum_tt_expr = f"SUM(CASE WHEN {ident} IS NOT NULL THEN t * t END)"
            sum_y_expr = f"SUM({ident})"
            sum_ty_expr = f"SUM(CASE WHEN {ident} IS NOT NULL THEN t * {ident} END)"
            denom_expr = f"(({count_expr}) * ({sum_tt_expr}) - POWER(({sum_t_expr}), 2))"
            aggregate_exprs.extend(
                [
                    f"AVG({ident}) AS {_quote_ident(level_alias)}",
                    f"{count_expr} AS {_quote_ident(n_alias)}",
                    (
                        f"CASE WHEN ({count_expr}) >= {int(growth_min_obs)} "
                        f"AND ({denom_expr}) > 0 "
                        f"THEN (({count_expr}) * ({sum_ty_expr}) - ({sum_t_expr}) * ({sum_y_expr})) "
                        f"/ ({denom_expr}) ELSE NULL END AS {_quote_ident(growth_alias)}"
                    ),
                ]
            )
            final_exprs.extend(
                [
                    f'agg.{_quote_ident(level_alias)}',
                    f'COALESCE(agg.{_quote_ident(n_alias)}, 0)::BIGINT AS {_quote_ident(n_alias)}',
                    f'agg.{_quote_ident(growth_alias)}',
                    (
                        f"CASE WHEN agg.{_quote_ident(level_alias)} IS NULL THEN 1 ELSE 0 END::TINYINT "
                        f"AS {_quote_ident(level_missing_alias)}"
                    ),
                    (
                        f"CASE WHEN agg.{_quote_ident(growth_alias)} IS NULL THEN 1 ELSE 0 END::TINYINT "
                        f"AS {_quote_ident(growth_missing_alias)}"
                    ),
                ]
            )
        sql = f"""
        WITH firms AS (
            SELECT DISTINCT c FROM annual_work
        ),
        agg AS (
            SELECT
                c,
                {", ".join(aggregate_exprs)}
            FROM annual_work
            GROUP BY 1
        )
        SELECT
            {", ".join(final_exprs)}
        FROM firms
        LEFT JOIN agg USING (c)
        ORDER BY firms.c
        """
        return con.sql(sql).df()
    finally:
        con.close()


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
    labels = ["lt10", "10_49", "50_249", "250_999", "1000p"]
    out = np.select(conditions, labels, default="unknown")
    return pd.Series(out, index=n_users.index, dtype="string")


def _load_company_meta_subset(
    path: Path,
    *,
    selected_firms: Optional[pd.DataFrame] = None,
    min_n_users: Optional[int] = None,
    eligible_only: bool = False,
) -> pd.DataFrame:
    con = ddb.connect()
    try:
        if selected_firms is not None:
            con.register("selected_firms_df", selected_firms[["c"]].drop_duplicates())
            con.sql(
                "CREATE OR REPLACE TEMP VIEW selected_firms AS "
                "SELECT CAST(c AS BIGINT) AS c FROM selected_firms_df"
            )
            join_sql = "JOIN selected_firms sf ON CAST(cm.rcid AS BIGINT) = sf.c"
        else:
            join_sql = ""

        filters = ["cm.rcid IS NOT NULL"]
        if min_n_users is not None:
            filters.append(f"COALESCE(cm.n_users, 0) >= {int(min_n_users)}")
        if eligible_only:
            filters.extend(
                [
                    "(cm.top_state IS NOT NULL OR cm.hq_state IS NOT NULL)",
                    "cm.naics_code IS NOT NULL",
                ]
            )
        sql = f"""
        SELECT
            CAST(cm.rcid AS BIGINT) AS c,
            CAST(cm.n_users AS DOUBLE) AS n_users,
            CAST(cm.top_state AS VARCHAR) AS top_state,
            CAST(cm.top_metro_area AS VARCHAR) AS top_metro_area,
            CAST(cm.hq_state AS VARCHAR) AS hq_state,
            CAST(cm.hq_region AS VARCHAR) AS hq_region,
            CAST(cm.naics_code AS VARCHAR) AS naics_code,
            CAST(cm.year_founded AS DOUBLE) AS year_founded
        FROM read_parquet('{_escape(path)}') cm
        {join_sql}
        WHERE {" AND ".join(filters)}
        """
        meta = con.sql(sql).df()
    finally:
        con.close()

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
        sampled = group.sample(n=min(n_target, len(group)), random_state=int(seed) + idx)
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
    analysis_ids = (
        pd.Series(pd.to_numeric(analysis_firms, errors="coerce")).dropna().astype(int).unique()
    )
    preferred_ids = set(
        pd.Series(pd.to_numeric(preferred_rcids, errors="coerce")).dropna().astype(int).tolist()
    )
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
        coarse_targets = coarse_targets.merge(
            already, on=["size_bucket", "company_state_feature"], how="left"
        )
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
        chosen_frames.append(remaining.sample(n=min(need, len(remaining)), random_state=seed + 3_000))

    sampled = pd.concat(chosen_frames, ignore_index=True) if chosen_frames else eligible.iloc[0:0].copy()
    sampled = sampled.drop_duplicates(subset=["c"], keep="first").reset_index(drop=True)
    sampled["outside_negative_candidate"] = 1
    return sampled


def _prepare_feature_firms(
    paths: FeaturePaths,
    *,
    cfg_full: dict,
    ratio: float,
    seed: int,
    min_n_users: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_firms = load_preferred_rcids(cfg=cfg_full)
    analysis_firms["in_analysis_universe"] = 1
    analysis_firms["outside_negative_candidate"] = 0

    analysis_meta = _load_company_meta_subset(
        paths.company_mapping,
        selected_firms=analysis_firms[["c"]],
    )
    candidate_meta = _load_company_meta_subset(
        paths.company_mapping,
        min_n_users=min_n_users,
        eligible_only=True,
    )
    outside = sample_outside_negative_firms(
        candidate_meta,
        analysis_firms["c"],
        analysis_firms["c"],
        ratio=ratio,
        seed=seed,
        min_n_users=min_n_users,
    )
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

    selected_meta = _load_company_meta_subset(
        paths.company_mapping,
        selected_firms=firms[["c"]],
    )
    return firms, outside, selected_meta


def _aggregate_school_features(
    school_flows: pd.DataFrame,
    school_benchmark: pd.DataFrame,
    school_map: pd.DataFrame,
) -> pd.DataFrame:
    if school_flows.empty:
        return pd.DataFrame(columns=["c", "t"])
    con = ddb.connect()
    try:
        con.register("school_flows_df", school_flows)
        con.register("school_map_df", school_map)
        con.register("school_benchmark_df", school_benchmark)
        bench_exprs = []
        for degree in DEGREE_GROUPS:
            col = f"opt_intensive_{degree}"
            if col in school_benchmark.columns:
                bench_exprs.append(f"COALESCE(TRY_CAST({_quote_ident(col)} AS INTEGER), 0) AS {_quote_ident(col)}")
            else:
                bench_exprs.append(f"0 AS {_quote_ident(col)}")
        grouped_degree_exprs: list[str] = []
        final_degree_exprs: list[str] = []
        for degree in DEGREE_GROUPS:
            grouped_degree_exprs.extend(
                [
                    (
                        f"SUM(CASE WHEN opt_intensive_{degree} = 1 "
                        f"THEN COALESCE(n_transitions, 0.0) ELSE 0.0 END) "
                        f"AS intensive_transitions_{degree}"
                    ),
                    (
                        f"SUM(CASE WHEN opt_intensive_{degree} = 1 "
                        f"THEN COALESCE(n_emp, 0.0) ELSE 0.0 END) "
                        f"AS intensive_emp_{degree}"
                    ),
                ]
            )
            final_degree_exprs.extend(
                [
                    (
                        f"CASE WHEN total_new_hires > 0 "
                        f"THEN intensive_transitions_{degree} / total_new_hires "
                        f"ELSE NULL END AS school_opt_share_new_hire_{degree}_annual"
                    ),
                    (
                        f"CASE WHEN total_emp > 0 "
                        f"THEN intensive_emp_{degree} / total_emp "
                        f"ELSE NULL END AS school_opt_share_tenured_{degree}_annual"
                    ),
                ]
            )
        sql = f"""
        WITH flows AS (
            SELECT
                CAST(c AS BIGINT) AS c,
                CAST(t AS INTEGER) AS t,
                CAST(university_raw AS VARCHAR) AS university_raw,
                TRY_CAST(n_transitions AS DOUBLE) AS n_transitions,
                TRY_CAST(n_emp AS DOUBLE) AS n_emp,
                TRY_CAST(total_new_hires AS DOUBLE) AS total_new_hires,
                NULLIF({sql_normalize_school_key('university_raw')}, '') AS university_raw_key
            FROM school_flows_df
        ),
        school_map AS (
            SELECT
                CAST(university_raw_key AS VARCHAR) AS university_raw_key,
                CAST(unitid AS VARCHAR) AS unitid
            FROM school_map_df
        ),
        school_benchmark AS (
            SELECT
                CAST(unitid AS VARCHAR) AS unitid,
                {", ".join(bench_exprs)}
            FROM school_benchmark_df
        ),
        mapped AS (
            SELECT
                f.c,
                f.t,
                f.n_transitions,
                f.n_emp,
                f.total_new_hires,
                f.university_raw_key,
                sm.unitid,
                COALESCE(sm.unitid, f.university_raw_key) AS school_key,
                {", ".join(f"sb.opt_intensive_{degree}" for degree in DEGREE_GROUPS)}
            FROM flows AS f
            LEFT JOIN school_map AS sm
              ON f.university_raw_key = sm.university_raw_key
            LEFT JOIN school_benchmark AS sb
              ON sm.unitid = sb.unitid
        ),
        grouped AS (
            SELECT
                c,
                t,
                MAX(total_new_hires) AS total_new_hires,
                SUM(COALESCE(n_emp, 0.0)) AS total_emp,
                COUNT(DISTINCT CASE WHEN COALESCE(n_transitions, 0.0) > 0 THEN school_key END) AS n_schools_new_hire_annual,
                COUNT(DISTINCT CASE WHEN COALESCE(n_emp, 0.0) > 0 THEN school_key END) AS n_schools_tenured_annual,
                {", ".join(grouped_degree_exprs)}
            FROM mapped
            GROUP BY 1, 2
        )
        SELECT
            c,
            t,
            total_new_hires,
            total_emp,
            CASE WHEN total_new_hires > 0 THEN intensive_transitions_masters / total_new_hires ELSE NULL END AS school_opt_share_new_hire_annual,
            CASE WHEN total_emp > 0 THEN intensive_emp_masters / total_emp ELSE NULL END AS school_opt_share_tenured_annual,
            n_schools_new_hire_annual,
            n_schools_tenured_annual,
            {", ".join(final_degree_exprs)}
        FROM grouped
        ORDER BY c, t
        """
        return con.sql(sql).df()
    finally:
        con.close()


def _build_static_feature_frame(company_meta: pd.DataFrame, firms: pd.DataFrame, *, feature_year_max: int) -> pd.DataFrame:
    company_age_ref_year = int(feature_year_max) + 1
    con = ddb.connect()
    try:
        con.register("company_meta_df", company_meta)
        con.register("firms_df", firms)
        sql = f"""
        WITH firms AS (
            SELECT
                CAST(c AS BIGINT) AS c,
                TRY_CAST(in_analysis_universe AS BIGINT) AS in_analysis_universe,
                TRY_CAST(preferred_rcid_source AS BIGINT) AS preferred_rcid_source,
                TRY_CAST(outside_negative_candidate AS BIGINT) AS outside_negative_candidate
            FROM firms_df
        ),
        meta AS (
            SELECT
                CAST(c AS BIGINT) AS c,
                CAST(naics2 AS VARCHAR) AS naics2,
                CAST(naics4 AS VARCHAR) AS naics4,
                CAST(company_state_feature AS VARCHAR) AS company_state_feature,
                CAST(company_metro_feature AS VARCHAR) AS company_metro_feature,
                CAST(company_hq_region AS VARCHAR) AS company_hq_region,
                TRY_CAST(year_founded AS DOUBLE) AS year_founded,
                TRY_CAST(n_users AS DOUBLE) AS n_users
            FROM company_meta_df
        )
        SELECT
            f.c,
            COALESCE(f.in_analysis_universe, 0) AS in_analysis_universe,
            COALESCE(f.preferred_rcid_source, 0) AS preferred_rcid_source,
            COALESCE(f.outside_negative_candidate, 0) AS outside_negative_candidate,
            m.naics2,
            m.naics4,
            m.company_state_feature,
            m.company_metro_feature,
            m.company_hq_region,
            CASE
                WHEN m.year_founded IS NOT NULL
                THEN GREATEST(0.0, {company_age_ref_year}.0 - m.year_founded)
                ELSE NULL
            END AS company_age_feature,
            CASE
                WHEN m.n_users IS NOT NULL THEN LN(1.0 + m.n_users)
                ELSE NULL
            END AS company_n_users_log1p
        FROM firms AS f
        LEFT JOIN meta AS m
          ON f.c = m.c
        ORDER BY f.c
        """
        return con.sql(sql).df()
    finally:
        con.close()


def _build_annual_feature_frame(
    wrds_annual: pd.DataFrame,
    school_annual: pd.DataFrame,
    opt_counts: pd.DataFrame,
    *,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    wrds_available = set(wrds_annual.columns)
    school_available = set(school_annual.columns)
    opt_available = set(opt_counts.columns)

    def _maybe_double(available: set[str], src: str, alias: str) -> str:
        if src in available:
            return f"TRY_CAST({_quote_ident(src)} AS DOUBLE) AS {_quote_ident(alias)}"
        return f"NULL::DOUBLE AS {_quote_ident(alias)}"

    wrds_fields = [
        ("total_headcount_wrds_annual", "total_headcount_annual"),
        ("total_headcount_wrds_annual", "firm_size_annual"),
        ("total_headcount_foreign_weighted_annual", "total_headcount_foreign_weighted_annual"),
        ("total_headcount_native_weighted_annual", "total_headcount_native_weighted_annual"),
        ("total_headcount_foreign_hard_annual", "total_headcount_foreign_hard_annual"),
        ("total_headcount_native_hard_annual", "total_headcount_native_hard_annual"),
        ("long_term_headcount_wrds_annual", "long_term_headcount_annual"),
        ("n_new_hires_wrds_annual", "n_new_hires_annual"),
        ("n_new_hires_foreign_weighted_annual", "n_new_hires_foreign_weighted_annual"),
        ("n_new_hires_native_weighted_annual", "n_new_hires_native_weighted_annual"),
        ("n_new_hires_foreign_hard_annual", "n_new_hires_foreign_hard_annual"),
        ("n_new_hires_native_hard_annual", "n_new_hires_native_hard_annual"),
        ("salary_mean_annual", "salary_mean_annual"),
        ("salary_var_annual", "salary_var_annual"),
        ("total_comp_mean_annual", "total_comp_mean_annual"),
        ("total_comp_var_annual", "total_comp_var_annual"),
        ("compensation_missing_share_annual", "compensation_missing_share_annual"),
        ("nonus_educ_share_annual", "nonus_educ_share_annual"),
        ("age_share_lt30_annual", "age_share_lt30_annual"),
        ("age_share_30_39_annual", "age_share_30_39_annual"),
        ("age_share_40_49_annual", "age_share_40_49_annual"),
        ("age_share_50_59_annual", "age_share_50_59_annual"),
        ("age_share_60p_annual", "age_share_60p_annual"),
        ("female_share_annual", "female_share_annual"),
        ("race_share_white_annual", "race_share_white_annual"),
        ("race_share_black_annual", "race_share_black_annual"),
        ("race_share_api_annual", "race_share_api_annual"),
        ("race_share_api_annual", "race_share_asian_annual"),
        ("race_share_hispanic_annual", "race_share_hispanic_annual"),
        ("race_share_native_annual", "race_share_native_annual"),
        ("race_share_multiple_annual", "race_share_multiple_annual"),
        ("seniority_mean_annual", "seniority_mean_annual"),
        ("avg_tenure_years_annual", "avg_tenure_years_annual"),
        ("occ_share_mgmt_annual", "occ_share_mgmt_annual"),
        ("occ_share_business_finance_annual", "occ_share_business_finance_annual"),
        ("occ_share_computing_math_annual", "occ_share_computing_math_annual"),
        ("occ_share_engineering_annual", "occ_share_engineering_annual"),
        ("occ_share_science_annual", "occ_share_science_annual"),
        ("occ_share_community_legal_education_annual", "occ_share_community_legal_education_annual"),
        ("occ_share_arts_media_annual", "occ_share_arts_media_annual"),
        ("occ_share_healthcare_annual", "occ_share_healthcare_annual"),
        ("occ_share_sales_office_annual", "occ_share_sales_office_annual"),
        ("occ_share_manual_annual", "occ_share_manual_annual"),
    ]
    school_fields = [
        ("total_new_hires", "total_new_hires"),
        ("total_emp", "total_emp"),
        ("school_opt_share_new_hire_annual", "school_opt_share_new_hire_annual"),
        ("school_opt_share_tenured_annual", "school_opt_share_tenured_annual"),
        ("n_schools_new_hire_annual", "n_schools_new_hire_annual"),
        ("n_schools_tenured_annual", "n_schools_tenured_annual"),
        ("school_opt_share_new_hire_bachelors_annual", "school_opt_share_new_hire_bachelors_annual"),
        ("school_opt_share_new_hire_masters_annual", "school_opt_share_new_hire_masters_annual"),
        ("school_opt_share_new_hire_phd_annual", "school_opt_share_new_hire_phd_annual"),
        ("school_opt_share_tenured_bachelors_annual", "school_opt_share_tenured_bachelors_annual"),
        ("school_opt_share_tenured_masters_annual", "school_opt_share_tenured_masters_annual"),
        ("school_opt_share_tenured_phd_annual", "school_opt_share_tenured_phd_annual"),
    ]
    opt_fields = [
        ("bachelors_opt_hire_count_annual", "bachelors_opt_hire_count_annual"),
        ("masters_opt_hire_count_annual", "masters_opt_hire_count_annual"),
        ("phd_opt_hire_count_annual", "phd_opt_hire_count_annual"),
        ("any_opt_hire_count_annual", "any_opt_hire_count_annual"),
    ]
    wrds_exprs = ",\n                ".join(
        _maybe_double(wrds_available, src, alias) for src, alias in wrds_fields
    )
    school_exprs = ",\n                ".join(
        _maybe_double(school_available, src, alias) for src, alias in school_fields
    )
    opt_exprs = ",\n                ".join(
        _maybe_double(opt_available, src, alias) for src, alias in opt_fields
    )

    con = ddb.connect()
    try:
        con.register("wrds_annual_df", wrds_annual)
        con.register("school_annual_df", school_annual)
        con.register("opt_counts_df", opt_counts)
        sql = f"""
        WITH wrds AS (
            SELECT
                CAST(c AS BIGINT) AS c,
                CAST(t AS INTEGER) AS t,
                {wrds_exprs}
            FROM wrds_annual_df
            WHERE TRY_CAST(t AS INTEGER) BETWEEN {int(year_min)} AND {int(year_max)}
        ),
        school AS (
            SELECT
                CAST(c AS BIGINT) AS c,
                CAST(t AS INTEGER) AS t,
                {school_exprs}
            FROM school_annual_df
        ),
        opt AS (
            SELECT
                CAST(c AS BIGINT) AS c,
                CAST(t AS INTEGER) AS t,
                {opt_exprs}
            FROM opt_counts_df
        ),
        annual AS (
            SELECT
                COALESCE(w.c, s.c, o.c) AS c,
                COALESCE(w.t, s.t, o.t) AS t,
                w.total_headcount_annual,
                w.firm_size_annual,
                w.total_headcount_foreign_weighted_annual,
                w.total_headcount_native_weighted_annual,
                w.total_headcount_foreign_hard_annual,
                w.total_headcount_native_hard_annual,
                w.long_term_headcount_annual,
                w.n_new_hires_annual,
                w.n_new_hires_foreign_weighted_annual,
                w.n_new_hires_native_weighted_annual,
                w.n_new_hires_foreign_hard_annual,
                w.n_new_hires_native_hard_annual,
                w.salary_mean_annual,
                w.salary_var_annual,
                w.total_comp_mean_annual,
                w.total_comp_var_annual,
                w.compensation_missing_share_annual,
                w.nonus_educ_share_annual,
                w.age_share_lt30_annual,
                w.age_share_30_39_annual,
                w.age_share_40_49_annual,
                w.age_share_50_59_annual,
                w.age_share_60p_annual,
                w.female_share_annual,
                w.race_share_white_annual,
                w.race_share_black_annual,
                w.race_share_api_annual,
                w.race_share_asian_annual,
                w.race_share_hispanic_annual,
                w.race_share_native_annual,
                w.race_share_multiple_annual,
                w.seniority_mean_annual,
                w.avg_tenure_years_annual,
                w.occ_share_mgmt_annual,
                w.occ_share_business_finance_annual,
                w.occ_share_computing_math_annual,
                w.occ_share_engineering_annual,
                w.occ_share_science_annual,
                w.occ_share_community_legal_education_annual,
                w.occ_share_arts_media_annual,
                w.occ_share_healthcare_annual,
                w.occ_share_sales_office_annual,
                w.occ_share_manual_annual,
                s.total_new_hires,
                s.total_emp,
                s.school_opt_share_new_hire_annual,
                s.school_opt_share_tenured_annual,
                s.n_schools_new_hire_annual,
                s.n_schools_tenured_annual,
                s.school_opt_share_new_hire_bachelors_annual,
                s.school_opt_share_new_hire_masters_annual,
                s.school_opt_share_new_hire_phd_annual,
                s.school_opt_share_tenured_bachelors_annual,
                s.school_opt_share_tenured_masters_annual,
                s.school_opt_share_tenured_phd_annual,
                o.bachelors_opt_hire_count_annual,
                o.masters_opt_hire_count_annual,
                o.phd_opt_hire_count_annual,
                o.any_opt_hire_count_annual
            FROM wrds AS w
            FULL OUTER JOIN school AS s
              ON w.c = s.c AND w.t = s.t
            FULL OUTER JOIN opt AS o
              ON COALESCE(w.c, s.c) = o.c AND COALESCE(w.t, s.t) = o.t
        ),
        annual_filled AS (
            SELECT
                a.c,
                a.t,
                a.total_headcount_annual,
                a.firm_size_annual,
                a.total_headcount_foreign_weighted_annual,
                a.total_headcount_native_weighted_annual,
                a.total_headcount_foreign_hard_annual,
                a.total_headcount_native_hard_annual,
                a.long_term_headcount_annual,
                a.n_new_hires_annual,
                a.n_new_hires_foreign_weighted_annual,
                a.n_new_hires_native_weighted_annual,
                a.n_new_hires_foreign_hard_annual,
                a.n_new_hires_native_hard_annual,
                a.salary_mean_annual,
                a.salary_var_annual,
                a.total_comp_mean_annual,
                a.total_comp_var_annual,
                a.compensation_missing_share_annual,
                a.nonus_educ_share_annual,
                a.age_share_lt30_annual,
                a.age_share_30_39_annual,
                a.age_share_40_49_annual,
                a.age_share_50_59_annual,
                a.age_share_60p_annual,
                a.female_share_annual,
                a.race_share_white_annual,
                a.race_share_black_annual,
                a.race_share_api_annual,
                a.race_share_asian_annual,
                a.race_share_hispanic_annual,
                a.race_share_native_annual,
                a.race_share_multiple_annual,
                a.seniority_mean_annual,
                a.avg_tenure_years_annual,
                a.occ_share_mgmt_annual,
                a.occ_share_business_finance_annual,
                a.occ_share_computing_math_annual,
                a.occ_share_engineering_annual,
                a.occ_share_science_annual,
                a.occ_share_community_legal_education_annual,
                a.occ_share_arts_media_annual,
                a.occ_share_healthcare_annual,
                a.occ_share_sales_office_annual,
                a.occ_share_manual_annual,
                a.total_new_hires,
                a.total_emp,
                a.school_opt_share_new_hire_annual,
                a.school_opt_share_tenured_annual,
                a.n_schools_new_hire_annual,
                a.n_schools_tenured_annual,
                a.school_opt_share_new_hire_bachelors_annual,
                a.school_opt_share_new_hire_masters_annual,
                a.school_opt_share_new_hire_phd_annual,
                a.school_opt_share_tenured_bachelors_annual,
                a.school_opt_share_tenured_masters_annual,
                a.school_opt_share_tenured_phd_annual,
                a.bachelors_opt_hire_count_annual,
                a.masters_opt_hire_count_annual,
                a.phd_opt_hire_count_annual,
                a.any_opt_hire_count_annual,
                COALESCE(a.total_headcount_annual, a.total_emp) AS total_headcount_filled,
                COALESCE(a.firm_size_annual, COALESCE(a.total_headcount_annual, a.total_emp)) AS firm_size_filled,
                COALESCE(a.n_new_hires_annual, a.total_new_hires) AS n_new_hires_filled
            FROM annual AS a
        ),
        annual_ready AS (
            SELECT
                a.c,
                a.t,
                a.total_headcount_filled AS total_headcount_annual,
                a.firm_size_filled AS firm_size_annual,
                a.total_headcount_foreign_weighted_annual,
                a.total_headcount_native_weighted_annual,
                a.total_headcount_foreign_hard_annual,
                a.total_headcount_native_hard_annual,
                a.long_term_headcount_annual,
                a.n_new_hires_filled AS n_new_hires_annual,
                a.n_new_hires_foreign_weighted_annual,
                a.n_new_hires_native_weighted_annual,
                a.n_new_hires_foreign_hard_annual,
                a.n_new_hires_native_hard_annual,
                a.salary_mean_annual,
                a.salary_var_annual,
                a.total_comp_mean_annual,
                a.total_comp_var_annual,
                a.compensation_missing_share_annual,
                a.nonus_educ_share_annual,
                a.age_share_lt30_annual,
                a.age_share_30_39_annual,
                a.age_share_40_49_annual,
                a.age_share_50_59_annual,
                a.age_share_60p_annual,
                a.female_share_annual,
                a.race_share_white_annual,
                a.race_share_black_annual,
                a.race_share_api_annual,
                a.race_share_asian_annual,
                a.race_share_hispanic_annual,
                a.race_share_native_annual,
                a.race_share_multiple_annual,
                a.seniority_mean_annual,
                a.avg_tenure_years_annual,
                a.occ_share_mgmt_annual,
                a.occ_share_business_finance_annual,
                a.occ_share_computing_math_annual,
                a.occ_share_engineering_annual,
                a.occ_share_science_annual,
                a.occ_share_community_legal_education_annual,
                a.occ_share_arts_media_annual,
                a.occ_share_healthcare_annual,
                a.occ_share_sales_office_annual,
                a.occ_share_manual_annual,
                a.total_new_hires,
                a.total_emp,
                a.school_opt_share_new_hire_annual,
                a.school_opt_share_tenured_annual,
                a.n_schools_new_hire_annual,
                a.n_schools_tenured_annual,
                a.school_opt_share_new_hire_bachelors_annual,
                a.school_opt_share_new_hire_masters_annual,
                a.school_opt_share_new_hire_phd_annual,
                a.school_opt_share_tenured_bachelors_annual,
                a.school_opt_share_tenured_masters_annual,
                a.school_opt_share_tenured_phd_annual,
                COALESCE(a.bachelors_opt_hire_count_annual, 0.0) AS bachelors_opt_hire_count_annual,
                COALESCE(a.masters_opt_hire_count_annual, 0.0) AS masters_opt_hire_count_annual,
                COALESCE(a.phd_opt_hire_count_annual, 0.0) AS phd_opt_hire_count_annual,
                COALESCE(a.any_opt_hire_count_annual, 0.0) AS any_opt_hire_count_annual
            FROM annual_filled AS a
        )
        SELECT
            a.*,
            CASE
                WHEN a.n_new_hires_annual > 0
                THEN a.bachelors_opt_hire_count_annual / a.n_new_hires_annual
                ELSE NULL
            END AS bachelors_opt_hire_rate_annual,
            CASE
                WHEN a.n_new_hires_annual > 0
                THEN a.masters_opt_hire_count_annual / a.n_new_hires_annual
                ELSE NULL
            END AS masters_opt_hire_rate_annual,
            CASE
                WHEN a.n_new_hires_annual > 0
                THEN a.phd_opt_hire_count_annual / a.n_new_hires_annual
                ELSE NULL
            END AS phd_opt_hire_rate_annual,
            CASE
                WHEN a.n_new_hires_annual > 0
                THEN a.any_opt_hire_count_annual / a.n_new_hires_annual
                ELSE NULL
            END AS any_opt_hire_rate_annual,
            a.masters_opt_hire_count_annual AS opt_hire_count_annual,
            CASE
                WHEN a.n_new_hires_annual > 0
                THEN a.masters_opt_hire_count_annual / a.n_new_hires_annual
                ELSE NULL
            END AS opt_hire_rate_annual
        FROM annual_ready AS a
        ORDER BY a.c, a.t
        """
        annual_df = con.sql(sql).df()
    finally:
        con.close()
    annual_df = annual_df.loc[:, ~annual_df.columns.duplicated()].copy()
    return annual_df


def _merge_feature_frames(static: pd.DataFrame, summarized: pd.DataFrame) -> pd.DataFrame:
    con = ddb.connect()
    try:
        con.register("static_df", static)
        con.register("summarized_df", summarized)
        return con.sql(
            """
            SELECT
                s.*,
                sm.* EXCLUDE (c)
            FROM static_df AS s
            LEFT JOIN summarized_df AS sm
              ON s.c = sm.c
            ORDER BY s.c
            """
        ).df()
    finally:
        con.close()


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
    source_paths = resolve_source_exposure_paths(cfg_full)
    feature_cfg = get_cfg_section(cfg_full, "revelio_company_features")
    year_min = int(feature_year_min if feature_year_min is not None else feature_cfg.get("feature_year_min", 2010))
    year_max = int(feature_year_max if feature_year_max is not None else feature_cfg.get("feature_year_max", 2015))
    validate_feature_window(year_min, year_max)
    school_map, school_map_meta = load_revelio_school_map(
        legacy_crosswalk=source_paths.revelio_inst_crosswalk,
        deterministic_triple_map=source_paths.revelio_inst_deterministic_map,
        ref_inst_catalog=source_paths.revelio_ref_inst_catalog,
    )
    firms, outside, selected_meta, universe_meta = load_or_build_source_firm_universe(
        cfg=cfg_full,
        force_rebuild=force_rebuild,
    )

    out_path = paths.company_features_out
    metadata = {
        "feature_year_min": year_min,
        "feature_year_max": year_max,
        "analysis_universe_method": ANALYSIS_UNIVERSE_METHOD,
        "opt_count_method": OPT_COUNT_METHOD,
        "school_benchmark_method": SCHOOL_BENCHMARK_METHOD,
        "degree_groups": list(DEGREE_GROUPS),
        "outside_negative_ratio": float(feature_cfg.get("outside_negative_ratio", 2.0)),
        "outside_negative_seed": int(feature_cfg.get("outside_negative_seed", 42)),
        "outside_negative_min_n_users": int(feature_cfg.get("outside_negative_min_n_users", 10)),
        "min_position_days": int(feature_cfg.get("min_position_days", 365)),
        "tenure_min_days": int(feature_cfg.get("tenure_min_days", 365)),
        "wrds_workforce_include_education_features": bool(
            feature_cfg.get("wrds_workforce_include_education_features", True)
        ),
        "new_hire_origin_method": NEW_HIRE_ORIGIN_METHOD,
        "revelio_school_map_method": school_map_meta["mapping_method"],
        "revelio_school_map_paths": {
            "deterministic_triple_map": school_map_meta.get("deterministic_triple_map"),
            "ref_inst_catalog": school_map_meta.get("ref_inst_catalog"),
            "legacy_crosswalk": school_map_meta.get("legacy_crosswalk"),
        },
        "shared_universe_meta": universe_meta,
    }
    if out_path.exists() and not force_rebuild:
        current_meta = _load_metadata(out_path)
        if (
            current_meta.get("feature_year_min") == metadata["feature_year_min"]
            and current_meta.get("feature_year_max") == metadata["feature_year_max"]
            and current_meta.get("analysis_universe_method") == metadata["analysis_universe_method"]
            and current_meta.get("opt_count_method") == metadata["opt_count_method"]
            and current_meta.get("school_benchmark_method") == metadata["school_benchmark_method"]
            and current_meta.get("outside_negative_ratio") == metadata["outside_negative_ratio"]
            and current_meta.get("outside_negative_seed") == metadata["outside_negative_seed"]
            and current_meta.get("outside_negative_min_n_users") == metadata["outside_negative_min_n_users"]
            and current_meta.get("min_position_days") == metadata["min_position_days"]
            and current_meta.get("tenure_min_days") == metadata["tenure_min_days"]
            and current_meta.get("wrds_workforce_include_education_features")
            == metadata["wrds_workforce_include_education_features"]
            and current_meta.get("new_hire_origin_method") == metadata["new_hire_origin_method"]
            and current_meta.get("revelio_school_map_method") == metadata["revelio_school_map_method"]
            and current_meta.get("revelio_school_map_paths") == metadata["revelio_school_map_paths"]
            and current_meta.get("shared_universe_meta") == metadata["shared_universe_meta"]
        ):
            print(f"[company_features] Reusing cached features from {out_path}")
            return pd.read_parquet(out_path)

    t0 = time.time()
    print(f"[company_features] Building source-based firm features for {year_min}–{year_max}")

    school_benchmark, school_meta = load_or_build_source_school_opt_benchmark(
        cfg=cfg_full,
        year_min=year_min,
        year_max=year_max,
        force_rebuild=force_rebuild,
    )
    print(f"[company_features] School benchmark metadata: {school_meta}")
    print(f"[company_features] Revelio school map metadata: {school_map_meta}")

    opt_counts, _ = load_or_build_source_opt_counts(
        cfg=cfg_full,
        year_min=year_min,
        year_max=year_max,
        force_rebuild=force_rebuild,
    )
    opt_counts = opt_counts.rename(
        columns={
            "bachelors_opt_hires_correction_aware": "bachelors_opt_hire_count_annual",
            "masters_opt_hires_correction_aware": "masters_opt_hire_count_annual",
            "phd_opt_hires_correction_aware": "phd_opt_hire_count_annual",
            "any_opt_hires_correction_aware": "any_opt_hire_count_annual",
        }
    )

    wrds_annual, workforce_meta = load_or_build_wrds_company_year_workforce_cache(
        cfg=cfg_full,
        year_min=year_min,
        year_max=year_max,
        force_rebuild=force_rebuild,
    )

    school_flows, school_flows_meta = load_or_build_wrds_school_flows_cache(
        cfg=cfg_full,
        year_min=year_min,
        year_max=year_max,
        force_rebuild=force_rebuild,
    )
    print("[company_features] Aggregating school flows in DuckDB")
    school_annual = _aggregate_school_features(school_flows, school_benchmark, school_map)
    print("[company_features] Building annual feature panel in DuckDB")
    annual = _build_annual_feature_frame(
        wrds_annual,
        school_annual,
        opt_counts,
        year_min=year_min,
        year_max=year_max,
    )

    feature_cols = [
        "opt_hire_count_annual",
        "opt_hire_rate_annual",
        "school_opt_share_new_hire_annual",
        "school_opt_share_tenured_annual",
        "bachelors_opt_hire_count_annual",
        "masters_opt_hire_count_annual",
        "phd_opt_hire_count_annual",
        "any_opt_hire_count_annual",
        "bachelors_opt_hire_rate_annual",
        "masters_opt_hire_rate_annual",
        "phd_opt_hire_rate_annual",
        "any_opt_hire_rate_annual",
        "school_opt_share_new_hire_bachelors_annual",
        "school_opt_share_new_hire_masters_annual",
        "school_opt_share_new_hire_phd_annual",
        "school_opt_share_tenured_bachelors_annual",
        "school_opt_share_tenured_masters_annual",
        "school_opt_share_tenured_phd_annual",
        "n_schools_new_hire_annual",
        "n_schools_tenured_annual",
        "firm_size_annual",
        "total_headcount_annual",
        "total_headcount_foreign_weighted_annual",
        "total_headcount_native_weighted_annual",
        "long_term_headcount_annual",
        "n_new_hires_annual",
        "n_new_hires_foreign_weighted_annual",
        "n_new_hires_native_weighted_annual",
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
        "female_share_annual",
        "race_share_white_annual",
        "race_share_black_annual",
        "race_share_api_annual",
        "race_share_asian_annual",
        "race_share_hispanic_annual",
        "race_share_native_annual",
        "race_share_multiple_annual",
        "seniority_mean_annual",
        "avg_tenure_years_annual",
        "occ_share_mgmt_annual",
        "occ_share_business_finance_annual",
        "occ_share_computing_math_annual",
        "occ_share_engineering_annual",
        "occ_share_science_annual",
        "occ_share_community_legal_education_annual",
        "occ_share_arts_media_annual",
        "occ_share_healthcare_annual",
        "occ_share_sales_office_annual",
        "occ_share_manual_annual",
    ]
    print("[company_features] Summarizing pre-period features in DuckDB")
    summarized = summarize_pre_period_features(
        annual,
        feature_cols,
        year_min=year_min,
        year_max=year_max,
    )
    print("[company_features] Building static feature frame in DuckDB")
    static = _build_static_feature_frame(selected_meta, firms, feature_year_max=year_max)
    print("[company_features] Merging static and summarized features in DuckDB")
    feature_df = _merge_feature_frames(static, summarized)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(out_path, index=False)
    print(f"[company_features] Wrote {out_path}")

    metadata.update(
        {
            "n_feature_firms": int(len(feature_df)),
            "n_analysis_firms": int(feature_df["in_analysis_universe"].fillna(0).sum()),
            "n_preferred_source_firms": int(feature_df["preferred_rcid_source"].fillna(0).sum()),
            "n_outside_negative_candidates": int(feature_df["outside_negative_candidate"].fillna(0).sum()),
            "shared_universe_meta": universe_meta,
            "workforce_cache_meta": workforce_meta,
            "school_flows_cache_meta": school_flows_meta,
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
    source_paths = resolve_source_exposure_paths(cfg_full)
    feature_cfg = get_cfg_section(cfg_full, "revelio_company_features")
    year_min = int(feature_year_min if feature_year_min is not None else feature_cfg.get("feature_year_min", 2010))
    year_max = int(feature_year_max if feature_year_max is not None else feature_cfg.get("feature_year_max", 2015))
    validate_feature_window(year_min, year_max)
    _, _, _, universe_meta = load_or_build_source_firm_universe(
        cfg=cfg_full,
        force_rebuild=force_rebuild,
    )
    expected_meta = {
        "feature_year_min": year_min,
        "feature_year_max": year_max,
        "analysis_universe_method": ANALYSIS_UNIVERSE_METHOD,
        "opt_count_method": OPT_COUNT_METHOD,
        "school_benchmark_method": SCHOOL_BENCHMARK_METHOD,
        "degree_groups": list(DEGREE_GROUPS),
        "outside_negative_ratio": float(feature_cfg.get("outside_negative_ratio", 2.0)),
        "outside_negative_seed": int(feature_cfg.get("outside_negative_seed", 42)),
        "outside_negative_min_n_users": int(feature_cfg.get("outside_negative_min_n_users", 10)),
        "min_position_days": int(feature_cfg.get("min_position_days", 365)),
        "tenure_min_days": int(feature_cfg.get("tenure_min_days", 365)),
        "wrds_workforce_include_education_features": bool(
            feature_cfg.get("wrds_workforce_include_education_features", True)
        ),
        "new_hire_origin_method": NEW_HIRE_ORIGIN_METHOD,
    }
    _, school_map_meta = load_revelio_school_map(
        legacy_crosswalk=source_paths.revelio_inst_crosswalk,
        deterministic_triple_map=source_paths.revelio_inst_deterministic_map,
        ref_inst_catalog=source_paths.revelio_ref_inst_catalog,
    )
    expected_meta["revelio_school_map_method"] = school_map_meta["mapping_method"]
    expected_meta["revelio_school_map_paths"] = {
        "deterministic_triple_map": school_map_meta.get("deterministic_triple_map"),
        "ref_inst_catalog": school_map_meta.get("ref_inst_catalog"),
        "legacy_crosswalk": school_map_meta.get("legacy_crosswalk"),
    }
    expected_meta["shared_universe_meta"] = universe_meta
    if paths.company_features_out.exists() and not force_rebuild:
        meta = _load_metadata(paths.company_features_out)
        if _metadata_compatible(meta, expected_meta):
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
