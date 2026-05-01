"""Source-rebuilt matched z_ct exposure design.

This module rebuilds the baseline shift-share exposure from source-side school
flows plus the existing school-shock methodology, then estimates a matched
high-vs-low exposure design with a common-break event study and a stacked DiD.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Iterable, Optional, Sequence

import duckdb as ddb
import numpy as np
import pandas as pd

try:
    import pyfixest as pf
except ImportError:  # pragma: no cover
    pf = None  # type: ignore[assignment]

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None  # type: ignore[assignment]

try:
    from company_shift_share.config_loader import (
        apply_testing_output_suffix,
        get_cfg_section,
        load_config,
    )
    from company_shift_share.exposure_event_study import (
        _build_post2016_target,
        _ensure_derived_outcome,
        _infer_feature_types,
        _select_index_feature_columns,
        fit_opt_probability_index,
    )
    from company_shift_share.institution_mapping import load_revelio_school_map
    from company_shift_share.revelio_company_features import (
        _build_annual_feature_frame,
        _build_static_feature_frame,
        _merge_feature_frames,
        summarize_pre_period_features,
    )
    from company_shift_share.source_exposure_data import (
        OPT_COUNT_COLUMNS,
        load_or_build_source_firm_universe,
        load_or_build_source_opt_counts,
        load_or_build_wrds_company_year_workforce_cache,
        load_or_build_wrds_school_flows_cache,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import (  # type: ignore[no-redef]
        apply_testing_output_suffix,
        get_cfg_section,
        load_config,
    )
    from company_shift_share.exposure_event_study import (  # type: ignore[no-redef]
        _build_post2016_target,
        _ensure_derived_outcome,
        _infer_feature_types,
        _select_index_feature_columns,
        fit_opt_probability_index,
    )
    from company_shift_share.institution_mapping import load_revelio_school_map  # type: ignore[no-redef]
    from company_shift_share.revelio_company_features import (  # type: ignore[no-redef]
        _build_annual_feature_frame,
        _build_static_feature_frame,
        _merge_feature_frames,
        summarize_pre_period_features,
    )
    from company_shift_share.source_exposure_data import (  # type: ignore[no-redef]
        OPT_COUNT_COLUMNS,
        load_or_build_source_firm_universe,
        load_or_build_source_opt_counts,
        load_or_build_wrds_company_year_workforce_cache,
        load_or_build_wrds_school_flows_cache,
    )


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "company_shift_share_matched_exposure_design.yaml"
)
DEFAULT_SHIFT_SHARE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "company_shift_share_apr2026.yaml"
)
DEFAULT_SOURCE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "company_shift_share_exposure_event_study.yaml"
)
MATCHING_FEATURE_FAMILY = "matching_fundamentals"
SUPPORTED_TRAINING_SAMPLE_MODES = ("preferred_plus_outside_negatives", "preferred_only")
SUPPORTED_EXPOSURE_METRICS = ("cum_z", "max_z", "mean_z", "active_share")
SUPPORTED_PROPENSITY_DESIGNS = ("opt_takeup", "treatment_propensity")
SUPPORTED_MATCHING_ALGORITHMS = ("optimal_max_cardinality_min_cost",)
SUPPORTED_MATCHING_COVERAGE_RULES = ("full_pre_coverage",)
DEFAULT_TRAJECTORY_SPECS = (
    {"name": "pre_policy", "start": 2010, "end": 2015, "run_matching": True, "run_regressions": True},
    {"name": "full_path", "start": 2010, "end": 2022, "run_matching": True, "run_regressions": True},
)


@dataclass(frozen=True)
class MatchedExposurePaths:
    out_dir: Path
    transition_shares_out: Path
    school_growth_out: Path
    instrument_panel_out: Path
    school_shift_sample_out: Path
    analysis_panel_out: Path
    matching_features_out: Path
    propensity_scores_out: Path
    trajectory_summary_out: Path
    matched_pairs_out: Path
    balance_table_out: Path
    balance_summary_out: Path
    matched_panel_out: Path
    final_matched_analysis_panel_out: Path
    common_break_results_out: Path
    stacked_panel_out: Path
    stacked_results_out: Path
    diagnostics_out: Path


def _resolve_path(paths_cfg: dict, key: str, *, allow_missing: bool = False) -> Path:
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"Config paths.{key} must be set.")
    path = Path(str(value))
    if not allow_missing and not path.exists():
        raise FileNotFoundError(f"Required path does not exist: {path}")
    return path


def _resolve_optional_path_value(value: object) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return Path(text)


def _resolve_output_path(cfg: dict, paths_cfg: dict, key: str) -> Path:
    return apply_testing_output_suffix(_resolve_path(paths_cfg, key, allow_missing=True), cfg)


def _resolve_output_path_with_default(
    cfg: dict,
    paths_cfg: dict,
    key: str,
    default: Optional[Path] = None,
) -> Path:
    if key in paths_cfg:
        return _resolve_output_path(cfg, paths_cfg, key)
    if default is None:
        raise ValueError(f"Config paths.{key} must be set.")
    return apply_testing_output_suffix(default, cfg)


def _resolve_paths(cfg: dict) -> MatchedExposurePaths:
    paths_cfg = get_cfg_section(cfg, "paths")
    out_dir = apply_testing_output_suffix(_resolve_path(paths_cfg, "out_dir", allow_missing=True), cfg)
    return MatchedExposurePaths(
        out_dir=out_dir,
        transition_shares_out=_resolve_output_path(cfg, paths_cfg, "transition_shares_out"),
        school_growth_out=_resolve_output_path(cfg, paths_cfg, "school_growth_out"),
        instrument_panel_out=_resolve_output_path(cfg, paths_cfg, "instrument_panel_out"),
        school_shift_sample_out=_resolve_output_path(cfg, paths_cfg, "school_shift_sample_out"),
        analysis_panel_out=_resolve_output_path(cfg, paths_cfg, "analysis_panel_out"),
        matching_features_out=_resolve_output_path(cfg, paths_cfg, "matching_features_out"),
        propensity_scores_out=_resolve_output_path(cfg, paths_cfg, "propensity_scores_out"),
        trajectory_summary_out=_resolve_output_path(cfg, paths_cfg, "trajectory_summary_out"),
        matched_pairs_out=_resolve_output_path(cfg, paths_cfg, "matched_pairs_out"),
        balance_table_out=_resolve_output_path(cfg, paths_cfg, "balance_table_out"),
        balance_summary_out=_resolve_output_path(cfg, paths_cfg, "balance_summary_out"),
        matched_panel_out=_resolve_output_path(cfg, paths_cfg, "matched_panel_out"),
        final_matched_analysis_panel_out=_resolve_output_path_with_default(
            cfg,
            paths_cfg,
            "final_matched_analysis_panel_out",
            default=out_dir / "final_matched_analysis_panel.parquet",
        ),
        common_break_results_out=_resolve_output_path(cfg, paths_cfg, "common_break_results_out"),
        stacked_panel_out=_resolve_output_path(cfg, paths_cfg, "stacked_panel_out"),
        stacked_results_out=_resolve_output_path(cfg, paths_cfg, "stacked_results_out"),
        diagnostics_out=_resolve_output_path(cfg, paths_cfg, "diagnostics_out"),
    )


def _shift_share_module():
    try:
        from company_shift_share import shift_share_analysis as ssa
    except ModuleNotFoundError:
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from company_shift_share import shift_share_analysis as ssa  # type: ignore[no-redef]
    return ssa


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[matched_exposure_design] Wrote {path}")


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[matched_exposure_design] Wrote {path}")


def _log_step(message: str) -> None:
    print(f"[matched_exposure_design] {message}")


def _elapsed_seconds(started: float) -> float:
    return float(time.perf_counter() - started)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[matched_exposure_design] Wrote {path}")


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"[matched_exposure_design] Wrote {path}")


def _slugify(value: object) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower()).strip("_")
    return text or "value"


def _df_to_markdown(df: pd.DataFrame, *, max_rows: Optional[int] = None) -> str:
    view = df.head(int(max_rows)).copy() if max_rows is not None else df.copy()
    if view.empty:
        return "_No rows._"
    try:
        return view.to_markdown(index=False)
    except Exception:  # pragma: no cover
        return "```\n" + view.to_string(index=False) + "\n```"


def _add_confidence_intervals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["coef"] = pd.to_numeric(out["coef"], errors="coerce")
    out["se"] = pd.to_numeric(out["se"], errors="coerce")
    out["ci_lower"] = out["coef"] - 1.96 * out["se"]
    out["ci_upper"] = out["coef"] + 1.96 * out["se"]
    return out


def _safe_json_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text()) if path.exists() else {}


def _maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(text)


def _empty_school_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["c", "t"])


def _first_present_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _add_matching_size_growth_features(
    feature_df: pd.DataFrame,
    *,
    size_feature: str = "match_firm_size_pre_level",
    growth_feature: str = "match_firm_size_pre_growth",
) -> pd.DataFrame:
    out = feature_df.copy()
    size_col = _first_present_column(
        out,
        [
            size_feature,
            "firm_size_annual_pre_level",
            "total_headcount_annual_pre_level",
            "total_headcount_wrds_annual_pre_level",
        ],
    )
    growth_col = _first_present_column(
        out,
        [
            growth_feature,
            "firm_size_annual_pre_growth",
            "total_headcount_annual_pre_growth",
            "total_headcount_wrds_annual_pre_growth",
        ],
    )
    out[size_feature] = (
        pd.to_numeric(out[size_col], errors="coerce")
        if size_col is not None
        else pd.Series(np.nan, index=out.index)
    )
    out[growth_feature] = (
        pd.to_numeric(out[growth_col], errors="coerce")
        if growth_col is not None
        else pd.Series(np.nan, index=out.index)
    )
    if "company_n_users_log1p" in out.columns:
        out[size_feature] = out[size_feature].fillna(
            pd.to_numeric(out["company_n_users_log1p"], errors="coerce")
        )
    return out


def _exclude_matching_feature(column: str) -> bool:
    return (
        column.startswith("school_")
        or column.startswith("n_schools_")
        or column.startswith("opt_")
        or "_opt_" in column
    )


def _parse_trajectory_specs(design_cfg: dict) -> list[dict[str, object]]:
    raw_specs = design_cfg.get("trajectory_specs")
    if raw_specs is None:
        return [dict(spec) for spec in DEFAULT_TRAJECTORY_SPECS]
    specs: list[dict[str, object]] = []
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("matched_exposure_design.trajectory_specs entries must be mappings.")
        name = str(raw_spec.get("name", "")).strip()
        if not name:
            raise ValueError("Each trajectory spec must set a non-empty 'name'.")
        specs.append(
            {
                "name": name,
                "start": int(raw_spec["start"]),
                "end": int(raw_spec["end"]),
                "run_matching": bool(raw_spec.get("run_matching", True)),
                "run_regressions": bool(raw_spec.get("run_regressions", True)),
            }
        )
    return specs


def _load_base_configs(cfg: dict) -> tuple[dict, dict]:
    design_cfg = get_cfg_section(cfg, "matched_exposure_design")
    shift_path = _resolve_optional_path_value(design_cfg.get("base_shift_share_config_path")) or DEFAULT_SHIFT_SHARE_CONFIG_PATH
    source_path = _resolve_optional_path_value(design_cfg.get("base_source_config_path")) or DEFAULT_SOURCE_CONFIG_PATH
    shift_cfg = load_config(shift_path)
    source_cfg = load_config(source_path)
    shift_pipeline = shift_cfg.setdefault("pipeline", {})
    for key in (
        "school_sample_window_start",
        "school_sample_window_end",
        "event_shock_pre_years",
        "event_shock_post_years",
        "share_year_min",
        "share_year_max",
    ):
        if key in design_cfg and design_cfg.get(key) is not None:
            shift_pipeline[key] = int(design_cfg[key])
    if "share_robustness_windows" in design_cfg and design_cfg.get("share_robustness_windows") is not None:
        shift_pipeline["share_robustness_windows"] = copy.deepcopy(design_cfg["share_robustness_windows"])
    if "reuse_cached_wrds_universe_only" in design_cfg:
        source_cfg["reuse_cached_wrds_universe_only"] = bool(design_cfg.get("reuse_cached_wrds_universe_only", False))
    if "testing" in cfg:
        shift_cfg["testing"] = copy.deepcopy(cfg["testing"])
        source_cfg["testing"] = copy.deepcopy(cfg["testing"])
    return shift_cfg, source_cfg


def build_source_analysis_panel_from_inputs(
    *,
    source_cfg: dict,
    data_min_t: int,
    data_max_t: int,
    force_rebuild_inputs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    firms, _, selected_meta, _ = load_or_build_source_firm_universe(
        cfg=source_cfg,
        force_rebuild=force_rebuild_inputs,
    )
    counts, _ = load_or_build_source_opt_counts(
        cfg=source_cfg,
        year_min=data_min_t,
        year_max=data_max_t,
        force_rebuild=force_rebuild_inputs,
    )
    workforce, _ = load_or_build_wrds_company_year_workforce_cache(
        cfg=source_cfg,
        year_min=data_min_t,
        year_max=data_max_t,
        force_rebuild=force_rebuild_inputs,
    )

    years = pd.DataFrame({"t": np.arange(int(data_min_t), int(data_max_t) + 1, dtype=int)})
    firms_work = firms.copy()
    firms_work["key"] = 1
    years["key"] = 1
    panel = firms_work.merge(years, on="key", how="inner").drop(columns=["key"])
    panel = panel.merge(counts[["c", "t"] + OPT_COUNT_COLUMNS], on=["c", "t"], how="left")
    panel = panel.merge(
        workforce[
            [
                "c",
                "t",
                "total_headcount_wrds_annual",
                "total_headcount_foreign_weighted_annual",
                "total_headcount_native_weighted_annual",
                "total_headcount_foreign_hard_annual",
                "total_headcount_native_hard_annual",
                "n_new_hires_wrds_annual",
                "n_new_hires_foreign_weighted_annual",
                "n_new_hires_native_weighted_annual",
                "n_new_hires_foreign_hard_annual",
                "n_new_hires_native_hard_annual",
            ]
        ],
        on=["c", "t"],
        how="left",
    )
    for col in OPT_COUNT_COLUMNS:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0).astype(int)
    panel["y_cst_lag0"] = pd.to_numeric(panel["total_headcount_wrds_annual"], errors="coerce").fillna(0.0)
    panel["y_cst_foreign_lag0"] = pd.to_numeric(
        panel["total_headcount_foreign_weighted_annual"], errors="coerce"
    ).fillna(0.0)
    panel["y_cst_native_lag0"] = pd.to_numeric(
        panel["total_headcount_native_weighted_annual"], errors="coerce"
    ).fillna(0.0)
    panel["y_cst_foreign_hard_lag0"] = pd.to_numeric(
        panel["total_headcount_foreign_hard_annual"], errors="coerce"
    ).fillna(0.0)
    panel["y_cst_native_hard_lag0"] = pd.to_numeric(
        panel["total_headcount_native_hard_annual"], errors="coerce"
    ).fillna(0.0)
    panel["y_new_hires_lag0"] = pd.to_numeric(panel["n_new_hires_wrds_annual"], errors="coerce").fillna(0.0)
    panel["y_new_hires_foreign_lag0"] = pd.to_numeric(
        panel["n_new_hires_foreign_weighted_annual"], errors="coerce"
    ).fillna(0.0)
    panel["y_new_hires_native_lag0"] = pd.to_numeric(
        panel["n_new_hires_native_weighted_annual"], errors="coerce"
    ).fillna(0.0)
    panel["y_new_hires_foreign_hard_lag0"] = pd.to_numeric(
        panel["n_new_hires_foreign_hard_annual"], errors="coerce"
    ).fillna(0.0)
    panel["y_new_hires_native_hard_lag0"] = pd.to_numeric(
        panel["n_new_hires_native_hard_annual"], errors="coerce"
    ).fillna(0.0)
    panel = panel.drop(
        columns=[
            "total_headcount_wrds_annual",
            "total_headcount_foreign_weighted_annual",
            "total_headcount_native_weighted_annual",
            "total_headcount_foreign_hard_annual",
            "total_headcount_native_hard_annual",
            "n_new_hires_wrds_annual",
            "n_new_hires_foreign_weighted_annual",
            "n_new_hires_native_weighted_annual",
            "n_new_hires_foreign_hard_annual",
            "n_new_hires_native_hard_annual",
        ]
    )
    return panel.reset_index(drop=True), firms.reset_index(drop=True), selected_meta.reset_index(drop=True), workforce


def build_matching_feature_frame_from_wrds(
    *,
    firms: pd.DataFrame,
    selected_meta: pd.DataFrame,
    wrds_annual: pd.DataFrame,
    feature_year_min: int,
    feature_year_max: int,
) -> pd.DataFrame:
    annual = _build_annual_feature_frame(
        wrds_annual=wrds_annual,
        school_annual=_empty_school_frame(),
        opt_counts=_empty_school_frame(),
        year_min=feature_year_min,
        year_max=feature_year_max,
    )
    annual_feature_cols = [
        col
        for col in annual.columns
        if col not in {"c", "t"} and not _exclude_matching_feature(str(col))
    ]
    summarized = summarize_pre_period_features(
        annual,
        annual_feature_cols,
        year_min=feature_year_min,
        year_max=feature_year_max,
    )
    static = _build_static_feature_frame(
        selected_meta,
        firms,
        feature_year_max=feature_year_max,
    )
    return _add_matching_size_growth_features(
        _merge_feature_frames(static, summarized),
    )


def build_transition_shares_from_source_flows(
    school_flows: pd.DataFrame,
    school_map: pd.DataFrame,
    firms: pd.DataFrame,
    *,
    share_period: str,
    share_base_year: int,
    share_year_min: int,
    share_year_max: int,
    robustness_windows: Sequence[Sequence[int]],
    min_universities_for_share: int,
) -> pd.DataFrame:
    ssa = _shift_share_module()
    con = ddb.connect()
    try:
        con.register("_school_flows_df", school_flows)
        con.register(
            "_school_map_df",
            school_map.loc[:, ["university_raw_key", "unitid"]].dropna(subset=["university_raw_key", "unitid"]),
        )
        con.register("_matched_firms_df", firms[["c"]].rename(columns={"c": "rcid"}))
        con.sql(
            """
            CREATE OR REPLACE TEMP VIEW revelio_transitions AS
            SELECT
                CAST(c AS BIGINT) AS rcid,
                CAST(university_raw AS VARCHAR) AS university_raw,
                CAST(t AS INTEGER) AS year,
                CAST(n_transitions AS DOUBLE) AS n_transitions
            FROM _school_flows_df
            WHERE c IS NOT NULL AND university_raw IS NOT NULL AND t IS NOT NULL
            """
        )
        con.sql(
            """
            CREATE OR REPLACE TEMP VIEW revelio_inst_cw AS
            SELECT
                CAST(university_raw_key AS VARCHAR) AS university_raw_norm,
                CAST(unitid AS VARCHAR) AS unitid
            FROM _school_map_df
            """
        )
        con.sql(
            """
            CREATE OR REPLACE TEMP VIEW matched_rcids AS
            SELECT DISTINCT CAST(rcid AS BIGINT) AS rcid
            FROM _matched_firms_df
            WHERE rcid IS NOT NULL
            """
        )
        ssa._build_transition_shares(
            con,
            share_period=share_period,
            share_base_year=share_base_year,
            share_year_min=share_year_min,
            share_year_max=share_year_max,
            robustness_windows=[(int(start), int(end)) for start, end in robustness_windows],
            exclude_unitids=None,
            min_universities_for_share=min_universities_for_share,
        )
        return con.sql("SELECT * FROM transition_shares ORDER BY c, k").df()
    finally:
        con.close()


def build_instrument_panel_from_transition_shares(
    transition_shares: pd.DataFrame,
    school_growth: pd.DataFrame,
) -> pd.DataFrame:
    ssa = _shift_share_module()
    con = ddb.connect()
    try:
        con.register("_transition_shares_df", transition_shares)
        con.register("_school_growth_df", school_growth)
        con.sql("CREATE OR REPLACE TEMP VIEW transition_shares AS SELECT * FROM _transition_shares_df")
        con.sql("CREATE OR REPLACE TEMP VIEW ipeds_unit_growth AS SELECT * FROM _school_growth_df")
        ssa._build_instrument(con)
        return con.sql("SELECT * FROM instrument_panel ORDER BY c, t").df()
    finally:
        con.close()


def build_shift_share_school_growth(
    *,
    shift_cfg: dict,
    data_min_t: int,
    data_max_t: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ssa = _shift_share_module()
    pipeline_cfg = get_cfg_section(shift_cfg, "pipeline")
    testing_cfg = get_cfg_section(shift_cfg, "testing")
    paths_cfg = get_cfg_section(shift_cfg, "paths")

    con = ddb.connect()
    try:
        ssa._load_inputs(con, paths_cfg, pipeline_cfg, testing_cfg)
        metric = str(pipeline_cfg.get("school_shift_metric", "ihmp_share"))
        degree_scope = str(pipeline_cfg.get("opt_shifts_degree_scope", "bachelors_masters"))
        metric_panel = ssa._build_school_metric_panel(
            con,
            metric=metric,
            degree_scope=degree_scope,
            exclude_unitids=None,
            opt_ihmp_ipeds_share_intl_threshold=float(
                pipeline_cfg.get("opt_ihmp_ipeds_share_intl_threshold", 0.30)
            ),
            opt_ihmp_foia_opt_share_threshold=float(
                pipeline_cfg.get("opt_ihmp_foia_opt_share_threshold", 0.50)
            ),
            opt_ihmp_min_program_f1_count=int(pipeline_cfg.get("opt_ihmp_min_program_f1_count", 10)),
        )
        school_name_lookup = ssa._school_name_lookup(con)
        school_classification = ssa._school_classification_lookup(con)
        sample_summary = ssa._build_school_shift_sample(
            metric_panel=metric_panel,
            metric=metric,
            school_name_lookup=school_name_lookup,
            school_classification=school_classification,
            opt_share_panel=None,
            n_shifted=int(pipeline_cfg.get("school_sample_n_shifted", 25)),
            window_start=int(pipeline_cfg.get("school_sample_window_start", 2014)),
            window_end=int(pipeline_cfg.get("school_sample_window_end", 2017)),
            event_pre_years=int(pipeline_cfg.get("event_shock_pre_years", 2)),
            event_post_years=int(pipeline_cfg.get("event_shock_post_years", 2)),
            control_positive_cap=float(pipeline_cfg.get("school_sample_control_positive_cap", 0.05)),
            match_on_carnegie_classification=bool(
                pipeline_cfg.get("match_school_pairs_by_carnegie_classification", True)
            ),
            min_school_size=int(pipeline_cfg.get("school_sample_min_size", 400)),
            opt_share_min_school_f1_count=int(pipeline_cfg.get("opt_share_min_school_f1_count", 50)),
            opt_share_max_yoy_drop=float(pipeline_cfg.get("opt_share_max_yoy_drop", 0.50)),
            restrict_treated_to_no_large_enrollment_jump=bool(
                pipeline_cfg.get("restrict_treated_to_no_large_enrollment_jump", False)
            ),
            max_yoy_size_jump=float(pipeline_cfg.get("school_sample_max_yoy_size_jump", 0.40)),
        )
        ssa._build_event_quantity_growth_view(
            con,
            metric_panel=metric_panel,
            sample_summary=sample_summary,
            opt_share_panel=None,
            year_min=data_min_t,
            year_max=data_max_t,
            event_window_start=int(pipeline_cfg.get("school_sample_window_start", 2014)),
            event_window_end=int(pipeline_cfg.get("school_sample_window_end", 2017)),
            event_pre_years=int(pipeline_cfg.get("event_shock_pre_years", 2)),
            event_post_years=int(pipeline_cfg.get("event_shock_post_years", 2)),
            falsification_lead_years=int(pipeline_cfg.get("falsification_lead_years", 4)),
        )
        school_growth = con.sql("SELECT * FROM ipeds_unit_growth ORDER BY k, t").df()
    finally:
        con.close()
    return school_growth, sample_summary


def build_shift_share_components_from_configs(
    *,
    shift_cfg: dict,
    source_cfg: dict,
    source_flow_year_min: int,
    source_flow_year_max: int,
    data_min_t: int,
    data_max_t: int,
    force_rebuild_inputs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipeline_cfg = get_cfg_section(shift_cfg, "pipeline")
    school_growth, school_shift_sample = build_shift_share_school_growth(
        shift_cfg=shift_cfg,
        data_min_t=data_min_t,
        data_max_t=data_max_t,
    )
    firms, _, _, _ = load_or_build_source_firm_universe(
        cfg=source_cfg,
        force_rebuild=force_rebuild_inputs,
    )
    school_flows, _ = load_or_build_wrds_school_flows_cache(
        cfg=source_cfg,
        year_min=source_flow_year_min,
        year_max=source_flow_year_max,
        force_rebuild=force_rebuild_inputs,
    )
    source_paths = get_cfg_section(source_cfg, "paths")
    school_map, _ = load_revelio_school_map(
        legacy_crosswalk=Path(str(source_paths["revelio_ipeds_foia_inst_crosswalk"])),
        deterministic_triple_map=_resolve_optional_path_value(source_paths.get("revelio_inst_deterministic_map")),
        ref_inst_catalog=_resolve_optional_path_value(source_paths.get("revelio_ref_inst_catalog")),
    )
    transition_shares = build_transition_shares_from_source_flows(
        school_flows=school_flows,
        school_map=school_map,
        firms=firms,
        share_period=str(pipeline_cfg.get("share_period", "pre_window")),
        share_base_year=int(pipeline_cfg.get("share_base_year", 2010)),
        share_year_min=int(pipeline_cfg.get("share_year_min", 2008)),
        share_year_max=int(pipeline_cfg.get("share_year_max", 2013)),
        robustness_windows=pipeline_cfg.get("share_robustness_windows", ((2008, 2010), (2011, 2013), (2008, 2013))),
        min_universities_for_share=int(pipeline_cfg.get("share_min_universities_for_share", 2)),
    )
    instrument_panel = build_instrument_panel_from_transition_shares(
        transition_shares=transition_shares,
        school_growth=school_growth,
    )
    return school_growth, school_shift_sample, transition_shares, instrument_panel


def merge_zct_into_analysis_panel(
    panel: pd.DataFrame,
    instrument_panel: pd.DataFrame,
) -> pd.DataFrame:
    merged = panel.merge(instrument_panel, on=["c", "t"], how="left")
    instrument_cols = [col for col in instrument_panel.columns if col not in {"c", "t"}]
    for col in instrument_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    _ensure_derived_outcome(merged, "x_bin_any_nonzero", x_source_col="any_opt_hires_correction_aware")
    for outcome_col in (
        "log1p_y_cst_lag0",
        "log1p_y_cst_foreign_lag0",
        "log1p_y_cst_native_lag0",
        "log1p_y_new_hires_lag0",
        "log1p_y_new_hires_foreign_lag0",
        "log1p_y_new_hires_native_lag0",
    ):
        _ensure_derived_outcome(merged, outcome_col, x_source_col="any_opt_hires_correction_aware")
    return merged


def summarize_zct_trajectory(
    panel: pd.DataFrame,
    *,
    window_start: int,
    window_end: int,
    trajectory_name: str,
) -> pd.DataFrame:
    window = panel.loc[panel["t"].between(int(window_start), int(window_end))].copy()
    if window.empty:
        raise ValueError(f"No panel rows found in trajectory window {window_start}-{window_end}.")
    grouped = (
        window.groupby("c", as_index=False)
        .agg(
            cum_z=("z_ct", "sum"),
            active_share=("z_ct", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).mean())),
            max_z=("z_ct", "max"),
            mean_z=("z_ct", "mean"),
            preferred_rcid_source=("preferred_rcid_source", "max"),
            outside_negative_candidate=("outside_negative_candidate", "max"),
            in_analysis_universe=("in_analysis_universe", "max"),
        )
    )
    grouped["trajectory_name"] = trajectory_name
    grouped["window_start"] = int(window_start)
    grouped["window_end"] = int(window_end)
    grouped["any_positive_z"] = (pd.to_numeric(grouped["cum_z"], errors="coerce").fillna(0.0) > 0).astype(int)
    return grouped


def _resolve_exposure_metric(
    trajectory_df: pd.DataFrame,
    metric_name: str,
) -> pd.Series:
    normalized = str(metric_name).strip().lower()
    if normalized not in SUPPORTED_EXPOSURE_METRICS:
        raise ValueError(
            f"Unsupported exposure metric {metric_name!r}. "
            f"Expected one of {SUPPORTED_EXPOSURE_METRICS}."
        )
    return pd.to_numeric(trajectory_df[normalized], errors="coerce").fillna(0.0)


def classify_zct_trajectory(
    trajectory_df: pd.DataFrame,
    *,
    high_quantile: float,
    low_quantile: float,
    high_exposure_metric: str,
    low_exposure_metric: str,
) -> pd.DataFrame:
    work = trajectory_df.copy()
    high_metric = _resolve_exposure_metric(work, high_exposure_metric)
    low_metric = _resolve_exposure_metric(work, low_exposure_metric)
    preferred_mask = work["preferred_rcid_source"].fillna(0).astype(int).eq(1)
    positive_reference = high_metric.loc[preferred_mask & high_metric.gt(0)]
    if positive_reference.empty:
        raise ValueError(
            "No positive preferred-firm z_ct trajectories available for thresholding "
            f"under high_exposure_metric={high_exposure_metric!r}."
        )
    low_cutoff = float(positive_reference.quantile(float(low_quantile)))
    high_cutoff = float(positive_reference.quantile(float(high_quantile)))
    is_high = preferred_mask & high_metric.ge(high_cutoff)
    is_low = (~is_high) & (low_metric.eq(0.0) | low_metric.le(low_cutoff))
    work["low_cutoff"] = low_cutoff
    work["high_cutoff"] = high_cutoff
    work["high_exposure_metric"] = str(high_exposure_metric)
    work["low_exposure_metric"] = str(low_exposure_metric)
    work["high_exposure_metric_value"] = high_metric
    work["low_exposure_metric_value"] = low_metric
    work["exposure_group"] = np.select(
        [is_high, is_low],
        ["high_exposure", "low_exposure"],
        default="middle_exposure",
    )
    work["high_exposure"] = is_high.astype(int)
    work["low_exposure"] = is_low.astype(int)
    return work


def _unpack_propensity_fit_result(
    fit_result: tuple[pd.DataFrame, dict] | tuple[pd.DataFrame, dict, dict[str, object]],
) -> tuple[pd.DataFrame, dict, dict[str, object]]:
    if len(fit_result) == 3:
        pred_df, diagnostics, artifacts = fit_result
        return pred_df, diagnostics, artifacts
    pred_df, diagnostics = fit_result
    return pred_df, diagnostics, {}


def _raw_feature_for_design_column(feature: str, raw_features: Sequence[str]) -> str:
    if feature.startswith("ix__"):
        interaction = feature.removeprefix("ix__")
        parts = [part for part in interaction.split("__x__") if part]
        return " x ".join(parts) if parts else feature
    for raw_feature in sorted((str(col) for col in raw_features), key=len, reverse=True):
        if feature == raw_feature or feature.startswith(f"{raw_feature}_"):
            return raw_feature
    return feature


def _propensity_feature_weight_records(
    diagnostics: dict[str, object],
    artifacts: dict[str, object],
    *,
    propensity_design: str,
    trajectory_name: Optional[str] = None,
) -> list[dict[str, object]]:
    weight_series = artifacts.get("weight_series")
    records: list[dict[str, object]] = []
    if isinstance(weight_series, pd.Series) and not weight_series.empty:
        raw_features = [str(col) for col in artifacts.get("feature_columns_raw", [])]
        standardized_columns = {str(col) for col in artifacts.get("standardized_feature_columns", [])}
        interaction_columns = {str(col) for col in artifacts.get("interaction_column_names", [])}
        model_method = str(diagnostics.get("model_method", "unknown"))
        weight_kind = "feature_importance" if model_method == "random_forest" else "coefficient"
        sorted_weights = weight_series.sort_values(key=lambda s: s.abs(), ascending=False)
        for rank, (feature, weight) in enumerate(sorted_weights.items(), start=1):
            weight_value = _safe_json_float(weight)
            if weight_value is None:
                continue
            feature_name = str(feature)
            records.append(
                {
                    "trajectory_name": trajectory_name,
                    "propensity_design": str(propensity_design),
                    "model_method": model_method,
                    "weight_kind": weight_kind,
                    "rank_abs_weight": int(rank),
                    "feature": feature_name,
                    "raw_feature": _raw_feature_for_design_column(feature_name, raw_features),
                    "is_interaction": int(feature_name in interaction_columns or feature_name.startswith("ix__")),
                    "standardized_input": int(feature_name in standardized_columns),
                    "weight": weight_value,
                    "abs_weight": abs(weight_value),
                }
            )
        return records

    for weight_key, weight_kind in (
        ("top_coefficients", "coefficient"),
        ("top_feature_importances", "feature_importance"),
    ):
        weights = diagnostics.get(weight_key)
        if not isinstance(weights, dict):
            continue
        sorted_items = sorted(
            ((str(feature), _safe_json_float(value)) for feature, value in weights.items()),
            key=lambda item: abs(item[1]) if item[1] is not None else -1,
            reverse=True,
        )
        for rank, (feature, value) in enumerate(sorted_items, start=1):
            if value is None:
                continue
            records.append(
                {
                    "trajectory_name": trajectory_name,
                    "propensity_design": str(propensity_design),
                    "model_method": str(diagnostics.get("model_method", "unknown")),
                    "weight_kind": weight_kind,
                    "rank_abs_weight": int(rank),
                    "feature": feature,
                    "raw_feature": feature,
                    "is_interaction": int(feature.startswith("ix__")),
                    "standardized_input": np.nan,
                    "weight": value,
                    "abs_weight": abs(value),
                }
            )
        if records:
            return records
    return records


def _attach_propensity_artifact_diagnostics(
    diagnostics: dict[str, object],
    artifacts: dict[str, object],
    *,
    propensity_design: str,
    trajectory_name: Optional[str] = None,
) -> dict[str, object]:
    out = dict(diagnostics)
    intercept = _safe_json_float(artifacts.get("intercept"))
    if intercept is not None:
        out["intercept"] = intercept
    feature_weights = _propensity_feature_weight_records(
        out,
        artifacts,
        propensity_design=propensity_design,
        trajectory_name=trajectory_name,
    )
    if feature_weights:
        out["feature_weights"] = feature_weights
        out["n_feature_weights"] = int(len(feature_weights))
        out["top_positive_weights"] = {
            str(row["feature"]): float(row["weight"])
            for row in sorted(feature_weights, key=lambda row: float(row["weight"]), reverse=True)[:15]
            if float(row["weight"]) > 0
        }
        out["top_negative_weights"] = {
            str(row["feature"]): float(row["weight"])
            for row in sorted(feature_weights, key=lambda row: float(row["weight"]))[:15]
            if float(row["weight"]) < 0
        }
    return out


def build_propensity_scores(
    feature_df: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    design_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    training_sample_mode = str(
        design_cfg.get("training_sample_mode", "preferred_plus_outside_negatives")
    ).strip() or "preferred_plus_outside_negatives"
    if training_sample_mode not in SUPPORTED_TRAINING_SAMPLE_MODES:
        raise ValueError(
            f"Unsupported training_sample_mode={training_sample_mode!r}. "
            f"Expected one of {SUPPORTED_TRAINING_SAMPLE_MODES}."
        )
    target_df = _build_post2016_target(
        panel,
        x_source_col=str(design_cfg.get("x_source_col", "any_opt_hires_correction_aware")),
        target_year_min=int(design_cfg.get("propensity_target_year_min", 2016)),
        target_year_max=int(design_cfg.get("propensity_target_year_max", 2022)),
    )
    feature_input = feature_df.copy()
    if training_sample_mode == "preferred_only":
        feature_input.loc[feature_input["outside_negative_candidate"].fillna(0).eq(1), "in_analysis_universe"] = 0
    pred_df, diagnostics, artifacts = _unpack_propensity_fit_result(
        fit_opt_probability_index(
            feature_input,
            target_df,
            model_method=str(design_cfg.get("propensity_model_method", "logit")),
            entry_mode=str(design_cfg.get("propensity_entry_mode", "continuous")),
            ntiles=int(design_cfg.get("propensity_ntiles", 2)),
            feature_year_min=int(design_cfg.get("propensity_feature_year_min", 2010)),
            feature_year_max=int(design_cfg.get("propensity_feature_year_max", 2013)),
            leaveout_enabled=bool(design_cfg.get("leaveout_enabled", False)),
            leaveout_share=float(design_cfg.get("leaveout_share", 0.25)),
            leaveout_seed=int(design_cfg.get("leaveout_seed", 42)),
            logit_class_weight=design_cfg.get("logit_class_weight", "balanced"),
            lasso_cv_folds=design_cfg.get("lasso_cv_folds"),
            lasso_n_cs=design_cfg.get("lasso_n_cs"),
            max_active_features=design_cfg.get("model_max_active_features"),
            max_feature_to_train_ratio=_maybe_float(design_cfg.get("model_max_feature_to_train_ratio")),
            feature_sample_seed=int(design_cfg.get("feature_sample_seed", 42)),
            feature_family=MATCHING_FEATURE_FAMILY,
            return_artifacts=True,
        )
    )
    pred_df["propensity_design"] = "opt_takeup"
    diagnostics = _attach_propensity_artifact_diagnostics(
        diagnostics,
        artifacts,
        propensity_design="opt_takeup",
    )
    diagnostics["propensity_design"] = "opt_takeup"
    diagnostics["training_sample_mode"] = training_sample_mode
    return pred_df, diagnostics


def build_treatment_propensity_scores(
    feature_df: pd.DataFrame,
    classified_trajectory: pd.DataFrame,
    *,
    design_cfg: dict,
    include_outside_negative_controls: bool,
) -> tuple[pd.DataFrame, dict]:
    trajectory_name = (
        str(classified_trajectory["trajectory_name"].iloc[0])
        if not classified_trajectory.empty and "trajectory_name" in classified_trajectory.columns
        else "unknown"
    )
    eligible_control = classified_trajectory["low_exposure"].eq(1)
    if not include_outside_negative_controls:
        eligible_control &= classified_trajectory["outside_negative_candidate"].fillna(0).ne(1)
    eligible_treated = (
        classified_trajectory["high_exposure"].eq(1)
        & classified_trajectory["preferred_rcid_source"].fillna(0).eq(1)
    )
    eligible_mask = eligible_treated | eligible_control

    feature_input = feature_df.copy()
    feature_input["in_analysis_universe"] = (
        feature_input["c"].isin(
            pd.to_numeric(
                classified_trajectory.loc[eligible_mask, "c"],
                errors="coerce",
            ).dropna().astype(int)
        )
    ).astype(int)
    target_df = classified_trajectory.loc[:, ["c", "high_exposure"]].rename(
        columns={"high_exposure": "post2016_any_opt"}
    ).copy()
    pred_df, diagnostics, artifacts = _unpack_propensity_fit_result(
        fit_opt_probability_index(
            feature_input,
            target_df,
            model_method=str(design_cfg.get("propensity_model_method", "logit")),
            entry_mode=str(design_cfg.get("propensity_entry_mode", "continuous")),
            ntiles=int(design_cfg.get("propensity_ntiles", 2)),
            feature_year_min=int(design_cfg.get("propensity_feature_year_min", 2010)),
            feature_year_max=int(design_cfg.get("propensity_feature_year_max", 2013)),
            leaveout_enabled=bool(design_cfg.get("leaveout_enabled", False)),
            leaveout_share=float(design_cfg.get("leaveout_share", 0.25)),
            leaveout_seed=int(design_cfg.get("leaveout_seed", 42)),
            logit_class_weight=design_cfg.get("logit_class_weight", "balanced"),
            lasso_cv_folds=design_cfg.get("lasso_cv_folds"),
            lasso_n_cs=design_cfg.get("lasso_n_cs"),
            max_active_features=design_cfg.get("model_max_active_features"),
            max_feature_to_train_ratio=_maybe_float(design_cfg.get("model_max_feature_to_train_ratio")),
            feature_sample_seed=int(design_cfg.get("feature_sample_seed", 42)),
            feature_family=MATCHING_FEATURE_FAMILY,
            return_artifacts=True,
        )
    )
    pred_df["propensity_design"] = "treatment_propensity"
    pred_df["trajectory_name"] = trajectory_name
    diagnostics = _attach_propensity_artifact_diagnostics(
        diagnostics,
        artifacts,
        propensity_design="treatment_propensity",
        trajectory_name=trajectory_name,
    )
    diagnostics["propensity_design"] = "treatment_propensity"
    diagnostics["trajectory_name"] = trajectory_name
    diagnostics["training_sample_mode"] = "eligible_match_pool"
    diagnostics["n_eligible_treated"] = int(eligible_treated.sum())
    diagnostics["n_eligible_controls"] = int(eligible_control.sum())
    diagnostics["include_outside_negative_controls"] = bool(include_outside_negative_controls)
    return pred_df, diagnostics


def _safe_logit(prob: pd.Series, eps: float = 1e-6) -> pd.Series:
    clipped = pd.to_numeric(prob, errors="coerce").clip(eps, 1 - eps)
    return np.log(clipped / (1.0 - clipped))


def _common_support_bounds(treated_scores: pd.Series, control_scores: pd.Series) -> tuple[float, float]:
    lower = max(float(treated_scores.min()), float(control_scores.min()))
    upper = min(float(treated_scores.max()), float(control_scores.max()))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("No common support overlap between treated and control logit scores.")
    return lower, upper


def _propensity_merge_keys(
    classified_trajectory: pd.DataFrame,
    propensity_df: pd.DataFrame,
) -> list[str]:
    if "trajectory_name" in classified_trajectory.columns and "trajectory_name" in propensity_df.columns:
        return ["c", "trajectory_name"]
    return ["c"]


def _build_matching_work_frame(
    classified_trajectory: pd.DataFrame,
    feature_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    *,
    size_match_feature: str,
    users_match_feature: str,
    coverage_feature: str,
) -> pd.DataFrame:
    merge_keys = _propensity_merge_keys(classified_trajectory, propensity_df)
    feature_cols = ["c"] + [
        col for col in feature_df.columns
        if col != "c" and col not in classified_trajectory.columns
    ]
    score_cols = merge_keys + [
        col
        for col in [
            "predicted_prob",
            "predicted_index",
            "preferred_rcid_source",
            "outside_negative_candidate",
            "propensity_design",
        ]
        if col in propensity_df.columns
    ]
    score_cols = list(dict.fromkeys(score_cols))
    scores = propensity_df[score_cols].copy()
    work = classified_trajectory.merge(feature_df[feature_cols], on="c", how="left")
    work = work.merge(
        scores.drop(columns=["preferred_rcid_source", "outside_negative_candidate"], errors="ignore"),
        on=merge_keys,
        how="left",
    )
    if "naics2" in work.columns:
        work["naics2"] = work["naics2"].fillna("__MISSING__").astype(str)
    if "company_hq_region" in work.columns:
        work["company_hq_region"] = work["company_hq_region"].fillna("__MISSING__").astype(str)
    work["predicted_prob"] = pd.to_numeric(work["predicted_prob"], errors="coerce")
    work = work.dropna(subset=["predicted_prob"]).copy()
    work["logit_score"] = _safe_logit(work["predicted_prob"])
    if users_match_feature in work.columns:
        work[users_match_feature] = pd.to_numeric(work[users_match_feature], errors="coerce")
        work["matching_users_value"] = pd.to_numeric(work[users_match_feature], errors="coerce")
    else:
        work["matching_users_value"] = np.nan
    size_values = (
        pd.to_numeric(work[size_match_feature], errors="coerce")
        if size_match_feature in work.columns
        else pd.Series(np.nan, index=work.index, dtype=float)
    )
    work["effective_size"] = size_values
    if users_match_feature in work.columns:
        work["effective_size"] = work["effective_size"].fillna(
            pd.to_numeric(work[users_match_feature], errors="coerce")
        )
    work["matching_coverage_years"] = (
        pd.to_numeric(work[coverage_feature], errors="coerce")
        if coverage_feature in work.columns
        else pd.Series(np.nan, index=work.index, dtype=float)
    )
    return work


def _matching_support_samples(
    work: pd.DataFrame,
    *,
    include_outside_negative_controls: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    treated_pre = work.loc[
        work["high_exposure"].eq(1) & work["preferred_rcid_source"].fillna(0).eq(1)
    ].copy()
    control_pre = work.loc[work["low_exposure"].eq(1)].copy()
    if not include_outside_negative_controls:
        control_pre = control_pre.loc[control_pre["outside_negative_candidate"].fillna(0).ne(1)].copy()
    if treated_pre.empty or control_pre.empty:
        raise ValueError(
            "No treated/control firms available for matching. "
            f"treated_pre={len(treated_pre)}, control_pre={len(control_pre)}, "
            f"n_scored={len(work)}, n_high={int(work['high_exposure'].eq(1).sum())}, "
            f"n_low={int(work['low_exposure'].eq(1).sum())}"
        )
    lower, upper = _common_support_bounds(treated_pre["logit_score"], control_pre["logit_score"])
    treated = treated_pre.loc[treated_pre["logit_score"].between(lower, upper)].copy()
    control = control_pre.loc[control_pre["logit_score"].between(lower, upper)].copy()
    if treated.empty or control.empty:
        raise ValueError(
            "No treated/control firms remain after common-support trimming. "
            f"treated_pre={len(treated_pre)}, control_pre={len(control_pre)}, "
            f"support=[{lower:.6f}, {upper:.6f}]"
        )
    return treated_pre, control_pre, treated, control, lower, upper


def _matching_row_constraint_mask(
    sample: pd.DataFrame,
    *,
    coverage_required_years: int,
    users_feature_required: bool,
) -> tuple[pd.Series, dict[str, int]]:
    coverage_ok = pd.to_numeric(sample["matching_coverage_years"], errors="coerce").ge(float(coverage_required_years))
    size_ok = pd.to_numeric(sample["effective_size"], errors="coerce").notna()
    if users_feature_required:
        users_ok = pd.to_numeric(sample["matching_users_value"], errors="coerce").notna()
    else:
        users_ok = pd.Series(True, index=sample.index)
    eligible = coverage_ok & size_ok & users_ok
    return eligible, {
        "dropped_low_coverage": int((~coverage_ok).sum()),
        "dropped_missing_effective_size": int((coverage_ok & ~size_ok).sum()),
        "dropped_missing_users_feature": int((coverage_ok & size_ok & ~users_ok).sum()),
    }


def _matching_eligible_samples(
    work: pd.DataFrame,
    *,
    include_outside_negative_controls: bool,
    coverage_required_years: int,
    users_feature_required: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, dict[str, int]]:
    treated_pre, control_pre, treated_support, control_support, lower, upper = _matching_support_samples(
        work,
        include_outside_negative_controls=include_outside_negative_controls,
    )
    treated_mask, treated_diag = _matching_row_constraint_mask(
        treated_support,
        coverage_required_years=coverage_required_years,
        users_feature_required=users_feature_required,
    )
    control_mask, control_diag = _matching_row_constraint_mask(
        control_support,
        coverage_required_years=coverage_required_years,
        users_feature_required=users_feature_required,
    )
    treated = treated_support.loc[treated_mask].copy()
    control = control_support.loc[control_mask].copy()
    if treated.empty or control.empty:
        raise ValueError(
            "No treated/control firms remain after hard matching constraints. "
            f"treated_after_support={len(treated_support)}, controls_after_support={len(control_support)}, "
            f"treated_after_constraints={len(treated)}, controls_after_constraints={len(control)}, "
            f"coverage_required_years={int(coverage_required_years)}"
        )
    diagnostics = {
        "n_treated_after_constraints": int(len(treated)),
        "n_controls_after_constraints": int(len(control)),
        "dropped_treated_low_coverage": int(treated_diag["dropped_low_coverage"]),
        "dropped_controls_low_coverage": int(control_diag["dropped_low_coverage"]),
        "dropped_treated_missing_effective_size": int(treated_diag["dropped_missing_effective_size"]),
        "dropped_controls_missing_effective_size": int(control_diag["dropped_missing_effective_size"]),
        "dropped_treated_missing_users_feature": int(treated_diag["dropped_missing_users_feature"]),
        "dropped_controls_missing_users_feature": int(control_diag["dropped_missing_users_feature"]),
    }
    return treated_pre, control_pre, treated_support, control_support, treated, control, lower, upper, diagnostics


def _matching_search_space_summary(
    treated: pd.DataFrame,
    control: pd.DataFrame,
) -> dict[str, float | int]:
    treated_counts = treated["naics2"].value_counts(dropna=False)
    control_counts = control["naics2"].value_counts(dropna=False)
    overlap = treated_counts.index.intersection(control_counts.index)
    if overlap.empty:
        return {
            "n_overlap_naics2": 0,
            "estimated_candidate_pairs": 0,
            "max_controls_in_overlap_naics2": 0,
            "median_controls_in_treated_naics2": 0.0,
        }
    overlap_control_counts = control_counts.loc[overlap].astype(float)
    overlap_treated_counts = treated_counts.loc[overlap].astype(float)
    candidate_pairs = int((overlap_treated_counts * overlap_control_counts).sum())
    return {
        "n_overlap_naics2": int(len(overlap)),
        "estimated_candidate_pairs": candidate_pairs,
        "max_controls_in_overlap_naics2": int(overlap_control_counts.max()),
        "median_controls_in_treated_naics2": float(overlap_control_counts.median()),
    }


def _safe_scale(*series_list: pd.Series) -> float:
    combined = pd.concat(
        [pd.to_numeric(series, errors="coerce") for series in series_list],
        ignore_index=True,
    ).dropna()
    scale = float(combined.std(ddof=0)) if not combined.empty else math.nan
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return scale


def _solve_optimal_match_assignment(
    candidate_control_positions: Sequence[np.ndarray],
    candidate_costs: Sequence[np.ndarray],
) -> tuple[list[tuple[int, int, float]], dict[str, float | int]]:
    if linear_sum_assignment is None:  # pragma: no cover
        raise ImportError("scipy is required for optimal matching in matched_exposure_design.")
    feasible_lists = [positions.astype(int, copy=False) for positions in candidate_control_positions if len(positions)]
    if not feasible_lists:
        return [], {"n_unique_feasible_controls": 0, "dummy_cost": 1.0}
    unique_controls = np.unique(np.concatenate(feasible_lists))
    n_rows = int(len(candidate_control_positions))
    n_real_controls = int(len(unique_controls))
    max_real_cost = max((float(np.max(costs)) for costs in candidate_costs if len(costs)), default=0.0)
    dummy_cost = float(max_real_cost + 1.0)
    infeasible_cost = float(dummy_cost * 2.0 + 1.0)
    cost_matrix = np.full((n_rows, n_real_controls + n_rows), infeasible_cost, dtype=np.float32)
    cost_matrix[:, n_real_controls:] = np.float32(dummy_cost)
    for row_idx, (positions, costs) in enumerate(zip(candidate_control_positions, candidate_costs, strict=False)):
        if not len(positions):
            continue
        compact_positions = np.searchsorted(unique_controls, positions.astype(int, copy=False))
        cost_matrix[row_idx, compact_positions] = costs.astype(np.float32, copy=False)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments: list[tuple[int, int, float]] = []
    for row_idx, col_idx in zip(row_ind, col_ind, strict=False):
        if col_idx >= n_real_controls:
            continue
        assigned_cost = float(cost_matrix[row_idx, col_idx])
        if assigned_cost >= dummy_cost:
            continue
        assignments.append((int(row_idx), int(unique_controls[col_idx]), assigned_cost))
    return assignments, {
        "n_unique_feasible_controls": n_real_controls,
        "dummy_cost": dummy_cost,
    }


def match_high_to_low_exposure(
    classified_trajectory: pd.DataFrame,
    feature_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    *,
    include_outside_negative_controls: bool,
    caliper_sd_multiplier: float,
    progress_every: Optional[int] = 500,
    size_match_feature: str = "match_firm_size_pre_level",
    growth_match_feature: str = "match_firm_size_pre_growth",
    size_match_weight: float = 0.35,
    growth_match_weight: float = 0.35,
    users_match_feature: str = "company_n_users_log1p",
    users_match_weight: float = 0.20,
    coverage_feature: str = "firm_size_annual_pre_n_years",
    coverage_rule: str = "full_pre_coverage",
    coverage_required_years: int = 4,
    matching_algorithm: str = "optimal_max_cardinality_min_cost",
    propensity_design: str = "opt_takeup",
) -> tuple[pd.DataFrame, dict]:
    matching_started = time.perf_counter()
    if str(matching_algorithm) not in SUPPORTED_MATCHING_ALGORITHMS:
        raise ValueError(
            f"Unsupported matching_algorithm={matching_algorithm!r}. "
            f"Expected one of {SUPPORTED_MATCHING_ALGORITHMS}."
        )
    if str(coverage_rule) not in SUPPORTED_MATCHING_COVERAGE_RULES:
        raise ValueError(
            f"Unsupported matching_coverage_rule={coverage_rule!r}. "
            f"Expected one of {SUPPORTED_MATCHING_COVERAGE_RULES}."
        )
    work = _build_matching_work_frame(
        classified_trajectory,
        feature_df,
        propensity_df,
        size_match_feature=size_match_feature,
        users_match_feature=users_match_feature,
        coverage_feature=coverage_feature,
    )
    if users_match_weight > 0 and users_match_feature not in work.columns:
        raise ValueError(
            f"Configured matching_users_feature={users_match_feature!r} is not available in the matching frame."
        )
    if coverage_feature not in work.columns:
        raise ValueError(
            f"Configured matching_coverage_feature={coverage_feature!r} is not available in the matching frame."
        )
    (
        treated_pre,
        control_pre,
        treated_support,
        control_support,
        treated,
        control,
        lower,
        upper,
        constraint_diag,
    ) = _matching_eligible_samples(
        work,
        include_outside_negative_controls=include_outside_negative_controls,
        coverage_required_years=int(coverage_required_years),
        users_feature_required=float(users_match_weight) > 0,
    )
    search_space = _matching_search_space_summary(treated, control)

    pooled_sd = float(pd.concat([treated["logit_score"], control["logit_score"]]).std(ddof=0))
    caliper = float(caliper_sd_multiplier) * pooled_sd
    if not np.isfinite(caliper) or caliper < 0:
        raise ValueError("Matching caliper is not finite.")

    n_treated = int(len(treated))
    n_control = int(len(control))
    _log_step(
        "matching start | "
        f"treated_before_support={len(treated_pre):,} controls_before_support={len(control_pre):,} "
        f"treated_after_support={len(treated_support):,} controls_after_support={len(control_support):,} "
        f"treated_after_constraints={n_treated:,} controls_after_constraints={n_control:,} "
        f"caliper={caliper:.6f} support=[{lower:.6f}, {upper:.6f}]"
    )
    _log_step(
        "matching search space | "
        f"overlap_naics2={int(search_space['n_overlap_naics2']):,} "
        f"estimated_candidate_pairs={int(search_space['estimated_candidate_pairs']):,} "
        f"max_controls_in_overlap_naics2={int(search_space['max_controls_in_overlap_naics2']):,} "
        f"median_controls_in_treated_naics2={float(search_space['median_controls_in_treated_naics2']):.1f}"
    )

    progress_interval = None
    if progress_every is not None:
        progress_interval = max(int(progress_every), 1)

    pairs: list[dict[str, object]] = []
    unmatched_reasons = {
        "no_exact_naics2_control": 0,
        "no_control_within_caliper": 0,
        "no_feasible_edge_after_constraints": 0,
        "unmatched_after_optimal_assignment": 0,
    }
    candidate_pool_total = 0
    within_caliper_total = 0
    feasible_edge_total = 0
    size_scale = _safe_scale(treated["effective_size"], control["effective_size"])
    growth_scale = (
        _safe_scale(treated[growth_match_feature], control[growth_match_feature])
        if growth_match_feature in treated.columns and growth_match_feature in control.columns
        else 1.0
    )
    users_scale = _safe_scale(treated["matching_users_value"], control["matching_users_value"])
    size_match_weight = max(float(size_match_weight), 0.0)
    growth_match_weight = max(float(growth_match_weight), 0.0)
    users_match_weight = max(float(users_match_weight), 0.0)
    treated = treated.sort_values(["naics2", "logit_score", "cum_z", "c"], ascending=[True, False, False, True]).copy()
    control_groups = {
        str(naics2): group.sort_values(["logit_score", "c"], ascending=[False, True]).reset_index(drop=True)
        for naics2, group in control.groupby("naics2", sort=False)
    }
    processed_treated = 0
    region_penalty = 1e-6
    for naics2, treated_block in treated.groupby("naics2", sort=False):
        treated_block = treated_block.reset_index(drop=True)
        control_block = control_groups.get(str(naics2))
        if control_block is None or control_block.empty:
            unmatched_reasons["no_exact_naics2_control"] += int(len(treated_block))
            processed_treated += int(len(treated_block))
            continue
        control_logit = pd.to_numeric(control_block["logit_score"], errors="coerce").to_numpy(dtype=float)
        control_effective_size = pd.to_numeric(control_block["effective_size"], errors="coerce").to_numpy(dtype=float)
        control_users = pd.to_numeric(control_block["matching_users_value"], errors="coerce").to_numpy(dtype=float)
        control_growth = (
            pd.to_numeric(control_block[growth_match_feature], errors="coerce").to_numpy(dtype=float)
            if growth_match_feature in control_block.columns
            else np.full(len(control_block), np.nan, dtype=float)
        )
        control_region = control_block["company_hq_region"].astype(str).to_numpy(dtype=object)
        candidate_positions: list[np.ndarray] = []
        candidate_costs: list[np.ndarray] = []
        for _, treated_row in treated_block.iterrows():
            processed_treated += 1
            candidate_pool_total += int(len(control_block))
            score_distance = np.abs(control_logit - float(treated_row["logit_score"]))
            within_caliper = score_distance <= float(caliper)
            within_caliper_total += int(within_caliper.sum())
            if not bool(within_caliper.any()):
                unmatched_reasons["no_control_within_caliper"] += 1
                candidate_positions.append(np.array([], dtype=int))
                candidate_costs.append(np.array([], dtype=float))
            else:
                treated_growth = (
                    pd.to_numeric(treated_row.get(growth_match_feature, np.nan), errors="coerce")
                    if growth_match_feature in treated_block.columns
                    else np.nan
                )
                feasible = within_caliper.copy()
                if growth_match_weight > 0 and pd.notna(treated_growth):
                    feasible &= np.isfinite(control_growth)
                positions = np.flatnonzero(feasible)
                if not len(positions):
                    unmatched_reasons["no_feasible_edge_after_constraints"] += 1
                    candidate_positions.append(np.array([], dtype=int))
                    candidate_costs.append(np.array([], dtype=float))
                else:
                    size_distance = np.abs(control_effective_size[positions] - float(treated_row["effective_size"]))
                    users_distance = np.abs(control_users[positions] - float(treated_row["matching_users_value"]))
                    if growth_match_weight > 0 and pd.notna(treated_growth):
                        growth_distance = np.abs(control_growth[positions] - float(treated_growth))
                    else:
                        growth_distance = np.zeros(len(positions), dtype=float)
                    total_cost = score_distance[positions].astype(float)
                    if size_match_weight > 0:
                        total_cost += float(size_match_weight) * (size_distance / float(size_scale))
                    if growth_match_weight > 0 and pd.notna(treated_growth):
                        total_cost += float(growth_match_weight) * (growth_distance / float(growth_scale))
                    if users_match_weight > 0:
                        total_cost += float(users_match_weight) * (users_distance / float(users_scale))
                    total_cost += region_penalty * (
                        control_region[positions] != str(treated_row["company_hq_region"])
                    ).astype(float)
                    feasible_edge_total += int(len(positions))
                    candidate_positions.append(positions.astype(int, copy=False))
                    candidate_costs.append(total_cost.astype(float, copy=False))
            if progress_interval is not None and (
                processed_treated == 1 or processed_treated % progress_interval == 0 or processed_treated == n_treated
            ):
                elapsed = time.perf_counter() - matching_started
                rate = processed_treated / elapsed if elapsed > 0 else math.nan
                remaining = n_treated - processed_treated
                eta_seconds = remaining / rate if rate and rate > 0 else math.nan
                avg_candidates = candidate_pool_total / processed_treated if processed_treated else 0.0
                avg_within_caliper = within_caliper_total / processed_treated if processed_treated else 0.0
                avg_after_constraints = feasible_edge_total / processed_treated if processed_treated else 0.0
                _log_step(
                    "matching progress | "
                    f"processed={processed_treated:,}/{n_treated:,} matched={len(pairs):,} "
                    f"avg_candidates_before_caliper={avg_candidates:.1f} "
                    f"avg_candidates_within_caliper={avg_within_caliper:.1f} "
                    f"avg_candidates_after_constraints={avg_after_constraints:.1f} "
                    f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s"
                )
        assignments, assignment_diag = _solve_optimal_match_assignment(candidate_positions, candidate_costs)
        unmatched_reasons["unmatched_after_optimal_assignment"] += int(
            sum(len(positions) > 0 for positions in candidate_positions) - len(assignments)
        )
        for treated_pos, control_pos, assigned_cost in assignments:
            treated_row = treated_block.iloc[int(treated_pos)]
            chosen = control_block.iloc[int(control_pos)]
            treated_growth = (
                pd.to_numeric(treated_row.get(growth_match_feature, np.nan), errors="coerce")
                if growth_match_feature in treated_block.columns
                else np.nan
            )
            control_growth_value = (
                pd.to_numeric(chosen.get(growth_match_feature, np.nan), errors="coerce")
                if growth_match_feature in control_block.columns
                else np.nan
            )
            pair_id = len(pairs) + 1
            pairs.append(
                {
                    "pair_id": pair_id,
                    "treated_c": int(treated_row["c"]),
                    "control_c": int(chosen["c"]),
                    "naics2": str(naics2),
                    "treated_logit_score": float(treated_row["logit_score"]),
                    "control_logit_score": float(chosen["logit_score"]),
                    "treated_predicted_prob": float(treated_row["predicted_prob"]),
                    "control_predicted_prob": float(chosen["predicted_prob"]),
                    "score_distance": float(abs(float(chosen["logit_score"]) - float(treated_row["logit_score"]))),
                    "size_distance": float(abs(float(chosen["effective_size"]) - float(treated_row["effective_size"]))),
                    "growth_distance": (
                        float(abs(float(control_growth_value) - float(treated_growth)))
                        if pd.notna(treated_growth) and pd.notna(control_growth_value)
                        else np.nan
                    ),
                    "users_distance": float(
                        abs(float(chosen["matching_users_value"]) - float(treated_row["matching_users_value"]))
                    ),
                    "match_distance": float(assigned_cost),
                    "same_region": int(str(chosen["company_hq_region"]) == str(treated_row["company_hq_region"])),
                    "control_source": (
                        "outside_negative"
                        if int(chosen.get("outside_negative_candidate", 0) or 0) == 1
                        else "preferred_low_exposure"
                    ),
                    "caliper": float(caliper),
                    "support_lower": float(lower),
                    "support_upper": float(upper),
                    "effective_treated_size": float(treated_row["effective_size"]),
                    "effective_control_size": float(chosen["effective_size"]),
                    "treated_coverage_years": float(pd.to_numeric(treated_row["matching_coverage_years"], errors="coerce")),
                    "control_coverage_years": float(pd.to_numeric(chosen["matching_coverage_years"], errors="coerce")),
                    "propensity_design": str(propensity_design),
                    "matching_algorithm": str(matching_algorithm),
                }
            )
    pair_df = pd.DataFrame(pairs)
    control_source_counts = pair_df["control_source"].value_counts().to_dict() if not pair_df.empty else {}
    elapsed_seconds = time.perf_counter() - matching_started
    avg_candidates_before_caliper = float(candidate_pool_total / n_treated) if n_treated else 0.0
    avg_candidates_within_caliper = float(within_caliper_total / n_treated) if n_treated else 0.0
    avg_candidates_after_constraints = float(feasible_edge_total / n_treated) if n_treated else 0.0
    diagnostics = {
        "n_treated_before_support": int(len(treated_pre)),
        "n_controls_before_support": int(len(control_pre)),
        "n_treated_after_support": int(len(treated_support)),
        "n_controls_after_support": int(len(control_support)),
        "n_pairs": int(len(pair_df)),
        "n_treated_unmatched": int(len(treated) - len(pair_df)),
        "caliper": float(caliper),
        "support_lower": float(lower),
        "support_upper": float(upper),
        "elapsed_seconds": float(elapsed_seconds),
        "progress_every": None if progress_interval is None else int(progress_interval),
        "n_overlap_naics2": int(search_space["n_overlap_naics2"]),
        "estimated_candidate_pairs": int(search_space["estimated_candidate_pairs"]),
        "max_controls_in_overlap_naics2": int(search_space["max_controls_in_overlap_naics2"]),
        "median_controls_in_treated_naics2": float(search_space["median_controls_in_treated_naics2"]),
        "avg_candidates_before_caliper": avg_candidates_before_caliper,
        "avg_candidates_within_caliper": avg_candidates_within_caliper,
        "avg_candidates_after_constraints": avg_candidates_after_constraints,
        "n_pairs_preferred_low_exposure": int(control_source_counts.get("preferred_low_exposure", 0)),
        "n_pairs_outside_negative": int(control_source_counts.get("outside_negative", 0)),
        "share_pairs_outside_negative": (
            float(control_source_counts.get("outside_negative", 0) / len(pair_df))
            if len(pair_df)
            else 0.0
        ),
        "size_match_feature": size_match_feature,
        "growth_match_feature": growth_match_feature,
        "users_match_feature": users_match_feature,
        "coverage_feature": coverage_feature,
        "coverage_rule": coverage_rule,
        "coverage_required_years": int(coverage_required_years),
        "matching_algorithm": str(matching_algorithm),
        "propensity_design": str(propensity_design),
        "size_match_weight": size_match_weight,
        "growth_match_weight": growth_match_weight,
        "users_match_weight": users_match_weight,
        **constraint_diag,
        **assignment_diag,
        **unmatched_reasons,
    }
    _log_step(
        "matching done | "
        f"pairs={len(pair_df):,} treated_unmatched={int(len(treated) - len(pair_df)):,} "
        f"elapsed={elapsed_seconds:.1f}s no_exact_naics2_control={unmatched_reasons['no_exact_naics2_control']:,} "
        f"no_control_within_caliper={unmatched_reasons['no_control_within_caliper']:,} "
        f"no_feasible_edge_after_constraints={unmatched_reasons['no_feasible_edge_after_constraints']:,} "
        f"unmatched_after_optimal_assignment={unmatched_reasons['unmatched_after_optimal_assignment']:,}"
    )
    return pair_df, diagnostics


def _balance_stat_row(
    treated_values: pd.Series,
    control_values: pd.Series,
    *,
    trajectory_name: str,
    match_stage: str,
    control_source: str,
    covariate: str,
    covariate_group: str,
    covariate_type: str,
) -> Optional[dict[str, object]]:
    treated_numeric = pd.to_numeric(treated_values, errors="coerce")
    control_numeric = pd.to_numeric(control_values, errors="coerce")
    n_treated = int(treated_numeric.notna().sum())
    n_control = int(control_numeric.notna().sum())
    if n_treated == 0 or n_control == 0:
        return None
    treated_mean = float(treated_numeric.mean())
    control_mean = float(control_numeric.mean())
    mean_diff = treated_mean - control_mean
    treated_var = float(treated_numeric.var(ddof=1)) if n_treated > 1 else 0.0
    control_var = float(control_numeric.var(ddof=1)) if n_control > 1 else 0.0
    pooled_var_num = max(0.0, ((n_treated - 1) * treated_var) + ((n_control - 1) * control_var))
    pooled_var_den = max(0, n_treated + n_control - 2)
    pooled_var = pooled_var_num / pooled_var_den if pooled_var_den > 0 else 0.0
    pooled_sd = math.sqrt(max(0.0, pooled_var))
    standardized_mean_diff = mean_diff / pooled_sd if pooled_sd > 0 else (0.0 if mean_diff == 0 else np.nan)
    variance_ratio = treated_var / control_var if control_var > 0 else np.nan
    se_diff = math.sqrt(max(0.0, treated_var / max(n_treated, 1) + control_var / max(n_control, 1)))
    z_stat = mean_diff / se_diff if se_diff > 0 else (0.0 if mean_diff == 0 else np.nan)
    p_value = math.erfc(abs(z_stat) / math.sqrt(2.0)) if np.isfinite(z_stat) else np.nan
    return {
        "trajectory_name": trajectory_name,
        "match_stage": match_stage,
        "control_source": control_source,
        "covariate": covariate,
        "covariate_group": covariate_group,
        "covariate_type": covariate_type,
        "n_treated": n_treated,
        "n_control": n_control,
        "treated_mean": treated_mean,
        "control_mean": control_mean,
        "mean_diff": mean_diff,
        "pooled_sd": pooled_sd,
        "standardized_mean_diff": standardized_mean_diff,
        "treated_variance": treated_var,
        "control_variance": control_var,
        "variance_ratio": variance_ratio,
        "mean_diff_z": z_stat,
        "mean_diff_pvalue": p_value,
    }


def _build_balance_rows(
    treated_df: pd.DataFrame,
    control_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    trajectory_name: str,
    match_stage: str,
    control_source: str,
) -> pd.DataFrame:
    if treated_df.empty or control_df.empty:
        return pd.DataFrame()
    available_feature_cols = [str(col) for col in feature_cols if str(col) in treated_df.columns and str(col) in control_df.columns]
    categorical_cols, numeric_cols = _infer_feature_types(treated_df, available_feature_cols)
    rows: list[dict[str, object]] = []

    for col in numeric_cols:
        row = _balance_stat_row(
            treated_df[col],
            control_df[col],
            trajectory_name=trajectory_name,
            match_stage=match_stage,
            control_source=control_source,
            covariate=str(col),
            covariate_group=str(col),
            covariate_type="numeric",
        )
        if row is not None:
            rows.append(row)

    for col in categorical_cols:
        treated_levels = (
            treated_df[col].astype("string").fillna("__MISSING__").replace({"": "__MISSING__"})
        )
        control_levels = (
            control_df[col].astype("string").fillna("__MISSING__").replace({"": "__MISSING__"})
        )
        levels = sorted(set(treated_levels.unique().tolist()) | set(control_levels.unique().tolist()))
        for level in levels:
            row = _balance_stat_row(
                treated_levels.eq(level).astype(float),
                control_levels.eq(level).astype(float),
                trajectory_name=trajectory_name,
                match_stage=match_stage,
                control_source=control_source,
                covariate=f"{col}={level}",
                covariate_group=str(col),
                covariate_type="categorical_indicator",
            )
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def summarize_balance_table(balance_df: pd.DataFrame) -> pd.DataFrame:
    if balance_df.empty:
        return pd.DataFrame(
            columns=[
                "trajectory_name",
                "match_stage",
                "control_source",
                "n_covariates",
                "max_abs_smd",
                "mean_abs_smd",
                "n_abs_smd_gt_0_10",
                "n_abs_smd_gt_0_25",
                "share_abs_smd_gt_0_10",
                "share_abs_smd_gt_0_25",
            ]
        )

    summary_rows: list[dict[str, object]] = []
    for keys, group in balance_df.groupby(["trajectory_name", "match_stage", "control_source"], dropna=False):
        abs_smd = pd.to_numeric(group["standardized_mean_diff"], errors="coerce").abs().dropna()
        n_covariates = int(len(abs_smd))
        n_abs_smd_gt_010 = int((abs_smd > 0.10).sum())
        n_abs_smd_gt_025 = int((abs_smd > 0.25).sum())
        summary_rows.append(
            {
                "trajectory_name": str(keys[0]),
                "match_stage": str(keys[1]),
                "control_source": str(keys[2]),
                "n_covariates": n_covariates,
                "max_abs_smd": float(abs_smd.max()) if not abs_smd.empty else np.nan,
                "mean_abs_smd": float(abs_smd.mean()) if not abs_smd.empty else np.nan,
                "n_abs_smd_gt_0_10": n_abs_smd_gt_010,
                "n_abs_smd_gt_0_25": n_abs_smd_gt_025,
                "share_abs_smd_gt_0_10": (float(n_abs_smd_gt_010 / n_covariates) if n_covariates else np.nan),
                "share_abs_smd_gt_0_25": (float(n_abs_smd_gt_025 / n_covariates) if n_covariates else np.nan),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(
        ["trajectory_name", "match_stage", "control_source"]
    ).reset_index(drop=True)


def build_matching_balance_tables(
    classified_trajectory: pd.DataFrame,
    feature_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    *,
    include_outside_negative_controls: bool,
    size_match_feature: str = "match_firm_size_pre_level",
    growth_match_feature: str = "match_firm_size_pre_growth",
    users_match_feature: str = "company_n_users_log1p",
    coverage_feature: str = "firm_size_annual_pre_n_years",
    coverage_required_years: int = 4,
    users_feature_required: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = _build_matching_work_frame(
        classified_trajectory,
        feature_df,
        propensity_df,
        size_match_feature=size_match_feature,
        users_match_feature=users_match_feature,
        coverage_feature=coverage_feature,
    )
    (
        _,
        _,
        _,
        _,
        treated_eligible,
        control_eligible,
        _,
        _,
        _,
    ) = _matching_eligible_samples(
        work,
        include_outside_negative_controls=include_outside_negative_controls,
        coverage_required_years=int(coverage_required_years),
        users_feature_required=bool(users_feature_required),
    )
    trajectory_name = str(classified_trajectory["trajectory_name"].iloc[0]) if not classified_trajectory.empty else "unknown"
    feature_cols = _select_index_feature_columns(feature_df, feature_family=MATCHING_FEATURE_FAMILY)
    balance_feature_cols = [
        col for col in feature_cols if col in work.columns
    ] + [
        col
        for col in (
            "predicted_prob",
            "logit_score",
            users_match_feature,
            size_match_feature,
            growth_match_feature,
            coverage_feature,
            "effective_size",
        )
        if col in work.columns
    ]
    balance_feature_cols = list(dict.fromkeys(balance_feature_cols))

    rows: list[pd.DataFrame] = []
    control_subsets_pre = {
        "overall": control_eligible,
        "preferred_low_exposure": control_eligible.loc[
            control_eligible["outside_negative_candidate"].fillna(0).ne(1)
        ].copy(),
    }
    outside_controls = control_eligible.loc[
        control_eligible["outside_negative_candidate"].fillna(0).eq(1)
    ].copy()
    if include_outside_negative_controls and not outside_controls.empty:
        control_subsets_pre["outside_negative"] = outside_controls
    for control_source, control_df in control_subsets_pre.items():
        balance_rows = _build_balance_rows(
            treated_eligible,
            control_df,
            feature_cols=balance_feature_cols,
            trajectory_name=trajectory_name,
            match_stage="pre_match",
            control_source=control_source,
        )
        if not balance_rows.empty:
            rows.append(balance_rows)

    work_indexed = work.drop_duplicates(subset=["c"]).set_index("c", drop=False)
    if not pair_df.empty:
        matched_treated = work_indexed.loc[
            sorted(pd.unique(pd.to_numeric(pair_df["treated_c"], errors="coerce").dropna().astype(int)))
        ].reset_index(drop=True)
        matched_control = work_indexed.loc[
            sorted(pd.unique(pd.to_numeric(pair_df["control_c"], errors="coerce").dropna().astype(int)))
        ].reset_index(drop=True)
        overall_post = _build_balance_rows(
            matched_treated,
            matched_control,
            feature_cols=balance_feature_cols,
            trajectory_name=trajectory_name,
            match_stage="post_match",
            control_source="overall",
        )
        if not overall_post.empty:
            rows.append(overall_post)
        for control_source in ("preferred_low_exposure", "outside_negative"):
            pair_subset = pair_df.loc[pair_df["control_source"].eq(control_source)].copy()
            if pair_subset.empty:
                continue
            treated_subset = work_indexed.loc[
                sorted(pd.unique(pd.to_numeric(pair_subset["treated_c"], errors="coerce").dropna().astype(int)))
            ].reset_index(drop=True)
            control_subset = work_indexed.loc[
                sorted(pd.unique(pd.to_numeric(pair_subset["control_c"], errors="coerce").dropna().astype(int)))
            ].reset_index(drop=True)
            balance_rows = _build_balance_rows(
                treated_subset,
                control_subset,
                feature_cols=balance_feature_cols,
                trajectory_name=trajectory_name,
                match_stage="post_match",
                control_source=control_source,
            )
            if not balance_rows.empty:
                rows.append(balance_rows)

    balance_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    balance_summary = summarize_balance_table(balance_table)
    return balance_table, balance_summary


def build_matched_panel(
    panel: pd.DataFrame,
    pair_df: pd.DataFrame,
    *,
    trajectory_name: str,
) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()
    treated_panel = pair_df[["pair_id", "treated_c"]].rename(columns={"treated_c": "c"}).merge(
        panel,
        on="c",
        how="left",
    )
    treated_panel["high_exposure"] = 1
    treated_panel["treated"] = 1
    control_panel = pair_df[["pair_id", "control_c"]].rename(columns={"control_c": "c"}).merge(
        panel,
        on="c",
        how="left",
    )
    control_panel["high_exposure"] = 0
    control_panel["treated"] = 0
    out = pd.concat([treated_panel, control_panel], ignore_index=True)
    out["trajectory_name"] = trajectory_name
    return out.sort_values(["pair_id", "c", "t"]).reset_index(drop=True)


def run_common_break_event_study(
    matched_panel: pd.DataFrame,
    *,
    outcome_cols: Sequence[str],
    ref_year: int,
) -> pd.DataFrame:
    if pf is None:  # pragma: no cover
        raise ImportError("pyfixest is required for the matched exposure regressions.")
    results: list[dict[str, object]] = []
    for outcome_col in outcome_cols:
        work = matched_panel.dropna(subset=[outcome_col]).drop_duplicates(subset=["pair_id", "c", "t"]).copy()
        if work.empty:
            continue
        years = sorted(int(y) for y in pd.unique(work["t"]))
        interaction_cols: list[str] = []
        for year in years:
            if year == int(ref_year):
                continue
            col = f"event_high_{year}"
            work[col] = ((work["t"] == year) & work["high_exposure"].eq(1)).astype(int)
            interaction_cols.append(col)
        if not interaction_cols:
            continue
        fit = pf.feols(
            f"{outcome_col} ~ {' + '.join(interaction_cols)} | c + t",
            data=work,
            vcov={"CRV1": "pair_id"},
            demeaner_backend="rust",
        )
        tidy = fit.tidy()
        for year in years:
            if year == int(ref_year):
                results.append(
                    {
                        "trajectory_name": str(work["trajectory_name"].iloc[0]),
                        "outcome_col": outcome_col,
                        "year": int(year),
                        "coef": 0.0,
                        "se": 0.0,
                        "n_pairs": int(work["pair_id"].nunique()),
                        "n_obs": int(len(work)),
                    }
                )
                continue
            col = f"event_high_{year}"
            if col not in tidy.index:
                continue
            results.append(
                {
                    "trajectory_name": str(work["trajectory_name"].iloc[0]),
                    "outcome_col": outcome_col,
                    "year": int(year),
                    "coef": float(tidy.loc[col, "Estimate"]),
                    "se": float(tidy.loc[col, "Std. Error"]),
                    "n_pairs": int(work["pair_id"].nunique()),
                    "n_obs": int(len(work)),
                }
            )
    return pd.DataFrame(results).sort_values(["trajectory_name", "outcome_col", "year"]).reset_index(drop=True)


def assign_persistent_entry_cohorts(
    panel: pd.DataFrame,
    *,
    cohort_min_year: int,
    cohort_max_year: int,
    pre_years_required: int,
    forward_window_years: int,
    min_positive_years_in_forward_window: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for firm_id, firm_df in panel.groupby("c", sort=True):
        z_by_year = (
            firm_df.loc[:, ["t", "z_ct"]]
            .assign(z_ct=lambda df: pd.to_numeric(df["z_ct"], errors="coerce").fillna(0.0))
            .drop_duplicates(subset=["t"])
            .set_index("t")["z_ct"]
            .to_dict()
        )
        cohort_year: Optional[int] = None
        for year in range(int(cohort_min_year), int(cohort_max_year) + 1):
            pre_years = [year - lag for lag in range(pre_years_required, 0, -1)]
            if any(z_by_year.get(pre_year, 0.0) > 0 for pre_year in pre_years):
                continue
            forward_years = list(range(year, year + int(forward_window_years)))
            positive_years = sum(z_by_year.get(forward_year, 0.0) > 0 for forward_year in forward_years)
            if z_by_year.get(year, 0.0) <= 0:
                continue
            if positive_years < int(min_positive_years_in_forward_window):
                continue
            cohort_year = year
            break
        rows.append({"c": int(firm_id), "g": cohort_year})
    return pd.DataFrame(rows)


def build_stacked_did_panel(
    matched_panel: pd.DataFrame,
    pair_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    *,
    pre_window: int,
    post_window: int,
) -> tuple[pd.DataFrame, dict]:
    if pair_df.empty or cohort_df.empty:
        return pd.DataFrame(), {"n_candidate_pairs": 0, "n_stacks": 0, "dropped_controls_turn_high": 0}
    cohort_lookup = cohort_df.dropna(subset=["g"]).copy()
    if cohort_lookup.empty:
        return pd.DataFrame(), {"n_candidate_pairs": int(len(pair_df)), "n_stacks": 0, "dropped_controls_turn_high": 0}
    pair_cohorts = pair_df.merge(
        cohort_lookup.rename(columns={"c": "treated_c", "g": "treated_g"}),
        on="treated_c",
        how="inner",
    )
    if pair_cohorts.empty:
        return pd.DataFrame(), {"n_candidate_pairs": int(len(pair_df)), "n_stacks": 0, "dropped_controls_turn_high": 0}
    control_z = matched_panel.loc[matched_panel["treated"].eq(0), ["pair_id", "t", "z_ct"]].copy()
    control_z["z_ct"] = pd.to_numeric(control_z["z_ct"], errors="coerce").fillna(0.0)

    stack_frames: list[pd.DataFrame] = []
    dropped_controls_turn_high = 0
    for _, pair_row in pair_cohorts.iterrows():
        g = int(pair_row["treated_g"])
        window_years = list(range(g - int(pre_window), g + int(post_window) + 1))
        control_window = control_z.loc[
            control_z["pair_id"].eq(int(pair_row["pair_id"])) & control_z["t"].isin(window_years)
        ].copy()
        if len(control_window) != len(window_years) or bool((control_window["z_ct"] > 0).any()):
            dropped_controls_turn_high += 1
            continue
        pair_panel = matched_panel.loc[
            matched_panel["pair_id"].eq(int(pair_row["pair_id"])) & matched_panel["t"].isin(window_years)
        ].copy()
        if pair_panel.empty or pair_panel["t"].nunique() != len(window_years):
            continue
        stack_id = int(pair_row["pair_id"])
        pair_panel["stack_id"] = stack_id
        pair_panel["g"] = g
        pair_panel["rel_time"] = pair_panel["t"] - g
        pair_panel["pair_stack_fe"] = pair_panel["pair_id"].astype(str) + "__" + pair_panel["stack_id"].astype(str)
        pair_panel["unit_stack_fe"] = pair_panel["c"].astype(str) + "__" + pair_panel["stack_id"].astype(str)
        pair_panel["year_stack_fe"] = pair_panel["t"].astype(str) + "__" + pair_panel["stack_id"].astype(str)
        stack_frames.append(pair_panel)
    if not stack_frames:
        return pd.DataFrame(), {
            "n_candidate_pairs": int(len(pair_cohorts)),
            "n_stacks": 0,
            "dropped_controls_turn_high": int(dropped_controls_turn_high),
        }
    stacked = pd.concat(stack_frames, ignore_index=True)
    diagnostics = {
        "n_candidate_pairs": int(len(pair_cohorts)),
        "n_stacks": int(stacked["stack_id"].nunique()),
        "dropped_controls_turn_high": int(dropped_controls_turn_high),
    }
    return stacked.sort_values(["stack_id", "c", "t"]).reset_index(drop=True), diagnostics


def run_stacked_did(
    stacked_panel: pd.DataFrame,
    *,
    outcome_cols: Sequence[str],
    ref_event_time: int = -1,
) -> pd.DataFrame:
    if stacked_panel.empty:
        return pd.DataFrame()
    if pf is None:  # pragma: no cover
        raise ImportError("pyfixest is required for the matched exposure regressions.")
    results: list[dict[str, object]] = []
    event_times = sorted(int(v) for v in pd.unique(stacked_panel["rel_time"]))
    non_ref_event_times = [event_time for event_time in event_times if event_time != int(ref_event_time)]
    for outcome_col in outcome_cols:
        work = stacked_panel.dropna(subset=[outcome_col]).drop_duplicates(
            subset=["stack_id", "pair_id", "c", "t"]
        ).copy()
        if work.empty:
            continue
        if "unit_stack_fe" not in work.columns and {"c", "stack_id"}.issubset(work.columns):
            work["unit_stack_fe"] = work["c"].astype(str) + "__" + work["stack_id"].astype(str)
        if "year_stack_fe" not in work.columns and {"t", "stack_id"}.issubset(work.columns):
            work["year_stack_fe"] = work["t"].astype(str) + "__" + work["stack_id"].astype(str)
        if "unit_stack_fe" not in work.columns or "year_stack_fe" not in work.columns:
            raise ValueError(
                "stacked_panel must include unit_stack_fe and year_stack_fe, "
                "or enough columns to construct them from c/stack_id and t/stack_id."
            )
        interaction_cols: list[str] = []
        for event_time in non_ref_event_times:
            suffix = f"m{abs(event_time)}" if event_time < 0 else f"p{event_time}"
            col = f"stack_treated_{suffix}"
            work[col] = ((work["rel_time"] == event_time) & work["treated"].eq(1)).astype(int)
            interaction_cols.append(col)
        if not interaction_cols:
            continue
        fit = pf.feols(
            f"{outcome_col} ~ {' + '.join(interaction_cols)} | unit_stack_fe + year_stack_fe",
            data=work,
            vcov={"CRV1": "pair_id"},
            demeaner_backend="rust",
        )
        tidy = fit.tidy()
        for event_time in event_times:
            if event_time == int(ref_event_time):
                results.append(
                    {
                        "trajectory_name": str(work["trajectory_name"].iloc[0]),
                        "outcome_col": outcome_col,
                        "rel_time": int(event_time),
                        "coef": 0.0,
                        "se": 0.0,
                        "n_pairs": int(work["pair_id"].nunique()),
                        "n_stacks": int(work["stack_id"].nunique()),
                        "n_obs": int(len(work)),
                    }
                )
                continue
            suffix = f"m{abs(event_time)}" if event_time < 0 else f"p{event_time}"
            col = f"stack_treated_{suffix}"
            if col not in tidy.index:
                continue
            results.append(
                {
                    "trajectory_name": str(work["trajectory_name"].iloc[0]),
                    "outcome_col": outcome_col,
                    "rel_time": int(event_time),
                    "coef": float(tidy.loc[col, "Estimate"]),
                    "se": float(tidy.loc[col, "Std. Error"]),
                    "n_pairs": int(work["pair_id"].nunique()),
                    "n_stacks": int(work["stack_id"].nunique()),
                    "n_obs": int(len(work)),
                }
            )
    return pd.DataFrame(results).sort_values(["trajectory_name", "outcome_col", "rel_time"]).reset_index(drop=True)


def _build_trajectory_group_counts(trajectory_summary: pd.DataFrame) -> pd.DataFrame:
    if trajectory_summary.empty:
        return pd.DataFrame()
    work = trajectory_summary.copy()
    grouped = (
        work.groupby(
            [
                "trajectory_name",
                "window_start",
                "window_end",
                "exposure_group",
                "preferred_rcid_source",
                "outside_negative_candidate",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_firms=("c", "nunique"),
            mean_cum_z=("cum_z", "mean"),
            mean_max_z=("max_z", "mean"),
            mean_active_share=("active_share", "mean"),
        )
        .sort_values(
            [
                "trajectory_name",
                "preferred_rcid_source",
                "outside_negative_candidate",
                "exposure_group",
            ]
        )
        .reset_index(drop=True)
    )
    return grouped


def _build_propensity_group_summary(
    trajectory_summary: pd.DataFrame,
    propensity_scores: pd.DataFrame,
) -> pd.DataFrame:
    if trajectory_summary.empty or propensity_scores.empty:
        return pd.DataFrame()
    merge_keys = ["c", "trajectory_name"] if "trajectory_name" in propensity_scores.columns else ["c"]
    score_cols = merge_keys + [
        col for col in ("predicted_prob", "predicted_index", "propensity_design") if col in propensity_scores.columns
    ]
    work = trajectory_summary.loc[:, ["c", "trajectory_name", "exposure_group"]].merge(
        propensity_scores.loc[:, score_cols],
        on=merge_keys,
        how="left",
    )
    if "predicted_prob" not in work.columns:
        return pd.DataFrame()
    agg_kwargs: dict[str, tuple[str, object]] = {
        "n_firms": ("c", "nunique"),
        "mean_predicted_prob": ("predicted_prob", "mean"),
        "median_predicted_prob": ("predicted_prob", "median"),
        "p25_predicted_prob": ("predicted_prob", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.25))),
        "p75_predicted_prob": ("predicted_prob", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.75))),
    }
    if "propensity_design" in work.columns:
        agg_kwargs["propensity_design"] = ("propensity_design", "first")
    grouped = (
        work.groupby(["trajectory_name", "exposure_group"], dropna=False, as_index=False)
        .agg(**agg_kwargs)
        .sort_values(["trajectory_name", "exposure_group"])
        .reset_index(drop=True)
    )
    return grouped


def _build_propensity_score_distribution_table(
    trajectory_summary: pd.DataFrame,
    propensity_scores: pd.DataFrame,
) -> pd.DataFrame:
    if trajectory_summary.empty or propensity_scores.empty:
        return pd.DataFrame()
    merge_keys = ["c", "trajectory_name"] if "trajectory_name" in propensity_scores.columns else ["c"]
    score_cols = merge_keys + [
        col for col in ("predicted_prob", "predicted_index", "propensity_design") if col in propensity_scores.columns
    ]
    base_cols = [
        "c",
        "trajectory_name",
        "exposure_group",
        "preferred_rcid_source",
        "outside_negative_candidate",
    ]
    work = trajectory_summary.loc[:, [col for col in base_cols if col in trajectory_summary.columns]].merge(
        propensity_scores.loc[:, score_cols],
        on=merge_keys,
        how="left",
    )
    if "predicted_prob" not in work.columns:
        return pd.DataFrame()
    work["source_group"] = np.select(
        [
            work["preferred_rcid_source"].fillna(0).astype(int).eq(1),
            work["outside_negative_candidate"].fillna(0).astype(int).eq(1),
        ],
        ["preferred_rcid_source", "outside_negative"],
        default="other_source",
    )
    group_cols = ["trajectory_name", "exposure_group", "source_group"]
    if "propensity_design" in work.columns:
        group_cols.append("propensity_design")

    def _quantile(prob: pd.Series, q: float) -> float:
        return float(pd.to_numeric(prob, errors="coerce").quantile(q))

    return (
        work.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n_firms=("c", "nunique"),
            mean_predicted_prob=("predicted_prob", "mean"),
            sd_predicted_prob=("predicted_prob", "std"),
            min_predicted_prob=("predicted_prob", "min"),
            p10_predicted_prob=("predicted_prob", lambda s: _quantile(s, 0.10)),
            p25_predicted_prob=("predicted_prob", lambda s: _quantile(s, 0.25)),
            median_predicted_prob=("predicted_prob", "median"),
            p75_predicted_prob=("predicted_prob", lambda s: _quantile(s, 0.75)),
            p90_predicted_prob=("predicted_prob", lambda s: _quantile(s, 0.90)),
            max_predicted_prob=("predicted_prob", "max"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def _iter_propensity_diagnostics(diagnostics: dict[str, object]) -> Iterable[tuple[Optional[str], dict[str, object]]]:
    top_level = diagnostics.get("propensity_diagnostics")
    if isinstance(top_level, dict):
        yield None, top_level
    trajectory_specs = diagnostics.get("trajectory_specs", {})
    if not isinstance(trajectory_specs, dict):
        return
    for trajectory_name, spec_payload in trajectory_specs.items():
        if not isinstance(spec_payload, dict):
            continue
        propensity_payload = spec_payload.get("propensity", {})
        if isinstance(propensity_payload, dict):
            yield str(trajectory_name), propensity_payload


def _build_propensity_model_diagnostics_table(diagnostics: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scalar_keys = [
        "propensity_design",
        "training_sample_mode",
        "model_method",
        "index_entry_mode",
        "feature_family",
        "logit_class_weight",
        "train_target_mean",
        "n_train_obs",
        "n_train_preferred_source",
        "n_train_outside_negative",
        "n_event_study_firms",
        "n_event_study_preferred_source",
        "n_event_study_outside_negative",
        "n_feature_columns_raw",
        "n_numeric_feature_columns_raw",
        "n_categorical_feature_columns_raw",
        "n_active_features",
        "n_standardized_features",
        "n_interaction_columns_added",
        "n_numeric_interaction_columns_added",
        "n_category_slope_interaction_columns_added",
        "feature_downsampled",
        "n_active_features_before_sampling",
        "n_active_features_after_sampling",
        "max_active_features",
        "max_feature_to_train_ratio",
        "selected_feature_seed",
        "lasso_cv_folds",
        "lasso_n_cs",
        "lasso_selected_c",
        "intercept",
        "evaluation_n",
        "evaluation_target_mean",
        "evaluation_auc",
        "evaluation_brier",
        "evaluation_class_1_share",
    ]
    for trajectory_name, payload in _iter_propensity_diagnostics(diagnostics):
        row = {"trajectory_name": trajectory_name}
        for key in scalar_keys:
            value = payload.get(key)
            if isinstance(value, (dict, list, tuple, set)):
                continue
            row[key] = value
        skipped = payload.get("skipped_interaction_source_columns", [])
        row["skipped_interaction_source_columns"] = (
            "; ".join(str(value) for value in skipped)
            if isinstance(skipped, (list, tuple))
            else skipped
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_propensity_feature_weight_table(diagnostics: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trajectory_name, payload in _iter_propensity_diagnostics(diagnostics):
        feature_weights = payload.get("feature_weights")
        if isinstance(feature_weights, list):
            for row in feature_weights:
                if not isinstance(row, dict):
                    continue
                out_row = dict(row)
                if out_row.get("trajectory_name") is None:
                    out_row["trajectory_name"] = trajectory_name
                rows.append(out_row)
            continue
        rows.extend(
            _propensity_feature_weight_records(
                payload,
                {},
                propensity_design=str(payload.get("propensity_design", diagnostics.get("propensity_design", "unknown"))),
                trajectory_name=trajectory_name,
            )
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if "abs_weight" not in out.columns and "weight" in out.columns:
        out["abs_weight"] = pd.to_numeric(out["weight"], errors="coerce").abs()
    return out.sort_values(
        ["trajectory_name", "propensity_design", "rank_abs_weight", "feature"],
        na_position="first",
    ).reset_index(drop=True)


def _build_propensity_feature_group_weight_table(feature_weights: pd.DataFrame) -> pd.DataFrame:
    if feature_weights.empty:
        return pd.DataFrame()
    work = feature_weights.copy()
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    work["abs_weight"] = pd.to_numeric(work["abs_weight"], errors="coerce")
    work = work.dropna(subset=["raw_feature", "abs_weight"])
    if work.empty:
        return pd.DataFrame()
    group_cols = ["trajectory_name", "propensity_design", "model_method", "weight_kind", "raw_feature"]
    idx = work.groupby(group_cols, dropna=False)["abs_weight"].idxmax()
    top_columns = (
        work.loc[idx, group_cols + ["feature", "weight", "abs_weight"]]
        .rename(
            columns={
                "feature": "top_design_column",
                "weight": "top_design_column_weight",
                "abs_weight": "top_design_column_abs_weight",
            }
        )
        .reset_index(drop=True)
    )
    grouped = (
        work.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n_design_columns=("feature", "nunique"),
            sum_abs_weight=("abs_weight", "sum"),
            max_abs_weight=("abs_weight", "max"),
            mean_abs_weight=("abs_weight", "mean"),
        )
        .merge(top_columns, on=group_cols, how="left")
        .sort_values(["trajectory_name", "propensity_design", "sum_abs_weight", "raw_feature"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )
    return grouped


def _build_matching_diagnostics_table(diagnostics: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    trajectory_specs = diagnostics.get("trajectory_specs", {})
    if not isinstance(trajectory_specs, dict):
        return pd.DataFrame()
    for trajectory_name, spec_payload in trajectory_specs.items():
        if not isinstance(spec_payload, dict):
            continue
        row: dict[str, object] = {
            "trajectory_name": str(trajectory_name),
            "propensity_design": diagnostics.get("propensity_design"),
            "n_high_exposure": spec_payload.get("n_high_exposure"),
            "n_low_exposure": spec_payload.get("n_low_exposure"),
        }
        propensity_payload = spec_payload.get("propensity", {})
        if isinstance(propensity_payload, dict):
            for key, value in propensity_payload.items():
                row[f"propensity_{key}"] = value
        matching_payload = spec_payload.get("matching", {})
        if isinstance(matching_payload, dict):
            for key, value in matching_payload.items():
                row[f"matching_{key}"] = value
        stacked_payload = spec_payload.get("stacked_did", {})
        if isinstance(stacked_payload, dict):
            for key, value in stacked_payload.items():
                row[f"stacked_did_{key}"] = value
        balance_payload = spec_payload.get("balance", {})
        if isinstance(balance_payload, dict):
            for match_stage, match_stage_payload in balance_payload.items():
                if not isinstance(match_stage_payload, dict):
                    continue
                overall = match_stage_payload.get("overall", {})
                if isinstance(overall, dict):
                    for key, value in overall.items():
                        row[f"balance_{match_stage}_overall_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trajectory_name").reset_index(drop=True) if rows else pd.DataFrame()


def _build_top_balance_table(balance_table: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if balance_table.empty:
        return pd.DataFrame()
    work = balance_table.copy()
    work["abs_smd"] = pd.to_numeric(work["standardized_mean_diff"], errors="coerce").abs()
    ranked = (
        work.sort_values(
            ["trajectory_name", "match_stage", "control_source", "abs_smd", "covariate"],
            ascending=[True, True, True, False, True],
        )
        .groupby(["trajectory_name", "match_stage", "control_source"], dropna=False, as_index=False, sort=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )
    return ranked


def _build_core_balance_table(
    balance_table: pd.DataFrame,
    *,
    core_covariates: Sequence[str],
) -> pd.DataFrame:
    if balance_table.empty:
        return pd.DataFrame()
    ordered_covariates = [str(value) for value in core_covariates if str(value)]
    if not ordered_covariates:
        return pd.DataFrame()
    order_lookup = {covariate: idx for idx, covariate in enumerate(ordered_covariates)}
    work = balance_table.loc[balance_table["covariate"].isin(ordered_covariates)].copy()
    if work.empty:
        return pd.DataFrame()
    work["covariate_order"] = work["covariate"].map(order_lookup).fillna(len(order_lookup)).astype(int)
    return work.sort_values(
        ["trajectory_name", "match_stage", "control_source", "covariate_order", "covariate"]
    ).drop(columns=["covariate_order"]).reset_index(drop=True)


def _summarize_common_break_results(
    common_break_results: pd.DataFrame,
    *,
    ref_year: int,
    event_year: int,
) -> pd.DataFrame:
    if common_break_results.empty:
        return pd.DataFrame()
    work = _add_confidence_intervals(common_break_results)
    rows: list[dict[str, object]] = []
    for keys, group in work.groupby(["trajectory_name", "outcome_col"], dropna=False):
        pre = group.loc[pd.to_numeric(group["year"], errors="coerce") < int(ref_year)]
        post = group.loc[pd.to_numeric(group["year"], errors="coerce") >= int(event_year)]
        event_row = group.loc[pd.to_numeric(group["year"], errors="coerce").eq(int(event_year))]
        n_pairs = (
            float(pd.to_numeric(group["n_pairs"], errors="coerce").max())
            if "n_pairs" in group.columns
            else np.nan
        )
        n_obs = (
            float(pd.to_numeric(group["n_obs"], errors="coerce").max())
            if "n_obs" in group.columns
            else np.nan
        )
        rows.append(
            {
                "trajectory_name": str(keys[0]),
                "outcome_col": str(keys[1]),
                "n_pairs": (int(n_pairs) if np.isfinite(n_pairs) else np.nan),
                "n_obs": (int(n_obs) if np.isfinite(n_obs) else np.nan),
                "mean_pre_coef": float(pd.to_numeric(pre["coef"], errors="coerce").mean()) if not pre.empty else np.nan,
                "max_abs_pre_coef": (
                    float(pd.to_numeric(pre["coef"], errors="coerce").abs().max()) if not pre.empty else np.nan
                ),
                "mean_post_coef": float(pd.to_numeric(post["coef"], errors="coerce").mean()) if not post.empty else np.nan,
                "max_abs_post_coef": (
                    float(pd.to_numeric(post["coef"], errors="coerce").abs().max()) if not post.empty else np.nan
                ),
                "coef_at_event_year": (
                    float(pd.to_numeric(event_row["coef"], errors="coerce").iloc[0]) if not event_row.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["trajectory_name", "outcome_col"]).reset_index(drop=True)


def _summarize_stacked_results(
    stacked_results: pd.DataFrame,
    *,
    ref_event_time: int,
) -> pd.DataFrame:
    if stacked_results.empty:
        return pd.DataFrame()
    work = _add_confidence_intervals(stacked_results)
    rows: list[dict[str, object]] = []
    for keys, group in work.groupby(["trajectory_name", "outcome_col"], dropna=False):
        rel_time = pd.to_numeric(group["rel_time"], errors="coerce")
        pre = group.loc[rel_time < int(ref_event_time)]
        post = group.loc[rel_time >= 0]
        event_row = group.loc[rel_time.eq(0)]
        n_pairs = (
            float(pd.to_numeric(group["n_pairs"], errors="coerce").max())
            if "n_pairs" in group.columns
            else np.nan
        )
        n_stacks = (
            float(pd.to_numeric(group["n_stacks"], errors="coerce").max())
            if "n_stacks" in group.columns
            else np.nan
        )
        n_obs = (
            float(pd.to_numeric(group["n_obs"], errors="coerce").max())
            if "n_obs" in group.columns
            else np.nan
        )
        rows.append(
            {
                "trajectory_name": str(keys[0]),
                "outcome_col": str(keys[1]),
                "n_pairs": (int(n_pairs) if np.isfinite(n_pairs) else np.nan),
                "n_stacks": (int(n_stacks) if np.isfinite(n_stacks) else np.nan),
                "n_obs": (int(n_obs) if np.isfinite(n_obs) else np.nan),
                "mean_pre_coef": float(pd.to_numeric(pre["coef"], errors="coerce").mean()) if not pre.empty else np.nan,
                "max_abs_pre_coef": (
                    float(pd.to_numeric(pre["coef"], errors="coerce").abs().max()) if not pre.empty else np.nan
                ),
                "mean_post_coef": float(pd.to_numeric(post["coef"], errors="coerce").mean()) if not post.empty else np.nan,
                "max_abs_post_coef": (
                    float(pd.to_numeric(post["coef"], errors="coerce").abs().max()) if not post.empty else np.nan
                ),
                "coef_at_event_time_0": (
                    float(pd.to_numeric(event_row["coef"], errors="coerce").iloc[0]) if not event_row.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["trajectory_name", "outcome_col"]).reset_index(drop=True)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except Exception:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    return ip.__class__.__name__.startswith("ZMQInteractiveShell")


def _display_figure_if_notebook(fig) -> None:
    if not _in_notebook():
        return
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        try:
            fig.show()
        except Exception:
            pass


def _plot_balance_summary_figures(balance_summary: pd.DataFrame, figures_dir: Path) -> list[Path]:
    if balance_summary.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        _log_step("matplotlib not available; skipping balance figures.")
        return []
    output_paths: list[Path] = []
    work = balance_summary.copy()
    work["stage_source"] = work["match_stage"].astype(str) + " | " + work["control_source"].astype(str)
    for trajectory_name, group in work.groupby("trajectory_name", sort=True):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        ordered = group.sort_values(["match_stage", "control_source"]).reset_index(drop=True)
        axes[0].barh(ordered["stage_source"], ordered["max_abs_smd"], color="#476c9b")
        axes[0].axvline(0.10, color="#a33", linestyle="--", linewidth=1)
        axes[0].set_title(f"{trajectory_name}: max |SMD|")
        axes[0].set_xlabel("Max absolute standardized mean difference")
        axes[1].barh(ordered["stage_source"], ordered["mean_abs_smd"], color="#4f8f5b")
        axes[1].axvline(0.10, color="#a33", linestyle="--", linewidth=1)
        axes[1].set_title(f"{trajectory_name}: mean |SMD|")
        axes[1].set_xlabel("Mean absolute standardized mean difference")
        fig.tight_layout()
        _display_figure_if_notebook(fig)
        out_path = figures_dir / f"balance_summary_{_slugify(trajectory_name)}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)
        _log_step(f"Wrote {out_path}")
    return output_paths


def _plot_dynamic_results(
    results_df: pd.DataFrame,
    *,
    x_col: str,
    ref_value: int,
    title_prefix: str,
    figures_dir: Path,
    filename_prefix: str,
    paired_raw_means_df: Optional[pd.DataFrame] = None,
    paired_raw_x_col: Optional[str] = None,
    paired_raw_ref_value: Optional[int] = None,
    paired_raw_title_prefix: Optional[str] = None,
    paired_raw_filename_prefix: Optional[str] = None,
) -> list[Path]:
    if results_df.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        _log_step(f"matplotlib not available; skipping {filename_prefix} figures.")
        return []
    output_paths: list[Path] = []
    work = _add_confidence_intervals(results_df)
    for keys, group in work.groupby(["trajectory_name", "outcome_col"], sort=True):
        plot_df = group.sort_values(x_col).copy()
        x = pd.to_numeric(plot_df[x_col], errors="coerce")
        coef = pd.to_numeric(plot_df["coef"], errors="coerce")
        lower = pd.to_numeric(plot_df["ci_lower"], errors="coerce")
        upper = pd.to_numeric(plot_df["ci_upper"], errors="coerce")
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.axvline(float(ref_value), color="grey", linestyle="--", linewidth=1)
        ax.plot(x, coef, marker="o", color="#2c5c85", linewidth=1.8)
        ax.fill_between(x, lower, upper, color="#8fb3d1", alpha=0.35)
        ax.set_title(f"{title_prefix}: {keys[0]} | {keys[1]}")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Coefficient")
        fig.tight_layout()
        _display_figure_if_notebook(fig)
        out_path = figures_dir / f"{filename_prefix}_{_slugify(keys[0])}_{_slugify(keys[1])}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)
        _log_step(f"Wrote {out_path}")

        if (
            paired_raw_means_df is not None
            and not paired_raw_means_df.empty
            and paired_raw_x_col is not None
            and paired_raw_ref_value is not None
            and paired_raw_title_prefix is not None
            and paired_raw_filename_prefix is not None
        ):
            raw_group = paired_raw_means_df.loc[
                paired_raw_means_df["trajectory_name"].astype(str).eq(str(keys[0]))
                & paired_raw_means_df["outcome_col"].astype(str).eq(str(keys[1]))
            ].copy()
            if not raw_group.empty:
                output_paths.extend(
                    _plot_raw_means(
                        raw_group,
                        x_col=paired_raw_x_col,
                        ref_value=int(paired_raw_ref_value),
                        title_prefix=paired_raw_title_prefix,
                        figures_dir=figures_dir,
                        filename_prefix=paired_raw_filename_prefix,
                    )
                )
    return output_paths


def _build_common_break_raw_means(
    matched_panel: pd.DataFrame,
    outcome_cols: Sequence[str],
) -> pd.DataFrame:
    if matched_panel.empty or not outcome_cols:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    base = matched_panel.drop_duplicates(subset=["pair_id", "c", "t"])
    for outcome_col in outcome_cols:
        if outcome_col not in base.columns:
            continue
        for (trajectory_name, year, treated), group in (
            base.loc[:, ["trajectory_name", "t", "treated", outcome_col]]
            .dropna(subset=["trajectory_name", "t", outcome_col, "treated"])
            .groupby(["trajectory_name", "t", "treated"], dropna=False)
        ):
            rows.append(
                {
                    "trajectory_name": str(trajectory_name),
                    "outcome_col": outcome_col,
                    "year": int(year),
                    "treated": int(treated),
                    "mean_outcome": float(pd.to_numeric(group[outcome_col], errors="coerce").mean()),
                    "n_obs": int(len(group)),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["trajectory_name", "outcome_col", "year", "treated"]).reset_index(drop=True)


def _build_stacked_raw_means(
    stacked_panel: pd.DataFrame,
    outcome_cols: Sequence[str],
) -> pd.DataFrame:
    if stacked_panel.empty or not outcome_cols:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    base = stacked_panel.drop_duplicates(subset=["stack_id", "pair_id", "c", "t"])
    for outcome_col in outcome_cols:
        if outcome_col not in base.columns:
            continue
        for (trajectory_name, rel_time, treated), group in (
            base.loc[:, ["trajectory_name", "rel_time", "treated", outcome_col]]
            .dropna(subset=["trajectory_name", "rel_time", outcome_col, "treated"])
            .groupby(["trajectory_name", "rel_time", "treated"], dropna=False)
        ):
            rows.append(
                {
                    "trajectory_name": str(trajectory_name),
                    "outcome_col": outcome_col,
                    "rel_time": int(rel_time),
                    "treated": int(treated),
                    "mean_outcome": float(pd.to_numeric(group[outcome_col], errors="coerce").mean()),
                    "n_obs": int(len(group)),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["trajectory_name", "outcome_col", "rel_time", "treated"]).reset_index(drop=True)


def _plot_raw_means(
    raw_means_df: pd.DataFrame,
    *,
    x_col: str,
    ref_value: int,
    title_prefix: str,
    figures_dir: Path,
    filename_prefix: str,
) -> list[Path]:
    if raw_means_df.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        _log_step(f"matplotlib not available; skipping {filename_prefix} raw mean figures.")
        return []

    output_paths: list[Path] = []
    for keys, group in raw_means_df.groupby(["trajectory_name", "outcome_col"], sort=True):
        plot_df = group.sort_values(x_col).copy()
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        treated_df = plot_df.loc[plot_df["treated"].eq(1)].copy()
        control_df = plot_df.loc[plot_df["treated"].eq(0)].copy()
        if not control_df.empty:
            ax.plot(
                pd.to_numeric(control_df[x_col], errors="coerce"),
                pd.to_numeric(control_df["mean_outcome"], errors="coerce"),
                marker="o",
                color="#3d6da3",
                label="Control",
                linewidth=1.6,
            )
        if not treated_df.empty:
            ax.plot(
                pd.to_numeric(treated_df[x_col], errors="coerce"),
                pd.to_numeric(treated_df["mean_outcome"], errors="coerce"),
                marker="o",
                color="#b53a3a",
                label="Treated",
                linewidth=1.6,
            )
        ax.axvline(float(ref_value), color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"{title_prefix}: {keys[0]} | {keys[1]}")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Raw outcome mean")
        ax.legend()
        fig.tight_layout()
        _display_figure_if_notebook(fig)
        out_path = figures_dir / f"{filename_prefix}_{_slugify(keys[0])}_{_slugify(keys[1])}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)
        _log_step(f"Wrote {out_path}")
    return output_paths


def write_analysis_reports(
    *,
    paths: MatchedExposurePaths,
    trajectory_summary: pd.DataFrame,
    propensity_scores: pd.DataFrame,
    diagnostics: dict[str, object],
    balance_table: pd.DataFrame,
    balance_summary: pd.DataFrame,
    common_break_results: pd.DataFrame,
    stacked_results: pd.DataFrame,
    matched_panel: pd.DataFrame,
    stacked_panel: pd.DataFrame,
    outcome_cols: Sequence[str],
    event_year: int,
    ref_year: int,
    ref_event_time: int,
    top_balance_rows: int,
    core_balance_covariates: Sequence[str],
) -> dict[str, object]:
    tables_dir = paths.out_dir / "tables"
    figures_dir = paths.out_dir / "figures"
    summary_path = paths.out_dir / "analysis_summary.md"

    trajectory_counts = _build_trajectory_group_counts(trajectory_summary)
    propensity_group_summary = _build_propensity_group_summary(trajectory_summary, propensity_scores)
    propensity_score_distribution = _build_propensity_score_distribution_table(trajectory_summary, propensity_scores)
    propensity_model_diagnostics = _build_propensity_model_diagnostics_table(diagnostics)
    propensity_feature_weights = _build_propensity_feature_weight_table(diagnostics)
    propensity_feature_group_weights = _build_propensity_feature_group_weight_table(propensity_feature_weights)
    matching_diagnostics = _build_matching_diagnostics_table(diagnostics)
    top_balance = _build_top_balance_table(balance_table, top_n=top_balance_rows)
    core_balance = _build_core_balance_table(balance_table, core_covariates=core_balance_covariates)
    common_break_results_wide = _add_confidence_intervals(common_break_results)
    common_break_summary = _summarize_common_break_results(
        common_break_results_wide,
        ref_year=ref_year,
        event_year=event_year,
    )
    stacked_results_wide = _add_confidence_intervals(stacked_results)
    stacked_summary = _summarize_stacked_results(
        stacked_results_wide,
        ref_event_time=ref_event_time,
    )
    common_break_raw_means = _build_common_break_raw_means(
        matched_panel,
        outcome_cols=outcome_cols,
    )
    stacked_raw_means = _build_stacked_raw_means(
        stacked_panel,
        outcome_cols=outcome_cols,
    )

    table_payloads = {
        "trajectory_group_counts.csv": trajectory_counts,
        "propensity_group_summary.csv": propensity_group_summary,
        "propensity_score_distribution.csv": propensity_score_distribution,
        "propensity_model_diagnostics.csv": propensity_model_diagnostics,
        "propensity_feature_weights.csv": propensity_feature_weights,
        "propensity_feature_group_weights.csv": propensity_feature_group_weights,
        "matching_diagnostics.csv": matching_diagnostics,
        "matching_balance_all_covariates.csv": balance_table,
        "matching_balance_summary.csv": balance_summary,
        "matching_core_balance.csv": core_balance,
        "matching_top_imbalances.csv": top_balance,
        "common_break_event_study.csv": common_break_results_wide,
        "common_break_event_study_summary.csv": common_break_summary,
        "stacked_did_event_study.csv": stacked_results_wide,
        "stacked_did_event_study_summary.csv": stacked_summary,
        "common_break_raw_means.csv": common_break_raw_means,
        "stacked_did_raw_means.csv": stacked_raw_means,
    }
    written_table_paths: list[Path] = []
    for filename, df in table_payloads.items():
        out_path = tables_dir / filename
        _write_csv(df, out_path)
        written_table_paths.append(out_path)

    figure_paths: list[Path] = []
    figure_paths.extend(_plot_balance_summary_figures(balance_summary, figures_dir))
    figure_paths.extend(
        _plot_dynamic_results(
            common_break_results_wide,
            x_col="year",
            ref_value=ref_year,
            title_prefix="Matched Common-Break Event Study",
            figures_dir=figures_dir,
            filename_prefix="common_break_event_study",
            paired_raw_means_df=common_break_raw_means,
            paired_raw_x_col="year",
            paired_raw_ref_value=ref_year,
            paired_raw_title_prefix="Matched Common-Break Raw Means",
            paired_raw_filename_prefix="common_break_raw_means",
        )
    )
    figure_paths.extend(
        _plot_dynamic_results(
            stacked_results_wide,
            x_col="rel_time",
            ref_value=ref_event_time,
            title_prefix="Matched Stacked DiD",
            figures_dir=figures_dir,
            filename_prefix="stacked_did",
            paired_raw_means_df=stacked_raw_means,
            paired_raw_x_col="rel_time",
            paired_raw_ref_value=ref_event_time,
            paired_raw_title_prefix="Matched Stacked DiD Raw Means",
            paired_raw_filename_prefix="stacked_did_raw_means",
        )
    )

    summary_sections = [
        "# Matched Exposure Design Summary",
        "## Trajectory Group Counts",
        _df_to_markdown(trajectory_counts, max_rows=30),
        "## Propensity Score Summary",
        _df_to_markdown(propensity_group_summary, max_rows=30),
        "## Propensity Score Distribution",
        _df_to_markdown(propensity_score_distribution, max_rows=40),
        "## Propensity Model Diagnostics",
        _df_to_markdown(propensity_model_diagnostics, max_rows=20),
        "## Top Propensity Feature Weights",
        _df_to_markdown(propensity_feature_weights, max_rows=30),
        "## Propensity Feature Group Weights",
        _df_to_markdown(propensity_feature_group_weights, max_rows=30),
        "## Matching Diagnostics",
        _df_to_markdown(matching_diagnostics, max_rows=20),
        "## Balance Summary",
        _df_to_markdown(balance_summary, max_rows=20),
        "## Full Balance Table Preview",
        _df_to_markdown(balance_table, max_rows=40),
        "## Core Balance Summary",
        _df_to_markdown(core_balance, max_rows=40),
        f"## Top {int(top_balance_rows)} Imbalances Per Stage/Source",
        _df_to_markdown(top_balance, max_rows=40),
        "## Common-Break Event Study Summary",
        _df_to_markdown(common_break_summary, max_rows=40),
        "## Stacked DiD Summary",
        _df_to_markdown(stacked_summary, max_rows=40),
        "## Common-Break Raw Means",
        _df_to_markdown(common_break_raw_means, max_rows=40),
        "## Stacked DiD Raw Means",
        _df_to_markdown(stacked_raw_means, max_rows=40),
        "## Output Directories",
        f"- Tables: `{tables_dir}`\n- Figures: `{figures_dir}`\n- Summary: `{summary_path}`",
    ]
    if figure_paths:
        summary_sections.extend(
            [
                "## Figure Files",
                "\n".join(f"- `{path.name}`" for path in figure_paths),
            ]
        )
    _write_text("\n\n".join(summary_sections), summary_path)
    return {
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "summary_path": str(summary_path),
        "table_paths": [str(path) for path in written_table_paths],
        "figure_paths": [str(path) for path in figure_paths],
    }


def _saved_output_paths(paths: MatchedExposurePaths) -> dict[str, Path]:
    return {
        "analysis_panel": paths.analysis_panel_out,
        "matching_features": paths.matching_features_out,
        "propensity_scores": paths.propensity_scores_out,
        "trajectory_summary": paths.trajectory_summary_out,
        "matched_pairs": paths.matched_pairs_out,
        "balance_table": paths.balance_table_out,
        "balance_summary": paths.balance_summary_out,
        "matched_panel": paths.matched_panel_out,
        "final_analysis_panel": paths.final_matched_analysis_panel_out,
        "common_break_results": paths.common_break_results_out,
        "stacked_panel": paths.stacked_panel_out,
        "stacked_results": paths.stacked_results_out,
    }


def _saved_matching_output_paths(paths: MatchedExposurePaths) -> dict[str, Path]:
    return {
        "propensity_scores": paths.propensity_scores_out,
        "trajectory_summary": paths.trajectory_summary_out,
        "matched_pairs": paths.matched_pairs_out,
        "balance_table": paths.balance_table_out,
        "balance_summary": paths.balance_summary_out,
        "matched_panel": paths.matched_panel_out,
        "final_analysis_panel": paths.final_matched_analysis_panel_out,
    }


def _can_reuse_saved_analysis_outputs(
    paths: MatchedExposurePaths,
    *,
    propensity_design: str,
) -> bool:
    if not all(path.exists() for path in _saved_output_paths(paths).values()):
        return False
    if not paths.diagnostics_out.exists():
        return False
    diagnostics = _read_json(paths.diagnostics_out)
    return str(diagnostics.get("propensity_design", "opt_takeup")) == str(propensity_design)


def _can_reuse_saved_matching_outputs(
    paths: MatchedExposurePaths,
    *,
    propensity_design: str,
) -> bool:
    if not all(path.exists() for path in _saved_matching_output_paths(paths).values()):
        return False
    if not paths.diagnostics_out.exists():
        return False
    diagnostics = _read_json(paths.diagnostics_out)
    return str(diagnostics.get("propensity_design", "opt_takeup")) == str(propensity_design)


def _load_saved_analysis_outputs(paths: MatchedExposurePaths) -> dict[str, object]:
    loaded: dict[str, object] = {
        key: pd.read_parquet(path)
        for key, path in _saved_output_paths(paths).items()
    }
    if not isinstance(loaded["final_analysis_panel"], pd.DataFrame) and isinstance(loaded.get("matched_panel"), pd.DataFrame):
        loaded["final_analysis_panel"] = loaded["matched_panel"]
    loaded["diagnostics"] = _read_json(paths.diagnostics_out)
    return loaded


def _load_saved_matching_outputs(paths: MatchedExposurePaths) -> dict[str, object]:
    loaded: dict[str, object] = {
        key: pd.read_parquet(path)
        for key, path in _saved_matching_output_paths(paths).items()
    }
    if not isinstance(loaded["final_analysis_panel"], pd.DataFrame) and isinstance(loaded.get("matched_panel"), pd.DataFrame):
        loaded["final_analysis_panel"] = loaded["matched_panel"]
    loaded["diagnostics"] = _read_json(paths.diagnostics_out)
    return loaded


def run_matched_exposure_design(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
) -> dict[str, object]:
    run_started = time.perf_counter()
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    design_cfg = get_cfg_section(cfg_full, "matched_exposure_design")
    paths = _resolve_paths(cfg_full)
    shift_cfg, source_cfg = _load_base_configs(cfg_full)

    data_min_t = int(design_cfg.get("data_min_t", 2010))
    data_max_t = int(design_cfg.get("data_max_t", 2022))
    source_flow_year_min = int(design_cfg.get("source_flow_year_min", 2008))
    source_flow_year_max = int(design_cfg.get("source_flow_year_max", 2022))
    feature_year_min = int(design_cfg.get("propensity_feature_year_min", 2010))
    feature_year_max = int(design_cfg.get("propensity_feature_year_max", 2013))
    force_rebuild_inputs = bool(design_cfg.get("force_rebuild_inputs", False))
    trajectory_specs = _parse_trajectory_specs(design_cfg)
    outcome_cols = [str(col) for col in design_cfg.get("outcome_cols", [])]
    reuse_saved_analysis_outputs = bool(design_cfg.get("reuse_saved_analysis_outputs", False))
    reuse_saved_matching_outputs = bool(design_cfg.get("reuse_saved_matching_outputs", False))
    propensity_design = str(design_cfg.get("propensity_design", "opt_takeup")).strip() or "opt_takeup"
    matching_algorithm = str(
        design_cfg.get("matching_algorithm", "optimal_max_cardinality_min_cost")
    ).strip() or "optimal_max_cardinality_min_cost"
    matching_size_feature = str(design_cfg.get("matching_size_feature", "match_firm_size_pre_level"))
    matching_growth_feature = str(design_cfg.get("matching_growth_feature", "match_firm_size_pre_growth"))
    matching_users_feature = str(design_cfg.get("matching_users_feature", "company_n_users_log1p"))
    matching_coverage_feature = str(design_cfg.get("matching_coverage_feature", "firm_size_annual_pre_n_years"))
    matching_coverage_rule = str(design_cfg.get("matching_coverage_rule", "full_pre_coverage")).strip() or "full_pre_coverage"
    matching_size_weight = float(design_cfg.get("matching_size_weight", 0.35))
    matching_growth_weight = float(design_cfg.get("matching_growth_weight", 0.35))
    matching_users_weight = float(design_cfg.get("matching_users_weight", 0.20))
    coverage_required_years = int(feature_year_max - feature_year_min + 1)
    core_balance_covariates = [
        "predicted_prob",
        "logit_score",
        matching_users_feature,
        matching_size_feature,
        "effective_size",
        matching_growth_feature,
        matching_coverage_feature,
    ]

    if propensity_design not in SUPPORTED_PROPENSITY_DESIGNS:
        raise ValueError(
            f"Unsupported propensity_design={propensity_design!r}. "
            f"Expected one of {SUPPORTED_PROPENSITY_DESIGNS}."
        )
    if matching_algorithm not in SUPPORTED_MATCHING_ALGORITHMS:
        raise ValueError(
            f"Unsupported matching_algorithm={matching_algorithm!r}. "
            f"Expected one of {SUPPORTED_MATCHING_ALGORITHMS}."
        )
    if matching_coverage_rule not in SUPPORTED_MATCHING_COVERAGE_RULES:
        raise ValueError(
            f"Unsupported matching_coverage_rule={matching_coverage_rule!r}. "
            f"Expected one of {SUPPORTED_MATCHING_COVERAGE_RULES}."
        )

    if reuse_saved_analysis_outputs and _can_reuse_saved_analysis_outputs(
        paths,
        propensity_design=propensity_design,
    ):
        _log_step("reusing saved analysis outputs from disk")
        loaded = _load_saved_analysis_outputs(paths)
        diagnostics = loaded.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        diagnostics["propensity_design"] = propensity_design
        diagnostics["reused_saved_analysis_outputs"] = True
        reports: dict[str, object] = {}
        final_analysis_panel = loaded.get("final_analysis_panel")
        if not isinstance(final_analysis_panel, pd.DataFrame) and isinstance(loaded.get("matched_panel"), pd.DataFrame):
            final_analysis_panel = loaded["matched_panel"]
        if bool(design_cfg.get("write_analysis_reports", True)):
            _log_step("rewriting analysis tables and figures from saved outputs")
            reports = write_analysis_reports(
                paths=paths,
                trajectory_summary=loaded["trajectory_summary"],  # type: ignore[arg-type]
                propensity_scores=loaded["propensity_scores"],  # type: ignore[arg-type]
                diagnostics=diagnostics,
                balance_table=loaded["balance_table"],  # type: ignore[arg-type]
                balance_summary=loaded["balance_summary"],  # type: ignore[arg-type]
                common_break_results=loaded["common_break_results"],  # type: ignore[arg-type]
                stacked_results=loaded["stacked_results"],  # type: ignore[arg-type]
                matched_panel=loaded.get("matched_panel", pd.DataFrame()),  # type: ignore[arg-type]
                stacked_panel=loaded.get("stacked_panel", pd.DataFrame()),  # type: ignore[arg-type]
                outcome_cols=outcome_cols,
                event_year=int(design_cfg.get("event_year", 2015)),
                ref_year=int(design_cfg.get("ref_year", 2014)),
                ref_event_time=-1,
                top_balance_rows=int(design_cfg.get("analysis_top_balance_rows", 20)),
                core_balance_covariates=core_balance_covariates,
            )
            diagnostics["analysis_reports"] = reports
        if isinstance(final_analysis_panel, pd.DataFrame) and not paths.final_matched_analysis_panel_out.exists():
            _write_parquet(final_analysis_panel, paths.final_matched_analysis_panel_out)
        _write_json(diagnostics, paths.diagnostics_out)
        _log_step(f"pipeline complete via reuse | elapsed={_elapsed_seconds(run_started):.1f}s")
        loaded["diagnostics"] = diagnostics
        loaded["analysis_reports"] = reports
        loaded["final_analysis_panel"] = final_analysis_panel
        return loaded
    if reuse_saved_analysis_outputs:
        _log_step("reuse_saved_analysis_outputs=true but saved outputs are incomplete; running full pipeline.")

    if reuse_saved_matching_outputs and _can_reuse_saved_matching_outputs(
        paths,
        propensity_design=propensity_design,
    ):
        _log_step("reusing saved matching outputs and rerunning downstream regressions")
        loaded = _load_saved_matching_outputs(paths)
        matched_pairs = loaded.get("matched_pairs", pd.DataFrame())
        matched_panel = loaded.get("matched_panel", pd.DataFrame())
        trajectory_summary = loaded.get("trajectory_summary", pd.DataFrame())
        propensity_scores = loaded.get("propensity_scores", pd.DataFrame())
        balance_table = loaded.get("balance_table", pd.DataFrame())
        balance_summary = loaded.get("balance_summary", pd.DataFrame())
        diagnostics = loaded.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        diagnostics.pop("analysis_reports", None)
        diagnostics.pop("reused_saved_analysis_outputs", None)
        diagnostics["propensity_design"] = propensity_design
        diagnostics["reused_saved_matching_outputs"] = True
        trajectory_diag_payload = diagnostics.get("trajectory_specs", {})
        if not isinstance(trajectory_diag_payload, dict):
            trajectory_diag_payload = {}
            diagnostics["trajectory_specs"] = trajectory_diag_payload
        common_break_frames: list[pd.DataFrame] = []
        stacked_panel_frames: list[pd.DataFrame] = []
        stacked_result_frames: list[pd.DataFrame] = []
        for spec in trajectory_specs:
            trajectory_name = str(spec["name"])
            pair_df = (
                matched_pairs.loc[matched_pairs["trajectory_name"].astype(str).eq(trajectory_name)].copy()
                if isinstance(matched_pairs, pd.DataFrame) and not matched_pairs.empty and "trajectory_name" in matched_pairs.columns
                else (matched_pairs.copy() if isinstance(matched_pairs, pd.DataFrame) else pd.DataFrame())
            )
            trajectory_matched_panel = (
                matched_panel.loc[matched_panel["trajectory_name"].astype(str).eq(trajectory_name)].copy()
                if isinstance(matched_panel, pd.DataFrame) and not matched_panel.empty and "trajectory_name" in matched_panel.columns
                else (matched_panel.copy() if isinstance(matched_panel, pd.DataFrame) else pd.DataFrame())
            )
            existing_spec_diag = trajectory_diag_payload.get(trajectory_name, {})
            spec_diag = dict(existing_spec_diag) if isinstance(existing_spec_diag, dict) else {}
            if not bool(spec.get("run_regressions", True)) or not outcome_cols or pair_df.empty or trajectory_matched_panel.empty:
                diagnostics["trajectory_specs"][trajectory_name] = spec_diag
                continue
            _log_step(
                "rerunning downstream regressions from saved matching outputs | "
                f"trajectory={trajectory_name} matched_pairs={len(pair_df):,} matched_panel_rows={len(trajectory_matched_panel):,}"
            )
            common_break_frames.append(
                run_common_break_event_study(
                    trajectory_matched_panel,
                    outcome_cols=outcome_cols,
                    ref_year=int(design_cfg.get("ref_year", 2014)),
                )
            )
            cohort_df = assign_persistent_entry_cohorts(
                trajectory_matched_panel.loc[trajectory_matched_panel["treated"].eq(1), ["c", "t", "z_ct"]],
                cohort_min_year=int(design_cfg.get("stacked_min_cohort_year", 2014)),
                cohort_max_year=int(design_cfg.get("stacked_max_cohort_year", 2019)),
                pre_years_required=int(design_cfg.get("stacked_pre_years", 3)),
                forward_window_years=int(design_cfg.get("stacked_post_years", 3)) + 1,
                min_positive_years_in_forward_window=int(
                    design_cfg.get("stacked_min_positive_years_in_forward_window", 2)
                ),
            )
            stacked_panel, stacked_diag = build_stacked_did_panel(
                trajectory_matched_panel,
                pair_df,
                cohort_df,
                pre_window=int(design_cfg.get("stacked_pre_years", 3)),
                post_window=int(design_cfg.get("stacked_post_years", 3)),
            )
            spec_diag["stacked_did"] = stacked_diag
            if not stacked_panel.empty:
                stacked_panel_frames.append(stacked_panel)
                stacked_result_frames.append(
                    run_stacked_did(
                        stacked_panel,
                        outcome_cols=outcome_cols,
                        ref_event_time=-1,
                    )
                )
            diagnostics["trajectory_specs"][trajectory_name] = spec_diag
        common_break_results = (
            pd.concat(common_break_frames, ignore_index=True)
            if common_break_frames
            else pd.DataFrame()
        )
        stacked_panel = (
            pd.concat(stacked_panel_frames, ignore_index=True)
            if stacked_panel_frames
            else pd.DataFrame()
        )
        stacked_results = (
            pd.concat(stacked_result_frames, ignore_index=True)
            if stacked_result_frames
            else pd.DataFrame()
        )
        _write_parquet(common_break_results, paths.common_break_results_out)
        _write_parquet(stacked_panel, paths.stacked_panel_out)
        _write_parquet(stacked_results, paths.stacked_results_out)
        reports: dict[str, object] = {}
        if bool(design_cfg.get("write_analysis_reports", True)):
            _log_step("writing analysis tables and figures from saved matching outputs")
            reports = write_analysis_reports(
                paths=paths,
                trajectory_summary=trajectory_summary if isinstance(trajectory_summary, pd.DataFrame) else pd.DataFrame(),
                propensity_scores=propensity_scores if isinstance(propensity_scores, pd.DataFrame) else pd.DataFrame(),
                diagnostics=diagnostics,
                balance_table=balance_table if isinstance(balance_table, pd.DataFrame) else pd.DataFrame(),
                balance_summary=balance_summary if isinstance(balance_summary, pd.DataFrame) else pd.DataFrame(),
                common_break_results=common_break_results,
                stacked_results=stacked_results,
                matched_panel=matched_panel if isinstance(matched_panel, pd.DataFrame) else pd.DataFrame(),
                stacked_panel=stacked_panel,
                outcome_cols=outcome_cols,
                event_year=int(design_cfg.get("event_year", 2015)),
                ref_year=int(design_cfg.get("ref_year", 2014)),
                ref_event_time=-1,
                top_balance_rows=int(design_cfg.get("analysis_top_balance_rows", 20)),
                core_balance_covariates=core_balance_covariates,
            )
            diagnostics["analysis_reports"] = reports
        final_analysis_panel = loaded.get("final_analysis_panel")
        if not isinstance(final_analysis_panel, pd.DataFrame):
            final_analysis_panel = matched_panel if isinstance(matched_panel, pd.DataFrame) else pd.DataFrame()
        if isinstance(final_analysis_panel, pd.DataFrame) and not paths.final_matched_analysis_panel_out.exists():
            _write_parquet(final_analysis_panel, paths.final_matched_analysis_panel_out)
        _write_json(diagnostics, paths.diagnostics_out)
        loaded["common_break_results"] = common_break_results
        loaded["stacked_panel"] = stacked_panel
        loaded["stacked_results"] = stacked_results
        loaded["diagnostics"] = diagnostics
        loaded["analysis_reports"] = reports
        loaded["final_analysis_panel"] = final_analysis_panel
        _log_step(f"pipeline complete via saved matching reuse | elapsed={_elapsed_seconds(run_started):.1f}s")
        return loaded
    if reuse_saved_matching_outputs:
        _log_step("reuse_saved_matching_outputs=true but saved matching outputs are incomplete; running full pipeline.")

    stage_started = time.perf_counter()
    _log_step(
        "building source analysis panel | "
        f"data_window={data_min_t}-{data_max_t} force_rebuild_inputs={force_rebuild_inputs}"
    )
    panel, firms, selected_meta, workforce = build_source_analysis_panel_from_inputs(
        source_cfg=source_cfg,
        data_min_t=data_min_t,
        data_max_t=data_max_t,
        force_rebuild_inputs=force_rebuild_inputs,
    )
    _log_step(
        "built source analysis panel | "
        f"panel_rows={len(panel):,} firms={len(firms):,} selected_meta_rows={len(selected_meta):,} "
        f"workforce_rows={len(workforce):,} elapsed={_elapsed_seconds(stage_started):.1f}s"
    )

    stage_started = time.perf_counter()
    _log_step(
        "building shift-share components | "
        f"source_flow_window={source_flow_year_min}-{source_flow_year_max} "
        f"data_window={data_min_t}-{data_max_t}"
    )
    school_growth, school_shift_sample, transition_shares, instrument_panel = build_shift_share_components_from_configs(
        shift_cfg=shift_cfg,
        source_cfg=source_cfg,
        source_flow_year_min=source_flow_year_min,
        source_flow_year_max=source_flow_year_max,
        data_min_t=data_min_t,
        data_max_t=data_max_t,
        force_rebuild_inputs=force_rebuild_inputs,
    )
    _log_step(
        "built shift-share components | "
        f"school_growth_rows={len(school_growth):,} school_shift_sample_rows={len(school_shift_sample):,} "
        f"transition_shares_rows={len(transition_shares):,} instrument_rows={len(instrument_panel):,} "
        f"elapsed={_elapsed_seconds(stage_started):.1f}s"
    )

    stage_started = time.perf_counter()
    _log_step("merging z_ct into firm-year analysis panel")
    analysis_panel = merge_zct_into_analysis_panel(panel, instrument_panel)
    _log_step(
        "merged z_ct into analysis panel | "
        f"analysis_panel_rows={len(analysis_panel):,} columns={len(analysis_panel.columns):,} "
        f"elapsed={_elapsed_seconds(stage_started):.1f}s"
    )

    stage_started = time.perf_counter()
    _log_step(
        "building matching features from WRDS | "
        f"feature_window={feature_year_min}-{feature_year_max}"
    )
    matching_features = build_matching_feature_frame_from_wrds(
        firms=firms,
        selected_meta=selected_meta,
        wrds_annual=workforce,
        feature_year_min=feature_year_min,
        feature_year_max=feature_year_max,
    )
    _log_step(
        "built matching features from WRDS | "
        f"rows={len(matching_features):,} columns={len(matching_features.columns):,} "
        f"elapsed={_elapsed_seconds(stage_started):.1f}s"
    )

    propensity_scores = pd.DataFrame()
    propensity_diagnostics: dict[str, object] = {}
    if propensity_design == "opt_takeup":
        stage_started = time.perf_counter()
        _log_step(
            "estimating propensity scores | "
            f"propensity_design={propensity_design} "
            f"training_sample_mode={str(design_cfg.get('training_sample_mode', 'preferred_plus_outside_negatives'))} "
            f"feature_family={MATCHING_FEATURE_FAMILY}"
        )
        propensity_scores, propensity_diagnostics = build_propensity_scores(
            matching_features,
            analysis_panel,
            design_cfg=design_cfg,
        )
        propensity_scores["propensity_design"] = "opt_takeup"
        _log_step(
            "estimated propensity scores | "
            f"rows={len(propensity_scores):,} columns={len(propensity_scores.columns):,} "
            f"elapsed={_elapsed_seconds(stage_started):.1f}s"
        )

    trajectory_frames: list[pd.DataFrame] = []
    propensity_frames: list[pd.DataFrame] = []
    matched_pair_frames: list[pd.DataFrame] = []
    balance_table_frames: list[pd.DataFrame] = []
    balance_summary_frames: list[pd.DataFrame] = []
    matched_panel_frames: list[pd.DataFrame] = []
    common_break_frames: list[pd.DataFrame] = []
    stacked_panel_frames: list[pd.DataFrame] = []
    stacked_result_frames: list[pd.DataFrame] = []
    diagnostics: dict[str, object] = {
        "propensity_design": propensity_design,
        "trajectory_specs": {},
    }
    if propensity_diagnostics:
        diagnostics["propensity_diagnostics"] = propensity_diagnostics

    for spec in trajectory_specs:
        trajectory = summarize_zct_trajectory(
            analysis_panel,
            window_start=int(spec["start"]),
            window_end=int(spec["end"]),
            trajectory_name=str(spec["name"]),
        )
        classified = classify_zct_trajectory(
            trajectory,
            high_quantile=float(design_cfg.get("high_exposure_quantile", 0.75)),
            low_quantile=float(design_cfg.get("low_exposure_quantile", 0.25)),
            high_exposure_metric=str(design_cfg.get("high_exposure_metric", "max_z")),
            low_exposure_metric=str(design_cfg.get("low_exposure_metric", "max_z")),
        )
        trajectory_frames.append(classified)
        spec_diag: dict[str, object] = {
            "n_high_exposure": int(classified["high_exposure"].sum()),
            "n_low_exposure": int(classified["low_exposure"].sum()),
        }
        trajectory_propensity_scores = propensity_scores
        if propensity_design == "treatment_propensity":
            stage_started = time.perf_counter()
            _log_step(
                "estimating propensity scores | "
                f"propensity_design={propensity_design} trajectory={spec['name']} "
                f"feature_family={MATCHING_FEATURE_FAMILY}"
            )
            trajectory_propensity_scores, trajectory_propensity_diag = build_treatment_propensity_scores(
                matching_features,
                classified,
                design_cfg=design_cfg,
                include_outside_negative_controls=bool(
                    design_cfg.get("include_outside_negative_controls", True)
                ),
            )
            trajectory_propensity_scores["propensity_design"] = "treatment_propensity"
            propensity_frames.append(trajectory_propensity_scores)
            spec_diag["propensity"] = trajectory_propensity_diag
            _log_step(
                "estimated propensity scores | "
                f"propensity_design={propensity_design} trajectory={spec['name']} "
                f"rows={len(trajectory_propensity_scores):,} columns={len(trajectory_propensity_scores.columns):,} "
                f"elapsed={_elapsed_seconds(stage_started):.1f}s"
            )
        if bool(spec.get("run_matching", True)):
            _log_step(
                "starting matching | "
                f"trajectory={spec['name']} window={int(spec['start'])}-{int(spec['end'])} "
                f"n_high_exposure={spec_diag['n_high_exposure']:,} n_low_exposure={spec_diag['n_low_exposure']:,}"
            )
            pair_df, matching_diag = match_high_to_low_exposure(
                classified,
                matching_features,
                trajectory_propensity_scores,
                include_outside_negative_controls=bool(
                    design_cfg.get("include_outside_negative_controls", True)
                ),
                caliper_sd_multiplier=float(design_cfg.get("caliper_sd_multiplier", 0.2)),
                progress_every=design_cfg.get("matching_progress_every", 500),
                size_match_feature=matching_size_feature,
                growth_match_feature=matching_growth_feature,
                size_match_weight=matching_size_weight,
                growth_match_weight=matching_growth_weight,
                users_match_feature=matching_users_feature,
                users_match_weight=matching_users_weight,
                coverage_feature=matching_coverage_feature,
                coverage_rule=matching_coverage_rule,
                coverage_required_years=coverage_required_years,
                matching_algorithm=matching_algorithm,
                propensity_design=propensity_design,
            )
            pair_df["trajectory_name"] = str(spec["name"])
            matched_pair_frames.append(pair_df)
            spec_diag["matching"] = matching_diag
            _log_step(f"building balance tables | trajectory={spec['name']}")
            balance_table, balance_summary = build_matching_balance_tables(
                classified,
                matching_features,
                trajectory_propensity_scores,
                pair_df,
                include_outside_negative_controls=bool(
                    design_cfg.get("include_outside_negative_controls", True)
                ),
                size_match_feature=matching_size_feature,
                growth_match_feature=matching_growth_feature,
                users_match_feature=matching_users_feature,
                coverage_feature=matching_coverage_feature,
                coverage_required_years=coverage_required_years,
                users_feature_required=matching_users_weight > 0,
            )
            if not balance_table.empty:
                balance_table_frames.append(balance_table)
            if not balance_summary.empty:
                balance_summary_frames.append(balance_summary)
                balance_diag: dict[str, dict[str, object]] = {}
                for match_stage, group in balance_summary.groupby("match_stage", sort=False):
                    stage_payload: dict[str, object] = {}
                    for _, row in group.iterrows():
                        stage_payload[str(row["control_source"])] = {
                            "n_covariates": int(row["n_covariates"]),
                            "max_abs_smd": (
                                None if pd.isna(row["max_abs_smd"]) else float(row["max_abs_smd"])
                            ),
                            "mean_abs_smd": (
                                None if pd.isna(row["mean_abs_smd"]) else float(row["mean_abs_smd"])
                            ),
                            "n_abs_smd_gt_0_10": int(row["n_abs_smd_gt_0_10"]),
                            "n_abs_smd_gt_0_25": int(row["n_abs_smd_gt_0_25"]),
                            "share_abs_smd_gt_0_10": (
                                None
                                if pd.isna(row["share_abs_smd_gt_0_10"])
                                else float(row["share_abs_smd_gt_0_10"])
                            ),
                            "share_abs_smd_gt_0_25": (
                                None
                                if pd.isna(row["share_abs_smd_gt_0_25"])
                                else float(row["share_abs_smd_gt_0_25"])
                            ),
                        }
                    balance_diag[str(match_stage)] = stage_payload
                spec_diag["balance"] = balance_diag
            _log_step(f"finished balance tables | trajectory={spec['name']}")
            if not pair_df.empty:
                matched_panel = build_matched_panel(
                    analysis_panel,
                    pair_df,
                    trajectory_name=str(spec["name"]),
                )
                matched_panel_frames.append(matched_panel)
                if bool(spec.get("run_regressions", True)) and outcome_cols:
                    common_break_frames.append(
                        run_common_break_event_study(
                            matched_panel,
                            outcome_cols=outcome_cols,
                            ref_year=int(design_cfg.get("ref_year", 2014)),
                        )
                    )
                    cohort_df = assign_persistent_entry_cohorts(
                        matched_panel.loc[matched_panel["treated"].eq(1), ["c", "t", "z_ct"]],
                        cohort_min_year=int(design_cfg.get("stacked_min_cohort_year", 2014)),
                        cohort_max_year=int(design_cfg.get("stacked_max_cohort_year", 2019)),
                        pre_years_required=int(design_cfg.get("stacked_pre_years", 3)),
                        forward_window_years=int(design_cfg.get("stacked_post_years", 3)) + 1,
                        min_positive_years_in_forward_window=int(
                            design_cfg.get("stacked_min_positive_years_in_forward_window", 2)
                        ),
                    )
                    stacked_panel, stacked_diag = build_stacked_did_panel(
                        matched_panel,
                        pair_df,
                        cohort_df,
                        pre_window=int(design_cfg.get("stacked_pre_years", 3)),
                        post_window=int(design_cfg.get("stacked_post_years", 3)),
                    )
                    spec_diag["stacked_did"] = stacked_diag
                    if not stacked_panel.empty:
                        stacked_panel_frames.append(stacked_panel)
                        stacked_result_frames.append(
                            run_stacked_did(
                                stacked_panel,
                                outcome_cols=outcome_cols,
                                ref_event_time=-1,
                            )
                        )
        diagnostics["trajectory_specs"][str(spec["name"])] = spec_diag

    if propensity_design == "treatment_propensity":
        propensity_scores = (
            pd.concat(propensity_frames, ignore_index=True)
            if propensity_frames
            else pd.DataFrame()
        )

    trajectory_summary = (
        pd.concat(trajectory_frames, ignore_index=True)
        if trajectory_frames
        else pd.DataFrame()
    )
    matched_pairs = (
        pd.concat(matched_pair_frames, ignore_index=True)
        if matched_pair_frames
        else pd.DataFrame()
    )
    balance_table = (
        pd.concat(balance_table_frames, ignore_index=True)
        if balance_table_frames
        else pd.DataFrame()
    )
    balance_summary = (
        pd.concat(balance_summary_frames, ignore_index=True)
        if balance_summary_frames
        else pd.DataFrame()
    )
    matched_panel = (
        pd.concat(matched_panel_frames, ignore_index=True)
        if matched_panel_frames
        else pd.DataFrame()
    )
    common_break_results = (
        pd.concat(common_break_frames, ignore_index=True)
        if common_break_frames
        else pd.DataFrame()
    )
    stacked_panel = (
        pd.concat(stacked_panel_frames, ignore_index=True)
        if stacked_panel_frames
        else pd.DataFrame()
    )
    stacked_results = (
        pd.concat(stacked_result_frames, ignore_index=True)
        if stacked_result_frames
        else pd.DataFrame()
    )

    _write_parquet(transition_shares, paths.transition_shares_out)
    _write_parquet(school_growth, paths.school_growth_out)
    _write_parquet(school_shift_sample, paths.school_shift_sample_out)
    _write_parquet(instrument_panel, paths.instrument_panel_out)
    _write_parquet(analysis_panel, paths.analysis_panel_out)
    _write_parquet(matching_features, paths.matching_features_out)
    _write_parquet(propensity_scores, paths.propensity_scores_out)
    _write_parquet(trajectory_summary, paths.trajectory_summary_out)
    _write_parquet(matched_pairs, paths.matched_pairs_out)
    _write_parquet(balance_table, paths.balance_table_out)
    _write_parquet(balance_summary, paths.balance_summary_out)
    _write_parquet(matched_panel, paths.matched_panel_out)
    _write_parquet(matched_panel, paths.final_matched_analysis_panel_out)
    _write_parquet(common_break_results, paths.common_break_results_out)
    _write_parquet(stacked_panel, paths.stacked_panel_out)
    _write_parquet(stacked_results, paths.stacked_results_out)
    reports: dict[str, object] = {}
    if bool(design_cfg.get("write_analysis_reports", True)):
        _log_step("writing analysis tables and figures")
        reports = write_analysis_reports(
            paths=paths,
            trajectory_summary=trajectory_summary,
            propensity_scores=propensity_scores,
            diagnostics=diagnostics,
            balance_table=balance_table,
            balance_summary=balance_summary,
            common_break_results=common_break_results,
            stacked_results=stacked_results,
            matched_panel=matched_panel,
            stacked_panel=stacked_panel,
            outcome_cols=outcome_cols,
            event_year=int(design_cfg.get("event_year", 2015)),
            ref_year=int(design_cfg.get("ref_year", 2014)),
            ref_event_time=-1,
            top_balance_rows=int(design_cfg.get("analysis_top_balance_rows", 20)),
            core_balance_covariates=core_balance_covariates,
        )
        diagnostics["analysis_reports"] = reports
    _write_json(diagnostics, paths.diagnostics_out)
    _log_step(f"pipeline complete | elapsed={_elapsed_seconds(run_started):.1f}s")

    return {
        "analysis_panel": analysis_panel,
        "matching_features": matching_features,
        "propensity_scores": propensity_scores,
        "trajectory_summary": trajectory_summary,
        "matched_pairs": matched_pairs,
        "balance_table": balance_table,
        "balance_summary": balance_summary,
        "matched_panel": matched_panel,
        "common_break_results": common_break_results,
        "stacked_panel": stacked_panel,
        "stacked_results": stacked_results,
        "final_analysis_panel": matched_panel,
        "diagnostics": diagnostics,
        "analysis_reports": reports,
    }


def _parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the source-rebuilt matched z_ct exposure design.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    if args is not None:
        return parser.parse_args(list(args))
    argv = sys.argv[1:]
    if Path(sys.argv[0]).name == "ipykernel_launcher.py" or "ipykernel" in sys.modules:
        parsed, _ = parser.parse_known_args(argv)
        return parsed
    return parser.parse_args(argv)


def main(args: Optional[Iterable[str]] = None) -> dict[str, object]:
    parsed = _parse_args(args)
    return run_matched_exposure_design(config_path=parsed.config)


if __name__ == "__main__":
    main()
