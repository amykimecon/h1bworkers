"""Source-based data builders for OPT exposure modeling and event studies.

This module rebuilds the firm-level treatment/outcome objects used by the OPT
exposure workflow directly from source FOIA and WRDS inputs. It is shared by:

  - company_shift_share.revelio_company_features
  - company_shift_share.exposure_event_study

No inputs here depend on downstream outputs from build_company_shift_share.py.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import time
from typing import Optional

import duckdb as ddb
import numpy as np
import pandas as pd

try:
    import wrds
except ImportError:  # pragma: no cover
    wrds = None  # type: ignore[assignment]

try:
    from company_shift_share.config_loader import (
        DEFAULT_CONFIG_PATH,
        apply_testing_output_suffix,
        get_cfg_section,
        load_config,
        testing_enabled,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import (  # type: ignore[no-redef]
        DEFAULT_CONFIG_PATH,
        apply_testing_output_suffix,
        get_cfg_section,
        load_config,
        testing_enabled,
    )

try:
    from helpers import degree_clean_regex_sql
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from helpers import degree_clean_regex_sql  # type: ignore[no-redef]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


_META_SUFFIX = ".meta.json"
DEGREE_GROUPS = ("bachelors", "masters", "phd")
OPT_COUNT_METHOD = "corrected_only_first_spell_split_ba_ma_phd_v2"
SCHOOL_BENCHMARK_METHOD = "split_ba_ma_phd_v1"
ANALYSIS_UNIVERSE_METHOD = "preferred_plus_outside_sample_v2"
NEW_HIRE_ORIGIN_METHOD = "heuristic_country_educ_position_v4_local_full_panel"
LOCAL_USER_PROFILE_CACHE_METHOD = "wrds_local_user_profile_origin_v2"
WORKFORCE_WRDS_EXTRACT_METHOD = "company_shift_share_workforce_wrds_extract_v3"
LOCAL_WORKFORCE_PANEL_METHOD = "duckdb_local_workforce_panel_v1"
LOCAL_SCHOOL_FLOWS_METHOD = "duckdb_local_school_flows_v1"
SOURCE_ANALYSIS_PANEL_METHOD = "firm_year_cartesian_with_origin_split_outcomes_v2"
DESIGN3_POSITION_OUTCOME_METHOD = "design3_position_outcomes_foia_soc2_recent_grad_v3_masters"
OPT_COUNT_COLUMNS = [f"{degree}_opt_hires_correction_aware" for degree in DEGREE_GROUPS] + [
    "any_opt_hires_correction_aware"
]
EDUCATION_DERIVED_WORKFORCE_COLUMNS = [
    "nonus_educ_share_annual",
    "age_share_lt30_annual",
    "age_share_30_39_annual",
    "age_share_40_49_annual",
    "age_share_50_59_annual",
    "age_share_60p_annual",
]
TOTAL_EMPLOYMENT_ORIGIN_WORKFORCE_COLUMNS = [
    "total_headcount_foreign_weighted_annual",
    "total_headcount_native_weighted_annual",
    "total_headcount_foreign_hard_annual",
    "total_headcount_native_hard_annual",
]
NEW_HIRE_ORIGIN_WORKFORCE_COLUMNS = [
    "n_new_hires_foreign_weighted_annual",
    "n_new_hires_native_weighted_annual",
    "n_new_hires_foreign_hard_annual",
    "n_new_hires_native_hard_annual",
]
ORIGIN_SPLIT_WORKFORCE_COLUMNS = (
    TOTAL_EMPLOYMENT_ORIGIN_WORKFORCE_COLUMNS + NEW_HIRE_ORIGIN_WORKFORCE_COLUMNS
)
LOCAL_PROFILE_WORKFORCE_COLUMNS = EDUCATION_DERIVED_WORKFORCE_COLUMNS + ORIGIN_SPLIT_WORKFORCE_COLUMNS
LOCAL_PROFILE_VALIDATION_COLUMNS = [
    "local_total_headcount_wrds_annual",
    "local_n_new_hires_wrds_annual",
]
_FEMALE_COL_CANDIDATES = ["f_prob", "female_prob", "gender_prob_female", "prob_female", "female"]
_RACE_PROB_COLS = [
    "white_prob",
    "black_prob",
    "api_prob",
    "hispanic_prob",
    "native_prob",
    "multiple_prob",
]
WORKFORCE_WRDS_USERS_COLUMNS = [
    "user_id",
    "user_location",
    "user_country",
    "female_prob",
    "white_prob",
    "black_prob",
    "api_prob",
    "hispanic_prob",
    "native_prob",
    "multiple_prob",
    "updated_dt",
    "university_name",
    "rsid",
    "education_number",
    "ed_startdate",
    "ed_enddate",
    "degree",
    "field",
    "university_country",
    "university_location",
    "university_raw",
    "degree_raw",
    "field_raw",
    "description",
]
WORKFORCE_WRDS_POSITION_HISTORY_COLUMNS = [
    "user_id",
    "country",
]
WORKFORCE_SELECTED_US_POSITIONS_COLUMNS = [
    "user_id",
    "position_id",
    "position_number",
    "rcid",
    "country",
    "startdate",
    "enddate",
    "onet_code",
    "seniority_raw",
    "salary",
    "total_compensation",
]
_LOCAL_WRDS_USERS_PATH_CANDIDATES = [
    "{root}/data/int/wrds_users_feb2026.parquet",
    "{root}/data/int/f1_indiv_merge/02_rev_import/wrds_users_apr2026v1.parquet",
]
_LOCAL_WRDS_POSITIONS_PATH_CANDIDATES = [
    "{root}/data/int/wrds_positions_jul31.parquet",
    "{root}/data/int/wrds_positions_feb2026.parquet",
    "{root}/data/int/f1_indiv_merge/02_rev_import/wrds_positions_apr2026v1.parquet",
]
_SOURCE_FIRM_UNIVERSE_CACHE: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]] = {}
_WRDS_WORKFORCE_CACHE: dict[str, tuple[pd.DataFrame, dict]] = {}
_WRDS_WORKFORCE_LOCAL_EXTRACT_CACHE: dict[str, tuple[Path, Path, Path, dict]] = {}
_LOCAL_USER_PROFILE_CACHE: dict[str, tuple[Path, dict]] = {}


@dataclass(frozen=True)
class SourceExposurePaths:
    foia_corrected: Path
    employer_crosswalk: Path
    preferred_rcids: Path
    company_mapping: Path
    f1_inst_unitid_crosswalk: Path
    revelio_inst_crosswalk: Path
    revelio_inst_deterministic_map: Path | None
    revelio_ref_inst_catalog: Path | None
    source_opt_counts_out: Path
    school_opt_benchmark_out: Path
    opt_exposure_analysis_panel_out: Path
    outside_negative_sample_out: Path
    wrds_company_year_workforce_out: Path
    wrds_school_flows_out: Path


@dataclass(frozen=True)
class WrdsQueryTask:
    rcids: tuple[int, ...]
    label: str
    year_min: int
    year_max: int
    history_year_min: int
    history_year_max: int


def _escape(path: Path) -> str:
    return str(path).replace("'", "''")


def _resolve_path(paths_cfg: dict, key: str, *, allow_missing: bool = False) -> Path:
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"Config paths.{key} must be set.")
    root = str(Path(__file__).resolve().parents[2])
    path = Path(str(value).replace("{root}", root))
    if not allow_missing and not path.exists():
        raise FileNotFoundError(f"Required path does not exist: {path}")
    return path


def _resolve_optional_path(paths_cfg: dict, key: str) -> Path | None:
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    root = str(Path(__file__).resolve().parents[2])
    return Path(str(value).replace("{root}", root))


def _resolve_optional_existing_path(paths_cfg: dict, key: str) -> Path | None:
    path = _resolve_optional_path(paths_cfg, key)
    if path is None or not path.exists():
        return None
    return path


def _candidate_existing_path(candidates: list[str]) -> Path | None:
    root = str(Path(__file__).resolve().parents[2])
    for candidate in candidates:
        path = Path(candidate.replace("{root}", root))
        if path.exists():
            return path
    return None


def _resolve_cfg_path_value(value: object) -> Path | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    root = str(Path(__file__).resolve().parents[2])
    return Path(str(value).replace("{root}", root))


def _cfg_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _normalize_design3_opt_likely_soc2(value: object) -> list[str]:
    out: list[str] = []
    for raw in _cfg_list(value):
        text = str(raw).strip()
        if not text:
            continue
        text = text.split("-")[0].strip()
        if text.isdigit():
            text = f"{int(text):02d}"
        if len(text) != 2:
            raise ValueError(f"Design 3 OPT-likely SOC2 values must be two-digit SOC codes; got {raw!r}")
        out.append(text)
    return sorted(dict.fromkeys(out))


def _selected_firms_hash(firms: pd.DataFrame) -> str:
    firm_ids = (
        pd.to_numeric(firms.get("c"), errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    payload = ",".join(str(v) for v in firm_ids).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _resolve_local_user_profile_refactor_settings(
    cfg: dict,
    *,
    workforce_out_path: Path,
) -> dict[str, object]:
    feature_cfg = get_cfg_section(cfg, "revelio_company_features")
    paths_cfg = get_cfg_section(cfg, "paths")
    requested = bool(feature_cfg.get("wrds_workforce_use_local_user_profile_cache", False))
    use_dedicated_extracts = bool(feature_cfg.get("wrds_workforce_build_local_extracts", False))
    configured_cache_path = _resolve_optional_path(paths_cfg, "wrds_user_profile_origin_out")
    cache_path = (
        apply_testing_output_suffix(configured_cache_path, cfg)
        if configured_cache_path is not None
        else workforce_out_path.with_name("wrds_user_profile_origin_cache.parquet")
    )
    extract_user_ids_path = _resolve_optional_path(paths_cfg, "wrds_workforce_user_ids_out")
    if extract_user_ids_path is None:
        extract_user_ids_path = workforce_out_path.with_name("wrds_workforce_user_ids.parquet")
    extract_user_ids_path = apply_testing_output_suffix(extract_user_ids_path, cfg)

    extract_users_path = _resolve_optional_path(paths_cfg, "wrds_workforce_users_out")
    if extract_users_path is None:
        extract_users_path = workforce_out_path.with_name("wrds_workforce_users.parquet")
    extract_users_path = apply_testing_output_suffix(extract_users_path, cfg)

    extract_positions_path = _resolve_optional_path(paths_cfg, "wrds_workforce_positions_out")
    if extract_positions_path is None:
        extract_positions_path = workforce_out_path.with_name("wrds_workforce_positions.parquet")
    extract_positions_path = apply_testing_output_suffix(extract_positions_path, cfg)

    extract_selected_positions_path = _resolve_optional_path(paths_cfg, "wrds_workforce_selected_us_positions_out")
    if extract_selected_positions_path is None:
        extract_selected_positions_path = workforce_out_path.with_name("wrds_workforce_selected_us_positions.parquet")
    extract_selected_positions_path = apply_testing_output_suffix(extract_selected_positions_path, cfg)

    extract_users_chunk_dir = _resolve_optional_path(paths_cfg, "wrds_workforce_users_chunk_dir")
    if extract_users_chunk_dir is None:
        extract_users_chunk_dir = workforce_out_path.with_name("wrds_workforce_users_chunks")
    extract_users_chunk_dir = apply_testing_output_suffix(extract_users_chunk_dir, cfg)

    extract_positions_chunk_dir = _resolve_optional_path(paths_cfg, "wrds_workforce_positions_chunk_dir")
    if extract_positions_chunk_dir is None:
        extract_positions_chunk_dir = workforce_out_path.with_name("wrds_workforce_positions_chunks")
    extract_positions_chunk_dir = apply_testing_output_suffix(extract_positions_chunk_dir, cfg)

    extract_selected_positions_chunk_dir = _resolve_optional_path(
        paths_cfg,
        "wrds_workforce_selected_us_positions_chunk_dir",
    )
    if extract_selected_positions_chunk_dir is None:
        extract_selected_positions_chunk_dir = workforce_out_path.with_name("wrds_workforce_selected_us_positions_chunks")
    extract_selected_positions_chunk_dir = apply_testing_output_suffix(extract_selected_positions_chunk_dir, cfg)

    snapshot_users_path = _resolve_optional_existing_path(paths_cfg, "wrds_users_local")
    if snapshot_users_path is None:
        snapshot_users_path = _candidate_existing_path(_LOCAL_WRDS_USERS_PATH_CANDIDATES)
    snapshot_positions_path = _resolve_optional_existing_path(paths_cfg, "wrds_positions_local")
    if snapshot_positions_path is None:
        snapshot_positions_path = _candidate_existing_path(_LOCAL_WRDS_POSITIONS_PATH_CANDIDATES)

    enabled = requested and (use_dedicated_extracts or (
        snapshot_users_path is not None and snapshot_positions_path is not None
    ))
    if requested and not enabled:
        _log(
            "[wrds_workforce_cache] Local user-profile refactor requested but local WRDS "
            "user/position parquet inputs were not found; falling back to the all-remote path"
        )
    return {
        "requested": requested,
        "enabled": enabled,
        "source_mode": "dedicated_wrds_extract" if use_dedicated_extracts else "snapshot",
        "use_dedicated_extracts": use_dedicated_extracts,
        "snapshot_users_path": snapshot_users_path,
        "snapshot_positions_path": snapshot_positions_path,
        "extract_user_ids_path": extract_user_ids_path,
        "extract_users_path": extract_users_path,
        "extract_positions_path": extract_positions_path,
        "extract_selected_positions_path": extract_selected_positions_path,
        "extract_users_chunk_dir": extract_users_chunk_dir,
        "extract_positions_chunk_dir": extract_positions_chunk_dir,
        "extract_selected_positions_chunk_dir": extract_selected_positions_chunk_dir,
        "extract_chunk_size": int(feature_cfg.get("wrds_workforce_extract_chunk_size", 20_000)),
        "extract_max_workers": max(1, int(feature_cfg.get("wrds_workforce_extract_max_workers", 3))),
        "cache_path": cache_path,
    }


def _coerce_year_like(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_shared_local_extract_year_window(
    cfg: dict,
    *,
    requested_year_min: int,
    requested_year_max: int,
) -> tuple[int, int, dict[str, object]]:
    requested_min = int(requested_year_min)
    requested_max = int(requested_year_max)
    if requested_min > requested_max:
        raise ValueError(
            "Requested WRDS local extract window is invalid: "
            f"{requested_min} > {requested_max}."
        )

    if testing_enabled(cfg):
        return requested_min, requested_max, {
            "shared_local_extract_year_window_method": "requested_window_testing_mode",
            "shared_local_extract_year_sources": [f"requested:{requested_min}-{requested_max}"],
        }

    windows: list[tuple[str, int, int]] = [("requested", requested_min, requested_max)]

    def _maybe_add_window(label: str, year_min_value: object, year_max_value: object) -> None:
        year_min = _coerce_year_like(year_min_value)
        year_max = _coerce_year_like(year_max_value)
        if year_min is None or year_max is None:
            return
        if year_min > year_max:
            raise ValueError(f"Invalid configured year window for {label}: {year_min} > {year_max}.")
        windows.append((label, year_min, year_max))

    feature_cfg = get_cfg_section(cfg, "revelio_company_features")
    _maybe_add_window(
        "revelio_company_features.feature_window",
        feature_cfg.get("feature_year_min"),
        feature_cfg.get("feature_year_max"),
    )

    exp_cfg = get_cfg_section(cfg, "exposure_event_study")
    _maybe_add_window(
        "exposure_event_study.legacy_exposure_window",
        exp_cfg.get("exposure_year_min"),
        exp_cfg.get("exposure_year_max"),
    )
    _maybe_add_window(
        "exposure_event_study.index_feature_window",
        exp_cfg.get("feature_year_min"),
        exp_cfg.get("feature_year_max"),
    )
    _maybe_add_window(
        "exposure_event_study.analysis_panel_window",
        exp_cfg.get("data_min_t"),
        exp_cfg.get("data_max_t"),
    )
    _maybe_add_window(
        "exposure_event_study.target_window",
        exp_cfg.get("target_year_min"),
        exp_cfg.get("target_year_max"),
    )

    extract_year_min = min(year_min for _, year_min, _ in windows)
    extract_year_max = max(year_max for _, _, year_max in windows)
    coverage_sources = [f"{label}:{year_min}-{year_max}" for label, year_min, year_max in windows]
    method = (
        "requested_plus_pipeline_windows"
        if extract_year_min != requested_min or extract_year_max != requested_max
        else "requested_window_only"
    )
    return extract_year_min, extract_year_max, {
        "shared_local_extract_year_window_method": method,
        "shared_local_extract_year_sources": coverage_sources,
    }


def resolve_source_exposure_paths(cfg: dict) -> SourceExposurePaths:
    paths_cfg = get_cfg_section(cfg, "paths")
    return SourceExposurePaths(
        foia_corrected=_resolve_path(paths_cfg, "foia_sevp_with_person_id_employment_corrected"),
        employer_crosswalk=_resolve_path(paths_cfg, "employer_crosswalk"),
        preferred_rcids=_resolve_path(paths_cfg, "preferred_rcids"),
        company_mapping=_resolve_path(paths_cfg, "revelio_company_mapping"),
        f1_inst_unitid_crosswalk=_resolve_path(paths_cfg, "f1_inst_unitid_crosswalk"),
        revelio_inst_crosswalk=_resolve_path(paths_cfg, "revelio_ipeds_foia_inst_crosswalk"),
        revelio_inst_deterministic_map=_resolve_optional_path(paths_cfg, "revelio_inst_deterministic_map"),
        revelio_ref_inst_catalog=_resolve_optional_path(paths_cfg, "revelio_ref_inst_catalog"),
        source_opt_counts_out=_resolve_path(paths_cfg, "source_opt_counts_out", allow_missing=True),
        school_opt_benchmark_out=_resolve_path(paths_cfg, "school_opt_benchmark_out", allow_missing=True),
        opt_exposure_analysis_panel_out=apply_testing_output_suffix(
            _resolve_path(paths_cfg, "opt_exposure_analysis_panel_out", allow_missing=True),
            cfg,
        ),
        outside_negative_sample_out=apply_testing_output_suffix(
            _resolve_path(paths_cfg, "outside_negative_sample_out", allow_missing=True),
            cfg,
        ),
        wrds_company_year_workforce_out=apply_testing_output_suffix(
            _resolve_path(paths_cfg, "wrds_company_year_workforce_out", allow_missing=True),
            cfg,
        ),
        wrds_school_flows_out=apply_testing_output_suffix(
            _resolve_path(paths_cfg, "wrds_school_flows_out", allow_missing=True),
            cfg,
        ),
    )


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + _META_SUFFIX)


def _legacy_cache_compat_mode() -> bool:
    return os.getenv("EXPOSURE_EVENT_STUDY_LEGACY_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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


def _log(message: str) -> None:
    print(message, flush=True)


def _reuse_cached_wrds_universe_only(cfg_full: dict) -> bool:
    return bool(cfg_full.get("reuse_cached_wrds_universe_only", False))


def _cache_meta_for_compatibility(
    expected_meta: dict,
    *,
    ignore_keys: Optional[set[str]] = None,
) -> dict:
    if not ignore_keys:
        return dict(expected_meta)
    return {k: v for k, v in expected_meta.items() if k not in ignore_keys}


def _wrds_universe_ignore_keys_for_extracts() -> set[str]:
    return {"selected_firms_hash", "selected_firms_n"}


def _wrds_universe_ignore_keys_for_panels() -> set[str]:
    return {
        "analysis_universe_method",
        "outside_negative_ratio",
        "outside_negative_seed",
        "outside_negative_min_n_users",
        "n_preferred_rcids",
        "preferred_sample_meta",
        "n_total_firms",
        "n_outside_negative_candidates",
        "analysis_firms_hash",
        "outside_negative_firms_hash",
    }


def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _source_firm_universe_cache_key(out_path: Path, metadata: dict) -> str:
    return json.dumps(
        {
            "out_path": str(out_path),
            "metadata": metadata,
        },
        sort_keys=True,
    )


def _clone_source_firm_universe_result(
    result: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    firms, outside, selected_meta, full_meta = result
    return (
        firms.copy(),
        outside.copy(),
        selected_meta.copy(),
        dict(full_meta),
    )


def _wrds_workforce_cache_key(out_path: Path, universe_meta: dict) -> str:
    return json.dumps(
        {
            "out_path": str(out_path),
            "universe_meta": universe_meta,
        },
        sort_keys=True,
    )


def _clone_wrds_workforce_cache_result(
    result: tuple[pd.DataFrame, dict],
) -> tuple[pd.DataFrame, dict]:
    df, meta = result
    return df.copy(), dict(meta)


def _wrds_workforce_local_extract_cache_key(
    user_ids_path: Path,
    users_out: Path,
    positions_out: Path,
    selected_positions_out: Path,
) -> str:
    return json.dumps(
        {
            "user_ids_path": str(user_ids_path),
            "users_out": str(users_out),
            "positions_out": str(positions_out),
            "selected_positions_out": str(selected_positions_out),
        },
        sort_keys=True,
    )


def _clone_wrds_workforce_local_extract_result(
    result: tuple[Path, Path, Path, dict],
) -> tuple[Path, Path, Path, dict]:
    users_out, positions_out, selected_positions_out, meta = result
    return users_out, positions_out, selected_positions_out, dict(meta)


def _local_user_profile_cache_key(out_path: Path) -> str:
    return json.dumps({"out_path": str(out_path)}, sort_keys=True)


def _clone_local_user_profile_cache_result(
    result: tuple[Path, dict],
) -> tuple[Path, dict]:
    out_path, meta = result
    return out_path, dict(meta)


def load_preferred_rcids(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = resolve_source_exposure_paths(cfg_full)
    rcids = pd.read_parquet(paths.preferred_rcids, columns=["preferred_rcid"]).rename(
        columns={"preferred_rcid": "c"}
    )
    rcids["c"] = pd.to_numeric(rcids["c"], errors="coerce")
    rcids = rcids.dropna(subset=["c"]).copy()
    rcids["c"] = rcids["c"].astype(int)
    return rcids.drop_duplicates(subset=["c"], keep="first").reset_index(drop=True)


def resolve_testing_sample_spec(
    cfg: dict,
    *,
    feature_year_min: Optional[int] = None,
    feature_year_max: Optional[int] = None,
) -> dict[str, int | bool]:
    testing_cfg = get_cfg_section(cfg, "testing")
    exp_cfg = get_cfg_section(cfg, "exposure_event_study")
    sample_year_min = int(
        testing_cfg.get(
            "analysis_sample_year_min",
            feature_year_min if feature_year_min is not None else exp_cfg.get("feature_year_min", 2010),
        )
    )
    sample_year_max = int(
        testing_cfg.get(
            "analysis_sample_year_max",
            feature_year_max if feature_year_max is not None else exp_cfg.get("feature_year_max", 2015),
        )
    )
    return {
        "enabled": bool(testing_cfg.get("enabled", False)),
        "analysis_sample_n": int(testing_cfg.get("analysis_sample_n", 50)),
        "analysis_sample_year_min": sample_year_min,
        "analysis_sample_year_max": sample_year_max,
        "analysis_min_post2016_positive": int(testing_cfg.get("analysis_min_post2016_positive", 10)),
        "analysis_min_post2016_nonpositive": int(testing_cfg.get("analysis_min_post2016_nonpositive", 10)),
        "random_seed": int(testing_cfg.get("random_seed", 42)),
        "target_year_min": int(exp_cfg.get("target_year_min", 2016)),
        "target_year_max": int(exp_cfg.get("target_year_max", 2022)),
    }


def _build_testing_analysis_firm_sample_from_counts(
    counts: pd.DataFrame,
    preferred_rcids: pd.DataFrame,
    *,
    sample_n: int,
    sample_year_min: int,
    sample_year_max: int,
    target_year_min: int,
    target_year_max: int,
    random_seed: int,
    min_positive: int,
    min_nonpositive: int,
) -> tuple[pd.DataFrame, dict]:
    if int(sample_n) <= 0:
        raise ValueError(f"testing.analysis_sample_n must be positive, got {sample_n}.")
    if int(sample_year_min) > int(sample_year_max):
        raise ValueError(
            "testing.analysis_sample_year_min must be <= testing.analysis_sample_year_max, "
            f"got {sample_year_min=} {sample_year_max=}."
        )

    preferred = preferred_rcids[["c"]].drop_duplicates().copy()
    preferred["c"] = pd.to_numeric(preferred["c"], errors="coerce")
    preferred = preferred.dropna(subset=["c"]).copy()
    preferred["c"] = preferred["c"].astype(int)

    work = counts.copy()
    work["c"] = pd.to_numeric(work["c"], errors="coerce")
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work["any_opt_hires_correction_aware"] = pd.to_numeric(
        work["any_opt_hires_correction_aware"],
        errors="coerce",
    ).fillna(0)
    work = work.dropna(subset=["c", "t"]).copy()
    work["c"] = work["c"].astype(int)
    work["t"] = work["t"].astype(int)

    eligible = (
        work[work["t"].between(sample_year_min, sample_year_max)][["c"]]
        .drop_duplicates()
        .merge(preferred, on="c", how="inner")
    )
    if eligible.empty:
        raise ValueError(
            "Testing-mode firm sample is empty: no preferred RCIDs appear in the requested "
            f"FOIA sample window [{sample_year_min}, {sample_year_max}]."
        )

    target = (
        work[work["t"].between(target_year_min, target_year_max)]
        .groupby("c", as_index=False)["any_opt_hires_correction_aware"]
        .max()
        .rename(columns={"any_opt_hires_correction_aware": "post2016_any_opt_raw"})
    )
    eligible = eligible.merge(target, on="c", how="left")
    eligible["post2016_any_opt"] = eligible["post2016_any_opt_raw"].fillna(0).gt(0).astype(int)

    rng = np.random.default_rng(int(random_seed))
    eligible["sample_draw"] = rng.random(len(eligible))
    eligible = eligible.sort_values("sample_draw").reset_index(drop=True)

    positive = eligible[eligible["post2016_any_opt"].eq(1)].copy()
    nonpositive = eligible[eligible["post2016_any_opt"].eq(0)].copy()
    n_requested = min(int(sample_n), len(eligible))

    n_positive = min(len(positive), max(0, min(int(min_positive), n_requested)))
    remaining = max(0, n_requested - n_positive)
    n_nonpositive = min(len(nonpositive), max(0, min(int(min_nonpositive), remaining)))

    selected = pd.concat(
        [
            positive.head(n_positive),
            nonpositive.head(n_nonpositive),
        ],
        ignore_index=True,
    )
    selected_ids = set(selected["c"].tolist())
    if len(selected) < n_requested:
        remainder = eligible[~eligible["c"].isin(selected_ids)].copy()
        if not remainder.empty:
            selected = pd.concat([selected, remainder.head(n_requested - len(selected))], ignore_index=True)

    selected = selected.drop_duplicates(subset=["c"], keep="first").reset_index(drop=True)
    meta = {
        "requested_sample_n": int(sample_n),
        "selected_sample_n": int(len(selected)),
        "eligible_sample_window_firms": int(len(eligible)),
        "eligible_post2016_positive_firms": int(len(positive)),
        "eligible_post2016_nonpositive_firms": int(len(nonpositive)),
        "selected_post2016_positive_firms": int(selected["post2016_any_opt"].sum()),
        "selected_post2016_nonpositive_firms": int(len(selected) - selected["post2016_any_opt"].sum()),
        "analysis_sample_year_min": int(sample_year_min),
        "analysis_sample_year_max": int(sample_year_max),
        "target_year_min": int(target_year_min),
        "target_year_max": int(target_year_max),
        "random_seed": int(random_seed),
        "analysis_min_post2016_positive": int(min_positive),
        "analysis_min_post2016_nonpositive": int(min_nonpositive),
    }
    return selected[["c"]].copy(), meta


def select_testing_analysis_firms(
    cfg: dict,
    *,
    feature_year_min: Optional[int] = None,
    feature_year_max: Optional[int] = None,
    force_rebuild_counts: bool = False,
) -> tuple[pd.DataFrame, dict]:
    preferred = load_preferred_rcids(cfg=cfg)
    spec = resolve_testing_sample_spec(
        cfg,
        feature_year_min=feature_year_min,
        feature_year_max=feature_year_max,
    )
    if not bool(spec["enabled"]):
        return preferred, {"testing_enabled": False, "selected_sample_n": int(len(preferred))}

    counts_year_min = min(int(spec["analysis_sample_year_min"]), int(spec["target_year_min"]))
    counts_year_max = max(int(spec["analysis_sample_year_max"]), int(spec["target_year_max"]))
    counts, counts_meta = load_or_build_source_opt_counts(
        cfg=cfg,
        year_min=counts_year_min,
        year_max=counts_year_max,
        force_rebuild=force_rebuild_counts,
    )
    selected, sample_meta = _build_testing_analysis_firm_sample_from_counts(
        counts,
        preferred,
        sample_n=int(spec["analysis_sample_n"]),
        sample_year_min=int(spec["analysis_sample_year_min"]),
        sample_year_max=int(spec["analysis_sample_year_max"]),
        target_year_min=int(spec["target_year_min"]),
        target_year_max=int(spec["target_year_max"]),
        random_seed=int(spec["random_seed"]),
        min_positive=int(spec["analysis_min_post2016_positive"]),
        min_nonpositive=int(spec["analysis_min_post2016_nonpositive"]),
    )
    return selected, {
        "testing_enabled": True,
        "sample_spec": spec,
        "sample_meta": sample_meta,
        "counts_meta": counts_meta,
    }


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
    preferred_rcids: pd.Series,
    *,
    ratio: float,
    seed: int,
    min_n_users: int,
) -> pd.DataFrame:
    preferred_ids = set(
        pd.Series(pd.to_numeric(preferred_rcids, errors="coerce")).dropna().astype(int).tolist()
    )
    analysis_meta = company_meta[company_meta["c"].isin(preferred_ids)].copy()
    eligible = company_meta[
        (~company_meta["c"].isin(preferred_ids))
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


def _shared_universe_settings(cfg_full: dict) -> dict[str, float | int]:
    feature_cfg = get_cfg_section(cfg_full, "revelio_company_features")
    return {
        "analysis_universe_method": ANALYSIS_UNIVERSE_METHOD,
        "outside_negative_ratio": float(feature_cfg.get("outside_negative_ratio", 2.0)),
        "outside_negative_seed": int(feature_cfg.get("outside_negative_seed", 42)),
        "outside_negative_min_n_users": int(feature_cfg.get("outside_negative_min_n_users", 10)),
    }


def load_or_build_source_firm_universe(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = resolve_source_exposure_paths(cfg_full)
    universe_meta = _shared_universe_settings(cfg_full)

    preferred, preferred_meta = select_testing_analysis_firms(
        cfg_full,
        force_rebuild_counts=force_rebuild,
    )
    preferred_firms_hash = _selected_firms_hash(preferred[["c"]])
    preferred["in_analysis_universe"] = 1
    preferred["preferred_rcid_source"] = 1
    preferred["outside_negative_candidate"] = 0

    outside_meta = {
        **universe_meta,
        "n_preferred_rcids": int(preferred["c"].nunique()),
        "preferred_firms_hash": preferred_firms_hash,
        "preferred_sample_meta": preferred_meta,
    }
    outside_path = paths.outside_negative_sample_out
    cache_key = _source_firm_universe_cache_key(outside_path, outside_meta)
    cached = _SOURCE_FIRM_UNIVERSE_CACHE.get(cache_key)
    if cached is not None:
        _log(f"[outside_negative_sample] Reusing in-process universe cache for {outside_path}")
        return _clone_source_firm_universe_result(cached)

    if outside_path.exists() and not force_rebuild:
        current_meta = _load_metadata(outside_path)
        if _metadata_compatible(current_meta, outside_meta):
            outside = pd.read_parquet(outside_path)
        else:
            outside = pd.DataFrame()
    else:
        outside = pd.DataFrame()

    if outside.empty:
        company_meta = _load_company_meta_subset(
            paths.company_mapping,
            min_n_users=int(universe_meta["outside_negative_min_n_users"]),
            eligible_only=True,
        )
        outside = sample_outside_negative_firms(
            company_meta,
            preferred["c"],
            ratio=float(universe_meta["outside_negative_ratio"]),
            seed=int(universe_meta["outside_negative_seed"]),
            min_n_users=int(universe_meta["outside_negative_min_n_users"]),
        )
        outside = outside[["c", "outside_negative_candidate"]].drop_duplicates(subset=["c"], keep="first")
        outside_path.parent.mkdir(parents=True, exist_ok=True)
        outside.to_parquet(outside_path, index=False)
        _write_metadata(
            outside_path,
            outside_meta | {"n_outside_negative_candidates": int(outside["c"].nunique())},
        )
        _log(f"[outside_negative_sample] Wrote {outside_path}")

    if outside.empty:
        outside = pd.DataFrame(columns=["c", "outside_negative_candidate"])
    outside["c"] = pd.to_numeric(outside["c"], errors="coerce")
    outside = outside.dropna(subset=["c"]).copy()
    outside["c"] = outside["c"].astype(int)
    outside = outside.drop_duplicates(subset=["c"], keep="first")
    outside["outside_negative_candidate"] = 1
    outside["preferred_rcid_source"] = 0
    outside["in_analysis_universe"] = 1

    firms = pd.concat(
        [
            preferred[["c", "in_analysis_universe", "preferred_rcid_source", "outside_negative_candidate"]],
            outside[["c", "in_analysis_universe", "preferred_rcid_source", "outside_negative_candidate"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["c"], keep="first")

    selected_meta = _load_company_meta_subset(
        paths.company_mapping,
        selected_firms=firms[["c"]],
    )
    full_meta = outside_meta | {
        "n_total_firms": int(firms["c"].nunique()),
        "n_outside_negative_candidates": int(outside["c"].nunique()),
        "analysis_firms_hash": _selected_firms_hash(firms[["c"]]),
        "outside_negative_firms_hash": _selected_firms_hash(outside[["c"]]),
    }
    result = (
        firms.reset_index(drop=True),
        outside.reset_index(drop=True),
        selected_meta,
        full_meta,
    )
    _SOURCE_FIRM_UNIVERSE_CACHE[cache_key] = _clone_source_firm_universe_result(result)
    return _clone_source_firm_universe_result(result)


def _has_column(con: ddb.DuckDBPyConnection, view: str, column: str) -> bool:
    rows = con.execute(f"PRAGMA table_info('{view}')").fetchall()
    cols = {str(r[1]).lower() for r in rows}
    return column.lower() in cols


def _degree_group_case(
    con: ddb.DuckDBPyConnection,
    view: str,
 ) -> str:
    if _has_column(con, view, "student_edu_level_desc"):
        return (
            "CASE "
            "WHEN LOWER(CAST(student_edu_level_desc AS VARCHAR)) IN ('bachelor''s', 'bachelors') "
            "  OR LOWER(CAST(student_edu_level_desc AS VARCHAR)) LIKE '%bachelor%' "
            "THEN 'bachelors' "
            "WHEN LOWER(CAST(student_edu_level_desc AS VARCHAR)) IN ('master''s', 'masters') "
            "  OR LOWER(CAST(student_edu_level_desc AS VARCHAR)) LIKE '%master%' "
            "THEN 'masters' "
            "WHEN LOWER(CAST(student_edu_level_desc AS VARCHAR)) IN ('doctorate', 'doctor', 'phd') "
            "  OR LOWER(CAST(student_edu_level_desc AS VARCHAR)) LIKE '%doctor%' "
            "  OR LOWER(CAST(student_edu_level_desc AS VARCHAR)) LIKE '%phd%' "
            "THEN 'phd' "
            "ELSE NULL END"
        )

    if _has_column(con, view, "awlevel_group"):
        return (
            "CASE "
            "WHEN LOWER(CAST(awlevel_group AS VARCHAR)) IN ('bachelor', 'bachelors') "
            "  OR LOWER(CAST(awlevel_group AS VARCHAR)) LIKE '%bachelor%' "
            "THEN 'bachelors' "
            "WHEN LOWER(CAST(awlevel_group AS VARCHAR)) IN ('master', 'masters') "
            "  OR LOWER(CAST(awlevel_group AS VARCHAR)) LIKE '%master%' "
            "THEN 'masters' "
            "WHEN LOWER(CAST(awlevel_group AS VARCHAR)) IN ('doctor', 'doctors', 'doctorate', 'doctorates', 'phd') "
            "  OR LOWER(CAST(awlevel_group AS VARCHAR)) LIKE '%doctor%' "
            "  OR LOWER(CAST(awlevel_group AS VARCHAR)) LIKE '%phd%' "
            "THEN 'phd' "
            "ELSE NULL END"
        )

    if _has_column(con, view, "awlevel"):
        return (
            "CASE "
            "WHEN CAST(awlevel AS INTEGER) = 5 THEN 'bachelors' "
            "WHEN CAST(awlevel AS INTEGER) = 7 THEN 'masters' "
            "WHEN CAST(awlevel AS INTEGER) IN (9, 17) THEN 'phd' "
            "ELSE NULL END"
        )

    return "NULL::VARCHAR"


def _sql_normalize(col: str) -> str:
    return (
        "TRIM(REGEXP_REPLACE(LOWER(CAST("
        f"{col}"
        " AS VARCHAR)), '[^a-z0-9]+', ' ', 'g'))"
    )


def _sql_clean_company_name(col: str) -> str:
    return (
        "TRIM(REGEXP_REPLACE("
        "REGEXP_REPLACE("
        "REGEXP_REPLACE("
        f"LOWER(CAST({col} AS VARCHAR)), "
        "'[^a-z0-9 ]+', ' ', 'g'), "
        "'\\b(inc|llc|ltd|corp|co|corporation|company|limited|incorporated)\\b', ' ', 'g'), "
        "'\\s+', ' ', 'g'))"
    )


def _sql_state_name_to_abbr(statecol: str) -> str:
    return f"UPPER(TRIM(CAST({statecol} AS VARCHAR)))"


def _sql_clean_zip(zipcol: str) -> str:
    zip_digits = f"TRIM(CAST(REGEXP_REPLACE({zipcol}, '[^0-9]', '', 'g') AS VARCHAR))"
    return f"""
        CASE
            WHEN LENGTH({zip_digits}) = 4 THEN '0' || {zip_digits}
            WHEN LENGTH({zip_digits}) >= 5 THEN SUBSTRING({zip_digits} FROM 1 FOR 5)
            ELSE {zip_digits}
        END
    """


def _date_parse_sql(col: str) -> str:
    return f"""
        COALESCE(
            TRY_CAST({col} AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%Y-%m-%d') AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%m/%d/%Y') AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%Y-%m-%d %H:%M:%S') AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%m/%d/%Y %H:%M:%S') AS DATE)
        )
    """


def _sql_country_token(expr: str) -> str:
    return (
        "LOWER(TRIM(REGEXP_REPLACE("
        "REGEXP_REPLACE("
        f"CAST({expr} AS VARCHAR), "
        "'[^A-Za-z]+', ' ', 'g'), "
        "'\\s+', ' ', 'g')))"
    )


def _sql_is_us_country(expr: str) -> str:
    token = _sql_country_token(expr)
    return (
        f"{token} IN ("
        "'united states', "
        "'united states of america', "
        "'usa', "
        "'us', "
        "'u s', "
        "'u s a'"
        ")"
    )


def _sql_nonus_country_signal_expr(expr: str) -> str:
    return f"""
        CASE
            WHEN {expr} IS NULL OR TRIM(CAST({expr} AS VARCHAR)) = '' THEN NULL::DOUBLE PRECISION
            WHEN {_sql_is_us_country(expr)} THEN 0.0::DOUBLE PRECISION
            ELSE 1.0::DOUBLE PRECISION
        END
    """


def _sql_new_hire_origin_probability_expr(signal_cols: list[str]) -> str:
    if not signal_cols:
        return "0.0::DOUBLE PRECISION"
    numerator = " + ".join(f"COALESCE({col}, 0.0)" for col in signal_cols)
    denominator = " + ".join(f"CASE WHEN {col} IS NULL THEN 0 ELSE 1 END" for col in signal_cols)
    return (
        f"COALESCE((({numerator}) / NULLIF(({denominator}), 0)), 0.0)::DOUBLE PRECISION"
    )


def _sql_new_hire_origin_hard_expr(prob_col: str, current_country_signal_col: str) -> str:
    return (
        "CASE "
        f"WHEN {prob_col} > 0.5 THEN 1 "
        f"WHEN ABS({prob_col} - 0.5) < 1e-12 AND {current_country_signal_col} = 1 THEN 1 "
        "ELSE 0 END"
    )


def _wrds_sql_column_or_null(
    alias: str,
    source_col: str,
    sql_type: str,
    available_cols: set[str],
    *,
    out_col: Optional[str] = None,
) -> str:
    target = out_col or source_col
    if source_col.lower() in available_cols:
        return f"CAST({alias}.{source_col} AS {sql_type}) AS {target}"
    return f"NULL::{sql_type} AS {target}"


def _resolve_wrds_extract_schema(
    db: wrds.Connection,
) -> dict[str, object]:
    user_cols = _table_columns(db, "individual_user")
    position_cols = _table_columns(db, "individual_positions")
    position_raw_cols = _table_columns(db, "individual_positions_raw")
    female_source_col = next((col for col in _FEMALE_COL_CANDIDATES if col in user_cols), None)
    return {
        "user_cols": user_cols,
        "position_cols": position_cols,
        "position_raw_cols": position_raw_cols,
        "female_source_col": female_source_col,
        "include_seniority": "seniority" in position_cols,
        "include_onet": "onet_code" in position_cols,
    }


def _local_parquet_columns(path: Path) -> set[str]:
    if not path.exists():
        return set()
    con = ddb.connect()
    try:
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_escape(path)}')"
        ).fetchall()
    finally:
        con.close()
    return {str(row[0]).lower() for row in rows}


def _register_source_foia_inputs(
    con: ddb.DuckDBPyConnection,
    paths: SourceExposurePaths,
) -> None:
    con.sql(
        f"CREATE OR REPLACE TEMP VIEW foia_raw AS "
        f"SELECT * FROM read_parquet('{_escape(paths.foia_corrected)}') "
        f"WHERE year_int > 2005"
    )
    con.sql(
        f"CREATE OR REPLACE TEMP VIEW preferred_rcids AS "
        f"SELECT DISTINCT CAST(preferred_rcid AS BIGINT) AS preferred_rcid "
        f"FROM read_parquet('{_escape(paths.preferred_rcids)}') "
        f"WHERE preferred_rcid IS NOT NULL"
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW employer_crosswalk AS
        SELECT ec.*
        FROM read_parquet('{_escape(paths.employer_crosswalk)}') AS ec
        JOIN preferred_rcids pr
          ON CAST(ec.preferred_rcid AS BIGINT) = pr.preferred_rcid
        """
    )


def build_source_opt_counts(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    year_min: int = 2006,
    year_max: int = 2023,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = resolve_source_exposure_paths(cfg_full)
    out_path = paths.source_opt_counts_out
    metadata = {
        "year_min": int(year_min),
        "year_max": int(year_max),
        "opt_count_method": OPT_COUNT_METHOD,
        "degree_groups": list(DEGREE_GROUPS),
    }
    if out_path.exists() and not force_rebuild:
        current_meta = _load_metadata(out_path)
        if _metadata_compatible(
            current_meta,
            {
                "year_min": metadata["year_min"],
                "year_max": metadata["year_max"],
                "opt_count_method": metadata["opt_count_method"],
            },
        ):
            return pd.read_parquet(out_path)

    con = ddb.connect()
    try:
        _register_source_foia_inputs(con, paths)
        degree_group_case = _degree_group_case(con, "foia_raw")

        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW foia_opt_authorizations_corrected AS
            WITH base AS (
                SELECT
                    person_id,
                    {_sql_clean_company_name('employer_name')} AS f1_empname_clean,
                    {_sql_normalize('employer_city')} AS f1_city_clean,
                    {_sql_state_name_to_abbr('employer_state')} AS f1_state_clean,
                    {_sql_clean_zip('employer_zip_code')} AS f1_zip_clean,
                    EXTRACT(YEAR FROM {_date_parse_sql('program_end_date')}) AS gradyear,
                    {degree_group_case} AS degree_group,
                    CASE
                        WHEN employer_name IS NOT NULL
                         AND COALESCE({_date_parse_sql('opt_employer_start_date')}, {_date_parse_sql('opt_authorization_start_date')}, {_date_parse_sql('authorization_start_date')})
                             >= {_date_parse_sql('program_end_date')}
                        THEN 1
                        ELSE 0
                    END AS valid_opt_hire,
                    COALESCE(
                        {_date_parse_sql('opt_employer_start_date')},
                        {_date_parse_sql('opt_authorization_start_date')},
                        {_date_parse_sql('authorization_start_date')}
                    ) AS spell_start_dt,
                    original_row_num
                FROM foia_raw
                WHERE person_id IS NOT NULL
                  AND employer_name IS NOT NULL
            ),
            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY person_id, f1_empname_clean, f1_city_clean, f1_state_clean, f1_zip_clean, degree_group
                        ORDER BY spell_start_dt ASC NULLS LAST, original_row_num ASC
                    ) AS spell_rank
                FROM base
                WHERE degree_group IS NOT NULL
            )
            SELECT
                person_id,
                f1_empname_clean,
                f1_city_clean,
                f1_state_clean,
                f1_zip_clean,
                gradyear,
                degree_group,
                valid_opt_hire
            FROM ranked
            WHERE spell_rank = 1
            """
        )
        counts = con.sql(
            f"""
            SELECT
                CAST(cw.preferred_rcid AS BIGINT) AS c,
                CAST(gradyear AS INTEGER) AS t,
                COUNT(DISTINCT CASE WHEN degree_group = 'bachelors' AND valid_opt_hire = 1 THEN person_id END)
                    AS bachelors_opt_hires_correction_aware,
                COUNT(DISTINCT CASE WHEN degree_group = 'masters' AND valid_opt_hire = 1 THEN person_id END)
                    AS masters_opt_hires_correction_aware,
                COUNT(DISTINCT CASE WHEN degree_group = 'phd' AND valid_opt_hire = 1 THEN person_id END)
                    AS phd_opt_hires_correction_aware,
                COUNT(DISTINCT CASE WHEN valid_opt_hire = 1 THEN person_id END)
                    AS any_opt_hires_correction_aware
            FROM foia_opt_authorizations_corrected AS f
            JOIN employer_crosswalk AS cw
              ON f.f1_empname_clean = cw.f1_empname_clean
             AND COALESCE(f.f1_city_clean, '') = COALESCE(cw.f1_city_clean, '')
             AND COALESCE(f.f1_state_clean, '') = COALESCE(cw.f1_state_clean, '')
             AND COALESCE(f.f1_zip_clean, '') = COALESCE(cw.f1_zip_clean, '')
            WHERE gradyear BETWEEN {int(year_min)} AND {int(year_max)}
              AND cw.preferred_rcid IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).df()
    finally:
        con.close()

    counts["c"] = pd.to_numeric(counts["c"], errors="coerce").astype(int)
    counts["t"] = pd.to_numeric(counts["t"], errors="coerce").astype(int)
    for col in OPT_COUNT_COLUMNS:
        counts[col] = pd.to_numeric(counts[col], errors="coerce").fillna(0).astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts.to_parquet(out_path, index=False)
    _write_metadata(out_path, metadata | {"n_rows": int(len(counts)), "n_firms": int(counts["c"].nunique())})
    print(f"[source_opt_counts] Wrote {out_path}")
    return counts


def load_or_build_source_opt_counts(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    year_min: int = 2006,
    year_max: int = 2023,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = resolve_source_exposure_paths(cfg_full)
    if paths.source_opt_counts_out.exists() and not force_rebuild:
        meta = _load_metadata(paths.source_opt_counts_out)
        if (
            meta.get("year_min") == int(year_min)
            and meta.get("year_max") == int(year_max)
            and meta.get("opt_count_method") == OPT_COUNT_METHOD
        ):
            return pd.read_parquet(paths.source_opt_counts_out), meta
    df = build_source_opt_counts(
        config_path=config_path,
        cfg=cfg_full,
        year_min=year_min,
        year_max=year_max,
        force_rebuild=force_rebuild,
    )
    return df, _load_metadata(paths.source_opt_counts_out)


def build_source_school_opt_benchmark(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    year_min: int = 2010,
    year_max: int = 2015,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = resolve_source_exposure_paths(cfg_full)
    out_path = paths.school_opt_benchmark_out
    metadata = {
        "year_min": int(year_min),
        "year_max": int(year_max),
        "school_benchmark_method": SCHOOL_BENCHMARK_METHOD,
        "degree_groups": list(DEGREE_GROUPS),
    }
    if out_path.exists() and not force_rebuild:
        current_meta = _load_metadata(out_path)
        if (
            current_meta.get("year_min") == metadata["year_min"]
            and current_meta.get("year_max") == metadata["year_max"]
            and current_meta.get("school_benchmark_method") == metadata["school_benchmark_method"]
        ):
            return pd.read_parquet(out_path)

    con = ddb.connect()
    try:
        con.sql(
            f"CREATE OR REPLACE TEMP VIEW foia_raw AS "
            f"SELECT * FROM read_parquet('{_escape(paths.foia_corrected)}') "
            f"WHERE year_int > 2005"
        )
        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW f1_inst_unitid_cw AS
            WITH base AS (
                SELECT
                    LOWER(TRIM(CAST(school_name AS VARCHAR))) AS school_name_key,
                    CAST(CAST(UNITID AS BIGINT) AS VARCHAR) AS unitid
                FROM read_parquet('{_escape(paths.f1_inst_unitid_crosswalk)}')
                WHERE school_name IS NOT NULL
                  AND UNITID IS NOT NULL
            )
            SELECT school_name_key, MIN(unitid) AS unitid
            FROM base
                GROUP BY 1
            """
        )
        degree_group_case = _degree_group_case(con, "foia_raw")

        school_rates = con.sql(
            f"""
            WITH base AS (
                SELECT
                    f.person_id,
                    CAST(EXTRACT(YEAR FROM {_date_parse_sql('program_end_date')}) AS INTEGER) AS gradyear,
                    cw.unitid,
                    {degree_group_case} AS degree_group,
                    CASE
                        WHEN employer_name IS NOT NULL
                         AND COALESCE({_date_parse_sql('opt_employer_start_date')}, {_date_parse_sql('opt_authorization_start_date')}, {_date_parse_sql('authorization_start_date')})
                             >= {_date_parse_sql('program_end_date')}
                        THEN 1
                        ELSE 0
                    END AS valid_opt_hire
                FROM foia_raw f
                LEFT JOIN f1_inst_unitid_cw cw
                  ON LOWER(TRIM(CAST(f.school_name AS VARCHAR))) = cw.school_name_key
                WHERE person_id IS NOT NULL
                  AND school_name IS NOT NULL
            ),
            dedup AS (
                SELECT
                    person_id,
                    gradyear,
                    unitid,
                    degree_group,
                    MAX(valid_opt_hire) AS valid_opt_hire
                FROM base
                WHERE gradyear BETWEEN {int(year_min)} AND {int(year_max)}
                  AND unitid IS NOT NULL
                  AND degree_group IS NOT NULL
                GROUP BY 1, 2, 3, 4
            ),
            school_year AS (
                SELECT
                    unitid,
                    gradyear,
                    degree_group,
                    COUNT(DISTINCT person_id) AS n_students,
                    COUNT(DISTINCT CASE WHEN valid_opt_hire = 1 THEN person_id END) AS n_opt_students,
                    CASE
                        WHEN COUNT(DISTINCT person_id) = 0 THEN NULL
                        ELSE COUNT(DISTINCT CASE WHEN valid_opt_hire = 1 THEN person_id END)::DOUBLE
                             / COUNT(DISTINCT person_id)::DOUBLE
                    END AS school_opt_rate_year
                FROM dedup
                GROUP BY 1, 2, 3
            ),
            school_pre AS (
                SELECT
                    unitid,
                    degree_group,
                    AVG(school_opt_rate_year) AS school_opt_rate,
                    SUM(n_students) AS n_students_total,
                    SUM(n_opt_students) AS n_opt_students_total
                FROM school_year
                GROUP BY 1, 2
            ),
            med AS (
                SELECT degree_group, MEDIAN(school_opt_rate) AS med_rate
                FROM school_pre
                GROUP BY 1
            ),
            labeled AS (
                SELECT
                    sp.unitid,
                    sp.degree_group,
                    sp.school_opt_rate,
                    sp.n_students_total,
                    sp.n_opt_students_total,
                    CASE WHEN sp.school_opt_rate > med.med_rate THEN 1 ELSE 0 END AS opt_intensive
                FROM school_pre sp
                JOIN med
                  ON sp.degree_group = med.degree_group
            )
            SELECT
                unitid,
                MAX(CASE WHEN degree_group = 'bachelors' THEN school_opt_rate END) AS school_opt_rate_bachelors,
                MAX(CASE WHEN degree_group = 'masters' THEN school_opt_rate END) AS school_opt_rate_masters,
                MAX(CASE WHEN degree_group = 'phd' THEN school_opt_rate END) AS school_opt_rate_phd,
                MAX(CASE WHEN degree_group = 'bachelors' THEN n_students_total END) AS n_students_total_bachelors,
                MAX(CASE WHEN degree_group = 'masters' THEN n_students_total END) AS n_students_total_masters,
                MAX(CASE WHEN degree_group = 'phd' THEN n_students_total END) AS n_students_total_phd,
                MAX(CASE WHEN degree_group = 'bachelors' THEN n_opt_students_total END) AS n_opt_students_total_bachelors,
                MAX(CASE WHEN degree_group = 'masters' THEN n_opt_students_total END) AS n_opt_students_total_masters,
                MAX(CASE WHEN degree_group = 'phd' THEN n_opt_students_total END) AS n_opt_students_total_phd,
                MAX(CASE WHEN degree_group = 'bachelors' THEN opt_intensive END) AS opt_intensive_bachelors,
                MAX(CASE WHEN degree_group = 'masters' THEN opt_intensive END) AS opt_intensive_masters,
                MAX(CASE WHEN degree_group = 'phd' THEN opt_intensive END) AS opt_intensive_phd
            FROM labeled
            GROUP BY 1
            """
        ).df()
    finally:
        con.close()

    school_rates["unitid"] = school_rates["unitid"].astype(str)
    for degree in DEGREE_GROUPS:
        col = f"opt_intensive_{degree}"
        school_rates[col] = pd.to_numeric(school_rates[col], errors="coerce").fillna(0).astype(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    school_rates.to_parquet(out_path, index=False)
    _write_metadata(out_path, metadata | {"n_schools": int(len(school_rates))})
    print(f"[school_opt_benchmark] Wrote {out_path}")
    return school_rates


def load_or_build_source_school_opt_benchmark(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    year_min: int = 2010,
    year_max: int = 2015,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = resolve_source_exposure_paths(cfg_full)
    if paths.school_opt_benchmark_out.exists() and not force_rebuild:
        meta = _load_metadata(paths.school_opt_benchmark_out)
        if _metadata_compatible(
            meta,
            {
                "year_min": int(year_min),
                "year_max": int(year_max),
                "school_benchmark_method": SCHOOL_BENCHMARK_METHOD,
            },
        ):
            return pd.read_parquet(paths.school_opt_benchmark_out), meta
    df = build_source_school_opt_benchmark(
        config_path=config_path,
        cfg=cfg_full,
        year_min=year_min,
        year_max=year_max,
        force_rebuild=force_rebuild,
    )
    return df, _load_metadata(paths.school_opt_benchmark_out)


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
            _log(f"[wrds] {label} attempt {attempt}/{attempts}: query start")
            t0 = time.time()
            _set_statement_timeout(db, query_timeout_ms)
            df = db.raw_sql(sql)
            _log(
                f"[wrds] {label} attempt {attempt}/{attempts}: query done "
                f"in {_format_elapsed(time.time() - t0)}"
            )
            return df, db
        except Exception as exc:
            last_exc = exc
            _log(
                f"[wrds] {label} failed on attempt {attempt}/{attempts} "
                f"after {_format_elapsed(time.time() - t0)}: {exc}"
            )
            if attempt < attempts:
                try:
                    db.close()
                except Exception:
                    pass
                db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
    assert last_exc is not None
    raise last_exc


def _wrds_exception_is_timeout(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "statement timeout" in text
        or "querycanceled" in text
        or "canceling statement due to statement timeout" in text
        or "could not receive data from server" in text
        or "connection timed out" in text
        or "ssl syscall error: connection timed out" in text
    )


def _resolve_singleton_query_timeout_ms(
    query_timeout_ms: Optional[int],
    singleton_query_timeout_ms: Optional[int],
) -> Optional[int]:
    base_timeout_ms = None if query_timeout_ms is None or int(query_timeout_ms) <= 0 else int(query_timeout_ms)
    if singleton_query_timeout_ms is not None:
        relaxed_timeout_ms = int(singleton_query_timeout_ms)
        if relaxed_timeout_ms <= 0:
            return None
        if base_timeout_ms is None or relaxed_timeout_ms > base_timeout_ms:
            return relaxed_timeout_ms
        return None
    if base_timeout_ms is None:
        return None
    # Large employers can still time out after the recursive batch splitter bottoms
    # out at a singleton firm. Give that final query a longer leash before failing.
    return max(base_timeout_ms, 60 * 60_000)


def _resolve_large_firm_n_users_threshold(
    large_firm_n_users_threshold: Optional[float],
) -> Optional[float]:
    if large_firm_n_users_threshold is None:
        return None
    threshold = float(large_firm_n_users_threshold)
    return None if threshold <= 0 else threshold


def _rcid_chunks(rcids: list[int], batch_size: int) -> list[list[int]]:
    if batch_size <= 0 or len(rcids) <= batch_size:
        return [rcids]
    return [rcids[i : i + batch_size] for i in range(0, len(rcids), batch_size)]


def _normalize_rcid_n_users(
    rcid_n_users: Optional[pd.DataFrame | dict[int, float]],
) -> dict[int, float]:
    if rcid_n_users is None:
        return {}
    if isinstance(rcid_n_users, dict):
        out: dict[int, float] = {}
        for key, value in rcid_n_users.items():
            try:
                rcid = int(key)
                n_users = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(n_users):
                out[rcid] = n_users
        return out
    if not isinstance(rcid_n_users, pd.DataFrame):
        raise TypeError("rcid_n_users must be a DataFrame, dict, or None.")
    required = {"c", "n_users"}
    missing = required - set(rcid_n_users.columns)
    if missing:
        raise ValueError(f"rcid_n_users is missing required columns: {sorted(missing)}")
    work = rcid_n_users[["c", "n_users"]].copy()
    work["c"] = pd.to_numeric(work["c"], errors="coerce")
    work["n_users"] = pd.to_numeric(work["n_users"], errors="coerce")
    work = work.dropna(subset=["c", "n_users"]).copy()
    work["c"] = work["c"].astype(int)
    work = work.drop_duplicates(subset=["c"], keep="first")
    return {
        int(row.c): float(row.n_users)
        for row in work.itertuples(index=False)
        if np.isfinite(float(row.n_users))
    }


def _year_windows(year_min: int, year_max: int, span_years: int) -> list[tuple[int, int]]:
    if int(year_min) > int(year_max):
        raise ValueError(f"year_min must be <= year_max, got {year_min=} {year_max=}.")
    span = max(1, int(span_years))
    windows: list[tuple[int, int]] = []
    start = int(year_min)
    while start <= int(year_max):
        end = min(int(year_max), start + span - 1)
        windows.append((start, end))
        start = end + 1
    return windows


def _selected_positions_extract_large_firm_year_span(year_min: int, year_max: int) -> int:
    return max(1, int(year_max) - int(year_min) + 1)


def _build_wrds_task_queue(
    rcids: list[int],
    *,
    label_prefix: str,
    rcid_batch_size: int,
    year_min: int,
    year_max: int,
    rcid_n_users: Optional[pd.DataFrame | dict[int, float]] = None,
    large_firm_n_users_threshold: Optional[float] = None,
    large_firm_year_span: int = 1,
) -> tuple[deque[WrdsQueryTask], dict[str, int | float | None]]:
    unique_rcids = sorted({int(v) for v in rcids})
    size_map = _normalize_rcid_n_users(rcid_n_users)
    threshold = _resolve_large_firm_n_users_threshold(large_firm_n_users_threshold)

    large_rcids: list[int] = []
    regular_rcids: list[int] = []
    for rcid in unique_rcids:
        n_users = size_map.get(rcid)
        if threshold is not None and n_users is not None and n_users >= threshold:
            large_rcids.append(rcid)
        else:
            regular_rcids.append(rcid)

    regular_batches = _rcid_chunks(regular_rcids, int(rcid_batch_size))
    task_queue: deque[WrdsQueryTask] = deque(
        WrdsQueryTask(
            rcids=tuple(batch),
            label=f"{label_prefix} batch {idx}/{len(regular_batches)}",
            year_min=int(year_min),
            year_max=int(year_max),
            history_year_min=int(year_min),
            history_year_max=int(year_max),
        )
        for idx, batch in enumerate(regular_batches, start=1)
    )

    year_windows = _year_windows(int(year_min), int(year_max), max(1, int(large_firm_year_span)))
    for rcid in sorted(large_rcids, key=lambda value: (-size_map.get(value, -1.0), value)):
        for window_min, window_max in year_windows:
            year_label = f"y{window_min}" if window_min == window_max else f"y{window_min}-{window_max}"
            task_queue.append(
                WrdsQueryTask(
                    rcids=(int(rcid),),
                    label=f"{label_prefix} large-firm rcid={int(rcid)} {year_label}",
                    year_min=int(window_min),
                    year_max=int(window_max),
                    history_year_min=int(year_min),
                    history_year_max=int(window_max),
                )
            )

    return task_queue, {
        "n_regular_batches": int(len(regular_batches)),
        "n_large_firms": int(len(large_rcids)),
        "n_large_firm_tasks": int(len(large_rcids) * len(year_windows)),
        "large_firm_n_users_threshold": threshold,
        "large_firm_year_span": int(max(1, int(large_firm_year_span))),
    }


def _table_columns(db: wrds.Connection, table: str) -> set[str]:
    try:
        cols = db.raw_sql(
            f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'revelio'
              AND table_name = '{table}'
            """
        )
    except Exception:
        return set()
    return {str(v).lower() for v in cols["column_name"].tolist()}


def _latest_role_lookup_table(db: wrds.Connection) -> Optional[str]:
    try:
        tables = db.raw_sql(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'revelio'
              AND table_name LIKE 'individual_role_lookup%'
            """
        )["table_name"].tolist()
    except Exception:
        return None
    if not tables:
        return None

    def _sort_key(name: str) -> tuple[int, str]:
        suffix = str(name).replace("individual_role_lookup", "").replace("_v", "").replace("_", "")
        return (int(suffix) if suffix.isdigit() else 0, str(name))

    return sorted((str(t) for t in tables), key=_sort_key)[-1]


def _build_wrds_company_year_workforce_query(
    rcids: list[int],
    *,
    year_min: int,
    year_max: int,
    user_prob_cols: list[str],
    include_seniority: bool,
    female_col: Optional[str],
    include_onet: bool,
    include_education_features: bool = True,
    history_year_min: Optional[int] = None,
    history_year_max: Optional[int] = None,
) -> str:
    rcid_sql = ", ".join(str(int(v)) for v in rcids)
    history_year_min_int = int(year_min if history_year_min is None else history_year_min)
    history_year_max_int = int(year_max if history_year_max is None else history_year_max)
    profile_select = ",\n                ".join(
        [f"CAST(u.{col} AS DOUBLE PRECISION) AS {col}" for col in user_prob_cols]
    )
    if not profile_select:
        profile_select = "NULL::DOUBLE PRECISION AS _dummy_prob"
    profile_outer_cols = ",\n            ".join(user_prob_cols) if user_prob_cols else "_dummy_prob"
    seniority_source = "NULL::VARCHAR AS seniority_raw"
    if include_seniority:
        seniority_source = "CAST(seniority AS VARCHAR) AS seniority_raw"
    demo_join_cols = []
    for col in user_prob_cols:
        demo_join_cols.append(f"MAX(up.{col}) AS {col}")
    if not demo_join_cols:
        demo_join_cols.append("NULL::DOUBLE PRECISION AS _dummy_prob")
    demo_join_sql = ",\n            ".join(demo_join_cols)

    # Occupation shares derived from O*NET codes (natively in individual_positions).
    # O*NET codes are formatted as "15-1132.00"; the first 2 chars give the SOC
    # major group (e.g. "15" = computing/math).  When onet_code is unavailable,
    # all occ_share columns are NULL so the missing indicators fire correctly.
    occ_share_cols = [
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
    if include_onet:
        role_share_cases = {
            "occ_share_mgmt_annual": "ay.soc2 = '11'",
            "occ_share_business_finance_annual": "ay.soc2 = '13'",
            "occ_share_computing_math_annual": "ay.soc2 = '15'",
            "occ_share_engineering_annual": "ay.soc2 = '17'",
            "occ_share_science_annual": "ay.soc2 = '19'",
            "occ_share_community_legal_education_annual": "ay.soc2 IN ('21', '23', '25')",
            "occ_share_arts_media_annual": "ay.soc2 = '27'",
            "occ_share_healthcare_annual": "ay.soc2 IN ('29', '31')",
            "occ_share_sales_office_annual": "ay.soc2 IN ('41', '43')",
            "occ_share_manual_annual": "ay.soc2 IN ('47', '49', '51', '53')",
        }
        role_share_sql = ",\n        ".join(
            [f"AVG(CASE WHEN {expr} THEN 1.0 ELSE 0.0 END) AS {col}"
             for col, expr in role_share_cases.items()]
        )
    else:
        role_share_sql = ",\n        ".join(
            [f"NULL::DOUBLE PRECISION AS {col}" for col in occ_share_cols]
        )

    race_cols_sql = []
    for src_col, out_col in [
        ("white_prob", "race_share_white_annual"),
        ("black_prob", "race_share_black_annual"),
        ("api_prob", "race_share_api_annual"),
        ("hispanic_prob", "race_share_hispanic_annual"),
        ("native_prob", "race_share_native_annual"),
        ("multiple_prob", "race_share_multiple_annual"),
    ]:
        if src_col in user_prob_cols:
            race_cols_sql.append(f"AVG({src_col}) AS {out_col}")
        else:
            race_cols_sql.append(f"NULL::DOUBLE PRECISION AS {out_col}")

    # female_col is the resolved column name (e.g. 'female_prob') or None.
    if female_col:
        female_sql = f"AVG({female_col}) AS female_share_annual"
    else:
        female_sql = "NULL::DOUBLE PRECISION AS female_share_annual"

    # soc2 = first 2 chars of onet_code (e.g. "15" from "15-1132.00").
    onet_select = "CAST(onet_code AS VARCHAR) AS onet_code," if include_onet else ""
    if include_onet:
        soc2_expr = """MAX(
                CASE
                    WHEN p.onet_code IS NULL OR TRIM(p.onet_code) = '' THEN NULL
                    ELSE SUBSTRING(p.onet_code FROM 1 FOR 2)
                END
            ) AS soc2,"""
    else:
        soc2_expr = "NULL::VARCHAR AS soc2,"

    education_ctes_sql = f"""
    educ_raw_dedup AS MATERIALIZED (
        SELECT
            r.user_id,
            r.education_number,
            MAX(NULLIF(TRIM(CAST(r.degree_raw AS VARCHAR)), '')) AS degree_raw,
            MAX(NULLIF(TRIM(CAST(r.field_raw AS VARCHAR)), '')) AS field_raw,
            MAX(NULLIF(TRIM(CAST(r.university_raw AS VARCHAR)), '')) AS university_raw
        FROM revelio.individual_user_education_raw AS r
        JOIN company_users cu
          ON cu.user_id = r.user_id
        GROUP BY 1, 2
    ),
    educ_joined AS MATERIALIZED (
        SELECT
            e.user_id,
            e.education_number,
            CAST(e.startdate AS VARCHAR) AS ed_startdate,
            CAST(e.enddate AS VARCHAR) AS ed_enddate,
            CAST(e.degree AS VARCHAR) AS degree,
            CAST(COALESCE(r.degree_raw, '') AS VARCHAR) AS degree_raw,
            CAST(COALESCE(r.field_raw, '') AS VARCHAR) AS field_raw,
            CAST(COALESCE(r.university_raw, '') AS VARCHAR) AS university_raw,
            CAST(e.university_country AS VARCHAR) AS university_country
        FROM revelio.individual_user_education AS e
        JOIN company_users cu
          ON cu.user_id = e.user_id
        LEFT JOIN educ_raw_dedup AS r
          ON e.user_id = r.user_id
         AND e.education_number = r.education_number
    ),
    raw_country_counts AS (
        SELECT
            LOWER(TRIM(university_raw)) AS raw_key,
            university_country,
            COUNT(*) AS n,
            ROW_NUMBER() OVER (
                PARTITION BY LOWER(TRIM(university_raw))
                ORDER BY COUNT(*) DESC, university_country ASC
            ) AS rn
        FROM educ_joined
        WHERE university_country IS NOT NULL
          AND TRIM(university_country) <> ''
          AND university_raw IS NOT NULL
          AND TRIM(university_raw) <> ''
        GROUP BY 1, 2
    ),
    raw_country_mode AS (
        SELECT raw_key, university_country AS modal_university_country
        FROM raw_country_counts
        WHERE rn = 1
    ),
    educ_enriched AS MATERIALIZED (
        SELECT
            ej.*,
            {degree_clean_regex_sql()} AS degree_clean,
            COALESCE(NULLIF(TRIM(ej.university_country), ''), rcm.modal_university_country) AS university_country_filled
        FROM educ_joined ej
        LEFT JOIN raw_country_mode rcm
          ON LOWER(TRIM(ej.university_raw)) = rcm.raw_key
    ),
    user_educ AS MATERIALIZED (
        SELECT
            user_id,
            MAX(
                CASE
                    WHEN university_country_filled IS NOT NULL
                     AND TRIM(university_country_filled) <> ''
                     AND NOT {_sql_is_us_country("university_country_filled")}
                    THEN 1
                    ELSE 0
                END
            ) AS has_nonus_educ,
            CASE
                WHEN MAX(CASE WHEN degree_clean = 'High School' THEN 1 ELSE 0 END) = 1
                    THEN MAX(
                        CASE
                            WHEN degree_clean = 'High School' AND ed_enddate IS NOT NULL
                                THEN SUBSTRING(ed_enddate, 1, 4)::INT - 18
                            ELSE NULL
                        END
                    )
                ELSE MIN(
                    CASE
                        WHEN degree_clean = 'Non-Degree' OR degree_clean = 'Master' OR degree_clean = 'Doctor' OR degree_clean = 'MBA'
                            THEN NULL
                        WHEN ed_startdate IS NOT NULL
                            THEN SUBSTRING(ed_startdate, 1, 4)::INT - 18
                        WHEN ed_enddate IS NOT NULL AND NOT degree_clean = 'Associate'
                            THEN SUBSTRING(ed_enddate, 1, 4)::INT - 23
                        WHEN ed_enddate IS NOT NULL
                            THEN SUBSTRING(ed_enddate, 1, 4)::INT - 21
                        ELSE NULL
                    END
                )
            END AS est_yob
        FROM educ_enriched
        GROUP BY 1
    ),"""
    if include_education_features:
        education_join_sql = """
        LEFT JOIN user_educ ue
          ON ue.user_id = p.user_id"""
        has_nonus_educ_sql = "MAX(COALESCE(ue.has_nonus_educ, 0)) AS has_nonus_educ,"
        est_yob_sql = "MAX(ue.est_yob) AS est_yob,"
        education_metric_sql = """
        AVG(CASE WHEN ay.has_nonus_educ = 1 THEN 1 ELSE 0 END) AS nonus_educ_share_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) < 30 THEN 1 ELSE 0 END) AS age_share_lt30_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) BETWEEN 30 AND 39 THEN 1 ELSE 0 END) AS age_share_30_39_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) BETWEEN 40 AND 49 THEN 1 ELSE 0 END) AS age_share_40_49_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) BETWEEN 50 AND 59 THEN 1 ELSE 0 END) AS age_share_50_59_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) >= 60 THEN 1 ELSE 0 END) AS age_share_60p_annual,"""
    else:
        education_join_sql = ""
        has_nonus_educ_sql = "NULL::DOUBLE PRECISION AS has_nonus_educ,"
        est_yob_sql = "NULL::INTEGER AS est_yob,"
        education_metric_sql = """
        NULL::DOUBLE PRECISION AS nonus_educ_share_annual,
        NULL::DOUBLE PRECISION AS age_share_lt30_annual,
        NULL::DOUBLE PRECISION AS age_share_30_39_annual,
        NULL::DOUBLE PRECISION AS age_share_40_49_annual,
        NULL::DOUBLE PRECISION AS age_share_50_59_annual,
        NULL::DOUBLE PRECISION AS age_share_60p_annual,"""

    return f"""
    WITH us_positions AS MATERIALIZED (
        SELECT
            user_id,
            rcid,
            startdate::DATE AS startdate,
            COALESCE(enddate::DATE, DATE '2025-12-31') AS enddate,
            {onet_select}
            CAST(salary AS VARCHAR) AS salary_raw,
            CAST(total_compensation AS VARCHAR) AS total_comp_raw,
            {seniority_source}
        FROM revelio.individual_positions
        WHERE {_sql_is_us_country("country")}
          AND rcid IN ({rcid_sql})
          AND startdate IS NOT NULL
          AND EXTRACT(YEAR FROM startdate)::INT <= {history_year_max_int}
          AND EXTRACT(YEAR FROM COALESCE(enddate::DATE, DATE '2025-12-31'))::INT >= {history_year_min_int}
    ),
    company_users AS MATERIALIZED (
        SELECT DISTINCT user_id
        FROM us_positions
    ),
    user_profile_raw AS MATERIALIZED (
        SELECT
            u.user_id,
            {profile_select},
            CAST(u.user_country AS VARCHAR) AS user_country_raw,
            ROW_NUMBER() OVER (
                PARTITION BY u.user_id
                ORDER BY NULLIF(TRIM(CAST(u.updated_dt AS VARCHAR)), '') DESC NULLS LAST,
                         CAST(u.user_country AS VARCHAR) DESC NULLS LAST,
                         u.user_id
            ) AS rn
        FROM revelio.individual_user AS u
        JOIN company_users cu
          ON cu.user_id = u.user_id
    ),
    user_profile AS MATERIALIZED (
        SELECT
            user_id,
            {profile_outer_cols},
            {_sql_nonus_country_signal_expr("user_country_raw")} AS signal_current_country_nonus
        FROM user_profile_raw
        WHERE rn = 1
    ),
    {education_ctes_sql}
    user_position_history AS MATERIALIZED (
        SELECT
            p.user_id,
            MAX(CASE WHEN p.country IS NOT NULL AND TRIM(CAST(p.country AS VARCHAR)) <> '' THEN 1 ELSE 0 END)
                AS has_any_position_country,
            MAX(
                CASE
                    WHEN p.country IS NOT NULL
                     AND TRIM(CAST(p.country AS VARCHAR)) <> ''
                     AND NOT {_sql_is_us_country("p.country")}
                    THEN 1
                    ELSE 0
                END
            ) AS has_nonus_position
        FROM revelio.individual_positions AS p
        JOIN company_users cu
          ON cu.user_id = p.user_id
        GROUP BY 1
    ),
    user_origin_signals AS MATERIALIZED (
        SELECT
            cu.user_id,
            up.signal_current_country_nonus,
            CASE
                WHEN ue.user_id IS NULL THEN NULL::DOUBLE PRECISION
                ELSE COALESCE(ue.has_nonus_educ, 0)::DOUBLE PRECISION
            END AS signal_nonus_educ,
            CASE
                WHEN COALESCE(uph.has_any_position_country, 0) = 0 THEN NULL::DOUBLE PRECISION
                ELSE COALESCE(uph.has_nonus_position, 0)::DOUBLE PRECISION
            END AS signal_nonus_position
        FROM company_users AS cu
        LEFT JOIN user_profile AS up
          ON up.user_id = cu.user_id
        LEFT JOIN user_educ AS ue
          ON ue.user_id = cu.user_id
        LEFT JOIN user_position_history AS uph
          ON uph.user_id = cu.user_id
    ),
    user_origin_scored AS MATERIALIZED (
        SELECT
            user_id,
            signal_current_country_nonus,
            signal_nonus_educ,
            signal_nonus_position,
            {_sql_new_hire_origin_probability_expr([
                "signal_current_country_nonus",
                "signal_nonus_educ",
                "signal_nonus_position",
            ])} AS p_likely_foreign
        FROM user_origin_signals
    ),
    user_origin AS MATERIALIZED (
        SELECT
            user_id,
            p_likely_foreign,
            (1.0 - p_likely_foreign)::DOUBLE PRECISION AS p_likely_native,
            {_sql_new_hire_origin_hard_expr("p_likely_foreign", "signal_current_country_nonus")}
                AS likely_foreign_hard
        FROM user_origin_scored
    ),
    company_user_first_start AS MATERIALIZED (
        SELECT
            user_id,
            rcid,
            MIN(startdate) AS first_start
        FROM us_positions
        GROUP BY 1, 2
    ),
    new_hire_origin AS MATERIALIZED (
        SELECT
            cufs.user_id,
            cufs.rcid,
            EXTRACT(YEAR FROM cufs.first_start)::INT AS year,
            uo.p_likely_foreign,
            uo.p_likely_native,
            uo.likely_foreign_hard
        FROM company_user_first_start AS cufs
        LEFT JOIN user_origin AS uo
          ON uo.user_id = cufs.user_id
        WHERE EXTRACT(YEAR FROM cufs.first_start)::INT BETWEEN {int(year_min)} AND {int(year_max)}
    ),
    new_hires AS (
        SELECT
            rcid,
            year,
            COUNT(DISTINCT user_id) AS n_new_hires_wrds_annual,
            SUM(COALESCE(p_likely_foreign, 0.0)) AS n_new_hires_foreign_weighted_annual,
            SUM(COALESCE(p_likely_native, 1.0)) AS n_new_hires_native_weighted_annual,
            SUM(COALESCE(likely_foreign_hard, 0)) AS n_new_hires_foreign_hard_annual,
            SUM(1 - COALESCE(likely_foreign_hard, 0)) AS n_new_hires_native_hard_annual
        FROM new_hire_origin
        GROUP BY 1, 2
    ),
    active_user_year AS (
        SELECT
            p.rcid,
            gs.year::INT AS year,
            p.user_id,
            MAX(NULLIF(TRIM(p.salary_raw), '')::DOUBLE PRECISION) AS salary,
            MAX(NULLIF(TRIM(p.total_comp_raw), '')::DOUBLE PRECISION) AS total_compensation,
            {has_nonus_educ_sql}
            {est_yob_sql}
            MAX(cufs.first_start) AS first_start,
            MAX(COALESCE(uo.p_likely_foreign, 0.0)) AS p_likely_foreign,
            MAX(COALESCE(uo.p_likely_native, 1.0)) AS p_likely_native,
            MAX(COALESCE(uo.likely_foreign_hard, 0)) AS likely_foreign_hard,
            {soc2_expr}
            MAX(
                CASE
                    WHEN p.seniority_raw IS NULL OR TRIM(p.seniority_raw) = '' THEN NULL
                    WHEN LOWER(p.seniority_raw) ~ '(intern|entry|junior|jr)' THEN 1.0
                    WHEN LOWER(p.seniority_raw) ~ '(associate|mid|ic)' THEN 2.0
                    WHEN LOWER(p.seniority_raw) ~ '(senior|sr|lead|principal|staff)' THEN 3.0
                    WHEN LOWER(p.seniority_raw) ~ '(manager|director|head|vp|vice president|chief|president|founder|partner|owner|executive)' THEN 4.0
                    WHEN NULLIF(REGEXP_REPLACE(TRIM(p.seniority_raw), '[^0-9\\.-]', '', 'g'), '') IS NOT NULL
                        THEN NULLIF(REGEXP_REPLACE(TRIM(p.seniority_raw), '[^0-9\\.-]', '', 'g'), '')::DOUBLE PRECISION
                    ELSE NULL
                END
            ) AS seniority_numeric,
            {demo_join_sql}
        FROM us_positions p
        JOIN company_user_first_start cufs
          ON cufs.user_id = p.user_id
         AND cufs.rcid = p.rcid
        JOIN LATERAL generate_series(
            GREATEST(EXTRACT(YEAR FROM p.startdate)::INT, {int(year_min)}),
            LEAST(EXTRACT(YEAR FROM p.enddate)::INT, {int(year_max)})
        ) AS gs(year) ON TRUE
        {education_join_sql}
        LEFT JOIN user_profile up
          ON up.user_id = p.user_id
        LEFT JOIN user_origin uo
          ON uo.user_id = p.user_id
        GROUP BY 1, 2, 3
    )
    SELECT
        CAST(ay.rcid AS BIGINT) AS c,
        CAST(ay.year AS INTEGER) AS t,
        COUNT(DISTINCT ay.user_id) AS total_headcount_wrds_annual,
        SUM(ay.p_likely_foreign)::DOUBLE PRECISION AS total_headcount_foreign_weighted_annual,
        SUM(ay.p_likely_native)::DOUBLE PRECISION AS total_headcount_native_weighted_annual,
        SUM(ay.likely_foreign_hard)::DOUBLE PRECISION AS total_headcount_foreign_hard_annual,
        SUM((1 - ay.likely_foreign_hard))::DOUBLE PRECISION AS total_headcount_native_hard_annual,
        COUNT(DISTINCT CASE WHEN make_date(ay.year, 12, 31) >= ay.first_start + INTERVAL '365 days' THEN ay.user_id END)
            AS long_term_headcount_wrds_annual,
        AVG(ay.salary) AS salary_mean_annual,
        VAR_SAMP(ay.salary) AS salary_var_annual,
        AVG(ay.total_compensation) AS total_comp_mean_annual,
        VAR_SAMP(ay.total_compensation) AS total_comp_var_annual,
        AVG(CASE WHEN ay.salary IS NULL AND ay.total_compensation IS NULL THEN 1 ELSE 0 END)
            AS compensation_missing_share_annual,
        {education_metric_sql}
        {female_sql},
        {", ".join(race_cols_sql)},
        AVG(ay.seniority_numeric) AS seniority_mean_annual,
        AVG(GREATEST(0.0, (make_date(ay.year, 12, 31) - ay.first_start)::DOUBLE PRECISION / 365.25)) AS avg_tenure_years_annual,
        {role_share_sql},
        MAX(nh.n_new_hires_wrds_annual)::DOUBLE PRECISION AS n_new_hires_wrds_annual,
        MAX(nh.n_new_hires_foreign_weighted_annual)::DOUBLE PRECISION AS n_new_hires_foreign_weighted_annual,
        MAX(nh.n_new_hires_native_weighted_annual)::DOUBLE PRECISION AS n_new_hires_native_weighted_annual,
        MAX(nh.n_new_hires_foreign_hard_annual)::DOUBLE PRECISION AS n_new_hires_foreign_hard_annual,
        MAX(nh.n_new_hires_native_hard_annual)::DOUBLE PRECISION AS n_new_hires_native_hard_annual
    FROM active_user_year ay
    LEFT JOIN new_hires nh
      ON nh.rcid = ay.rcid
     AND nh.year = ay.year
    GROUP BY 1, 2
    ORDER BY 1, 2
    """


def _build_wrds_company_year_workforce_query_base_only(
    rcids: list[int],
    *,
    year_min: int,
    year_max: int,
    user_prob_cols: list[str],
    include_seniority: bool,
    female_col: Optional[str],
    include_onet: bool,
    history_year_min: Optional[int] = None,
    history_year_max: Optional[int] = None,
) -> str:
    rcid_sql = ", ".join(str(int(v)) for v in rcids)
    history_year_min_int = int(year_min if history_year_min is None else history_year_min)
    history_year_max_int = int(year_max if history_year_max is None else history_year_max)
    profile_select = ",\n                ".join(
        [f"CAST(u.{col} AS DOUBLE PRECISION) AS {col}" for col in user_prob_cols]
    )
    if not profile_select:
        profile_select = "NULL::DOUBLE PRECISION AS _dummy_prob"
    profile_outer_cols = ",\n            ".join(user_prob_cols) if user_prob_cols else "_dummy_prob"
    seniority_source = "NULL::VARCHAR AS seniority_raw"
    if include_seniority:
        seniority_source = "CAST(seniority AS VARCHAR) AS seniority_raw"
    demo_join_cols = []
    for col in user_prob_cols:
        demo_join_cols.append(f"MAX(up.{col}) AS {col}")
    if not demo_join_cols:
        demo_join_cols.append("NULL::DOUBLE PRECISION AS _dummy_prob")
    demo_join_sql = ",\n            ".join(demo_join_cols)

    occ_share_cols = [
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
    if include_onet:
        role_share_cases = {
            "occ_share_mgmt_annual": "ay.soc2 = '11'",
            "occ_share_business_finance_annual": "ay.soc2 = '13'",
            "occ_share_computing_math_annual": "ay.soc2 = '15'",
            "occ_share_engineering_annual": "ay.soc2 = '17'",
            "occ_share_science_annual": "ay.soc2 = '19'",
            "occ_share_community_legal_education_annual": "ay.soc2 IN ('21', '23', '25')",
            "occ_share_arts_media_annual": "ay.soc2 = '27'",
            "occ_share_healthcare_annual": "ay.soc2 IN ('29', '31')",
            "occ_share_sales_office_annual": "ay.soc2 IN ('41', '43')",
            "occ_share_manual_annual": "ay.soc2 IN ('47', '49', '51', '53')",
        }
        role_share_sql = ",\n        ".join(
            [
                f"AVG(CASE WHEN {expr} THEN 1.0 ELSE 0.0 END) AS {col}"
                for col, expr in role_share_cases.items()
            ]
        )
    else:
        role_share_sql = ",\n        ".join(
            [f"NULL::DOUBLE PRECISION AS {col}" for col in occ_share_cols]
        )

    race_cols_sql = []
    for src_col, out_col in [
        ("white_prob", "race_share_white_annual"),
        ("black_prob", "race_share_black_annual"),
        ("api_prob", "race_share_api_annual"),
        ("hispanic_prob", "race_share_hispanic_annual"),
        ("native_prob", "race_share_native_annual"),
        ("multiple_prob", "race_share_multiple_annual"),
    ]:
        if src_col in user_prob_cols:
            race_cols_sql.append(f"AVG({src_col}) AS {out_col}")
        else:
            race_cols_sql.append(f"NULL::DOUBLE PRECISION AS {out_col}")

    if female_col:
        female_sql = f"AVG({female_col}) AS female_share_annual"
    else:
        female_sql = "NULL::DOUBLE PRECISION AS female_share_annual"

    onet_select = "CAST(onet_code AS VARCHAR) AS onet_code," if include_onet else ""
    if include_onet:
        soc2_expr = """MAX(
                CASE
                    WHEN p.onet_code IS NULL OR TRIM(p.onet_code) = '' THEN NULL
                    ELSE SUBSTRING(p.onet_code FROM 1 FOR 2)
                END
            ) AS soc2,"""
    else:
        soc2_expr = "NULL::VARCHAR AS soc2,"

    return f"""
    WITH us_positions AS MATERIALIZED (
        SELECT
            user_id,
            rcid,
            startdate::DATE AS startdate,
            COALESCE(enddate::DATE, DATE '2025-12-31') AS enddate,
            {onet_select}
            CAST(salary AS VARCHAR) AS salary_raw,
            CAST(total_compensation AS VARCHAR) AS total_comp_raw,
            {seniority_source}
        FROM revelio.individual_positions
        WHERE {_sql_is_us_country("country")}
          AND rcid IN ({rcid_sql})
          AND startdate IS NOT NULL
          AND EXTRACT(YEAR FROM startdate)::INT <= {history_year_max_int}
          AND EXTRACT(YEAR FROM COALESCE(enddate::DATE, DATE '2025-12-31'))::INT >= {history_year_min_int}
    ),
    company_users AS MATERIALIZED (
        SELECT DISTINCT user_id
        FROM us_positions
    ),
    user_profile_raw AS MATERIALIZED (
        SELECT
            u.user_id,
            {profile_select},
            ROW_NUMBER() OVER (
                PARTITION BY u.user_id
                ORDER BY NULLIF(TRIM(CAST(u.updated_dt AS VARCHAR)), '') DESC NULLS LAST,
                         u.user_id
            ) AS rn
        FROM revelio.individual_user AS u
        JOIN company_users cu
          ON cu.user_id = u.user_id
    ),
    user_profile AS MATERIALIZED (
        SELECT
            user_id,
            {profile_outer_cols}
        FROM user_profile_raw
        WHERE rn = 1
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
    ),
    active_user_year AS (
        SELECT
            p.rcid,
            gs.year::INT AS year,
            p.user_id,
            MAX(NULLIF(TRIM(p.salary_raw), '')::DOUBLE PRECISION) AS salary,
            MAX(NULLIF(TRIM(p.total_comp_raw), '')::DOUBLE PRECISION) AS total_compensation,
            MAX(cufs.first_start) AS first_start,
            {soc2_expr}
            MAX(
                CASE
                    WHEN p.seniority_raw IS NULL OR TRIM(p.seniority_raw) = '' THEN NULL
                    WHEN LOWER(p.seniority_raw) ~ '(intern|entry|junior|jr)' THEN 1.0
                    WHEN LOWER(p.seniority_raw) ~ '(associate|mid|ic)' THEN 2.0
                    WHEN LOWER(p.seniority_raw) ~ '(senior|sr|lead|principal|staff)' THEN 3.0
                    WHEN LOWER(p.seniority_raw) ~ '(manager|director|head|vp|vice president|chief|president|founder|partner|owner|executive)' THEN 4.0
                    WHEN NULLIF(REGEXP_REPLACE(TRIM(p.seniority_raw), '[^0-9\\.-]', '', 'g'), '') IS NOT NULL
                        THEN NULLIF(REGEXP_REPLACE(TRIM(p.seniority_raw), '[^0-9\\.-]', '', 'g'), '')::DOUBLE PRECISION
                    ELSE NULL
                END
            ) AS seniority_numeric,
            {demo_join_sql}
        FROM us_positions p
        JOIN company_user_first_start cufs
          ON cufs.user_id = p.user_id
         AND cufs.rcid = p.rcid
        JOIN LATERAL generate_series(
            GREATEST(EXTRACT(YEAR FROM p.startdate)::INT, {int(year_min)}),
            LEAST(EXTRACT(YEAR FROM p.enddate)::INT, {int(year_max)})
        ) AS gs(year) ON TRUE
        LEFT JOIN user_profile up
          ON up.user_id = p.user_id
        GROUP BY 1, 2, 3
    )
    SELECT
        CAST(ay.rcid AS BIGINT) AS c,
        CAST(ay.year AS INTEGER) AS t,
        COUNT(DISTINCT ay.user_id) AS total_headcount_wrds_annual,
        NULL::DOUBLE PRECISION AS total_headcount_foreign_weighted_annual,
        NULL::DOUBLE PRECISION AS total_headcount_native_weighted_annual,
        NULL::DOUBLE PRECISION AS total_headcount_foreign_hard_annual,
        NULL::DOUBLE PRECISION AS total_headcount_native_hard_annual,
        COUNT(DISTINCT CASE WHEN make_date(ay.year, 12, 31) >= ay.first_start + INTERVAL '365 days' THEN ay.user_id END)
            AS long_term_headcount_wrds_annual,
        AVG(ay.salary) AS salary_mean_annual,
        VAR_SAMP(ay.salary) AS salary_var_annual,
        AVG(ay.total_compensation) AS total_comp_mean_annual,
        VAR_SAMP(ay.total_compensation) AS total_comp_var_annual,
        AVG(CASE WHEN ay.salary IS NULL AND ay.total_compensation IS NULL THEN 1 ELSE 0 END)
            AS compensation_missing_share_annual,
        NULL::DOUBLE PRECISION AS nonus_educ_share_annual,
        NULL::DOUBLE PRECISION AS age_share_lt30_annual,
        NULL::DOUBLE PRECISION AS age_share_30_39_annual,
        NULL::DOUBLE PRECISION AS age_share_40_49_annual,
        NULL::DOUBLE PRECISION AS age_share_50_59_annual,
        NULL::DOUBLE PRECISION AS age_share_60p_annual,
        {female_sql},
        {", ".join(race_cols_sql)},
        AVG(ay.seniority_numeric) AS seniority_mean_annual,
        AVG(GREATEST(0.0, (make_date(ay.year, 12, 31) - ay.first_start)::DOUBLE PRECISION / 365.25)) AS avg_tenure_years_annual,
        {role_share_sql},
        MAX(nh.n_new_hires_wrds_annual)::DOUBLE PRECISION AS n_new_hires_wrds_annual,
        NULL::DOUBLE PRECISION AS n_new_hires_foreign_weighted_annual,
        NULL::DOUBLE PRECISION AS n_new_hires_native_weighted_annual,
        NULL::DOUBLE PRECISION AS n_new_hires_foreign_hard_annual,
        NULL::DOUBLE PRECISION AS n_new_hires_native_hard_annual
    FROM active_user_year ay
    LEFT JOIN new_hires nh
      ON nh.rcid = ay.rcid
     AND nh.year = ay.year
    GROUP BY 1, 2
    ORDER BY 1, 2
    """


def build_wrds_company_year_workforce(
    rcids: list[int],
    *,
    wrds_username: str,
    year_min: int,
    year_max: int,
    rcid_batch_size: int,
    rcid_n_users: Optional[pd.DataFrame | dict[int, float]] = None,
    large_firm_n_users_threshold: Optional[float] = None,
    large_firm_year_span: int = 1,
    include_education_features: bool = True,
    query_timeout_ms: Optional[int],
    singleton_query_timeout_ms: Optional[int] = None,
    query_max_retries: int,
    compute_origin_and_education_locally: bool = False,
) -> pd.DataFrame:
    if not rcids:
        return pd.DataFrame(columns=["c", "t"] + ORIGIN_SPLIT_WORKFORCE_COLUMNS)
    if wrds is None:  # pragma: no cover
        raise ImportError("wrds is not installed.")

    build_t0 = time.time()
    relaxed_singleton_timeout_ms = _resolve_singleton_query_timeout_ms(
        query_timeout_ms=query_timeout_ms,
        singleton_query_timeout_ms=singleton_query_timeout_ms,
    )
    _log(
        f"[wrds_workforce] START: {len(rcids):,} firm ids | years {int(year_min)}–{int(year_max)} | "
        f"batch_size={int(rcid_batch_size)}"
    )
    if compute_origin_and_education_locally:
        _log(
            "[wrds_workforce] Using light remote base query for workforce totals/demographics; "
            "annual education/origin metrics will be merged from the local cached user-profile path"
        )
    if not include_education_features:
        _log(
            "[wrds_workforce] Education-derived features disabled: "
            "skipping annual age/non-US education features"
        )
    if relaxed_singleton_timeout_ms is not None and (
        query_timeout_ms is None or int(relaxed_singleton_timeout_ms) > int(query_timeout_ms)
    ):
        _log(
            "[wrds_workforce] Singleton fallback timeout: "
            f"{int(round(relaxed_singleton_timeout_ms / 60_000))} minutes"
        )
    db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
    positions_cols = _table_columns(db, "individual_positions")
    user_cols = _table_columns(db, "individual_user")
    include_seniority = "seniority" in positions_cols

    # Detect female probability column — the column name changed across WRDS releases.
    _female_col_candidates = ["f_prob", "female_prob", "gender_prob_female", "prob_female", "female"]
    female_col: Optional[str] = next((c for c in _female_col_candidates if c in user_cols), None)
    if female_col is None:
        _log(
            "[wrds_workforce] WARNING: none of the expected female-probability columns "
            f"({_female_col_candidates}) found in revelio.individual_user. "
            "female_share_annual will be NULL for all firms/years."
        )

    user_prob_cols = [
        col
        for col in [
            female_col,
            "white_prob",
            "black_prob",
            "api_prob",
            "hispanic_prob",
            "native_prob",
            "multiple_prob",
        ]
        if col is not None and col in user_cols
    ]

    # O*NET codes are natively in individual_positions and map directly to 2-digit SOC
    # major groups (first 2 chars, e.g. "15-1132.00" → "15" = computing/math).
    include_onet = "onet_code" in positions_cols
    if not include_onet:
        _log(
            "[wrds_workforce] WARNING: onet_code not found in revelio.individual_positions. "
            "occ_share columns will be NULL."
        )

    batch_queue, task_meta = _build_wrds_task_queue(
        rcids,
        label_prefix="workforce",
        rcid_batch_size=int(rcid_batch_size),
        year_min=int(year_min),
        year_max=int(year_max),
        rcid_n_users=rcid_n_users,
        large_firm_n_users_threshold=large_firm_n_users_threshold,
        large_firm_year_span=int(large_firm_year_span),
    )
    if int(task_meta["n_large_firms"]) > 0:
        threshold = task_meta["large_firm_n_users_threshold"]
        span = int(task_meta["large_firm_year_span"])
        _log(
            "[wrds_workforce] Proactive large-firm slicing: "
            f"{int(task_meta['n_large_firms']):,} firm(s) with n_users >= {int(threshold):,} "
            f"scheduled as {int(task_meta['n_large_firm_tasks']):,} singleton task(s) "
            f"using {span}-year windows"
        )
    frames: list[pd.DataFrame] = []
    try:
        while batch_queue:
            task = batch_queue.popleft()
            batch = list(task.rcids)
            label = task.label
            batch_t0 = time.time()
            batch_min = min(batch)
            batch_max = max(batch)
            task_years_msg = (
                f" | output years {int(task.year_min)}–{int(task.year_max)}"
                if int(task.year_min) != int(year_min) or int(task.year_max) != int(year_max)
                else ""
            )
            _log(
                f"[wrds_workforce] START {label}: {len(batch):,} firms | "
                f"rcid range {batch_min:,}–{batch_max:,}{task_years_msg}"
            )
            query_builder = (
                _build_wrds_company_year_workforce_query_base_only
                if compute_origin_and_education_locally
                else _build_wrds_company_year_workforce_query
            )
            query_kwargs = {
                "year_min": task.year_min,
                "year_max": task.year_max,
                "user_prob_cols": user_prob_cols,
                "include_seniority": include_seniority,
                "female_col": female_col,
                "include_onet": include_onet,
                "history_year_min": task.history_year_min,
                "history_year_max": task.history_year_max,
            }
            if not compute_origin_and_education_locally:
                query_kwargs["include_education_features"] = include_education_features
            sql = query_builder(
                batch,
                **query_kwargs,
            )
            try:
                df, db = _run_sql_with_retries(
                    db,
                    sql,
                    wrds_username=wrds_username,
                    query_timeout_ms=query_timeout_ms,
                    max_retries=query_max_retries,
                    label=label,
                )
            except Exception as exc:
                is_timeout = _wrds_exception_is_timeout(exc)
                if len(batch) > 1 and is_timeout:
                    mid = len(batch) // 2
                    left = batch[:mid]
                    right = batch[mid:]
                    _log(
                        f"[wrds_workforce] SPLIT {label}: timeout after "
                        f"{_format_elapsed(time.time() - batch_t0)} | "
                        f"{len(batch):,} firms -> {len(left):,} + {len(right):,} firms"
                    )
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
                    batch_queue.appendleft(
                        WrdsQueryTask(
                            rcids=tuple(right),
                            label=f"{label}.2",
                            year_min=int(task.year_min),
                            year_max=int(task.year_max),
                            history_year_min=int(task.history_year_min),
                            history_year_max=int(task.history_year_max),
                        )
                    )
                    batch_queue.appendleft(
                        WrdsQueryTask(
                            rcids=tuple(left),
                            label=f"{label}.1",
                            year_min=int(task.year_min),
                            year_max=int(task.year_max),
                            history_year_min=int(task.history_year_min),
                            history_year_max=int(task.history_year_max),
                        )
                    )
                    continue
                if len(batch) == 1 and is_timeout and relaxed_singleton_timeout_ms is not None:
                    relaxed_label = f"{label} singleton-fallback"
                    slice_windows = _year_windows(int(task.year_min), int(task.year_max), 1)
                    slice_mode = (
                        "with relaxed timeout"
                        if len(slice_windows) == 1
                        else "as 1-year windows with relaxed timeout"
                    )
                    _log(
                        f"[wrds_workforce] RETRY {label}: timed out after "
                        f"{_format_elapsed(time.time() - batch_t0)} | "
                        f"retrying rcid {batch[0]:,} {slice_mode} "
                        f"({int(round(relaxed_singleton_timeout_ms / 60_000))} minutes)"
                    )
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = _open_wrds_connection(
                        wrds_username=wrds_username,
                        query_timeout_ms=relaxed_singleton_timeout_ms,
                    )
                    try:
                        year_frames: list[pd.DataFrame] = []
                        for slice_year_min, slice_year_max in slice_windows:
                            slice_suffix = (
                                f"y{slice_year_min}"
                                if slice_year_min == slice_year_max
                                else f"y{slice_year_min}-{slice_year_max}"
                            )
                            slice_label = (
                                relaxed_label
                                if len(slice_windows) == 1
                                else f"{relaxed_label} {slice_suffix}"
                            )
                            slice_query_kwargs = {
                                "year_min": slice_year_min,
                                "year_max": slice_year_max,
                                "history_year_min": task.history_year_min,
                                "history_year_max": slice_year_max,
                                "user_prob_cols": user_prob_cols,
                                "include_seniority": include_seniority,
                                "female_col": female_col,
                                "include_onet": include_onet,
                            }
                            if not compute_origin_and_education_locally:
                                slice_query_kwargs["include_education_features"] = include_education_features
                            slice_sql = query_builder(
                                batch,
                                **slice_query_kwargs,
                            )
                            slice_df, db = _run_sql_with_retries(
                                db,
                                slice_sql,
                                wrds_username=wrds_username,
                                query_timeout_ms=relaxed_singleton_timeout_ms,
                                max_retries=query_max_retries,
                                label=slice_label,
                            )
                            year_frames.append(slice_df)
                        df = (
                            pd.concat(year_frames, ignore_index=True)
                            if year_frames
                            else pd.DataFrame(columns=["c", "t"])
                        )
                    except Exception as relaxed_exc:
                        if _wrds_exception_is_timeout(relaxed_exc):
                            _log(
                                f"[wrds_workforce] FATAL {label}: single-firm batch timed out for rcid "
                                f"{batch[0]:,} even after refined singleton fallback "
                                f"({_format_elapsed(time.time() - batch_t0)})"
                            )
                        raise
                elif len(batch) == 1 and is_timeout:
                    _log(
                        f"[wrds_workforce] FATAL {label}: single-firm batch timed out for rcid "
                        f"{batch[0]:,} after {_format_elapsed(time.time() - batch_t0)}"
                    )
                    raise
                else:
                    raise
            if not df.empty:
                df["c"] = pd.to_numeric(df["c"], errors="coerce").astype("Int64")
                df["t"] = pd.to_numeric(df["t"], errors="coerce").astype("Int64")
                df = df.dropna(subset=["c", "t"]).copy()
                df["c"] = df["c"].astype(int)
                df["t"] = df["t"].astype(int)
            frames.append(df)
            _log(
                f"[wrds_workforce] DONE  {label}: {len(df):,} rows | "
                f"elapsed {_format_elapsed(time.time() - batch_t0)}"
            )
    finally:
        try:
            db.close()
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["c", "t"])
    out = pd.concat(frames, ignore_index=True)
    _log(
        f"[wrds_workforce] DONE: {len(out):,} rows before dedup | "
        f"elapsed {_format_elapsed(time.time() - build_t0)}"
    )
    return out.drop_duplicates(subset=["c", "t"], keep="first")


def _build_wrds_workforce_user_ids_query(
    rcids: list[int],
    *,
    history_year_min: int,
    history_year_max: int,
) -> str:
    rcid_sql = ", ".join(str(int(v)) for v in rcids)
    return f"""
    SELECT DISTINCT
        CAST(user_id AS BIGINT) AS user_id
    FROM revelio.individual_positions
    WHERE {_sql_is_us_country("country")}
      AND rcid IN ({rcid_sql})
      AND startdate IS NOT NULL
      AND EXTRACT(YEAR FROM startdate)::INT <= {int(history_year_max)}
      AND EXTRACT(YEAR FROM COALESCE(enddate::DATE, DATE '2025-12-31'))::INT >= {int(history_year_min)}
    ORDER BY 1
    """


def _format_user_id_sql_list(user_ids: list[int]) -> str:
    ids = [str(int(user_id)) for user_id in user_ids]
    if not ids:
        raise ValueError("Cannot build WRDS extract query with an empty user_id list.")
    return ",".join(ids)


def _build_wrds_users_extract_query(
    user_ids: list[int],
    *,
    user_cols: set[str],
    female_source_col: Optional[str],
) -> str:
    userid_subset = _format_user_id_sql_list(user_ids)
    female_sql = (
        f"CAST(u.{female_source_col} AS DOUBLE PRECISION) AS female_prob"
        if female_source_col is not None
        else "NULL::DOUBLE PRECISION AS female_prob"
    )
    race_prob_sql = ",\n                    ".join(
        _wrds_sql_column_or_null("u", col, "DOUBLE PRECISION", user_cols)
        for col in _RACE_PROB_COLS
    )
    return f"""
        WITH user_profile_raw AS (
            SELECT
                CAST(u.user_id AS BIGINT) AS user_id,
                {_wrds_sql_column_or_null("u", "user_location", "VARCHAR", user_cols)},
                {_wrds_sql_column_or_null("u", "user_country", "VARCHAR", user_cols)},
                {female_sql},
                {race_prob_sql},
                {_wrds_sql_column_or_null("u", "updated_dt", "VARCHAR", user_cols)},
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(u.user_id AS BIGINT)
                    ORDER BY NULLIF(TRIM(CAST(u.updated_dt AS VARCHAR)), '') DESC NULLS LAST,
                             CAST(u.user_id AS BIGINT)
                ) AS rn
            FROM revelio.individual_user AS u
            WHERE u.user_id IN ({userid_subset})
        ),
        user_profile AS (
            SELECT
                user_id,
                user_location,
                user_country,
                female_prob,
                white_prob,
                black_prob,
                api_prob,
                hispanic_prob,
                native_prob,
                multiple_prob,
                updated_dt
            FROM user_profile_raw
            WHERE rn = 1
        ),
        educ_raw_dedup AS (
            SELECT
                user_id,
                education_number,
                MAX(NULLIF(TRIM(CAST(university_raw AS VARCHAR)), '')) AS university_raw,
                MAX(NULLIF(TRIM(CAST(degree_raw AS VARCHAR)), '')) AS degree_raw,
                MAX(NULLIF(TRIM(CAST(field_raw AS VARCHAR)), '')) AS field_raw,
                MAX(NULLIF(TRIM(CAST(description AS VARCHAR)), '')) AS description
            FROM revelio.individual_user_education_raw
            WHERE user_id IN ({userid_subset})
            GROUP BY 1, 2
        ),
        educ_joined AS (
            SELECT
                CAST(e.user_id AS BIGINT) AS user_id,
                CAST(e.university_name AS VARCHAR) AS university_name,
                CAST(e.rsid AS VARCHAR) AS rsid,
                CAST(e.education_number AS BIGINT) AS education_number,
                CAST(e.startdate AS VARCHAR) AS ed_startdate,
                CAST(e.enddate AS VARCHAR) AS ed_enddate,
                CAST(e.degree AS VARCHAR) AS degree,
                CAST(e.field AS VARCHAR) AS field,
                CAST(e.university_country AS VARCHAR) AS university_country,
                CAST(e.university_location AS VARCHAR) AS university_location,
                CAST(r.university_raw AS VARCHAR) AS university_raw,
                CAST(r.degree_raw AS VARCHAR) AS degree_raw,
                CAST(r.field_raw AS VARCHAR) AS field_raw,
                CAST(r.description AS VARCHAR) AS description
            FROM revelio.individual_user_education AS e
            LEFT JOIN educ_raw_dedup AS r
              ON e.user_id = r.user_id
             AND e.education_number = r.education_number
            WHERE e.user_id IN ({userid_subset})
        )
        SELECT
            COALESCE(up.user_id, ej.user_id) AS user_id,
            up.user_location,
            up.user_country,
            up.female_prob,
            up.white_prob,
            up.black_prob,
            up.api_prob,
            up.hispanic_prob,
            up.native_prob,
            up.multiple_prob,
            up.updated_dt,
            ej.university_name,
            ej.rsid,
            ej.education_number,
            ej.ed_startdate,
            ej.ed_enddate,
            ej.degree,
            ej.field,
            ej.university_country,
            ej.university_location,
            ej.university_raw,
            ej.degree_raw,
            ej.field_raw,
            ej.description
        FROM user_profile AS up
        FULL JOIN educ_joined AS ej
          ON up.user_id = ej.user_id
    """


def _build_wrds_positions_extract_query(
    user_ids: list[int],
    *,
    position_cols: set[str],
) -> str:
    userid_subset = _format_user_id_sql_list(user_ids)
    return f"""
        SELECT
            CAST(a.user_id AS BIGINT) AS user_id,
            CAST(a.country AS VARCHAR) AS country
        FROM revelio.individual_positions AS a
        WHERE a.user_id IN ({userid_subset})
    """


def _build_wrds_selected_us_positions_query(
    rcids: list[int],
    *,
    history_year_min: int,
    history_year_max: int,
    include_seniority: bool,
    include_onet: bool,
) -> str:
    rcid_sql = ", ".join(str(int(v)) for v in rcids)
    onet_select = "CAST(onet_code AS VARCHAR) AS onet_code," if include_onet else "NULL::VARCHAR AS onet_code,"
    seniority_select = (
        "CAST(seniority AS VARCHAR) AS seniority_raw,"
        if include_seniority
        else "NULL::VARCHAR AS seniority_raw,"
    )
    return f"""
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            CAST(position_id AS BIGINT) AS position_id,
            CAST(position_number AS BIGINT) AS position_number,
            CAST(rcid AS BIGINT) AS rcid,
            CAST(country AS VARCHAR) AS country,
            CAST(startdate AS VARCHAR) AS startdate,
            CAST(enddate AS VARCHAR) AS enddate,
            {onet_select}
            {seniority_select}
            CAST(salary AS VARCHAR) AS salary,
            CAST(total_compensation AS VARCHAR) AS total_compensation
        FROM revelio.individual_positions
        WHERE {_sql_is_us_country("country")}
          AND rcid IN ({rcid_sql})
          AND startdate IS NOT NULL
          AND EXTRACT(YEAR FROM startdate)::INT <= {int(history_year_max)}
          AND EXTRACT(YEAR FROM COALESCE(enddate::DATE, DATE '2025-12-31'))::INT >= {int(history_year_min)}
    """


def _list_chunk_files(chunk_dir: Path) -> list[Path]:
    if not chunk_dir.exists():
        return []
    return sorted(path for path in chunk_dir.rglob("*.parquet") if path.is_file())


def _clear_chunk_dir(chunk_dir: Path) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for path in _list_chunk_files(chunk_dir):
        path.unlink()
    nested_dirs = sorted(
        (path for path in chunk_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in nested_dirs:
        if path == chunk_dir:
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def _write_empty_parquet(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_parquet(path, index=False)


def _count_rows_from_files(paths: list[Path]) -> int:
    if not paths:
        return 0
    con = ddb.connect()
    try:
        return int(con.execute("SELECT COUNT(*) FROM read_parquet(?)", [[str(path) for path in paths]]).fetchone()[0])
    finally:
        con.close()


def _count_parquet_rows(path: Path) -> int:
    con = ddb.connect()
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{_escape(path)}')").fetchone()[0])
    finally:
        con.close()


def _merge_chunk_dir_to_output(
    *,
    chunk_dir: Path,
    output_parquet: Path,
    empty_columns: list[str],
    overwrite: bool,
    distinct: bool = False,
    log_label: Optional[str] = None,
) -> None:
    chunk_files = _list_chunk_files(chunk_dir)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_parquet.exists():
        output_parquet.unlink()
    if not chunk_files:
        _write_empty_parquet(output_parquet, empty_columns)
        return
    merge_t0 = time.time()
    if log_label:
        _log(
            f"[{log_label}] START merge: {len(chunk_files):,} chunk files -> {output_parquet} | "
            f"distinct={int(bool(distinct))}"
        )
    con = ddb.connect()
    try:
        select_body = "SELECT DISTINCT *" if distinct else "SELECT *"
        con.execute(
            f"""
            COPY (
                {select_body}
                FROM read_parquet(?)
            )
            TO '{_escape(output_parquet)}' (FORMAT PARQUET)
            """,
            [[str(path) for path in chunk_files]],
        )
    finally:
        con.close()
    if log_label:
        _log(
            f"[{log_label}] DONE merge: {output_parquet} | "
            f"elapsed {_format_elapsed(time.time() - merge_t0)}"
        )


def build_wrds_workforce_user_ids(
    rcids: list[int],
    *,
    wrds_username: str,
    year_min: int,
    year_max: int,
    rcid_batch_size: int,
    rcid_n_users: Optional[pd.DataFrame | dict[int, float]] = None,
    large_firm_n_users_threshold: Optional[float] = None,
    large_firm_year_span: int = 1,
    query_timeout_ms: Optional[int],
    singleton_query_timeout_ms: Optional[int] = None,
    query_max_retries: int,
) -> pd.DataFrame:
    if not rcids:
        return pd.DataFrame(columns=["user_id"])
    if wrds is None:  # pragma: no cover
        raise ImportError("wrds is not installed.")

    build_t0 = time.time()
    relaxed_singleton_timeout_ms = _resolve_singleton_query_timeout_ms(
        query_timeout_ms=query_timeout_ms,
        singleton_query_timeout_ms=singleton_query_timeout_ms,
    )
    _log(
        f"[wrds_workforce_extract] START user-id scan: {len(rcids):,} firm ids | "
        f"years {int(year_min)}–{int(year_max)} | batch_size={int(rcid_batch_size)}"
    )
    db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
    batch_queue, task_meta = _build_wrds_task_queue(
        rcids,
        label_prefix="workforce-users",
        rcid_batch_size=int(rcid_batch_size),
        year_min=int(year_min),
        year_max=int(year_max),
        rcid_n_users=rcid_n_users,
        large_firm_n_users_threshold=large_firm_n_users_threshold,
        large_firm_year_span=int(large_firm_year_span),
    )
    if int(task_meta["n_large_firms"]) > 0:
        threshold = task_meta["large_firm_n_users_threshold"]
        span = int(task_meta["large_firm_year_span"])
        _log(
            "[wrds_workforce_extract] Proactive large-firm slicing for user-id scan: "
            f"{int(task_meta['n_large_firms']):,} firm(s) with n_users >= {int(threshold):,} "
            f"scheduled as {int(task_meta['n_large_firm_tasks']):,} singleton task(s) "
            f"using {span}-year windows"
        )

    frames: list[pd.DataFrame] = []
    try:
        while batch_queue:
            task = batch_queue.popleft()
            batch = list(task.rcids)
            label = task.label
            batch_t0 = time.time()
            _log(
                f"[wrds_workforce_extract] START {label}: {len(batch):,} firms | "
                f"history years {int(task.history_year_min)}–{int(task.history_year_max)}"
            )
            sql = _build_wrds_workforce_user_ids_query(
                batch,
                history_year_min=int(task.history_year_min),
                history_year_max=int(task.history_year_max),
            )
            try:
                df, db = _run_sql_with_retries(
                    db,
                    sql,
                    wrds_username=wrds_username,
                    query_timeout_ms=query_timeout_ms,
                    max_retries=query_max_retries,
                    label=label,
                )
            except Exception as exc:
                is_timeout = _wrds_exception_is_timeout(exc)
                if len(batch) > 1 and is_timeout:
                    mid = len(batch) // 2
                    left = batch[:mid]
                    right = batch[mid:]
                    _log(
                        f"[wrds_workforce_extract] SPLIT {label}: timeout after "
                        f"{_format_elapsed(time.time() - batch_t0)} | "
                        f"{len(batch):,} firms -> {len(left):,} + {len(right):,} firms"
                    )
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
                    batch_queue.appendleft(
                        WrdsQueryTask(
                            rcids=tuple(right),
                            label=f"{label}.2",
                            year_min=int(task.year_min),
                            year_max=int(task.year_max),
                            history_year_min=int(task.history_year_min),
                            history_year_max=int(task.history_year_max),
                        )
                    )
                    batch_queue.appendleft(
                        WrdsQueryTask(
                            rcids=tuple(left),
                            label=f"{label}.1",
                            year_min=int(task.year_min),
                            year_max=int(task.year_max),
                            history_year_min=int(task.history_year_min),
                            history_year_max=int(task.history_year_max),
                        )
                    )
                    continue
                if len(batch) == 1 and is_timeout and relaxed_singleton_timeout_ms is not None:
                    relaxed_label = f"{label} singleton-fallback"
                    slice_windows = _year_windows(int(task.year_min), int(task.year_max), 1)
                    _log(
                        f"[wrds_workforce_extract] RETRY {label}: timed out after "
                        f"{_format_elapsed(time.time() - batch_t0)} | retrying rcid {batch[0]:,} "
                        f"as 1-year windows with relaxed timeout "
                        f"({int(round(relaxed_singleton_timeout_ms / 60_000))} minutes)"
                    )
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = _open_wrds_connection(
                        wrds_username=wrds_username,
                        query_timeout_ms=relaxed_singleton_timeout_ms,
                    )
                    try:
                        year_frames: list[pd.DataFrame] = []
                        for slice_year_min, slice_year_max in slice_windows:
                            slice_label = f"{relaxed_label} y{slice_year_min}"
                            slice_sql = _build_wrds_workforce_user_ids_query(
                                batch,
                                history_year_min=int(task.history_year_min),
                                history_year_max=int(slice_year_max),
                            )
                            slice_df, db = _run_sql_with_retries(
                                db,
                                slice_sql,
                                wrds_username=wrds_username,
                                query_timeout_ms=relaxed_singleton_timeout_ms,
                                max_retries=query_max_retries,
                                label=slice_label,
                            )
                            year_frames.append(slice_df)
                        df = (
                            pd.concat(year_frames, ignore_index=True)
                            if year_frames
                            else pd.DataFrame(columns=["user_id"])
                        )
                    except Exception:
                        raise
                else:
                    raise
            if not df.empty:
                df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
                df = df.dropna(subset=["user_id"]).copy()
                df["user_id"] = df["user_id"].astype(int)
            frames.append(df)
            _log(
                f"[wrds_workforce_extract] DONE  {label}: {len(df):,} user ids | "
                f"elapsed {_format_elapsed(time.time() - batch_t0)}"
            )
    finally:
        try:
            db.close()
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["user_id"])
    out = pd.concat(frames, ignore_index=True)
    out["user_id"] = pd.to_numeric(out["user_id"], errors="coerce")
    out = out.dropna(subset=["user_id"]).copy()
    out["user_id"] = out["user_id"].astype(int)
    out = out.drop_duplicates(subset=["user_id"], keep="first").sort_values("user_id").reset_index(drop=True)
    _log(
        f"[wrds_workforce_extract] DONE user-id scan: {len(out):,} unique users | "
        f"elapsed {_format_elapsed(time.time() - build_t0)}"
    )
    return out


def _write_wrds_extract_chunks_for_user_ids(
    *,
    user_ids: list[int],
    wrds_username: str,
    users_chunk_dir: Path,
    positions_chunk_dir: Path,
    chunk_size: int,
    max_workers: int,
    overwrite: bool,
) -> dict[str, int]:
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if int(max_workers) <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers}.")
    users_chunk_dir.mkdir(parents=True, exist_ok=True)
    positions_chunk_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _clear_chunk_dir(users_chunk_dir)
        _clear_chunk_dir(positions_chunk_dir)

    if not user_ids:
        _write_empty_parquet(users_chunk_dir / "wrds_users_chunk_00000.parquet", WORKFORCE_WRDS_USERS_COLUMNS)
        _write_empty_parquet(
            positions_chunk_dir / "wrds_positions_chunk_00000.parquet",
            WORKFORCE_WRDS_POSITION_HISTORY_COLUMNS,
        )
        return {"wrds_query_chunks": 0, "wrds_user_rows": 0, "wrds_position_rows": 0}

    user_chunks = _rcid_chunks(sorted({int(user_id) for user_id in user_ids}), int(chunk_size))
    schema_db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=None)
    try:
        schema = _resolve_wrds_extract_schema(schema_db)
    finally:
        try:
            schema_db.close()
        except Exception:
            pass

    user_cols = schema["user_cols"]  # type: ignore[assignment]
    female_source_col = schema["female_source_col"]  # type: ignore[assignment]
    position_cols = schema["position_cols"]  # type: ignore[assignment]
    worker_count = min(len(user_chunks), max(1, int(max_workers)))
    chunk_jobs: list[list[tuple[int, list[int]]]] = [[] for _ in range(worker_count)]
    for job_idx, user_chunk in enumerate(user_chunks):
        chunk_jobs[job_idx % worker_count].append((job_idx, user_chunk))

    _log(
        f"[wrds_workforce_extract] START user-history extract: {len(user_ids):,} user ids | "
        f"{len(user_chunks):,} chunks | workers={worker_count}"
    )
    build_t0 = time.time()

    def _run_worker(worker_idx: int, assigned_jobs: list[tuple[int, list[int]]]) -> int:
        if not assigned_jobs:
            return 0
        db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=None)
        try:
            for idx, user_chunk in assigned_jobs:
                users_chunk_path = users_chunk_dir / f"wrds_users_chunk_{idx:05d}.parquet"
                positions_chunk_path = positions_chunk_dir / f"wrds_positions_chunk_{idx:05d}.parquet"
                if not users_chunk_path.exists() or overwrite:
                    users_df, db = _run_sql_with_retries(
                        db,
                        _build_wrds_users_extract_query(
                            user_chunk,
                            user_cols=user_cols,
                            female_source_col=female_source_col,
                        ),
                        wrds_username=wrds_username,
                        query_timeout_ms=None,
                        max_retries=1,
                        label=f"workforce users extract chunk {idx + 1}/{len(user_chunks)}",
                    )
                    users_df.to_parquet(users_chunk_path, index=False)
                if not positions_chunk_path.exists() or overwrite:
                    positions_df, db = _run_sql_with_retries(
                        db,
                        _build_wrds_positions_extract_query(
                            user_chunk,
                            position_cols=position_cols,
                        ),
                        wrds_username=wrds_username,
                        query_timeout_ms=None,
                        max_retries=1,
                        label=f"workforce positions extract chunk {idx + 1}/{len(user_chunks)}",
                    )
                    positions_df.to_parquet(positions_chunk_path, index=False)
                _log(
                    f"[wrds_workforce_extract] worker {worker_idx}/{worker_count} import chunk "
                    f"{idx + 1:05d}/{len(user_chunks):05d}: user_ids={len(user_chunk):,}"
                )
        finally:
            try:
                db.close()
            except Exception:
                pass
        return len(assigned_jobs)

    if worker_count == 1:
        _run_worker(1, chunk_jobs[0])
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="wrds_extract") as executor:
            futures = [
                executor.submit(_run_worker, worker_idx + 1, assigned_jobs)
                for worker_idx, assigned_jobs in enumerate(chunk_jobs)
                if assigned_jobs
            ]
            for future in as_completed(futures):
                future.result()

    _log(
        f"[wrds_workforce_extract] DONE user-history extract: {len(user_chunks):,} chunks | "
        f"elapsed {_format_elapsed(time.time() - build_t0)} | workers={worker_count}"
    )

    user_chunk_files = _list_chunk_files(users_chunk_dir)
    position_chunk_files = _list_chunk_files(positions_chunk_dir)
    return {
        "wrds_query_chunks": len(user_chunks),
        "wrds_user_rows": _count_rows_from_files(user_chunk_files),
        "wrds_position_rows": _count_rows_from_files(position_chunk_files),
    }


def _distinct_user_ids_from_positions_parquet(positions_path: Path) -> pd.DataFrame:
    if not positions_path.exists():
        return pd.DataFrame(columns=["user_id"])
    con = ddb.connect()
    try:
        out = con.sql(
            f"""
            SELECT DISTINCT
                CAST(user_id AS BIGINT) AS user_id
            FROM read_parquet('{_escape(positions_path)}')
            WHERE user_id IS NOT NULL
            ORDER BY 1
            """
        ).df()
    finally:
        con.close()
    if out.empty:
        return pd.DataFrame(columns=["user_id"])
    out["user_id"] = pd.to_numeric(out["user_id"], errors="coerce")
    out = out.dropna(subset=["user_id"]).copy()
    out["user_id"] = out["user_id"].astype(int)
    return out


def _extract_wrds_selected_us_positions(
    *,
    rcids: list[int],
    wrds_username: str,
    selected_positions_chunk_dir: Path,
    rcid_batch_size: int,
    rcid_n_users: Optional[pd.DataFrame | dict[int, float]],
    large_firm_n_users_threshold: Optional[float],
    large_firm_year_span: int,
    year_min: int,
    year_max: int,
    query_timeout_ms: Optional[int],
    singleton_query_timeout_ms: Optional[int],
    query_max_retries: int,
    overwrite: bool,
) -> dict[str, int | bool]:
    if not rcids:
        selected_positions_chunk_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            _clear_chunk_dir(selected_positions_chunk_dir)
        _write_empty_parquet(
            selected_positions_chunk_dir / "wrds_selected_us_positions_chunk_00000.parquet",
            WORKFORCE_SELECTED_US_POSITIONS_COLUMNS,
        )
        return {
            "wrds_selected_us_position_chunks": 1,
            "selected_positions_include_seniority": False,
            "selected_positions_include_onet": False,
        }
    if wrds is None:  # pragma: no cover
        raise ImportError("wrds is not installed.")
    selected_positions_chunk_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _clear_chunk_dir(selected_positions_chunk_dir)

    build_t0 = time.time()
    relaxed_singleton_timeout_ms = _resolve_singleton_query_timeout_ms(
        query_timeout_ms=query_timeout_ms,
        singleton_query_timeout_ms=singleton_query_timeout_ms,
    )
    _log(
        f"[wrds_workforce_extract] START selected-us-positions scan: {len(rcids):,} firm ids | "
        f"years {int(year_min)}–{int(year_max)} | batch_size={int(rcid_batch_size)}"
    )
    db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
    schema = _resolve_wrds_extract_schema(db)
    include_seniority = bool(schema["include_seniority"])
    include_onet = bool(schema["include_onet"])
    selected_positions_large_firm_year_span = _selected_positions_extract_large_firm_year_span(
        int(year_min),
        int(year_max),
    )
    batch_queue, task_meta = _build_wrds_task_queue(
        rcids,
        label_prefix="workforce-selected-us-positions",
        rcid_batch_size=int(rcid_batch_size),
        year_min=int(year_min),
        year_max=int(year_max),
        rcid_n_users=rcid_n_users,
        large_firm_n_users_threshold=large_firm_n_users_threshold,
        large_firm_year_span=selected_positions_large_firm_year_span,
    )
    if int(task_meta["n_large_firms"]) > 0:
        threshold = task_meta["large_firm_n_users_threshold"]
        span = int(task_meta["large_firm_year_span"])
        _log(
            "[wrds_workforce_extract] Proactive large-firm slicing for selected-us-positions: "
            f"{int(task_meta['n_large_firms']):,} firm(s) with n_users >= {int(threshold):,} "
            f"scheduled as {int(task_meta['n_large_firm_tasks']):,} singleton task(s) "
            f"using {span}-year windows"
        )

    chunk_idx = 0
    try:
        while batch_queue:
            task = batch_queue.popleft()
            batch = list(task.rcids)
            label = task.label
            batch_t0 = time.time()
            _log(
                f"[wrds_workforce_extract] START {label}: {len(batch):,} firms | "
                f"history years {int(task.history_year_min)}–{int(task.history_year_max)}"
            )
            sql = _build_wrds_selected_us_positions_query(
                batch,
                history_year_min=int(task.history_year_min),
                history_year_max=int(task.history_year_max),
                include_seniority=include_seniority,
                include_onet=include_onet,
            )
            try:
                df, db = _run_sql_with_retries(
                    db,
                    sql,
                    wrds_username=wrds_username,
                    query_timeout_ms=query_timeout_ms,
                    max_retries=query_max_retries,
                    label=label,
                )
            except Exception as exc:
                is_timeout = _wrds_exception_is_timeout(exc)
                if len(batch) > 1 and is_timeout:
                    mid = len(batch) // 2
                    left = batch[:mid]
                    right = batch[mid:]
                    _log(
                        f"[wrds_workforce_extract] SPLIT {label}: timeout after "
                        f"{_format_elapsed(time.time() - batch_t0)} | "
                        f"{len(batch):,} firms -> {len(left):,} + {len(right):,} firms"
                    )
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
                    batch_queue.appendleft(
                        WrdsQueryTask(
                            rcids=tuple(right),
                            label=f"{label}.2",
                            year_min=int(task.year_min),
                            year_max=int(task.year_max),
                            history_year_min=int(task.history_year_min),
                            history_year_max=int(task.history_year_max),
                        )
                    )
                    batch_queue.appendleft(
                        WrdsQueryTask(
                            rcids=tuple(left),
                            label=f"{label}.1",
                            year_min=int(task.year_min),
                            year_max=int(task.year_max),
                            history_year_min=int(task.history_year_min),
                            history_year_max=int(task.history_year_max),
                        )
                    )
                    continue
                if len(batch) == 1 and is_timeout and relaxed_singleton_timeout_ms is not None:
                    relaxed_label = f"{label} singleton-fallback"
                    slice_windows = _year_windows(int(task.year_min), int(task.year_max), 1)
                    _log(
                        f"[wrds_workforce_extract] RETRY {label}: timed out after "
                        f"{_format_elapsed(time.time() - batch_t0)} | retrying rcid {batch[0]:,} "
                        f"as 1-year windows with relaxed timeout "
                        f"({int(round(relaxed_singleton_timeout_ms / 60_000))} minutes)"
                    )
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = _open_wrds_connection(
                        wrds_username=wrds_username,
                        query_timeout_ms=relaxed_singleton_timeout_ms,
                    )
                    year_frames: list[pd.DataFrame] = []
                    for slice_year_min, slice_year_max in slice_windows:
                        slice_label = f"{relaxed_label} y{slice_year_min}"
                        slice_sql = _build_wrds_selected_us_positions_query(
                            batch,
                            history_year_min=int(slice_year_min),
                            history_year_max=int(slice_year_max),
                            include_seniority=include_seniority,
                            include_onet=include_onet,
                        )
                        slice_df, db = _run_sql_with_retries(
                            db,
                            slice_sql,
                            wrds_username=wrds_username,
                            query_timeout_ms=relaxed_singleton_timeout_ms,
                            max_retries=query_max_retries,
                            label=slice_label,
                        )
                        year_frames.append(slice_df)
                    df = (
                        pd.concat(year_frames, ignore_index=True)
                        if year_frames
                        else pd.DataFrame(columns=WORKFORCE_SELECTED_US_POSITIONS_COLUMNS)
                    )
                    if not df.empty:
                        df = df.drop_duplicates(ignore_index=True)
                else:
                    raise
            if not df.empty:
                chunk_path = selected_positions_chunk_dir / f"wrds_selected_us_positions_chunk_{chunk_idx:05d}.parquet"
                df.to_parquet(chunk_path, index=False)
                chunk_idx += 1
            _log(
                f"[wrds_workforce_extract] DONE  {label}: {len(df):,} selected-us-position rows | "
                f"elapsed {_format_elapsed(time.time() - batch_t0)}"
            )
    finally:
        try:
            db.close()
        except Exception:
            pass

    chunk_files = _list_chunk_files(selected_positions_chunk_dir)
    _log(
        f"[wrds_workforce_extract] DONE selected-us-positions scan: {len(chunk_files):,} chunks | "
        f"elapsed {_format_elapsed(time.time() - build_t0)}"
    )
    return {
        "wrds_selected_us_position_chunks": len(chunk_files),
        "selected_positions_include_seniority": include_seniority,
        "selected_positions_include_onet": include_onet,
    }


def load_or_build_wrds_workforce_local_extracts(
    *,
    cfg: dict,
    settings: dict[str, object],
    selected_firms: pd.DataFrame,
    rcid_n_users: Optional[pd.DataFrame | dict[int, float]],
    year_min: int,
    year_max: int,
    force_rebuild: bool = False,
) -> tuple[Path, Path, Path, dict]:
    feature_cfg = get_cfg_section(cfg, "revelio_company_features")
    selected = _clean_selected_firm_ids(selected_firms)
    user_ids_path = settings["extract_user_ids_path"]  # type: ignore[assignment]
    users_out = settings["extract_users_path"]  # type: ignore[assignment]
    positions_out = settings["extract_positions_path"]  # type: ignore[assignment]
    selected_positions_out = settings["extract_selected_positions_path"]  # type: ignore[assignment]
    users_chunk_dir = settings["extract_users_chunk_dir"]  # type: ignore[assignment]
    positions_chunk_dir = settings["extract_positions_chunk_dir"]  # type: ignore[assignment]
    selected_positions_chunk_dir = settings["extract_selected_positions_chunk_dir"]  # type: ignore[assignment]
    wrds_username = str(feature_cfg.get("wrds_username", "")).strip()
    extract_chunk_size = int(settings["extract_chunk_size"])
    extract_max_workers = int(settings["extract_max_workers"])
    query_timeout_ms = int(float(feature_cfg.get("query_timeout_minutes", 10)) * 60_000)
    singleton_query_timeout_ms = (
        int(float(feature_cfg.get("wrds_singleton_query_timeout_minutes")) * 60_000)
        if feature_cfg.get("wrds_singleton_query_timeout_minutes") is not None
        else None
    )
    query_max_retries = int(feature_cfg.get("query_max_retries", 1))

    expected_meta = {
        "year_min": int(year_min),
        "year_max": int(year_max),
        "selected_firms_hash": _selected_firms_hash(selected),
        "selected_firms_n": int(len(selected)),
        "wrds_username": wrds_username,
        "wrds_rcid_batch_size": int(feature_cfg.get("wrds_rcid_batch_size", 100)),
        "wrds_large_firm_n_users_threshold": feature_cfg.get("wrds_large_firm_n_users_threshold", 75_000),
        "wrds_large_firm_year_span": int(feature_cfg.get("wrds_large_firm_year_span", 1)),
        "wrds_workforce_extract_chunk_size": extract_chunk_size,
        "workforce_wrds_extract_method": WORKFORCE_WRDS_EXTRACT_METHOD,
        "new_hire_origin_method": NEW_HIRE_ORIGIN_METHOD,
    }
    reuse_cached_universe_only = _reuse_cached_wrds_universe_only(cfg)
    expected_meta_compare = _cache_meta_for_compatibility(
        expected_meta,
        ignore_keys=(
            _wrds_universe_ignore_keys_for_extracts()
            if reuse_cached_universe_only
            else None
        ),
    )
    cache_key = _wrds_workforce_local_extract_cache_key(
        user_ids_path,
        users_out,
        positions_out,
        selected_positions_out,
    )
    cached_result = _WRDS_WORKFORCE_LOCAL_EXTRACT_CACHE.get(cache_key)
    if cached_result is not None:
        cached_users_out, cached_positions_out, cached_selected_positions_out, cached_meta = (
            _clone_wrds_workforce_local_extract_result(cached_result)
        )
        if (
            user_ids_path.exists()
            and cached_users_out.exists()
            and cached_positions_out.exists()
            and cached_selected_positions_out.exists()
            and _metadata_compatible(
                cached_meta,
                {k: v for k, v in expected_meta_compare.items() if k not in {"year_min", "year_max"}},
            )
            and cached_meta.get("year_min") is not None
            and cached_meta.get("year_max") is not None
            and int(cached_meta["year_min"]) <= int(year_min)
            and int(cached_meta["year_max"]) >= int(year_max)
        ):
            _log(
                f"[wrds_workforce_extract] REUSE in-process: {cached_users_out}, "
                f"{cached_positions_out}, and {cached_selected_positions_out}"
            )
            return (
                cached_users_out,
                cached_positions_out,
                cached_selected_positions_out,
                cached_meta,
            )
    if users_out.exists() and positions_out.exists() and selected_positions_out.exists() and not force_rebuild:
        meta = _load_metadata(users_out)
        if (
            _metadata_compatible(
                meta,
                {k: v for k, v in expected_meta_compare.items() if k not in {"year_min", "year_max"}},
            )
            and meta.get("year_min") is not None
            and meta.get("year_max") is not None
            and int(meta["year_min"]) <= int(year_min)
            and int(meta["year_max"]) >= int(year_max)
        ):
            _WRDS_WORKFORCE_LOCAL_EXTRACT_CACHE[cache_key] = _clone_wrds_workforce_local_extract_result(
                (users_out, positions_out, selected_positions_out, meta)
            )
            _log(
                f"[wrds_workforce_extract] REUSE: {users_out}, {positions_out}, "
                f"and {selected_positions_out}"
            )
            if reuse_cached_universe_only and (
                meta.get("selected_firms_hash") != expected_meta.get("selected_firms_hash")
            ):
                _log(
                    "[wrds_workforce_extract] REUSE under reuse_cached_wrds_universe_only=true; "
                    "ignoring selected_firms_hash mismatch for testing."
                )
            return users_out, positions_out, selected_positions_out, meta

    selected_positions_stats = _extract_wrds_selected_us_positions(
        rcids=selected["c"].tolist(),
        wrds_username=wrds_username,
        selected_positions_chunk_dir=selected_positions_chunk_dir,
        rcid_batch_size=int(feature_cfg.get("wrds_rcid_batch_size", 100)),
        rcid_n_users=rcid_n_users,
        large_firm_n_users_threshold=feature_cfg.get("wrds_large_firm_n_users_threshold", 75_000),
        large_firm_year_span=int(feature_cfg.get("wrds_large_firm_year_span", 1)),
        year_min=int(year_min),
        year_max=int(year_max),
        query_timeout_ms=query_timeout_ms,
        singleton_query_timeout_ms=singleton_query_timeout_ms,
        query_max_retries=query_max_retries,
        overwrite=True,
    )
    _merge_chunk_dir_to_output(
        chunk_dir=selected_positions_chunk_dir,
        output_parquet=selected_positions_out,
        empty_columns=WORKFORCE_SELECTED_US_POSITIONS_COLUMNS,
        overwrite=True,
        distinct=False,
        log_label="wrds_workforce_extract",
    )

    user_ids = _distinct_user_ids_from_positions_parquet(selected_positions_out)
    user_ids_path.parent.mkdir(parents=True, exist_ok=True)
    user_ids.to_parquet(user_ids_path, index=False)

    chunk_stats = _write_wrds_extract_chunks_for_user_ids(
        user_ids=user_ids["user_id"].astype(int).tolist(),
        wrds_username=wrds_username,
        users_chunk_dir=users_chunk_dir,
        positions_chunk_dir=positions_chunk_dir,
        chunk_size=extract_chunk_size,
        max_workers=extract_max_workers,
        overwrite=True,
    )
    _merge_chunk_dir_to_output(
        chunk_dir=users_chunk_dir,
        output_parquet=users_out,
        empty_columns=WORKFORCE_WRDS_USERS_COLUMNS,
        overwrite=True,
        log_label="wrds_workforce_extract",
    )
    _merge_chunk_dir_to_output(
        chunk_dir=positions_chunk_dir,
        output_parquet=positions_out,
        empty_columns=WORKFORCE_WRDS_POSITION_HISTORY_COLUMNS,
        overwrite=True,
        log_label="wrds_workforce_extract",
    )
    meta = expected_meta | {
        "n_selected_user_ids": int(len(user_ids)),
        "wrds_user_rows": _count_parquet_rows(users_out),
        "wrds_position_rows": _count_parquet_rows(positions_out),
        "wrds_selected_us_position_rows": _count_parquet_rows(selected_positions_out),
        "wrds_query_chunks": int(chunk_stats["wrds_query_chunks"]),
        "wrds_workforce_extract_max_workers": extract_max_workers,
        "user_ids_path": str(user_ids_path),
        "users_chunk_dir": str(users_chunk_dir),
        "positions_chunk_dir": str(positions_chunk_dir),
        "selected_positions_chunk_dir": str(selected_positions_chunk_dir),
        "selected_positions_path": str(selected_positions_out),
        **selected_positions_stats,
    }
    _write_metadata(user_ids_path, meta)
    _write_metadata(users_out, meta)
    _write_metadata(positions_out, meta)
    _write_metadata(selected_positions_out, meta)
    _WRDS_WORKFORCE_LOCAL_EXTRACT_CACHE[cache_key] = _clone_wrds_workforce_local_extract_result(
        (users_out, positions_out, selected_positions_out, meta)
    )
    _log(
        f"[wrds_workforce_extract] DONE: wrote {users_out}, {positions_out}, and "
        f"{selected_positions_out} | "
        f"user_ids={int(len(user_ids)):,} | user_rows={int(meta['wrds_user_rows']):,} | "
        f"position_rows={int(meta['wrds_position_rows']):,} | "
        f"selected_us_position_rows={int(meta['wrds_selected_us_position_rows']):,}"
    )
    return users_out, positions_out, selected_positions_out, meta


def _clean_selected_firm_ids(selected_firms: pd.DataFrame) -> pd.DataFrame:
    cleaned = pd.DataFrame({"c": pd.to_numeric(selected_firms.get("c"), errors="coerce")})
    cleaned = cleaned.dropna(subset=["c"]).copy()
    cleaned["c"] = cleaned["c"].astype(int)
    return cleaned.drop_duplicates(subset=["c"], keep="first").reset_index(drop=True)


def _build_local_wrds_user_profile_cache(
    *,
    selected_firms: pd.DataFrame,
    users_path: Path,
    positions_path: Path,
    out_path: Path,
    year_min: int,
    year_max: int,
) -> None:
    selected = _clean_selected_firm_ids(selected_firms)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    position_cols = _local_parquet_columns(positions_path)
    if {"rcid", "startdate", "enddate"}.issubset(position_cols):
        company_users_sql = f"""
        selected_us_positions AS (
            SELECT
                CAST(p.user_id AS BIGINT) AS user_id
            FROM read_parquet('{_escape(positions_path)}') AS p
            JOIN selected_firms sf
              ON CAST(p.rcid AS BIGINT) = sf.c
            WHERE p.user_id IS NOT NULL
              AND {_sql_is_us_country("p.country")}
              AND {_date_parse_sql("p.startdate")} IS NOT NULL
              AND EXTRACT(YEAR FROM {_date_parse_sql("p.startdate")})::INT <= {int(year_max)}
              AND EXTRACT(
                    YEAR FROM COALESCE({_date_parse_sql("p.enddate")}, DATE '2025-12-31')
                  )::INT >= {int(year_min)}
        ),
        company_users AS (
            SELECT DISTINCT user_id
            FROM selected_us_positions
        ),"""
    else:
        company_users_sql = f"""
        company_users AS (
            SELECT DISTINCT
                CAST(user_id AS BIGINT) AS user_id
            FROM read_parquet('{_escape(positions_path)}')
            WHERE user_id IS NOT NULL
        ),"""
    con = ddb.connect()
    try:
        con.register("selected_firms_df", selected)
        con.sql(
            "CREATE OR REPLACE TEMP VIEW selected_firms AS "
            "SELECT CAST(c AS BIGINT) AS c FROM selected_firms_df"
        )
        sql = f"""
        WITH {company_users_sql}
        user_profile_raw AS (
            SELECT
                CAST(u.user_id AS BIGINT) AS user_id,
                CAST(u.user_country AS VARCHAR) AS user_country_raw,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(u.user_id AS BIGINT)
                    ORDER BY NULLIF(TRIM(CAST(u.updated_dt AS VARCHAR)), '') DESC NULLS LAST,
                             CAST(u.user_country AS VARCHAR) DESC NULLS LAST,
                             CAST(u.user_id AS BIGINT)
                ) AS rn
            FROM read_parquet('{_escape(users_path)}') AS u
            JOIN company_users cu
              ON CAST(u.user_id AS BIGINT) = cu.user_id
        ),
        user_profile AS (
            SELECT
                user_id,
                {_sql_nonus_country_signal_expr("user_country_raw")} AS signal_current_country_nonus
            FROM user_profile_raw
            WHERE rn = 1
        ),
        educ_joined AS (
            SELECT
                CAST(u.user_id AS BIGINT) AS user_id,
                CAST(u.education_number AS BIGINT) AS education_number,
                CAST(u.ed_startdate AS VARCHAR) AS ed_startdate,
                CAST(u.ed_enddate AS VARCHAR) AS ed_enddate,
                CAST(u.degree AS VARCHAR) AS degree,
                CAST(COALESCE(u.degree_raw, '') AS VARCHAR) AS degree_raw,
                CAST(COALESCE(u.field_raw, '') AS VARCHAR) AS field_raw,
                CAST(COALESCE(u.university_raw, '') AS VARCHAR) AS university_raw,
                CAST(u.university_country AS VARCHAR) AS university_country
            FROM read_parquet('{_escape(users_path)}') AS u
            JOIN company_users cu
              ON CAST(u.user_id AS BIGINT) = cu.user_id
            WHERE u.education_number IS NOT NULL
        ),
        raw_country_counts AS (
            SELECT
                LOWER(TRIM(university_raw)) AS raw_key,
                university_country,
                COUNT(*) AS n,
                ROW_NUMBER() OVER (
                    PARTITION BY LOWER(TRIM(university_raw))
                    ORDER BY COUNT(*) DESC, university_country ASC
                ) AS rn
            FROM educ_joined
            WHERE university_country IS NOT NULL
              AND TRIM(university_country) <> ''
              AND university_raw IS NOT NULL
              AND TRIM(university_raw) <> ''
            GROUP BY 1, 2
        ),
        raw_country_mode AS (
            SELECT raw_key, university_country AS modal_university_country
            FROM raw_country_counts
            WHERE rn = 1
        ),
        educ_enriched AS (
            SELECT
                ej.*,
                {degree_clean_regex_sql()} AS degree_clean,
                COALESCE(
                    NULLIF(TRIM(ej.university_country), ''),
                    rcm.modal_university_country
                ) AS university_country_filled
            FROM educ_joined ej
            LEFT JOIN raw_country_mode rcm
              ON LOWER(TRIM(ej.university_raw)) = rcm.raw_key
        ),
        user_educ AS (
            SELECT
                user_id,
                MAX(
                    CASE
                        WHEN university_country_filled IS NOT NULL
                         AND TRIM(university_country_filled) <> ''
                         AND NOT {_sql_is_us_country("university_country_filled")}
                        THEN 1
                        ELSE 0
                    END
                ) AS has_nonus_educ,
                CASE
                    WHEN MAX(CASE WHEN degree_clean = 'High School' THEN 1 ELSE 0 END) = 1
                        THEN MAX(
                            CASE
                                WHEN degree_clean = 'High School' AND ed_enddate IS NOT NULL
                                    THEN SUBSTRING(ed_enddate, 1, 4)::INT - 18
                                ELSE NULL
                            END
                        )
                    ELSE MIN(
                        CASE
                            WHEN degree_clean = 'Non-Degree' OR degree_clean = 'Master'
                              OR degree_clean = 'Doctor' OR degree_clean = 'MBA'
                                THEN NULL
                            WHEN ed_startdate IS NOT NULL
                                THEN SUBSTRING(ed_startdate, 1, 4)::INT - 18
                            WHEN ed_enddate IS NOT NULL AND NOT degree_clean = 'Associate'
                                THEN SUBSTRING(ed_enddate, 1, 4)::INT - 23
                            WHEN ed_enddate IS NOT NULL
                                THEN SUBSTRING(ed_enddate, 1, 4)::INT - 21
                            ELSE NULL
                        END
                    )
                END AS est_yob
            FROM educ_enriched
            GROUP BY 1
        ),
        user_position_history AS (
            SELECT
                CAST(p.user_id AS BIGINT) AS user_id,
                MAX(
                    CASE
                        WHEN p.country IS NOT NULL AND TRIM(CAST(p.country AS VARCHAR)) <> '' THEN 1
                        ELSE 0
                    END
                ) AS has_any_position_country,
                MAX(
                    CASE
                        WHEN p.country IS NOT NULL
                         AND TRIM(CAST(p.country AS VARCHAR)) <> ''
                         AND NOT {_sql_is_us_country("p.country")}
                        THEN 1
                        ELSE 0
                    END
                ) AS has_nonus_position
            FROM read_parquet('{_escape(positions_path)}') AS p
            JOIN company_users cu
              ON CAST(p.user_id AS BIGINT) = cu.user_id
            GROUP BY 1
        ),
        user_origin_signals AS (
            SELECT
                cu.user_id,
                COALESCE(ue.has_nonus_educ, 0)::DOUBLE PRECISION AS has_nonus_educ,
                ue.est_yob,
                up.signal_current_country_nonus,
                CASE
                    WHEN ue.user_id IS NULL THEN NULL::DOUBLE PRECISION
                    ELSE COALESCE(ue.has_nonus_educ, 0)::DOUBLE PRECISION
                END AS signal_nonus_educ,
                CASE
                    WHEN COALESCE(uph.has_any_position_country, 0) = 0 THEN NULL::DOUBLE PRECISION
                    ELSE COALESCE(uph.has_nonus_position, 0)::DOUBLE PRECISION
                END AS signal_nonus_position
            FROM company_users cu
            LEFT JOIN user_profile up
              ON up.user_id = cu.user_id
            LEFT JOIN user_educ ue
              ON ue.user_id = cu.user_id
            LEFT JOIN user_position_history uph
              ON uph.user_id = cu.user_id
        ),
        user_origin_scored AS (
            SELECT
                user_id,
                has_nonus_educ,
                est_yob,
                signal_current_country_nonus,
                signal_nonus_educ,
                signal_nonus_position,
                {_sql_new_hire_origin_probability_expr([
                    "signal_current_country_nonus",
                    "signal_nonus_educ",
                    "signal_nonus_position",
                ])} AS p_likely_foreign
            FROM user_origin_signals
        )
        SELECT
            user_id,
            has_nonus_educ,
            est_yob,
            signal_current_country_nonus,
            signal_nonus_educ,
            signal_nonus_position,
            p_likely_foreign,
            (1.0 - p_likely_foreign)::DOUBLE PRECISION AS p_likely_native,
            {_sql_new_hire_origin_hard_expr("p_likely_foreign", "signal_current_country_nonus")}
                AS likely_foreign_hard
        FROM user_origin_scored
        """
        con.sql(f"COPY ({sql}) TO '{_escape(out_path)}' (FORMAT PARQUET)")
    finally:
        con.close()


def load_or_build_local_wrds_user_profile_cache(
    *,
    selected_firms: pd.DataFrame,
    users_path: Path,
    positions_path: Path,
    out_path: Path,
    year_min: int,
    year_max: int,
    force_rebuild: bool = False,
) -> tuple[Path, dict]:
    selected = _clean_selected_firm_ids(selected_firms)
    expected_meta = {
        "year_min": int(year_min),
        "year_max": int(year_max),
        "users_path": str(users_path),
        "positions_path": str(positions_path),
        "selected_firms_hash": _selected_firms_hash(selected),
        "selected_firms_n": int(len(selected)),
        "local_user_profile_cache_method": LOCAL_USER_PROFILE_CACHE_METHOD,
        "new_hire_origin_method": NEW_HIRE_ORIGIN_METHOD,
    }
    cache_key = _local_user_profile_cache_key(out_path)
    cached_result = _LOCAL_USER_PROFILE_CACHE.get(cache_key)
    if cached_result is not None:
        cached_path, cached_meta = _clone_local_user_profile_cache_result(cached_result)
        if (
            cached_path.exists()
            and _metadata_compatible(
                cached_meta,
                {k: v for k, v in expected_meta.items() if k not in {"year_min", "year_max"}},
            )
            and cached_meta.get("year_min") is not None
            and cached_meta.get("year_max") is not None
            and int(cached_meta["year_min"]) <= int(year_min)
            and int(cached_meta["year_max"]) >= int(year_max)
        ):
            _log(f"[local_user_profile_cache] REUSE in-process: {cached_path}")
            return cached_path, cached_meta
    if out_path.exists() and not force_rebuild:
        meta = _load_metadata(out_path)
        if (
            _metadata_compatible(
                meta,
                {k: v for k, v in expected_meta.items() if k not in {"year_min", "year_max"}},
            )
            and meta.get("year_min") is not None
            and meta.get("year_max") is not None
            and int(meta["year_min"]) <= int(year_min)
            and int(meta["year_max"]) >= int(year_max)
        ):
            _LOCAL_USER_PROFILE_CACHE[cache_key] = _clone_local_user_profile_cache_result((out_path, meta))
            _log(f"[local_user_profile_cache] REUSE: {out_path}")
            return out_path, meta

    _log(
        f"[local_user_profile_cache] BUILD: {out_path} | {len(selected):,} firms | "
        f"years {int(year_min)}–{int(year_max)}"
    )
    _build_local_wrds_user_profile_cache(
        selected_firms=selected,
        users_path=users_path,
        positions_path=positions_path,
        out_path=out_path,
        year_min=year_min,
        year_max=year_max,
    )
    con = ddb.connect()
    try:
        n_rows = int(
            con.sql(f"SELECT COUNT(*) AS n_rows FROM read_parquet('{_escape(out_path)}')").fetchone()[0]
        )
    finally:
        con.close()
    meta = expected_meta | {"n_rows": n_rows}
    _write_metadata(out_path, meta)
    _LOCAL_USER_PROFILE_CACHE[cache_key] = _clone_local_user_profile_cache_result((out_path, meta))
    _log(f"[local_user_profile_cache] DONE: wrote {out_path} | rows {n_rows:,}")
    return out_path, meta


def build_local_company_year_profile_metrics(
    *,
    selected_firms: pd.DataFrame,
    positions_path: Path,
    user_profile_cache_path: Path,
    year_min: int,
    year_max: int,
    include_education_features: bool = True,
) -> pd.DataFrame:
    selected = _clean_selected_firm_ids(selected_firms)
    if selected.empty:
        return pd.DataFrame(
            columns=["c", "t"] + LOCAL_PROFILE_VALIDATION_COLUMNS + LOCAL_PROFILE_WORKFORCE_COLUMNS
        )

    if include_education_features:
        education_metric_sql = """
        AVG(CASE WHEN ay.has_nonus_educ = 1 THEN 1 ELSE 0 END) AS nonus_educ_share_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) < 30 THEN 1 ELSE 0 END) AS age_share_lt30_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) BETWEEN 30 AND 39 THEN 1 ELSE 0 END) AS age_share_30_39_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) BETWEEN 40 AND 49 THEN 1 ELSE 0 END) AS age_share_40_49_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) BETWEEN 50 AND 59 THEN 1 ELSE 0 END) AS age_share_50_59_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) >= 60 THEN 1 ELSE 0 END) AS age_share_60p_annual,"""
    else:
        education_metric_sql = """
        NULL::DOUBLE PRECISION AS nonus_educ_share_annual,
        NULL::DOUBLE PRECISION AS age_share_lt30_annual,
        NULL::DOUBLE PRECISION AS age_share_30_39_annual,
        NULL::DOUBLE PRECISION AS age_share_40_49_annual,
        NULL::DOUBLE PRECISION AS age_share_50_59_annual,
        NULL::DOUBLE PRECISION AS age_share_60p_annual,"""

    con = ddb.connect()
    try:
        con.register("selected_firms_df", selected)
        con.sql(
            "CREATE OR REPLACE TEMP VIEW selected_firms AS "
            "SELECT CAST(c AS BIGINT) AS c FROM selected_firms_df"
        )
        metrics = con.sql(
            f"""
            WITH us_positions AS (
                SELECT
                    CAST(p.user_id AS BIGINT) AS user_id,
                    CAST(p.rcid AS BIGINT) AS rcid,
                    {_date_parse_sql("p.startdate")} AS startdate,
                    COALESCE({_date_parse_sql("p.enddate")}, DATE '2025-12-31') AS enddate
                FROM read_parquet('{_escape(positions_path)}') AS p
                JOIN selected_firms sf
                  ON CAST(p.rcid AS BIGINT) = sf.c
                WHERE p.user_id IS NOT NULL
                  AND {_sql_is_us_country("p.country")}
                  AND {_date_parse_sql("p.startdate")} IS NOT NULL
                  AND EXTRACT(YEAR FROM {_date_parse_sql("p.startdate")})::INT <= {int(year_max)}
                  AND EXTRACT(
                        YEAR FROM COALESCE({_date_parse_sql("p.enddate")}, DATE '2025-12-31')
                      )::INT >= {int(year_min)}
            ),
            company_user_first_start AS (
                SELECT
                    user_id,
                    rcid,
                    MIN(startdate) AS first_start
                FROM us_positions
                GROUP BY 1, 2
            ),
            new_hire_origin AS (
                SELECT
                    cufs.user_id,
                    cufs.rcid,
                    EXTRACT(YEAR FROM cufs.first_start)::INT AS year,
                    up.p_likely_foreign,
                    up.p_likely_native,
                    up.likely_foreign_hard
                FROM company_user_first_start cufs
                LEFT JOIN read_parquet('{_escape(user_profile_cache_path)}') AS up
                  ON cufs.user_id = CAST(up.user_id AS BIGINT)
                WHERE EXTRACT(YEAR FROM cufs.first_start)::INT BETWEEN {int(year_min)} AND {int(year_max)}
            ),
            new_hires AS (
                SELECT
                    rcid,
                    year,
                    COUNT(DISTINCT user_id) AS n_new_hires_wrds_annual,
                    SUM(COALESCE(p_likely_foreign, 0.0)) AS n_new_hires_foreign_weighted_annual,
                    SUM(COALESCE(p_likely_native, 1.0)) AS n_new_hires_native_weighted_annual,
                    SUM(COALESCE(likely_foreign_hard, 0)) AS n_new_hires_foreign_hard_annual,
                    SUM(1 - COALESCE(likely_foreign_hard, 0)) AS n_new_hires_native_hard_annual
                FROM new_hire_origin
                GROUP BY 1, 2
            ),
            active_user_year AS (
                SELECT
                    p.rcid,
                    gs.year::INT AS year,
                    p.user_id,
                    MAX(COALESCE(up.has_nonus_educ, 0.0)) AS has_nonus_educ,
                    MAX(up.est_yob) AS est_yob,
                    MAX(COALESCE(up.p_likely_foreign, 0.0)) AS p_likely_foreign,
                    MAX(COALESCE(up.p_likely_native, 1.0)) AS p_likely_native,
                    MAX(COALESCE(up.likely_foreign_hard, 0)) AS likely_foreign_hard
                FROM us_positions p
                JOIN company_user_first_start cufs
                  ON cufs.user_id = p.user_id
                 AND cufs.rcid = p.rcid
                JOIN LATERAL generate_series(
                    GREATEST(EXTRACT(YEAR FROM p.startdate)::INT, {int(year_min)}),
                    LEAST(EXTRACT(YEAR FROM p.enddate)::INT, {int(year_max)})
                ) AS gs(year) ON TRUE
                LEFT JOIN read_parquet('{_escape(user_profile_cache_path)}') AS up
                  ON p.user_id = CAST(up.user_id AS BIGINT)
                GROUP BY 1, 2, 3
            )
            SELECT
                CAST(ay.rcid AS BIGINT) AS c,
                CAST(ay.year AS INTEGER) AS t,
                COUNT(DISTINCT ay.user_id)::DOUBLE PRECISION AS local_total_headcount_wrds_annual,
                {education_metric_sql}
                SUM(ay.p_likely_foreign)::DOUBLE PRECISION AS total_headcount_foreign_weighted_annual,
                SUM(ay.p_likely_native)::DOUBLE PRECISION AS total_headcount_native_weighted_annual,
                SUM(ay.likely_foreign_hard)::DOUBLE PRECISION AS total_headcount_foreign_hard_annual,
                SUM((1 - ay.likely_foreign_hard))::DOUBLE PRECISION AS total_headcount_native_hard_annual,
                COALESCE(MAX(nh.n_new_hires_wrds_annual), 0)::DOUBLE PRECISION AS local_n_new_hires_wrds_annual,
                COALESCE(MAX(nh.n_new_hires_foreign_weighted_annual), 0.0)::DOUBLE PRECISION AS n_new_hires_foreign_weighted_annual,
                COALESCE(MAX(nh.n_new_hires_native_weighted_annual), 0.0)::DOUBLE PRECISION AS n_new_hires_native_weighted_annual,
                COALESCE(MAX(nh.n_new_hires_foreign_hard_annual), 0.0)::DOUBLE PRECISION AS n_new_hires_foreign_hard_annual,
                COALESCE(MAX(nh.n_new_hires_native_hard_annual), 0.0)::DOUBLE PRECISION AS n_new_hires_native_hard_annual
            FROM active_user_year ay
            LEFT JOIN new_hires nh
              ON nh.rcid = ay.rcid
             AND nh.year = ay.year
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).df()
    finally:
        con.close()

    if not metrics.empty:
        metrics["c"] = pd.to_numeric(metrics["c"], errors="coerce").astype(int)
        metrics["t"] = pd.to_numeric(metrics["t"], errors="coerce").astype(int)
    return metrics


def build_local_wrds_company_year_workforce(
    *,
    selected_firms: pd.DataFrame,
    users_path: Path,
    selected_positions_path: Path,
    user_profile_cache_path: Path,
    year_min: int,
    year_max: int,
    include_education_features: bool = True,
) -> pd.DataFrame:
    selected = _clean_selected_firm_ids(selected_firms)
    if selected.empty:
        return pd.DataFrame(columns=["c", "t"] + ORIGIN_SPLIT_WORKFORCE_COLUMNS)

    user_cols = _local_parquet_columns(users_path)
    female_sql = _wrds_sql_column_or_null("u", "female_prob", "DOUBLE PRECISION", user_cols)
    race_sql = ",\n                    ".join(
        _wrds_sql_column_or_null("u", col, "DOUBLE PRECISION", user_cols)
        for col in _RACE_PROB_COLS
    )

    if include_education_features:
        education_metric_sql = """
        AVG(CASE WHEN ay.has_nonus_educ = 1 THEN 1 ELSE 0 END) AS nonus_educ_share_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) < 30 THEN 1 ELSE 0 END) AS age_share_lt30_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) BETWEEN 30 AND 39 THEN 1 ELSE 0 END) AS age_share_30_39_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) BETWEEN 40 AND 49 THEN 1 ELSE 0 END) AS age_share_40_49_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) BETWEEN 50 AND 59 THEN 1 ELSE 0 END) AS age_share_50_59_annual,
        AVG(CASE WHEN ay.est_yob IS NULL THEN NULL WHEN (ay.year - ay.est_yob) >= 60 THEN 1 ELSE 0 END) AS age_share_60p_annual,"""
    else:
        education_metric_sql = """
        NULL::DOUBLE PRECISION AS nonus_educ_share_annual,
        NULL::DOUBLE PRECISION AS age_share_lt30_annual,
        NULL::DOUBLE PRECISION AS age_share_30_39_annual,
        NULL::DOUBLE PRECISION AS age_share_40_49_annual,
        NULL::DOUBLE PRECISION AS age_share_50_59_annual,
        NULL::DOUBLE PRECISION AS age_share_60p_annual,"""

    con = ddb.connect()
    try:
        con.register("selected_firms_df", selected)
        con.sql(
            "CREATE OR REPLACE TEMP VIEW selected_firms AS "
            "SELECT CAST(c AS BIGINT) AS c FROM selected_firms_df"
        )
        workforce = con.sql(
            f"""
            WITH us_positions AS (
                SELECT
                    CAST(p.user_id AS BIGINT) AS user_id,
                    CAST(p.position_id AS BIGINT) AS position_id,
                    CAST(p.position_number AS BIGINT) AS position_number,
                    CAST(p.rcid AS BIGINT) AS rcid,
                    {_date_parse_sql("p.startdate")} AS startdate,
                    COALESCE({_date_parse_sql("p.enddate")}, DATE '2025-12-31') AS enddate,
                    CAST(p.salary AS VARCHAR) AS salary_raw,
                    CAST(p.total_compensation AS VARCHAR) AS total_comp_raw,
                    CAST(p.seniority_raw AS VARCHAR) AS seniority_raw,
                    CAST(p.onet_code AS VARCHAR) AS onet_code
                FROM read_parquet('{_escape(selected_positions_path)}') AS p
                JOIN selected_firms sf
                  ON CAST(p.rcid AS BIGINT) = sf.c
                WHERE p.user_id IS NOT NULL
                  AND {_date_parse_sql("p.startdate")} IS NOT NULL
                  AND EXTRACT(YEAR FROM {_date_parse_sql("p.startdate")})::INT <= {int(year_max)}
                  AND EXTRACT(
                        YEAR FROM COALESCE({_date_parse_sql("p.enddate")}, DATE '2025-12-31')
                      )::INT >= {int(year_min)}
            ),
            user_demo_raw AS (
                SELECT
                    CAST(u.user_id AS BIGINT) AS user_id,
                    {female_sql},
                    {race_sql},
                    CAST(u.updated_dt AS VARCHAR) AS updated_dt_raw,
                    ROW_NUMBER() OVER (
                        PARTITION BY CAST(u.user_id AS BIGINT)
                        ORDER BY NULLIF(TRIM(CAST(u.updated_dt AS VARCHAR)), '') DESC NULLS LAST,
                                 CAST(u.user_id AS BIGINT)
                    ) AS rn
                FROM read_parquet('{_escape(users_path)}') AS u
                WHERE u.user_id IS NOT NULL
            ),
            user_demo AS (
                SELECT
                    user_id,
                    female_prob,
                    white_prob,
                    black_prob,
                    api_prob,
                    hispanic_prob,
                    native_prob,
                    multiple_prob
                FROM user_demo_raw
                WHERE rn = 1
            ),
            company_user_first_start AS (
                SELECT
                    user_id,
                    rcid,
                    MIN(startdate) AS first_start
                FROM us_positions
                GROUP BY 1, 2
            ),
            new_hire_origin AS (
                SELECT
                    cufs.user_id,
                    cufs.rcid,
                    EXTRACT(YEAR FROM cufs.first_start)::INT AS year,
                    up.p_likely_foreign,
                    up.p_likely_native,
                    up.likely_foreign_hard
                FROM company_user_first_start cufs
                LEFT JOIN read_parquet('{_escape(user_profile_cache_path)}') AS up
                  ON cufs.user_id = CAST(up.user_id AS BIGINT)
                WHERE EXTRACT(YEAR FROM cufs.first_start)::INT BETWEEN {int(year_min)} AND {int(year_max)}
            ),
            new_hires AS (
                SELECT
                    rcid,
                    year,
                    COUNT(DISTINCT user_id) AS n_new_hires_wrds_annual,
                    SUM(COALESCE(p_likely_foreign, 0.0)) AS n_new_hires_foreign_weighted_annual,
                    SUM(COALESCE(p_likely_native, 1.0)) AS n_new_hires_native_weighted_annual,
                    SUM(COALESCE(likely_foreign_hard, 0)) AS n_new_hires_foreign_hard_annual,
                    SUM(1 - COALESCE(likely_foreign_hard, 0)) AS n_new_hires_native_hard_annual
                FROM new_hire_origin
                GROUP BY 1, 2
            ),
            active_user_year AS (
                SELECT
                    p.rcid,
                    gs.year::INT AS year,
                    p.user_id,
                    MAX(TRY_CAST(NULLIF(TRIM(CAST(p.salary_raw AS VARCHAR)), '') AS DOUBLE PRECISION)) AS salary,
                    MAX(TRY_CAST(NULLIF(TRIM(CAST(p.total_comp_raw AS VARCHAR)), '') AS DOUBLE PRECISION)) AS total_compensation,
                    MAX(cufs.first_start) AS first_start,
                    MAX(COALESCE(up.has_nonus_educ, 0.0)) AS has_nonus_educ,
                    MAX(up.est_yob) AS est_yob,
                    MAX(COALESCE(up.p_likely_foreign, 0.0)) AS p_likely_foreign,
                    MAX(COALESCE(up.p_likely_native, 1.0)) AS p_likely_native,
                    MAX(COALESCE(up.likely_foreign_hard, 0)) AS likely_foreign_hard,
                    MAX(
                        CASE
                            WHEN p.onet_code IS NULL OR TRIM(CAST(p.onet_code AS VARCHAR)) = '' THEN NULL
                            ELSE SUBSTRING(CAST(p.onet_code AS VARCHAR) FROM 1 FOR 2)
                        END
                    ) AS soc2,
                    MAX(
                        CASE
                            WHEN p.seniority_raw IS NULL OR TRIM(CAST(p.seniority_raw AS VARCHAR)) = '' THEN NULL
                            WHEN regexp_matches(LOWER(CAST(p.seniority_raw AS VARCHAR)), '(intern|entry|junior|jr)') THEN 1.0
                            WHEN regexp_matches(LOWER(CAST(p.seniority_raw AS VARCHAR)), '(associate|mid|ic)') THEN 2.0
                            WHEN regexp_matches(LOWER(CAST(p.seniority_raw AS VARCHAR)), '(senior|sr|lead|principal|staff)') THEN 3.0
                            WHEN regexp_matches(LOWER(CAST(p.seniority_raw AS VARCHAR)), '(manager|director|head|vp|vice president|chief|president|founder|partner|owner|executive)') THEN 4.0
                            WHEN NULLIF(regexp_replace(TRIM(CAST(p.seniority_raw AS VARCHAR)), '[^0-9\\.-]', '', 'g'), '') IS NOT NULL
                                THEN TRY_CAST(NULLIF(regexp_replace(TRIM(CAST(p.seniority_raw AS VARCHAR)), '[^0-9\\.-]', '', 'g'), '') AS DOUBLE PRECISION)
                            ELSE NULL
                        END
                    ) AS seniority_numeric,
                    MAX(ud.female_prob) AS female_prob,
                    MAX(ud.white_prob) AS white_prob,
                    MAX(ud.black_prob) AS black_prob,
                    MAX(ud.api_prob) AS api_prob,
                    MAX(ud.hispanic_prob) AS hispanic_prob,
                    MAX(ud.native_prob) AS native_prob,
                    MAX(ud.multiple_prob) AS multiple_prob
                FROM us_positions p
                JOIN company_user_first_start cufs
                  ON cufs.user_id = p.user_id
                 AND cufs.rcid = p.rcid
                JOIN LATERAL generate_series(
                    GREATEST(EXTRACT(YEAR FROM p.startdate)::INT, {int(year_min)}),
                    LEAST(EXTRACT(YEAR FROM p.enddate)::INT, {int(year_max)})
                ) AS gs(year) ON TRUE
                LEFT JOIN read_parquet('{_escape(user_profile_cache_path)}') AS up
                  ON p.user_id = CAST(up.user_id AS BIGINT)
                LEFT JOIN user_demo AS ud
                  ON p.user_id = ud.user_id
                GROUP BY 1, 2, 3
            )
            SELECT
                CAST(ay.rcid AS BIGINT) AS c,
                CAST(ay.year AS INTEGER) AS t,
                COUNT(DISTINCT ay.user_id)::DOUBLE PRECISION AS total_headcount_wrds_annual,
                SUM(ay.p_likely_foreign)::DOUBLE PRECISION AS total_headcount_foreign_weighted_annual,
                SUM(ay.p_likely_native)::DOUBLE PRECISION AS total_headcount_native_weighted_annual,
                SUM(ay.likely_foreign_hard)::DOUBLE PRECISION AS total_headcount_foreign_hard_annual,
                SUM((1 - ay.likely_foreign_hard))::DOUBLE PRECISION AS total_headcount_native_hard_annual,
                COUNT(DISTINCT CASE WHEN make_date(ay.year, 12, 31) >= ay.first_start + INTERVAL '365 days' THEN ay.user_id END)
                    AS long_term_headcount_wrds_annual,
                AVG(ay.salary) AS salary_mean_annual,
                VAR_SAMP(ay.salary) AS salary_var_annual,
                AVG(ay.total_compensation) AS total_comp_mean_annual,
                VAR_SAMP(ay.total_compensation) AS total_comp_var_annual,
                AVG(CASE WHEN ay.salary IS NULL AND ay.total_compensation IS NULL THEN 1 ELSE 0 END)
                    AS compensation_missing_share_annual,
                {education_metric_sql}
                AVG(ay.female_prob) AS female_share_annual,
                AVG(ay.white_prob) AS race_share_white_annual,
                AVG(ay.black_prob) AS race_share_black_annual,
                AVG(ay.api_prob) AS race_share_api_annual,
                AVG(ay.hispanic_prob) AS race_share_hispanic_annual,
                AVG(ay.native_prob) AS race_share_native_annual,
                AVG(ay.multiple_prob) AS race_share_multiple_annual,
                AVG(ay.seniority_numeric) AS seniority_mean_annual,
                AVG(GREATEST(0.0, (make_date(ay.year, 12, 31) - ay.first_start)::DOUBLE PRECISION / 365.25)) AS avg_tenure_years_annual,
                AVG(CASE WHEN ay.soc2 = '11' THEN 1.0 ELSE 0.0 END) AS occ_share_mgmt_annual,
                AVG(CASE WHEN ay.soc2 = '13' THEN 1.0 ELSE 0.0 END) AS occ_share_business_finance_annual,
                AVG(CASE WHEN ay.soc2 = '15' THEN 1.0 ELSE 0.0 END) AS occ_share_computing_math_annual,
                AVG(CASE WHEN ay.soc2 = '17' THEN 1.0 ELSE 0.0 END) AS occ_share_engineering_annual,
                AVG(CASE WHEN ay.soc2 = '19' THEN 1.0 ELSE 0.0 END) AS occ_share_science_annual,
                AVG(CASE WHEN ay.soc2 IN ('21', '23', '25') THEN 1.0 ELSE 0.0 END) AS occ_share_community_legal_education_annual,
                AVG(CASE WHEN ay.soc2 = '27' THEN 1.0 ELSE 0.0 END) AS occ_share_arts_media_annual,
                AVG(CASE WHEN ay.soc2 IN ('29', '31') THEN 1.0 ELSE 0.0 END) AS occ_share_healthcare_annual,
                AVG(CASE WHEN ay.soc2 IN ('41', '43') THEN 1.0 ELSE 0.0 END) AS occ_share_sales_office_annual,
                AVG(CASE WHEN ay.soc2 IN ('47', '49', '51', '53') THEN 1.0 ELSE 0.0 END) AS occ_share_manual_annual,
                COALESCE(MAX(nh.n_new_hires_wrds_annual), 0)::DOUBLE PRECISION AS n_new_hires_wrds_annual,
                COALESCE(MAX(nh.n_new_hires_foreign_weighted_annual), 0.0)::DOUBLE PRECISION AS n_new_hires_foreign_weighted_annual,
                COALESCE(MAX(nh.n_new_hires_native_weighted_annual), 0.0)::DOUBLE PRECISION AS n_new_hires_native_weighted_annual,
                COALESCE(MAX(nh.n_new_hires_foreign_hard_annual), 0.0)::DOUBLE PRECISION AS n_new_hires_foreign_hard_annual,
                COALESCE(MAX(nh.n_new_hires_native_hard_annual), 0.0)::DOUBLE PRECISION AS n_new_hires_native_hard_annual
            FROM active_user_year ay
            LEFT JOIN new_hires nh
              ON nh.rcid = ay.rcid
             AND nh.year = ay.year
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).df()
    finally:
        con.close()

    if not workforce.empty:
        workforce["c"] = pd.to_numeric(workforce["c"], errors="coerce").astype(int)
        workforce["t"] = pd.to_numeric(workforce["t"], errors="coerce").astype(int)
    return workforce


def build_local_wrds_school_flows(
    *,
    selected_firms: pd.DataFrame,
    selected_positions_path: Path,
    users_path: Path,
    year_min: int,
    year_max: int,
    min_position_days: int,
    tenure_min_days: int,
) -> pd.DataFrame:
    selected = _clean_selected_firm_ids(selected_firms)
    if selected.empty:
        return pd.DataFrame(columns=["university_raw", "c", "t", "n_transitions", "n_emp", "total_new_hires"])

    con = ddb.connect()
    try:
        con.register("selected_firms_df", selected)
        con.sql(
            "CREATE OR REPLACE TEMP VIEW selected_firms AS "
            "SELECT CAST(c AS BIGINT) AS c FROM selected_firms_df"
        )
        out = con.sql(
            f"""
            WITH us_positions AS (
                SELECT
                    CAST(p.user_id AS BIGINT) AS user_id,
                    CAST(p.rcid AS BIGINT) AS rcid,
                    {_date_parse_sql("p.startdate")} AS startdate,
                    COALESCE({_date_parse_sql("p.enddate")}, DATE '2025-12-31') AS enddate
                FROM read_parquet('{_escape(selected_positions_path)}') AS p
                JOIN selected_firms sf
                  ON CAST(p.rcid AS BIGINT) = sf.c
                WHERE p.user_id IS NOT NULL
                  AND {_date_parse_sql("p.startdate")} IS NOT NULL
                  AND EXTRACT(YEAR FROM {_date_parse_sql("p.startdate")})::INT <= {int(year_max)}
                  AND EXTRACT(
                        YEAR FROM COALESCE({_date_parse_sql("p.enddate")}, DATE '2025-12-31')
                      )::INT >= {int(year_min)}
            ),
            education AS (
                SELECT
                    CAST(u.user_id AS BIGINT) AS user_id,
                    CAST(u.education_number AS BIGINT) AS education_number,
                    MAX(NULLIF(TRIM(CAST(u.university_raw AS VARCHAR)), '')) AS university_raw,
                    MAX({_date_parse_sql("u.ed_enddate")}) AS grad_date
                FROM read_parquet('{_escape(users_path)}') AS u
                WHERE u.user_id IS NOT NULL
                  AND u.education_number IS NOT NULL
                GROUP BY 1, 2
                HAVING MAX({_date_parse_sql("u.ed_enddate")}) IS NOT NULL
                   AND MAX(NULLIF(TRIM(CAST(u.university_raw AS VARCHAR)), '')) IS NOT NULL
            ),
            long_positions AS (
                SELECT user_id, rcid, startdate, enddate
                FROM us_positions
                WHERE enddate >= startdate + INTERVAL '{int(min_position_days)} days'
            ),
            positions_after_grad AS (
                SELECT
                    e.user_id,
                    e.university_raw,
                    e.education_number,
                    p.rcid,
                    p.startdate,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.user_id, e.university_raw
                        ORDER BY p.startdate
                    ) AS rank_after_grad
                FROM education e
                JOIN long_positions p
                  ON e.user_id = p.user_id
                 AND p.startdate >= e.grad_date
                 AND p.startdate <= e.grad_date + INTERVAL '1 years'
            ),
            first_jobs AS (
                SELECT *
                FROM positions_after_grad
                WHERE rank_after_grad = 1
                  AND EXTRACT(YEAR FROM startdate)::INT BETWEEN {int(year_min)} AND {int(year_max)}
            ),
            user_company_bounds AS (
                SELECT
                    user_id,
                    rcid,
                    MIN(startdate) AS first_startdate
                FROM us_positions
                GROUP BY 1, 2
            ),
            tenure_positions AS (
                SELECT
                    up.user_id,
                    up.rcid,
                    up.startdate,
                    up.enddate,
                    b.first_startdate
                FROM us_positions up
                JOIN user_company_bounds b
                  USING (user_id, rcid)
            ),
            long_term_employees AS (
                SELECT
                    tp.user_id,
                    tp.rcid,
                    tp.first_startdate,
                    tp.startdate,
                    tp.enddate,
                    edu.university_raw
                FROM tenure_positions tp
                JOIN education edu
                  ON tp.user_id = edu.user_id
                 AND edu.grad_date <= tp.startdate
            ),
            transition_counts AS (
                SELECT
                    university_raw,
                    rcid,
                    EXTRACT(YEAR FROM startdate)::INT AS year,
                    COUNT(DISTINCT user_id) AS n_transitions
                FROM first_jobs
                GROUP BY university_raw, rcid, year
            ),
            employee_counts AS (
                SELECT
                    university_raw,
                    rcid,
                    gs.year,
                    COUNT(DISTINCT user_id) AS n_emp
                FROM long_term_employees,
                LATERAL generate_series(
                    GREATEST(EXTRACT(YEAR FROM startdate)::INT, {int(year_min)}),
                    LEAST(EXTRACT(YEAR FROM enddate)::INT, {int(year_max)})
                ) AS gs(year)
                WHERE make_date(gs.year, 12, 31) >= first_startdate + INTERVAL '{int(tenure_min_days)} days'
                GROUP BY university_raw, rcid, gs.year
            ),
            new_hires AS (
                SELECT
                    user_id,
                    rcid,
                    MIN(startdate) AS first_start
                FROM long_positions
                GROUP BY user_id, rcid
            ),
            new_hire_counts AS (
                SELECT
                    rcid,
                    EXTRACT(YEAR FROM first_start)::INT AS year,
                    COUNT(DISTINCT user_id) AS total_new_hires
                FROM new_hires
                WHERE EXTRACT(YEAR FROM first_start)::INT BETWEEN {int(year_min)} AND {int(year_max)}
                GROUP BY rcid, year
            )
            SELECT
                COALESCE(t.university_raw, e.university_raw) AS university_raw,
                COALESCE(CAST(t.rcid AS BIGINT), CAST(e.rcid AS BIGINT)) AS c,
                COALESCE(CAST(t.year AS INTEGER), CAST(e.year AS INTEGER)) AS t,
                CAST(t.n_transitions AS DOUBLE PRECISION) AS n_transitions,
                CAST(e.n_emp AS DOUBLE PRECISION) AS n_emp,
                CAST(n.total_new_hires AS DOUBLE PRECISION) AS total_new_hires
            FROM transition_counts t
            FULL OUTER JOIN employee_counts e
              ON t.rcid = e.rcid
             AND t.university_raw = e.university_raw
             AND t.year = e.year
            LEFT JOIN new_hire_counts n
              ON COALESCE(t.rcid, e.rcid) = n.rcid
             AND COALESCE(t.year, e.year) = n.year
            ORDER BY 2, 3
            """
        ).df()
    finally:
        con.close()

    if out.empty:
        return pd.DataFrame(columns=["university_raw", "c", "t", "n_transitions", "n_emp", "total_new_hires"])
    out["c"] = pd.to_numeric(out["c"], errors="coerce").astype(int)
    out["t"] = pd.to_numeric(out["t"], errors="coerce").astype(int)
    return out.drop_duplicates(subset=["c", "t", "university_raw"], keep="first").reset_index(drop=True)


def _merge_local_profile_metrics_into_workforce(
    base_workforce: pd.DataFrame,
    local_metrics: pd.DataFrame,
) -> pd.DataFrame:
    validation = base_workforce[["c", "t", "total_headcount_wrds_annual", "n_new_hires_wrds_annual"]].merge(
        local_metrics[["c", "t"] + LOCAL_PROFILE_VALIDATION_COLUMNS],
        on=["c", "t"],
        how="left",
    )
    if validation["local_total_headcount_wrds_annual"].isna().any():
        missing = int(validation["local_total_headcount_wrds_annual"].isna().sum())
        raise ValueError(
            "Local workforce profile metrics are missing headcount rows for "
            f"{missing:,} company-year observations."
        )
    if validation["local_n_new_hires_wrds_annual"].isna().any():
        missing = int(validation["local_n_new_hires_wrds_annual"].isna().sum())
        raise ValueError(
            "Local workforce profile metrics are missing new-hire rows for "
            f"{missing:,} company-year observations."
        )

    headcount_diff = (
        pd.to_numeric(validation["total_headcount_wrds_annual"], errors="coerce")
        - pd.to_numeric(validation["local_total_headcount_wrds_annual"], errors="coerce")
    ).abs()
    if bool((headcount_diff > 1e-6).any()):
        max_diff = float(headcount_diff.max())
        raise ValueError(
            "Local workforce profile metrics do not match WRDS total headcount totals; "
            f"max absolute difference = {max_diff:.6f}."
        )

    new_hires_diff = (
        pd.to_numeric(validation["n_new_hires_wrds_annual"], errors="coerce").fillna(0.0)
        - pd.to_numeric(validation["local_n_new_hires_wrds_annual"], errors="coerce").fillna(0.0)
    ).abs()
    if bool((new_hires_diff > 1e-6).any()):
        max_diff = float(new_hires_diff.max())
        raise ValueError(
            "Local workforce profile metrics do not match WRDS new-hire totals; "
            f"max absolute difference = {max_diff:.6f}."
        )

    merged = base_workforce.drop(columns=LOCAL_PROFILE_WORKFORCE_COLUMNS, errors="ignore").merge(
        local_metrics.drop(columns=LOCAL_PROFILE_VALIDATION_COLUMNS, errors="ignore"),
        on=["c", "t"],
        how="left",
    )
    return merged


def _build_wrds_school_flow_query(
    rcids: list[int],
    *,
    year_min: int,
    year_max: int,
    min_position_days: int,
    tenure_min_days: int,
    history_year_min: Optional[int] = None,
    history_year_max: Optional[int] = None,
) -> str:
    rcid_sql = ", ".join(str(int(v)) for v in rcids)
    history_year_min_int = int(year_min if history_year_min is None else history_year_min)
    history_year_max_int = int(year_max if history_year_max is None else history_year_max)
    return f"""
    WITH us_positions AS MATERIALIZED (
        SELECT
            user_id,
            rcid,
            startdate::DATE AS startdate,
            COALESCE(enddate::DATE, DATE '2025-12-31') AS enddate
        FROM revelio.individual_positions
        WHERE country = 'United States'
          AND rcid IN ({rcid_sql})
          AND startdate IS NOT NULL
          AND EXTRACT(YEAR FROM startdate)::INT <= {history_year_max_int}
          AND EXTRACT(YEAR FROM COALESCE(enddate::DATE, DATE '2025-12-31'))::INT >= {history_year_min_int}
    ),
    company_users AS MATERIALIZED (
        SELECT DISTINCT user_id
        FROM us_positions
    ),
    education_clean AS (
        SELECT
            e.user_id,
            e.education_number,
            e.enddate::DATE AS grad_date
        FROM revelio.individual_user_education AS e
        JOIN company_users cu
          ON cu.user_id = e.user_id
        WHERE enddate IS NOT NULL
    ),
    education_raw AS (
        SELECT
            r.user_id,
            r.education_number,
            MAX(NULLIF(TRIM(CAST(r.university_raw AS VARCHAR)), '')) AS university_raw
        FROM revelio.individual_user_education_raw AS r
        JOIN company_users cu
          ON cu.user_id = r.user_id
        WHERE university_raw IS NOT NULL
          AND TRIM(university_raw) <> ''
        GROUP BY 1, 2
    ),
    education AS MATERIALIZED (
        SELECT
            e.user_id,
            e.education_number,
            r.university_raw,
            e.grad_date
        FROM education_clean e
        JOIN education_raw r
          ON e.user_id = r.user_id
         AND e.education_number = r.education_number
        WHERE e.grad_date IS NOT NULL
    ),
    long_positions AS MATERIALIZED (
        SELECT user_id, rcid, startdate
        FROM us_positions
        WHERE enddate >= startdate + INTERVAL '{int(min_position_days)} days'
    ),
    positions_after_grad AS (
        SELECT
            e.user_id,
            e.university_raw,
            e.education_number,
            p.rcid,
            p.startdate,
            ROW_NUMBER() OVER (
                PARTITION BY e.user_id, e.university_raw
                ORDER BY p.startdate
            ) AS rank_after_grad
        FROM education e
        JOIN long_positions p
          ON e.user_id = p.user_id
         AND p.startdate >= e.grad_date
         AND p.startdate <= e.grad_date + INTERVAL '1 years'
    ),
    first_jobs AS MATERIALIZED (
        SELECT *
        FROM positions_after_grad
        WHERE rank_after_grad = 1
          AND EXTRACT(YEAR FROM startdate)::INT BETWEEN {int(year_min)} AND {int(year_max)}
    ),
    user_company_bounds AS (
        SELECT
            user_id,
            rcid,
            MIN(startdate) AS first_startdate
        FROM us_positions
        GROUP BY user_id, rcid
    ),
    tenure_positions AS (
        SELECT
            up.user_id,
            up.rcid,
            up.startdate,
            up.enddate,
            b.first_startdate
        FROM us_positions up
        JOIN user_company_bounds b
          USING (user_id, rcid)
    ),
    long_term_employees AS MATERIALIZED (
        SELECT
            tp.user_id,
            tp.rcid,
            tp.first_startdate,
            tp.startdate,
            tp.enddate,
            edu.university_raw
        FROM tenure_positions tp
        JOIN education edu
          ON tp.user_id = edu.user_id
         AND edu.university_raw IS NOT NULL
         AND edu.grad_date <= tp.startdate
    ),
    transition_counts AS (
        SELECT
            university_raw,
            rcid,
            EXTRACT(YEAR FROM startdate)::INT AS year,
            COUNT(DISTINCT user_id) AS n_transitions
        FROM first_jobs
        GROUP BY university_raw, rcid, year
    ),
    employee_counts AS (
        SELECT
            university_raw,
            rcid,
            gs.year,
            COUNT(DISTINCT user_id) AS n_emp
        FROM long_term_employees,
        LATERAL generate_series(
            GREATEST(EXTRACT(YEAR FROM startdate)::INT, {int(year_min)}),
            LEAST(EXTRACT(YEAR FROM enddate)::INT, {int(year_max)})
        ) AS gs(year)
        WHERE make_date(gs.year, 12, 31) >= first_startdate + INTERVAL '{int(tenure_min_days)} days'
        GROUP BY university_raw, rcid, gs.year
    ),
    new_hires AS (
        SELECT
            user_id,
            rcid,
            MIN(startdate) AS first_start
        FROM long_positions
        GROUP BY user_id, rcid
    ),
    new_hire_counts AS (
        SELECT
            rcid,
            EXTRACT(YEAR FROM first_start)::INT AS year,
            COUNT(DISTINCT user_id) AS total_new_hires
        FROM new_hires
        WHERE EXTRACT(YEAR FROM first_start)::INT BETWEEN {int(year_min)} AND {int(year_max)}
        GROUP BY rcid, year
    )
    SELECT
        COALESCE(t.university_raw, e.university_raw) AS university_raw,
        COALESCE(CAST(t.rcid AS BIGINT), CAST(e.rcid AS BIGINT)) AS c,
        COALESCE(CAST(t.year AS INTEGER), CAST(e.year AS INTEGER)) AS t,
        CAST(t.n_transitions AS DOUBLE PRECISION) AS n_transitions,
        CAST(e.n_emp AS DOUBLE PRECISION) AS n_emp,
        CAST(n.total_new_hires AS DOUBLE PRECISION) AS total_new_hires
    FROM transition_counts t
    FULL OUTER JOIN employee_counts e
      ON t.rcid = e.rcid
     AND t.university_raw = e.university_raw
     AND t.year = e.year
    LEFT JOIN new_hire_counts n
      ON COALESCE(t.rcid, e.rcid) = n.rcid
     AND COALESCE(t.year, e.year) = n.year
    ORDER BY 2, 3
    """


def build_wrds_school_flows(
    rcids: list[int],
    *,
    wrds_username: str,
    year_min: int,
    year_max: int,
    rcid_batch_size: int,
    rcid_n_users: Optional[pd.DataFrame | dict[int, float]] = None,
    large_firm_n_users_threshold: Optional[float] = None,
    large_firm_year_span: int = 1,
    query_timeout_ms: Optional[int],
    singleton_query_timeout_ms: Optional[int] = None,
    query_max_retries: int,
    min_position_days: int,
    tenure_min_days: int,
) -> pd.DataFrame:
    if not rcids:
        return pd.DataFrame(columns=["c", "t", "university_raw"])
    if wrds is None:  # pragma: no cover
        raise ImportError("wrds is not installed.")

    build_t0 = time.time()
    relaxed_singleton_timeout_ms = _resolve_singleton_query_timeout_ms(
        query_timeout_ms=query_timeout_ms,
        singleton_query_timeout_ms=singleton_query_timeout_ms,
    )
    _log(
        f"[wrds_school_flows] START: {len(rcids):,} firm ids | years {int(year_min)}–{int(year_max)} | "
        f"batch_size={int(rcid_batch_size)}"
    )
    if relaxed_singleton_timeout_ms is not None and (
        query_timeout_ms is None or int(relaxed_singleton_timeout_ms) > int(query_timeout_ms)
    ):
        _log(
            "[wrds_school_flows] Singleton fallback timeout: "
            f"{int(round(relaxed_singleton_timeout_ms / 60_000))} minutes"
        )
    db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
    batch_queue, task_meta = _build_wrds_task_queue(
        rcids,
        label_prefix="school-flow",
        rcid_batch_size=int(rcid_batch_size),
        year_min=int(year_min),
        year_max=int(year_max),
        rcid_n_users=rcid_n_users,
        large_firm_n_users_threshold=large_firm_n_users_threshold,
        large_firm_year_span=int(large_firm_year_span),
    )
    if int(task_meta["n_large_firms"]) > 0:
        threshold = task_meta["large_firm_n_users_threshold"]
        span = int(task_meta["large_firm_year_span"])
        _log(
            "[wrds_school_flows] Proactive large-firm slicing: "
            f"{int(task_meta['n_large_firms']):,} firm(s) with n_users >= {int(threshold):,} "
            f"scheduled as {int(task_meta['n_large_firm_tasks']):,} singleton task(s) "
            f"using {span}-year windows"
        )
    frames: list[pd.DataFrame] = []
    try:
        while batch_queue:
            task = batch_queue.popleft()
            batch = list(task.rcids)
            label = task.label
            batch_t0 = time.time()
            batch_min = min(batch)
            batch_max = max(batch)
            task_years_msg = (
                f" | output years {int(task.year_min)}–{int(task.year_max)}"
                if int(task.year_min) != int(year_min) or int(task.year_max) != int(year_max)
                else ""
            )
            _log(
                f"[wrds_school_flows] START {label}: {len(batch):,} firms | "
                f"rcid range {batch_min:,}–{batch_max:,}{task_years_msg}"
            )
            sql = _build_wrds_school_flow_query(
                batch,
                year_min=task.year_min,
                year_max=task.year_max,
                min_position_days=min_position_days,
                tenure_min_days=tenure_min_days,
                history_year_min=task.history_year_min,
                history_year_max=task.history_year_max,
            )
            try:
                df, db = _run_sql_with_retries(
                    db,
                    sql,
                    wrds_username=wrds_username,
                    query_timeout_ms=query_timeout_ms,
                    max_retries=query_max_retries,
                    label=label,
                )
            except Exception as exc:
                is_timeout = _wrds_exception_is_timeout(exc)
                if len(batch) > 1 and is_timeout:
                    mid = len(batch) // 2
                    left = batch[:mid]
                    right = batch[mid:]
                    _log(
                        f"[wrds_school_flows] SPLIT {label}: timeout after "
                        f"{_format_elapsed(time.time() - batch_t0)} | "
                        f"{len(batch):,} firms -> {len(left):,} + {len(right):,} firms"
                    )
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
                    batch_queue.appendleft(
                        WrdsQueryTask(
                            rcids=tuple(right),
                            label=f"{label}.2",
                            year_min=int(task.year_min),
                            year_max=int(task.year_max),
                            history_year_min=int(task.history_year_min),
                            history_year_max=int(task.history_year_max),
                        )
                    )
                    batch_queue.appendleft(
                        WrdsQueryTask(
                            rcids=tuple(left),
                            label=f"{label}.1",
                            year_min=int(task.year_min),
                            year_max=int(task.year_max),
                            history_year_min=int(task.history_year_min),
                            history_year_max=int(task.history_year_max),
                        )
                    )
                    continue
                if len(batch) == 1 and is_timeout and relaxed_singleton_timeout_ms is not None:
                    relaxed_label = f"{label} singleton-fallback"
                    slice_windows = _year_windows(int(task.year_min), int(task.year_max), 1)
                    slice_mode = (
                        "with relaxed timeout"
                        if len(slice_windows) == 1
                        else "as 1-year windows with relaxed timeout"
                    )
                    _log(
                        f"[wrds_school_flows] RETRY {label}: timed out after "
                        f"{_format_elapsed(time.time() - batch_t0)} | "
                        f"retrying rcid {batch[0]:,} {slice_mode} "
                        f"({int(round(relaxed_singleton_timeout_ms / 60_000))} minutes)"
                    )
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = _open_wrds_connection(
                        wrds_username=wrds_username,
                        query_timeout_ms=relaxed_singleton_timeout_ms,
                    )
                    try:
                        year_frames: list[pd.DataFrame] = []
                        for slice_year_min, slice_year_max in slice_windows:
                            slice_suffix = (
                                f"y{slice_year_min}"
                                if slice_year_min == slice_year_max
                                else f"y{slice_year_min}-{slice_year_max}"
                            )
                            slice_label = (
                                relaxed_label
                                if len(slice_windows) == 1
                                else f"{relaxed_label} {slice_suffix}"
                            )
                            slice_sql = _build_wrds_school_flow_query(
                                batch,
                                year_min=slice_year_min,
                                year_max=slice_year_max,
                                history_year_min=task.history_year_min,
                                history_year_max=slice_year_max,
                                min_position_days=min_position_days,
                                tenure_min_days=tenure_min_days,
                            )
                            slice_df, db = _run_sql_with_retries(
                                db,
                                slice_sql,
                                wrds_username=wrds_username,
                                query_timeout_ms=relaxed_singleton_timeout_ms,
                                max_retries=query_max_retries,
                                label=slice_label,
                            )
                            year_frames.append(slice_df)
                        df = (
                            pd.concat(year_frames, ignore_index=True)
                            if year_frames
                            else pd.DataFrame(columns=["c", "t", "university_raw"])
                        )
                    except Exception as relaxed_exc:
                        if _wrds_exception_is_timeout(relaxed_exc):
                            _log(
                                f"[wrds_school_flows] FATAL {label}: single-firm batch timed out for rcid "
                                f"{batch[0]:,} even after refined singleton fallback "
                                f"({_format_elapsed(time.time() - batch_t0)})"
                            )
                        raise
                elif len(batch) == 1 and is_timeout:
                    _log(
                        f"[wrds_school_flows] FATAL {label}: single-firm batch timed out for rcid "
                        f"{batch[0]:,} after {_format_elapsed(time.time() - batch_t0)}"
                    )
                    raise
                else:
                    raise
            if not df.empty:
                df["c"] = pd.to_numeric(df["c"], errors="coerce").astype("Int64")
                df["t"] = pd.to_numeric(df["t"], errors="coerce").astype("Int64")
                df = df.dropna(subset=["c", "t"]).copy()
                df["c"] = df["c"].astype(int)
                df["t"] = df["t"].astype(int)
            frames.append(df)
            _log(
                f"[wrds_school_flows] DONE  {label}: {len(df):,} rows | "
                f"elapsed {_format_elapsed(time.time() - batch_t0)}"
            )
    finally:
        try:
            db.close()
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["c", "t", "university_raw"])
    out = pd.concat(frames, ignore_index=True)
    _log(
        f"[wrds_school_flows] DONE: {len(out):,} rows before dedup | "
        f"elapsed {_format_elapsed(time.time() - build_t0)}"
    )
    return out.drop_duplicates(subset=["c", "t", "university_raw"], keep="first")


def load_or_build_wrds_company_year_workforce_cache(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    year_min: int,
    year_max: int,
    force_rebuild: bool = False,
    cache_only: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cache_t0 = time.time()
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = resolve_source_exposure_paths(cfg_full)
    feature_cfg = get_cfg_section(cfg_full, "revelio_company_features")
    firms, _, selected_meta, universe_meta = load_or_build_source_firm_universe(
        cfg=cfg_full,
        force_rebuild=force_rebuild,
    )
    include_education_features = bool(feature_cfg.get("wrds_workforce_include_education_features", True))
    local_profile_settings = _resolve_local_user_profile_refactor_settings(
        cfg_full,
        workforce_out_path=paths.wrds_company_year_workforce_out,
    )
    use_full_local_workforce = bool(local_profile_settings["enabled"]) and bool(
        local_profile_settings.get("use_dedicated_extracts")
    )
    extract_year_min = int(year_min)
    extract_year_max = int(year_max)
    extract_window_meta: dict[str, object] = {}
    if use_full_local_workforce:
        extract_year_min, extract_year_max, extract_window_meta = _resolve_shared_local_extract_year_window(
            cfg_full,
            requested_year_min=int(year_min),
            requested_year_max=int(year_max),
        )
    expected_meta = {
        "year_min": int(year_min),
        "year_max": int(year_max),
        "wrds_workforce_include_education_features": include_education_features,
        "new_hire_origin_method": NEW_HIRE_ORIGIN_METHOD,
        "local_user_profile_refactor_enabled": bool(local_profile_settings["enabled"]),
        "local_user_profile_source_mode": (
            str(local_profile_settings["source_mode"]) if bool(local_profile_settings["enabled"]) else None
        ),
        "local_user_profile_cache_method": (
            LOCAL_USER_PROFILE_CACHE_METHOD if bool(local_profile_settings["enabled"]) else None
        ),
        "local_workforce_panel_method": (
            LOCAL_WORKFORCE_PANEL_METHOD
            if bool(local_profile_settings["enabled"]) and bool(local_profile_settings.get("use_dedicated_extracts"))
            else None
        ),
        "local_wrds_users_path": (
            (
                str(local_profile_settings["extract_users_path"])
                if local_profile_settings.get("use_dedicated_extracts")
                else str(local_profile_settings["snapshot_users_path"])
            )
            if local_profile_settings["enabled"]
            else None
        ),
        "local_wrds_positions_path": (
            (
                str(local_profile_settings["extract_positions_path"])
                if local_profile_settings.get("use_dedicated_extracts")
                else str(local_profile_settings["snapshot_positions_path"])
            )
            if local_profile_settings["enabled"]
            else None
        ),
        "local_selected_us_positions_path": (
            str(local_profile_settings["extract_selected_positions_path"])
            if bool(local_profile_settings["enabled"]) and bool(local_profile_settings.get("use_dedicated_extracts"))
            else None
        ),
        "workforce_wrds_extract_method": (
            WORKFORCE_WRDS_EXTRACT_METHOD if local_profile_settings.get("use_dedicated_extracts") else None
        ),
        "local_extract_year_min": extract_year_min if use_full_local_workforce else None,
        "local_extract_year_max": extract_year_max if use_full_local_workforce else None,
        "local_profile_cache_year_min": extract_year_min if use_full_local_workforce else None,
        "local_profile_cache_year_max": extract_year_max if use_full_local_workforce else None,
        **extract_window_meta,
        **universe_meta,
    }
    reuse_cached_universe_only = _reuse_cached_wrds_universe_only(cfg_full)
    expected_meta_compare = _cache_meta_for_compatibility(
        expected_meta,
        ignore_keys=(
            _wrds_universe_ignore_keys_for_panels()
            if reuse_cached_universe_only
            else None
        ),
    )
    out_path = paths.wrds_company_year_workforce_out
    cache_key = _wrds_workforce_cache_key(out_path, universe_meta)

    if cache_only:
        if not out_path.exists():
            raise FileNotFoundError(
                f"[wrds_workforce_cache] cache_only=True but cache file is missing: {out_path}"
            )

        cached = pd.read_parquet(out_path)
        required_cols = {"c", "t", "avg_tenure_years_annual"}
        if not required_cols.issubset(set(cached.columns)):
            raise RuntimeError(
                "[wrds_workforce_cache] cache_only=True but cached workforce file does not contain "
                f"required columns: {sorted(required_cols - set(cached.columns))}"
            )
        cached["t"] = pd.to_numeric(cached["t"], errors="coerce")
        if cached["t"].dropna().empty:
            raise RuntimeError("[wrds_workforce_cache] cache_only=True but cached workforce file has no parseable t values.")
        cached_min = int(cached["t"].min())
        cached_max = int(cached["t"].max())
        if cached_min > int(year_min) or cached_max < int(year_max):
            raise RuntimeError(
                "[wrds_workforce_cache] cache_only=True but cached workforce coverage "
                f"({cached_min}–{cached_max}) does not cover requested ({int(year_min)}–{int(year_max)})."
            )
        cached = cached[cached["t"].between(int(year_min), int(year_max))].copy()
        cached_meta = _load_metadata(out_path) or {"year_min": cached_min, "year_max": cached_max}
        _WRDS_WORKFORCE_CACHE[cache_key] = _clone_wrds_workforce_cache_result((cached, cached_meta))
        _log(
            f"[wrds_workforce_cache] REUSE (cache_only): {out_path} | rows {len(cached):,} | "
            f"requested years {int(year_min)}–{int(year_max)}"
        )
        return cached.reset_index(drop=True), cached_meta

    cached_result = _WRDS_WORKFORCE_CACHE.get(cache_key)
    if cached_result is not None:
        cached_df, cached_meta = _clone_wrds_workforce_cache_result(cached_result)
        if (
            _metadata_compatible(
                cached_meta,
                {k: v for k, v in expected_meta_compare.items() if k not in {"year_min", "year_max"}},
            )
            and (
                cached_meta.get("year_min") is not None
                and cached_meta.get("year_max") is not None
                and int(cached_meta["year_min"]) <= int(year_min)
                and int(cached_meta["year_max"]) >= int(year_max)
            )
        ):
            cached_df = cached_df[cached_df["t"].between(int(year_min), int(year_max))].copy()
            _log(
                f"[wrds_workforce_cache] REUSE in-process: {out_path} | rows {len(cached_df):,} | "
                f"elapsed {_format_elapsed(time.time() - cache_t0)}"
            )
            return cached_df.reset_index(drop=True), cached_meta

    if out_path.exists() and not force_rebuild:
        meta = _load_metadata(out_path)
        compatible = _metadata_compatible(
            meta,
            {k: v for k, v in expected_meta_compare.items() if k not in {"year_min", "year_max"}},
        )
        if compatible and meta.get("year_min") is not None and meta.get("year_max") is not None:
            cached_min = int(meta["year_min"])
            cached_max = int(meta["year_max"])
            if cached_min <= int(year_min) and cached_max >= int(year_max):
                cached = pd.read_parquet(out_path)
                _WRDS_WORKFORCE_CACHE[cache_key] = _clone_wrds_workforce_cache_result((cached, meta))
                cached = cached[cached["t"].between(int(year_min), int(year_max))].copy()
                _log(
                    f"[wrds_workforce_cache] REUSE: {out_path} | rows {len(cached):,} | "
                    f"elapsed {_format_elapsed(time.time() - cache_t0)}"
                )
                if reuse_cached_universe_only and (
                    meta.get("analysis_firms_hash") != expected_meta.get("analysis_firms_hash")
                ):
                    _log(
                        "[wrds_workforce_cache] REUSE under reuse_cached_wrds_universe_only=true; "
                        "ignoring firm-universe hash mismatch for testing."
                    )
                return cached.reset_index(drop=True), meta

    _log(
        f"[wrds_workforce_cache] BUILD: {out_path} | {len(firms):,} firms | "
        f"years {int(year_min)}–{int(year_max)}"
    )
    if use_full_local_workforce and (extract_year_min != int(year_min) or extract_year_max != int(year_max)):
        _log(
            "[wrds_workforce_cache] Shared local WRDS extract coverage expanded to "
            f"{extract_year_min}–{extract_year_max} so the same extract can serve both "
            f"the requested {int(year_min)}–{int(year_max)} build and broader event-study stages."
        )
    if use_full_local_workforce:
        local_users_path, local_positions_path, local_selected_positions_path, extract_meta = (
            load_or_build_wrds_workforce_local_extracts(
                cfg=cfg_full,
                settings=local_profile_settings,
                selected_firms=firms[["c"]],
                rcid_n_users=selected_meta[["c", "n_users"]] if "n_users" in selected_meta.columns else None,
                year_min=extract_year_min,
                year_max=extract_year_max,
                force_rebuild=force_rebuild,
            )
        )
        expected_meta["workforce_wrds_extract_meta"] = extract_meta
        profile_cache_path, profile_cache_meta = load_or_build_local_wrds_user_profile_cache(
            selected_firms=firms[["c"]],
            users_path=local_users_path,
            positions_path=local_positions_path,
            out_path=local_profile_settings["cache_path"],  # type: ignore[arg-type]
            year_min=extract_year_min,
            year_max=extract_year_max,
            force_rebuild=force_rebuild,
        )
        expected_meta["local_user_profile_cache_meta"] = profile_cache_meta
        df = build_local_wrds_company_year_workforce(
            selected_firms=firms[["c"]],
            users_path=local_users_path,
            selected_positions_path=local_selected_positions_path,
            user_profile_cache_path=profile_cache_path,
            year_min=int(year_min),
            year_max=int(year_max),
            include_education_features=include_education_features,
        )
    else:
        df = build_wrds_company_year_workforce(
            firms["c"].dropna().astype(int).tolist(),
            wrds_username=str(feature_cfg.get("wrds_username", "")).strip(),
            year_min=int(year_min),
            year_max=int(year_max),
            rcid_batch_size=int(feature_cfg.get("wrds_rcid_batch_size", 100)),
            rcid_n_users=selected_meta[["c", "n_users"]] if "n_users" in selected_meta.columns else None,
            large_firm_n_users_threshold=feature_cfg.get("wrds_large_firm_n_users_threshold", 75_000),
            large_firm_year_span=int(feature_cfg.get("wrds_large_firm_year_span", 1)),
            include_education_features=include_education_features,
            query_timeout_ms=int(float(feature_cfg.get("query_timeout_minutes", 10)) * 60_000),
            singleton_query_timeout_ms=(
                int(float(feature_cfg.get("wrds_singleton_query_timeout_minutes")) * 60_000)
                if feature_cfg.get("wrds_singleton_query_timeout_minutes") is not None
                else None
            ),
            query_max_retries=int(feature_cfg.get("query_max_retries", 1)),
            compute_origin_and_education_locally=bool(local_profile_settings["enabled"]),
        )
    if bool(local_profile_settings["enabled"]):
        if not use_full_local_workforce:
            local_users_path: Path
            local_positions_path: Path
            local_users_path = local_profile_settings["snapshot_users_path"]  # type: ignore[assignment]
            local_positions_path = local_profile_settings["snapshot_positions_path"]  # type: ignore[assignment]
            profile_cache_path, profile_cache_meta = load_or_build_local_wrds_user_profile_cache(
                selected_firms=firms[["c"]],
                users_path=local_users_path,
                positions_path=local_positions_path,
                out_path=local_profile_settings["cache_path"],  # type: ignore[arg-type]
                year_min=int(year_min),
                year_max=int(year_max),
                force_rebuild=force_rebuild,
            )
            local_metrics = build_local_company_year_profile_metrics(
                selected_firms=firms[["c"]],
                positions_path=local_positions_path,
                user_profile_cache_path=profile_cache_path,
                year_min=int(year_min),
                year_max=int(year_max),
                include_education_features=include_education_features,
            )
            df = _merge_local_profile_metrics_into_workforce(df, local_metrics)
            expected_meta["local_user_profile_cache_meta"] = profile_cache_meta
    df = df.merge(
        firms[["c", "in_analysis_universe", "preferred_rcid_source", "outside_negative_candidate"]],
        on="c",
        how="left",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    meta = expected_meta | {
        "n_rows": int(len(df)),
        "n_firms": int(df["c"].nunique()) if "c" in df.columns else 0,
    }
    _write_metadata(out_path, meta)
    _WRDS_WORKFORCE_CACHE[cache_key] = _clone_wrds_workforce_cache_result((df, meta))
    _log(
        f"[wrds_workforce_cache] DONE: wrote {out_path} | rows {len(df):,} | "
        f"elapsed {_format_elapsed(time.time() - cache_t0)}"
    )
    return _clone_wrds_workforce_cache_result((df, meta))


def load_or_build_wrds_school_flows_cache(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    year_min: int,
    year_max: int,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cache_t0 = time.time()
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = resolve_source_exposure_paths(cfg_full)
    feature_cfg = get_cfg_section(cfg_full, "revelio_company_features")
    firms, _, selected_meta, universe_meta = load_or_build_source_firm_universe(
        cfg=cfg_full,
        force_rebuild=force_rebuild,
    )
    local_profile_settings = _resolve_local_user_profile_refactor_settings(
        cfg_full,
        workforce_out_path=paths.wrds_company_year_workforce_out,
    )
    use_full_local_school_flows = bool(local_profile_settings["enabled"]) and bool(
        local_profile_settings.get("use_dedicated_extracts")
    )
    extract_year_min = int(year_min)
    extract_year_max = int(year_max)
    extract_window_meta: dict[str, object] = {}
    if use_full_local_school_flows:
        extract_year_min, extract_year_max, extract_window_meta = _resolve_shared_local_extract_year_window(
            cfg_full,
            requested_year_min=int(year_min),
            requested_year_max=int(year_max),
        )
    expected_meta = {
        "year_min": int(year_min),
        "year_max": int(year_max),
        "min_position_days": int(feature_cfg.get("min_position_days", 365)),
        "tenure_min_days": int(feature_cfg.get("tenure_min_days", 365)),
        "local_school_flows_method": (
            LOCAL_SCHOOL_FLOWS_METHOD
            if bool(local_profile_settings["enabled"]) and bool(local_profile_settings.get("use_dedicated_extracts"))
            else None
        ),
        "workforce_wrds_extract_method": (
            WORKFORCE_WRDS_EXTRACT_METHOD if local_profile_settings.get("use_dedicated_extracts") else None
        ),
        "local_extract_year_min": extract_year_min if use_full_local_school_flows else None,
        "local_extract_year_max": extract_year_max if use_full_local_school_flows else None,
        **extract_window_meta,
        **universe_meta,
    }
    reuse_cached_universe_only = _reuse_cached_wrds_universe_only(cfg_full)
    expected_meta_compare = _cache_meta_for_compatibility(
        expected_meta,
        ignore_keys=(
            _wrds_universe_ignore_keys_for_panels()
            if reuse_cached_universe_only
            else None
        ),
    )
    out_path = paths.wrds_school_flows_out
    if out_path.exists() and not force_rebuild:
        meta = _load_metadata(out_path)
        if (
            _metadata_compatible(
                meta,
                {k: v for k, v in expected_meta_compare.items() if k not in {"year_min", "year_max"}},
            )
            and meta.get("year_min") is not None
            and meta.get("year_max") is not None
            and int(meta["year_min"]) <= int(year_min)
            and int(meta["year_max"]) >= int(year_max)
        ):
            cached = pd.read_parquet(out_path)
            cached = cached[cached["t"].between(int(year_min), int(year_max))].copy()
            _log(
                f"[wrds_school_flows_cache] REUSE: {out_path} | rows {len(cached):,} | "
                f"elapsed {_format_elapsed(time.time() - cache_t0)}"
            )
            if reuse_cached_universe_only and (
                meta.get("analysis_firms_hash") != expected_meta.get("analysis_firms_hash")
            ):
                _log(
                    "[wrds_school_flows_cache] REUSE under reuse_cached_wrds_universe_only=true; "
                    "ignoring firm-universe hash mismatch for testing."
                )
            return cached, meta

    _log(
        f"[wrds_school_flows_cache] BUILD: {out_path} | {len(firms):,} firms | "
        f"years {int(year_min)}–{int(year_max)}"
    )
    if use_full_local_school_flows and (extract_year_min != int(year_min) or extract_year_max != int(year_max)):
        _log(
            "[wrds_school_flows_cache] Shared local WRDS extract coverage expanded to "
            f"{extract_year_min}–{extract_year_max} so school-flow builds reuse the same "
            "superset extract as the analysis panel."
        )
    if use_full_local_school_flows:
        local_users_path, _, local_selected_positions_path, extract_meta = (
            load_or_build_wrds_workforce_local_extracts(
                cfg=cfg_full,
                settings=local_profile_settings,
                selected_firms=firms[["c"]],
                rcid_n_users=selected_meta[["c", "n_users"]] if "n_users" in selected_meta.columns else None,
                year_min=extract_year_min,
                year_max=extract_year_max,
                force_rebuild=force_rebuild,
            )
        )
        expected_meta["workforce_wrds_extract_meta"] = extract_meta
        df = build_local_wrds_school_flows(
            selected_firms=firms[["c"]],
            selected_positions_path=local_selected_positions_path,
            users_path=local_users_path,
            year_min=int(year_min),
            year_max=int(year_max),
            min_position_days=int(feature_cfg.get("min_position_days", 365)),
            tenure_min_days=int(feature_cfg.get("tenure_min_days", 365)),
        )
    else:
        df = build_wrds_school_flows(
            firms["c"].dropna().astype(int).tolist(),
            wrds_username=str(feature_cfg.get("wrds_username", "")).strip(),
            year_min=int(year_min),
            year_max=int(year_max),
            rcid_batch_size=int(feature_cfg.get("wrds_rcid_batch_size", 100)),
            rcid_n_users=selected_meta[["c", "n_users"]] if "n_users" in selected_meta.columns else None,
            large_firm_n_users_threshold=feature_cfg.get("wrds_large_firm_n_users_threshold", 75_000),
            large_firm_year_span=int(feature_cfg.get("wrds_large_firm_year_span", 1)),
            query_timeout_ms=int(float(feature_cfg.get("query_timeout_minutes", 10)) * 60_000),
            singleton_query_timeout_ms=(
                int(float(feature_cfg.get("wrds_singleton_query_timeout_minutes")) * 60_000)
                if feature_cfg.get("wrds_singleton_query_timeout_minutes") is not None
                else None
            ),
            query_max_retries=int(feature_cfg.get("query_max_retries", 1)),
            min_position_days=int(feature_cfg.get("min_position_days", 365)),
            tenure_min_days=int(feature_cfg.get("tenure_min_days", 365)),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    meta = expected_meta | {
        "n_rows": int(len(df)),
        "n_firms": int(df["c"].nunique()) if "c" in df.columns else 0,
    }
    _write_metadata(out_path, meta)
    _log(
        f"[wrds_school_flows_cache] DONE: wrote {out_path} | rows {len(df):,} | "
        f"elapsed {_format_elapsed(time.time() - cache_t0)}"
    )
    return df, meta


def _default_design3_position_outcomes_path(cfg: dict) -> Path:
    paths_cfg = get_cfg_section(cfg, "paths")
    configured = paths_cfg.get("design3_position_outcomes_out")
    if configured:
        root = str(Path(__file__).resolve().parents[2])
        return Path(str(configured).replace("{root}", root))
    workforce_path = _resolve_path(paths_cfg, "wrds_company_year_workforce_out", allow_missing=True)
    return workforce_path.with_name("design3_position_outcomes.parquet")


def build_design3_position_outcomes_from_local_caches(
    *,
    selected_positions_path: Path,
    position_history_path: Path | str | None = None,
    users_path: Path,
    user_profile_path: Path,
    year_min: int,
    year_max: int,
    firm_ids: Optional[pd.Series | pd.DataFrame | list[int]] = None,
    opt_likely_soc2: Optional[list[str] | tuple[str, ...] | str] = None,
    top_n_soc2: int = 4,
    intern_max_days: int = 183,
) -> tuple[pd.DataFrame, dict]:
    """Build Design 3 firm-year outcomes from local WRDS/Revelio cache files."""
    selected_positions_path = Path(selected_positions_path)
    position_history_path = _resolve_cfg_path_value(position_history_path)
    users_path = Path(users_path)
    user_profile_path = Path(user_profile_path)
    if not selected_positions_path.exists():
        raise FileNotFoundError(f"Missing selected positions cache: {selected_positions_path}")
    if not users_path.exists():
        raise FileNotFoundError(f"Missing WRDS users cache: {users_path}")
    if not user_profile_path.exists():
        raise FileNotFoundError(f"Missing user-profile origin cache: {user_profile_path}")
    fixed_opt_likely_soc2 = _normalize_design3_opt_likely_soc2(opt_likely_soc2)

    if isinstance(firm_ids, pd.DataFrame):
        selected = _clean_selected_firm_ids(firm_ids)
    elif isinstance(firm_ids, pd.Series):
        selected = _clean_selected_firm_ids(pd.DataFrame({"c": firm_ids}))
    elif firm_ids is None:
        selected = pd.DataFrame(columns=["c"])
    else:
        selected = _clean_selected_firm_ids(pd.DataFrame({"c": firm_ids}))

    con = ddb.connect()
    try:
        profile_cols = set(
            con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{_escape(user_profile_path)}') LIMIT 1"
            ).df()["column_name"].astype(str)
        )
        signal_current_country_ref = (
            "COALESCE(TRY_CAST(signal_current_country_nonus AS DOUBLE), NULL)"
            if "signal_current_country_nonus" in profile_cols
            else "NULL::DOUBLE"
        )
        if "likely_foreign_hard" in profile_cols:
            likely_foreign_hard_expr = "COALESCE(TRY_CAST(likely_foreign_hard AS INTEGER), 0)"
        else:
            likely_foreign_hard_expr = _sql_new_hire_origin_hard_expr(
                "COALESCE(TRY_CAST(p_likely_foreign AS DOUBLE), 0.0)",
                signal_current_country_ref,
            )
        if not selected.empty:
            con.register("selected_firms_df", selected)
            firm_join = "JOIN selected_firms sf ON CAST(p.rcid AS BIGINT) = sf.c"
            con.sql(
                "CREATE OR REPLACE TEMP VIEW selected_firms AS "
                "SELECT CAST(c AS BIGINT) AS c FROM selected_firms_df"
            )
        else:
            firm_join = ""
        if fixed_opt_likely_soc2:
            values_sql = ", ".join(f"('{soc2}')" for soc2 in fixed_opt_likely_soc2)
            opt_like_soc2_cte = f"""
        opt_like_soc2 AS MATERIALIZED (
            SELECT CAST(soc2 AS VARCHAR) AS soc2
            FROM (VALUES {values_sql}) AS v(soc2)
        ),
            """
            opt_likely_source = "fixed_config"
        else:
            opt_like_soc2_cte = f"""
        opt_like_soc2 AS MATERIALIZED (
            SELECT soc2
            FROM (
                SELECT
                    soc2,
                    COUNT(DISTINCT user_id) AS n_users,
                    ROW_NUMBER() OVER (ORDER BY COUNT(DISTINCT user_id) DESC, soc2 ASC) AS rn
                FROM new_grad_hires
                WHERE soc2 IS NOT NULL
                  AND first_start <= grad_date + INTERVAL '365 days'
                GROUP BY 1
            )
            WHERE rn <= {int(top_n_soc2)}
        ),
            """
            opt_likely_source = "top_n_recent_grad"

        base_ctes = f"""
        WITH pos AS MATERIALIZED (
            SELECT
                CAST(p.rcid AS BIGINT) AS c,
                CAST(p.user_id AS BIGINT) AS user_id,
                CAST(p.position_id AS BIGINT) AS position_id,
                TRY_CAST(p.startdate AS DATE) AS startdate,
                COALESCE(TRY_CAST(p.enddate AS DATE), DATE '2026-12-31') AS enddate,
                CASE
                    WHEN p.onet_code IS NULL OR TRIM(CAST(p.onet_code AS VARCHAR)) = '' THEN NULL
                    ELSE SUBSTRING(CAST(p.onet_code AS VARCHAR), 1, 2)
                END AS soc2
            FROM read_parquet('{_escape(selected_positions_path)}') AS p
            {firm_join}
            WHERE p.user_id IS NOT NULL
              AND p.rcid IS NOT NULL
              AND TRY_CAST(p.startdate AS DATE) IS NOT NULL
	              AND EXTRACT(YEAR FROM TRY_CAST(p.startdate AS DATE))::INT <= {int(year_max)}
	              AND EXTRACT(YEAR FROM COALESCE(TRY_CAST(p.enddate AS DATE), DATE '2026-12-31'))::INT >= {int(year_min)}
	        ),
	        profile AS MATERIALIZED (
	            SELECT
	                CAST(user_id AS BIGINT) AS user_id,
	                COALESCE(TRY_CAST(p_likely_foreign AS DOUBLE), 0.0) AS p_likely_foreign,
	                CAST({likely_foreign_hard_expr} AS INTEGER) AS likely_foreign_hard
	            FROM read_parquet('{_escape(user_profile_path)}')
	            WHERE user_id IS NOT NULL
	        ),
	        education_base AS MATERIALIZED (
	            SELECT
	                CAST(user_id AS BIGINT) AS user_id,
	                TRY_CAST(ed_startdate AS DATE) AS education_start_date,
	                TRY_CAST(ed_enddate AS DATE) AS grad_date,
	                NULLIF(TRIM(CAST(degree AS VARCHAR)), '') AS degree,
	                COALESCE(NULLIF(TRIM(CAST(degree_raw AS VARCHAR)), ''), '') AS degree_raw,
	                COALESCE(NULLIF(TRIM(CAST(field_raw AS VARCHAR)), ''), '') AS field_raw,
	                COALESCE(NULLIF(TRIM(CAST(university_raw AS VARCHAR)), ''), '') AS university_raw
	            FROM read_parquet('{_escape(users_path)}')
	            WHERE user_id IS NOT NULL
	              AND TRY_CAST(ed_enddate AS DATE) IS NOT NULL
	        ),
	        education_clean AS MATERIALIZED (
	            SELECT
	                user_id,
	                education_start_date,
	                grad_date,
	                COALESCE({degree_clean_regex_sql()}, 'Missing') AS degree_clean
	            FROM education_base
	        ),
	        education_ranked AS MATERIALIZED (
	            SELECT *
	            FROM (
	                SELECT
	                    user_id,
	                    education_start_date,
	                    grad_date,
	                    degree_clean,
	                    CASE WHEN degree_clean = 'Master' THEN 1 ELSE 0 END AS is_masters,
	                    ROW_NUMBER() OVER (
	                        PARTITION BY user_id
	                        ORDER BY grad_date DESC NULLS LAST,
	                                 education_start_date ASC NULLS LAST
	                    ) AS rn
	                FROM education_clean
	            )
	            WHERE rn = 1
	        ),
	        grad AS MATERIALIZED (
	            SELECT user_id, education_start_date, grad_date, degree_clean, is_masters
	            FROM education_ranked
	            WHERE grad_date IS NOT NULL
	        ),
	        first_position AS MATERIALIZED (
	            SELECT *
            FROM (
                SELECT
                    p.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.c, p.user_id
                        ORDER BY p.startdate ASC, p.position_id ASC NULLS LAST
                    ) AS rn
                FROM pos p
            )
            WHERE rn = 1
        ),
        company_user_bounds AS MATERIALIZED (
            SELECT
                p.c,
                p.user_id,
                MIN(p.startdate) AS first_start,
                MAX(p.enddate) AS last_end
	            FROM pos p
	            GROUP BY 1, 2
	        ),
	        new_grad_hires AS MATERIALIZED (
	            SELECT
	                fp.c,
	                fp.user_id,
                fp.position_id,
	                fp.startdate AS first_start,
	                b.last_end,
	                fp.soc2,
	                EXTRACT(YEAR FROM fp.startdate)::INTEGER AS t,
	                COALESCE(pr.p_likely_foreign, 0.0) AS p_likely_foreign,
	                COALESCE(pr.likely_foreign_hard, 0) AS likely_foreign_hard,
	                COALESCE(g.is_masters, 0) AS is_masters,
	                g.education_start_date,
	                g.grad_date
	            FROM first_position fp
	            JOIN company_user_bounds b
	              ON b.c = fp.c
	             AND b.user_id = fp.user_id
	            JOIN grad g
	              ON g.user_id = fp.user_id
	            LEFT JOIN profile pr
	              ON pr.user_id = fp.user_id
	            WHERE EXTRACT(YEAR FROM fp.startdate)::INT BETWEEN {int(year_min)} AND {int(year_max)}
	              AND fp.startdate >= g.grad_date
	              AND fp.startdate <= g.grad_date + INTERVAL '365 days'
	        ),
	        {opt_like_soc2_cte}
	        new_grad_hire_outcomes AS (
	            SELECT
	                c,
	                t,
	                SUM(COALESCE(likely_foreign_hard, 0))::DOUBLE AS y_new_hires_foreign_lag0,
	                SUM(1 - COALESCE(likely_foreign_hard, 0))::DOUBLE AS y_new_hires_native_lag0,
	                SUM(CASE WHEN soc2 IN (SELECT soc2 FROM opt_like_soc2) THEN COALESCE(likely_foreign_hard, 0) ELSE 0 END)::DOUBLE
	                    AS y_new_hires_foreign_opt_likely_lag0,
	                SUM(CASE WHEN is_masters = 1 THEN COALESCE(likely_foreign_hard, 0) ELSE 0 END)::DOUBLE
	                    AS y_new_hires_foreign_masters_lag0,
	                SUM(CASE WHEN is_masters = 1 THEN 1 - COALESCE(likely_foreign_hard, 0) ELSE 0 END)::DOUBLE
	                    AS y_new_hires_native_masters_lag0,
	                SUM(CASE
	                        WHEN is_masters = 1 AND soc2 IN (SELECT soc2 FROM opt_like_soc2) THEN COALESCE(likely_foreign_hard, 0)
	                        ELSE 0
	                    END)::DOUBLE AS y_new_hires_foreign_opt_likely_masters_lag0,
	                AVG(GREATEST(0.0, (last_end - first_start)::DOUBLE / 365.25))
	                    AS avg_tenure_new_hires_lag0,
	                AVG(
	                    CASE
	                        WHEN is_masters = 1 THEN
	                            GREATEST(0.0, (last_end - first_start)::DOUBLE / 365.25)
	                        ELSE NULL
	                    END
	                ) AS avg_tenure_new_hires_masters_lag0,
	                AVG(
	                    CASE
	                        WHEN COALESCE(likely_foreign_hard, 0) = 1 THEN
	                            GREATEST(0.0, (last_end - first_start)::DOUBLE / 365.25)
	                        ELSE NULL
	                    END
	                ) AS avg_tenure_foreign_new_hires_lag0
	                ,
	                AVG(
	                    CASE
	                        WHEN is_masters = 1 AND COALESCE(likely_foreign_hard, 0) = 1 THEN
	                            GREATEST(0.0, (last_end - first_start)::DOUBLE / 365.25)
	                        ELSE NULL
	                    END
	                ) AS avg_tenure_foreign_new_hires_masters_lag0
	            FROM new_grad_hires
	            GROUP BY 1, 2
	        ),
        active_user_year AS MATERIALIZED (
            SELECT
                p.c,
                gs.year::INTEGER AS t,
                p.user_id,
                MAX(CASE WHEN p.soc2 IN (SELECT soc2 FROM opt_like_soc2) THEN 1 ELSE 0 END) AS opt_likely_job,
                MAX(b.first_start) AS first_start
            FROM pos p
            JOIN company_user_bounds b
              ON b.c = p.c
             AND b.user_id = p.user_id
            JOIN LATERAL generate_series(
                GREATEST(EXTRACT(YEAR FROM p.startdate)::INT, {int(year_min)}),
                LEAST(EXTRACT(YEAR FROM p.enddate)::INT, {int(year_max)})
            ) AS gs(year) ON TRUE
            GROUP BY 1, 2, 3
        ),
        opt_likely_tenure AS (
            SELECT
                c,
                t,
                AVG(GREATEST(0.0, (make_date(t, 12, 31) - first_start)::DOUBLE / 365.25))
                    AS avg_tenure_opt_likely_jobs_lag0
            FROM active_user_year
            WHERE opt_likely_job = 1
            GROUP BY 1, 2
        ),
        intern_user_year AS MATERIALIZED (
            SELECT
                p.c,
                EXTRACT(YEAR FROM p.startdate)::INTEGER AS t,
                p.user_id,
                MAX(CASE WHEN p.soc2 IN (SELECT soc2 FROM opt_like_soc2) THEN 1 ELSE 0 END) AS opt_likely_intern,
                MAX(COALESCE(pr.likely_foreign_hard, 0)) AS likely_foreign_hard
            FROM pos p
            JOIN grad g
              ON g.user_id = p.user_id
            LEFT JOIN profile pr
              ON pr.user_id = p.user_id
            WHERE p.enddate < p.startdate + INTERVAL '{int(intern_max_days)} days'
              AND p.startdate < g.grad_date
              AND g.education_start_date IS NOT NULL
              AND p.startdate >= g.education_start_date
              AND EXTRACT(YEAR FROM p.startdate)::INT BETWEEN {int(year_min)} AND {int(year_max)}
            GROUP BY 1, 2, 3
        ),
        intern_outcomes AS (
            SELECT
                c,
                t,
                SUM(opt_likely_intern)::DOUBLE AS y_intern_positions_opt_likely_lag0,
                SUM(likely_foreign_hard)::DOUBLE AS y_intern_positions_foreign_lag0,
                SUM(CASE WHEN opt_likely_intern = 1 THEN likely_foreign_hard ELSE 0 END)::DOUBLE
                    AS y_intern_positions_opt_likely_foreign_lag0
            FROM intern_user_year
            GROUP BY 1, 2
        )
        """
        out = con.sql(
            base_ctes
            + """
            SELECT
                COALESCE(nh.c, ot.c, i.c) AS c,
                COALESCE(nh.t, ot.t, i.t) AS t,
                nh.y_new_hires_foreign_lag0,
                nh.y_new_hires_native_lag0,
                nh.y_new_hires_foreign_opt_likely_lag0,
                nh.y_new_hires_foreign_masters_lag0,
                nh.y_new_hires_native_masters_lag0,
                nh.y_new_hires_foreign_opt_likely_masters_lag0,
                ot.avg_tenure_opt_likely_jobs_lag0,
                nh.avg_tenure_foreign_new_hires_lag0,
                nh.avg_tenure_new_hires_lag0,
                nh.avg_tenure_foreign_new_hires_masters_lag0,
                nh.avg_tenure_new_hires_masters_lag0,
                i.y_intern_positions_opt_likely_lag0,
                i.y_intern_positions_foreign_lag0,
                i.y_intern_positions_opt_likely_foreign_lag0,
                (SELECT string_agg(soc2, ',' ORDER BY soc2) FROM opt_like_soc2) AS _opt_likely_soc2_csv
            FROM new_grad_hire_outcomes nh
            FULL OUTER JOIN opt_likely_tenure ot
              ON ot.c = nh.c
             AND ot.t = nh.t
            FULL OUTER JOIN intern_outcomes i
              ON i.c = COALESCE(nh.c, ot.c)
             AND i.t = COALESCE(nh.t, ot.t)
            ORDER BY 1, 2
            """
        ).df()
    finally:
        con.close()

    if "_opt_likely_soc2_csv" in out.columns and out["_opt_likely_soc2_csv"].notna().any():
        top_soc2_values = [
            v
            for v in str(out["_opt_likely_soc2_csv"].dropna().iloc[0]).split(",")
            if v
        ]
        out = out.drop(columns=["_opt_likely_soc2_csv"])
    else:
        top_soc2_values = []
    if not out.empty:
        out["c"] = pd.to_numeric(out["c"], errors="coerce").astype("Int64")
        out["t"] = pd.to_numeric(out["t"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["c", "t"]).copy()
        out["c"] = out["c"].astype(int)
        out["t"] = out["t"].astype(int)
    meta = {
        "design3_position_outcome_method": DESIGN3_POSITION_OUTCOME_METHOD,
        "year_min": int(year_min),
        "year_max": int(year_max),
        "top_n_soc2": int(top_n_soc2),
        "opt_likely_soc2": top_soc2_values,
        "opt_likely_soc2_source": opt_likely_source,
        "intern_max_days": int(intern_max_days),
        "selected_positions_path": str(selected_positions_path),
        "position_history_path": str(position_history_path) if position_history_path is not None else None,
        "users_path": str(users_path),
        "user_profile_path": str(user_profile_path),
        "n_rows": int(len(out)),
    }
    return out.reset_index(drop=True), meta


def load_or_build_design3_position_outcomes_cache(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    year_min: int,
    year_max: int,
    force_rebuild: bool = False,
    firm_ids: Optional[pd.Series | pd.DataFrame | list[int]] = None,
    position_history_path: Path | str | None = None,
    opt_likely_soc2: Optional[list[str] | tuple[str, ...] | str] = None,
    intern_max_days: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths_cfg = get_cfg_section(cfg_full, "paths")
    out_path = _default_design3_position_outcomes_path(cfg_full)
    selected_positions_path = _resolve_path(paths_cfg, "wrds_workforce_selected_us_positions_out")
    users_path = _resolve_path(paths_cfg, "wrds_workforce_users_out")
    user_profile_path = _resolve_path(paths_cfg, "wrds_user_profile_origin_out")
    feature_cfg = get_cfg_section(cfg_full, "revelio_company_features")
    configured_history_path = (
        _resolve_cfg_path_value(position_history_path)
        or _resolve_optional_existing_path(paths_cfg, "design3_position_history_path")
    )
    top_n_soc2 = int(feature_cfg.get("design3_opt_likely_top_n_soc2", 4))
    if opt_likely_soc2 is None:
        opt_likely_soc2 = feature_cfg.get("design3_opt_likely_soc2")
    fixed_opt_likely_soc2 = _normalize_design3_opt_likely_soc2(opt_likely_soc2)
    if intern_max_days is None:
        intern_max_days = int(feature_cfg.get("design3_intern_max_days", 183))
    else:
        intern_max_days = int(intern_max_days)
    selected_hash = None
    if firm_ids is not None:
        if isinstance(firm_ids, pd.DataFrame):
            selected_hash = _selected_firms_hash(firm_ids)
        elif isinstance(firm_ids, pd.Series):
            selected_hash = _selected_firms_hash(pd.DataFrame({"c": firm_ids}))
        else:
            selected_hash = _selected_firms_hash(pd.DataFrame({"c": firm_ids}))
    expected_meta = {
        "design3_position_outcome_method": DESIGN3_POSITION_OUTCOME_METHOD,
        "year_min": int(year_min),
        "year_max": int(year_max),
        "top_n_soc2": int(top_n_soc2),
        "opt_likely_soc2": fixed_opt_likely_soc2,
        "opt_likely_soc2_source": "fixed_config" if fixed_opt_likely_soc2 else "top_n_recent_grad",
        "intern_max_days": int(intern_max_days),
        "selected_positions_path": str(selected_positions_path),
        "position_history_path": str(configured_history_path) if configured_history_path is not None else None,
        "users_path": str(users_path),
        "user_profile_path": str(user_profile_path),
        "selected_firms_hash": selected_hash,
    }
    if out_path.exists() and not force_rebuild:
        meta = _load_metadata(out_path)
        if _metadata_compatible(meta, expected_meta):
            cached = pd.read_parquet(out_path)
            cached["t"] = pd.to_numeric(cached["t"], errors="coerce")
            cached = cached[cached["t"].between(int(year_min), int(year_max))].copy()
            _log(f"[design3_position_outcomes] REUSE: {out_path} | rows {len(cached):,}")
            return cached.reset_index(drop=True), meta or {}

    _log(f"[design3_position_outcomes] BUILD: {out_path}")
    out, meta = build_design3_position_outcomes_from_local_caches(
        selected_positions_path=selected_positions_path,
        position_history_path=configured_history_path,
        users_path=users_path,
        user_profile_path=user_profile_path,
        year_min=int(year_min),
        year_max=int(year_max),
        firm_ids=firm_ids,
        opt_likely_soc2=fixed_opt_likely_soc2,
        top_n_soc2=top_n_soc2,
        intern_max_days=intern_max_days,
    )
    meta = {**expected_meta, **meta}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    _write_metadata(out_path, meta)
    _log(f"[design3_position_outcomes] DONE: wrote {out_path} | rows {len(out):,}")
    return out, meta


def build_source_analysis_panel(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    data_min_t: int = 2010,
    data_max_t: int = 2022,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = resolve_source_exposure_paths(cfg_full)
    out_path = paths.opt_exposure_analysis_panel_out
    firms, _, _, universe_meta = load_or_build_source_firm_universe(
        cfg=cfg_full,
        force_rebuild=force_rebuild,
    )
    metadata = {
        "data_min_t": int(data_min_t),
        "data_max_t": int(data_max_t),
        "source_analysis_panel_method": SOURCE_ANALYSIS_PANEL_METHOD,
        "opt_count_method": OPT_COUNT_METHOD,
        "degree_groups": list(DEGREE_GROUPS),
        "new_hire_origin_method": NEW_HIRE_ORIGIN_METHOD,
        **universe_meta,
    }
    if out_path.exists() and not force_rebuild:
        current_meta = _load_metadata(out_path)
        if _metadata_compatible(current_meta, metadata):
            return pd.read_parquet(out_path)

    counts, _ = load_or_build_source_opt_counts(
        cfg=cfg_full,
        year_min=data_min_t,
        year_max=data_max_t,
        force_rebuild=force_rebuild,
    )
    workforce, workforce_meta = load_or_build_wrds_company_year_workforce_cache(
        cfg=cfg_full,
        year_min=data_min_t,
        year_max=data_max_t,
        force_rebuild=force_rebuild,
    )

    years = pd.DataFrame({"t": np.arange(int(data_min_t), int(data_max_t) + 1, dtype=int)})
    firms = firms.copy()
    firms["key"] = 1
    years["key"] = 1
    panel = firms.merge(years, on="key", how="inner").drop(columns=["key"])
    panel = panel.merge(
        counts[["c", "t"] + OPT_COUNT_COLUMNS],
        on=["c", "t"],
        how="left",
    )
    workforce_cols = [
        col
        for col in [
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
            "avg_tenure_years_annual",
        ]
        if col in workforce.columns
    ]
    panel = panel.merge(
        workforce[workforce_cols],
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
    if "avg_tenure_years_annual" in panel.columns:
        panel["avg_tenure_years_lag0"] = pd.to_numeric(
            panel["avg_tenure_years_annual"], errors="coerce"
        )
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
            "avg_tenure_years_annual",
        ],
        errors="ignore",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    _write_metadata(
        out_path,
        metadata
        | {
            "n_rows": int(len(panel)),
            "n_firms": int(panel["c"].nunique()),
            "workforce_cache_year_min": workforce_meta.get("year_min"),
            "workforce_cache_year_max": workforce_meta.get("year_max"),
        },
    )
    print(f"[source_analysis_panel] Wrote {out_path}")
    return panel


def load_or_build_source_analysis_panel(
    config_path: str | Path | None = None,
    *,
    cfg: Optional[dict] = None,
    data_min_t: int = 2010,
    data_max_t: int = 2022,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    paths = resolve_source_exposure_paths(cfg_full)
    _, _, _, universe_meta = load_or_build_source_firm_universe(
        cfg=cfg_full,
        force_rebuild=force_rebuild,
    )
    expected_meta = {
        "data_min_t": int(data_min_t),
        "data_max_t": int(data_max_t),
        "source_analysis_panel_method": SOURCE_ANALYSIS_PANEL_METHOD,
        "opt_count_method": OPT_COUNT_METHOD,
        "degree_groups": list(DEGREE_GROUPS),
        "new_hire_origin_method": NEW_HIRE_ORIGIN_METHOD,
        **universe_meta,
    }
    if paths.opt_exposure_analysis_panel_out.exists() and not force_rebuild:
        meta = _load_metadata(paths.opt_exposure_analysis_panel_out)
        if _metadata_compatible(meta, expected_meta):
            return pd.read_parquet(paths.opt_exposure_analysis_panel_out), meta
    df = build_source_analysis_panel(
        config_path=config_path,
        cfg=cfg_full,
        data_min_t=data_min_t,
        data_max_t=data_max_t,
        force_rebuild=force_rebuild,
    )
    return df, _load_metadata(paths.opt_exposure_analysis_panel_out)
