"""YAML-backed settings for the standalone relabel_indiv_model WRDS extract."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import root  # noqa: E402

DEFAULT_CONFIG_PATH = Path(root) / "h1bworkers" / "code" / "configs" / "relabel_indiv_model.yaml"
REPO_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "relabel_indiv_model.yaml"
ENV_CONFIG_PATH = os.environ.get("RELABEL_INDIV_MODEL_CONFIG")
_VAR_PATTERN = re.compile(r"\$(\w+)|\$\{([^}]+)\}")
ACTIVE_CONFIG_PATH: str | None = None


def _replace_known_vars(value: str) -> str:
    return value.replace("{root}", str(root))


def _lookup_var(data: dict[str, Any], key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _interpolate(value: str, data: dict[str, Any]) -> str:
    def _sub(match: re.Match) -> str:
        key = match.group(1) or match.group(2)
        if not key:
            return match.group(0)
        resolved = _lookup_var(data, key)
        return str(resolved) if resolved is not None else match.group(0)

    return _VAR_PATTERN.sub(_sub, value)


def _walk_and_replace(obj: Any, data: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_and_replace(v, data) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_replace(v, data) for v in obj]
    if isinstance(obj, str):
        expanded = os.path.expanduser(_replace_known_vars(obj))
        return _interpolate(expanded, data)
    return obj


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    global ACTIVE_CONFIG_PATH

    cfg_path = (
        Path(path)
        if path
        else (Path(ENV_CONFIG_PATH) if ENV_CONFIG_PATH else DEFAULT_CONFIG_PATH)
    )
    if not cfg_path.exists() and path is None and not ENV_CONFIG_PATH and REPO_CONFIG_PATH.exists():
        cfg_path = REPO_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    ACTIVE_CONFIG_PATH = str(cfg_path.resolve())
    data = yaml.safe_load(cfg_path.read_text()) or {}
    data = _walk_and_replace(data, data)
    return _walk_and_replace(data, data)


CFG = load_config()
PATHS = CFG.get("paths", {})
BUILD_CFG = CFG.get("build", {})
ANALYSIS_CFG = CFG.get("analysis", {})
WRDS_CFG = CFG.get("wrds", {})
TESTING_CFG = CFG.get("testing", {})
RUN_TAG = str(CFG.get("run_tag", "apr2026"))


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    as_str = str(value).strip()
    if as_str.lower() in {"", "none", "null"}:
        return None
    return as_str


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    out: list[str] = []
    for item in values:
        item_str = _as_optional_str(item)
        if item_str is not None:
            out.append(item_str)
    return out


def _as_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        values = default
    elif isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    out: list[int] = []
    for item in values:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out or default


def _as_str_int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, item in value.items():
        key_str = _as_optional_str(key)
        if key_str is None:
            continue
        try:
            out[key_str] = int(item)
        except (TypeError, ValueError):
            continue
    return out


RELABELS_PARQUET = PATHS.get("relabels_parquet", "")
IPEDS_CROSSWALK_PARQUET = PATHS.get("ipeds_crosswalk_parquet", "")
IPEDS_COMPLETIONS_PARQUET = PATHS.get("ipeds_completions_parquet", "")

EVENT_SCHOOLS_PARQUET = PATHS.get("event_schools_parquet", "")
RSID_CANDIDATES_PARQUET = PATHS.get("rsid_candidates_parquet", "")
SELECTED_RSID_PARQUET = PATHS.get("selected_rsid_parquet", "")
MATCHED_EDUCATION_PARQUET = PATHS.get("matched_education_parquet", "")
MATCHED_POSITIONS_PARQUET = PATHS.get("matched_positions_parquet", "")

NEVER_TREATED_SCHOOLS_PARQUET = PATHS.get("never_treated_schools_parquet", "")
NEVER_TREATED_RSID_CANDIDATES_PARQUET = PATHS.get("never_treated_rsid_candidates_parquet", "")
NEVER_TREATED_SELECTED_RSID_PARQUET = PATHS.get("never_treated_selected_rsid_parquet", "")
NEVER_TREATED_EDUCATION_PARQUET = PATHS.get("never_treated_education_parquet", "")
NEVER_TREATED_POSITIONS_PARQUET = PATHS.get("never_treated_positions_parquet", "")
ANALYSIS_PANEL_PARQUET = PATHS.get("analysis_panel_parquet", "")
ANALYSIS_SCHOOL_EVENT_TIME_PARQUET = PATHS.get("analysis_school_event_time_parquet", "")
ANALYSIS_EVENT_STUDY_PARQUET = PATHS.get("analysis_event_study_parquet", "")
ANALYSIS_REGRESSION_EVENT_STUDY_PARQUET = PATHS.get("analysis_regression_event_study_parquet", "")
ANALYSIS_OUTPUT_DIR = PATHS.get("analysis_output_dir", "")

BUILD_OVERWRITE = _as_bool(BUILD_CFG.get("overwrite"), True)
BUILD_CONTROL_ONLY = _as_bool(BUILD_CFG.get("control_only"), False)
BUILD_REUSE_EXISTING_CANDIDATE_PARQUETS = _as_bool(
    BUILD_CFG.get("reuse_existing_candidate_parquets"),
    False,
)
BUILD_POSITION_CHUNK_SIZE = max(1, _as_int(BUILD_CFG.get("position_chunk_size"), 10000))
BUILD_RSID_LOOKUP_LIMIT = max(1, _as_int(BUILD_CFG.get("rsid_lookup_limit"), 10000))
BUILD_EDUCATION_EVENT_WINDOW_YEARS = max(0, _as_int(BUILD_CFG.get("education_event_window_years"), 5))
BUILD_MANUAL_RSID_OVERRIDES = _as_str_int_dict(BUILD_CFG.get("manual_rsid_overrides"))
BUILD_CONTROL_YEAR_MIN = _as_int(BUILD_CFG.get("control_year_min"), 2010)
BUILD_CONTROL_YEAR_MAX = _as_int(BUILD_CFG.get("control_year_max"), 2024)
ANALYSIS_HORIZONS = sorted({h for h in _as_int_list(ANALYSIS_CFG.get("horizons"), [1, 3, 5, 10]) if h >= 0})
ANALYSIS_CAP_TO_LATEST_AVAILABLE_YEAR = _as_bool(ANALYSIS_CFG.get("cap_to_latest_available_year"), True)
ANALYSIS_REGRESSION_REFERENCE_EVENT_TIME = _as_int(ANALYSIS_CFG.get("regression_reference_event_time"), -1)
ANALYSIS_FIELD_FILTER = _as_optional_str(ANALYSIS_CFG.get("field_filter"))

WRDS_USERNAME = _as_optional_str(WRDS_CFG.get("username"))
WRDS_PGPASS_PATH = _as_optional_str(WRDS_CFG.get("pgpass_path"))

TESTING_ENABLED = _as_bool(TESTING_CFG.get("enabled"), False)
TESTING_SAMPLE_N_SCHOOLS = max(1, _as_int(TESTING_CFG.get("sample_n_schools"), 3))
TESTING_RANDOM_SEED = _as_optional_int(TESTING_CFG.get("random_seed"))
TESTING_SCHOOL_NAMES = _as_str_list(TESTING_CFG.get("school_names"))
