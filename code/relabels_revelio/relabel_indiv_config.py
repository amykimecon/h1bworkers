"""YAML-backed settings for the relabels_revelio analysis pipeline."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import root  # noqa: E402

DEFAULT_CONFIG_PATH = Path(root) / "h1bworkers" / "code" / "configs" / "relabel_indiv.yaml"
REPO_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "relabel_indiv.yaml"
ENV_CONFIG_PATH = os.environ.get("RELABEL_INDIV_CONFIG")
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
RUN_TAG = str(CFG.get("run_tag", "mar2026"))
BUILD_CFG = CFG.get("build", {})
TESTING_CFG = CFG.get("testing", {})


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


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_lower_str(value: Any, default: str) -> str:
    if value is None:
        return default
    as_str = str(value).strip().lower()
    return as_str or default


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


def choose_path(primary: str, fallback: str | None = None) -> str:
    """Return existing primary path, else fallback (if given), else primary."""
    if os.path.exists(primary):
        return primary
    if fallback is not None:
        return fallback
    return primary


# --- INPUT PATHS ---
RELABELS_PARQUET = PATHS.get("relabels_parquet", "")
REVELIO_IPEDS_INST_CW_PARQUET = PATHS.get("revelio_ipeds_inst_cw_parquet", "")
STAGE04_MERGE_READY_PARQUET = PATHS.get("stage04_merge_ready_parquet", "")
STAGE05_PERSON_BASELINE_PARQUET = PATHS.get("stage05_person_baseline_parquet", "")
# Blended nationality (name model + institution country) for ALL Revelio users.
# Broader than rev_indiv (not filtered to OPT/H-1B pipeline).
# Columns: user_id, nationality_country, nationality_score.
REV_USER_NATIONALITY_PARQUET = choose_path(
    PATHS.get("rev_user_nationality_parquet", ""),
    PATHS.get("rev_user_nationality_parquet_legacy"),
)
REV_EDUC_LONG_PARQUET = choose_path(
    PATHS.get("rev_educ_long_parquet", ""),
    PATHS.get("rev_educ_long_parquet_legacy"),
)
REV_POS_PARQUET = choose_path(
    PATHS.get("rev_pos_parquet", ""),
    PATHS.get("rev_pos_parquet_legacy"),
)

# --- OUTPUT PATHS ---
OUTPUT_PANEL_PARQUET = PATHS.get("output_panel_parquet", "")
OUTPUT_DID_RESULTS_PARQUET = PATHS.get("output_did_results_parquet", "")
OUTPUT_DIR = PATHS.get("output_dir", "")

# --- BUILD PARAMS ---
BUILD_OVERWRITE = _as_bool(BUILD_CFG.get("overwrite"), True)
BUILD_EVENT_WINDOW = _as_int(BUILD_CFG.get("event_window"), 5)
BUILD_COHORT_WINDOW = _as_int(BUILD_CFG.get("cohort_window"), 3)
BUILD_OUTCOME_HORIZONS = sorted(
    {h for h in _as_int_list(BUILD_CFG.get("outcome_horizons"), [3]) if h >= 0}
)
BUILD_CAP_TO_LATEST_AVAILABLE_YEAR = _as_bool(
    BUILD_CFG.get("cap_to_latest_available_year"),
    True,
)
BUILD_FIELD_MATCH_PATTERNS: list[str] = BUILD_CFG.get("field_match_patterns", ["%econom%"])
BUILD_SAMPLE_GRADYEAR_WINDOW = _as_int(BUILD_CFG.get("sample_gradyear_window"), 5)
BUILD_SAMPLE_CIP_PREFIXES: list[str] = [
    str(value).strip()
    for value in BUILD_CFG.get("sample_cip_prefixes", ["4506"])
    if str(value).strip()
]
BUILD_SAMPLE_VARIANTS: list[str] = [
    str(value).strip()
    for value in BUILD_CFG.get(
        "sample_variants",
        ["stage04_all", "foia_linked_person_baseline"],
    )
    if str(value).strip()
]
BUILD_MIN_POS_DURATION_DAYS = _as_int(BUILD_CFG.get("min_pos_duration_days"), 1)
BUILD_RUN_DID = _as_bool(BUILD_CFG.get("run_did"), True)
BUILD_DID_MODEL = _as_lower_str(BUILD_CFG.get("did_model"), "simple")
BUILD_EXCLUDE_US_NATIONALS = _as_bool(BUILD_CFG.get("exclude_us_nationals"), True)
BUILD_EXCLUDE_US_COUNTRY_VALUE = _as_optional_str(BUILD_CFG.get("exclude_us_country_value")) or "United States"

# --- TESTING PARAMS ---
TESTING_ENABLED = _as_bool(TESTING_CFG.get("enabled"), False)
TESTING_SAMPLE_N_INSTITUTIONS = max(1, _as_int(TESTING_CFG.get("sample_n_institutions"), 5))
TESTING_RANDOM_SEED = _as_optional_int(TESTING_CFG.get("random_seed"))
