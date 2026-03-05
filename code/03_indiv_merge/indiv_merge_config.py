"""YAML-backed settings for 03_indiv_merge scripts."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import root  # noqa: E402

DEFAULT_CONFIG_PATH = Path(root) / "h1bworkers" / "code" / "configs" / "indiv_merge.yaml"
REPO_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "indiv_merge.yaml"
ENV_CONFIG_PATH = os.environ.get("INDIV_MERGE_CONFIG")
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
RUN_TAG = str(CFG.get("run_tag", "feb2026"))
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

FOIA_INDIV_PARQUET = PATHS["foia_indiv_parquet"]
FOIA_INDIV_PARQUET_LEGACY = PATHS["foia_indiv_parquet_legacy"]
REV_INDIV_PARQUET = PATHS["rev_indiv_parquet"]
REV_INDIV_PARQUET_LEGACY = PATHS["rev_indiv_parquet_legacy"]
REV_EDUC_LONG_PARQUET = PATHS["rev_educ_long_parquet"]
REV_EDUC_LONG_PARQUET_LEGACY = PATHS["rev_educ_long_parquet_legacy"]
MERGED_POS_CLEAN_PARQUET = PATHS["merged_pos_clean_parquet"]
MERGED_POS_CLEAN_PARQUET_LEGACY = PATHS["merged_pos_clean_parquet_legacy"]

MERGE_FILT_BASELINE_PARQUET = PATHS["merge_filt_baseline_parquet"]
MERGE_FILT_BASELINE_PARQUET_LEGACY = PATHS["merge_filt_baseline_parquet_legacy"]
MERGE_FILT_PREFILT_PARQUET = PATHS["merge_filt_prefilt_parquet"]
MERGE_FILT_PREFILT_PARQUET_LEGACY = PATHS["merge_filt_prefilt_parquet_legacy"]
MERGE_FILT_MULT2_PARQUET = PATHS["merge_filt_mult2_parquet"]
MERGE_FILT_MULT2_PARQUET_LEGACY = PATHS["merge_filt_mult2_parquet_legacy"]
MERGE_FILT_MULT4_PARQUET = PATHS["merge_filt_mult4_parquet"]
MERGE_FILT_MULT4_PARQUET_LEGACY = PATHS["merge_filt_mult4_parquet_legacy"]
MERGE_FILT_MULT6_PARQUET = PATHS["merge_filt_mult6_parquet"]
MERGE_FILT_MULT6_PARQUET_LEGACY = PATHS["merge_filt_mult6_parquet_legacy"]

MERGE_FILT_STRICT_PARQUET = PATHS["merge_filt_strict_parquet"]

BUILD_OVERWRITE = _as_bool(BUILD_CFG.get("overwrite"), False)
BUILD_PREFILT_SQL = _as_optional_str(BUILD_CFG.get("prefilt_sql")) or ""
BUILD_FIRM_YEAR_USER_DEDUP = _as_bool(BUILD_CFG.get("firm_year_user_dedup"), True)
BUILD_AMBIGUITY_WEIGHT_GAP_CUTOFF = _as_float(
    BUILD_CFG.get("ambiguity_weight_gap_cutoff"), 0.03
)
BUILD_NO_COUNTRY_MIN_SUBREGION_SCORE = _as_float(
    BUILD_CFG.get("no_country_min_subregion_score"), 0.20
)
BUILD_NO_COUNTRY_MIN_TOTAL_SCORE = _as_float(
    BUILD_CFG.get("no_country_min_total_score"), 0.12
)
BUILD_NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL = _as_float(
    BUILD_CFG.get("no_country_min_f_score_if_est_yob_null"), 0.30
)
BUILD_BAD_MATCH_GUARD_ENABLED = _as_bool(
    BUILD_CFG.get("bad_match_guard_enabled"), True
)
BUILD_BAD_MATCH_GUARD_SUBREGION_SCORE_LT = _as_float(
    BUILD_CFG.get("bad_match_guard_subregion_score_lt"), 0.15
)
BUILD_BAD_MATCH_GUARD_F_SCORE_LT = _as_float(
    BUILD_CFG.get("bad_match_guard_f_score_lt"), 0.30
)
BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT = _as_float(
    BUILD_CFG.get("bad_match_guard_total_score_lt"), 0.10
)
BUILD_FIRM_NAME_MATCH_THRESHOLD = _as_float(
    BUILD_CFG.get("firm_name_match_threshold"), 0.85
)
BUILD_SUBREGION_BOOST_ALPHA = _as_float(
    BUILD_CFG.get("subregion_boost_alpha"), 0.4
)
BUILD_W_COUNTRY = _as_float(BUILD_CFG.get("w_country"), 0.70)
BUILD_W_YOB = _as_float(BUILD_CFG.get("w_yob"), 0.20)
BUILD_W_GENDER = _as_float(BUILD_CFG.get("w_gender"), 0.10)
BUILD_W_OCC = _as_float(BUILD_CFG.get("w_occ"), 0.0)
BUILD_OCC_SCORE_HALFLIFE = _as_float(BUILD_CFG.get("occ_score_halflife"), 500.0)
BUILD_FOIA_OCC_RANK_CUTOFF = _as_optional_int(BUILD_CFG.get("foia_occ_rank_cutoff"))

# Strict sample thresholds (post-hoc filter on baseline for high-precision analysis)
STRICT_MIN_WEIGHT_NORM = _as_float(BUILD_CFG.get("strict_min_weight_norm"), 0.90)
STRICT_MIN_TOTAL_SCORE = _as_float(BUILD_CFG.get("strict_min_total_score"), 0.55)
STRICT_MIN_FIRM_QUALITY = _as_float(BUILD_CFG.get("strict_min_firm_quality"), 0.80)
STRICT_MIN_COUNTRY_SCORE = _as_float(BUILD_CFG.get("strict_min_country_score"), 0.50)
STRICT_REQUIRE_EST_YOB = _as_bool(BUILD_CFG.get("strict_require_est_yob"), True)
STRICT_MAX_N_MATCH_FILT = _as_optional_int(BUILD_CFG.get("strict_max_n_match_filt"))

TESTING_ENABLED = _as_bool(TESTING_CFG.get("enabled"), False)
TESTING_SAMPLE_MATCHES = max(1, _as_int(TESTING_CFG.get("sample_matches"), 5))
TESTING_RANDOM_SEED = _as_optional_int(TESTING_CFG.get("random_seed"))
TESTING_FIRM_UID = _as_optional_str(TESTING_CFG.get("firm_uid"))
TESTING_LOTTERY_YEAR = _as_optional_str(TESTING_CFG.get("lottery_year"))
TESTING_MATERIALIZE_INTERMEDIATE_TABLES = _as_bool(
    TESTING_CFG.get("materialize_intermediate_tables"), True
)
TESTING_TABLE_PREFIX = _as_optional_str(TESTING_CFG.get("table_prefix")) or "imt"


def choose_path(primary: str, fallback: str | None = None) -> str:
    """Return existing primary path, else fallback (if given), else primary."""
    if os.path.exists(primary):
        return primary
    if fallback is not None:
        return fallback
    return primary
