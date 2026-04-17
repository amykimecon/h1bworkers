"""YAML-backed settings for f1_indiv_merge scripts."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import root  # noqa: E402

DEFAULT_CONFIG_PATH = Path(root) / "h1bworkers" / "code" / "configs" / "f1_indiv_merge.yaml"
REPO_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "f1_indiv_merge.yaml"
ENV_CONFIG_PATH = os.environ.get("F1_INDIV_MERGE_CONFIG")
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


# --- INPUT PATHS ---
F1_FOIA_PARQUET = PATHS["f1_foia_parquet"]
REV_INDIV_PARQUET = PATHS["rev_indiv_parquet"]
REV_INDIV_PARQUET_LEGACY = PATHS["rev_indiv_parquet_legacy"]
REV_EDUC_LONG_PARQUET = PATHS["rev_educ_long_parquet"]
REV_EDUC_LONG_PARQUET_LEGACY = PATHS["rev_educ_long_parquet_legacy"]
REV_POS_PARQUET = PATHS.get("rev_pos_parquet", "")
REV_POS_PARQUET_LEGACY = PATHS.get("rev_pos_parquet_legacy", "")
F1_REV_SCHOOL_CROSSWALK_PARQUET = PATHS["f1_rev_school_crosswalk_parquet"]
F1_INST_UNITID_CROSSWALK_PARQUET = PATHS.get("f1_inst_unitid_crosswalk_parquet", "")
REVELIO_IPEDS_FOIA_INST_CROSSWALK_PARQUET = PATHS.get("revelio_ipeds_foia_inst_crosswalk_parquet", "")
F1_REVELIO_SCHOOL_CROSSWALK_CANONICAL_PARQUET = PATHS.get("f1_revelio_school_crosswalk_parquet", "")
F1_REVELIO_IPEDS_RESOLUTION_PARQUET = PATHS.get("f1_revelio_ipeds_resolution_parquet", "")
F1_EMPLOYER_ENTITY_CROSSWALK_PARQUET = PATHS.get("f1_employer_entity_crosswalk_parquet", "")
F1_EMPLOYER_PREBUILT_CROSSWALK_PARQUET = PATHS.get("f1_employer_prebuilt_crosswalk_parquet", "")
F1_OPT_EMPLOYER_LOOKUP_PARQUET = PATHS.get("f1_opt_employer_lookup_parquet", "")
F1_REV_SCHOOL_RESOLUTION_PARQUET = PATHS.get("f1_rev_school_resolution_parquet", "")

# --- OUTPUT PATHS (spell-level) ---
F1_MERGE_BASELINE_PARQUET = PATHS["f1_merge_baseline_parquet"]
F1_MERGE_MULT2_PARQUET = PATHS["f1_merge_mult2_parquet"]
F1_MERGE_MULT4_PARQUET = PATHS["f1_merge_mult4_parquet"]
F1_MERGE_MULT6_PARQUET = PATHS["f1_merge_mult6_parquet"]
F1_MERGE_STRICT_PARQUET = PATHS["f1_merge_strict_parquet"]

# --- OUTPUT PATHS (person-level) ---
F1_MERGE_PERSON_BASELINE_PARQUET = PATHS.get("f1_merge_person_baseline_parquet", "")
F1_MERGE_PERSON_STRICT_PARQUET = PATHS.get("f1_merge_person_strict_parquet", "")

# --- BUILD PARAMS ---
BUILD_OVERWRITE = _as_bool(BUILD_CFG.get("overwrite"), False)
BUILD_SCHOOL_MATCH_THRESHOLD = _as_float(BUILD_CFG.get("school_match_threshold"), 0.88)
BUILD_SCHOOL_AMBIGUITY_SCORE_GAP = _as_float(BUILD_CFG.get("school_ambiguity_score_gap"), 0.03)
BUILD_SCHOOL_BLOCK_MODE = _as_lower_str(BUILD_CFG.get("school_block_mode"), "off")
if BUILD_SCHOOL_BLOCK_MODE not in {"off", "campus_unique"}:
    BUILD_SCHOOL_BLOCK_MODE = "off"
BUILD_PERSON_USER_DEDUP = _as_bool(BUILD_CFG.get("person_user_dedup"), True)

# Hard join filter: |f1_prog_start_year - rev_educ_start_year| <= this (NULL passes through)
BUILD_YEAR_HARD_BUFFER = _as_int(BUILD_CFG.get("year_hard_buffer"), 2)

# Employer weight: exponential saturation w_emp_eff = w_emp_max * (1 - exp(-n / emp_n_scale))
# The remaining (1 - w_emp_eff) is split among w_country/w_degree/w_date/w_field (renormalized).
# When employer_score is NULL, w_emp_eff = 0 and the four weights fill the full score.
BUILD_W_EMP_MAX   = _as_float(BUILD_CFG.get("w_emp_max"),   0.85)
BUILD_EMP_N_SCALE = _as_float(BUILD_CFG.get("emp_n_scale"), 3.0)
BUILD_W_COUNTRY = _as_float(BUILD_CFG.get("w_country"), 0.35)
BUILD_W_DEGREE  = _as_float(BUILD_CFG.get("w_degree"),  0.12)
BUILD_W_DATE    = _as_float(BUILD_CFG.get("w_date"),    0.08)
BUILD_W_FIELD   = _as_float(BUILD_CFG.get("w_field"),   0.10)
BUILD_DEGREE_SCORE_NULL_DEFAULT = _as_float(BUILD_CFG.get("degree_score_null_default"), 0.5)
BUILD_DATE_SCORE_NULL_DEFAULT   = _as_float(BUILD_CFG.get("date_score_null_default"),   0.5)

# Employer sequence scoring
BUILD_EMP_FUZZY_THRESHOLD = _as_float(BUILD_CFG.get("emp_fuzzy_threshold"), 0.70)
BUILD_EMP_IDF_SMOOTHING = _as_float(BUILD_CFG.get("emp_idf_smoothing"), 1.0)

# Subset token-overlap matching
BUILD_EMP_SUBSET_MATCH_THRESHOLD = _as_float(BUILD_CFG.get("emp_subset_match_threshold"), 0.40)
BUILD_EMP_SUBSET_MIN_TOKENS = _as_int(BUILD_CFG.get("emp_subset_min_tokens"), 2)
BUILD_EMP_TOKEN_MIN_IDF = _as_float(BUILD_CFG.get("emp_token_min_idf"), 0.05)

BUILD_AMBIGUITY_WEIGHT_GAP_CUTOFF = _as_float(BUILD_CFG.get("ambiguity_weight_gap_cutoff"), 0.05)
BUILD_BAD_MATCH_GUARD_ENABLED = _as_bool(BUILD_CFG.get("bad_match_guard_enabled"), True)
BUILD_BAD_MATCH_GUARD_COUNTRY_SCORE_LT = _as_float(
    BUILD_CFG.get("bad_match_guard_country_score_lt"), 0.15
)
BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT = _as_float(
    BUILD_CFG.get("bad_match_guard_total_score_lt"), 0.10
)
BUILD_MULT_CUTOFFS = BUILD_CFG.get("mult_cutoffs", [2, 4, 6])

# Strict sample thresholds (spell-level)
STRICT_MIN_WEIGHT_NORM  = _as_float(BUILD_CFG.get("strict_min_weight_norm"),  0.85)
STRICT_MIN_TOTAL_SCORE  = _as_float(BUILD_CFG.get("strict_min_total_score"),  0.50)
STRICT_MIN_COUNTRY_SCORE = _as_float(BUILD_CFG.get("strict_min_country_score"), 0.50)
STRICT_MAX_N_MATCH_FILT = _as_optional_int(BUILD_CFG.get("strict_max_n_match_filt"))

# Employer crosswalk (used by deps_f1_employer_crosswalk.py)
BUILD_EMPLOYER_MATCH_THRESHOLD = _as_float(BUILD_CFG.get("employer_match_threshold"), 0.88)
BUILD_EMPLOYER_MATCH_JW_ONLY = _as_bool(
    BUILD_CFG.get("employer_match_jw_only"), False
)
BUILD_EMPLOYER_ENABLE_FUZZY_MATCHING = _as_bool(
    BUILD_CFG.get("employer_enable_fuzzy_matching"), True
)

# Person-level aggregation strict threshold
STRICT_PERSON_MIN_WEIGHT_NORM = _as_float(BUILD_CFG.get("strict_person_min_weight_norm"), 0.85)

# --- TESTING PARAMS ---
TESTING_ENABLED = _as_bool(TESTING_CFG.get("enabled"), False)
TESTING_SAMPLE_N_PERSONS = max(1, _as_int(TESTING_CFG.get("sample_n_persons"), 1000))
TESTING_RANDOM_SEED = _as_optional_int(TESTING_CFG.get("random_seed"))
TESTING_SCHOOL = _as_optional_str(TESTING_CFG.get("test_school"))
TESTING_COUNTRY = _as_optional_str(TESTING_CFG.get("test_country"))
TESTING_MATERIALIZE_INTERMEDIATE_TABLES = _as_bool(
    TESTING_CFG.get("materialize_intermediate_tables"), True
)
TESTING_TABLE_PREFIX = _as_optional_str(TESTING_CFG.get("table_prefix")) or "f1mt"


def choose_path(primary: str, fallback: str | None = None) -> str:
    """Return existing primary path, else fallback (if given), else primary."""
    if os.path.exists(primary):
        return primary
    if fallback is not None:
        return fallback
    return primary
