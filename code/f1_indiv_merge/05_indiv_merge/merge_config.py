"""Pipeline-config-backed settings for the local 05_indiv_merge stage."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = STAGE_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
for _path in (STAGE_DIR, PIPELINE_ROOT, REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import src.config_loader as cfg_loader  # noqa: E402
from src.config_loader import get_stage_config, load_config  # noqa: E402
from src.pipeline_runtime import coerce_bool  # noqa: E402

STAGE_NAME = "05_indiv_merge"
ACTIVE_CONFIG_PATH: str | None = cfg_loader.ACTIVE_CONFIG_PATH


def _load_pipeline_config() -> dict[str, Any]:
    config_path = os.environ.get("F1_INDIV_PIPELINE_CONFIG")
    cfg = load_config(config_path)
    global ACTIVE_CONFIG_PATH
    ACTIVE_CONFIG_PATH = cfg_loader.ACTIVE_CONFIG_PATH
    return cfg


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
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _as_optional_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    items: list[Any]
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: list[int] = []
    for item in items:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out or None


def _as_optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    items: list[Any]
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        text = _as_optional_str(item)
        if text is not None:
            out.append(text)
    return out or None


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
    text = str(value).strip().lower()
    return text or default


def _existing_path(path: str | None) -> str | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    return text if Path(text).exists() else None


def choose_path(primary: str | None, fallback: str | None = None) -> str:
    """Return the existing primary path, else fallback when available."""
    primary_existing = _existing_path(primary)
    if primary_existing:
        return primary_existing
    fallback_existing = _existing_path(fallback)
    if fallback_existing:
        return fallback_existing
    return str(primary or fallback or "")


CFG = _load_pipeline_config()
RUN_TAG = str(CFG.get("run_tag", "apr2026v1"))
BUILD_CFG = CFG.get("build", {})
TESTING_CFG = CFG.get("testing", {})
STAGE_CFG = get_stage_config(CFG, STAGE_NAME)

# --- INPUT PATHS ---
F1_FOIA_PARQUET = choose_path(
    STAGE_CFG.get("f1_foia_input_parquet"),
    STAGE_CFG.get("f1_foia_fallback_parquet"),
)
REV_USERS_CORE_PARQUET = _as_optional_str(STAGE_CFG.get("rev_users_core_input_parquet"))
REV_MATCH_READY_PARQUET = _as_optional_str(STAGE_CFG.get("rev_match_ready_input_parquet"))
REV_EDUC_LONG_PARQUET = _as_optional_str(STAGE_CFG.get("rev_educ_clean_long_input_parquet"))
REV_POS_PARQUET = _as_optional_str(STAGE_CFG.get("rev_pos_clean_long_input_parquet"))
REV_INDIV_PARQUET_LEGACY = _as_optional_str(STAGE_CFG.get("legacy_rev_indiv_parquet"))
REV_EDUC_LONG_PARQUET_LEGACY = _as_optional_str(STAGE_CFG.get("legacy_rev_educ_long_parquet"))
REV_POS_PARQUET_LEGACY = _as_optional_str(STAGE_CFG.get("legacy_rev_pos_parquet"))
F1_REV_SCHOOL_CROSSWALK_PARQUET = _as_optional_str(STAGE_CFG.get("school_family_crosswalk_input_parquet")) or ""
F1_REV_SCHOOL_RESOLUTION_PARQUET = _as_optional_str(STAGE_CFG.get("school_resolution_input_parquet")) or ""
F1_INST_UNITID_CROSSWALK_PARQUET = _as_optional_str(STAGE_CFG.get("f1_inst_unitid_crosswalk_input_parquet")) or ""
F1_OPT_EMPLOYER_LOOKUP_PARQUET = _as_optional_str(STAGE_CFG.get("employer_lookup_input_parquet")) or ""
ECON_RELABELS_PARQUET = _as_optional_str(STAGE_CFG.get("econ_relabels_input_parquet"))
RELABEL_SAMPLE_CACHE_PARQUET = _as_optional_str(STAGE_CFG.get("relabel_sample_cache_parquet"))
RELABEL_SAMPLE_FORCE_REBUILD = coerce_bool(STAGE_CFG.get("relabel_sample_force_rebuild"), False)

# --- OUTPUT PATHS (spell-level) ---
F1_MERGE_BASELINE_PARQUET = str(STAGE_CFG["baseline_parquet"])
F1_MERGE_MULT2_PARQUET = str(STAGE_CFG["mult2_parquet"])
F1_MERGE_MULT4_PARQUET = str(STAGE_CFG["mult4_parquet"])
F1_MERGE_MULT6_PARQUET = str(STAGE_CFG["mult6_parquet"])
F1_MERGE_STRICT_PARQUET = str(STAGE_CFG["strict_parquet"])

# --- OUTPUT PATHS (person-level) ---
F1_MERGE_PERSON_BASELINE_PARQUET = str(STAGE_CFG["person_baseline_parquet"])
F1_MERGE_PERSON_STRICT_PARQUET = str(STAGE_CFG["person_strict_parquet"])

# --- OPTIONAL REFERENCE OUTPUTS ---
COMPARE_TO_REFERENCE_OUTPUTS = coerce_bool(STAGE_CFG.get("compare_to_reference_outputs"), False)
REFERENCE_SPELL_BASELINE_PARQUET = _as_optional_str(STAGE_CFG.get("reference_spell_baseline_parquet"))
REFERENCE_SPELL_STRICT_PARQUET = _as_optional_str(STAGE_CFG.get("reference_spell_strict_parquet"))
REFERENCE_PERSON_BASELINE_PARQUET = _as_optional_str(STAGE_CFG.get("reference_person_baseline_parquet"))

# --- BUILD PARAMS ---
BUILD_OVERWRITE = coerce_bool(BUILD_CFG.get("overwrite"), False)
BUILD_SCHOOL_MATCH_THRESHOLD = _as_float(BUILD_CFG.get("school_match_threshold"), 0.88)
BUILD_SCHOOL_AMBIGUITY_SCORE_GAP = _as_float(BUILD_CFG.get("school_ambiguity_score_gap"), 0.03)
BUILD_SCHOOL_BLOCK_MODE = _as_lower_str(BUILD_CFG.get("school_block_mode"), "off")
if BUILD_SCHOOL_BLOCK_MODE not in {"off", "campus_unique"}:
    BUILD_SCHOOL_BLOCK_MODE = "off"
BUILD_PERSON_USER_DEDUP = coerce_bool(BUILD_CFG.get("person_user_dedup"), True)
BUILD_YEAR_HARD_BUFFER = _as_int(BUILD_CFG.get("year_hard_buffer"), 2)
BUILD_GRADYR_SCORE_DECAY_POWER = max(
    0.01,
    _as_float(BUILD_CFG.get("gradyr_score_decay_power"), 1.0),
)
BUILD_W_EMP_MAX = _as_float(BUILD_CFG.get("w_emp_max"), 0.85)
BUILD_EMP_N_SCALE = _as_float(BUILD_CFG.get("emp_n_scale"), 3.0)
BUILD_MULTIPLICATIVE_SCORE = coerce_bool(BUILD_CFG.get("multiplicative_score"), False)
BUILD_SUBREGION_BOOST_ALPHA = _as_float(BUILD_CFG.get("subregion_boost_alpha"), 0.4)
BUILD_COUNTRY_COMPETITION_WEIGHT = _as_float(
    BUILD_CFG.get("country_competition_weight"),
    0.0,
)
BUILD_COUNTRY_COMPETITION_THRESHOLD = _as_float(
    BUILD_CFG.get("country_competition_threshold"),
    0.0,
)
BUILD_W_COUNTRY = _as_float(BUILD_CFG.get("w_country"), 0.35)
BUILD_W_DEGREE = _as_float(BUILD_CFG.get("w_degree"), 0.12)
BUILD_W_DATE = _as_float(BUILD_CFG.get("w_date"), 0.08)
BUILD_W_FIELD = _as_float(BUILD_CFG.get("w_field"), 0.10)
BUILD_DEGREE_SCORE_NULL_DEFAULT = _as_float(BUILD_CFG.get("degree_score_null_default"), 0.5)
BUILD_DATE_SCORE_NULL_DEFAULT = _as_float(BUILD_CFG.get("date_score_null_default"), 0.5)
BUILD_EMP_FUZZY_THRESHOLD = _as_float(BUILD_CFG.get("emp_fuzzy_threshold"), 0.70)
BUILD_EMP_IDF_SMOOTHING = _as_float(BUILD_CFG.get("emp_idf_smoothing"), 1.0)
BUILD_EMP_SUBSET_MATCH_THRESHOLD = _as_float(BUILD_CFG.get("emp_subset_match_threshold"), 0.40)
BUILD_EMP_SUBSET_MIN_TOKENS = _as_int(BUILD_CFG.get("emp_subset_min_tokens"), 2)
BUILD_EMP_TOKEN_MIN_IDF = _as_float(BUILD_CFG.get("emp_token_min_idf"), 0.05)
BUILD_ENFORCE_INDIVIDUAL_ONE_TO_ONE = coerce_bool(
    BUILD_CFG.get("enforce_individual_one_to_one"),
    True,
)
BUILD_AMBIGUITY_WEIGHT_GAP_CUTOFF = _as_float(BUILD_CFG.get("ambiguity_weight_gap_cutoff"), 0.05)
BUILD_BAD_MATCH_GUARD_ENABLED = coerce_bool(BUILD_CFG.get("bad_match_guard_enabled"), True)
BUILD_BAD_MATCH_GUARD_COUNTRY_SCORE_LT = _as_float(
    BUILD_CFG.get("bad_match_guard_country_score_lt"),
    0.15,
)
BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT = _as_float(
    BUILD_CFG.get("bad_match_guard_total_score_lt"),
    0.10,
)
BUILD_MULT_CUTOFFS = BUILD_CFG.get("mult_cutoffs", [2, 4, 6])
STRICT_MIN_WEIGHT_NORM = _as_float(BUILD_CFG.get("strict_min_weight_norm"), 0.85)
STRICT_MIN_TOTAL_SCORE = _as_float(BUILD_CFG.get("strict_min_total_score"), 0.50)
STRICT_MIN_COUNTRY_SCORE = _as_float(BUILD_CFG.get("strict_min_country_score"), 0.50)
STRICT_MAX_N_MATCH_FILT = _as_optional_int(BUILD_CFG.get("strict_max_n_match_filt"))
STRICT_PERSON_MIN_WEIGHT_NORM = _as_float(BUILD_CFG.get("strict_person_min_weight_norm"), 0.85)
RESTRICT_TO_RELABEL_PROGRAMS = coerce_bool(STAGE_CFG.get("restrict_to_relabel_programs"), False)
RELABEL_GRADYEAR_WINDOW = _as_int(STAGE_CFG.get("relabel_gradyear_window"), 5)
EMPLOYMENT_HISTORY_FILTER_ENABLED = coerce_bool(
    STAGE_CFG.get("employment_history_filter_enabled"),
    True,
)
EMPLOYER_MATCH_YEAR_BUFFER = max(0, _as_int(STAGE_CFG.get("employer_match_year_buffer"), 2))
RELATIVE_SCORE_FILTER_ENABLED = coerce_bool(
    STAGE_CFG.get("relative_score_filter_enabled"),
    True,
)
EMPLOYER_SCORE_RELATIVE_BUFFER = _as_float(
    STAGE_CFG.get("employer_score_relative_buffer"),
    0.15,
)
EMPLOYER_SCORE_RELATIVE_APPLY_MIN = _as_float(
    STAGE_CFG.get("employer_score_relative_apply_min"),
    0.35,
)
FIELD_SCORE_RELATIVE_BUFFER = _as_float(
    STAGE_CFG.get("field_score_relative_buffer"),
    0.20,
)
FIELD_SCORE_RELATIVE_APPLY_MIN = _as_float(
    STAGE_CFG.get("field_score_relative_apply_min"),
    0.75,
)
COUNTRY_SCORE_RELATIVE_BUFFER = _as_float(
    STAGE_CFG.get("country_score_relative_buffer"),
    0.20,
)
COUNTRY_SCORE_RELATIVE_APPLY_MIN = _as_float(
    STAGE_CFG.get("country_score_relative_apply_min"),
    0.35,
)
BUILD_W_GRADYR = _as_float(BUILD_CFG.get("w_gradyr"), _as_float(BUILD_CFG.get("w_date"), 0.15))
BUILD_W_INST = _as_float(BUILD_CFG.get("w_inst"), 0.15)
BUILD_FIELD_TARGET_CIP4 = _as_int(BUILD_CFG.get("field_target_cip4"), 4506)
BUILD_FIELD_NON_TARGET_CAP = _as_float(BUILD_CFG.get("field_non_target_cap"), 0.85)
BUILD_FIELD_FILTER_MIN_SCORE = _as_float(BUILD_CFG.get("field_filter_min_score"), 0.25)
BUILD_FIELD_CIP2_MATCH_MULTIPLIER = _as_float(BUILD_CFG.get("field_cip2_match_multiplier"), 0.70)
FIELD_CANDIDATE_FILTER_ENABLED = coerce_bool(
    STAGE_CFG.get("field_candidate_filter_enabled"),
    True,
)

# --- TESTING PARAMS ---
TESTING_ENABLED = coerce_bool(TESTING_CFG.get("enabled"), False)
TESTING_SAMPLE_N_PERSONS = max(1, _as_int(TESTING_CFG.get("sample_n_persons"), 100))
TESTING_RANDOM_SEED = _as_optional_int(TESTING_CFG.get("random_seed"))
TESTING_INDIVIDUAL_KEYS = _as_optional_str_list(TESTING_CFG.get("individual_keys"))
TESTING_PERSON_IDS = _as_optional_int_list(TESTING_CFG.get("person_ids"))
TESTING_SCHOOL = _as_optional_str(TESTING_CFG.get("test_school"))
TESTING_COUNTRY = _as_optional_str(TESTING_CFG.get("test_country"))
TESTING_MATERIALIZE_INTERMEDIATE_TABLES = coerce_bool(
    TESTING_CFG.get("materialize_intermediate_tables"),
    True,
)
TESTING_TABLE_PREFIX = _as_optional_str(TESTING_CFG.get("table_prefix")) or "f1mt"
