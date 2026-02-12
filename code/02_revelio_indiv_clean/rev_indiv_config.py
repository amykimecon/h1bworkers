"""YAML-backed shared settings for 02_revelio_indiv_clean scripts."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import root, wrds_out  # noqa: E402

DEFAULT_CONFIG_PATH = Path(root) / "h1bworkers" / "code" / "configs" / "rev_indiv_clean.yaml"
REPO_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "rev_indiv_clean.yaml"
ENV_CONFIG_PATH = os.environ.get("REV_INDIV_CONFIG")
_VAR_PATTERN = re.compile(r"\$(\w+)|\$\{([^}]+)\}")
ACTIVE_CONFIG_PATH: str | None = None


def _replace_known_vars(value: str) -> str:
    return value.replace("{root}", str(root)).replace("{wrds_out}", str(wrds_out))


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
TESTING = CFG.get("testing", {})


def _test_cfg(section: str) -> dict[str, Any]:
    cfg = TESTING.get(section, {})
    if not isinstance(cfg, dict):
        return {}
    return cfg

# Main run tag for refreshed outputs in this folder.
RUN_TAG = str(CFG.get("run_tag", "feb2026"))

# Legacy tags/files used before migration.
LEGACY_WRDS_USERS_TAG = str(CFG.get("legacy_tags", {}).get("wrds_users", "sep2"))
LEGACY_WRDS_POSITIONS_TAG = str(CFG.get("legacy_tags", {}).get("wrds_positions", "aug1"))
LEGACY_INST_COUNTRIES_TAG = str(CFG.get("legacy_tags", {}).get("inst_countries", "jun30"))
LEGACY_NAMETRACE_LONG_TAG = str(CFG.get("legacy_tags", {}).get("nametrace_long", "jul8"))

# Shared upstream inputs
DUP_RCIDS_CSV = PATHS["dup_rcids_csv"]
GOOD_MATCH_IDS_CSV = PATHS["good_match_ids_csv"]
LLM_CROSSWALK_CSV = PATHS["llm_crosswalk_csv"]

# WRDS users
WRDS_USERS_PARQUET = PATHS["wrds_users_parquet"]
WRDS_USERS_PARQUET_LEGACY = PATHS["wrds_users_parquet_legacy"]
WRDS_USERS_CHUNK_STUB = PATHS["wrds_users_chunk_stub"]

# WRDS positions
WRDS_POSITIONS_PARQUET = PATHS["wrds_positions_parquet"]
WRDS_POSITIONS_PARQUET_LEGACY = PATHS["wrds_positions_parquet_legacy"]
WRDS_POSITIONS_CHUNK_STUB = PATHS["wrds_positions_chunk_stub"]

# NameTrace outputs
NAMETRACE_CHUNK_DIR = PATHS["nametrace_chunk_dir"]
NAMETRACE_CHUNK_STUB = PATHS["nametrace_chunk_stub"]
NAMETRACE_WIDE_PARQUET = PATHS["nametrace_wide_parquet"]
NAMETRACE_LONG_PARQUET = PATHS["nametrace_long_parquet"]
NAMETRACE_LONG_PARQUET_LEGACY = PATHS["nametrace_long_parquet_legacy"]

# Name2Nat outputs
NAME2NAT_CHUNK_STUB = PATHS["name2nat_chunk_stub"]
NAME2NAT_CHUNK_STUB_TEST = PATHS["name2nat_chunk_stub_test"]
NAME2NAT_PARQUET = PATHS["name2nat_parquet"]
NAME2NAT_PARQUET_TEST = PATHS.get(
    "name2nat_parquet_test",
    NAME2NAT_PARQUET.replace(".parquet", "_test.parquet"),
)

# Institution-country outputs
REV_INST_COUNTRIES_PARQUET = PATHS["rev_inst_countries_parquet"]
REV_INST_COUNTRIES_PARQUET_LEGACY = PATHS["rev_inst_countries_parquet_legacy"]

# clean_revelio_institutions intermediates
OPENALEX_MATCH_FILT_PARQUET = PATHS["openalex_match_filt_parquet"]
DEDUP_MATCH_FILT_PARQUET = PATHS["dedup_match_filt_parquet"]
ALL_TOKEN_FREQS_PARQUET = PATHS["all_token_freqs_parquet"]
GEONAMES_TOKEN_MERGE_PARQUET = PATHS["geonames_token_merge_parquet"]
OPENALEX_MATCH_FILT_PARQUET_LEGACY = PATHS["openalex_match_filt_parquet_legacy"]
DEDUP_MATCH_FILT_PARQUET_LEGACY = PATHS["dedup_match_filt_parquet_legacy"]

# rev_users_clean intermediates
REV_EDUC_LONG_PARQUET = PATHS["rev_educ_long_parquet"]
MERGED_POS_CLEAN_PARQUET = PATHS["merged_pos_clean_parquet"]
COMPANY_MERGE_SAMPLE_PARQUET = PATHS["company_merge_sample_parquet"]
FOIA_INDIV_PARQUET = PATHS["foia_indiv_parquet"]
REV_INDIV_PARQUET = PATHS["rev_indiv_parquet"]

# Legacy WRDS shard format (used as fallback by some scripts)
_legacy_shard_count = int(CFG.get("legacy_wrds_user_merge_shard_count", 10))
LEGACY_WRDS_USER_MERGE_SHARDS = [
    f"{wrds_out}/rev_user_merge{i}.parquet" for i in range(_legacy_shard_count)
]

# Script testing toggles/knobs
_WRDS_USERS_TEST = _test_cfg("wrds_users")
WRDS_USERS_TEST = bool(_WRDS_USERS_TEST.get("enabled", False))
WRDS_USERS_TEST_RCID_LIMIT = int(_WRDS_USERS_TEST.get("rcid_limit", 20))
WRDS_USERS_CHUNKS = int(_WRDS_USERS_TEST.get("chunks", 20))
WRDS_USERS_CHUNK_SIZE = int(_WRDS_USERS_TEST.get("chunk_size", 10000))

_WRDS_POSITIONS_TEST = _test_cfg("wrds_positions")
WRDS_POSITIONS_TEST = bool(_WRDS_POSITIONS_TEST.get("enabled", False))
WRDS_POSITIONS_TEST_RCID_LIMIT = int(_WRDS_POSITIONS_TEST.get("rcid_limit", 10))
WRDS_POSITIONS_CHUNKS = int(_WRDS_POSITIONS_TEST.get("chunks", 20))
WRDS_POSITIONS_CHUNK_SIZE = int(_WRDS_POSITIONS_TEST.get("chunk_size", 10000))

_NAME2NAT_TEST = _test_cfg("rev_indiv_name2nat")
NAME2NAT_TEST = bool(_NAME2NAT_TEST.get("enabled", False))
NAME2NAT_TESTN = int(_NAME2NAT_TEST.get("testn", 1000))
NAME2NAT_CHUNKS = int(_NAME2NAT_TEST.get("chunks", 20))
NAME2NAT_CHUNK_SIZE = int(_NAME2NAT_TEST.get("chunk_size", 100000))

_NAMETRACE_TEST = _test_cfg("rev_indiv_nametrace")
NAMETRACE_TEST = bool(_NAMETRACE_TEST.get("enabled", False))
NAMETRACE_TESTN = int(_NAMETRACE_TEST.get("testn", 1000))
NAMETRACE_CHUNKS = int(_NAMETRACE_TEST.get("chunks", 20))
NAMETRACE_CHUNK_SIZE = int(_NAMETRACE_TEST.get("chunk_size", 100000))

_CLEAN_INST_TEST = _test_cfg("clean_revelio_institutions")
CLEAN_INST_TEST = bool(_CLEAN_INST_TEST.get("enabled", False))
CLEAN_INST_RANDOM_UNIV_SAMPLE_N = int(_CLEAN_INST_TEST.get("random_univ_sample_n", 100))
CLEAN_INST_TOKENMATCH = bool(_CLEAN_INST_TEST.get("tokenmatch", False))

_REV_USERS_CLEAN_TEST = _test_cfg("rev_users_clean")
REV_USERS_CLEAN_TEST = bool(_REV_USERS_CLEAN_TEST.get("enabled", False))
REV_USERS_CLEAN_TEST_USER = float(_REV_USERS_CLEAN_TEST.get("test_user", 322249231.0))
REV_USERS_CLEAN_RANDOM_USER_SAMPLE_N = int(_REV_USERS_CLEAN_TEST.get("random_user_sample_n", 100))


def choose_path(primary: str, fallback: str | None = None) -> str:
    """Return existing primary path, else fallback (if given), else primary."""
    if os.path.exists(primary):
        return primary
    if fallback is not None:
        return fallback
    return primary
