"""Stage runner for 04_rev_user_clean."""

from __future__ import annotations

import argparse
import copy
import re
import sys
import tempfile
import time
from builtins import print as _print
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = STAGE_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
for _path in (STAGE_DIR, PIPELINE_ROOT, REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from local_name2nat import run_name2nat
from local_nametrace import run_nametrace
from common import (
    atomic_duckdb_copy_to_parquet,
    escape_sql_literal,
    get_duckdb_connection,
    resolve_existing_path,
    resolve_user_shard_spec,
    shard_output_path,
    sql_user_shard_predicate,
)
from src.config_loader import get_stage_config, load_config
from src.pipeline_runtime import coerce_bool, sanitize_ipykernel_argv
from src.progress_tracker import mark_stage_complete
from user_clean import Stage04Session, assemble_clean_user_artifacts

print = partial(_print, flush=True)

STAGE_NAME = "04_rev_user_clean"
_SHARDABLE_OUTPUT_KEYS = (
    "name2nat_parquet",
    "nametrace_wide_parquet",
    "nametrace_long_parquet",
    "rev_users_core_parquet",
    "rev_educ_clean_long_parquet",
    "rev_pos_clean_long_parquet",
    "rev_match_ready_parquet",
    "rev_educ_inst_candidates_long_parquet",
    "rev_educ_cip_candidates_long_parquet",
)
_DET_DEGREE_TYPE_ALIASES = {
    "master": ("masters",),
    "masters": ("masters",),
    "mba": ("masters",),
    "bachelor": ("bachelors",),
    "bachelors": ("bachelors",),
    "associate": ("associates",),
    "associates": ("associates",),
    "doctor": ("doctor", "doctors"),
    "doctors": ("doctor", "doctors"),
    "doctorate": ("doctor", "doctors"),
    "phd": ("doctor", "doctors"),
    "ph.d": ("doctor", "doctors"),
    "high school": ("hs_or_below",),
    "hs_or_below": ("hs_or_below",),
    "non-degree": ("non_degree",),
    "non_degree": ("non_degree",),
    "unknown": ("unknown",),
}


def _sql_string_literal(value: str) -> str:
    return f"'{escape_sql_literal(value)}'"


def _normalize_key_sql(column: str) -> str:
    return (
        "trim("
        "regexp_replace("
        "regexp_replace("
        f"lower(strip_accents(coalesce(CAST({column} AS VARCHAR), ''))), "
        "'[^a-z0-9\\s]+', ' ', 'g'"
        "), "
        "'\\s+', ' ', 'g'"
        ")"
        ")"
    )


def _fmt_elapsed(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _describe_parquet_columns(path: str) -> list[str]:
    con = get_duckdb_connection()
    return [
        row[0]
        for row in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{escape_sql_literal(path)}')"
        ).fetchall()
    ]


def _normalize_degree_type_filter_values(values: Any) -> list[str]:
    if values is None:
        values = ["masters"]
    if isinstance(values, str):
        items = [values]
    elif isinstance(values, (list, tuple, set)):
        items = list(values)
    else:
        items = [str(values)]
    expanded: list[str] = []
    seen: set[str] = set()
    for value in items:
        text = str(value).strip().lower()
        if not text:
            continue
        candidates = _DET_DEGREE_TYPE_ALIASES.get(text, (text,))
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded or ["masters"]


def _normalize_cip_prefix_filter_values(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        items = [values]
    elif isinstance(values, (list, tuple, set)):
        items = list(values)
    else:
        items = [str(values)]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in items:
        text = str(value).strip()
        if not text:
            continue
        text = re.sub(r"(?i)[x*]+$", "", text)
        digits = re.sub(r"[^0-9]", "", text)
        if not digits or digits in seen:
            continue
        seen.add(digits)
        normalized.append(digits)
    return normalized


def _degree_type_array_has_allowed_predicate(array_expr: str, allowed_types: list[str]) -> str:
    checks = [f"list_contains({array_expr}, {_sql_string_literal(value)})" for value in allowed_types]
    checks_sql = " OR ".join(checks) if checks else "FALSE"
    return (
        f"{array_expr} IS NOT NULL AND array_length({array_expr}) > 0 "
        f"AND ({checks_sql})"
    )


def _degree_type_array_has_informative_predicate(array_expr: str) -> str:
    return (
        f"{array_expr} IS NOT NULL AND array_length({array_expr}) > 0 "
        f"AND NOT (array_length({array_expr}) = 1 AND lower(CAST({array_expr}[1] AS VARCHAR)) = 'unknown')"
    )


def _cip_candidate_array_has_required_prefix_predicate(array_expr: str, required_prefixes: list[str]) -> str:
    checks = [
        (
            "REGEXP_REPLACE(COALESCE(CAST(candidate.cip_code AS VARCHAR), ''), '[^0-9]', '', 'g') "
            f"LIKE {_sql_string_literal(prefix + '%')}"
        )
        for prefix in required_prefixes
    ]
    checks_sql = " OR ".join(checks) if checks else "FALSE"
    return (
        f"{array_expr} IS NOT NULL AND array_length({array_expr}) > 0 "
        f"AND list_count(list_filter({array_expr}, candidate -> ({checks_sql}))) > 0"
    )


def _resolve_filter_source_paths(cfg: dict[str, Any], stage_cfg: dict[str, Any]) -> dict[str, str | None]:
    stage02_cfg = cfg.get("stages", {}).get("02_rev_import", {})
    paths_cfg = cfg.get("paths", {})
    return {
        "raw_users": resolve_existing_path(
            stage_cfg.get("wrds_users_input_parquet"),
            stage02_cfg.get("wrds_users_parquet"),
        ),
        "raw_positions": resolve_existing_path(
            stage_cfg.get("wrds_positions_input_parquet"),
            stage02_cfg.get("wrds_positions_parquet"),
        ),
        "legacy_rev_indiv": resolve_existing_path(
            stage_cfg.get("legacy_rev_indiv_parquet"),
            paths_cfg.get("legacy_rev_indiv_parquet"),
        ),
        "legacy_rev_educ": resolve_existing_path(
            stage_cfg.get("legacy_rev_educ_long_parquet"),
            paths_cfg.get("legacy_rev_educ_long_parquet"),
        ),
        "legacy_rev_pos": resolve_existing_path(
            stage_cfg.get("legacy_rev_pos_parquet"),
            paths_cfg.get("legacy_rev_pos_parquet"),
        ),
    }


def _configure_user_shard(
    cfg: dict[str, Any],
    *,
    shard_count: int | None,
    shard_id: int | None,
) -> dict[str, int | str] | None:
    stage_cfg = get_stage_config(cfg, STAGE_NAME)
    if shard_count is not None or shard_id is not None:
        if shard_count is None or shard_id is None:
            raise ValueError("Stage-04 sharding requires both `shard_count` and `shard_id`.")
        stage_cfg["user_shard_count"] = int(shard_count)
        stage_cfg["user_shard_id"] = int(shard_id)
    shard_spec = resolve_user_shard_spec(stage_cfg)
    if not shard_spec:
        return None
    for key in _SHARDABLE_OUTPUT_KEYS:
        path = stage_cfg.get(key)
        if not path:
            continue
        stage_cfg[key] = shard_output_path(
            str(path),
            shard_count=int(shard_spec["user_shard_count"]),
            shard_id=int(shard_spec["user_shard_id"]),
        )
    return shard_spec


def _merge_stage04_sharded_outputs(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    *,
    shard_count: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(pipeline_cfg or load_config(config_path))
    stage_cfg = get_stage_config(cfg, STAGE_NAME)
    effective_shard_count = shard_count
    if effective_shard_count is None:
        raw_value = stage_cfg.get("user_shard_count")
        if raw_value in (None, "", 0):
            raise ValueError("Shard merge requires `shard_count` (or `user_shard_count` in stage config).")
        effective_shard_count = int(raw_value)
    if int(effective_shard_count) < 2:
        raise ValueError("Shard merge requires `shard_count >= 2`.")

    con = get_duckdb_connection()
    merged: dict[str, Any] = {
        "user_shard_count": int(effective_shard_count),
        "merged_outputs": [],
        "skipped_outputs": [],
    }
    for key in _SHARDABLE_OUTPUT_KEYS:
        base_path = stage_cfg.get(key)
        if not base_path:
            continue
        shard_paths = [
            shard_output_path(
                str(base_path),
                shard_count=int(effective_shard_count),
                shard_id=shard_id,
            )
            for shard_id in range(int(effective_shard_count))
        ]
        existing = [Path(path).exists() for path in shard_paths]
        if not any(existing):
            merged["skipped_outputs"].append(key)
            continue
        if not all(existing):
            missing = [path for path, exists in zip(shard_paths, existing) if not exists]
            raise FileNotFoundError(
                f"Cannot merge `{key}` because some shard outputs are missing: {missing}"
            )
        print(
            f"[{STAGE_NAME}] Merging {key} from {int(effective_shard_count)} shard files -> {base_path}"
        )
        atomic_duckdb_copy_to_parquet(
            con,
            "SELECT * FROM read_parquet(?)",
            str(base_path),
            [[str(path) for path in shard_paths]],
        )
        merged["merged_outputs"].append(key)
        merged[key] = str(base_path)
    return merged


def _guard_unfiltered_raw_stage_run(cfg: dict[str, Any], *, testing: bool) -> None:
    if testing:
        return
    stage_cfg = get_stage_config(cfg, STAGE_NAME)
    if coerce_bool(stage_cfg.get("user_degree_filter_enabled"), False):
        return
    shard_spec = resolve_user_shard_spec(stage_cfg)
    if shard_spec is not None:
        return
    if coerce_bool(stage_cfg.get("allow_unfiltered_raw_stage02_run"), False):
        return
    source_paths = _resolve_filter_source_paths(cfg, stage_cfg)
    if not (source_paths["raw_users"] and source_paths["raw_positions"]):
        return
    raise RuntimeError(
        f"{STAGE_NAME} aborted before execution: `user_degree_filter_enabled=false` with the full "
        "stage-02 raw users/positions inputs will expand the run to the entire raw Revelio pull and "
        "has been observed to crash DuckDB/Python with a native segmentation fault. "
        "If you only want to remove the downstream field filter, leave stage 04 unchanged and instead "
        "set `stages.05_indiv_merge.field_candidate_filter_enabled=false`, "
        "`stages.05_indiv_merge.field_score_relative_apply_min=1.01`, and optionally `build.w_field=0.0`. "
        "If you truly want the full unfiltered raw stage-04 rerun, set "
        "`stages.04_rev_user_clean.allow_unfiltered_raw_stage02_run=true` and use a smaller custom raw input "
        "or a testing slice first."
    )


def _copy_filtered_user_rows(
    *,
    con,
    source_path: str,
    out_path: Path,
    keep_users_query: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    t0 = time.perf_counter()
    print(f"[{STAGE_NAME}] Writing filtered rows: {source_path} -> {out_path}")
    con.sql(
        f"""
        COPY (
            WITH keep_users AS (
                {keep_users_query}
            )
            SELECT src.*
            FROM read_parquet('{escape_sql_literal(source_path)}') AS src
            INNER JOIN keep_users AS ku
              ON TRY_CAST(src.user_id AS BIGINT) = ku.user_id
        )
        TO '{escape_sql_literal(out_path)}' (FORMAT PARQUET)
        """
    )
    print(f"[{STAGE_NAME}] Finished writing {out_path.name} in {_fmt_elapsed(time.perf_counter() - t0)}")


def _prepare_degree_filtered_stage_inputs(
    cfg: dict[str, Any],
    *,
    workdir: Path,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    stage_cfg = get_stage_config(cfg, STAGE_NAME)
    shard_spec = resolve_user_shard_spec(stage_cfg)
    enabled = coerce_bool(stage_cfg.get("user_degree_filter_enabled"), False)
    allowed_degree_types = _normalize_degree_type_filter_values(
        stage_cfg.get("user_degree_filter_allowed_degree_types", stage_cfg.get("user_degree_filter_allowed_degree_clean"))
    )
    required_cip_prefixes = _normalize_cip_prefix_filter_values(
        stage_cfg.get(
            "user_degree_filter_required_deterministic_cip_prefixes",
            stage_cfg.get("user_degree_filter_required_deterministic_cip_prefix"),
        )
    )
    include_no_degree = coerce_bool(stage_cfg.get("user_degree_filter_include_no_degree"), True)
    shard_predicate = (
        sql_user_shard_predicate(
            "user_id",
            shard_count=int(shard_spec["user_shard_count"]),
            shard_id=int(shard_spec["user_shard_id"]),
        )
        if shard_spec
        else ""
    )
    shard_where_sql = f" AND {shard_predicate}" if shard_predicate else ""
    stats: dict[str, Any] = {
        "user_degree_filter_enabled": enabled,
        "user_degree_filter_allowed_degree_types": allowed_degree_types,
        "user_degree_filter_required_deterministic_cip_prefixes": required_cip_prefixes,
        "user_degree_filter_include_no_degree": include_no_degree,
    }
    if shard_spec:
        stats.update(
            {
                "user_shard_count": int(shard_spec["user_shard_count"]),
                "user_shard_id": int(shard_spec["user_shard_id"]),
                "user_shard_label": str(shard_spec["user_shard_label"]),
            }
        )
    if not enabled:
        print(f"[{STAGE_NAME}] User degree filter disabled")
        return stats

    source_paths = _resolve_filter_source_paths(cfg, stage_cfg)
    raw_users_path = source_paths["raw_users"]
    raw_positions_path = source_paths["raw_positions"]
    legacy_indiv_path = source_paths["legacy_rev_indiv"]
    legacy_educ_path = source_paths["legacy_rev_educ"]
    legacy_pos_path = source_paths["legacy_rev_pos"]
    allowed_predicate = _degree_type_array_has_allowed_predicate("educ.deterministic_degree_types", allowed_degree_types)
    informative_predicate = _degree_type_array_has_informative_predicate("educ.deterministic_degree_types")
    required_cip_predicate = (
        _cip_candidate_array_has_required_prefix_predicate(
            "educ.deterministic_cip_candidates",
            required_cip_prefixes,
        )
        if required_cip_prefixes
        else ""
    )
    keep_predicate = "has_allowed_degree = 1"
    if include_no_degree:
        keep_predicate += " OR has_informative_degree = 0"
    keep_predicate = f"({keep_predicate})"
    if required_cip_prefixes:
        keep_predicate += " AND has_required_cip = 1"
    educ_required_cols_sql = ", deterministic_cip_candidates" if required_cip_prefixes else ""
    agg_required_sql = (
        f", MAX(CASE WHEN {required_cip_predicate} THEN 1 ELSE 0 END) AS has_required_cip"
        if required_cip_prefixes
        else ""
    )

    con = get_duckdb_connection()
    if raw_users_path and raw_positions_path:
        source_mode = "raw_users"
        print(
            f"[{STAGE_NAME}] Preparing degree-filtered raw inputs "
            f"(users={raw_users_path}, positions={raw_positions_path})"
        )
        all_users_count_query = (
            f"SELECT COUNT(DISTINCT TRY_CAST(user_id AS BIGINT)) "
            f"FROM read_parquet('{escape_sql_literal(raw_users_path)}') "
            f"WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL"
            f"{shard_where_sql}"
        )
        raw_cols = set(_describe_parquet_columns(raw_users_path))
        triple_map_path = resolve_existing_path(stage_cfg.get("deterministic_triple_map_input_parquet"))
        use_triple_map = triple_map_path is not None
        if not use_triple_map and "deterministic_degree_types" not in raw_cols:
            raise ValueError(
                f"{STAGE_NAME} user degree filter requires `deterministic_degree_types` on raw users input: "
                f"{raw_users_path}"
            )
        if not use_triple_map and required_cip_prefixes and "deterministic_cip_candidates" not in raw_cols:
            raise ValueError(
                f"{STAGE_NAME} user degree filter requires `deterministic_cip_candidates` on raw users input "
                f"when CIP prefixes are configured: {raw_users_path}"
            )
        if use_triple_map:
            triple_cols = set(_describe_parquet_columns(triple_map_path))
            schema_con = get_duckdb_connection()
            triple_schema = {
                str(row[0]): str(row[1])
                for row in schema_con.sql(
                    f"DESCRIBE SELECT * FROM read_parquet('{escape_sql_literal(triple_map_path)}')"
                ).fetchall()
            }
            schema_con.close()
            degree_types_col = (
                "candidate_degree_types"
                if "candidate_degree_types" in triple_cols
                else "degree_types"
                if "degree_types" in triple_cols
                else None
            )
            cip_candidates_col = (
                "candidate_cip_codes"
                if "candidate_cip_codes" in triple_cols
                else "cip_candidates"
                if "cip_candidates" in triple_cols
                else None
            )
            degree_element_expr = "x.degree_type"
            if degree_types_col is not None and "STRUCT(" not in triple_schema.get(degree_types_col, ""):
                degree_element_expr = "x"
            cip_score_expr = "x.hybrid_score"
            if cip_candidates_col is not None and "hybrid_score" not in triple_schema.get(cip_candidates_col, ""):
                cip_score_expr = "x.score"
            educ_select_sql = f"""
                SELECT
                    TRY_CAST(u.user_id AS BIGINT) AS user_id,
                    CASE
                        WHEN __DEGREE_COL__ IS NULL THEN NULL::VARCHAR[]
                        ELSE list_filter(
                            list_transform(
                                __DEGREE_COL__,
                                x -> lower(trim(CAST(__DEGREE_VALUE__ AS VARCHAR)))
                            ),
                            x -> x IS NOT NULL AND x != ''
                        )
                    END AS deterministic_degree_types
                    {"," if required_cip_prefixes else ""}
                    {"CASE WHEN __CIP_COL__ IS NULL THEN NULL::STRUCT(cip_code VARCHAR, hybrid_score DOUBLE)[] ELSE list_transform(__CIP_COL__, x -> struct_pack(cip_code := CAST(x.cip_code AS VARCHAR), hybrid_score := TRY_CAST(__CIP_SCORE__ AS DOUBLE))) END AS deterministic_cip_candidates" if required_cip_prefixes else ""}
                FROM read_parquet('{escape_sql_literal(raw_users_path)}') AS u
                LEFT JOIN read_parquet('{escape_sql_literal(triple_map_path)}') AS dtm
                  ON CAST(dtm.degree_key AS VARCHAR) = NULLIF({_normalize_key_sql('u.degree_raw')}, '')
                 AND CAST(dtm.field_key AS VARCHAR) = NULLIF({_normalize_key_sql('u.field_raw')}, '')
                 AND CAST(dtm.inst_key AS VARCHAR) = NULLIF({_normalize_key_sql('u.university_raw')}, '')
                WHERE TRY_CAST(u.user_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(u.education_number AS BIGINT) IS NOT NULL
            """
            educ_select_sql = educ_select_sql.replace("__DEGREE_COL__", f"dtm.{degree_types_col}" if degree_types_col else "NULL")
            educ_select_sql = educ_select_sql.replace("__DEGREE_VALUE__", degree_element_expr)
            educ_select_sql = educ_select_sql.replace("__CIP_COL__", f"dtm.{cip_candidates_col}" if cip_candidates_col else "NULL")
            educ_select_sql = educ_select_sql.replace("__CIP_SCORE__", cip_score_expr)
        else:
            educ_select_sql = f"""
                SELECT
                    TRY_CAST(user_id AS BIGINT) AS user_id,
                    deterministic_degree_types
                    {educ_required_cols_sql}
                FROM read_parquet('{escape_sql_literal(raw_users_path)}')
                WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(education_number AS BIGINT) IS NOT NULL
            """
        keep_users_query = f"""
            WITH users AS (
                SELECT DISTINCT
                    TRY_CAST(user_id AS BIGINT) AS user_id
                FROM read_parquet('{escape_sql_literal(raw_users_path)}')
                WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL
            ),
            educ AS (
                {educ_select_sql}
            ),
            agg AS (
                SELECT
                    users.user_id,
                    MAX(CASE WHEN {allowed_predicate} THEN 1 ELSE 0 END) AS has_allowed_degree,
                    MAX(CASE WHEN {informative_predicate} THEN 1 ELSE 0 END) AS has_informative_degree
                    {agg_required_sql}
                FROM users
                LEFT JOIN educ
                  ON educ.user_id = users.user_id
                GROUP BY users.user_id
            )
            SELECT user_id
            FROM agg
            WHERE {keep_predicate}
              {shard_where_sql}
        """
        filtered_users_path = workdir / "wrds_users_degree_filtered.parquet"
        _copy_filtered_user_rows(
            con=con,
            source_path=raw_users_path,
            out_path=filtered_users_path,
            keep_users_query=keep_users_query,
        )
        stage_cfg["wrds_users_input_parquet"] = str(filtered_users_path)
        stage_cfg["name_source_parquet"] = str(filtered_users_path)
        filtered_positions_path = workdir / "wrds_positions_degree_filtered.parquet"
        _copy_filtered_user_rows(
            con=con,
            source_path=raw_positions_path,
            out_path=filtered_positions_path,
            keep_users_query=keep_users_query,
        )
        stage_cfg["wrds_positions_input_parquet"] = str(filtered_positions_path)
    elif legacy_indiv_path and legacy_educ_path:
        source_mode = "legacy_fallback"
        print(
            f"[{STAGE_NAME}] Preparing degree-filtered legacy inputs "
            f"(indiv={legacy_indiv_path}, educ={legacy_educ_path}, pos={legacy_pos_path})"
        )
        legacy_educ_cols = set(_describe_parquet_columns(legacy_educ_path))
        if "deterministic_degree_types" not in legacy_educ_cols:
            raise ValueError(
                f"{STAGE_NAME} user degree filter requires `deterministic_degree_types` on legacy education input: "
                f"{legacy_educ_path}"
            )
        if required_cip_prefixes and "deterministic_cip_candidates" not in legacy_educ_cols:
            raise ValueError(
                f"{STAGE_NAME} user degree filter requires `deterministic_cip_candidates` on legacy education input "
                f"when CIP prefixes are configured: {legacy_educ_path}"
            )
        all_users_count_query = (
            f"SELECT COUNT(DISTINCT TRY_CAST(user_id AS BIGINT)) "
            f"FROM read_parquet('{escape_sql_literal(legacy_indiv_path)}') "
            f"WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL"
            f"{shard_where_sql}"
        )
        keep_users_query = f"""
            WITH users AS (
                SELECT DISTINCT
                    TRY_CAST(user_id AS BIGINT) AS user_id
                FROM read_parquet('{escape_sql_literal(legacy_indiv_path)}')
                WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL
            ),
            educ AS (
                SELECT
                    TRY_CAST(user_id AS BIGINT) AS user_id,
                    deterministic_degree_types
                    {educ_required_cols_sql}
                FROM read_parquet('{escape_sql_literal(legacy_educ_path)}')
                WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL
            ),
            agg AS (
                SELECT
                    users.user_id,
                    MAX(CASE WHEN {allowed_predicate} THEN 1 ELSE 0 END) AS has_allowed_degree,
                    MAX(CASE WHEN {informative_predicate} THEN 1 ELSE 0 END) AS has_informative_degree
                    {agg_required_sql}
                FROM users
                LEFT JOIN educ
                  ON educ.user_id = users.user_id
                GROUP BY users.user_id
            )
            SELECT user_id
            FROM agg
            WHERE {keep_predicate}
              {shard_where_sql}
        """
        filtered_indiv_path = workdir / "legacy_rev_indiv_degree_filtered.parquet"
        filtered_educ_path = workdir / "legacy_rev_educ_degree_filtered.parquet"
        _copy_filtered_user_rows(
            con=con,
            source_path=legacy_indiv_path,
            out_path=filtered_indiv_path,
            keep_users_query=keep_users_query,
        )
        _copy_filtered_user_rows(
            con=con,
            source_path=legacy_educ_path,
            out_path=filtered_educ_path,
            keep_users_query=keep_users_query,
        )
        stage_cfg["legacy_rev_indiv_parquet"] = str(filtered_indiv_path)
        stage_cfg["legacy_rev_educ_long_parquet"] = str(filtered_educ_path)
        stage_cfg["name_source_parquet"] = str(filtered_indiv_path)
        if legacy_pos_path:
            filtered_pos_path = workdir / "legacy_rev_pos_degree_filtered.parquet"
            _copy_filtered_user_rows(
                con=con,
                source_path=legacy_pos_path,
                out_path=filtered_pos_path,
                keep_users_query=keep_users_query,
            )
            stage_cfg["legacy_rev_pos_parquet"] = str(filtered_pos_path)
    else:
        raise FileNotFoundError(
            f"{STAGE_NAME} user_degree_filter_enabled=true, but no usable raw or legacy inputs were found."
        )

    total_users = int(con.sql(all_users_count_query).fetchone()[0])
    kept_users = int(con.sql(f"SELECT COUNT(*) FROM ({keep_users_query})").fetchone()[0])
    dropped_users = max(total_users - kept_users, 0)
    if kept_users <= 0:
        raise ValueError(
            f"{STAGE_NAME} user degree filter removed every user. "
            f"Allowed deterministic degree types: {allowed_degree_types}; include_no_degree={include_no_degree}"
        )

    stats.update(
        {
            "user_degree_filter_source_mode": source_mode,
            "user_degree_filter_total_users": total_users,
            "user_degree_filter_kept_users": kept_users,
            "user_degree_filter_dropped_users": dropped_users,
        }
    )
    print(
        f"[{STAGE_NAME}] User degree filter kept {kept_users:,} / {total_users:,} users "
        f"(allowed={allowed_degree_types}, required_cip_prefixes={required_cip_prefixes or '[]'}, "
        f"include_no_degree={include_no_degree}, source={source_mode}) "
        f"in {_fmt_elapsed(time.perf_counter() - t0)}"
    )
    return stats


def build_clean_users(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    testing: bool | None = None,
    run_name2nat_models: bool = True,
    run_nametrace_model: bool = True,
    use_mock_name_models: bool | None = None,
    in_memory_only: bool = False,
    return_session: bool = False,
    shard_count: int | None = None,
    shard_id: int | None = None,
) -> dict[str, Any] | Stage04Session:
    stage_t0 = time.perf_counter()
    cfg = copy.deepcopy(pipeline_cfg or load_config(config_path))
    shard_spec = _configure_user_shard(cfg, shard_count=shard_count, shard_id=shard_id)
    stage_cfg = get_stage_config(cfg, STAGE_NAME)
    effective_testing = coerce_bool(cfg.get("testing", {}).get("enabled"), False) if testing is None else bool(testing)
    _guard_unfiltered_raw_stage_run(cfg, testing=effective_testing)

    outputs: dict[str, Any] = {
        "testing": effective_testing,
        "stage_status": stage_cfg.get("status"),
    }
    if shard_spec:
        outputs.update(
            {
                "user_shard_count": int(shard_spec["user_shard_count"]),
                "user_shard_id": int(shard_spec["user_shard_id"]),
                "user_shard_label": str(shard_spec["user_shard_label"]),
            }
        )
    print(
        f"[{STAGE_NAME}] build_clean_users start "
        f"(testing={effective_testing}, run_name2nat_models={run_name2nat_models}, "
        f"run_nametrace_model={run_nametrace_model}, in_memory_only={in_memory_only}, "
        f"return_session={return_session}, "
        f"user_shard={shard_spec['user_shard_label'] if shard_spec else 'off'})"
    )
    cleanup = ExitStack()
    try:
        filter_workdir = Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix="stage04_degree_filter_")))
        filter_t0 = time.perf_counter()
        outputs.update(_prepare_degree_filtered_stage_inputs(cfg, workdir=filter_workdir))
        print(f"[{STAGE_NAME}] Degree-filter preparation finished in {_fmt_elapsed(time.perf_counter() - filter_t0)}")

        if run_name2nat_models:
            t0 = time.perf_counter()
            print(f"[{STAGE_NAME}] Running local name2nat step")
            outputs.update(
                run_name2nat(
                    config_path=config_path,
                    pipeline_cfg=cfg,
                    testing=effective_testing,
                    use_mock=use_mock_name_models,
                )
            )
            print(f"[{STAGE_NAME}] Local name2nat step finished in {_fmt_elapsed(time.perf_counter() - t0)}")

        if run_nametrace_model:
            t0 = time.perf_counter()
            print(f"[{STAGE_NAME}] Running local nametrace step")
            outputs.update(
                run_nametrace(
                    config_path=config_path,
                    pipeline_cfg=cfg,
                    testing=effective_testing,
                    use_mock=use_mock_name_models,
                )
            )
            print(f"[{STAGE_NAME}] Local nametrace step finished in {_fmt_elapsed(time.perf_counter() - t0)}")

        t0 = time.perf_counter()
        print(f"[{STAGE_NAME}] Building cleaned user, education, position, and match-ready artifacts")
        assembly_result = assemble_clean_user_artifacts(
            config_path=config_path,
            pipeline_cfg=cfg,
            testing=effective_testing,
            in_memory_only=in_memory_only or return_session,
            return_session=return_session,
        )
        if isinstance(assembly_result, Stage04Session):
            assembly_result.outputs.update(outputs)
            print(f"[{STAGE_NAME}] Artifact assembly finished in {_fmt_elapsed(time.perf_counter() - t0)}")
            print(f"[{STAGE_NAME}] build_clean_users complete in {_fmt_elapsed(time.perf_counter() - stage_t0)}")
            return assembly_result
        outputs.update(assembly_result)
        print(f"[{STAGE_NAME}] Artifact assembly finished in {_fmt_elapsed(time.perf_counter() - t0)}")

        for key in (
            "rev_users_core_parquet",
            "rev_educ_clean_long_parquet",
            "rev_pos_clean_long_parquet",
            "rev_match_ready_parquet",
        ):
            if key in outputs:
                print(f"[{STAGE_NAME}] {key} -> {outputs[key]}")
        if outputs.get("in_memory_only"):
            print(f"[{STAGE_NAME}] In-memory tables -> {sorted(outputs.get('duckdb_tables', {}).keys())}")
        print(f"[{STAGE_NAME}] build_clean_users complete in {_fmt_elapsed(time.perf_counter() - stage_t0)}")
        return outputs
    finally:
        cleanup.close()


def run(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    testing: bool | None = None,
) -> dict[str, Any]:
    return build_clean_users(
        config_path=config_path,
        pipeline_cfg=pipeline_cfg,
        testing=testing,
    )


def _get_stage04_connection(session_or_connection: Stage04Session | Any):
    if isinstance(session_or_connection, Stage04Session):
        return session_or_connection.connection
    return session_or_connection


def _running_in_ipykernel() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


def _publish_interactive_namespace(namespace: dict[str, Any]) -> None:
    globals().update(namespace)
    try:
        from IPython import get_ipython
    except ImportError:
        return
    shell = get_ipython()
    user_ns = getattr(shell, "user_ns", None)
    if isinstance(user_ns, dict):
        user_ns.update(namespace)


def _format_match_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_match_ready_details(
    session_or_connection: Stage04Session | Any,
    *,
    user_id: int,
    education_number: int,
) -> None:
    con = _get_stage04_connection(session_or_connection)
    detail_row = con.execute(
        """
        WITH unitid_display AS (
            SELECT unitid, alias_norm AS ipeds_name
            FROM (
                SELECT
                    unitid,
                    alias_norm,
                    ROW_NUMBER() OVER (
                        PARTITION BY unitid
                        ORDER BY LENGTH(alias_norm), alias_norm
                    ) AS rn
                FROM unitid_aliases
            )
            WHERE rn = 1
        )
        SELECT
            uc.fullname,
            uc.fullname_clean,
            uc.top_country_candidate,
            uc.top_country_score,
            uc.country_uncertain_ind,
            ec.degree_raw,
            ec.degree_clean,
            ec.deterministic_degree_types,
            ec.field_raw,
            ec.field_clean,
            ec.university_raw,
            ec.unitid,
            ec.unitid_mapping_source,
            ec.unitid_selection_reason,
            ec.unitid_score,
            ec.school_match_score,
            ec.unitid_stage03,
            ec.school_match_score_stage03,
            ec.unitid_top1,
            ec.unitid_top1_score,
            ec.unitid_candidate_count,
            ud.ipeds_name,
            ec.cip,
            ctl.title_norm,
            ec.cip_mapping_source,
            ec.cip_score,
            ec.cip_stage03,
            ec.cip_stage03_candidate_score,
            ec.cip_top1,
            ec.cip_top1_score,
            ec.cip_candidate_count
        FROM educ_clean AS ec
        LEFT JOIN users_core AS uc
          ON uc.user_id = ec.user_id
        LEFT JOIN unitid_display AS ud
          ON ud.unitid = ec.unitid
        LEFT JOIN cip_title_lookup AS ctl
          ON ctl.cip = ec.cip
        WHERE ec.user_id = ? AND ec.education_number = ?
        """,
        [int(user_id), int(education_number)],
    ).fetchone()
    if detail_row is None:
        print(f"[{STAGE_NAME}] No educ_clean row found for user_id={user_id}, education_number={education_number}")
        return

    (
        fullname,
        fullname_clean,
        top_country_candidate,
        top_country_score,
        country_uncertain_ind,
        degree_raw,
        degree_clean,
        deterministic_degree_types,
        field_raw,
        field_clean,
        university_raw,
        unitid,
        unitid_mapping_source,
        unitid_selection_reason,
        unitid_score,
        school_match_score,
        unitid_stage03,
        school_match_score_stage03,
        unitid_top1,
        unitid_top1_score,
        unitid_candidate_count,
        ipeds_name,
        cip,
        cip_title,
        cip_mapping_source,
        cip_score,
        cip_stage03,
        cip_stage03_candidate_score,
        cip_top1,
        cip_top1_score,
        cip_candidate_count,
    ) = detail_row

    country_rows = con.execute(
        """
        SELECT
            country_rank,
            country_candidate,
            country_score,
            nanat_score,
            institution_score,
            nametrace_score,
            nanat_subregion_score,
            nt_subregion_score,
            subregion_candidate,
            country_uncertain_ind
        FROM country_candidates
        WHERE user_id = ?
        ORDER BY country_rank
        """,
        [int(user_id)],
    ).fetchall()
    cip_rows = con.execute(
        """
        SELECT
            r.candidate_rank,
            r.cip,
            c.title_norm AS cip_title,
            r.candidate_score,
            r.text_match_rank,
            r.original_rank,
            r.candidate_count,
            CASE WHEN ec.cip = r.cip THEN 1 ELSE 0 END AS selected_ind
        FROM cip_candidate_rows AS r
        JOIN educ_clean AS ec
          ON ec.user_id = r.user_id
         AND ec.education_number = r.education_number
        LEFT JOIN cip_title_lookup AS c
          ON c.cip = r.cip
        WHERE r.user_id = ? AND r.education_number = ?
        ORDER BY r.candidate_rank
        """,
        [int(user_id), int(education_number)],
    ).fetchall()
    unitid_rows = con.execute(
        """
        WITH unitid_display AS (
            SELECT unitid, alias_norm AS ipeds_name
            FROM (
                SELECT
                    unitid,
                    alias_norm,
                    ROW_NUMBER() OVER (
                        PARTITION BY unitid
                        ORDER BY LENGTH(alias_norm), alias_norm
                    ) AS rn
                FROM unitid_aliases
            )
            WHERE rn = 1
        )
        SELECT
            r.candidate_rank,
            r.unitid,
            u.ipeds_name,
            r.candidate_score,
            r.text_match_rank,
            r.alias_jw_max,
            r.alias_token_overlap_max,
            r.alias_shared_token_count_max,
            r.selection_bucket,
            r.original_rank,
            r.candidate_count,
            CASE WHEN ec.unitid = r.unitid THEN 1 ELSE 0 END AS selected_ind
        FROM unitid_candidate_rows AS r
        JOIN educ_clean AS ec
          ON ec.user_id = r.user_id
         AND ec.education_number = r.education_number
        LEFT JOIN unitid_display AS u
          ON u.unitid = r.unitid
        WHERE r.user_id = ? AND r.education_number = ?
        ORDER BY r.candidate_rank
        """,
        [int(user_id), int(education_number)],
    ).fetchall()

    print("")
    print(f"=== Match Detail: user_id={user_id}, education_number={education_number} ===")
    print(f"fullname: {_format_match_value(fullname)}")
    print(f"fullname_clean: {_format_match_value(fullname_clean)}")
    print(
        "top_country: "
        f"{_format_match_value(top_country_candidate)} "
        f"(score={_format_match_value(top_country_score)}, uncertain={_format_match_value(country_uncertain_ind)})"
    )
    print(f"degree_raw: {_format_match_value(degree_raw)}")
    print(f"matched_degree: {_format_match_value(degree_clean)}")
    print(f"matched_degree_types: {_format_match_value(deterministic_degree_types)}")
    print(f"field_raw: {_format_match_value(field_raw)}")
    print(f"matched_field: {_format_match_value(field_clean)}")
    print(f"university_raw: {_format_match_value(university_raw)}")
    print(
        "selected_unitid: "
        f"{_format_match_value(unitid)} "
        f"name={_format_match_value(ipeds_name)} "
        f"mapping={_format_match_value(unitid_mapping_source)} "
        f"reason={_format_match_value(unitid_selection_reason)} "
        f"score={_format_match_value(unitid_score)} "
        f"school_match_score={_format_match_value(school_match_score)} "
        f"stage03_unitid={_format_match_value(unitid_stage03)} "
        f"stage03_score={_format_match_value(school_match_score_stage03)} "
        f"top1_unitid={_format_match_value(unitid_top1)} "
        f"top1_score={_format_match_value(unitid_top1_score)} "
        f"candidate_count={_format_match_value(unitid_candidate_count)}"
    )
    print(
        "selected_cip: "
        f"{_format_match_value(cip)} "
        f"title={_format_match_value(cip_title)} "
        f"mapping={_format_match_value(cip_mapping_source)} "
        f"score={_format_match_value(cip_score)} "
        f"stage03_cip={_format_match_value(cip_stage03)} "
        f"stage03_score={_format_match_value(cip_stage03_candidate_score)} "
        f"top1_cip={_format_match_value(cip_top1)} "
        f"top1_score={_format_match_value(cip_top1_score)} "
        f"candidate_count={_format_match_value(cip_candidate_count)}"
    )
    print("country_candidates:")
    for row in country_rows:
        (
            country_rank,
            country_candidate,
            country_score,
            nanat_score,
            institution_score,
            nametrace_score,
            nanat_subregion_score,
            nt_subregion_score,
            subregion_candidate,
            row_uncertain_ind,
        ) = row
        print(
            f"  [{country_rank}] {_format_match_value(country_candidate)} "
            f"score={_format_match_value(country_score)} "
            f"nanat={_format_match_value(nanat_score)} "
            f"inst={_format_match_value(institution_score)} "
            f"nametrace={_format_match_value(nametrace_score)} "
            f"nanat_subregion={_format_match_value(nanat_subregion_score)} "
            f"nt_subregion={_format_match_value(nt_subregion_score)} "
            f"subregion={_format_match_value(subregion_candidate)} "
            f"uncertain={_format_match_value(row_uncertain_ind)}"
        )
    print("cip_candidates:")
    if not cip_rows:
        print("  <none>")
    for row in cip_rows:
        candidate_rank, candidate_cip, candidate_title, candidate_score, text_match_rank, original_rank, candidate_count, selected_ind = row
        marker = "*" if selected_ind else " "
        print(
            f" {marker}[{candidate_rank}] cip={_format_match_value(candidate_cip)} "
            f"title={_format_match_value(candidate_title)} "
            f"candidate_score={_format_match_value(candidate_score)} "
            f"text_match_rank={_format_match_value(text_match_rank)} "
            f"original_rank={_format_match_value(original_rank)} "
            f"candidate_count={_format_match_value(candidate_count)}"
        )
    print("unitid_candidates:")
    if not unitid_rows:
        print("  <none>")
    for row in unitid_rows:
        (
            candidate_rank,
            candidate_unitid,
            candidate_name,
            candidate_score,
            text_match_rank,
            alias_jw_max,
            alias_token_overlap_max,
            alias_shared_token_count_max,
            selection_bucket,
            original_rank,
            candidate_count,
            selected_ind,
        ) = row
        marker = "*" if selected_ind else " "
        print(
            f" {marker}[{candidate_rank}] unitid={_format_match_value(candidate_unitid)} "
            f"name={_format_match_value(candidate_name)} "
            f"candidate_score={_format_match_value(candidate_score)} "
            f"text_match_rank={_format_match_value(text_match_rank)} "
            f"alias_jw_max={_format_match_value(alias_jw_max)} "
            f"alias_overlap_max={_format_match_value(alias_token_overlap_max)} "
            f"alias_shared_tokens={_format_match_value(alias_shared_token_count_max)} "
            f"selection_bucket={_format_match_value(selection_bucket)} "
            f"original_rank={_format_match_value(original_rank)} "
            f"candidate_count={_format_match_value(candidate_count)}"
        )


def print_match_ready_sample_details(
    session_or_connection: Stage04Session | Any,
    *,
    sample_size: int = 5,
    seed: int | None = None,
) -> list[tuple[int, int]]:
    con = _get_stage04_connection(session_or_connection)
    limit_n = max(1, int(sample_size))
    if seed is None:
        order_sql = "ORDER BY random()"
    else:
        order_sql = f"ORDER BY hash(TRY_CAST(user_id AS BIGINT), TRY_CAST(education_number AS BIGINT), {int(seed)})"
    rows = con.execute(
        f"""
        SELECT DISTINCT user_id, education_number
        FROM match_ready
        WHERE user_id IS NOT NULL
          AND education_number IS NOT NULL
        {order_sql}
        LIMIT {limit_n}
        """
    ).fetchall()
    sample_keys = [(int(row[0]), int(row[1])) for row in rows]
    if not sample_keys:
        print(f"[{STAGE_NAME}] match_ready is empty; no rows to sample.")
        return []
    for user_id, education_number in sample_keys:
        print_match_ready_details(
            con,
            user_id=user_id,
            education_number=education_number,
        )
    return sample_keys


def build_clean_users_session(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    testing: bool | None = None,
    run_name2nat_models: bool = False,
    run_nametrace_model: bool = False,
    use_mock_name_models: bool | None = None,
    shard_count: int | None = None,
    shard_id: int | None = None,
) -> Stage04Session:
    result = build_clean_users(
        config_path=config_path,
        pipeline_cfg=pipeline_cfg,
        testing=testing,
        run_name2nat_models=run_name2nat_models,
        run_nametrace_model=run_nametrace_model,
        use_mock_name_models=use_mock_name_models,
        in_memory_only=True,
        return_session=True,
        shard_count=shard_count,
        shard_id=shard_id,
    )
    if not isinstance(result, Stage04Session):
        raise TypeError(f"{STAGE_NAME} expected a Stage04Session but received {type(result).__name__}")
    return result


def launch_ipython_session(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    testing: bool | None = None,
    run_name2nat_models: bool = False,
    run_nametrace_model: bool = False,
    use_mock_name_models: bool | None = None,
    shard_count: int | None = None,
    shard_id: int | None = None,
) -> Stage04Session:
    session = build_clean_users_session(
        config_path=config_path,
        pipeline_cfg=pipeline_cfg,
        testing=testing,
        run_name2nat_models=run_name2nat_models,
        run_nametrace_model=run_nametrace_model,
        use_mock_name_models=use_mock_name_models,
        shard_count=shard_count,
        shard_id=shard_id,
    )
    namespace = {
        "session": session,
        "con": session.connection,
        "outputs": session.outputs,
        "tables": session.tables,
        "print_match_ready_details": print_match_ready_details,
        "print_match_ready_sample_details": print_match_ready_sample_details,
    }
    if _running_in_ipykernel():
        _publish_interactive_namespace(namespace)
        print(
            f"[{STAGE_NAME}] Detected ipykernel; published session, con, outputs, tables, "
            f"print_match_ready_details, and print_match_ready_sample_details to the interactive namespace."
        )
        return session
    try:
        from IPython import start_ipython
    except ImportError as exc:
        session.close()
        raise RuntimeError("IPython is not installed in this environment.") from exc
    print(
        f"[{STAGE_NAME}] Launching IPython with session, con, outputs, tables "
        f"({len(session.tables)} tables)"
    )
    try:
        start_ipython(argv=[], user_ns=namespace)
    finally:
        session.close()
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 04_rev_user_clean stage.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-name2nat", action="store_true")
    parser.add_argument("--skip-nametrace", action="store_true")
    parser.add_argument("--use-mock-name-models", action="store_true")
    parser.add_argument("--in-memory-only", action="store_true")
    parser.add_argument("--ipython", action="store_true")
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--merge-shards", action="store_true")
    testing_group = parser.add_mutually_exclusive_group()
    testing_group.add_argument("--testing", dest="testing", action="store_true")
    testing_group.add_argument("--no-testing", dest="testing", action="store_false")
    parser.set_defaults(testing=None)
    clean_argv = sanitize_ipykernel_argv()
    args = parser.parse_args(clean_argv)
    auto_ipython = _running_in_ipykernel() and not clean_argv

    cfg = load_config(args.config)
    effective_testing = coerce_bool(cfg.get("testing", {}).get("enabled"), False) if args.testing is None else bool(args.testing)
    sharding_requested = bool(args.shard_count is not None or args.shard_id is not None)
    if not sharding_requested:
        stage_shard_spec = resolve_user_shard_spec(get_stage_config(cfg, STAGE_NAME))
        sharding_requested = stage_shard_spec is not None
    t0 = time.perf_counter()
    if args.merge_shards:
        if args.shard_id is not None:
            raise ValueError("`--merge-shards` does not accept `--shard-id`.")
        merge_outputs = _merge_stage04_sharded_outputs(
            config_path=args.config,
            pipeline_cfg=cfg,
            shard_count=args.shard_count,
        )
        print(f"[{STAGE_NAME}] merged shard outputs: {merge_outputs.get('merged_outputs', [])}")
        if not effective_testing and not args.in_memory_only:
            mark_stage_complete(STAGE_NAME, time.perf_counter() - t0)
        return
    if args.ipython or auto_ipython:
        if auto_ipython and not args.ipython:
            print(f"[{STAGE_NAME}] Detected ipykernel execution; auto-launching interactive session setup.")
        else:
            print("Launching IPython session instead of running full stage execution...")
        launch_ipython_session(
            config_path=args.config,
            pipeline_cfg=cfg,
            testing=True,
            run_name2nat_models=False,
            run_nametrace_model=False,
            use_mock_name_models=args.use_mock_name_models,
            shard_count=args.shard_count,
            shard_id=args.shard_id,
        )
        return
    build_clean_users(
        config_path=args.config,
        pipeline_cfg=cfg,
        testing=args.testing,
        run_name2nat_models=not args.skip_name2nat,
        run_nametrace_model=not args.skip_nametrace,
        use_mock_name_models=args.use_mock_name_models,
        in_memory_only=args.in_memory_only,
        shard_count=args.shard_count,
        shard_id=args.shard_id,
    )
    if not effective_testing and not args.in_memory_only and not sharding_requested:
        mark_stage_complete(STAGE_NAME, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
