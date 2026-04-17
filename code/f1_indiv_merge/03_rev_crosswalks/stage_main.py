"""Build school, field, and employer crosswalks for stage 03_rev_crosswalks."""

from __future__ import annotations

import argparse
import os
import sys
import time
from builtins import print as _print
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import duckdb

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
for _path in (PIPELINE_ROOT, REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from external_us_school_matching import stage_external_us_school_matching_artifacts
from helpers import cip_code_to_cip4_sql, field_clean_regex_sql, field_clean_to_cip4_sql, inst_clean_regex_sql
from src.config_loader import get_stage_config, load_config
from src.pipeline_runtime import StageDeferredError, coerce_bool, sanitize_ipykernel_argv
from src.progress_tracker import mark_stage_complete

print = partial(_print, flush=True)

STAGE_NAME = "03_rev_crosswalks"
REQUIRED_STAGE_OUTPUTS = {
    "school_family_crosswalk_parquet",
    "school_resolution_parquet",
    "f1_inst_unitid_crosswalk_parquet",
    "employer_lookup_parquet",
    "employer_key_map_parquet",
}


def _escape(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _describe_parquet_columns(con: duckdb.DuckDBPyConnection, path: str | Path) -> list[str]:
    return [
        row[0]
        for row in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{_escape(path)}')"
        ).fetchall()
    ]


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _required_column(
    columns: Iterable[str],
    candidates: Iterable[str],
    *,
    label: str,
    source_path: str | Path,
) -> str:
    selected = _first_present(columns, candidates)
    if selected is None:
        raise ValueError(
            f"Missing required {label} in {source_path}. "
            f"Tried columns: {', '.join(candidates)}"
        )
    return selected


def _optional_select(column_name: str | None, alias: str, cast: str = "VARCHAR") -> str:
    if column_name is None:
        return f"NULL::{cast} AS {alias}"
    return f"CAST({column_name} AS {cast}) AS {alias}"


def _alpha_text_sql(expr: str) -> str:
    return (
        "TRIM("
        "REGEXP_REPLACE("
        "REGEXP_REPLACE("
        f"strip_accents(lower(CAST({expr} AS VARCHAR))), "
        "'[^a-z]+', ' ', 'g'"
        "), "
        "'\\s+', ' ', 'g'"
        ")"
        ")"
    )


def _copy_parquet(con: duckdb.DuckDBPyConnection, src: str | Path, dest: str | Path) -> None:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        dest_path.unlink()
    con.sql(
        f"""
        COPY (
            SELECT *
            FROM read_parquet('{_escape(src)}')
        )
        TO '{_escape(dest_path)}' (FORMAT PARQUET)
        """
    )


def _write_query(
    con: duckdb.DuckDBPyConnection,
    *,
    query: str,
    out_path: str | Path,
) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()
    con.sql(f"COPY ({query}) TO '{_escape(out_file)}' (FORMAT PARQUET)")


def _require_output_columns(
    con: duckdb.DuckDBPyConnection,
    path: str | Path,
    required_cols: Iterable[str],
    *,
    label: str,
) -> None:
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"{label} was not written: {path_obj}")
    cols = set(_describe_parquet_columns(con, path_obj))
    missing = [col for col in required_cols if col not in cols]
    if missing:
        raise ValueError(
            f"{label} is missing required columns {missing}. "
            f"Observed columns: {sorted(cols)}"
        )
    n_rows = int(
        con.sql(
            f"SELECT COUNT(*) FROM read_parquet('{_escape(path_obj)}')"
        ).fetchone()[0]
    )
    if n_rows <= 0:
        raise ValueError(f"{label} is empty: {path_obj}")


def _load_local_school_builder(legacy_config_path: str | None):
    if legacy_config_path:
        os.environ["F1_INDIV_MERGE_CONFIG"] = legacy_config_path
    import importlib

    import deps_f1_school_crosswalk

    return importlib.reload(deps_f1_school_crosswalk)


def _load_local_employer_builder(legacy_config_path: str | None):
    if legacy_config_path:
        os.environ["F1_INDIV_MERGE_CONFIG"] = legacy_config_path
    import importlib

    import deps_f1_employer_crosswalk

    return importlib.reload(deps_f1_employer_crosswalk)


def _resolve_existing_path(*candidates: Any) -> str | None:
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple)):
            nested = _resolve_existing_path(*candidate)
            if nested:
                return nested
            continue
        text = str(candidate).strip()
        if not text:
            continue
        if Path(text).exists():
            return text
    return None


def _resolve_external_cleaned_education_path(stage_cfg: dict[str, Any]) -> str:
    candidates = [
        stage_cfg.get("external_cleaned_education_artifact"),
        *(stage_cfg.get("external_cleaned_education_fallbacks") or []),
    ]
    resolved = _resolve_existing_path(*candidates)
    if resolved:
        return resolved

    configured = list(dict.fromkeys(str(path) for path in candidates if path))
    expected_output = stage_cfg.get("external_cleaned_education_upstream_output") or "rev_educ_clean_long parquet"
    build_hint = stage_cfg.get("external_cleaned_education_upstream_hint") or (
        "Run the revelio-cleaning education-object cleaning step on the stage-02 WRDS users pull, "
        f"then produce {expected_output} before rerunning stage 03."
    )
    checked = ", ".join(configured) if configured else "(no configured paths)"
    raise StageDeferredError(
        f"{STAGE_NAME} cannot run because the required revelio-cleaning output is missing. "
        f"Expected cleaned education artifact: {expected_output}. Checked: {checked}. {build_hint}"
    )


def _maybe_stage_external_school_artifacts(stage_cfg: dict[str, Any]) -> dict[str, Path] | None:
    artifact_paths = stage_cfg.get("external_school_artifacts") or {}
    if not isinstance(artifact_paths, dict) or not artifact_paths:
        return None
    if any(not str(value).strip() for value in artifact_paths.values()):
        return None
    try:
        return stage_external_us_school_matching_artifacts(
            artifact_paths={key: str(value) for key, value in artifact_paths.items()},
            revelio_cleaning_repo_root=stage_cfg.get("revelio_cleaning_repo_root"),
            verbose=True,
        )
    except FileNotFoundError as exc:
        print(f"[{STAGE_NAME}] WARNING: external school artifacts unavailable; using local fallback.")
        print(f"[{STAGE_NAME}] {exc}")
        return None


def _legacy_rev_school_unitid_query(
    con: duckdb.DuckDBPyConnection,
    *,
    stage_cfg: dict[str, Any],
) -> str:
    source_path = _resolve_existing_path(
        stage_cfg.get("legacy_rev_school_unitid_source_parquet"),
        stage_cfg.get("legacy_school_resolution_source_parquet"),
    )
    if source_path is None:
        school_builder = _load_local_school_builder(stage_cfg.get("legacy_config_path"))
        school_builder.build_school_crosswalk(overwrite=coerce_bool(stage_cfg.get("overwrite"), True), con=con)
        source_path = _resolve_existing_path(
            stage_cfg.get("legacy_rev_school_unitid_source_parquet"),
            getattr(school_builder.cfg, "F1_REVELIO_IPEDS_RESOLUTION_PARQUET", ""),
        )
    if source_path is None:
        raise ValueError("Unable to resolve any source for the Revelio school-to-UNITID crosswalk.")

    cols = _describe_parquet_columns(con, source_path)
    university_col = _required_column(
        cols,
        ["university_raw", "rev_university_raw"],
        label="Revelio raw university",
        source_path=source_path,
    )
    unitid_col = _required_column(cols, ["unitid", "UNITID", "main_unitid"], label="UNITID", source_path=source_path)
    rev_clean_col = _first_present(cols, ["rev_instname_clean"])
    f1_row_col = _first_present(cols, ["f1_row_num"])
    ipeds_col = _first_present(cols, ["ipeds_instname_clean", "ipeds_name"])
    source_col = _first_present(cols, ["rev_match_source"])
    match_type_col = _first_present(cols, ["rev_matchtype"])
    match_group_col = _first_present(cols, ["match_group"])
    score_col = _first_present(cols, ["rev_jw_score", "school_match_score", "match_score"])

    return f"""
        WITH base AS (
            SELECT
                NULLIF(TRIM(CAST({university_col} AS VARCHAR)), '') AS university_raw,
                LOWER(NULLIF(TRIM(CAST({university_col} AS VARCHAR)), '')) AS university_raw_key,
                {_optional_select(rev_clean_col, 'rev_instname_clean')},
                TRY_CAST({unitid_col} AS BIGINT) AS unitid,
                {_optional_select(f1_row_col, 'f1_row_num_candidate', cast='BIGINT')},
                {_optional_select(ipeds_col, 'ipeds_instname_clean')},
                {_optional_select(source_col, 'rev_match_source')},
                {_optional_select(match_type_col, 'rev_matchtype')},
                {_optional_select(match_group_col, 'match_group')},
                {_optional_select(score_col, 'school_match_score', cast='DOUBLE')}
            FROM read_parquet('{_escape(source_path)}')
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY university_raw, unitid
                    ORDER BY school_match_score DESC NULLS LAST, f1_row_num_candidate
                ) AS row_rank
            FROM base
            WHERE university_raw IS NOT NULL
              AND unitid IS NOT NULL
        )
        SELECT
            university_raw,
            university_raw_key,
            rev_instname_clean,
            unitid,
            f1_row_num_candidate,
            ipeds_instname_clean,
            rev_match_source,
            rev_matchtype,
            match_group,
            school_match_score,
            'raw_string_level' AS artifact_grain,
            '{_escape(source_path)}' AS source_artifact,
            'legacy_fallback' AS mapping_source,
            NULL::DOUBLE AS deterministic_candidate_score
        FROM ranked
        WHERE row_rank = 1
    """


def _legacy_field_cip_query(
    con: duckdb.DuckDBPyConnection,
    *,
    stage_cfg: dict[str, Any],
    cleaned_education_path: str,
) -> str:
    clean_cols = _describe_parquet_columns(con, cleaned_education_path)
    clean_user_col = _required_column(clean_cols, ["user_id"], label="cleaned user id", source_path=cleaned_education_path)
    clean_educ_col = _required_column(
        clean_cols,
        ["education_number"],
        label="cleaned education_number",
        source_path=cleaned_education_path,
    )
    field_clean_col = _required_column(
        clean_cols,
        ["field_clean"],
        label="field_clean",
        source_path=cleaned_education_path,
    )

    use_raw_wrds = coerce_bool(stage_cfg.get("use_raw_wrds_users_for_field_crosswalk"), False)
    raw_users_path = _resolve_existing_path(stage_cfg.get("raw_wrds_users_parquet")) if use_raw_wrds else None
    if raw_users_path:
        raw_cols = _describe_parquet_columns(con, raw_users_path)
        raw_user_col = _required_column(raw_cols, ["user_id"], label="raw user id", source_path=raw_users_path)
        raw_educ_col = _required_column(
            raw_cols,
            ["education_number"],
            label="raw education_number",
            source_path=raw_users_path,
        )
        raw_field_col = _required_column(raw_cols, ["field_raw"], label="field_raw", source_path=raw_users_path)
        return f"""
            WITH cleaned AS (
                SELECT DISTINCT
                    TRY_CAST({clean_user_col} AS BIGINT) AS user_id,
                    TRY_CAST({clean_educ_col} AS BIGINT) AS education_number,
                    NULLIF(TRIM(CAST({field_clean_col} AS VARCHAR)), '') AS field_clean
                FROM read_parquet('{_escape(cleaned_education_path)}')
            ),
            raw_users AS (
                SELECT DISTINCT
                    TRY_CAST({raw_user_col} AS BIGINT) AS user_id,
                    TRY_CAST({raw_educ_col} AS BIGINT) AS education_number,
                    NULLIF(TRIM(CAST({raw_field_col} AS VARCHAR)), '') AS field_raw
                FROM read_parquet('{_escape(raw_users_path)}')
                WHERE NULLIF(TRIM(CAST({raw_field_col} AS VARCHAR)), '') IS NOT NULL
            ),
            joined AS (
                SELECT
                    ru.field_raw,
                    cl.field_clean,
                    COALESCE(cl.field_clean, ru.field_raw) AS source_field_text
                FROM cleaned AS cl
                LEFT JOIN raw_users AS ru
                  ON ru.user_id = cl.user_id
                 AND ru.education_number = cl.education_number
            ),
            field_values AS (
                SELECT DISTINCT
                    field_raw,
                    field_clean,
                    source_field_text
                FROM joined
                WHERE source_field_text IS NOT NULL
            ),
            normalized AS (
                SELECT
                    field_raw,
                    field_clean,
                    source_field_text,
                    NULLIF(TRIM({field_clean_regex_sql('source_field_text')}), '') AS source_field_norm
                FROM field_values
            ),
            collapsed AS (
                SELECT
                    source_field_norm,
                    MIN(source_field_text) AS source_field_text,
                    MIN(field_raw) AS field_raw,
                    MIN(field_clean) AS field_clean,
                    COUNT(*) AS n_source_strings
                FROM normalized
                WHERE source_field_norm IS NOT NULL
                GROUP BY source_field_norm
            ),
            mapped AS (
                SELECT
                    field_raw,
                    field_clean,
                    source_field_text,
                    source_field_norm,
                    n_source_strings,
                    {field_clean_to_cip4_sql('source_field_norm')} AS cip_code
                FROM collapsed
            )
            SELECT
                field_raw,
                field_clean,
                source_field_text,
                source_field_norm,
                n_source_strings,
                cip_code,
                'normalized_string_level' AS artifact_grain,
                'legacy_raw_plus_cleaned' AS mapping_source,
                NULL::DOUBLE AS deterministic_candidate_score
            FROM mapped
            WHERE cip_code IS NOT NULL
        """

    return f"""
        WITH field_values AS (
            SELECT DISTINCT
                NULL::VARCHAR AS field_raw,
                NULLIF(TRIM(CAST({field_clean_col} AS VARCHAR)), '') AS field_clean,
                NULLIF(TRIM(CAST({field_clean_col} AS VARCHAR)), '') AS source_field_text
            FROM read_parquet('{_escape(cleaned_education_path)}')
            WHERE NULLIF(TRIM(CAST({field_clean_col} AS VARCHAR)), '') IS NOT NULL
        ),
        normalized AS (
            SELECT
                field_raw,
                field_clean,
                source_field_text,
                NULLIF(TRIM({field_clean_regex_sql('source_field_text')}), '') AS source_field_norm
            FROM field_values
        ),
        collapsed AS (
            SELECT
                source_field_norm,
                MIN(source_field_text) AS source_field_text,
                MIN(field_raw) AS field_raw,
                MIN(field_clean) AS field_clean,
                COUNT(*) AS n_source_strings
            FROM normalized
            WHERE source_field_norm IS NOT NULL
            GROUP BY source_field_norm
        ),
        mapped AS (
            SELECT
                field_raw,
                field_clean,
                source_field_text,
                source_field_norm,
                n_source_strings,
                {field_clean_to_cip4_sql('source_field_norm')} AS cip_code
            FROM collapsed
        )
        SELECT
            field_raw,
            field_clean,
            source_field_text,
            source_field_norm,
            n_source_strings,
            cip_code,
            'normalized_string_level' AS artifact_grain,
            'legacy_cleaned_only' AS mapping_source,
            NULL::DOUBLE AS deterministic_candidate_score
        FROM mapped
        WHERE cip_code IS NOT NULL
    """


def _build_f1_inst_unitid_crosswalk(
    con: duckdb.DuckDBPyConnection,
    *,
    stage_cfg: dict[str, Any],
) -> str:
    out_path = stage_cfg["f1_inst_unitid_crosswalk_parquet"]
    source_path = _resolve_existing_path(
        stage_cfg.get("legacy_f1_inst_unitid_source_parquet"),
        stage_cfg.get("legacy_school_resolution_source_parquet"),
    )
    if source_path is None:
        school_builder = _load_local_school_builder(stage_cfg.get("legacy_config_path"))
        school_builder.build_school_crosswalk(overwrite=coerce_bool(stage_cfg.get("overwrite"), True), con=con)
        source_path = _resolve_existing_path(
            stage_cfg.get("legacy_f1_inst_unitid_source_parquet"),
            getattr(school_builder.cfg, "F1_REVELIO_IPEDS_RESOLUTION_PARQUET", ""),
        )
    if source_path is None:
        raise ValueError("Unable to resolve any source for the F1 institution-to-UNITID crosswalk.")

    cols = _describe_parquet_columns(con, source_path)
    f1_row_col = _required_column(cols, ["f1_row_num"], label="F1 row id", source_path=source_path)
    unitid_col = _required_column(cols, ["unitid", "UNITID", "main_unitid"], label="UNITID", source_path=source_path)
    school_col = _first_present(cols, ["school_name", "f1_school_name"])
    inst_clean_col = _first_present(cols, ["f1_instname_clean"])
    city_col = _first_present(cols, ["f1_city_clean"])
    state_col = _first_present(cols, ["f1_state_clean"])
    zip_col = _first_present(cols, ["f1_zip_clean"])
    match_col = _first_present(cols, ["matchtype", "f1_matchtype"])

    query = f"""
        SELECT DISTINCT
            TRY_CAST({f1_row_col} AS BIGINT) AS f1_row_num,
            {_optional_select(school_col, 'school_name')},
            {_optional_select(inst_clean_col, 'f1_instname_clean')},
            {_optional_select(city_col, 'f1_city_clean')},
            {_optional_select(state_col, 'f1_state_clean')},
            {_optional_select(zip_col, 'f1_zip_clean')},
            TRY_CAST({unitid_col} AS BIGINT) AS unitid,
            {_optional_select(match_col, 'match_type')},
            'row_level' AS artifact_grain,
            '{_escape(source_path)}' AS source_artifact
        FROM read_parquet('{_escape(source_path)}')
        WHERE TRY_CAST({f1_row_col} AS BIGINT) IS NOT NULL
          AND TRY_CAST({unitid_col} AS BIGINT) IS NOT NULL
        ORDER BY f1_row_num
    """
    _write_query(con, query=query, out_path=out_path)
    _require_output_columns(
        con,
        out_path,
        ["f1_row_num", "unitid", "school_name", "artifact_grain"],
        label="F1 institution-to-UNITID crosswalk",
    )
    return out_path


def _resolve_school_artifact_sources(
    stage_cfg: dict[str, Any],
    *,
    con: duckdb.DuckDBPyConnection,
) -> tuple[str | None, str | None]:
    external_school_cfg = stage_cfg.get("external_school_artifacts") or {}
    family_source = _resolve_existing_path(
        stage_cfg.get("legacy_school_crosswalk_source_parquet"),
        external_school_cfg.get("f1_revelio_school_crosswalk"),
    )
    resolution_source = _resolve_existing_path(
        stage_cfg.get("legacy_school_resolution_source_parquet"),
        external_school_cfg.get("f1_revelio_ipeds_resolution"),
    )
    if family_source and resolution_source:
        return family_source, resolution_source

    school_builder = _load_local_school_builder(stage_cfg.get("legacy_config_path"))
    school_builder.build_school_crosswalk(
        overwrite=coerce_bool(stage_cfg.get("overwrite"), True),
        con=con,
    )
    family_source = _resolve_existing_path(
        family_source,
        getattr(school_builder.cfg, "F1_REV_SCHOOL_CROSSWALK_PARQUET", ""),
        getattr(school_builder.cfg, "F1_REVELIO_SCHOOL_CROSSWALK_CANONICAL_PARQUET", ""),
    )
    resolution_source = _resolve_existing_path(
        resolution_source,
        getattr(school_builder.cfg, "F1_REV_SCHOOL_RESOLUTION_PARQUET", ""),
        getattr(school_builder.cfg, "F1_REVELIO_IPEDS_RESOLUTION_PARQUET", ""),
    )
    return family_source, resolution_source


def _build_school_family_crosswalk(
    con: duckdb.DuckDBPyConnection,
    *,
    stage_cfg: dict[str, Any],
) -> str:
    out_path = stage_cfg["school_family_crosswalk_parquet"]
    family_source, _ = _resolve_school_artifact_sources(stage_cfg, con=con)
    if family_source is None:
        raise ValueError("Unable to resolve any source for the school-family crosswalk artifact.")
    _copy_parquet(con, family_source, out_path)
    _require_output_columns(
        con,
        out_path,
        ["f1_school_name", "rev_university_raw", "rev_instname_clean", "match_score"],
        label="School-family crosswalk",
    )
    return out_path


def _build_school_resolution(
    con: duckdb.DuckDBPyConnection,
    *,
    stage_cfg: dict[str, Any],
) -> str:
    out_path = stage_cfg["school_resolution_parquet"]
    _, resolution_source = _resolve_school_artifact_sources(stage_cfg, con=con)
    if resolution_source is None:
        raise ValueError("Unable to resolve any source for the row-level school-resolution artifact.")
    _copy_parquet(con, resolution_source, out_path)
    _require_output_columns(
        con,
        out_path,
        ["f1_school_name", "f1_row_num", "UNITID", "rev_university_raw", "rev_instname_clean"],
        label="School-resolution artifact",
    )
    return out_path


def _build_rev_school_unitid_crosswalk(
    con: duckdb.DuckDBPyConnection,
    *,
    stage_cfg: dict[str, Any],
) -> str:
    out_path = stage_cfg["rev_school_unitid_crosswalk_parquet"]
    legacy_query = _legacy_rev_school_unitid_query(con, stage_cfg=stage_cfg)
    raw_users_path = _resolve_existing_path(stage_cfg.get("raw_wrds_users_parquet"))
    ipeds_name_path = _resolve_existing_path(stage_cfg.get("ipeds_name_crosswalk_input_parquet"))
    ipeds_cols = _describe_parquet_columns(con, ipeds_name_path) if ipeds_name_path else []
    ipeds_unit_expr = None
    if "main_unitid" in set(ipeds_cols) and "UNITID" in set(ipeds_cols):
        ipeds_unit_expr = "COALESCE(main_unitid, UNITID)"
    elif "main_unitid" in set(ipeds_cols):
        ipeds_unit_expr = "main_unitid"
    elif "UNITID" in set(ipeds_cols):
        ipeds_unit_expr = "UNITID"
    elif "unitid" in set(ipeds_cols):
        ipeds_unit_expr = "unitid"
    ipeds_name_col = _first_present(ipeds_cols, ["instname", "ipeds_name", "display_name", "ipeds_instname_clean"])
    raw_cols = _describe_parquet_columns(con, raw_users_path) if raw_users_path else []
    has_deterministic_candidates = bool(raw_users_path) and {
        "university_raw",
        "deterministic_inst_candidates",
    }.issubset(set(raw_cols))

    if has_deterministic_candidates and ipeds_name_path and ipeds_unit_expr and ipeds_name_col:
        raw_users_expr = "CAST(university_raw AS VARCHAR)"
        query = f"""
            WITH ipeds_names AS (
                SELECT DISTINCT
                    TRY_CAST({ipeds_unit_expr} AS BIGINT) AS unitid,
                    NULLIF({_alpha_text_sql(ipeds_name_col)}, '') AS name_alpha_norm
                FROM read_parquet('{_escape(ipeds_name_path)}')
                WHERE {ipeds_unit_expr} IS NOT NULL
                  AND {ipeds_name_col} IS NOT NULL
            ),
            deterministic_base AS (
                SELECT DISTINCT
                    NULLIF(TRIM({raw_users_expr}), '') AS university_raw,
                    LOWER(NULLIF(TRIM({raw_users_expr}), '')) AS university_raw_key,
                    NULLIF(TRIM({inst_clean_regex_sql(raw_users_expr)}), '') AS rev_instname_clean,
                    NULLIF({_alpha_text_sql(raw_users_expr)}, '') AS source_alpha_norm,
                    deterministic_inst_candidates
                FROM read_parquet('{_escape(raw_users_path)}')
                WHERE array_length(deterministic_inst_candidates) > 0
                  AND NULLIF(TRIM({raw_users_expr}), '') IS NOT NULL
            ),
            deterministic_expanded AS (
                SELECT DISTINCT
                    university_raw,
                    university_raw_key,
                    rev_instname_clean,
                    source_alpha_norm,
                    TRY_CAST(deterministic_inst_candidates[idx].unitid AS BIGINT) AS unitid,
                    TRY_CAST(deterministic_inst_candidates[idx].hybrid_score AS DOUBLE) AS deterministic_candidate_score,
                    idx AS raw_candidate_rank
                FROM deterministic_base
                CROSS JOIN generate_series(1, array_length(deterministic_inst_candidates)) AS gs(idx)
                WHERE TRY_CAST(deterministic_inst_candidates[idx].unitid AS BIGINT) IS NOT NULL
            ),
            deterministic_scored AS (
                SELECT
                    de.university_raw,
                    de.university_raw_key,
                    de.rev_instname_clean,
                    de.unitid,
                    de.deterministic_candidate_score,
                    de.raw_candidate_rank,
                    COALESCE(
                        MAX(
                            CASE
                                WHEN ip.name_alpha_norm = de.source_alpha_norm THEN 2
                                WHEN list_has_all(string_split(ip.name_alpha_norm, ' '), string_split(de.source_alpha_norm, ' '))
                                  OR list_has_all(string_split(de.source_alpha_norm, ' '), string_split(ip.name_alpha_norm, ' '))
                                    THEN 1
                                ELSE 0
                            END
                        ),
                        0
                    ) AS text_match_rank
                FROM deterministic_expanded AS de
                LEFT JOIN ipeds_names AS ip
                  ON de.unitid = ip.unitid
                GROUP BY
                    de.university_raw,
                    de.university_raw_key,
                    de.rev_instname_clean,
                    de.unitid,
                    de.deterministic_candidate_score,
                    de.raw_candidate_rank
            ),
            deterministic_ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY university_raw_key
                        ORDER BY text_match_rank DESC, deterministic_candidate_score DESC NULLS LAST, raw_candidate_rank, university_raw, unitid
                    ) AS row_rank
                FROM deterministic_scored
            ),
            deterministic_selected AS (
                SELECT
                    university_raw,
                    university_raw_key,
                    rev_instname_clean,
                    unitid,
                    NULL::BIGINT AS f1_row_num_candidate,
                    NULL::VARCHAR AS ipeds_instname_clean,
                    'deterministic_candidates' AS rev_match_source,
                    'deterministic_top1' AS rev_matchtype,
                    'deterministic' AS match_group,
                    deterministic_candidate_score AS school_match_score,
                    'raw_string_level' AS artifact_grain,
                    '{_escape(raw_users_path)}' AS source_artifact,
                    'deterministic_top1' AS mapping_source,
                    deterministic_candidate_score
                FROM deterministic_ranked
                WHERE row_rank = 1
            ),
            legacy_selected AS (
                SELECT *
                FROM ({legacy_query})
            )
            SELECT
                university_raw,
                rev_instname_clean,
                unitid,
                f1_row_num_candidate,
                ipeds_instname_clean,
                rev_match_source,
                rev_matchtype,
                match_group,
                school_match_score,
                artifact_grain,
                source_artifact,
                mapping_source,
                deterministic_candidate_score
            FROM deterministic_selected

            UNION ALL

            SELECT
                legacy_selected.university_raw,
                legacy_selected.rev_instname_clean,
                legacy_selected.unitid,
                legacy_selected.f1_row_num_candidate,
                legacy_selected.ipeds_instname_clean,
                legacy_selected.rev_match_source,
                legacy_selected.rev_matchtype,
                legacy_selected.match_group,
                legacy_selected.school_match_score,
                legacy_selected.artifact_grain,
                legacy_selected.source_artifact,
                legacy_selected.mapping_source,
                legacy_selected.deterministic_candidate_score
            FROM legacy_selected
            LEFT JOIN (
                SELECT DISTINCT university_raw_key
                FROM deterministic_selected
            ) AS det
              ON legacy_selected.university_raw_key = det.university_raw_key
            WHERE det.university_raw_key IS NULL

            ORDER BY university_raw, unitid
        """
    elif has_deterministic_candidates:
        raw_users_expr = "CAST(university_raw AS VARCHAR)"
        query = f"""
            WITH deterministic_base AS (
                SELECT DISTINCT
                    NULLIF(TRIM({raw_users_expr}), '') AS university_raw,
                    LOWER(NULLIF(TRIM({raw_users_expr}), '')) AS university_raw_key,
                    NULLIF(TRIM({inst_clean_regex_sql(raw_users_expr)}), '') AS rev_instname_clean,
                    TRY_CAST((deterministic_inst_candidates[1]).unitid AS BIGINT) AS unitid,
                    TRY_CAST((deterministic_inst_candidates[1]).hybrid_score AS DOUBLE) AS deterministic_candidate_score
                FROM read_parquet('{_escape(raw_users_path)}')
                WHERE array_length(deterministic_inst_candidates) > 0
                  AND NULLIF(TRIM({raw_users_expr}), '') IS NOT NULL
                  AND TRY_CAST((deterministic_inst_candidates[1]).unitid AS BIGINT) IS NOT NULL
            ),
            deterministic_ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY university_raw_key
                        ORDER BY deterministic_candidate_score DESC NULLS LAST, university_raw, unitid
                    ) AS row_rank
                FROM deterministic_base
            ),
            deterministic_selected AS (
                SELECT
                    university_raw,
                    university_raw_key,
                    rev_instname_clean,
                    unitid,
                    NULL::BIGINT AS f1_row_num_candidate,
                    NULL::VARCHAR AS ipeds_instname_clean,
                    'deterministic_candidates' AS rev_match_source,
                    'deterministic_top1' AS rev_matchtype,
                    'deterministic' AS match_group,
                    deterministic_candidate_score AS school_match_score,
                    'raw_string_level' AS artifact_grain,
                    '{_escape(raw_users_path)}' AS source_artifact,
                    'deterministic_top1' AS mapping_source,
                    deterministic_candidate_score
                FROM deterministic_ranked
                WHERE row_rank = 1
            ),
            legacy_selected AS (
                SELECT *
                FROM ({legacy_query})
            )
            SELECT
                university_raw,
                rev_instname_clean,
                unitid,
                f1_row_num_candidate,
                ipeds_instname_clean,
                rev_match_source,
                rev_matchtype,
                match_group,
                school_match_score,
                artifact_grain,
                source_artifact,
                mapping_source,
                deterministic_candidate_score
            FROM deterministic_selected

            UNION ALL

            SELECT
                legacy_selected.university_raw,
                legacy_selected.rev_instname_clean,
                legacy_selected.unitid,
                legacy_selected.f1_row_num_candidate,
                legacy_selected.ipeds_instname_clean,
                legacy_selected.rev_match_source,
                legacy_selected.rev_matchtype,
                legacy_selected.match_group,
                legacy_selected.school_match_score,
                legacy_selected.artifact_grain,
                legacy_selected.source_artifact,
                legacy_selected.mapping_source,
                legacy_selected.deterministic_candidate_score
            FROM legacy_selected
            LEFT JOIN (
                SELECT DISTINCT university_raw_key
                FROM deterministic_selected
            ) AS det
              ON legacy_selected.university_raw_key = det.university_raw_key
            WHERE det.university_raw_key IS NULL

            ORDER BY university_raw, unitid
        """
    else:
        query = f"""
            SELECT
                university_raw,
                rev_instname_clean,
                unitid,
                f1_row_num_candidate,
                ipeds_instname_clean,
                rev_match_source,
                rev_matchtype,
                match_group,
                school_match_score,
                artifact_grain,
                source_artifact,
                mapping_source,
                deterministic_candidate_score
            FROM ({legacy_query})
            ORDER BY university_raw, unitid
        """
    _write_query(con, query=query, out_path=out_path)
    _require_output_columns(
        con,
        out_path,
        ["university_raw", "unitid", "artifact_grain", "mapping_source", "deterministic_candidate_score"],
        label="Revelio school-to-UNITID crosswalk",
    )
    return out_path


def _build_field_cip_crosswalk(
    con: duckdb.DuckDBPyConnection,
    *,
    stage_cfg: dict[str, Any],
    cleaned_education_path: str,
) -> str:
    out_path = stage_cfg["field_cip_crosswalk_parquet"]
    legacy_query = _legacy_field_cip_query(
        con,
        stage_cfg=stage_cfg,
        cleaned_education_path=cleaned_education_path,
    )
    raw_users_path = _resolve_existing_path(stage_cfg.get("raw_wrds_users_parquet"))
    cip_reference_path = _resolve_existing_path(stage_cfg.get("cip_reference_input_path"))
    raw_cols = _describe_parquet_columns(con, raw_users_path) if raw_users_path else []
    has_deterministic_candidates = bool(raw_users_path) and "deterministic_cip_candidates" in set(raw_cols)
    field_source_expr = "COALESCE(field_raw, field, field_key, '')" if "field_key" in set(raw_cols) else "COALESCE(field_raw, field, '')"

    if has_deterministic_candidates and cip_reference_path:
        query = f"""
            WITH cip_titles AS (
                SELECT DISTINCT
                    TRY_CAST(SUBSTRING(REGEXP_REPLACE(CAST(CIPCode AS VARCHAR), '[^0-9]', '', 'g'), 1, 4) AS BIGINT) AS cip_code,
                    NULLIF({_alpha_text_sql("CIPTitle")}, '') AS title_alpha_norm
                FROM read_csv_auto('{_escape(cip_reference_path)}', HEADER=TRUE, ALL_VARCHAR=TRUE)
                WHERE LENGTH(REGEXP_REPLACE(CAST(CIPCode AS VARCHAR), '[^0-9]', '', 'g')) = 4
                  AND CIPTitle IS NOT NULL
            ),
            deterministic_base AS (
                SELECT DISTINCT
                    NULLIF(TRIM(CAST(field_raw AS VARCHAR)), '') AS field_raw,
                    NULLIF(TRIM(CAST({field_source_expr} AS VARCHAR)), '') AS source_field_text,
                    NULLIF(TRIM({field_clean_regex_sql(field_source_expr)}), '') AS source_field_norm,
                    NULLIF({_alpha_text_sql(field_source_expr)}, '') AS source_alpha_norm,
                    deterministic_cip_candidates
                FROM read_parquet('{_escape(raw_users_path)}')
                WHERE array_length(deterministic_cip_candidates) > 0
            ),
            source_counts AS (
                SELECT
                    source_field_norm,
                    COUNT(*) AS n_source_strings
                FROM deterministic_base
                WHERE source_field_norm IS NOT NULL
                GROUP BY source_field_norm
            ),
            deterministic_expanded AS (
                SELECT DISTINCT
                    field_raw,
                    source_field_text,
                    source_field_norm,
                    source_alpha_norm,
                    {cip_code_to_cip4_sql("deterministic_cip_candidates[idx].cip_code")} AS cip_code,
                    TRY_CAST(deterministic_cip_candidates[idx].hybrid_score AS DOUBLE) AS candidate_score,
                    idx AS raw_candidate_rank
                FROM deterministic_base
                CROSS JOIN generate_series(1, array_length(deterministic_cip_candidates)) AS gs(idx)
                WHERE {cip_code_to_cip4_sql("deterministic_cip_candidates[idx].cip_code")} IS NOT NULL
            ),
            deterministic_collapsed AS (
                SELECT
                    field_raw,
                    source_field_text,
                    source_field_norm,
                    source_alpha_norm,
                    cip_code,
                    MAX(candidate_score) AS deterministic_candidate_score,
                    MIN(raw_candidate_rank) AS raw_candidate_rank
                FROM deterministic_expanded
                GROUP BY
                    field_raw,
                    source_field_text,
                    source_field_norm,
                    source_alpha_norm,
                    cip_code
            ),
            deterministic_scored AS (
                SELECT
                    dc.field_raw,
                    dc.source_field_text,
                    dc.source_field_norm,
                    dc.cip_code,
                    dc.deterministic_candidate_score,
                    dc.raw_candidate_rank,
                    sc.n_source_strings,
                    COALESCE(
                        MAX(
                            CASE
                                WHEN ct.title_alpha_norm = dc.source_alpha_norm THEN 2
                                WHEN list_has_all(string_split(ct.title_alpha_norm, ' '), string_split(dc.source_alpha_norm, ' '))
                                  OR list_has_all(string_split(dc.source_alpha_norm, ' '), string_split(ct.title_alpha_norm, ' '))
                                    THEN 1
                                ELSE 0
                            END
                        ),
                        0
                    ) AS text_match_rank
                FROM deterministic_collapsed AS dc
                LEFT JOIN source_counts AS sc
                  ON dc.source_field_norm = sc.source_field_norm
                LEFT JOIN cip_titles AS ct
                  ON dc.cip_code = ct.cip_code
                GROUP BY
                    dc.field_raw,
                    dc.source_field_text,
                    dc.source_field_norm,
                    dc.source_alpha_norm,
                    dc.cip_code,
                    sc.n_source_strings,
                    dc.deterministic_candidate_score,
                    dc.raw_candidate_rank
            ),
            deterministic_ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY source_field_norm
                        ORDER BY text_match_rank DESC, deterministic_candidate_score DESC NULLS LAST, raw_candidate_rank, source_field_text, cip_code
                    ) AS row_rank
                FROM deterministic_scored
                WHERE source_field_norm IS NOT NULL
                  AND cip_code IS NOT NULL
            ),
            deterministic_selected AS (
                SELECT
                    field_raw,
                    source_field_norm AS field_clean,
                    source_field_text,
                    source_field_norm,
                    n_source_strings,
                    cip_code,
                    'normalized_string_level' AS artifact_grain,
                    'deterministic_top1' AS mapping_source,
                    deterministic_candidate_score
                FROM deterministic_ranked
                WHERE row_rank = 1
            ),
            legacy_selected AS (
                SELECT *
                FROM ({legacy_query})
            )
            SELECT
                field_raw,
                field_clean,
                source_field_text,
                source_field_norm,
                n_source_strings,
                cip_code,
                artifact_grain,
                mapping_source,
                deterministic_candidate_score
            FROM deterministic_selected

            UNION ALL

            SELECT
                legacy_selected.field_raw,
                legacy_selected.field_clean,
                legacy_selected.source_field_text,
                legacy_selected.source_field_norm,
                legacy_selected.n_source_strings,
                legacy_selected.cip_code,
                legacy_selected.artifact_grain,
                legacy_selected.mapping_source,
                legacy_selected.deterministic_candidate_score
            FROM legacy_selected
            LEFT JOIN (
                SELECT DISTINCT source_field_norm
                FROM deterministic_selected
            ) AS det
              ON legacy_selected.source_field_norm = det.source_field_norm
            WHERE det.source_field_norm IS NULL
        """
    elif has_deterministic_candidates:
        query = f"""
            WITH deterministic_base AS (
                SELECT DISTINCT
                    NULLIF(TRIM(CAST(field_raw AS VARCHAR)), '') AS field_raw,
                    NULLIF(TRIM(CAST({field_source_expr} AS VARCHAR)), '') AS source_field_text,
                    NULLIF(TRIM({field_clean_regex_sql(field_source_expr)}), '') AS source_field_norm,
                    {cip_code_to_cip4_sql("(deterministic_cip_candidates[1]).cip_code")} AS cip_code,
                    TRY_CAST((deterministic_cip_candidates[1]).hybrid_score AS DOUBLE) AS deterministic_candidate_score
                FROM read_parquet('{_escape(raw_users_path)}')
                WHERE array_length(deterministic_cip_candidates) > 0
            ),
            deterministic_ranked AS (
                SELECT
                    *,
                    COUNT(*) OVER (PARTITION BY source_field_norm) AS n_source_strings,
                    ROW_NUMBER() OVER (
                        PARTITION BY source_field_norm
                        ORDER BY deterministic_candidate_score DESC NULLS LAST, source_field_text, cip_code
                    ) AS row_rank
                FROM deterministic_base
                WHERE source_field_norm IS NOT NULL
                  AND cip_code IS NOT NULL
            ),
            deterministic_selected AS (
                SELECT
                    field_raw,
                    source_field_norm AS field_clean,
                    source_field_text,
                    source_field_norm,
                    n_source_strings,
                    cip_code,
                    'normalized_string_level' AS artifact_grain,
                    'deterministic_top1' AS mapping_source,
                    deterministic_candidate_score
                FROM deterministic_ranked
                WHERE row_rank = 1
            ),
            legacy_selected AS (
                SELECT *
                FROM ({legacy_query})
            )
            SELECT
                field_raw,
                field_clean,
                source_field_text,
                source_field_norm,
                n_source_strings,
                cip_code,
                artifact_grain,
                mapping_source,
                deterministic_candidate_score
            FROM deterministic_selected

            UNION ALL

            SELECT
                legacy_selected.field_raw,
                legacy_selected.field_clean,
                legacy_selected.source_field_text,
                legacy_selected.source_field_norm,
                legacy_selected.n_source_strings,
                legacy_selected.cip_code,
                legacy_selected.artifact_grain,
                legacy_selected.mapping_source,
                legacy_selected.deterministic_candidate_score
            FROM legacy_selected
            LEFT JOIN (
                SELECT DISTINCT source_field_norm
                FROM deterministic_selected
            ) AS det
              ON legacy_selected.source_field_norm = det.source_field_norm
            WHERE det.source_field_norm IS NULL
        """
    else:
        query = legacy_query

    _write_query(con, query=query, out_path=out_path)
    _require_output_columns(
        con,
        out_path,
        [
            "source_field_text",
            "cip_code",
            "artifact_grain",
            "mapping_source",
            "deterministic_candidate_score",
        ],
        label="Field-to-CIP crosswalk",
    )
    return out_path


def _build_employer_lookup(
    con: duckdb.DuckDBPyConnection,
    *,
    stage_cfg: dict[str, Any],
) -> str:
    out_path = stage_cfg["employer_lookup_parquet"]
    source_path = _resolve_existing_path(stage_cfg.get("legacy_employer_lookup_parquet"))
    if source_path is None:
        employer_builder = _load_local_employer_builder(stage_cfg.get("legacy_config_path"))
        employer_builder.build_employer_crosswalk(
            overwrite=coerce_bool(stage_cfg.get("overwrite"), True),
            con=con,
        )
        source_path = _resolve_existing_path(
            stage_cfg.get("legacy_employer_lookup_parquet"),
            getattr(employer_builder.cfg, "F1_OPT_EMPLOYER_LOOKUP_PARQUET", ""),
        )
    if source_path is None:
        raise ValueError("Unable to resolve any source for the row-level employer lookup.")

    cols = _describe_parquet_columns(con, source_path)
    employer_name_col = _required_column(cols, ["employer_name"], label="employer_name", source_path=source_path)
    employer_name_clean_col = _required_column(cols, ["employer_name_clean"], label="employer_name_clean", source_path=source_path)
    employer_city_col = _required_column(cols, ["employer_city_clean"], label="employer_city_clean", source_path=source_path)
    employer_state_col = _required_column(cols, ["employer_state_clean"], label="employer_state_clean", source_path=source_path)
    employer_zip_col = _required_column(cols, ["employer_zip_clean"], label="employer_zip_clean", source_path=source_path)
    row_id_col = _required_column(cols, ["foia_row_uid", "f1_emp_row_num"], label="row identifier", source_path=source_path)
    firm_id_col = _required_column(cols, ["foia_firm_uid", "f1_emp_entity_id"], label="firm identifier", source_path=source_path)
    rcid_col = _required_column(cols, ["rcid"], label="rcid", source_path=source_path)
    matched_name_col = _first_present(cols, ["matched_company_name"])
    match_type_col = _first_present(cols, ["match_type"])
    match_score_col = _first_present(cols, ["match_score"])
    lookup_count_col = _first_present(cols, ["lookup_rcid_count"])
    lookup_ambig_col = _first_present(cols, ["lookup_rcid_ambiguous_ind"])
    lookup_direct_col = _first_present(cols, ["lookup_has_direct_ind"])

    query = f"""
        SELECT
            CAST({employer_name_col} AS VARCHAR) AS employer_name,
            CAST({employer_name_clean_col} AS VARCHAR) AS employer_name_clean,
            CAST({employer_city_col} AS VARCHAR) AS employer_city_clean,
            CAST({employer_state_col} AS VARCHAR) AS employer_state_clean,
            CAST({employer_zip_col} AS VARCHAR) AS employer_zip_clean,
            CAST({row_id_col} AS VARCHAR) AS foia_row_uid,
            CAST({firm_id_col} AS VARCHAR) AS foia_firm_uid,
            TRY_CAST({rcid_col} AS BIGINT) AS rcid,
            {_optional_select(matched_name_col, 'matched_company_name')},
            {_optional_select(match_type_col, 'match_type')},
            {_optional_select(match_score_col, 'match_score', cast='DOUBLE')},
            {_optional_select(lookup_count_col, 'lookup_rcid_count', cast='BIGINT')},
            {_optional_select(lookup_ambig_col, 'lookup_rcid_ambiguous_ind', cast='BIGINT')},
            {_optional_select(lookup_direct_col, 'lookup_has_direct_ind', cast='BIGINT')},
            '{_escape(source_path)}' AS source_artifact
        FROM read_parquet('{_escape(source_path)}')
        WHERE TRY_CAST({rcid_col} AS BIGINT) IS NOT NULL
    """
    _write_query(con, query=query, out_path=out_path)
    _require_output_columns(
        con,
        out_path,
        ["foia_row_uid", "foia_firm_uid", "rcid"],
        label="Employer lookup artifact",
    )
    return out_path


def _build_employer_key_map(
    con: duckdb.DuckDBPyConnection,
    *,
    stage_cfg: dict[str, Any],
) -> str:
    out_path = stage_cfg["employer_key_map_parquet"]
    source_path = stage_cfg["employer_lookup_parquet"]
    cols = _describe_parquet_columns(con, source_path)
    rcid_col = _required_column(cols, ["rcid"], label="rcid", source_path=source_path)
    firm_uid_col = _first_present(cols, ["foia_firm_uid"])
    matched_name_col = _first_present(cols, ["matched_company_name"])
    cleaned_name_col = _first_present(cols, ["employer_name_clean"])
    raw_name_col = _first_present(cols, ["employer_name"])
    match_type_col = _first_present(cols, ["match_type"])

    normalized_name_expr = "COALESCE("
    normalized_parts = []
    for column in (matched_name_col, cleaned_name_col, raw_name_col):
        if column is not None:
            normalized_parts.append(f"NULLIF(TRIM(CAST({column} AS VARCHAR)), '')")
    normalized_name_expr += ", ".join(normalized_parts) + ")" if normalized_parts else "NULL)"

    query = f"""
        WITH base AS (
            SELECT
                TRY_CAST({rcid_col} AS BIGINT) AS rcid,
                {normalized_name_expr} AS normalized_employer_name,
                {_optional_select(match_type_col, 'representative_match_type')},
                {_optional_select(firm_uid_col, 'foia_firm_uid')}
            FROM read_parquet('{_escape(source_path)}')
            WHERE TRY_CAST({rcid_col} AS BIGINT) IS NOT NULL
        ),
        agg AS (
            SELECT
                rcid,
                normalized_employer_name,
                representative_match_type,
                COUNT(*) OVER (PARTITION BY rcid) AS n_employer_rows,
                COUNT(DISTINCT foia_firm_uid) OVER (PARTITION BY rcid) AS n_foia_firms,
                ROW_NUMBER() OVER (
                    PARTITION BY rcid
                    ORDER BY CASE WHEN normalized_employer_name IS NULL THEN 1 ELSE 0 END,
                             normalized_employer_name
                ) AS rcid_rank
            FROM base
        )
        SELECT
            rcid,
            normalized_employer_name,
            representative_match_type,
            n_employer_rows,
            n_foia_firms,
            'employer_level' AS artifact_grain,
            '{_escape(source_path)}' AS source_artifact
        FROM agg
        WHERE rcid_rank = 1
        ORDER BY rcid
    """
    _write_query(con, query=query, out_path=out_path)
    _require_output_columns(
        con,
        out_path,
        ["rcid", "normalized_employer_name", "artifact_grain"],
        label="Employer key map",
    )
    return out_path


def _build_openalex_ipeds_crosswalk(
    con: duckdb.DuckDBPyConnection,
    *,
    stage_cfg: dict[str, Any],
    testing: bool = False,
) -> tuple[str, str]:
    """Match OpenAlex institutions to IPEDS UNITIDs; write crosswalk + exploded name table.

    Matching runs in three passes:
      1. Exact: cleaned name string equality
      2. Subset: one cleaned name is a full substring of the other (length/token guards apply)
      3. JW fuzzy: jaro_winkler_similarity >= threshold, blocked on shared uncommon tokens

    Writes two parquet files:
      - crosswalk: one row per (main_unitid, openalex_id) match with match_type + score
      - names: one row per (main_unitid, name_variant) from both IPEDS and matched OpenAlex
    """
    import json
    import pandas as pd

    t0 = time.perf_counter()
    crosswalk_path = stage_cfg["openalex_ipeds_crosswalk_parquet"]
    names_path = stage_cfg["openalex_ipeds_names_parquet"]

    ipeds_path = _resolve_existing_path(stage_cfg.get("ipeds_main_institutions_parquet"))
    if ipeds_path is None:
        raise ValueError("[03_rev_crosswalks] ipeds_main_institutions_parquet not found; check config")
    openalex_path = _resolve_existing_path(stage_cfg.get("openalex_institutions_jsonl"))
    if openalex_path is None:
        raise ValueError("[03_rev_crosswalks] openalex_institutions_jsonl not found; check config")

    idf_threshold = float(stage_cfg.get("openalex_idf_threshold", 2.0))
    jw_threshold = float(stage_cfg.get("openalex_jw_threshold", 0.92))
    subset_min_len = int(stage_cfg.get("openalex_subset_min_len", 10))
    subset_min_tokens = int(stage_cfg.get("openalex_subset_min_tokens", 2))
    subset_min_score = float(stage_cfg.get("openalex_subset_min_score", 0.6))
    max_oa_records = 5000 if testing else None

    # --- Load OpenAlex JSONL into a flat DataFrame of (openalex_id, name, name_source) ---
    # Only keep institutions with country_code == 'US' or missing/null (to match IPEDS scope).
    print(f"[03_rev_crosswalks] Loading OpenAlex institutions from {openalex_path} (US or unknown country only)")
    oa_rows: list[dict[str, str]] = []
    n_oa_skipped = 0
    with open(openalex_path, encoding="utf-8") as fh:
        for line_idx, raw_line in enumerate(fh):
            if max_oa_records is not None and line_idx >= max_oa_records:
                break
            text = raw_line.strip()
            if not text:
                continue
            d = json.loads(text)
            oid = str(d.get("openalex_id") or "").strip()
            if not oid:
                continue
            # Filter: keep only US institutions or those with no country specified
            country = str(d.get("country_code") or "").strip().upper()
            if country and country != "US":
                n_oa_skipped += 1
                continue
            display = str(d.get("display_name") or "").strip()
            if display:
                oa_rows.append({"openalex_id": oid, "name": display, "name_source": "openalex_display"})
            for alt in d.get("alternative_names") or []:
                alt_s = str(alt).strip()
                if alt_s:
                    oa_rows.append({"openalex_id": oid, "name": alt_s, "name_source": "openalex_altname"})
            for acr in d.get("acronyms") or []:
                acr_s = str(acr).strip()
                if acr_s:
                    oa_rows.append({"openalex_id": oid, "name": acr_s, "name_source": "openalex_acronym"})

    oa_df = pd.DataFrame(oa_rows, columns=["openalex_id", "name", "name_source"])
    n_oa_inst = oa_df["openalex_id"].nunique()
    print(f"[03_rev_crosswalks]   {len(oa_df):,} OpenAlex name rows ({n_oa_inst:,} institutions; {n_oa_skipped:,} non-US skipped)")
    con.register("_oa_names_raw", oa_df)

    # --- Build cleaned IPEDS name table (explode name variants + alias names) ---
    # In testing mode, limit to first 500 institutions by main_unitid.
    ipeds_base_subq = (
        f"SELECT * FROM read_parquet('{_escape(ipeds_path)}') ORDER BY main_unitid LIMIT 500"
        if testing
        else f"SELECT * FROM read_parquet('{_escape(ipeds_path)}')"
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _ipeds_clean AS
        WITH ipeds_exploded AS (
            SELECT main_unitid,
                   unnest(ipeds_name_raw_variants) AS name,
                   'ipeds_variant'::VARCHAR AS name_source
            FROM ({ipeds_base_subq}) WHERE ipeds_name_raw_variants IS NOT NULL
            UNION ALL
            SELECT main_unitid,
                   unnest(ipeds_alias_names) AS name,
                   'ipeds_alias'::VARCHAR AS name_source
            FROM ({ipeds_base_subq}) WHERE ipeds_alias_names IS NOT NULL
        )
        SELECT main_unitid, name, name_source,
               NULLIF(TRIM({inst_clean_regex_sql('name')}), '') AS name_clean
        FROM ipeds_exploded
        WHERE NULLIF(TRIM(name), '') IS NOT NULL
    """)
    n_ipeds_inst = con.execute("SELECT COUNT(DISTINCT main_unitid) FROM _ipeds_clean").fetchone()[0]
    n_ipeds_rows = con.execute("SELECT COUNT(*) FROM _ipeds_clean").fetchone()[0]
    print(f"[03_rev_crosswalks]   {n_ipeds_rows:,} IPEDS name rows ({n_ipeds_inst:,} institutions)")

    # --- Build cleaned OpenAlex name table ---
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _oa_clean AS
        SELECT openalex_id, name, name_source,
               NULLIF(TRIM({inst_clean_regex_sql('name')}), '') AS name_clean
        FROM _oa_names_raw
        WHERE NULLIF(TRIM(name), '') IS NOT NULL
          AND len(NULLIF(TRIM(name), '')) >= 4
    """)

    # --- Compute token IDF over IPEDS name strings (one row per distinct unitid × token) ---
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _ipeds_tokens AS
        SELECT DISTINCT main_unitid, token
        FROM (
            SELECT main_unitid, unnest(str_split(name_clean, ' ')) AS token
            FROM _ipeds_clean WHERE name_clean IS NOT NULL
        )
        WHERE len(token) > 1
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _oa_tokens AS
        SELECT DISTINCT openalex_id, token
        FROM (
            SELECT openalex_id, unnest(str_split(name_clean, ' ')) AS token
            FROM _oa_clean WHERE name_clean IS NOT NULL
        )
        WHERE len(token) > 1
    """)
    # Uncommon tokens: log(N / (1 + df)) > idf_threshold
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _uncommon_tokens AS
        SELECT token
        FROM (
            SELECT token, COUNT(DISTINCT main_unitid) AS doc_count
            FROM _ipeds_tokens GROUP BY token
        ) AS tf
        CROSS JOIN (SELECT COUNT(DISTINCT main_unitid) AS n FROM _ipeds_clean) AS tot
        WHERE ln(CAST(n AS DOUBLE) / (1.0 + CAST(doc_count AS DOUBLE))) > {idf_threshold}
    """)
    n_uncommon = con.execute("SELECT COUNT(*) FROM _uncommon_tokens").fetchone()[0]
    print(f"[03_rev_crosswalks]   {n_uncommon:,} uncommon tokens (IDF > {idf_threshold})")

    # --- Build blocking pairs: (main_unitid, openalex_id) sharing at least one uncommon token ---
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _blocking_pairs AS
        SELECT DISTINCT it.main_unitid, ot.openalex_id
        FROM _ipeds_tokens it
        JOIN _uncommon_tokens u ON it.token = u.token
        JOIN _oa_tokens ot ON it.token = ot.token
    """)
    n_blocking = con.execute("SELECT COUNT(*) FROM _blocking_pairs").fetchone()[0]
    print(f"[03_rev_crosswalks]   {n_blocking:,} blocking candidate pairs")

    # --- Step 1: Exact matches (cleaned name equality, institution-wide) ---
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _exact_matches AS
        SELECT DISTINCT ON (i.main_unitid, o.openalex_id)
            i.main_unitid,
            o.openalex_id,
            'exact'::VARCHAR AS match_type,
            1.0::DOUBLE AS match_score,
            i.name AS ipeds_name_match,
            o.name AS openalex_name_match
        FROM _ipeds_clean i
        JOIN _oa_clean o ON i.name_clean = o.name_clean
        WHERE i.name_clean IS NOT NULL AND len(i.name_clean) >= 4
        ORDER BY i.main_unitid, o.openalex_id, i.name_source, i.name, o.name_source, o.name
    """)
    n_exact = con.execute("SELECT COUNT(*) FROM _exact_matches").fetchone()[0]
    print(f"[03_rev_crosswalks]   Step 1 (exact): {n_exact:,} pairs")

    # --- Step 2: Subset matches (blocked, then CONTAINS check; skip already-exact-matched) ---
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _subset_matches AS
        SELECT DISTINCT ON (bp.main_unitid, bp.openalex_id)
            bp.main_unitid,
            bp.openalex_id,
            'subset'::VARCHAR AS match_type,
            CAST(LEAST(len(i.name_clean), len(o.name_clean)) AS DOUBLE) /
                CAST(GREATEST(len(i.name_clean), len(o.name_clean)) AS DOUBLE) AS match_score,
            i.name AS ipeds_name_match,
            o.name AS openalex_name_match
        FROM _blocking_pairs bp
        JOIN _ipeds_clean i ON bp.main_unitid = i.main_unitid
        JOIN _oa_clean o ON bp.openalex_id = o.openalex_id
        LEFT JOIN _exact_matches em ON bp.main_unitid = em.main_unitid
            AND bp.openalex_id = em.openalex_id
        WHERE em.main_unitid IS NULL
          AND (contains(i.name_clean, o.name_clean) OR contains(o.name_clean, i.name_clean))
          AND len(i.name_clean) >= {subset_min_len}
          AND len(o.name_clean) >= {subset_min_len}
          AND len(str_split(i.name_clean, ' ')) >= {subset_min_tokens}
          AND len(str_split(o.name_clean, ' ')) >= {subset_min_tokens}
          AND CAST(LEAST(len(i.name_clean), len(o.name_clean)) AS DOUBLE) /
                CAST(GREATEST(len(i.name_clean), len(o.name_clean)) AS DOUBLE) >= {subset_min_score}
        ORDER BY bp.main_unitid, bp.openalex_id,
            CAST(LEAST(len(i.name_clean), len(o.name_clean)) AS DOUBLE) /
                CAST(GREATEST(len(i.name_clean), len(o.name_clean)) AS DOUBLE) DESC
    """)
    n_subset = con.execute("SELECT COUNT(*) FROM _subset_matches").fetchone()[0]
    print(f"[03_rev_crosswalks]   Step 2 (subset): {n_subset:,} pairs")

    # --- Step 3: JW fuzzy matches (blocked, skip already-matched, keep >= jw_threshold) ---
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _jw_matches AS
        WITH jw_candidates AS (
            SELECT
                bp.main_unitid,
                bp.openalex_id,
                i.name AS ipeds_name_match,
                o.name AS openalex_name_match,
                jaro_winkler_similarity(i.name_clean, o.name_clean) AS jw_score
            FROM _blocking_pairs bp
            JOIN _ipeds_clean i ON bp.main_unitid = i.main_unitid
            JOIN _oa_clean o ON bp.openalex_id = o.openalex_id
            LEFT JOIN _exact_matches em ON bp.main_unitid = em.main_unitid
                AND bp.openalex_id = em.openalex_id
            LEFT JOIN _subset_matches sm ON bp.main_unitid = sm.main_unitid
                AND bp.openalex_id = sm.openalex_id
            WHERE em.main_unitid IS NULL
              AND sm.main_unitid IS NULL
              AND i.name_clean IS NOT NULL
              AND o.name_clean IS NOT NULL
        ),
        jw_best AS (
            SELECT DISTINCT ON (main_unitid, openalex_id)
                main_unitid, openalex_id, ipeds_name_match, openalex_name_match, jw_score AS match_score
            FROM jw_candidates
            ORDER BY main_unitid, openalex_id, jw_score DESC
        )
        SELECT main_unitid, openalex_id,
               'jw_fuzzy'::VARCHAR AS match_type,
               match_score,
               ipeds_name_match, openalex_name_match
        FROM jw_best
        WHERE match_score >= {jw_threshold}
    """)
    n_jw = con.execute("SELECT COUNT(*) FROM _jw_matches").fetchone()[0]
    print(f"[03_rev_crosswalks]   Step 3 (jw_fuzzy): {n_jw:,} pairs")

    # --- Write crosswalk parquet ---
    print(f"[03_rev_crosswalks] Writing crosswalk -> {crosswalk_path}")
    _write_query(
        con,
        query="""
            SELECT main_unitid, openalex_id, match_type, match_score,
                   ipeds_name_match, openalex_name_match
            FROM _exact_matches
            UNION ALL
            SELECT main_unitid, openalex_id, match_type, match_score,
                   ipeds_name_match, openalex_name_match
            FROM _subset_matches
            UNION ALL
            SELECT main_unitid, openalex_id, match_type, match_score,
                   ipeds_name_match, openalex_name_match
            FROM _jw_matches
            ORDER BY main_unitid, openalex_id
        """,
        out_path=crosswalk_path,
    )
    n_unitids_matched = con.execute(
        f"SELECT COUNT(DISTINCT main_unitid) FROM read_parquet('{_escape(crosswalk_path)}')"
    ).fetchone()[0]
    n_oa_matched = con.execute(
        f"SELECT COUNT(DISTINCT openalex_id) FROM read_parquet('{_escape(crosswalk_path)}')"
    ).fetchone()[0]
    n_total_pairs = n_exact + n_subset + n_jw
    print(
        f"[03_rev_crosswalks]   {n_total_pairs:,} total match pairs "
        f"({n_unitids_matched:,} IPEDS unitids, {n_oa_matched:,} OpenAlex IDs)"
    )

    # --- Write exploded names parquet ---
    # All IPEDS name variants for every institution, plus all OpenAlex name variants
    # for each matched institution, keyed by main_unitid.
    print(f"[03_rev_crosswalks] Writing exploded names -> {names_path}")
    _write_query(
        con,
        query="""
            SELECT
                i.main_unitid,
                NULL::VARCHAR AS openalex_id,
                i.name AS name_variant,
                i.name_clean,
                i.name_source
            FROM _ipeds_clean i

            UNION ALL

            SELECT
                cw.main_unitid,
                o.openalex_id,
                o.name AS name_variant,
                o.name_clean,
                o.name_source
            FROM _oa_clean o
            JOIN (
                SELECT DISTINCT main_unitid, openalex_id FROM _exact_matches
                UNION ALL
                SELECT DISTINCT main_unitid, openalex_id FROM _subset_matches
                UNION ALL
                SELECT DISTINCT main_unitid, openalex_id FROM _jw_matches
            ) cw ON o.openalex_id = cw.openalex_id

            ORDER BY main_unitid, openalex_id NULLS FIRST, name_source, name_variant
        """,
        out_path=names_path,
    )
    n_names_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{_escape(names_path)}')"
    ).fetchone()[0]
    print(f"[03_rev_crosswalks]   {n_names_rows:,} rows in exploded names table")

    # --- Validate outputs ---
    _require_output_columns(
        con, crosswalk_path,
        ["main_unitid", "openalex_id", "match_type", "match_score"],
        label="OpenAlex–IPEDS crosswalk",
    )
    _require_output_columns(
        con, names_path,
        ["main_unitid", "name_variant", "name_clean", "name_source"],
        label="OpenAlex–IPEDS exploded names",
    )

    # Clean up temp objects
    for tbl in [
        "_exact_matches", "_subset_matches", "_jw_matches",
        "_blocking_pairs",
        "_ipeds_tokens", "_oa_tokens", "_uncommon_tokens",
        "_oa_clean", "_ipeds_clean",
    ]:
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
    try:
        con.unregister("_oa_names_raw")
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    print(f"[03_rev_crosswalks] OpenAlex–IPEDS crosswalk complete in {elapsed:.1f}s")
    return crosswalk_path, names_path


def _validate_stage_outputs(
    con: duckdb.DuckDBPyConnection,
    stage_cfg: dict[str, Any],
    *,
    include_school: bool,
    include_field: bool,
    include_employer: bool,
) -> None:
    if include_school:
        _require_output_columns(
            con,
            stage_cfg["school_family_crosswalk_parquet"],
            ["f1_school_name", "rev_university_raw", "rev_instname_clean"],
            label="School-family crosswalk",
        )
        _require_output_columns(
            con,
            stage_cfg["school_resolution_parquet"],
            ["f1_school_name", "f1_row_num", "UNITID", "rev_university_raw"],
            label="School-resolution artifact",
        )
        _require_output_columns(
            con,
            stage_cfg["f1_inst_unitid_crosswalk_parquet"],
            ["f1_row_num", "unitid", "school_name"],
            label="F1 institution-to-UNITID crosswalk",
        )
    if include_field:
        _require_output_columns(
            con,
            stage_cfg["field_cip_crosswalk_parquet"],
            ["source_field_text", "cip_code"],
            label="Field-to-CIP crosswalk",
        )
    if include_employer:
        _require_output_columns(
            con,
            stage_cfg["employer_lookup_parquet"],
            ["foia_row_uid", "foia_firm_uid", "rcid"],
            label="Employer lookup",
        )
        _require_output_columns(
            con,
            stage_cfg["employer_key_map_parquet"],
            ["rcid", "normalized_employer_name"],
            label="Employer key map",
        )


def build_crosswalks(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    testing: bool | None = None,
    build_school: bool = True,
    build_field: bool = False,
    build_employer: bool = True,
    build_openalex: bool = False,
) -> dict[str, str]:
    cfg = pipeline_cfg or load_config(config_path)
    stage_cfg = dict(get_stage_config(cfg, STAGE_NAME))
    required_keys: set[str] = set()
    if build_school:
        required_keys.update(
            {
                "school_family_crosswalk_parquet",
                "school_resolution_parquet",
                "f1_inst_unitid_crosswalk_parquet",
            }
        )
    if build_field:
        required_keys.add("field_cip_crosswalk_parquet")
    if build_employer:
        required_keys.update({"employer_lookup_parquet", "employer_key_map_parquet"})
    if build_openalex:
        required_keys.update({"openalex_ipeds_crosswalk_parquet", "openalex_ipeds_names_parquet"})

    for key in required_keys:
        if not stage_cfg.get(key):
            raise ValueError(f"Missing required config entry for {STAGE_NAME}: {key}")

    stage_cfg["overwrite"] = coerce_bool(stage_cfg.get("overwrite"), coerce_bool(cfg.get("build", {}).get("overwrite"), True))
    testing_enabled = coerce_bool(cfg.get("testing", {}).get("enabled"), False) if testing is None else testing
    cleaned_education_path = _resolve_external_cleaned_education_path(stage_cfg) if build_field else ""
    staged_school_artifacts = _maybe_stage_external_school_artifacts(stage_cfg)
    if staged_school_artifacts:
        print(f"[{STAGE_NAME}] External school artifacts staged from {staged_school_artifacts['f1_revelio_school_crosswalk'].parent}")

    con = duckdb.connect()
    con.execute("SET threads = 8")
    outputs: dict[str, str] = {}

    if build_school:
        outputs["school_family_crosswalk_parquet"] = _build_school_family_crosswalk(con, stage_cfg=stage_cfg)
        outputs["school_resolution_parquet"] = _build_school_resolution(con, stage_cfg=stage_cfg)
        outputs["f1_inst_unitid_crosswalk_parquet"] = _build_f1_inst_unitid_crosswalk(con, stage_cfg=stage_cfg)
    if build_field:
        outputs["field_cip_crosswalk_parquet"] = _build_field_cip_crosswalk(
            con,
            stage_cfg=stage_cfg,
            cleaned_education_path=cleaned_education_path,
        )
    if build_employer:
        outputs["employer_lookup_parquet"] = _build_employer_lookup(con, stage_cfg=stage_cfg)
        outputs["employer_key_map_parquet"] = _build_employer_key_map(con, stage_cfg=stage_cfg)
    if build_openalex:
        cw_path, names_path = _build_openalex_ipeds_crosswalk(
            con, stage_cfg=stage_cfg, testing=testing_enabled
        )
        outputs["openalex_ipeds_crosswalk_parquet"] = cw_path
        outputs["openalex_ipeds_names_parquet"] = names_path

    _validate_stage_outputs(
        con,
        stage_cfg,
        include_school=build_school or Path(stage_cfg["f1_inst_unitid_crosswalk_parquet"]).exists(),
        include_field=build_field and Path(stage_cfg["field_cip_crosswalk_parquet"]).exists(),
        include_employer=build_employer or Path(stage_cfg["employer_lookup_parquet"]).exists(),
    )
    print(f"[{STAGE_NAME}] Crosswalk artifacts ready (testing={testing_enabled})")
    for key in sorted(outputs):
        print(f"[{STAGE_NAME}] {key} -> {outputs[key]}")
    return outputs


def run(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    testing: bool | None = None,
    build_school: bool = True,
    build_field: bool = False,
    build_employer: bool = True,
    build_openalex: bool = False,
) -> dict[str, str]:
    return build_crosswalks(
        config_path=config_path,
        pipeline_cfg=pipeline_cfg,
        testing=testing,
        build_school=build_school,
        build_field=build_field,
        build_employer=build_employer,
        build_openalex=build_openalex,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 03_rev_crosswalks.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-school", action="store_true")
    parser.add_argument("--skip-field", action="store_true")
    parser.add_argument("--skip-employer", action="store_true")
    parser.add_argument("--skip-openalex", action="store_true")
    testing_group = parser.add_mutually_exclusive_group()
    testing_group.add_argument("--testing", dest="testing", action="store_true")
    testing_group.add_argument("--no-testing", dest="testing", action="store_false")
    parser.set_defaults(testing=None)
    args = parser.parse_args(sanitize_ipykernel_argv())

    cfg = load_config(args.config)
    effective_testing = coerce_bool(cfg.get("testing", {}).get("enabled"), False) if args.testing is None else args.testing
    t0 = time.perf_counter()
    try:
        run(
            config_path=args.config,
            pipeline_cfg=cfg,
            testing=args.testing,
            build_school=not args.skip_school,
            build_field=not args.skip_field,
            build_employer=not args.skip_employer,
            build_openalex=not args.skip_openalex,
        )
    except StageDeferredError as exc:
        print(exc)
        raise SystemExit(2) from exc
    if not effective_testing:
        mark_stage_complete(STAGE_NAME, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
