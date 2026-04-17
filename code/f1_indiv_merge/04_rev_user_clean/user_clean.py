from __future__ import annotations

import sys
import time
from builtins import print as _print
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import duckdb

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = STAGE_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
for _path in (STAGE_DIR, PIPELINE_ROOT, REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from common import (  # noqa: E402
    ANGLO_COUNTRIES,
    atomic_duckdb_copy_to_parquet,
    ensure_parent_dir,
    escape_sql_literal,
    get_duckdb_connection,
    register_country_support_views,
    resolve_user_shard_spec,
    resolve_existing_path,
    sql_user_shard_predicate,
    sql_clean_company_name_expr,
    sql_normalize_alpha_expr,
    sql_string_literal,
    sql_text_relation_rank_expr,
)
from helpers import (  # noqa: E402
    cip_code_to_cip4_sql,
    degree_clean_regex_sql,
    field_category_to_cip4_sql,
    field_clean_regex_sql,
    field_clean_to_cip4_sql,
    fullname_clean_regex_sql,
    inst_clean_regex_sql,
    stem_ind_regex_sql,
)
from src.config_loader import get_stage_config, load_config  # noqa: E402
from src.pipeline_runtime import coerce_bool  # noqa: E402
from config import root  # noqa: E402

STAGE_NAME = "04_rev_user_clean"
MATCH_READY_REQUIRED_COLS = {
    "user_id",
    "unitid",
    "degree_clean",
    "country_candidate",
    "cip",
    "employer_key",
}
_COUNTRY_NULL_LITERALS = ("na", "n/a", "none", "null", "<na>", "nan")
_EMPTY_CIP_LOOKUP_QUERY = """
    SELECT
        CAST(NULL AS BIGINT) AS cip,
        CAST(NULL AS VARCHAR) AS title_norm
    WHERE FALSE
"""
_EMPTY_UNITID_ALIAS_QUERY = """
    SELECT
        CAST(NULL AS BIGINT) AS unitid,
        CAST(NULL AS VARCHAR) AS alias_norm
    WHERE FALSE
"""

print = partial(_print, flush=True)


@dataclass
class Stage04Session:
    connection: duckdb.DuckDBPyConnection
    outputs: dict[str, Any]
    tables: dict[str, str]

    def close(self) -> None:
        self.connection.close()


def _fmt_elapsed(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _first_present(columns: list[str], candidates: list[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _sql_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _describe_parquet_columns(path: str) -> list[str]:
    con = get_duckdb_connection()
    return [
        row[0]
        for row in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{escape_sql_literal(path)}')"
        ).fetchall()
    ]


def _describe_parquet_schema(path: str) -> dict[str, str]:
    con = get_duckdb_connection()
    return {
        str(row[0]): str(row[1])
        for row in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{escape_sql_literal(path)}')"
        ).fetchall()
    }


def _create_temp_table(con: duckdb.DuckDBPyConnection, name: str, query_sql: str) -> None:
    con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {query_sql}")


def _create_empty_table(
    con: duckdb.DuckDBPyConnection,
    name: str,
    columns: list[tuple[str, str]],
) -> None:
    select_terms = [f"CAST(NULL AS {dtype}) AS {column}" for column, dtype in columns]
    _create_temp_table(con, name, "SELECT " + ", ".join(select_terms) + " WHERE FALSE")


def _relation_count(con: duckdb.DuckDBPyConnection, relation_name: str) -> int:
    return int(con.execute(f"SELECT COUNT(*) FROM {relation_name}").fetchone()[0])


def _relation_columns(con: duckdb.DuckDBPyConnection, relation_name: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation_name}").fetchall()]


def _relation_exists(con: duckdb.DuckDBPyConnection, relation_name: str) -> bool:
    try:
        con.execute(f"DESCRIBE SELECT * FROM {relation_name}").fetchall()
    except duckdb.Error:
        return False
    return True


def _country_standardize_expr(value_sql: str, map_alias: str) -> str:
    trimmed = f"TRIM(CAST({value_sql} AS VARCHAR))"
    null_literals = ", ".join(sql_string_literal(value) for value in _COUNTRY_NULL_LITERALS)
    return f"""
        CASE
            WHEN {value_sql} IS NULL THEN NULL
            WHEN {trimmed} = '' THEN NULL
            WHEN LOWER({trimmed}) IN ({null_literals}) THEN NULL
            ELSE COALESCE(
                {map_alias}.std_country,
                CASE
                    WHEN {trimmed} = UPPER({trimmed}) THEN title({trimmed})
                    ELSE {trimmed}
                END
            )
        END
    """


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


def _normalized_degree_type_expr(value_sql: str) -> str:
    normalized = f"LOWER(TRIM(CAST({value_sql} AS VARCHAR)))"
    return f"""
        CASE
            WHEN {value_sql} IS NULL THEN NULL
            WHEN {normalized} = '' THEN NULL
            WHEN {normalized} IN ('non_degree', 'non-degree') THEN 'non_degree'
            WHEN {normalized} IN ('hs_or_below', 'high school') THEN 'hs_or_below'
            WHEN {normalized} IN ('associate', 'associates') THEN 'associates'
            WHEN {normalized} IN ('bachelor', 'bachelors') THEN 'bachelors'
            WHEN {normalized} IN ('master', 'masters', 'mba') THEN 'masters'
            WHEN {normalized} IN ('doctor', 'doctors', 'doctorate', 'phd', 'ph.d') THEN 'doctor'
            WHEN {normalized} = 'unknown' THEN 'unknown'
            ELSE {normalized}
        END
    """


def _match_ready_degree_clean_expr(
    deterministic_array_sql: str = "e.deterministic_degree_types",
    fallback_degree_clean_sql: str = "e.degree_clean",
) -> str:
    deterministic_top1 = f"{deterministic_array_sql}[1]"
    return f"""
        COALESCE(
            CASE
                WHEN {deterministic_array_sql} IS NOT NULL
                 AND array_length({deterministic_array_sql}) > 0
                    THEN {_normalized_degree_type_expr(deterministic_top1)}
                ELSE NULL
            END,
            {_normalized_degree_type_expr(fallback_degree_clean_sql)}
        )
    """


def _create_keep_users_table(
    con: duckdb.DuckDBPyConnection,
    *,
    source_sql: str,
    testing_max_users: int | None,
    shard_count: int | None = None,
    shard_id: int | None = None,
    table_name: str = "keep_users",
) -> bool:
    if (testing_max_users is None or testing_max_users <= 0) and (
        shard_count is None or shard_id is None
    ):
        return False
    shard_where_sql = ""
    if shard_count is not None and shard_id is not None:
        shard_where_sql = (
            "AND "
            + sql_user_shard_predicate(
                "user_id",
                shard_count=int(shard_count),
                shard_id=int(shard_id),
            )
        )
    limit_sql = ""
    if testing_max_users is not None and testing_max_users > 0:
        limit_sql = f"ORDER BY user_id LIMIT {int(testing_max_users)}"
    _create_temp_table(
        con,
        table_name,
        f"""
        SELECT DISTINCT user_id
        FROM ({source_sql})
        WHERE user_id IS NOT NULL
          {shard_where_sql}
        {limit_sql}
        """,
    )
    return True


def _create_raw_sources(
    con: duckdb.DuckDBPyConnection,
    *,
    raw_users_path: str,
    raw_positions_path: str,
    deterministic_triple_map_path: str | None,
    testing_max_users: int | None,
    shard_count: int | None = None,
    shard_id: int | None = None,
) -> None:
    raw_cols = set(_describe_parquet_columns(raw_users_path))
    field_norm_source = "COALESCE(field_raw, field, '')"
    degree_key_expr = _normalize_key_sql("u.degree_raw")
    field_key_expr = _normalize_key_sql("u.field_raw")
    inst_key_expr = _normalize_key_sql("u.university_raw")
    triple_join = ""
    if deterministic_triple_map_path is not None:
        triple_cols = set(_describe_parquet_columns(deterministic_triple_map_path))
        triple_schema = _describe_parquet_schema(deterministic_triple_map_path)
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
        inst_candidates_col = (
            "candidate_ref_inst_ids"
            if "candidate_ref_inst_ids" in triple_cols
            else "inst_candidates"
            if "inst_candidates" in triple_cols
            else None
        )
        degree_element_expr = "x.degree_type"
        if degree_types_col is not None and "STRUCT(" not in triple_schema.get(degree_types_col, ""):
            degree_element_expr = "x"
        cip_score_expr = "x.hybrid_score"
        if cip_candidates_col is not None and "hybrid_score" not in triple_schema.get(cip_candidates_col, ""):
            cip_score_expr = "x.score"
        inst_score_expr = "x.hybrid_score"
        if inst_candidates_col is not None and "hybrid_score" not in triple_schema.get(inst_candidates_col, ""):
            inst_score_expr = "x.score"
        triple_join = f"""
        LEFT JOIN read_parquet('{escape_sql_literal(deterministic_triple_map_path)}') AS dtm
          ON CAST(dtm.degree_key AS VARCHAR) = {degree_key_expr}
         AND CAST(dtm.field_key AS VARCHAR) = {field_key_expr}
         AND CAST(dtm.inst_key AS VARCHAR) = {inst_key_expr}
        """
        det_cip_expr = """
            CASE
                WHEN __CIP_COL__ IS NULL THEN NULL::STRUCT(cip_code VARCHAR, hybrid_score DOUBLE)[]
                ELSE list_filter(
                    list_transform(
                        __CIP_COL__,
                        x -> struct_pack(
                            cip_code := CAST(x.cip_code AS VARCHAR),
                            hybrid_score := TRY_CAST(__CIP_SCORE__ AS DOUBLE)
                        )
                    ),
                    x -> x.cip_code IS NOT NULL AND TRIM(CAST(x.cip_code AS VARCHAR)) != ''
                )
            END
        """
        det_ref_inst_expr = """
            CASE
                WHEN __INST_COL__ IS NULL THEN NULL::STRUCT(ref_inst_id VARCHAR, hybrid_score DOUBLE)[]
                ELSE list_filter(
                    list_transform(
                        __INST_COL__,
                        x -> struct_pack(
                            ref_inst_id := CAST(x.ref_inst_id AS VARCHAR),
                            hybrid_score := TRY_CAST(__INST_SCORE__ AS DOUBLE)
                        )
                    ),
                    x -> x.ref_inst_id IS NOT NULL AND TRIM(CAST(x.ref_inst_id AS VARCHAR)) != ''
                )
            END
        """
        det_degree_expr = f"""
            CASE
                WHEN __DEGREE_COL__ IS NULL THEN NULL::VARCHAR[]
                ELSE list_filter(
                    list_transform(
                        __DEGREE_COL__,
                        x -> {_normalized_degree_type_expr("__DEGREE_VALUE__")}
                    ),
                    x -> x IS NOT NULL AND TRIM(CAST(x AS VARCHAR)) != ''
                )
            END
        """
        det_cip_expr = det_cip_expr.replace("__CIP_COL__", f"dtm.{cip_candidates_col}" if cip_candidates_col else "NULL")
        det_cip_expr = det_cip_expr.replace("__CIP_SCORE__", cip_score_expr)
        det_ref_inst_expr = det_ref_inst_expr.replace("__INST_COL__", f"dtm.{inst_candidates_col}" if inst_candidates_col else "NULL")
        det_ref_inst_expr = det_ref_inst_expr.replace("__INST_SCORE__", inst_score_expr)
        det_degree_expr = det_degree_expr.replace("__DEGREE_COL__", f"dtm.{degree_types_col}" if degree_types_col else "NULL")
        det_degree_expr = det_degree_expr.replace("__DEGREE_VALUE__", degree_element_expr)
    else:
        det_cip_expr = (
            "deterministic_cip_candidates"
            if "deterministic_cip_candidates" in raw_cols
            else "NULL::STRUCT(cip_code VARCHAR, hybrid_score DOUBLE)[]"
        )
        det_ref_inst_expr = (
            "list_transform(deterministic_inst_candidates, x -> struct_pack(ref_inst_id := CAST(x.unitid AS VARCHAR), hybrid_score := TRY_CAST(x.hybrid_score AS DOUBLE)))"
            if "deterministic_inst_candidates" in raw_cols
            else "NULL::STRUCT(ref_inst_id VARCHAR, hybrid_score DOUBLE)[]"
        )
        det_degree_expr = (
            "deterministic_degree_types"
            if "deterministic_degree_types" in raw_cols
            else "NULL::VARCHAR[]"
        )
    keep_users_enabled = _create_keep_users_table(
        con,
        source_sql=f"""
            SELECT TRY_CAST(user_id AS BIGINT) AS user_id
            FROM read_parquet('{escape_sql_literal(raw_users_path)}')
            WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL
        """,
        testing_max_users=testing_max_users,
        shard_count=shard_count,
        shard_id=shard_id,
    )
    keep_users_join = "JOIN keep_users ku ON ku.user_id = TRY_CAST(u.user_id AS BIGINT)" if keep_users_enabled else ""
    _create_temp_table(
        con,
        "src_users_work",
        f"""
        SELECT
            TRY_CAST(u.user_id AS BIGINT) AS user_id,
            CAST(u.fullname AS VARCHAR) AS fullname,
            CASE
                WHEN u.fullname ~ '.*[A-z].*' THEN {fullname_clean_regex_sql('u.fullname')}
                ELSE ''
            END AS fullname_clean,
            CAST(u.profile_linkedin_url AS VARCHAR) AS profile_linkedin_url,
            CAST(u.user_location AS VARCHAR) AS user_location,
            CAST(u.user_country AS VARCHAR) AS user_country,
            {_country_standardize_expr('u.user_country', 'ucm')} AS user_country_std,
            TRY_CAST(u.f_prob AS DOUBLE) AS f_prob,
            CAST(u.updated_dt AS VARCHAR) AS updated_dt,
            TRY_CAST(u.updated_dt AS TIMESTAMP) AS updated_dt_ts,
            CAST(u.university_name AS VARCHAR) AS university_name,
            CAST(u.rsid AS VARCHAR) AS rsid,
            TRY_CAST(u.education_number AS BIGINT) AS education_number,
            CAST(u.ed_startdate AS VARCHAR) AS ed_startdate,
            CAST(u.ed_enddate AS VARCHAR) AS ed_enddate,
            CAST(u.degree AS VARCHAR) AS degree,
            CAST(u.field AS VARCHAR) AS field,
            CAST(u.university_country AS VARCHAR) AS university_country,
            {_country_standardize_expr('u.university_country', 'univcm')} AS university_country_std,
            CAST(u.university_location AS VARCHAR) AS university_location,
            CAST(u.university_raw AS VARCHAR) AS university_raw,
            NULLIF(LOWER(TRIM(CAST(u.university_raw AS VARCHAR))), '') AS university_raw_key,
            CAST(u.degree_raw AS VARCHAR) AS degree_raw,
            CAST(u.field_raw AS VARCHAR) AS field_raw,
            CAST(u.description AS VARCHAR) AS description,
            NULLIF({degree_key_expr}, '') AS degree_key,
            NULLIF({field_key_expr}, '') AS field_key,
            NULLIF({inst_key_expr}, '') AS inst_key,
            {det_cip_expr} AS deterministic_cip_candidates,
            {det_ref_inst_expr} AS deterministic_ref_inst_candidates,
            {det_degree_expr} AS deterministic_degree_types,
            COALESCE({degree_clean_regex_sql()}, 'Missing') AS degree_clean,
            NULLIF(TRIM({field_clean_regex_sql(field_norm_source)}), '') AS field_clean,
            NULLIF(TRIM({field_clean_regex_sql(field_norm_source)}), '') AS field_norm,
            NULLIF(TRIM({inst_clean_regex_sql('u.university_raw')}), '') AS university_raw_clean,
            {stem_ind_regex_sql()} AS stem_ind
        FROM read_parquet('{escape_sql_literal(raw_users_path)}') AS u
        {keep_users_join}
        {triple_join}
        LEFT JOIN country_label_map AS ucm
          ON UPPER(TRIM(CAST(u.user_country AS VARCHAR))) = ucm.lookup_upper
        LEFT JOIN country_label_map AS univcm
          ON UPPER(TRIM(CAST(u.university_country AS VARCHAR))) = univcm.lookup_upper
        WHERE TRY_CAST(u.user_id AS BIGINT) IS NOT NULL
        """,
    )
    _create_temp_table(
        con,
        "profiles_src",
        """
        SELECT
            user_id,
            fullname,
            COALESCE(NULLIF(TRIM(fullname_clean), ''), '') AS fullname_clean,
            profile_linkedin_url,
            user_location,
            user_country,
            user_country_std,
            f_prob,
            updated_dt,
            updated_dt_ts
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY updated_dt_ts DESC NULLS LAST
                ) AS rn
            FROM src_users_work
        )
        WHERE rn = 1
        """,
    )
    _create_temp_table(
        con,
        "educ_src",
        """
        SELECT
            user_id,
            education_number,
            university_name,
            rsid,
            ed_startdate,
            ed_enddate,
            degree,
            field,
            university_country,
            university_country_std,
            university_location,
            university_raw,
            university_raw_key,
            degree_raw,
            field_raw,
            description,
            degree_key,
            field_key,
            inst_key,
            deterministic_cip_candidates,
            deterministic_ref_inst_candidates,
            deterministic_degree_types,
            degree_clean,
            field_clean,
            field_norm,
            university_raw_clean,
            stem_ind,
            CASE WHEN degree_clean = 'High School' THEN 1 ELSE 0 END AS high_school_ind,
            CASE WHEN degree_clean = 'Non-Degree' THEN 1 ELSE 0 END AS non_degree_ind,
            CASE WHEN degree_clean IN ('High School', 'Non-Degree') THEN 0 ELSE 1 END AS match_eligible_ind,
            TRY_CAST(SUBSTRING(ed_startdate, 1, 4) AS BIGINT) AS ed_start_year,
            TRY_CAST(SUBSTRING(ed_enddate, 1, 4) AS BIGINT) AS ed_end_year
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, education_number
                    ORDER BY updated_dt_ts DESC NULLS LAST
                ) AS rn
            FROM src_users_work
            WHERE education_number IS NOT NULL
        )
        WHERE rn = 1
        """,
    )

    pos_cols = set(_describe_parquet_columns(raw_positions_path))
    keep_positions_join = "JOIN keep_users ku ON ku.user_id = TRY_CAST(p.user_id AS BIGINT)" if keep_users_enabled else ""
    _create_temp_table(
        con,
        "positions_src",
        f"""
        SELECT
            TRY_CAST(p.user_id AS BIGINT) AS user_id,
            CAST(p.position_id AS VARCHAR) AS position_id,
            TRY_CAST(p.position_number AS BIGINT) AS position_number,
            TRY_CAST(p.rcid AS BIGINT) AS rcid,
            CAST(p.country AS VARCHAR) AS country,
            {_country_standardize_expr('p.country', 'pcm')} AS country_std,
            CAST(p.startdate AS VARCHAR) AS startdate,
            CAST(p.enddate AS VARCHAR) AS enddate,
            TRY_CAST(SUBSTRING(CAST(p.startdate AS VARCHAR), 1, 4) AS BIGINT) AS start_year,
            TRY_CAST(SUBSTRING(CAST(p.enddate AS VARCHAR), 1, 4) AS BIGINT) AS end_year,
            CAST(p.role_k17000_v3 AS VARCHAR) AS role_k17000_v3,
            TRY_CAST(p.salary AS DOUBLE) AS salary,
            TRY_CAST(p.total_compensation AS DOUBLE) AS total_compensation,
            CAST(p.company_raw AS VARCHAR) AS company_raw,
            NULLIF(LOWER(TRIM(CAST(p.company_raw AS VARCHAR))), '') AS company_raw_clean,
            {sql_clean_company_name_expr('p.company_raw')} AS company_raw_normed,
            CAST(p.title_raw AS VARCHAR) AS title_raw
        FROM read_parquet('{escape_sql_literal(raw_positions_path)}') AS p
        {keep_positions_join}
        LEFT JOIN country_label_map AS pcm
          ON UPPER(TRIM(CAST(p.country AS VARCHAR))) = pcm.lookup_upper
        WHERE TRY_CAST(p.user_id AS BIGINT) IS NOT NULL
        """,
    )


def _create_legacy_sources(
    con: duckdb.DuckDBPyConnection,
    *,
    legacy_rev_indiv_path: str,
    legacy_rev_educ_path: str,
    legacy_rev_pos_path: str,
    testing_max_users: int | None,
    shard_count: int | None = None,
    shard_id: int | None = None,
) -> None:
    legacy_indiv_cols = _describe_parquet_columns(legacy_rev_indiv_path)
    country_col = _first_present(legacy_indiv_cols, ["country_std", "country", "nationality_country"])
    score_col = _first_present(legacy_indiv_cols, ["total_score", "nationality_score", "country_score"])
    subregion_col = _first_present(legacy_indiv_cols, ["subregion"])
    nanat_col = _first_present(legacy_indiv_cols, ["nanat_score", "nationality_score", "total_score"])
    nanat_subregion_col = _first_present(legacy_indiv_cols, ["nanat_subregion_score"])
    nt_subregion_col = _first_present(legacy_indiv_cols, ["nt_subregion_score"])
    uncertain_col = _first_present(legacy_indiv_cols, ["country_uncertain_ind"])
    full_clean_expr = (
        "NULLIF(TRIM(CAST(fullname_clean AS VARCHAR)), '')"
        if "fullname_clean" in legacy_indiv_cols
        else f"CASE WHEN fullname ~ '.*[A-z].*' THEN {fullname_clean_regex_sql('fullname')} ELSE '' END"
    )

    keep_users_enabled = _create_keep_users_table(
        con,
        source_sql=f"""
            SELECT TRY_CAST(user_id AS BIGINT) AS user_id
            FROM read_parquet('{escape_sql_literal(legacy_rev_indiv_path)}')
            WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL
        """,
        testing_max_users=testing_max_users,
        shard_count=shard_count,
        shard_id=shard_id,
    )
    keep_indiv_join = "JOIN keep_users ku ON ku.user_id = TRY_CAST(i.user_id AS BIGINT)" if keep_users_enabled else ""
    _create_temp_table(
        con,
        "legacy_indiv_work",
        f"""
        SELECT
            TRY_CAST(i.user_id AS BIGINT) AS user_id,
            CAST(i.fullname AS VARCHAR) AS fullname,
            {full_clean_expr} AS fullname_clean,
            CAST(i.user_location AS VARCHAR) AS user_location,
            CAST(i.user_country AS VARCHAR) AS user_country,
            {_country_standardize_expr('i.user_country', 'ucm')} AS user_country_std,
            TRY_CAST(i.f_prob AS DOUBLE) AS f_prob,
            {f"CAST(i.{_sql_identifier(country_col)} AS VARCHAR)" if country_col else "NULL::VARCHAR"} AS country_candidate_raw,
            {f"TRY_CAST(i.{_sql_identifier(score_col)} AS DOUBLE)" if score_col else "1.0"} AS country_score_raw,
            {f"TRY_CAST(i.{_sql_identifier(nanat_col)} AS DOUBLE)" if nanat_col else "NULL::DOUBLE"} AS nanat_score_raw,
            {f"TRY_CAST(i.{_sql_identifier(nanat_subregion_col)} AS DOUBLE)" if nanat_subregion_col else "NULL::DOUBLE"} AS nanat_subregion_score_raw,
            {f"TRY_CAST(i.{_sql_identifier(nt_subregion_col)} AS DOUBLE)" if nt_subregion_col else "NULL::DOUBLE"} AS nt_subregion_score_raw,
            {f"TRY_CAST(i.{_sql_identifier(uncertain_col)} AS BIGINT)" if uncertain_col else "0"} AS country_uncertain_ind_raw,
            {f"CAST(i.{_sql_identifier(subregion_col)} AS VARCHAR)" if subregion_col else "NULL::VARCHAR"} AS subregion_candidate_raw
        FROM read_parquet('{escape_sql_literal(legacy_rev_indiv_path)}') AS i
        {keep_indiv_join}
        LEFT JOIN country_label_map AS ucm
          ON UPPER(TRIM(CAST(i.user_country AS VARCHAR))) = ucm.lookup_upper
        WHERE TRY_CAST(i.user_id AS BIGINT) IS NOT NULL
        """,
    )
    _create_temp_table(
        con,
        "profiles_src",
        """
        SELECT
            user_id,
            fullname,
            COALESCE(NULLIF(TRIM(fullname_clean), ''), '') AS fullname_clean,
            CAST(NULL AS VARCHAR) AS profile_linkedin_url,
            user_location,
            user_country,
            user_country_std,
            f_prob,
            CAST(NULL AS VARCHAR) AS updated_dt,
            CAST(NULL AS TIMESTAMP) AS updated_dt_ts
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY country_score_raw DESC NULLS LAST, user_id
                ) AS rn
            FROM legacy_indiv_work
        )
        WHERE rn = 1
        """,
    )
    _create_temp_table(
        con,
        "country_candidates_seed",
        """
        WITH standardized AS (
            SELECT
                user_id,
                CASE
                    WHEN country_candidate_raw IS NULL OR TRIM(country_candidate_raw) = '' THEN NULL
                    ELSE COALESCE(cm.std_country, country_candidate_raw)
                END AS country_candidate,
                COALESCE(country_score_raw, 1.0) AS country_score_raw,
                COALESCE(nanat_score_raw, country_score_raw, 1.0) AS nanat_score,
                0.0::DOUBLE AS institution_score,
                0.0::DOUBLE AS nametrace_score,
                COALESCE(nanat_subregion_score_raw, nanat_score_raw, country_score_raw, 1.0) AS nanat_subregion_score,
                COALESCE(nt_subregion_score_raw, 0.0) AS nt_subregion_score,
                COALESCE(country_uncertain_ind_raw, 0) AS country_uncertain_ind,
                COALESCE(subregion_candidate_raw, sm.subregion_candidate) AS subregion_candidate,
                0.0::DOUBLE AS anglo_pressure
            FROM legacy_indiv_work li
            LEFT JOIN country_label_map AS cm
              ON UPPER(TRIM(li.country_candidate_raw)) = cm.lookup_upper
            LEFT JOIN subregion_map AS sm
              ON COALESCE(cm.std_country, li.country_candidate_raw) = sm.country_candidate
        ),
        grouped AS (
            SELECT
                user_id,
                country_candidate,
                MAX(country_score_raw) AS country_score_raw,
                MAX(nanat_score) AS nanat_score,
                MAX(institution_score) AS institution_score,
                MAX(nametrace_score) AS nametrace_score,
                MAX(nanat_subregion_score) AS nanat_subregion_score,
                MAX(nt_subregion_score) AS nt_subregion_score,
                MAX(country_uncertain_ind) AS country_uncertain_ind,
                MAX(subregion_candidate) AS subregion_candidate,
                MAX(anglo_pressure) AS anglo_pressure
            FROM standardized
            WHERE country_candidate IS NOT NULL
            GROUP BY 1, 2
        ),
        scored AS (
            SELECT
                *,
                country_score_raw / NULLIF(SUM(country_score_raw) OVER (PARTITION BY user_id), 0.0) AS country_score
            FROM grouped
        )
        SELECT
            user_id,
            country_candidate,
            country_score,
            nanat_score,
            institution_score,
            nametrace_score,
            nanat_subregion_score,
            nt_subregion_score,
            subregion_candidate,
            country_uncertain_ind,
            anglo_pressure,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY country_score DESC NULLS LAST, country_candidate
            ) AS country_rank
        FROM scored
        """,
    )

    legacy_educ_schema = _describe_parquet_schema(legacy_rev_educ_path)
    legacy_educ_cols = list(legacy_educ_schema.keys())
    keep_educ_join = "JOIN keep_users ku ON ku.user_id = TRY_CAST(e.user_id AS BIGINT)" if keep_users_enabled else ""
    education_number_expr = (
        "TRY_CAST(e.education_number AS BIGINT)"
        if "education_number" in legacy_educ_cols
        else "ROW_NUMBER() OVER (ORDER BY TRY_CAST(e.user_id AS BIGINT), CAST(e.university_raw AS VARCHAR), CAST(e.ed_startdate AS VARCHAR), CAST(e.ed_enddate AS VARCHAR))"
    )
    degree_clean_expr = (
        "CAST(e.degree_clean AS VARCHAR)"
        if "degree_clean" in legacy_educ_cols
        else "COALESCE(CAST(e.degree AS VARCHAR), 'Missing')"
    )
    university_name_expr = "CAST(e.university_name AS VARCHAR)" if "university_name" in legacy_educ_cols else "NULL::VARCHAR"
    rsid_expr = "CAST(e.rsid AS VARCHAR)" if "rsid" in legacy_educ_cols else "NULL::VARCHAR"
    degree_expr = "CAST(e.degree AS VARCHAR)" if "degree" in legacy_educ_cols else "NULL::VARCHAR"
    field_expr = "CAST(e.field AS VARCHAR)" if "field" in legacy_educ_cols else "NULL::VARCHAR"
    university_country_expr = (
        "CAST(e.university_country AS VARCHAR)" if "university_country" in legacy_educ_cols else "NULL::VARCHAR"
    )
    university_location_expr = (
        "CAST(e.university_location AS VARCHAR)" if "university_location" in legacy_educ_cols else "NULL::VARCHAR"
    )
    degree_raw_expr = "CAST(e.degree_raw AS VARCHAR)" if "degree_raw" in legacy_educ_cols else "NULL::VARCHAR"
    field_raw_expr = "CAST(e.field_raw AS VARCHAR)" if "field_raw" in legacy_educ_cols else "NULL::VARCHAR"
    description_expr = "CAST(e.description AS VARCHAR)" if "description" in legacy_educ_cols else "NULL::VARCHAR"
    degree_key_expr = (
        f"NULLIF({_normalize_key_sql('e.degree_raw')}, '')"
        if "degree_raw" in legacy_educ_cols
        else "NULL::VARCHAR"
    )
    field_key_expr = (
        f"NULLIF({_normalize_key_sql('e.field_raw')}, '')"
        if "field_raw" in legacy_educ_cols
        else "NULL::VARCHAR"
    )
    inst_key_expr = (
        f"NULLIF({_normalize_key_sql('e.university_raw')}, '')"
        if "university_raw" in legacy_educ_cols
        else "NULL::VARCHAR"
    )
    stem_ind_expr = "TRY_CAST(e.stem_ind AS BIGINT)" if "stem_ind" in legacy_educ_cols else "NULL::BIGINT"
    univ_raw_clean_expr = (
        "CAST(e.univ_raw_clean AS VARCHAR)" if "univ_raw_clean" in legacy_educ_cols else "NULL::VARCHAR"
    )
    field_clean_source = _first_present(legacy_educ_cols, ["field_clean", "field_raw", "field"])
    field_clean_expr = f"CAST(e.{_sql_identifier(field_clean_source)} AS VARCHAR)" if field_clean_source else "NULL::VARCHAR"
    university_country_source = _first_present(legacy_educ_cols, ["match_country", "university_country"])
    det_cip_expr = (
        "e.deterministic_cip_candidates"
        if "deterministic_cip_candidates" in legacy_educ_schema
        and "STRUCT(" in legacy_educ_schema["deterministic_cip_candidates"]
        else "NULL::STRUCT(cip_code VARCHAR, hybrid_score DOUBLE)[]"
    )
    det_inst_expr = (
        "e.deterministic_inst_candidates"
        if "deterministic_inst_candidates" in legacy_educ_schema
        and "STRUCT(" in legacy_educ_schema["deterministic_inst_candidates"]
        else "NULL::STRUCT(unitid VARCHAR, hybrid_score DOUBLE)[]"
    )
    _create_temp_table(
        con,
        "educ_src",
        f"""
        SELECT
            TRY_CAST(e.user_id AS BIGINT) AS user_id,
            {education_number_expr} AS education_number,
            {university_name_expr} AS university_name,
            {rsid_expr} AS rsid,
            CAST(e.ed_startdate AS VARCHAR) AS ed_startdate,
            CAST(e.ed_enddate AS VARCHAR) AS ed_enddate,
            {degree_expr} AS degree,
            {field_expr} AS field,
            {university_country_expr} AS university_country,
            {_country_standardize_expr(f'e.{_sql_identifier(university_country_source)}', 'ucm') if university_country_source else 'NULL::VARCHAR'} AS university_country_std,
            {university_location_expr} AS university_location,
            CAST(e.university_raw AS VARCHAR) AS university_raw,
            NULLIF(LOWER(TRIM(CAST(e.university_raw AS VARCHAR))), '') AS university_raw_key,
            {degree_raw_expr} AS degree_raw,
            {field_raw_expr} AS field_raw,
            {description_expr} AS description,
            {degree_key_expr} AS degree_key,
            {field_key_expr} AS field_key,
            {inst_key_expr} AS inst_key,
            {det_cip_expr} AS deterministic_cip_candidates,
            NULL::STRUCT(ref_inst_id VARCHAR, hybrid_score DOUBLE)[] AS deterministic_ref_inst_candidates,
            {"e.deterministic_degree_types" if "deterministic_degree_types" in legacy_educ_cols else "NULL::VARCHAR[]"} AS deterministic_degree_types,
            COALESCE({degree_clean_expr}, 'Missing') AS degree_clean,
            NULLIF(TRIM({field_clean_expr}), '') AS field_clean,
            NULLIF(TRIM({field_clean_regex_sql(f'COALESCE({field_clean_expr}, {field_expr}, {field_raw_expr}, \'\')')}), '') AS field_norm,
            NULLIF(TRIM(COALESCE({univ_raw_clean_expr}, {inst_clean_regex_sql('e.university_raw')})), '') AS university_raw_clean,
            {stem_ind_expr} AS stem_ind,
            CASE WHEN COALESCE({degree_clean_expr}, 'Missing') = 'High School' THEN 1 ELSE 0 END AS high_school_ind,
            CASE WHEN COALESCE({degree_clean_expr}, 'Missing') = 'Non-Degree' THEN 1 ELSE 0 END AS non_degree_ind,
            CASE WHEN COALESCE({degree_clean_expr}, 'Missing') IN ('High School', 'Non-Degree') THEN 0 ELSE 1 END AS match_eligible_ind,
            TRY_CAST(SUBSTRING(CAST(e.ed_startdate AS VARCHAR), 1, 4) AS BIGINT) AS ed_start_year,
            TRY_CAST(SUBSTRING(CAST(e.ed_enddate AS VARCHAR), 1, 4) AS BIGINT) AS ed_end_year
        FROM read_parquet('{escape_sql_literal(legacy_rev_educ_path)}') AS e
        {keep_educ_join}
        {"LEFT JOIN country_label_map AS ucm ON UPPER(TRIM(CAST(e." + _sql_identifier(university_country_source) + " AS VARCHAR))) = ucm.lookup_upper" if university_country_source else ""}
        WHERE TRY_CAST(e.user_id AS BIGINT) IS NOT NULL
        """,
    )

    legacy_pos_cols = _describe_parquet_columns(legacy_rev_pos_path)
    keep_pos_join = "JOIN keep_users ku ON ku.user_id = TRY_CAST(p.user_id AS BIGINT)" if keep_users_enabled else ""
    position_number_expr = (
        "TRY_CAST(p.position_number AS BIGINT)"
        if "position_number" in legacy_pos_cols
        else "ROW_NUMBER() OVER (ORDER BY TRY_CAST(p.user_id AS BIGINT), CAST(p.startdate AS VARCHAR), CAST(p.enddate AS VARCHAR), CAST(p.company_raw AS VARCHAR))"
    )
    position_id_expr = (
        "CAST(p.position_id AS VARCHAR)"
        if "position_id" in legacy_pos_cols
        else f"CAST({position_number_expr} AS VARCHAR)"
    )
    role_expr = "CAST(p.role_k17000_v3 AS VARCHAR)" if "role_k17000_v3" in legacy_pos_cols else "NULL::VARCHAR"
    salary_expr = "TRY_CAST(p.salary AS DOUBLE)" if "salary" in legacy_pos_cols else "NULL::DOUBLE"
    total_comp_expr = (
        "TRY_CAST(p.total_compensation AS DOUBLE)" if "total_compensation" in legacy_pos_cols else "NULL::DOUBLE"
    )
    _create_temp_table(
        con,
        "positions_src",
        f"""
        SELECT
            TRY_CAST(p.user_id AS BIGINT) AS user_id,
            {position_id_expr} AS position_id,
            {position_number_expr} AS position_number,
            TRY_CAST(p.rcid AS BIGINT) AS rcid,
            CAST(p.country AS VARCHAR) AS country,
            {_country_standardize_expr('p.country', 'pcm')} AS country_std,
            CAST(p.startdate AS VARCHAR) AS startdate,
            CAST(p.enddate AS VARCHAR) AS enddate,
            TRY_CAST(SUBSTRING(CAST(p.startdate AS VARCHAR), 1, 4) AS BIGINT) AS start_year,
            TRY_CAST(SUBSTRING(CAST(p.enddate AS VARCHAR), 1, 4) AS BIGINT) AS end_year,
            {role_expr} AS role_k17000_v3,
            {salary_expr} AS salary,
            {total_comp_expr} AS total_compensation,
            CAST(p.company_raw AS VARCHAR) AS company_raw,
            NULLIF(LOWER(TRIM(CAST(p.company_raw AS VARCHAR))), '') AS company_raw_clean,
            {sql_clean_company_name_expr('p.company_raw')} AS company_raw_normed,
            CAST(p.title_raw AS VARCHAR) AS title_raw
        FROM read_parquet('{escape_sql_literal(legacy_rev_pos_path)}') AS p
        {keep_pos_join}
        LEFT JOIN country_label_map AS pcm
          ON UPPER(TRIM(CAST(p.country AS VARCHAR))) = pcm.lookup_upper
        WHERE TRY_CAST(p.user_id AS BIGINT) IS NOT NULL
        """,
    )


def _create_school_crosswalk_table(con: duckdb.DuckDBPyConnection, path: str | None) -> None:
    if path is None:
        _create_empty_table(
            con,
            "school_crosswalk",
            [
                ("university_raw_key", "VARCHAR"),
                ("unitid_stage03", "BIGINT"),
                ("school_match_score_stage03", "DOUBLE"),
                ("rev_instname_clean_stage03", "VARCHAR"),
                ("school_mapping_source_stage03", "VARCHAR"),
                ("unitid_stage03_candidate_score", "DOUBLE"),
            ],
        )
        return
    columns = set(_describe_parquet_columns(path))
    if "university_raw" not in columns or "unitid" not in columns:
        _create_empty_table(
            con,
            "school_crosswalk",
            [
                ("university_raw_key", "VARCHAR"),
                ("unitid_stage03", "BIGINT"),
                ("school_match_score_stage03", "DOUBLE"),
                ("rev_instname_clean_stage03", "VARCHAR"),
                ("school_mapping_source_stage03", "VARCHAR"),
                ("unitid_stage03_candidate_score", "DOUBLE"),
            ],
        )
        return
    school_match_expr = "TRY_CAST(school_match_score AS DOUBLE)" if "school_match_score" in columns else "NULL::DOUBLE"
    rev_instname_expr = "CAST(rev_instname_clean AS VARCHAR)" if "rev_instname_clean" in columns else "NULL::VARCHAR"
    mapping_source_expr = "CAST(mapping_source AS VARCHAR)" if "mapping_source" in columns else "NULL::VARCHAR"
    det_score_expr = (
        "TRY_CAST(deterministic_candidate_score AS DOUBLE)"
        if "deterministic_candidate_score" in columns
        else "NULL::DOUBLE"
    )
    _create_temp_table(
        con,
        "school_crosswalk",
        f"""
        SELECT
            university_raw_key,
            unitid AS unitid_stage03,
            school_match_score AS school_match_score_stage03,
            rev_instname_clean AS rev_instname_clean_stage03,
            mapping_source AS school_mapping_source_stage03,
            deterministic_candidate_score AS unitid_stage03_candidate_score
        FROM (
            SELECT
                LOWER(TRIM(CAST(university_raw AS VARCHAR))) AS university_raw_key,
                TRY_CAST(unitid AS BIGINT) AS unitid,
                {school_match_expr} AS school_match_score,
                {rev_instname_expr} AS rev_instname_clean,
                {mapping_source_expr} AS mapping_source,
                {det_score_expr} AS deterministic_candidate_score,
                ROW_NUMBER() OVER (
                    PARTITION BY LOWER(TRIM(CAST(university_raw AS VARCHAR)))
                    ORDER BY {school_match_expr} DESC NULLS LAST, TRY_CAST(unitid AS BIGINT)
                ) AS rn
            FROM read_parquet('{escape_sql_literal(path)}')
        )
        WHERE university_raw_key IS NOT NULL
          AND university_raw_key != ''
          AND unitid IS NOT NULL
          AND rn = 1
        """,
    )


def _create_field_crosswalk_table(con: duckdb.DuckDBPyConnection, path: str | None) -> None:
    if path is None:
        _create_empty_table(
            con,
            "field_crosswalk",
            [
                ("field_norm", "VARCHAR"),
                ("cip_stage03", "BIGINT"),
                ("field_mapping_source_stage03", "VARCHAR"),
                ("cip_stage03_candidate_score", "DOUBLE"),
            ],
        )
        return
    columns = _describe_parquet_columns(path)
    norm_col = _first_present(columns, ["source_field_norm", "field_clean", "source_field_text"])
    cip_col = _first_present(columns, ["cip_code", "cip"])
    if norm_col is None or cip_col is None:
        _create_empty_table(
            con,
            "field_crosswalk",
            [
                ("field_norm", "VARCHAR"),
                ("cip_stage03", "BIGINT"),
                ("field_mapping_source_stage03", "VARCHAR"),
                ("cip_stage03_candidate_score", "DOUBLE"),
            ],
        )
        return
    mapping_source_expr = "CAST(mapping_source AS VARCHAR)" if "mapping_source" in columns else "NULL::VARCHAR"
    det_score_expr = (
        "TRY_CAST(deterministic_candidate_score AS DOUBLE)"
        if "deterministic_candidate_score" in columns
        else "NULL::DOUBLE"
    )
    cip_digits_expr = f"REGEXP_REPLACE(CAST({_sql_identifier(cip_col)} AS VARCHAR), '[^0-9]', '', 'g')"
    _create_temp_table(
        con,
        "field_crosswalk",
        f"""
        SELECT
            field_norm,
            cip AS cip_stage03,
            field_mapping_source AS field_mapping_source_stage03,
            deterministic_candidate_score AS cip_stage03_candidate_score
        FROM (
            SELECT
                field_norm,
                cip,
                field_mapping_source,
                deterministic_candidate_score,
                ROW_NUMBER() OVER (
                    PARTITION BY field_norm
                    ORDER BY deterministic_candidate_score DESC NULLS LAST, cip
                ) AS rn
            FROM (
                SELECT
                    LOWER(TRIM(CAST({_sql_identifier(norm_col)} AS VARCHAR))) AS field_norm,
                    CASE
                        WHEN NULLIF({cip_digits_expr}, '') IS NULL THEN NULL
                        WHEN LENGTH({cip_digits_expr}) >= 4 THEN TRY_CAST(SUBSTRING({cip_digits_expr}, 1, 4) AS BIGINT)
                        WHEN LENGTH({cip_digits_expr}) = 3 THEN TRY_CAST(LPAD({cip_digits_expr}, 4, '0') AS BIGINT)
                        ELSE {field_clean_to_cip4_sql('LOWER(TRIM(CAST(' + _sql_identifier(norm_col) + ' AS VARCHAR)))')}
                    END AS cip,
                    {mapping_source_expr} AS field_mapping_source,
                    {det_score_expr} AS deterministic_candidate_score
                FROM read_parquet('{escape_sql_literal(path)}')
            )
        )
        WHERE field_norm IS NOT NULL
          AND field_norm != ''
          AND cip IS NOT NULL
          AND rn = 1
        """,
    )


def _create_employer_key_map_table(
    con: duckdb.DuckDBPyConnection,
    *,
    key_map_path: str | None,
    employer_lookup_path: str | None,
) -> None:
    if key_map_path is not None:
        columns = set(_describe_parquet_columns(key_map_path))
        if "rcid" in columns:
            employer_key_expr = (
                "NULLIF(TRIM(CAST(normalized_employer_name AS VARCHAR)), '')"
                if "normalized_employer_name" in columns
                else "NULL::VARCHAR"
            )
            match_type_expr = (
                "CAST(representative_match_type AS VARCHAR)"
                if "representative_match_type" in columns
                else "NULL::VARCHAR"
            )
            _create_temp_table(
                con,
                "employer_key_map",
                f"""
                SELECT
                    TRY_CAST(rcid AS BIGINT) AS rcid,
                    {employer_key_expr} AS employer_key,
                    {match_type_expr} AS representative_match_type
                FROM read_parquet('{escape_sql_literal(key_map_path)}')
                WHERE TRY_CAST(rcid AS BIGINT) IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY TRY_CAST(rcid AS BIGINT) ORDER BY TRY_CAST(rcid AS BIGINT)) = 1
                """,
            )
            return
    if employer_lookup_path is not None:
        columns = set(_describe_parquet_columns(employer_lookup_path))
        if "rcid" in columns:
            employer_key_candidates: list[str] = []
            for candidate in ("matched_company_name", "employer_name_clean", "employer_name"):
                if candidate in columns:
                    employer_key_candidates.append(f"NULLIF(TRIM(CAST({candidate} AS VARCHAR)), '')")
            employer_key_expr = f"COALESCE({', '.join(employer_key_candidates)})" if employer_key_candidates else "NULL::VARCHAR"
            match_type_expr = "CAST(match_type AS VARCHAR)" if "match_type" in columns else "NULL::VARCHAR"
            _create_temp_table(
                con,
                "employer_key_map",
                f"""
                SELECT
                    TRY_CAST(rcid AS BIGINT) AS rcid,
                    {employer_key_expr} AS employer_key,
                    {match_type_expr} AS representative_match_type
                FROM read_parquet('{escape_sql_literal(employer_lookup_path)}')
                WHERE TRY_CAST(rcid AS BIGINT) IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY TRY_CAST(rcid AS BIGINT) ORDER BY TRY_CAST(rcid AS BIGINT)) = 1
                """,
            )
            return
    _create_empty_table(
        con,
        "employer_key_map",
        [("rcid", "BIGINT"), ("employer_key", "VARCHAR"), ("representative_match_type", "VARCHAR")],
    )


def _create_unitid_alias_table(
    con: duckdb.DuckDBPyConnection,
    *,
    ipeds_path: str | None,
    openalex_path: str | None,
) -> None:
    union_queries: list[str] = []
    if ipeds_path is not None:
        columns = _describe_parquet_columns(ipeds_path)
        unitid_col = _first_present(columns, ["UNITID", "main_unitid", "unitid"])
        if unitid_col is not None:
            for name_col in ("instname", "display_name", "ipeds_name", "ipeds_instname_clean", "instname_raw", "name_variant", "name_clean"):
                if name_col in columns:
                    union_queries.append(
                        f"""
                        SELECT
                            TRY_CAST({_sql_identifier(unitid_col)} AS BIGINT) AS unitid,
                            {sql_normalize_alpha_expr(_sql_identifier(name_col))} AS alias_norm
                        FROM read_parquet('{escape_sql_literal(ipeds_path)}')
                        """
                    )
    if openalex_path is not None:
        columns = _describe_parquet_columns(openalex_path)
        unitid_col = _first_present(columns, ["main_unitid", "UNITID", "unitid"])
        if unitid_col is not None:
            for name_col in ("name_variant", "name_clean"):
                if name_col in columns:
                    union_queries.append(
                        f"""
                        SELECT
                            TRY_CAST({_sql_identifier(unitid_col)} AS BIGINT) AS unitid,
                            {sql_normalize_alpha_expr(_sql_identifier(name_col))} AS alias_norm
                        FROM read_parquet('{escape_sql_literal(openalex_path)}')
                        """
                    )
    if not union_queries:
        _create_temp_table(con, "unitid_aliases", _EMPTY_UNITID_ALIAS_QUERY)
        return
    _create_temp_table(
        con,
        "unitid_aliases",
        """
        SELECT DISTINCT
            unitid,
            alias_norm
        FROM (
        """
        + "\nUNION ALL\n".join(union_queries)
        + """
        )
        WHERE unitid IS NOT NULL
          AND alias_norm IS NOT NULL
          AND alias_norm != ''
        """,
    )


def _create_ref_inst_link_table(con: duckdb.DuckDBPyConnection, path: str | None) -> None:
    if path is None:
        _create_empty_table(
            con,
            "ref_inst_links",
            [("ref_inst_id", "VARCHAR"), ("openalex_id", "VARCHAR"), ("main_unitid", "BIGINT")],
        )
        return
    columns = _describe_parquet_columns(path)
    ref_col = _first_present(columns, ["ref_inst_id"])
    openalex_col = _first_present(columns, ["openalex_id", "openalexid"])
    unitid_col = _first_present(columns, ["main_unitid", "UNITID", "unitid"])
    if ref_col is None:
        _create_empty_table(
            con,
            "ref_inst_links",
            [("ref_inst_id", "VARCHAR"), ("openalex_id", "VARCHAR"), ("main_unitid", "BIGINT")],
        )
        return
    _create_temp_table(
        con,
        "ref_inst_links",
        f"""
        SELECT DISTINCT
            NULLIF(TRIM(CAST({_sql_identifier(ref_col)} AS VARCHAR)), '') AS ref_inst_id,
            {f"NULLIF(TRIM(CAST({_sql_identifier(openalex_col)} AS VARCHAR)), '')" if openalex_col else "NULL::VARCHAR"} AS openalex_id,
            {f"TRY_CAST({_sql_identifier(unitid_col)} AS BIGINT)" if unitid_col else "NULL::BIGINT"} AS main_unitid
        FROM read_parquet('{escape_sql_literal(path)}')
        WHERE NULLIF(TRIM(CAST({_sql_identifier(ref_col)} AS VARCHAR)), '') IS NOT NULL
        """,
    )


def _create_openalex_reference_tables(con: duckdb.DuckDBPyConnection, path: str | None) -> None:
    if path is None:
        _create_empty_table(
            con,
            "openalex_entities",
            [("openalex_id", "VARCHAR"), ("country_candidate", "VARCHAR")],
        )
        _create_empty_table(
            con,
            "openalex_aliases",
            [("openalex_id", "VARCHAR"), ("alias_norm", "VARCHAR")],
        )
        return
    path_obj = Path(path)
    escaped_path = escape_sql_literal(path)
    if path_obj.suffix.lower() in {".parquet", ".pq"}:
        source_sql = f"SELECT * FROM read_parquet('{escaped_path}')"
        columns = _describe_parquet_columns(path)
    else:
        source_sql = f"SELECT * FROM read_json_auto('{escaped_path}', format='newline_delimited')"
        columns = [
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_json_auto('{escaped_path}', format='newline_delimited')"
            ).fetchall()
        ]
    id_expr = (
        "COALESCE(openalex_id, id)"
        if "openalex_id" in columns and "id" in columns
        else "openalex_id"
        if "openalex_id" in columns
        else "id"
        if "id" in columns
        else "NULL::VARCHAR"
    )
    display_expr = (
        "display_name"
        if "display_name" in columns
        else "name"
        if "name" in columns
        else "NULL::VARCHAR"
    )
    country_code_expr = "src.country_code" if "country_code" in columns else "NULL::VARCHAR"
    country_name_expr = (
        "src.country_name"
        if "country_name" in columns
        else "src.country"
        if "country" in columns
        else "NULL::VARCHAR"
    )
    _create_temp_table(
        con,
        "iso_country_code_map",
        f"""
        SELECT
            UPPER(TRIM(CAST("alpha-2" AS VARCHAR))) AS alpha2,
            CAST(name AS VARCHAR) AS country_name
        FROM read_csv_auto('{escape_sql_literal(Path(root) / "data" / "crosswalks" / "iso_country_codes.csv")}', HEADER=TRUE)
        WHERE "alpha-2" IS NOT NULL
          AND name IS NOT NULL
        """,
    )
    _create_temp_table(
        con,
        "openalex_entities",
        f"""
        WITH src AS (
            {source_sql}
        )
        SELECT DISTINCT
            NULLIF(TRIM(CAST({id_expr} AS VARCHAR)), '') AS openalex_id,
            COALESCE(
                clm.std_country,
                icm.country_name,
                NULLIF(TRIM(CAST({country_name_expr} AS VARCHAR)), '')
            ) AS country_candidate
        FROM src
        LEFT JOIN iso_country_code_map AS icm
          ON UPPER(TRIM(CAST({country_code_expr} AS VARCHAR))) = icm.alpha2
        LEFT JOIN country_label_map AS clm
          ON UPPER(TRIM(COALESCE(icm.country_name, CAST({country_name_expr} AS VARCHAR)))) = clm.lookup_upper
        WHERE NULLIF(TRIM(CAST({id_expr} AS VARCHAR)), '') IS NOT NULL
        """,
    )
    alias_queries = [
        f"""
        SELECT
            NULLIF(TRIM(CAST({id_expr} AS VARCHAR)), '') AS openalex_id,
            CAST({display_expr} AS VARCHAR) AS raw_name
        FROM src
        """
    ]
    if "alternative_names" in columns:
        alias_queries.append(
            f"""
            SELECT
                NULLIF(TRIM(CAST({id_expr} AS VARCHAR)), '') AS openalex_id,
                CAST(unnest(alternative_names) AS VARCHAR) AS raw_name
            FROM src
            WHERE alternative_names IS NOT NULL
            """
        )
    elif "aliases_seed" in columns:
        alias_queries.append(
            f"""
            SELECT
                NULLIF(TRIM(CAST({id_expr} AS VARCHAR)), '') AS openalex_id,
                CAST(unnest(json_extract_string(aliases_seed, '$[*]')) AS VARCHAR) AS raw_name
            FROM src
            WHERE aliases_seed IS NOT NULL
              AND json_valid(aliases_seed)
            """
        )
    if "acronyms" in columns:
        alias_queries.append(
            f"""
            SELECT
                NULLIF(TRIM(CAST({id_expr} AS VARCHAR)), '') AS openalex_id,
                CAST(unnest(acronyms) AS VARCHAR) AS raw_name
            FROM src
            WHERE acronyms IS NOT NULL
            """
        )
    elif "acronyms_seed" in columns:
        alias_queries.append(
            f"""
            SELECT
                NULLIF(TRIM(CAST({id_expr} AS VARCHAR)), '') AS openalex_id,
                CAST(unnest(json_extract_string(acronyms_seed, '$[*]')) AS VARCHAR) AS raw_name
            FROM src
            WHERE acronyms_seed IS NOT NULL
              AND json_valid(acronyms_seed)
            """
        )
    _create_temp_table(
        con,
        "openalex_aliases",
        f"""
        WITH src AS (
            {source_sql}
        ),
        exploded AS (
            {' UNION ALL '.join(alias_queries)}
        )
        SELECT DISTINCT
            openalex_id,
            {sql_normalize_alpha_expr('raw_name')} AS alias_norm
        FROM exploded
        WHERE openalex_id IS NOT NULL
          AND raw_name IS NOT NULL
          AND TRIM(raw_name) != ''
          AND {sql_normalize_alpha_expr('raw_name')} != ''
        """,
    )


def _create_ref_inst_alias_table(con: duckdb.DuckDBPyConnection) -> None:
    _create_temp_table(
        con,
        "ref_inst_aliases",
        """
        SELECT DISTINCT
            l.ref_inst_id,
            a.alias_norm
        FROM ref_inst_links AS l
        JOIN unitid_aliases AS a
          ON a.unitid = l.main_unitid
        WHERE l.ref_inst_id IS NOT NULL

        UNION ALL

        SELECT DISTINCT
            l.ref_inst_id,
            a.alias_norm
        FROM ref_inst_links AS l
        JOIN openalex_aliases AS a
          ON a.openalex_id = l.openalex_id
        WHERE l.ref_inst_id IS NOT NULL
        """,
    )


def _create_cip_title_lookup_table(con: duckdb.DuckDBPyConnection, path: str | None) -> None:
    if path is None:
        _create_temp_table(con, "cip_title_lookup", _EMPTY_CIP_LOOKUP_QUERY)
        return
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        columns = _describe_parquet_columns(path)
        cip_col = _first_present(columns, ["cip_code_4", "cip_code", "cip"])
        title_col = _first_present(columns, ["cip_title", "title", "CIPTitle"])
        if cip_col is None or title_col is None:
            _create_temp_table(con, "cip_title_lookup", _EMPTY_CIP_LOOKUP_QUERY)
            return
        _create_temp_table(
            con,
            "cip_title_lookup",
            f"""
            SELECT DISTINCT
                TRY_CAST({_sql_identifier(cip_col)} AS BIGINT) AS cip,
                {sql_normalize_alpha_expr(_sql_identifier(title_col))} AS title_norm
            FROM read_parquet('{escape_sql_literal(path)}')
            WHERE TRY_CAST({_sql_identifier(cip_col)} AS BIGINT) IS NOT NULL
              AND {sql_normalize_alpha_expr(_sql_identifier(title_col))} != ''
            """,
        )
        return
    header_line = path_obj.read_text(encoding="utf-8", errors="replace").splitlines()[0] if path_obj.exists() else ""
    use_simple_two_col_csv = "CIPCode" in header_line and "CIPTitle" in header_line and "CIPFamily" not in header_line
    if use_simple_two_col_csv:
        _create_temp_table(
            con,
            "cip_title_lookup",
            f"""
            WITH raw AS (
                SELECT
                    REGEXP_REPLACE(CIPCode, '[^0-9]', '', 'g') AS cip_digits,
                    {sql_normalize_alpha_expr('CIPTitle')} AS title_norm
                FROM read_csv(
                    '{escape_sql_literal(path)}',
                    columns={{
                        'CIPCode':'VARCHAR',
                        'CIPTitle':'VARCHAR'
                    }},
                    header=TRUE,
                    auto_detect=FALSE,
                    delim=',',
                    quote='\"',
                    escape='\"',
                    strict_mode=false,
                    null_padding=true,
                    ignore_errors=true
                )
            )
            SELECT DISTINCT
                TRY_CAST(cip_digits AS BIGINT) AS cip,
                title_norm
            FROM raw
            WHERE LENGTH(cip_digits) = 4
              AND title_norm != ''
            """,
        )
        return
    _create_temp_table(
        con,
        "cip_title_lookup",
        f"""
        WITH raw AS (
            SELECT
                REGEXP_REPLACE(CIPCode, '[^0-9]', '', 'g') AS cip_digits,
                {sql_normalize_alpha_expr('CIPTitle')} AS title_norm
            FROM read_csv(
                '{escape_sql_literal(path)}',
                columns={{
                    'CIPFamily':'VARCHAR',
                    'CIPCode':'VARCHAR',
                    'Action':'VARCHAR',
                    'TextChange':'VARCHAR',
                    'CIPTitle':'VARCHAR',
                    'CIPDefinition':'VARCHAR',
                    'CrossReferences':'VARCHAR',
                    'Examples':'VARCHAR'
                }},
                header=TRUE,
                auto_detect=FALSE,
                delim=',',
                quote='\"',
                escape='\"',
                strict_mode=false,
                null_padding=true,
                ignore_errors=true
            )
        )
        SELECT DISTINCT
            TRY_CAST(cip_digits AS BIGINT) AS cip,
            title_norm
        FROM raw
        WHERE LENGTH(cip_digits) = 4
          AND title_norm != ''
        """,
    )


def _create_name2nat_table(
    con: duckdb.DuckDBPyConnection,
    *,
    name2nat_path: str | None,
    stage_cfg: dict[str, Any],
) -> None:
    if name2nat_path is None:
        _create_empty_table(
            con,
            "name2nat_scores",
            [
                ("user_id", "BIGINT"),
                ("country_candidate", "VARCHAR"),
                ("nanat_score", "DOUBLE"),
                ("anglo_pressure", "DOUBLE"),
                ("nanat_anglo_crowding_ind", "BIGINT"),
            ],
        )
        return
    raw_cols = _describe_parquet_columns(name2nat_path)
    full_col = _first_present(raw_cols, ["pred_nats_full_json", "pred_nats_full", "pred_nats_name_json", "pred_nats_name"])
    first_col = _first_present(raw_cols, ["pred_nats_first_json", "pred_nats_first"])
    last_col = _first_present(raw_cols, ["pred_nats_last_json", "pred_nats_last"])
    if full_col is None:
        _create_empty_table(
            con,
            "name2nat_scores",
            [
                ("user_id", "BIGINT"),
                ("country_candidate", "VARCHAR"),
                ("nanat_score", "DOUBLE"),
                ("anglo_pressure", "DOUBLE"),
                ("nanat_anglo_crowding_ind", "BIGINT"),
            ],
        )
        return
    anglo_cutoff = float(stage_cfg.get("name2nat_anglo_pressure_cutoff", 0.35))
    w_full_default = float(stage_cfg.get("name2nat_weight_full_default", 0.55))
    w_last_default = float(stage_cfg.get("name2nat_weight_last_default", 0.35))
    w_full_crowded = float(stage_cfg.get("name2nat_weight_full_anglo_crowded", 0.30))
    w_last_crowded = float(stage_cfg.get("name2nat_weight_last_anglo_crowded", 0.60))
    w_first = float(stage_cfg.get("name2nat_weight_first", 0.10))
    anglo_list = ", ".join(sql_string_literal(country) for country in ANGLO_COUNTRIES)
    _create_temp_table(
        con,
        "name2nat_scores",
        f"""
        WITH raw AS (
            SELECT
                p.user_id,
                p.fullname_clean,
                CAST(a.{_sql_identifier(full_col)} AS VARCHAR) AS full_json,
                {f"CAST(a.{_sql_identifier(first_col)} AS VARCHAR)" if first_col else "NULL::VARCHAR"} AS first_json,
                {f"CAST(a.{_sql_identifier(last_col)} AS VARCHAR)" if last_col else "NULL::VARCHAR"} AS last_json
            FROM profiles_src AS p
            JOIN read_parquet('{escape_sql_literal(name2nat_path)}') AS a
              ON TRIM(CAST(a.fullname_clean AS VARCHAR)) = p.fullname_clean
        ),
        full_raw AS (
            SELECT
                user_id,
                fullname_clean,
                key AS raw_country,
                TRY_CAST(json_extract(full_json, '$."' || key || '"') AS DOUBLE) AS prob
            FROM raw,
                 UNNEST(CASE WHEN full_json IS NOT NULL AND json_valid(full_json) THEN json_keys(full_json) ELSE []::VARCHAR[] END) AS keys(key)
        ),
        first_raw AS (
            SELECT
                user_id,
                fullname_clean,
                key AS raw_country,
                TRY_CAST(json_extract(first_json, '$."' || key || '"') AS DOUBLE) AS prob
            FROM raw,
                 UNNEST(CASE WHEN first_json IS NOT NULL AND json_valid(first_json) THEN json_keys(first_json) ELSE []::VARCHAR[] END) AS keys(key)
        ),
        last_raw AS (
            SELECT
                user_id,
                fullname_clean,
                key AS raw_country,
                TRY_CAST(json_extract(last_json, '$."' || key || '"') AS DOUBLE) AS prob
            FROM raw,
                 UNNEST(CASE WHEN last_json IS NOT NULL AND json_valid(last_json) THEN json_keys(last_json) ELSE []::VARCHAR[] END) AS keys(key)
        ),
        full_std AS (
            SELECT
                fr.user_id,
                fr.fullname_clean,
                COALESCE(cm.std_country, fr.raw_country) AS country_candidate,
                SUM(fr.prob) AS prob
            FROM full_raw AS fr
            LEFT JOIN country_label_map AS cm
              ON UPPER(TRIM(fr.raw_country)) = cm.lookup_upper
            WHERE fr.prob > 0
            GROUP BY 1, 2, 3
        ),
        first_std AS (
            SELECT
                fr.user_id,
                fr.fullname_clean,
                COALESCE(cm.std_country, fr.raw_country) AS country_candidate,
                SUM(fr.prob) AS prob
            FROM first_raw AS fr
            LEFT JOIN country_label_map AS cm
              ON UPPER(TRIM(fr.raw_country)) = cm.lookup_upper
            WHERE fr.prob > 0
            GROUP BY 1, 2, 3
        ),
        last_std AS (
            SELECT
                lr.user_id,
                lr.fullname_clean,
                COALESCE(cm.std_country, lr.raw_country) AS country_candidate,
                SUM(lr.prob) AS prob
            FROM last_raw AS lr
            LEFT JOIN country_label_map AS cm
              ON UPPER(TRIM(lr.raw_country)) = cm.lookup_upper
            WHERE lr.prob > 0
            GROUP BY 1, 2, 3
        ),
        full_norm AS (
            SELECT
                user_id,
                fullname_clean,
                country_candidate,
                prob / NULLIF(SUM(prob) OVER (PARTITION BY user_id, fullname_clean), 0.0) AS prob,
                'full' AS prob_type
            FROM full_std
        ),
        first_norm AS (
            SELECT
                user_id,
                fullname_clean,
                country_candidate,
                prob / NULLIF(SUM(prob) OVER (PARTITION BY user_id, fullname_clean), 0.0) AS prob,
                'first' AS prob_type
            FROM first_std
        ),
        last_norm AS (
            SELECT
                user_id,
                fullname_clean,
                country_candidate,
                prob / NULLIF(SUM(prob) OVER (PARTITION BY user_id, fullname_clean), 0.0) AS prob,
                'last' AS prob_type
            FROM last_std
        ),
        merged AS (
            SELECT
                user_id,
                fullname_clean,
                country_candidate,
                SUM(CASE WHEN prob_type = 'full' THEN prob ELSE 0.0 END) AS nanat_prob_full,
                SUM(CASE WHEN prob_type = 'first' THEN prob ELSE 0.0 END) AS nanat_prob_first,
                SUM(CASE WHEN prob_type = 'last' THEN prob ELSE 0.0 END) AS nanat_prob_last
            FROM (
                SELECT * FROM full_norm
                UNION ALL
                SELECT * FROM first_norm
                UNION ALL
                SELECT * FROM last_norm
            )
            GROUP BY 1, 2, 3
        ),
        anglo AS (
            SELECT
                user_id,
                fullname_clean,
                SUM(CASE WHEN country_candidate IN ({anglo_list}) THEN nanat_prob_full ELSE 0.0 END) AS anglo_prob_full,
                SUM(CASE WHEN country_candidate IN ({anglo_list}) THEN nanat_prob_last ELSE 0.0 END) AS anglo_prob_last
            FROM merged
            GROUP BY 1, 2
        ),
        scored AS (
            SELECT
                m.user_id,
                m.fullname_clean,
                m.country_candidate,
                GREATEST(a.anglo_prob_full - a.anglo_prob_last, 0.0) AS anglo_pressure,
                CASE WHEN GREATEST(a.anglo_prob_full - a.anglo_prob_last, 0.0) > {anglo_cutoff} THEN 1 ELSE 0 END AS nanat_anglo_crowding_ind,
                (
                    CASE WHEN GREATEST(a.anglo_prob_full - a.anglo_prob_last, 0.0) > {anglo_cutoff} THEN {w_full_crowded} ELSE {w_full_default} END
                ) * m.nanat_prob_full
                + (
                    CASE WHEN GREATEST(a.anglo_prob_full - a.anglo_prob_last, 0.0) > {anglo_cutoff} THEN {w_last_crowded} ELSE {w_last_default} END
                ) * m.nanat_prob_last
                + {w_first} * m.nanat_prob_first AS nanat_score_raw
            FROM merged AS m
            JOIN anglo AS a
              ON a.user_id = m.user_id
             AND a.fullname_clean = m.fullname_clean
        ),
        normalized AS (
            SELECT
                user_id,
                fullname_clean,
                country_candidate,
                anglo_pressure,
                nanat_anglo_crowding_ind,
                nanat_score_raw / NULLIF(SUM(nanat_score_raw) OVER (PARTITION BY user_id, fullname_clean), 0.0) AS nanat_score
            FROM scored
        )
        SELECT
            user_id,
            country_candidate,
            MAX(nanat_score) AS nanat_score,
            MAX(anglo_pressure) AS anglo_pressure,
            MAX(nanat_anglo_crowding_ind) AS nanat_anglo_crowding_ind
        FROM normalized
        GROUP BY 1, 2
        """,
    )


def _create_nametrace_tables(
    con: duckdb.DuckDBPyConnection,
    *,
    nametrace_wide_path: str | None,
    nametrace_long_path: str | None,
) -> None:
    if nametrace_wide_path is None:
        _create_empty_table(con, "gender_scores", [("user_id", "BIGINT"), ("f_prob_nt", "DOUBLE")])
    else:
        _create_temp_table(
            con,
            "gender_scores",
            f"""
            SELECT
                p.user_id,
                MAX(TRY_CAST(w.f_prob_nt AS DOUBLE)) AS f_prob_nt
            FROM profiles_src AS p
            LEFT JOIN read_parquet('{escape_sql_literal(nametrace_wide_path)}') AS w
              ON TRIM(CAST(w.fullname_clean AS VARCHAR)) = p.fullname_clean
            GROUP BY 1
            """,
        )
    if nametrace_long_path is None:
        _create_empty_table(
            con,
            "nametrace_country_scores",
            [("user_id", "BIGINT"), ("country_candidate", "VARCHAR"), ("nametrace_score", "DOUBLE")],
        )
        _create_empty_table(
            con,
            "user_region_scores",
            [("user_id", "BIGINT"), ("subregion_candidate", "VARCHAR"), ("nt_subregion_score", "DOUBLE")],
        )
        return
    _create_temp_table(
        con,
        "user_region_scores",
        f"""
        SELECT
            p.user_id,
            CAST(l.region AS VARCHAR) AS subregion_candidate,
            MAX(TRY_CAST(l.prob AS DOUBLE)) AS nt_subregion_score
        FROM profiles_src AS p
        JOIN read_parquet('{escape_sql_literal(nametrace_long_path)}') AS l
          ON TRIM(CAST(l.fullname_clean AS VARCHAR)) = p.fullname_clean
        WHERE CAST(l.region AS VARCHAR) IS NOT NULL
          AND TRY_CAST(l.prob AS DOUBLE) > 0
        GROUP BY 1, 2
        """,
    )
    _create_temp_table(
        con,
        "nametrace_country_scores",
        """
        SELECT
            urs.user_id,
            sm.country_candidate,
            SUM(urs.nt_subregion_score / NULLIF(cbs.n_countries, 0)) AS nametrace_score
        FROM user_region_scores AS urs
        JOIN subregion_map AS sm
          ON sm.subregion_candidate = urs.subregion_candidate
        JOIN countries_by_subregion AS cbs
          ON cbs.subregion_candidate = urs.subregion_candidate
        GROUP BY 1, 2
        """,
    )


def _create_institution_country_scores(con: duckdb.DuckDBPyConnection, stage_cfg: dict[str, Any]) -> None:
    weights = dict(stage_cfg.get("institution_country_degree_weights", {}))
    default_weight = float(stage_cfg.get("institution_country_default_weight", 0.4))
    exclude_us = coerce_bool(stage_cfg.get("institution_country_exclude_us", True), True)
    degree_cases = " ".join(
        f"WHEN degree_clean = {sql_string_literal(label)} THEN {float(weight)}"
        for label, weight in weights.items()
    )
    exclude_us_clause = "AND university_country_std != 'United States'" if exclude_us else ""
    _create_temp_table(
        con,
        "institution_country_scores",
        f"""
        WITH source AS (
            SELECT
                user_id,
                institution_country_std AS country_candidate,
                CASE
                    {degree_cases}
                    ELSE {default_weight}
                END AS institution_weight_raw
            FROM educ_clean_all
            WHERE institution_country_std IS NOT NULL
              {exclude_us_clause}
        ),
        grouped AS (
            SELECT
                user_id,
                country_candidate,
                MAX(institution_weight_raw) AS institution_weight_raw
            FROM source
            GROUP BY 1, 2
        )
        SELECT
            user_id,
            country_candidate,
            institution_weight_raw / NULLIF(SUM(institution_weight_raw) OVER (PARTITION BY user_id), 0.0) AS institution_score
        FROM grouped
        """,
    )


def _create_country_candidates(
    con: duckdb.DuckDBPyConnection,
    *,
    stage_cfg: dict[str, Any],
    use_seed: bool,
) -> None:
    if use_seed:
        _create_temp_table(
            con,
            "country_candidates",
            """
            SELECT
                user_id,
                country_candidate,
                country_score,
                nanat_score,
                institution_score,
                nametrace_score,
                nanat_subregion_score,
                nt_subregion_score,
                subregion_candidate,
                country_uncertain_ind,
                anglo_pressure,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY country_score DESC NULLS LAST, country_candidate
                ) AS country_rank
            FROM country_candidates_seed
            """,
        )
        return
    weight_cfg = dict(stage_cfg.get("country_score_weights", {}))
    w_nanat = float(weight_cfg.get("name2nat", 0.5))
    w_inst = float(weight_cfg.get("institution", 0.35))
    w_nt = float(weight_cfg.get("nametrace", 0.15))
    fallback_weight = float(weight_cfg.get("user_country_fallback", 0.25))
    gap_cutoff = float(stage_cfg.get("country_uncertainty_gap", 0.10))
    top_n = int(stage_cfg.get("top_country_candidates_per_user", 5))
    min_score = float(stage_cfg.get("min_country_candidate_score", 0.02))
    _create_temp_table(
        con,
        "country_candidates",
        f"""
        WITH base AS (
            SELECT
                user_id,
                country_candidate,
                MAX(nanat_score) AS nanat_score,
                MAX(anglo_pressure) AS anglo_pressure,
                MAX(nanat_anglo_crowding_ind) AS nanat_anglo_crowding_ind,
                0.0::DOUBLE AS nametrace_score,
                0.0::DOUBLE AS institution_score
            FROM name2nat_scores
            GROUP BY 1, 2
            UNION ALL
            SELECT
                user_id,
                country_candidate,
                0.0::DOUBLE AS nanat_score,
                0.0::DOUBLE AS anglo_pressure,
                0::BIGINT AS nanat_anglo_crowding_ind,
                MAX(nametrace_score) AS nametrace_score,
                0.0::DOUBLE AS institution_score
            FROM nametrace_country_scores
            GROUP BY 1, 2
            UNION ALL
            SELECT
                user_id,
                country_candidate,
                0.0::DOUBLE AS nanat_score,
                0.0::DOUBLE AS anglo_pressure,
                0::BIGINT AS nanat_anglo_crowding_ind,
                0.0::DOUBLE AS nametrace_score,
                MAX(institution_score) AS institution_score
            FROM institution_country_scores
            GROUP BY 1, 2
        ),
        grouped AS (
            SELECT
                user_id,
                country_candidate,
                MAX(nanat_score) AS nanat_score,
                MAX(nametrace_score) AS nametrace_score,
                MAX(institution_score) AS institution_score,
                MAX(anglo_pressure) AS anglo_pressure,
                MAX(nanat_anglo_crowding_ind) AS nanat_anglo_crowding_ind,
                {w_nanat} * MAX(nanat_score)
                    + {w_inst} * MAX(institution_score)
                    + {w_nt} * MAX(nametrace_score) AS country_score_raw
            FROM base
            GROUP BY 1, 2
        ),
        score_totals AS (
            SELECT
                user_id,
                SUM(country_score_raw) AS raw_total
            FROM grouped
            GROUP BY 1
        ),
        fallback_rows AS (
            SELECT
                p.user_id,
                p.user_country_std AS country_candidate,
                0.0::DOUBLE AS nanat_score,
                0.0::DOUBLE AS nametrace_score,
                0.0::DOUBLE AS institution_score,
                0.0::DOUBLE AS anglo_pressure,
                0::BIGINT AS nanat_anglo_crowding_ind,
                CASE WHEN {fallback_weight} > 0 THEN {fallback_weight} ELSE 1.0 END AS country_score_raw
            FROM profiles_src AS p
            LEFT JOIN score_totals AS st
              ON st.user_id = p.user_id
            WHERE p.user_country_std IS NOT NULL
              AND COALESCE(st.raw_total, 0.0) <= 0.0
        ),
        regrouped AS (
            SELECT
                user_id,
                country_candidate,
                MAX(nanat_score) AS nanat_score,
                MAX(nametrace_score) AS nametrace_score,
                MAX(institution_score) AS institution_score,
                MAX(anglo_pressure) AS anglo_pressure,
                MAX(nanat_anglo_crowding_ind) AS nanat_anglo_crowding_ind,
                SUM(country_score_raw) AS country_score_raw
            FROM (
                SELECT * FROM grouped
                UNION ALL
                SELECT * FROM fallback_rows
            )
            WHERE country_candidate IS NOT NULL
            GROUP BY 1, 2
        ),
        scored AS (
            SELECT
                g.user_id,
                g.country_candidate,
                g.nanat_score,
                g.institution_score,
                g.nametrace_score,
                COALESCE(sm.subregion_candidate, g.country_candidate) AS subregion_candidate,
                g.anglo_pressure,
                g.nanat_anglo_crowding_ind,
                g.country_score_raw / NULLIF(SUM(g.country_score_raw) OVER (PARTITION BY g.user_id), 0.0) AS country_score
            FROM regrouped AS g
            LEFT JOIN subregion_map AS sm
              ON sm.country_candidate = g.country_candidate
        ),
        nanat_subregion AS (
            SELECT
                user_id,
                subregion_candidate,
                SUM(nanat_score) AS nanat_subregion_score
            FROM scored
            GROUP BY 1, 2
        ),
        enriched AS (
            SELECT
                s.user_id,
                s.country_candidate,
                s.country_score,
                s.nanat_score,
                s.institution_score,
                s.nametrace_score,
                COALESCE(ns.nanat_subregion_score, s.nanat_score) AS nanat_subregion_score,
                COALESCE(urs.nt_subregion_score, 0.0) AS nt_subregion_score,
                s.subregion_candidate,
                s.anglo_pressure,
                s.nanat_anglo_crowding_ind
            FROM scored AS s
            LEFT JOIN nanat_subregion AS ns
              ON ns.user_id = s.user_id
             AND ns.subregion_candidate = s.subregion_candidate
            LEFT JOIN user_region_scores AS urs
              ON urs.user_id = s.user_id
             AND urs.subregion_candidate = s.subregion_candidate
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY country_score DESC NULLS LAST, country_candidate
                ) AS country_rank
            FROM enriched
        ),
        gap_base AS (
            SELECT
                user_id,
                MAX(CASE WHEN country_rank = 1 THEN country_score ELSE 0.0 END) AS top_country_score,
                MAX(CASE WHEN country_rank = 2 THEN country_score ELSE 0.0 END) AS second_country_score,
                MAX(nanat_anglo_crowding_ind) AS crowding_ind
            FROM ranked
            GROUP BY 1
        ),
        gap_flags AS (
            SELECT
                user_id,
                CASE
                    WHEN (top_country_score - second_country_score) < {gap_cutoff} OR crowding_ind > 0 THEN 1
                    ELSE 0
                END AS country_uncertain_ind
            FROM gap_base
        )
        SELECT
            r.user_id,
            r.country_candidate,
            r.country_score,
            r.nanat_score,
            r.institution_score,
            r.nametrace_score,
            r.nanat_subregion_score,
            r.nt_subregion_score,
            r.subregion_candidate,
            gf.country_uncertain_ind,
            r.anglo_pressure,
            r.country_rank
        FROM ranked AS r
        JOIN gap_flags AS gf
          ON gf.user_id = r.user_id
        WHERE r.country_rank <= {top_n}
          AND (r.country_score >= {min_score} OR r.country_rank = 1)
        """,
    )


def _create_candidate_rerank_tables(
    con: duckdb.DuckDBPyConnection,
    stage_cfg: dict[str, Any],
) -> None:
    unitid_rank_expr = sql_text_relation_rank_expr("r.source_text", "a.alias_norm")
    cip_rank_expr = sql_text_relation_rank_expr("r.source_text", "c.title_norm")
    unitid_jw_threshold = float(stage_cfg.get("unitid_soft_gate_jw_threshold", 0.92))
    unitid_overlap_threshold = float(stage_cfg.get("unitid_soft_gate_token_overlap_threshold", 0.5))
    unitid_overlap_min_shared = int(stage_cfg.get("unitid_soft_gate_min_shared_tokens", 2))
    _create_temp_table(
        con,
        "ref_inst_candidate_rows",
        f"""
        WITH raw_source AS (
            SELECT
                user_id,
                education_number,
                university_raw AS source_text,
                university_country_std,
                UNNEST(deterministic_ref_inst_candidates) AS candidate,
                generate_subscripts(deterministic_ref_inst_candidates, 1) AS original_rank
            FROM educ_src
            WHERE deterministic_ref_inst_candidates IS NOT NULL
              AND array_length(deterministic_ref_inst_candidates) > 0
        ),
        raw AS (
            SELECT
                user_id,
                education_number,
                source_text,
                university_country_std,
                CAST(candidate.ref_inst_id AS VARCHAR) AS ref_inst_id,
                TRY_CAST(candidate.hybrid_score AS DOUBLE) AS candidate_score,
                original_rank
            FROM raw_source
        ),
        linked AS (
            SELECT
                r.user_id,
                r.education_number,
                r.source_text,
                r.university_country_std,
                r.ref_inst_id,
                l.openalex_id,
                l.main_unitid AS unitid,
                r.candidate_score,
                r.original_rank,
                oe.country_candidate AS openalex_country_candidate
            FROM raw AS r
            LEFT JOIN ref_inst_links AS l
              ON l.ref_inst_id = r.ref_inst_id
            LEFT JOIN openalex_entities AS oe
              ON oe.openalex_id = l.openalex_id
        ),
        scored AS (
            SELECT
                r.user_id,
                r.education_number,
                r.ref_inst_id,
                r.openalex_id,
                r.unitid,
                r.openalex_country_candidate,
                r.university_country_std,
                r.candidate_score,
                r.original_rank,
                COALESCE(MAX({unitid_rank_expr}), 0) AS text_match_rank,
                COALESCE(
                    MAX(
                        CASE
                            WHEN {sql_normalize_alpha_expr("r.source_text")} = '' OR a.alias_norm = '' THEN NULL
                            ELSE jaro_winkler_similarity({sql_normalize_alpha_expr("r.source_text")}, a.alias_norm)
                        END
                    ),
                    0.0
                ) AS alias_jw_max,
                COALESCE(
                    MAX(
                        CASE
                            WHEN list_count(
                                list_filter(
                                    str_split({sql_normalize_alpha_expr("r.source_text")}, ' '),
                                    x -> x <> ''
                                )
                            ) = 0
                             OR list_count(
                                list_filter(
                                    str_split(a.alias_norm, ' '),
                                    x -> x <> ''
                                )
                            ) = 0
                                THEN 0.0
                            ELSE CAST(
                                list_count(
                                    list_filter(
                                        list_distinct(
                                            list_filter(
                                                str_split({sql_normalize_alpha_expr("r.source_text")}, ' '),
                                                x -> x <> ''
                                            )
                                        ),
                                        x -> list_contains(
                                            list_distinct(
                                                list_filter(
                                                    str_split(a.alias_norm, ' '),
                                                    x -> x <> ''
                                                )
                                            ),
                                            x
                                        )
                                    )
                                ) AS DOUBLE
                            ) / CAST(
                                LEAST(
                                    list_count(
                                        list_distinct(
                                            list_filter(
                                                str_split({sql_normalize_alpha_expr("r.source_text")}, ' '),
                                                x -> x <> ''
                                            )
                                        )
                                    ),
                                    list_count(
                                        list_distinct(
                                            list_filter(
                                                str_split(a.alias_norm, ' '),
                                                x -> x <> ''
                                            )
                                        )
                                    )
                                ) AS DOUBLE
                            )
                        END
                    ),
                    0.0
                ) AS alias_token_overlap_max,
                COALESCE(
                    MAX(
                        list_count(
                            list_filter(
                                list_distinct(
                                    list_filter(
                                        str_split({sql_normalize_alpha_expr("r.source_text")}, ' '),
                                        x -> x <> ''
                                    )
                                ),
                                x -> list_contains(
                                    list_distinct(
                                        list_filter(
                                            str_split(a.alias_norm, ' '),
                                            x -> x <> ''
                                        )
                                    ),
                                    x
                                )
                            )
                        )
                    ),
                    0
                ) AS alias_shared_token_count_max
            FROM linked AS r
            LEFT JOIN ref_inst_aliases AS a
              ON a.ref_inst_id = r.ref_inst_id
            GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
        ),
        ranked AS (
        SELECT
            user_id,
            education_number,
            ref_inst_id,
            openalex_id,
            unitid,
            COALESCE(openalex_country_candidate, university_country_std) AS institution_country_candidate,
            candidate_score,
            original_rank,
            text_match_rank,
            alias_jw_max,
            alias_token_overlap_max,
            alias_shared_token_count_max,
            CASE
                WHEN text_match_rank >= 1 THEN 2
                WHEN alias_jw_max >= {unitid_jw_threshold}
                  OR (
                    alias_token_overlap_max >= {unitid_overlap_threshold}
                    AND alias_shared_token_count_max >= {unitid_overlap_min_shared}
                  )
                    THEN 1
                ELSE 0
            END AS selection_bucket,
            ROW_NUMBER() OVER (
                PARTITION BY user_id, education_number
                ORDER BY
                    CASE
                        WHEN text_match_rank >= 1 THEN 2
                        WHEN alias_jw_max >= {unitid_jw_threshold}
                          OR (
                            alias_token_overlap_max >= {unitid_overlap_threshold}
                            AND alias_shared_token_count_max >= {unitid_overlap_min_shared}
                          )
                            THEN 1
                        ELSE 0
                    END DESC,
                    candidate_score DESC NULLS LAST,
                    text_match_rank DESC,
                    alias_jw_max DESC,
                    alias_token_overlap_max DESC,
                    alias_shared_token_count_max DESC,
                    original_rank,
                    unitid
            ) AS candidate_rank,
            COUNT(*) OVER (PARTITION BY user_id, education_number) AS candidate_count
        FROM scored
        )
        SELECT * FROM ranked
        """,
    )
    _create_temp_table(
        con,
        "unitid_candidate_rows",
        """
        WITH collapsed AS (
            SELECT
                user_id,
                education_number,
                unitid,
                MAX(candidate_score) AS candidate_score,
                MIN(original_rank) AS original_rank,
                MAX(text_match_rank) AS text_match_rank,
                MAX(alias_jw_max) AS alias_jw_max,
                MAX(alias_token_overlap_max) AS alias_token_overlap_max,
                MAX(alias_shared_token_count_max) AS alias_shared_token_count_max,
                MAX(selection_bucket) AS selection_bucket
            FROM ref_inst_candidate_rows
            WHERE unitid IS NOT NULL
            GROUP BY 1, 2, 3
        )
        SELECT
            user_id,
            education_number,
            unitid,
            candidate_score,
            original_rank,
            text_match_rank,
            alias_jw_max,
            alias_token_overlap_max,
            alias_shared_token_count_max,
            selection_bucket,
            ROW_NUMBER() OVER (
                PARTITION BY user_id, education_number
                ORDER BY selection_bucket DESC,
                         candidate_score DESC NULLS LAST,
                         text_match_rank DESC,
                         alias_jw_max DESC,
                         alias_token_overlap_max DESC,
                         alias_shared_token_count_max DESC,
                         original_rank,
                         unitid
            ) AS candidate_rank,
            COUNT(*) OVER (PARTITION BY user_id, education_number) AS candidate_count
        FROM collapsed
        """,
    )
    _create_temp_table(
        con,
        "unitid_top1_summary",
        f"""
        WITH exact_subset AS (
            SELECT
                user_id,
                education_number,
                unitid,
                candidate_score,
                candidate_count,
                text_match_rank,
                alias_jw_max,
                alias_token_overlap_max,
                alias_shared_token_count_max,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, education_number
                    ORDER BY
                        candidate_score DESC NULLS LAST,
                        text_match_rank DESC,
                        alias_jw_max DESC,
                        alias_token_overlap_max DESC,
                        alias_shared_token_count_max DESC,
                        original_rank,
                        unitid
                ) AS rn
            FROM unitid_candidate_rows
            WHERE text_match_rank >= 1
        ),
        soft_gate AS (
            SELECT
                user_id,
                education_number,
                unitid,
                candidate_score,
                candidate_count,
                text_match_rank,
                alias_jw_max,
                alias_token_overlap_max,
                alias_shared_token_count_max,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, education_number
                    ORDER BY
                        candidate_score DESC NULLS LAST,
                        alias_jw_max DESC,
                        alias_token_overlap_max DESC,
                        alias_shared_token_count_max DESC,
                        original_rank,
                        unitid
                ) AS rn
            FROM unitid_candidate_rows
            WHERE text_match_rank < 1
              AND selection_bucket = 1
        ),
        candidate_counts AS (
            SELECT
                user_id,
                education_number,
                MAX(candidate_count) AS unitid_candidate_count
            FROM unitid_candidate_rows
            GROUP BY 1, 2
        )
        SELECT
            ids.user_id,
            ids.education_number,
            CASE
                WHEN es.unitid IS NOT NULL THEN es.unitid
                WHEN sg.unitid IS NOT NULL THEN sg.unitid
                ELSE NULL
            END AS unitid_top1,
            CASE
                WHEN es.unitid IS NOT NULL THEN es.candidate_score
                WHEN sg.unitid IS NOT NULL THEN sg.candidate_score
                ELSE NULL
            END AS unitid_top1_score,
            COALESCE(cc.unitid_candidate_count, 0) AS unitid_candidate_count,
            CASE
                WHEN es.unitid IS NOT NULL THEN 'exact_or_subset_alias'
                WHEN sg.unitid IS NOT NULL THEN 'soft_alias_gate'
                WHEN COALESCE(cc.unitid_candidate_count, 0) = 0 THEN 'no_deterministic_candidates'
                ELSE 'failed_alias_gate'
            END AS unitid_selection_reason
        FROM (
            SELECT DISTINCT user_id, education_number
            FROM educ_src
        ) AS ids
        LEFT JOIN exact_subset AS es
          ON es.user_id = ids.user_id
         AND es.education_number = ids.education_number
         AND es.rn = 1
        LEFT JOIN soft_gate AS sg
          ON sg.user_id = ids.user_id
         AND sg.education_number = ids.education_number
         AND sg.rn = 1
        LEFT JOIN candidate_counts AS cc
          ON cc.user_id = ids.user_id
         AND cc.education_number = ids.education_number
        """,
    )
    _create_temp_table(
        con,
        "openalex_top1_summary",
        """
        SELECT
            user_id,
            education_number,
            openalex_id AS openalex_top1,
            institution_country_candidate AS institution_country_candidate
        FROM (
            SELECT
                user_id,
                education_number,
                openalex_id,
                institution_country_candidate,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, education_number
                    ORDER BY selection_bucket DESC,
                             candidate_score DESC NULLS LAST,
                             text_match_rank DESC,
                             alias_jw_max DESC,
                             alias_token_overlap_max DESC,
                             alias_shared_token_count_max DESC,
                             original_rank,
                             openalex_id
                ) AS rn
            FROM ref_inst_candidate_rows
            WHERE openalex_id IS NOT NULL
              AND institution_country_candidate IS NOT NULL
        )
        WHERE rn = 1
        """,
    )
    _create_temp_table(
        con,
        "cip_candidate_rows",
        f"""
        WITH raw_source AS (
            SELECT
                user_id,
                education_number,
                COALESCE(field_raw, field, field_clean) AS source_text,
                UNNEST(deterministic_cip_candidates) AS candidate,
                generate_subscripts(deterministic_cip_candidates, 1) AS original_rank
            FROM educ_src
            WHERE deterministic_cip_candidates IS NOT NULL
              AND array_length(deterministic_cip_candidates) > 0
        ),
        raw AS (
            SELECT
                user_id,
                education_number,
                source_text,
                {cip_code_to_cip4_sql('candidate.cip_code')} AS cip,
                TRY_CAST(candidate.hybrid_score AS DOUBLE) AS candidate_score,
                original_rank
            FROM raw_source
        ),
        collapsed AS (
            SELECT
                user_id,
                education_number,
                source_text,
                cip,
                candidate_score,
                original_rank
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id, education_number, cip
                        ORDER BY candidate_score DESC NULLS LAST, original_rank
                    ) AS rn
                FROM raw
                WHERE cip IS NOT NULL
            )
            WHERE rn = 1
        ),
        scored AS (
            SELECT
                r.user_id,
                r.education_number,
                r.cip,
                r.candidate_score,
                r.original_rank,
                COALESCE(MAX({cip_rank_expr}), 0) AS text_match_rank
            FROM collapsed AS r
            LEFT JOIN cip_title_lookup AS c
              ON c.cip = r.cip
            GROUP BY 1, 2, 3, 4, 5
        )
        SELECT
            user_id,
            education_number,
            cip,
            candidate_score,
            original_rank,
            text_match_rank,
            ROW_NUMBER() OVER (
                PARTITION BY user_id, education_number
                ORDER BY text_match_rank DESC, candidate_score DESC NULLS LAST, original_rank, cip
            ) AS candidate_rank,
            COUNT(*) OVER (PARTITION BY user_id, education_number) AS candidate_count
        FROM scored
        """,
    )
    _create_temp_table(
        con,
        "cip_top1_summary",
        """
        SELECT
            user_id,
            education_number,
            cip AS cip_top1,
            CASE
                WHEN text_match_rank >= 2 THEN 1.0
                ELSE candidate_score
            END AS cip_top1_score,
            candidate_count AS cip_candidate_count
        FROM cip_candidate_rows
        WHERE candidate_rank = 1
        """,
    )


def _create_educ_clean_all(con: duckdb.DuckDBPyConnection) -> None:
    cip_field_key_null_heuristic_expr = field_category_to_cip4_sql("e.field")
    cip_field_key_null_heuristic_match_expr = (
        f"(e.field_key IS NULL AND ({cip_field_key_null_heuristic_expr}) IS NOT NULL)"
    )
    _create_temp_table(
        con,
        "educ_clean_all",
        f"""
        SELECT
            e.*,
            s.unitid_stage03,
            s.school_match_score_stage03,
            s.rev_instname_clean_stage03,
            s.school_mapping_source_stage03,
            s.unitid_stage03_candidate_score,
            u.unitid_top1,
            u.unitid_top1_score,
            COALESCE(u.unitid_candidate_count, 0) AS unitid_candidate_count,
            COALESCE(u.unitid_top1, s.unitid_stage03) AS unitid,
            CASE
                WHEN u.unitid_top1 IS NOT NULL THEN 'deterministic_top1'
                WHEN s.unitid_stage03 IS NOT NULL THEN 'stage03_fallback'
                ELSE NULL
            END AS unitid_mapping_source,
            u.unitid_selection_reason,
            COALESCE(u.unitid_top1_score, s.unitid_stage03_candidate_score) AS school_match_score,
            COALESCE(u.unitid_top1_score, s.unitid_stage03_candidate_score) AS unitid_score,
            COALESCE(s.rev_instname_clean_stage03, e.university_raw_clean) AS rev_instname_clean,
            CASE WHEN COALESCE(u.unitid_top1, s.unitid_stage03) IS NOT NULL THEN 1 ELSE 0 END AS school_mapped_ind,
            f.cip_stage03,
            f.field_mapping_source_stage03,
            f.cip_stage03_candidate_score,
            c.cip_top1,
            c.cip_top1_score,
            CASE
                WHEN c.cip_top1 IS NOT NULL THEN COALESCE(c.cip_candidate_count, 0)
                WHEN {cip_field_key_null_heuristic_match_expr} THEN 1
                ELSE 0
            END AS cip_candidate_count,
            CASE
                WHEN c.cip_top1 IS NOT NULL THEN c.cip_top1
                WHEN {cip_field_key_null_heuristic_match_expr} THEN {cip_field_key_null_heuristic_expr}
                ELSE f.cip_stage03
            END AS cip,
            CASE
                WHEN c.cip_top1 IS NOT NULL THEN 'deterministic_top1'
                WHEN {cip_field_key_null_heuristic_match_expr} THEN 'field_key_null_heuristic'
                WHEN f.cip_stage03 IS NOT NULL THEN 'stage03_fallback'
                ELSE NULL
            END AS cip_mapping_source,
            CASE
                WHEN c.cip_top1 IS NOT NULL THEN 'deterministic_top1'
                WHEN {cip_field_key_null_heuristic_match_expr} THEN 'field_key_null_heuristic'
                ELSE f.field_mapping_source_stage03
            END AS field_mapping_source,
            CASE
                WHEN c.cip_top1 IS NOT NULL THEN c.cip_top1_score
                WHEN {cip_field_key_null_heuristic_match_expr} THEN 1.0
                ELSE f.cip_stage03_candidate_score
            END AS cip_score,
            CASE
                WHEN c.cip_top1 IS NOT NULL THEN 1
                WHEN {cip_field_key_null_heuristic_match_expr} THEN 1
                WHEN f.cip_stage03 IS NOT NULL THEN 1
                ELSE 0
            END AS field_mapped_ind,
            COALESCE(o.institution_country_candidate, e.university_country_std) AS institution_country_std
        FROM educ_src AS e
        LEFT JOIN school_crosswalk AS s
          ON s.university_raw_key = e.university_raw_key
        LEFT JOIN field_crosswalk AS f
          ON f.field_norm = e.field_norm
        LEFT JOIN unitid_top1_summary AS u
          ON u.user_id = e.user_id
         AND u.education_number = e.education_number
        LEFT JOIN openalex_top1_summary AS o
          ON o.user_id = e.user_id
         AND o.education_number = e.education_number
        LEFT JOIN cip_top1_summary AS c
          ON c.user_id = e.user_id
         AND c.education_number = e.education_number
        """,
    )


def _create_pruned_educ_clean(con: duckdb.DuckDBPyConnection) -> None:
    _create_temp_table(
        con,
        "educ_clean",
        """
        SELECT *
        FROM educ_clean_all
        WHERE unitid IS NOT NULL
        """,
    )


def _create_positions_clean(con: duckdb.DuckDBPyConnection) -> None:
    _create_temp_table(
        con,
        "positions_clean",
        """
        SELECT
            p.*,
            COALESCE(ekm.employer_key, p.company_raw_normed) AS employer_key,
            CASE WHEN COALESCE(ekm.employer_key, p.company_raw_normed) IS NOT NULL THEN 1 ELSE 0 END AS employer_mapped_ind,
            ekm.representative_match_type
        FROM positions_src AS p
        LEFT JOIN employer_key_map AS ekm
          ON ekm.rcid = p.rcid
        """,
    )


def _create_users_core(con: duckdb.DuckDBPyConnection) -> None:
    _create_temp_table(
        con,
        "educ_with_grad_year",
        """
        SELECT
            *,
            CASE
                WHEN ed_end_year IS NOT NULL THEN ed_end_year
                WHEN ed_start_year IS NOT NULL AND degree_clean = 'Doctor' THEN ed_start_year + 4
                WHEN ed_start_year IS NOT NULL AND degree_clean IN ('Master', 'MBA', 'Associate') THEN ed_start_year + 2
                WHEN ed_start_year IS NOT NULL THEN ed_start_year + 4
                ELSE NULL
            END AS grad_year
        FROM educ_clean
        """,
    )
    _create_temp_table(
        con,
        "educ_summary",
        """
        WITH degree_ranked AS (
            SELECT
                user_id,
                CASE
                    WHEN degree_clean IN ('Doctor', 'Master', 'MBA', 'Bachelor', 'Associate', 'Other', 'Missing') THEN degree_clean
                    ELSE 'Other'
                END AS degree_label,
                CASE
                    WHEN degree_clean = 'Doctor' THEN 5
                    WHEN degree_clean IN ('Master', 'MBA') THEN 4
                    WHEN degree_clean = 'Bachelor' THEN 3
                    WHEN degree_clean = 'Associate' THEN 2
                    WHEN degree_clean = 'Missing' THEN 0
                    ELSE 1
                END AS degree_rank
            FROM educ_with_grad_year
        ),
        highest_degree AS (
            SELECT
                user_id,
                degree_label AS highest_ed_level
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY degree_rank DESC, degree_label
                    ) AS rn
                FROM degree_ranked
            )
            WHERE rn = 1
        ),
        field_json AS (
            SELECT
                user_id,
                CAST(array_to_json(list_sort(list_distinct(list(field_clean)))) AS VARCHAR) AS fields_json
            FROM educ_with_grad_year
            WHERE field_clean IS NOT NULL
              AND TRIM(field_clean) != ''
            GROUP BY 1
        )
        SELECT
            e.user_id,
            CASE
                WHEN MAX(CASE WHEN degree_clean = 'High School' AND ed_end_year IS NOT NULL THEN 1 ELSE 0 END) = 1
                    THEN MAX(CASE WHEN degree_clean = 'High School' AND ed_end_year IS NOT NULL THEN ed_end_year - 18 ELSE NULL END)
                ELSE MIN(
                    CASE
                        WHEN degree_clean IN ('Non-Degree', 'Master', 'Doctor', 'MBA') THEN NULL
                        WHEN ed_start_year IS NOT NULL THEN ed_start_year - 18
                        WHEN ed_end_year IS NOT NULL AND degree_clean <> 'Associate' THEN ed_end_year - 23
                        WHEN ed_end_year IS NOT NULL THEN ed_end_year - 21
                        ELSE NULL
                    END
                )
            END AS est_yob,
            hd.highest_ed_level,
            COALESCE(fj.fields_json, '[]') AS fields_json,
            COALESCE(MAX(CASE WHEN match_eligible_ind = 1 THEN COALESCE(stem_ind, 0) ELSE 0 END), 0) AS stem_ind_any,
            COUNT(*) AS n_education_records,
            MAX(CASE WHEN university_country_std = 'United States' AND high_school_ind = 1 THEN 1 ELSE 0 END) AS us_hs_exact,
            MAX(CASE WHEN university_country_std = 'United States' AND match_eligible_ind = 1 THEN 1 ELSE 0 END) AS us_educ,
            MAX(CASE WHEN university_country_std = 'United States' AND degree_clean IN ('Master', 'MBA', 'Doctor') AND grad_year IS NOT NULL THEN 1 ELSE 0 END) AS ade_ind,
            MIN(CASE WHEN university_country_std = 'United States' AND degree_clean IN ('Master', 'MBA', 'Doctor') THEN grad_year ELSE NULL END) AS ade_year,
            MAX(CASE WHEN university_country_std = 'United States' THEN grad_year ELSE NULL END) AS last_grad_year
        FROM educ_with_grad_year AS e
        LEFT JOIN highest_degree AS hd
          ON hd.user_id = e.user_id
        LEFT JOIN field_json AS fj
          ON fj.user_id = e.user_id
        GROUP BY 1, 3, 4
        """,
    )
    _create_temp_table(
        con,
        "pos_summary",
        """
        SELECT
            user_id,
            COUNT(position_id) AS n_position_records,
            COUNT(DISTINCT employer_key) FILTER (WHERE employer_key IS NOT NULL) AS n_distinct_employer_keys
        FROM positions_clean
        GROUP BY 1
        """,
    )
    _create_temp_table(
        con,
        "top_country",
        """
        SELECT
            user_id,
            country_candidate AS top_country_candidate,
            country_score AS top_country_score,
            country_uncertain_ind
        FROM country_candidates
        WHERE country_rank = 1
        """,
    )
    _create_temp_table(
        con,
        "country_json",
        """
        SELECT
            user_id,
            CAST(
                array_to_json(
                    list(
                        json_object(
                            'country_candidate', country_candidate,
                            'country_score', ROUND(country_score, 6),
                            'nanat_score', ROUND(nanat_score, 6),
                            'institution_score', ROUND(institution_score, 6),
                            'nametrace_score', ROUND(nametrace_score, 6)
                        )
                        ORDER BY country_rank
                    )
                ) AS VARCHAR
            ) AS country_candidates_json
        FROM country_candidates
        GROUP BY 1
        """,
    )
    _create_temp_table(
        con,
        "users_core",
        """
        SELECT
            p.user_id,
            p.fullname,
            p.fullname_clean,
            p.profile_linkedin_url,
            p.user_location,
            p.user_country,
            p.user_country_std,
            p.f_prob,
            p.updated_dt,
            p.updated_dt_ts,
            es.est_yob,
            es.highest_ed_level,
            COALESCE(es.fields_json, '[]') AS fields_json,
            COALESCE(es.stem_ind_any, 0) AS stem_ind_any,
            COALESCE(es.n_education_records, 0) AS n_education_records,
            COALESCE(es.us_hs_exact, 0) AS us_hs_exact,
            COALESCE(es.us_educ, 0) AS us_educ,
            COALESCE(es.ade_ind, 0) AS ade_ind,
            es.ade_year,
            es.last_grad_year,
            COALESCE(ps.n_position_records, 0) AS n_position_records,
            COALESCE(ps.n_distinct_employer_keys, 0) AS n_distinct_employer_keys,
            gs.f_prob_nt,
            tc.top_country_candidate,
            tc.top_country_score,
            COALESCE(tc.country_uncertain_ind, 0) AS country_uncertain_ind,
            COALESCE(cj.country_candidates_json, '[]') AS country_candidates_json
        FROM profiles_src AS p
        LEFT JOIN educ_summary AS es
          ON es.user_id = p.user_id
        LEFT JOIN pos_summary AS ps
          ON ps.user_id = p.user_id
        LEFT JOIN gender_scores AS gs
          ON gs.user_id = p.user_id
        LEFT JOIN top_country AS tc
          ON tc.user_id = p.user_id
        LEFT JOIN country_json AS cj
          ON cj.user_id = p.user_id
        """,
    )


def _create_positions_agg(con: duckdb.DuckDBPyConnection, include_null_employer: bool) -> None:
    _create_temp_table(
        con,
        "positions_agg_base",
        """
        SELECT
            user_id,
            employer_key,
            MAX(rcid) AS rcid,
            COUNT(position_id) AS n_positions,
            MAX(company_raw) AS representative_company_raw,
            MIN(startdate) AS first_position_startdate,
            MAX(enddate) AS last_position_enddate
        FROM positions_clean
        GROUP BY 1, 2
        """,
    )
    if include_null_employer:
        _create_temp_table(
            con,
            "positions_agg",
            """
            SELECT * FROM positions_agg_base
            UNION ALL
            SELECT
                u.user_id,
                CAST(NULL AS VARCHAR) AS employer_key,
                CAST(NULL AS BIGINT) AS rcid,
                0 AS n_positions,
                CAST(NULL AS VARCHAR) AS representative_company_raw,
                CAST(NULL AS VARCHAR) AS first_position_startdate,
                CAST(NULL AS VARCHAR) AS last_position_enddate
            FROM users_core AS u
            LEFT JOIN (
                SELECT DISTINCT user_id
                FROM positions_agg_base
            ) AS p
              ON p.user_id = u.user_id
            WHERE p.user_id IS NULL
            """,
        )
    else:
        _create_temp_table(
            con,
            "positions_agg",
            "SELECT * FROM positions_agg_base WHERE employer_key IS NOT NULL",
        )


def _create_match_ready(con: duckdb.DuckDBPyConnection, stage_cfg: dict[str, Any]) -> None:
    include_null_unitid = coerce_bool(stage_cfg.get("match_ready_include_null_unitid"), True)
    unitid_clause = "" if include_null_unitid else "AND e.unitid IS NOT NULL"
    match_ready_degree_clean_expr = _match_ready_degree_clean_expr()
    _create_temp_table(
        con,
        "match_ready",
        f"""
        SELECT DISTINCT
            e.user_id,
            e.education_number,
            e.unitid,
            {match_ready_degree_clean_expr} AS degree_clean,
            c.country_candidate,
            c.country_score,
            c.nanat_score,
            c.institution_score,
            c.nametrace_score,
            c.nanat_subregion_score,
            c.nt_subregion_score,
            c.subregion_candidate,
            c.country_uncertain_ind,
            e.cip,
            e.cip_score,
            e.field_mapped_ind,
            p.employer_key,
            p.rcid,
            p.n_positions,
            p.representative_company_raw,
            p.first_position_startdate,
            p.last_position_enddate,
            e.university_raw,
            e.field_clean,
            e.ed_startdate,
            e.ed_enddate,
            e.school_match_score
        FROM educ_clean AS e
        JOIN country_candidates AS c
          ON c.user_id = e.user_id
        LEFT JOIN positions_agg AS p
          ON p.user_id = e.user_id
        WHERE e.match_eligible_ind = 1
          {unitid_clause}
        """,
    )


def _create_candidate_long_outputs(
    con: duckdb.DuckDBPyConnection,
    stage_cfg: dict[str, Any],
    *,
    in_memory_only: bool = False,
) -> dict[str, Any]:
    if not coerce_bool(stage_cfg.get("write_candidate_long_artifacts"), False):
        return {}
    top_k = max(0, int(stage_cfg.get("candidate_long_top_k", 5) or 0))
    _create_temp_table(
        con,
        "inst_candidate_long",
        f"""
        SELECT
            r.user_id,
            r.education_number,
            r.candidate_rank,
            r.unitid,
            r.candidate_score,
            r.text_match_rank,
            r.alias_jw_max,
            r.alias_token_overlap_max,
            r.alias_shared_token_count_max,
            r.selection_bucket,
            'deterministic_candidate' AS mapping_source,
            CASE
                WHEN e.unitid_mapping_source = 'deterministic_top1'
                 AND e.unitid = r.unitid THEN 1
                ELSE 0
            END AS selected_top1_ind
        FROM unitid_candidate_rows AS r
        JOIN educ_clean AS e
          ON e.user_id = r.user_id
         AND e.education_number = r.education_number
        WHERE r.candidate_rank <= {top_k}
        """,
    )
    _create_temp_table(
        con,
        "cip_candidate_long",
        f"""
        SELECT
            r.user_id,
            r.education_number,
            r.candidate_rank,
            r.cip,
            r.candidate_score,
            'deterministic_candidate' AS mapping_source,
            CASE
                WHEN e.cip_mapping_source = 'deterministic_top1'
                 AND r.candidate_rank = 1
                 AND e.cip = r.cip THEN 1
                ELSE 0
            END AS selected_top1_ind
        FROM cip_candidate_rows AS r
        JOIN educ_clean AS e
          ON e.user_id = r.user_id
         AND e.education_number = r.education_number
        WHERE r.candidate_rank <= {top_k}
        """,
    )
    outputs = {
        "rev_educ_inst_candidates_long_rows": _relation_count(con, "inst_candidate_long"),
        "rev_educ_cip_candidates_long_rows": _relation_count(con, "cip_candidate_long"),
    }
    if in_memory_only:
        outputs.update(
            {
                "rev_educ_inst_candidates_long_table": "inst_candidate_long",
                "rev_educ_cip_candidates_long_table": "cip_candidate_long",
            }
        )
        return outputs

    inst_out = ensure_parent_dir(stage_cfg["rev_educ_inst_candidates_long_parquet"])
    cip_out = ensure_parent_dir(stage_cfg["rev_educ_cip_candidates_long_parquet"])
    atomic_duckdb_copy_to_parquet(con, "SELECT * FROM inst_candidate_long", inst_out)
    atomic_duckdb_copy_to_parquet(con, "SELECT * FROM cip_candidate_long", cip_out)
    outputs.update(
        {
            "rev_educ_inst_candidates_long_parquet": str(inst_out),
            "rev_educ_cip_candidates_long_parquet": str(cip_out),
        }
    )
    return outputs


def _validate_stage_outputs(con: duckdb.DuckDBPyConnection) -> None:
    if _relation_count(con, "users_core") <= 0:
        raise ValueError(f"{STAGE_NAME} produced an empty rev_users_core artifact.")
    if _relation_count(con, "country_candidates") <= 0:
        raise ValueError(f"{STAGE_NAME} produced no country candidates.")
    match_ready_columns = set(_relation_columns(con, "match_ready"))
    missing_cols = sorted(MATCH_READY_REQUIRED_COLS - match_ready_columns)
    if missing_cols:
        raise ValueError(f"{STAGE_NAME} match-ready artifact is missing required columns: {missing_cols}")
    if _relation_count(con, "match_ready") <= 0:
        raise ValueError(f"{STAGE_NAME} produced an empty match-ready artifact.")


def _resolve_stage_paths(cfg: dict[str, Any], stage_cfg: dict[str, Any]) -> dict[str, str | None]:
    stage02_cfg = cfg.get("stages", {}).get("02_rev_import", {})
    stage03_cfg = cfg.get("stages", {}).get("03_rev_crosswalks", {})
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
        "name2nat": resolve_existing_path(stage_cfg.get("name2nat_parquet")),
        "nametrace_wide": resolve_existing_path(stage_cfg.get("nametrace_wide_parquet")),
        "nametrace_long": resolve_existing_path(stage_cfg.get("nametrace_long_parquet")),
        "school_crosswalk": resolve_existing_path(
            stage_cfg.get("rev_school_unitid_crosswalk_input_parquet"),
            stage03_cfg.get("rev_school_unitid_crosswalk_parquet"),
        ),
        "field_crosswalk": resolve_existing_path(
            stage_cfg.get("field_cip_crosswalk_input_parquet"),
            stage03_cfg.get("field_cip_crosswalk_parquet"),
        ),
        "deterministic_triple_map": resolve_existing_path(stage_cfg.get("deterministic_triple_map_input_parquet")),
        "ref_inst_link": resolve_existing_path(stage_cfg.get("ref_inst_link_input_parquet")),
        "openalex_institutions": resolve_existing_path(stage_cfg.get("openalex_institutions_input_jsonl")),
        "cip_reference": resolve_existing_path(stage_cfg.get("cip_reference_input_path")),
        "ipeds_name_crosswalk": resolve_existing_path(stage_cfg.get("ipeds_name_crosswalk_input_parquet")),
        "employer_lookup": resolve_existing_path(
            stage_cfg.get("employer_lookup_input_parquet"),
            stage03_cfg.get("employer_lookup_parquet"),
        ),
        "employer_key_map": resolve_existing_path(
            stage_cfg.get("employer_key_map_input_parquet"),
            stage03_cfg.get("employer_key_map_parquet"),
        ),
        "legacy_rev_indiv": resolve_existing_path(
            stage_cfg.get("legacy_rev_indiv_parquet"),
            paths_cfg.get("legacy_rev_indiv_parquet"),
        ),
        "legacy_rev_educ_long": resolve_existing_path(
            stage_cfg.get("legacy_rev_educ_long_parquet"),
            paths_cfg.get("legacy_rev_educ_long_parquet"),
        ),
        "legacy_rev_pos": resolve_existing_path(
            stage_cfg.get("legacy_rev_pos_parquet"),
            paths_cfg.get("legacy_rev_pos_parquet"),
        ),
    }


def assemble_clean_user_artifacts(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    testing: bool | None = None,
    in_memory_only: bool = False,
    return_session: bool = False,
) -> dict[str, Any] | Stage04Session:
    in_memory_only = bool(in_memory_only or return_session)
    stage_t0 = time.perf_counter()
    cfg = pipeline_cfg or load_config(config_path)
    stage_cfg = get_stage_config(cfg, STAGE_NAME)
    effective_testing = coerce_bool(cfg.get("testing", {}).get("enabled"), False) if testing is None else bool(testing)
    testing_max_users = int(stage_cfg.get("testing_max_users", 0)) if effective_testing else None
    shard_spec = resolve_user_shard_spec(stage_cfg)
    resolved_paths = _resolve_stage_paths(cfg, stage_cfg)

    print(
        f"[{STAGE_NAME}] assemble_clean_user_artifacts start "
        f"(testing={effective_testing}, testing_max_users={testing_max_users or 0}, "
        f"user_shard={shard_spec['user_shard_label'] if shard_spec else 'off'})"
    )
    con = get_duckdb_connection()
    register_country_support_views(con)

    load_t0 = time.perf_counter()
    raw_mode = resolved_paths["raw_users"] is not None and resolved_paths["raw_positions"] is not None
    if raw_mode:
        print(f"[{STAGE_NAME}] Loading raw users from {resolved_paths['raw_users']}")
        _create_raw_sources(
            con,
            raw_users_path=str(resolved_paths["raw_users"]),
            raw_positions_path=str(resolved_paths["raw_positions"]),
            deterministic_triple_map_path=resolved_paths["deterministic_triple_map"],
            testing_max_users=testing_max_users,
            shard_count=int(shard_spec["user_shard_count"]) if shard_spec else None,
            shard_id=int(shard_spec["user_shard_id"]) if shard_spec else None,
        )
        source_mode = "raw_stage02"
        country_seed = False
        print(f"[{STAGE_NAME}] Loading raw positions from {resolved_paths['raw_positions']}")
    else:
        allow_legacy = coerce_bool(cfg.get("build", {}).get("allow_legacy_fallbacks"), True)
        if not allow_legacy:
            raise FileNotFoundError(
                f"{STAGE_NAME} could not find stage-02 raw users/positions and legacy fallbacks are disabled."
            )
        if not all(
            [
                resolved_paths["legacy_rev_indiv"],
                resolved_paths["legacy_rev_educ_long"],
                resolved_paths["legacy_rev_pos"],
            ]
        ):
            raise FileNotFoundError(
                f"{STAGE_NAME} could not find stage-02 raw users/positions and the required legacy fallbacks "
                "(`legacy_rev_indiv_parquet`, `legacy_rev_educ_long_parquet`, `legacy_rev_pos_parquet`) were not all present."
            )
        print(
            f"[{STAGE_NAME}] Loading legacy fallback artifacts "
            f"(indiv={resolved_paths['legacy_rev_indiv']}, educ={resolved_paths['legacy_rev_educ_long']}, "
            f"pos={resolved_paths['legacy_rev_pos']})"
        )
        _create_legacy_sources(
            con,
            legacy_rev_indiv_path=str(resolved_paths["legacy_rev_indiv"]),
            legacy_rev_educ_path=str(resolved_paths["legacy_rev_educ_long"]),
            legacy_rev_pos_path=str(resolved_paths["legacy_rev_pos"]),
            testing_max_users=testing_max_users,
            shard_count=int(shard_spec["user_shard_count"]) if shard_spec else None,
            shard_id=int(shard_spec["user_shard_id"]) if shard_spec else None,
        )
        source_mode = "legacy_fallback"
        country_seed = True

    print(
        f"[{STAGE_NAME}] Loaded source frames in {_fmt_elapsed(time.perf_counter() - load_t0)} "
        f"(source_mode={source_mode}, profiles={_relation_count(con, 'profiles_src'):,}, "
        f"educ={_relation_count(con, 'educ_src'):,}, positions={_relation_count(con, 'positions_src'):,})"
    )

    crosswalk_t0 = time.perf_counter()
    print(f"[{STAGE_NAME}] Loading institution/employer reference inputs")
    _create_school_crosswalk_table(con, resolved_paths["school_crosswalk"])
    _create_field_crosswalk_table(con, resolved_paths["field_crosswalk"])
    _create_ref_inst_link_table(con, resolved_paths["ref_inst_link"])
    _create_openalex_reference_tables(con, resolved_paths["openalex_institutions"])
    _create_cip_title_lookup_table(con, resolved_paths["cip_reference"])
    _create_unitid_alias_table(
        con,
        ipeds_path=resolved_paths["ipeds_name_crosswalk"],
        openalex_path=None,
    )
    _create_ref_inst_alias_table(con)
    _create_employer_key_map_table(
        con,
        key_map_path=resolved_paths["employer_key_map"],
        employer_lookup_path=resolved_paths["employer_lookup"],
    )
    print(
        f"[{STAGE_NAME}] Loaded crosswalk inputs in {_fmt_elapsed(time.perf_counter() - crosswalk_t0)} "
        f"(school={_relation_count(con, 'school_crosswalk'):,}, field={_relation_count(con, 'field_crosswalk'):,}, "
        f"ref_inst_links={_relation_count(con, 'ref_inst_links'):,}, openalex={_relation_count(con, 'openalex_entities'):,}, "
        f"employer={_relation_count(con, 'employer_key_map'):,}, cip_titles={_relation_count(con, 'cip_title_lookup'):,}, "
        f"ipeds_unitids={int(con.execute('SELECT COUNT(DISTINCT unitid) FROM unitid_aliases').fetchone()[0] or 0):,})"
    )

    transform_t0 = time.perf_counter()
    print(f"[{STAGE_NAME}] Applying education crosswalks and employer mapping")
    _create_candidate_rerank_tables(con, stage_cfg)
    _create_educ_clean_all(con)
    _create_positions_clean(con)
    print(
        f"[{STAGE_NAME}] Applied mappings in {_fmt_elapsed(time.perf_counter() - transform_t0)} "
        f"(educ_clean_all={_relation_count(con, 'educ_clean_all'):,}, positions_clean={_relation_count(con, 'positions_clean'):,})"
    )

    country_t0 = time.perf_counter()
    if country_seed:
        print(f"[{STAGE_NAME}] Using precomputed legacy country candidates")
    else:
        print(f"[{STAGE_NAME}] Building country candidates from name/institution signals")
        _create_name2nat_table(
            con,
            name2nat_path=resolved_paths["name2nat"],
            stage_cfg=stage_cfg,
        )
        _create_nametrace_tables(
            con,
            nametrace_wide_path=resolved_paths["nametrace_wide"],
            nametrace_long_path=resolved_paths["nametrace_long"],
        )
        _create_institution_country_scores(con, stage_cfg)
    if country_seed:
        _create_empty_table(con, "gender_scores", [("user_id", "BIGINT"), ("f_prob_nt", "DOUBLE")])
        _create_empty_table(
            con,
            "user_region_scores",
            [("user_id", "BIGINT"), ("subregion_candidate", "VARCHAR"), ("nt_subregion_score", "DOUBLE")],
        )
    _create_country_candidates(con, stage_cfg=stage_cfg, use_seed=country_seed)
    print(
        f"[{STAGE_NAME}] Country candidate step complete in {_fmt_elapsed(time.perf_counter() - country_t0)} "
        f"(country_candidates={_relation_count(con, 'country_candidates'):,}, gender_rows={_relation_count(con, 'gender_scores'):,})"
    )

    build_t0 = time.perf_counter()
    print(f"[{STAGE_NAME}] Building users_core, positions aggregation, and match_ready")
    _create_pruned_educ_clean(con)
    _create_users_core(con)
    _create_positions_agg(
        con,
        include_null_employer=coerce_bool(stage_cfg.get("match_ready_include_null_employer"), True),
    )
    _create_match_ready(con, stage_cfg)
    print(
        f"[{STAGE_NAME}] Built final in-memory artifacts in {_fmt_elapsed(time.perf_counter() - build_t0)} "
        f"(users_core={_relation_count(con, 'users_core'):,}, positions_agg={_relation_count(con, 'positions_agg'):,}, "
        f"match_ready={_relation_count(con, 'match_ready'):,})"
    )

    validate_t0 = time.perf_counter()
    print(f"[{STAGE_NAME}] Validating stage outputs")
    _validate_stage_outputs(con)
    print(f"[{STAGE_NAME}] Validation complete in {_fmt_elapsed(time.perf_counter() - validate_t0)}")

    candidate_t0 = time.perf_counter()
    candidate_long_outputs = _create_candidate_long_outputs(
        con,
        stage_cfg,
        in_memory_only=in_memory_only,
    )
    if candidate_long_outputs:
        print(
            f"[{STAGE_NAME}] Candidate long artifacts {'kept in memory' if in_memory_only else 'written'} "
            f"in {_fmt_elapsed(time.perf_counter() - candidate_t0)} "
            f"({candidate_long_outputs})"
        )
    else:
        print(f"[{STAGE_NAME}] Candidate long artifacts skipped in {_fmt_elapsed(time.perf_counter() - candidate_t0)}")

    educ_drop_cols = {
        "deterministic_cip_candidates",
        "deterministic_ref_inst_candidates",
    }
    educ_write_cols = [column for column in _relation_columns(con, "educ_clean") if column not in educ_drop_cols]

    outputs = {
        "rev_users_core_rows": _relation_count(con, "users_core"),
        "rev_educ_clean_long_rows": _relation_count(con, "educ_clean"),
        "rev_pos_clean_long_rows": _relation_count(con, "positions_clean"),
        "rev_match_ready_rows": _relation_count(con, "match_ready"),
        "country_candidate_rows": _relation_count(con, "country_candidates"),
        "source_mode": source_mode,
    }
    if shard_spec:
        outputs.update(
            {
                "user_shard_count": int(shard_spec["user_shard_count"]),
                "user_shard_id": int(shard_spec["user_shard_id"]),
                "user_shard_label": str(shard_spec["user_shard_label"]),
            }
        )
    outputs.update(candidate_long_outputs)
    tables = {
        name: name
        for name in (
            "profiles_src",
            "educ_src",
            "positions_src",
            "school_crosswalk",
            "field_crosswalk",
            "ref_inst_links",
            "openalex_entities",
            "openalex_aliases",
            "ref_inst_aliases",
            "cip_title_lookup",
            "unitid_aliases",
            "employer_key_map",
            "ref_inst_candidate_rows",
            "unitid_candidate_rows",
            "cip_candidate_rows",
            "educ_clean_all",
            "educ_clean",
            "positions_clean",
            "name2nat_scores",
            "gender_scores",
            "user_region_scores",
            "institution_country_scores",
            "country_candidates",
            "users_core",
            "positions_agg",
            "match_ready",
            "inst_candidate_long",
            "cip_candidate_long",
        )
        if _relation_exists(con, name)
    }
    if in_memory_only:
        outputs["in_memory_only"] = True
        outputs["duckdb_tables"] = tables
        print(f"[{STAGE_NAME}] Skipping parquet writes because in_memory_only=True")
    else:
        users_out = ensure_parent_dir(stage_cfg["rev_users_core_parquet"])
        educ_out = ensure_parent_dir(stage_cfg["rev_educ_clean_long_parquet"])
        pos_out = ensure_parent_dir(stage_cfg["rev_pos_clean_long_parquet"])
        match_out = ensure_parent_dir(stage_cfg["rev_match_ready_parquet"])
        write_t0 = time.perf_counter()
        print(f"[{STAGE_NAME}] Writing parquet outputs")
        atomic_duckdb_copy_to_parquet(con, "SELECT * FROM users_core", users_out)
        atomic_duckdb_copy_to_parquet(
            con,
            "SELECT " + ", ".join(_sql_identifier(column) for column in educ_write_cols) + " FROM educ_clean",
            educ_out,
        )
        atomic_duckdb_copy_to_parquet(con, "SELECT * FROM positions_clean", pos_out)
        atomic_duckdb_copy_to_parquet(con, "SELECT * FROM match_ready", match_out)
        print(
            f"[{STAGE_NAME}] Wrote parquet outputs in {_fmt_elapsed(time.perf_counter() - write_t0)} "
            f"(users={users_out}, educ={educ_out}, pos={pos_out}, match_ready={match_out})"
        )
        outputs.update(
            {
                "rev_users_core_parquet": str(users_out),
                "rev_educ_clean_long_parquet": str(educ_out),
                "rev_pos_clean_long_parquet": str(pos_out),
                "rev_match_ready_parquet": str(match_out),
            }
        )
    print(
        f"[{STAGE_NAME}] assemble_clean_user_artifacts complete in {_fmt_elapsed(time.perf_counter() - stage_t0)}"
    )
    if return_session:
        outputs["in_memory_only"] = True
        outputs["duckdb_tables"] = tables
        return Stage04Session(connection=con, outputs=outputs, tables=tables)
    con.close()
    return outputs
