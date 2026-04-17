"""Local education matching helpers for stage 02_rev_import."""

from __future__ import annotations

from pathlib import Path

import duckdb as ddb
import pandas as pd

from common import (
    escape_sql_literal,
    sql_degree_clean_expr,
    sql_extract_year_expr,
    sql_inst_clean_expr,
    sql_us_or_null_country_expr,
)


def materialize_local_match_tables(
    *,
    con: ddb.DuckDBPyConnection,
    source_relation: str,
    regex_pattern: str,
    source_label: str,
    matched_educ_table: str = "matched_educ_stage2",
    matched_users_table: str = "matched_user_ids_stage2",
) -> None:
    escaped_pattern = regex_pattern.replace("'", "''") if regex_pattern else r"(?!)"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE wrds_users_stage2 AS
        WITH base AS (
            SELECT
                TRY_CAST(user_id AS BIGINT) AS user_id,
                TRY_CAST(education_number AS BIGINT) AS education_number,
                ed_startdate,
                ed_enddate,
                degree,
                field,
                university_country,
                university_raw,
                degree_raw,
                field_raw,
                {sql_inst_clean_expr('university_raw')} AS university_raw_clean,
                {sql_inst_clean_expr('degree_raw')} AS degree_raw_clean,
                {sql_inst_clean_expr('field_raw')} AS field_raw_clean,
                {sql_degree_clean_expr()} AS degree_clean_stage2,
                {sql_extract_year_expr('ed_startdate')} AS ed_start_year,
                {sql_extract_year_expr('ed_enddate')} AS ed_end_year
            FROM {source_relation}
            WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL
        ),
        filtered AS (
            SELECT
                *,
                CASE WHEN degree_clean_stage2 NOT IN ('High School', 'Associate', 'Non-Degree') THEN 1 ELSE 0 END AS degree_keep_ind,
                CASE
                    WHEN ed_start_year > 2000 OR ed_end_year > 2004 OR (ed_start_year IS NULL AND ed_end_year IS NULL)
                    THEN 1 ELSE 0
                END AS date_keep_ind
            FROM base
        )
        SELECT
            *,
            CASE WHEN regexp_matches(COALESCE(university_raw_clean, ''), '{escaped_pattern}') THEN 1 ELSE 0 END AS university_match_ind,
            CASE WHEN regexp_matches(COALESCE(degree_raw_clean, ''), '{escaped_pattern}') THEN 1 ELSE 0 END AS degree_match_ind,
            CASE WHEN regexp_matches(COALESCE(field_raw_clean, ''), '{escaped_pattern}') THEN 1 ELSE 0 END AS field_match_ind
        FROM filtered
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {matched_educ_table} AS
        SELECT
            *,
            CASE
                WHEN university_match_ind = 1 THEN 'university_raw'
                WHEN degree_match_ind = 1 THEN 'degree_raw'
                WHEN field_match_ind = 1 THEN 'field_raw'
                ELSE NULL
            END AS regex_match_source,
            '{source_label.replace("'", "''")}' AS provisional_match_source
        FROM wrds_users_stage2
        WHERE degree_keep_ind = 1
          AND date_keep_ind = 1
          AND (university_match_ind = 1 OR degree_match_ind = 1 OR field_match_ind = 1)
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {matched_users_table} AS
        SELECT
            user_id,
            COUNT(*) AS n_matched_education_rows,
            MAX(university_match_ind) AS any_university_match_ind,
            MAX(degree_match_ind) AS any_degree_match_ind,
            MAX(field_match_ind) AS any_field_match_ind
        FROM {matched_educ_table}
        GROUP BY user_id
        ORDER BY user_id
        """
    )


def match_wrds_users_dataframe(
    *,
    df: pd.DataFrame,
    regex_pattern: str,
    source_label: str,
    filtered_out_sample_n: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | pd.DataFrame] | None]:
    con = ddb.connect()
    con.register("_wrds_users_df", df)
    materialize_local_match_tables(
        con=con,
        source_relation="_wrds_users_df",
        regex_pattern=regex_pattern,
        source_label=source_label,
    )
    matched_educ = con.execute("SELECT * FROM matched_educ_stage2").df()
    matched_users = con.execute("SELECT * FROM matched_user_ids_stage2").df()
    debug_stats: dict[str, int | pd.DataFrame] | None = None
    if filtered_out_sample_n > 0:
        filtered_out_count = con.execute(
            """
            SELECT COUNT(*)
            FROM wrds_users_stage2
            WHERE NOT (
                degree_keep_ind = 1
                AND date_keep_ind = 1
                AND (university_match_ind = 1 OR degree_match_ind = 1 OR field_match_ind = 1)
            )
            """
        ).fetchone()[0]
        filtered_out_sample = con.execute(
            f"""
            SELECT
                user_id,
                education_number,
                ed_startdate,
                ed_enddate,
                degree,
                field,
                university_country,
                university_raw,
                degree_raw,
                field_raw,
                degree_clean_stage2,
                degree_keep_ind,
                date_keep_ind,
                university_match_ind,
                degree_match_ind,
                field_match_ind,
                CASE
                    WHEN degree_keep_ind != 1 THEN 'degree_filter'
                    WHEN date_keep_ind != 1 THEN 'date_filter'
                    WHEN university_match_ind != 1 AND degree_match_ind != 1 AND field_match_ind != 1 THEN 'regex_no_match'
                    ELSE 'other'
                END AS filtered_out_reason
            FROM wrds_users_stage2
            WHERE NOT (
                degree_keep_ind = 1
                AND date_keep_ind = 1
                AND (university_match_ind = 1 OR degree_match_ind = 1 OR field_match_ind = 1)
            )
            ORDER BY user_id, education_number
            LIMIT {int(filtered_out_sample_n)}
            """
        ).df()
        debug_stats = {
            "filtered_out_count": int(filtered_out_count),
            "filtered_out_sample": filtered_out_sample,
        }
    return matched_educ, matched_users, debug_stats


def build_provisional_matched_user_artifacts(
    *,
    wrds_users_source_path: str | Path,
    compiled_regex_txt: str | Path,
    matched_education_parquet: str | Path,
    matched_user_list_parquet: str | Path,
    matched_user_list_csv: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, int | str]:
    wrds_users_path = Path(wrds_users_source_path)
    regex_txt_path = Path(compiled_regex_txt)
    matched_educ_path = Path(matched_education_parquet)
    matched_user_path = Path(matched_user_list_parquet)
    matched_user_csv_path = Path(matched_user_list_csv) if matched_user_list_csv else None

    if not wrds_users_path.exists():
        raise FileNotFoundError(f"Legacy WRDS users parquet not found: {wrds_users_path}")
    if not regex_txt_path.exists():
        raise FileNotFoundError(f"Compiled FOIA regex artifact not found: {regex_txt_path}")

    matched_educ_path.parent.mkdir(parents=True, exist_ok=True)
    matched_user_path.parent.mkdir(parents=True, exist_ok=True)
    if matched_user_csv_path is not None:
        matched_user_csv_path.parent.mkdir(parents=True, exist_ok=True)

    regex_pattern = regex_txt_path.read_text().strip() or r"(?!)"
    con = ddb.connect()
    escaped_wrds_users = escape_sql_literal(wrds_users_path)

    materialize_local_match_tables(
        con=con,
        source_relation=f"read_parquet('{escaped_wrds_users}')",
        regex_pattern=regex_pattern,
        source_label="legacy_wrds_users_fallback",
    )

    for path in [matched_educ_path, matched_user_path, matched_user_csv_path]:
        if path is not None and path.exists() and overwrite:
            path.unlink()

    if overwrite or not matched_educ_path.exists():
        con.execute(
            f"""
            COPY matched_educ_stage2
            TO '{escape_sql_literal(matched_educ_path)}' (FORMAT PARQUET)
            """
        )
    if overwrite or not matched_user_path.exists():
        con.execute(
            f"""
            COPY matched_user_ids_stage2
            TO '{escape_sql_literal(matched_user_path)}' (FORMAT PARQUET)
            """
        )
    if matched_user_csv_path is not None and (overwrite or not matched_user_csv_path.exists()):
        con.execute(
            f"""
            COPY matched_user_ids_stage2
            TO '{escape_sql_literal(matched_user_csv_path)}' (HEADER, DELIMITER ',')
            """
        )

    counts = con.execute(
        """
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT user_id) AS n_users,
            COUNT(
                DISTINCT CAST(user_id AS VARCHAR) || '|' || COALESCE(CAST(education_number AS VARCHAR), '')
            ) AS n_education_numbers
        FROM matched_educ_stage2
        """
    ).fetchone()

    return {
        "wrds_users_source_path": str(wrds_users_path),
        "matched_education_parquet": str(matched_educ_path),
        "matched_user_list_parquet": str(matched_user_path),
        "matched_user_list_csv": str(matched_user_csv_path) if matched_user_csv_path else "",
        "matched_education_rows": int(counts[0]),
        "matched_user_count": int(counts[1]),
        "matched_education_numbers": int(counts[2]),
    }
