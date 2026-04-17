"""Consolidate shard-level education matches for stage 02_rev_import."""

from __future__ import annotations

from pathlib import Path

import duckdb as ddb

from common import escape_sql_literal


def _count_match_outputs(
    *,
    matched_education_parquet: str | Path,
    matched_user_list_parquet: str | Path,
) -> dict[str, int]:
    con = ddb.connect()
    educ_path = escape_sql_literal(matched_education_parquet)
    user_path = escape_sql_literal(matched_user_list_parquet)
    educ_counts = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT user_id) AS n_users,
            COUNT(
                DISTINCT CAST(user_id AS VARCHAR) || '|' || COALESCE(CAST(education_number AS VARCHAR), '')
            ) AS n_education_numbers
        FROM read_parquet('{educ_path}')
        """
    ).fetchone()
    user_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{user_path}')
        """
    ).fetchone()[0]
    return {
        "matched_education_rows": int(educ_counts[0]),
        "matched_user_count": int(educ_counts[1]),
        "matched_education_numbers": int(educ_counts[2]),
        "matched_user_rows": int(user_count),
    }


def compile_matched_user_artifacts(
    *,
    matched_education_shard_dir: str | Path,
    matched_user_shard_dir: str | Path,
    matched_education_parquet: str | Path,
    matched_user_list_parquet: str | Path,
    matched_user_list_csv: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, int | str]:
    educ_dir = Path(matched_education_shard_dir)
    user_dir = Path(matched_user_shard_dir)
    educ_files = sorted(educ_dir.glob("*.parquet"))
    user_files = sorted(user_dir.glob("*.parquet"))
    if not educ_files:
        raise FileNotFoundError(f"No shard-level matched education parquet files found in {educ_dir}")
    if not user_files:
        raise FileNotFoundError(f"No shard-level matched user parquet files found in {user_dir}")

    educ_out = Path(matched_education_parquet)
    user_out = Path(matched_user_list_parquet)
    user_csv_out = Path(matched_user_list_csv) if matched_user_list_csv else None
    educ_out.parent.mkdir(parents=True, exist_ok=True)
    user_out.parent.mkdir(parents=True, exist_ok=True)
    if user_csv_out is not None:
        user_csv_out.parent.mkdir(parents=True, exist_ok=True)

    if educ_out.exists() and user_out.exists() and (user_csv_out is None or user_csv_out.exists()) and not overwrite:
        counts = _count_match_outputs(
            matched_education_parquet=educ_out,
            matched_user_list_parquet=user_out,
        )
        return {
            "matched_education_shard_dir": str(educ_dir),
            "matched_user_shard_dir": str(user_dir),
            "matched_education_parquet": str(educ_out),
            "matched_user_list_parquet": str(user_out),
            "matched_user_list_csv": str(user_csv_out) if user_csv_out else "",
            "matched_education_shard_files": len(educ_files),
            "matched_user_shard_files": len(user_files),
            **counts,
        }

    for path in [educ_out, user_out, user_csv_out]:
        if path is not None and path.exists() and overwrite:
            path.unlink()

    con = ddb.connect()
    educ_glob = escape_sql_literal(educ_dir / "*.parquet")
    user_glob = escape_sql_literal(user_dir / "*.parquet")

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet('{educ_glob}')
            ORDER BY user_id, education_number, shard_id
        )
        TO '{escape_sql_literal(educ_out)}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT
                TRY_CAST(user_id AS BIGINT) AS user_id,
                SUM(COALESCE(n_matched_education_rows, 0)) AS n_matched_education_rows,
                MAX(COALESCE(any_university_match_ind, 0)) AS any_university_match_ind,
                MAX(COALESCE(any_degree_match_ind, 0)) AS any_degree_match_ind,
                MAX(COALESCE(any_field_match_ind, 0)) AS any_field_match_ind,
                COUNT(*) AS n_shards_with_matches
            FROM read_parquet('{user_glob}')
            WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        )
        TO '{escape_sql_literal(user_out)}' (FORMAT PARQUET)
        """
    )
    if user_csv_out is not None:
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{escape_sql_literal(user_out)}')
                ORDER BY user_id
            )
            TO '{escape_sql_literal(user_csv_out)}' (HEADER, DELIMITER ',')
            """
        )

    counts = _count_match_outputs(
        matched_education_parquet=educ_out,
        matched_user_list_parquet=user_out,
    )
    return {
        "matched_education_shard_dir": str(educ_dir),
        "matched_user_shard_dir": str(user_dir),
        "matched_education_parquet": str(educ_out),
        "matched_user_list_parquet": str(user_out),
        "matched_user_list_csv": str(user_csv_out) if user_csv_out else "",
        "matched_education_shard_files": len(educ_files),
        "matched_user_shard_files": len(user_files),
        **counts,
    }
