"""WRDS query builders for stage 02_rev_import."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


def _resolve_pgpass_path(pgpass_path: str | Path | None = None) -> Path:
    if pgpass_path is not None:
        return Path(pgpass_path).expanduser()
    env_pgpass = os.environ.get("PGPASSFILE")
    if env_pgpass:
        return Path(env_pgpass).expanduser()
    return Path.home() / ".pgpass"


def _parse_pgpass_line(line: str) -> tuple[str, str, str, str, str] | None:
    parts: list[str] = []
    buffer: list[str] = []
    escaped = False

    for char in line.rstrip("\n"):
        if escaped:
            buffer.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == ":" and len(parts) < 4:
            parts.append("".join(buffer))
            buffer = []
            continue
        buffer.append(char)

    parts.append("".join(buffer))
    if len(parts) != 5:
        return None
    return tuple(parts)  # type: ignore[return-value]


def infer_wrds_username_from_pgpass(pgpass_path: str | Path | None = None) -> str | None:
    resolved_path = _resolve_pgpass_path(pgpass_path)
    if not resolved_path.exists():
        return None

    preferred_usernames: list[str] = []
    fallback_usernames: list[str] = []
    for raw_line in resolved_path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parsed = _parse_pgpass_line(stripped)
        if parsed is None:
            continue
        host, _port, database, username, _password = parsed
        username = username.strip()
        if not username or username == "*":
            continue

        host_lower = host.lower()
        database_lower = database.lower()
        if "wrds" in host_lower or "wharton" in host_lower or database_lower == "wrds":
            preferred_usernames.append(username)
        else:
            fallback_usernames.append(username)

    if preferred_usernames:
        return preferred_usernames[0]
    if fallback_usernames:
        return fallback_usernames[0]
    return None


def get_wrds_connection(
    wrds_username: str | None = None,
    pgpass_path: str | Path | None = None,
):
    import wrds

    kwargs = {}
    resolved_username = wrds_username or infer_wrds_username_from_pgpass(pgpass_path)
    if resolved_username:
        kwargs["wrds_username"] = resolved_username
    return wrds.Connection(**kwargs)


def format_user_id_sql_list(user_ids: Iterable[int]) -> str:
    ids = [str(int(user_id)) for user_id in user_ids]
    if not ids:
        raise ValueError("Cannot build WRDS query with an empty user_id list.")
    return ",".join(ids)


def build_user_id_bounds_query(
    source_relation: str = "revelio.individual_user_education",
) -> str:
    return f"""
        SELECT
            MIN(user_id) AS user_id_min,
            MAX(user_id) AS user_id_max
        FROM {source_relation}
        WHERE user_id IS NOT NULL
    """


def fetch_user_id_bounds(
    db,
    source_relation: str = "revelio.individual_user_education",
) -> dict[str, int | str]:
    out = raw_sql_dataframe(db, build_user_id_bounds_query(source_relation=source_relation))
    if out.empty:
        raise ValueError(f"WRDS bounds query returned no rows for {source_relation}.")

    user_id_min = out.loc[0, "user_id_min"]
    user_id_max = out.loc[0, "user_id_max"]
    if pd.isna(user_id_min) or pd.isna(user_id_max):
        raise ValueError(f"WRDS bounds query returned null user_id bounds for {source_relation}.")

    return {
        "user_id_bounds_source_relation": source_relation,
        "resolved_user_id_min": int(user_id_min),
        "resolved_user_id_max": int(user_id_max),
    }


def build_education_scan_query(
    *,
    user_id_lower_bound: int,
    user_id_upper_bound: int,
    start_year_threshold: int = 2000,
    end_year_threshold: int = 2004,
    row_limit: int | None = None,
) -> str:
    limit_sql = f"\nLIMIT {int(row_limit)}" if row_limit is not None else ""
    start_date_floor = f"{int(start_year_threshold) + 1:04d}-01-01"
    end_date_floor = f"{int(end_year_threshold) + 1:04d}-01-01"
    return f"""
        WITH base AS (
            SELECT
                a.user_id,
                a.university_name,
                a.rsid,
                a.education_number,
                a.startdate AS ed_startdate,
                a.enddate AS ed_enddate,
                a.degree,
                a.field,
                a.university_country,
                a.university_location
            FROM revelio.individual_user_education AS a
            WHERE a.user_id >= {int(user_id_lower_bound)}
              AND a.user_id <= {int(user_id_upper_bound)}
        ),
        filtered AS (
            SELECT *
            FROM base
            WHERE (
                university_country IS NULL
                OR university_country = 'United States'
            )
              AND (
                ed_startdate >= DATE '{start_date_floor}'
                OR ed_enddate >= DATE '{end_date_floor}'
                OR (ed_startdate IS NULL AND ed_enddate IS NULL)
              )
              AND (
                degree IS NULL
                OR degree NOT IN (
                    'High School', 'Associate'
                )
              )
        )
        SELECT
            f.user_id,
            f.education_number,
            f.ed_startdate,
            f.ed_enddate,
            f.degree,
            f.field,
            f.university_country,
            r.university_raw,
            r.degree_raw,
            r.field_raw
        FROM filtered AS f
        LEFT JOIN revelio.individual_user_education_raw AS r
            ON f.user_id = r.user_id
           AND f.education_number = r.education_number
        {limit_sql}
    """


def build_wrds_users_query(user_ids: Iterable[int]) -> str:
    userid_subset = format_user_id_sql_list(user_ids)
    return f"""
        SELECT
            CASE WHEN a.user_id IS NOT NULL THEN a.user_id ELSE b.user_id END AS user_id,
            fullname,
            profile_linkedin_url,
            user_location,
            user_country,
            f_prob,
            updated_dt,
            university_name,
            rsid,
            education_number,
            ed_startdate,
            ed_enddate,
            degree,
            field,
            university_country,
            university_location,
            university_raw,
            degree_raw,
            field_raw,
            description
        FROM (
            (
                SELECT
                    user_id,
                    fullname,
                    profile_linkedin_url,
                    user_location,
                    user_country,
                    f_prob,
                    updated_dt
                FROM revelio.individual_user
                WHERE user_id IN ({userid_subset})
            ) AS a
            FULL JOIN (
                SELECT
                    b_educ.user_id,
                    b_educ.education_number,
                    university_name,
                    rsid,
                    ed_startdate,
                    ed_enddate,
                    degree,
                    field,
                    university_country,
                    university_location,
                    university_raw,
                    degree_raw,
                    field_raw,
                    description
                FROM (
                    (
                        SELECT
                            user_id,
                            university_name,
                            rsid,
                            education_number,
                            startdate AS ed_startdate,
                            enddate AS ed_enddate,
                            degree,
                            field,
                            university_country,
                            university_location
                        FROM revelio.individual_user_education
                        WHERE user_id IN ({userid_subset})
                    ) AS b_educ
                    LEFT JOIN (
                        SELECT
                            user_id,
                            university_raw,
                            education_number,
                            degree_raw,
                            field_raw,
                            description
                        FROM revelio.individual_user_education_raw
                        WHERE user_id IN ({userid_subset})
                    ) AS b_educ_raw
                    ON b_educ.user_id = b_educ_raw.user_id
                   AND b_educ.education_number = b_educ_raw.education_number
                )
            ) AS b
            ON a.user_id = b.user_id
        )
    """


def build_wrds_positions_query(user_ids: Iterable[int]) -> str:
    userid_subset = format_user_id_sql_list(user_ids)
    return f"""
        SELECT
            a.user_id,
            a.position_id,
            a.position_number,
            rcid,
            country,
            startdate,
            enddate,
            role_k17000_v3,
            salary,
            total_compensation,
            company_raw,
            title_raw
        FROM revelio.individual_positions AS a
        LEFT JOIN revelio.individual_positions_raw AS b
            ON a.position_id = b.position_id
        WHERE a.user_id IN ({userid_subset})
    """


def raw_sql_dataframe(db, query: str) -> pd.DataFrame:
    return db.raw_sql(query)
