"""WRDS user and position import helpers for stage 02_rev_import."""

from __future__ import annotations

from builtins import print as _print
from functools import partial
from pathlib import Path
from typing import Iterable

import duckdb as ddb
import pandas as pd

from common import chunk_values, escape_sql_literal
from wrds_query import (
    build_wrds_positions_query,
    build_wrds_users_query,
    get_wrds_connection,
    raw_sql_dataframe,
)

print = partial(_print, flush=True)

WRDS_USERS_COLUMNS = [
    "user_id",
    "fullname",
    "profile_linkedin_url",
    "user_location",
    "user_country",
    "f_prob",
    "updated_dt",
    "university_name",
    "rsid",
    "education_number",
    "ed_startdate",
    "ed_enddate",
    "degree",
    "field",
    "university_country",
    "university_location",
    "university_raw",
    "degree_raw",
    "field_raw",
    "description",
]

WRDS_POSITIONS_COLUMNS = [
    "user_id",
    "position_id",
    "position_number",
    "rcid",
    "country",
    "startdate",
    "enddate",
    "role_k17000_v3",
    "salary",
    "total_compensation",
    "company_raw",
    "title_raw",
]


def _list_chunk_files(chunk_dir: str | Path) -> list[Path]:
    root = Path(chunk_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def has_wrds_import_chunks(chunk_dir: str | Path) -> bool:
    return bool(_list_chunk_files(chunk_dir))


def _clear_chunk_dir(chunk_dir: Path) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for path in _list_chunk_files(chunk_dir):
        path.unlink()
    nested_dirs = sorted(
        (path for path in chunk_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in nested_dirs:
        if path == chunk_dir:
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def _read_distinct_user_ids(
    *,
    matched_user_list_parquet: str | Path,
    max_users: int | None = None,
) -> list[int]:
    con = ddb.connect()
    user_ids = con.execute(
        f"""
        SELECT DISTINCT TRY_CAST(user_id AS BIGINT) AS user_id
        FROM read_parquet('{escape_sql_literal(matched_user_list_parquet)}')
        WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL
        ORDER BY user_id
        """
    ).df()["user_id"].astype(int).tolist()
    if max_users is not None:
        user_ids = user_ids[: int(max_users)]
    return user_ids


def _write_empty_parquet(path: str | Path, columns: list[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_parquet(out_path, index=False)


def _count_rows_from_files(paths: Iterable[Path]) -> int:
    file_list = [str(path) for path in paths]
    if not file_list:
        return 0
    con = ddb.connect()
    return int(con.execute("SELECT COUNT(*) FROM read_parquet(?)", [file_list]).fetchone()[0])


def _merge_chunk_dir_to_output(
    *,
    chunk_dir: str | Path,
    output_parquet: str | Path,
    empty_columns: list[str],
) -> None:
    chunk_files = _list_chunk_files(chunk_dir)
    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not chunk_files:
        _write_empty_parquet(out_path, empty_columns)
        return

    con = ddb.connect()
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet(?)
        )
        TO '{escape_sql_literal(out_path)}' (FORMAT PARQUET)
        """,
        [[str(path) for path in chunk_files]],
    )


def _count_parquet_rows(path: str | Path) -> int:
    con = ddb.connect()
    return int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{escape_sql_literal(path)}')"
        ).fetchone()[0]
    )


def write_wrds_import_chunks_for_user_ids(
    *,
    user_ids: Iterable[int],
    wrds_users_chunk_dir: str | Path,
    wrds_positions_chunk_dir: str | Path,
    wrds_username: str | None = None,
    chunk_size: int = 10000,
    chunk_subdir: str | Path | None = None,
    overwrite: bool = False,
    db=None,
) -> dict[str, int | str]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    users_chunk_root = Path(wrds_users_chunk_dir)
    positions_chunk_root = Path(wrds_positions_chunk_dir)
    users_chunk_dir = users_chunk_root / Path(chunk_subdir) if chunk_subdir else users_chunk_root
    positions_chunk_dir = positions_chunk_root / Path(chunk_subdir) if chunk_subdir else positions_chunk_root

    if overwrite:
        _clear_chunk_dir(users_chunk_dir)
        _clear_chunk_dir(positions_chunk_dir)
    else:
        users_chunk_dir.mkdir(parents=True, exist_ok=True)
        positions_chunk_dir.mkdir(parents=True, exist_ok=True)

    distinct_user_ids = sorted({int(user_id) for user_id in user_ids if user_id is not None})
    if not distinct_user_ids:
        users_empty = users_chunk_dir / "wrds_users_chunk_00000.parquet"
        positions_empty = positions_chunk_dir / "wrds_positions_chunk_00000.parquet"
        if overwrite or not users_empty.exists():
            _write_empty_parquet(users_empty, WRDS_USERS_COLUMNS)
        if overwrite or not positions_empty.exists():
            _write_empty_parquet(positions_empty, WRDS_POSITIONS_COLUMNS)
        return {
            "wrds_users_chunk_dir": str(users_chunk_dir),
            "wrds_positions_chunk_dir": str(positions_chunk_dir),
            "final_import_user_count": 0,
            "wrds_user_rows": 0,
            "wrds_position_rows": 0,
            "wrds_query_chunks": 0,
        }

    user_chunks = chunk_values(distinct_user_ids, chunk_size=chunk_size)
    own_db = db is None
    if own_db:
        db = get_wrds_connection(wrds_username=wrds_username)

    label = str(chunk_subdir) if chunk_subdir is not None else "all_users"
    try:
        for idx, user_chunk in enumerate(user_chunks):
            users_chunk_path = users_chunk_dir / f"wrds_users_chunk_{idx:05d}.parquet"
            positions_chunk_path = positions_chunk_dir / f"wrds_positions_chunk_{idx:05d}.parquet"

            if not users_chunk_path.exists() or overwrite:
                users_df = raw_sql_dataframe(db, build_wrds_users_query(user_chunk))
                users_df.to_parquet(users_chunk_path, index=False)
            if not positions_chunk_path.exists() or overwrite:
                positions_df = raw_sql_dataframe(db, build_wrds_positions_query(user_chunk))
                positions_df.to_parquet(positions_chunk_path, index=False)

            print(
                f"[02_rev_import] import chunk {idx + 1:05d}/{len(user_chunks):05d} "
                f"for {label}: user_ids={len(user_chunk):,}"
            )
    finally:
        if own_db and db is not None:
            try:
                db.close()
            except Exception:
                pass

    user_chunk_files = sorted(users_chunk_dir.glob("*.parquet"))
    position_chunk_files = sorted(positions_chunk_dir.glob("*.parquet"))
    return {
        "wrds_users_chunk_dir": str(users_chunk_dir),
        "wrds_positions_chunk_dir": str(positions_chunk_dir),
        "final_import_user_count": len(distinct_user_ids),
        "wrds_user_rows": _count_rows_from_files(user_chunk_files),
        "wrds_position_rows": _count_rows_from_files(position_chunk_files),
        "wrds_query_chunks": len(user_chunks),
    }


def consolidate_wrds_user_and_position_artifacts(
    *,
    wrds_users_parquet: str | Path,
    wrds_positions_parquet: str | Path,
    wrds_users_chunk_dir: str | Path,
    wrds_positions_chunk_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, int | str]:
    users_out = Path(wrds_users_parquet)
    positions_out = Path(wrds_positions_parquet)
    users_chunk_root = Path(wrds_users_chunk_dir)
    positions_chunk_root = Path(wrds_positions_chunk_dir)

    if users_out.exists() and positions_out.exists() and not overwrite:
        return {
            "wrds_users_parquet": str(users_out),
            "wrds_positions_parquet": str(positions_out),
            "wrds_users_chunk_dir": str(users_chunk_root),
            "wrds_positions_chunk_dir": str(positions_chunk_root),
            "wrds_user_rows": _count_parquet_rows(users_out),
            "wrds_position_rows": _count_parquet_rows(positions_out),
            "wrds_query_chunks": len(_list_chunk_files(users_chunk_root)),
        }

    for path in [users_out, positions_out]:
        if overwrite and path.exists():
            path.unlink()

    _merge_chunk_dir_to_output(
        chunk_dir=users_chunk_root,
        output_parquet=users_out,
        empty_columns=WRDS_USERS_COLUMNS,
    )
    _merge_chunk_dir_to_output(
        chunk_dir=positions_chunk_root,
        output_parquet=positions_out,
        empty_columns=WRDS_POSITIONS_COLUMNS,
    )

    return {
        "wrds_users_parquet": str(users_out),
        "wrds_positions_parquet": str(positions_out),
        "wrds_users_chunk_dir": str(users_chunk_root),
        "wrds_positions_chunk_dir": str(positions_chunk_root),
        "wrds_user_rows": _count_parquet_rows(users_out),
        "wrds_position_rows": _count_parquet_rows(positions_out),
        "wrds_query_chunks": len(_list_chunk_files(users_chunk_root)),
    }


def import_wrds_user_and_position_artifacts(
    *,
    matched_user_list_parquet: str | Path,
    wrds_users_parquet: str | Path,
    wrds_positions_parquet: str | Path,
    wrds_users_chunk_dir: str | Path,
    wrds_positions_chunk_dir: str | Path,
    wrds_username: str | None = None,
    chunk_size: int = 10000,
    max_users: int | None = None,
    overwrite: bool = False,
    db=None,
) -> dict[str, int | str]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    matched_users_path = Path(matched_user_list_parquet)
    if not matched_users_path.exists():
        raise FileNotFoundError(f"Matched user list parquet not found: {matched_users_path}")

    users_out = Path(wrds_users_parquet)
    positions_out = Path(wrds_positions_parquet)
    users_chunk_dir = Path(wrds_users_chunk_dir)
    positions_chunk_dir = Path(wrds_positions_chunk_dir)

    if users_out.exists() and positions_out.exists() and not overwrite:
        return {
            "wrds_users_parquet": str(users_out),
            "wrds_positions_parquet": str(positions_out),
            "wrds_users_chunk_dir": str(users_chunk_dir),
            "wrds_positions_chunk_dir": str(positions_chunk_dir),
            "final_import_user_count": len(
                _read_distinct_user_ids(
                    matched_user_list_parquet=matched_users_path,
                    max_users=max_users,
                )
            ),
            "wrds_user_rows": _count_parquet_rows(users_out),
            "wrds_position_rows": _count_parquet_rows(positions_out),
            "wrds_query_chunks": len(_list_chunk_files(users_chunk_dir)),
        }

    if overwrite:
        for path in [users_out, positions_out]:
            if path.exists():
                path.unlink()

    user_ids = _read_distinct_user_ids(
        matched_user_list_parquet=matched_users_path,
        max_users=max_users,
    )
    chunk_stats = write_wrds_import_chunks_for_user_ids(
        user_ids=user_ids,
        wrds_users_chunk_dir=users_chunk_dir,
        wrds_positions_chunk_dir=positions_chunk_dir,
        wrds_username=wrds_username,
        chunk_size=chunk_size,
        overwrite=overwrite,
        db=db,
    )
    merge_stats = consolidate_wrds_user_and_position_artifacts(
        wrds_users_parquet=users_out,
        wrds_positions_parquet=positions_out,
        wrds_users_chunk_dir=users_chunk_dir,
        wrds_positions_chunk_dir=positions_chunk_dir,
        overwrite=overwrite,
    )
    return {
        **merge_stats,
        "final_import_user_count": len(user_ids),
        "wrds_query_chunks": int(chunk_stats["wrds_query_chunks"]),
    }
