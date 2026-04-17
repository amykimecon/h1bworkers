"""Shard-level WRDS education scanning for stage 02_rev_import."""

from __future__ import annotations

import json
from builtins import print as _print
from functools import partial
from pathlib import Path
from typing import Iterable

import duckdb as ddb
import pandas as pd

from common import coerce_int_list, escape_sql_literal
from final_imports import write_wrds_import_chunks_for_user_ids
from legacy_match import match_wrds_users_dataframe
from wrds_query import build_education_scan_query, get_wrds_connection, raw_sql_dataframe

print = partial(_print, flush=True)


def _load_shard_ids(
    *,
    shard_manifest_parquet: str | Path,
    selected_shard_ids: Iterable[int] | None = None,
    max_shards: int | None = None,
) -> list[dict[str, int]]:
    manifest_path = Path(shard_manifest_parquet)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Shard manifest parquet not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    required_cols = {"shard_id", "user_id_lower_bound", "user_id_upper_bound"}
    missing_cols = required_cols - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"Shard manifest is missing required columns: {sorted(missing_cols)}")

    manifest = manifest.loc[:, ["shard_id", "user_id_lower_bound", "user_id_upper_bound"]].copy()
    manifest["shard_id"] = manifest["shard_id"].astype(int)
    manifest["user_id_lower_bound"] = manifest["user_id_lower_bound"].astype(int)
    manifest["user_id_upper_bound"] = manifest["user_id_upper_bound"].astype(int)
    manifest = manifest.sort_values("shard_id").reset_index(drop=True)
    if selected_shard_ids:
        selected = set(coerce_int_list(selected_shard_ids))
        manifest = manifest.loc[manifest["shard_id"].isin(selected)].reset_index(drop=True)
    if max_shards is not None:
        manifest = manifest.head(int(max_shards)).reset_index(drop=True)
    return manifest.to_dict("records")


def _load_marker_stats(marker_path: Path) -> dict[str, int] | None:
    if not marker_path.exists():
        return None
    try:
        raw = json.loads(marker_path.read_text())
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    out: dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value)
        except Exception:
            continue
    return out


def scan_wrds_education_shards(
    *,
    compiled_regex_txt: str | Path,
    shard_manifest_parquet: str | Path,
    matched_education_shard_dir: str | Path | None = None,
    matched_user_shard_dir: str | Path | None = None,
    wrds_users_chunk_dir: str | Path | None = None,
    wrds_positions_chunk_dir: str | Path | None = None,
    wrds_username: str | None = None,
    selected_shard_ids: Iterable[int] | None = None,
    max_shards: int | None = None,
    row_limit_per_shard: int | None = None,
    start_year_threshold: int = 2000,
    end_year_threshold: int = 2004,
    filtered_out_sample_n: int = 0,
    final_import_chunk_size: int = 10000,
    write_match_outputs: bool = False,
    overwrite: bool = False,
    db=None,
) -> dict[str, int | str]:
    regex_path = Path(compiled_regex_txt)
    if not regex_path.exists():
        raise FileNotFoundError(f"Compiled regex artifact not found: {regex_path}")

    matched_educ_dir = Path(matched_education_shard_dir) if matched_education_shard_dir else None
    matched_users_dir = Path(matched_user_shard_dir) if matched_user_shard_dir else None
    if write_match_outputs:
        if matched_educ_dir is None or matched_users_dir is None:
            raise ValueError("matched shard output dirs are required when write_match_outputs=True")
        matched_educ_dir.mkdir(parents=True, exist_ok=True)
        matched_users_dir.mkdir(parents=True, exist_ok=True)

    users_chunk_root = Path(wrds_users_chunk_dir) if wrds_users_chunk_dir else None
    positions_chunk_root = Path(wrds_positions_chunk_dir) if wrds_positions_chunk_dir else None
    direct_wrds_import = users_chunk_root is not None and positions_chunk_root is not None
    if not write_match_outputs and not direct_wrds_import:
        raise ValueError("scan_wrds_education_shards requires shard match outputs or direct WRDS shard outputs.")

    marker_dir = None
    if direct_wrds_import:
        users_chunk_root.mkdir(parents=True, exist_ok=True)
        positions_chunk_root.mkdir(parents=True, exist_ok=True)
        marker_dir = users_chunk_root / "_shard_markers"
        marker_dir.mkdir(parents=True, exist_ok=True)

    regex_pattern = regex_path.read_text().strip() or r"(?!)"
    shard_specs = _load_shard_ids(
        shard_manifest_parquet=shard_manifest_parquet,
        selected_shard_ids=selected_shard_ids,
        max_shards=max_shards,
    )
    if not shard_specs:
        raise ValueError("No shards selected for education scan.")

    own_db = db is None
    if own_db:
        db = get_wrds_connection(wrds_username=wrds_username)

    total_raw_rows = 0
    total_matched_rows = 0
    total_matched_users = 0
    total_wrds_user_rows = 0
    total_wrds_position_rows = 0
    total_wrds_query_chunks = 0
    scanned_shards = 0

    try:
        for shard_spec in shard_specs:
            shard_id = int(shard_spec["shard_id"])
            user_id_lower_bound = int(shard_spec["user_id_lower_bound"])
            user_id_upper_bound = int(shard_spec["user_id_upper_bound"])
            shard_label = f"shard_{shard_id:05d}"

            marker_path = marker_dir / f"{shard_label}.json" if marker_dir is not None else None
            users_shard_dir = users_chunk_root / shard_label if users_chunk_root is not None else None
            positions_shard_dir = positions_chunk_root / shard_label if positions_chunk_root is not None else None
            if overwrite and marker_path is not None and marker_path.exists():
                marker_path.unlink()

            if direct_wrds_import and marker_path is not None and marker_path.exists() and not overwrite:
                marker_stats = _load_marker_stats(marker_path)
                users_ready = users_shard_dir is not None and any(users_shard_dir.glob("*.parquet"))
                positions_ready = positions_shard_dir is not None and any(positions_shard_dir.glob("*.parquet"))
                if marker_stats is not None and users_ready and positions_ready:
                    total_raw_rows += int(marker_stats.get("raw_education_rows", 0))
                    total_matched_rows += int(marker_stats.get("matched_education_rows", 0))
                    total_matched_users += int(marker_stats.get("matched_user_rows", 0))
                    total_wrds_user_rows += int(marker_stats.get("wrds_user_rows", 0))
                    total_wrds_position_rows += int(marker_stats.get("wrds_position_rows", 0))
                    total_wrds_query_chunks += int(marker_stats.get("wrds_query_chunks", 0))
                    scanned_shards += 1
                    continue

            educ_out = (
                matched_educ_dir / f"matched_education_shard_{shard_id:05d}.parquet"
                if matched_educ_dir is not None
                else None
            )
            user_out = (
                matched_users_dir / f"matched_user_ids_shard_{shard_id:05d}.parquet"
                if matched_users_dir is not None
                else None
            )
            if (
                write_match_outputs
                and not direct_wrds_import
                and educ_out is not None
                and user_out is not None
                and educ_out.exists()
                and user_out.exists()
                and not overwrite
            ):
                shard_con = ddb.connect()
                raw_counts = shard_con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{escape_sql_literal(educ_out)}')"
                ).fetchone()[0]
                user_counts = shard_con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{escape_sql_literal(user_out)}')"
                ).fetchone()[0]
                total_matched_rows += int(raw_counts)
                total_matched_users += int(user_counts)
                scanned_shards += 1
                continue

            query = build_education_scan_query(
                user_id_lower_bound=user_id_lower_bound,
                user_id_upper_bound=user_id_upper_bound,
                start_year_threshold=int(start_year_threshold),
                end_year_threshold=int(end_year_threshold),
                row_limit=row_limit_per_shard,
            )
            shard_df = raw_sql_dataframe(db, query)
            shard_raw_rows = int(len(shard_df))
            total_raw_rows += shard_raw_rows

            matched_educ_df, matched_users_df, debug_stats = match_wrds_users_dataframe(
                df=shard_df,
                regex_pattern=regex_pattern,
                source_label=f"wrds_shard_mod_{shard_id}",
                filtered_out_sample_n=filtered_out_sample_n,
            )
            if "shard_id" not in matched_educ_df.columns:
                matched_educ_df["shard_id"] = shard_id
            if "shard_id" not in matched_users_df.columns:
                matched_users_df["shard_id"] = shard_id

            if write_match_outputs and educ_out is not None and user_out is not None:
                matched_educ_df.to_parquet(educ_out, index=False)
                matched_users_df.to_parquet(user_out, index=False)

            shard_matched_rows = int(len(matched_educ_df))
            shard_matched_users = int(len(matched_users_df))
            total_matched_rows += shard_matched_rows
            total_matched_users += shard_matched_users

            shard_import_stats = {
                "final_import_user_count": 0,
                "wrds_user_rows": 0,
                "wrds_position_rows": 0,
                "wrds_query_chunks": 0,
            }
            if direct_wrds_import and users_chunk_root is not None and positions_chunk_root is not None:
                shard_import_stats = write_wrds_import_chunks_for_user_ids(
                    user_ids=matched_users_df["user_id"].dropna().astype(int).tolist(),
                    wrds_users_chunk_dir=users_chunk_root,
                    wrds_positions_chunk_dir=positions_chunk_root,
                    wrds_username=wrds_username,
                    chunk_size=int(final_import_chunk_size),
                    chunk_subdir=shard_label,
                    overwrite=overwrite,
                    db=db,
                )
                total_wrds_user_rows += int(shard_import_stats["wrds_user_rows"])
                total_wrds_position_rows += int(shard_import_stats["wrds_position_rows"])
                total_wrds_query_chunks += int(shard_import_stats["wrds_query_chunks"])
                if marker_path is not None:
                    marker_path.write_text(
                        json.dumps(
                            {
                                "raw_education_rows": shard_raw_rows,
                                "matched_education_rows": shard_matched_rows,
                                "matched_user_rows": shard_matched_users,
                                "wrds_user_rows": int(shard_import_stats["wrds_user_rows"]),
                                "wrds_position_rows": int(shard_import_stats["wrds_position_rows"]),
                                "wrds_query_chunks": int(shard_import_stats["wrds_query_chunks"]),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                    )

            scanned_shards += 1
            print(
                f"[02_rev_import] shard {shard_id:05d} "
                f"[{user_id_lower_bound:,}, {user_id_upper_bound:,}]: raw_rows={shard_raw_rows:,} "
                f"matched_rows={shard_matched_rows:,} matched_users={shard_matched_users:,} "
                f"wrds_user_rows={int(shard_import_stats['wrds_user_rows']):,} "
                f"wrds_position_rows={int(shard_import_stats['wrds_position_rows']):,}"
            )
            if debug_stats is not None:
                filtered_out_count = int(debug_stats["filtered_out_count"])
                filtered_out_sample = debug_stats["filtered_out_sample"]
                print(
                    f"[02_rev_import] shard {shard_id:05d}: "
                    f"filtered_out_rows={filtered_out_count:,}"
                )
                if isinstance(filtered_out_sample, pd.DataFrame) and not filtered_out_sample.empty:
                    print(
                        filtered_out_sample.to_string(
                            index=False,
                            max_colwidth=80,
                        )
                    )
    finally:
        if own_db and db is not None:
            try:
                db.close()
            except Exception:
                pass

    return {
        "compiled_regex_txt": str(regex_path),
        "matched_education_shard_dir": str(matched_educ_dir) if matched_educ_dir is not None else "",
        "matched_user_shard_dir": str(matched_users_dir) if matched_users_dir is not None else "",
        "wrds_users_chunk_dir": str(users_chunk_root) if users_chunk_root is not None else "",
        "wrds_positions_chunk_dir": str(positions_chunk_root) if positions_chunk_root is not None else "",
        "scanned_shards": int(scanned_shards),
        "raw_education_rows_scanned": int(total_raw_rows),
        "matched_education_rows_scanned": int(total_matched_rows),
        "matched_user_rows_scanned": int(total_matched_users),
        "wrds_user_rows_scanned": int(total_wrds_user_rows),
        "wrds_position_rows_scanned": int(total_wrds_position_rows),
        "wrds_query_chunks_scanned": int(total_wrds_query_chunks),
    }
