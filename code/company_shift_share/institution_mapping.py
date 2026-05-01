"""Helpers for mapping Revelio raw school strings to IPEDS UNITIDs."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import duckdb as ddb
import pandas as pd

DETERMINISTIC_MAPPING_METHOD = "deterministic_ref_inst_catalog_v1"
DETERMINISTIC_WITH_FALLBACK_METHOD = "deterministic_ref_inst_catalog_with_legacy_fallback_v1"
LEGACY_MAPPING_METHOD = "legacy_revelio_ipeds_foia_v1"


def _escape(path: Path) -> str:
    return str(path).replace("'", "''")


def _describe_parquet_columns(con: ddb.DuckDBPyConnection, path: Path) -> list[str]:
    return [
        row[0]
        for row in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{_escape(path)}')"
        ).fetchall()
    ]


def _first_present(columns: list[str], candidates: list[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def sql_normalize_school_key(expr: str) -> str:
    return (
        "TRIM("
        "REGEXP_REPLACE("
        "REGEXP_REPLACE("
        f"LOWER(strip_accents(COALESCE(CAST({expr} AS VARCHAR), ''))), "
        "'[^a-z0-9\\s]+', ' ', 'g'"
        "), "
        "'\\s+', ' ', 'g'"
        ")"
        ")"
    )


def normalize_school_key_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def normalize_school_key_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_school_key_value).astype("string")


def _load_deterministic_school_map(
    deterministic_triple_map: Path,
    ref_inst_catalog: Path,
) -> pd.DataFrame:
    con = ddb.connect()
    try:
        df = con.sql(
            f"""
            WITH base AS (
                SELECT DISTINCT
                    NULLIF(TRIM(CAST(inst_key AS VARCHAR)), '') AS university_raw_key,
                    array_length(inst_candidates) AS n_candidates,
                    inst_candidates
                FROM read_parquet('{_escape(deterministic_triple_map)}')
                WHERE NULLIF(TRIM(CAST(inst_key AS VARCHAR)), '') IS NOT NULL
                  AND inst_candidates IS NOT NULL
                  AND array_length(inst_candidates) > 0
            ),
            expanded AS (
                SELECT
                    b.university_raw_key,
                    idx AS original_rank,
                    CAST(b.inst_candidates[idx].ref_inst_id AS VARCHAR) AS ref_inst_id,
                    TRY_CAST(b.inst_candidates[idx].hybrid_score AS DOUBLE) AS candidate_score,
                    TRY_CAST(ref.main_unitid AS BIGINT) AS unitid
                FROM base AS b
                CROSS JOIN generate_series(1, b.n_candidates) AS gs(idx)
                LEFT JOIN read_parquet('{_escape(ref_inst_catalog)}') AS ref
                  ON CAST(b.inst_candidates[idx].ref_inst_id AS VARCHAR) = CAST(ref.ref_inst_id AS VARCHAR)
            ),
            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY university_raw_key
                        ORDER BY
                            CASE WHEN unitid IS NOT NULL THEN 0 ELSE 1 END,
                            original_rank,
                            candidate_score DESC NULLS LAST,
                            ref_inst_id
                    ) AS rn
                FROM expanded
            )
            SELECT
                university_raw_key,
                CAST(unitid AS VARCHAR) AS unitid,
                ref_inst_id,
                candidate_score,
                original_rank,
                '{DETERMINISTIC_MAPPING_METHOD}' AS row_mapping_method
            FROM ranked
            WHERE rn = 1
              AND unitid IS NOT NULL
            ORDER BY university_raw_key
            """
        ).df()
    finally:
        con.close()
    return df


def _load_legacy_school_map(legacy_crosswalk: Path) -> pd.DataFrame:
    con = ddb.connect()
    try:
        columns = _describe_parquet_columns(con, legacy_crosswalk)
        university_col = _first_present(columns, ["university_raw", "rev_university_raw"])
        unitid_col = _first_present(columns, ["UNITID", "unitid", "main_unitid"])
        if university_col is None or unitid_col is None:
            raise ValueError(
                f"Legacy Revelio school crosswalk is missing required columns: {legacy_crosswalk}"
            )
        df = con.sql(
            f"""
            WITH base AS (
                SELECT
                    NULLIF({sql_normalize_school_key(university_col)}, '') AS university_raw_key,
                    CAST(TRY_CAST({unitid_col} AS BIGINT) AS VARCHAR) AS unitid
                FROM read_parquet('{_escape(legacy_crosswalk)}')
                WHERE TRY_CAST({unitid_col} AS BIGINT) IS NOT NULL
            )
            SELECT
                university_raw_key,
                MIN(unitid) AS unitid,
                NULL::VARCHAR AS ref_inst_id,
                NULL::DOUBLE AS candidate_score,
                NULL::BIGINT AS original_rank,
                '{LEGACY_MAPPING_METHOD}' AS row_mapping_method
            FROM base
            WHERE university_raw_key IS NOT NULL
            GROUP BY university_raw_key
            ORDER BY university_raw_key
            """
        ).df()
    finally:
        con.close()
    return df


def load_revelio_school_map(
    *,
    legacy_crosswalk: Path | None,
    deterministic_triple_map: Path | None = None,
    ref_inst_catalog: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    deterministic_available = (
        deterministic_triple_map is not None
        and ref_inst_catalog is not None
        and deterministic_triple_map.exists()
        and ref_inst_catalog.exists()
    )
    legacy_available = legacy_crosswalk is not None and legacy_crosswalk.exists()

    deterministic_error: str | None = None
    deterministic_df = pd.DataFrame()
    if deterministic_available:
        try:
            deterministic_df = _load_deterministic_school_map(
                deterministic_triple_map=deterministic_triple_map,
                ref_inst_catalog=ref_inst_catalog,
            )
        except Exception as exc:
            deterministic_error = str(exc)

    legacy_df = pd.DataFrame()
    if legacy_available:
        legacy_df = _load_legacy_school_map(legacy_crosswalk=legacy_crosswalk)

    if not deterministic_df.empty:
        mapping_method = (
            DETERMINISTIC_WITH_FALLBACK_METHOD if legacy_available else DETERMINISTIC_MAPPING_METHOD
        )
        if not legacy_df.empty:
            legacy_extra = legacy_df.loc[
                ~legacy_df["university_raw_key"].isin(deterministic_df["university_raw_key"])
            ].copy()
            school_map = pd.concat([deterministic_df, legacy_extra], ignore_index=True)
        else:
            school_map = deterministic_df.copy()
    elif not legacy_df.empty:
        mapping_method = LEGACY_MAPPING_METHOD
        school_map = legacy_df.copy()
    else:
        checked = [
            str(path)
            for path in (deterministic_triple_map, ref_inst_catalog, legacy_crosswalk)
            if path is not None
        ]
        raise FileNotFoundError(
            "Unable to resolve any Revelio school mapping source. "
            f"Checked: {checked or ['(no configured paths)']}"
        )

    school_map["university_raw_key"] = school_map["university_raw_key"].astype("string")
    school_map["unitid"] = school_map["unitid"].astype("string")
    school_map = school_map.dropna(subset=["university_raw_key", "unitid"]).drop_duplicates(
        subset=["university_raw_key"], keep="first"
    )

    metadata: dict[str, Any] = {
        "mapping_method": mapping_method,
        "deterministic_triple_map": str(deterministic_triple_map) if deterministic_triple_map else None,
        "ref_inst_catalog": str(ref_inst_catalog) if ref_inst_catalog else None,
        "legacy_crosswalk": str(legacy_crosswalk) if legacy_crosswalk else None,
        "n_rows": int(len(school_map)),
    }
    if deterministic_error:
        metadata["deterministic_error"] = deterministic_error
    return school_map.reset_index(drop=True), metadata
