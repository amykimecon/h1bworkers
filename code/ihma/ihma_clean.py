"""Reusable data cleaning pipeline for IHMA analysis work."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable as IterableABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple, Union

try:  # optional dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm missing
    tqdm = None

import duckdb as ddb
import pandas as pd
import jellyfish

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *

import json
from collections.abc import Iterable as IterableABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple, Union

import duckdb as ddb
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *

from helpers import match_fields_to_cip, get_std_country as _helpers_get_std_country  # proxy into 10_misc.cip_matching



DEFAULT_CLUSTER_MAPPING_PATH = Path(root) / "data" / "clean" / "wrds_clusters_oct2025.csv"
DEFAULT_POSITION_GEOGRAPHY_PATH = Path(root) / "data" / "crosswalks" / "geonames" / "cities500.txt"
COUNTRY_DICT_PATH = Path(root) / "data" / "crosswalks" / "country_dict.json"
GEONAMES_CITIES_PATH = Path(root) / "data" / "crosswalks" / "geonames" / "cities500.txt"
POSITION_GEO_LOOKUP_PATH = Path(root) / "data" / "int" / "position_geography_lookup.parquet"
RSID_GEONAME_CROSSWALK_PATH = Path(root) / "data" / "int" / "rsid_geoname_cw.parquet"
IPEDS_NAME_CW_PATH = Path(root) / "data" / "raw" / "ipeds_name_cw_2021.xlsx"
IPEDS_CW_PATH = Path(root) / "data" / "raw" / "ipeds_cw_2021.csv"
DEFAULT_CIP_CACHE_PATH = Path(root) / "data" / "int" / "cip_match_cache.json"
_GEONAMES_CACHE: Optional[pd.DataFrame] = None
_IPEDS_ALIAS_CACHE: Optional[pd.DataFrame] = None
_TEXT_NORMALIZE_RE = re.compile(r"[^0-9a-z]+")
_CLUSTER_MAPPING_CACHE: Optional[pd.DataFrame] = None
with COUNTRY_DICT_PATH.open("r", encoding="utf-8") as _country_handle:
    _COUNTRY_CODE_DICT = json.load(_country_handle)

_SPECIAL_CLUSTER_PREFIXES = ("generic::", "non_degree::", "nondegree::")

CBSA_SHARE_START_YEAR = 1991
CBSA_SHARE_END_YEAR = 2022
POS_START_YEAR = 2000
POS_END_YEAR = 2025
CBSA_SHARE_TARGET_YEAR = 2005
MAX_RELATIVE_YEARS = 9


def _parse_group_option(value: str, allowed: Set[str], label: str) -> Tuple[str, bool]:
    text = value.strip().lower()
    use_cip = False
    for suffix in ("_x_cip", "_cip"):
        if text.endswith(suffix):
            use_cip = True
            text = text[: -len(suffix)]
            break
    if text not in allowed:
        raise ValueError(
            f"{label} must be one of {sorted(allowed)} optionally suffixed with '_x_cip' to include CIP codes."
        )
    return text, use_cip


def _cip_prefix_expr(column: str) -> str:
    """Return SQL snippet that extracts the two-digit CIP prefix (digits only)."""

    sanitized = f"regexp_replace(coalesce({column}, ''), '[^0-9]', '', 'g')"
    return f"NULLIF(substr({sanitized}, 1, 2), '')"


def _ensure_std_country_function(con: ddb.DuckDBPyConnection) -> None:
    if getattr(_ensure_std_country_function, "_registered", False):
        return
    con.create_function(
        "get_std_country",
        lambda value: _helpers_get_std_country(value, _COUNTRY_CODE_DICT),
        ["VARCHAR"],
        "VARCHAR",
    )
    setattr(_ensure_std_country_function, "_registered", True)


def _normalize_metro_sql(column: str) -> str:
    """
    Return a SQL snippet that trims a metro name and removes common suffixes.

    Keeping this logic centralized ensures positions, crosswalks, and analysis
    views all normalize text in the same way before attempting geoname matches.
    """

    base = f"TRIM({column})"
    return (
        f"TRIM(LOWER(REPLACE(REPLACE(REPLACE(REPLACE({base}, 'nonmetropolitan area', ''), "
        f"'metropolitan area', ''), 'metro area', ''), 'metro', '')))"
    )

@dataclass(frozen=True)
class IHMACleanTables:
    """Names of temp views that downstream code can query."""

    ihma_samp: str
    ihma_educ: str
    ihma_us_educ_clean: str
    ihma_user: str
    ihma_pos: str
    ihma_pos_clean: str
    pos_clean_msa: str
    earn_clean: str
    positions_msa: str
    ipeds_ma: str
    ipeds_ma_nodist: str
    ipeds_geoname: str
    cbsa_shares: str
    cbsa_shares_main: str
    educ_cbsa_shares: str
    educ_cbsa_shares_main: str
    ipeds_cbsa_growth: str
    instrument_components: str
    instrument_panel: str
    indep_constr: str
    dep_constr: str


def prepare_clean_data(
    con: Optional[ddb.DuckDBPyConnection] = None,
    *,
    cluster_mapping_path: Optional[Path] = DEFAULT_CLUSTER_MAPPING_PATH,
    position_geography_path: Optional[Path] = DEFAULT_POSITION_GEOGRAPHY_PATH,
    cip_batch_size: int = 1000,
    cip_digit_length: int = 6,
    cip_cache_only: bool = True,
    cip_cache_path: Optional[Path] = None,
    group_positions_by_geoname: bool = True,
    verbose: bool = True,
    save_intermediate: bool = True,
    intermediate_dir: Optional[Path] = None,
    from_intermediate: bool = False,
    origin_group_by: str = "origin_rsid",
    us_group_by: str = "us_cluster_geo_city_id",
    force_resave: bool = False,
) -> IHMACleanTables:
    """
    Build cleaned IHMA views inside the provided DuckDB connection.

    Parameters
    ----------
    con
        Existing DuckDB connection. When omitted, a new in-memory connection
        is created.
    cluster_mapping_path
        Path to a precomputed cluster assignment table containing at least
        `cluster_root` and a member-name column. Entries are matched back to the
        IHMA source views instead of clustering inside this function.
    position_geography_path
        Optional parquet produced by `clean_revelio_institutions.py` that maps
        each position_id to a geonames identifier and city metadata. When
        provided, the positions table is enriched with those columns.
    cip_batch_size
        Number of unique fields to send through the CIP matcher per batch.
    cip_digit_length
        CIP digit granularity forwarded to match_fields_to_cip (default 6-digit codes).
    cip_cache_only
        When True, reuse cached CIP matches without scoring new fields.
    cip_cache_path
        Optional override for the CIP cache location used when cache_only is enabled.
        When omitted or empty, defaults to data/int/cip_match_cache.json.
    origin_group_by
        Either "origin_rsid" or "origin_cluster_geo_city_id". Append "_x_cip" (or "_cip")
        to include CIP codes in the grouping key.
    us_group_by
        Either "us_cluster_geo_city_id" or "us_cluster_ipeds_ids". Append "_x_cip"
        (or "_cip") to include CIP codes in the grouping key.
    verbose
        When True, log high-level progress for long-running steps.

    Returns
    -------
    IHMACleanTables
        Helper containing the temp-view names that were created.
    """

    connection = con or ddb.connect()
    _ensure_std_country_function(connection)
    cluster_mapping = _read_cluster_mapping(cluster_mapping_path)
    cache_root = Path(intermediate_dir) if intermediate_dir is not None else (Path(root) / "data" / "int" / "ihma_intermediate")
    origin_base, origin_use_cip = _parse_group_option(
        origin_group_by,
        {"origin_rsid", "origin_cluster_geo_city_id"},
        "origin_group_by",
    )
    us_base, us_use_cip = _parse_group_option(
        us_group_by,
        {"us_cluster_geo_city_id", "us_cluster_ipeds_ids"},
        "us_group_by",
    )
    cache_dir = cache_root / f"{origin_base}{'_cip' if origin_use_cip else ''}__{us_base}{'_cip' if us_use_cip else ''}"

    _create_base_views(connection)
    geonames_view = _register_position_geography_view(connection, position_geography_path)

    samp_view = _attach_cluster_metadata(
        connection,
        source_view="ihma_samp_base",
        name_column="institution_name_for_cluster",
        prefix="origin",
        cluster_mapping=cluster_mapping,
        verbose=verbose,
    )
    print(samp_view)
    educ_view = _attach_cluster_metadata(
        connection,
        source_view="ihma_educ_base",
        name_column="institution_name_for_cluster",
        prefix="us",
        cluster_mapping=cluster_mapping,
        verbose=verbose,
    )

    resolved_cip_cache_path = cip_cache_path or DEFAULT_CIP_CACHE_PATH

    samp_view = _attach_cip_matches(
        connection,
        source_view=samp_view,
        batch_size=cip_batch_size,
        digit_length=cip_digit_length,
        verbose=verbose,
        cache_only=cip_cache_only,
        cache_path=resolved_cip_cache_path,
    )

    educ_view = _attach_cip_matches(
        connection,
        source_view=educ_view,
        batch_size=cip_batch_size,
        digit_length=cip_digit_length,
        verbose=verbose,
        cache_only=cip_cache_only,
        cache_path=resolved_cip_cache_path,
    )

    base_tables = _create_downstream_views(
        connection,
        samp_view=samp_view,
        educ_view=educ_view,
        geonames_view=geonames_view,
        group_positions_by_geoname=group_positions_by_geoname,
        verbose=verbose,
        save_intermediate=save_intermediate,
        from_intermediate=from_intermediate,
        cache_dir=cache_dir,
        origin_group_by=origin_base,
        origin_group_by_use_cip=origin_use_cip,
        us_group_by=us_base,
        us_group_by_use_cip=us_use_cip,
        force_resave=force_resave,
    )

    analysis_tables = _create_analysis_views(
        connection,
        verbose=verbose,
        save_intermediate=save_intermediate,
        cache_dir=cache_dir,
        from_intermediate=from_intermediate,
        origin_group_by=origin_base,
        us_group_by=us_base,
        force_resave=force_resave,
    )

    print("IHMA cleaning complete.")
    
    return IHMACleanTables(**base_tables, **analysis_tables)


def _create_base_views(con: ddb.DuckDBPyConnection) -> None:
    """Register the parquet inputs that every downstream step needs."""

    sources = {
        "ihma_samp_base": Path(root) / "data/int/ihma_users_all_nov2025.parquet",
        "ihma_educ_base": Path(root) / "data/int/ihma_educ_all_oct20.parquet",
        "ihma_pos_raw": Path(root) / "data/int/ihma_positions_all_oct20.parquet",
        "positions_msa": Path(root) / "data/int/positions_msa_cw.parquet",
        "ipeds_ma": Path(root) / "data/int/ipeds_ma_only_grp.parquet",
        "ipeds_ma_nodist": Path(root) / "data/int/ipeds_ma_only_nodist_grp.parquet",
        "ipeds_geoname": Path(root) / "data/int/ipeds_geoname_cw.parquet",
    }

    for view, path in sources.items():
        parquet_scan = _parquet_scan(path)
        # TEMP: filter samp
        if view == "ihma_samp_base":
            con.sql(f"""
                CREATE OR REPLACE TEMP VIEW {view} AS   
                SELECT *, {help.degree_clean_regex_sql()} AS degree_clean, {help.field_clean_regex_sql("field_raw")} AS field_clean_pre, university_raw AS institution_name_for_cluster FROM {parquet_scan} WHERE NOT university_country = 'United States'
                """)
        elif view in {"ihma_samp_base", "ihma_educ_base"}:
            con.sql(
                f"""
                CREATE OR REPLACE TEMP VIEW {view} AS
                SELECT
                    *, {help.degree_clean_regex_sql()} AS degree_clean, {help.field_clean_regex_sql("field_raw")} AS field_clean_pre,
                    university_raw AS institution_name_for_cluster
                FROM {parquet_scan}
                """
            )
        else:
            con.sql(f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT * FROM {parquet_scan}")


def _register_position_geography_view(
    con: ddb.DuckDBPyConnection,
    geo_path: Optional[Path],
) -> Optional[str]:
    if geo_path is None:
        return None
    path = Path(geo_path).expanduser()
    if not path.exists():
        print(f"Geonames file {path} not found; skipping position geocode enrichment.")
        return None

    cached_lookup = POSITION_GEO_LOOKUP_PATH
    crosswalk_view = "position_geography_lookup"
    if cached_lookup.exists():
        escaped_cache = str(cached_lookup).replace("'", "''")
        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW {crosswalk_view} AS
            SELECT * FROM read_parquet('{escaped_cache}')
            """
        )
        print(f"Loaded cached position geography lookup from {cached_lookup}")
        return crosswalk_view

    escaped = str(path).replace("'", "''")

    geonames_raw_view = "geonames_city_lookup"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP View {geonames_raw_view} AS
        SELECT
            CAST(column00 AS VARCHAR) AS geoname_id,
            column01 AS city_name,
            get_std_country(UPPER(TRIM(column08))) AS country_norm,
            TRY_CAST(column14 AS BIGINT) AS population,
            LOWER(TRIM(column01)) AS city_norm,
            LOWER(TRIM(column02)) AS city_ascii_norm
        FROM read_csv_auto(
            '{escaped}',
            delim='\t'
        )
        """
    )
    metro_norm_expr = _normalize_metro_sql("metro_area")
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {crosswalk_view} AS
        WITH metros AS (
            SELECT metro_area AS metro_raw, country AS country_raw,
                {metro_norm_expr} AS metro_norm,
                get_std_country(TRIM(country)) AS country_norm
            FROM (
                SELECT DISTINCT metro_area, country
                FROM ihma_pos_raw
                WHERE metro_area IS NOT NULL
                  AND country IS NOT NULL
            )
        ),
        ranked AS (
            SELECT
                m.metro_raw,
                m.metro_norm,
                m.country_raw,
                m.country_norm,
                g.geoname_id,
                g.city_name,
                g.country_norm AS geo_country_norm,
                COALESCE(g.population, 0) AS pop,
                ROW_NUMBER() OVER (
                    PARTITION BY m.metro_norm, m.country_norm
                    ORDER BY COALESCE(g.population, 0) DESC, g.city_name
                ) AS rn
            FROM metros m
            JOIN {geonames_raw_view} AS g
              ON m.country_norm = g.country_norm
             AND (
                m.metro_norm = g.city_norm
                OR m.metro_norm = g.city_ascii_norm
             )
        ),
        matched AS (
            SELECT
                metro_raw,
                metro_norm,
                country_raw,
                country_norm,
                geoname_id AS pos_geo_geoname_id,
                city_name AS pos_geo_city_name,
                geo_country_norm AS pos_geo_country_name
            FROM ranked
            WHERE rn = 1
        ),
        unmatched AS (
            SELECT
                m.metro_raw,
                m.metro_norm,
                m.country_raw,
                m.country_norm,
                CONCAT(
                    '999',
                    COALESCE(NULLIF(m.country_norm, ''), 'UNK')
                ) AS pos_geo_geoname_id,
                COALESCE(NULLIF(m.metro_raw, ''), 'Nonmetropolitan Area') AS pos_geo_city_name,
                m.country_norm AS pos_geo_country_name
            FROM metros AS m
            LEFT JOIN matched AS mt
              ON m.metro_norm = mt.metro_norm
             AND m.country_norm = mt.country_norm
            WHERE mt.metro_norm IS NULL
        )
        SELECT * FROM matched
        UNION ALL
        SELECT * FROM unmatched
        """
    )

    try:
        cached_lookup.parent.mkdir(parents=True, exist_ok=True)
        escaped_cache = str(cached_lookup).replace("'", "''")
        con.sql(
            f"""
            COPY (SELECT * FROM {crosswalk_view})
            TO '{escaped_cache}' (FORMAT 'parquet')
            """
        )
        print(f"Persisted position geography lookup to {cached_lookup}")
    except Exception as exc:  # pragma: no cover - best effort cache
        print(f"Warning: failed to persist position geography lookup to {cached_lookup} ({exc})")

    return crosswalk_view


def _maybe_save_view(
    con: ddb.DuckDBPyConnection,
    view_name: str,
    cache_dir: Path,
    enabled: bool,
    force: bool = False,
) -> None:
    if not enabled:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{view_name}.parquet"
        if path.exists() and not force:
            print(f"Parquet cache already exists for {view_name} at {path}; skipping save.")
            return
        escaped = str(path).replace("'", "''")
        con.sql(
            f"""
            COPY (SELECT * FROM {view_name})
            TO '{escaped}' (FORMAT 'parquet', COMPRESSION 'ZSTD')
            """
        )
        print(f"Saved {view_name} to {path}")
    except Exception as exc:  # pragma: no cover - cache best-effort
        print(f"Warning: failed to save {view_name} to parquet ({exc})")


def _maybe_load_view(con: ddb.DuckDBPyConnection, view_name: str, cache_dir: Path, enabled: bool) -> bool:
    if not enabled:
        return False
    path = cache_dir / f"{view_name}.parquet"
    if not path.exists():
        print(f"Parquet cache not found for {view_name} at {path}; skipping load.")
        return False
    escaped = str(path).replace("'", "''")
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        SELECT * FROM read_parquet('{escaped}')
        """
    )
    print(f"Loaded cached view {view_name} from {path}")
    return True


def _parquet_scan(path: Path) -> str:
    escaped = str(path).replace("'", "''")
    return f"read_parquet('{escaped}')"


def _attach_cluster_metadata(
    con: ddb.DuckDBPyConnection,
    *,
    source_view: str,
    name_column: str,
    prefix: str,
    cluster_mapping: pd.DataFrame,
    verbose: bool,
) -> str:
    """Attach precomputed cluster metadata to the requested source view."""

    if cluster_mapping.empty:
        print(f"Warning: cluster mapping table is empty; skipping cluster attachment for {source_view}")
        return source_view

    col_expr = f"{name_column}"
    trimmed_col_expr = f"TRIM({col_expr})"

    names_count = con.sql(
        f"""
        SELECT COUNT(DISTINCT {trimmed_col_expr})
        FROM {source_view}
        WHERE {col_expr} IS NOT NULL AND LENGTH({trimmed_col_expr}) > 0
        """
    ).fetchone()[0]
    if names_count < 2:
        if verbose:
            print(f"Skipping cluster attachment for {source_view} (not enough names).")
        return source_view

    if verbose:
        print(f"Linking {names_count:,} unique institutions to precomputed clusters for {source_view}")

    map_name = f"{prefix}_cluster_map"
    con.register(map_name, cluster_mapping)

    filtered_map_view = f"{map_name}_filtered"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {filtered_map_view} AS
        SELECT DISTINCT
            map.cluster_member_name,
            map.cluster_root,
            map.cluster_display_name,
            map.matched_city,
            map.matched_institution_id,
            map.ipeds_ids,
            map.cluster_size,
            map.matched_geo_city_id,
            map.matched_country_code,
            map.matched_geo_city_name,
            map.primary_ipeds_id
        FROM {map_name} AS map
        JOIN (
            SELECT DISTINCT TRIM({col_expr}) AS cluster_member_name
            FROM {source_view}
            WHERE {col_expr} IS NOT NULL AND LENGTH(TRIM({col_expr})) > 0
        ) AS names
        ON map.cluster_member_name = names.cluster_member_name
        """
    )
    match_count = con.sql(f"SELECT COUNT(*) FROM {filtered_map_view}").fetchone()[0]
    if match_count == 0:
        print(f"Warning: no cluster assignments found for {source_view}")
        return source_view

    base_view = f"{source_view}_{prefix}_cluster_join"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {base_view} AS
        SELECT
            src.*,
            map.cluster_root AS {prefix}_cluster_root_original,
            map.cluster_display_name AS {prefix}_cluster_name,
            map.cluster_size AS {prefix}_cluster_size,
            map.matched_city AS {prefix}_cluster_city,
            map.matched_institution_id AS {prefix}_cluster_institution_id,
            map.ipeds_ids AS {prefix}_cluster_ipeds_ids,
            map.matched_country_code AS {prefix}_cluster_country_code,
            map.matched_geo_city_id AS {prefix}_cluster_geo_city_id,
            map.matched_geo_city_name AS {prefix}_cluster_geo_city_name,
            map.primary_ipeds_id AS {prefix}_primary_ipeds_id
        FROM {source_view} AS src
        LEFT JOIN {filtered_map_view} AS map
        ON TRIM(src.{col_expr}) = map.cluster_member_name
        """
    )

    target_view = f"{source_view}_{prefix}_clustered"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {target_view} AS
        WITH member_counts AS (
            SELECT
                {prefix}_cluster_root_original AS cluster_root,
                TRIM({col_expr}) AS cluster_member_name,
                COUNT(*) AS member_count
            FROM {base_view}
            WHERE {prefix}_cluster_root_original IS NOT NULL
              AND {col_expr} IS NOT NULL
              AND LENGTH(TRIM({col_expr})) > 0
            GROUP BY 1, 2
        ),
        ranked_members AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY cluster_root
                    ORDER BY member_count DESC, cluster_member_name
                ) AS rn
            FROM member_counts
        ),
        root_choice AS (
            SELECT cluster_root, cluster_member_name AS cluster_display_name
            FROM ranked_members
            WHERE rn = 1
        )
        SELECT
            base.* EXCLUDE ({prefix}_cluster_root_original),
            COALESCE(root_choice.cluster_display_name, base.{prefix}_cluster_root_original) AS {prefix}_cluster_root
        FROM {base_view} AS base
        LEFT JOIN root_choice
            ON base.{prefix}_cluster_root_original = root_choice.cluster_root
        """
    )
    return target_view


def _read_cluster_mapping(cluster_mapping_path: Optional[Path]) -> pd.DataFrame:
    global _CLUSTER_MAPPING_CACHE
    if _CLUSTER_MAPPING_CACHE is not None:
        return _CLUSTER_MAPPING_CACHE.copy()
    if cluster_mapping_path is None:
        raise ValueError("cluster_mapping_path must be provided.")
    mapping_path = Path(cluster_mapping_path).expanduser()
    if not mapping_path.exists():
        raise FileNotFoundError(f"Cluster mapping file not found: {mapping_path}")
    frame = pd.read_csv(mapping_path)
    if frame.empty:
        print(f"Warning: cluster mapping file {mapping_path} is empty.")
        _CLUSTER_MAPPING_CACHE = pd.DataFrame(columns=[
            "cluster_member_name",
            "cluster_root",
            "matched_city",
            "matched_institution_id",
            "ipeds_ids",
            "cluster_size",
            "matched_geo_city_id",
            "primary_ipeds_id",
        ])
        return _CLUSTER_MAPPING_CACHE.copy()

    normalized = _standardize_cluster_mapping(frame)
    finalized = _finalize_cluster_mapping(normalized)
    annotated = _annotate_cluster_primary_ipeds(finalized)
    _CLUSTER_MAPPING_CACHE = annotated
    return annotated.copy()


def _standardize_cluster_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    lower_cols = {col.lower(): col for col in frame.columns}

    def _get_column(candidates: Sequence[str]) -> Optional[str]:
        for cand in candidates:
            col = lower_cols.get(cand.lower())
            if col:
                return col
        return None

    member_col = _get_column(["cluster_member_name", "university_name", "institution_name", "name"])
    root_col = _get_column(["cluster_root", "cluster_id", "root"])
    if member_col is None or root_col is None:
        raise ValueError("Cluster mapping must contain columns for member names and cluster roots.")

    rename_map = {
        member_col: "cluster_member_name",
        root_col: "cluster_root",
    }
    optional_column_map = {
        "matched_city": ["matched_city", "cluster_city", "city"],
        "matched_institution_id": ["matched_institution_id", "institution_id", "matched_institution"],
        "ipeds_ids": ["ipeds_ids", "ipeds", "ipeds_id"],
        "cluster_size": ["cluster_size"],
        "matched_geo_city_id": ["matched_geo_city_id", "geo_city_id", "matched_geoname_id"],
    }
    for canonical, candidates in optional_column_map.items():
        col_name = _get_column(candidates)
        if col_name is not None:
            rename_map[col_name] = canonical

    standardized = frame.rename(columns=rename_map)
    standardized["cluster_member_name"] = standardized["cluster_member_name"].apply(_strip_special_label)
    standardized["cluster_root"] = standardized["cluster_root"].apply(_drop_special_identifier)
    standardized = standardized.dropna(subset=["cluster_member_name", "cluster_root"]).copy()
    standardized["cluster_member_name"] = standardized["cluster_member_name"].astype(str)
    standardized["cluster_root"] = standardized["cluster_root"].astype(str)
    standardized["ipeds_ids"] = standardized.get("ipeds_ids")

    for required in ["matched_city", "matched_institution_id", "ipeds_ids", "cluster_size"]:
        if required not in standardized.columns:
            standardized[required] = None

    standardized["cluster_member_name"] = standardized["cluster_member_name"].astype(str).str.strip()
    standardized = standardized[standardized["cluster_member_name"] != ""]
    standardized = standardized.drop_duplicates(subset=["cluster_member_name"], keep="first")
    standardized["is_generic_cluster"] = standardized["cluster_root"].str.startswith("generic::")
    standardized["is_generic_member"] = standardized["cluster_member_name"].str.startswith("generic::")
    standardized["is_generic"] = standardized["is_generic_cluster"] | standardized["is_generic_member"]
    standardized["cluster_root"] = standardized["cluster_root"].mask(standardized["is_generic_cluster"])
    standardized["matched_institution_id"] = standardized["matched_institution_id"].mask(standardized["is_generic"])
    standardized["cluster_member_name"] = standardized["cluster_member_name"].where(
        ~standardized["is_generic_member"],
        standardized["cluster_member_name"].str.split("::").str[-1],
    )
    return standardized[
        [
            "cluster_member_name",
            "cluster_root",
            "matched_city",
            "matched_institution_id",
            "ipeds_ids",
            "cluster_size",
            "matched_geo_city_id",
        ]
    ]


def _finalize_cluster_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["cluster_member_name", "cluster_root", "matched_city", "matched_institution_id", "ipeds_ids", "cluster_size"])

    working = frame.copy()
    if "cluster_size" not in working.columns:
        working["cluster_size"] = working.groupby("cluster_root")["cluster_member_name"].transform("count")
    else:
        working["cluster_size"] = working["cluster_size"].fillna(
            working.groupby("cluster_root")["cluster_member_name"].transform("count")
        )

    optional_cols = ["matched_city", "matched_institution_id", "ipeds_ids", "matched_geo_city_id"]
    for col in optional_cols:
        if col not in working.columns:
            working[col] = None

    working["cluster_display_name"] = working["cluster_root"].where(working["cluster_root"].notna(), working["cluster_member_name"])

    working["matched_institution_id"] = working["matched_institution_id"].apply(_clean_cluster_string)
    working["matched_institution_id"] = working["matched_institution_id"].apply(_drop_special_identifier)
    working["matched_geo_city_id"] = working["matched_geo_city_id"].apply(_normalize_geo_id)
    geoname_meta = _load_geonames_city_metadata()
    if not working["matched_geo_city_id"].isnull().all():
        enriched = working.merge(
            geoname_meta,
            how="left",
            left_on="matched_geo_city_id",
            right_on="geo_city_id",
        )
        enriched = enriched.drop(columns=["geo_city_id"])
        working = enriched
    if "matched_country_code" not in working.columns:
        working["matched_country_code"] = None
    if "matched_geo_city_name" not in working.columns:
        working["matched_geo_city_name"] = None

    columns = [
        "cluster_member_name",
        "cluster_root",
        "cluster_display_name",
        "matched_city",
        "matched_institution_id",
        "ipeds_ids",
        "cluster_size",
        "matched_geo_city_id",
        "matched_country_code",
        "matched_geo_city_name",
    ]
    return working[columns]


def _load_geonames_city_metadata() -> pd.DataFrame:
    global _GEONAMES_CACHE
    if _GEONAMES_CACHE is not None:
        return _GEONAMES_CACHE
    if not GEONAMES_CITIES_PATH.exists():
        print(f"Warning: Geonames cities file not found at {GEONAMES_CITIES_PATH}; matched country enrichment will be skipped.")
        _GEONAMES_CACHE = pd.DataFrame(
            columns=["geo_city_id", "matched_country_code", "matched_geo_city_name"]
        )
        return _GEONAMES_CACHE
    try:
        df = pd.read_csv(
            GEONAMES_CITIES_PATH,
            sep="\t",
            header=None,
            usecols=[0, 1, 8],
            names=["geo_city_id", "matched_geo_city_name", "matched_country_code"],
            dtype=str,
            keep_default_na=False,
        )
        df["geo_city_id"] = df["geo_city_id"].astype(str).str.strip()
        df["matched_country_code"] = df["matched_country_code"].astype(str).str.strip()
        df["matched_geo_city_name"] = df["matched_geo_city_name"].astype(str).str.strip()
    except Exception as exc:
        print(f"Warning: failed to load geonames city metadata ({exc}); country enrichment skipped.")
        df = pd.DataFrame(columns=["geo_city_id", "matched_country_code", "matched_geo_city_name"])
    _GEONAMES_CACHE = df
    return _GEONAMES_CACHE.copy()


def _normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = value.lower()
    text = _TEXT_NORMALIZE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_ipeds_ids(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(',') if part.strip()]
    cleaned = []
    for part in parts:
        digits = re.sub(r"[^0-9]", "", part)
        if digits:
            cleaned.append(digits)
    return cleaned


def _load_ipeds_alias_table() -> pd.DataFrame:
    global _IPEDS_ALIAS_CACHE
    if _IPEDS_ALIAS_CACHE is not None:
        return _IPEDS_ALIAS_CACHE.copy()
    if not IPEDS_NAME_CW_PATH.exists() or not IPEDS_CW_PATH.exists():
        print(
            f"Warning: IPEDS alias sources not found ({IPEDS_NAME_CW_PATH} / {IPEDS_CW_PATH}); "
            "IPED-based grouping will fall back to raw IDs."
        )
        _IPEDS_ALIAS_CACHE = pd.DataFrame(columns=["unitid", "instname", "alias_clean"])
        return _IPEDS_ALIAS_CACHE.copy()
    try:
        univ_cw = pd.read_excel(
            IPEDS_NAME_CW_PATH,
            sheet_name="Crosswalk",
            usecols=["OPEID", "IPEDSMatch", "PEPSSchname", "PEPSLocname", "IPEDSInstnm", "OPEIDMain", "IPEDSMain"],
        )
        univ_cw["UNITID"] = univ_cw["IPEDSMatch"].astype(str).str.replace("No match", "-1", regex=False).astype(int)
    except Exception as exc:
        print(f"Warning: failed to load IPEDS name crosswalk ({exc}); IPED-based grouping disabled.")
        _IPEDS_ALIAS_CACHE = pd.DataFrame(columns=["unitid", "instname", "alias_clean"])
        return _IPEDS_ALIAS_CACHE.copy()
    try:
        zip_cw = pd.read_csv(
            IPEDS_CW_PATH,
            usecols=["UNITID", "OPEID", "INSTNM", "CITY", "STABBR", "ZIP", "ALIAS"],
        )
    except Exception as exc:
        print(f"Warning: failed to load IPEDS alias csv ({exc}); IPED-based grouping disabled.")
        _IPEDS_ALIAS_CACHE = pd.DataFrame(columns=["unitid", "instname", "alias_clean"])
        return _IPEDS_ALIAS_CACHE.copy()

    merged = (
        univ_cw[univ_cw["UNITID"] != -1]
        .merge(zip_cw, on=["UNITID", "OPEID"], how="left")
        .melt(
            id_vars=["UNITID", "CITY", "STABBR", "ZIP"],
            value_vars=["PEPSSchname", "PEPSLocname", "IPEDSInstnm", "INSTNM", "ALIAS"],
            var_name="source",
            value_name="instname",
        )
        .dropna(subset=["instname"])
        .drop_duplicates(subset=["UNITID", "instname"])
        .reset_index(drop=True)
    )
    merged["unitid"] = merged["UNITID"].astype(str).str.strip()
    merged["instname"] = merged["instname"].astype(str).str.strip()
    merged["alias_clean"] = merged["instname"].map(_normalize_text)
    merged = merged[merged["alias_clean"].ne("")]
    alias_df = merged[["unitid", "instname", "alias_clean"]].drop_duplicates()
    _IPEDS_ALIAS_CACHE = alias_df
    return alias_df.copy()


def _annotate_cluster_primary_ipeds(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame["primary_ipeds_id"] = None
        return frame

    alias_table = _load_ipeds_alias_table()
    if alias_table.empty:
        frame = frame.copy()
        frame["primary_ipeds_id"] = frame["ipeds_ids"].map(lambda value: _parse_ipeds_ids(value)[0] if _parse_ipeds_ids(value) else None)
        return frame

    cluster_subset = frame[[
        "cluster_root",
        "ipeds_ids",
    ]].drop_duplicates().copy()

    conn = ddb.connect()
    conn.register("cluster_ipeds_source", cluster_subset)
    conn.register("ipeds_alias_lookup", alias_table)

    conn.execute(
        """
        CREATE OR REPLACE TEMP VIEW cluster_ipeds_expanded AS
        WITH exploded AS (
            SELECT
                cluster_root,
                REGEXP_REPLACE(REGEXP_REPLACE(value, '^\\s+|\\s+$', ''), '[^0-9]', '') AS raw_unitid
        FROM cluster_ipeds_source,
        UNNEST(
            regexp_split_to_array(
                regexp_replace(regexp_replace(COALESCE(ipeds_ids, ''), '[()]', ''), '''', ''),
                ','
            )
        ) AS t(value)
        )
        SELECT DISTINCT
            cluster_root,
            REGEXP_REPLACE(raw_unitid, '[^0-9]', '') AS unitid
        FROM exploded
        WHERE unitid <> ''
        """
    )

    singletons = conn.execute(
        """
        SELECT cluster_root, unitid AS primary_ipeds_id
        FROM (
            SELECT
                cluster_root,
                unitid,
                COUNT(*) OVER (PARTITION BY cluster_root) AS unit_count
            FROM cluster_ipeds_expanded
        )
        WHERE unit_count = 1
        """
    ).df()

    multiples = conn.execute(
        """
        SELECT cluster_root, unitid
        FROM (
            SELECT
                cluster_root,
                unitid,
                COUNT(*) OVER (PARTITION BY cluster_root) AS unit_count
            FROM cluster_ipeds_expanded
        )
        WHERE unit_count > 1
        """
    ).df()

    if multiples.empty:
        crosswalk = singletons
    else:
        conn.register("multiples_unitids", multiples)
        conn.execute(
            """
            CREATE OR REPLACE TEMP VIEW cluster_alias_scores AS
            SELECT
                m.cluster_root,
                m.unitid,
                jaro_similarity(
                    LOWER(REGEXP_REPLACE(c.cluster_root, '[^0-9a-z ]', ' ')),
                    alias.alias_clean
                ) AS jw_score
            FROM multiples_unitids AS m
            JOIN cluster_ipeds_source AS c USING(cluster_root)
            JOIN ipeds_alias_lookup AS alias
              ON alias.unitid = m.unitid
            """
        )
        best = conn.execute(
            """
            SELECT cluster_root, unitid AS primary_ipeds_id
            FROM (
                SELECT
                    cluster_root,
                    unitid,
                    jw_score,
                    ROW_NUMBER() OVER (PARTITION BY cluster_root ORDER BY jw_score DESC, unitid) AS rn
                FROM cluster_alias_scores
            )
            WHERE rn = 1
            """
        ).df()
        crosswalk = pd.concat([singletons, best], ignore_index=True, sort=False)
    conn.close()

    frame = frame.copy()
    frame = frame.merge(crosswalk, on="cluster_root", how="left")
    return frame


def _ensure_ipeds_alias_view(con: ddb.DuckDBPyConnection) -> bool:
    alias_df = _load_ipeds_alias_table()
    if alias_df.empty:
        return False
    con.register("ipeds_alias_df", alias_df)
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW ipeds_alias_lookup AS
        SELECT
            unitid,
            instname,
            alias_clean
        FROM ipeds_alias_df
        """
    )
    return True




def _clean_cluster_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _strip_special_label(value: Any) -> Optional[str]:
    clean = _clean_cluster_string(value)
    if clean is None:
        return None
    for prefix in _SPECIAL_CLUSTER_PREFIXES:
        if clean.startswith(prefix):
            parts = clean.split("::")
            return parts[-1] if parts else None
    return clean


def _drop_special_identifier(value: Any) -> Optional[str]:
    clean = _clean_cluster_string(value)
    if clean is None:
        return None
    for prefix in _SPECIAL_CLUSTER_PREFIXES:
        if clean.startswith(prefix):
            return None
    return clean


def _normalize_geo_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float):
        if pd.isna(value):
            return None
        if value.is_integer():
            return str(int(value))
        return str(value).strip()
    text = str(value).strip()
    return text or None


def _tuple_to_csv(value: Optional[Iterable[str]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, IterableABC):
        parts = [str(v).strip() for v in value if v]
        parts = [part for part in parts if part]
        return ",".join(parts) if parts else None
    return str(value)


def _load_cip_cache_dataframe(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"CIP cache not found at {cache_path}. Run matching once without cache_only to seed it."
        )
    raw_text = cache_path.read_text(encoding="utf-8")
    payload: Dict[str, Any]
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - tolerate truncated cache
        trimmed = raw_text[: exc.pos]
        entry_start = trimmed.rfind('\n  "')
        if entry_start == -1:
            raise ValueError(f"Failed to parse CIP cache at {cache_path}: {exc}") from exc
        trimmed = trimmed[:entry_start].rstrip(", \n")
        repaired_text = trimmed + "\n}\n"
        try:
            payload = json.loads(repaired_text)
            print(
                f"Warning: CIP cache at {cache_path} appeared truncated; "
                "ignored incomplete tail and loaded remaining entries."
            )
        except json.JSONDecodeError as repaired_exc:
            raise ValueError(
                f"Failed to parse CIP cache at {cache_path}: {exc}"
            ) from repaired_exc
    if not isinstance(payload, dict):
        return pd.DataFrame(
            columns=[
                "field_clean_pre",
                "field_clean",
                "cip_code",
                "cip_title",
                "cip_match_score",
                "cip_match_source",
                "cip_jaro_winkler",
            ]
        )
    rows: list[dict[str, Any]] = []
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        record = dict(value)
        canonical = record.get("field_clean")
        if not isinstance(canonical, str) or not canonical.strip():
            canonical = key
        record.setdefault("field_clean", canonical)
        record["field_clean_pre"] = record.get("field_clean_pre") or canonical
        rows.append(record)
    if not rows:
        return pd.DataFrame(
            columns=[
                "field_clean_pre",
                "field_clean",
                "cip_code",
                "cip_title",
                "cip_match_score",
                "cip_match_source",
                "cip_jaro_winkler",
            ]
        )
    cache_df = pd.DataFrame(rows)
    needed = [
        "field_clean_pre",
        "field_clean",
        "cip_code",
        "cip_title",
        "cip_match_score",
        "cip_match_source",
        "cip_jaro_winkler",
    ]
    for col in needed:
        if col not in cache_df.columns:
            cache_df[col] = pd.NA
    cache_df = cache_df[needed]
    cache_df = cache_df.dropna(subset=["field_clean_pre"]).drop_duplicates("field_clean_pre")
    return cache_df.reset_index(drop=True)


def _attach_cip_matches(
    con: ddb.DuckDBPyConnection,
    *,
    source_view: str,
    batch_size: int,
    digit_length: int,
    verbose: bool,
    cache_only: bool,
    cache_path: Optional[Path],
) -> str:
    """Match raw fields to CIP codes in batches and join back to the sample."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if cache_only:
        if cache_path is None:
            raise ValueError("cache_only=True requires a valid cache_path.")
        matches = _load_cip_cache_dataframe(cache_path)
        if matches.empty:
            print(
                f"Warning: CIP cache at {cache_path} is empty; no matches will be joined."
            )
        elif verbose:
            print(
                f"Joining {len(matches):,} cached CIP matches from {cache_path.name}."
            )
    else:
        distinct_fields = con.sql(
            f"""
            SELECT DISTINCT field_clean_pre
            FROM {source_view}
            WHERE field_clean_pre IS NOT NULL AND LENGTH(TRIM(field_clean_pre)) > 0
            ORDER BY field_clean_pre
            """
        ).df()

        if distinct_fields.empty:
            print(f"Warning: no raw fields found for CIP matching in {source_view}")
            return source_view

        if verbose:
            print(f"Matching {len(distinct_fields):,} unique fields to CIP codes...")

        matches = _batch_match_fields_to_cip(
            distinct_fields.copy(),
            batch_size=batch_size,
            digit_length=digit_length,
            cache_only=False,
            cache_path=cache_path,
        )
        matches.to_parquet(f"{root}/data/int/ihma_cip_matches.parquet", index=False)
    
    cip_view = "ihma_samp_field_cip"
    con.register(cip_view, matches)
    target_view = f"{source_view}_cip"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {target_view} AS
        SELECT
            src.*,
            cip.field_clean_pre,
            cip.field_clean,
            cip.cip_code,
            cip.cip_title,
            cip.cip_match_score,
            cip.cip_jaro_winkler
        FROM {source_view} AS src
        LEFT JOIN {cip_view} AS cip
        ON src.field_clean_pre = cip.field_clean_pre
        """
    )

    return target_view


def _batch_match_fields_to_cip(
    df: pd.DataFrame,
    *,
    batch_size: int,
    digit_length: int,
    max_workers: Optional[int] = None,
    cache_only: bool = False,
    cache_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Chunked wrapper over helpers.match_fields_to_cip to limit working-set size."""

    if df.empty:
        return pd.DataFrame(
            columns=["field_clean_pre", "field_clean", "cip_code", "cip_title", "cip_match_score", "cip_jaro_winkler"]
        )

    def _process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
        match_kwargs: dict[str, Union[bool, int, str, Path]] = {
            "digit_length": digit_length,
            "field_column": "field_clean_pre",
            "return_debug": False,
            "cache_only": cache_only,
        }
        if cache_path is not None:
            match_kwargs["cache_path"] = cache_path
        return match_fields_to_cip(
            chunk_df,
            **match_kwargs,
        )

    jobs: list[tuple[int, pd.DataFrame]] = []
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        chunk = df.iloc[start:end].copy()
        jobs.append((start, chunk))

    worker_count = max(1, max_workers or (os.cpu_count() or 1))
    parts: list[tuple[int, pd.DataFrame]] = []

    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(jobs), desc="CIP matching", unit="chunk", leave=False)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(_process_chunk, chunk): idx for idx, chunk in jobs}
        for future in as_completed(future_map):
            idx = future_map[future]
            parts.append((idx, future.result()))
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    parts.sort(key=lambda item: item[0])
    merged = pd.concat([part for _, part in parts], ignore_index=True)

    needed_cols = ["field_clean_pre", "field_clean", "cip_code", "cip_title", "cip_match_score", "cip_jaro_winkler"]
    for col in needed_cols:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[needed_cols]
    return merged


def _create_downstream_views(
    con: ddb.DuckDBPyConnection,
    *,
    samp_view: str,
    educ_view: str,
    geonames_view: Optional[str],
    group_positions_by_geoname: bool,
    verbose: bool,
    save_intermediate: bool,
    from_intermediate: bool,
    cache_dir: Path,
    origin_group_by: str,
    origin_group_by_use_cip: bool,
    us_group_by: str,
    us_group_by_use_cip: bool,
    force_resave: bool,
) -> dict[str, str]:
    """Reproduce the SQL transforms that ihma_explore previously owned."""

    origin_group_expr_base = (
        "COALESCE(CASE WHEN a.rsid IS NOT NULL THEN CAST(a.rsid AS VARCHAR) ELSE a.origin_cluster_institution_id END, a.origin_cluster_institution_id)"
        if origin_group_by == "origin_rsid"
        else "COALESCE(rg.geoname_id, a.origin_cluster_geo_city_id)"
    )
    origin_cip_expr = _cip_prefix_expr("a.cip_code")
    if origin_group_by_use_cip:
        origin_group_expr = (
            f"CONCAT(COALESCE(CAST(({origin_group_expr_base}) AS VARCHAR), 'unknown_origin'), "
            "'::cip::', "
            f"COALESCE({origin_cip_expr}, 'uncoded'))"
        )
    else:
        origin_group_expr = origin_group_expr_base

    us_group_expr_base = (
        "b.us_cluster_geo_city_id"
        if us_group_by == "us_cluster_geo_city_id"
        else "b.us_primary_ipeds_id"
    )
    us_cip_expr = _cip_prefix_expr("b.us_cip_code")
    if us_group_by_use_cip:
        us_group_expr = (
            f"CONCAT(COALESCE(CAST(({us_group_expr_base}) AS VARCHAR), 'unknown_us'), "
            "'::cip::', "
            f"COALESCE({us_cip_expr}, 'uncoded'))"
        )
    else:
        us_group_expr = us_group_expr_base

    if verbose:
        print("Constructing ihma_us_educ_clean, ihma_user, ihma_pos_clean, pos_clean_msa, and earn_clean views...")

    con.sql(f"CREATE OR REPLACE TEMP VIEW ihma_samp AS SELECT * FROM {samp_view}")

    con.sql(f"CREATE OR REPLACE TEMP VIEW ihma_educ AS SELECT * FROM {educ_view}")

    # only keep education records where the grad year is after the origin grad year (main US education database)
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW ihma_us_educ_clean AS
        SELECT a.user_id, CASE WHEN a.rsid IS NOT NULL THEN CAST(a.rsid AS VARCHAR) ELSE us_cluster_institution_id END AS us_rsid, a.university_raw AS us_university_raw, a.gradyr AS us_gradyr, degree_clean AS us_degree_clean, field_clean_pre AS us_field_clean_pre, a.cip_code AS us_cip_code, a.cip_title AS us_cip_title, a.cip_match_score AS us_cip_match_score, us_cluster_root, us_cluster_name, us_cluster_geo_city_name, us_cluster_geo_city_id, us_cluster_country_code, us_cluster_institution_id, us_cluster_ipeds_ids, us_primary_ipeds_id
        FROM {educ_view} AS a
        LEFT JOIN (
            SELECT gradyr AS origin_gradyr, user_id AS origin_user_id
            FROM {samp_view}
            WHERE gradyr IS NOT NULL
        ) AS b
        ON a.user_id = b.origin_user_id
        WHERE a.gradyr > b.origin_gradyr
        """
    )
    
    # build ihma_user view combining samp and educ info (at user level), excluding US origin universities
    if not _maybe_load_view(con, "ihma_user", cache_dir, from_intermediate):
        rsid_lookup_view = "rsid_geoname_lookup"
        if RSID_GEONAME_CROSSWALK_PATH.exists():
            escaped_crosswalk = str(RSID_GEONAME_CROSSWALK_PATH).replace("'", "''")
            con.sql(
                f"""
                CREATE OR REPLACE TEMP VIEW {rsid_lookup_view} AS
                SELECT
                    CAST(rsid AS VARCHAR) AS rsid,
                    CAST(geoname_id AS VARCHAR) AS geoname_id,
                    CAST(geo_name AS VARCHAR) AS geo_name,
                    CAST(campus_state AS VARCHAR) AS campus_state,
                    CAST(campus_city AS VARCHAR) AS campus_city
                FROM read_parquet('{escaped_crosswalk}')
                """
            )
        else:
            print(f"Warning: RSID→geoname crosswalk missing at {RSID_GEONAME_CROSSWALK_PATH}; using cluster geonames only.")
            con.sql(
                f"""
                CREATE OR REPLACE TEMP VIEW {rsid_lookup_view} AS
                SELECT CAST(NULL AS VARCHAR) AS rsid, CAST(NULL AS VARCHAR) AS geoname_id WHERE FALSE
                """
            )

        con.sql(
            f"""
            CREATE OR REPLACE TEMP TABLE ihma_user_base AS
            SELECT
                a.user_id,
                CASE WHEN a.rsid IS NOT NULL THEN CAST(a.rsid AS VARCHAR) ELSE a.origin_cluster_institution_id END AS origin_rsid,
                {origin_group_expr} AS origin_group_id,
                COALESCE(rg.geoname_id, a.origin_cluster_geo_city_id) AS origin_geo_city_id_final,
                rg.geoname_id AS origin_rsid_geoname_id,
                CASE
                    WHEN '{origin_group_by}' = 'origin_rsid' THEN COALESCE(a.origin_cluster_name, a.origin_cluster_root)
                    ELSE COALESCE(rg.geo_name, a.origin_cluster_geo_city_name)
                END AS origin_group_name,
                CASE
                    WHEN '{origin_group_by}' = 'origin_rsid' THEN COALESCE(a.origin_cluster_country_code, 'US')
                    ELSE COALESCE(a.origin_cluster_country_code, 'US')
                END AS origin_group_country,
                a.gradyr AS origin_gradyr,
                a.university_raw AS origin_university_raw,
                a.degree_clean AS origin_degree_clean,
                a.field_clean_pre AS origin_field_clean_pre,
                a.cip_code AS origin_cip_code,
                a.cip_title AS origin_cip_title,
                a.cip_match_score AS origin_cip_match_score,
                a.origin_cluster_root,
                a.origin_cluster_name,
                a.origin_cluster_geo_city_name,
                a.origin_cluster_geo_city_id,
                a.origin_cluster_country_code,
                a.origin_cluster_institution_id,
                a.origin_cluster_ipeds_ids,
                b.us_rsid,
                b.us_university_raw,
                b.us_gradyr,
                b.us_degree_clean,
                b.us_field_clean_pre,
                b.us_cip_code,
                b.us_cip_title,
                b.us_cip_match_score,
                b.us_cluster_root,
                b.us_cluster_geo_city_name,
                b.us_cluster_geo_city_id,
                b.us_cluster_country_code,
                b.us_cluster_institution_id,
                b.us_cluster_ipeds_ids,
                b.us_primary_ipeds_id,
                {us_group_expr} AS us_group_id,
                CASE
                    WHEN '{us_group_by}' = 'us_cluster_geo_city_id' THEN b.us_cluster_geo_city_name
                    ELSE b.us_cluster_name
                END AS us_group_name_base,
                CASE
                    WHEN '{us_group_by}' = 'us_cluster_geo_city_id' THEN COALESCE(b.us_cluster_country_code, 'US')
                    ELSE COALESCE(b.us_cluster_country_code, 'US')
                END AS us_group_country_base
            FROM (
                SELECT
                    user_id,
                    rsid,
                    gradyr,
                    university_raw,
                    degree_clean,
                    field_clean_pre,
                    cip_code,
                    cip_title,
                    cip_match_score,
                    origin_cluster_root,
                    origin_cluster_name,
                    origin_cluster_geo_city_name,
                    origin_cluster_geo_city_id,
                    origin_cluster_country_code,
                    origin_cluster_institution_id,
                    origin_cluster_ipeds_ids
                FROM {samp_view}
            ) AS a
            LEFT JOIN ihma_us_educ_clean AS b
                ON a.user_id = b.user_id
            LEFT JOIN {rsid_lookup_view} AS rg
                ON CAST(a.rsid AS VARCHAR) = rg.rsid
            WHERE a.gradyr IS NOT NULL
            """
        )
        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW ihma_user AS
            SELECT
                base.* EXCLUDE (us_group_name_base, us_group_country_base),
                CASE
                    WHEN '{us_group_by}' = 'us_cluster_geo_city_id' THEN base.us_cluster_geo_city_name
                    ELSE base.us_group_name_base
                END AS us_group_name,
                CASE
                    WHEN '{us_group_by}' = 'us_cluster_geo_city_id' THEN COALESCE(base.us_cluster_country_code, 'US')
                    ELSE base.us_group_country_base
                END AS us_group_country,
                base.us_group_id AS us_group_id
            FROM ihma_user_base AS base
            """
        )
        _maybe_save_view(con, "ihma_user", cache_dir, save_intermediate, force=force_resave)

    # clean position data: parse dates, filter invalid records, extract years
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW ihma_pos_clean_base AS
        WITH norm AS (
            SELECT
                *,
                CAST(startdate AS DATE) AS sd,
                CAST(COALESCE(enddate, '2025-12-01') AS DATE) AS ed
            FROM ihma_pos_raw
            WHERE startdate IS NOT NULL
              AND (enddate IS NULL OR startdate <= enddate)
        ),
        windowed AS (
            SELECT
                *,
                CAST(GREATEST(EXTRACT(YEAR FROM sd), {POS_START_YEAR}) AS INTEGER) AS start_y,
                CAST(LEAST(EXTRACT(YEAR FROM ed), {POS_END_YEAR}) AS INTEGER) AS end_y
            FROM norm
        )
        SELECT *
        FROM windowed
        WHERE start_y IS NOT NULL
          AND end_y IS NOT NULL
          AND start_y <= end_y
        """
    )

    if not _maybe_load_view(con, "ihma_pos_clean", cache_dir, from_intermediate):
        if geonames_view is not None:
            con.sql(
                f"""
                CREATE OR REPLACE TEMP VIEW ihma_pos_clean AS
                SELECT
                    base.*,
                    geo.pos_geo_geoname_id,
                    geo.pos_geo_city_name,
                    geo.pos_geo_country_name
                FROM ihma_pos_clean_base AS base
                LEFT JOIN {geonames_view} AS geo
                    ON base.metro_area = geo.metro_raw
                AND base.country = geo.country_raw
                """
            )
        else:
            con.sql(
                """
                CREATE OR REPLACE TEMP VIEW ihma_pos_clean AS
                SELECT
                    base.*,
                    CAST(NULL AS VARCHAR) AS pos_geo_geoname_id,
                    CAST(NULL AS VARCHAR) AS pos_geo_city_name,
                    CAST(NULL AS VARCHAR) AS pos_geo_country_name
                FROM ihma_pos_clean_base AS base
                """
            )
        _maybe_save_view(con, "ihma_pos_clean", cache_dir, save_intermediate, force=force_resave)

    use_geoname_grouping = geonames_view is not None and group_positions_by_geoname
    if not use_geoname_grouping:
        raise RuntimeError(
            "Geoname grouping is required for pos_clean_msa; ensure position_geography_lookup is available."
        )

    pos_clean_sql = f"""
        CREATE OR REPLACE TEMP VIEW pos_clean_msa AS
        WITH expanded AS (
            SELECT
                n.user_id,
                n.pos_geo_geoname_id,
                n.pos_geo_city_name,
                n.pos_geo_country_name,
                n.total_compensation,
                y AS year
            FROM ihma_pos_clean AS n,
            generate_series(start_y, end_y) AS t(y)
            WHERE n.pos_geo_geoname_id IS NOT NULL
              AND n.start_y IS NOT NULL
              AND n.end_y IS NOT NULL
              AND n.start_y <= n.end_y
        ),
        grouped AS (
            SELECT
                user_id,
                pos_geo_geoname_id,
                pos_geo_city_name,
                pos_geo_country_name,
                year,
                SUM(total_compensation) AS total_compensation,
                COUNT(*) AS n_records
            FROM expanded
            GROUP BY user_id, pos_geo_geoname_id, pos_geo_city_name, pos_geo_country_name, year
        )
        SELECT
            g.user_id,
            b.origin_group_id,
            b.origin_group_name,
            b.origin_group_country,
            g.pos_geo_geoname_id,
            g.pos_geo_city_name,
            g.pos_geo_country_name,
            g.year,
            g.total_compensation,
            g.n_records,
            1 AS worked_in_geo,
            b.origin_gradyr,
            b.origin_rsid
        FROM grouped AS g
        JOIN ihma_user AS b
            ON g.user_id = b.user_id
        WHERE b.origin_gradyr <= g.year
    """
    if not _maybe_load_view(con, "pos_clean_msa", cache_dir, from_intermediate):
        con.sql(pos_clean_sql)
        _maybe_save_view(con, "pos_clean_msa", cache_dir, save_intermediate, force=force_resave)

    # expand position data to get yearly earnings (at years since graduation x user level)
    if not _maybe_load_view(con, "earn_clean", cache_dir, from_intermediate):
        con.sql(
        """
        CREATE OR REPLACE TEMP VIEW earn_clean AS
        WITH expanded AS (
            SELECT
                y.user_id,
                y.total_compensation,
                gs.y AS year
            FROM ihma_pos_clean y,
            generate_series(y.start_y, y.end_y) AS gs(y)
            WHERE y.start_y <= y.end_y
              AND y.start_y IS NOT NULL
              AND y.end_y IS NOT NULL
        ),
        comp_by_year AS (
            SELECT
                user_id,
                year,
                MAX(total_compensation) AS max_comp,
                COUNT(*) AS n_jobs
            FROM expanded
            GROUP BY user_id, year
        ),
        year_bounds AS (
            SELECT
                user_id,
                MIN(start_y) AS min_year,
                MAX(end_y) AS max_year
            FROM ihma_pos_clean
            GROUP BY user_id
        ),
        user_years AS (
            SELECT
                b.user_id,
                gs.y AS year
            FROM year_bounds b,
            generate_series(b.min_year, b.max_year) AS gs(y)
            WHERE b.min_year <= b.max_year
              AND b.min_year IS NOT NULL
              AND b.max_year IS NOT NULL
        )
        SELECT
            u.user_id,
            b.origin_gradyr,
            b.origin_group_id,
            b.origin_group_name,
            b.origin_group_country,
            CASE WHEN b.origin_rsid IS NOT NULL THEN CAST(b.origin_rsid AS VARCHAR) ELSE b.origin_cluster_institution_id END AS origin_rsid,
            b.origin_cluster_institution_id,
            u.year,
            COALESCE(c.max_comp, 0) AS total_compensation,
            c.n_jobs,
            u.year - b.origin_gradyr AS t
        FROM user_years u
        LEFT JOIN comp_by_year c USING (user_id, year)
        JOIN ihma_user AS b
            ON u.user_id = b.user_id
        ORDER BY u.user_id, year
        """
        )
        _maybe_save_view(con, "earn_clean", cache_dir, save_intermediate, force=force_resave)

    return {
        "ihma_samp": "ihma_samp",
        "ihma_educ": "ihma_educ",
        "ihma_us_educ_clean": "ihma_us_educ_clean",
        "ihma_user": "ihma_user",
        "ihma_pos": "ihma_pos_raw",
        "ihma_pos_clean": "ihma_pos_clean",
        "pos_clean_msa": "pos_clean_msa",
        "earn_clean": "earn_clean",
        "positions_msa": "positions_msa",
        "ipeds_ma": "ipeds_ma",
        "ipeds_ma_nodist": "ipeds_ma_nodist",
        "ipeds_geoname": "ipeds_geoname",
    }


def _create_analysis_views(
    con: ddb.DuckDBPyConnection,
    *,
    verbose: bool,
    save_intermediate: bool,
    cache_dir: Path,
    from_intermediate: bool,
    origin_group_by: str,
    us_group_by: str,
    force_resave: bool,
) -> dict[str, str]:
    """Materialize the shift-share helper views used in ihma_explore."""

    if verbose:
        print("Constructing geoname share, instrument, and regression helper views...")

    pos_share_view = "ihma_pos_geo_shares"
    pos_share_main = f"{pos_share_view}_{CBSA_SHARE_TARGET_YEAR}"
    pos_loaded = _maybe_load_view(con, pos_share_view, cache_dir, from_intermediate)
    pos_main_loaded = _maybe_load_view(con, pos_share_main, cache_dir, from_intermediate)
    if pos_loaded and pos_main_loaded:
        cbsa_shares = pos_share_view
        cbsa_shares_main = pos_share_main
    else:
        cbsa_shares, cbsa_shares_main = _create_cbsa_share_view(
            con,
            source_view="pos_clean_msa",
            rsid_column="origin_group_id",
            location_column="pos_geo_geoname_id",
            year_column="year",
            user_id_column="user_id",
            view_prefix="ihma_pos",
            rsid_name_column="origin_group_name",
            rsid_country_column="origin_group_country",
            location_name_column="pos_geo_city_name",
            location_country_column="pos_geo_country_name",
        )
        _maybe_save_view(con, cbsa_shares, cache_dir, save_intermediate, force=force_resave)
        _maybe_save_view(con, cbsa_shares_main, cache_dir, save_intermediate, force=force_resave)

    educ_share_view = "ihma_educ_geo_shares"
    educ_share_main = f"{educ_share_view}_{CBSA_SHARE_TARGET_YEAR}"
    educ_loaded = _maybe_load_view(con, educ_share_view, cache_dir, from_intermediate)
    educ_main_loaded = _maybe_load_view(con, educ_share_main, cache_dir, from_intermediate)
    if educ_loaded and educ_main_loaded:
        educ_shares = educ_share_view
        educ_shares_main = educ_share_main
    else:
        educ_shares, educ_shares_main = _create_cbsa_share_view(
            con,
            source_view="ihma_user",
            rsid_column="origin_group_id",
            location_column="us_group_id",
            year_column="us_gradyr",
            user_id_column="user_id",
            view_prefix="ihma_educ",
            rsid_name_column="origin_group_name",
            rsid_country_column="origin_group_country",
            location_name_column="us_group_name",
            location_country_column="us_group_country",
        )
        _maybe_save_view(con, educ_shares, cache_dir, save_intermediate, force=force_resave)
        _maybe_save_view(con, educ_shares_main, cache_dir, save_intermediate, force=force_resave)

    if us_group_by == "us_cluster_geo_city_id":
        selected_share_main = cbsa_shares_main
        growth_view_name = "ipeds_geo_growth"
        if not _maybe_load_view(con, growth_view_name, cache_dir, from_intermediate):
            ipeds_growth = _create_ipeds_growth_view(con)
            _maybe_save_view(con, growth_view_name, cache_dir, save_intermediate, force=force_resave)
        else:
            ipeds_growth = growth_view_name
    else:
        selected_share_main = educ_shares_main
        growth_view_name = "ipeds_unit_growth"
        if not _maybe_load_view(con, growth_view_name, cache_dir, from_intermediate):
            ipeds_growth = _create_ipeds_unit_growth_view(con)
            _maybe_save_view(con, growth_view_name, cache_dir, save_intermediate, force=force_resave)
        else:
            ipeds_growth = growth_view_name

    instrument_components = "ihma_instrument_components"
    instrument_panel = "ihma_instrument_panel"
    instrument_loaded = _maybe_load_view(con, instrument_components, cache_dir, from_intermediate) and _maybe_load_view(
        con, instrument_panel, cache_dir, from_intermediate
    )
    if not instrument_loaded:
        instrument_components, instrument_panel = _create_instrument_views(
            con,
            share_main_view=selected_share_main,
            ipeds_growth_view=ipeds_growth,
        )
        _maybe_save_view(con, instrument_components, cache_dir, save_intermediate, force=force_resave)
        _maybe_save_view(con, instrument_panel, cache_dir, save_intermediate, force=force_resave)

    dep_view, indep_view = _create_dep_indep_views(
        con,
        source_view = "ihma_user",
        rsid_column = "origin_group_id",
        grad_year_column = "origin_gradyr",
        us_group_column = "us_group_id")
    _maybe_save_view(con, dep_view, cache_dir, save_intermediate, force=force_resave)
    _maybe_save_view(con, indep_view, cache_dir, save_intermediate, force=force_resave)

    return {
        "cbsa_shares": cbsa_shares,
        "cbsa_shares_main": cbsa_shares_main,
        "educ_cbsa_shares": educ_shares,
        "educ_cbsa_shares_main": educ_shares_main,
        "ipeds_cbsa_growth": ipeds_growth,
        "instrument_components": instrument_components,
        "instrument_panel": instrument_panel,
        "dep_constr": dep_view,
        "indep_constr": indep_view,
    }


def _create_cbsa_share_view(
    con: ddb.DuckDBPyConnection,
    *,
    source_view: str,
    rsid_column: str,
    location_column: str,
    year_column: str,
    user_id_column: str,
    view_prefix: str,
    rsid_name_column: Optional[str] = None,
    rsid_country_column: Optional[str] = None,
    location_name_column: Optional[str] = None,
    location_country_column: Optional[str] = None,
) -> tuple[str, str]:
    """Construct rolling geoname share views for position or education data."""

    share_view = f"{view_prefix}_geo_shares"
    main_view = f"{share_view}_{CBSA_SHARE_TARGET_YEAR}"

    rsid_name_expr = rsid_name_column if rsid_name_column else "NULL"
    rsid_country_expr = rsid_country_column if rsid_country_column else "NULL"
    loc_name_expr = location_name_column if location_name_column else "NULL"
    loc_country_expr = location_country_column if location_country_column else "NULL"

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {share_view} AS
        WITH typed AS (
            SELECT
                {rsid_column} AS rsid,
                {location_column} AS geo_city_id,
                CAST({year_column} AS INTEGER) AS year,
                {user_id_column} AS user_id,
                {rsid_name_expr} AS rsid_name,
                {rsid_country_expr} AS rsid_country,
                {loc_name_expr} AS location_name,
                {loc_country_expr} AS location_country
            FROM {source_view}
            WHERE {location_column} IS NOT NULL
              AND {year_column} IS NOT NULL
        ),
        base AS (
            SELECT
                rsid,
                geo_city_id,
                year,
                MAX(rsid_name) AS rsid_name,
                MAX(rsid_country) AS rsid_country,
                MAX(location_name) AS location_name,
                MAX(location_country) AS location_country,
                COUNT(DISTINCT user_id) AS n_cbsa
            FROM typed
            WHERE year BETWEEN {CBSA_SHARE_START_YEAR} AND {CBSA_SHARE_END_YEAR}
            GROUP BY rsid, geo_city_id, year
        ),
        bounds AS (
            SELECT
                rsid,
                geo_city_id,
                MIN(year) AS min_year,
                MAX(year) AS max_year
            FROM base
            GROUP BY rsid, geo_city_id
        ),
        expanded AS (
            SELECT
                b.rsid,
                b.geo_city_id,
                gs.year
            FROM bounds b,
            LATERAL generate_series(b.min_year, b.max_year) AS gs(year)
        ),
        combined AS (
            SELECT
                e.rsid,
                e.geo_city_id,
                e.year,
                COALESCE(b.n_cbsa, 0) AS n_cbsa,
                b.rsid_name,
                b.rsid_country,
                b.location_name,
                b.location_country
            FROM expanded e
            LEFT JOIN base b
              ON e.rsid = b.rsid
             AND e.geo_city_id = b.geo_city_id
             AND e.year = b.year
        ),
        totals AS (
            SELECT
                rsid,
                geo_city_id,
                year,
                n_cbsa,
                SUM(n_cbsa) OVER (PARTITION BY rsid, year) AS n_tot,
                rsid_name,
                rsid_country,
                location_name,
                location_country
            FROM combined
        ),
        windowed AS (
            SELECT
                *,
                SUM(n_cbsa) OVER (
                    PARTITION BY rsid, geo_city_id
                    ORDER BY year
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS num_5yr,
                SUM(n_tot) OVER (
                    PARTITION BY rsid, geo_city_id
                    ORDER BY year
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS num_tot_5yr
            FROM totals
        )
        SELECT
            rsid AS i,
            geo_city_id AS k,
            rsid_name,
            rsid_country,
            location_name,
            location_country,
            year,
            n_cbsa,
            n_tot,
            CASE
                WHEN num_tot_5yr = 0 THEN NULL
                ELSE num_5yr::DOUBLE / NULLIF(num_tot_5yr, 0)
            END AS rolling_share_geo
        FROM windowed
        ORDER BY rsid, geo_city_id, year
        """
    )

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {main_view} AS
        SELECT
            i,
            k,
            rsid_name,
            rsid_country,
            location_name,
            location_country,
            rolling_share_geo AS share_geo_2001_2005
        FROM {share_view}
        WHERE year = {CBSA_SHARE_TARGET_YEAR}
        """
    )

    return share_view, main_view


def _create_ipeds_growth_view(con: ddb.DuckDBPyConnection) -> str:
    """Build the IPEDS growth view keyed directly by geoname identifiers."""

    view_name = "ipeds_geo_growth"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        WITH joined AS (
            SELECT
                CAST(g.geoname_id AS VARCHAR) AS geoname_id,
                CAST(m.year AS INTEGER) AS year,
                SUM(m.cnralt) AS cnralt
            FROM ipeds_ma AS m
            LEFT JOIN ipeds_geoname AS g
                ON m.unitid = g.unitid
            WHERE g.geoname_id IS NOT NULL
            GROUP BY CAST(g.geoname_id AS VARCHAR), CAST(m.year AS INTEGER)
        ),
        bounds AS (
            SELECT
                geoname_id,
                MIN(year) AS min_year,
                MAX(year) AS max_year
            FROM joined
            GROUP BY geoname_id
        ),
        expanded AS (
            SELECT
                b.geoname_id,
                gs.year
            FROM bounds b,
            LATERAL generate_series(b.min_year, b.max_year) AS gs(year)
        ),
        filled AS (
            SELECT
                e.geoname_id,
                e.year,
                COALESCE(j.cnralt, 0) AS cnralt
            FROM expanded e
            LEFT JOIN joined j
                ON e.geoname_id = j.geoname_id
                AND e.year = j.year
        ),
        diffed AS (
            SELECT
                geoname_id,
                year,
                cnralt - LAG(cnralt) OVER (PARTITION BY geoname_id ORDER BY year) AS g_kt
            FROM filled
        )
        SELECT
            geoname_id AS k,
            year - 2 AS t,
            g_kt
        FROM diffed
        WHERE g_kt IS NOT NULL
        """
    )
    return view_name


def _create_ipeds_unit_growth_view(con: ddb.DuckDBPyConnection) -> str:
    """Build the IPEDS growth view keyed by UNITID (used when grouping by IPEDS)."""

    view_name = "ipeds_unit_growth"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        WITH joined AS (
            SELECT
                CAST(unitid AS VARCHAR) AS unitid,
                CAST(year AS INTEGER) AS year,
                SUM(cnralt) AS cnralt
            FROM ipeds_ma
            WHERE unitid IS NOT NULL
            GROUP BY CAST(unitid AS VARCHAR), CAST(year AS INTEGER)
        ),
        bounds AS (
            SELECT
                unitid,
                MIN(year) AS min_year,
                MAX(year) AS max_year
            FROM joined
            GROUP BY unitid
        ),
        expanded AS (
            SELECT
                b.unitid,
                gs.year
            FROM bounds b,
            LATERAL generate_series(b.min_year, b.max_year) AS gs(year)
        ),
        filled AS (
            SELECT
                e.unitid,
                e.year,
                COALESCE(j.cnralt, 0) AS cnralt
            FROM expanded e
            LEFT JOIN joined j
                ON e.unitid = j.unitid
               AND e.year = j.year
        ),
        diffed AS (
            SELECT
                unitid,
                year,
                cnralt - LAG(cnralt) OVER (PARTITION BY unitid ORDER BY year) AS g_kt
            FROM filled
        )
        SELECT
            unitid AS k,
            year - 2 AS t,
            g_kt
        FROM diffed
        WHERE g_kt IS NOT NULL
        """
    )
    return view_name


def _create_instrument_views(
    con: ddb.DuckDBPyConnection,
    *,
    share_main_view: str,
    ipeds_growth_view: str,
) -> tuple[str, str]:
    """Combine CBSA shares with IPEDS growth to form the instrument."""

    components_view = "ihma_instrument_components"
    panel_view = "ihma_instrument_panel"

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {components_view} AS
        SELECT
            share.i,
            share.k,
            growth.t,
            share.share_geo_2001_2005,
            growth.g_kt,
            share.share_geo_2001_2005 * growth.g_kt AS z_it
        FROM {share_main_view} AS share
        JOIN {ipeds_growth_view} AS growth
            ON share.k = growth.k
        WHERE growth.g_kt IS NOT NULL
        """
    )

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {panel_view} AS
        SELECT
            i,
            t,
            SUM(z_it) AS z_it
        FROM {components_view}
        GROUP BY i, t
        """
    )

    return components_view, panel_view


def _create_dep_indep_views(
    con: ddb.DuckDBPyConnection,
    *,
    source_view: str,
    rsid_column: str,
    us_group_column: str,
    grad_year_column: str,
    max_relative_years: int = MAX_RELATIVE_YEARS,
) -> tuple[str, str]:
    """Create the dependent and independent variable summary views."""

    indep_view = "ihma_indep_constr"
    dep_view = "ihma_dep_constr"

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {indep_view} AS
        SELECT
            {rsid_column} AS i,
            {grad_year_column} AS t,
            MAX(origin_group_name) AS origin_group_name,
            COUNT({us_group_column}) AS x_it,
            COUNT(*) AS n
        FROM {source_view}
        GROUP BY {rsid_column}, {grad_year_column}
        """
    )

    total_comp_expr = ",\n                ".join(
        f"MAX(CASE WHEN e.t = {offset} THEN e.total_compensation END) AS total_comp{offset}"
        for offset in range(1, max_relative_years + 1)
    )
    avg_expr = ",\n            ".join(
        f"AVG(total_comp{offset}) AS y_it{offset}" for offset in range(1, max_relative_years + 1)
    )

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {dep_view} AS
        WITH earnings AS (
            SELECT
                user_id,
                t,
                total_compensation
            FROM earn_clean
            WHERE t BETWEEN 1 AND {max_relative_years}
        ),
        user_comp AS (
            SELECT
                u.user_id,
                {grad_year_column} AS gradyr,
                {rsid_column} AS rsid,
                MAX(u.origin_group_name) AS origin_group_name,
                {total_comp_expr}
            FROM {source_view} AS u
            LEFT JOIN earnings AS e
                ON u.user_id = e.user_id
            GROUP BY u.user_id, {grad_year_column}, {rsid_column}
        )
        SELECT
            gradyr AS t,
            rsid AS i,
            MAX(origin_group_name) AS origin_group_name,
            {avg_expr}
        FROM user_comp
        GROUP BY gradyr, rsid
        """
    )

    return dep_view, indep_view


# connection = ddb.connect()
# verbose = True
# connection.create_function(
#     "get_std_country",
#     lambda value: _helpers_get_std_country(value, _COUNTRY_CODE_DICT),
#     ["VARCHAR"],
#     "VARCHAR",
# )
# cluster_mapping = _read_cluster_mapping(DEFAULT_CLUSTER_MAPPING_PATH)
# _create_base_views(connection)
# position_geo_view = _register_position_geography_view(connection, DEFAULT_POSITION_GEOGRAPHY_PATH)
# samp_view = _attach_cluster_metadata(
#         connection,
#         source_view="ihma_samp_base",
#         name_column="institution_name_for_cluster",
#         prefix="origin",
#         cluster_mapping=cluster_mapping,
#         verbose=True,
# )
# educ_view = _attach_cluster_metadata(
#     connection,
#     source_view="ihma_educ_base",
#     name_column="institution_name_for_cluster",
#     prefix="us",
#     cluster_mapping=cluster_mapping,
#     verbose=verbose,
# )
# base_tables = _create_downstream_views(
#     connection,
#     samp_view=samp_view,
#     educ_view=educ_view,
#     verbose=verbose,
#     geonames_view=position_geo_view
# )
# con = connection
# con.sql("SELECT * FROM position_geography_lookup").df()
# y=connection.sql("SELECT DISTINCT field_clean_pre FROM ihma_samp_base_origin_clustered").fetch_df_chunk(10)
# z = _batch_match_fields_to_cip(y, batch_size = 1000, digit_length = 6)
