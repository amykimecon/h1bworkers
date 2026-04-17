"""Standalone WRDS extract for relabel-school education and position histories."""

from __future__ import annotations

import importlib.util
import numbers
import os
import sys
import time
from pathlib import Path

import duckdb as ddb
import numpy as np
import pandas as pd

# Ensure progress logs flush immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import relabels_revelio.relabel_indiv_model_config as cfg


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_WRDS_HELPER_DIR = _REPO_ROOT / "f1_indiv_merge" / "02_rev_import"
_wrds_query = _load_module_from_path("_relabel_indiv_model_wrds_query", _WRDS_HELPER_DIR / "wrds_query.py")
_wrds_common = _load_module_from_path("_relabel_indiv_model_wrds_common", _WRDS_HELPER_DIR / "common.py")

get_wrds_connection = _wrds_query.get_wrds_connection
chunk_values = _wrds_common.chunk_values
clean_institution_text = _wrds_common.clean_institution_text
escape_sql_literal = _wrds_common.escape_sql_literal

print(f"[relabel_indiv_model] Using config: {cfg.ACTIVE_CONFIG_PATH}")
print(
    "[relabel_indiv_model] "
    f"run_tag={cfg.RUN_TAG} "
    f"testing={cfg.TESTING_ENABLED} "
    f"control_only={cfg.BUILD_CONTROL_ONLY} "
    f"reuse_existing_candidate_parquets={cfg.BUILD_REUSE_EXISTING_CANDIDATE_PARQUETS}"
)

EVENT_SCHOOLS_COLUMNS = [
    "unitid",
    "relabel_year",
    "relabel_type",
    "school_name",
    "school_name_clean",
]

RSID_CANDIDATE_COLUMNS = [
    "unitid",
    "school_name",
    "school_name_clean",
    "university_name",
    "university_name_clean",
    "rsid",
    "candidate_count",
    "name_match_score",
    "candidate_rank",
]

SELECTED_RSID_COLUMNS = RSID_CANDIDATE_COLUMNS.copy()

MATCHED_EDUCATION_COLUMNS = [
    "unitid",
    "school_name",
    "school_name_clean",
    "relabel_year",
    "relabel_type",
    "event_rsid",
    "event_rsid_university_name",
    "rsid_candidate_count",
    "rsid_name_match_score",
    "user_id",
    "fullname",
    "rsid",
    "university_name",
    "education_number",
    "ed_startdate",
    "ed_enddate",
    "ed_end_year",
    "degree",
    "field",
    "university_country",
    "university_location",
    "university_raw",
    "degree_raw",
    "field_raw",
    "description",
    "exclude_immediate_same_inst_phd_after_master_ind",
]

MATCHED_POSITIONS_COLUMNS = [
    "user_id",
    "position_id",
    "position_number",
    "rcid",
    "country",
    "startdate",
    "enddate",
    "salary",
    "total_compensation",
    "company_raw",
    "location_raw",
    "title_raw",
    "description",
]

IMMEDIATE_PHD_WINDOW_DAYS = 365


def _elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


def _normalize_school_name(value: object) -> str:
    cleaned = clean_institution_text(None if pd.isna(value) else str(value))
    return cleaned or ""


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _sql_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "NULL"
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))
        return repr(as_float)
    return f"'{_sql_quote(str(value))}'"


def _build_values_sql(rows: list[tuple[object, ...]]) -> str:
    if not rows:
        raise ValueError("Cannot build VALUES SQL from an empty row set.")
    return ",\n".join("(" + ", ".join(_sql_value(value) for value in row) + ")" for row in rows)


def _normalize_sql_expr(col: str) -> str:
    return f"""
    TRIM(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    LOWER(COALESCE({col}::TEXT, '')),
                    E'[&+]',
                    ' and ',
                    'g'
                ),
                E'[^a-z0-9\\s]',
                ' ',
                'g'
            ),
            E'\\s+',
            ' ',
            'g'
        )
    )
    """


def _prepare_output_path(path_str: str) -> Path:
    out_path = Path(path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not cfg.BUILD_OVERWRITE:
        raise FileExistsError(f"Output exists and overwrite=false: {out_path}")
    return out_path


def _write_parquet(df: pd.DataFrame, path_str: str, columns: list[str]) -> Path:
    out_path = _prepare_output_path(path_str)
    out_df = df.copy()
    for col in columns:
        if col not in out_df.columns:
            out_df[col] = pd.NA
    out_df = out_df.loc[:, columns]
    out_df.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}")
    return out_path


def _load_parquet_with_columns(path_str: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path_str).copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df.loc[:, columns]


def _coerce_rsid_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["unitid", "rsid", "candidate_count", "name_match_score", "candidate_rank"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def _wrds_sql_to_df(db, sql: str, label: str) -> pd.DataFrame:
    t0 = time.time()
    # psycopg2 treats bare '%' as parameter syntax even for driver-level SQL.
    safe_sql = sql.replace("%", "%%")
    result = db.connection.exec_driver_sql(safe_sql)
    rows = result.fetchall()
    cols = list(result.keys())
    df = pd.DataFrame(rows, columns=cols)
    print(f"  {label}: {len(df):,} rows in {_elapsed(t0)}")
    return df


def _build_school_sample_query(search_name: str) -> str:
    search_name_sql = _sql_quote(search_name.strip().lower())
    normalize_name_expr = _normalize_sql_expr("a.university_name")
    normalize_raw_expr = _normalize_sql_expr("r.university_raw")
    return f"""
        SELECT
            a.university_name,
            {normalize_name_expr} AS university_name_clean,
            r.university_raw,
            {normalize_raw_expr} AS university_raw_clean,
            CAST(a.rsid AS BIGINT) AS rsid,
            CAST(a.user_id AS BIGINT) AS user_id,
            CAST(a.education_number AS BIGINT) AS education_number
        FROM revelio.individual_user_education AS a
        JOIN revelio.individual_user_education_raw AS r
          ON a.user_id = r.user_id
         AND a.education_number = r.education_number
        WHERE r.university_raw IS NOT NULL
          AND a.rsid IS NOT NULL
          AND POSITION('{search_name_sql}' IN LOWER(r.university_raw)) > 0
        ORDER BY a.user_id, a.education_number, a.rsid, r.university_raw, a.university_name
        LIMIT {cfg.BUILD_RSID_LOOKUP_LIMIT}
    """


def _build_school_exact_query(search_name: str) -> str:
    search_name_sql = _sql_quote(search_name.strip().lower())
    search_name_clean_sql = _sql_quote(_normalize_school_name(search_name))
    normalize_name_expr = _normalize_sql_expr("a.university_name")
    normalize_raw_expr = _normalize_sql_expr("r.university_raw")
    return f"""
        SELECT
            a.university_name,
            {normalize_name_expr} AS university_name_clean,
            r.university_raw,
            {normalize_raw_expr} AS university_raw_clean,
            CAST(a.rsid AS BIGINT) AS rsid,
            CAST(a.user_id AS BIGINT) AS user_id,
            CAST(a.education_number AS BIGINT) AS education_number
        FROM revelio.individual_user_education AS a
        JOIN revelio.individual_user_education_raw AS r
          ON a.user_id = r.user_id
         AND a.education_number = r.education_number
        WHERE r.university_raw IS NOT NULL
          AND a.rsid IS NOT NULL
          AND (
                LOWER(BTRIM(r.university_raw)) = '{search_name_sql}'
             OR {normalize_raw_expr} = '{search_name_clean_sql}'
          )
        ORDER BY a.user_id, a.education_number, a.rsid, r.university_raw, a.university_name
        LIMIT {cfg.BUILD_RSID_LOOKUP_LIMIT}
    """


def _apply_testing_filter(event_schools: pd.DataFrame) -> pd.DataFrame:
    if not cfg.TESTING_ENABLED:
        return event_schools

    unique_schools = (
        event_schools.loc[:, ["unitid", "school_name"]]
        .drop_duplicates()
        .sort_values(["school_name", "unitid"])
        .reset_index(drop=True)
    )
    if unique_schools.empty:
        return event_schools

    if cfg.TESTING_SCHOOL_NAMES:
        want = {name.strip().lower() for name in cfg.TESTING_SCHOOL_NAMES}
        keep = unique_schools.loc[
            unique_schools["school_name"].str.lower().isin(want),
            "unitid",
        ].astype("int64").tolist()
        if not keep:
            raise ValueError(
                "Testing school filter matched no event schools. "
                f"Requested={cfg.TESTING_SCHOOL_NAMES}"
            )
        print(f"[test] Filtering to schools: {sorted(unique_schools.loc[unique_schools['unitid'].isin(keep), 'school_name'].tolist())}")
    else:
        rng = np.random.default_rng(cfg.TESTING_RANDOM_SEED)
        n_sample = min(cfg.TESTING_SAMPLE_N_SCHOOLS, len(unique_schools))
        chosen_idx = rng.choice(len(unique_schools), size=n_sample, replace=False)
        keep = (
            unique_schools.iloc[sorted(chosen_idx)]["unitid"]
            .astype("int64")
            .tolist()
        )
        sampled_names = unique_schools.loc[unique_schools["unitid"].isin(keep), "school_name"].tolist()
        print(f"[test] Sampled {n_sample} schools: {sampled_names}")

    return (
        event_schools.loc[event_schools["unitid"].isin(keep)]
        .sort_values(["relabel_year", "school_name", "relabel_type"])
        .reset_index(drop=True)
    )


def load_event_schools() -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 1: Loading treated relabel schools ───────────────────────────")

    relabels_path = Path(cfg.RELABELS_PARQUET)
    ipeds_path = Path(cfg.IPEDS_CROSSWALK_PARQUET)
    if not relabels_path.exists():
        raise FileNotFoundError(f"Missing relabel parquet: {relabels_path}")
    if not ipeds_path.exists():
        raise FileNotFoundError(f"Missing IPEDS crosswalk parquet: {ipeds_path}")

    con = ddb.connect()
    event_schools = con.sql(
        f"""
        WITH treated_events AS (
            SELECT DISTINCT
                CAST(unitid AS BIGINT) AS unitid,
                CAST(relabel_year AS INTEGER) AS relabel_year,
                CAST(relabel_type AS VARCHAR) AS relabel_type
            FROM read_parquet('{escape_sql_literal(relabels_path)}')
            WHERE event_flag = 1
              AND unitid IS NOT NULL
              AND relabel_year IS NOT NULL
        ),
        canonical_names AS (
            SELECT
                CAST(UNITID AS BIGINT) AS unitid,
                instname AS school_name,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(UNITID AS BIGINT)
                    ORDER BY CASE WHEN COALESCE(ALIAS, FALSE) THEN 1 ELSE 0 END, instname
                ) AS school_rank
            FROM read_parquet('{escape_sql_literal(ipeds_path)}')
            WHERE UNITID IS NOT NULL
              AND instname IS NOT NULL
        )
        SELECT
            te.unitid,
            te.relabel_year,
            te.relabel_type,
            cn.school_name
        FROM treated_events AS te
        LEFT JOIN canonical_names AS cn
          ON te.unitid = cn.unitid
         AND cn.school_rank = 1
        ORDER BY te.relabel_year, cn.school_name, te.relabel_type
        """
    ).df()
    con.close()

    if event_schools["school_name"].isna().any():
        missing = event_schools.loc[event_schools["school_name"].isna(), "unitid"].drop_duplicates().tolist()
        raise ValueError(f"Missing canonical IPEDS school_name for treated unitids: {missing}")

    event_schools["school_name_clean"] = event_schools["school_name"].map(_normalize_school_name)
    event_schools = event_schools.loc[:, EVENT_SCHOOLS_COLUMNS]
    event_schools = _apply_testing_filter(event_schools)

    print(f"  treated event rows: {len(event_schools):,}")
    print(f"  unique schools: {event_schools['unitid'].nunique():,}")
    print(f"  year range: {event_schools['relabel_year'].min()} - {event_schools['relabel_year'].max()}")
    print(f"  Step 1 done in {_elapsed(t0)}")
    return event_schools


# Source-CIP filter for "economics" programs (mirrors econ_relabels_opt_usage.py RELABEL_SPECS).
# CIP prefix 4506 = Economics; exclude 450603 (Econometrics, the *target* CIP of the relabel).
_CONTROL_SOURCE_CIP_PREFIX = "4506"
_CONTROL_SOURCE_CIP_EXCLUDE = ["450603"]


def load_never_treated_schools(event_schools: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1b: Load never-treated econ control schools from IPEDS.

    Identifies IPEDS institutions that have source-CIP economics master's programs
    (CIP prefix 4506, excluding econometrics 450603) within the configured control
    year range, but do NOT appear as treated schools in the relabels parquet.
    Returns a DataFrame with the same schema as event_schools but with
    relabel_year = None and relabel_type = 'never_treated'.
    """
    t0 = time.time()
    print("\n── Step 1b: Loading never-treated econ control schools ───────────────")

    completions_path = Path(cfg.IPEDS_COMPLETIONS_PARQUET)
    ipeds_path = Path(cfg.IPEDS_CROSSWALK_PARQUET)
    if not completions_path.exists():
        raise FileNotFoundError(f"Missing IPEDS completions parquet: {completions_path}")
    if not ipeds_path.exists():
        raise FileNotFoundError(f"Missing IPEDS crosswalk parquet: {ipeds_path}")

    treated_unitids = sorted(event_schools["unitid"].dropna().astype("int64").unique().tolist())
    treated_sql = ", ".join(str(u) for u in treated_unitids) if treated_unitids else "NULL"
    exclude_cips = ", ".join(f"'{c}'" for c in _CONTROL_SOURCE_CIP_EXCLUDE)
    min_year = cfg.BUILD_CONTROL_YEAR_MIN
    max_year = cfg.BUILD_CONTROL_YEAR_MAX

    con = ddb.connect()
    never_treated = con.sql(
        f"""
        WITH source_units AS (
            SELECT DISTINCT CAST(unitid AS BIGINT) AS unitid
            FROM read_parquet('{escape_sql_literal(completions_path)}')
            WHERE unitid IS NOT NULL
              AND cipcode IS NOT NULL
              AND CAST(awlevel AS INTEGER) = 7
              AND LPAD(CAST(cipcode AS VARCHAR), 6, '0') LIKE '{_CONTROL_SOURCE_CIP_PREFIX}%'
              AND LPAD(CAST(cipcode AS VARCHAR), 6, '0') NOT IN ({exclude_cips})
              AND CAST(year AS INTEGER) BETWEEN {min_year} AND {max_year}
              AND CAST(unitid AS BIGINT) NOT IN ({treated_sql})
        ),
        canonical_names AS (
            SELECT
                CAST(UNITID AS BIGINT) AS unitid,
                instname AS school_name,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(UNITID AS BIGINT)
                    ORDER BY CASE WHEN COALESCE(ALIAS, FALSE) THEN 1 ELSE 0 END, instname
                ) AS school_rank
            FROM read_parquet('{escape_sql_literal(ipeds_path)}')
            WHERE UNITID IS NOT NULL AND instname IS NOT NULL
        )
        SELECT
            su.unitid,
            NULL::INTEGER AS relabel_year,
            'never_treated' AS relabel_type,
            cn.school_name
        FROM source_units su
        LEFT JOIN canonical_names cn ON su.unitid = cn.unitid AND cn.school_rank = 1
        ORDER BY cn.school_name
        """
    ).df()
    con.close()

    missing = never_treated.loc[never_treated["school_name"].isna(), "unitid"].tolist()
    if missing:
        print(f"  Warning: no IPEDS name for {len(missing)} control unitid(s), dropping: {missing[:20]}")
        never_treated = never_treated.loc[never_treated["school_name"].notna()].copy()

    never_treated["school_name_clean"] = never_treated["school_name"].map(_normalize_school_name)
    never_treated = never_treated.loc[:, EVENT_SCHOOLS_COLUMNS].reset_index(drop=True)

    # Testing: sample a small subset of control schools.
    if cfg.TESTING_ENABLED:
        n_sample = min(cfg.TESTING_SAMPLE_N_SCHOOLS, len(never_treated))
        rng = np.random.default_rng(cfg.TESTING_RANDOM_SEED)
        chosen = sorted(rng.choice(len(never_treated), size=n_sample, replace=False).tolist())
        never_treated = never_treated.iloc[chosen].reset_index(drop=True)
        print(f"[test] Sampled {n_sample} never-treated schools: {never_treated['school_name'].tolist()}")

    print(f"  never-treated control schools: {never_treated['unitid'].nunique():,}")
    print(f"  control year range: {min_year} – {max_year}")
    print(f"  Step 1b done in {_elapsed(t0)}")
    return never_treated


def build_school_search_names(event_schools: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 2: Building school search names ─────────────────────────────")

    unitid_rows = sorted(event_schools["unitid"].dropna().astype("int64").unique().tolist())
    if not unitid_rows:
        raise ValueError("No treated schools available after filtering.")

    con = ddb.connect()
    unitid_sql = ",".join(str(int(unitid)) for unitid in unitid_rows)
    ipeds_names = con.sql(
        f"""
        SELECT
            CAST(UNITID AS BIGINT) AS unitid,
            instname AS search_name,
            COALESCE(ALIAS, FALSE) AS alias_ind
        FROM read_parquet('{escape_sql_literal(cfg.IPEDS_CROSSWALK_PARQUET)}')
        WHERE UNITID IS NOT NULL
          AND instname IS NOT NULL
          AND CAST(UNITID AS BIGINT) IN ({unitid_sql})
        ORDER BY CAST(UNITID AS BIGINT), COALESCE(ALIAS, FALSE), instname
        """
    ).df()
    con.close()

    canonical = (
        event_schools.loc[:, ["unitid", "school_name"]]
        .drop_duplicates()
        .rename(columns={"school_name": "search_name"})
        .assign(alias_ind=False)
    )
    search_names = pd.concat([ipeds_names, canonical], ignore_index=True)
    search_names = search_names.merge(
        event_schools.loc[:, ["unitid", "school_name", "school_name_clean"]].drop_duplicates(),
        on="unitid",
        how="left",
    )
    search_names["search_name_clean"] = search_names["search_name"].map(_normalize_school_name)
    search_names = search_names.loc[search_names["search_name_clean"].ne("")].copy()
    search_names = search_names.drop_duplicates(
        subset=["unitid", "school_name", "school_name_clean", "search_name_clean"]
    ).reset_index(drop=True)

    print(f"  search names: {len(search_names):,} rows across {search_names['unitid'].nunique():,} schools")
    print(f"  Step 2 done in {_elapsed(t0)}")
    return search_names


def build_manual_rsid_candidates(event_schools: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 3: Applying manual school → rsid overrides ─────────────────")

    overrides = cfg.BUILD_MANUAL_RSID_OVERRIDES
    if not overrides:
        print("  manual overrides configured: 0")
        print(f"  Step 3 done in {_elapsed(t0)}")
        return pd.DataFrame(columns=RSID_CANDIDATE_COLUMNS)

    normalized_overrides = {
        _normalize_school_name(name): (name, rsid)
        for name, rsid in overrides.items()
        if _normalize_school_name(name)
    }

    rows: list[dict[str, object]] = []
    unique_schools = (
        event_schools.loc[:, ["unitid", "school_name", "school_name_clean"]]
        .drop_duplicates()
        .sort_values(["school_name", "unitid"])
    )
    for school_row in unique_schools.itertuples(index=False):
        school_name = str(school_row.school_name)
        school_name_clean = str(school_row.school_name_clean)
        override_rsid = overrides.get(school_name)
        override_name = school_name
        if override_rsid is None:
            match = normalized_overrides.get(school_name_clean)
            if match is not None:
                override_name, override_rsid = match
        if override_rsid is None:
            continue
        rows.append(
            {
                "unitid": int(school_row.unitid),
                "school_name": school_name,
                "school_name_clean": school_name_clean,
                "university_name": override_name,
                "university_name_clean": _normalize_school_name(override_name),
                "rsid": int(override_rsid),
                "candidate_count": pd.NA,
                "name_match_score": pd.NA,
                "candidate_rank": 1,
            }
        )

    manual_candidates = pd.DataFrame(rows, columns=RSID_CANDIDATE_COLUMNS)
    if not manual_candidates.empty:
        for col in ["unitid", "rsid", "candidate_rank"]:
            manual_candidates[col] = pd.to_numeric(manual_candidates[col], errors="coerce").astype("Int64")
        for col in ["candidate_count", "name_match_score"]:
            manual_candidates[col] = pd.to_numeric(manual_candidates[col], errors="coerce").astype("Int64")
        print(f"  manual overrides used for schools: {manual_candidates['unitid'].nunique():,}")
        print(
            manual_candidates.loc[:, ["school_name", "rsid"]]
            .sort_values(["school_name", "rsid"])
            .to_string(index=False)
        )
    else:
        print("  manual overrides matched schools: 0")
    print(f"  Step 3 done in {_elapsed(t0)}")
    return manual_candidates


def load_or_build_candidates(
    event_schools: pd.DataFrame,
    path_str: str,
    label: str,
) -> tuple[pd.DataFrame, bool, str]:
    path = Path(path_str)
    school_unitids = (
        event_schools["unitid"].dropna().astype("int64").drop_duplicates().tolist()
        if not event_schools.empty
        else []
    )

    if cfg.BUILD_REUSE_EXISTING_CANDIDATE_PARQUETS and path.exists():
        t0 = time.time()
        print(f"\n── Step 3: Reusing existing {label} rsid candidates ────────────────")
        candidates = _load_parquet_with_columns(path_str, RSID_CANDIDATE_COLUMNS)
        candidates = _coerce_rsid_candidate_columns(candidates)
        if school_unitids:
            candidates = candidates.loc[
                candidates["unitid"].isin(school_unitids)
            ].copy()
        else:
            candidates = candidates.iloc[0:0].copy()
        print(f"  loaded {len(candidates):,} candidate rows from {path}")
        print(f"  schools in cached candidates: {candidates['unitid'].nunique() if not candidates.empty else 0:,}")
        print(f"  Step 3 done in {_elapsed(t0)}")
        return candidates, True, f"existing candidate parquet: {path}"

    candidates = build_manual_rsid_candidates(event_schools)
    return candidates, False, "build.manual_rsid_overrides"


def print_manual_only_rsid_summary(
    event_schools: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_source: str,
) -> None:
    """
    Step 4: Report coverage when rsid selection is restricted to YAML overrides.

    Schools not matched by build.manual_rsid_overrides are intentionally skipped
    and will not trigger any sampled school -> rsid lookup in this script.
    """
    t0 = time.time()
    print("\n── Step 4: Skipping school → rsid candidate lookup ─────────────────")

    unique_schools = (
        event_schools.loc[:, ["unitid", "school_name"]]
        .drop_duplicates()
        .sort_values(["school_name", "unitid"])
        .reset_index(drop=True)
    )
    covered_unitids = (
        candidates["unitid"].dropna().astype("int64").drop_duplicates().tolist()
        if not candidates.empty
        else []
    )
    missing = unique_schools.loc[~unique_schools["unitid"].isin(covered_unitids)].copy()

    print("  in-script school -> rsid lookup disabled")
    print(f"  candidate source: {candidate_source}")
    print(f"  schools with candidate rows: {len(covered_unitids):,}")
    print(f"  schools skipped without candidate rows: {len(missing):,}")
    print(f"  Step 4 done in {_elapsed(t0)}")


def query_rsid_candidates(db, search_names: pd.DataFrame, excluded_unitids: set[int] | None = None) -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 4: Querying school → rsid candidates ───────────────────────")

    if search_names.empty:
        return pd.DataFrame(columns=RSID_CANDIDATE_COLUMNS)

    excluded_unitids = excluded_unitids or set()

    search_rows = search_names.loc[
        :,
        ["unitid", "school_name", "school_name_clean", "search_name", "search_name_clean", "alias_ind"],
    ].drop_duplicates()
    if excluded_unitids:
        search_rows = search_rows.loc[~search_rows["unitid"].astype("int64").isin(sorted(excluded_unitids))].copy()
    if search_rows.empty:
        print(f"  schools skipped due to manual rsid overrides: {len(excluded_unitids):,}")
        print(f"  Step 4 done in {_elapsed(t0)}")
        return pd.DataFrame(columns=RSID_CANDIDATE_COLUMNS)
    search_rows["is_canonical"] = (search_rows["search_name"] == search_rows["school_name"]).astype(int)
    search_rows["search_len"] = search_rows["search_name"].astype(str).str.len()
    search_rows = search_rows.sort_values(
        ["unitid", "is_canonical", "search_len", "alias_ind", "search_name"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)

    candidate_frames: list[pd.DataFrame] = []
    resolved_unitids: list[int] = []
    unresolved_rows: list[tuple[int, str]] = []

    grouped = search_rows.groupby(["unitid", "school_name", "school_name_clean"], sort=False)
    for (unitid, school_name, school_name_clean), school_search in grouped:
        sample_df = _wrds_sql_to_df(
            db,
            _build_school_exact_query(str(school_name)),
            label=f"rsid exact query [{school_name}] via full name '{school_name}'",
        )
        matched_search_name = None
        matched_search_name_clean = None

        if not sample_df.empty:
            matched_search_name = str(school_name)
            matched_search_name_clean = str(school_name_clean)
        else:
            for search_row in school_search.itertuples(index=False):
                query = _build_school_sample_query(search_row.search_name)
                sample_df = _wrds_sql_to_df(
                    db,
                    query,
                    label=f"rsid sample query [{school_name}] via '{search_row.search_name}'",
                )
                if not sample_df.empty:
                    matched_search_name = str(search_row.search_name)
                    matched_search_name_clean = str(search_row.search_name_clean)
                    break

        if sample_df.empty or matched_search_name is None or matched_search_name_clean is None:
            unresolved_rows.append((int(unitid), str(school_name)))
            continue

        sample_df = sample_df.copy().reset_index(drop=True)
        sample_df["sample_order"] = np.arange(1, len(sample_df) + 1, dtype=int)
        sample_df["rsid"] = pd.to_numeric(sample_df["rsid"], errors="coerce").astype("Int64")
        sample_df = sample_df.loc[sample_df["rsid"].notna()].copy()
        if sample_df.empty:
            unresolved_rows.append((int(unitid), str(school_name)))
            continue

        sample_df["name_match_score"] = np.select(
            [
                sample_df["university_raw_clean"].eq(school_name_clean),
                sample_df["university_raw_clean"].eq(matched_search_name_clean),
                sample_df["university_raw_clean"].str.contains(matched_search_name_clean, regex=False)
                | pd.Series(
                    [
                        matched_search_name_clean in value if isinstance(value, str) else False
                        for value in sample_df["university_raw_clean"]
                    ],
                    index=sample_df.index,
                ),
            ],
            [3, 3, 2],
            default=1,
        )

        rep_names = (
            sample_df.groupby(["rsid", "university_name", "university_name_clean"], as_index=False)
            .agg(
                university_name_count=("sample_order", "size"),
                first_sample_order=("sample_order", "min"),
            )
            .sort_values(
                ["rsid", "university_name_count", "first_sample_order", "university_name"],
                ascending=[True, False, True, True],
            )
            .drop_duplicates(subset=["rsid"], keep="first")
            .loc[:, ["rsid", "university_name", "university_name_clean"]]
        )

        rsid_counts = (
            sample_df.groupby("rsid", as_index=False)
            .agg(
                candidate_count=("sample_order", "size"),
                name_match_score=("name_match_score", "max"),
            )
            .merge(rep_names, on="rsid", how="left")
        )
        rsid_counts["unitid"] = int(unitid)
        rsid_counts["school_name"] = str(school_name)
        rsid_counts["school_name_clean"] = str(school_name_clean)
        candidate_frames.append(
            rsid_counts.loc[
                :,
                [
                    "unitid",
                    "school_name",
                    "school_name_clean",
                    "university_name",
                    "university_name_clean",
                    "rsid",
                    "candidate_count",
                    "name_match_score",
                ],
            ]
        )
        resolved_unitids.append(int(unitid))

    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(columns=RSID_CANDIDATE_COLUMNS)
    if not candidates.empty:
        candidates["unitid"] = pd.to_numeric(candidates["unitid"], errors="coerce").astype("Int64")
        candidates["rsid"] = pd.to_numeric(candidates["rsid"], errors="coerce").astype("Int64")
        candidates["candidate_count"] = pd.to_numeric(candidates["candidate_count"], errors="coerce").astype("Int64")
        candidates["name_match_score"] = pd.to_numeric(candidates["name_match_score"], errors="coerce").astype("Int64")
        candidates = candidates.sort_values(
            ["unitid", "candidate_count", "name_match_score", "rsid", "university_name"],
            ascending=[True, False, False, True, True],
        ).reset_index(drop=True)
        candidates["candidate_rank"] = (candidates.groupby("unitid").cumcount() + 1).astype("Int64")

    print(f"  schools skipped due to manual rsid overrides: {len(excluded_unitids):,}")
    print(f"  schools with sampled candidates: {len(set(resolved_unitids)):,}")
    print(f"  unresolved schools after sampled lookup: {len(unresolved_rows):,}")
    print(f"  Step 4 done in {_elapsed(t0)}")
    return candidates


def select_top_rsid(candidates: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 5: Selecting one rsid per school ────────────────────────────")

    if candidates.empty:
        selected = pd.DataFrame(columns=SELECTED_RSID_COLUMNS)
    else:
        selected = (
            candidates.loc[candidates["candidate_rank"] == 1, SELECTED_RSID_COLUMNS]
            .sort_values(["school_name", "unitid"])
            .reset_index(drop=True)
        )
    print(f"  selected schools: {len(selected):,}")
    if not selected.empty:
        print(
            selected.loc[:, ["school_name", "rsid", "candidate_count", "name_match_score"]]
            .to_string(index=False)
        )
    print(f"  Step 5 done in {_elapsed(t0)}")
    return selected


def query_matched_education(db, event_schools: pd.DataFrame, selected_rsid: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 6: Querying matched education rows ──────────────────────────")

    if event_schools.empty or selected_rsid.empty:
        return pd.DataFrame(columns=MATCHED_EDUCATION_COLUMNS)

    events_sql = _build_values_sql(
        list(
            event_schools.loc[:, EVENT_SCHOOLS_COLUMNS]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    )
    selected_sql = _build_values_sql(
        list(
            selected_rsid.loc[:, SELECTED_RSID_COLUMNS]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    )

    sql = f"""
        WITH treated_events (
            unitid,
            relabel_year,
            relabel_type,
            school_name,
            school_name_clean
        ) AS (
            VALUES
            {events_sql}
        ),
        selected_rsid (
            unitid,
            school_name,
            school_name_clean,
            university_name,
            university_name_clean,
            rsid,
            candidate_count,
            name_match_score,
            candidate_rank
        ) AS (
            VALUES
            {selected_sql}
        ),
        event_rsid AS (
            SELECT
                e.unitid,
                e.relabel_year,
                e.relabel_type,
                e.school_name,
                e.school_name_clean,
                s.university_name AS event_rsid_university_name,
                s.rsid AS event_rsid,
                s.candidate_count AS rsid_candidate_count,
                s.name_match_score AS rsid_name_match_score
            FROM treated_events AS e
            JOIN selected_rsid AS s
              ON e.unitid = s.unitid
        ),
        doctorate_spells AS (
            SELECT
                a.user_id,
                CAST(a.rsid AS BIGINT) AS rsid,
                a.education_number,
                a.startdate,
                a.enddate
            FROM revelio.individual_user_education AS a
            JOIN (
                SELECT DISTINCT event_rsid
                FROM event_rsid
            ) AS er2
              ON CAST(a.rsid AS BIGINT) = er2.event_rsid
            WHERE a.degree = 'Doctor'
              AND a.startdate IS NOT NULL
        )
        SELECT
            er.unitid,
            er.school_name,
            er.school_name_clean,
            er.relabel_year,
            er.relabel_type,
            er.event_rsid,
            er.event_rsid_university_name,
            er.rsid_candidate_count,
            er.rsid_name_match_score,
            a.user_id,
            u.fullname,
            CAST(a.rsid AS BIGINT) AS rsid,
            a.university_name,
            a.education_number,
            a.startdate AS ed_startdate,
            a.enddate AS ed_enddate,
            EXTRACT(YEAR FROM a.enddate)::INTEGER AS ed_end_year,
            a.degree,
            a.field,
            a.university_country,
            a.university_location,
            r.university_raw,
            r.degree_raw,
            r.field_raw,
            r.description,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM doctorate_spells AS p
                    WHERE p.user_id = a.user_id
                      AND p.rsid = CAST(a.rsid AS BIGINT)
                      AND COALESCE(p.education_number, -1) <> COALESCE(a.education_number, -1)
                      AND a.startdate IS NOT NULL
                      AND a.enddate IS NOT NULL
                      AND p.startdate <= a.enddate + INTERVAL '{IMMEDIATE_PHD_WINDOW_DAYS} day'
                      AND COALESCE(p.enddate, p.startdate) >= a.startdate
                ) THEN 1
                ELSE 0
            END AS exclude_immediate_same_inst_phd_after_master_ind
        FROM event_rsid AS er
        JOIN revelio.individual_user_education AS a
          ON CAST(a.rsid AS BIGINT) = er.event_rsid
        LEFT JOIN revelio.individual_user AS u
          ON a.user_id = u.user_id
        LEFT JOIN revelio.individual_user_education_raw AS r
          ON a.user_id = r.user_id
         AND a.education_number = r.education_number
        WHERE a.degree = 'Master'
          AND a.enddate IS NOT NULL
          AND ABS(EXTRACT(YEAR FROM a.enddate)::INTEGER - er.relabel_year) <= {cfg.BUILD_EDUCATION_EVENT_WINDOW_YEARS}
        ORDER BY er.relabel_year, er.school_name, a.user_id, a.enddate, a.education_number
    """
    matched_education = _wrds_sql_to_df(db, sql, label="education extract query")

    if not matched_education.empty:
        for col in [
            "unitid",
            "relabel_year",
            "event_rsid",
            "rsid_candidate_count",
            "rsid_name_match_score",
            "user_id",
            "rsid",
            "education_number",
            "ed_end_year",
            "exclude_immediate_same_inst_phd_after_master_ind",
        ]:
            if col in matched_education.columns:
                matched_education[col] = pd.to_numeric(matched_education[col], errors="coerce").astype("Int64")

    print(f"  matched users: {matched_education['user_id'].nunique() if not matched_education.empty else 0:,}")
    if "exclude_immediate_same_inst_phd_after_master_ind" in matched_education.columns:
        print(
            "  same-institution doctoral continuers flagged: "
            f"{int(matched_education['exclude_immediate_same_inst_phd_after_master_ind'].fillna(0).sum()):,}"
        )
    print(f"  Step 6 done in {_elapsed(t0)}")
    return matched_education


def query_control_education(db, control_schools: pd.DataFrame, selected_rsid: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6b: Query Revelio education rows for never-treated control schools.

    Like query_matched_education but without an event-window filter. Instead,
    pulls all Master's graduates from cfg.BUILD_CONTROL_YEAR_MIN to
    cfg.BUILD_CONTROL_YEAR_MAX so the analysis script can apply matching later.
    relabel_year is NULL and relabel_type is 'never_treated' in the output.
    """
    t0 = time.time()
    print("\n── Step 6b: Querying never-treated control school education rows ─────")

    if control_schools.empty or selected_rsid.empty:
        print("  no control schools or rsids — skipping")
        print(f"  Step 6b done in {_elapsed(t0)}")
        return pd.DataFrame(columns=MATCHED_EDUCATION_COLUMNS)

    # Only pass unitid/school_name/school_name_clean — relabel_year is not used.
    schools_sql = _build_values_sql(
        list(
            control_schools.loc[:, ["unitid", "school_name", "school_name_clean"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    )
    selected_sql = _build_values_sql(
        list(
            selected_rsid.loc[:, SELECTED_RSID_COLUMNS]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    )

    min_year = cfg.BUILD_CONTROL_YEAR_MIN
    max_year = cfg.BUILD_CONTROL_YEAR_MAX

    sql = f"""
        WITH control_schools (
            unitid,
            school_name,
            school_name_clean
        ) AS (
            VALUES
            {schools_sql}
        ),
        selected_rsid (
            unitid,
            school_name,
            school_name_clean,
            university_name,
            university_name_clean,
            rsid,
            candidate_count,
            name_match_score,
            candidate_rank
        ) AS (
            VALUES
            {selected_sql}
        ),
        event_rsid AS (
            SELECT
                cs.unitid,
                NULL::INTEGER AS relabel_year,
                'never_treated' AS relabel_type,
                cs.school_name,
                cs.school_name_clean,
                s.university_name AS event_rsid_university_name,
                s.rsid AS event_rsid,
                s.candidate_count AS rsid_candidate_count,
                s.name_match_score AS rsid_name_match_score
            FROM control_schools AS cs
            JOIN selected_rsid AS s ON cs.unitid = s.unitid
        ),
        doctorate_spells AS (
            SELECT
                a.user_id,
                CAST(a.rsid AS BIGINT) AS rsid,
                a.education_number,
                a.startdate,
                a.enddate
            FROM revelio.individual_user_education AS a
            JOIN (
                SELECT DISTINCT event_rsid
                FROM event_rsid
            ) AS er2
              ON CAST(a.rsid AS BIGINT) = er2.event_rsid
            WHERE a.degree = 'Doctor'
              AND a.startdate IS NOT NULL
        )
        SELECT
            er.unitid,
            er.school_name,
            er.school_name_clean,
            er.relabel_year,
            er.relabel_type,
            er.event_rsid,
            er.event_rsid_university_name,
            er.rsid_candidate_count,
            er.rsid_name_match_score,
            a.user_id,
            u.fullname,
            CAST(a.rsid AS BIGINT) AS rsid,
            a.university_name,
            a.education_number,
            a.startdate AS ed_startdate,
            a.enddate AS ed_enddate,
            EXTRACT(YEAR FROM a.enddate)::INTEGER AS ed_end_year,
            a.degree,
            a.field,
            a.university_country,
            a.university_location,
            r.university_raw,
            r.degree_raw,
            r.field_raw,
            r.description,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM doctorate_spells AS p
                    WHERE p.user_id = a.user_id
                      AND p.rsid = CAST(a.rsid AS BIGINT)
                      AND COALESCE(p.education_number, -1) <> COALESCE(a.education_number, -1)
                      AND a.startdate IS NOT NULL
                      AND a.enddate IS NOT NULL
                      AND p.startdate <= a.enddate + INTERVAL '{IMMEDIATE_PHD_WINDOW_DAYS} day'
                      AND COALESCE(p.enddate, p.startdate) >= a.startdate
                ) THEN 1
                ELSE 0
            END AS exclude_immediate_same_inst_phd_after_master_ind
        FROM event_rsid AS er
        JOIN revelio.individual_user_education AS a
          ON CAST(a.rsid AS BIGINT) = er.event_rsid
        LEFT JOIN revelio.individual_user AS u
          ON a.user_id = u.user_id
        LEFT JOIN revelio.individual_user_education_raw AS r
          ON a.user_id = r.user_id
         AND a.education_number = r.education_number
        WHERE a.degree = 'Master'
          AND a.enddate IS NOT NULL
          AND EXTRACT(YEAR FROM a.enddate)::INTEGER BETWEEN {min_year} AND {max_year}
        ORDER BY er.school_name, a.user_id, a.enddate, a.education_number
    """
    control_education = _wrds_sql_to_df(db, sql, label="control education extract")

    if not control_education.empty:
        for col in [
            "unitid",
            "relabel_year",
            "event_rsid",
            "rsid_candidate_count",
            "rsid_name_match_score",
            "user_id",
            "rsid",
            "education_number",
            "ed_end_year",
            "exclude_immediate_same_inst_phd_after_master_ind",
        ]:
            if col in control_education.columns:
                control_education[col] = pd.to_numeric(control_education[col], errors="coerce").astype("Int64")

    print(f"  control users: {control_education['user_id'].nunique() if not control_education.empty else 0:,}")
    if "exclude_immediate_same_inst_phd_after_master_ind" in control_education.columns:
        print(
            "  same-institution doctoral continuers flagged: "
            f"{int(control_education['exclude_immediate_same_inst_phd_after_master_ind'].fillna(0).sum()):,}"
        )
    print(f"  Step 6b done in {_elapsed(t0)}")
    return control_education


def _build_positions_query(user_ids: list[int]) -> str:
    user_id_sql = ",".join(str(int(user_id)) for user_id in user_ids)
    return f"""
        SELECT
            a.user_id,
            a.position_id,
            a.position_number,
            a.rcid,
            a.country,
            a.startdate,
            a.enddate,
            a.salary,
            a.total_compensation,
            b.company_raw,
            b.location_raw,
            b.title_raw,
            b.description
        FROM revelio.individual_positions AS a
        LEFT JOIN revelio.individual_positions_raw AS b
          ON a.position_id = b.position_id
        WHERE a.user_id IN ({user_id_sql})
        ORDER BY a.user_id, a.position_number, a.startdate, a.position_id
    """


def query_matched_positions(db, matched_education: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 7: Querying full position histories ─────────────────────────")

    user_ids = (
        matched_education["user_id"].dropna().astype("int64").drop_duplicates().sort_values().tolist()
        if not matched_education.empty
        else []
    )
    if not user_ids:
        return pd.DataFrame(columns=MATCHED_POSITIONS_COLUMNS)

    chunks = chunk_values(user_ids, cfg.BUILD_POSITION_CHUNK_SIZE)
    frames: list[pd.DataFrame] = []
    for idx, user_chunk in enumerate(chunks, start=1):
        chunk_df = _wrds_sql_to_df(
            db,
            _build_positions_query(user_chunk),
            label=f"positions chunk {idx}/{len(chunks)}",
        )
        frames.append(chunk_df)

    positions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MATCHED_POSITIONS_COLUMNS)
    if not positions.empty:
        for col in ["user_id", "position_id", "position_number", "rcid"]:
            positions[col] = pd.to_numeric(positions[col], errors="coerce").astype("Int64")

    print(f"  matched position rows: {len(positions):,}")
    print(f"  Step 7 done in {_elapsed(t0)}")
    return positions


def print_summary(event_schools: pd.DataFrame, candidates: pd.DataFrame, selected_rsid: pd.DataFrame, matched_education: pd.DataFrame, matched_positions: pd.DataFrame) -> None:
    unique_schools = event_schools.loc[:, ["unitid", "school_name"]].drop_duplicates()
    candidate_school_count = candidates["unitid"].nunique() if not candidates.empty else 0
    selected_school_count = selected_rsid["unitid"].nunique() if not selected_rsid.empty else 0
    missing = unique_schools.loc[~unique_schools["unitid"].isin(selected_rsid["unitid"] if not selected_rsid.empty else pd.Series(dtype="Int64"))]

    print("\n── Summary ───────────────────────────────────────────────────────────")
    print(f"  relabel schools: {len(unique_schools):,}")
    print(f"  schools with rsid candidates: {candidate_school_count:,}")
    print(f"  schools with selected rsid: {selected_school_count:,}")
    print(f"  schools without selected rsid: {len(missing):,}")
    if not missing.empty:
        print("  missing schools:")
        print(missing.sort_values(["school_name", "unitid"]).to_string(index=False))
    print(f"  matched education rows: {len(matched_education):,}")
    print(f"  matched education users: {matched_education['user_id'].nunique() if not matched_education.empty else 0:,}")
    print(f"  matched position rows: {len(matched_positions):,}")


def main() -> None:
    t_main = time.time()
    db = None
    try:
        event_schools = load_event_schools()

        if not cfg.BUILD_CONTROL_ONLY:
            candidates, reused_treated_candidates, treated_candidate_source = load_or_build_candidates(
                event_schools,
                cfg.RSID_CANDIDATES_PARQUET,
                "treated",
            )
            print_manual_only_rsid_summary(event_schools, candidates, treated_candidate_source)

            _write_parquet(event_schools, cfg.EVENT_SCHOOLS_PARQUET, EVENT_SCHOOLS_COLUMNS)

            db = get_wrds_connection(
                wrds_username=cfg.WRDS_USERNAME,
                pgpass_path=cfg.WRDS_PGPASS_PATH,
            )

            if not candidates.empty:
                candidates = candidates.loc[:, RSID_CANDIDATE_COLUMNS].sort_values(
                    ["school_name", "candidate_rank", "rsid"],
                    ascending=[True, True, True],
                ).reset_index(drop=True)
            selected_rsid = select_top_rsid(candidates)
            matched_education = query_matched_education(db, event_schools, selected_rsid)
            matched_positions = query_matched_positions(db, matched_education)

            if not reused_treated_candidates:
                _write_parquet(candidates, cfg.RSID_CANDIDATES_PARQUET, RSID_CANDIDATE_COLUMNS)
            else:
                print(f"  keeping existing {cfg.RSID_CANDIDATES_PARQUET}")
            _write_parquet(selected_rsid, cfg.SELECTED_RSID_PARQUET, SELECTED_RSID_COLUMNS)
            _write_parquet(matched_education, cfg.MATCHED_EDUCATION_PARQUET, MATCHED_EDUCATION_COLUMNS)
            _write_parquet(matched_positions, cfg.MATCHED_POSITIONS_PARQUET, MATCHED_POSITIONS_COLUMNS)

            print_summary(event_schools, candidates, selected_rsid, matched_education, matched_positions)
        else:
            print("\n── Treated Schools ─────────────────────────────────────────────────")
            print("  control_only=true: skipping treated-school candidate, education, and position outputs")

        # ── Never-treated control schools ─────────────────────────────────────
        print("\n\n════ Control Schools (never-treated econ) ════════════════════════════")
        control_schools = load_never_treated_schools(event_schools)
        _write_parquet(control_schools, cfg.NEVER_TREATED_SCHOOLS_PARQUET, EVENT_SCHOOLS_COLUMNS)

        control_candidates, reused_control_candidates, control_candidate_source = load_or_build_candidates(
            control_schools,
            cfg.NEVER_TREATED_RSID_CANDIDATES_PARQUET,
            "control",
        )
        print_manual_only_rsid_summary(control_schools, control_candidates, control_candidate_source)
        if db is None:
            db = get_wrds_connection(
                wrds_username=cfg.WRDS_USERNAME,
                pgpass_path=cfg.WRDS_PGPASS_PATH,
            )
        if not control_candidates.empty:
            control_candidates = control_candidates.loc[:, RSID_CANDIDATE_COLUMNS].sort_values(
                ["school_name", "candidate_rank", "rsid"],
                ascending=[True, True, True],
            ).reset_index(drop=True)
        control_selected_rsid = select_top_rsid(control_candidates)
        control_education = query_control_education(db, control_schools, control_selected_rsid)
        control_positions = query_matched_positions(db, control_education)

        if not reused_control_candidates:
            _write_parquet(control_candidates, cfg.NEVER_TREATED_RSID_CANDIDATES_PARQUET, RSID_CANDIDATE_COLUMNS)
        else:
            print(f"  keeping existing {cfg.NEVER_TREATED_RSID_CANDIDATES_PARQUET}")
        _write_parquet(control_selected_rsid, cfg.NEVER_TREATED_SELECTED_RSID_PARQUET, SELECTED_RSID_COLUMNS)
        _write_parquet(control_education, cfg.NEVER_TREATED_EDUCATION_PARQUET, MATCHED_EDUCATION_COLUMNS)
        _write_parquet(control_positions, cfg.NEVER_TREATED_POSITIONS_PARQUET, MATCHED_POSITIONS_COLUMNS)

        print_summary(control_schools, control_candidates, control_selected_rsid, control_education, control_positions)
        print(f"\nTotal runtime: {_elapsed(t_main)}")
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
