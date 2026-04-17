"""
f1_indiv_merge.py
=================
F1 FOIA × Revelio individual merge pipeline.

Links F1 FOIA student records (identified by foia_person_id) to Revelio
individual profiles (user_id) at the education-spell level:
  unit = (person_id, school_name, degree_level, program_start_year)

Primary join key: F1 school name → Revelio university_raw (via pre-built
fuzzy name crosswalk in deps_f1_school_crosswalk.py) × country_of_birth.

Candidate generation uses hard filters on all join dimensions:
  - School: via school crosswalk (fuzzy-matched, score threshold)
  - Country: must match (enforces tractability of the cross-join)
  - Degree level: must match (Unknown/Missing passes through)
  - Program start year: |f1_prog_start_year - rev_educ_start_year| ≤ year_hard_buffer

Scoring (primary): IDF-weighted recall of F1 employers in Revelio position history.
  employer_score = Σ(idf_weight_i × matched_i) / Σ(idf_weight_i)
  where idf_weight = 1/log(smoothing + n_rev_users_at_employer)
  Employer match via entity-ID first, Jaro-Winkler fuzzy fallback for uncrosswalked employers.
  Persons with no employer data: employer_score excluded (field + name weights rebalanced).

Scoring (tiebreakers):
  field_score:      1.0 if 2-digit CIP matches, 0.5 if either unknown, 0.0 otherwise
  country_score: name-model confidence in nationality (nanat/nt model blend)

Output variants (see build_f1_merge_inputs()):
  baseline   — all rank-1 matches
  mult2/4/6  — baseline restricted to spells with ≤2/4/6 candidates
  strict     — high-precision filter (weight_norm ≥ 0.85, strict thresholds)

Usage (iPython):
    import importlib, sys
    sys.path.insert(0, '/home/yk0581/h1bworkers/code/f1_indiv_merge')
    import f1_indiv_merge as m
    importlib.reload(m)
    m.build_f1_merge_inputs()

    # Or with testing mode:
    m.build_f1_merge_inputs(testing=True)
"""

import json
import os
import sys
import time
from builtins import print as _print
from functools import partial

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import f1_indiv_merge_config as cfg  # noqa: E402
from config import root as _root  # noqa: E402
from deps_f1_school_crosswalk import _sql_clean_inst_name_expr  # noqa: E402
from employer_entity_sql import (  # noqa: E402
    sql_clean_zip_expr,
    sql_normalize_expr,
    sql_state_name_to_abbr_expr,
)
from helpers import field_clean_to_cip2_sql  # noqa: E402

# Flush progress output immediately so redirected logs stay live.
print = partial(_print, flush=True)


# ---------------------------------------------------------------------------
# Country standardization: load shared crosswalk (same as rev_users_clean.py)
# ---------------------------------------------------------------------------
_COUNTRY_DICT_PATH = os.path.join(_root, 'data', 'crosswalks', 'country_dict.json')
with open(_COUNTRY_DICT_PATH) as _f:
    _COUNTRY_CW_UPPER = {k.upper(): v for k, v in json.load(_f).items()}


def _std_f1_country(country: str):
    """Standardize an F1 FOIA country string to the Revelio country format.

    F1 uses UPPERCASE country names. Looks up via case-insensitive match
    against the shared country_dict.json crosswalk (same as rev_users_clean.py).
    Returns None if input is None; returns title-cased input if not found in crosswalk.
    """
    if country is None:
        return None
    return _COUNTRY_CW_UPPER.get(country.strip().upper(), country.strip().title())


# ---------------------------------------------------------------------------
# Company name normalization SQL expression
# ---------------------------------------------------------------------------

def _company_norm_sql(col: str) -> str:
    """Return a SQL expression that normalizes a company name column for matching.

    Applies (in order):
      1. Lowercase + trim
      2. Strip punctuation (replace with space)
      3. Standardize verbose suffixes to short forms (longest patterns first)
      4. Remove legal-entity type words (inc, llc, corp, ltd, etc.)
      5. Remove common stopwords (the, of, and, a, an, ...)
      6. Remove common company/sector filler words (group, services, global, ...)
      7. Collapse whitespace

    The result is a cleaned string suitable for JW similarity or token overlap
    matching. Original 'clean' columns (lower+trim only) are kept separately.

    Note: '\\b' in Python string literal → '\b' in string value → SQL literal '\b'
    → DuckDB passes backslash+b to RE2 → word boundary. Same logic for '\\s' → '\s'.
    """
    s = f"lower(trim({col}))"
    # Strip punctuation → space (inside [] most chars are literal, no extra escaping needed)
    s = f"regexp_replace({s}, '[,\\.;:()/&\\[\\]\\-]', ' ', 'g')"
    # Standardize verbose suffix forms (longest first to avoid partial matches)
    for pat, repl in [
        ('\\bincorporated\\b', 'inc'),
        ('\\bcorporation\\b',  'corp'),
        ('\\blimited\\b',      'ltd'),
        ('\\bcompany\\b',      'co'),
    ]:
        s = f"regexp_replace({s}, '{pat}', '{repl}', 'gi')"
    # Remove legal-entity suffixes.
    # '\\b' in Python string literal → '\b' in the string value → SQL literal '\b' →
    # DuckDB passes '\b' (backslash + b) to RE2 → RE2 interprets as word boundary ✓
    s = f"regexp_replace({s}, '\\b(inc|llc|corp|ltd|co|plc|lp|llp|pllc|pc|pa|na|nv|sa|ag|bv|gmbh)\\b', ' ', 'gi')"
    # Remove general stopwords
    s = f"regexp_replace({s}, '\\b(the|of|and|a|an|for|in|at|by|with|de|le|la|les|el|los)\\b', ' ', 'gi')"
    # Remove common company/sector filler words
    s = (
        f"regexp_replace({s}, "
        "'\\b(group|services|solutions|technologies|technology|global|international|"
        "national|american|north|south|east|west|systems|consulting|management|"
        "partners|associates|enterprises|holdings)\\b', ' ', 'gi')"
    )
    # Collapse whitespace ('\s' in Python → '\s' in SQL → RE2 whitespace class ✓)
    s = f"trim(regexp_replace({s}, '\\s+', ' ', 'g'))"
    return s


# ---------------------------------------------------------------------------
# DuckDB connection
# ---------------------------------------------------------------------------
def _configure_duckdb_runtime(con):
    con.execute("SET threads = 8")
    # Use home filesystem for temp (has ~40TB free); /tmp is a tiny 7.8GB partition
    con.execute("SET temp_directory = '/home/yk0581/tmp/duckdb_f1_merge'")
    con.execute("SET memory_limit = '48GB'")
    # initcap (capitalize first letter of each word) is absent in DuckDB 1.2.x
    con.create_function("initcap", lambda s: s.title() if s else s, [str], str)


con_f1 = duckdb.connect()
_configure_duckdb_runtime(con_f1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _sql_escape_path(path: str) -> str:
    return path.replace("'", "''")


_EMPLOYER_LOOKUP_REQUIRED_COLS = {
    "employer_name",
    "employer_city_clean",
    "employer_state_clean",
    "employer_zip_clean",
    "foia_row_uid",
    "foia_firm_uid",
    "rcid",
    "lookup_rcid_count",
    "lookup_rcid_ambiguous_ind",
    "lookup_has_direct_ind",
}

_SCHOOL_RESOLUTION_REQUIRED_COLS = {
    "f1_school_name",
    "f1_row_num",
    "UNITID",
    "f1_city_clean",
    "f1_state_clean",
    "f1_zip_clean",
    "rev_university_raw",
    "rev_instname_clean",
    "school_match_score",
    "rev_matchtype",
    "match_group",
    "rev_match_source",
}


def _sql_f1_emp_lookup_join_clause(f_alias: str = "f", lu_alias: str = "lu") -> str:
    return f"""
        lower(trim({f_alias}.employer_name)) = lower(trim({lu_alias}.employer_name))
        AND COALESCE({sql_normalize_expr(f'{f_alias}.employer_city')}, '') = COALESCE({lu_alias}.employer_city_clean, '')
        AND COALESCE({sql_state_name_to_abbr_expr(f'{f_alias}.employer_state')}, '') = COALESCE({lu_alias}.employer_state_clean, '')
        AND COALESCE({sql_clean_zip_expr(f'{f_alias}.employer_zip_code')}, '') = COALESCE({lu_alias}.employer_zip_clean, '')
    """


def _build_school_crosswalk_family_query(
    cw_tab: str,
    source_cols: list[str] | None = None,
    SCORE_GAP: float = cfg.BUILD_SCHOOL_AMBIGUITY_SCORE_GAP,
) -> str:
    """Collapse raw school crosswalk rows to kept cleaned-name families.

    The source crosswalk may contain multiple raw Revelio university strings for
    the same cleaned school family. This query first collapses to
    (f1_school_name, rev_instname_clean), then keeps only the top family plus
    near-ties within SCORE_GAP of the school's best family-level match_score.
    """
    source_cols = source_cols or []
    n_rev_users_col = _first_present_col(
        source_cols,
        ["n_rev_users", "n_revelio_institution_records", "n_revelio_inst_raw_variants", "n_users_right"],
    )
    match_score_col = _first_present_col(source_cols, ["match_score", "school_match_score", "rev_jw_score"])
    n_rev_users_expr = (
        f"COALESCE(CAST(src.{n_rev_users_col} AS BIGINT), 0)"
        if n_rev_users_col is not None
        else "0::BIGINT"
    )
    match_score_expr = (
        f"COALESCE(CAST(src.{match_score_col} AS DOUBLE), 1.0)"
        if match_score_col is not None
        else "1.0::DOUBLE"
    )

    return f"""
    WITH base AS (
        SELECT
            src.f1_school_name,
            src.f1_instname_clean,
            src.rev_university_raw,
            src.rev_instname_clean,
            {n_rev_users_expr} AS n_rev_users,
            {match_score_expr} AS match_score
        FROM {cw_tab} AS src
        WHERE src.f1_school_name IS NOT NULL
          AND src.rev_instname_clean IS NOT NULL
          AND trim(src.rev_instname_clean) != ''
    ),
    raw_ranked AS (
        SELECT *,
            ROW_NUMBER() OVER(
                PARTITION BY f1_school_name, rev_instname_clean
                ORDER BY n_rev_users DESC, match_score DESC, rev_university_raw
            ) AS raw_variant_rank
        FROM base
    ),
    families AS (
        SELECT
            f1_school_name,
            MAX(f1_instname_clean) AS f1_instname_clean,
            rev_instname_clean,
            MAX(CASE WHEN raw_variant_rank = 1 THEN rev_university_raw END) AS rev_university_raw,
            SUM(n_rev_users) AS n_rev_users,
            COUNT(DISTINCT rev_university_raw) AS n_rev_university_raw_variants,
            MAX(match_score) AS match_score
        FROM raw_ranked
        GROUP BY f1_school_name, rev_instname_clean
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER(
                PARTITION BY f1_school_name
                ORDER BY match_score DESC, n_rev_users DESC, rev_instname_clean
            ) AS school_match_rank,
            MAX(match_score) OVER(PARTITION BY f1_school_name) AS top_match_score
        FROM families
    ),
    filtered AS (
        SELECT *,
            top_match_score - match_score AS match_score_gap_from_top
        FROM ranked
        WHERE school_match_rank = 1
           OR match_score >= top_match_score - {SCORE_GAP}
    )
    SELECT
        f1_school_name,
        f1_instname_clean,
        rev_university_raw,
        rev_instname_clean,
        n_rev_users,
        match_score,
        n_rev_university_raw_variants,
        school_match_rank,
        match_score_gap_from_top,
        CASE
            WHEN COUNT(*) OVER(PARTITION BY f1_school_name) > 1 THEN 1
            ELSE 0
        END AS match_ambiguous_ind
    FROM filtered
    """


def _f1_merge_stage_counts(query: str, con=con_f1) -> dict:
    """Compute summary stats for a merge stage query."""
    df = con.sql(f"""
        SELECT
            COUNT(*)                AS n_rows,
            COUNT(DISTINCT spell_id) AS n_spells,
            COUNT(DISTINCT person_id) AS n_persons,
            COUNT(DISTINCT user_id)   AS n_users
        FROM ({query})
    """).df().iloc[0]
    mult = round(df["n_rows"] / max(1, df["n_spells"]), 2)
    return {
        "n_rows":    int(df["n_rows"]),
        "n_spells":  int(df["n_spells"]),
        "n_persons": int(df["n_persons"]),
        "n_users":   int(df["n_users"]),
        "mult":      mult,
    }


def _print_merge_stage(label: str, counts: dict) -> None:
    print(
        f"  {label:<20s}: "
        f"{counts['n_rows']:>10,} rows | "
        f"{counts['n_spells']:>8,} spells | "
        f"{counts['n_persons']:>8,} persons | "
        f"{counts['n_users']:>8,} users | "
        f"{counts['mult']:>6.2f}x mult"
    )


def materialize_table(table_name: str, query: str, con=con_f1) -> int:
    t0 = time.perf_counter()
    con.sql(f"CREATE OR REPLACE TABLE {table_name} AS {query}")
    n = int(con.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    elapsed = time.perf_counter() - t0
    print(f"  Materialized {table_name}: {n:,} rows ({_fmt_elapsed(elapsed)})")
    return n


def write_query_to_parquet(query: str, out_path: str, overwrite: bool = False, con=con_f1) -> None:
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        if os.path.getsize(out_path) == 0:
            print(f"  Removing empty file: {out_path}")
            os.remove(out_path)
        elif not overwrite:
            print(f"  Skipping (exists): {out_path}")
            return
        else:
            os.remove(out_path)
    esc = _sql_escape_path(out_path)
    con.sql(f"COPY ({query}) TO '{esc}' (FORMAT parquet)")
    elapsed = time.perf_counter() - t0
    print(f"  Wrote: {out_path} ({_fmt_elapsed(elapsed)})")


# ---------------------------------------------------------------------------
# Stage 1: Build F1 education spell summary
# ---------------------------------------------------------------------------

def _build_f1_school_rows_query(
    f1_foia_tab: str,
    cw_tab: str,
    resolution_tab: str | None = None,
    school_block_mode: str = cfg.BUILD_SCHOOL_BLOCK_MODE,
) -> str:
    """Build row-level F1 school records with optional campus-location resolution.

    This stage preserves the raw F1 row grain long enough to exploit the
    row-level campus-location-aware school resolution available in the rich
    school artifact. In `campus_unique` mode, each F1 row is matched against
    the row-level school artifact on (school_name, campus_state, optional
    campus_city / campus_zip). If exactly one upstream f1_row_num remains, the
    row is marked as uniquely resolved; otherwise it falls back to school-name-
    only handling.

    In `off` mode, or when the rich artifact is unavailable, the stage reduces
    to a lightweight filter that retains only rows whose school_name appears in
    the legacy school crosswalk.
    """
    if school_block_mode != "campus_unique" or resolution_tab is None:
        return f"""
        WITH school_pre_counts AS (
            SELECT
                f1_school_name,
                COUNT(DISTINCT rev_instname_clean) AS school_match_count_pre
            FROM {cw_tab}
            GROUP BY f1_school_name
        )
        SELECT
            f.*,
            spc.school_match_count_pre,
            spc.school_match_count_pre AS school_match_count_post_row,
            NULL::BIGINT               AS resolved_f1_row_num_row,
            NULL::BIGINT               AS resolved_unitid_row
        FROM {f1_foia_tab} AS f
        JOIN school_pre_counts AS spc
            ON f.school_name = spc.f1_school_name
        """

    campus_city_clean = sql_normalize_expr("f.campus_city")
    campus_state_clean = sql_state_name_to_abbr_expr("f.campus_state")
    campus_zip_clean = sql_clean_zip_expr("f.campus_zip_code")

    return f"""
    WITH school_pre_counts AS (
        SELECT
            f1_school_name,
            COUNT(DISTINCT rev_instname_clean) AS school_match_count_pre
        FROM {cw_tab}
        GROUP BY f1_school_name
    ),
    resolution_pairs AS (
        SELECT DISTINCT
            sr.f1_school_name,
            CAST(sr.f1_row_num AS BIGINT) AS f1_row_num,
            CAST(sr.UNITID AS BIGINT)     AS UNITID,
            sr.f1_city_clean,
            sr.f1_state_clean,
            sr.f1_zip_clean,
            cw.rev_instname_clean
        FROM {resolution_tab} AS sr
        JOIN {cw_tab} AS cw
            ON sr.f1_school_name = cw.f1_school_name
           AND sr.rev_instname_clean = cw.rev_instname_clean
    ),
    row_post_counts AS (
        SELECT
            f1_row_num,
            COUNT(DISTINCT rev_instname_clean) AS school_match_count_post_row
        FROM resolution_pairs
        GROUP BY f1_row_num
    ),
    raw_rows AS (
        SELECT
            ROW_NUMBER() OVER() AS f1_merge_row_id,
            f.*,
            {campus_city_clean}  AS campus_city_clean,
            {campus_state_clean} AS campus_state_clean,
            {campus_zip_clean}   AS campus_zip_clean
        FROM {f1_foia_tab} AS f
        JOIN school_pre_counts AS spc
            ON f.school_name = spc.f1_school_name
    ),
    compatible_rows AS (
        SELECT DISTINCT
            rr.f1_merge_row_id,
            rp.f1_row_num,
            rp.UNITID
        FROM raw_rows AS rr
        JOIN resolution_pairs AS rp
            ON rr.school_name = rp.f1_school_name
            AND COALESCE(rr.campus_state_clean, '') != ''
            AND COALESCE(rr.campus_state_clean, '') = COALESCE(rp.f1_state_clean, '')
            AND (
                COALESCE(rr.campus_city_clean, '') = ''
                OR COALESCE(rr.campus_city_clean, '') = COALESCE(rp.f1_city_clean, '')
            )
            AND (
                COALESCE(rr.campus_zip_clean, '') = ''
                OR COALESCE(rr.campus_zip_clean, '') = COALESCE(rp.f1_zip_clean, '')
            )
    ),
    row_resolution AS (
        SELECT
            rr.f1_merge_row_id,
            CASE
                WHEN COUNT(DISTINCT cr.f1_row_num) = 1 THEN MIN(cr.f1_row_num)
                ELSE NULL
            END AS resolved_f1_row_num_row,
            CASE
                WHEN COUNT(DISTINCT cr.f1_row_num) = 1 THEN MIN(cr.UNITID)
                ELSE NULL
            END AS resolved_unitid_row
        FROM raw_rows AS rr
        LEFT JOIN compatible_rows AS cr
            ON rr.f1_merge_row_id = cr.f1_merge_row_id
        GROUP BY rr.f1_merge_row_id
    )
    SELECT
        rr.* EXCLUDE (campus_city_clean, campus_state_clean, campus_zip_clean),
        spc.school_match_count_pre,
        CASE
            WHEN rrn.resolved_f1_row_num_row IS NOT NULL
                THEN COALESCE(rpc.school_match_count_post_row, spc.school_match_count_pre)
            ELSE spc.school_match_count_pre
        END AS school_match_count_post_row,
        rrn.resolved_f1_row_num_row,
        rrn.resolved_unitid_row
    FROM raw_rows AS rr
    JOIN school_pre_counts AS spc
        ON rr.school_name = spc.f1_school_name
    LEFT JOIN row_resolution AS rrn
        ON rr.f1_merge_row_id = rrn.f1_merge_row_id
    LEFT JOIN row_post_counts AS rpc
        ON rrn.resolved_f1_row_num_row = rpc.f1_row_num
    """


def _build_f1_educ_spells_query(f1_school_rows_tab: str) -> str:
    """Collapse row-level F1 school records → education spell level.

    Unit: (person_id, school_name, degree_level, program_start_year)
    Each person can have multiple spells (e.g., Bachelor's then Master's).

    The input rows already encode whether campus location uniquely resolved the
    spell to one upstream f1_row_num. This stage conservatively applies school
    blocking only when every row in the spell resolves to the same f1_row_num.
    """
    return f"""
    WITH spell_rows AS (
        SELECT
            f.person_id,
            f.school_name,
            -- Standardize country_of_birth to Revelio Title Case format via crosswalk JOIN
            COALESCE(cw_c.std_country, initcap(trim(f.country_of_birth))) AS f1_country_std,
            CASE
                WHEN upper(f.student_edu_level_desc) = 'DOCTORATE' THEN 'Doctor'
                WHEN upper(f.student_edu_level_desc) LIKE '%MASTER%' THEN 'Master'
                WHEN upper(f.student_edu_level_desc) LIKE '%BACHELOR%' THEN 'Bachelor'
                WHEN upper(f.student_edu_level_desc) LIKE '%ASSOCIATE%' THEN 'Associate'
                ELSE 'Other'
            END AS f1_degree_level,
            YEAR(TRY_CAST(f.program_start_date AS DATE)) AS f1_prog_start_year,
            YEAR(TRY_CAST(f.program_end_date AS DATE))   AS f1_prog_end_year,
            f.major_1_cip_code,
            f.employment_opt_type,
            f.year,
            f.year_int,
            f.school_match_count_pre,
            f.school_match_count_post_row,
            f.resolved_f1_row_num_row,
            f.resolved_unitid_row
        FROM {f1_school_rows_tab} AS f
        -- Join to country crosswalk for vectorized country normalization (no UDF per row)
        LEFT JOIN _country_cw AS cw_c ON upper(trim(f.country_of_birth)) = cw_c.key_upper
        WHERE f.person_id IS NOT NULL
          AND f.country_of_birth IS NOT NULL
          AND f.school_name IS NOT NULL
    ),
    spells AS (
        SELECT
            person_id,
            school_name,
            f1_country_std,
            f1_degree_level,
            f1_prog_start_year,
            MAX(f1_prog_end_year) AS f1_prog_end_year,
            -- CIP code (modal value across rows within the spell)
            MODE(major_1_cip_code) AS f1_cip6,
            -- 2-digit CIP family (integer part before decimal, divided by 100)
            CASE
                WHEN MODE(major_1_cip_code) IS NULL THEN NULL
                ELSE TRY_CAST(
                    FLOOR(TRY_CAST(REPLACE(MODE(major_1_cip_code), '-', '.') AS FLOAT))
                    AS INTEGER)
            END AS f1_cip2,
            -- Employer type retained for diagnostics only
            MODE(employment_opt_type) AS f1_opt_type,
            -- Metadata
            COUNT(DISTINCT year) AS n_f1_years,
            MIN(year_int)        AS f1_year_min,
            MAX(year_int)        AS f1_year_max,
            CASE
                WHEN COUNT(*) = COUNT(resolved_f1_row_num_row)
                 AND COUNT(DISTINCT resolved_f1_row_num_row) = 1
                    THEN 'location_unique'
                ELSE 'name_only'
            END AS school_resolution_status,
            CASE
                WHEN COUNT(*) = COUNT(resolved_f1_row_num_row)
                 AND COUNT(DISTINCT resolved_f1_row_num_row) = 1
                    THEN MIN(resolved_f1_row_num_row)
                ELSE NULL
            END AS resolved_f1_row_num,
            CASE
                WHEN COUNT(*) = COUNT(resolved_f1_row_num_row)
                 AND COUNT(DISTINCT resolved_f1_row_num_row) = 1
                    THEN MIN(resolved_unitid_row)
                ELSE NULL
            END AS resolved_unitid,
            CASE
                WHEN COUNT(*) = COUNT(resolved_f1_row_num_row)
                 AND COUNT(DISTINCT resolved_f1_row_num_row) = 1
                    THEN 1
                ELSE 0
            END AS school_block_applied_ind,
            MAX(school_match_count_pre) AS school_match_count_pre,
            CASE
                WHEN COUNT(*) = COUNT(resolved_f1_row_num_row)
                 AND COUNT(DISTINCT resolved_f1_row_num_row) = 1
                    THEN MAX(school_match_count_post_row)
                ELSE MAX(school_match_count_pre)
            END AS school_match_count_post
        FROM spell_rows
        GROUP BY
            person_id,
            school_name,
            f1_country_std,
            f1_degree_level,
            f1_prog_start_year
    )
    SELECT
        *,
        -- Spell ID: stable identifier for (person_id, school, country, degree, prog_start)
        ROW_NUMBER() OVER(
            ORDER BY person_id, school_name, f1_country_std, f1_degree_level, f1_prog_start_year
        ) AS spell_id
    FROM spells
    """


# ---------------------------------------------------------------------------
# Stage 2: Build Revelio education × school summary
# ---------------------------------------------------------------------------

def _build_rev_educ_school_query(
    rev_educ_tab: str,
    cw_tab: str,
) -> str:
    """Build Revelio education × school table, crosswalked to F1 school names.

    Keeps ALL education records per (user_id, university_raw) — not just the
    highest degree. This allows Stage 3 to score each F1 spell against the
    best-matching Revelio education record (e.g., a user who got a BS then MS
    at the same school will have both records available, so an F1 spell for the
    BS years scores correctly on degree and date).

    Stage 4a's existing dedup (PARTITION BY spell_id, user_id ORDER BY
    total_score DESC) naturally picks the best-matching record downstream.

    This is the broad school_name-level join used both in `off` mode and as the
    fallback path for unresolved (`name_only`) spells in `campus_unique` mode.
    The school crosswalk is applied at the cleaned-school-family level
    (`rev_instname_clean`) so all raw Revelio variants in the retained family
    remain eligible while weaker family alternatives are pruned upstream.

    Only Revelio education records with a crosswalk match are retained (INNER
    JOIN). One Revelio school can map to multiple F1 schools, yielding multiple
    rows — all are kept as merge candidates.

    CIP2 is derived directly from the per-record field_clean column in rev_educ,
    giving the field of study for that specific degree rather than a user-level
    aggregate across all degrees.
    """
    cip2_sql = field_clean_to_cip2_sql("e.field_clean")
    rev_clean_expr = _sql_clean_inst_name_expr("e.university_raw")
    return f"""
    WITH educ_base AS (
        SELECT
            e.user_id,
            e.university_raw AS rev_university_raw,
            {rev_clean_expr} AS rev_instname_clean,
            e.degree_clean          AS rev_degree_clean,
            YEAR(TRY_CAST(e.ed_startdate AS DATE)) AS rev_educ_start_year,
            YEAR(TRY_CAST(e.ed_enddate AS DATE))   AS rev_educ_end_year,
            e.ed_startdate AS rev_educ_start_raw,
            e.ed_enddate   AS rev_educ_end_raw,
            {cip2_sql} AS rev_cip2
        FROM {rev_educ_tab} AS e
        WHERE e.university_raw IS NOT NULL
          AND e.degree_clean NOT IN ('Non-Degree', 'High School')
    )
    SELECT
        eb.user_id,
        eb.rev_university_raw,
        eb.rev_instname_clean,
        cw.f1_school_name,
        cw.match_score          AS school_match_score,
        cw.match_ambiguous_ind  AS school_match_ambiguous_ind,
        eb.rev_degree_clean,
        eb.rev_educ_start_year,
        eb.rev_educ_end_year,
        eb.rev_educ_start_raw,
        eb.rev_educ_end_raw,
        eb.rev_cip2
    FROM educ_base AS eb
    JOIN {cw_tab} AS cw
        ON eb.rev_instname_clean = cw.rev_instname_clean
       AND eb.rev_instname_clean IS NOT NULL
    """


def _build_rev_educ_school_row_query(
    rev_educ_school_tab: str,
    cw_tab: str,
    resolution_tab: str,
) -> str:
    """Expand the legacy Revelio school table to row-level f1_row_num mappings.

    This is used only for `location_unique` spells. The mapping expands each
    resolved row to the kept cleaned Revelio school family (`rev_instname_clean`),
    so blocked candidates remain a subset of the baseline school-name candidate
    universe without depending on one exact raw-string variant.
    """
    return f"""
    WITH row_map AS (
        SELECT DISTINCT
            sr.f1_school_name,
            cw.rev_instname_clean,
            CAST(sr.f1_row_num AS BIGINT) AS f1_row_num
        FROM {resolution_tab} AS sr
        JOIN {cw_tab} AS cw
            ON sr.f1_school_name = cw.f1_school_name
           AND sr.rev_instname_clean = cw.rev_instname_clean
    )
    SELECT DISTINCT
        re.user_id,
        re.rev_university_raw,
        re.rev_instname_clean,
        re.f1_school_name,
        rm.f1_row_num,
        re.school_match_score,
        re.school_match_ambiguous_ind,
        re.rev_degree_clean,
        re.rev_educ_start_year,
        re.rev_educ_end_year,
        re.rev_educ_start_raw,
        re.rev_educ_end_raw,
        re.rev_cip2
    FROM {rev_educ_school_tab} AS re
    JOIN row_map AS rm
        ON re.f1_school_name = rm.f1_school_name
       AND re.rev_instname_clean = rm.rev_instname_clean
    """


# ---------------------------------------------------------------------------
# Stage 1b: F1 employers (all — with and without entity crosswalk match)
# ---------------------------------------------------------------------------

def _build_f1_opt_employers_all_query(f1_foia_tab: str, emp_lookup_tab: str) -> str:
    """All distinct F1 employers per person, with only unambiguous lookup rcids.

    LEFT JOIN to the employer entity lookup so that ALL observed employers are
    retained — not just those that crosswalked to a Revelio rcid. Employers
    without a crosswalk match get rcid = NULL and will be handled by the
    fuzzy name-matching fallback in the employer sequence scoring stage.

    Employer lookup rows are first aggregated at the downstream join grain
    (employer_name + cleaned location). If that effective employer key maps to
    multiple rcids in the lookup artifact, rcid is nulled out here so the row
    is no longer treated as strong entity evidence downstream. The employer
    itself is still retained and can match via the fuzzy/subset name paths.

    max_f1_year: MAX(year_int) per person_id — used downstream to restrict
    Revelio positions to those started at or before this year.
    min_f1_year: MIN(year_int) per person_id — used downstream to restrict
    Revelio positions to those ending at or after this year.

    employer_name_normed: cleaned form with suffixes standardized, stopwords
    and common company words removed — used for JW and subset token matching.
    """
    join_clause = _sql_f1_emp_lookup_join_clause("f", "lu")
    normed_sql = _company_norm_sql("f.employer_name")
    return f"""
    WITH person_year_range AS (
        SELECT person_id,
               MIN(year_int) AS min_f1_year,
               MAX(year_int) AS max_f1_year
        FROM {f1_foia_tab}
        WHERE person_id IS NOT NULL
        GROUP BY person_id
    ),
    lookup_summary AS (
        SELECT
            employer_name,
            employer_city_clean,
            employer_state_clean,
            employer_zip_clean,
            MAX(CAST(lookup_rcid_count AS BIGINT))           AS lookup_rcid_count,
            MAX(CAST(lookup_rcid_ambiguous_ind AS INTEGER))  AS lookup_rcid_ambiguous_ind,
            MAX(CAST(lookup_has_direct_ind AS INTEGER))      AS lookup_has_direct_ind,
            CASE
                WHEN MAX(CAST(lookup_rcid_ambiguous_ind AS INTEGER)) = 1 THEN NULL
                WHEN COUNT(DISTINCT rcid) FILTER (WHERE rcid IS NOT NULL) = 1
                    THEN MAX(CAST(rcid AS BIGINT))
                ELSE NULL
            END AS rcid
        FROM {emp_lookup_tab}
        GROUP BY employer_name, employer_city_clean, employer_state_clean, employer_zip_clean
    )
    SELECT DISTINCT
        f.person_id,
        lower(trim(f.employer_name))  AS employer_name_clean,
        {normed_sql}                  AS employer_name_normed,
        lu.rcid,
        COALESCE(lu.lookup_rcid_count, 0)          AS lookup_rcid_count,
        COALESCE(lu.lookup_rcid_ambiguous_ind, 0)  AS lookup_rcid_ambiguous_ind,
        COALESCE(lu.lookup_has_direct_ind, 0)      AS lookup_has_direct_ind,
        py.min_f1_year,
        py.max_f1_year
    FROM {f1_foia_tab} AS f
    LEFT JOIN lookup_summary AS lu
        ON {join_clause}
    LEFT JOIN person_year_range AS py
        ON f.person_id = py.person_id
    WHERE f.employer_name IS NOT NULL
      AND trim(f.employer_name) != ''
    """


# ---------------------------------------------------------------------------
# Stage 2b: Revelio position history (full, filtered to candidate users)
# ---------------------------------------------------------------------------

def _build_rev_pos_full_query(rev_pos_tab: str, candidates_tab: str) -> str:
    """All distinct (user_id, rcid, company_raw_clean, company_raw_normed,
    pos_start_year, pos_end_year) from Revelio positions, restricted to
    user_ids that appear in merge_candidates.

    Filtering to candidate users keeps this table tractable; the full rev_pos
    dataset can be very large.

    Keep positions even when rcid is NULL: the entity path requires rcid, but
    the fuzzy-name and subset-token paths should still be able to use uncoded
    positions as employer evidence.

    company_raw_clean: lower+trim only — kept for display.
    company_raw_normed: fully cleaned (suffixes standardized, stopwords removed) — used
      for JW similarity and subset token matching.
    pos_start_year: YEAR(startdate) — used to filter positions to those started at or
      before each person's max F1 year.
    pos_end_year: YEAR(enddate) — used to filter positions to those ending at or after
      each person's min F1 year. NULL end dates are treated as the current year.
    """
    normed_sql = _company_norm_sql("rp.company_raw")
    return f"""
    SELECT DISTINCT
        rp.user_id,
        CAST(rp.rcid AS BIGINT)     AS rcid,
        lower(trim(rp.company_raw)) AS company_raw_clean,
        {normed_sql}                AS company_raw_normed,
        YEAR(TRY_CAST(rp.startdate AS DATE)) AS pos_start_year,
        YEAR(TRY_CAST(rp.enddate   AS DATE)) AS pos_end_year
    FROM {rev_pos_tab} AS rp
    WHERE rp.user_id IN (SELECT DISTINCT user_id FROM {candidates_tab})
      AND rp.company_raw IS NOT NULL
    """


def _build_emp_idf_query(rev_pos_tab: str, smoothing: float) -> str:
    """Per-rcid IDF weights computed over ALL Revelio positions (global employer frequency).

    idf_weight = 1 / log(smoothing + n_distinct_users_at_rcid)

    Uses the full rev_pos (not just candidate users) so that IDF reflects
    how discriminative an employer is across the entire Revelio universe.
    """
    return f"""
    SELECT
        CAST(rcid AS BIGINT) AS rcid,
        COUNT(DISTINCT user_id)                                     AS n_rev_users,
        1.0 / LOG({smoothing} + COUNT(DISTINCT user_id)::FLOAT)     AS idf_weight
    FROM {rev_pos_tab}
    WHERE rcid IS NOT NULL
      AND user_id IS NOT NULL
    GROUP BY rcid
    """


def _build_token_idf_query(rev_pos_tab: str, min_idf: float) -> str:
    """Per-token IDF weights computed over ALL Revelio company names (global token frequency).

    token_idf = 1 / log(1 + n_distinct_companies_containing_token)

    Uses the full rev_pos corpus (not just candidate users) so that token IDF
    reflects global company name frequency. Only tokens with idf >= min_idf are
    retained — this excludes ultra-common tokens (e.g. 'the', 'and') that survive
    normalization and would dilute subset matching scores.

    Company names are normalized before tokenization (same _company_norm_sql applied
    at query-build time) so tokens match what appears in rev_pos_full.company_raw_normed.
    """
    normed_sql = _company_norm_sql("company_raw")
    return f"""
    WITH normed_companies AS (
        SELECT DISTINCT {normed_sql} AS company_normed
        FROM {rev_pos_tab}
        WHERE company_raw IS NOT NULL
    ),
    token_counts AS (
        SELECT token, COUNT(*) AS n_companies
        FROM normed_companies,
             UNNEST(string_split(company_normed, ' ')) t(token)
        WHERE token != ''
        GROUP BY token
    )
    SELECT
        token,
        n_companies,
        1.0 / LOG(1.0 + n_companies::FLOAT) AS token_idf
    FROM token_counts
    WHERE 1.0 / LOG(1.0 + n_companies::FLOAT) >= {min_idf}
    """


# ---------------------------------------------------------------------------
# Stage 3: Candidate generation (hard-filter cross-join)
# ---------------------------------------------------------------------------

def _build_candidates_query(
    f1_spells_tab: str,
    rev_educ_school_tab: str,
    rev_indiv_tab: str,
    rev_educ_school_row_tab: str | None = None,
    SCHOOL_MATCH_THRESHOLD: float = cfg.BUILD_SCHOOL_MATCH_THRESHOLD,
    YEAR_HARD_BUFFER: int = cfg.BUILD_YEAR_HARD_BUFFER,
    DEGREE_SCORE_NULL_DEFAULT: float = cfg.BUILD_DEGREE_SCORE_NULL_DEFAULT,
    DATE_SCORE_NULL_DEFAULT: float = cfg.BUILD_DATE_SCORE_NULL_DEFAULT,
) -> str:
    """Hard-filter join: F1 spells × Revelio education → (spell_id, user_id) candidates.

    Join conditions (all enforced as hard filters — necessary to keep cross-join tractable
    and to eliminate obviously wrong matches):
      1. School:  f1_school_name matches via crosswalk, score > threshold
      2. Country: ri.country_std = f1.f1_country_std
      3. Degree:  levels must agree; 'Other'/'Missing'/'Non-Degree' pass through
      4. Year:    |f1_prog_start_year - rev_educ_start_year| <= YEAR_HARD_BUFFER (NULL passes through)

    Soft scoring signals are computed here and combined in _build_merge_scored_query:
      country_score: name-model confidence in the matched nationality (NOT country match itself)
      degree_score:  1.0 if levels match exactly, null_default if either unknown
      date_score:    linear decay within the YEAR_HARD_BUFFER window (0.5 if either NULL)
      field_score:   1.0 if 2-digit CIP matches, 0.5 if either unknown, 0.0 otherwise

    n_match_raw: count of distinct candidates per spell_id.
    """
    # country_score: confidence of Revelio name-nationality model for the matched country
    country_score_expr = """
        CASE
            WHEN ri.nanat_score IS NULL AND ri.nanat_subregion_score IS NULL
                AND ri.nt_subregion_score IS NULL
                THEN 0.5
            WHEN COALESCE(ri.country_uncertain_ind, 0) = 1
                THEN 0.7 * LEAST(1.0, GREATEST(
                    COALESCE(ri.nanat_subregion_score, 0),
                    COALESCE(ri.nt_subregion_score, 0)
                ))
            ELSE
                LEAST(1.0,
                    0.4 * LEAST(1.0, GREATEST(
                        COALESCE(ri.nanat_subregion_score, 0),
                        COALESCE(ri.nt_subregion_score, 0)
                    )) +
                    0.6 * COALESCE(ri.nanat_score, 0.5)
                )
        END"""

    degree_score_expr = f"""
        CASE
            WHEN f1.f1_degree_level IS NULL OR f1.f1_degree_level = 'Other'
                 OR re.rev_degree_clean IS NULL OR re.rev_degree_clean = 'Missing'
                THEN {DEGREE_SCORE_NULL_DEFAULT}
            WHEN f1.f1_degree_level = 'Doctor'    AND re.rev_degree_clean = 'Doctor'    THEN 1.0
            WHEN f1.f1_degree_level = 'Master'    AND re.rev_degree_clean IN ('Master', 'MBA') THEN 1.0
            WHEN f1.f1_degree_level = 'Bachelor'  AND re.rev_degree_clean = 'Bachelor'  THEN 1.0
            WHEN f1.f1_degree_level = 'Associate' AND re.rev_degree_clean = 'Associate' THEN 1.0
            ELSE {DEGREE_SCORE_NULL_DEFAULT}
        END"""

    # date_score: linear decay within ±YEAR_HARD_BUFFER window; 0.5 if either year is NULL
    date_score_expr = f"""
        CASE
            WHEN f1.f1_prog_start_year IS NULL OR re.rev_educ_start_year IS NULL
                THEN {DATE_SCORE_NULL_DEFAULT}
            ELSE 1.0 - ABS(f1.f1_prog_start_year - re.rev_educ_start_year)::FLOAT
                       / ({YEAR_HARD_BUFFER} + 1.0)
        END"""

    field_score_expr = """
        CASE
            WHEN f1.f1_cip2 IS NULL OR re.rev_cip2 IS NULL THEN 0.5
            WHEN f1.f1_cip2 = re.rev_cip2 THEN 1.0
            ELSE 0.0
        END"""

    def _candidate_branch_sql(
        re_tab: str,
        school_join_sql: str,
        school_match_f1_row_sql: str,
        spell_status_sql: str,
    ) -> str:
        return f"""
        SELECT
            f1.spell_id,
            f1.person_id,
            f1.school_name             AS f1_school_name,
            re.rev_university_raw,
            re.school_match_score,
            re.school_match_ambiguous_ind,
            f1.f1_country_std,
            f1.f1_degree_level,
            f1.f1_prog_start_year,
            f1.f1_prog_end_year,
            f1.f1_cip6,
            f1.f1_cip2,
            f1.f1_opt_type,
            f1.n_f1_years,
            f1.f1_year_min,
            f1.f1_year_max,
            f1.school_resolution_status,
            f1.resolved_f1_row_num,
            f1.resolved_unitid,
            f1.school_block_applied_ind,
            f1.school_match_count_pre,
            f1.school_match_count_post,
            ri.user_id,
            ri.fullname,
            ri.country                 AS rev_country,
            ri.subregion,
            ri.nanat_score,
            ri.nanat_subregion_score,
            ri.nt_subregion_score,
            ri.country_uncertain_ind,
            ri.est_yob,
            ri.stem_ind,
            ri.f_prob                  AS f_prob_avg,
            ri.fields                  AS rev_fields,
            ri.highest_ed_level        AS rev_highest_ed_level,
            re.rev_degree_clean,
            re.rev_educ_start_year,
            re.rev_educ_end_year,
            re.rev_educ_start_raw,
            re.rev_educ_end_raw,
            {school_match_f1_row_sql}  AS school_match_f1_row_num,
            re.rev_cip2,
            ({country_score_expr})     AS country_score,
            ({degree_score_expr})      AS degree_score,
            ({date_score_expr})        AS date_score,
            ({field_score_expr})       AS field_score
        FROM {f1_spells_tab} AS f1
        JOIN {re_tab} AS re
            ON {school_join_sql}
            AND re.school_match_score > {SCHOOL_MATCH_THRESHOLD}
            AND (
                f1.f1_degree_level = re.rev_degree_clean
                OR f1.f1_degree_level = 'Other'
                OR re.rev_degree_clean IN ('Missing', 'Non-Degree')
                OR (f1.f1_degree_level = 'Master' AND re.rev_degree_clean = 'MBA')
            )
            AND (
                f1.f1_prog_start_year IS NULL
                OR re.rev_educ_start_year IS NULL
                OR ABS(f1.f1_prog_start_year - re.rev_educ_start_year) <= {YEAR_HARD_BUFFER}
            )
        JOIN {rev_indiv_tab} AS ri
            ON re.user_id = ri.user_id
           AND ri.country_std = f1.f1_country_std
        WHERE {spell_status_sql}
        """

    if rev_educ_school_row_tab is None:
        broad_branch = _candidate_branch_sql(
            re_tab=rev_educ_school_tab,
            school_join_sql="f1.school_name = re.f1_school_name",
            school_match_f1_row_sql="NULL::BIGINT",
            spell_status_sql="1 = 1",
        )
        return f"""
        WITH candidate_union AS (
            {broad_branch}
        )
        SELECT
            cu.*,
            COUNT(*) OVER(PARTITION BY cu.spell_id) AS n_match_raw
        FROM candidate_union AS cu
        """

    broad_branch = _candidate_branch_sql(
        re_tab=rev_educ_school_tab,
        school_join_sql="f1.school_name = re.f1_school_name",
        school_match_f1_row_sql="NULL::BIGINT",
        spell_status_sql="f1.school_resolution_status != 'location_unique'",
    )
    blocked_branch = _candidate_branch_sql(
        re_tab=rev_educ_school_row_tab,
        school_join_sql=(
            "f1.resolved_f1_row_num IS NOT NULL "
            "AND re.f1_row_num = f1.resolved_f1_row_num"
        ),
        school_match_f1_row_sql="re.f1_row_num",
        spell_status_sql="f1.school_resolution_status = 'location_unique'",
    )
    return f"""
    WITH candidate_union AS (
        {broad_branch}
        UNION ALL
        {blocked_branch}
    )
    SELECT
        cu.*,
        COUNT(*) OVER(PARTITION BY cu.spell_id) AS n_match_raw
    FROM candidate_union AS cu
    """


# ---------------------------------------------------------------------------
# Stage 3b: Employer sequence scoring
# ---------------------------------------------------------------------------

def _build_match_pairs_query(
    candidates_tab: str,
    f1_opt_emp_tab: str,
    rev_pos_full_tab: str,
    emp_idf_tab: str,
    token_idf_tab: str,
    EMP_FUZZY_THRESHOLD: float = cfg.BUILD_EMP_FUZZY_THRESHOLD,
    EMP_IDF_SMOOTHING: float = cfg.BUILD_EMP_IDF_SMOOTHING,
    EMP_SUBSET_MATCH_THRESHOLD: float = cfg.BUILD_EMP_SUBSET_MATCH_THRESHOLD,
    EMP_SUBSET_MIN_TOKENS: int = cfg.BUILD_EMP_SUBSET_MIN_TOKENS,
) -> str:
    """Build bipartite match pairs across all three employer matching paths.

    Returns SQL producing one row per (person_id, user_id, employer_key, rev_key)
    with the best match_score across all paths, where:
      - employer_key = COALESCE(employer_name_normed, employer_name_clean): the
        deduplicated logical F1 employer — multiple raw name variants that normalize
        to the same string are treated as a single employer.
      - rev_key = Revelio company identifier (rcid as VARCHAR for entity path;
        company_raw_normed for fuzzy/subset paths).

    This table feeds _solve_employer_assignment(), which uses the Hungarian
    algorithm to assign each Revelio company to at most one F1 employer,
    maximizing total IDF-weighted match score.

    Three matching paths, all producing continuous scores in [0, 1]:

      Path 1 (entity): F1 employer has rcid AND user has that rcid in rev_pos.
        Score = 1.0 (unambiguous entity link).

      Path 2 (fuzzy):  F1 employer has no rcid; JW(normed names) >= threshold.
        Score = max JW similarity per (F1 employer, Revelio company) pair.

      Path 3 (subset): F1 employer has no rcid; token overlap score >= threshold.
        Score = overlap_idf / min(f1_name_idf, rev_name_idf), where idf weights
        come from token_idf_tab (token frequency across all Revelio companies).
        Only applied when the shorter name has >= EMP_SUBSET_MIN_TOKENS tokens.

    Year filter: Revelio positions are restricted to pos_start_year <= max_f1_year
    AND COALESCE(pos_end_year, current_year) >= min_f1_year (i.e. the position
    must overlap with the person's F1 activity window).
    IDF fallback: employers not in emp_idf get idf = 1/log(smoothing + 1).
    Token IDF fallback: tokens not in token_idf_tab get token_idf = 1/log(2).
    """
    default_idf = f"1.0 / LOG({EMP_IDF_SMOOTHING} + 1.0)"
    default_token_idf = "1.0 / LOG(2.0)"   # token appearing in 1 company
    return f"""
    WITH
    -- Distinct (person_id, user_id) pairs from candidates
    candidate_pairs AS (
        SELECT DISTINCT person_id, user_id
        FROM {candidates_tab}
    ),

    -- All F1 employers for each candidate pair, with IDF weight + year range
    f1_emp_for_candidates AS (
        SELECT DISTINCT
            cp.person_id,
            cp.user_id,
            e.employer_name_clean,
            e.employer_name_normed,
            e.rcid,
            e.min_f1_year,
            e.max_f1_year,
            COALESCE(idf.idf_weight, {default_idf}) AS idf_weight
        FROM candidate_pairs AS cp
        JOIN {f1_opt_emp_tab} AS e ON cp.person_id = e.person_id
        LEFT JOIN {emp_idf_tab} AS idf ON CAST(e.rcid AS BIGINT) = idf.rcid
    ),

    -- Path 1: entity-ID match (rcid match) → score = 1.0.
    -- An rcid match is an unambiguous entity link; using JW on names would
    -- penalise pairs where the company raw name differs from the F1 record
    -- (e.g. abbreviated vs. full legal name) despite being the same employer.
    -- Year filter: only positions started at or before the person's last F1 year.
    entity_match_raw AS (
        SELECT DISTINCT
            f.person_id,
            f.user_id,
            f.employer_name_clean,
            CAST(f.rcid AS VARCHAR) AS rev_key,
            1.0                     AS raw_score
        FROM f1_emp_for_candidates AS f
        JOIN {rev_pos_full_tab} AS rp
            ON f.user_id = rp.user_id
            AND CAST(f.rcid AS BIGINT) = rp.rcid
            AND (rp.pos_start_year IS NULL OR rp.pos_start_year <= f.max_f1_year)
            AND COALESCE(rp.pos_end_year, YEAR(CURRENT_DATE)) >= f.min_f1_year
        WHERE f.rcid IS NOT NULL
    ),

    -- Path 2: fuzzy name match for employers without a crosswalk rcid.
    -- Score = actual JW similarity (not binary); year filter applied.
    -- One row per (person, user, F1 employer, Revelio company) pair.
    fuzzy_match_raw AS (
        SELECT
            f.person_id,
            f.user_id,
            f.employer_name_clean,
            rp.company_raw_normed AS rev_key,
            MAX(jaro_winkler_similarity(
                COALESCE(f.employer_name_normed, f.employer_name_clean),
                COALESCE(rp.company_raw_normed,  rp.company_raw_clean)
            )) AS raw_score
        FROM f1_emp_for_candidates AS f
        JOIN {rev_pos_full_tab} AS rp
            ON f.user_id = rp.user_id
            AND jaro_winkler_similarity(
                    COALESCE(f.employer_name_normed, f.employer_name_clean),
                    COALESCE(rp.company_raw_normed,  rp.company_raw_clean)
                ) >= {EMP_FUZZY_THRESHOLD}
            AND (rp.pos_start_year IS NULL OR rp.pos_start_year <= f.max_f1_year)
            AND COALESCE(rp.pos_end_year, YEAR(CURRENT_DATE)) >= f.min_f1_year
        WHERE f.rcid IS NULL
          AND f.employer_name_normed IS NOT NULL
          AND rp.company_raw_normed IS NOT NULL
        GROUP BY f.person_id, f.user_id, f.employer_name_clean, rp.company_raw_normed
    ),

    -- Path 3: subset token matching for employers without a crosswalk rcid.
    -- Score = overlap_token_idf / min(f1_name_token_idf, rev_name_token_idf).
    -- Only for names with >= EMP_SUBSET_MIN_TOKENS tokens after normalization.

    -- Tokenize F1 employer normed names (rcid IS NULL only).
    -- Carry min/max_f1_year from f1_emp_for_candidates so the year filter can be
    -- applied directly in token_overlap_raw without a correlated subquery.
    f1_emp_tokens AS (
        SELECT DISTINCT
            f.person_id,
            f.user_id,
            f.employer_name_clean,
            f.employer_name_normed,
            f.min_f1_year,
            f.max_f1_year,
            tok.token
        FROM f1_emp_for_candidates AS f,
             UNNEST(string_split(COALESCE(f.employer_name_normed, ''), ' ')) tok(token)
        WHERE f.rcid IS NULL
          AND f.employer_name_normed IS NOT NULL
          AND tok.token != ''
    ),

    -- Tokenize rev_pos normed names (candidate users only, from rev_pos_full)
    rev_pos_tokens AS (
        SELECT DISTINCT
            rp.user_id,
            rp.company_raw_normed,
            rp.pos_start_year,
            rp.pos_end_year,
            tok.token
        FROM {rev_pos_full_tab} AS rp,
             UNNEST(string_split(COALESCE(rp.company_raw_normed, ''), ' ')) tok(token)
        WHERE rp.company_raw_normed IS NOT NULL
          AND tok.token != ''
    ),

    -- Token IDF totals per F1 employer name (denominator for subset score)
    f1_emp_token_totals AS (
        SELECT
            person_id,
            employer_name_clean,
            employer_name_normed,
            COUNT(DISTINCT token)                              AS n_tokens,
            SUM(COALESCE(ti.token_idf, {default_token_idf}))  AS f1_idf_total
        FROM (SELECT DISTINCT person_id, employer_name_clean, employer_name_normed, token
              FROM f1_emp_tokens)
        LEFT JOIN {token_idf_tab} AS ti USING (token)
        GROUP BY person_id, employer_name_clean, employer_name_normed
    ),

    -- Token IDF totals per Revelio company name (denominator for subset score).
    -- Reuse rev_pos_tokens to avoid re-tokenizing.
    rev_pos_token_totals AS (
        SELECT
            company_raw_normed,
            SUM(COALESCE(ti.token_idf, {default_token_idf})) AS rev_idf_total
        FROM (SELECT DISTINCT company_raw_normed, token FROM rev_pos_tokens)
        LEFT JOIN {token_idf_tab} AS ti USING (token)
        GROUP BY company_raw_normed
    ),

    -- Distinct overlapping tokens per (person, user, f1_employer, rev_company) pair.
    -- Deduping at the token level prevents repeated positions at the same company
    -- from inflating the subset score above 1.0.
    token_overlap_tokens AS (
        SELECT DISTINCT
            fe.person_id,
            fe.user_id,
            fe.employer_name_clean,
            rpt.company_raw_normed,
            fe.token
        FROM f1_emp_tokens AS fe
        JOIN rev_pos_tokens AS rpt
            ON rpt.user_id = fe.user_id
            AND rpt.token  = fe.token
            AND (rpt.pos_start_year IS NULL OR rpt.pos_start_year <= fe.max_f1_year)
            AND COALESCE(rpt.pos_end_year, YEAR(CURRENT_DATE)) >= fe.min_f1_year
    ),

    -- Token overlap IDF per (person, user, f1_employer, rev_company) pair.
    token_overlap_raw AS (
        SELECT
            tot.person_id,
            tot.user_id,
            tot.employer_name_clean,
            tot.company_raw_normed,
            SUM(COALESCE(ti.token_idf, {default_token_idf})) AS overlap_idf
        FROM token_overlap_tokens AS tot
        LEFT JOIN {token_idf_tab} AS ti ON tot.token = ti.token
        GROUP BY tot.person_id, tot.user_id, tot.employer_name_clean, tot.company_raw_normed
    ),

    -- Subset score per (person, user, F1 employer, Revelio company).
    -- token_overlap_raw is already at this grain, so no MAX/GROUP BY needed —
    -- just filter by threshold and token count.
    subset_match_raw AS (
        SELECT
            tor.person_id,
            tor.user_id,
            tor.employer_name_clean,
            tor.company_raw_normed AS rev_key,
            tor.overlap_idf / NULLIF(LEAST(fi.f1_idf_total, ri.rev_idf_total), 0) AS raw_score
        FROM token_overlap_raw AS tor
        JOIN f1_emp_token_totals AS fi
            ON tor.person_id = fi.person_id
            AND tor.employer_name_clean = fi.employer_name_clean
        JOIN rev_pos_token_totals AS ri
            ON tor.company_raw_normed = ri.company_raw_normed
        WHERE fi.n_tokens >= {EMP_SUBSET_MIN_TOKENS}
          AND tor.overlap_idf / NULLIF(LEAST(fi.f1_idf_total, ri.rev_idf_total), 0)
              >= {EMP_SUBSET_MATCH_THRESHOLD}
    )

    -- Best score per (person, user, logical F1 employer key, Revelio company).
    -- employer_key = normed name collapses raw-name variants of the same employer.
    SELECT
        amp.person_id,
        amp.user_id,
        COALESCE(e.employer_name_normed, e.employer_name_clean) AS employer_key,
        amp.rev_key,
        MAX(LEAST(1.0, GREATEST(0.0, amp.raw_score))) AS match_score
    FROM (
        SELECT person_id, user_id, employer_name_clean, rev_key, raw_score
        FROM entity_match_raw
        UNION ALL
        SELECT person_id, user_id, employer_name_clean, rev_key, raw_score
        FROM fuzzy_match_raw
        UNION ALL
        SELECT person_id, user_id, employer_name_clean, rev_key, raw_score
        FROM subset_match_raw
    ) AS amp
    JOIN f1_emp_for_candidates AS e
        ON amp.person_id        = e.person_id
        AND amp.user_id         = e.user_id
        AND amp.employer_name_clean = e.employer_name_clean
    GROUP BY amp.person_id, amp.user_id,
             COALESCE(e.employer_name_normed, e.employer_name_clean),
             amp.rev_key
    """


def _build_f1_keys_query(
    candidates_tab: str,
    f1_opt_emp_tab: str,
    emp_idf_tab: str,
    EMP_IDF_SMOOTHING: float = cfg.BUILD_EMP_IDF_SMOOTHING,
) -> str:
    """All deduplicated logical F1 employers per candidate pair, with IDF weights.

    employer_key = COALESCE(employer_name_normed, employer_name_clean).
    Multiple raw FOIA employer names that normalize to the same string are
    collapsed to one employer_key; idf_weight is the max across variants.

    This table is the denominator for n_f1_employers and employer_score.
    """
    default_idf = f"1.0 / LOG({EMP_IDF_SMOOTHING} + 1.0)"
    return f"""
    WITH
    candidate_pairs AS (
        SELECT DISTINCT person_id, user_id
        FROM {candidates_tab}
    ),
    f1_emp_for_candidates AS (
        SELECT DISTINCT
            cp.person_id,
            cp.user_id,
            e.employer_name_clean,
            e.employer_name_normed,
            e.rcid,
            COALESCE(idf.idf_weight, {default_idf}) AS idf_weight
        FROM candidate_pairs AS cp
        JOIN {f1_opt_emp_tab} AS e ON cp.person_id = e.person_id
        LEFT JOIN {emp_idf_tab} AS idf ON CAST(e.rcid AS BIGINT) = idf.rcid
    )
    SELECT
        person_id,
        user_id,
        COALESCE(employer_name_normed, employer_name_clean) AS employer_key,
        MAX(idf_weight)                                     AS idf_weight
    FROM f1_emp_for_candidates
    GROUP BY person_id, user_id, COALESCE(employer_name_normed, employer_name_clean)
    """


def _solve_employer_assignment(
    con,
    match_pairs_tab: str,
    f1_keys_tab: str,
) -> pd.DataFrame:
    """Solve max-weight bipartite employer assignment per (person_id, user_id).

    Each Revelio company (rev_key) is assigned to at most one logical F1 employer
    (employer_key). The assignment maximizes SUM(idf_weight * match_score) across
    all assigned pairs using scipy's implementation of the Hungarian algorithm.

    Returns DataFrame with columns (person_id, user_id, employer_key, match_score).
    """
    from scipy.optimize import linear_sum_assignment

    # Join match pairs with idf weights; idf_weight is constant per employer_key
    pairs_df = con.sql(f"""
        SELECT
            amp.person_id,
            amp.user_id,
            amp.employer_key,
            amp.rev_key,
            amp.match_score,
            fk.idf_weight
        FROM {match_pairs_tab} AS amp
        JOIN {f1_keys_tab} AS fk
            ON amp.person_id     = fk.person_id
            AND amp.user_id      = fk.user_id
            AND amp.employer_key = fk.employer_key
    """).df()

    if pairs_df.empty:
        return pd.DataFrame(
            columns=["person_id", "user_id", "employer_key", "match_score"]
        )

    results = []
    for (person_id, user_id), grp in pairs_df.groupby(["person_id", "user_id"]):
        f1_keys  = grp["employer_key"].unique()
        rev_keys = grp["rev_key"].unique()
        f1_idx   = {k: i for i, k in enumerate(f1_keys)}
        rev_idx  = {k: i for i, k in enumerate(rev_keys)}

        idf_by_key = (
            grp[["employer_key", "idf_weight"]]
            .drop_duplicates("employer_key")
            .set_index("employer_key")["idf_weight"]
        )

        # Cost matrix: rows = F1 employer keys, cols = Revelio companies.
        # Negate because scipy minimizes; maximize SUM(idf_weight * match_score).
        cost = np.zeros((len(f1_keys), len(rev_keys)))
        for _, row in grp.iterrows():
            i = f1_idx[row["employer_key"]]
            j = rev_idx[row["rev_key"]]
            bounded_score = float(np.clip(row["match_score"], 0.0, 1.0))
            weighted = idf_by_key[row["employer_key"]] * bounded_score
            if weighted > -cost[i, j]:   # keep the highest weighted score per cell
                cost[i, j] = -weighted

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 0:   # skip zero-weight (unmatched) assignments
                key = f1_keys[r]
                idf = idf_by_key[key]
                results.append({
                    "person_id":    person_id,
                    "user_id":      user_id,
                    "employer_key": key,
                    "match_score":  float(np.clip(((-cost[r, c] / idf) if idf > 0 else 0.0), 0.0, 1.0)),
                })

    return pd.DataFrame(
        results, columns=["person_id", "user_id", "employer_key", "match_score"]
    )


def _build_emp_scores_from_assignment_query(
    f1_keys_tab: str,
    assigned_pairs_tab: str,
) -> str:
    """Aggregate assigned employer pairs into per-(person, user) employer scores.

    Uses f1_keys_tab as the denominator (n_f1_employers, idf_weight sum) and
    left-joins the scipy-assigned pairs to get match_score per employer_key.

    employer_score = Σ(idf_weight_i × match_score_i) / Σ(idf_weight_i)
      where the sum is over all deduplicated logical F1 employers.
    """
    return f"""
    SELECT
        ak.person_id,
        ak.user_id,
        COUNT(*)                                                             AS n_f1_employers,
        SUM(CASE WHEN am.match_score > 0 THEN 1 ELSE 0 END)                 AS n_emp_matched,
        MAX(CASE WHEN am.match_score > 0 THEN 1 ELSE 0 END)                 AS has_any_emp_match,
        CASE
            WHEN SUM(ak.idf_weight) = 0 THEN NULL
            ELSE LEAST(
                1.0,
                GREATEST(
                    0.0,
                    SUM(ak.idf_weight * COALESCE(am.match_score, 0.0))
                        / SUM(ak.idf_weight)
                )
            )
        END                                                                  AS employer_score
    FROM {f1_keys_tab} AS ak
    LEFT JOIN {assigned_pairs_tab} AS am
        ON ak.person_id     = am.person_id
        AND ak.user_id      = am.user_id
        AND ak.employer_key = am.employer_key
    GROUP BY ak.person_id, ak.user_id
    """


# ---------------------------------------------------------------------------
# Stage 4: Combine candidates + employer scores → total_score
# ---------------------------------------------------------------------------

def _build_merge_scored_query(
    candidates_tab: str,
    emp_scores_tab: str,
    W_EMP_MAX:   float = cfg.BUILD_W_EMP_MAX,
    EMP_N_SCALE: float = cfg.BUILD_EMP_N_SCALE,
    W_COUNTRY:   float = cfg.BUILD_W_COUNTRY,
    W_DEGREE:    float = cfg.BUILD_W_DEGREE,
    W_DATE:      float = cfg.BUILD_W_DATE,
    W_FIELD:     float = cfg.BUILD_W_FIELD,
) -> str:
    """Join candidates with employer sequence scores and compute total_score.

    Employer weight uses exponential saturation so that it increases with the
    number of F1 employers (more evidence = employer signal dominates more):

        w_emp_eff = W_EMP_MAX * (1 - exp(-n_f1_employers / EMP_N_SCALE))

    Total score formula:
      With employer data (employer_score IS NOT NULL):
        school_match_score × (
          w_emp_eff × employer_score
          + (1 - w_emp_eff) × (w_country × cs + w_degree × ds + w_date × dt + w_field × fld)
                             / (w_country + w_degree + w_date + w_field)
        )
      Without employer data (employer_score = NULL, person has no employer records):
        school_match_score × (
          w_country × cs + w_degree × ds + w_date × dt + w_field × fld
        ) / (w_country + w_degree + w_date + w_field)

    emp_score_available_ind = 1 if the person had any employer data.
    w_emp_eff is included as an output column for diagnostics.
    """
    w_no_emp = W_COUNTRY + W_DEGREE + W_DATE + W_FIELD   # sum for renormalization
    # Inline SQL expression for variable employer weight (avoids repeating formula)
    _w_eff = (
        f"({W_EMP_MAX} * (1.0 - EXP(-CAST(es.n_f1_employers AS DOUBLE) / {EMP_N_SCALE})))"
    )
    return f"""
    SELECT
        mc.*,
        es.employer_score,
        COALESCE(es.n_f1_employers,    0) AS n_f1_employers,
        COALESCE(es.n_emp_matched,     0) AS n_emp_matched,
        COALESCE(es.has_any_emp_match, 0) AS has_any_emp_match,
        CASE WHEN es.employer_score IS NOT NULL THEN 1 ELSE 0 END AS emp_score_available_ind,
        -- Effective employer weight for this row (diagnostic)
        CASE WHEN es.employer_score IS NOT NULL THEN {_w_eff} ELSE 0.0 END AS w_emp_eff,
        -- Total score: school quality × weighted combination of all signals
        mc.school_match_score * CASE
            WHEN es.employer_score IS NOT NULL THEN
                {_w_eff} * es.employer_score
                + (1.0 - {_w_eff})
                    * (   {W_COUNTRY} * mc.country_score
                        + {W_DEGREE}  * mc.degree_score
                        + {W_DATE}    * mc.date_score
                        + {W_FIELD}   * mc.field_score
                      ) / {w_no_emp}
            ELSE
                (   {W_COUNTRY} * mc.country_score
                  + {W_DEGREE}  * mc.degree_score
                  + {W_DATE}    * mc.date_score
                  + {W_FIELD}   * mc.field_score
                ) / {w_no_emp}
        END AS total_score
    FROM {candidates_tab} AS mc
    LEFT JOIN {emp_scores_tab} AS es
        ON mc.person_id = es.person_id AND mc.user_id = es.user_id
    """


# ---------------------------------------------------------------------------
# Stage 4: Filtering and weighting
# ---------------------------------------------------------------------------

def _build_stage_match_filt_sql(source_tab: str) -> str:
    """Build stage_match_filt SQL: keep best Revelio education record per (spell_id, user_id).

    Country, degree, and year are now hard join filters applied upstream in
    _build_candidates_query, so no additional filtering is needed here.
    The dedup picks the best-scoring education record when a user has multiple
    Revelio education rows that survived the hard filters for the same spell.
    """
    return f"""
    WITH base AS (
        SELECT *,
            ROW_NUMBER() OVER(
                PARTITION BY spell_id, user_id
                ORDER BY total_score DESC, rev_degree_clean ASC, rev_educ_start_year ASC
            ) AS spell_user_rn
        FROM {source_tab}
    )
    SELECT * EXCLUDE spell_user_rn,
        COUNT(*) OVER(PARTITION BY spell_id) AS n_match_filt
    FROM base
    WHERE spell_user_rn = 1
    """


def _build_stage_weighted_sql(
    source_tab: str,
    person_user_dedup: bool = cfg.BUILD_PERSON_USER_DEDUP,
) -> str:
    """Build stage_weighted SQL: add weight_norm; optional person-level dedup."""
    dedup_clause = (
        "QUALIFY ROW_NUMBER() OVER(PARTITION BY person_id, user_id "
        "ORDER BY total_score DESC, spell_id) = 1"
        if person_user_dedup else ""
    )
    return f"""
    SELECT *,
        total_score / SUM(total_score) OVER(PARTITION BY spell_id) AS weight_norm
    FROM {source_tab}
    {dedup_clause}
    """


def _build_stage_final_sql(
    source_tab: str,
    AMBIGUITY_WEIGHT_GAP_CUTOFF: float = cfg.BUILD_AMBIGUITY_WEIGHT_GAP_CUTOFF,
    BAD_MATCH_GUARD_ENABLED: bool = cfg.BUILD_BAD_MATCH_GUARD_ENABLED,
    BAD_MATCH_GUARD_COUNTRY_SCORE_LT: float = cfg.BUILD_BAD_MATCH_GUARD_COUNTRY_SCORE_LT,
    BAD_MATCH_GUARD_TOTAL_SCORE_LT: float = cfg.BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT,
) -> str:
    """Build stage_final SQL: prune, renormalize, rank, and flag ambiguity by spell.

    Recomputes n_match_filt and weight_norm on the post-pruning candidate set so
    both fields reflect the final spell-level universe after person-user dedup
    and the bad-match guard.
    """
    bad_match_where = ""
    if BAD_MATCH_GUARD_ENABLED:
        bad_match_where = (
            f"WHERE NOT ("
            f"country_score < {BAD_MATCH_GUARD_COUNTRY_SCORE_LT} "
            f"AND total_score < {BAD_MATCH_GUARD_TOTAL_SCORE_LT}"
            f")"
        )
    return f"""
    WITH filtered AS (
        SELECT * FROM {source_tab}
        {bad_match_where}
    ),
    renormalized AS (
        SELECT
            * EXCLUDE (n_match_filt, weight_norm),
            COUNT(*) OVER(PARTITION BY spell_id) AS n_match_filt,
            CASE
                WHEN SUM(total_score) OVER(PARTITION BY spell_id) > 0 THEN
                    total_score / SUM(total_score) OVER(PARTITION BY spell_id)
                WHEN COUNT(*) OVER(PARTITION BY spell_id) > 0 THEN
                    1.0 / COUNT(*) OVER(PARTITION BY spell_id)
                ELSE NULL
            END AS weight_norm
        FROM filtered
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER(
                PARTITION BY spell_id
                ORDER BY weight_norm DESC, total_score DESC, country_score DESC, user_id
            ) AS match_rank,
            LEAD(weight_norm) OVER(
                PARTITION BY spell_id
                ORDER BY weight_norm DESC, total_score DESC, country_score DESC, user_id
            ) AS next_weight_norm
        FROM renormalized
    ),
    annotated AS (
        SELECT *,
            CASE
                WHEN match_rank = 1
                    THEN weight_norm - COALESCE(next_weight_norm, 0)
                ELSE NULL
            END AS top_weight_gap,
            CASE
                WHEN match_rank = 1
                 AND n_match_filt >= 2
                 AND weight_norm - COALESCE(next_weight_norm, 0) <= {AMBIGUITY_WEIGHT_GAP_CUTOFF}
                THEN 1 ELSE 0
            END AS top_match_ambiguous_ind
        FROM ranked
    )
    SELECT *,
        MAX(top_match_ambiguous_ind) OVER(PARTITION BY spell_id) AS spell_ambiguous_ind
    FROM annotated
    """


def _build_f1_merge_filt_stage_queries(merge_scored_tab: str, **kwargs) -> dict:
    """Build full nested filtering+weighting+ranking pipeline as a single SQL string.

    Used in the non-testing (production) path where we materialize the
    whole pipeline in one shot. Returns a dict with 'stage_final' key.

    Partitioning unit: spell_id (analogous to foia_indiv_id in H1B merge).
    """
    match_filt_q = _build_stage_match_filt_sql(f"({merge_scored_tab})")
    weighted_q = _build_stage_weighted_sql(f"({match_filt_q})", **{
        k: v for k, v in kwargs.items() if k == "person_user_dedup"
    })
    final_q = _build_stage_final_sql(f"({weighted_q})", **{
        k: v for k, v in kwargs.items()
        if k in ("AMBIGUITY_WEIGHT_GAP_CUTOFF", "BAD_MATCH_GUARD_ENABLED",
                 "BAD_MATCH_GUARD_COUNTRY_SCORE_LT", "BAD_MATCH_GUARD_TOTAL_SCORE_LT")
    })
    return {
        "stage_match_filt": match_filt_q,
        "stage_final":      final_q,
    }


# ---------------------------------------------------------------------------
# Stage 5: Strict filter
# ---------------------------------------------------------------------------

def _build_f1_stage_strict_query(
    baseline_tab: str,
    min_weight_norm: float = cfg.STRICT_MIN_WEIGHT_NORM,
    min_total_score: float = cfg.STRICT_MIN_TOTAL_SCORE,
    min_country_score: float = cfg.STRICT_MIN_COUNTRY_SCORE,
    max_n_match_filt=cfg.STRICT_MAX_N_MATCH_FILT,
) -> str:
    """Post-hoc high-precision filter on baseline.

    Keeps rank-1 matches satisfying all strict thresholds, trading recall
    for very low false-positive rates.
    """
    conditions = [
        "match_rank = 1",
        f"weight_norm >= {min_weight_norm}",
        f"total_score >= {min_total_score}",
        f"country_score >= {min_country_score}",
    ]
    if max_n_match_filt is not None:
        conditions.append(f"n_match_filt <= {max_n_match_filt}")
    where_clause = " AND ".join(conditions)
    return f"SELECT * FROM {baseline_tab} WHERE {where_clause}"


# ---------------------------------------------------------------------------
# Stage 5: Person-level aggregation
# ---------------------------------------------------------------------------

def _build_person_agg_query(match_filt_tab: str) -> str:
    """Aggregate filtered candidates to (person_id, user_id) level.

    Sums total_score across all spell-level evidence units per (person_id, user_id).
    A Revelio user who matches the same F1 person on multiple spells gets a higher
    person_score_sum, rewarding consistent multi-spell evidence.

    person_weight_norm: share of person_score_sum for this user among all candidate
                        users for this person_id — analogous to weight_norm at spell level.
    person_match_rank:  rank-1 = best user for each person_id.
    """
    return f"""
    WITH agg AS (
        SELECT
            person_id,
            user_id,
            SUM(total_score)                                AS person_score_sum,
            COUNT(*)                                        AS n_evidence_units,
            COUNT(DISTINCT spell_id)                        AS n_spell_matches,
            SUM(COALESCE(n_emp_matched, 0))                 AS total_emp_matches,
            MAX(COALESCE(has_any_emp_match, 0))             AS has_employer_match_ind
        FROM {match_filt_tab}
        GROUP BY person_id, user_id
    )
    SELECT *,
        person_score_sum / SUM(person_score_sum) OVER(PARTITION BY person_id) AS person_weight_norm,
        ROW_NUMBER() OVER(
            PARTITION BY person_id
            ORDER BY person_score_sum DESC, n_evidence_units DESC, user_id
        ) AS person_match_rank
    FROM agg
    """


# ---------------------------------------------------------------------------
# Testing: spotcheck pretty-print
# ---------------------------------------------------------------------------

def _print_f1_testing_spotcheck(final_query: str, sample_n: int = 5, con=con_f1) -> None:
    """Print human-readable spotcheck of top matches for sampled F1 persons."""
    sample_n = max(1, int(sample_n))
    df = con.sql(f"""
        WITH matches AS (SELECT * FROM ({final_query})),
        sampled_spells AS (
            SELECT DISTINCT spell_id
            FROM matches
            ORDER BY RANDOM()
            LIMIT {sample_n}
        )
        SELECT m.*
        FROM matches AS m
        JOIN sampled_spells AS s ON m.spell_id = s.spell_id
        ORDER BY m.spell_id, m.match_rank
    """).df()

    if df.empty:
        print("  [spotcheck] No matches found in testing subset.")
        return

    for spell_id, grp in df.groupby("spell_id"):
        top = grp.iloc[0]
        print(f"\n  {'='*70}")
        print(f"  SPELL {spell_id} | person_id={top['person_id']}")
        print(f"  F1:  school={top['f1_school_name']} | degree={top['f1_degree_level']} "
              f"| year={top['f1_prog_start_year']} | country={top['f1_country_std']} "
              f"| CIP2={top['f1_cip2']}")
        print(f"       school_match_score={top['school_match_score']:.3f} "
              f"(rev_univ={top['rev_university_raw']})")
        if "school_resolution_status" in grp.columns:
            print(f"       school_resolution={top['school_resolution_status']} "
                  f"| blocked={int(top.get('school_block_applied_ind') or 0)} "
                  f"| school_matches={int(top.get('school_match_count_pre') or 0)}"
                  f"→{int(top.get('school_match_count_post') or 0)}")
        n_emps = int(top.get('n_f1_employers') or 0)
        emp_avail = bool(top.get('emp_score_available_ind') or 0)
        print(f"       n_f1_employers={n_emps} | emp_score_available={emp_avail}")
        print(f"  Candidates ({len(grp)}):")
        for _, row in grp.head(5).iterrows():
            emp_sc = row.get('employer_score')
            emp_sc_str = f"{emp_sc:.3f}" if emp_sc is not None else "  N/A"
            n_matched = int(row.get('n_emp_matched') or 0)
            print(
                f"    #{int(row['match_rank'])} user={row['user_id']} "
                f"name={str(row.get('fullname',''))[:30]:<30} "
                f"country={row['rev_country']} "
                f"deg={row['rev_degree_clean']} "
                f"yr={row.get('rev_educ_start_year','')} "
                f"cip2={row.get('rev_cip2','')} | "
                f"emp={emp_sc_str}({n_matched}/{n_emps}matched) "
                f"cs={row['country_score']:.3f} "
                f"deg={row['degree_score']:.3f} "
                f"dt={row['date_score']:.3f} "
                f"fld={row['field_score']:.3f} "
                f"tot={row['total_score']:.3f} "
                f"wt={row['weight_norm']:.3f}"
            )


# ---------------------------------------------------------------------------
# Top-level: load data
# ---------------------------------------------------------------------------

def _load_data(con=con_f1):
    """Load all source data into DuckDB views. Returns detected column flags."""
    print("\n[Loading data]")

    # F1 FOIA
    con.sql(f"CREATE OR REPLACE VIEW f1_foia AS SELECT * FROM read_parquet('{cfg.F1_FOIA_PARQUET}')")
    n_f1 = int(con.sql("SELECT COUNT(*) FROM f1_foia").fetchone()[0])
    n_persons = int(con.sql("SELECT COUNT(DISTINCT person_id) FROM f1_foia").fetchone()[0])
    print(f"  f1_foia:          {n_f1:>12,} rows | {n_persons:>8,} distinct person_ids")

    # School crosswalk
    cw_path = cfg.F1_REV_SCHOOL_CROSSWALK_PARQUET
    if not os.path.exists(cw_path):
        raise FileNotFoundError(
            f"School crosswalk not found: {cw_path}\n"
            f"Run deps_f1_school_crosswalk.build_school_crosswalk() first."
        )
    con.sql(f"CREATE OR REPLACE VIEW f1_rev_school_cw AS SELECT * FROM read_parquet('{cw_path}')")
    n_cw = int(con.sql("SELECT COUNT(*) FROM f1_rev_school_cw").fetchone()[0])
    n_f1_schools = int(con.sql("SELECT COUNT(DISTINCT f1_school_name) FROM f1_rev_school_cw").fetchone()[0])
    school_cw_cols = _describe_relation_columns(con, "f1_rev_school_cw")
    print(f"  school_crosswalk: {n_cw:>12,} rows | {n_f1_schools:>8,} F1 schools matched")
    con.sql(
        "CREATE OR REPLACE VIEW f1_rev_school_cw_filt AS "
        + _build_school_crosswalk_family_query("f1_rev_school_cw", source_cols=school_cw_cols)
    )
    n_cw_filt = int(con.sql("SELECT COUNT(*) FROM f1_rev_school_cw_filt").fetchone()[0])
    n_cw_filt_amb = int(con.sql(
        "SELECT COUNT(DISTINCT f1_school_name) FROM f1_rev_school_cw_filt WHERE match_ambiguous_ind = 1"
    ).fetchone()[0])
    print(f"  school_crosswalk_filt:{n_cw_filt:>8,} kept families "
          f"| {n_cw_filt_amb:>8,} ambiguous F1 schools after pruning")

    resolution_path = cfg.F1_REV_SCHOOL_RESOLUTION_PARQUET
    has_school_resolution = bool(resolution_path) and os.path.exists(resolution_path)
    has_school_resolution_schema = False
    if has_school_resolution:
        con.sql(
            f"CREATE OR REPLACE VIEW f1_rev_school_resolution AS "
            f"SELECT * FROM read_parquet('{resolution_path}')"
        )
        resolution_cols = {r[0] for r in con.sql("DESCRIBE f1_rev_school_resolution").fetchall()}
        has_school_resolution_schema = _SCHOOL_RESOLUTION_REQUIRED_COLS.issubset(resolution_cols)
        if has_school_resolution_schema:
            n_res = int(con.sql("SELECT COUNT(*) FROM f1_rev_school_resolution").fetchone()[0])
            n_res_f1_rows = int(con.sql("SELECT COUNT(DISTINCT f1_row_num) FROM f1_rev_school_resolution").fetchone()[0])
            print(f"  school_resolution:{n_res:>12,} rows | {n_res_f1_rows:>8,} F1 rows matched")
        else:
            missing_resolution_cols = ", ".join(
                sorted(_SCHOOL_RESOLUTION_REQUIRED_COLS - resolution_cols)
            )
            print("  school_resolution: FOUND but missing required columns")
            print(f"    → Missing: {missing_resolution_cols}")
    else:
        print("  school_resolution: NOT FOUND (campus_unique school blocking unavailable)")

    # Revelio education
    rev_educ_path = cfg.choose_path(cfg.REV_EDUC_LONG_PARQUET, cfg.REV_EDUC_LONG_PARQUET_LEGACY)
    con.sql(f"CREATE OR REPLACE VIEW rev_educ AS SELECT * FROM read_parquet('{rev_educ_path}')")
    n_rev_educ = int(con.sql("SELECT COUNT(*) FROM rev_educ").fetchone()[0])
    n_rev_educ_users = int(con.sql("SELECT COUNT(DISTINCT user_id) FROM rev_educ").fetchone()[0])
    print(f"  rev_educ:         {n_rev_educ:>12,} rows | {n_rev_educ_users:>8,} distinct user_ids")

    # Revelio individual
    rev_indiv_path = cfg.choose_path(cfg.REV_INDIV_PARQUET, cfg.REV_INDIV_PARQUET_LEGACY)
    con.sql(f"CREATE OR REPLACE VIEW rev_indiv AS SELECT * FROM read_parquet('{rev_indiv_path}')")
    n_rev_indiv = int(con.sql("SELECT COUNT(*) FROM rev_indiv").fetchone()[0])
    print(f"  rev_indiv:        {n_rev_indiv:>12,} rows")

    # Country crosswalk table + normalized rev_indiv view
    # Load crosswalk dict into DuckDB so country normalization uses a vectorized SQL JOIN
    # rather than a per-row Python UDF call during the cross-join.
    _cw_df = pd.DataFrame(list(_COUNTRY_CW_UPPER.items()), columns=['key_upper', 'std_country'])
    con.register('_country_cw_src', _cw_df)
    con.execute("CREATE OR REPLACE TABLE _country_cw AS SELECT * FROM _country_cw_src")
    con.execute("CREATE OR REPLACE VIEW rev_indiv_norm AS "
                "SELECT ri.*, COALESCE(cw.std_country, initcap(ri.country)) AS country_std "
                "FROM rev_indiv ri "
                "LEFT JOIN _country_cw cw ON upper(ri.country) = cw.key_upper")
    print(f"  country_cw:       {len(_cw_df):>12,} entries → _country_cw + rev_indiv_norm")

    # Detect optional columns
    rev_indiv_cols = [r[0] for r in con.sql("DESCRIBE rev_indiv").fetchall()]
    has_country_uncertain = "country_uncertain_ind" in rev_indiv_cols
    has_nanat_score = "nanat_score" in rev_indiv_cols
    has_fields = "fields" in rev_indiv_cols
    print(f"  rev_indiv cols detected: country_uncertain={has_country_uncertain}, "
          f"nanat_score={has_nanat_score}, fields={has_fields}")

    # Revelio positions (for employer sequence scoring)
    rev_pos_path = cfg.choose_path(cfg.REV_POS_PARQUET, cfg.REV_POS_PARQUET_LEGACY) if cfg.REV_POS_PARQUET else ""
    has_rev_pos = bool(rev_pos_path) and os.path.exists(rev_pos_path)
    if has_rev_pos:
        con.sql(f"CREATE OR REPLACE VIEW rev_pos AS SELECT * FROM read_parquet('{rev_pos_path}')")
        n_rev_pos = int(con.sql("SELECT COUNT(*) FROM rev_pos").fetchone()[0])
        print(f"  rev_pos:          {n_rev_pos:>12,} rows")
    else:
        print(f"  rev_pos:          NOT FOUND (employer sequence scoring will be skipped)")

    # F1 employer lookup (built by deps_f1_employer_crosswalk.py)
    emp_lookup_path = cfg.F1_OPT_EMPLOYER_LOOKUP_PARQUET
    has_employer_lookup = bool(emp_lookup_path) and os.path.exists(emp_lookup_path)
    has_employer_lookup_schema = False
    if has_employer_lookup:
        con.sql(f"CREATE OR REPLACE VIEW f1_opt_emp_lookup AS "
                f"SELECT * FROM read_parquet('{emp_lookup_path}')")
        emp_lookup_cols = {r[0] for r in con.sql("DESCRIBE f1_opt_emp_lookup").fetchall()}
        has_employer_lookup_schema = _EMPLOYER_LOOKUP_REQUIRED_COLS.issubset(emp_lookup_cols)
        if has_employer_lookup_schema:
            n_emp_lookup = int(con.sql("SELECT COUNT(*) FROM f1_opt_emp_lookup").fetchone()[0])
            n_lookup_rcids = int(con.sql("SELECT COUNT(DISTINCT rcid) FROM f1_opt_emp_lookup").fetchone()[0])
            n_lookup_entities = int(con.sql("SELECT COUNT(DISTINCT foia_firm_uid) FROM f1_opt_emp_lookup").fetchone()[0])
            print(f"  f1_opt_emp_lookup:{n_emp_lookup:>12,} rows "
                  f"({n_lookup_rcids:,} distinct rcids | {n_lookup_entities:,} distinct FOIA firms)")
        else:
            missing_lookup_cols = ", ".join(sorted(_EMPLOYER_LOOKUP_REQUIRED_COLS - emp_lookup_cols))
            print("  f1_opt_emp_lookup: FOUND but missing required entity-aware columns")
            print(f"    → Missing: {missing_lookup_cols}")
            print("    → Rebuild deps_f1_employer_crosswalk.build_employer_crosswalk() first.")
    else:
        print(f"  f1_opt_emp_lookup: NOT FOUND")
        print(f"    → Run deps_f1_employer_crosswalk.build_employer_crosswalk() first.")
        print(f"    → Employer sequence scoring will be skipped.")

    # Detect employer key columns in f1_foia
    f1_foia_cols = [r[0] for r in con.sql("DESCRIBE f1_foia").fetchall()]
    has_f1_employer_col = "employer_name" in f1_foia_cols
    required_f1_employer_cols = {"employer_name", "employer_city", "employer_state", "employer_zip_code"}
    has_f1_employer_key_cols = required_f1_employer_cols.issubset(set(f1_foia_cols))
    if not has_f1_employer_key_cols:
        missing_f1_cols = ", ".join(sorted(required_f1_employer_cols - set(f1_foia_cols)))
        print(f"  f1_foia: missing employer lookup columns ({missing_f1_cols}) — employer stages skipped.")

    return {
        "has_school_resolution": has_school_resolution,
        "has_school_resolution_schema": has_school_resolution_schema,
        "has_country_uncertain": has_country_uncertain,
        "has_nanat_score": has_nanat_score,
        "has_fields": has_fields,
        "has_rev_pos": has_rev_pos,
        "has_employer_lookup": has_employer_lookup,
        "has_employer_lookup_schema": has_employer_lookup_schema,
        "has_f1_employer_col": has_f1_employer_col,
        "has_f1_employer_key_cols": has_f1_employer_key_cols,
    }


# ---------------------------------------------------------------------------
# Top-level: build merge
# ---------------------------------------------------------------------------

def build_f1_merge_inputs(
    testing: bool = None,
    overwrite: bool = None,
    con=con_f1,
) -> None:
    """Build all F1 × Revelio merge outputs.

    Steps:
      1.   Load all source data into DuckDB views
      1b.  Build f1_opt_employers_all (all F1 employers per person_id, rcid where crosswalked)
      2.   Build rev_educ_school (user_id × school, ALL degree records)
      3.   Build merge_candidates (hard-filter join: school + country + degree + year ±2)
      2b.  Build rev_pos_full (positions for candidate users) + emp_idf (global IDF weights)
      3b.  Build employer_seq_scores (IDF-weighted employer recall per person × user)
      4.   Build merge_scored (join candidates + employer scores → total_score)
      4a.  Apply filtering + weighting + ranking (baseline spell-level)
      5.   Person-level aggregation (sum scores across spell-level evidence units)
      6.  Write spell-level (baseline/mult/strict) + person-level parquets

    In testing mode (testing=True or config enabled):
      - Runs on a random sample of N person_ids
      - Materializes all intermediate tables for interactive inspection
      - Prints spell-level and person-level spotchecks
      - Does NOT write parquet output files

    Args:
        testing:  Override testing mode (None = use config value)
        overwrite: Override overwrite setting (None = use config value)
        con:      DuckDB connection to use
    """
    if testing is None:
        testing = cfg.TESTING_ENABLED
    if overwrite is None:
        overwrite = cfg.BUILD_OVERWRITE

    t_total = time.perf_counter()
    print("=" * 70)
    print("f1_indiv_merge: building F1 × Revelio merge")
    print(f"  run_tag:  {cfg.RUN_TAG}")
    print(f"  testing:  {testing}")
    print(f"  overwrite:{overwrite}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    col_flags = _load_data(con=con)
    school_block_mode = cfg.BUILD_SCHOOL_BLOCK_MODE
    if school_block_mode == "campus_unique" and not (
        col_flags["has_school_resolution"] and col_flags["has_school_resolution_schema"]
    ):
        print("\n[School blocking] Requested campus_unique mode, but rich school-resolution artifact is unavailable or invalid.")
        print("  Falling back to school_block_mode=off")
        school_block_mode = "off"
    print(f"  school_block_mode: {school_block_mode}")

    # ------------------------------------------------------------------
    # 2. Build row-level school resolution + education spells
    # ------------------------------------------------------------------
    print("\n[Stage 1: F1 school rows + education spells]")
    t0 = time.perf_counter()

    # In testing mode: restrict to a random sample of person_ids
    if testing:
        seed = cfg.TESTING_RANDOM_SEED or 42
        n_sample = cfg.TESTING_SAMPLE_N_PERSONS
        test_filter_parts = [f"WHERE person_id IN (sample_pids.person_id)"]

        # Optional school/country pin
        school_pin = cfg.TESTING_SCHOOL
        country_pin = cfg.TESTING_COUNTRY

        # Build test person_ids CTE
        test_pid_cte = f"""
        WITH sample_pids AS (
            SELECT DISTINCT person_id
            FROM f1_foia
            WHERE person_id IS NOT NULL
            {'AND school_name = ' + repr(school_pin) if school_pin else ''}
            {'AND upper(trim(country_of_birth)) = upper(trim(' + repr(country_pin) + '))' if country_pin else ''}
            USING SAMPLE {n_sample} ROWS (reservoir, {seed})
        )
        """
        f1_foia_src = f"""(
            {test_pid_cte}
            SELECT f.* FROM f1_foia AS f
            JOIN sample_pids AS s ON f.person_id = s.person_id
        )"""
        print(f"  [TESTING] Using sample of {n_sample} person_ids (seed={seed})")
    else:
        f1_foia_src = "f1_foia"

    school_rows_q = _build_f1_school_rows_query(
        f1_foia_src,
        "f1_rev_school_cw_filt",
        resolution_tab=("f1_rev_school_resolution" if school_block_mode == "campus_unique" else None),
        school_block_mode=school_block_mode,
    )

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        materialize_table(f"{pfx}_f1_school_rows", school_rows_q, con=con)
        f1_school_rows_tab = f"{pfx}_f1_school_rows"
    else:
        materialize_table("_f1_school_rows", school_rows_q, con=con)
        f1_school_rows_tab = "_f1_school_rows"

    spells_q = _build_f1_educ_spells_query(f1_school_rows_tab)

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        materialize_table(f"{pfx}_f1_educ_spells", spells_q, con=con)
        f1_spells_tab = f"{pfx}_f1_educ_spells"
    else:
        # Materialize in production to avoid re-running for each downstream query
        materialize_table("_f1_educ_spells", spells_q, con=con)
        f1_spells_tab = "_f1_educ_spells"

    _spells_ref = f1_spells_tab if not f1_spells_tab.startswith("(") else f"({f1_spells_tab}) _s"
    n_spells = int(con.sql(f"SELECT COUNT(*) FROM {_spells_ref}").fetchone()[0])
    n_persons_spells = int(con.sql(f"SELECT COUNT(DISTINCT person_id) FROM {_spells_ref}").fetchone()[0])
    print(f"  {n_spells:,} education spells from {n_persons_spells:,} person_ids "
          f"({_fmt_elapsed(time.perf_counter() - t0)})")
    school_res_summary = con.sql(f"""
        SELECT
            school_resolution_status,
            COUNT(*) AS n_spells,
            SUM(school_block_applied_ind) AS n_blocked_spells,
            ROUND(AVG(school_match_count_pre), 2) AS avg_match_count_pre,
            ROUND(AVG(school_match_count_post), 2) AS avg_match_count_post
        FROM {f1_spells_tab}
        GROUP BY school_resolution_status
        ORDER BY school_resolution_status
    """).df()
    print("  School resolution summary:")
    print(school_res_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 1b. Build f1_opt_employers_all (all F1 employers per person, rcid where available)
    # ------------------------------------------------------------------
    has_employer_data = (
        col_flags["has_employer_lookup"]
        and col_flags["has_employer_lookup_schema"]
        and col_flags["has_f1_employer_key_cols"]
    )

    if has_employer_data:
        print("\n[Stage 1b: F1 employers (all, with rcid where crosswalked)]")
        t0 = time.perf_counter()
        opt_emp_q = _build_f1_opt_employers_all_query(f1_foia_src, "f1_opt_emp_lookup")
        pfx = cfg.TESTING_TABLE_PREFIX if testing else ""
        tname = f"{pfx}_f1_opt_employers" if testing else "_f1_opt_employers"
        materialize_table(tname, opt_emp_q, con=con)
        f1_opt_emp_tab = tname
        n_opt_emp = int(con.sql(f"SELECT COUNT(*) FROM {f1_opt_emp_tab}").fetchone()[0])
        n_opt_persons = int(con.sql(
            f"SELECT COUNT(DISTINCT person_id) FROM {f1_opt_emp_tab}").fetchone()[0])
        n_with_rcid = int(con.sql(
            f"SELECT COUNT(*) FROM {f1_opt_emp_tab} WHERE rcid IS NOT NULL").fetchone()[0])
        n_ambig_rcid = int(con.sql(
            f"SELECT COUNT(*) FROM {f1_opt_emp_tab} WHERE lookup_rcid_ambiguous_ind = 1").fetchone()[0])
        n_name_only = n_opt_emp - n_with_rcid - n_ambig_rcid
        print(f"  {n_opt_emp:,} (person_id, employer) rows from {n_opt_persons:,} persons "
              f"({n_with_rcid:,} with unambiguous crosswalk rcid, "
              f"{n_ambig_rcid:,} ambiguous lookup rows forced to name matching, "
              f"{n_name_only:,} with no usable lookup rcid) "
              f"({_fmt_elapsed(time.perf_counter() - t0)})")
    else:
        f1_opt_emp_tab = None
        print("\n[Stage 1b: skipped — employer lookup unavailable or f1_foia missing location cols]")

    # ------------------------------------------------------------------
    # 2. Build rev_educ_school
    # ------------------------------------------------------------------
    print("\n[Stage 2: Revelio education × school summary]")
    t0 = time.perf_counter()

    rev_educ_school_q = _build_rev_educ_school_query(
        "rev_educ",
        "f1_rev_school_cw_filt",
    )

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        # Filter to schools present in the test F1 spells to keep testing fast
        rev_educ_school_q_filt = f"""
        SELECT res.* FROM ({rev_educ_school_q}) AS res
        INNER JOIN (SELECT DISTINCT school_name FROM {f1_spells_tab}) AS sp
            ON res.f1_school_name = sp.school_name
        """
        materialize_table(f"{pfx}_rev_educ_school", rev_educ_school_q_filt, con=con)
        rev_educ_school_tab = f"{pfx}_rev_educ_school"
    else:
        # Materialize in production — rev_educ_school is expensive (CIP lookup via array unnest)
        materialize_table("_rev_educ_school", rev_educ_school_q, con=con)
        rev_educ_school_tab = "_rev_educ_school"

    n_rev_educ_school = int(con.sql(f"SELECT COUNT(*) FROM {rev_educ_school_tab}").fetchone()[0])
    print(f"  {n_rev_educ_school:,} Revelio user×school records ({_fmt_elapsed(time.perf_counter() - t0)})")

    if school_block_mode == "campus_unique":
        rev_educ_school_row_q = _build_rev_educ_school_row_query(
            rev_educ_school_tab=rev_educ_school_tab,
            cw_tab="f1_rev_school_cw_filt",
            resolution_tab="f1_rev_school_resolution",
        )
        if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
            pfx = cfg.TESTING_TABLE_PREFIX
            materialize_table(f"{pfx}_rev_educ_school_rows", rev_educ_school_row_q, con=con)
            rev_educ_school_row_tab = f"{pfx}_rev_educ_school_rows"
        else:
            materialize_table("_rev_educ_school_rows", rev_educ_school_row_q, con=con)
            rev_educ_school_row_tab = "_rev_educ_school_rows"
        n_rev_educ_school_rows = int(con.sql(f"SELECT COUNT(*) FROM {rev_educ_school_row_tab}").fetchone()[0])
        print(f"  {n_rev_educ_school_rows:,} blocked user×school×row records")
    else:
        rev_educ_school_row_tab = None

    # ------------------------------------------------------------------
    # 3. Build merge_candidates (hard-filter join: school + country + degree + year ±2)
    # ------------------------------------------------------------------
    print("\n[Stage 3: Candidate generation (hard-filter join)]")
    t0 = time.perf_counter()

    candidates_q = _build_candidates_query(
        f1_spells_tab=f1_spells_tab,
        rev_educ_school_tab=rev_educ_school_tab,
        rev_indiv_tab="rev_indiv_norm",
        rev_educ_school_row_tab=rev_educ_school_row_tab,
    )

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        materialize_table(f"{pfx}_candidates", candidates_q, con=con)
        candidates_tab = f"{pfx}_candidates"
    else:
        print("  Materializing candidates (hard-filter cross-join)...")
        materialize_table("_f1_candidates", candidates_q, con=con)
        candidates_tab = "_f1_candidates"

    cand_counts = _f1_merge_stage_counts(f"SELECT * FROM {candidates_tab}", con=con)
    _print_merge_stage("candidates", cand_counts)
    candidate_school_block_summary = con.sql(f"""
        SELECT
            school_resolution_status,
            COUNT(DISTINCT spell_id) AS n_spells,
            ROUND(AVG(n_match_raw), 1) AS avg_candidates_per_spell,
            ROUND(AVG(school_match_count_pre), 2) AS avg_school_match_count_pre,
            ROUND(AVG(school_match_count_post), 2) AS avg_school_match_count_post
        FROM {candidates_tab}
        GROUP BY school_resolution_status
        ORDER BY school_resolution_status
    """).df()
    print("  Candidate summary by school resolution:")
    print(candidate_school_block_summary.to_string(index=False))
    print(f"  ({_fmt_elapsed(time.perf_counter() - t0)})")

    # ------------------------------------------------------------------
    # 2b. Build rev_pos_full + emp_idf (after candidates so we can filter to candidate users)
    #     Then build employer_seq_scores
    # ------------------------------------------------------------------
    if col_flags["has_rev_pos"] and f1_opt_emp_tab is not None:
        print("\n[Stage 2b: Revelio position history (candidate users) + IDF weights]")
        t0 = time.perf_counter()

        pfx = cfg.TESTING_TABLE_PREFIX if (testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES) else ""
        rev_pos_full_name = f"{pfx}_rev_pos_full" if pfx else "_f1_rev_pos_full"
        emp_idf_name      = f"{pfx}_emp_idf"      if pfx else "_f1_emp_idf"
        token_idf_name    = "_f1_token_idf"  # always global (not test-prefixed)

        materialize_table(
            rev_pos_full_name,
            _build_rev_pos_full_query("rev_pos", candidates_tab),
            con=con,
        )
        materialize_table(
            emp_idf_name,
            _build_emp_idf_query("rev_pos", cfg.BUILD_EMP_IDF_SMOOTHING),
            con=con,
        )
        materialize_table(
            token_idf_name,
            _build_token_idf_query("rev_pos", cfg.BUILD_EMP_TOKEN_MIN_IDF),
            con=con,
        )
        n_rev_pos_full = int(con.sql(f"SELECT COUNT(*) FROM {rev_pos_full_name}").fetchone()[0])
        n_idf_rcids    = int(con.sql(f"SELECT COUNT(*) FROM {emp_idf_name}").fetchone()[0])
        n_token_idf    = int(con.sql(f"SELECT COUNT(*) FROM {token_idf_name}").fetchone()[0])
        print(f"  {n_rev_pos_full:,} (user_id, rcid, company) rows for candidate users | "
              f"{n_idf_rcids:,} rcids with IDF weights | "
              f"{n_token_idf:,} tokens with IDF weights "
              f"({_fmt_elapsed(time.perf_counter() - t0)})")

        print("\n[Stage 3b: Employer sequence scoring (IDF-weighted, 3-path, optimal assignment)]")
        t0 = time.perf_counter()

        emp_score_name   = f"{pfx}_emp_seq_scores"   if pfx else "_f1_emp_seq_scores"
        match_pairs_name = f"{pfx}_emp_match_pairs"  if pfx else "_f1_emp_match_pairs"
        f1_keys_name     = f"{pfx}_f1_employer_keys" if pfx else "_f1_employer_keys"

        # i. Build (logical F1 employer key × Revelio company) match pairs
        materialize_table(
            match_pairs_name,
            _build_match_pairs_query(
                candidates_tab=candidates_tab,
                f1_opt_emp_tab=f1_opt_emp_tab,
                rev_pos_full_tab=rev_pos_full_name,
                emp_idf_tab=emp_idf_name,
                token_idf_tab=token_idf_name,
            ),
            con=con,
        )
        # ii. Build deduplicated F1 employer keys (denominator for scoring)
        materialize_table(
            f1_keys_name,
            _build_f1_keys_query(
                candidates_tab=candidates_tab,
                f1_opt_emp_tab=f1_opt_emp_tab,
                emp_idf_tab=emp_idf_name,
            ),
            con=con,
        )
        n_match_pairs = int(con.sql(f"SELECT COUNT(*) FROM {match_pairs_name}").fetchone()[0])
        n_f1_keys     = int(con.sql(f"SELECT COUNT(*) FROM {f1_keys_name}").fetchone()[0])
        print(f"  {n_match_pairs:,} (person, user, F1 employer, Revelio company) match pairs | "
              f"{n_f1_keys:,} deduplicated (person, user, F1 employer) keys")

        # iii. Solve max-weight bipartite assignment per (person, user) via scipy
        print("  solving employer assignment...", end=" ", flush=True)
        t_assign = time.perf_counter()
        assigned_df = _solve_employer_assignment(con, match_pairs_name, f1_keys_name)
        con.execute("DROP VIEW IF EXISTS _assigned_df_view")
        con.register("_assigned_df_view", assigned_df)
        assigned_name = f"{pfx}_emp_assigned" if pfx else "_f1_emp_assigned"
        materialize_table(assigned_name, "SELECT * FROM _assigned_df_view", con=con)
        con.execute("DROP VIEW IF EXISTS _assigned_df_view")
        print(f"{len(assigned_df):,} assigned pairs "
              f"({_fmt_elapsed(time.perf_counter() - t_assign)})")

        # iv. Aggregate assigned pairs → per-(person, user) employer score
        materialize_table(
            emp_score_name,
            _build_emp_scores_from_assignment_query(
                f1_keys_tab=f1_keys_name,
                assigned_pairs_tab=assigned_name,
            ),
            con=con,
        )
        emp_score_tab = emp_score_name
        n_emp_scored = int(con.sql(f"SELECT COUNT(*) FROM {emp_score_tab}").fetchone()[0])
        n_with_match = int(con.sql(
            f"SELECT COUNT(*) FROM {emp_score_tab} WHERE has_any_emp_match = 1").fetchone()[0])
        avg_score = con.sql(
            f"SELECT ROUND(AVG(employer_score), 3) FROM {emp_score_tab}"
        ).fetchone()[0]
        print(f"  {n_emp_scored:,} (person, user) pairs scored | "
              f"{n_with_match:,} with ≥1 employer match | avg_score={avg_score} "
              f"({_fmt_elapsed(time.perf_counter() - t0)})")
    else:
        emp_score_tab = None
        if not col_flags["has_rev_pos"]:
            print("\n[Stages 2b/3b: skipped — rev_pos not found]")
        else:
            print("\n[Stages 2b/3b: skipped — no employer lookup (employer_score will be NULL for all)]")

    # ------------------------------------------------------------------
    # 4. Combine candidates + employer scores → merge_scored
    # ------------------------------------------------------------------
    print("\n[Stage 4: Scoring (combine candidates + employer sequence scores)]")
    t0 = time.perf_counter()

    # emp_score_tab is None → merge_scored_q will LEFT JOIN on NULL tab;
    # handle by passing an empty table substitute when no employer data
    if emp_score_tab is None:
        # Create an empty placeholder with the right schema
        con.execute("""
            CREATE OR REPLACE TABLE _f1_emp_scores_empty AS
            SELECT
                NULL::BIGINT AS person_id, NULL::BIGINT AS user_id,
                NULL::BIGINT AS n_f1_employers, NULL::BIGINT AS n_emp_matched,
                NULL::INTEGER AS has_any_emp_match, NULL::FLOAT AS employer_score
            WHERE 1=0
        """)
        emp_score_tab = "_f1_emp_scores_empty"

    scored_q = _build_merge_scored_query(
        candidates_tab=candidates_tab,
        emp_scores_tab=emp_score_tab,
    )

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        materialize_table(f"{pfx}_merge_scored", scored_q, con=con)
        merge_scored_tab = f"{pfx}_merge_scored"
    else:
        print("  Materializing merge_scored...")
        materialize_table("_f1_merge_scored", scored_q, con=con)
        merge_scored_tab = "_f1_merge_scored"

    # Score distribution diagnostics
    score_stats = con.sql(f"""
        SELECT
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY country_score), 3) AS cs_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY degree_score),  3) AS deg_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY date_score),    3) AS dt_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY field_score),   3) AS fld_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY
                CASE WHEN employer_score IS NOT NULL THEN employer_score END), 3)  AS emp_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_score),   3) AS tot_p50,
            ROUND(AVG(emp_score_available_ind), 3)                                 AS pct_with_emp,
            ROUND(AVG(CASE WHEN w_emp_eff > 0 THEN w_emp_eff END), 3)             AS w_emp_eff_mean
        FROM {merge_scored_tab}
    """).df().to_string(index=False)
    print(f"\n  Score distributions (merge_scored):\n  {score_stats}")
    print(f"  ({_fmt_elapsed(time.perf_counter() - t0)})")

    # ------------------------------------------------------------------
    # 5. Filtering, weighting, ranking
    # ------------------------------------------------------------------
    print("\n[Stage 4a: Filtering, weighting, ranking]")
    t0 = time.perf_counter()

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        materialize_table(f"{pfx}_match_filt",
                          _build_stage_match_filt_sql(merge_scored_tab), con=con)
        filt_counts = _f1_merge_stage_counts(f"SELECT * FROM {pfx}_match_filt", con=con)
        _print_merge_stage("match_filt", filt_counts)

        materialize_table(f"{pfx}_weighted",
                          _build_stage_weighted_sql(f"{pfx}_match_filt"), con=con)

        materialize_table(f"{pfx}_final",
                          _build_stage_final_sql(f"{pfx}_weighted"), con=con)
        match_filt_tab = f"{pfx}_match_filt"
        baseline_tab = f"{pfx}_final"
    else:
        # Materialize match_filt separately — needed for person-level aggregation (Stage 5)
        print("  Materializing match_filt (spell×user dedup)...")
        materialize_table("_f1_match_filt",
                          _build_stage_match_filt_sql(f"SELECT * FROM {merge_scored_tab}"),
                          con=con)
        print("  Materializing baseline (weighted + final)...")
        materialize_table("_f1_baseline",
                          _build_stage_final_sql(
                              _build_stage_weighted_sql("_f1_match_filt")
                          ), con=con)
        match_filt_tab = "_f1_match_filt"
        baseline_tab = "_f1_baseline"

    base_counts = _f1_merge_stage_counts(f"SELECT * FROM {baseline_tab}", con=con)
    _print_merge_stage("baseline", base_counts)

    mult_dist = con.sql(f"""
        SELECT n_match_filt, COUNT(DISTINCT spell_id) AS n_spells
        FROM {baseline_tab}
        GROUP BY n_match_filt
        ORDER BY n_match_filt
        LIMIT 15
    """).df()
    print("\n  Multiplicity distribution (n_match_filt):")
    print(mult_dist.to_string(index=False))

    wn_stats = con.sql(f"""
        SELECT
            ROUND(PERCENTILE_CONT(0.1)  WITHIN GROUP (ORDER BY weight_norm), 3) AS p10,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY weight_norm), 3) AS p25,
            ROUND(PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY weight_norm), 3) AS p50,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY weight_norm), 3) AS p75,
            ROUND(PERCENTILE_CONT(0.9)  WITHIN GROUP (ORDER BY weight_norm), 3) AS p90,
            ROUND(AVG(CASE WHEN top_match_ambiguous_ind = 1 THEN 1.0 ELSE 0 END), 3) AS pct_ambiguous
        FROM {baseline_tab}
        WHERE match_rank = 1
    """).df().to_string(index=False)
    print(f"\n  Weight_norm (rank-1):\n  {wn_stats}")
    print(f"  ({_fmt_elapsed(time.perf_counter() - t0)})")

    # ------------------------------------------------------------------
    # 5. Person-level aggregation
    # ------------------------------------------------------------------
    print("\n[Stage 5: Person-level aggregation]")
    t0 = time.perf_counter()

    person_agg_q = _build_person_agg_query(match_filt_tab)

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        materialize_table(f"{pfx}_person_agg", person_agg_q, con=con)
        person_agg_tab = f"{pfx}_person_agg"
    else:
        print("  Materializing person_agg...")
        materialize_table("_f1_person_agg", person_agg_q, con=con)
        person_agg_tab = "_f1_person_agg"

    person_rank1 = con.sql(f"""
        SELECT
            COUNT(DISTINCT person_id)   AS n_persons,
            COUNT(DISTINCT user_id)     AS n_users,
            SUM(has_employer_match_ind) AS n_with_employer,
            ROUND(AVG(person_weight_norm), 3)       AS avg_weight_norm,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY person_score_sum), 3) AS median_score
        FROM {person_agg_tab}
        WHERE person_match_rank = 1
    """).df().iloc[0]
    pct_emp = 100 * person_rank1["n_with_employer"] / max(1, person_rank1["n_persons"])
    print(f"  person_agg rank-1: {int(person_rank1['n_persons']):,} persons, "
          f"{int(person_rank1['n_users']):,} distinct user_ids "
          f"({_fmt_elapsed(time.perf_counter() - t0)})")
    print(f"  With employer confirmation: {int(person_rank1['n_with_employer']):,} ({pct_emp:.1f}%)")
    print(f"  avg person_weight_norm={person_rank1['avg_weight_norm']:.3f}, "
          f"median person_score_sum={person_rank1['median_score']:.3f}")

    # ------------------------------------------------------------------
    # 6. Testing mode: spotcheck and exit
    # ------------------------------------------------------------------
    if testing:
        print("\n[Spotcheck: top spell-level matches for sampled person_ids]")
        _print_f1_testing_spotcheck(
            final_query=f"SELECT * FROM {baseline_tab}",
            sample_n=min(10, cfg.TESTING_SAMPLE_N_PERSONS),
            con=con,
        )

        print("\n[Spotcheck: person-level aggregation (rank-1 per person_id)]")
        person_sample = con.sql(f"""
            SELECT person_id, user_id, person_score_sum, n_evidence_units,
                   n_spell_matches, total_emp_matches, has_employer_match_ind,
                   person_weight_norm, person_match_rank
            FROM {person_agg_tab}
            WHERE person_id IN (
                SELECT DISTINCT person_id FROM {person_agg_tab}
                ORDER BY RANDOM() LIMIT {min(10, cfg.TESTING_SAMPLE_N_PERSONS)}
            )
            ORDER BY person_id, person_match_rank
            LIMIT 30
        """).df()
        print(person_sample.to_string(index=False))

        print("\n[TESTING MODE: no parquet output written]")
        print(f"\nTotal elapsed: {_fmt_elapsed(time.perf_counter() - t_total)}")
        return

    # ------------------------------------------------------------------
    # 7. Write output parquets
    # ------------------------------------------------------------------
    print("\n[Stage 6: Writing output parquets]")

    # Spell-level: baseline (all rank-1 matches)
    print("  Writing spell-level outputs...")
    write_query_to_parquet(
        query=f"SELECT * FROM {baseline_tab} WHERE match_rank = 1",
        out_path=cfg.F1_MERGE_BASELINE_PARQUET,
        overwrite=overwrite, con=con,
    )

    # Mult2/4/6 variants
    for cutoff, out_path in [
        (2, cfg.F1_MERGE_MULT2_PARQUET),
        (4, cfg.F1_MERGE_MULT4_PARQUET),
        (6, cfg.F1_MERGE_MULT6_PARQUET),
    ]:
        mult_q = f"SELECT * FROM {baseline_tab} WHERE match_rank = 1 AND n_match_filt <= {cutoff}"
        mult_counts = _f1_merge_stage_counts(mult_q, con=con)
        _print_merge_stage(f"mult{cutoff}", mult_counts)
        write_query_to_parquet(query=mult_q, out_path=out_path, overwrite=overwrite, con=con)

    # Strict spell-level variant
    strict_q = _build_f1_stage_strict_query(baseline_tab)
    strict_counts = _f1_merge_stage_counts(strict_q, con=con)
    _print_merge_stage("strict", strict_counts)
    write_query_to_parquet(query=strict_q, out_path=cfg.F1_MERGE_STRICT_PARQUET, overwrite=overwrite, con=con)

    # Person-level outputs
    print("  Writing person-level outputs...")
    write_query_to_parquet(
        query=f"SELECT * FROM {person_agg_tab} WHERE person_match_rank = 1",
        out_path=cfg.F1_MERGE_PERSON_BASELINE_PARQUET,
        overwrite=overwrite, con=con,
    )

    strict_person_min_wn = cfg.STRICT_PERSON_MIN_WEIGHT_NORM
    strict_person_q = (
        f"SELECT * FROM {person_agg_tab} "
        f"WHERE person_match_rank = 1 AND person_weight_norm >= {strict_person_min_wn}"
    )
    strict_person_counts = con.sql(f"""
        SELECT COUNT(DISTINCT person_id) AS n_persons, COUNT(DISTINCT user_id) AS n_users
        FROM ({strict_person_q})
    """).df().iloc[0]
    print(f"  person_strict: {int(strict_person_counts['n_persons']):,} persons, "
          f"{int(strict_person_counts['n_users']):,} users "
          f"(person_weight_norm >= {strict_person_min_wn})")
    write_query_to_parquet(
        query=strict_person_q,
        out_path=cfg.F1_MERGE_PERSON_STRICT_PARQUET,
        overwrite=overwrite, con=con,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("F1 MERGE COMPLETE — Summary")
    print(f"  run_tag:  {cfg.RUN_TAG}")
    print(f"  spell/baseline: {base_counts['n_persons']:,} persons, "
          f"{base_counts['n_spells']:,} spells, "
          f"{base_counts['mult']:.2f}x mult")
    print(f"  spell/strict:   {strict_counts['n_persons']:,} persons, "
          f"{strict_counts['n_spells']:,} spells, "
          f"{strict_counts['mult']:.2f}x mult")
    print(f"  person/baseline:{int(person_rank1['n_persons']):,} persons, "
          f"{int(person_rank1['n_users']):,} users "
          f"({pct_emp:.1f}% with employer confirmation)")
    print(f"  person/strict:  {int(strict_person_counts['n_persons']):,} persons, "
          f"{int(strict_person_counts['n_users']):,} users "
          f"(weight_norm >= {strict_person_min_wn})")
    print(f"  Total elapsed: {_fmt_elapsed(time.perf_counter() - t_total)}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Diagnostic: check a single person_id
# ---------------------------------------------------------------------------

def check_person(
    person_id,
    top_n: int = 5,
    testing: bool = None,
    person_agg_tab: str = None,
    educ_spells_tab: str = None,
    opt_emp_tab: str = None,
    baseline_tab: str = None,
    rev_pos_tab: str = None,
    con=con_f1,
) -> None:
    """Inspect a single person_id after the merge pipeline has been run.

    Prints:
      1. F1 education spells for the person
      2. F1 employers for the person
      3. Top N Revelio match candidates (by person_match_rank) with:
           - aggregate scores and spell-level breakdown
           - Revelio employer history, flagging which employers matched F1

    Args:
        person_id:       The F1 FOIA person_id to inspect.
        top_n:           Number of top match candidates to show (default 5).
        testing:         If True, use testing-mode table names (default: auto from config).
                         Pass False to force production table names.
        person_agg_tab:  Override table name for person-level aggregation.
        educ_spells_tab: Override table name for F1 education spells.
        opt_emp_tab:     Override table name for F1 employers.
        baseline_tab:    Override table name for scored/ranked spell-level matches.
        rev_pos_tab:     Override table name for Revelio position history of candidate users.
        con:             DuckDB connection (default: module-level con_f1).

    Usage (iPython, after build_f1_merge_inputs()):
        m.check_person(12345)           # auto-detects testing vs. production tables
        m.check_person(12345, testing=False)  # force production table names
    """
    # Auto-resolve table names based on testing mode
    if testing is None:
        testing = cfg.TESTING_ENABLED
    pfx = cfg.TESTING_TABLE_PREFIX if testing else None

    def _tab(test_name, prod_name, override):
        if override is not None:
            return override
        return test_name if testing else prod_name

    person_agg_tab  = _tab(f"{pfx}_person_agg",     "_f1_person_agg",     person_agg_tab)
    educ_spells_tab = _tab(f"{pfx}_f1_educ_spells", "_f1_educ_spells",    educ_spells_tab)
    opt_emp_tab     = _tab(f"{pfx}_f1_opt_employers","_f1_opt_employers",  opt_emp_tab)
    baseline_tab    = _tab(f"{pfx}_final",           "_f1_baseline",       baseline_tab)
    rev_pos_tab     = _tab(f"{pfx}_rev_pos_full",    "_f1_rev_pos_full",   rev_pos_tab)
    SEP = "=" * 70
    sep = "-" * 70

    print(f"\n{SEP}")
    print(f"PERSON CHECK: person_id={person_id}")
    print(SEP)

    # ------------------------------------------------------------------
    # 1. F1 education spells
    # ------------------------------------------------------------------
    try:
        spells_df = con.sql(f"""
            SELECT spell_id, school_name AS f1_school_name, f1_degree_level,
                   f1_prog_start_year, f1_country_std, f1_cip2
            FROM {educ_spells_tab}
            WHERE person_id = {person_id}
            ORDER BY f1_prog_start_year, spell_id
        """).df()
    except Exception as e:
        print(f"[F1 education spells] Could not query {educ_spells_tab}: {e}")
        spells_df = None

    print("\nF1 Education Spells:")
    if spells_df is None or spells_df.empty:
        print("  (none found)")
    else:
        for _, row in spells_df.iterrows():
            print(f"  spell={row['spell_id']} | school={row['f1_school_name']} "
                  f"| degree={row['f1_degree_level']} | year={row['f1_prog_start_year']} "
                  f"| country={row['f1_country_std']} | CIP2={row['f1_cip2']}")

    # ------------------------------------------------------------------
    # 2. F1 employers
    # ------------------------------------------------------------------
    try:
        emp_df = con.sql(f"""
            SELECT DISTINCT employer_name_clean,
                   COALESCE(employer_name_normed, employer_name_clean) AS employer_name_normed,
                   CAST(rcid AS BIGINT) AS rcid,
                   max_f1_year
            FROM {opt_emp_tab}
            WHERE person_id = {person_id}
            ORDER BY employer_name_clean
        """).df()
    except Exception as e:
        print(f"\n[F1 employers] Could not query {opt_emp_tab}: {e}")
        emp_df = None

    print(f"\nF1 Employers ({len(emp_df) if emp_df is not None else 0}):")
    if emp_df is None or emp_df.empty:
        print("  (none — employer data not available or person has no employer records)")
    else:
        for i, row in emp_df.iterrows():
            rcid_str = f"rcid={row['rcid']}" if row['rcid'] is not None else "no rcid"
            max_yr = f"max_yr={int(row['max_f1_year'])}" if row['max_f1_year'] is not None else "max_yr=?"
            normed = row['employer_name_normed']
            normed_str = f" → normed: '{normed}'" if normed != row['employer_name_clean'] else ""
            print(f"  {i+1}. {row['employer_name_clean']}  ({rcid_str}, {max_yr}){normed_str}")

    # ------------------------------------------------------------------
    # 2b. Employer match decomposition (F1 employer × Revelio position)
    # ------------------------------------------------------------------
    # Shows, for each candidate user × each F1 employer: which Revelio position
    # matched, via which path (entity/fuzzy/subset), the JW score, and the
    # IDF weight — so employer_score = SUM(idf × match_score) / SUM(idf) is transparent.
    # ------------------------------------------------------------------
    emp_idf_tab  = _tab(f"{pfx}_emp_idf", "_f1_emp_idf", None)

    if emp_df is not None and not emp_df.empty:
        try:
            fuz_thresh   = cfg.BUILD_EMP_FUZZY_THRESHOLD
            sub_min_tok  = cfg.BUILD_EMP_SUBSET_MIN_TOKENS
            idf_smoothing = cfg.BUILD_EMP_IDF_SMOOTHING
            default_idf_val = f"1.0 / LOG({idf_smoothing} + 1.0)"
            # Top-N candidate users (need them before the main query below)
            top_uids_df = con.sql(f"""
                SELECT user_id FROM {person_agg_tab}
                WHERE person_id = {person_id}
                ORDER BY person_match_rank LIMIT {top_n}
            """).df()
            if not top_uids_df.empty:
                uid_list_for_decomp = ", ".join(str(u) for u in top_uids_df["user_id"].tolist())
                decomp_df = con.sql(f"""
                    WITH f1_emps AS (
                        SELECT DISTINCT
                            employer_name_clean,
                            COALESCE(employer_name_normed, employer_name_clean) AS emp_normed,
                            CAST(rcid AS BIGINT) AS f1_rcid,
                            min_f1_year,
                            max_f1_year
                        FROM {opt_emp_tab}
                        WHERE person_id = {person_id}
                    ),
                    emp_idf AS (
                        SELECT rcid, idf_weight FROM {emp_idf_tab}
                    ),
                    rev_pos AS (
                        SELECT user_id, rcid AS rev_rcid, company_raw_clean,
                               COALESCE(company_raw_normed, company_raw_clean) AS rev_normed,
                               pos_start_year,
                               pos_end_year
                        FROM {rev_pos_tab}
                        WHERE user_id IN ({uid_list_for_decomp})
                    ),
                    pairs AS (
                        SELECT
                            rp.user_id,
                            fe.employer_name_clean,
                            fe.emp_normed,
                            fe.f1_rcid,
                            fe.min_f1_year,
                            fe.max_f1_year,
                            COALESCE(ei.idf_weight, {default_idf_val}) AS emp_idf,
                            rp.company_raw_clean,
                            rp.rev_normed,
                            rp.rev_rcid,
                            rp.pos_start_year,
                            rp.pos_end_year,
                            ((rp.pos_start_year IS NULL OR rp.pos_start_year <= fe.max_f1_year)
                             AND COALESCE(rp.pos_end_year, YEAR(CURRENT_DATE)) >= fe.min_f1_year) AS year_ok,
                            (fe.f1_rcid IS NOT NULL AND fe.f1_rcid = rp.rev_rcid)              AS entity_match,
                            jaro_winkler_similarity(fe.emp_normed, rp.rev_normed)              AS jw_score,
                            (list_has_all(
                                list_filter(string_split(rp.rev_normed, ' '), x -> x != ''),
                                list_filter(string_split(fe.emp_normed, ' '), x -> x != '')
                            ) OR list_has_all(
                                list_filter(string_split(fe.emp_normed, ' '), x -> x != ''),
                                list_filter(string_split(rp.rev_normed, ' '), x -> x != '')
                            ))                                                                  AS subset_flag,
                            LEAST(
                                len(list_filter(string_split(fe.emp_normed, ' '), x -> x != '')),
                                len(list_filter(string_split(rp.rev_normed, ' '), x -> x != ''))
                            )                                                                  AS shorter_n_tok
                        FROM f1_emps fe
                        LEFT JOIN emp_idf ei ON fe.f1_rcid = ei.rcid
                        CROSS JOIN rev_pos rp
                    ),
                    pair_scored AS (
                        SELECT *,
                            CASE
                                WHEN NOT year_ok                                           THEN 'year_filtered'
                                WHEN entity_match                                          THEN 'entity'
                                WHEN jw_score >= {fuz_thresh}                              THEN 'fuzzy'
                                WHEN subset_flag AND shorter_n_tok >= {sub_min_tok}        THEN 'subset'
                                ELSE NULL
                            END AS match_source,
                            CASE
                                WHEN NOT year_ok                                           THEN 0.0
                                WHEN entity_match                                          THEN 1.0
                                WHEN jw_score >= {fuz_thresh}                              THEN jw_score
                                WHEN subset_flag AND shorter_n_tok >= {sub_min_tok}        THEN jw_score
                                ELSE 0.0
                            END AS match_score
                        FROM pairs
                    ),
                    -- Best Revelio position per (user, logical F1 employer key).
                    -- Groups by emp_normed to collapse raw-name variants (e.g.
                    -- "deluxe entertainment services" → "deluxe entertainment")
                    -- so n_matched and emp_score are not inflated.
                    best AS (
                        SELECT
                            user_id,
                            emp_normed,
                            MAX(emp_idf)                                           AS emp_idf,
                            MAX(CASE WHEN match_source = 'entity'  THEN 1.0         ELSE 0 END) AS entity_score,
                            MAX(CASE WHEN match_source = 'fuzzy'   THEN match_score ELSE 0 END) AS fuzzy_score,
                            MAX(CASE WHEN match_source = 'subset'  THEN match_score ELSE 0 END) AS subset_score,
                            MAX(CASE WHEN match_source IS NOT NULL
                                      AND match_source != 'year_filtered' THEN match_score ELSE 0 END) AS best_score,
                            arg_max(company_raw_clean,
                                CASE WHEN match_source IS NOT NULL
                                      AND match_source != 'year_filtered' THEN match_score ELSE -1 END
                            )                                                      AS best_rev_company,
                            arg_max(match_source,
                                CASE WHEN match_source IS NOT NULL
                                      AND match_source != 'year_filtered' THEN match_score ELSE -1 END
                            )                                                      AS best_source
                        FROM pair_scored
                        GROUP BY user_id, emp_normed
                    )
                    SELECT *,
                        emp_idf * best_score AS score_contribution
                    FROM best
                    ORDER BY user_id, emp_normed
                """).df()

                # Print per-user breakdown
                print(f"\n{sep}")
                print("Employer Match Decomposition (per candidate user):")
                print(sep)
                for uid in top_uids_df["user_id"].tolist():
                    u_rows = decomp_df[decomp_df["user_id"] == uid]
                    if u_rows.empty:
                        continue
                    idf_total = u_rows["emp_idf"].sum()
                    score_sum = u_rows["score_contribution"].sum()
                    emp_score = score_sum / idf_total if idf_total > 0 else 0
                    n_matched = (u_rows["best_score"] > 0).sum()
                    print(f"\n  user_id={uid}  employer_score={emp_score:.3f}  "
                          f"({n_matched}/{len(u_rows)} matched)  "
                          f"[idf_total={idf_total:.3f}]")
                    print(f"  {'F1 employer':<38s} {'idf':>6} {'score':>6} {'src':<7} best Revelio match")
                    print(f"  {'-'*38} {'-'*6} {'-'*6} {'-'*7} {'-'*30}")
                    for _, r in u_rows.iterrows():
                        src = r["best_source"] if r["best_score"] > 0 else "—"
                        rev_co = (str(r["best_rev_company"])[:30]
                                  if r["best_rev_company"] is not None and r["best_score"] > 0
                                  else "—")
                        print(f"  {str(r['emp_normed']):<38s} "
                              f"{r['emp_idf']:>6.3f} "
                              f"{r['best_score']:>6.3f} "
                              f"{src:<7s} "
                              f"{rev_co}")
        except Exception as e:
            print(f"\n[Employer decomp] Error: {e}")

    # ------------------------------------------------------------------
    # 3. Top-N candidates from person_agg
    # ------------------------------------------------------------------
    try:
        cand_df = con.sql(f"""
            SELECT person_id, user_id, person_score_sum, n_evidence_units,
                   n_spell_matches, total_emp_matches, has_employer_match_ind,
                   person_weight_norm, person_match_rank
            FROM {person_agg_tab}
            WHERE person_id = {person_id}
            ORDER BY person_match_rank
            LIMIT {top_n}
        """).df()
    except Exception as e:
        print(f"\n[Candidates] Could not query {person_agg_tab}: {e}")
        return

    print(f"\n{sep}")
    print(f"Top {top_n} Candidates (from person_agg):")
    print(sep)

    if cand_df.empty:
        print("  (no candidates found for this person_id)")
        return

    # Fetch spell-level breakdown for this person from baseline_tab
    try:
        spell_scores_df = con.sql(f"""
            SELECT spell_id, user_id, match_rank, total_score,
                   employer_score, country_score, degree_score, date_score,
                   field_score, weight_norm, n_f1_employers, n_emp_matched,
                   f1_school_name, f1_degree_level, f1_prog_start_year
            FROM {baseline_tab}
            WHERE person_id = {person_id}
              AND user_id IN (
                  SELECT user_id FROM {person_agg_tab}
                  WHERE person_id = {person_id}
                  ORDER BY person_match_rank
                  LIMIT {top_n}
              )
            ORDER BY user_id, spell_id
        """).df()
        has_spell_scores = True
    except Exception as e:
        print(f"  [Note] Could not query {baseline_tab} for spell breakdown: {e}")
        spell_scores_df = None
        has_spell_scores = False

    # Fetch Revelio position history for candidate users
    candidate_user_ids = cand_df["user_id"].tolist()
    uid_list_sql = ", ".join(str(u) for u in candidate_user_ids)

    # Fetch Revelio positions with normed names + year range for match-source annotation
    try:
        rev_pos_df = con.sql(f"""
            SELECT user_id, rcid, company_raw_clean,
                   COALESCE(company_raw_normed, company_raw_clean) AS company_raw_normed,
                   pos_start_year,
                   pos_end_year
            FROM {rev_pos_tab}
            WHERE user_id IN ({uid_list_sql})
            ORDER BY user_id, company_raw_clean
        """).df()
        has_rev_pos = True
    except Exception as e:
        print(f"  [Note] Could not query {rev_pos_tab} for Revelio employers: {e}")
        rev_pos_df = None
        has_rev_pos = False

    # Build per-(user_id, company_raw_clean) → list of match annotations using SQL.
    # For each (rev_pos, f1_employer) pair compute entity/fuzzy/subset match source.
    # list_has_all(a, b) checks that every element of b appears in a (subset check).
    match_detail_df = None
    if has_rev_pos and emp_df is not None and not emp_df.empty:
        try:
            fuz_thresh = cfg.BUILD_EMP_FUZZY_THRESHOLD
            sub_min_tok = cfg.BUILD_EMP_SUBSET_MIN_TOKENS
            match_detail_df = con.sql(f"""
                WITH f1_emps AS (
                    SELECT DISTINCT
                        employer_name_clean,
                        COALESCE(employer_name_normed, employer_name_clean) AS emp_normed,
                        CAST(rcid AS BIGINT) AS f1_rcid,
                        min_f1_year,
                        max_f1_year
                    FROM {opt_emp_tab}
                    WHERE person_id = {person_id}
                ),
                rev_pos AS (
                    SELECT user_id, rcid AS rev_rcid, company_raw_clean,
                           COALESCE(company_raw_normed, company_raw_clean) AS rev_normed,
                           pos_start_year,
                           pos_end_year
                    FROM {rev_pos_tab}
                    WHERE user_id IN ({uid_list_sql})
                ),
                pairs AS (
                    SELECT
                        rp.user_id,
                        rp.company_raw_clean,
                        rp.rev_normed,
                        rp.rev_rcid,
                        rp.pos_start_year,
                        rp.pos_end_year,
                        fe.employer_name_clean AS f1_emp_clean,
                        fe.emp_normed          AS f1_normed,
                        fe.f1_rcid,
                        fe.min_f1_year,
                        fe.max_f1_year,
                        -- Year filter: position must overlap the person's F1 activity window
                        ((rp.pos_start_year IS NULL OR rp.pos_start_year <= fe.max_f1_year)
                         AND COALESCE(rp.pos_end_year, YEAR(CURRENT_DATE)) >= fe.min_f1_year) AS year_ok,
                        -- Entity match: same rcid
                        (fe.f1_rcid IS NOT NULL
                         AND fe.f1_rcid = rp.rev_rcid)                   AS entity_match,
                        -- JW score on normed names
                        jaro_winkler_similarity(fe.emp_normed, rp.rev_normed) AS jw_score,
                        -- Subset flag: shorter normed name tokens all appear in longer
                        (list_has_all(
                            string_split(rp.rev_normed, ' '),
                            list_filter(string_split(fe.emp_normed, ' '), x -> x != '')
                        ) OR list_has_all(
                            string_split(fe.emp_normed, ' '),
                            list_filter(string_split(rp.rev_normed, ' '), x -> x != '')
                        ))                                                AS subset_flag,
                        -- Length of shorter normed name (for min-token check)
                        LEAST(
                            len(list_filter(string_split(fe.emp_normed, ' '), x -> x != '')),
                            len(list_filter(string_split(rp.rev_normed, ' '), x -> x != ''))
                        )                                                 AS shorter_n_tokens
                    FROM f1_emps fe CROSS JOIN rev_pos rp
                )
                SELECT
                    user_id,
                    company_raw_clean,
                    f1_emp_clean,
                    f1_rcid,
                    rev_rcid,
                    pos_start_year,
                    pos_end_year,
                    min_f1_year,
                    max_f1_year,
                    year_ok,
                    entity_match,
                    jw_score,
                    subset_flag,
                    shorter_n_tokens,
                    CASE
                        WHEN NOT year_ok                                         THEN 'year_filtered'
                        WHEN entity_match                                        THEN 'entity'
                        WHEN jw_score >= {fuz_thresh}                            THEN 'fuzzy'
                        WHEN subset_flag AND shorter_n_tokens >= {sub_min_tok}   THEN 'subset'
                        ELSE NULL
                    END AS match_source
                FROM pairs
                WHERE entity_match
                   OR jw_score >= {fuz_thresh}
                   OR (subset_flag AND shorter_n_tokens >= {sub_min_tok})
                   OR NOT year_ok
                ORDER BY user_id, company_raw_clean, f1_emp_clean
            """).df()
        except Exception as e:
            print(f"  [Note] Could not compute match-source details: {e}")
            match_detail_df = None

    for _, cand in cand_df.iterrows():
        uid = cand["user_id"]
        print(f"\n  #{int(cand['person_match_rank'])} user_id={uid}"
              f" | person_weight_norm={cand['person_weight_norm']:.3f}"
              f" | score_sum={cand['person_score_sum']:.4f}"
              f" | n_spells={int(cand['n_spell_matches'])}"
              f" | emp_matches={int(cand['total_emp_matches'])}"
              f" | has_emp_match={int(cand['has_employer_match_ind'])}")

        # Spell-level breakdown
        if has_spell_scores and spell_scores_df is not None:
            user_spells = spell_scores_df[spell_scores_df["user_id"] == uid]
            if not user_spells.empty:
                print("     Spell-level breakdown:")
                for _, sp in user_spells.iterrows():
                    emp_str = (
                        f"emp={sp['employer_score']:.3f}({int(sp['n_emp_matched'] or 0)}/"
                        f"{int(sp['n_f1_employers'] or 0)}matched)"
                        if sp["employer_score"] is not None
                        else "emp=N/A"
                    )
                    print(f"       spell={sp['spell_id']} rank={int(sp['match_rank'])}"
                          f" | school={sp['f1_school_name']}"
                          f" deg={sp['f1_degree_level']} yr={sp['f1_prog_start_year']}"
                          f" | {emp_str}"
                          f" cs={sp['country_score']:.3f}"
                          f" deg={sp['degree_score']:.3f}"
                          f" dt={sp['date_score']:.3f}"
                          f" fld={sp['field_score']:.3f}"
                          f" tot={sp['total_score']:.4f}"
                          f" wt={sp['weight_norm']:.3f}")

        # Revelio employer history with match-source annotations
        if has_rev_pos and rev_pos_df is not None:
            user_pos = rev_pos_df[rev_pos_df["user_id"] == uid]
            print(f"     Revelio employers ({len(user_pos)}):")
            if user_pos.empty:
                print("       (none in rev_pos_full for this user)")
            else:
                # Build lookup: company_raw_clean → list of match annotation strings
                match_annots: dict[str, list[str]] = {}
                if match_detail_df is not None:
                    user_details = match_detail_df[match_detail_df["user_id"] == uid]
                    for _, md in user_details.iterrows():
                        co = md["company_raw_clean"]
                        src = md["match_source"]
                        if src == "year_filtered":
                            ann = (f"[year_filtered: pos_start={md['pos_start_year']} "
                                   f"pos_end={md['pos_end_year']} "
                                   f"f1_window={md['min_f1_year']}-{md['max_f1_year']}]")
                        elif src == "entity":
                            ann = (f"✓ entity   f1='{md['f1_emp_clean']}' "
                                   f"(rcid={md['f1_rcid']}, jw={md['jw_score']:.2f})")
                        elif src == "fuzzy":
                            ann = (f"✓ fuzzy    f1='{md['f1_emp_clean']}' "
                                   f"(jw={md['jw_score']:.2f})")
                        elif src == "subset":
                            ann = (f"✓ subset   f1='{md['f1_emp_clean']}' "
                                   f"(jw={md['jw_score']:.2f}, n_tok={md['shorter_n_tokens']})")
                        else:
                            continue
                        match_annots.setdefault(co, []).append(ann)

                for _, pos in user_pos.iterrows():
                    co = pos["company_raw_clean"]
                    rcid_str = f"rcid={pos['rcid']}" if pd.notna(pos["rcid"]) else "no rcid"
                    start_str = (f"{int(pos['pos_start_year'])}"
                                 if pd.notna(pos["pos_start_year"]) else "?")
                    end_str   = (f"{int(pos['pos_end_year'])}"
                                 if pd.notna(pos["pos_end_year"]) else "now")
                    yr_str = f"yr={start_str}-{end_str}"
                    annots = match_annots.get(co, [])
                    print(f"       {str(co):<45s}  ({rcid_str}, {yr_str})")
                    for ann in annots:
                        print(f"           {ann}")
        elif not has_rev_pos:
            print("     Revelio employers: (rev_pos_full not available)")

    print(f"\n{SEP}\n")
