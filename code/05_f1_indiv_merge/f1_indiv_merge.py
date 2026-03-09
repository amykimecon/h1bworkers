"""
f1_indiv_merge.py
=================
F1 FOIA × Revelio individual merge pipeline.

Links F1 FOIA student records (identified by foia_person_id) to Revelio
individual profiles (user_id) at the education-spell level:
  unit = (person_id, school_name, degree_level, program_start_year)

Primary join key: F1 school name → Revelio university_raw (via pre-built
fuzzy name crosswalk in deps_f1_school_crosswalk.py) × country_of_birth.

Analogous to 03_indiv_merge/indiv_merge.py for H-1B data, but with:
  - School (not employer) as the primary join key
  - country_of_birth directly available (not imputed from name)
  - No gender or DOB signals
  - Additional signals: degree level, CIP/field, program year

Scoring signals (weights configurable in configs/f1_indiv_merge.yaml):
  country_score (0.55): name-model confidence in nationality assignment
  degree_score  (0.20): degree level match (Doctor / Master / Bachelor)
  date_score    (0.15): program year proximity to Revelio educ year
  field_score   (0.10): 2-digit CIP category match

Output variants (see build_f1_merge_inputs()):
  baseline   — all rank-1 matches
  mult2/4/6  — baseline restricted to spells with ≤2/4/6 candidates
  strict     — high-precision filter (weight_norm ≥ 0.85, strict thresholds)

Usage (iPython):
    import importlib, sys
    sys.path.insert(0, '/home/yk0581/h1bworkers/code/05_f1_indiv_merge')
    import f1_indiv_merge as m
    importlib.reload(m)
    m.build_f1_merge_inputs()

    # Or with testing mode:
    m.build_f1_merge_inputs(testing=True)
"""

import os
import sys
import time

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))

import f1_indiv_merge_config as cfg  # noqa: E402
from helpers import field_clean_to_cip2_sql  # noqa: E402


# ---------------------------------------------------------------------------
# DuckDB connection
# ---------------------------------------------------------------------------
def _configure_duckdb_runtime(con):
    con.execute("SET threads = 8")
    con.execute(f"SET temp_directory = '/tmp/duckdb_f1_merge'")
    con.execute("SET memory_limit = '48GB'")


con_f1 = duckdb.connect()
_configure_duckdb_runtime(con_f1)

# ---------------------------------------------------------------------------
# Country standardization: F1 uses UPPERCASE, Revelio uses Title Case.
# Map known mismatches; title-case everything else.
# ---------------------------------------------------------------------------
_F1_COUNTRY_OVERRIDES = {
    # F1 UPPERCASE name              → Revelio Title Case name
    "REPUBLIC OF KOREA (SOUTH KOREA)": "South Korea",
    "KOREA, SOUTH": "South Korea",
    "KOREA, REPUBLIC OF": "South Korea",
    "KOREA (SOUTH)": "South Korea",
    "KOREA, NORTH": "North Korea",
    "TURKEY": "Turkiye",
    "BAHAMAS, THE": "Bahamas",
    "BURMA": "Myanmar",
    "CZECH REPUBLIC": "Czechia",
    "CONGO (KINSHASA)": "Democratic Republic of the Congo",
    "CONGO, DEMOCRATIC REPUBLIC OF THE": "Democratic Republic of the Congo",
    "CONGO, REPUBLIC OF": "Republic of the Congo",
    "CONGO (BRAZZAVILLE)": "Republic of the Congo",
    "GAMBIA, THE": "Gambia",
    "MACAU": "Macao",
    "CAPE VERDE": "Cabo Verde",
    "SWAZILAND": "Eswatini",
    "TAIWAN": "Taiwan",
    "HONG KONG": "Hong Kong",
    "IRAN": "Iran",
    "NORTH KOREA": "North Korea",
    "LAOS": "Laos",
    "MOLDOVA": "Moldova",
    "RUSSIA": "Russia",
    "SYRIA": "Syria",
    "VENEZUELA": "Venezuela",
    "VIETNAM": "Vietnam",
    "MACEDONIA": "North Macedonia",
    "TANZANIA": "Tanzania",
    "BOLIVIA": "Bolivia",
}


def _build_country_std_sql(col: str) -> str:
    """Return SQL CASE expression normalizing F1 country_of_birth → Revelio country format.

    Maps known UPPERCASE F1 mismatches to Revelio Title Case, then applies
    initcap() as a fallback for standard countries.
    """
    cases = "\n".join(
        f"        WHEN upper(trim({col})) = '{f1_name.upper()}' THEN '{rev_name}'"
        for f1_name, rev_name in _F1_COUNTRY_OVERRIDES.items()
    )
    return f"""
    CASE
{cases}
        ELSE initcap(lower(trim({col})))
    END"""


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

def _build_f1_educ_spells_query(f1_foia_tab: str, cw_tab: str) -> str:
    """Collapse F1 FOIA records → education spell level.

    Unit: (person_id, school_name [from crosswalk], degree_level, program_start_year)
    Each person can have multiple spells (e.g., Bachelor's then Master's).

    Joins to school crosswalk on school_name to get the Revelio-side match;
    if a school has no crosswalk match, the spell is excluded from the merge.
    """
    country_std = _build_country_std_sql("f.country_of_birth")
    return f"""
    WITH spells AS (
        SELECT
            f.person_id,
            f.school_name,
            cw.rev_university_raw,
            cw.match_score AS school_match_score,
            cw.match_ambiguous_ind AS school_match_ambiguous_ind,
            -- Standardize country_of_birth to Revelio Title Case format
            ({country_std}) AS f1_country_std,
            -- Highest degree achieved at this school by this person
            CASE
                WHEN MAX(CASE WHEN upper(f.student_edu_level_desc) = 'DOCTORATE' THEN 1 ELSE 0 END) = 1
                    THEN 'Doctor'
                WHEN MAX(CASE WHEN upper(f.student_edu_level_desc) LIKE '%MASTER%' THEN 1 ELSE 0 END) = 1
                    THEN 'Master'
                WHEN MAX(CASE WHEN upper(f.student_edu_level_desc) LIKE '%BACHELOR%' THEN 1 ELSE 0 END) = 1
                    THEN 'Bachelor'
                WHEN MAX(CASE WHEN upper(f.student_edu_level_desc) LIKE '%ASSOCIATE%' THEN 1 ELSE 0 END) = 1
                    THEN 'Associate'
                ELSE 'Other'
            END AS f1_degree_level,
            -- Program date range
            YEAR(MIN(TRY_CAST(f.program_start_date AS DATE))) AS f1_prog_start_year,
            YEAR(MAX(TRY_CAST(f.program_end_date AS DATE)))   AS f1_prog_end_year,
            -- CIP code (modal value across years)
            MODE(f.major_1_cip_code)                          AS f1_cip6,
            -- 2-digit CIP family (integer part before decimal, divided by 100)
            CASE
                WHEN MODE(f.major_1_cip_code) IS NULL THEN NULL
                ELSE TRY_CAST(
                    FLOOR(TRY_CAST(REPLACE(MODE(f.major_1_cip_code), '-', '.') AS FLOAT))
                    AS INTEGER)
            END AS f1_cip2,
            -- OPT employer match (for informational purposes)
            MODE(f.employment_opt_type) AS f1_opt_type,
            -- Metadata
            COUNT(DISTINCT f.year) AS n_f1_years,
            MIN(f.year_int)        AS f1_year_min,
            MAX(f.year_int)        AS f1_year_max
        FROM {f1_foia_tab} AS f
        -- Join to school crosswalk — only matched schools are included
        INNER JOIN {cw_tab} AS cw ON f.school_name = cw.f1_school_name
        WHERE f.person_id IS NOT NULL
          AND f.country_of_birth IS NOT NULL
          AND f.school_name IS NOT NULL
        GROUP BY
            f.person_id,
            f.school_name,
            cw.rev_university_raw,
            cw.match_score,
            cw.match_ambiguous_ind,
            ({country_std})
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

def _build_rev_educ_school_query(rev_educ_tab: str, rev_indiv_tab: str) -> str:
    """Collapse Revelio education records → user × school level.

    For each (user_id, university_raw) pair, take the best education record
    (most recent non-trivial degree with dates, preferring Doctor > Master > Bachelor).

    Joins to rev_indiv to get the CIP2 field mapping from the user's fields array.
    """
    cip2_sql = field_clean_to_cip2_sql("f_elem")
    return f"""
    WITH educ_ranked AS (
        SELECT
            e.user_id,
            e.university_raw,
            e.degree_clean,
            e.ed_startdate,
            e.ed_enddate,
            -- Prefer Doctor > Master/MBA > Bachelor > other
            CASE e.degree_clean
                WHEN 'Doctor'   THEN 1
                WHEN 'Master'   THEN 2
                WHEN 'MBA'      THEN 2
                WHEN 'Bachelor' THEN 3
                WHEN 'Associate' THEN 4
                ELSE 5
            END AS degree_rank,
            ROW_NUMBER() OVER(
                PARTITION BY e.user_id, e.university_raw
                ORDER BY
                    CASE e.degree_clean
                        WHEN 'Doctor'   THEN 1
                        WHEN 'Master'   THEN 2
                        WHEN 'MBA'      THEN 2
                        WHEN 'Bachelor' THEN 3
                        WHEN 'Associate' THEN 4
                        ELSE 5
                    END,
                    TRY_CAST(e.ed_enddate AS DATE) DESC NULLS LAST,
                    TRY_CAST(e.ed_startdate AS DATE) DESC NULLS LAST
            ) AS rn
        FROM {rev_educ_tab} AS e
        WHERE e.university_raw IS NOT NULL
          AND e.degree_clean NOT IN ('Non-Degree', 'High School')
    ),
    best_educ AS (
        SELECT
            user_id,
            university_raw,
            degree_clean AS rev_degree_clean,
            YEAR(TRY_CAST(ed_startdate AS DATE)) AS rev_educ_start_year,
            YEAR(TRY_CAST(ed_enddate AS DATE))   AS rev_educ_end_year,
            ed_startdate AS rev_educ_start_raw,
            ed_enddate   AS rev_educ_end_raw
        FROM educ_ranked
        WHERE rn = 1
    ),
    -- Derive 2-digit CIP from user's fields array in rev_indiv
    user_cip AS (
        SELECT
            ri.user_id,
            -- Apply CIP mapping to each element of the fields array and take the mode
            -- fields is an array of field_clean strings like ['Engineering', 'Computer Science']
            (SELECT MODE(mapped_cip)
             FROM (
                 SELECT {cip2_sql} AS mapped_cip
                 FROM UNNEST(ri.fields) AS t(f_elem)
             ) WHERE mapped_cip IS NOT NULL
            ) AS rev_cip2
        FROM {rev_indiv_tab} AS ri
        WHERE ri.fields IS NOT NULL AND ARRAY_LENGTH(ri.fields) > 0
    )
    SELECT
        b.user_id,
        b.university_raw,
        b.rev_degree_clean,
        b.rev_educ_start_year,
        b.rev_educ_end_year,
        b.rev_educ_start_raw,
        b.rev_educ_end_raw,
        uc.rev_cip2
    FROM best_educ AS b
    LEFT JOIN user_cip AS uc ON b.user_id = uc.user_id
    """


# ---------------------------------------------------------------------------
# Stage 3: Raw cross-join (merge_raw)
# ---------------------------------------------------------------------------

def _build_merge_raw_query(
    f1_spells_tab: str,
    rev_educ_school_tab: str,
    rev_indiv_tab: str,
    W_COUNTRY: float = cfg.BUILD_W_COUNTRY,
    W_DEGREE: float = cfg.BUILD_W_DEGREE,
    W_DATE: float = cfg.BUILD_W_DATE,
    W_FIELD: float = cfg.BUILD_W_FIELD,
    DATE_SCORE_YEAR_BUFFER: int = cfg.BUILD_DATE_SCORE_YEAR_BUFFER,
    DATE_SCORE_NULL_DEFAULT: float = cfg.BUILD_DATE_SCORE_NULL_DEFAULT,
    DEGREE_SCORE_NULL_DEFAULT: float = cfg.BUILD_DEGREE_SCORE_NULL_DEFAULT,
) -> str:
    """Build the raw F1 × Revelio cross-join query with scores.

    Join condition: rev_university_raw (from F1 school crosswalk) = re.university_raw
                  AND lower(ri.country) = lower(f1.f1_country_std)

    Including country in the join key is necessary to keep the cross-join
    tractable (without it: ~169B rows across all schools).

    Scores:
      country_score: name-model confidence (nanat/nt scores), NOT country match
                     (that's enforced by the join)
      degree_score:  1.0 if degree levels agree, DEGREE_NULL_DEFAULT if either missing, 0.0 otherwise
      date_score:    linear decay on |f1_prog_start_year - rev_educ_start_year|
      field_score:   1.0 if 2-digit CIP matches, 0.5 if either null, 0.0 otherwise
      total_score:   weighted sum × school_match_score (quality multiplier)
    """
    date_score_expr = f"""
        CASE
            WHEN f1.f1_prog_start_year IS NULL OR re.rev_educ_start_year IS NULL
                THEN {DATE_SCORE_NULL_DEFAULT}
            WHEN ABS(f1.f1_prog_start_year - re.rev_educ_start_year) <= {DATE_SCORE_YEAR_BUFFER}
                THEN 1.0 - ABS(f1.f1_prog_start_year - re.rev_educ_start_year)::FLOAT
                         / ({DATE_SCORE_YEAR_BUFFER} + 1.0)
            ELSE 0.0
        END"""

    degree_score_expr = f"""
        CASE
            WHEN f1.f1_degree_level IS NULL OR f1.f1_degree_level = 'Other'
                 OR re.rev_degree_clean IS NULL OR re.rev_degree_clean = 'Missing'
                THEN {DEGREE_SCORE_NULL_DEFAULT}
            WHEN f1.f1_degree_level = 'Doctor'   AND re.rev_degree_clean = 'Doctor'   THEN 1.0
            WHEN f1.f1_degree_level = 'Master'   AND re.rev_degree_clean IN ('Master', 'MBA') THEN 1.0
            WHEN f1.f1_degree_level = 'Bachelor' AND re.rev_degree_clean = 'Bachelor' THEN 1.0
            WHEN f1.f1_degree_level = 'Associate' AND re.rev_degree_clean = 'Associate' THEN 1.0
            ELSE 0.0
        END"""

    field_score_expr = """
        CASE
            WHEN f1.f1_cip2 IS NULL OR re.rev_cip2 IS NULL THEN 0.5
            WHEN f1.f1_cip2 = re.rev_cip2 THEN 1.0
            ELSE 0.0
        END"""

    # country_score: confidence of Revelio name-nationality model for the matched country.
    # Since country is in the JOIN condition, country_score reflects quality of evidence,
    # not whether a country match exists.
    country_score_expr = """
        CASE
            WHEN ri.nanat_score IS NULL AND ri.nanat_subregion_score IS NULL
                AND ri.nt_subregion_score IS NULL
                THEN 0.5   -- no name model data → neutral
            WHEN COALESCE(ri.country_uncertain_ind, 0) = 1
                -- uncertain nationality → discount by 30%
                THEN 0.7 * LEAST(1.0, GREATEST(
                    COALESCE(ri.nanat_subregion_score, 0),
                    COALESCE(ri.nt_subregion_score, 0)
                ))
            ELSE
                -- certain country: blend subregion evidence and specificity
                LEAST(1.0,
                    0.4 * LEAST(1.0, GREATEST(
                        COALESCE(ri.nanat_subregion_score, 0),
                        COALESCE(ri.nt_subregion_score, 0)
                    )) +
                    0.6 * COALESCE(ri.nanat_score, 0.5)
                )
        END"""

    return f"""
    SELECT
        -- F1 spell identifiers
        f1.spell_id,
        f1.person_id,
        f1.school_name             AS f1_school_name,
        f1.rev_university_raw,     -- Revelio university matched to this F1 school
        f1.school_match_score,
        f1.school_match_ambiguous_ind,
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
        -- Revelio candidate
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
        -- Revelio education at this school
        re.rev_degree_clean,
        re.rev_educ_start_year,
        re.rev_educ_end_year,
        re.rev_educ_start_raw,
        re.rev_educ_end_raw,
        re.rev_cip2,
        -- Scores
        ({country_score_expr})     AS country_score,
        ({degree_score_expr})      AS degree_score,
        ({date_score_expr})        AS date_score,
        ({field_score_expr})       AS field_score,
        -- Total score = weighted sum × school quality multiplier
        (   ({country_score_expr}) * {W_COUNTRY}
          + ({degree_score_expr})  * {W_DEGREE}
          + ({date_score_expr})    * {W_DATE}
          + ({field_score_expr})   * {W_FIELD}
        ) * f1.school_match_score  AS total_score,
        -- Raw candidate count per spell (before filtering)
        COUNT(*) OVER(PARTITION BY f1.spell_id) AS n_match_raw
    FROM {f1_spells_tab} AS f1
    -- Join Revelio education records that match this school name
    JOIN {rev_educ_school_tab} AS re ON f1.rev_university_raw = re.university_raw
    -- Join Revelio individual, enforcing country match to keep join tractable
    JOIN {rev_indiv_tab} AS ri
        ON re.user_id = ri.user_id
        AND lower(ri.country) = lower(f1.f1_country_std)
    """


# ---------------------------------------------------------------------------
# Stage 4: Filtering and weighting
# ---------------------------------------------------------------------------

def _build_stage_match_filt_sql(
    source_tab: str,
    COUNTRY_SCORE_CUTOFF: float = cfg.BUILD_COUNTRY_SCORE_CUTOFF,
    NO_COUNTRY_MIN_TOTAL_SCORE: float = cfg.BUILD_NO_COUNTRY_MIN_TOTAL_SCORE,
) -> str:
    """Build stage_match_filt SQL: keep best (spell_id, user_id) pair; apply country threshold."""
    return f"""
    WITH base AS (
        SELECT *,
            ROW_NUMBER() OVER(
                PARTITION BY spell_id, user_id
                ORDER BY total_score DESC, rev_degree_clean ASC, rev_educ_start_year ASC
            ) AS spell_user_rn,
            MAX(country_score) OVER(PARTITION BY spell_id) AS max_country_score_spell
        FROM {source_tab}
    ),
    deduped AS (
        SELECT * EXCLUDE spell_user_rn FROM base WHERE spell_user_rn = 1
    )
    SELECT *,
        COUNT(*) OVER(PARTITION BY spell_id) AS n_match_filt
    FROM deduped
    WHERE country_score > {COUNTRY_SCORE_CUTOFF}
       OR (
            max_country_score_spell <= {COUNTRY_SCORE_CUTOFF}
            AND total_score >= {NO_COUNTRY_MIN_TOTAL_SCORE}
       )
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
    """Build stage_final SQL: rank candidates per spell; flag ambiguity; bad match guard."""
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
        FROM filtered
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


def _build_f1_merge_filt_stage_queries(merge_raw_tab: str, **kwargs) -> dict:
    """Build full nested filtering+weighting+ranking pipeline as a single SQL string.

    Used in the non-testing (production) path where we materialize the
    whole pipeline in one shot. Returns a dict with 'stage_final' key.

    Partitioning unit: spell_id (analogous to foia_indiv_id in H1B merge).
    """
    match_filt_q = _build_stage_match_filt_sql(f"({merge_raw_tab})", **{
        k: v for k, v in kwargs.items()
        if k in ("COUNTRY_SCORE_CUTOFF", "NO_COUNTRY_MIN_TOTAL_SCORE")
    })
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
        print(f"  Candidates ({len(grp)}):")
        for _, row in grp.head(5).iterrows():
            print(
                f"    #{int(row['match_rank'])} user={row['user_id']} "
                f"name={str(row.get('fullname',''))[:30]:<30} "
                f"country={row['rev_country']} "
                f"deg={row['rev_degree_clean']} "
                f"yr={row.get('rev_educ_start_year','')} "
                f"cip2={row.get('rev_cip2','')} | "
                f"country_sc={row['country_score']:.3f} "
                f"deg_sc={row['degree_score']:.3f} "
                f"date_sc={row['date_score']:.3f} "
                f"field_sc={row['field_score']:.3f} "
                f"total={row['total_score']:.3f} "
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
    print(f"  school_crosswalk: {n_cw:>12,} rows | {n_f1_schools:>8,} F1 schools matched")

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

    # Detect optional columns
    rev_indiv_cols = [r[0] for r in con.sql("DESCRIBE rev_indiv").fetchall()]
    has_country_uncertain = "country_uncertain_ind" in rev_indiv_cols
    has_nanat_score = "nanat_score" in rev_indiv_cols
    has_fields = "fields" in rev_indiv_cols
    print(f"  rev_indiv cols detected: country_uncertain={has_country_uncertain}, "
          f"nanat_score={has_nanat_score}, fields={has_fields}")

    return {
        "has_country_uncertain": has_country_uncertain,
        "has_nanat_score": has_nanat_score,
        "has_fields": has_fields,
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
      1. Load all source data into DuckDB views
      2. Build f1_educ_spells (person_id × school × degree × year)
      3. Build rev_educ_school (user_id × school summary with CIP)
      4. Build merge_raw (cross-join on school × country with scores)
      5. Apply filtering + weighting + ranking (baseline)
      6. Write baseline, mult2/4/6, strict parquet outputs

    In testing mode (testing=True or config enabled):
      - Runs on a random sample of N person_ids
      - Materializes all intermediate tables for interactive inspection
      - Prints a human-readable spotcheck of top matches
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

    # ------------------------------------------------------------------
    # 2. Build f1_educ_spells
    # ------------------------------------------------------------------
    print("\n[Stage 1: F1 education spells]")
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
            USING SAMPLE {n_sample} ROWS (bernoulli, {seed})
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

    spells_q = _build_f1_educ_spells_query(f1_foia_src, "f1_rev_school_cw")

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

    # ------------------------------------------------------------------
    # 3. Build rev_educ_school
    # ------------------------------------------------------------------
    print("\n[Stage 2: Revelio education × school summary]")
    t0 = time.perf_counter()

    rev_educ_school_q = _build_rev_educ_school_query("rev_educ", "rev_indiv")

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        # Filter to schools present in the test F1 spells to keep testing fast
        rev_educ_school_q_filt = f"""
        SELECT * FROM ({rev_educ_school_q})
        WHERE university_raw IN (
            SELECT DISTINCT rev_university_raw FROM {f1_spells_tab}
        )
        """
        materialize_table(f"{pfx}_rev_educ_school", rev_educ_school_q_filt, con=con)
        rev_educ_school_tab = f"{pfx}_rev_educ_school"
    else:
        # Materialize in production — rev_educ_school is expensive (CIP lookup via array unnest)
        materialize_table("_rev_educ_school", rev_educ_school_q, con=con)
        rev_educ_school_tab = "_rev_educ_school"

    _rev_ref = rev_educ_school_tab if not rev_educ_school_tab.startswith("(") else f"({rev_educ_school_tab}) _r"
    n_rev_educ_school = int(con.sql(f"SELECT COUNT(*) FROM {_rev_ref}").fetchone()[0])
    print(f"  {n_rev_educ_school:,} Revelio user×school records ({_fmt_elapsed(time.perf_counter() - t0)})")

    # ------------------------------------------------------------------
    # 4. Build merge_raw
    # ------------------------------------------------------------------
    print("\n[Stage 3: Raw cross-join (merge_raw)]")
    t0 = time.perf_counter()

    merge_raw_q = _build_merge_raw_query(
        f1_spells_tab=f1_spells_tab,
        rev_educ_school_tab=rev_educ_school_tab,
        rev_indiv_tab="rev_indiv",
    )

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        materialize_table(f"{pfx}_merge_raw", merge_raw_q, con=con)
        merge_raw_tab = f"{pfx}_merge_raw"
    else:
        # Materialize always — the cross-join is heavy and referenced multiple times
        print("  Materializing merge_raw (may take a few minutes)...")
        materialize_table("_f1_merge_raw", merge_raw_q, con=con)
        merge_raw_tab = "_f1_merge_raw"

    raw_counts = _f1_merge_stage_counts(f"SELECT * FROM {merge_raw_tab}", con=con)
    _print_merge_stage("merge_raw", raw_counts)
    print(f"  ({_fmt_elapsed(time.perf_counter() - t0)})")

    # Print score distributions for debugging
    print("\n  Score distributions (merge_raw):")
    score_stats = con.sql(f"""
        SELECT
            ROUND(MIN(country_score), 3) AS cs_min,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY country_score), 3) AS cs_p25,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY country_score), 3) AS cs_p50,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY country_score), 3) AS cs_p75,
            ROUND(MAX(country_score), 3) AS cs_max,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY degree_score), 3) AS deg_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY date_score), 3) AS date_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY field_score), 3) AS field_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_score), 3) AS total_p50
        FROM {merge_raw_tab}
    """).df().to_string(index=False)
    print(f"  {score_stats}")

    # ------------------------------------------------------------------
    # 5. Build filtering/weighting/ranking pipeline
    # ------------------------------------------------------------------
    print("\n[Stage 4: Filtering, weighting, ranking]")
    t0 = time.perf_counter()

    stages = _build_f1_merge_filt_stage_queries(merge_raw_tab=f"SELECT * FROM {merge_raw_tab}")

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        # Materialize each stage sequentially, referencing the previous materialized table
        materialize_table(f"{pfx}_match_filt",
                          _build_stage_match_filt_sql(merge_raw_tab), con=con)
        filt_counts = _f1_merge_stage_counts(f"SELECT * FROM {pfx}_match_filt", con=con)
        _print_merge_stage("match_filt", filt_counts)

        materialize_table(f"{pfx}_weighted",
                          _build_stage_weighted_sql(f"{pfx}_match_filt"), con=con)

        materialize_table(f"{pfx}_final",
                          _build_stage_final_sql(f"{pfx}_weighted"), con=con)
        baseline_tab = f"{pfx}_final"
    else:
        print("  Materializing baseline (match_filt + weighted + final)...")
        materialize_table("_f1_baseline", stages["stage_final"], con=con)
        baseline_tab = "_f1_baseline"

    base_counts = _f1_merge_stage_counts(f"SELECT * FROM {baseline_tab}", con=con)
    _print_merge_stage("baseline", base_counts)

    # Multiplicity distribution
    mult_dist = con.sql(f"""
        SELECT n_match_filt, COUNT(DISTINCT spell_id) AS n_spells
        FROM {baseline_tab}
        GROUP BY n_match_filt
        ORDER BY n_match_filt
        LIMIT 15
    """).df()
    print("\n  Multiplicity distribution (n_match_filt):")
    print(mult_dist.to_string(index=False))

    print(f"\n  Weight_norm for rank-1 matches:")
    wn_stats = con.sql(f"""
        SELECT
            ROUND(PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY weight_norm), 3) AS p10,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY weight_norm), 3) AS p25,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY weight_norm), 3) AS p50,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY weight_norm), 3) AS p75,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY weight_norm), 3) AS p90,
            ROUND(AVG(CASE WHEN top_match_ambiguous_ind = 1 THEN 1.0 ELSE 0 END), 3) AS pct_ambiguous
        FROM {baseline_tab}
        WHERE match_rank = 1
    """).df().to_string(index=False)
    print(f"  {wn_stats}")

    # ------------------------------------------------------------------
    # 6. Testing mode: spotcheck and exit
    # ------------------------------------------------------------------
    if testing:
        print("\n[Spotcheck: top matches for sampled person_ids]")
        _print_f1_testing_spotcheck(
            final_query=f"SELECT * FROM {baseline_tab}",
            sample_n=min(10, cfg.TESTING_SAMPLE_N_PERSONS),
            con=con,
        )
        print("\n[TESTING MODE: no parquet output written]")
        print(f"\nTotal elapsed: {_fmt_elapsed(time.perf_counter() - t_total)}")
        return

    # ------------------------------------------------------------------
    # 7. Write output parquets
    # ------------------------------------------------------------------
    print("\n[Stage 5: Writing output parquets]")

    # Baseline (all rank-1 matches)
    print("  Writing baseline...")
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

    # Strict variant
    strict_q = _build_f1_stage_strict_query(baseline_tab)
    strict_counts = _f1_merge_stage_counts(strict_q, con=con)
    _print_merge_stage("strict", strict_counts)
    write_query_to_parquet(query=strict_q, out_path=cfg.F1_MERGE_STRICT_PARQUET, overwrite=overwrite, con=con)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("F1 MERGE COMPLETE — Summary")
    print(f"  run_tag:  {cfg.RUN_TAG}")
    print(f"  baseline: {base_counts['n_persons']:,} persons, "
          f"{base_counts['n_spells']:,} spells, "
          f"{base_counts['mult']:.2f}x mult")
    print(f"  strict:   {strict_counts['n_persons']:,} persons, "
          f"{strict_counts['n_spells']:,} spells, "
          f"{strict_counts['mult']:.2f}x mult")
    print(f"  Total elapsed: {_fmt_elapsed(time.perf_counter() - t_total)}")
    print("=" * 70)
