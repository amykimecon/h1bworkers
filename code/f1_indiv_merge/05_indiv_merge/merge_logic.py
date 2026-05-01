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
  - Grad year: |f1_prog_end_year - rev_educ_end_year| ≤ year_hard_buffer

Scoring (primary): IDF-weighted recall of F1 employers in Revelio position history.
  employer_score = Σ(idf_weight_i × matched_i) / Σ(idf_weight_i)
  where idf_weight = 1/log(smoothing + n_rev_users_at_employer)
  Employer match via entity-ID first, Jaro-Winkler fuzzy fallback for uncrosswalked employers.
  Persons with no employer data: employer_score excluded (field + name weights rebalanced).

Scoring (tiebreakers):
  gradyr_score:   closeness of F1 and Revelio graduation years
  country_score:  H-1B-style country evidence score
  inst_score:     institution-country evidence input to country_score
  field_score:    stage-04 field candidate score, scaled by F1/Revelio CIP agreement
                  (1.0 on CIP4 match, 0.7 on CIP2-only match, 0.0 on CIP2 mismatch)

Output variants (see build_f1_merge_inputs()):
  baseline   — all rank-1 matches
  mult2/4/6  — baseline restricted to spells with ≤2/4/6 candidates
  strict     — high-precision filter (weight_norm ≥ 0.85, strict thresholds)

Usage (iPython):
    import importlib, sys
    sys.path.insert(0, '/home/yk0581/h1bworkers/code/f1_indiv_merge/05_indiv_merge')
    import merge_logic as m
    importlib.reload(m)
    m.build_f1_merge_inputs()

    # Or with testing mode:
    m.build_f1_merge_inputs(testing=True)
"""

import json
import os
import shutil
import sys
import tempfile
import time
import weakref
from builtins import print as _print
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import merge_config as cfg  # noqa: E402
from config import root as _root  # noqa: E402
from deps_f1_school_crosswalk import _sql_clean_inst_name_expr  # noqa: E402
from employer_entity_sql import (  # noqa: E402
    sql_clean_zip_expr,
    sql_normalize_expr,
    sql_state_name_to_abbr_expr,
)
from helpers import cip_code_to_cip4_sql, field_clean_to_cip4_sql  # noqa: E402
from src.duckdb_runtime import get_duckdb_memory_limit_sql_literal  # noqa: E402

try:  # noqa: E402
    import f1_foia.econ_relabels_opt_usage as relabel_base
    import f1_foia.econ_relabels_opt_usage_v2 as relabel_v2
except ModuleNotFoundError:  # pragma: no cover - only relevant in broken path setups
    relabel_base = None
    relabel_v2 = None

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
def _cleanup_duckdb_temp_dir(path: str | Path) -> None:
    shutil.rmtree(Path(path), ignore_errors=True)


def _create_duckdb_temp_dir() -> Path:
    temp_root = Path(_root) / ".tmp" / "f1_indiv_merge_duckdb"
    temp_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"stage05_pid{os.getpid()}_", dir=str(temp_root)))


def _configure_duckdb_runtime(con):
    con.execute("SET threads = 8")
    con.execute(f"SET memory_limit = '{get_duckdb_memory_limit_sql_literal()}'")
    temp_dir = _create_duckdb_temp_dir()
    weakref.finalize(con, _cleanup_duckdb_temp_dir, temp_dir)
    escaped_temp_dir = str(temp_dir).replace("'", "''")
    con.execute(f"SET temp_directory = '{escaped_temp_dir}'")
    # initcap (capitalize first letter of each word) is absent in DuckDB 1.2.x
    con.create_function("initcap", lambda s: s.title() if s else s, [str], str)


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    _configure_duckdb_runtime(con)
    return con


con_f1 = get_duckdb_connection()


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


def _describe_relation_columns(con, relation: str) -> list[str]:
    return [row[0] for row in con.sql(f"DESCRIBE {relation}").fetchall()]


def _first_present_col(columns: list[str], candidates: list[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _optional_select_expr(columns: list[str], candidates: list[str], alias: str, cast: str = "VARCHAR") -> str:
    selected = _first_present_col(columns, candidates)
    if selected is None:
        return f"NULL::{cast} AS {alias}"
    return f"CAST({selected} AS {cast}) AS {alias}"


def _resolve_rev_educ_relation_sql(con) -> str:
    try:
        _describe_relation_columns(con, "rev_educ")
        return "rev_educ"
    except Exception:
        rev_educ_path = cfg.choose_path(cfg.REV_EDUC_LONG_PARQUET, cfg.REV_EDUC_LONG_PARQUET_LEGACY)
        if not rev_educ_path or not os.path.exists(rev_educ_path):
            raise FileNotFoundError("Revelio education input not available for field_raw backfill.")
        return f"read_parquet('{_sql_escape_path(rev_educ_path)}')"


def _build_field_raw_lookup_cte_sql(
    con,
    *,
    user_filter_sql: str,
    cte_name: str = "field_raw_lookup",
) -> str:
    rev_educ_rel = _resolve_rev_educ_relation_sql(con)
    rev_educ_cols = {
        row[0]
        for row in con.sql(f"DESCRIBE SELECT * FROM {rev_educ_rel}").fetchall()
    }

    def _first_expr(candidates: list[tuple[str, str]]) -> str:
        for col, expr in candidates:
            if col in rev_educ_cols:
                return expr
        return "NULL::BIGINT"

    rev_university_expr = (
        "CAST(rev_university_raw AS VARCHAR)"
        if "rev_university_raw" in rev_educ_cols
        else "CAST(university_raw AS VARCHAR)"
    )
    rev_degree_expr = (
        "CAST(rev_degree_clean AS VARCHAR)"
        if "rev_degree_clean" in rev_educ_cols
        else "CAST(degree_clean AS VARCHAR)"
    )
    rev_start_year_expr = _first_expr(
        [
            ("rev_educ_start_year", "CAST(rev_educ_start_year AS BIGINT)"),
            ("ed_start_year", "CAST(ed_start_year AS BIGINT)"),
            ("ed_startdate", "CAST(EXTRACT(YEAR FROM TRY_CAST(ed_startdate AS DATE)) AS BIGINT)"),
            ("startdate", "CAST(EXTRACT(YEAR FROM TRY_CAST(startdate AS DATE)) AS BIGINT)"),
        ]
    )
    rev_end_year_expr = _first_expr(
        [
            ("rev_educ_end_year", "CAST(rev_educ_end_year AS BIGINT)"),
            ("ed_end_year", "CAST(ed_end_year AS BIGINT)"),
            ("ed_enddate", "CAST(EXTRACT(YEAR FROM TRY_CAST(ed_enddate AS DATE)) AS BIGINT)"),
            ("enddate", "CAST(EXTRACT(YEAR FROM TRY_CAST(enddate AS DATE)) AS BIGINT)"),
        ]
    )
    field_raw_source_expr = (
        "CAST(field_raw AS VARCHAR)"
        if "field_raw" in rev_educ_cols
        else (
            "CAST(field_clean AS VARCHAR)"
            if "field_clean" in rev_educ_cols
            else "NULL::VARCHAR"
        )
    )
    return f"""
    {cte_name} AS (
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            {rev_university_expr} AS rev_university_raw,
            {rev_degree_expr} AS rev_degree_clean,
            {rev_start_year_expr} AS rev_educ_start_year,
            {rev_end_year_expr} AS rev_educ_end_year,
            MIN({field_raw_source_expr}) FILTER (
                WHERE {field_raw_source_expr} IS NOT NULL
                  AND trim({field_raw_source_expr}) <> ''
            ) AS field_raw
        FROM {rev_educ_rel}
        WHERE user_id IN ({user_filter_sql})
          AND {rev_university_expr} IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5
    )
    """


def _maybe_float(value) -> float | None:
    try:
        if value is None or not pd.notna(value):
            return None
        return float(value)
    except Exception:
        return None


def _cip2_from_cip4_sql(value_sql: str) -> str:
    return f"""
        CASE
            WHEN {value_sql} IS NULL THEN NULL
            ELSE TRY_CAST(FLOOR(CAST({value_sql} AS DOUBLE) / 100.0) AS INTEGER)
        END
    """


def _field_score_expr(
    f1_cip4_sql: str,
    rev_cip4_sql: str,
    rev_cip_score_sql: str,
    *,
    cip2_match_multiplier: float,
) -> str:
    f1_cip2_expr = _cip2_from_cip4_sql(f1_cip4_sql)
    rev_cip2_expr = _cip2_from_cip4_sql(rev_cip4_sql)
    bounded_rev_score_expr = (
        f"LEAST(1.0, GREATEST(0.0, COALESCE({rev_cip_score_sql}, 0.0)))"
    )
    return f"""
        CASE
            WHEN {rev_cip_score_sql} IS NULL THEN 0.0
            WHEN {f1_cip4_sql} IS NULL OR {rev_cip4_sql} IS NULL THEN {bounded_rev_score_expr}
            WHEN {rev_cip4_sql} = {f1_cip4_sql} THEN {bounded_rev_score_expr}
            WHEN {rev_cip2_expr} = {f1_cip2_expr}
                THEN LEAST(1.0, GREATEST(0.0, {cip2_match_multiplier} * {bounded_rev_score_expr}))
            ELSE 0.0
        END
    """


def _h1b_country_base_expr(
    inst_score_sql: str,
    position_score_sql: str,
    nanat_score_sql: str,
    nanat_subregion_sql: str,
    nt_subregion_sql: str,
    alpha: float = cfg.BUILD_SUBREGION_BOOST_ALPHA,
) -> str:
    subregion_expr = (
        "LEAST(1.0, GREATEST("
        f"COALESCE({nanat_subregion_sql}, 0.0), "
        f"COALESCE({nt_subregion_sql}, 0.0)"
        "))"
    )
    specificity_expr = (
        "LEAST(1.0, "
        f"COALESCE({nanat_score_sql}, 0.0) / "
        f"GREATEST(COALESCE({nanat_subregion_sql}, 0.0), 0.01)"
        ")"
    )
    return f"""
        GREATEST(
            COALESCE({inst_score_sql}, 0.0),
            COALESCE({position_score_sql}, 0.0),
            {alpha} * {subregion_expr}
                + (1.0 - {alpha}) * {specificity_expr}
        )
    """


def _build_rev_pos_country_query(rev_pos_tab: str) -> str:
    return f"""
    SELECT DISTINCT
        CAST(rp.user_id AS BIGINT) AS user_id,
        COALESCE(cw.std_country, initcap(CAST(rp.country AS VARCHAR))) AS country_std,
        1.0::DOUBLE AS position_score
    FROM {rev_pos_tab} AS rp
    LEFT JOIN _country_cw AS cw
      ON upper(CAST(rp.country AS VARCHAR)) = cw.key_upper
    WHERE rp.user_id IS NOT NULL
      AND rp.country IS NOT NULL
      AND trim(CAST(rp.country AS VARCHAR)) <> ''
    """


def _build_rev_indiv_norm_query(
    rev_indiv_tab: str,
    rev_indiv_cols: list[str],
    rev_pos_country_tab: str,
    alpha: float = cfg.BUILD_SUBREGION_BOOST_ALPHA,
    competition_weight: float = cfg.BUILD_COUNTRY_COMPETITION_WEIGHT,
    competition_threshold: float = cfg.BUILD_COUNTRY_COMPETITION_THRESHOLD,
) -> str:
    country_score_stage4_expr = _optional_select_expr(
        rev_indiv_cols,
        ["country_score"],
        "country_score_stage4",
        "DOUBLE",
    )
    institution_score_expr = _optional_select_expr(
        rev_indiv_cols,
        ["institution_score", "inst_score"],
        "institution_score",
        "DOUBLE",
    )
    subregion_expr = _optional_select_expr(rev_indiv_cols, ["subregion"], "subregion")
    nanat_score_expr = _optional_select_expr(
        rev_indiv_cols,
        ["nanat_score"],
        "nanat_score",
        "DOUBLE",
    )
    nanat_subregion_expr = _optional_select_expr(
        rev_indiv_cols,
        ["nanat_subregion_score"],
        "nanat_subregion_score",
        "DOUBLE",
    )
    nt_subregion_expr = _optional_select_expr(
        rev_indiv_cols,
        ["nt_subregion_score"],
        "nt_subregion_score",
        "DOUBLE",
    )
    country_uncertain_expr = _optional_select_expr(
        rev_indiv_cols,
        ["country_uncertain_ind"],
        "country_uncertain_ind",
        "INTEGER",
    )
    exclude_cols = {
        "country_score",
        "institution_score",
        "inst_score",
        "subregion",
        "nanat_score",
        "nanat_subregion_score",
        "nt_subregion_score",
        "country_uncertain_ind",
    }
    present_exclude = [col for col in rev_indiv_cols if col in exclude_cols]
    star_expr = "ri.*"
    if present_exclude:
        star_expr = "ri.* EXCLUDE (" + ", ".join(present_exclude) + ")"

    country_base_expr = _h1b_country_base_expr(
        inst_score_sql="b.institution_score",
        position_score_sql="pc.position_score",
        nanat_score_sql="b.nanat_score",
        nanat_subregion_sql="b.nanat_subregion_score",
        nt_subregion_sql="b.nt_subregion_score",
        alpha=alpha,
    )

    if competition_weight != 0.0:
        max_other_expr = (
            "CASE "
            "WHEN sb.country_std = uc.max1_country_std THEN uc.max2_cs "
            "ELSE COALESCE(uc.max1_cs, 0.0) "
            "END"
        )
        competition_mult_expr = (
            f"GREATEST(0.0, 1.0 - {competition_weight} "
            f"* GREATEST(0.0, ({max_other_expr}) - {competition_threshold}))"
        )
        competition_ctes = f""",
    user_country_ranked AS (
        SELECT
            user_id,
            country_std,
            country_score_h1b_base,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY country_score_h1b_base DESC, country_std
            ) AS cs_rank
        FROM scored_base
    ),
    user_top2_country AS (
        SELECT
            user_id,
            MAX(CASE WHEN cs_rank = 1 THEN country_score_h1b_base END) AS max1_cs,
            MAX(CASE WHEN cs_rank = 1 THEN country_std END) AS max1_country_std,
            COALESCE(MAX(CASE WHEN cs_rank = 2 THEN country_score_h1b_base END), 0.0) AS max2_cs
        FROM user_country_ranked
        WHERE cs_rank <= 2
        GROUP BY user_id
    )"""
        competition_join = "LEFT JOIN user_top2_country AS uc ON sb.user_id = uc.user_id"
    else:
        max_other_expr = "NULL::DOUBLE"
        competition_mult_expr = "1.0"
        competition_ctes = ""
        competition_join = ""

    return f"""
    WITH base AS (
        SELECT
            {star_expr},
            {country_score_stage4_expr},
            {institution_score_expr},
            {subregion_expr},
            {nanat_score_expr},
            {nanat_subregion_expr},
            {nt_subregion_expr},
            {country_uncertain_expr},
            COALESCE(cw.std_country, initcap(CAST(ri.country AS VARCHAR))) AS country_std
        FROM {rev_indiv_tab} AS ri
        LEFT JOIN _country_cw AS cw
          ON upper(CAST(ri.country AS VARCHAR)) = cw.key_upper
    ),
    scored_base AS (
        SELECT
            b.*,
            COALESCE(pc.position_score, 0.0) AS position_score,
            {country_base_expr} AS country_score_h1b_base
        FROM base AS b
        LEFT JOIN {rev_pos_country_tab} AS pc
          ON b.user_id = pc.user_id
         AND b.country_std = pc.country_std
    ){competition_ctes}
    SELECT
        sb.*,
        {competition_mult_expr} AS country_competition_mult,
        {max_other_expr} AS max_other_country_score,
        LEAST(
            1.0,
            GREATEST(0.0, sb.country_score_h1b_base * {competition_mult_expr})
        ) AS country_score
    FROM scored_base AS sb
    {competition_join}
    """


def _gradyr_score_expr(
    f1_year_sql: str,
    rev_year_sql: str,
    *,
    null_default: float,
    year_buffer: int,
    decay_power: float = cfg.BUILD_GRADYR_SCORE_DECAY_POWER,
) -> str:
    linear_expr = (
        f"1.0 - ABS({f1_year_sql} - {rev_year_sql})::FLOAT / ({year_buffer} + 1.0)"
    )
    return f"""
        CASE
            WHEN {f1_year_sql} IS NULL OR {rev_year_sql} IS NULL
                THEN {null_default}
            ELSE POWER(GREATEST(0.0, {linear_expr}), {decay_power})
        END
    """


def _format_total_score_breakdown(row) -> list[str]:
    total = _maybe_float(row.get("total_score"))
    if total is None:
        return []

    grad = _maybe_float(row.get("gradyr_score"))
    country = _maybe_float(row.get("country_score"))
    inst = _maybe_float(row.get("inst_score"))
    field = _maybe_float(row.get("field_score"))
    emp = _maybe_float(row.get("employer_score"))
    n_f1_employers = _maybe_float(row.get("n_f1_employers"))
    w_emp_eff = _maybe_float(row.get("w_emp_eff"))

    if grad is None or country is None or inst is None or field is None:
        return [f"total = {total:.4f} (score components unavailable)"]

    w_grad = float(cfg.BUILD_W_GRADYR)
    w_country = float(cfg.BUILD_W_COUNTRY)
    w_inst = float(cfg.BUILD_W_INST)
    w_field = float(cfg.BUILD_W_FIELD)
    w_no_emp = w_grad + w_country + w_inst + w_field
    if w_no_emp <= 0:
        return [f"total = {total:.4f} (invalid non-employer weights)"]

    nonemp_num = (
        w_grad * grad
        + w_country * country
        + w_inst * inst
        + w_field * field
    )
    nonemp_avg = nonemp_num / w_no_emp if w_no_emp > 0 else 0.0

    if cfg.BUILD_MULTIPLICATIVE_SCORE:
        floor = 1e-9

        def _fmt_pow_term(label: str, score: float, exponent: float) -> str:
            if score <= floor:
                return f"{label}(max({score:.3f},1e-9))^{exponent:.3f}"
            return f"{label}({score:.3f})^{exponent:.3f}"

        if emp is not None:
            if w_emp_eff is None and n_f1_employers is not None:
                w_emp_eff = float(
                    cfg.BUILD_W_EMP_MAX
                    * (1.0 - np.exp(-float(n_f1_employers) / float(cfg.BUILD_EMP_N_SCALE)))
                )
            if w_emp_eff is None:
                w_emp_eff = 0.0
            nonemp_weight = 1.0 - w_emp_eff
            return [
                (
                    "total = "
                    f"{_fmt_pow_term('emp', emp, w_emp_eff)} * "
                    f"{_fmt_pow_term('nonemp', nonemp_avg, nonemp_weight)} "
                    f"=> {total:.4f}"
                ),
                (
                    f"nonemp = ({w_grad:.2f}*grad({grad:.3f}) + "
                    f"{w_country:.2f}*country({country:.3f}) + "
                    f"{w_inst:.2f}*inst({inst:.3f}) + "
                    f"{w_field:.2f}*field({field:.3f})) / {w_no_emp:.2f}"
                ),
            ]

        return [
            (
                f"total = nonemp({nonemp_avg:.3f}) => {total:.4f}"
            ),
            (
                f"nonemp = ({w_grad:.2f}*grad({grad:.3f}) + "
                f"{w_country:.2f}*country({country:.3f}) + "
                f"{w_inst:.2f}*inst({inst:.3f}) + "
                f"{w_field:.2f}*field({field:.3f})) / {w_no_emp:.2f}"
            ),
        ]

    if emp is not None:
        if w_emp_eff is None and n_f1_employers is not None:
            w_emp_eff = float(
                cfg.BUILD_W_EMP_MAX
                * (1.0 - np.exp(-float(n_f1_employers) / float(cfg.BUILD_EMP_N_SCALE)))
            )
        if w_emp_eff is None:
            w_emp_eff = 0.0
        emp_part = w_emp_eff * emp
        nonemp_weight = 1.0 - w_emp_eff
        nonemp_part = nonemp_weight * nonemp_avg
        return [
            (
                f"total = {w_emp_eff:.3f}*emp({emp:.3f})={emp_part:.3f} + "
                f"{nonemp_weight:.3f}*nonemp({nonemp_avg:.3f})={nonemp_part:.3f} "
                f"=> {total:.4f}"
            ),
            (
                f"nonemp = ({w_grad:.2f}*grad({grad:.3f}) + "
                f"{w_country:.2f}*country({country:.3f}) + "
                f"{w_inst:.2f}*inst({inst:.3f}) + "
                f"{w_field:.2f}*field({field:.3f})) / {w_no_emp:.2f}"
            ),
        ]

    return [
        (
            f"total = nonemp({nonemp_avg:.3f}) => {total:.4f}"
        ),
        (
            f"nonemp = ({w_grad:.2f}*grad({grad:.3f}) + "
            f"{w_country:.2f}*country({country:.3f}) + "
            f"{w_inst:.2f}*inst({inst:.3f}) + "
            f"{w_field:.2f}*field({field:.3f})) / {w_no_emp:.2f}"
        ),
    ]


def _sql_rev_degree_bucket_expr(col: str) -> str:
    """Map stage-04 cleaned degree labels to the stage-05 degree buckets."""
    col_l = f"lower(trim(COALESCE(CAST({col} AS VARCHAR), '')))"
    return f"""
        CASE
            WHEN {col_l} IN ('doctor', 'doctoral', 'doctorate', 'phd', 'ph.d') THEN 'Doctor'
            WHEN {col_l} = 'mba' THEN 'MBA'
            WHEN {col_l} IN ('master', 'masters', 'master''s') THEN 'Master'
            WHEN {col_l} IN ('bachelor', 'bachelors', 'bachelor''s') THEN 'Bachelor'
            WHEN {col_l} IN ('associate', 'associates', 'associate''s') THEN 'Associate'
            WHEN {col_l} = 'non_degree' THEN 'Non-Degree'
            WHEN {col_l} IN ('missing', 'unknown', '') THEN 'Missing'
            WHEN {col_l} = 'hs_or_below' THEN 'High School'
            ELSE initcap(trim(CAST({col} AS VARCHAR)))
        END
    """


def _sql_emp_year_overlap_expr(
    pos_start_col: str,
    pos_end_col: str,
    min_f1_col: str,
    max_f1_col: str,
    year_buffer: int = cfg.EMPLOYER_MATCH_YEAR_BUFFER,
) -> str:
    """Return SQL for buffered overlap between Revelio position years and the F1 window."""
    year_buffer = max(0, int(year_buffer))
    return (
        f"(({pos_start_col} IS NULL OR {pos_start_col} <= {max_f1_col} + {year_buffer}) "
        f"AND COALESCE({pos_end_col}, YEAR(CURRENT_DATE)) >= {min_f1_col} - {year_buffer})"
    )


def get_runtime_table_names(testing: bool | None = None) -> dict[str, str]:
    """Resolve the materialized runtime table names for the current config."""
    if testing is None:
        testing = cfg.TESTING_ENABLED
    use_testing_prefix = bool(testing) and bool(cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES)
    pfx = cfg.TESTING_TABLE_PREFIX if use_testing_prefix else None

    def _name(test_suffix: str, prod_name: str) -> str:
        if pfx:
            return f"{pfx}_{test_suffix}"
        return prod_name

    return {
        "f1_school_rows": _name("f1_school_rows", "_f1_school_rows"),
        "f1_educ_spells": _name("f1_educ_spells", "_f1_educ_spells"),
        "f1_opt_employers": _name("f1_opt_employers", "_f1_opt_employers"),
        "rev_educ_school": _name("rev_educ_school", "_rev_educ_school"),
        "rev_educ_school_rows": _name("rev_educ_school_rows", "_rev_educ_school_rows"),
        "candidates": _name("candidates", "_f1_candidates"),
        "rev_pos_full": _name("rev_pos_full", "_f1_rev_pos_full"),
        "rev_pos_user_stats": _name("rev_pos_user_stats", "_f1_rev_pos_user_stats"),
        "emp_idf": _name("emp_idf", "_f1_emp_idf"),
        "f1_employer_keys": _name("f1_employer_keys", "_f1_employer_keys"),
        "emp_match_pairs": _name("emp_match_pairs", "_f1_emp_match_pairs"),
        "emp_assigned": _name("emp_assigned", "_f1_emp_assigned"),
        "emp_seq_scores": _name("emp_seq_scores", "_f1_emp_seq_scores"),
        "merge_scored": _name("merge_scored", "_f1_merge_scored"),
        "merge_scored_emp_filt": _name("merge_scored_emp_filt", "_f1_merge_scored_emp_filt"),
        "match_filt": _name("match_filt", "_f1_match_filt"),
        "weighted": _name("weighted", "_f1_weighted"),
        "baseline": _name("final", "_f1_baseline"),
        "person_agg": _name("person_agg", "_f1_person_agg"),
    }


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

_F1_INST_UNITID_REQUIRED_COLS = {
    "f1_row_num",
    "unitid",
}

_RELABEL_SAMPLE_REQUIRED_COLS = {
    "original_row_num",
    "person_id",
    "sample_unitid",
    "sample_grad_year",
    "sample_relabel_year",
    "sample_relabel_type",
    "sample_role",
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


def _atomic_tmp_path(path: str | Path) -> Path:
    out_path = Path(path)
    return out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")


def write_query_to_parquet(query: str, out_path: str, overwrite: bool = False, con=con_f1) -> None:
    t0 = time.perf_counter()
    final_path = Path(out_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        if final_path.stat().st_size == 0:
            print(f"  Removing empty file: {final_path}")
            final_path.unlink()
        elif not overwrite:
            print(f"  Skipping (exists): {final_path}")
            return
    tmp_path = _atomic_tmp_path(final_path)
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        esc = _sql_escape_path(str(tmp_path))
        con.sql(f"COPY ({query}) TO '{esc}' (FORMAT parquet)")
        tmp_path.replace(final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    elapsed = time.perf_counter() - t0
    print(f"  Wrote: {final_path} ({_fmt_elapsed(elapsed)})")


def _relabel_sample_cache_is_usable(
    con,
    cache_path: str | None,
    *,
    required_cols: set[str] = _RELABEL_SAMPLE_REQUIRED_COLS,
) -> bool:
    if not cache_path or not os.path.exists(cache_path):
        return False
    try:
        cols = {
            row[0]
            for row in con.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{_sql_escape_path(cache_path)}')"
            ).fetchall()
        }
    except Exception:
        return False
    return required_cols.issubset(cols)


def _materialize_relabel_sample(
    con,
    relabel_sample_q: str,
    *,
    cache_path: str | None = cfg.RELABEL_SAMPLE_CACHE_PARQUET,
    force_rebuild: bool = cfg.RELABEL_SAMPLE_FORCE_REBUILD,
    table_name: str = "_f1_foia_relabel_sample",
) -> str:
    use_cache = _relabel_sample_cache_is_usable(con, cache_path) and not force_rebuild
    if use_cache:
        print(f"  Reusing cached relabel sample: {cache_path}")
        materialize_table(
            table_name,
            f"SELECT * FROM read_parquet('{_sql_escape_path(cache_path)}')",
            con=con,
        )
        return table_name

    if force_rebuild and cache_path:
        print(f"  Force rebuilding relabel sample: {cache_path}")
    else:
        print("  Building relabel sample parquet cache...")
    materialize_table(table_name, relabel_sample_q, con=con)
    if cache_path:
        write_query_to_parquet(
            query=f"SELECT * FROM {table_name}",
            out_path=cache_path,
            overwrite=True,
            con=con,
        )
    return table_name


def _load_relabel_program_artifacts(con=con_f1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load treated events + matched controls from the econ relabel v2 artifact."""
    if relabel_base is None or relabel_v2 is None:
        raise ModuleNotFoundError(
            "Could not import f1_foia.econ_relabels_opt_usage(_v2); "
            "relabel-program sample restriction is unavailable."
        )
    relabel_path = cfg.ECON_RELABELS_PARQUET
    if not relabel_path or not os.path.exists(relabel_path):
        raise FileNotFoundError(
            "Econ relabel v2 parquet not found. "
            f"Checked: {relabel_path}"
        )

    relabel_df = con.sql(
        f"SELECT * FROM read_parquet('{_sql_escape_path(relabel_path)}')"
    ).df()
    if relabel_df.empty:
        raise ValueError(f"Econ relabel v2 parquet is empty: {relabel_path}")

    treated_events = (
        relabel_df.loc[
            relabel_df["event_flag"] == 1,
            ["unitid", "relabel_year", "relabel_type"],
        ]
        .dropna(subset=["unitid", "relabel_year", "relabel_type"])
        .drop_duplicates()
        .copy()
    )
    if treated_events.empty:
        raise ValueError(
            "No treated relabel events found in econ relabel v2 parquet."
        )
    treated_events["unitid"] = pd.to_numeric(
        treated_events["unitid"], errors="coerce"
    ).astype("Int64")
    treated_events["relabel_year"] = pd.to_numeric(
        treated_events["relabel_year"], errors="coerce"
    ).astype("Int64")
    treated_events = treated_events.dropna(subset=["unitid", "relabel_year"]).copy()
    treated_events["unitid"] = treated_events["unitid"].astype("int64")
    treated_events["relabel_year"] = treated_events["relabel_year"].astype("int64")

    matched_pairs = relabel_v2._match_treated_to_untreated_cohorts(con=con, relabel_df=relabel_df)
    if matched_pairs.empty:
        raise ValueError("No treated/control pairs returned by the econ relabel v2 matcher.")

    return treated_events, matched_pairs


def _build_f1_relabel_sample_query(
    f1_foia_tab: str,
    f1_inst_unitid_tab: str,
    treated_events_tab: str,
    matched_pairs_tab: str,
    gradyear_window: int,
) -> str:
    """Restrict FOIA rows to treated + matched-control relabel programs around event time."""
    if relabel_base is None or relabel_v2 is None:
        raise ModuleNotFoundError(
            "Could not import the econ relabel helper modules required for sample restriction."
        )

    norm_cip_expr = relabel_base.normalize_cip_sql("f.major_1_cip_code")
    school_name_key_expr = "lower(trim(CAST(f.school_name AS VARCHAR)))"
    inst_school_name_key_expr = "lower(trim(CAST(u.school_name AS VARCHAR)))"
    campus_city_clean_expr = sql_normalize_expr("f.campus_city")
    campus_state_clean_expr = sql_state_name_to_abbr_expr("f.campus_state")
    campus_zip_clean_expr = sql_clean_zip_expr("f.campus_zip_code")
    treated_branches: list[str] = []
    control_branches: list[str] = []
    for spec in relabel_base.RELABEL_SPECS:
        # For the merge sample, keep the full 45.06xx family on both treated and matched-control
        # sides. This intentionally does not exclude 45.0603.
        sample_cip_pred = relabel_base._cip_prefix_pred("cip6", spec["source_prefix"])
        treated_branches.append(
            f"""
            SELECT
                f.foia_row_num,
                f.grad_year,
                f.unitid,
                te.relabel_year,
                te.relabel_type,
                'treated' AS sample_role
            FROM foia_base AS f
            JOIN {treated_events_tab} AS te
              ON f.unitid = CAST(te.unitid AS BIGINT)
             AND te.relabel_type = '{spec["name"]}'
            WHERE ({sample_cip_pred})
              AND ABS(f.grad_year - CAST(te.relabel_year AS INTEGER)) <= {gradyear_window}
            """
        )
        control_branches.append(
            f"""
            SELECT
                f.foia_row_num,
                f.grad_year,
                f.unitid,
                mp.relabel_year,
                mp.relabel_type,
                'matched_control' AS sample_role
            FROM foia_base AS f
            JOIN {matched_pairs_tab} AS mp
              ON f.unitid = CAST(mp.control_unitid AS BIGINT)
             AND mp.relabel_type = '{spec["name"]}'
            WHERE ({sample_cip_pred})
              AND ABS(f.grad_year - CAST(mp.relabel_year AS INTEGER)) <= {gradyear_window}
            """
        )

    sample_union_sql = "\nUNION ALL\n".join(treated_branches + control_branches)
    return f"""
    WITH foia_filtered AS (
        SELECT
            f.*,
            CAST(f.original_row_num AS BIGINT) AS foia_row_num,
            CAST(EXTRACT(YEAR FROM f.program_end_date) AS INTEGER) AS grad_year,
            LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0') AS cip6,
            {school_name_key_expr} AS school_name_key,
            {campus_city_clean_expr} AS campus_city_clean,
            {campus_state_clean_expr} AS campus_state_clean,
            {campus_zip_clean_expr} AS campus_zip_clean
        FROM {f1_foia_tab} AS f
        WHERE f.original_row_num IS NOT NULL
          AND f.program_end_date IS NOT NULL
          AND (f.year_int IS NULL OR f.year_int = CAST(EXTRACT(YEAR FROM f.program_end_date) AS INTEGER))
          AND upper(COALESCE(f.student_edu_level_desc, '')) LIKE '%MASTER%'
          AND {norm_cip_expr} IS NOT NULL
    ),
    inst_unitid AS (
        SELECT DISTINCT
            CAST(u.f1_row_num AS BIGINT) AS f1_row_num,
            CAST(u.unitid AS BIGINT) AS unitid,
            CAST(u.school_name AS VARCHAR) AS school_name,
            {inst_school_name_key_expr} AS school_name_key,
            CAST(u.f1_city_clean AS VARCHAR) AS f1_city_clean,
            CAST(u.f1_state_clean AS VARCHAR) AS f1_state_clean,
            CAST(u.f1_zip_clean AS VARCHAR) AS f1_zip_clean
        FROM {f1_inst_unitid_tab} AS u
        WHERE CAST(u.school_name AS VARCHAR) IS NOT NULL
          AND CAST(u.unitid AS BIGINT) IS NOT NULL
    ),
    school_unitid_counts AS (
        SELECT
            school_name_key,
            COUNT(DISTINCT unitid) AS n_unitids
        FROM inst_unitid
        GROUP BY school_name_key
    ),
    site_candidates AS (
        SELECT DISTINCT
            f.foia_row_num,
            i.unitid
        FROM foia_filtered AS f
        JOIN inst_unitid AS i
          ON f.school_name_key = i.school_name_key
         AND COALESCE(f.campus_state_clean, '') != ''
         AND COALESCE(f.campus_state_clean, '') = COALESCE(i.f1_state_clean, '')
         AND (
             COALESCE(f.campus_city_clean, '') = ''
             OR COALESCE(f.campus_city_clean, '') = COALESCE(i.f1_city_clean, '')
         )
         AND (
             COALESCE(f.campus_zip_clean, '') = ''
             OR COALESCE(f.campus_zip_clean, '') = COALESCE(i.f1_zip_clean, '')
         )
    ),
    school_unique_fallback AS (
        SELECT
            f.foia_row_num,
            MIN(i.unitid) AS unitid
        FROM foia_filtered AS f
        JOIN inst_unitid AS i
          ON f.school_name_key = i.school_name_key
        JOIN school_unitid_counts AS s
          ON f.school_name_key = s.school_name_key
        WHERE s.n_unitids = 1
        GROUP BY f.foia_row_num
    ),
    resolved_unitid AS (
        SELECT
            foia_row_num,
            MIN(unitid) AS unitid
        FROM (
            SELECT foia_row_num, unitid FROM site_candidates
            UNION ALL
            SELECT sf.foia_row_num, sf.unitid
            FROM school_unique_fallback AS sf
            WHERE sf.foia_row_num NOT IN (SELECT foia_row_num FROM site_candidates)
        )
        GROUP BY foia_row_num
        HAVING COUNT(DISTINCT unitid) = 1
    ),
    foia_base AS (
        SELECT
            f.*,
            r.unitid
        FROM foia_filtered AS f
        JOIN resolved_unitid AS r
          ON f.foia_row_num = r.foia_row_num
    ),
    sample_union AS (
        {sample_union_sql}
    ),
    ranked AS (
        SELECT
            su.*,
            ROW_NUMBER() OVER(
                PARTITION BY su.foia_row_num
                ORDER BY
                    ABS(su.grad_year - CAST(su.relabel_year AS INTEGER)),
                    CASE WHEN su.sample_role = 'treated' THEN 0 ELSE 1 END,
                    su.relabel_year,
                    su.relabel_type,
                    su.unitid
            ) AS sample_rank
        FROM sample_union AS su
    )
    SELECT
        f.*,
        r.unitid              AS sample_unitid,
        r.grad_year           AS sample_grad_year,
        CAST(r.relabel_year AS INTEGER) AS sample_relabel_year,
        r.relabel_type        AS sample_relabel_type,
        r.sample_role
    FROM {f1_foia_tab} AS f
    JOIN ranked AS r
      ON CAST(f.original_row_num AS BIGINT) = r.foia_row_num
    WHERE r.sample_rank = 1
    """


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


def _build_f1_school_rows_unitid_query(
    f1_foia_tab: str,
    f1_inst_unitid_tab: str,
    cw_tab: str | None = None,
) -> str:
    """Build row-level F1 school records with institution-level UNITID resolution.

    This path is used by the stage-04 `rev_match_ready` candidate join. It resolves
    each FOIA row to a unique UNITID using the institution crosswalk on
    (school_name, campus_state, optional campus_city / campus_zip), with a fallback
    to school-name-only when the school maps to exactly one UNITID in the crosswalk.
    """
    school_pre_counts_cte = ""
    school_pre_counts_join = ""
    school_match_count_pre_expr = "CASE WHEN rrn.resolved_unitid_row IS NOT NULL THEN 1 ELSE 0 END"
    if cw_tab:
        school_pre_counts_cte = f"""
    school_pre_counts AS (
        SELECT
            f1_school_name,
            COUNT(DISTINCT rev_instname_clean) AS school_match_count_pre
        FROM {cw_tab}
        GROUP BY f1_school_name
    ),
"""
        school_pre_counts_join = """
    LEFT JOIN school_pre_counts AS spc
        ON rr.school_name = spc.f1_school_name
"""
        school_match_count_pre_expr = (
            "COALESCE(spc.school_match_count_pre, "
            "CASE WHEN rrn.resolved_unitid_row IS NOT NULL THEN 1 ELSE 0 END)"
        )

    campus_city_clean = sql_normalize_expr("f.campus_city")
    campus_state_clean = sql_state_name_to_abbr_expr("f.campus_state")
    campus_zip_clean = sql_clean_zip_expr("f.campus_zip_code")
    inst_school_name_key_expr = "lower(trim(CAST(u.school_name AS VARCHAR)))"

    return f"""
    WITH
{school_pre_counts_cte}
    inst_unitid AS (
        SELECT DISTINCT
            CAST(u.f1_row_num AS BIGINT) AS f1_row_num,
            CAST(u.unitid AS BIGINT) AS unitid,
            {inst_school_name_key_expr} AS school_name_key,
            CAST(u.f1_city_clean AS VARCHAR) AS f1_city_clean,
            CAST(u.f1_state_clean AS VARCHAR) AS f1_state_clean,
            CAST(u.f1_zip_clean AS VARCHAR) AS f1_zip_clean
        FROM {f1_inst_unitid_tab} AS u
        WHERE CAST(u.school_name AS VARCHAR) IS NOT NULL
          AND CAST(u.unitid AS BIGINT) IS NOT NULL
    ),
    school_unitid_counts AS (
        SELECT
            school_name_key,
            COUNT(DISTINCT unitid) AS n_unitids
        FROM inst_unitid
        GROUP BY school_name_key
    ),
    raw_rows AS (
        SELECT
            ROW_NUMBER() OVER() AS f1_merge_row_id,
            f.*,
            lower(trim(CAST(f.school_name AS VARCHAR))) AS school_name_key,
            {campus_city_clean}  AS campus_city_clean,
            {campus_state_clean} AS campus_state_clean,
            {campus_zip_clean}   AS campus_zip_clean
        FROM {f1_foia_tab} AS f
        WHERE f.school_name IS NOT NULL
    ),
    site_candidates AS (
        SELECT DISTINCT
            rr.f1_merge_row_id,
            iu.f1_row_num,
            iu.unitid
        FROM raw_rows AS rr
        JOIN inst_unitid AS iu
          ON rr.school_name_key = iu.school_name_key
         AND COALESCE(rr.campus_state_clean, '') != ''
         AND COALESCE(rr.campus_state_clean, '') = COALESCE(iu.f1_state_clean, '')
         AND (
             COALESCE(rr.campus_city_clean, '') = ''
             OR COALESCE(rr.campus_city_clean, '') = COALESCE(iu.f1_city_clean, '')
         )
         AND (
             COALESCE(rr.campus_zip_clean, '') = ''
             OR COALESCE(rr.campus_zip_clean, '') = COALESCE(iu.f1_zip_clean, '')
         )
    ),
    school_unique_fallback AS (
        SELECT
            rr.f1_merge_row_id,
            MIN(iu.unitid) AS unitid
        FROM raw_rows AS rr
        JOIN inst_unitid AS iu
          ON rr.school_name_key = iu.school_name_key
        JOIN school_unitid_counts AS suc
          ON rr.school_name_key = suc.school_name_key
        WHERE suc.n_unitids = 1
        GROUP BY rr.f1_merge_row_id
    ),
    row_resolution AS (
        SELECT
            resolved.f1_merge_row_id,
            NULL::BIGINT AS resolved_f1_row_num_row,
            MIN(resolved.unitid) AS resolved_unitid_row
        FROM (
            SELECT sc.f1_merge_row_id, sc.unitid
            FROM site_candidates AS sc
            UNION ALL
            SELECT suf.f1_merge_row_id, suf.unitid
            FROM school_unique_fallback AS suf
            WHERE suf.f1_merge_row_id NOT IN (SELECT f1_merge_row_id FROM site_candidates)
        ) AS resolved
        GROUP BY resolved.f1_merge_row_id
        HAVING COUNT(DISTINCT resolved.unitid) = 1
    )
    SELECT
        rr.* EXCLUDE (school_name_key, campus_city_clean, campus_state_clean, campus_zip_clean),
        {school_match_count_pre_expr} AS school_match_count_pre,
        {school_match_count_pre_expr} AS school_match_count_post_row,
        rrn.resolved_f1_row_num_row,
        rrn.resolved_unitid_row
    FROM raw_rows AS rr
    LEFT JOIN row_resolution AS rrn
        ON rr.f1_merge_row_id = rrn.f1_merge_row_id
{school_pre_counts_join}
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
            -- 4-digit CIP family
            {cip_code_to_cip4_sql('MODE(major_1_cip_code)')} AS f1_cip4,
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
                WHEN COUNT(*) = COUNT(resolved_unitid_row)
                 AND COUNT(DISTINCT resolved_unitid_row) = 1
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
# Stage 1a: Build targeted FOIA rows for matching
# ---------------------------------------------------------------------------

def _build_f1_match_unit_rows_query(
    f1_foia_tab: str,
    *,
    person_shard_count: int | None = None,
    person_shard_id: int | None = None,
) -> str:
    """Restrict FOIA rows to the intended matching unit and remap stage-05 person_id.

    The stage-05 matching target is one raw ``individual_key`` × filing year after
    keeping only rows where the FOIA filing year equals the program end year. This
    avoids pooling the same ``individual_key`` across multiple completion years
    into one stage-local target. We preserve the upstream linked ``person_id`` as
    ``linked_person_id`` for diagnostics, but overwrite ``person_id`` in this
    stage-local source so downstream joins and aggregations operate at the
    individual-key × year grain.
    """
    shard_where_sql = ""
    if person_shard_count is not None or person_shard_id is not None:
        if person_shard_count is None or person_shard_id is None:
            raise ValueError("Shard filtering on match-unit rows requires both `person_shard_count` and `person_shard_id`.")
        shard_where_sql = (
            f"WHERE MOD(ABS(CAST(km.match_person_id AS BIGINT)), {int(person_shard_count)}) = {int(person_shard_id)}"
        )

    return f"""
    WITH eligible AS (
        SELECT *
        FROM {f1_foia_tab} AS f
        WHERE f.program_end_date IS NOT NULL
          AND f.year_int IS NOT NULL
          AND f.year_int = CAST(EXTRACT(YEAR FROM TRY_CAST(f.program_end_date AS DATE)) AS INTEGER)
          AND CAST(f.individual_key AS VARCHAR) IS NOT NULL
          AND trim(CAST(f.individual_key AS VARCHAR)) <> ''
    ),
    key_map AS (
        SELECT
            individual_key_norm,
            match_year,
            ROW_NUMBER() OVER (ORDER BY individual_key_norm, match_year) AS match_person_id
        FROM (
            SELECT DISTINCT
                trim(CAST(individual_key AS VARCHAR)) AS individual_key_norm,
                year_int AS match_year
            FROM eligible
        )
    )
    SELECT
        e.* EXCLUDE (person_id),
        e.person_id AS linked_person_id,
        km.match_person_id AS person_id
    FROM eligible AS e
    JOIN key_map AS km
      ON trim(CAST(e.individual_key AS VARCHAR)) = km.individual_key_norm
     AND e.year_int = km.match_year
    {shard_where_sql}
    """


def _build_testing_f1_foia_source_query(
    f1_foia_base_src: str,
    *,
    n_sample: int,
    seed: int,
    individual_keys_pin: list[str] | None = None,
    person_ids_pin: list[int] | None = None,
    school_pin: str | None = None,
    country_pin: str | None = None,
) -> str:
    """Build the Stage-05 testing source query over F1 FOIA rows."""
    individual_keys_pin = _normalize_testing_individual_keys(individual_keys_pin)
    person_ids_pin = _normalize_testing_person_ids(person_ids_pin)
    eligible_keys_query = _build_testing_eligible_keys_query(
        f1_foia_base_src,
        school_pin=school_pin,
        country_pin=country_pin,
    )

    # Hash-order sampling is deterministic and avoids DuckDB sampling semantics
    # that can over-weight high-row-count keys before DISTINCT is materialized.
    sample_key_predicates = [
        "f.program_end_date IS NOT NULL",
        "f.year_int IS NOT NULL",
        "f.year_int = CAST(EXTRACT(YEAR FROM TRY_CAST(f.program_end_date AS DATE)) AS INTEGER)",
        "CAST(f.individual_key AS VARCHAR) IS NOT NULL",
        "trim(CAST(f.individual_key AS VARCHAR)) <> ''",
    ]
    if school_pin:
        sample_key_predicates.append(f"f.school_name = {repr(school_pin)}")
    if country_pin:
        sample_key_predicates.append(
            f"upper(trim(f.country_of_birth)) = upper(trim({repr(country_pin)}))"
        )
    sample_key_where_sql = " AND ".join(sample_key_predicates)

    if individual_keys_pin:
        individual_key_list_sql = ", ".join(repr(individual_key) for individual_key in individual_keys_pin)
        test_key_cte = f"""
        WITH sample_keys AS (
            SELECT DISTINCT trim(CAST(f.individual_key AS VARCHAR)) AS individual_key
            FROM {f1_foia_base_src} AS f
            WHERE {sample_key_where_sql}
              AND trim(CAST(f.individual_key AS VARCHAR)) IN ({individual_key_list_sql})
        )
        """
        sample_join_sql = "JOIN sample_keys AS s ON trim(CAST(f.individual_key AS VARCHAR)) = s.individual_key"
    elif person_ids_pin:
        person_id_list_sql = ", ".join(str(person_id) for person_id in person_ids_pin)
        test_key_cte = f"""
        WITH sample_person_ids AS (
            SELECT DISTINCT person_id
            FROM {f1_foia_base_src} AS f
            WHERE person_id IN ({person_id_list_sql})
              AND f.program_end_date IS NOT NULL
              AND f.year_int IS NOT NULL
              AND f.year_int = CAST(EXTRACT(YEAR FROM TRY_CAST(f.program_end_date AS DATE)) AS INTEGER)
        )
        """
        sample_join_sql = "JOIN sample_person_ids AS s ON f.person_id = s.person_id"
    else:
        test_key_cte = f"""
        WITH eligible_keys AS (
            {eligible_keys_query}
        ),
        sample_keys AS (
            SELECT individual_key
            FROM eligible_keys
            ORDER BY hash(individual_key, {int(seed)})
            LIMIT {int(n_sample)}
        )
        """
        sample_join_sql = "JOIN sample_keys AS s ON trim(CAST(f.individual_key AS VARCHAR)) = s.individual_key"

    return f"""(
        {test_key_cte}
        SELECT f.*
        FROM {f1_foia_base_src} AS f
        {sample_join_sql}
        WHERE f.program_end_date IS NOT NULL
          AND f.year_int IS NOT NULL
          AND f.year_int = CAST(EXTRACT(YEAR FROM TRY_CAST(f.program_end_date AS DATE)) AS INTEGER)
    )"""


def _normalize_testing_individual_keys(individual_keys: list[str] | None) -> list[str]:
    return [
        str(individual_key).strip()
        for individual_key in (individual_keys or [])
        if str(individual_key).strip()
    ]


def _normalize_testing_person_ids(person_ids: list[int] | None) -> list[int]:
    return [int(person_id) for person_id in (person_ids or [])]


def _build_testing_eligible_keys_query(
    f1_foia_base_src: str,
    *,
    school_pin: str | None = None,
    country_pin: str | None = None,
) -> str:
    sample_key_predicates = [
        "f.program_end_date IS NOT NULL",
        "f.year_int IS NOT NULL",
        "f.year_int = CAST(EXTRACT(YEAR FROM TRY_CAST(f.program_end_date AS DATE)) AS INTEGER)",
        "CAST(f.individual_key AS VARCHAR) IS NOT NULL",
        "trim(CAST(f.individual_key AS VARCHAR)) <> ''",
    ]
    if school_pin:
        sample_key_predicates.append(f"f.school_name = {repr(school_pin)}")
    if country_pin:
        sample_key_predicates.append(
            f"upper(trim(f.country_of_birth)) = upper(trim({repr(country_pin)}))"
        )
    sample_key_where_sql = " AND ".join(sample_key_predicates)
    return f"""
    SELECT DISTINCT trim(CAST(f.individual_key AS VARCHAR)) AS individual_key
    FROM {f1_foia_base_src} AS f
    WHERE {sample_key_where_sql}
    """


# ---------------------------------------------------------------------------
# Stage 2: Build Revelio education × school summary
# ---------------------------------------------------------------------------

def _build_rev_educ_school_query(
    rev_educ_tab: str,
    cw_tab: str,
    *,
    rev_educ_has_cip: bool,
    rev_educ_has_cip_score: bool,
    rev_educ_has_field_mapped_ind: bool,
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

    CIP4 is taken from the stage-04 scalar `cip` column when available, with a
    field-text heuristic fallback for legacy artifacts.
    """
    if rev_educ_has_cip:
        rev_cip_sql = "COALESCE(TRY_CAST(e.cip AS INTEGER), " + field_clean_to_cip4_sql("e.field_clean") + ")"
    else:
        rev_cip_sql = field_clean_to_cip4_sql("e.field_clean")
    rev_cip_score_sql = "TRY_CAST(e.cip_score AS DOUBLE)" if rev_educ_has_cip_score else "NULL::DOUBLE"
    rev_field_mapped_sql = (
        "TRY_CAST(e.field_mapped_ind AS INTEGER)"
        if rev_educ_has_field_mapped_ind
        else "NULL::INTEGER"
    )
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
            {rev_cip_sql} AS rev_cip4,
            {rev_cip_score_sql} AS rev_cip_score,
            {rev_field_mapped_sql} AS rev_field_mapped_ind
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
        eb.rev_cip4,
        eb.rev_cip_score,
        eb.rev_field_mapped_ind
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
        re.rev_cip4,
        re.rev_cip_score,
        re.rev_field_mapped_ind
    FROM {rev_educ_school_tab} AS re
    JOIN row_map AS rm
        ON re.f1_school_name = rm.f1_school_name
       AND re.rev_instname_clean = rm.rev_instname_clean
    """


def _build_rev_match_ready_educ_query(
    rev_match_ready_tab: str,
    rev_match_ready_cols: list[str],
) -> str:
    """Build a stage-04 merge_ready education table keyed directly by UNITID."""
    rev_cip_expr = _optional_select_expr(rev_match_ready_cols, ["cip"], "rev_cip_raw", "BIGINT")
    rev_cip_score_expr = _optional_select_expr(rev_match_ready_cols, ["cip_score"], "rev_cip_score", "DOUBLE")
    rev_field_mapped_expr = _optional_select_expr(
        rev_match_ready_cols,
        ["field_mapped_ind"],
        "rev_field_mapped_ind",
        "INTEGER",
    )
    school_match_expr = _optional_select_expr(
        rev_match_ready_cols,
        ["school_match_score"],
        "school_match_score",
        "DOUBLE",
    )
    university_raw_expr = _optional_select_expr(
        rev_match_ready_cols,
        ["university_raw"],
        "rev_university_raw",
    )
    degree_expr = _optional_select_expr(rev_match_ready_cols, ["degree_clean"], "degree_raw")
    ed_start_expr = _optional_select_expr(rev_match_ready_cols, ["ed_startdate"], "rev_educ_start_raw")
    ed_end_expr = _optional_select_expr(rev_match_ready_cols, ["ed_enddate"], "rev_educ_end_raw")
    education_number_expr = _optional_select_expr(
        rev_match_ready_cols,
        ["education_number"],
        "education_number",
        "BIGINT",
    )
    re_country_score_expr = _optional_select_expr(
        rev_match_ready_cols,
        ["country_score"],
        "re_country_score",
        "DOUBLE",
    )
    re_inst_score_expr = _optional_select_expr(
        rev_match_ready_cols,
        ["institution_score"],
        "re_inst_score",
        "DOUBLE",
    )
    nanat_score_expr = _optional_select_expr(
        rev_match_ready_cols,
        ["nanat_score"],
        "nanat_score",
        "DOUBLE",
    )
    nanat_subregion_expr = _optional_select_expr(
        rev_match_ready_cols,
        ["nanat_subregion_score"],
        "nanat_subregion_score",
        "DOUBLE",
    )
    nt_subregion_expr = _optional_select_expr(
        rev_match_ready_cols,
        ["nt_subregion_score"],
        "nt_subregion_score",
        "DOUBLE",
    )
    country_uncertain_expr = _optional_select_expr(
        rev_match_ready_cols,
        ["country_uncertain_ind"],
        "country_uncertain_ind",
        "INTEGER",
    )
    return f"""
    WITH mr_base AS (
        SELECT DISTINCT
            CAST(user_id AS BIGINT) AS user_id,
            CAST(unitid AS BIGINT) AS rev_unitid,
            CAST(country_candidate AS VARCHAR) AS country_candidate,
            {degree_expr},
            {ed_start_expr},
            {ed_end_expr},
            {university_raw_expr},
            {school_match_expr},
            {education_number_expr},
            {re_country_score_expr},
            {re_inst_score_expr},
            {nanat_score_expr},
            {nanat_subregion_expr},
            {nt_subregion_expr},
            {country_uncertain_expr},
            {rev_cip_expr},
            {rev_cip_score_expr},
            {rev_field_mapped_expr}
        FROM {rev_match_ready_tab}
        WHERE user_id IS NOT NULL
          AND country_candidate IS NOT NULL
          AND unitid IS NOT NULL
    )
    SELECT
        mr.user_id,
        mr.rev_university_raw,
        NULL::VARCHAR AS rev_instname_clean,
        mr.rev_unitid,
        COALESCE(mr.school_match_score, 1.0) AS school_match_score,
        0::INTEGER AS school_match_ambiguous_ind,
        COALESCE(cw.std_country, initcap(mr.country_candidate)) AS country_std,
        {_sql_rev_degree_bucket_expr("mr.degree_raw")} AS rev_degree_clean,
        YEAR(TRY_CAST(mr.rev_educ_start_raw AS DATE)) AS rev_educ_start_year,
        YEAR(TRY_CAST(mr.rev_educ_end_raw AS DATE)) AS rev_educ_end_year,
        mr.rev_educ_start_raw,
        mr.rev_educ_end_raw,
        mr.re_country_score,
        mr.re_inst_score,
        mr.nanat_score,
        mr.nanat_subregion_score,
        mr.nt_subregion_score,
        mr.country_uncertain_ind,
        TRY_CAST(mr.rev_cip_raw AS INTEGER) AS rev_cip4,
        mr.rev_cip_score,
        mr.rev_field_mapped_ind
    FROM mr_base AS mr
    LEFT JOIN _country_cw AS cw
      ON upper(mr.country_candidate) = cw.key_upper
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
    GRADYR_SCORE_DECAY_POWER: float = cfg.BUILD_GRADYR_SCORE_DECAY_POWER,
    FIELD_TARGET_CIP4: int = cfg.BUILD_FIELD_TARGET_CIP4,
    FIELD_NON_TARGET_CAP: float = cfg.BUILD_FIELD_NON_TARGET_CAP,
    FIELD_FILTER_MIN_SCORE: float = cfg.BUILD_FIELD_FILTER_MIN_SCORE,
    FIELD_CIP2_MATCH_MULTIPLIER: float = cfg.BUILD_FIELD_CIP2_MATCH_MULTIPLIER,
) -> str:
    """Hard-filter join: F1 spells × Revelio education → (spell_id, user_id) candidates.

    Join conditions (all enforced as hard filters — necessary to keep cross-join tractable
    and to eliminate obviously wrong matches):
      1. School:  f1_school_name matches via crosswalk, score > threshold
      2. Country: ri.country_std = f1.f1_country_std
      3. Degree:  levels must agree; 'Other'/'Missing'/'Non-Degree' pass through
      4. Gradyr:  |f1_prog_end_year - rev_educ_end_year| <= YEAR_HARD_BUFFER (NULL passes through)

    Soft scoring signals are computed here and combined in _build_merge_scored_query:
      country_score: H-1B-style country evidence score using the same formula as 03_indiv_merge
      inst_score:    institution-country evidence input used in that country score
      gradyr_score:  power-decayed year proximity within the YEAR_HARD_BUFFER window
                     (0.5 if either NULL)
      field_score:   stage-04 field candidate score, scaled by F1/Revelio CIP agreement
      degree_score:  retained for diagnostics; kept as a hard block upstream

    n_match_raw: count of distinct candidates per spell_id.
    """
    country_score_expr = "COALESCE(ri.country_score, 0.0)"
    inst_score_expr = "COALESCE(ri.institution_score, 0.0)"

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

    gradyr_score_expr = _gradyr_score_expr(
        "f1.f1_prog_end_year",
        "re.rev_educ_end_year",
        null_default=DATE_SCORE_NULL_DEFAULT,
        year_buffer=YEAR_HARD_BUFFER,
        decay_power=GRADYR_SCORE_DECAY_POWER,
    )
    field_score_expr = _field_score_expr(
        "f1.f1_cip4",
        "re.rev_cip4",
        "re.rev_cip_score",
        cip2_match_multiplier=FIELD_CIP2_MATCH_MULTIPLIER,
    )
    field_pass_expr = f"""
        CASE
            WHEN re.rev_field_mapped_ind = 1 AND COALESCE(({field_score_expr}), 0.0) < {FIELD_FILTER_MIN_SCORE}
                THEN 0
            ELSE 1
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
            f1.f1_cip4,
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
            ri.country_score_stage4    AS rev_country_score_raw,
            ri.institution_score       AS rev_inst_score_raw,
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
            re.rev_cip4,
            re.rev_cip_score,
            re.rev_field_mapped_ind,
            ({country_score_expr})     AS country_score,
            ({inst_score_expr})        AS inst_score,
            ({degree_score_expr})      AS degree_score,
            ({gradyr_score_expr})      AS gradyr_score,
            ({field_score_expr})       AS field_score,
            ({field_pass_expr})        AS field_candidate_pass_ind
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
                f1.f1_prog_end_year IS NULL
                OR re.rev_educ_end_year IS NULL
                OR ABS(f1.f1_prog_end_year - re.rev_educ_end_year) <= {YEAR_HARD_BUFFER}
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


def _build_candidates_unitid_query(
    f1_spells_tab: str,
    rev_match_ready_educ_tab: str,
    rev_indiv_tab: str,
    YEAR_HARD_BUFFER: int = cfg.BUILD_YEAR_HARD_BUFFER,
    DEGREE_SCORE_NULL_DEFAULT: float = cfg.BUILD_DEGREE_SCORE_NULL_DEFAULT,
    DATE_SCORE_NULL_DEFAULT: float = cfg.BUILD_DATE_SCORE_NULL_DEFAULT,
    GRADYR_SCORE_DECAY_POWER: float = cfg.BUILD_GRADYR_SCORE_DECAY_POWER,
    FIELD_TARGET_CIP4: int = cfg.BUILD_FIELD_TARGET_CIP4,
    FIELD_NON_TARGET_CAP: float = cfg.BUILD_FIELD_NON_TARGET_CAP,
    FIELD_FILTER_MIN_SCORE: float = cfg.BUILD_FIELD_FILTER_MIN_SCORE,
    FIELD_CIP2_MATCH_MULTIPLIER: float = cfg.BUILD_FIELD_CIP2_MATCH_MULTIPLIER,
) -> str:
    """Hard-filter join: F1 spells × stage-04 merge_ready on UNITID + country + degree + year."""
    country_score_expr = "COALESCE(ri.country_score, 0.0)"
    inst_score_expr = "COALESCE(re.re_inst_score, ri.institution_score, 0.0)"

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

    gradyr_score_expr = _gradyr_score_expr(
        "f1.f1_prog_end_year",
        "re.rev_educ_end_year",
        null_default=DATE_SCORE_NULL_DEFAULT,
        year_buffer=YEAR_HARD_BUFFER,
        decay_power=GRADYR_SCORE_DECAY_POWER,
    )
    field_score_expr = _field_score_expr(
        "f1.f1_cip4",
        "re.rev_cip4",
        "re.rev_cip_score",
        cip2_match_multiplier=FIELD_CIP2_MATCH_MULTIPLIER,
    )
    field_pass_expr = f"""
        CASE
            WHEN re.rev_field_mapped_ind = 1 AND COALESCE(({field_score_expr}), 0.0) < {FIELD_FILTER_MIN_SCORE}
                THEN 0
            ELSE 1
        END"""

    return f"""
    WITH candidate_base AS (
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
            f1.f1_cip4,
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
            re.user_id,
            ri.fullname,
            COALESCE(ri.country, f1.f1_country_std) AS rev_country,
            ri.subregion,
            COALESCE(re.re_country_score, ri.country_score_stage4) AS rev_country_score_raw,
            COALESCE(re.re_inst_score, ri.institution_score) AS rev_inst_score_raw,
            COALESCE(re.nanat_score, ri.nanat_score) AS nanat_score,
            COALESCE(re.nanat_subregion_score, ri.nanat_subregion_score) AS nanat_subregion_score,
            COALESCE(re.nt_subregion_score, ri.nt_subregion_score) AS nt_subregion_score,
            COALESCE(re.country_uncertain_ind, ri.country_uncertain_ind) AS country_uncertain_ind,
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
            NULL::BIGINT               AS school_match_f1_row_num,
            re.rev_cip4,
            re.rev_cip_score,
            re.rev_field_mapped_ind,
            ({country_score_expr})     AS country_score,
            ({inst_score_expr})        AS inst_score,
            ({degree_score_expr})      AS degree_score,
            ({gradyr_score_expr})      AS gradyr_score,
            ({field_score_expr})       AS field_score,
            ({field_pass_expr})        AS field_candidate_pass_ind
        FROM {f1_spells_tab} AS f1
        JOIN {rev_match_ready_educ_tab} AS re
          ON f1.resolved_unitid IS NOT NULL
         AND re.rev_unitid = f1.resolved_unitid
         AND re.country_std = f1.f1_country_std
         AND (
             f1.f1_degree_level = re.rev_degree_clean
             OR f1.f1_degree_level = 'Other'
             OR re.rev_degree_clean IN ('Missing', 'Non-Degree')
             OR (f1.f1_degree_level = 'Master' AND re.rev_degree_clean = 'MBA')
         )
         AND (
             f1.f1_prog_end_year IS NULL
             OR re.rev_educ_end_year IS NULL
             OR ABS(f1.f1_prog_end_year - re.rev_educ_end_year) <= {YEAR_HARD_BUFFER}
         )
        LEFT JOIN {rev_indiv_tab} AS ri
          ON re.user_id = ri.user_id
         AND ri.country_std = re.country_std
    )
    SELECT
        cb.*,
        COUNT(*) OVER(PARTITION BY cb.spell_id) AS n_match_raw
    FROM candidate_base AS cb
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
    EMPLOYER_MATCH_YEAR_BUFFER: int = cfg.EMPLOYER_MATCH_YEAR_BUFFER,
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

    Year filter: Revelio positions must overlap the person's F1 activity window,
    allowing a symmetric +/- EMPLOYER_MATCH_YEAR_BUFFER slack on both ends.
    IDF fallback: employers not in emp_idf get idf = 1/log(smoothing + 1).
    Token IDF fallback: tokens not in token_idf_tab get token_idf = 1/log(2).
    """
    default_idf = f"1.0 / LOG({EMP_IDF_SMOOTHING} + 1.0)"
    default_token_idf = "1.0 / LOG(2.0)"   # token appearing in 1 company
    year_overlap_expr_entity = _sql_emp_year_overlap_expr(
        "rp.pos_start_year",
        "rp.pos_end_year",
        "f.min_f1_year",
        "f.max_f1_year",
        EMPLOYER_MATCH_YEAR_BUFFER,
    )
    year_overlap_expr_subset = _sql_emp_year_overlap_expr(
        "rpt.pos_start_year",
        "rpt.pos_end_year",
        "fe.min_f1_year",
        "fe.max_f1_year",
        EMPLOYER_MATCH_YEAR_BUFFER,
    )
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
            AND {year_overlap_expr_entity}
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
            AND {year_overlap_expr_entity}
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
            AND {year_overlap_expr_subset}
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
    rev_pos_user_stats_tab: str | None = None,
    W_EMP_MAX:   float = cfg.BUILD_W_EMP_MAX,
    EMP_N_SCALE: float = cfg.BUILD_EMP_N_SCALE,
    MULTIPLICATIVE_SCORE: bool = cfg.BUILD_MULTIPLICATIVE_SCORE,
    W_GRADYR:    float = cfg.BUILD_W_GRADYR,
    W_COUNTRY:   float = cfg.BUILD_W_COUNTRY,
    W_INST:      float = cfg.BUILD_W_INST,
    W_FIELD:     float = cfg.BUILD_W_FIELD,
) -> str:
    """Join candidates with employer sequence scores and compute total_score.

    Employer weight uses exponential saturation so that it increases with the
    number of F1 employers (more evidence = employer signal dominates more):

        w_emp_eff = W_EMP_MAX * (1 - exp(-n_f1_employers / EMP_N_SCALE))

    Total score formula:
      Additive mode:
        With employer data:
          w_emp_eff × employer_score
          + (1 - w_emp_eff) × weighted_avg(gradyr, country, inst, field)
        Without employer data:
          weighted_avg(gradyr, country, inst, field)

      Multiplicative mode:
        With employer data:
          employer_score ^ w_emp_eff
          × weighted_avg(gradyr, country, inst, field) ^ (1 - w_emp_eff)
        Without employer data:
          weighted_avg(gradyr, country, inst, field)

    emp_score_available_ind = 1 if the person had any employer data.
    w_emp_eff is included as an output column for diagnostics.
    """
    w_no_emp = W_GRADYR + W_COUNTRY + W_INST + W_FIELD
    if w_no_emp <= 0:
        raise ValueError("Non-employer weights must sum to a positive value.")
    # Inline SQL expression for variable employer weight (avoids repeating formula)
    _w_eff = (
        f"({W_EMP_MAX} * (1.0 - EXP(-CAST(COALESCE(es.n_f1_employers, 0) AS DOUBLE) / {EMP_N_SCALE})))"
    )
    nonemp_additive_expr = (
        f"(   {W_GRADYR} * COALESCE(mc.gradyr_score, 0.0)"
        f"  + {W_COUNTRY} * COALESCE(mc.country_score, 0.0)"
        f"  + {W_INST}    * COALESCE(mc.inst_score, 0.0)"
        f"  + {W_FIELD}   * COALESCE(mc.field_score, 0.0)"
        f") / {w_no_emp}"
    )

    def _pow_term(score_sql: str, exponent_sql: str) -> str:
        return f"POWER(GREATEST(COALESCE({score_sql}, 0.0), 1e-9), {exponent_sql})"

    multiplicative_emp_expr = " * ".join([
        _pow_term("es.employer_score", _w_eff),
        _pow_term(f"({nonemp_additive_expr})", f"(1.0 - {_w_eff})"),
    ])
    total_score_expr = f"""
        CASE
            WHEN es.employer_score IS NOT NULL THEN
                {_w_eff} * es.employer_score
                + (1.0 - {_w_eff}) * {nonemp_additive_expr}
            ELSE
                {nonemp_additive_expr}
        END
    """
    if MULTIPLICATIVE_SCORE:
        total_score_expr = f"""
        CASE
            WHEN es.employer_score IS NOT NULL THEN
                {multiplicative_emp_expr}
            ELSE
                {nonemp_additive_expr}
        END
        """
    rev_pos_join = ""
    rev_pos_select = "0::BIGINT AS n_rev_positions,"
    rev_pos_count_expr = "0"
    if rev_pos_user_stats_tab:
        rev_pos_join = (
            f"LEFT JOIN {rev_pos_user_stats_tab} AS rp "
            "ON mc.user_id = rp.user_id"
        )
        rev_pos_select = "COALESCE(rp.n_rev_positions, 0) AS n_rev_positions,"
        rev_pos_count_expr = "COALESCE(rp.n_rev_positions, 0)"
    return f"""
    SELECT
        mc.*,
        es.employer_score,
        {rev_pos_select}
        COALESCE(es.n_f1_employers,    0) AS n_f1_employers,
        COALESCE(es.n_emp_matched,     0) AS n_emp_matched,
        COALESCE(es.has_any_emp_match, 0) AS has_any_emp_match,
        CASE WHEN es.employer_score IS NOT NULL THEN 1 ELSE 0 END AS emp_score_available_ind,
        -- Effective employer weight for this row (diagnostic)
        CASE WHEN es.employer_score IS NOT NULL THEN {_w_eff} ELSE 0.0 END AS w_emp_eff,
        CASE
            WHEN COALESCE(es.n_f1_employers, 0) > 0
             AND {rev_pos_count_expr} > 0
             AND COALESCE(es.has_any_emp_match, 0) = 0
                THEN 0
            ELSE 1
        END AS employment_history_pass_ind,
        -- Total score: additive blend or employer-vs-nonemployer geometric blend, depending on config.
        {total_score_expr} AS total_score
    FROM {candidates_tab} AS mc
    LEFT JOIN {emp_scores_tab} AS es
        ON mc.person_id = es.person_id AND mc.user_id = es.user_id
    {rev_pos_join}
    """


def _build_employment_history_filtered_query(
    source_tab: str,
    enabled: bool = cfg.EMPLOYMENT_HISTORY_FILTER_ENABLED,
) -> str:
    if not enabled:
        return f"SELECT * FROM {source_tab}"
    return f"SELECT * FROM {source_tab} WHERE employment_history_pass_ind = 1"


def _build_candidate_filtered_query(
    source_tab: str,
    *,
    employment_filter_enabled: bool = cfg.EMPLOYMENT_HISTORY_FILTER_ENABLED,
    field_filter_enabled: bool = cfg.FIELD_CANDIDATE_FILTER_ENABLED,
    relative_score_filter_enabled: bool = cfg.RELATIVE_SCORE_FILTER_ENABLED,
    employer_score_relative_buffer: float = cfg.EMPLOYER_SCORE_RELATIVE_BUFFER,
    employer_score_relative_apply_min: float = cfg.EMPLOYER_SCORE_RELATIVE_APPLY_MIN,
    field_score_relative_buffer: float = cfg.FIELD_SCORE_RELATIVE_BUFFER,
    field_score_relative_apply_min: float = cfg.FIELD_SCORE_RELATIVE_APPLY_MIN,
    country_score_relative_buffer: float = cfg.COUNTRY_SCORE_RELATIVE_BUFFER,
    country_score_relative_apply_min: float = cfg.COUNTRY_SCORE_RELATIVE_APPLY_MIN,
) -> str:
    """Apply hard candidate filters, then sequential relative score cutoffs by spell.

    Relative cutoffs are applied on the surviving set in this order:
      1. employer_score
      2. field_score
      3. country_score

    This lets a strong target-field hit survive even when its country score is
    lower than another candidate that was already removed by the field filter.
    """
    if not (
        employment_filter_enabled
        or field_filter_enabled
        or relative_score_filter_enabled
    ):
        return f"SELECT * FROM {source_tab}"

    return f"""
    WITH base AS (
        SELECT
            *,
            CASE
                WHEN {1 if not employment_filter_enabled else 0} = 1 THEN 1
                WHEN employment_history_pass_ind = 1 THEN 1
                ELSE 0
            END AS employment_filter_pass_ind,
            CASE
                WHEN {1 if not field_filter_enabled else 0} = 1 THEN 1
                WHEN field_candidate_pass_ind = 1 THEN 1
                ELSE 0
            END AS field_absolute_pass_ind
        FROM {source_tab}
    ),
    hard_kept AS (
        SELECT *
        FROM base
        WHERE employment_filter_pass_ind = 1
          AND field_absolute_pass_ind = 1
    ),
    employer_scored AS (
        SELECT
            *,
            MAX(CASE WHEN emp_score_available_ind = 1 THEN employer_score END)
                OVER (PARTITION BY spell_id) AS spell_max_employer_score
        FROM hard_kept
    ),
    employer_kept AS (
        SELECT
            *,
            CASE
                WHEN {1 if not relative_score_filter_enabled else 0} = 1 THEN 1
                WHEN COALESCE(spell_max_employer_score, -1.0) < {employer_score_relative_apply_min} THEN 1
                WHEN COALESCE(employer_score, -1.0) >= spell_max_employer_score - {employer_score_relative_buffer} THEN 1
                ELSE 0
            END AS employer_relative_pass_ind
        FROM employer_scored
    ),
    field_scored AS (
        SELECT
            *,
            MAX(field_score) OVER (PARTITION BY spell_id) AS spell_max_field_score
        FROM employer_kept
        WHERE employer_relative_pass_ind = 1
    ),
    field_kept AS (
        SELECT
            *,
            CASE
                WHEN {1 if not relative_score_filter_enabled else 0} = 1 THEN 1
                WHEN COALESCE(spell_max_field_score, -1.0) < {field_score_relative_apply_min} THEN 1
                WHEN field_score >= spell_max_field_score - {field_score_relative_buffer} THEN 1
                ELSE 0
            END AS field_relative_pass_ind
        FROM field_scored
    ),
    country_scored AS (
        SELECT
            *,
            MAX(country_score) OVER (PARTITION BY spell_id) AS spell_max_country_score
        FROM field_kept
        WHERE field_relative_pass_ind = 1
    )
    SELECT
        *,
        CASE
            WHEN {1 if not relative_score_filter_enabled else 0} = 1 THEN 1
            WHEN COALESCE(spell_max_country_score, -1.0) < {country_score_relative_apply_min} THEN 1
            WHEN country_score >= spell_max_country_score - {country_score_relative_buffer} THEN 1
            ELSE 0
        END AS country_relative_pass_ind
    FROM country_scored
    WHERE field_relative_pass_ind = 1
      AND (
          CASE
              WHEN {1 if not relative_score_filter_enabled else 0} = 1 THEN 1
              WHEN COALESCE(spell_max_country_score, -1.0) < {country_score_relative_apply_min} THEN 1
              WHEN country_score >= spell_max_country_score - {country_score_relative_buffer} THEN 1
              ELSE 0
          END
      ) = 1
    """


# ---------------------------------------------------------------------------
# Stage 4: Filtering and weighting
# ---------------------------------------------------------------------------

def _build_stage_match_filt_sql(source_tab: str) -> str:
    """Build stage_match_filt SQL: keep best Revelio education record per (spell_id, user_id).

    School, country, degree, and grad-year are now hard join filters applied
    upstream in _build_candidates_query, and employment-history pruning is
    applied immediately before this stage.
    The dedup picks the best-scoring education record when a user has multiple
    Revelio education rows that survived the hard filters for the same spell.
    """
    return f"""
    WITH base AS (
        SELECT *,
            ROW_NUMBER() OVER(
                PARTITION BY spell_id, user_id
                ORDER BY total_score DESC, rev_degree_clean ASC, rev_educ_end_year ASC
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


def _solve_individual_assignment(
    con,
    person_agg_tab: str,
    enforce_one_to_one: bool = cfg.BUILD_ENFORCE_INDIVIDUAL_ONE_TO_ONE,
) -> pd.DataFrame:
    """Resolve global person_id <-> user_id matches from person-level scores.

    When enforce_one_to_one is True, solve a max-weight bipartite assignment
    over the sparse (person_id, user_id) candidate graph using person_score_sum
    as the edge weight. The solve is done component-by-component so disconnected
    subgraphs do not create a huge dense matrix.

    When enforce_one_to_one is False, keep the best user for each person_id
    independently (person_match_rank = 1 behavior).
    """
    pairs_df = con.sql(f"""
        SELECT
            person_id,
            user_id,
            person_score_sum,
            n_evidence_units,
            n_spell_matches,
            total_emp_matches,
            has_employer_match_ind,
            person_weight_norm,
            person_match_rank
        FROM {person_agg_tab}
    """).df()

    if pairs_df.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "user_id",
                "person_score_sum",
                "n_evidence_units",
                "n_spell_matches",
                "total_emp_matches",
                "has_employer_match_ind",
                "person_weight_norm",
                "person_match_rank",
            ]
        )

    if not enforce_one_to_one:
        return (
            pairs_df.sort_values(
                ["person_id", "person_match_rank", "person_score_sum", "user_id"],
                ascending=[True, True, False, True],
                kind="mergesort",
            )
            .drop_duplicates("person_id", keep="first")
            .reset_index(drop=True)
        )

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    dummy_cost = 1e-12

    person_to_users: dict[int, set[int]] = {}
    user_to_persons: dict[int, set[int]] = {}
    pair_lookup: dict[tuple[int, int], dict] = {}

    for row in pairs_df.to_dict("records"):
        person_id = row["person_id"]
        user_id = row["user_id"]
        person_to_users.setdefault(person_id, set()).add(user_id)
        user_to_persons.setdefault(user_id, set()).add(person_id)
        pair_lookup[(person_id, user_id)] = row

    seen_people: set[int] = set()
    seen_users: set[int] = set()
    results: list[dict] = []

    for start_person in person_to_users:
        if start_person in seen_people:
            continue

        comp_people: set[int] = set()
        comp_users: set[int] = set()
        queue: list[tuple[str, int]] = [("person", start_person)]

        while queue:
            kind, node = queue.pop()
            if kind == "person":
                if node in seen_people:
                    continue
                seen_people.add(node)
                comp_people.add(node)
                for uid in person_to_users.get(node, ()):
                    if uid not in seen_users:
                        queue.append(("user", uid))
            else:
                if node in seen_users:
                    continue
                seen_users.add(node)
                comp_users.add(node)
                for pid in user_to_persons.get(node, ()):
                    if pid not in seen_people:
                        queue.append(("person", pid))

        if not comp_people or not comp_users:
            continue

        people = sorted(comp_people)
        users = sorted(comp_users)
        p_idx = {pid: i for i, pid in enumerate(people)}
        u_idx = {uid: j for j, uid in enumerate(users)}
        row_idx: list[int] = []
        col_idx: list[int] = []
        data: list[float] = []

        for pid in people:
            person_index = p_idx[pid]
            for uid in person_to_users.get(pid, ()):
                if uid not in u_idx:
                    continue
                row = pair_lookup[(pid, uid)]
                raw_score = row.get("person_score_sum")
                score = float(raw_score) if pd.notna(raw_score) else 0.0
                if not np.isfinite(score) or score <= 0.0:
                    continue
                row_idx.append(person_index)
                col_idx.append(u_idx[uid])
                data.append(-score)

            # Give every person one private dummy column so leaving them unmatched
            # stays feasible without expanding to a dense people x users matrix.
            row_idx.append(person_index)
            col_idx.append(len(users) + person_index)
            data.append(dummy_cost)

        cost = coo_matrix(
            (np.asarray(data, dtype=np.float64), (np.asarray(row_idx), np.asarray(col_idx))),
            shape=(len(people), len(users) + len(people)),
        ).tocsr()

        row_ind, col_ind = min_weight_full_bipartite_matching(cost)
        for r, c in zip(row_ind, col_ind):
            if c >= len(users):
                continue
            pid = people[r]
            uid = users[c]
            results.append(pair_lookup[(pid, uid)])

    assigned = pd.DataFrame(results)
    if assigned.empty:
        return pd.DataFrame(columns=pairs_df.columns.tolist())

    assigned = assigned.sort_values(
        ["person_id", "person_score_sum", "n_evidence_units", "user_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    assigned["person_match_rank"] = 1
    assigned["person_weight_norm"] = 1.0
    return assigned[pairs_df.columns.tolist()]


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
              f"| gradyr={top['f1_prog_end_year']} | country={top['f1_country_std']} "
              f"| CIP4={top['f1_cip4']}")
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
                f"gradyr={row.get('rev_educ_end_year','')} "
                f"cip4={row.get('rev_cip4','')} | "
                f"emp={emp_sc_str}({n_matched}/{n_emps}matched) "
                f"cs={row['country_score']:.3f} "
                f"inst={row['inst_score']:.3f} "
                f"fld={row['field_score']:.3f} "
                f"deg={row['degree_score']:.3f} "
                f"grad={row['gradyr_score']:.3f} "
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
    if not cfg.F1_FOIA_PARQUET or not os.path.exists(cfg.F1_FOIA_PARQUET):
        raise FileNotFoundError(f"F1 FOIA input not found: {cfg.F1_FOIA_PARQUET}")
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

    f1_inst_unitid_path = cfg.F1_INST_UNITID_CROSSWALK_PARQUET
    has_f1_inst_unitid = bool(f1_inst_unitid_path) and os.path.exists(f1_inst_unitid_path)
    has_f1_inst_unitid_schema = False
    if has_f1_inst_unitid:
        con.sql(
            f"CREATE OR REPLACE VIEW f1_inst_unitid AS "
            f"SELECT * FROM read_parquet('{f1_inst_unitid_path}')"
        )
        f1_inst_cols = {r[0] for r in con.sql("DESCRIBE f1_inst_unitid").fetchall()}
        has_f1_inst_unitid_schema = _F1_INST_UNITID_REQUIRED_COLS.issubset(f1_inst_cols)
        if has_f1_inst_unitid_schema:
            n_inst_unitid = int(con.sql("SELECT COUNT(*) FROM f1_inst_unitid").fetchone()[0])
            n_inst_rows = int(
                con.sql("SELECT COUNT(DISTINCT f1_row_num) FROM f1_inst_unitid").fetchone()[0]
            )
            print(f"  f1_inst_unitid:   {n_inst_unitid:>12,} rows | {n_inst_rows:>8,} F1 rows with UNITID")
        else:
            missing_inst_cols = ", ".join(sorted(_F1_INST_UNITID_REQUIRED_COLS - f1_inst_cols))
            print("  f1_inst_unitid:   FOUND but missing required columns")
            print(f"    → Missing: {missing_inst_cols}")
    else:
        print("  f1_inst_unitid:   NOT FOUND")

    # Revelio education
    rev_educ_path = cfg.choose_path(cfg.REV_EDUC_LONG_PARQUET, cfg.REV_EDUC_LONG_PARQUET_LEGACY)
    if not rev_educ_path or not os.path.exists(rev_educ_path):
        raise FileNotFoundError(
            "Revelio education input not found. "
            f"Checked stage-04 cleaned artifact and legacy fallback: {rev_educ_path}"
        )
    con.sql(f"CREATE OR REPLACE VIEW rev_educ AS SELECT * FROM read_parquet('{rev_educ_path}')")
    n_rev_educ = int(con.sql("SELECT COUNT(*) FROM rev_educ").fetchone()[0])
    n_rev_educ_users = int(con.sql("SELECT COUNT(DISTINCT user_id) FROM rev_educ").fetchone()[0])
    print(f"  rev_educ:         {n_rev_educ:>12,} rows | {n_rev_educ_users:>8,} distinct user_ids")
    rev_educ_cols = _describe_relation_columns(con, "rev_educ")
    has_rev_educ_cip = "cip" in rev_educ_cols
    has_rev_educ_cip_score = "cip_score" in rev_educ_cols
    has_rev_educ_field_mapped_ind = "field_mapped_ind" in rev_educ_cols
    print(
        "  rev_educ cols detected: "
        f"cip={has_rev_educ_cip}, "
        f"cip_score={has_rev_educ_cip_score}, "
        f"field_mapped_ind={has_rev_educ_field_mapped_ind}"
    )

    # Revelio individual / country-candidate view
    has_stage4_rev_indiv = (
        bool(cfg.REV_USERS_CORE_PARQUET)
        and bool(cfg.REV_MATCH_READY_PARQUET)
        and os.path.exists(str(cfg.REV_USERS_CORE_PARQUET))
        and os.path.exists(str(cfg.REV_MATCH_READY_PARQUET))
    )
    rev_indiv_source_mode = "legacy_fallback"
    if has_stage4_rev_indiv:
        users_core_path = str(cfg.REV_USERS_CORE_PARQUET)
        match_ready_path = str(cfg.REV_MATCH_READY_PARQUET)
        con.sql(f"CREATE OR REPLACE VIEW rev_users_core AS SELECT * FROM read_parquet('{users_core_path}')")
        con.sql(f"CREATE OR REPLACE VIEW rev_match_ready AS SELECT * FROM read_parquet('{match_ready_path}')")
        uc_cols = _describe_relation_columns(con, "rev_users_core")
        mr_cols = _describe_relation_columns(con, "rev_match_ready")
        if "user_id" not in mr_cols or "country_candidate" not in mr_cols:
            raise ValueError(
                "Stage-04 rev_match_ready artifact is missing required columns "
                "('user_id', 'country_candidate')."
            )
        con.execute(
            f"""
            CREATE OR REPLACE VIEW rev_indiv AS
            WITH mr_country_ranked AS (
                SELECT
                    CAST(user_id AS BIGINT) AS user_id,
                    CAST(country_candidate AS VARCHAR) AS country,
                    {_optional_select_expr(mr_cols, ['subregion_candidate', 'subregion'], 'subregion')},
                    {_optional_select_expr(mr_cols, ['country_score'], 'country_score', 'DOUBLE')},
                    {_optional_select_expr(mr_cols, ['nanat_score', 'country_score'], 'nanat_score', 'DOUBLE')},
                    {_optional_select_expr(mr_cols, ['institution_score'], 'institution_score', 'DOUBLE')},
                    {_optional_select_expr(mr_cols, ['nanat_subregion_score'], 'nanat_subregion_score', 'DOUBLE')},
                    {_optional_select_expr(mr_cols, ['nt_subregion_score'], 'nt_subregion_score', 'DOUBLE')},
                    {_optional_select_expr(mr_cols, ['country_uncertain_ind'], 'country_uncertain_ind', 'INTEGER')},
                    ROW_NUMBER() OVER(
                        PARTITION BY CAST(user_id AS BIGINT), CAST(country_candidate AS VARCHAR)
                        ORDER BY
                            COALESCE(CAST(country_score AS DOUBLE), 0.0) DESC,
                            COALESCE(CAST(institution_score AS DOUBLE), 0.0) DESC,
                            COALESCE(CAST(nanat_score AS DOUBLE), 0.0) DESC
                    ) AS country_rn
                FROM rev_match_ready
                WHERE country_candidate IS NOT NULL
            ),
            mr_country AS (
                SELECT * EXCLUDE(country_rn)
                FROM mr_country_ranked
                WHERE country_rn = 1
            )
            SELECT
                mr.user_id,
                {_optional_select_expr(uc_cols, ['fullname', 'full_name', 'fullname_clean'], 'fullname')},
                mr.country,
                mr.subregion,
                mr.country_score,
                mr.nanat_score,
                mr.institution_score,
                mr.nanat_subregion_score,
                mr.nt_subregion_score,
                mr.country_uncertain_ind,
                {_optional_select_expr(uc_cols, ['est_yob'], 'est_yob', 'BIGINT')},
                {_optional_select_expr(uc_cols, ['stem_ind_any', 'stem_ind'], 'stem_ind', 'INTEGER')},
                {_optional_select_expr(uc_cols, ['f_prob', 'f_prob_nt'], 'f_prob', 'DOUBLE')},
                {_optional_select_expr(uc_cols, ['fields_json', 'fields'], 'fields')},
                {_optional_select_expr(uc_cols, ['highest_ed_level'], 'highest_ed_level')}
            FROM mr_country AS mr
            LEFT JOIN rev_users_core AS uc
              ON mr.user_id = CAST(uc.user_id AS BIGINT)
            """
        )
        rev_indiv_source_mode = "stage04_outputs"
    else:
        rev_indiv_path = cfg.choose_path(None, cfg.REV_INDIV_PARQUET_LEGACY)
        if not rev_indiv_path or not os.path.exists(rev_indiv_path):
            raise FileNotFoundError(
                "Revelio individual/country input not found. "
                "Expected stage-04 `rev_users_core` + `rev_match_ready` or a legacy indiv parquet."
            )
        con.sql(f"CREATE OR REPLACE VIEW rev_indiv AS SELECT * FROM read_parquet('{rev_indiv_path}')")
    n_rev_indiv = int(con.sql("SELECT COUNT(*) FROM rev_indiv").fetchone()[0])
    print(f"  rev_indiv:        {n_rev_indiv:>12,} rows ({rev_indiv_source_mode})")

    # Country crosswalk table
    # Load crosswalk dict into DuckDB so country normalization uses a vectorized SQL JOIN
    # rather than a per-row Python UDF call during the cross-join.
    _cw_df = pd.DataFrame(list(_COUNTRY_CW_UPPER.items()), columns=['key_upper', 'std_country'])
    con.register('_country_cw_src', _cw_df)
    con.execute("CREATE OR REPLACE TABLE _country_cw AS SELECT * FROM _country_cw_src")
    rev_indiv_cols = _describe_relation_columns(con, "rev_indiv")
    print(f"  country_cw:       {len(_cw_df):>12,} entries → _country_cw")

    # Detect optional columns
    has_country_uncertain = "country_uncertain_ind" in rev_indiv_cols
    has_nanat_score = "nanat_score" in rev_indiv_cols
    has_fields = "fields" in rev_indiv_cols
    print(f"  rev_indiv cols detected: country_uncertain={has_country_uncertain}, "
          f"nanat_score={has_nanat_score}, fields={has_fields}")

    # Revelio positions (for employer sequence scoring)
    rev_pos_path = cfg.choose_path(cfg.REV_POS_PARQUET, cfg.REV_POS_PARQUET_LEGACY)
    has_rev_pos = bool(rev_pos_path) and os.path.exists(rev_pos_path)
    if has_rev_pos:
        con.sql(f"CREATE OR REPLACE VIEW rev_pos AS SELECT * FROM read_parquet('{rev_pos_path}')")
        n_rev_pos = int(con.sql("SELECT COUNT(*) FROM rev_pos").fetchone()[0])
        print(f"  rev_pos:          {n_rev_pos:>12,} rows")
        con.execute(
            "CREATE OR REPLACE VIEW rev_pos_country AS "
            + _build_rev_pos_country_query("rev_pos")
        )
    else:
        print(f"  rev_pos:          NOT FOUND (employer sequence scoring will be skipped)")
        con.execute("""
            CREATE OR REPLACE VIEW rev_pos_country AS
            SELECT
                NULL::BIGINT AS user_id,
                NULL::VARCHAR AS country_std,
                0.0::DOUBLE AS position_score
            WHERE 1=0
        """)

    con.execute(
        "CREATE OR REPLACE VIEW rev_indiv_norm AS "
        + _build_rev_indiv_norm_query(
            "rev_indiv",
            rev_indiv_cols,
            "rev_pos_country",
        )
    )
    country_scoring_mode = "multiplicative" if cfg.BUILD_MULTIPLICATIVE_SCORE else "additive"
    print(
        "  rev_indiv_norm:   H1B-style country scoring "
        f"(alpha={cfg.BUILD_SUBREGION_BOOST_ALPHA:.2f}, "
        f"competition_weight={cfg.BUILD_COUNTRY_COMPETITION_WEIGHT:.2f}) | "
        f"stage4 total-score mode={country_scoring_mode}"
    )

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
        "has_stage4_rev_indiv": has_stage4_rev_indiv,
        "has_school_resolution": has_school_resolution,
        "has_school_resolution_schema": has_school_resolution_schema,
        "has_f1_inst_unitid": has_f1_inst_unitid,
        "has_f1_inst_unitid_schema": has_f1_inst_unitid_schema,
        "has_country_uncertain": has_country_uncertain,
        "has_nanat_score": has_nanat_score,
        "has_fields": has_fields,
        "has_rev_educ_cip": has_rev_educ_cip,
        "has_rev_educ_cip_score": has_rev_educ_cip_score,
        "has_rev_educ_field_mapped_ind": has_rev_educ_field_mapped_ind,
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
    person_shard_count: int | None = None,
    person_shard_id: int | None = None,
    shard_output_paths: dict[str, str] | None = None,
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
      - Runs on a random sample of N original individual_keys
      - Materializes all intermediate tables for interactive inspection
      - Prints spell-level and person-level spotchecks
      - Does NOT write parquet output files

    Args:
        testing:  Override testing mode (None = use config value)
        overwrite: Override overwrite setting (None = use config value)
        con:      DuckDB connection to use
        person_shard_count: Optional exact person-shard count for production chunking
        person_shard_id: Optional exact person-shard id for production chunking
        shard_output_paths: Optional shard artifact output paths; when provided,
            the run writes shard baseline + person_agg artifacts and skips the
            final global person assignment and final output parquet writes
    """
    if testing is None:
        testing = cfg.TESTING_ENABLED
    if overwrite is None:
        overwrite = cfg.BUILD_OVERWRITE
    shard_mode = (
        person_shard_count is not None
        or person_shard_id is not None
        or shard_output_paths is not None
    )
    if shard_mode:
        if person_shard_count is None or person_shard_id is None:
            raise ValueError("Stage-05 shard mode requires both `person_shard_count` and `person_shard_id`.")
        if shard_output_paths is None:
            raise ValueError("Stage-05 shard mode requires `shard_output_paths`.")
        if int(person_shard_count) < 2:
            raise ValueError("Stage-05 shard mode requires `person_shard_count >= 2`.")
        if int(person_shard_id) < 0 or int(person_shard_id) >= int(person_shard_count):
            raise ValueError("Stage-05 shard mode requires `0 <= person_shard_id < person_shard_count`.")
        person_shard_label = f"shard{int(person_shard_id):04d}of{int(person_shard_count):04d}"
    else:
        person_shard_label = "off"

    t_total = time.perf_counter()
    print("=" * 70)
    print("f1_indiv_merge: building F1 × Revelio merge")
    print(f"  run_tag:  {cfg.RUN_TAG}")
    print(f"  testing:  {testing}")
    print(f"  overwrite:{overwrite}")
    if shard_mode:
        print(f"  person_shard:{person_shard_label}")
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
    use_unitid_candidate_join = (
        col_flags["has_stage4_rev_indiv"]
        and col_flags["has_f1_inst_unitid"]
        and col_flags["has_f1_inst_unitid_schema"]
    )
    if use_unitid_candidate_join:
        print("  candidate_join_mode: unitid_stage04_merge_ready")
    else:
        print("  candidate_join_mode: legacy_school_crosswalk")

    f1_foia_base_src = "f1_foia"
    if cfg.RESTRICT_TO_RELABEL_PROGRAMS:
        print("\n[Relabel sample restriction]")
        if not (col_flags["has_f1_inst_unitid"] and col_flags["has_f1_inst_unitid_schema"]):
            raise FileNotFoundError(
                "restrict_to_relabel_programs=true, but the F1 institution-to-UNITID crosswalk "
                "is unavailable or missing required columns."
            )
        treated_events_df, matched_pairs_df = _load_relabel_program_artifacts(con=con)
        con.execute("DROP VIEW IF EXISTS _relabel_treated_events_py")
        con.execute("DROP VIEW IF EXISTS _relabel_matched_pairs_py")
        con.register("_relabel_treated_events_py", treated_events_df)
        con.register("_relabel_matched_pairs_py", matched_pairs_df)
        con.execute(
            "CREATE OR REPLACE TEMP VIEW _relabel_treated_events AS "
            "SELECT * FROM _relabel_treated_events_py"
        )
        con.execute(
            "CREATE OR REPLACE TEMP VIEW _relabel_matched_pairs AS "
            "SELECT * FROM _relabel_matched_pairs_py"
        )
        relabel_sample_q = _build_f1_relabel_sample_query(
            "f1_foia",
            "f1_inst_unitid",
            "_relabel_treated_events",
            "_relabel_matched_pairs",
            cfg.RELABEL_GRADYEAR_WINDOW,
        )
        relabel_sample_src = _materialize_relabel_sample(
            con,
            relabel_sample_q,
            cache_path=cfg.RELABEL_SAMPLE_CACHE_PARQUET,
            force_rebuild=cfg.RELABEL_SAMPLE_FORCE_REBUILD,
            table_name="_f1_foia_relabel_sample",
        )
        relabel_sample_stats = con.sql(
            f"""
            SELECT
                COUNT(*) AS n_rows,
                COUNT(DISTINCT person_id) AS n_persons,
                COUNT(DISTINCT sample_unitid) AS n_unitids,
                COUNT(DISTINCT sample_relabel_year) AS n_relabel_years
            FROM {relabel_sample_src}
            """
        ).df().iloc[0]
        relabel_role_stats = con.sql(
            f"""
            SELECT sample_role, COUNT(*) AS n_rows
            FROM {relabel_sample_src}
            GROUP BY sample_role
            ORDER BY sample_role
            """
        ).df()
        print(
            f"  {int(relabel_sample_stats['n_rows']):,} FOIA rows | "
            f"{int(relabel_sample_stats['n_persons']):,} persons | "
            f"{int(relabel_sample_stats['n_unitids']):,} unitids | "
            f"{int(relabel_sample_stats['n_relabel_years']):,} relabel cohorts"
        )
        print("  Sample roles:")
        print(relabel_role_stats.to_string(index=False))
        if int(relabel_sample_stats["n_rows"]) <= 0:
            if testing:
                print(
                    "  Relabel-program restriction produced zero rows in testing mode; "
                    "falling back to the full FOIA source for interactive inspection."
                )
                f1_foia_base_src = "f1_foia"
            else:
                raise ValueError("Relabel-program FOIA sample restriction produced zero rows.")
        else:
            f1_foia_base_src = "_f1_foia_relabel_sample"
            print("  Using relabel-restricted FOIA sample")

    # ------------------------------------------------------------------
    # 2. Build row-level school resolution + education spells
    # ------------------------------------------------------------------
    print("\n[Stage 1: F1 school rows + education spells]")
    t0 = time.perf_counter()

    # In testing mode: restrict to rows keyed by original individual_key
    # where the FOIA filing year matches the program end year.
    if testing:
        seed = cfg.TESTING_RANDOM_SEED or 42
        n_sample = cfg.TESTING_SAMPLE_N_PERSONS
        individual_keys_pin = list(getattr(cfg, "TESTING_INDIVIDUAL_KEYS", []) or [])
        person_ids_pin = list(getattr(cfg, "TESTING_PERSON_IDS", []) or [])
        school_pin = cfg.TESTING_SCHOOL
        country_pin = cfg.TESTING_COUNTRY
        if school_pin or country_pin:
            print(
                "  [TESTING] Active filters: "
                f"school={school_pin!r}, country={country_pin!r}"
            )
        if not individual_keys_pin and not person_ids_pin:
            eligible_keys_q = _build_testing_eligible_keys_query(
                f1_foia_base_src,
                school_pin=school_pin,
                country_pin=country_pin,
            )
            n_eligible_keys = int(
                con.sql(f"SELECT COUNT(*) FROM ({eligible_keys_q}) AS eligible_keys").fetchone()[0]
            )
            print(
                "  [TESTING] Eligible distinct individual_keys after filters: "
                f"{n_eligible_keys:,}"
            )
            if n_eligible_keys <= 0:
                raise ValueError(
                    "Testing sample produced zero eligible individual_keys before sampling. "
                    f"source={f1_foia_base_src}, school={school_pin!r}, country={country_pin!r}"
                )
        f1_foia_src = _build_testing_f1_foia_source_query(
            f1_foia_base_src,
            n_sample=n_sample,
            seed=seed,
            individual_keys_pin=individual_keys_pin,
            person_ids_pin=person_ids_pin,
            school_pin=school_pin,
            country_pin=country_pin,
        )
        if individual_keys_pin:
            print(f"  [TESTING] Using explicit individual_key list: {individual_keys_pin}")
        elif person_ids_pin:
            print(f"  [TESTING] Using explicit person_id list: {person_ids_pin}")
        if not individual_keys_pin and not person_ids_pin:
            print(f"  [TESTING] Using sample of {n_sample} individual_keys (seed={seed})")
    else:
        f1_foia_src = f1_foia_base_src

    print("  Materializing match-unit FOIA rows (individual_key x year with year == program_end_year)...")
    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        materialize_table(
            f"{pfx}_f1_match_unit_rows",
            _build_f1_match_unit_rows_query(
                f1_foia_src,
                person_shard_count=(int(person_shard_count) if shard_mode else None),
                person_shard_id=(int(person_shard_id) if shard_mode else None),
            ),
            con=con,
        )
        f1_match_unit_tab = f"{pfx}_f1_match_unit_rows"
    else:
        materialize_table(
            "_f1_match_unit_rows",
            _build_f1_match_unit_rows_query(
                f1_foia_src,
                person_shard_count=(int(person_shard_count) if shard_mode else None),
                person_shard_id=(int(person_shard_id) if shard_mode else None),
            ),
            con=con,
        )
        f1_match_unit_tab = "_f1_match_unit_rows"

    match_unit_stats = con.sql(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT individual_key) AS n_individual_keys,
            COUNT(DISTINCT person_id) AS n_match_person_ids,
            COUNT(DISTINCT linked_person_id) AS n_linked_person_ids
        FROM {f1_match_unit_tab}
        """
    ).df().iloc[0]
    print(
        f"  {int(match_unit_stats['n_rows']):,} targeted FOIA rows | "
        f"{int(match_unit_stats['n_individual_keys']):,} distinct individual_keys | "
        f"{int(match_unit_stats['n_match_person_ids']):,} individual-key x year targets | "
        f"{int(match_unit_stats['n_linked_person_ids']):,} upstream linked person_ids"
    )

    if use_unitid_candidate_join:
        school_rows_q = _build_f1_school_rows_unitid_query(
            f1_match_unit_tab,
            "f1_inst_unitid",
            cw_tab="f1_rev_school_cw_filt",
        )
    else:
        school_rows_q = _build_f1_school_rows_query(
            f1_match_unit_tab,
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
    print(f"  {n_spells:,} education spells from {n_persons_spells:,} individual-key x year targets "
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
        opt_emp_q = _build_f1_opt_employers_all_query(f1_match_unit_tab, "f1_opt_emp_lookup")
        use_testing_prefix = bool(testing) and bool(cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES)
        pfx = cfg.TESTING_TABLE_PREFIX if use_testing_prefix else ""
        tname = f"{pfx}_f1_opt_employers" if use_testing_prefix else "_f1_opt_employers"
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
        print(f"  {n_opt_emp:,} (target, employer) rows from {n_opt_persons:,} individual-key x year targets "
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
    print("\n[Stage 2: Revelio education summary]")
    t0 = time.perf_counter()
    if use_unitid_candidate_join:
        rev_educ_school_q = _build_rev_match_ready_educ_query(
            "rev_match_ready",
            _describe_relation_columns(con, "rev_match_ready"),
        )
    else:
        rev_educ_school_q = _build_rev_educ_school_query(
            "rev_educ",
            "f1_rev_school_cw_filt",
            rev_educ_has_cip=col_flags.get("has_rev_educ_cip", False),
            rev_educ_has_cip_score=col_flags.get("has_rev_educ_cip_score", False),
            rev_educ_has_field_mapped_ind=col_flags.get("has_rev_educ_field_mapped_ind", False),
        )

    rev_educ_school_q_eff = rev_educ_school_q
    if testing or (shard_mode and use_unitid_candidate_join):
        if use_unitid_candidate_join:
            rev_educ_school_q_eff = f"""
            SELECT res.* FROM ({rev_educ_school_q}) AS res
            INNER JOIN (
                SELECT DISTINCT resolved_unitid
                FROM {f1_spells_tab}
                WHERE resolved_unitid IS NOT NULL
            ) AS sp
                ON res.rev_unitid = sp.resolved_unitid
            """
        elif testing:
            rev_educ_school_q_eff = f"""
            SELECT res.* FROM ({rev_educ_school_q}) AS res
            INNER JOIN (SELECT DISTINCT school_name FROM {f1_spells_tab}) AS sp
                ON res.f1_school_name = sp.school_name
            """

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        materialize_table(f"{pfx}_rev_educ_school", rev_educ_school_q_eff, con=con)
        rev_educ_school_tab = f"{pfx}_rev_educ_school"
    else:
        # Materialize in production — the upstream education relation is expensive.
        materialize_table("_rev_educ_school", rev_educ_school_q_eff, con=con)
        rev_educ_school_tab = "_rev_educ_school"

    n_rev_educ_school = int(con.sql(f"SELECT COUNT(*) FROM {rev_educ_school_tab}").fetchone()[0])
    print(f"  {n_rev_educ_school:,} Revelio user×school records ({_fmt_elapsed(time.perf_counter() - t0)})")

    if school_block_mode == "campus_unique" and not use_unitid_candidate_join:
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

    if use_unitid_candidate_join:
        candidates_q = _build_candidates_unitid_query(
            f1_spells_tab=f1_spells_tab,
            rev_match_ready_educ_tab=rev_educ_school_tab,
            rev_indiv_tab="rev_indiv_norm",
        )
    else:
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
        rev_pos_user_stats_name = f"{pfx}_rev_pos_user_stats" if pfx else "_f1_rev_pos_user_stats"
        emp_idf_name      = f"{pfx}_emp_idf"      if pfx else "_f1_emp_idf"
        token_idf_name    = "_f1_token_idf"  # always global (not test-prefixed)

        materialize_table(
            rev_pos_full_name,
            _build_rev_pos_full_query("rev_pos", candidates_tab),
            con=con,
        )
        materialize_table(
            rev_pos_user_stats_name,
            f"""
            SELECT user_id, COUNT(*) AS n_rev_positions
            FROM {rev_pos_full_name}
            GROUP BY user_id
            """,
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
        rev_pos_user_stats_name = None
        if not col_flags["has_rev_pos"]:
            print("\n[Stages 2b/3b: skipped — rev_pos not found]")
        else:
            print("\n[Stages 2b/3b: skipped — no employer lookup (employer_score will be NULL for all)]")

    # ------------------------------------------------------------------
    # 4. Combine candidates + employer scores → merge_scored
    # ------------------------------------------------------------------
    print("\n[Stage 4: Scoring (combine candidates + employer sequence scores)]")
    print(
        "  scoring_mode: "
        f"{'multiplicative' if cfg.BUILD_MULTIPLICATIVE_SCORE else 'additive'}"
        f" | gradyr_decay_power={cfg.BUILD_GRADYR_SCORE_DECAY_POWER:.2f}"
    )
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
        rev_pos_user_stats_tab=rev_pos_user_stats_name,
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
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY inst_score),    3) AS inst_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY gradyr_score),  3) AS grad_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY field_score),   3) AS field_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY
                CASE WHEN employer_score IS NOT NULL THEN employer_score END), 3)  AS emp_p50,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_score),   3) AS tot_p50,
            ROUND(AVG(emp_score_available_ind), 3)                                 AS pct_with_emp,
            ROUND(AVG(CASE WHEN w_emp_eff > 0 THEN w_emp_eff END), 3)             AS w_emp_eff_mean
        FROM {merge_scored_tab}
    """).df().to_string(index=False)
    print(f"\n  Score distributions (merge_scored):\n  {score_stats}")
    print(f"  ({_fmt_elapsed(time.perf_counter() - t0)})")

    print("\n[Stage 4b: Candidate evidence filters]")
    t0 = time.perf_counter()
    emp_filtered_q = _build_candidate_filtered_query(merge_scored_tab)
    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        materialize_table(f"{pfx}_merge_scored_emp_filt", emp_filtered_q, con=con)
        merge_scored_filt_tab = f"{pfx}_merge_scored_emp_filt"
    else:
        materialize_table("_f1_merge_scored_emp_filt", emp_filtered_q, con=con)
        merge_scored_filt_tab = "_f1_merge_scored_emp_filt"
    emp_filter_stats = con.sql(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT spell_id) AS n_spells,
            COALESCE(SUM(CASE WHEN emp_score_available_ind = 1 THEN 1 ELSE 0 END), 0) AS n_rows_with_emp,
            COALESCE(SUM(CASE WHEN emp_score_available_ind = 1 AND employment_history_pass_ind = 1 THEN 1 ELSE 0 END), 0) AS n_rows_emp_pass,
            COALESCE(SUM(CASE WHEN field_candidate_pass_ind = 1 THEN 1 ELSE 0 END), 0) AS n_rows_field_pass,
            COALESCE(SUM(CASE WHEN field_candidate_pass_ind = 0 THEN 1 ELSE 0 END), 0) AS n_rows_field_fail,
            COALESCE(SUM(CASE WHEN employer_relative_pass_ind = 1 THEN 1 ELSE 0 END), 0) AS n_rows_emp_rel_pass,
            COALESCE(SUM(CASE WHEN employer_relative_pass_ind = 0 THEN 1 ELSE 0 END), 0) AS n_rows_emp_rel_fail,
            COALESCE(SUM(CASE WHEN field_relative_pass_ind = 1 THEN 1 ELSE 0 END), 0) AS n_rows_field_rel_pass,
            COALESCE(SUM(CASE WHEN field_relative_pass_ind = 0 THEN 1 ELSE 0 END), 0) AS n_rows_field_rel_fail,
            COALESCE(SUM(CASE WHEN country_relative_pass_ind = 1 THEN 1 ELSE 0 END), 0) AS n_rows_country_rel_pass,
            COALESCE(SUM(CASE WHEN country_relative_pass_ind = 0 THEN 1 ELSE 0 END), 0) AS n_rows_country_rel_fail
        FROM {merge_scored_filt_tab}
        """
    ).df().iloc[0]
    filt_counts = _f1_merge_stage_counts(f"SELECT * FROM {merge_scored_filt_tab}", con=con)
    _print_merge_stage("candidate_filt", filt_counts)
    print(
        f"  employment filter kept "
        f"{int(emp_filter_stats['n_rows_emp_pass']):,} / {int(emp_filter_stats['n_rows_with_emp']):,} "
        "rows with employer-history evidence"
    )
    print(
        f"  field filter kept "
        f"{int(emp_filter_stats['n_rows_field_pass']):,} rows "
        f"and dropped {int(emp_filter_stats['n_rows_field_fail']):,} weak mapped-field rows"
    )
    print(
        f"  relative employer filter kept "
        f"{int(emp_filter_stats['n_rows_emp_rel_pass']):,} rows "
        f"and dropped {int(emp_filter_stats['n_rows_emp_rel_fail']):,}"
    )
    print(
        f"  relative field filter kept "
        f"{int(emp_filter_stats['n_rows_field_rel_pass']):,} rows "
        f"and dropped {int(emp_filter_stats['n_rows_field_rel_fail']):,}"
    )
    print(
        f"  relative country filter kept "
        f"{int(emp_filter_stats['n_rows_country_rel_pass']):,} rows "
        f"and dropped {int(emp_filter_stats['n_rows_country_rel_fail']):,}"
    )
    print(f"  ({_fmt_elapsed(time.perf_counter() - t0)})")

    # ------------------------------------------------------------------
    # 5. Filtering, weighting, ranking
    # ------------------------------------------------------------------
    print("\n[Stage 4a: Filtering, weighting, ranking]")
    t0 = time.perf_counter()

    if testing and cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES:
        pfx = cfg.TESTING_TABLE_PREFIX
        materialize_table(f"{pfx}_match_filt",
                          _build_stage_match_filt_sql(merge_scored_filt_tab), con=con)
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
                          _build_stage_match_filt_sql(merge_scored_filt_tab),
                          con=con)
        print("  Materializing weighted...")
        materialize_table("_f1_weighted",
                          _build_stage_weighted_sql("_f1_match_filt"),
                          con=con)
        print("  Materializing baseline...")
        materialize_table("_f1_baseline",
                          _build_stage_final_sql("_f1_weighted"),
                          con=con)
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

    if shard_mode:
        print("\n[Stage 6: Writing shard artifacts]")
        baseline_shard_q = f"SELECT * FROM {baseline_tab} WHERE match_rank = 1"
        baseline_shard_counts = _f1_merge_stage_counts(baseline_shard_q, con=con)
        _print_merge_stage("baseline_rank1_shard", baseline_shard_counts)
        print("  Writing shard spell-level baseline artifact...")
        write_query_to_parquet(
            query=baseline_shard_q,
            out_path=shard_output_paths["baseline_parquet"],
            overwrite=overwrite,
            con=con,
        )
        person_agg_rows = int(con.sql(f"SELECT COUNT(*) FROM {person_agg_tab}").fetchone()[0])
        print(f"  Writing shard person-level aggregate artifact ({person_agg_rows:,} rows)...")
        write_query_to_parquet(
            query=f"SELECT * FROM {person_agg_tab}",
            out_path=shard_output_paths["person_agg_parquet"],
            overwrite=overwrite,
            con=con,
        )
        print("\n" + "=" * 70)
        print("F1 MERGE SHARD COMPLETE — Summary")
        print(f"  run_tag:       {cfg.RUN_TAG}")
        print(f"  person_shard:  {person_shard_label}")
        print(
            f"  shard baseline:{baseline_shard_counts['n_persons']:,} persons, "
            f"{baseline_shard_counts['n_spells']:,} spells, "
            f"{baseline_shard_counts['mult']:.2f}x mult"
        )
        print(f"  shard person_agg rows: {person_agg_rows:,}")
        print(f"  Total elapsed: {_fmt_elapsed(time.perf_counter() - t_total)}")
        print("=" * 70)
        return

    assignment_label = (
        "global 1:1 assignment"
        if cfg.BUILD_ENFORCE_INDIVIDUAL_ONE_TO_ONE
        else "per-person rank-1 only"
    )
    print(f"  Resolving person-user matches via {assignment_label}...", end=" ", flush=True)
    t_assign = time.perf_counter()
    assigned_people_df = _solve_individual_assignment(
        con,
        person_agg_tab,
        enforce_one_to_one=cfg.BUILD_ENFORCE_INDIVIDUAL_ONE_TO_ONE,
    )
    con.execute("DROP VIEW IF EXISTS _person_assigned_df_view")
    con.register("_person_assigned_df_view", assigned_people_df)
    assigned_people_tab = f"{cfg.TESTING_TABLE_PREFIX}_person_assigned" if testing else "_f1_person_assigned"
    materialize_table(assigned_people_tab, "SELECT * FROM _person_assigned_df_view", con=con)
    con.execute("DROP VIEW IF EXISTS _person_assigned_df_view")
    print(f"{len(assigned_people_df):,} assigned pairs ({_fmt_elapsed(time.perf_counter() - t_assign)})")

    person_rank1 = con.sql(f"""
        SELECT
            COUNT(DISTINCT person_id)   AS n_persons,
            COUNT(DISTINCT user_id)     AS n_users,
            SUM(has_employer_match_ind) AS n_with_employer,
            ROUND(AVG(person_weight_norm), 3)       AS avg_weight_norm,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY person_score_sum), 3) AS median_score
        FROM {assigned_people_tab}
    """).df().iloc[0]
    pct_emp = 100 * person_rank1["n_with_employer"] / max(1, person_rank1["n_persons"])
    print(f"  person matches: {int(person_rank1['n_persons']):,} persons, "
          f"{int(person_rank1['n_users']):,} distinct user_ids "
          f"({_fmt_elapsed(time.perf_counter() - t0)})")
    print(f"  With employer confirmation: {int(person_rank1['n_with_employer']):,} ({pct_emp:.1f}%)")
    print(f"  avg person_weight_norm={person_rank1['avg_weight_norm']:.3f}, "
          f"median person_score_sum={person_rank1['median_score']:.3f}")

    assigned_join = (
        f"JOIN {assigned_people_tab} AS ap "
        "ON b.person_id = ap.person_id AND b.user_id = ap.user_id"
    )

    # ------------------------------------------------------------------
    # 6. Testing mode: spotcheck and exit
    # ------------------------------------------------------------------
    if testing:
        testing_baseline_q = f"SELECT b.* FROM {baseline_tab} AS b {assigned_join} WHERE b.match_rank = 1"
        print("\n[Spotcheck: top spell-level matches for sampled person_ids]")
        _print_f1_testing_spotcheck(
            final_query=testing_baseline_q,
            sample_n=min(10, cfg.TESTING_SAMPLE_N_PERSONS),
            con=con,
        )

        print("\n[Spotcheck: person-level aggregation (rank-1 per person_id)]")
        person_sample = con.sql(f"""
            SELECT person_id, user_id, person_score_sum, n_evidence_units,
                   n_spell_matches, total_emp_matches, has_employer_match_ind,
                   person_weight_norm, person_match_rank
            FROM {assigned_people_tab}
            WHERE person_id IN (
                SELECT DISTINCT person_id FROM {assigned_people_tab}
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
    baseline_output_q = f"SELECT b.* FROM {baseline_tab} AS b {assigned_join} WHERE b.match_rank = 1"
    write_query_to_parquet(
        query=baseline_output_q,
        out_path=cfg.F1_MERGE_BASELINE_PARQUET,
        overwrite=overwrite, con=con,
    )

    # Mult2/4/6 variants
    for cutoff, out_path in [
        (2, cfg.F1_MERGE_MULT2_PARQUET),
        (4, cfg.F1_MERGE_MULT4_PARQUET),
        (6, cfg.F1_MERGE_MULT6_PARQUET),
    ]:
        mult_q = (
            f"SELECT b.* FROM {baseline_tab} AS b {assigned_join} "
            f"WHERE b.match_rank = 1 AND b.n_match_filt <= {cutoff}"
        )
        mult_counts = _f1_merge_stage_counts(mult_q, con=con)
        _print_merge_stage(f"mult{cutoff}", mult_counts)
        write_query_to_parquet(query=mult_q, out_path=out_path, overwrite=overwrite, con=con)

    # Strict spell-level variant
    strict_q = (
        f"SELECT s.* FROM ({_build_f1_stage_strict_query(baseline_tab)}) AS s "
        f"JOIN {assigned_people_tab} AS ap "
        "ON s.person_id = ap.person_id AND s.user_id = ap.user_id"
    )
    strict_counts = _f1_merge_stage_counts(strict_q, con=con)
    _print_merge_stage("strict", strict_counts)
    write_query_to_parquet(query=strict_q, out_path=cfg.F1_MERGE_STRICT_PARQUET, overwrite=overwrite, con=con)

    base_counts = _f1_merge_stage_counts(baseline_output_q, con=con)

    # Person-level outputs
    print("  Writing person-level outputs...")
    write_query_to_parquet(
        query=f"SELECT * FROM {assigned_people_tab}",
        out_path=cfg.F1_MERGE_PERSON_BASELINE_PARQUET,
        overwrite=overwrite, con=con,
    )

    strict_person_min_wn = cfg.STRICT_PERSON_MIN_WEIGHT_NORM
    strict_person_q = (
        f"SELECT * FROM {assigned_people_tab} "
        f"WHERE person_weight_norm >= {strict_person_min_wn}"
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
# Diagnostic: audit raw candidates for a single person_id
# ---------------------------------------------------------------------------

def audit_person_candidates(
    person_id,
    testing: bool | None = None,
    educ_spells_tab: str | None = None,
    candidates_tab: str | None = None,
    merge_scored_tab: str | None = None,
    con=con_f1,
) -> None:
    """Print all raw candidate rows after the join and before downstream filters."""
    table_names = get_runtime_table_names(testing=testing)
    educ_spells_tab = educ_spells_tab or table_names["f1_educ_spells"]
    candidates_tab = candidates_tab or table_names["candidates"]
    merge_scored_tab = merge_scored_tab or table_names["merge_scored"]

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"RAW CANDIDATE AUDIT: person_id={person_id}")
    print(sep)

    try:
        spells_df = con.sql(f"""
            SELECT spell_id, school_name AS f1_school_name, f1_degree_level,
                   f1_prog_end_year, f1_country_std, f1_cip4
            FROM {educ_spells_tab}
            WHERE person_id = {person_id}
            ORDER BY f1_prog_end_year, spell_id
        """).df()
    except Exception as e:
        print(f"[F1 education spells] Could not query {educ_spells_tab}: {e}")
        spells_df = None

    print("\nF1 Education Spells:")
    if spells_df is None or spells_df.empty:
        print("  (none found)")
    else:
        for _, row in spells_df.iterrows():
            print(
                f"  spell={row['spell_id']} | school={row['f1_school_name']} "
                f"| degree={row['f1_degree_level']} | gradyr={row['f1_prog_end_year']} "
                f"| country={row['f1_country_std']} | CIP4={row['f1_cip4']}"
            )

    candidate_source_tab = candidates_tab
    try:
        source_columns = _describe_relation_columns(con, merge_scored_tab)
        candidate_source_tab = merge_scored_tab
    except Exception:
        source_columns = _describe_relation_columns(con, candidates_tab)
    source_field_raw_col = _first_present_col(source_columns, ["field_raw"])
    source_field_raw_expr = (
        f"CAST(src.{source_field_raw_col} AS VARCHAR)"
        if source_field_raw_col is not None
        else "NULL::VARCHAR"
    )

    detail_q = f"""
        WITH target_rows AS (
            SELECT *
            FROM {candidate_source_tab}
            WHERE person_id = {person_id}
        ),
        {_build_field_raw_lookup_cte_sql(
            con,
            user_filter_sql="SELECT DISTINCT user_id FROM target_rows",
        )}
        SELECT
            src.spell_id,
            src.person_id,
            src.user_id,
            src.fullname,
            src.f1_school_name,
            src.f1_degree_level,
            src.f1_prog_end_year,
            src.f1_country_std,
            src.rev_university_raw,
            src.rev_country,
            src.rev_degree_clean,
            COALESCE({source_field_raw_expr}, fr.field_raw) AS field_raw,
            src.rev_educ_start_year,
            src.rev_educ_end_year,
            src.school_match_score,
            src.school_resolution_status,
            src.school_block_applied_ind,
            src.school_match_count_pre,
            src.school_match_count_post,
            src.n_match_raw,
            src.country_score,
            src.inst_score,
            src.degree_score,
            src.gradyr_score,
            src.field_score,
            {_optional_select_expr(source_columns, ['n_f1_employers'], 'n_f1_employers', 'BIGINT')},
            {_optional_select_expr(source_columns, ['w_emp_eff'], 'w_emp_eff', 'DOUBLE')},
            {_optional_select_expr(source_columns, ['employer_score'], 'employer_score', 'DOUBLE')},
            {_optional_select_expr(source_columns, ['total_score'], 'total_score', 'DOUBLE')},
            {_optional_select_expr(source_columns, ['employment_history_pass_ind'], 'employment_history_pass_ind', 'INTEGER')}
        FROM target_rows AS src
        LEFT JOIN field_raw_lookup AS fr
          ON src.user_id = fr.user_id
         AND COALESCE(CAST(src.rev_university_raw AS VARCHAR), '') = COALESCE(fr.rev_university_raw, '')
         AND COALESCE(CAST(src.rev_degree_clean AS VARCHAR), '') = COALESCE(fr.rev_degree_clean, '')
         AND COALESCE(CAST(src.rev_educ_start_year AS BIGINT), -1) = COALESCE(fr.rev_educ_start_year, -1)
         AND COALESCE(CAST(src.rev_educ_end_year AS BIGINT), -1) = COALESCE(fr.rev_educ_end_year, -1)
        ORDER BY
            src.spell_id,
            total_score DESC NULLS LAST,
            src.school_match_score DESC NULLS LAST,
            src.country_score DESC NULLS LAST,
            src.user_id,
            src.rev_educ_end_year NULLS LAST
    """
    try:
        cand_df = con.sql(detail_q).df()
    except Exception as e:
        print(f"\n[Raw candidates] Could not query {candidate_source_tab}: {e}")
        return

    print(f"\nCandidate source: {candidate_source_tab}")
    if cand_df.empty:
        print("  (no candidates found for this person_id)")
        return

    for spell_id, grp in cand_df.groupby("spell_id", sort=False):
        top = grp.iloc[0]
        unique_users = int(grp["user_id"].nunique())
        print(f"\n{'-' * 70}")
        print(
            f"SPELL {spell_id} | raw candidates={len(grp)} rows | unique users={unique_users} "
            f"| n_match_raw={int(top['n_match_raw'])}"
        )
        print(
            f"F1: school={top['f1_school_name']} | degree={top['f1_degree_level']} "
            f"| gradyr={top['f1_prog_end_year']} | country={top['f1_country_std']}"
        )
        if "school_resolution_status" in grp.columns:
            print(
                f"     school_resolution={top['school_resolution_status']} "
                f"| blocked={int(top.get('school_block_applied_ind') or 0)} "
                f"| school_matches={int(top.get('school_match_count_pre') or 0)}"
                f"→{int(top.get('school_match_count_post') or 0)}"
            )
        for idx, (_, row) in enumerate(grp.iterrows(), start=1):
            emp_score = row.get("employer_score")
            total_score = row.get("total_score")
            emp_pass = row.get("employment_history_pass_ind")
            emp_str = f"{emp_score:.3f}" if pd.notna(emp_score) else "N/A"
            total_str = f"{total_score:.3f}" if pd.notna(total_score) else "N/A"
            emp_pass_str = str(int(emp_pass)) if pd.notna(emp_pass) else "N/A"
            print(
                f"  #{idx} user={row['user_id']} "
                f"name={str(row.get('fullname', ''))[:30]:<30} "
                f"rev_school={row['rev_university_raw']} "
                f"| rev_degree={row['rev_degree_clean']} "
                f"| field_raw={row.get('field_raw')} "
                f"| rev_gradyr={row['rev_educ_end_year']} "
                f"| country={row['rev_country']}"
            )
            print(
                f"     school={row['school_match_score']:.3f} "
                f"country={row['country_score']:.3f} "
                f"inst={row['inst_score']:.3f} "
                f"deg={row['degree_score']:.3f} "
                f"grad={row['gradyr_score']:.3f} "
                f"field={row['field_score']:.3f} "
                f"emp={emp_str} "
                f"total={total_str} "
                f"emp_pass={emp_pass_str}"
            )
            for breakdown_line in _format_total_score_breakdown(row):
                print(f"     {breakdown_line}")


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
    table_names = get_runtime_table_names(testing=testing)

    person_agg_tab = person_agg_tab or table_names["person_agg"]
    educ_spells_tab = educ_spells_tab or table_names["f1_educ_spells"]
    opt_emp_tab = opt_emp_tab or table_names["f1_opt_employers"]
    baseline_tab = baseline_tab or table_names["baseline"]
    rev_pos_tab = rev_pos_tab or table_names["rev_pos_full"]
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
                   f1_prog_end_year, f1_country_std, f1_cip4
            FROM {educ_spells_tab}
            WHERE person_id = {person_id}
            ORDER BY f1_prog_end_year, spell_id
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
                  f"| degree={row['f1_degree_level']} | gradyr={row['f1_prog_end_year']} "
                  f"| country={row['f1_country_std']} | CIP4={row['f1_cip4']}")

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
    emp_idf_tab = table_names["emp_idf"]

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
                            {_sql_emp_year_overlap_expr('rp.pos_start_year', 'rp.pos_end_year', 'fe.min_f1_year', 'fe.max_f1_year')} AS year_ok,
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
                                -- Mirror the real matcher: fuzzy/subset are only
                                -- available when the F1 employer has no usable rcid.
                                WHEN f1_rcid IS NULL AND jw_score >= {fuz_thresh}          THEN 'fuzzy'
                                WHEN f1_rcid IS NULL
                                     AND subset_flag
                                     AND shorter_n_tok >= {sub_min_tok}                    THEN 'subset'
                                ELSE NULL
                            END AS match_source,
                            CASE
                                WHEN NOT year_ok                                           THEN 0.0
                                WHEN entity_match                                          THEN 1.0
                                WHEN f1_rcid IS NULL AND jw_score >= {fuz_thresh}          THEN jw_score
                                WHEN f1_rcid IS NULL
                                     AND subset_flag
                                     AND shorter_n_tok >= {sub_min_tok}                    THEN jw_score
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
            WITH cand AS (
                SELECT person_id, user_id, person_score_sum, n_evidence_units,
                       n_spell_matches, total_emp_matches, has_employer_match_ind,
                       person_weight_norm, person_match_rank
                FROM {person_agg_tab}
                WHERE person_id = {person_id}
            ),
            baseline_user AS (
                SELECT
                    user_id,
                    arg_max(fullname, total_score) AS fullname,
                    arg_max(country_score, total_score) AS country_score
                FROM {baseline_tab}
                WHERE person_id = {person_id}
                GROUP BY user_id
            )
            SELECT
                cand.*,
                bu.fullname,
                bu.country_score
            FROM cand
            LEFT JOIN baseline_user AS bu
              ON cand.user_id = bu.user_id
            ORDER BY cand.person_match_rank
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
        baseline_columns = _describe_relation_columns(con, baseline_tab)
        baseline_field_raw_col = _first_present_col(baseline_columns, ["field_raw"])
        baseline_field_raw_expr = (
            f"CAST(src.{baseline_field_raw_col} AS VARCHAR)"
            if baseline_field_raw_col is not None
            else "NULL::VARCHAR"
        )
        spell_scores_df = con.sql(f"""
            WITH target_rows AS (
                SELECT *
                FROM {baseline_tab}
                WHERE person_id = {person_id}
                  AND user_id IN (
                      SELECT user_id FROM {person_agg_tab}
                      WHERE person_id = {person_id}
                      ORDER BY person_match_rank
                      LIMIT {top_n}
                  )
            ),
            {_build_field_raw_lookup_cte_sql(
                con,
                user_filter_sql="SELECT DISTINCT user_id FROM target_rows",
            )}
            SELECT
                src.spell_id,
                src.user_id,
                src.match_rank,
                src.total_score,
                src.employer_score,
                src.country_score,
                src.inst_score,
                src.field_score,
                src.degree_score,
                src.gradyr_score,
                src.weight_norm,
                src.n_f1_employers,
                src.n_emp_matched,
                src.w_emp_eff,
                src.f1_school_name,
                src.f1_degree_level,
                src.f1_prog_end_year,
                src.rev_university_raw,
                src.rev_degree_clean,
                COALESCE({baseline_field_raw_expr}, fr.field_raw) AS field_raw
            FROM target_rows AS src
            LEFT JOIN field_raw_lookup AS fr
              ON src.user_id = fr.user_id
             AND COALESCE(CAST(src.rev_university_raw AS VARCHAR), '') = COALESCE(fr.rev_university_raw, '')
             AND COALESCE(CAST(src.rev_degree_clean AS VARCHAR), '') = COALESCE(fr.rev_degree_clean, '')
             AND COALESCE(CAST(src.rev_educ_start_year AS BIGINT), -1) = COALESCE(fr.rev_educ_start_year, -1)
             AND COALESCE(CAST(src.rev_educ_end_year AS BIGINT), -1) = COALESCE(fr.rev_educ_end_year, -1)
            ORDER BY src.user_id, src.spell_id
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
                        -- Year filter: position must overlap the person's F1 activity window,
                        -- with a symmetric slack buffer.
                        {_sql_emp_year_overlap_expr('rp.pos_start_year', 'rp.pos_end_year', 'fe.min_f1_year', 'fe.max_f1_year')} AS year_ok,
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
        fullname = str(cand.get("fullname") or "").strip()
        name_str = f" | name={fullname}" if fullname else ""
        country_score = cand.get("country_score")
        country_score_str = (
            f" | country_score={country_score:.3f}"
            if pd.notna(country_score)
            else ""
        )
        print(f"\n  #{int(cand['person_match_rank'])} user_id={uid}"
              f"{name_str}"
              f"{country_score_str}"
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
                          f" deg={sp['f1_degree_level']} gradyr={sp['f1_prog_end_year']}"
                          f" | rev_school={sp['rev_university_raw']}"
                          f" rev_degree={sp['rev_degree_clean']}"
                          f" field_raw={sp['field_raw']}"
                          f" | {emp_str}"
                          f" cs={sp['country_score']:.3f}"
                          f" inst={sp['inst_score']:.3f}"
                          f" fld={sp['field_score']:.3f}"
                          f" deg={sp['degree_score']:.3f}"
                          f" grad={sp['gradyr_score']:.3f}"
                          f" tot={sp['total_score']:.4f}"
                          f" wt={sp['weight_norm']:.3f}")
                    for breakdown_line in _format_total_score_breakdown(sp):
                        print(f"         {breakdown_line}")

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
