# File Description: Individual-level labor market outcomes around economics→econometrics
#   relabel events in IPEDS/FOIA data. Identifies Revelio individuals who attended
#   relabeled programs and tracks staying-in-US, number of positions, and imputed salary
#   at fixed horizons after graduation. Produces cohort-based event-study plots and a DiD.
#
# Pipeline:
#   Step 1 - Detect relabel events (reuse econ_relabels_opt_usage_v2) + aggregate plots
#   Step 2 - Load full-sample match-ready Revelio education sample
#   Step 3 - Match individuals to relabel events (treated group)
#   Step 4 - Build individual × horizon outcome panel
#   Step 5 - Event study aggregation + plots (treated only)
#   Step 6 - Control group via never-treated econ institution matching
#   Step 7 - Treated vs. control event study plots
#   Step 8 - Staggered DiD

import math
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
# Ensure progress logs flush immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


# ── path setup so we can import from repo root ──────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import relabels_revelio.relabel_indiv_config as cfg
import relabels_revelio.relabel_events_generalized as generalized
import f1_foia.econ_relabels_opt_usage_v2 as v2
import laborlunch_plot_style as llstyle

# ── global config ────────────────────────────────────────────────────────────
print(f"[relabel_indiv] Using config: {cfg.ACTIVE_CONFIG_PATH}")
print(f"[relabel_indiv] run_tag={cfg.RUN_TAG}  testing={cfg.TESTING_ENABLED}")
print(f"[relabel_indiv] horizons={getattr(cfg, 'BUILD_OUTCOME_HORIZONS', [3])}")

OUTCOMES = [
    "in_us",
    "n_pos",
    "n_employers",
    "avg_employer_tenure_years",
    "in_school",
    "salary_imputed",
    "linkedin_active_through_target_year",
    "n_internship_positions",
]
HORIZON_PROFILE_EXCLUDED_OUTCOMES = {"n_internship_positions"}
OUTCOME_LABELS = {
    "in_us":           "Share with active US position",
    "n_pos":           "Mean cumulative post-grad positions",
    "n_employers":     "Mean cumulative post-grad employers",
    "avg_employer_tenure_years": "Mean post-grad tenure per employer (years)",
    "in_school":       "Share enrolled in school",
    "salary_imputed":  "Mean imputed annual compensation (USD)",
    "linkedin_active_through_target_year": "Share active on LinkedIn through target year",
    "n_internship_positions": "Mean education-spell positions",
}
OUTCOME_FILE_LABELS = {
    "in_us": "active_us",
    "n_pos": "active_positions",
    "n_employers": "unique_employers",
    "avg_employer_tenure_years": "employer_tenure",
    "in_school": "in_school",
    "salary_imputed": "compensation",
    "linkedin_active_through_target_year": "linkedin_active",
    "n_internship_positions": "internship_positions",
}
OUTPUT_DIR = Path(cfg.OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_FONT_SIZE = llstyle.BASE_FONT_SIZE
DI_D_PLOT_FONT_SIZE = llstyle.AXIS_LABEL_FONT_SIZE
DI_D_PLOT_MARKER_SIZE = llstyle.MARKER_SIZE
DI_D_ERRORBAR_ALPHA = llstyle.ERRORBAR_ALPHA
DI_D_EVENT_LINE_X = -0.5
EVENT_PLOT_MARKER_SIZE = llstyle.MULTI_MARKER_SIZE
REVELIO_FULL_SAMPLE_COLOR = "#3977B7"
REVELIO_FOIA_LINKED_COLOR = "#B63D4A"
REVELIO_FULL_SAMPLE_FOREIGN_COLOR = "#083B73"
REVELIO_FULL_SAMPLE_NON_FOREIGN_COLOR = "#C7E1F7"
REVELIO_FOIA_LINKED_FOREIGN_COLOR = REVELIO_FOIA_LINKED_COLOR
REVELIO_FOIA_LINKED_NON_FOREIGN_COLOR = "#E8A6AF"
sns.set(style="whitegrid")
llstyle.apply_style()

t0 = time.time()
SUPPORTED_SAMPLE_VARIANTS = {
    "stage04_all",
    "foia_linked_person_baseline",
}
VALID_EVENT_SOURCE_MODES = {"econ_v2", "generalized_final_sample"}
ESTIMATOR_DID = "did"
ESTIMATOR_STACKED_TREATED = "stacked_treated"
ESTIMATOR_BOTH = "both"
VALID_DID_ESTIMATORS = {ESTIMATOR_DID, ESTIMATOR_STACKED_TREATED, ESTIMATOR_BOTH}
PLOT_MODE_EVENT_STUDY = "event_study_by_cohort"
PLOT_MODE_POOLED_POST = "pooled_post_by_horizon"
POOLED_POST_EVENT_MIN = -1
POOLED_POST_EVENT_MAX = 3
POOLED_STATS_HORIZON = 3
VALID_DID_PLOT_MODES = {PLOT_MODE_EVENT_STUDY, PLOT_MODE_POOLED_POST}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _elapsed(since: float) -> str:
    return f"{time.time() - since:.1f}s"


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _event_source_mode() -> str:
    mode = str(getattr(cfg, "BUILD_EVENT_SOURCE_MODE", "econ_v2") or "econ_v2").strip().lower()
    if mode not in VALID_EVENT_SOURCE_MODES:
        raise ValueError(
            f"Unsupported event_source_mode '{mode}'. "
            f"Expected one of {sorted(VALID_EVENT_SOURCE_MODES)}."
        )
    return mode


def _uses_generalized_events() -> bool:
    return _event_source_mode() == "generalized_final_sample"


def _control_group() -> str:
    return generalized._normalize_control_group(  # noqa: SLF001
        getattr(cfg, "BUILD_CONTROL_GROUP", generalized.DEFAULT_CONTROL_GROUP)
    )


def _horizon_profile_outcomes() -> list[str]:
    return [out for out in OUTCOMES if out not in HORIZON_PROFILE_EXCLUDED_OUTCOMES]


def _variant_slug(label: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in str(label).strip().lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "variant"


def _save_and_show(fig: plt.Figure, name: str, analysis_variant: str | None = None) -> Path:
    """Save figure to OUTPUT_DIR and display it."""
    if analysis_variant:
        name = f"{name}_{_variant_slug(analysis_variant)}"
    out = OUTPUT_DIR / f"{name}.png"
    llstyle.savefig(fig, out, dpi=150)
    print(f"  → saved {out}")
    return out


def _soften_errorbar_interval(errorbar_container, alpha: float = DI_D_ERRORBAR_ALPHA) -> None:
    """Apply alpha to Matplotlib errorbar interval artists without fading markers/lines."""
    if errorbar_container is None or not hasattr(errorbar_container, "lines"):
        return
    for artist_group in errorbar_container.lines[1:]:
        if artist_group is None:
            continue
        try:
            iterator = iter(artist_group)
        except TypeError:
            iterator = iter([artist_group])
        for artist in iterator:
            if hasattr(artist, "set_alpha"):
                artist.set_alpha(alpha)


def _table_exists(con: ddb.DuckDBPyConnection, name: str) -> bool:
    return name in {row[0] for row in con.sql("SHOW TABLES").fetchall()}


def _create_empty_temp_view(
    con: ddb.DuckDBPyConnection,
    name: str,
    columns: list[tuple[str, str]],
) -> None:
    selects = ", ".join(
        f"CAST(NULL AS {dtype}) AS {col}" for col, dtype in columns
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {name} AS
        SELECT {selects}
        WHERE 1 = 0
        """
    )


def _cip_prefix_where_clause(col: str = "cip") -> str:
    prefixes = [str(prefix).strip() for prefix in cfg.BUILD_SAMPLE_CIP_PREFIXES if str(prefix).strip()]
    if not prefixes:
        return "TRUE"
    digits_expr = f"regexp_replace(CAST({col} AS VARCHAR), '[^0-9]', '', 'g')"
    parts = [f"{digits_expr} LIKE '{_escape_sql_literal(prefix)}%'" for prefix in prefixes]
    return "(" + " OR ".join(parts) + ")"


def _normalize_cip_sql(col: str) -> str:
    digits_expr = f"regexp_replace(CAST({col} AS VARCHAR), '[^0-9]', '', 'g')"
    return f"CASE WHEN NULLIF({digits_expr}, '') IS NULL THEN NULL ELSE LPAD({digits_expr}, 6, '0') END"


def _clean_institution_name_sql(value_sql: str) -> str:
    stripped = (
        f"REGEXP_REPLACE(COALESCE(CAST({value_sql} AS VARCHAR), ''), "
        r"'(?i)\b(at|campus|inc|the)\b|\([^\)]*\)|\[[^\]]*\]', ' ', 'g')"
    )
    return f"""
        NULLIF(
            TRIM(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(
                        REGEXP_REPLACE(
                            strip_accents(LOWER({stripped})),
                            '\\s?(&|\\+)\\s?', ' and ', 'g'
                        ),
                        '[^a-z0-9]+', ' ', 'g'
                    ),
                    '\\s+', ' ', 'g'
                )
            ),
            ''
        )
    """


def _prepare_institution_alias_quality_view(con: ddb.DuckDBPyConnection) -> None:
    """Build UNITID-level F1/IPEDS aliases for conservative institution checks."""
    if _table_exists(con, "institution_match_aliases"):
        return

    alias_schema = [("unitid", "BIGINT"), ("alias_clean", "VARCHAR")]
    union_parts: list[str] = []

    rev_cw_path = getattr(cfg, "REVELIO_IPEDS_INST_CW_PARQUET", "")
    if rev_cw_path and os.path.exists(rev_cw_path):
        rev_cw_sql = _escape_sql_literal(rev_cw_path)
        union_parts.extend(
            [
                f"""
                SELECT
                    CAST(UNITID AS BIGINT) AS unitid,
                    {_clean_institution_name_sql("f1_instname_clean")} AS alias_clean
                FROM read_parquet('{rev_cw_sql}')
                WHERE UNITID IS NOT NULL AND f1_instname_clean IS NOT NULL
                """,
                f"""
                SELECT
                    CAST(UNITID AS BIGINT) AS unitid,
                    {_clean_institution_name_sql("ipeds_instname_clean")} AS alias_clean
                FROM read_parquet('{rev_cw_sql}')
                WHERE UNITID IS NOT NULL AND ipeds_instname_clean IS NOT NULL
                """,
            ]
        )

    ipeds_crosswalk_path = (
        getattr(cfg, "IPEDS_CROSSWALK_PARQUET", "")
        or getattr(v2, "CROSSWALK_PATH", "")
    )
    if ipeds_crosswalk_path and os.path.exists(ipeds_crosswalk_path):
        ipeds_cw_sql = _escape_sql_literal(ipeds_crosswalk_path)
        union_parts.extend(
            [
                f"""
                SELECT
                    CAST(UNITID AS BIGINT) AS unitid,
                    {_clean_institution_name_sql("instname")} AS alias_clean
                FROM read_parquet('{ipeds_cw_sql}')
                WHERE UNITID IS NOT NULL AND instname IS NOT NULL
                """,
                f"""
                SELECT
                    CAST(UNITID AS BIGINT) AS unitid,
                    {_clean_institution_name_sql("instname_raw")} AS alias_clean
                FROM read_parquet('{ipeds_cw_sql}')
                WHERE UNITID IS NOT NULL AND instname_raw IS NOT NULL
                """,
            ]
        )

    if not union_parts:
        _create_empty_temp_view(con, "institution_match_aliases", alias_schema)
        return

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW institution_match_aliases AS
        SELECT DISTINCT
            unitid,
            alias_clean
        FROM (
            {" UNION ALL ".join(union_parts)}
        )
        WHERE unitid IS NOT NULL
          AND alias_clean IS NOT NULL
          AND LENGTH(alias_clean) >= 3
        """
    )


def _prepare_stage04_rsid_lookup_view(con: ddb.DuckDBPyConnection) -> None:
    """Expose education-level RSID for full-sample rows when available."""
    if _table_exists(con, "stage04_rsid_lookup"):
        return

    empty_schema = [
        ("user_id", "BIGINT"),
        ("education_number", "BIGINT"),
        ("rsid", "BIGINT"),
    ]
    educ_path = getattr(cfg, "REV_EDUC_CLEAN_LONG_PARQUET", "") or getattr(
        cfg, "REV_EDUC_LONG_PARQUET", ""
    )
    if not educ_path or not os.path.exists(educ_path):
        _create_empty_temp_view(con, "stage04_rsid_lookup", empty_schema)
        return

    educ_sql = _escape_sql_literal(educ_path)
    try:
        cols = {
            row[0].lower()
            for row in con.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{educ_sql}')"
            ).fetchall()
        }
    except Exception as exc:
        print(f"  Warning: failed reading education-long schema for rsid lookup: {exc}")
        _create_empty_temp_view(con, "stage04_rsid_lookup", empty_schema)
        return

    if not {"user_id", "education_number", "rsid"}.issubset(cols):
        _create_empty_temp_view(con, "stage04_rsid_lookup", empty_schema)
        return

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW stage04_rsid_lookup AS
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            CAST(education_number AS BIGINT) AS education_number,
            MIN(TRY_CAST(rsid AS BIGINT)) AS rsid
        FROM read_parquet('{educ_sql}')
        WHERE user_id IS NOT NULL
          AND education_number IS NOT NULL
          AND TRY_CAST(rsid AS BIGINT) IS NOT NULL
        GROUP BY
            CAST(user_id AS BIGINT),
            CAST(education_number AS BIGINT)
        """
    )


def _cip_match_level_sql(col: str) -> str:
    digits_expr = f"regexp_replace(CAST({col} AS VARCHAR), '[^0-9]', '', 'g')"
    return f"""
        CASE
            WHEN NULLIF({digits_expr}, '') IS NULL THEN NULL
            WHEN LENGTH({digits_expr}) >= 5 THEN 'cip6'
            WHEN LENGTH({digits_expr}) >= 3 THEN 'cip4'
            ELSE 'cip2'
        END
    """


def _cip_match_code_sql(col: str) -> str:
    digits_expr = f"regexp_replace(CAST({col} AS VARCHAR), '[^0-9]', '', 'g')"
    return f"""
        CASE
            WHEN NULLIF({digits_expr}, '') IS NULL THEN NULL
            WHEN LENGTH({digits_expr}) >= 5 THEN SUBSTR(LPAD({digits_expr}, 6, '0'), 1, 6)
            WHEN LENGTH({digits_expr}) >= 3 THEN SUBSTR(LPAD({digits_expr}, 4, '0'), 1, 4)
            ELSE SUBSTR(LPAD({digits_expr}, 2, '0'), 1, 2)
        END
    """


def _stage04_degree_type_expr(alias: str = "mr") -> str:
    degree_expr = f"LOWER(COALESCE(CAST({alias}.degree_clean AS VARCHAR), ''))"
    compact_expr = f"regexp_replace({degree_expr}, '\\s+', '', 'g')"
    return f"""
        CASE
            WHEN {degree_expr} LIKE '%phd%'
              OR {degree_expr} LIKE '%ph d%'
              OR {degree_expr} LIKE '%doctor%'
              OR {degree_expr} LIKE '%doctoral%'
              OR {degree_expr} LIKE '%jd%'
              OR {degree_expr} LIKE '%md%'
              OR {degree_expr} LIKE '%edd%'
              OR {degree_expr} LIKE '%dba%'
                THEN 'Doctor'
            WHEN {compact_expr} IN ('ma', 'ms', 'mba', 'meng', 'mpp', 'mph', 'mpa', 'mfin', 'msc', 'macc')
              OR {degree_expr} LIKE '%master%'
              OR {degree_expr} LIKE '%masters%'
              OR {degree_expr} LIKE '%m a%'
              OR {degree_expr} LIKE '%m s%'
              OR {degree_expr} LIKE '%ms %'
              OR {degree_expr} LIKE '%ma %'
              OR {degree_expr} LIKE '%mba%'
              OR {degree_expr} LIKE '%meng%'
              OR {degree_expr} LIKE '%m eng%'
              OR {degree_expr} LIKE '%mpp%'
              OR {degree_expr} LIKE '%mph%'
              OR {degree_expr} LIKE '%mpa%'
              OR {degree_expr} LIKE '%mfin%'
                THEN 'Master'
            WHEN {compact_expr} IN ('ba', 'bs', 'bba', 'ab', 'sb')
              OR {degree_expr} LIKE '%bachelor%'
              OR {degree_expr} LIKE '%bachelors%'
              OR {degree_expr} LIKE '%undergraduate%'
              OR {degree_expr} LIKE '%undergrad%'
              OR {degree_expr} LIKE '%b a%'
              OR {degree_expr} LIKE '%b s%'
              OR {degree_expr} LIKE '%ba %'
              OR {degree_expr} LIKE '%bs %'
                THEN 'Bachelor'
            ELSE 'Other'
        END
    """


def _event_id_from_parts(
    *,
    unitid: object,
    relabel_year: object,
    relabel_type: object,
    degree_type: object = pd.NA,
    broad_pair_bin: object = pd.NA,
    awlevel: object = pd.NA,
) -> str:
    parts = [
        str(unitid),
        str(relabel_year),
        str(relabel_type),
        str(degree_type),
        str(broad_pair_bin),
        str(awlevel),
    ]
    return "|".join(parts)


def _attach_event_ids(events_df: pd.DataFrame, *, unitid_col: str = "unitid") -> pd.DataFrame:
    if events_df.empty:
        out = events_df.copy()
        out["event_id"] = pd.Series(dtype="string")
        return out
    out = events_df.copy()
    out["event_id"] = [
        _event_id_from_parts(
            unitid=row.get(unitid_col, pd.NA),
            relabel_year=row.get("relabel_year", pd.NA),
            relabel_type=row.get("relabel_type", pd.NA),
            degree_type=row.get("degree_type", pd.NA),
            broad_pair_bin=row.get("broad_pair_bin", pd.NA),
            awlevel=row.get("awlevel", pd.NA),
        )
        for _, row in out.iterrows()
    ]
    return out


def _generalized_treated_events(relabel_df: pd.DataFrame) -> pd.DataFrame:
    treated_events = generalized.build_broad_treated_events(relabel_df).copy()
    if treated_events.empty:
        return _attach_event_ids(treated_events)
    treated_events = _attach_event_ids(treated_events)
    if "event_flag" not in treated_events.columns:
        treated_events["event_flag"] = 1
    return treated_events


def _event_identity_value(frame: pd.DataFrame) -> pd.Series:
    if "pair_id" in frame.columns:
        pair_id = pd.to_numeric(frame["pair_id"], errors="coerce")
        if pair_id.notna().any():
            return pair_id.fillna(-1).astype("int64").astype(str)
    if "event_id" in frame.columns:
        event_id = frame["event_id"].astype("string")
        if event_id.notna().any():
            return event_id.fillna("__missing_event_id__").astype(str)
    return (
        pd.to_numeric(frame["relabel_year"], errors="coerce").fillna(-1).astype("int64").astype(str)
        + "||"
        + frame.get("relabel_type", pd.Series("__missing_relabel_type__", index=frame.index)).fillna("__missing_relabel_type__").astype(str)
    )


def _sample_view_name(analysis_variant: str) -> str:
    if analysis_variant == "stage04_all":
        return "stage04_sample_all"
    if analysis_variant == "foia_linked_person_baseline":
        return "stage04_sample_foia_linked_person_baseline"
    return f"stage04_sample_{_variant_slug(analysis_variant)}"


def _analysis_horizons() -> list[int]:
    horizons = sorted({int(h) for h in getattr(cfg, "BUILD_OUTCOME_HORIZONS", [3]) if int(h) >= 0})
    if not horizons:
        raise ValueError("No non-negative outcome horizons configured.")
    return horizons


def _first_present_lower(cols: list[str], candidates: list[str]) -> str | None:
    lowered = {str(col).lower(): str(col) for col in cols}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _prepare_revelio_user_features_view(con: ddb.DuckDBPyConnection) -> None:
    if _table_exists(con, "revelio_user_features"):
        return

    empty_schema = [
        ("user_id", "BIGINT"),
        ("female_prob_raw", "DOUBLE"),
        ("origin_country_raw", "VARCHAR"),
        ("est_yob", "INTEGER"),
        ("linkedin_last_education_date", "DATE"),
        ("linkedin_last_position_date", "DATE"),
        ("linkedin_last_activity_date", "DATE"),
        ("linkedin_last_activity_year", "INTEGER"),
    ]

    user_core_path = cfg.REV_USERS_CORE_PARQUET
    if not user_core_path or not os.path.exists(user_core_path):
        print(f"  Warning: Revelio user-core parquet not found: {user_core_path}")
        _create_empty_temp_view(con, "revelio_user_features", empty_schema)
        return

    user_core_sql = _escape_sql_literal(user_core_path)
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW rev_users_core_raw AS
        SELECT * FROM read_parquet('{user_core_sql}')
        """
    )
    core_cols = [row[0] for row in con.sql("DESCRIBE rev_users_core_raw").fetchall()]
    core_user_id = _first_present_lower(core_cols, ["user_id"])
    if core_user_id is None:
        raise ValueError("rev_users_core parquet must contain user_id")
    core_f_prob = _first_present_lower(core_cols, ["f_prob", "f_prob_avg"])
    core_country = _first_present_lower(core_cols, ["top_country_candidate", "origin_country", "user_country_std", "user_country"])
    core_yob = _first_present_lower(core_cols, ["est_yob"])

    core_selects = [
        f"CAST({core_user_id} AS BIGINT) AS user_id",
        (
            f"TRY_CAST({core_f_prob} AS DOUBLE) AS female_prob_raw"
            if core_f_prob is not None else
            "NULL::DOUBLE AS female_prob_raw"
        ),
        (
            f"CAST({core_country} AS VARCHAR) AS origin_country_raw"
            if core_country is not None else
            "NULL::VARCHAR AS origin_country_raw"
        ),
        (
            f"TRY_CAST({core_yob} AS INTEGER) AS est_yob"
            if core_yob is not None else
            "NULL::INTEGER AS est_yob"
        ),
    ]
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW rev_users_core_features AS
        SELECT DISTINCT
            {", ".join(core_selects)}
        FROM rev_users_core_raw
        WHERE {core_user_id} IS NOT NULL
        """
    )

    educ_path = cfg.REV_EDUC_CLEAN_LONG_PARQUET
    if educ_path and os.path.exists(educ_path):
        educ_sql = _escape_sql_literal(educ_path)
        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW rev_educ_clean_long_raw AS
            SELECT * FROM read_parquet('{educ_sql}')
            """
        )
        educ_cols = [row[0] for row in con.sql("DESCRIBE rev_educ_clean_long_raw").fetchall()]
        educ_user = _first_present_lower(educ_cols, ["user_id"])
        educ_start = _first_present_lower(educ_cols, ["ed_startdate", "startdate"])
        educ_end = _first_present_lower(educ_cols, ["ed_enddate", "enddate"])
        if educ_user is None:
            raise ValueError("rev_educ_clean_long parquet must contain user_id")
        date_terms = []
        if educ_start is not None:
            date_terms.append(
                f"SELECT CAST({educ_user} AS BIGINT) AS user_id, TRY_CAST({educ_start} AS DATE) AS activity_date FROM rev_educ_clean_long_raw"
            )
        if educ_end is not None:
            date_terms.append(
                f"SELECT CAST({educ_user} AS BIGINT) AS user_id, TRY_CAST({educ_end} AS DATE) AS activity_date FROM rev_educ_clean_long_raw"
            )
        if date_terms:
            con.sql(
                f"""
                CREATE OR REPLACE TEMP VIEW revelio_last_education_activity AS
                SELECT
                    user_id,
                    MAX(activity_date) AS linkedin_last_education_date
                FROM (
                    {" UNION ALL ".join(date_terms)}
                )
                WHERE activity_date IS NOT NULL
                GROUP BY user_id
                """
            )
        else:
            _create_empty_temp_view(
                con,
                "revelio_last_education_activity",
                [("user_id", "BIGINT"), ("linkedin_last_education_date", "DATE")],
            )
    else:
        print(f"  Warning: Revelio education-long parquet not found: {educ_path}")
        _create_empty_temp_view(
            con,
            "revelio_last_education_activity",
            [("user_id", "BIGINT"), ("linkedin_last_education_date", "DATE")],
        )

    pos_long_path = cfg.REV_POS_CLEAN_LONG_PARQUET or cfg.REV_POS_PARQUET
    if pos_long_path and os.path.exists(pos_long_path):
        pos_long_sql = _escape_sql_literal(pos_long_path)
        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW rev_pos_clean_long_raw AS
            SELECT * FROM read_parquet('{pos_long_sql}')
            """
        )
        pos_cols = [row[0] for row in con.sql("DESCRIBE rev_pos_clean_long_raw").fetchall()]
        pos_user = _first_present_lower(pos_cols, ["user_id"])
        pos_start = _first_present_lower(pos_cols, ["startdate", "start_date", "position_startdate"])
        pos_end = _first_present_lower(pos_cols, ["enddate", "end_date", "position_enddate"])
        if pos_user is None:
            raise ValueError("rev_pos_clean_long parquet must contain user_id")
        date_terms = []
        if pos_start is not None:
            date_terms.append(
                f"SELECT CAST({pos_user} AS BIGINT) AS user_id, TRY_CAST({pos_start} AS DATE) AS activity_date FROM rev_pos_clean_long_raw"
            )
        if pos_end is not None:
            date_terms.append(
                f"SELECT CAST({pos_user} AS BIGINT) AS user_id, TRY_CAST({pos_end} AS DATE) AS activity_date FROM rev_pos_clean_long_raw"
            )
        if date_terms:
            con.sql(
                f"""
                CREATE OR REPLACE TEMP VIEW revelio_last_position_activity AS
                SELECT
                    user_id,
                    MAX(activity_date) AS linkedin_last_position_date
                FROM (
                    {" UNION ALL ".join(date_terms)}
                )
                WHERE activity_date IS NOT NULL
                GROUP BY user_id
                """
            )
        else:
            _create_empty_temp_view(
                con,
                "revelio_last_position_activity",
                [("user_id", "BIGINT"), ("linkedin_last_position_date", "DATE")],
            )
    else:
        print(f"  Warning: Revelio position-long parquet not found: {pos_long_path}")
        _create_empty_temp_view(
            con,
            "revelio_last_position_activity",
            [("user_id", "BIGINT"), ("linkedin_last_position_date", "DATE")],
        )

    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW revelio_user_features AS
        WITH all_users AS (
            SELECT user_id FROM rev_users_core_features
            UNION
            SELECT user_id FROM revelio_last_education_activity
            UNION
            SELECT user_id FROM revelio_last_position_activity
        )
        SELECT
            u.user_id,
            c.female_prob_raw,
            c.origin_country_raw,
            c.est_yob,
            e.linkedin_last_education_date,
            p.linkedin_last_position_date,
            CASE
                WHEN e.linkedin_last_education_date IS NULL THEN p.linkedin_last_position_date
                WHEN p.linkedin_last_position_date IS NULL THEN e.linkedin_last_education_date
                ELSE GREATEST(e.linkedin_last_education_date, p.linkedin_last_position_date)
            END AS linkedin_last_activity_date,
            CAST(
                EXTRACT(
                    YEAR FROM CASE
                        WHEN e.linkedin_last_education_date IS NULL THEN p.linkedin_last_position_date
                        WHEN p.linkedin_last_position_date IS NULL THEN e.linkedin_last_education_date
                        ELSE GREATEST(e.linkedin_last_education_date, p.linkedin_last_position_date)
                    END
                ) AS INTEGER
            ) AS linkedin_last_activity_year
        FROM all_users u
        LEFT JOIN rev_users_core_features c USING (user_id)
        LEFT JOIN revelio_last_education_activity e USING (user_id)
        LEFT JOIN revelio_last_position_activity p USING (user_id)
        """
    )


def _load_instsize_hd_panel(min_year: int | None, max_year: int | None) -> pd.DataFrame:
    hd_dir = cfg.IPEDS_HD_DIR
    if not hd_dir or not os.path.isdir(hd_dir) or min_year is None or max_year is None:
        return pd.DataFrame(columns=["unitid", "grad_year", "instsize_hd"])

    frames: list[pd.DataFrame] = []
    for year in range(max(2004, int(min_year)), min(2024, int(max_year)) + 1):
        path = Path(hd_dir) / f"hd{year}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, dtype=str, encoding="latin1", low_memory=False)
        except Exception as exc:
            print(f"  Warning: failed reading {path}: {exc}")
            continue
        unitid_col = next((col for col in df.columns if str(col).strip().upper() == "UNITID"), None)
        instsize_col = next((col for col in df.columns if str(col).strip().upper() == "INSTSIZE"), None)
        if unitid_col is None or instsize_col is None:
            continue
        slim = df[[unitid_col, instsize_col]].rename(columns={unitid_col: "unitid", instsize_col: "instsize_hd"})
        slim["grad_year"] = year
        frames.append(slim)
    if not frames:
        return pd.DataFrame(columns=["unitid", "grad_year", "instsize_hd"])

    instsize = pd.concat(frames, ignore_index=True)
    instsize["unitid"] = pd.to_numeric(instsize["unitid"], errors="coerce").astype("Int64")
    instsize["grad_year"] = pd.to_numeric(instsize["grad_year"], errors="coerce").astype("Int64")
    instsize["instsize_hd"] = instsize["instsize_hd"].astype("string").str.strip()
    instsize = instsize.dropna(subset=["unitid", "grad_year"]).copy()
    instsize = instsize[instsize["instsize_hd"].notna() & (instsize["instsize_hd"] != "")]
    instsize = instsize.drop_duplicates(["unitid", "grad_year"], keep="first")
    return instsize


def _prepare_school_year_controls_view(con: ddb.DuckDBPyConnection) -> None:
    if _table_exists(con, "school_year_controls"):
        return

    empty_schema = [
        ("unitid", "BIGINT"),
        ("grad_year", "INTEGER"),
        ("instsize_hd", "VARCHAR"),
        ("c21basic_lab", "VARCHAR"),
    ]
    try:
        min_year, max_year = con.sql(
            """
            SELECT MIN(grad_year), MAX(grad_year)
            FROM stage04_educ_base
            """
        ).fetchone()
    except Exception:
        min_year, max_year = (None, None)

    instsize = _load_instsize_hd_panel(min_year, max_year)

    c21_frames = pd.DataFrame(columns=["unitid", "c21basic_lab"])
    ipeds_path = getattr(v2.base, "IPEDS_PATH", "")
    if ipeds_path and os.path.exists(ipeds_path):
        try:
            c21 = pd.read_parquet(ipeds_path, columns=["unitid", "c21basic_lab"])
            c21["unitid"] = pd.to_numeric(c21["unitid"], errors="coerce").astype("Int64")
            c21["c21basic_lab"] = c21["c21basic_lab"].astype("string").str.strip()
            c21 = c21[c21["unitid"].notna() & c21["c21basic_lab"].notna() & (c21["c21basic_lab"] != "")]
            c21_frames = c21.drop_duplicates(["unitid"], keep="first")
        except Exception as exc:
            print(f"  Warning: failed reading IPEDS completions parquet for school controls: {exc}")

    if instsize.empty and c21_frames.empty:
        _create_empty_temp_view(con, "school_year_controls", empty_schema)
        return

    years = []
    if min_year is not None and max_year is not None:
        years = list(range(max(2004, int(min_year)), min(2024, int(max_year)) + 1))

    if instsize.empty:
        if c21_frames.empty or not years:
            _create_empty_temp_view(con, "school_year_controls", empty_schema)
            return
        years_df = pd.DataFrame({"grad_year": years})
        c21_frames["key"] = 1
        years_df["key"] = 1
        school = c21_frames.merge(years_df, on="key", how="inner").drop(columns=["key"])
        school["instsize_hd"] = pd.NA
        school = school[["unitid", "grad_year", "instsize_hd", "c21basic_lab"]]
    else:
        school = instsize.merge(c21_frames, on="unitid", how="left")
    school["unitid"] = pd.to_numeric(school["unitid"], errors="coerce").astype("Int64")
    school["grad_year"] = pd.to_numeric(school["grad_year"], errors="coerce").astype("Int64")
    school["instsize_hd"] = school.get("instsize_hd", pd.Series(dtype="string")).astype("string")
    school["c21basic_lab"] = school.get("c21basic_lab", pd.Series(dtype="string")).astype("string")
    school = school.dropna(subset=["unitid", "grad_year"], how="any")
    school = school.drop_duplicates(["unitid", "grad_year"], keep="first")
    if school.empty:
        _create_empty_temp_view(con, "school_year_controls", empty_schema)
        return

    con.register("school_year_controls_py", school)
    con.sql("CREATE OR REPLACE TEMP VIEW school_year_controls AS SELECT * FROM school_year_controls_py")


def _ensure_enrichment_views(con: ddb.DuckDBPyConnection) -> None:
    _prepare_revelio_user_features_view(con)
    _prepare_school_year_controls_view(con)


def _finalize_variant_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel

    out = panel.copy()
    out["female_prob"] = pd.to_numeric(out.get("female_prob_raw"), errors="coerce").fillna(0.5)

    raw_country = out.get("origin_country_raw", pd.Series(index=out.index, dtype="object"))
    raw_country = raw_country.fillna("").astype(str).str.strip()
    raw_country = raw_country.where(raw_country != "", "Unknown")
    top_n = max(1, int(getattr(cfg, "BUILD_DID_COUNTRY_TOP_N", 20)))
    top_countries = raw_country[raw_country != "Unknown"].value_counts().head(top_n).index
    out["origin_country_raw"] = raw_country
    out["origin_country_bucket"] = raw_country.where(raw_country.isin(top_countries) | (raw_country == "Unknown"), "Other")
    origin_norm = (
        raw_country
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )
    us_origin = origin_norm.isin(
        {
            "united states",
            "united states of america",
            "usa",
            "us",
            "u s",
            "u s a",
            "america",
        }
    )
    observed_origin = raw_country.ne("Unknown")
    variant_raw = out.get("analysis_variant", pd.Series(index=out.index, dtype="object"))
    variant_str = variant_raw.fillna("").astype(str)
    foia_linked_variant = variant_str.eq("foia_linked_person_baseline")
    out["imputed_foreign_ind"] = (
        foia_linked_variant | (observed_origin & ~us_origin)
    ).astype(int)
    out["imputed_foreign_label"] = np.where(
        out["imputed_foreign_ind"].eq(1),
        "Foreign",
        "Non-foreign",
    )
    out["imputed_foreign_source"] = np.where(
        foia_linked_variant,
        "foia_linked_f1",
        "revelio_origin_country",
    )

    age_raw = pd.to_numeric(out.get("age_at_grad_raw"), errors="coerce")
    age_valid = age_raw.where(age_raw.between(15, 80))
    age_median = age_valid.median()
    if pd.isna(age_median):
        age_median = 28.0
    out["age_missing_ind"] = age_valid.isna().astype(int)
    out["age_at_grad"] = age_valid.fillna(float(age_median))

    out["instsize_hd"] = out.get("instsize_hd", pd.Series(index=out.index, dtype="object")).fillna("Unknown").astype(str)
    out["instsize_hd"] = out["instsize_hd"].replace({"": "Unknown"})
    out["c21basic_lab"] = out.get("c21basic_lab", pd.Series(index=out.index, dtype="object")).fillna("Unknown").astype(str)
    out["c21basic_lab"] = out["c21basic_lab"].replace({"": "Unknown"})
    out["linkedin_active_through_target_year"] = pd.to_numeric(
        out.get("linkedin_active_through_target_year"),
        errors="coerce",
    ).fillna(0).astype(int)
    return out


def _analysis_variant_label(variant: object) -> str:
    variant_str = str(variant)
    labels = {
        "stage04_all": "Full-sample",
        "foia_linked_person_baseline": "FOIA-linked",
        "stage04_all_foreign": "Full-sample: foreign",
        "stage04_all_non_foreign": "Full-sample: non-foreign",
        "foia_linked_person_baseline_foreign": "Linked: foreign",
        "foia_linked_person_baseline_non_foreign": "Linked: non-foreign",
    }
    return labels.get(variant_str, variant_str.replace("_", " "))


def _analysis_variant_color(variant: object) -> str:
    variant_str = str(variant)
    colors = {
        "stage04_all": REVELIO_FULL_SAMPLE_COLOR,
        "foia_linked_person_baseline": REVELIO_FOIA_LINKED_COLOR,
        "stage04_all_foreign": REVELIO_FULL_SAMPLE_FOREIGN_COLOR,
        "stage04_all_non_foreign": REVELIO_FULL_SAMPLE_NON_FOREIGN_COLOR,
        "foia_linked_person_baseline_foreign": REVELIO_FOIA_LINKED_FOREIGN_COLOR,
        "foia_linked_person_baseline_non_foreign": REVELIO_FOIA_LINKED_NON_FOREIGN_COLOR,
    }
    return colors.get(variant_str, llstyle.color(2))


def _variant_comparison_order(variants: Sequence[object]) -> list[str]:
    present = {str(variant) for variant in variants}
    preferred = [
        "stage04_all",
        "stage04_all_foreign",
        "foia_linked_person_baseline",
        "stage04_all_non_foreign",
        "foia_linked_person_baseline_foreign",
        "foia_linked_person_baseline_non_foreign",
    ]
    ordered = [variant for variant in preferred if variant in present]
    ordered.extend(str(variant) for variant in variants if str(variant) not in set(ordered))
    return ordered


def _variant_comparison_marker(variant: object) -> str:
    markers = {
        "stage04_all": "o",
        "foia_linked_person_baseline": "s",
        "stage04_all_foreign": "o",
        "stage04_all_non_foreign": "D",
        "foia_linked_person_baseline_foreign": "s",
        "foia_linked_person_baseline_non_foreign": "^",
    }
    return markers.get(str(variant), "o")


def _variant_comparison_offset(variant: object, variants: Sequence[object]) -> float:
    variant_str = str(variant)
    present = {str(value) for value in variants}
    overlay_offsets = {
        "stage04_all": -0.14,
        "stage04_all_foreign": -0.14,
        "stage04_all_non_foreign": 0.0,
        "foia_linked_person_baseline": 0.14,
    }
    if variant_str in overlay_offsets:
        return overlay_offsets[variant_str]
    ordered = _variant_comparison_order(list(present))
    if len(ordered) <= 1:
        return 0.0
    fallback_offsets = {
        value: float(offset)
        for value, offset in zip(ordered, np.linspace(-0.14, 0.14, num=len(ordered)))
    }
    return fallback_offsets.get(variant_str, 0.0)


def _hex_to_rgb(color: str) -> tuple[int, int, int] | None:
    value = str(color).strip().lstrip("#")
    if len(value) != 6:
        return None
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _blend_hex(color: str, target: str, amount: float) -> str:
    rgb = _hex_to_rgb(color)
    target_rgb = _hex_to_rgb(target)
    if rgb is None or target_rgb is None:
        return color
    weight = min(1.0, max(0.0, float(amount)))
    mixed = [
        int(round((1.0 - weight) * channel + weight * target_channel))
        for channel, target_channel in zip(rgb, target_rgb)
    ]
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def _variant_comparison_ylim_map(
    results_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
) -> dict[tuple[object, ...], tuple[float, float]]:
    """Return shared y-axis limits for variant-comparison DiD plots."""
    if results_df.empty or not set(group_cols).issubset(results_df.columns):
        return {}
    work = results_df.copy()
    work["coef"] = pd.to_numeric(work.get("coef"), errors="coerce")
    work["se"] = pd.to_numeric(work.get("se"), errors="coerce").fillna(0.0)
    work = work.dropna(subset=["coef"])
    if work.empty:
        return {}
    out: dict[tuple[object, ...], tuple[float, float]] = {}
    for key, grp in work.groupby(list(group_cols), dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        lower = (grp["coef"] - 1.96 * grp["se"]).min()
        upper = (grp["coef"] + 1.96 * grp["se"]).max()
        if pd.isna(lower) or pd.isna(upper):
            continue
        lower = min(float(lower), 0.0)
        upper = max(float(upper), 0.0)
        span = upper - lower
        pad = 0.08 * span if span > 0 else max(abs(upper), abs(lower), 1.0) * 0.08
        out[key_tuple] = (lower - pad, upper + pad)
    return out


def _is_foreign_heterogeneity_variant(variant: object) -> bool:
    variant_str = str(variant)
    return variant_str.endswith("_foreign") or variant_str.endswith("_non_foreign")


def _build_did_fe_group(df: pd.DataFrame) -> pd.Series:
    unitid = pd.to_numeric(df.get("cluster_unitid"), errors="coerce").astype("Int64").astype(str)
    if not {"broad_pair_bin", "degree_type"}.issubset(df.columns):
        return unitid
    broad_pair_bin = df["broad_pair_bin"].fillna("missing").astype(str)
    degree_type = df["degree_type"].fillna("missing").astype(str)
    return unitid + "||" + broad_pair_bin + "||" + degree_type


def _did_fe_var(df: pd.DataFrame | None) -> str:
    if df is None:
        return "cluster_unitid"
    if "did_fe_group" in df.columns and df["did_fe_group"].dropna().nunique() >= 2:
        return "did_fe_group"
    return "cluster_unitid"


def _did_estimator_mode() -> str:
    estimator = str(getattr(cfg, "BUILD_DID_ESTIMATOR", ESTIMATOR_DID)).strip().lower()
    if estimator not in VALID_DID_ESTIMATORS:
        print(f"  Warning: unsupported did_estimator={estimator!r}; defaulting to 'did'")
        return ESTIMATOR_DID
    return estimator


def _did_estimators_to_run() -> list[str]:
    estimator = _did_estimator_mode()
    if estimator == ESTIMATOR_BOTH:
        return [ESTIMATOR_DID, ESTIMATOR_STACKED_TREATED]
    return [estimator]


def _did_plot_mode() -> str:
    plot_mode = str(
        getattr(cfg, "BUILD_DID_PLOT_MODE", PLOT_MODE_EVENT_STUDY)
        or PLOT_MODE_EVENT_STUDY
    ).strip().lower()
    if plot_mode not in VALID_DID_PLOT_MODES:
        print(
            f"  Warning: unsupported did_plot_mode={plot_mode!r}; "
            f"defaulting to '{PLOT_MODE_EVENT_STUDY}'"
        )
        return PLOT_MODE_EVENT_STUDY
    return plot_mode


def _did_formula(
    reference_cohort_t: int,
    reg_df: pd.DataFrame | None = None,
    *,
    estimator: str = ESTIMATOR_DID,
) -> str:
    fe_var = _did_fe_var(reg_df)
    event_term = (
        f"C(cohort_t, Treatment(reference={reference_cohort_t}))"
        if estimator == ESTIMATOR_STACKED_TREATED
        else f"C(cohort_t, Treatment(reference={reference_cohort_t}))*treated_ind"
    )
    terms = [
        event_term,
        f"C({fe_var})",
        "C(grad_year)",
    ]
    if getattr(cfg, "BUILD_DID_INCLUDE_INDIVIDUAL_CONTROLS", False):
        terms.extend(
            [
                "female_prob",
                "age_at_grad",
                "age_missing_ind",
                "C(origin_country_bucket)",
            ]
        )
    if getattr(cfg, "BUILD_DID_INCLUDE_SCHOOL_CHAR_GRADYEAR_CONTROLS", False):
        terms.extend(
            [
                "C(grad_year):C(instsize_hd)",
                "C(grad_year):C(c21basic_lab)",
            ]
        )
    return " + ".join(terms)


def _pooled_post_formula(reg_df: pd.DataFrame | None = None) -> str:
    fe_var = _did_fe_var(reg_df)
    terms = [
        # treated_ind is absorbed by school-style fixed effects in the pooled setup,
        # so keep only the post main effect plus the DiD interaction.
        "post_ind",
        "post_ind:treated_ind",
        f"C({fe_var})",
        "C(grad_year)",
    ]
    if getattr(cfg, "BUILD_DID_INCLUDE_INDIVIDUAL_CONTROLS", False):
        terms.extend(
            [
                "female_prob",
                "age_at_grad",
                "age_missing_ind",
                "C(origin_country_bucket)",
            ]
        )
    if getattr(cfg, "BUILD_DID_INCLUDE_SCHOOL_CHAR_GRADYEAR_CONTROLS", False):
        terms.extend(
            [
                "C(grad_year):C(instsize_hd)",
                "C(grad_year):C(c21basic_lab)",
            ]
        )
    return " + ".join(terms)


def _pooled_post_event_bounds() -> tuple[int, int]:
    post_min = int(getattr(cfg, "BUILD_POOLED_POST_EVENT_MIN", POOLED_POST_EVENT_MIN))
    post_max = int(getattr(cfg, "BUILD_POOLED_POST_EVENT_MAX", POOLED_POST_EVENT_MAX))
    if post_min > post_max:
        raise ValueError(
            f"BUILD_POOLED_POST_EVENT_MIN ({post_min}) cannot exceed "
            f"BUILD_POOLED_POST_EVENT_MAX ({post_max})."
        )
    return post_min, post_max


def _agg_cohort_time(
    panel: pd.DataFrame,
    group_col: str | None = None,
    *,
    observed_only: bool = True,
) -> pd.DataFrame:
    """Aggregate outcome panel by cohort_t and horizon_years."""
    if panel.empty:
        keys = ["horizon_years", "cohort_t"] if group_col is None else ["horizon_years", "cohort_t", group_col]
        return pd.DataFrame(columns=keys + ["n"])

    work = panel.copy()
    if observed_only and "target_year_observed" in work.columns:
        work = work[work["target_year_observed"] == 1].copy()
    if work.empty:
        keys = ["horizon_years", "cohort_t"] if group_col is None else ["horizon_years", "cohort_t", group_col]
        return pd.DataFrame(columns=keys + ["n"])

    keys = ["horizon_years", "cohort_t"] if group_col is None else ["horizon_years", "cohort_t", group_col]
    rows = []
    for grp_vals, grp in work.groupby(keys, dropna=False):
        if not isinstance(grp_vals, tuple):
            grp_vals = (grp_vals,)
        row: dict[str, object] = dict(zip(keys, grp_vals))
        row["n"] = len(grp)
        if "target_year_observed" in grp.columns:
            row["target_year_observed_share"] = float(grp["target_year_observed"].mean())
        if "used_latest_avail" in grp.columns:
            row["used_latest_avail_share"] = float(grp["used_latest_avail"].mean())
        for out in OUTCOMES:
            if out in grp.columns:
                row[f"{out}_mean"] = grp[out].mean()
                row[f"{out}_se"] = grp[out].sem()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _coerce_plot_frame(df: pd.DataFrame, cols: list[str], sort_col: str) -> pd.DataFrame:
    """Convert plotting columns to plain numeric values before handing them to Matplotlib."""
    plot_df = df.loc[:, cols].copy()
    for col in cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    return plot_df.dropna(subset=cols).sort_values(sort_col)


def _append_reference_plot_row(
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    reference_x: int,
    extra_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Ensure a plotted event-study line includes the omitted reference point."""
    out = plot_df.copy()
    if out.empty:
        row = {col: np.nan for col in out.columns}
        row[x_col] = float(reference_x)
        row[y_col] = 0.0
        for key, value in (extra_values or {}).items():
            row[key] = value
        return pd.DataFrame([row])

    x_vals = pd.to_numeric(out[x_col], errors="coerce")
    ref_mask = x_vals == float(reference_x)
    if ref_mask.any():
        out.loc[ref_mask, y_col] = 0.0
        for key, value in (extra_values or {}).items():
            if key in out.columns:
                out.loc[ref_mask, key] = value
    else:
        row = {col: np.nan for col in out.columns}
        row[x_col] = float(reference_x)
        row[y_col] = 0.0
        for key, value in (extra_values or {}).items():
            row[key] = value
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
    out = out.dropna(subset=[x_col, y_col]).sort_values(x_col).reset_index(drop=True)
    return out


def _did_models_to_run() -> list[str]:
    model = str(getattr(cfg, "BUILD_DID_MODEL", "simple")).strip().lower()
    if model == "simple":
        return ["simple"]
    if model in {"panel", "both"}:
        print(
            "  Warning: panel DiD is not identified under horizon-based outcomes; "
            "using the clustered simple FE specification instead."
        )
        return ["simple"]
    print(f"  Warning: unsupported did_model={model!r}; defaulting to 'simple'")
    return ["simple"]


def _did_results_output_path() -> Path:
    if getattr(cfg, "OUTPUT_DID_RESULTS_PARQUET", ""):
        return Path(cfg.OUTPUT_DID_RESULTS_PARQUET)
    return OUTPUT_DIR / f"relabel_did_results_{cfg.RUN_TAG}.parquet"


def _horizon_file_suffix(horizon: int, available_horizons: list[int]) -> str:
    return "" if len(set(int(h) for h in available_horizons)) == 1 else f"_t{int(horizon)}"


def _outcome_file_label(outcome: str) -> str:
    return OUTCOME_FILE_LABELS.get(outcome, _variant_slug(outcome))


def _supported_did_cohorts(
    df: pd.DataFrame,
    *,
    reference_cohort_t: int = -2,
) -> list[int]:
    counts = df.groupby(["cohort_t", "treated_ind"]).size().unstack(fill_value=0)
    if reference_cohort_t not in counts.index:
        return []
    if counts.loc[reference_cohort_t].get(0, 0) == 0 or counts.loc[reference_cohort_t].get(1, 0) == 0:
        return []

    supported: list[int] = []
    for cohort_t in sorted(int(v) for v in counts.index.tolist()):
        if cohort_t == reference_cohort_t:
            continue
        if counts.loc[cohort_t].get(0, 0) > 0 and counts.loc[cohort_t].get(1, 0) > 0:
            supported.append(int(cohort_t))
    return supported


def _supported_stacked_cohorts(
    df: pd.DataFrame,
    *,
    reference_cohort_t: int = -2,
) -> list[int]:
    counts = df.groupby("cohort_t").size()
    if reference_cohort_t not in counts.index or int(counts.loc[reference_cohort_t]) == 0:
        return []
    return [
        int(cohort_t)
        for cohort_t in sorted(int(v) for v in counts.index.tolist())
        if int(cohort_t) != reference_cohort_t and int(counts.loc[cohort_t]) > 0
    ]


def _choose_reference_cohort_t(
    df: pd.DataFrame,
    *,
    default: int = -2,
) -> int:
    counts = df.groupby(["cohort_t", "treated_ind"]).size().unstack(fill_value=0)
    if default in counts.index and counts.loc[default].get(0, 0) > 0 and counts.loc[default].get(1, 0) > 0:
        return int(default)

    negative_supported = [
        int(cohort_t)
        for cohort_t in counts.index.tolist()
        if int(cohort_t) < 0 and counts.loc[cohort_t].get(0, 0) > 0 and counts.loc[cohort_t].get(1, 0) > 0
    ]
    if negative_supported:
        return max(negative_supported)

    supported = [
        int(cohort_t)
        for cohort_t in counts.index.tolist()
        if counts.loc[cohort_t].get(0, 0) > 0 and counts.loc[cohort_t].get(1, 0) > 0
    ]
    if supported:
        return min(supported)
    return int(default)


def _choose_stacked_reference_cohort_t(
    df: pd.DataFrame,
    *,
    default: int = -2,
) -> int:
    counts = df.groupby("cohort_t").size()
    if default in counts.index and int(counts.loc[default]) > 0:
        return int(default)
    negative_supported = [int(cohort_t) for cohort_t in counts.index.tolist() if int(cohort_t) < 0]
    if negative_supported:
        return max(negative_supported)
    if len(counts.index):
        return min(int(cohort_t) for cohort_t in counts.index.tolist())
    return int(default)


def _find_did_interaction_param(
    params: pd.Series,
    cohort_t: int,
    reference_cohort_t: int,
) -> str | None:
    term = f"C(cohort_t, Treatment(reference={reference_cohort_t}))"
    candidates = [
        f"{term}[T.{cohort_t}]:treated_ind",
        f"{term}[{cohort_t}]:treated_ind",
        f"treated_ind:{term}[T.{cohort_t}]",
        f"treated_ind:{term}[{cohort_t}]",
    ]
    for candidate in candidates:
        if candidate in params.index:
            return candidate
    return None


def _find_event_time_param(
    params: pd.Series,
    cohort_t: int,
    reference_cohort_t: int,
) -> str | None:
    term = f"C(cohort_t, Treatment(reference={reference_cohort_t}))"
    candidates = [
        f"{term}[T.{cohort_t}]",
        f"{term}[{cohort_t}]",
    ]
    for candidate in candidates:
        if candidate in params.index:
            return candidate
    tokens = {str(int(cohort_t)), f"{float(cohort_t)}", f"{cohort_t:.1f}"}
    for name in map(str, params.index):
        if term not in name:
            continue
        if any(f"[T.{token}]" in name or f"[{token}]" in name for token in tokens):
            return name
    for name in map(str, params.index):
        if "treated" in name or "cohort_t" not in name:
            continue
        if any(token in name for token in tokens):
            return name
    return None


def _normal_pvalue_from_coef_se(coef: float, se: float) -> float:
    if pd.isna(coef) or pd.isna(se) or se <= 0:
        return float("nan")
    z_val = abs(float(coef) / float(se))
    return float(math.erfc(z_val / math.sqrt(2.0)))


def _format_compact_number(value: float) -> str:
    if pd.isna(value):
        return "na"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _series_short_label(label: object) -> str:
    text = str(label).strip()
    replacements = {
        "Full-sample": "Full",
        "FOIA-linked": "Linked",
        "Full-sample: foreign": "Full-F",
        "Full-sample: non-foreign": "Full-NF",
        "Linked: foreign": "Linked-F",
        "Linked: non-foreign": "Linked-NF",
    }
    return replacements.get(text, text)


def _pooled_stats_box_text(
    stats_df: pd.DataFrame,
    *,
    series_col: str = "series_label",
) -> str:
    if stats_df.empty:
        return ""

    work = stats_df.copy()
    if series_col not in work.columns:
        work[series_col] = "Series"
    work["horizon_years"] = pd.to_numeric(work["horizon_years"], errors="coerce")
    work = work.dropna(subset=["horizon_years"]).sort_values([series_col, "horizon_years"])
    if work.empty:
        return ""

    series_values = list(dict.fromkeys(work[series_col].astype(str).tolist()))
    multiple_series = len(series_values) > 1
    lines = [
        "Series        Base    Coef    Eff"
        if multiple_series
        else "Horizon       Base    Coef    Eff"
    ]
    for _, row in work.iterrows():
        series_label = _series_short_label(row.get(series_col, "Series"))
        row_label = f"{series_label} h{int(row['horizon_years'])}" if multiple_series else f"h{int(row['horizon_years'])}"
        effect = row.get("effect_size", np.nan)
        effect_text = "na" if pd.isna(effect) else f"{100.0 * float(effect):.1f}%"
        lines.append(
            f"{row_label:<12} "
            f"{_format_compact_number(row.get('baseline_mean', np.nan)):>6} "
            f"{_format_compact_number(row.get('coef', np.nan)):>7} "
            f"{effect_text:>7}"
        )
    return "\n".join(lines)


def _annotate_pooled_stats_box(
    ax: plt.Axes,
    stats_df: pd.DataFrame,
    *,
    series_col: str = "series_label",
) -> None:
    text = _pooled_stats_box_text(stats_df, series_col=series_col)
    if not text:
        return
    n_rows = max(1, text.count("\n") + 1)
    font_size = 8 if n_rows <= 7 else 7
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=font_size,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#9aa0a6",
            "alpha": 0.92,
        },
        zorder=10,
    )


def _plot_did_coefficients(
    results_df: pd.DataFrame,
    *,
    analysis_variant: str | None = None,
) -> None:
    """Plot cohort-based DiD coefficients for one analysis variant."""
    if results_df.empty:
        return

    order = [out for out in OUTCOMES if out in set(results_df["outcome"])]
    if not order:
        return

    reference_cohort_t = (
        int(results_df["reference_cohort_t"].dropna().iloc[0])
        if "reference_cohort_t" in results_df.columns
        else -2
    )
    available_horizons = sorted(results_df["horizon_years"].dropna().astype(int).unique().tolist())
    for did_model, model_grp in results_df.groupby("did_model", sort=False):
        for horizon, grp in model_grp.groupby("horizon_years", sort=True):
            cohort_ticks = sorted(
                {int(v) for v in grp["cohort_t"].dropna().tolist()} | {reference_cohort_t}
            )
            if not cohort_ticks:
                continue

            for outcome in order:
                plot_df = grp[grp["outcome"] == outcome].dropna(subset=["coef", "se"])
                if plot_df.empty:
                    continue
                line_df = _coerce_plot_frame(plot_df, ["cohort_t", "coef", "se"], sort_col="cohort_t")
                if line_df.empty:
                    continue
                line_df = _append_reference_plot_row(
                    line_df,
                    x_col="cohort_t",
                    y_col="coef",
                    reference_x=reference_cohort_t,
                    extra_values={"se": 0.0},
                )

                fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
                errorbar_container = ax.errorbar(
                    line_df["cohort_t"].to_numpy(dtype=float),
                    line_df["coef"].to_numpy(dtype=float),
                    yerr=1.96 * line_df["se"].to_numpy(dtype=float),
                    fmt="o-",
                    color=llstyle.color(0),
                    ecolor=llstyle.rgba(llstyle.color(0), DI_D_ERRORBAR_ALPHA),
                    elinewidth=DI_D_PLOT_MARKER_SIZE,
                    capsize=0,
                    markersize=DI_D_PLOT_MARKER_SIZE,
                    linewidth=1.6,
                )
                _soften_errorbar_interval(errorbar_container)
                ax.scatter(
                    [reference_cohort_t],
                    [0.0],
                    facecolors="white",
                    edgecolors="black",
                    linewidths=1.4,
                    s=70,
                    zorder=4,
                )
                ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
                ax.axvline(x=DI_D_EVENT_LINE_X, linestyle=":", color="gray", linewidth=1)
                ax.set_xticks(cohort_ticks)
                ax.set_title("")
                ax.set_xlabel(
                    "Graduation year relative to relabel event",
                    fontsize=DI_D_PLOT_FONT_SIZE,
                )
                ax.set_ylabel(
                    f"did coef: {OUTCOME_LABELS.get(outcome, outcome)}",
                    fontsize=DI_D_PLOT_FONT_SIZE,
                )
                _save_and_show(
                    fig,
                    f"did_att_{_outcome_file_label(outcome)}{_horizon_file_suffix(int(horizon), available_horizons)}",
                    analysis_variant=analysis_variant,
                )


def _plot_did_variant_comparison(
    results_df: pd.DataFrame,
    *,
    file_tag: str = "variant",
    title_label: str = "match sample",
    ylim_by_key: dict[tuple[object, ...], tuple[float, float]] | None = None,
) -> None:
    """Plot DiD coefficients across analysis variants for each model spec."""
    if results_df.empty or results_df["analysis_variant"].nunique() < 2:
        return

    order = [out for out in OUTCOMES if out in set(results_df["outcome"])]
    if not order:
        return

    variant_order = _variant_comparison_order(results_df["analysis_variant"].tolist())
    reference_cohort_t = (
        int(results_df["reference_cohort_t"].dropna().iloc[0])
        if "reference_cohort_t" in results_df.columns
        else -2
    )
    available_horizons = sorted(results_df["horizon_years"].dropna().astype(int).unique().tolist())

    for did_model, model_grp in results_df.groupby("did_model", sort=False):
        for horizon, grp in model_grp.groupby("horizon_years", sort=True):
            cohort_ticks = sorted(
                {int(v) for v in grp["cohort_t"].dropna().tolist()} | {reference_cohort_t}
            )
            if not cohort_ticks:
                continue

            for outcome in order:
                outcome_grp = grp[grp["outcome"] == outcome]
                if outcome_grp.empty:
                    continue

                fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
                plotted_any = False
                legend_handles = None
                legend_labels = None

                for idx, variant in enumerate(variant_order):
                    sub = outcome_grp[outcome_grp["analysis_variant"] == variant].dropna(subset=["coef", "se"])
                    if sub.empty:
                        continue
                    line_df = _coerce_plot_frame(sub, ["cohort_t", "coef", "se"], sort_col="cohort_t")
                    if line_df.empty:
                        continue
                    line_df = _append_reference_plot_row(
                        line_df,
                        x_col="cohort_t",
                        y_col="coef",
                        reference_x=reference_cohort_t,
                        extra_values={"se": 0.0},
                    )
                    plotted_any = True
                    color = _analysis_variant_color(variant)
                    x_vals = line_df["cohort_t"].to_numpy(dtype=float) + _variant_comparison_offset(variant, variant_order)
                    errorbar_container = ax.errorbar(
                        x_vals,
                        line_df["coef"].to_numpy(dtype=float),
                        yerr=1.96 * line_df["se"].to_numpy(dtype=float),
                        fmt="-",
                        color=color,
                        ecolor=llstyle.rgba(color, DI_D_ERRORBAR_ALPHA),
                        elinewidth=DI_D_PLOT_MARKER_SIZE,
                        capsize=0,
                        markersize=DI_D_PLOT_MARKER_SIZE,
                        linewidth=1.5,
                        marker=_variant_comparison_marker(variant),
                        label=_analysis_variant_label(variant),
                        zorder=3 + idx,
                    )
                    _soften_errorbar_interval(errorbar_container)

                ax.scatter(
                    [reference_cohort_t],
                    [0.0],
                    facecolors="white",
                    edgecolors="black",
                    linewidths=1.3,
                    s=65,
                    zorder=4,
                )
                ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
                ax.axvline(x=DI_D_EVENT_LINE_X, linestyle=":", color="gray", linewidth=1)
                ax.set_xticks(cohort_ticks)
                ax.set_title("")
                ax.set_xlabel(
                    "Graduation year relative to relabel event",
                    fontsize=DI_D_PLOT_FONT_SIZE,
                )
                ax.set_ylabel(
                    f"did coef: {OUTCOME_LABELS.get(outcome, outcome)}",
                    fontsize=DI_D_PLOT_FONT_SIZE,
                )
                ylim = (ylim_by_key or {}).get((did_model, horizon, outcome))
                if ylim is not None:
                    ax.set_ylim(*ylim)
                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()

                if not plotted_any:
                    plt.close(fig)
                    continue

                if legend_handles:
                    llstyle.right_legend(ax)
                _save_and_show(
                    fig,
                    f"did_att_by_{file_tag}_{_outcome_file_label(outcome)}"
                    f"{_horizon_file_suffix(int(horizon), available_horizons)}",
                )


def _plot_did_horizon_comparison(
    results_df: pd.DataFrame,
    *,
    analysis_variant: str,
    file_tag: str,
    title_label: str,
) -> None:
    """Plot one analysis variant with separate lines for each outcome horizon."""
    if results_df.empty:
        return

    work = results_df[results_df["analysis_variant"].eq(analysis_variant)].copy()
    if work.empty or work["horizon_years"].nunique() < 2:
        return

    order = [out for out in _horizon_profile_outcomes() if out in set(work["outcome"])]
    if not order:
        return

    reference_cohort_t = (
        int(work["reference_cohort_t"].dropna().iloc[0])
        if "reference_cohort_t" in work.columns and work["reference_cohort_t"].notna().any()
        else -2
    )
    horizon_order = sorted(work["horizon_years"].dropna().astype(int).unique().tolist())
    palette = [llstyle.color(idx) for idx in range(len(horizon_order))]
    marker_cycle = ["o", "s", "D", "^", "P", "X"]
    horizon_offsets = (
        np.linspace(-0.16, 0.16, num=len(horizon_order))
        if len(horizon_order) > 1
        else np.array([0.0])
    )

    for did_model, model_grp in work.groupby("did_model", sort=False):
        for outcome in order:
            outcome_grp = model_grp[model_grp["outcome"] == outcome]
            if outcome_grp.empty:
                continue

            cohort_ticks = sorted(
                {int(v) for v in outcome_grp["cohort_t"].dropna().tolist()} | {reference_cohort_t}
            )
            if not cohort_ticks:
                continue

            fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
            plotted_any = False
            legend_handles = None
            legend_labels = None

            for idx, (color, horizon) in enumerate(zip(palette, horizon_order)):
                sub = outcome_grp[
                    outcome_grp["horizon_years"].astype(int).eq(int(horizon))
                ].dropna(subset=["coef", "se"])
                if sub.empty:
                    continue
                line_df = _coerce_plot_frame(sub, ["cohort_t", "coef", "se"], sort_col="cohort_t")
                if line_df.empty:
                    continue
                line_df = _append_reference_plot_row(
                    line_df,
                    x_col="cohort_t",
                    y_col="coef",
                    reference_x=reference_cohort_t,
                    extra_values={"se": 0.0},
                )
                plotted_any = True
                x_vals = line_df["cohort_t"].to_numpy(dtype=float) + float(horizon_offsets[idx])
                errorbar_container = ax.errorbar(
                    x_vals,
                    line_df["coef"].to_numpy(dtype=float),
                    yerr=1.96 * line_df["se"].to_numpy(dtype=float),
                    fmt="-",
                    color=color,
                    ecolor=llstyle.rgba(color, DI_D_ERRORBAR_ALPHA),
                    elinewidth=DI_D_PLOT_MARKER_SIZE,
                    capsize=0,
                    markersize=DI_D_PLOT_MARKER_SIZE,
                    linewidth=1.5,
                    marker=marker_cycle[idx % len(marker_cycle)],
                    label=f"{int(horizon)} years after grad",
                    zorder=3 + idx,
                )
                _soften_errorbar_interval(errorbar_container)

            ax.scatter(
                [reference_cohort_t],
                [0.0],
                facecolors="white",
                edgecolors="black",
                linewidths=1.3,
                s=65,
                zorder=4,
            )
            ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
            ax.axvline(x=DI_D_EVENT_LINE_X, linestyle=":", color="gray", linewidth=1)
            ax.set_xticks(cohort_ticks)
            ax.set_title("")
            ax.set_xlabel(
                "Graduation year relative to relabel event",
                fontsize=DI_D_PLOT_FONT_SIZE,
            )
            ax.set_ylabel(
                f"did coef: {OUTCOME_LABELS.get(outcome, outcome)}",
                fontsize=DI_D_PLOT_FONT_SIZE,
            )
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            if not plotted_any:
                plt.close(fig)
                continue

            if legend_handles:
                llstyle.right_legend(ax)
            _save_and_show(
                fig,
                f"did_att_{file_tag}_{_outcome_file_label(outcome)}",
            )


def _fit_pooled_post_result(
    sub: pd.DataFrame,
    *,
    outcome: str,
    analysis_variant: str | None,
    horizon: int,
) -> dict[str, object] | None:
    if len(sub) < 5:
        return None
    if sub["treated_ind"].nunique() < 2 or sub["post_ind"].nunique() < 2:
        return None
    post_min, post_max = _pooled_post_event_bounds()
    treated_pre = sub.loc[
        (sub["treated_ind"] == 1) & (sub["cohort_t"] < post_min),
        outcome,
    ]
    treated_post = sub.loc[
        (sub["treated_ind"] == 1) & (sub["cohort_t"].between(post_min, post_max)),
        outcome,
    ]
    control_pre = sub.loc[
        (sub["treated_ind"] == 0) & (sub["cohort_t"] < post_min),
        outcome,
    ]
    control_post = sub.loc[
        (sub["treated_ind"] == 0) & (sub["cohort_t"].between(post_min, post_max)),
        outcome,
    ]
    if treated_pre.empty or treated_post.empty or control_pre.empty or control_post.empty:
        return None

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None

    formula = f"{outcome} ~ {_pooled_post_formula(sub)}"
    try:
        if sub["cluster_unitid"].nunique() >= 2:
            try:
                result = smf.ols(formula=formula, data=sub).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": sub["cluster_unitid"]},
                )
            except Exception as exc:
                print(
                    f"    {outcome} [pooled] h={int(horizon)}: "
                    f"clustered SE failed ({exc}); falling back to HC1"
                )
                result = smf.ols(formula=formula, data=sub).fit(cov_type="HC1")
        else:
            result = smf.ols(formula=formula, data=sub).fit(cov_type="HC1")
    except Exception as exc:
        print(f"    {outcome} [pooled] h={int(horizon)}: regression failed ({exc})")
        return None

    param = None
    for candidate in ("post_ind:treated_ind", "treated_ind:post_ind"):
        if candidate in result.params.index:
            param = candidate
            break
    if param is None:
        return None

    coef = float(result.params.get(param, float("nan")))
    se = float(result.bse.get(param, float("nan")))
    pval = _normal_pvalue_from_coef_se(coef, se)
    baseline_mean = float(treated_pre.mean())
    effect_size = (
        float(coef / baseline_mean)
        if pd.notna(baseline_mean) and not np.isclose(baseline_mean, 0.0)
        else float("nan")
    )

    return {
        "analysis_variant": analysis_variant or "unspecified",
        "did_model": "simple",
        "did_estimator": ESTIMATOR_DID,
        "outcome": outcome,
        "horizon_years": int(horizon),
        "coef": coef,
        "se": se,
        "pval": pval,
        "ci_lower": coef - 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan,
        "ci_upper": coef + 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan,
        "baseline_mean": baseline_mean,
        "effect_size": effect_size,
        "pooled_post_event_min": post_min,
        "pooled_post_event_max": post_max,
        "treated_pre_mean": baseline_mean,
        "treated_post_mean": float(treated_post.mean()),
        "control_pre_mean": float(control_pre.mean()),
        "control_post_mean": float(control_post.mean()),
        "n_obs": int(result.nobs),
        "n_users": int(sub["user_id"].nunique()),
        "n_entities": int(sub["did_entity_id"].nunique()),
        "n_unitids": int(sub["cluster_unitid"].nunique()),
        "formula": formula,
        "did_include_individual_controls": int(
            getattr(cfg, "BUILD_DID_INCLUDE_INDIVIDUAL_CONTROLS", False)
        ),
        "did_include_school_char_gradyear_controls": int(
            getattr(cfg, "BUILD_DID_INCLUDE_SCHOOL_CHAR_GRADYEAR_CONTROLS", False)
        ),
    }


def _plot_pooled_horizon_profile(
    results_df: pd.DataFrame,
    *,
    analysis_variant: str,
    file_tag: str,
    title_label: str,
) -> None:
    work = results_df[results_df["analysis_variant"].eq(analysis_variant)].copy()
    if work.empty:
        return

    order = [out for out in _horizon_profile_outcomes() if out in set(work["outcome"])]
    if not order:
        return

    for did_model, model_grp in work.groupby("did_model", sort=False):
        for outcome in order:
            outcome_grp = model_grp[model_grp["outcome"] == outcome].dropna(subset=["coef", "se"])
            if outcome_grp.empty:
                continue
            line_df = _coerce_plot_frame(outcome_grp, ["horizon_years", "coef", "se"], sort_col="horizon_years")
            if line_df.empty:
                continue

            fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
            errorbar_container = ax.errorbar(
                line_df["horizon_years"].to_numpy(dtype=float),
                line_df["coef"].to_numpy(dtype=float),
                yerr=1.96 * line_df["se"].to_numpy(dtype=float),
                fmt="o-",
                color=llstyle.color(0),
                ecolor=llstyle.rgba(llstyle.color(0), DI_D_ERRORBAR_ALPHA),
                elinewidth=DI_D_PLOT_MARKER_SIZE,
                capsize=0,
                markersize=DI_D_PLOT_MARKER_SIZE,
                linewidth=1.6,
            )
            _soften_errorbar_interval(errorbar_container)
            ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
            ax.set_xticks(sorted(line_df["horizon_years"].astype(int).unique().tolist()))
            ax.set_xlabel(
                "Calendar year relative to graduation",
                fontsize=DI_D_PLOT_FONT_SIZE,
            )
            ax.set_ylabel(
                f"did coef: {OUTCOME_LABELS.get(outcome, outcome)}",
                fontsize=DI_D_PLOT_FONT_SIZE,
            )
            ax.set_title("")
            stats_df = outcome_grp[
                pd.to_numeric(outcome_grp["horizon_years"], errors="coerce").eq(POOLED_STATS_HORIZON)
            ].copy()
            if not stats_df.empty:
                _annotate_pooled_stats_box(ax, stats_df)
            _save_and_show(
                fig,
                f"did_att_{_outcome_file_label(outcome)}_{file_tag}",
            )


def _plot_pooled_variant_comparison(
    results_df: pd.DataFrame,
    *,
    file_tag: str,
    title_label: str,
    ylim_by_key: dict[tuple[object, ...], tuple[float, float]] | None = None,
) -> None:
    if results_df.empty or results_df["analysis_variant"].nunique() < 2:
        return

    order = [out for out in _horizon_profile_outcomes() if out in set(results_df["outcome"])]
    if not order:
        return

    variant_order = _variant_comparison_order(results_df["analysis_variant"].tolist())

    for did_model, model_grp in results_df.groupby("did_model", sort=False):
        for outcome in order:
            outcome_grp = model_grp[model_grp["outcome"] == outcome].copy()
            if outcome_grp.empty:
                continue
            horizon_ticks = sorted(outcome_grp["horizon_years"].dropna().astype(int).unique().tolist())
            if not horizon_ticks:
                continue

            fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
            plotted_any = False

            for idx, variant in enumerate(variant_order):
                sub = outcome_grp[outcome_grp["analysis_variant"] == variant].dropna(subset=["coef", "se"])
                if sub.empty:
                    continue
                line_df = _coerce_plot_frame(sub, ["horizon_years", "coef", "se"], sort_col="horizon_years")
                if line_df.empty:
                    continue
                plotted_any = True
                color = _analysis_variant_color(variant)
                x_vals = line_df["horizon_years"].to_numpy(dtype=float) + _variant_comparison_offset(variant, variant_order)
                errorbar_container = ax.errorbar(
                    x_vals,
                    line_df["coef"].to_numpy(dtype=float),
                    yerr=1.96 * line_df["se"].to_numpy(dtype=float),
                    fmt="-",
                    color=color,
                    ecolor=llstyle.rgba(color, DI_D_ERRORBAR_ALPHA),
                    elinewidth=DI_D_PLOT_MARKER_SIZE,
                    capsize=0,
                    markersize=DI_D_PLOT_MARKER_SIZE,
                    linewidth=1.5,
                    marker=_variant_comparison_marker(variant),
                    label=_analysis_variant_label(variant),
                    zorder=3 + idx,
                )
                _soften_errorbar_interval(errorbar_container)

            if not plotted_any:
                plt.close(fig)
                continue

            ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
            ax.set_xticks(horizon_ticks)
            ax.set_xlabel(
                "Calendar year relative to graduation",
                fontsize=DI_D_PLOT_FONT_SIZE,
            )
            ax.set_ylabel(
                f"did coef: {OUTCOME_LABELS.get(outcome, outcome)}",
                fontsize=DI_D_PLOT_FONT_SIZE,
            )
            ylim = (ylim_by_key or {}).get((did_model, outcome))
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_title("")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                llstyle.right_legend(ax)
            stats_df = outcome_grp[
                pd.to_numeric(outcome_grp["horizon_years"], errors="coerce").eq(POOLED_STATS_HORIZON)
            ].copy()
            if not stats_df.empty:
                stats_df["series_label"] = stats_df["analysis_variant"].map(_analysis_variant_label)
                _annotate_pooled_stats_box(ax, stats_df, series_col="series_label")
            _save_and_show(
                fig,
                f"did_att_by_{file_tag}_{_outcome_file_label(outcome)}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Detect relabel events + aggregate FOIA/IPEDS plots
# ─────────────────────────────────────────────────────────────────────────────

def step1_relabels(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load relabel events/panel for the configured event source.
    Reproduces the legacy aggregate FOIA/IPEDS plots only for econ_v2 mode.
    """
    t = time.time()
    print("\n── Step 1: Detecting relabel events ──────────────────────────────────")
    event_mode = _event_source_mode()
    print(f"  event_source_mode={event_mode}")

    if event_mode == "econ_v2":
        relabels_path = cfg.RELABELS_PARQUET
        if os.path.exists(relabels_path) and not cfg.BUILD_OVERWRITE:
            print(f"  Loading cached relabels from {relabels_path}")
            relabel_df = pd.read_parquet(relabels_path)
        else:
            print("  Running v2 relabel detector (IPEDS + FOIA)...")
            relabel_df = v2.detect_econ_relabels(con)
            if relabel_df.empty:
                raise RuntimeError("No relabel events found. Check IPEDS data path.")
            os.makedirs(os.path.dirname(relabels_path), exist_ok=True)
            relabel_df.to_parquet(relabels_path, index=False)
            print(f"  Saved relabel panel → {relabels_path}")
        treated_events = relabel_df[relabel_df["event_flag"] == 1].copy()
    else:
        panel_path = cfg.GENERALIZED_RELABELS_PANEL_PARQUET
        if not panel_path or not os.path.exists(panel_path):
            raise FileNotFoundError(
                f"Generalized relabel panel parquet not found: {panel_path}"
            )
        print(f"  Loading finalized generalized relabel panel from {panel_path}")
        relabel_df = pd.read_parquet(panel_path)
        treated_events = _generalized_treated_events(relabel_df)
        if treated_events.empty:
            raise RuntimeError("No generalized treated relabel events found in finalized panel.")

    n_inst = treated_events["unitid"].nunique()
    n_events = len(treated_events)
    print(f"  Relabel events: {n_events} events at {n_inst} institutions")
    print(f"  Year range: {treated_events['relabel_year'].min()} – {treated_events['relabel_year'].max()}")
    print(f"  By type:\n{treated_events.groupby('relabel_type').size().to_string()}")

    if event_mode == "generalized_final_sample":
        if "degree_type" in treated_events.columns:
            print(
                "  By degree type:\n"
                + treated_events.groupby("degree_type").size().sort_values(ascending=False).to_string()
            )
        if "broad_pair_bin" in treated_events.columns:
            print(
                "  By broad bin:\n"
                + treated_events.groupby("broad_pair_bin").size().sort_values(ascending=False).to_string()
            )

    print("  Generating aggregate diagnostics...")
    try:
        fig_hist, ax_hist = plt.subplots(figsize=llstyle.FIGSIZE)
        hist_hue = "degree_type" if event_mode == "generalized_final_sample" else "relabel_type"
        sns.histplot(
            data=treated_events,
            x="relabel_year",
            bins=range(
                int(treated_events["relabel_year"].min()),
                int(treated_events["relabel_year"].max()) + 2,
            ),
            discrete=True,
            hue=hist_hue,
            multiple="dodge",
            ax=ax_hist,
        )
        ax_hist.set_xlabel("Relabel year")
        ax_hist.set_ylabel("Count of relabel events")
        llstyle.right_legend(ax_hist)
        _save_and_show(fig_hist, "relabel_year_histogram")
    except Exception as exc:
        print(f"  Warning: relabel-year histogram failed ({exc}); continuing.")

    if event_mode == "econ_v2":
        try:
            opt_usage = v2.compute_opt_usage(
                con,
                relabel_df,
                foia_person_panel_path=cfg.FOIA_PERSON_PANEL_PARQUET,
                stage05_person_baseline_path=cfg.STAGE05_PERSON_BASELINE_PARQUET,
                ipeds_cost_panel_path=cfg.IPEDS_COST_PANEL_PARQUET,
                ipeds_tuition_col=cfg.BUILD_COHORT_EXTERNAL_TUITION_COL,
            )
            if not opt_usage.empty:
                opt_usage_event = v2.compute_opt_usage_event_time(opt_usage)

                cohort_yvars = list(getattr(cfg, "BUILD_COHORT_PLOT_YVARS", [])) or [
                    "opt_share",
                    "opt_stem_share",
                    "avg_tuition",
                ]

                for yvar in cohort_yvars:
                    try:
                        fig_path = v2.plot_opt_usage(
                            opt_usage, yvar=yvar, show=True, save=True
                        )
                        if fig_path:
                            print(f"    saved OPT plot ({yvar}) → {fig_path}")
                    except Exception as e:
                        print(f"    Warning: opt_usage plot ({yvar}) failed: {e}")

                try:
                    ctrl_phys = v2.compute_control_opt_usage_event_time(
                        con,
                        relabel_df,
                        foia_person_panel_path=cfg.FOIA_PERSON_PANEL_PARQUET,
                        stage05_person_baseline_path=cfg.STAGE05_PERSON_BASELINE_PARQUET,
                        ipeds_cost_panel_path=cfg.IPEDS_COST_PANEL_PARQUET,
                        ipeds_tuition_col=cfg.BUILD_COHORT_EXTERNAL_TUITION_COL,
                    )
                    v2.plot_opt_usage_event_time_with_control_label(
                        opt_usage_event=opt_usage_event,
                        control_event=ctrl_phys,
                        control_label="Physical Sciences",
                        yvar="opt_share",
                        show=True,
                        save=True,
                        file_tag="physical_sciences",
                        make_treated_only_plot=True,
                    )
                except Exception as e:
                    print(f"    Warning: physical-sciences control plot failed: {e}")

                try:
                    ctrl_never = v2.compute_never_treated_econ_control_event_time(
                        con,
                        relabel_df,
                        foia_person_panel_path=cfg.FOIA_PERSON_PANEL_PARQUET,
                        stage05_person_baseline_path=cfg.STAGE05_PERSON_BASELINE_PARQUET,
                        ipeds_cost_panel_path=cfg.IPEDS_COST_PANEL_PARQUET,
                        ipeds_tuition_col=cfg.BUILD_COHORT_EXTERNAL_TUITION_COL,
                    )
                    made_treated_only_plot = False
                    for yvar in cohort_yvars:
                        try:
                            v2.plot_opt_usage_event_time_with_control_label(
                                opt_usage_event=opt_usage_event,
                                control_event=ctrl_never,
                                control_label="Never-treated Economics",
                                yvar=yvar,
                                show=True,
                                save=True,
                                file_tag="never_treated_econ",
                                make_treated_only_plot=not made_treated_only_plot,
                            )
                            made_treated_only_plot = True
                        except Exception as exc:
                            print(f"    Warning: never-treated econ raw trend plot ({yvar}) failed: {exc}")

                    did_panel = v2.compute_never_treated_econ_did_panel(
                        con,
                        relabel_df,
                        foia_person_panel_path=cfg.FOIA_PERSON_PANEL_PARQUET,
                        stage05_person_baseline_path=cfg.STAGE05_PERSON_BASELINE_PARQUET,
                        ipeds_cost_panel_path=cfg.IPEDS_COST_PANEL_PARQUET,
                        ipeds_tuition_col=cfg.BUILD_COHORT_EXTERNAL_TUITION_COL,
                    )
                    if did_panel.empty:
                        print("    Warning: never-treated econ DiD panel empty; skipping cohort DiD plots.")
                    else:
                        for yvar in cohort_yvars:
                            try:
                                did_event_study = v2.compute_did_event_study(
                                    did_panel=did_panel,
                                    yvar=yvar,
                                )
                                if did_event_study.empty:
                                    print(
                                        f"    Warning: never-treated econ DiD event study empty for {yvar}; skipping"
                                    )
                                    continue
                                did_out = v2.plot_did_event_study(
                                    did_event_study=did_event_study,
                                    yvar=yvar,
                                    show=True,
                                    save=True,
                                    file_tag=v2.DID_CONTROL_FILE_TAG,
                                )
                                if did_out:
                                    print(f"    saved never-treated econ DiD plot ({yvar}) → {did_out}")
                            except Exception as exc:
                                print(f"    Warning: never-treated econ DiD plot ({yvar}) failed: {exc}")
                except Exception as e:
                    print(f"    Warning: never-treated econ control plot failed: {e}")
            else:
                print("    Warning: OPT usage data is empty; skipping aggregate plots.")
        except Exception as e:
            print(f"  Warning: aggregate plots failed ({e}); continuing.")

    print(f"  Step 1 done in {_elapsed(t)}")
    return relabel_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 - Load full-sample match-ready education sample
# ─────────────────────────────────────────────────────────────────────────────

def _stage04_grad_year_expr(alias: str = "mr") -> str:
    degree_expr = f"LOWER(COALESCE(CAST({alias}.degree_clean AS VARCHAR), ''))"
    start_year_expr = f"CAST(EXTRACT(YEAR FROM TRY_CAST({alias}.ed_startdate AS DATE)) AS INTEGER)"
    end_year_expr = f"CAST(EXTRACT(YEAR FROM TRY_CAST({alias}.ed_enddate AS DATE)) AS INTEGER)"
    return f"""
        CASE
            WHEN TRY_CAST({alias}.ed_enddate AS DATE) IS NOT NULL THEN {end_year_expr}
            WHEN TRY_CAST({alias}.ed_startdate AS DATE) IS NULL THEN NULL::INTEGER
            WHEN {degree_expr} IN ('master', 'masters', 'mba', 'associate', 'associates')
                THEN {start_year_expr} + 2
            WHEN {degree_expr} IN ('doctor', 'doctors', 'doctoral', 'phd', 'ph.d', 'bachelor', 'bachelors')
                THEN {start_year_expr} + 4
            ELSE {start_year_expr} + 4
        END
    """


def step2_prepare_stage04_samples(con: ddb.DuckDBPyConnection) -> list[str]:
    """Build full-sample-based sample views used by all analysis variants."""
    t = time.time()
    print("\n-- Step 2: Loading full-sample/stage-05 sample inputs ------------------")

    requested_variants = list(cfg.BUILD_SAMPLE_VARIANTS or ["stage04_all"])
    invalid_variants = sorted(set(requested_variants) - SUPPORTED_SAMPLE_VARIANTS)
    if invalid_variants:
        raise ValueError(f"Unsupported sample_variants: {invalid_variants}")

    stage04_path = cfg.STAGE04_MERGE_READY_PARQUET
    if not stage04_path or not os.path.exists(stage04_path):
        raise FileNotFoundError(f"Full-sample merge_ready parquet not found: {stage04_path}")

    stage04_path_sql = _escape_sql_literal(stage04_path)
    raw_cols = {
        row[0].lower()
        for row in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{stage04_path_sql}')"
        ).fetchall()
    }
    required_cols = {
        "user_id",
        "education_number",
        "unitid",
        "degree_clean",
        "cip",
        "university_raw",
        "field_clean",
        "ed_startdate",
        "ed_enddate",
        "school_match_score",
    }
    missing_cols = sorted(required_cols - raw_cols)
    if missing_cols:
        raise ValueError(f"Full-sample merge_ready missing required columns: {missing_cols}")

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW stage04_merge_ready_raw AS
        SELECT
            user_id,
            education_number,
            unitid,
            degree_clean,
            cip,
            university_raw,
            field_clean,
            ed_startdate,
            ed_enddate,
            school_match_score
        FROM read_parquet('{stage04_path_sql}')
        """
    )
    _prepare_stage04_rsid_lookup_view(con)
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW stage04_educ_base AS
        SELECT DISTINCT
            CAST(mr.user_id AS BIGINT) AS user_id,
            CAST(mr.education_number AS BIGINT) AS education_number,
            CAST(mr.unitid AS BIGINT) AS unitid,
            CAST(mr.degree_clean AS VARCHAR) AS degree_clean,
            CAST(mr.cip AS VARCHAR) AS cip,
            {_normalize_cip_sql("mr.cip")} AS cip6,
            {_cip_match_level_sql("mr.cip")} AS cip_match_level,
            {_cip_match_code_sql("mr.cip")} AS cip_match_code,
            {_stage04_degree_type_expr("mr")} AS degree_type,
            CAST(mr.university_raw AS VARCHAR) AS university_raw,
            CAST(mr.field_clean AS VARCHAR) AS field_clean,
            TRY_CAST(mr.ed_startdate AS DATE) AS ed_startdate,
            TRY_CAST(mr.ed_enddate AS DATE) AS ed_enddate,
            TRY_CAST(mr.school_match_score AS DOUBLE) AS school_match_score,
            TRY_CAST(rsid.rsid AS BIGINT) AS rsid,
            {_stage04_grad_year_expr("mr")} AS grad_year
        FROM stage04_merge_ready_raw AS mr
        LEFT JOIN stage04_rsid_lookup AS rsid
          ON CAST(mr.user_id AS BIGINT) = rsid.user_id
         AND CAST(mr.education_number AS BIGINT) = rsid.education_number
        WHERE mr.user_id IS NOT NULL
        """
    )

    cip_where = "TRUE" if _uses_generalized_events() else _cip_prefix_where_clause("cip")
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW stage04_sample_all AS
        SELECT *
        FROM stage04_educ_base
        WHERE unitid IS NOT NULL
          AND {cip_where}
          AND grad_year IS NOT NULL
        """
    )

    stats = con.sql(
        f"""
        SELECT
            COUNT(*) AS rows_total,
            COUNT(DISTINCT user_id) AS users_total,
            COUNT(*) FILTER (WHERE unitid IS NOT NULL) AS rows_with_unitid,
            COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where}) AS rows_after_cip_filter,
            COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where} AND grad_year IS NULL) AS rows_missing_grad_year,
            COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where} AND grad_year IS NOT NULL) AS rows_stage04_all,
            COUNT(DISTINCT user_id) FILTER (WHERE unitid IS NOT NULL AND {cip_where} AND grad_year IS NOT NULL)
                AS users_stage04_all
        FROM stage04_educ_base
        """
    ).fetchone()
    if stats is not None:
        print(
            "  full-sample rows: "
            f"{int(stats[0] or 0):,} rows | {int(stats[1] or 0):,} users"
        )
        print(f"  rows with non-null unitid: {int(stats[2] or 0):,}")
        print(f"  rows after CIP filter:     {int(stats[3] or 0):,}")
        print(f"  rows dropped missing grad_year: {int(stats[4] or 0):,}")
        print(
            "  full-sample analysis rows: "
            f"{int(stats[5] or 0):,} rows | {int(stats[6] or 0):,} users"
        )

    stage05_path = cfg.STAGE05_PERSON_BASELINE_PARQUET
    if stage05_path and os.path.exists(stage05_path):
        stage05_path_sql = _escape_sql_literal(stage05_path)
        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW stage05_person_baseline_raw AS
            SELECT * FROM read_parquet('{stage05_path_sql}')
            """
        )
        stage05_cols = {
            row[0].lower()
            for row in con.sql("DESCRIBE stage05_person_baseline_raw").fetchall()
        }
        if "user_id" not in stage05_cols:
            raise ValueError("Stage-05 person baseline parquet must contain user_id")
        rank_filter = ""
        if "person_match_rank" in stage05_cols:
            rank_filter = "AND CAST(person_match_rank AS BIGINT) = 1"
        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW stage05_person_baseline_users AS
            SELECT DISTINCT CAST(user_id AS BIGINT) AS user_id
            FROM stage05_person_baseline_raw
            WHERE user_id IS NOT NULL
              {rank_filter}
            """
        )
        con.sql(
            """
            CREATE OR REPLACE TEMP VIEW stage04_sample_foia_linked_person_baseline AS
            SELECT s.*
            FROM stage04_sample_all AS s
            JOIN stage05_person_baseline_users AS u
              ON s.user_id = u.user_id
            """
        )
        overlap_stats = con.sql(
            """
            SELECT
                (SELECT COUNT(*) FROM stage05_person_baseline_users) AS baseline_users,
                (SELECT COUNT(DISTINCT user_id) FROM stage04_sample_all) AS stage04_users,
                (SELECT COUNT(DISTINCT s.user_id)
                 FROM stage04_sample_all AS s
                 JOIN stage05_person_baseline_users AS u USING (user_id)) AS overlap_users,
                (SELECT COUNT(*) FROM stage04_sample_foia_linked_person_baseline) AS overlap_rows
            """
        ).fetchone()
        if overlap_stats is not None:
            print(
                "  FOIA-linked overlap: "
                f"{int(overlap_stats[2] or 0):,} overlapping users "
                f"({int(overlap_stats[3] or 0):,} education rows) "
                f"vs {int(overlap_stats[0] or 0):,} stage-05 users"
            )
    elif "foia_linked_person_baseline" in requested_variants:
        raise FileNotFoundError(
            f"Stage-05 person baseline parquet not found: {stage05_path}"
        )
    else:
        print("  Stage-05 person baseline parquet not found; FOIA-linked diagnostics skipped")

    print(f"  Step 2 done in {_elapsed(t)}")
    return requested_variants


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Match individuals to relabel events
# ─────────────────────────────────────────────────────────────────────────────

def _empty_indiv_event_matches() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "user_id",
            "unitid",
            "education_number",
            "ed_startdate",
            "ed_enddate",
            "grad_year",
            "relabel_year",
            "relabel_type",
            "cohort_t",
            "treated_ind",
            "pair_id",
            "event_id",
            "broad_pair_bin",
            "degree_type",
            "rsid",
            "rsid_unitid_match_count",
            "rsid_unitid_total_with_rsid",
            "rsid_unitid_match_share",
            "rsid_unitid_required_count",
        ]
    )


def _generalized_membership_rows(
    con: ddb.DuckDBPyConnection,
    sample_view: str,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    del con, sample_view
    empty_cols = ["broad_pair_bin", "cip_match_level", "cip_match_code"]
    if events_df.empty or "broad_pair_bin" not in events_df.columns:
        return pd.DataFrame(columns=empty_cols)
    valid_bins = {
        str(value)
        for value in events_df["broad_pair_bin"].dropna().astype(str).tolist()
    }
    if not valid_bins:
        return pd.DataFrame(columns=empty_cols)

    cip_universe: list[str] = []
    ipeds_path = getattr(generalized.base, "IPEDS_PATH", "")
    if ipeds_path and os.path.exists(ipeds_path):
        try:
            cip_universe = list(generalized._load_ipeds_cip_map(ipeds_path).keys())
        except Exception as exc:
            print(f"  Warning: failed loading IPEDS CIP map for generalized membership rows: {exc}")
    if not cip_universe:
        for col in ("source_cip6", "target_cip6", "event_source_cip6"):
            if col in events_df.columns:
                cip_universe.extend(
                    str(value).zfill(6)
                    for value in events_df[col].dropna().tolist()
                )
    cip_universe = sorted({str(value).zfill(6) for value in cip_universe if pd.notna(value)})
    if not cip_universe:
        return pd.DataFrame(columns=empty_cols)

    membership = generalized.build_broad_bin_membership(cip_universe)
    rows: list[dict[str, str]] = []
    for broad_pair_bin in sorted(valid_bins):
        spec = membership.get(broad_pair_bin, {})
        for cip6 in spec.get("all_cips", ()):
            cip6_str = str(cip6).zfill(6)
            rows.extend(
                [
                    {
                        "broad_pair_bin": broad_pair_bin,
                        "cip_match_level": "cip6",
                        "cip_match_code": cip6_str,
                    },
                    {
                        "broad_pair_bin": broad_pair_bin,
                        "cip_match_level": "cip4",
                        "cip_match_code": cip6_str[:4],
                    },
                    {
                        "broad_pair_bin": broad_pair_bin,
                        "cip_match_level": "cip2",
                        "cip_match_code": cip6_str[:2],
                    },
                ]
            )
    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _match_individuals_to_events(
    con: ddb.DuckDBPyConnection,
    sample_view: str,
    events_df: pd.DataFrame,
    *,
    treated_ind: int,
    group_label: str,
) -> pd.DataFrame:
    if events_df.empty:
        return _empty_indiv_event_matches()
    if not _table_exists(con, sample_view):
        raise ValueError(f"Sample view not found: {sample_view}")

    quality_gate_enabled = bool(
        getattr(cfg, "BUILD_INSTITUTION_MATCH_QUALITY_GATE", True)
    )
    if quality_gate_enabled:
        _prepare_institution_alias_quality_view(con)

    try:
        sample_cols = {
            row[0].lower()
            for row in con.sql(f"DESCRIBE {sample_view}").fetchall()
        }
    except Exception:
        sample_cols = set()

    events_view = f"events_{_variant_slug(group_label)}"
    normalized_events = events_df.copy()
    for column in ("event_id", "broad_pair_bin", "degree_type", "relabel_type"):
        if column in normalized_events.columns:
            normalized_events[column] = normalized_events[column].astype("string")
    con.register(f"{events_view}_py", normalized_events)
    con.sql(f"CREATE OR REPLACE TEMP VIEW {events_view} AS SELECT * FROM {events_view}_py")

    use_generalized_match = _uses_generalized_events() and {
        "broad_pair_bin",
        "degree_type",
    }.issubset(normalized_events.columns)
    join_clauses = ["CAST(s.unitid AS BIGINT) = CAST(ev.unitid AS BIGINT)"]
    match_partition_cols = ["user_id", "relabel_year"]
    matched_cols = [
        "s.user_id",
        "CAST(s.unitid AS BIGINT) AS unitid",
        "CAST(s.education_number AS BIGINT) AS education_number",
        "TRY_CAST(s.ed_startdate AS DATE) AS ed_startdate",
        "TRY_CAST(s.ed_enddate AS DATE) AS ed_enddate",
        "CAST(s.university_raw AS VARCHAR) AS university_raw",
        "TRY_CAST(s.school_match_score AS DOUBLE) AS school_match_score",
        (
            "TRY_CAST(s.rsid AS BIGINT) AS rsid"
            if "rsid" in sample_cols
            else "NULL::BIGINT AS rsid"
        ),
        "CAST(s.grad_year AS INTEGER) AS grad_year",
        "CAST(ev.relabel_year AS INTEGER) AS relabel_year",
        "CAST(ev.relabel_type AS VARCHAR) AS relabel_type",
        "CAST(s.grad_year AS INTEGER) - CAST(ev.relabel_year AS INTEGER) AS cohort_t",
        (
            "TRY_CAST(ev.pair_id AS BIGINT) AS pair_id"
            if "pair_id" in normalized_events.columns
            else "NULL::BIGINT AS pair_id"
        ),
        (
            "CAST(ev.event_id AS VARCHAR) AS event_id"
            if "event_id" in normalized_events.columns
            else "NULL::VARCHAR AS event_id"
        ),
        (
            "CAST(ev.broad_pair_bin AS VARCHAR) AS broad_pair_bin"
            if "broad_pair_bin" in normalized_events.columns
            else "NULL::VARCHAR AS broad_pair_bin"
        ),
        (
            "CAST(ev.degree_type AS VARCHAR) AS degree_type"
            if "degree_type" in normalized_events.columns
            else "NULL::VARCHAR AS degree_type"
        ),
    ]
    extra_order_cols = ["relabel_type"]

    membership_view = None
    if use_generalized_match:
        membership_rows = _generalized_membership_rows(con, sample_view, normalized_events)
        membership_view = f"event_membership_{_variant_slug(group_label)}"
        con.register(f"{membership_view}_py", membership_rows)
        con.sql(
            f"CREATE OR REPLACE TEMP VIEW {membership_view} AS SELECT * FROM {membership_view}_py"
        )
        join_clauses.extend(
            [
                "CAST(COALESCE(s.degree_type, 'Other') AS VARCHAR) = CAST(COALESCE(ev.degree_type, 'Other') AS VARCHAR)",
            ]
        )
        match_partition_cols = ["user_id", "event_id"]
        extra_order_cols = ["broad_pair_bin", "relabel_type"]

    year_window = cfg.BUILD_SAMPLE_GRADYEAR_WINDOW
    join_sql = f"JOIN {events_view} AS ev ON " + " AND ".join(join_clauses)
    if membership_view is not None:
        join_sql += (
            f"\n            JOIN {membership_view} AS m"
            "\n              ON CAST(ev.broad_pair_bin AS VARCHAR) = CAST(m.broad_pair_bin AS VARCHAR)"
            "\n             AND CAST(s.cip_match_level AS VARCHAR) = CAST(m.cip_match_level AS VARCHAR)"
            "\n             AND CAST(s.cip_match_code AS VARCHAR) = CAST(m.cip_match_code AS VARCHAR)"
        )

    where_conditions = [
        f"ABS(CAST(s.grad_year AS INTEGER) - CAST(ev.relabel_year AS INTEGER)) <= {year_window}"
    ]
    if not use_generalized_match:
        where_conditions.append("CAST(COALESCE(s.degree_type, '') AS VARCHAR) = 'Master'")
    where_sql = "\n              AND ".join(where_conditions)
    quality_cte_sql = ""
    ranked_source = "matched"
    if quality_gate_enabled:
        score_threshold = float(getattr(cfg, "BUILD_INSTITUTION_MATCH_SCORE_MIN", 0.85))
        jw_threshold = float(getattr(cfg, "BUILD_INSTITUTION_ALIAS_JW_MIN", 0.92))
        university_clean_expr = _clean_institution_name_sql("matched.university_raw")
        quality_cte_sql = f"""
        ,
        matched_with_quality AS (
            SELECT
                matched.*,
                MAX(
                    CASE
                        WHEN a.alias_clean IS NULL OR {university_clean_expr} IS NULL THEN NULL
                        ELSE jaro_winkler_similarity({university_clean_expr}, a.alias_clean)
                    END
                ) AS institution_alias_jw_max
            FROM matched
            LEFT JOIN institution_match_aliases AS a
              ON CAST(matched.unitid AS BIGINT) = CAST(a.unitid AS BIGINT)
            GROUP BY ALL
        ),
        quality_filtered AS (
            SELECT *
            FROM matched_with_quality
            WHERE COALESCE(school_match_score, 0.0) >= {score_threshold}
               OR COALESCE(institution_alias_jw_max, 0.0) >= {jw_threshold}
        )
        """
        ranked_source = "quality_filtered"

    rsid_support_share = float(getattr(cfg, "BUILD_RSID_SUPPORT_MIN_SHARE", 0.05))
    rsid_support_min_count = int(getattr(cfg, "BUILD_RSID_SUPPORT_MIN_COUNT", 10))
    rsid_gate_enabled = bool(getattr(cfg, "BUILD_RSID_SUPPORT_GATE", False))
    if "rsid" in sample_cols:
        rsid_support_sql = f"""
            SELECT
                CAST(unitid AS BIGINT) AS unitid,
                TRY_CAST(rsid AS BIGINT) AS rsid,
                COUNT(*) AS rsid_unitid_match_count,
                SUM(COUNT(*)) OVER (PARTITION BY CAST(unitid AS BIGINT)) AS rsid_unitid_total_with_rsid
            FROM {sample_view}
            WHERE unitid IS NOT NULL
              AND TRY_CAST(rsid AS BIGINT) IS NOT NULL
            GROUP BY
                CAST(unitid AS BIGINT),
                TRY_CAST(rsid AS BIGINT)
        """
    else:
        rsid_support_sql = """
            SELECT
                NULL::BIGINT AS unitid,
                NULL::BIGINT AS rsid,
                0::BIGINT AS rsid_unitid_match_count,
                0::BIGINT AS rsid_unitid_total_with_rsid
            WHERE FALSE
        """

    rsid_cte_sql = f"""
        ,
        rsid_support AS (
            {rsid_support_sql}
        ),
        matched_with_rsid_support AS (
            SELECT
                src.*,
                sup.rsid_unitid_match_count,
                sup.rsid_unitid_total_with_rsid,
                CASE
                    WHEN sup.rsid_unitid_total_with_rsid > 0
                    THEN CAST(sup.rsid_unitid_match_count AS DOUBLE)
                         / CAST(sup.rsid_unitid_total_with_rsid AS DOUBLE)
                    ELSE NULL::DOUBLE
                END AS rsid_unitid_match_share,
                GREATEST(
                    {rsid_support_min_count},
                    CAST(CEIL(COALESCE(sup.rsid_unitid_total_with_rsid, 0) * {rsid_support_share}) AS BIGINT)
                ) AS rsid_unitid_required_count
            FROM {ranked_source} AS src
            LEFT JOIN rsid_support AS sup
              ON CAST(src.unitid AS BIGINT) = sup.unitid
             AND TRY_CAST(src.rsid AS BIGINT) = sup.rsid
        )
    """
    ranked_source = "matched_with_rsid_support"
    if rsid_gate_enabled:
        rsid_cte_sql += """
        ,
        rsid_support_filtered AS (
            SELECT *
            FROM matched_with_rsid_support
            WHERE rsid IS NOT NULL
              AND COALESCE(rsid_unitid_match_count, 0) >= rsid_unitid_required_count
        )
        """
        ranked_source = "rsid_support_filtered"

    out = con.sql(
        f"""
        WITH matched AS (
            SELECT
                {", ".join(matched_cols)}
            FROM {sample_view} AS s
            {join_sql}
            WHERE {where_sql}
        )
        {quality_cte_sql}
        {rsid_cte_sql},
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY {", ".join(match_partition_cols)}
                    ORDER BY
                        ABS(cohort_t),
                        CASE WHEN ed_enddate IS NULL THEN 1 ELSE 0 END,
                        ed_enddate DESC,
                        CASE WHEN education_number IS NULL THEN 1 ELSE 0 END,
                        education_number,
                        unitid,
                        {", ".join(extra_order_cols)}
                ) AS match_rank
            FROM {ranked_source}
        )
        SELECT
            user_id,
            unitid,
            education_number,
            ed_startdate,
            ed_enddate,
            grad_year,
            relabel_year,
            relabel_type,
            cohort_t,
            {treated_ind} AS treated_ind,
            pair_id,
            event_id,
            broad_pair_bin,
            degree_type,
            rsid,
            rsid_unitid_match_count,
            rsid_unitid_total_with_rsid,
            rsid_unitid_match_share,
            rsid_unitid_required_count
        FROM ranked
        WHERE match_rank = 1
        ORDER BY user_id, relabel_year, COALESCE(pair_id, -1), COALESCE(event_id, '')
        """
    ).df()
    if out.empty:
        return _empty_indiv_event_matches()
    return out


def _treated_events_for_matching(
    relabel_df: pd.DataFrame,
    control_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if _uses_generalized_events():
        if control_events is not None and not control_events.empty and "treated_unitid" in control_events.columns:
            treated_cols = [
                "treated_unitid",
                "relabel_year",
                "relabel_type",
                "pair_id",
                "event_id",
                "broad_pair_bin",
                "degree_type",
            ]
            optional_cols = [col for col in ("awlevel", "source_cip6", "target_cip6") if col in control_events.columns]
            treated_events = control_events[treated_cols + optional_cols].drop_duplicates().rename(
                columns={"treated_unitid": "unitid"}
            )
            return treated_events.reset_index(drop=True)
        return _generalized_treated_events(relabel_df)
    treated_events = relabel_df[relabel_df["event_flag"] == 1][
        ["unitid", "relabel_year", "relabel_type"]
    ].drop_duplicates()
    return _attach_event_ids(treated_events)


def step3_match_treated(
    con: ddb.DuckDBPyConnection,
    relabel_df: pd.DataFrame,
    sample_view: str,
    analysis_variant: str,
    testing_unitids: list[int] | None = None,
    control_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join the full-sample education sample to treated relabel events.
    Returns one deduplicated row per user × event identity.
    """
    t = time.time()
    print(
        f"\n── Step 3: Matching treated individuals [{analysis_variant}] "
        "────────────────"
    )

    treated_events = _treated_events_for_matching(relabel_df, control_events=control_events)
    if testing_unitids is not None:
        treated_events = treated_events[treated_events["unitid"].isin(testing_unitids)].copy()
        print(f"  [test] Restricting to {len(testing_unitids)} sample institutions")

    treated_indiv = _match_individuals_to_events(
        con,
        sample_view,
        treated_events,
        treated_ind=1,
        group_label=f"treated_{analysis_variant}",
    )
    n = len(treated_indiv)
    n_users = treated_indiv["user_id"].nunique() if not treated_indiv.empty else 0
    n_unitids = treated_indiv["unitid"].nunique() if not treated_indiv.empty else 0
    print(
        "  Treated individuals: "
        f"{n:,} (user × event) | {n_users:,} users | {n_unitids:,} schools"
    )
    print(f"  Step 3 done in {_elapsed(t)}")
    return treated_indiv


def build_control_events(
    con: ddb.DuckDBPyConnection,
    relabel_df: pd.DataFrame,
    testing_unitids: list[int] | None = None,
) -> pd.DataFrame:
    """Assign pseudo relabel years to matched never-treated control schools."""
    t = time.time()
    control_group = _control_group()
    control_label = control_group.replace("_", "-")
    print(f"\n── Control Setup: Matching {control_label} control schools ─────────────")

    if _uses_generalized_events():
        matched_pairs = generalized.match_treated_to_never_treated(
            con=con,
            relabel_panel=relabel_df,
            control_group=control_group,
        )
        if matched_pairs.empty:
            print("  Warning: no generalized matched control pairs found.")
            return pd.DataFrame(
                columns=[
                    "unitid",
                    "treated_unitid",
                    "relabel_year",
                    "relabel_type",
                    "pair_id",
                    "event_id",
                    "broad_pair_bin",
                    "degree_type",
                ]
            )
        if testing_unitids is not None:
            matched_pairs = matched_pairs[
                matched_pairs["treated_unitid"].isin(testing_unitids)
            ].copy()
            print(f"  [test] Restricted to {len(matched_pairs)} matched pairs")
        if matched_pairs.empty:
            print("  Warning: no generalized control pairs remain after test filter.")
            return pd.DataFrame(
                columns=[
                    "unitid",
                    "treated_unitid",
                    "relabel_year",
                    "relabel_type",
                    "pair_id",
                    "event_id",
                    "broad_pair_bin",
                    "degree_type",
                ]
            )
        control_events = matched_pairs.copy()
        control_events["unitid"] = pd.to_numeric(control_events["control_unitid"], errors="coerce").astype("Int64")
        control_events = _attach_event_ids(control_events, unitid_col="treated_unitid")
        print(f"  Matched control pairs: {len(control_events):,}")
        print(
            control_events[
                ["pair_id", "relabel_type", "relabel_year", "treated_unitid", "control_unitid"]
            ].head(10).to_string(index=False)
        )
        print(
            "  Control pseudo-events: "
            f"{len(control_events):,} rows | {control_events['unitid'].nunique():,} schools"
        )
        print(f"  Control setup done in {_elapsed(t)}")
        return control_events

    if control_group != generalized.CONTROL_GROUP_NEVER_TREATED:
        raise ValueError(
            f"control_group='{control_group}' requires event_source_mode='generalized_final_sample'. "
            "econ_v2 Revelio controls only support never_treated."
        )

    matched_pairs = v2._match_treated_to_untreated_cohorts(con=con, relabel_df=relabel_df)
    if matched_pairs.empty:
        print("  Warning: no matched control pairs found.")
        return pd.DataFrame(columns=["unitid", "relabel_year", "relabel_type", "event_id"])

    if testing_unitids is not None:
        matched_pairs = matched_pairs[
            matched_pairs["treated_unitid"].isin(testing_unitids)
        ].copy()
        print(f"  [test] Restricted to {len(matched_pairs)} matched pairs")
    if matched_pairs.empty:
        print("  Warning: no control pairs remain after test filter.")
        return pd.DataFrame(columns=["unitid", "relabel_year", "relabel_type", "event_id"])

    print(f"  Matched control pairs: {len(matched_pairs):,}")
    print(
        matched_pairs[
            ["relabel_type", "relabel_year", "treated_unitid", "control_unitid"]
        ].head(10).to_string(index=False)
    )

    control_events = matched_pairs.rename(columns={"control_unitid": "unitid"})[
        ["unitid", "relabel_year", "relabel_type"]
    ].drop_duplicates()
    control_events = _attach_event_ids(control_events)
    print(
        "  Control pseudo-events: "
        f"{len(control_events):,} rows | {control_events['unitid'].nunique():,} schools"
    )
    print(f"  Control setup done in {_elapsed(t)}")
    return control_events


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Build individual × horizon outcome panel
# ─────────────────────────────────────────────────────────────────────────────

def _latest_available_position_year(
    con: ddb.DuckDBPyConnection,
    pos_view: str,
    startdate_col: str,
    enddate_col: str | None,
) -> int:
    if enddate_col is None:
        query = f"""
            SELECT MAX(obs_year) AS latest_available_year
            FROM (
                SELECT EXTRACT(YEAR FROM TRY_CAST({startdate_col} AS DATE)) AS obs_year
                FROM {pos_view}
                WHERE TRY_CAST({startdate_col} AS DATE) IS NOT NULL
            )
        """
    else:
        query = f"""
            WITH years AS (
                SELECT EXTRACT(YEAR FROM TRY_CAST({startdate_col} AS DATE)) AS obs_year
                FROM {pos_view}
                WHERE TRY_CAST({startdate_col} AS DATE) IS NOT NULL
                UNION ALL
                SELECT EXTRACT(YEAR FROM TRY_CAST({enddate_col} AS DATE)) AS obs_year
                FROM {pos_view}
                WHERE TRY_CAST({enddate_col} AS DATE) IS NOT NULL
            )
            SELECT MAX(obs_year) AS latest_available_year
            FROM years
            WHERE obs_year IS NOT NULL
        """
    latest_available_year = con.sql(query).fetchone()[0]
    if latest_available_year is None:
        raise ValueError("Could not determine latest available year from Revelio positions.")
    return int(latest_available_year)


def step4_build_outcome_panel(
    con: ddb.DuckDBPyConnection,
    indiv_events: pd.DataFrame,
    group_label: str = "treated",
    analysis_variant: str | None = None,
) -> pd.DataFrame:
    """
    For each matched user-event row, compute outcomes at fixed horizons after graduation.

    Outcomes:
        in_us          : 1 if any active position in US in the evaluation year
        n_pos          : count of distinct post-graduation positions held up to and including the evaluation year
        n_employers    : count of distinct post-graduation employers up to and including the evaluation year
        avg_employer_tenure_years : mean cumulative post-graduation tenure per known employer through the evaluation year
        in_school      : 1 if any education spell overlaps the evaluation year
        salary_imputed : SUM(total_compensation * frac_of_year_active) in the evaluation year
        n_internship_positions : count of positions overlapping the education spell

    Returns long panel with columns:
        user_id, relabel_year, relabel_type, grad_year, cohort_t, horizon_years,
        target_year, eval_year, target_year_observed, used_latest_avail,
        treated_ind, analysis_variant, unitid, education_number,
        pair_id, event_id, broad_pair_bin, degree_type,
        in_us, n_pos, n_employers, avg_employer_tenure_years, in_school,
        salary_imputed, n_internship_positions
    """
    t = time.time()
    print(
        f"\n── Step 4: Building {group_label} outcome panel"
        + (f" [{analysis_variant}]" if analysis_variant else "")
        + " ──────────────────────"
    )

    if indiv_events.empty:
        print("  No individual-event rows supplied; returning empty panel.")
        return pd.DataFrame(
            columns=[
                "user_id",
                "relabel_year",
                "relabel_type",
                "grad_year",
                "cohort_t",
                "horizon_years",
                "target_year",
                "eval_year",
                "latest_available_year",
                "target_year_observed",
                "used_latest_avail",
                "treated_ind",
                "analysis_variant",
                "unitid",
                "education_number",
                "ed_startdate",
                "ed_enddate",
                "pair_id",
                "event_id",
                "broad_pair_bin",
                "degree_type",
                "rsid",
                "rsid_unitid_match_count",
                "rsid_unitid_total_with_rsid",
                "rsid_unitid_match_share",
                "rsid_unitid_required_count",
                "female_prob_raw",
                "origin_country_raw",
                "origin_country_bucket",
                "est_yob",
                "age_at_grad_raw",
                "age_at_grad",
                "age_missing_ind",
                "linkedin_last_education_date",
                "linkedin_last_position_date",
                "linkedin_last_activity_date",
                "linkedin_last_activity_year",
                "instsize_hd",
                "c21basic_lab",
                "in_us",
                "n_pos",
                "n_employers",
                "avg_employer_tenure_years",
                "in_school",
                "salary_imputed",
                "linkedin_active_through_target_year",
                "n_internship_positions",
            ]
        )

    _ensure_enrichment_views(con)

    pos_path = cfg.REV_POS_PARQUET
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"Revelio positions not found: {pos_path}")

    pos_path_sql = _escape_sql_literal(pos_path)
    con.sql(f"CREATE OR REPLACE TEMP VIEW rev_pos_raw AS SELECT * FROM read_parquet('{pos_path_sql}')")
    pos_cols = [r[0].lower() for r in con.sql("DESCRIBE rev_pos_raw").fetchall()]
    print(f"  rev_pos columns: {pos_cols}")

    # Resolve position column names
    country_col = next((c for c in pos_cols if c == "country"), None)
    startdate_col = next((c for c in pos_cols if c in {"startdate", "start_date", "position_startdate"}), None)
    enddate_col = next((c for c in pos_cols if c in {"enddate", "end_date", "position_enddate"}), None)
    rcid_col = next((c for c in pos_cols if c == "rcid"), None)
    comp_col = next((c for c in pos_cols if c in {"total_compensation", "compensation", "salary"}), None)

    if startdate_col is None:
        raise ValueError(f"Cannot find startdate column in rev_pos. Columns: {pos_cols}")
    if enddate_col is None:
        enddate_date_expr = "NULL::DATE"
        print("  Warning: no enddate column found; treating all positions as open-ended")
    else:
        enddate_date_expr = f"TRY_CAST(p.{enddate_col} AS DATE)"

    # Build compensation expression
    if comp_col:
        comp_expr = f"TRY_CAST(p.{comp_col} AS DOUBLE)"
    else:
        comp_expr = "NULL::DOUBLE"
        print("  Warning: no compensation column found; salary_imputed will be NULL")

    # Build country expression
    if country_col:
        in_us_expr = (
            f"CASE WHEN LOWER(TRIM(CAST(p.{country_col} AS VARCHAR))) = 'united states' "
            "THEN 1 ELSE 0 END"
        )
    else:
        in_us_expr = "0"
        print("  Warning: no country column found; in_us will be 0")

    # rcid expression for distinct position count
    if rcid_col:
        rcid_expr = f"p.{rcid_col}"
    else:
        rcid_expr = "NULL"

    educ_path = cfg.REV_EDUC_CLEAN_LONG_PARQUET or cfg.REV_EDUC_LONG_PARQUET
    if educ_path and os.path.exists(educ_path):
        educ_path_sql = _escape_sql_literal(educ_path)
        con.sql(f"CREATE OR REPLACE TEMP VIEW rev_educ_spells_raw AS SELECT * FROM read_parquet('{educ_path_sql}')")
        educ_cols = [r[0] for r in con.sql("DESCRIBE rev_educ_spells_raw").fetchall()]
        educ_user_col = _first_present_lower(educ_cols, ["user_id"])
        educ_start_col = _first_present_lower(educ_cols, ["ed_startdate", "startdate", "start_date"])
        educ_end_col = _first_present_lower(educ_cols, ["ed_enddate", "enddate", "end_date"])
        if educ_user_col is None:
            raise ValueError("rev_educ_long parquet must contain user_id")
        if educ_start_col is None and educ_end_col is None:
            print("  Warning: no education date columns found; in_school will be 0")
            _create_empty_temp_view(
                con,
                "rev_educ_spells",
                [("user_id", "BIGINT"), ("school_startdate", "DATE"), ("school_enddate", "DATE")],
            )
        else:
            start_expr = f"TRY_CAST({educ_start_col} AS DATE)" if educ_start_col is not None else "NULL::DATE"
            end_expr = f"TRY_CAST({educ_end_col} AS DATE)" if educ_end_col is not None else "NULL::DATE"
            con.sql(
                f"""
                CREATE OR REPLACE TEMP VIEW rev_educ_spells AS
                SELECT
                    CAST({educ_user_col} AS BIGINT) AS user_id,
                    COALESCE({start_expr}, {end_expr}) AS school_startdate,
                    COALESCE({end_expr}, {start_expr}) AS school_enddate
                FROM rev_educ_spells_raw
                WHERE {educ_user_col} IS NOT NULL
                  AND COALESCE({start_expr}, {end_expr}) IS NOT NULL
                  AND COALESCE({end_expr}, {start_expr}) IS NOT NULL
                """
            )
    else:
        print(f"  Warning: Revelio education-long parquet not found: {educ_path}; in_school will be 0")
        _create_empty_temp_view(
            con,
            "rev_educ_spells",
            [("user_id", "BIGINT"), ("school_startdate", "DATE"), ("school_enddate", "DATE")],
        )

    horizons = _analysis_horizons()
    latest_available_year = _latest_available_position_year(
        con,
        "rev_pos_raw",
        startdate_col,
        enddate_col,
    )
    print(f"  horizons after graduation: {horizons}")
    print(f"  latest available position year: {latest_available_year}")

    min_duration_days = max(1, int(getattr(cfg, "BUILD_MIN_POS_DURATION_DAYS", 1)))
    analysis_variant_sql = _escape_sql_literal(analysis_variant or "unspecified")
    horizons_sql = ",\n            ".join(f"({int(h)})" for h in horizons)
    events_view = f"indiv_events_{_variant_slug(group_label)}"

    indiv_events = indiv_events.copy()
    for missing_date_col in ("ed_startdate", "ed_enddate"):
        if missing_date_col not in indiv_events.columns:
            indiv_events[missing_date_col] = pd.NaT
    con.register("indiv_events_py", indiv_events)
    con.sql(f"CREATE OR REPLACE TEMP VIEW {events_view} AS SELECT * FROM indiv_events_py")

    # Evaluate outcomes at fixed horizons after graduation, not at years since relabel.
    # Future target years are flagged so downstream plots / regressions can drop them.
    outcome_panel = con.sql(
        f"""
        WITH horizons (horizon_years) AS (
            VALUES
            {horizons_sql}
        ),
        indiv_horizons AS (
            SELECT
                ie.user_id,
                ie.relabel_year,
                ie.relabel_type,
                ie.grad_year,
                ie.cohort_t,
                h.horizon_years,
                ie.grad_year + h.horizon_years AS target_year,
                CASE
                    WHEN {1 if getattr(cfg, "BUILD_CAP_TO_LATEST_AVAILABLE_YEAR", True) else 0} = 1
                    THEN LEAST(ie.grad_year + h.horizon_years, {latest_available_year})
                    ELSE ie.grad_year + h.horizon_years
                END AS eval_year,
                {latest_available_year} AS latest_available_year,
                CASE
                    WHEN ie.grad_year + h.horizon_years <= {latest_available_year} THEN 1
                    ELSE 0
                END AS target_year_observed,
                CASE
                    WHEN {1 if getattr(cfg, "BUILD_CAP_TO_LATEST_AVAILABLE_YEAR", True) else 0} = 1
                     AND ie.grad_year + h.horizon_years > {latest_available_year} THEN 1
                    ELSE 0
                END AS used_latest_avail,
                ie.treated_ind,
                '{analysis_variant_sql}' AS analysis_variant,
                COALESCE(CAST(ie.unitid AS BIGINT), -1) AS unitid,
                COALESCE(CAST(ie.education_number AS BIGINT), -1) AS education_number,
                TRY_CAST(ie.ed_startdate AS DATE) AS ed_startdate,
                TRY_CAST(ie.ed_enddate AS DATE) AS ed_enddate,
                TRY_CAST(ie.pair_id AS BIGINT) AS pair_id,
                CAST(ie.event_id AS VARCHAR) AS event_id,
                CAST(ie.broad_pair_bin AS VARCHAR) AS broad_pair_bin,
                CAST(ie.degree_type AS VARCHAR) AS degree_type,
                TRY_CAST(ie.rsid AS BIGINT) AS rsid,
                TRY_CAST(ie.rsid_unitid_match_count AS BIGINT) AS rsid_unitid_match_count,
                TRY_CAST(ie.rsid_unitid_total_with_rsid AS BIGINT) AS rsid_unitid_total_with_rsid,
                TRY_CAST(ie.rsid_unitid_match_share AS DOUBLE) AS rsid_unitid_match_share,
                TRY_CAST(ie.rsid_unitid_required_count AS BIGINT) AS rsid_unitid_required_count
            FROM {events_view} ie
            CROSS JOIN horizons h
        ),
        pos_base AS (
            SELECT
                ih.user_id,
                ih.relabel_year,
                ih.relabel_type,
                ih.grad_year,
                ih.cohort_t,
                ih.horizon_years,
                ih.target_year,
                ih.eval_year,
                ih.latest_available_year,
                ih.target_year_observed,
                ih.used_latest_avail,
                ih.treated_ind,
                ih.analysis_variant,
                ih.unitid,
                ih.education_number,
                ih.ed_startdate,
                ih.ed_enddate,
                ih.pair_id,
                ih.event_id,
                ih.broad_pair_bin,
                ih.degree_type,
                ih.rsid,
                ih.rsid_unitid_match_count,
                ih.rsid_unitid_total_with_rsid,
                ih.rsid_unitid_match_share,
                ih.rsid_unitid_required_count,
                COALESCE(ih.ed_startdate, MAKE_DATE(ih.grad_year - 4, 1, 1)) AS education_start_date,
                COALESCE(ih.ed_enddate, MAKE_DATE(ih.grad_year, 12, 31)) AS education_end_date,
                {in_us_expr} AS is_us_pos,
                COALESCE({rcid_expr}, -1) AS pos_rcid,
                TRY_CAST(p.{startdate_col} AS DATE) AS pos_startdate,
                COALESCE({enddate_date_expr}, DATE '9999-12-31') AS pos_enddate,
                {comp_expr} AS comp
            FROM indiv_horizons ih
            JOIN rev_pos_raw p
              ON p.user_id = ih.user_id
            WHERE TRY_CAST(p.{startdate_col} AS DATE) IS NOT NULL
        ),
        pos_eval_year_active AS (
            SELECT
                *,
                GREATEST(0,
                    (LEAST(pos_enddate, MAKE_DATE(eval_year, 12, 31))
                     - GREATEST(pos_startdate, MAKE_DATE(eval_year, 1, 1)) + 1)
                ) AS overlap_days,
                GREATEST(0,
                    (LEAST(pos_enddate, MAKE_DATE(eval_year, 12, 31))
                     - GREATEST(pos_startdate, MAKE_DATE(eval_year, 1, 1)) + 1)
                ) / 365.0 AS frac_year
            FROM pos_base
            WHERE pos_startdate <= MAKE_DATE(eval_year, 12, 31)
              AND pos_enddate >= MAKE_DATE(eval_year, 1, 1)
        ),
        pos_postgrad_to_eval AS (
            SELECT
                *,
                GREATEST(0,
                    (LEAST(pos_enddate, MAKE_DATE(eval_year, 12, 31))
                     - GREATEST(pos_startdate, MAKE_DATE(grad_year, 1, 1)) + 1)
                ) AS postgrad_overlap_days
            FROM pos_base
            WHERE pos_startdate <= MAKE_DATE(eval_year, 12, 31)
              AND pos_enddate >= MAKE_DATE(grad_year, 1, 1)
        ),
        pos_education_spell AS (
            SELECT
                *,
                GREATEST(0,
                    (LEAST(pos_enddate, education_end_date)
                     - GREATEST(pos_startdate, education_start_date) + 1)
                ) AS education_overlap_days
            FROM pos_base
            WHERE pos_startdate <= education_end_date
              AND pos_enddate >= education_start_date
        ),
        school_at_horizon_agg AS (
            SELECT
                ih.user_id,
                ih.relabel_year,
                ih.relabel_type,
                ih.grad_year,
                ih.cohort_t,
                ih.horizon_years,
                ih.target_year,
                ih.eval_year,
                ih.latest_available_year,
                ih.target_year_observed,
                ih.used_latest_avail,
                ih.treated_ind,
                ih.analysis_variant,
                ih.unitid,
                ih.education_number,
                ih.pair_id,
                ih.event_id,
                ih.broad_pair_bin,
                ih.degree_type,
                ih.rsid,
                ih.rsid_unitid_match_count,
                ih.rsid_unitid_total_with_rsid,
                ih.rsid_unitid_match_share,
                ih.rsid_unitid_required_count,
                1 AS in_school
            FROM indiv_horizons ih
            JOIN rev_educ_spells e
              ON e.user_id = ih.user_id
            WHERE e.school_startdate <= MAKE_DATE(ih.eval_year, 12, 31)
              AND e.school_enddate >= MAKE_DATE(ih.eval_year, 1, 1)
            GROUP BY ih.user_id, ih.relabel_year, ih.relabel_type, ih.grad_year, ih.cohort_t,
                     ih.horizon_years, ih.target_year, ih.eval_year, ih.latest_available_year,
                     ih.target_year_observed, ih.used_latest_avail, ih.treated_ind,
                     ih.analysis_variant, ih.unitid, ih.education_number,
                     ih.pair_id, ih.event_id, ih.broad_pair_bin, ih.degree_type,
                     ih.rsid, ih.rsid_unitid_match_count, ih.rsid_unitid_total_with_rsid,
                     ih.rsid_unitid_match_share, ih.rsid_unitid_required_count
        ),
        yearly_agg AS (
            SELECT
                user_id,
                relabel_year,
                relabel_type,
                grad_year,
                cohort_t,
                horizon_years,
                target_year,
                eval_year,
                latest_available_year,
                target_year_observed,
                used_latest_avail,
                treated_ind,
                analysis_variant,
                unitid,
                education_number,
                pair_id,
                event_id,
                broad_pair_bin,
                degree_type,
                rsid,
                rsid_unitid_match_count,
                rsid_unitid_total_with_rsid,
                rsid_unitid_match_share,
                rsid_unitid_required_count,
                MAX(is_us_pos) AS in_us,
                SUM(COALESCE(comp, 0) * frac_year) AS salary_imputed_raw
            FROM pos_eval_year_active
            WHERE overlap_days >= {min_duration_days}
            GROUP BY user_id, relabel_year, relabel_type, grad_year, cohort_t,
                     horizon_years, target_year, eval_year, latest_available_year,
                     target_year_observed, used_latest_avail, treated_ind,
                     analysis_variant, unitid, education_number,
                     pair_id, event_id, broad_pair_bin, degree_type,
                     rsid, rsid_unitid_match_count, rsid_unitid_total_with_rsid,
                     rsid_unitid_match_share, rsid_unitid_required_count
        ),
        postgrad_position_agg AS (
            SELECT
                user_id,
                relabel_year,
                relabel_type,
                grad_year,
                cohort_t,
                horizon_years,
                target_year,
                eval_year,
                latest_available_year,
                target_year_observed,
                used_latest_avail,
                treated_ind,
                analysis_variant,
                unitid,
                education_number,
                pair_id,
                event_id,
                broad_pair_bin,
                degree_type,
                rsid,
                rsid_unitid_match_count,
                rsid_unitid_total_with_rsid,
                rsid_unitid_match_share,
                rsid_unitid_required_count,
                COUNT(DISTINCT (pos_rcid, pos_startdate)) AS n_pos,
                COUNT(DISTINCT CASE WHEN pos_rcid != -1 THEN pos_rcid END) AS n_employers
            FROM pos_postgrad_to_eval
            WHERE postgrad_overlap_days >= {min_duration_days}
            GROUP BY user_id, relabel_year, relabel_type, grad_year, cohort_t,
                     horizon_years, target_year, eval_year, latest_available_year,
                     target_year_observed, used_latest_avail, treated_ind,
                     analysis_variant, unitid, education_number,
                     pair_id, event_id, broad_pair_bin, degree_type,
                     rsid, rsid_unitid_match_count, rsid_unitid_total_with_rsid,
                     rsid_unitid_match_share, rsid_unitid_required_count
        ),
        postgrad_employer_spells AS (
            SELECT
                user_id,
                relabel_year,
                relabel_type,
                grad_year,
                cohort_t,
                horizon_years,
                target_year,
                eval_year,
                latest_available_year,
                target_year_observed,
                used_latest_avail,
                treated_ind,
                analysis_variant,
                unitid,
                education_number,
                pair_id,
                event_id,
                broad_pair_bin,
                degree_type,
                rsid,
                rsid_unitid_match_count,
                rsid_unitid_total_with_rsid,
                rsid_unitid_match_share,
                rsid_unitid_required_count,
                pos_rcid,
                SUM(postgrad_overlap_days) AS employer_tenure_days
            FROM pos_postgrad_to_eval
            WHERE postgrad_overlap_days >= {min_duration_days}
              AND pos_rcid != -1
            GROUP BY user_id, relabel_year, relabel_type, grad_year, cohort_t,
                     horizon_years, target_year, eval_year, latest_available_year,
                     target_year_observed, used_latest_avail, treated_ind,
                     analysis_variant, unitid, education_number,
                     pair_id, event_id, broad_pair_bin, degree_type,
                     rsid, rsid_unitid_match_count, rsid_unitid_total_with_rsid,
                     rsid_unitid_match_share, rsid_unitid_required_count, pos_rcid
        ),
        postgrad_tenure_agg AS (
            SELECT
                user_id,
                relabel_year,
                relabel_type,
                grad_year,
                cohort_t,
                horizon_years,
                target_year,
                eval_year,
                latest_available_year,
                target_year_observed,
                used_latest_avail,
                treated_ind,
                analysis_variant,
                unitid,
                education_number,
                pair_id,
                event_id,
                broad_pair_bin,
                degree_type,
                rsid,
                rsid_unitid_match_count,
                rsid_unitid_total_with_rsid,
                rsid_unitid_match_share,
                rsid_unitid_required_count,
                AVG(employer_tenure_days / 365.0) AS avg_employer_tenure_years
            FROM postgrad_employer_spells
            GROUP BY user_id, relabel_year, relabel_type, grad_year, cohort_t,
                     horizon_years, target_year, eval_year, latest_available_year,
                     target_year_observed, used_latest_avail, treated_ind,
                     analysis_variant, unitid, education_number,
                     pair_id, event_id, broad_pair_bin, degree_type,
                     rsid, rsid_unitid_match_count, rsid_unitid_total_with_rsid,
                     rsid_unitid_match_share, rsid_unitid_required_count
        ),
        postgrad_agg AS (
            SELECT
                p.user_id,
                p.relabel_year,
                p.relabel_type,
                p.grad_year,
                p.cohort_t,
                p.horizon_years,
                p.target_year,
                p.eval_year,
                p.latest_available_year,
                p.target_year_observed,
                p.used_latest_avail,
                p.treated_ind,
                p.analysis_variant,
                p.unitid,
                p.education_number,
                p.pair_id,
                p.event_id,
                p.broad_pair_bin,
                p.degree_type,
                p.rsid,
                p.rsid_unitid_match_count,
                p.rsid_unitid_total_with_rsid,
                p.rsid_unitid_match_share,
                p.rsid_unitid_required_count,
                p.n_pos,
                p.n_employers,
                t.avg_employer_tenure_years
            FROM postgrad_position_agg p
            LEFT JOIN postgrad_tenure_agg t
              ON  t.user_id             = p.user_id
              AND t.relabel_year        = p.relabel_year
              AND t.relabel_type        = p.relabel_type
              AND t.analysis_variant    = p.analysis_variant
              AND t.unitid              = p.unitid
              AND t.education_number    = p.education_number
              AND COALESCE(t.pair_id, -1) = COALESCE(p.pair_id, -1)
              AND COALESCE(t.event_id, '') = COALESCE(p.event_id, '')
              AND COALESCE(t.broad_pair_bin, '') = COALESCE(p.broad_pair_bin, '')
              AND COALESCE(t.degree_type, '') = COALESCE(p.degree_type, '')
              AND COALESCE(t.rsid, -1) = COALESCE(p.rsid, -1)
              AND t.horizon_years       = p.horizon_years
        ),
        internship_agg AS (
            SELECT
                user_id,
                relabel_year,
                relabel_type,
                grad_year,
                cohort_t,
                treated_ind,
                analysis_variant,
                unitid,
                education_number,
                pair_id,
                event_id,
                broad_pair_bin,
                degree_type,
                rsid,
                rsid_unitid_match_count,
                rsid_unitid_total_with_rsid,
                rsid_unitid_match_share,
                rsid_unitid_required_count,
                COUNT(DISTINCT (pos_rcid, pos_startdate)) AS n_internship_positions
            FROM pos_education_spell
            WHERE education_overlap_days >= {min_duration_days}
            GROUP BY user_id, relabel_year, relabel_type, grad_year, cohort_t,
                     treated_ind, analysis_variant, unitid, education_number,
                     pair_id, event_id, broad_pair_bin, degree_type,
                     rsid, rsid_unitid_match_count, rsid_unitid_total_with_rsid,
                     rsid_unitid_match_share, rsid_unitid_required_count
        ),
        agg AS (
            SELECT
                COALESCE(y.user_id, p.user_id) AS user_id,
                COALESCE(y.relabel_year, p.relabel_year) AS relabel_year,
                COALESCE(y.relabel_type, p.relabel_type) AS relabel_type,
                COALESCE(y.grad_year, p.grad_year) AS grad_year,
                COALESCE(y.cohort_t, p.cohort_t) AS cohort_t,
                COALESCE(y.horizon_years, p.horizon_years) AS horizon_years,
                COALESCE(y.target_year, p.target_year) AS target_year,
                COALESCE(y.eval_year, p.eval_year) AS eval_year,
                COALESCE(y.latest_available_year, p.latest_available_year) AS latest_available_year,
                COALESCE(y.target_year_observed, p.target_year_observed) AS target_year_observed,
                COALESCE(y.used_latest_avail, p.used_latest_avail) AS used_latest_avail,
                COALESCE(y.treated_ind, p.treated_ind) AS treated_ind,
                COALESCE(y.analysis_variant, p.analysis_variant) AS analysis_variant,
                COALESCE(y.unitid, p.unitid) AS unitid,
                COALESCE(y.education_number, p.education_number) AS education_number,
                COALESCE(y.pair_id, p.pair_id) AS pair_id,
                COALESCE(y.event_id, p.event_id) AS event_id,
                COALESCE(y.broad_pair_bin, p.broad_pair_bin) AS broad_pair_bin,
                COALESCE(y.degree_type, p.degree_type) AS degree_type,
                COALESCE(y.rsid, p.rsid) AS rsid,
                COALESCE(y.rsid_unitid_match_count, p.rsid_unitid_match_count) AS rsid_unitid_match_count,
                COALESCE(y.rsid_unitid_total_with_rsid, p.rsid_unitid_total_with_rsid) AS rsid_unitid_total_with_rsid,
                COALESCE(y.rsid_unitid_match_share, p.rsid_unitid_match_share) AS rsid_unitid_match_share,
                COALESCE(y.rsid_unitid_required_count, p.rsid_unitid_required_count) AS rsid_unitid_required_count,
                y.in_us,
                p.n_pos,
                p.n_employers,
                p.avg_employer_tenure_years,
                y.salary_imputed_raw
            FROM yearly_agg y
            FULL OUTER JOIN postgrad_agg p
              ON  p.user_id             = y.user_id
              AND p.relabel_year        = y.relabel_year
              AND p.relabel_type        = y.relabel_type
              AND p.analysis_variant    = y.analysis_variant
              AND p.unitid              = y.unitid
              AND p.education_number    = y.education_number
              AND COALESCE(p.pair_id, -1) = COALESCE(y.pair_id, -1)
              AND COALESCE(p.event_id, '') = COALESCE(y.event_id, '')
              AND COALESCE(p.broad_pair_bin, '') = COALESCE(y.broad_pair_bin, '')
              AND COALESCE(p.degree_type, '') = COALESCE(y.degree_type, '')
              AND COALESCE(p.rsid, -1) = COALESCE(y.rsid, -1)
              AND p.horizon_years       = y.horizon_years
        ),
        full_panel AS (
            SELECT
                ih.user_id,
                ih.relabel_year,
                ih.relabel_type,
                ih.grad_year,
                ih.cohort_t,
                ih.horizon_years,
                ih.target_year,
                ih.eval_year,
                ih.latest_available_year,
                ih.target_year_observed,
                ih.used_latest_avail,
                ih.treated_ind,
                ih.analysis_variant,
                ih.unitid,
                ih.education_number,
                ih.pair_id,
                ih.event_id,
                ih.broad_pair_bin,
                ih.degree_type,
                ih.rsid,
                ih.rsid_unitid_match_count,
                ih.rsid_unitid_total_with_rsid,
                ih.rsid_unitid_match_share,
                ih.rsid_unitid_required_count,
                COALESCE(uf.female_prob_raw, 0.5) AS female_prob_raw,
                COALESCE(NULLIF(TRIM(CAST(uf.origin_country_raw AS VARCHAR)), ''), 'Unknown') AS origin_country_raw,
                TRY_CAST(uf.est_yob AS INTEGER) AS est_yob,
                CASE
                    WHEN TRY_CAST(uf.est_yob AS INTEGER) IS NULL THEN NULL::DOUBLE
                    WHEN ih.grad_year - TRY_CAST(uf.est_yob AS INTEGER) BETWEEN 15 AND 80
                    THEN CAST(ih.grad_year - TRY_CAST(uf.est_yob AS INTEGER) AS DOUBLE)
                    ELSE NULL::DOUBLE
                END AS age_at_grad_raw,
                uf.linkedin_last_education_date,
                uf.linkedin_last_position_date,
                uf.linkedin_last_activity_date,
                uf.linkedin_last_activity_year,
                COALESCE(CAST(syc.instsize_hd AS VARCHAR), 'Unknown') AS instsize_hd,
                COALESCE(CAST(syc.c21basic_lab AS VARCHAR), 'Unknown') AS c21basic_lab,
                CASE
                    WHEN uf.linkedin_last_activity_date IS NOT NULL
                     AND uf.linkedin_last_activity_date >= MAKE_DATE(ih.target_year, 12, 31) THEN 1
                    ELSE 0
                END AS linkedin_active_through_target_year,
                COALESCE(a.in_us, 0)             AS in_us,
                COALESCE(a.n_pos, 0)             AS n_pos,
                COALESCE(a.n_employers, 0)       AS n_employers,
                COALESCE(a.avg_employer_tenure_years, 0) AS avg_employer_tenure_years,
                COALESCE(s.in_school, 0) AS in_school,
                COALESCE(i.n_internship_positions, 0) AS n_internship_positions,
                CASE
                    WHEN a.salary_imputed_raw IS NOT NULL AND a.salary_imputed_raw > 0
                    THEN a.salary_imputed_raw
                    ELSE NULL
                END AS salary_imputed
            FROM indiv_horizons ih
            LEFT JOIN agg a
              ON  a.user_id             = ih.user_id
              AND a.relabel_year        = ih.relabel_year
              AND a.relabel_type        = ih.relabel_type
              AND a.analysis_variant    = ih.analysis_variant
              AND a.unitid              = ih.unitid
              AND a.education_number    = ih.education_number
              AND COALESCE(a.pair_id, -1) = COALESCE(ih.pair_id, -1)
              AND COALESCE(a.event_id, '') = COALESCE(ih.event_id, '')
              AND COALESCE(a.broad_pair_bin, '') = COALESCE(ih.broad_pair_bin, '')
              AND COALESCE(a.degree_type, '') = COALESCE(ih.degree_type, '')
              AND COALESCE(a.rsid, -1) = COALESCE(ih.rsid, -1)
              AND a.horizon_years       = ih.horizon_years
            LEFT JOIN internship_agg i
              ON  i.user_id             = ih.user_id
              AND i.relabel_year        = ih.relabel_year
              AND i.relabel_type        = ih.relabel_type
              AND i.analysis_variant    = ih.analysis_variant
              AND i.unitid              = ih.unitid
              AND i.education_number    = ih.education_number
              AND COALESCE(i.pair_id, -1) = COALESCE(ih.pair_id, -1)
              AND COALESCE(i.event_id, '') = COALESCE(ih.event_id, '')
              AND COALESCE(i.broad_pair_bin, '') = COALESCE(ih.broad_pair_bin, '')
              AND COALESCE(i.degree_type, '') = COALESCE(ih.degree_type, '')
              AND COALESCE(i.rsid, -1) = COALESCE(ih.rsid, -1)
            LEFT JOIN school_at_horizon_agg s
              ON  s.user_id             = ih.user_id
              AND s.relabel_year        = ih.relabel_year
              AND s.relabel_type        = ih.relabel_type
              AND s.analysis_variant    = ih.analysis_variant
              AND s.unitid              = ih.unitid
              AND s.education_number    = ih.education_number
              AND COALESCE(s.pair_id, -1) = COALESCE(ih.pair_id, -1)
              AND COALESCE(s.event_id, '') = COALESCE(ih.event_id, '')
              AND COALESCE(s.broad_pair_bin, '') = COALESCE(ih.broad_pair_bin, '')
              AND COALESCE(s.degree_type, '') = COALESCE(ih.degree_type, '')
              AND COALESCE(s.rsid, -1) = COALESCE(ih.rsid, -1)
              AND s.horizon_years       = ih.horizon_years
            LEFT JOIN revelio_user_features uf
              ON uf.user_id = ih.user_id
            LEFT JOIN school_year_controls syc
              ON syc.unitid = ih.unitid
             AND syc.grad_year = ih.grad_year
        )
        SELECT * FROM full_panel
        ORDER BY user_id, relabel_year, treated_ind, COALESCE(pair_id, -1), COALESCE(event_id, ''), unitid, horizon_years
        """
    ).df()

    n = len(outcome_panel)
    n_users = outcome_panel["user_id"].nunique()
    print(f"  Outcome panel ({group_label}): {n:,} rows | {n_users:,} users")
    observed_panel = (
        outcome_panel[outcome_panel["target_year_observed"] == 1].copy()
        if "target_year_observed" in outcome_panel.columns
        else outcome_panel
    )
    print(f"  observed rows:      {len(observed_panel):,}")
    print(f"  in_us mean:         {observed_panel['in_us'].mean():.3f}" if not observed_panel.empty else "  in_us mean:         nan")
    print(f"  n_pos mean:         {observed_panel['n_pos'].mean():.2f}" if not observed_panel.empty else "  n_pos mean:         nan")
    print(f"  n_employers mean:   {observed_panel['n_employers'].mean():.2f}" if not observed_panel.empty else "  n_employers mean:   nan")
    print(f"  avg employer tenure mean: {observed_panel['avg_employer_tenure_years'].mean():.2f}" if not observed_panel.empty else "  avg employer tenure mean: nan")
    print(f"  in_school mean:     {observed_panel['in_school'].mean():.3f}" if not observed_panel.empty else "  in_school mean:     nan")
    print(f"  education-spell positions mean: {observed_panel['n_internship_positions'].mean():.2f}" if not observed_panel.empty else "  education-spell positions mean: nan")
    print(
        f"  linkedin_active mean: {observed_panel['linkedin_active_through_target_year'].mean():.3f}"
        if not observed_panel.empty else
        "  linkedin_active mean: nan"
    )
    sal = observed_panel["salary_imputed"].dropna() if not observed_panel.empty else pd.Series(dtype=float)
    if len(sal) > 0:
        print(f"  salary_imputed mean: {sal.mean():,.0f}  (non-null: {len(sal):,} rows)")
    else:
        print("  salary_imputed: all NULL (no compensation data in positions)")
    print(f"  Step 4 done in {_elapsed(t)}")
    return outcome_panel


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Event study aggregation + plots (treated only)
# ─────────────────────────────────────────────────────────────────────────────

def step5_event_study_plots(
    panel: pd.DataFrame,
    label: str = "treated",
    analysis_variant: str | None = None,
) -> None:
    """Aggregate by cohort_t and plot one figure per outcome."""
    t = time.time()
    print(
        f"\n── Step 5: Event study plots ({label})"
        + (f" [{analysis_variant}]" if analysis_variant else "")
        + " ──────────────────────────────"
    )

    agg = _agg_cohort_time(panel)
    window = cfg.BUILD_EVENT_WINDOW
    agg = agg[agg["cohort_t"].between(-window, window)]
    if agg.empty:
        print("  No observed outcome rows remain after cohort-window filter.")
        return

    horizons = sorted(agg["horizon_years"].dropna().astype(int).unique().tolist())
    palette = [llstyle.color(idx) for idx in range(max(1, len(horizons)))]
    if "target_year_observed" in panel.columns:
        n_dropped = int((pd.to_numeric(panel["target_year_observed"], errors="coerce") == 0).sum())
        if n_dropped:
            print(f"  dropped {n_dropped:,} future target-year rows before aggregation")

    for outcome in OUTCOMES:
        mean_col = f"{outcome}_mean"
        se_col = f"{outcome}_se"
        if mean_col not in agg.columns:
            print(f"  Skipping {outcome} (not in panel)")
            continue

        fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
        for color, (horizon, grp) in zip(palette, agg.groupby("horizon_years", sort=True)):
            line_df = _coerce_plot_frame(grp, ["cohort_t", mean_col], sort_col="cohort_t")
            if line_df.empty:
                continue
            ax.plot(
                line_df["cohort_t"].to_numpy(dtype=float),
                line_df[mean_col].to_numpy(dtype=float),
                marker="o",
                markersize=EVENT_PLOT_MARKER_SIZE,
                linewidth=2,
                color=color,
                label=f"{int(horizon)} years after grad",
            )
            valid_ci = grp[se_col].notna() & (pd.to_numeric(grp[se_col], errors="coerce") > 0)
            if valid_ci.any():
                ci_df = _coerce_plot_frame(
                    grp.loc[valid_ci],
                    ["cohort_t", mean_col, se_col],
                    sort_col="cohort_t",
                )
                ax.fill_between(
                    ci_df["cohort_t"].to_numpy(dtype=float),
                    ci_df[mean_col].to_numpy(dtype=float) - 1.96 * ci_df[se_col].to_numpy(dtype=float),
                    ci_df[mean_col].to_numpy(dtype=float) + 1.96 * ci_df[se_col].to_numpy(dtype=float),
                    alpha=0.20,
                    color=color,
                )
        ax.axvline(x=DI_D_EVENT_LINE_X, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel(
            "Graduation year relative to relabel event",
            fontsize=DI_D_PLOT_FONT_SIZE,
        )
        ax.set_ylabel(
            OUTCOME_LABELS.get(outcome, outcome),
            fontsize=DI_D_PLOT_FONT_SIZE,
        )
        ax.set_title("")
        if len(horizons) > 1:
            llstyle.right_legend(ax)
        _save_and_show(fig, f"{outcome}_event_study_{label}", analysis_variant=analysis_variant)
        print(f"  n per horizon × cohort_t:\n{agg[['horizon_years', 'cohort_t', 'n']].to_string(index=False)}")

    print(f"  Step 5 done in {_elapsed(t)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Control group (never-treated econ institutions)
# ─────────────────────────────────────────────────────────────────────────────

def step6_control_group(
    con: ddb.DuckDBPyConnection,
    sample_view: str,
    control_events: pd.DataFrame,
    analysis_variant: str,
) -> pd.DataFrame:
    """
    Build never-treated control individuals from the same full-sample education sample.
    Returns control outcome panel (treated_ind=0).
    """
    t = time.time()
    print(f"\n── Step 6: Building control group [{analysis_variant}] ───────────────")

    if control_events.empty:
        print("  Warning: control pseudo-events are empty; skipping control group.")
        return pd.DataFrame()

    control_indiv = _match_individuals_to_events(
        con,
        sample_view,
        control_events,
        treated_ind=0,
        group_label=f"control_{analysis_variant}",
    )

    if control_indiv.empty:
        print("  Warning: no Revelio individuals found at control institutions.")
        return pd.DataFrame()

    n = len(control_indiv)
    n_users = control_indiv["user_id"].nunique()
    n_unitids = control_indiv["unitid"].nunique()
    print(
        "  Control individuals: "
        f"{n:,} (user × event) | {n_users:,} users | {n_unitids:,} schools"
    )

    control_panel = step4_build_outcome_panel(
        con,
        control_indiv,
        group_label=f"control_{_variant_slug(analysis_variant)}",
        analysis_variant=analysis_variant,
    )
    print(f"  Step 6 done in {_elapsed(t)}")
    return control_panel


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 – Treated vs. control event study plots
# ─────────────────────────────────────────────────────────────────────────────

def step7_treated_vs_control_plots(
    treated_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    analysis_variant: str | None = None,
) -> None:
    """Plot treated and control cohort series together for each outcome."""
    t = time.time()
    print(
        "\n── Step 7: Treated vs. control event study plots"
        + (f" [{analysis_variant}]" if analysis_variant else "")
        + " ─────────────────────"
    )

    if control_panel.empty:
        print("  Control panel is empty; skipping treated-vs-control plots.")
        return

    window = cfg.BUILD_EVENT_WINDOW

    treated_agg = _agg_cohort_time(treated_panel)
    treated_agg = treated_agg[treated_agg["cohort_t"].between(-window, window)].copy()
    treated_label = (
        "Treated (Matched relabel events)"
        if _uses_generalized_events()
        else "Treated (Econ→Econometrics)"
    )
    control_label = (
        "Control (Matched never-treated schools)"
        if _uses_generalized_events()
        else "Control (Never-treated Econ)"
    )
    treated_agg["series"] = treated_label

    ctrl_agg = _agg_cohort_time(control_panel)
    ctrl_agg = ctrl_agg[ctrl_agg["cohort_t"].between(-window, window)].copy()
    ctrl_agg["series"] = control_label
    if treated_agg.empty or ctrl_agg.empty:
        print("  No overlapping observed rows remain after cohort-window filter.")
        return

    sample_color = _analysis_variant_color(analysis_variant or "stage04_all")
    colors = {treated_label: sample_color, control_label: sample_color}
    horizons = sorted(
        set(treated_agg["horizon_years"].dropna().astype(int).unique().tolist())
        | set(ctrl_agg["horizon_years"].dropna().astype(int).unique().tolist())
    )
    keep_horizons = {0, 3}
    horizons = [horizon for horizon in horizons if horizon in keep_horizons]
    if not horizons:
        print("  No zero- or three-year horizon rows remain for raw means plots.")
        return

    for outcome in OUTCOMES:
        mean_col = f"{outcome}_mean"
        se_col = f"{outcome}_se"
        if mean_col not in treated_agg.columns:
            continue

        fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
        for agg_df in [treated_agg, ctrl_agg]:
            if mean_col not in agg_df.columns:
                continue
            label = agg_df["series"].iloc[0]
            color = colors.get(label, "#4c78a8")
            is_treated_series = label == treated_label
            linestyle = "-" if is_treated_series else "--"
            role_label = "Treated" if is_treated_series else "Control"
            marker = "o" if is_treated_series else "x"
            for idx, (horizon, grp) in enumerate(agg_df.groupby("horizon_years", sort=True)):
                if int(horizon) not in keep_horizons:
                    continue
                line_df = _coerce_plot_frame(grp, ["cohort_t", mean_col], sort_col="cohort_t")
                if line_df.empty:
                    continue
                line_label = f"{role_label}, {int(horizon)}yr"
                horizon_color = (
                    _blend_hex(color, "#FFFFFF", 0.62)
                    if int(horizon) == 0
                    else _blend_hex(color, "#000000", 0.32)
                )
                ax.plot(
                    line_df["cohort_t"].to_numpy(dtype=float),
                    line_df[mean_col].to_numpy(dtype=float),
                    marker=marker,
                    markersize=EVENT_PLOT_MARKER_SIZE,
                    linewidth=1.6,
                    label=line_label,
                    color=horizon_color,
                    linestyle=linestyle,
                    alpha=1.0,
                )
                valid_ci = grp[se_col].notna() & (pd.to_numeric(grp[se_col], errors="coerce") > 0)
                if valid_ci.any():
                    ci_df = _coerce_plot_frame(
                        grp.loc[valid_ci],
                        ["cohort_t", mean_col, se_col],
                        sort_col="cohort_t",
                    )
                    ax.fill_between(
                        ci_df["cohort_t"].to_numpy(dtype=float),
                        ci_df[mean_col].to_numpy(dtype=float) - 1.96 * ci_df[se_col].to_numpy(dtype=float),
                        ci_df[mean_col].to_numpy(dtype=float) + 1.96 * ci_df[se_col].to_numpy(dtype=float),
                        alpha=0.12,
                        color=horizon_color,
                    )

        ax.axvline(x=DI_D_EVENT_LINE_X, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel(
            "Graduation year relative to relabel event",
            fontsize=DI_D_PLOT_FONT_SIZE,
        )
        ax.set_ylabel(
            OUTCOME_LABELS.get(outcome, outcome),
            fontsize=DI_D_PLOT_FONT_SIZE,
        )
        ax.set_title("")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            llstyle.right_legend(ax)
        _save_and_show(
            fig,
            f"{outcome}_event_study_treated_vs_control",
            analysis_variant=analysis_variant,
        )

    print(f"  Step 7 done in {_elapsed(t)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 – Staggered DiD
# ─────────────────────────────────────────────────────────────────────────────


def step8_pooled_post_by_horizon(
    treated_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    analysis_variant: str | None = None,
) -> pd.DataFrame:
    """Run pooled treat x post regressions separately by post-graduation horizon."""
    t = time.time()
    print(
        "\n── Step 8: Pooled treat x post by horizon"
        + (f" [{analysis_variant}]" if analysis_variant else "")
        + " ─────────────────────────────"
    )

    if not cfg.BUILD_RUN_DID:
        print("  Skipping pooled DiD (run_did=false in config).")
        return pd.DataFrame()
    if control_panel.empty:
        print("  Skipping pooled DiD: control panel is empty.")
        return pd.DataFrame()

    combined = pd.concat([treated_panel, control_panel], ignore_index=True)
    if "target_year_observed" in combined.columns:
        n_before = len(combined)
        combined = combined[pd.to_numeric(combined["target_year_observed"], errors="coerce") == 1].copy()
        n_dropped = n_before - len(combined)
        if n_dropped:
            print(f"  dropped {n_dropped:,} future target-year rows before pooled DiD")

    combined = combined.dropna(
        subset=["cohort_t", "treated_ind", "relabel_year", "grad_year", "horizon_years"]
    ).copy()
    combined["cohort_t"] = pd.to_numeric(combined["cohort_t"], errors="coerce")
    combined["relabel_year"] = pd.to_numeric(combined["relabel_year"], errors="coerce")
    combined["grad_year"] = pd.to_numeric(combined["grad_year"], errors="coerce")
    combined["horizon_years"] = pd.to_numeric(combined["horizon_years"], errors="coerce")
    combined = combined.dropna(subset=["cohort_t", "relabel_year", "grad_year", "horizon_years"])
    combined["cohort_t"] = combined["cohort_t"].astype(int)
    combined["relabel_year"] = combined["relabel_year"].astype(int)
    combined["grad_year"] = combined["grad_year"].astype(int)
    combined["horizon_years"] = combined["horizon_years"].astype(int)

    window = cfg.BUILD_EVENT_WINDOW
    combined = combined[combined["cohort_t"].between(-window, window)].copy()
    post_min, post_max = _pooled_post_event_bounds()
    combined = combined[
        combined["cohort_t"].lt(post_min)
        | combined["cohort_t"].between(post_min, post_max)
    ].copy()
    combined["post_ind"] = combined["cohort_t"].between(post_min, post_max).astype(int)
    if combined.empty:
        print("  Pooled DiD panel is empty after cohort-window filter; skipping.")
        return pd.DataFrame()

    unitid_series = (
        combined["unitid"]
        if "unitid" in combined.columns
        else pd.Series(-1, index=combined.index, dtype="int64")
    )
    education_number_series = (
        combined["education_number"]
        if "education_number" in combined.columns
        else pd.Series(-1, index=combined.index, dtype="int64")
    )
    combined["cluster_unitid"] = pd.to_numeric(unitid_series, errors="coerce").fillna(-1).astype("int64")
    combined["cluster_education_number"] = pd.to_numeric(
        education_number_series, errors="coerce"
    ).fillna(-1).astype("int64")
    combined["did_fe_group"] = _build_did_fe_group(combined)
    event_identity = _event_identity_value(combined)
    entity_keys = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "user_id": pd.to_numeric(combined["user_id"], errors="coerce").fillna(-1).astype("int64"),
                "relabel_year": pd.to_numeric(combined["relabel_year"], errors="coerce").fillna(-1).astype("int64"),
                "treated_ind": pd.to_numeric(combined["treated_ind"], errors="coerce").fillna(-1).astype("int64"),
                "horizon_years": pd.to_numeric(combined["horizon_years"], errors="coerce").fillna(-1).astype("int64"),
                "event_identity": event_identity.astype(str),
                "cluster_unitid": combined["cluster_unitid"],
                "cluster_education_number": combined["cluster_education_number"],
            }
        )
    )
    combined["did_entity_id"] = pd.factorize(entity_keys, sort=False)[0].astype("int64")

    print(
        f"  Pooled DiD panel: {len(combined):,} rows | "
        f"{combined['user_id'].nunique():,} users | "
        f"{combined['did_entity_id'].nunique():,} event-entities | "
        f"{combined['cluster_unitid'].nunique():,} schools | "
        f"{combined['horizon_years'].nunique()} horizons"
    )
    print(f"  Fixed effects: C({_did_fe_var(combined)}) + C(grad_year); clustered by cluster_unitid")
    print(f"  Pooled post period: cohort_t {post_min} through {post_max}")

    results_rows: list[dict[str, object]] = []
    available_horizons = sorted(combined["horizon_years"].dropna().astype(int).unique().tolist())
    print("\n  Pooled results:")
    for outcome in OUTCOMES:
        if outcome not in combined.columns:
            continue
        base = combined.dropna(subset=[outcome]).copy()
        if len(base) < 5:
            continue
        for horizon in available_horizons:
            sub = base[base["horizon_years"] == horizon].copy()
            result_row = _fit_pooled_post_result(
                sub,
                outcome=outcome,
                analysis_variant=analysis_variant,
                horizon=int(horizon),
            )
            if result_row is None:
                continue
            stars = (
                "***" if pd.notna(result_row["pval"]) and float(result_row["pval"]) < 0.01 else
                "**" if pd.notna(result_row["pval"]) and float(result_row["pval"]) < 0.05 else
                "*" if pd.notna(result_row["pval"]) and float(result_row["pval"]) < 0.10 else ""
            )
            print(
                f"    {outcome:<20s} h={int(horizon):>2d}  "
                f"coef={float(result_row['coef']):>10.4f}  "
                f"se={float(result_row['se']):>8.4f}  "
                f"p={float(result_row['pval']):.3f} {stars}  "
                f"base={_format_compact_number(float(result_row['baseline_mean']))}"
            )
            results_rows.append(result_row)

    results_df = (
        pd.DataFrame(results_rows)
        .sort_values(["did_model", "outcome", "horizon_years"])
        .reset_index(drop=True)
        if results_rows
        else pd.DataFrame()
    )
    if results_df.empty:
        print("  No pooled DiD estimates were produced.")
        print(f"  Step 8 done in {_elapsed(t)}")
        return results_df

    title_label = _analysis_variant_label(analysis_variant or "unspecified")
    file_tag = _variant_slug(analysis_variant or "unspecified")
    _plot_pooled_horizon_profile(
        results_df,
        analysis_variant=analysis_variant or "unspecified",
        file_tag=file_tag,
        title_label=title_label,
    )
    if analysis_variant == "foia_linked_person_baseline":
        _plot_pooled_horizon_profile(
            results_df,
            analysis_variant=analysis_variant,
            file_tag="foia_linked_horizons",
            title_label="FOIA-linked sample",
        )

    print(f"  Step 8 done in {_elapsed(t)}")
    return results_df


def step8_did(
    treated_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    analysis_variant: str | None = None,
) -> pd.DataFrame:
    """
    Run cohort-based DiD or treated-only stacked event-study regressions.

    For each configured post-graduation horizon, estimate the same FE structure
    as the generalized relabel coefficient plots:
        outcome ~ C(cohort_t, Treatment(reference=<reference_cohort_t>))*treated_ind
                + C(unitid x broad_pair_bin x degree_type) + C(grad_year)
    or, for stacked_treated, the same regression without controls or treated_ind.
    with school-clustered SEs.
    """
    t = time.time()
    print(
        "\n── Step 8: Staggered DiD"
        + (f" [{analysis_variant}]" if analysis_variant else "")
        + " ─────────────────────────────────────────────"
    )

    if not cfg.BUILD_RUN_DID:
        print("  Skipping DiD (run_did=false in config).")
        return pd.DataFrame()
    estimators_to_run = _did_estimators_to_run()
    if control_panel.empty and ESTIMATOR_STACKED_TREATED not in estimators_to_run:
        print("  Skipping DiD: control panel is empty.")
        return pd.DataFrame()
    if control_panel.empty and ESTIMATOR_DID in estimators_to_run:
        print("  Control panel is empty; running stacked_treated only.")
        estimators_to_run = [ESTIMATOR_STACKED_TREATED]

    combined = pd.concat([treated_panel, control_panel], ignore_index=True)
    if "target_year_observed" in combined.columns:
        n_before = len(combined)
        combined = combined[pd.to_numeric(combined["target_year_observed"], errors="coerce") == 1].copy()
        n_dropped = n_before - len(combined)
        if n_dropped:
            print(f"  dropped {n_dropped:,} future target-year rows before DiD")
    combined = combined.dropna(
        subset=["cohort_t", "treated_ind", "relabel_year", "grad_year", "horizon_years"]
    ).copy()
    combined["cohort_t"] = pd.to_numeric(combined["cohort_t"], errors="coerce")
    combined["relabel_year"] = pd.to_numeric(combined["relabel_year"], errors="coerce")
    combined["grad_year"] = pd.to_numeric(combined["grad_year"], errors="coerce")
    combined["horizon_years"] = pd.to_numeric(combined["horizon_years"], errors="coerce")
    combined = combined.dropna(subset=["cohort_t", "relabel_year", "grad_year", "horizon_years"])
    combined["cohort_t"] = combined["cohort_t"].astype(int)
    combined["relabel_year"] = combined["relabel_year"].astype(int)
    combined["grad_year"] = combined["grad_year"].astype(int)
    combined["horizon_years"] = combined["horizon_years"].astype(int)

    window = cfg.BUILD_EVENT_WINDOW
    combined = combined[combined["cohort_t"].between(-window, window)].copy()
    if combined.empty:
        print("  DiD panel is empty after cohort-window filter; skipping.")
        return pd.DataFrame()

    unitid_series = (
        combined["unitid"]
        if "unitid" in combined.columns
        else pd.Series(-1, index=combined.index, dtype="int64")
    )
    education_number_series = (
        combined["education_number"]
        if "education_number" in combined.columns
        else pd.Series(-1, index=combined.index, dtype="int64")
    )
    combined["cluster_unitid"] = pd.to_numeric(unitid_series, errors="coerce").fillna(-1).astype("int64")
    combined["cluster_education_number"] = pd.to_numeric(
        education_number_series, errors="coerce"
    ).fillna(-1).astype("int64")
    combined["did_fe_group"] = _build_did_fe_group(combined)
    event_identity = _event_identity_value(combined)
    entity_keys = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "user_id": pd.to_numeric(combined["user_id"], errors="coerce").fillna(-1).astype("int64"),
                "relabel_year": pd.to_numeric(combined["relabel_year"], errors="coerce").fillna(-1).astype("int64"),
                "treated_ind": pd.to_numeric(combined["treated_ind"], errors="coerce").fillna(-1).astype("int64"),
                "horizon_years": pd.to_numeric(combined["horizon_years"], errors="coerce").fillna(-1).astype("int64"),
                "event_identity": event_identity.astype(str),
                "cluster_unitid": combined["cluster_unitid"],
                "cluster_education_number": combined["cluster_education_number"],
            }
        )
    )
    combined["did_entity_id"] = pd.factorize(entity_keys, sort=False)[0].astype("int64")

    print(f"  DiD panel: {len(combined):,} rows | "
          f"{combined['user_id'].nunique():,} users | "
          f"{combined['did_entity_id'].nunique():,} event-entities | "
          f"{combined['cluster_unitid'].nunique():,} schools | "
          f"{combined['horizon_years'].nunique()} horizons")
    print(f"  Treated rows: {combined['treated_ind'].sum():,} | "
          f"Control rows: {(combined['treated_ind'] == 0).sum():,}")
    fe_label = _did_fe_var(combined)
    print(f"  Horizons in DiD: {sorted(combined['horizon_years'].unique().tolist())}")
    print(f"  DiD model setting: {cfg.BUILD_DID_MODEL}")
    print(f"  Estimator setting: {getattr(cfg, 'BUILD_DID_ESTIMATOR', ESTIMATOR_DID)}")
    print(f"  Fixed effects: C({fe_label}) + C(grad_year); clustered by cluster_unitid")

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        smf = None
    if smf is None:
        print("  statsmodels not available; skipping DiD.")
        return pd.DataFrame()

    models_to_run = _did_models_to_run()
    print("\n  Event-study results:")
    results_rows = []
    available_horizons = sorted(combined["horizon_years"].dropna().astype(int).unique().tolist())
    for outcome in OUTCOMES:
        if outcome not in combined.columns:
            continue

        base = combined.dropna(subset=[outcome]).copy()
        if len(base) < 5:
            print(f"  {outcome}: too few obs ({len(base)}), skipping")
            continue
        for horizon in available_horizons:
            horizon_base = base[base["horizon_years"] == horizon].copy()
            if len(horizon_base) < 5:
                continue

            for estimator_kind in estimators_to_run:
                sub_source = (
                    horizon_base[pd.to_numeric(horizon_base["treated_ind"], errors="coerce").eq(1)].copy()
                    if estimator_kind == ESTIMATOR_STACKED_TREATED
                    else horizon_base.copy()
                )
                if len(sub_source) < 5:
                    continue
                if estimator_kind == ESTIMATOR_DID and sub_source["treated_ind"].nunique() < 2:
                    print(f"  {outcome} [h={int(horizon)}]: insufficient treated/control variation, skipping DiD")
                    continue

                reference_cohort_t = (
                    _choose_stacked_reference_cohort_t(sub_source, default=-2)
                    if estimator_kind == ESTIMATOR_STACKED_TREATED
                    else _choose_reference_cohort_t(sub_source, default=-2)
                )
                supported_cohorts = (
                    _supported_stacked_cohorts(sub_source, reference_cohort_t=reference_cohort_t)
                    if estimator_kind == ESTIMATOR_STACKED_TREATED
                    else _supported_did_cohorts(sub_source, reference_cohort_t=reference_cohort_t)
                )
                if not supported_cohorts:
                    print(
                        f"  {outcome} [{estimator_kind}] h={int(horizon)}: "
                        "no supported cohort_t coefficients, skipping"
                    )
                    continue

                cohort_values = sorted({reference_cohort_t, *supported_cohorts})
                sub = sub_source[sub_source["cohort_t"].isin(cohort_values)].copy()
                treated_ref_mean = sub.loc[
                    (sub["treated_ind"] == 1) & (sub["cohort_t"] == reference_cohort_t),
                    outcome,
                ].mean()
                control_ref_mean = (
                    sub.loc[
                        (sub["treated_ind"] == 0) & (sub["cohort_t"] == reference_cohort_t),
                        outcome,
                    ].mean()
                    if estimator_kind == ESTIMATOR_DID else np.nan
                )

                for did_model in models_to_run:
                    model_label = did_model if estimator_kind == ESTIMATOR_DID else f"{did_model}_{estimator_kind}"
                    try:
                        formula = f"{outcome} ~ {_did_formula(reference_cohort_t, sub, estimator=estimator_kind)}"
                        if sub["cluster_unitid"].nunique() >= 2:
                            try:
                                result = smf.ols(formula=formula, data=sub).fit(
                                    cov_type="cluster",
                                    cov_kwds={"groups": sub["cluster_unitid"]},
                                )
                                cov = result.cov_params()
                            except Exception as exc:
                                print(
                                    f"    {outcome} [{model_label}] h={int(horizon)}: "
                                    f"clustered SE failed ({exc}); falling back to HC1"
                                )
                                result = smf.ols(formula=formula, data=sub).fit(cov_type="HC1")
                                cov = result.cov_params()
                        else:
                            result = smf.ols(formula=formula, data=sub).fit(cov_type="HC1")
                            cov = result.cov_params()
                        n_obs = int(result.nobs)
                        param_finder = (
                            _find_event_time_param
                            if estimator_kind == ESTIMATOR_STACKED_TREATED
                            else _find_did_interaction_param
                        )
                        ref_param = param_finder(
                            result.params,
                            cohort_t=reference_cohort_t,
                            reference_cohort_t=reference_cohort_t,
                        )
                        ref_coef = float(result.params[ref_param]) if ref_param is not None else 0.0

                        for cohort_t in supported_cohorts:
                            param = param_finder(
                                result.params,
                                cohort_t=cohort_t,
                                reference_cohort_t=reference_cohort_t,
                            )
                            if param is None:
                                continue
                            raw_coef = float(result.params.get(param, float("nan")))
                            if ref_param is None:
                                coef = raw_coef
                                var = float(cov.loc[param, param])
                            else:
                                coef = raw_coef - ref_coef
                                var = float(
                                    cov.loc[param, param]
                                    + cov.loc[ref_param, ref_param]
                                    - 2 * cov.loc[param, ref_param]
                                )
                            se = float(max(var, 0.0) ** 0.5)
                            pval = _normal_pvalue_from_coef_se(coef, se)
                            treated_event_mean = sub.loc[
                                (sub["treated_ind"] == 1) & (sub["cohort_t"] == cohort_t),
                                outcome,
                            ].mean()
                            control_event_mean = (
                                sub.loc[
                                    (sub["treated_ind"] == 0) & (sub["cohort_t"] == cohort_t),
                                    outcome,
                                ].mean()
                                if estimator_kind == ESTIMATOR_DID else np.nan
                            )
                            stars = (
                                "***" if pd.notna(pval) and pval < 0.01 else
                                "**" if pd.notna(pval) and pval < 0.05 else
                                "*" if pd.notna(pval) and pval < 0.10 else ""
                            )
                            print(
                                f"    {outcome:<20s} [{model_label:<22s}] "
                                f"h={int(horizon):>2d}  cohort_t={cohort_t:+d}  "
                                f"coef={coef:>10.4f}  se={se:>8.4f}  "
                                f"p={pval:.3f} {stars}  n={n_obs:,}"
                            )
                            results_rows.append(
                                {
                                    "analysis_variant": analysis_variant or "unspecified",
                                    "did_model": model_label,
                                    "did_estimator": estimator_kind,
                                    "outcome": outcome,
                                    "horizon_years": int(horizon),
                                    "cohort_t": int(cohort_t),
                                    "reference_cohort_t": reference_cohort_t,
                                    "coef": coef,
                                    "se": se,
                                    "pval": pval,
                                    "ci_lower": coef - 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan,
                                    "ci_upper": coef + 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan,
                                    "n_obs": n_obs,
                                    "n_users": int(sub["user_id"].nunique()),
                                    "n_entities": int(sub["did_entity_id"].nunique()),
                                    "n_unitids": int(sub["cluster_unitid"].nunique()),
                                    "formula": formula,
                                    "did_include_individual_controls": int(
                                        getattr(cfg, "BUILD_DID_INCLUDE_INDIVIDUAL_CONTROLS", False)
                                    ),
                                    "did_include_school_char_gradyear_controls": int(
                                        getattr(cfg, "BUILD_DID_INCLUDE_SCHOOL_CHAR_GRADYEAR_CONTROLS", False)
                                    ),
                                    "treated_ref_mean": treated_ref_mean,
                                    "control_ref_mean": control_ref_mean,
                                    "treated_event_mean": treated_event_mean,
                                    "control_event_mean": control_event_mean,
                                }
                            )
                    except Exception as e:
                        print(f"    {outcome} [{model_label}] h={int(horizon)}: event-study failed ({e})")

    results_df = (
        pd.DataFrame(results_rows)
        .sort_values(["did_model", "outcome", "horizon_years", "cohort_t"])
        .reset_index(drop=True)
        if results_rows
        else pd.DataFrame()
    )
    if not results_df.empty:
        _plot_did_coefficients(results_df, analysis_variant=analysis_variant)
    else:
        print("  No DiD estimates were produced.")

    print("\n  Cohort-specific treated series (aggregated by relabel_year):")
    cohort_agg = _agg_cohort_time(
        treated_panel[treated_panel["cohort_t"].between(-window, window)],
        group_col="relabel_year",
    )
    if "relabel_year" in cohort_agg.columns:
        for outcome in OUTCOMES:
            mean_col = f"{outcome}_mean"
            if mean_col not in cohort_agg.columns:
                continue
            for horizon in available_horizons:
                horizon_grp = cohort_agg[cohort_agg["horizon_years"] == horizon].copy()
                if horizon_grp.empty:
                    continue
                fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
                for cohort_year, grp in horizon_grp.groupby("relabel_year"):
                    line_df = _coerce_plot_frame(grp, ["cohort_t", mean_col], sort_col="cohort_t")
                    if line_df.empty:
                        continue
                    ax.plot(
                        line_df["cohort_t"].to_numpy(dtype=float),
                        line_df[mean_col].to_numpy(dtype=float),
                        marker=".",
                        linewidth=1.5,
                        alpha=0.7,
                        label=str(cohort_year),
                    )
                ax.axvline(x=DI_D_EVENT_LINE_X, linestyle="--", color="gray", linewidth=1)
                ax.set_xlabel(
                    "Graduation year relative to relabel event",
                    fontsize=DI_D_PLOT_FONT_SIZE,
                )
                ax.set_ylabel(
                    OUTCOME_LABELS.get(outcome, outcome),
                    fontsize=DI_D_PLOT_FONT_SIZE,
                )
                ax.set_title("")
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    llstyle.right_legend(ax, title="Relabel year")
                _save_and_show(
                    fig,
                    f"{outcome}_cohort_event_time{_horizon_file_suffix(int(horizon), available_horizons)}",
                    analysis_variant=analysis_variant,
                )

    print(f"  Step 8 done in {_elapsed(t)}")
    return results_df


def _run_foreign_heterogeneity_did(
    variant_panel: pd.DataFrame,
    *,
    analysis_variant: str,
) -> list[pd.DataFrame]:
    if not getattr(cfg, "BUILD_FOREIGN_HETEROGENEITY", True):
        return []
    if variant_panel.empty or "imputed_foreign_ind" not in variant_panel.columns:
        return []

    out: list[pd.DataFrame] = []
    foreign_flag = pd.to_numeric(variant_panel["imputed_foreign_ind"], errors="coerce")
    for value, suffix, label in (
        (1, "foreign", "foreign"),
        (0, "non_foreign", "non-foreign"),
    ):
        subgroup = variant_panel[foreign_flag.eq(value)].copy()
        if subgroup.empty:
            print(f"  Skipping {label} heterogeneity for {analysis_variant}: no rows.")
            continue
        treated_sub = subgroup[subgroup["treated_ind"] == 1].copy()
        control_sub = subgroup[subgroup["treated_ind"] == 0].copy()
        if treated_sub.empty or control_sub.empty:
            print(
                f"  Skipping {label} heterogeneity for {analysis_variant}: "
                "missing treated or control rows."
            )
            continue
        did_variant = f"{analysis_variant}_{suffix}"
        if _did_plot_mode() == PLOT_MODE_POOLED_POST:
            result = step8_pooled_post_by_horizon(
                treated_sub,
                control_sub,
                analysis_variant=did_variant,
            )
        else:
            result = step8_did(
                treated_sub,
                control_sub,
                analysis_variant=did_variant,
            )
        if not result.empty:
            out.append(result)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: sample-level inspection
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_sample(
    con: ddb.DuckDBPyConnection,
    relabel_df: pd.DataFrame,
    analysis_variant: str,
    sample_view: str,
    treated_indiv: pd.DataFrame,
    control_events: pd.DataFrame | None = None,
    n_sample_people: int = 3,
) -> None:
    """
    Print full-sample/stage-05 sample diagnostics for a single analysis variant.
    """
    print("\n" + "═" * 70)
    print(f"DIAGNOSTIC: sample inspection [{analysis_variant}]")
    print("═" * 70)
    cip_where = _cip_prefix_where_clause("cip")
    print("\n[1] Full-sample funnel:")
    try:
        funnel = con.sql(
            f"""
            SELECT
                COUNT(*) AS rows_total,
                COUNT(*) FILTER (WHERE unitid IS NOT NULL) AS rows_with_unitid,
                COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where}) AS rows_after_cip_filter,
                COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where} AND grad_year IS NULL) AS rows_missing_grad_year,
                COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where} AND grad_year IS NOT NULL) AS rows_stage04_all,
                COUNT(*) FILTER (WHERE user_id IN (SELECT user_id FROM {sample_view})) AS rows_variant,
                COUNT(DISTINCT user_id) FILTER (WHERE user_id IN (SELECT user_id FROM {sample_view})) AS users_variant
            FROM stage04_educ_base
            """
        ).df()
        print(funnel.to_string(index=False))
    except Exception as e:
        print(f"  Could not compute full-sample funnel: {e}")

    print("\n[2] Treated/control school coverage:")
    treated_total = int(
        pd.to_numeric(
            relabel_df.loc[relabel_df["event_flag"] == 1, "unitid"],
            errors="coerce",
        ).dropna().nunique()
    )
    treated_hit = int(treated_indiv["unitid"].nunique()) if not treated_indiv.empty else 0
    print(f"  treated schools with relabel events: {treated_total:,}")
    print(f"  treated schools represented in sample: {treated_hit:,}")
    if control_events is not None and not control_events.empty:
        con.register("control_events_diag_py", control_events)
        con.sql("CREATE OR REPLACE TEMP VIEW control_events_diag AS SELECT * FROM control_events_diag_py")
        try:
            control_hit = int(
                con.sql(
                    f"""
                    SELECT COUNT(DISTINCT s.unitid)
                    FROM {sample_view} AS s
                    JOIN control_events_diag AS c
                      ON s.unitid = CAST(c.unitid AS BIGINT)
                    """
                ).fetchone()[0] or 0
            )
            print(f"  control pseudo-event schools: {control_events['unitid'].nunique():,}")
            print(f"  control schools represented in sample: {control_hit:,}")
        except Exception as e:
            print(f"  Could not compute control coverage: {e}")
    else:
        print("  control pseudo-event schools: 0")

    print("\n[3] FOIA-linked overlap counts:")
    if _table_exists(con, "stage05_person_baseline_users"):
        try:
            overlap = con.sql(
                f"""
                SELECT
                    (SELECT COUNT(*) FROM stage05_person_baseline_users) AS baseline_users,
                    (SELECT COUNT(DISTINCT user_id) FROM stage04_sample_all) AS stage04_all_users,
                    (SELECT COUNT(DISTINCT s.user_id)
                     FROM stage04_sample_all AS s
                     JOIN stage05_person_baseline_users AS u USING (user_id)) AS overlap_users,
                    (SELECT COUNT(DISTINCT user_id) FROM {sample_view}) AS variant_users
                """
            ).df()
            print(overlap.to_string(index=False))
        except Exception as e:
            print(f"  Could not compute FOIA overlap counts: {e}")
    else:
        print("  stage05_person_baseline_users not available")

    print(f"\n[4] Sample matched education histories from `{sample_view}`:")
    if treated_indiv.empty:
        try:
            sample_rows = con.sql(
                f"""
                SELECT user_id, unitid, grad_year, degree_clean, cip, university_raw, field_clean,
                       ed_startdate, ed_enddate
                FROM {sample_view}
                ORDER BY user_id, education_number
                LIMIT 10
                """
            ).df()
            if sample_rows.empty:
                print("  No rows in sample view.")
            else:
                print(sample_rows.to_string(index=False))
        except Exception as e:
            print(f"  Could not query sample view: {e}")
        print("\n" + "═" * 70)
        return

    sample_users = treated_indiv["user_id"].drop_duplicates().sample(
        min(n_sample_people, treated_indiv["user_id"].nunique()),
        random_state=42,
    ).tolist()
    for uid in sample_users:
        matched_row = treated_indiv[treated_indiv["user_id"] == uid].iloc[0]
        print(
            f"\n  ── user_id={uid} unitid={matched_row['unitid']} grad_year={matched_row['grad_year']} "
            f"relabel_year={matched_row['relabel_year']} cohort_t={matched_row['cohort_t']}"
        )
        try:
            educ = con.sql(
                f"""
                SELECT
                    education_number,
                    unitid,
                    degree_clean,
                    cip,
                    university_raw,
                    field_clean,
                    grad_year,
                    ed_startdate,
                    ed_enddate,
                    school_match_score
                FROM stage04_educ_base
                WHERE user_id = {int(uid)}
                ORDER BY ed_startdate, education_number
                """
            ).df()
            print(educ.to_string(index=False, max_rows=10))
        except Exception as e:
            print(f"    Could not fetch education history: {e}")

    print("\n" + "═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_main = time.time()
    con = ddb.connect()
    # Configure DuckDB for performance
    n_threads = max(1, os.cpu_count() or 1)
    con.sql(f"PRAGMA threads={n_threads}")
    con.sql("PRAGMA preserve_insertion_order=false")

    # ── Step 1: Detect relabels + aggregate plots ────────────────────────────
    relabel_df = step1_relabels(con)

    # Testing: optionally restrict to a sample of treated institutions
    testing_unitids: list[int] | None = None
    if cfg.TESTING_ENABLED:
        rng = np.random.default_rng(cfg.TESTING_RANDOM_SEED)
        all_treated_unitids = relabel_df.loc[
            relabel_df["event_flag"] == 1, "unitid"
        ].unique().tolist()
        n_sample = min(cfg.TESTING_SAMPLE_N_INSTITUTIONS, len(all_treated_unitids))
        testing_unitids = sorted(
            [int(x) for x in rng.choice(all_treated_unitids, size=n_sample, replace=False)]
        )
        print(f"\n[TEST MODE] Sampled {n_sample} treated institutions: {testing_unitids}")

    # Step 2: Load full-sample/stage-05 sample views
    sample_variants = step2_prepare_stage04_samples(con)
    _ensure_enrichment_views(con)

    # ── Control pseudo-events (shared across variants) ──────────────────────
    control_events = build_control_events(con, relabel_df, testing_unitids=testing_unitids)

    combined_panels: list[pd.DataFrame] = []
    combined_did_results: list[pd.DataFrame] = []
    for analysis_variant in sample_variants:
        sample_view = _sample_view_name(analysis_variant)
        if not _table_exists(con, sample_view):
            raise ValueError(f"Missing sample view for variant {analysis_variant}: {sample_view}")

        treated_indiv = step3_match_treated(
            con,
            relabel_df,
            sample_view,
            analysis_variant,
            testing_unitids=testing_unitids,
            control_events=control_events,
        )
        diagnose_sample(
            con,
            relabel_df,
            analysis_variant,
            sample_view,
            treated_indiv,
            control_events=control_events,
        )

        if treated_indiv.empty:
            print(f"\nNo treated individuals found for variant {analysis_variant}; skipping.")
            continue

        treated_panel = step4_build_outcome_panel(
            con,
            treated_indiv,
            group_label=f"treated_{_variant_slug(analysis_variant)}",
            analysis_variant=analysis_variant,
        )
        step5_event_study_plots(
            treated_panel,
            label="treated",
            analysis_variant=analysis_variant,
        )

        control_panel = step6_control_group(
            con,
            sample_view,
            control_events,
            analysis_variant,
        )
        if not control_panel.empty:
            variant_panel = pd.concat([treated_panel, control_panel], ignore_index=True)
        else:
            variant_panel = treated_panel.copy()
        variant_panel = _finalize_variant_panel(variant_panel)
        treated_panel = variant_panel[variant_panel["treated_ind"] == 1].copy()
        control_panel = variant_panel[variant_panel["treated_ind"] == 0].copy()

        step7_treated_vs_control_plots(
            treated_panel,
            control_panel,
            analysis_variant=analysis_variant,
        )
        if _did_plot_mode() == PLOT_MODE_POOLED_POST:
            did_results = step8_pooled_post_by_horizon(
                treated_panel,
                control_panel,
                analysis_variant=analysis_variant,
            )
        else:
            did_results = step8_did(
                treated_panel,
                control_panel,
                analysis_variant=analysis_variant,
            )
        if not did_results.empty:
            combined_did_results.append(did_results)
        combined_did_results.extend(
            _run_foreign_heterogeneity_did(
                variant_panel,
                analysis_variant=analysis_variant,
            )
        )

        combined_panels.append(variant_panel)

    if not combined_panels:
        print("\nNo variant produced a treated sample. Check full-sample/stage-05 inputs.")
        return

    combined = pd.concat(combined_panels, ignore_index=True)

    panel_out = cfg.OUTPUT_PANEL_PARQUET
    os.makedirs(os.path.dirname(panel_out), exist_ok=True)
    combined.to_parquet(panel_out, index=False)
    print(f"\nSaved combined panel → {panel_out}")

    if combined_did_results:
        did_combined = pd.concat(combined_did_results, ignore_index=True)
        did_out = _did_results_output_path()
        did_out.parent.mkdir(parents=True, exist_ok=True)
        did_combined.to_parquet(did_out, index=False)
        did_csv = did_out.with_suffix(".csv")
        did_combined.to_csv(did_csv, index=False)
        print(f"Saved DiD results → {did_out}")
        print(f"Saved DiD results CSV → {did_csv}")
        base_compare = did_combined[
            ~did_combined["analysis_variant"].map(_is_foreign_heterogeneity_variant)
        ].copy()
        heterogeneity_compare = did_combined[
            did_combined["analysis_variant"].map(_is_foreign_heterogeneity_variant)
        ].copy()
        if _did_plot_mode() == PLOT_MODE_POOLED_POST:
            _plot_pooled_variant_comparison(
                base_compare,
                file_tag="variant",
                title_label="match sample",
            )
            _plot_pooled_variant_comparison(
                heterogeneity_compare,
                file_tag="foreign_status",
                title_label="imputed foreign status and match sample",
            )
        else:
            _plot_did_variant_comparison(base_compare)
            _plot_did_horizon_comparison(
                base_compare,
                analysis_variant="foia_linked_person_baseline",
                file_tag="foia_linked_horizons",
                title_label="FOIA-linked sample",
            )
            _plot_did_variant_comparison(
                heterogeneity_compare,
                file_tag="foreign_status",
                title_label="imputed foreign status and match sample",
            )
    else:
        print("No DiD results saved.")
    print(f"\nTotal time: {_elapsed(t_main)}")


if __name__ == "__main__":
    main()
