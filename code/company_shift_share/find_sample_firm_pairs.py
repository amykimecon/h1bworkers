"""Find sample firm pairs with similar size and different university shock timing.

Notebook-friendly script:
- no CLI args, no main()
- edit top-level parameter block and re-run.

Selection logic:
1) Pick each firm's dominant university by transition share.
2) For each university, identify years with a large shock:
   - (g_t - g_{t-1}) / g_{t-1} >= min_growth_ratio AND (g_t - g_{t-1}) >= min_level_change
     OR
   - new program creation (prev <= 0 and g_t >= min_level_change), if enabled.
3) Require each firm to have at least one OPT hire in HIRE_BASE_T and at least one in a later year.
4) Form firm pairs with similar initial size, different dominant universities, and
   shock years sufficiently far apart.
"""

from __future__ import annotations

from pathlib import Path
import sys

import duckdb as ddb
import matplotlib.pyplot as plt
import pandas as pd

try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config


def _resolve_path(paths_cfg: dict, key: str) -> Path:
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"Config paths.{key} must be set.")
    return Path(value)


def _find_size_col(con: ddb.DuckDBPyConnection) -> str:
    cols = con.sql("PRAGMA table_info('analysis_panel')").df()["name"].tolist()
    for candidate in ("y_cst_lagm1", "y_cst_lag0", "y_cst_lag1"):
        if candidate in cols:
            return candidate
    raise ValueError(f"No usable size column found in analysis_panel. Columns: {cols}")


def _find_outcome_col(con: ddb.DuckDBPyConnection, preferred: str | None) -> str:
    cols = con.sql("PRAGMA table_info('analysis_panel')").df()["name"].tolist()
    if preferred and preferred in cols:
        return preferred
    for candidate in ("y_cst_lag0", "y", "masters_opt_hires"):
        if candidate in cols:
            return candidate
    raise ValueError(f"No usable outcome column found in analysis_panel. Columns: {cols}")


def _esc(p: Path) -> str:
    return str(p).replace("'", "''")


def _build_sql(
    *,
    share_col: str,
    g_col: str,
    top_n: int,
    min_share: float,
    size_log_tol: float,
    min_event_year_gap: int,
    max_event_year_gap: int | None,
    min_level_change: float,
    min_growth_ratio: float,
    allow_new_program: bool,
    size_col: str,
    hire_base_t: int,
    min_firm_size: float,
    size_match_year: int,
    min_share_num: int,
) -> str:
    share_num_col = "n_transitions_full" if share_col == "share_ck_full" else "n_transitions"
    new_program_clause = "FALSE"
    if allow_new_program:
        new_program_clause = f"(COALESCE(prev_g, 0) <= 0 AND g_t >= {float(min_level_change)})"

    return f"""
    WITH firm_init AS (
        SELECT c, MIN(t) AS t0
        FROM analysis_panel
        GROUP BY c
    ),
    firm_name_filter AS (
        SELECT
            ap.c
        FROM (SELECT DISTINCT c FROM analysis_panel) ap
        LEFT JOIN firm_meta fm USING (c)
        WHERE fm.firm_name IS NULL
           OR NOT regexp_matches(LOWER(fm.firm_name), '(hospital|university|college|school|health|bluecross|medical)')
    ),
    firm_hire_filter AS (
        SELECT c
        FROM analysis_panel
        GROUP BY c
        HAVING
            MAX(CASE WHEN t = {int(hire_base_t)} AND COALESCE(valid_masters_opt_hires, 0) >= 1 THEN 1 ELSE 0 END) = 1
            AND MAX(CASE WHEN t > {int(hire_base_t)} AND COALESCE(valid_masters_opt_hires, 0) >= 1 THEN 1 ELSE 0 END) = 1
    ),
    firm_size_year AS (
        SELECT ap.c, ap.{size_col} AS size_match
        FROM analysis_panel ap
        JOIN firm_name_filter fnf USING (c)
        JOIN firm_hire_filter fhf USING (c)
        WHERE ap.t = {int(size_match_year)}
          AND ap.{size_col} IS NOT NULL
          AND ap.{size_col} >= {float(min_firm_size)}
    ),
    firm_top_univ AS (
        SELECT c, k, {share_col} AS share_val, {share_num_col} AS share_num
        FROM (
            SELECT
                c,
                k,
                {share_col},
                {share_num_col},
                ROW_NUMBER() OVER (PARTITION BY c ORDER BY {share_col} DESC NULLS LAST) AS rn
            FROM transition_unit_shares
            WHERE {share_col} IS NOT NULL
        )
        WHERE rn = 1
          AND share_val >= {float(min_share)}
          AND COALESCE(share_num, 0) >= {int(min_share_num)}
    ),
    univ_growth AS (
        SELECT
            k,
            t,
            CAST({g_col} AS DOUBLE) AS g_t,
            LAG(CAST({g_col} AS DOUBLE)) OVER (PARTITION BY k ORDER BY t) AS prev_g
        FROM ipeds_unit_growth
        WHERE {g_col} IS NOT NULL
          AND t > 2010
    ),
    univ_events AS (
        SELECT
            k,
            t AS event_t,
            g_t,
            prev_g,
            g_t - COALESCE(prev_g, 0) AS delta_g,
            CASE
                WHEN prev_g > 0 THEN (g_t - prev_g) / prev_g
                ELSE NULL
            END AS growth_ratio,
            CASE WHEN {new_program_clause} THEN 1 ELSE 0 END AS is_new_program
        FROM univ_growth
    ),
    univ_events_qualified AS (
        SELECT *
        FROM univ_events
        WHERE event_t > 2011 AND event_t < 2020
          AND (
            (
                prev_g > 0
                AND growth_ratio >= {float(min_growth_ratio)}
                AND delta_g >= {float(min_level_change)}
            )
            OR is_new_program = 1
          )
    ),
    univ_event_choice AS (
        SELECT
            k,
            event_t,
            g_t,
            prev_g,
            delta_g,
            growth_ratio,
            is_new_program
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY k
                    ORDER BY is_new_program DESC, delta_g DESC, growth_ratio DESC NULLS LAST, event_t ASC
                ) AS rn
            FROM univ_events_qualified
        )
        WHERE rn = 1
    ),
    firm_profile AS (
        SELECT
            ftu.c,
            ftu.k,
            ftu.share_val,
            ftu.share_num,
            uec.event_t,
            uec.delta_g,
            uec.growth_ratio,
            uec.is_new_program,
            fsy.size_match
        FROM firm_top_univ ftu
        JOIN univ_event_choice uec USING (k)
        JOIN firm_size_year fsy USING (c)
    ),
    pairs AS (
        SELECT
            a.c AS firm_a,
            b.c AS firm_b,
            COALESCE(fma.firm_name, CAST(a.c AS VARCHAR)) AS firm_a_name,
            fma.firm_city AS firm_a_city,
            fma.firm_state AS firm_a_state,
            COALESCE(fmb.firm_name, CAST(b.c AS VARCHAR)) AS firm_b_name,
            fmb.firm_city AS firm_b_city,
            fmb.firm_state AS firm_b_state,
            a.size_match AS size_match_a,
            b.size_match AS size_match_b,
            a.k AS univ_a,
            b.k AS univ_b,
            COALESCE(uma.univ_name, a.k) AS univ_a_name,
            COALESCE(umb.univ_name, b.k) AS univ_b_name,
            a.share_val AS share_a,
            b.share_val AS share_b,
            a.share_num AS share_num_a,
            b.share_num AS share_num_b,
            a.event_t AS event_t_a,
            b.event_t AS event_t_b,
            a.delta_g AS delta_g_a,
            b.delta_g AS delta_g_b,
            a.growth_ratio AS growth_ratio_a,
            b.growth_ratio AS growth_ratio_b,
            a.is_new_program AS is_new_program_a,
            b.is_new_program AS is_new_program_b,
            ABS(LN(a.size_match + 1) - LN(b.size_match + 1)) AS size_log_diff,
            ABS(a.event_t - b.event_t) AS event_year_gap,
            (a.share_val + b.share_val) AS share_score,
            (a.delta_g + b.delta_g) AS shock_score
        FROM firm_profile a
        JOIN firm_profile b
          ON a.c < b.c
         AND a.k <> b.k
        LEFT JOIN firm_meta fma ON a.c = fma.c
        LEFT JOIN firm_meta fmb ON b.c = fmb.c
        LEFT JOIN univ_meta uma ON a.k = uma.k
        LEFT JOIN univ_meta umb ON b.k = umb.k
        WHERE ABS(LN(a.size_match + 1) - LN(b.size_match + 1)) <= {float(size_log_tol)}
          AND ABS(a.event_t - b.event_t) >= {int(min_event_year_gap)}
          {"AND ABS(a.event_t - b.event_t) <= " + str(int(max_event_year_gap)) if max_event_year_gap is not None else ""}
    )
    SELECT *
    FROM pairs
    ORDER BY GREATEST(size_match_a, size_match_b) DESC, shock_score DESC, share_score DESC, size_log_diff ASC
    LIMIT {int(top_n)}
    """


def _build_stage_counts_sql(
    *,
    share_col: str,
    g_col: str,
    min_share: float,
    size_log_tol: float,
    min_event_year_gap: int,
    max_event_year_gap: int | None,
    min_level_change: float,
    min_growth_ratio: float,
    allow_new_program: bool,
    size_col: str,
    hire_base_t: int,
    min_firm_size: float,
    size_match_year: int,
    min_share_num: int,
) -> str:
    new_program_clause = "FALSE"
    if allow_new_program:
        new_program_clause = f"(COALESCE(prev_g, 0) <= 0 AND g_t >= {float(min_level_change)})"

    return f"""
    WITH firm_init AS (
        SELECT c, MIN(t) AS t0
        FROM analysis_panel
        GROUP BY c
    ),
    firm_name_filter AS (
        SELECT
            ap.c
        FROM (SELECT DISTINCT c FROM analysis_panel) ap
        LEFT JOIN firm_meta fm USING (c)
        WHERE fm.firm_name IS NULL
           OR NOT regexp_matches(LOWER(fm.firm_name), '(hospital|university|college|school|health|bluecross|medical)')
    ),
    firm_hire_filter AS (
        SELECT c
        FROM analysis_panel
        GROUP BY c
        HAVING
            MAX(CASE WHEN t = {int(hire_base_t)} AND COALESCE(valid_masters_opt_hires, 0) >= 1 THEN 1 ELSE 0 END) = 1
            AND MAX(CASE WHEN t > {int(hire_base_t)} AND COALESCE(valid_masters_opt_hires, 0) >= 1 THEN 1 ELSE 0 END) = 1
    ),
    firm_size_year AS (
        SELECT ap.c, ap.{size_col} AS size_match
        FROM analysis_panel ap
        JOIN firm_name_filter fnf USING (c)
        JOIN firm_hire_filter fhf USING (c)
        WHERE ap.t = {int(size_match_year)}
          AND ap.{size_col} IS NOT NULL
          AND ap.{size_col} >= {float(min_firm_size)}
    ),
    firm_top_univ AS (
        SELECT c, k, {share_col} AS share_val
        FROM (
            SELECT
                c,
                k,
                {share_col},
                {"n_transitions_full" if share_col == "share_ck_full" else "n_transitions"} AS share_num,
                ROW_NUMBER() OVER (PARTITION BY c ORDER BY {share_col} DESC NULLS LAST) AS rn
            FROM transition_unit_shares
            WHERE {share_col} IS NOT NULL
        )
        WHERE rn = 1
          AND share_val >= {float(min_share)}
          AND COALESCE(share_num, 0) >= {int(min_share_num)}
    ),
    univ_growth AS (
        SELECT
            k,
            t,
            CAST({g_col} AS DOUBLE) AS g_t,
            LAG(CAST({g_col} AS DOUBLE)) OVER (PARTITION BY k ORDER BY t) AS prev_g
        FROM ipeds_unit_growth
        WHERE {g_col} IS NOT NULL
          AND t > 2010
    ),
    univ_events AS (
        SELECT
            k,
            t AS event_t,
            g_t,
            prev_g,
            g_t - COALESCE(prev_g, 0) AS delta_g,
            CASE
                WHEN prev_g > 0 THEN (g_t - prev_g) / prev_g
                ELSE NULL
            END AS growth_ratio,
            CASE WHEN {new_program_clause} THEN 1 ELSE 0 END AS is_new_program
        FROM univ_growth
    ),
    univ_events_qualified AS (
        SELECT *
        FROM univ_events
        WHERE event_t > 2011 AND event_t < 2020
          AND (
            (
                prev_g > 0
                AND growth_ratio >= {float(min_growth_ratio)}
                AND delta_g >= {float(min_level_change)}
            )
            OR is_new_program = 1
          )
    ),
    univ_event_choice AS (
        SELECT
            k,
            event_t,
            delta_g,
            growth_ratio,
            is_new_program
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY k
                    ORDER BY is_new_program DESC, delta_g DESC, growth_ratio DESC NULLS LAST, event_t ASC
                ) AS rn
            FROM univ_events_qualified
        )
        WHERE rn = 1
    ),
    firm_profile AS (
        SELECT
            ftu.c,
            ftu.k,
            ftu.share_val,
            uec.event_t,
            uec.delta_g,
            uec.growth_ratio,
            uec.is_new_program,
            fsy.size_match
        FROM firm_top_univ ftu
        JOIN univ_event_choice uec USING (k)
        JOIN firm_size_year fsy USING (c)
    ),
    pairs_raw AS (
        SELECT
            a.c AS firm_a,
            b.c AS firm_b,
            ABS(LN(a.size_match + 1) - LN(b.size_match + 1)) AS size_log_diff,
            ABS(a.event_t - b.event_t) AS event_year_gap
        FROM firm_profile a
        JOIN firm_profile b
          ON a.c < b.c
         AND a.k <> b.k
    ),
    pairs_after_size AS (
        SELECT *
        FROM pairs_raw
        WHERE size_log_diff <= {float(size_log_tol)}
    ),
    pairs_after_gap AS (
        SELECT *
        FROM pairs_after_size
        WHERE event_year_gap >= {int(min_event_year_gap)}
          {"AND event_year_gap <= " + str(int(max_event_year_gap)) if max_event_year_gap is not None else ""}
    )
    SELECT * FROM (
        SELECT 'firms_in_analysis_panel' AS stage, COUNT(DISTINCT c) AS n FROM analysis_panel
        UNION ALL
        SELECT 'firms_after_name_filter', COUNT(*) FROM firm_name_filter
        UNION ALL
        SELECT 'firms_after_hire_filter', COUNT(*) FROM firm_hire_filter
        UNION ALL
        SELECT 'firms_with_size_match_year', COUNT(DISTINCT c) FROM firm_size_year
        UNION ALL
        SELECT 'firms_with_top_univ', COUNT(DISTINCT c) FROM firm_top_univ
        UNION ALL
        SELECT 'universities_with_event', COUNT(DISTINCT k) FROM univ_event_choice
        UNION ALL
        SELECT 'firms_in_firm_profile', COUNT(DISTINCT c) FROM firm_profile
        UNION ALL
        SELECT 'candidate_pairs_raw', COUNT(*) FROM pairs_raw
        UNION ALL
        SELECT 'candidate_pairs_after_size', COUNT(*) FROM pairs_after_size
        UNION ALL
        SELECT 'candidate_pairs_after_year_gap', COUNT(*) FROM pairs_after_gap
    )
    """


# -----------------------------------------------------------------------------
# Notebook-style parameters (edit these, then re-run)
# -----------------------------------------------------------------------------
CONFIG_PATH: Path | None = None
ANALYSIS_PANEL_PATH: Path | None = None
INSTRUMENT_COMPONENTS_PATH: Path | None = None
EMPLOYER_CROSSWALK_PATH: Path | None = None
THREE_WAY_CROSSWALK_PATH: Path | None = None
TRANSITION_SHARES_PATH: Path | None = None
GROWTH_PANEL_PATH: Path | None = None
OUTPUT_PATH: Path | None = None

# Optional config section: find_sample_firm_pairs
SHARE_COL = None
G_COL = None
TOP_N = None
MIN_SHARE = None
SIZE_LOG_TOL = None
MIN_EVENT_YEAR_GAP = None
MAX_EVENT_YEAR_GAP = None
MIN_LEVEL_CHANGE = None
MIN_GROWTH_RATIO = None
ALLOW_NEW_PROGRAM = None
HIRE_BASE_T = None
REPORT_STAGE_COUNTS = None
MIN_FIRM_SIZE = None
SIZE_MATCH_YEAR = None
MIN_SHARE_NUM = None
OUTCOME_COL = None
X_COL = None
EVENT_WINDOW = None
PAIR_RANDOM_SEED = None

cfg = load_config(CONFIG_PATH)
paths_cfg = get_cfg_section(cfg, "paths")
pairs_cfg = get_cfg_section(cfg, "find_sample_firm_pairs")
build_cfg = get_cfg_section(cfg, "build_company_shift_share")
reg_cfg = get_cfg_section(cfg, "shift_share_regressions")

analysis_panel_path = ANALYSIS_PANEL_PATH or _resolve_path(paths_cfg, "analysis_panel")
instrument_components_path = INSTRUMENT_COMPONENTS_PATH or _resolve_path(paths_cfg, "instrument_components")
employer_crosswalk_path = EMPLOYER_CROSSWALK_PATH or _resolve_path(paths_cfg, "employer_crosswalk")
three_way_crosswalk_path = THREE_WAY_CROSSWALK_PATH or _resolve_path(paths_cfg, "revelio_ipeds_foia_inst_crosswalk")
transition_shares_path = TRANSITION_SHARES_PATH
growth_path = GROWTH_PANEL_PATH

share_col = SHARE_COL or pairs_cfg.get("share_col", "share_ck_full")
g_col = G_COL or pairs_cfg.get("g_col", "g_kt")
top_n = int(TOP_N if TOP_N is not None else pairs_cfg.get("top_n", 20))
min_share = float(MIN_SHARE if MIN_SHARE is not None else pairs_cfg.get("min_share", 0.20))
size_log_tol = float(SIZE_LOG_TOL if SIZE_LOG_TOL is not None else pairs_cfg.get("size_log_tol", 0.5))
min_event_year_gap = int(MIN_EVENT_YEAR_GAP if MIN_EVENT_YEAR_GAP is not None else pairs_cfg.get("min_event_year_gap", 2))
max_event_year_gap_cfg = pairs_cfg.get("max_event_year_gap")
max_event_year_gap = (
    int(MAX_EVENT_YEAR_GAP)
    if MAX_EVENT_YEAR_GAP is not None
    else (int(max_event_year_gap_cfg) if max_event_year_gap_cfg is not None else None)
)
min_level_change = float(MIN_LEVEL_CHANGE if MIN_LEVEL_CHANGE is not None else pairs_cfg.get("min_level_change", 10.0))
min_growth_ratio = float(MIN_GROWTH_RATIO if MIN_GROWTH_RATIO is not None else pairs_cfg.get("min_growth_ratio", 1.0))
allow_new_program = bool(ALLOW_NEW_PROGRAM if ALLOW_NEW_PROGRAM is not None else pairs_cfg.get("allow_new_program", True))
default_hire_base_t = int(build_cfg.get("share_base_year", 2010)) + 1
hire_base_t = int(HIRE_BASE_T if HIRE_BASE_T is not None else pairs_cfg.get("hire_base_t", default_hire_base_t))
report_stage_counts = bool(REPORT_STAGE_COUNTS if REPORT_STAGE_COUNTS is not None else pairs_cfg.get("report_stage_counts", True))
min_firm_size = float(MIN_FIRM_SIZE if MIN_FIRM_SIZE is not None else pairs_cfg.get("min_firm_size", 10))
size_match_year = int(SIZE_MATCH_YEAR if SIZE_MATCH_YEAR is not None else pairs_cfg.get("size_match_year", hire_base_t))
min_share_num = int(MIN_SHARE_NUM if MIN_SHARE_NUM is not None else pairs_cfg.get("min_share_num", 10))
default_outcome_col = f"{reg_cfg.get('outcome_prefix', 'y_cst_lag')}0"
outcome_col = OUTCOME_COL or pairs_cfg.get("outcome_col", default_outcome_col)
x_col = X_COL or pairs_cfg.get("x_col", reg_cfg.get("dependent", "masters_opt_hires"))
event_window = int(EVENT_WINDOW if EVENT_WINDOW is not None else pairs_cfg.get("event_window", 5))
pair_random_seed = PAIR_RANDOM_SEED if PAIR_RANDOM_SEED is not None else pairs_cfg.get("pair_random_seed")

CON = ddb.connect()
CON.sql(f"CREATE OR REPLACE VIEW analysis_panel AS SELECT * FROM read_parquet('{_esc(analysis_panel_path)}')")
CON.sql(f"CREATE OR REPLACE VIEW instrument_components_input AS SELECT * FROM read_parquet('{_esc(instrument_components_path)}')")
CON.sql(f"CREATE OR REPLACE VIEW employer_crosswalk_input AS SELECT * FROM read_parquet('{_esc(employer_crosswalk_path)}')")
CON.sql(f"CREATE OR REPLACE VIEW three_way_crosswalk_input AS SELECT * FROM read_parquet('{_esc(three_way_crosswalk_path)}')")

# firm name/location (mode per preferred_rcid)
CON.sql(
    """
    CREATE OR REPLACE VIEW firm_meta AS
    SELECT
        CAST(preferred_rcid AS BIGINT) AS c,
        f1_empname_clean AS firm_name,
        f1_city_clean AS firm_city,
        f1_state_clean AS firm_state
    FROM (
        SELECT
            preferred_rcid,
            f1_empname_clean,
            f1_city_clean,
            f1_state_clean,
            COUNT(*) AS n,
            ROW_NUMBER() OVER (
                PARTITION BY preferred_rcid
                ORDER BY COUNT(*) DESC, f1_empname_clean, f1_city_clean, f1_state_clean
            ) AS rn
        FROM employer_crosswalk_input
        WHERE preferred_rcid IS NOT NULL
        GROUP BY preferred_rcid, f1_empname_clean, f1_city_clean, f1_state_clean
    )
    WHERE rn = 1
    """
)

# university name per UNITID (mode over three-way crosswalk)
CON.sql(
    """
    CREATE OR REPLACE VIEW univ_meta AS
    SELECT
        CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k,
        COALESCE(ipeds_instname_clean, f1_instname_clean, rev_instname_clean, university_raw) AS univ_name
    FROM (
        SELECT
            unitid,
            ipeds_instname_clean,
            f1_instname_clean,
            rev_instname_clean,
            university_raw,
            COUNT(*) AS n,
            ROW_NUMBER() OVER (
                PARTITION BY unitid
                ORDER BY COUNT(*) DESC, COALESCE(ipeds_instname_clean, f1_instname_clean, rev_instname_clean, university_raw)
            ) AS rn
        FROM three_way_crosswalk_input
        WHERE unitid IS NOT NULL
        GROUP BY unitid, ipeds_instname_clean, f1_instname_clean, rev_instname_clean, university_raw
    )
    WHERE rn = 1
    """
)

if transition_shares_path is not None:
    CON.sql(f"CREATE OR REPLACE VIEW transition_unit_shares AS SELECT * FROM read_parquet('{_esc(transition_shares_path)}')")
else:
    CON.sql(
        """
        CREATE OR REPLACE VIEW transition_unit_shares AS
        SELECT
            c,
            k,
            MAX(share_ck) AS share_ck,
            MAX(share_ck_base) AS share_ck_base,
            MAX(share_ck_full) AS share_ck_full,
            MAX(n_transitions) AS n_transitions,
            MAX(n_transitions_full) AS n_transitions_full
        FROM instrument_components_input
        GROUP BY c, k
        """
    )

if growth_path is not None:
    CON.sql(f"CREATE OR REPLACE VIEW ipeds_unit_growth AS SELECT * FROM read_parquet('{_esc(growth_path)}')")
else:
    CON.sql(
        """
        CREATE OR REPLACE VIEW ipeds_unit_growth AS
        SELECT
            k,
            t,
            MAX(g_kt) AS g_kt,
            MAX(g_kt_all) AS g_kt_all,
            MAX(g_kt_intl) AS g_kt_intl
        FROM instrument_components_input
        GROUP BY k, t
        """
    )

size_col = _find_size_col(CON)
outcome_col = _find_outcome_col(CON, outcome_col)
x_col = _find_outcome_col(CON, x_col)
SQL = _build_sql(
    share_col=share_col,
    g_col=g_col,
    top_n=top_n,
    min_share=min_share,
    size_log_tol=size_log_tol,
    min_event_year_gap=min_event_year_gap,
    max_event_year_gap=max_event_year_gap,
    min_level_change=min_level_change,
    min_growth_ratio=min_growth_ratio,
    allow_new_program=allow_new_program,
    size_col=size_col,
    hire_base_t=hire_base_t,
    min_firm_size=min_firm_size,
    size_match_year=size_match_year,
    min_share_num=min_share_num,
)

if report_stage_counts:
    stage_sql = _build_stage_counts_sql(
        share_col=share_col,
        g_col=g_col,
        min_share=min_share,
        size_log_tol=size_log_tol,
        min_event_year_gap=min_event_year_gap,
        max_event_year_gap=max_event_year_gap,
        min_level_change=min_level_change,
        min_growth_ratio=min_growth_ratio,
        allow_new_program=allow_new_program,
        size_col=size_col,
        hire_base_t=hire_base_t,
        min_firm_size=min_firm_size,
        size_match_year=size_match_year,
        min_share_num=min_share_num,
    )
    STAGE_COUNTS = CON.sql(stage_sql).df()
    print("Stage counts:")
    print(STAGE_COUNTS.to_string(index=False))

TOP_PAIRS = CON.sql(SQL).df()
if TOP_PAIRS.empty:
    print("No pairs found with the current thresholds.")
else:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(TOP_PAIRS.to_string(index=False))
    if pair_random_seed is None:
        top_pair = TOP_PAIRS.sample(n=1).iloc[0]
    else:
        top_pair = TOP_PAIRS.sample(n=1, random_state=int(pair_random_seed)).iloc[0]
    print("Selected random pair for plotting:")
    print(top_pair.to_frame().T.to_string(index=False))
    firm_a = int(top_pair["firm_a"])
    firm_b = int(top_pair["firm_b"])
    event_t_a = int(top_pair["event_t_a"])
    event_t_b = int(top_pair["event_t_b"])
    firm_a_label = str(top_pair["firm_a_name"])
    firm_b_label = str(top_pair["firm_b_name"])
    event_sql = f"""
    SELECT
        c,
        t,
        CAST({outcome_col} AS DOUBLE) AS outcome,
        CAST({x_col} AS DOUBLE) AS x_val
    FROM analysis_panel
    WHERE c IN ({firm_a}, {firm_b})
      AND (CAST({outcome_col} AS DOUBLE) IS NOT NULL OR CAST({x_col} AS DOUBLE) IS NOT NULL)
    ORDER BY c, t
    """
    EVENT_PANEL = CON.sql(event_sql).df()
    EVENT_PANEL["firm_label"] = EVENT_PANEL["c"].map({firm_a: firm_a_label, firm_b: firm_b_label})
    if EVENT_PANEL.empty:
        print("Top pair has no non-null outcome observations in the requested event window.")
    else:
        EVENT_STUDY_SERIES = EVENT_PANEL.sort_values(["firm_label", "t"])
        firm_styles = {
            firm_a_label: {"color": "tab:blue", "event_t": event_t_a},
            firm_b_label: {"color": "tab:orange", "event_t": event_t_b},
        }
        outcome_plot_df = EVENT_STUDY_SERIES.dropna(subset=["outcome"])
        if outcome_plot_df.empty:
            print("Top pair has no non-null outcome observations in the requested event window.")
        else:
            fig, ax = plt.subplots(figsize=(9, 5))
            for label, sub in outcome_plot_df.groupby("firm_label", sort=False):
                style = firm_styles[label]
                ax.plot(sub["t"], sub["outcome"], marker="o", linewidth=2, color=style["color"], label=label)
                ax.axvline(
                    style["event_t"],
                    color=style["color"],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    label=f"{label} event ({style['event_t']})",
                )
            ax.set_xlabel("Calendar year (t)")
            ax.set_ylabel(outcome_col)
            ax.set_title("Top pair outcome paths (real time)")
            ax.legend(frameon=False)
            ax.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()

        x_plot_df = EVENT_STUDY_SERIES.dropna(subset=["x_val"])
        if x_plot_df.empty:
            print("Top pair has no non-null x observations in the requested event window.")
        else:
            fig, ax = plt.subplots(figsize=(9, 5))
            for label, sub in x_plot_df.groupby("firm_label", sort=False):
                style = firm_styles[label]
                ax.plot(sub["t"], sub["x_val"], marker="o", linewidth=2, color=style["color"], label=label)
                ax.axvline(
                    style["event_t"],
                    color=style["color"],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    label=f"{label} event ({style['event_t']})",
                )
            ax.set_xlabel("Calendar year (t)")
            ax.set_ylabel(x_col)
            ax.set_title("Top pair x paths (real time)")
            ax.legend(frameon=False)
            ax.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()

if OUTPUT_PATH is not None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOP_PAIRS.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
