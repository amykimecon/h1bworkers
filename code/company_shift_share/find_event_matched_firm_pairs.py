"""Find treated/control firm pairs matched at event time.

Notebook-friendly script:
- no CLI args, no main()
- edit top-level parameter block and re-run

Pair design (calendar-time matching):
1) Assign each firm its dominant university by transition share (with min share cutoff).
2) Define large university shocks from IPEDS growth.
3) Treated firm-year is (firm c, calendar year t) when c's dominant university has an isolated large shock at t.
4) Control firm-year at same calendar t has dominant university with no large shock in +/- window around t.
5) Match treated/control on similar firm size and similar masters_opt_hires at t-1 and t-2.
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


def _find_col(con: ddb.DuckDBPyConnection, preferred: str | None, candidates: list[str], table: str) -> str:
    cols = con.sql(f"PRAGMA table_info('{table}')").df()["name"].tolist()
    if preferred and preferred in cols:
        return preferred
    for candidate in candidates:
        if candidate in cols:
            return candidate
    raise ValueError(f"No usable column found in {table}. Candidates={candidates}; available={cols}")


def _esc(p: Path) -> str:
    return str(p).replace("'", "''")


def _build_sql(
    *,
    share_col: str,
    g_col: str,
    size_col: str,
    top_n: int,
    min_share: float,
    min_share_num: int,
    min_level_change: float,
    min_growth_ratio: float,
    window_min_level_change: float,
    window_min_growth_ratio: float,
    allow_new_program: bool,
    min_calendar_t: int,
    event_exclusion_window: int,
    size_log_tol: float,
    x_log_tol: float,
    size_change_log_tol: float,
    x_change_log_tol: float,
    min_firm_size: float,
    require_nonzero_x: bool,
    exclude_name_regex: str,
    opt_hire_window: int,
) -> str:
    share_num_col = "n_transitions_full" if share_col == "share_ck_full" else "n_transitions"
    new_program_clause = "FALSE"
    if allow_new_program:
        new_program_clause = f"(COALESCE(prev_g, 0) <= 0 AND g_t >= {float(min_level_change)})"
    window_new_program_clause = "FALSE"
    if allow_new_program:
        window_new_program_clause = f"(COALESCE(prev_g, 0) <= 0 AND g_t >= {float(window_min_level_change)})"

    x_nonzero_clause = ""
    if require_nonzero_x:
        x_nonzero_clause = """
          AND tf.x_t1 >= 1
          AND tf.x_t2 >= 1
          AND cf.x_t1 >= 1
          AND cf.x_t2 >= 1
        """

    return f"""
    WITH firm_name_filter AS (
        SELECT
            ap.c
        FROM (SELECT DISTINCT c FROM analysis_panel) ap
        LEFT JOIN firm_meta fm USING (c)
        WHERE fm.firm_name IS NULL
           OR NOT regexp_matches(LOWER(fm.firm_name), '{exclude_name_regex}')
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
    firm_top_univ_filtered AS (
        SELECT ftu.*
        FROM firm_top_univ ftu
        JOIN firm_name_filter fnf USING (c)
    ),
    univ_growth AS (
        SELECT
            k,
            t,
            CAST({g_col} AS DOUBLE) AS g_t,
            LAG(CAST({g_col} AS DOUBLE)) OVER (PARTITION BY k ORDER BY t) AS prev_g
        FROM ipeds_unit_growth
        WHERE {g_col} IS NOT NULL
          AND t >= {int(min_calendar_t) - 1}
    ),
    univ_events AS (
        SELECT
            k,
            t AS calendar_t,
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
        WHERE calendar_t >= {int(min_calendar_t)}
          AND (
            (
                prev_g > 0
                AND growth_ratio >= {float(min_growth_ratio)}
                AND delta_g >= {float(min_level_change)}
            )
            OR is_new_program = 1
          )
    ),
    univ_events_window_qualified AS (
        SELECT
            *,
            CASE WHEN {window_new_program_clause} THEN 1 ELSE 0 END AS is_window_new_program
        FROM univ_events
        WHERE calendar_t >= {int(min_calendar_t)}
          AND (
            (
                prev_g > 0
                AND growth_ratio >= {float(window_min_growth_ratio)}
                AND delta_g >= {float(window_min_level_change)}
            )
            OR ({window_new_program_clause})
          )
    ),
    treated_events_isolated AS (
        SELECT e.*
        FROM univ_events_qualified e
        WHERE NOT EXISTS (
            SELECT 1
            FROM univ_events_window_qualified e2
            WHERE e2.k = e.k
              AND ABS(e2.calendar_t - e.calendar_t) <= {int(event_exclusion_window)}
              AND e2.calendar_t <> e.calendar_t
        )
    ),
    treated_firm_events AS (
        SELECT
            ftu.c AS treated_firm,
            ftu.k AS treated_univ,
            ftu.share_val AS treated_share,
            ftu.share_num AS treated_share_num,
            te.calendar_t,
            te.calendar_t - 1 AS pre_t1_year,
            te.calendar_t - 2 AS pre_t2_year,
            te.delta_g AS treated_delta_g,
            te.growth_ratio AS treated_growth_ratio,
            te.is_new_program AS treated_is_new_program
        FROM firm_top_univ_filtered ftu
        JOIN treated_events_isolated te
          ON ftu.k = te.k
    ),
    treated_features AS (
        SELECT
            tfe.*,
            t1.{size_col} AS size_t1,
            t2.{size_col} AS size_t2,
            t1.masters_opt_hires AS x_t1,
            t2.masters_opt_hires AS x_t2
        FROM treated_firm_events tfe
        JOIN analysis_panel t1
          ON t1.c = tfe.treated_firm AND t1.t = tfe.pre_t1_year
        JOIN analysis_panel t2
          ON t2.c = tfe.treated_firm AND t2.t = tfe.pre_t2_year
        WHERE t1.{size_col} IS NOT NULL
          AND t2.{size_col} IS NOT NULL
          AND t1.{size_col} >= {float(min_firm_size)}
          AND t2.{size_col} >= {float(min_firm_size)}
    ),
    controls_pool AS (
        SELECT
            tf.*,
            cfu.c AS control_firm,
            cfu.k AS control_univ,
            cfu.share_val AS control_share,
            cfu.share_num AS control_share_num
        FROM treated_features tf
        JOIN firm_top_univ_filtered cfu
          ON cfu.c <> tf.treated_firm
         AND cfu.k <> tf.treated_univ
        WHERE NOT EXISTS (
            SELECT 1
            FROM univ_events_window_qualified ce
            WHERE ce.k = cfu.k
              AND ABS(ce.calendar_t - tf.calendar_t) <= {int(event_exclusion_window)}
        )
    ),
    pair_features AS (
        SELECT
            cp.*,
            c1.{size_col} AS control_size_t1,
            c2.{size_col} AS control_size_t2,
            c1.masters_opt_hires AS control_x_t1,
            c2.masters_opt_hires AS control_x_t2
        FROM controls_pool cp
        JOIN analysis_panel c1
          ON c1.c = cp.control_firm AND c1.t = cp.pre_t1_year
        JOIN analysis_panel c2
          ON c2.c = cp.control_firm AND c2.t = cp.pre_t2_year
        WHERE c1.{size_col} IS NOT NULL
          AND c2.{size_col} IS NOT NULL
          AND c1.{size_col} >= {float(min_firm_size)}
          AND c2.{size_col} >= {float(min_firm_size)}
    ),
    matched AS (
        SELECT
            pf.calendar_t AS t,
            pf.pre_t1_year,
            pf.pre_t2_year,
            pf.treated_firm AS firm_a,
            pf.control_firm AS firm_b,
            COALESCE(fma.firm_name, CAST(pf.treated_firm AS VARCHAR)) AS firm_a_name,
            fma.firm_city AS firm_a_city,
            fma.firm_state AS firm_a_state,
            COALESCE(fmb.firm_name, CAST(pf.control_firm AS VARCHAR)) AS firm_b_name,
            fmb.firm_city AS firm_b_city,
            fmb.firm_state AS firm_b_state,
            pf.treated_univ AS univ_a,
            pf.control_univ AS univ_b,
            COALESCE(uma.univ_name, pf.treated_univ) AS univ_a_name,
            COALESCE(umb.univ_name, pf.control_univ) AS univ_b_name,
            pf.treated_share AS share_a,
            pf.control_share AS share_b,
            pf.treated_share_num AS share_num_a,
            pf.control_share_num AS share_num_b,
            pf.treated_delta_g AS delta_g_a,
            pf.treated_growth_ratio AS growth_ratio_a,
            pf.treated_is_new_program AS is_new_program_a,
            pf.size_t1 AS size_a_t1,
            pf.size_t2 AS size_a_t2,
            pf.control_size_t1 AS size_b_t1,
            pf.control_size_t2 AS size_b_t2,
            pf.x_t1 AS x_a_t1,
            pf.x_t2 AS x_a_t2,
            pf.control_x_t1 AS x_b_t1,
            pf.control_x_t2 AS x_b_t2,
            ABS(LN(pf.size_t1 + 1) - LN(pf.control_size_t1 + 1)) AS size_diff_t1,
            ABS(LN(pf.size_t2 + 1) - LN(pf.control_size_t2 + 1)) AS size_diff_t2,
            ABS(LN(pf.x_t1 + 1) - LN(pf.control_x_t1 + 1)) AS x_diff_t1,
            ABS(LN(pf.x_t2 + 1) - LN(pf.control_x_t2 + 1)) AS x_diff_t2,
            (LN(pf.size_t1 + 1) - LN(pf.size_t2 + 1)) AS size_change_a,
            (LN(pf.control_size_t1 + 1) - LN(pf.control_size_t2 + 1)) AS size_change_b,
            (LN(pf.x_t1 + 1) - LN(pf.x_t2 + 1)) AS x_change_a,
            (LN(pf.control_x_t1 + 1) - LN(pf.control_x_t2 + 1)) AS x_change_b
        FROM pair_features pf
        LEFT JOIN firm_meta fma ON pf.treated_firm = fma.c
        LEFT JOIN firm_meta fmb ON pf.control_firm = fmb.c
        LEFT JOIN univ_meta uma ON pf.treated_univ = uma.k
        LEFT JOIN univ_meta umb ON pf.control_univ = umb.k
        WHERE ABS(LN(pf.size_t1 + 1) - LN(pf.control_size_t1 + 1)) <= {float(size_log_tol)}
          AND ABS(LN(pf.size_t2 + 1) - LN(pf.control_size_t2 + 1)) <= {float(size_log_tol)}
          AND ABS(LN(pf.x_t1 + 1) - LN(pf.control_x_t1 + 1)) <= {float(x_log_tol)}
          AND ABS(LN(pf.x_t2 + 1) - LN(pf.control_x_t2 + 1)) <= {float(x_log_tol)}
          AND (
              (LN(pf.size_t1 + 1) - LN(pf.size_t2 + 1))
              * (LN(pf.control_size_t1 + 1) - LN(pf.control_size_t2 + 1))
          ) >= 0
          AND ABS(
              (LN(pf.size_t1 + 1) - LN(pf.size_t2 + 1))
              - (LN(pf.control_size_t1 + 1) - LN(pf.control_size_t2 + 1))
          ) <= {float(size_change_log_tol)}
          AND (
              (LN(pf.x_t1 + 1) - LN(pf.x_t2 + 1))
              * (LN(pf.control_x_t1 + 1) - LN(pf.control_x_t2 + 1))
          ) >= 0
          AND ABS(
              (LN(pf.x_t1 + 1) - LN(pf.x_t2 + 1))
              - (LN(pf.control_x_t1 + 1) - LN(pf.control_x_t2 + 1))
          ) <= {float(x_change_log_tol)}
          AND (
              SELECT COUNT(*)
              FROM analysis_panel ap
              WHERE ap.c = pf.treated_firm
                AND ap.t BETWEEN pf.calendar_t - {int(opt_hire_window)} AND pf.calendar_t + {int(opt_hire_window)}
                AND COALESCE(ap.masters_opt_hires, 0) >= 1
          ) = {int(2 * opt_hire_window + 1)}
          AND (
              SELECT COUNT(*)
              FROM analysis_panel ap
              WHERE ap.c = pf.control_firm
                AND ap.t BETWEEN pf.calendar_t - {int(opt_hire_window)} AND pf.calendar_t + {int(opt_hire_window)}
                AND COALESCE(ap.masters_opt_hires, 0) >= 1
          ) = {int(2 * opt_hire_window + 1)}
          {x_nonzero_clause}
    )
    SELECT *
    FROM matched
    ORDER BY t ASC, GREATEST(size_a_t1, size_b_t1) DESC, (size_diff_t1 + size_diff_t2 + x_diff_t1 + x_diff_t2) ASC
    LIMIT {int(top_n)}
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

SHARE_COL = None
G_COL = None
TOP_N = None
MIN_SHARE = None
MIN_SHARE_NUM = None
MIN_LEVEL_CHANGE = None
MIN_GROWTH_RATIO = None
WINDOW_MIN_LEVEL_CHANGE = None
WINDOW_MIN_GROWTH_RATIO = None
ALLOW_NEW_PROGRAM = None
MIN_EVENT_T = None
MIN_CALENDAR_T = None
EVENT_EXCLUSION_WINDOW = None
SIZE_LOG_TOL = None
X_LOG_TOL = None
SIZE_CHANGE_LOG_TOL = None
X_CHANGE_LOG_TOL = None
MIN_FIRM_SIZE = None
REQUIRE_NONZERO_X = None
EXCLUDE_NAME_REGEX = None
OPT_HIRE_WINDOW = None
OUTCOME_COL = None
X_COL = None
PLOT_WINDOW = None
PAIR_RANDOM_SEED = None

cfg = load_config(CONFIG_PATH or DEFAULT_CONFIG_PATH)
paths_cfg = get_cfg_section(cfg, "paths")
pairs_cfg = get_cfg_section(cfg, "find_event_matched_firm_pairs")
reg_cfg = get_cfg_section(cfg, "shift_share_regressions")

analysis_panel_path = ANALYSIS_PANEL_PATH or _resolve_path(paths_cfg, "analysis_panel")
instrument_components_path = INSTRUMENT_COMPONENTS_PATH or _resolve_path(paths_cfg, "instrument_components")
employer_crosswalk_path = EMPLOYER_CROSSWALK_PATH or _resolve_path(paths_cfg, "employer_crosswalk")
three_way_crosswalk_path = THREE_WAY_CROSSWALK_PATH or _resolve_path(paths_cfg, "revelio_ipeds_foia_inst_crosswalk")
transition_shares_path = TRANSITION_SHARES_PATH
growth_path = GROWTH_PANEL_PATH

share_col = SHARE_COL or pairs_cfg.get("share_col", "share_ck_full")
g_col = G_COL or pairs_cfg.get("g_col", "g_kt")
top_n = int(TOP_N if TOP_N is not None else pairs_cfg.get("top_n", 100))
min_share = float(MIN_SHARE if MIN_SHARE is not None else pairs_cfg.get("min_share", 0.1))
min_share_num = int(MIN_SHARE_NUM if MIN_SHARE_NUM is not None else pairs_cfg.get("min_share_num", 5))
min_level_change = float(MIN_LEVEL_CHANGE if MIN_LEVEL_CHANGE is not None else pairs_cfg.get("min_level_change", 40.0))
min_growth_ratio = float(MIN_GROWTH_RATIO if MIN_GROWTH_RATIO is not None else pairs_cfg.get("min_growth_ratio", 0.1))
window_min_level_change = float(
    WINDOW_MIN_LEVEL_CHANGE
    if WINDOW_MIN_LEVEL_CHANGE is not None
    else pairs_cfg.get("window_min_level_change", min_level_change / 1.0)
)
window_min_growth_ratio = float(
    WINDOW_MIN_GROWTH_RATIO
    if WINDOW_MIN_GROWTH_RATIO is not None
    else pairs_cfg.get("window_min_growth_ratio", min_growth_ratio / 2.0)
)
allow_new_program = bool(ALLOW_NEW_PROGRAM if ALLOW_NEW_PROGRAM is not None else pairs_cfg.get("allow_new_program", True))
_min_cal_t_cfg = pairs_cfg.get("min_calendar_t", pairs_cfg.get("min_event_t", 2012))
min_calendar_t = int(
    MIN_CALENDAR_T
    if MIN_CALENDAR_T is not None
    else (MIN_EVENT_T if MIN_EVENT_T is not None else _min_cal_t_cfg)
)
event_exclusion_window = int(
    EVENT_EXCLUSION_WINDOW
    if EVENT_EXCLUSION_WINDOW is not None
    else pairs_cfg.get("event_exclusion_window", 3)
)
size_log_tol = float(SIZE_LOG_TOL if SIZE_LOG_TOL is not None else pairs_cfg.get("size_log_tol", 1))
x_log_tol = float(X_LOG_TOL if X_LOG_TOL is not None else pairs_cfg.get("x_log_tol", 1.0))
size_change_log_tol = float(
    SIZE_CHANGE_LOG_TOL
    if SIZE_CHANGE_LOG_TOL is not None
    else pairs_cfg.get("size_change_log_tol", size_log_tol)
)
x_change_log_tol = float(
    X_CHANGE_LOG_TOL
    if X_CHANGE_LOG_TOL is not None
    else pairs_cfg.get("x_change_log_tol", x_log_tol)
)
min_firm_size = float(MIN_FIRM_SIZE if MIN_FIRM_SIZE is not None else pairs_cfg.get("min_firm_size", 5000.0))
require_nonzero_x = bool(
    REQUIRE_NONZERO_X if REQUIRE_NONZERO_X is not None else pairs_cfg.get("require_nonzero_x", False)
)
opt_hire_window = int(
    OPT_HIRE_WINDOW if OPT_HIRE_WINDOW is not None else pairs_cfg.get("opt_hire_window", event_exclusion_window)
)
exclude_name_regex = EXCLUDE_NAME_REGEX or pairs_cfg.get(
    "exclude_name_regex",
    "(hospital|university|college|school|health)",
)
default_outcome_col = f"{reg_cfg.get('outcome_prefix', 'y_cst_lag')}0"
outcome_col = OUTCOME_COL or pairs_cfg.get("outcome_col", default_outcome_col)
x_col = X_COL or pairs_cfg.get("x_col", reg_cfg.get("dependent", "masters_opt_hires"))
plot_window = int(PLOT_WINDOW if PLOT_WINDOW is not None else pairs_cfg.get("plot_window", 10))
pair_random_seed = PAIR_RANDOM_SEED if PAIR_RANDOM_SEED is not None else pairs_cfg.get("pair_random_seed")

CON = ddb.connect()
CON.sql(f"CREATE OR REPLACE VIEW analysis_panel AS SELECT * FROM read_parquet('{_esc(analysis_panel_path)}')")
CON.sql(f"CREATE OR REPLACE VIEW instrument_components_input AS SELECT * FROM read_parquet('{_esc(instrument_components_path)}')")
CON.sql(f"CREATE OR REPLACE VIEW employer_crosswalk_input AS SELECT * FROM read_parquet('{_esc(employer_crosswalk_path)}')")
CON.sql(f"CREATE OR REPLACE VIEW three_way_crosswalk_input AS SELECT * FROM read_parquet('{_esc(three_way_crosswalk_path)}')")

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
outcome_col = _find_col(CON, outcome_col, ["y_cst_lag0", "y", "masters_opt_hires"], "analysis_panel")
x_col = _find_col(CON, x_col, ["masters_opt_hires", "valid_masters_opt_hires"], "analysis_panel")
g_col = _find_col(CON, g_col, ["g_kt", "g_kt_all", "g_kt_intl"], "ipeds_unit_growth")
SQL = _build_sql(
    share_col=share_col,
    g_col=g_col,
    size_col=size_col,
    top_n=top_n,
    min_share=min_share,
    min_share_num=min_share_num,
    min_level_change=min_level_change,
    min_growth_ratio=min_growth_ratio,
    window_min_level_change=window_min_level_change,
    window_min_growth_ratio=window_min_growth_ratio,
    allow_new_program=allow_new_program,
    min_calendar_t=min_calendar_t,
    event_exclusion_window=event_exclusion_window,
    size_log_tol=size_log_tol,
    x_log_tol=x_log_tol,
    size_change_log_tol=size_change_log_tol,
    x_change_log_tol=x_change_log_tol,
    min_firm_size=min_firm_size,
    require_nonzero_x=require_nonzero_x,
    exclude_name_regex=exclude_name_regex,
    opt_hire_window=opt_hire_window,
)

PAIR_CANDIDATES = CON.sql(SQL).df()
if PAIR_CANDIDATES.empty:
    print("No pairs found with the current thresholds.")
else:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print(PAIR_CANDIDATES.to_string(index=False))
    if pair_random_seed is None:
        selected_pair = PAIR_CANDIDATES.sample(n=1).iloc[0]
    else:
        selected_pair = PAIR_CANDIDATES.sample(n=1, random_state=int(pair_random_seed)).iloc[0]
    print("Selected pair for plots:")
    print(selected_pair.to_frame().T.to_string(index=False))

    t0 = int(selected_pair["t"])
    firm_a = int(selected_pair["firm_a"])
    firm_b = int(selected_pair["firm_b"])
    univ_a = str(selected_pair["univ_a"])
    univ_b = str(selected_pair["univ_b"])
    firm_a_label = str(selected_pair["firm_a_name"])
    firm_b_label = str(selected_pair["firm_b_name"])
    univ_a_label = str(selected_pair["univ_a_name"])
    univ_b_label = str(selected_pair["univ_b_name"])
    firm_a_label_plot = f"{firm_a_label} (treated)"
    firm_b_label_plot = f"{firm_b_label} (control)"
    univ_a_label_plot = f"{univ_a_label} (treated)"
    univ_b_label_plot = f"{univ_b_label} (control)"

    yx_df = CON.sql(
        f"""
        SELECT
            c,
            t,
            CAST({outcome_col} AS DOUBLE) AS y_val,
            CAST({x_col} AS DOUBLE) AS x_val
        FROM analysis_panel
        WHERE c IN ({firm_a}, {firm_b})
          AND t BETWEEN {t0 - plot_window} AND {t0 + plot_window}
        ORDER BY c, t
        """
    ).df()
    yx_df["firm_label"] = yx_df["c"].map({firm_a: firm_a_label_plot, firm_b: firm_b_label_plot})

    g_df = CON.sql(
        f"""
        SELECT
            k,
            t,
            CAST({g_col} AS DOUBLE) AS g_val
        FROM ipeds_unit_growth
        WHERE k IN ('{univ_a.replace("'", "''")}', '{univ_b.replace("'", "''")}')
          AND t BETWEEN {t0 - plot_window} AND {t0 + plot_window}
        ORDER BY k, t
        """
    ).df()
    g_df["univ_label"] = g_df["k"].map({univ_a: univ_a_label_plot, univ_b: univ_b_label_plot})

    firm_styles = {firm_a_label_plot: "tab:blue", firm_b_label_plot: "tab:orange"}
    univ_styles = {univ_a_label_plot: "tab:blue", univ_b_label_plot: "tab:orange"}

    y_plot = yx_df.dropna(subset=["y_val"])
    if y_plot.empty:
        print("No non-null y observations for selected pair in plotting window.")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, sub in y_plot.groupby("firm_label", sort=False):
            ax.plot(sub["t"], sub["y_val"], marker="o", linewidth=2, color=firm_styles.get(label), label=label)
        ax.axvline(t0, color="black", linestyle="--", linewidth=1.5, label=f"event t={t0}")
        ax.set_xlabel("Calendar year (t)")
        ax.set_ylabel(outcome_col)
        ax.set_title("Matched pair: y over time")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    x_plot = yx_df.dropna(subset=["x_val"])
    if x_plot.empty:
        print("No non-null x observations for selected pair in plotting window.")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, sub in x_plot.groupby("firm_label", sort=False):
            ax.plot(sub["t"], sub["x_val"], marker="o", linewidth=2, color=firm_styles.get(label), label=label)
        ax.axvline(t0, color="black", linestyle="--", linewidth=1.5, label=f"event t={t0}")
        ax.set_xlabel("Calendar year (t)")
        ax.set_ylabel(x_col)
        ax.set_title("Matched pair: x over time")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    g_plot = g_df.dropna(subset=["g_val"])
    if g_plot.empty:
        print("No non-null g observations for selected universities in plotting window.")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, sub in g_plot.groupby("univ_label", sort=False):
            ax.plot(sub["t"], sub["g_val"], marker="o", linewidth=2, color=univ_styles.get(label), label=label)
        ax.axvline(t0, color="black", linestyle="--", linewidth=1.5, label=f"event t={t0}")
        ax.set_xlabel("Calendar year (t)")
        ax.set_ylabel(g_col)
        ax.set_title("Matched pair universities: g over time")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()

if OUTPUT_PATH is not None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAIR_CANDIDATES.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
