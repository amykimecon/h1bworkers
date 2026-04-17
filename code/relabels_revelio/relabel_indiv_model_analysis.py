"""Downstream analysis panel and plots for relabel_indiv_model Revelio extracts."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb as ddb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import v2 for IPEDS-based treated→control matching.
try:
    import f1_foia.econ_relabels_opt_usage_v2 as _v2
except ModuleNotFoundError:
    import importlib.util as _ilu
    _here = Path(__file__).resolve()
    for _cand in [_here.parents[1], *_here.parents[1].parents]:
        _p = _cand / "f1_foia" / "econ_relabels_opt_usage_v2.py"
        if _p.exists():
            _spec = _ilu.spec_from_file_location("_v2_mod", _p)
            _v2 = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_v2)
            break
    else:
        _v2 = None  # type: ignore[assignment]

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

print(f"[relabel_indiv_model_analysis] Using config: {cfg.ACTIVE_CONFIG_PATH}")
print(f"[relabel_indiv_model_analysis] run_tag={cfg.RUN_TAG} horizons={cfg.ANALYSIS_HORIZONS}")

sns.set(style="whitegrid")
plt.rcParams.update({"font.size": 12})

PANEL_COLUMNS = [
    "user_id",
    "unitid",
    "school_name",
    "school_name_clean",
    "relabel_year",
    "relabel_type",
    "event_rsid",
    "grad_year",
    "cohort_t",
    "horizon_years",
    "target_year",
    "eval_year",
    "latest_available_year",
    "target_year_observed",
    "used_latest_avail",
    "still_in_us",
]

EVENT_STUDY_COLUMNS = [
    "horizon_years",
    "cohort_t",
    "n_school_events",
    "n_users",
    "still_in_us_mean",
    "still_in_us_se",
    "still_in_us_ci_low",
    "still_in_us_ci_high",
    "used_latest_avail_share",
    "target_year_observed_share",
    "eval_year_min",
    "eval_year_max",
]

REGRESSION_EVENT_STUDY_COLUMNS = [
    "horizon_years",
    "cohort_t",
    "is_reference",
    "reference_cohort_t",
    "n_school_events",
    "n_users",
    "n_schools",
    "coef",
    "se",
    "ci_low",
    "ci_high",
]

SCHOOL_EVENT_TIME_COLUMNS = [
    "unitid",
    "school_name",
    "school_name_clean",
    "relabel_year",
    "relabel_type",
    "event_rsid",
    "cohort_t",
    "horizon_years",
    "n_users",
    "still_in_us_mean",
    "used_latest_avail_share",
    "target_year_observed_share",
    "eval_year_min",
    "eval_year_max",
]


def _elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _parquet_has_column(path: Path, column_name: str) -> bool:
    con = ddb.connect()
    try:
        cols = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_escape_sql_literal(str(path))}')"
        ).df()["column_name"].astype(str)
        return column_name in set(cols.tolist())
    finally:
        con.close()


def _prepare_output_path(path_str: str) -> Path:
    out_path = Path(path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not cfg.BUILD_OVERWRITE:
        raise FileExistsError(f"Output exists and overwrite=false: {out_path}")
    return out_path


def _analysis_panel_path() -> Path:
    if cfg.ANALYSIS_PANEL_PARQUET:
        return _prepare_output_path(cfg.ANALYSIS_PANEL_PARQUET)
    fallback = Path(cfg.MATCHED_EDUCATION_PARQUET).resolve().with_name(
        f"relabel_indiv_model_analysis_panel_{cfg.RUN_TAG}.parquet"
    )
    return _prepare_output_path(str(fallback))


def _analysis_event_study_path() -> Path:
    if cfg.ANALYSIS_EVENT_STUDY_PARQUET:
        return _prepare_output_path(cfg.ANALYSIS_EVENT_STUDY_PARQUET)
    fallback = Path(cfg.MATCHED_EDUCATION_PARQUET).resolve().with_name(
        f"relabel_indiv_model_analysis_event_study_{cfg.RUN_TAG}.parquet"
    )
    return _prepare_output_path(str(fallback))


def _analysis_regression_event_study_path() -> Path:
    if cfg.ANALYSIS_REGRESSION_EVENT_STUDY_PARQUET:
        return _prepare_output_path(cfg.ANALYSIS_REGRESSION_EVENT_STUDY_PARQUET)
    fallback = Path(cfg.MATCHED_EDUCATION_PARQUET).resolve().with_name(
        f"relabel_indiv_model_analysis_regression_event_study_{cfg.RUN_TAG}.parquet"
    )
    return _prepare_output_path(str(fallback))


def _analysis_school_event_time_path() -> Path:
    if cfg.ANALYSIS_SCHOOL_EVENT_TIME_PARQUET:
        return _prepare_output_path(cfg.ANALYSIS_SCHOOL_EVENT_TIME_PARQUET)
    fallback = Path(cfg.MATCHED_EDUCATION_PARQUET).resolve().with_name(
        f"relabel_indiv_model_school_event_time_{cfg.RUN_TAG}.parquet"
    )
    return _prepare_output_path(str(fallback))


def _analysis_output_dir() -> Path:
    if cfg.ANALYSIS_OUTPUT_DIR:
        out_dir = Path(cfg.ANALYSIS_OUTPUT_DIR)
    else:
        out_dir = _analysis_panel_path().parent / f"relabel_indiv_model_plots_{cfg.RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_parquet(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    out_df = df.copy()
    for col in columns:
        if col not in out_df.columns:
            out_df[col] = pd.NA
    out_df = out_df.loc[:, columns]
    out_df.to_parquet(path, index=False)
    print(f"  wrote {path}")


def _save_figure(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return out_path


def _coerce_plot_frame(
    df: pd.DataFrame,
    cols: list[str],
    *,
    sort_col: str,
) -> pd.DataFrame:
    """Convert plotting columns to plain numeric values before handing them to Matplotlib."""
    plot_df = df.loc[:, cols].copy()
    for col in cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    return plot_df.dropna(subset=cols).sort_values(sort_col)


def _find_did_interaction_param(
    params: pd.Series,
    cohort_t: int,
    reference_event_time: int,
) -> str | None:
    term = f"C(cohort_t, Treatment(reference={reference_event_time}))"
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


def _validate_inputs() -> tuple[Path, Path]:
    education_path = Path(cfg.MATCHED_EDUCATION_PARQUET)
    positions_path = Path(cfg.MATCHED_POSITIONS_PARQUET)
    if not education_path.exists():
        raise FileNotFoundError(f"Missing matched education parquet: {education_path}")
    if not positions_path.exists():
        raise FileNotFoundError(f"Missing matched positions parquet: {positions_path}")
    if not cfg.ANALYSIS_HORIZONS:
        raise ValueError("No analysis horizons configured.")
    return education_path, positions_path


def register_sources(
    con: ddb.DuckDBPyConnection,
    education_path: Path,
    positions_path: Path,
) -> None:
    if _parquet_has_column(education_path, "exclude_immediate_same_inst_phd_after_master_ind"):
        education_select = (
            f"SELECT * FROM read_parquet('{_escape_sql_literal(str(education_path))}')"
        )
    else:
        education_select = (
            "SELECT src.*, 0::INTEGER AS exclude_immediate_same_inst_phd_after_master_ind "
            f"FROM read_parquet('{_escape_sql_literal(str(education_path))}') AS src"
        )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW matched_education AS
        {education_select}
        """
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW matched_positions AS
        SELECT * FROM read_parquet('{_escape_sql_literal(str(positions_path))}')
        """
    )
def build_base_events(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 1: Building base user-event matches ─────────────────────────")

    if cfg.ANALYSIS_FIELD_FILTER:
        escaped = _escape_sql_literal(cfg.ANALYSIS_FIELD_FILTER.lower())
        field_filter_clause = (
            f"LOWER(COALESCE(degree_raw, '')) LIKE '%{escaped}%'"
            f" OR LOWER(COALESCE(field_raw, '')) LIKE '%{escaped}%'"
        )
        print(f"  field filter: degree_raw or field_raw LIKE '%{escaped}%'")
    else:
        field_filter_clause = "1=1"
        print("  field filter: none")

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW base_event_candidates AS
        SELECT
            CAST(unitid AS BIGINT) AS unitid,
            school_name,
            school_name_clean,
            CAST(relabel_year AS INTEGER) AS relabel_year,
            relabel_type,
            CAST(event_rsid AS BIGINT) AS event_rsid,
            CAST(user_id AS BIGINT) AS user_id,
            CAST(education_number AS BIGINT) AS education_number,
            TRY_CAST(ed_enddate AS DATE) AS ed_enddate,
            CAST(ed_end_year AS INTEGER) AS grad_year,
            CAST(ed_end_year AS INTEGER) - CAST(relabel_year AS INTEGER) AS cohort_t,
            COALESCE(CAST(exclude_immediate_same_inst_phd_after_master_ind AS INTEGER), 0)
                AS exclude_immediate_same_inst_phd_after_master
        FROM matched_education
        WHERE user_id IS NOT NULL
          AND relabel_year IS NOT NULL
          AND ed_end_year IS NOT NULL
          AND (
            {field_filter_clause}
          )
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW base_events AS
        WITH ranked AS (
            SELECT
                unitid,
                school_name,
                school_name_clean,
                relabel_year,
                relabel_type,
                event_rsid,
                user_id,
                education_number,
                ed_enddate,
                grad_year,
                cohort_t,
                ROW_NUMBER() OVER (
                    -- Match the old treated-panel behavior: keep at most one
                    -- user-event match per user_id × relabel_year.
                    PARTITION BY user_id, relabel_year
                    ORDER BY
                        ABS(grad_year - relabel_year),
                        CASE WHEN ed_enddate IS NULL THEN 1 ELSE 0 END,
                        ed_enddate DESC,
                        CASE WHEN education_number IS NULL THEN 1 ELSE 0 END,
                        education_number,
                        unitid,
                        school_name,
                        relabel_type
                ) AS match_rank
            FROM base_event_candidates
            WHERE exclude_immediate_same_inst_phd_after_master = 0
        )
        SELECT
            user_id,
            unitid,
            school_name,
            school_name_clean,
            relabel_year,
            relabel_type,
            event_rsid,
            grad_year,
            cohort_t
        FROM ranked
        WHERE match_rank = 1
        """
    )
    candidate_counts = con.sql(
        """
        SELECT
            COUNT(*) AS n_candidates,
            COUNT(*) FILTER (WHERE exclude_immediate_same_inst_phd_after_master = 1) AS n_excluded
        FROM base_event_candidates
        """
    ).fetchone()
    base_events = con.sql(
        """
        SELECT *
        FROM base_events
        ORDER BY relabel_year, cohort_t, school_name, user_id
        """
    ).df()

    print(f"  base user-event rows: {len(base_events):,}")
    if candidate_counts is not None:
        print(
            "  same-institution doctoral-continuation exclusions: "
            f"{int(candidate_counts[1] or 0):,} of {int(candidate_counts[0] or 0):,} candidate rows"
        )
    print(f"  unique users: {base_events['user_id'].nunique() if not base_events.empty else 0:,}")
    if not base_events.empty:
        print(f"  cohort_t range: {base_events['cohort_t'].min()} to {base_events['cohort_t'].max()}")
    print(f"  Step 1 done in {_elapsed(t0)}")
    return base_events


def build_positions_view(con: ddb.DuckDBPyConnection) -> None:
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW parsed_positions AS
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            LOWER(COALESCE(country, '')) AS country_lc,
            TRY_CAST(startdate AS DATE) AS startdate,
            TRY_CAST(enddate AS DATE) AS enddate
        FROM matched_positions
        WHERE user_id IS NOT NULL
          AND TRY_CAST(startdate AS DATE) IS NOT NULL
        """
    )


def get_latest_available_year(con: ddb.DuckDBPyConnection) -> int:
    t0 = time.time()
    print("\n── Step 2: Determining latest observed work-history year ────────────")

    latest_available_year = con.sql(
        """
        WITH years AS (
            SELECT EXTRACT(YEAR FROM startdate)::INTEGER AS obs_year
            FROM parsed_positions
            UNION ALL
            SELECT EXTRACT(YEAR FROM enddate)::INTEGER AS obs_year
            FROM parsed_positions
            WHERE enddate IS NOT NULL
        )
        SELECT MAX(obs_year) AS latest_available_year
        FROM years
        WHERE obs_year IS NOT NULL
        """
    ).fetchone()[0]

    if latest_available_year is None:
        raise ValueError("Could not determine latest available year from matched positions.")

    latest_available_year = int(latest_available_year)
    print(f"  latest available year: {latest_available_year}")
    print(f"  Step 2 done in {_elapsed(t0)}")
    return latest_available_year


def build_analysis_panel(con: ddb.DuckDBPyConnection, latest_available_year: int) -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 3: Building still-in-US horizon panel ───────────────────────")

    horizons_sql = ",\n            ".join(f"({int(h)})" for h in cfg.ANALYSIS_HORIZONS)
    # target_year is horizon_years after *graduation* (not after the relabel event).
    # Using relabel_year + horizon_years would evaluate the outcome at different points
    # relative to graduation depending on cohort_t, causing spurious dips where
    # cohort_t == horizon_years (eval year lands exactly on grad year).
    if cfg.ANALYSIS_CAP_TO_LATEST_AVAILABLE_YEAR:
        eval_year_expr = f"LEAST(be.grad_year + h.horizon_years, {latest_available_year})"
        used_latest_expr = f"CASE WHEN be.grad_year + h.horizon_years > {latest_available_year} THEN 1 ELSE 0 END"
    else:
        eval_year_expr = "be.grad_year + h.horizon_years"
        used_latest_expr = "0"

    panel = con.sql(
        f"""
        WITH horizons (horizon_years) AS (
            VALUES
            {horizons_sql}
        ),
        event_horizons AS (
            SELECT
                be.user_id,
                be.unitid,
                be.school_name,
                be.school_name_clean,
                be.relabel_year,
                be.relabel_type,
                be.event_rsid,
                be.grad_year,
                be.cohort_t,
                h.horizon_years,
                be.grad_year + h.horizon_years AS target_year,
                {eval_year_expr} AS eval_year,
                {latest_available_year} AS latest_available_year,
                CASE
                    WHEN be.grad_year + h.horizon_years <= {latest_available_year} THEN 1
                    ELSE 0
                END AS target_year_observed,
                {used_latest_expr} AS used_latest_avail
            FROM base_events AS be
            CROSS JOIN horizons AS h
        ),
        analysis_years AS (
            SELECT DISTINCT
                eval_year
            FROM event_horizons
        ),
        us_presence AS (
            SELECT
                p.user_id,
                ay.eval_year,
                MAX(
                    CASE WHEN p.country_lc = 'united states' THEN 1 ELSE 0 END
                ) AS still_in_us
            FROM analysis_years AS ay
            JOIN parsed_positions AS p
              ON p.startdate <= MAKE_DATE(ay.eval_year, 12, 31)
             AND (p.enddate IS NULL OR p.enddate >= MAKE_DATE(ay.eval_year, 1, 1))
            GROUP BY
                p.user_id,
                ay.eval_year
        )
        SELECT
            eh.user_id,
            eh.unitid,
            eh.school_name,
            eh.school_name_clean,
            eh.relabel_year,
            eh.relabel_type,
            eh.event_rsid,
            eh.grad_year,
            eh.cohort_t,
            eh.horizon_years,
            eh.target_year,
            eh.eval_year,
            eh.latest_available_year,
            eh.target_year_observed,
            eh.used_latest_avail,
            COALESCE(up.still_in_us, 0) AS still_in_us
        FROM event_horizons AS eh
        LEFT JOIN us_presence AS up
          ON up.user_id = eh.user_id
         AND up.eval_year = eh.eval_year
        ORDER BY
            eh.user_id,
            eh.relabel_year,
            eh.horizon_years,
            eh.cohort_t,
            eh.school_name
        """
    ).df()

    int_cols = [
        "user_id",
        "unitid",
        "relabel_year",
        "event_rsid",
        "grad_year",
        "cohort_t",
        "horizon_years",
        "target_year",
        "eval_year",
        "latest_available_year",
        "target_year_observed",
        "used_latest_avail",
        "still_in_us",
    ]
    for col in int_cols:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("Int64")

    print(f"  panel rows: {len(panel):,}")
    print(f"  users in panel: {panel['user_id'].nunique() if not panel.empty else 0:,}")
    if not panel.empty:
        print(
            panel.groupby("horizon_years", dropna=False)["still_in_us"]
            .mean()
            .rename("still_in_us_mean")
            .round(3)
            .to_string()
        )
    print(f"  Step 3 done in {_elapsed(t0)}")
    return panel


def build_school_event_time(panel: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 4: Aggregating to school-event × cohort_t ───────────────────")

    if panel.empty:
        school_event_time = pd.DataFrame(columns=SCHOOL_EVENT_TIME_COLUMNS)
    else:
        school_event_time = (
            panel.groupby(
                [
                    "unitid",
                    "school_name",
                    "school_name_clean",
                    "relabel_year",
                    "relabel_type",
                    "event_rsid",
                    "cohort_t",
                    "horizon_years",
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                n_users=("user_id", "size"),
                still_in_us_mean=("still_in_us", "mean"),
                used_latest_avail_share=("used_latest_avail", "mean"),
                target_year_observed_share=("target_year_observed", "mean"),
                eval_year_min=("eval_year", "min"),
                eval_year_max=("eval_year", "max"),
            )
            .sort_values(["school_name", "relabel_year", "horizon_years", "cohort_t"])
            .reset_index(drop=True)
        )
        school_event_time = school_event_time.loc[:, SCHOOL_EVENT_TIME_COLUMNS]

    print(f"  school-event-time rows: {len(school_event_time):,}")
    print(f"  Step 4 done in {_elapsed(t0)}")
    return school_event_time


def build_event_study(school_event_time: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 5: Aggregating school-level event-study series ──────────────")

    if school_event_time.empty:
        event_study = pd.DataFrame(columns=EVENT_STUDY_COLUMNS)
    else:
        # Exclude cohorts where the target year is beyond the latest available data.
        # Those rows have no position data and resolve to still_in_us=0 via COALESCE,
        # which would artificially deflate the average for late cohort_t values.
        n_before = len(school_event_time)
        agg_df = school_event_time[school_event_time["target_year_observed_share"] > 0].copy()
        n_dropped = n_before - len(agg_df)
        if n_dropped:
            print(f"  dropped {n_dropped:,} school-event-time rows with target_year_observed_share=0 (future outcome)")
        event_study = (
            agg_df.groupby(["horizon_years", "cohort_t"], dropna=False, as_index=False)
            .agg(
                n_school_events=("unitid", "size"),
                n_users=("n_users", "sum"),
                still_in_us_mean=("still_in_us_mean", "mean"),
                still_in_us_se=("still_in_us_mean", "sem"),
                used_latest_avail_share=("used_latest_avail_share", "mean"),
                target_year_observed_share=("target_year_observed_share", "mean"),
                eval_year_min=("eval_year_min", "min"),
                eval_year_max=("eval_year_max", "max"),
            )
            .sort_values(["horizon_years", "cohort_t"])
            .reset_index(drop=True)
        )
        event_study["still_in_us_ci_low"] = (
            event_study["still_in_us_mean"] - 1.96 * event_study["still_in_us_se"]
        ).clip(lower=0)
        event_study["still_in_us_ci_high"] = (
            event_study["still_in_us_mean"] + 1.96 * event_study["still_in_us_se"]
        ).clip(upper=1)
        event_study = event_study.loc[:, EVENT_STUDY_COLUMNS]

    print(f"  event-study rows: {len(event_study):,}")
    print(f"  Step 5 done in {_elapsed(t0)}")
    return event_study


def build_regression_event_study(school_event_time: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    print("\n── Step 6: Running school-FE event-time regressions ─────────────────")

    if school_event_time.empty:
        regression_event_study = pd.DataFrame(columns=REGRESSION_EVENT_STUDY_COLUMNS)
        print("  school-event-time summary is empty; skipping regressions")
        print(f"  Step 6 done in {_elapsed(t0)}")
        return regression_event_study

    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        print(f"  statsmodels unavailable; skipping regressions ({exc})")
        print(f"  Step 6 done in {_elapsed(t0)}")
        return pd.DataFrame(columns=REGRESSION_EVENT_STUDY_COLUMNS)

    reference_event_time = int(cfg.ANALYSIS_REGRESSION_REFERENCE_EVENT_TIME)
    rows: list[dict[str, object]] = []

    # Exclude future-outcome rows before regression (same reason as in build_event_study).
    school_event_time = school_event_time[school_event_time["target_year_observed_share"] > 0].copy()

    for horizon in sorted(school_event_time["horizon_years"].dropna().astype(int).unique().tolist()):
        reg_df = (
            school_event_time.loc[school_event_time["horizon_years"].astype("Int64") == horizon]
            .copy()
            .sort_values(["school_name", "relabel_year", "cohort_t"])
            .reset_index(drop=True)
        )
        if reg_df.empty:
            continue
        reg_df["cohort_t"] = pd.to_numeric(reg_df["cohort_t"], errors="coerce").astype(int)
        reg_df["unitid"] = pd.to_numeric(reg_df["unitid"], errors="coerce").astype(int)
        reg_df["relabel_year"] = pd.to_numeric(reg_df["relabel_year"], errors="coerce").astype(int)
        reg_df["n_users"] = pd.to_numeric(reg_df["n_users"], errors="coerce").astype(int)
        reg_df["still_in_us_mean"] = pd.to_numeric(reg_df["still_in_us_mean"], errors="coerce").astype(float)
        # grad_year FE absorbs calendar-time trends; grad_year = relabel_year + cohort_t.
        reg_df["grad_year"] = reg_df["relabel_year"] + reg_df["cohort_t"]

        cohort_values = sorted(reg_df["cohort_t"].dropna().astype(int).unique().tolist())
        n_schools = reg_df["unitid"].dropna().nunique()
        if reference_event_time not in cohort_values:
            print(
                f"  horizon {horizon}: missing reference cohort_t={reference_event_time}; "
                "skipping regression"
            )
            continue
        if len(cohort_values) < 2 or n_schools < 2:
            print(
                f"  horizon {horizon}: insufficient variation for regression "
                f"(cohort_t={len(cohort_values)}, schools={n_schools})"
            )
            continue

        formula = (
            "still_in_us_mean ~ "
            f"C(cohort_t, Treatment(reference={reference_event_time})) + C(unitid) + C(grad_year)"
        )
        result = smf.ols(formula, data=reg_df).fit(
            cov_type="cluster",
            cov_kwds={"groups": reg_df["unitid"]},
        )

        cell_counts = (
            reg_df.loc[:, ["cohort_t", "n_users"]]
            .groupby("cohort_t", as_index=False)
            .agg(
                n_school_events=("cohort_t", "size"),
                n_users=("n_users", "sum"),
            )
        )
        count_lookup = {
            int(row.cohort_t): (int(row.n_school_events), int(row.n_users))
            for row in cell_counts.itertuples(index=False)
        }

        for cohort_t in cohort_values:
            n_school_events, n_users = count_lookup.get(int(cohort_t), (0, 0))
            if cohort_t == reference_event_time:
                coef = 0.0
                se = 0.0
            else:
                param_name = (
                    f"C(cohort_t, Treatment(reference={reference_event_time}))[T.{int(cohort_t)}]"
                )
                coef = float(result.params.get(param_name, float("nan")))
                se = float(result.bse.get(param_name, float("nan")))
            rows.append(
                {
                    "horizon_years": horizon,
                    "cohort_t": int(cohort_t),
                    "is_reference": int(cohort_t == reference_event_time),
                    "reference_cohort_t": reference_event_time,
                    "n_school_events": n_school_events,
                    "n_users": n_users,
                    "n_schools": int(n_schools),
                    "coef": coef,
                    "se": se,
                    "ci_low": coef - 1.96 * se,
                    "ci_high": coef + 1.96 * se,
                }
            )

    regression_event_study = pd.DataFrame(rows, columns=REGRESSION_EVENT_STUDY_COLUMNS)
    if not regression_event_study.empty:
        int_cols = [
            "horizon_years",
            "cohort_t",
            "is_reference",
            "reference_cohort_t",
            "n_school_events",
            "n_users",
            "n_schools",
        ]
        for col in int_cols:
            regression_event_study[col] = (
                pd.to_numeric(regression_event_study[col], errors="coerce").astype("Int64")
            )
        regression_event_study = regression_event_study.sort_values(
            ["horizon_years", "cohort_t"]
        ).reset_index(drop=True)

    print(f"  regression rows: {len(regression_event_study):,}")
    print(f"  reference cohort_t: {reference_event_time}")
    print(f"  Step 6 done in {_elapsed(t0)}")
    return regression_event_study


def plot_university_stay_rates(
    university_name: str,
    horizons: list[int] | None = None,
    panel_path: str | Path | None = None,
) -> None:
    """Plot still-in-US stay rates by graduation year for a specific university.

    X-axis: actual graduation year.
    Y-axis: mean still_in_US rate across users at each grad year.
    One line per horizon (default: 1, 3, 5 years after graduation).

    Only cohorts where the target year was observed in the data are included.

    Args:
        university_name: Substring match (case-insensitive) against school_name_clean
                         or school_name.  Prints matching names found if ambiguous.
        horizons:        Horizon years to plot (default [1, 3, 5]).
        panel_path:      Path to the analysis panel parquet.  Defaults to the
                         path derived from the active config.
    """
    if horizons is None:
        horizons = [1, 3, 5]

    # ── Load panel ────────────────────────────────────────────────────────────
    if panel_path is None:
        if cfg.ANALYSIS_PANEL_PARQUET:
            panel_path = Path(cfg.ANALYSIS_PANEL_PARQUET)
        else:
            panel_path = Path(cfg.MATCHED_EDUCATION_PARQUET).resolve().with_name(
                f"relabel_indiv_model_analysis_panel_{cfg.RUN_TAG}.parquet"
            )
    panel_path = Path(panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel parquet not found: {panel_path}\n"
            "Run main() first to build and save the panel."
        )
    panel = pd.read_parquet(panel_path)
    print(f"Loaded panel: {len(panel):,} rows from {panel_path.name}")

    # ── Match university name ─────────────────────────────────────────────────
    name_lower = university_name.lower()
    mask = panel["school_name_clean"].str.lower().str.contains(name_lower, na=False) | \
           panel["school_name"].str.lower().str.contains(name_lower, na=False)
    uni_df = panel[mask].copy()
    if uni_df.empty:
        available = sorted(panel["school_name_clean"].dropna().unique().tolist())
        print(f"No rows matched '{university_name}'. Available school names:")
        for n in available:
            print(f"  {n}")
        return

    matched_names = sorted(uni_df["school_name_clean"].dropna().unique().tolist())
    if len(matched_names) > 1:
        print(f"Matched {len(matched_names)} school_name_clean values:")
        for n in matched_names:
            print(f"  {n}")
    else:
        print(f"Matched school: {matched_names[0]}")
    print(f"Rows after name filter: {len(uni_df):,}")

    # ── Filter to requested horizons and observed outcomes ────────────────────
    uni_df = uni_df[uni_df["horizon_years"].isin(horizons)]
    uni_df = uni_df[uni_df["target_year_observed"] == 1]

    if uni_df.empty:
        print("No observed-outcome rows remain after horizon/observed filter.")
        return

    # ── Aggregate by grad_year × horizon ─────────────────────────────────────
    uni_df["grad_year"] = pd.to_numeric(uni_df["grad_year"], errors="coerce")
    uni_df["still_in_us"] = pd.to_numeric(uni_df["still_in_us"], errors="coerce")
    uni_df["horizon_years"] = pd.to_numeric(uni_df["horizon_years"], errors="coerce").astype(int)

    agg = (
        uni_df.groupby(["grad_year", "horizon_years"], as_index=False)
        .agg(
            still_in_us_mean=("still_in_us", "mean"),
            n_users=("user_id", "size"),
        )
        .sort_values(["horizon_years", "grad_year"])
    )

    print(f"\nAggregated rows: {len(agg)}")
    print(agg.to_string(index=False))

    # ── Relabel event years (for vertical lines) ──────────────────────────────
    relabel_years = sorted(
        pd.to_numeric(uni_df["relabel_year"], errors="coerce").dropna().astype(int).unique().tolist()
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    palette = sns.color_palette("deep", n_colors=len(horizons))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for color, horizon in zip(palette, sorted(horizons)):
        grp = agg[agg["horizon_years"] == horizon].sort_values("grad_year")
        if grp.empty:
            continue
        ax.plot(
            grp["grad_year"],
            grp["still_in_us_mean"],
            marker="o",
            linewidth=2,
            color=color,
            label=f"{horizon}yr after grad",
        )

    for i, ry in enumerate(relabel_years):
        ax.axvline(
            x=ry,
            linestyle=":",
            color="gray",
            linewidth=1.5,
            label="relabel event" if i == 0 else None,
        )

    title_name = matched_names[0] if len(matched_names) == 1 else university_name
    ax.set_title(title_name)
    ax.set_xlabel("Graduation year")
    ax.set_ylabel("Fraction still in US")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_n_users_event_study(school_event_time: pd.DataFrame, out_dir: Path) -> None:
    """Plot raw and school-FE regression of n_users by cohort_t.

    Uses a single horizon (the smallest available) per school-event to avoid
    double-counting users across horizons.  This makes the n_users count
    independent of outcome availability.
    """
    t0 = time.time()
    print("\n── Step 7: Writing n_users event-study plots ────────────────────────")

    if school_event_time.empty:
        print("  school-event-time is empty; skipping n_users plots")
        return

    # Collapse to one row per school × relabel_year × cohort_t by taking the
    # smallest horizon (n_users is the same across horizons before outcome filtering).
    min_horizon = int(
        pd.to_numeric(school_event_time["horizon_years"], errors="coerce").min()
    )
    base = school_event_time[
        school_event_time["horizon_years"].astype("Int64") == min_horizon
    ].copy()
    base["cohort_t"] = pd.to_numeric(base["cohort_t"], errors="coerce").astype(int)
    base["n_users"] = pd.to_numeric(base["n_users"], errors="coerce").astype(int)
    base["unitid"] = pd.to_numeric(base["unitid"], errors="coerce").astype(int)
    base["relabel_year"] = pd.to_numeric(base["relabel_year"], errors="coerce").astype(int)
    # grad_year FE absorbs calendar-time trends; grad_year = relabel_year + cohort_t.
    base["grad_year"] = base["relabel_year"] + base["cohort_t"]

    # ── Raw plot ──────────────────────────────────────────────────────────────
    raw = (
        base.groupby("cohort_t", as_index=False)
        .agg(n_users=("n_users", "sum"))
        .sort_values("cohort_t")
    )
    xticks = sorted(raw["cohort_t"].unique().tolist())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(raw["cohort_t"], raw["n_users"], color=sns.color_palette("deep")[0])
    ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Graduation year relative to relabel event")
    ax.set_ylabel("Number of users")
    ax.set_title("N users by graduation cohort (raw)")
    ax.set_xticks(xticks)
    fig.tight_layout()
    _save_figure(fig, out_dir, "n_users_by_cohort_raw")

    # ── School-FE regression plot ─────────────────────────────────────────────
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        print(f"  statsmodels unavailable; skipping n_users regression ({exc})")
        print(f"  Step 7 done in {_elapsed(t0)}")
        return

    reference_event_time = int(cfg.ANALYSIS_REGRESSION_REFERENCE_EVENT_TIME)
    cohort_values = sorted(base["cohort_t"].dropna().unique().tolist())
    n_schools = base["unitid"].dropna().nunique()

    if reference_event_time not in cohort_values or len(cohort_values) < 2 or n_schools < 2:
        print(
            f"  insufficient variation for n_users regression "
            f"(cohort_t={len(cohort_values)}, schools={n_schools})"
        )
        print(f"  Step 7 done in {_elapsed(t0)}")
        return

    formula = (
        "n_users ~ "
        f"C(cohort_t, Treatment(reference={reference_event_time})) + C(unitid) + C(grad_year)"
    )
    result = smf.ols(formula, data=base).fit(
        cov_type="cluster",
        cov_kwds={"groups": base["unitid"]},
    )

    reg_rows = []
    for ct in cohort_values:
        if ct == reference_event_time:
            coef, se = 0.0, 0.0
        else:
            param = f"C(cohort_t, Treatment(reference={reference_event_time}))[T.{ct}]"
            coef = float(result.params.get(param, float("nan")))
            se = float(result.bse.get(param, float("nan")))
        reg_rows.append({"cohort_t": ct, "coef": coef, "se": se})
    reg_df = pd.DataFrame(reg_rows).sort_values("cohort_t")
    reg_df["ci_low"] = reg_df["coef"] - 1.96 * reg_df["se"]
    reg_df["ci_high"] = reg_df["coef"] + 1.96 * reg_df["se"]

    color = sns.color_palette("deep")[1]
    fig, ax = plt.subplots(figsize=(9, 5))
    line_df = _coerce_plot_frame(reg_df, ["cohort_t", "coef"], sort_col="cohort_t")
    ax.plot(
        line_df["cohort_t"].to_numpy(dtype=float),
        line_df["coef"].to_numpy(dtype=float),
        marker="o",
        linewidth=2,
        color=color,
    )
    valid_ci = reg_df["se"].notna() & (reg_df["se"] > 0)
    if valid_ci.any():
        ci = _coerce_plot_frame(
            reg_df.loc[valid_ci],
            ["cohort_t", "ci_low", "ci_high"],
            sort_col="cohort_t",
        )
        ax.fill_between(
            ci["cohort_t"].to_numpy(dtype=float),
            ci["ci_low"].to_numpy(dtype=float),
            ci["ci_high"].to_numpy(dtype=float),
            alpha=0.25,
            color=color,
        )
    ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
    ax.axhline(y=0, linestyle=":", color="gray", linewidth=1)
    ax.set_xlabel("Graduation year relative to relabel event")
    ax.set_ylabel(
        f"Difference vs cohort_t={reference_event_time} (school + event-year FE, clustered by school)"
    )
    ax.set_title("N users by graduation cohort (school-FE regression)")
    ax.set_xticks(xticks)
    fig.tight_layout()
    _save_figure(fig, out_dir, "n_users_by_cohort_school_fe")

    print(f"  Step 7 done in {_elapsed(t0)}")


def plot_n_users_to_ipeds_ratio(school_event_time: pd.DataFrame, out_dir: Path) -> None:
    """Plot ratio of Revelio econ users to IPEDS econ graduates by cohort_t.

    Joins school_event_time (n_users per school × grad_year) against IPEDS
    completions (summed over all econ CIPs) by unitid × year.  Aggregates to
    cohort_t by averaging the school-level ratios.
    """
    t0 = time.time()
    print("\n── Step 8b: Writing n_users / IPEDS n_grads ratio plot ─────────────")

    completions_path = Path(cfg.IPEDS_COMPLETIONS_PARQUET)
    if not completions_path.exists():
        print(f"  IPEDS completions parquet not found: {completions_path}; skipping")
        return

    if school_event_time.empty:
        print("  school-event-time is empty; skipping ratio plot")
        return

    # ── Build school-level n_users by (unitid, grad_year) ─────────────────────
    # Use smallest horizon (n_users is horizon-independent before outcome filtering).
    min_horizon = int(
        pd.to_numeric(school_event_time["horizon_years"], errors="coerce").min()
    )
    base = school_event_time[
        school_event_time["horizon_years"].astype("Int64") == min_horizon
    ].copy()
    base["unitid"] = pd.to_numeric(base["unitid"], errors="coerce").astype("Int64")
    base["cohort_t"] = pd.to_numeric(base["cohort_t"], errors="coerce").astype(int)
    base["relabel_year"] = pd.to_numeric(base["relabel_year"], errors="coerce").astype(int)
    base["n_users"] = pd.to_numeric(base["n_users"], errors="coerce").astype(float)
    # grad_year = relabel_year + cohort_t
    base["grad_year"] = base["relabel_year"] + base["cohort_t"]

    # ── Load IPEDS econ completions ───────────────────────────────────────────
    # CIP 45.06 = Economics; MA only to match the relabel event population.
    ipeds = pd.read_parquet(
        completions_path,
        columns=["unitid", "year", "ctotalt", "cip4dig", "awlevel_group"],
    )
    print(f"  IPEDS completions rows: {len(ipeds):,}")

    ipeds_econ = ipeds[
        (pd.to_numeric(ipeds["cip4dig"], errors="coerce") == 4506)
        & (ipeds["awlevel_group"] == "Master")
    ].copy()
    print(f"  IPEDS CIP 45.06 MA rows: {len(ipeds_econ):,}")

    ipeds_econ["unitid"] = pd.to_numeric(ipeds_econ["unitid"], errors="coerce").astype("Int64")
    ipeds_econ["year"] = pd.to_numeric(ipeds_econ["year"], errors="coerce")
    ipeds_econ["ctotalt"] = pd.to_numeric(ipeds_econ["ctotalt"], errors="coerce").fillna(0)

    # Sum across all econ CIPs per (unitid, year).
    ipeds_agg = (
        ipeds_econ.groupby(["unitid", "year"], as_index=False)
        .agg(n_grads=("ctotalt", "sum"))
    )
    ipeds_agg["year"] = ipeds_agg["year"].astype(int)

    # ── Join and compute ratio ────────────────────────────────────────────────
    merged = base.merge(
        ipeds_agg,
        left_on=["unitid", "grad_year"],
        right_on=["unitid", "year"],
        how="left",
    )
    n_matched = merged["n_grads"].notna().sum()
    print(f"  School-cohort rows with IPEDS match: {n_matched:,} / {len(merged):,}")

    merged = merged[merged["n_grads"].notna() & (merged["n_grads"] > 0)].copy()
    merged["ratio"] = merged["n_users"] / merged["n_grads"]

    if merged.empty:
        print("  No matched rows with positive n_grads; skipping ratio plot")
        return

    # Aggregate to cohort_t (mean school-level ratio).
    ratio_agg = (
        merged.groupby("cohort_t", as_index=False)
        .agg(ratio_mean=("ratio", "mean"), n_schools=("unitid", "nunique"))
        .sort_values("cohort_t")
    )
    print(ratio_agg.to_string(index=False))

    xticks = sorted(ratio_agg["cohort_t"].unique().tolist())
    color = sns.color_palette("deep")[2]

    fig, ax = plt.subplots(figsize=(9, 5))
    line_df = _coerce_plot_frame(ratio_agg, ["cohort_t", "ratio_mean"], sort_col="cohort_t")
    ax.plot(
        line_df["cohort_t"].to_numpy(dtype=float),
        line_df["ratio_mean"].to_numpy(dtype=float),
        marker="o",
        linewidth=2,
        color=color,
    )
    ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
    ax.axhline(y=0, linestyle=":", color="gray", linewidth=0.8)
    ax.set_xlabel("Graduation year relative to relabel event")
    ax.set_ylabel("Revelio econ users / IPEDS econ graduates (school avg)")
    ax.set_title("Coverage ratio: Revelio users to IPEDS econ graduates")
    ax.set_xticks(xticks)
    fig.tight_layout()
    _save_figure(fig, out_dir, "n_users_to_ipeds_grads_ratio")

    print(f"  Step 8b done in {_elapsed(t0)}")


def plot_event_study(event_study: pd.DataFrame, out_dir: Path) -> None:
    t0 = time.time()
    print("\n── Step 8: Writing raw event-study plots ─────────────────────────────")

    if event_study.empty:
        print("  event-study summary is empty; skipping plots")
        return

    palette = sns.color_palette("deep", n_colors=event_study["horizon_years"].nunique())
    xticks = sorted(event_study["cohort_t"].dropna().astype(int).unique().tolist())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for color, (horizon, grp) in zip(palette, event_study.groupby("horizon_years", sort=True)):
        line_df = _coerce_plot_frame(grp, ["cohort_t", "still_in_us_mean"], sort_col="cohort_t")
        if line_df.empty:
            continue
        ax.plot(
            line_df["cohort_t"].to_numpy(dtype=float),
            line_df["still_in_us_mean"].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            color=color,
            label=f"{int(horizon)} years after event",
        )
    ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Graduation year relative to relabel event")
    ax.set_ylabel("Mean school-level probability still in US")
    ax.set_xticks(xticks)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, out_dir, "still_in_us_event_study_by_horizon_school_level")

    for color, (horizon, grp) in zip(palette, event_study.groupby("horizon_years", sort=True)):
        line_df = _coerce_plot_frame(grp, ["cohort_t", "still_in_us_mean"], sort_col="cohort_t")
        if line_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.plot(
            line_df["cohort_t"].to_numpy(dtype=float),
            line_df["still_in_us_mean"].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            color=color,
        )
        valid_ci = grp["still_in_us_se"].notna()
        if valid_ci.any():
            ci_grp = _coerce_plot_frame(
                grp.loc[valid_ci],
                ["cohort_t", "still_in_us_ci_low", "still_in_us_ci_high"],
                sort_col="cohort_t",
            )
            ax.fill_between(
                ci_grp["cohort_t"].to_numpy(dtype=float),
                ci_grp["still_in_us_ci_low"].to_numpy(dtype=float),
                ci_grp["still_in_us_ci_high"].to_numpy(dtype=float),
                alpha=0.25,
                color=color,
            )
        ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("Graduation year relative to relabel event")
        ax.set_ylabel("Mean school-level probability still in US")
        ax.set_xticks(xticks)
        ax.set_title(f"{int(horizon)} years after event")
        fig.tight_layout()
        _save_figure(fig, out_dir, f"still_in_us_event_study_t{int(horizon)}_school_level")

    print(f"  Step 7 done in {_elapsed(t0)}")


def plot_regression_event_study(regression_event_study: pd.DataFrame, out_dir: Path) -> None:
    t0 = time.time()
    print("\n── Step 9: Writing regression event-study plots ─────────────────────")

    if regression_event_study.empty:
        print("  regression event-study summary is empty; skipping plots")
        return

    reference_event_time = int(
        regression_event_study["reference_cohort_t"].dropna().iloc[0]
    )
    palette = sns.color_palette("deep", n_colors=regression_event_study["horizon_years"].nunique())
    xticks = sorted(regression_event_study["cohort_t"].dropna().astype(int).unique().tolist())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for color, (horizon, grp) in zip(
        palette,
        regression_event_study.groupby("horizon_years", sort=True),
    ):
        line_df = _coerce_plot_frame(grp, ["cohort_t", "coef"], sort_col="cohort_t")
        if line_df.empty:
            continue
        ax.plot(
            line_df["cohort_t"].to_numpy(dtype=float),
            line_df["coef"].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            color=color,
            label=f"{int(horizon)} years after event",
        )
        valid_ci = grp["se"].notna()
        if valid_ci.any():
            ci_grp = _coerce_plot_frame(
                grp.loc[valid_ci],
                ["cohort_t", "ci_low", "ci_high"],
                sort_col="cohort_t",
            )
            ax.fill_between(
                ci_grp["cohort_t"].to_numpy(dtype=float),
                ci_grp["ci_low"].to_numpy(dtype=float),
                ci_grp["ci_high"].to_numpy(dtype=float),
                alpha=0.20,
                color=color,
            )
    ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
    ax.axhline(y=0, linestyle=":", color="gray", linewidth=1)
    ax.set_xlabel("Graduation year relative to relabel event")
    ax.set_ylabel(
        f"Difference vs cohort_t = {reference_event_time} (school + event-year FE, clustered by school)"
    )
    ax.set_xticks(xticks)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, out_dir, "still_in_us_event_study_by_horizon_school_fe_clustered")

    for color, (horizon, grp) in zip(
        palette,
        regression_event_study.groupby("horizon_years", sort=True),
    ):
        line_df = _coerce_plot_frame(grp, ["cohort_t", "coef"], sort_col="cohort_t")
        if line_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.plot(
            line_df["cohort_t"].to_numpy(dtype=float),
            line_df["coef"].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            color=color,
        )
        valid_ci = grp["se"].notna()
        if valid_ci.any():
            ci_grp = _coerce_plot_frame(
                grp.loc[valid_ci],
                ["cohort_t", "ci_low", "ci_high"],
                sort_col="cohort_t",
            )
            ax.fill_between(
                ci_grp["cohort_t"].to_numpy(dtype=float),
                ci_grp["ci_low"].to_numpy(dtype=float),
                ci_grp["ci_high"].to_numpy(dtype=float),
                alpha=0.20,
                color=color,
            )
        ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
        ax.axhline(y=0, linestyle=":", color="gray", linewidth=1)
        ax.set_xlabel("Graduation year relative to relabel event")
        ax.set_ylabel(
            f"Difference vs cohort_t = {reference_event_time} (school + event-year FE, clustered by school)"
        )
        ax.set_xticks(xticks)
        ax.set_title(f"{int(horizon)} years after event")
        fig.tight_layout()
        _save_figure(fig, out_dir, f"still_in_us_event_study_t{int(horizon)}_school_fe_clustered")

    print(f"  Step 8 done in {_elapsed(t0)}")


def print_summary(
    base_events: pd.DataFrame,
    panel: pd.DataFrame,
    school_event_time: pd.DataFrame,
    event_study: pd.DataFrame,
    regression_event_study: pd.DataFrame,
    latest_available_year: int,
) -> None:
    print("\n── Summary ───────────────────────────────────────────────────────────")
    print(f"  base user-events: {len(base_events):,}")
    print(f"  latest available work-history year: {latest_available_year}")
    print(f"  panel rows: {len(panel):,}")
    print(f"  school-event-time rows: {len(school_event_time):,}")
    print(f"  regression rows: {len(regression_event_study):,}")
    print(f"  horizons: {cfg.ANALYSIS_HORIZONS}")
    if not event_study.empty:
        overall = (
            school_event_time.groupby("horizon_years", as_index=False)
            .agg(
                n_school_events=("unitid", "size"),
                n_users=("n_users", "sum"),
                still_in_us_mean=("still_in_us_mean", "mean"),
                used_latest_avail_share=("used_latest_avail_share", "mean"),
            )
            .sort_values("horizon_years")
        )
        print("  by horizon:")
        print(overall.to_string(index=False))


def _has_control_inputs() -> bool:
    """Return True if the never-treated control parquets produced by relabel_indiv_model.py exist."""
    return bool(
        cfg.NEVER_TREATED_EDUCATION_PARQUET
        and Path(cfg.NEVER_TREATED_EDUCATION_PARQUET).exists()
        and cfg.NEVER_TREATED_POSITIONS_PARQUET
        and Path(cfg.NEVER_TREATED_POSITIONS_PARQUET).exists()
    )


def assign_control_relabel_years(
    con: ddb.DuckDBPyConnection,
    control_education: pd.DataFrame,
    relabels_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match control schools to treated cohorts using IPEDS pre-treatment econ size.

    Calls v2._match_treated_to_untreated_cohorts (nearest-neighbour on source-econ
    size in t-1, by relabel_year × relabel_type) to obtain matched (treated, control)
    pairs, then assigns each control school the relabel_year of its matched treated
    school.  Returns control_education rows augmented with non-null relabel_year and
    relabel_type.  Schools not matched by IPEDS are dropped.
    """
    t0 = time.time()
    print("\n── Control: Assigning relabel_year via IPEDS matching ───────────────")

    if _v2 is None:
        print("  Warning: could not import econ_relabels_opt_usage_v2; skipping control matching.")
        return pd.DataFrame()

    matched_pairs = _v2._match_treated_to_untreated_cohorts(con=con, relabel_df=relabels_df)
    if matched_pairs.empty:
        print("  Warning: no IPEDS control pairs found; skipping control analysis.")
        return pd.DataFrame()

    print(f"  IPEDS matched pairs: {len(matched_pairs):,}")
    print(
        matched_pairs[["relabel_type", "relabel_year", "treated_unitid", "control_unitid"]]
        .head(10)
        .to_string(index=False)
    )

    # Build a lookup: control_unitid → [(relabel_year, relabel_type), ...]
    # A control school may be matched to multiple treated cohorts (with replacement).
    pairs = matched_pairs[["control_unitid", "relabel_year", "relabel_type"]].copy()
    pairs["control_unitid"] = pd.to_numeric(pairs["control_unitid"], errors="coerce").astype("Int64")
    pairs["relabel_year"] = pd.to_numeric(pairs["relabel_year"], errors="coerce").astype("Int64")

    educ = control_education.copy()
    educ["unitid"] = pd.to_numeric(educ["unitid"], errors="coerce").astype("Int64")
    # Drop the placeholder nulls written by relabel_indiv_model.py; will be replaced by matching.
    educ = educ.drop(columns=["relabel_year", "relabel_type"], errors="ignore")

    educ_with_years = educ.merge(
        pairs.rename(columns={"control_unitid": "unitid"}),
        on="unitid",
        how="inner",  # drop control schools not matched to any treated cohort
    )

    n_sch = educ_with_years["unitid"].nunique()
    n_total = control_education["unitid"].nunique()
    print(f"  Control schools in Revelio data: {n_total}  |  matched to IPEDS: {n_sch}")
    print(f"  Control education rows after matching: {len(educ_with_years):,}")
    print(f"  Done in {_elapsed(t0)}")
    return educ_with_years


def build_control_pipeline(
    con: ddb.DuckDBPyConnection,
    control_education_with_years: pd.DataFrame,
    control_positions_path: Path,
    latest_available_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run build_base_events → build_analysis_panel → build_school_event_time →
    build_event_study for the never-treated control schools.

    Temporarily overwrites the 'matched_education' and 'matched_positions' DuckDB
    views to point at control data, reusing all existing build functions unchanged.
    Returns (control_school_event_time, control_event_study).
    """
    t0 = time.time()
    print("\n── Control: Building control analysis panel ─────────────────────────")

    pos_path_str = _escape_sql_literal(str(control_positions_path))
    control_education_py = control_education_with_years.copy()
    if "exclude_immediate_same_inst_phd_after_master_ind" not in control_education_py.columns:
        control_education_py["exclude_immediate_same_inst_phd_after_master_ind"] = 0
    con.register("_control_education_py", control_education_py)
    con.sql("CREATE OR REPLACE TEMP VIEW matched_education AS SELECT * FROM _control_education_py")
    con.sql(
        f"CREATE OR REPLACE TEMP VIEW matched_positions AS "
        f"SELECT * FROM read_parquet('{pos_path_str}')"
    )
    build_positions_view(con)

    control_base_events = build_base_events(con)
    if control_base_events.empty:
        print("  No control base events found; skipping control panel.")
        return pd.DataFrame(), pd.DataFrame()

    control_panel = build_analysis_panel(con, latest_available_year)
    control_school_event_time = build_school_event_time(control_panel)
    control_event_study = build_event_study(control_school_event_time)

    print(f"  Control pipeline done in {_elapsed(t0)}")
    return control_school_event_time, control_event_study


def plot_treated_vs_control_event_study(
    treated_event_study: pd.DataFrame,
    control_event_study: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Overlay treated and never-treated-econ control still-in-US event study series.
    One figure per horizon that exists in both datasets.
    """
    t0 = time.time()
    print("\n── Treated vs. control event study plots ────────────────────────────")

    if treated_event_study.empty or control_event_study.empty:
        print("  One or both event studies empty; skipping comparison plots.")
        return

    treated_horizons = set(treated_event_study["horizon_years"].dropna().astype(int).unique())
    control_horizons = set(control_event_study["horizon_years"].dropna().astype(int).unique())
    horizons = sorted(treated_horizons & control_horizons)
    if not horizons:
        print("  No horizon overlap between treated and control; skipping.")
        return

    all_cohort_t = sorted(
        set(treated_event_study["cohort_t"].dropna().astype(int).unique()) |
        set(control_event_study["cohort_t"].dropna().astype(int).unique())
    )

    TREATED_COLOR = "#2e8b57"
    CONTROL_COLOR = "#e07a5f"

    for horizon in horizons:
        t_grp = (
            treated_event_study[treated_event_study["horizon_years"].astype("Int64") == horizon]
        )
        c_grp = (
            control_event_study[control_event_study["horizon_years"].astype("Int64") == horizon]
        )

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for grp, color, label in [
            (t_grp, TREATED_COLOR, "Treated (Econ→Econometrics)"),
            (c_grp, CONTROL_COLOR, "Never-treated Econ (matched)"),
        ]:
            line_df = _coerce_plot_frame(grp, ["cohort_t", "still_in_us_mean"], sort_col="cohort_t")
            if line_df.empty:
                continue
            ax.plot(
                line_df["cohort_t"].to_numpy(dtype=float),
                line_df["still_in_us_mean"].to_numpy(dtype=float),
                marker="o", linewidth=2, color=color, label=label,
            )
            valid_se = grp["still_in_us_se"].notna() & (grp["still_in_us_se"] > 0)
            if valid_se.any():
                ci = _coerce_plot_frame(
                    grp.loc[valid_se],
                    ["cohort_t", "still_in_us_ci_low", "still_in_us_ci_high"],
                    sort_col="cohort_t",
                )
                ax.fill_between(
                    ci["cohort_t"].to_numpy(dtype=float),
                    ci["still_in_us_ci_low"].to_numpy(dtype=float),
                    ci["still_in_us_ci_high"].to_numpy(dtype=float),
                    alpha=0.20, color=color,
                )

        ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("Graduation year relative to relabel event")
        ax.set_ylabel("Mean school-level probability still in US")
        ax.set_xticks(all_cohort_t)
        ax.set_title(f"{int(horizon)} years after event")
        ax.legend(frameon=False)
        fig.tight_layout()
        _save_figure(fig, out_dir, f"still_in_us_treated_vs_control_t{int(horizon)}")

    print(f"  Done in {_elapsed(t0)}")


def plot_did_regression_event_study(
    treated_school_event_time: pd.DataFrame,
    control_school_event_time: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    DiD event-study regression using stacked treated + matched-control school data.

    Spec (per horizon):
        still_in_us_mean ~ C(cohort_t, Treatment(ref=-1)) × treated_ind
                           + C(unitid) + C(grad_year) + C(relabel_year)
    clustered by school (unitid).

    The C(cohort_t) × treated_ind interaction coefficients give the differential
    treated-minus-control gap at each cohort_t relative to cohort_t = -1.
    Produces one figure per horizon, saved as
        still_in_us_did_event_study_t{h}.png
    """
    t0 = time.time()
    print("\n── Treated vs. control DiD event-study regression ───────────────────")

    if treated_school_event_time.empty or control_school_event_time.empty:
        print("  One or both school-event-time tables empty; skipping DiD regression.")
        return

    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        print(f"  statsmodels unavailable; skipping DiD regression ({exc})")
        return

    reference_event_time = int(cfg.ANALYSIS_REGRESSION_REFERENCE_EVENT_TIME)

    # Stack treated (treated_ind=1) and control (treated_ind=0).
    t_df = treated_school_event_time.copy()
    t_df["treated_ind"] = 1
    c_df = control_school_event_time.copy()
    c_df["treated_ind"] = 0
    stacked = pd.concat([t_df, c_df], ignore_index=True)

    # Exclude future-outcome rows (same filter as build_event_study).
    stacked = stacked[stacked["target_year_observed_share"] > 0].copy()

    for col in ["cohort_t", "unitid", "relabel_year", "n_users", "still_in_us_mean"]:
        stacked[col] = pd.to_numeric(stacked[col], errors="coerce")
    stacked["cohort_t"] = stacked["cohort_t"].astype(int)
    stacked["unitid"] = stacked["unitid"].astype(int)
    stacked["relabel_year"] = stacked["relabel_year"].astype(int)
    stacked["still_in_us_mean"] = stacked["still_in_us_mean"].astype(float)
    # grad_year FE absorbs calendar-time trends; grad_year = relabel_year + cohort_t.
    stacked["grad_year"] = stacked["relabel_year"] + stacked["cohort_t"]

    horizons = sorted(
        set(treated_school_event_time["horizon_years"].dropna().astype(int).unique()) &
        set(control_school_event_time["horizon_years"].dropna().astype(int).unique())
    )
    if not horizons:
        print("  No horizon overlap; skipping DiD regression.")
        return

    COLOR = "#4c78a8"  # single color for DiD coefficient series

    for horizon in horizons:
        reg_df = stacked[stacked["horizon_years"].astype("Int64") == horizon].copy()
        if reg_df.empty:
            continue

        treated_cohort_values = set(
            reg_df.loc[reg_df["treated_ind"] == 1, "cohort_t"].dropna().astype(int).unique().tolist()
        )
        control_cohort_values = set(
            reg_df.loc[reg_df["treated_ind"] == 0, "cohort_t"].dropna().astype(int).unique().tolist()
        )
        cohort_values = sorted(treated_cohort_values & control_cohort_values)
        reg_df = reg_df[reg_df["cohort_t"].isin(cohort_values)].copy()
        n_schools = reg_df["unitid"].nunique()

        if reference_event_time not in cohort_values:
            print(
                f"  horizon {horizon}: reference cohort_t={reference_event_time} not present; "
                "skipping"
            )
            continue
        if len(cohort_values) < 2 or n_schools < 2:
            print(
                f"  horizon {horizon}: insufficient variation (cohort_t={len(cohort_values)}, "
                f"schools={n_schools}); skipping"
            )
            continue

        # Interaction spec: treated-control gap at each cohort_t, absorbing
        # school, graduation-year, and relabel-year fixed effects. Coefficients
        # are then re-centered relative to the reference cohort_t.
        formula = (
            "still_in_us_mean ~ "
            f"C(cohort_t, Treatment(reference={reference_event_time})):treated_ind "
            f"+ C(unitid) + C(grad_year) + C(relabel_year)"
        )
        try:
            result = smf.ols(formula, data=reg_df).fit(
                cov_type="cluster",
                cov_kwds={"groups": reg_df["unitid"]},
            )
        except Exception as exc:
            print(f"  horizon {horizon}: regression failed ({exc}); skipping")
            continue

        cov = result.cov_params()
        ref_param = _find_did_interaction_param(
            result.params,
            cohort_t=reference_event_time,
            reference_event_time=reference_event_time,
        )
        ref_coef = (
            float(result.params[ref_param])
            if ref_param is not None
            else 0.0
        )

        rows = []
        for ct in cohort_values:
            if ct == reference_event_time:
                coef, se = 0.0, 0.0
            else:
                param = _find_did_interaction_param(
                    result.params,
                    cohort_t=ct,
                    reference_event_time=reference_event_time,
                )
                if param is None:
                    coef, se = float("nan"), float("nan")
                else:
                    raw_coef = float(result.params[param])
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
            rows.append({"cohort_t": ct, "coef": coef, "se": se})

        reg_results = pd.DataFrame(rows).sort_values("cohort_t")
        reg_results["ci_low"] = reg_results["coef"] - 1.96 * reg_results["se"]
        reg_results["ci_high"] = reg_results["coef"] + 1.96 * reg_results["se"]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        line_df = _coerce_plot_frame(reg_results, ["cohort_t", "coef"], sort_col="cohort_t")
        ax.plot(
            line_df["cohort_t"].to_numpy(dtype=float),
            line_df["coef"].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            color=COLOR,
        )
        valid_ci = reg_results["se"].notna() & (reg_results["se"] > 0)
        if valid_ci.any():
            ci = _coerce_plot_frame(
                reg_results.loc[valid_ci],
                ["cohort_t", "ci_low", "ci_high"],
                sort_col="cohort_t",
            )
            ax.fill_between(
                ci["cohort_t"].to_numpy(dtype=float),
                ci["ci_low"].to_numpy(dtype=float),
                ci["ci_high"].to_numpy(dtype=float),
                alpha=0.25, color=COLOR,
            )
        ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
        ax.axhline(y=0, linestyle=":", color="gray", linewidth=1)
        ax.set_xlabel("Graduation year relative to relabel event")
        ax.set_ylabel(
            f"DiD coefficient vs cohort_t={reference_event_time}\n"
            "(school + grad-year + relabel-year FE, clustered by school)"
        )
        ax.set_title(f"{int(horizon)} years after event")
        ax.set_xticks(cohort_values)
        fig.tight_layout()
        _save_figure(fig, out_dir, f"still_in_us_did_event_study_t{int(horizon)}")

    print(f"  Done in {_elapsed(t0)}")


def main() -> None:
    t_main = time.time()
    education_path, positions_path = _validate_inputs()
    panel_path = _analysis_panel_path()
    school_event_time_path = _analysis_school_event_time_path()
    event_study_path = _analysis_event_study_path()
    regression_event_study_path = _analysis_regression_event_study_path()
    output_dir = _analysis_output_dir()

    con = ddb.connect()
    try:
        register_sources(con, education_path, positions_path)
        build_positions_view(con)
        base_events = build_base_events(con)
        latest_available_year = get_latest_available_year(con)
        panel = build_analysis_panel(con, latest_available_year)
        school_event_time = build_school_event_time(panel)
        event_study = build_event_study(school_event_time)
        regression_event_study = build_regression_event_study(school_event_time)

        _write_parquet(panel, panel_path, PANEL_COLUMNS)
        _write_parquet(school_event_time, school_event_time_path, SCHOOL_EVENT_TIME_COLUMNS)
        _write_parquet(event_study, event_study_path, EVENT_STUDY_COLUMNS)
        _write_parquet(
            regression_event_study,
            regression_event_study_path,
            REGRESSION_EVENT_STUDY_COLUMNS,
        )
        plot_n_users_event_study(school_event_time, output_dir)
        plot_n_users_to_ipeds_ratio(school_event_time, output_dir)
        plot_event_study(event_study, output_dir)
        plot_regression_event_study(regression_event_study, output_dir)

        print_summary(
            base_events,
            panel,
            school_event_time,
            event_study,
            regression_event_study,
            latest_available_year,
        )

        # ── Never-treated control pipeline ───────────────────────────────────
        if _has_control_inputs():
            print("\n\n════ Control Schools (never-treated econ) ════════════════════════════")
            control_education = pd.read_parquet(cfg.NEVER_TREATED_EDUCATION_PARQUET)
            print(f"  Control education: {len(control_education):,} rows | "
                  f"{control_education['unitid'].nunique():,} unique schools")

            # Load the detected relabels (needed for IPEDS matching).
            if cfg.RELABELS_PARQUET and Path(cfg.RELABELS_PARQUET).exists():
                relabels_df = pd.read_parquet(cfg.RELABELS_PARQUET)
            else:
                print(f"  Warning: relabels parquet not found at {cfg.RELABELS_PARQUET}; "
                      "cannot run IPEDS matching — skipping control analysis.")
                relabels_df = pd.DataFrame()

            if not relabels_df.empty:
                control_education_with_years = assign_control_relabel_years(
                    con, control_education, relabels_df
                )
                if not control_education_with_years.empty:
                    control_positions_path = Path(cfg.NEVER_TREATED_POSITIONS_PARQUET)
                    control_school_event_time, control_event_study = build_control_pipeline(
                        con,
                        control_education_with_years,
                        control_positions_path,
                        latest_available_year,
                    )
                    plot_treated_vs_control_event_study(
                        event_study, control_event_study, output_dir
                    )
                    plot_did_regression_event_study(
                        school_event_time, control_school_event_time, output_dir
                    )
                    # Restore treated views so the connection is clean on exit.
                    register_sources(con, education_path, positions_path)
                    build_positions_view(con)
        else:
            print(
                "\nControl school parquets not found — skipping control analysis."
                f"\n(Run relabel_indiv_model.py first to extract never-treated control data.)"
                f"\nExpected: {cfg.NEVER_TREATED_EDUCATION_PARQUET}"
            )

        print(f"\nTotal runtime: {_elapsed(t_main)}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
