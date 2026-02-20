"""
Company-level shift-share pipeline.

Builds z_ct = Σ_k s_ckr * g_kt for companies (rcids), treatment counts of
Master's OPT hires, and outcomes based on Revelio headcounts. Inputs live
under root/data; outputs are written to root/data/out/company_shift_share.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import seaborn as sns
from pathlib import Path
import sys
from typing import Iterable, Optional

import duckdb as ddb

try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
except ModuleNotFoundError:
    # Allow direct execution when repo root is not already on PYTHONPATH.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config

@dataclass(frozen=True)
class PipelinePaths:
    transitions: Path
    headcount: Path
    revelio_ipeds_foia_inst_crosswalk: Path
    ipeds_ma_only: Path
    foia_sevp_with_person_id: Path
    foia_sevp_with_person_id_employment_corrected: Path
    employer_crosswalk: Path
    preferred_rcids: Path
    instrument_components_out: Path
    instrument_panel_out: Path
    treatment_out: Path
    outcomes_out: Path
    analysis_panel_out: Path


def _resolve_path(paths_cfg: dict, key: str) -> Path:
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"Config paths.{key} must be set.")
    return Path(value)


def _resolve_pipeline_paths(cfg: dict) -> PipelinePaths:
    paths_cfg = get_cfg_section(cfg, "paths")
    return PipelinePaths(
        transitions=_resolve_path(paths_cfg, "transitions_out"),
        headcount=_resolve_path(paths_cfg, "headcounts_out"),
        revelio_ipeds_foia_inst_crosswalk=_resolve_path(paths_cfg, "revelio_ipeds_foia_inst_crosswalk"),
        ipeds_ma_only=_resolve_path(paths_cfg, "ipeds_ma_only"),
        foia_sevp_with_person_id=_resolve_path(paths_cfg, "foia_sevp_with_person_id"),
        foia_sevp_with_person_id_employment_corrected=_resolve_path(paths_cfg, "foia_sevp_with_person_id_employment_corrected"),
        employer_crosswalk=_resolve_path(paths_cfg, "employer_crosswalk"),
        preferred_rcids=_resolve_path(paths_cfg, "preferred_rcids"),
        instrument_components_out=_resolve_path(paths_cfg, "instrument_components_out"),
        instrument_panel_out=_resolve_path(paths_cfg, "instrument_panel_out"),
        treatment_out=_resolve_path(paths_cfg, "treatment_out"),
        outcomes_out=_resolve_path(paths_cfg, "outcomes_out"),
        analysis_panel_out=_resolve_path(paths_cfg, "analysis_panel_out"),
    )


def _escape(path: Path) -> str:
    return str(path).replace("'", "''")


def _parse_unitids(raw: str | None) -> list[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _sql_in_list(values: list[str]) -> str:
    if not values:
        return "()"
    escaped = [v.replace("'", "''") for v in values]
    return "(" + ",".join(f"'{v}'" for v in escaped) + ")"

def _check_paths(paths: dict[str, Path]) -> None:
    missing = [f"{label}: {p}" for label, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def _ensure_out_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_view(con: ddb.DuckDBPyConnection, view: str, path: Path) -> None:
    _ensure_out_dir(path)
    con.sql(f"COPY (SELECT * FROM {view}) TO '{_escape(path)}' (FORMAT PARQUET)")
    print(f"Wrote {path}")


def _has_column(con: ddb.DuckDBPyConnection, view: str, column: str) -> bool:
    rows = con.execute(f"PRAGMA table_info('{view}')").fetchall()
    cols = {r[1].lower() for r in rows}
    return column.lower() in cols


def _masters_predicate(con: ddb.DuckDBPyConnection, view: str = "foia_raw") -> Optional[str]:
    """
    Return a SQL predicate to filter Master's records if columns exist.
    Prioritized checks follow common FOIA clean outputs.
    """
    if _has_column(con, view, "student_edu_level_desc"):
        return "LOWER(CAST(student_edu_level_desc AS VARCHAR)) = 'master''s'"
    if _has_column(con, view, "awlevel_group"):
        return "LOWER(CAST(awlevel_group AS VARCHAR)) = 'master'"
    if _has_column(con, view, "awlevel"):
        return "CAST(awlevel AS INTEGER) = 7"
    for candidate in ("education_level", "degree_level"):
        if _has_column(con, view, candidate):
            return f"LOWER(CAST({candidate} AS VARCHAR)) LIKE '%master%'"
    return None


def _register_inputs(con: ddb.DuckDBPyConnection, paths: PipelinePaths) -> None:
    _check_paths(
        {
            "transitions": paths.transitions,
            "headcount": paths.headcount,
            "revelio_ipeds_foia_inst_crosswalk": paths.revelio_ipeds_foia_inst_crosswalk,
            "ipeds_ma_only": paths.ipeds_ma_only,
            "foia_raw_full": paths.foia_sevp_with_person_id,
            "foia_raw_corrected": paths.foia_sevp_with_person_id_employment_corrected,
            "employer_cw": paths.employer_crosswalk,
            "preferred_rcids": paths.preferred_rcids,
        }
    )

    con.sql(f"CREATE OR REPLACE TEMP VIEW revelio_transitions AS SELECT * FROM read_parquet('{_escape(paths.transitions)}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW revelio_headcount AS SELECT * FROM read_parquet('{_escape(paths.headcount)}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW revelio_inst_cw_raw AS SELECT * FROM read_parquet('{_escape(paths.revelio_ipeds_foia_inst_crosswalk)}')")
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW revelio_inst_cw AS
        WITH norm AS (
            SELECT
                LOWER(TRIM(CAST(university_raw AS VARCHAR))) AS university_raw_norm,
                CAST(CAST(UNITID AS BIGINT) AS VARCHAR) AS unitid
            FROM revelio_inst_cw_raw
            WHERE university_raw IS NOT NULL
              AND TRIM(CAST(university_raw AS VARCHAR)) <> ''
              AND UNITID IS NOT NULL
        )
        SELECT
            university_raw_norm,
            MIN(unitid) AS unitid
        FROM norm
        GROUP BY university_raw_norm
        """
    )
    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw AS SELECT * FROM read_parquet('{_escape(paths.ipeds_ma_only)}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{_escape(paths.foia_sevp_with_person_id_employment_corrected)}') WHERE year_int > 2005")
    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw_full AS SELECT * FROM read_parquet('{_escape(paths.foia_sevp_with_person_id)}') WHERE year_int > 2005")
    con.sql(f"CREATE OR REPLACE TEMP VIEW preferred_rcids AS SELECT DISTINCT preferred_rcid FROM read_parquet('{_escape(paths.preferred_rcids)}')")
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW employer_crosswalk AS
        SELECT ec.*
        FROM read_parquet('{_escape(paths.employer_crosswalk)}') AS ec
        JOIN preferred_rcids AS pr
          ON ec.preferred_rcid = pr.preferred_rcid
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW matched_rcids AS
        SELECT DISTINCT preferred_rcid AS rcid
        FROM employer_crosswalk
        WHERE preferred_rcid IS NOT NULL
        """
    )


def _create_ipeds_growth_view(
    con: ddb.DuckDBPyConnection,
    use_changes: bool = False,
    demean_by_school: bool = False,
    exclude_unitids: list[str] | None = None,
) -> str:
    """
    Recompute g_kt using the IHMA rules (international-heavy master's programs).
    """
    view_name = "ipeds_unit_growth"
    g_expr = "g_kt"
    g_all_expr = "g_kt_all"
    g_intl_expr = "g_kt_intl"
    if use_changes:
        g_expr = "ASINH(g_kt) - ASINH(g_kt_lag)"
        g_all_expr = "ASINH(g_kt_all) - ASINH(g_kt_all_lag)"
        g_intl_expr = "ASINH(g_kt_intl) - ASINH(g_kt_intl_lag)"
    demean_cte = ""
    final_view = "raw_out"
    if demean_by_school:
        demean_cte = """
        ,
        demeaned AS (
            SELECT
                k,
                t,
                g_kt - AVG(g_kt) OVER (PARTITION BY k) AS g_kt,
                g_kt_all - AVG(g_kt_all) OVER (PARTITION BY k) AS g_kt_all,
                g_kt_intl - AVG(g_kt_intl) OVER (PARTITION BY k) AS g_kt_intl
            FROM raw_out
        )
        """
        final_view = "demeaned"

    exclude_unitids = exclude_unitids or []
    exclude_clause = ""
    if exclude_unitids:
        exclude_clause = f"AND CAST(unitid AS VARCHAR) NOT IN {_sql_in_list(exclude_unitids)}"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        WITH base AS (
            SELECT
                CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k,
                CAST(cipcode AS VARCHAR) AS cipcode,
                CAST(year AS INTEGER) AS year,
                CAST(cnralt AS DOUBLE) AS cnralt,
                CAST(ctotalt AS DOUBLE) AS ctotalt,
                CAST(share_intl AS DOUBLE) AS share_intl
            FROM ipeds_raw
            WHERE unitid IS NOT NULL
              {exclude_clause}
        ),
        program_flags AS (
            SELECT
                k,
                cipcode,
                MAX(
                    CASE
                        WHEN year > 2010
                         AND share_intl >= 0.5
                         AND ctotalt >= 10
                        THEN 1
                        ELSE 0
                    END
                ) AS international_heavy
            FROM base
            WHERE cipcode IS NOT NULL
            GROUP BY k, cipcode
        ),
        joined AS (
            SELECT
                b.k,
                b.year,
                SUM(b.cnralt) AS tot_intl_students,
                SUM(COALESCE(b.ctotalt, 0) * COALESCE(p.international_heavy, 0)) AS tot_seats_ihma,
                SUM(COALESCE(b.cnralt, 0) * COALESCE(p.international_heavy, 0)) AS tot_intl_seats_ihma
            FROM base AS b
            LEFT JOIN program_flags AS p
                ON b.k = p.k
               AND b.cipcode = p.cipcode
            GROUP BY b.k, b.year
        ),
        bounds AS (
            SELECT
                k,
                MIN(year) AS min_year,
                MAX(year) AS max_year
            FROM joined
            GROUP BY k
        ),
        expanded AS (
            SELECT
                b.k,
                gs.year
            FROM bounds b,
            LATERAL generate_series(b.min_year, b.max_year) AS gs(year)
        ),
        filled AS (
            SELECT
                e.k,
                e.year,
                COALESCE(j.tot_intl_students, 0) AS tot_intl_students,
                COALESCE(j.tot_seats_ihma, 0) AS tot_seats_ihma,
                COALESCE(j.tot_intl_seats_ihma, 0) AS tot_intl_seats_ihma
            FROM expanded e
            LEFT JOIN joined j
                ON e.k = j.k
               AND e.year = j.year
        ),
        diffed AS (
            SELECT
                k,
                year,
                tot_intl_students AS g_kt_all,
                tot_seats_ihma AS g_kt, 
                tot_intl_seats_ihma AS g_kt_intl
            FROM filled
        ),
        with_lags AS (
            SELECT
                k,
                year,
                g_kt,
                g_kt_all,
                g_kt_intl,
                LAG(g_kt) OVER (PARTITION BY k ORDER BY year) AS g_kt_lag,
                LAG(g_kt_all) OVER (PARTITION BY k ORDER BY year) AS g_kt_all_lag,
                LAG(g_kt_intl) OVER (PARTITION BY k ORDER BY year) AS g_kt_intl_lag
            FROM diffed
        ),
        base_out AS (
            SELECT
                k,
                year AS t,
                {g_expr} AS g_kt,
                {g_all_expr} AS g_kt_all,
                {g_intl_expr} AS g_kt_intl
            FROM with_lags
            WHERE g_kt IS NOT NULL
        ),
        raw_out AS (
            SELECT * FROM base_out
        )
        {demean_cte}
        SELECT
            k,
            t,
            g_kt,
            g_kt_all,
            g_kt_intl
        FROM {final_view}
        """
    )
    return view_name


def _create_transition_share_view(
    con: ddb.DuckDBPyConnection,
    share_base_year: int,
    exclude_unitids: list[str] | None = None,
) -> str:
    """
    Build base-year shares s_ck: share of new hires at company c from university k
    in the base-year window.
    Also builds share_ck_full using all available years.
    """
    window_start = int(share_base_year) - 5
    window_end = int(share_base_year)
    exclude_unitids = exclude_unitids or []
    exclude_clause = ""
    if exclude_unitids:
        exclude_clause = f"AND CAST(cw.unitid AS VARCHAR) NOT IN {_sql_in_list(exclude_unitids)}"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW transition_unit_shares AS
        WITH transitions_window AS (
            SELECT
                CAST(t.rcid AS INTEGER) AS c,
                CAST(CAST(cw.unitid AS BIGINT) AS VARCHAR) AS k,
                CAST(t.year AS INTEGER) AS year,
                t.n_transitions,
                t.total_new_hires AS total_new_hires_alt,
                SUM(t.n_transitions) OVER(PARTITION BY t.year, t.rcid) AS total_new_hires
            FROM revelio_transitions AS t
            JOIN matched_rcids mr ON t.rcid = mr.rcid
            JOIN revelio_inst_cw AS cw
              ON LOWER(TRIM(CAST(t.university_raw AS VARCHAR))) = cw.university_raw_norm
            WHERE t.n_transitions IS NOT NULL
              AND t.total_new_hires IS NOT NULL
              AND cw.unitid IS NOT NULL
              {exclude_clause}
              AND CAST(t.year AS INTEGER) BETWEEN {window_start} AND {window_end}
        ),
        transitions_all AS (
            SELECT
                CAST(t.rcid AS INTEGER) AS c,
                CAST(CAST(cw.unitid AS BIGINT) AS VARCHAR) AS k,
                CAST(t.year AS INTEGER) AS year,
                t.n_transitions,
                SUM(t.n_transitions) OVER (PARTITION BY t.year, t.rcid) AS total_new_hires
            FROM revelio_transitions AS t
            JOIN matched_rcids mr ON t.rcid = mr.rcid
            JOIN revelio_inst_cw AS cw
              ON LOWER(TRIM(CAST(t.university_raw AS VARCHAR))) = cw.university_raw_norm
            WHERE t.n_transitions IS NOT NULL
              AND cw.unitid IS NOT NULL
              AND t.year IS NOT NULL
              {exclude_clause}
        ),
        base_companies AS (
            SELECT
                c,
                SUM(total_new_hires_year) AS total_new_hires,
                SUM(total_new_hires_alt_year) AS total_new_hires_alt
            FROM (
                SELECT
                    c,
                    year,
                    MAX(total_new_hires) AS total_new_hires_year,
                    MAX(total_new_hires_alt) AS total_new_hires_alt_year
                FROM transitions_window
                GROUP BY c, year
            ) AS company_year_totals_window
            GROUP BY c
            HAVING SUM(total_new_hires_year) IS NOT NULL
        ),
        agg AS (
            SELECT
                c,
                k,
                SUM(n_transitions) AS n_transitions
            FROM transitions_window
            GROUP BY c, k
        ),
        company_year_totals_full AS (
            SELECT
                c,
                year,
                MAX(total_new_hires) AS total_new_hires_year
            FROM transitions_all
            GROUP BY c, year
        ),
        base_companies_full AS (
            SELECT
                c,
                SUM(total_new_hires_year) AS total_new_hires_full
            FROM company_year_totals_full
            GROUP BY c
        ),
        agg_full AS (
            SELECT
                c,
                k,
                SUM(n_transitions) AS n_transitions_full
            FROM transitions_all
            GROUP BY c, k
        ),
        pair_support AS (
            SELECT c, k FROM agg_full
            UNION
            SELECT c, k FROM agg
        )
        SELECT
            ps.c,
            ps.k,
            COALESCE(a.n_transitions, 0) AS n_transitions,
            COALESCE(af.n_transitions_full, 0) AS n_transitions_full,
            bc.total_new_hires,
            bc.total_new_hires_alt,
            bcf.total_new_hires_full,
            COALESCE(a.n_transitions, 0) / NULLIF(bc.total_new_hires_alt, 0) AS share_ck,
            COALESCE(a.n_transitions, 0) / NULLIF(bc.total_new_hires, 0) AS share_ck_base,
            COALESCE(af.n_transitions_full, 0) / NULLIF(bcf.total_new_hires_full, 0) AS share_ck_full
        FROM pair_support AS ps
        LEFT JOIN base_companies AS bc
          ON ps.c = bc.c
        LEFT JOIN agg AS a
          ON ps.c = a.c
         AND ps.k = a.k
        LEFT JOIN agg_full AS af
          ON ps.c = af.c
         AND ps.k = af.k
        LEFT JOIN base_companies_full AS bcf
          ON ps.c = bcf.c
        """
    )
    return "transition_unit_shares"


def _create_instrument_views(con: ddb.DuckDBPyConnection, shares_view: str, growth_view: str) -> tuple[str, str]:
    components = "company_instrument_components"
    panel = "company_instrument_panel"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {components} AS
        SELECT
            s.c,
            s.k,
            g.t,
            CASE WHEN s.share_ck > 1 THEN NULL ELSE s.share_ck END AS share_ck,
            CASE WHEN s.share_ck_base > 1 THEN NULL ELSE s.share_ck_base END AS share_ck_base,
            CASE WHEN s.share_ck_full > 1 THEN NULL ELSE s.share_ck_full END AS share_ck_full,
            s.n_transitions,
            s.n_transitions_full,
            s.total_new_hires,
            s.total_new_hires_alt,
            s.total_new_hires_full,
            g.g_kt,
            g.g_kt_all,
            g.g_kt_intl,
            CASE WHEN s.share_ck > 1 THEN NULL ELSE s.share_ck END * g.g_kt AS z_ct_component,
            CASE WHEN s.share_ck > 1 THEN NULL ELSE s.share_ck END * g.g_kt_all AS z_ct_all_component,
            CASE WHEN s.share_ck > 1 THEN NULL ELSE s.share_ck END * g.g_kt_intl AS z_ct_intl_component,
            CASE WHEN s.share_ck_full > 1 THEN NULL ELSE s.share_ck_full END * g.g_kt AS z_ct_full_component,
            CASE WHEN s.share_ck_full > 1 THEN NULL ELSE s.share_ck_full END * g.g_kt_all AS z_ct_all_full_component,
            CASE WHEN s.share_ck_full > 1 THEN NULL ELSE s.share_ck_full END * g.g_kt_intl AS z_ct_intl_full_component
        FROM {shares_view} AS s
        JOIN {growth_view} AS g
          ON s.k = g.k
        WHERE g.g_kt IS NOT NULL
        """
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {panel} AS
        SELECT
            c,
            t,
            SUM(z_ct_component) AS z_ct,
            SUM(z_ct_all_component) AS z_ct_all,
            SUM(z_ct_intl_component) AS z_ct_intl,
            SUM(z_ct_full_component) AS z_ct_full,
            SUM(z_ct_all_full_component) AS z_ct_all_full,
            SUM(z_ct_intl_full_component) AS z_ct_intl_full,
            COUNT(
                DISTINCT CASE
                    WHEN share_ck IS NOT NULL AND share_ck <> 0
                     AND g_kt IS NOT NULL AND g_kt <> 0
                    THEN k
                    ELSE NULL
                END
            ) AS n_universities
        FROM {components}
        GROUP BY c, t
        """
    )
    return components, panel


def _sql_normalize(colname: str) -> str:
    return f"""
        TRIM(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    LOWER({colname}),
                    '[^a-z0-9 ]', ' ', 'g'
                ),
                '\\\\s+', ' ', 'g'
            )
        )
    """


def _sql_clean_company_name(companycol: str) -> str:
    suffix_regex = (
        "(?i)\\b("
        "inc|inc\\.|incorporated|llc|l\\.l\\.c|llp|l\\.l\\.p|lp|l\\.p|"
        "ltd|ltd\\.|limited|corp|corp\\.|corporation|company|co|co\\.|"
        "pllc|plc|pc|pc\\.|gmbh|ag|sa"
        ")\\b"
    )
    return _sql_normalize(f"REGEXP_REPLACE({companycol}, '{suffix_regex}', ' ', 'g')")


def _sql_state_name_to_abbr(statecol: str) -> str:
    mapping = {
        "alabama": "AL",
        "alaska": "AK",
        "arizona": "AZ",
        "arkansas": "AR",
        "california": "CA",
        "colorado": "CO",
        "connecticut": "CT",
        "delaware": "DE",
        "district of columbia": "DC",
        "washington dc": "DC",
        "dc": "DC",
        "florida": "FL",
        "georgia": "GA",
        "hawaii": "HI",
        "idaho": "ID",
        "illinois": "IL",
        "indiana": "IN",
        "iowa": "IA",
        "kansas": "KS",
        "kentucky": "KY",
        "louisiana": "LA",
        "maine": "ME",
        "maryland": "MD",
        "massachusetts": "MA",
        "michigan": "MI",
        "minnesota": "MN",
        "mississippi": "MS",
        "missouri": "MO",
        "montana": "MT",
        "nebraska": "NE",
        "nevada": "NV",
        "new hampshire": "NH",
        "new jersey": "NJ",
        "new mexico": "NM",
        "new york": "NY",
        "north carolina": "NC",
        "north dakota": "ND",
        "ohio": "OH",
        "oklahoma": "OK",
        "oregon": "OR",
        "pennsylvania": "PA",
        "rhode island": "RI",
        "south carolina": "SC",
        "south dakota": "SD",
        "tennessee": "TN",
        "texas": "TX",
        "utah": "UT",
        "vermont": "VT",
        "virginia": "VA",
        "washington": "WA",
        "west virginia": "WV",
        "wisconsin": "WI",
        "wyoming": "WY",
        "puerto rico": "PR",
        "guam": "GU",
        "american samoa": "AS",
        "northern mariana islands": "MP",
        "us virgin islands": "VI",
    }
    cases = "\n".join([f"WHEN LOWER(TRIM({statecol})) = '{name}' THEN '{abbr}'" for name, abbr in mapping.items()])
    return f"""
        CASE
            {cases}
            ELSE UPPER(TRIM({statecol}))
        END
    """


def _sql_clean_zip(zipcol: str) -> str:
    zipcolclean = f"TRIM(CAST(REGEXP_REPLACE({zipcol}, '[^0-9]', '', 'g') AS VARCHAR))"
    return f"""
        CASE
            WHEN LENGTH(TRIM(CAST({zipcolclean} AS VARCHAR))) = 4 THEN '0' || TRIM(CAST({zipcolclean} AS VARCHAR))
            WHEN LENGTH(TRIM(CAST({zipcolclean} AS VARCHAR))) >= 5 THEN SUBSTRING(TRIM(CAST({zipcolclean} AS VARCHAR)) FROM 1 FOR 5)
            ELSE TRIM(CAST({zipcolclean} AS VARCHAR))
        END
    """


def _date_parse_sql(col: str) -> str:
    return f"""
        COALESCE(
            TRY_CAST({col} AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%Y-%m-%d') AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%m/%d/%Y') AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%Y-%m-%d %H:%M:%S') AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%m/%d/%Y %H:%M:%S') AS DATE)
        )
    """


def _create_opt_counts(
    con: ddb.DuckDBPyConnection,
    outcome_lag_start: int,
    outcome_lag_end: int,
    use_changes: bool = False,
    include_non_masters: bool = False,
) -> str:
    # auth_start = f"COALESCE({_date_parse_sql('authorization_start_date')},{_date_parse_sql('opt_authorization_start_date')})"
    masters_clause = ""
    if not include_non_masters:
        predicate = _masters_predicate(con, view="foia_raw")
        if predicate:
            masters_clause = f"AND {predicate}"

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW foia_opt_authorizations_old AS
            SELECT person_id,
                {_sql_clean_company_name('employer_name')} AS f1_empname_clean,
                {_sql_normalize('employer_city')} AS f1_city_clean,
                {_sql_state_name_to_abbr('employer_state')} AS f1_state_clean,
                {_sql_clean_zip('employer_zip_code')} AS f1_zip_clean,
                MIN(EXTRACT(YEAR FROM program_end_date)) AS gradyear,
                MAX(CASE WHEN opt_employer_start_date >= program_end_date THEN 1 ELSE 0 END) AS valid_opt_hire
            FROM foia_raw_full
            WHERE employer_name IS NOT NULL {masters_clause}
            GROUP BY person_id, employer_name, employer_city, employer_state, employer_zip_code
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW person_post2014_correction_status AS
        SELECT
            f.person_id,
            MAX(
                CASE
                    WHEN f.year_int >= 2015 AND c.original_row_num IS NULL THEN 1
                    ELSE 0
                END
            ) AS has_post2014_correction
        FROM foia_raw_full AS f
        LEFT JOIN foia_raw AS c
          ON f.original_row_num = c.original_row_num
        WHERE f.person_id IS NOT NULL
        GROUP BY f.person_id
        """
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW foia_opt_authorizations_first_spell AS
        WITH base AS (
            SELECT
                person_id,
                {_sql_clean_company_name('employer_name')} AS f1_empname_clean,
                {_sql_normalize('employer_city')} AS f1_city_clean,
                {_sql_state_name_to_abbr('employer_state')} AS f1_state_clean,
                {_sql_clean_zip('employer_zip_code')} AS f1_zip_clean,
                EXTRACT(YEAR FROM program_end_date) AS gradyear,
                CASE WHEN opt_employer_start_date >= program_end_date THEN 1 ELSE 0 END AS valid_opt_hire,
                COALESCE(
                    {_date_parse_sql('opt_employer_start_date')},
                    {_date_parse_sql('opt_authorization_start_date')},
                    {_date_parse_sql('authorization_start_date')}
                ) AS spell_start_dt,
                original_row_num
            FROM foia_raw
            WHERE employer_name IS NOT NULL {masters_clause}
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY person_id, f1_empname_clean, f1_city_clean, f1_state_clean, f1_zip_clean
                    ORDER BY spell_start_dt ASC NULLS LAST, original_row_num ASC
                ) AS spell_rank
            FROM base
        )
        SELECT
            person_id,
            f1_empname_clean,
            f1_city_clean,
            f1_state_clean,
            f1_zip_clean,
            gradyear,
            valid_opt_hire
        FROM ranked
        WHERE spell_rank = 1
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW foia_opt_authorizations_correction_aware AS
        SELECT
            o.*,
            'old_fallback' AS correction_source
        FROM foia_opt_authorizations_old AS o
        LEFT JOIN person_post2014_correction_status AS pcs
          ON o.person_id = pcs.person_id
        WHERE COALESCE(pcs.has_post2014_correction, 0) = 0
        UNION ALL
        SELECT
            f.*,
            'first_spell' AS correction_source
        FROM foia_opt_authorizations_first_spell AS f
        JOIN person_post2014_correction_status AS pcs
          ON f.person_id = pcs.person_id
        WHERE pcs.has_post2014_correction = 1
        """
    )
    # con.sql(
    #     f"""
    #     CREATE OR REPLACE TEMP VIEW foia_opt_authorizations AS
    #     SELECT individual_key,
    #         {_sql_clean_company_name('employer_name')} AS f1_empname_clean,
    #         {_sql_normalize('employer_city')} AS f1_city_clean,
    #         {_sql_state_name_to_abbr('employer_state')} AS f1_state_clean,
    #         {_sql_clean_zip('employer_zip_code')} AS f1_zip_clean,
    #         {auth_start} AS auth_start
    #     FROM foia_raw
    #     WHERE employer_name IS NOT NULL AND EXTRACT(YEAR FROM {auth_start}) = year
    #       {masters_clause}
    #     """
    # )
    
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW opt_new_hires_old AS
        SELECT
            cw.preferred_rcid AS c,
            gradyear::INT AS t,
            COUNT(DISTINCT person_id) AS masters_opt_hires,
            COUNT(DISTINCT CASE WHEN valid_opt_hire = 1 THEN person_id END) AS valid_masters_opt_hires
        FROM foia_opt_authorizations_old AS f
        JOIN employer_crosswalk AS cw
          ON f.f1_empname_clean = cw.f1_empname_clean
         AND f.f1_city_clean = cw.f1_city_clean
         AND f.f1_state_clean = cw.f1_state_clean
         AND f.f1_zip_clean = cw.f1_zip_clean
        WHERE gradyear IS NOT NULL
          AND cw.preferred_rcid IS NOT NULL
        GROUP BY cw.preferred_rcid, gradyear::INT
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW opt_new_hires_correction_aware AS
        SELECT
            cw.preferred_rcid AS c,
            gradyear::INT AS t,
            COUNT(
                DISTINCT CASE
                    WHEN correction_source = 'old_fallback' AND valid_opt_hire = 1 THEN person_id
                    WHEN correction_source = 'first_spell' AND valid_opt_hire = 1 THEN person_id
                    ELSE NULL
                END
            ) AS masters_opt_hires_correction_aware
        FROM foia_opt_authorizations_correction_aware AS f
        JOIN employer_crosswalk AS cw
          ON f.f1_empname_clean = cw.f1_empname_clean
         AND f.f1_city_clean = cw.f1_city_clean
         AND f.f1_state_clean = cw.f1_state_clean
         AND f.f1_zip_clean = cw.f1_zip_clean
        WHERE gradyear IS NOT NULL
          AND cw.preferred_rcid IS NOT NULL
        GROUP BY cw.preferred_rcid, gradyear::INT
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW opt_new_hires_base AS
        SELECT
            COALESCE(o.c, n.c) AS c,
            COALESCE(o.t, n.t) AS t,
            COALESCE(o.masters_opt_hires, 0) AS masters_opt_hires,
            COALESCE(o.valid_masters_opt_hires, 0) AS valid_masters_opt_hires,
            COALESCE(n.masters_opt_hires_correction_aware, 0) AS masters_opt_hires_correction_aware
        FROM opt_new_hires_old AS o
        FULL OUTER JOIN opt_new_hires_correction_aware AS n
          ON o.c = n.c
         AND o.t = n.t
        """
    )
    lag_start = int(outcome_lag_start)
    lag_end = int(outcome_lag_end)
    x_lag_year_min = 2005
    x_lag_year_max = 2022
    x_lag_cols = ",\n            ".join(
        [
            (
                f"CASE WHEN (t + {lag}) < {x_lag_year_min} OR (t + {lag}) > {x_lag_year_max} "
                f"THEN NULL ELSE COALESCE(MAX(CASE WHEN lag = {lag} THEN x_lag END), 0) END "
                f"AS x_cst_lag{'m' + str(abs(lag)) if lag < 0 else str(lag)}"
            )
            for lag in range(lag_start, lag_end + 1)
        ]
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW opt_new_hires AS
        WITH base AS (
            SELECT
                c,
                t,
                COALESCE(masters_opt_hires, 0) AS masters_opt_hires,
                COALESCE(valid_masters_opt_hires, 0) AS valid_masters_opt_hires,
                COALESCE(masters_opt_hires_correction_aware, 0) AS masters_opt_hires_correction_aware
            FROM opt_new_hires_base
            WHERE c IS NOT NULL
              AND t IS NOT NULL
        ),
        bounds AS (
            SELECT
                c,
                MIN(t) AS min_t,
                MAX(t) AS max_t
            FROM base
            GROUP BY c
        ),
        expanded AS (
            SELECT
                b.c,
                gs.year AS t
            FROM bounds b,
            LATERAL generate_series(b.min_t, b.max_t) AS gs(year)
        ),
        filled AS (
            SELECT
                e.c,
                e.t,
                COALESCE(b.masters_opt_hires, 0) AS masters_opt_hires,
                COALESCE(b.valid_masters_opt_hires, 0) AS valid_masters_opt_hires,
                COALESCE(b.masters_opt_hires_correction_aware, 0) AS masters_opt_hires_correction_aware
            FROM expanded e
            LEFT JOIN base b
              ON e.c = b.c
             AND e.t = b.t
        ),
        long_lags AS (
            SELECT
                f.c,
                f.t,
                f.masters_opt_hires,
                f.valid_masters_opt_hires,
                f.masters_opt_hires_correction_aware,
                lag.lag AS lag,
                COALESCE(f2.masters_opt_hires_correction_aware, 0) AS x_lag
            FROM filled AS f
            CROSS JOIN LATERAL generate_series({lag_start}, {lag_end}) AS lag(lag)
            LEFT JOIN filled AS f2
              ON f.c = f2.c
             AND f2.t = f.t + lag.lag
        )
        SELECT
            c,
            t,
            MAX(masters_opt_hires) AS masters_opt_hires,
            MAX(valid_masters_opt_hires) AS valid_masters_opt_hires,
            MAX(masters_opt_hires_correction_aware) AS masters_opt_hires_correction_aware,
            {x_lag_cols}
        FROM long_lags
        GROUP BY c, t
        """
    )
    return "opt_new_hires"


def _create_outcome_views(
    con: ddb.DuckDBPyConnection,
    outcome_lag_start: int,
    outcome_lag_end: int,
    use_changes: bool = False,
) -> tuple[str, str]:
    """
    Map Revelio headcounts to hire-year t using an outcome-year lag range.
    If lag = 0, t == outcome year. If lag = 2, outcome_year = t + 2.
    Returns (outcomes_long_view, outcomes_wide_view).
    """
    lag_start = int(outcome_lag_start)
    lag_end = int(outcome_lag_end)

    def _lag_suffix(lag: int) -> str:
        return f"m{abs(lag)}" if lag < 0 else str(lag)

    y_expr = "y_cst"
    y_new_hires_expr = "y_new_hires"
    if use_changes:
        y_expr = "ASINH(y_cst) - ASINH(y_cst_lag)"
        y_new_hires_expr = "ASINH(y_new_hires) - ASINH(y_new_hires_lag)"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW outcomes_long AS
        WITH base_headcount_raw AS (
            SELECT
                CAST(rcid AS INTEGER) AS c,
                CAST(year AS INTEGER) AS outcome_year,
                CAST(total_headcount AS DOUBLE) AS y_cst
            FROM revelio_headcount
            WHERE rcid IN (SELECT rcid FROM matched_rcids)
              AND year IS NOT NULL
        ),
        base_headcount AS (
            SELECT
                c,
                outcome_year,
                COALESCE(MAX(y_cst), 0) AS y_cst
            FROM base_headcount_raw
            GROUP BY c, outcome_year
        ),
        base_new_hires_raw AS (
            SELECT
                CAST(rcid AS INTEGER) AS c,
                CAST(year AS INTEGER) AS outcome_year,
                CAST(total_new_hires AS DOUBLE) AS y_new_hires
            FROM revelio_transitions
            WHERE rcid IN (SELECT rcid FROM matched_rcids)
              AND year IS NOT NULL
              AND total_new_hires IS NOT NULL
        ),
        base_new_hires AS (
            SELECT
                c,
                outcome_year,
                COALESCE(MAX(y_new_hires), 0) AS y_new_hires
            FROM base_new_hires_raw
            GROUP BY c, outcome_year
        ),
        base_joined AS (
            SELECT
                COALESCE(h.c, n.c) AS c,
                COALESCE(h.outcome_year, n.outcome_year) AS outcome_year,
                COALESCE(h.y_cst, 0) AS y_cst,
                COALESCE(n.y_new_hires, 0) AS y_new_hires
            FROM base_headcount AS h
            FULL OUTER JOIN base_new_hires AS n
              ON h.c = n.c
             AND h.outcome_year = n.outcome_year
        ),
        bounds AS (
            SELECT
                c,
                MIN(outcome_year) AS min_year,
                MAX(outcome_year) AS max_year
            FROM base_joined
            GROUP BY c
        ),
        expanded AS (
            SELECT
                b.c,
                gs.year AS outcome_year
            FROM bounds b,
            LATERAL generate_series(b.min_year, b.max_year) AS gs(year)
        ),
        filled AS (
            SELECT
                e.c,
                e.outcome_year,
                COALESCE(b.y_cst, 0) AS y_cst,
                COALESCE(b.y_new_hires, 0) AS y_new_hires
            FROM expanded e
            LEFT JOIN base_joined b
              ON e.c = b.c
             AND e.outcome_year = b.outcome_year
        ),
        with_lags AS (
            SELECT
                c,
                outcome_year,
                y_cst,
                y_new_hires,
                LAG(y_cst) OVER (PARTITION BY c ORDER BY outcome_year) AS y_cst_lag,
                LAG(y_new_hires) OVER (PARTITION BY c ORDER BY outcome_year) AS y_new_hires_lag
            FROM filled
        )
        SELECT
            b.c,
            b.outcome_year AS s,
            b.outcome_year - lag.lag AS t,
            {y_expr} AS y_cst,
            {y_new_hires_expr} AS y_new_hires,
            lag.lag AS lag
        FROM with_lags AS b
        CROSS JOIN LATERAL generate_series({lag_start}, {lag_end}) AS lag(lag)
        """
    )

    outcome_cols_y = ",\n            ".join(
        [
            f"COALESCE(MAX(CASE WHEN lag = {lag} THEN y_cst END), 0) AS y_cst_lag{_lag_suffix(lag)}"
            for lag in range(lag_start, lag_end + 1)
        ]
    )
    outcome_cols_y_new_hires = ",\n            ".join(
        [
            f"COALESCE(MAX(CASE WHEN lag = {lag} THEN y_new_hires END), 0) AS y_new_hires_lag{_lag_suffix(lag)}"
            for lag in range(lag_start, lag_end + 1)
        ]
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW outcomes_wide AS
        SELECT
            c,
            t,
            {outcome_cols_y},
            {outcome_cols_y_new_hires}
        FROM outcomes_long
        WHERE t IS NOT NULL
        GROUP BY c, t
        """
    )
    return "outcomes_long", "outcomes_wide"


def _create_analysis_panel(
    con: ddb.DuckDBPyConnection,
    outcome_lag_start: int,
    outcome_lag_end: int,
    use_log_y: bool = False,
) -> str:
    def _lag_suffix(lag: int) -> str:
        return f"m{abs(lag)}" if lag < 0 else str(lag)

    lag_range = range(int(outcome_lag_start), int(outcome_lag_end) + 1)
    x_lag_year_min = 2005
    x_lag_year_max = 2022

    outcome_cols = []
    outcome_cols_new_hires = []
    x_lag_cols = []
    for lag in lag_range:
        suffix = _lag_suffix(lag)
        if use_log_y:
            outcome_cols.append(f"ASINH(o.y_cst_lag{suffix}) AS y_cst_lag{suffix}")
            outcome_cols_new_hires.append(f"ASINH(o.y_new_hires_lag{suffix}) AS y_new_hires_lag{suffix}")
        else:
            outcome_cols.append(f"o.y_cst_lag{suffix}")
            outcome_cols_new_hires.append(f"o.y_new_hires_lag{suffix}")
        x_lag_cols.append(
            f"CASE WHEN (o.t + {lag}) < {x_lag_year_min} OR (o.t + {lag}) > {x_lag_year_max} "
            f"THEN NULL ELSE COALESCE(x.x_cst_lag{suffix}, 0) END AS x_cst_lag{suffix}"
        )
    outcome_cols = ",\n            ".join(outcome_cols)
    outcome_cols_new_hires = ",\n            ".join(outcome_cols_new_hires)
    x_lag_cols = ",\n            ".join(x_lag_cols)

    x_expr = "COALESCE(x.masters_opt_hires, 0)"
    x_valid_expr = "COALESCE(x.valid_masters_opt_hires, 0)"
    x_corrected_expr = "COALESCE(x.masters_opt_hires_correction_aware, 0)"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW analysis_panel AS
        SELECT
            o.c,
            o.t,
            {outcome_cols},
            {outcome_cols_new_hires},
            {x_expr} AS masters_opt_hires,
            {x_valid_expr} AS valid_masters_opt_hires,
            {x_corrected_expr} AS masters_opt_hires_correction_aware,
            {x_lag_cols},
            instr.z_ct,
            instr.z_ct_all,
            instr.z_ct_intl,
            instr.z_ct_full,
            instr.z_ct_all_full,
            instr.z_ct_intl_full,
            instr.n_universities
        FROM (SELECT * FROM outcomes_wide WHERE t > 2005 AND t < 2025) AS o
        LEFT JOIN (SELECT * FROM opt_new_hires WHERE t > 2005 AND t < 2025) AS x USING (c, t)
        LEFT JOIN (SELECT * FROM company_instrument_panel WHERE t > 2005 AND t < 2025) AS instr USING (c, t)
        WHERE o.t IS NOT NULL
        """
    )
    return "analysis_panel"


def run_diagnostics(con: ddb.DuckDBPyConnection, plot: bool = True) -> None:
    """
    Run some basic diagnostics on the built outputs.
    """

    # Shifts
    print("Running diagnostics on shifts...")
    print(f"---Number of unique universities: {con.sql('SELECT COUNT(DISTINCT k) FROM ipeds_unit_growth').fetchone()[0]}")
    print(f"---Year range: {con.sql('SELECT MIN(t), MAX(t) FROM ipeds_unit_growth').fetchone()}")
    print(f"---Average shifts (g_kt, g_kt_all, g_kt_intl) across all k in 2015: {con.sql('SELECT AVG(g_kt), AVG(g_kt_all), AVG(g_kt_intl) FROM ipeds_unit_growth WHERE t = 2015').fetchone()}")
    
    if plot:
        sns.histplot(con.sql('SELECT g_kt AS "g_kt in 2015" FROM ipeds_unit_growth WHERE t = 2015 AND g_kt < 1000').df()['g_kt in 2015'], bins=50)
    
    # Shares
    print("Running diagnostics on shares...")
    print(f"---Number of companies with shares: {con.sql('SELECT COUNT(DISTINCT c) FROM transition_unit_shares').fetchone()[0]}")
    print(f"---Number of universities with shares: {con.sql('SELECT COUNT(DISTINCT k) FROM transition_unit_shares').fetchone()[0]}")
    print(f"---Mean of average company share (share_ck, share_ck_full): {con.sql('SELECT AVG(share_ck), AVG(share_ck_full) FROM (SELECT AVG(share_ck) AS share_ck, AVG(share_ck_full) AS share_ck_full FROM transition_unit_shares GROUP BY c)').fetchone()}")
    
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        sns.histplot(con.sql('SELECT share_ck AS "Average share_ck by company" FROM (SELECT AVG(share_ck) AS share_ck, AVG(share_ck_full) AS share_ck_full FROM transition_unit_shares GROUP BY c) WHERE share_ck <= 1').df()['Average share_ck by company'], bins=50)
        
    # Instruments
    print("Running diagnostics on instruments...")
    print(f"---Number of universities contributing to instruments: {con.sql('SELECT COUNT(DISTINCT k) FROM company_instrument_components').fetchone()[0]}")
    print(f"---Number of companies with instruments: {con.sql('SELECT COUNT(DISTINCT c) FROM company_instrument_panel').fetchone()[0]}")
    print(f"Year range of instruments: {con.sql('SELECT MIN(t), MAX(t) FROM company_instrument_panel').fetchone()}")
    print(f"---Mean z_ct, z_ct_all, z_ct_intl, z_ct_full, z_ct_all_full, z_ct_intl_full: {con.sql('SELECT AVG(z_ct), AVG(z_ct_all), AVG(z_ct_intl), AVG(z_ct_full), AVG(z_ct_all_full), AVG(z_ct_intl_full) FROM company_instrument_panel').fetchone()}")
    
    # Independent Variable
    print("Running diagnostics on independent variable (treatment)...")
    print(f"---Number of companies with Master's OPT hires: {con.sql('SELECT COUNT(DISTINCT c) FROM opt_new_hires').fetchone()[0]}")
    print(f"---Number of company-year observations with Master's OPT hires: {con.sql('SELECT COUNT(*) FROM opt_new_hires').fetchone()[0]}")
    print(f"---Year range of Master's OPT hires: {con.sql('SELECT MIN(t), MAX(t) FROM opt_new_hires').fetchone()}")
    print(f"---Mean Master's OPT hires (hires, valid hires) per company-year: {con.sql('SELECT AVG(masters_opt_hires), AVG(valid_masters_opt_hires) FROM opt_new_hires').fetchone()}")
    
    # Dependent Variable
    print("Running diagnostics on dependent variable (outcomes)...")
    print(f"---Number of companies with outcomes: {con.sql('SELECT COUNT(DISTINCT c) FROM outcomes_long').fetchone()[0]}")
    print(f"---Number of company-year observations with outcomes: {con.sql('SELECT COUNT(*) FROM outcomes_long').fetchone()[0]}")
    print(f"---Year range of outcomes: {con.sql('SELECT MIN(s), MAX(s) FROM outcomes_long').fetchone()}")
    lag_cols = con.sql("PRAGMA table_info('outcomes_wide')").fetchall()
    lag_col_names = [r[1] for r in lag_cols if r[1].startswith("y_cst_lag0")]
    for col in lag_col_names:
        print(f"---Mean outcome for {col}: {con.sql(f'SELECT AVG({col}) FROM outcomes_wide WHERE t < 2020 AND t > 2010').fetchone()[0]}")
    
    # Analysis Panel
    print("Running diagnostics on analysis panel...")
    print(f"---Number of companies in analysis panel: {con.sql('SELECT COUNT(DISTINCT c) FROM analysis_panel WHERE z_ct IS NOT NULL AND masters_opt_hires IS NOT NULL').fetchone()[0]}")
    print(f"---Number of company-year observations in analysis panel: {con.sql('SELECT COUNT(*) FROM analysis_panel WHERE z_ct IS NOT NULL AND masters_opt_hires IS NOT NULL').fetchone()[0]}")
    print(f"---Year range of analysis panel: {con.sql('SELECT MIN(t), MAX(t) FROM analysis_panel').fetchone()}")
    
    # First Stage
    print("Running first-stage diagnostics...")
    fs = con.sql("SELECT z_ct, z_ct_all, z_ct_intl, masters_opt_hires, valid_masters_opt_hires, c, t FROM analysis_panel WHERE z_ct IS NOT NULL AND masters_opt_hires IS NOT NULL").df()
    # overlapping histograms of z_ct and masters_opt_hires
    if plot:
        plt.figure()
        sns.histplot(fs['z_ct'], bins=50)
        sns.histplot(fs['masters_opt_hires'], bins=50)
    
    # correlation matrix between z_ct, z_ct_all, z_ct_intl and masters_opt_hires, valid_masters_opt_hires
    corr_matrix = fs[['z_ct', 'z_ct_all', 'z_ct_intl', 'masters_opt_hires', 'valid_masters_opt_hires']].corr()
    print(f"---Correlation matrix:\n{corr_matrix}")
    
    # binned scatterplot
    if plot:
        plt.figure()
        sns.regplot(x='z_ct_intl', y='valid_masters_opt_hires', data=fs, lowess=True, scatter_kws={'s':10}, line_kws={'color':'red'})

def build_pipeline(
    paths: PipelinePaths,
    *,
    outcome_lag_start: int = 0,
    outcome_lag_end: int = 5,
    share_base_year: int = 2010,
    save_outputs: bool = True,
    verbose: bool = True,
    plot_diagnostics: bool = True,
    use_changes: bool = False,
    use_log_y: bool = False,
    include_non_masters: bool = False,
    exclude_unitids: list[str] | None = None,
) -> None:
    con = ddb.connect()
    _register_inputs(con, paths)

    if use_changes and use_log_y:
        raise ValueError("use_changes and use_log_y are mutually exclusive.")

    growth_view = _create_ipeds_growth_view(
        con,
        use_changes=use_changes,
        demean_by_school=use_log_y,
        exclude_unitids=exclude_unitids,
    )
    shares_view = _create_transition_share_view(con, share_base_year, exclude_unitids=exclude_unitids)
    _create_instrument_views(con, shares_view, growth_view)
    if outcome_lag_end < outcome_lag_start:
        raise ValueError("outcome_lag_end must be >= outcome_lag_start.")
    _create_outcome_views(con, outcome_lag_start, outcome_lag_end, use_changes=use_changes)
    _create_opt_counts(
        con,
        outcome_lag_start=outcome_lag_start,
        outcome_lag_end=outcome_lag_end,
        use_changes=use_changes,
        include_non_masters=include_non_masters,
    )
    _create_analysis_panel(con, outcome_lag_start, outcome_lag_end, use_log_y=use_log_y)

    if not save_outputs:
        print("Skipping writes (save_outputs=False).")
        return

    _write_view(con, "company_instrument_components", paths.instrument_components_out)
    _write_view(con, "company_instrument_panel", paths.instrument_panel_out)
    _write_view(con, "opt_new_hires", paths.treatment_out)
    _write_view(con, "outcomes_long", paths.outcomes_out)
    _write_view(con, "analysis_panel", paths.analysis_panel_out)
    
    if verbose:
        run_diagnostics(con, plot=plot_diagnostics)
        


def _parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build company-level shift-share pipeline outputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--outcome-lag",
        type=int,
        default=None,
        help="Outcome year minus hire year (overrides lag range when set).",
    )
    parser.add_argument(
        "--share-base-year",
        type=int,
        default=None,
        help="Base year to compute company-university shares (default: 2010).",
    )
    parser.add_argument(
        "--outcome-lag-start",
        type=int,
        default=None,
        help="Minimum outcome lag (inclusive).",
    )
    parser.add_argument(
        "--outcome-lag-end",
        type=int,
        default=None,
        help="Maximum outcome lag (inclusive).",
    )
    parser.add_argument(
        "--include-non-masters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use all FOIA records (skip master's-only filter).",
    )
    parser.add_argument(
        "--no-write",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip writing parquet outputs (useful for quick checks).",
    )
    parser.add_argument(
        "--use-changes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use IHS differences for g_kt and outcomes: asinh(x) - asinh(lag(x)).",
    )
    parser.add_argument(
        "--use-log-y",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Transform outcomes with inverse hyperbolic sine: y=asinh(y).",
    )
    parser.add_argument(
        "--exclude-unitids",
        default="",
        help="Comma-separated UNITIDs to exclude from both growth and transition shares.",
    )
    return parser.parse_args(args)

def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(cli_args)
    cfg = load_config(args.config)
    paths = _resolve_pipeline_paths(cfg)
    pipeline_cfg = get_cfg_section(cfg, "build_company_shift_share")

    share_base_year = args.share_base_year if args.share_base_year is not None else pipeline_cfg.get("share_base_year", 2010)
    lag_start_default = pipeline_cfg.get("outcome_lag_start", 0)
    lag_end_default = pipeline_cfg.get("outcome_lag_end", 5)
    include_non_masters = (
        args.include_non_masters
        if args.include_non_masters is not None
        else pipeline_cfg.get("include_non_masters", False)
    )
    save_outputs = (
        not args.no_write
        if args.no_write is not None
        else pipeline_cfg.get("save_outputs", True)
    )
    use_changes = args.use_changes if args.use_changes is not None else pipeline_cfg.get("use_changes", False)
    use_log_y = args.use_log_y if args.use_log_y is not None else pipeline_cfg.get("use_log_y", pipeline_cfg.get("use_log_rate", False))
    exclude_unitids = _parse_unitids(args.exclude_unitids) if args.exclude_unitids else pipeline_cfg.get("exclude_unitids", [])
    if args.outcome_lag is not None:
        outcome_lag_start = args.outcome_lag
        outcome_lag_end = args.outcome_lag
    else:
        outcome_lag_start = args.outcome_lag_start if args.outcome_lag_start is not None else lag_start_default
        outcome_lag_end = args.outcome_lag_end if args.outcome_lag_end is not None else lag_end_default
    build_pipeline(
        paths=paths,
        outcome_lag_start=outcome_lag_start,
        outcome_lag_end=outcome_lag_end,
        share_base_year=share_base_year,
        save_outputs=save_outputs,
        verbose=pipeline_cfg.get("verbose", True),
        plot_diagnostics=pipeline_cfg.get("plot_diagnostics", True),
        use_changes=use_changes,
        use_log_y=use_log_y,
        include_non_masters=include_non_masters,
        exclude_unitids=exclude_unitids,
    )


# if __name__ == "__main__":
#     main()

# TEMP -- ipython
cfg = load_config(DEFAULT_CONFIG_PATH)
paths = _resolve_pipeline_paths(cfg)

con = ddb.connect()
_register_inputs(con, paths)

use_changes = False
use_log_y = False
include_non_masters = False
exclude_unitids = None
share_base_year = 2010
outcome_lag_start = -3
outcome_lag_end = 6

growth_view = _create_ipeds_growth_view(
    con,
    use_changes=use_changes,
    demean_by_school=use_log_y,
    exclude_unitids=exclude_unitids,
)

shares_view = _create_transition_share_view(con, share_base_year, exclude_unitids=exclude_unitids)
_create_instrument_views(con, shares_view, growth_view)
if outcome_lag_end < outcome_lag_start:
    raise ValueError("outcome_lag_end must be >= outcome_lag_start.")
_create_outcome_views(con, outcome_lag_start, outcome_lag_end, use_changes=use_changes)
_create_opt_counts(
    con,
    outcome_lag_start=outcome_lag_start,
    outcome_lag_end=outcome_lag_end,
    use_changes=use_changes,
    include_non_masters=include_non_masters,
)
_create_analysis_panel(con, outcome_lag_start, outcome_lag_end, use_log_y=use_log_y)


_write_view(con, "company_instrument_components", paths.instrument_components_out)
_write_view(con, "company_instrument_panel", paths.instrument_panel_out)
_write_view(con, "opt_new_hires", paths.treatment_out)
_write_view(con, "outcomes_long", paths.outcomes_out)
_write_view(con, "analysis_panel", paths.analysis_panel_out)

#run_diagnostics(con, plot=True)
