"""Shift-share regressions using OPT authorizations and Revelio new hires.

The unit of observation is RCID × year (t = calendar year - 2 to match
the IPEDS growth timing used in ihma_clean.py). Shares are computed from
Revelio school→employer transitions (mapped to UNITIDs), shifts from the
IPEDS growth series, the outcome is total new hires, and the endogenous
variable is the count of OPT authorizations that start in the year.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import duckdb as ddb
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *  # noqa: F401,F403

INT_FOLDER = Path(root) / "data" / "int" / "int_files_nov2025"
TRANSITIONS_PATH = INT_FOLDER / "revelio_school_to_employer_transitions.parquet"
FOIA_RAW_PATH = INT_FOLDER / "foia_sevp_combined_raw.parquet"
EMPLOYER_CROSSWALK_PATH = INT_FOLDER / "f1_employer_final_crosswalk.parquet"
RSID_UNITID_CROSSWALK_PATH = Path(root) / "data" / "int" / "rsid_ipeds_cw.parquet"
IPEDS_MA_PATH = INT_FOLDER / "ipeds_completions_all.parquet"

STATE_NAME_TO_ABBR = {
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


def _escape(path: Path) -> str:
    return str(path).replace("'", "''")


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


def _sql_clean_zip(zipcol: str) -> str:
    zipcolclean = f"TRIM(CAST(REGEXP_REPLACE({zipcol}, '[^0-9]', '', 'g') AS VARCHAR))"
    return f"""
        CASE
            WHEN LENGTH(TRIM(CAST({zipcolclean} AS VARCHAR))) = 4 THEN '0' || TRIM(CAST({zipcolclean} AS VARCHAR))
            WHEN LENGTH(TRIM(CAST({zipcolclean} AS VARCHAR))) >= 5 THEN SUBSTRING(TRIM(CAST({zipcolclean} AS VARCHAR)) FROM 1 FOR 5)
            ELSE TRIM(CAST({zipcolclean} AS VARCHAR))
        END
    """


def _sql_state_name_to_abbr(statecol: str) -> str:
    cases = "\n".join([f"WHEN LOWER(TRIM({statecol})) = '{name}' THEN '{abbr}'" for name, abbr in STATE_NAME_TO_ABBR.items()])
    return f"""
        CASE
            {cases}
            ELSE UPPER(TRIM({statecol}))
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


def _check_paths(paths: Iterable[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        names = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required input files:\n{names}")


def _register_inputs(con: ddb.DuckDBPyConnection) -> None:
    _check_paths(
        [
            TRANSITIONS_PATH,
            FOIA_RAW_PATH,
            EMPLOYER_CROSSWALK_PATH,
            RSID_UNITID_CROSSWALK_PATH,
            IPEDS_MA_PATH,
        ]
    )
    con.sql(f"CREATE OR REPLACE TEMP VIEW revelio_transitions AS SELECT * FROM read_parquet('{_escape(TRANSITIONS_PATH)}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW rsid_unitid AS SELECT * FROM read_parquet('{_escape(RSID_UNITID_CROSSWALK_PATH)}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{_escape(FOIA_RAW_PATH)}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW employer_crosswalk AS SELECT * FROM read_parquet('{_escape(EMPLOYER_CROSSWALK_PATH)}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw AS SELECT * FROM read_parquet('{_escape(IPEDS_MA_PATH)}')")
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW matched_rcids AS
        SELECT DISTINCT preferred_rcid AS rcid
        FROM employer_crosswalk
        WHERE preferred_rcid IS NOT NULL
        """
    )


def _create_ipeds_unit_growth(con: ddb.DuckDBPyConnection) -> str:
    view_name = "ipeds_unit_growth"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        WITH base AS (
            SELECT
                CAST(unitid AS VARCHAR) AS k,
                CAST(cipcode AS VARCHAR) AS cipcode,
                CAST(year AS INTEGER) AS year,
                CAST(cnralt AS DOUBLE) AS cnralt,
                CAST(ctotalt AS DOUBLE) AS ctotalt,
                CAST(share_intl AS DOUBLE) AS share_intl,
                CAST(STEMOPT AS INTEGER) AS STEMOPT
            FROM ipeds_raw
            WHERE unitid IS NOT NULL AND awlevel_group = 'Master'
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
                         AND STEMOPT = 1
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
                SUM(COALESCE(b.ctotalt, 0) * COALESCE(p.international_heavy, 0)) AS tot_seats_ihma
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
                COALESCE(j.tot_seats_ihma, 0) AS tot_seats_ihma
            FROM expanded e
            LEFT JOIN joined j
                ON e.k = j.k
               AND e.year = j.year
        )
        SELECT
            k,
            year AS t,
            tot_intl_students AS g_kt_all,
            tot_seats_ihma AS g_kt
        FROM filled
        WHERE g_kt IS NOT NULL
        """
    )
    return view_name


def _create_share_view(con: ddb.DuckDBPyConnection) -> str:
    """Compute rcid×unitid shares using Revelio transitions."""

    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW transitions_unitid AS
        SELECT
            t.rcid AS i,
            cw.unitid AS k,
            t.year,
            t.n_transitions,
            t.total_new_hires
        FROM revelio_transitions AS t
        JOIN matched_rcids mr ON t.rcid = mr.rcid
        LEFT JOIN rsid_unitid AS cw
          ON CAST(t.rsid AS VARCHAR) = CAST(cw.rsid AS VARCHAR)
        WHERE t.rcid IS NOT NULL
          AND cw.unitid IS NOT NULL
          AND t.n_transitions IS NOT NULL
        """
    )

    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW rcid_total_hires AS
        SELECT
            rcid AS i,
            year,
            MAX(total_new_hires) AS total_new_hires
        FROM revelio_transitions
        WHERE rcid IS NOT NULL
          AND total_new_hires IS NOT NULL
          AND rcid IN (SELECT rcid FROM matched_rcids)
        GROUP BY rcid, year
        """
    )

    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW employer_unit_shares AS
        WITH totals AS (
            SELECT i, SUM(total_new_hires) AS total_new_hires
            FROM rcid_total_hires
            GROUP BY i
        )
        SELECT
            trans.i,
            trans.k,
            SUM(trans.n_transitions) AS n_transitions,
            totals.total_new_hires,
            SUM(trans.n_transitions) / NULLIF(totals.total_new_hires, 0) AS share_ik
        FROM transitions_unitid AS trans
        JOIN totals USING (i)
        GROUP BY trans.i, trans.k, totals.total_new_hires
        HAVING totals.total_new_hires IS NOT NULL
        """
    )
    return "employer_unit_shares"


def _create_instrument_views(con: ddb.DuckDBPyConnection, shares_view: str, growth_view: str) -> tuple[str, str]:
    components = "employer_instrument_components"
    panel = "employer_instrument_panel"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {components} AS
        SELECT
            s.i,
            g.k,
            g.t,
            s.share_ik,
            g.g_kt,
            g.g_kt_all,
            s.share_ik * g.g_kt AS z_it,
            s.share_ik * g.g_kt_all AS z_it_all
        FROM {shares_view} AS s
        JOIN {growth_view} AS g
          ON s.k = g.k
        """
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {panel} AS
        SELECT
            i,
            t,
            SUM(z_it) AS z_it,
            SUM(z_it_all) AS z_it_all
        FROM {components}
        GROUP BY i, t
        """
    )
    return components, panel


def _create_opt_counts(con: ddb.DuckDBPyConnection) -> str:
    """Compute OPT authorizations (x_it) by rcid and year."""

    auth_start = f"COALESCE({_date_parse_sql('opt_authorization_start_date')}, {_date_parse_sql('authorization_start_date')})"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW foia_opt_authorizations AS
        SELECT
            {_sql_clean_company_name('employer_name')} AS f1_empname_clean,
            {_sql_normalize('employer_city')} AS f1_city_clean,
            {_sql_state_name_to_abbr('employer_state')} AS f1_state_clean,
            {_sql_clean_zip('employer_zip_code')} AS f1_zip_clean,
            {auth_start} AS auth_start
        FROM foia_raw
        WHERE employer_name IS NOT NULL
        """
    )

    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW opt_new_hires AS
        SELECT
            cw.preferred_rcid AS i,
            EXTRACT(YEAR FROM auth_start)::INT - 2 AS t,
            COUNT(*) AS opt_new_hires
        FROM foia_opt_authorizations AS f
        JOIN employer_crosswalk AS cw
          ON f.f1_empname_clean = cw.f1_empname_clean
         AND f.f1_city_clean = cw.f1_city_clean
         AND f.f1_state_clean = cw.f1_state_clean
         AND f.f1_zip_clean = cw.f1_zip_clean
        WHERE auth_start IS NOT NULL
          AND cw.preferred_rcid IS NOT NULL
        GROUP BY cw.preferred_rcid, EXTRACT(YEAR FROM auth_start)::INT - 2
        """
    )
    return "opt_new_hires"


def _create_outcome_view(con: ddb.DuckDBPyConnection) -> str:
    """Total new hires per rcid-year (aligned t = year - 2)."""

    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW employer_outcomes AS
        SELECT
            rcid AS i,
            year - 2 AS t,
            MAX(total_new_hires) AS y_it
        FROM revelio_transitions
        WHERE rcid IS NOT NULL
          AND total_new_hires IS NOT NULL
          AND rcid IN (SELECT rcid FROM matched_rcids)
        GROUP BY rcid, year
        """
    )
    return "employer_outcomes"


def _build_regression_panel(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    panel = con.sql(
        """
        SELECT
            y.i,
            y.t,
            y.y_it,
            x.opt_new_hires AS x_it,
            instr.z_it,
            instr.z_it_all
        FROM employer_outcomes AS y
        LEFT JOIN opt_new_hires AS x USING (i, t)
        LEFT JOIN employer_instrument_panel AS instr USING (i, t)
        WHERE y.y_it IS NOT NULL
        """
    ).df()
    panel["i"] = panel["i"].astype(str)
    panel["x_it"] = panel["x_it"].fillna(0)
    panel = panel.dropna(subset=["z_it", "y_it"])
    return panel


def run_regressions(panel: pd.DataFrame) -> dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
    work = panel.copy()
    fe_term = " + C(i) + C(t)"

    ols = sm.OLS.from_formula(f"y_it ~ x_it{fe_term}", data=work).fit()
    first_stage = sm.OLS.from_formula(f"x_it ~ z_it{fe_term}", data=work).fit()
    reduced = sm.OLS.from_formula(f"y_it ~ z_it{fe_term}", data=work).fit()
    iv = IV2SLS.from_formula(
        f"y_it ~ 1{fe_term} + [x_it ~ z_it]",
        data=work,
    ).fit(cov_type="robust")

    return {"ols": ols, "first_stage": first_stage, "reduced": reduced, "iv": iv}


def main() -> tuple[pd.DataFrame, dict[str, sm.regression.linear_model.RegressionResultsWrapper]]:
    con = ddb.connect()
    _register_inputs(con)

    growth_view = _create_ipeds_unit_growth(con)
    shares_view = _create_share_view(con)
    _create_instrument_views(con, shares_view, growth_view)
    _create_opt_counts(con)
    _create_outcome_view(con)

    panel = _build_regression_panel(con)
    if panel.empty:
        raise RuntimeError("Regression panel is empty; check inputs and joins.")

    results = run_regressions(panel)
    print(f"Panel rows: {len(panel):,} | rcids: {panel['i'].nunique():,} | years: {panel['t'].nunique():,}")
    print("OLS coef (x_it):", results["ols"].params.get("x_it"))
    print("IV coef (x_it):", results["iv"].params.get("x_it"))
    return panel, results


if __name__ == "__main__":
    main()
