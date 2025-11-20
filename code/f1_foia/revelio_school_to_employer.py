# Script: Generate school-to-employer transition shares from Revelio
# Pulls US positions and education from WRDS, keeps first post-graduation
# long-duration job per education, and aggregates flows by school (rsid),
# employer (rcid), and start year.

import os
import sys
from typing import Tuple

import pandas as pd
import wrds

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *  # noqa: F401,F403

INT_FOLDER = f"{root}/data/int/int_files_nov2025"

# Output locations
TRANSITION_OUTFILE = f"{INT_FOLDER}/revelio_school_to_employer_transitions.parquet"

# Parameters
MIN_POSITION_DAYS = 365  # drop internships/roles shorter than this


def run_queries(db: wrds.Connection, test = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the WRDS queries that produce:
      1) School -> employer transitions with new-hire counts and shares.
      2) Employer-level new-hire counts by year (for all rcids, even
         those without transitions from the education sample).
    """
    if test is not None:
        testlimit = f"LIMIT {test}"
    else:
        testlimit = ""

    base_cte = f"""
    WITH us_positions AS (
        SELECT
            user_id,
            rcid,
            startdate::DATE AS startdate,
            COALESCE(enddate::DATE, '2025-12-31') AS enddate
        FROM revelio.individual_positions
        WHERE country = 'United States'
          AND rcid IS NOT NULL
          AND startdate IS NOT NULL
            {testlimit}
    ),
    long_positions AS (
        SELECT user_id, rcid, startdate
        FROM us_positions
        WHERE enddate >= startdate + INTERVAL '{MIN_POSITION_DAYS} days'
    ),
    education AS (
        SELECT
            user_id,
            education_number,
            rsid,
            enddate::DATE AS grad_date
        FROM revelio.individual_user_education
        WHERE rsid IS NOT NULL
          AND enddate IS NOT NULL
    ),
    positions_after_grad AS (
        SELECT
            e.user_id,
            e.rsid,
            e.education_number,
            p.rcid,
            p.startdate,
            -- for now, partition by rsid (will take earliest grad x position, may count positions while still at school)
            ROW_NUMBER() OVER (
                PARTITION BY e.user_id, e.rsid
                ORDER BY p.startdate
            ) AS rank_after_grad
        FROM education e
        JOIN long_positions p
          ON e.user_id = p.user_id
         AND p.startdate >= e.grad_date
         AND p.startdate <= e.grad_date + INTERVAL '1 years'
    ),
    first_jobs AS (
        SELECT *
        FROM positions_after_grad
        WHERE rank_after_grad = 1
    ),
    new_hires AS (
        SELECT user_id, rcid, MIN(startdate) AS first_start
        FROM long_positions
        GROUP BY user_id, rcid
    )
    """

    transition_query = f"""
    {base_cte},
    transition_counts AS (
        SELECT
            rsid,
            rcid,
            EXTRACT(YEAR FROM startdate)::INT AS year,
            COUNT(DISTINCT user_id) AS n_transitions
        FROM first_jobs
        GROUP BY rsid, rcid, year
    ),
    new_hire_counts AS (
        SELECT
            rcid,
            EXTRACT(YEAR FROM first_start)::INT AS year,
            COUNT(DISTINCT user_id) AS total_new_hires
        FROM new_hires
        GROUP BY rcid, year
    )
    SELECT
        t.rsid,
        t.rcid,
        t.year,
        t.n_transitions,
        n.total_new_hires,
        CASE
            WHEN n.total_new_hires IS NULL OR n.total_new_hires = 0 THEN NULL
            ELSE t.n_transitions::DECIMAL / n.total_new_hires
        END AS share_of_new_hires
    FROM transition_counts t
    LEFT JOIN new_hire_counts n
      ON t.rcid = n.rcid AND t.year = n.year
    ORDER BY t.rsid, t.rcid, t.year
    """

    transitions = db.raw_sql(transition_query)
    return transitions

def save_with_fallback(df: pd.DataFrame, parquet_path: str) -> None:
    """
    Save to parquet when possible; fall back to CSV if parquet engines are
    unavailable in the runtime environment.
    """
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Wrote {parquet_path}")
    except Exception as exc:  # pragma: no cover - best-effort convenience
        csv_path = os.path.splitext(parquet_path)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"Parquet save failed ({exc}); wrote {csv_path} instead")


def main() -> None:
    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username="amykimecon")
    print("Running transition and new-hire queries (this can take a while)...")
    transitions = run_queries(db)

    save_with_fallback(transitions, TRANSITION_OUTFILE)
    print("Done.")


if __name__ == "__main__":
    main()
