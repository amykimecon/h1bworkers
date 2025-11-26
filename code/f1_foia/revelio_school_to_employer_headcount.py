# Script: Compute employer headcounts using existing school-to-employer transitions
# Uses rcids from a precomputed transitions parquet and queries WRDS for US
# positions to count long-duration employees by rcid-year and rcid-rsid-year.

from __future__ import annotations

import argparse
import os
from typing import Iterable

import pandas as pd
import wrds

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *  # noqa: F401,F403
import helpers as help  # noqa: E402

INT_FOLDER = f"{root}/data/int/int_files_nov2025"
TRANSITION_OUTFILE = f"{INT_FOLDER}/revelio_school_to_employer_transitions.parquet"
HEADCOUNT_OUTFILE = f"{INT_FOLDER}/revelio_school_to_employer_headcount.parquet"
HEADCOUNT_BY_RSID_OUTFILE = f"{INT_FOLDER}/revelio_school_to_employer_origin_headcount.parquet"
MIN_POSITION_DAYS = 365


def _load_rcids_from_parquet(path: str, verbose: bool) -> list[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transitions parquet not found: {path}")
    df = pd.read_parquet(path, columns=["rcid"])
    rcids = sorted({int(x) for x in df["rcid"].dropna().unique().tolist()})
    if verbose:
        print(f"Loaded {len(rcids):,} rcids from {path}")
    return rcids


def _values_clause(seq: Iterable[int]) -> str:
    cleaned = [int(x) for x in seq]
    if not cleaned:
        return ""
    return ",".join(f"({x})" for x in cleaned)


def compute_headcount(
    db: wrds.Connection,
    rcids: Iterable[int],
    *,
    min_position_days: int = MIN_POSITION_DAYS,
    verbose: bool = True,
) -> pd.DataFrame:
    values_clause = _values_clause(rcids)
    if not values_clause:
        return pd.DataFrame(columns=["rcid", "year", "total_us_employees"])

    query = f"""
        WITH rcids AS (
            SELECT column1::INT AS rcid
            FROM (VALUES {values_clause}) AS v(column1)
        ),
        us_positions AS (
            SELECT
                p.user_id,
                p.rcid,
                p.startdate::DATE AS startdate,
                COALESCE(p.enddate::DATE, '2025-12-31') AS enddate
            FROM revelio.individual_positions AS p
            JOIN rcids USING (rcid)
            WHERE p.country = 'United States'
              AND p.rcid IS NOT NULL
              AND p.startdate IS NOT NULL
        ),
        long_positions AS (
            SELECT *
            FROM us_positions
            WHERE enddate >= startdate + INTERVAL '{int(min_position_days)} days'
        ),
        headcount AS (
            SELECT
                lp.rcid,
                gs.year,
                COUNT(DISTINCT lp.user_id) AS total_us_employees
            FROM long_positions AS lp,
            LATERAL generate_series(EXTRACT(YEAR FROM lp.startdate)::INT, EXTRACT(YEAR FROM lp.enddate)::INT) AS gs(year)
            GROUP BY lp.rcid, gs.year
        )
        SELECT rcid, year, total_us_employees
        FROM headcount
        ORDER BY rcid, year
    """
    if verbose:
        print("Running rcid-year headcount query...")
    return db.raw_sql(query)


def compute_origin_headcount(
    db: wrds.Connection,
    rcids: Iterable[int],
    *,
    min_position_days: int = MIN_POSITION_DAYS,
    verbose: bool = True,
) -> pd.DataFrame:
    values_clause = _values_clause(rcids)
    if not values_clause:
        return pd.DataFrame(columns=["rcid", "rsid", "year", "total_us_employees_from_rsid"])

    query = f"""
        WITH rcids AS (
            SELECT column1::INT AS rcid
            FROM (VALUES {values_clause}) AS v(column1)
        ),
        us_positions AS (
            SELECT
                p.user_id,
                p.rcid,
                p.startdate::DATE AS startdate,
                COALESCE(p.enddate::DATE, '2025-12-31') AS enddate
            FROM revelio.individual_positions AS p
            JOIN rcids USING (rcid)
            WHERE p.country = 'United States'
              AND p.rcid IS NOT NULL
              AND p.startdate IS NOT NULL
        ),
        long_positions AS (
            SELECT *
            FROM us_positions
            WHERE enddate >= startdate + INTERVAL '{int(min_position_days)} days'
        ),
        primary_rsid AS (
            SELECT user_id, rsid
            FROM (
                SELECT
                    user_id,
                    rsid,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY enddate::DATE DESC NULLS LAST
                    ) AS rn
                FROM revelio.individual_user_education
                WHERE rsid IS NOT NULL
            )
            WHERE rn = 1
        ),
        origin_headcount AS (
            SELECT
                lp.rcid,
                pr.rsid,
                gs.year,
                COUNT(DISTINCT lp.user_id) AS total_us_employees_from_rsid
            FROM long_positions AS lp
            JOIN primary_rsid AS pr
              ON lp.user_id = pr.user_id
            ,
            LATERAL generate_series(EXTRACT(YEAR FROM lp.startdate)::INT, EXTRACT(YEAR FROM lp.enddate)::INT) AS gs(year)
            GROUP BY lp.rcid, pr.rsid, gs.year
        )
        SELECT rcid, rsid, year, total_us_employees_from_rsid
        FROM origin_headcount
        ORDER BY rcid, rsid, year
    """
    if verbose:
        print("Running rcid-rsid-year headcount query...")
    return db.raw_sql(query)


def _chunked_compute(
    rcids: list[int],
    compute_fn,
    *,
    db: wrds.Connection,
    min_position_days: int,
    chunks: int,
    chunk_size: int,
    verbose: bool,
) -> pd.DataFrame:
    """Helper to chunk rcid list when calling compute_fn."""
    if not rcids:
        return pd.DataFrame()

    rcid_df = pd.DataFrame({"rcid": rcids})

    def _run_chunk(subset: pd.DataFrame) -> pd.DataFrame:
        if subset.empty:
            return pd.DataFrame()
        return compute_fn(
            db,
            subset["rcid"].tolist(),
            min_position_days=min_position_days,
            verbose=False,
        )

    chunked = help.chunk_query(
        rcid_df,
        j=max(chunks, 1),
        fun=_run_chunk,
        d=chunk_size,
        verbose=verbose,
    )
    if chunked is None or chunked.empty:
        return pd.DataFrame()

    # Deduplicate/aggregate in case of overlap across chunks
    group_cols = [c for c in chunked.columns if c in {"rcid", "rsid", "year"}]
    agg_cols = [c for c in chunked.columns if c not in group_cols]
    if group_cols and agg_cols:
        chunked = chunked.groupby(group_cols, as_index=False)[agg_cols].sum()
    return chunked


def save_with_fallback(df: pd.DataFrame, parquet_path: str) -> None:
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Wrote {parquet_path}")
    except Exception as exc:  # pragma: no cover - best effort
        csv_path = os.path.splitext(parquet_path)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"Parquet save failed ({exc}); wrote {csv_path} instead")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute employer headcounts using existing transitions rcids.")
    parser.add_argument("--transitions-path", default=TRANSITION_OUTFILE, help="Path to transitions parquet with rcids.")
    parser.add_argument("--min-position-days", type=int, default=MIN_POSITION_DAYS, help="Minimum position length to treat as employment.")
    parser.add_argument("--headcount", action="store_true", help="Compute rcid-year headcounts.")
    parser.add_argument("--origin-headcount", action="store_true", help="Compute rcid-rsid-year headcounts.")
    parser.add_argument("--chunks", type=int, default=20, help="Number of chunks for rcid queries.")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Approximate rows per chunk for rcid queries.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")
    args = parser.parse_args()

    do_headcount = args.headcount or (not args.headcount and not args.origin_headcount)
    do_origin = args.origin_headcount

    rcids = _load_rcids_from_parquet(args.transitions_path, args.verbose)
    db = wrds.Connection(wrds_username="amykimecon")

    if do_headcount:
        headcount = _chunked_compute(
            rcids,
            compute_headcount,
            db=db,
            min_position_days=args.min_position_days,
            chunks=args.chunks,
            chunk_size=args.chunk_size,
            verbose=args.verbose,
        )
        save_with_fallback(headcount, HEADCOUNT_OUTFILE)

    if do_origin:
        origin_headcount = _chunked_compute(
            rcids,
            compute_origin_headcount,
            db=db,
            min_position_days=args.min_position_days,
            chunks=args.chunks,
            chunk_size=args.chunk_size,
            verbose=args.verbose,
        )
        save_with_fallback(origin_headcount, HEADCOUNT_BY_RSID_OUTFILE)

    if args.verbose:
        print("Done.")


if __name__ == "__main__":
    main()
