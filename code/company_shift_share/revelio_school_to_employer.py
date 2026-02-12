# Script: Generate school-to-employer transition shares and firm-year headcounts from Revelio
# Pulls US positions and education from WRDS, keeps first post-graduation
# long-duration job per education, aggregates flows by school (rsid),
# employer (rcid), and start year, and also computes firm headcounts
# (total and long-term). The transition output now includes a per-year
# snapshot of employees at a firm by prior rsid.

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import wrds

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *  # noqa: F401,F403
from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config

def _require_config_keys(section_name: str, section: dict, keys: list[str]) -> None:
    missing = [k for k in keys if k not in section]
    if missing:
        raise ValueError(f"Missing required config keys in '{section_name}': {', '.join(missing)}")

def _to_optional_int(value, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"", "none", "null"}:
            return None
        try:
            return int(v)
        except ValueError as exc:
            raise ValueError(f"Invalid integer for {name}: {value!r}") from exc
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def _load_preferred_rcids(path: str) -> list[int]:
    """
    Load preferred rcids from the employer crosswalk, if available.
    Returns an empty list when the file or column is missing.
    """
    if not os.path.exists(path):
        print(f"Preferred rcid filter skipped: {path} not found.")
        return []
    try:
        df = pd.read_parquet(path, columns=None)
    except Exception as exc:  # pragma: no cover - file may exist but parquet engine missing
        print(f"Preferred rcid filter skipped: failed to read {path} ({exc}).")
        return []

    col_candidates = [c for c in df.columns if c.lower() == "preferred_rcid"]
    if not col_candidates:
        print(f"Preferred rcid filter skipped: no preferred_rcid column in {path}.")
        return []

    rcids = pd.to_numeric(df[col_candidates[0]], errors="coerce").dropna().astype(int).unique().tolist()
    return rcids


def _rcid_chunks(rcids: list[int], batch_size: Optional[int]) -> list[list[int]]:
    if batch_size is None or batch_size <= 0 or len(rcids) <= batch_size:
        return [rcids]
    return [rcids[i : i + batch_size] for i in range(0, len(rcids), batch_size)]


def _batch_path(kind: str, idx: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{kind}_batch_{idx:03d}.parquet"


def _set_statement_timeout(db: wrds.Connection, timeout_ms: Optional[int]) -> None:
    """
    Best-effort session statement timeout setup across wrds client variants.
    """
    if timeout_ms is None or int(timeout_ms) <= 0:
        return
    timeout_ms_int = int(timeout_ms)
    set_sql = f"SET statement_timeout = {timeout_ms_int}"
    fallback_sql = (
        f"SELECT set_config('statement_timeout', '{timeout_ms_int}', false) AS statement_timeout"
    )
    last_exc: Optional[Exception] = None

    try:
        conn = getattr(db, "connection", None)
        if conn is not None and hasattr(conn, "cursor"):
            with conn.cursor() as cur:
                cur.execute(set_sql)
            return
    except Exception as exc:
        last_exc = exc

    try:
        conn = getattr(db, "conn", None)
        if conn is not None and hasattr(conn, "cursor"):
            with conn.cursor() as cur:
                cur.execute(set_sql)
            return
    except Exception as exc:
        last_exc = exc

    try:
        engine = getattr(db, "engine", None)
        if engine is not None:
            with engine.connect() as conn:
                conn.exec_driver_sql(set_sql)
            return
    except Exception as exc:
        last_exc = exc

    try:
        # Works even when wrappers do not expose low-level DB handles.
        db.raw_sql(fallback_sql)
        return
    except Exception as exc:
        last_exc = exc

    if last_exc is not None:
        print(f"Warning: failed to set WRDS statement_timeout={timeout_ms_int}ms ({last_exc}).")


def _wrds_connect_args(query_timeout_ms: Optional[int]) -> dict:
    if query_timeout_ms is None or int(query_timeout_ms) <= 0:
        return {}
    timeout_ms_int = int(query_timeout_ms)
    # Set timeout at connect-time so every query on the session inherits it.
    return {
        "wrds_connect_args": {
            "options": f"-c statement_timeout={timeout_ms_int} -c lock_timeout={timeout_ms_int}",
        }
    }


def _open_wrds_connection(wrds_username: str, query_timeout_ms: Optional[int]) -> wrds.Connection:
    return wrds.Connection(
        wrds_username=wrds_username,
        **_wrds_connect_args(query_timeout_ms),
    )


def _reconnect_wrds(
    db: wrds.Connection,
    wrds_username: str,
    query_timeout_ms: Optional[int],
) -> wrds.Connection:
    try:
        db.close()
    except Exception:
        pass
    print("  Reconnecting to WRDS...")
    return _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)


def _run_sql_with_retries(
    db: wrds.Connection,
    sql: str,
    wrds_username: str,
    params: Optional[dict[str, object]] = None,
    query_timeout_ms: Optional[int] = None,
    max_retries: int = 1,
    retry_sleep_seconds: int = 5,
    label: str = "query",
    before_attempt: Optional[Callable[[wrds.Connection], None]] = None,
) -> tuple[pd.DataFrame, wrds.Connection]:
    """
    Execute SQL with timeout + reconnect/retry to avoid indefinite stalls.
    """
    attempts = max(1, int(max_retries) + 1)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            if before_attempt is not None:
                before_attempt(db)
            _set_statement_timeout(db, query_timeout_ms)
            df = db.raw_sql(sql, params=params or None)
            return df, db
        except Exception as exc:
            last_exc = exc
            print(f"  {label} failed on attempt {attempt}/{attempts}: {exc}")
            if attempt < attempts:
                time.sleep(max(0, int(retry_sleep_seconds)))
                db = _reconnect_wrds(
                    db,
                    wrds_username=wrds_username,
                    query_timeout_ms=query_timeout_ms,
                )
    assert last_exc is not None  # pragma: no cover - defensive, attempts >= 1
    raise last_exc


def _sql_in_list(values: list, numeric: bool) -> str:
    if numeric:
        return ", ".join(str(int(v)) for v in values)
    escaped = [str(v).replace("'", "''") for v in values]
    return ", ".join(f"'{v}'" for v in escaped)


def _prepare_university_filter_table(
    db: wrds.Connection,
    university_filter_values: list[str],
    table_name: str = "tmp_university_filter_values",
    chunk_size: int = 5000,
) -> None:
    """
    Upload university filter values to a session temp table to avoid passing
    a huge array parameter on every batch query.
    """
    cleaned_values = (
        pd.Series(university_filter_values, dtype="string")
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .loc[lambda s: s.ne("")]
        .drop_duplicates()
        .tolist()
    )
    if not cleaned_values:
        return

    conn = getattr(db, "connection", None)
    if conn is None or not hasattr(conn, "exec_driver_sql"):
        raise RuntimeError("WRDS connection does not expose exec_driver_sql for temp table setup.")

    conn.exec_driver_sql(f"DROP TABLE IF EXISTS pg_temp.{table_name}")
    conn.exec_driver_sql(
        f"CREATE TEMP TABLE pg_temp.{table_name} (university_raw TEXT PRIMARY KEY) ON COMMIT PRESERVE ROWS"
    )

    insert_sql = f"""
    INSERT INTO pg_temp.{table_name} (university_raw)
    SELECT DISTINCT u
    FROM unnest(%(vals)s::text[]) AS t(u)
    ON CONFLICT (university_raw) DO NOTHING
    """
    for i in range(0, len(cleaned_values), max(1, int(chunk_size))):
        chunk = cleaned_values[i : i + max(1, int(chunk_size))]
        conn.exec_driver_sql(insert_sql, {"vals": chunk})


def _detect_positions_role_column(db: wrds.Connection) -> Optional[str]:
    try:
        cols = db.raw_sql(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'revelio'
              AND table_name = 'individual_positions'
            """
        )
        col_list = cols["column_name"].tolist()
    except Exception as exc:  # pragma: no cover - best-effort for schema discovery
        print(f"Role column detection failed; skipping occupation subset ({exc}).")
        return None

    if "role_k17000_v3" in col_list:
        return "role_k17000_v3"
    if "role_k17000_v3_id" in col_list:
        return "role_k17000_v3_id"
    return None


def _count_positions(
    db: wrds.Connection,
    rcid_list_full: list[int],
    rcid_batch_size: Optional[int],
    role_filter_sql: str,
) -> int:
    total = 0
    rcid_batches = _rcid_chunks(rcid_list_full, rcid_batch_size) if rcid_list_full else [None]
    for batch in rcid_batches:
        batch_filter = ""
        if batch is not None:
            rcid_str = ", ".join(str(int(r)) for r in batch)
            batch_filter = f" AND rcid IN ({rcid_str})"
        sql = f"""
        SELECT COUNT(*) AS n_positions
        FROM revelio.individual_positions
        WHERE country = 'United States'
          AND rcid IS NOT NULL
          AND startdate IS NOT NULL
          {role_filter_sql}
          {batch_filter}
        """
        df = db.raw_sql(sql)
        total += int(df["n_positions"].iloc[0]) if not df.empty else 0
    return total


def _load_university_filter_values(path: str) -> list[str]:
    if not path:
        return []
    if not os.path.exists(path):
        print(f"University filter skipped: {path} not found.")
        return []
    try:
        if path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - best-effort convenience
        print(f"University filter skipped: failed to read {path} ({exc}).")
        return []
    if "university_raw" not in df.columns:
        print(f"University filter skipped: no university_raw column in {path}.")
        return []
    vals = (
        df["university_raw"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .str.lower()
        .unique()
        .tolist()
    )
    return vals


def _load_foia_job_title_counts(foia_path: str, min_year: int) -> pd.DataFrame:
    foia = pd.read_parquet(foia_path)
    foia = foia[foia["year"].astype(int) >= min_year]
    foia = foia[foia["job_title"].notna()].copy()
    foia["job_title"] = foia["job_title"].astype(str).str.strip()
    occs = (
        foia.groupby("job_title")["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "n_persons"})
    )
    return occs


def _load_role_lookup(db: wrds.Connection) -> pd.DataFrame:
    return db.raw_sql("SELECT * FROM revelio.individual_role_lookup_v3")


def _build_role_k17000_subset(
    db: wrds.Connection,
    foia_path: str,
    min_year: int,
    max_cutoff: int,
    mean_cutoff: int,
) -> dict[str, object]:
    occs = _load_foia_job_title_counts(foia_path, min_year)
    roles = _load_role_lookup(db)

    role_cols = [c for c in roles.columns if c.startswith("role_k") and c.endswith("_v3")]
    role_cols = [c for c in role_cols if c != "role_k10_v3"]
    if "role_k50_v3" not in role_cols:
        raise ValueError("role_k50_v3 not found in individual_role_lookup_v3 columns.")

    id_vars = ["role_k50_v3"]
    value_vars = [c for c in role_cols if c != "role_k50_v3"]
    role_long = roles[id_vars + value_vars].melt(
        id_vars="role_k50_v3",
        value_vars=value_vars,
        var_name="role_level",
        value_name="role_name",
    )
    role_long = role_long.dropna(subset=["role_k50_v3", "role_name"]).copy()
    role_long["role_name"] = role_long["role_name"].astype(str).str.strip()

    merged = role_long.merge(occs, left_on="role_name", right_on="job_title", how="inner")
    role_k50_stats = (
        merged.groupby("role_k50_v3")["n_persons"]
        .agg(["max", "mean", "sum"])
        .reset_index()
        .rename(
            columns={
                "max": "max_person_id_count",
                "mean": "mean_person_id_count",
                "sum": "sum_person_id_count",
            }
        )
    )
    keep_role_k50 = role_k50_stats[
        (role_k50_stats["max_person_id_count"] >= max_cutoff)
        | (role_k50_stats["mean_person_id_count"] >= mean_cutoff)
    ]["role_k50_v3"]

    roles_kept = roles[roles["role_k50_v3"].isin(keep_role_k50)]
    role_k17000_names = (
        roles_kept["role_k17000_v3"].dropna().astype(str).unique().tolist()
        if "role_k17000_v3" in roles_kept.columns
        else []
    )
    role_k17000_total_names = (
        roles["role_k17000_v3"].dropna().astype(str).unique().tolist()
        if "role_k17000_v3" in roles.columns
        else []
    )
    role_k17000_ids = (
        pd.to_numeric(roles_kept["role_k17000_v3_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
        if "role_k17000_v3_id" in roles_kept.columns
        else []
    )
    role_k17000_total_ids = (
        pd.to_numeric(roles["role_k17000_v3_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
        if "role_k17000_v3_id" in roles.columns
        else []
    )

    return {
        "role_k17000_names": role_k17000_names,
        "role_k17000_total_names": role_k17000_total_names,
        "role_k17000_ids": role_k17000_ids,
        "role_k17000_total_ids": role_k17000_total_ids,
        "role_k50_stats": role_k50_stats,
    }


def run_queries(
    db: wrds.Connection,
    wrds_username: str,
    employer_cw_path: str,
    min_position_days: int,
    tenure_min_days: int,
    test: Optional[int] = None,
    use_preferred_rcids: bool = True,
    rcid_limit: Optional[int] = None,
    limit_education_to_positions: bool = False,
    batch_out_dir: Optional[Path] = None,
    resume_batches: bool = False,
    rcid_batch_size: Optional[int] = None,
    role_filter_values: Optional[list] = None,
    role_filter_column: Optional[str] = None,
    role_filter_numeric: bool = False,
    university_filter_values: Optional[list[str]] = None,
    query_timeout_ms: Optional[int] = None,
    query_max_retries: int = 1,
    split_failed_batches: bool = True,
    skip_failed_batches: bool = True,
) -> pd.DataFrame:
    """
    Run the WRDS queries that produce:
      1) School -> employer transitions with new-hire counts and shares.
      2) Employer-level new-hire counts by year (for all rcids, even
         those without transitions from the education sample).

    If use_preferred_rcids is True (default), restrict positions to rcids that
    appear as preferred_rcid in the F-1 employer crosswalk.
    """
    rcid_filter_sql = ""
    rcid_list_full: list[int] = []
    if use_preferred_rcids:
        preferred_rcids = _load_preferred_rcids(employer_cw_path)
        if preferred_rcids:
            if rcid_limit is not None and rcid_limit < len(preferred_rcids):
                preferred_rcids = preferred_rcids[:rcid_limit]
                print(f"Test mode: trimming rcid list to {len(preferred_rcids)} entries.")
            rcid_list_full = preferred_rcids
            if rcid_batch_size is None or rcid_batch_size <= 0:
                # No batching: apply the full filter once here.
                rcid_list = ", ".join(str(int(r)) for r in preferred_rcids)
                rcid_filter_sql = f" AND rcid IN ({rcid_list})"
                print(f"Filtering to {len(preferred_rcids)} preferred rcids from crosswalk (no batching).")
            else:
                print(f"Prepared {len(preferred_rcids)} preferred rcids for batching.")
        else:
            print("No preferred rcids loaded; skipping rcid filter.")

    if test is not None:
        testlimit = f"LIMIT {test}"
    else:
        testlimit = ""

    role_filter_sql = ""
    if role_filter_values and role_filter_column:
        role_filter_sql = (
            f" AND {role_filter_column} IN ({_sql_in_list(role_filter_values, role_filter_numeric)})"
        )
    university_filter_sql = ""
    before_attempt: Optional[Callable[[wrds.Connection], None]] = None
    query_params: dict[str, object] = {}
    if university_filter_values:
        query_params["university_filter_values"] = list(university_filter_values)
        # WRDS sessions may run in read-only transactions; avoid temp-table DDL.
        university_filter_sql = (
            "  AND EXISTS (\n"
            "      SELECT 1\n"
            "      FROM unnest(%(university_filter_values)s::text[]) AS uf(university_raw)\n"
            "      WHERE uf.university_raw = LOWER(TRIM(r.university_raw))\n"
            "  )\n"
        )

    edu_position_filter = ""
    if limit_education_to_positions:
        edu_position_filter = "  AND EXISTS (SELECT 1 FROM us_positions up WHERE up.user_id = e.user_id)\n"

    print("Assembling transition query SQL...")
    base_cte = f"""
    WITH us_positions AS MATERIALIZED (
        SELECT
            user_id,
            rcid,
            startdate::DATE AS startdate,
            COALESCE(enddate::DATE, '2025-12-31') AS enddate
        FROM revelio.individual_positions
        WHERE country = 'United States'
          AND rcid IS NOT NULL
          AND startdate IS NOT NULL
            {rcid_filter_sql}
            {role_filter_sql}
            {{rcid_filter_override}}
        {testlimit}
    ),
    education_clean AS (
        SELECT
            user_id,
            education_number,
            rsid,
            enddate::DATE AS grad_date
        FROM revelio.individual_user_education
        WHERE enddate IS NOT NULL
    ),
    education_raw AS (
        SELECT
            user_id,
            education_number,
            university_raw
        FROM revelio.individual_user_education_raw
        WHERE university_raw IS NOT NULL
          AND TRIM(university_raw) <> ''
    ),
    education AS MATERIALIZED (
        SELECT
            e.user_id,
            e.education_number,
            e.rsid,
            r.university_raw,
            e.grad_date
        FROM education_clean AS e
        LEFT JOIN education_raw AS r
          ON e.user_id = r.user_id
         AND e.education_number = r.education_number
        WHERE e.grad_date IS NOT NULL
          AND r.university_raw IS NOT NULL
          AND TRIM(r.university_raw) <> ''
    {university_filter_sql}
    {edu_position_filter}
    ),
    long_positions AS MATERIALIZED (
        SELECT user_id, rcid, startdate
        FROM us_positions
        WHERE enddate >= startdate + INTERVAL '{int(min_position_days)} days'
    ),
    positions_after_grad AS (
        SELECT
            e.user_id,
            e.university_raw,
            e.education_number,
            p.rcid,
            p.startdate,
            -- take earliest post-grad position per user x university_raw
            ROW_NUMBER() OVER (
                PARTITION BY e.user_id, e.university_raw
                ORDER BY p.startdate
            ) AS rank_after_grad
        FROM education e
        JOIN long_positions p
          ON e.user_id = p.user_id
         AND p.startdate >= e.grad_date
         AND p.startdate <= e.grad_date + INTERVAL '1 years'
    ),
    first_jobs AS MATERIALIZED (
        SELECT *
        FROM positions_after_grad
        WHERE rank_after_grad = 1
    ),
    user_company_bounds AS (
        -- First start at the firm per user (used for tenure check)
        SELECT
            user_id,
            rcid,
            MIN(startdate) AS first_startdate
        FROM us_positions
        GROUP BY user_id, rcid
    ),
    tenure_positions AS (
        -- All positions (including short spells) keyed with the first start date at the firm
        SELECT
            up.user_id,
            up.rcid,
            up.startdate,
            up.enddate,
            b.first_startdate
        FROM us_positions AS up
        JOIN user_company_bounds AS b
          USING (user_id, rcid)
    ),
    long_term_employees AS MATERIALIZED (
        -- Employees whose education (rsid) ended on/before a given position start; tenure tested at the year level
        SELECT
            tp.user_id,
            tp.rcid,
            tp.first_startdate,
            tp.startdate,
            tp.enddate,
            edu.university_raw
        FROM tenure_positions AS tp
        JOIN education AS edu
          ON tp.user_id = edu.user_id
         AND edu.university_raw IS NOT NULL
         AND edu.grad_date <= tp.startdate
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
            university_raw,
            rcid,
            EXTRACT(YEAR FROM startdate)::INT AS year,
            COUNT(DISTINCT user_id) AS n_transitions
        FROM first_jobs
        GROUP BY university_raw, rcid, year
    ),
    employee_counts AS (
        SELECT
            university_raw,
            rcid,
            gs.year,
            COUNT(DISTINCT user_id) AS n_emp
        FROM long_term_employees,
        LATERAL generate_series(EXTRACT(YEAR FROM startdate)::INT, EXTRACT(YEAR FROM enddate)::INT) AS gs(year)
        WHERE make_date(gs.year, 12, 31) >= first_startdate + INTERVAL '{int(tenure_min_days)} days'
        GROUP BY university_raw, rcid, gs.year
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
        t.university_raw,
        t.rcid,
        t.year,
        t.n_transitions,
        e.n_emp,
        n.total_new_hires,
        CASE
            WHEN n.total_new_hires IS NULL OR n.total_new_hires = 0 THEN NULL
            ELSE t.n_transitions::DECIMAL / n.total_new_hires
        END AS share_of_new_hires
    FROM transition_counts t
    LEFT JOIN employee_counts e
      ON t.rcid = e.rcid AND t.university_raw = e.university_raw AND t.year = e.year
    LEFT JOIN new_hire_counts n
      ON t.rcid = n.rcid AND t.year = n.year
    ORDER BY t.university_raw, t.rcid, t.year
    """

    print("Executing transition query...")
    results = []
    rcid_batches = _rcid_chunks(rcid_list_full, rcid_batch_size) if rcid_list_full else [None]
    for idx, batch in enumerate(rcid_batches, start=1):
        batch_filter = ""
        if batch is not None:
            rcid_str = ", ".join(str(int(r)) for r in batch)
            batch_filter = f" AND rcid IN ({rcid_str})"
            print(f"  Running transition batch {idx}/{len(rcid_batches)} with {len(batch)} rcids...")
        sql = transition_query.replace("{rcid_filter_override}", batch_filter)
        batch_path = _batch_path("transitions", idx, batch_out_dir) if batch_out_dir else None
        if resume_batches and batch_path and batch_path.exists():
            print(f"  Loading existing transition batch {idx} from {batch_path}")
            df = pd.read_parquet(batch_path)
        else:
            try:
                df, db = _run_sql_with_retries(
                    db=db,
                    sql=sql,
                    wrds_username=wrds_username,
                    params=query_params or None,
                    query_timeout_ms=query_timeout_ms,
                    max_retries=query_max_retries,
                    label=f"transition batch {idx}/{len(rcid_batches)}",
                    before_attempt=before_attempt,
                )
            except Exception as exc:
                if split_failed_batches and batch is not None and len(batch) > 1:
                    print(f"  Splitting failed transition batch {idx} into singleton rcid queries ({exc})")
                    split_frames = []
                    for rcid in batch:
                        singleton_filter = f" AND rcid IN ({int(rcid)})"
                        singleton_sql = transition_query.replace("{rcid_filter_override}", singleton_filter)
                        try:
                            singleton_df, db = _run_sql_with_retries(
                                db=db,
                                sql=singleton_sql,
                                wrds_username=wrds_username,
                                params=query_params or None,
                                query_timeout_ms=query_timeout_ms,
                                max_retries=query_max_retries,
                                label=f"transition singleton rcid={int(rcid)} (parent batch {idx})",
                                before_attempt=before_attempt,
                            )
                            split_frames.append(singleton_df)
                        except Exception as inner_exc:
                            msg = (
                                f"  Failed transition singleton rcid={int(rcid)} "
                                f"in batch {idx}: {inner_exc}"
                            )
                            if skip_failed_batches:
                                print(f"{msg}; skipping.")
                                continue
                            raise RuntimeError(msg) from inner_exc
                    df = pd.concat(split_frames, ignore_index=True) if split_frames else pd.DataFrame()
                elif skip_failed_batches:
                    print(f"  Transition batch {idx} failed and will be skipped ({exc}).")
                    df = pd.DataFrame()
                else:
                    raise

            print(f"  Batch {idx} returned {len(df):,} rows.")
            if batch_path:
                df.to_parquet(batch_path, index=False)
                print(f"  Saved batch {idx} to {batch_path}")
        results.append(df)
    transitions = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    print(f"Transitions query returned {len(transitions):,} total rows.")
    return transitions


def run_headcounts(
    db: wrds.Connection,
    wrds_username: str,
    employer_cw_path: str,
    min_position_days: int,
    test: Optional[int] = None,
    use_preferred_rcids: bool = True,
    rcid_limit: Optional[int] = None,
    batch_out_dir: Optional[Path] = None,
    resume_batches: bool = False,
    rcid_batch_size: Optional[int] = None,
    role_filter_values: Optional[list] = None,
    role_filter_column: Optional[str] = None,
    role_filter_numeric: bool = False,
    query_timeout_ms: Optional[int] = None,
    query_max_retries: int = 1,
    split_failed_batches: bool = True,
    skip_failed_batches: bool = True,
) -> pd.DataFrame:
    """
    Compute rcid-year headcounts:
      - total_headcount: all US employees
      - long_term_headcount: employees on positions lasting >= MIN_POSITION_DAYS
    """
    rcid_filter_sql = ""
    rcid_list_full: list[int] = []
    if use_preferred_rcids:
        preferred_rcids = _load_preferred_rcids(employer_cw_path)
        if preferred_rcids:
            if rcid_limit is not None and rcid_limit < len(preferred_rcids):
                preferred_rcids = preferred_rcids[:rcid_limit]
                print(f"Test mode: trimming rcid list to {len(preferred_rcids)} entries for headcounts.")
            rcid_list_full = preferred_rcids
            if rcid_batch_size is None or rcid_batch_size <= 0:
                rcid_list = ", ".join(str(int(r)) for r in preferred_rcids)
                rcid_filter_sql = f" AND rcid IN ({rcid_list})"
                print(f"Filtering headcounts to {len(preferred_rcids)} preferred rcids from crosswalk (no batching).")
            else:
                print(f"Prepared {len(preferred_rcids)} preferred rcids for batching (headcounts).")
        else:
            print("No preferred rcids loaded for headcounts; skipping rcid filter.")

    if test is not None:
        testlimit = f"LIMIT {test}"
    else:
        testlimit = ""

    role_filter_sql = ""
    if role_filter_values and role_filter_column:
        role_filter_sql = (
            f" AND {role_filter_column} IN ({_sql_in_list(role_filter_values, role_filter_numeric)})"
        )

    print("Assembling headcount query SQL...")
    query = f"""
    WITH us_positions AS MATERIALIZED (
        SELECT
            user_id,
            rcid,
            startdate::DATE AS startdate,
            COALESCE(enddate::DATE, '2025-12-31') AS enddate
        FROM revelio.individual_positions
        WHERE country = 'United States'
          AND rcid IS NOT NULL
          AND startdate IS NOT NULL
            {rcid_filter_sql}
            {role_filter_sql}
            {{rcid_filter_override}}
        {testlimit}
    ),
    long_positions AS MATERIALIZED (
        SELECT *
        FROM us_positions
        WHERE enddate >= startdate + INTERVAL '{int(min_position_days)} days'
    ),
    total_headcount AS (
        SELECT
            rcid,
            gs.year,
            COUNT(DISTINCT user_id) AS total_headcount
        FROM us_positions,
        LATERAL generate_series(EXTRACT(YEAR FROM startdate)::INT, EXTRACT(YEAR FROM enddate)::INT) AS gs(year)
        GROUP BY rcid, gs.year
    ),
    long_term_headcount AS (
        SELECT
            rcid,
            gs.year,
            COUNT(DISTINCT user_id) AS long_term_headcount
        FROM long_positions,
        LATERAL generate_series(EXTRACT(YEAR FROM startdate)::INT, EXTRACT(YEAR FROM enddate)::INT) AS gs(year)
        GROUP BY rcid, gs.year
    )
    SELECT
        COALESCE(t.rcid, l.rcid) AS rcid,
        COALESCE(t.year, l.year) AS year,
        t.total_headcount,
        l.long_term_headcount
    FROM total_headcount t
    FULL OUTER JOIN long_term_headcount l
      ON t.rcid = l.rcid AND t.year = l.year
    ORDER BY rcid, year
    """
    print("Executing headcount query...")
    results = []
    rcid_batches = _rcid_chunks(rcid_list_full, rcid_batch_size) if rcid_list_full else [None]
    for idx, batch in enumerate(rcid_batches, start=1):
        batch_filter = ""
        if batch is not None:
            rcid_str = ", ".join(str(int(r)) for r in batch)
            batch_filter = f" AND rcid IN ({rcid_str})"
            print(f"  Running headcount batch {idx}/{len(rcid_batches)} with {len(batch)} rcids...")
        sql = query.replace("{rcid_filter_override}", batch_filter)
        batch_path = _batch_path("headcounts", idx, batch_out_dir) if batch_out_dir else None
        if resume_batches and batch_path and batch_path.exists():
            print(f"  Loading existing headcount batch {idx} from {batch_path}")
            df = pd.read_parquet(batch_path)
        else:
            try:
                df, db = _run_sql_with_retries(
                    db=db,
                    sql=sql,
                    wrds_username=wrds_username,
                    query_timeout_ms=query_timeout_ms,
                    max_retries=query_max_retries,
                    label=f"headcount batch {idx}/{len(rcid_batches)}",
                )
            except Exception as exc:
                if split_failed_batches and batch is not None and len(batch) > 1:
                    print(f"  Splitting failed headcount batch {idx} into singleton rcid queries ({exc})")
                    split_frames = []
                    for rcid in batch:
                        singleton_filter = f" AND rcid IN ({int(rcid)})"
                        singleton_sql = query.replace("{rcid_filter_override}", singleton_filter)
                        try:
                            singleton_df, db = _run_sql_with_retries(
                                db=db,
                                sql=singleton_sql,
                                wrds_username=wrds_username,
                                query_timeout_ms=query_timeout_ms,
                                max_retries=query_max_retries,
                                label=f"headcount singleton rcid={int(rcid)} (parent batch {idx})",
                            )
                            split_frames.append(singleton_df)
                        except Exception as inner_exc:
                            msg = (
                                f"  Failed headcount singleton rcid={int(rcid)} "
                                f"in batch {idx}: {inner_exc}"
                            )
                            if skip_failed_batches:
                                print(f"{msg}; skipping.")
                                continue
                            raise RuntimeError(msg) from inner_exc
                    df = pd.concat(split_frames, ignore_index=True) if split_frames else pd.DataFrame()
                elif skip_failed_batches:
                    print(f"  Headcount batch {idx} failed and will be skipped ({exc}).")
                    df = pd.DataFrame()
                else:
                    raise

            print(f"  Batch {idx} returned {len(df):,} rows.")
            if batch_path:
                df.to_parquet(batch_path, index=False)
                print(f"  Saved batch {idx} to {batch_path}")
        results.append(df)
    headcounts = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    print(f"Headcount query returned {len(headcounts):,} total rows.")
    return headcounts

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
    parser = argparse.ArgumentParser(description="Build school-to-employer transitions and headcounts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--role-filter",
        dest="role_filter",
        action="store_true",
        default=None,
        help="Restrict to occupation subset derived from FOIA + role lookup (default: on).",
    )
    parser.add_argument(
        "--no-role-filter",
        dest="role_filter",
        action="store_false",
        help="Disable occupation subset filtering (writes baseline outputs).",
    )
    parser.add_argument(
        "--university-filter-path",
        type=str,
        default=None,
        help="Optional path to parquet/csv containing university_raw; transitions restricted to these names.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_cfg_section(cfg, "paths")
    revelio_cfg = get_cfg_section(cfg, "revelio_school_to_employer")
    _require_config_keys(
        "paths",
        paths,
        [
            "transitions_out",
            "headcounts_out",
            "transitions_occ_subset_out",
            "headcounts_occ_subset_out",
            "foia_with_person_id",
            "preferred_rcids",
        ],
    )
    _require_config_keys(
        "revelio_school_to_employer",
        revelio_cfg,
        [
            "wrds_username",
            "min_position_days",
            "tenure_min_days",
            "test_rcid_limit",
            "test_limit",
            "batch_size",
            "batch_out_dir",
            "role_filter_default",
            "occ_subset_min_year",
            "role_k50_max_cutoff",
            "role_k50_mean_cutoff",
            "limit_education_to_positions",
            "resume_batches",
            "use_preferred_rcids",
        ],
    )

    transitions_outfile = paths["transitions_out"]
    headcount_outfile = paths["headcounts_out"]
    occ_subset_transition_outfile = paths["transitions_occ_subset_out"]
    occ_subset_headcount_outfile = paths["headcounts_occ_subset_out"]
    foia_path = paths["foia_with_person_id"]
    employer_cw_path = paths["preferred_rcids"]

    occ_subset_min_year = int(revelio_cfg["occ_subset_min_year"])
    role_k50_max_cutoff = int(revelio_cfg["role_k50_max_cutoff"])
    role_k50_mean_cutoff = int(revelio_cfg["role_k50_mean_cutoff"])
    role_filter_default = bool(revelio_cfg["role_filter_default"])
    min_position_days = int(revelio_cfg["min_position_days"])
    tenure_min_days = int(revelio_cfg["tenure_min_days"])
    test_rcid_limit = _to_optional_int(revelio_cfg["test_rcid_limit"], "revelio_school_to_employer.test_rcid_limit")
    test_limit = _to_optional_int(revelio_cfg["test_limit"], "revelio_school_to_employer.test_limit")
    batch_size = _to_optional_int(revelio_cfg["batch_size"], "revelio_school_to_employer.batch_size")
    batch_out_dir = Path(revelio_cfg["batch_out_dir"])
    timeout_minutes = _to_optional_int(
        revelio_cfg.get("query_timeout_minutes", 45),
        "revelio_school_to_employer.query_timeout_minutes",
    )
    query_timeout_ms = None if timeout_minutes is None else int(timeout_minutes) * 60 * 1000
    query_max_retries = int(revelio_cfg.get("query_max_retries", 1))
    split_failed_batches = bool(revelio_cfg.get("split_failed_batches", True))
    skip_failed_batches = bool(revelio_cfg.get("skip_failed_batches", True))

    role_filter_enabled = role_filter_default if args.role_filter is None else args.role_filter
    default_matched_university_path = paths.get("revelio_matched_university_raws")
    university_filter_path = (
        args.university_filter_path
        or revelio_cfg.get("university_filter_path")
        or default_matched_university_path
    )
    university_filter_values = _load_university_filter_values(university_filter_path) if university_filter_path else []
    if university_filter_values:
        print(f"University filter active: {len(university_filter_values):,} university_raw values.")

    print("Connecting to WRDS...")
    wrds_username = str(revelio_cfg["wrds_username"])
    db = _open_wrds_connection(wrds_username=wrds_username, query_timeout_ms=query_timeout_ms)
    if query_timeout_ms is not None:
        print(
            f"Query timeout enabled: {int(query_timeout_ms / 60000)} minutes per SQL call; "
            f"retries={query_max_retries}."
        )
    role_filter_values = None
    role_filter_column = None
    role_filter_numeric = False
    preferred_rcids_for_counts: list[int] = []
    if role_filter_enabled:
        try:
            role_subset = _build_role_k17000_subset(
                db,
                foia_path=foia_path,
                min_year=occ_subset_min_year,
                max_cutoff=role_k50_max_cutoff,
                mean_cutoff=role_k50_mean_cutoff,
            )
            role_filter_column = _detect_positions_role_column(db)
            if role_filter_column == "role_k17000_v3":
                role_filter_values = role_subset["role_k17000_names"]
                role_filter_numeric = False
                total_roles = len(role_subset["role_k17000_total_names"])
                kept_roles = len(role_filter_values)
            elif role_filter_column == "role_k17000_v3_id":
                role_filter_values = role_subset["role_k17000_ids"]
                role_filter_numeric = True
                total_roles = len(role_subset["role_k17000_total_ids"])
                kept_roles = len(role_filter_values)
            else:
                print("Role column not found in individual_positions; disabling role filter.")
                role_filter_column = None
                role_filter_values = None
                total_roles = 0
                kept_roles = 0
            if total_roles > 0:
                excluded = total_roles - kept_roles
                excluded_pct = excluded / total_roles * 100
                print(
                    f"Role filter excludes {excluded:,} of {total_roles:,} "
                    f"role_k17000 occupations ({excluded_pct:.1f}%)."
                )
                print(
                    f"Role filter active on {role_filter_column}: keeping {kept_roles:,} "
                    f"of {total_roles:,} role_k17000 occupations."
                )
        except Exception as exc:  # pragma: no cover - best-effort, should not block baseline outputs
            print(f"Occupation subset setup failed; disabling role filter ({exc}).")
            role_filter_column = None
            role_filter_values = None

    if role_filter_enabled and role_filter_values and role_filter_column:
        batch_out_dir = Path(Path(transitions_outfile).parent) / "revelio_occ_subset_batches"

        preferred_rcids_for_counts = _load_preferred_rcids(employer_cw_path)
        if test_rcid_limit is not None and preferred_rcids_for_counts:
            preferred_rcids_for_counts = preferred_rcids_for_counts[:int(test_rcid_limit)]
        role_filter_sql = (
            f" AND {role_filter_column} IN ({_sql_in_list(role_filter_values, role_filter_numeric)})"
        )
        try:
            print("Counting positions with and without role filter (this may take a bit)...")
            total_positions = _count_positions(
                db,
                preferred_rcids_for_counts,
                batch_size,
                role_filter_sql="",
            )
            filtered_positions = _count_positions(
                db,
                preferred_rcids_for_counts,
                batch_size,
                role_filter_sql=role_filter_sql,
            )
            if total_positions > 0:
                pct = filtered_positions / total_positions * 100
                print(
                    f"Positions kept after role filter: {filtered_positions:,} "
                    f"of {total_positions:,} ({pct:.1f}%)."
                )
        except Exception as exc:  # pragma: no cover - best-effort diagnostic
            print(f"Position count check failed; continuing without counts ({exc}).")
            
    print("Running transition and new-hire queries (this can take a while)...")
    transitions = run_queries(
        db,
        wrds_username=wrds_username,
        employer_cw_path=employer_cw_path,
        min_position_days=min_position_days,
        tenure_min_days=tenure_min_days,
        use_preferred_rcids=bool(revelio_cfg["use_preferred_rcids"]),
        test=test_limit,
        rcid_limit=test_rcid_limit,
        limit_education_to_positions=bool(revelio_cfg["limit_education_to_positions"]),
        batch_out_dir=batch_out_dir,
        resume_batches=bool(revelio_cfg["resume_batches"]),
        rcid_batch_size=batch_size,
        role_filter_values=role_filter_values,
        role_filter_column=role_filter_column,
        role_filter_numeric=role_filter_numeric,
        university_filter_values=university_filter_values,
        query_timeout_ms=query_timeout_ms,
        query_max_retries=query_max_retries,
        split_failed_batches=split_failed_batches,
        skip_failed_batches=skip_failed_batches,
    )
    print("Running headcount queries (this can take a while)...")
    headcounts = run_headcounts(
        db,
        wrds_username=wrds_username,
        employer_cw_path=employer_cw_path,
        min_position_days=min_position_days,
        use_preferred_rcids=bool(revelio_cfg["use_preferred_rcids"]),
        test=test_limit,
        rcid_limit=test_rcid_limit,
        batch_out_dir=batch_out_dir,
        resume_batches=bool(revelio_cfg["resume_batches"]),
        rcid_batch_size=batch_size,
        role_filter_values=role_filter_values,
        role_filter_column=role_filter_column,
        role_filter_numeric=role_filter_numeric,
        query_timeout_ms=query_timeout_ms,
        query_max_retries=query_max_retries,
        split_failed_batches=split_failed_batches,
        skip_failed_batches=skip_failed_batches,
    )

    if role_filter_enabled and role_filter_values and role_filter_column:
        save_with_fallback(transitions, occ_subset_transition_outfile)
        save_with_fallback(headcounts, occ_subset_headcount_outfile)
    else:
        save_with_fallback(transitions, transitions_outfile)
        save_with_fallback(headcounts, headcount_outfile)
    print("Done.")


if __name__ == "__main__":
    main()
