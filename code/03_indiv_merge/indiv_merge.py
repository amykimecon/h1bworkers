# File Description: Merging H-1B and Revelio Individual Data
# Author: Amy Kim
# Date Created: Jun 30 2025

# Imports and Paths
import argparse
import datetime
import duckdb as ddb
import pandas as pd
import numpy as np
import os
import re
import sys
import time

# Used as default enddatenull in get_long_by_year — update dynamically so positions with
# null enddates are imputed as "still active through the current calendar year."
CURRENT_YEAR = str(datetime.datetime.now().year)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

if "__file__" in globals():
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    _THIS_DIR = os.path.join(os.getcwd(), "03_indiv_merge")

sys.path.append(_THIS_DIR)
sys.path.append(os.path.dirname(_THIS_DIR))
import helpers as help
import indiv_merge_config as icfg

con_indiv = ddb.connect()


def _configure_duckdb_runtime(connection):
    threads = max(1, os.cpu_count() or 1)
    connection.sql(f"PRAGMA threads={threads}")
    connection.sql("PRAGMA preserve_insertion_order=false")
    tmp_dir = os.path.join(os.path.expanduser("~"), ".tmp", "duckdb")
    os.makedirs(tmp_dir, exist_ok=True)
    escaped_tmp = tmp_dir.replace("'", "''")
    connection.sql(f"PRAGMA temp_directory='{escaped_tmp}'")


_configure_duckdb_runtime(con_indiv)

#####################
# IMPORTING DATA
#####################
## foia
foia_indiv = con_indiv.read_parquet(
    icfg.choose_path(icfg.FOIA_INDIV_PARQUET, icfg.FOIA_INDIV_PARQUET_LEGACY)
)

## revelio
rev_indiv = con_indiv.read_parquet(
    icfg.choose_path(icfg.REV_INDIV_PARQUET, icfg.REV_INDIV_PARQUET_LEGACY)
)
REV_INDIV_COLS = set(rev_indiv.columns)
HAS_COUNTRY_UNCERTAIN_COL = "country_uncertain_ind" in REV_INDIV_COLS
COUNTRY_UNCERTAIN_EXPR = "COALESCE(country_uncertain_ind, 0)" if HAS_COUNTRY_UNCERTAIN_COL else "0"
COUNTRY_UNCERTAIN_SELECT = "country_uncertain_ind" if HAS_COUNTRY_UNCERTAIN_COL else "0 AS country_uncertain_ind"
if not HAS_COUNTRY_UNCERTAIN_COL:
    print(
        "rev_indiv missing country_uncertain_ind; defaulting to 0 for merge ranking/outputs."
    )
HAS_LLM_MATCH_SCORE_COL = "llm_match_score" in REV_INDIV_COLS
if not HAS_LLM_MATCH_SCORE_COL:
    print(
        "rev_indiv missing llm_match_score; defaulting firm-match quality factor to neutral (1.0)."
    )

## revelio education data
rev_educ = con_indiv.read_parquet(
    icfg.choose_path(icfg.REV_EDUC_LONG_PARQUET, icfg.REV_EDUC_LONG_PARQUET_LEGACY)
)

# collapsing to user x institution (for now use country with top score)
rev_educ_clean = con_indiv.sql("SELECT *, ed_startdate AS startdate, ed_enddate AS enddate FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY user_id, education_number ORDER BY matchscore DESC) AS match_order FROM rev_educ WHERE degree_clean != 'Non-Degree') WHERE match_order = 1")

# Importing User x Position-level Data (all positions, cleaned and deduplicated)
merged_pos = con_indiv.read_parquet(
    icfg.choose_path(icfg.MERGED_POS_CLEAN_PARQUET, icfg.MERGED_POS_CLEAN_PARQUET_LEGACY)
)

# removing duplicates, setting alt enddate as enddate if missing
merged_pos_clean = con_indiv.sql("SELECT * EXCLUDE (enddate), CASE WHEN alt_enddate IS NULL THEN enddate ELSE alt_enddate END AS enddate FROM merged_pos WHERE pos_dup_ind IS NULL OR pos_dup_ind = 0")

#####################
# WRAPPER FUNCTIONS FOR MERGE
#####################
LAST_TESTING_TABLES = {}


def _fmt_elapsed(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds / 60:.2f}m"


def _sql_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _safe_identifier(text):
    out = re.sub(r"[^0-9a-zA-Z_]", "_", str(text)).strip("_").lower()
    out = re.sub(r"_+", "_", out)
    if not out:
        out = "tmp"
    if out[0].isdigit():
        out = f"t_{out}"
    return out


def get_duckdb_connection():
    """Return the shared DuckDB connection used by this module."""
    return con_indiv


def list_testing_tables(prefix = None, con = con_indiv):
    """List materialized testing tables currently available in this DuckDB session."""
    tables = con.sql("SHOW TABLES").df()
    if prefix is not None:
        safe_prefix = _safe_identifier(prefix)
        tables = tables[tables["name"].str.startswith(safe_prefix + "_")]
    return tables.sort_values("name").reset_index(drop = True)


def _materialize_testing_tables(table_query_pairs, table_prefix = "imt", con = con_indiv):
    global LAST_TESTING_TABLES

    safe_prefix = _safe_identifier(table_prefix)
    materialized = {}
    print("")
    print("Materializing intermediate tables for interactive inspection...")
    for label, query in table_query_pairs:
        t0 = time.perf_counter()
        table_name = f"{safe_prefix}_{_safe_identifier(label)}"
        con.sql(f"CREATE OR REPLACE TEMP TABLE {table_name} AS {query}")
        n_rows = int(con.sql(f"SELECT COUNT(*) AS n FROM {table_name}").df().iloc[0, 0])
        elapsed = time.perf_counter() - t0
        materialized[label] = table_name
        print(f"[table] {table_name}: {n_rows:,} rows ({_fmt_elapsed(elapsed)})")

    LAST_TESTING_TABLES = materialized
    print("Access tips:")
    print("  - con = indiv_merge.get_duckdb_connection()")
    print("  - indiv_merge.list_testing_tables()")
    if "final" in materialized:
        print(f"  - con.sql(\"SELECT * FROM {materialized['final']} LIMIT 20\").df()")
    return materialized


def _merge_stage_counts(query, con = con_indiv):
    out = con.sql(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT foia_indiv_id) AS n_apps,
            COUNT(DISTINCT user_id) AS n_users
        FROM ({query})
        """
    ).df().iloc[0]
    n_rows = int(out["n_rows"])
    n_apps = int(out["n_apps"])
    n_users = int(out["n_users"])
    mult = round(n_rows / n_apps, 2) if n_apps else np.nan
    return n_rows, n_apps, n_users, mult


def _print_merge_stage(label, query, con = con_indiv):
    n_rows, n_apps, n_users, mult = _merge_stage_counts(query, con = con)
    mult_txt = "NA" if pd.isna(mult) else f"{mult}"
    print(
        f"[{label}] rows={n_rows:,} | apps={n_apps:,} | users={n_users:,} | multiplicity={mult_txt}"
    )


def _print_testing_spotcheck(df, con=con_indiv, max_cands=8, max_pos=4, max_educ=3):
    """Print a human-readable spotcheck of merge results after a testing-mode run.

    Uses the merge output DataFrame (df) plus con to pull employer names,
    position history, and education records for each candidate.
    """
    if df is None or df.empty:
        print("\n[spotcheck] No matches found.")
        return

    print("\n" + "=" * 80)
    print("SPOTCHECK: Testing Mode Match Results")
    print("=" * 80)

    # Pull employer name lookup from foia_indiv
    foia_ids_sql = ", ".join(f"'{x}'" for x in df.foia_indiv_id.unique())
    foia_meta = con.sql(f"""
        SELECT DISTINCT ON (foia_indiv_id) foia_indiv_id, employer_name, job_title, highest_ed_level, field_clean, n_apps
        FROM foia_indiv
        WHERE foia_indiv_id IN ({foia_ids_sql})
        QUALIFY ROW_NUMBER() OVER (PARTITION BY foia_indiv_id ORDER BY foia_indiv_id) = 1
    """).df().set_index("foia_indiv_id")

    # Pull position history for all candidate users
    user_ids = df.user_id.dropna().unique().tolist()
    user_ids_sql = ", ".join(str(u) for u in user_ids)
    pos_df = con.sql(f"""
        SELECT user_id, rcid, company_raw, title_raw, startdate, enddate, country
        FROM merged_pos_clean
        WHERE user_id IN ({user_ids_sql})
        ORDER BY user_id, startdate DESC
    """).df() if user_ids else pd.DataFrame()

    # Pull education history
    educ_df = con.sql(f"""
        SELECT user_id, university_raw, degree_clean, ed_startdate, ed_enddate, match_country
        FROM rev_educ_clean
        WHERE user_id IN ({user_ids_sql})
        ORDER BY user_id, education_number
    """).df() if user_ids else pd.DataFrame()

    # Pull firm rcids for position matching
    foia_firm_uids = df.foia_firm_uid.dropna().unique().tolist()
    fuids_sql = ", ".join(f"'{x}'" for x in foia_firm_uids)
    firm_rcids_df = con.sql(f"""
        SELECT foia_firm_uid, LIST(DISTINCT rcid ORDER BY rcid) AS rcids
        FROM foia_indiv
        WHERE foia_firm_uid IN ({fuids_sql}) AND rcid IS NOT NULL
        GROUP BY foia_firm_uid
    """).df()
    firm_rcids_map = dict(zip(firm_rcids_df.foia_firm_uid, firm_rcids_df.rcids))

    def _fmt(v, decimals=3):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "null"
        return f"{float(v):.{decimals}f}"

    def _trunc(v, n=40):
        s = str(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else ""
        return s[:n] + ("…" if len(s) > n else "")

    for foia_id, group in df.groupby("foia_indiv_id"):
        group = group.sort_values("weight_norm", ascending=False).head(max_cands)
        n_cands = len(group)
        first = group.iloc[0]
        meta = foia_meta.loc[foia_id] if foia_id in foia_meta.index else None

        print(f"\n{'='*80}")
        print(f"foia_indiv_id={foia_id}  lottery_year={first.lottery_year}  n_candidates={first.n_match_filt}")
        employer = meta.employer_name if meta is not None else first.foia_firm_uid
        n_apps = int(meta.n_apps) if meta is not None else "?"
        print(f"FOIA: employer={employer}  country={first.foia_country}  yob={first.yob}  n_apps(firm×year)={n_apps}")
        if meta is not None:
            print(f"      job={_trunc(meta.job_title)}  edu={meta.highest_ed_level}  field={_trunc(meta.field_clean)}")

        firm_rcids = firm_rcids_map.get(str(first.foia_firm_uid), [])

        for rank, (_, row) in enumerate(group.iterrows(), 1):
            uid = row.user_id
            print(f"\n  [{rank}] user_id={uid}  name={row.fullname}")
            print(f"      weight={_fmt(row.weight_norm, 4)}  total_score={_fmt(row.total_score)}  "
                  f"country_score={_fmt(row.country_score)}  f_score={_fmt(row.f_score)}")
            occ_s = _fmt(row.get("occ_score")) if "occ_score" in row.index else "n/a"
            print(f"      rev_country={row.rev_country}  subregion={row.subregion}  "
                  f"est_yob={row.est_yob}  occ_score={occ_s}")
            print(f"      stem_ind={row.stem_ind}  foia_occ_ind={row.foia_occ_ind}  "
                  f"min_h1b_occ_rank={row.min_h1b_occ_rank}  firm_mult={_fmt(row.firm_match_quality_mult)}")
            print(f"      fields={_trunc(str(row.fields), 80)}")

            # Education
            if not educ_df.empty:
                ue = educ_df[educ_df.user_id == uid].head(max_educ)
                if not ue.empty:
                    educ_strs = [
                        f"{e.university_raw or '?'} | {e.degree_clean} ({e.ed_startdate or '?'}→{e.ed_enddate or '?'})"
                        for _, e in ue.iterrows()
                    ]
                    print(f"      educ: " + " || ".join(_trunc(s, 60) for s in educ_strs))

            # Positions at firm rcids
            if not pos_df.empty and firm_rcids:
                at_firm = pos_df[(pos_df.user_id == uid) & (pos_df.rcid.isin(firm_rcids))].head(max_pos)
                if not at_firm.empty:
                    pos_strs = [
                        f"{str(p.startdate)[:7]}→{str(p.enddate)[:7]} {p.company_raw} | {p.title_raw}"
                        for _, p in at_firm.iterrows()
                    ]
                    print(f"      *** AT FIRM: " + " || ".join(_trunc(s, 70) for s in pos_strs))
                else:
                    # Show any positions regardless
                    up = pos_df[pos_df.user_id == uid].head(max_pos)
                    if not up.empty:
                        pos_strs = [
                            f"{str(p.startdate)[:7]}→{str(p.enddate)[:7]} {p.company_raw} | {p.title_raw}"
                            for _, p in up.iterrows()
                        ]
                        print(f"      pos: " + " || ".join(_trunc(s, 70) for s in pos_strs))

    print("\n" + "=" * 80)
    print("Spotcheck done.")


def _combine_foia_prefilt(foia_prefilt, extra_condition):
    base = (foia_prefilt or "").strip().rstrip(";")
    if not base:
        return f"WHERE {extra_condition}"
    if base.lower().startswith("where"):
        return f"{base} AND ({extra_condition})"
    return f"WHERE ({base}) AND ({extra_condition})"


def _pick_testing_subset(
    foia_tab = "foia_indiv",
    foia_prefilt = "",
    test_firm_uid = None,
    test_lottery_year = None,
    test_random_seed = None,
    con = con_indiv,
):
    if (test_firm_uid is None) != (test_lottery_year is None):
        raise ValueError("Provide both test_firm_uid and test_lottery_year, or neither.")

    if test_firm_uid is not None and test_lottery_year is not None:
        return str(test_firm_uid), str(test_lottery_year)

    order_expr = "RANDOM()"
    if test_random_seed is not None:
        order_expr = f"HASH(foia_firm_uid, lottery_year, {int(test_random_seed)})"

    pick_query = f"""
    SELECT
        foia_firm_uid::VARCHAR AS foia_firm_uid,
        lottery_year::VARCHAR AS lottery_year
    FROM {foia_tab} {foia_prefilt}
    WHERE foia_firm_uid IS NOT NULL AND lottery_year IS NOT NULL
    GROUP BY foia_firm_uid, lottery_year
    ORDER BY {order_expr}
    LIMIT 1
    """
    picked = con.sql(pick_query).df()
    if picked.empty:
        raise RuntimeError("No non-null foia_firm_uid x lottery_year group found in FOIA sample.")

    return str(picked.iloc[0]["foia_firm_uid"]), str(picked.iloc[0]["lottery_year"])


def _build_stage_weighted_query(stage_input_query, YOB_BUFFER = 5, F_PROB_BUFFER = 0.8, firm_year_user_dedup = True, W_COUNTRY = 0.70, W_YOB = 0.20, W_GENDER = 0.10, W_OCC = 0.0, OCC_SCORE_HALFLIFE = 500.0, MULTIPLICATIVE_SCORE = False):
    # Principled firm quality multiplier: P(right firm) in [0, 1].
    # - NULL score (legacy matches): COALESCE to 100 → mult = 1.0 (unchanged)
    # - has_name_match_pos = 1: position history confirms firm → override to 1.0
    # - Otherwise: raw score/100, no artificial floor (score=40 → 0.40, score=10 → 0.10)
    llm_match_score_norm_expr = "GREATEST(LEAST(COALESCE(llm_match_score, 100), 100), 0) / 100.0"
    firm_quality_mult_expr = f"CASE WHEN COALESCE(has_name_match_pos, 0) = 1 THEN 1.0 ELSE ({llm_match_score_norm_expr}) END"
    dedup_clause = (
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY lottery_year, user_id ORDER BY total_score DESC, foia_indiv_id) = 1"
        if firm_year_user_dedup else ""
    )
    # Occupation score: hyperbolic decay on H-1B occupation rank.
    # Formula: 1 / (1 + (rank-1) / K), where K = OCC_SCORE_HALFLIFE (rank at which score = 0.5).
    # NULL min_h1b_occ_rank (no position data) → neutral 0.5. W_OCC=0 is backward-compatible.
    occ_score_expr = f"""CASE WHEN min_h1b_occ_rank IS NULL THEN 0.5
             ELSE 1.0 / (1.0 + (GREATEST(1, min_h1b_occ_rank) - 1)::FLOAT / {OCC_SCORE_HALFLIFE})
        END"""
    # YOB score: extracted to a variable so it can be reused in both scoring modes.
    yob_score_expr = f"""CASE
                WHEN est_yob IS NULL THEN 0.5
                WHEN ABS(est_yob - yob::INTEGER) <= {YOB_BUFFER}
                    THEN 1.0 - ABS(est_yob - yob::INTEGER)::FLOAT / ({YOB_BUFFER} + 1.0)
                ELSE 0.0
            END"""
    if MULTIPLICATIVE_SCORE:
        # Weighted geometric mean: score^weight for each signal. Equivalent to a weighted sum in
        # log-space, so the same weights (w_country=0.70, w_yob=0.20, w_gender=0.10) preserve the
        # relative downweighting of YOB and gender vs country. A small floor (1e-9) prevents any
        # single signal from zeroing the total score (e.g. yob outside buffer but known).
        # w_occ: since it's typically 0, use a hybrid blend for the occ component (same as additive).
        base_score_expr = (
            f"POWER(GREATEST(country_score, 1e-9), {W_COUNTRY})"
            f" * POWER(GREATEST({yob_score_expr}, 1e-9), {W_YOB})"
            f" * POWER(GREATEST(f_score, 1e-9), {W_GENDER})"
        )
        total_score_expr = f"({base_score_expr} * (1 - {W_OCC}) + ({occ_score_expr}) * {W_OCC}) * {firm_quality_mult_expr}"
    else:
        # Additive (weighted sum): original formula.
        total_score_expr = (
            f"(({yob_score_expr} * {W_YOB} + f_score * {W_GENDER} + country_score * {W_COUNTRY})"
            f" * (1 - {W_OCC}) + ({occ_score_expr}) * {W_OCC}) * {firm_quality_mult_expr}"
        )
    return f"""
    SELECT *, total_score/(SUM(total_score) OVER(PARTITION BY foia_indiv_id)) AS weight_norm FROM (
        SELECT foia_indiv_id, foia_firm_uid, FEIN, lottery_year, rcid, llm_match_score, user_id, fullname, foia_country, rev_country, subregion, country_score, subregion_score, {COUNTRY_UNCERTAIN_SELECT}, country_rank_score, female_ind, f_prob_avg, f_score, yob, est_yob, max_yob, n_match_raw, startdatediff, enddatediff, updatediff, updatediff_activity, stem_ind, foia_occ_ind, n_unique_country, min_h1b_occ_rank, months_since_grad, n_apps, status_type, ade_ind, ade_year, last_grad_year, foia_highest_ed_level, rev_highest_ed_level, prev_visa, high_rep_emp_ind, no_rep_emp_ind, field_clean, fields, positions, rcids, DOT_CODE, JOB_TITLE, n_match_filt, COALESCE(has_name_match_pos, 0) AS has_name_match_pos,
            (COUNT(DISTINCT foia_indiv_id) OVER(PARTITION BY foia_firm_uid, lottery_year))/n_apps AS share_apps_matched_emp,
            COUNT(DISTINCT user_id) OVER(PARTITION BY foia_firm_uid, lottery_year) AS n_rev_users_emp,
            (COUNT(*) OVER(PARTITION BY foia_firm_uid, lottery_year))/(COUNT(DISTINCT foia_indiv_id) OVER(PARTITION BY foia_firm_uid, lottery_year)) AS match_mult_emp,
            (COUNT(*) OVER(PARTITION BY foia_firm_uid, lottery_year))/(COUNT(DISTINCT user_id) OVER(PARTITION BY foia_firm_uid, lottery_year)) AS rev_mult_emp,
            COUNT(DISTINCT foia_indiv_id) OVER(PARTITION BY foia_firm_uid, lottery_year) AS n_apps_matched_emp,
            COUNT(DISTINCT status_type) OVER(PARTITION BY foia_firm_uid, lottery_year) AS n_unique_wintype_emp,
            {llm_match_score_norm_expr} AS llm_match_score_norm,
            {firm_quality_mult_expr} AS firm_match_quality_mult,
            {occ_score_expr} AS occ_score,
            {total_score_expr} AS total_score
        FROM ({stage_input_query})
        {dedup_clause}
    )
    """


def _build_stage_final_query(
    stage_weighted_query,
    postfilt = "none",
    MATCH_MULT_CUTOFF = 4,
    REV_MULT_COEFF = 1,
    AMBIGUITY_WEIGHT_GAP_CUTOFF = icfg.BUILD_AMBIGUITY_WEIGHT_GAP_CUTOFF,
    BAD_MATCH_GUARD_ENABLED = icfg.BUILD_BAD_MATCH_GUARD_ENABLED,
    BAD_MATCH_GUARD_SUBREGION_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_SUBREGION_SCORE_LT,
    BAD_MATCH_GUARD_F_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_F_SCORE_LT,
    BAD_MATCH_GUARD_TOTAL_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT,
):
    predicates = []
    if postfilt in ("indiv", "emp"):
        predicates.extend(
            [
                "share_apps_matched_emp = 1",
                f"rev_mult_emp < {REV_MULT_COEFF}*n_apps_matched_emp",
                "n_unique_wintype_emp > 1",
            ]
        )
    if postfilt == "emp":
        predicates.append(f"match_mult_emp <= {MATCH_MULT_CUTOFF}")
    if BAD_MATCH_GUARD_ENABLED:
        predicates.append(
            "NOT ("
            f"country_score = 0 AND subregion_score < {BAD_MATCH_GUARD_SUBREGION_SCORE_LT} "
            f"AND (est_yob IS NULL OR f_score < {BAD_MATCH_GUARD_F_SCORE_LT}) "
            f"AND total_score < {BAD_MATCH_GUARD_TOTAL_SCORE_LT}"
            ")"
        )

    where_sql = ""
    if predicates:
        where_sql = "WHERE " + " AND ".join(predicates)

    return f"""
    WITH filtered AS (
        SELECT *
        FROM ({stage_weighted_query})
        {where_sql}
    ),
    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER(
                PARTITION BY foia_indiv_id
                ORDER BY
                    weight_norm DESC,
                    total_score DESC,
                    user_id,
                    rcid,
                    rev_country,
                    subregion,
                    fullname,
                    llm_match_score DESC NULLS LAST,
                    est_yob ASC NULLS LAST
            ) AS match_rank,
            LEAD(weight_norm) OVER(
                PARTITION BY foia_indiv_id
                ORDER BY
                    weight_norm DESC,
                    total_score DESC,
                    user_id,
                    rcid,
                    rev_country,
                    subregion,
                    fullname,
                    llm_match_score DESC NULLS LAST,
                    est_yob ASC NULLS LAST
            ) AS next_weight_norm
        FROM filtered
    ),
    annotated AS (
        SELECT
            *,
            CASE
                WHEN match_rank = 1 THEN weight_norm - COALESCE(next_weight_norm, 0)
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
        MAX(top_match_ambiguous_ind) OVER(PARTITION BY foia_indiv_id) AS app_ambiguous_ind
    FROM annotated
    """


def _build_merge_filt_stage_queries(
    merge_raw_tab,
    postfilt = "none",
    MATCH_MULT_CUTOFF = 4,
    REV_MULT_COEFF = 1,
    COUNTRY_SCORE_CUTOFF = 0.03,
    COUNTRY_FALLBACK_TOPN = 3,
    COUNTRY_TOTAL_MARGIN = 0.08,
    MONTH_BUFFER = 0,  # deprecated: date filters are now year-level; kept for API compatibility
    YOB_BUFFER = 5,
    F_PROB_BUFFER = 0.8,
    GRAD_YEAR_BUFFER = (15, 35),
    firm_year_user_dedup = False,
    W_COUNTRY = 0.70,
    W_YOB = 0.20,
    W_GENDER = 0.10,
    W_OCC = 0.0,
    OCC_SCORE_HALFLIFE = 500.0,
    FOIA_OCC_RANK_CUTOFF = None,
    MULTIPLICATIVE_SCORE = False,
    NO_COUNTRY_MIN_SUBREGION_SCORE = icfg.BUILD_NO_COUNTRY_MIN_SUBREGION_SCORE,
    NO_COUNTRY_MIN_TOTAL_SCORE = icfg.BUILD_NO_COUNTRY_MIN_TOTAL_SCORE,
    NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL = icfg.BUILD_NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL,
    AMBIGUITY_WEIGHT_GAP_CUTOFF = icfg.BUILD_AMBIGUITY_WEIGHT_GAP_CUTOFF,
    BAD_MATCH_GUARD_ENABLED = icfg.BUILD_BAD_MATCH_GUARD_ENABLED,
    BAD_MATCH_GUARD_SUBREGION_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_SUBREGION_SCORE_LT,
    BAD_MATCH_GUARD_F_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_F_SCORE_LT,
    BAD_MATCH_GUARD_TOTAL_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT,
):
    stage_match_order = f"""
    SELECT *,
        CASE
            WHEN {COUNTRY_UNCERTAIN_EXPR} = 1 THEN 0.6*country_score + 0.4*subregion_score
            ELSE country_score
        END AS country_rank_score,
        ROW_NUMBER() OVER(
            PARTITION BY foia_indiv_id, user_id
            ORDER BY
                CASE
                    WHEN {COUNTRY_UNCERTAIN_EXPR} = 1 THEN 0.6*country_score + 0.4*subregion_score
                    ELSE country_score
                END DESC,
                total_score DESC,
                rev_country ASC,
                subregion ASC,
                rcid ASC,
                fullname ASC,
                llm_match_score DESC NULLS LAST,
                est_yob ASC NULLS LAST
        ) AS match_order_ind,
        MAX(country_score) OVER(PARTITION BY foia_indiv_id) AS max_country_score
    FROM {merge_raw_tab}
    WHERE f_score >= 1 - {F_PROB_BUFFER}
      AND (ABS(yob::INTEGER - est_yob) <= {YOB_BUFFER} OR (est_yob IS NULL AND yob::INTEGER <= max_yob))
      -- Year-level date filters (robust to LinkedIn 01-01 placeholder dates)
      AND YEAR(first_startdate) - (lottery_year::INT - 1) BETWEEN -4 AND 0
      AND YEAR(last_enddate) >= (lottery_year::INT - 1)
      AND (last_grad_year IS NULL OR (last_grad_year::INTEGER - yob::INTEGER) BETWEEN {GRAD_YEAR_BUFFER[0]} AND {GRAD_YEAR_BUFFER[1]})
      AND (grad_years_since IS NULL OR grad_years_since <= 4)
    """

    stage_match_filt = f"""
    WITH base AS (
        SELECT *,
            MAX(country_score) OVER(PARTITION BY foia_indiv_id) AS max_country_score_app,
            MAX(total_score) OVER(PARTITION BY foia_indiv_id) AS max_total_score_app,
            ROW_NUMBER() OVER(
                PARTITION BY foia_indiv_id
                ORDER BY
                    country_rank_score DESC,
                    total_score DESC,
                    user_id ASC,
                    rcid ASC,
                    rev_country ASC,
                    subregion ASC,
                    fullname ASC,
                    llm_match_score DESC NULLS LAST,
                    est_yob ASC NULLS LAST
            ) AS app_country_rank
        FROM ({stage_match_order})
        WHERE match_order_ind = 1
          AND (stem_ind IS NULL OR stem_ind = 1 OR foia_occ_ind IS NULL OR (foia_occ_ind = 1{f" AND min_h1b_occ_rank <= {FOIA_OCC_RANK_CUTOFF}" if FOIA_OCC_RANK_CUTOFF is not None else ""}))
    )
    SELECT *,
        COUNT(*) OVER(PARTITION BY foia_indiv_id) AS n_match_filt
    FROM base
    WHERE country_score > {COUNTRY_SCORE_CUTOFF}
       OR (
            max_country_score_app <= {COUNTRY_SCORE_CUTOFF}
            AND subregion_score >= {NO_COUNTRY_MIN_SUBREGION_SCORE}
            AND total_score >= {NO_COUNTRY_MIN_TOTAL_SCORE}
            AND (est_yob IS NOT NULL OR f_score >= {NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL})
       )
       OR (
            {COUNTRY_UNCERTAIN_EXPR} = 1
            AND (
                app_country_rank <= {COUNTRY_FALLBACK_TOPN}
                OR total_score >= max_total_score_app - {COUNTRY_TOTAL_MARGIN}
            )
       )
    """

    stage_after_indiv_cutoff = stage_match_filt
    if postfilt == "indiv":
        stage_after_indiv_cutoff = (
            f"SELECT * FROM ({stage_match_filt}) WHERE n_match_filt <= {MATCH_MULT_CUTOFF}"
        )

    stage_weighted = _build_stage_weighted_query(
        stage_after_indiv_cutoff,
        YOB_BUFFER = YOB_BUFFER,
        F_PROB_BUFFER = F_PROB_BUFFER,
        firm_year_user_dedup = firm_year_user_dedup,
        W_COUNTRY = W_COUNTRY,
        W_YOB = W_YOB,
        W_GENDER = W_GENDER,
        W_OCC = W_OCC,
        OCC_SCORE_HALFLIFE = OCC_SCORE_HALFLIFE,
        MULTIPLICATIVE_SCORE = MULTIPLICATIVE_SCORE,
    )
    stage_final = _build_stage_final_query(
        stage_weighted,
        postfilt = postfilt,
        MATCH_MULT_CUTOFF = MATCH_MULT_CUTOFF,
        REV_MULT_COEFF = REV_MULT_COEFF,
        AMBIGUITY_WEIGHT_GAP_CUTOFF = AMBIGUITY_WEIGHT_GAP_CUTOFF,
        BAD_MATCH_GUARD_ENABLED = BAD_MATCH_GUARD_ENABLED,
        BAD_MATCH_GUARD_SUBREGION_SCORE_LT = BAD_MATCH_GUARD_SUBREGION_SCORE_LT,
        BAD_MATCH_GUARD_F_SCORE_LT = BAD_MATCH_GUARD_F_SCORE_LT,
        BAD_MATCH_GUARD_TOTAL_SCORE_LT = BAD_MATCH_GUARD_TOTAL_SCORE_LT,
    )

    return {
        "match_order": stage_match_order,
        "match_filt": stage_match_filt,
        "after_indiv_cutoff": stage_after_indiv_cutoff,
        "weighted": stage_weighted,
        "final": stage_final,
    }


def _build_stage_queries_from_match_filt(
    match_filt_tab,
    postfilt = "none",
    MATCH_MULT_CUTOFF = 4,
    REV_MULT_COEFF = 1,
    YOB_BUFFER = 5,
    F_PROB_BUFFER = 0.8,
    firm_year_user_dedup = False,
    W_COUNTRY = 0.70,
    W_YOB = 0.20,
    W_GENDER = 0.10,
    W_OCC = 0.0,
    OCC_SCORE_HALFLIFE = 500.0,
    MULTIPLICATIVE_SCORE = False,
    AMBIGUITY_WEIGHT_GAP_CUTOFF = icfg.BUILD_AMBIGUITY_WEIGHT_GAP_CUTOFF,
    BAD_MATCH_GUARD_ENABLED = icfg.BUILD_BAD_MATCH_GUARD_ENABLED,
    BAD_MATCH_GUARD_SUBREGION_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_SUBREGION_SCORE_LT,
    BAD_MATCH_GUARD_F_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_F_SCORE_LT,
    BAD_MATCH_GUARD_TOTAL_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT,
):
    stage_match_filt = f"SELECT * FROM {match_filt_tab}"
    stage_after_indiv_cutoff = stage_match_filt
    if postfilt == "indiv":
        stage_after_indiv_cutoff = (
            f"SELECT * FROM {match_filt_tab} WHERE n_match_filt <= {MATCH_MULT_CUTOFF}"
        )

    stage_weighted = _build_stage_weighted_query(
        stage_after_indiv_cutoff,
        YOB_BUFFER = YOB_BUFFER,
        F_PROB_BUFFER = F_PROB_BUFFER,
        firm_year_user_dedup = firm_year_user_dedup,
        W_COUNTRY = W_COUNTRY,
        W_YOB = W_YOB,
        W_GENDER = W_GENDER,
        W_OCC = W_OCC,
        OCC_SCORE_HALFLIFE = OCC_SCORE_HALFLIFE,
        MULTIPLICATIVE_SCORE = MULTIPLICATIVE_SCORE,
    )
    stage_final = _build_stage_final_query(
        stage_weighted,
        postfilt = postfilt,
        MATCH_MULT_CUTOFF = MATCH_MULT_CUTOFF,
        REV_MULT_COEFF = REV_MULT_COEFF,
        AMBIGUITY_WEIGHT_GAP_CUTOFF = AMBIGUITY_WEIGHT_GAP_CUTOFF,
        BAD_MATCH_GUARD_ENABLED = BAD_MATCH_GUARD_ENABLED,
        BAD_MATCH_GUARD_SUBREGION_SCORE_LT = BAD_MATCH_GUARD_SUBREGION_SCORE_LT,
        BAD_MATCH_GUARD_F_SCORE_LT = BAD_MATCH_GUARD_F_SCORE_LT,
        BAD_MATCH_GUARD_TOTAL_SCORE_LT = BAD_MATCH_GUARD_TOTAL_SCORE_LT,
    )
    return {
        "match_filt": stage_match_filt,
        "after_indiv_cutoff": stage_after_indiv_cutoff,
        "weighted": stage_weighted,
        "final": stage_final,
    }


def _build_stage_strict_query(
    baseline_tab,
    min_weight_norm = icfg.STRICT_MIN_WEIGHT_NORM,
    min_total_score = icfg.STRICT_MIN_TOTAL_SCORE,
    min_firm_quality = icfg.STRICT_MIN_FIRM_QUALITY,
    min_country_score = icfg.STRICT_MIN_COUNTRY_SCORE,
    require_est_yob = icfg.STRICT_REQUIRE_EST_YOB,
    max_n_match_filt = icfg.STRICT_MAX_N_MATCH_FILT,
):
    """Post-hoc strict filter on baseline weighted+ranked table.

    Keeps only rank=1 matches that satisfy all strict thresholds, minimizing
    false positives at the cost of recall. Configurable via indiv_merge.yaml.
      - weight_norm: relative dominance (>=0.9 means 90% of score mass on this candidate)
      - total_score: absolute match quality floor (guards against weight_norm=1 on a single weak candidate)
      - firm_match_quality_mult: firm confidence (1.0 if name match; else llm_match_score_norm — handles rebrands)
      - country_score: strong country-level evidence required (not just subregion proximity)
      - est_yob: birth year must be non-null (yob component not imputed to 0.5 default)
    """
    conditions = [
        "match_rank = 1",
        f"weight_norm >= {min_weight_norm}",
        f"total_score >= {min_total_score}",
        f"firm_match_quality_mult >= {min_firm_quality}",
        f"country_score >= {min_country_score}",
    ]
    if require_est_yob:
        conditions.append("est_yob IS NOT NULL")
    if max_n_match_filt is not None:
        conditions.append(f"n_match_filt <= {max_n_match_filt}")
    where_clause = " AND ".join(conditions)
    return f"SELECT * FROM {baseline_tab} WHERE {where_clause}"


def _print_readable_testing_samples(
    final_query,
    foia_tab = "foia_indiv",
    sample_n = 5,
    con = con_indiv,
):
    sample_n = max(1, int(sample_n))
    sample_query = f"""
    WITH matches AS (
        SELECT * FROM ({final_query})
    ),
    candidate_matches AS (
        SELECT DISTINCT
            m.foia_indiv_id,
            m.foia_firm_uid,
            m.FEIN,
            m.lottery_year,
            m.status_type,
            m.foia_country,
            m.yob,
            m.female_ind,
            m.user_id::BIGINT AS user_id,
            m.fullname,
            m.rev_country,
            m.est_yob,
            m.f_prob_avg,
            m.rcid,
            m.llm_match_score,
            m.llm_match_score_norm,
            m.firm_match_quality_mult,
            m.weight_norm
        FROM matches AS m
    ),
    sampled AS (
        SELECT
            *
        FROM candidate_matches
        ORDER BY weight_norm DESC NULLS LAST, RANDOM()
        LIMIT {sample_n}
    ),
    sampled_users AS (
        SELECT DISTINCT user_id
        FROM sampled
    ),
    foia_info AS (
        SELECT foia_indiv_id, MAX(employer_name) AS employer_name
        FROM {foia_tab}
        GROUP BY foia_indiv_id
    ),
    educ_hist AS (
        SELECT
            e.user_id::BIGINT AS user_id,
            string_agg(
                COALESCE(degree_clean, 'Unknown degree') || ' @ ' || COALESCE(university_raw, 'Unknown school') ||
                ' (' || COALESCE(match_country, 'Unknown country') || ', ' ||
                COALESCE(SUBSTRING(CAST(startdate AS VARCHAR), 1, 4), '?') || '-' ||
                COALESCE(SUBSTRING(CAST(enddate AS VARCHAR), 1, 4), 'present') || ')',
                ' | ' ORDER BY startdate DESC
            ) AS educ_history
        FROM rev_educ_clean AS e
        JOIN sampled_users AS su
            ON e.user_id::BIGINT = su.user_id
        GROUP BY e.user_id
    ),
    work_ranked AS (
        SELECT
            p.user_id::BIGINT AS user_id,
            COALESCE(p.company_raw, 'Unknown company') AS company_raw,
            COALESCE(p.title_raw, 'Unknown title') AS title_raw,
            COALESCE(p.country, 'Unknown country') AS country,
            p.startdate,
            p.enddate,
            ROW_NUMBER() OVER (
                PARTITION BY p.user_id
                ORDER BY p.startdate DESC NULLS LAST, p.position_number DESC NULLS LAST
            ) AS rn
        FROM merged_pos_clean AS p
        JOIN sampled_users AS su
            ON p.user_id::BIGINT = su.user_id
    ),
    work_hist AS (
        SELECT
            user_id,
            string_agg(
                company_raw || ' - ' || title_raw ||
                ' (' || country || ', ' ||
                COALESCE(SUBSTRING(CAST(startdate AS VARCHAR), 1, 4), '?') || '-' ||
                COALESCE(SUBSTRING(CAST(enddate AS VARCHAR), 1, 4), 'present') || ')',
                ' | ' ORDER BY rn
            ) AS work_history
        FROM work_ranked
        WHERE rn <= 6
        GROUP BY user_id
    )
    SELECT
        s.*,
        f.employer_name AS foia_company_name,
        e.educ_history,
        w.work_history
    FROM sampled AS s
    LEFT JOIN foia_info AS f ON s.foia_indiv_id = f.foia_indiv_id
    LEFT JOIN educ_hist AS e ON s.user_id = e.user_id
    LEFT JOIN work_hist AS w ON s.user_id = w.user_id
    ORDER BY s.weight_norm DESC NULLS LAST, s.foia_indiv_id
    """
    sample_df = con.sql(sample_query).df()

    if sample_df.empty:
        print("No matches found in testing subset after filtering.")
        return

    print("")
    print("Sample matches (readable):")
    print("-" * 80)

    for i, row in sample_df.reset_index(drop = True).iterrows():
        foia_gender = "Female" if row["female_ind"] == 1 else ("Male" if row["female_ind"] == 0 else "Unknown")
        if pd.isna(row["f_prob_avg"]):
            rev_gender = "Unknown"
        elif row["f_prob_avg"] >= 0.5:
            rev_gender = f"Female (p={row['f_prob_avg']:.2f})"
        else:
            rev_gender = f"Male (p={1 - row['f_prob_avg']:.2f})"
        foia_company = (
            row["foia_company_name"]
            if pd.notna(row["foia_company_name"])
            else "Unknown company"
        )
        rev_yob = "Unknown" if pd.isna(row["est_yob"]) else f"{int(row['est_yob'])}"

        work_history = row["work_history"] if pd.notna(row["work_history"]) else "No work history found"
        educ_history = row["educ_history"] if pd.notna(row["educ_history"]) else "No education history found"
        rev_company = "Unknown company"
        if work_history and work_history != "No work history found":
            rev_company = work_history.split(" | ")[0].split(" - ")[0]

        print(f"Match {i + 1}")
        print(
            f"Person A (foia): {foia_company} | yob={row['yob']} | gender={foia_gender} | "
            f"country={row['foia_country']} | year={row['lottery_year']} | status={row['status_type']} | "
            f"foia_indiv_id={row['foia_indiv_id']}"
        )
        print(
            f"Person B (potential match in revelio): company={rev_company} | user_id={row['user_id']} | "
            f"name={row['fullname']} | matched_rcid={row['rcid']} | imputed_yob={rev_yob} | "
            f"imputed_gender={rev_gender} | imputed_country={row['rev_country']}"
        )
        if pd.notna(row["llm_match_score"]):
            print(
                f"  Firm match quality: raw={row['llm_match_score']:.2f} "
                f"norm={row['llm_match_score_norm']:.3f} "
                f"mult={row['firm_match_quality_mult']:.3f}"
            )
        print(f"  Educ history: {educ_history}")
        print(f"  Work history: {work_history}")
        print(f"  Match weight: {row['weight_norm']:.4f}" if pd.notna(row["weight_norm"]) else "  Match weight: NA")
        print("-" * 80)


# GET DF 
def merge_df(rev_tab = 'rev_indiv', foia_tab = 'foia_indiv', with_t_vars = False, postfilt = 'none', MATCH_MULT_CUTOFF = 4, REV_MULT_COEFF = 1, foia_prefilt = '', subregion = True, COUNTRY_SCORE_CUTOFF = 0.03, COUNTRY_FALLBACK_TOPN = 3, COUNTRY_TOTAL_MARGIN = 0.08, MONTH_BUFFER = 0, YOB_BUFFER = 5, F_PROB_BUFFER = 0.8, GRAD_YEAR_BUFFER = (15, 35), NO_COUNTRY_MIN_SUBREGION_SCORE = None, NO_COUNTRY_MIN_TOTAL_SCORE = None, NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL = None, AMBIGUITY_WEIGHT_GAP_CUTOFF = None, BAD_MATCH_GUARD_ENABLED = None, BAD_MATCH_GUARD_SUBREGION_SCORE_LT = None, BAD_MATCH_GUARD_F_SCORE_LT = None, BAD_MATCH_GUARD_TOTAL_SCORE_LT = None, verbose = False, testing = False, test_sample_matches = 5, test_random_seed = None, test_firm_uid = None, test_lottery_year = None, test_materialize_intermediate_tables = None, test_table_prefix = None, con = con_indiv):
    return con.sql(
        merge(
            rev_tab = rev_tab,
            foia_tab = foia_tab,
            with_t_vars = with_t_vars,
            postfilt = postfilt,
            MATCH_MULT_CUTOFF = MATCH_MULT_CUTOFF,
            REV_MULT_COEFF = REV_MULT_COEFF,
            foia_prefilt = foia_prefilt,
            subregion = subregion,
            COUNTRY_SCORE_CUTOFF = COUNTRY_SCORE_CUTOFF,
            COUNTRY_FALLBACK_TOPN = COUNTRY_FALLBACK_TOPN,
            COUNTRY_TOTAL_MARGIN = COUNTRY_TOTAL_MARGIN,
            MONTH_BUFFER = MONTH_BUFFER,
            YOB_BUFFER = YOB_BUFFER,
            F_PROB_BUFFER = F_PROB_BUFFER,
            GRAD_YEAR_BUFFER = GRAD_YEAR_BUFFER,
            NO_COUNTRY_MIN_SUBREGION_SCORE = NO_COUNTRY_MIN_SUBREGION_SCORE,
            NO_COUNTRY_MIN_TOTAL_SCORE = NO_COUNTRY_MIN_TOTAL_SCORE,
            NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL = NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL,
            AMBIGUITY_WEIGHT_GAP_CUTOFF = AMBIGUITY_WEIGHT_GAP_CUTOFF,
            BAD_MATCH_GUARD_ENABLED = BAD_MATCH_GUARD_ENABLED,
            BAD_MATCH_GUARD_SUBREGION_SCORE_LT = BAD_MATCH_GUARD_SUBREGION_SCORE_LT,
            BAD_MATCH_GUARD_F_SCORE_LT = BAD_MATCH_GUARD_F_SCORE_LT,
            BAD_MATCH_GUARD_TOTAL_SCORE_LT = BAD_MATCH_GUARD_TOTAL_SCORE_LT,
            verbose = verbose,
            testing = testing,
            test_sample_matches = test_sample_matches,
            test_random_seed = test_random_seed,
            test_firm_uid = test_firm_uid,
            test_lottery_year = test_lottery_year,
            test_materialize_intermediate_tables = test_materialize_intermediate_tables,
            test_table_prefix = test_table_prefix,
            con = con,
        )
    ).df()

# WRAPPER
def merge(rev_tab = 'rev_indiv', foia_tab = 'foia_indiv', with_t_vars = False, postfilt = 'none', MATCH_MULT_CUTOFF = 4, REV_MULT_COEFF = 1, foia_prefilt = '', subregion = True, FIRM_NAME_MATCH_THRESHOLD = None, COUNTRY_SCORE_CUTOFF = 0.03, COUNTRY_FALLBACK_TOPN = 3, COUNTRY_TOTAL_MARGIN = 0.08, MONTH_BUFFER = 0, YOB_BUFFER = 5, F_PROB_BUFFER = 0.8, GRAD_YEAR_BUFFER = (15, 35), NO_COUNTRY_MIN_SUBREGION_SCORE = None, NO_COUNTRY_MIN_TOTAL_SCORE = None, NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL = None, AMBIGUITY_WEIGHT_GAP_CUTOFF = None, BAD_MATCH_GUARD_ENABLED = None, BAD_MATCH_GUARD_SUBREGION_SCORE_LT = None, BAD_MATCH_GUARD_F_SCORE_LT = None, BAD_MATCH_GUARD_TOTAL_SCORE_LT = None, verbose = False, testing = False, test_sample_matches = 5, test_random_seed = None, test_firm_uid = None, test_lottery_year = None, test_materialize_intermediate_tables = None, test_table_prefix = None, con = con_indiv):
    if NO_COUNTRY_MIN_SUBREGION_SCORE is None:
        NO_COUNTRY_MIN_SUBREGION_SCORE = icfg.BUILD_NO_COUNTRY_MIN_SUBREGION_SCORE
    if NO_COUNTRY_MIN_TOTAL_SCORE is None:
        NO_COUNTRY_MIN_TOTAL_SCORE = icfg.BUILD_NO_COUNTRY_MIN_TOTAL_SCORE
    if NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL is None:
        NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL = icfg.BUILD_NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL
    if AMBIGUITY_WEIGHT_GAP_CUTOFF is None:
        AMBIGUITY_WEIGHT_GAP_CUTOFF = icfg.BUILD_AMBIGUITY_WEIGHT_GAP_CUTOFF
    if BAD_MATCH_GUARD_ENABLED is None:
        BAD_MATCH_GUARD_ENABLED = icfg.BUILD_BAD_MATCH_GUARD_ENABLED
    if BAD_MATCH_GUARD_SUBREGION_SCORE_LT is None:
        BAD_MATCH_GUARD_SUBREGION_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_SUBREGION_SCORE_LT
    if BAD_MATCH_GUARD_F_SCORE_LT is None:
        BAD_MATCH_GUARD_F_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_F_SCORE_LT
    if BAD_MATCH_GUARD_TOTAL_SCORE_LT is None:
        BAD_MATCH_GUARD_TOTAL_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT

    test_foia_prefilt = foia_prefilt
    if testing:
        if test_materialize_intermediate_tables is None:
            test_materialize_intermediate_tables = icfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES
        if test_table_prefix is None:
            test_table_prefix = icfg.TESTING_TABLE_PREFIX

        print("")
        print("========== TESTING MODE: indiv_merge ==========")
        print("Selecting foia_firm_uid x lottery_year subset...")
        pick_uid, pick_year = _pick_testing_subset(
            foia_tab = foia_tab,
            foia_prefilt = foia_prefilt,
            test_firm_uid = test_firm_uid,
            test_lottery_year = test_lottery_year,
            test_random_seed = test_random_seed,
            con = con,
        )
        subset_condition = (
            f"foia_firm_uid = {_sql_quote(pick_uid)} AND lottery_year = {_sql_quote(pick_year)}"
        )
        test_foia_prefilt = _combine_foia_prefilt(foia_prefilt, subset_condition)
        print(f"Selected subset: foia_firm_uid={pick_uid} | lottery_year={pick_year}")
        n_subset_apps = con.sql(
            f"SELECT COUNT(DISTINCT foia_indiv_id) AS n FROM {foia_tab} {test_foia_prefilt}"
        ).df().iloc[0, 0]
        print(f"Subset size: {int(n_subset_apps):,} FOIA applications")
        print("==============================================")

    raw_query = merge_raw_func(
        rev_tab,
        foia_tab,
        foia_prefilt = test_foia_prefilt,
        subregion = subregion,
        FIRM_NAME_MATCH_THRESHOLD = FIRM_NAME_MATCH_THRESHOLD,
    )
    stage_queries = _build_merge_filt_stage_queries(
        f"({raw_query})",
        postfilt = postfilt,
        MATCH_MULT_CUTOFF = MATCH_MULT_CUTOFF,
        REV_MULT_COEFF = REV_MULT_COEFF,
        COUNTRY_SCORE_CUTOFF = COUNTRY_SCORE_CUTOFF,
        COUNTRY_FALLBACK_TOPN = COUNTRY_FALLBACK_TOPN,
        COUNTRY_TOTAL_MARGIN = COUNTRY_TOTAL_MARGIN,
        MONTH_BUFFER = MONTH_BUFFER,
        YOB_BUFFER = YOB_BUFFER,
        F_PROB_BUFFER = F_PROB_BUFFER,
        GRAD_YEAR_BUFFER = GRAD_YEAR_BUFFER,
        NO_COUNTRY_MIN_SUBREGION_SCORE = NO_COUNTRY_MIN_SUBREGION_SCORE,
        NO_COUNTRY_MIN_TOTAL_SCORE = NO_COUNTRY_MIN_TOTAL_SCORE,
        NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL = NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL,
        AMBIGUITY_WEIGHT_GAP_CUTOFF = AMBIGUITY_WEIGHT_GAP_CUTOFF,
        BAD_MATCH_GUARD_ENABLED = BAD_MATCH_GUARD_ENABLED,
        BAD_MATCH_GUARD_SUBREGION_SCORE_LT = BAD_MATCH_GUARD_SUBREGION_SCORE_LT,
        BAD_MATCH_GUARD_F_SCORE_LT = BAD_MATCH_GUARD_F_SCORE_LT,
        BAD_MATCH_GUARD_TOTAL_SCORE_LT = BAD_MATCH_GUARD_TOTAL_SCORE_LT
    )
    base_final_query = stage_queries["final"]
    str_out = base_final_query

    if with_t_vars:
        # Materialize base merge once to avoid recomputing a heavy subquery multiple times.
        str_out = f"""
        WITH base AS ({base_final_query})
        SELECT * EXCLUDE (b.foia_indiv_id, b.user_id)
        FROM base AS a
        LEFT JOIN ({get_rel_year_inds_wide_by_t('base')}) AS b
            ON a.foia_indiv_id = b.foia_indiv_id AND a.user_id = b.user_id
        """
    
    if verbose:
        print(f"Main H-1B Sample Size: {con.sql(f"SELECT COUNT(DISTINCT foia_indiv_id) FROM {foia_tab} {test_foia_prefilt}").df().iloc[0,0]} Applications")
        print(f"Main LinkedIn Sample Size: {con.sql(f"SELECT COUNT(DISTINCT user_id) FROM {rev_tab} WHERE (stem_ind IS NULL OR stem_ind = 1) AND (foia_occ_ind IS NULL OR foia_occ_ind = 1)").df().iloc[0,0]} Users")

        ns = con.sql(f"SELECT COUNT(*), COUNT(DISTINCT foia_indiv_id) FROM ({str_out})").df()
        print(f"Resulting Merge Size: {ns.iloc[0,0]} potential matches for {ns.iloc[0,1]} applications (multiplicity {round(ns.iloc[0,0]/ns.iloc[0,1],2)})")

    if testing:
        print("")
        print("Testing progress by stage:")
        _print_merge_stage("raw_join", raw_query, con = con)
        _print_merge_stage("after_filters_and_best_match", stage_queries["match_filt"], con = con)
        if postfilt == "indiv":
            _print_merge_stage("after_indiv_match_cutoff", stage_queries["after_indiv_cutoff"], con = con)
        _print_merge_stage("after_weighting", stage_queries["weighted"], con = con)
        _print_merge_stage("final_postfilter", base_final_query, con = con)
        if with_t_vars:
            _print_merge_stage("with_t_vars_joined", str_out, con = con)
        if test_materialize_intermediate_tables:
            table_pairs = [
                ("foia_scope", f"SELECT * FROM {foia_tab} {test_foia_prefilt}"),
                ("raw_join", raw_query),
                ("after_filters_and_best_match", stage_queries["match_filt"]),
                ("after_weighting", stage_queries["weighted"]),
                ("final", base_final_query),
            ]
            if postfilt == "indiv":
                table_pairs.insert(3, ("after_indiv_match_cutoff", stage_queries["after_indiv_cutoff"]))
            if with_t_vars:
                table_pairs.append(("with_t_vars", str_out))
            _materialize_testing_tables(
                table_pairs,
                table_prefix = test_table_prefix,
                con = con,
            )
        _print_readable_testing_samples(
            base_final_query,
            foia_tab = foia_tab,
            sample_n = test_sample_matches,
            con = con,
        )

    return str_out

#####################
# REV X FOIA MERGE FUNCTIONS
#####################
# MERGE 
def merge_raw_func(rev_tab, foia_tab, foia_prefilt = '', subregion = True, pos_tab = 'merged_pos_clean', FIRM_NAME_MATCH_THRESHOLD = None, ALPHA = 0.4, COMPETITION_WEIGHT = 0.0, COMPETITION_THRESHOLD = 0.0):
    if subregion:
        mergekey = 'subregion'
    else:
        mergekey = 'country'
    llm_match_score_expr = "TRY_CAST(b.llm_match_score AS DOUBLE)" if HAS_LLM_MATCH_SCORE_COL else "NULL::DOUBLE"
    if FIRM_NAME_MATCH_THRESHOLD is None:
        FIRM_NAME_MATCH_THRESHOLD = icfg.BUILD_FIRM_NAME_MATCH_THRESHOLD
    # Pre-compute name_match_pos efficiently:
    # 1. Distinct (foia_firm_uid, company_raw) pairs from candidate positions — constrains
    #    jaro_winkler to ~N_firms × avg_distinct_companies_per_cand (low millions, not billions).
    # 2. Apply jaro only to that small set, then map back to users.
    name_match_cte = f"""
    firm_names AS (
        SELECT DISTINCT foia_firm_uid, lower(employer_name) AS emp_name
        FROM {foia_tab}
        WHERE employer_name IS NOT NULL
    ),
    firm_cand_companies AS (
        SELECT DISTINCT u.foia_firm_uid, lower(p.company_raw) AS co_name
        FROM (SELECT DISTINCT user_id, foia_firm_uid FROM {rev_tab}) u
        JOIN {pos_tab} p ON u.user_id = p.user_id
        WHERE p.company_raw IS NOT NULL
    ),
    name_matched_companies AS (
        SELECT fcc.foia_firm_uid, fcc.co_name
        FROM firm_cand_companies fcc
        JOIN firm_names fn ON fcc.foia_firm_uid = fn.foia_firm_uid
        WHERE jaro_winkler_similarity(fn.emp_name, fcc.co_name) >= {FIRM_NAME_MATCH_THRESHOLD}
    ),
    name_match_pos AS (
        SELECT DISTINCT u.user_id, u.foia_firm_uid
        FROM (SELECT DISTINCT user_id, foia_firm_uid FROM {rev_tab}) u
        JOIN {pos_tab} p ON u.user_id = p.user_id
        JOIN name_matched_companies nmc
            ON u.foia_firm_uid = nmc.foia_firm_uid
            AND lower(p.company_raw) = nmc.co_name
    ),
    user_lottery_years AS (
        SELECT DISTINCT b.user_id, a.lottery_year::INT AS lottery_year
        FROM {foia_tab} a
        JOIN {rev_tab} b ON a.foia_firm_uid = b.foia_firm_uid
    ),
    latest_educ_enddate AS (
        SELECT uly.user_id, uly.lottery_year, e.ed_enddate AS grad_enddate
        FROM user_lottery_years uly
        JOIN rev_educ_clean e ON uly.user_id = e.user_id
        WHERE e.ed_enddate IS NOT NULL
          AND SUBSTRING(e.ed_startdate, 1, 4)::INT < uly.lottery_year
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY uly.user_id, uly.lottery_year
            ORDER BY e.education_number DESC, e.ed_enddate DESC
        ) = 1
    ),
    pos_country AS (
        SELECT DISTINCT user_id, country
        FROM {pos_tab}
    ),
    -- Country competition: for each (user_id, country), compute a country score equivalent using
    -- the same formula as the main country_score (incl. position term). Used to identify users
    -- with strong evidence for a different country than the current FOIA match.
    country_cs AS (
        SELECT ri.user_id, ri.country,
               GREATEST(COALESCE(ri.inst_score, 0.0),
                   (pc_all.user_id IS NOT NULL)::FLOAT,
                   {ALPHA} * LEAST(1.0, GREATEST(ri.nanat_subregion_score, ri.nt_subregion_score))
                   + (1.0 - {ALPHA}) * LEAST(1.0, ri.nanat_score / GREATEST(ri.nanat_subregion_score, 0.01))
               ) AS cs_equiv
        FROM {rev_tab} ri
        LEFT JOIN pos_country pc_all ON ri.user_id = pc_all.user_id AND ri.country = pc_all.country
    ),
    user_country_ranked AS (
        SELECT user_id, country, cs_equiv,
               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY cs_equiv DESC, country) AS cs_rank
        FROM country_cs
    ),
    user_top2_cs AS (
        -- Top-2 pattern: for each (user, match_country), max_other_cs = rank2 if current==rank1, else rank1.
        SELECT user_id,
               MAX(CASE WHEN cs_rank = 1 THEN cs_equiv ELSE NULL END) AS max1_cs,
               MAX(CASE WHEN cs_rank = 1 THEN country ELSE NULL END) AS max1_country,
               COALESCE(MAX(CASE WHEN cs_rank = 2 THEN cs_equiv ELSE NULL END), 0.0) AS max2_cs
        FROM user_country_ranked
        WHERE cs_rank <= 2
        GROUP BY user_id
    ),
    -- Most recent startdate across all positions AND educations per user.
    -- Better proxy for "when did this user last update their profile" than updated_dt
    -- (which reflects Revelio's scrape refresh, not the user's own profile activity).
    max_activity_startdate AS (
        SELECT user_id, MAX(startdate_all) AS max_startdate_all
        FROM (
            SELECT user_id, startdate AS startdate_all FROM {pos_tab}
            UNION ALL
            SELECT user_id, ed_startdate AS startdate_all FROM rev_educ_clean
        )
        GROUP BY user_id
    )"""
    # Build max_other_cs expression (reused in country_score and as a standalone column)
    max_other_cs_expr = f"CASE WHEN b.country = uts.max1_country THEN uts.max2_cs ELSE COALESCE(uts.max1_cs, 0.0) END"
    # Competition penalty multiplier: 1 when disabled (weight=0) or max_other below threshold
    if COMPETITION_WEIGHT == 0.0:
        competition_mult_expr = "1.0"
    else:
        competition_mult_expr = (
            f"GREATEST(0.0, 1.0 - {COMPETITION_WEIGHT}"
            f" * GREATEST(0.0, ({max_other_cs_expr}) - {COMPETITION_THRESHOLD}))"
        )
    str_out = f"""WITH {name_match_cte}
    SELECT *, a.country AS foia_country, b.country AS rev_country, COUNT(*) OVER(PARTITION BY foia_indiv_id) AS n_match_raw,
            DATEDIFF('month', ((a.lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, first_startdate) AS startdatediff,
            DATEDIFF('month', ((a.lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, last_enddate) AS enddatediff,
            DATEDIFF('month', led.grad_enddate::DATETIME,
                     ((a.lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME) AS months_since_grad,
            (a.lottery_year::INT - 1) - YEAR(led.grad_enddate::DATE) AS grad_years_since,
            SUBSTRING(min_startdate, 1, 4)::INTEGER - 16 AS max_yob,
            DATEDIFF('month', ((a.lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, updated_dt::DATETIME) AS updatediff,
            -- updatediff_activity: months from pre-lottery reference to the most recent position/educ startdate.
            -- Captures when the user last *added* an entry, not just when Revelio refreshed the profile.
            DATEDIFF('month', ((a.lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, mas.max_startdate_all::DATETIME) AS updatediff_activity,
            (f_prob + COALESCE(f_prob_nt, f_prob))/2 AS f_prob_avg,
            1 - ABS(female_ind - (f_prob + COALESCE(f_prob_nt, f_prob))/2) AS f_score,
            {llm_match_score_expr} AS llm_match_score,
            -- subregion_evidence: bounded [0,1]; combines both name models
            LEAST(1.0, GREATEST(nanat_subregion_score, nt_subregion_score)) AS subregion_score,
            -- country_score: alpha-blend of subregion_evidence and country_specificity,
            --   with institution and position overrides (GREATEST natural since inst_score penalises ambiguity).
            --   Optionally downweighted by competition_mult when user has strong other-country evidence.
            CASE WHEN a.country = b.country THEN GREATEST(
                COALESCE(inst_score, 0.0),
                (pc.user_id IS NOT NULL)::FLOAT,
                {ALPHA} * LEAST(1.0, GREATEST(nanat_subregion_score, nt_subregion_score))
                + (1.0 - {ALPHA}) * LEAST(1.0, nanat_score / GREATEST(nanat_subregion_score, 0.01))
            ) * {competition_mult_expr} ELSE 0.0 END AS country_score,
            -- max country score for other countries (for inspection and debugging)
            {max_other_cs_expr} AS max_other_country_score,
            a.highest_ed_level AS foia_highest_ed_level, b.highest_ed_level AS rev_highest_ed_level,
            COALESCE((nmp.user_id IS NOT NULL), FALSE)::INTEGER AS has_name_match_pos,
            (pc.user_id IS NOT NULL)::INTEGER AS position_score
        FROM (SELECT * FROM {foia_tab} {foia_prefilt} QUALIFY ROW_NUMBER() OVER (PARTITION BY foia_indiv_id ORDER BY rcid) = 1) AS a
        LEFT JOIN {rev_tab} AS b ON a.foia_firm_uid = b.foia_firm_uid AND a.{mergekey} = b.{mergekey}
        LEFT JOIN name_match_pos nmp ON b.user_id = nmp.user_id AND a.foia_firm_uid = nmp.foia_firm_uid
        LEFT JOIN latest_educ_enddate led ON b.user_id = led.user_id AND a.lottery_year::INT = led.lottery_year
        LEFT JOIN pos_country pc ON b.user_id = pc.user_id AND a.country = pc.country
        LEFT JOIN user_top2_cs uts ON b.user_id = uts.user_id
        LEFT JOIN max_activity_startdate mas ON b.user_id = mas.user_id"""
    return str_out

# FILTERS
# postfilt can be 'indiv' (filter on n_match_filt (# matches per app), then post-filter), 'emp' (filter on everything at employer level including match_mult_emp (avg # matches per app at employer level)), or anything else (no post-filtering)

def merge_filt_func(merge_raw_tab, postfilt = 'none', MATCH_MULT_CUTOFF = 4, REV_MULT_COEFF = 1, COUNTRY_SCORE_CUTOFF = 0.03, COUNTRY_FALLBACK_TOPN = 3, COUNTRY_TOTAL_MARGIN = 0.08, MONTH_BUFFER = 0, YOB_BUFFER = 5, F_PROB_BUFFER = 0.8, GRAD_YEAR_BUFFER = (15, 35), NO_COUNTRY_MIN_SUBREGION_SCORE = icfg.BUILD_NO_COUNTRY_MIN_SUBREGION_SCORE, NO_COUNTRY_MIN_TOTAL_SCORE = icfg.BUILD_NO_COUNTRY_MIN_TOTAL_SCORE, NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL = icfg.BUILD_NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL, AMBIGUITY_WEIGHT_GAP_CUTOFF = icfg.BUILD_AMBIGUITY_WEIGHT_GAP_CUTOFF, BAD_MATCH_GUARD_ENABLED = icfg.BUILD_BAD_MATCH_GUARD_ENABLED, BAD_MATCH_GUARD_SUBREGION_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_SUBREGION_SCORE_LT, BAD_MATCH_GUARD_F_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_F_SCORE_LT, BAD_MATCH_GUARD_TOTAL_SCORE_LT = icfg.BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT, W_COUNTRY = icfg.BUILD_W_COUNTRY, W_YOB = icfg.BUILD_W_YOB, W_GENDER = icfg.BUILD_W_GENDER):

    return _build_merge_filt_stage_queries(
        merge_raw_tab,
        postfilt = postfilt,
        MATCH_MULT_CUTOFF = MATCH_MULT_CUTOFF,
        REV_MULT_COEFF = REV_MULT_COEFF,
        COUNTRY_SCORE_CUTOFF = COUNTRY_SCORE_CUTOFF,
        COUNTRY_FALLBACK_TOPN = COUNTRY_FALLBACK_TOPN,
        COUNTRY_TOTAL_MARGIN = COUNTRY_TOTAL_MARGIN,
        MONTH_BUFFER = MONTH_BUFFER,
        YOB_BUFFER = YOB_BUFFER,
        F_PROB_BUFFER = F_PROB_BUFFER,
        GRAD_YEAR_BUFFER = GRAD_YEAR_BUFFER,
        NO_COUNTRY_MIN_SUBREGION_SCORE = NO_COUNTRY_MIN_SUBREGION_SCORE,
        NO_COUNTRY_MIN_TOTAL_SCORE = NO_COUNTRY_MIN_TOTAL_SCORE,
        NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL = NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL,
        AMBIGUITY_WEIGHT_GAP_CUTOFF = AMBIGUITY_WEIGHT_GAP_CUTOFF,
        BAD_MATCH_GUARD_ENABLED = BAD_MATCH_GUARD_ENABLED,
        BAD_MATCH_GUARD_SUBREGION_SCORE_LT = BAD_MATCH_GUARD_SUBREGION_SCORE_LT,
        BAD_MATCH_GUARD_F_SCORE_LT = BAD_MATCH_GUARD_F_SCORE_LT,
        BAD_MATCH_GUARD_TOTAL_SCORE_LT = BAD_MATCH_GUARD_TOTAL_SCORE_LT,
        W_COUNTRY = W_COUNTRY,
        W_YOB = W_YOB,
        W_GENDER = W_GENDER,
    )["final"]



#####################
# LONG POSITION/EDUC MERGE
#####################
def get_rel_year_inds_wide(merge_tab, t0 = -1, t1 = 2, pos_tab = 'merged_pos_clean', educ_tab = 'rev_educ_clean'):

    # join position and education data, unpivot long on variable, then pivot wide on variable x t
    str_out = f"""
    PIVOT 
        (UNPIVOT     
            (SELECT * EXCLUDE(b.foia_indiv_id, b.user_id, b.t) 
            FROM ({get_rel_year_inds_pos(merge_tab, pos_tab = pos_tab, t0=t0, t1=t1)}) AS a JOIN ({get_rel_year_inds_educ(merge_tab, educ_tab = educ_tab, t0=t0, t1=t1)}) AS b ON a.foia_indiv_id = b.foia_indiv_id AND a.user_id = b.user_id AND a.t = b.t)
        ON * EXCLUDE(foia_indiv_id, user_id, t) 
        INTO NAME var VALUE val)
    ON var || t USING FIRST(val)
    """

    return str_out


def get_rel_year_inds_wide_by_t(merge_tab, t0 = -1, t1 = 2, pos_tab = 'merged_pos_clean', educ_tab = 'rev_educ_clean'):
    """Faster wide construction via one pass + conditional aggregation."""
    pos_cols = [
        "no_positions", "in_us", "in_home_country", "loc_null",
        "change_company", "change_position", "agg_compensation",
        "still_at_firm", "promoted",
        "n_pos", "n_pos_startafter", "frac_t",
    ]
    educ_cols = [
        "no_educations", "educ_in_us", "educ_in_home_country", "masters", "doctors",
        "new_educ_in_us", "new_educ_in_home_country", "new_masters", "new_doctors", "new_educ",
    ]
    wide_exprs = []
    for t in range(t0, t1 + 1):
        for c in pos_cols + educ_cols:
            # Keep legacy naming convention: <var><t>, e.g. change_company1, no_positions-1
            wide_exprs.append(f"MAX(CASE WHEN t = {t} THEN {c} END) AS \"{c}{t}\"")

    return f"""
    WITH pos AS ({get_rel_year_inds_pos(merge_tab, pos_tab = pos_tab, t0 = t0, t1 = t1)}),
         educ AS ({get_rel_year_inds_educ(merge_tab, educ_tab = educ_tab, t0 = t0, t1 = t1)}),
         joined AS (
             SELECT p.foia_indiv_id, p.user_id, p.t,
                    {", ".join([f"p.{c}" for c in pos_cols])},
                    {", ".join([f"e.{c}" for c in educ_cols])}
             FROM pos AS p
             JOIN educ AS e
               ON p.foia_indiv_id = e.foia_indiv_id
              AND p.user_id = e.user_id
              AND p.t = e.t
         )
    SELECT foia_indiv_id, user_id, {", ".join(wide_exprs)}
    FROM joined
    GROUP BY foia_indiv_id, user_id
    """

def get_rel_year_inds_educ(merge_tab, educ_tab = 'rev_educ_clean', t0 = -1, t1 = 5, rawmerge_tab = None):
    """ Takes output of merge function and a table with user_id x education data and returns table at merge x t level with relevant variables (where t is relative to lottery year)

    Parameters
    -----------
    merge_tab : str
        Name or SQL string representing table that is output of merge functions
    educ_tab: str
        Name or SQL string representing table with user_id x education data
    t0, t1: optional inputs to help.long_by_year

    Returns
    -------
    String representing SQL query for table at merge x t level with relevant education variables
    """

    educlong = f"""
        SELECT *,
            -- indicator for education being in US
                CASE WHEN match_country = 'United States' THEN 1 ELSE 0 END AS educ_in_us,
            -- indicator for education being in country of birth
                CASE WHEN match_country = foia_country THEN 1 ELSE 0 END AS educ_in_home_country,
            -- indicator for masters
                CASE WHEN degree_clean = 'Master' OR degree_clean = 'MBA' THEN 1 ELSE 0 END AS masters,
            -- indicator for doctors
                CASE WHEN degree_clean = 'Doctor' THEN 1 ELSE 0 END AS doctors,
            -- indicator for education having been started before reference year
                CASE WHEN (MIN(t) OVER(PARTITION BY foia_indiv_id, user_id, education_number)) < 0 THEN 1 ELSE 0 END AS start_before,
            -- total number of educations in t
                COUNT(DISTINCT education_number) OVER(PARTITION BY foia_indiv_id, user_id, t) AS n_educ_t
        FROM ({get_long_by_year(merge_tab, educ_tab, long_tab_vars = ', degree_clean, match_country, startdate, enddate, education_number', t0 = t0, t1 = t1, enddatenull = "(CASE WHEN degree_clean = 'Master' OR degree_clean = 'MBA' THEN SUBSTRING(startdate, 1, 4)::INT + 2 ELSE SUBSTRING(startdate, 1, 4)::INT + 4 END)", rawmerge_tab = rawmerge_tab)})
        ORDER BY foia_indiv_id, user_id, t 
    """

    educgroup = f""" 
    SELECT 
        foia_indiv_id, user_id, t,
        CASE WHEN COUNT(DISTINCT education_number) = 0 THEN 1 ELSE 0 END AS no_educations,
        MAX(educ_in_us) AS educ_in_us, MAX(educ_in_home_country) AS educ_in_home_country,
        MAX(masters) AS masters, MAX(doctors) AS doctors,
        MAX(CASE WHEN start_before = 0 AND educ_in_us = 1 THEN 1 ELSE 0 END) AS new_educ_in_us,
        MAX(CASE WHEN start_before = 0 AND educ_in_home_country = 1 THEN 1 ELSE 0 END) AS new_educ_in_home_country, 
        MAX(CASE WHEN start_before = 0 AND masters = 1 THEN 1 ELSE 0 END) AS new_masters, 
        MAX(CASE WHEN start_before = 0 AND doctors = 1 THEN 1 ELSE 0 END) AS new_doctors,
        MAX(CASE WHEN start_before = 0 THEN 1 ELSE 0 END) AS new_educ
    FROM ({educlong}) WHERE t IS NOT NULL
    GROUP BY foia_indiv_id, user_id, t"""

    return educgroup


def get_rel_year_inds_pos(merge_tab, pos_tab = 'merged_pos_clean', t0 = -1, t1 = 5, rcid_lookup_tab = None, rawmerge_tab = None):
    """ Takes output of merge function and a table with user_id x position data and returns table at merge x t level with relevant variables (where t is relative to lottery year)

    Parameters
    -----------
    merge_tab : str
        Name or SQL string representing table that is output of merge functions
    pos_tab: str
        Name or SQL string representing table with user_id x position data
    t0, t1: optional inputs to help.long_by_year

    Returns
    -------
    String representing SQL query for table at merge x t level with relevant position variables
    """

    # enrich pos_tab with foia_firm_uid via rcid lookup in rev_indiv
    _rcid_source = rcid_lookup_tab if rcid_lookup_tab is not None else "(SELECT DISTINCT rcid, foia_firm_uid FROM rev_indiv)"
    enriched_pos_tab = f"""(
        SELECT p.*, r.foia_firm_uid AS pos_foia_firm_uid
        FROM {pos_tab} AS p
        LEFT JOIN {_rcid_source} AS r ON p.rcid = r.rcid
    )"""

    # merging long on position x t
    poslong = f"""
        SELECT *,
        -- indicator for position being in US
            CASE WHEN country = 'United States' THEN 1 ELSE 0 END AS in_us,
        -- indicator for position being null
            CASE WHEN country IS NULL THEN 1 ELSE 0 END AS loc_null,
        -- indicator for position being in country of birth
            CASE WHEN country = foia_country THEN 1 ELSE 0 END AS in_home_country,
        -- indicator for being at same company as matched on in lottery
            CASE WHEN pos_foia_firm_uid = ref_foia_firm_uid THEN 1 ELSE 0 END AS same_company,
        -- creating reference position number variable (first when all positions with t <= 0 ordered by t desc, position number asc)
            MAX(CASE WHEN ref_pos_priority = 1 THEN position_number ELSE 0 END) OVER(PARTITION BY foia_indiv_id, user_id) AS ref_position_number,
        -- indicator for still being at reference position
            CASE WHEN position_number = (MAX(CASE WHEN ref_pos_priority = 1 THEN position_number ELSE 0 END) OVER(PARTITION BY foia_indiv_id, user_id)) THEN 1 ELSE 0 END AS same_position,
        -- indicator for position being started before reference position
            CASE WHEN position_number < (MAX(CASE WHEN ref_pos_priority = 1 THEN position_number ELSE 0 END) OVER(PARTITION BY foia_indiv_id, user_id)) THEN 1 ELSE 0 END AS start_before,
        -- imputed total compensation of reference position
            CASE WHEN position_number = (MAX(CASE WHEN ref_pos_priority = 1 THEN position_number ELSE 0 END) OVER(PARTITION BY foia_indiv_id, user_id)) THEN total_compensation ELSE 0 END AS ref_comp,
        -- total number of positions in t
            COUNT(DISTINCT position_id) OVER(PARTITION BY foia_indiv_id, user_id, t) AS n_pos_t,
        -- first relative year this position appears (identifies post-lottery new positions)
            MIN(t) OVER(PARTITION BY foia_indiv_id, user_id, position_number) AS pos_start_t
        FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY foia_indiv_id, user_id ORDER BY (CASE WHEN t <= 0 THEN 1 ELSE 0 END) DESC, t DESC, position_number) AS ref_pos_priority FROM ({
            get_long_by_year(merge_tab, enriched_pos_tab, long_tab_vars = ', position_id, position_number, b.pos_foia_firm_uid, startdate, enddate, title_raw, company_raw, country, total_compensation', t0 = t0, t1 = t1, rawmerge_tab = rawmerge_tab)}))
        ORDER BY foia_indiv_id, user_id, t
    """

    # Filtering and grouping by t
    posgroup = f""" 
    SELECT 
        foia_indiv_id, user_id, t,
        CASE WHEN COUNT(DISTINCT position_number) = 0 THEN 1 ELSE 0 END AS no_positions,
        MAX(in_us) AS in_us, MAX(in_home_country) AS in_home_country,
        MAX(loc_null) AS loc_null,
        -- at_diff_firm: started a new position at a DIFFERENT firm post-lottery (pos_start_t >= 1)
        MAX(CASE WHEN pos_start_t >= 1 AND same_company = 0 THEN 1 ELSE 0 END) AS change_company,
        MAX(CASE WHEN start_before = 0 AND same_position = 0 THEN 1 ELSE 0 END) AS change_position,
        -- still_at_firm: has any active position at same foia_firm_uid in this year (incl. original position)
        MAX(CASE WHEN same_company = 1 THEN 1 ELSE 0 END) AS still_at_firm,
        -- promoted: new position at same firm started post-lottery (pos_start_t >= 1), different role
        MAX(CASE WHEN pos_start_t >= 1 AND same_company = 1 AND same_position = 0 THEN 1 ELSE 0 END) AS promoted,
        SUM(total_compensation * frac_t) AS agg_compensation,
        COUNT(*) AS n_pos,
        COUNT(CASE WHEN start_before = 0 THEN 1 END) AS n_pos_startafter,
        SUM(frac_t) AS frac_t,
        -- Non-updating bias diagnostics:
        -- Fraction of active positions in year t with null enddate (still-active from imputation, not observed enddate)
        AVG(CASE WHEN enddate IS NULL THEN 1.0 ELSE 0.0 END) AS frac_null_enddate,
        -- Binary: at original firm ONLY due to null-enddate imputation (no new post-lottery position added at that firm)
        MAX(CASE WHEN same_company = 1 AND enddate IS NULL AND pos_start_t < 1
                      AND has_new_pos_same_firm = 0
                 THEN 1 ELSE 0 END) AS null_enddate_stayer
    FROM (
        SELECT *,
            COUNT(CASE WHEN start_before = 0 THEN 1 ELSE NULL END) OVER(PARTITION BY foia_indiv_id, user_id, t) AS n_start_after,
            -- has_new_pos_same_firm: 1 if any post-lottery position at the same FOIA firm exists in this year
            MAX(CASE WHEN same_company = 1 AND pos_start_t >= 1 THEN 1 ELSE 0 END) OVER(PARTITION BY foia_indiv_id, user_id, t) AS has_new_pos_same_firm
        FROM ({poslong})
        ) 
    WHERE t IS NOT NULL AND (start_before = 0 OR n_start_after = 0)
    GROUP BY foia_indiv_id, user_id, t"""

    return posgroup
    

def _build_rawmerge_query(merge_tab, long_tab, long_tab_vars):
    """Return the SQL for the base rawmerge join (merge × events), used both inline
    and for pre-materialization before the expensive long_by_year expansion."""
    return f"""
        SELECT foia_indiv_id, a.user_id, foia_country,
            lottery_year::INT - 1 AS ref_year, a.foia_firm_uid AS ref_foia_firm_uid {long_tab_vars}
        FROM {merge_tab} AS a LEFT JOIN {long_tab} AS b ON a.user_id = b.user_id
    """


def _pos_rawmerge_query(merge_tab, pos_tab, rcid_lookup_tab = None):
    """Rawmerge SQL for positions (includes pos_foia_firm_uid enrichment).
    Pre-materialize this before calling get_rel_year_inds_pos with rawmerge_tab."""
    _rcid_source = rcid_lookup_tab if rcid_lookup_tab is not None else "(SELECT DISTINCT rcid, foia_firm_uid FROM rev_indiv)"
    enriched_pos = f"(SELECT p.*, r.foia_firm_uid AS pos_foia_firm_uid FROM {pos_tab} AS p LEFT JOIN {_rcid_source} AS r ON p.rcid = r.rcid)"
    return _build_rawmerge_query(
        merge_tab, enriched_pos,
        ', position_id, position_number, b.pos_foia_firm_uid, startdate, enddate, title_raw, company_raw, country, total_compensation',
    )


def _educ_rawmerge_query(merge_tab, educ_tab):
    """Rawmerge SQL for education.
    Pre-materialize this before calling get_rel_year_inds_educ with rawmerge_tab."""
    return _build_rawmerge_query(
        merge_tab, educ_tab,
        ', degree_clean, match_country, startdate, enddate, education_number',
    )


def get_long_by_year(merge_tab, long_tab, long_tab_vars, t0 = -1, t1 = 5, enddatenull = CURRENT_YEAR, rawmerge_tab = None):
    """ Takes output of merge function and a table long on user_id x event and returns SQL string for table long on merge x event x t where t is relative to lottery year

    Parameters
    -----------
    merge_tab : str
        Name or SQL string representing table that is output of merge functions
    long_tab: str
        Name or SQL string representing table long on user_id x event where event has start and end date (e.g. position, education)
    long_tab_vars: str
        Additional vars from long_tab to keep for future steps (must start with comma)
    t0, t1, enddatenull: optional inputs to help.long_by_year
    rawmerge_tab : str, optional
        Pre-materialized rawmerge table name.  When provided, long_tab / long_tab_vars
        are ignored and merge_tab is used only for the time-grid distinct-id lookup.

    Returns
    -------
    String representing SQL query for table long on merge x event x t
    """

    if rawmerge_tab is not None:
        return help.long_by_year(
            tab = rawmerge_tab, t0 = t0, t1 = t1, t_ref = 'x.ref_year',
            enddatenull = enddatenull, joinids = 'user_id, foia_indiv_id',
            distinct_ids_tab = merge_tab,
        )

    rawmerge = _build_rawmerge_query(merge_tab, long_tab, long_tab_vars)
    return help.long_by_year(tab = f'({rawmerge})', t0 = t0, t1 = t1, t_ref = 'x.ref_year', enddatenull = enddatenull, joinids = 'user_id, foia_indiv_id')


def _sql_escape_path(path):
    return path.replace("'", "''")


def write_query_to_parquet(query, out_path, overwrite = False, con = con_indiv):
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(out_path), exist_ok = True)

    if os.path.exists(out_path):
        if os.path.getsize(out_path) == 0:
            print(f"Found empty file, rebuilding: {out_path}")
            os.remove(out_path)
        elif not overwrite:
            print(f"Skipping existing file: {out_path}")
            return
        else:
            os.remove(out_path)
    escaped_path = _sql_escape_path(out_path)
    con.sql(f"COPY ({query}) TO '{escaped_path}' (FORMAT parquet)")
    elapsed = time.perf_counter() - t0
    print(f"Wrote: {out_path} ({_fmt_elapsed(elapsed)})")


def _apply_optimal_dedup(con, weighted_table, out_table = "_optimal_dedup_pairs"):
    """Maximum weight bipartite matching for firm-year-user deduplication.

    For each lottery_year, finds the 1:1 assignment (user_id <-> foia_indiv_id) that
    maximizes the sum of total_score over all matched pairs, subject to:
      - Each user_id is matched to at most one foia_indiv_id per year
      - Each foia_indiv_id is matched to at most one user_id per year (new constraint)

    Algorithm: scipy.optimize.linear_sum_assignment (LAPJV, Jonker–Volgenant 1987)
    applied per connected component within each year. The bipartite CC decomposition
    decomposes the graph into independent subproblems, making this tractable even at
    scale — most components are tiny since candidates are clustered within firms.

    References:
      - Kuhn (1955) / Munkres (1957): Hungarian algorithm
      - Jonker & Volgenant (1987): LAPJV, O(n^2 * m), used by scipy

    Args:
        con: DuckDB connection
        weighted_table: name of a materialized DuckDB table containing at minimum
            columns (lottery_year, foia_indiv_id, user_id, total_score)
        out_table: name of the DuckDB table to create with winning pairs

    Creates DuckDB table `out_table` with columns (lottery_year, foia_indiv_id, user_id).
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    t0 = time.perf_counter()

    # Aggregate to unique triplets (take max score if multiple rows per triplet)
    df = con.sql(f"""
        SELECT lottery_year, foia_indiv_id, user_id, MAX(total_score) AS total_score
        FROM {weighted_table}
        GROUP BY lottery_year, foia_indiv_id, user_id
    """).df()

    n_years = df["lottery_year"].nunique()
    print(f"  Optimal dedup: {len(df):,} candidate (year, app, user) triplets across {n_years} years")

    winning_pairs = []

    for year, grp in df.groupby("lottery_year"):
        grp = grp.reset_index(drop=True)

        # Build integer indices for apps and users in this year
        unique_apps = grp["foia_indiv_id"].unique()
        unique_users = grp["user_id"].unique()
        app_to_idx = {a: i for i, a in enumerate(unique_apps)}
        user_to_idx = {u: i for i, u in enumerate(unique_users)}
        n_apps = len(unique_apps)
        n_users = len(unique_users)

        app_idxs = grp["foia_indiv_id"].map(app_to_idx).values
        user_idxs = grp["user_id"].map(user_to_idx).values
        scores = grp["total_score"].values

        # Build bipartite adjacency for connected component decomposition.
        # Represent as (n_apps + n_users) x (n_apps + n_users) graph:
        #   Apps occupy nodes 0..n_apps-1, Users occupy nodes n_apps..n_apps+n_users-1
        n_total = n_apps + n_users
        bi_rows = np.concatenate([app_idxs, user_idxs + n_apps])
        bi_cols = np.concatenate([user_idxs + n_apps, app_idxs])
        adj = csr_matrix(
            (np.ones(len(bi_rows), dtype=np.float32), (bi_rows, bi_cols)),
            shape=(n_total, n_total),
        )
        n_comp, labels = connected_components(adj, directed=False)

        n_matched = 0
        for comp_id in range(n_comp):
            comp_nodes = np.where(labels == comp_id)[0]
            comp_app_nodes = comp_nodes[comp_nodes < n_apps]
            comp_user_nodes = comp_nodes[comp_nodes >= n_apps] - n_apps

            if len(comp_app_nodes) == 0 or len(comp_user_nodes) == 0:
                continue

            # Find all edges in this component
            mask = np.isin(app_idxs, comp_app_nodes) & np.isin(user_idxs, comp_user_nodes)
            nc_a = len(comp_app_nodes)
            nc_u = len(comp_user_nodes)
            app_local = {a: i for i, a in enumerate(comp_app_nodes)}
            user_local = {u: i for i, u in enumerate(comp_user_nodes)}

            # Build dense cost matrix for this component
            cost = np.zeros((nc_a, nc_u), dtype=np.float64)
            for ai, ui, sc in zip(app_idxs[mask], user_idxs[mask], scores[mask]):
                cost[app_local[ai], user_local[ui]] = sc

            # Maximize total score: negate cost matrix for linear_sum_assignment (minimizes)
            # Rectangular matrices OK — unmatched nodes on the larger side get no assignment
            row_ind, col_ind = linear_sum_assignment(-cost)

            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] > 0:
                    winning_pairs.append((
                        year,
                        unique_apps[comp_app_nodes[ri]],
                        unique_users[comp_user_nodes[ci]],
                    ))
                    n_matched += 1

        print(f"    Year {year}: {len(grp):,} edges, {n_comp:,} components → {n_matched:,} matched pairs")

    # Register result as DuckDB table
    pairs_df = pd.DataFrame(winning_pairs, columns=["lottery_year", "foia_indiv_id", "user_id"])
    con.register("_optimal_dedup_pairs_tmp", pairs_df)
    con.sql(f"CREATE OR REPLACE TABLE {out_table} AS SELECT * FROM _optimal_dedup_pairs_tmp")
    con.unregister("_optimal_dedup_pairs_tmp")

    elapsed = time.perf_counter() - t0
    print(f"  Optimal dedup complete: {len(pairs_df):,} winning pairs ({_fmt_elapsed(elapsed)})")


def materialize_table(table_name, query, con = con_indiv):
    t0 = time.perf_counter()
    con.sql(f"CREATE OR REPLACE TABLE {table_name} AS {query}")
    n = con.sql(f"SELECT COUNT(*) AS n FROM {table_name}").df().iloc[0, 0]
    elapsed = time.perf_counter() - t0
    print(f"Materialized {table_name}: {n:,} rows ({_fmt_elapsed(elapsed)})")


def _tvars_pivot_query(pos_long_tab, educ_long_tab, t0 = -1, t1 = 2):
    """Build the final pivot from already-materialized pos/educ long tables.

    Splits the expensive part of get_rel_year_inds_wide_by_t into a cheap
    final step so pos and educ can be materialized separately first.
    """
    pos_cols = [
        "no_positions", "in_us", "in_home_country", "loc_null",
        "change_company", "change_position", "agg_compensation",
        "still_at_firm", "promoted",
        "n_pos", "n_pos_startafter", "frac_t",
        # Non-updating bias diagnostics:
        "frac_null_enddate",   # fraction of active positions with null enddate (still-active from imputation)
        "null_enddate_stayer", # at original firm only due to null-enddate imputation
    ]
    educ_cols = [
        "no_educations", "educ_in_us", "educ_in_home_country", "masters", "doctors",
        "new_educ_in_us", "new_educ_in_home_country", "new_masters", "new_doctors", "new_educ",
    ]
    wide_exprs = []
    for t in range(t0, t1 + 1):
        for c in pos_cols + educ_cols:
            wide_exprs.append(f"MAX(CASE WHEN t = {t} THEN {c} END) AS \"{c}{t}\"")

    return f"""
    WITH joined AS (
        SELECT p.foia_indiv_id, p.user_id, p.t,
               {", ".join([f"p.{c}" for c in pos_cols])},
               {", ".join([f"e.{c}" for c in educ_cols])}
        FROM {pos_long_tab} AS p
        JOIN {educ_long_tab} AS e
          ON p.foia_indiv_id = e.foia_indiv_id
         AND p.user_id = e.user_id
         AND p.t = e.t
    )
    SELECT foia_indiv_id, user_id, {", ".join(wide_exprs)}
    FROM joined
    GROUP BY foia_indiv_id, user_id
    """


def with_t_vars_query(base_table, t0 = -1, t1 = 2, pos_tab = 'merged_pos_clean', educ_tab = 'rev_educ_clean'):
    return f"""
    SELECT * EXCLUDE (b.foia_indiv_id, b.user_id)
    FROM {base_table} AS a
    LEFT JOIN ({get_rel_year_inds_wide_by_t(base_table, t0=t0, t1=t1, pos_tab = pos_tab, educ_tab = educ_tab)}) AS b
        ON a.foia_indiv_id = b.foia_indiv_id AND a.user_id = b.user_id
    """


def with_t_vars_from_table_query(base_table, tvars_table):
    return f"""
    SELECT * EXCLUDE (b.foia_indiv_id, b.user_id)
    FROM {base_table} AS a
    LEFT JOIN {tvars_table} AS b
        ON a.foia_indiv_id = b.foia_indiv_id AND a.user_id = b.user_id
    """


def build_reg_inputs(
    overwrite = None,
    testing = None,
    test_sample_matches = None,
    test_random_seed = None,
    test_firm_uid = None,
    test_lottery_year = None,
    test_materialize_intermediate_tables = None,
    test_table_prefix = None,
    strict_only = False,
    con = con_indiv,
):
    """Builds merge outputs consumed by 04_analysis/reg.py."""
    pipeline_t0 = time.perf_counter()
    print(f"Using config: {icfg.ACTIVE_CONFIG_PATH}")

    if overwrite is None:
        overwrite = icfg.BUILD_OVERWRITE
    if testing is None:
        testing = icfg.TESTING_ENABLED
    if test_sample_matches is None:
        test_sample_matches = icfg.TESTING_SAMPLE_MATCHES
    if test_random_seed is None:
        test_random_seed = icfg.TESTING_RANDOM_SEED
    if test_firm_uid is None:
        test_firm_uid = icfg.TESTING_FIRM_UID
    if test_lottery_year is None:
        test_lottery_year = icfg.TESTING_LOTTERY_YEAR
    if test_materialize_intermediate_tables is None:
        test_materialize_intermediate_tables = icfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES
    if test_table_prefix is None:
        test_table_prefix = icfg.TESTING_TABLE_PREFIX

    if testing:
        print("Testing mode enabled: running merge only on one foia_firm_uid x lottery_year subset.")
        print("No parquet outputs will be written in testing mode.")
        print(
            f"Testing options: sample_matches={test_sample_matches}, "
            f"seed={test_random_seed}, firm_uid={test_firm_uid}, lottery_year={test_lottery_year}, "
            f"materialize_intermediate_tables={test_materialize_intermediate_tables}, "
            f"table_prefix={test_table_prefix}"
        )
        test_result_df = merge_df(
            with_t_vars = False,
            verbose = True,
            testing = True,
            test_sample_matches = test_sample_matches,
            test_random_seed = test_random_seed,
            test_firm_uid = test_firm_uid,
            test_lottery_year = test_lottery_year,
            test_materialize_intermediate_tables = test_materialize_intermediate_tables,
            test_table_prefix = test_table_prefix,
            con = con,
        )
        _print_testing_spotcheck(test_result_df, con=con)
        print(f"Total testing runtime: {_fmt_elapsed(time.perf_counter() - pipeline_t0)}")
        return

    if strict_only:
        # Load the already-built baseline parquet (which has t-vars pre-joined) and apply strict filter.
        # This avoids re-running the full merge pipeline.
        baseline_path = icfg.choose_path(icfg.MERGE_FILT_BASELINE_PARQUET, icfg.MERGE_FILT_BASELINE_PARQUET_LEGACY)
        print(f"strict_only mode: loading baseline from {baseline_path}")
        con.execute(f"CREATE OR REPLACE TABLE _merge_baseline_base AS SELECT * FROM read_parquet('{baseline_path}')")
        strict_sql = _build_stage_strict_query("_merge_baseline_base")
        materialize_table("_merge_strict_base", strict_sql, con=con)
        n_strict = con.execute("SELECT COUNT(*), COUNT(DISTINCT foia_indiv_id) FROM _merge_strict_base").fetchone()
        n_baseline = con.execute("SELECT COUNT(DISTINCT foia_indiv_id) FROM _merge_baseline_base WHERE match_rank = 1").fetchone()
        pct = 100.0 * n_strict[1] / n_baseline[0] if n_baseline[0] > 0 else 0.0
        print(f"  Strict sample: {n_strict[1]:,} apps ({pct:.1f}% of baseline rank-1), {n_strict[0]:,} rows")
        # Baseline parquet already has t-vars joined; save strict rows directly (no second t-vars join)
        write_query_to_parquet("SELECT * FROM _merge_strict_base", icfg.MERGE_FILT_STRICT_PARQUET, overwrite=overwrite, con=con)
        print(f"Total strict_only runtime: {_fmt_elapsed(time.perf_counter() - pipeline_t0)}")
        return

    prefilt = (icfg.BUILD_PREFILT_SQL or "").strip()
    if prefilt:
        print(f"Using configured prefilter: {prefilt}")
    else:
        print("No prefilter configured; prefilt output will mirror baseline.")

    country_score_cutoff = 0.03
    country_fallback_topn = 3
    country_total_margin = 0.08
    month_buffer = 0  # deprecated: date filters are now year-level; kept for API compatibility
    yob_buffer = 5
    f_prob_buffer = 0.8
    rev_mult_coeff = 1
    no_country_min_subregion = icfg.BUILD_NO_COUNTRY_MIN_SUBREGION_SCORE
    no_country_min_total = icfg.BUILD_NO_COUNTRY_MIN_TOTAL_SCORE
    no_country_min_f_score_if_yob_null = icfg.BUILD_NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL
    ambiguity_weight_gap_cutoff = icfg.BUILD_AMBIGUITY_WEIGHT_GAP_CUTOFF
    bad_match_guard_enabled = icfg.BUILD_BAD_MATCH_GUARD_ENABLED
    bad_match_guard_subregion_lt = icfg.BUILD_BAD_MATCH_GUARD_SUBREGION_SCORE_LT
    bad_match_guard_f_score_lt = icfg.BUILD_BAD_MATCH_GUARD_F_SCORE_LT
    bad_match_guard_total_score_lt = icfg.BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT
    firm_year_user_dedup = icfg.BUILD_FIRM_YEAR_USER_DEDUP
    firm_year_user_dedup_optimal = icfg.BUILD_FIRM_YEAR_USER_DEDUP_OPTIMAL
    alpha = icfg.BUILD_SUBREGION_BOOST_ALPHA
    w_country = icfg.BUILD_W_COUNTRY
    w_yob = icfg.BUILD_W_YOB
    w_gender = icfg.BUILD_W_GENDER
    w_occ = icfg.BUILD_W_OCC
    occ_score_halflife = icfg.BUILD_OCC_SCORE_HALFLIFE
    foia_occ_rank_cutoff = icfg.BUILD_FOIA_OCC_RANK_CUTOFF
    multiplicative_score = icfg.BUILD_MULTIPLICATIVE_SCORE
    competition_weight = icfg.BUILD_COUNTRY_COMPETITION_WEIGHT
    competition_threshold = icfg.BUILD_COUNTRY_COMPETITION_THRESHOLD
    print(f"firm_year_user_dedup: {firm_year_user_dedup}, optimal: {firm_year_user_dedup_optimal}")
    print(
        "Merge quality controls: "
        f"no_country_min_subregion={no_country_min_subregion}, "
        f"no_country_min_total={no_country_min_total}, "
        f"no_country_min_f_score_if_est_yob_null={no_country_min_f_score_if_yob_null}, "
        f"bad_match_guard_enabled={bad_match_guard_enabled}, "
        f"ambiguity_weight_gap_cutoff={ambiguity_weight_gap_cutoff}"
    )
    print(f"Occupation score: w_occ={w_occ}, occ_score_halflife={occ_score_halflife}, foia_occ_rank_cutoff={foia_occ_rank_cutoff}")
    print(f"Score mode: {'multiplicative (weighted geometric mean)' if multiplicative_score else 'additive (weighted sum)'}")
    print(f"Country competition: weight={competition_weight}, threshold={competition_threshold}")

    print("Building shared raw/base stages...")
    materialize_table(
        "_merge_raw_base",
        merge_raw_func("rev_indiv", "foia_indiv", foia_prefilt = "", subregion = True, ALPHA = alpha, COMPETITION_WEIGHT = competition_weight, COMPETITION_THRESHOLD = competition_threshold),
        con = con,
    )
    base_stage = _build_merge_filt_stage_queries(
        "_merge_raw_base",
        postfilt = "none",
        MATCH_MULT_CUTOFF = 4,
        REV_MULT_COEFF = rev_mult_coeff,
        COUNTRY_SCORE_CUTOFF = country_score_cutoff,
        COUNTRY_FALLBACK_TOPN = country_fallback_topn,
        COUNTRY_TOTAL_MARGIN = country_total_margin,
        MONTH_BUFFER = month_buffer,
        YOB_BUFFER = yob_buffer,
        F_PROB_BUFFER = f_prob_buffer,
        firm_year_user_dedup = firm_year_user_dedup,
        NO_COUNTRY_MIN_SUBREGION_SCORE = no_country_min_subregion,
        NO_COUNTRY_MIN_TOTAL_SCORE = no_country_min_total,
        NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL = no_country_min_f_score_if_yob_null,
        AMBIGUITY_WEIGHT_GAP_CUTOFF = ambiguity_weight_gap_cutoff,
        BAD_MATCH_GUARD_ENABLED = bad_match_guard_enabled,
        BAD_MATCH_GUARD_SUBREGION_SCORE_LT = bad_match_guard_subregion_lt,
        BAD_MATCH_GUARD_F_SCORE_LT = bad_match_guard_f_score_lt,
        BAD_MATCH_GUARD_TOTAL_SCORE_LT = bad_match_guard_total_score_lt,
        W_COUNTRY = w_country,
        W_YOB = w_yob,
        W_GENDER = w_gender,
        W_OCC = w_occ,
        OCC_SCORE_HALFLIFE = occ_score_halflife,
        FOIA_OCC_RANK_CUTOFF = foia_occ_rank_cutoff,
        MULTIPLICATIVE_SCORE = multiplicative_score,
    )
    materialize_table("_merge_match_filt_base", base_stage["match_filt"], con = con)

    # For optimal dedup: compute scores on the full candidate set, run maximum weight
    # bipartite matching, then pre-filter match_filt to winning pairs only. All downstream
    # stages then use the pre-filtered table with firm_year_user_dedup_sql=False (the SQL
    # QUALIFY clause is no longer needed since dedup was already applied in Python).
    match_filt_tab_base = "_merge_match_filt_base"
    firm_year_user_dedup_sql = firm_year_user_dedup
    if firm_year_user_dedup and firm_year_user_dedup_optimal:
        print("Computing optimal dedup for base branch (maximum weight bipartite matching)...")
        _tmp_weighted_q = _build_stage_weighted_query(
            "SELECT * FROM _merge_match_filt_base",
            firm_year_user_dedup = False,
            YOB_BUFFER = yob_buffer,
            F_PROB_BUFFER = f_prob_buffer,
            W_COUNTRY = w_country,
            W_YOB = w_yob,
            W_GENDER = w_gender,
            W_OCC = w_occ,
            OCC_SCORE_HALFLIFE = occ_score_halflife,
            MULTIPLICATIVE_SCORE = multiplicative_score,
        )
        materialize_table("_merge_weighted_tmp", _tmp_weighted_q, con = con)
        _apply_optimal_dedup(con, "_merge_weighted_tmp", out_table = "_optimal_dedup_pairs_base")
        # Save slim pre-dedup edge table so spotcheck_matches.py --od can inspect networks post-hoc
        if icfg.OPTIMAL_DEDUP_PREDEDUP_PARQUET:
            write_query_to_parquet("""
                SELECT
                    w.lottery_year, w.foia_indiv_id, w.user_id,
                    MAX(w.total_score)          AS total_score,
                    MAX(w.weight_norm)          AS weight_norm,
                    ANY_VALUE(w.foia_firm_uid)  AS foia_firm_uid,
                    ANY_VALUE(w.foia_country)   AS foia_country,
                    ANY_VALUE(w.yob)            AS yob,
                    ANY_VALUE(w.fullname)       AS fullname,
                    ANY_VALUE(w.est_yob)        AS est_yob,
                    ANY_VALUE(w.rev_country)    AS rev_country,
                    MAX(CASE WHEN p.foia_indiv_id IS NOT NULL THEN 1 ELSE 0 END) AS is_selected
                FROM _merge_weighted_tmp w
                LEFT JOIN _optimal_dedup_pairs_base p
                    USING (lottery_year, foia_indiv_id, user_id)
                GROUP BY w.lottery_year, w.foia_indiv_id, w.user_id
            """, icfg.OPTIMAL_DEDUP_PREDEDUP_PARQUET, overwrite=True, con=con)
            print(f"  Saved pre-dedup edges to {icfg.OPTIMAL_DEDUP_PREDEDUP_PARQUET}")
        materialize_table(
            "_merge_match_filt_deduped",
            "SELECT m.* FROM _merge_match_filt_base m JOIN _optimal_dedup_pairs_base p USING (lottery_year, foia_indiv_id, user_id)",
            con = con,
        )
        con.sql("DROP TABLE IF EXISTS _merge_weighted_tmp")
        match_filt_tab_base = "_merge_match_filt_deduped"
        firm_year_user_dedup_sql = False  # SQL QUALIFY dedup already applied via pre-filter

    materialize_table(
        "_merge_baseline_base",
        _build_stage_queries_from_match_filt(
            match_filt_tab_base,
            postfilt = "none",
            MATCH_MULT_CUTOFF = 4,
            REV_MULT_COEFF = rev_mult_coeff,
            YOB_BUFFER = yob_buffer,
            F_PROB_BUFFER = f_prob_buffer,
            firm_year_user_dedup = firm_year_user_dedup_sql,
            W_COUNTRY = w_country,
            W_YOB = w_yob,
            W_GENDER = w_gender,
            W_OCC = w_occ,
            OCC_SCORE_HALFLIFE = occ_score_halflife,
            MULTIPLICATIVE_SCORE = multiplicative_score,
            AMBIGUITY_WEIGHT_GAP_CUTOFF = ambiguity_weight_gap_cutoff,
            BAD_MATCH_GUARD_ENABLED = bad_match_guard_enabled,
            BAD_MATCH_GUARD_SUBREGION_SCORE_LT = bad_match_guard_subregion_lt,
            BAD_MATCH_GUARD_F_SCORE_LT = bad_match_guard_f_score_lt,
            BAD_MATCH_GUARD_TOTAL_SCORE_LT = bad_match_guard_total_score_lt,
        )["final"],
        con = con,
    )

    if prefilt:
        print("Building prefilt base with one extra raw pass...")
        materialize_table(
            "_merge_raw_prefilt",
            merge_raw_func("rev_indiv", "foia_indiv", foia_prefilt = prefilt, subregion = True, ALPHA = alpha, COMPETITION_WEIGHT = competition_weight, COMPETITION_THRESHOLD = competition_threshold),
            con = con,
        )
        prefilt_stage = _build_merge_filt_stage_queries(
            "_merge_raw_prefilt",
            postfilt = "none",
            MATCH_MULT_CUTOFF = 4,
            REV_MULT_COEFF = rev_mult_coeff,
            COUNTRY_SCORE_CUTOFF = country_score_cutoff,
            COUNTRY_FALLBACK_TOPN = country_fallback_topn,
            COUNTRY_TOTAL_MARGIN = country_total_margin,
            MONTH_BUFFER = month_buffer,
            YOB_BUFFER = yob_buffer,
            F_PROB_BUFFER = f_prob_buffer,
            firm_year_user_dedup = firm_year_user_dedup,
            NO_COUNTRY_MIN_SUBREGION_SCORE = no_country_min_subregion,
            NO_COUNTRY_MIN_TOTAL_SCORE = no_country_min_total,
            NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL = no_country_min_f_score_if_yob_null,
            AMBIGUITY_WEIGHT_GAP_CUTOFF = ambiguity_weight_gap_cutoff,
            BAD_MATCH_GUARD_ENABLED = bad_match_guard_enabled,
            BAD_MATCH_GUARD_SUBREGION_SCORE_LT = bad_match_guard_subregion_lt,
            BAD_MATCH_GUARD_F_SCORE_LT = bad_match_guard_f_score_lt,
            BAD_MATCH_GUARD_TOTAL_SCORE_LT = bad_match_guard_total_score_lt,
            W_COUNTRY = w_country,
            W_YOB = w_yob,
            W_GENDER = w_gender,
            W_OCC = w_occ,
            OCC_SCORE_HALFLIFE = occ_score_halflife,
            FOIA_OCC_RANK_CUTOFF = foia_occ_rank_cutoff,
            MULTIPLICATIVE_SCORE = multiplicative_score,
        )
        materialize_table("_merge_match_filt_prefilt", prefilt_stage["match_filt"], con = con)

        match_filt_tab_prefilt = "_merge_match_filt_prefilt"
        if firm_year_user_dedup and firm_year_user_dedup_optimal:
            print("Computing optimal dedup for prefilt branch...")
            _tmp_weighted_q = _build_stage_weighted_query(
                "SELECT * FROM _merge_match_filt_prefilt",
                firm_year_user_dedup = False,
                YOB_BUFFER = yob_buffer,
                F_PROB_BUFFER = f_prob_buffer,
                W_COUNTRY = w_country,
                W_YOB = w_yob,
                W_GENDER = w_gender,
                W_OCC = w_occ,
                OCC_SCORE_HALFLIFE = occ_score_halflife,
                MULTIPLICATIVE_SCORE = multiplicative_score,
            )
            materialize_table("_merge_weighted_tmp", _tmp_weighted_q, con = con)
            _apply_optimal_dedup(con, "_merge_weighted_tmp", out_table = "_optimal_dedup_pairs_prefilt")
            materialize_table(
                "_merge_match_filt_prefilt_deduped",
                "SELECT m.* FROM _merge_match_filt_prefilt m JOIN _optimal_dedup_pairs_prefilt p USING (lottery_year, foia_indiv_id, user_id)",
                con = con,
            )
            con.sql("DROP TABLE IF EXISTS _merge_weighted_tmp")
            match_filt_tab_prefilt = "_merge_match_filt_prefilt_deduped"

        materialize_table(
            "_merge_prefilt_base",
            _build_stage_queries_from_match_filt(
                match_filt_tab_prefilt,
                postfilt = "none",
                MATCH_MULT_CUTOFF = 4,
                REV_MULT_COEFF = rev_mult_coeff,
                YOB_BUFFER = yob_buffer,
                F_PROB_BUFFER = f_prob_buffer,
                firm_year_user_dedup = firm_year_user_dedup_sql,
                W_COUNTRY = w_country,
                W_YOB = w_yob,
                W_GENDER = w_gender,
                W_OCC = w_occ,
                OCC_SCORE_HALFLIFE = occ_score_halflife,
                MULTIPLICATIVE_SCORE = multiplicative_score,
                AMBIGUITY_WEIGHT_GAP_CUTOFF = ambiguity_weight_gap_cutoff,
                BAD_MATCH_GUARD_ENABLED = bad_match_guard_enabled,
                BAD_MATCH_GUARD_SUBREGION_SCORE_LT = bad_match_guard_subregion_lt,
                BAD_MATCH_GUARD_F_SCORE_LT = bad_match_guard_f_score_lt,
                BAD_MATCH_GUARD_TOTAL_SCORE_LT = bad_match_guard_total_score_lt,
            )["final"],
            con = con,
        )
    else:
        materialize_table("_merge_prefilt_base", "SELECT * FROM _merge_baseline_base", con = con)

    for cutoff in (2, 4, 6):
        print(f"Building mult{cutoff} base from shared stage...")
        materialize_table(
            f"_merge_mult{cutoff}_base",
            _build_stage_queries_from_match_filt(
                match_filt_tab_base,
                postfilt = "indiv",
                MATCH_MULT_CUTOFF = cutoff,
                REV_MULT_COEFF = rev_mult_coeff,
                YOB_BUFFER = yob_buffer,
                F_PROB_BUFFER = f_prob_buffer,
                firm_year_user_dedup = firm_year_user_dedup_sql,
                W_COUNTRY = w_country,
                W_YOB = w_yob,
                W_GENDER = w_gender,
                W_OCC = w_occ,
                OCC_SCORE_HALFLIFE = occ_score_halflife,
                MULTIPLICATIVE_SCORE = multiplicative_score,
                AMBIGUITY_WEIGHT_GAP_CUTOFF = ambiguity_weight_gap_cutoff,
                BAD_MATCH_GUARD_ENABLED = bad_match_guard_enabled,
                BAD_MATCH_GUARD_SUBREGION_SCORE_LT = bad_match_guard_subregion_lt,
                BAD_MATCH_GUARD_F_SCORE_LT = bad_match_guard_f_score_lt,
                BAD_MATCH_GUARD_TOTAL_SCORE_LT = bad_match_guard_total_score_lt,
            )["final"],
            con = con,
        )

    print("Materializing baseline user subset...")
    materialize_table("_merge_baseline_users", "SELECT DISTINCT user_id FROM _merge_baseline_base", con = con)
    materialize_table(
        "_merge_pos_subset",
        "SELECT p.* FROM merged_pos_clean AS p JOIN _merge_baseline_users AS u ON p.user_id = u.user_id",
        con = con,
    )
    materialize_table(
        "_merge_educ_subset",
        "SELECT e.* FROM rev_educ_clean AS e JOIN _merge_baseline_users AS u ON e.user_id = u.user_id",
        con = con,
    )

    print("Building shared t-vars table from baseline...")
    materialize_table(
        "_merge_rcid_fuid_lookup",
        "SELECT DISTINCT p.rcid, r.foia_firm_uid FROM _merge_pos_subset AS p LEFT JOIN (SELECT DISTINCT rcid, foia_firm_uid FROM rev_indiv) AS r ON p.rcid = r.rcid",
        con = con,
    )
    materialize_table(
        "_merge_rawmerge_pos",
        _pos_rawmerge_query("_merge_baseline_base", "_merge_pos_subset", rcid_lookup_tab = "_merge_rcid_fuid_lookup"),
        con = con,
    )
    materialize_table(
        "_merge_rawmerge_educ",
        _educ_rawmerge_query("_merge_baseline_base", "_merge_educ_subset"),
        con = con,
    )
    materialize_table(
        "_merge_pos_long",
        get_rel_year_inds_pos("_merge_baseline_base", t0 = -1, t1 = 2, pos_tab = "_merge_pos_subset", rcid_lookup_tab = "_merge_rcid_fuid_lookup", rawmerge_tab = "_merge_rawmerge_pos"),
        con = con,
    )
    materialize_table(
        "_merge_educ_long",
        get_rel_year_inds_educ("_merge_baseline_base", t0 = -1, t1 = 2, educ_tab = "_merge_educ_subset", rawmerge_tab = "_merge_rawmerge_educ"),
        con = con,
    )
    materialize_table(
        "_merge_baseline_tvars",
        _tvars_pivot_query("_merge_pos_long", "_merge_educ_long", t0 = -1, t1 = 2),
        con = con,
    )

    # Build strict sample as post-hoc filter on baseline (rank=1, high weight/score/firm/country thresholds)
    print("Building strict base from baseline...")
    materialize_table("_merge_strict_base", _build_stage_strict_query("_merge_baseline_base"), con=con)
    n_strict = con.execute("SELECT COUNT(*), COUNT(DISTINCT foia_indiv_id) FROM _merge_strict_base").fetchone()
    n_baseline = con.execute("SELECT COUNT(DISTINCT foia_indiv_id) FROM _merge_baseline_base WHERE match_rank = 1").fetchone()
    pct = 100.0 * n_strict[1] / n_baseline[0] if n_baseline[0] > 0 else 0.0
    print(f"  Strict sample: {n_strict[1]:,} apps ({pct:.1f}% of baseline rank-1), {n_strict[0]:,} rows")

    outputs = [
        ("baseline", "_merge_baseline_base", icfg.MERGE_FILT_BASELINE_PARQUET),
        ("prefilt", "_merge_prefilt_base", icfg.MERGE_FILT_PREFILT_PARQUET),
        ("mult2", "_merge_mult2_base", icfg.MERGE_FILT_MULT2_PARQUET),
        ("mult4", "_merge_mult4_base", icfg.MERGE_FILT_MULT4_PARQUET),
        ("mult6", "_merge_mult6_base", icfg.MERGE_FILT_MULT6_PARQUET),
        ("strict", "_merge_strict_base", icfg.MERGE_FILT_STRICT_PARQUET),
    ]
    # Optimal dedup outputs go to separate files with _opt suffix so they don't
    # overwrite the standard (greedy) outputs.
    if firm_year_user_dedup and firm_year_user_dedup_optimal:
        outputs = [(name, tab, path.replace(".parquet", "_opt.parquet")) for name, tab, path in outputs]

    for name, base_table, out_path in outputs:
        print(f"Building {name}...")
        write_query_to_parquet(
            with_t_vars_from_table_query(base_table, "_merge_baseline_tvars"),
            out_path,
            overwrite = overwrite,
            con = con,
        )

    print(f"Total indiv_merge build runtime: {_fmt_elapsed(time.perf_counter() - pipeline_t0)}")


def compare_quality_control_impact(
    merge_path = None,
    country_score_cutoff = 0.03,
    country_fallback_topn = 3,
    country_total_margin = 0.08,
    no_country_min_subregion_score = None,
    no_country_min_total_score = None,
    no_country_min_f_score_if_est_yob_null = None,
    ambiguity_weight_gap_cutoff = None,
    bad_match_guard_enabled = None,
    bad_match_guard_subregion_score_lt = None,
    bad_match_guard_f_score_lt = None,
    bad_match_guard_total_score_lt = None,
    con = con_indiv,
):
    """Compare before/after effects of quality controls on an existing merge parquet."""
    if merge_path is None:
        merge_path = icfg.choose_path(icfg.MERGE_FILT_BASELINE_PARQUET, icfg.MERGE_FILT_BASELINE_PARQUET_LEGACY)
    if no_country_min_subregion_score is None:
        no_country_min_subregion_score = icfg.BUILD_NO_COUNTRY_MIN_SUBREGION_SCORE
    if no_country_min_total_score is None:
        no_country_min_total_score = icfg.BUILD_NO_COUNTRY_MIN_TOTAL_SCORE
    if no_country_min_f_score_if_est_yob_null is None:
        no_country_min_f_score_if_est_yob_null = icfg.BUILD_NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL
    if ambiguity_weight_gap_cutoff is None:
        ambiguity_weight_gap_cutoff = icfg.BUILD_AMBIGUITY_WEIGHT_GAP_CUTOFF
    if bad_match_guard_enabled is None:
        bad_match_guard_enabled = icfg.BUILD_BAD_MATCH_GUARD_ENABLED
    if bad_match_guard_subregion_score_lt is None:
        bad_match_guard_subregion_score_lt = icfg.BUILD_BAD_MATCH_GUARD_SUBREGION_SCORE_LT
    if bad_match_guard_f_score_lt is None:
        bad_match_guard_f_score_lt = icfg.BUILD_BAD_MATCH_GUARD_F_SCORE_LT
    if bad_match_guard_total_score_lt is None:
        bad_match_guard_total_score_lt = icfg.BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT

    guard_expr = "TRUE"
    if bad_match_guard_enabled:
        guard_expr = (
            "NOT ("
            f"country_score = 0 AND subregion_score < {bad_match_guard_subregion_score_lt} "
            f"AND (est_yob IS NULL OR f_score < {bad_match_guard_f_score_lt}) "
            f"AND total_score < {bad_match_guard_total_score_lt}"
            ")"
        )

    work_con = con
    close_after = False
    if con is con_indiv:
        work_con = ddb.connect()
        close_after = True
    work_con.sql("PRAGMA threads=1")
    work_con.sql("PRAGMA preserve_insertion_order=true")

    q = f"""
    WITH src AS (
        SELECT * FROM read_parquet('{_sql_escape_path(merge_path)}')
    ),
    before_ranked AS (
        SELECT *,
            ROW_NUMBER() OVER(
                PARTITION BY foia_indiv_id
                ORDER BY
                    weight_norm DESC,
                    total_score DESC,
                    user_id,
                    rcid,
                    rev_country,
                    subregion,
                    fullname,
                    llm_match_score DESC NULLS LAST,
                    est_yob ASC NULLS LAST
            ) AS rn,
            LEAD(weight_norm) OVER(
                PARTITION BY foia_indiv_id
                ORDER BY
                    weight_norm DESC,
                    total_score DESC,
                    user_id,
                    rcid,
                    rev_country,
                    subregion,
                    fullname,
                    llm_match_score DESC NULLS LAST,
                    est_yob ASC NULLS LAST
            ) AS next_weight
        FROM src
    ),
    before_top AS (
        SELECT *,
            weight_norm - COALESCE(next_weight, 0) AS weight_gap
        FROM before_ranked
        WHERE rn = 1
    ),
    after_candidates_scored AS (
        SELECT *,
            MAX(country_score) OVER(PARTITION BY foia_indiv_id) AS max_country_score_app,
            MAX(total_score) OVER(PARTITION BY foia_indiv_id) AS max_total_score_app,
            ROW_NUMBER() OVER(
                PARTITION BY foia_indiv_id
                ORDER BY
                    country_rank_score DESC,
                    total_score DESC,
                    user_id,
                    rcid,
                    rev_country,
                    subregion,
                    fullname,
                    llm_match_score DESC NULLS LAST,
                    est_yob ASC NULLS LAST
            ) AS app_country_rank
        FROM src
    ),
    after_candidates AS (
        SELECT *
        FROM after_candidates_scored
        WHERE (
                country_score > {country_score_cutoff}
                OR (
                    max_country_score_app <= {country_score_cutoff}
                    AND subregion_score >= {no_country_min_subregion_score}
                    AND total_score >= {no_country_min_total_score}
                    AND (est_yob IS NOT NULL OR f_score >= {no_country_min_f_score_if_est_yob_null})
                )
                OR (
                    COALESCE(country_uncertain_ind, 0) = 1
                    AND (
                        app_country_rank <= {country_fallback_topn}
                        OR total_score >= max_total_score_app - {country_total_margin}
                    )
                )
            )
            AND ({guard_expr})
    ),
    after_ranked AS (
        SELECT *,
            ROW_NUMBER() OVER(
                PARTITION BY foia_indiv_id
                ORDER BY
                    weight_norm DESC,
                    total_score DESC,
                    user_id,
                    rcid,
                    rev_country,
                    subregion,
                    fullname,
                    llm_match_score DESC NULLS LAST,
                    est_yob ASC NULLS LAST
            ) AS rn,
            LEAD(weight_norm) OVER(
                PARTITION BY foia_indiv_id
                ORDER BY
                    weight_norm DESC,
                    total_score DESC,
                    user_id,
                    rcid,
                    rev_country,
                    subregion,
                    fullname,
                    llm_match_score DESC NULLS LAST,
                    est_yob ASC NULLS LAST
            ) AS next_weight
        FROM after_candidates
    ),
    after_top AS (
        SELECT *,
            weight_norm - COALESCE(next_weight, 0) AS weight_gap
        FROM after_ranked
        WHERE rn = 1
    ),
    before_stats AS (
        SELECT
            'before' AS scenario,
            COUNT(*) AS n_apps,
            SUM(CASE WHEN country_score = 0 THEN 1 ELSE 0 END) AS top_country_score_zero,
            SUM(CASE WHEN est_yob IS NULL THEN 1 ELSE 0 END) AS top_est_yob_null,
            SUM(CASE WHEN n_match_filt >= 2 AND weight_gap <= {ambiguity_weight_gap_cutoff} THEN 1 ELSE 0 END) AS top_ambiguous,
            SUM(CASE WHEN {guard_expr} THEN 0 ELSE 1 END) AS top_bad_guard_hits
        FROM before_top
    ),
    after_stats AS (
        SELECT
            'after' AS scenario,
            COUNT(*) AS n_apps,
            SUM(CASE WHEN country_score = 0 THEN 1 ELSE 0 END) AS top_country_score_zero,
            SUM(CASE WHEN est_yob IS NULL THEN 1 ELSE 0 END) AS top_est_yob_null,
            SUM(CASE WHEN n_match_filt >= 2 AND weight_gap <= {ambiguity_weight_gap_cutoff} THEN 1 ELSE 0 END) AS top_ambiguous,
            SUM(CASE WHEN {guard_expr} THEN 0 ELSE 1 END) AS top_bad_guard_hits
        FROM after_top
    )
    SELECT * FROM before_stats
    UNION ALL
    SELECT * FROM after_stats
    """
    out_df = work_con.sql(q).df()
    if close_after:
        work_con.close()
    return out_df


# x = con_indiv.sql(merge(with_t_vars = True, MATCH_MULT_CUTOFF = 2))
# full_df = con.sql(f"COPY ({merge(with_t_vars=True)}) TO '{root}/data/int/merge_filt_base_with_t_vars_aug21.parquet'")

#mergetest = con.sql(merge(postfilt = 'indiv'))

#out = con.sql(get_rel_year_inds_wide('mergetest'))


# mergetest = con.sql(merge(postfilt = 'indiv'))

# out = con.sql(get_rel_year_inds_wide('mergetest'))

# for c in [2,4,6]:
#     con_indiv.sql(f"COPY ({merge(with_t_vars = True, MATCH_MULT_CUTOFF = c, postfilt = 'indiv')}) TO '{root}/data/int/merge_filt_mult{c}_sep8.parquet'")

# # con_indiv.sql(f"COPY ({merge(with_t_vars = True)}) TO '{root}/data/int/merge_filt_baseline_sep8.parquet'")

# #####################
# # DIFFERENT MERGE VERSIONS
# # #####################
# # con.sql(f"COPY ({merge(with_t_vars=True)}) TO '{root}/data/int/merge_filt_base_jul30.parquet'")
# con_indiv.sql(f"COPY ({merge(foia_prefilt = "WHERE subregion != 'Southern Asia' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'China' AND country != 'Taiwan'", with_t_vars = True)}) TO '{root}/data/int/merge_filt_prefilt_sep8.parquet'")
# con.sql(f"COPY ({merge(postfilt='indiv')}) TO '{root}/data/int/merge_filt_postfilt_jul30.parquet'")





# con.sql(f"""CREATE OR REPLACE TABLE merge_raw AS {merge_raw_func('rev_indiv', 'foia_indiv')}""")

# con.sql(f"""CREATE OR REPLACE TABLE merge_raw_subregion AS {merge_raw_func('rev_indiv', 'foia_indiv', subregion=True)}""")

# con.sql(f"""CREATE OR REPLACE TABLE merge_raw_prefilt AS {merge_raw_func('rev_indiv', "(SELECT * FROM foia_indiv WHERE country != 'China' AND country != 'India' AND country != 'Taiwan' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'Nepal' AND country != 'Pakistan')")}""")


# con.sql(f"""CREATE OR REPLACE TABLE merge_raw_prefilt_subregion AS {merge_raw_func('rev_indiv', "(SELECT * FROM foia_indiv WHERE country != 'China' AND country != 'India' AND country != 'Taiwan' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'Nepal' AND country != 'Pakistan')", subregion = True)}""")

# merge_filt_base = con.sql(merge_filt_func('merge_raw'))
# merge_filt_subregion = con.sql(merge_filt_func('merge_raw_subregion'))

# merge_filt_prefilt = con.sql(merge_filt_func('merge_raw_prefilt'))

# merge_filt_prefilt_subregion = con.sql(merge_filt_func('merge_raw_prefilt_subregion'))

# MATCH_MULT_CUTOFF = 4
# REV_MULT_COEFF = 1
# # version 1: filtering on avg match_mult at the firm x year level
# merge_filt_postfilt = con.sql(f'SELECT * FROM ({merge_filt_func('merge_raw')}) WHERE share_apps_matched_emp = 1 AND match_mult_emp <= {MATCH_MULT_CUTOFF} AND rev_mult_emp < {REV_MULT_COEFF}*n_apps_matched_emp  AND n_unique_wintype_emp > 1')
# #print(merge_filt_postfilt.shape)

# # version 2: filtering on match_mult at the foia app level, then filtering on firm stuff (more restrictive)
# merge_filt_postfilt2 = con.sql(f'SELECT * FROM (SELECT (COUNT(DISTINCT foia_temp_id) OVER(PARTITION BY FEIN, lottery_year))/n_apps AS share_apps_matched_emp, (COUNT(*) OVER(PARTITION BY FEIN, lottery_year))/(COUNT(DISTINCT user_id) OVER(PARTITION BY FEIN, lottery_year)) AS rev_mult_emp, COUNT(DISTINCT foia_temp_id) OVER(PARTITION BY FEIN, lottery_year) AS n_apps_matched_emp,COUNT(DISTINCT status_type) OVER(PARTITION BY FEIN, lottery_year) AS n_unique_wintype_emp, * FROM ({merge_filt_func('merge_raw')}) WHERE n_match_filt <= {MATCH_MULT_CUTOFF}) WHERE share_apps_matched_emp = 1 AND rev_mult_emp < {REV_MULT_COEFF}*n_apps_matched_emp  AND n_unique_wintype_emp > 1')
# #print(merge_filt_postfilt2.shape)


# con.sql(f"COPY foia_indiv TO '{root}/data/int/foia_merge_samp_jul23.parquet'")
# con.sql(f"COPY merge_filt_base TO '{root}/data/int/merge_filt_base_jul23.parquet'")
# con.sql(f"COPY merge_filt_postfilt2 TO '{root}/data/int/merge_filt_postfilt_jul23.parquet'")
# con.sql(f"COPY merge_filt_prefilt TO '{root}/data/int/merge_filt_prefilt_jul23.parquet'")

# To regenerate files used by 04_analysis/reg.py, run this script directly.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Build indiv-merge parquet inputs, or run a subset testing/debug mode."
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        dest = "overwrite",
        action = "store_true",
        default = None,
        help = "Override config and force overwriting existing parquet outputs.",
    )
    overwrite_group.add_argument(
        "--no-overwrite",
        dest = "overwrite",
        action = "store_false",
        help = "Override config and skip overwriting existing parquet outputs.",
    )

    testing_group = parser.add_mutually_exclusive_group()
    testing_group.add_argument(
        "--testing",
        dest = "testing",
        action = "store_true",
        default = None,
        help = "Override config and run on one foia_firm_uid x lottery_year subset.",
    )
    testing_group.add_argument(
        "--no-testing",
        dest = "testing",
        action = "store_false",
        help = "Override config and disable testing mode.",
    )
    parser.add_argument(
        "--test-sample-matches",
        type = int,
        default = None,
        help = "Override config: number of readable sample matches to print in testing mode.",
    )
    parser.add_argument(
        "--test-seed",
        type = int,
        default = None,
        help = "Override config: optional seed for deterministic test subset selection.",
    )
    parser.add_argument(
        "--test-firm-uid",
        type = str,
        default = None,
        help = "Override config: explicit foia_firm_uid for testing mode.",
    )
    parser.add_argument(
        "--test-lottery-year",
        type = str,
        default = None,
        help = "Override config: explicit lottery_year for testing mode.",
    )
    materialize_group = parser.add_mutually_exclusive_group()
    materialize_group.add_argument(
        "--test-materialize-intermediate-tables",
        dest = "test_materialize_intermediate_tables",
        action = "store_true",
        default = None,
        help = "Override config: materialize intermediate testing tables in DuckDB.",
    )
    materialize_group.add_argument(
        "--no-test-materialize-intermediate-tables",
        dest = "test_materialize_intermediate_tables",
        action = "store_false",
        help = "Override config: do not materialize intermediate testing tables.",
    )
    parser.add_argument(
        "--test-table-prefix",
        type = str,
        default = None,
        help = "Override config: table prefix for materialized testing tables.",
    )
    parser.add_argument(
        "--strict-only",
        dest = "strict_only",
        action = "store_true",
        default = False,
        help = "Skip the full merge pipeline: load existing baseline parquet and output only the strict parquet.",
    )

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        # Jupyter/ipykernel injects argv like "-f <kernel.json>".
        # Ignore those only in notebook contexts; keep strict parsing otherwise.
        if "ipykernel" in sys.modules:
            print(f"Ignoring notebook kernel args: {unknown_args}")
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")

    build_reg_inputs(
        overwrite = args.overwrite,
        testing = args.testing,
        test_sample_matches = args.test_sample_matches,
        test_random_seed = args.test_seed,
        test_firm_uid = args.test_firm_uid,
        test_lottery_year = args.test_lottery_year,
        test_materialize_intermediate_tables = args.test_materialize_intermediate_tables,
        test_table_prefix = args.test_table_prefix,
        strict_only = args.strict_only,
    )
