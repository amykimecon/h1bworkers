"""Spot-check raw match data: shows original FOIA + Revelio source fields for sampled matches.

Modes
-----
(default)  False-positive spotcheck: sample individual matches and review them.
--fn       False-negative firm×year panel: for a sample of small firm×years, display all
           FOIA applicants and all Revelio candidates side-by-side, optionally label the
           real match interactively, and report TP/FP/FN statistics vs. the algorithm.

Usage
-----
  python spotcheck_matches.py [--seed N]
  python spotcheck_matches.py --fn [--fn-firms N] [--seed N]
"""

import argparse
import duckdb as ddb
import pandas as pd
import re
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import indiv_merge_config as icfg
from config import root as data_root  # noqa: E402

# ---------- CLI args ----------
# When running inside an IPython/Jupyter kernel, sys.argv contains kernel
# connection args (e.g. ['-f', '/tmp/kernel-xxx.json']) that confuse argparse.
# Detect IPython via get_ipython() and parse from an empty argv instead so all
# options fall back to their defaults.  Override args interactively as needed:
#   args.fn = True; args.fn_firms = 5; run_fn_analysis(args.fn_firms)
try:
    _ipython = get_ipython()  # type: ignore[name-defined]  # noqa: F821
except NameError:
    _ipython = None

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--fn", action="store_true", help="Run false-negative firm×year panel")
parser.add_argument("--fn-firms", type=int, default=3, metavar="N",
                    help="Number of firm×years to sample in --fn mode (default 3)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
parser.add_argument("--n-singles",      type=int, default=2, help="SINGLE samples (default 2)")
parser.add_argument("--n-contested",    type=int, default=2, help="CONTESTED samples (default 2)")
parser.add_argument("--n-highmult",     type=int, default=1, help="HIGH-MULT samples (default 1)")
parser.add_argument("--n-no-firm-pos",  type=int, default=1, help="NO-FIRM-POS samples: matched user has no positions at FOIA firm rcid (default 1)")
parser.add_argument("--firms", nargs="+", metavar="FIRM_UID",
                    help="Restrict spotcheck to these foia_firm_uid values (space-separated)")
parser.add_argument("--max-cands",   type=int, default=5,
                    help="Max Revelio candidates shown per applicant (default 5)")
parser.add_argument("--max-pos",     type=int, default=5,
                    help="Max position rows shown per user (default 5)")
parser.add_argument("--max-educ",    type=int, default=3,
                    help="Max education rows shown per user (default 3)")
args, _ = parser.parse_known_args([] if _ipython is not None else None)

random.seed(args.seed)
con = ddb.connect()

# ---------- Data paths ----------
BLOOMBERG_CSV = os.path.join(str(data_root), "data", "raw", "foia_bloomberg",
                             "foia_bloomberg_all_withids.csv")

# ---------- Load tables ----------
con.register("baseline",
    con.read_parquet(icfg.choose_path(icfg.MERGE_FILT_BASELINE_PARQUET, icfg.MERGE_FILT_BASELINE_PARQUET_LEGACY))
)
con.register("foia_indiv",
    con.read_parquet(icfg.choose_path(icfg.FOIA_INDIV_PARQUET, icfg.FOIA_INDIV_PARQUET_LEGACY))
)
con.register("rev_indiv",
    con.read_parquet(icfg.REV_INDIV_PARQUET)
)
con.register("merged_pos_clean",
    con.read_parquet(icfg.choose_path(icfg.MERGED_POS_CLEAN_PARQUET, icfg.MERGED_POS_CLEAN_PARQUET_LEGACY))
)
con.register("rev_educ_long",
    con.read_parquet(icfg.choose_path(icfg.REV_EDUC_LONG_PARQUET, icfg.REV_EDUC_LONG_PARQUET_LEGACY))
)

# Bloomberg FOIA extra fields: primary field of study, education level definition, wage.
# Joins on foia_unique_id == foia_indiv_id (verified 1:1 match).
HAS_BLOOMBERG = os.path.exists(BLOOMBERG_CSV)
if HAS_BLOOMBERG:
    con.sql(f"""
    CREATE OR REPLACE VIEW bloomberg_extra AS
    SELECT
        CAST(foia_unique_id AS BIGINT) AS foia_indiv_id,
        CASE WHEN TRIM(JOB_TITLE)            IN ('', 'NA') THEN NULL ELSE TRIM(JOB_TITLE)            END AS raw_job_title,
        CASE WHEN TRIM(BEN_PFIELD_OF_STUDY)  IN ('', 'NA') THEN NULL ELSE TRIM(BEN_PFIELD_OF_STUDY)  END AS pfield_of_study,
        CASE WHEN TRIM(ED_LEVEL_DEFINITION)  IN ('', 'NA') THEN NULL ELSE TRIM(ED_LEVEL_DEFINITION)  END AS ed_level_def,
        TRY_CAST(NULLIF(TRIM(CAST(WAGE_AMT AS VARCHAR)), 'NA') AS DOUBLE) AS wage_amt,
        CASE WHEN TRIM(WAGE_UNIT) IN ('', 'NA', '(b)(3) (b)(6) (b)(7)(c)') THEN NULL ELSE TRIM(WAGE_UNIT) END AS wage_unit
    FROM read_csv('{BLOOMBERG_CSV}', ignore_errors=true)
    """)
else:
    print(f"[warning] Bloomberg CSV not found: {BLOOMBERG_CSV}")

# rev_user: collapse country AND rcid fan-out — one row per (user_id, foia_firm_uid).
# A user can have multiple rcids pointing to the same foia_firm_uid, so we aggregate.
con.sql("""
CREATE OR REPLACE VIEW rev_user AS
SELECT
    user_id,
    foia_firm_uid,
    ANY_VALUE(f_prob)                AS f_prob,
    ANY_VALUE(f_prob_nt)             AS f_prob_nt,
    ANY_VALUE(est_yob)               AS est_yob,
    ANY_VALUE(fullname)              AS fullname,
    MIN(first_startdate)             AS first_startdate,
    MAX(last_enddate)                AS last_enddate,
    MIN(min_startdate)               AS min_startdate,
    MIN(min_startdate_us)            AS min_startdate_us,
    MAX(updated_dt)                  AS updated_dt,
    ANY_VALUE(fields)                AS fields,
    ANY_VALUE(highest_ed_level)      AS highest_ed_level,
    ANY_VALUE(country_uncertain_ind) AS country_uncertain_ind,
    ANY_VALUE(max_anglo_pressure)    AS max_anglo_pressure,
    ANY_VALUE(hs_ind)                AS hs_ind,
    ANY_VALUE(valid_postsec)         AS valid_postsec,
    ANY_VALUE(stem_ind)              AS stem_ind,
    ANY_VALUE(ade_ind)               AS ade_ind,
    ANY_VALUE(ade_year)              AS ade_year,
    ANY_VALUE(last_grad_year)        AS last_grad_year,
    ANY_VALUE(positions)             AS positions,
    ANY_VALUE(rcids)                 AS rcids
FROM rev_indiv
GROUP BY user_id, foia_firm_uid
""")


# ---------- Formatting helpers ----------

def _yob_score(est_yob, foia_yob, yob_buffer=5):
    """Replicate the YOB component of total_score (matches indiv_merge.py logic)."""
    try:
        ey = int(est_yob)
        fy = int(foia_yob)
    except (TypeError, ValueError):
        return 0.5  # est_yob IS NULL → 0.5
    if pd.isna(est_yob):
        return 0.5
    diff = abs(ey - fy)
    if diff <= yob_buffer:
        return 1.0 - diff / (yob_buffer + 1.0)
    return 0.0


def fmt_date(val):
    if pd.isna(val):
        return "null"
    return str(val)[:10]


def fmt_float(val, decimals=3):
    if pd.isna(val):
        return "null"
    return f"{float(val):.{decimals}f}"


def fmt_wage(amt, unit):
    """Format wage as '$XX,XXX/YEAR' or '$XX/HOUR' etc."""
    if amt is None or (isinstance(amt, float) and pd.isna(amt)):
        return "null"
    unit_str = f"/{unit}" if unit and not pd.isna(unit) else ""
    return f"${int(amt):,}{unit_str}"


def print_divider(char="=", n=80):
    print(char * n)


def _bloomberg_fields(row):
    """Return (field_str, ed_str, job_str, wage_str) enriched with Bloomberg data."""
    ed     = row.foia_highest_ed_level
    field  = row.foia_field_clean
    job    = row.foia_job_title
    if not HAS_BLOOMBERG:
        return field, ed, job, "null"
    pfield   = getattr(row, "bloomberg_field_study", None)
    ed_def   = getattr(row, "bloomberg_ed_level",    None)
    raw_job  = getattr(row, "bloomberg_job_title",   None)
    wage_amt = getattr(row, "bloomberg_wage_amt",    None)
    wage_u   = getattr(row, "bloomberg_wage_unit",   None)
    # Append bloomberg value in brackets when it adds info
    na_vals = {None, "na", "none", "null", ""}
    def _maybe_append(base, extra):
        if extra is None or (isinstance(extra, float) and pd.isna(extra)):
            return str(base)
        if str(extra).lower() in na_vals:
            return str(base)
        if str(extra) == str(base):
            return str(base)
        return f"{base} [{extra}]"
    return (
        _maybe_append(field, pfield),
        _maybe_append(ed,    ed_def),
        _maybe_append(job,   raw_job),
        fmt_wage(wage_amt, wage_u),
    )


# ---------- Soft firm-name matching ----------

_FIRM_IGNORE = {
    # legal suffixes
    'inc', 'llc', 'corp', 'co', 'ltd', 'lp', 'plc', 'llp',
    # 3-letter stop words
    'the', 'and', 'for', 'new', 'usa', 'us', 'of',
    # generic descriptors (4+ chars)
    'group', 'services', 'solutions', 'systems', 'international', 'global',
    'management', 'consulting', 'technology', 'technologies', 'associates',
    'holdings', 'enterprises', 'partners', 'partnership',
}


def _firm_tokens(name) -> set:
    """Return significant word tokens from a company name."""
    if not name or (isinstance(name, float) and pd.isna(name)):
        return set()
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', str(name))  # split camelCase before lowercasing
    clean = re.sub(r'[^a-z0-9 ]', ' ', s.lower())
    return {t for t in clean.split() if len(t) >= 3 and t not in _FIRM_IGNORE}


def _soft_firm_match(foia_name, company_raw) -> bool:
    """Return True if foia_name and company_raw share at least one significant token."""
    ft = _firm_tokens(foia_name)
    rt = _firm_tokens(company_raw)
    return bool(ft and rt and ft & rt)


# ---------- Core print for a single match candidate ----------

def print_match(row, pos_rows, educ_rows, bucket, firm_rcids=None):
    print_divider()
    print(f"[{bucket}]  foia_indiv_id={row.foia_indiv_id}  lottery_year={row.lottery_year}")

    # --- FOIA APPLICATION ---
    print_divider("-", 60)
    print("  FOIA APPLICATION")
    print_divider("-", 60)
    print(f"  employer  : {row.employer_name}  (state=N/A, FEIN={row.FEIN}, foia_firm_uid={row.foia_firm_uid})")
    print(f"  cob       : {row.foia_country}  subregion={row.subregion}")
    print(f"  yob       : {row.foia_yob}    gender (female_ind)={row.foia_female_ind}    status={row.status_type}")
    field_str, ed_str, job_str, wage_str = _bloomberg_fields(row)
    print(f"  edu       : {ed_str}    field={field_str}")
    print(f"  job       : {job_str}    wage={wage_str}")
    print(f"  n_apps(firm×year)={row.n_apps}")
    print()

    # --- REVELIO MATCH ---
    print_divider("-", 60)
    print(f"  REVELIO MATCH  user_id={row.user_id}  weight={fmt_float(row.weight_norm)}  total_score={fmt_float(row.total_score)}  n_candidates={int(row.n_match_filt)}")
    # Score component breakdown
    _ys  = _yob_score(getattr(row, "rev_est_yob_raw", None), getattr(row, "foia_yob", None))
    _fs  = float(row.f_score)  if not pd.isna(row.f_score)  else 0.0
    _cs  = float(row.country_score) if not pd.isna(row.country_score) else 0.0
    _fm  = float(row.firm_match_quality_mult) if not pd.isna(row.firm_match_quality_mult) else 1.0
    _base = _cs * 0.70 + _fs * 0.10 + _ys * 0.20
    print(f"               country={fmt_float(_cs)}×0.70={fmt_float(_cs*0.70)}"
          f"  gender={fmt_float(_fs)}×0.10={fmt_float(_fs*0.10)}"
          f"  yob={fmt_float(_ys)}×0.20={fmt_float(_ys*0.20)}"
          f"  sum={fmt_float(_base)}  ×firm_mult={fmt_float(_fm)}"
          f"  =total={fmt_float(_base*_fm)}")
    print_divider("-", 60)

    # RAW section
    print("  [RAW]")
    print(f"    name           : {row.rev_fullname}")

    # Revelio education records (capped)
    max_educ = args.max_educ
    if len(educ_rows) > 0:
        shown = educ_rows.head(max_educ)
        tail  = f"  (+{len(educ_rows) - max_educ} more)" if len(educ_rows) > max_educ else ""
        print(f"\n    Education ({len(educ_rows)} record(s){tail}):")
        for _, e in shown.iterrows():
            yrs = f"{e.ed_startdate or '?'} → {e.ed_enddate or '?'}"
            print(f"      [{yrs}]  {e.university_raw}  |  {e.degree_clean}  |  {e.match_country}")
    else:
        print("    Education      : [none]")

    # Firm-match quality: positions at FOIA employer's rcids (hard match) or name match (soft)
    if firm_rcids is not None:
        at_firm = pos_rows[pos_rows.rcid.isin(firm_rcids)] if len(pos_rows) > 0 else pd.DataFrame()
        if not at_firm.empty:
            print(f"\n    Positions AT FOIA FIRM ({len(at_firm)} record(s), rcids={firm_rcids}):")
            for _, p in at_firm.iterrows():
                sal = f"  salary={int(p.salary)}" if not pd.isna(p.salary) and p.salary > 0 else ""
                print(f"      [{fmt_date(p.startdate)} → {fmt_date(p.enddate)}]  {p.company_raw}  |  {p.title_raw}  |  {p.country}{sal}")
        else:
            # Check for soft name matches among all positions
            employer = getattr(row, 'employer_name', None)
            soft_matches = pd.DataFrame()
            if len(pos_rows) > 0 and employer:
                mask = pos_rows.company_raw.apply(lambda c: _soft_firm_match(employer, c))
                soft_matches = pos_rows[mask]
            if not soft_matches.empty:
                print(f"\n    Positions via SOFT FIRM MATCH (name-based, rcid not in crosswalk)"
                      f"  [crosswalk rcids={firm_rcids}]:")
                for _, p in soft_matches.iterrows():
                    sal = f"  salary={int(p.salary)}" if not pd.isna(p.salary) and p.salary > 0 else ""
                    print(f"      [{fmt_date(p.startdate)} → {fmt_date(p.enddate)}]  {p.company_raw}"
                          f"  |  {p.title_raw}  |  {p.country}{sal}  rcid={p.rcid}")
            else:
                print(f"\n    *** FIRM-MATCH WARNING: no positions at FOIA firm rcids {firm_rcids} ***")

    # Full position history (capped)
    max_pos = args.max_pos
    if len(pos_rows) == 0:
        print("    Positions      : [none]")
    else:
        shown = pos_rows.head(max_pos)
        tail  = f"  (+{len(pos_rows) - max_pos} more)" if len(pos_rows) > max_pos else ""
        print(f"\n    All positions ({len(pos_rows)} records{tail}):")
        for _, p in shown.iterrows():
            sal = f"  salary={int(p.salary)}" if not pd.isna(p.salary) and p.salary > 0 else ""
            print(f"      [{fmt_date(p.startdate)} → {fmt_date(p.enddate)}]  {p.company_raw}  |  {p.title_raw}  |  {p.country}  |  {p.role_k17000_v3}{sal}  rcid={p.rcid}")

    # IMPUTED section
    print()
    print("  [IMPUTED]")
    nanat  = fmt_float(getattr(row, "nanat_score",           float("nan")))
    nt_sub = fmt_float(getattr(row, "nt_subregion_score",    float("nan")))
    nt_nat = fmt_float(getattr(row, "nanat_subregion_score", float("nan")))
    print(f"    cob            : {row.rev_country}  (country_score={fmt_float(row.country_score)}, subregion_score={fmt_float(row.subregion_score)}, uncertain={row.rev_country_uncertain})")
    print(f"    name signals   : nanat={nanat}  nt_subregion={nt_sub}  nanat_subregion={nt_nat}")
    print(f"    yob            : {row.rev_est_yob_raw}  (foia_yob={row.foia_yob})")
    print(f"    gender         : f_prob={fmt_float(row.rev_f_prob)}  f_prob_nt={fmt_float(row.rev_f_prob_nt)}  f_score={fmt_float(row.f_score)}")
    llm_raw  = fmt_float(getattr(row, "llm_match_score",      float("nan")))
    llm_norm = fmt_float(getattr(row, "llm_match_score_norm", float("nan")))
    fm_mult  = fmt_float(getattr(row, "firm_match_quality_mult", float("nan")))
    print(f"    llm_match_score: raw={llm_raw}  norm={llm_norm}  firm_quality_mult={fm_mult}")
    print(f"    stem_ind       : {row.rev_stem_ind}  max_anglo={fmt_float(row.max_anglo_pressure)}")
    print(f"    edu            : {row.rev_highest_ed_level}  hs_ind={row.hs_ind}  valid_postsec={row.valid_postsec}")
    print(f"    ade_ind={row.ade_ind}  ade_year={row.ade_year}  last_grad_year={row.last_grad_year}")
    print(f"    dates          : first_start={fmt_date(row.first_startdate)}  last_end={fmt_date(row.last_enddate)}")
    print(f"                     min_start={fmt_date(row.min_startdate)}  min_start_us={fmt_date(row.min_startdate_us)}  updated={fmt_date(row.updated_dt)}")
    startdiff = fmt_float(getattr(row, "startdatediff",                    float("nan")), 0)
    enddiff   = fmt_float(getattr(row, "enddatediff",                      float("nan")), 0)
    msg       = fmt_float(getattr(row, "months_since_grad", float("nan")), 0)
    print(f"                     startdatediff={startdiff}  enddatediff={enddiff}  months_since_grad={msg}")
    print(f"    fields         : {row.rev_fields}")


# =============================================================================
# PART A: False-positive spotcheck (default mode)
# =============================================================================

def run_spotcheck():
    # --- Sample selection ---
    firm_filter = ""
    if args.firms:
        firms_sql = ", ".join(f"'{f}'" for f in args.firms)
        firm_filter = f"WHERE foia_firm_uid IN ({firms_sql})"

    mult_df = con.sql(f"""
        SELECT foia_indiv_id, COUNT(*) AS n_candidates, MAX(weight_norm) AS max_weight
        FROM baseline
        {firm_filter}
        GROUP BY foia_indiv_id
    """).df()

    singles_ids   = mult_df[mult_df.n_candidates == 1].foia_indiv_id.tolist()
    contested_ids = mult_df[(mult_df.n_candidates.between(2, 5)) & (mult_df.max_weight > 0.5)].foia_indiv_id.tolist()
    highmult_ids  = mult_df[mult_df.n_candidates > 20].foia_indiv_id.tolist()

    # NO-FIRM-POS: top-weight match has no position at the FOIA firm's rcid(s)
    # Also exclude cases where a soft name match exists (crosswalk gap, not a real failure).
    no_firm_pos_raw = con.sql(f"""
        WITH top_match AS (
            SELECT foia_indiv_id, foia_firm_uid, user_id
            FROM baseline
            {firm_filter}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY foia_indiv_id ORDER BY weight_norm DESC) = 1
        ),
        firm_rcids AS (
            SELECT DISTINCT foia_firm_uid, rcid FROM foia_indiv WHERE rcid IS NOT NULL
        ),
        user_firm_pos AS (
            SELECT DISTINCT p.user_id, fr.foia_firm_uid
            FROM merged_pos_clean p
            JOIN firm_rcids fr ON p.rcid = fr.rcid
        )
        SELECT t.foia_indiv_id, t.foia_firm_uid, t.user_id
        FROM top_match t
        LEFT JOIN user_firm_pos ufp ON t.user_id = ufp.user_id AND t.foia_firm_uid = ufp.foia_firm_uid
        WHERE ufp.user_id IS NULL
    """).df()

    if no_firm_pos_raw.empty:
        no_firm_pos_ids = []
    else:
        # Sample a pool (much larger than needed) then soft-filter for efficiency
        pool_size = min(len(no_firm_pos_raw), max(args.n_no_firm_pos * 20, 100))
        pool = no_firm_pos_raw.sample(pool_size, random_state=args.seed)

        pool_uids_sql = ", ".join(str(u) for u in pool.user_id.tolist())
        pool_fids_sql = ", ".join(f"'{x}'" for x in pool.foia_firm_uid.unique().tolist())

        pool_pos_df = con.sql(f"""
            SELECT user_id, company_raw FROM merged_pos_clean
            WHERE user_id IN ({pool_uids_sql})
        """).df()
        pool_emp_df = con.sql(f"""
            SELECT foia_firm_uid, ANY_VALUE(employer_name) AS employer_name
            FROM foia_indiv
            WHERE foia_firm_uid IN ({pool_fids_sql})
            GROUP BY foia_firm_uid
        """).df()

        _emp_map  = dict(zip(pool_emp_df.foia_firm_uid, pool_emp_df.employer_name))
        _user_cos = pool_pos_df.groupby('user_id')['company_raw'].apply(list).to_dict()

        def _has_soft(r):
            emp = _emp_map.get(r.foia_firm_uid, '')
            return any(_soft_firm_match(emp, c) for c in _user_cos.get(r.user_id, []))

        no_firm_pos_ids = pool[~pool.apply(_has_soft, axis=1)].foia_indiv_id.tolist()

    sample_singles      = random.sample(singles_ids,      min(args.n_singles,      len(singles_ids)))
    sample_contested    = random.sample(contested_ids,    min(args.n_contested,    len(contested_ids)))
    sample_highmult     = random.sample(highmult_ids,     min(args.n_highmult,     len(highmult_ids)))
    sample_no_firm_pos  = random.sample(no_firm_pos_ids,  min(args.n_no_firm_pos,  len(no_firm_pos_ids)))

    all_sampled = sample_singles + sample_contested + sample_highmult + sample_no_firm_pos
    labels = (
        ["SINGLE"]       * len(sample_singles)     +
        ["CONTESTED"]    * len(sample_contested)   +
        ["HIGH-MULT"]    * len(sample_highmult)    +
        ["NO-FIRM-POS"]  * len(sample_no_firm_pos)
    )

    # --- Fetch full match data ---
    ids_sql = ", ".join(f"'{x}'" for x in all_sampled)

    bloomberg_join = "LEFT JOIN bloomberg_extra bx ON f.foia_indiv_id = bx.foia_indiv_id" if HAS_BLOOMBERG else ""

    # months_since_grad may not exist in older baseline parquet files
    _baseline_cols = con.sql("SELECT * FROM baseline LIMIT 0").columns
    months_since_grad_col = "b.months_since_grad" if "months_since_grad" in _baseline_cols else "NULL AS months_since_grad"
    bloomberg_cols = """
        bx.raw_job_title   AS bloomberg_job_title,
        bx.pfield_of_study AS bloomberg_field_study,
        bx.ed_level_def    AS bloomberg_ed_level,
        bx.wage_amt        AS bloomberg_wage_amt,
        bx.wage_unit       AS bloomberg_wage_unit,""" if HAS_BLOOMBERG else ""

    match_df = con.sql(f"""
    WITH foia_dedup AS (
        -- foia_indiv has one row per (foia_indiv_id, rcid); take one row per foia_indiv_id
        SELECT DISTINCT ON (foia_indiv_id)
            foia_indiv_id, foia_firm_uid, FEIN, lottery_year,
            country, subregion, female_ind, yob, status_type,
            employer_name, highest_ed_level, field_clean, job_title, n_apps
        FROM foia_indiv
        QUALIFY ROW_NUMBER() OVER (PARTITION BY foia_indiv_id ORDER BY foia_indiv_id) = 1
    )
    SELECT
        b.foia_indiv_id,
        b.foia_firm_uid,
        b.lottery_year,
        b.user_id,
        b.fullname        AS rev_fullname,
        b.foia_country,
        b.rev_country,
        b.subregion,
        b.country_score,
        b.subregion_score,
        b.country_uncertain_ind AS b_country_uncertain,
        b.f_prob_avg,
        b.f_score,
        b.yob             AS foia_yob,
        b.est_yob         AS rev_est_yob,
        b.stem_ind,
        b.total_score,
        b.weight_norm,
        b.n_match_filt,
        b.llm_match_score,
        b.llm_match_score_norm,
        b.firm_match_quality_mult,
        b.startdatediff,
        b.enddatediff,
        {months_since_grad_col},
        -- raw FOIA fields
        f.employer_name,
        f.FEIN,
        f.female_ind      AS foia_female_ind,
        f.status_type,
        f.highest_ed_level AS foia_highest_ed_level,
        f.field_clean     AS foia_field_clean,
        f.job_title       AS foia_job_title,
        f.n_apps,
        {bloomberg_cols}
        -- raw Revelio fields
        r.f_prob          AS rev_f_prob,
        r.f_prob_nt       AS rev_f_prob_nt,
        r.est_yob         AS rev_est_yob_raw,
        r.first_startdate,
        r.last_enddate,
        r.min_startdate,
        r.min_startdate_us,
        r.updated_dt,
        r.fields          AS rev_fields,
        r.highest_ed_level AS rev_highest_ed_level,
        r.country_uncertain_ind AS rev_country_uncertain,
        r.max_anglo_pressure,
        r.hs_ind,
        r.valid_postsec,
        r.stem_ind        AS rev_stem_ind,
        r.ade_ind,
        r.ade_year,
        r.last_grad_year,
        r.positions       AS rev_positions_arr,
        r.rcids           AS rev_rcids,
        -- name-signal scores for matched country (from rev_indiv)
        ri.nanat_score,
        ri.nanat_subregion_score,
        ri.nt_subregion_score
    FROM baseline b
    JOIN foia_dedup f
        ON b.foia_indiv_id = f.foia_indiv_id
    {bloomberg_join}
    LEFT JOIN rev_user r
        ON b.user_id = r.user_id AND b.foia_firm_uid = r.foia_firm_uid
    LEFT JOIN (
        SELECT user_id, foia_firm_uid, country, nanat_score, nanat_subregion_score, nt_subregion_score
        FROM rev_indiv
    ) ri ON b.user_id = ri.user_id AND b.foia_firm_uid = ri.foia_firm_uid AND b.rev_country = ri.country
    WHERE b.foia_indiv_id IN ({ids_sql})
    ORDER BY b.foia_indiv_id, b.weight_norm DESC
    """).df()

    user_ids = match_df.user_id.dropna().unique().tolist()
    user_ids_sql = ", ".join(str(u) for u in user_ids)

    pos_df = con.sql(f"""
    SELECT user_id, rcid, company_raw, title_raw, startdate, enddate, country, role_k17000_v3,
        ROUND(salary, 0) AS salary
    FROM merged_pos_clean
    WHERE user_id IN ({user_ids_sql})
    ORDER BY user_id, startdate
    """).df()

    educ_df = con.sql(f"""
    SELECT user_id, university_raw, degree_clean, ed_startdate, ed_enddate, match_country, educ_order
    FROM rev_educ_long
    WHERE user_id IN ({user_ids_sql})
    ORDER BY user_id, educ_order
    """).df()

    # --- Build firm_rcids_map: foia_firm_uid -> list of rcids ---
    firm_uids = match_df.foia_firm_uid.dropna().unique().tolist()
    firm_uids_sql = ", ".join(f"'{x}'" for x in firm_uids)
    firm_rcids_df = con.sql(f"""
    SELECT foia_firm_uid, LIST(DISTINCT rcid ORDER BY rcid) AS rcids
    FROM foia_indiv
    WHERE foia_firm_uid IN ({firm_uids_sql}) AND rcid IS NOT NULL
    GROUP BY foia_firm_uid
    """).df()
    firm_rcids_map = dict(zip(firm_rcids_df.foia_firm_uid, firm_rcids_df.rcids))

    # --- Print ---
    label_map = dict(zip(all_sampled, labels))

    for foia_id in all_sampled:
        bucket = label_map[foia_id]
        group = match_df[match_df.foia_indiv_id == foia_id]
        if group.empty:
            continue
        n_cands = len(group)
        max_cands = args.max_cands
        print()
        print_divider("=")
        tail = f"  (showing top {max_cands})" if n_cands > max_cands else ""
        print(f"=== {bucket}  —  {n_cands} candidate(s){tail} ===")

        for i, (_, row) in enumerate(group.head(max_cands).iterrows()):
            user_pos   = pos_df[pos_df.user_id == row.user_id].copy()
            user_educ  = educ_df[educ_df.user_id == row.user_id].copy()
            firm_rcids = firm_rcids_map.get(row.foia_firm_uid, [])
            print()
            if n_cands > 1:
                print(f"  --- Candidate {i+1}/{min(n_cands, max_cands)} ---")
            print_match(row, user_pos, user_educ, bucket, firm_rcids=firm_rcids)

    print()
    print_divider()
    print("Done.")


# =============================================================================
# PART B: False-negative firm×year panel (--fn mode)
# =============================================================================

def _foia_row_summary(fa):
    """One-line summary of a FOIA applicant row for the panel view."""
    ed    = getattr(fa, "highest_ed_level", None) or "?"
    field = getattr(fa, "field_clean",      None) or "?"
    job   = getattr(fa, "job_title",        None) or "?"
    if HAS_BLOOMBERG:
        pfield   = getattr(fa, "pfield_of_study", None)
        ed_def   = getattr(fa, "ed_level_def",    None)
        raw_job  = getattr(fa, "raw_job_title",   None)
        wage_amt = getattr(fa, "wage_amt",         None)
        wage_u   = getattr(fa, "wage_unit",        None)
        na_vals  = {None, "na", "none", "null", ""}
        def _enrich(base, extra):
            if extra is None or (isinstance(extra, float) and pd.isna(extra)):
                return str(base)
            if str(extra).lower() in na_vals or str(extra) == str(base):
                return str(base)
            return f"{base} [{extra}]"
        ed    = _enrich(ed, ed_def)
        field = _enrich(field, pfield)
        job   = _enrich(job,   raw_job)
        wage  = fmt_wage(wage_amt, wage_u)
    else:
        wage = "null"
    return (f"country={fa.country}/{fa.subregion}, yob={fa.yob}, {fa.status_type}\n"
            f"       edu={ed}, field={field}\n"
            f"       job={job}  wage={wage}")


def _print_fn_panel(firm_uid, year, employer, foia_rows, rev_list,
                    baseline_firm, rev_pos, rev_educ_df):
    """Display a firm×year panel and optionally collect interactive labels."""
    print()
    print_divider("=")
    print(f"=== FIRM×YEAR: {employer}  ({firm_uid}, year={year}) ===")
    print(f"    {len(foia_rows)} FOIA applicant(s)  |  {len(rev_list)} Revelio candidate(s)")

    # --- Numbered Revelio candidates ---
    print("\n--- Revelio Candidates ---")
    for i, r in rev_list.iterrows():
        num = i + 1
        rpos = rev_pos[rev_pos.user_id == r.user_id] if not rev_pos.empty else pd.DataFrame()
        reduc = rev_educ_df[rev_educ_df.user_id == r.user_id] if not rev_educ_df.empty else pd.DataFrame()
        pos_strs = [
            f"{fmt_date(p.startdate)}→{fmt_date(p.enddate)} {str(p.title_raw)[:28] if pd.notna(p.title_raw) else ''}"
            for _, p in rpos.iterrows()
        ]
        educ_strs = [
            f"{str(e.university_raw)[:35] if pd.notna(e.university_raw) else '?'}"
            f" ({e.degree_clean}, {e.ed_enddate or '?'})"
            for _, e in reduc.iterrows()
        ]
        print(f"  [{num:2d}] user_id={r.user_id}  {r.fullname}"
              f"  yob={r.est_yob}  f_prob={fmt_float(r.f_prob)}"
              f"  edu={r.highest_ed_level}")
        if educ_strs:
            print(f"        educ : " + " | ".join(educ_strs[:2]))
        if pos_strs:
            print(f"        pos  : " + " | ".join(pos_strs[:3]))

    # --- Per-applicant: show details, algorithm pick, prompt ---
    ground_truth = {}  # foia_indiv_id -> user_id or None

    for _, fa in foia_rows.iterrows():
        foia_id = fa.foia_indiv_id
        print()
        print_divider("-")
        print(f"  FOIA applicant  id={foia_id}")
        print(f"       {_foia_row_summary(fa)}")

        # Algorithm's candidates for this applicant
        app_matches = baseline_firm[baseline_firm.foia_indiv_id == foia_id].copy()
        if app_matches.empty:
            print("  Algorithm: [no candidates in baseline]")
        else:
            top = app_matches.iloc[0]
            top_idx = rev_list[rev_list.user_id == top.user_id].index
            top_num = (top_idx[0] + 1) if len(top_idx) > 0 else None
            ref = f" → candidate #{top_num}" if top_num is not None else " [not in candidate list above]"
            print(f"  Algorithm top : user_id={top.user_id} ({top.fullname})"
                  f"  weight={fmt_float(top.weight_norm)}{ref}")
            for _, o in app_matches.iloc[1:4].iterrows():
                on = rev_list[rev_list.user_id == o.user_id].index
                onum = on[0] + 1 if len(on) > 0 else "?"
                print(f"    also: user_id={o.user_id} ({o.fullname})"
                      f"  weight={fmt_float(o.weight_norm)}  →  #{onum}")

        # Interactive prompt
        try:
            resp = input(f"\n  Your pick — enter # (1–{len(rev_list)}), 'n'=no match, Enter=skip: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [interactive input not available, skipping]")
            continue

        if resp == "n":
            ground_truth[foia_id] = None
        elif resp.isdigit():
            idx = int(resp) - 1
            if 0 <= idx < len(rev_list):
                ground_truth[foia_id] = rev_list.iloc[idx].user_id
            else:
                print(f"  [invalid index {resp}, skipping]")
        # blank Enter = skip (no ground truth recorded)

    # --- Summary ---
    if not ground_truth:
        print("\n  [no ground truth entered — skipping summary]")
        return

    print()
    print_divider("-")
    print("  === Match Quality Summary ===")
    n_tp = n_fp = n_fn_missing = n_fn_ranked = n_tn = 0

    for foia_id, gt_user in ground_truth.items():
        app_matches  = baseline_firm[baseline_firm.foia_indiv_id == foia_id]
        algo_top     = app_matches.iloc[0].user_id if not app_matches.empty else None
        algo_all     = set(app_matches.user_id.tolist())

        if gt_user is None:
            if algo_top is not None:
                tag = "FP       (algo matched; human says no real match)"
                n_fp += 1
            else:
                tag = "TN       (neither algo nor human found a match)"
                n_tn += 1
        elif gt_user == algo_top:
            tag = "TP  ✓    (algo top == human pick)"
            n_tp += 1
        elif gt_user in algo_all:
            tag = "FN-ranked  (correct user in candidates but not algo's top pick)"
            n_fn_ranked += 1
        else:
            tag = "FN-missing (correct user absent from algo's candidate set)"
            n_fn_missing += 1

        def _name(uid):
            if uid is None:
                return "none"
            row = rev_list[rev_list.user_id == uid]
            return row.fullname.values[0] if not row.empty else str(uid)

        print(f"    foia={foia_id}: human={gt_user} ({_name(gt_user)})"
              f"  algo={algo_top} ({_name(algo_top)})")
        print(f"      → {tag}")

    n_lab = len(ground_truth)
    denom_p = n_tp + n_fp
    denom_r = n_tp + n_fn_missing + n_fn_ranked
    precision = n_tp / denom_p if denom_p > 0 else float("nan")
    recall    = n_tp / denom_r  if denom_r > 0 else float("nan")
    print()
    print(f"  Labeled: {n_lab}  TP={n_tp}  FP={n_fp}"
          f"  FN-missing={n_fn_missing}  FN-ranked={n_fn_ranked}  TN={n_tn}")
    print(f"  Precision = {precision:.2f}    Recall = {recall:.2f}"
          f"    (based on {n_lab} labeled applicants)")


def run_fn_analysis(n_firms):
    """Sample small firm×years and interactively evaluate algorithm matches."""

    # Sample firm×years with 2–8 apps and 2–6 matched Revelio users (tractable for review)
    all_fy = con.sql("""
        SELECT b.foia_firm_uid, b.lottery_year,
            COUNT(DISTINCT b.foia_indiv_id) AS n_apps,
            COUNT(DISTINCT b.user_id)       AS n_rev_matched
        FROM baseline b
        GROUP BY b.foia_firm_uid, b.lottery_year
        HAVING n_apps BETWEEN 2 AND 8 AND n_rev_matched BETWEEN 2 AND 6
    """).df()

    if all_fy.empty:
        print("[error] No firm×years in baseline match the size criteria.")
        return

    sampled_fy = all_fy.sample(min(n_firms, len(all_fy)), random_state=args.seed)

    bloomberg_join = ("LEFT JOIN bloomberg_extra bx ON fi.foia_indiv_id = bx.foia_indiv_id"
                      if HAS_BLOOMBERG else "")
    bloomberg_cols = (", bx.raw_job_title, bx.pfield_of_study, bx.ed_level_def, bx.wage_amt, bx.wage_unit"
                      if HAS_BLOOMBERG else "")

    for _, fy in sampled_fy.iterrows():
        firm_uid = fy.foia_firm_uid
        year     = int(fy.lottery_year)

        # All FOIA applicants at this firm×year (one row per foia_indiv_id)
        foia_rows = con.sql(f"""
        SELECT fi.foia_indiv_id, fi.country, fi.subregion, fi.yob, fi.female_ind,
            fi.status_type, fi.employer_name, fi.highest_ed_level, fi.field_clean, fi.job_title
            {bloomberg_cols}
        FROM foia_indiv fi
        {bloomberg_join}
        WHERE fi.foia_firm_uid = '{firm_uid}' AND fi.lottery_year = {year}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY fi.foia_indiv_id ORDER BY fi.foia_indiv_id) = 1
        """).df()

        employer = foia_rows.employer_name.iloc[0] if not foia_rows.empty else firm_uid

        # All Revelio candidates at this firm (from rev_user VIEW, already 1 row per user_id)
        rev_rows = con.sql(f"""
        SELECT user_id, fullname, f_prob, f_prob_nt, est_yob,
            fields, highest_ed_level, min_startdate, min_startdate_us, country_uncertain_ind
        FROM rev_user
        WHERE foia_firm_uid = '{firm_uid}'
        ORDER BY fullname
        """).df()

        if rev_rows.empty:
            print(f"\n[skip] No Revelio candidates for {employer} ({firm_uid}, {year})")
            continue

        rev_list = rev_rows.reset_index(drop=True)

        # Algorithm's candidates for this firm×year (all rows, ranked by weight)
        baseline_firm = con.sql(f"""
        SELECT foia_indiv_id, user_id, fullname, weight_norm, total_score, n_match_filt
        FROM baseline
        WHERE foia_firm_uid = '{firm_uid}' AND lottery_year = {year}
        ORDER BY foia_indiv_id, weight_norm DESC
        """).df()

        # Positions at this firm's rcids for all Revelio candidates
        firm_rcids = con.sql(f"""
        SELECT DISTINCT rcid FROM foia_indiv
        WHERE foia_firm_uid = '{firm_uid}' AND rcid IS NOT NULL
        """).df().rcid.dropna().tolist()

        uids_sql  = ", ".join(str(u) for u in rev_list.user_id.tolist())
        rcids_sql = ", ".join(str(r) for r in firm_rcids)

        if firm_rcids:
            rev_pos = con.sql(f"""
            SELECT user_id, rcid, company_raw, title_raw, startdate, enddate, country
            FROM merged_pos_clean
            WHERE user_id IN ({uids_sql}) AND rcid IN ({rcids_sql})
            ORDER BY user_id, startdate
            """).df()
        else:
            rev_pos = pd.DataFrame()

        rev_educ_df = con.sql(f"""
        SELECT user_id, university_raw, degree_clean, ed_startdate, ed_enddate,
            match_country, educ_order
        FROM rev_educ_long
        WHERE user_id IN ({uids_sql})
        ORDER BY user_id, educ_order
        """).df()

        _print_fn_panel(firm_uid, year, employer, foia_rows, rev_list,
                        baseline_firm, rev_pos, rev_educ_df)

    print()
    print_divider()
    print("Done.")


# ---------- Entry point ----------

if args.fn:
    run_fn_analysis(args.fn_firms)
else:
    run_spotcheck()
