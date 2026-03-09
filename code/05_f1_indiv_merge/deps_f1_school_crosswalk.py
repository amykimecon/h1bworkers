"""
deps_f1_school_crosswalk.py
===========================
Pre-requisite dependency for the F1 individual merge pipeline.

Builds a direct fuzzy-match crosswalk between:
  - F1 FOIA school names (from the FOIA SEVP panel)
  - Revelio university_raw names (from the Revelio education panel)

This crosswalk is intentionally built WITHOUT using IPEDS as an intermediary,
so that non-US schools (which are not in IPEDS) are also matched.

Approach:
  1. Extract distinct F1 school names → clean with _sql_clean_inst_name logic
  2. Extract distinct Revelio university_raw → clean with inst_clean_regex_sql
  3. Fuzzy join using fuzzy_join_lev_jw() (Levenshtein + Jaro-Winkler)
  4. Save crosswalk parquet

Output: f1_rev_school_crosswalk_{run_tag}.parquet
  Columns: f1_school_name, f1_instname_clean, rev_university_raw,
           rev_instname_clean, match_score, match_ambiguous_ind

Usage (from iPython):
    import importlib, sys
    sys.path.insert(0, '/home/yk0581/h1bworkers/code/05_f1_indiv_merge')
    import deps_f1_school_crosswalk
    importlib.reload(deps_f1_school_crosswalk)
    deps_f1_school_crosswalk.build_school_crosswalk()
"""

import os
import sys
import time

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))

import f1_indiv_merge_config as cfg  # noqa: E402 – loaded after path setup
from helpers import fuzzy_join_lev_jw  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _sql_clean_inst_name_expr(col: str) -> str:
    """SQL expression that cleans an institution name for fuzzy matching.

    Applies the same transformations as deps_foia_clean._sql_clean_inst_name:
      - lowercase
      - remove words: at, campus, inc, the, and content in parentheses
      - strip accents (DuckDB strip_accents)
      - remove apostrophes, periods
      - convert & / + to 'and'
      - remove other punctuation (replace with space)
      - normalize whitespace
    """
    # Remove stopwords and parenthetical content first
    stripped = (
        f"REGEXP_REPLACE({col}, "
        r"'(?i)\b(at|campus|inc|the)\b|\([^\)]*\)', ' ', 'g')"
    )
    # Then apply the same chain as inst_clean_regex_sql from helpers.py
    return f"""TRIM(REGEXP_REPLACE(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(
                        strip_accents(lower({stripped})),
                    '\\s*(\\(|\\[)[^\\)\\]]*(\\)|\\])\\s*', ' ', 'g'),
                $$'|'|\\.$$, '', 'g'),
            '\\s?(&|\\+)\\s?', ' and ', 'g'),
        '[^A-z0-9\\s]', ' ', 'g'),
    '\\s+', ' ', 'g'))"""


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_school_crosswalk(overwrite: bool = None, con=None) -> pd.DataFrame:
    """Build F1 → Revelio school name crosswalk via fuzzy matching.

    Returns the crosswalk DataFrame and writes it to parquet.
    """
    out_path = cfg.F1_REV_SCHOOL_CROSSWALK_PARQUET
    if overwrite is None:
        overwrite = cfg.BUILD_OVERWRITE
    threshold = cfg.BUILD_SCHOOL_MATCH_THRESHOLD

    if os.path.exists(out_path) and not overwrite:
        print(f"[crosswalk] Skipping — file exists: {out_path}")
        return pd.read_parquet(out_path)

    if con is None:
        con = duckdb.connect()

    t_total = time.perf_counter()
    print("=" * 70)
    print("deps_f1_school_crosswalk: building F1 → Revelio school crosswalk")
    print(f"  F1 FOIA:         {cfg.F1_FOIA_PARQUET}")
    rev_educ_path = cfg.choose_path(cfg.REV_EDUC_LONG_PARQUET, cfg.REV_EDUC_LONG_PARQUET_LEGACY)
    print(f"  Revelio educ:    {rev_educ_path}")
    print(f"  Match threshold: {threshold}")
    print(f"  Output:          {out_path}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Extract distinct F1 school names
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print("\n[1/4] Extracting distinct F1 school names...")

    f1_clean_expr = _sql_clean_inst_name_expr("school_name")
    f1_schools_df = con.sql(f"""
        SELECT
            school_name AS f1_school_name,
            {f1_clean_expr} AS f1_instname_clean,
            COUNT(DISTINCT person_id) AS n_persons
        FROM read_parquet('{cfg.F1_FOIA_PARQUET}')
        WHERE school_name IS NOT NULL
          AND TRIM(school_name) != ''
        GROUP BY school_name, f1_instname_clean
        ORDER BY n_persons DESC
    """).df()

    print(f"  {len(f1_schools_df):,} distinct F1 school names "
          f"({f1_schools_df['n_persons'].sum():,.0f} total person-records)")

    # ------------------------------------------------------------------
    # 2. Extract distinct Revelio university_raw names
    # ------------------------------------------------------------------
    print(f"\n[2/4] Extracting distinct Revelio university names from: {rev_educ_path}")
    rev_clean_expr = _sql_clean_inst_name_expr("university_raw")
    rev_schools_df = con.sql(f"""
        SELECT
            university_raw AS rev_university_raw,
            {rev_clean_expr} AS rev_instname_clean,
            COUNT(DISTINCT user_id) AS n_users
        FROM read_parquet('{rev_educ_path}')
        WHERE university_raw IS NOT NULL
          AND TRIM(university_raw) != ''
        GROUP BY university_raw, rev_instname_clean
        ORDER BY n_users DESC
    """).df()

    print(f"  {len(rev_schools_df):,} distinct Revelio university names "
          f"({rev_schools_df['n_users'].sum():,.0f} total user-records)")

    elapsed = time.perf_counter() - t0
    print(f"  Data extraction done ({_fmt_elapsed(elapsed)})")

    # ------------------------------------------------------------------
    # 3. Fuzzy join
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print(f"\n[3/4] Fuzzy joining school names (threshold={threshold})...")

    cw_df = fuzzy_join_lev_jw(
        left=f1_schools_df,
        right=rev_schools_df,
        left_on="f1_instname_clean",
        right_on="rev_instname_clean",
        threshold=threshold,
        top_n=3,           # keep up to 3 Revelio candidates per F1 school
        jw_top_k=30,
        keep_subscores=True,
    )

    elapsed = time.perf_counter() - t0
    print(f"  Fuzzy join done: {len(cw_df):,} raw matches ({_fmt_elapsed(elapsed)})")

    # ------------------------------------------------------------------
    # 4. Post-process: rename columns, flag ambiguous matches
    # ------------------------------------------------------------------
    print("\n[4/4] Post-processing crosswalk...")

    # Rename the suffixed columns back to clean names
    col_map = {
        "f1_school_name_left": "f1_school_name",
        "f1_instname_clean_left": "f1_instname_clean",
        "n_persons_left": "n_f1_persons",
        "rev_university_raw_right": "rev_university_raw",
        "rev_instname_clean_right": "rev_instname_clean",
        "n_users_right": "n_rev_users",
        "_score": "match_score",
        "_lev_sim": "lev_sim",
        "_jw_sim": "jw_sim",
    }
    cw_df = cw_df.rename(columns=col_map)

    # Drop fuzzy-join internal columns (_n suffix from normalization)
    drop_cols = [c for c in cw_df.columns if c.endswith("_n_left") or c.endswith("_n_right")]
    cw_df = cw_df.drop(columns=drop_cols, errors="ignore")

    # Flag ambiguous matches: F1 school has ≥2 Revelio matches
    match_counts = cw_df.groupby("f1_school_name")["rev_university_raw"].transform("count")
    cw_df["match_ambiguous_ind"] = (match_counts > 1).astype(int)

    # Keep only desired columns
    keep_cols = [
        "f1_school_name", "f1_instname_clean", "n_f1_persons",
        "rev_university_raw", "rev_instname_clean", "n_rev_users",
        "match_score", "lev_sim", "jw_sim", "match_ambiguous_ind",
    ]
    cw_df = cw_df[[c for c in keep_cols if c in cw_df.columns]]

    # Summary stats
    n_f1_matched = cw_df["f1_school_name"].nunique()
    n_f1_total = f1_schools_df["f1_school_name"].nunique()
    n_rev_matched = cw_df["rev_university_raw"].nunique()
    n_f1_ambiguous = cw_df[cw_df["match_ambiguous_ind"] == 1]["f1_school_name"].nunique()

    print(f"  F1 schools matched:     {n_f1_matched:,} / {n_f1_total:,} "
          f"({100*n_f1_matched/max(1, n_f1_total):.1f}%)")
    print(f"  Revelio schools matched: {n_rev_matched:,}")
    print(f"  Ambiguous F1 schools:    {n_f1_ambiguous:,} "
          f"({100*n_f1_ambiguous/max(1, n_f1_matched):.1f}% of matched)")
    print(f"  Score distribution:")
    print(cw_df["match_score"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

    # Sample the top and bottom of the crosswalk for manual inspection
    print("\n  Top matches (highest score):")
    print(cw_df.nlargest(10, "match_score")[
        ["f1_school_name", "rev_university_raw", "match_score", "match_ambiguous_ind"]
    ].to_string(index=False))

    print("\n  Lowest accepted matches:")
    print(cw_df.nsmallest(10, "match_score")[
        ["f1_school_name", "rev_university_raw", "match_score", "match_ambiguous_ind"]
    ].to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Write parquet
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    cw_df.to_parquet(out_path, index=False)

    total_elapsed = time.perf_counter() - t_total
    print(f"\n[done] Wrote crosswalk: {out_path} ({_fmt_elapsed(total_elapsed)})")
    print(f"       {len(cw_df):,} rows")

    return cw_df


# ---------------------------------------------------------------------------
# Entry point (iPython-compatible: just call the function)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # When run as a script (not required but convenient)
    build_school_crosswalk()
