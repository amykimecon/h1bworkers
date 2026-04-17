"""
deps_f1_school_crosswalk.py
===========================
Pre-requisite dependency for the F1 individual merge pipeline.

Builds the F1 FOIA school-name → Revelio university crosswalk used by
`f1_indiv_merge.py`.

Default behavior:
  - Keep the existing lightweight school-name crosswalk build unchanged:
    direct fuzzy join between F1 school names and Revelio `university_raw`.
  - Additionally, when the upstream company_shift_share/IPEDS-backed files
    are present, emit a second rich row-level artifact preserving
    f1_row_num / UNITID / campus-location-aware school resolution.

The fallback path still avoids requiring IPEDS coverage for every school, so
non-US schools can be matched when the direct fuzzy join is used.

Approach:
  1. Extract distinct F1 school names and clean them
  2. Extract distinct Revelio university_raw names and clean them
  3. Run fuzzy_join_lev_jw() (Levenshtein + Jaro-Winkler)
  4. Save the lightweight crosswalk parquet
  5. If available, also derive the rich row-level resolution artifact from
     the pre-built company_shift_share inputs

Outputs:
  1. f1_rev_school_crosswalk_{run_tag}.parquet
     Columns: f1_school_name, f1_instname_clean, rev_university_raw,
              rev_instname_clean, match_score, match_ambiguous_ind, ...
     The lightweight artifact is collapsed to cleaned Revelio school families
     (`rev_instname_clean`), with a representative raw school name retained for
     inspection and weak family alternatives pruned by score gap.
  2. f1_rev_school_resolution_{run_tag}.parquet
     Rich row-level artifact preserving f1_row_num / UNITID / location-aware
     school resolution signals for conservative candidate blocking in
     f1_indiv_merge.py.

Usage (from iPython):
    import importlib, sys
    sys.path.insert(0, '/home/yk0581/h1bworkers/code/f1_indiv_merge')
    import deps_f1_school_crosswalk
    importlib.reload(deps_f1_school_crosswalk)
    deps_f1_school_crosswalk.build_school_crosswalk()
"""

import os
import sys
import time
from builtins import print as _print
from functools import partial

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))

import f1_indiv_merge_config as cfg  # noqa: E402 – loaded after path setup
from helpers import fuzzy_join_lev_jw  # noqa: E402

# Flush progress output immediately so redirected logs stay live.
print = partial(_print, flush=True)


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


def _f1_school_person_counts(con) -> pd.DataFrame:
    return con.sql(f"""
        SELECT
            school_name AS f1_school_name,
            COUNT(DISTINCT person_id) AS n_f1_persons
        FROM read_parquet('{cfg.F1_FOIA_PARQUET}')
        WHERE school_name IS NOT NULL
          AND TRIM(school_name) != ''
        GROUP BY school_name
    """).df()


def _build_crosswalk_from_canonical(
    canonical_school_path: str,
    con,
) -> pd.DataFrame:
    school_esc = canonical_school_path.replace("'", "''")
    cw_df = con.sql(f"""
        SELECT DISTINCT
            f1_school_name,
            f1_instname_clean,
            rev_university_raw,
            rev_instname_clean,
            COALESCE(CAST(n_revelio_institution_records AS BIGINT), CAST(n_revelio_inst_raw_variants AS BIGINT), 0) AS n_rev_users,
            COALESCE(match_score, 1.0) AS match_score,
            COALESCE(lev_sim, match_score, 1.0) AS lev_sim,
            COALESCE(jw_sim, match_score, 1.0) AS jw_sim,
            COALESCE(CAST(match_ambiguous_ind AS INTEGER), 0) AS match_ambiguous_ind,
            COALESCE(CAST(school_match_rank AS BIGINT), 1) AS school_match_rank,
            COALESCE(match_score_gap_from_top, 0.0) AS match_score_gap_from_top,
            COALESCE(CAST(n_revelio_inst_raw_variants AS BIGINT), 1) AS n_rev_university_raw_variants
        FROM read_parquet('{school_esc}')
        WHERE f1_school_name IS NOT NULL
          AND rev_instname_clean IS NOT NULL
    """).df()
    if cw_df.empty:
        cw_df["n_f1_persons"] = pd.Series(dtype="int64")
        return cw_df

    f1_counts = _f1_school_person_counts(con)
    cw_df = cw_df.merge(f1_counts, on="f1_school_name", how="left")
    keep_cols = [
        "f1_school_name", "f1_instname_clean", "n_f1_persons",
        "rev_university_raw", "rev_instname_clean", "n_rev_users",
        "match_score", "lev_sim", "jw_sim", "match_ambiguous_ind",
        "school_match_rank", "match_score_gap_from_top", "n_rev_university_raw_variants",
    ]
    return cw_df[[c for c in keep_cols if c in cw_df.columns]].sort_values(
        ["f1_school_name", "school_match_rank", "rev_instname_clean"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def _build_resolution_from_canonical(
    canonical_resolution_path: str,
    con,
) -> pd.DataFrame:
    resolution_esc = canonical_resolution_path.replace("'", "''")
    return con.sql(f"""
        SELECT *
        FROM read_parquet('{resolution_esc}')
        ORDER BY f1_school_name, f1_row_num, rev_instname_clean
    """).df()


# ---------------------------------------------------------------------------
# Fast path: derive crosswalk from existing company_shift_share files
# ---------------------------------------------------------------------------

def _build_resolution_from_existing(
    f1_unitid_path: str,
    rev_ipeds_foia_path: str,
    con,
) -> pd.DataFrame:
    """Derive rich row-level F1 → Revelio school resolution from fast-path files.

    Joins f1_inst_unitid_crosswalk (school_name → f1_row_num) with
    revelio_ipeds_foia_inst_crosswalk (f1_row_num → university_raw) to
    preserve the row-level fields needed for campus-location-aware blocking in
    the merge pipeline.

    Direct matches (rev_matchtype = 'direct'/'direct_tie') have NULL rev_jw_score
    and are assigned match_score = 1.0. Fuzzy matches are kept regardless of score;
    match_score propagates the uncertainty as a multiplier on total_score downstream.

    Coverage: ~87% of F1 FOIA school names (IPEDS-matched US schools only).
    """
    f1_esc = f1_unitid_path.replace("'", "''")
    rev_esc = rev_ipeds_foia_path.replace("'", "''")

    return con.sql(f"""
        SELECT DISTINCT
            f.school_name            AS f1_school_name,
            f.f1_row_num,
            f.UNITID,
            f.f1_instname_clean,
            f.f1_city_clean,
            f.f1_state_clean,
            f.f1_zip_clean,
            cw.university_raw        AS rev_university_raw,
            cw.rev_instname_clean,
            COALESCE(cw.rev_jw_score, 1.0) AS school_match_score,
            cw.rev_matchtype,
            cw.match_group,
            cw.rev_match_source
        FROM read_parquet('{f1_esc}') AS f
        JOIN read_parquet('{rev_esc}') AS cw ON f.f1_row_num = cw.f1_row_num
        WHERE cw.university_raw IS NOT NULL
        ORDER BY f.school_name, school_match_score DESC, cw.university_raw
    """).df()


def _build_crosswalk_from_resolution_df(resolution_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the rich resolution artifact to the legacy lightweight schema."""
    cw_df = (
        resolution_df[[
            "f1_school_name",
            "f1_instname_clean",
            "rev_university_raw",
            "rev_instname_clean",
            "school_match_score",
        ]]
        .drop_duplicates()
        .rename(columns={"school_match_score": "match_score"})
        .sort_values(["f1_school_name", "match_score", "rev_university_raw"],
                     ascending=[True, False, True])
        .reset_index(drop=True)
    )
    match_counts = cw_df.groupby("f1_school_name")["rev_university_raw"].transform("count")
    cw_df["match_ambiguous_ind"] = (match_counts > 1).astype(int)
    return cw_df


def _build_school_family_query(source_tab: str, score_gap: float) -> str:
    """Collapse raw crosswalk rows to cleaned-school families and prune weak ones."""
    return f"""
    WITH base AS (
        SELECT
            f1_school_name,
            f1_instname_clean,
            rev_university_raw,
            rev_instname_clean,
            COALESCE(CAST(n_rev_users AS BIGINT), 0) AS n_rev_users,
            match_score,
            COALESCE(lev_sim, match_score) AS lev_sim,
            COALESCE(jw_sim, match_score)  AS jw_sim
        FROM {source_tab}
        WHERE f1_school_name IS NOT NULL
          AND rev_instname_clean IS NOT NULL
          AND trim(rev_instname_clean) != ''
    ),
    raw_ranked AS (
        SELECT *,
            ROW_NUMBER() OVER(
                PARTITION BY f1_school_name, rev_instname_clean
                ORDER BY n_rev_users DESC, match_score DESC, rev_university_raw
            ) AS raw_variant_rank
        FROM base
    ),
    families AS (
        SELECT
            f1_school_name,
            MAX(f1_instname_clean) AS f1_instname_clean,
            rev_instname_clean,
            MAX(CASE WHEN raw_variant_rank = 1 THEN rev_university_raw END) AS rev_university_raw,
            SUM(n_rev_users) AS n_rev_users,
            COUNT(DISTINCT rev_university_raw) AS n_rev_university_raw_variants,
            MAX(match_score) AS match_score,
            MAX(lev_sim)     AS lev_sim,
            MAX(jw_sim)      AS jw_sim
        FROM raw_ranked
        GROUP BY f1_school_name, rev_instname_clean
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER(
                PARTITION BY f1_school_name
                ORDER BY match_score DESC, n_rev_users DESC, rev_instname_clean
            ) AS school_match_rank,
            MAX(match_score) OVER(PARTITION BY f1_school_name) AS top_match_score
        FROM families
    ),
    filtered AS (
        SELECT *,
            top_match_score - match_score AS match_score_gap_from_top,
            CASE
                WHEN school_match_rank = 1 THEN 1
                WHEN match_score >= top_match_score - {score_gap} THEN 1
                ELSE 0
            END AS match_keep_ind
        FROM ranked
    ),
    kept AS (
        SELECT *,
            COUNT(*) OVER(PARTITION BY f1_school_name) AS kept_family_count
        FROM filtered
        WHERE match_keep_ind = 1
    )
    SELECT
        f1_school_name,
        f1_instname_clean,
        rev_university_raw,
        rev_instname_clean,
        n_rev_users,
        match_score,
        lev_sim,
        jw_sim,
        CASE WHEN kept_family_count > 1 THEN 1 ELSE 0 END AS match_ambiguous_ind,
        school_match_rank,
        match_score_gap_from_top,
        n_rev_university_raw_variants
    FROM kept
    ORDER BY f1_school_name, school_match_rank, rev_instname_clean
    """


def _collapse_school_crosswalk_df(cw_df: pd.DataFrame, score_gap: float, con) -> pd.DataFrame:
    """Collapse raw crosswalk rows to kept cleaned-school families."""
    if cw_df.empty:
        out = cw_df.copy()
        out["school_match_rank"] = pd.Series(dtype="int64")
        out["match_score_gap_from_top"] = pd.Series(dtype="float64")
        out["n_rev_university_raw_variants"] = pd.Series(dtype="int64")
        return out

    con.register("_cw_raw_df", cw_df)
    return con.sql(_build_school_family_query("_cw_raw_df", score_gap)).df()


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_school_crosswalk(overwrite: bool = None, con=None) -> pd.DataFrame:
    """Build the lightweight school crosswalk and, when available, the rich artifact."""
    out_path = cfg.F1_REV_SCHOOL_CROSSWALK_PARQUET
    resolution_out_path = cfg.F1_REV_SCHOOL_RESOLUTION_PARQUET
    canonical_school_path = cfg.F1_REVELIO_SCHOOL_CROSSWALK_CANONICAL_PARQUET
    canonical_resolution_path = cfg.F1_REVELIO_IPEDS_RESOLUTION_PARQUET
    if overwrite is None:
        overwrite = cfg.BUILD_OVERWRITE
    threshold = cfg.BUILD_SCHOOL_MATCH_THRESHOLD
    score_gap = cfg.BUILD_SCHOOL_AMBIGUITY_SCORE_GAP

    if con is None:
        con = duckdb.connect()

    # ------------------------------------------------------------------
    # Fast path: derive from pre-built company_shift_share files
    # ------------------------------------------------------------------
    f1_unitid_path = cfg.F1_INST_UNITID_CROSSWALK_PARQUET
    rev_ipeds_foia_path = cfg.REVELIO_IPEDS_FOIA_INST_CROSSWALK_PARQUET
    use_canonical = bool(canonical_school_path) and os.path.exists(canonical_school_path)
    use_existing = (
        bool(f1_unitid_path) and os.path.exists(f1_unitid_path)
        and bool(rev_ipeds_foia_path) and os.path.exists(rev_ipeds_foia_path)
    )

    if os.path.exists(out_path) and not overwrite:
        if use_canonical and resolution_out_path and canonical_resolution_path and os.path.exists(canonical_resolution_path) and not os.path.exists(resolution_out_path):
            print("[crosswalk] Lightweight crosswalk exists; building missing canonical rich resolution artifact only.")
            resolution_df = _build_resolution_from_canonical(canonical_resolution_path, con)
            os.makedirs(os.path.dirname(resolution_out_path), exist_ok=True)
            if os.path.exists(resolution_out_path):
                os.remove(resolution_out_path)
            resolution_df.to_parquet(resolution_out_path, index=False)
            print(f"[crosswalk] Wrote rich resolution artifact: {resolution_out_path}")
        elif use_existing and resolution_out_path and not os.path.exists(resolution_out_path):
            print("[crosswalk] Lightweight crosswalk exists; building missing rich resolution artifact only.")
            resolution_df = _build_resolution_from_existing(f1_unitid_path, rev_ipeds_foia_path, con)
            os.makedirs(os.path.dirname(resolution_out_path), exist_ok=True)
            if os.path.exists(resolution_out_path):
                os.remove(resolution_out_path)
            resolution_df.to_parquet(resolution_out_path, index=False)
            print(f"[crosswalk] Wrote rich resolution artifact: {resolution_out_path}")
        else:
            print(f"[crosswalk] Skipping — file exists: {out_path}")
        return pd.read_parquet(out_path)

    t_total = time.perf_counter()
    print("=" * 70)
    print("deps_f1_school_crosswalk: building F1 → Revelio school crosswalk")
    print(f"  F1 FOIA:         {cfg.F1_FOIA_PARQUET}")
    rev_educ_path = cfg.choose_path(cfg.REV_EDUC_LONG_PARQUET, cfg.REV_EDUC_LONG_PARQUET_LEGACY)
    print(f"  Revelio educ:    {rev_educ_path}")
    print(f"  Match threshold: {threshold}")
    print(f"  Ambig gap keep:  {score_gap}")
    print(f"  Output:          {out_path}")
    if use_canonical and resolution_out_path:
        print(f"  Canonical path:  {canonical_school_path}")
        if canonical_resolution_path:
            print(f"  Resolution src:  {canonical_resolution_path}")
        print(f"  Resolution out:  {resolution_out_path}")
    elif use_existing and resolution_out_path:
        print(f"  Resolution out:  {resolution_out_path}")
    print("=" * 70)

    if use_canonical:
        print("\n[fast-path] Using canonical shared-IPEDS school mappings.")
        t0 = time.perf_counter()
        cw_df = _build_crosswalk_from_canonical(canonical_school_path, con)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            os.remove(out_path)
        cw_df.to_parquet(out_path, index=False)

        n_f1_matched = cw_df["f1_school_name"].nunique() if not cw_df.empty else 0
        n_rev_matched = cw_df["rev_instname_clean"].nunique() if not cw_df.empty else 0
        n_ambiguous = cw_df.loc[cw_df["match_ambiguous_ind"] == 1, "f1_school_name"].nunique() if not cw_df.empty else 0
        print(f"  Wrote {len(cw_df):,} canonical school-family rows "
              f"({n_f1_matched:,} F1 schools; {n_rev_matched:,} Revelio families; "
              f"{n_ambiguous:,} ambiguous F1 schools) "
              f"({_fmt_elapsed(time.perf_counter() - t0)})")

        if resolution_out_path and canonical_resolution_path and os.path.exists(canonical_resolution_path):
            print("\n[fast-path] Copying canonical row-level resolution artifact...")
            t0 = time.perf_counter()
            resolution_df = _build_resolution_from_canonical(canonical_resolution_path, con)
            os.makedirs(os.path.dirname(resolution_out_path), exist_ok=True)
            if os.path.exists(resolution_out_path):
                os.remove(resolution_out_path)
            resolution_df.to_parquet(resolution_out_path, index=False)
            print(f"  Wrote {len(resolution_df):,} rich rows "
                  f"({resolution_df['f1_row_num'].nunique():,} F1 rows; "
                  f"{resolution_df['f1_school_name'].nunique():,} F1 schools) "
                  f"({_fmt_elapsed(time.perf_counter() - t0)})")
        elif resolution_out_path and os.path.exists(resolution_out_path):
            os.remove(resolution_out_path)
            print(f"  Removed stale rich resolution artifact: {resolution_out_path}")

        total_elapsed = time.perf_counter() - t_total
        print(f"\n[done] Wrote crosswalk: {out_path} ({_fmt_elapsed(total_elapsed)})")
        print(f"       {len(cw_df):,} rows")
        return cw_df

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

    raw_match_count = len(cw_df)
    raw_family_count = cw_df[["f1_school_name", "rev_instname_clean"]].drop_duplicates().shape[0]
    cw_df = _collapse_school_crosswalk_df(cw_df, score_gap=score_gap, con=con)
    cw_df = cw_df.merge(
        f1_schools_df[["f1_school_name", "n_persons"]]
        .rename(columns={"n_persons": "n_f1_persons"}),
        on="f1_school_name",
        how="left",
    )

    keep_cols = [
        "f1_school_name", "f1_instname_clean", "n_f1_persons",
        "rev_university_raw", "rev_instname_clean", "n_rev_users",
        "match_score", "lev_sim", "jw_sim", "match_ambiguous_ind",
        "school_match_rank", "match_score_gap_from_top", "n_rev_university_raw_variants",
    ]
    cw_df = cw_df[[c for c in keep_cols if c in cw_df.columns]]

    # Summary stats
    n_f1_matched = cw_df["f1_school_name"].nunique()
    n_f1_total = f1_schools_df["f1_school_name"].nunique()
    n_rev_matched = cw_df["rev_instname_clean"].nunique()
    n_f1_ambiguous = cw_df[cw_df["match_ambiguous_ind"] == 1]["f1_school_name"].nunique()
    n_kept_families = len(cw_df)

    print(f"  F1 schools matched:     {n_f1_matched:,} / {n_f1_total:,} "
          f"({100*n_f1_matched/max(1, n_f1_total):.1f}%)")
    print(f"  Raw crosswalk rows:     {raw_match_count:,}")
    print(f"  Raw cleaned families:   {raw_family_count:,}")
    print(f"  Kept cleaned families:  {n_kept_families:,}")
    print(f"  Revelio families kept:  {n_rev_matched:,}")
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
    if use_existing and resolution_out_path:
        print("\n[5/5] Building rich row-level school-resolution artifact...")
        t0 = time.perf_counter()
        resolution_df = _build_resolution_from_existing(f1_unitid_path, rev_ipeds_foia_path, con)
        os.makedirs(os.path.dirname(resolution_out_path), exist_ok=True)
        if os.path.exists(resolution_out_path):
            os.remove(resolution_out_path)
        resolution_df.to_parquet(resolution_out_path, index=False)
        print(f"  Wrote {len(resolution_df):,} rich rows "
              f"({resolution_df['f1_row_num'].nunique():,} F1 rows; "
              f"{resolution_df['f1_school_name'].nunique():,} F1 schools) "
              f"({_fmt_elapsed(time.perf_counter() - t0)})")
    elif resolution_out_path and os.path.exists(resolution_out_path):
        os.remove(resolution_out_path)
        print(f"  Removed stale rich resolution artifact: {resolution_out_path}")

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
