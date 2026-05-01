"""
deps_f1_employer_crosswalk.py
=============================
Pre-requisite dependency for the F1 individual merge pipeline.

Builds a lookup table mapping row-level F1 OPT employer records to candidate
Revelio company IDs (rcid), used as the join key in Branch B
(school + employer matching) of f1_indiv_merge.py.

Approach:
  1. Extract distinct FOIA employer rows at the original employer-name + location
     grain and attach `foia_row_uid` / `foia_firm_uid` from the staged external
     employer crosswalk.
  2. Use the staged final crosswalk as a strong path only when a FOIA firm has
     exactly one preferred rcid.
  3. For uncovered or ambiguous rows, fall back to clean exact and optional
     fuzzy matching against Revelio positions.

Output: f1_opt_employer_lookup_{run_tag}.parquet
  Columns:
    employer_name, employer_name_clean,
    employer_city_clean, employer_state_clean, employer_zip_clean,
    foia_row_uid, foia_firm_uid,
    rcid, matched_company_name, match_type, match_score,
    lookup_rcid_count, lookup_rcid_ambiguous_ind, lookup_has_direct_ind
  One row per FOIA employer row x rcid candidate.

Usage (from iPython):
    import importlib, sys
    sys.path.insert(0, '/home/yk0581/h1bworkers/code/f1_indiv_merge')
    import deps_f1_employer_crosswalk
    importlib.reload(deps_f1_employer_crosswalk)
    deps_f1_employer_crosswalk.build_employer_crosswalk()
"""

import math
import os
import sys
import time
from builtins import print as _print
from functools import partial

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import distance, process

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import f1_indiv_merge_config as cfg  # noqa: E402
from employer_entity_sql import (  # noqa: E402
    sql_clean_company_name_expr,
    sql_clean_zip_expr,
    sql_normalize_expr,
    sql_state_name_to_abbr_expr,
)
from helpers import fuzzy_join_lev_jw  # noqa: E402
from src.duckdb_runtime import get_duckdb_memory_limit_sql_literal  # noqa: E402

# Flush progress output immediately so redirected logs stay live.
print = partial(_print, flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOOKUP_BASE_COLS = [
    "employer_name",
    "employer_name_clean",
    "employer_city_clean",
    "employer_state_clean",
    "employer_zip_clean",
    "foia_row_uid",
    "foia_firm_uid",
    "rcid",
    "matched_company_name",
    "match_type",
    "match_score",
]
LOOKUP_FLAG_COLS = [
    "lookup_rcid_count",
    "lookup_rcid_ambiguous_ind",
    "lookup_has_direct_ind",
]
LOOKUP_COLS = LOOKUP_BASE_COLS + LOOKUP_FLAG_COLS

_ROW_ID_COL = "_lookup_row_id"
_FUZZY_BLOCK_PREFIX_LEN = 5
_FUZZY_MAX_PAIRS_PER_CHUNK = 10_000_000


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _empty_lookup_df() -> pd.DataFrame:
    return pd.DataFrame(columns=LOOKUP_COLS)


def _coerce_nullable_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.array(df[col], dtype="Int64")
    return df


def _build_lookup_row_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["employer_name"].fillna("").astype(str).str.lower().str.strip() + "||"
        + df["employer_city_clean"].fillna("").astype(str).str.strip() + "||"
        + df["employer_state_clean"].fillna("").astype(str).str.strip() + "||"
        + df["employer_zip_clean"].fillna("").astype(str).str.strip() + "||"
        + df["foia_row_uid"].fillna("").astype(str) + "||"
        + df["foia_firm_uid"].fillna("").astype(str)
    )


def _build_block_key(series: pd.Series, prefix_len: int = _FUZZY_BLOCK_PREFIX_LEN) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str[:prefix_len]


def _chunked_fuzzy_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    threshold: float,
    top_n: int,
    use_jw_only: bool,
    jw_top_k: int,
    keep_subscores: bool,
    block_key: str,
    max_pairs_per_chunk: int = _FUZZY_MAX_PAIRS_PER_CHUNK,
) -> pd.DataFrame:
    if left.empty or right.empty:
        return pd.DataFrame()

    left_blocks = left[left[block_key].notna() & left[block_key].ne("")].copy()
    right_blocks = right[right[block_key].notna() & right[block_key].ne("")].copy()
    if left_blocks.empty or right_blocks.empty:
        return pd.DataFrame()

    groups = sorted(set(left_blocks[block_key].unique()) & set(right_blocks[block_key].unique()))
    if not groups:
        return pd.DataFrame()

    out_parts = []
    n_groups = len(groups)

    for i, group in enumerate(groups, start=1):
        left_block = left_blocks[left_blocks[block_key] == group].reset_index(drop=True)
        right_block = right_blocks[right_blocks[block_key] == group].reset_index(drop=True)
        if left_block.empty or right_block.empty:
            continue

        pair_count = len(left_block) * len(right_block)
        left_chunk_size = max(1, math.floor(max_pairs_per_chunk / max(1, len(right_block))))
        n_chunks = max(1, math.ceil(len(left_block) / left_chunk_size))

        if n_chunks > 1:
            print(
                f"    block '{group}': {len(left_block):,} left x {len(right_block):,} right "
                f"({pair_count:,} pairs) -> {n_chunks:,} chunks"
            )

        for start in range(0, len(left_block), left_chunk_size):
            left_chunk = left_block.iloc[start:start + left_chunk_size].copy()
            if use_jw_only:
                lv = left_chunk[left_on].fillna("").astype(str).tolist()
                rv = right_block[right_on].fillna("").astype(str).tolist()
                jw_sim = process.cdist(
                    lv,
                    rv,
                    scorer=distance.JaroWinkler.normalized_similarity,
                    workers=-1,
                )
                m_left, m_right = jw_sim.shape
                if m_left == 0 or m_right == 0:
                    continue

                if top_n == 1:
                    best_cols = jw_sim.argmax(axis=1)
                    best_scores = jw_sim[np.arange(m_left), best_cols]
                    keep = best_scores >= threshold
                    if keep.any():
                        left_sel = left_chunk.loc[keep].reset_index(drop=True)
                        right_sel = right_block.iloc[best_cols[keep]].reset_index(drop=True)
                        matched = pd.concat(
                            [left_sel.add_suffix("_left"), right_sel.add_suffix("_right")],
                            axis=1,
                        )
                        matched["_score"] = best_scores[keep]
                        if keep_subscores:
                            matched["_jw_sim"] = best_scores[keep]
                    else:
                        matched = pd.DataFrame()
                else:
                    k = min(max(1, top_n), m_right)
                    top_idx = np.argpartition(-jw_sim, kth=k - 1, axis=1)[:, :k]
                    rows, cols = np.indices(top_idx.shape)
                    top_scores = jw_sim[rows, top_idx]
                    keep_mask = top_scores >= threshold
                    if keep_mask.any():
                        li = left_chunk.index.values[rows[keep_mask]]
                        ri = right_block.index.values[top_idx[keep_mask]]
                        left_multi = left_chunk.loc[li].reset_index(drop=True)
                        right_multi = right_block.loc[ri].reset_index(drop=True)
                        matched = pd.concat(
                            [left_multi.add_suffix("_left"), right_multi.add_suffix("_right")],
                            axis=1,
                        )
                        matched["_score"] = top_scores[keep_mask]
                        if keep_subscores:
                            matched["_jw_sim"] = top_scores[keep_mask]
                    else:
                        matched = pd.DataFrame()
            else:
                matched = fuzzy_join_lev_jw(
                    left=left_chunk,
                    right=right_block,
                    left_on=left_on,
                    right_on=right_on,
                    threshold=threshold,
                    top_n=top_n,
                    jw_top_k=jw_top_k,
                    keep_subscores=keep_subscores,
                    normalize=False,
                )
            if not matched.empty:
                out_parts.append(matched)

        if i % 200 == 0 or i == n_groups:
            print(f"    processed {i:,} / {n_groups:,} fuzzy blocks")

    if not out_parts:
        return pd.DataFrame()
    return pd.concat(out_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_employer_crosswalk(overwrite: bool = None, do_fuzzy: bool | None = None, con=None) -> pd.DataFrame:
    """Build row-level F1 OPT employer -> Revelio rcid lookup table."""
    out_path = cfg.F1_OPT_EMPLOYER_LOOKUP_PARQUET
    if overwrite is None:
        overwrite = cfg.BUILD_OVERWRITE
    threshold = cfg.BUILD_EMPLOYER_MATCH_THRESHOLD
    use_jw_only = cfg.BUILD_EMPLOYER_MATCH_JW_ONLY
    if do_fuzzy is None:
        do_fuzzy = cfg.BUILD_EMPLOYER_ENABLE_FUZZY_MATCHING

    if not out_path:
        raise ValueError("f1_opt_employer_lookup_parquet not configured in f1_indiv_merge.yaml")

    if os.path.exists(out_path) and not overwrite:
        print(f"[employer crosswalk] Skipping — file exists: {out_path}")
        return pd.read_parquet(out_path)

    if con is None:
        con = duckdb.connect()
        con.execute("SET threads = 8")
        con.execute(f"SET memory_limit = '{get_duckdb_memory_limit_sql_literal()}'")

    crosswalk_path = cfg.F1_EMPLOYER_ENTITY_CROSSWALK_PARQUET
    rev_pos_path = cfg.choose_path(cfg.REV_POS_PARQUET, cfg.REV_POS_PARQUET_LEGACY)
    f1_foia_path = cfg.F1_FOIA_PARQUET

    t_total = time.perf_counter()
    print("=" * 70)
    print("deps_f1_employer_crosswalk: building row-level F1 OPT employer -> rcid lookup")
    print(f"  Staged final crosswalk:    {crosswalk_path}")
    print(f"  F1 FOIA:                   {f1_foia_path}")
    print(f"  Revelio positions:         {rev_pos_path}")
    print(f"  Fuzzy enabled:             {do_fuzzy}")
    print(f"  Fuzzy metric:              {'jw_only' if use_jw_only else 'lev_jw'}")
    print(f"  Fuzzy threshold:           {threshold}")
    print(f"  Output:                    {out_path}")
    print("=" * 70)

    print("\n[1/4] Extracting distinct F1 OPT employer rows and staged firm ids...")
    t0 = time.perf_counter()

    f1_esc = f1_foia_path.replace("'", "''")
    crosswalk_join = ""
    crosswalk_cols = (
        "NULL::VARCHAR AS foia_row_uid, "
        "NULL::VARCHAR AS foia_firm_uid, "
        "NULL::BIGINT AS preferred_rcid, "
        "NULL::VARCHAR AS preferred_company_name, "
        "NULL::VARCHAR AS preferred_match_source, "
        "0::BIGINT AS staged_match_count"
    )
    use_crosswalk = bool(crosswalk_path) and os.path.exists(crosswalk_path)
    if use_crosswalk:
        staged_df = pd.read_parquet(crosswalk_path).copy()
        staged_df["foia_row_uid"] = staged_df.get("foia_row_uid", pd.Series(dtype="string")).astype("string")
        staged_df["foia_firm_uid"] = staged_df.get("foia_firm_uid", pd.Series(dtype="string")).astype("string")
        staged_df["preferred_rcid"] = pd.to_numeric(staged_df.get("preferred_rcid"), errors="coerce").astype("Int64")
        staged_df["preferred_company_name"] = staged_df.get("preferred_company_name", pd.Series(dtype="string")).astype("string")
        staged_df["preferred_match_source"] = staged_df.get("preferred_match_source", pd.Series(dtype="string")).astype("string")
        staged_df["staged_match_count"] = staged_df.get("matched_rcids", pd.Series(dtype="object")).map(
            lambda v: len(v) if isinstance(v, (list, tuple, set)) else (len(v.tolist()) if hasattr(v, "tolist") and not isinstance(v, str) else (0 if v is None or (isinstance(v, float) and pd.isna(v)) else 1))
        )
        staged_df = staged_df[
            [
                "employer_name",
                "f1_empname_clean",
                "f1_city_clean",
                "f1_state_clean",
                "f1_zip_clean",
                "foia_row_uid",
                "foia_firm_uid",
                "preferred_rcid",
                "preferred_company_name",
                "preferred_match_source",
                "staged_match_count",
            ]
        ].drop_duplicates(
            subset=["employer_name", "f1_empname_clean", "f1_city_clean", "f1_state_clean", "f1_zip_clean"],
            keep="first",
        )
        con.register("staged_employer_crosswalk_df", staged_df)
        crosswalk_join = """
        LEFT JOIN staged_employer_crosswalk_df AS em
          ON lower(trim(base.employer_name)) = lower(trim(em.employer_name))
         AND COALESCE(base.employer_city_clean, '') = COALESCE(em.f1_city_clean, '')
         AND COALESCE(base.employer_state_clean, '') = COALESCE(em.f1_state_clean, '')
         AND COALESCE(base.employer_zip_clean, '') = COALESCE(em.f1_zip_clean, '')
        """
        crosswalk_cols = (
            "em.foia_row_uid, "
            "em.foia_firm_uid, "
            "CAST(em.preferred_rcid AS BIGINT) AS preferred_rcid, "
            "em.preferred_company_name, "
            "em.preferred_match_source, "
            "CAST(em.staged_match_count AS BIGINT) AS staged_match_count"
        )

    f1_employers_df = con.sql(f"""
        WITH base AS (
            SELECT
                employer_name,
                {sql_clean_company_name_expr('employer_name')} AS employer_name_clean,
                {sql_normalize_expr('employer_city')} AS employer_city_clean,
                {sql_state_name_to_abbr_expr('employer_state')} AS employer_state_clean,
                {sql_clean_zip_expr('employer_zip_code')} AS employer_zip_clean,
                COUNT(DISTINCT person_id) AS n_persons
            FROM read_parquet('{f1_esc}')
            WHERE employer_name IS NOT NULL
              AND TRIM(employer_name) != ''
            GROUP BY 1, 2, 3, 4, 5
        )
        SELECT
            base.*,
            {crosswalk_cols}
        FROM base
        {crosswalk_join}
        ORDER BY n_persons DESC, base.employer_name
    """).df()
    f1_employers_df = _coerce_nullable_int(f1_employers_df, ["preferred_rcid"])
    f1_employers_df[_ROW_ID_COL] = _build_lookup_row_id(f1_employers_df)

    n_f1_total = len(f1_employers_df)
    n_row_uids = int(f1_employers_df["foia_row_uid"].notna().sum())
    n_firms = int(f1_employers_df["foia_firm_uid"].nunique(dropna=True))
    print(
        f"  {n_f1_total:,} distinct F1 OPT employer rows "
        f"({f1_employers_df['n_persons'].sum():,.0f} total person-records, {_fmt_elapsed(time.perf_counter() - t0)})"
    )
    if use_crosswalk:
        print(f"  FOIA ids staged: {n_row_uids:,} rows across {n_firms:,} upstream FOIA firms")
    else:
        print("  WARNING: staged final employer crosswalk not found; foia_row_uid/foia_firm_uid will be null in fallback rows.")

    con.register("f1_employer_rows_df", f1_employers_df)

    print("\n[2/4] Applying staged external employer matches (strong path)...")
    t0 = time.perf_counter()

    prebuilt_results = _empty_lookup_df()
    if use_crosswalk:
        strong_mask = (
            f1_employers_df["preferred_rcid"].notna()
            & pd.to_numeric(f1_employers_df["staged_match_count"], errors="coerce").fillna(0).eq(1)
        )
        if strong_mask.any():
            prebuilt_results = f1_employers_df.loc[
                strong_mask,
                [
                    "employer_name",
                    "employer_name_clean",
                    "employer_city_clean",
                    "employer_state_clean",
                    "employer_zip_clean",
                    "foia_row_uid",
                    "foia_firm_uid",
                    "preferred_rcid",
                    "preferred_company_name",
                ],
            ].copy()
            prebuilt_results = prebuilt_results.rename(
                columns={
                    "preferred_rcid": "rcid",
                    "preferred_company_name": "matched_company_name",
                }
            )
            prebuilt_results["match_type"] = "staged_direct"
            prebuilt_results["match_score"] = 1.0
            prebuilt_results = prebuilt_results[LOOKUP_BASE_COLS].copy()
            prebuilt_results = _coerce_nullable_int(prebuilt_results, ["rcid"])
            prebuilt_results[_ROW_ID_COL] = _build_lookup_row_id(prebuilt_results)
        n_prebuilt_keys = prebuilt_results[_ROW_ID_COL].nunique() if not prebuilt_results.empty else 0
        print(
            f"  Strong staged matches: {len(prebuilt_results):,} rows across {n_prebuilt_keys:,} employer rows "
            f"({100 * n_prebuilt_keys / max(1, n_f1_total):.1f}% of rows, {_fmt_elapsed(time.perf_counter() - t0)})"
        )
    else:
        print("  WARNING: staged final employer crosswalk not found.")
        print("           Skipping strong path — uncovered employer rows will use exact/fuzzy fallback.")

    matched_row_ids = set(prebuilt_results[_ROW_ID_COL]) if not prebuilt_results.empty else set()
    unmatched_df = f1_employers_df[~f1_employers_df[_ROW_ID_COL].isin(matched_row_ids)].copy()

    print(f"\n[3/4] Matching {len(unmatched_df):,} uncovered or ambiguous employer rows against Revelio positions...")
    t0 = time.perf_counter()

    clean_exact_results = _empty_lookup_df()
    fuzzy_results = _empty_lookup_df()

    if len(unmatched_df) > 0 and bool(rev_pos_path) and os.path.exists(rev_pos_path):
        rev_esc = rev_pos_path.replace("'", "''")
        rev_exact_df = con.sql(f"""
            SELECT
                company_raw AS matched_company_name,
                {sql_clean_company_name_expr('company_raw')} AS rev_company_name_clean,
                CAST(rcid AS BIGINT) AS rcid,
                COUNT(DISTINCT user_id) AS n_users
            FROM read_parquet('{rev_esc}')
            WHERE company_raw IS NOT NULL
              AND TRIM(company_raw) != ''
              AND rcid IS NOT NULL
            GROUP BY matched_company_name, rev_company_name_clean, rcid
        """).df()
        rev_exact_df = rev_exact_df[
            rev_exact_df["rev_company_name_clean"].notna()
            & rev_exact_df["rev_company_name_clean"].str.strip().ne("")
        ].copy()
        rev_exact_df = (
            rev_exact_df
            .sort_values(["rev_company_name_clean", "rcid", "n_users", "matched_company_name"],
                         ascending=[True, True, False, True])
            .drop_duplicates(subset=["rev_company_name_clean", "rcid"], keep="first")
        )
        rev_cos_df = (
            rev_exact_df
            .sort_values(["rev_company_name_clean", "n_users", "matched_company_name", "rcid"],
                         ascending=[True, False, True, True])
            .drop_duplicates(subset=["rev_company_name_clean"], keep="first")
            [["rev_company_name_clean", "matched_company_name", "rcid"]]
            .copy()
        )

        print(
            f"  Revelio company universe: {len(rev_exact_df):,} clean-name x rcid rows | "
            f"{rev_cos_df['rev_company_name_clean'].nunique():,} distinct cleaned names"
        )

        clean_mask = (
            unmatched_df["employer_name_clean"].notna()
            & unmatched_df["employer_name_clean"].str.strip().ne("")
        )
        unmatched_clean_df = (
            unmatched_df.loc[clean_mask, ["employer_name_clean"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        clean_exact_matched = unmatched_clean_df.merge(
            rev_exact_df[["rev_company_name_clean", "matched_company_name", "rcid"]],
            left_on="employer_name_clean",
            right_on="rev_company_name_clean",
            how="inner",
        )[["employer_name_clean", "matched_company_name", "rcid"]].drop_duplicates()

        if not clean_exact_matched.empty:
            clean_exact_results = unmatched_df.merge(
                clean_exact_matched,
                on="employer_name_clean",
                how="inner",
            )
            clean_exact_results["match_type"] = "clean_exact"
            clean_exact_results["match_score"] = 1.0
            clean_exact_results = clean_exact_results[LOOKUP_BASE_COLS].copy()
            clean_exact_results = _coerce_nullable_int(clean_exact_results, ["rcid"])
            clean_exact_results[_ROW_ID_COL] = _build_lookup_row_id(clean_exact_results)

        n_clean_exact_keys = clean_exact_results[_ROW_ID_COL].nunique() if not clean_exact_results.empty else 0
        print(
            f"  Clean exact matches: {len(clean_exact_results):,} rows across {n_clean_exact_keys:,} employer rows "
            f"({100 * n_clean_exact_keys / max(1, len(unmatched_df)):.1f}% of uncovered rows)"
        )

        clean_exact_row_ids = set(clean_exact_results[_ROW_ID_COL]) if not clean_exact_results.empty else set()
        fuzzy_input_df = unmatched_df[~unmatched_df[_ROW_ID_COL].isin(clean_exact_row_ids)].copy()
        fuzzy_input_clean_df = (
            fuzzy_input_df[["employer_name_clean"]]
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
        )
        fuzzy_input_clean_df = fuzzy_input_clean_df[
            fuzzy_input_clean_df["employer_name_clean"].str.strip().ne("")
        ].copy()

        if not do_fuzzy:
            print("  Fuzzy matching disabled — keeping pre-built + clean exact matches only.")
        elif fuzzy_input_clean_df.empty:
            print("  No employer rows remain after clean exact matching — fuzzy step skipped.")
        else:
            fuzzy_input_clean_df["_fuzzy_block"] = _build_block_key(fuzzy_input_clean_df["employer_name_clean"])
            rev_cos_df["_fuzzy_block"] = _build_block_key(rev_cos_df["rev_company_name_clean"])

            matched_df = _chunked_fuzzy_join(
                left=fuzzy_input_clean_df,
                right=rev_cos_df,
                left_on="employer_name_clean",
                right_on="rev_company_name_clean",
                threshold=threshold,
                top_n=1,
                use_jw_only=use_jw_only,
                jw_top_k=25,
                keep_subscores=False,
                block_key="_fuzzy_block",
            )

            if len(matched_df) > 0:
                matched_clean = matched_df.rename(columns={
                    "employer_name_clean_left": "employer_name_clean",
                    "matched_company_name_right": "matched_company_name",
                    "rcid_right": "rcid",
                    "_score": "match_score",
                })[["employer_name_clean", "matched_company_name", "rcid", "match_score"]].drop_duplicates()

                fuzzy_results = fuzzy_input_df.merge(
                    matched_clean,
                    on="employer_name_clean",
                    how="inner",
                )
                fuzzy_results["match_type"] = "fuzzy"
                fuzzy_results = fuzzy_results[LOOKUP_BASE_COLS].copy()
                fuzzy_results = _coerce_nullable_int(fuzzy_results, ["rcid"])
                fuzzy_results[_ROW_ID_COL] = _build_lookup_row_id(fuzzy_results)

        n_fuzzy_keys = fuzzy_results[_ROW_ID_COL].nunique() if not fuzzy_results.empty else 0
        print(
            f"  Fuzzy matches: {len(fuzzy_results):,} rows across {n_fuzzy_keys:,} employer rows "
            f"({100 * n_fuzzy_keys / max(1, len(unmatched_df)):.1f}% of uncovered rows, {_fmt_elapsed(time.perf_counter() - t0)})"
        )
    elif len(unmatched_df) > 0:
        print(f"  WARNING: Revelio positions parquet not found at: {rev_pos_path}")
        print("           Skipping exact/fuzzy fallback.")
    else:
        print("  All employer rows covered by pre-built crosswalk — no fallback matching needed.")

    print("\n[4/4] Combining results and writing parquet...")

    lookup_df = pd.concat(
        [prebuilt_results[LOOKUP_BASE_COLS], clean_exact_results[LOOKUP_BASE_COLS], fuzzy_results[LOOKUP_BASE_COLS]],
        ignore_index=True,
    ) if (not prebuilt_results.empty or not clean_exact_results.empty or not fuzzy_results.empty) else _empty_lookup_df()

    if not lookup_df.empty:
        lookup_df = _coerce_nullable_int(lookup_df, ["rcid"])
        lookup_df[_ROW_ID_COL] = _build_lookup_row_id(lookup_df)
        lookup_df = lookup_df.drop_duplicates(subset=[_ROW_ID_COL, "rcid"], keep="first")
        row_stats = (
            lookup_df.groupby(_ROW_ID_COL, dropna=False)
            .agg(
                lookup_rcid_count=("rcid", "nunique"),
                lookup_has_direct_ind=("match_type", lambda s: int((s == "staged_direct").any())),
            )
            .reset_index()
        )
        row_stats["lookup_rcid_ambiguous_ind"] = (row_stats["lookup_rcid_count"] > 1).astype(int)
        lookup_df = lookup_df.merge(row_stats, on=_ROW_ID_COL, how="left")
        n_matched_employer_rows = int(row_stats[_ROW_ID_COL].nunique())
        n_ambiguous_rows = int(row_stats["lookup_rcid_ambiguous_ind"].sum())

        # Deduplicate to (foia_firm_uid, rcid) — multiple FOIA employer rows can
        # share the same firm+rcid after upstream canonical matching. Keeping
        # one row per entity×rcid prevents the join in rev_pos_summary from exploding.
        # Uses the same score ordering as _build_rev_pos_summary_query: NULL → 1.0.
        entity_mask = lookup_df["foia_firm_uid"].notna() & lookup_df["rcid"].notna()
        entity_rows = (
            lookup_df[entity_mask]
            .assign(_sort_score=lambda d: d["match_score"].fillna(1.0))
            .sort_values("_sort_score", ascending=False)
            .drop_duplicates(subset=["foia_firm_uid", "rcid"], keep="first")
            .drop(columns=["_sort_score"])
        )
        lookup_df = pd.concat(
            [entity_rows, lookup_df[~entity_mask]], ignore_index=True
        )
        lookup_df = lookup_df.drop(columns=[_ROW_ID_COL])
        lookup_df = lookup_df[LOOKUP_COLS].copy()
    else:
        n_matched_employer_rows = 0
        n_ambiguous_rows = 0

    n_prebuilt = int((lookup_df["match_type"] == "staged_direct").sum()) if not lookup_df.empty else 0
    n_clean_exact = int((lookup_df["match_type"] == "clean_exact").sum()) if not lookup_df.empty else 0
    n_fuzzy = int((lookup_df["match_type"] == "fuzzy").sum()) if not lookup_df.empty else 0
    n_matched_rows = len(lookup_df)
    n_unmatched_rows = n_f1_total - n_matched_employer_rows
    n_entity_ids = int(lookup_df["foia_firm_uid"].nunique(dropna=True)) if not lookup_df.empty else 0

    print(f"  Pre-built rows:    {n_prebuilt:,}")
    print(f"  Clean exact rows:  {n_clean_exact:,}")
    print(f"  Fuzzy rows:        {n_fuzzy:,}")
    print(
        f"  Total matched:     {n_matched_rows:,} rows across {n_matched_employer_rows:,} / {n_f1_total:,} employer rows "
        f"({100 * n_matched_employer_rows / max(1, n_f1_total):.1f}% of rows)"
    )
    print(f"  Unmatched rows:    {n_unmatched_rows:,}")
    print(f"  Distinct rcids:    {lookup_df['rcid'].nunique() if not lookup_df.empty else 0:,}")
    print(f"  Distinct firms:    {n_entity_ids:,}")
    print(f"  Ambiguous employer rows: {n_ambiguous_rows:,}")
    print(f"  Rows after entity×rcid dedup: {n_matched_rows:,}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    lookup_df.to_parquet(out_path, index=False)

    print(f"\n[done] Wrote employer lookup: {out_path} ({_fmt_elapsed(time.perf_counter() - t_total)})")
    return lookup_df


if __name__ == "__main__":
    build_employer_crosswalk()
