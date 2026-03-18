"""
build_trk_rcid_crosswalk.py
============================
Build a foia_firm_uid → rcid crosswalk for pre-2021 TRK_12704 firms so that
firm_outcomes.py can link them to Revelio headcounts.

Background
----------
pre2021_foia.parquet (built by build_pre2021_foia.py) has main_rcid=NaN for all
rows because TRK_12704 petition records have no individual-level Revelio IDs.
This script provides a firm-level crosswalk to fill that gap.

There are two types of TRK firms:
  is_new=False  – matched to an existing Bloomberg foia_firm_uid when
                  extend_foia_dedup_trk12704.py ran.  Their rcid already exists
                  in the LLM-reviewed crosswalk; we just collapse it to firm
                  level and propagate.
  is_new=True   – brand-new foia_firm_uid values minted from TRK names only.
                  We fuzzy-match these against the Revelio company roster using
                  the same token_sort_ratio approach as lca_firm_match.py.

Pipeline
--------
Step 1 — is_new=False firms
  Load trk12704_to_foiafirm.csv, keep is_new=False unique foia_firm_uid.
  Join to llm_review_all_foia_to_rcid_crosswalk.csv (valid_match only).
  Collapse to firm-level mode rcid, tag match_type='existing_llm'.

Step 2 — is_new=True firms
  Load foia_firms_dedup_extended.csv (source='trk12704').
  Load revelio_rcid_entities.csv, filter to US companies with non-null state.
  Deduplicate Revelio roster to one row per (name_base, top_state_norm).
  Block on first block_prefix_len chars of name_base.
  Score candidate pairs with rapidfuzz token_sort_ratio.
  Apply state bonus (+5 pts) when hq_state_mode matches top_state_norm.
  Keep best-scoring match per TRK firm where adjusted_score >= match_threshold.
  Tag match_type='fuzzy_name'.

Step 3 — Union and save
  Combine both steps; deduplicate (prefer existing_llm over fuzzy_name).
  Save as pre2021_rcid_crosswalk.parquet.
  Print coverage stats.

Output schema
-------------
  foia_firm_uid   str     – TRK firm identifier
  rcid            float   – Revelio company ID
  match_type      str     – 'existing_llm' | 'fuzzy_name'
  match_score     float   – 1.0 for existing_llm; token_sort_ratio/100 for fuzzy
  is_new          bool    – False = matched to existing Bloomberg UID; True = new

Usage
-----
Run from repo root:
    python revelio_h1b_company_matching/build_trk_rcid_crosswalk.py
Or import and call main() in an iPython session.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Path setup — make company_name_cleaning and config importable
# ---------------------------------------------------------------------------
if "__file__" in globals():
    _THIS_DIR = Path(__file__).resolve().parent
else:
    _THIS_DIR = Path.cwd() / "revelio_h1b_company_matching"

_CODE_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_CODE_DIR))

from company_name_cleaning import normalize_state  # noqa: E402
from config import root  # noqa: E402

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
_CFG_PATH = _CODE_DIR / "configs" / "build_trk_rcid_crosswalk.yaml"


def _load_config() -> dict:
    raw = yaml.safe_load(_CFG_PATH.read_text()) or {}

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            return obj.replace("{root}", str(root))
        return obj

    return _expand(raw)


CFG = _load_config()


# ---------------------------------------------------------------------------
# US state abbreviation set (for filtering Revelio to US companies)
# ---------------------------------------------------------------------------
_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}


# ---------------------------------------------------------------------------
# Step 1: is_new=False firms — reuse existing LLM crosswalk
# ---------------------------------------------------------------------------

def _build_existing_llm_rcids(cfg: dict) -> pd.DataFrame:
    """
    For TRK firms matched to an existing Bloomberg foia_firm_uid (is_new=False),
    look up their rcid from the LLM-reviewed crosswalk and collapse to firm level.

    Returns DataFrame with columns:
        foia_firm_uid, rcid, match_type, match_score, is_new
    """
    print("\n--- Step 1: is_new=False firms (LLM crosswalk lookup) ---", flush=True)
    t0 = time.time()

    # Load TRK crosswalk, filter to is_new=False
    trk = pd.read_csv(cfg["trk_crosswalk"], dtype=str)
    trk["is_new"] = trk["is_new"].map({"True": True, "False": False, True: True, False: False})
    existing = trk[~trk["is_new"].fillna(True)][["foia_firm_uid"]].drop_duplicates()
    print(f"  is_new=False unique UIDs: {len(existing):,}")

    # Load LLM crosswalk, keep valid_match rows only
    llm = pd.read_csv(
        cfg["llm_rcid_crosswalk"],
        usecols=["foia_firm_uid", "rcid", "crosswalk_validity_label"],
        dtype={"rcid": str},
    )
    llm = llm[llm["crosswalk_validity_label"] == "valid_match"].copy()
    llm["rcid"] = pd.to_numeric(llm["rcid"], errors="coerce")
    llm = llm.dropna(subset=["rcid"])

    # Join by foia_firm_uid
    merged = existing.merge(llm[["foia_firm_uid", "rcid"]], on="foia_firm_uid", how="left")

    # Collapse to firm-level mode rcid (a firm may appear in multiple fein_year rows)
    def _mode_rcid(s):
        s = s.dropna()
        return float(s.mode().iloc[0]) if not s.empty else np.nan

    rcid_by_firm = (
        merged.groupby("foia_firm_uid")["rcid"]
        .agg(_mode_rcid)
        .reset_index()
        .rename(columns={"rcid": "rcid"})
    )

    rcid_by_firm = rcid_by_firm.dropna(subset=["rcid"])
    rcid_by_firm["match_type"]  = "existing_llm"
    rcid_by_firm["match_score"] = 1.0
    rcid_by_firm["is_new"]      = False

    n_total   = len(existing)
    n_matched = len(rcid_by_firm)
    print(
        f"  Matched {n_matched:,} / {n_total:,} is_new=False firms to rcid "
        f"({n_matched / max(n_total, 1) * 100:.1f}%)  ({time.time()-t0:.1f}s)"
    )
    return rcid_by_firm[["foia_firm_uid", "rcid", "match_type", "match_score", "is_new"]]


# ---------------------------------------------------------------------------
# Step 2: is_new=True firms — fuzzy match against Revelio
# ---------------------------------------------------------------------------

def _load_revelio_roster(cfg: dict) -> pd.DataFrame:
    """
    Load Revelio rcid entity table, filter to US companies, and deduplicate to
    one row per (name_base, top_state_norm) keeping the rcid with most users.

    Returns DataFrame with columns: rcid, company, name_base, top_state_norm
    """
    print("\nLoading Revelio company roster...", flush=True)
    t0 = time.time()

    rev = pd.read_csv(
        cfg["revelio_rcid_entities"],
        usecols=["rcid", "company", "name_base", "top_state_norm", "n_users"],
        dtype={"rcid": str, "top_state_norm": str},
        low_memory=False,
    )
    print(f"  {len(rev):,} Revelio rows loaded ({time.time()-t0:.1f}s)")

    # Normalize state abbreviation; filter to US
    rev["top_state_norm"] = rev["top_state_norm"].apply(
        lambda x: normalize_state(x, to="abbr") if pd.notna(x) and str(x).strip() != "" else None
    )
    rev = rev[rev["top_state_norm"].isin(_US_STATES)].copy()
    print(f"  {len(rev):,} rows after filtering to US companies")

    # Convert rcid to float and drop invalid
    rev["rcid"] = pd.to_numeric(rev["rcid"], errors="coerce")
    rev = rev.dropna(subset=["rcid", "name_base"])
    rev["name_base"] = rev["name_base"].astype(str).str.strip()
    rev = rev[rev["name_base"] != ""].copy()

    # Fill n_users for dedup tiebreak
    rev["n_users"] = pd.to_numeric(rev["n_users"], errors="coerce").fillna(0)

    # Deduplicate: keep one rcid per (name_base, top_state_norm) — largest n_users
    rev = (
        rev.sort_values("n_users", ascending=False)
        .drop_duplicates(subset=["name_base", "top_state_norm"], keep="first")
        .reset_index(drop=True)
    )
    print(f"  {len(rev):,} unique (name_base, state) Revelio entries")
    return rev[["rcid", "company", "name_base", "top_state_norm"]]


def _load_trk_new_roster(cfg: dict) -> pd.DataFrame:
    """
    Load the extended FOIA dedup, filtered to new TRK firms (source='trk12704').
    Normalize state to abbreviation.

    Returns DataFrame with columns:
        foia_firm_uid, canonical_name_clean, name_base, state_abbr
    """
    print("\nLoading new TRK firm roster from extended dedup...", flush=True)
    t0 = time.time()

    dedup = pd.read_csv(cfg["foia_firms_dedup_extended"], dtype=str)
    print(f"  {len(dedup):,} total entries in extended dedup ({time.time()-t0:.1f}s)")

    # Filter to TRK-sourced firms only (source='trk12704')
    trk_only = dedup[dedup.get("source", pd.Series(dtype=str)) == "trk12704"].copy()
    if trk_only.empty:
        # Fallback: use is_new flag from trk crosswalk to identify new UIDs
        print("  [WARN] source='trk12704' filter returned 0 rows — will fall back to trk crosswalk UIDs")
        trk_cw = pd.read_csv(cfg["trk_crosswalk"], dtype=str)
        trk_cw["is_new"] = trk_cw["is_new"].map({"True": True, "False": False, True: True, False: False})
        new_uids = set(trk_cw[trk_cw["is_new"].fillna(True)]["foia_firm_uid"].unique())
        trk_only = dedup[dedup["foia_firm_uid"].isin(new_uids)].copy()

    print(f"  {len(trk_only):,} new TRK firm entries")

    # Keep one row per foia_firm_uid (should already be deduplicated)
    trk_only = trk_only.drop_duplicates(subset=["foia_firm_uid"]).copy()

    # Normalize names
    trk_only["name_base"] = trk_only["canonical_name_base"].fillna("").astype(str).str.strip()
    trk_only["canonical_name_clean"] = trk_only["canonical_name_clean"].fillna("").astype(str)

    # Normalize state
    trk_only["state_abbr"] = trk_only["hq_state_mode"].apply(
        lambda x: normalize_state(x, to="abbr") if pd.notna(x) and str(x).strip() not in ("", "nan") else None
    )

    return trk_only[["foia_firm_uid", "canonical_name_clean", "name_base", "state_abbr"]].reset_index(drop=True)


def _fuzzy_match_new_firms(
    trk_roster: pd.DataFrame,
    rev_roster: pd.DataFrame,
    match_threshold: float,
    block_prefix_len: int,
    max_block_size: int,
) -> pd.DataFrame:
    """
    Fuzzy-match new TRK firms (is_new=True) against the Revelio company roster.

    Blocking: first block_prefix_len chars of name_base.
    Scoring: rapidfuzz token_sort_ratio on (canonical_name_clean, company).
    State bonus: +5 pts when TRK state_abbr matches Revelio top_state_norm.
    Threshold: adjusted_score >= match_threshold.

    Returns DataFrame with columns:
        foia_firm_uid, rcid, rev_company, match_type, match_score, is_new
    """
    print(
        f"\n--- Step 2: is_new=True fuzzy match "
        f"(threshold={match_threshold}, block_prefix={block_prefix_len}) ---",
        flush=True,
    )
    t0 = time.time()

    # Build block key
    trk_roster = trk_roster.copy()
    rev_roster  = rev_roster.copy()
    trk_roster["block_key"] = trk_roster["name_base"].str[:block_prefix_len]
    rev_roster["block_key"]  = rev_roster["name_base"].str[:block_prefix_len]

    blocks = sorted(trk_roster["block_key"].dropna().unique())
    print(f"  {len(trk_roster):,} TRK firms × {len(rev_roster):,} Revelio entries → {len(blocks):,} blocks")

    results = []
    n_skipped_blocks = 0
    n_processed = 0

    for block in blocks:
        trk_block = trk_roster[trk_roster["block_key"] == block]
        rev_block  = rev_roster[rev_roster["block_key"] == block]

        if rev_block.empty:
            continue

        # Skip oversized blocks to avoid excessive compute
        if len(trk_block) * len(rev_block) > max_block_size * max_block_size:
            n_skipped_blocks += 1
            continue

        trk_names = trk_block["canonical_name_clean"].tolist()
        rev_names  = rev_block["company"].tolist()

        # Score matrix: rows=TRK, cols=Revelio
        scores = process.cdist(trk_names, rev_names, scorer=fuzz.token_sort_ratio)

        for i, trk_row in enumerate(trk_block.itertuples()):
            row_scores = scores[i].copy()

            # Apply state bonus: +5 pts when states match
            trk_state = trk_row.state_abbr
            if pd.notna(trk_state) and trk_state in _US_STATES:
                for j, rev_row in enumerate(rev_block.itertuples()):
                    if rev_row.top_state_norm == trk_state:
                        row_scores[j] = min(100.0, row_scores[j] + 5)

            best_score = float(row_scores.max())
            if best_score < match_threshold:
                continue

            best_idx = int(row_scores.argmax())
            rev_row  = rev_block.iloc[best_idx]

            results.append({
                "foia_firm_uid": trk_row.foia_firm_uid,
                "rcid":          float(rev_row["rcid"]),
                "rev_company":   rev_row["company"],
                "match_type":    "fuzzy_name",
                "match_score":   best_score / 100.0,
                "is_new":        True,
            })
            n_processed += 1

    if n_skipped_blocks > 0:
        print(f"  [INFO] {n_skipped_blocks} oversized blocks skipped (>{max_block_size}×{max_block_size} pairs)")

    matched = pd.DataFrame(results) if results else pd.DataFrame(
        columns=["foia_firm_uid", "rcid", "rev_company", "match_type", "match_score", "is_new"]
    )

    n_total   = len(trk_roster)
    n_matched = len(matched)
    print(
        f"  Matched {n_matched:,} / {n_total:,} is_new=True firms "
        f"({n_matched / max(n_total, 1) * 100:.1f}%)  ({time.time()-t0:.1f}s)"
    )
    return matched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()
    cfg = _load_config()
    testing = cfg.get("testing", {}).get("enabled", False)
    seed    = cfg.get("testing", {}).get("seed", 0)
    n_test  = cfg.get("testing", {}).get("n_firms", 500)

    print("=" * 60)
    print("build_trk_rcid_crosswalk.py")
    print(f"  run_tag : {cfg.get('run_tag', 'n/a')}")
    print(f"  testing : {testing}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: is_new=False firms — reuse existing LLM crosswalk
    # ------------------------------------------------------------------
    llm_rcids = _build_existing_llm_rcids(cfg)

    # ------------------------------------------------------------------
    # Step 2: is_new=True firms — fuzzy match against Revelio
    # ------------------------------------------------------------------
    trk_roster = _load_trk_new_roster(cfg)

    if testing:
        import random
        random.seed(seed)
        sample_uids = set(random.sample(trk_roster["foia_firm_uid"].tolist(),
                                        min(n_test, len(trk_roster))))
        trk_roster = trk_roster[trk_roster["foia_firm_uid"].isin(sample_uids)].copy()
        print(f"  [TESTING] Sampled {len(trk_roster):,} is_new=True firms")

    rev_roster = _load_revelio_roster(cfg)

    fuzzy_rcids = _fuzzy_match_new_firms(
        trk_roster     = trk_roster,
        rev_roster     = rev_roster,
        match_threshold = cfg.get("match_threshold", 90.0),
        block_prefix_len = int(cfg.get("block_prefix_len", 10)),
        max_block_size   = int(cfg.get("max_block_size", 500)),
    )

    # ------------------------------------------------------------------
    # Step 3: Union and save
    # ------------------------------------------------------------------
    print("\n--- Step 3: Union and save ---", flush=True)

    # Align columns before concat
    keep_cols = ["foia_firm_uid", "rcid", "match_type", "match_score", "is_new"]
    llm_rcids["rev_company"] = np.nan  # not available for LLM path
    parts = [llm_rcids[keep_cols]]
    if not fuzzy_rcids.empty:
        parts.append(fuzzy_rcids[keep_cols])

    combined = pd.concat(parts, ignore_index=True)

    # Deduplicate: prefer existing_llm over fuzzy_name
    priority = {"existing_llm": 0, "fuzzy_name": 1}
    combined["_priority"] = combined["match_type"].map(priority).fillna(99)
    combined = (
        combined.sort_values(["foia_firm_uid", "_priority", "match_score"],
                             ascending=[True, True, False])
        .drop_duplicates(subset=["foia_firm_uid"], keep="first")
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )

    # Ensure correct dtypes
    combined["rcid"]        = combined["rcid"].astype(float)
    combined["match_score"] = combined["match_score"].astype(float)
    combined["is_new"]      = combined["is_new"].astype(bool)

    # Coverage summary
    total_trk_uids = (
        len(pd.read_csv(cfg["trk_crosswalk"], usecols=["foia_firm_uid"])["foia_firm_uid"].unique())
    )
    n_matched_total   = len(combined)
    n_llm_matched     = (combined["match_type"] == "existing_llm").sum()
    n_fuzzy_matched   = (combined["match_type"] == "fuzzy_name").sum()
    print(f"\n  Total TRK unique UIDs   : {total_trk_uids:,}")
    print(f"  Matched via existing_llm: {n_llm_matched:,}")
    print(f"  Matched via fuzzy_name  : {n_fuzzy_matched:,}")
    print(f"  Total matched           : {n_matched_total:,} ({n_matched_total/max(total_trk_uids,1)*100:.1f}%)")
    print(f"  Unmatched               : {total_trk_uids - n_matched_total:,}")

    # Save
    out_path = Path(cfg["output_path"])
    if testing:
        stem = out_path.stem
        out_path = out_path.with_name(f"{stem}_test.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"\n  Saved {len(combined):,} rows → {out_path}  ({time.time()-t_total:.1f}s total)")

    return combined


if __name__ == "__main__":
    main()
