"""
extend_foia_dedup_trk12704.py
==============================
Extend the existing FOIA employer deduplication to include old H-1B petition
approval data from the TRK_12704 Excel files (FY2000–2023).

The Bloomberg-based `foia_firms_dedup.csv` only covers the lottery era (FY2021+).
The TRK_12704 files cover FY2000–2023 and have `PET_FIRM_NAME`, `PET_CITY`,
`PET_STATE` (no FEIN, no address). This script:

  1. Loads TRK_12704 for specified years → build unique firm roster.
  2. Cleans names with `clean_company_name()` (same as the existing pipeline).
  3. Deduplicates TRK_12704 firms against each other (blocking + union-find).
  4. Matches each canonical TRK_12704 firm against existing `foia_firms_dedup`:
       - Name similarity >= match_threshold AND same state → assign existing UID.
       - Otherwise → mint new UID via stable_hash_id.
  5. Appends new-only rows to produce `foia_firms_dedup_extended.csv` (superset).
     **Existing foia_firm_uid values are never modified.**

Output
------
  {output_dir}/foia_firms_dedup_extended.csv
      Superset of foia_firms_dedup.csv. New TRK_12704-only rows have
      source="trk12704"; original rows carry source="bloomberg" (added).

  {output_dir}/trk12704_to_foiafirm.csv
      One row per canonical TRK_12704 firm → foia_firm_uid crosswalk.
      Columns: pet_firm_name, pet_state, foia_firm_uid, match_type,
               match_score, is_new, n_apps, n_years, canonical_name_base

Usage
-----
Run from repo root:
    python revelio_h1b_company_matching/extend_foia_dedup_trk12704.py
Or import and call main() in an iPython session.
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Path setup — make company_name_cleaning importable
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_CODE_DIR  = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_CODE_DIR))

from company_name_cleaning import (  # noqa: E402
    CleanName,
    clean_company_name,
    name_similarity,
    normalize_state,
    stable_hash_id,
)
from dedupe_utils import (  # noqa: E402
    DedupeConfig,
    aggregate_components,
    assign_components,
    find_duplicate_edges,
)
from config import root  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CFG_PATH = _CODE_DIR / "configs" / "extend_foia_dedup.yaml"


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
# TRK_12704 file registry
# FY2000: "TRK_12704_FY2000 clean.xlsx" (no "redacted")
# FY2001+: "TRK_12704_FY{YEAR} clean redacted.xlsx"
# Sheet name pattern: "TRK_12704_FY{YEAR}.csv"
# ---------------------------------------------------------------------------

def _trk_filename(year: int) -> str:
    if year == 2000:
        return f"TRK_12704_FY{year} clean.xlsx"
    return f"TRK_12704_FY{year} clean redacted.xlsx"


def _trk_sheetname(year: int) -> str:
    return f"TRK_12704_FY{year}.csv"


# ---------------------------------------------------------------------------
# Step A: Load TRK_12704 files
# ---------------------------------------------------------------------------

def load_trk12704(cfg: dict) -> pd.DataFrame:
    """Load and union TRK_12704 Excel files for configured years."""
    raw_dir = Path(cfg["h1b_raw_dir"])
    years   = cfg["h1b_years"]
    testing = cfg.get("testing", {}).get("enabled", False)
    nrows   = cfg["testing"].get("n_firms") if testing else None

    print(f"Loading TRK_12704 for years {years}...")
    frames = []
    t0 = time.time()

    for year in years:
        fname = _trk_filename(year)
        path  = raw_dir / fname
        if not path.exists():
            print(f"  FY{year}: {fname} not found — skipping")
            continue

        sheet = _trk_sheetname(year)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_excel(path, sheet_name=sheet, dtype=str,
                                   nrows=nrows, engine="openpyxl")
        except Exception as e:
            print(f"  FY{year}: ERROR loading {fname}: {e} — skipping")
            continue

        # Normalise column names
        df.columns = [c.strip().upper() for c in df.columns]

        keep = {}
        for want, candidates in [
            ("PET_FIRM_NAME",    ["PET_FIRM_NAME"]),
            ("PET_CITY",         ["PET_CITY"]),
            ("PET_STATE",        ["PET_STATE"]),
            ("FIRST_DECISION_FY",["FIRST_DECISION_FY"]),
        ]:
            col = next((c for c in candidates if c in df.columns), None)
            if col:
                keep[want] = df[col]

        if "PET_FIRM_NAME" not in keep:
            print(f"  FY{year}: PET_FIRM_NAME column not found — skipping")
            continue

        chunk = pd.DataFrame(keep)
        chunk["fiscal_year"] = year
        n_before = len(chunk)
        chunk = chunk[chunk["PET_FIRM_NAME"].notna() & (chunk["PET_FIRM_NAME"].str.strip() != "")].copy()
        print(f"  FY{year}: {n_before:,} rows → {len(chunk):,} with non-null firm name")
        frames.append(chunk)

    if not frames:
        raise ValueError("No TRK_12704 files loaded. Check h1b_raw_dir and h1b_years in config.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(combined):,} rows across {combined['fiscal_year'].nunique()} years "
          f"({time.time()-t0:.1f}s)")
    return combined


# ---------------------------------------------------------------------------
# Step B: Build unique firm × state roster from TRK_12704
# ---------------------------------------------------------------------------

def _mode_or_first(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if not s.empty else None


def build_trk_roster(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse TRK_12704 rows to unique (PET_FIRM_NAME, PET_STATE).
    Applies clean_company_name() and normalize_state() for normalization.
    Returns a DataFrame with cleaned name fields + aggregated counts.
    """
    print("\nBuilding unique firm × state roster from TRK_12704...")
    t0 = time.time()

    # Standardize state to full name for comparability with foia_firms_dedup
    raw = raw.copy()
    raw["state_norm"] = raw["PET_STATE"].apply(lambda x: normalize_state(x, to="name"))

    # Aggregate to unique (PET_FIRM_NAME, state_norm)
    def mode_city(s):
        return _mode_or_first(s)

    agg = (
        raw.groupby(["PET_FIRM_NAME", "state_norm"], dropna=False)
        .agg(
            n_apps  = ("PET_FIRM_NAME",  "count"),
            n_years = ("fiscal_year",    "nunique"),
            city    = ("PET_CITY",       mode_city),
        )
        .reset_index()
    )
    print(f"  {len(agg):,} unique (firm, state) pairs")

    # Apply clean_company_name() to each firm name
    print("  Cleaning firm names via clean_company_name()...", flush=True)
    cleaned = agg["PET_FIRM_NAME"].apply(lambda n: clean_company_name(n))
    agg["name_clean"]    = [c.clean       for c in cleaned]
    agg["name_stub"]     = [c.stub        for c in cleaned]
    agg["name_base"]     = [c.base        for c in cleaned]
    agg["token_stream"]  = [c.token_stream for c in cleaned]

    # Drop rows where cleaning yields an empty name_base (unusable for matching)
    n_before = len(agg)
    agg = agg[agg["name_base"].str.strip() != ""].copy().reset_index(drop=True)
    if len(agg) < n_before:
        print(f"  Dropped {n_before - len(agg)} rows with empty name_base after cleaning")

    print(f"  Roster: {len(agg):,} entries ({time.time()-t0:.1f}s)")
    return agg


# ---------------------------------------------------------------------------
# Step C: Deduplicate TRK_12704 firms against each other
# ---------------------------------------------------------------------------

def dedup_trk_roster(roster: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deduplicate TRK_12704 firms within themselves using blocking + union-find.
    Returns (canonical_df, mapping_df).
    canonical_df: one row per component (representative firm)
    mapping_df: (PET_FIRM_NAME, state_norm) -> canonical row index
    """
    print("\nDeduplicating TRK_12704 firms against each other...")
    t0 = time.time()

    dcfg_raw = cfg.get("dedup", {})
    dcfg = DedupeConfig(
        strong_name     = float(dcfg_raw.get("strong_name",     98.0)),
        name_and_state  = float(dcfg_raw.get("name_and_state",  95.0)),
        name_and_domain = float(dcfg_raw.get("name_and_domain", 93.0)),
        name_and_naics2 = float(dcfg_raw.get("name_and_naics2", 96.0)),
        block_prefix_len= int(dcfg_raw.get("block_prefix_len",  10)),
        max_block_size  = int(dcfg_raw.get("max_block_size",   500)),
    )

    edges = find_duplicate_edges(
        roster,
        base_col  = "name_base",
        stub_col  = "name_stub",
        state_col = "state_norm",
        cfg       = dcfg,
    )
    print(f"  Found {len(edges)} duplicate edges among TRK_12704 firms")

    labels = assign_components(len(roster), edges)
    n_components = len(set(labels))
    n_merged = len(roster) - n_components
    print(f"  {len(roster):,} unique firms → {n_components:,} components "
          f"({n_merged:,} merged as duplicates, {time.time()-t0:.1f}s)")

    # Aggregate components: representative = highest n_apps row, list PET_FIRM_NAMEs
    canon, mapping = aggregate_components(
        roster,
        comp_labels      = labels,
        id_cols_for_uid  = ["name_base", "state_norm"],  # temporary UID (overwritten later)
        uid_prefix       = "TRK",
        weight_col       = "n_apps",
        keep_cols        = ["PET_FIRM_NAME", "state_norm", "name_clean", "name_stub",
                            "name_base", "token_stream", "city"],
        list_cols        = ["PET_FIRM_NAME"],
    )
    # Aggregate counts across merged firms
    roster["_comp"] = labels
    comp_counts = (
        roster.groupby("_comp")
        .agg(n_apps_total=("n_apps", "sum"), n_years_total=("n_years", "sum"))
        .reset_index()
        .rename(columns={"_comp": "component"})
    )
    canon = canon.merge(comp_counts, on="component", how="left")

    print(f"  Canonical TRK firms: {len(canon):,}")
    return canon, mapping


# ---------------------------------------------------------------------------
# Step D: Match canonical TRK_12704 firms → existing foia_firms_dedup
# ---------------------------------------------------------------------------

def match_trk_to_existing(canon: pd.DataFrame, foia_dedup: pd.DataFrame,
                           cfg: dict) -> pd.DataFrame:
    """
    For each canonical TRK_12704 firm, find the best matching foia_firm_uid in
    the existing dedup. Assigns either:
      - An existing foia_firm_uid (match_type="existing") if score >= threshold AND state matches
      - A new foia_firm_uid (match_type="new") otherwise

    Returns canon with foia_firm_uid, match_type, match_score, is_new added.
    """
    match_threshold = float(cfg.get("match_threshold", 90.0))
    prefix_len      = int(cfg.get("dedup", {}).get("block_prefix_len", 10))
    max_block       = int(cfg.get("dedup", {}).get("max_block_size",   500))

    print(f"\nMatching {len(canon):,} canonical TRK firms → {len(foia_dedup):,} existing foia firms "
          f"(threshold={match_threshold})...")
    t0 = time.time()

    # Build prefix index on existing foia firms
    foia_dedup = foia_dedup.copy()
    # foia_firms_dedup.csv has: foia_firm_uid, canonical_name_base, canonical_name_stub,
    # canonical_name_clean, hq_state_mode (full state name), etc.
    base_col  = "canonical_name_base"
    stub_col  = "canonical_name_stub"
    state_col = "hq_state_mode"

    # Build inverted index: prefix → list of row indices in foia_dedup
    prefix_index: Dict[str, List[int]] = {}
    for i, base in enumerate(foia_dedup[base_col].fillna("").astype(str)):
        key = base[:prefix_len]
        if key:
            prefix_index.setdefault(key, []).append(i)

    matched_uids   = []
    match_types    = []
    match_scores   = []

    for _, row in canon.iterrows():
        trk_base  = str(row["name_base"] or "")
        trk_stub  = str(row["name_stub"] or "")
        trk_state = str(row["state_norm"] or "")

        key = trk_base[:prefix_len]
        candidates = prefix_index.get(key, [])

        best_score = 0.0
        best_idx   = None

        if 0 < len(candidates) <= max_block:
            for ci in candidates:
                ex_base  = str(foia_dedup.at[ci, base_col] or "")
                ex_stub  = str(foia_dedup.at[ci, stub_col] or "")
                ex_state = str(foia_dedup.at[ci, state_col] or "")

                sim = name_similarity(trk_stub, trk_base, ex_stub, ex_base)

                # State agreement required: both states non-null and matching
                state_ok = (trk_state and ex_state and
                            trk_state.casefold() == ex_state.casefold())

                if sim >= match_threshold and state_ok and sim > best_score:
                    best_score = sim
                    best_idx   = ci

        if best_idx is not None:
            # Assign existing foia_firm_uid
            matched_uids.append(foia_dedup.at[best_idx, "foia_firm_uid"])
            match_types.append("existing")
            match_scores.append(best_score)
        else:
            # Mint new foia_firm_uid from name_base + state
            key_parts = [trk_base + ":" + trk_state.casefold()]
            new_uid = stable_hash_id("FOIAFIRM", key_parts)
            matched_uids.append(new_uid)
            match_types.append("new")
            match_scores.append(0.0)

    canon = canon.copy()
    canon["foia_firm_uid"] = matched_uids
    canon["match_type"]    = match_types
    canon["match_score"]   = match_scores
    canon["is_new"]        = [mt == "new" for mt in match_types]

    n_existing = sum(1 for mt in match_types if mt == "existing")
    n_new      = sum(1 for mt in match_types if mt == "new")
    print(f"  Matched to existing: {n_existing:,}  |  New UIDs minted: {n_new:,} "
          f"({time.time()-t0:.1f}s)")

    # Safety check: no existing foia_firm_uid should be duplicated (each TRK firm
    # mapped to one foia_firm_uid; multiple TRK firms can map to the same existing one).
    new_uid_dupes = canon[canon["is_new"]]["foia_firm_uid"].duplicated().sum()
    if new_uid_dupes > 0:
        print(f"  WARNING: {new_uid_dupes} new foia_firm_uid collisions detected — "
              f"review name_base:state uniqueness in TRK_12704 data.")

    return canon


# ---------------------------------------------------------------------------
# Step E: Assemble extended dedup and save outputs
# ---------------------------------------------------------------------------

def assemble_and_save(canon: pd.DataFrame, roster: pd.DataFrame,
                      foia_dedup: pd.DataFrame,
                      cfg: dict) -> Tuple[Path, Path]:
    """
    Build foia_firms_dedup_extended.csv (superset) and trk12704_to_foiafirm.csv.
    Original foia_firms_dedup rows are left completely unchanged.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    testing = cfg.get("testing", {}).get("enabled", False)
    suffix  = "_test" if testing else ""

    # --- foia_firms_dedup_extended.csv ---
    # Tag original rows with source="bloomberg", new rows with source="trk12704"
    orig = foia_dedup.copy()
    orig["source"] = "bloomberg"

    # Build new rows (only those with is_new=True)
    new_only = canon[canon["is_new"]].copy()
    new_rows = pd.DataFrame({
        "foia_firm_uid":         new_only["foia_firm_uid"].values,
        "canonical_name_clean":  new_only["name_clean"].values,
        "canonical_name_stub":   new_only["name_stub"].values,
        "canonical_name_base":   new_only["name_base"].values,
        "token_stream":          new_only["token_stream"].values,
        "hq_state_mode":         new_only["state_norm"].values,
        "raw_name_example":      new_only["PET_FIRM_NAME"].values,
        "n_apps_trk":            new_only["n_apps_total"].values,
        "n_years_trk":           new_only["n_years_total"].values,
        "source":                "trk12704",
    })

    extended = pd.concat([orig, new_rows], ignore_index=True)
    ext_path = output_dir / f"foia_firms_dedup_extended{suffix}.csv"
    extended.to_csv(ext_path, index=False)
    print(f"\nSaved extended dedup → {ext_path}  ({len(extended):,} rows; "
          f"{len(orig):,} original + {len(new_rows):,} new TRK_12704 firms)")

    # --- trk12704_to_foiafirm.csv ---
    # roster already has _comp (added by dedup_trk_roster); map directly to foia_firm_uid
    comp_to_uid   = dict(zip(canon["component"], canon["foia_firm_uid"]))
    comp_to_mtype = dict(zip(canon["component"], canon["match_type"]))
    comp_to_score = dict(zip(canon["component"], canon["match_score"]))
    comp_to_isnew = dict(zip(canon["component"], canon["is_new"]))

    roster_with_comp = roster.copy()
    roster_with_comp["foia_firm_uid"] = roster_with_comp["_comp"].map(comp_to_uid)
    roster_with_comp["match_type"]    = roster_with_comp["_comp"].map(comp_to_mtype)
    roster_with_comp["match_score"]   = roster_with_comp["_comp"].map(comp_to_score)
    roster_with_comp["is_new"]        = roster_with_comp["_comp"].map(comp_to_isnew)

    xwalk = roster_with_comp[[
        "PET_FIRM_NAME", "state_norm", "foia_firm_uid",
        "match_type", "match_score", "is_new",
        "n_apps", "n_years", "name_base",
    ]].rename(columns={
        "PET_FIRM_NAME": "pet_firm_name",
        "state_norm":    "pet_state",
        "name_base":     "canonical_name_base",
    }).sort_values(["pet_state", "pet_firm_name"])

    xwalk_path = output_dir / f"trk12704_to_foiafirm{suffix}.csv"
    xwalk.to_csv(xwalk_path, index=False)
    print(f"Saved TRK crosswalk   → {xwalk_path}  ({len(xwalk):,} rows)")

    return ext_path, xwalk_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 70)
    print("Extend FOIA Dedup with TRK_12704 Old H-1B Data")
    print(f"Config: {_CFG_PATH}")
    testing = CFG.get("testing", {}).get("enabled", False)
    print(f"Testing mode: {testing}")
    print("=" * 70)

    # Load existing foia_firms_dedup
    foia_dedup_path = Path(CFG["foia_firms_dedup"])
    print(f"\nLoading existing foia_firms_dedup from {foia_dedup_path}...")
    foia_dedup = pd.read_csv(foia_dedup_path, dtype=str)
    print(f"  {len(foia_dedup):,} existing foia_firm_uid rows loaded")

    # Step A: Load TRK_12704
    print("\n--- Step A: Load TRK_12704 ---")
    raw = load_trk12704(CFG)

    # Step B: Build unique firm × state roster
    print("\n--- Step B: Build firm roster ---")
    roster = build_trk_roster(raw)

    if testing:
        n = CFG["testing"].get("n_firms", 500)
        roster = roster.sample(n=min(n, len(roster)), random_state=0).copy().reset_index(drop=True)
        print(f"  [TESTING] Subsampled to {len(roster):,} firms")

    # Step C: Dedup within TRK_12704
    print("\n--- Step C: Dedup within TRK_12704 ---")
    canon, mapping = dedup_trk_roster(roster, CFG)

    # Step D: Match against existing foia_firms_dedup
    print("\n--- Step D: Match TRK_12704 → existing foia_firm_uid ---")
    canon = match_trk_to_existing(canon, foia_dedup, CFG)

    # Step E: Save
    print("\n--- Step E: Save outputs ---")
    ext_path, xwalk_path = assemble_and_save(canon, roster, foia_dedup, CFG)

    # Summary
    print("\n=== Summary ===")
    print(f"  Original foia_firm_uid count:  {len(foia_dedup):,}")
    n_new = canon["is_new"].sum()
    print(f"  New TRK_12704 firms added:     {n_new:,}")
    print(f"  Extended total:                {len(foia_dedup) + n_new:,}")
    print(f"  TRK firms matched to existing: {(~canon['is_new']).sum():,}")
    print(f"\nDone. Total elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
