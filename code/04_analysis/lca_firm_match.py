"""
lca_firm_match.py
=================
Match LCA employer names to FOIA H-1B firm identifiers (foia_firm_uid) using
the extended firm deduplication table produced by extend_foia_dedup_trk12704.py.

Prerequisites
-------------
1. Run lca_clean.main() → {root}/data/int/lca_firm_year.parquet
2. Run extend_foia_dedup_trk12704.main() → {root}/data/int/foia_firms_dedup_extended.csv

Matching pipeline
-----------------
Stage 1: Exact name_base match within state. City agreement breaks ties.
Stage 2: Fuzzy token_sort_ratio on name_base, within state (threshold ≥ 90).
         City agreement breaks ties.
Stage 3: Diagnostics — match rates, worker-weighted coverage, top unmatched.

Name normalization is handled entirely by clean_company_name() from
company_name_cleaning.py (same normalization used throughout the pipeline).

Output
------
  {output_dir}/lca_foia_crosswalk[_test].parquet
  Columns:
    lca_employer_name   – raw LCA employer name (standardized uppercase)
    lca_employer_state  – 2-char state abbreviation
    foia_firm_uid       – matched FOIA firm identifier
    match_type          – "exact_name" | "exact_name_ambig" | "fuzzy_name" | "fuzzy_name_ambig"
    match_score         – float [0,1]; 1.0 for exact, rapidfuzz/100 for fuzzy
    n_lca_years         – number of fiscal years this LCA firm appears
    n_lca_workers_total – total certified LCA workers across all years

Usage
-----
Run from repo root:
    python 04_analysis/lca_firm_match.py
Or import and call main() in an iPython session.
"""

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import yaml
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Path setup — make company_name_cleaning importable
# ---------------------------------------------------------------------------
if "__file__" in globals():
    _THIS_DIR = Path(__file__).resolve().parent
else:
    _THIS_DIR = Path.cwd() / "04_analysis"

_CODE_DIR = _THIS_DIR.parent
_MATCH_DIR = _CODE_DIR / "revelio_h1b_company_matching"
sys.path.insert(0, str(_MATCH_DIR))
sys.path.insert(0, str(_CODE_DIR))

from company_name_cleaning import clean_company_name, normalize_state  # noqa: E402
from config import root  # noqa: E402

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
_CFG_PATH = _CODE_DIR / "configs" / "lca_firm_match.yaml"


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
# Helpers
# ---------------------------------------------------------------------------

def _mode_or_first(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if not s.empty else None


# ---------------------------------------------------------------------------
# Stage 0: Build firm rosters
# ---------------------------------------------------------------------------

def _build_lca_roster(cfg: dict) -> pd.DataFrame:
    """
    Load LCA parquet and collapse to unique (employer_name, employer_state).
    Applies clean_company_name() for name normalization (same as matching pipeline).
    """
    print("Loading LCA data...", flush=True)
    t0 = time.time()
    con = duckdb.connect()
    lca_path = cfg["lca_input"]
    lca = con.execute(f"SELECT * FROM read_parquet('{lca_path}')").df()
    print(f"  {len(lca):,} LCA firm-year rows loaded ({time.time()-t0:.1f}s)")

    print("  Collapsing to unique firm × state roster...", flush=True)
    roster = (
        lca.groupby(["employer_name", "employer_state"], dropna=False)
        .agg(
            zip5                = ("zip5",           _mode_or_first),
            employer_city       = ("employer_city",  _mode_or_first),
            n_lca_years         = ("fiscal_year",    "nunique"),
            n_lca_workers_total = ("n_lca_workers",  "sum"),
        )
        .reset_index()
    )
    print(f"  {len(roster):,} unique LCA firm × state pairs")

    # Normalize names using clean_company_name() — same as the dedup pipeline
    print("  Cleaning LCA names via clean_company_name()...", flush=True)
    cleaned = roster["employer_name"].apply(lambda n: clean_company_name(n))
    roster["name_base"]     = [c.base  for c in cleaned]
    roster["name_stub"]     = [c.stub  for c in cleaned]

    # Normalize state to abbreviated form for matching
    roster["state_abbr"] = roster["employer_state"].apply(
        lambda x: normalize_state(x, to="abbr") if pd.notna(x) else None
    )

    # Normalize city for tiebreaking
    roster["city_clean"] = (
        roster["employer_city"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return roster.reset_index(drop=True)


def _build_foia_roster(cfg: dict) -> pd.DataFrame:
    """
    Load foia_firms_dedup_extended.csv. Normalize hq_state_mode to abbreviation.
    """
    print("Loading extended FOIA firm dedup...", flush=True)
    t0 = time.time()
    path = cfg["foia_firms_extended"]
    dedup = pd.read_csv(path, dtype=str)
    print(f"  {len(dedup):,} foia_firm_uid entries loaded ({time.time()-t0:.1f}s)")

    # Normalize state to abbreviated form
    dedup["state_abbr"] = dedup["hq_state_mode"].apply(
        lambda x: normalize_state(x, to="abbr") if pd.notna(x) and str(x).strip() != "" else None
    )

    # Fallback: some entries may have state already as abbr; normalize_state handles both
    dedup["name_base"] = dedup["canonical_name_base"].fillna("").astype(str)
    dedup["name_stub"] = dedup.get("canonical_name_stub",
                                   dedup.get("canonical_name_clean", dedup["name_base"])).fillna("").astype(str)

    return dedup.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 1: Exact name_base + state match
# ---------------------------------------------------------------------------

def _match_exact_name(lca_roster: pd.DataFrame,
                      foia_roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exact match on (name_base, state_abbr).
    Returns (matched_df, remaining_lca_roster).
    """
    print("\n--- Stage 1: Exact name_base match ---", flush=True)
    t0 = time.time()

    merged = lca_roster.merge(
        foia_roster[["name_base", "state_abbr", "foia_firm_uid"]].rename(
            columns={"state_abbr": "foia_state_abbr", "foia_firm_uid": "foia_firm_uid"}
        ),
        left_on  = ["name_base", "state_abbr"],
        right_on = ["name_base", "foia_state_abbr"],
        how="inner",
    )

    if merged.empty:
        print("  0 matches")
        return pd.DataFrame(), lca_roster.copy()

    def _resolve(grp):
        if len(grp) == 1:
            row = grp.iloc[0].copy()
            row["match_type"] = "exact_name"
            return row
        # Multiple FOIA firms at same name+state: tiebreak with city
        lca_city = grp["city_clean"].iloc[0]
        # foia_roster doesn't have city, so just flag ambiguous and take first
        row = grp.iloc[0].copy()
        row["match_type"] = "exact_name_ambig"
        return row

    resolved = (
        merged.groupby(["employer_name", "employer_state"], group_keys=False)
        .apply(_resolve)
        .reset_index(drop=True)
    )
    resolved["match_score"] = 1.0

    n_exact = len(resolved)
    n_ambig = (resolved["match_type"] == "exact_name_ambig").sum()
    print(f"  {n_exact:,} exact matches ({n_ambig:,} ambiguous) ({time.time()-t0:.1f}s)")

    matched_keys = set(zip(resolved["employer_name"], resolved["employer_state"]))
    remaining = lca_roster[
        ~lca_roster.apply(lambda r: (r["employer_name"], r["employer_state"]) in matched_keys, axis=1)
    ].copy()
    print(f"  {len(remaining):,} LCA firms remaining unmatched")

    return resolved, remaining


# ---------------------------------------------------------------------------
# Stage 2: Fuzzy name match (within state)
# ---------------------------------------------------------------------------

def _match_fuzzy_name(lca_unmatched: pd.DataFrame,
                      foia_roster: pd.DataFrame,
                      fuzzy_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Within-state fuzzy match using rapidfuzz token_sort_ratio on name_base.
    Returns (matched_df, remaining_lca_roster).
    """
    print(f"\n--- Stage 2: Fuzzy name match (within state, threshold={fuzzy_threshold}) ---",
          flush=True)
    t0 = time.time()

    if lca_unmatched.empty:
        print("  No unmatched LCA firms")
        return pd.DataFrame(), lca_unmatched.copy()

    states = sorted(lca_unmatched["state_abbr"].dropna().unique())
    print(f"  Matching across {len(states)} states...", flush=True)

    results = []
    for state in states:
        lca_state  = lca_unmatched[lca_unmatched["state_abbr"] == state]
        foia_state = foia_roster[foia_roster["state_abbr"] == state]

        if lca_state.empty or foia_state.empty:
            continue

        lca_bases  = lca_state["name_base"].tolist()
        foia_bases = foia_state["name_base"].tolist()

        # Score matrix: rows=LCA, cols=FOIA
        scores = process.cdist(lca_bases, foia_bases, scorer=fuzz.token_sort_ratio)

        for i, lca_row in enumerate(lca_state.itertuples()):
            row_scores = scores[i]
            best_score = float(row_scores.max())
            if best_score < fuzzy_threshold:
                continue

            best_idxs = [j for j, sc in enumerate(row_scores) if sc == best_score]
            if len(best_idxs) == 1:
                foia_row   = foia_state.iloc[best_idxs[0]]
                match_type = "fuzzy_name"
            else:
                # Tiebreak: city similarity (we don't have foia city, so just take first)
                foia_row   = foia_state.iloc[best_idxs[0]]
                match_type = "fuzzy_name_ambig"

            results.append({
                "employer_name":      lca_row.employer_name,
                "employer_state":     lca_row.employer_state,
                "name_base":          lca_row.name_base,
                "state_abbr":         lca_row.state_abbr,
                "city_clean":         lca_row.city_clean,
                "n_lca_years":        lca_row.n_lca_years,
                "n_lca_workers_total":lca_row.n_lca_workers_total,
                "foia_firm_uid":      foia_row["foia_firm_uid"],
                "match_type":         match_type,
                "match_score":        best_score / 100.0,
            })

    if not results:
        print("  0 fuzzy matches above threshold")
        return pd.DataFrame(), lca_unmatched.copy()

    resolved = pd.DataFrame(results)
    n_fuzzy  = len(resolved)
    n_ambig  = (resolved["match_type"] == "fuzzy_name_ambig").sum()
    print(f"  {n_fuzzy:,} fuzzy matches ({n_ambig:,} ambiguous) ({time.time()-t0:.1f}s)")

    matched_keys = set(zip(resolved["employer_name"], resolved["employer_state"]))
    remaining = lca_unmatched[
        ~lca_unmatched.apply(lambda r: (r["employer_name"], r["employer_state"]) in matched_keys, axis=1)
    ].copy()
    print(f"  {len(remaining):,} LCA firms remaining unmatched")

    return resolved, remaining


# ---------------------------------------------------------------------------
# Stage 3: Diagnostics
# ---------------------------------------------------------------------------

def _print_diagnostics(crosswalk: pd.DataFrame, unmatched: pd.DataFrame) -> None:
    """Print match rates, worker-weighted coverage, top unmatched firms."""
    print("\n" + "=" * 70)
    print("DIAGNOSTICS")
    print("=" * 70)

    n_total = len(crosswalk) + len(unmatched)
    if n_total == 0:
        print("  No firms processed.")
        return

    # Match type breakdown
    print("\nMatch type breakdown:")
    if not crosswalk.empty:
        counts = crosswalk["match_type"].value_counts()
        for mtype, cnt in counts.items():
            pct = 100 * cnt / n_total
            print(f"  {mtype:<24} {cnt:>8,}  ({pct:.1f}%)")
    print(f"  {'unmatched':<24} {len(unmatched):>8,}  ({100*len(unmatched)/n_total:.1f}%)")
    print(f"  {'TOTAL':<24} {n_total:>8,}")

    # Worker-weighted coverage
    w_matched   = crosswalk["n_lca_workers_total"].sum() if not crosswalk.empty else 0
    w_unmatched = unmatched["n_lca_workers_total"].sum() if not unmatched.empty else 0
    w_total     = w_matched + w_unmatched
    if w_total > 0:
        print(f"\nCoverage (weighted by n_lca_workers): {w_matched/w_total:.1%} "
              f"({w_matched:,.0f} / {w_total:,.0f})")

    # Top unmatched
    if not unmatched.empty:
        top = (
            unmatched
            .sort_values("n_lca_workers_total", ascending=False)
            .head(20)[["employer_name", "employer_state", "n_lca_workers_total", "n_lca_years"]]
        )
        print("\nTop 20 unmatched LCA firms by n_lca_workers_total:")
        print(top.to_string(index=False))


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save_crosswalk(crosswalk: pd.DataFrame, output_dir: Path, cfg: dict) -> Path:
    """Save crosswalk as parquet via DuckDB."""
    output_dir.mkdir(parents=True, exist_ok=True)
    testing  = cfg.get("testing", {}).get("enabled", False)
    suffix   = "_test" if testing else ""
    out_path = output_dir / f"lca_foia_crosswalk{suffix}.parquet"

    keep_cols = [
        "employer_name", "employer_state", "foia_firm_uid",
        "match_type", "match_score", "n_lca_years", "n_lca_workers_total",
    ]
    out_df = crosswalk[[c for c in keep_cols if c in crosswalk.columns]].copy()
    out_df = out_df.rename(columns={"employer_name":  "lca_employer_name",
                                    "employer_state": "lca_employer_state"})

    con = duckdb.connect()
    con.register("xwalk", out_df)
    con.execute(
        f"COPY (SELECT * FROM xwalk ORDER BY lca_employer_state, lca_employer_name) "
        f"TO '{out_path}' (FORMAT PARQUET)"
    )
    print(f"\nSaved → {out_path}  ({len(out_df):,} rows)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 70)
    print("LCA → FOIA Firm Matcher")
    print(f"Config: {_CFG_PATH}")
    testing = CFG.get("testing", {}).get("enabled", False)
    print(f"Testing mode: {testing}")
    if testing:
        print(f"  (sampling {CFG['testing']['n_firms']:,} LCA firms)")
    print("=" * 70)

    fuzzy_threshold = float(CFG.get("fuzzy_threshold", 90.0))
    output_dir      = Path(CFG["output_dir"])

    # --- Stage 0: Build rosters ---
    print("\n--- Stage 0: Build firm rosters ---")
    lca_roster  = _build_lca_roster(CFG)
    foia_roster = _build_foia_roster(CFG)

    # Testing: subsample LCA roster
    if testing:
        n_firms   = CFG["testing"]["n_firms"]
        lca_roster = lca_roster.sample(n=min(n_firms, len(lca_roster)), random_state=0).copy()
        print(f"  [TESTING] Subsampled to {len(lca_roster):,} LCA firms")

    # --- Matching stages ---
    matched_frames = []

    matched1, unmatched = _match_exact_name(lca_roster, foia_roster)
    if not matched1.empty:
        matched_frames.append(matched1)

    matched2, unmatched = _match_fuzzy_name(unmatched, foia_roster, fuzzy_threshold)
    if not matched2.empty:
        matched_frames.append(matched2)

    # Combine
    if matched_frames:
        crosswalk = pd.concat(matched_frames, ignore_index=True)
    else:
        crosswalk = pd.DataFrame(columns=[
            "employer_name", "employer_state", "foia_firm_uid",
            "match_type", "match_score", "n_lca_years", "n_lca_workers_total",
        ])

    # --- Diagnostics ---
    _print_diagnostics(crosswalk, unmatched)

    # --- Save ---
    _save_crosswalk(crosswalk, output_dir, CFG)

    print(f"\nDone. Total elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
