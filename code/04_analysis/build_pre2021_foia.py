"""
build_pre2021_foia.py
=====================
Build a pre-2021 FOIA application parquet from TRK_12704 petition data.

Background
----------
The Bloomberg FOIA CSV (foia_bloomberg_all_withids.csv) only covers the
electronic-registration era (FY2021+), so firm_outcomes.py's pre2021_lca_proxy
branch has no data to work with.  Before FY2021, FOIA records only approved
petitions (winners).  TRK_12704 Excel files contain approved H-1B beneficiary
rows for FY2000–2023.  The extend_foia_dedup_trk12704.py script already mapped
TRK_12704 firm names → foia_firm_uid via trk12704_to_foiafirm.csv.

This script:
  1. Loads TRK_12704 Excel files for configured years (default FY2008–2019).
  2. Filters to approved petitions only.
  3. Joins each petition row to foia_firm_uid via trk12704_to_foiafirm.csv,
     using (PET_FIRM_NAME, normalized_state) as the join key.
  4. Assigns synthetic foia_indiv_id integers (offset to avoid collision with
     Bloomberg integer IDs).
  5. Sets status_type = "SELECTED" (all TRK_12704 rows = petition winners).
  6. Saves the result as pre2021_foia[_test].parquet in output_dir.

Output schema (matches firm_outcomes.py _load_foia_core requirements):
  foia_indiv_id    int64   – synthetic unique row ID
  foia_firm_uid    str     – from trk12704_to_foiafirm crosswalk
  lottery_year     int64   – FIRST_DECISION_FY from TRK_12704
  status_type      str     – always "SELECTED"
  main_rcid        float   – always NaN (no Revelio ID at petition level)
  ade_lottery      float   – always NaN (ADE not recorded pre-2021)

Usage
-----
Run from repo root:
    python 04_analysis/build_pre2021_foia.py
Or import and call main() in an iPython session.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).resolve().parent.parent
_MATCH_DIR = _CODE_DIR / "revelio_h1b_company_matching"
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_MATCH_DIR))

from config import root  # noqa: E402
from company_name_cleaning import normalize_state  # noqa: E402

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
_CFG_PATH = _CODE_DIR / "configs" / "build_pre2021_foia.yaml"


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
# File registry helpers (mirrors extend_foia_dedup_trk12704.py)
# ---------------------------------------------------------------------------

def _trk_filename(year: int) -> str:
    if year == 2000:
        return f"TRK_12704_FY{year} clean.xlsx"
    return f"TRK_12704_FY{year} clean redacted.xlsx"


def _trk_sheetname(year: int) -> str:
    return f"TRK_12704_FY{year}.csv"


# ---------------------------------------------------------------------------
# Step 1: Load TRK_12704 raw petition rows
# ---------------------------------------------------------------------------

def load_trk_approved(cfg: dict) -> pd.DataFrame:
    """
    Load TRK_12704 Excel files for configured years.
    Keeps only approved petitions and returns one row per beneficiary.
    """
    raw_dir   = Path(cfg["h1b_raw_dir"])
    years     = cfg["h1b_years"]
    approved  = str(cfg.get("approved_status", "Approved")).strip()
    testing   = cfg.get("testing", {}).get("enabled", False)
    nrows     = cfg["testing"].get("n_rows") if testing else None

    print(f"Loading TRK_12704 for years {years} (approved only)...")
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
                df = pd.read_excel(
                    path, sheet_name=sheet, dtype=str,
                    nrows=nrows, engine="openpyxl",
                )
        except Exception as e:
            print(f"  FY{year}: ERROR loading {fname}: {e} — skipping")
            continue

        # Normalize column names
        df.columns = [c.strip().upper() for c in df.columns]

        # Extract needed columns
        keep = {}
        for want, candidates in [
            ("PET_FIRM_NAME",     ["PET_FIRM_NAME"]),
            ("PET_STATE",         ["PET_STATE"]),
            ("FIRST_DECISION_FY", ["FIRST_DECISION_FY"]),
            ("FIRST_DECISION",    ["FIRST_DECISION"]),
        ]:
            col = next((c for c in candidates if c in df.columns), None)
            if col:
                keep[want] = df[col]

        if "PET_FIRM_NAME" not in keep:
            print(f"  FY{year}: PET_FIRM_NAME not found — skipping")
            continue
        if "FIRST_DECISION" not in keep:
            print(f"  FY{year}: WARNING — FIRST_DECISION column not found; keeping all rows")

        chunk = pd.DataFrame(keep)
        n_raw = len(chunk)

        # Filter to approved petitions
        if "FIRST_DECISION" in chunk.columns:
            chunk = chunk[chunk["FIRST_DECISION"].str.strip() == approved].copy()
        n_approved = len(chunk)

        # Filter to rows with a usable firm name
        chunk = chunk[chunk["PET_FIRM_NAME"].notna() & (chunk["PET_FIRM_NAME"].str.strip() != "")].copy()

        # Parse fiscal year from FIRST_DECISION_FY; fall back to file year
        if "FIRST_DECISION_FY" in chunk.columns:
            chunk["lottery_year"] = pd.to_numeric(chunk["FIRST_DECISION_FY"], errors="coerce")
            chunk["lottery_year"] = chunk["lottery_year"].fillna(year).astype("int64")
        else:
            chunk["lottery_year"] = year

        print(
            f"  FY{year}: {n_raw:,} raw → {n_approved:,} approved → {len(chunk):,} with firm name"
        )
        frames.append(chunk)

    if not frames:
        raise ValueError("No TRK_12704 files loaded. Check h1b_raw_dir and h1b_years in config.")

    combined = pd.concat(frames, ignore_index=True)
    print(
        f"\n  Total: {len(combined):,} approved petition rows across "
        f"{combined['lottery_year'].nunique()} fiscal years ({time.time() - t0:.1f}s)"
    )
    return combined


# ---------------------------------------------------------------------------
# Step 2: Join to foia_firm_uid via crosswalk
# ---------------------------------------------------------------------------

def join_crosswalk(df: pd.DataFrame, crosswalk_path: str) -> pd.DataFrame:
    """
    Attach foia_firm_uid to each row by joining on (PET_FIRM_NAME, state_norm).

    The crosswalk was built from unique (PET_FIRM_NAME, normalized_state) pairs
    in the TRK_12704 data, so the join is exact and produces at most one match
    per raw row.
    """
    xwalk = pd.read_csv(
        crosswalk_path,
        usecols=["pet_firm_name", "pet_state", "foia_firm_uid"],
        dtype=str,
    )
    print(f"  Crosswalk loaded: {len(xwalk):,} (pet_firm_name, pet_state) entries")

    # Normalize state in raw data to full name (mirrors build_trk_roster logic)
    df = df.copy()
    df["state_norm"] = df["PET_STATE"].apply(
        lambda x: normalize_state(x, to="name") if pd.notna(x) else None
    )

    n_before = len(df)
    merged = df.merge(
        xwalk.rename(columns={"pet_firm_name": "PET_FIRM_NAME", "pet_state": "state_norm"}),
        on=["PET_FIRM_NAME", "state_norm"],
        how="left",
    )

    n_matched   = merged["foia_firm_uid"].notna().sum()
    n_unmatched = n_before - n_matched
    pct = 100 * n_matched / n_before if n_before else 0
    print(
        f"  Joined {n_before:,} rows → {n_matched:,} matched ({pct:.1f}%), "
        f"{n_unmatched:,} unmatched (dropped)"
    )

    # Drop rows without a foia_firm_uid (firm not in crosswalk — likely very rare)
    merged = merged[merged["foia_firm_uid"].notna()].copy()
    return merged


# ---------------------------------------------------------------------------
# Step 3: Build output schema and save
# ---------------------------------------------------------------------------

def build_and_save(df: pd.DataFrame, cfg: dict) -> Path:
    """
    Construct the final FOIA-schema parquet and save it.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    testing = cfg.get("testing", {}).get("enabled", False)
    suffix  = "_test" if testing else ""
    out_path = output_dir / f"pre2021_foia{suffix}.parquet"

    id_offset = int(cfg.get("indiv_id_offset", 10_000_000))

    # Assign synthetic foia_indiv_id: offset + integer row index
    out = pd.DataFrame()
    out["foia_indiv_id"] = np.arange(id_offset, id_offset + len(df), dtype="int64")
    out["foia_firm_uid"] = df["foia_firm_uid"].values
    out["lottery_year"]  = df["lottery_year"].values
    out["status_type"]   = "SELECTED"
    out["main_rcid"]     = np.nan
    out["ade_lottery"]   = np.nan

    # Summary before saving
    print(f"\n  Output: {len(out):,} rows")
    year_summary = (
        out.groupby("lottery_year")
        .agg(n_petitions=("foia_indiv_id", "count"), n_firms=("foia_firm_uid", "nunique"))
        .reset_index()
        .sort_values("lottery_year")
    )
    print(year_summary.to_string(index=False))

    out.to_parquet(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 70)
    print("Build Pre-2021 FOIA Parquet from TRK_12704")
    print(f"Config: {_CFG_PATH}")
    testing = CFG.get("testing", {}).get("enabled", False)
    print(f"Testing mode: {testing}")
    if testing:
        print(f"  (limiting to {CFG['testing']['n_rows']:,} rows per file)")
    print("=" * 70)

    print(f"\nYears   : {CFG['h1b_years']}")
    print(f"Raw dir : {CFG['h1b_raw_dir']}")
    print(f"Crosswalk: {CFG['trk_crosswalk']}")
    print(f"Output  : {CFG['output_dir']}")

    # Step 1: Load raw approved petitions
    print("\n--- Step 1: Load TRK_12704 approved petitions ---")
    raw = load_trk_approved(CFG)

    # Step 2: Join to foia_firm_uid
    print("\n--- Step 2: Join crosswalk → foia_firm_uid ---")
    merged = join_crosswalk(raw, CFG["trk_crosswalk"])

    # Step 3: Build output and save
    print("\n--- Step 3: Build output schema and save ---")
    out_path = build_and_save(merged, CFG)

    print(f"\nDone. Total elapsed: {time.time() - t_start:.0f}s")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
