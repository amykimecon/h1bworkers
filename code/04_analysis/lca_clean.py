"""
lca_clean.py
============
Download and clean H-1B LCA (Labor Condition Application) disclosure data from
the DOL OFLC website for fiscal years 2005–2019.

Purpose
-------
Pre-2020 H-1B administrative FOIA data records only petition approvals, not
lottery losers, so the denominator (total applications per firm) is unknown.
Following the literature (Kerr & Lincoln 2010; Peri, Shih & Sparber 2015),
we use certified LCA filings as a proxy for total H-1B applications per firm
per year.

Output
------
  {output_dir}/lca_firm_year[_test].parquet
  Columns:
    employer_name        – standardized uppercase firm name
    employer_state       – 2-char US state abbreviation
    employer_address     – street address (nullable for older years)
    employer_city        – city (nullable for older years)
    employer_postal_code – ZIP / postal code (nullable for older years)
    zip5                 – 5-digit ZIP prefix extracted from employer_postal_code
    fiscal_year          – integer fiscal year
    n_lca_workers        – sum of certified LCA worker positions (main proxy)
    n_lca_cases          – count of certified LCA case filings (secondary proxy)

Usage
-----
Run from the repo root:
    python 04_analysis/lca_clean.py
Or import and call main() in an iPython session.
"""

import os
import re
import sys
import shutil
import subprocess
import time
from pathlib import Path
import warnings

import duckdb
import pandas as pd
import requests
import yaml

# ---------------------------------------------------------------------------
# Resolve root & code dir (mirrors pattern in firm_outcomes.py)
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_DIR))
from config import root  # noqa: E402

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
_CFG_PATH = _CODE_DIR / "configs" / "lca_clean.yaml"


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
# File registry
# Each entry: filename, DOL URL (None = use legacy local file), column family,
#             whether a VISA_CLASS filter is needed, and whether there are
#             multiple source files for the year (list of dicts).
# ---------------------------------------------------------------------------
DOL_BASE = CFG["dol_base_url"].rstrip("/")

# Column families define how to map raw column names to standard names.
# We detect the actual family at runtime in case DOL slightly changes headers.
COL_FAMILIES = {
    # FY2005-2007: old efile/fax CSV format
    "legacy_csv": {
        "employer_name":        ["EMPLOYER_NAME", "NAME"],
        "total_workers":        ["NBR_IMMIGRANTS", "NBR_WORKERS"],
        "case_status":          ["CASE_STATUS"],
        "employer_state":       ["STATE"],
        "employer_address":     ["ADDRESS_1", "ADDRESS1"],
        "employer_city":        ["CITY"],
        "employer_postal_code": ["ZIP_CODE", "POSTAL_CODE"],
        "visa_class":           [],   # no visa class col – file is H-1B only
    },
    # FY2008-2009: iCert old format
    "icert_old": {
        "employer_name":        ["EMPLOYER_NAME", "NAME"],
        "total_workers":        ["NBR_IMMIGRANTS", "NBR_WORKERS", "TOTAL_WORKERS"],
        "case_status":          ["CASE_STATUS", "STATUS"],
        "employer_state":       ["EMPLOYER_STATE", "STATE"],
        "employer_address":     ["EMPLOYER_ADDRESS1", "EMPLOYER_ADDRESS", "ADDRESS1"],
        "employer_city":        ["EMPLOYER_CITY", "CITY"],
        "employer_postal_code": ["EMPLOYER_POSTAL_CODE", "POSTAL_CODE"],
        "visa_class":           [],   # H-1B only
    },
    # FY2010-2014: iCert new format (LCA_CASE_* prefix)
    "icert_new": {
        "employer_name":        ["LCA_CASE_EMPLOYER_NAME"],
        "total_workers":        ["LCA_CASE_WORKER_COUNT", "TOTAL_WORKERS"],
        "case_status":          ["STATUS", "CASE_STATUS"],
        "employer_state":       ["LCA_CASE_EMPLOYER_STATE", "EMPLOYER_STATE"],
        "employer_address":     ["LCA_CASE_EMPLOYER_ADDRESS", "EMPLOYER_ADDRESS"],
        "employer_city":        ["LCA_CASE_EMPLOYER_CITY", "EMPLOYER_CITY"],
        "employer_postal_code": ["LCA_CASE_EMPLOYER_POSTAL_CODE", "EMPLOYER_POSTAL_CODE"],
        "visa_class":           ["VISA_CLASS"],
    },
    # FY2015-2019: new OFLC disclosure format
    "oflc_new": {
        "employer_name":        ["EMPLOYER_NAME"],
        "total_workers":        ["TOTAL_WORKERS"],  # space normalised to underscore below
        "case_status":          ["CASE_STATUS"],
        "employer_state":       ["EMPLOYER_STATE"],
        "employer_address":     ["EMPLOYER_ADDRESS", "EMPLOYER_ADDRESS1"],
        "employer_city":        ["EMPLOYER_CITY"],
        "employer_postal_code": ["EMPLOYER_POSTAL_CODE"],
        "visa_class":           ["VISA_CLASS"],
    },
}

# Registry: year → list of file specs (most years have 1 file; FY2009 has 2)
FILE_REGISTRY = {
    # ---- Legacy local files -----------------------------------------------
    # FY2005–2006 stored as Access MDB locally; no DOL URL.
    # Will attempt mdb-export conversion at runtime; skipped if mdb-tools absent.
    2005: [{"filename": "H1B_efile_FY05.mdb",           "url": None, "fmt": "mdb",  "family": "legacy_csv", "visa_filter": False}],
    2006: [{"filename": "H1B_efile_FY06.mdb",           "url": None, "fmt": "mdb",  "family": "legacy_csv", "visa_filter": False}],
    2007: [{"filename": "EFILE_FY2007_DATA.csv",         "url": None, "fmt": "csv",  "family": "legacy_csv", "visa_filter": False}],
    # ---- DOL XLSX downloads -----------------------------------------------
    2008: [{"filename": "H-1B_Case_Data_FY2008.xlsx",            "url": f"{DOL_BASE}/H-1B_Case_Data_FY2008.xlsx",            "fmt": "xlsx", "family": "icert_old", "visa_filter": False}],
    # FY2009: two files (legacy + iCert transition year); both are H-1B only
    2009: [
        {"filename": "H-1B_Case_Data_FY2009.xlsx",       "url": f"{DOL_BASE}/H-1B_Case_Data_FY2009.xlsx",       "fmt": "xlsx", "family": "icert_old", "visa_filter": False},
        {"filename": "Icert_LCA_FY2009.xlsx",            "url": f"{DOL_BASE}/Icert_LCA_FY2009.xlsx",            "fmt": "xlsx", "family": "icert_new", "visa_filter": True},
    ],
    2010: [{"filename": "H-1B_FY2010.xlsx",                      "url": f"{DOL_BASE}/H-1B_FY2010.xlsx",                      "fmt": "xlsx", "family": "icert_new", "visa_filter": True}],
    2011: [{"filename": "H-1B_iCert_LCA_FY2011_Q4.xlsx",         "url": f"{DOL_BASE}/H-1B_iCert_LCA_FY2011_Q4.xlsx",         "fmt": "xlsx", "family": "icert_new", "visa_filter": True}],
    2012: [{"filename": "LCA_FY2012_Q4.xlsx",                     "url": f"{DOL_BASE}/LCA_FY2012_Q4.xlsx",                     "fmt": "xlsx", "family": "icert_new", "visa_filter": True}],
    2013: [{"filename": "LCA_FY2013.xlsx",                        "url": f"{DOL_BASE}/LCA_FY2013.xlsx",                        "fmt": "xlsx", "family": "icert_new", "visa_filter": True}],
    2014: [{"filename": "H-1B_FY14_Q4.xlsx",                     "url": f"{DOL_BASE}/H-1B_FY14_Q4.xlsx",                     "fmt": "xlsx", "family": "icert_new", "visa_filter": True}],
    2015: [{"filename": "H-1B_Disclosure_Data_FY15_Q4.xlsx",     "url": f"{DOL_BASE}/H-1B_Disclosure_Data_FY15_Q4.xlsx",     "fmt": "xlsx", "family": "oflc_new",  "visa_filter": True}],
    2016: [{"filename": "H-1B_Disclosure_Data_FY16.xlsx",        "url": f"{DOL_BASE}/H-1B_Disclosure_Data_FY16.xlsx",        "fmt": "xlsx", "family": "oflc_new",  "visa_filter": True}],
    2017: [{"filename": "H-1B_Disclosure_Data_FY17.xlsx",        "url": f"{DOL_BASE}/H-1B_Disclosure_Data_FY17.xlsx",        "fmt": "xlsx", "family": "oflc_new",  "visa_filter": True}],
    2018: [{"filename": "H-1B_Disclosure_Data_FY2018_EOY.xlsx",  "url": f"{DOL_BASE}/H-1B_Disclosure_Data_FY2018_EOY.xlsx",  "fmt": "xlsx", "family": "oflc_new",  "visa_filter": True}],
    2019: [{"filename": "H-1B_Disclosure_Data_FY2019.xlsx",      "url": f"{DOL_BASE}/H-1B_Disclosure_Data_FY2019.xlsx",      "fmt": "xlsx", "family": "oflc_new",  "visa_filter": True}],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase all column names, collapse whitespace to underscore."""
    df.columns = [re.sub(r"\s+", "_", c.strip().upper()) for c in df.columns]
    return df


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _standardize_name(s: pd.Series) -> pd.Series:
    """Uppercase, strip, collapse whitespace, drop trailing punctuation."""
    s = s.fillna("").astype(str).str.upper().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"[,\.;]+$", "", regex=True).str.strip()
    return s


def _standardize_state(s: pd.Series) -> pd.Series:
    """Keep only 2-char alpha state codes; blank out anything else."""
    s = s.fillna("").astype(str).str.upper().str.strip()
    return s.where(s.str.match(r"^[A-Z]{2}$"), other=None)


def _mode_or_first(s: pd.Series) -> str | None:
    """Return mode value (first if tie), or None if all NaN."""
    s = s.dropna()
    if s.empty:
        return None
    return s.mode().iloc[0]


# ---------------------------------------------------------------------------
# Step 3: Download
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Stream-download url to dest. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading {url} (attempt {attempt})...")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100 * downloaded / total
                            print(f"    {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.0f}%)", end="\r")
            print(f"  -> Saved to {dest} ({downloaded / 1e6:.1f} MB)")
            return True
        except Exception as e:
            print(f"  WARNING: attempt {attempt} failed: {e}")
            if dest.exists():
                dest.unlink()
            if attempt < retries:
                time.sleep(5)
    print(f"  ERROR: all {retries} download attempts failed for {url}. Skipping year.")
    return False


def _export_mdb(mdb_path: Path, dest_csv: Path) -> bool:
    """Use mdb-export (mdb-tools) to dump the main table from an MDB file to CSV."""
    try:
        tables_raw = subprocess.check_output(
            ["mdb-tables", "-1", str(mdb_path)], stderr=subprocess.DEVNULL
        ).decode().splitlines()
        tables = [t for t in tables_raw if t.strip()]
        if not tables:
            print(f"  WARNING: no tables found in {mdb_path.name}")
            return False
        # Pick the largest table (most rows) as the main LCA table
        target = tables[0]
        print(f"  Exporting MDB table '{target}' from {mdb_path.name}...")
        output = subprocess.check_output(
            ["mdb-export", str(mdb_path), target], stderr=subprocess.DEVNULL
        )
        dest_csv.write_bytes(output)
        print(f"  -> Saved to {dest_csv}")
        return True
    except FileNotFoundError:
        print(f"  WARNING: mdb-tools not installed; cannot process {mdb_path.name}. Skipping.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: mdb-export failed for {mdb_path.name}: {e}")
        return False


def download_all(years: list[int], raw_dir: Path, legacy_dir: Path) -> None:
    """Download / locate all files, placing them in raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for year in sorted(years):
        specs = FILE_REGISTRY.get(year, [])
        if not specs:
            print(f"FY{year}: no registry entry — skipping")
            continue

        for spec in specs:
            fname = spec["filename"]
            dest = raw_dir / fname

            if dest.exists():
                print(f"FY{year}: {fname} already present — skipping download")
                continue

            url = spec["url"]

            # DOL download
            if url is not None:
                _download_file(url, dest)
                continue

            # Legacy local file (MDB or CSV in lca_legacy_dir)
            fmt = spec["fmt"]
            legacy_src = legacy_dir / fname

            if fmt == "csv":
                if legacy_src.exists():
                    shutil.copy2(legacy_src, dest)
                    print(f"FY{year}: copied {fname} from legacy dir")
                else:
                    print(f"FY{year}: WARNING — {fname} not found in legacy dir; skipping")

            elif fmt == "mdb":
                # Convert MDB → CSV and save as .csv
                csv_dest = dest.with_suffix(".csv")
                if csv_dest.exists():
                    print(f"FY{year}: {csv_dest.name} already present — skipping MDB export")
                    continue
                if legacy_src.exists():
                    _export_mdb(legacy_src, csv_dest)
                else:
                    print(f"FY{year}: WARNING — {fname} not found in legacy dir; skipping")

    print(f"\nDownload phase complete ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# Step 4: Process each year
# ---------------------------------------------------------------------------

def _load_file(path: Path, fmt: str, nrows: int | None) -> pd.DataFrame:
    """Load XLSX or CSV into a DataFrame. nrows=None for full load."""
    if fmt in ("xlsx",):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_excel(path, nrows=nrows, engine="openpyxl", dtype=str)
    else:
        # csv (including MDB exports)
        df = pd.read_csv(path, nrows=nrows, dtype=str, on_bad_lines="skip", encoding_errors="replace")
    return df


def _process_year_file(spec: dict, raw_dir: Path, cfg: dict) -> pd.DataFrame | None:
    """
    Load, filter, and aggregate one year's LCA file.
    Returns a firm-state-level DataFrame with standardized columns, or None on failure.
    """
    fname = spec["filename"]
    fmt   = spec["fmt"]
    year  = spec["_year"]

    # MDB was converted to CSV at download time
    if fmt == "mdb":
        fname = Path(fname).stem + ".csv"
        fmt   = "csv"

    path = raw_dir / fname
    if not path.exists():
        print(f"  FY{year} [{fname}]: file not found — skipping")
        return None

    testing = cfg.get("testing", {}).get("enabled", False)
    nrows   = cfg["testing"]["n_rows"] if testing else None

    # --- Load ---
    print(f"  FY{year} [{fname}]: loading...", end=" ", flush=True)
    try:
        df = _load_file(path, fmt, nrows)
    except Exception as e:
        print(f"\n  ERROR loading {fname}: {e} — skipping")
        return None

    df = _normalize_cols(df)
    n_raw = len(df)
    print(f"{n_raw:,} rows")

    # --- Map columns from family definition ---
    family_map = COL_FAMILIES[spec["family"]]
    std = {}
    for std_name, candidates in family_map.items():
        col = _resolve_col(df, candidates)
        if col is not None:
            std[std_name] = col
        # Missing cols are allowed (address fields optional; visa_class may be absent)

    if "employer_name" not in std:
        print(f"  ERROR: cannot find employer_name column in {fname} (cols: {df.columns.tolist()}) — skipping")
        return None
    if "case_status" not in std:
        print(f"  ERROR: cannot find case_status column in {fname} — skipping")
        return None

    # --- Visa class filter ---
    visa_classes = [v.upper() for v in cfg.get("visa_class_filter", ["H-1B"])]
    if spec["visa_filter"] and "visa_class" in std:
        n_before = len(df)
        df = df[df[std["visa_class"]].str.upper().str.strip().isin(visa_classes)].copy()
        print(f"  FY{year}: visa filter {n_before:,} -> {len(df):,}")

    # --- Certified status filter ---
    incl_withdrawn = cfg.get("include_withdrawn", False)
    status_col = df[std["case_status"]].str.upper().str.strip()
    if incl_withdrawn:
        mask = status_col.str.startswith("CERTIFIED")
    else:
        mask = status_col == "CERTIFIED"
    n_before = len(df)
    df = df[mask].copy()
    print(f"  FY{year}: cert filter {n_before:,} -> {len(df):,}")

    if df.empty:
        print(f"  FY{year}: WARNING — 0 certified rows; skipping")
        return None

    # --- Extract & standardize fields ---
    out = pd.DataFrame()
    out["employer_name"] = _standardize_name(df[std["employer_name"]])

    state_col = std.get("employer_state")
    out["employer_state"] = _standardize_state(df[state_col]) if state_col else None

    addr_col = std.get("employer_address")
    out["employer_address"] = df[addr_col].str.strip().str.upper() if addr_col else None

    city_col = std.get("employer_city")
    out["employer_city"] = df[city_col].str.strip().str.upper() if city_col else None

    zip_col = std.get("employer_postal_code")
    out["employer_postal_code"] = df[zip_col].str.strip() if zip_col else None
    out["zip5"] = out["employer_postal_code"].str.extract(r"(\d{5})")[0] if zip_col else None

    # Workers: coerce to int, fill missing with 1 (conservative)
    if "total_workers" in std:
        out["total_workers"] = pd.to_numeric(df[std["total_workers"]], errors="coerce").fillna(1).astype(int)
    else:
        out["total_workers"] = 1

    out["fiscal_year"] = year

    # --- Aggregate to employer_name × employer_state ---
    groupby_cols = ["employer_name", "employer_state", "fiscal_year"]
    agg = (
        out.groupby(groupby_cols, dropna=False)
        .agg(
            n_lca_workers        = ("total_workers",        "sum"),
            n_lca_cases          = ("total_workers",        "count"),
            employer_address     = ("employer_address",     _mode_or_first),
            employer_city        = ("employer_city",        _mode_or_first),
            employer_postal_code = ("employer_postal_code", _mode_or_first),
            zip5                 = ("zip5",                 _mode_or_first),
        )
        .reset_index()
    )

    print(f"  FY{year}: {len(agg):,} employer-state combinations after aggregation")
    return agg


def process_all_years(years: list[int], raw_dir: Path, cfg: dict) -> list[pd.DataFrame]:
    """Process all years in the registry; return list of aggregated DataFrames."""
    frames = []
    t0 = time.time()

    for year in sorted(years):
        specs = FILE_REGISTRY.get(year, [])
        if not specs:
            continue
        year_frames = []
        for spec in specs:
            spec["_year"] = year
            df = _process_year_file(spec, raw_dir, cfg)
            if df is not None:
                year_frames.append(df)

        if not year_frames:
            continue

        if len(year_frames) > 1:
            # FY2009: combine iCert + legacy; sum workers for same employer-state (dedup)
            combined = pd.concat(year_frames, ignore_index=True)
            groupby_cols = ["employer_name", "employer_state", "fiscal_year"]
            combined = (
                combined.groupby(groupby_cols, dropna=False)
                .agg(
                    n_lca_workers        = ("n_lca_workers",        "sum"),
                    n_lca_cases          = ("n_lca_cases",          "sum"),
                    employer_address     = ("employer_address",      _mode_or_first),
                    employer_city        = ("employer_city",         _mode_or_first),
                    employer_postal_code = ("employer_postal_code",  _mode_or_first),
                    zip5                 = ("zip5",                  _mode_or_first),
                )
                .reset_index()
            )
            print(f"  FY{year}: combined {len(combined):,} firm-state rows from {len(year_frames)} files")
            frames.append(combined)
        else:
            frames.append(year_frames[0])

    print(f"\nProcessing phase complete ({time.time() - t0:.0f}s)")
    return frames


# ---------------------------------------------------------------------------
# Step 5: Combine & save
# ---------------------------------------------------------------------------

def combine_and_save(frames: list[pd.DataFrame], output_dir: Path, cfg: dict) -> Path:
    """Union all year frames via DuckDB and save as parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    testing = cfg.get("testing", {}).get("enabled", False)
    suffix  = "_test" if testing else ""
    out_path = output_dir / f"lca_firm_year{suffix}.parquet"

    print(f"\nCombining {len(frames)} year frames via DuckDB...")
    t0 = time.time()

    con = duckdb.connect()

    # Register each frame as a DuckDB view
    for i, df in enumerate(frames):
        con.register(f"frame_{i}", df)

    union_sql = " UNION ALL ".join([f"SELECT * FROM frame_{i}" for i in range(len(frames))])
    combined = con.execute(f"SELECT * FROM ({union_sql})").df()
    n_total = len(combined)
    n_years = combined["fiscal_year"].nunique()
    print(f"  Combined: {n_total:,} rows across {n_years} fiscal years")

    # Save
    con.execute(f"""
        COPY (SELECT * FROM ({union_sql}) ORDER BY fiscal_year, employer_state, employer_name)
        TO '{out_path}' (FORMAT PARQUET)
    """)
    print(f"  Saved -> {out_path}")
    print(f"  Combine phase complete ({time.time() - t0:.0f}s)")

    # --- Summary ---
    print("\n=== Summary: rows per fiscal year ===")
    summary = (
        combined.groupby("fiscal_year")
        .agg(n_firms=("employer_name", "nunique"), n_lca_workers=("n_lca_workers", "sum"))
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\n=== Top 15 employers by total certified LCA workers (all years) ===")
    top = (
        combined.groupby("employer_name")["n_lca_workers"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    print(top.to_string(index=False))

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 70)
    print("LCA Disclosure Data Cleaner")
    print(f"Config: {_CFG_PATH}")
    testing = CFG.get("testing", {}).get("enabled", False)
    print(f"Testing mode: {testing}")
    if testing:
        print(f"  (limiting to {CFG['testing']['n_rows']:,} rows per file)")
    print("=" * 70)

    years       = CFG["years"]
    raw_dir     = Path(CFG["lca_raw_dir"])
    legacy_dir  = Path(CFG["lca_legacy_dir"])
    output_dir  = Path(CFG["output_dir"])

    print(f"\nYears to process: {years}")
    print(f"Raw dir:    {raw_dir}")
    print(f"Legacy dir: {legacy_dir}")
    print(f"Output dir: {output_dir}\n")

    # Step 3: Download missing files
    print("--- Step 1: Download / locate files ---")
    download_all(years, raw_dir, legacy_dir)

    # Step 4: Process each year
    print("\n--- Step 2: Process each year ---")
    frames = process_all_years(years, raw_dir, CFG)

    if not frames:
        print("ERROR: no data processed. Exiting.")
        return

    # Step 5: Combine & save
    print("\n--- Step 3: Combine & save ---")
    out_path = combine_and_save(frames, output_dir, CFG)

    print(f"\nDone. Total elapsed: {time.time() - t_start:.0f}s")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
