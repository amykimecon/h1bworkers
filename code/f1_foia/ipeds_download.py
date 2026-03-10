# SCRIPT TO DOWNLOAD AND MERGE IPEDS DATA ACROSS YEARS
# Components: Fall Enrollment (EF) and Institutional Costs (IC_AY)
# Also downloads Stata programs to extract variable/value labels for harmonization.
# Years: 2000-2024
import os
import sys
import re
import json
import zipfile
import requests
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *

IPEDS_BASE_URL = "https://nces.ed.gov/ipeds/datacenter/data"
RAW_PATH = f"{root}/data/raw/ipeds"
STATA_PATH = os.path.join(RAW_PATH, "stata_programs")
os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(STATA_PATH, exist_ok=True)

YEARS = range(2000, 2025)

# Each component maps to its zip URL stem and the filename inside the zip.
# IPEDS uses lowercase filenames inside the zip for most years; we try both.
COMPONENTS = {
    "enrollment_fall": {
        "zip_stem": "EF{year}A",    # EF2023A.zip
        "csv_stem": "ef{year}a",    # ef2023a.csv inside zip
    },
    "enrollment_12mo": {
        "zip_stem": "EFFY{year}",   # EFFY2023.zip
        "csv_stem": "effy{year}",   # effy2023.csv inside zip
    },
    "cost": {
        "zip_stem": "IC{year}_AY",  # IC2023_AY.zip
        "csv_stem": "ic{year}_ay",  # ic2023_ay.csv inside zip
    },
}


# ---------------------------------------------------------------------------
# HELPERS: finding files inside zips
# ---------------------------------------------------------------------------

def _find_file_in_zip(zf: zipfile.ZipFile, preferred_stem: str, ext: str) -> str | None:
    """Return the name of a file inside a zip by trying lower/uppercase stems."""
    names = zf.namelist()
    candidates = [
        preferred_stem.lower() + ext,
        preferred_stem.upper() + ext,
        preferred_stem.lower() + "_rv" + ext,  # revised release
        preferred_stem.upper() + "_rv" + ext,
    ]
    for c in candidates:
        if c in names:
            return c
    # Fallback: any file with matching extension that isn't a dict/label file
    matches = [
        n for n in names
        if n.lower().endswith(ext)
        and "_dict" not in n.lower()
        and "_label" not in n.lower()
    ]
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# STATA PROGRAM: download + parse
# ---------------------------------------------------------------------------

def _download_zip(url: str, local_path: str, label: str, force: bool) -> bool:
    """Download a zip to local_path. Returns True on success."""
    if not force and os.path.exists(local_path):
        print(f"  [{label}] cached")
        return True
    print(f"  [{label}] downloading ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"SKIP ({e})")
        return False
    except requests.RequestException as e:
        print(f"ERROR ({e})")
        return False
    with open(local_path, "wb") as f:
        f.write(resp.content)
    print("done")
    return True


def download_stata_program(component: str, year: int, force: bool = False) -> str | None:
    """
    Download IPEDS Stata program zip for a component/year.
    Returns the text content of the .do file, or None on failure.
    """
    cfg = COMPONENTS[component]
    zip_stem = cfg["zip_stem"].format(year=year)
    stata_zip = f"{zip_stem}_Stata.zip"
    local_zip = os.path.join(STATA_PATH, stata_zip)

    ok = _download_zip(f"{IPEDS_BASE_URL}/{stata_zip}", local_zip, f"Stata/{year}", force)
    if not ok:
        return None

    try:
        with zipfile.ZipFile(local_zip) as zf:
            do_name = _find_file_in_zip(zf, cfg["csv_stem"].format(year=year), ".do")
            if do_name is None:
                # Also try matching the zip stem
                do_name = _find_file_in_zip(zf, zip_stem.lower(), ".do")
            if do_name is None:
                do_files = [n for n in zf.namelist() if n.lower().endswith(".do")]
                do_name = do_files[0] if do_files else None
            if do_name is None:
                print(f"  WARNING: no .do file in {stata_zip}")
                return None
            return zf.read(do_name).decode("latin-1")
    except zipfile.BadZipFile:
        print(f"  WARNING: bad zip {local_zip}")
        return None


def parse_stata_labels(do_text: str) -> dict:
    """
    Parse a Stata .do file and return:
      var_labels:          {varname -> "Variable label"}
      val_label_defs:      {labelname -> {int_code -> "string label"}}
      val_label_assigns:   {varname -> labelname}
    """
    # Normalize Stata line-continuation (///)
    text = re.sub(r"\s*///\s*\n\s*", " ", do_text)

    # --- variable labels ---
    var_labels: dict[str, str] = {}
    for m in re.finditer(r'label\s+variable\s+(\w+)\s+"([^"]*)"', text, re.IGNORECASE):
        var_labels[m.group(1).lower()] = m.group(2)

    # --- value label definitions ---
    # Pattern: label define <name> [-]N "label" [-]N "label" ... [,add]
    # Stata uses ",add" to append entries to an existing label set across multiple lines;
    # accumulate into the same dict entry rather than overwriting on each match.
    val_label_defs: dict[str, dict[int, str]] = {}
    for m in re.finditer(
        r'label\s+define\s+(\w+)\s+((?:(?:-?\d+)\s+"[^"]*"\s*)+)',
        text,
        re.IGNORECASE,
    ):
        labelname = m.group(1).lower()
        pairs = val_label_defs.get(labelname, {})  # accumulate across ,add lines
        for pm in re.finditer(r'(-?\d+)\s+"([^"]*)"', m.group(2)):
            pairs[int(pm.group(1))] = pm.group(2)
        if pairs:
            val_label_defs[labelname] = pairs

    # --- value label assignments ---
    val_label_assigns: dict[str, str] = {}
    for m in re.finditer(r'label\s+values\s+(\w+)\s+(\w+)', text, re.IGNORECASE):
        val_label_assigns[m.group(1).lower()] = m.group(2).lower()

    return {
        "var_labels": var_labels,
        "val_label_defs": val_label_defs,
        "val_label_assigns": val_label_assigns,
    }


def apply_labels(df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """
    Given a DataFrame and parsed Stata labels, add `_label` columns for every
    numerically-coded variable that has a value-label assignment.
    Variable labels are stored as column .attrs metadata.
    """
    df = df.copy()
    var_labels = labels["var_labels"]
    val_label_defs = labels["val_label_defs"]
    val_label_assigns = labels["val_label_assigns"]

    for col in list(df.columns):
        if col in var_labels:
            df[col].attrs["label"] = var_labels[col]
        if col in val_label_assigns:
            labelname = val_label_assigns[col]
            if labelname in val_label_defs:
                df[f"{col}_label"] = df[col].map(val_label_defs[labelname])

    return df


# ---------------------------------------------------------------------------
# CROSS-YEAR HARMONIZATION CHECK
# ---------------------------------------------------------------------------

def check_label_consistency(per_year_labels: dict[int, dict]) -> pd.DataFrame:
    """
    Given {year: labels_dict}, identify variables whose value encodings
    changed across years. Returns a DataFrame summarizing inconsistencies.
    """
    # Collect all (varname, labelname) assignments per year
    records = []
    for year, labels in per_year_labels.items():
        for varname, labelname in labels["val_label_assigns"].items():
            mapping = labels["val_label_defs"].get(labelname, {})
            records.append({
                "year": year,
                "varname": varname,
                "labelname": labelname,
                "mapping_hash": hash(json.dumps(mapping, sort_keys=True)),
                "n_values": len(mapping),
            })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # Flag variables that have more than one unique mapping hash across years
    inconsistent = (
        df.groupby("varname")["mapping_hash"]
        .nunique()
        .reset_index()
        .rename(columns={"mapping_hash": "n_distinct_encodings"})
        .query("n_distinct_encodings > 1")
    )
    return inconsistent.sort_values("varname")


# ---------------------------------------------------------------------------
# MAIN DOWNLOAD FUNCTIONS
# ---------------------------------------------------------------------------

def download_year(
    component: str, year: int, force: bool = False
) -> tuple[pd.DataFrame | None, dict | None]:
    """
    Download one year of one IPEDS component + its Stata program.
    Returns (DataFrame, labels_dict). Either may be None on failure.
    """
    cfg = COMPONENTS[component]
    zip_stem = cfg["zip_stem"].format(year=year)
    csv_stem = cfg["csv_stem"].format(year=year)
    zip_filename = f"{zip_stem}.zip"
    local_zip_path = os.path.join(RAW_PATH, zip_filename)

    print(f"\n  -- {year} --")

    # Download data zip
    ok = _download_zip(f"{IPEDS_BASE_URL}/{zip_filename}", local_zip_path, "data", force)
    if not ok:
        return None, None

    # Read CSV from zip
    try:
        with zipfile.ZipFile(local_zip_path) as zf:
            csv_name = _find_file_in_zip(zf, csv_stem, ".csv")
            if csv_name is None:
                print(f"  WARNING: no data CSV in {zip_filename}")
                return None, None
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, encoding="latin-1", low_memory=False)
    except zipfile.BadZipFile:
        print(f"  WARNING: bad zip {local_zip_path}")
        return None, None

    # Lowercase + strip whitespace; some IPEDS CSVs have trailing spaces in headers
    # which create spurious duplicate columns on concat (e.g. "efnralw " vs "efnralw")
    df.columns = df.columns.str.lower().str.strip()
    # Strip UTF-8 BOM that shows up as 'ï»¿' when the file is decoded as latin-1
    df.columns = df.columns.str.replace(r'^ï»¿', '', regex=True)
    # Drop pandas auto-renamed duplicate columns (e.g. "xefgndru.1" when "xefgndru"
    # also exists) — these arise when the CSV itself contains two identically-named columns
    auto_renamed = [c for c in df.columns if re.search(r'\.\d+$', c) and c.rsplit('.', 1)[0] in df.columns]
    if auto_renamed:
        df = df.drop(columns=auto_renamed)
    # Drop any remaining true duplicates (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]
    df["year"] = year
    print(f"  [data] {len(df):,} rows from {csv_name}")

    # Download and parse Stata program
    do_text = download_stata_program(component, year, force=force)
    labels = None
    if do_text:
        labels = parse_stata_labels(do_text)
        n_vars = len(labels["var_labels"])
        n_coded = len(labels["val_label_assigns"])
        print(f"  [Stata] {n_vars} var labels, {n_coded} coded vars")
        df = apply_labels(df, labels)

    return df, labels


def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    After concat, object columns may contain mixed int/str values
    (e.g. numeric in most years, suppression codes like 'R' in others).
    PyArrow rejects these. Strategy: try numeric; if any non-null strings
    remain, cast the whole column to string (keeping NaN as NA).
    """
    for col in df.select_dtypes(include="object").columns:
        # Treat Stata's '.' missing-value convention as NaN before numeric conversion
        cleaned = df[col].replace(".", np.nan)
        converted = pd.to_numeric(cleaned, errors="coerce")
        # All non-null values survived numeric conversion -> use numeric
        if converted.notna().sum() >= cleaned.notna().sum():
            df[col] = converted
        else:
            # Heterogeneous: cast to string, preserve actual nulls
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    return df


def download_and_merge(
    component: str, years=YEARS, force: bool = False
) -> pd.DataFrame:
    """
    Download all years for a component, apply labels, concatenate into a panel.
    Runs a cross-year encoding consistency check and saves a metadata JSON.
    """
    print(f"\n{'='*60}")
    print(f"  Component: {component.upper()}")
    print(f"{'='*60}")

    frames = []
    per_year_labels = {}

    for year in years:
        df, labels = download_year(component, year, force=force)
        if df is not None:
            frames.append(df)
        if labels is not None:
            per_year_labels[year] = labels

    if not frames:
        raise RuntimeError(f"No data downloaded for component '{component}'")

    panel = pd.concat(frames, ignore_index=True)
    panel = _normalize_dtypes(panel)

    # Drop _label columns that are entirely null across all years
    # (common for imputation-flag columns like xefrac* which have no value label assignments)
    null_label_cols = [c for c in panel.columns if c.endswith("_label") and panel[c].isna().all()]
    if null_label_cols:
        panel = panel.drop(columns=null_label_cols)
        print(f"\n  Dropped {len(null_label_cols)} all-null _label columns")

    print(f"\n  Panel: {len(panel):,} rows | {panel.shape[1]} cols | years {panel['year'].min()}-{panel['year'].max()}")

    # Cross-year consistency check
    if per_year_labels:
        inconsistent = check_label_consistency(per_year_labels)
        if len(inconsistent):
            print(f"\n  WARNING: {len(inconsistent)} variables changed encoding across years:")
            print(inconsistent.to_string(index=False))
        else:
            print("\n  All value encodings consistent across years.")

        # Save label metadata for inspection
        meta_path = os.path.join(RAW_PATH, f"ipeds_{component}_labels.json")
        # json.dumps can't serialize int keys -> convert
        meta = {
            str(yr): {
                "var_labels": lbl["var_labels"],
                "val_label_defs": {
                    k: {str(code): lab for code, lab in v.items()}
                    for k, v in lbl["val_label_defs"].items()
                },
                "val_label_assigns": lbl["val_label_assigns"],
            }
            for yr, lbl in per_year_labels.items()
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Label metadata -> {meta_path}")

    return panel


if __name__ == "__main__":
    for component in COMPONENTS:
        panel = download_and_merge(component, force = True)
        out_path = os.path.join(RAW_PATH, f"ipeds_{component}_panel.parquet")
        panel.to_parquet(out_path, index=False)
        print(f"\n  Saved panel -> {out_path}")
