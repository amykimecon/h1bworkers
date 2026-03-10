# File Description: Scrape historical SEVP-certified school lists and compute
# accreditation events (schools joining or leaving the SEVP list) across time.
#
# Strategy:
#   1. Use the Wayback Machine CDX API to enumerate all archived snapshots of
#      the "certified school list" PDF published by Study in the States.
#   2. Download each unique PDF (deduped by content digest).
#   3. Parse PDFs with pdfplumber to extract school-level rows.
#   4. Save one CSV per snapshot date.
#   5. Compare consecutive snapshots to produce a panel of accreditation events.
#
# Output (under {root}/data/raw/sevp/):
#   snapshots/          -- one CSV per snapshot date
#   events.parquet      -- accreditation events panel

import os
import re
import sys
import time
import requests
import pdfplumber
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *

###############################################################################
# PATHS & CONSTANTS
###############################################################################

RAW_PATH = Path(f"{root}/data/raw/sevp")
SNAPSHOT_PATH = RAW_PATH / "snapshots"
PDF_PATH = RAW_PATH / "pdfs"
RAW_PATH.mkdir(parents=True, exist_ok=True)
SNAPSHOT_PATH.mkdir(parents=True, exist_ok=True)
PDF_PATH.mkdir(parents=True, exist_ok=True)

# Certified school list is published at a date-stamped URL, e.g.:
# https://studyinthestates.dhs.gov/assets/certified-school-list-01-14-26.pdf
# DHS updates this periodically; the file lives under /assets/ as of early 2026.
SITS_BASE = "https://studyinthestates.dhs.gov"
SITS_ASSETS_BASE = "studyinthestates.dhs.gov/assets"

# Wayback Machine CDX API -- returns JSON arrays of capture metadata.
# We search for any URL matching the certified-school-list pattern across all time.
CDX_API = "https://web.archive.org/cdx/search/cdx"

# Polite crawl delay (seconds) between HTTP requests
CRAWL_DELAY = 2

# Extra one-off PDFs to incorporate as additional snapshots.
# Format: (url, snapshot_date_str "YYYY-MM-DD", format_tag)
# format_tag drives which parser is used.
EXTRA_PDFS = [
    (
        "https://www.immigration.com/sites/default/files/sevpapprovedSchools.pdf",
        "2010-01-11",   # "As of Monday, January 11, 2010"
        "immigration2010",
    ),
    (
        "https://www.kidambi.com/resources/SEVP%20ApprovedSchools.pdf",
        "2011-12-16",   # "As of Friday, December 16, 2011"
        "immigration2010",  # same column layout as the 2010 PDF
    ),
]


###############################################################################
# STEP 1 -- DISCOVER HISTORICAL SNAPSHOTS VIA WAYBACK CDX API
###############################################################################

def query_cdx(url_pattern: str, limit: int = 500) -> list[dict]:
    """
    Query the Wayback Machine CDX API for all captures of a URL pattern.
    Returns a list of dicts with keys: timestamp, original, statuscode, digest.
    """
    params = {
        "url": url_pattern,
        "output": "json",
        "fl": "timestamp,original,statuscode,digest",
        "collapse": "digest",   # deduplicate identical content
        "filter": "statuscode:200",
        "limit": limit,
    }
    print(f"  Querying CDX API for: {url_pattern}")
    resp = requests.get(CDX_API, params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
    if len(rows) <= 1:
        return []
    # First row is the header
    header, *data = rows
    return [dict(zip(header, row)) for row in data]


def discover_snapshots() -> pd.DataFrame:
    """
    Use the CDX API to find all unique archived versions of the SEVP
    certified school list PDF. Returns a DataFrame sorted by timestamp.

    DHS has used at least two URL roots over time:
      /assets/certified-school-list-*.pdf   (2024-present)
      /sites/default/files/certified-school-list-*.pdf  (older Drupal path)
    We query both, plus a broad wildcard to catch any renamings.
    """
    print("\n=== Discovering historical snapshots via Wayback Machine CDX API ===")

    patterns = [
        f"{SITS_ASSETS_BASE}/certified-school-list*",           # current path
        f"studyinthestates.dhs.gov/sites/default/files/certified-school-list*",  # legacy
        f"studyinthestates.dhs.gov/*certified*school*list*",    # broad fallback
    ]

    all_records = []
    for pat in patterns:
        records = query_cdx(pat)
        all_records.extend(records)
        time.sleep(CRAWL_DELAY)

    if not all_records:
        print("  WARNING: no snapshots found via CDX. "
              "Try adding more URL patterns or check for robots restrictions.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M%S")
    df = df.sort_values("timestamp").drop_duplicates(subset="digest")
    print(f"  Found {len(df)} unique snapshots ({df['timestamp'].min().date()} "
          f"– {df['timestamp'].max().date()})")
    return df


###############################################################################
# STEP 1b -- DOWNLOAD CURRENT PDF DIRECTLY (AS SEED IF CDX FINDS NOTHING)
###############################################################################

def download_current_pdf(force: bool = False) -> Path | None:
    """
    Download the current certified school list PDF directly from DHS.
    The page https://studyinthestates.dhs.gov/school-search links to a
    date-stamped PDF under /assets/. We scrape that link dynamically.
    """
    from html.parser import HTMLParser

    class LinkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.pdf_url = None
        def handle_starttag(self, tag, attrs):
            if tag == "a" and self.pdf_url is None:
                attrs_dict = dict(attrs)
                href = attrs_dict.get("href", "")
                if "certified-school-list" in href and href.endswith(".pdf"):
                    self.pdf_url = href if href.startswith("http") else f"{SITS_BASE}{href}"

    print("\n=== Fetching current certified school list from DHS ===")
    try:
        resp = requests.get(f"{SITS_BASE}/school-search", timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Could not fetch school-search page: {e}")
        return None

    parser = LinkParser()
    parser.feed(resp.text)

    if not parser.pdf_url:
        # Fallback: try a predictable URL based on today's date
        print("  Could not auto-detect PDF URL from page; trying known URL pattern.")
        today = datetime.today()
        filename = f"certified-school-list-{today.strftime('%m-%d-%y')}.pdf"
        parser.pdf_url = f"{SITS_BASE}/assets/{filename}"

    print(f"  PDF URL: {parser.pdf_url}")
    filename = parser.pdf_url.split("/")[-1]
    local = PDF_PATH / filename
    if not force and local.exists():
        print(f"  Cached: {local}")
        return local

    print(f"  Downloading ...", end=" ", flush=True)
    try:
        resp = requests.get(parser.pdf_url, timeout=120)
        resp.raise_for_status()
        with open(local, "wb") as f:
            f.write(resp.content)
        print("done")
        return local
    except requests.RequestException as e:
        print(f"FAILED ({e})")
        return None


###############################################################################
# STEP 2 -- DOWNLOAD PDFs
###############################################################################

def wayback_url(timestamp: str, original: str) -> str:
    """Construct a Wayback Machine URL for a given capture."""
    return f"https://web.archive.org/web/{timestamp}if_/{original}"


def download_pdf(timestamp: str, original_url: str, force: bool = False) -> Path | None:
    """
    Download one archived PDF from the Wayback Machine.
    Returns the local Path on success, None on failure.
    """
    ts_str = timestamp if isinstance(timestamp, str) else timestamp.strftime("%Y%m%d%H%M%S")
    local = PDF_PATH / f"{ts_str}.pdf"
    if not force and local.exists():
        print(f"  [{ts_str}] cached")
        return local

    url = wayback_url(ts_str, original_url)
    print(f"  [{ts_str}] downloading {original_url} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(local, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        print("done")
        return local
    except requests.RequestException as e:
        print(f"FAILED ({e})")
        return None


###############################################################################
# STEP 3 -- PARSE PDFs
###############################################################################

###############################################################################
# STEP 3a -- PARSER FOR 2010-FORMAT PDF (immigration.com snapshot)
###############################################################################

# The 2010 PDF is a fixed-width 4-column layout (no embedded table metadata):
#
#   Institution Name   Campus Name   City/State   Date Approved
#   x0 ≈ 22            x0 ≈ 229      x0 ≈ 419     x0 ≈ 536
#
# We use pdfplumber's word-coordinate extraction to assign each word to the
# correct column by x0 position, then reconstruct cell values row by row.

# Column x0 boundaries (left edge of each column), derived from header words.
_2010_COL_BOUNDS = [
    ("institution_name", 0,   228),
    ("campus_name",      229, 418),
    ("city_state",       419, 535),
    ("date_approved",    536, 9999),
]

_DATE_APPROVED_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")


def parse_2010_pdf(pdf_path: Path) -> pd.DataFrame | None:
    """
    Parse the immigration.com 2010-format SEVP PDF using word coordinates.

    Returns a DataFrame with columns:
      institution_name, campus_name, city, state, date_approved
    """
    # Collect words across all pages, grouped by (page_num, row_top)
    page_rows: dict[tuple, dict] = {}   # (page, row_top) -> {col: [word, ...]}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                words = page.extract_words()
                for w in words:
                    x0 = w["x0"]
                    top = round(w["top"])   # y-position; round to group words on same line
                    text = w["text"]

                    # Determine column
                    col = None
                    for col_name, lo, hi in _2010_COL_BOUNDS:
                        if lo <= x0 <= hi:
                            col = col_name
                            break
                    if col is None:
                        continue

                    key = (page_num, top)
                    if key not in page_rows:
                        page_rows[key] = {c: [] for c, _, _ in _2010_COL_BOUNDS}
                    page_rows[key][col].append(text)
    except Exception as e:
        print(f"  WARNING: could not parse {pdf_path.name}: {e}")
        return None

    rows = []
    for (page_num, top), cols in sorted(page_rows.items()):
        inst = " ".join(cols["institution_name"]).strip()
        campus = " ".join(cols["campus_name"]).strip()
        city_state = " ".join(cols["city_state"]).strip()
        date_str = " ".join(cols["date_approved"]).strip()

        # Skip header rows, section dividers, and lines without a valid date
        if not _DATE_APPROVED_RE.match(date_str):
            continue
        # Skip the column header row itself
        if "Institution" in inst or "Campus" in inst:
            continue

        # Split "City, ST" -> city, state
        city_state_parts = city_state.rsplit(",", 1)
        if len(city_state_parts) == 2:
            city = city_state_parts[0].strip()
            state = city_state_parts[1].strip()
        else:
            city = city_state.strip()
            state = ""

        rows.append({
            "institution_name": inst,
            "campus_name": campus,
            "city": city,
            "state": state,
            "date_approved": date_str,
        })

    if not rows:
        print(f"  WARNING: no data rows extracted from {pdf_path.name}")
        return None

    df = pd.DataFrame(rows)
    df["date_approved"] = pd.to_datetime(df["date_approved"], format="%m/%d/%Y", errors="coerce")
    print(f"  Parsed {len(df)} campus rows ({df['state'].nunique()} states, "
          f"date_approved {df['date_approved'].min().date()} – {df['date_approved'].max().date()})")
    return df


def normalize_2010_snapshot(df: pd.DataFrame, snapshot_date: datetime) -> pd.DataFrame:
    """
    Convert the 2010 DataFrame into the standard snapshot schema so it can
    be fed into compute_events() alongside the DHS PDFs.

    We use institution_name as school_name (campus-level granularity is
    preserved in a separate column but not used for the join/leave key).
    The date_approved field is retained for use as a more precise event date.
    """
    out = pd.DataFrame({
        "school_name": df["institution_name"],
        "campus_name": df["campus_name"],
        "city": df["city"],
        "state": df["state"],
        "date_approved": df["date_approved"].dt.date.astype(str),
        "snapshot_date": str(snapshot_date.date()),
    })
    out = out[out["school_name"].str.strip().ne("")]
    return out


###############################################################################
# STEP 3b -- DHS TEXT-BASED PARSER (coordinate-based fallback for DHS PDFs)
###############################################################################

# The DHS "SEVP Certified Schools" PDFs from ~2013-2024 use a fixed-width
# text layout with no embedded table borders.  Observed column positions:
#
#   SCHOOL NAME  CAMPUS NAME  F  M  CITY         ST  CAMPUS ID
#   x0≈20        x0≈255      406 419 x0≈432    511   x0≈535
#
# We auto-detect exact column breaks from the header row on the first page,
# so minor layout shifts across years are handled automatically.

_STATE_RE = re.compile(r"^[A-Z]{2}$")


def _detect_dhs_col_bounds(pdf_path: Path) -> list[tuple[str, int, int]] | None:
    """
    Find the x0 position of each column header on the first page, then
    build column boundary ranges.  Returns None if no header is found.
    """
    header_keywords = {
        "school": "school_name",
        "campus": "campus_name",
        "city": "city",
        "st": "state",
        "f": "f_visa",
        "m": "m_visa",
        "id": "campus_id",
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:3]:  # header is always on early pages
                words = page.extract_words()
                # Find the header row: a row where "SCHOOL" or "SCHOOL NAME" appears
                # Group words by top-coordinate
                rows: dict[int, list] = {}
                for w in words:
                    top = round(w["top"])
                    rows.setdefault(top, []).append(w)

                for top, row_words in sorted(rows.items()):
                    texts = [w["text"].lower() for w in row_words]
                    if "school" in texts or any("school" in t for t in texts):
                        # This is the header row — record x0 of each column header word
                        anchors: list[tuple[float, str]] = []
                        for w in row_words:
                            t = w["text"].lower()
                            # Map to canonical column name
                            col = None
                            for kw, name in header_keywords.items():
                                if t == kw or t.startswith(kw):
                                    col = name
                                    break
                            if col:
                                anchors.append((w["x0"], col))

                        if len(anchors) < 3:
                            continue  # not enough columns found, try next row

                        # Sort by x0; deduplicate by column name.
                        # Special case: "CAMPUS ID" produces two anchors —
                        # (x0_campus, "campus_name") and (x0_id, "campus_id").
                        # We want campus_id to START at x0_campus (the left edge of
                        # the two-word header), not x0_id. So when we encounter a
                        # duplicate "campus_name", check whether the next anchor is
                        # "campus_id"; if so, remap the duplicate to "campus_id" using
                        # the "CAMPUS" word's x0 and drop the separate "id" anchor.
                        anchors.sort()
                        seen_cols: set[str] = set()
                        deduped = []
                        skip_next = False
                        for i, (x0, col) in enumerate(anchors):
                            if skip_next:
                                skip_next = False
                                continue
                            if col in seen_cols:
                                # Duplicate — use it as campus_id start if followed by "campus_id"
                                nxt_col = anchors[i + 1][1] if i + 1 < len(anchors) else None
                                if nxt_col == "campus_id":
                                    deduped.append((x0, "campus_id"))
                                    skip_next = True
                                # else: discard the duplicate entirely
                            else:
                                deduped.append((x0, col))
                                seen_cols.add(col)
                        anchors = deduped

                        bounds = []
                        for i, (x0, col) in enumerate(anchors):
                            lo = int(x0) - 2
                            hi = int(anchors[i + 1][0]) - 3 if i + 1 < len(anchors) else 9999
                            bounds.append((col, lo, hi))
                        return bounds
    except Exception:
        pass
    return None


def _parse_dhs_text_pdf(pdf_path: Path, col_bounds: list[tuple]) -> pd.DataFrame | None:
    """
    Coordinate-based parser for DHS text-layout PDFs.
    Works identically to parse_2010_pdf but uses auto-detected column bounds.
    """
    col_names = [c for c, _, _ in col_bounds]
    page_rows: dict[tuple, dict] = {}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                for w in page.extract_words():
                    x0 = w["x0"]
                    top = round(w["top"])
                    col = None
                    for col_name, lo, hi in col_bounds:
                        if lo <= x0 <= hi:
                            col = col_name
                            break
                    if col is None:
                        continue
                    key = (page_num, top)
                    if key not in page_rows:
                        page_rows[key] = {c: [] for c in col_names}
                    page_rows[key][col].append(w["text"])
    except Exception as e:
        print(f"  WARNING: coordinate parse failed on {pdf_path.name}: {e}")
        return None

    rows = []
    for (_, _top), cols in sorted(page_rows.items()):
        school = " ".join(cols.get("school_name", [])).strip()
        campus = " ".join(cols.get("campus_name", [])).strip()
        city   = " ".join(cols.get("city", [])).strip()
        state  = " ".join(cols.get("state", [])).strip()
        f_visa = " ".join(cols.get("f_visa", [])).strip()
        m_visa = " ".join(cols.get("m_visa", [])).strip()
        cid    = " ".join(cols.get("campus_id", [])).strip()

        # Skip header rows and non-data rows (state must be a 2-letter code)
        if not _STATE_RE.match(state):
            continue
        if not school or school.upper() in ("SCHOOL NAME", "SCHOOL"):
            continue

        rows.append({
            "school_name": school,
            "campus_name": campus,
            "city": city,
            "state": state,
            "f_visa": f_visa,
            "m_visa": m_visa,
            "sevis_id": cid,
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


###############################################################################
# STEP 3c -- GENERIC TABLE-BASED PARSER + FALLBACK (DHS PDFs)
###############################################################################

def _clean_text(s) -> str:
    """Normalize whitespace in a cell value."""
    if s is None:
        return ""
    return " ".join(str(s).split())


def parse_school_pdf(pdf_path: Path) -> pd.DataFrame | None:
    """
    Extract school rows from one DHS certified-school-list PDF.

    Tries two strategies in order:
      1. pdfplumber extract_tables() -- works for newer PDFs with embedded borders.
      2. Coordinate-based word extraction -- works for older text-layout PDFs
         (the dominant format for DHS snapshots from ~2013 onward).

    Returns a raw DataFrame with lowercase column names, or None on failure.
    """
    # --- Strategy 1: table extraction ---
    rows = []
    header = None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    if not table:
                        continue
                    for row in table:
                        cells = [_clean_text(c) for c in row]
                        if not any(cells):
                            continue
                        if header is None:
                            if any("school" in c.lower() for c in cells):
                                header = cells
                            continue
                        if sum(bool(c) for c in cells) >= 2:
                            rows.append(cells)
    except Exception as e:
        print(f"  WARNING: table parse failed on {pdf_path.name}: {e}")

    if rows:
        n = len(header) if header else max(len(r) for r in rows)
        if header is None:
            header = [f"col_{i}" for i in range(n)]
        aligned = []
        for row in rows:
            if len(row) < n:
                row = row + [""] * (n - len(row))
            aligned.append(row[:n])

        df = pd.DataFrame(aligned, columns=header)
        df.columns = [c.lower().strip() for c in df.columns]
        return df

    # --- Strategy 2: coordinate-based fallback ---
    col_bounds = _detect_dhs_col_bounds(pdf_path)
    if col_bounds is None:
        print(f"  WARNING: could not detect column layout in {pdf_path.name}")
        return None

    df = _parse_dhs_text_pdf(pdf_path, col_bounds)
    if df is None:
        print(f"  WARNING: no data rows extracted from {pdf_path.name}")
    return df


def _identify_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first df column name that contains any candidate substring."""
    for col in df.columns:
        for cand in candidates:
            if cand in col:
                return col
    return None


def normalize_snapshot(df: pd.DataFrame, snapshot_date: datetime) -> pd.DataFrame:
    """
    Standardize column names across PDF vintages into a common schema:
      school_name, city, state, school_type, f_visa, m_visa, sevis_id (if present)
    """
    name_col = _identify_column(df, ["school name", "name"])
    city_col = _identify_column(df, ["city"])
    state_col = _identify_column(df, ["state"])
    type_col = _identify_column(df, ["school type", "education type", "type"])
    f_col = _identify_column(df, ["f visa", "f-1", "f1"])
    m_col = _identify_column(df, ["m visa", "m-1", "m1"])
    sevis_col = _identify_column(df, ["sevis", "school code", "campus id"])

    mapping = {
        name_col: "school_name",
        city_col: "city",
        state_col: "state",
        type_col: "school_type",
        f_col: "f_visa",
        m_col: "m_visa",
        sevis_col: "sevis_id",
    }
    rename = {k: v for k, v in mapping.items() if k is not None}
    df = df.rename(columns=rename)

    # Keep only standardized columns (drop whatever we couldn't map)
    keep = [c for c in ["school_name", "city", "state", "school_type",
                         "f_visa", "m_visa", "sevis_id"] if c in df.columns]
    df = df[keep].copy()

    # Drop rows with no school name
    if "school_name" in df.columns:
        df = df[df["school_name"].str.strip().ne("")]

    df["snapshot_date"] = snapshot_date.date()
    return df


###############################################################################
# STEP 4 -- PROCESS ALL SNAPSHOTS
###############################################################################

def process_snapshots(
    snapshots: pd.DataFrame, force: bool = False, reparse: bool = False
) -> list[pd.DataFrame]:
    """
    Download PDFs and parse them. Returns a list of normalized DataFrames,
    one per successfully processed snapshot.

    force:   re-download and re-parse everything
    reparse: re-parse already-downloaded PDFs without re-downloading
    """
    frames = []
    for _, row in snapshots.iterrows():
        ts = row["timestamp"]
        ts_str = ts.strftime("%Y%m%d%H%M%S")

        # Check if we already have a parsed snapshot CSV
        csv_out = SNAPSHOT_PATH / f"{ts_str}.csv"
        if not force and not reparse and csv_out.exists():
            print(f"  [{ts_str}] loading cached CSV")
            df = pd.read_csv(csv_out, dtype=str)
            frames.append(df)
            continue

        # Download PDF (skip re-download if only reparsing)
        pdf_path = download_pdf(ts_str, row["original"], force=force)
        if not reparse:
            time.sleep(CRAWL_DELAY)
        if pdf_path is None:
            continue

        # Parse PDF
        print(f"  [{ts_str}] parsing PDF ...", end=" ", flush=True)
        df_raw = parse_school_pdf(pdf_path)
        if df_raw is None:
            continue
        print(f"{len(df_raw)} rows")

        # Normalize
        df = normalize_snapshot(df_raw, ts)
        df.to_csv(csv_out, index=False)
        frames.append(df)

    return frames


###############################################################################
# STEP 4b -- DOWNLOAD AND PROCESS EXTRA ONE-OFF PDFs
###############################################################################

PARSERS = {
    "immigration2010": (parse_2010_pdf, normalize_2010_snapshot),
    "dhs": (parse_school_pdf, normalize_snapshot),
}


def download_and_process_extra_pdfs(force: bool = False, reparse: bool = False) -> list[pd.DataFrame]:
    """
    Download and parse PDFs listed in EXTRA_PDFS.
    Returns a list of normalized DataFrames (same schema as DHS snapshots).
    """
    frames = []
    for url, date_str, fmt_tag in EXTRA_PDFS:
        snap_date = datetime.strptime(date_str, "%Y-%m-%d")
        slug = date_str.replace("-", "")
        local_pdf = PDF_PATH / f"extra_{slug}_{fmt_tag}.pdf"
        csv_out = SNAPSHOT_PATH / f"extra_{slug}.csv"

        if not force and not reparse and csv_out.exists():
            print(f"  [extra {date_str}] loading cached CSV")
            frames.append(pd.read_csv(csv_out, dtype=str))
            continue

        # Download
        if not force and local_pdf.exists():
            print(f"  [extra {date_str}] PDF cached")
        else:
            print(f"  [extra {date_str}] downloading {url} ...", end=" ", flush=True)
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                with open(local_pdf, "wb") as f:
                    f.write(resp.content)
                print("done")
            except requests.RequestException as e:
                print(f"FAILED ({e})")
                continue

        # Parse
        parse_fn, normalize_fn = PARSERS.get(fmt_tag, PARSERS["dhs"])
        print(f"  [extra {date_str}] parsing ({fmt_tag}) ...", end=" ", flush=True)
        df_raw = parse_fn(local_pdf)
        if df_raw is None:
            continue

        df = normalize_fn(df_raw, snap_date)
        df.to_csv(csv_out, index=False)
        frames.append(df)

    return frames


###############################################################################
# STEP 5 -- COMPUTE ACCREDITATION EVENTS
###############################################################################

def _school_key(df: pd.DataFrame) -> pd.Series:
    """
    Build a canonical school identifier. Prefer SEVIS ID when available;
    fall back to normalized (school_name, state) pair.
    """
    if "sevis_id" in df.columns and df["sevis_id"].notna().any():
        key = df["sevis_id"].str.strip().str.upper()
    else:
        name = df.get("school_name", pd.Series([""] * len(df))).str.strip().str.upper()
        state = df.get("state", pd.Series([""] * len(df))).str.strip().str.upper()
        key = name + "|" + state
    return key


MIN_SCHOOLS_PER_SNAPSHOT = 10_000  # snapshots with fewer rows are likely partial loads


def _deduplicate_frames(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    1. Drop snapshots with too few schools (partial page loads from Wayback Machine).
    2. When multiple snapshots share the same date, keep only the one with the
       most schools (maximizes signal, avoids churn between same-day archives).
    """
    # Filter clearly partial snapshots
    frames = [df for df in frames if len(df) >= MIN_SCHOOLS_PER_SNAPSHOT]

    # Group by date, keep largest
    by_date: dict[str, pd.DataFrame] = {}
    for df in frames:
        date = str(df["snapshot_date"].iloc[0])
        if date not in by_date or len(df) > len(by_date[date]):
            by_date[date] = df

    kept = sorted(by_date.values(), key=lambda d: d["snapshot_date"].iloc[0])
    dropped = len(frames) - len(kept)
    if dropped:
        print(f"  [dedup] kept {len(kept)} unique-date snapshots "
              f"(dropped {dropped} same-day duplicates / partial loads)")
    return kept


def compute_events(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Given a list of snapshot DataFrames, compute accreditation events:
      event_type = 'join'  -- school appears for the first time (or re-appears)
      event_type = 'leave' -- school disappears from the next snapshot

    Returns a DataFrame with columns:
      school_key, school_name, city, state, school_type,
      event_type, event_date, prev_date
    """
    if not frames:
        return pd.DataFrame()

    # Drop partial snapshots and deduplicate to one per date
    frames = _deduplicate_frames(frames)
    if not frames:
        return pd.DataFrame()

    # Sort by snapshot_date
    frames = sorted(frames, key=lambda d: d["snapshot_date"].iloc[0])

    events = []

    prev_keys = set()
    prev_date = None
    prev_meta: dict[str, dict] = {}   # key -> metadata row

    for df in frames:
        snap_date = df["snapshot_date"].iloc[0]
        df = df.copy()
        df["_key"] = _school_key(df)
        df = df.drop_duplicates(subset="_key")
        cur_keys = set(df["_key"])

        # Build metadata lookup for current snapshot
        meta_cols = [c for c in ["school_name", "city", "state", "school_type"] if c in df.columns]
        cur_meta = df.set_index("_key")[meta_cols].to_dict("index")

        # Build a date_approved lookup for this snapshot if available
        # (populated from the 2010 immigration.com PDF which has per-school cert dates)
        date_approved_lookup: dict[str, str] = {}
        if "date_approved" in df.columns:
            date_approved_lookup = df.set_index("_key")["date_approved"].to_dict()

        if prev_keys:
            # Schools present now but not before -> joined
            for key in cur_keys - prev_keys:
                meta = cur_meta.get(key, {})
                # Use per-school date_approved when available (more precise than snap_date)
                join_date = date_approved_lookup.get(key) or snap_date
                events.append({
                    "school_key": key,
                    **meta,
                    "event_type": "join",
                    "event_date": join_date,
                    "snapshot_date": snap_date,
                    "prev_date": prev_date,
                })
            # Schools present before but not now -> left
            for key in prev_keys - cur_keys:
                meta = prev_meta.get(key, {})
                events.append({
                    "school_key": key,
                    **meta,
                    "event_type": "leave",
                    "event_date": snap_date,
                    "snapshot_date": snap_date,
                    "prev_date": prev_date,
                })
        else:
            # First snapshot: everyone is a "join"
            for key in cur_keys:
                meta = cur_meta.get(key, {})
                join_date = date_approved_lookup.get(key) or snap_date
                events.append({
                    "school_key": key,
                    **meta,
                    "event_type": "join",
                    "event_date": join_date,
                    "snapshot_date": snap_date,
                    "prev_date": None,
                })

        prev_keys = cur_keys
        prev_date = snap_date
        prev_meta = cur_meta

    events_df = pd.DataFrame(events)
    events_df["event_date"] = pd.to_datetime(events_df["event_date"], errors="coerce")
    # event_year uses the precise event_date (e.g. date_approved for 2010 schools)
    events_df["event_year"] = events_df["event_date"].dt.year
    events_df = events_df.sort_values(["event_date", "event_type", "school_key"])
    return events_df


###############################################################################
# STEP 6 -- SUMMARY STATISTICS
###############################################################################

def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a year-level summary: number of schools joining and leaving per year.
    """
    summary = (
        events.groupby(["event_year", "event_type"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"join": "n_join", "leave": "n_leave"})
        .reset_index()
    )
    summary["net"] = summary.get("n_join", 0) - summary.get("n_leave", 0)
    return summary


###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape SEVP school lists and build accreditation events panel.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-parse even if cached.")
    parser.add_argument("--reparse", action="store_true", help="Re-parse cached PDFs without re-downloading.")
    parser.add_argument("--limit", type=int, default=500, help="Max CDX snapshots to retrieve (default 500).")
    args = parser.parse_args()

    # --- 1. Download current list directly (always a useful seed) ---
    current_pdf = download_current_pdf(force=args.force)
    if current_pdf:
        ts_now = datetime.today().strftime("%Y%m%d%H%M%S")
        csv_out = SNAPSHOT_PATH / f"{ts_now}.csv"
        if args.force or not csv_out.exists():
            print(f"  Parsing current PDF ...", end=" ", flush=True)
            df_raw = parse_school_pdf(current_pdf)
            if df_raw is not None:
                print(f"{len(df_raw)} rows")
                df = normalize_snapshot(df_raw, datetime.today())
                df.to_csv(csv_out, index=False)

    # --- 2. Discover historical snapshots via Wayback Machine ---
    snapshots = discover_snapshots()

    frames = []

    if not snapshots.empty:
        # Save snapshot index
        idx_path = RAW_PATH / "snapshot_index.csv"
        snapshots.to_csv(idx_path, index=False)
        print(f"  Snapshot index -> {idx_path}")

        # --- 3. Download and parse ---
        print(f"\n=== Downloading and parsing {len(snapshots)} PDFs from Wayback Machine ===")
        frames = process_snapshots(snapshots, force=args.force, reparse=args.reparse)
        print(f"\n  Successfully parsed {len(frames)} snapshots")
    else:
        print("\nNo Wayback Machine snapshots found; proceeding with current snapshot only.")

    # --- 3. Process extra one-off PDFs (e.g. 2010 immigration.com list) ---
    print("\n=== Processing extra historical PDFs ===")
    download_and_process_extra_pdfs(force=args.force, reparse=args.reparse)

    # Load all cached snapshot CSVs (includes current + historical + extras)
    all_csvs = sorted(SNAPSHOT_PATH.glob("*.csv"))
    frames = [pd.read_csv(f, dtype=str) for f in all_csvs]
    print(f"\n  Total snapshots available: {len(frames)}")

    if not frames:
        print("No parseable snapshots. Exiting.")
        sys.exit(1)

    # --- 4. Compute events ---
    print("\n=== Computing accreditation events ===")
    events = compute_events(frames)
    print(f"  Total events: {len(events):,} "
          f"({events['event_type'].value_counts().to_dict()})")

    # Save
    events_path = RAW_PATH / "events.parquet"
    events.to_parquet(events_path, index=False)
    print(f"  Events -> {events_path}")

    events_csv = RAW_PATH / "events.csv"
    events.to_csv(events_csv, index=False)
    print(f"  Events -> {events_csv}")

    # --- 5. Summary ---
    summary = summarize_events(events)
    print("\n=== Year-level summary ===")
    print(summary.to_string(index=False))

    summary_path = RAW_PATH / "events_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n  Summary -> {summary_path}")
