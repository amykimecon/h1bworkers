# File Description: Ex-post match quality evaluation for H-1B × Revelio individual merge
# Author: Amy Kim
# Date Created: Mar 2026
#
# Compares observable FOIA applicant characteristics — institution (ADE applicants),
# field of study, education level, and job title — to linked Revelio profile data,
# producing weighted match rates as a function of merge confidence (weight_norm).
#
# Key insight: the merged parquet already carries field_clean, job_title,
# foia_highest_ed_level, and ade_ind from the FOIA side, so TRK loading is only
# needed for institution_txt (available only in TRK_12704 raw xlsx files).
#
# Entry points:
#   run_match_quality_check(variant_name, run_tag, output_dir, parquet_path)
#     — called from reg_new.py per variant
#   Standalone: run the file directly in an iPython session

import os
import sys
import time
import glob
import re
import warnings
from pathlib import Path

import itertools
import duckdb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# ---------------------------------------------------------------------------
# Path setup — resolve code root and load helpers/icfg
# ---------------------------------------------------------------------------
if "__file__" in globals():
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    _THIS_DIR = os.path.join(os.getcwd(), "04_analysis")

_CODE_DIR = os.path.dirname(_THIS_DIR)
sys.path.append(_CODE_DIR)
sys.path.append(os.path.join(_CODE_DIR, "03_indiv_merge"))

import indiv_merge_config as icfg
from config import root
from helpers import fuzzy_join_lev_jw, _norm, field_clean_regex_sql, inst_clean_regex_sql

# ---------------------------------------------------------------------------
# Load match quality config
# ---------------------------------------------------------------------------
_MQ_CONFIG_PATH = Path(_CODE_DIR) / "configs" / "match_quality.yaml"


def _load_mq_config() -> dict:
    raw = yaml.safe_load(_MQ_CONFIG_PATH.read_text()) or {}

    def _expand(v):
        if isinstance(v, str):
            return v.replace("{root}", str(root))
        if isinstance(v, dict):
            return {k: _expand(vv) for k, vv in v.items()}
        return v

    return {k: _expand(v) for k, v in raw.items()}


MQ_CFG = _load_mq_config()
TESTING_CFG = MQ_CFG.get("testing", {})
TESTING_ENABLED = bool(TESTING_CFG.get("enabled", False))
TESTING_N_APPS = int(TESTING_CFG.get("n_apps", 200))
TESTING_SEED = int(TESTING_CFG.get("seed", 42))
THRESHOLDS = MQ_CFG.get("thresholds", {})
INST_THRESHOLD = float(THRESHOLDS.get("institution", 0.85))
FIELD_STR_THRESHOLD = float(THRESHOLDS.get("field_str", 0.80))
TITLE_THRESHOLD = float(THRESHOLDS.get("title", 0.75))
TITLE_TOP_N = int(MQ_CFG.get("title_top_n", 5))
FORCE_REBUILD = bool(MQ_CFG.get("force_rebuild_supplement", False))

print(f"=== match_quality_check.py loaded ===")
print(f"  testing: {'ENABLED (n=' + str(TESTING_N_APPS) + ')' if TESTING_ENABLED else 'disabled'}")
print(f"  thresholds: inst={INST_THRESHOLD}, field={FIELD_STR_THRESHOLD}, title={TITLE_THRESHOLD}")
print(f"  title_top_n: {TITLE_TOP_N}")


###############################################################################
# EDUCATION LEVEL MAPPING
###############################################################################

def _ed_category(s) -> str | None:
    """Map raw education level string → hs / bachelor / master / doctor / None."""
    if pd.isna(s) or str(s).strip() == "":
        return None
    s_low = str(s).lower().strip()
    if any(x in s_low for x in ("doctor", "ph.d", "phd", "d.phil", "j.d", "juris", "md ", "m.d.")):
        return "doctor"
    if any(x in s_low for x in ("master", "mba", "m.s", "m.b", "m.a.", " ms ", "meng")):
        return "master"
    if any(x in s_low for x in ("bachelor", "b.s", "b.a.", " bs ", " ba ", "btech", "b.tech", "undergraduate")):
        return "bachelor"
    if any(x in s_low for x in ("high school", "associate", "diploma", "hs ", "none", "some college")):
        return "hs"
    return None

# Ordered rank for ed categories: higher = more advanced degree
_ED_RANK = {"hs": 1, "bachelor": 2, "master": 3, "doctor": 4}


###############################################################################
# FIELD OF STUDY BROAD CATEGORY MAPPING
###############################################################################

# Checked in order — first match wins.
# STEM is split into four sub-categories for finer-grained comparison:
#   cs, engineering, math_stats, natural_science
# Full taxonomy: law, cs, engineering, math_stats, natural_science,
#                business, social_sciences, health, humanities, education, other
_FIELD_CATS = [
    # --- Law (before business/social_sciences to avoid false positives) ---
    ("law", re.compile(
        r"law|legal|juris|attorney|paralegal|regulatory|compliance|litigation|"
        r"intellectual property|patent|tax law|corporate law|criminal law|"
        r"international law|constitutional|civil rights", re.I)),

    # --- CS / IT / Data (before engineering to catch "computer engineering",
    #     "software engineering", "data engineering" as CS-adjacent) ---
    ("cs", re.compile(
        r"computer science|software engineering|software development|"
        r"information technology|information systems|computer engineering|"
        r"electrical and computer|computing|programming|"
        r"machine learning|artificial intelligence|deep learning|neural network|"
        r"data science|data engineering|data analytics|"
        r"cybersecurity|cyber security|network security|information security|"
        r"cloud computing|devops|web development|database|"
        r"human.computer interaction|bioinformatics|"
        r"computer|software|informatics", re.I)),

    # --- Engineering (catches anything with "engineer" + arch/construction/materials) ---
    # "biomedical engineering", "electrical engineering", "environmental engineering" etc.
    # all fall here because "engineer" appears before natural_science checks "biology"/"environment".
    ("engineering", re.compile(
        r"engineer|architecture|architectural|construction management|"
        r"robotics|automation|semiconductor|telecommunications|telecom|"
        r"nanotechnology|photonics|surveying|urban design|structural|"
        r"petroleum|nuclear energy|materials science|materials engineering|"
        r"manufacturing|mechatronics|systems design", re.I)),

    # --- Math & Statistics (before natural_science to catch "mathematical science") ---
    ("math_stats", re.compile(
        r"mathematics|\bmath\b|statistics|\bstatistical\b|actuarial|"
        r"quantitative|operations research|computational mathematics|"
        r"computational applied|applied mathematics|applied math|"
        r"econometrics|mathematical", re.I)),

    # --- Natural Sciences ---
    ("natural_science", re.compile(
        r"biology|physics|chemistry|biochemistry|biophysics|"
        r"environmental science|ecology|geology|geophysics|geoscience|"
        r"oceanography|meteorology|astronomy|astrophysics|"
        r"genomics|genetics|molecular biology|cell biology|microbiology|"
        r"neuroscience|agricultural|food science|earth science|atmospheric|"
        r"materials|zoology|botany|entomology|evolutionary|"
        r"natural science|physical science|science", re.I)),

    # --- Business ---
    ("business", re.compile(
        r"business|management|accounting|finance|marketing|commerce|"
        r"administration|mba|supply chain|operations|entrepreneurship|strategy|"
        r"project management|organizational|human resources|\bhr\b|real estate|"
        r"logistics|hospitality|retail|banking|investment|audit|taxation|\btax\b|"
        r"international business|e-commerce|ecommerce|consulting|"
        r"financial|managerial|corporate|business analytics|"
        r"business administration|bus admin|bus analytics|bus mgmt|"
        r"business information|healthcare management|public administration", re.I)),

    # --- Social Sciences ---
    ("social_sciences", re.compile(
        r"economics|psychology|sociology|political|public policy|international relations|"
        r"anthropology|geography|criminology|social work|urban planning|urban studies|"
        r"development studies|gender studies|cultural studies|ethnic studies|religious studies|"
        r"diplomacy|global affairs|international development|area studies|"
        r"demography|population|public affairs|government|social science|"
        r"behavioral science|cognitive science", re.I)),

    # --- Health ---
    ("health", re.compile(
        r"medicine|medical|nursing|pharmacy|health|clinical|dental|physician|"
        r"surgery|epidemiology|nutrition|physical therapy|"
        r"occupational therapy|radiology|pathology|veterinary|optometry|"
        r"kinesiology|dietetics|speech language|audiology|exercise science|"
        r"sports medicine|mental health|counseling|physical education|"
        r"healthcare|rehabilitation|respiratory|anesthesia|midwifery|"
        r"pharmaceutical|pre-med|premed|pre med|"
        r"health informatics|genetic counseling|"
        r"community health|global health|health policy", re.I)),

    # --- Humanities ---
    ("humanities", re.compile(
        r"english|history|philosophy|literature|arts|linguistics|communication|"
        r"journalism|music|religion|fine arts|theater|film|media studies|"
        r"creative writing|graphic design|fashion|performing arts|"
        r"foreign language|spanish|french|chinese|arabic|german|japanese|"
        r"korean|portuguese|translation|interpretation|rhetoric|cultural|"
        r"classics|medieval|american studies|writing|visual arts|studio art|"
        r"photography|illustration|animation|interior design|game design|"
        r"digital media|broadcasting|public relations|advertising|"
        r"library|museum|art history", re.I)),

    # --- Education ---
    ("education", re.compile(
        r"education|teaching|curriculum|pedagogy|instructional|"
        r"early childhood|special education|educational|higher education|"
        r"student affairs|school counseling|school psychology|"
        r"training and development|adult education", re.I)),
]


def _field_category(s) -> str | None:
    """Map field string → law / cs / engineering / math_stats / natural_science /
    business / social_sciences / health / humanities / education / other.
    Returns None for null/empty/trivial inputs."""
    if pd.isna(s) or str(s).strip() in ("", "na", "n/a", "none", "general", "general studies"):
        return None
    s_str = str(s)
    for cat, pat in _FIELD_CATS:
        if pat.search(s_str):
            return cat
    return "other"


# ---------------------------------------------------------------------------
# Direct mapping for Revelio's 17 pre-categorized `field` values.
# Must use the same category names as _FIELD_CATS above.
# ---------------------------------------------------------------------------
_REV_FIELD_CAT: dict[str, str] = {
    "Business":               "business",
    "Engineering":            "engineering",
    "Education":              "education",
    "Accounting":             "business",
    "Economics":              "social_sciences",
    "Finance":                "business",
    "Marketing":              "business",
    "Nursing":                "health",
    "Law":                    "law",
    "Information Technology": "cs",
    "Biology":                "natural_science",
    "Chemistry":              "natural_science",
    "Mathematics":            "math_stats",
    "Architecture":           "engineering",
    "Medicine":               "health",
    "Physics":                "natural_science",
    "Statistics":             "math_stats",
}


###############################################################################
# TRK INSTITUTION SUPPLEMENT
###############################################################################

def _load_trk_raw(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load TRK_12704 xlsx files, keep only rows with INSTITUTION_TXT,
    normalize all column names to UPPERCASE, and cache as parquet.

    No firm crosswalk, no FOIA join, no institution cleaning — just raw TRK
    data filtered to rows that have an institution. The exact-match join to
    foia_indiv happens in run_match_quality_check() using 15 raw FOIA columns.

    Returns DataFrame with all available TRK columns (uppercased).
    """
    cache_path = MQ_CFG.get(
        "trk_raw_cache",
        str(Path(root) / "data" / "int" / "trk_raw.parquet")
    )

    if not force_rebuild and os.path.exists(cache_path):
        print(f"  [TRK raw] Loading cached ({cache_path})")
        trk = pd.read_parquet(cache_path)
        print(f"  [TRK raw] {len(trk):,} rows")
        return trk

    print(f"  [TRK raw] Building from raw xlsx files (this may take a few minutes)...")
    t0 = time.time()

    trk_dir = MQ_CFG.get("trk_data_dir", str(Path(root) / "data" / "raw" / "H-1B Data"))
    all_trk_files = sorted(glob.glob(os.path.join(trk_dir, "TRK_12704_FY*.xlsx")))
    print(f"  Found {len(all_trk_files)} TRK xlsx files in {trk_dir}")

    # --- Filter to lottery years present in foia_indiv (SELECTED only) ---
    foia_path_tmp = icfg.choose_path(icfg.FOIA_INDIV_PARQUET, icfg.FOIA_INDIV_PARQUET_LEGACY)
    _con_tmp = duckdb.connect()
    _relevant_years = set(_con_tmp.execute(f"""
        SELECT DISTINCT CAST(lottery_year AS VARCHAR) AS ly
        FROM read_parquet('{foia_path_tmp}')
        WHERE status_type = 'SELECTED'
    """).df()["ly"].tolist())
    _con_tmp.close()

    def _fy_from_fname(f):
        m = re.search(r'FY(\d{4})', os.path.basename(f))
        return m.group(1) if m else None

    trk_files = [f for f in all_trk_files if _fy_from_fname(f) in _relevant_years]
    print(f"  Relevant lottery years ({len(_relevant_years)}): {sorted(_relevant_years)}")
    print(f"  Loading {len(trk_files)}/{len(all_trk_files)} xlsx files matching relevant years")

    # --- Load all columns from each file, normalize column names to UPPERCASE ---
    frames = []
    for f in trk_files:
        try:
            df = pd.read_excel(f, engine="openpyxl")
            df.columns = [c.strip().upper() for c in df.columns]
            n_inst = df["INSTITUTION_TXT"].notna().sum() if "INSTITUTION_TXT" in df.columns else 0
            print(f"    {os.path.basename(f)}: {len(df):,} rows, {n_inst:,} with INSTITUTION_TXT")
            frames.append(df)
        except Exception as e:
            print(f"    [WARN] Could not read {os.path.basename(f)}: {e}")

    if not frames:
        print("  [ERROR] No TRK files loaded — returning empty DataFrame")
        return pd.DataFrame()

    trk = pd.concat(frames, ignore_index=True)
    print(f"  Total TRK rows loaded: {len(trk):,}")

    # Keep only rows with INSTITUTION_TXT
    trk = trk[
        trk["INSTITUTION_TXT"].notna() &
        (trk["INSTITUTION_TXT"].astype(str).str.strip() != "")
    ].copy()
    print(f"  Rows with INSTITUTION_TXT: {len(trk):,}")

    # Restrict to cap-subject ADE applications (S3Q1 = 'M')
    if "S3Q1" in trk.columns:
        trk["S3Q1"] = trk["S3Q1"].astype(str).str.strip().str.upper()
        n_before = len(trk)
        trk = trk[trk["S3Q1"] == "M"].copy()
        print(f"  Rows after S3Q1='M' (cap-subject) filter: {len(trk):,} "
              f"(dropped {n_before - len(trk):,})")
    else:
        print("  [WARN] S3Q1 column not found in TRK data — cap-subject filter skipped")

    # Cast mixed-type object columns to string to avoid PyArrow int64 inference errors
    # (e.g. FIRST_DECISION_FY has both integers and redacted strings like "(b)(3) (b)(6)")
    for _col in trk.select_dtypes(include="object").columns:
        trk[_col] = trk[_col].astype(str).where(trk[_col].notna(), other=None)

    trk.to_parquet(cache_path, index=False)
    print(f"  [TRK raw] {len(trk):,} rows saved to {cache_path} ({time.time()-t0:.1f}s)")
    return trk


###############################################################################
# WEIGHTED MEAN HELPER
###############################################################################

def _wtd_mean_by_app(df: pd.DataFrame, match_col: str,
                     id_col: str = "foia_indiv_id",
                     weight_col: str = "weight_norm") -> pd.Series:
    """
    Vectorized weighted mean of match_col grouped by id_col.
    Rows where match_col is NaN are excluded from both numerator and denominator.
    Returns a Series indexed by id_col.
    """
    valid = df[match_col].notna() & df[weight_col].notna()
    sub = df.loc[valid, [id_col, match_col, weight_col]].copy()
    if sub.empty:
        return pd.Series(dtype=float, name=match_col)
    sub["_num"] = sub[match_col] * sub[weight_col]
    num = sub.groupby(id_col)["_num"].sum()
    den = sub.groupby(id_col)[weight_col].sum()
    return (num / den).rename(match_col)


###############################################################################
# MAIN ENTRY POINT
###############################################################################

def run_match_quality_check(variant_name: str, run_tag: str,
                             output_dir: Path, parquet_path: str):
    """
    Compute ex-post match quality for one merged parquet variant.

    Args:
        variant_name:  e.g. "baseline", "mult6"
        run_tag:       e.g. "feb2026"
        output_dir:    Path to reg output dir (LaTeX tables go here/tables/)
        parquet_path:  Resolved path to the merged parquet for this variant
    """
    print(f"\n{'─'*60}")
    print(f"Match quality check: {variant_name}  run_tag={run_tag}")
    print(f"Parquet: {os.path.basename(parquet_path)}")
    print(f"{'─'*60}")
    t0 = time.time()

    output_dir = Path(output_dir)
    mq_dir = Path(MQ_CFG.get("output_dir", str(Path(root) / "output" / "match_quality")))
    mq_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load merged parquet
    # -----------------------------------------------------------------------
    con = duckdb.connect()
    print(f"\n  [1/7] Loading merged parquet...")
    merged = con.execute(f"""
        SELECT foia_indiv_id,
               user_id,
               weight_norm,
               foia_firm_uid,
               rcid,
               field_clean         AS field_clean_foia,
               job_title           AS job_title_foia,
               foia_highest_ed_level,
               COALESCE(ade_ind, 0) AS ade_ind,
               CAST(lottery_year AS VARCHAR) AS lottery_year,
               status_type
        FROM read_parquet('{parquet_path}')
        WHERE weight_norm IS NOT NULL
    """).df()
    print(f"  Merged rows: {len(merged):,} | "
          f"apps: {merged['foia_indiv_id'].nunique():,} | "
          f"users: {merged['user_id'].nunique():,}")

    # Testing mode: subsample applicants
    if TESTING_ENABLED:
        rng = np.random.default_rng(TESTING_SEED)
        app_ids = merged["foia_indiv_id"].unique()
        sample = rng.choice(app_ids, size=min(TESTING_N_APPS, len(app_ids)), replace=False)
        merged = merged[merged["foia_indiv_id"].isin(sample)].copy()
        print(f"  [TESTING] Subsampled to {merged['foia_indiv_id'].nunique():,} apps, "
              f"{len(merged):,} rows")

    user_ids = merged["user_id"].unique().tolist()
    foia_ids = merged["foia_indiv_id"].unique().tolist()
    print(f"  Unique user_ids in sample: {len(user_ids):,}")

    # -----------------------------------------------------------------------
    # 1b. Attach raw FOIA match columns needed for TRK exact-match join.
    #     Primary: check if they're already in the merged parquet.
    #     Fallback: load from foia_raw_match_{run_tag}.parquet (saved by
    #               rev_users_clean.py; only that script needs to be rerun,
    #               not indiv_merge.py).
    # -----------------------------------------------------------------------
    _raw_match_cols = [
        "FIRST_DECISION", "BASIS_FOR_CLASSIFICATION", "BEN_SEX",
        "BEN_COUNTRY_OF_BIRTH", "S3Q1", "ED_LEVEL_DEFINITION",
        "BEN_PFIELD_OF_STUDY", "i129_employer_name",
        "PET_CITY", "PET_STATE", "JOB_TITLE",
        "DOT_CODE", "BEN_COMP_PAID", "valid_from", "valid_to",
    ]
    _cols_missing = [c for c in _raw_match_cols if c not in merged.columns]

    if not _cols_missing:
        print(f"\n  [1b] Raw FOIA match columns already present in merged parquet — skipping load")
    else:
        print(f"\n  [1b] Loading raw FOIA match columns from foia_raw_match "
              f"({len(_cols_missing)} missing from merged)...")
        _raw_match_tmpl = MQ_CFG.get(
            "foia_raw_match_parquet",
            str(Path(root) / "data" / "int" / "foia_raw_match_{run_tag}.parquet")
        )
        _raw_match_path = _raw_match_tmpl.replace("{run_tag}", run_tag)
        _load_cols = ["foia_indiv_id"] + _cols_missing
        try:
            foia_raw = pd.read_parquet(_raw_match_path, columns=_load_cols)
        except Exception:
            print(f"  [WARN] foia_raw_match not found — re-run rev_users_clean.py to generate "
                  f"foia_raw_match_{run_tag}.parquet")
            foia_raw = pd.DataFrame({"foia_indiv_id": foia_ids})
            for c in _cols_missing:
                foia_raw[c] = None

        foia_raw = foia_raw[foia_raw["foia_indiv_id"].isin(foia_ids)]
        merged = merged.merge(foia_raw, on="foia_indiv_id", how="left")
        print(f"  Raw FOIA cols merged: {foia_raw['foia_indiv_id'].nunique():,} apps")

    # Normalize all raw match columns to uppercase strings (align with TRK convention).
    # "NA" (pandas nullable string representation) must also be treated as null.
    _NULL_VALS = {"NAN", "NONE", "NA", "N/A", "<NA>", ""}
    for c in _raw_match_cols:
        if c in merged.columns:
            merged[c] = merged[c].astype(str).str.strip().str.upper()
            merged[c] = merged[c].apply(lambda x: None if x in _NULL_VALS else x)

    # -----------------------------------------------------------------------
    # 2. Load Revelio education (university + degree)
    # -----------------------------------------------------------------------
    print(f"\n  [2/7] Loading Revelio education data...")
    rev_educ_path = icfg.choose_path(icfg.REV_EDUC_LONG_PARQUET, icfg.REV_EDUC_LONG_PARQUET_LEGACY)
    user_ids_df = pd.DataFrame({"user_id": user_ids})
    con.register("_filter_users", user_ids_df)

    rev_educ = con.execute(f"""
        SELECT r.user_id, r.education_number, r.university_raw, r.degree_clean, r.ed_enddate
        FROM read_parquet('{rev_educ_path}') r
        INNER JOIN _filter_users f ON r.user_id = f.user_id
        WHERE r.degree_clean IS NOT NULL
          AND r.degree_clean != 'Non-Degree'
    """).df()
    print(f"  Revelio education records: {len(rev_educ):,} "
          f"for {rev_educ['user_id'].nunique():,} users")

    # -----------------------------------------------------------------------
    # 3. Load Revelio field of study (from wrds_users — has 'field' column)
    # -----------------------------------------------------------------------
    print(f"\n  [3/7] Loading Revelio field-of-study data...")
    wrds_path_tmpl = MQ_CFG.get("wrds_users_parquet",
                                 str(Path(root) / "data" / "int" / "wrds_users_{run_tag}.parquet"))
    wrds_path = wrds_path_tmpl.replace("{run_tag}", run_tag)
    if not os.path.exists(wrds_path):
        print(f"  [WARN] wrds_users parquet not found: {wrds_path} — skipping field match")
        rev_field = pd.DataFrame(columns=["user_id", "education_number",
                                          "field_rev", "field_raw_clean_rev"])
    else:
        # field      — Revelio's 17-value pre-categorized column (for category match)
        # field_raw  — free-text entry (for string fuzzy match after normalization)
        field_raw_sql = field_clean_regex_sql("field_raw")
        rev_field = con.execute(f"""
            SELECT r.user_id,
                   r.education_number,
                   r.field                AS field_rev,
                   {field_raw_sql}        AS field_raw_clean_rev
            FROM read_parquet('{wrds_path}') r
            INNER JOIN _filter_users f ON r.user_id = f.user_id
            WHERE r.field IS NOT NULL AND TRIM(r.field) != ''
        """).df()
        # Deduplicate by (user_id, education_number) — keep first
        rev_field = rev_field.drop_duplicates(subset=["user_id", "education_number"])
        n_with_raw = rev_field["field_raw_clean_rev"].notna().sum()
        print(f"  Revelio field records: {len(rev_field):,} "
              f"for {rev_field['user_id'].nunique():,} users "
              f"({n_with_raw:,} with field_raw for string match)")

    # -----------------------------------------------------------------------
    # 4. Load Revelio positions (for title match at FOIA-firm rcid)
    # -----------------------------------------------------------------------
    print(f"\n  [4/7] Loading Revelio positions data...")
    pos_path = icfg.choose_path(icfg.MERGED_POS_CLEAN_PARQUET, icfg.MERGED_POS_CLEAN_PARQUET_LEGACY)

    # Register FOIA-firm (user_id, rcid) pairs to filter positions
    foia_rcids = merged[["user_id", "rcid"]].drop_duplicates()
    con.register("_foia_rcids", foia_rcids)

    positions = con.execute(f"""
        SELECT p.user_id, p.rcid, p.title_raw
        FROM read_parquet('{pos_path}') p
        INNER JOIN _filter_users f ON p.user_id = f.user_id
        INNER JOIN _foia_rcids fr ON p.user_id = fr.user_id AND p.rcid = fr.rcid
        WHERE p.title_raw IS NOT NULL AND TRIM(p.title_raw) != ''
    """).df()
    positions = positions.drop_duplicates(subset=["user_id", "rcid", "title_raw"])
    print(f"  Revelio FOIA-firm positions: {len(positions):,} "
          f"for {positions['user_id'].nunique():,} users")
    con.close()

    # -----------------------------------------------------------------------
    # 5. Education level match
    # -----------------------------------------------------------------------
    print(f"\n  [5/7] Computing match indicators...")
    print(f"    5a. Education level match...")

    merged["ed_cat_foia"] = merged["foia_highest_ed_level"].map(_ed_category)
    rev_educ["ed_cat_rev"] = rev_educ["degree_clean"].map(_ed_category)

    # Expand to (foia_indiv_id, user_id) × Revelio education records
    ed_pairs = merged[["foia_indiv_id", "user_id", "weight_norm", "ed_cat_foia",
                        "foia_highest_ed_level", "lottery_year"]].merge(
        rev_educ[["user_id", "ed_cat_rev", "degree_clean", "ed_enddate"]].dropna(subset=["ed_cat_rev"]),
        on="user_id", how="left"
    )
    # Keep only education records completed before the lottery (April of fiscal_year - 1,
    # since lottery_year is a fiscal year: FY2024 lottery occurs in April 2023)
    # Null ed_enddate is kept conservatively (end date unknown)
    lottery_cutoff = (ed_pairs["lottery_year"].str[:4].astype(int) - 1).astype(str) + "-04"
    ed_enddate_str = ed_pairs["ed_enddate"].fillna("").str[:7]
    n_before = len(ed_pairs)
    ed_pairs = ed_pairs[ed_pairs["ed_enddate"].isna() | (ed_enddate_str < lottery_cutoff)]
    print(f"      Education records after lottery-date filter: {len(ed_pairs):,} "
          f"(dropped {n_before - len(ed_pairs):,} post-lottery records)")
    # Map Revelio ed categories to numeric rank to identify highest pre-lottery degree
    _rank_to_cat = {v: k for k, v in _ED_RANK.items()}
    ed_pairs["ed_rank_rev"] = ed_pairs["ed_cat_rev"].map(_ED_RANK)

    # Reduce to one row per (foia_indiv_id, user_id): Revelio's highest pre-lottery degree.
    # max() on ranks returns NaN if user has no valid pre-lottery ed records.
    rev_highest = (
        ed_pairs
        .groupby(["foia_indiv_id", "user_id", "weight_norm", "ed_cat_foia"])["ed_rank_rev"]
        .max()
        .reset_index()
    )
    rev_highest["ed_cat_rev_highest"] = rev_highest["ed_rank_rev"].map(_rank_to_cat)

    # Compare FOIA highest degree vs Revelio highest pre-lottery degree (one comparison per pair)
    rev_highest["ed_level_match"] = (
        (rev_highest["ed_cat_foia"] == rev_highest["ed_cat_rev_highest"]) &
        rev_highest["ed_cat_foia"].notna() &
        rev_highest["ed_cat_rev_highest"].notna()
    ).astype(float)
    # NaN where FOIA side is missing
    rev_highest.loc[rev_highest["ed_cat_foia"].isna(), "ed_level_match"] = np.nan
    
    # NAN WHERE REV HIGH IS MISSING? (conservative: if we have no pre-lottery ed data, don't say it's a mismatch)   
    rev_highest.loc[rev_highest["ed_cat_rev_highest"].isna(), "ed_level_match"] = np.nan

    ed_match = rev_highest[["foia_indiv_id", "user_id", "weight_norm", "ed_level_match"]]

    n_foia_ed = rev_highest['ed_level_match'].notna().sum() #merged["ed_cat_foia"].notna().sum()
    print(f"    Ed level: {100*n_foia_ed/len(merged):.1f}% of pairs have FOIA ed data "
          f"(comparing FOIA highest vs Revelio highest pre-lottery degree)")

    # -----------------------------------------------------------------------
    # 5b. Institution match (ADE applicants only)
    #     Exact-match TRK raw data to foia_indiv on 15 raw columns, then
    #     clean institution_txt for fuzzy comparison to Revelio universities.
    # -----------------------------------------------------------------------
    print(f"    5b. Institution match (ADE only)...")

    # Load raw TRK (institution_txt rows only, all columns uppercased)
    trk_raw = _load_trk_raw(force_rebuild=FORCE_REBUILD)

    _JOIN_COLS = [
        "FIRST_DECISION", "BASIS_FOR_CLASSIFICATION", "BEN_SEX",
        "BEN_COUNTRY_OF_BIRTH", "S3Q1", "ED_LEVEL_DEFINITION",
        "BEN_PFIELD_OF_STUDY", "i129_employer_name",
        "PET_CITY", "PET_STATE", "JOB_TITLE",
        "DOT_CODE", "BEN_COMP_PAID", "valid_from", "valid_to",
    ]

    if len(trk_raw) > 0:
        # Rename TRK columns to match FOIA column names
        trk_raw = trk_raw.rename(columns={
            "PET_FIRM_NAME": "i129_employer_name",
            "VALID_FROM":    "valid_from",
            "VALID_TO":      "valid_to",
        })
        # Normalize TRK join columns to uppercase strings.
        # TRK redacts some fields with FOIA exemption codes like "(B)(3) (B)(6) (B)(7)(C)".
        # Treat these as NULL so IS NOT DISTINCT FROM matches Bloomberg FOIA nulls.
        _NULL_VALS = {"NAN", "NONE", "NA", "N/A", "<NA>", ""}
        import re as _re
        _foia_exempt_pat = _re.compile(r'^\(B\)\(')
        for c in _JOIN_COLS:
            if c in trk_raw.columns:
                trk_raw[c] = trk_raw[c].astype(str).str.strip().str.upper()
                trk_raw[c] = trk_raw[c].apply(
                    lambda x: None if (x in _NULL_VALS or _foia_exempt_pat.match(x or "")) else x
                )

        # Derive ade_ind_trk from BASIS_FOR_CLASSIFICATION before the join
        if "BASIS_FOR_CLASSIFICATION" in trk_raw.columns:
            trk_raw["ade_ind_trk"] = (trk_raw["S3Q1"] == 'M')
        else:
            trk_raw["ade_ind_trk"] = 0

        # Exact-match join: merged (with raw FOIA cols) × TRK on 15 columns.
        # Use DuckDB with IS NOT DISTINCT FROM so NULL == NULL matches correctly.
        _con_inst = duckdb.connect()
        # Use only foia_indiv_id + join cols from merged (one row per app)
        _avail_join = [c for c in _JOIN_COLS if c in merged.columns]
        _missing = [c for c in _JOIN_COLS if c not in merged.columns]
        if _missing:
            print(f"    [WARN] Missing FOIA join cols (need pipeline re-run): {_missing}")

        # Normalize date columns to YYYY-MM-DD strings on both sides.
        # TRK xlsx reads valid_from/valid_to as datetime64 → astype(str) gives
        # "2020-10-01 00:00:00", while Bloomberg FOIA stores them as "2020-10-01".
        _date_cols = ["valid_from", "valid_to"]
        for _dc in _date_cols:
            for _df in [merged, trk_raw]:
                if _dc in _df.columns:
                    _df[_dc] = (pd.to_datetime(_df[_dc], errors="coerce")
                                  .dt.strftime("%Y-%m-%d")
                                  .where(_df[_dc].notna(), other=None))
                    
        # Normalize compensation columns to float
        if "BEN_COMP_PAID" in trk_raw.columns:
            trk_raw["BEN_COMP_PAID"] = pd.to_numeric(trk_raw["BEN_COMP_PAID"], errors="coerce")
        if "BEN_COMP_PAID" in merged.columns:
            merged["BEN_COMP_PAID"] = pd.to_numeric(merged["BEN_COMP_PAID"], errors="coerce")

        _foia_sub = merged[(merged['status_type']=="SELECTED")&(merged['S3Q1']=='M')][["foia_indiv_id"] + _avail_join].drop_duplicates("foia_indiv_id")
        _trk_sub = trk_raw[[c for c in _JOIN_COLS if c in trk_raw.columns] +
                            ["INSTITUTION_TXT", "ade_ind_trk"]]

        # Diagnostic: show sample values for a few join columns on each side
        print(f"    [DEBUG] _avail_join ({len(_avail_join)} cols): {_avail_join}")
        for _dc in _JOIN_COLS: #["FIRST_DECISION", "valid_from", "valid_to", "BEN_COUNTRY_OF_BIRTH"]:
            if _dc in _foia_sub.columns:
                _f_vals = _foia_sub[_dc].dropna().unique()[:3].tolist()
                print(f"      FOIA {_dc}: {_f_vals}")
            if _dc in _trk_sub.columns:
                _t_vals = _trk_sub[_dc].dropna().unique()[:3].tolist()
                print(f"      TRK  {_dc}: {_t_vals}")

        _con_inst.register("_foia_join", _foia_sub)
        _con_inst.register("_trk_join", _trk_sub)

        # Asymmetric join: TRK NULL acts as wildcard (skip that column); TRK non-NULL
        # must exactly match FOIA. This handles TRK FOIA-exemption redactions without
        # causing massive fan-out from symmetric NULL=NULL matching.
        # Also require TRK row to have at least MIN_TRK_NONNULL non-NULL join columns
        # so that rows redacted on most columns don't match too broadly.
        _MIN_TRK_NONNULL = MQ_CFG.get("trk_min_nonnull_join_cols", 8)
        _join_conds = " AND ".join(
            [f"((t.{c} = f.{c}) OR (t.{c} IS NULL AND f.{c} IS NULL))" for c in _avail_join]
        )
        _nonnull_expr = " + ".join(
            [f"(t.{c} IS NOT NULL)::INT" for c in _avail_join]
        )
        trk_matched = _con_inst.execute(f"""
            SELECT f.foia_indiv_id,
                   t.INSTITUTION_TXT AS institution_txt,
                   t.ade_ind_trk,
                   ({_nonnull_expr}) AS n_trk_nonnull
            FROM _foia_join f
            INNER JOIN _trk_join t ON {_join_conds}
            WHERE ({_nonnull_expr}) >= {_MIN_TRK_NONNULL}
        """).df()
        _con_inst.close()
        print(f"    TRK exact-match joined: {len(trk_matched):,} rows "
              f"(min non-null TRK cols: {_MIN_TRK_NONNULL})")
        if len(trk_matched):
            print(f"    n_trk_nonnull: median={trk_matched['n_trk_nonnull'].median():.0f}, "
                  f"min={trk_matched['n_trk_nonnull'].min()}")

        # Deduplicate: for each foia_indiv_id, keep the TRK row with the most
        # non-NULL join columns (highest n_trk_nonnull = best / most specific match).
        # Drop only when two TRK rows are tied at the top score (truly ambiguous).
        _dup_mask = trk_matched.duplicated("foia_indiv_id", keep=False)
        _dup_ids = trk_matched.loc[_dup_mask, "foia_indiv_id"].unique()

        if len(_dup_ids):
            _dup_df = trk_matched[_dup_mask].copy()
            # For each app, find the max non-null score among its TRK matches
            _max_score = (_dup_df.groupby("foia_indiv_id")["n_trk_nonnull"]
                          .max().rename("_max_score"))
            _dup_df = _dup_df.join(_max_score, on="foia_indiv_id")
            # Keep only rows that hit the max score for their app
            _dup_df = _dup_df[_dup_df["n_trk_nonnull"] == _dup_df["_max_score"]]
            # If multiple TRK rows share the top score → still ambiguous → drop
            _still_dup = _dup_df.duplicated("foia_indiv_id", keep=False)
            _tie_ids = _dup_df.loc[_still_dup, "foia_indiv_id"].unique()
            _rescued = (_dup_df[~_still_dup]
                        .drop(columns="_max_score")
                        .drop_duplicates("foia_indiv_id"))
        else:
            _rescued = trk_matched[~_dup_mask].copy()
            _tie_ids = []

        trk_matched = pd.concat([
            trk_matched[~_dup_mask],
            _rescued
        ], ignore_index=True)
        print(f"    After dedup: {len(trk_matched):,} unique rows "
              f"({len(_dup_ids):,} initially ambiguous, "
              f"{len(_rescued):,} rescued by best-match score, "
              f"{len(_tie_ids):,} unresolvable ties dropped)")

        # Clean institution text
        _con_clean = duckdb.connect()
        _con_clean.register("_tmp_inst", trk_matched)
        _inst_sql = inst_clean_regex_sql("institution_txt")
        inst_supplement = _con_clean.execute(f"""
            SELECT foia_indiv_id,
                   {_inst_sql} AS institution_clean,
                   ade_ind_trk
            FROM _tmp_inst
            WHERE institution_txt IS NOT NULL
              AND TRIM(CAST(institution_txt AS VARCHAR)) != ''
        """).df()
        _con_clean.close()
    else:
        print(f"    [WARN] TRK raw returned empty — no institution supplement available")
        inst_supplement = pd.DataFrame(columns=["foia_indiv_id", "institution_clean", "ade_ind_trk"])

    ade_pairs = merged[merged["ade_ind"] == 1][["foia_indiv_id", "user_id", "weight_norm"]].copy()
    ade_pairs = ade_pairs.merge(
        inst_supplement[["foia_indiv_id", "institution_clean"]],
        on="foia_indiv_id", how="left"
    )
    n_ade_total = len(ade_pairs)
    n_ade_with_inst = ade_pairs["institution_clean"].notna().sum()
    print(f"    ADE pairs: {n_ade_total:,}, with TRK institution: "
          f"{n_ade_with_inst:,} ({100*n_ade_with_inst/max(n_ade_total,1):.1f}%)")

    # Start inst_match as NaN for all pairs (non-ADE stays NaN)
    inst_match = merged[["foia_indiv_id", "user_id", "weight_norm"]].copy()
    inst_match["inst_match"] = np.nan

    if n_ade_with_inst > 0:
        ade_with_inst = ade_pairs[ade_pairs["institution_clean"].notna()].copy()

        # Fuzzy join at unique-string level: FOIA institutions × Revelio universities
        foia_insts_uniq = (ade_with_inst[["institution_clean"]]
                           .drop_duplicates().reset_index(drop=True))
        rev_univs_uniq = (rev_educ[rev_educ["university_raw"].notna()][["university_raw"]]
                          .drop_duplicates().reset_index(drop=True))

        print(f"    Fuzzy matching {len(foia_insts_uniq):,} FOIA institutions × "
              f"{len(rev_univs_uniq):,} Revelio universities (threshold={INST_THRESHOLD})...")
        t_inst = time.time()

        inst_fuzzy = fuzzy_join_lev_jw(
            foia_insts_uniq, rev_univs_uniq,
            left_on="institution_clean", right_on="university_raw",
            threshold=INST_THRESHOLD, top_n=1
        )
        print(f"    Fuzzy join done: {len(inst_fuzzy):,} matched pairs ({time.time()-t_inst:.1f}s)")

        # Subset match: fire when all tokens of the shorter name appear in the longer
        # name's token set, provided the shorter name has >2 tokens and at least one
        # token that is not in the common stopword set (to avoid e.g. "State University"
        # matching everything).
        _COMMON_TOKENS = {
            "university", "college", "state", "science", "sciences",
            "technology", "technologies", "institute", "institution",
            "school", "department", "faculty", "national", "international",
            "american", "technical", "polytechnic",
            "of", "the", "and", "for", "at", "a", "an", "in",
        }

        def _inst_subset_match(a_norm: str, b_norm: str) -> bool:
            """Return True if all tokens of the shorter string appear in the longer
            string's token set, subject to quality guards on the shorter side."""
            toks_a = set(a_norm.split())
            toks_b = set(b_norm.split())
            shorter, longer = (toks_a, toks_b) if len(toks_a) <= len(toks_b) else (toks_b, toks_a)
            if len(shorter) <= 2:
                return False
            if not (shorter - _COMMON_TOKENS):  # all tokens are common
                return False
            return shorter.issubset(longer)

        t_subset = time.time()
        foia_norms = [(s, _norm(s)) for s in foia_insts_uniq["institution_clean"]]
        rev_norms  = [(s, _norm(s)) for s in rev_univs_uniq["university_raw"]]
        inst_subset_pairs: set = set()
        for (f_raw, f_norm), (r_raw, r_norm) in itertools.product(foia_norms, rev_norms):
            if _inst_subset_match(f_norm, r_norm):
                inst_subset_pairs.add((f_raw, r_raw))
        print(f"    Subset match pairs: {len(inst_subset_pairs):,} "
              f"({time.time()-t_subset:.1f}s)")

        if len(inst_fuzzy) > 0 or inst_subset_pairs:
            # Union of fuzzy-score matches and token-subset matches
            inst_match_pairs = set(zip(
                inst_fuzzy["institution_clean_left"],
                inst_fuzzy["university_raw_right"]
            )) | inst_subset_pairs
            print(f"    Combined match pairs (fuzzy ∪ subset): {len(inst_match_pairs):,}")

            # Expand ADE pairs with all Revelio universities per user
            ade_exp = ade_with_inst.merge(
                rev_educ[["user_id", "university_raw"]].dropna(subset=["university_raw"]),
                on="user_id", how="left"
            )
            # Vectorized match: check if (institution_clean, university_raw) in lookup set
            keys = list(zip(ade_exp["institution_clean"].fillna("__NULL__"),
                            ade_exp["university_raw"].fillna("__NULL__")))
            ade_exp["inst_match"] = [
                int(k in inst_match_pairs)
                if (pd.notna(r_inst) and pd.notna(r_univ)) else np.nan
                for k, r_inst, r_univ in zip(
                    keys,
                    ade_exp["institution_clean"],
                    ade_exp["university_raw"]
                )
            ]

            # Max match across Revelio records per pair
            pair_inst = (ade_exp
                         .groupby(["foia_indiv_id", "user_id"])["inst_match"]
                         .max()
                         .reset_index(name="inst_match_new"))

            inst_match = inst_match.merge(pair_inst, on=["foia_indiv_id", "user_id"], how="left")
            # Fill inst_match for ADE rows that have TRK institution data
            has_inst = inst_match["foia_indiv_id"].isin(
                ade_with_inst["foia_indiv_id"].unique()
            )
            inst_match.loc[has_inst, "inst_match"] = inst_match.loc[has_inst, "inst_match_new"]
            inst_match = inst_match.drop(columns=["inst_match_new"])

        n_inst_match = (inst_match["inst_match"] == 1).sum()
        n_inst_valid = inst_match["inst_match"].notna().sum()
        print(f"    Institution match rate (ADE, valid): "
              f"{100*n_inst_match/max(n_inst_valid,1):.1f}%")

    # -----------------------------------------------------------------------
    # 5c. Field of study match (string + category, composite = max)
    # -----------------------------------------------------------------------
    print(f"    5c. Field of study match...")

    # Merge pairs: include both field_rev (category match) and field_raw_clean_rev (string match)
    # Drop rows where Revelio has no field at all; field_raw_clean_rev may still be NaN separately
    field_pairs = merged[["foia_indiv_id", "user_id", "weight_norm", "field_clean_foia"]].merge(
        rev_field[["user_id", "field_rev", "field_raw_clean_rev"]].dropna(subset=["field_rev"]),
        on="user_id", how="left"
    )
    n_foia_field = merged["field_clean_foia"].notna().sum()
    # Exclude the literal string "na" that field_clean_regex_sql can produce from null values
    n_foia_field_valid = merged["field_clean_foia"].apply(
        lambda x: pd.notna(x) and str(x).strip() not in ("", "na", "n/a", "none")
    ).sum()
    print(f"    FOIA field non-null: {100*n_foia_field/len(merged):.1f}% of pairs "
          f"({100*n_foia_field_valid/len(merged):.1f}% non-trivial)")

    # --- Tier 1: fuzzy string match — field_clean_foia × field_raw_clean_rev ---
    # Both sides are free-text normalized by field_clean_regex_sql, making this
    # apples-to-apples (e.g. "computer science" vs "computer science").
    foia_fields_uniq = (
        merged[merged["field_clean_foia"].notna() &
               ~merged["field_clean_foia"].isin(["na", "n/a", "none", ""])][["field_clean_foia"]]
        .drop_duplicates().reset_index(drop=True)
    )
    rev_fields_raw_uniq = (
        rev_field[rev_field["field_raw_clean_rev"].notna() &
                  (rev_field["field_raw_clean_rev"].str.strip() != "")][["field_raw_clean_rev"]]
        .drop_duplicates().reset_index(drop=True)
    )

    field_match_pairs_str = set()
    if len(foia_fields_uniq) > 0 and len(rev_fields_raw_uniq) > 0:
        print(f"    Fuzzy matching {len(foia_fields_uniq):,} FOIA fields × "
              f"{len(rev_fields_raw_uniq):,} Revelio field_raw strings "
              f"(threshold={FIELD_STR_THRESHOLD})...")
        t_field = time.time()
        field_fuzzy = fuzzy_join_lev_jw(
            foia_fields_uniq, rev_fields_raw_uniq,
            left_on="field_clean_foia", right_on="field_raw_clean_rev",
            threshold=FIELD_STR_THRESHOLD, top_n=1
        )
        print(f"    Field fuzzy join: {len(field_fuzzy):,} matched pairs "
              f"({time.time()-t_field:.1f}s)")
        if len(field_fuzzy) > 0:
            field_match_pairs_str = set(zip(
                field_fuzzy["field_clean_foia_left"],
                field_fuzzy["field_raw_clean_rev_right"]
            ))

    keys_f = list(zip(field_pairs["field_clean_foia"].fillna("__NULL__"),
                      field_pairs["field_raw_clean_rev"].fillna("__NULL__")))
    trivial_foia = {"na", "n/a", "none", ""}
    field_pairs["field_match_str"] = [
        int(k in field_match_pairs_str)
        if (pd.notna(r_f) and str(r_f).strip() not in trivial_foia and pd.notna(r_r)) else np.nan
        for k, r_f, r_r in zip(
            keys_f, field_pairs["field_clean_foia"], field_pairs["field_raw_clean_rev"]
        )
    ]
    # NaN where FOIA field is missing or trivial
    field_pairs.loc[
        field_pairs["field_clean_foia"].isna() |
        field_pairs["field_clean_foia"].isin(trivial_foia),
        "field_match_str"
    ] = np.nan

    # --- Tier 2: broad category match ---
    # FOIA: apply _field_category() to cleaned free-text
    # Revelio: look up the pre-categorized `field` value directly in _REV_FIELD_CAT
    # This avoids running regex on already-standardized Revelio category labels.
    field_pairs["field_cat_foia"] = field_pairs["field_clean_foia"].map(_field_category)
    field_pairs["field_cat_rev"] = field_pairs["field_rev"].map(_REV_FIELD_CAT)
    field_pairs["field_match_cat"] = (
        (field_pairs["field_cat_foia"] == field_pairs["field_cat_rev"]) &
        field_pairs["field_cat_foia"].notna() &
        field_pairs["field_cat_rev"].notna()
    ).astype(float)
    field_pairs.loc[
        field_pairs["field_cat_foia"].isna() | field_pairs["field_cat_rev"].isna(),
        "field_match_cat"
    ] = np.nan

    # Composite: max(string, category)
    field_pairs["field_match"] = field_pairs[["field_match_str", "field_match_cat"]].max(axis=1)

    # Aggregate to (foia_indiv_id, user_id) level
    field_agg = (field_pairs
                 .groupby(["foia_indiv_id", "user_id", "weight_norm"])[
                     ["field_match_str", "field_match_cat", "field_match"]]
                 .max()
                 .reset_index())

    str_rate = field_agg["field_match_str"].dropna().mean()
    cat_rate = field_agg["field_match_cat"].dropna().mean()
    comp_rate = field_agg["field_match"].dropna().mean()
    print(f"    Field match — string: {100*str_rate:.1f}%, "
          f"category: {100*cat_rate:.1f}%, composite: {100*comp_rate:.1f}%")

    # -----------------------------------------------------------------------
    # 5d. Job title match
    # -----------------------------------------------------------------------
    print(f"    5d. Job title match...")

    title_pairs = merged[["foia_indiv_id", "user_id", "weight_norm",
                           "rcid", "job_title_foia"]].merge(
        positions[["user_id", "rcid", "title_raw"]].dropna(subset=["title_raw"]),
        on=["user_id", "rcid"], how="left"
    )
    n_foia_title = merged["job_title_foia"].notna().sum()
    print(f"    FOIA job title non-null: {100*n_foia_title/len(merged):.1f}% of pairs")

    # Fuzzy match on unique string pairs
    foia_titles_uniq = (merged[merged["job_title_foia"].notna()][["job_title_foia"]]
                        .drop_duplicates().reset_index(drop=True))
    rev_titles_uniq = (positions[positions["title_raw"].notna()][["title_raw"]]
                       .drop_duplicates().reset_index(drop=True))

    title_match_pairs = set()
    if len(foia_titles_uniq) > 0 and len(rev_titles_uniq) > 0:
        print(f"    Fuzzy matching {len(foia_titles_uniq):,} FOIA titles × "
              f"{len(rev_titles_uniq):,} Revelio titles "
              f"(threshold={TITLE_THRESHOLD}, top_n={TITLE_TOP_N})...")
        t_title = time.time()
        title_fuzzy = fuzzy_join_lev_jw(
            foia_titles_uniq, rev_titles_uniq,
            left_on="job_title_foia", right_on="title_raw",
            threshold=TITLE_THRESHOLD, top_n=TITLE_TOP_N
        )
        print(f"    Title fuzzy join: {len(title_fuzzy):,} matched pairs "
              f"({time.time()-t_title:.1f}s)")
        if len(title_fuzzy) > 0:
            title_match_pairs = set(zip(
                title_fuzzy["job_title_foia_left"],
                title_fuzzy["title_raw_right"]
            ))

    keys_t = list(zip(title_pairs["job_title_foia"].fillna("__NULL__"),
                      title_pairs["title_raw"].fillna("__NULL__")))
    title_pairs["title_match"] = [
        int(k in title_match_pairs)
        if (pd.notna(r_j) and pd.notna(r_t)) else np.nan
        for k, r_j, r_t in zip(
            keys_t, title_pairs["job_title_foia"], title_pairs["title_raw"]
        )
    ]
    title_pairs.loc[title_pairs["job_title_foia"].isna(), "title_match"] = np.nan

    title_agg = (title_pairs
                 .groupby(["foia_indiv_id", "user_id", "weight_norm"])["title_match"]
                 .max()
                 .reset_index())

    t_rate = title_agg["title_match"].dropna().mean()
    print(f"    Title match rate: {100*t_rate:.1f}%")

    # -----------------------------------------------------------------------
    # 6. Merge all match indicators to pair level and aggregate to app level
    # -----------------------------------------------------------------------
    print(f"\n  [6/7] Aggregating to application level...")

    scores = merged[["foia_indiv_id", "user_id", "weight_norm"]].copy()
    scores = scores.merge(ed_match.rename(columns={"ed_level_match": "ed_level_match"}),
                          on=["foia_indiv_id", "user_id", "weight_norm"], how="left")
    scores = scores.merge(inst_match[["foia_indiv_id", "user_id", "inst_match"]],
                          on=["foia_indiv_id", "user_id"], how="left")
    scores = scores.merge(field_agg[["foia_indiv_id", "user_id",
                                      "field_match_str", "field_match_cat", "field_match"]],
                          on=["foia_indiv_id", "user_id"], how="left")
    scores = scores.merge(title_agg[["foia_indiv_id", "user_id", "title_match"]],
                          on=["foia_indiv_id", "user_id"], how="left")

    # any_match: 1 if any of the three main indicators (ed, field, title) is 1
    match_cols = ["ed_level_match", "field_match", "title_match"]
    scores["any_match"] = scores[match_cols].max(axis=1)
    # all_match: 1 only if all three are 1 (excludes NaN rows from evaluation)
    scores["all_match"] = scores[match_cols].min(axis=1)
    # ed_plus_any_match: ed level matched AND any of (institution, field, title) matched
    _other_any = scores[["inst_match", "field_match", "title_match"]].max(axis=1)
    scores["ed_plus_any_match"] = np.where(
        scores["ed_level_match"].isna(), np.nan,
        np.where(scores["ed_level_match"] == 0, 0.0, _other_any)
    )

    # Weighted mean per foia_indiv_id
    id_col = "foia_indiv_id"
    app_scores_parts = []
    for col in ["ed_level_match", "inst_match",
                "field_match_str", "field_match_cat", "field_match",
                "title_match", "any_match", "all_match", "ed_plus_any_match"]:
        s = _wtd_mean_by_app(scores, col, id_col)
        app_scores_parts.append(s.rename(f"{col}_w"))

    app_scores = pd.concat(app_scores_parts, axis=1)
    # pd.concat can drop the index name if any part is an empty Series (e.g. inst_match_w
    # when TRK supplement has 0 rows). Restore it explicitly before reset_index().
    app_scores.index.name = id_col
    app_scores = app_scores.reset_index()

    # Attach max weight_norm per app (for quartile plot)
    wn_max = scores.groupby(id_col)["weight_norm"].max().rename("weight_norm_max")
    app_scores = app_scores.merge(wn_max.reset_index(), on=id_col, how="left")

    # Coverage flags: % of apps with non-null FOIA data for each dimension
    ed_cov = merged.groupby("foia_indiv_id")["foia_highest_ed_level"].first().notna().astype(int)
    field_cov = merged.groupby("foia_indiv_id")["field_clean_foia"].first().notna().astype(int)
    title_cov = merged.groupby("foia_indiv_id")["job_title_foia"].first().notna().astype(int)
    inst_cov = (inst_supplement["foia_indiv_id"].isin(merged["foia_indiv_id"])
                .groupby(inst_supplement["foia_indiv_id"]).any().astype(int)
                if len(inst_supplement) > 0
                else pd.Series(0, index=merged["foia_indiv_id"].unique()))

    for cov, name in [(ed_cov, "ed_foia_valid"), (field_cov, "field_foia_valid"),
                      (title_cov, "title_foia_valid")]:
        app_scores = app_scores.merge(
            cov.rename(name).reset_index(), on=id_col, how="left"
        )

    print(f"  App-level scores: {len(app_scores):,} apps computed")

    # Print match rates among apps with successful TRK match (inst_match_w not NaN)
    trk_apps = app_scores[app_scores["inst_match_w"].notna()]
    n_trk = len(trk_apps)
    print(f"\n  TRK-matched apps (ADE institution data found): {n_trk:,} "
          f"({100*n_trk/len(app_scores):.1f}% of total)")
    if n_trk > 0:
        dims_print = [
            ("Institution",   "inst_match_w"),
            ("Field",         "field_match_w"),
            ("Ed level",      "ed_level_match_w"),
            ("Job title",     "title_match_w"),
            ("Any match",     "any_match_w"),
            ("Ed + any other","ed_plus_any_match_w"),
        ]
        print(f"  Match rates among TRK-matched apps:")
        print(f"    {'Dimension':<16}  {'Wtd mean':>8}  {'Max>0 share':>11}  {'n':>7}")
        print(f"    {'-'*16}  {'-'*8}  {'-'*11}  {'-'*7}")
        for label, col in dims_print:
            if col in trk_apps.columns:
                valid = trk_apps[col].dropna()
                rate_mean = valid.mean()         # avg weighted match rate per app
                rate_max = (valid > 0).mean()    # share of apps with at least one pair match
                n_valid = len(valid)
                print(f"    {label:<16}  {100*rate_mean:7.1f}%  {100*rate_max:10.1f}%  {n_valid:>7,}")

    # Save scores parquet
    scores_path = mq_dir / f"match_quality_scores_{variant_name}_{run_tag}.parquet"
    app_scores.to_parquet(scores_path, index=False)
    print(f"  Scores saved to {scores_path}")

    # -----------------------------------------------------------------------
    # 7. Summary table + plots
    # -----------------------------------------------------------------------
    print(f"\n  [7/7] Saving outputs...")
    _save_summary_table(app_scores, variant_name, run_tag, output_dir)
    _save_plots(app_scores, variant_name, run_tag, mq_dir)

    print(f"\n  Match quality check done: {variant_name} ({time.time()-t0:.1f}s)")
    print(f"  Outputs: {mq_dir}")


###############################################################################
# SUMMARY TABLE (LaTeX)
###############################################################################

def _save_summary_table(app_scores: pd.DataFrame, variant_name: str,
                         run_tag: str, output_dir: Path):
    """Build and save LaTeX table: match rates by dimension and weight_norm quartile."""

    df = app_scores.copy()

    # Compute weight_norm quartiles, dropping NaN
    wn_valid = df["weight_norm_max"].dropna()
    if len(wn_valid) >= 4:
        # duplicates="drop" can reduce the number of bins below 4; generate labels dynamically
        _, bins = pd.qcut(wn_valid, q=4, retbins=True, duplicates="drop")
        n_bins = len(bins) - 1
        q_labels = [f"Q{i}" for i in range(1, n_bins + 1)]
        df["wn_q"] = pd.qcut(df["weight_norm_max"], q=4,
                              labels=q_labels, duplicates="drop")
    else:
        df["wn_q"] = "Q1"

    dims = [
        ("Education level",    "ed_level_match_w", "ed_foia_valid"),
        ("Institution (ADE)",  "inst_match_w",     None),
        ("Field (string)",     "field_match_str_w", "field_foia_valid"),
        ("Field (category)",   "field_match_cat_w", "field_foia_valid"),
        ("Field (composite)",  "field_match_w",     "field_foia_valid"),
        ("Job title",          "title_match_w",     "title_foia_valid"),
        ("Any match",          "any_match_w",       None),
        ("Ed + any other",     "ed_plus_any_match_w", None),
    ]

    col_header = (r"Dimension & FOIA cov. & Wtd match rate & "
                  r"Q1 & Q2 & Q3 & Q4 \\")

    rows = []
    for label, col, valid_col in dims:
        if col not in df.columns:
            continue
        pct_foia = (f"{100*df[valid_col].mean():.0f}\\%"
                    if valid_col and valid_col in df.columns else "---")
        overall = f"{100*df[col].dropna().mean():.1f}\\%"
        q_rates = df.groupby("wn_q")[col].mean()
        q_strs = " & ".join(
            f"{100*q_rates.get(q, np.nan):.1f}\\%" if not np.isnan(q_rates.get(q, np.nan))
            else "---"
            for q in ["Q1", "Q2", "Q3", "Q4"]
        )
        rows.append(rf"  {label} & {pct_foia} & {overall} & {q_strs} \\")

    n_apps = len(df)
    latex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Ex-Post Match Quality: \texttt{{{variant_name}}} variant ({run_tag}). "
        rf"N = {n_apps:,} applications.}}",
        rf"\label{{tab:match_quality_{variant_name}}}",
        r"\footnotesize",
        r"\begin{tabular}{lcccccc}",
        r"\hline\hline",
        col_header,
        r"\hline",
        *rows,
        r"\hline\hline",
        (r"\multicolumn{7}{l}{\footnotesize \textit{Notes}: "
         r"Q1--Q4 = weight\_norm quartiles (Q4 = highest-confidence matches). "
         r"Weighted match rate = weighted mean of binary match indicator, "
         r"using weight\_norm as weights. FOIA cov.\ = share of applicants "
         r"with non-null FOIA characteristic.} \\"),
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = output_dir / "tables" / f"match_quality_{variant_name}.tex"
    out_path.write_text(latex)
    print(f"  LaTeX table: {out_path}")


###############################################################################
# PLOTS
###############################################################################

def _save_plots(app_scores: pd.DataFrame, variant_name: str,
                run_tag: str, mq_dir: Path):
    """
    Two plots:
    1. Bar chart: overall weighted match rate by dimension
    2. Line chart: match rate vs weight_norm decile (validates merge quality)
    """
    sns.set_style("whitegrid")

    dim_labels = {
        "ed_level_match_w": "Ed Level",
        "inst_match_w":     "Institution\n(ADE)",
        "field_match_w":    "Field\n(composite)",
        "title_match_w":    "Job Title",
        "any_match_w":      "Any Match",
    }

    # --- Plot 1: Bar chart of overall match rates ---
    bar_data = []
    for col, label in dim_labels.items():
        if col not in app_scores.columns:
            continue
        mean_val = app_scores[col].dropna().mean()
        n = app_scores[col].notna().sum()
        if not np.isnan(mean_val):
            bar_data.append({"Dimension": label, "Match Rate": mean_val, "n": n})

    if bar_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        df_bar = pd.DataFrame(bar_data)
        sns.barplot(data=df_bar, x="Dimension", y="Match Rate", ax=ax,
                    color="steelblue", width=0.6)
        ax.set_ylabel("Weighted Match Rate")
        ax.set_title(f"Ex-Post Match Quality — {variant_name} ({run_tag})")
        ax.set_ylim(0, 1)
        for patch in ax.patches:
            h = patch.get_height()
            ax.text(patch.get_x() + patch.get_width() / 2, h + 0.01,
                    f"{100*h:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        bar_path = mq_dir / f"match_quality_bar_{variant_name}_{run_tag}.png"
        fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Bar chart: {bar_path}")

    # --- Plot 2: Match rate by weight_norm decile ---
    df2 = app_scores.copy()
    try:
        _, bins2 = pd.qcut(df2["weight_norm_max"].dropna(), q=10, retbins=True, duplicates="drop")
        n_dec = len(bins2) - 1
        df2["wn_decile"] = pd.qcut(df2["weight_norm_max"],
                                    q=10,
                                    labels=[f"D{i}" for i in range(1, n_dec + 1)],
                                    duplicates="drop")
    except ValueError:
        df2["wn_decile"] = "D1"

    dec_parts = []
    for col, label in dim_labels.items():
        if col not in df2.columns:
            continue
        grp = df2.groupby("wn_decile")[col].mean().reset_index()
        grp.columns = ["Decile", "Match Rate"]
        grp["Dimension"] = label
        dec_parts.append(grp)

    if dec_parts:
        df_dec = pd.concat(dec_parts, ignore_index=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df_dec, x="Decile", y="Match Rate",
                     hue="Dimension", marker="o", ax=ax)
        ax.set_xlabel("weight_norm Decile (D10 = highest confidence)")
        ax.set_ylabel("Match Rate")
        ax.set_title(f"Match Rate by Merge Confidence — {variant_name} ({run_tag})")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        line_path = mq_dir / f"match_quality_decile_{variant_name}_{run_tag}.png"
        fig.savefig(line_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Decile plot: {line_path}")


###############################################################################
# COMBINED PAPER TABLE
###############################################################################

# Rows shown in the paper match-quality table, in order.
_PAPER_MQ_DIMS: list[tuple[str, str, str | None]] = [
    ("Education level",    "ed_level_match_w",    "ed_foia_valid"),
    ("Institution (ADE)",  "inst_match_w",         None),
    ("Field exact",        "field_match_str_w",    "field_foia_valid"),
    ("Job title",          "title_match_w",        "title_foia_valid"),
    (r"Ed.\ + any other",  "ed_plus_any_match_w",  None),
]

# Default columns for the paper table (variant_name → display label).
_PAPER_MQ_VARIANTS: list[tuple[str, str]] = [
    ("us_educ",         "U.S.-educ."),
    ("us_educ_prefilt", "Pre-filtered"),
    ("us_educ_opt",     "One-to-one"),
    ("strict_med",      "Strict Q50"),
]


def build_paper_match_quality_table(
    variants: list[tuple[str, str]] | None = None,
    run_tag: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """Build and save a combined match quality tabular fragment for the paper.

    Loads saved score parquets from the match quality output directory and
    writes a single LaTeX tabular (no wrapping \\begin{table}) with one column
    per variant to output_dir/tables/paper_match_quality.tex.

    Parameters
    ----------
    variants   : list of (variant_name, display_label); defaults to _PAPER_MQ_VARIANTS
    run_tag    : defaults to icfg.RUN_TAG
    output_dir : where to write tables/ subdir; defaults to reg output_dir from reg.yaml
    """
    if variants is None:
        variants = _PAPER_MQ_VARIANTS
    if run_tag is None:
        run_tag = icfg.RUN_TAG
    if output_dir is None:
        _reg_cfg_path = Path(_CODE_DIR) / "configs" / "reg.yaml"
        _reg_cfg = yaml.safe_load(_reg_cfg_path.read_text()) or {}
        output_dir = Path(
            _reg_cfg.get("output_dir", f"{root}/output/reg").replace("{root}", str(root))
        )

    mq_dir = Path(MQ_CFG["output_dir"])
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load score DataFrames
    dfs: list[tuple[str, str, pd.DataFrame | None]] = []
    for vname, vlabel in variants:
        spath = mq_dir / f"match_quality_scores_{vname}_{run_tag}.parquet"
        if spath.exists():
            dfs.append((vname, vlabel, pd.read_parquet(spath)))
        else:
            warnings.warn(f"Score parquet not found, skipping: {spath}")
            dfs.append((vname, vlabel, None))

    if not any(df is not None for _, _, df in dfs):
        print("  [SKIP] build_paper_match_quality_table: no score parquets found")
        return

    ncols = len(dfs)
    col_spec = "l" + "r" * (ncols + 1)  # dim col + FOIA cov + one per variant

    header = (
        "Dimension & FOIA cov. & "
        + " & ".join(vlabel for _, vlabel, _ in dfs)
        + r" \\"
    )

    body_rows: list[str] = []
    for label, match_col, valid_col in _PAPER_MQ_DIMS:
        # FOIA coverage: range across variants that have the column
        cov_vals = [
            df[valid_col].mean()
            for _, _, df in dfs
            if df is not None and valid_col and valid_col in df.columns
        ]
        if cov_vals:
            lo, hi = min(cov_vals) * 100, max(cov_vals) * 100
            cov_str = (
                rf"{lo:.0f}\%" if hi - lo < 1.0
                else rf"{lo:.0f}--{hi:.0f}\%"
            )
        else:
            cov_str = "---"

        # Weighted match rate per variant
        rate_strs = []
        for _, _, df in dfs:
            if df is None or match_col not in df.columns:
                rate_strs.append("---")
            else:
                val = df[match_col].dropna().mean()
                rate_strs.append(rf"{val*100:.1f}\%" if not np.isnan(val) else "---")

        body_rows.append(
            f"  {label} & {cov_str} & " + " & ".join(rate_strs) + r" \\"
        )

    # N row
    n_strs = [f"{len(df):,}" if df is not None else "---" for _, _, df in dfs]
    body_rows.append(r"  \midrule")
    body_rows.append(r"  $N$ (applications) & & " + " & ".join(n_strs) + r" \\")

    latex = "\n".join([
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])

    out_path = tables_dir / "paper_match_quality.tex"
    out_path.write_text(latex)
    print(f"  Saved combined match quality table → {out_path}")


###############################################################################
# STANDALONE ENTRY POINT
###############################################################################

if __name__ == "__main__" or "__file__" not in globals():
    _t_main = time.time()

    _reg_cfg_path = Path(_CODE_DIR) / "configs" / "reg.yaml"
    _reg_cfg = yaml.safe_load(_reg_cfg_path.read_text()) or {}
    _variants = _reg_cfg.get("variants", [])
    _run_tag = icfg.RUN_TAG
    _use_opt = bool(_reg_cfg.get("use_optimal_dedup", False))
    _output_dir = Path(
        _reg_cfg.get("output_dir", f"{root}/output/reg").replace("{root}", str(root))
    )
    _output_dir.mkdir(parents=True, exist_ok=True)
    (_output_dir / "tables").mkdir(exist_ok=True)

    print(f"\n=== match_quality_check.py standalone ===")
    print(f"run_tag: {_run_tag}")
    print(f"use_optimal_dedup: {_use_opt}")
    print(f"testing: {'ENABLED (n=' + str(TESTING_N_APPS) + ')' if TESTING_ENABLED else 'disabled'}")
    print(f"variants: {[v['name'] for v in _variants]}")
    print()

    for _variant in _variants:
        _pk = _variant["parquet_key"]
        _ppath = getattr(icfg, _pk, None)
        if _ppath is None:
            print(f"[SKIP] Unknown parquet key: {_pk}")
            continue
        if _use_opt:
            _ppath = _ppath.replace(".parquet", "_opt.parquet")
            _legacy = None
        else:
            _legacy = getattr(icfg, _pk + "_LEGACY", None)
        _ppath = icfg.choose_path(_ppath, _legacy)
        if not os.path.exists(_ppath):
            print(f"[SKIP] Parquet not found: {_ppath}")
            continue
        run_match_quality_check(_variant["name"], _run_tag, _output_dir, _ppath)

    build_paper_match_quality_table(run_tag=_run_tag, output_dir=_output_dir)
    print(f"\nAll variants done. Total time: {time.time()-_t_main:.1f}s")
