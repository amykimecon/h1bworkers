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
FORCE_REBUILD = bool(MQ_CFG.get("force_rebuild_supplement", False))

print(f"=== match_quality_check.py loaded ===")
print(f"  testing: {'ENABLED (n=' + str(TESTING_N_APPS) + ')' if TESTING_ENABLED else 'disabled'}")
print(f"  thresholds: inst={INST_THRESHOLD}, field={FIELD_STR_THRESHOLD}, title={TITLE_THRESHOLD}")


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


###############################################################################
# FIELD OF STUDY BROAD CATEGORY MAPPING
###############################################################################

_FIELD_CATS = [
    ("stem", re.compile(
        r"computer|software|engineer|mathematics|math|physics|chemistry|biology|science|"
        r"data|statistics|electrical|mechanical|civil|chemical|information|technology|"
        r"computing|programming|neural|machine learning|artificial intelligence|"
        r"environmental|materials|aerospace|biotech|genomics", re.I)),
    ("business", re.compile(
        r"business|management|accounting|finance|economics|marketing|commerce|"
        r"administration|mba|supply chain|operations|entrepreneurship|strategy", re.I)),
    ("health", re.compile(
        r"medicine|medical|nursing|pharmacy|health|clinical|dental|physician|"
        r"surgery|biochem|biomedical|public health|epidemiology|nutrition|"
        r"physical therapy|occupational therapy", re.I)),
    ("social_sciences", re.compile(
        r"psychology|sociology|political|public policy|international|anthropology|"
        r"geography|criminology|social work|urban|urban planning", re.I)),
    ("humanities", re.compile(
        r"english|history|philosophy|literature|arts|linguistics|communication|"
        r"journalism|music|religion|fine arts|theater|film|media", re.I)),
    ("education", re.compile(
        r"education|teaching|curriculum|pedagogy|instructional|school", re.I)),
]


def _field_category(s) -> str | None:
    """Map field string → stem / business / health / social_sciences / humanities / education / other."""
    if pd.isna(s) or str(s).strip() == "":
        return None
    s_str = str(s)
    for cat, pat in _FIELD_CATS:
        if pat.search(s_str):
            return cat
    return "other"


###############################################################################
# TRK INSTITUTION SUPPLEMENT
###############################################################################

def _load_trk_institution_supplement(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load TRK_12704 xlsx files and join to foia_indiv to obtain institution_txt
    (normalized) per foia_indiv_id for ADE applicants.

    Caches result to parquet. Subsequent calls use the cache unless force_rebuild=True.

    Returns DataFrame with columns:
        foia_indiv_id, institution_clean, ade_ind_trk
    """
    cache_path = MQ_CFG.get(
        "trk_supplement_cache",
        str(Path(root) / "data" / "int" / "trk_institution_supplement.parquet")
    )

    if not force_rebuild and os.path.exists(cache_path):
        print(f"  [TRK supplement] Loading cached supplement ({cache_path})")
        supp = pd.read_parquet(cache_path)
        print(f"  [TRK supplement] {len(supp):,} rows")
        return supp

    print(f"  [TRK supplement] Building from raw xlsx files (this may take a few minutes)...")
    t0 = time.time()

    trk_dir = MQ_CFG.get("trk_data_dir", str(Path(root) / "data" / "raw" / "H-1B Data"))
    trk_files = sorted(glob.glob(os.path.join(trk_dir, "TRK_12704_FY*.xlsx")))
    print(f"  Found {len(trk_files)} TRK xlsx files in {trk_dir}")

    # --- Load and concatenate all TRK files ---
    keep_cols = ["FIRST_DECISION_FY", "PET_FIRM_NAME", "PET_STATE",
                 "BEN_COUNTRY_OF_BIRTH", "BEN_SEX",
                 "institution_txt", "BASIS_FOR_CLASSIFICATION"]
    frames = []
    for f in trk_files:
        try:
            df = pd.read_excel(f, engine="openpyxl", usecols=lambda c: c in keep_cols)
            frames.append(df)
            n_inst = df["institution_txt"].notna().sum() if "institution_txt" in df.columns else 0
            print(f"    {os.path.basename(f)}: {len(df):,} rows, {n_inst:,} with institution_txt")
        except Exception as e:
            print(f"    [WARN] Could not read {os.path.basename(f)}: {e}")

    if not frames:
        print("  [ERROR] No TRK files loaded — returning empty supplement")
        return pd.DataFrame(columns=["foia_indiv_id", "institution_clean", "ade_ind_trk"])

    trk = pd.concat(frames, ignore_index=True)
    print(f"  Total TRK rows loaded: {len(trk):,}")

    # Keep only rows with institution_txt (ADE applicants have it; general category may not)
    trk_ade = trk[trk["institution_txt"].notna() &
                  (trk["institution_txt"].astype(str).str.strip() != "")].copy()
    print(f"  Rows with institution_txt: {len(trk_ade):,}")

    # --- Join to crosswalk: PET_FIRM_NAME + PET_STATE → foia_firm_uid ---
    xwalk_path = MQ_CFG.get("trk_crosswalk",
                             str(Path(root) / "data" / "int" / "trk12704_to_foiafirm.csv"))
    xwalk = pd.read_csv(xwalk_path)[["pet_firm_name", "pet_state", "foia_firm_uid"]].copy()
    xwalk["_firm_key"] = xwalk["pet_firm_name"].astype(str).str.upper().str.strip()
    xwalk["_state_key"] = xwalk["pet_state"].astype(str).str.strip()

    trk_ade["_firm_key"] = trk_ade["PET_FIRM_NAME"].astype(str).str.upper().str.strip()
    trk_ade["_state_key"] = trk_ade["PET_STATE"].astype(str).str.strip()

    trk_ade = trk_ade.merge(
        xwalk[["_firm_key", "_state_key", "foia_firm_uid"]],
        on=["_firm_key", "_state_key"],
        how="left"
    )
    n_with_firm = trk_ade["foia_firm_uid"].notna().sum()
    print(f"  Crosswalk match: {n_with_firm:,}/{len(trk_ade):,} "
          f"({100*n_with_firm/max(len(trk_ade),1):.1f}%)")
    trk_ade = trk_ade[trk_ade["foia_firm_uid"].notna()].copy()

    # --- Normalize join keys ---
    trk_ade["lottery_year"] = trk_ade["FIRST_DECISION_FY"].astype(str).str.strip()
    trk_ade["country_up"] = trk_ade["BEN_COUNTRY_OF_BIRTH"].astype(str).str.upper().str.strip()
    # BEN_SEX: F→1, M→0, unknown→-1 (will be excluded from sex-constrained join)
    trk_ade["female_ind"] = trk_ade["BEN_SEX"].map({"F": 1, "M": 0}).fillna(-1).astype(int)
    trk_ade["ade_ind_trk"] = (
        trk_ade["BASIS_FOR_CLASSIFICATION"].astype(str)
        .str.contains("Advanced Degree|ADE|advanced degree", case=False, na=False)
        .astype(int)
    )

    # --- Load foia_indiv (selected/winners only) ---
    foia_path = icfg.choose_path(icfg.FOIA_INDIV_PARQUET, icfg.FOIA_INDIV_PARQUET_LEGACY)
    con = duckdb.connect()
    foia_sel = con.execute(f"""
        SELECT foia_indiv_id,
               foia_firm_uid,
               CAST(lottery_year AS VARCHAR) AS lottery_year,
               UPPER(TRIM(country))          AS country_up,
               CAST(female_ind AS INTEGER)   AS female_ind
        FROM read_parquet('{foia_path}')
        WHERE status_type = 'SELECTED'
    """).df()
    print(f"  foia_indiv selected rows: {len(foia_sel):,}")

    # --- Join: split by known/unknown sex to avoid excluding unknowns ---
    trk_known = trk_ade[trk_ade["female_ind"] != -1]
    trk_unkn = trk_ade[trk_ade["female_ind"] == -1]

    joined_known = trk_known.merge(
        foia_sel,
        on=["foia_firm_uid", "lottery_year", "country_up", "female_ind"],
        how="inner"
    )
    joined_unkn = trk_unkn.merge(
        foia_sel[["foia_indiv_id", "foia_firm_uid", "lottery_year", "country_up"]],
        on=["foia_firm_uid", "lottery_year", "country_up"],
        how="inner"
    )
    joined = pd.concat([joined_known, joined_unkn], ignore_index=True)
    print(f"  TRK → foia_indiv joined: {len(joined):,} rows")

    # Flag and exclude multi-matches (same foia_indiv_id matched to >1 TRK row)
    dup_ids = joined[joined.duplicated("foia_indiv_id", keep=False)]["foia_indiv_id"].unique()
    n_dup = len(dup_ids)
    joined_uniq = joined[~joined["foia_indiv_id"].isin(dup_ids)].copy()
    print(f"  Excluded {n_dup:,} ambiguous foia_indiv_ids → {len(joined_uniq):,} unique rows kept")

    # --- Clean institution text via DuckDB ---
    con2 = duckdb.connect()
    con2.register("_tmp_inst", joined_uniq[["foia_indiv_id", "institution_txt", "ade_ind_trk"]])
    inst_sql = inst_clean_regex_sql("institution_txt")
    result = con2.execute(f"""
        SELECT foia_indiv_id,
               {inst_sql} AS institution_clean,
               ade_ind_trk
        FROM _tmp_inst
        WHERE institution_txt IS NOT NULL
          AND TRIM(CAST(institution_txt AS VARCHAR)) != ''
    """).df()
    con2.close()
    con.close()

    # Cache
    result.to_parquet(cache_path, index=False)
    print(f"  TRK supplement: {len(result):,} rows saved to {cache_path} "
          f"({time.time()-t0:.1f}s)")
    return result


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
               COALESCE(ade_ind, 0) AS ade_ind
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
    # 2. Load Revelio education (university + degree)
    # -----------------------------------------------------------------------
    print(f"\n  [2/7] Loading Revelio education data...")
    rev_educ_path = icfg.choose_path(icfg.REV_EDUC_LONG_PARQUET, icfg.REV_EDUC_LONG_PARQUET_LEGACY)
    user_ids_df = pd.DataFrame({"user_id": user_ids})
    con.register("_filter_users", user_ids_df)

    rev_educ = con.execute(f"""
        SELECT r.user_id, r.education_number, r.university_raw, r.degree_clean
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
        rev_field = pd.DataFrame(columns=["user_id", "education_number", "field_clean_rev"])
    else:
        field_sql = field_clean_regex_sql("field")
        rev_field = con.execute(f"""
            SELECT r.user_id,
                   r.education_number,
                   {field_sql} AS field_clean_rev
            FROM read_parquet('{wrds_path}') r
            INNER JOIN _filter_users f ON r.user_id = f.user_id
            WHERE r.field IS NOT NULL AND TRIM(r.field) != ''
        """).df()
        # Deduplicate by (user_id, education_number) — keep first
        rev_field = rev_field.drop_duplicates(subset=["user_id", "education_number"])
        print(f"  Revelio field records: {len(rev_field):,} "
              f"for {rev_field['user_id'].nunique():,} users")

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
    ed_pairs = merged[["foia_indiv_id", "user_id", "weight_norm", "ed_cat_foia"]].merge(
        rev_educ[["user_id", "ed_cat_rev"]].dropna(subset=["ed_cat_rev"]),
        on="user_id", how="left"
    )
    ed_pairs["ed_level_match"] = (
        (ed_pairs["ed_cat_foia"] == ed_pairs["ed_cat_rev"]) &
        ed_pairs["ed_cat_foia"].notna() &
        ed_pairs["ed_cat_rev"].notna()
    ).astype(float)
    # NaN where FOIA side is missing
    ed_pairs.loc[ed_pairs["ed_cat_foia"].isna(), "ed_level_match"] = np.nan

    # Max match across Revelio records per (foia_indiv_id, user_id)
    ed_match = (ed_pairs
                .groupby(["foia_indiv_id", "user_id", "weight_norm"])["ed_level_match"]
                .max()
                .reset_index())

    n_foia_ed = merged["ed_cat_foia"].notna().sum()
    print(f"    Ed level: {100*n_foia_ed/len(merged):.1f}% of pairs have FOIA ed data")

    # -----------------------------------------------------------------------
    # 5b. Institution match (ADE applicants only)
    # -----------------------------------------------------------------------
    print(f"    5b. Institution match (ADE only)...")
    inst_supplement = _load_trk_institution_supplement(force_rebuild=FORCE_REBUILD)

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

        if len(inst_fuzzy) > 0:
            # Build lookup set of matched (institution_clean, university_raw) pairs
            inst_match_pairs = set(zip(
                inst_fuzzy["institution_clean_left"],
                inst_fuzzy["university_raw_right"]
            ))

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

    field_pairs = merged[["foia_indiv_id", "user_id", "weight_norm", "field_clean_foia"]].merge(
        rev_field[["user_id", "field_clean_rev"]].dropna(subset=["field_clean_rev"]),
        on="user_id", how="left"
    )
    n_foia_field = merged["field_clean_foia"].notna().sum()
    print(f"    FOIA field non-null: "
          f"{100*n_foia_field/len(merged):.1f}% of pairs")

    # --- Tier 1: fuzzy string match on unique strings ---
    foia_fields_uniq = (merged[merged["field_clean_foia"].notna()][["field_clean_foia"]]
                        .drop_duplicates().reset_index(drop=True))
    rev_fields_uniq = (rev_field[rev_field["field_clean_rev"].notna()][["field_clean_rev"]]
                       .drop_duplicates().reset_index(drop=True))

    field_match_pairs_str = set()
    if len(foia_fields_uniq) > 0 and len(rev_fields_uniq) > 0:
        print(f"    Fuzzy matching {len(foia_fields_uniq):,} FOIA fields × "
              f"{len(rev_fields_uniq):,} Revelio fields (threshold={FIELD_STR_THRESHOLD})...")
        t_field = time.time()
        field_fuzzy = fuzzy_join_lev_jw(
            foia_fields_uniq, rev_fields_uniq,
            left_on="field_clean_foia", right_on="field_clean_rev",
            threshold=FIELD_STR_THRESHOLD, top_n=1
        )
        print(f"    Field fuzzy join: {len(field_fuzzy):,} matched pairs "
              f"({time.time()-t_field:.1f}s)")
        if len(field_fuzzy) > 0:
            field_match_pairs_str = set(zip(
                field_fuzzy["field_clean_foia_left"],
                field_fuzzy["field_clean_rev_right"]
            ))

    keys_f = list(zip(field_pairs["field_clean_foia"].fillna("__NULL__"),
                      field_pairs["field_clean_rev"].fillna("__NULL__")))
    field_pairs["field_match_str"] = [
        int(k in field_match_pairs_str)
        if (pd.notna(r_f) and pd.notna(r_r)) else np.nan
        for k, r_f, r_r in zip(
            keys_f, field_pairs["field_clean_foia"], field_pairs["field_clean_rev"]
        )
    ]
    # NaN where FOIA field is missing (we can't evaluate)
    field_pairs.loc[field_pairs["field_clean_foia"].isna(), "field_match_str"] = np.nan

    # --- Tier 2: broad category match ---
    field_pairs["field_cat_foia"] = field_pairs["field_clean_foia"].map(_field_category)
    field_pairs["field_cat_rev"] = field_pairs["field_clean_rev"].map(_field_category)
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
              f"{len(rev_titles_uniq):,} Revelio titles (threshold={TITLE_THRESHOLD})...")
        t_title = time.time()
        title_fuzzy = fuzzy_join_lev_jw(
            foia_titles_uniq, rev_titles_uniq,
            left_on="job_title_foia", right_on="title_raw",
            threshold=TITLE_THRESHOLD, top_n=1
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

    # Weighted mean per foia_indiv_id
    id_col = "foia_indiv_id"
    app_scores_parts = []
    for col in ["ed_level_match", "inst_match",
                "field_match_str", "field_match_cat", "field_match",
                "title_match", "any_match", "all_match"]:
        s = _wtd_mean_by_app(scores, col, id_col)
        app_scores_parts.append(s.rename(f"{col}_w"))

    app_scores = pd.concat(app_scores_parts, axis=1).reset_index()

    # Attach max weight_norm per app (for quartile plot)
    wn_max = scores.groupby(id_col)["weight_norm"].max().rename("weight_norm_max")
    app_scores = app_scores.merge(wn_max, on=id_col, how="left")

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
        df["wn_q"] = pd.qcut(df["weight_norm_max"], q=4,
                              labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
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
        df2["wn_decile"] = pd.qcut(df2["weight_norm_max"], q=10,
                                    labels=[f"D{i}" for i in range(1, 11)],
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

    print(f"\nAll variants done. Total time: {time.time()-_t_main:.1f}s")
