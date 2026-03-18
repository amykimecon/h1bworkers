# File Description: Firm-level employment analysis — DIGe-style lottery wins effect
# Author: Amy Kim
# Date Created: Mar 2026
#
# Replicates and extends Doren, Isen & Gelber's approach to estimating the causal
# effect of H-1B lottery wins on firm-level employment (as measured in Revelio).
#
# Estimating equation:
#   log(Y_{i,t,tau}) = beta_0 + beta_1 * U_{i,tau} + firm FE + lottery-year FE + eps
#
# Where U_{i,tau} = "unexpected wins" = actual_wins - expected_wins
#   expected_wins = win_rate_ADE_tau * n_ADE_apps + win_rate_nonADE_tau * n_nonADE_apps
#   (Conditional on application mix, wins are quasi-randomly assigned by the lottery)
#
# ADE/non-ADE split:
#   - For SELECTED applicants: ADE status from ade_lottery column in FOIA data (direct).
#   - For non-selected (ELIGIBLE/CREATED): ADE status is unavailable in the raw FOIA
#     extract. We proxy it using ade_ind from the merged Revelio-FOIA parquet, which
#     covers the matched subset of applicants (~209K). Unmatched applicants have unknown
#     ADE status and are treated as proportionally split (see build_hib_panel).
#
# Pipeline:
#   1. Build H-1B firm-year panel from FOIA parquet
#   2. Query Revelio headcounts from WRDS (3 versions, Jan 1 snapshot)
#   3. Merge to analysis panel; compute U_{i,tau} and log employment
#   4. Run PanelOLS regressions for each lag k and headcount version
#   5. Output LaTeX table and event study plot

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
# Only force the non-interactive Agg backend when not running inside IPython
# (e.g. a plain batch job).  In IPython the inline backend is already active
# and forcing Agg would suppress all inline plot display.
try:
    from IPython import get_ipython as _get_ipython
    _IN_IPYTHON = _get_ipython() is not None
except Exception:
    _IN_IPYTHON = False
if not _IN_IPYTHON:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
if "__file__" in globals():
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    _THIS_DIR = os.path.join(os.getcwd(), "04_analysis")

_CODE_DIR = os.path.dirname(_THIS_DIR)
sys.path.append(_CODE_DIR)

from config import root

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
_CFG_PATH = Path(_CODE_DIR) / "configs" / "firm_outcomes.yaml"


def _save_and_show(fig, path, dpi=150, **kwargs):
    """Save figure to *path*, display inline if in IPython, then close."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    if _IN_IPYTHON:
        plt.show()
    plt.close(fig)


def _load_config() -> dict:
    raw = yaml.safe_load(_CFG_PATH.read_text()) or {}
    # Expand {root} in all string values
    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            return obj.replace("{root}", str(root))
        return obj
    return _expand(raw)


def _normalize_fe_spec(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "").replace("-", "_")
    aliases = {
        "none": "none",
        "nofe": "none",
        "no_fe": "none",
        "firm": "firm_only",
        "entity": "firm_only",
        "firm_only": "firm_only",
        "firmonly": "firm_only",
        "year": "year_only",
        "time": "year_only",
        "year_only": "year_only",
        "yearonly": "year_only",
        "firm_year": "firm_year",
        "firmyear": "firm_year",
        "firm+year": "firm_year",
        "entity_time": "firm_year",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown FE spec '{value}'. Valid specs: none, firm_only, year_only, firm_year"
        )
    return aliases[key]


def _fe_label(fe_spec: str) -> str:
    labels = {
        "none": "None",
        "firm_only": "Firm FE",
        "year_only": "Lottery-year FE",
        "firm_year": "Firm FE + Lottery-year FE",
    }
    return labels.get(fe_spec, fe_spec)


def _assign_quartile(series: pd.Series, q: int = 4, labels=None) -> pd.Series:
    """
    Robust quantile assignment that tolerates ties/low variation.
    Returns categorical labels or NaN when quantiles cannot be formed.
    """
    if labels is None:
        labels = [f"Q{i}" for i in range(1, q + 1)]
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype="object")
    valid = s.notna()
    if valid.sum() < q:
        return out
    ranks = s[valid].rank(method="first")
    try:
        bins = pd.qcut(ranks, q=q, labels=labels, duplicates="drop")
    except Exception:
        return out
    out.loc[valid] = bins.astype(str)
    return out


CFG = _load_config()

TESTING         = CFG.get("testing", {})
TESTING_ENABLED = bool(TESTING.get("enabled", False))
TESTING_N_FIRMS = int(TESTING.get("n_firms", 200))
TESTING_SEED    = int(TESTING.get("seed", 42))

OUTPUT_DIR      = Path(CFG["output_dir"])
FOIA_PARQUET    = CFG["foia_parquet"]
FOIA_FIRM_MAP_CSV = CFG.get(
    "foia_firm_map_csv",
    str(Path(root) / "data" / "int" / "company_matching_jan28" / "foia_fein_to_firm.csv"),
)
FOIA_RCID_MAP_CSV = CFG.get(
    "foia_rcid_map_csv",
    str(Path(root) / "data" / "int" / "company_matching_jan28" / "llm_review_all_foia_to_rcid_crosswalk.csv"),
)
FOIA_GOOD_MATCH_IDS_CSV = CFG.get(
    "foia_good_match_ids_csv",
    str(Path(root) / "data" / "int" / "good_match_ids_mar20.csv"),
)
MERGE_PARQUET   = CFG.get("merge_parquet", "")
WRDS_USER       = CFG.get("wrds_username", "amykimecon")
CHUNK_SIZE      = int(CFG.get("wrds_chunk_size", 300))

SNAP_MONTH      = int(CFG.get("snapshot_month", 1))
SNAP_DAY        = int(CFG.get("snapshot_day", 1))
HC_YEAR_MIN     = int(CFG.get("headcount_year_min", 2020))
HC_YEAR_MAX     = int(CFG.get("headcount_year_max", 2028))
TENURE_MONTHS   = int(CFG.get("tenure_min_months", 6))
ENDDATE_FALLBACK = CFG.get("enddate_fallback", "2030-12-31")
H1B_OCC_FILTER  = CFG.get("h1b_occ_filter", {})
H1B_OCC_ENABLED = bool(H1B_OCC_FILTER.get("enabled", True))
H1B_OCC_CROSSWALK = H1B_OCC_FILTER.get(
    "foia_occ_crosswalk",
    str(Path(root) / "data" / "crosswalks" / "rev_occ_to_foia_freq.csv"),
)
_H1B_OCC_MAX_RANK_RAW = H1B_OCC_FILTER.get("max_rank", None)
if isinstance(_H1B_OCC_MAX_RANK_RAW, str) and _H1B_OCC_MAX_RANK_RAW.strip().lower() in {"", "none", "null"}:
    H1B_OCC_MAX_RANK = None
elif _H1B_OCC_MAX_RANK_RAW is None:
    H1B_OCC_MAX_RANK = None
else:
    H1B_OCC_MAX_RANK = float(_H1B_OCC_MAX_RANK_RAW)
H1B_OCC_MIN_SHARE = float(H1B_OCC_FILTER.get("min_share", 0.0))

K_LAGS          = CFG.get("k_lags", [-1, 0, 1, 2, 3])
MIN_APPS        = int(CFG.get("min_apps", 1))
_MAX_APPS_RAW   = CFG.get("max_apps", 500)
if _MAX_APPS_RAW is None or (
    isinstance(_MAX_APPS_RAW, str)
    and _MAX_APPS_RAW.strip().lower() in {"", "none", "null"}
):
    MAX_APPS = None
else:
    MAX_APPS = int(_MAX_APPS_RAW)
if MAX_APPS is not None and MAX_APPS < MIN_APPS:
    raise ValueError(f"max_apps ({MAX_APPS}) must be >= min_apps ({MIN_APPS})")
LOG_OUTCOMES    = bool(CFG.get("log_outcomes", True))
FORCE_REQUERY   = bool(CFG.get("force_requery", False))
MISSING_OUTCOME_POLICY = str(CFG.get("missing_outcome_policy", "zero_impute")).strip().lower()
if MISSING_OUTCOME_POLICY not in {"zero_impute", "balanced_panel"}:
    raise ValueError(
        f"Unknown missing_outcome_policy='{MISSING_OUTCOME_POLICY}'. "
        "Valid values: zero_impute, balanced_panel"
    )
PRINT_HETEROGENEITY_TABLES = bool(CFG.get("print_heterogeneity_tables", True))
HETEROGENEITY_OUTCOME = str(CFG.get("heterogeneity_outcome", "out_emp_all"))
_HETEROGENEITY_FE_RAW = CFG.get("heterogeneity_fe_spec", "firm_year")
HETEROGENEITY_FE_SPEC = _normalize_fe_spec(_HETEROGENEITY_FE_RAW)
_HETEROGENEITY_K_RAW = CFG.get("heterogeneity_k", None)
if _HETEROGENEITY_K_RAW is None or (isinstance(_HETEROGENEITY_K_RAW, str) and _HETEROGENEITY_K_RAW.strip().lower() in {"", "none", "null"}):
    HETEROGENEITY_K = None
else:
    HETEROGENEITY_K = int(_HETEROGENEITY_K_RAW)
TOP_SIZE_TRIM_PCT = float(CFG.get("top_size_trim_pct", 0.0))
_BASELINE_SIZE_YEAR_RAW = CFG.get("baseline_size_year", None)
if _BASELINE_SIZE_YEAR_RAW is None or (
    isinstance(_BASELINE_SIZE_YEAR_RAW, str)
    and _BASELINE_SIZE_YEAR_RAW.strip().lower() in {"", "none", "null", "auto"}
):
    BASELINE_SIZE_YEAR = None
else:
    BASELINE_SIZE_YEAR = int(_BASELINE_SIZE_YEAR_RAW)
BASELINE_SIZE_COL = str(CFG.get("baseline_size_col", "emp_all")).strip()
BASELINE_SIZE_DROP_MISSING = bool(CFG.get("baseline_size_drop_missing", False))
PLOT_SIZE_DECILE_EVENT_STUDY = bool(CFG.get("plot_size_decile_event_study", True))
SIZE_DECILE_EVENT_OUTCOME = str(CFG.get("size_decile_event_outcome", HETEROGENEITY_OUTCOME))
_SIZE_DECILE_EVENT_FE_RAW = CFG.get("size_decile_event_fe_spec", HETEROGENEITY_FE_SPEC)
SIZE_DECILE_EVENT_FE_SPEC = _normalize_fe_spec(_SIZE_DECILE_EVENT_FE_RAW)
DIAG_CFG = CFG.get("diagnostics", {}) or {}
DIAG_ENABLED = bool(DIAG_CFG.get("enabled", True))
DIAG_FOCUS_OUTCOME = str(DIAG_CFG.get("focus_outcome", "out_emp_all"))
DIAG_TRIM_QUANTILE = float(DIAG_CFG.get("trim_quantile", 0.99))
DIAG_COMPARE_POLICIES = bool(DIAG_CFG.get("compare_missing_policies", True))
DIAG_RUN_SENSITIVITY_GRID = bool(DIAG_CFG.get("run_sensitivity_grid", True))
DIAG_SCATTER_SAMPLE_N = int(DIAG_CFG.get("scatter_sample_n", 50000))
_FE_SPECS_RAW   = CFG.get("fe_specs", ["none", "firm_year"])
if isinstance(_FE_SPECS_RAW, str):
    _FE_SPECS_RAW = [_FE_SPECS_RAW]
FE_SPECS        = [_normalize_fe_spec(v) for v in _FE_SPECS_RAW]
if not FE_SPECS:
    FE_SPECS = ["none", "firm_year"]
FE_SPECS = list(dict.fromkeys(FE_SPECS))
_INDIV_RESTRICT_CFG        = CFG.get("indiv_sample_restrict", {}) or {}
INDIV_SAMPLE_RESTRICT_ENABLED  = bool(_INDIV_RESTRICT_CFG.get("enabled", False))
_INDIV_RESTRICT_PARQUET_RAW    = _INDIV_RESTRICT_CFG.get("merge_parquet", "")
# Fall back to top-level merge_parquet when the sub-key is empty/omitted
INDIV_SAMPLE_RESTRICT_PARQUET  = (
    _INDIV_RESTRICT_PARQUET_RAW.strip()
    if isinstance(_INDIV_RESTRICT_PARQUET_RAW, str) and _INDIV_RESTRICT_PARQUET_RAW.strip()
    else MERGE_PARQUET
)

STATUS_SEL      = CFG.get("status_selected", "SELECTED")
STATUS_ELIG     = CFG.get("status_eligible", "ELIGIBLE")

_PRE2021_CFG    = CFG.get("pre2021_lca_proxy", {}) or {}
PRE2021_ENABLED = bool(_PRE2021_CFG.get("enabled", False))
PRE2021_LCA_PARQUET   = _PRE2021_CFG.get("lca_parquet", "")
PRE2021_CROSSWALK_PARQUET = _PRE2021_CFG.get("lca_crosswalk_parquet", "")
PRE2021_MAX_YEAR  = int(_PRE2021_CFG.get("max_lottery_year", 2020))
PRE2021_PROXY_COL = str(_PRE2021_CFG.get("proxy_col", "n_lca_workers"))
# Optional pre-2021 FOIA parquet (built by build_pre2021_foia.py from TRK_12704 data).
# When present, these rows are concatenated with the main FOIA input so that the
# pre-2021 SELECTED split in build_hib_panel has non-zero data.
FOIA_PRE2021_PARQUET = str(CFG.get("foia_pre2021_parquet", ""))
# Optional firm-level foia_firm_uid → rcid crosswalk for pre-2021 TRK_12704 firms.
# Built by revelio_h1b_company_matching/build_trk_rcid_crosswalk.py.
# Pre-2021 rows have main_rcid=NaN; this crosswalk fills in Revelio IDs at the
# firm level so pre-2021 firms can be linked to WRDS headcounts.
PRE2021_RCID_CROSSWALK = str(CFG.get("pre2021_rcid_crosswalk", ""))

print("=== firm_outcomes.py ===")
print(f"run_tag : {CFG.get('run_tag', 'feb2026')}")
print(f"testing : {'ENABLED (n_firms=' + str(TESTING_N_FIRMS) + ')' if TESTING_ENABLED else 'disabled'}")
print(f"force_requery: {FORCE_REQUERY}")
print(f"foia_input: {FOIA_PARQUET}")
print(f"foia_rcid_map: {FOIA_RCID_MAP_CSV}")
print(f"foia_good_match_ids: {FOIA_GOOD_MATCH_IDS_CSV}")
print(f"k_lags  : {K_LAGS}")
if MAX_APPS is None:
    print(f"apps filter: n_apps >= {MIN_APPS}")
else:
    print(f"apps filter: {MIN_APPS} <= n_apps <= {MAX_APPS}")
print(f"log_outcomes: {LOG_OUTCOMES}")
print(f"missing_outcome_policy: {MISSING_OUTCOME_POLICY}")
print(f"fe_specs : {FE_SPECS}")
if TOP_SIZE_TRIM_PCT > 0:
    byear = "auto" if BASELINE_SIZE_YEAR is None else str(BASELINE_SIZE_YEAR)
    print(
        "baseline size trim: enabled "
        f"(top {TOP_SIZE_TRIM_PCT:.3f}% by {BASELINE_SIZE_COL} at year={byear}, "
        f"drop_missing={BASELINE_SIZE_DROP_MISSING})"
    )
else:
    print("baseline size trim: disabled")
if PRINT_HETEROGENEITY_TABLES:
    hk = "auto" if HETEROGENEITY_K is None else str(HETEROGENEITY_K)
    print(f"heterogeneity: enabled (outcome={HETEROGENEITY_OUTCOME}, fe={HETEROGENEITY_FE_SPEC}, k={hk})")
else:
    print("heterogeneity: disabled")
if PLOT_SIZE_DECILE_EVENT_STUDY:
    print(
        "size-decile event study: enabled "
        f"(outcome={SIZE_DECILE_EVENT_OUTCOME}, fe={SIZE_DECILE_EVENT_FE_SPEC})"
    )
else:
    print("size-decile event study: disabled")
if DIAG_ENABLED:
    print(
        "diagnostics: enabled "
        f"(focus_outcome={DIAG_FOCUS_OUTCOME}, trim_q={DIAG_TRIM_QUANTILE:.3f}, "
        f"compare_policies={DIAG_COMPARE_POLICIES}, sensitivity_grid={DIAG_RUN_SENSITIVITY_GRID})"
    )
else:
    print("diagnostics: disabled")
if H1B_OCC_ENABLED:
    rank_label = "all ranked" if H1B_OCC_MAX_RANK is None else f"rank<={H1B_OCC_MAX_RANK:g}"
    share_label = "share>0" if H1B_OCC_MIN_SHARE <= 0 else f"share>={H1B_OCC_MIN_SHARE:g}"
    print(f"h1b_occ : enabled (auto via FOIA occ rank/share; {rank_label}, {share_label})")
else:
    print("h1b_occ : disabled")
if PRE2021_ENABLED:
    print(
        f"pre2021_lca_proxy: enabled "
        f"(max_lottery_year={PRE2021_MAX_YEAR}, proxy_col={PRE2021_PROXY_COL})"
    )
else:
    print("pre2021_lca_proxy: disabled")
if FOIA_PRE2021_PARQUET:
    _pre21_exists = Path(FOIA_PRE2021_PARQUET).exists()
    print(f"foia_pre2021_parquet: {FOIA_PRE2021_PARQUET} ({'found' if _pre21_exists else 'NOT FOUND'})")
else:
    print("foia_pre2021_parquet: not configured")
if INDIV_SAMPLE_RESTRICT_ENABLED:
    _restrict_exists = Path(INDIV_SAMPLE_RESTRICT_PARQUET).exists() if INDIV_SAMPLE_RESTRICT_PARQUET else False
    print(
        f"indiv_sample_restrict: ENABLED  "
        f"(parquet={INDIV_SAMPLE_RESTRICT_PARQUET}, "
        f"{'found' if _restrict_exists else 'NOT FOUND'})"
    )
else:
    print("indiv_sample_restrict: disabled")
print()


###############################################################################
# STEP 1: H-1B FIRM-YEAR PANEL
###############################################################################

def _load_lca_proxy_panel(
    lca_path: str,
    crosswalk_path: str,
    proxy_col: str,
    max_year: int,
) -> pd.DataFrame:
    """
    Build a (foia_firm_uid, lottery_year, n_lca_apps) panel from LCA data.

    Joins lca_firm_year (employer_name x employer_state x fiscal_year) with
    lca_foia_crosswalk (lca_employer_name x lca_employer_state -> foia_firm_uid),
    filters to fiscal_year <= max_year, and aggregates to (foia_firm_uid, fiscal_year)
    summing proxy_col as n_lca_apps.

    Returns DataFrame: foia_firm_uid (str), lottery_year (int), n_lca_apps (float).
    """
    t0 = time.time()
    lca_path = Path(lca_path)
    xwalk_path = Path(crosswalk_path)

    if not lca_path.exists():
        raise FileNotFoundError(f"LCA firm-year parquet not found: {lca_path}")
    if not xwalk_path.exists():
        raise FileNotFoundError(f"LCA-FOIA crosswalk not found: {xwalk_path}")

    lca = pd.read_parquet(lca_path, columns=["employer_name", "employer_state", "fiscal_year", proxy_col])
    xwalk = pd.read_parquet(xwalk_path, columns=["lca_employer_name", "lca_employer_state", "foia_firm_uid"])

    print(f"  LCA proxy: loaded {len(lca):,} firm-year rows from {lca_path.name} ({time.time()-t0:.1f}s)")
    print(f"  LCA crosswalk: {len(xwalk):,} matched firm-name records")

    # Filter to pre-2021 years
    lca = lca[lca["fiscal_year"] <= max_year].copy()
    print(f"  LCA proxy: {len(lca):,} rows with fiscal_year <= {max_year}")

    # Join LCA to crosswalk on (employer_name, employer_state)
    lca = lca.rename(columns={"employer_name": "lca_employer_name", "employer_state": "lca_employer_state"})
    merged = lca.merge(
        xwalk[["lca_employer_name", "lca_employer_state", "foia_firm_uid"]],
        on=["lca_employer_name", "lca_employer_state"],
        how="left",
    )

    n_matched = merged["foia_firm_uid"].notna().sum()
    n_total   = len(merged)
    print(
        f"  LCA proxy: {n_matched:,} / {n_total:,} firm-year rows matched to foia_firm_uid "
        f"({n_matched/n_total*100:.1f}% by row count)"
    )

    merged = merged[merged["foia_firm_uid"].notna()].copy()
    merged["fiscal_year"] = merged["fiscal_year"].astype(int)
    merged[proxy_col] = pd.to_numeric(merged[proxy_col], errors="coerce").fillna(0)

    # Aggregate to (foia_firm_uid, lottery_year)
    proxy_panel = (
        merged.groupby(["foia_firm_uid", "fiscal_year"])[proxy_col]
        .sum()
        .reset_index()
        .rename(columns={"fiscal_year": "lottery_year", proxy_col: "n_lca_apps"})
    )
    proxy_panel["lottery_year"] = proxy_panel["lottery_year"].astype(int)

    print(
        f"  LCA proxy panel: {len(proxy_panel):,} firm-year rows, "
        f"{proxy_panel['foia_firm_uid'].nunique():,} unique firms "
        f"(fiscal years {proxy_panel['lottery_year'].min()}–{proxy_panel['lottery_year'].max()})"
    )
    return proxy_panel


def _load_ade_proxy(merge_path: str) -> pd.DataFrame:
    """
    Build a (foia_indiv_id → ade) lookup from the merged Revelio-FOIA parquet.

    Uses the same ADE definition as reg_new.py:
      ade = 1 if ade_ind==1 AND (ade_year is null OR ade_year <= lottery_year - 1)

    Returns a DataFrame with columns [foia_indiv_id, ade_proxy] deduplicated to
    one row per applicant (takes first value; ade_ind is constant per applicant).
    """
    if not merge_path or not Path(merge_path).exists():
        print("  [WARN] merge_parquet not found — ADE proxy unavailable, using pooled win rate")
        return pd.DataFrame(columns=["foia_indiv_id", "ade_proxy"])

    cols = ["foia_indiv_id", "ade_ind", "ade_year", "lottery_year"]
    df = pd.read_parquet(merge_path, columns=cols)
    df = df[["foia_indiv_id", "ade_ind", "ade_year", "lottery_year"]].drop_duplicates(
        subset="foia_indiv_id"
    )
    df["lottery_year"] = df["lottery_year"].astype(int)
    df["ade_proxy"] = np.where(
        df["ade_ind"].notna() & (df["ade_ind"] == 1),
        np.where(
            df["ade_year"].isna() | (df["ade_year"] <= df["lottery_year"] - 1),
            1, 0
        ),
        0
    )
    return df[["foia_indiv_id", "ade_proxy"]]


def _clean_fein_series(series: pd.Series) -> pd.Series:
    """Normalize FEIN-like strings to integer keys used by company-matching tables."""
    digits = series.fillna("").astype(str).str.replace(r"\D", "", regex=True)
    return pd.to_numeric(digits, errors="coerce").fillna(0).astype("int64")


def _clean_fein_series_nullable(series: pd.Series) -> pd.Series:
    """
    FEIN cleaner that preserves missing values (vs. coercing to 0).
    Used when reproducing rev_users_clean FEIN-year bridge logic.
    """
    digits = series.fillna("").astype(str).str.replace(r"\D", "", regex=True)
    digits = digits.replace("", np.nan)
    return pd.to_numeric(digits, errors="coerce")


def _normalize_foia_uid(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.replace({"": np.nan, "nan": np.nan, "None": np.nan})


def _read_tabular(path: Path, usecols: list | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=usecols)
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _load_rcid_map_simple(rcid_map_path: Path) -> pd.DataFrame:
    """Fallback loader for pre-collapsed foia_firm_uid x year -> rcid tables."""
    if rcid_map_path.suffix.lower() == ".parquet":
        rcmap = pd.read_parquet(rcid_map_path)
    else:
        rcid_cols = ["foia_firm_uid", "fein_year", "lottery_year", "rcid", "score", "confidence"]
        rcid_avail = pd.read_csv(rcid_map_path, nrows=1).columns.tolist()
        keep = [c for c in rcid_cols if c in rcid_avail]
        rcmap = pd.read_csv(rcid_map_path, usecols=keep, low_memory=False)

    if "fein_year" not in rcmap.columns and "lottery_year" in rcmap.columns:
        rcmap["fein_year"] = rcmap["lottery_year"]
    if not {"foia_firm_uid", "fein_year", "rcid"}.issubset(rcmap.columns):
        return pd.DataFrame(columns=["foia_firm_uid", "fein_year", "rcid"])

    rcmap = rcmap.copy()
    rcmap["foia_firm_uid"] = _normalize_foia_uid(rcmap["foia_firm_uid"])
    rcmap["fein_year"] = pd.to_numeric(rcmap["fein_year"], errors="coerce")
    rcmap["rcid"] = pd.to_numeric(rcmap["rcid"], errors="coerce")
    if "match_source" in rcmap.columns:
        rcmap["rcid_source"] = rcmap["match_source"].astype(str).str.strip()
    elif "source" in rcmap.columns:
        rcmap["rcid_source"] = rcmap["source"].astype(str).str.strip()
    else:
        rcmap["rcid_source"] = np.nan
    if "match_priority_bucket" in rcmap.columns:
        rcmap["rcid_match_priority"] = rcmap["match_priority_bucket"].astype(str).str.strip()
    else:
        rcmap["rcid_match_priority"] = np.nan
    if "match_priority_bucket" in rcmap.columns:
        bucket_rank = {"exact_llm": 1, "legacy_good": 2, "llm": 3}
        rcmap["category_priority"] = (
            rcmap["match_priority_bucket"].astype(str).str.strip().map(bucket_rank).fillna(9)
        )
    else:
        rcmap["category_priority"] = 9
    if "llm_match_score" in rcmap.columns:
        rcmap["score"] = pd.to_numeric(rcmap["llm_match_score"], errors="coerce")
    elif "score" in rcmap.columns:
        rcmap["score"] = pd.to_numeric(rcmap["score"], errors="coerce")
    else:
        rcmap["score"] = np.nan
    if "confidence" in rcmap.columns:
        rcmap["confidence"] = pd.to_numeric(rcmap["confidence"], errors="coerce")
    else:
        rcmap["confidence"] = np.nan

    rcmap = rcmap.dropna(subset=["foia_firm_uid", "fein_year", "rcid"]).copy()
    rcmap = rcmap.sort_values(
        ["foia_firm_uid", "fein_year", "category_priority", "score", "confidence", "rcid"],
        ascending=[True, True, True, False, False, True],
    )
    rcmap = rcmap.drop_duplicates(subset=["foia_firm_uid", "fein_year"])
    return rcmap[["foia_firm_uid", "fein_year", "rcid", "rcid_source", "rcid_match_priority"]]


def _load_rcid_map_rev_users_union(llm_crosswalk_path: Path, good_match_ids_path: str) -> pd.DataFrame:
    """
    Mirror rev_users_clean mapping logic:
      - start from llm_review_all_foia_to_rcid_crosswalk.csv
      - keep llm valid_match rows (exact=priority1, non-exact=priority3)
      - map legacy good_match_ids FEIN x year -> foia_firm_uid via llm bridge (priority2)
      - collapse to one best rcid per foia_firm_uid x fein_year
    """
    llm_cols = [
        "foia_firm_uid", "fein_clean", "fein_year", "rcid",
        "crosswalk_validity_label", "firm_status", "score", "confidence",
    ]
    llm = _read_tabular(llm_crosswalk_path, usecols=llm_cols).copy()
    llm["foia_firm_uid"] = _normalize_foia_uid(llm["foia_firm_uid"])
    llm["fein_clean"] = pd.to_numeric(llm["fein_clean"], errors="coerce")
    llm["fein_year"] = pd.to_numeric(llm["fein_year"], errors="coerce")
    llm["rcid"] = pd.to_numeric(llm["rcid"], errors="coerce")
    llm["crosswalk_validity_label"] = (
        llm["crosswalk_validity_label"].astype(str).str.strip().str.lower()
    )
    llm["firm_status"] = llm["firm_status"].astype(str).str.strip().str.lower()
    llm["score"] = pd.to_numeric(llm["score"], errors="coerce")
    llm["confidence"] = pd.to_numeric(llm["confidence"], errors="coerce")

    uid_bridge = (
        llm.dropna(subset=["foia_firm_uid", "fein_clean", "fein_year"])
        [["foia_firm_uid", "fein_clean", "fein_year"]]
        .drop_duplicates()
    )

    llm_keep = llm[
        llm["foia_firm_uid"].notna()
        & llm["fein_year"].notna()
        & llm["rcid"].notna()
        & (llm["crosswalk_validity_label"] == "valid_match")
    ][["foia_firm_uid", "fein_year", "rcid", "firm_status", "score", "confidence"]].copy()
    llm_keep["category_priority"] = np.where(llm_keep["firm_status"] == "exact match", 1, 3)
    llm_keep["rcid_source"] = "llm"
    llm_keep["rcid_match_priority"] = np.where(
        llm_keep["firm_status"] == "exact match", "exact_llm", "llm"
    )
    union_parts = [llm_keep]

    good_path = Path(good_match_ids_path)
    if good_path.exists():
        good_cols = pd.read_csv(good_path, nrows=1).columns.tolist()
        needed = {"FEIN", "lottery_year", "rcid"}
        if needed.issubset(good_cols):
            good = pd.read_csv(good_path, usecols=["FEIN", "lottery_year", "rcid"], low_memory=False)
            good["fein_clean"] = _clean_fein_series_nullable(good["FEIN"])
            good["fein_year"] = pd.to_numeric(good["lottery_year"], errors="coerce")
            good["rcid"] = pd.to_numeric(good["rcid"], errors="coerce")
            good = good.dropna(subset=["fein_clean", "fein_year", "rcid"]).copy()
            good = good.merge(uid_bridge, on=["fein_clean", "fein_year"], how="left")
            good = good.dropna(subset=["foia_firm_uid"])[["foia_firm_uid", "fein_year", "rcid"]].drop_duplicates()
            good["category_priority"] = 2
            good["score"] = np.nan
            good["confidence"] = np.nan
            good["rcid_source"] = "legacy_good"
            good["rcid_match_priority"] = "legacy_good"
            union_parts.append(good)
        else:
            print(f"  [WARN] {good_path} missing one of {sorted(needed)}; skipping legacy-good union")
    else:
        print(f"  [WARN] foia_good_match_ids_csv not found: {good_path}; using LLM matches only")

    union = pd.concat(union_parts, ignore_index=True) if union_parts else pd.DataFrame()
    if union.empty:
        return pd.DataFrame(columns=["foia_firm_uid", "fein_year", "rcid"])

    union = union.dropna(subset=["foia_firm_uid", "fein_year", "rcid"]).copy()
    union = union.sort_values(
        ["foia_firm_uid", "fein_year", "category_priority", "score", "confidence", "rcid"],
        ascending=[True, True, True, False, False, True],
    )
    best = union.drop_duplicates(subset=["foia_firm_uid", "fein_year"], keep="first").copy()
    src_counts = best["rcid_source"].value_counts().to_dict()
    print(
        "  RCID map (rev_users_union): "
        f"{best['foia_firm_uid'].nunique():,} firms, "
        f"{len(best):,} firm-year rows, "
        f"{best['rcid'].nunique():,} unique rcids, "
        f"sources={src_counts}"
    )
    return best[["foia_firm_uid", "fein_year", "rcid", "rcid_source", "rcid_match_priority"]]


def _load_rcid_map_for_raw_foia() -> pd.DataFrame:
    """Load foia_firm_uid x year -> rcid map, preferring rev_users-style union logic when possible."""
    rcid_map_path = Path(FOIA_RCID_MAP_CSV)
    if not rcid_map_path.exists():
        print(f"  [WARN] foia_rcid_map_csv not found: {rcid_map_path} (main_rcid will be missing)")
        return pd.DataFrame(columns=["foia_firm_uid", "fein_year", "rcid", "rcid_source", "rcid_match_priority"])

    if rcid_map_path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
            cols = set(pq.ParquetFile(rcid_map_path).schema.names)
        except Exception:
            cols = set(pd.read_parquet(rcid_map_path).columns)
    else:
        cols = set(pd.read_csv(rcid_map_path, nrows=1).columns.tolist())

    # Prebuilt firm-year mapping output from rev_users_clean (company_merge_sample).
    if {"foia_firm_uid", "lottery_year", "rcid"}.issubset(cols):
        rcmap = _load_rcid_map_simple(rcid_map_path)
        print(
            "  RCID map (prebuilt firm-year): "
            f"{rcmap['foia_firm_uid'].nunique():,} firms, "
            f"{len(rcmap):,} firm-year rows, "
            f"{rcmap['rcid'].nunique():,} unique rcids"
        )
        return rcmap

    # LLM full crosswalk with labels -> replicate rev_users_clean union behavior.
    if {"foia_firm_uid", "fein_clean", "fein_year", "rcid", "crosswalk_validity_label"}.issubset(cols):
        return _load_rcid_map_rev_users_union(rcid_map_path, FOIA_GOOD_MATCH_IDS_CSV)

    # Fallback: plain pre-collapsed map.
    rcmap = _load_rcid_map_simple(rcid_map_path)
    print(
        "  RCID map (simple): "
        f"{rcmap['foia_firm_uid'].nunique():,} firms, "
        f"{len(rcmap):,} firm-year rows, "
        f"{rcmap['rcid'].nunique():,} unique rcids"
    )
    return rcmap


def _load_foia_core(foia_path: str) -> pd.DataFrame:
    """
    Load FOIA data into a standard schema required by build_hib_panel:
      foia_indiv_id, foia_firm_uid, lottery_year, status_type, main_rcid, ade_lottery

    Supports either:
      1) Clean parquet already in this schema
      2) Raw FOIA Bloomberg CSV (~1.8M rows), mapped to firm/rcid using company matching tables
    """
    path = Path(foia_path)
    req_cols = ["foia_indiv_id", "foia_firm_uid", "lottery_year", "status_type", "main_rcid", "ade_lottery"]
    opt_cols = ["industry", "NAICS4", "NAICS_CODE"]

    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
            cols_available = set(pq.ParquetFile(path).schema.names)
        except Exception:
            # Fallback if pyarrow metadata access is unavailable.
            cols_available = set(pd.read_parquet(path).columns)
        if set(req_cols).issubset(cols_available):
            keep_cols = req_cols + [c for c in opt_cols if c in cols_available]
            return pd.read_parquet(path, columns=keep_cols)

        raise ValueError(
            f"{path} is parquet but missing required columns for firm_outcomes: "
            f"{sorted(set(req_cols) - cols_available)}"
        )

    if path.suffix.lower() in {".csv", ".txt"}:
        # Full FOIA raw sample path (expected ~1.8M apps)
        csv_cols = ["foia_unique_id", "FEIN", "lottery_year", "status_type", "NAICS_CODE"]
        raw = pd.read_csv(path, usecols=csv_cols, low_memory=False)
        raw = raw.rename(columns={"foia_unique_id": "foia_indiv_id"}).copy()
        raw["foia_indiv_id"] = pd.to_numeric(raw["foia_indiv_id"], errors="coerce")
        raw = raw[raw["foia_indiv_id"].notna()].copy()
        raw["foia_indiv_id"] = raw["foia_indiv_id"].astype("int64")

        # Build firm key from FEIN x lottery_year
        raw["lottery_year"] = pd.to_numeric(raw["lottery_year"], errors="coerce")
        raw = raw[raw["lottery_year"].notna()].copy()
        raw["lottery_year"] = raw["lottery_year"].round().astype("int64")
        raw["fein_clean"] = _clean_fein_series(raw["FEIN"])
        raw["fein_year"] = raw["lottery_year"].astype("int64")

        firm_map_path = Path(FOIA_FIRM_MAP_CSV)
        if not firm_map_path.exists():
            raise FileNotFoundError(
                f"Raw FOIA CSV selected, but foia_firm_map_csv not found: {firm_map_path}"
            )
        firm_map = pd.read_csv(firm_map_path, usecols=["fein_clean", "fein_year", "foia_firm_uid"])
        firm_map = firm_map.drop_duplicates(subset=["fein_clean", "fein_year"])
        raw = raw.merge(firm_map, on=["fein_clean", "fein_year"], how="left")

        rcmap = _load_rcid_map_for_raw_foia()
        if not rcmap.empty:
            raw = raw.merge(rcmap, on=["foia_firm_uid", "fein_year"], how="left")
        else:
            raw["rcid"] = np.nan

        raw = raw.drop_duplicates(subset=["foia_indiv_id"]).copy()
        raw["main_rcid"] = raw["rcid"]
        raw["ade_lottery"] = np.nan
        out_cols = ["foia_indiv_id", "foia_firm_uid", "lottery_year", "status_type", "main_rcid", "ade_lottery"]
        if "NAICS_CODE" in raw.columns:
            out_cols.append("NAICS_CODE")
        out = raw[out_cols].copy()
        return out

    raise ValueError(f"Unsupported foia input file type: {path.suffix} ({path})")


def build_hib_panel(foia_path: str, merge_path: str = "", return_stats: bool = False):
    """
    Aggregate FOIA individual data to firm-year level.

    ADE/non-ADE split:
      - For SELECTED applicants: uses ade_lottery from FOIA (direct measure).
      - For non-selected (ELIGIBLE/CREATED): uses ade_proxy from merged parquet
        where available; remaining unknown applicants are allocated proportionally
        to the firm's known ADE share.

    Pre-2021 lottery years (when pre2021_lca_proxy.enabled=true):
      - FOIA only records SELECTED (winners) before FY2021 electronic registration.
      - n_apps is proxied by n_lca_workers from certified LCA filings, matched to
        foia_firm_uid via lca_foia_crosswalk.parquet.
      - ADE split for the denominator is fully unknown (LCA doesn't record ADE
        eligibility); national ADE share is used for all pre-2021 firms.

    Returns one row per (foia_firm_uid, lottery_year) with:
      - n_apps, n_wins           : total and winning applications
      - n_ade_apps, n_nonADE_apps: ADE/non-ADE application counts (estimated)
      - n_ade_wins, n_nonADE_wins: ADE/non-ADE win counts
      - U_itau                   : unexpected wins (actual - expected)
      - rcid                     : Revelio company ID
    """
    t0 = time.time()
    stats = {}
    print("Step 1: Building H-1B firm-year panel from FOIA data...")

    df = _load_foia_core(foia_path)
    print(f"  Loaded {len(df):,} rows from FOIA input ({time.time()-t0:.1f}s)")

    # Optionally prepend pre-2021 TRK_12704 petition rows (built by build_pre2021_foia.py).
    # These have status_type="SELECTED" and lottery_year <= PRE2021_MAX_YEAR, so they
    # will feed the pre-2021 LCA-proxy branch of build_hib_panel.
    if FOIA_PRE2021_PARQUET and Path(FOIA_PRE2021_PARQUET).exists():
        _req = ["foia_indiv_id", "foia_firm_uid", "lottery_year", "status_type", "main_rcid", "ade_lottery"]
        pre21_df = pd.read_parquet(FOIA_PRE2021_PARQUET, columns=_req)
        df = pd.concat([pre21_df, df], ignore_index=True)
        print(
            f"  Pre-2021 FOIA rows appended: {len(pre21_df):,} "
            f"(total now {len(df):,} rows)"
        )

        # Backfill main_rcid for pre-2021 rows using the firm-level crosswalk
        # (pre2021_foia.parquet has main_rcid=NaN for all rows by construction).
        if PRE2021_RCID_CROSSWALK and Path(PRE2021_RCID_CROSSWALK).exists():
            _cw = pd.read_parquet(PRE2021_RCID_CROSSWALK, columns=["foia_firm_uid", "rcid"])
            _cw = _cw.rename(columns={"rcid": "_pre21_rcid"})
            _cw["_pre21_rcid"] = pd.to_numeric(_cw["_pre21_rcid"], errors="coerce")
            df = df.merge(_cw, on="foia_firm_uid", how="left")
            _pre_year = pd.to_numeric(df["lottery_year"], errors="coerce")
            _pre_mask = _pre_year <= PRE2021_MAX_YEAR
            _fill_mask = _pre_mask & df["main_rcid"].isna() & df["_pre21_rcid"].notna()
            df.loc[_fill_mask, "main_rcid"] = df.loc[_fill_mask, "_pre21_rcid"]
            df = df.drop(columns=["_pre21_rcid"])
            n_filled = int(_fill_mask.sum())
            n_pre21_rows = int(_pre_mask.sum())
            print(
                f"  Pre-2021 rcid backfill: {n_filled:,} / {n_pre21_rows:,} rows filled "
                f"({n_filled / max(n_pre21_rows, 1) * 100:.1f}%)"
            )
        else:
            print("  [INFO] pre2021_rcid_crosswalk not configured or not found; pre-2021 rcid will be NaN")

    stats["foia_rows_loaded"] = int(len(df))
    stats["foia_unique_apps_loaded"] = int(df["foia_indiv_id"].nunique())
    stats["foia_unique_firms_loaded"] = int(df["foia_firm_uid"].nunique())
    stats["foia_rows_missing_firm_uid"] = int(df["foia_firm_uid"].isna().sum())
    df_year = pd.to_numeric(df["lottery_year"], errors="coerce")
    stats["audit_foia_loaded_by_year"] = (
        df.assign(_lottery_year=df_year)
        .dropna(subset=["_lottery_year"])
        .groupby("_lottery_year")
        .agg(
            n_rows=("foia_indiv_id", "size"),
            n_apps=("foia_indiv_id", "nunique"),
            n_firms=("foia_firm_uid", "nunique"),
            n_rows_missing_firm_uid=("foia_firm_uid", lambda s: int(s.isna().sum())),
        )
        .reset_index()
        .rename(columns={"_lottery_year": "lottery_year"})
        .astype({"lottery_year": int})
        .sort_values("lottery_year")
    )
    if stats["foia_rows_missing_firm_uid"] > 0:
        print(f"  [WARN] rows with missing foia_firm_uid: {stats['foia_rows_missing_firm_uid']:,}")

    # If pre-2021 LCA proxy is enabled, extract pre-2021 SELECTED rows separately and
    # restrict the main pool to lottery_year > PRE2021_MAX_YEAR.  Before FY2021 there is
    # no ELIGIBLE/CREATED pool, so keeping those rows in the main path would set
    # n_apps = n_wins (all observed rows are winners), producing U_itau ≈ 0 everywhere.
    _df_year_num = pd.to_numeric(df["lottery_year"], errors="coerce")
    if PRE2021_ENABLED:
        _pre_mask   = (_df_year_num <= PRE2021_MAX_YEAR) & (df["status_type"] == STATUS_SEL)
        df_pre21    = df[_pre_mask].copy()
        df_post21   = df[_df_year_num > PRE2021_MAX_YEAR].copy()
        print(
            f"  Pre-{PRE2021_MAX_YEAR + 1} SELECTED rows for LCA proxy: {len(df_pre21):,}  |  "
            f"Post-{PRE2021_MAX_YEAR} rows (main path): {len(df_post21):,}"
        )
    else:
        df_post21 = df
        df_pre21  = None

    # Restrict to lottery-pool participants.
    # In FY2022+, non-selected registrants have status ELIGIBLE.
    # In FY2021, the same group has status CREATED (the label changed between years).
    STATUS_CREATED = "CREATED"
    pool = df_post21[df_post21["status_type"].isin([STATUS_SEL, STATUS_ELIG, STATUS_CREATED])].copy()
    print(f"  Lottery-pool rows (SELECTED + ELIGIBLE + CREATED): {len(pool):,}")
    stats["lottery_pool_rows"] = int(len(pool))
    stats["lottery_pool_unique_apps"] = int(pool["foia_indiv_id"].nunique())
    stats["lottery_pool_unique_firms"] = int(pool["foia_firm_uid"].nunique())
    pool_year = pd.to_numeric(pool["lottery_year"], errors="coerce")
    stats["audit_lottery_pool_by_year"] = (
        pool.assign(_lottery_year=pool_year)
        .dropna(subset=["_lottery_year"])
        .groupby("_lottery_year")
        .agg(
            n_rows=("foia_indiv_id", "size"),
            n_apps=("foia_indiv_id", "nunique"),
            n_firms=("foia_firm_uid", "nunique"),
        )
        .reset_index()
        .rename(columns={"_lottery_year": "lottery_year"})
        .astype({"lottery_year": int})
        .sort_values("lottery_year")
    )
    pool["main_rcid"] = pd.to_numeric(pool["main_rcid"], errors="coerce")
    # Industry group (used for heterogeneity tables).
    if "industry" in pool.columns and pool["industry"].notna().any():
        pool["industry_group"] = pool["industry"].astype(str).str.strip()
        pool["industry_group"] = pool["industry_group"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    elif "NAICS4" in pool.columns and pool["NAICS4"].notna().any():
        naics2 = pool["NAICS4"].astype(str).str.extract(r"(\d{2})", expand=False)
        pool["industry_group"] = np.where(naics2.notna(), "NAICS " + naics2, np.nan)
    elif "NAICS_CODE" in pool.columns and pool["NAICS_CODE"].notna().any():
        naics2 = pool["NAICS_CODE"].astype(str).str.extract(r"(\d{2})", expand=False)
        pool["industry_group"] = np.where(naics2.notna(), "NAICS " + naics2, np.nan)
    else:
        pool["industry_group"] = np.nan

    pool["lottery_year"] = pool["lottery_year"].astype(str)
    pool["is_selected"] = (pool["status_type"] == STATUS_SEL).astype(int)

    # --- ADE status resolution ---
    # ADE status = whether an applicant is eligible for the advanced-degree cap.
    # Primary source: ade_proxy from merged parquet (based on ade_ind in FOIA application).
    # Supplement: anyone who won via the ADE pool (ade_lottery==1) is definitively ADE-eligible
    #   even if not in the merged data.
    # NOTE: ade_lottery==0 means "won via regular pool" — it does NOT mean non-ADE, because
    #   ADE-eligible applicants can also win in round 1 (regular pool).
    ade_proxy = _load_ade_proxy(merge_path)
    n_proxy = len(ade_proxy)
    print(f"  ADE proxy: {n_proxy:,} applicants from merged parquet")

    pool = pool.merge(ade_proxy, on="foia_indiv_id", how="left")

    # ADE status: use proxy as primary; supplement with ade_lottery==1 for unmatched winners
    pool["ade_status"] = pool["ade_proxy"]
    pool.loc[pool["ade_proxy"].isna() & (pool["ade_lottery"] == 1), "ade_status"] = 1
    # ade_status: 1 = ADE-eligible, 0 = non-ADE, NaN = unknown

    n_known = pool["ade_status"].notna().sum()
    n_total = len(pool)
    print(f"  ADE known: {n_known:,} / {n_total:,} applicants ({n_known/n_total*100:.1f}%)")

    # Indicator columns for aggregation (NaN = unknown → not counted)
    pool["is_ade_app"]    = np.where(pool["ade_status"] == 1, 1.0, np.where(pool["ade_status"] == 0, 0.0, np.nan))
    pool["is_nonADE_app"] = np.where(pool["ade_status"] == 0, 1.0, np.where(pool["ade_status"] == 1, 0.0, np.nan))
    # Wins among ADE/nonADE applicants with known status
    pool["is_ade_win"]    = np.where(pool["is_selected"] == 1, pool["is_ade_app"],    np.nan)
    pool["is_nonADE_win"] = np.where(pool["is_selected"] == 1, pool["is_nonADE_app"], np.nan)

    # --- Firm-year aggregation ---
    grp = pool.groupby(["foia_firm_uid", "lottery_year"])
    panel = grp.agg(
        n_apps           =("is_selected",    "count"),
        n_wins           =("is_selected",    "sum"),
        n_ade_apps_obs   =("is_ade_app",     "sum"),   # known-ADE apps (NaN → 0 via sum)
        n_nonADE_apps_obs=("is_nonADE_app",  "sum"),   # known-nonADE apps
        n_ade_wins_obs   =("is_ade_win",     "sum"),   # wins among known-ADE apps
        n_nonADE_wins_obs=("is_nonADE_win",  "sum"),   # wins among known-nonADE apps
    ).reset_index()

    # Impute unknown-ADE apps by scaling up proportionally to firm's known ADE share.
    # For firms with no known-ADE info, fall back to national ADE share (computed later).
    panel["n_ade_apps_obs"]    = panel["n_ade_apps_obs"].fillna(0)
    panel["n_nonADE_apps_obs"] = panel["n_nonADE_apps_obs"].fillna(0)
    panel["n_known_apps"]      = panel["n_ade_apps_obs"] + panel["n_nonADE_apps_obs"]
    panel["n_unknown_apps"]    = panel["n_apps"] - panel["n_known_apps"]

    # Firm-level ADE share among known apps (NaN if no known apps)
    panel["firm_ade_share"] = np.where(
        panel["n_known_apps"] > 0,
        panel["n_ade_apps_obs"] / panel["n_known_apps"],
        np.nan
    )

    # Most common rcid per firm
    rcid_mode = (
        pool.dropna(subset=["main_rcid"])
        .groupby("foia_firm_uid")["main_rcid"]
        .agg(lambda x: int(x.mode().iloc[0]) if len(x) > 0 else np.nan)
        .reset_index()
        .rename(columns={"main_rcid": "rcid"})
    )
    panel = panel.merge(rcid_mode, on="foia_firm_uid", how="left")
    if "rcid_source" in pool.columns:
        src_mode = (
            pool.dropna(subset=["rcid_source"])
            .groupby(["foia_firm_uid", "lottery_year"])["rcid_source"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
            .reset_index()
        )
        panel = panel.merge(src_mode, on=["foia_firm_uid", "lottery_year"], how="left")
    if "rcid_match_priority" in pool.columns:
        pri_mode = (
            pool.dropna(subset=["rcid_match_priority"])
            .groupby(["foia_firm_uid", "lottery_year"])["rcid_match_priority"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
            .reset_index()
        )
        panel = panel.merge(pri_mode, on=["foia_firm_uid", "lottery_year"], how="left")
    # Most common industry group per firm-year
    ind_mode = (
        pool.dropna(subset=["industry_group"])
        .groupby(["foia_firm_uid", "lottery_year"])["industry_group"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
        .reset_index()
    )
    panel = panel.merge(ind_mode, on=["foia_firm_uid", "lottery_year"], how="left")

    # --- Pre-2021 sub-panel (LCA-proxied application denominator) ---
    # Firms with certified LCA filings in pre-2021 years are included as observations.
    # n_apps = n_lca_workers (LCA proxy); n_wins from FOIA SELECTED records.
    # Firms in LCA but with zero FOIA wins are valid all-loser observations (U_itau < 0).
    # Firms in FOIA SELECTED but not matched in the LCA crosswalk are dropped (no denominator).
    if PRE2021_ENABLED and df_pre21 is not None and len(df_pre21) > 0:
        print(f"  Building pre-{PRE2021_MAX_YEAR + 1} sub-panel using LCA proxy ({PRE2021_PROXY_COL})...")

        pre21 = df_pre21.copy()
        pre21["main_rcid"] = pd.to_numeric(pre21["main_rcid"], errors="coerce")

        # Industry group (same logic as main pool above)
        if "industry" in pre21.columns and pre21["industry"].notna().any():
            pre21["industry_group"] = pre21["industry"].astype(str).str.strip().replace(
                {"": np.nan, "nan": np.nan, "None": np.nan}
            )
        elif "NAICS4" in pre21.columns and pre21["NAICS4"].notna().any():
            _n2 = pre21["NAICS4"].astype(str).str.extract(r"(\d{2})", expand=False)
            pre21["industry_group"] = np.where(_n2.notna(), "NAICS " + _n2, np.nan)
        elif "NAICS_CODE" in pre21.columns and pre21["NAICS_CODE"].notna().any():
            _n2 = pre21["NAICS_CODE"].astype(str).str.extract(r"(\d{2})", expand=False)
            pre21["industry_group"] = np.where(_n2.notna(), "NAICS " + _n2, np.nan)
        else:
            pre21["industry_group"] = np.nan

        pre21["lottery_year"] = pre21["lottery_year"].astype(str)

        # ADE status for pre-2021 winners (reuse ade_proxy already loaded above)
        pre21 = pre21.merge(ade_proxy, on="foia_indiv_id", how="left")
        pre21["ade_status"] = pre21["ade_proxy"]
        pre21.loc[pre21["ade_proxy"].isna() & (pre21["ade_lottery"] == 1), "ade_status"] = 1
        pre21["is_ade_win"]    = np.where(pre21["ade_status"] == 1, 1.0,
                                  np.where(pre21["ade_status"] == 0, 0.0, np.nan))
        pre21["is_nonADE_win"] = np.where(pre21["ade_status"] == 0, 1.0,
                                  np.where(pre21["ade_status"] == 1, 0.0, np.nan))

        # Aggregate wins by firm-year
        pre21_wins = (
            pre21.groupby(["foia_firm_uid", "lottery_year"])
            .agg(
                n_wins           =("foia_indiv_id",  "count"),
                n_ade_wins_obs   =("is_ade_win",     "sum"),
                n_nonADE_wins_obs=("is_nonADE_win",  "sum"),
            )
            .reset_index()
        )
        pre21_wins["n_ade_wins_obs"]    = pre21_wins["n_ade_wins_obs"].fillna(0)
        pre21_wins["n_nonADE_wins_obs"] = pre21_wins["n_nonADE_wins_obs"].fillna(0)

        # rcid mode per firm (across pre-2021 SELECTED rows, firm-level not firm-year-level)
        pre21_rcid = (
            pre21.dropna(subset=["main_rcid"])
            .groupby("foia_firm_uid")["main_rcid"]
            .agg(lambda x: int(x.mode().iloc[0]) if len(x) > 0 else np.nan)
            .reset_index()
            .rename(columns={"main_rcid": "rcid"})
        )

        # Industry group mode per firm-year
        pre21_ind = (
            pre21.dropna(subset=["industry_group"])
            .groupby(["foia_firm_uid", "lottery_year"])["industry_group"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
            .reset_index()
        )

        # Load LCA proxy panel and convert lottery_year to string to match panel dtype
        lca_proxy = _load_lca_proxy_panel(
            PRE2021_LCA_PARQUET, PRE2021_CROSSWALK_PARQUET, PRE2021_PROXY_COL, PRE2021_MAX_YEAR
        )
        lca_proxy["lottery_year"] = lca_proxy["lottery_year"].astype(str)

        # Left-join wins onto LCA proxy (LCA is left: pure-loser firms get n_wins=0)
        panel_pre21 = lca_proxy.merge(pre21_wins, on=["foia_firm_uid", "lottery_year"], how="left")

        # Report FOIA firms that couldn't be matched to LCA (no denominator → excluded)
        _foia_firms = set(pre21_wins["foia_firm_uid"].unique())
        _lca_firms  = set(panel_pre21[panel_pre21["n_wins"].notna() & (panel_pre21["n_wins"] > 0)]["foia_firm_uid"].unique())
        _unmatched  = len(_foia_firms - _lca_firms)
        if _unmatched > 0:
            print(
                f"  [WARN] {_unmatched} pre-2021 FOIA-selected firms not in LCA crosswalk — "
                "dropped (no LCA denominator available)"
            )

        panel_pre21["n_wins"]            = panel_pre21["n_wins"].fillna(0).astype(float)
        panel_pre21["n_ade_wins_obs"]    = panel_pre21["n_ade_wins_obs"].fillna(0)
        panel_pre21["n_nonADE_wins_obs"] = panel_pre21["n_nonADE_wins_obs"].fillna(0)

        # n_apps = LCA proxy; ADE split is entirely unknown (LCA doesn't record ADE eligibility)
        panel_pre21["n_apps"]            = panel_pre21["n_lca_apps"]
        panel_pre21["n_ade_apps_obs"]    = 0.0
        panel_pre21["n_nonADE_apps_obs"] = 0.0
        panel_pre21["n_known_apps"]      = 0.0
        panel_pre21["n_unknown_apps"]    = panel_pre21["n_lca_apps"]
        panel_pre21["firm_ade_share"]    = np.nan   # national ADE share will be used downstream

        panel_pre21 = panel_pre21.merge(pre21_rcid, on="foia_firm_uid", how="left")
        panel_pre21 = panel_pre21.merge(pre21_ind,  on=["foia_firm_uid", "lottery_year"], how="left")
        panel_pre21 = panel_pre21.drop(columns=["n_lca_apps"])

        # Align columns to match the post-2021 panel (add optional cols as NaN)
        for _col in panel.columns:
            if _col not in panel_pre21.columns:
                panel_pre21[_col] = np.nan
        panel_pre21 = panel_pre21[panel.columns].copy()

        print(
            f"  Pre-2021 sub-panel: {len(panel_pre21):,} firm-year rows, "
            f"{panel_pre21['foia_firm_uid'].nunique():,} unique firms, "
            f"n_wins total={panel_pre21['n_wins'].sum():,.0f}, "
            f"n_apps (LCA) total={panel_pre21['n_apps'].sum():,.0f}"
        )

        panel = pd.concat([panel_pre21, panel], ignore_index=True)
        print(
            f"  Combined panel: {len(panel):,} firm-year rows "
            f"({panel['lottery_year'].nunique()} lottery years)"
        )

    stats["panel_firm_years_pre_test"] = int(len(panel))
    stats["panel_firms_pre_test"] = int(panel["foia_firm_uid"].nunique())

    # Apply testing subsample
    if TESTING_ENABLED:
        rng = np.random.default_rng(TESTING_SEED)
        firms = panel["foia_firm_uid"].unique()
        sampled = rng.choice(firms, size=min(TESTING_N_FIRMS, len(firms)), replace=False)
        panel = panel[panel["foia_firm_uid"].isin(sampled)].copy()
        print(f"  [TEST] Subsampled to {panel['foia_firm_uid'].nunique():,} firms")
    stats["panel_firm_years_post_test"] = int(len(panel))
    stats["panel_firms_post_test"] = int(panel["foia_firm_uid"].nunique())

    # --- National win rates by ADE/non-ADE ---
    # Win counts are direct from FOIA (ade_lottery column), reliable.
    # App counts: use observed counts + imputed unknowns via national ADE share.
    yr_grp = panel.groupby("lottery_year")
    yr_stats = yr_grp.agg(
        total_apps           =("n_apps",             "sum"),
        total_wins           =("n_wins",              "sum"),
        total_ade_apps_obs   =("n_ade_apps_obs",      "sum"),
        total_nonADE_apps_obs=("n_nonADE_apps_obs",   "sum"),
        total_unknown_apps   =("n_unknown_apps",      "sum"),
        total_ade_wins_obs   =("n_ade_wins_obs",      "sum"),
        total_nonADE_wins_obs=("n_nonADE_wins_obs",   "sum"),
    ).reset_index()

    # National ADE share (for imputing unknown apps)
    yr_stats["nat_known_apps"]  = yr_stats["total_ade_apps_obs"] + yr_stats["total_nonADE_apps_obs"]
    yr_stats["nat_ade_share"]   = np.where(
        yr_stats["nat_known_apps"] > 0,
        yr_stats["total_ade_apps_obs"] / yr_stats["nat_known_apps"],
        0.5
    )
    # Total ADE/nonADE apps (observed + imputed unknowns)
    yr_stats["total_ade_apps"]    = (yr_stats["total_ade_apps_obs"]
                                     + yr_stats["total_unknown_apps"] * yr_stats["nat_ade_share"])
    yr_stats["total_nonADE_apps"] = (yr_stats["total_nonADE_apps_obs"]
                                     + yr_stats["total_unknown_apps"] * (1 - yr_stats["nat_ade_share"]))

    # Win rates
    # Scale observed wins proportionally: unmatched winners are allocated to ADE/nonADE
    # using the observed ADE share among known-status winners.
    yr_stats["obs_wins"]           = yr_stats["total_ade_wins_obs"] + yr_stats["total_nonADE_wins_obs"]
    yr_stats["win_ade_share"]      = np.where(
        yr_stats["obs_wins"] > 0,
        yr_stats["total_ade_wins_obs"] / yr_stats["obs_wins"],
        yr_stats["nat_ade_share"]
    )
    unknown_wins = yr_stats["total_wins"] - yr_stats["obs_wins"]
    yr_stats["total_ade_wins"]    = yr_stats["total_ade_wins_obs"]    + unknown_wins * yr_stats["win_ade_share"]
    yr_stats["total_nonADE_wins"] = yr_stats["total_nonADE_wins_obs"] + unknown_wins * (1 - yr_stats["win_ade_share"])

    yr_stats["win_rate_ADE"]    = yr_stats["total_ade_wins"]    / yr_stats["total_ade_apps"]
    yr_stats["win_rate_nonADE"] = yr_stats["total_nonADE_wins"] / yr_stats["total_nonADE_apps"]
    yr_stats["win_rate_pooled"] = yr_stats["total_wins"]        / yr_stats["total_apps"]

    panel = panel.merge(
        yr_stats[["lottery_year", "win_rate_ADE", "win_rate_nonADE",
                  "win_rate_pooled", "nat_ade_share", "win_ade_share"]],
        on="lottery_year", how="left"
    )

    # Firm-level ADE app counts (use firm share if available, else national share)
    panel["ade_share"]         = panel["firm_ade_share"].fillna(panel["nat_ade_share"])
    panel["n_ade_apps_est"]    = (panel["n_ade_apps_obs"]
                                   + panel["n_unknown_apps"] * panel["ade_share"])
    panel["n_nonADE_apps_est"] = (panel["n_nonADE_apps_obs"]
                                   + panel["n_unknown_apps"] * (1 - panel["ade_share"]))

    # U_{i,tau} = actual wins - expected wins (positive = lucky firm)
    panel["expected_wins"] = (panel["win_rate_ADE"]    * panel["n_ade_apps_est"]
                              + panel["win_rate_nonADE"] * panel["n_nonADE_apps_est"])
    panel["U_itau"] = panel["n_wins"] - panel["expected_wins"]
    known_win_denom = panel["n_ade_wins_obs"] + panel["n_nonADE_wins_obs"]
    panel["share_ade_wins_obs"] = np.where(
        known_win_denom > 0,
        panel["n_ade_wins_obs"] / known_win_denom,
        np.nan,
    )
    panel["share_ade_wins_est"] = panel["share_ade_wins_obs"].fillna(panel["win_ade_share"])

    # Filter by application-count thresholds
    panel_pre_app = panel.copy()
    app_mask = panel["n_apps"] >= MIN_APPS
    if MAX_APPS is not None:
        app_mask &= panel["n_apps"] <= MAX_APPS
    panel = panel[app_mask].copy()
    stats["panel_firm_years_post_min_apps"] = int(len(panel))
    stats["panel_firms_post_min_apps"] = int(panel["foia_firm_uid"].nunique())
    stats["panel_rcid_nonmissing_post_min_apps"] = int(panel["rcid"].notna().sum())
    stats["panel_rcid_missing_post_min_apps"] = int(panel["rcid"].isna().sum())
    stats["panel_firm_years_dropped_below_min_apps"] = int((panel_pre_app["n_apps"] < MIN_APPS).sum())
    if MAX_APPS is not None:
        stats["panel_firm_years_dropped_above_max_apps"] = int((panel_pre_app["n_apps"] > MAX_APPS).sum())
    else:
        stats["panel_firm_years_dropped_above_max_apps"] = 0

    # Optionally restrict to (foia_firm_uid, lottery_year) pairs present in the
    # individual merge sample.  Useful for apples-to-apples comparison with the
    # individual-level estimates.
    if INDIV_SAMPLE_RESTRICT_ENABLED:
        _restrict_path = INDIV_SAMPLE_RESTRICT_PARQUET
        if _restrict_path and Path(_restrict_path).exists():
            _indiv_keys = pd.read_parquet(_restrict_path, columns=["foia_firm_uid", "lottery_year"])
            _indiv_keys["lottery_year"] = _indiv_keys["lottery_year"].astype(str)
            _indiv_keys = _indiv_keys.drop_duplicates()
            _n_before_restrict = len(panel)
            panel = panel.merge(_indiv_keys, on=["foia_firm_uid", "lottery_year"], how="inner")
            _n_after_restrict  = len(panel)
            _n_dropped_restrict = _n_before_restrict - _n_after_restrict
            print(
                f"  [indiv_sample_restrict] {_n_after_restrict:,} / {_n_before_restrict:,} firm-years "
                f"kept ({_n_dropped_restrict:,} dropped — not in individual sample)"
            )
            stats["panel_firm_years_dropped_indiv_restrict"] = _n_dropped_restrict
            stats["panel_firm_years_post_indiv_restrict"]    = _n_after_restrict
            stats["panel_firms_post_indiv_restrict"]         = int(panel["foia_firm_uid"].nunique())
        else:
            print(f"  [WARN] indiv_sample_restrict enabled but parquet not found: {_restrict_path}")
            stats["panel_firm_years_dropped_indiv_restrict"] = 0
            stats["panel_firm_years_post_indiv_restrict"]    = int(len(panel))
            stats["panel_firms_post_indiv_restrict"]         = int(panel["foia_firm_uid"].nunique())
    else:
        stats["panel_firm_years_dropped_indiv_restrict"] = 0
        stats["panel_firm_years_post_indiv_restrict"]    = int(len(panel))
        stats["panel_firms_post_indiv_restrict"]         = int(panel["foia_firm_uid"].nunique())

    panel_by_year = (
        panel.groupby("lottery_year")
        .agg(
            firm_years=("foia_firm_uid", "size"),
            firms=("foia_firm_uid", "nunique"),
            firm_years_with_rcid=("rcid", lambda s: int(s.notna().sum())),
            firm_years_missing_rcid=("rcid", lambda s: int(s.isna().sum())),
            mean_n_apps=("n_apps", "mean"),
            p50_n_apps=("n_apps", "median"),
        )
        .reset_index()
        .sort_values("lottery_year")
    )
    firms_with_rcid = (
        panel[panel["rcid"].notna()]
        .groupby("lottery_year")["foia_firm_uid"]
        .nunique()
        .rename("firms_with_rcid")
        .reset_index()
    )
    panel_by_year = panel_by_year.merge(firms_with_rcid, on="lottery_year", how="left")
    panel_by_year["firms_with_rcid"] = panel_by_year["firms_with_rcid"].fillna(0).astype(int)
    panel_by_year["firms_missing_rcid"] = panel_by_year["firms"] - panel_by_year["firms_with_rcid"]
    stats["audit_panel_by_year"] = panel_by_year

    # Safety check
    degen = yr_stats[yr_stats["win_rate_pooled"].ge(0.99) | yr_stats["win_rate_pooled"].le(0.01)]
    if not degen.empty:
        print(f"  [WARN] Degenerate win rates in years: {degen['lottery_year'].tolist()} — check data")

    panel["lottery_year"] = panel["lottery_year"].astype(int)

    n_firms = panel["foia_firm_uid"].nunique()
    n_yrs   = panel["lottery_year"].nunique()
    print(f"  Panel: {len(panel):,} firm-years | {n_firms:,} firms | {n_yrs} lottery years")
    print(f"  Win rates by year:")
    for _, row in yr_stats.iterrows():
        print(f"    {row['lottery_year']}: ADE={row['win_rate_ADE']:.3f}  "
              f"nonADE={row['win_rate_nonADE']:.3f}  pooled={row['win_rate_pooled']:.3f}  "
              f"(ADE share: {row['nat_ade_share']:.2f})")
    print(f"  U_itau: mean={panel['U_itau'].mean():.4f}  sd={panel['U_itau'].std():.4f}")
    print(f"  rcid missing: {panel['rcid'].isna().sum():,} / {len(panel):,}")
    print()

    if return_stats:
        return panel, stats
    return panel


###############################################################################
# STEP 2: REVELIO HEADCOUNT FROM WRDS
###############################################################################

def _values_clause(ids: list) -> str:
    """Format a list of IDs as a SQL VALUES clause for use in a WITH block."""
    return ",".join(f"({int(x)})" for x in ids if pd.notna(x))


def _load_h1b_role_values_from_foia(crosswalk_path: str) -> list:
    """
    Build the role_k17000_v3 whitelist from FOIA occupation rank/share metadata.

    Expected columns (either naming scheme is accepted):
      - rank:  foia_occ_rank  OR  min_rank
      - share: foia_occ_share OR  max_share_foia
    """
    path = Path(crosswalk_path)
    if not path.exists():
        print(f"  [WARN] h1b_occ_filter enabled but crosswalk not found: {path}")
        return []

    cw = pd.read_csv(path)
    role_col = "role_k17000_v3"
    if role_col not in cw.columns:
        print(f"  [WARN] {path} missing '{role_col}' column; skipping H-1B occupation headcount")
        return []

    rank_col = "foia_occ_rank" if "foia_occ_rank" in cw.columns else (
        "min_rank" if "min_rank" in cw.columns else None
    )
    share_col = "foia_occ_share" if "foia_occ_share" in cw.columns else (
        "max_share_foia" if "max_share_foia" in cw.columns else None
    )
    if rank_col is None and share_col is None:
        print(f"  [WARN] {path} missing foia rank/share columns; skipping H-1B occupation headcount")
        return []

    keep_cols = [role_col]
    if rank_col is not None:
        keep_cols.append(rank_col)
    if share_col is not None:
        keep_cols.append(share_col)
    sub = cw[keep_cols].copy()
    sub = sub[sub[role_col].notna()].copy()
    sub[role_col] = sub[role_col].astype(str).str.strip()
    sub = sub[sub[role_col] != ""]

    mask = pd.Series(True, index=sub.index)
    if rank_col is not None:
        rank = pd.to_numeric(sub[rank_col], errors="coerce")
        if H1B_OCC_MAX_RANK is None:
            # create_occ_cw.py uses rank=1000 for unmatched roles
            mask &= rank.notna() & (rank < 1000)
        else:
            mask &= rank.notna() & (rank <= H1B_OCC_MAX_RANK)

    if share_col is not None:
        share = pd.to_numeric(sub[share_col], errors="coerce").fillna(0)
        if H1B_OCC_MIN_SHARE <= 0:
            mask &= share > 0
        else:
            mask &= share >= H1B_OCC_MIN_SHARE

    roles = sorted(sub.loc[mask, role_col].drop_duplicates().tolist())
    print(f"  H-1B occupation filter from {path.name}: {len(roles):,} role values selected")
    return roles


def _headcount_query(rcids: list, year_min: int, year_max: int) -> str:
    """
    Build a WRDS SQL query that computes firm-year headcounts via Jan 1 snapshot.

    Returns:
      rcid, year, emp_all, emp_tenure
      (emp_h1b_occ is added separately when H-1B occupation filtering is active)
    """
    vals = _values_clause(rcids)
    snap = f"{SNAP_MONTH:02d}-{SNAP_DAY:02d}"
    tenure_interval = f"{TENURE_MONTHS} months"

    query = f"""
    WITH rcids AS (
        SELECT column1::BIGINT AS rcid
        FROM (VALUES {vals}) AS v(column1)
    ),
    positions AS (
        SELECT
            p.user_id,
            p.rcid,
            p.startdate::DATE AS startdate,
            COALESCE(p.enddate::DATE, '{ENDDATE_FALLBACK}'::DATE) AS enddate,
            p.role_k17000_v3
        FROM revelio.individual_positions AS p
        JOIN rcids USING (rcid)
        WHERE p.country = 'United States'
          AND p.startdate IS NOT NULL
    ),
    years AS (
        SELECT generate_series({year_min}, {year_max}) AS year
    ),
    snapshot AS (
        SELECT
            p.rcid,
            y.year,
            COUNT(DISTINCT p.user_id) AS emp_all,
            COUNT(DISTINCT CASE
                WHEN p.startdate <= (make_date(y.year, {SNAP_MONTH}, {SNAP_DAY})
                                     - INTERVAL '{tenure_interval}')
                THEN p.user_id END) AS emp_tenure
        FROM positions p
        CROSS JOIN years y
        WHERE p.startdate <= make_date(y.year, {SNAP_MONTH}, {SNAP_DAY})
          AND p.enddate   >= make_date(y.year, {SNAP_MONTH}, {SNAP_DAY})
        GROUP BY p.rcid, y.year
    )
    SELECT rcid, year, emp_all, emp_tenure
    FROM snapshot
    ORDER BY rcid, year
    """
    return query


def _headcount_query_h1b_occ(rcids: list, year_min: int, year_max: int,
                               role_values: list) -> str:
    """
    Headcount restricted to H-1B occupations using role_k17000_v3 values
    derived from FOIA occupation rank/share metadata.
    """
    vals = _values_clause(rcids)
    rc_vals = ",".join("'" + str(v).replace("'", "''") + "'" for v in role_values)
    query = f"""
    WITH rcids AS (
        SELECT column1::BIGINT AS rcid
        FROM (VALUES {vals}) AS v(column1)
    ),
    positions AS (
        SELECT
            p.user_id,
            p.rcid,
            p.startdate::DATE AS startdate,
            COALESCE(p.enddate::DATE, '{ENDDATE_FALLBACK}'::DATE) AS enddate
        FROM revelio.individual_positions AS p
        JOIN rcids USING (rcid)
        WHERE p.country = 'United States'
          AND p.startdate IS NOT NULL
          AND p.role_k17000_v3 IN ({rc_vals})
    ),
    years AS (
        SELECT generate_series({year_min}, {year_max}) AS year
    ),
    snapshot AS (
        SELECT
            p.rcid,
            y.year,
            COUNT(DISTINCT p.user_id) AS emp_h1b_occ
        FROM positions p
        CROSS JOIN years y
        WHERE p.startdate <= make_date(y.year, {SNAP_MONTH}, {SNAP_DAY})
          AND p.enddate   >= make_date(y.year, {SNAP_MONTH}, {SNAP_DAY})
        GROUP BY p.rcid, y.year
    )
    SELECT rcid, year, emp_h1b_occ
    FROM snapshot
    ORDER BY rcid, year
    """
    return query


def query_headcounts_wrds(rcids: list, output_path: Path, return_stats: bool = False):
    """
    Query Revelio headcounts from WRDS for the given list of rcids.
    Results are cached to:
      - output_path/headcount_panel.parquet in full mode
      - output_path/headcount_panel_test.parquet in testing mode

    Returns DataFrame with (rcid, year, emp_all, emp_tenure[, emp_h1b_occ]).
    """
    stats = {}
    cache_name = "headcount_panel_test.parquet" if TESTING_ENABLED else "headcount_panel.parquet"
    cache = output_path / cache_name
    stats["headcount_cache_path"] = str(cache)

    if cache.exists() and not FORCE_REQUERY:
        print(f"  Loading cached headcount panel from {cache}")
        hc = pd.read_parquet(cache)
        stats["headcount_loaded_from_cache"] = True
        stats["headcount_rows"] = int(len(hc))
        stats["headcount_firms"] = int(hc["rcid"].nunique()) if "rcid" in hc.columns else 0
        stats["headcount_years"] = int(hc["year"].nunique()) if "year" in hc.columns else 0
        if return_stats:
            return hc, stats
        return hc
    if cache.exists() and FORCE_REQUERY:
        print(f"  force_requery=True; ignoring existing cache at {cache}")
    stats["headcount_loaded_from_cache"] = False

    import wrds
    print(f"  Connecting to WRDS as '{WRDS_USER}'...")
    db = wrds.Connection(wrds_username=WRDS_USER)

    # Filter to valid integer rcids
    valid_rcids = sorted({int(r) for r in rcids if pd.notna(r)})
    stats["rcids_requested"] = int(len(valid_rcids))
    print(f"  Querying headcounts for {len(valid_rcids):,} rcids "
          f"({HC_YEAR_MIN}–{HC_YEAR_MAX}, Jan {SNAP_DAY} snapshot)...")

    h1b_role_values = []
    if H1B_OCC_ENABLED:
        h1b_role_values = _load_h1b_role_values_from_foia(H1B_OCC_CROSSWALK)
        if not h1b_role_values:
            print("  [WARN] No H-1B occupation roles selected; emp_h1b_occ will be skipped")
    else:
        print("  H-1B occupation filtering disabled; emp_h1b_occ will be skipped")

    # Chunk the rcid list to avoid WRDS query size limits
    chunks = [valid_rcids[i:i+CHUNK_SIZE]
              for i in range(0, len(valid_rcids), CHUNK_SIZE)]
    n_chunks = len(chunks)
    stats["headcount_chunks"] = int(n_chunks)

    parts_main = []
    parts_occ  = []

    for i, chunk in enumerate(chunks):
        t_chunk = time.time()
        print(f"    Chunk {i+1}/{n_chunks} ({len(chunk)} rcids)...", end="", flush=True)

        q_main = _headcount_query(chunk, HC_YEAR_MIN, HC_YEAR_MAX)
        df_main = db.raw_sql(q_main)
        parts_main.append(df_main)

        if h1b_role_values:
            q_occ = _headcount_query_h1b_occ(chunk, HC_YEAR_MIN, HC_YEAR_MAX,
                                              h1b_role_values)
            df_occ = db.raw_sql(q_occ)
            parts_occ.append(df_occ)

        print(f" {len(df_main):,} rows ({time.time()-t_chunk:.1f}s)")

    if parts_main:
        hc = pd.concat(parts_main, ignore_index=True)
    else:
        hc = pd.DataFrame(columns=["rcid", "year", "emp_all", "emp_tenure"])
    hc["rcid"] = hc["rcid"].astype("Int64")
    hc["year"] = hc["year"].astype(int)

    if h1b_role_values and parts_occ:
        hc_occ = pd.concat(parts_occ, ignore_index=True)
        hc_occ["rcid"] = hc_occ["rcid"].astype("Int64")
        hc_occ["year"] = hc_occ["year"].astype(int)
        hc = hc.merge(hc_occ, on=["rcid", "year"], how="left")

    db.close()

    print(f"  Headcount panel: {len(hc):,} rows, "
          f"{hc['rcid'].nunique():,} firms, {hc['year'].nunique()} years")
    print(f"  emp_all:    min={hc['emp_all'].min():.0f}  "
          f"mean={hc['emp_all'].mean():.1f}  max={hc['emp_all'].max():.0f}")
    print(f"  emp_tenure: min={hc['emp_tenure'].min():.0f}  "
          f"mean={hc['emp_tenure'].mean():.1f}  max={hc['emp_tenure'].max():.0f}")
    stats["headcount_rows"] = int(len(hc))
    stats["headcount_firms"] = int(hc["rcid"].nunique())
    stats["headcount_years"] = int(hc["year"].nunique())

    output_path.mkdir(parents=True, exist_ok=True)
    hc.to_parquet(cache, index=False)
    print(f"  Saved: {cache}")
    print()

    if return_stats:
        return hc, stats
    return hc


###############################################################################
# STEP 3: BUILD ANALYSIS PANEL
###############################################################################

def trim_top_baseline_size_firms(
    panel: pd.DataFrame,
    hc: pd.DataFrame,
    return_stats: bool = False,
):
    """
    Optional sample trim: drop the top x% of firms by baseline pre-lottery size.
    Size is measured using BASELINE_SIZE_COL at baseline year:
      - baseline_size_year (if set), otherwise min(lottery_year)-1.
    """
    stats = {
        "baseline_trim_enabled": TOP_SIZE_TRIM_PCT > 0,
        "baseline_trim_pct": TOP_SIZE_TRIM_PCT,
        "baseline_trim_col": BASELINE_SIZE_COL,
        "baseline_trim_drop_missing": BASELINE_SIZE_DROP_MISSING,
    }

    if TOP_SIZE_TRIM_PCT <= 0:
        if return_stats:
            return panel, stats
        return panel

    print("Step 2b: Baseline-size trimming...")
    p = panel.copy()
    stats["baseline_trim_rows_before"] = int(len(p))
    stats["baseline_trim_firms_before"] = int(p["foia_firm_uid"].nunique())

    if p.empty:
        print("  [SKIP] panel is empty")
        if return_stats:
            return p, stats
        return p

    if BASELINE_SIZE_COL not in hc.columns:
        print(f"  [WARN] baseline_size_col='{BASELINE_SIZE_COL}' not found in headcount; skipping trim")
        if return_stats:
            return p, stats
        return p

    byear = BASELINE_SIZE_YEAR
    if byear is None:
        ly = pd.to_numeric(p["lottery_year"], errors="coerce")
        ly = ly.dropna()
        if ly.empty:
            print("  [WARN] lottery_year missing; skipping trim")
            if return_stats:
                return p, stats
            return p
        byear = int(ly.min()) - 1

    stats["baseline_trim_year"] = int(byear)
    base = hc.loc[hc["year"] == byear, ["rcid", BASELINE_SIZE_COL]].copy()
    base = base.rename(columns={BASELINE_SIZE_COL: "baseline_size"})
    base["baseline_size"] = pd.to_numeric(base["baseline_size"], errors="coerce")
    base = base.dropna(subset=["rcid"]).copy()
    base = base.drop_duplicates(subset=["rcid"])

    firm_rcid = p[["foia_firm_uid", "rcid"]].drop_duplicates().copy()
    firm_rcid = firm_rcid.dropna(subset=["foia_firm_uid"]).copy()
    firm_size = firm_rcid.merge(base, on="rcid", how="left")
    firm_size = firm_size.drop_duplicates(subset=["foia_firm_uid"])

    stats["baseline_trim_firms_missing_baseline"] = int(firm_size["baseline_size"].isna().sum())
    eligible = firm_size[firm_size["baseline_size"].notna()].copy()
    stats["baseline_trim_firms_with_baseline"] = int(len(eligible))

    if eligible.empty:
        print("  [WARN] No firms with non-missing baseline size; skipping trim")
        if return_stats:
            return p, stats
        return p

    q = 1.0 - TOP_SIZE_TRIM_PCT / 100.0
    q = min(max(q, 0.0), 1.0)
    cutoff = float(eligible["baseline_size"].quantile(q))
    stats["baseline_trim_cutoff"] = cutoff

    drop_firms = set(eligible.loc[eligible["baseline_size"] >= cutoff, "foia_firm_uid"].tolist())
    if BASELINE_SIZE_DROP_MISSING:
        missing_firms = set(firm_size.loc[firm_size["baseline_size"].isna(), "foia_firm_uid"].tolist())
        drop_firms |= missing_firms
        stats["baseline_trim_firms_missing_dropped"] = int(len(missing_firms))
    else:
        stats["baseline_trim_firms_missing_dropped"] = 0

    before_by_year = (
        p.groupby("lottery_year")
        .agg(n_rows=("foia_firm_uid", "size"), n_firms=("foia_firm_uid", "nunique"))
        .reset_index()
    )
    out = p[~p["foia_firm_uid"].isin(drop_firms)].copy()
    after_by_year = (
        out.groupby("lottery_year")
        .agg(n_rows_after_trim=("foia_firm_uid", "size"), n_firms_after_trim=("foia_firm_uid", "nunique"))
        .reset_index()
    )
    audit = before_by_year.merge(after_by_year, on="lottery_year", how="left").fillna(0)
    audit["n_rows_dropped"] = audit["n_rows"] - audit["n_rows_after_trim"]
    audit["n_firms_dropped"] = audit["n_firms"] - audit["n_firms_after_trim"]
    stats["audit_panel_after_baseline_trim_by_year"] = audit

    stats["baseline_trim_firms_dropped"] = int(len(drop_firms))
    stats["baseline_trim_rows_after"] = int(len(out))
    stats["baseline_trim_firms_after"] = int(out["foia_firm_uid"].nunique())
    stats["baseline_trim_rows_dropped"] = int(stats["baseline_trim_rows_before"] - stats["baseline_trim_rows_after"])

    print(
        f"  Baseline year={byear}, col={BASELINE_SIZE_COL}, cutoff={cutoff:.3f} "
        f"(top {TOP_SIZE_TRIM_PCT:.3f}%)"
    )
    print(
        f"  Dropped {stats['baseline_trim_firms_dropped']:,} firms "
        f"({stats['baseline_trim_rows_dropped']:,} firm-year rows)"
    )
    print(
        f"  Remaining: {stats['baseline_trim_firms_after']:,} firms, "
        f"{stats['baseline_trim_rows_after']:,} firm-year rows"
    )
    print()

    if return_stats:
        return out, stats
    return out


def build_analysis_panel(panel: pd.DataFrame, hc: pd.DataFrame,
                          output_path: Path, return_stats: bool = False):
    """
    For each lag k, merge headcount at outcome year (lottery_year + k) onto the
    firm-year panel. Returns a long DataFrame with columns:
      foia_firm_uid, lottery_year, n_apps, n_wins, U_itau, k,
      emp_all, emp_tenure[, emp_h1b_occ], log_emp_all, log_emp_tenure[, log_emp_h1b_occ]
    """
    print("Step 3: Building analysis panel...")
    t0 = time.time()
    stats = {}

    # emp versions available
    emp_cols = ["emp_all", "emp_tenure"]
    if "emp_h1b_occ" in hc.columns:
        emp_cols.append("emp_h1b_occ")

    rows = []
    for k in K_LAGS:
        sub = panel.copy()
        sub["outcome_year"] = sub["lottery_year"] + k
        sub["k"] = k

        # Merge headcount at outcome_year
        hc_k = hc[["rcid", "year"] + emp_cols].rename(columns={"year": "outcome_year"})
        sub = sub.merge(hc_k, on=["rcid", "outcome_year"], how="left")

        rows.append(sub)

    adf = pd.concat(rows, ignore_index=True)
    stats["audit_analysis_pre_rcid_by_year"] = (
        adf.groupby("lottery_year")
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values("lottery_year")
    )
    stats["audit_analysis_pre_rcid_by_year_k"] = (
        adf.groupby(["lottery_year", "k"])
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["lottery_year", "k"])
    )
    stats["analysis_rows_pre_rcid_filter"] = int(len(adf))
    stats["analysis_rcid_missing_rows_dropped"] = int(adf["rcid"].isna().sum())
    # Keep only firms that are matched to rcid.
    adf = adf[adf["rcid"].notna()].copy()
    stats["audit_analysis_post_rcid_by_year"] = (
        adf.groupby("lottery_year")
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values("lottery_year")
    )
    stats["audit_analysis_post_rcid_by_year_k"] = (
        adf.groupby(["lottery_year", "k"])
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["lottery_year", "k"])
    )
    stats["analysis_rows_post_rcid_filter"] = int(len(adf))
    stats["analysis_firms_post_rcid_filter"] = int(adf["foia_firm_uid"].nunique())
    stats["analysis_lottery_years_post_rcid_filter"] = int(adf["lottery_year"].nunique())
    stats["analysis_zero_imputed_rows"] = {}
    stats["analysis_balanced_retained_rows"] = {}
    stats["audit_outcome_nonmissing_by_year_k"] = {}

    # Compute outcome columns; cast to float first to handle nullable Int64 from WRDS.
    # Missing-outcome handling follows MISSING_OUTCOME_POLICY.
    for col in emp_cols:
        out_col = f"out_{col}"
        raw_vals = pd.to_numeric(adf[col], errors="coerce")
        raw_vals = raw_vals.where(np.isfinite(raw_vals), np.nan)
        raw_vals = raw_vals.where(raw_vals >= 0, np.nan)

        if MISSING_OUTCOME_POLICY == "zero_impute":
            rcids_with_positive = set(adf.loc[raw_vals > 0, "rcid"].dropna().unique().tolist())
            impute_mask = raw_vals.isna() & adf["rcid"].isin(rcids_with_positive)
            vals = raw_vals.copy()
            vals.loc[impute_mask] = 0.0
            stats["analysis_zero_imputed_rows"][out_col] = int(impute_mask.sum())
            stats["analysis_balanced_retained_rows"][out_col] = np.nan

            if LOG_OUTCOMES:
                adf[out_col] = np.where(vals > 0, np.log(vals), np.where(vals == 0, 0.0, np.nan))
            else:
                adf[out_col] = vals

        elif MISSING_OUTCOME_POLICY == "balanced_panel":
            # Keep a balanced (firm-year x k) block for this outcome:
            # if any lag is missing for a firm-year, drop all lags for that firm-year.
            complete_mask = (
                raw_vals.notna()
                .groupby([adf["foia_firm_uid"], adf["lottery_year"]])
                .transform("all")
            )
            vals = raw_vals.where(complete_mask, np.nan)
            stats["analysis_zero_imputed_rows"][out_col] = 0
            stats["analysis_balanced_retained_rows"][out_col] = int(complete_mask.sum())

            if LOG_OUTCOMES:
                adf[out_col] = np.where(vals > 0, np.log(vals), np.nan)
            else:
                adf[out_col] = vals

        nonmiss_by_yk = (
            adf.assign(_nonmiss=adf[out_col].notna().astype(int))
            .groupby(["lottery_year", "k"])["_nonmiss"]
            .sum()
            .reset_index()
            .rename(columns={"_nonmiss": "n_nonmissing"})
            .sort_values(["lottery_year", "k"])
        )
        stats["audit_outcome_nonmissing_by_year_k"][out_col] = nonmiss_by_yk

    # Heterogeneity bins (at foia_firm_uid x lottery_year level)
    key_cols = ["foia_firm_uid", "lottery_year"]
    key_df = adf[key_cols + ["n_apps", "share_ade_wins_est", "industry_group"]].drop_duplicates(subset=key_cols).copy()
    key_df["apps_q"] = _assign_quartile(key_df["n_apps"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    key_df["ade_win_share_q"] = _assign_quartile(
        key_df["share_ade_wins_est"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    # Pre-period size quartile uses emp_all at k=-1 when available, else nearest pre-period lag.
    pre_lags = sorted([k for k in K_LAGS if k < 0], reverse=True)
    k_pre = -1 if -1 in K_LAGS else (pre_lags[0] if pre_lags else min(K_LAGS))
    pre_emp = (
        adf[adf["k"] == k_pre][key_cols + ["emp_all"]]
        .drop_duplicates(subset=key_cols)
        .rename(columns={"emp_all": "_pre_emp_all"})
    )
    pre_emp["pre_emp_all_q"] = _assign_quartile(pre_emp["_pre_emp_all"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    pre_emp["pre_emp_all_d"] = _assign_quartile(
        pre_emp["_pre_emp_all"],
        q=10,
        labels=[f"D{i:02d}" for i in range(1, 11)],
    )
    key_df = key_df.merge(pre_emp[key_cols + ["pre_emp_all_q", "pre_emp_all_d"]], on=key_cols, how="left")
    adf = adf.merge(
        key_df[key_cols + ["apps_q", "ade_win_share_q", "pre_emp_all_q", "pre_emp_all_d"]],
        on=key_cols,
        how="left",
    )

    # Summary
    outcome_label = "log_emp" if LOG_OUTCOMES else "emp_level"
    print(f"  Analysis panel: {len(adf):,} rows ({len(K_LAGS)} lags × firm-years)  [outcomes: {outcome_label}]")
    print(f"  rcid-missing rows dropped: {stats['analysis_rcid_missing_rows_dropped']:,}")
    outcome_nonmissing = {}
    for col in emp_cols:
        n_nonmiss = adf[f"out_{col}"].notna().sum()
        outcome_nonmissing[f"out_{col}"] = int(n_nonmiss)
        n_imp = stats["analysis_zero_imputed_rows"].get(f"out_{col}", 0)
        if MISSING_OUTCOME_POLICY == "zero_impute":
            print(f"  {f'out_{col}':20s}: {n_nonmiss:,} non-missing obs | zero-imputed={n_imp:,}")
        else:
            n_bal = stats["analysis_balanced_retained_rows"].get(f"out_{col}", np.nan)
            n_bal_txt = f"{int(n_bal):,}" if pd.notna(n_bal) else "NA"
            print(f"  {f'out_{col}':20s}: {n_nonmiss:,} non-missing obs | balanced-retained={n_bal_txt}")
    stats["analysis_outcome_nonmissing"] = outcome_nonmissing

    out = output_path / "analysis_panel.parquet"
    adf.to_parquet(out, index=False)
    print(f"  Saved: {out}  ({time.time()-t0:.1f}s)")
    print()

    if return_stats:
        return adf, stats
    return adf


###############################################################################
# STEP 4: REGRESSIONS
###############################################################################

def _build_formula(outcome: str, fe_spec: str) -> str:
    rhs = "U_itau"
    if fe_spec == "firm_only":
        rhs += " + EntityEffects"
    elif fe_spec == "year_only":
        rhs += " + TimeEffects"
    elif fe_spec == "firm_year":
        rhs += " + EntityEffects + TimeEffects"
    elif fe_spec != "none":
        raise ValueError(f"Unsupported FE spec: {fe_spec}")
    return f"{outcome} ~ {rhs}"


def _fit_panel_model(sub: pd.DataFrame, outcome: str, fe_spec: str, quiet: bool = False):
    """
    Fit a PanelOLS model on a pre-filtered subset.
    Returns PanelOLSResults or None.
    """
    work = sub[["foia_firm_uid", "lottery_year", "U_itau", "n_apps", outcome]].dropna().copy()
    if len(work) < 10:
        return None
    if work["foia_firm_uid"].nunique() < 2 or work["lottery_year"].nunique() < 2:
        return None

    work["_year_int"] = work["lottery_year"].astype(int)
    pdata = work.set_index(["foia_firm_uid", "_year_int"])
    try:
        model = PanelOLS.from_formula(_build_formula(outcome, fe_spec), data=pdata)
        return model.fit(cov_type="clustered", cluster_entity=True)
    except Exception as e:
        if not quiet:
            print(f"    [ERR] {outcome} fe={fe_spec}: {e}")
        return None


def run_regression(df: pd.DataFrame, outcome: str, k: int, fe_spec: str):
    """
    Run PanelOLS for a single (outcome, k, FE spec) combination.
    Entity = foia_firm_uid, Time = lottery_year.
    Formula: out_col ~ U_itau + FE terms

    Returns PanelOLSResults or None.
    """
    sub = df[df["k"] == k].copy()
    res = _fit_panel_model(sub, outcome, fe_spec, quiet=True)
    if res is None:
        # Print context-aware error line only when this looks like a model failure vs. low-support skip.
        tmp = sub[["foia_firm_uid", "lottery_year", "U_itau", "n_apps", outcome]].dropna()
        if len(tmp) >= 10 and tmp["foia_firm_uid"].nunique() >= 2 and tmp["lottery_year"].nunique() >= 2:
            print(f"    [ERR] {outcome} k={k} fe={fe_spec}: model fit failed")
    return res


def run_all_regressions(adf: pd.DataFrame, emp_cols: list, fe_specs: list) -> dict:
    """
    Run regressions for all (fe_spec, outcome, k) combinations.
    Returns nested dict: {fe_spec: {out_col: {k: result_or_None}}}.
    """
    print("Step 4: Running regressions...")
    results = {}

    for fe_spec in fe_specs:
        results[fe_spec] = {}
        print(f"  FE spec: {fe_spec} ({_fe_label(fe_spec)})")
        for col in emp_cols:
            out_col = f"out_{col}"
            if out_col not in adf.columns:
                continue
            results[fe_spec][out_col] = {}
            print(f"    Outcome: {out_col}")
            for k in K_LAGS:
                res = run_regression(adf, out_col, k, fe_spec=fe_spec)
                results[fe_spec][out_col][k] = res
                if res is not None:
                    b   = res.params.get("U_itau", np.nan)
                    se  = res.std_errors.get("U_itau", np.nan)
                    pv  = res.pvalues.get("U_itau", np.nan)
                    n   = int(res.nobs)
                    stars = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""
                    print(f"      k={k:+d}:  beta={b:+.4f}{stars}  SE={se:.4f}  N={n:,}")
                else:
                    print(f"      k={k:+d}:  [skip]")

    print()
    return results


###############################################################################
# STEP 5: TABLES AND PLOTS
###############################################################################

def make_regression_table(
    results: dict,
    emp_cols: list,
    output_dir: Path,
    fe_spec: str,
    write_legacy: bool = False,
) -> str:
    """
    Format a LaTeX regression table.
    Rows = lags k; Columns = headcount versions.
    """
    out_cols = [f"out_{c}" for c in emp_cols if f"out_{c}" in results]
    n_cols = len(out_cols)

    # Header row labels for each headcount version
    col_labels = {
        "out_emp_all":     "All employees",
        "out_emp_tenure":  r"Tenured ($\geq$6 mo.)",
        "out_emp_h1b_occ": "H-1B occupations",
    }

    dv_label = "log employment" if LOG_OUTCOMES else "employment level"
    fe_label = _fe_label(fe_spec)
    latex  = f"% Firm-level employment effects of lottery wins (DIGe-style); DV: {dv_label}; FE: {fe_label}\n"
    latex += "\\begin{tabular}{l" + "c" * n_cols + "}\n"
    latex += "\\toprule\n"
    latex += "Lag $k$ & " + " & ".join(
        col_labels.get(c, c) for c in out_cols
    ) + " \\\\\n"
    latex += "\\midrule\n"

    for k in K_LAGS:
        coefs, ses = [], []
        for lc in out_cols:
            res = results.get(lc, {}).get(k)
            if res is None or "U_itau" not in res.params:
                coefs.append("--")
                ses.append("")
            else:
                b  = res.params["U_itau"]
                se = res.std_errors["U_itau"]
                pv = res.pvalues["U_itau"]
                stars = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""
                coefs.append(f"{b:.4f}{stars}")
                ses.append(f"({se:.4f})")
        latex += f"$k={k:+d}$ & " + " & ".join(coefs) + " \\\\\n"
        if any(s for s in ses):
            latex += " & " + " & ".join(ses) + " \\\\\n"

    # N row (use k=1 for reference N per column)
    latex += "\\midrule\n"
    n_strs = []
    for lc in out_cols:
        res = results.get(lc, {}).get(1) or results.get(lc, {}).get(0)
        n_strs.append(str(int(res.nobs)) if res is not None else "")
    latex += "\\textbf{N} & " + " & ".join(n_strs) + " \\\\\n"
    latex += "\\textit{Fixed effects} & " + " & ".join([fe_label] * n_cols) + " \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"

    path = output_dir / "tables" / f"reg_firm_outcomes_{fe_spec}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex)
    print(f"\n--- {path.name} ---")
    print(latex)
    print(f"\n  Saved: {path}")
    if write_legacy:
        legacy = output_dir / "tables" / "reg_firm_outcomes.tex"
        legacy.write_text(latex)
        print(f"  Saved legacy copy: {legacy}")

    return latex


def make_event_study_plot(
    results: dict,
    emp_cols: list,
    output_dir: Path,
    fe_spec: str,
    write_legacy: bool = False,
):
    """
    Event study plot: beta_k ± 1.96*SE vs k, one line per headcount version.
    Saved to output_dir/plots/event_study_{fe_spec}.png.
    """
    out_cols = [f"out_{c}" for c in emp_cols if f"out_{c}" in results]
    if not out_cols:
        print("  [SKIP] No results for event study plot")
        return

    col_labels = {
        "out_emp_all":     "All employees",
        "out_emp_tenure":  "Tenured (≥6 mo.)",
        "out_emp_h1b_occ": "H-1B occupations",
    }

    plot_data = []
    for lc in out_cols:
        for k in K_LAGS:
            res = results.get(lc, {}).get(k)
            if res is not None and "U_itau" in res.params:
                b  = res.params["U_itau"]
                se = res.std_errors["U_itau"]
                plot_data.append({
                    "k": k,
                    "beta": b,
                    "ci_lo": b - 1.96 * se,
                    "ci_hi": b + 1.96 * se,
                    "version": col_labels.get(lc, lc),
                })

    if not plot_data:
        print("  [SKIP] No valid estimates for event study plot")
        return

    pdf = pd.DataFrame(plot_data)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 5))

    palette = sns.color_palette("tab10", n_colors=len(out_cols))
    versions = pdf["version"].unique()

    for i, ver in enumerate(versions):
        sub = pdf[pdf["version"] == ver].sort_values("k")
        ax.plot(sub["k"], sub["beta"], marker="o", label=ver, color=palette[i])
        ax.fill_between(sub["k"], sub["ci_lo"], sub["ci_hi"],
                        alpha=0.15, color=palette[i])

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(-0.5, color="gray", linewidth=0.5, linestyle=":")  # pre/post separator
    y_label = ("$\\hat{\\beta}_k$ (effect of unexpected win on log employment)"
               if LOG_OUTCOMES else
               "$\\hat{\\beta}_k$ (effect of unexpected win on employment level)")
    ax.set_xlabel("Lag $k$ (years after lottery year $\\tau$)")
    ax.set_ylabel(y_label)
    ax.set_title(f"H-1B Lottery Wins and Firm Employment (DIGe-style, {_fe_label(fe_spec)})")
    ax.set_xticks(K_LAGS)
    ax.legend(title="Headcount version", loc="upper left")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"event_study_{fe_spec}.png"
    if write_legacy:
        legacy = plots_dir / "event_study.png"
        fig.savefig(legacy, dpi=150, bbox_inches="tight")
        print(f"  Saved legacy copy: {legacy}")
    _save_and_show(fig, out, dpi=150)
    print(f"  Saved: {out}")


def make_size_decile_event_study_plot(adf: pd.DataFrame, output_dir: Path):
    """
    Event-study coefficients by pre-period (t-1) firm size decile.
    Uses the configured outcome/FE spec and estimates beta_k within each decile.
    """
    if not PLOT_SIZE_DECILE_EVENT_STUDY:
        return

    if "pre_emp_all_d" not in adf.columns:
        print("  [SKIP] size-decile event study: pre_emp_all_d not available")
        return

    out_cols = [c for c in adf.columns if c.startswith("out_")]
    if not out_cols:
        print("  [SKIP] size-decile event study: no outcome columns")
        return
    outcome = SIZE_DECILE_EVENT_OUTCOME if SIZE_DECILE_EVENT_OUTCOME in out_cols else out_cols[0]
    fe_spec = SIZE_DECILE_EVENT_FE_SPEC if SIZE_DECILE_EVENT_FE_SPEC in {"none", "firm_only", "year_only", "firm_year"} else "firm_year"

    deciles = (
        adf["pre_emp_all_d"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": np.nan})
        .dropna()
        .unique()
        .tolist()
    )
    if not deciles:
        print("  [SKIP] size-decile event study: no non-missing deciles")
        return
    deciles = sorted(deciles, key=lambda x: int(str(x).replace("D", "")))

    rows = []
    for dec in deciles:
        sub_dec = adf[adf["pre_emp_all_d"] == dec].copy()
        for k in K_LAGS:
            res = run_regression(sub_dec, outcome, k, fe_spec=fe_spec)
            if res is not None and "U_itau" in res.params:
                b = float(res.params["U_itau"])
                se = float(res.std_errors["U_itau"])
                pv = float(res.pvalues["U_itau"])
                nobs = int(res.nobs)
                n_firms = int(
                    sub_dec[sub_dec["k"] == k][["foia_firm_uid", "lottery_year", outcome]]
                    .dropna()["foia_firm_uid"]
                    .nunique()
                )
                rows.append(
                    {
                        "decile": dec,
                        "k": k,
                        "beta": b,
                        "se": se,
                        "pval": pv,
                        "nobs": nobs,
                        "n_firms": n_firms,
                        "ci_lo": b - 1.96 * se,
                        "ci_hi": b + 1.96 * se,
                    }
                )
            else:
                rows.append(
                    {
                        "decile": dec,
                        "k": k,
                        "beta": np.nan,
                        "se": np.nan,
                        "pval": np.nan,
                        "nobs": 0,
                        "n_firms": 0,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                    }
                )

    est = pd.DataFrame(rows)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    table_out = plots_dir / f"event_study_by_size_decile_{outcome}_{fe_spec}.csv"
    est.to_csv(table_out, index=False)

    n_dec = len(deciles)
    ncols = 5
    nrows = int(np.ceil(n_dec / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.7 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for i, dec in enumerate(deciles):
        ax = axes[i]
        sub = est[(est["decile"] == dec) & est["beta"].notna()].sort_values("k")
        if not sub.empty:
            ax.plot(sub["k"], sub["beta"], marker="o", linewidth=1.3)
            ax.fill_between(sub["k"], sub["ci_lo"], sub["ci_hi"], alpha=0.2)
            n_firms_dec = int(sub["n_firms"].max())
            ax.set_title(f"{dec} (firms~{n_firms_dec:,})", fontsize=9)
        else:
            ax.set_title(f"{dec} (no estimates)", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.axvline(-0.5, color="gray", linewidth=0.6, linestyle=":")
        ax.set_xticks(K_LAGS)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Event Study by Pre-period Firm Size Decile (outcome={outcome}, FE={_fe_label(fe_spec)})",
        y=1.02,
    )
    fig.text(0.5, 0.01, "Lag k", ha="center")
    fig.text(0.01, 0.5, "Beta on U_itau", va="center", rotation="vertical")
    fig.tight_layout()

    out = plots_dir / f"event_study_by_size_decile_{outcome}_{fe_spec}.png"
    _save_and_show(fig, out, dpi=170)
    print(f"  Saved: {out}")
    print(f"  Saved: {table_out}")


def _print_heterogeneity_table(
    dfk: pd.DataFrame,
    outcome: str,
    fe_spec: str,
    group_col: str,
    title: str,
    ordered_groups: list | None = None,
    include_missing: bool = False,
):
    if group_col not in dfk.columns:
        print(f"\n--- Heterogeneity: {title} [skip: missing {group_col}] ---")
        return

    work = dfk.copy()
    if include_missing:
        work[group_col] = work[group_col].fillna("Missing")
    else:
        work = work[work[group_col].notna()].copy()
    work[group_col] = work[group_col].astype(str).str.strip()
    work = work[work[group_col] != ""].copy()
    if work.empty:
        print(f"\n--- Heterogeneity: {title} [skip: no rows after group filtering] ---")
        return
    if ordered_groups is None:
        groups = work[group_col].value_counts().index.tolist()
    else:
        seen = set(work[group_col].unique())
        groups = [g for g in ordered_groups if g in seen]
        groups += [g for g in work[group_col].value_counts().index.tolist() if g not in groups]

    rows = []
    for g in groups:
        sub = work[work[group_col] == g].copy()
        prep = sub[["foia_firm_uid", "lottery_year", "U_itau", "n_apps", outcome]].dropna()
        n_obs = int(len(prep))
        n_firms = int(prep["foia_firm_uid"].nunique()) if n_obs > 0 else 0
        n_years = int(prep["lottery_year"].nunique()) if n_obs > 0 else 0

        res = _fit_panel_model(sub, outcome, fe_spec, quiet=True)
        if res is not None and "U_itau" in res.params:
            b = float(res.params["U_itau"])
            se = float(res.std_errors["U_itau"])
            pv = float(res.pvalues["U_itau"])
            stars = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""
            beta_txt = f"{b:.4f}{stars}"
            se_txt = f"{se:.4f}"
            pv_txt = f"{pv:.4f}"
        else:
            beta_txt = "--"
            se_txt = ""
            pv_txt = ""

        rows.append(
            {
                "group": g,
                "N": n_obs,
                "firms": n_firms,
                "years": n_years,
                "beta_U_itau": beta_txt,
                "se": se_txt,
                "pval": pv_txt,
            }
        )

    out = pd.DataFrame(rows)
    print(f"\n--- Heterogeneity: {title} ---")
    print(out.to_string(index=False))


def print_heterogeneity_tables(adf: pd.DataFrame):
    """
    Print heterogeneity regressions to stdout (not saved to disk).
    """
    if not PRINT_HETEROGENEITY_TABLES:
        return

    out_cols = [c for c in adf.columns if c.startswith("out_")]
    if not out_cols:
        print("\n--- Heterogeneity: [skip] no outcome columns ---")
        return

    outcome = HETEROGENEITY_OUTCOME if HETEROGENEITY_OUTCOME in out_cols else out_cols[0]
    if HETEROGENEITY_FE_SPEC in FE_SPECS:
        fe_spec = HETEROGENEITY_FE_SPEC
    else:
        fe_spec = FE_SPECS[0]

    if HETEROGENEITY_K is None:
        if 1 in K_LAGS:
            k_target = 1
        else:
            nonneg = sorted([k for k in K_LAGS if k >= 0])
            k_target = nonneg[0] if nonneg else K_LAGS[0]
    else:
        if HETEROGENEITY_K in K_LAGS:
            k_target = HETEROGENEITY_K
        else:
            k_target = min(K_LAGS, key=lambda x: abs(x - HETEROGENEITY_K))
            print(f"  [WARN] heterogeneity_k={HETEROGENEITY_K} not in k_lags; using nearest k={k_target}")

    dfk = adf[adf["k"] == k_target].copy()
    if dfk.empty:
        print(f"\n--- Heterogeneity: [skip] no rows at k={k_target} ---")
        return

    print("\nStep 6: Heterogeneity tables (printed only)")
    print(f"  outcome={outcome} | fe={fe_spec} ({_fe_label(fe_spec)}) | k={k_target:+d}")

    # Requested groups
    _print_heterogeneity_table(
        dfk, outcome, fe_spec, "pre_emp_all_q",
        "Pre-period Firm Size Quartile",
        ordered_groups=["Q1", "Q2", "Q3", "Q4"],
    )
    _print_heterogeneity_table(
        dfk, outcome, fe_spec, "ade_win_share_q",
        "Share of Winners ADE Quartile",
        ordered_groups=["Q1", "Q2", "Q3", "Q4"],
    )

    # Industry buckets (top groups + Other)
    if "industry_group" in dfk.columns:
        ind = dfk["industry_group"].fillna("Unknown").astype(str).str.strip()
        ind = ind.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        top = ind.value_counts().head(8).index.tolist()
        dfk["industry_bucket"] = np.where(ind.isin(top), ind, "Other")
    else:
        dfk["industry_bucket"] = "Unknown"
    _print_heterogeneity_table(dfk, outcome, fe_spec, "industry_bucket", "Industry (Top Buckets)")

    # Extra: application-intensity quartile and lottery year
    _print_heterogeneity_table(
        dfk, outcome, fe_spec, "apps_q",
        "Application Count Quartile",
        ordered_groups=["Q1", "Q2", "Q3", "Q4"],
    )
    _print_heterogeneity_table(dfk, outcome, fe_spec, "lottery_year", "Lottery Year")


###############################################################################
# STEP 7: DEEP-DIVE DIAGNOSTICS
###############################################################################

def _resolve_focus_outcome(adf: pd.DataFrame) -> str | None:
    out_cols = [c for c in adf.columns if c.startswith("out_")]
    if not out_cols:
        return None
    if DIAG_FOCUS_OUTCOME in out_cols:
        return DIAG_FOCUS_OUTCOME
    return out_cols[0]


def _write_stage_audit_files(step1_stats: dict, step3_stats: dict, debug_dir: Path):
    tables = []

    def _append(name: str, df: pd.DataFrame | None):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return
        t = df.copy()
        if "lottery_year" in t.columns:
            t["lottery_year"] = pd.to_numeric(t["lottery_year"], errors="coerce")
            t = t.dropna(subset=["lottery_year"]).copy()
            t["lottery_year"] = t["lottery_year"].astype(int)
        t.insert(0, "stage", name)
        tables.append(t)

    _append("01_foia_loaded", step1_stats.get("audit_foia_loaded_by_year"))
    _append("02_lottery_pool", step1_stats.get("audit_lottery_pool_by_year"))
    _append("03_panel_after_min_apps", step1_stats.get("audit_panel_by_year"))
    trim_df = step1_stats.get("audit_panel_after_baseline_trim_by_year")
    if isinstance(trim_df, pd.DataFrame) and not trim_df.empty:
        t = pd.DataFrame(
            {
                "lottery_year": trim_df["lottery_year"],
                "n_rows": (
                    trim_df["n_rows_after_trim"]
                    if "n_rows_after_trim" in trim_df.columns
                    else trim_df.get("n_rows", np.nan)
                ),
                "n_firms": (
                    trim_df["n_firms_after_trim"]
                    if "n_firms_after_trim" in trim_df.columns
                    else trim_df.get("n_firms", np.nan)
                ),
                "n_rows_dropped": trim_df.get("n_rows_dropped", np.nan),
                "n_firms_dropped": trim_df.get("n_firms_dropped", np.nan),
            }
        )
        _append("03b_panel_after_baseline_trim", t)
    _append("04_analysis_pre_rcid", step3_stats.get("audit_analysis_pre_rcid_by_year"))
    _append("05_analysis_post_rcid", step3_stats.get("audit_analysis_post_rcid_by_year"))

    if tables:
        out = pd.concat(tables, ignore_index=True, sort=False)
        out = out.sort_values(["lottery_year", "stage"])
        out_path = debug_dir / "stage_audit_by_year.csv"
        out.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    pre_yk = step3_stats.get("audit_analysis_pre_rcid_by_year_k")
    if isinstance(pre_yk, pd.DataFrame) and not pre_yk.empty:
        out_path = debug_dir / "analysis_pre_rcid_by_year_k.csv"
        pre_yk.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    post_yk = step3_stats.get("audit_analysis_post_rcid_by_year_k")
    if isinstance(post_yk, pd.DataFrame) and not post_yk.empty:
        out_path = debug_dir / "analysis_post_rcid_by_year_k.csv"
        post_yk.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    nonmiss = step3_stats.get("audit_outcome_nonmissing_by_year_k", {})
    if isinstance(nonmiss, dict):
        for outcome, df in nonmiss.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                out_path = debug_dir / f"nonmissing_by_year_k_{outcome}.csv"
                df.to_csv(out_path, index=False)
                print(f"  Saved: {out_path}")


def _plot_outcome_level_diagnostics(adf: pd.DataFrame, outcome: str, debug_dir: Path):
    req = ["lottery_year", "outcome_year", "k", outcome]
    work = adf[req].dropna().copy()
    if work.empty:
        print(f"  [WARN] No non-missing rows for {outcome}; skipping raw trend plots")
        return

    # Overall means over outcome year.
    by_outcome_year = (
        work.groupby("outcome_year")[outcome]
        .agg(mean="mean", median="median", n="size")
        .reset_index()
        .sort_values("outcome_year")
    )
    by_outcome_year.to_csv(debug_dir / f"raw_means_by_outcome_year_{outcome}.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(by_outcome_year["outcome_year"], by_outcome_year["mean"], marker="o", label="Mean")
    ax.plot(by_outcome_year["outcome_year"], by_outcome_year["median"], marker="s", label="Median")
    ax.set_xlabel("Outcome year")
    ax.set_ylabel(outcome)
    ax.set_title(f"Raw {outcome} over outcome year")
    ax.legend(loc="best")
    _save_and_show(fig, debug_dir / f"raw_means_by_outcome_year_{outcome}.png")

    # Means by lottery cohort over outcome year.
    by_cohort_year = (
        work.groupby(["lottery_year", "outcome_year"])[outcome]
        .agg(mean="mean", n="size")
        .reset_index()
        .sort_values(["lottery_year", "outcome_year"])
    )
    by_cohort_year.to_csv(
        debug_dir / f"raw_means_by_lottery_year_over_outcome_year_{outcome}.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for ly, sub in by_cohort_year.groupby("lottery_year"):
        ax.plot(sub["outcome_year"], sub["mean"], marker="o", label=str(int(ly)))
    ax.set_xlabel("Outcome year")
    ax.set_ylabel(outcome)
    ax.set_title(f"Raw {outcome} over outcome year, by lottery cohort")
    ax.legend(title="Lottery year", loc="best")
    _save_and_show(fig, debug_dir / f"raw_means_by_lottery_year_over_outcome_year_{outcome}.png")

    # Event-time means.
    by_k = (
        work.groupby("k")[outcome]
        .agg(mean="mean", median="median", n="size")
        .reset_index()
        .sort_values("k")
    )
    by_k.to_csv(debug_dir / f"raw_means_by_k_{outcome}.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(by_k["k"], by_k["mean"], marker="o", label="Mean")
    ax.plot(by_k["k"], by_k["median"], marker="s", label="Median")
    ax.axvline(-0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Event time k")
    ax.set_ylabel(outcome)
    ax.set_title(f"Raw {outcome} by event time")
    ax.legend(loc="best")
    _save_and_show(fig, debug_dir / f"raw_means_by_k_{outcome}.png")

    by_cohort_k = (
        work.groupby(["lottery_year", "k"])[outcome]
        .agg(mean="mean", n="size")
        .reset_index()
        .sort_values(["lottery_year", "k"])
    )
    by_cohort_k.to_csv(debug_dir / f"raw_means_by_lottery_year_k_{outcome}.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ly, sub in by_cohort_k.groupby("lottery_year"):
        ax.plot(sub["k"], sub["mean"], marker="o", label=str(int(ly)))
    ax.axvline(-0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Event time k")
    ax.set_ylabel(outcome)
    ax.set_title(f"Raw {outcome} by event time and cohort")
    ax.legend(title="Lottery year", loc="best")
    _save_and_show(fig, debug_dir / f"raw_means_by_lottery_year_k_{outcome}.png")

    # Quantile bands over outcome year.
    q = (
        work.groupby("outcome_year")[outcome]
        .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        .unstack()
        .reset_index()
        .rename(columns={0.1: "p10", 0.25: "p25", 0.5: "p50", 0.75: "p75", 0.9: "p90"})
        .sort_values("outcome_year")
    )
    q.to_csv(debug_dir / f"raw_quantiles_by_outcome_year_{outcome}.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(q["outcome_year"], q["p10"], q["p90"], alpha=0.2, label="p10-p90")
    ax.fill_between(q["outcome_year"], q["p25"], q["p75"], alpha=0.25, label="p25-p75")
    ax.plot(q["outcome_year"], q["p50"], color="black", linewidth=1.5, label="Median")
    ax.set_xlabel("Outcome year")
    ax.set_ylabel(outcome)
    ax.set_title(f"Distribution of {outcome} over outcome year")
    ax.legend(loc="best")
    _save_and_show(fig, debug_dir / f"raw_quantiles_by_outcome_year_{outcome}.png")


def _u_itau_diagnostics(adf: pd.DataFrame, debug_dir: Path):
    key = adf[["foia_firm_uid", "lottery_year", "U_itau", "n_apps"]].drop_duplicates().copy()
    if key.empty:
        return

    summary = (
        key.groupby("lottery_year")["U_itau"]
        .agg(
            n="size",
            mean="mean",
            sd="std",
            p01=lambda s: s.quantile(0.01),
            p10=lambda s: s.quantile(0.10),
            p50=lambda s: s.quantile(0.50),
            p90=lambda s: s.quantile(0.90),
            p99=lambda s: s.quantile(0.99),
        )
        .reset_index()
        .sort_values("lottery_year")
    )
    summary.to_csv(debug_dir / "u_itau_summary_by_lottery_year.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(key["U_itau"], bins=80, kde=False, ax=ax)
    ax.set_title("Distribution of U_itau (all cohorts)")
    ax.set_xlabel("U_itau")
    _save_and_show(fig, debug_dir / "u_itau_hist_all.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=key, x="lottery_year", y="U_itau", ax=ax)
    ax.set_title("U_itau by lottery year")
    _save_and_show(fig, debug_dir / "u_itau_box_by_lottery_year.png")

    sample_n = min(len(key), max(1000, DIAG_SCATTER_SAMPLE_N))
    scat = key.sample(sample_n, random_state=42) if len(key) > sample_n else key
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(np.log1p(scat["n_apps"]), scat["U_itau"], s=6, alpha=0.15)
    ax.set_xlabel("log(1 + n_apps)")
    ax.set_ylabel("U_itau")
    ax.set_title("U_itau vs application count")
    _save_and_show(fig, debug_dir / "u_itau_vs_n_apps.png")

    pre_lags = sorted([k for k in K_LAGS if k < 0], reverse=True)
    if pre_lags:
        k_pre = -1 if -1 in pre_lags else pre_lags[0]
        pre = (
            adf[adf["k"] == k_pre][["foia_firm_uid", "lottery_year", "emp_all"]]
            .drop_duplicates()
            .rename(columns={"emp_all": "pre_emp_all"})
        )
        scat2 = key.merge(pre, on=["foia_firm_uid", "lottery_year"], how="left")
        scat2 = scat2.dropna(subset=["pre_emp_all"])
        if not scat2.empty:
            sample_n2 = min(len(scat2), max(1000, DIAG_SCATTER_SAMPLE_N))
            scat2 = scat2.sample(sample_n2, random_state=42) if len(scat2) > sample_n2 else scat2
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(np.log1p(scat2["pre_emp_all"]), scat2["U_itau"], s=6, alpha=0.15)
            ax.set_xlabel(f"log(1 + emp_all at k={k_pre})")
            ax.set_ylabel("U_itau")
            ax.set_title("U_itau vs pre-period firm size")
            _save_and_show(fig, debug_dir / "u_itau_vs_pre_emp_all.png")


def _matching_diagnostics(panel: pd.DataFrame, debug_dir: Path):
    p = panel.copy()
    if p.empty:
        return
    by_year = (
        p.groupby("lottery_year")
        .agg(
            firm_years=("foia_firm_uid", "size"),
            firms=("foia_firm_uid", "nunique"),
            firm_years_with_rcid=("rcid", lambda s: int(s.notna().sum())),
            firm_years_missing_rcid=("rcid", lambda s: int(s.isna().sum())),
            mean_n_apps=("n_apps", "mean"),
            p50_n_apps=("n_apps", "median"),
        )
        .reset_index()
        .sort_values("lottery_year")
    )
    by_year["share_firm_years_with_rcid"] = by_year["firm_years_with_rcid"] / by_year["firm_years"]
    by_year.to_csv(debug_dir / "matching_by_lottery_year.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(by_year["lottery_year"], by_year["share_firm_years_with_rcid"], marker="o")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share with non-missing rcid")
    ax.set_xlabel("Lottery year")
    ax.set_title("Match coverage by lottery year")
    _save_and_show(fig, debug_dir / "matching_share_by_lottery_year.png")

    if "rcid_source" in p.columns:
        src = (
            p[p["rcid"].notna()]
            .groupby(["lottery_year", "rcid_source"])
            .size()
            .rename("firm_years")
            .reset_index()
            .sort_values(["lottery_year", "rcid_source"])
        )
        src.to_csv(debug_dir / "matching_source_by_lottery_year.csv", index=False)

    firm_map = p[["foia_firm_uid", "rcid"]].drop_duplicates().dropna()
    if not firm_map.empty:
        rcid_to_firms = (
            firm_map.groupby("rcid")["foia_firm_uid"].nunique().rename("n_firms_per_rcid")
        )
        dup_summary = pd.DataFrame(
            {
                "n_rcids": [int(rcid_to_firms.size)],
                "n_rcids_with_multi_firms": [int((rcid_to_firms > 1).sum())],
                "max_firms_per_rcid": [int(rcid_to_firms.max())],
                "share_firms_in_multi_rcid_groups": [
                    float(rcid_to_firms[rcid_to_firms > 1].sum() / len(firm_map))
                ],
            }
        )
        dup_summary.to_csv(debug_dir / "many_to_one_rcid_summary.csv", index=False)


def _simulate_outcome_policy(adf: pd.DataFrame, emp_col: str, policy: str) -> pd.Series:
    raw_vals = pd.to_numeric(adf[emp_col], errors="coerce")
    raw_vals = raw_vals.where(np.isfinite(raw_vals), np.nan)
    raw_vals = raw_vals.where(raw_vals >= 0, np.nan)

    if policy == "zero_impute":
        rcids_with_positive = set(adf.loc[raw_vals > 0, "rcid"].dropna().unique().tolist())
        vals = raw_vals.copy()
        vals.loc[raw_vals.isna() & adf["rcid"].isin(rcids_with_positive)] = 0.0
        if LOG_OUTCOMES:
            return pd.Series(np.where(vals > 0, np.log(vals), np.where(vals == 0, 0.0, np.nan)), index=adf.index)
        return vals

    if policy == "balanced_panel":
        complete_mask = (
            raw_vals.notna()
            .groupby([adf["foia_firm_uid"], adf["lottery_year"]])
            .transform("all")
        )
        vals = raw_vals.where(complete_mask, np.nan)
        if LOG_OUTCOMES:
            return pd.Series(np.where(vals > 0, np.log(vals), np.nan), index=adf.index)
        return vals

    raise ValueError(f"Unknown policy: {policy}")


def _missingness_policy_diagnostics(adf: pd.DataFrame, outcome: str, debug_dir: Path):
    if not outcome.startswith("out_"):
        return
    emp_col = outcome.replace("out_", "", 1)
    if emp_col not in adf.columns:
        return

    records = []
    heat = {}
    for policy in ["zero_impute", "balanced_panel"]:
        vals = _simulate_outcome_policy(adf, emp_col, policy)
        tmp = adf[["lottery_year", "k"]].copy()
        tmp["_nonmiss"] = vals.notna().astype(int)
        tmp["_value"] = vals
        g = (
            tmp.groupby(["lottery_year", "k"])
            .agg(
                n_rows=("_nonmiss", "size"),
                n_nonmissing=("_nonmiss", "sum"),
                mean_value=("_value", "mean"),
            )
            .reset_index()
            .sort_values(["lottery_year", "k"])
        )
        g["policy"] = policy
        g["missing_rate"] = 1.0 - (g["n_nonmissing"] / g["n_rows"])
        records.append(g)
        heat[policy] = g.pivot(index="lottery_year", columns="k", values="missing_rate")

    compare = pd.concat(records, ignore_index=True)
    compare.to_csv(debug_dir / f"policy_compare_missingness_{outcome}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for i, policy in enumerate(["zero_impute", "balanced_panel"]):
        sns.heatmap(
            heat[policy],
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            cbar=(i == 1),
            ax=axes[i],
            vmin=0,
            vmax=max(0.01, float(np.nanmax(heat[policy].values))),
        )
        axes[i].set_title(f"Missing rate by year x k ({policy})")
        axes[i].set_xlabel("k")
        axes[i].set_ylabel("lottery_year")
    _save_and_show(fig, debug_dir / f"policy_compare_missingness_{outcome}.png")


def _run_sensitivity_grid(adf: pd.DataFrame, outcome: str, debug_dir: Path):
    if outcome not in adf.columns:
        return

    q = min(max(DIAG_TRIM_QUANTILE, 0.5), 0.9999)
    u_cut = adf["U_itau"].abs().quantile(q)
    apps_cut = adf["n_apps"].quantile(q)
    samples = {
        "base": adf,
        f"trim_abs_u_q{q:.3f}": adf[adf["U_itau"].abs() <= u_cut].copy(),
        f"trim_n_apps_q{q:.3f}": adf[adf["n_apps"] <= apps_cut].copy(),
    }
    fe_specs = ["none", "year_only", "firm_only", "firm_year"]

    rows = []
    for sample_name, sdf in samples.items():
        for fe_spec in fe_specs:
            for k in K_LAGS:
                res = run_regression(sdf, outcome, k, fe_spec=fe_spec)
                if res is None:
                    rows.append(
                        {
                            "sample": sample_name,
                            "fe_spec": fe_spec,
                            "k": k,
                            "beta": np.nan,
                            "se": np.nan,
                            "pval": np.nan,
                            "nobs": 0,
                        }
                    )
                else:
                    rows.append(
                        {
                            "sample": sample_name,
                            "fe_spec": fe_spec,
                            "k": k,
                            "beta": float(res.params.get("U_itau", np.nan)),
                            "se": float(res.std_errors.get("U_itau", np.nan)),
                            "pval": float(res.pvalues.get("U_itau", np.nan)),
                            "nobs": int(res.nobs),
                        }
                    )

    grid = pd.DataFrame(rows)
    grid.to_csv(debug_dir / f"sensitivity_grid_{outcome}.csv", index=False)

    base = grid[(grid["sample"] == "base") & grid["beta"].notna()].copy()
    if base.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for fe_spec, sub in base.groupby("fe_spec"):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["beta"], marker="o", label=fe_spec)
        ax.fill_between(sub["k"], sub["beta"] - 1.96 * sub["se"], sub["beta"] + 1.96 * sub["se"], alpha=0.12)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(-0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("k")
    ax.set_ylabel("beta on U_itau")
    ax.set_title(f"Sensitivity by FE spec ({outcome}, base sample)")
    ax.legend(title="FE spec", loc="best")
    _save_and_show(fig, debug_dir / f"sensitivity_grid_plot_{outcome}.png")


def run_deep_dive_diagnostics(
    panel: pd.DataFrame,
    adf: pd.DataFrame,
    step1_stats: dict,
    step3_stats: dict,
    output_dir: Path,
):
    if not DIAG_ENABLED:
        return

    print("\nStep 7: Deep-dive diagnostics")
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    outcome = _resolve_focus_outcome(adf)
    if outcome is None:
        print("  [WARN] No outcome columns available; skipping diagnostics")
        return
    print(f"  focus outcome: {outcome}")

    _write_stage_audit_files(step1_stats, step3_stats, debug_dir)
    _plot_outcome_level_diagnostics(adf, outcome, debug_dir)
    _u_itau_diagnostics(adf, debug_dir)
    _matching_diagnostics(panel, debug_dir)

    if DIAG_COMPARE_POLICIES:
        _missingness_policy_diagnostics(adf, outcome, debug_dir)

    if DIAG_RUN_SENSITIVITY_GRID:
        _run_sensitivity_grid(adf, outcome, debug_dir)

    print(f"  Diagnostics written to: {debug_dir}")


def write_sample_construction_summary(
    step1_stats: dict,
    step2_stats: dict,
    step3_stats: dict,
    results_by_fe: dict,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Save a stage-by-stage sample construction summary to CSV.
    """
    rows = [
        {
            "stage": "01_foia_loaded",
            "n_rows": step1_stats.get("foia_rows_loaded"),
            "n_firms": step1_stats.get("foia_unique_firms_loaded"),
            "n_apps": step1_stats.get("foia_unique_apps_loaded"),
            "n_years": np.nan,
            "notes": "Rows loaded from configured FOIA input file.",
        },
        {
            "stage": "01b_foia_rows_missing_firm_uid",
            "n_rows": step1_stats.get("foia_rows_missing_firm_uid"),
            "n_firms": np.nan,
            "n_apps": np.nan,
            "n_years": np.nan,
            "notes": "Rows missing foia_firm_uid before status filter.",
        },
        {
            "stage": "02_lottery_pool_status_filter",
            "n_rows": step1_stats.get("lottery_pool_rows"),
            "n_firms": step1_stats.get("lottery_pool_unique_firms"),
            "n_apps": step1_stats.get("lottery_pool_unique_apps"),
            "n_years": np.nan,
            "notes": f"Kept status in [{STATUS_SEL}, {STATUS_ELIG}, CREATED].",
        },
        {
            "stage": "03_firm_year_aggregation_pre_test",
            "n_rows": step1_stats.get("panel_firm_years_pre_test"),
            "n_firms": step1_stats.get("panel_firms_pre_test"),
            "n_apps": np.nan,
            "n_years": np.nan,
            "notes": "FOIA panel aggregated to firm x lottery_year.",
        },
        {
            "stage": "04_after_testing_subsample",
            "n_rows": step1_stats.get("panel_firm_years_post_test"),
            "n_firms": step1_stats.get("panel_firms_post_test"),
            "n_apps": np.nan,
            "n_years": np.nan,
            "notes": f"Testing mode={'on' if TESTING_ENABLED else 'off'}.",
        },
        {
            "stage": "05_after_min_apps_filter",
            "n_rows": step1_stats.get("panel_firm_years_post_min_apps"),
            "n_firms": step1_stats.get("panel_firms_post_min_apps"),
            "n_apps": np.nan,
            "n_years": np.nan,
            "notes": (
                f"Kept n_apps >= {MIN_APPS}."
                if MAX_APPS is None
                else f"Kept {MIN_APPS} <= n_apps <= {MAX_APPS}."
            ),
        },
        {
            "stage": "05a_rows_dropped_below_min_apps",
            "n_rows": step1_stats.get("panel_firm_years_dropped_below_min_apps"),
            "n_firms": np.nan,
            "n_apps": np.nan,
            "n_years": np.nan,
            "notes": "Firm-year rows removed by lower app-count threshold.",
        },
        {
            "stage": "05aa_rows_dropped_above_max_apps",
            "n_rows": step1_stats.get("panel_firm_years_dropped_above_max_apps"),
            "n_firms": np.nan,
            "n_apps": np.nan,
            "n_years": np.nan,
            "notes": "Firm-year rows removed by upper app-count threshold.",
        },
        {
            "stage": "05b_after_indiv_sample_restrict",
            "n_rows": step1_stats.get("panel_firm_years_post_indiv_restrict"),
            "n_firms": step1_stats.get("panel_firms_post_indiv_restrict"),
            "n_apps": np.nan,
            "n_years": np.nan,
            "notes": (
                "Restricted to (foia_firm_uid, lottery_year) pairs in individual sample."
                if INDIV_SAMPLE_RESTRICT_ENABLED
                else "indiv_sample_restrict disabled."
            ),
        },
        {
            "stage": "05b_rows_dropped_indiv_restrict",
            "n_rows": step1_stats.get("panel_firm_years_dropped_indiv_restrict"),
            "n_firms": np.nan,
            "n_apps": np.nan,
            "n_years": np.nan,
            "notes": "Firm-years not present in individual merge sample (dropped).",
        },
        {
            "stage": "06_panel_rcid_nonmissing",
            "n_rows": step1_stats.get("panel_rcid_nonmissing_post_min_apps"),
            "n_firms": np.nan,
            "n_apps": np.nan,
            "n_years": np.nan,
            "notes": "Firm-years with non-missing rcid in panel.",
        },
        {
            "stage": "07_headcount_panel",
            "n_rows": step2_stats.get("headcount_rows"),
            "n_firms": step2_stats.get("headcount_firms"),
            "n_apps": np.nan,
            "n_years": step2_stats.get("headcount_years"),
            "notes": (
                f"WRDS headcount (cache={'yes' if step2_stats.get('headcount_loaded_from_cache') else 'no'})."
            ),
        },
        {
            "stage": "08_analysis_panel_pre_rcid_filter",
            "n_rows": step3_stats.get("analysis_rows_pre_rcid_filter"),
            "n_firms": np.nan,
            "n_apps": np.nan,
            "n_years": np.nan,
            "notes": "Long panel (firm-year x k) before rcid handling.",
        },
        {
            "stage": "09_analysis_panel_post_rcid_filter",
            "n_rows": step3_stats.get("analysis_rows_post_rcid_filter"),
            "n_firms": step3_stats.get("analysis_firms_post_rcid_filter"),
            "n_apps": np.nan,
            "n_years": step3_stats.get("analysis_lottery_years_post_rcid_filter"),
            "notes": "Rows after dropping missing-rcid firms.",
        },
        {
            "stage": "09b_analysis_rows_dropped_missing_rcid",
            "n_rows": step3_stats.get("analysis_rcid_missing_rows_dropped"),
            "n_firms": np.nan,
            "n_apps": np.nan,
            "n_years": np.nan,
            "notes": "Rows removed due to missing rcid.",
        },
    ]

    if bool(step1_stats.get("baseline_trim_enabled", False)):
        byear = step1_stats.get("baseline_trim_year")
        byear_txt = "auto" if byear is None else str(int(byear))
        pct_val = step1_stats.get("baseline_trim_pct", np.nan)
        pct_txt = f"{float(pct_val):.3f}" if pd.notna(pct_val) else "NA"
        cut_val = step1_stats.get("baseline_trim_cutoff", np.nan)
        cut_txt = f"{float(cut_val):.3f}" if pd.notna(cut_val) else "NA"
        rows.insert(
            5,
            {
                "stage": "05b_after_baseline_size_trim",
                "n_rows": step1_stats.get("baseline_trim_rows_after"),
                "n_firms": step1_stats.get("baseline_trim_firms_after"),
                "n_apps": np.nan,
                "n_years": np.nan,
                "notes": (
                    f"Dropped top {pct_txt}% firms "
                    f"by {step1_stats.get('baseline_trim_col', BASELINE_SIZE_COL)} "
                    f"at baseline year {byear_txt} "
                    f"(cutoff={cut_txt})."
                ),
            }
        )
        rows.insert(
            6,
            {
                "stage": "05c_baseline_size_trim_dropped_rows",
                "n_rows": step1_stats.get("baseline_trim_rows_dropped"),
                "n_firms": step1_stats.get("baseline_trim_firms_dropped"),
                "n_apps": np.nan,
                "n_years": np.nan,
                "notes": "Rows/firms removed by baseline size trim.",
            }
        )

    if MISSING_OUTCOME_POLICY == "zero_impute":
        for outcome, n_imp in step3_stats.get("analysis_zero_imputed_rows", {}).items():
            rows.append(
                {
                    "stage": f"10_zero_imputed_{outcome}",
                    "n_rows": int(n_imp),
                    "n_firms": np.nan,
                    "n_apps": np.nan,
                    "n_years": np.nan,
                    "notes": "Missing outcome cells set to 0 (rcid has positive headcount in other years).",
                }
            )
    elif MISSING_OUTCOME_POLICY == "balanced_panel":
        for outcome, n_bal in step3_stats.get("analysis_balanced_retained_rows", {}).items():
            if pd.isna(n_bal):
                continue
            rows.append(
                {
                    "stage": f"10_balanced_retained_{outcome}",
                    "n_rows": int(n_bal),
                    "n_firms": np.nan,
                    "n_apps": np.nan,
                    "n_years": np.nan,
                    "notes": "Rows retained after enforcing balanced panel for this outcome.",
                }
            )

    for outcome, n_nonmiss in step3_stats.get("analysis_outcome_nonmissing", {}).items():
        rows.append(
            {
                "stage": f"11_nonmissing_{outcome}",
                "n_rows": int(n_nonmiss),
                "n_firms": np.nan,
                "n_apps": np.nan,
                "n_years": np.nan,
                "notes": f"Non-missing outcome rows for {outcome}.",
            }
        )

    for fe_spec, fe_results in results_by_fe.items():
        for outcome, k_map in fe_results.items():
            for k in K_LAGS:
                res = k_map.get(k)
                rows.append(
                    {
                        "stage": f"12_reg_nobs_{fe_spec}_{outcome}_k{k:+d}",
                        "n_rows": int(res.nobs) if res is not None else 0,
                        "n_firms": np.nan,
                        "n_apps": np.nan,
                        "n_years": np.nan,
                        "notes": "PanelOLS nobs after model-specific drops.",
                    }
                )

    sdf = pd.DataFrame(rows)
    out = output_dir / "tables" / "sample_construction.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sdf.to_csv(out, index=False)
    print(f"  Saved: {out}")

    print("  Sample construction (key stages):")
    for stage in rows[:11]:
        n_val = stage["n_rows"]
        n_txt = f"{int(n_val):,}" if pd.notna(n_val) else "NA"
        print(f"    {stage['stage']}: {n_txt}")

    return sdf


###############################################################################
# MAIN
###############################################################################

def main():
    t_total = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: H-1B firm-year panel ---
    panel, step1_stats = build_hib_panel(FOIA_PARQUET, MERGE_PARQUET, return_stats=True)

    # --- Step 2: Revelio headcounts from WRDS ---
    rcids = panel["rcid"].dropna().unique().tolist()
    hc, step2_stats = query_headcounts_wrds(rcids, OUTPUT_DIR, return_stats=True)

    # --- Step 2b: Optional baseline-size trimming ---
    panel, trim_stats = trim_top_baseline_size_firms(panel, hc, return_stats=True)
    step1_stats.update(trim_stats)

    # --- Step 3: Analysis panel ---
    emp_cols = ["emp_all", "emp_tenure"]
    if "emp_h1b_occ" in hc.columns:
        emp_cols.append("emp_h1b_occ")

    adf, step3_stats = build_analysis_panel(panel, hc, OUTPUT_DIR, return_stats=True)

    # --- Step 4: Regressions ---
    results = run_all_regressions(adf, emp_cols, FE_SPECS)

    # --- Step 5: Tables and plots ---
    print("Step 5: Generating outputs...")
    for i, fe_spec in enumerate(FE_SPECS):
        fe_results = results.get(fe_spec, {})
        write_legacy = (i == 0)
        make_regression_table(
            fe_results, emp_cols, OUTPUT_DIR, fe_spec=fe_spec, write_legacy=write_legacy
        )
        make_event_study_plot(
            fe_results, emp_cols, OUTPUT_DIR, fe_spec=fe_spec, write_legacy=write_legacy
        )
    make_size_decile_event_study_plot(adf, OUTPUT_DIR)
    print_heterogeneity_tables(adf)
    run_deep_dive_diagnostics(panel, adf, step1_stats, step3_stats, OUTPUT_DIR)
    write_sample_construction_summary(step1_stats, step2_stats, step3_stats, results, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"Done. Total time: {time.time()-t_total:.1f}s")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
