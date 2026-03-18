# File Description: Clean regression analysis for H-1B lottery outcomes
# Author: Amy Kim
# Date Created: Mar 2026
#
# Runs weighted panel regressions for each merged parquet variant (baseline, mult2/4/6,
# prefilt, strict) with three FE specifications: none, firm×year, firm+year.
# Outputs regression tables (LaTeX) and optional balance tables to output/reg/tables/.

import os
import sys
import time
import warnings
from pathlib import Path

# Ensure stdout is line-buffered so nohup logs update in real time (no-op in IPython)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from linearmodels.panel import PanelOLS
from linearmodels.panel import compare

# ---------------------------------------------------------------------------
# Path setup — resolve code root and config dirs
# ---------------------------------------------------------------------------
if "__file__" in globals():
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    _THIS_DIR = os.path.join(os.getcwd(), "04_analysis")

_CODE_DIR = os.path.dirname(_THIS_DIR)
sys.path.append(_CODE_DIR)
sys.path.append(_THIS_DIR)
sys.path.append(os.path.join(_CODE_DIR, "03_indiv_merge"))

import indiv_merge_config as icfg
from config import root

# ---------------------------------------------------------------------------
# Load reg config
# ---------------------------------------------------------------------------
_REG_CONFIG_PATH = Path(_CODE_DIR) / "configs" / "reg.yaml"


def _load_reg_config() -> dict:
    raw = yaml.safe_load(_REG_CONFIG_PATH.read_text()) or {}
    # Expand {root} in output_dir
    if "output_dir" in raw:
        raw["output_dir"] = raw["output_dir"].replace("{root}", str(root))
    return raw


def _normalize_outcome_groups(raw_groups: dict) -> dict:
    """
    Normalize config outcomes into a uniform list of specs per group.

    Supported YAML entries:
      - plain string outcome name
      - dict with:
          name: regression/table label
          source_outcome: column in df_app to regress on
          subset_var: optional column used to subset the sample
          subset_value: value required in subset_var (default = 1)
    """
    normalized = {}

    for group_name, entries in (raw_groups or {}).items():
        normalized[group_name] = []
        for entry in entries or []:
            if isinstance(entry, str):
                normalized[group_name].append({
                    "name": entry,
                    "source_outcome": entry,
                    "subset_var": None,
                    "subset_value": None,
                })
                continue

            if not isinstance(entry, dict):
                raise TypeError(
                    f"Outcome entry in group '{group_name}' must be a string or dict, got {type(entry)}"
                )

            source_outcome = entry.get("source_outcome") or entry.get("outcome") or entry.get("name")
            name = entry.get("name") or source_outcome
            if not source_outcome or not name:
                raise ValueError(
                    f"Outcome entry in group '{group_name}' must define 'name' and/or 'source_outcome'"
                )

            subset_var = entry.get("subset_var")
            subset_value = entry.get("subset_value", 1 if subset_var is not None else None)
            normalized[group_name].append({
                "name": name,
                "source_outcome": source_outcome,
                "subset_var": subset_var,
                "subset_value": subset_value,
            })

    return normalized


CFG = _load_reg_config()
TESTING = CFG.get("testing", {})
TESTING_ENABLED = bool(TESTING.get("enabled", False))
TESTING_N_APPS = int(TESTING.get("n_apps", 500))
TESTING_SEED = int(TESTING.get("seed", 42))
OUTPUT_DIR = Path(CFG["output_dir"])
PRIMARY_FE_SPEC = CFG.get("primary_fe_spec", "firm_year")
BALANCE_TABLE = bool(CFG.get("balance_table", True))
HET_EFFECTS = bool(CFG.get("het_effects", True))
IN_US_HET_EFFECTS = bool(CFG.get("in_us_het_effects", False))
RUN_UPDATING_DIAGNOSTICS = bool(CFG.get("run_updating_diagnostics", True))
RUN_MATCH_QUALITY_CHECK = bool(CFG.get("run_match_quality_check", False))
OUTCOME_GROUPS: dict = _normalize_outcome_groups(CFG.get("outcomes", {}))
FE_SPECS: list = CFG.get("fe_specs", ["firm_year"])
VARIANTS: list = CFG.get("variants", [])
USE_OPTIMAL_DEDUP: bool = bool(CFG.get("use_optimal_dedup", False))
INCLUDE_WINNER_ADE_INTERACTION: bool = bool(CFG.get("include_winner_ade_interaction", True))
ADE_AGG: str = CFG.get("ade_agg", "max")  # "max" or "weighted_avg"
USE_PYFIXEST: bool = bool(CFG.get("use_pyfixest", True))
EVENT_TIME_CFG: dict = CFG.get("event_time_graphs", {})

# Try importing pyfixest once at module load; fall back to linearmodels gracefully.
_PYFIXEST_AVAILABLE = False
if USE_PYFIXEST:
    try:
        import pyfixest as pf
        _PYFIXEST_AVAILABLE = True
    except ImportError:
        pass

print(f"=== reg_new.py ===")
print(f"run_tag:         {CFG.get('run_tag', 'feb2026')}")
print(f"use_optimal_dedup: {USE_OPTIMAL_DEDUP}")
print(f"testing:         {'ENABLED (n=' + str(TESTING_N_APPS) + ')' if TESTING_ENABLED else 'disabled'}")
print(f"variants: {[v['name'] for v in VARIANTS]}")
print(f"fe_specs: {FE_SPECS}")
print(f"include_winner_ade_interaction: {INCLUDE_WINNER_ADE_INTERACTION}")
print(f"ade_agg:         {ADE_AGG}")
print(f"regression backend: {'pyfixest' if _PYFIXEST_AVAILABLE else 'linearmodels'}")
print(f"event_time_graphs: {'enabled' if EVENT_TIME_CFG.get('enabled') else 'disabled'}")
print()


###############################################################################
# HELPER FUNCTIONS
###############################################################################

def _with_firm_key(df: pd.DataFrame) -> pd.DataFrame:
    """Create firm_key from foia_firm_uid only (never FEIN)."""
    out = df.copy()
    if "foia_firm_uid" in out.columns:
        uid = out["foia_firm_uid"].astype(str).str.strip()
        uid = uid.where(~uid.isin(["", "None", "nan", "NaN"]))
        out["firm_key"] = uid.fillna("")
    else:
        out["firm_key"] = ""
    return out


def _weighted_average(values: pd.Series, weights: pd.Series):
    """Weighted average with NaN handling."""
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    w = weights[mask]
    denom = w.sum()
    if denom == 0:
        return np.nan
    return (values[mask] * w).sum() / denom


###############################################################################
# DATA LOADING AND CLEANING
###############################################################################

def load_and_clean(parquet_path: str) -> pd.DataFrame:
    """
    Load raw merged parquet and create all outcome and control variables.
    Returns a row-level DataFrame (multiple rows per applicant = one per candidate match).
    """
    t0 = time.time()
    print(f"  Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} rows, {df['foia_indiv_id'].nunique():,} unique applicants "
          f"({time.time()-t0:.1f}s)")

    df = _with_firm_key(df)

    # --- Treatment indicators ---
    df["winner"] = (df["status_type"] == "SELECTED").astype(int)

    # ADE: valid if ade_ind==1 AND ade_year is null or before lottery year
    df["ade"] = np.where(
        df["ade_ind"].notna() & (df["ade_ind"] == 1),
        np.where(
            df["ade_year"].isna() | (df["ade_year"] <= df["lottery_year"].astype(int) - 1),
            1, 0
        ),
        0
    )

    # --- Demographics ---
    df["age"] = df["lottery_year"].astype(int) - df["yob"].astype(int)

    # --- Time since graduation ---
    df["graddiff"] = df["lottery_year"].astype(int) - 1 - df["last_grad_year"]
    # Weighted average graddiff at applicant level (for heterogeneous effects)
    df["graddiff_agg"] = (
        df.groupby("foia_indiv_id")[["graddiff", "weight_norm"]]
        .apply(lambda g: _weighted_average(g["graddiff"], g["weight_norm"]))
        .reindex(df["foia_indiv_id"])
        .values
    )

    # Outcome variables come directly from the merged parquet (built in indiv_merge.py).

    # --- Profile location indicators (static, from user_location/user_country) ---
    if "user_country" in df.columns:
        us_country_codes = {"US", "USA", "United States"}
        df["profile_in_us"] = df["user_country"].isin(us_country_codes).astype(int)
        df["profile_in_home_country"] = (
            df["user_country"].str.upper() == df["foia_country"].str.upper()
        ).fillna(False).astype(int)
        df["profile_non_us"] = (
            (df["profile_in_us"] == 0) & df["user_country"].notna()
        ).astype(int)
        df["profile_loc_null"] = df["user_country"].isna().astype(int)

    # --- Panel index columns ---
    df["lottery_year"] = df["lottery_year"].astype(int)
    df["firm_year_fe"] = df["firm_key"].astype(str) + "_" + df["lottery_year"].astype(str)

    return df


def collapse_to_app_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse row-level data to one row per applicant (unique at foia_indiv_id level).

    Outcomes are aggregated via weighted average across ALL candidate rows for each
    applicant. The firm_key used for FEs is taken from the firm with the highest
    total weight_norm for that applicant.
    """
    # Candidate outcomes to aggregate (only include if present in data)
    candidate_outcomes = [
        # WORK STATUS (exhaustive: still_at_firm + diff_firm + other = 1)
        "still_at_firm1", "still_at_firm2", "still_at_firm3",
        "diff_firm1", "diff_firm2", "diff_firm3",           # active position at different firm
        "new_diff_firm1", "new_diff_firm2", "new_diff_firm3",   # diff_firm, position started post-lottery
        "old_diff_firm1", "old_diff_firm2", "old_diff_firm3",   # diff_firm, position started pre-lottery
        "other1", "other2", "other3",                       # no active position anywhere
        # CONDITIONAL ON still_at_firm (exhaustive within still_at_firm=1)
        "same_firm_new_position1", "same_firm_new_position2", "same_firm_new_position3",
        "same_pos_null_end1", "same_pos_null_end2",         # same position, null enddate (imputed active)
        "same_pos_nonnull_end1", "same_pos_nonnull_end2",   # same position, explicit non-null enddate
        # LOCATION BASELINE (combined position + education; in_us + in_home_country + non_us_non_home + loc_null = 1)
        "in_us1", "in_us2", "in_us3",
        "in_home_country1", "in_home_country2", "in_home_country3",
        "non_us_non_home1", "non_us_non_home2", "non_us_non_home3",
        # EDUCATION (exhaustive: new_educ + continuing_educ + no_educations = 1)
        "new_educ1", "new_educ2", "new_educ3",
        "continuing_educ1", "continuing_educ2", "continuing_educ3",
        # COMPENSATION
        "agg_compensation1", "agg_compensation2", "agg_compensation3",
        # NON-UPDATING BIAS DIAGNOSTICS (weighted avg across candidate matches)
        "updatediff",          # months from pre-lottery ref to Revelio's profile refresh date
        "updatediff_activity", # months from pre-lottery ref to most recent position/educ startdate
        "frac_null_enddate1", "frac_null_enddate2",
        "null_enddate_stayer1", "null_enddate_stayer2",
    ]
    outcomes = [v for v in candidate_outcomes if v in df.columns]

    # Applicant-level constants (same value across all rows for a given applicant)
    id_cols = [
        "foia_indiv_id", "female_ind", "yob", "age",
        "lottery_year", "foia_country", "winner",
        "n_apps", "n_unique_country", "high_rep_emp_ind", "no_rep_emp_ind",
        "graddiff_agg",
    ]
    id_cols = [c for c in id_cols if c in df.columns]

    # ade is binarized separately after the main collapse (see below).
    has_ade = "ade" in df.columns
    weighted_avg_cols = []  # ade excluded from weighted avg loop; handled below

    # Step 1: Identify best firm per applicant (highest total weight_norm)
    firm_weights = (
        df.groupby(["foia_indiv_id", "firm_key"])["weight_norm"]
        .sum()
        .reset_index()
        .sort_values("weight_norm", ascending=False)
        .drop_duplicates(subset="foia_indiv_id")
        .rename(columns={"weight_norm": "_best_firm_weight"})
        [["foia_indiv_id", "firm_key"]]
    )

    # Step 2: Collapse outcomes to applicant level via weighted average.
    # Vectorized: for each outcome col, compute (col * weight_norm) then groupby-sum
    # and divide by sum of weights. This avoids per-group df.loc indexing which is
    # O(n_applicants * n_outcomes) label lookups and is extremely slow.
    wa_cols = outcomes + weighted_avg_cols
    g = df.groupby("foia_indiv_id")

    # Sum of weights per applicant (also needed for the division)
    weight_sum = g["weight_norm"].sum().rename("weight_norm")

    # For each outcome, compute weighted sum of non-null values and sum of weights
    # over non-null observations only (so NaNs don't dilute the average).
    wa_results = {}
    for var in wa_cols:
        mask_col = f"_w_{var}"
        val_col  = f"_wv_{var}"
        notna = df[var].notna()
        df[mask_col] = np.where(notna, df["weight_norm"], 0.0)
        df[val_col]  = np.where(notna, df[var] * df["weight_norm"], 0.0)
        wsum  = df.groupby("foia_indiv_id")[mask_col].sum()
        wvsum = df.groupby("foia_indiv_id")[val_col].sum()
        wa_results[var] = (wvsum / wsum).where(wsum > 0)   # NaN when all obs are NaN
        df.drop(columns=[mask_col, val_col], inplace=True)

    app_df = pd.DataFrame(wa_results)
    app_df.index.name = "foia_indiv_id"
    app_df = app_df.reset_index()
    app_df = app_df.merge(weight_sum.reset_index(), on="foia_indiv_id", how="left")

    # ade binarization: always produces a 0/1 column.
    #   "max"       → 1 if any candidate is ADE-eligible
    #   "threshold" → 1 if weighted avg of ADE >= 0.5
    if has_ade:
        if ADE_AGG == "max":
            ade_series = df.groupby("foia_indiv_id")["ade"].max()
        else:  # "threshold": weighted avg >= 0.5
            notna = df["ade"].notna()
            df["_w_ade"]  = np.where(notna, df["weight_norm"], 0.0)
            df["_wv_ade"] = np.where(notna, df["ade"] * df["weight_norm"], 0.0)
            wsum  = df.groupby("foia_indiv_id")["_w_ade"].sum()
            wvsum = df.groupby("foia_indiv_id")["_wv_ade"].sum()
            df.drop(columns=["_w_ade", "_wv_ade"], inplace=True)
            ade_series = ((wvsum / wsum.where(wsum > 0)) >= 0.5).astype(float)
        ade_series = ade_series.rename("ade")
        app_df = app_df.merge(ade_series.reset_index(), on="foia_indiv_id", how="left")
        print(f"  [ade_agg={ADE_AGG}] ade=1: {int(app_df['ade'].sum()):,}  "
              f"ade=0: {int((app_df['ade'] == 0).sum()):,}")

    # Merge applicant-level constants back in (take first value per applicant)
    id_cols_no_id = [c for c in id_cols if c != "foia_indiv_id"]
    id_vals = df.groupby("foia_indiv_id")[id_cols_no_id].first().reset_index()

    # Diagnostic: flag any id_cols that actually vary across rows for the same applicant.
    # These should be constant (all come from FOIA table keyed by foia_indiv_id), but
    # if any vary, .first() silently picks an arbitrary value — useful to know.
    _vary_check = df.groupby("foia_indiv_id")[id_cols_no_id].nunique()
    _varying = [c for c in id_cols_no_id if (_vary_check[c] > 1).any()]
    if _varying:
        print(f"  [WARN] id_cols that vary per applicant (using .first()): {_varying}")
        for c in _varying:
            n_bad = int((_vary_check[c] > 1).sum())
            print(f"    {c}: {n_bad:,} applicants with >1 unique value")
    else:
        print("  [OK] All id_cols are constant per applicant")

    app_df = app_df.merge(id_vals, on="foia_indiv_id", how="left")

    # Step 3: Attach best firm_key
    app_df = app_df.merge(firm_weights, on="foia_indiv_id", how="left")

    # Derived columns for regression
    app_df["const"] = 1
    app_df["firm_year_fe"] = app_df["firm_key"].astype(str) + "_" + app_df["lottery_year"].astype(str)

    assert app_df["foia_indiv_id"].nunique() == len(app_df), \
        "collapse_to_app_level: result is not unique at foia_indiv_id level"

    print(f"  Collapsed to {len(app_df):,} app-level rows (one per applicant)")
    return app_df


def maybe_sample(df: pd.DataFrame) -> pd.DataFrame:
    """If testing mode is on, subsample to TESTING_N_APPS unique applicants."""
    if not TESTING_ENABLED:
        return df
    rng = np.random.default_rng(TESTING_SEED)
    unique_ids = df["foia_indiv_id"].unique()
    sampled_ids = rng.choice(unique_ids, size=min(TESTING_N_APPS, len(unique_ids)), replace=False)
    out = df[df["foia_indiv_id"].isin(sampled_ids)].copy()
    print(f"  [TEST] Sampled to {len(out):,} rows, {out['foia_indiv_id'].nunique():,} applicants")
    return out


###############################################################################
# PANEL REGRESSION
###############################################################################

class _RegressionResult:
    """
    Thin adapter that normalises pyfixest or linearmodels results to a uniform interface.

    Attributes:
        params       pd.Series  — coefficient estimates, indexed by variable name
        std_errors   pd.Series  — standard errors
        pvalues      pd.Series  — p-values
        nobs         int        — number of observations
        model        object     — linearmodels model object (None for pyfixest path)
    """
    def __init__(self, params, std_errors, pvalues, nobs, model=None):
        self.params = params
        self.std_errors = std_errors
        self.pvalues = pvalues
        self.nobs = nobs
        self.model = model  # only populated on the linearmodels path

    @classmethod
    def from_linearmodels(cls, res):
        obj = cls(res.params, res.std_errors, res.pvalues, int(res.nobs), model=res.model)
        obj._raw_lm = res  # preserve raw result for linearmodels.compare()
        return obj

    @classmethod
    def from_pyfixest(cls, fit):
        tidy = fit.tidy()
        # pyfixest versions differ: older ones return "Coefficient" as a column;
        # newer ones return it as the index, or use "term" as the column name.
        if "Coefficient" in tidy.columns:
            tidy = tidy.set_index("Coefficient")
        elif "term" in tidy.columns:
            tidy = tidy.set_index("term")
        # else: coefficient names are already the index — use as-is

        # Column names also vary across versions; try known aliases.
        def _col(candidates):
            for c in candidates:
                if c in tidy.columns:
                    return tidy[c]
            raise ValueError(f"None of {candidates} found in pyfixest tidy() columns: {list(tidy.columns)}")

        obj = cls(
            params=_col(["Estimate", "coef", "estimate"]),
            std_errors=_col(["Std. Error", "se", "std_error"]),
            pvalues=_col(["Pr(>|t|)", "pvalue", "p_value", "Pr(>|z|)"]),
            nobs=int(fit._N),
            model=None,
        )
        obj._raw_pf = fit  # preserve for pf.etable() display
        return obj


def _build_panel_df(df: pd.DataFrame, fe_spec: str) -> pd.DataFrame:
    """
    Set the MultiIndex required by linearmodels.PanelOLS.

    - none / firm_year : entity = firm_year_fe,  time = foia_indiv_id (int)
    - firm_plus_year   : entity = firm_key,       time = foia_indiv_id (int)
      year dummies are added as explicit columns and included in the formula.
    """
    out = df.copy()

    # Convert foia_indiv_id to integer codes for PanelOLS time dimension
    out["_indiv_int"] = pd.Categorical(out["foia_indiv_id"]).codes

    if fe_spec in ("none", "firm_year"):
        return out.set_index(["firm_year_fe", "_indiv_int"])
    elif fe_spec == "firm_plus_year":
        # Add year dummies (drop first for identification)
        years = sorted(out["lottery_year"].unique())[1:]  # skip first year (base)
        for yr in years:
            out[f"yr_{yr}"] = (out["lottery_year"] == yr).astype(int)
        return out.set_index(["firm_key", "_indiv_int"])
    else:
        raise ValueError(f"Unknown fe_spec: {fe_spec}")


def _fe_formula(
    outcome: str,
    fe_spec: str,
    df: pd.DataFrame,
    include_interaction: bool = True,
) -> str:
    """Build the PanelOLS formula string for a given FE spec."""
    if include_interaction:
        base = f"{outcome} ~ winner + ade + winner:ade"
    else:
        base = f"{outcome} ~ winner + ade"
    if fe_spec == "none":
        return base + " + 1"
    elif fe_spec == "firm_year":
        return base + " + EntityEffects"
    elif fe_spec == "firm_plus_year":
        # EntityEffects = firm FEs; explicit year dummies for year FEs
        years = sorted(df["lottery_year"].unique())[1:]
        year_terms = " + ".join(f"yr_{yr}" for yr in years)
        return base + f" + EntityEffects + {year_terms}"
    else:
        raise ValueError(f"Unknown fe_spec: {fe_spec}")


def _run_pyfixest(
    sub: pd.DataFrame,
    reg_outcome: str,
    fe_spec: str,
    include_interaction: bool,
) -> "_RegressionResult | None":
    """
    Run one regression via pyfixest.feols().

    Uses | separator for FEs; CRV1 clustering at firm level; fixef_rm='singleton'
    handles singleton entity removal automatically inside pyfixest.
    """
    rhs = "winner + ade + winner:ade" if include_interaction else "winner + ade"
    fe_map = {
        "none":           "",
        "firm_year":      " | firm_year_fe",
        "firm_plus_year": " | firm_key + lottery_year",
    }
    fml = f"{reg_outcome} ~ {rhs}{fe_map[fe_spec]}"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyfixest")
            warnings.filterwarnings("ignore", message=".*singleton.*", category=UserWarning)
            fit = pf.feols(
                fml,
                data=sub,
                vcov={"CRV1": "firm_key"},
                weights="weight_norm",
                fixef_rm="singleton",
            )
        return _RegressionResult.from_pyfixest(fit)
    except Exception as e:
        print(f"    [ERR pyfixest] {reg_outcome} ({fe_spec}): {e}")
        return None


def _run_linearmodels(
    sub: pd.DataFrame,
    reg_outcome: str,
    fe_spec: str,
    include_interaction: bool,
) -> "_RegressionResult | None":
    """Run one regression via linearmodels.PanelOLS (original backend)."""
    # Add year dummies before building panel index (firm_plus_year only)
    if fe_spec == "firm_plus_year":
        years = sorted(sub["lottery_year"].unique())[1:]
        for yr in years:
            sub[f"yr_{yr}"] = (sub["lottery_year"] == yr).astype(int)

    panel_df = _build_panel_df(sub, fe_spec)
    formula  = _fe_formula(reg_outcome, fe_spec, sub, include_interaction=include_interaction)

    try:
        model = PanelOLS.from_formula(
            formula,
            data=panel_df,
            weights=panel_df["weight_norm"],
        )
        if fe_spec == "firm_plus_year":
            result = model.fit(cov_type="clustered", cluster_entity=True)
        else:
            result = model.fit(cov_type="clustered", clusters=panel_df["firm_key"])
        return _RegressionResult.from_linearmodels(result)
    except Exception as e:
        print(f"    [ERR linearmodels] {reg_outcome} ({fe_spec}): {e}")
        return None


def _pre_build_spec_panel(df_app: pd.DataFrame, fe_spec: str) -> dict:
    """
    Pre-compute singleton mask once per (df_app, fe_spec) to avoid repeated
    groupby().transform() calls inside run_regression() for every outcome.

    Returns a dict with key 'non_singleton_mask': a pd.Series[bool] aligned to
    df_app's index, True for rows NOT in singleton entities.
    For fe_spec=="none" the mask is None (singletons don't matter without entity FEs).
    """
    if fe_spec == "none":
        return {"non_singleton_mask": None}
    entity_col = "firm_key" if fe_spec == "firm_plus_year" else "firm_year_fe"
    counts = df_app.groupby(entity_col)[entity_col].transform("count")
    mask = counts > 1
    n_singleton_rows = int((~mask).sum())
    if n_singleton_rows:
        print(f"    [pre-build {fe_spec}] {n_singleton_rows:,} singleton rows pre-flagged")
    return {"non_singleton_mask": mask}


def run_regression(
    df: pd.DataFrame,
    outcome_spec: dict,
    fe_spec: str,
    include_interaction: bool = True,
    pre_built_panel: dict | None = None,
) -> "_RegressionResult | None":
    """
    Run a single weighted regression (pyfixest or linearmodels).

    Parameters
    ----------
    df                  applicant-level DataFrame (post-collapse, one row per applicant)
    outcome_spec        normalized outcome dict: name, source_outcome, subset_var, subset_value
    fe_spec             one of "none", "firm_year", "firm_plus_year"
    include_interaction if False, omit winner:ade from formula
    pre_built_panel     optional dict from _pre_build_spec_panel(); when provided,
                        the singleton mask is reused instead of recomputed (speed-up)

    Returns _RegressionResult or None if outcome missing / too few observations.
    """
    outcome_name   = outcome_spec["name"]
    source_outcome = outcome_spec.get("source_outcome", outcome_name)
    subset_var     = outcome_spec.get("subset_var")
    subset_value   = outcome_spec.get("subset_value")

    if source_outcome not in df.columns or df[source_outcome].isna().all():
        return None

    # --- Build working subset ---
    keep_cols = [source_outcome, "winner", "ade", "weight_norm", "lottery_year",
                 "firm_key", "firm_year_fe", "foia_indiv_id"]
    if subset_var is not None:
        keep_cols.append(subset_var)
    missing_cols = [c for c in keep_cols if c not in df.columns]
    if missing_cols:
        print(f"    [SKIP] {outcome_name} ({fe_spec}): missing columns {missing_cols}")
        return None

    sub = df[keep_cols].copy()
    if subset_var is not None:
        sub = sub[sub[subset_var] == subset_value].copy()
    sub = sub.dropna(subset=[source_outcome, "winner", "ade", "weight_norm"])

    reg_outcome = source_outcome
    if outcome_name != source_outcome:
        sub[outcome_name] = sub[source_outcome]
        reg_outcome = outcome_name

    if len(sub) < 10:
        print(f"    [SKIP] {outcome_name} ({fe_spec}): too few obs ({len(sub)})")
        return None

    # --- Singleton dropping ---
    # For pyfixest, fixef_rm='singleton' handles this internally; skip here.
    # For linearmodels, use pre-computed mask if available (no subset) else recompute.
    if not _PYFIXEST_AVAILABLE and fe_spec != "none":
        if pre_built_panel is not None and subset_var is None:
            # Reuse pre-computed mask (aligned to df's original index)
            mask = pre_built_panel.get("non_singleton_mask")
            if mask is not None:
                n_before = len(sub)
                sub = sub[mask.reindex(sub.index, fill_value=False)].copy()
                if len(sub) < n_before:
                    print(f"    [INFO] {outcome_name} ({fe_spec}): dropped "
                          f"{n_before - len(sub):,} singleton rows (pre-built mask)")
        else:
            # Recompute singleton mask for this subset
            entity_col = "firm_key" if fe_spec == "firm_plus_year" else "firm_year_fe"
            entity_counts = sub.groupby(entity_col)[entity_col].transform("count")
            n_before = len(sub)
            sub = sub[entity_counts > 1].copy()
            if len(sub) < n_before:
                print(f"    [INFO] {outcome_name} ({fe_spec}): dropped "
                      f"{n_before - len(sub):,} singleton {entity_col}s")
        if len(sub) < 10:
            print(f"    [SKIP] {outcome_name} ({fe_spec}): too few obs after singletons ({len(sub)})")
            return None

    # --- Dispatch to backend ---
    if _PYFIXEST_AVAILABLE:
        result = _run_pyfixest(sub, reg_outcome, fe_spec, include_interaction)
    else:
        result = _run_linearmodels(sub, reg_outcome, fe_spec, include_interaction)

    # Attach ctrl_mean (loser mean of outcome) while sub is still in scope.
    if result is not None and "winner" in sub.columns:
        ctrl_mask = sub["winner"] == 0
        result.ctrl_mean = float(sub.loc[ctrl_mask, reg_outcome].mean()) if ctrl_mask.any() else None
    return result


def run_all_outcomes(
    df_app: pd.DataFrame,
    outcome_groups: dict,
    fe_specs: list,
    include_interaction: bool = True,
) -> dict:
    """
    Run regressions for all outcomes × all FE specs.

    Pre-builds singleton mask once per spec (avoids redundant groupby per outcome).
    Returns nested dict: {fe_spec: {group_name: {outcome: result_or_None}}}.
    """
    results = {spec: {} for spec in fe_specs}

    for spec in fe_specs:
        print(f"  FE spec: {spec}")
        # Pre-compute singleton mask once for this spec (speed-up for linearmodels path)
        pre_built = _pre_build_spec_panel(df_app, spec)

        for group, outcome_specs in outcome_groups.items():
            results[spec][group] = {}
            for outcome_spec in outcome_specs:
                outcome_name = outcome_spec["name"]
                res = run_regression(
                    df_app, outcome_spec, spec,
                    include_interaction=include_interaction,
                    pre_built_panel=pre_built,
                )
                results[spec][group][outcome_name] = res
                if res is not None:
                    coef = res.params.get("winner", np.nan)
                    se   = res.std_errors.get("winner", np.nan)
                    n    = int(res.nobs)
                    if include_interaction:
                        coef_int = res.params.get("winner:ade", np.nan)
                        se_int   = res.std_errors.get("winner:ade", np.nan)
                        print(f"    {outcome_name:30s}  winner={coef:+.4f} ({se:.4f})  "
                              f"winner:ade={coef_int:+.4f} ({se_int:.4f})  N={n:,}")
                    else:
                        print(f"    {outcome_name:30s}  winner={coef:+.4f} ({se:.4f})  N={n:,}")

    return results


###############################################################################
# TABLE FORMATTING
###############################################################################

def _coef_row(results: list, var: str) -> tuple[list, list]:
    """Return (coef strings with stars, SE strings) for a given variable."""
    coefs, ses = [], []
    for res in results:
        if res is None or var not in res.params:
            coefs.append("")
            ses.append("")
            continue
        b = res.params[var]
        se = res.std_errors[var]
        p = res.pvalues[var]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        star_str = f"\\sym{{{stars}}}" if stars else ""
        sign = "$-$" if b < 0 else ""
        coefs.append(f"{sign}{abs(b):.4f}{star_str}")
        ses.append(f"({se:.4f})")
    return coefs, ses


def make_regression_table(
    results: list,
    col_labels: list,
    row_vars: list = None,
    title: str = "",
    verbose: bool = True,
    include_interaction: bool = True,
) -> str:
    """
    Format a list of regression results into a LaTeX table string.

    Parameters
    ----------
    results             list of _RegressionResult (None entries produce blank cells)
    col_labels          column header strings
    row_vars            list of coefficient names to display; defaults to
                        ['winner', 'ade', 'winner:ade'] when include_interaction=True,
                        else ['winner', 'ade']
    title               optional title comment at top of LaTeX
    verbose             if True, print linearmodels compare() table to console
                        (only works on the linearmodels path; skipped for pyfixest)
    include_interaction whether winner:ade was included in the regressions
    """
    if row_vars is None:
        row_vars = ["winner", "ade", "winner:ade"] if include_interaction else ["winner", "ade"]

    valid = {lab: res for lab, res in zip(col_labels, results) if res is not None}

    if verbose and valid:
        # pyfixest path: use pf.etable() for a side-by-side markdown summary
        pf_fits = [res._raw_pf for res in valid.values() if hasattr(res, "_raw_pf")]
        if pf_fits and _PYFIXEST_AVAILABLE:
            try:
                print(pf.etable(pf_fits, type="md",
                                model_heads=list(valid.keys()),
                                keep=["winner", "ade", "winner:ade"]))
            except Exception as e:
                print(f"  [etable warn] {e}")
        else:
            # linearmodels path fallback
            lm_valid = {lab: res._raw_lm for lab, res in valid.items()
                        if res.model is not None and hasattr(res, "_raw_lm")}
            if lm_valid:
                print(compare(lm_valid, stars=True, precision="std_errors"))

    ncols = len(results)
    latex = f"% {title}\n" if title else ""
    latex += "\\begin{tabular}{l" + "c" * ncols + "}\n"
    latex += "\\toprule\n"
    latex += "& " + " & ".join(col_labels) + " \\\\\n"
    latex += "& (" + ") & (".join(str(i) for i in range(1, ncols + 1)) + ") \\\\\n"
    latex += "\\midrule\n"

    for var in row_vars:
        coefs, ses = _coef_row(results, var)
        latex += var + " & " + " & ".join(coefs) + " \\\\\n"
        latex += " & " + " & ".join(ses) + " \\\\\n"

    latex += "\\midrule\n"

    ctrl_means = []
    for res in results:
        if res is None:
            ctrl_means.append("")
        elif hasattr(res, "ctrl_mean") and res.ctrl_mean is not None:
            ctrl_means.append(f"{res.ctrl_mean:.3f}")
        elif res.model is not None:
            # linearmodels fallback (no ctrl_mean attribute)
            dv_series = res.model.dependent.dataframe.squeeze()
            exog_df = res.model.exog.dataframe
            if "winner" in exog_df.columns:
                ctrl_means.append(f"{dv_series[exog_df['winner'] == 0].mean():.3f}")
            else:
                ctrl_means.append("")
        else:
            ctrl_means.append("")
    latex += "\\textbf{Control mean} & " + " & ".join(ctrl_means) + " \\\\\n"

    obs_strs = [str(int(r.nobs)) if r is not None else "" for r in results]
    latex += "\\textbf{N} & " + " & ".join(obs_strs) + " \\\\\n"

    if verbose and any(m != "" for m in ctrl_means):
        print("Control mean (losers): " + "  ".join(
            f"{lab}={m}" for lab, m in zip(col_labels, ctrl_means) if m != ""
        ))

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}"
    return latex


# ---------------------------------------------------------------------------
# PAPER TABLE CONFIG
# ---------------------------------------------------------------------------
# Each entry: (outcome_group, outcome_var, column_label_for_paper)
# These define the 6 columns used in the main results and robustness tables.
PAPER_MAIN_COLS = [
    ("location",               "in_us2",                               "In U.S."),
    ("location",               "in_home_country2",                     "In home ctry"),
    ("retention",              "still_at_firm2",                       "Same firm"),
    ("within_firm_conditional","same_firm_new_position_conditional2",  "New pos.\\ (cond.)"),
    ("mobility",               "new_diff_firm2",                       "New diff.\\ firm"),
    ("education",              "new_educ2",                            "New educ."),
]

# Panels for the robustness table, in display order.
# Each entry: (variant_name, panel_label)
PAPER_ROBUSTNESS_PANELS = [
    ("us_educ",        "U.S.-educated"),
    ("us_educ_opt",    "One-to-one (optimal bipartite matching)"),
    ("us_educ_prefilt","Pre-filtered (excl.\\ Southern Asia, East Asia, UK, CA, AU)"),
    ("strict_low",     "Strict (Q25 score threshold)"),
    ("strict_med",     "Strict (Q50 score threshold)"),
    ("strict_high",    "Strict (Q75 score threshold)"),
]

# Variants to include as panels in the combined heterogeneity table.
PAPER_HET_PANELS = [
    ("us_educ",        "U.S.-educated"),
    ("us_educ_opt",    "One-to-one"),
    ("strict_high",    "Strict (Q75)"),
]


def save_table(latex: str, path: Path, verbose: bool = True):
    """Write LaTeX string to file and optionally print it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex)
    if verbose:
        print(f"\n--- {path.name} ---")
        print(latex)
    print(f"  Saved: {path}")


###############################################################################
# BALANCE TABLE
###############################################################################

def build_paper_tables(all_variant_results: dict, all_het_results: dict, output_dir: Path):
    """
    Assemble paper-ready LaTeX tabular files from regression results.

    Writes two files to output_dir/tables/:
      paper_main_results.tex  — 6-column main results (us_educ, primary FE spec)
      paper_robustness.tex    — winner row across three sample panels

    Parameters
    ----------
    all_variant_results : dict  variant_name -> run_all_outcomes() output
    output_dir          : base output directory
    """
    spec = PRIMARY_FE_SPEC
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    def _col_result(variant_results, group, outcome):
        return variant_results.get(spec, {}).get(group, {}).get(outcome)

    # ------------------------------------------------------------------
    # Main results table (us_educ, 6 columns)
    # ------------------------------------------------------------------
    if "us_educ" not in all_variant_results:
        print("  [SKIP] paper_main_results.tex: us_educ variant not in results")
    else:
        res_us = all_variant_results["us_educ"]
        col_results = [_col_result(res_us, grp, out) for grp, out, _ in PAPER_MAIN_COLS]
        col_labels  = [lbl for _, _, lbl in PAPER_MAIN_COLS]
        latex = make_regression_table(
            col_results, col_labels,
            title="Main results — us_educ — FE: firm_year",
            verbose=False,
            include_interaction=INCLUDE_WINNER_ADE_INTERACTION,
        )
        save_table(latex, tables_dir / "paper_main_results.tex")

    # ------------------------------------------------------------------
    # Robustness table (winner row only, one panel per variant)
    # ------------------------------------------------------------------
    panels = [(name, lbl) for name, lbl in PAPER_ROBUSTNESS_PANELS
              if name in all_variant_results]
    if not panels:
        print("  [SKIP] paper_robustness.tex: no robustness variants in results")
        return

    ncols = len(PAPER_MAIN_COLS)
    col_header = " & ".join(f"({i})" for i in range(1, ncols + 1))
    col_names  = " & ".join(lbl for _, _, lbl in PAPER_MAIN_COLS)

    latex  = "% Robustness — winner effects across sample variants — FE: firm_year\n"
    latex += f"\\begin{{tabular}}{{l{'c' * ncols}}}\n"
    latex += "\\toprule\n"
    latex += f"& {col_header} \\\\\n"
    latex += f"& {col_names} \\\\\n"
    latex += "\\midrule\n"

    for idx, (var_name, panel_label) in enumerate(panels):
        res_v = all_variant_results[var_name]
        col_results = [_col_result(res_v, grp, out) for grp, out, _ in PAPER_MAIN_COLS]

        n_obs = next((int(r.nobs) for r in col_results if r is not None), None)
        n_str = f"{n_obs:,}" if n_obs is not None else "---"

        latex += f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\textit{{Panel: {panel_label} (N = {n_str})}}}}\\\\\n"

        coefs, ses = _coef_row(col_results, "winner")
        latex += "Winner   & " + " & ".join(coefs) + " \\\\\n"
        latex += "         & " + " & ".join(ses) + " \\\\\n"

        ctrl_means = []
        for res in col_results:
            if res is None:
                ctrl_means.append("")
            elif hasattr(res, "ctrl_mean") and res.ctrl_mean is not None:
                ctrl_means.append(f"{res.ctrl_mean:.3f}")
            elif res.model is not None:
                dv = res.model.dependent.dataframe.squeeze()
                exog = res.model.exog.dataframe
                if "winner" in exog.columns:
                    ctrl_means.append(f"{dv[exog['winner'] == 0].mean():.3f}")
                else:
                    ctrl_means.append("")
            else:
                ctrl_means.append("")
        sep = " \\\\[6pt]\n" if idx < len(panels) - 1 else " \\\\\n"
        latex += "Ctrl.\\ mean & " + " & ".join(ctrl_means) + sep

    latex += "\\bottomrule\n\\end{tabular}"
    save_table(latex, tables_dir / "paper_robustness.tex")

    # ------------------------------------------------------------------
    # Combined match quality table
    # ------------------------------------------------------------------
    try:
        from match_quality_check import build_paper_match_quality_table
        build_paper_match_quality_table(output_dir=output_dir)
    except Exception as e:
        print(f"  [WARN] build_paper_match_quality_table failed: {e}")

    # ------------------------------------------------------------------
    # Combined het table (still_at_firm2, multi-panel)
    # ------------------------------------------------------------------
    het_panels = [(name, lbl) for name, lbl in PAPER_HET_PANELS if name in all_het_results]
    if not het_panels:
        print("  [SKIP] paper_het_in_us2.tex: no het results available")
        return

    # Use col_labels from first available panel
    _, first_col_labels = all_het_results[het_panels[0][0]]
    ncols = len(first_col_labels)

    latex  = "% Combined het table — in_us2 — multi-panel\n"
    latex += f"\\begin{{tabular}}{{l{'c' * ncols}}}\n"
    latex += "\\toprule\n"
    latex += "& " + " & ".join(first_col_labels) + " \\\\\n"
    latex += "& (" + ") & (".join(str(i) for i in range(1, ncols + 1)) + ") \\\\\n"
    latex += "\\midrule\n"

    for idx, (var_name, panel_label) in enumerate(het_panels):
        res_list, col_labels = all_het_results[var_name]
        n_obs = next((int(r.nobs) for r in res_list if r is not None), None)
        n_str = f"{n_obs:,}" if n_obs is not None else "---"

        latex += f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\textit{{Panel: {panel_label} (N = {n_str})}}}}\\\\\n"

        coefs, ses = _coef_row(res_list, "winner")
        latex += "Winner   & " + " & ".join(coefs) + " \\\\\n"
        latex += "         & " + " & ".join(ses) + " \\\\\n"

        ctrl_means = []
        for res in res_list:
            if res is None:
                ctrl_means.append("")
            elif hasattr(res, "ctrl_mean") and res.ctrl_mean is not None:
                ctrl_means.append(f"{res.ctrl_mean:.3f}")
            else:
                ctrl_means.append("")
        sep = " \\\\[6pt]\n" if idx < len(het_panels) - 1 else " \\\\\n"
        latex += "Ctrl.\\ mean & " + " & ".join(ctrl_means) + sep

    latex += "\\bottomrule\n\\end{tabular}"
    save_table(latex, tables_dir / "paper_het_in_us2.tex")


def make_balance_table(df_app: pd.DataFrame, variant_name: str, output_dir: Path):
    """
    Compare pre-lottery covariates by winner status, separately within ADE=0 and ADE=1 subgroups.

    For all variables except `ade` itself, means and t-test p-values are computed within each
    ADE subgroup (ADE=0 and ADE=1) to assess within-group balance.
    For `ade`, the comparison is done on the full sample (conditioning on ADE is meaningless there).

    LaTeX output has 7 columns: Variable | ADE=0 Loser | ADE=0 Winner | ADE=0 p | ADE=1 Loser | ADE=1 Winner | ADE=1 p
    """
    balance_vars = [
        ("age",              "Age at lottery"),
        ("ade",              "Advanced degree eligible"),
        ("graddiff_agg",     "Years since graduation"),
        ("female_ind",       "Female"),
        ("n_apps",           "# applications (multiplicity)"),
        ("n_unique_country", "# unique countries (match)"),
        ("high_rep_emp_ind", "High-reputation employer"),
    ]

    df_ade0 = df_app[df_app["ade"] == 0]
    df_ade1 = df_app[df_app["ade"] == 1]

    def _compute_row(col, label, df):
        """Compute balance stats for one variable in a given subsample."""
        if col not in df.columns:
            return None
        g0 = df.loc[df["winner"] == 0, col].dropna()
        g1 = df.loc[df["winner"] == 1, col].dropna()
        if len(g0) < 2 or len(g1) < 2:
            return None
        _, pval = stats.ttest_ind(g0, g1, equal_var=False)
        return {
            "label": label,
            "mean_0": g0.mean(),
            "mean_1": g1.mean(),
            "pval": pval,
            "n0": len(g0),
            "n1": len(g1),
        }

    rows = []
    for col, label in balance_vars:
        if col not in df_app.columns:
            continue
        if col == "ade":
            # Full-sample comparison for ADE itself
            r_full = _compute_row(col, label, df_app)
            if r_full is None:
                continue
            rows.append({"label": label, "col": col, "full": r_full, "ade0": None, "ade1": None})
        else:
            r0 = _compute_row(col, label, df_ade0)
            r1 = _compute_row(col, label, df_ade1)
            if r0 is None and r1 is None:
                continue
            rows.append({"label": label, "col": col, "full": None, "ade0": r0, "ade1": r1})

    if not rows:
        print(f"  [balance] No variables available for {variant_name}")
        return

    # --- Console print ---
    def _stars(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

    print(f"\n=== Balance Table: {variant_name} ===")
    hdr = (f"{'Variable':<35} {'ADE=0 L':>8} {'ADE=0 W':>8} {'p':>7}  "
           f"{'ADE=1 L':>8} {'ADE=1 W':>8} {'p':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["col"] == "ade":
            f = r["full"]
            print(f"  {r['label']:<33} {f['mean_0']:8.3f} {f['mean_1']:8.3f} "
                  f"{f['pval']:7.3f}{_stars(f['pval'])}  (full sample)")
        else:
            r0, r1 = r["ade0"], r["ade1"]
            s0 = (f"{r0['mean_0']:8.3f} {r0['mean_1']:8.3f} {r0['pval']:7.3f}{_stars(r0['pval'])}"
                  if r0 else f"{'—':>8} {'—':>8} {'—':>7}")
            s1 = (f"{r1['mean_0']:8.3f} {r1['mean_1']:8.3f} {r1['pval']:7.3f}{_stars(r1['pval'])}"
                  if r1 else f"{'—':>8} {'—':>8} {'—':>7}")
            print(f"  {r['label']:<33} {s0}  {s1}")

    # --- LaTeX ---
    latex = f"% Balance table: {variant_name}\n"
    latex += "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    latex += ("& \\multicolumn{3}{c}{ADE = 0} & \\multicolumn{3}{c}{ADE = 1} \\\\\n"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
    latex += "Variable & Loser & Winner & p-val & Loser & Winner & p-val \\\\\n\\midrule\n"

    for r in rows:
        if r["col"] == "ade":
            f = r["full"]
            stars = _stars(f["pval"])
            # Span across all 6 data columns: show full-sample stats in first 3, dashes in last 3
            latex += (f"{r['label']} & {f['mean_0']:.3f} & {f['mean_1']:.3f} & "
                      f"{f['pval']:.3f}{stars} & \\multicolumn{{3}}{{c}}{{(full sample)}} \\\\\n")
        else:
            r0, r1 = r["ade0"], r["ade1"]
            if r0:
                s0 = f"{r0['mean_0']:.3f} & {r0['mean_1']:.3f} & {r0['pval']:.3f}{_stars(r0['pval'])}"
            else:
                s0 = "— & — & —"
            if r1:
                s1 = f"{r1['mean_0']:.3f} & {r1['mean_1']:.3f} & {r1['pval']:.3f}{_stars(r1['pval'])}"
            else:
                s1 = "— & — & —"
            latex += f"{r['label']} & {s0} & {s1} \\\\\n"

    # N footer: ADE=0 and ADE=1 separately
    n0_l = int((df_ade0["winner"] == 0).sum())
    n0_w = int((df_ade0["winner"] == 1).sum())
    n1_l = int((df_ade1["winner"] == 0).sum())
    n1_w = int((df_ade1["winner"] == 1).sum())
    latex += "\\midrule\n"
    latex += (f"N (ADE=0: losers / winners) & {n0_l:,} & {n0_w:,} & & & & \\\\\n")
    latex += (f"N (ADE=1: losers / winners) & & & & {n1_l:,} & {n1_w:,} & \\\\\n")
    latex += "\\bottomrule\n\\end{tabular}"

    save_table(latex, output_dir / "tables" / f"balance_{variant_name}.tex", verbose=False)


###############################################################################
# HETEROGENEOUS EFFECTS
###############################################################################

def run_het_effects(
    df_app: pd.DataFrame,
    variant_name: str,
    output_dir: Path,
):
    """
    Heterogeneous effects on retention outcomes using the primary FE spec.

    Two dimensions:
      1. Employer reputation: all / high_rep / no_rep
      2. Years since graduation: all / graddiff<3 / graddiff>=3 / interaction term
    """
    ret_outcome_specs = [
        spec for spec in OUTCOME_GROUPS.get("retention", [])
        if spec.get("source_outcome") in df_app.columns
    ]
    if not ret_outcome_specs:
        return

    spec = PRIMARY_FE_SPEC

    # --- Dimension 1: employer reputation subgroups ---
    print(f"\n  [het] Employer reputation subgroups ({variant_name})")
    rep_subgroups = {
        "all": df_app,
        "high\\_rep": df_app[df_app.get("high_rep_emp_ind", pd.Series(dtype=int)) == 1]
                      if "high_rep_emp_ind" in df_app.columns else None,
        "no\\_rep":  df_app[df_app.get("no_rep_emp_ind",  pd.Series(dtype=int)) == 1]
                      if "no_rep_emp_ind"  in df_app.columns else None,
    }

    for outcome_spec in ret_outcome_specs:
        outcome_name = outcome_spec["name"]
        res_list, col_labels = [], []
        for label, sub in rep_subgroups.items():
            if sub is None or len(sub) < 10:
                continue
            res = run_regression(sub, outcome_spec, spec,
                                 include_interaction=INCLUDE_WINNER_ADE_INTERACTION)
            res_list.append(res)
            col_labels.append(label)

        if any(r is not None for r in res_list):
            title = f"Het effects (employer rep) — {outcome_name} — {variant_name}"
            latex = make_regression_table(res_list, col_labels, title=title, verbose=True,
                                          include_interaction=INCLUDE_WINNER_ADE_INTERACTION)
            save_table(
                latex,
                output_dir / "tables" / f"het_{variant_name}_rep_{outcome_name}.tex",
                verbose=False,
            )

    # --- Dimension 2: graddiff subgroups + interaction ---
    if "graddiff_agg" not in df_app.columns:
        return

    print(f"\n  [het] graddiff subgroups ({variant_name})")
    grad_subgroups = {
        "all":        df_app,
        "graddiff=0": df_app[df_app["graddiff_agg"].round() == 0],
        "graddiff=1": df_app[df_app["graddiff_agg"].round() == 1],
        "graddiff=2": df_app[df_app["graddiff_agg"].round() == 2],
        "graddiff=3": df_app[df_app["graddiff_agg"].round() == 3],
    }

    for outcome_spec in ret_outcome_specs:
        outcome_name = outcome_spec["name"]
        source_outcome = outcome_spec.get("source_outcome", outcome_name)
        res_list, col_labels = [], []

        # Subgroup regressions
        for label, sub in grad_subgroups.items():
            if len(sub) < 10:
                continue
            res = run_regression(sub, outcome_spec, spec,
                                 include_interaction=INCLUDE_WINNER_ADE_INTERACTION)
            res_list.append(res)
            col_labels.append(label)

        if any(r is not None for r in res_list):
            title = f"Het effects (graddiff) — {outcome_name} — {variant_name}"
            latex = make_regression_table(
                res_list, col_labels,
                row_vars=["winner", "ade", "winner:ade"] if INCLUDE_WINNER_ADE_INTERACTION
                         else ["winner", "ade"],
                title=title, verbose=True,
            )
            save_table(
                latex,
                output_dir / "tables" / f"het_{variant_name}_graddiff_{outcome_name}.tex",
                verbose=False,
            )


def run_in_us_het_table(
    df_app: pd.DataFrame,
    variant_name: str,
    output_dir: Path,
):
    """
    Combined heterogeneity table for in_us2: graddiff=0/1/2/3 subgroups followed by
    norep/highrep employer-reputation subgroups, all in a single table.
    """
    if "in_us2" not in df_app.columns:
        return

    outcome_spec = {
        "name": "in_us2",
        "source_outcome": "in_us2",
        "subset_var": None,
        "subset_value": None,
    }
    spec = PRIMARY_FE_SPEC

    subgroups = {}
    if "graddiff_agg" in df_app.columns:
        for g in [0, 1, 2, 3]:
            subgroups[f"graddiff$={g}$"] = df_app[df_app["graddiff_agg"].round() == g]
    if "no_rep_emp_ind" in df_app.columns:
        subgroups["no\\_rep"] = df_app[df_app["no_rep_emp_ind"] == 1]
    if "high_rep_emp_ind" in df_app.columns:
        subgroups["high\\_rep"] = df_app[df_app["high_rep_emp_ind"] == 1]

    res_list, col_labels = [], []
    for label, sub in subgroups.items():
        if len(sub) < 10:
            continue
        res = run_regression(sub, outcome_spec, spec,
                             include_interaction=INCLUDE_WINNER_ADE_INTERACTION)
        res_list.append(res)
        col_labels.append(label)

    if not any(r is not None for r in res_list):
        return None, None

    title = f"Het effects (in_us2) — graddiff + rep — {variant_name}"
    latex = make_regression_table(
        res_list, col_labels,
        row_vars=["winner", "ade"],
        title=title, verbose=True,
        include_interaction=INCLUDE_WINNER_ADE_INTERACTION,
    )
    save_table(
        latex,
        output_dir / "tables" / f"het_{variant_name}_in_us2_combined.tex",
        verbose=False,
    )
    return res_list, col_labels


###############################################################################
# NON-UPDATING BIAS DIAGNOSTICS
###############################################################################

def run_profile_recency_subsample(df_app: pd.DataFrame, variant_name: str, output_dir: Path):
    """
    Robustness check for non-updating bias: split sample by whether the Revelio
    profile shows evidence of recent activity before vs. after the lottery, then
    compare treatment effects on main mobility outcomes across subgroups.

    Uses updatediff_activity (months from pre-lottery ref to most recent position/educ
    startdate) if available, falls back to updatediff (Revelio refresh date).

    - updatediff_activity <= 0: last profile activity was BEFORE the pre-lottery
      reference date (lottery_year-1, March) → profile was stale, any updating
      cannot be caused by winning the lottery. If treatment effects persist in
      this subsample, differential updating is NOT the main driver.
    - updatediff_activity >= 12: last activity at least 1 year after the reference
      date → profile was updated post-lottery. Effects here may be contaminated.
    """
    uvar = (
        "updatediff_activity" if "updatediff_activity" in df_app.columns
        else "updatediff" if "updatediff" in df_app.columns
        else None
    )
    if uvar is None:
        print("  [diag] updatediff not available, skipping profile recency diagnostics")
        return

    print(f"  [diag] Running profile recency subsample analysis using '{uvar}'")
    mobility_outcomes = [
        o for o in ["diff_firm1", "diff_firm2", "new_diff_firm1", "new_diff_firm2",
                    "same_firm_new_position1", "same_firm_new_position2"]
        if o in df_app.columns
    ]
    if not mobility_outcomes:
        return

    spec = PRIMARY_FE_SPEC
    subgroups = {
        "All":                    df_app,
        "Pre-lottery activity":   df_app[df_app[uvar] <= 0],
        "Post-lottery activity":  df_app[df_app[uvar] >= 12],
    }

    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    for outcome in mobility_outcomes:
        res_list, col_labels = [], []
        outcome_spec = {"name": outcome, "source_outcome": outcome,
                        "subset_var": None, "subset_value": None}
        for label, sub in subgroups.items():
            if len(sub) < 50:
                print(f"    [diag] Skipping '{label}' for {outcome} (n={len(sub)} < 50)")
                continue
            res = run_regression(sub, outcome_spec, spec,
                                 include_interaction=INCLUDE_WINNER_ADE_INTERACTION)
            res_list.append(res)
            col_labels.append(label)
        if any(r is not None for r in res_list):
            latex = make_regression_table(
                res_list, col_labels,
                title=f"Profile recency robustness — {outcome} — {variant_name}",
                verbose=True,
            )
            out_path = output_dir / "tables" / f"diag_recency_{variant_name}_{outcome}.tex"
            save_table(latex, out_path, verbose=False)
            print(f"    [diag] Saved {out_path.name}")


###############################################################################
# EVENT-TIME GRAPHS
###############################################################################

def run_event_time_graphs(
    df_app: pd.DataFrame,
    variant_name: str,
    output_dir: Path,
    et_cfg: dict,
    include_interaction: bool = True,
):
    """
    For each configured outcome base, run regressions at t=1, 2, 3 and plot
    the winner coefficient ± 95% CI error bars vs. years post-lottery.

    X-axis: years post-lottery (t = 1, 2, 3)
    Y-axis: winner coefficient from  outcome_t ~ winner + ade [+ winner:ade] + FE
    Error bars: ±1.96 × SE (firm-clustered)

    Saves PNGs to: {output_dir}/graphs/event_time_{variant_name}_{outcome_base}.png
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    spec       = PRIMARY_FE_SPEC
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    outcome_bases = et_cfg.get("outcome_bases", ["in_us", "still_at_firm"])
    print(f"  [event_time] variant={variant_name}  spec={spec}  "
          f"bases={outcome_bases}")

    for base in outcome_bases:
        periods, coefs, ses = [], [], []

        for t in [-2, -1, 0, 1, 2, 3]:
            col = f"{base}{t}"
            if col not in df_app.columns:
                print(f"    [SKIP] {col} not in data")
                continue
            outcome_spec = {"name": col, "source_outcome": col,
                            "subset_var": None, "subset_value": None}
            res = run_regression(
                df_app, outcome_spec, spec,
                include_interaction=include_interaction,
            )
            if res is None:
                print(f"    [SKIP] {col}: regression returned None")
                continue
            if "winner" not in res.params.index:
                print(f"    [SKIP] {col}: 'winner' not in params")
                continue

            coef = float(res.params["winner"])
            se   = float(res.std_errors["winner"])
            periods.append(t)
            coefs.append(coef)
            ses.append(se)
            print(f"    {col}: winner={coef:+.4f}  SE={se:.4f}  N={res.nobs:,}")

        if len(periods) < 2:
            print(f"    [SKIP] {base}: fewer than 2 periods estimated, skipping plot")
            continue

        # --- Build plot ---
        coefs_arr = np.array(coefs)
        ses_arr   = np.array(ses)
        ci95      = 1.96 * ses_arr

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(5, 4))

        ax.errorbar(
            periods, coefs_arr,
            yerr=ci95,
            fmt="o-",
            capsize=5,
            linewidth=1.5,
            markersize=6,
            color=sns.color_palette("deep")[0],
            label="Winner coef. ± 1.96 SE",
        )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(periods)
        ax.set_xlabel("Years post-lottery (t)")
        ax.set_ylabel("Winner coefficient")
        ax.set_title(f"Event-time: {base}\n{variant_name} — {spec}")
        ax.legend(fontsize=9)
        plt.tight_layout()

        out_path = graphs_dir / f"event_time_{variant_name}_{base}.png"
        plt.savefig(out_path, dpi=150)
        plt.show()
        plt.close()
        print(f"    Saved: {out_path}")


###############################################################################
# MAIN LOOP
###############################################################################

def process_variant(variant: dict, output_dir: Path):
    """Full pipeline for one merged parquet variant."""
    name = variant["name"]
    parquet_key = variant["parquet_key"]

    # Resolve parquet path via icfg; optionally redirect to _opt files
    primary_path = getattr(icfg, parquet_key, None)
    legacy_path = getattr(icfg, parquet_key + "_LEGACY", None)

    if primary_path is None:
        print(f"  [SKIP] Unknown parquet key: {parquet_key}")
        return
    if USE_OPTIMAL_DEDUP:
        primary_path = primary_path.replace(".parquet", "_opt.parquet")
        legacy_path = None  # no legacy variant for optimal dedup outputs
    parquet_path = icfg.choose_path(primary_path, legacy_path)
    if not os.path.exists(parquet_path):
        print(f"  [SKIP] Parquet not found: {parquet_path}")
        return

    print(f"\n{'='*60}")
    print(f"Variant: {name}")
    print(f"{'='*60}")
    t_start = time.time()

    # 1. Load + clean (row-level)
    df_raw = load_and_clean(parquet_path)

    # 2. Collapse to application level
    df_app = collapse_to_app_level(df_raw)

    # 3. Testing mode: subsample
    df_app = maybe_sample(df_app)

    # 4. Balance table
    if BALANCE_TABLE:
        make_balance_table(df_app, name, output_dir)

    # 5. Run all regressions
    print(f"\n  Running regressions...")
    results = run_all_outcomes(
        df_app, OUTCOME_GROUPS, FE_SPECS,
        include_interaction=INCLUDE_WINNER_ADE_INTERACTION,
    )

    # 6. Save regression tables (one per FE spec per outcome group)
    for spec in FE_SPECS:
        is_primary = (spec == PRIMARY_FE_SPEC)
        for group, group_outcomes in OUTCOME_GROUPS.items():
            group_results = results[spec].get(group, {})
            res_list = [group_results.get(o["name"]) for o in group_outcomes]
            col_labels = [o["name"] for o in group_outcomes]

            # Skip if all None
            if all(r is None for r in res_list):
                continue

            title = f"{name} — {group} outcomes — FE: {spec}"
            latex = make_regression_table(
                res_list, col_labels,
                title=title,
                verbose=is_primary,   # print compare() only for primary spec
                include_interaction=INCLUDE_WINNER_ADE_INTERACTION,
            )
            out_path = output_dir / "tables" / f"reg_{name}_{group}_{spec}.tex"
            save_table(latex, out_path, verbose=is_primary)

    # 7. Heterogeneous effects (primary spec only)
    if HET_EFFECTS:
        print(f"\n  Running heterogeneous effects...")
        run_het_effects(df_app, name, output_dir)

    # 7b. In-US heterogeneity table (graddiff=0/1/2/3 + norep/highrep, in_us2 only)
    het_results = None
    if IN_US_HET_EFFECTS:
        print(f"\n  Running in_us2 heterogeneity table...")
        het_results = run_in_us_het_table(df_app, name, output_dir)

    # 8. Non-updating bias diagnostics
    if RUN_UPDATING_DIAGNOSTICS:
        print(f"\n  Running non-updating bias diagnostics...")
        run_profile_recency_subsample(df_app, name, output_dir)

    # 9. Ex-post match quality check
    if RUN_MATCH_QUALITY_CHECK:
        print(f"\n  Running ex-post match quality check...")
        from match_quality_check import run_match_quality_check
        run_match_quality_check(name, CFG.get("run_tag", icfg.RUN_TAG), output_dir, parquet_path)

    # 10. Event-time coefficient graphs
    if EVENT_TIME_CFG.get("enabled", False):
        et_variants = EVENT_TIME_CFG.get("variants")  # None = all variants
        if et_variants is None or name in et_variants:
            print(f"\n  Running event-time graphs...")
            run_event_time_graphs(
                df_app, name, output_dir,
                et_cfg=EVENT_TIME_CFG,
                include_interaction=INCLUDE_WINNER_ADE_INTERACTION,
            )

    print(f"\n  Done with {name} ({time.time()-t_start:.1f}s total)")
    return results, het_results


def main():
    t0 = time.time()
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    all_variant_results = {}
    all_het_results = {}
    for variant in VARIANTS:
        out = process_variant(variant, output_dir)
        if out is not None:
            results, het = out
            all_variant_results[variant["name"]] = results
            if het is not None:
                res_list, col_labels = het
                all_het_results[variant["name"]] = (res_list, col_labels)

    print(f"\n{'='*60}")
    print("Building paper tables...")
    build_paper_tables(all_variant_results, all_het_results, output_dir)

    print(f"\n{'='*60}")
    print(f"All variants complete. Total time: {time.time()-t0:.1f}s")
    print(f"Tables saved to: {output_dir / 'tables'}")


main()
