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
from pathlib import Path

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


CFG = _load_reg_config()
TESTING = CFG.get("testing", {})
TESTING_ENABLED = bool(TESTING.get("enabled", False))
TESTING_N_APPS = int(TESTING.get("n_apps", 500))
TESTING_SEED = int(TESTING.get("seed", 42))
OUTPUT_DIR = Path(CFG["output_dir"])
PRIMARY_FE_SPEC = CFG.get("primary_fe_spec", "firm_year")
BALANCE_TABLE = bool(CFG.get("balance_table", True))
HET_EFFECTS = bool(CFG.get("het_effects", True))
OUTCOME_GROUPS: dict = CFG.get("outcomes", {})
FE_SPECS: list = CFG.get("fe_specs", ["firm_year"])
VARIANTS: list = CFG.get("variants", [])

print(f"=== reg_new.py ===")
print(f"run_tag:  {CFG.get('run_tag', 'feb2026')}")
print(f"testing:  {'ENABLED (n=' + str(TESTING_N_APPS) + ')' if TESTING_ENABLED else 'disabled'}")
print(f"variants: {[v['name'] for v in VARIANTS]}")
print(f"fe_specs: {FE_SPECS}")
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

    # Outcome variables (still_at_firm, promoted, change_company, in_us, in_home_country,
    # new_educ, agg_compensation) come directly from the merged parquet (built in indiv_merge.py).

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
        "still_at_firm1", "still_at_firm2", "still_at_firm3",   # still at same foia_firm_uid
        "change_company1", "change_company2", "change_company3", # at different foia_firm_uid (post-lottery)
        "promoted1", "promoted2", "promoted3",                   # same firm, new position (post-lottery)
        "in_us1", "in_us2", "in_us3",
        "in_home_country1", "in_home_country2", "in_home_country3",
        "new_educ1", "new_educ2", "new_educ3",
        "agg_compensation1", "agg_compensation2", "agg_compensation3",
        "graddiff",
    ]
    outcomes = [v for v in candidate_outcomes if v in df.columns]

    # Applicant-level constants (same value across all rows for a given applicant)
    id_cols = [
        "foia_indiv_id", "female_ind", "yob", "age",
        "lottery_year", "foia_country", "winner", "ade",
        "n_apps", "n_unique_country", "high_rep_emp_ind", "no_rep_emp_ind",
        "graddiff_agg",
    ]
    id_cols = [c for c in id_cols if c in df.columns]

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

    # Step 2: Collapse outcomes to applicant level via weighted average
    agg_dict = {
        var: (var, lambda x, var=var: (
            np.average(x.dropna(), weights=df.loc[x.dropna().index, "weight_norm"])
            if x.notna().any() else np.nan
        ))
        for var in outcomes
    }
    agg_dict["weight_norm"] = ("weight_norm", "sum")  # total match weight per applicant

    app_df = df.groupby(id_cols).agg(**agg_dict).reset_index()

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


def _fe_formula(outcome: str, fe_spec: str, df: pd.DataFrame) -> str:
    """Build the PanelOLS formula string for a given FE spec."""
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


def run_regression(
    df: pd.DataFrame,
    outcome: str,
    fe_spec: str,
):
    """
    Run a single weighted PanelOLS regression.
    Returns PanelOLSResults or None if the outcome column is missing / all NaN.
    """
    if outcome not in df.columns or df[outcome].isna().all():
        return None

    # Drop rows where outcome or key regressors are NaN
    keep_cols = [outcome, "winner", "ade", "weight_norm", "lottery_year",
                 "firm_key", "firm_year_fe", "foia_indiv_id"]
    sub = df[keep_cols].dropna(subset=[outcome, "winner", "ade", "weight_norm"])

    if len(sub) < 10:
        print(f"    [SKIP] {outcome} ({fe_spec}): too few obs ({len(sub)})")
        return None

    # Add year dummies before building panel index (needed for firm_plus_year)
    if fe_spec == "firm_plus_year":
        years = sorted(sub["lottery_year"].unique())[1:]
        for yr in years:
            sub[f"yr_{yr}"] = (sub["lottery_year"] == yr).astype(int)

    panel_df = _build_panel_df(sub, fe_spec)
    formula = _fe_formula(outcome, fe_spec, sub)

    try:
        model = PanelOLS.from_formula(
            formula,
            data=panel_df,
            weights=panel_df["weight_norm"],
        )
        result = model.fit(cov_type="clustered", cluster_entity=True)
        return result
    except Exception as e:
        print(f"    [ERR] {outcome} ({fe_spec}): {e}")
        return None


def run_all_outcomes(
    df_app: pd.DataFrame,
    outcome_groups: dict,
    fe_specs: list,
) -> dict:
    """
    Run regressions for all outcomes × all FE specs.
    Returns nested dict: {fe_spec: {group_name: {outcome: result_or_None}}}.
    """
    results = {spec: {} for spec in fe_specs}
    all_outcomes = [o for grp in outcome_groups.values() for o in grp]

    for spec in fe_specs:
        print(f"  FE spec: {spec}")
        for group, outcomes in outcome_groups.items():
            results[spec][group] = {}
            for outcome in outcomes:
                res = run_regression(df_app, outcome, spec)
                results[spec][group][outcome] = res
                if res is not None:
                    coef = res.params.get("winner", np.nan)
                    se = res.std_errors.get("winner", np.nan)
                    n = int(res.nobs)
                    print(f"    {outcome:30s}  winner={coef:+.4f} ({se:.4f})  N={n:,}")

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
        coefs.append(f"{b:.4f}{stars}")
        ses.append(f"({se:.4f})")
    return coefs, ses


def make_regression_table(
    results: list,
    col_labels: list,
    row_vars: list = None,
    title: str = "",
    verbose: bool = True,
) -> str:
    """
    Format a list of PanelOLSResults into a LaTeX table string.

    Parameters
    ----------
    results   : list of PanelOLSResults (None entries produce blank cells)
    col_labels: column header strings
    row_vars  : list of coefficient names to display (default: ['winner', 'ade'])
    title     : optional title comment at top of LaTeX
    verbose   : if True, also print linearmodels compare() table to console
    """
    if row_vars is None:
        row_vars = ["winner", "ade"]

    valid = {lab: res for lab, res in zip(col_labels, results) if res is not None}

    if verbose and valid:
        print(compare(valid, stars=True, precision="std_errors"))

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
    obs_strs = [str(int(r.nobs)) if r is not None else "" for r in results]
    latex += "\\textbf{N} & " + " & ".join(obs_strs) + " \\\\\n"

    dv_means = []
    for res in results:
        if res is None:
            dv_means.append("")
        else:
            dv_series = res.model.dependent.dataframe.squeeze()
            dv_means.append(f"{dv_series.mean():.3f}")
    latex += "\\textbf{DV mean} & " + " & ".join(dv_means) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}"
    return latex


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

def make_balance_table(df_app: pd.DataFrame, variant_name: str, output_dir: Path):
    """
    Compare pre-lottery covariates by winner status.
    Prints and saves a LaTeX table with mean(winner=0), mean(winner=1), p-value.
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

    rows = []
    for col, label in balance_vars:
        if col not in df_app.columns:
            continue
        g0 = df_app.loc[df_app["winner"] == 0, col].dropna()
        g1 = df_app.loc[df_app["winner"] == 1, col].dropna()
        if len(g0) < 2 or len(g1) < 2:
            continue
        _, pval = stats.ttest_ind(g0, g1, equal_var=False)
        rows.append({
            "label": label,
            "mean_0": g0.mean(),
            "mean_1": g1.mean(),
            "diff": g1.mean() - g0.mean(),
            "pval": pval,
            "n0": len(g0),
            "n1": len(g1),
        })

    if not rows:
        print(f"  [balance] No variables available for {variant_name}")
        return

    print(f"\n=== Balance Table: {variant_name} ===")
    hdr = f"{'Variable':<35} {'Loser':>8} {'Winner':>8} {'Diff':>8} {'p-val':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        stars = "***" if r["pval"] < 0.01 else "**" if r["pval"] < 0.05 else "*" if r["pval"] < 0.1 else ""
        print(f"  {r['label']:<33} {r['mean_0']:8.3f} {r['mean_1']:8.3f} "
              f"{r['diff']:+8.3f} {r['pval']:7.3f}{stars}")

    # LaTeX
    latex = f"% Balance table: {variant_name}\n"
    latex += "\\begin{tabular}{lrrrr}\n\\toprule\n"
    latex += "Variable & Loser & Winner & Diff & p-val \\\\\n\\midrule\n"
    for r in rows:
        stars = "***" if r["pval"] < 0.01 else "**" if r["pval"] < 0.05 else "*" if r["pval"] < 0.1 else ""
        latex += (f"{r['label']} & {r['mean_0']:.3f} & {r['mean_1']:.3f} & "
                  f"{r['diff']:+.3f} & {r['pval']:.3f}{stars} \\\\\n")
    n0 = rows[0]["n0"] if rows else 0
    n1 = rows[0]["n1"] if rows else 0
    latex += "\\midrule\n"
    latex += f"N (losers / winners) & {n0:,} & {n1:,} & & \\\\\n"
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
    ret_outcomes = [o for o in OUTCOME_GROUPS.get("retention", []) if o in df_app.columns]
    if not ret_outcomes:
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

    for outcome in ret_outcomes:
        res_list, col_labels = [], []
        for label, sub in rep_subgroups.items():
            if sub is None or len(sub) < 10:
                continue
            res = run_regression(sub, outcome, spec)
            res_list.append(res)
            col_labels.append(label)

        if any(r is not None for r in res_list):
            title = f"Het effects (employer rep) — {outcome} — {variant_name}"
            latex = make_regression_table(res_list, col_labels, title=title, verbose=True)
            save_table(
                latex,
                output_dir / "tables" / f"het_{variant_name}_rep_{outcome}.tex",
                verbose=False,
            )

    # --- Dimension 2: graddiff subgroups + interaction ---
    if "graddiff_agg" not in df_app.columns:
        return

    print(f"\n  [het] graddiff subgroups ({variant_name})")
    grad_subgroups = {
        "all":        df_app,
        "graddiff$<$3": df_app[(df_app["graddiff_agg"].round() < 3) &
                                (df_app["graddiff_agg"] >= -1)],
        "graddiff$>$3": df_app[(df_app["graddiff_agg"].round() > 3) &
                                (df_app["graddiff_agg"] <= 6)],
    }

    for outcome in ret_outcomes:
        res_list, col_labels = [], []

        # Subgroup regressions
        for label, sub in grad_subgroups.items():
            if len(sub) < 10:
                continue
            res = run_regression(sub, outcome, spec)
            res_list.append(res)
            col_labels.append(label)

        # Interaction specification: winner * graddiff3 (graddiff_agg rounded == 3)
        df_int = df_app.copy()
        df_int["graddiff3"] = (df_int["graddiff_agg"].round() == 3).astype(int)
        keep_cols = [outcome, "winner", "ade", "graddiff3", "weight_norm",
                     "lottery_year", "firm_key", "firm_year_fe", "foia_indiv_id"]
        sub_int = df_int[[c for c in keep_cols if c in df_int.columns]].dropna(
            subset=[outcome, "winner", "ade", "weight_norm"]
        )

        if len(sub_int) >= 10:
            if spec == "firm_plus_year":
                years = sorted(sub_int["lottery_year"].unique())[1:]
                for yr in years:
                    sub_int[f"yr_{yr}"] = (sub_int["lottery_year"] == yr).astype(int)

            sub_int["_indiv_int"] = pd.Categorical(sub_int["foia_indiv_id"]).codes

            if spec in ("none", "firm_year"):
                panel_sub = sub_int.set_index(["firm_year_fe", "_indiv_int"])
                fe_term = "EntityEffects" if spec == "firm_year" else "1"
            else:
                panel_sub = sub_int.set_index(["firm_key", "_indiv_int"])
                yr_terms = " + ".join(f"yr_{yr}" for yr in years)
                fe_term = f"EntityEffects + {yr_terms}"

            formula = f"{outcome} ~ winner * graddiff3 + ade + {fe_term}"
            try:
                int_res = PanelOLS.from_formula(
                    formula, data=panel_sub, weights=panel_sub["weight_norm"]
                ).fit(cov_type="clustered", cluster_entity=True)
                res_list.append(int_res)
                col_labels.append("interact")
            except Exception as e:
                print(f"    [ERR] interact {outcome}: {e}")

        if any(r is not None for r in res_list):
            title = f"Het effects (graddiff) — {outcome} — {variant_name}"
            latex = make_regression_table(
                res_list, col_labels,
                row_vars=["winner", "ade", "graddiff3", "winner:graddiff3"],
                title=title, verbose=True,
            )
            save_table(
                latex,
                output_dir / "tables" / f"het_{variant_name}_graddiff_{outcome}.tex",
                verbose=False,
            )


###############################################################################
# MAIN LOOP
###############################################################################

def process_variant(variant: dict, output_dir: Path):
    """Full pipeline for one merged parquet variant."""
    name = variant["name"]
    parquet_key = variant["parquet_key"]

    # Resolve parquet path via icfg
    primary_path = getattr(icfg, parquet_key, None)
    legacy_path = getattr(icfg, parquet_key + "_LEGACY", None)

    if primary_path is None:
        print(f"  [SKIP] Unknown parquet key: {parquet_key}")
        return
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
    results = run_all_outcomes(df_app, OUTCOME_GROUPS, FE_SPECS)

    # 6. Save regression tables (one per FE spec per outcome group)
    for spec in FE_SPECS:
        is_primary = (spec == PRIMARY_FE_SPEC)
        for group, group_outcomes in OUTCOME_GROUPS.items():
            group_results = results[spec].get(group, {})
            res_list = [group_results.get(o) for o in group_outcomes]
            col_labels = group_outcomes

            # Skip if all None
            if all(r is None for r in res_list):
                continue

            title = f"{name} — {group} outcomes — FE: {spec}"
            latex = make_regression_table(
                res_list, col_labels,
                title=title,
                verbose=is_primary,   # print compare() only for primary spec
            )
            out_path = output_dir / "tables" / f"reg_{name}_{group}_{spec}.tex"
            save_table(latex, out_path, verbose=False)

    # 7. Heterogeneous effects (primary spec only)
    if HET_EFFECTS:
        print(f"\n  Running heterogeneous effects...")
        run_het_effects(df_app, name, output_dir)

    print(f"\n  Done with {name} ({time.time()-t_start:.1f}s total)")


def main():
    t0 = time.time()
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    for variant in VARIANTS:
        process_variant(variant, output_dir)

    print(f"\n{'='*60}")
    print(f"All variants complete. Total time: {time.time()-t0:.1f}s")
    print(f"Tables saved to: {output_dir / 'tables'}")


main()
