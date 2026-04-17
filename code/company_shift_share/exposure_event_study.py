"""Exposure-based firm-level event study around the 2016 OPT policy change.

Instead of grouping firms by the *timing* of their OPT shock (as in opt_event_study_2016.py),
this script computes a static, pre-period firm-level "OPT exposure" measure and stratifies
firms into ntile groups. This supports a cleaner diff-in-diff interpretation: do high-exposure
firms respond differently to the 2016 policy event than low-exposure firms?

Two exposure measures are available (configured via exposure_event_study.exposure_version):

  1. opt_hire_rate — OPT hire rate = OPT hires / total new hires, each summed over
       the exposure window (default 2010–2015). Purely from analysis_panel.parquet.

  2. school_opt_share — share of new hires from "OPT-intensive" schools, where schools
       are classified above/below the median school OPT usage rate (mean g_kt over the
       exposure window). g_kt comes from instrument_components.parquet; n_transitions_full
       per (firm, school) are used as hiring weights.

Raw plots:
  Mean outcome by ntile group over calendar years, demeaned to ref_year.

Regression:
  y_jt = firm_FE + Σ_{τ≠ref} δ_τ 1[t=τ]
       + Σ_{q>1, τ≠ref} β_{τq} 1[t=τ] 1[ntile_j=q] + ε_jt

  Coefficients β_{τq} (year × ntile interactions) are plotted with 95% CIs.

Interactive use:
    import company_shift_share.exposure_event_study as e
    e.main([])
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Flush stdout/stderr immediately for clean progress logging in interactive sessions.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    PanelOLS = None  # type: ignore[assignment,misc]

try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import (  # type: ignore[no-redef]
        DEFAULT_CONFIG_PATH,
        get_cfg_section,
        load_config,
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _in_ipython() -> bool:
    """Return True when running inside an IPython / Jupyter kernel."""
    try:
        get_ipython()  # type: ignore[name-defined]
        return True
    except NameError:
        return False


def _escape(path: Path) -> str:
    """Escape single quotes in a path string for use in DuckDB SQL literals."""
    return str(path).replace("'", "''")


def _resolve_path(paths_cfg: dict, key: str) -> Path:
    """Resolve a path from the config, substituting {root} with the repo root."""
    value = paths_cfg.get(key)
    if not value or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"Config paths.{key} must be set.")
    root = str(Path(__file__).resolve().parents[2])
    return Path(str(value).replace("{root}", root))


def _savefig(name: str, slides_out: Optional[Path], oc_tag: str, version_tag: str) -> None:
    """Save current matplotlib figure with outcome + version tags appended to filename."""
    if slides_out is None:
        return
    stem, _, ext = name.rpartition(".")
    tagged = f"{stem}_{oc_tag}_{version_tag}.{ext}"
    plt.savefig(slides_out / tagged, dpi=150, bbox_inches="tight")
    print(f"[info] Saved {tagged}")


def _ensure_derived_outcome(df: pd.DataFrame, col: str, x_source_col: str) -> None:
    """Derive binary or log-transformed outcome columns in-place if not already present."""
    if col in df.columns:
        return
    if col.startswith("log1p_"):
        base = col[len("log1p_"):]
        if base in df.columns:
            df[col] = np.where(df[base].notna() & (df[base] >= 0), np.log1p(df[base]), np.nan)
        return
    if x_source_col not in df.columns:
        return
    if col == "x_bin_any_nonzero":
        df[col] = (df[x_source_col].fillna(0) != 0).astype("int8")


# ---------------------------------------------------------------------------
# Exposure measure computation
# ---------------------------------------------------------------------------

def _compute_exposure_opt_hire_rate(
    panel: pd.DataFrame, year_min: int, year_max: int
) -> pd.Series:
    """
    Exposure version 1 (opt_hire_rate): OPT hires / total new hires, aggregated over
    [year_min, year_max] per firm.

    Numerator: sum of masters_opt_hires_correction_aware (NaN → 0).
    Denominator: sum of y_new_hires_lag0.
    Firms where the denominator is zero or entirely missing are dropped.

    Returns: pd.Series indexed by c (int firm id), values in [0, ∞).
    """
    for col in ("masters_opt_hires_correction_aware", "y_new_hires_lag0"):
        if col not in panel.columns:
            raise ValueError(
                f"Analysis panel missing required column for opt_hire_rate exposure: '{col}'.\n"
                f"Available columns: {list(panel.columns)}"
            )

    win = panel[panel["t"].between(year_min, year_max)].copy()
    print(f"[exposure opt_hire_rate] Firm-year obs in [{year_min}, {year_max}]: {len(win)}, "
          f"firms: {win['c'].nunique()}")

    # Sum over exposure window; pandas sum(skipna=True) treats NaN as 0 for numerator —
    # correct because a missing OPT hire count means zero OPT hires.
    agg = (
        win.groupby("c", as_index=False)
        .agg(
            opt_hires_sum=("masters_opt_hires_correction_aware", "sum"),
            new_hires_sum=("y_new_hires_lag0", "sum"),
        )
    )
    n_total = len(agg)
    # Drop firms where total new hires (denominator) is zero or null.
    agg = agg[agg["new_hires_sum"].notna() & (agg["new_hires_sum"] > 0)]
    n_dropped = n_total - len(agg)
    if n_dropped:
        print(f"[exposure opt_hire_rate] Dropped {n_dropped} firms with zero/null new-hire denominator.")
    print(f"[exposure opt_hire_rate] Firms with valid exposure: {len(agg)}")

    agg["exposure"] = agg["opt_hires_sum"] / agg["new_hires_sum"]
    print(f"[exposure opt_hire_rate] Distribution:\n{agg['exposure'].describe().to_string()}")

    return agg.set_index("c")["exposure"]


def _compute_exposure_school_opt_share(
    components: pd.DataFrame, year_min: int, year_max: int
) -> pd.Series:
    """
    Exposure version 2 (school_opt_share): share of new hires from OPT-intensive schools.

    Algorithm:
      1. Average g_kt over [year_min, year_max] per school k → school_opt_rate_k.
      2. Classify schools above/below the cross-school median → OPT-intensive flag.
      3. For each firm: sum(n_transitions_full for OPT-intensive schools) / total_new_hires_full.

    g_kt and n_transitions_full come from instrument_components.parquet (already built by
    build_company_shift_share.py). n_transitions_full is the all-years aggregate — the best
    approximation available without going back to raw per-year transitions.

    Returns: pd.Series indexed by c (int firm id), values in [0, 1].
    """
    for col in ("k", "t", "g_kt", "n_transitions_full", "total_new_hires_full"):
        if col not in components.columns:
            raise ValueError(
                f"instrument_components missing required column for school_opt_share: '{col}'.\n"
                f"Available: {list(components.columns)}"
            )

    # Step 1: compute school-level mean OPT rate over exposure window.
    win = components[components["t"].between(year_min, year_max)].copy()
    print(f"[exposure school_opt_share] Component rows in [{year_min}, {year_max}]: {len(win)}")

    school_rate = (
        win.groupby("k", as_index=False)["g_kt"]
        .mean()
        .rename(columns={"g_kt": "school_opt_rate"})
    )
    print(f"[exposure school_opt_share] Schools with g_kt in window: {len(school_rate)}")

    # Step 2: classify as OPT-intensive (above median cross-school rate).
    median_rate = school_rate["school_opt_rate"].median()
    print(f"[exposure school_opt_share] Median school OPT rate: {median_rate:.4f}")
    school_rate["opt_intensive"] = school_rate["school_opt_rate"] > median_rate
    n_intensive = int(school_rate["opt_intensive"].sum())
    print(f"[exposure school_opt_share] OPT-intensive schools: {n_intensive} / {len(school_rate)}")

    # Step 3: collapse components to one row per (c, k) — n_transitions_full is constant over t.
    comp_ck = (
        components.groupby(["c", "k"], as_index=False)
        .agg(
            n_transitions_full=("n_transitions_full", "first"),
            total_new_hires_full=("total_new_hires_full", "first"),
        )
    )
    comp_ck = comp_ck.merge(school_rate[["k", "opt_intensive"]], on="k", how="left")
    comp_ck["opt_intensive"] = comp_ck["opt_intensive"].fillna(False)

    # Step 4: sum transitions from OPT-intensive schools per firm.
    intensive = (
        comp_ck[comp_ck["opt_intensive"]]
        .groupby("c", as_index=False)["n_transitions_full"]
        .sum()
        .rename(columns={"n_transitions_full": "intensive_transitions"})
    )
    # Total new hires is constant per firm (take the first non-null value across schools).
    firm_totals = comp_ck.groupby("c", as_index=False)["total_new_hires_full"].first()

    firm_exp = firm_totals.merge(intensive, on="c", how="left")
    firm_exp["intensive_transitions"] = firm_exp["intensive_transitions"].fillna(0)

    n_total = len(firm_exp)
    firm_exp = firm_exp[
        firm_exp["total_new_hires_full"].notna() & (firm_exp["total_new_hires_full"] > 0)
    ]
    n_dropped = n_total - len(firm_exp)
    if n_dropped:
        print(f"[exposure school_opt_share] Dropped {n_dropped} firms with zero/null denominator.")
    print(f"[exposure school_opt_share] Firms with valid exposure: {len(firm_exp)}")

    firm_exp["exposure"] = firm_exp["intensive_transitions"] / firm_exp["total_new_hires_full"]
    print(f"[exposure school_opt_share] Distribution:\n{firm_exp['exposure'].describe().to_string()}")

    return firm_exp.set_index("c")["exposure"]


# ---------------------------------------------------------------------------
# Ntile assignment
# ---------------------------------------------------------------------------

def _assign_ntiles(exposure: pd.Series, ntiles: int, zero_separate: bool = False) -> pd.DataFrame:
    """
    Assign firms to exposure ntile groups.

    If zero_separate=True (used for opt_hire_rate, which has a mass of exactly-zero
    firms), firms with zero exposure are placed in their own group (Q1 = "no OPT hires")
    and positive-exposure firms are split into ntiles-1 equal groups (Q2…Qntiles) by qcut.

    If zero_separate=False (default, used for school_opt_share), standard qcut is applied
    to all firms.

    Returns DataFrame with columns: c (int), ntile (1-indexed int), ntile_label (str).
    Firms with null exposure are excluded.
    """
    valid = exposure.dropna()
    n_dropped = len(exposure) - len(valid)
    if n_dropped:
        print(f"[ntiles] Dropped {n_dropped} firms with null exposure.")

    parts: list[pd.DataFrame] = []

    if zero_separate:
        zero_mask = valid <= 0
        n_zero = int(zero_mask.sum())
        pos = valid[~zero_mask]
        n_pos = len(pos)
        print(f"[ntiles] Zero-exposure firms: {n_zero} | Positive-exposure firms: {n_pos}")

        # Q1: all firms with zero exposure.
        if n_zero > 0:
            parts.append(pd.DataFrame({"c": valid.index[zero_mask], "ntile": 1}))

        # Q2 … Qntiles: qcut of positive-exposure firms.
        n_pos_groups = ntiles - 1 if n_zero > 0 else ntiles
        offset = 2 if n_zero > 0 else 1
        if n_pos > 0 and n_pos_groups > 0:
            codes, bins = pd.qcut(pos, n_pos_groups, labels=False, retbins=True, duplicates="drop")
            actual_pos_groups = int(codes.max()) + 1
            if actual_pos_groups < n_pos_groups:
                print(f"[ntiles] Note: requested {n_pos_groups} positive-exposure bins but got "
                      f"{actual_pos_groups} due to ties.")
            parts.append(pd.DataFrame({"c": pos.index, "ntile": (codes + offset).astype(int)}))
            for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:]), start=offset):
                print(f"  Q{i}: ({lo:.4g}, {hi:.4g}]")
        elif n_pos == 0:
            print("[ntiles] No positive-exposure firms — all firms placed in Q1 (zero).")

        has_zero_group = n_zero > 0

    else:
        # Rank-based ntile assignment: rank firms by exposure (ties broken by first occurrence)
        # then cut into equal-count bins on the rank. This guarantees exactly `ntiles` groups
        # regardless of mass points in the exposure distribution (e.g. many firms at 1.0 for
        # school_opt_share would collapse Q3/Q4 under value-based pd.qcut + duplicates="drop").
        print(f"[ntiles] Firms: {len(valid)}")
        ranks = valid.rank(method="first")
        codes = pd.cut(ranks, bins=ntiles, labels=False, include_lowest=True)
        actual_groups = int(codes.max()) + 1
        if actual_groups < ntiles:
            print(f"[ntiles] Note: requested {ntiles} bins but got {actual_groups}.")
        parts.append(pd.DataFrame({"c": valid.index, "ntile": (codes + 1).astype(int)}))
        # Print approximate exposure range per ntile for reference.
        tmp = pd.DataFrame({"exposure": valid.values, "ntile": (codes + 1).values})
        for q in sorted(tmp["ntile"].unique()):
            sub = tmp[tmp["ntile"] == q]["exposure"]
            print(f"  Q{q}: [{sub.min():.4g}, {sub.max():.4g}]  (n={len(sub)})")
        has_zero_group = False

    if not parts:
        raise ValueError("No firms remain after ntile assignment.")

    df = pd.concat(parts, ignore_index=True)
    actual_ntiles = int(df["ntile"].max())

    def _label(q: int, n: int) -> str:
        if has_zero_group and q == 1:
            return "Q1 (no OPT hires)"
        if q == n:
            return f"Q{n} (highest)"
        if q == (2 if has_zero_group else 1):
            return f"Q{q} (lowest positive)" if has_zero_group else "Q1 (lowest)"
        return f"Q{q}"

    df["ntile_label"] = df["ntile"].apply(lambda q: _label(q, actual_ntiles))

    print(f"[ntiles] Firm counts per ntile:")
    print(df.groupby(["ntile", "ntile_label"])["c"].count().to_string())

    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_NTILE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def _ntile_color_map(ntile_order: Sequence[str]) -> dict[str, str]:
    return {label: _NTILE_COLORS[i % len(_NTILE_COLORS)] for i, label in enumerate(ntile_order)}


def _plot_raw_means(
    panel: pd.DataFrame,
    outcome_col: str,
    event_year: int,
    ref_year: int,
    ntile_order: Sequence[str],
    ntile_colors: dict[str, str],
    slides_out: Optional[Path],
    oc_tag: str,
    version_tag: str,
) -> None:
    """
    Plot mean outcome by calendar year, stratified by ntile group.
    Each group's series is demeaned to its mean in ref_year.
    """
    # Aggregate mean, SE per (ntile_label, year).
    grp = panel.groupby(["ntile_label", "t"])[outcome_col]
    means = grp.mean().reset_index(name="mean_outcome")
    counts = grp.count().reset_index(name="n_obs")
    stds = grp.std().reset_index(name="std_outcome")

    stats = means.merge(counts, on=["ntile_label", "t"]).merge(stds, on=["ntile_label", "t"])
    stats["se"] = stats["std_outcome"] / np.sqrt(stats["n_obs"].clip(lower=1))

    # Demean each group's series relative to its mean in ref_year.
    ref = (
        stats[stats["t"] == ref_year][["ntile_label", "mean_outcome"]]
        .rename(columns={"mean_outcome": "ref_mean"})
    )
    stats = stats.merge(ref, on="ntile_label", how="left")
    stats["mean_demeaned"] = stats["mean_outcome"] - stats["ref_mean"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label in ntile_order:
        s = stats[stats["ntile_label"] == label].sort_values("t")
        if s.empty:
            continue
        n_firms = panel[panel["ntile_label"] == label]["c"].nunique()
        ax.errorbar(
            s["t"],
            s["mean_demeaned"],
            yerr=1.96 * s["se"].fillna(0),
            fmt="o-",
            capsize=3,
            color=ntile_colors.get(label),
            label=f"{label} (n={n_firms})",
        )

    ax.axvline(event_year, color="black", linestyle="--", linewidth=1, label=f"Event ({event_year})")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Mean {outcome_col} (demeaned to {ref_year})")
    ax.set_title(f"Outcome by OPT exposure ntile — {version_tag}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig("exposure_es_raw.png", slides_out, oc_tag, version_tag)
    plt.show()
    plt.close(fig)


def _plot_regression(
    coef_df: pd.DataFrame,
    outcome_col: str,
    event_year: int,
    ref_year: int,
    non_ref_ntiles: Sequence[str],
    ntile_colors: dict[str, str],
    slides_out: Optional[Path],
    oc_tag: str,
    version_tag: str,
) -> None:
    """
    Plot year × ntile interaction coefficients β_{τq} on a single axes.
    Each ntile q > Q1 gets one line with 95% CI error bars.
    Reference: Q1 firms in ref_year (coefficient = 0 by construction).
    """
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for ntile_label in non_ref_ntiles:
        s = coef_df[coef_df["ntile_label"] == ntile_label].sort_values("year")
        if s.empty:
            continue
        ax.errorbar(
            s["year"],
            s["coef"],
            yerr=1.96 * s["se"].fillna(0),
            fmt="o-",
            capsize=3,
            color=ntile_colors.get(ntile_label),
            label=ntile_label,
        )

    ax.axvline(event_year, color="black", linestyle="--", linewidth=1, label=f"Event ({event_year})")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Year × ntile coef vs Q1 in {ref_year} (± 1.96 SE)")
    ax.set_title(
        f"Year × OPT-exposure ntile interactions — {version_tag}\nOutcome: {outcome_col}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig("exposure_es_reg.png", slides_out, oc_tag, version_tag)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def _run_regression(
    panel: pd.DataFrame,
    outcome_col: str,
    ref_year: int,
    data_min_t: int,
    data_max_t: int,
) -> Optional[pd.DataFrame]:
    """
    Panel regression with firm fixed effects, year dummies, and year × ntile interactions.

    Model (after absorbing firm FE):
        y_jt = Σ_{τ≠ref} δ_τ 1[t=τ]
             + Σ_{q>1, τ≠ref} β_{τq} 1[t=τ] 1[ntile_j=q]
             + ε_jt

    Year dummies (δ_τ) capture the common year effect for the reference ntile (Q1).
    Interactions (β_{τq}) are the differential year-τ effect for ntile q vs Q1,
    relative to ref_year. Standard errors are clustered by firm.

    Note: ntile main effects are absorbed by firm FE (ntile is time-invariant per firm).

    Returns DataFrame with columns: year, ntile_label, coef, se, tstat, pval.
    Returns None if the regression cannot be run.
    """
    if PanelOLS is None:
        print("[regression] linearmodels not installed. Install with: pip install linearmodels")
        return None

    # Filter to data window and drop missing outcomes.
    work = (
        panel[panel["t"].between(data_min_t, data_max_t)]
        .dropna(subset=[outcome_col])
        .drop_duplicates(subset=["c", "t"])  # ensure one row per firm-year
        .reset_index(drop=True)
    )
    work[outcome_col] = work[outcome_col].astype(float)

    n_obs = len(work)
    n_firms = work["c"].nunique()
    n_years = work["t"].nunique()
    print(f"\n[regression] Outcome: {outcome_col} | N obs: {n_obs} | "
          f"N firms: {n_firms} | N years: {n_years}")

    if n_obs < 10:
        print("[regression] Too few observations. Skipping.")
        return None

    # Determine ntile order (sorted by ntile number, deduped).
    ntile_order_df = (
        work[["ntile", "ntile_label"]]
        .drop_duplicates()
        .sort_values("ntile")
    )
    ntile_order_all: list[str] = ntile_order_df["ntile_label"].tolist()
    ntile_ref = ntile_order_all[0]       # Q1 (lowest) — reference group
    non_ref_ntiles = ntile_order_all[1:]  # Q2, Q3, ... — groups we estimate interactions for

    years = sorted(work["t"].unique())
    non_ref_years = [y for y in years if y != ref_year]

    if not non_ref_years:
        print("[regression] No non-reference years in data window. Skipping.")
        return None
    if not non_ref_ntiles:
        print("[regression] Only one ntile group — no interactions to estimate. Skipping.")
        return None

    # Build year dummies (omit ref_year → reference year for Q1).
    year_dummies = pd.get_dummies(work["t"], prefix="yr", dtype=float)
    ref_yr_col = f"yr_{ref_year}"
    if ref_yr_col in year_dummies.columns:
        year_dummies = year_dummies.drop(columns=[ref_yr_col])

    # Build year × ntile interactions (omit ref_year and Q1 → already captured by year dummies).
    interact_cols: dict[str, pd.Series] = {}
    for yr in non_ref_years:
        for ntile_label in non_ref_ntiles:
            # Construct a safe column name (no spaces or parentheses).
            safe = ntile_label.replace(" ", "_").replace("(", "").replace(")", "")
            col_name = f"yr{yr}_x_{safe}"
            interact_cols[col_name] = (
                (work["t"] == yr).astype(float) * (work["ntile_label"] == ntile_label).astype(float)
            )

    exog = pd.concat(
        [year_dummies, pd.DataFrame(interact_cols, index=work.index)],
        axis=1,
    )

    # Drop zero-variance columns (e.g., if some ntile has no firms in a given year).
    zero_var = exog.columns[exog.std() == 0].tolist()
    if zero_var:
        print(f"[regression] Dropping {len(zero_var)} zero-variance columns: {zero_var[:5]}...")
        exog = exog.drop(columns=zero_var)

    # Set MultiIndex (c, t) required by PanelOLS.
    idx = pd.MultiIndex.from_arrays([work["c"].values, work["t"].values], names=["c", "t"])
    dep = pd.Series(work[outcome_col].values, index=idx, name=outcome_col)
    exog_indexed = exog.set_index(idx)

    try:
        model = PanelOLS(dependent=dep, exog=exog_indexed, entity_effects=True)
        result = model.fit(cov_type="clustered", cluster_entity=True)
    except Exception as exc:
        print(f"[regression] PanelOLS failed: {exc}")
        return None

    # Extract interaction coefficients and build result table.
    records: list[dict] = []
    for yr in non_ref_years:
        for ntile_label in non_ref_ntiles:
            safe = ntile_label.replace(" ", "_").replace("(", "").replace(")", "")
            col_name = f"yr{yr}_x_{safe}"
            if col_name not in result.params.index:
                continue  # might have been dropped as zero-variance
            records.append({
                "year": yr,
                "ntile_label": ntile_label,
                "coef": float(result.params[col_name]),
                "se": float(result.std_errors[col_name]),
                "tstat": float(result.tstats[col_name]),
                "pval": float(result.pvalues[col_name]),
            })

    if not records:
        print("[regression] No interaction coefficients extracted.")
        return None

    coef_df = pd.DataFrame(records)
    print(f"\n[regression] Interaction coefficients (year × ntile vs Q1 in {ref_year}):")
    print(coef_df.to_string(index=False))
    return coef_df


# ---------------------------------------------------------------------------
# Per-version / per-outcome pipeline
# ---------------------------------------------------------------------------

def _run_one_version(
    panel: pd.DataFrame,
    components: Optional[pd.DataFrame],
    outcome_col: str,
    version: str,
    cfg: dict,
    slides_out: Optional[Path],
    oc_tag: str,
    do_plot: bool,
) -> None:
    """
    Run the full exposure event study for one exposure version and one outcome column.

    panel is passed by value (caller should pass panel.copy() to avoid cross-version
    contamination from _ensure_derived_outcome).
    """
    year_min = int(cfg.get("exposure_year_min", 2010))
    year_max = int(cfg.get("exposure_year_max", 2015))
    ntiles = int(cfg.get("ntiles", 4))
    event_year = int(cfg.get("event_year", 2016))
    ref_year = int(cfg.get("ref_year", 2015))
    data_min_t = int(cfg.get("data_min_t", 2010))
    data_max_t = int(cfg.get("data_max_t", 2022))
    x_source_col = str(cfg.get("x_source_col", "masters_opt_hires_correction_aware"))
    version_tag = version.replace("_", "-")

    print(f"\n{'='*65}")
    print(f" Version: {version_tag}  |  Outcome: {outcome_col}")
    print(f"{'='*65}")

    # Derive outcome column if it doesn't exist yet (e.g. x_bin_any_nonzero).
    _ensure_derived_outcome(panel, outcome_col, x_source_col)
    if outcome_col not in panel.columns:
        print(f"[warn] Outcome column '{outcome_col}' not available. Skipping.")
        return

    # --- Compute exposure measure ---
    print(f"\n--- Computing exposure ({version}) ---")
    if version == "opt_hire_rate":
        exposure = _compute_exposure_opt_hire_rate(panel, year_min, year_max)
    elif version == "school_opt_share":
        if components is None:
            print("[warn] instrument_components not loaded — required for school_opt_share. Skipping.")
            return
        exposure = _compute_exposure_school_opt_share(components, year_min, year_max)
    else:
        raise ValueError(f"Unknown exposure version: {version!r}. "
                         "Choose 'opt_hire_rate', 'school_opt_share', or 'both'.")

    # --- Assign ntile groups ---
    # Separate zero-exposure firms into their own Q1 only for opt_hire_rate, where a
    # mass of exactly-zero firms would cause pd.qcut to collapse all bins.
    print(f"\n--- Assigning {ntiles} ntile groups ---")
    ntile_df = _assign_ntiles(exposure, ntiles, zero_separate=(version == "opt_hire_rate"))

    # Merge ntile labels onto the full panel (inner join → only firms with exposure).
    panel_with_ntile = panel.merge(ntile_df[["c", "ntile", "ntile_label"]], on="c", how="inner")
    print(f"\n[merge] Panel rows with ntile assignment: {len(panel_with_ntile)} | "
          f"firms: {panel_with_ntile['c'].nunique()} | "
          f"years: {panel_with_ntile['t'].min()}–{panel_with_ntile['t'].max()}")

    # Build ordered ntile label list and color map.
    ntile_order_df = (
        panel_with_ntile[["ntile", "ntile_label"]].drop_duplicates().sort_values("ntile")
    )
    ntile_order: list[str] = ntile_order_df["ntile_label"].tolist()
    ntile_colors = _ntile_color_map(ntile_order)
    non_ref_ntiles = ntile_order[1:]  # Q1 is reference in regression

    # Print summary of outcome by ntile.
    print(f"\n[summary] Mean {outcome_col} by ntile (full panel):")
    summary = panel_with_ntile.groupby("ntile_label")[outcome_col].agg(["mean", "std", "count"])
    print(summary.to_string())

    # --- Plot 1: raw means by ntile over time ---
    if do_plot:
        print(f"\n--- Plot 1: raw means by ntile over time ---")
        _plot_raw_means(
            panel_with_ntile,
            outcome_col,
            event_year,
            ref_year,
            ntile_order,
            ntile_colors,
            slides_out,
            oc_tag,
            version_tag,
        )

    # --- Regression: year FE + year × ntile interactions + firm FE ---
    print(f"\n--- Regression: year × ntile interactions + firm FE ---")
    coef_df = _run_regression(panel_with_ntile, outcome_col, ref_year, data_min_t, data_max_t)

    # Add ref_year as explicit zero-points so it appears on the regression plot.
    # (ref_year is the omitted year in the regression, so its coefficient is zero by construction.)
    if coef_df is not None and not coef_df.empty:
        ref_rows = pd.DataFrame([
            {"year": ref_year, "ntile_label": lbl, "coef": 0.0, "se": 0.0,
             "tstat": np.nan, "pval": np.nan}
            for lbl in non_ref_ntiles
        ])
        coef_df = pd.concat([ref_rows, coef_df], ignore_index=True)

    # --- Plot 2: regression interaction coefficients ---
    if do_plot and coef_df is not None and not coef_df.empty:
        print(f"\n--- Plot 2: regression interaction coefficients ---")
        _plot_regression(
            coef_df,
            outcome_col,
            event_year,
            ref_year,
            non_ref_ntiles,
            ntile_colors,
            slides_out,
            oc_tag,
            version_tag,
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exposure-based firm-level event study around 2016 OPT policy."
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    p.add_argument(
        "--outcome-col", type=str, default=None,
        help="Comma-separated outcome column(s). Overrides exposure_event_study.outcome_cols.",
    )
    p.add_argument(
        "--exposure-version", type=str, default=None,
        choices=("opt_hire_rate", "school_opt_share", "both"),
        help="Exposure measure to use. Overrides exposure_event_study.exposure_version.",
    )
    p.add_argument(
        "--ntiles", type=int, default=None,
        help="Number of ntile groups. Overrides exposure_event_study.ntiles.",
    )
    p.add_argument(
        "--no-plot", action="store_true", default=False,
        help="Skip all plot generation (regression still runs).",
    )

    # IPython/Jupyter-safe: avoid crashing on kernel flags like '-f kernel.json'.
    if args is None:
        argv0 = Path(sys.argv[0]).name.lower() if sys.argv else ""
        has_kernel_argv = (
            len(sys.argv) >= 3
            and sys.argv[1] == "-f"
            and str(sys.argv[2]).lower().endswith(".json")
        )
        in_ipython_ctx = (
            "IPython" in sys.modules
            or "ipykernel" in sys.modules
            or "ipykernel_launcher" in argv0
            or has_kernel_argv
        )
        if in_ipython_ctx:
            parsed, unknown = p.parse_known_args()
            if unknown:
                print(f"[info] Ignoring unknown IPython args: {unknown}")
            return parsed
    return p.parse_args(args)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(cli_args: Optional[Iterable[str]] = None) -> None:
    """Run the exposure event study. Pass cli_args=[] to run from iPython with config defaults."""
    import time
    t0 = time.time()

    args = _parse_args(cli_args)
    cfg_full = load_config(args.config)
    paths_cfg = get_cfg_section(cfg_full, "paths")
    exp_cfg = get_cfg_section(cfg_full, "exposure_event_study")

    # Override config with CLI flags.
    if args.ntiles is not None:
        exp_cfg["ntiles"] = args.ntiles
    if args.exposure_version is not None:
        exp_cfg["exposure_version"] = args.exposure_version

    # Resolve slides output directory.
    slides_raw = str(exp_cfg.get("slides_out_dir", "")).strip()
    if slides_raw and slides_raw.lower() not in {"none", "null", ""}:
        root = str(Path(__file__).resolve().parents[2])
        slides_out: Optional[Path] = Path(slides_raw.replace("{root}", root))
        slides_out.mkdir(parents=True, exist_ok=True)
        print(f"[main] Slides output: {slides_out}")
    else:
        slides_out = None
        print("[main] slides_out_dir not set — plots will display only, not saved.")

    # Determine exposure version(s).
    version_raw = str(exp_cfg.get("exposure_version", "opt_hire_rate"))
    versions: list[str] = (
        ["opt_hire_rate", "school_opt_share"] if version_raw == "both" else [version_raw]
    )

    # Determine outcome columns.
    outcome_raw = args.outcome_col or exp_cfg.get(
        "outcome_cols", ["x_bin_any_nonzero", "log1p_y_cst_lag0", "log1p_y_new_hires_lag0"]
    )
    if isinstance(outcome_raw, str):
        outcome_cols = [v.strip() for v in outcome_raw.split(",") if v.strip()]
    else:
        outcome_cols = [str(v).strip() for v in outcome_raw if str(v).strip()]

    do_plot = not args.no_plot

    print(f"[main] Exposure versions: {versions}")
    print(f"[main] Outcome columns:   {outcome_cols}")
    print(f"[main] Ntiles: {exp_cfg.get('ntiles', 4)} | "
          f"Exposure window: {exp_cfg.get('exposure_year_min', 2010)}–"
          f"{exp_cfg.get('exposure_year_max', 2015)} | "
          f"Event year: {exp_cfg.get('event_year', 2016)}")

    # -----------------------------------------------------------------------
    # Load analysis panel via DuckDB.
    # -----------------------------------------------------------------------
    panel_path = _resolve_path(paths_cfg, "analysis_panel")
    print(f"\n[main] Loading analysis panel from:\n  {panel_path}")
    con = ddb.connect()
    panel: pd.DataFrame = con.execute(
        f"SELECT * FROM read_parquet('{_escape(panel_path)}')"
    ).df()
    con.close()

    # Coerce types; drop rows with null firm or year identifier.
    panel["c"] = pd.to_numeric(panel["c"], errors="coerce")
    panel["t"] = pd.to_numeric(panel["t"], errors="coerce")
    panel = panel.dropna(subset=["c", "t"])
    panel["c"] = panel["c"].astype(int)
    panel["t"] = panel["t"].astype(int)
    print(f"[main] Panel: {len(panel):,} rows | {panel['c'].nunique():,} firms | "
          f"years {int(panel['t'].min())}–{int(panel['t'].max())}")
    print(f"[main] Panel columns: {list(panel.columns)}")

    # -----------------------------------------------------------------------
    # Load instrument_components if school_opt_share version is requested.
    # -----------------------------------------------------------------------
    components: Optional[pd.DataFrame] = None
    if "school_opt_share" in versions:
        comp_path = _resolve_path(paths_cfg, "instrument_components")
        print(f"\n[main] Loading instrument_components from:\n  {comp_path}")
        con2 = ddb.connect()
        components = con2.execute(
            f"SELECT c, k, t, g_kt, n_transitions_full, total_new_hires_full "
            f"FROM read_parquet('{_escape(comp_path)}')"
        ).df()
        con2.close()
        components["c"] = pd.to_numeric(components["c"], errors="coerce")
        components["t"] = pd.to_numeric(components["t"], errors="coerce")
        components = components.dropna(subset=["c", "t", "k"])
        components["c"] = components["c"].astype(int)
        components["t"] = components["t"].astype(int)
        print(f"[main] Components: {len(components):,} rows | "
              f"{components['k'].nunique():,} schools | "
              f"{components['c'].nunique():,} firms")

    # -----------------------------------------------------------------------
    # Run pipeline: for each version × each outcome column.
    # -----------------------------------------------------------------------
    total_passes = len(versions) * len(outcome_cols)
    pass_n = 0
    for version in versions:
        for outcome_col in outcome_cols:
            pass_n += 1
            print(f"\n[main] Pass {pass_n}/{total_passes}: "
                  f"version={version}, outcome={outcome_col}")
            oc_tag = outcome_col.replace("/", "_").replace(" ", "_")
            _run_one_version(
                panel.copy(),  # copy so derived-column side effects don't bleed across passes
                components,
                outcome_col,
                version,
                exp_cfg,
                slides_out,
                oc_tag,
                do_plot,
            )

    elapsed = time.time() - t0
    print(f"\n[main] Done. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
