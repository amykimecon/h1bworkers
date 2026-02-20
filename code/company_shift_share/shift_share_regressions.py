"""Company-level shift-share regressions using the analysis panel output."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

try:
    from linearmodels.panel import PanelOLS
    try:
        from linearmodels.panel import IV2SLS as PanelIV2SLS
    except ImportError:
        PanelIV2SLS = None
    from linearmodels.iv import IV2SLS
except ImportError:  # pragma: no cover - handled at runtime with a clear message
    PanelOLS = None
    PanelIV2SLS = None
    IV2SLS = None

try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
except ModuleNotFoundError:
    # Allow direct execution when repo root is not already on PYTHONPATH.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config


def _resolve_cfg_path(paths_cfg: dict, key: str) -> Path:
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"Config paths.{key} must be set.")
    return Path(value)


def _parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shift-share regressions on company analysis panel.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--analysis-panel",
        type=Path,
        default=None,
        help="Path to analysis_panel.parquet (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--outcome-prefix",
        default=None,
        help="Outcome column prefix (default: y_cst_lag).",
    )
    parser.add_argument(
        "--lag-start",
        type=int,
        default=None,
        help="Minimum lag to include (inclusive).",
    )
    parser.add_argument(
        "--lag-end",
        type=int,
        default=None,
        help="Maximum lag to include (inclusive).",
    )
    parser.add_argument(
        "--instrument",
        default=None,
        choices=("z_ct", "z_ct_all", "z_ct_intl", "z_ct_full", "z_ct_all_full", "z_ct_intl_full"),
        help="Instrument column to use.",
    )
    parser.add_argument(
        "--dependent",
        default=None,
        help="Endogenous/dependent variable for IV (default: masters_opt_hires).",
    )
    parser.add_argument(
        "--start-t",
        type=int,
        default=None,
        help="Minimum t to include (inclusive).",
    )
    parser.add_argument(
        "--end-t",
        type=int,
        default=None,
        help="Maximum t to include (inclusive).",
    )
    parser.add_argument(
        "--no-fe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip company and time fixed effects.",
    )
    parser.add_argument(
        "--controls",
        default=None,
        help="Comma-separated controls to add (default: none).",
    )
    parser.add_argument(
        "--balanced-panel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restrict to firms observed in every sampled year after filters.",
    )
    parser.add_argument(
        "--min-employees-2010",
        type=int,
        default=None,
        help="Restrict to firms with at least this many employees in t=2010 (uses lagged outcome).",
    )
    parser.add_argument(
        "--employees-lag",
        type=int,
        default=None,
        help="Lag to use for the 2010 employee filter (default: 0 uses y_cst_lag0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write regression summary CSV.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Optional path to write a coefficient plot (e.g., .png or .pdf).",
    )
    parser.add_argument(
        "--plot-coef",
        default="iv",
        choices=("iv", "ols", "rf", "first_stage"),
        help="Which coefficient to plot (default: iv).",
    )
    parser.add_argument(
        "--plot-title",
        default=None,
        help="Optional title for the coefficient plot.",
    )
    parser.add_argument(
        "--latex-out",
        type=Path,
        default=None,
        help="Optional path to write a LaTeX table.",
    )
    parser.add_argument(
        "--latex-caption",
        default="Shift-share regression results",
        help="Caption for the LaTeX table.",
    )
    parser.add_argument(
        "--latex-label",
        default="tab:shift_share",
        help="Label for the LaTeX table.",
    )
    parser.add_argument(
        "--latex-size",
        default="\\scriptsize",
        help="LaTeX size command for the table (default: \\scriptsize).",
    )
    parser.add_argument(
        "--latex-lag",
        type=int,
        default=None,
        help="Optional lag to show a single-row table (e.g., 2).",
    )
    parser.add_argument(
        "--latex-x-label",
        default=None,
        help="Optional label for the endogenous variable row (defaults to --dependent).",
    )
    parser.add_argument(
        "--latex-z-label",
        default=None,
        help="Optional label for the instrument row (defaults to --instrument).",
    )
    parser.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run shift-share diagnostics (balance, Rotemberg weights, top shifts/shares).",
    )
    parser.add_argument(
        "--instrument-components",
        type=Path,
        default=None,
        help="Path to instrument_components.parquet for diagnostics.",
    )
    parser.add_argument(
        "--diagnostics-top-n",
        type=int,
        default=None,
        help="Top N items to show in diagnostics tables.",
    )
    parser.add_argument(
        "--diagnostics-pre-lag-1",
        type=int,
        default=None,
        help="Pre-period lag 1 for balance test (default: 1).",
    )
    parser.add_argument(
        "--diagnostics-pre-lag-2",
        type=int,
        default=None,
        help="Pre-period lag 2 for balance test (default: 2).",
    )
    parser.add_argument(
        "--diagnostics-component-col",
        default=None,
        help="Component column in instrument_components (default: inferred from --instrument).",
    )
    parser.add_argument(
        "--diagnostics-share-col",
        default=None,
        help="Share column in instrument_components (default: share_ck).",
    )
    parser.add_argument(
        "--require-nonzero-x-all-years",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restrict to firms with dependent variable > 0 in every sampled year.",
    )
    parser.add_argument(
        "--fill-missing-xy-zero",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replace missing dependent (x) and outcome (y) values with 0 before regressions.",
    )
    parser.add_argument(
        "--median-regression",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run median (quantile=0.5) regressions instead of IV/OLS pipeline.",
    )
    parser.add_argument(
        "--ihs-y",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply inverse-hyperbolic-sine transform to outcome y columns before regressions.",
    )
    return parser.parse_args(args)


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython
    except Exception:
        return False
    return get_ipython() is not None


def _lag_suffix(lag: int) -> str:
    return f"m{abs(lag)}" if lag < 0 else str(lag)


def _outcome_columns(prefix: str, lag_start: int, lag_end: int) -> list[str]:
    return [f"{prefix}{_lag_suffix(lag)}" for lag in range(int(lag_start), int(lag_end) + 1)]


def _parse_controls(raw: str) -> list[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _filter_panel(panel: pd.DataFrame, start_t: int | None, end_t: int | None) -> pd.DataFrame:
    if start_t is None and end_t is None:
        return panel
    mask = pd.Series(True, index=panel.index)
    if start_t is not None:
        mask &= panel["t"] >= start_t
    if end_t is not None:
        mask &= panel["t"] <= end_t
    return panel[mask].copy()


def _filter_by_employees_2010(panel: pd.DataFrame, min_employees: int, employees_lag: int) -> pd.DataFrame:
    outcome_col = f"y_cst_lag{_lag_suffix(int(employees_lag))}"
    if outcome_col not in panel.columns:
        raise ValueError(f"{outcome_col} not found in panel for employee filter.")
    base = panel[panel["t"] == 2012][["c", outcome_col]].dropna()
    keep = base[base[outcome_col] >= min_employees]["c"].unique()
    return panel[panel["c"].isin(keep)].copy()


def _filter_firms_nonzero_in_all_years(panel: pd.DataFrame, x_col: str) -> pd.DataFrame:
    if x_col not in panel.columns:
        raise ValueError(f"{x_col} not found in panel for nonzero-every-year filter.")
    work = panel.copy()
    required_years = work["t"].dropna().unique()
    n_years = len(required_years)
    if n_years == 0:
        return work
    by_firm = (
        work.groupby("c")
        .agg(
            n_years=("t", "nunique"),
            n_nonzero=(x_col, lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
        )
        .reset_index()
    )
    keep = by_firm[(by_firm["n_years"] == n_years) & (by_firm["n_nonzero"] == n_years)]["c"].unique()
    return work[work["c"].isin(keep)].copy()


def _fill_missing_xy_with_zero(panel: pd.DataFrame, x_col: str, y_cols: Sequence[str]) -> pd.DataFrame:
    work = panel.copy()
    cols = [x_col] + list(y_cols)
    missing = [col for col in cols if col not in work.columns]
    if missing:
        raise ValueError(f"Columns not found for missing-value fill: {missing}")
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    return work


def _transform_y_ihs(panel: pd.DataFrame, y_cols: Sequence[str]) -> pd.DataFrame:
    work = panel.copy()
    missing = [col for col in y_cols if col not in work.columns]
    if missing:
        raise ValueError(f"Outcome columns not found for IHS transform: {missing}")
    for col in y_cols:
        work[col] = np.arcsinh(pd.to_numeric(work[col], errors="coerce"))
    return work


def _prepare_panel_for_instrument(
    panel: pd.DataFrame,
    instrument: str,
    enforce_balanced_panel: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    if instrument not in panel.columns:
        raise ValueError(f"Instrument column '{instrument}' not found in panel.")
    work = panel.copy()
    firms_before = int(work["c"].nunique())
    rows_before = int(len(work))

    # Drop firms where the chosen instrument is missing in every sampled year.
    has_any_instrument_series = work.groupby("c")[instrument].apply(lambda s: s.notna().any())
    keep_instrument_firms = has_any_instrument_series[has_any_instrument_series].index
    dropped_all_null_firms = firms_before - int(len(keep_instrument_firms))
    work = work[work["c"].isin(keep_instrument_firms)].copy()
    print(f"Dropped all-null instrument firms: {dropped_all_null_firms}")
    
    # Fill partial missing years with zeros.
    print(f"Filled partial missing instrument values with zeros: {work[instrument].isna().sum()} missing values before fill.")
    work[instrument] = work[instrument].fillna(0)

    dropped_unbalanced_firms = 0
    if not enforce_balanced_panel:
        if verbose:
            print(
                f"Panel preprocessing ({instrument}): firms {firms_before:,} -> {work['c'].nunique():,}, "
                f"rows {rows_before:,} -> {len(work):,}; "
                f"dropped all-null instrument firms={dropped_all_null_firms:,}, dropped for balance=0."
            )
        return work

    required_years = work["t"].dropna().unique()
    n_years = len(required_years)
    if n_years == 0:
        if verbose:
            print(
                f"Panel preprocessing ({instrument}): firms {firms_before:,} -> {work['c'].nunique():,}, "
                f"rows {rows_before:,} -> {len(work):,}; "
                f"dropped all-null instrument firms={dropped_all_null_firms:,}, dropped for balance=0."
            )
        return work
    c_year_counts = work.groupby("c")["t"].nunique()
    keep_companies = c_year_counts[c_year_counts == n_years].index
    firms_after_instrument = int(work["c"].nunique())
    dropped_unbalanced_firms = firms_after_instrument - int(len(keep_companies))
    work = work[work["c"].isin(keep_companies)].copy()
    if verbose:
        print(
            f"Panel preprocessing ({instrument}): firms {firms_before:,} -> {work['c'].nunique():,}, "
            f"rows {rows_before:,} -> {len(work):,}; "
            f"dropped all-null instrument firms={dropped_all_null_firms:,}, "
            f"dropped for balance={dropped_unbalanced_firms:,}."
        )
    return work


_STATE_NAME_TO_ABBR = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "dc": "DC",
}

_STATE_ABBR_ORDER = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


def _normalize_state_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if len(raw) == 2:
        code = raw.upper()
        return code if code in _STATE_ABBR_ORDER else None
    return _STATE_NAME_TO_ABBR.get(raw.lower())


def _mode_state_by_company(
    employer_crosswalk: pd.DataFrame,
    company_col: str = "preferred_rcid",
    state_col: str = "f1_state_clean",
) -> pd.DataFrame:
    needed = {company_col, state_col}
    missing = needed.difference(employer_crosswalk.columns)
    if missing:
        raise ValueError(f"Missing required columns in employer_crosswalk: {sorted(missing)}")
    state_map = employer_crosswalk[[company_col, state_col]].dropna().copy()
    state_map[company_col] = pd.to_numeric(state_map[company_col], errors="coerce")
    state_map = state_map.dropna(subset=[company_col])
    state_map["state"] = state_map[state_col].map(_normalize_state_code)
    state_map = state_map.dropna(subset=["state"])
    if state_map.empty:
        return pd.DataFrame(columns=["c", "state", "state_obs"])
    counts = (
        state_map.groupby([company_col, "state"], as_index=False)
        .size()
        .rename(columns={"size": "state_obs"})
        .sort_values([company_col, "state_obs", "state"], ascending=[True, False, True])
    )
    mode = counts.groupby(company_col, as_index=False).first()
    mode = mode.rename(columns={company_col: "c"})
    mode["c"] = mode["c"].astype(int)
    return mode[["c", "state", "state_obs"]]


def _mode_name_by_company(
    employer_crosswalk: pd.DataFrame,
    company_col: str = "preferred_rcid",
    name_col: str = "f1_empname_clean",
) -> pd.DataFrame:
    needed = {company_col, name_col}
    missing = needed.difference(employer_crosswalk.columns)
    if missing:
        raise ValueError(f"Missing required columns in employer_crosswalk: {sorted(missing)}")
    name_map = employer_crosswalk[[company_col, name_col]].dropna().copy()
    name_map[company_col] = pd.to_numeric(name_map[company_col], errors="coerce")
    name_map = name_map.dropna(subset=[company_col])
    name_map["firm_name"] = name_map[name_col].astype(str).str.strip()
    name_map = name_map[name_map["firm_name"] != ""]
    if name_map.empty:
        return pd.DataFrame(columns=["c", "firm_name", "name_obs"])
    counts = (
        name_map.groupby([company_col, "firm_name"], as_index=False)
        .size()
        .rename(columns={"size": "name_obs"})
        .sort_values([company_col, "name_obs", "firm_name"], ascending=[True, False, True])
    )
    mode = counts.groupby(company_col, as_index=False).first()
    mode = mode.rename(columns={company_col: "c"})
    mode["c"] = mode["c"].astype(int)
    return mode[["c", "firm_name", "name_obs"]]


def join_company_names(
    panel: pd.DataFrame,
    employer_crosswalk: pd.DataFrame | None = None,
    employer_crosswalk_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Join a modal company name (firm_name) onto panel using company id c.
    """
    if "c" not in panel.columns:
        raise ValueError("Panel must include column 'c' to join company names.")
    if employer_crosswalk is None and employer_crosswalk_path is not None:
        employer_crosswalk = pd.read_parquet(Path(employer_crosswalk_path))
    if employer_crosswalk is None:
        raise ValueError("Provide employer_crosswalk or employer_crosswalk_path to join company names.")

    name_map = _mode_name_by_company(employer_crosswalk)
    work = panel.copy()
    work["c"] = pd.to_numeric(work["c"], errors="coerce")
    work = work.dropna(subset=["c"])
    work["c"] = work["c"].astype(int)
    return work.merge(name_map[["c", "firm_name"]], on="c", how="left")


def firm_sample_diagnostics_plots(
    panel: pd.DataFrame,
    employer_crosswalk: pd.DataFrame | None = None,
    employer_crosswalk_path: Path | str | None = None,
    size_col: str = "y_cst_lag0",
    year_a: int = 2011,
    year_b: int = 2019,
    out_dir: Path | str | None = None,
    hist_bins: int = 50,
    show: bool = False,
) -> dict[str, object]:
    """
    Plot diagnostics for firms in the shift-share sample:
      1) Histograms of firm size in year_a and year_b.
      2) Heat map of firm counts by state (dominant state per firm).
    """
    required_panel = {"c", "t", size_col}
    missing_panel = required_panel.difference(panel.columns)
    if missing_panel:
        raise ValueError(f"Panel is missing required columns: {sorted(missing_panel)}")

    outdir = None if out_dir is None else Path(out_dir)
    auto_show = bool(show or outdir is None)
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    work = panel[["c", "t", size_col]].copy()
    work["c"] = pd.to_numeric(work["c"], errors="coerce")
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work[size_col] = np.log(pd.to_numeric(work[size_col], errors="coerce") + 1)
    work = work.dropna(subset=["c", "t", size_col])
    work["c"] = work["c"].astype(int)
    work["t"] = work["t"].astype(int)

    firm_size_a = (
        work[work["t"] == int(year_a)]
        .groupby("c", as_index=False)[size_col]
        .max()
        .rename(columns={size_col: "firm_size"})
    )
    firm_size_b = (
        work[work["t"] == int(year_b)]
        .groupby("c", as_index=False)[size_col]
        .max()
        .rename(columns={size_col: "firm_size"})
    )

    fig_hist, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    axes[0].hist(firm_size_a["firm_size"], bins=int(hist_bins), color="#4C78A8", alpha=0.85)
    axes[0].set_title(f"Firm size in {int(year_a)}")
    axes[0].set_xlabel(size_col)
    axes[0].set_ylabel("Number of firms")
    axes[1].hist(firm_size_b["firm_size"], bins=int(hist_bins), color="#F58518", alpha=0.85)
    axes[1].set_title(f"Firm size in {int(year_b)}")
    axes[1].set_xlabel(size_col)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig_hist.tight_layout()

    hist_path = None
    if outdir is not None:
        hist_path = outdir / f"firm_size_hist_{int(year_a)}_{int(year_b)}.png"
        fig_hist.savefig(hist_path, dpi=160, bbox_inches="tight")
    if auto_show:
        plt.show()
    plt.close(fig_hist)

    if employer_crosswalk is None and employer_crosswalk_path is not None:
        employer_crosswalk = pd.read_parquet(Path(employer_crosswalk_path))
    if employer_crosswalk is None:
        raise ValueError("Provide employer_crosswalk or employer_crosswalk_path to build state heat map.")

    state_map = _mode_state_by_company(employer_crosswalk)
    sample_firms = pd.DataFrame({"c": sorted(work["c"].unique())})
    firm_states = sample_firms.merge(state_map, on="c", how="left")
    firm_states = firm_states.dropna(subset=["state"])
    state_counts = (
        firm_states.groupby("state", as_index=False)["c"]
        .nunique()
        .rename(columns={"c": "n_firms"})
    )

    heat_df = pd.DataFrame({"state": _STATE_ABBR_ORDER})
    heat_df = heat_df.merge(state_counts, on="state", how="left")
    heat_df["n_firms"] = heat_df["n_firms"].fillna(0).astype(int)

    heat_path = None
    try:
        import plotly.express as px  # type: ignore

        fig_map = px.choropleth(
            heat_df,
            locations="state",
            locationmode="USA-states",
            color="n_firms",
            scope="usa",
            color_continuous_scale="YlOrRd",
            labels={"n_firms": "Firm count", "state": "State"},
            title="Firms by state",
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        if outdir is not None:
            heat_path = outdir / "firm_state_map.html"
            fig_map.write_html(str(heat_path), include_plotlyjs="cdn")
        if auto_show:
            fig_map.show()
    except Exception:
        # Fallback if plotly is unavailable in the runtime.
        arr = heat_df["n_firms"].to_numpy().reshape(-1, 1)
        fig_heat, ax = plt.subplots(figsize=(3.6, 10.5))
        im = ax.imshow(arr, aspect="auto", cmap="YlOrRd")
        ax.set_title("Firms by state")
        ax.set_xticks([0])
        ax.set_xticklabels(["n_firms"])
        ax.set_yticks(np.arange(len(heat_df)))
        ax.set_yticklabels(heat_df["state"])
        cbar = fig_heat.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cbar.set_label("Firm count")
        fig_heat.tight_layout()
        if outdir is not None:
            heat_path = outdir / "firm_state_heatmap.png"
            fig_heat.savefig(heat_path, dpi=160, bbox_inches="tight")
        if auto_show:
            plt.show()
        plt.close(fig_heat)

    return {
        "firm_size_year_a": firm_size_a,
        "firm_size_year_b": firm_size_b,
        "state_counts": state_counts.sort_values("n_firms", ascending=False),
        "histogram_path": hist_path,
        "heatmap_path": heat_path,
    }


def _build_exog(panel: pd.DataFrame, controls: Sequence[str], include_const: bool) -> pd.DataFrame:
    exog = panel[list(controls)].copy() if controls else pd.DataFrame(index=panel.index)
    if include_const:
        exog = sm.add_constant(exog, has_constant="add")
    return exog


def _demean_two_way(panel: pd.DataFrame, cols: Sequence[str], entity: str = "c", time: str = "t") -> pd.DataFrame:
    """
    Exact two-way within transformation for additive entity/time effects.
    """
    cols = list(dict.fromkeys(cols))
    overall = panel[cols].mean()
    entity_mean = panel.groupby(entity)[cols].transform("mean")
    time_mean = panel.groupby(time)[cols].transform("mean")
    panel[cols] = panel[cols] - entity_mean - time_mean + overall
    return panel


def _balance_on_pretrend(
    panel: pd.DataFrame,
    instrument: str,
    pre_col_1: str,
    pre_col_2: str,
    include_fixed_effects: bool = True,
    controls: Sequence[str] | None = None,
) -> pd.DataFrame:
    controls = list(controls or [])
    needed = ["c", "t", instrument, pre_col_1, pre_col_2] + controls
    work = panel[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work.empty:
        return pd.DataFrame()
    work["pre_change"] = work[pre_col_1] - work[pre_col_2]
    if include_fixed_effects:
        work = work.set_index(["c", "t"])
        exog = _build_exog(work, controls, include_const=False).join(work[[instrument]])
        res = PanelOLS(
            dependent=work["pre_change"],
            exog=exog,
            entity_effects=True,
            time_effects=True,
        ).fit(cov_type="clustered", cluster_entity=True)
        coef = res.params[instrument]
        se = res.std_errors[instrument]
        tstat = res.tstats[instrument]
    else:
        exog = _build_exog(work, controls, include_const=True)
        exog = exog.join(work[[instrument]])
        res = sm.OLS(work["pre_change"], exog).fit()
        coef = res.params[instrument]
        se = res.bse[instrument]
        tstat = res.tvalues[instrument]
    pval = _pvalue_from_tstat(float(tstat))
    return pd.DataFrame(
        [
            {
                "pre_col_1": pre_col_1,
                "pre_col_2": pre_col_2,
                "coef": coef,
                "se": se,
                "p_value": pval,
                "n_obs": int(work.shape[0]),
            }
        ]
    )


def _rotemberg_weights(
    panel: pd.DataFrame,
    components: pd.DataFrame,
    instrument: str,
    dependent: str,
    include_fixed_effects: bool = True,
    controls: Sequence[str] | None = None,
    component_col: str = "z_ct_component",
) -> pd.DataFrame:
    controls = list(controls or [])
    needed = ["c", "t", dependent, instrument] + controls
    work = panel[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work.empty:
        return pd.DataFrame()
    work = work.set_index(["c", "t"])
    if include_fixed_effects:
        exog = _build_exog(work, controls, include_const=False)
        if exog.empty:
            x_tilde = _demean_two_way(work.reset_index(), [dependent]).set_index(["c", "t"])[dependent]
        else:
            res = PanelOLS(
                dependent=work[dependent],
                exog=exog,
                entity_effects=True,
                time_effects=True,
            ).fit(cov_type="clustered", cluster_entity=True)
            x_tilde = res.resids
    else:
        exog = _build_exog(work, controls, include_const=True)
        res = sm.OLS(work[dependent], exog).fit()
        x_tilde = res.resid

    merged = components.merge(
        x_tilde.reset_index().rename(columns={0: "x_tilde", dependent: "x_tilde"}),
        on=["c", "t"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    merged = merged[["k", "c", "t", component_col, "x_tilde"]].dropna()
    denom = (merged[component_col] * merged["x_tilde"]).sum()
    if denom == 0 or not np.isfinite(denom):
        return pd.DataFrame()
    weights = (
        merged.groupby("k", as_index=False)
        .apply(lambda df: (df[component_col] * df["x_tilde"]).sum(), include_groups=False)
        .rename(columns={None: "weight"})
    )
    weights["abs_weight"] = weights["weight"].abs()
    weights = weights.sort_values("abs_weight", ascending=False)
    weights["weight"] = weights["weight"] / denom
    weights["abs_weight"] = weights["abs_weight"] / abs(denom)
    return weights


def _top_component_contributions(
    components: pd.DataFrame,
    component_col: str,
    top_n: int = 10,
) -> pd.DataFrame:
    if components.empty:
        return pd.DataFrame()
    contrib = (
        components.groupby(["k", "t"], as_index=False)[component_col]
        .sum()
        .rename(columns={component_col: "contribution"})
    )
    contrib["abs_contribution"] = contrib["contribution"].abs()
    return contrib.sort_values("abs_contribution", ascending=False).head(top_n)


def _top_share_weights(
    components: pd.DataFrame,
    share_col: str,
    top_n: int = 10,
) -> pd.DataFrame:
    if components.empty:
        return pd.DataFrame()
    shares = (
        components.groupby("k", as_index=False)[share_col]
        .mean()
        .rename(columns={share_col: "avg_share"})
    )
    return shares.sort_values("avg_share", ascending=False).head(top_n)


def shift_share_diagnostics(
    panel: pd.DataFrame,
    instrument_components: pd.DataFrame | None = None,
    instrument: str = "z_ct",
    dependent: str = "masters_opt_hires",
    outcome_prefix: str = "y_cst_lag",
    pre_lag_1: int = 1,
    pre_lag_2: int = 2,
    include_fixed_effects: bool = True,
    controls: Sequence[str] | None = None,
    component_col: str | None = None,
    share_col: str = "share_ck",
    top_n: int = 10,
    print_summary: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Produce shift-share diagnostics:
      1) Balance of instrument on pre-period change in outcome (lag1 - lag2).
      2) Rotemberg-style weights by shock k (based on cov(z_k, x_tilde)).
      3) Largest shock-time contributions to the instrument.
      4) Universities with largest average base shares.
    """
    component_col = component_col or f"{instrument}_component"
    diagnostics: dict[str, pd.DataFrame] = {}
    pre_col_1 = f"{outcome_prefix}{_lag_suffix(pre_lag_1)}"
    pre_col_2 = f"{outcome_prefix}{_lag_suffix(pre_lag_2)}"
    if pre_col_1 in panel.columns and pre_col_2 in panel.columns and instrument in panel.columns:
        balance = _balance_on_pretrend(
            panel,
            instrument=instrument,
            pre_col_1=pre_col_1,
            pre_col_2=pre_col_2,
            include_fixed_effects=include_fixed_effects,
            controls=controls,
        )
    else:
        balance = pd.DataFrame()
    diagnostics["balance"] = balance

    if instrument_components is not None and not instrument_components.empty:
        panel_years = panel["t"].dropna().unique()
        panel_companies = panel["c"].dropna().unique()
        components_sample = instrument_components
        if "t" in components_sample.columns:
            components_sample = components_sample[components_sample["t"].isin(panel_years)]
        if "c" in components_sample.columns:
            components_sample = components_sample[components_sample["c"].isin(panel_companies)]
        if component_col in components_sample.columns:
            rotemberg = _rotemberg_weights(
                panel,
                components_sample,
                instrument=instrument,
                dependent=dependent,
                include_fixed_effects=include_fixed_effects,
                controls=controls,
                component_col=component_col,
            )
            diagnostics["rotemberg_weights"] = rotemberg.head(top_n)
            diagnostics["top_shifts"] = _top_component_contributions(
                components_sample, component_col=component_col, top_n=top_n
            )
        else:
            diagnostics["rotemberg_weights"] = pd.DataFrame()
            diagnostics["top_shifts"] = pd.DataFrame()
        if share_col in components_sample.columns:
            diagnostics["top_shares"] = _top_share_weights(
                components_sample, share_col=share_col, top_n=top_n
            )
        else:
            diagnostics["top_shares"] = pd.DataFrame()
    else:
        diagnostics["rotemberg_weights"] = pd.DataFrame()
        diagnostics["top_shifts"] = pd.DataFrame()
        diagnostics["top_shares"] = pd.DataFrame()

    if print_summary:
        if not diagnostics["balance"].empty:
            print("Pre-trend balance:")
            print(diagnostics["balance"].to_string(index=False))
        if instrument_components is None or instrument_components.empty:
            print("Diagnostics note: instrument_components not provided or empty; skipping weights/shifts/shares.")
        else:
            if component_col not in instrument_components.columns:
                print(f"Diagnostics note: '{component_col}' not in instrument_components; skipping weights/shifts.")
            if share_col not in instrument_components.columns:
                print(f"Diagnostics note: '{share_col}' not in instrument_components; skipping share table.")
        if not diagnostics["rotemberg_weights"].empty:
            print("Top Rotemberg-style weights:")
            print(diagnostics["rotemberg_weights"].to_string(index=False))
        if not diagnostics["top_shifts"].empty:
            print("Top shift contributions (k,t):")
            print(diagnostics["top_shifts"].to_string(index=False))
        if not diagnostics["top_shares"].empty:
            print("Top average base shares by k:")
            print(diagnostics["top_shares"].to_string(index=False))

    return diagnostics

def _extract_lag(outcome: str, prefix: str) -> int | None:
    if not outcome.startswith(prefix):
        return None
    suffix = outcome[len(prefix):]
    if suffix.isdigit():
        return int(suffix)
    if suffix.startswith("m") and suffix[1:].isdigit():
        return -int(suffix[1:])
    return None


def _plot_results(
    results: pd.DataFrame,
    outcome_prefix: str,
    coef_key: str,
    out_path: Path | None = None,
    title: str | None = None,
    show: bool = False,
) -> None:
    coef_map = {
        "iv": ("iv_b", "iv_se", "IV"),
        "ols": ("ols_b", "ols_se", "OLS"),
        "rf": ("rf_b", "rf_se", "Reduced-form"),
        "first_stage": ("first_stage_b", "first_stage_se", "First-stage"),
    }
    coef_col, se_col, label = coef_map[coef_key]
    work = results.copy()
    work["lag"] = [
        _extract_lag(outcome, outcome_prefix) if isinstance(outcome, str) else None
        for outcome in work["outcome"]
    ]
    if work["lag"].isna().all():
        work["lag"] = np.arange(len(work))
    work = work.sort_values("lag")

    y = work[coef_col].astype(float)
    se = work[se_col].astype(float)
    ci = 1.96 * se

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.errorbar(work["lag"], y, yerr=ci, fmt="o", color="#2F5597", ecolor="#B7C4E0", capsize=3)
    ax.axhline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("Lag")
    ax.set_ylabel(f"{label} coefficient")
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)


def _pvalue_from_tstat(tstat: float) -> float:
    if not np.isfinite(tstat):
        return np.nan
    cdf = 0.5 * (1.0 + math.erf(abs(tstat) / math.sqrt(2.0)))
    return 2.0 * (1.0 - cdf)


def _extract_kp_rk_wald_f(iv_result: object) -> float:
    """
    Try to pull the Kleibergen-Paap rk Wald F statistic from a linearmodels IV result.
    Returns NaN if unavailable.
    """
    if iv_result is None:
        return np.nan
    candidate = None
    for attr in ("weak_instrument_test", "weak_instrument"):
        if hasattr(iv_result, attr):
            candidate = getattr(iv_result, attr)
            if callable(candidate):
                try:
                    candidate = candidate()
                except Exception:
                    candidate = None
            break
    if candidate is None:
        return np.nan
    for name in ("stat", "statistic", "stat_value", "value"):
        if hasattr(candidate, name):
            try:
                return float(getattr(candidate, name))
            except Exception:
                pass
    for key in ("stat", "statistic", "stat_value", "value"):
        try:
            return float(candidate[key])
        except Exception:
            continue
    return np.nan


def _kp_from_result_or_refit(iv_model: object, iv_result: object) -> float:
    """
    Return KP rk Wald F if available; otherwise refit with robust cov_type and try again.
    """
    kp = _extract_kp_rk_wald_f(iv_result)
    if np.isfinite(kp) or iv_model is None:
        return kp
    if not hasattr(iv_model, "fit"):
        return np.nan
    try:
        robust_res = iv_model.fit(cov_type="robust")
    except Exception:
        return np.nan
    return _extract_kp_rk_wald_f(robust_res)


def _stars_from_pvalue(pval: float) -> str:
    if not np.isfinite(pval):
        return ""
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def _format_estimate(b: float, se: float, digits: int = 3, add_stars: bool = True) -> str:
    if pd.isna(b) or pd.isna(se):
        return ""
    stars = ""
    if add_stars and se != 0:
        tstat = float(b) / float(se)
        stars = _stars_from_pvalue(_pvalue_from_tstat(tstat))
    return f"{b:.{digits}f}{stars} ({se:.{digits}f})"


def _render_latex_table(
    results: pd.DataFrame,
    outcome_prefix: str,
    caption: str,
    label: str,
    size_cmd: str,
    lag: int | None = None,
    x_label: str = "x",
    z_label: str = "z",
    add_stars: bool = True,
    add_notes: bool = True,
) -> str:
    work = results.copy()
    work["lag"] = [
        _extract_lag(outcome, outcome_prefix) if isinstance(outcome, str) else None
        for outcome in work["outcome"]
    ]
    if work["lag"].isna().all():
        work["lag"] = np.arange(len(work))
    if lag is None and work["lag"].nunique() > 1:
        raise ValueError("latex_lag must be set when multiple lags are present.")
    if lag is not None:
        work = work[work["lag"] == lag]
    work = work.sort_values("lag")

    lines: list[str] = []
    lines.append("\\begin{table}[!ht]")
    lines.append("\\centering")
    if size_cmd:
        lines.append(size_cmd)
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append(" & First-stage & OLS & RF & IV \\\\")
    lines.append("\\hline")
    row = work.iloc[0] if not work.empty else None
    if row is None:
        lines.append(f"{x_label} &  &  &  &  \\\\")
        lines.append(f"{z_label} &  &  &  &  \\\\")
    else:
        ols = _format_estimate(row["ols_b"], row["ols_se"], add_stars=add_stars)
        iv = _format_estimate(row["iv_b"], row["iv_se"], add_stars=add_stars)
        rf = _format_estimate(row["rf_b"], row["rf_se"], add_stars=add_stars)
        fs = _format_estimate(row["first_stage_b"], row["first_stage_se"], add_stars=add_stars)
        lines.append(f"{x_label} &  & {ols} &  & {iv} \\\\")
        lines.append(f"{z_label} & {fs} &  & {rf} &  \\\\")
        nobs = int(row["n_obs"]) if pd.notna(row["n_obs"]) else ""
        fs_r2 = "" if pd.isna(row["first_stage_r2"]) else f"{float(row['first_stage_r2']):.3f}"
        ols_r2 = "" if pd.isna(row["ols_r2"]) else f"{float(row['ols_r2']):.3f}"
        rf_r2 = "" if pd.isna(row["rf_r2"]) else f"{float(row['rf_r2']):.3f}"
        iv_r2 = "" if pd.isna(row["iv_r2"]) else f"{float(row['iv_r2']):.3f}"
        lines.append(f"Observations & {nobs} & {nobs} & {nobs} & {nobs} \\\\")
        lines.append(f"$R^2$ & {fs_r2} & {ols_r2} & {rf_r2} & {iv_r2} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    if add_notes:
        note_parts = ["Notes: Clustered standard errors at the company level in parentheses."]
        if add_stars:
            note_parts.append("* p<0.10, ** p<0.05, *** p<0.01.")
        note_text = " ".join(note_parts)
        lines.append("\\vspace{0.2em}")
        lines.append(f"\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize {note_text}\\end{{minipage}}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def _latex_table(
    results: pd.DataFrame,
    outcome_prefix: str,
    caption: str,
    label: str,
    size_cmd: str,
    out_path: Path,
    lag: int | None = None,
    x_label: str = "x",
    z_label: str = "z",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _render_latex_table(
            results,
            outcome_prefix=outcome_prefix,
            caption=caption,
            label=label,
            size_cmd=size_cmd,
            lag=lag,
            x_label=x_label,
            z_label=z_label,
        )
    )


def run_regressions(
    panel: pd.DataFrame,
    outcomes: Sequence[str],
    instr: str = "z_ct",
    dependent: str = "masters_opt_hires",
    include_fixed_effects: bool = True,
    controls: Sequence[str] | None = None,
) -> pd.DataFrame:
    controls = list(controls or [])
    if PanelOLS is None or IV2SLS is None:
        raise ImportError(
            "linearmodels is required for shift_share_regressions. "
            "Install it with: pip install linearmodels"
        )

    records: list[dict[str, object]] = []
    for outcome in outcomes:
        needed = ["c", "t", dependent, instr, outcome] + list(controls)
        work = panel[needed].replace([np.inf, -np.inf], np.nan).copy()
        work = work.dropna()
        if work.empty:
            continue
        work = work.set_index(["c", "t"])
        include_const = not include_fixed_effects
        exog = _build_exog(work, controls, include_const)

        ols = PanelOLS(
            dependent=work[outcome],
            exog=exog.join(work[[dependent]]),
            entity_effects=include_fixed_effects,
            time_effects=include_fixed_effects,
        ).fit(cov_type="clustered", cluster_entity=True)
        first_stage = PanelOLS(
            dependent=work[dependent],
            exog=exog.join(work[[instr]]),
            entity_effects=include_fixed_effects,
            time_effects=include_fixed_effects,
        ).fit(cov_type="clustered", cluster_entity=True)
        reduced = PanelOLS(
            dependent=work[outcome],
            exog=exog.join(work[[instr]]),
            entity_effects=include_fixed_effects,
            time_effects=include_fixed_effects,
        ).fit(cov_type="clustered", cluster_entity=True)

        iv_exog = exog if not exog.empty else None
        try:
            if PanelIV2SLS is not None:
                iv_model = PanelIV2SLS(
                    dependent=work[outcome],
                    exog=iv_exog,
                    endog=work[[dependent]],
                    instruments=work[[instr]],
                    entity_effects=include_fixed_effects,
                    time_effects=include_fixed_effects,
                )
                iv = iv_model.fit(cov_type="clustered", cluster_entity=True)
            else:
                clusters = work.reset_index()["c"]
                if include_fixed_effects:
                    # Fall back to exact within transformation when panel IV isn't available.
                    demean_cols = [outcome, dependent, instr] + list(controls)
                    work_iv = _demean_two_way(work.reset_index(), demean_cols).set_index(["c", "t"])
                    iv_exog = _build_exog(work_iv, controls, include_const=False)
                    iv_model = IV2SLS(
                        dependent=work_iv[outcome],
                        exog=iv_exog if not iv_exog.empty else None,
                        endog=work_iv[[dependent]],
                        instruments=work_iv[[instr]],
                    )
                    iv = iv_model.fit(cov_type="clustered", clusters=clusters)
                else:
                    iv_model = IV2SLS(
                        dependent=work[outcome],
                        exog=iv_exog,
                        endog=work[[dependent]],
                        instruments=work[[instr]],
                    )
                    iv = iv_model.fit(cov_type="clustered", clusters=clusters)
        except np.linalg.LinAlgError as exc:
            print(f"Warning: IV fit failed for '{outcome}': {exc}")
            continue

        records.append(
            {
                "outcome": outcome,
                "n_obs": int(work.shape[0]),
                "ols_b": ols.params[dependent],
                "ols_se": ols.std_errors[dependent],
                "ols_r2": getattr(ols, "rsquared", np.nan),
                "first_stage_b": first_stage.params[instr],
                "first_stage_se": first_stage.std_errors[instr],
                "first_stage_fstat": float(first_stage.tstats[instr] ** 2),
                "first_stage_r2": getattr(first_stage, "rsquared", np.nan),
                "kp_rk_wald_f": _kp_from_result_or_refit(iv_model, iv),
                "rf_b": reduced.params[instr],
                "rf_se": reduced.std_errors[instr],
                "rf_r2": getattr(reduced, "rsquared", np.nan),
                "iv_b": iv.params[dependent],
                "iv_se": iv.std_errors[dependent],
                "iv_r2": getattr(iv, "rsquared", np.nan),
            }
        )

    return pd.DataFrame.from_records(records)


def run_median_regressions(
    panel: pd.DataFrame,
    outcomes: Sequence[str],
    dependent: str = "masters_opt_hires",
    include_fixed_effects: bool = True,
    controls: Sequence[str] | None = None,
) -> pd.DataFrame:
    controls = list(controls or [])
    if include_fixed_effects:
        print("Median regression note: firm/year fixed effects are not applied in median mode.")

    records: list[dict[str, object]] = []
    for outcome in outcomes:
        needed = ["c", "t", dependent, outcome] + list(controls)
        work = panel[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if work.empty:
            continue
        exog = _build_exog(work, controls, include_const=True).join(work[[dependent]])
        try:
            med = sm.QuantReg(work[outcome], exog).fit(q=0.5)
        except Exception as exc:
            print(f"Warning: median regression failed for '{outcome}': {exc}")
            continue

        records.append(
            {
                "outcome": outcome,
                "n_obs": int(work.shape[0]),
                "ols_b": med.params.get(dependent, np.nan),
                "ols_se": med.bse.get(dependent, np.nan),
                "ols_r2": np.nan,
                "first_stage_b": np.nan,
                "first_stage_se": np.nan,
                "first_stage_fstat": np.nan,
                "first_stage_r2": np.nan,
                "kp_rk_wald_f": np.nan,
                "rf_b": np.nan,
                "rf_se": np.nan,
                "rf_r2": np.nan,
                "iv_b": np.nan,
                "iv_se": np.nan,
                "iv_r2": np.nan,
                "regression_type": "median",
            }
        )
    return pd.DataFrame.from_records(records)


def main(cli_args: Iterable[str] | None = None) -> pd.DataFrame:
    if cli_args is None and _in_ipython():
        cli_args = []
    args = _parse_args(cli_args)
    cfg = load_config(args.config)
    paths_cfg = get_cfg_section(cfg, "paths")
    reg_cfg = get_cfg_section(cfg, "shift_share_regressions")

    analysis_panel = args.analysis_panel or _resolve_cfg_path(paths_cfg, "analysis_panel")
    outcome_prefix = args.outcome_prefix or reg_cfg.get("outcome_prefix", "y_cst_lag")
    lag_start = args.lag_start if args.lag_start is not None else reg_cfg.get("lag_start", 0)
    lag_end = args.lag_end if args.lag_end is not None else reg_cfg.get("lag_end", 5)
    instrument = args.instrument or reg_cfg.get("instrument", "z_ct")
    dependent = args.dependent or reg_cfg.get("dependent", "masters_opt_hires")
    no_fe = args.no_fe if args.no_fe is not None else reg_cfg.get("no_fe", False)
    enforce_balanced_panel = (
        args.balanced_panel
        if args.balanced_panel is not None
        else reg_cfg.get("enforce_balanced_panel", True)
    )
    controls = args.controls if args.controls is not None else reg_cfg.get("controls", [])
    employees_lag = args.employees_lag if args.employees_lag is not None else reg_cfg.get("employees_lag", 0)
    diagnostics = args.diagnostics if args.diagnostics is not None else reg_cfg.get("diagnostics", False)
    diagnostics_top_n = args.diagnostics_top_n if args.diagnostics_top_n is not None else reg_cfg.get("diagnostics_top_n", 10)
    diagnostics_pre_lag_1 = args.diagnostics_pre_lag_1 if args.diagnostics_pre_lag_1 is not None else reg_cfg.get("diagnostics_pre_lag_1", 1)
    diagnostics_pre_lag_2 = args.diagnostics_pre_lag_2 if args.diagnostics_pre_lag_2 is not None else reg_cfg.get("diagnostics_pre_lag_2", 2)
    diagnostics_share_col = args.diagnostics_share_col or reg_cfg.get("diagnostics_share_col", "share_ck")
    diagnostics_component_col = args.diagnostics_component_col if args.diagnostics_component_col is not None else reg_cfg.get("diagnostics_component_col")
    require_nonzero_x_all_years = (
        args.require_nonzero_x_all_years
        if args.require_nonzero_x_all_years is not None
        else reg_cfg.get("require_nonzero_x_all_years", False)
    )
    median_regression = (
        args.median_regression
        if args.median_regression is not None
        else reg_cfg.get("median_regression", False)
    )
    fill_missing_xy_zero = (
        args.fill_missing_xy_zero
        if args.fill_missing_xy_zero is not None
        else reg_cfg.get("fill_missing_xy_zero", False)
    )
    ihs_y = args.ihs_y if args.ihs_y is not None else reg_cfg.get("ihs_y", False)

    if lag_end < lag_start:
        raise ValueError("lag_end must be >= lag_start.")

    panel_path = analysis_panel
    if not panel_path.is_absolute():
        panel_path = Path.cwd() / panel_path
    panel = pd.read_parquet(panel_path)
    employer_crosswalk_path = _resolve_cfg_path(paths_cfg, "employer_crosswalk")
    if not employer_crosswalk_path.is_absolute():
        employer_crosswalk_path = Path.cwd() / employer_crosswalk_path
    if employer_crosswalk_path.exists():
        panel = join_company_names(panel, employer_crosswalk_path=employer_crosswalk_path)
    else:
        print(f"Name-join note: employer_crosswalk not found at {employer_crosswalk_path}; skipping firm_name join.")

    outcomes = _outcome_columns(outcome_prefix, lag_start, lag_end)
    panel = _filter_panel(panel, args.start_t or reg_cfg.get("start_t"), args.end_t or reg_cfg.get("end_t"))
    if args.min_employees_2010 is not None:
        panel = _filter_by_employees_2010(panel, args.min_employees_2010, employees_lag)
    elif reg_cfg.get("min_employees_2010") is not None:
        panel = _filter_by_employees_2010(panel, reg_cfg.get("min_employees_2010"), employees_lag)
    if fill_missing_xy_zero:
        panel = _fill_missing_xy_with_zero(panel, dependent, outcomes)
    if ihs_y:
        panel = _transform_y_ihs(panel, outcomes)
    panel = _prepare_panel_for_instrument(
        panel,
        instrument=instrument,
        enforce_balanced_panel=enforce_balanced_panel,
    )
    if require_nonzero_x_all_years:
        firms_before = panel["c"].nunique()
        panel = _filter_firms_nonzero_in_all_years(panel, dependent)
        firms_after = panel["c"].nunique()
        print(
            f"Applied nonzero-x-all-years filter ({dependent} > 0): "
            f"firms {firms_before:,} -> {firms_after:,}, rows={len(panel):,}."
        )
    controls = _parse_controls(controls) if isinstance(controls, str) else list(controls)

    if median_regression:
        results = run_median_regressions(
            panel,
            outcomes,
            dependent=dependent,
            include_fixed_effects=not no_fe,
            controls=controls,
        )
    else:
        results = run_regressions(
            panel,
            outcomes,
            instr=instrument,
            dependent=dependent,
            include_fixed_effects=not no_fe,
            controls=controls,
        )

    if results.empty:
        print("No regression results produced (check missing data or outcome columns).")
        return results

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(results.to_string(index=False))
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Wrote regression summary to {output_path}")

    if args.plot_out is not None:
        plot_path = args.plot_out
        if str(plot_path) == "-":
            _plot_results(
                results,
                outcome_prefix=outcome_prefix,
                coef_key=args.plot_coef,
                out_path=None,
                title=args.plot_title,
                show=True,
            )
        else:
            if not plot_path.is_absolute():
                plot_path = Path.cwd() / plot_path
            _plot_results(
                results,
                outcome_prefix=outcome_prefix,
                coef_key=args.plot_coef,
                out_path=plot_path,
                title=args.plot_title,
            )
            print(f"Wrote plot to {plot_path}")

    if args.latex_out is not None:
        latex_path = args.latex_out
        x_label = args.latex_x_label or args.dependent
        z_label = args.latex_z_label or args.instrument
        if str(latex_path) == "-":
            print(
                _render_latex_table(
                    results,
                    outcome_prefix=outcome_prefix,
                    caption=args.latex_caption,
                    label=args.latex_label,
                    size_cmd=args.latex_size,
                    lag=args.latex_lag,
                    x_label=x_label,
                    z_label=z_label,
                )
            )
        else:
            if not latex_path.is_absolute():
                latex_path = Path.cwd() / latex_path
            _latex_table(
                results,
                outcome_prefix=outcome_prefix,
                caption=args.latex_caption,
                label=args.latex_label,
                size_cmd=args.latex_size,
                out_path=latex_path,
                lag=args.latex_lag,
                x_label=x_label,
                z_label=z_label,
            )
            print(f"Wrote LaTeX table to {latex_path}")

    if diagnostics:
        components_path = args.instrument_components or _resolve_cfg_path(paths_cfg, "instrument_components")
        if not components_path.is_absolute():
            components_path = Path.cwd() / components_path
        if components_path.exists():
            instrument_components = pd.read_parquet(components_path)
        else:
            print(f"Diagnostics note: instrument_components not found at {components_path}.")
            instrument_components = None
        shift_share_diagnostics(
            panel,
            instrument_components=instrument_components,
            instrument=instrument,
            dependent=dependent,
            outcome_prefix=outcome_prefix,
            pre_lag_1=diagnostics_pre_lag_1,
            pre_lag_2=diagnostics_pre_lag_2,
            include_fixed_effects=not no_fe,
            controls=controls,
            component_col=diagnostics_component_col,
            share_col=diagnostics_share_col,
            top_n=diagnostics_top_n,
            print_summary=True,
        )

    return results


def run_from_panel_path(
    panel_path: Path | str | None = None,
    outcomes: Sequence[str] | None = None,
    outcome_prefix: str = "y_cst_lag",
    lag_start: int = -2,
    lag_end: int = 5,
    instr: str = "z_ct",
    dependent: str = "valid_masters_opt_hires",
    include_fixed_effects: bool = True,
    controls: Sequence[str] | None = None,
    start_t: 2012 | None = None,
    end_t: 2018 | None = None,
    enforce_balanced_panel: bool = True,
    min_employees_2010: int | None = None,
    employees_lag: int = 0,
    plot_out: Path | str | bool | None = None,
    plot_coef: str = "iv",
    plot_title: str | None = None,
    latex_out: Path | str | bool | None = None,
    latex_caption: str = "Shift-share regression results",
    latex_label: str = "tab:shift_share",
    latex_size: str = "\\scriptsize",
    latex_lag: int | None = None,
    latex_x_label: str | None = None,
    latex_z_label: str | None = None,
    run_diagnostics: bool = False,
    instrument_components: pd.DataFrame | None = None,
    instrument_components_path: Path | str | None = None,
    diagnostics_top_n: int = 10,
    diagnostics_pre_lag_1: int = -1,
    diagnostics_pre_lag_2: int = -2,
    diagnostics_component_col: str | None = None,
    diagnostics_share_col: str = "share_ck",
    diagnostics_print: bool = True,
    require_nonzero_x_all_years: bool = False,
    fill_missing_xy_zero: bool = False,
    median_regression: bool = False,
    ihs_y: bool = False,
    config_path: Path | str | None = None,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    paths_cfg = get_cfg_section(cfg, "paths")
    if panel_path is None:
        panel_path = _resolve_cfg_path(paths_cfg, "analysis_panel")
    panel_path = Path(panel_path)
    if not panel_path.is_absolute():
        panel_path = Path.cwd() / panel_path
    panel = pd.read_parquet(panel_path)
    employer_crosswalk_path_cfg = _resolve_cfg_path(paths_cfg, "employer_crosswalk")
    if not employer_crosswalk_path_cfg.is_absolute():
        employer_crosswalk_path_cfg = Path.cwd() / employer_crosswalk_path_cfg
    if employer_crosswalk_path_cfg.exists():
        panel = join_company_names(panel, employer_crosswalk_path=employer_crosswalk_path_cfg)
    else:
        print(
            f"Name-join note: employer_crosswalk not found at {employer_crosswalk_path_cfg}; "
            "skipping firm_name join."
        )
    panel = _filter_panel(panel, start_t, end_t)
    if min_employees_2010 is not None:
        panel = _filter_by_employees_2010(panel, min_employees_2010, employees_lag)
    outcomes = list(outcomes or _outcome_columns(outcome_prefix, lag_start, lag_end))
    if fill_missing_xy_zero:
        panel = _fill_missing_xy_with_zero(panel, dependent, outcomes)
    if ihs_y:
        panel = _transform_y_ihs(panel, outcomes)
    panel = _prepare_panel_for_instrument(
        panel,
        instrument=instr,
        enforce_balanced_panel=enforce_balanced_panel,
    )
    if require_nonzero_x_all_years:
        panel = _filter_firms_nonzero_in_all_years(panel, dependent)
    if median_regression:
        results = run_median_regressions(
            panel,
            outcomes,
            dependent=dependent,
            include_fixed_effects=include_fixed_effects,
            controls=controls,
        )
    else:
        results = run_regressions(
            panel,
            outcomes,
            instr=instr,
            dependent=dependent,
            include_fixed_effects=include_fixed_effects,
            controls=controls,
        )
    if plot_out is not None:
        if plot_out is True or str(plot_out) == "-":
            _plot_results(
                results,
                outcome_prefix=outcome_prefix,
                coef_key=plot_coef,
                out_path=None,
                title=plot_title,
                show=True,
            )
        else:
            plot_path = Path(plot_out)
            if not plot_path.is_absolute():
                plot_path = Path.cwd() / plot_path
            _plot_results(
                results,
                outcome_prefix=outcome_prefix,
                coef_key=plot_coef,
                out_path=plot_path,
                title=plot_title,
            )
    if latex_out is not None:
        x_label = latex_x_label or dependent
        z_label = latex_z_label or instr
        if latex_out is True or str(latex_out) == "-":
            print(
                _render_latex_table(
                    results,
                    outcome_prefix=outcome_prefix,
                    caption=latex_caption,
                    label=latex_label,
                    size_cmd=latex_size,
                    lag=latex_lag,
                    x_label=x_label,
                    z_label=z_label,
                )
            )
        else:
            latex_path = Path(latex_out)
            if not latex_path.is_absolute():
                latex_path = Path.cwd() / latex_path
            _latex_table(
                results,
                outcome_prefix=outcome_prefix,
                caption=latex_caption,
                label=latex_label,
                size_cmd=latex_size,
                out_path=latex_path,
                lag=latex_lag,
                x_label=x_label,
                z_label=z_label,
            )
    if run_diagnostics:
        if instrument_components is None:
            if instrument_components_path is None:
                instrument_components_path = _resolve_cfg_path(paths_cfg, "instrument_components")
            comp_path = Path(instrument_components_path)
            if not comp_path.is_absolute():
                comp_path = Path.cwd() / comp_path
            if comp_path.exists():
                instrument_components = pd.read_parquet(comp_path)
            else:
                print(f"Diagnostics note: instrument_components not found at {comp_path}.")
                instrument_components = None
        shift_share_diagnostics(
            panel,
            instrument_components=instrument_components,
            instrument=instr,
            dependent=dependent,
            outcome_prefix=outcome_prefix,
            pre_lag_1=diagnostics_pre_lag_1,
            pre_lag_2=diagnostics_pre_lag_2,
            include_fixed_effects=include_fixed_effects,
            controls=controls,
            component_col=diagnostics_component_col,
            share_col=diagnostics_share_col,
            top_n=diagnostics_top_n,
            print_summary=diagnostics_print,
        )
    return results


# if __name__ == "__main__":
#     main()
start_t = 2012
end_t = 2018
min_employees_2010 = 0 #15
employees_lag = 0
enforce_balanced_panel = False
outcome_prefix = "y_cst_lag"
lag_start = -2
lag_end = 5
instr = "z_ct_full"
dependent = "valid_masters_opt_hires"
include_fixed_effects = True
controls = []
plot_out = True
plot_coef = "iv"
plot_title = None
latex_out = None
latex_caption = "Shift-share regression results"
latex_label = "tab:shift_share"
latex_size = "\\scriptsize"
latex_lag = 0
latex_x_label = None
latex_z_label = None
run_diagnostics = True
instrument_components = None
instrument_components_path = None
diagnostics_top_n: int = 10
diagnostics_pre_lag_1: int = -1
diagnostics_pre_lag_2: int = -2
diagnostics_component_col: str | None = None
diagnostics_share_col: str = "share_ck"
diagnostics_print: bool = True
require_nonzero_x_all_years = False
median_regression = True
ihs_y  = False

cfg = load_config(DEFAULT_CONFIG_PATH)
paths_cfg = get_cfg_section(cfg, "paths")
panel_path = _resolve_cfg_path(paths_cfg, "analysis_panel")
panel_path = Path(panel_path)
if not panel_path.is_absolute():
    panel_path = Path.cwd() / panel_path
panel = pd.read_parquet(panel_path)
panel = _filter_panel(panel, start_t, end_t)
if min_employees_2010 is not None:
    panel = _filter_by_employees_2010(panel, min_employees_2010, employees_lag)
panel = _prepare_panel_for_instrument(
    panel,
    instrument=instr,
    enforce_balanced_panel=enforce_balanced_panel,
)
if require_nonzero_x_all_years:
    firms_before = panel["c"].nunique()
    panel = _filter_firms_nonzero_in_all_years(panel, dependent)
    firms_after = panel["c"].nunique()
    print(
        f"Applied nonzero-x-all-years filter ({dependent} > 0): "
        f"firms {firms_before:,} -> {firms_after:,}, rows={len(panel):,}."
    )
outcomes = list(_outcome_columns(outcome_prefix, lag_start, lag_end))

if ihs_y:
        panel = _transform_y_ihs(panel, outcomes)

if median_regression:
    results = run_median_regressions(
        panel,
        outcomes,
        dependent=dependent,
        include_fixed_effects=True,
        controls=controls,
    )
else:
    results = run_regressions(
        panel,
        outcomes,
        instr=instr,
        dependent=dependent,
        include_fixed_effects=include_fixed_effects,
        controls=controls,
    )
if plot_out is not None:
    if plot_out is True or str(plot_out) == "-":
        _plot_results(
            results,
            outcome_prefix=outcome_prefix,
            coef_key=plot_coef,
            out_path=None,
            title=plot_title,
            show=True,
        )
    else:
        plot_path = Path(plot_out)
        if not plot_path.is_absolute():
            plot_path = Path.cwd() / plot_path
        _plot_results(
            results,
            outcome_prefix=outcome_prefix,
            coef_key=plot_coef,
            out_path=plot_path,
            title=plot_title,
        )
if latex_out is not None:
    x_label = latex_x_label or dependent
    z_label = latex_z_label or instr
    if latex_out is True or str(latex_out) == "-":
        print(
            _render_latex_table(
                results,
                outcome_prefix=outcome_prefix,
                caption=latex_caption,
                label=latex_label,
                size_cmd=latex_size,
                lag=latex_lag,
                x_label=x_label,
                z_label=z_label,
            )
        )
    else:
        latex_path = Path(latex_out)
        if not latex_path.is_absolute():
            latex_path = Path.cwd() / latex_path
        _latex_table(
            results,
            outcome_prefix=outcome_prefix,
            caption=latex_caption,
            label=latex_label,
            size_cmd=latex_size,
            out_path=latex_path,
            lag=latex_lag,
            x_label=x_label,
            z_label=z_label,
        )
if run_diagnostics:
    if instrument_components is None:
        if instrument_components_path is None:
            instrument_components_path = _resolve_cfg_path(paths_cfg, "instrument_components")
        comp_path = Path(instrument_components_path)
        if not comp_path.is_absolute():
            comp_path = Path.cwd() / comp_path
        if comp_path.exists():
            instrument_components = pd.read_parquet(comp_path)
        else:
            print(f"Diagnostics note: instrument_components not found at {comp_path}.")
            instrument_components = None
    shift_share_diagnostics(
        panel,
        instrument_components=instrument_components,
        instrument=instr,
        dependent=dependent,
        outcome_prefix=outcome_prefix,
        pre_lag_1=diagnostics_pre_lag_1,
        pre_lag_2=diagnostics_pre_lag_2,
        include_fixed_effects=include_fixed_effects,
        controls=controls,
        component_col=diagnostics_component_col,
        share_col=diagnostics_share_col,
        top_n=diagnostics_top_n,
        print_summary=diagnostics_print,
    )

# diag = firm_sample_diagnostics_plots(panel, 
#                               employer_crosswalk_path=_resolve_cfg_path(paths_cfg, "employer_crosswalk"),
#                               size_col="y_cst_lag0",
#                               year_a = 2012,
#                               year_b = 2018,
#                               out_dir = None)
