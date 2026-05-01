"""Build slide-ready shift-share assets for slides_laborlunch_20260507.

This wrapper is intentionally downstream-only: it reads saved outputs from the
shift-share pipeline and writes figures/tables for the labor lunch deck without
rebuilding the firm panel or rerunning the full regression suite.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import laborlunch_plot_style as llstyle


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/yk0581/data/out/company_shift_share")
APR_ROOT = DATA_ROOT / "apr2026"
EXPOSURE_ROOT = Path("/home/yk0581/data/out/company_shift_share_apr2026")
DEFAULT_OUT_DIR = CODE_ROOT / "output" / "company_shift_share" / "slides_20260507_shift_share"
DEFAULT_DECK = Path("/home/yk0581/writing/slides/slides_laborlunch_20260507/slides_laborlunch_20260507.tex")
ERRORBAR_INTERVAL_ALPHA = llstyle.ERRORBAR_ALPHA


OUTCOME_LABELS = {
    "y_cst_lag0": "Employment",
    "log1p_y_cst_lag0": "Log employment",
    "y_new_hires_lag0": "New hires",
    "log1p_y_new_hires_lag0": "Log new hires",
    "y_new_hires_foreign_lag0": "New hires, foreign",
    "log1p_y_new_hires_foreign_lag0": "Log new hires, foreign",
    "y_new_hires_native_lag0": "New hires, native",
    "log1p_y_new_hires_native_lag0": "Log new hires, native",
    "avg_tenure_years_lag0": "Avg tenure",
    "avg_tenure_years_annual": "Avg tenure",
    "total_comp_mean_annual": "Avg compensation",
}

VARIANT_LABELS = {
    "z_ct": "Raw annual flow",
    "z_ct_event_pulse": "Event pulse",
    "z_ct_flow_diff": "First-difference innovation",
    "z_ct_flow_ar_resid": "AR-residual innovation",
    "z_ct_flow_diff_cumulative": "Cumulative first-diff pool",
    "z_ct_flow_ar_resid_cumulative": "Cumulative AR-residual pool",
    "z_ct_common_base_level": "Common-base level",
    "z_ct_common_base_asinh": "Common-base asinh",
    "z_ct_event_step_dose": "Event-step dose",
    "z_ct_share_2008_2010": "Shares 2008-2010",
    "z_ct_share_2011_2013": "Shares 2011-2013",
    "z_ct_full": "Full-sample shares",
    "z_ct_falsification_lead4_broad": "Lead-4 placebo",
}


def _set_plot_style() -> None:
    llstyle.apply_style()


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fmt_num(value: object, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(val):
        return ""
    if abs(val) >= 1000:
        return f"{val:,.0f}"
    if abs(val) < 0.001 and val != 0:
        return f"{val:.2e}"
    return f"{val:.{digits}f}"


def _coef_se_cell(coef: object, se: object, digits: int = 4) -> str:
    coef_text = _fmt_num(coef, digits)
    se_text = _fmt_num(se, digits)
    if not coef_text:
        return ""
    if not se_text:
        return coef_text
    return rf"{coef_text} ({se_text})"


def _latex_escape(text: object) -> str:
    out = str(text)
    for old, new in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }.items():
        out = out.replace(old, new)
    return out


def _savefig(fig: plt.Figure, path: Path) -> None:
    llstyle.savefig(fig, path)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _iterative_residualize(
    df: pd.DataFrame,
    value_col: str,
    fe_cols: list[str],
    max_iter: int = 40,
    tol: float = 1e-9,
) -> pd.Series:
    """Residualize a column against several high-dimensional FEs by alternating projections."""
    resid = _safe_num(df[value_col]).astype(float)
    resid = resid - resid.mean()
    valid_fe_cols = [col for col in fe_cols if col in df.columns]
    if not valid_fe_cols:
        return resid
    for _ in range(max_iter):
        old = resid.copy()
        for fe_col in valid_fe_cols:
            means = resid.groupby(df[fe_col], sort=False, dropna=False).transform("mean")
            resid = resid - means
        if float((resid - old).abs().max()) < tol:
            break
    return resid


def _prepare_preferred_panel_for_binscatter() -> pd.DataFrame:
    panel = _read_table(APR_ROOT / "analysis_panel.parquet")
    required = {"c", "t", "z_ct", "masters_opt_hires_correction_aware", "y_cst_lag0", "y_cst_lag1"}
    if panel.empty or not required.issubset(panel.columns):
        return pd.DataFrame()
    optional = {
        "z_ct_v4_matched_step",
        "z_ct_v5_matched_pulse",
        "y_new_hires_lag0",
        "y_new_hires_foreign_lag0",
        "y_new_hires_native_lag0",
        "avg_tenure_years_lag0",
    }
    keep_cols = sorted(required | (optional & set(panel.columns)))
    work = panel.loc[:, keep_cols].copy()
    work["c"] = work["c"].astype(str)
    work["t_num"] = _safe_num(work["t"])
    work = work.loc[work["t_num"].between(2008, 2022)].copy()
    work = work.loc[work["z_ct"].notna()].copy()
    from company_shift_share.shift_share_analysis import _prepare_first_stage_state_panel

    work = _prepare_first_stage_state_panel(
        work,
        baseline_window_start=2008,
        baseline_window_end=2013,
        current_size_bins=10,
        current_growth_bins=5,
        joint_size_growth_bins=3,
        baseline_growth_bins=5,
        use_log_y_panel=False,
    )
    work = work.loc[work["t_num"].between(2013, 2022)].copy()
    n_years = work["t_num"].nunique()
    balanced = work.groupby("c")["t_num"].nunique()
    balanced_firms = balanced.loc[balanced.eq(n_years)].index
    work = work.loc[work["c"].isin(balanced_firms)].copy()
    work["t"] = work["t_num"].astype("Int64").astype(str)
    work = work.loc[work["baseline_size_growth_year_fe"].notna()].copy()
    work["z_ct"] = _safe_num(work["z_ct"]).fillna(0.0)
    for z_col in ["z_ct", "z_ct_v4_matched_step", "z_ct_v5_matched_pulse"]:
        if z_col not in work.columns:
            continue
        work[z_col] = _safe_num(work[z_col]).fillna(0.0)
        work[f"log1p_{z_col}"] = np.log1p(work[z_col].where(work[z_col] >= -1))
    work["x_ct"] = _safe_num(work["masters_opt_hires_correction_aware"]).fillna(0.0)
    work["log1p_y_cst_lag0"] = np.log1p(_safe_num(work["y_cst_lag0"]).clip(lower=0))
    for outcome in [
        "y_new_hires_lag0",
        "y_new_hires_foreign_lag0",
        "y_new_hires_native_lag0",
    ]:
        if outcome in work.columns:
            work[f"log1p_{outcome}"] = np.log1p(_safe_num(work[outcome]).clip(lower=0))
    if "avg_tenure_years_lag0" in work.columns:
        work["avg_tenure_years_lag0"] = _safe_num(work["avg_tenure_years_lag0"])
    return work


def _plot_residualized_binscatter(
    panel: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_png: Path,
    out_csv: Path,
    n_bins: int = 50,
) -> None:
    if panel.empty or x_col not in panel.columns or y_col not in panel.columns:
        return
    work = panel.loc[:, ["c", "t", "baseline_size_growth_year_fe", x_col, y_col]].dropna().copy()
    if work.empty or work[x_col].nunique() < 2:
        return
    fe_cols = ["c", "t", "baseline_size_growth_year_fe"]
    work["x_resid"] = _iterative_residualize(work, x_col, fe_cols)
    work["y_resid"] = _iterative_residualize(work, y_col, fe_cols)
    work = work.loc[work["x_resid"].notna() & work["y_resid"].notna()].copy()
    if work.empty or work["x_resid"].nunique() < 2:
        return
    ranks = work["x_resid"].rank(method="first")
    work["bin"] = pd.qcut(ranks, q=min(n_bins, work["x_resid"].nunique()), labels=False, duplicates="drop")
    bins = (
        work.groupby("bin", as_index=False)
        .agg(
            x_resid=("x_resid", "mean"),
            y_resid=("y_resid", "mean"),
            n=("x_resid", "size"),
        )
        .sort_values("x_resid")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    bins.to_csv(out_csv, index=False)
    slope, intercept = np.polyfit(work["x_resid"], work["y_resid"], 1)
    x_line = np.linspace(float(bins["x_resid"].min()), float(bins["x_resid"].max()), 100)
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    ax.scatter(bins["x_resid"], bins["y_resid"], s=llstyle.marker_area(llstyle.MULTI_MARKER_SIZE), alpha=0.85, color=llstyle.color(2))
    ax.plot(x_line, intercept + slope * x_line, color=llstyle.color(1), linewidth=1.8)
    ax.axhline(0, color="0.45", linestyle=":", linewidth=1.1)
    ax.axvline(0, color="0.45", linestyle=":", linewidth=1.1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("")
    ax.text(
        0.03,
        0.95,
        f"FWL slope: {slope:.4g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    _savefig(fig, out_png)


def _soften_errorbar_interval(errorbar_container, alpha: float = ERRORBAR_INTERVAL_ALPHA) -> None:
    """Apply alpha to errorbar intervals/caps without fading the coefficient line."""
    if errorbar_container is None or not hasattr(errorbar_container, "lines"):
        return
    for artist_group in errorbar_container.lines[1:]:
        if artist_group is None:
            continue
        artists = artist_group if isinstance(artist_group, (tuple, list)) else [artist_group]
        for artist in artists:
            if hasattr(artist, "set_alpha"):
                artist.set_alpha(alpha)


def _event_year_from_row(row: pd.Series) -> Optional[int]:
    best_year: Optional[int] = None
    best_change = -np.inf
    for col in row.index:
        match = re.match(r"metric_share_change_(\d{4})_(\d{4})$", str(col))
        if not match:
            continue
        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if pd.isna(val):
            continue
        if float(val) > best_change:
            best_change = float(val)
            best_year = int(match.group(2))
    return best_year


def _school_event_map(sample: pd.DataFrame) -> pd.DataFrame:
    if sample.empty:
        return pd.DataFrame(columns=["k", "sample_role", "matched_pair_id", "event_year"])
    work = sample.copy()
    work["k"] = work["k"].astype(str)
    work["sample_role"] = work.get("sample_role", "").fillna("").astype(str)
    work = work.loc[work["sample_role"].isin(["treated", "control"])].copy()
    if work.empty:
        return pd.DataFrame(columns=["k", "sample_role", "matched_pair_id", "event_year"])
    work["event_year_raw"] = work.apply(_event_year_from_row, axis=1)
    treated = work.loc[work["sample_role"].eq("treated"), ["k", "event_year_raw"]].dropna()
    treated_map = dict(zip(treated["k"], treated["event_year_raw"].astype(int)))
    if "matched_school_k" in work.columns:
        work["matched_school_k"] = work["matched_school_k"].astype(str)
        work["event_year"] = np.where(
            work["sample_role"].eq("control"),
            work["matched_school_k"].map(treated_map),
            work["event_year_raw"],
        )
    else:
        work["event_year"] = work["event_year_raw"]
    work["event_year"] = _safe_num(work["event_year"]).astype("Int64")
    return work.loc[work["event_year"].notna(), ["k", "sample_role", "matched_pair_id", "event_year"]].copy()


def _plot_event_line(
    stats: pd.DataFrame,
    y_col: str,
    yerr_col: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    color_map = {"treated": llstyle.color(2), "control": llstyle.color(1)}
    label_map = {"treated": "Treated schools", "control": "Matched controls"}
    roles = [role for role in ["control", "treated"] if not stats.loc[stats["sample_role"].eq(role)].empty]
    role_offsets = {role: float(offset) for role, offset in zip(roles, llstyle.offsets(len(roles)))}
    for role in roles:
        sub = stats.loc[stats["sample_role"].eq(role)].sort_values("event_time")
        if sub.empty:
            continue
        x_vals = sub["event_time"].astype(float) + role_offsets.get(role, 0.0)
        errorbar_container = ax.errorbar(
            x_vals,
            sub[y_col],
            yerr=1.96 * sub[yerr_col].fillna(0),
            fmt="o-",
            color=color_map[role],
            ecolor=llstyle.rgba(color_map[role]),
            elinewidth=llstyle.MARKER_SIZE,
            capsize=0,
            markersize=llstyle.MARKER_SIZE,
            label=label_map[role],
        )
        _soften_errorbar_interval(errorbar_container)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.4)
    ax.axhline(0, color="0.35", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Years relative to school treatment year")
    ax.set_ylabel(ylabel)
    ax.set_title("")
    llstyle.right_legend(ax)
    _savefig(fig, path)


def build_school_event_figures(out_dir: Path) -> None:
    sample = _read_table(DATA_ROOT / "school_shift_sample.parquet")
    panel = _read_table(DATA_ROOT / "school_shift_metric_panel.parquet")
    if sample.empty or panel.empty:
        return
    events = _school_event_map(sample)
    if events.empty:
        return
    keep = panel.merge(events, on="k", how="inner", suffixes=("", "_event"))
    if "sample_role_event" in keep.columns:
        keep["sample_role"] = keep["sample_role_event"]
    keep["t"] = _safe_num(keep["t"]).astype("Int64")
    keep["event_year"] = _safe_num(keep["event_year"]).astype("Int64")
    keep = keep.loc[keep["t"].notna() & keep["event_year"].notna()].copy()
    keep["event_time"] = keep["t"].astype(int) - keep["event_year"].astype(int)
    keep = keep.loc[keep["event_time"].between(-5, 5)].copy()
    if keep.empty:
        return
    keep["metric_level"] = _safe_num(keep["metric_level"])
    keep["metric_share"] = _safe_num(keep["metric_share"])
    base = (
        keep.loc[keep["event_time"].between(-3, -1)]
        .groupby("k", as_index=False)
        .agg(base_level=("metric_level", "mean"), base_share=("metric_share", "mean"))
    )
    keep = keep.merge(base, on="k", how="left")
    keep["gkt_minus_pre"] = keep["metric_level"] - keep["base_level"]
    keep["ihmp_share_minus_pre_pp"] = 100.0 * (keep["metric_share"] - keep["base_share"])
    stats = (
        keep.groupby(["sample_role", "event_time"], as_index=False)
        .agg(
            mean_gkt=("gkt_minus_pre", "mean"),
            se_gkt=("gkt_minus_pre", lambda s: s.std(ddof=1) / np.sqrt(max(s.notna().sum(), 1))),
            mean_share=("ihmp_share_minus_pre_pp", "mean"),
            se_share=("ihmp_share_minus_pre_pp", lambda s: s.std(ddof=1) / np.sqrt(max(s.notna().sum(), 1))),
            n_schools=("k", "nunique"),
        )
    )
    stats.to_csv(out_dir / "school_event_time_stats.csv", index=False)
    _plot_event_line(
        stats,
        "mean_gkt",
        "se_gkt",
        "IHMP seats minus pre-event mean",
        "Zeroth stage: school shift around treatment year",
        out_dir / "zeroth_stage_gkt_event_time.png",
    )
    _plot_event_line(
        stats,
        "mean_share",
        "se_share",
        "IHMP share minus pre-event mean (pp)",
        "IHMP share around treatment year",
        out_dir / "ihmp_share_event_time.png",
    )


def build_zct_selected_firm_figure(out_dir: Path, top_n: int = 35) -> None:
    sample = _read_table(DATA_ROOT / "school_shift_sample.parquet")
    comp = _read_table(DATA_ROOT / "instrument_components.parquet")
    panel = _read_table(DATA_ROOT / "instrument_panel.parquet")
    if sample.empty or comp.empty or panel.empty:
        return
    events = _school_event_map(sample)
    if events.empty:
        return
    events = events.loc[events["sample_role"].eq("treated"), ["k", "event_year"]].drop_duplicates()
    work = comp.merge(events, on="k", how="inner")
    if work.empty:
        return
    work["exposure_mass"] = _safe_num(work.get("share_ck", 0)).fillna(0) * _safe_num(work.get("g_kt", 0)).fillna(0).abs()
    top = (
        work.groupby(["event_year", "c"], as_index=False)["exposure_mass"].sum()
        .sort_values(["event_year", "exposure_mass"], ascending=[True, False])
        .groupby("event_year", as_index=False)
        .head(top_n)
    )
    firm_panel = panel.merge(top[["event_year", "c"]].drop_duplicates(), on="c", how="inner")
    firm_panel["t"] = _safe_num(firm_panel["t"]).astype(int)
    firm_panel["z_ct"] = _safe_num(firm_panel["z_ct"])
    stats = (
        firm_panel.groupby(["event_year", "t"], as_index=False)
        .agg(
            mean_z=("z_ct", "mean"),
            se_z=("z_ct", lambda s: s.std(ddof=1) / np.sqrt(max(s.notna().sum(), 1))),
            n_firms=("c", "nunique"),
        )
    )
    stats.to_csv(out_dir / "zct_selected_firms_by_event_year.csv", index=False)
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    event_years = sorted(stats["event_year"].dropna().unique())
    event_offsets = {year: float(offset) for year, offset in zip(event_years, llstyle.offsets(len(event_years), span=0.50))}
    for i, event_year in enumerate(event_years):
        sub = stats.loc[stats["event_year"].eq(event_year)].sort_values("t")
        color = llstyle.color(i)
        errorbar_container = ax.errorbar(
            sub["t"].astype(float) + event_offsets.get(event_year, 0.0),
            sub["mean_z"],
            yerr=1.96 * sub["se_z"].fillna(0),
            fmt="o-",
            color=color,
            ecolor=llstyle.rgba(color),
            elinewidth=llstyle.MULTI_MARKER_SIZE,
            capsize=0,
            markersize=llstyle.MULTI_MARKER_SIZE,
            label=f"Event {int(event_year)}",
        )
        _soften_errorbar_interval(errorbar_container)
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Mean firm shift-share exposure, z_ct")
    ax.set_title("")
    llstyle.right_legend(ax)
    _savefig(fig, out_dir / "zct_selected_firms_by_event_year.png")


def build_raw_first_stage_figure(out_dir: Path) -> None:
    panel = _read_table(DATA_ROOT / "analysis_panel.parquet")
    if panel.empty or not {"c", "t", "z_ct", "masters_opt_hires_correction_aware"}.issubset(panel.columns):
        return
    work = panel.loc[:, ["c", "t", "z_ct", "masters_opt_hires_correction_aware"]].copy()
    work["t"] = _safe_num(work["t"])
    work["z_ct"] = _safe_num(work["z_ct"]).fillna(0.0)
    work["x"] = _safe_num(work["masters_opt_hires_correction_aware"]).fillna(0.0)
    work = work.loc[work["t"].between(2008, 2022)].copy()
    exposure = (
        work.loc[work["t"].between(2014, 2017)]
        .groupby("c", as_index=False)["z_ct"]
        .sum()
        .rename(columns={"z_ct": "z_total"})
    )
    pos = exposure.loc[exposure["z_total"].gt(0), "z_total"]
    if pos.nunique() < 2:
        return
    q25, q75 = pos.quantile([0.25, 0.75])
    exposure["exposure_bin"] = np.where(
        exposure["z_total"].ge(q75),
        "High exposure",
        np.where(exposure["z_total"].le(q25), "Low/zero exposure", "Middle"),
    )
    work = work.merge(exposure[["c", "exposure_bin"]], on="c", how="left")
    work["exposure_bin"] = work["exposure_bin"].fillna("Low/zero exposure")
    work = work.loc[work["exposure_bin"].isin(["Low/zero exposure", "High exposure"])].copy()
    base = (
        work.loc[work["t"].between(2011, 2013)]
        .groupby("c", as_index=False)["x"]
        .mean()
        .rename(columns={"x": "x_pre"})
    )
    work = work.merge(base, on="c", how="left")
    work["x_minus_pre"] = work["x"] - work["x_pre"]
    stats = (
        work.groupby(["t", "exposure_bin"], as_index=False)
        .agg(
            mean_x=("x_minus_pre", "mean"),
            se_x=("x_minus_pre", lambda s: s.std(ddof=1) / np.sqrt(max(s.notna().sum(), 1))),
            n_firms=("c", "nunique"),
        )
    )
    stats.to_csv(out_dir / "raw_first_stage_by_exposure_bin.csv", index=False)
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    colors = {"Low/zero exposure": llstyle.NEUTRAL, "High exposure": llstyle.color(2)}
    labels = ["Low/zero exposure", "High exposure"]
    label_offsets = {label: float(offset) for label, offset in zip(labels, llstyle.offsets(len(labels)))}
    for label in labels:
        sub = stats.loc[stats["exposure_bin"].eq(label)].sort_values("t")
        errorbar_container = ax.errorbar(
            sub["t"].astype(float) + label_offsets.get(label, 0.0),
            sub["mean_x"],
            yerr=1.96 * sub["se_x"].fillna(0),
            fmt="o-",
            color=colors[label],
            ecolor=llstyle.rgba(colors[label]),
            elinewidth=llstyle.MARKER_SIZE,
            capsize=0,
            markersize=llstyle.MARKER_SIZE,
            label=label,
        )
        _soften_errorbar_interval(errorbar_container)
    ax.axvline(2016, color="black", linestyle="--", linewidth=1.4)
    ax.axhline(0, color="0.35", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("OPT hires minus firm 2011-2013 mean")
    ax.set_title("")
    llstyle.right_legend(ax)
    _savefig(fig, out_dir / "raw_first_stage_by_exposure_bin.png")


def build_dynamic_figures(out_dir: Path) -> None:
    dyn = _read_table(APR_ROOT / "dynamic_effect_coefficients.csv")
    if dyn.empty:
        return
    dyn["coef"] = _safe_num(dyn["coef"])
    dyn["se"] = _safe_num(dyn["se"])
    dyn.to_csv(out_dir / "dynamic_effect_coefficients.csv", index=False)
    for family, filename, title in [
        ("first_stage", "dynamic_first_stage_coefficients.png", "Dynamic first stage"),
        ("reduced_form", "dynamic_reduced_form_coefficients.png", "Dynamic reduced form"),
    ]:
        sub = dyn.loc[dyn["family"].eq(family)].sort_values("horizon")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
        errorbar_container = ax.errorbar(
            sub["horizon"],
            sub["coef"],
            yerr=1.96 * sub["se"].fillna(0),
            fmt="o-",
            color=llstyle.color(2),
            ecolor=llstyle.rgba(llstyle.color(2)),
            elinewidth=llstyle.MARKER_SIZE,
            capsize=0,
            markersize=llstyle.MARKER_SIZE,
        )
        _soften_errorbar_interval(errorbar_container)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.4)
        ax.axhline(0, color="0.35", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Outcome horizon relative to z_ct year")
        ax.set_ylabel("Coefficient on z_ct")
        ax.set_title("")
        _savefig(fig, out_dir / filename)


def _write_latex_table(path: Path, rows: list[list[str]], headers: list[str], note: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colspec = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.13}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\hline",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    if note:
        lines.append(rf"\\[-0.2em]{{\tiny {_latex_escape(note)}}}")
    lines.append(r"\endgroup")
    path.write_text("\n".join(lines) + "\n")


def _write_econ_regression_table(
    path: Path,
    columns: list[dict[str, object]],
    coef_var_label: str,
    note: str,
) -> None:
    """Write a paper-style regression table with models as columns and variables as rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [""] + [_latex_escape(col.get("title", "")) for col in columns]
    colspec = "l" + "c" * len(columns)
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
        _latex_escape(coef_var_label)
        + " & "
        + " & ".join(_latex_escape(_fmt_num(col.get("coef"), 4)) for col in columns)
        + r" \\",
        ""
        + " & "
        + " & ".join(
            _latex_escape(f"({_fmt_num(col.get('se'), 4)})") if _fmt_num(col.get("se"), 4) else ""
            for col in columns
        )
        + r" \\",
        r"\hline",
        "Mean dep. var. & " + " & ".join(_latex_escape(_fmt_num(col.get("dep_mean"), 3)) for col in columns) + r" \\",
        "Observations & " + " & ".join(_latex_escape(_fmt_num(col.get("n_obs"), 0)) for col in columns) + r" \\",
        "Firm FE & " + " & ".join("Yes" for _ in columns) + r" \\",
        "Year FE & " + " & ".join("Yes" for _ in columns) + r" \\",
        "Baseline size-growth $\\times$ year FE & " + " & ".join("Yes" for _ in columns) + r" \\",
        r"\hline",
        r"\end{tabular}",
        rf"\\[-0.2em]{{\tiny {_latex_escape(note)}}}",
        r"\endgroup",
    ]
    path.write_text("\n".join(lines) + "\n")


def _fit_slide_model(panel: pd.DataFrame, lhs: str, rhs: str, estimator: str) -> dict[str, object]:
    try:
        import pyfixest as pf
    except Exception as exc:
        return {"error": str(exc)}
    work = panel.loc[:, ["c", "t", "baseline_size_growth_year_fe", lhs, rhs]].dropna().copy()
    if work.empty:
        return {"error": "empty model panel"}
    if estimator == "ppml":
        work = work.loc[_safe_num(work[lhs]).ge(0)].copy()
    if work.empty or work[lhs].nunique(dropna=True) <= 1 or work[rhs].nunique(dropna=True) <= 1:
        return {"error": "insufficient variation"}
    formula = f"{lhs} ~ {rhs} | c + t + baseline_size_growth_year_fe"
    try:
        fit = pf.fepois(formula, data=work, vcov={"CRV1": "c"}) if estimator == "ppml" else pf.feols(formula, data=work, vcov={"CRV1": "c"})
        return {
            "coef": float(fit.coef().loc[rhs]),
            "se": float(fit.se().loc[rhs]),
            "n_obs": int(getattr(fit, "_N", getattr(fit, "nobs", len(work)))),
            "dep_mean": float(_safe_num(work[lhs]).mean()),
            "estimator": estimator,
        }
    except Exception as exc:
        return {"error": str(exc), "n_obs": len(work)}


def _build_slide_regression_columns(
    panel: pd.DataFrame,
    rhs: str,
    title_suffix: str = "",
) -> list[dict[str, object]]:
    specs = [
        ("OPT hires", "x_ct", "ppml"),
        ("Log emp.", "log1p_y_cst_lag0", "ols"),
        ("Log hires", "log1p_y_new_hires_lag0", "ols"),
        ("Log foreign hires", "log1p_y_new_hires_foreign_lag0", "ols"),
        ("Log native hires", "log1p_y_new_hires_native_lag0", "ols"),
        ("Avg tenure", "avg_tenure_years_lag0", "ols"),
    ]
    columns: list[dict[str, object]] = []
    for title, lhs, estimator in specs:
        if lhs not in panel.columns:
            continue
        result = _fit_slide_model(panel, lhs=lhs, rhs=rhs, estimator=estimator)
        result["title"] = f"{title}{title_suffix}"
        result["lhs"] = lhs
        result["rhs"] = rhs
        columns.append(result)
    return columns


def build_preferred_binscatter_assets(out_dir: Path) -> None:
    panel = _prepare_preferred_panel_for_binscatter()
    if panel.empty:
        return
    scatter_dir = out_dir / "residualized_binscatter"
    outcome_specs = [
        (
            "first_stage",
            "x_ct",
            "Residualized master's OPT hires",
            "First stage",
        ),
        (
            "reduced_form",
            "log1p_y_cst_lag0",
            "Residualized log employment",
            "Reduced form: employment",
        ),
        (
            "new_hires",
            "log1p_y_new_hires_lag0",
            "Residualized log new hires",
            "Reduced form: new hires",
        ),
        (
            "foreign_new_hires",
            "log1p_y_new_hires_foreign_lag0",
            "Residualized log foreign new hires",
            "Reduced form: foreign new hires",
        ),
        (
            "native_new_hires",
            "log1p_y_new_hires_native_lag0",
            "Residualized log native new hires",
            "Reduced form: native new hires",
        ),
        (
            "avg_tenure",
            "avg_tenure_years_lag0",
            "Residualized average tenure",
            "Reduced form: average tenure",
        ),
    ]
    x_specs = [
        (
            "preferred",
            "log1p_z_ct",
            "Residualized log(1 + predicted OPT supply), log(1+z_ct)",
            "Preferred log z_ct",
        ),
        (
            "matched_step",
            "log1p_z_ct_v4_matched_step",
            "Residualized log(1 + matched predicted OPT supply)",
            "Matched step log z_ct",
        ),
    ]
    for x_prefix, x_col, x_label, title_prefix in x_specs:
        if x_col not in panel.columns:
            continue
        for outcome_name, y_col, y_label, outcome_title in outcome_specs:
            if y_col not in panel.columns:
                continue
            _plot_residualized_binscatter(
                panel=panel,
                x_col=x_col,
                y_col=y_col,
                x_label=x_label,
                y_label=y_label,
                title=f"{title_prefix}: {outcome_title}",
                out_png=scatter_dir / f"{x_prefix}_{outcome_name}_twfe_binscatter.png",
                out_csv=scatter_dir / f"{x_prefix}_{outcome_name}_twfe_binscatter.csv",
            )
    # Keep the exact regression-generated cumulative plots close to the slide assets.
    for src, dst_name in [
        (
            APR_ROOT / "cumulative_innovation_specs" / "residualized_binscatter" / "first_stage_twfe_binscatter.png",
            "cumulative_diff_first_stage_twfe_binscatter.png",
        ),
        (
            APR_ROOT / "cumulative_innovation_specs" / "residualized_binscatter" / "reduced_form_twfe_binscatter.png",
            "cumulative_diff_reduced_form_twfe_binscatter.png",
        ),
    ]:
        _copy_if_exists(src, scatter_dir / dst_name)


def build_preferred_static_table(out_dir: Path) -> None:
    panel = _prepare_preferred_panel_for_binscatter()
    if panel.empty or "log1p_z_ct" not in panel.columns:
        return
    columns = _build_slide_regression_columns(panel, rhs="log1p_z_ct")
    _write_econ_regression_table(
        out_dir / "preferred_static_regression_table.tex",
        columns,
        "log(1+z_ct)",
        "Preferred specification. First-stage column is PPML; other columns are OLS. All columns absorb firm, year, and baseline-size-decile by baseline-growth-quantile by year fixed effects. Standard errors are clustered by firm.",
    )
    pd.DataFrame(columns).to_csv(out_dir / "preferred_static_regression_table.csv", index=False)


def build_matched_design_tables(out_dir: Path) -> None:
    panel = _prepare_preferred_panel_for_binscatter()
    if panel.empty:
        return
    for rhs, label, filename in [
        (
            "log1p_z_ct_v4_matched_step",
            "log(1+matched step z_ct)",
            "matched_step_regression_table.tex",
        ),
        (
            "log1p_z_ct_v5_matched_pulse",
            "log(1+matched pulse z_ct)",
            "matched_pulse_regression_table.tex",
        ),
    ]:
        if rhs not in panel.columns:
            continue
        columns = _build_slide_regression_columns(panel, rhs=rhs)
        _write_econ_regression_table(
            out_dir / filename,
            columns,
            label,
            "Matched-school design specification. First-stage column is PPML; other columns are OLS. All columns absorb firm, year, and baseline-size-decile by baseline-growth-quantile by year fixed effects. Standard errors are clustered by firm.",
        )
        pd.DataFrame(columns).to_csv(out_dir / filename.replace(".tex", ".csv"), index=False)


def build_variant_robustness_tables(out_dir: Path) -> None:
    main = _read_table(APR_ROOT / "regression_variant_headline_summary.csv")
    cumul = _read_table(APR_ROOT / "cumulative_innovation_specs" / "regression_variant_headline_summary.csv")
    frames = [df for df in [main, cumul] if not df.empty]
    if not frames:
        return
    summary = pd.concat(frames, ignore_index=True)
    summary = summary.loc[summary["outcome_col"].eq("y_cst_lag0")].copy()
    summary = summary.drop_duplicates(subset=["instrument_variant"], keep="last")
    variant_order = [
        "z_ct",
        "z_ct_event_pulse",
        "z_ct_flow_diff",
        "z_ct_flow_ar_resid",
        "z_ct_flow_diff_cumulative",
        "z_ct_flow_ar_resid_cumulative",
        "z_ct_common_base_level",
        "z_ct_common_base_asinh",
        "z_ct_event_step_dose",
        "z_ct_share_2008_2010",
        "z_ct_share_2011_2013",
        "z_ct_full",
    ]
    rows: list[list[str]] = []
    for variant in variant_order:
        hit = summary.loc[summary["instrument_variant"].eq(variant)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        rows.append(
            [
                _latex_escape(VARIANT_LABELS.get(variant, row.get("variant_label", variant))),
                _latex_escape(_coef_se_cell(row.get("fs_twfe_coef"), row.get("fs_twfe_se"))),
                _latex_escape(_fmt_num(row.get("fs_twfe_f"), 2)),
                _latex_escape(_coef_se_cell(row.get("rf_twfe_coef"), row.get("rf_twfe_se"))),
            ]
        )
    _write_latex_table(
        out_dir / "variant_robustness_table.tex",
        rows,
        ["Exposure", "First stage", "FS F", "RF: log employment"],
        "All rows use the configured firm, year, and baseline-size/growth by year fixed effects where available.",
    )

    cumul_rows: list[list[str]] = []
    if not cumul.empty:
        for _, row in cumul.iterrows():
            cumul_rows.append(
                [
                    _latex_escape(OUTCOME_LABELS.get(row.get("outcome_col"), row.get("outcome_col", ""))),
                    _latex_escape(VARIANT_LABELS.get(row.get("instrument_variant"), row.get("variant_label", ""))),
                    _latex_escape(_coef_se_cell(row.get("fs_twfe_coef"), row.get("fs_twfe_se"))),
                    _latex_escape(_coef_se_cell(row.get("rf_twfe_coef"), row.get("rf_twfe_se"))),
                ]
            )
    _write_latex_table(
        out_dir / "cumulative_innovation_table.tex",
        cumul_rows,
        ["Outcome", "Cumulative exposure", "First stage", "Reduced form"],
        "Cumulative exposures sum the first-difference or AR-residual innovation shocks through year t.",
    )


def build_distributed_lag_table(out_dir: Path) -> None:
    rows: list[list[str]] = []
    for variant in ["z_ct_flow_diff", "z_ct_flow_ar_resid"]:
        dl = _read_table(
            APR_ROOT / "regression_variants" / "y_cst_lag0" / variant / "distributed_lag_coefficients.csv"
        )
        if dl.empty:
            continue
        for family, label in [("first_stage", "First stage"), ("reduced_form", "Reduced form")]:
            sub = dl.loc[dl["family"].eq(family)].copy()
            if sub.empty:
                continue
            last = sub.sort_values("lag").iloc[-1]
            lag0 = sub.loc[sub["lag"].eq(0)].iloc[0] if sub["lag"].eq(0).any() else last
            rows.append(
                [
                    _latex_escape(VARIANT_LABELS.get(variant, variant)),
                    _latex_escape(label),
                    _latex_escape(_coef_se_cell(lag0.get("coef"), lag0.get("se"))),
                    _latex_escape(_coef_se_cell(
                        last.get("cumulative_coef_through_lag"),
                        last.get("cumulative_se_independence_proxy"),
                    )),
                    _latex_escape(_fmt_num(last.get("n_obs"), 0)),
                ]
            )
    _write_latex_table(
        out_dir / "distributed_lag_summary_table.tex",
        rows,
        ["Innovation", "Equation", "Lag 0", "Cumulative lag 0-3", "N"],
        "Cumulative SE is the saved independence-proxy SE; the plotted coefficients use the regression output files.",
    )
    for variant in ["z_ct_flow_diff", "z_ct_flow_ar_resid"]:
        src_dir = APR_ROOT / "regression_variants" / "y_cst_lag0" / variant
        prefix = "flow_diff" if variant.endswith("flow_diff") else "flow_ar_resid"
        for name in ["distributed_lag_first_stage_coefficients.png", "distributed_lag_reduced_form_coefficients.png"]:
            _copy_if_exists(src_dir / name, out_dir / f"{prefix}_{name}")


def build_diagnostic_tables(out_dir: Path) -> None:
    autocorr = _read_table(APR_ROOT / "diagnostics" / "shock_autocorrelation_diagnostics.csv")
    if not autocorr.empty:
        rows: list[list[str]] = []
        variants = ["z_ct", "z_ct_event_pulse", "z_ct_flow_diff", "z_ct_flow_ar_resid"]
        for variant in variants:
            school = autocorr.loc[
                autocorr["level"].eq("school_shock")
                & autocorr["variant"].eq(variant.replace("z_ct", "g_kt", 1))
                & autocorr["lag"].eq(1)
            ]
            firm = autocorr.loc[
                autocorr["level"].eq("firm_shift_share")
                & autocorr["variant"].eq(variant)
                & autocorr["lag"].eq(1)
            ]
            rows.append(
                [
                    _latex_escape(VARIANT_LABELS.get(variant, variant)),
                    _latex_escape(_fmt_num(school["autocorrelation"].iloc[0], 3) if not school.empty else ""),
                    _latex_escape(_fmt_num(firm["autocorrelation"].iloc[0], 3) if not firm.empty else ""),
                    _latex_escape(_fmt_num(firm["n_units"].iloc[0], 0) if not firm.empty else ""),
                ]
            )
        _write_latex_table(
            out_dir / "serial_correlation_table.tex",
            rows,
            ["Exposure", "School lag-1 rho", "Firm z lag-1 rho", "Firms"],
            "Innovation measures sharply reduce serial correlation relative to the raw annual flow.",
        )

    shock_diag = _read_table(APR_ROOT / "diagnostics" / "shock_level_diagnostics_by_variant.csv")
    if not shock_diag.empty:
        rows = []
        for variant in ["z_ct", "z_ct_event_pulse", "z_ct_flow_diff", "z_ct_flow_ar_resid", "z_ct_common_base_level"]:
            hit = shock_diag.loc[shock_diag["instrument_variant"].eq(variant)]
            if hit.empty:
                continue
            row = hit.iloc[0]
            rows.append(
                [
                    _latex_escape(VARIANT_LABELS.get(variant, variant)),
                    _latex_escape(_fmt_num(row.get("active_schools"), 0)),
                    _latex_escape(_fmt_num(row.get("effective_schools_by_abs_component_mass"), 1)),
                    _latex_escape(_fmt_num(row.get("top5_school_abs_component_share"), 3)),
                ]
            )
        _write_latex_table(
            out_dir / "shock_level_diagnostics_table.tex",
            rows,
            ["Exposure", "Active schools", "Effective schools", "Top-5 share"],
            "Effective school count is based on absolute component-mass concentration.",
        )

    fals = _read_table(APR_ROOT / "falsification_tests" / "falsification_headline_summary.csv")
    if not fals.empty:
        rows = []
        for _, row in fals.iterrows():
            rows.append(
                [
                    _latex_escape(VARIANT_LABELS.get(row.get("instrument_variant"), row.get("variant_label", ""))),
                    _latex_escape(_coef_se_cell(row.get("fs_twfe_coef"), row.get("fs_twfe_se"))),
                    _latex_escape(_fmt_num(row.get("fs_twfe_f"), 2)),
                    _latex_escape(_coef_se_cell(row.get("rf_twfe_coef"), row.get("rf_twfe_se"))),
                ]
            )
        _write_latex_table(
            out_dir / "falsification_table.tex",
            rows,
            ["Placebo", "First stage", "FS F", "RF: log employment"],
            "Placebo shock timing is shifted before the actual treatment window.",
        )


def build_first_stage_table(out_dir: Path) -> None:
    reg = _read_table(APR_ROOT / "reg_table.csv")
    comp = _read_table(APR_ROOT / "first_stage_outcome_comparison.csv")
    panel = _read_table(DATA_ROOT / "analysis_panel.parquet")
    base_mean = np.nan
    if not panel.empty and "masters_opt_hires_correction_aware" in panel.columns:
        pre = panel.loc[_safe_num(panel.get("t")).between(2011, 2013), "masters_opt_hires_correction_aware"]
        base_mean = float(_safe_num(pre).mean())
    rows: list[list[str]] = []
    if not reg.empty:
        for label in ["first_stage_no_fe", "first_stage_twfe"]:
            hit = reg.loc[reg["label"].eq(label)]
            if hit.empty:
                continue
            row = hit.iloc[0]
            rows.append(
                [
                    _latex_escape("PPML, no FE" if label.endswith("no_fe") else "PPML, firm + year FE"),
                    _fmt_num(row.get("coef_instrument"), 4),
                    f"({_fmt_num(row.get('se_instrument'), 4)})",
                    _fmt_num(row.get("n_obs"), 0),
                    "No" if label.endswith("no_fe") else "Yes",
                    "No",
                    _fmt_num(base_mean, 3),
                ]
            )
    if not comp.empty:
        fs_labels = {
            "fs0": "LPM/asinh, firm + year FE",
            "fs1": "Add baseline size x year FE",
            "fs2": "Add current size x year FE",
            "fs4": "Add size x growth x year FE",
        }
        for outcome in ["x_bin", "x_asinh"]:
            hit = comp.loc[comp["outcome"].eq(outcome)]
            if hit.empty:
                continue
            for code, label in fs_labels.items():
                rows.append(
                    [
                        _latex_escape(f"{label} ({outcome})"),
                        _fmt_num(hit.iloc[0].get(f"{code}_coef"), 4),
                        f"({_fmt_num(hit.iloc[0].get(f'{code}_se'), 4)})",
                        _fmt_num(hit.iloc[0].get(f"{code}_n"), 0),
                        "Yes",
                        "Yes" if code in {"fs1", "fs2", "fs4"} else "No",
                        _fmt_num(base_mean, 3),
                    ]
                )
    _write_latex_table(
        out_dir / "first_stage_regression_table.tex",
        rows[:8],
        ["Spec", "Coef", "SE", "N", "Firm/year FE", "Year x size FE", "Base mean"],
        "Coefficient on z_ct. Baseline mean is the 2011-2013 firm-year mean of master OPT hires in the saved panel.",
    )


def build_reduced_form_table(out_dir: Path) -> None:
    estimates = _estimate_preferred_reduced_form()
    rows: list[list[str]] = []
    for _, row in estimates.iterrows():
        rows.append(
            [
                _latex_escape(row.get("label", "")),
                _fmt_num(row.get("coef"), 4),
                f"({_fmt_num(row.get('se'), 4)})" if pd.notna(row.get("se")) else "",
                _fmt_num(row.get("n_obs"), 0),
                _fmt_num(row.get("baseline_mean"), 2),
            ]
        )
    _write_latex_table(
        out_dir / "reduced_form_regression_table.tex",
        rows,
        ["Outcome", "Coef", "SE", "N", "Base mean"],
        "Preferred spec is firm FE plus year x baseline-size FE. Count outcomes are log1p-transformed; compensation is log-transformed.",
    )
    if not estimates.empty:
        estimates.to_csv(out_dir / "reduced_form_regression_table.csv", index=False)


def _estimate_preferred_reduced_form() -> pd.DataFrame:
    """Estimate preferred static RF specs from saved panels for deck tables."""
    try:
        import pyfixest as pf
    except Exception as exc:
        print(f"[shift-share slides] pyfixest unavailable; reduced-form coefficients skipped: {exc}")
        return pd.DataFrame()

    z = _read_table(DATA_ROOT / "instrument_panel.parquet")
    base_panel = _read_table(DATA_ROOT / "analysis_panel.parquet")
    panel = _read_table(EXPOSURE_ROOT / "opt_exposure_analysis_panel.parquet")
    workforce = _read_table(EXPOSURE_ROOT / "wrds_company_year_workforce.parquet")
    if z.empty or base_panel.empty or panel.empty:
        return pd.DataFrame()

    z = z.loc[:, ["c", "t", "z_ct"]].copy()
    z["c"] = z["c"].astype(str)
    z["t"] = _safe_num(z["t"]).astype("Int64")
    z["z_ct"] = _safe_num(z["z_ct"]).fillna(0.0)

    base = base_panel.loc[:, ["c", "t", "y_cst_lag0"]].copy()
    base["c"] = base["c"].astype(str)
    base["t"] = _safe_num(base["t"]).astype("Int64")
    balanced = (
        base.loc[base["t"].between(2008, 2022)]
        .groupby("c")["t"]
        .nunique()
        .loc[lambda s: s.eq(15)]
        .index
    )
    baseline_size = (
        base.loc[base["c"].isin(balanced) & base["t"].between(2011, 2013)]
        .groupby("c", as_index=False)["y_cst_lag0"]
        .mean()
        .rename(columns={"y_cst_lag0": "baseline_size"})
    )
    baseline_size["baseline_size_decile"] = pd.qcut(
        baseline_size["baseline_size"].rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    ).astype(int) + 1

    panel = panel.copy()
    panel["c"] = panel["c"].astype(str)
    panel["t"] = _safe_num(panel["t"]).astype("Int64")
    panel = panel.loc[panel["c"].isin(balanced) & panel["t"].between(2010, 2022)].copy()
    keep_cols = [
        "c",
        "t",
        "y_cst_lag0",
        "y_new_hires_lag0",
        "y_new_hires_foreign_lag0",
        "y_new_hires_native_lag0",
    ]
    panel = panel.loc[:, [c for c in keep_cols if c in panel.columns]]
    if not workforce.empty:
        wf = workforce.loc[:, [
            c
            for c in ["c", "t", "avg_tenure_years_annual", "total_comp_mean_annual"]
            if c in workforce.columns
        ]].copy()
        wf["c"] = wf["c"].astype(str)
        wf["t"] = _safe_num(wf["t"]).astype("Int64")
        panel = panel.merge(wf, on=["c", "t"], how="left")

    reg = panel.merge(z, on=["c", "t"], how="left").merge(
        baseline_size[["c", "baseline_size_decile"]], on="c", how="inner"
    )
    reg["z_ct"] = _safe_num(reg["z_ct"]).fillna(0.0)
    reg["size_year_fe"] = reg["t"].astype(str) + "_d" + reg["baseline_size_decile"].astype(str)
    outcome_specs = [
        ("Employment", "y_cst_lag0", "log1p_y_cst_lag0", "log1p"),
        ("New hires", "y_new_hires_lag0", "log1p_y_new_hires_lag0", "log1p"),
        ("New hires, foreign", "y_new_hires_foreign_lag0", "log1p_y_new_hires_foreign_lag0", "log1p"),
        ("New hires, native", "y_new_hires_native_lag0", "log1p_y_new_hires_native_lag0", "log1p"),
        ("Avg tenure", "avg_tenure_years_annual", "avg_tenure_years_annual", "level"),
        ("Avg compensation", "total_comp_mean_annual", "log_total_comp_mean_annual", "log"),
    ]
    rows: list[dict[str, object]] = []
    for label, source_col, lhs, transform in outcome_specs:
        if source_col not in reg.columns:
            rows.append({"label": label, "source_col": source_col})
            continue
        work = reg.loc[:, ["c", "size_year_fe", "z_ct", source_col]].copy()
        work[source_col] = _safe_num(work[source_col])
        baseline_mean = float(work[source_col].mean())
        if transform == "log1p":
            work[lhs] = np.log1p(work[source_col].clip(lower=0))
        elif transform == "log":
            work[lhs] = np.log(work[source_col].where(work[source_col] > 0))
        else:
            work[lhs] = work[source_col]
        work = work.dropna(subset=[lhs, "z_ct", "c", "size_year_fe"])
        if work.empty:
            rows.append({"label": label, "source_col": source_col, "baseline_mean": baseline_mean})
            continue
        try:
            fit = pf.feols(f"{lhs} ~ z_ct | c + size_year_fe", data=work, vcov={"CRV1": "c"})
            coef = float(fit.coef().loc["z_ct"])
            se = float(fit.se().loc["z_ct"])
            n_obs = int(getattr(fit, "nobs", len(work)))
        except Exception as exc:
            print(f"[shift-share slides] RF estimate failed for {label}: {exc}")
            coef, se, n_obs = np.nan, np.nan, len(work)
        rows.append(
            {
                "label": label,
                "source_col": source_col,
                "lhs": lhs,
                "transform": transform,
                "coef": coef,
                "se": se,
                "n_obs": n_obs,
                "baseline_mean": baseline_mean,
            }
        )
    return pd.DataFrame(rows)


def build_heterogeneity_figure(out_dir: Path) -> None:
    het = _read_table(APR_ROOT / "first_stage_heterogeneity_diagnostics.csv")
    if het.empty or "interaction_slope" not in het.columns:
        return
    plot = het.loc[het["interaction_kind"].isin(["baseline_size", "current_size", "growth", "joint_size_growth"])].copy()
    if plot.empty:
        return
    label_map = {
        "baseline_size": "Baseline size",
        "current_size": "Current size",
        "growth": "Growth",
        "joint_size_growth": "Size x growth",
    }
    plot["label"] = plot["interaction_kind"].map(label_map).fillna(plot["interaction_kind"])
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    ax.bar(plot["label"], _safe_num(plot["interaction_slope"]), color=[llstyle.color(idx) for idx in range(len(plot))])
    ax.axhline(0, color="0.35", linestyle=":", linewidth=1.2)
    ax.set_ylabel("Slope through interaction coefficients")
    ax.set_title("")
    ax.tick_params(axis="x", rotation=20)
    _savefig(fig, out_dir / "first_stage_heterogeneity_summary.png")


def build_assets(out_dir: Path = DEFAULT_OUT_DIR) -> None:
    _set_plot_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    build_school_event_figures(out_dir)
    build_zct_selected_firm_figure(out_dir)
    build_raw_first_stage_figure(out_dir)
    build_dynamic_figures(out_dir)
    build_heterogeneity_figure(out_dir)
    build_preferred_binscatter_assets(out_dir)
    build_preferred_static_table(out_dir)
    build_matched_design_tables(out_dir)
    build_variant_robustness_tables(out_dir)
    build_distributed_lag_table(out_dir)
    build_diagnostic_tables(out_dir)
    build_first_stage_table(out_dir)
    build_reduced_form_table(out_dir)
    print(f"[shift-share slides] wrote assets to {out_dir}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shift-share slide assets.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    build_assets(args.out_dir)


if __name__ == "__main__":
    main()
