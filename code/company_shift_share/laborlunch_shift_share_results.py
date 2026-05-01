"""Build the labor-lunch firm shift-share slides and slide assets.

This is a downstream helper for slides_laborlunch_20260507.  It reads the
saved April 2026 shift-share outputs, uses the IHMP-share level instrument
where available, and writes the two TeX files already included by the deck:

  output/company_shift_share/slides_20260507_shift_share/firm_level_main.tex
  output/company_shift_share/slides_20260507_shift_share/firm_level_appendix.tex
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import laborlunch_plot_style as llstyle

from company_shift_share.first_stage_variant_sweep import (
    APR_ROOT,
    FE_SPECS,
    _add_component_built_exposures,
    _base_analysis_panel,
    _fit_one,
    _load_panel,
)
from company_shift_share.shift_share_deck_assets import (
    DEFAULT_OUT_DIR,
    _plot_residualized_binscatter,
    _set_plot_style,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
VARIANT_ROOT = CODE_ROOT / "output" / "company_shift_share" / "first_stage_variant_sweep_apr2026"
IHMP_VARIANT_ROOT = CODE_ROOT / "output" / "company_shift_share" / "first_stage_variant_sweep_ihmp_share_apr2026"
RF_NH_IHMP_ROOT = CODE_ROOT / "output" / "company_shift_share" / "reduced_form_new_hires_ihmp_share_sweep_apr2026"
DIAG_ROOT = APR_ROOT / "diagnostics"
SOURCE_EXPOSURE_ROOT = Path("/home/yk0581/data/out/company_shift_share_apr2026")
MATCHED_EXPOSURE_ROOT = SOURCE_EXPOSURE_ROOT / "matched_exposure_design"
MATCHED_EXPOSURE_FIGURE = (
    MATCHED_EXPOSURE_ROOT / "figures" / "common_break_event_study_full_path_any_opt_hires_correction_aware.png"
)
Z_COL = "z_ihmp_share"
RICH_FE = "firm_year_size_growth_fe"
RICH_FE_LABEL = "Firm + year + year x baseline size x baseline growth FE"
ZCT_ES_EVENT_YEAR = 2015
ZCT_ES_REF_YEAR = 2014
ZCT_ES_START_YEAR = 2013
ZCT_ES_END_YEAR = 2022
ZCT_ES_GROUP_ORDER = [
    "Zero exposure",
    "Low positive exposure",
    "Middle positive exposure",
    "High positive exposure",
]
ZCT_ES_COLORS = {
    "Zero exposure": llstyle.NEUTRAL,
    "Low positive exposure": llstyle.color(2),
    "Middle positive exposure": llstyle.color(4),
    "High positive exposure": llstyle.color(1),
}
ZCT_ES_OUTCOME_LABELS = {
    "any_opt_hires_correction_aware": "Any OPT hires",
    "log1p_y_cst_lag0": "Log employment",
    "log1p_y_new_hires_lag0": "Log new hires",
    "log1p_y_new_hires_foreign_lag0": "Log foreign new hires",
    "log1p_y_new_hires_native_lag0": "Log native new hires",
    "log1p_y_cst_foreign_lag0": "Log foreign employment",
    "log1p_y_cst_native_lag0": "Log native employment",
}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fmt(value: object, digits: int = 3) -> str:
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


def _fmt_signed(value: object, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(val):
        return ""
    return f"{val:+.{digits}f}"


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def _savefig(fig: plt.Figure, path: Path) -> None:
    llstyle.savefig(fig, path)


def _load_ihmp_panel() -> pd.DataFrame:
    panel = _load_panel(APR_ROOT)
    panel = _add_component_built_exposures(panel, APR_ROOT)
    panel = _base_analysis_panel(panel, 2013, 2022)
    panel[Z_COL] = _safe_num(panel[Z_COL]).fillna(0.0)
    panel["x_asinh"] = np.arcsinh(_safe_num(panel["masters_opt_hires_correction_aware"]).fillna(0.0))
    panel["x_bin"] = (_safe_num(panel["masters_opt_hires_correction_aware"]).fillna(0.0) > 0).astype(float)
    panel["t"] = panel["t_fe"].astype(str)
    exposure_panel_path = SOURCE_EXPOSURE_ROOT / "opt_exposure_analysis_panel.parquet"
    if exposure_panel_path.exists():
        add_cols = [
            "c",
            "t",
            "any_opt_hires_correction_aware",
            "y_cst_foreign_lag0",
            "y_cst_native_lag0",
        ]
        exposure_cols = _read_table(exposure_panel_path)
        exposure_cols = exposure_cols.loc[:, [c for c in add_cols if c in exposure_cols.columns]].copy()
        if {"c", "t"}.issubset(exposure_cols.columns):
            exposure_cols["c"] = exposure_cols["c"].astype(str)
            exposure_cols["t_num"] = pd.to_numeric(exposure_cols["t"], errors="coerce")
            exposure_cols = exposure_cols.drop(columns=["t"]).drop_duplicates(["c", "t_num"])
            panel = panel.merge(exposure_cols, on=["c", "t_num"], how="left")
    return panel


def _static_summary_to_columns(static: pd.DataFrame, title_map: dict[str, str]) -> list[dict[str, object]]:
    columns: list[dict[str, object]] = []
    for fe_key, title in title_map.items():
        hit = static.loc[static["fe_spec"].eq(fe_key)]
        row = hit.iloc[0] if not hit.empty else pd.Series(dtype=object)
        columns.append(
            {
                "title": title,
                "coef": row.get("coef"),
                "se": row.get("se"),
                "n": row.get("n_obs"),
                "mean": row.get("lhs_mean"),
                "f": row.get("f_stat"),
                "firm_fe": "Yes" if fe_key != "no_fe" else "No",
                "year_fe": "Yes" if fe_key != "no_fe" else "No",
                "rich_fe": "Yes" if fe_key == RICH_FE else "No",
            }
        )
    return columns


def _write_econ_table(
    path: Path,
    dep_var: str,
    coef_label: str,
    columns: list[dict[str, object]],
    note: str,
) -> None:
    headers = [""] + [str(c["title"]) for c in columns]
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        rf"\begin{{tabular}}{{l{'c' * len(columns)}}}",
        r"\hline\hline",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\hline",
        coef_label
        + " & "
        + " & ".join(_fmt(c.get("coef"), 4) for c in columns)
        + r" \\",
        ""
        + " & "
        + " & ".join(f"({_fmt(c.get('se'), 4)})" if _fmt(c.get("se"), 4) else "" for c in columns)
        + r" \\",
        r"\hline",
        "Dependent variable mean & " + " & ".join(_fmt(c.get("mean"), 3) for c in columns) + r" \\",
        "Observations & " + " & ".join(_fmt(c.get("n"), 0) for c in columns) + r" \\",
        "F-statistic & " + " & ".join(_fmt(c.get("f"), 2) for c in columns) + r" \\",
        "Firm FE & " + " & ".join(str(c.get("firm_fe", "")) for c in columns) + r" \\",
        "Year FE & " + " & ".join(str(c.get("year_fe", "")) for c in columns) + r" \\",
        "Year x baseline size x growth FE & "
        + " & ".join(str(c.get("rich_fe", "")) for c in columns)
        + r" \\",
        r"\hline\hline",
        r"\end{tabular}",
        rf"\\[-0.2em]{{\tiny Dependent variable: {_latex_escape(dep_var)}. {note}}}",
        r"\endgroup",
    ]
    _write_text(path, "\n".join(lines))


def _rich_row(df: pd.DataFrame, variant: str | None = None) -> pd.Series:
    if df.empty or "fe_spec" not in df.columns:
        return pd.Series(dtype=object)
    hit = df.loc[df["fe_spec"].eq(RICH_FE)].copy()
    if variant is not None and "variant" in hit.columns:
        hit = hit.loc[hit["variant"].eq(variant)]
    if hit.empty:
        return pd.Series(dtype=object)
    return hit.iloc[0]


def _sample_label(row: pd.Series, *, pairs: bool = False) -> str:
    if row.empty:
        return ""
    if pairs:
        return f"{_fmt(row.get('n_pairs'), 0)} pairs"
    firms = _fmt(row.get("n_companies"), 0)
    obs = _fmt(row.get("n_obs"), 0)
    if firms and obs:
        return f"{firms} firms; {obs} obs."
    return firms or obs


def _first_stage_metric(row: pd.Series) -> str:
    if row.empty:
        return ""
    coef = _fmt(row.get("coef"), 4)
    se = _fmt(row.get("se"), 4)
    f_stat = _fmt(row.get("f_stat"), 1)
    effect = _fmt(row.get("effect_per_1sd_pct_or_level", row.get("effect_per_1sd_pct_or_pp")), 1)
    parts = []
    if coef:
        parts.append(f"b={coef}" + (f" ({se})" if se else ""))
    if f_stat:
        parts.append(f"F={f_stat}")
    if effect:
        parts.append(f"1 SD={effect}%")
    return "; ".join(parts)


def _strength_label(row: pd.Series) -> str:
    if row.empty:
        return ""
    coef = row.get("coef")
    f_stat = row.get("f_stat")
    try:
        coef_val = float(coef)
        f_val = float(f_stat)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(coef_val):
        return ""
    if f_val >= 10:
        strength = "Strong"
    elif f_val >= 5:
        strength = "Moderate"
    else:
        strength = "Weak"
    if coef_val > 0:
        return f"{strength} +"
    if coef_val < 0:
        return f"{strength} -"
    return f"{strength} 0"


def build_first_stage_design_comparison(out_dir: Path) -> None:
    ihmp_static = _read_table(IHMP_VARIANT_ROOT / "first_stage_variant_static_summary.csv")
    variant_static = _read_table(VARIANT_ROOT / "first_stage_variant_static_summary.csv")
    common_break_summary = _read_table(MATCHED_EXPOSURE_ROOT / "tables" / "common_break_event_study_summary.csv")
    common_break = _read_table(MATCHED_EXPOSURE_ROOT / "tables" / "common_break_event_study.csv")
    diagnostics = _read_json(MATCHED_EXPOSURE_ROOT / "design_diagnostics.json")

    rows: list[dict[str, str]] = []
    ihmp = _rich_row(ihmp_static, "ihmp_share")
    rows.append(
        {
            "design_family": "School-shock shift-share",
            "exposure_variant": "IHMP-share level",
            "first_stage_metric": _first_stage_metric(ihmp),
            "strength_sign": _strength_label(ihmp),
            "sample": _sample_label(ihmp),
        }
    )

    variant_labels = [
        ("raw_ppml", "Raw annual flow"),
        ("matched_step", "Matched step"),
        ("matched_pulse", "Matched pulse"),
        ("first_diff", "First diff."),
        ("ar_residual", "AR residual"),
        ("event_pulse", "Event pulse"),
        ("event_step_broad", "Event step"),
        ("common_base_level", "Common base"),
        ("full_sample_shares", "Full-sample shares"),
    ]
    for key, label in variant_labels:
        row = _rich_row(variant_static, key)
        rows.append(
            {
                "design_family": "Timing/comparison variants",
                "exposure_variant": label,
                "first_stage_metric": _first_stage_metric(row),
                "strength_sign": _strength_label(row),
                "sample": _sample_label(row),
            }
        )

    direct_summary = common_break_summary.loc[
        common_break_summary.get("outcome_col", pd.Series(dtype=object)).eq("any_opt_hires_correction_aware")
    ]
    direct_row = direct_summary.iloc[0] if not direct_summary.empty else pd.Series(dtype=object)
    direct_event = common_break.loc[
        common_break.get("outcome_col", pd.Series(dtype=object)).eq("any_opt_hires_correction_aware")
    ].copy()
    coef_2015 = direct_event.loc[direct_event.get("year", pd.Series(dtype=object)).eq(2015), "coef"]
    coef_2016 = direct_event.loc[direct_event.get("year", pd.Series(dtype=object)).eq(2016), "coef"]
    auc = (
        diagnostics.get("trajectory_specs", {})
        .get("full_path", {})
        .get("propensity", {})
        .get("evaluation_auc")
    )
    if auc is None:
        auc = diagnostics.get("propensity", {}).get("evaluation_auc")
    rows.append(
        {
            "design_family": "Direct firm-exposure model",
            "exposure_variant": "Predicted high exposure x policy break",
            "first_stage_metric": (
                f"2015 {_fmt_signed(coef_2015.iloc[0] if not coef_2015.empty else np.nan)}; "
                f"2016 {_fmt_signed(coef_2016.iloc[0] if not coef_2016.empty else np.nan)}; "
                f"AUC={_fmt(auc, 3)}"
            ),
            "strength_sign": "Policy-break +",
            "sample": _sample_label(direct_row, pairs=True),
        }
    )

    table_df = pd.DataFrame(rows)
    table_df.to_csv(out_dir / "first_stage_design_comparison.csv", index=False)

    lines = [
        r"\begingroup",
        r"\tiny",
        r"\setlength{\tabcolsep}{2.1pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{p{0.19\textwidth}p{0.19\textwidth}p{0.31\textwidth}p{0.12\textwidth}p{0.13\textwidth}}",
        r"\hline",
        r"\textbf{Design family} & \textbf{Exposure variant} & \textbf{First-stage metric} & \textbf{Strength/sign} & \textbf{Sample} \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                _latex_escape(row[col])
                for col in ["design_family", "exposure_variant", "first_stage_metric", "strength_sign", "sample"]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            rf"\\[-0.2em]{{\tiny Shift-share rows use PPML on master's OPT hires with {RICH_FE_LABEL}. Direct-exposure row reports matched common-break effects on any OPT hire, relative to 2014.}}",
            r"\endgroup",
        ]
    )
    _write_text(out_dir / "first_stage_design_comparison_table.tex", "\n".join(lines))


def build_first_stage_tables(panel: pd.DataFrame, out_dir: Path) -> None:
    title_map = {
        "no_fe": "No FE",
        "firm_year_fe": "Firm + year FE",
        RICH_FE: "Rich FE",
    }
    static = _read_table(IHMP_VARIANT_ROOT / "first_stage_variant_static_summary.csv")
    cols = _static_summary_to_columns(static, title_map)
    _write_econ_table(
        out_dir / "first_stage_ppml_ihmp_share_table.tex",
        "Master's OPT hires",
        r"$z_{ct}$: IHMP share level",
        cols,
        "PPML estimates. The shock is $g_{kt}=$ IHMP share in levels, aggregated with pre-period firm-school shares.",
    )

    for lhs, dep, filename in [
        ("x_asinh", "IHS master's OPT hires", "first_stage_continuous_ols_ihmp_share_table.tex"),
        ("x_bin", "1[any master's OPT hire]", "first_stage_binary_ols_ihmp_share_table.tex"),
    ]:
        rows: list[dict[str, object]] = []
        for fe_key, title, fe_cols in FE_SPECS:
            try:
                _fit, work, coef, se, f_stat = _fit_one(panel, lhs, Z_COL, "ols", fe_cols)
                nobs = len(work)
                mean = float(_safe_num(work[lhs]).mean())
            except Exception as exc:
                print(f"[laborlunch] {filename} {fe_key} failed: {exc}")
                coef, se, f_stat, nobs, mean = np.nan, np.nan, np.nan, np.nan, np.nan
            rows.append(
                {
                    "title": title if fe_key != RICH_FE else "Rich FE",
                    "coef": coef,
                    "se": se,
                    "f": f_stat,
                    "n": nobs,
                    "mean": mean,
                    "firm_fe": "Yes" if fe_key != "no_fe" else "No",
                    "year_fe": "Yes" if fe_key != "no_fe" else "No",
                    "rich_fe": "Yes" if fe_key == RICH_FE else "No",
                }
            )
        _write_econ_table(
            out_dir / filename,
            dep,
            r"$z_{ct}$: IHMP share level",
            rows,
            "OLS estimates with firm-clustered standard errors.",
        )


def build_reduced_form_table(panel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    specs = [
        ("IHS employment", "y_cst_lag0", "ihs_y_cst_lag0", "ihs"),
        ("IHS new hires", "y_new_hires_lag0", "ihs_y_new_hires_lag0", "ihs"),
        ("IHS foreign new hires", "y_new_hires_foreign_lag0", "ihs_y_new_hires_foreign_lag0", "ihs"),
        ("IHS native new hires", "y_new_hires_native_lag0", "ihs_y_new_hires_native_lag0", "ihs"),
        ("Avg tenure", "avg_tenure_years_lag0", "avg_tenure_years_lag0", "level"),
    ]
    rows: list[dict[str, object]] = []
    rich_cols = [cols for key, _label, cols in FE_SPECS if key == RICH_FE][0]
    for label, source, lhs, transform in specs:
        if source not in panel.columns:
            rows.append({"title": label})
            continue
        work_panel = panel.copy()
        work_panel[source] = _safe_num(work_panel[source])
        if transform == "ihs":
            work_panel[lhs] = np.arcsinh(work_panel[source].clip(lower=0.0))
        try:
            mean = float(_safe_num(work_panel[source]).mean())
            _fit, work, coef, se, f_stat = _fit_one(work_panel, lhs, Z_COL, "ols", rich_cols)
            nobs = len(work)
        except Exception as exc:
            print(f"[laborlunch] RF {label} failed: {exc}")
            coef, se, f_stat, nobs, mean = np.nan, np.nan, np.nan, np.nan, np.nan
        rows.append({"title": label, "coef": coef, "se": se, "f": f_stat, "n": nobs, "mean": mean})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "reduced_form_ihmp_share_table.csv", index=False)

    columns = [
        {
            "title": r["title"],
            "coef": r["coef"],
            "se": r["se"],
            "f": r["f"],
            "n": r["n"],
            "mean": r["mean"],
            "firm_fe": "Yes",
            "year_fe": "Yes",
            "rich_fe": "Yes",
        }
        for r in rows
    ]
    _write_econ_table(
        out_dir / "reduced_form_ihmp_share_table.tex",
        "Column outcome",
        r"$z_{ct}$: IHMP share level",
        columns,
        f"All columns use {RICH_FE_LABEL.lower()} and firm-clustered standard errors. Count outcomes are IHS-transformed.",
    )
    return table


def _plot_dynamic_series(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    y_label: str,
    group_col: str | None = None,
    label_col: str | None = None,
) -> None:
    plot = df.loc[df["coef"].notna() & df["se"].notna()].copy()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    if group_col and group_col in plot.columns:
        groups = [(key, sub.sort_values("horizon")) for key, sub in plot.groupby(group_col, sort=False)]
        group_offsets = {
            key: float(offset)
            for (key, _sub), offset in zip(groups, llstyle.offsets(len(groups)))
        }
        for idx, (_key, sub) in enumerate(groups):
            sub = sub.sort_values("horizon")
            label = str(sub[label_col].iloc[0] if label_col and label_col in sub.columns else _key)
            color = llstyle.color(idx)
            ax.errorbar(
                sub["horizon"].astype(float) + group_offsets.get(_key, 0.0),
                sub["coef"],
                yerr=1.96 * sub["se"].fillna(0.0),
                marker="o",
                linewidth=1.5,
                capsize=0,
                markersize=llstyle.MARKER_SIZE,
                color=color,
                ecolor=llstyle.rgba(color),
                elinewidth=llstyle.MARKER_SIZE,
                label=label,
            )
        llstyle.right_legend(ax)
    else:
        plot = plot.sort_values("horizon")
        ax.errorbar(
            plot["horizon"],
            plot["coef"],
            yerr=1.96 * plot["se"].fillna(0.0),
            marker="o",
            linewidth=1.5,
            capsize=0,
            markersize=llstyle.MARKER_SIZE,
            color=llstyle.color(2),
            ecolor=llstyle.rgba(llstyle.color(2)),
            elinewidth=llstyle.MARKER_SIZE,
        )
    ax.axhline(0, color="0.25", linestyle=":", linewidth=1.1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Outcome horizon relative to exposure year")
    ax.set_ylabel(y_label)
    ax.set_title("")
    _savefig(fig, out_path)


def build_binscatters(panel: pd.DataFrame, out_dir: Path) -> None:
    scatter_dir = out_dir / "residualized_binscatter"
    event_groups = _zct_event_group_assignments(panel)
    fig, axes = plt.subplots(1, 2, figsize=llstyle.PANEL_FIGSIZE)
    counts = event_groups["zct_exposure_group"].value_counts().reindex(ZCT_ES_GROUP_ORDER).fillna(0)
    axes[0].bar(
        range(len(counts)),
        counts.values,
        color=[ZCT_ES_COLORS.get(label, llstyle.color(idx)) for idx, label in enumerate(counts.index)],
    )
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(["Zero", "Low +", "Mid +", "High +"], rotation=0)
    axes[0].set_ylabel("Firms")
    axes[0].set_title("")
    for idx, value in enumerate(counts.values):
        axes[0].text(idx, value, f"{int(value):,}", ha="center", va="bottom", fontsize=10)

    z_pos = event_groups.loc[
        event_groups["zct_exposure_group"].astype(str).ne(ZCT_ES_GROUP_ORDER[0]),
        "zct_event_exposure",
    ].dropna()
    axes[1].hist(z_pos, bins=40, color=llstyle.color(2), alpha=0.88)
    axes[1].set_xlabel(r"$z_{ct}$ among positive-exposure firms")
    axes[1].set_ylabel("Firms")
    axes[1].set_title("")
    _savefig(fig, out_dir / "dist_z_ct.png")

    _plot_residualized_binscatter(
        panel,
        Z_COL,
        "masters_opt_hires_correction_aware",
        "Residualized IHMP-share exposure",
        "Residualized master's OPT hires",
        "First stage, rich FE",
        scatter_dir / "ihmp_share_first_stage_rich_binscatter.png",
        scatter_dir / "ihmp_share_first_stage_rich_binscatter.csv",
    )
    for source, lhs, label, stem in [
        ("y_cst_lag0", "ihs_y_cst_lag0", "IHS employment", "employment"),
        ("y_new_hires_lag0", "ihs_y_new_hires_lag0", "IHS new hires", "new_hires"),
    ]:
        if source not in panel.columns:
            continue
        panel[lhs] = np.arcsinh(_safe_num(panel[source]).clip(lower=0.0))
        _plot_residualized_binscatter(
            panel,
            Z_COL,
            lhs,
            "Residualized IHMP-share exposure",
            f"Residualized {label}",
            f"Reduced form: {label}, rich FE",
            scatter_dir / f"ihmp_share_rf_{stem}_rich_binscatter.png",
            scatter_dir / f"ihmp_share_rf_{stem}_rich_binscatter.csv",
        )


def _ensure_zct_event_outcome(panel: pd.DataFrame, outcome_col: str) -> bool:
    if outcome_col in panel.columns:
        panel[outcome_col] = _safe_num(panel[outcome_col])
        return True
    if outcome_col.startswith("log1p_"):
        source_col = outcome_col.removeprefix("log1p_")
        if source_col in panel.columns:
            panel[outcome_col] = np.log1p(_safe_num(panel[source_col]).clip(lower=0.0))
            return True
    return False


def _zct_event_group_assignments(panel: pd.DataFrame) -> pd.DataFrame:
    event_scores = (
        panel.loc[panel["t_num"].eq(ZCT_ES_EVENT_YEAR), ["c", Z_COL]]
        .dropna(subset=["c", Z_COL])
        .drop_duplicates("c")
        .copy()
    )
    event_scores[Z_COL] = _safe_num(event_scores[Z_COL]).fillna(0.0)
    event_scores["zct_exposure_group"] = ZCT_ES_GROUP_ORDER[0]
    pos = event_scores[Z_COL].gt(0)
    if pos.any():
        ranks = event_scores.loc[pos, Z_COL].rank(method="first", pct=True)
        bins = pd.cut(
            ranks,
            bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            labels=ZCT_ES_GROUP_ORDER[1:],
            include_lowest=True,
        )
        event_scores.loc[pos, "zct_exposure_group"] = bins.astype(str)
    event_scores["zct_exposure_group"] = pd.Categorical(
        event_scores["zct_exposure_group"],
        categories=ZCT_ES_GROUP_ORDER,
        ordered=True,
    )
    return event_scores[["c", Z_COL, "zct_exposure_group"]].rename(columns={Z_COL: "zct_event_exposure"})


def _zct_event_panel(panel: pd.DataFrame, outcome_cols: list[str]) -> pd.DataFrame:
    work = panel.loc[panel["t_num"].between(ZCT_ES_START_YEAR, ZCT_ES_END_YEAR)].copy()
    for outcome_col in outcome_cols:
        _ensure_zct_event_outcome(work, outcome_col)
    groups = _zct_event_group_assignments(panel)
    work = work.merge(groups, on="c", how="inner")
    work["t_year"] = pd.to_numeric(work["t_num"], errors="coerce").astype("Int64")
    work["t_fe"] = work["t_year"].astype("string")
    work["c"] = work["c"].astype(str)
    return work


def _plot_zct_raw_means(
    event_panel: pd.DataFrame,
    outcome_col: str,
    out_path: Path,
    title: str,
    *,
    ax: plt.Axes | None = None,
) -> None:
    own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    else:
        fig = ax.figure
    stats = (
        event_panel.dropna(subset=[outcome_col, "zct_exposure_group", "t_year"])
        .groupby(["zct_exposure_group", "t_year"], observed=True)[outcome_col]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    stats["se"] = stats["sd"] / np.sqrt(stats["n"].clip(lower=1))
    groups = [group for group in ZCT_ES_GROUP_ORDER if not stats.loc[stats["zct_exposure_group"].astype(str).eq(group)].empty]
    group_offsets = {group: float(offset) for group, offset in zip(groups, llstyle.offsets(len(groups)))}
    for idx, group in enumerate(groups):
        sub = stats.loc[stats["zct_exposure_group"].astype(str).eq(group)].sort_values("t_year")
        if sub.empty:
            continue
        n_firms = event_panel.loc[event_panel["zct_exposure_group"].astype(str).eq(group), "c"].nunique()
        color = ZCT_ES_COLORS.get(group) or llstyle.color(idx)
        ax.errorbar(
            sub["t_year"].astype(float) + group_offsets.get(group, 0.0),
            sub["mean"],
            yerr=1.96 * sub["se"].fillna(0.0),
            marker="o",
            linewidth=1.5,
            capsize=0,
            markersize=llstyle.MARKER_SIZE,
            color=color,
            ecolor=llstyle.rgba(color),
            elinewidth=llstyle.MARKER_SIZE,
            label=f"{group} (n={n_firms:,})",
        )
    ax.axvline(ZCT_ES_EVENT_YEAR, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Year")
    ax.set_ylabel(ZCT_ES_OUTCOME_LABELS.get(outcome_col, outcome_col))
    ax.set_title("")
    llstyle.right_legend(ax)
    if own_fig:
        _savefig(fig, out_path)


def _run_zct_high_low_regression(event_panel: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    import pyfixest as pf

    keep_groups = [ZCT_ES_GROUP_ORDER[0], ZCT_ES_GROUP_ORDER[-1]]
    work = event_panel.loc[
        event_panel["zct_exposure_group"].astype(str).isin(keep_groups),
        ["c", "t_fe", "t_year", "zct_exposure_group", outcome_col],
    ].dropna(subset=["c", "t_fe", "t_year", "zct_exposure_group", outcome_col]).copy()
    work[outcome_col] = _safe_num(work[outcome_col])
    work["zct_high"] = work["zct_exposure_group"].astype(str).eq(ZCT_ES_GROUP_ORDER[-1]).astype(float)
    years = sorted(int(y) for y in work["t_year"].dropna().unique())
    interaction_cols: list[str] = []
    for year in years:
        if year == ZCT_ES_REF_YEAR:
            continue
        col = f"es_y{year}_high"
        work[col] = work["zct_high"] * work["t_year"].astype(int).eq(year).astype(float)
        if work[col].std() > 0:
            interaction_cols.append(col)
    if not interaction_cols:
        return pd.DataFrame()
    formula = f"{outcome_col} ~ {' + '.join(interaction_cols)} | c + t_fe"
    fit = pf.feols(formula, data=work, vcov={"CRV1": "c"})
    coefs = fit.coef()
    ses = fit.se()
    rows = [{"year": ZCT_ES_REF_YEAR, "coef": 0.0, "se": 0.0}]
    for col in interaction_cols:
        year = int(col.removeprefix("es_y").removesuffix("_high"))
        if col in coefs.index:
            rows.append({"year": year, "coef": float(coefs.loc[col]), "se": float(ses.loc[col])})
    return pd.DataFrame(rows).sort_values("year")


def _draw_zct_regression(
    ax: plt.Axes,
    coef_df: pd.DataFrame,
    outcome_col: str,
    title: str,
    *,
    show_ylabel: bool = True,
) -> None:
    if coef_df.empty:
        ax.text(0.5, 0.5, "No estimable coefficients", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    plot = coef_df.sort_values("year")
    ax.errorbar(
        plot["year"],
        plot["coef"],
        yerr=1.96 * plot["se"].fillna(0.0),
        marker="o",
        linewidth=1.5,
        capsize=0,
        markersize=llstyle.MARKER_SIZE,
        color=llstyle.color(1),
        ecolor=llstyle.rgba(llstyle.color(1)),
        elinewidth=llstyle.MARKER_SIZE,
    )
    ax.axhline(0, color="0.25", linestyle=":", linewidth=1.1)
    ax.axvline(ZCT_ES_EVENT_YEAR, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Year")
    if show_ylabel:
        ax.set_ylabel(f"High vs zero exposure, ref. {ZCT_ES_REF_YEAR}")
    ax.set_title("")


def _plot_zct_regression(coef_df: pd.DataFrame, outcome_col: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    _draw_zct_regression(ax, coef_df, outcome_col, title)
    _savefig(fig, out_path)


def _plot_zct_combined_regression(
    coef_by_outcome: dict[str, pd.DataFrame],
    outcomes: list[tuple[str, str]],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(outcomes), figsize=llstyle.PANEL_FIGSIZE, sharey=False)
    if len(outcomes) == 1:
        axes = [axes]  # type: ignore[assignment]
    for idx, (ax, (outcome_col, panel_title)) in enumerate(zip(axes, outcomes)):
        _draw_zct_regression(
            ax,
            coef_by_outcome.get(outcome_col, pd.DataFrame()),
            outcome_col,
            panel_title,
            show_ylabel=idx == 0,
        )
    _savefig(fig, out_path)


def build_zct_exposure_event_study_figures(panel: pd.DataFrame, out_dir: Path) -> None:
    outcomes = [
        "any_opt_hires_correction_aware",
        "log1p_y_cst_lag0",
        "log1p_y_new_hires_lag0",
        "log1p_y_new_hires_foreign_lag0",
        "log1p_y_new_hires_native_lag0",
        "log1p_y_cst_foreign_lag0",
        "log1p_y_cst_native_lag0",
    ]
    event_panel = _zct_event_panel(panel, outcomes)
    available = [outcome for outcome in outcomes if outcome in event_panel.columns]
    coef_by_outcome: dict[str, pd.DataFrame] = {}
    for outcome_col in available:
        label = ZCT_ES_OUTCOME_LABELS.get(outcome_col, outcome_col)
        stem = f"zct_exposure_es_{outcome_col}"
        _plot_zct_raw_means(
            event_panel,
            outcome_col,
            out_dir / f"zct_exposure_es_raw_{outcome_col}.png",
            f"{label} by $z_{{ct}}$ exposure group",
        )
        try:
            coef_df = _run_zct_high_low_regression(event_panel, outcome_col)
        except Exception as exc:
            print(f"[laborlunch] z_ct event-study regression failed for {outcome_col}: {exc}")
            coef_df = pd.DataFrame()
        coef_by_outcome[outcome_col] = coef_df
        if not coef_df.empty:
            coef_df.assign(outcome=outcome_col).to_csv(out_dir / f"{stem}.csv", index=False)
        _plot_zct_regression(
            coef_df,
            outcome_col,
            out_dir / f"zct_exposure_es_reg_{outcome_col}.png",
            f"{label}: high versus zero $z_{{ct}}$ exposure",
        )

    _plot_zct_combined_regression(
        coef_by_outcome,
        [
            ("log1p_y_new_hires_foreign_lag0", "Foreign new hires"),
            ("log1p_y_new_hires_native_lag0", "Native new hires"),
        ],
        out_dir / "zct_exposure_es_reg_foreign_native_new_hires.png",
        "Hiring response by worker origin",
    )


def build_dynamic_figures(out_dir: Path) -> None:
    dyn_ihmp = _read_table(IHMP_VARIANT_ROOT / "first_stage_variant_dynamic_summary.csv")
    if not dyn_ihmp.empty:
        sub = dyn_ihmp.loc[dyn_ihmp["fe_spec"].eq(RICH_FE)].copy()
        _plot_dynamic_series(
            sub,
            out_dir / "dynamic_first_stage_ihmp_share_rich.png",
            "Dynamic first stage: IHMP-share levels",
            "Coefficient on IHMP-share exposure",
        )

    dyn_methods = _read_table(VARIANT_ROOT / "first_stage_variant_dynamic_summary.csv")
    method_labels = {
        "first_diff": "First difference",
        "ar_residual": "AR residual",
        "common_base_level": "Common-base level",
        "event_pulse": "Event pulse",
    }
    if not dyn_methods.empty:
        all_methods = dyn_methods.loc[
            dyn_methods["variant"].isin(method_labels) & dyn_methods["fe_spec"].eq(RICH_FE)
        ].copy()
        all_methods["method_label"] = all_methods["variant"].map(method_labels)
        _plot_dynamic_series(
            all_methods,
            out_dir / "dynamic_first_stage_serial_methods_comparison.png",
            "Dynamic first stage: serial-correlation adjustments",
            "Coefficient on adjusted exposure",
            group_col="variant",
            label_col="method_label",
        )
        for key, label in method_labels.items():
            sub = all_methods.loc[all_methods["variant"].eq(key)]
            _plot_dynamic_series(
                sub,
                out_dir / f"dynamic_first_stage_{key}.png",
                f"Dynamic first stage: {label}",
                "Coefficient on adjusted exposure",
            )

    _build_dynamic_reduced_form_panels(out_dir)


def _read_dynamic_rf(outcome: str) -> pd.DataFrame:
    ar_path = (
        CODE_ROOT
        / "output"
        / "company_shift_share"
        / f"reduced_form_ar_resid_{outcome}"
        / "reduced_form_variant_dynamic_summary.csv"
    )
    dyn_ar = _read_table(ar_path)
    if not dyn_ar.empty:
        dyn_ar = dyn_ar.loc[
            dyn_ar.get("equation", "").eq("reduced_form") & dyn_ar.get("fe_spec", "").eq(RICH_FE)
        ].copy()
        if not dyn_ar.empty:
            dyn_ar["family"] = "reduced_form"
            dyn_ar["serial_correction"] = "AR residual"
            return dyn_ar

    if outcome == "y_cst_lag0":
        path = APR_ROOT / "dynamic_effect_coefficients.csv"
    else:
        path = APR_ROOT / "outcome_variants" / outcome / "dynamic_effect_coefficients.csv"
    dyn = _read_table(path)
    if dyn.empty:
        return dyn
    dyn["serial_correction"] = "Raw exposure"
    return dyn.loc[dyn["family"].eq("reduced_form")].copy()


def _build_dynamic_reduced_form_panels(out_dir: Path) -> None:
    outcome_specs = [
        ("y_cst_lag0", "Employment", llstyle.color(2), "o"),
        ("y_new_hires_lag0", "New hires", llstyle.color(1), "s"),
        ("y_new_hires_foreign_lag0", "Foreign new hires", llstyle.color(0), "o"),
        ("y_new_hires_native_lag0", "Native new hires", llstyle.color(3), "s"),
        ("avg_tenure_years_lag0", "Avg tenure", llstyle.color(2), "o"),
    ]
    frames: list[pd.DataFrame] = []
    for outcome, label, _color, _marker in outcome_specs:
        dyn = _read_dynamic_rf(outcome)
        if dyn.empty:
            continue
        dyn["outcome_label"] = label
        dyn["outcome"] = outcome
        frames.append(dyn)
    if not frames:
        return
    all_dyn = pd.concat(frames, ignore_index=True)
    all_dyn.to_csv(out_dir / "dynamic_reduced_form_outcomes.csv", index=False)
    panels = [
        ("dynamic_rf_employment_new_hires.png", ["Employment", "New hires"], "Dynamic reduced form: scale margins, AR-residual shock"),
        (
            "dynamic_rf_foreign_native_new_hires.png",
            ["Foreign new hires", "Native new hires"],
            "Dynamic reduced form: hiring composition, AR-residual shock",
        ),
        ("dynamic_rf_avg_tenure.png", ["Avg tenure"], "Dynamic reduced form: average tenure, AR-residual shock"),
    ]
    colors = {label: llstyle.color(idx) for idx, (_outcome, label, _color, _marker) in enumerate(outcome_specs)}
    markers = {label: marker for _outcome, label, _color, marker in outcome_specs}
    for filename, labels, title in panels:
        plot = all_dyn.loc[all_dyn["outcome_label"].isin(labels)].copy()
        if plot.empty:
            continue
        fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
        label_offsets = {label: float(offset) for label, offset in zip(labels, llstyle.offsets(len(labels)))}
        for label in labels:
            sub = plot.loc[plot["outcome_label"].eq(label)].sort_values("horizon")
            if sub.empty:
                continue
            color = colors.get(label) or llstyle.color(0)
            ax.errorbar(
                sub["horizon"].astype(float) + label_offsets.get(label, 0.0),
                sub["coef"],
                yerr=1.96 * sub["se"].fillna(0.0),
                marker=markers.get(label, "o"),
                linewidth=1.5,
                capsize=0,
                markersize=llstyle.MARKER_SIZE,
                color=color,
                ecolor=llstyle.rgba(color),
                elinewidth=llstyle.MARKER_SIZE,
                label=label,
            )
        ax.axhline(0, color="0.25", linestyle=":", linewidth=1.1)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.1)
        ax.set_xlabel("Outcome horizon relative to exposure year")
        ax.set_ylabel("Reduced-form coefficient")
        ax.set_title("")
        llstyle.right_legend(ax)
        _savefig(fig, out_dir / filename)


def build_robustness_figures(out_dir: Path) -> None:
    influence = _read_table(DIAG_ROOT / "school_influence_diagnostics.csv")
    if not influence.empty and "rotemberg_proxy_abs_share" in influence.columns:
        plot = influence.sort_values("rotemberg_proxy_abs_share", ascending=False).head(15).copy()
        plot["school_name"] = plot["school_name"].fillna(plot["k"].astype(str))
        fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
        ax.barh(plot["school_name"][::-1], 100 * _safe_num(plot["rotemberg_proxy_abs_share"])[::-1], color=llstyle.color(2))
        ax.set_xlabel("Rotemberg-style absolute weight (%)")
        ax.set_title("")
        _savefig(fig, out_dir / "top_schools_rotemberg_weights.png")

    balance = _read_table(DIAG_ROOT / "school_pretrend_balance.csv")
    if not balance.empty and {"delta_share_event", "metric_share_pre_slope"}.issubset(balance.columns):
        work = balance.dropna(subset=["delta_share_event", "metric_share_pre_slope"]).copy()
        fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
        ax.scatter(work["delta_share_event"], work["metric_share_pre_slope"], s=llstyle.marker_area(5), alpha=0.45, color=llstyle.NEUTRAL)
        if len(work) > 2:
            x = _safe_num(work["delta_share_event"]).to_numpy()
            y = _safe_num(work["metric_share_pre_slope"]).to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 2:
                b, a = np.polyfit(x[ok], y[ok], 1)
                xs = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 100)
                ax.plot(xs, a + b * xs, color=llstyle.color(1), linewidth=1.8)
        ax.axhline(0, color="0.25", linestyle=":", linewidth=1.1)
        ax.set_xlabel("Event-window IHMP-share shock")
        ax.set_ylabel("Pre-period IHMP-share slope")
        ax.set_title("")
        _savefig(fig, out_dir / "bhj_shock_balance_pretrend.png")

    shock_diag = _read_table(DIAG_ROOT / "shock_level_diagnostics_by_variant.csv")
    if not shock_diag.empty:
        keep = ["z_ct", "z_ct_flow_diff", "z_ct_flow_ar_resid", "z_ct_common_base_level", "z_ct_event_pulse"]
        plot = shock_diag.loc[shock_diag["instrument_variant"].isin(keep)].copy()
        if not plot.empty:
            fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
            ax.bar(
                plot["instrument_variant"],
                _safe_num(plot["effective_schools_by_abs_component_mass"]),
                color=llstyle.color(0),
            )
            ax.tick_params(axis="x", rotation=22)
            ax.set_ylabel("Effective schools")
            ax.set_title("")
            _savefig(fig, out_dir / "shock_level_effective_schools.png")

    for src, dst in [
        (DIAG_ROOT / "shock_decomposition_scatter.png", out_dir / "shock_decomposition_scatter.png"),
        (DIAG_ROOT / "firm_instrument_concentration.png", out_dir / "firm_instrument_concentration.png"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)


def write_main_tex(out_dir: Path) -> None:
    text = r"""
\section{Firm-Level Effects: Shift-Share Design}

\begin{frame}[t,shrink=7]
    \label{firm_comparable_first_stage_table}
    \frametitle{Comparable First Stages Across Firm-Level Designs}
    \centering
    \IfFileExists{\companyoutput/slides_20260507_shift_share/first_stage_design_comparison_table.tex}{
        \input{\companyoutput/slides_20260507_shift_share/first_stage_design_comparison_table.tex}
    }{
        \scriptsize First-stage comparison table not found. Run \texttt{python -m company_shift_share.laborlunch_shift_share_results}.
    }

    \vspace{0.25em}
    \hyperlink{firm_first_stage_design_visuals}{\beamergotobutton{Visual Checks}}
    \hyperlink{firm_first_stage_ppml_table}{\beamergotobutton{School-Shift Table}}
\end{frame}

\begin{frame}
    \label{firm_first_stage_design_visuals}
    \frametitle{First-Stage Visual Checks by Design Family}
    \begin{columns}[T,onlytextwidth]
        \begin{column}{0.49\textwidth}
            \centering
            {\scriptsize Timing/comparison variants}

            \vspace{0.15em}
            \maybegraphic{\linewidth}{\companyoutput/slides_20260507_shift_share/dynamic_first_stage_serial_methods_comparison.png}{Insert serial-correlation adjustment comparison.}
        \end{column}
        \begin{column}{0.49\textwidth}
            \centering
            {\scriptsize Direct firm-exposure common break}

            \vspace{0.15em}
            \maybegraphic{\linewidth}{__MATCHED_EXPOSURE_FIGURE__}{Insert direct firm-exposure common-break first stage.}
        \end{column}
    \end{columns}

    \vspace{0.15em}
    {\tiny Both panels show first-stage evidence only: OPT-hiring outcomes against exposure, with later reduced-form interpretation separated from this roadmap.}
\end{frame}

\begin{frame}
    \label{firm_shift_share_empirical_strategy}
    \frametitle{Empirical Strategy: Firms Exposed to IHMP-Share Growth}
    \[
        z_{ct} = \sum_k s_{ck}^{pre} g_{kt}, \qquad g_{kt}=\text{IHMP share}_{kt}
    \]
    \[
        Y_{ct} = \alpha_c + \lambda_t + \lambda_{t \times q(c) \times r(c)}
        + \beta z_{ct} + \varepsilon_{ct}
    \]
    \begin{itemize}
        \item $s_{ck}^{pre}$ is firm $c$'s pre-period hiring share from school $k$.
        \item The main shock is the school IHMP share in levels, not a count flow.
        \item The richest specification absorbs firm, year, and year $\times$ baseline size $\times$ baseline growth fixed effects.
        \item Firm-clustered standard errors are shown in the static tables.
    \end{itemize}
    \hyperlink{app_firm_zct_histogram}{\beamergotobutton{Histogram of $z_{ct}$}}
\end{frame}

\begin{frame}[t,shrink=8]
    \label{firm_first_stage_ppml_table}
    \frametitle{First Stage: PPML on Continuous OPT Hires}
    \centering
    \IfFileExists{\companyoutput/slides_20260507_shift_share/first_stage_ppml_ihmp_share_table.tex}{
        \input{\companyoutput/slides_20260507_shift_share/first_stage_ppml_ihmp_share_table.tex}
    }{
        \scriptsize First-stage table not found. Run \texttt{python -m company_shift_share.laborlunch_shift_share_results}.
    }

    \vspace{0.25em}
    \hyperlink{app_firm_fs_continuous_ols}{\beamergotobutton{Continuous OLS}}
    \hyperlink{app_firm_fs_binary_ols}{\beamergotobutton{0--1 Binned}}
    \hyperlink{app_firm_fs_binplot}{\beamergotobutton{Binplot}}
\end{frame}

\begin{frame}[t,shrink=8]
    \label{firm_reduced_form_table}
    \frametitle{Reduced Form: Richest FE Specification Across Outcomes}
    \centering
    \IfFileExists{\companyoutput/slides_20260507_shift_share/reduced_form_ihmp_share_table.tex}{
        \input{\companyoutput/slides_20260507_shift_share/reduced_form_ihmp_share_table.tex}
    }{
        \scriptsize Reduced-form table not found. Run \texttt{python -m company_shift_share.laborlunch_shift_share_results}.
    }

    \vspace{0.25em}
    \hyperlink{app_firm_rf_binplot}{\beamergotobutton{Regression Binplot}}
\end{frame}

\begin{frame}
    \label{firm_dynamic_first_stage}
    \frametitle{Dynamic First Stage: Persistent Exposure Creates Pretrends}
    \centering
    \maybegraphic{0.82\textwidth}{\companyoutput/slides_20260507_shift_share/dynamic_first_stage_ihmp_share_rich.png}{Insert dynamic first-stage plot.}

    \vspace{0.25em}
    {\scriptsize The pre-period coefficients are visibly nonzero because school IHMP-share shocks are serially correlated.}
\end{frame}

\begin{frame}
    \label{firm_dynamic_first_stage_serial_methods}
    \frametitle{Addressing Serial Correlation in the First Stage}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/dynamic_first_stage_serial_methods_comparison.png}{Insert serial-correlation adjustment comparison.}

    \vspace{0.25em}
    \hyperlink{app_firm_dynamic_first_diff}{\beamergotobutton{First Difference}}
    \hyperlink{app_firm_dynamic_ar_resid}{\beamergotobutton{AR Residual}}
    \hyperlink{app_firm_dynamic_common_base}{\beamergotobutton{Common Base}}
    \hyperlink{app_firm_dynamic_event_pulse}{\beamergotobutton{Event Pulse}}
\end{frame}

\begin{frame}
    \label{firm_dynamic_reduced_form_scale}
    \frametitle{Dynamic Reduced Form: Employment and New Hires}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/dynamic_rf_employment_new_hires.png}{Insert dynamic reduced form for employment and new hires.}

    \vspace{0.25em}
    \hyperlink{app_firm_dynamic_rf_binplots}{\beamergotobutton{RF Binplots}}
    \hyperlink{app_firm_dynamic_serial_note}{\beamergotobutton{Serial-Correlation Note}}
\end{frame}

\begin{frame}
    \label{firm_dynamic_reduced_form_hiring}
    \frametitle{Dynamic Reduced Form: Foreign and Native New Hires}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/dynamic_rf_foreign_native_new_hires.png}{Insert dynamic reduced form for foreign and native new hires.}
\end{frame}

\begin{frame}
    \label{firm_dynamic_reduced_form_tenure}
    \frametitle{Dynamic Reduced Form: Average Tenure}
    \centering
    \maybegraphic{0.82\textwidth}{\companyoutput/slides_20260507_shift_share/dynamic_rf_avg_tenure.png}{Insert dynamic reduced form for average tenure.}
\end{frame}

\begin{frame}
    \label{firm_zct_event_opt_hires}
    \frametitle{Firms With Higher $z_{ct}$ Exposure Hire More OPT Workers}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_reg_any_opt_hires_correction_aware.png}{Insert $z_{ct}$ exposure event-study plot for OPT hires.}

    \vspace{0.25em}
    \hyperlink{app_firm_zct_es_raw_any_opt_hires}{\beamergotobutton{Raw Means}}
    \hyperlink{app_firm_zct_histogram}{\beamergotobutton{Exposure Distribution}}
\end{frame}

\begin{frame}
    \label{firm_zct_event_employment}
    \frametitle{Higher $z_{ct}$ Exposure Predicts Employment Growth}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_reg_log1p_y_cst_lag0.png}{Insert $z_{ct}$ exposure event-study plot for employment.}

    \vspace{0.25em}
    \hyperlink{app_firm_zct_es_raw_employment}{\beamergotobutton{Raw Means}}
    \hyperlink{app_firm_zct_es_foreign_employment}{\beamergotobutton{Foreign Employment}}
    \hyperlink{app_firm_zct_es_native_employment}{\beamergotobutton{Native Employment}}
\end{frame}

\begin{frame}
    \label{firm_zct_event_hiring_composition}
    \frametitle{Hiring Response Splits by Foreign and Native Hires}
    \centering
    \maybegraphic{0.90\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_reg_foreign_native_new_hires.png}{Insert $z_{ct}$ exposure event-study plot for foreign and native hires.}

    \vspace{0.25em}
    \hyperlink{app_firm_zct_es_total_new_hires}{\beamergotobutton{Total New Hires}}
    \hyperlink{app_firm_zct_es_raw_foreign_new_hires}{\beamergotobutton{Foreign Raw Means}}
    \hyperlink{app_firm_zct_es_raw_native_new_hires}{\beamergotobutton{Native Raw Means}}
\end{frame}

\begin{frame}
    \label{firm_robustness_checks}
    \frametitle{Robustness and Shift-Share Diagnostics}
    \scriptsize
    \begin{itemize}
        \item Rotemberg-style weights identify whether a few schools dominate the identifying variation; the current top-weight plot shows this concentration directly.
        \item BHJ-style shock-level balance asks whether school shocks are correlated with pre-period school trends; the balance plot is the first-pass visual check.
        \item Shock concentration, serial-correlation, placebo timing, and alternative share-window checks are in the appendix.
        \item Takeaway: read robustness by first checking the first stage, then whether signs survive nearby exposure definitions and shock-level diagnostics.
    \end{itemize}
    \vspace{0.25em}
    \hyperlink{app_firm_rotemberg_weights}{\beamergotobutton{Rotemberg Weights}}
    \hyperlink{app_firm_bhj_balance}{\beamergotobutton{BHJ Balance}}
    \hyperlink{app_firm_shock_concentration}{\beamergotobutton{Shock Concentration}}
    \hyperlink{app_firm_serial_correlation}{\beamergotobutton{Serial Correlation}}
\end{frame}
"""
    text = text.replace("__MATCHED_EXPOSURE_FIGURE__", MATCHED_EXPOSURE_FIGURE.as_posix())
    _write_text(out_dir / "firm_level_main.tex", text)


def write_appendix_tex(out_dir: Path) -> None:
    text = r"""
\begin{frame}
    \label{app_firm_zct_histogram}
    \frametitle{Appendix: Histogram of Firm Exposure $z_{ct}$ \hyperlink{firm_shift_share_empirical_strategy}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/dist_z_ct.png}{Insert $z_{ct}$ histogram.}
\end{frame}

\begin{frame}
    \label{app_firm_zct_es_raw_any_opt_hires}
    \frametitle{Appendix: OPT Hires by $z_{ct}$ Exposure Group \hyperlink{firm_zct_event_opt_hires}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_raw_any_opt_hires_correction_aware.png}{Insert raw means for OPT hires by $z_{ct}$ exposure group.}
\end{frame}

\begin{frame}
    \label{app_firm_zct_es_raw_employment}
    \frametitle{Appendix: Employment by $z_{ct}$ Exposure Group \hyperlink{firm_zct_event_employment}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_raw_log1p_y_cst_lag0.png}{Insert raw means for employment by $z_{ct}$ exposure group.}
\end{frame}

\begin{frame}
    \label{app_firm_zct_es_total_new_hires}
    \frametitle{Appendix: Total New Hires, High vs Zero $z_{ct}$ Exposure \hyperlink{firm_zct_event_hiring_composition}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_reg_log1p_y_new_hires_lag0.png}{Insert event-study coefficients for total new hires.}

    \vspace{0.25em}
    \hyperlink{app_firm_zct_es_raw_total_new_hires}{\beamergotobutton{Raw Means}}
\end{frame}

\begin{frame}
    \label{app_firm_zct_es_raw_total_new_hires}
    \frametitle{Appendix: Total New Hires by $z_{ct}$ Exposure Group \hyperlink{app_firm_zct_es_total_new_hires}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_raw_log1p_y_new_hires_lag0.png}{Insert raw means for total new hires.}
\end{frame}

\begin{frame}
    \label{app_firm_zct_es_raw_foreign_new_hires}
    \frametitle{Appendix: Foreign New Hires by $z_{ct}$ Exposure Group \hyperlink{firm_zct_event_hiring_composition}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_raw_log1p_y_new_hires_foreign_lag0.png}{Insert raw means for foreign new hires.}
\end{frame}

\begin{frame}
    \label{app_firm_zct_es_raw_native_new_hires}
    \frametitle{Appendix: Native New Hires by $z_{ct}$ Exposure Group \hyperlink{firm_zct_event_hiring_composition}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_raw_log1p_y_new_hires_native_lag0.png}{Insert raw means for native new hires.}
\end{frame}

\begin{frame}
    \label{app_firm_zct_es_foreign_employment}
    \frametitle{Appendix: Foreign Employment, High vs Zero $z_{ct}$ Exposure \hyperlink{firm_zct_event_employment}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_reg_log1p_y_cst_foreign_lag0.png}{Insert event-study coefficients for foreign employment.}

    \vspace{0.25em}
    \hyperlink{app_firm_zct_es_raw_foreign_employment}{\beamergotobutton{Raw Means}}
\end{frame}

\begin{frame}
    \label{app_firm_zct_es_raw_foreign_employment}
    \frametitle{Appendix: Foreign Employment by $z_{ct}$ Exposure Group \hyperlink{app_firm_zct_es_foreign_employment}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_raw_log1p_y_cst_foreign_lag0.png}{Insert raw means for foreign employment.}
\end{frame}

\begin{frame}
    \label{app_firm_zct_es_native_employment}
    \frametitle{Appendix: Native Employment, High vs Zero $z_{ct}$ Exposure \hyperlink{firm_zct_event_employment}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_reg_log1p_y_cst_native_lag0.png}{Insert event-study coefficients for native employment.}

    \vspace{0.25em}
    \hyperlink{app_firm_zct_es_raw_native_employment}{\beamergotobutton{Raw Means}}
\end{frame}

\begin{frame}
    \label{app_firm_zct_es_raw_native_employment}
    \frametitle{Appendix: Native Employment by $z_{ct}$ Exposure Group \hyperlink{app_firm_zct_es_native_employment}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/zct_exposure_es_raw_log1p_y_cst_native_lag0.png}{Insert raw means for native employment.}
\end{frame}

\begin{frame}[t,shrink=8]
    \label{app_firm_fs_continuous_ols}
    \frametitle{Appendix: First Stage, Continuous OLS \hyperlink{firm_first_stage_ppml_table}{\beamerreturnbutton{Return}}}
    \centering
    \input{\companyoutput/slides_20260507_shift_share/first_stage_continuous_ols_ihmp_share_table.tex}
\end{frame}

\begin{frame}[t,shrink=8]
    \label{app_firm_fs_binary_ols}
    \frametitle{Appendix: First Stage, 0--1 Binned Outcome \hyperlink{firm_first_stage_ppml_table}{\beamerreturnbutton{Return}}}
    \centering
    \input{\companyoutput/slides_20260507_shift_share/first_stage_binary_ols_ihmp_share_table.tex}
\end{frame}

\begin{frame}
    \label{app_firm_fs_binplot}
    \frametitle{Appendix: First-Stage Regression Binplot \hyperlink{firm_first_stage_ppml_table}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/residualized_binscatter/ihmp_share_first_stage_rich_binscatter.png}{Insert first-stage binplot.}
\end{frame}

\begin{frame}
    \label{app_firm_rf_binplot}
    \frametitle{Appendix: Reduced-Form Regression Binplot \hyperlink{firm_reduced_form_table}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/residualized_binscatter/ihmp_share_rf_employment_rich_binscatter.png}{Insert reduced-form binplot.}
\end{frame}

\begin{frame}
    \label{app_firm_dynamic_first_diff}
    \frametitle{Appendix: Dynamic First Stage, First Difference \hyperlink{firm_dynamic_first_stage_serial_methods}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/dynamic_first_stage_first_diff.png}{Insert first-difference dynamic first stage.}
\end{frame}

\begin{frame}
    \label{app_firm_dynamic_ar_resid}
    \frametitle{Appendix: Dynamic First Stage, AR Residual \hyperlink{firm_dynamic_first_stage_serial_methods}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/dynamic_first_stage_ar_residual.png}{Insert AR-residual dynamic first stage.}
\end{frame}

\begin{frame}
    \label{app_firm_dynamic_common_base}
    \frametitle{Appendix: Dynamic First Stage, Common Base \hyperlink{firm_dynamic_first_stage_serial_methods}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/dynamic_first_stage_common_base_level.png}{Insert common-base dynamic first stage.}
\end{frame}

\begin{frame}
    \label{app_firm_dynamic_event_pulse}
    \frametitle{Appendix: Dynamic First Stage, Event Pulse \hyperlink{firm_dynamic_first_stage_serial_methods}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/dynamic_first_stage_event_pulse.png}{Insert event-pulse dynamic first stage.}
\end{frame}

\begin{frame}
    \label{app_firm_dynamic_rf_binplots}
    \frametitle{Appendix: Reduced-Form New-Hires Binplot \hyperlink{firm_dynamic_reduced_form_scale}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/residualized_binscatter/ihmp_share_rf_new_hires_rich_binscatter.png}{Insert new-hires reduced-form binplot.}
\end{frame}

\begin{frame}
    \label{app_firm_dynamic_serial_note}
    \frametitle{Appendix: Serial-Correlation Interpretation \hyperlink{firm_dynamic_reduced_form_scale}{\beamerreturnbutton{Return}}}
    \scriptsize
    \begin{itemize}
        \item The raw IHMP-share exposure is persistent, so dynamic coefficients at negative horizons should not be read as clean anticipation effects.
        \item First-difference and AR-residual first stages are the cleanest timing checks because they remove much of the serially correlated school component.
        \item The AR-residual first stage has the strongest rich-FE first stage among the innovation-style corrections in the saved sweep, so it is the preferred serial-correlation adjustment.
        \item The dynamic reduced-form slides use the AR-residual shock for the five requested outcomes, keeping the richest fixed-effect specification fixed across panels.
    \end{itemize}
\end{frame}

\begin{frame}
    \label{app_firm_rotemberg_weights}
    \frametitle{Appendix: Top Schools by Rotemberg-Style Weights \hyperlink{firm_robustness_checks}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.84\textwidth}{\companyoutput/slides_20260507_shift_share/top_schools_rotemberg_weights.png}{Insert top-school Rotemberg plot.}
\end{frame}

\begin{frame}
    \label{app_firm_bhj_balance}
    \frametitle{Appendix: BHJ-Style Shock-Level Balance \hyperlink{firm_robustness_checks}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.82\textwidth}{\companyoutput/slides_20260507_shift_share/bhj_shock_balance_pretrend.png}{Insert shock-level balance plot.}
\end{frame}

\begin{frame}
    \label{app_firm_shock_concentration}
    \frametitle{Appendix: Shock-Level Concentration \hyperlink{firm_robustness_checks}{\beamerreturnbutton{Return}}}
    \centering
    \maybegraphic{0.82\textwidth}{\companyoutput/slides_20260507_shift_share/shock_level_effective_schools.png}{Insert shock concentration plot.}
\end{frame}

\begin{frame}
    \label{app_firm_serial_correlation}
    \frametitle{Appendix: Serial-Correlation Table \hyperlink{firm_robustness_checks}{\beamerreturnbutton{Return}}}
    \centering
    \IfFileExists{\companyoutput/slides_20260507_shift_share/serial_correlation_table.tex}{
        \input{\companyoutput/slides_20260507_shift_share/serial_correlation_table.tex}
    }{
        \scriptsize Serial-correlation table not found. Run the base deck asset wrapper.
    }
\end{frame}
"""
    _write_text(out_dir / "firm_level_appendix.tex", text)


def build_assets(out_dir: Path = DEFAULT_OUT_DIR) -> None:
    _set_plot_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_ihmp_panel()
    build_first_stage_tables(panel, out_dir)
    build_first_stage_design_comparison(out_dir)
    build_reduced_form_table(panel, out_dir)
    build_binscatters(panel, out_dir)
    build_dynamic_figures(out_dir)
    build_zct_exposure_event_study_figures(panel, out_dir)
    build_robustness_figures(out_dir)
    write_main_tex(out_dir)
    write_appendix_tex(out_dir)
    print(f"[laborlunch] wrote firm-section assets to {out_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build labor-lunch shift-share result slides.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    build_assets(args.out_dir)


if __name__ == "__main__":
    main()
