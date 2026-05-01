"""
Shift-share variant sweep.

Reads the saved April 2026 shift-share panel and runs comparable first-stage or
reduced-form specifications across exposure variants. Outputs coefficient/F-stat
summaries, dynamic coefficients, and per-variant plots.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from company_shift_share.shift_share_analysis import (
    _coef_for_term,
    _model_nobs,
    _prepare_first_stage_state_panel,
    _single_term_f_stat,
)


APR_ROOT = Path("/home/yk0581/data/out/company_shift_share/apr2026")
DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "company_shift_share"
    / "first_stage_variant_sweep_apr2026"
)
X_COL = "masters_opt_hires_correction_aware"
POLY_UNITID = "193900"


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    z_col: str
    estimator: str = "ppml"
    x_col: str = X_COL
    sample_filter: str = "all"
    notes: str = ""


FE_SPECS = [
    ("no_fe", "No FE", []),
    ("firm_year_fe", "Company + year FE", ["c", "t_fe"]),
    (
        "firm_year_size_growth_fe",
        "Company + year + baseline-size-decile x baseline-growth-quintile x year FE",
        ["c", "t_fe", "baseline_size_growth_year_fe"],
    ),
]


VARIANTS = [
    Variant(
        "raw_ppml",
        "Raw annual school g_kt, PPML x",
        "z_raw_flow",
        notes="Former/current broad annual-flow exposure: sum_k share_ck * raw school metric level.",
    ),
    Variant(
        "raw_binary",
        "Raw annual school g_kt, binary x",
        "z_raw_flow",
        estimator="ols_lpm",
        x_col="x_bin",
        notes="Same exposure as raw_ppml; outcome is 1[x_ct > 0].",
    ),
    Variant(
        "ihmp_share",
        "Raw annual IHMP-share g_kt",
        "z_ihmp_share",
        notes="Uses school metric_share instead of metric_level: sum_k share_ck * IHMP share_kt.",
    ),
    Variant(
        "raw_asinh_school_g",
        "ASINH(raw school g_kt), PPML x",
        "z_asinh_raw_g",
        notes="Transforms the school shock before aggregation: sum_k share_ck * asinh(raw g_kt).",
    ),
    Variant(
        "raw_middle50_2008_size",
        "Raw g_kt, middle 50% by 2008 firm size",
        "z_raw_flow",
        sample_filter="middle50_2008_size",
        notes="Keeps firms between the 25th and 75th percentiles of 2008 employment.",
    ),
    Variant(
        "raw_drop_polytechnic",
        "Raw g_kt, drop Polytechnic Institute of New York University",
        "z_raw_drop_polytechnic",
        notes="Recomputes z after removing UNITID 193900 from the shift-share sum.",
    ),
    Variant(
        "matched_step",
        "Matched-school persistent step",
        "z_ct_v4_matched_step",
        notes="Only matched treated/control schools contribute; treated shock is a persistent post-event step.",
    ),
    Variant(
        "matched_pulse",
        "Matched-school event pulse",
        "z_ct_v5_matched_pulse",
        notes="Only matched treated/control schools contribute; treated shock occurs only in the event year.",
    ),
    Variant(
        "matched_pulse_growth_rate",
        "Matched-school event-pulse growth rate",
        "z_matched_pulse_growth_rate",
        notes="Only matched treated/control schools contribute; event-year shock uses (post - pre) / pre.",
    ),
    Variant(
        "first_diff",
        "First-difference school g_kt",
        "z_ct_flow_diff",
        notes="Uses annual first differences in the school metric.",
    ),
    Variant(
        "ar_residual",
        "AR-residual school g_kt",
        "z_ct_flow_ar_resid",
        notes="Uses residualized AR(1)-style school metric innovations.",
    ),
    Variant(
        "event_pulse",
        "Event-pulse school growth rate",
        "z_event_pulse_growth_rate",
        notes="Event-time-specific candidate: shock only in the detected event year, using (post - pre) / pre.",
    ),
    Variant(
        "event_step_broad",
        "Broad event-step dose",
        "z_ct_event_step_dose",
        notes="All event-detected schools contribute a persistent post-event step.",
    ),
    Variant(
        "common_base_level",
        "Common-base level deviation",
        "z_ct_common_base_level",
        notes="School metric level minus 2011-2013 common baseline, then shift-share aggregated.",
    ),
    Variant(
        "full_sample_shares",
        "Raw g_kt with full-sample firm-school shares",
        "z_ct_full",
        notes="Raw school g_kt with firm-school shares computed on the full available transition window.",
    ),
]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")


def _fe_term(fe_cols: list[str]) -> str:
    return "" if not fe_cols else "| " + " + ".join(fe_cols)


def _load_panel(apr_root: Path) -> pd.DataFrame:
    panel = pd.read_parquet(apr_root / "analysis_panel.parquet")
    panel["c"] = panel["c"].astype(str)
    panel["t_num"] = pd.to_numeric(panel["t"], errors="coerce")
    panel["t_fe"] = panel["t_num"].astype("Int64").astype("string")
    panel[X_COL] = pd.to_numeric(panel[X_COL], errors="coerce").fillna(0.0)
    panel["x_bin"] = (panel[X_COL] > 0).astype(float)
    panel = _prepare_first_stage_state_panel(
        panel,
        baseline_window_start=2008,
        baseline_window_end=2013,
        current_size_bins=10,
        current_growth_bins=5,
        joint_size_growth_bins=3,
        baseline_growth_bins=5,
        use_log_y_panel=False,
    )
    for col in ["baseline_size_growth_year_fe", "c", "t_fe"]:
        if col in panel.columns:
            panel[col] = panel[col].astype("string")
    if "z_ct_raw_flow" in panel.columns:
        panel["z_raw_flow"] = pd.to_numeric(panel["z_ct_raw_flow"], errors="coerce")
    elif "z_ct" in panel.columns:
        panel["z_raw_flow"] = pd.to_numeric(panel["z_ct"], errors="coerce")
    return panel


def _read_available_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
        use_cols = [col for col in columns if col in available]
    except Exception:
        use_cols = columns
    return pd.read_parquet(path, columns=use_cols)


def _add_component_built_exposures(panel: pd.DataFrame, apr_root: Path) -> pd.DataFrame:
    needed = [
        "c", "k", "t", "share_ck", "g_kt_raw_flow", "g_kt_event_pulse",
        "metric_share", "event_pre_size", "event_pre_share", "event_level_growth_rate",
        "selected_for_instrument", "z_ct_component_raw_flow",
    ]
    comp = _read_available_parquet_columns(apr_root / "instrument_components.parquet", needed)
    comp["c"] = comp["c"].astype(str)
    comp["t_num"] = pd.to_numeric(comp["t"], errors="coerce")
    comp["share_ck"] = pd.to_numeric(comp["share_ck"], errors="coerce")
    comp["g_kt_raw_flow"] = pd.to_numeric(comp["g_kt_raw_flow"], errors="coerce")
    if "metric_share" in comp.columns:
        comp["metric_share"] = pd.to_numeric(comp["metric_share"], errors="coerce")
    else:
        comp["metric_share"] = np.nan
    comp["g_kt_event_pulse"] = pd.to_numeric(comp["g_kt_event_pulse"], errors="coerce")
    comp["event_pre_size"] = pd.to_numeric(comp["event_pre_size"], errors="coerce")
    comp["event_pre_share"] = pd.to_numeric(comp["event_pre_share"], errors="coerce")
    if "selected_for_instrument" in comp.columns:
        selected_for_instrument = pd.to_numeric(comp["selected_for_instrument"], errors="coerce").fillna(0.0)
    else:
        selected_for_instrument = pd.Series(0.0, index=comp.index)
    comp["z_ct_component_raw_flow"] = pd.to_numeric(comp["z_ct_component_raw_flow"], errors="coerce")
    comp["z_ihmp_share_component"] = comp["share_ck"] * comp["metric_share"].fillna(0.0)
    comp["z_asinh_raw_g_component"] = comp["share_ck"] * np.arcsinh(comp["g_kt_raw_flow"])
    event_pre_level = comp["event_pre_size"] * comp["event_pre_share"]
    if "event_level_growth_rate" in comp.columns:
        event_level_growth_rate = pd.to_numeric(comp["event_level_growth_rate"], errors="coerce")
        comp["g_event_pulse_growth_rate"] = event_level_growth_rate.where(
            pd.to_numeric(comp["g_kt_event_pulse"], errors="coerce").ne(0),
            0.0,
        )
    else:
        comp["g_event_pulse_growth_rate"] = (
            comp["g_kt_event_pulse"] / event_pre_level.where(event_pre_level > 0)
        )
    comp["g_event_pulse_growth_rate"] = (
        comp["g_event_pulse_growth_rate"].replace([float("inf"), float("-inf")], np.nan).fillna(0.0)
    )
    comp["z_event_pulse_growth_rate_component"] = comp["share_ck"] * comp["g_event_pulse_growth_rate"]
    comp["z_matched_pulse_growth_rate_component"] = (
        comp["share_ck"] * comp["g_event_pulse_growth_rate"] * selected_for_instrument.gt(0).astype(float)
    )
    comp["z_raw_drop_polytechnic_component"] = comp["z_ct_component_raw_flow"].where(
        comp["k"].astype(str) != POLY_UNITID,
        0.0,
    )
    built = (
        comp.groupby(["c", "t_num"], as_index=False)
        .agg(
            z_ihmp_share=("z_ihmp_share_component", "sum"),
            z_asinh_raw_g=("z_asinh_raw_g_component", "sum"),
            z_event_pulse_growth_rate=("z_event_pulse_growth_rate_component", "sum"),
            z_matched_pulse_growth_rate=("z_matched_pulse_growth_rate_component", "sum"),
            z_raw_drop_polytechnic=("z_raw_drop_polytechnic_component", "sum"),
        )
    )
    return panel.merge(built, on=["c", "t_num"], how="left")


def _base_analysis_panel(panel: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    work = panel.loc[panel["t_num"].between(start_year, end_year)].copy()
    years = int(work["t_num"].nunique())
    counts = work.groupby("c")["t_num"].nunique()
    keep = counts.loc[counts.eq(years)].index
    work = work.loc[work["c"].isin(keep)].copy()
    return work


def _apply_variant_filter(base: pd.DataFrame, full_panel: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    if variant.sample_filter != "middle50_2008_size":
        return base.copy()
    sizes = (
        full_panel.loc[full_panel["t_num"].eq(2008), ["c", "y_cst_lag0"]]
        .dropna()
        .assign(size_2008=lambda d: pd.to_numeric(d["y_cst_lag0"], errors="coerce"))
        .dropna(subset=["size_2008"])
        .drop_duplicates("c")
    )
    q25, q75 = sizes["size_2008"].quantile([0.25, 0.75])
    keep = set(sizes.loc[sizes["size_2008"].between(q25, q75), "c"])
    return base.loc[base["c"].isin(keep)].copy()


def _fit_one(df: pd.DataFrame, lhs: str, rhs: str, estimator: str, fe_cols: list[str]):
    import pyfixest as pf

    cols = [lhs, rhs, "c", *fe_cols]
    cols = list(dict.fromkeys([c for c in cols if c in df.columns]))
    work = df.loc[df[cols].notna().all(axis=1), cols].copy()
    work[lhs] = pd.to_numeric(work[lhs], errors="coerce")
    work[rhs] = pd.to_numeric(work[rhs], errors="coerce")
    work = work.loc[work[[lhs, rhs]].notna().all(axis=1)].copy()
    if estimator == "ppml":
        work = work.loc[work[lhs].ge(0)].copy()
    if work.empty or work[rhs].nunique(dropna=True) < 2 or work[lhs].nunique(dropna=True) < 2:
        raise ValueError("No estimable variation after filtering.")
    formula = f"{lhs} ~ {rhs} {_fe_term(fe_cols)}"
    fit = (
        pf.fepois(formula, data=work, vcov={"CRV1": "c"})
        if estimator == "ppml"
        else pf.feols(formula, data=work, vcov={"CRV1": "c"})
    )
    coef, se = _coef_for_term(fit, rhs)
    f_stat = _single_term_f_stat(coef, se)
    return fit, work, coef, se, f_stat


def _interpretation(
    coef: float | None,
    z_sd: float,
    z_iqr: float,
    percent_effect: bool,
) -> tuple[float | None, float | None]:
    if coef is None or not math.isfinite(float(coef)):
        return None, None
    if percent_effect:
        return 100.0 * math.expm1(float(coef) * z_sd), 100.0 * math.expm1(float(coef) * z_iqr)
    return 100.0 * float(coef) * z_sd, 100.0 * float(coef) * z_iqr


def _make_dynamic_panel(base: pd.DataFrame, full_panel: pd.DataFrame, lhs_base: str, horizon: int) -> pd.Series:
    lookup = full_panel.loc[:, ["c", "t_num", lhs_base]].copy()
    lookup["target_t"] = pd.to_numeric(lookup["t_num"], errors="coerce")
    lookup = lookup.dropna(subset=["c", "target_t", lhs_base])
    mapper = {
        (str(c), float(t)): v
        for c, t, v in lookup.loc[:, ["c", "target_t", lhs_base]].itertuples(index=False, name=None)
    }
    return pd.Series(
        [
            mapper.get((str(c), float(t) + int(horizon)), np.nan)
            if pd.notna(t) else np.nan
            for c, t in base.loc[:, ["c", "t_num"]].itertuples(index=False, name=None)
        ],
        index=base.index,
    )


def _plot_dynamic(dyn: pd.DataFrame, variant: Variant, out_path: Path, equation_label: str) -> None:
    plot = dyn.loc[dyn["coef"].notna() & dyn["se"].notna()].copy()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for spec_key, spec_df in plot.groupby("fe_spec", sort=False):
        spec_df = spec_df.sort_values("horizon")
        ax.errorbar(
            spec_df["horizon"],
            spec_df["coef"],
            yerr=1.96 * spec_df["se"],
            marker="o",
            linewidth=1.8,
            capsize=2,
            label=spec_df["fe_label"].iloc[0],
            alpha=0.9,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Outcome horizon relative to exposure year")
    ax.set_ylabel(f"Coefficient on {variant.z_col}")
    ax.set_title(f"Dynamic {equation_label}: {variant.label}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _run_variant(
    variant: Variant,
    base_panel: pd.DataFrame,
    full_panel: pd.DataFrame,
    out_dir: Path,
    horizons: Iterable[int],
    equation: str,
    outcome_col: str,
    use_log_outcome: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if variant.z_col not in base_panel.columns:
        raise KeyError(f"Missing exposure column {variant.z_col}")
    panel = _apply_variant_filter(base_panel, full_panel, variant)
    panel = panel.loc[panel[variant.z_col].notna()].copy()
    z = pd.to_numeric(panel[variant.z_col], errors="coerce")
    z_sd = float(z.std()) if z.notna().any() else float("nan")
    z_iqr = float(z.quantile(0.75) - z.quantile(0.25)) if z.notna().any() else float("nan")

    static_rows: list[dict[str, object]] = []
    dynamic_rows: list[dict[str, object]] = []
    if equation == "first_stage":
        lhs_col = variant.x_col
        estimator = variant.estimator
        percent_effect = estimator == "ppml"
        equation_label = "first stage"
        file_suffix = "first_stage"
    elif equation == "reduced_form":
        if outcome_col not in panel.columns:
            raise KeyError(f"Missing outcome column {outcome_col}")
        lhs_col = f"log1p_{outcome_col}" if use_log_outcome else outcome_col
        if use_log_outcome:
            panel[lhs_col] = np.log1p(pd.to_numeric(panel[outcome_col], errors="coerce").clip(lower=0))
            full_panel = full_panel.copy()
            full_panel[lhs_col] = np.log1p(pd.to_numeric(full_panel[outcome_col], errors="coerce").clip(lower=0))
        estimator = "ols"
        percent_effect = use_log_outcome
        equation_label = "reduced form"
        file_suffix = "reduced_form"
    else:
        raise ValueError(f"Unknown equation: {equation}")
    for fe_key, fe_label, fe_cols in FE_SPECS:
        try:
            _, fit_panel, coef, se, f_stat = _fit_one(
                panel,
                lhs=lhs_col,
                rhs=variant.z_col,
                estimator=estimator,
                fe_cols=fe_cols,
            )
            sd_effect, iqr_effect = _interpretation(coef, z_sd, z_iqr, percent_effect)
            static_rows.append({
                "variant": variant.key,
                "label": variant.label,
                "z_col": variant.z_col,
                "lhs_col": lhs_col,
                "source_outcome_col": outcome_col if equation == "reduced_form" else "",
                "equation": equation,
                "estimator": estimator,
                "fe_spec": fe_key,
                "fe_label": fe_label,
                "coef": coef,
                "se": se,
                "t_stat": (coef / se) if coef is not None and se not in {None, 0} else np.nan,
                "f_stat": f_stat,
                "n_obs": len(fit_panel),
                "n_companies": fit_panel["c"].nunique(),
                "lhs_mean": float(pd.to_numeric(fit_panel[lhs_col], errors="coerce").mean()),
                "z_mean": float(z.mean()),
                "z_sd": z_sd,
                "z_iqr": z_iqr,
                "effect_per_1sd_pct_or_level": sd_effect,
                "effect_per_iqr_pct_or_level": iqr_effect,
                "notes": variant.notes,
            })
        except Exception as exc:
            static_rows.append({
                "variant": variant.key,
                "label": variant.label,
                "z_col": variant.z_col,
                "lhs_col": lhs_col,
                "source_outcome_col": outcome_col if equation == "reduced_form" else "",
                "equation": equation,
                "estimator": estimator,
                "fe_spec": fe_key,
                "fe_label": fe_label,
                "error": str(exc),
                "notes": variant.notes,
            })
            continue

        for horizon in horizons:
            dyn_lhs = f"dyn_{lhs_col}_h{horizon:+d}".replace("+", "p").replace("-", "m")
            panel[dyn_lhs] = _make_dynamic_panel(panel, full_panel, lhs_col, int(horizon))
            if equation == "first_stage" and estimator == "ols_lpm" and variant.x_col == "x_bin":
                panel[dyn_lhs] = (pd.to_numeric(panel[dyn_lhs], errors="coerce").fillna(0.0) > 0).astype(float)
            try:
                _, dyn_fit_panel, coef, se, f_stat = _fit_one(
                    panel,
                    lhs=dyn_lhs,
                    rhs=variant.z_col,
                    estimator=estimator,
                    fe_cols=fe_cols,
                )
                dynamic_rows.append({
                    "variant": variant.key,
                    "label": variant.label,
                    "z_col": variant.z_col,
                    "lhs_col": lhs_col,
                    "source_outcome_col": outcome_col if equation == "reduced_form" else "",
                    "equation": equation,
                    "estimator": estimator,
                    "fe_spec": fe_key,
                    "fe_label": fe_label,
                    "horizon": int(horizon),
                    "coef": coef,
                    "se": se,
                    "f_stat": f_stat,
                    "n_obs": _model_nobs(_, len(dyn_fit_panel)),
                    "n_companies": dyn_fit_panel["c"].nunique(),
                })
            except Exception as exc:
                dynamic_rows.append({
                    "variant": variant.key,
                    "label": variant.label,
                    "z_col": variant.z_col,
                    "lhs_col": lhs_col,
                    "source_outcome_col": outcome_col if equation == "reduced_form" else "",
                    "equation": equation,
                    "estimator": estimator,
                    "fe_spec": fe_key,
                    "fe_label": fe_label,
                    "horizon": int(horizon),
                    "error": str(exc),
                })

    static = pd.DataFrame(static_rows)
    dynamic = pd.DataFrame(dynamic_rows)
    variant_dir = out_dir / "variants" / _safe_name(variant.key)
    variant_dir.mkdir(parents=True, exist_ok=True)
    static.to_csv(variant_dir / f"static_{file_suffix}.csv", index=False)
    dynamic.to_csv(variant_dir / f"dynamic_{file_suffix}.csv", index=False)
    _plot_dynamic(dynamic, variant, variant_dir / f"dynamic_{file_suffix}.png", equation_label)
    return static, dynamic


def _write_matched_methodology(out_dir: Path) -> None:
    text = """# Matched Exposure Methodology

The matched variants use the saved `school_shift_sample.parquet` from the April
pipeline. Schools are screened over the 2014-2017 event window using the IHMP
share metric. Treated schools are the high positive-shift schools, ranked by the
largest event-window increase in the metric. Controls must have full window
coverage, meet the minimum school-size requirement, and stay below the configured
positive-shift cap. When available, controls are matched within Carnegie
classification; the matching objective then chooses nearby controls by school
size/log-size and related pre-event observables.

The `matched_step` exposure uses only the selected treated/control schools. For
treated schools, the school shock is a persistent post-event dose equal to
`pre_event_school_size * event_delta_share`; controls have zero treated shock.
The firm exposure is `sum_k pre_period_share_ck * matched_step_g_kt`.

The `matched_pulse` exposure uses the same selected school sample and the same
shock size, but the dose is nonzero only in the treated school's event year.

These matched variants are therefore much narrower than the broad raw-flow
instrument. They trade off a cleaner treated/control school design against fewer
active schools and greater school-level concentration.
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matched_methodology.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apr-root", type=Path, default=APR_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--equation", choices=["first_stage", "reduced_form"], default="first_stage")
    parser.add_argument("--outcome-col", default="y_new_hires_lag0")
    parser.add_argument("--no-log-outcome", action="store_true")
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--horizon-start", type=int, default=-4)
    parser.add_argument("--horizon-end", type=int, default=5)
    parser.add_argument("--variants", nargs="*", default=None, help="Optional subset of variant keys.")
    args = parser.parse_args()

    selected = VARIANTS
    if args.variants:
        wanted = set(args.variants)
        selected = [v for v in VARIANTS if v.key in wanted]
        missing = sorted(wanted - {v.key for v in selected})
        if missing:
            raise ValueError(f"Unknown variant keys: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_panel(args.apr_root)
    panel = _add_component_built_exposures(panel, args.apr_root)
    base = _base_analysis_panel(panel, args.start_year, args.end_year)
    horizons = list(range(args.horizon_start, args.horizon_end + 1))
    use_log_outcome = not args.no_log_outcome

    all_static: list[pd.DataFrame] = []
    all_dynamic: list[pd.DataFrame] = []
    for variant in selected:
        print(f"[variant] {variant.key}: {variant.label}")
        static, dynamic = _run_variant(
            variant,
            base,
            panel,
            args.out_dir,
            horizons,
            equation=args.equation,
            outcome_col=args.outcome_col,
            use_log_outcome=use_log_outcome,
        )
        all_static.append(static)
        all_dynamic.append(dynamic)

    static_all = pd.concat(all_static, ignore_index=True) if all_static else pd.DataFrame()
    dynamic_all = pd.concat(all_dynamic, ignore_index=True) if all_dynamic else pd.DataFrame()
    static_all.to_csv(args.out_dir / f"{args.equation}_variant_static_summary.csv", index=False)
    dynamic_all.to_csv(args.out_dir / f"{args.equation}_variant_dynamic_summary.csv", index=False)
    pd.DataFrame([v.__dict__ for v in selected]).to_csv(args.out_dir / "variant_definitions.csv", index=False)
    _write_matched_methodology(args.out_dir)
    print(f"[done] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
