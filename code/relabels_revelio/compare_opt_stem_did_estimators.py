"""Compare staggered-DiD estimators for the generalized relabel opt-STEM outcome.

This is intentionally a test/diagnostic script rather than a production pipeline
step. It builds or loads the FOIA-based generalized relabel DiD panel for
``opt_stem_share`` and compares several dynamic treatment-effect estimators:

* current TWFE event-time regression used by ``relabel_events_generalized``;
* Sun-Abraham style interaction-weighted cohort/event coefficients;
* Callaway-Sant'Anna style group-time ATTs against never-treated controls;
* stacked cohort DiD with stack-specific entity and calendar-year fixed effects.

Outputs are written to ``output/relabel_indiv/did_estimator_comparison`` by
default: long estimates CSV/parquet, a wide comparison CSV, a Markdown report,
and a graph with coefficient and standard-error paths.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb as ddb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import relabels_revelio.relabel_events_generalized as generalized


YVAR = "opt_stem_share"
DEFAULT_OUT_DIR = (
    Path(generalized.DEFAULT_OUTPUT_DIR) / "did_estimator_comparison"
)
DEFAULT_REFERENCE_EVENT_TIME = generalized.DI_D_REFERENCE_EVENT_TIME
DEFAULT_EVENT_MIN = generalized.PLOT_EVENT_MIN
DEFAULT_EVENT_MAX = generalized.PLOT_EVENT_MAX


@dataclass(frozen=True)
class EstimateSpec:
    name: str
    label: str
    notes: str


ESTIMATOR_SPECS = {
    "twfe": EstimateSpec(
        "twfe",
        "TWFE",
        "Current event-time x treated regression with relabel design fixed effects and calendar-year fixed effects.",
    ),
    "sun_abraham_iw": EstimateSpec(
        "sun_abraham_iw",
        "Sun-Abraham IW",
        "pyfixest saturated event-study, aggregated from cohort/event coefficients with treated exposure weights.",
    ),
    "callaway_santanna": EstimateSpec(
        "callaway_santanna",
        "Callaway-Sant'Anna",
        "csdid ATTgt with dynamic aggregation, never-treated controls, DR estimation, and cluster bootstrap SEs.",
    ),
    "stacked_did": EstimateSpec(
        "stacked_did",
        "Stacked DiD",
        "Cohort-stack event-time x treated regression with stack-specific entity and calendar-year fixed effects.",
    ),
}


def _elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "x"


def _event_slug(event_t: int) -> str:
    return f"m{abs(int(event_t))}" if int(event_t) < 0 else f"p{int(event_t)}"


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{100.0 * float(value):.2f}"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce").astype(float)
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    ok = y.notna() & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(y[ok], weights=w[ok]))


def _weighted_sum(values: pd.Series, weights: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return 0.0
    return float(np.sum(y[ok] * w[ok]))


def _result_params_and_cov(result: Any) -> tuple[pd.Series, pd.DataFrame]:
    params = result.coef()
    try:
        cov = result.vcov()
    except Exception:
        cov = None
    if cov is None:
        se = result.se()
        cov = pd.DataFrame(np.diag(np.square(se)), index=se.index, columns=se.index)
    return params.astype(float), cov.astype(float)


def _fit_feols(
    formula: str,
    *,
    data: pd.DataFrame,
    cluster_var: str,
    weight_var: str | None,
) -> Any:
    from pyfixest.estimation import feols

    kwargs: dict[str, Any] = {
        "fml": formula,
        "data": data,
        "vcov": {"CRV1": cluster_var},
        "copy_data": False,
        "store_data": False,
    }
    if weight_var is not None:
        kwargs["weights"] = weight_var
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return feols(**kwargs)


def _find_event_interaction_param(params: pd.Series, event_t: int) -> str | None:
    names = [str(name) for name in params.index]
    event = int(event_t)
    exact_patterns = [
        rf"\[T\.{re.escape(str(event))}\]",
        rf"\[{re.escape(str(event))}\]",
        rf"::{re.escape(str(event))}(?=($|[^0-9.]))",
        rf"\[T\.{re.escape(f'{float(event):.1f}')}\]",
        rf"\[{re.escape(f'{float(event):.1f}')}\]",
        rf"::{re.escape(f'{float(event):.1f}')}(?=($|[^0-9.]))",
    ]
    for name in names:
        low = name.lower()
        if "event_t" not in low or "treated" not in low:
            continue
        if any(re.search(pattern, name) for pattern in exact_patterns):
            return name
    return None


def _fe_var(df: pd.DataFrame, did_spec: str) -> str:
    return generalized._did_fe_var(df, did_spec=did_spec)  # noqa: SLF001


def _prepare_regression_df(
    did_panel: pd.DataFrame,
    *,
    did_spec: str,
    event_min: int,
    event_max: int,
) -> pd.DataFrame:
    reg_df = generalized._prepare_did_regression_df(  # noqa: SLF001
        did_panel,
        yvar=YVAR,
        did_spec=did_spec,
        event_time_min=event_min,
        event_time_max=event_max,
    )
    if reg_df.empty:
        return reg_df
    reg_df = reg_df.copy()
    reg_df["event_t"] = pd.to_numeric(reg_df["event_t"], errors="coerce").astype(int)
    reg_df["calendar_year"] = pd.to_numeric(reg_df["calendar_year"], errors="coerce").astype(int)
    reg_df["relabel_year"] = pd.to_numeric(reg_df["relabel_year"], errors="coerce").astype(int)
    reg_df["treated"] = pd.to_numeric(reg_df["treated"], errors="coerce").astype(int)
    reg_df["total_grads"] = pd.to_numeric(reg_df["total_grads"], errors="coerce").fillna(0.0)
    if "pair_id" not in reg_df.columns:
        reg_df["pair_id"] = reg_df["unitid"].astype(str)
    return reg_df


def _build_package_panel(reg_df: pd.DataFrame, *, did_spec: str, use_weights: bool) -> pd.DataFrame:
    """Collapse to one entity-time row for package estimators.

    The source generalized panel can be student/stack level, with many rows for
    the same school/design/year. Package DiD APIs expect a panel entity to have
    exactly one cohort and at most one row per time period, so we define entities
    at the cohort-stack/design level and collapse outcomes using total_grads.
    """
    if reg_df.empty:
        return pd.DataFrame()
    work = reg_df.copy()
    fe_var = _fe_var(work, did_spec)
    broad = work.get("broad_pair_bin", pd.Series("bin", index=work.index)).fillna("bin").astype(str)
    degree = work.get("degree_type", pd.Series("degree", index=work.index)).fillna("degree").astype(str)
    work["pkg_stack_id"] = (
        work["relabel_year"].astype(int).astype(str)
        + "||"
        + broad
        + "||"
        + degree
        + "||"
        + work["treated"].astype(int).astype(str)
    )
    work["pkg_entity"] = (
        work["pkg_stack_id"]
        + "||"
        + work[fe_var].astype(str)
        + "||"
        + work["unitid"].astype(int).astype(str)
    )
    grouped = (
        work.groupby(
            ["pkg_entity", "calendar_year", "treated", "relabel_year", "unitid"],
            as_index=False,
        )
        .apply(
            lambda g: pd.Series(
                {
                    YVAR: _weighted_mean(g[YVAR], g["total_grads"]) if use_weights else float(g[YVAR].mean()),
                    "total_grads": float(pd.to_numeric(g["total_grads"], errors="coerce").fillna(0.0).sum()),
                    "event_t": int(g["event_t"].iloc[0]),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    grouped = grouped.dropna(subset=[YVAR, "event_t", "calendar_year", "relabel_year", "unitid"]).copy()
    grouped = grouped[grouped["total_grads"] > 0].copy()
    grouped["pkg_entity_id"] = pd.factorize(grouped["pkg_entity"], sort=False)[0].astype("int64")
    grouped["calendar_year"] = grouped["calendar_year"].astype(int)
    grouped["calendar_year_ref"] = grouped["calendar_year"] + 1
    grouped["unitid"] = grouped["unitid"].astype(int)
    grouped["treated"] = grouped["treated"].astype(int)
    grouped["relabel_year"] = grouped["relabel_year"].astype(int)
    grouped["g_pyfixest"] = np.where(grouped["treated"].eq(1), grouped["relabel_year"], 0).astype(int)
    grouped["g_csdid"] = grouped["g_pyfixest"].astype(int)
    return grouped


def load_or_build_panel(args: argparse.Namespace) -> pd.DataFrame:
    cache_path = Path(args.panel_cache)
    if cache_path.exists() and not args.rebuild_panel:
        print(f"[load] using cached DiD panel: {cache_path}")
        return pd.read_parquet(cache_path)

    print("[build] constructing generalized opt-STEM DiD panel")
    relabel_panel = pd.read_parquet(args.relabel_panel)
    degree_type = None if args.degree_type.lower() in {"pooled", "all", "none"} else args.degree_type
    con = ddb.connect()
    try:
        did_panel = generalized.compute_generalized_did_panel(
            con,
            relabel_panel,
            degree_type=degree_type,
            did_spec=args.did_spec,
        )
    finally:
        con.close()
    if did_panel.empty:
        raise RuntimeError("Generalized DiD panel is empty; no estimator comparison can be run.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    did_panel.to_parquet(cache_path, index=False)
    print(f"[build] cached DiD panel: {cache_path} ({len(did_panel):,} rows)")
    return did_panel


def estimate_twfe(
    did_panel: pd.DataFrame,
    *,
    did_spec: str,
    reference_event_time: int,
    event_min: int,
    event_max: int,
    use_weights: bool,
) -> pd.DataFrame:
    out = generalized.compute_did_event_study_generalized(
        did_panel,
        yvar=YVAR,
        did_spec=did_spec,
        reference_event_time=reference_event_time,
        event_time_min=event_min,
        event_time_max=event_max,
        use_weights=use_weights,
    )
    if out.empty:
        return out
    out = out.copy()
    out["estimator"] = "twfe"
    out["estimator_label"] = ESTIMATOR_SPECS["twfe"].label
    return out


def _parse_pyfixest_saturated_name(name: object) -> tuple[int, int] | None:
    text = str(name)
    event_match = re.search(r"rel_time.*?\[T\.(-?\d+(?:\.\d+)?)\]", text)
    cohort_match = re.search(r"cohort_dummy_(\d+)", text)
    if event_match is None or cohort_match is None:
        return None
    shifted_event_t = int(float(event_match.group(1)))
    cohort = int(cohort_match.group(1))
    return cohort, shifted_event_t - 1


def estimate_sun_abraham_iw(
    pkg_panel: pd.DataFrame,
    *,
    reference_event_time: int,
    count_lookup: pd.DataFrame,
    use_weights: bool,
) -> pd.DataFrame:
    if pkg_panel.empty:
        return pd.DataFrame()

    try:
        import pyfixest as pf
    except Exception as exc:
        print(f"[warn] pyfixest unavailable for Sun-Abraham saturated event-study: {exc}")
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = pf.event_study(
            pkg_panel,
            yname=YVAR,
            idname="pkg_entity_id",
            tname="calendar_year_ref",
            gname="g_pyfixest",
            cluster="unitid",
            estimator="saturated",
            att=False,
        )
    params, cov = _result_params_and_cov(result)

    weight_rows = (
        pkg_panel[pkg_panel["treated"].eq(1)]
        .groupby(["relabel_year", "event_t"], as_index=False)
        .agg(weight=("total_grads", "sum"), n_rows=("unitid", "size"), n_schools=("unitid", "nunique"))
    )
    weight_lookup = {
        (int(row.relabel_year), int(row.event_t)): float(row.weight)
        for row in weight_rows.itertuples(index=False)
    }
    param_meta = {
        str(name): parsed
        for name in params.index
        if (parsed := _parse_pyfixest_saturated_name(name)) is not None
    }
    event_values = sorted(pkg_panel["event_t"].dropna().astype(int).unique())

    rows: list[dict[str, object]] = []
    for event_t in event_values:
        if event_t == reference_event_time:
            rows.append(_base_result_row(pkg_panel, event_t, "sun_abraham_iw", 0.0, 0.0, reference_event_time, count_lookup))
            continue
        cols = [name for name, (_, e) in param_meta.items() if e == int(event_t) and name in params.index]
        if not cols:
            rows.append(_base_result_row(pkg_panel, event_t, "sun_abraham_iw", float("nan"), float("nan"), reference_event_time, count_lookup))
            continue
        raw_weights = np.array([weight_lookup.get((param_meta[col][0], int(event_t)), 0.0) for col in cols], dtype=float)
        if raw_weights.sum() <= 0:
            raw_weights = np.ones(len(cols), dtype=float)
        weights = raw_weights / raw_weights.sum()
        beta = params.reindex(cols).astype(float).to_numpy()
        coef = float(np.dot(weights, beta))
        subcov = cov.reindex(index=cols, columns=cols).fillna(0.0).to_numpy(dtype=float)
        var = float(weights @ subcov @ weights)
        se = math.sqrt(var) if var >= 0 and np.isfinite(var) else float("nan")
        rows.append(_base_result_row(pkg_panel, event_t, "sun_abraham_iw", coef, se, reference_event_time, count_lookup))
    return pd.DataFrame(rows)


def _callaway_dynamic(
    df: pd.DataFrame,
    *,
    reference_event_time: int,
    event_values: list[int],
    row_weight_col: str,
) -> pd.DataFrame:
    work = df.copy()
    work["_row_weight"] = pd.to_numeric(work["total_grads"], errors="coerce").fillna(0.0)
    if row_weight_col != "total_grads":
        work["_row_weight"] *= pd.to_numeric(work[row_weight_col], errors="coerce").fillna(0.0)
    sums = (
        work.groupby(["relabel_year", "treated", "event_t"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "mean": _weighted_mean(g[YVAR], g["_row_weight"]),
                    "weight": float(g["_row_weight"].sum()),
                    "n_schools": int(g["unitid"].nunique()),
                    "n_rows": int(len(g)),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    lookup = {
        (int(row.relabel_year), int(row.treated), int(row.event_t)): row
        for row in sums.itertuples(index=False)
    }
    cohorts = sorted(work.loc[work["treated"].eq(1), "relabel_year"].dropna().astype(int).unique())
    rows: list[dict[str, object]] = []
    for event_t in event_values:
        cohort_rows: list[dict[str, float]] = []
        for cohort in cohorts:
            treat_t = lookup.get((cohort, 1, int(event_t)))
            treat_base = lookup.get((cohort, 1, int(reference_event_time)))
            control_t = lookup.get((cohort, 0, int(event_t)))
            control_base = lookup.get((cohort, 0, int(reference_event_time)))
            if any(x is None for x in (treat_t, treat_base, control_t, control_base)):
                continue
            if any(pd.isna(x.mean) for x in (treat_t, treat_base, control_t, control_base)):
                continue
            att = (float(treat_t.mean) - float(treat_base.mean)) - (
                float(control_t.mean) - float(control_base.mean)
            )
            cohort_rows.append(
                {
                    "att": att,
                    "weight": float(treat_t.weight),
                    "treated_n_school_years": float(treat_t.n_rows),
                    "control_n_school_years": float(control_t.n_rows),
                    "treated_n_schools": float(treat_t.n_schools),
                    "control_n_schools": float(control_t.n_schools),
                }
            )
        if event_t == reference_event_time:
            coef = 0.0
        elif cohort_rows:
            att_df = pd.DataFrame(cohort_rows)
            coef = _weighted_mean(att_df["att"], att_df["weight"])
        else:
            coef = float("nan")
        rows.append({"event_t": int(event_t), "coef": coef, "n_cohorts": len(cohort_rows)})
    return pd.DataFrame(rows)


def estimate_callaway_santanna(
    pkg_panel: pd.DataFrame,
    *,
    reference_event_time: int,
    bootstrap_reps: int,
    random_seed: int,
    count_lookup: pd.DataFrame,
    use_weights: bool,
) -> pd.DataFrame:
    if pkg_panel.empty:
        return pd.DataFrame()

    try:
        from csdid.att_gt import ATTgt
    except Exception as exc:
        print(f"[warn] csdid unavailable for Callaway-Sant'Anna: {exc}")
        return pd.DataFrame()

    cs_df = pkg_panel[
        ["pkg_entity_id", "calendar_year", "unitid", "g_csdid", YVAR, "total_grads"]
    ].copy()
    cs_df = cs_df.rename(columns={"pkg_entity_id": "entity_id"})
    anticipation = max(0, abs(int(reference_event_time)) - 1)
    bstrap = int(bootstrap_reps) > 0
    biters = max(1, int(bootstrap_reps)) if bstrap else 0

    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            np.random.seed(int(random_seed))
            model = ATTgt(
                yname=YVAR,
                tname="calendar_year",
                idname="entity_id",
                gname="g_csdid",
                data=cs_df,
                control_group="nevertreated",
                panel=True,
                allow_unbalanced_panel=True,
                clustervar="unitid",
                weights_name="total_grads",
                anticipation=anticipation,
                cband=False,
                biters=biters,
            )
            model.fit(est_method="dr", base_period="universal", bstrap=bstrap)
            model.aggte(
                typec="dynamic",
                min_e=int(pkg_panel["event_t"].min()),
                max_e=int(pkg_panel["event_t"].max()),
                na_rm=True,
                bstrap=bstrap,
                biters=biters if bstrap else None,
                cband=False,
            )

    event_values = sorted(pkg_panel["event_t"].dropna().astype(int).unique())
    att_lookup = {
        int(event_t): (float(att), float(se))
        for event_t, att, se in zip(
            list(model.atte.get("egt", [])),
            list(model.atte.get("att_egt", [])),
            np.asarray(model.atte.get("se_egt", [[]]), dtype=float).reshape(-1).tolist(),
        )
    }
    rows = []
    for event_t in event_values:
        if event_t == reference_event_time:
            coef, se = 0.0, 0.0
        else:
            coef, se = att_lookup.get(int(event_t), (float("nan"), float("nan")))
        rows.append(
            _base_result_row(
                pkg_panel,
                int(event_t),
                "callaway_santanna",
                coef,
                se,
                reference_event_time,
                count_lookup,
            )
        )
    return pd.DataFrame(rows)


def estimate_stacked_did(
    reg_df: pd.DataFrame,
    *,
    did_spec: str,
    reference_event_time: int,
    use_weights: bool,
    count_lookup: pd.DataFrame,
) -> pd.DataFrame:
    if reg_df.empty:
        return pd.DataFrame()
    work = reg_df.copy()
    fe_var = _fe_var(work, did_spec)
    stack_parts = [
        work["relabel_year"].astype(str),
        work.get("broad_pair_bin", pd.Series("bin", index=work.index)).fillna("bin").astype(str),
        work.get("degree_type", pd.Series("degree", index=work.index)).fillna("degree").astype(str),
    ]
    work["stack_id"] = stack_parts[0] + "||" + stack_parts[1] + "||" + stack_parts[2]
    work["stack_entity"] = work["stack_id"] + "||" + work[fe_var].astype(str)
    work["stack_year"] = work["stack_id"] + "||" + work["grad_year"].astype(str)

    formula = f"{YVAR} ~ i(event_t, treated, ref={int(reference_event_time)}) | stack_entity + stack_year"
    result = _fit_feols(
        formula,
        data=work,
        cluster_var="unitid",
        weight_var="total_grads" if use_weights else None,
    )
    params, cov = _result_params_and_cov(result)
    event_values = sorted(work["event_t"].dropna().astype(int).unique())
    rows: list[dict[str, object]] = []
    for event_t in event_values:
        if int(event_t) == int(reference_event_time):
            coef, se = 0.0, 0.0
        else:
            param = _find_event_interaction_param(params, int(event_t))
            coef = float(params.get(param, float("nan"))) if param is not None else float("nan")
            var = float(cov.loc[param, param]) if param is not None and param in cov.index else float("nan")
            se = math.sqrt(var) if var >= 0 and np.isfinite(var) else float("nan")
        rows.append(_base_result_row(work, int(event_t), "stacked_did", coef, se, reference_event_time, count_lookup))
    return pd.DataFrame(rows)


def _base_result_row(
    df: pd.DataFrame,
    event_t: int,
    estimator: str,
    coef: float,
    se: float,
    reference_event_time: int,
    count_lookup: pd.DataFrame,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    sub_counts = count_lookup[count_lookup["event_t"].eq(int(event_t))]
    treated = sub_counts[sub_counts["treated"].eq(1)]
    control = sub_counts[sub_counts["treated"].eq(0)]
    row = {
        "event_t": int(event_t),
        "estimator": estimator,
        "estimator_label": ESTIMATOR_SPECS[estimator].label,
        "coef": float(coef) if coef is not None else float("nan"),
        "se": float(se) if se is not None else float("nan"),
        "ci_low": float(coef) - 1.96 * float(se) if pd.notna(coef) and pd.notna(se) else float("nan"),
        "ci_high": float(coef) + 1.96 * float(se) if pd.notna(coef) and pd.notna(se) else float("nan"),
        "reference_event_t": int(reference_event_time),
        "nobs": int(len(df)),
        "n_schools_total": int(df["unitid"].nunique()),
        "treated_n_school_years": int(treated["n_school_years"].sum()) if not treated.empty else 0,
        "control_n_school_years": int(control["n_school_years"].sum()) if not control.empty else 0,
        "treated_n_schools": int(treated["n_schools"].sum()) if not treated.empty else 0,
        "control_n_schools": int(control["n_schools"].sum()) if not control.empty else 0,
        "treated_total_grads": float(treated["total_grads"].sum()) if not treated.empty else 0.0,
        "control_total_grads": float(control["total_grads"].sum()) if not control.empty else 0.0,
    }
    if extra:
        row.update(extra)
    return row


def _count_lookup(reg_df: pd.DataFrame) -> pd.DataFrame:
    return (
        reg_df.groupby(["event_t", "treated"], as_index=False)
        .agg(
            n_school_years=("unitid", "size"),
            n_schools=("unitid", "nunique"),
            total_grads=("total_grads", "sum"),
        )
    )


def plot_comparison(results: pd.DataFrame, out_path: Path) -> Path:
    plot_df = results.dropna(subset=["event_t", "coef"]).copy()
    if plot_df.empty:
        raise RuntimeError("No coefficient estimates available to plot.")
    plot_df = plot_df.sort_values(["estimator", "event_t"])
    labels = [name for name in ESTIMATOR_SPECS if name in set(plot_df["estimator"])]
    colors = {
        "twfe": "#1f77b4",
        "sun_abraham_iw": "#d62728",
        "callaway_santanna": "#2ca02c",
        "stacked_did": "#9467bd",
    }
    markers = {
        "twfe": "o",
        "sun_abraham_iw": "s",
        "callaway_santanna": "^",
        "stacked_did": "D",
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax, ax_se = axes
    for estimator in labels:
        sub = plot_df[plot_df["estimator"].eq(estimator)].sort_values("event_t")
        label = ESTIMATOR_SPECS[estimator].label
        color = colors.get(estimator)
        ax.plot(sub["event_t"], 100.0 * sub["coef"], label=label, color=color, linewidth=1.8)
        ax.scatter(sub["event_t"], 100.0 * sub["coef"], color=color, marker=markers.get(estimator, "o"), s=36)
        with_se = sub.dropna(subset=["se"])
        if not with_se.empty:
            ax.errorbar(
                with_se["event_t"],
                100.0 * with_se["coef"],
                yerr=1.96 * 100.0 * with_se["se"],
                fmt="none",
                ecolor=color,
                alpha=0.20,
                linewidth=1.4,
                capsize=0,
            )
            ax_se.plot(with_se["event_t"], 100.0 * with_se["se"], label=label, color=color, linewidth=1.8)
            ax_se.scatter(with_se["event_t"], 100.0 * with_se["se"], color=color, marker=markers.get(estimator, "o"), s=28)
    for axis in axes:
        axis.axhline(0, color="gray", linestyle="--", linewidth=0.9)
        axis.axvline(-0.5, color="gray", linestyle="--", linewidth=0.9)
        axis.grid(True, axis="y", alpha=0.25)
    ax.set_ylabel("Effect on OPT STEM share (pp)")
    ax.set_title("Dynamic DiD estimator comparison: opt_stem_share")
    ax.legend(ncol=2, frameon=False)
    ax_se.set_xlabel("Graduation Cohort Relative to Relabel Event")
    ax_se.set_ylabel("SE (pp)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_report(
    *,
    args: argparse.Namespace,
    reg_df: pd.DataFrame,
    results: pd.DataFrame,
    wide: pd.DataFrame,
    out_paths: dict[str, Path],
) -> Path:
    report_path = out_paths["report"]
    lines: list[str] = []
    lines.append("# opt_stem_share DiD Estimator Comparison")
    lines.append("")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Design")
    lines.append("")
    lines.append(f"- Outcome: `{YVAR}`")
    lines.append(f"- DiD spec: `{args.did_spec}`")
    lines.append(f"- Degree sample: `{args.degree_type}`")
    lines.append(f"- Event window: [{args.event_min}, {args.event_max}]")
    lines.append(f"- Reference event time: {args.reference_event_time}")
    lines.append(f"- Weighted by `total_grads`: {not args.no_weights}")
    lines.append(f"- Callaway-Sant'Anna cluster bootstrap reps: {args.bootstrap_reps}")
    lines.append("")
    lines.append("## Panel")
    lines.append("")
    lines.append(f"- Regression rows: {len(reg_df):,}")
    lines.append(f"- Schools/clusters: {reg_df['unitid'].nunique():,}")
    lines.append(f"- Treated relabel cohorts: {reg_df.loc[reg_df['treated'].eq(1), 'relabel_year'].nunique():,}")
    lines.append(f"- Calendar years: {int(reg_df['calendar_year'].min())}-{int(reg_df['calendar_year'].max())}")
    lines.append(f"- Treated rows: {int(reg_df['treated'].sum()):,}")
    lines.append(f"- Control rows: {int((1 - reg_df['treated']).sum()):,}")
    lines.append("")
    lines.append("## Estimators")
    lines.append("")
    for spec in ESTIMATOR_SPECS.values():
        if spec.name in set(results["estimator"]):
            lines.append(f"- **{spec.label}**: {spec.notes}")
    lines.append("")
    lines.append("## Post-Period Summary")
    lines.append("")
    summary_rows = []
    for estimator, grp in results.groupby("estimator", sort=False):
        post = grp[grp["event_t"] >= -1].dropna(subset=["coef"])
        pre = grp[grp["event_t"] < args.reference_event_time].dropna(subset=["coef"])
        summary_rows.append(
            {
                "estimator": ESTIMATOR_SPECS[estimator].label,
                "post_mean_pp": _format_pct(post["coef"].mean() if not post.empty else float("nan")),
                "post_max_pp": _format_pct(post["coef"].max() if not post.empty else float("nan")),
                "pre_rms_pp": _format_pct(math.sqrt(float(np.mean(np.square(pre["coef"])))) if not pre.empty else float("nan")),
                "median_se_pp": _format_pct(grp["se"].replace(0.0, np.nan).median()),
            }
        )
    lines.append(pd.DataFrame(summary_rows).to_markdown(index=False))
    lines.append("")
    if "twfe" in set(results["estimator"]):
        lines.append("## Differences Versus TWFE")
        lines.append("")
        twfe = results[results["estimator"].eq("twfe")][["event_t", "coef", "se"]].rename(
            columns={"coef": "twfe_coef", "se": "twfe_se"}
        )
        diff_rows = []
        for estimator, grp in results[~results["estimator"].eq("twfe")].groupby("estimator", sort=False):
            merged = grp.merge(twfe, on="event_t", how="inner")
            merged = merged.dropna(subset=["coef", "twfe_coef"])
            post = merged[merged["event_t"] >= -1]
            diff_rows.append(
                {
                    "estimator": ESTIMATOR_SPECS[estimator].label,
                    "max_abs_path_diff_pp": _format_pct((merged["coef"] - merged["twfe_coef"]).abs().max()),
                    "post_mean_diff_pp": _format_pct((post["coef"] - post["twfe_coef"]).mean() if not post.empty else float("nan")),
                    "median_se_ratio": f"{(merged['se'] / merged['twfe_se'].replace(0.0, np.nan)).median():.2f}",
                }
            )
        lines.append(pd.DataFrame(diff_rows).to_markdown(index=False))
        lines.append("")
    lines.append("## Dynamic Estimates")
    lines.append("")
    table = wide.copy()
    display_cols = ["event_t"]
    for estimator in ESTIMATOR_SPECS:
        if f"{estimator}_coef" in table.columns:
            table[f"{estimator}_coef_pp"] = table[f"{estimator}_coef"].map(_format_pct)
            table[f"{estimator}_se_pp"] = table[f"{estimator}_se"].map(_format_pct)
            display_cols.extend([f"{estimator}_coef_pp", f"{estimator}_se_pp"])
    lines.append(table[display_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for key, path in out_paths.items():
        lines.append(f"- {key}: `{path}`")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_wide(results: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for estimator, grp in results.groupby("estimator", sort=False):
        sub = grp[["event_t", "coef", "se", "ci_low", "ci_high"]].copy()
        sub = sub.rename(
            columns={
                "coef": f"{estimator}_coef",
                "se": f"{estimator}_se",
                "ci_low": f"{estimator}_ci_low",
                "ci_high": f"{estimator}_ci_high",
            }
        )
        pieces.append(sub)
    wide = pieces[0]
    for piece in pieces[1:]:
        wide = wide.merge(piece, on="event_t", how="outer")
    return wide.sort_values("event_t").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dynamic DiD estimators for generalized relabel opt_stem_share."
    )
    parser.add_argument(
        "--relabel-panel",
        type=str,
        default=str(generalized.DEFAULT_PANEL_PARQUET),
        help="Generalized relabel panel parquet produced by relabel_events_generalized.",
    )
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--panel-cache",
        type=str,
        default="",
        help="Cached generalized DiD panel. Defaults inside --out-dir.",
    )
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument(
        "--did-spec",
        choices=generalized.VALID_DID_SPECS,
        default=generalized.DEFAULT_DID_SPEC,
    )
    parser.add_argument("--degree-type", default="pooled", help="pooled, Bachelor, Master, or Doctor.")
    parser.add_argument("--event-min", type=int, default=DEFAULT_EVENT_MIN)
    parser.add_argument("--event-max", type=int, default=DEFAULT_EVENT_MAX)
    parser.add_argument("--reference-event-time", type=int, default=DEFAULT_REFERENCE_EVENT_TIME)
    parser.add_argument("--bootstrap-reps", type=int, default=199)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-weights", action="store_true", help="Run regressions unweighted instead of weighting by total_grads.")
    return parser.parse_args()


def main() -> None:
    start = time.time()
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.panel_cache:
        args.panel_cache = str(out_dir / f"opt_stem_did_panel_{_slug(args.degree_type)}_{_slug(args.did_spec)}.parquet")

    did_panel = load_or_build_panel(args)
    reg_df = _prepare_regression_df(
        did_panel,
        did_spec=args.did_spec,
        event_min=args.event_min,
        event_max=args.event_max,
    )
    if reg_df.empty:
        raise RuntimeError("Prepared regression panel is empty after event-window and outcome filters.")
    counts = _count_lookup(reg_df)
    use_weights = not args.no_weights
    pkg_panel = _build_package_panel(reg_df, did_spec=args.did_spec, use_weights=use_weights)
    if pkg_panel.empty:
        raise RuntimeError("Package estimator panel is empty after collapsing to entity-time rows.")
    pkg_counts = _count_lookup(pkg_panel)
    print(
        f"[run] rows={len(reg_df):,}, schools={reg_df['unitid'].nunique():,}, "
        f"event_t={sorted(reg_df['event_t'].unique().tolist())}"
    )
    print(
        f"[run] package panel rows={len(pkg_panel):,}, "
        f"entities={pkg_panel['pkg_entity_id'].nunique():,}"
    )

    frames: list[pd.DataFrame] = []
    print("[estimate] TWFE")
    frames.append(
        estimate_twfe(
            did_panel,
            did_spec=args.did_spec,
            reference_event_time=args.reference_event_time,
            event_min=args.event_min,
            event_max=args.event_max,
            use_weights=use_weights,
        )
    )
    print("[estimate] Sun-Abraham interaction-weighted")
    frames.append(
        estimate_sun_abraham_iw(
            pkg_panel,
            reference_event_time=args.reference_event_time,
            count_lookup=pkg_counts,
            use_weights=use_weights,
        )
    )
    print("[estimate] Callaway-Sant'Anna group-time ATT")
    frames.append(
        estimate_callaway_santanna(
            pkg_panel,
            reference_event_time=args.reference_event_time,
            bootstrap_reps=args.bootstrap_reps,
            random_seed=args.random_seed,
            count_lookup=pkg_counts,
            use_weights=use_weights,
        )
    )
    print("[estimate] stacked DiD")
    frames.append(
        estimate_stacked_did(
            reg_df,
            did_spec=args.did_spec,
            reference_event_time=args.reference_event_time,
            use_weights=use_weights,
            count_lookup=counts,
        )
    )
    results = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True)
    if results.empty:
        raise RuntimeError("No estimates were produced.")
    results = results.sort_values(["estimator", "event_t"]).reset_index(drop=True)
    wide = build_wide(results)

    stem = f"opt_stem_{_slug(args.degree_type)}_{_slug(args.did_spec)}"
    out_paths = {
        "panel_cache": Path(args.panel_cache),
        "estimates_csv": out_dir / f"{stem}_did_estimator_comparison_long.csv",
        "estimates_parquet": out_dir / f"{stem}_did_estimator_comparison_long.parquet",
        "wide_csv": out_dir / f"{stem}_did_estimator_comparison_wide.csv",
        "figure": out_dir / f"{stem}_dynamic_effects_and_se_paths.png",
        "report": out_dir / f"{stem}_did_estimator_comparison_report.md",
    }
    results.to_csv(out_paths["estimates_csv"], index=False)
    results.to_parquet(out_paths["estimates_parquet"], index=False)
    wide.to_csv(out_paths["wide_csv"], index=False)
    plot_comparison(results, out_paths["figure"])
    write_report(args=args, reg_df=reg_df, results=results, wide=wide, out_paths=out_paths)
    print(f"[done] wrote report: {out_paths['report']}")
    print(f"[done] wrote figure: {out_paths['figure']}")
    print(f"[done] completed in {_elapsed(start)}")


if __name__ == "__main__":
    main()
