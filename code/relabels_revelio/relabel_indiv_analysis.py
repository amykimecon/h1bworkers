# File Description: Individual-level labor market outcomes around economics→econometrics
#   relabel events in IPEDS/FOIA data. Identifies Revelio individuals who attended
#   relabeled programs and tracks staying-in-US, number of positions, and imputed salary
#   at fixed horizons after graduation. Produces cohort-based event-study plots and a DiD.
#
# Pipeline:
#   Step 1 - Detect relabel events (reuse econ_relabels_opt_usage_v2) + aggregate plots
#   Step 2 - Load stage-04 match-ready Revelio education sample
#   Step 3 - Match individuals to relabel events (treated group)
#   Step 4 - Build individual × horizon outcome panel
#   Step 5 - Event study aggregation + plots (treated only)
#   Step 6 - Control group via never-treated econ institution matching
#   Step 7 - Treated vs. control event study plots
#   Step 8 - Staggered DiD

import math
import os
import sys
import time
from pathlib import Path

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Ensure progress logs flush immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


# ── path setup so we can import from repo root ──────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import relabels_revelio.relabel_indiv_config as cfg
import f1_foia.econ_relabels_opt_usage_v2 as v2

# ── global config ────────────────────────────────────────────────────────────
print(f"[relabel_indiv] Using config: {cfg.ACTIVE_CONFIG_PATH}")
print(f"[relabel_indiv] run_tag={cfg.RUN_TAG}  testing={cfg.TESTING_ENABLED}")
print(f"[relabel_indiv] horizons={getattr(cfg, 'BUILD_OUTCOME_HORIZONS', [3])}")

OUTCOMES = ["in_us", "n_pos", "salary_imputed"]
OUTCOME_LABELS = {
    "in_us":           "Share with active US position",
    "n_pos":           "Mean number of active positions",
    "salary_imputed":  "Mean imputed annual compensation (USD)",
}
OUTCOME_FILE_LABELS = {
    "in_us": "active_us",
    "n_pos": "active_positions",
    "salary_imputed": "compensation",
}

OUTPUT_DIR = Path(cfg.OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_FONT_SIZE = 12 * 1.4
sns.set(style="whitegrid")
plt.rcParams.update({"font.size": BASE_FONT_SIZE})

t0 = time.time()
SUPPORTED_SAMPLE_VARIANTS = {
    "stage04_all",
    "foia_linked_person_baseline",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _elapsed(since: float) -> str:
    return f"{time.time() - since:.1f}s"


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _variant_slug(label: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in str(label).strip().lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "variant"


def _save_and_show(fig: plt.Figure, name: str, analysis_variant: str | None = None) -> Path:
    """Save figure to OUTPUT_DIR and display it."""
    if analysis_variant:
        name = f"{name}_{_variant_slug(analysis_variant)}"
    out = OUTPUT_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  → saved {out}")
    return out


def _table_exists(con: ddb.DuckDBPyConnection, name: str) -> bool:
    return name in {row[0] for row in con.sql("SHOW TABLES").fetchall()}


def _cip_prefix_where_clause(col: str = "cip") -> str:
    prefixes = [str(prefix).strip() for prefix in cfg.BUILD_SAMPLE_CIP_PREFIXES if str(prefix).strip()]
    if not prefixes:
        return "TRUE"
    digits_expr = f"regexp_replace(CAST({col} AS VARCHAR), '[^0-9]', '', 'g')"
    parts = [f"{digits_expr} LIKE '{_escape_sql_literal(prefix)}%'" for prefix in prefixes]
    return "(" + " OR ".join(parts) + ")"


def _sample_view_name(analysis_variant: str) -> str:
    if analysis_variant == "stage04_all":
        return "stage04_sample_all"
    if analysis_variant == "foia_linked_person_baseline":
        return "stage04_sample_foia_linked_person_baseline"
    return f"stage04_sample_{_variant_slug(analysis_variant)}"


def _analysis_horizons() -> list[int]:
    horizons = sorted({int(h) for h in getattr(cfg, "BUILD_OUTCOME_HORIZONS", [3]) if int(h) >= 0})
    if not horizons:
        raise ValueError("No non-negative outcome horizons configured.")
    return horizons


def _agg_cohort_time(
    panel: pd.DataFrame,
    group_col: str | None = None,
    *,
    observed_only: bool = True,
) -> pd.DataFrame:
    """Aggregate outcome panel by cohort_t and horizon_years."""
    if panel.empty:
        keys = ["horizon_years", "cohort_t"] if group_col is None else ["horizon_years", "cohort_t", group_col]
        return pd.DataFrame(columns=keys + ["n"])

    work = panel.copy()
    if observed_only and "target_year_observed" in work.columns:
        work = work[work["target_year_observed"] == 1].copy()
    if work.empty:
        keys = ["horizon_years", "cohort_t"] if group_col is None else ["horizon_years", "cohort_t", group_col]
        return pd.DataFrame(columns=keys + ["n"])

    keys = ["horizon_years", "cohort_t"] if group_col is None else ["horizon_years", "cohort_t", group_col]
    rows = []
    for grp_vals, grp in work.groupby(keys, dropna=False):
        if not isinstance(grp_vals, tuple):
            grp_vals = (grp_vals,)
        row: dict[str, object] = dict(zip(keys, grp_vals))
        row["n"] = len(grp)
        if "target_year_observed" in grp.columns:
            row["target_year_observed_share"] = float(grp["target_year_observed"].mean())
        if "used_latest_avail" in grp.columns:
            row["used_latest_avail_share"] = float(grp["used_latest_avail"].mean())
        for out in OUTCOMES:
            if out in grp.columns:
                row[f"{out}_mean"] = grp[out].mean()
                row[f"{out}_se"] = grp[out].sem()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _coerce_plot_frame(df: pd.DataFrame, cols: list[str], sort_col: str) -> pd.DataFrame:
    """Convert plotting columns to plain numeric values before handing them to Matplotlib."""
    plot_df = df.loc[:, cols].copy()
    for col in cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    return plot_df.dropna(subset=cols).sort_values(sort_col)


def _did_models_to_run() -> list[str]:
    model = str(getattr(cfg, "BUILD_DID_MODEL", "simple")).strip().lower()
    if model == "simple":
        return ["simple"]
    if model in {"panel", "both"}:
        print(
            "  Warning: panel DiD is not identified under horizon-based outcomes; "
            "using the clustered simple FE specification instead."
        )
        return ["simple"]
    print(f"  Warning: unsupported did_model={model!r}; defaulting to 'simple'")
    return ["simple"]


def _did_results_output_path() -> Path:
    if getattr(cfg, "OUTPUT_DID_RESULTS_PARQUET", ""):
        return Path(cfg.OUTPUT_DID_RESULTS_PARQUET)
    return OUTPUT_DIR / f"relabel_did_results_{cfg.RUN_TAG}.parquet"


def _horizon_file_suffix(horizon: int, available_horizons: list[int]) -> str:
    return "" if len(set(int(h) for h in available_horizons)) == 1 else f"_t{int(horizon)}"


def _outcome_file_label(outcome: str) -> str:
    return OUTCOME_FILE_LABELS.get(outcome, _variant_slug(outcome))


def _supported_did_cohorts(
    df: pd.DataFrame,
    *,
    reference_cohort_t: int = -1,
) -> list[int]:
    counts = df.groupby(["cohort_t", "treated_ind"]).size().unstack(fill_value=0)
    if reference_cohort_t not in counts.index:
        return []
    if counts.loc[reference_cohort_t].get(0, 0) == 0 or counts.loc[reference_cohort_t].get(1, 0) == 0:
        return []

    supported: list[int] = []
    for cohort_t in sorted(int(v) for v in counts.index.tolist()):
        if cohort_t == reference_cohort_t:
            continue
        if counts.loc[cohort_t].get(0, 0) > 0 and counts.loc[cohort_t].get(1, 0) > 0:
            supported.append(int(cohort_t))
    return supported


def _find_did_interaction_param(
    params: pd.Series,
    cohort_t: int,
    reference_cohort_t: int,
) -> str | None:
    term = f"C(cohort_t, Treatment(reference={reference_cohort_t}))"
    candidates = [
        f"{term}[T.{cohort_t}]:treated_ind",
        f"{term}[{cohort_t}]:treated_ind",
        f"treated_ind:{term}[T.{cohort_t}]",
        f"treated_ind:{term}[{cohort_t}]",
    ]
    for candidate in candidates:
        if candidate in params.index:
            return candidate
    return None


def _normal_pvalue_from_coef_se(coef: float, se: float) -> float:
    if pd.isna(coef) or pd.isna(se) or se <= 0:
        return float("nan")
    z_val = abs(float(coef) / float(se))
    return float(math.erfc(z_val / math.sqrt(2.0)))


def _plot_did_coefficients(
    results_df: pd.DataFrame,
    *,
    analysis_variant: str | None = None,
) -> None:
    """Plot cohort-based DiD coefficients for one analysis variant."""
    if results_df.empty:
        return

    order = [out for out in OUTCOMES if out in set(results_df["outcome"])]
    if not order:
        return

    reference_cohort_t = (
        int(results_df["reference_cohort_t"].dropna().iloc[0])
        if "reference_cohort_t" in results_df.columns
        else -1
    )
    available_horizons = sorted(results_df["horizon_years"].dropna().astype(int).unique().tolist())
    for did_model, model_grp in results_df.groupby("did_model", sort=False):
        for horizon, grp in model_grp.groupby("horizon_years", sort=True):
            cohort_ticks = sorted(
                {int(v) for v in grp["cohort_t"].dropna().tolist()} | {reference_cohort_t}
            )
            if not cohort_ticks:
                continue

            for outcome in order:
                plot_df = grp[grp["outcome"] == outcome].dropna(subset=["coef", "se"])
                if plot_df.empty:
                    continue
                line_df = _coerce_plot_frame(plot_df, ["cohort_t", "coef", "se"], sort_col="cohort_t")
                if line_df.empty:
                    continue

                fig, ax = plt.subplots(figsize=(7.6, 4.8))
                ax.errorbar(
                    line_df["cohort_t"].to_numpy(dtype=float),
                    line_df["coef"].to_numpy(dtype=float),
                    yerr=1.96 * line_df["se"].to_numpy(dtype=float),
                    fmt="o-",
                    color="#2e8b57",
                    ecolor="#7aa78a",
                    elinewidth=2,
                    capsize=4,
                    markersize=6,
                    linewidth=1.5,
                )
                ax.scatter(
                    [reference_cohort_t],
                    [0.0],
                    facecolors="white",
                    edgecolors="black",
                    linewidths=1.2,
                    s=42,
                    zorder=4,
                )
                ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
                ax.axvline(x=0, linestyle=":", color="gray", linewidth=1)
                ax.set_xticks(cohort_ticks)
                ax.set_title(OUTCOME_LABELS.get(outcome, outcome))
                ax.set_xlabel("Graduation year relative to relabel event")
                ax.set_ylabel(f"DiD coefficient vs cohort_t={reference_cohort_t}")
                fig.suptitle(
                    f"DiD coefficients ({did_model}, {int(horizon)} years after graduation)",
                    y=1.02,
                )
                fig.tight_layout()
                _save_and_show(
                    fig,
                    f"did_att_{_outcome_file_label(outcome)}{_horizon_file_suffix(int(horizon), available_horizons)}",
                    analysis_variant=analysis_variant,
                )


def _plot_did_variant_comparison(results_df: pd.DataFrame) -> None:
    """Plot DiD coefficients across analysis variants for each model spec."""
    if results_df.empty or results_df["analysis_variant"].nunique() < 2:
        return

    order = [out for out in OUTCOMES if out in set(results_df["outcome"])]
    if not order:
        return

    variant_order = list(dict.fromkeys(results_df["analysis_variant"].tolist()))
    palette = sns.color_palette("Set2", n_colors=len(variant_order))
    marker_cycle = ["o", "s", "D", "^", "P", "X"]
    variant_offsets = np.linspace(-0.14, 0.14, num=len(variant_order)) if len(variant_order) > 1 else np.array([0.0])
    reference_cohort_t = (
        int(results_df["reference_cohort_t"].dropna().iloc[0])
        if "reference_cohort_t" in results_df.columns
        else -1
    )
    available_horizons = sorted(results_df["horizon_years"].dropna().astype(int).unique().tolist())

    for did_model, model_grp in results_df.groupby("did_model", sort=False):
        for horizon, grp in model_grp.groupby("horizon_years", sort=True):
            cohort_ticks = sorted(
                {int(v) for v in grp["cohort_t"].dropna().tolist()} | {reference_cohort_t}
            )
            if not cohort_ticks:
                continue

            for outcome in order:
                outcome_grp = grp[grp["outcome"] == outcome]
                if outcome_grp.empty:
                    continue

                fig, ax = plt.subplots(figsize=(7.8, 4.8))
                plotted_any = False
                legend_handles = None
                legend_labels = None

                for idx, (color, variant) in enumerate(zip(palette, variant_order)):
                    sub = outcome_grp[outcome_grp["analysis_variant"] == variant].dropna(subset=["coef", "se"])
                    if sub.empty:
                        continue
                    line_df = _coerce_plot_frame(sub, ["cohort_t", "coef", "se"], sort_col="cohort_t")
                    if line_df.empty:
                        continue
                    plotted_any = True
                    x_vals = line_df["cohort_t"].to_numpy(dtype=float) + float(variant_offsets[idx])
                    ax.errorbar(
                        x_vals,
                        line_df["coef"].to_numpy(dtype=float),
                        yerr=1.96 * line_df["se"].to_numpy(dtype=float),
                        fmt="-",
                        color=color,
                        ecolor=color,
                        elinewidth=1.5,
                        capsize=3,
                        markersize=5,
                        linewidth=1.3,
                        marker=marker_cycle[idx % len(marker_cycle)],
                        label=variant.replace("_", " "),
                        zorder=3 + idx,
                    )

                ax.scatter(
                    [reference_cohort_t],
                    [0.0],
                    facecolors="white",
                    edgecolors="black",
                    linewidths=1.1,
                    s=38,
                    zorder=4,
                )
                ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
                ax.axvline(x=0, linestyle=":", color="gray", linewidth=1)
                ax.set_xticks(cohort_ticks)
                ax.set_title(OUTCOME_LABELS.get(outcome, outcome))
                ax.set_xlabel("Graduation year relative to relabel event")
                ax.set_ylabel(f"DiD coefficient vs cohort_t={reference_cohort_t}")
                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()

                if not plotted_any:
                    plt.close(fig)
                    continue

                if legend_handles:
                    fig.legend(
                        legend_handles,
                        legend_labels,
                        loc="lower center",
                        ncol=len(legend_labels),
                        frameon=False,
                    )
                fig.suptitle(
                    f"DiD by match sample ({did_model}, {int(horizon)} years after graduation)",
                    y=1.03,
                )
                fig.tight_layout()
                if legend_handles:
                    fig.subplots_adjust(bottom=0.20)
                _save_and_show(
                    fig,
                    f"did_att_by_variant_{_outcome_file_label(outcome)}"
                    f"{_horizon_file_suffix(int(horizon), available_horizons)}",
                )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Detect relabel events + aggregate FOIA/IPEDS plots
# ─────────────────────────────────────────────────────────────────────────────

def step1_relabels(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load or compute relabel events using the v2 detector.
    Also reproduces the aggregate FOIA/IPEDS plots from econ_relabels_opt_usage_v2.
    """
    t = time.time()
    print("\n── Step 1: Detecting relabel events ──────────────────────────────────")

    relabels_path = cfg.RELABELS_PARQUET
    if os.path.exists(relabels_path) and not cfg.BUILD_OVERWRITE:
        print(f"  Loading cached relabels from {relabels_path}")
        relabel_df = pd.read_parquet(relabels_path)
    else:
        print("  Running v2 relabel detector (IPEDS + FOIA)...")
        relabel_df = v2.detect_econ_relabels(con)
        if relabel_df.empty:
            raise RuntimeError("No relabel events found. Check IPEDS data path.")
        os.makedirs(os.path.dirname(relabels_path), exist_ok=True)
        relabel_df.to_parquet(relabels_path, index=False)
        print(f"  Saved relabel panel → {relabels_path}")

    treated_events = relabel_df[relabel_df["event_flag"] == 1]
    n_inst = treated_events["unitid"].nunique()
    n_events = len(treated_events)
    print(f"  Relabel events: {n_events} events at {n_inst} institutions")
    print(f"  Year range: {treated_events['relabel_year'].min()} – {treated_events['relabel_year'].max()}")
    print(f"  By type:\n{treated_events.groupby('relabel_type').size().to_string()}")

    # ── Aggregate FOIA/IPEDS plots from v2 (reuse existing functions) ─────────
    print("  Generating aggregate FOIA/IPEDS plots...")
    try:
        # Histogram of relabel years
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        sns.histplot(
            data=treated_events,
            x="relabel_year",
            bins=range(
                int(treated_events["relabel_year"].min()),
                int(treated_events["relabel_year"].max()) + 2,
            ),
            discrete=True,
            hue="relabel_type",
            multiple="dodge",
            ax=ax_hist,
        )
        ax_hist.set_xlabel("Relabel year")
        ax_hist.set_ylabel("Count of relabel events")
        fig_hist.tight_layout()
        _save_and_show(fig_hist, "relabel_year_histogram")

        # OPT usage event-time plots using v2 functions
        opt_usage = v2.compute_opt_usage(con, relabel_df)
        if not opt_usage.empty:
            opt_usage_event = v2.compute_opt_usage_event_time(opt_usage)

            # Plot OPT usage (treated only) for several yvars
            for yvar in ["opt_share", "opt_stem_share", "avg_tuition"]:
                try:
                    fig_path = v2.plot_opt_usage(
                        opt_usage, yvar=yvar, show=True, save=True
                    )
                    if fig_path:
                        print(f"    saved OPT plot ({yvar}) → {fig_path}")
                except Exception as e:
                    print(f"    Warning: opt_usage plot ({yvar}) failed: {e}")

            # Treated vs. physical-sciences control
            try:
                ctrl_phys = v2.compute_control_opt_usage_event_time(con, relabel_df)
                v2.plot_opt_usage_event_time_with_control_label(
                    opt_usage_event=opt_usage_event,
                    control_event=ctrl_phys,
                    control_label="Physical Sciences",
                    yvar="opt_share",
                    show=True,
                    save=True,
                    file_tag="physical_sciences",
                    make_treated_only_plot=True,
                )
            except Exception as e:
                print(f"    Warning: physical-sciences control plot failed: {e}")

            # Treated vs. never-treated econ control
            try:
                ctrl_never = v2.compute_never_treated_econ_control_event_time(con, relabel_df)
                v2.plot_opt_usage_event_time_with_control_label(
                    opt_usage_event=opt_usage_event,
                    control_event=ctrl_never,
                    control_label="Never-treated Economics",
                    yvar="opt_share",
                    show=True,
                    save=True,
                    file_tag="never_treated_econ",
                    make_treated_only_plot=False,
                )
            except Exception as e:
                print(f"    Warning: never-treated econ control plot failed: {e}")
        else:
            print("    Warning: OPT usage data is empty; skipping aggregate plots.")
    except Exception as e:
        print(f"  Warning: aggregate plots failed ({e}); continuing.")

    print(f"  Step 1 done in {_elapsed(t)}")
    return relabel_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Load stage-04 match-ready education sample
# ─────────────────────────────────────────────────────────────────────────────

def _stage04_grad_year_expr(alias: str = "mr") -> str:
    degree_expr = f"LOWER(COALESCE(CAST({alias}.degree_clean AS VARCHAR), ''))"
    start_year_expr = f"CAST(EXTRACT(YEAR FROM TRY_CAST({alias}.ed_startdate AS DATE)) AS INTEGER)"
    end_year_expr = f"CAST(EXTRACT(YEAR FROM TRY_CAST({alias}.ed_enddate AS DATE)) AS INTEGER)"
    return f"""
        CASE
            WHEN TRY_CAST({alias}.ed_enddate AS DATE) IS NOT NULL THEN {end_year_expr}
            WHEN TRY_CAST({alias}.ed_startdate AS DATE) IS NULL THEN NULL::INTEGER
            WHEN {degree_expr} IN ('master', 'masters', 'mba', 'associate', 'associates')
                THEN {start_year_expr} + 2
            WHEN {degree_expr} IN ('doctor', 'doctors', 'doctoral', 'phd', 'ph.d', 'bachelor', 'bachelors')
                THEN {start_year_expr} + 4
            ELSE {start_year_expr} + 4
        END
    """


def step2_prepare_stage04_samples(con: ddb.DuckDBPyConnection) -> list[str]:
    """Build stage-04-based sample views used by all analysis variants."""
    t = time.time()
    print("\n── Step 2: Loading stage-04/stage-05 sample inputs ──────────────────")

    requested_variants = list(cfg.BUILD_SAMPLE_VARIANTS or ["stage04_all"])
    invalid_variants = sorted(set(requested_variants) - SUPPORTED_SAMPLE_VARIANTS)
    if invalid_variants:
        raise ValueError(f"Unsupported sample_variants: {invalid_variants}")

    stage04_path = cfg.STAGE04_MERGE_READY_PARQUET
    if not stage04_path or not os.path.exists(stage04_path):
        raise FileNotFoundError(f"Stage-04 merge_ready parquet not found: {stage04_path}")

    stage04_path_sql = _escape_sql_literal(stage04_path)
    raw_cols = {
        row[0].lower()
        for row in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{stage04_path_sql}')"
        ).fetchall()
    }
    required_cols = {
        "user_id",
        "education_number",
        "unitid",
        "degree_clean",
        "cip",
        "university_raw",
        "field_clean",
        "ed_startdate",
        "ed_enddate",
        "school_match_score",
    }
    missing_cols = sorted(required_cols - raw_cols)
    if missing_cols:
        raise ValueError(f"Stage-04 merge_ready missing required columns: {missing_cols}")

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW stage04_merge_ready_raw AS
        SELECT
            user_id,
            education_number,
            unitid,
            degree_clean,
            cip,
            university_raw,
            field_clean,
            ed_startdate,
            ed_enddate,
            school_match_score
        FROM read_parquet('{stage04_path_sql}')
        """
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW stage04_educ_base AS
        SELECT DISTINCT
            CAST(mr.user_id AS BIGINT) AS user_id,
            CAST(mr.education_number AS BIGINT) AS education_number,
            CAST(mr.unitid AS BIGINT) AS unitid,
            CAST(mr.degree_clean AS VARCHAR) AS degree_clean,
            CAST(mr.cip AS VARCHAR) AS cip,
            CAST(mr.university_raw AS VARCHAR) AS university_raw,
            CAST(mr.field_clean AS VARCHAR) AS field_clean,
            TRY_CAST(mr.ed_startdate AS DATE) AS ed_startdate,
            TRY_CAST(mr.ed_enddate AS DATE) AS ed_enddate,
            TRY_CAST(mr.school_match_score AS DOUBLE) AS school_match_score,
            {_stage04_grad_year_expr("mr")} AS grad_year
        FROM stage04_merge_ready_raw AS mr
        WHERE mr.user_id IS NOT NULL
        """
    )

    cip_where = _cip_prefix_where_clause("cip")
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW stage04_sample_all AS
        SELECT *
        FROM stage04_educ_base
        WHERE unitid IS NOT NULL
          AND {cip_where}
          AND grad_year IS NOT NULL
        """
    )

    stats = con.sql(
        f"""
        SELECT
            COUNT(*) AS rows_total,
            COUNT(DISTINCT user_id) AS users_total,
            COUNT(*) FILTER (WHERE unitid IS NOT NULL) AS rows_with_unitid,
            COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where}) AS rows_after_cip_filter,
            COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where} AND grad_year IS NULL) AS rows_missing_grad_year,
            COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where} AND grad_year IS NOT NULL) AS rows_stage04_all,
            COUNT(DISTINCT user_id) FILTER (WHERE unitid IS NOT NULL AND {cip_where} AND grad_year IS NOT NULL)
                AS users_stage04_all
        FROM stage04_educ_base
        """
    ).fetchone()
    if stats is not None:
        print(
            "  stage04 rows: "
            f"{int(stats[0] or 0):,} rows | {int(stats[1] or 0):,} users"
        )
        print(f"  rows with non-null unitid: {int(stats[2] or 0):,}")
        print(f"  rows after CIP filter:     {int(stats[3] or 0):,}")
        print(f"  rows dropped missing grad_year: {int(stats[4] or 0):,}")
        print(
            "  stage04_all sample:        "
            f"{int(stats[5] or 0):,} rows | {int(stats[6] or 0):,} users"
        )

    stage05_path = cfg.STAGE05_PERSON_BASELINE_PARQUET
    if stage05_path and os.path.exists(stage05_path):
        stage05_path_sql = _escape_sql_literal(stage05_path)
        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW stage05_person_baseline_raw AS
            SELECT * FROM read_parquet('{stage05_path_sql}')
            """
        )
        stage05_cols = {
            row[0].lower()
            for row in con.sql("DESCRIBE stage05_person_baseline_raw").fetchall()
        }
        if "user_id" not in stage05_cols:
            raise ValueError("Stage-05 person baseline parquet must contain user_id")
        rank_filter = ""
        if "person_match_rank" in stage05_cols:
            rank_filter = "AND CAST(person_match_rank AS BIGINT) = 1"
        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW stage05_person_baseline_users AS
            SELECT DISTINCT CAST(user_id AS BIGINT) AS user_id
            FROM stage05_person_baseline_raw
            WHERE user_id IS NOT NULL
              {rank_filter}
            """
        )
        con.sql(
            """
            CREATE OR REPLACE TEMP VIEW stage04_sample_foia_linked_person_baseline AS
            SELECT s.*
            FROM stage04_sample_all AS s
            JOIN stage05_person_baseline_users AS u
              ON s.user_id = u.user_id
            """
        )
        overlap_stats = con.sql(
            """
            SELECT
                (SELECT COUNT(*) FROM stage05_person_baseline_users) AS baseline_users,
                (SELECT COUNT(DISTINCT user_id) FROM stage04_sample_all) AS stage04_users,
                (SELECT COUNT(DISTINCT s.user_id)
                 FROM stage04_sample_all AS s
                 JOIN stage05_person_baseline_users AS u USING (user_id)) AS overlap_users,
                (SELECT COUNT(*) FROM stage04_sample_foia_linked_person_baseline) AS overlap_rows
            """
        ).fetchone()
        if overlap_stats is not None:
            print(
                "  FOIA-linked overlap: "
                f"{int(overlap_stats[2] or 0):,} overlapping users "
                f"({int(overlap_stats[3] or 0):,} education rows) "
                f"vs {int(overlap_stats[0] or 0):,} stage-05 users"
            )
    elif "foia_linked_person_baseline" in requested_variants:
        raise FileNotFoundError(
            f"Stage-05 person baseline parquet not found: {stage05_path}"
        )
    else:
        print("  Stage-05 person baseline parquet not found; FOIA-linked diagnostics skipped")

    print(f"  Step 2 done in {_elapsed(t)}")
    return requested_variants


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Match individuals to relabel events
# ─────────────────────────────────────────────────────────────────────────────

def _match_individuals_to_events(
    con: ddb.DuckDBPyConnection,
    sample_view: str,
    events_df: pd.DataFrame,
    *,
    treated_ind: int,
    group_label: str,
) -> pd.DataFrame:
    columns = [
        "user_id",
        "unitid",
        "education_number",
        "ed_enddate",
        "grad_year",
        "relabel_year",
        "relabel_type",
        "cohort_t",
        "treated_ind",
    ]
    if events_df.empty:
        return pd.DataFrame(columns=columns)
    if not _table_exists(con, sample_view):
        raise ValueError(f"Sample view not found: {sample_view}")

    events_view = f"events_{_variant_slug(group_label)}"
    con.register(f"{events_view}_py", events_df)
    con.sql(f"CREATE OR REPLACE TEMP VIEW {events_view} AS SELECT * FROM {events_view}_py")

    year_window = cfg.BUILD_SAMPLE_GRADYEAR_WINDOW
    return con.sql(
        f"""
        WITH matched AS (
            SELECT
                s.user_id,
                CAST(s.unitid AS BIGINT) AS unitid,
                CAST(s.education_number AS BIGINT) AS education_number,
                TRY_CAST(s.ed_enddate AS DATE) AS ed_enddate,
                CAST(s.grad_year AS INTEGER) AS grad_year,
                CAST(ev.relabel_year AS INTEGER) AS relabel_year,
                ev.relabel_type,
                CAST(s.grad_year AS INTEGER) - CAST(ev.relabel_year AS INTEGER) AS cohort_t
            FROM {sample_view} AS s
            JOIN {events_view} AS ev
              ON CAST(s.unitid AS BIGINT) = CAST(ev.unitid AS BIGINT)
            WHERE ABS(CAST(s.grad_year AS INTEGER) - CAST(ev.relabel_year AS INTEGER)) <= {year_window}
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, relabel_year
                    ORDER BY
                        ABS(cohort_t),
                        CASE WHEN ed_enddate IS NULL THEN 1 ELSE 0 END,
                        ed_enddate DESC,
                        CASE WHEN education_number IS NULL THEN 1 ELSE 0 END,
                        education_number,
                        unitid,
                        relabel_type
                ) AS match_rank
            FROM matched
        )
        SELECT
            user_id,
            unitid,
            education_number,
            ed_enddate,
            grad_year,
            relabel_year,
            relabel_type,
            cohort_t,
            {treated_ind} AS treated_ind
        FROM ranked
        WHERE match_rank = 1
        ORDER BY user_id, relabel_year
        """
    ).df()


def step3_match_treated(
    con: ddb.DuckDBPyConnection,
    relabel_df: pd.DataFrame,
    sample_view: str,
    analysis_variant: str,
    testing_unitids: list[int] | None = None,
) -> pd.DataFrame:
    """
    Join the stage-04 education sample to treated relabel events.
    Returns one deduplicated row per user_id × relabel_year.
    """
    t = time.time()
    print(
        f"\n── Step 3: Matching treated individuals [{analysis_variant}] "
        "────────────────"
    )

    treated_events = relabel_df[relabel_df["event_flag"] == 1][
        ["unitid", "relabel_year", "relabel_type"]
    ].drop_duplicates()
    if testing_unitids is not None:
        treated_events = treated_events[treated_events["unitid"].isin(testing_unitids)].copy()
        print(f"  [test] Restricting to {len(testing_unitids)} sample institutions")

    treated_indiv = _match_individuals_to_events(
        con,
        sample_view,
        treated_events,
        treated_ind=1,
        group_label=f"treated_{analysis_variant}",
    )
    n = len(treated_indiv)
    n_users = treated_indiv["user_id"].nunique() if not treated_indiv.empty else 0
    n_unitids = treated_indiv["unitid"].nunique() if not treated_indiv.empty else 0
    print(
        "  Treated individuals: "
        f"{n:,} (user × relabel_year) | {n_users:,} users | {n_unitids:,} schools"
    )
    print(f"  Step 3 done in {_elapsed(t)}")
    return treated_indiv


def build_control_events(
    con: ddb.DuckDBPyConnection,
    relabel_df: pd.DataFrame,
    testing_unitids: list[int] | None = None,
) -> pd.DataFrame:
    """Assign pseudo relabel years to matched never-treated control schools."""
    t = time.time()
    print("\n── Control Setup: Matching never-treated econ schools ────────────────")

    matched_pairs = v2._match_treated_to_untreated_cohorts(con=con, relabel_df=relabel_df)
    if matched_pairs.empty:
        print("  Warning: no matched control pairs found.")
        return pd.DataFrame(columns=["unitid", "relabel_year", "relabel_type"])

    if testing_unitids is not None:
        matched_pairs = matched_pairs[
            matched_pairs["treated_unitid"].isin(testing_unitids)
        ].copy()
        print(f"  [test] Restricted to {len(matched_pairs)} matched pairs")
    if matched_pairs.empty:
        print("  Warning: no control pairs remain after test filter.")
        return pd.DataFrame(columns=["unitid", "relabel_year", "relabel_type"])

    print(f"  Matched control pairs: {len(matched_pairs):,}")
    print(
        matched_pairs[
            ["relabel_type", "relabel_year", "treated_unitid", "control_unitid"]
        ].head(10).to_string(index=False)
    )

    control_events = matched_pairs.rename(columns={"control_unitid": "unitid"})[
        ["unitid", "relabel_year", "relabel_type"]
    ].drop_duplicates()
    print(
        "  Control pseudo-events: "
        f"{len(control_events):,} rows | {control_events['unitid'].nunique():,} schools"
    )
    print(f"  Control setup done in {_elapsed(t)}")
    return control_events


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Build individual × horizon outcome panel
# ─────────────────────────────────────────────────────────────────────────────

def _latest_available_position_year(
    con: ddb.DuckDBPyConnection,
    pos_view: str,
    startdate_col: str,
    enddate_col: str | None,
) -> int:
    if enddate_col is None:
        query = f"""
            SELECT MAX(obs_year) AS latest_available_year
            FROM (
                SELECT EXTRACT(YEAR FROM TRY_CAST({startdate_col} AS DATE)) AS obs_year
                FROM {pos_view}
                WHERE TRY_CAST({startdate_col} AS DATE) IS NOT NULL
            )
        """
    else:
        query = f"""
            WITH years AS (
                SELECT EXTRACT(YEAR FROM TRY_CAST({startdate_col} AS DATE)) AS obs_year
                FROM {pos_view}
                WHERE TRY_CAST({startdate_col} AS DATE) IS NOT NULL
                UNION ALL
                SELECT EXTRACT(YEAR FROM TRY_CAST({enddate_col} AS DATE)) AS obs_year
                FROM {pos_view}
                WHERE TRY_CAST({enddate_col} AS DATE) IS NOT NULL
            )
            SELECT MAX(obs_year) AS latest_available_year
            FROM years
            WHERE obs_year IS NOT NULL
        """
    latest_available_year = con.sql(query).fetchone()[0]
    if latest_available_year is None:
        raise ValueError("Could not determine latest available year from Revelio positions.")
    return int(latest_available_year)


def step4_build_outcome_panel(
    con: ddb.DuckDBPyConnection,
    indiv_events: pd.DataFrame,
    group_label: str = "treated",
    analysis_variant: str | None = None,
) -> pd.DataFrame:
    """
    For each (user_id, relabel_year), compute outcomes at fixed horizons after graduation.

    Outcomes:
        in_us          : 1 if any active position in US in the evaluation year
        n_pos          : count of distinct active (rcid, startdate) pairs in the evaluation year
        salary_imputed : SUM(total_compensation * frac_of_year_active) in the evaluation year

    Returns long panel with columns:
        user_id, relabel_year, relabel_type, grad_year, cohort_t, horizon_years,
        target_year, eval_year, target_year_observed, used_latest_avail,
        treated_ind, analysis_variant, unitid, education_number,
        in_us, n_pos, salary_imputed
    """
    t = time.time()
    print(
        f"\n── Step 4: Building {group_label} outcome panel"
        + (f" [{analysis_variant}]" if analysis_variant else "")
        + " ──────────────────────"
    )

    if indiv_events.empty:
        print("  No individual-event rows supplied; returning empty panel.")
        return pd.DataFrame(
            columns=[
                "user_id",
                "relabel_year",
                "relabel_type",
                "grad_year",
                "cohort_t",
                "horizon_years",
                "target_year",
                "eval_year",
                "latest_available_year",
                "target_year_observed",
                "used_latest_avail",
                "treated_ind",
                "analysis_variant",
                "unitid",
                "education_number",
                "in_us",
                "n_pos",
                "salary_imputed",
            ]
        )

    pos_path = cfg.REV_POS_PARQUET
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"Revelio positions not found: {pos_path}")

    pos_path_sql = _escape_sql_literal(pos_path)
    con.sql(f"CREATE OR REPLACE TEMP VIEW rev_pos_raw AS SELECT * FROM read_parquet('{pos_path_sql}')")
    pos_cols = [r[0].lower() for r in con.sql("DESCRIBE rev_pos_raw").fetchall()]
    print(f"  rev_pos columns: {pos_cols}")

    # Resolve position column names
    country_col = next((c for c in pos_cols if c == "country"), None)
    startdate_col = next((c for c in pos_cols if c in {"startdate", "start_date", "position_startdate"}), None)
    enddate_col = next((c for c in pos_cols if c in {"enddate", "end_date", "position_enddate"}), None)
    rcid_col = next((c for c in pos_cols if c == "rcid"), None)
    comp_col = next((c for c in pos_cols if c in {"total_compensation", "compensation", "salary"}), None)

    if startdate_col is None:
        raise ValueError(f"Cannot find startdate column in rev_pos. Columns: {pos_cols}")
    if enddate_col is None:
        enddate_date_expr = "NULL::DATE"
        print("  Warning: no enddate column found; treating all positions as open-ended")
    else:
        enddate_date_expr = f"TRY_CAST(p.{enddate_col} AS DATE)"

    # Build compensation expression
    if comp_col:
        comp_expr = f"TRY_CAST(p.{comp_col} AS DOUBLE)"
    else:
        comp_expr = "NULL::DOUBLE"
        print("  Warning: no compensation column found; salary_imputed will be NULL")

    # Build country expression
    if country_col:
        in_us_expr = (
            f"CASE WHEN LOWER(TRIM(CAST(p.{country_col} AS VARCHAR))) = 'united states' "
            "THEN 1 ELSE 0 END"
        )
    else:
        in_us_expr = "0"
        print("  Warning: no country column found; in_us will be 0")

    # rcid expression for distinct position count
    if rcid_col:
        rcid_expr = f"p.{rcid_col}"
    else:
        rcid_expr = "NULL"

    horizons = _analysis_horizons()
    latest_available_year = _latest_available_position_year(
        con,
        "rev_pos_raw",
        startdate_col,
        enddate_col,
    )
    print(f"  horizons after graduation: {horizons}")
    print(f"  latest available position year: {latest_available_year}")

    min_duration_days = max(1, int(getattr(cfg, "BUILD_MIN_POS_DURATION_DAYS", 1)))
    analysis_variant_sql = _escape_sql_literal(analysis_variant or "unspecified")
    horizons_sql = ",\n            ".join(f"({int(h)})" for h in horizons)
    events_view = f"indiv_events_{_variant_slug(group_label)}"

    con.register("indiv_events_py", indiv_events)
    con.sql(f"CREATE OR REPLACE TEMP VIEW {events_view} AS SELECT * FROM indiv_events_py")

    # Evaluate outcomes at fixed horizons after graduation, not at years since relabel.
    # Future target years are flagged so downstream plots / regressions can drop them.
    outcome_panel = con.sql(
        f"""
        WITH horizons (horizon_years) AS (
            VALUES
            {horizons_sql}
        ),
        indiv_horizons AS (
            SELECT
                ie.user_id,
                ie.relabel_year,
                ie.relabel_type,
                ie.grad_year,
                ie.cohort_t,
                h.horizon_years,
                ie.grad_year + h.horizon_years AS target_year,
                CASE
                    WHEN {1 if getattr(cfg, "BUILD_CAP_TO_LATEST_AVAILABLE_YEAR", True) else 0} = 1
                    THEN LEAST(ie.grad_year + h.horizon_years, {latest_available_year})
                    ELSE ie.grad_year + h.horizon_years
                END AS eval_year,
                {latest_available_year} AS latest_available_year,
                CASE
                    WHEN ie.grad_year + h.horizon_years <= {latest_available_year} THEN 1
                    ELSE 0
                END AS target_year_observed,
                CASE
                    WHEN {1 if getattr(cfg, "BUILD_CAP_TO_LATEST_AVAILABLE_YEAR", True) else 0} = 1
                     AND ie.grad_year + h.horizon_years > {latest_available_year} THEN 1
                    ELSE 0
                END AS used_latest_avail,
                ie.treated_ind,
                '{analysis_variant_sql}' AS analysis_variant,
                COALESCE(CAST(ie.unitid AS BIGINT), -1) AS unitid,
                COALESCE(CAST(ie.education_number AS BIGINT), -1) AS education_number
            FROM {events_view} ie
            CROSS JOIN horizons h
        ),
        pos_overlap AS (
            SELECT
                ih.user_id,
                ih.relabel_year,
                ih.relabel_type,
                ih.grad_year,
                ih.cohort_t,
                ih.horizon_years,
                ih.target_year,
                ih.eval_year,
                ih.latest_available_year,
                ih.target_year_observed,
                ih.used_latest_avail,
                ih.treated_ind,
                ih.analysis_variant,
                ih.unitid,
                ih.education_number,
                {in_us_expr} AS is_us_pos,
                COALESCE({rcid_expr}, -1) AS pos_rcid,
                TRY_CAST(p.{startdate_col} AS DATE) AS pos_startdate,
                GREATEST(0,
                    (LEAST(
                        COALESCE({enddate_date_expr}, DATE '9999-12-31'),
                        MAKE_DATE(ih.eval_year, 12, 31)
                    ) - GREATEST(
                        TRY_CAST(p.{startdate_col} AS DATE),
                        MAKE_DATE(ih.eval_year, 1, 1)
                    ) + 1)
                ) AS overlap_days,
                {comp_expr} AS comp
            FROM indiv_horizons ih
            JOIN rev_pos_raw p
              ON p.user_id = ih.user_id
            WHERE TRY_CAST(p.{startdate_col} AS DATE) IS NOT NULL
              AND TRY_CAST(p.{startdate_col} AS DATE) <= MAKE_DATE(ih.eval_year, 12, 31)
              AND (
                {enddate_date_expr} IS NULL
                OR {enddate_date_expr} >= MAKE_DATE(ih.eval_year, 1, 1)
              )
        ),
        pos_active AS (
            SELECT
                *,
                overlap_days / 365.0 AS frac_year
            FROM pos_overlap
            WHERE overlap_days >= {min_duration_days}
        ),
        agg AS (
            SELECT
                user_id,
                relabel_year,
                relabel_type,
                grad_year,
                cohort_t,
                horizon_years,
                target_year,
                eval_year,
                latest_available_year,
                target_year_observed,
                used_latest_avail,
                treated_ind,
                analysis_variant,
                unitid,
                education_number,
                MAX(is_us_pos)                                AS in_us,
                COUNT(DISTINCT (pos_rcid, pos_startdate))     AS n_pos,
                SUM(COALESCE(comp, 0) * frac_year)            AS salary_imputed_raw
            FROM pos_active
            GROUP BY user_id, relabel_year, relabel_type, grad_year, cohort_t,
                     horizon_years, target_year, eval_year, latest_available_year,
                     target_year_observed, used_latest_avail, treated_ind,
                     analysis_variant, unitid, education_number
        ),
        full_panel AS (
            SELECT
                ih.user_id,
                ih.relabel_year,
                ih.relabel_type,
                ih.grad_year,
                ih.cohort_t,
                ih.horizon_years,
                ih.target_year,
                ih.eval_year,
                ih.latest_available_year,
                ih.target_year_observed,
                ih.used_latest_avail,
                ih.treated_ind,
                ih.analysis_variant,
                ih.unitid,
                ih.education_number,
                COALESCE(a.in_us, 0)             AS in_us,
                COALESCE(a.n_pos, 0)             AS n_pos,
                CASE
                    WHEN a.salary_imputed_raw IS NOT NULL AND a.salary_imputed_raw > 0
                    THEN a.salary_imputed_raw
                    ELSE NULL
                END AS salary_imputed
            FROM indiv_horizons ih
            LEFT JOIN agg a
              ON  a.user_id             = ih.user_id
              AND a.relabel_year        = ih.relabel_year
              AND a.analysis_variant    = ih.analysis_variant
              AND a.unitid              = ih.unitid
              AND a.education_number    = ih.education_number
              AND a.horizon_years       = ih.horizon_years
        )
        SELECT * FROM full_panel
        ORDER BY user_id, relabel_year, treated_ind, unitid, horizon_years
        """
    ).df()

    n = len(outcome_panel)
    n_users = outcome_panel["user_id"].nunique()
    print(f"  Outcome panel ({group_label}): {n:,} rows | {n_users:,} users")
    observed_panel = (
        outcome_panel[outcome_panel["target_year_observed"] == 1].copy()
        if "target_year_observed" in outcome_panel.columns
        else outcome_panel
    )
    print(f"  observed rows:      {len(observed_panel):,}")
    print(f"  in_us mean:         {observed_panel['in_us'].mean():.3f}" if not observed_panel.empty else "  in_us mean:         nan")
    print(f"  n_pos mean:         {observed_panel['n_pos'].mean():.2f}" if not observed_panel.empty else "  n_pos mean:         nan")
    sal = observed_panel["salary_imputed"].dropna() if not observed_panel.empty else pd.Series(dtype=float)
    if len(sal) > 0:
        print(f"  salary_imputed mean: {sal.mean():,.0f}  (non-null: {len(sal):,} rows)")
    else:
        print("  salary_imputed: all NULL (no compensation data in positions)")
    print(f"  Step 4 done in {_elapsed(t)}")
    return outcome_panel


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Event study aggregation + plots (treated only)
# ─────────────────────────────────────────────────────────────────────────────

def step5_event_study_plots(
    panel: pd.DataFrame,
    label: str = "treated",
    analysis_variant: str | None = None,
) -> None:
    """Aggregate by cohort_t and plot one figure per outcome."""
    t = time.time()
    print(
        f"\n── Step 5: Event study plots ({label})"
        + (f" [{analysis_variant}]" if analysis_variant else "")
        + " ──────────────────────────────"
    )

    agg = _agg_cohort_time(panel)
    window = cfg.BUILD_EVENT_WINDOW
    agg = agg[agg["cohort_t"].between(-window, window)]
    if agg.empty:
        print("  No observed outcome rows remain after cohort-window filter.")
        return

    horizons = sorted(agg["horizon_years"].dropna().astype(int).unique().tolist())
    palette = sns.color_palette("deep", n_colors=max(1, len(horizons)))
    if "target_year_observed" in panel.columns:
        n_dropped = int((pd.to_numeric(panel["target_year_observed"], errors="coerce") == 0).sum())
        if n_dropped:
            print(f"  dropped {n_dropped:,} future target-year rows before aggregation")

    for outcome in OUTCOMES:
        mean_col = f"{outcome}_mean"
        se_col = f"{outcome}_se"
        if mean_col not in agg.columns:
            print(f"  Skipping {outcome} (not in panel)")
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        for color, (horizon, grp) in zip(palette, agg.groupby("horizon_years", sort=True)):
            line_df = _coerce_plot_frame(grp, ["cohort_t", mean_col], sort_col="cohort_t")
            if line_df.empty:
                continue
            ax.plot(
                line_df["cohort_t"].to_numpy(dtype=float),
                line_df[mean_col].to_numpy(dtype=float),
                marker="o",
                linewidth=2,
                color=color,
                label=f"{int(horizon)} years after grad",
            )
            valid_ci = grp[se_col].notna() & (pd.to_numeric(grp[se_col], errors="coerce") > 0)
            if valid_ci.any():
                ci_df = _coerce_plot_frame(
                    grp.loc[valid_ci],
                    ["cohort_t", mean_col, se_col],
                    sort_col="cohort_t",
                )
                ax.fill_between(
                    ci_df["cohort_t"].to_numpy(dtype=float),
                    ci_df[mean_col].to_numpy(dtype=float) - 1.96 * ci_df[se_col].to_numpy(dtype=float),
                    ci_df[mean_col].to_numpy(dtype=float) + 1.96 * ci_df[se_col].to_numpy(dtype=float),
                    alpha=0.20,
                    color=color,
                )
        ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("Graduation year relative to relabel event")
        ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome))
        ax.set_title("")
        if len(horizons) > 1:
            ax.legend(title=None, fontsize=10)
        fig.tight_layout()
        _save_and_show(fig, f"{outcome}_event_study_{label}", analysis_variant=analysis_variant)
        print(f"  n per horizon × cohort_t:\n{agg[['horizon_years', 'cohort_t', 'n']].to_string(index=False)}")

    print(f"  Step 5 done in {_elapsed(t)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Control group (never-treated econ institutions)
# ─────────────────────────────────────────────────────────────────────────────

def step6_control_group(
    con: ddb.DuckDBPyConnection,
    sample_view: str,
    control_events: pd.DataFrame,
    analysis_variant: str,
) -> pd.DataFrame:
    """
    Build never-treated control individuals from the same stage-04 education sample.
    Returns control outcome panel (treated_ind=0).
    """
    t = time.time()
    print(f"\n── Step 6: Building control group [{analysis_variant}] ───────────────")

    if control_events.empty:
        print("  Warning: control pseudo-events are empty; skipping control group.")
        return pd.DataFrame()

    control_indiv = _match_individuals_to_events(
        con,
        sample_view,
        control_events,
        treated_ind=0,
        group_label=f"control_{analysis_variant}",
    )

    if control_indiv.empty:
        print("  Warning: no Revelio individuals found at control institutions.")
        return pd.DataFrame()

    n = len(control_indiv)
    n_users = control_indiv["user_id"].nunique()
    n_unitids = control_indiv["unitid"].nunique()
    print(
        "  Control individuals: "
        f"{n:,} (user × relabel_year) | {n_users:,} users | {n_unitids:,} schools"
    )

    control_panel = step4_build_outcome_panel(
        con,
        control_indiv,
        group_label=f"control_{_variant_slug(analysis_variant)}",
        analysis_variant=analysis_variant,
    )
    print(f"  Step 6 done in {_elapsed(t)}")
    return control_panel


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 – Treated vs. control event study plots
# ─────────────────────────────────────────────────────────────────────────────

def step7_treated_vs_control_plots(
    treated_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    analysis_variant: str | None = None,
) -> None:
    """Plot treated and control cohort series together for each outcome."""
    t = time.time()
    print(
        "\n── Step 7: Treated vs. control event study plots"
        + (f" [{analysis_variant}]" if analysis_variant else "")
        + " ─────────────────────"
    )

    if control_panel.empty:
        print("  Control panel is empty; skipping treated-vs-control plots.")
        return

    window = cfg.BUILD_EVENT_WINDOW

    treated_agg = _agg_cohort_time(treated_panel)
    treated_agg = treated_agg[treated_agg["cohort_t"].between(-window, window)].copy()
    treated_agg["series"] = "Treated (Econ→Econometrics)"

    ctrl_agg = _agg_cohort_time(control_panel)
    ctrl_agg = ctrl_agg[ctrl_agg["cohort_t"].between(-window, window)].copy()
    ctrl_agg["series"] = "Control (Never-treated Econ)"
    if treated_agg.empty or ctrl_agg.empty:
        print("  No overlapping observed rows remain after cohort-window filter.")
        return

    colors = {"Treated (Econ→Econometrics)": "#2e8b57", "Control (Never-treated Econ)": "#e07a5f"}
    horizons = sorted(
        set(treated_agg["horizon_years"].dropna().astype(int).unique().tolist())
        | set(ctrl_agg["horizon_years"].dropna().astype(int).unique().tolist())
    )
    linestyles = ["-", "--", "-.", ":"]

    for outcome in OUTCOMES:
        mean_col = f"{outcome}_mean"
        se_col = f"{outcome}_se"
        if mean_col not in treated_agg.columns:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        for agg_df in [treated_agg, ctrl_agg]:
            if mean_col not in agg_df.columns:
                continue
            label = agg_df["series"].iloc[0]
            color = colors.get(label, "#4c78a8")
            for idx, (horizon, grp) in enumerate(agg_df.groupby("horizon_years", sort=True)):
                line_df = _coerce_plot_frame(grp, ["cohort_t", mean_col], sort_col="cohort_t")
                if line_df.empty:
                    continue
                line_label = label
                if len(horizons) > 1:
                    line_label = f"{label}, {int(horizon)}yr"
                ax.plot(
                    line_df["cohort_t"].to_numpy(dtype=float),
                    line_df[mean_col].to_numpy(dtype=float),
                    marker="o",
                    linewidth=2,
                    label=line_label,
                    color=color,
                    linestyle=linestyles[idx % len(linestyles)],
                )
                valid_ci = grp[se_col].notna() & (pd.to_numeric(grp[se_col], errors="coerce") > 0)
                if valid_ci.any():
                    ci_df = _coerce_plot_frame(
                        grp.loc[valid_ci],
                        ["cohort_t", mean_col, se_col],
                        sort_col="cohort_t",
                    )
                    ax.fill_between(
                        ci_df["cohort_t"].to_numpy(dtype=float),
                        ci_df[mean_col].to_numpy(dtype=float) - 1.96 * ci_df[se_col].to_numpy(dtype=float),
                        ci_df[mean_col].to_numpy(dtype=float) + 1.96 * ci_df[se_col].to_numpy(dtype=float),
                        alpha=0.12,
                        color=color,
                    )

        ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("Graduation year relative to relabel event")
        ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome))
        ax.set_title("")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(title=None, fontsize=10)
        fig.tight_layout()
        _save_and_show(
            fig,
            f"{outcome}_event_study_treated_vs_control",
            analysis_variant=analysis_variant,
        )

    print(f"  Step 7 done in {_elapsed(t)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 – Staggered DiD
# ─────────────────────────────────────────────────────────────────────────────

def step8_did(
    treated_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    analysis_variant: str | None = None,
) -> pd.DataFrame:
    """
    Run cohort-based DiD regressions on the treated/control outcome panel.

    For each configured post-graduation horizon, estimate:
        outcome ~ C(cohort_t, Treatment(reference=-1)):treated_ind
                + C(unitid) + C(grad_year) + C(relabel_year)
    with school-clustered SEs.
    """
    t = time.time()
    print(
        "\n── Step 8: Staggered DiD"
        + (f" [{analysis_variant}]" if analysis_variant else "")
        + " ─────────────────────────────────────────────"
    )

    if not cfg.BUILD_RUN_DID:
        print("  Skipping DiD (run_did=false in config).")
        return pd.DataFrame()
    if control_panel.empty:
        print("  Skipping DiD: control panel is empty.")
        return pd.DataFrame()

    combined = pd.concat([treated_panel, control_panel], ignore_index=True)
    if "target_year_observed" in combined.columns:
        n_before = len(combined)
        combined = combined[pd.to_numeric(combined["target_year_observed"], errors="coerce") == 1].copy()
        n_dropped = n_before - len(combined)
        if n_dropped:
            print(f"  dropped {n_dropped:,} future target-year rows before DiD")
    combined = combined.dropna(
        subset=["cohort_t", "treated_ind", "relabel_year", "grad_year", "horizon_years"]
    ).copy()
    combined["cohort_t"] = pd.to_numeric(combined["cohort_t"], errors="coerce")
    combined["relabel_year"] = pd.to_numeric(combined["relabel_year"], errors="coerce")
    combined["grad_year"] = pd.to_numeric(combined["grad_year"], errors="coerce")
    combined["horizon_years"] = pd.to_numeric(combined["horizon_years"], errors="coerce")
    combined = combined.dropna(subset=["cohort_t", "relabel_year", "grad_year", "horizon_years"])
    combined["cohort_t"] = combined["cohort_t"].astype(int)
    combined["relabel_year"] = combined["relabel_year"].astype(int)
    combined["grad_year"] = combined["grad_year"].astype(int)
    combined["horizon_years"] = combined["horizon_years"].astype(int)

    window = cfg.BUILD_EVENT_WINDOW
    combined = combined[combined["cohort_t"].between(-window, window)].copy()
    if combined.empty:
        print("  DiD panel is empty after cohort-window filter; skipping.")
        return pd.DataFrame()

    unitid_series = (
        combined["unitid"]
        if "unitid" in combined.columns
        else pd.Series(-1, index=combined.index, dtype="int64")
    )
    education_number_series = (
        combined["education_number"]
        if "education_number" in combined.columns
        else pd.Series(-1, index=combined.index, dtype="int64")
    )
    combined["cluster_unitid"] = pd.to_numeric(unitid_series, errors="coerce").fillna(-1).astype("int64")
    combined["cluster_education_number"] = pd.to_numeric(
        education_number_series, errors="coerce"
    ).fillna(-1).astype("int64")
    entity_keys = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "user_id": pd.to_numeric(combined["user_id"], errors="coerce").fillna(-1).astype("int64"),
                "relabel_year": pd.to_numeric(combined["relabel_year"], errors="coerce").fillna(-1).astype("int64"),
                "treated_ind": pd.to_numeric(combined["treated_ind"], errors="coerce").fillna(-1).astype("int64"),
                "horizon_years": pd.to_numeric(combined["horizon_years"], errors="coerce").fillna(-1).astype("int64"),
                "cluster_unitid": combined["cluster_unitid"],
                "cluster_education_number": combined["cluster_education_number"],
            }
        )
    )
    combined["did_entity_id"] = pd.factorize(entity_keys, sort=False)[0].astype("int64")

    print(f"  DiD panel: {len(combined):,} rows | "
          f"{combined['user_id'].nunique():,} users | "
          f"{combined['did_entity_id'].nunique():,} event-entities | "
          f"{combined['cluster_unitid'].nunique():,} schools | "
          f"{combined['horizon_years'].nunique()} horizons")
    print(f"  Treated rows: {combined['treated_ind'].sum():,} | "
          f"Control rows: {(combined['treated_ind'] == 0).sum():,}")
    print("  Omitted reference cohort: cohort_t=-1")
    print(f"  Horizons in DiD: {sorted(combined['horizon_years'].unique().tolist())}")
    print(f"  DiD model setting: {cfg.BUILD_DID_MODEL}")

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        smf = None
    if smf is None:
        print("  statsmodels not available; skipping DiD.")
        return pd.DataFrame()

    models_to_run = _did_models_to_run()
    print("\n  DiD results:")
    results_rows = []
    reference_cohort_t = -1
    available_horizons = sorted(combined["horizon_years"].dropna().astype(int).unique().tolist())
    for outcome in OUTCOMES:
        if outcome not in combined.columns:
            continue

        base = combined.dropna(subset=[outcome]).copy()
        if len(base) < 5:
            print(f"  {outcome}: too few obs ({len(base)}), skipping")
            continue
        if base["treated_ind"].nunique() < 2:
            print(f"  {outcome}: insufficient treated/control variation, skipping")
            continue

        for horizon in available_horizons:
            sub = base[base["horizon_years"] == horizon].copy()
            if len(sub) < 5:
                continue
            supported_cohorts = _supported_did_cohorts(
                sub,
                reference_cohort_t=reference_cohort_t,
            )
            if not supported_cohorts:
                print(
                    f"  {outcome} [h={int(horizon)}]: no supported treated/control cohort_t coefficients, skipping"
                )
                continue

            cohort_values = sorted({reference_cohort_t, *supported_cohorts})
            sub = sub[sub["cohort_t"].isin(cohort_values)].copy()
            treated_ref_mean = sub.loc[
                (sub["treated_ind"] == 1) & (sub["cohort_t"] == reference_cohort_t),
                outcome,
            ].mean()
            control_ref_mean = sub.loc[
                (sub["treated_ind"] == 0) & (sub["cohort_t"] == reference_cohort_t),
                outcome,
            ].mean()

            for did_model in models_to_run:
                try:
                    formula = (
                        f"{outcome} ~ "
                        f"C(cohort_t, Treatment(reference={reference_cohort_t})):treated_ind "
                        "+ C(cluster_unitid) + C(grad_year) + C(relabel_year)"
                    )
                    if sub["cluster_unitid"].nunique() >= 2:
                        try:
                            result = smf.ols(formula=formula, data=sub).fit(
                                cov_type="cluster",
                                cov_kwds={"groups": sub["cluster_unitid"]},
                            )
                            cov = result.cov_params()
                        except Exception as exc:
                            print(
                                f"    {outcome} [{did_model}] h={int(horizon)}: "
                                f"clustered SE failed ({exc}); falling back to HC1"
                            )
                            result = smf.ols(formula=formula, data=sub).fit(cov_type="HC1")
                            cov = result.cov_params()
                    else:
                        result = smf.ols(formula=formula, data=sub).fit(cov_type="HC1")
                        cov = result.cov_params()
                    n_obs = int(result.nobs)
                    ref_param = _find_did_interaction_param(
                        result.params,
                        cohort_t=reference_cohort_t,
                        reference_cohort_t=reference_cohort_t,
                    )
                    ref_coef = float(result.params[ref_param]) if ref_param is not None else 0.0

                    for cohort_t in supported_cohorts:
                        param = _find_did_interaction_param(
                            result.params,
                            cohort_t=cohort_t,
                            reference_cohort_t=reference_cohort_t,
                        )
                        if param is None:
                            continue
                        raw_coef = float(result.params.get(param, float("nan")))
                        if ref_param is None:
                            coef = raw_coef
                            var = float(cov.loc[param, param])
                        else:
                            coef = raw_coef - ref_coef
                            var = float(
                                cov.loc[param, param]
                                + cov.loc[ref_param, ref_param]
                                - 2 * cov.loc[param, ref_param]
                            )
                        se = float(max(var, 0.0) ** 0.5)
                        pval = _normal_pvalue_from_coef_se(coef, se)
                        treated_event_mean = sub.loc[
                            (sub["treated_ind"] == 1) & (sub["cohort_t"] == cohort_t),
                            outcome,
                        ].mean()
                        control_event_mean = sub.loc[
                            (sub["treated_ind"] == 0) & (sub["cohort_t"] == cohort_t),
                            outcome,
                        ].mean()
                        stars = (
                            "***" if pd.notna(pval) and pval < 0.01 else
                            "**" if pd.notna(pval) and pval < 0.05 else
                            "*" if pd.notna(pval) and pval < 0.10 else ""
                        )
                        print(
                            f"    {outcome:<20s} [{did_model:<6s}] "
                            f"h={int(horizon):>2d}  cohort_t={cohort_t:+d}  "
                            f"coef={coef:>10.4f}  se={se:>8.4f}  "
                            f"p={pval:.3f} {stars}  n={n_obs:,}"
                        )
                        results_rows.append(
                            {
                                "analysis_variant": analysis_variant or "unspecified",
                                "did_model": did_model,
                                "outcome": outcome,
                                "horizon_years": int(horizon),
                                "cohort_t": int(cohort_t),
                                "reference_cohort_t": reference_cohort_t,
                                "coef": coef,
                                "se": se,
                                "pval": pval,
                                "ci_lower": coef - 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan,
                                "ci_upper": coef + 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan,
                                "n_obs": n_obs,
                                "n_users": int(sub["user_id"].nunique()),
                                "n_entities": int(sub["did_entity_id"].nunique()),
                                "n_unitids": int(sub["cluster_unitid"].nunique()),
                                "treated_ref_mean": treated_ref_mean,
                                "control_ref_mean": control_ref_mean,
                                "treated_event_mean": treated_event_mean,
                                "control_event_mean": control_event_mean,
                            }
                        )
                except Exception as e:
                    print(f"    {outcome} [{did_model}] h={int(horizon)}: DiD failed ({e})")

    results_df = (
        pd.DataFrame(results_rows)
        .sort_values(["did_model", "outcome", "horizon_years", "cohort_t"])
        .reset_index(drop=True)
        if results_rows
        else pd.DataFrame()
    )
    if not results_df.empty:
        _plot_did_coefficients(results_df, analysis_variant=analysis_variant)
    else:
        print("  No DiD estimates were produced.")

    print("\n  Cohort-specific treated series (aggregated by relabel_year):")
    cohort_agg = _agg_cohort_time(
        treated_panel[treated_panel["cohort_t"].between(-window, window)],
        group_col="relabel_year",
    )
    if "relabel_year" in cohort_agg.columns:
        for outcome in OUTCOMES:
            mean_col = f"{outcome}_mean"
            if mean_col not in cohort_agg.columns:
                continue
            for horizon in available_horizons:
                horizon_grp = cohort_agg[cohort_agg["horizon_years"] == horizon].copy()
                if horizon_grp.empty:
                    continue
                fig, ax = plt.subplots(figsize=(9, 5))
                for cohort_year, grp in horizon_grp.groupby("relabel_year"):
                    line_df = _coerce_plot_frame(grp, ["cohort_t", mean_col], sort_col="cohort_t")
                    if line_df.empty:
                        continue
                    ax.plot(
                        line_df["cohort_t"].to_numpy(dtype=float),
                        line_df[mean_col].to_numpy(dtype=float),
                        marker=".",
                        linewidth=1.5,
                        alpha=0.7,
                        label=str(cohort_year),
                    )
                ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
                ax.set_xlabel("Graduation year relative to relabel event")
                ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome))
                ax.set_title(f"Cohort-specific: {outcome} ({int(horizon)} years after grad)")
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(title="Relabel year", fontsize=8, ncol=2)
                fig.tight_layout()
                _save_and_show(
                    fig,
                    f"{outcome}_cohort_event_time{_horizon_file_suffix(int(horizon), available_horizons)}",
                    analysis_variant=analysis_variant,
                )

    print(f"  Step 8 done in {_elapsed(t)}")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: sample-level inspection
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_sample(
    con: ddb.DuckDBPyConnection,
    relabel_df: pd.DataFrame,
    analysis_variant: str,
    sample_view: str,
    treated_indiv: pd.DataFrame,
    control_events: pd.DataFrame | None = None,
    n_sample_people: int = 3,
) -> None:
    """
    Print stage-04/stage-05 sample diagnostics for a single analysis variant.
    """
    print("\n" + "═" * 70)
    print(f"DIAGNOSTIC: sample inspection [{analysis_variant}]")
    print("═" * 70)
    cip_where = _cip_prefix_where_clause("cip")
    print("\n[1] Stage-04 sample funnel:")
    try:
        funnel = con.sql(
            f"""
            SELECT
                COUNT(*) AS rows_total,
                COUNT(*) FILTER (WHERE unitid IS NOT NULL) AS rows_with_unitid,
                COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where}) AS rows_after_cip_filter,
                COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where} AND grad_year IS NULL) AS rows_missing_grad_year,
                COUNT(*) FILTER (WHERE unitid IS NOT NULL AND {cip_where} AND grad_year IS NOT NULL) AS rows_stage04_all,
                COUNT(*) FILTER (WHERE user_id IN (SELECT user_id FROM {sample_view})) AS rows_variant,
                COUNT(DISTINCT user_id) FILTER (WHERE user_id IN (SELECT user_id FROM {sample_view})) AS users_variant
            FROM stage04_educ_base
            """
        ).df()
        print(funnel.to_string(index=False))
    except Exception as e:
        print(f"  Could not compute stage-04 funnel: {e}")

    print("\n[2] Treated/control school coverage:")
    treated_total = int(
        pd.to_numeric(
            relabel_df.loc[relabel_df["event_flag"] == 1, "unitid"],
            errors="coerce",
        ).dropna().nunique()
    )
    treated_hit = int(treated_indiv["unitid"].nunique()) if not treated_indiv.empty else 0
    print(f"  treated schools with relabel events: {treated_total:,}")
    print(f"  treated schools represented in sample: {treated_hit:,}")
    if control_events is not None and not control_events.empty:
        con.register("control_events_diag_py", control_events)
        con.sql("CREATE OR REPLACE TEMP VIEW control_events_diag AS SELECT * FROM control_events_diag_py")
        try:
            control_hit = int(
                con.sql(
                    f"""
                    SELECT COUNT(DISTINCT s.unitid)
                    FROM {sample_view} AS s
                    JOIN control_events_diag AS c
                      ON s.unitid = CAST(c.unitid AS BIGINT)
                    """
                ).fetchone()[0] or 0
            )
            print(f"  control pseudo-event schools: {control_events['unitid'].nunique():,}")
            print(f"  control schools represented in sample: {control_hit:,}")
        except Exception as e:
            print(f"  Could not compute control coverage: {e}")
    else:
        print("  control pseudo-event schools: 0")

    print("\n[3] FOIA-linked overlap counts:")
    if _table_exists(con, "stage05_person_baseline_users"):
        try:
            overlap = con.sql(
                f"""
                SELECT
                    (SELECT COUNT(*) FROM stage05_person_baseline_users) AS baseline_users,
                    (SELECT COUNT(DISTINCT user_id) FROM stage04_sample_all) AS stage04_all_users,
                    (SELECT COUNT(DISTINCT s.user_id)
                     FROM stage04_sample_all AS s
                     JOIN stage05_person_baseline_users AS u USING (user_id)) AS overlap_users,
                    (SELECT COUNT(DISTINCT user_id) FROM {sample_view}) AS variant_users
                """
            ).df()
            print(overlap.to_string(index=False))
        except Exception as e:
            print(f"  Could not compute FOIA overlap counts: {e}")
    else:
        print("  stage05_person_baseline_users not available")

    print(f"\n[4] Sample matched education histories from `{sample_view}`:")
    if treated_indiv.empty:
        try:
            sample_rows = con.sql(
                f"""
                SELECT user_id, unitid, grad_year, degree_clean, cip, university_raw, field_clean,
                       ed_startdate, ed_enddate
                FROM {sample_view}
                ORDER BY user_id, education_number
                LIMIT 10
                """
            ).df()
            if sample_rows.empty:
                print("  No rows in sample view.")
            else:
                print(sample_rows.to_string(index=False))
        except Exception as e:
            print(f"  Could not query sample view: {e}")
        print("\n" + "═" * 70)
        return

    sample_users = treated_indiv["user_id"].drop_duplicates().sample(
        min(n_sample_people, treated_indiv["user_id"].nunique()),
        random_state=42,
    ).tolist()
    for uid in sample_users:
        matched_row = treated_indiv[treated_indiv["user_id"] == uid].iloc[0]
        print(
            f"\n  ── user_id={uid} unitid={matched_row['unitid']} grad_year={matched_row['grad_year']} "
            f"relabel_year={matched_row['relabel_year']} cohort_t={matched_row['cohort_t']}"
        )
        try:
            educ = con.sql(
                f"""
                SELECT
                    education_number,
                    unitid,
                    degree_clean,
                    cip,
                    university_raw,
                    field_clean,
                    grad_year,
                    ed_startdate,
                    ed_enddate,
                    school_match_score
                FROM stage04_educ_base
                WHERE user_id = {int(uid)}
                ORDER BY ed_startdate, education_number
                """
            ).df()
            print(educ.to_string(index=False, max_rows=10))
        except Exception as e:
            print(f"    Could not fetch education history: {e}")

    print("\n" + "═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_main = time.time()
    con = ddb.connect()
    # Configure DuckDB for performance
    n_threads = max(1, os.cpu_count() or 1)
    con.sql(f"PRAGMA threads={n_threads}")
    con.sql("PRAGMA preserve_insertion_order=false")

    # ── Step 1: Detect relabels + aggregate plots ────────────────────────────
    relabel_df = step1_relabels(con)

    # Testing: optionally restrict to a sample of treated institutions
    testing_unitids: list[int] | None = None
    if cfg.TESTING_ENABLED:
        rng = np.random.default_rng(cfg.TESTING_RANDOM_SEED)
        all_treated_unitids = relabel_df.loc[
            relabel_df["event_flag"] == 1, "unitid"
        ].unique().tolist()
        n_sample = min(cfg.TESTING_SAMPLE_N_INSTITUTIONS, len(all_treated_unitids))
        testing_unitids = sorted(
            [int(x) for x in rng.choice(all_treated_unitids, size=n_sample, replace=False)]
        )
        print(f"\n[TEST MODE] Sampled {n_sample} treated institutions: {testing_unitids}")

    # ── Step 2: Load stage-04/stage-05 sample views ─────────────────────────
    sample_variants = step2_prepare_stage04_samples(con)

    # ── Control pseudo-events (shared across variants) ──────────────────────
    control_events = build_control_events(con, relabel_df, testing_unitids=testing_unitids)

    combined_panels: list[pd.DataFrame] = []
    combined_did_results: list[pd.DataFrame] = []
    for analysis_variant in sample_variants:
        sample_view = _sample_view_name(analysis_variant)
        if not _table_exists(con, sample_view):
            raise ValueError(f"Missing sample view for variant {analysis_variant}: {sample_view}")

        treated_indiv = step3_match_treated(
            con,
            relabel_df,
            sample_view,
            analysis_variant,
            testing_unitids=testing_unitids,
        )
        diagnose_sample(
            con,
            relabel_df,
            analysis_variant,
            sample_view,
            treated_indiv,
            control_events=control_events,
        )

        if treated_indiv.empty:
            print(f"\nNo treated individuals found for variant {analysis_variant}; skipping.")
            continue

        treated_panel = step4_build_outcome_panel(
            con,
            treated_indiv,
            group_label=f"treated_{_variant_slug(analysis_variant)}",
            analysis_variant=analysis_variant,
        )
        step5_event_study_plots(
            treated_panel,
            label="treated",
            analysis_variant=analysis_variant,
        )

        control_panel = step6_control_group(
            con,
            sample_view,
            control_events,
            analysis_variant,
        )
        step7_treated_vs_control_plots(
            treated_panel,
            control_panel,
            analysis_variant=analysis_variant,
        )
        did_results = step8_did(
            treated_panel,
            control_panel,
            analysis_variant=analysis_variant,
        )
        if not did_results.empty:
            combined_did_results.append(did_results)

        if not control_panel.empty:
            variant_panel = pd.concat([treated_panel, control_panel], ignore_index=True)
        else:
            variant_panel = treated_panel.copy()
        combined_panels.append(variant_panel)

    if not combined_panels:
        print("\nNo variant produced a treated sample. Check stage-04/stage-05 inputs.")
        return

    combined = pd.concat(combined_panels, ignore_index=True)

    panel_out = cfg.OUTPUT_PANEL_PARQUET
    os.makedirs(os.path.dirname(panel_out), exist_ok=True)
    combined.to_parquet(panel_out, index=False)
    print(f"\nSaved combined panel → {panel_out}")

    if combined_did_results:
        did_combined = pd.concat(combined_did_results, ignore_index=True)
        did_out = _did_results_output_path()
        did_out.parent.mkdir(parents=True, exist_ok=True)
        did_combined.to_parquet(did_out, index=False)
        did_csv = did_out.with_suffix(".csv")
        did_combined.to_csv(did_csv, index=False)
        print(f"Saved DiD results → {did_out}")
        print(f"Saved DiD results CSV → {did_csv}")
        _plot_did_variant_comparison(did_combined)
    else:
        print("No DiD results saved.")
    print(f"\nTotal time: {_elapsed(t_main)}")


if __name__ == "__main__":
    main()
