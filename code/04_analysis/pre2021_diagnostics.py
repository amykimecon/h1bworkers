"""
pre2021_diagnostics.py
======================
Read-only audit of the pre-2021 firm outcomes pipeline.

Loads each intermediate file and prints key statistics at five stages:
  Stage 1  TRK_12704 → foia_firm_uid crosswalk  (trk12704_to_foiafirm.csv)
  Stage 2  Pre-2021 FOIA parquet                (pre2021_foia.parquet)
  Stage 3  LCA firm-year parquet                (lca_firm_year.parquet)
  Stage 4  LCA-FOIA crosswalk                   (lca_foia_crosswalk.parquet)
  Stage 5  Pre-2021 panel assembly              (replicates firm_outcomes.py logic)

No WRDS queries, no re-running pipeline steps. Reads paths from existing
configs/firm_outcomes.yaml and configs/build_pre2021_foia.yaml.

Usage
-----
  run 04_analysis/pre2021_diagnostics.py   # in iPython
  python 04_analysis/pre2021_diagnostics.py
"""

import sys
import time
from pathlib import Path

import matplotlib
try:
    from IPython import get_ipython as _get_ipython
    _IN_IPYTHON = _get_ipython() is not None
except Exception:
    _IN_IPYTHON = False
if not _IN_IPYTHON:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
if "__file__" in globals():
    _THIS_DIR = Path(__file__).resolve().parent
else:
    _THIS_DIR = Path.cwd() / "04_analysis"

_CODE_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_CODE_DIR))
from config import root  # noqa: E402

# ---------------------------------------------------------------------------
# Load configs
# ---------------------------------------------------------------------------

def _expand(obj):
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return obj.replace("{root}", str(root))
    return obj


def _load_cfg(name):
    path = _CODE_DIR / "configs" / name
    return _expand(yaml.safe_load(path.read_text()) or {})


FO_CFG  = _load_cfg("firm_outcomes.yaml")          # firm_outcomes config
PRE_CFG = _load_cfg("build_pre2021_foia.yaml")     # build_pre2021_foia config

# Key paths
TRK_CROSSWALK   = PRE_CFG["trk_crosswalk"]
PRE21_FOIA      = FO_CFG.get("foia_pre2021_parquet", "")
LCA_FY          = FO_CFG["pre2021_lca_proxy"]["lca_parquet"]
LCA_XWALK       = FO_CFG["pre2021_lca_proxy"]["lca_crosswalk_parquet"]
PROXY_COL       = FO_CFG["pre2021_lca_proxy"].get("proxy_col", "n_lca_workers")
MAX_YEAR        = int(FO_CFG["pre2021_lca_proxy"].get("max_lottery_year", 2020))
OUTPUT_DIR      = Path(FO_CFG["output_dir"]) / "debug"

_SEP = "=" * 70


def _section(title):
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _check_exists(path, label):
    p = Path(path)
    if not p.exists():
        print(f"  [SKIP] {label} not found: {path}")
        return False
    size_mb = p.stat().st_size / 1e6
    print(f"  File: {path}  ({size_mb:.1f} MB)")
    return True


def _pct_table(series, name="n", pcts=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    """Print a percentile row for a numeric series."""
    vals = series.dropna()
    if vals.empty:
        print(f"  {name}: (empty)")
        return
    qs = np.percentile(vals, pcts)
    header = "  " + "  ".join(f"p{p:>3}" for p in pcts)
    row    = "  " + "  ".join(f"{v:>5,.0f}" for v in qs)
    print(f"  {name} percentiles (n={len(vals):,}):")
    print(header)
    print(row)


# ===========================================================================
# Stage 1: TRK_12704 → foia_firm_uid crosswalk
# ===========================================================================

def stage1_trk_crosswalk():
    _section("STAGE 1: TRK_12704 → foia_firm_uid crosswalk")
    if not _check_exists(TRK_CROSSWALK, "trk_crosswalk"):
        return

    t0 = time.time()
    df = pd.read_csv(TRK_CROSSWALK, dtype=str, low_memory=False)
    print(f"  Rows: {len(df):,}  ({time.time()-t0:.1f}s)")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Unique foia_firm_uid: {df['foia_firm_uid'].nunique():,}")

    # is_new breakdown (True = brand-new UID, False = matched to existing Bloomberg UID)
    if "is_new" in df.columns:
        new_counts = df["is_new"].value_counts(dropna=False)
        print("\n  is_new breakdown (True = new UID added from TRK, False = matched existing):")
        for val, cnt in new_counts.items():
            pct = 100 * cnt / len(df)
            print(f"    {str(val):<10}  {cnt:>8,}  ({pct:.1f}%)")

    # n_apps distribution
    if "n_apps" in df.columns:
        df["n_apps_num"] = pd.to_numeric(df["n_apps"], errors="coerce")
        _pct_table(df["n_apps_num"], name="n_apps (petitions per canonical TRK firm)")

        print("\n  Top 15 firms by n_apps:")
        top = (
            df.dropna(subset=["n_apps_num"])
            .nlargest(15, "n_apps_num")[["pet_firm_name", "pet_state", "n_apps_num", "is_new"]]
        )
        print(top.to_string(index=False))

    # State distribution
    if "pet_state" in df.columns:
        top_states = df["pet_state"].value_counts().head(15)
        print("\n  Top 15 states by n_crosswalk_entries:")
        print(top_states.to_string())


# ===========================================================================
# Stage 2: Pre-2021 FOIA parquet
# ===========================================================================

def stage2_pre2021_foia():
    _section("STAGE 2: Pre-2021 FOIA parquet (pre2021_foia.parquet)")
    if not PRE21_FOIA:
        print("  [SKIP] foia_pre2021_parquet not configured in firm_outcomes.yaml")
        return
    if not _check_exists(PRE21_FOIA, "pre2021_foia"):
        return

    t0 = time.time()
    df = pd.read_parquet(PRE21_FOIA)
    print(f"  Rows: {len(df):,}  ({time.time()-t0:.1f}s)")
    print(f"  Columns: {df.columns.tolist()}")

    # ID range check
    if "foia_indiv_id" in df.columns:
        id_min = df["foia_indiv_id"].min()
        id_max = df["foia_indiv_id"].max()
        print(f"\n  foia_indiv_id range: {id_min:,} – {id_max:,}  (expected min ~10,000,000)")

    # Missing firm UID
    if "foia_firm_uid" in df.columns:
        n_missing = df["foia_firm_uid"].isna().sum()
        print(f"  Rows with missing foia_firm_uid: {n_missing:,}  ({100*n_missing/len(df):.2f}%)")

    # status_type check (should all be SELECTED)
    if "status_type" in df.columns:
        print(f"\n  status_type breakdown:")
        print(df["status_type"].value_counts(dropna=False).to_string())

    # ade_lottery check (should all be NaN)
    if "ade_lottery" in df.columns:
        n_ade_notna = df["ade_lottery"].notna().sum()
        print(f"  ade_lottery non-null: {n_ade_notna:,}  (expected 0 — all NaN for pre-2021)")

    # Per-year summary
    if "lottery_year" in df.columns and "foia_firm_uid" in df.columns:
        year_summary = (
            df.groupby("lottery_year")
            .agg(
                n_petitions=("foia_indiv_id", "count"),
                n_firms    =("foia_firm_uid", "nunique"),
            )
            .reset_index()
            .sort_values("lottery_year")
        )
        print("\n  Per-year summary:")
        print(year_summary.to_string(index=False))

        # Petitions per firm distribution
        pet_per_firm = (
            df.groupby("foia_firm_uid")["foia_indiv_id"]
            .count()
            .rename("petitions_per_firm")
        )
        _pct_table(pet_per_firm, name="petitions per firm (all years)")


# ===========================================================================
# Stage 3: LCA firm-year parquet
# ===========================================================================

def stage3_lca_firm_year():
    _section("STAGE 3: LCA firm-year parquet (lca_firm_year.parquet)")
    if not _check_exists(LCA_FY, "lca_firm_year"):
        return

    t0 = time.time()
    df = pd.read_parquet(LCA_FY)
    print(f"  Rows: {len(df):,}  ({time.time()-t0:.1f}s)")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Fiscal year range: {int(df['fiscal_year'].min())} – {int(df['fiscal_year'].max())}")

    # Per-year table
    year_tbl = (
        df.groupby("fiscal_year")
        .agg(
            n_firms      =("employer_name",   "nunique"),
            n_lca_workers=("n_lca_workers",   "sum"),
            n_lca_cases  =("n_lca_cases",     "sum"),
        )
        .reset_index()
        .sort_values("fiscal_year")
    )
    print("\n  Per fiscal year:")
    print(year_tbl.to_string(index=False))

    # Worker distribution
    _pct_table(df["n_lca_workers"], name="n_lca_workers (per firm-year row)")

    # Top 15 employers
    top = (
        df.groupby("employer_name")["n_lca_workers"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    print("\n  Top 15 employers by total n_lca_workers (all years):")
    print(top.to_string(index=False))


# ===========================================================================
# Stage 4: LCA-FOIA crosswalk
# ===========================================================================

def stage4_lca_crosswalk():
    _section("STAGE 4: LCA-FOIA crosswalk (lca_foia_crosswalk.parquet)")
    if not _check_exists(LCA_XWALK, "lca_foia_crosswalk"):
        return

    t0 = time.time()
    df = pd.read_parquet(LCA_XWALK)
    print(f"  Rows: {len(df):,}  ({time.time()-t0:.1f}s)")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Unique foia_firm_uid: {df['foia_firm_uid'].nunique():,}")

    # Match type breakdown
    if "match_type" in df.columns:
        print("\n  Match type breakdown:")
        mc = df["match_type"].value_counts(dropna=False)
        for mt, cnt in mc.items():
            pct = 100 * cnt / len(df)
            print(f"    {str(mt):<28}  {cnt:>7,}  ({pct:.1f}%)")

    # Match score distribution (fuzzy only)
    if "match_score" in df.columns:
        fuzzy = df[df["match_type"].str.startswith("fuzzy", na=False)]["match_score"]
        if not fuzzy.empty:
            _pct_table(fuzzy, name="fuzzy match_score")

    # Worker-weighted coverage (vs total LCA workers in lca_firm_year)
    if "n_lca_workers_total" in df.columns and _check_exists(LCA_FY, "lca_firm_year (for coverage)"):
        lca_all = pd.read_parquet(LCA_FY, columns=["employer_name", "employer_state", "n_lca_workers"])
        lca_all = lca_all[lca_all["employer_name"].notna()]
        total_workers = lca_all["n_lca_workers"].sum()
        matched_workers = df["n_lca_workers_total"].sum()
        pct_cov = 100 * matched_workers / total_workers if total_workers else 0
        print(f"\n  Worker-weighted coverage: {matched_workers:,.0f} / {total_workers:,.0f} ({pct_cov:.1f}%)")

        # Unmatched firms: LCA firms not in crosswalk
        xwalk_keys = set(zip(df["lca_employer_name"], df["lca_employer_state"]))
        lca_roster = (
            lca_all.groupby(["employer_name", "employer_state"])["n_lca_workers"]
            .sum()
            .reset_index()
        )
        lca_roster["is_matched"] = lca_roster.apply(
            lambda r: (r["employer_name"], r["employer_state"]) in xwalk_keys, axis=1
        )
        unmatched = lca_roster[~lca_roster["is_matched"]]
        print(f"  Unmatched LCA firms (no foia_firm_uid): {len(unmatched):,} / {len(lca_roster):,}")
        if not unmatched.empty:
            top_unmatched = unmatched.nlargest(20, "n_lca_workers")[
                ["employer_name", "employer_state", "n_lca_workers"]
            ]
            print("\n  Top 20 unmatched LCA firms by n_lca_workers:")
            print(top_unmatched.to_string(index=False))

    # n_lca_years distribution
    if "n_lca_years" in df.columns:
        _pct_table(df["n_lca_years"], name="n_lca_years per crosswalk entry")


# ===========================================================================
# Stage 5: Pre-2021 panel assembly
# ===========================================================================

def stage5_panel_assembly():
    _section("STAGE 5: Pre-2021 panel assembly (replicates firm_outcomes.py logic)")

    ok_lca   = _check_exists(LCA_FY,    "lca_firm_year")
    ok_xwalk = _check_exists(LCA_XWALK, "lca_foia_crosswalk")
    ok_foia  = PRE21_FOIA and _check_exists(PRE21_FOIA, "pre2021_foia")
    if not (ok_lca and ok_xwalk and ok_foia):
        print("  [SKIP] One or more required files missing — cannot assemble panel")
        return

    t0 = time.time()

    # ---------- LCA proxy panel (mirrors _load_lca_proxy_panel) ----------
    print(f"\n  --- 5a: Build LCA proxy panel (fiscal_year <= {MAX_YEAR}) ---")
    lca = pd.read_parquet(LCA_FY,
                          columns=["employer_name", "employer_state", "fiscal_year", PROXY_COL])
    xwalk = pd.read_parquet(LCA_XWALK,
                             columns=["lca_employer_name", "lca_employer_state", "foia_firm_uid"])
    print(f"  LCA firm-year rows loaded: {len(lca):,}")
    print(f"  LCA crosswalk entries: {len(xwalk):,}")

    lca = lca[lca["fiscal_year"] <= MAX_YEAR].copy()
    print(f"  After fiscal_year <= {MAX_YEAR}: {len(lca):,} rows")

    lca = lca.rename(columns={"employer_name": "lca_employer_name",
                               "employer_state": "lca_employer_state"})
    merged_lca = lca.merge(xwalk, on=["lca_employer_name", "lca_employer_state"], how="left")
    n_matched  = merged_lca["foia_firm_uid"].notna().sum()
    print(f"  Matched to foia_firm_uid: {n_matched:,} / {len(merged_lca):,} "
          f"({100*n_matched/len(merged_lca):.1f}%)")

    matched_lca = merged_lca[merged_lca["foia_firm_uid"].notna()].copy()
    matched_lca[PROXY_COL] = pd.to_numeric(matched_lca[PROXY_COL], errors="coerce").fillna(0)
    lca_proxy = (
        matched_lca.groupby(["foia_firm_uid", "fiscal_year"])[PROXY_COL]
        .sum()
        .reset_index()
        .rename(columns={"fiscal_year": "lottery_year", PROXY_COL: "n_lca_apps"})
    )
    lca_proxy["lottery_year"] = lca_proxy["lottery_year"].astype(int)
    print(f"  LCA proxy panel: {len(lca_proxy):,} firm-year rows, "
          f"{lca_proxy['foia_firm_uid'].nunique():,} unique firms")
    print(f"  Lottery years: {sorted(lca_proxy['lottery_year'].unique())}")

    # ---------- Pre-2021 FOIA wins ----------
    print(f"\n  --- 5b: Aggregate pre-2021 FOIA wins ---")
    pre21 = pd.read_parquet(PRE21_FOIA)
    pre21_wins = (
        pre21.groupby(["foia_firm_uid", "lottery_year"])["foia_indiv_id"]
        .count()
        .reset_index()
        .rename(columns={"foia_indiv_id": "n_wins"})
    )
    pre21_wins["lottery_year"] = pre21_wins["lottery_year"].astype(int)
    print(f"  Pre-2021 FOIA wins: {len(pre21_wins):,} firm-year rows with wins")
    print(f"  Unique firms with at least one win: {pre21_wins['foia_firm_uid'].nunique():,}")

    # ---------- Left-join ----------
    print(f"\n  --- 5c: Left-join (LCA is left) ---")
    lca_proxy_str = lca_proxy.copy()
    lca_proxy_str["lottery_year_str"] = lca_proxy_str["lottery_year"].astype(str)
    pre21_wins_str = pre21_wins.copy()
    pre21_wins_str["lottery_year_str"] = pre21_wins_str["lottery_year"].astype(str)

    panel = lca_proxy.merge(
        pre21_wins[["foia_firm_uid", "lottery_year", "n_wins"]],
        on=["foia_firm_uid", "lottery_year"],
        how="left",
    )
    panel["n_wins"] = panel["n_wins"].fillna(0)

    n_zero_wins = (panel["n_wins"] == 0).sum()
    print(f"  Panel rows: {len(panel):,}  (LCA firm-years that controlled the sample)")
    print(f"  Firm-years with ZERO FOIA wins (pure losers): {n_zero_wins:,} "
          f"({100*n_zero_wins/len(panel):.1f}%)")

    # FOIA firms not in LCA (dropped — no denominator)
    foia_firms    = set(pre21_wins["foia_firm_uid"].unique())
    lca_firms_in_panel = set(panel[panel["n_wins"] > 0]["foia_firm_uid"].unique())
    dropped_foia  = foia_firms - lca_firms_in_panel
    print(f"  FOIA-win firms NOT in LCA crosswalk (dropped): {len(dropped_foia):,}")

    # ---------- Per-year stats ----------
    print(f"\n  --- 5d: Per-lottery-year panel stats ---")
    panel["win_rate"] = np.where(panel["n_lca_apps"] > 0,
                                  panel["n_wins"] / panel["n_lca_apps"], np.nan)

    year_stats = (
        panel.groupby("lottery_year")
        .agg(
            n_firm_years  =("foia_firm_uid",  "count"),
            n_firms       =("foia_firm_uid",  "nunique"),
            n_wins_total  =("n_wins",         "sum"),
            n_lca_total   =("n_lca_apps",     "sum"),
            win_rate_mean =("win_rate",        "mean"),
            win_rate_med  =("win_rate",        "median"),
            n_lca_p25     =("n_lca_apps",      lambda s: s.quantile(0.25)),
            n_lca_p50     =("n_lca_apps",      "median"),
            n_lca_p75     =("n_lca_apps",      lambda s: s.quantile(0.75)),
            n_lca_p95     =("n_lca_apps",      lambda s: s.quantile(0.95)),
        )
        .reset_index()
        .sort_values("lottery_year")
    )
    year_stats["overall_wr"] = year_stats["n_wins_total"] / year_stats["n_lca_total"]
    print(year_stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---------- Overall ----------
    print(f"\n  --- 5e: Overall panel summary ---")
    total_firm_years = len(panel)
    total_firms      = panel["foia_firm_uid"].nunique()
    total_wins       = panel["n_wins"].sum()
    total_lca_apps   = panel["n_lca_apps"].sum()
    overall_wr       = total_wins / total_lca_apps if total_lca_apps > 0 else np.nan
    print(f"  Total firm-years : {total_firm_years:>10,}")
    print(f"  Unique firms     : {total_firms:>10,}")
    print(f"  Total wins (TRK) : {total_wins:>10,.0f}")
    print(f"  Total LCA apps   : {total_lca_apps:>10,.0f}")
    print(f"  Overall win rate : {overall_wr:>10.3f}  ({100*overall_wr:.1f}%)")

    _pct_table(panel["n_wins"],    name="n_wins per firm-year")
    _pct_table(panel["n_lca_apps"], name="n_lca_apps per firm-year")
    _pct_table(panel["win_rate"].dropna(), name="win_rate (n_wins/n_lca_apps)")

    print(f"\n  Panel assembly complete ({time.time()-t0:.1f}s)")

    # ---------- Plot ----------
    print(f"\n  --- 5f: Win-rate distribution plot ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = panel[panel["win_rate"].notna() & (panel["n_lca_apps"] >= 1)].copy()
    plot_df["lottery_year"] = plot_df["lottery_year"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram of win rate across all years
    ax = axes[0]
    sns.histplot(
        x=plot_df["win_rate"].clip(upper=1.0),
        bins=50, ax=ax, color="steelblue", alpha=0.8,
    )
    ax.set_xlabel("Win rate (n_wins / n_lca_apps)")
    ax.set_ylabel("Firm-years")
    ax.set_title("Win rate distribution (all years)")

    # Right: median win rate by lottery year
    ax2 = axes[1]
    yr_med = (
        plot_df.groupby("lottery_year")["win_rate"]
        .median()
        .reset_index()
        .sort_values("lottery_year")
    )
    sns.barplot(data=yr_med, x="lottery_year", y="win_rate", color="steelblue", ax=ax2)
    ax2.set_xlabel("Lottery year")
    ax2.set_ylabel("Median win rate")
    ax2.set_title("Median win rate by lottery year")
    ax2.tick_params(axis="x", rotation=45)

    fig.suptitle("Pre-2021 panel: LCA-proxied win rates", fontsize=12, fontweight="bold")
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "pre2021_winrate_diag.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    if _IN_IPYTHON:
        plt.show()
    plt.close(fig)
    print(f"  Plot saved → {plot_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_start = time.time()
    print(_SEP)
    print("  Pre-2021 Firm Outcomes Pipeline Diagnostics")
    print(f"  root = {root}")
    print(f"  proxy_col = {PROXY_COL}  |  max_lottery_year = {MAX_YEAR}")
    print(_SEP)

    stage1_trk_crosswalk()
    stage2_pre2021_foia()
    stage3_lca_firm_year()
    stage4_lca_crosswalk()
    stage5_panel_assembly()

    print(f"\n{_SEP}")
    print(f"  Done. Total elapsed: {time.time()-t_start:.0f}s")
    print(_SEP)


if __name__ == "__main__":
    main()
