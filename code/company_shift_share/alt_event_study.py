"""Standalone alternate event-study script with event-level control assignment.

Changes relative to the in-script alt_event_study logic:
1) Treatment events are kept if they are isolated within +/- local_window years
   (not necessarily unique across the full event window). A firm can contribute
   multiple treated events if they are far enough apart.
2) Control assignment is done per treated event from firms with no
   z_pct_change > control_cutoff within +/- local_window years of that event.
   Controls are matched on y_cst_lagm3 and y_cst_lagm2 when enabled.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import time

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

# Ensure progress logs flush immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config


def _ensure_derived_outcome_col(df: pd.DataFrame, col: str, x_source_col: str) -> None:
    if col in df.columns:
        return

    if col == "log1p_y_cst_lag0" and "y_cst_lag0" in df.columns:
        df[col] = np.where(
            df["y_cst_lag0"].notna() & (df["y_cst_lag0"] >= 0),
            np.log1p(df["y_cst_lag0"]),
            np.nan,
        )
        return

    if col == "x_bin_any_nonzero" and x_source_col in df.columns:
        df[col] = (df[x_source_col].fillna(0) != 0).astype("int8")
        return

    if col == "x_bin_above_year_median" and x_source_col in df.columns:
        med = df.groupby("t")[x_source_col].transform("median")
        df[col] = pd.Series(pd.NA, index=df.index, dtype="Int8")
        mask = df[x_source_col].notna() & med.notna()
        if mask.any():
            idx = df.index[mask]
            df.loc[idx, col] = (df.loc[idx, x_source_col] > med.loc[idx]).astype("int8")
        return

    if col == "x_bin_topbot_quartile" and x_source_col in df.columns:
        p25 = df.groupby("t")[x_source_col].transform(lambda s: s.quantile(0.25))
        p75 = df.groupby("t")[x_source_col].transform(lambda s: s.quantile(0.75))
        df[col] = pd.Series(pd.NA, index=df.index, dtype="Int8")
        mask = df[x_source_col].notna() & p25.notna() & p75.notna()
        if mask.any():
            idx = df.index[mask]
            xv = df.loc[idx, x_source_col]
            p25v = p25.loc[idx]
            p75v = p75.loc[idx]
            df.loc[idx[xv <= p25v], col] = 0
            df.loc[idx[xv >= p75v], col] = 1
        return


def _isolated_event_rows(df: pd.DataFrame, window: int) -> pd.DataFrame:
    # df must have columns c, t and contain only treated-event candidates.
    if df.empty:
        return pd.DataFrame(columns=["c", "t"])
    out = df[["c", "t"]].drop_duplicates().sort_values(["c", "t"]).copy()
    prev_t = out.groupby("c")["t"].shift(1)
    next_t = out.groupby("c")["t"].shift(-1)
    keep = ((prev_t.isna()) | ((out["t"] - prev_t) > window)) & (
        (next_t.isna()) | ((next_t - out["t"]) > window)
    )
    return out.loc[keep, ["c", "t"]].reset_index(drop=True)


def _filter_treated_events_by_abs_window(
    treated_df: pd.DataFrame, abs_event_df: pd.DataFrame, window: int
) -> pd.DataFrame:
    # Keep treated (c,t) only if it is the only |z_pct_change|-large event
    # in [t-window, t+window] for that firm.
    if treated_df.empty:
        return pd.DataFrame(columns=["c", "t"])
    abs_years_by_firm = {
        firm: np.sort(g["t"].to_numpy(dtype=int))
        for firm, g in abs_event_df.groupby("c", sort=False)
    }
    kept_rows = []
    for _, row in treated_df.iterrows():
        firm = row["c"]
        year = int(row["t"])
        yrs = abs_years_by_firm.get(firm)
        if yrs is None or yrs.size == 0:
            continue
        lo = int(np.searchsorted(yrs, year - window, side="left"))
        hi = int(np.searchsorted(yrs, year + window, side="right"))
        if (hi - lo) == 1:
            kept_rows.append({"c": firm, "t": year})
    if not kept_rows:
        return pd.DataFrame(columns=["c", "t"])
    return pd.DataFrame(kept_rows).drop_duplicates().reset_index(drop=True)


def main() -> None:
    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:8.2f}s] {msg}")

    _log("Loading config...")
    cfg = load_config(DEFAULT_CONFIG_PATH)
    paths_cfg = get_cfg_section(cfg, "paths")
    reg_cfg = get_cfg_section(cfg, "shift_share_regressions")

    treatment_col_default = str(
        reg_cfg.get("treatment_col", reg_cfg.get("dependent", "masters_opt_hires"))
    )
    # Source column used to construct derived x-bin outcomes.
    x_source_col = str(reg_cfg.get("alt_event_x_source_col", "masters_opt_hires_correction_aware"))
    if re.fullmatch(r"x_bin_(any_nonzero|above_year_median|topbot_quartile)", x_source_col):
        x_source_col = "masters_opt_hires_correction_aware"

    alt_event_instrument_col = str(reg_cfg.get("alt_event_instrument_col", "z_ct_full"))
    alt_event_treat_pctile = float(reg_cfg.get("alt_event_treat_pctile", 90.0))
    alt_event_control_pctile = float(reg_cfg.get("alt_event_control_pctile", 50.0))
    alt_event_time_min = int(reg_cfg.get("alt_event_time_min", -3))
    alt_event_time_max = int(reg_cfg.get("alt_event_time_max", 3))
    alt_event_seed = int(reg_cfg.get("alt_event_seed", 42))
    alt_event_data_min_t = int(reg_cfg.get("alt_event_data_min_t", 2008))
    alt_event_data_max_t = int(reg_cfg.get("alt_event_data_max_t", 2022))
    alt_event_event_min_t = int(reg_cfg.get("alt_event_event_min_t", 2012))
    alt_event_event_max_t = int(reg_cfg.get("alt_event_event_max_t", 2018))
    alt_event_local_window = int(reg_cfg.get("alt_event_local_window", 3))
    alt_event_match_controls_on_y_cst_lagm3 = bool(
        reg_cfg.get("alt_event_match_controls_on_y_cst_lagm3", True)
    )
    alt_event_control_without_replacement = bool(
        reg_cfg.get("alt_event_control_without_replacement", True)
    )
    alt_event_use_log_y = bool(reg_cfg.get("alt_event_use_log_y", False))

    alt_outcome_cfg = reg_cfg.get("alt_event_outcome_col", None)
    if alt_outcome_cfg is None or str(alt_outcome_cfg).strip().lower() in {"", "none", "null"}:
        alt_event_outcome_col = treatment_col_default
    else:
        alt_event_outcome_col = str(alt_outcome_cfg)
    if alt_event_use_log_y and alt_event_outcome_col == "y_cst_lag0":
        alt_event_outcome_col = "log1p_y_cst_lag0"

    con = ddb.connect()
    panel_path = Path(paths_cfg["analysis_panel"])
    if not panel_path.is_absolute():
        panel_path = Path.cwd() / panel_path
    panel_path_esc = str(panel_path).replace("'", "''")
    con.sql(f"CREATE OR REPLACE VIEW analysis_panel AS SELECT * FROM read_parquet('{panel_path_esc}')")

    available_cols = set(con.sql("DESCRIBE analysis_panel").df()["column_name"].tolist())

    # Pull only columns needed for this script to avoid expensive SELECT * scans.
    required_cols = {"c", "t", alt_event_instrument_col, "y_cst_lagm3", "y_cst_lagm2", "y_cst_lagm1", "n_universities"}
    # Potentially needed to construct derived outcomes.
    required_cols.update({"y_cst_lag0", x_source_col, alt_event_outcome_col})
    select_cols = sorted(c for c in required_cols if c in available_cols)
    missing_for_read = sorted(c for c in required_cols if c not in available_cols)
    if missing_for_read:
        _log(
            "Columns not found in analysis_panel (will derive later if needed): "
            + ", ".join(missing_for_read)
        )
    cols_sql = ", ".join(select_cols)
    _log("Reading event-study source from parquet...")
    es_source = con.sql(
        f"""
        SELECT {cols_sql}
        FROM analysis_panel
        WHERE t BETWEEN {alt_event_data_min_t} AND {alt_event_data_max_t}
        """
    ).df()
    _log(f"Loaded es_source: {len(es_source):,} rows x {len(es_source.columns)} cols")

    if alt_event_instrument_col not in es_source.columns:
        raise ValueError(
            f"alt_event_instrument_col '{alt_event_instrument_col}' not found in event-study source columns."
        )
    _ensure_derived_outcome_col(es_source, alt_event_outcome_col, x_source_col)
    if alt_event_outcome_col not in es_source.columns:
        raise ValueError(
            f"alt_event_outcome_col '{alt_event_outcome_col}' not found in event-study source columns."
        )
    if alt_event_use_log_y:
        _ensure_derived_outcome_col(es_source, "log1p_y_cst_lag0", x_source_col)
    if "y_cst_lagm3" not in es_source.columns or "y_cst_lagm2" not in es_source.columns:
        raise ValueError("Columns 'y_cst_lagm3' and 'y_cst_lagm2' are required for control matching.")
    if not (0 < alt_event_control_pctile < alt_event_treat_pctile < 100):
        raise ValueError("Require 0 < alt_event_control_pctile < alt_event_treat_pctile < 100.")
    if alt_event_time_min >= alt_event_time_max:
        raise ValueError("alt_event_time_min must be < alt_event_time_max.")
    if alt_event_event_min_t > alt_event_event_max_t:
        raise ValueError("alt_event_event_min_t must be <= alt_event_event_max_t.")
    if alt_event_local_window < 0:
        raise ValueError("alt_event_local_window must be >= 0.")

    es_df = es_source[["c", "t", alt_event_instrument_col]].copy().sort_values(["c", "t"])
    es_df["z_lag1"] = es_df.groupby("c")[alt_event_instrument_col].shift(1)
    es_df["z_pct_change"] = np.where(
        es_df["z_lag1"].notna() & (es_df["z_lag1"] != 0),
        (es_df[alt_event_instrument_col] - es_df["z_lag1"]) / es_df["z_lag1"],
        np.nan,
    )
    es_df.loc[es_df["z_pct_change"] > 10, "z_pct_change"] = np.nan
    es_df["z_abs_pct_change"] = es_df["z_pct_change"].abs()
    es_df["event_year_eligible"] = es_df["t"].between(alt_event_event_min_t, alt_event_event_max_t)

    pct_valid = es_df.loc[es_df["event_year_eligible"], "z_pct_change"].dropna()
    pct_valid_abs = es_df.loc[es_df["event_year_eligible"], "z_abs_pct_change"].dropna()
    if pct_valid.empty:
        raise ValueError("No valid z_pct_change values in event window.")
    if pct_valid_abs.empty:
        raise ValueError("No valid |z_pct_change| values in event window.")

    treat_cutoff = float(np.nanpercentile(pct_valid, alt_event_treat_pctile))
    control_cutoff = float(np.nanpercentile(pct_valid_abs, alt_event_control_pctile))

    # Top-of-output diagnostic: distribution of z_pct_change in the event window.
    plt.figure(figsize=(8, 4.5))
    plt.hist(pct_valid, bins=80, alpha=0.85, color="tab:blue")
    plt.axvline(control_cutoff, color="tab:orange", linestyle="--", linewidth=1.5, label=f"|Control cutoff| p{alt_event_control_pctile:g}")
    plt.axvline(-control_cutoff, color="tab:orange", linestyle="--", linewidth=1.5)
    plt.axvline(treat_cutoff, color="tab:red", linestyle="--", linewidth=1.5, label=f"Treat cutoff p{alt_event_treat_pctile:g}")
    plt.xlabel("z_pct_change")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    es_df["is_treat_event_raw"] = es_df["event_year_eligible"] & (es_df["z_pct_change"] > treat_cutoff)
    es_df["is_above_control_abs"] = es_df["z_abs_pct_change"] > control_cutoff

    treated_raw = es_df.loc[es_df["is_treat_event_raw"], ["c", "t"]].copy()
    abs_events = es_df.loc[es_df["is_above_control_abs"], ["c", "t"]].copy()
    treated_kept = _filter_treated_events_by_abs_window(
        treated_raw, abs_events, window=alt_event_local_window
    )
    if treated_kept.empty:
        raise ValueError("No treated events remain after local-window isolation rule.")
    _log(f"Treated events kept: {len(treated_kept):,}")

    treated_events = treated_kept.rename(columns={"t": "event_year"}).sort_values(["event_year", "c"]).reset_index(drop=True)
    treated_events["event_id"] = np.arange(len(treated_events), dtype=int)

    y_lookup = es_source[["c", "t", "y_cst_lagm3", "y_cst_lagm2"]].rename(columns={"t": "event_year"})
    treated_events = treated_events.merge(y_lookup, on=["c", "event_year"], how="left")
    treated_events = treated_events.rename(
        columns={
            "y_cst_lagm3": "treated_y_cst_lagm3",
            "y_cst_lagm2": "treated_y_cst_lagm2",
            "c": "treated_firm",
        }
    )

    all_firms = set(es_df["c"].dropna().unique().tolist())
    is_above = es_df.loc[es_df["is_above_control_abs"].fillna(False), ["c", "t"]].copy()

    eligible_controls_by_year: dict[int, set] = {}
    for yr in range(alt_event_event_min_t, alt_event_event_max_t + 1):
        lo = yr - alt_event_local_window
        hi = yr + alt_event_local_window
        bad = set(is_above.loc[is_above["t"].between(lo, hi), "c"].tolist())
        eligible_controls_by_year[yr] = set(all_firms - bad)

    rng = np.random.default_rng(alt_event_seed)
    used_controls: set = set()
    control_rows = []
    _log("Preparing year-level control pools...")
    y_lookup_by_year = {int(yr): g.copy() for yr, g in y_lookup.groupby("event_year", sort=False)}
    pool_by_year: dict[int, dict[str, np.ndarray]] = {}
    for yr in range(alt_event_event_min_t, alt_event_event_max_t + 1):
        g = y_lookup_by_year.get(yr)
        if g is None or g.empty:
            pool_by_year[yr] = {
                "c_nonmiss": np.array([], dtype=object),
                "y3_nonmiss": np.array([], dtype=float),
                "y2_nonmiss": np.array([], dtype=float),
                "c_all": np.array([], dtype=object),
            }
            continue
        elig = eligible_controls_by_year.get(yr, set())
        if not elig:
            pool_by_year[yr] = {
                "c_nonmiss": np.array([], dtype=object),
                "y3_nonmiss": np.array([], dtype=float),
                "y2_nonmiss": np.array([], dtype=float),
                "c_all": np.array([], dtype=object),
            }
            continue
        gy = g.loc[g["c"].isin(elig), ["c", "y_cst_lagm3", "y_cst_lagm2"]].copy()
        if gy.empty:
            pool_by_year[yr] = {
                "c_nonmiss": np.array([], dtype=object),
                "y3_nonmiss": np.array([], dtype=float),
                "y2_nonmiss": np.array([], dtype=float),
                "c_all": np.array([], dtype=object),
            }
            continue
        gnm = gy.loc[gy["y_cst_lagm3"].notna() & gy["y_cst_lagm2"].notna()].copy()
        pool_by_year[yr] = {
            "c_nonmiss": gnm["c"].to_numpy(),
            "y3_nonmiss": gnm["y_cst_lagm3"].to_numpy(dtype=float),
            "y2_nonmiss": gnm["y_cst_lagm2"].to_numpy(dtype=float),
            "c_all": gy["c"].to_numpy(),
        }

    _log("Assigning matched controls...")
    treated_sorted = treated_events.sort_values(["event_year", "event_id"]).reset_index(drop=True)
    n_treated = len(treated_sorted)
    for i, (_, tr) in enumerate(treated_sorted.iterrows(), start=1):
        yr = int(tr["event_year"])
        treated_firm = tr["treated_firm"]
        treated_y3 = tr["treated_y_cst_lagm3"]
        treated_y2 = tr["treated_y_cst_lagm2"]

        pool = pool_by_year.get(yr)
        if pool is None:
            continue
        c_nonmiss = pool["c_nonmiss"]
        y3_nonmiss = pool["y3_nonmiss"]
        y2_nonmiss = pool["y2_nonmiss"]
        c_all = pool["c_all"]
        if c_all.size == 0:
            continue

        chosen_c = None
        if (
            alt_event_match_controls_on_y_cst_lagm3
            and pd.notna(treated_y3)
            and pd.notna(treated_y2)
            and y3_nonmiss.size > 0
        ):
            valid_mask = np.fromiter(
                (
                    (c != treated_firm)
                    and ((not alt_event_control_without_replacement) or (c not in used_controls))
                    for c in c_nonmiss
                ),
                dtype=bool,
                count=c_nonmiss.size,
            )
            valid_idx = np.flatnonzero(valid_mask)
            if valid_idx.size > 0:
                d = (y3_nonmiss[valid_idx] - float(treated_y3)) ** 2 + (y2_nonmiss[valid_idx] - float(treated_y2)) ** 2
                chosen_c = c_nonmiss[valid_idx[int(np.argmin(d))]]

        if chosen_c is None:
            # Fallback: random eligible control in year-specific pool.
            if alt_event_control_without_replacement:
                avail = [c for c in c_all if c != treated_firm and c not in used_controls]
            else:
                avail = [c for c in c_all if c != treated_firm]
            if not avail:
                continue
            chosen_c = avail[int(rng.integers(0, len(avail)))]

        if alt_event_control_without_replacement:
            used_controls.add(chosen_c)
        control_rows.append({"event_id": int(tr["event_id"]), "c": chosen_c, "event_year": yr, "treated": 0})
        if i % 2000 == 0 or i == n_treated:
            _log(f"Matched {i:,}/{n_treated:,} treated events; controls assigned: {len(control_rows):,}")

    if not control_rows:
        raise ValueError("No controls could be assigned under current settings.")
    _log(f"Assigned controls: {len(control_rows):,}")

    controls = pd.DataFrame(control_rows)
    treated_panel_rows = treated_events[["event_id", "treated_firm", "event_year"]].rename(
        columns={"treated_firm": "c"}
    )
    treated_panel_rows["treated"] = 1

    event_firms = pd.concat(
        [treated_panel_rows[["event_id", "c", "event_year", "treated"]], controls],
        ignore_index=True,
    )
    event_firms["group"] = np.where(event_firms["treated"] == 1, "treated", "control")

    event_counts = (
        event_firms.groupby(["event_year", "group"], as_index=False)
        .agg(n_events=("event_id", "nunique"), n_firms=("c", "nunique"))
        .sort_values(["event_year", "group"])
    )
    print("\n[alt_event_study_counts_by_event_year]")
    print(event_counts)

    event_level_chars = event_firms.merge(
        es_source[["c", "t", "y_cst_lagm3", "y_cst_lagm1", "n_universities"]].rename(columns={"t": "event_year"}),
        on=["c", "event_year"],
        how="left",
    )
    lagm3_summary = (
        event_level_chars.groupby("group", as_index=False)
        .agg(
            mean_y_cst_lagm3=("y_cst_lagm3", "mean"),
            n_events=("event_id", "nunique"),
            n_nonmissing_y_cst_lagm3=("y_cst_lagm3", "count"),
        )
        .sort_values("group")
    )
    print("\n[alt_event_study_mean_y_cst_lagm3_by_group]")
    print(lagm3_summary)

    # Histograms of baseline characteristics by treatment status.
    for _col in ["y_cst_lagm1", "n_universities"]:
        if _col not in event_level_chars.columns:
            print(f"\n[warn] Column '{_col}' not available; skipping histogram.")
            continue
        hdf = event_level_chars[["group", _col]].dropna()
        if hdf.empty:
            print(f"\n[warn] No non-missing values for '{_col}'; skipping histogram.")
            continue
        plt.figure(figsize=(8, 4.5))
        for grp, color in [("control", "tab:blue"), ("treated", "tab:orange")]:
            vals = hdf.loc[hdf["group"] == grp, _col]
            if vals.empty:
                continue
            plt.hist(vals, bins=40, alpha=0.45, label=grp.capitalize(), color=color)
        plt.xlabel(_col)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Build only event-window rows (event_id x event_time) to avoid an expensive
    # many-to-many merge on firm id.
    offsets = np.arange(alt_event_time_min, alt_event_time_max + 1, dtype=int)
    base_events = event_firms[["event_id", "c", "event_year", "treated"]].copy()
    rep_n = len(offsets)
    es_windows = base_events.loc[base_events.index.repeat(rep_n)].copy()
    es_windows["event_time"] = np.tile(offsets, len(base_events))
    es_windows["t"] = es_windows["event_year"] + es_windows["event_time"]

    z_pct_map = es_df[["c", "t", "z_pct_change"]].copy()
    source_with_z = es_source.merge(z_pct_map, on=["c", "t"], how="left")
    es_sample = es_windows.merge(source_with_z, on=["c", "t"], how="inner")
    if es_sample.empty:
        raise ValueError("Event-study sample is empty after event-time filtering.")
    _log(f"Constructed event-time sample: {len(es_sample):,} rows")

    # (0) Raw means of z_pct_change by event time for treated vs control.
    z_means = (
        es_sample.groupby(["event_time", "treated"], as_index=False)["z_pct_change"]
        .mean()
        .rename(columns={"z_pct_change": "mean_z_pct_change"})
    )
    plt.figure(figsize=(8, 4.5))
    for tr, label in [(0, "Control"), (1, "Treated")]:
        s = z_means.loc[z_means["treated"] == tr].sort_values("event_time")
        if not s.empty:
            plt.plot(s["event_time"], s["mean_z_pct_change"], marker="o", label=label)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Event time")
    plt.ylabel("Mean z_pct_change")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (0b) Raw means of instrument level by event time for treated vs control.
    instr_means = (
        es_sample.groupby(["event_time", "treated"], as_index=False)[alt_event_instrument_col]
        .mean()
        .rename(columns={alt_event_instrument_col: "mean_instrument"})
    )
    plt.figure(figsize=(8, 4.5))
    for tr, label in [(0, "Control"), (1, "Treated")]:
        s = instr_means.loc[instr_means["treated"] == tr].sort_values("event_time")
        if not s.empty:
            plt.plot(s["event_time"], s["mean_instrument"], marker="o", label=label)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Event time")
    plt.ylabel(f"Mean {alt_event_instrument_col}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    y0_outcome_for_plots = "log1p_y_cst_lag0" if alt_event_use_log_y else "y_cst_lag0"
    outcomes_to_plot = []
    for _outcome in [alt_event_outcome_col, y0_outcome_for_plots]:
        if _outcome in es_sample.columns and _outcome not in outcomes_to_plot:
            outcomes_to_plot.append(_outcome)

    # Event-time indicators used by all TWFE outcome variants.
    event_times = sorted(int(k) for k in es_sample["event_time"].dropna().unique())
    ref_k = -1 if -1 in event_times else min(event_times)
    rhs_terms = []
    kept_ks = []
    for k in event_times:
        if k == ref_k:
            continue
        col = f"evt_treat_{'m' + str(abs(k)) if k < 0 else 'p' + str(k)}"
        es_sample[col] = ((es_sample["event_time"] == k) & (es_sample["treated"] == 1)).astype("int8")
        rhs_terms.append(col)
        kept_ks.append((k, col))
    if not rhs_terms:
        raise ValueError("No non-reference event-time indicators for TWFE plot.")

    for outcome_col in outcomes_to_plot:
        # (1) Raw means by event time for treated vs control.
        raw_means = (
            es_sample.groupby(["event_time", "treated"], as_index=False)[outcome_col]
            .mean()
            .rename(columns={outcome_col: "mean_outcome"})
        )
        plt.figure(figsize=(8, 4.5))
        for tr, label in [(0, "Control"), (1, "Treated")]:
            s = raw_means.loc[raw_means["treated"] == tr].sort_values("event_time")
            if not s.empty:
                plt.plot(s["event_time"], s["mean_outcome"], marker="o", label=label)
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Event time")
        plt.ylabel(f"Mean {outcome_col}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # (2) TWFE event study: coefficients on treated x event-time.
        twfe_fit = pf.feols(
            f"{outcome_col} ~ {' + '.join(rhs_terms)} | c + t",
            data=es_sample,
            vcov={"CRV1": "c"},
            demeaner_backend="rust",
        )
        _log(f"Estimated TWFE model for outcome={outcome_col}.")
        twfe_tidy = twfe_fit.tidy()
        twfe_rows = []
        for k, col in kept_ks:
            if col in twfe_tidy.index:
                est = float(twfe_tidy.loc[col, "Estimate"])
                se = float(twfe_tidy.loc[col, "Std. Error"])
                twfe_rows.append(
                    {"event_time": k, "coef": est, "se": se, "lo": est - 1.96 * se, "hi": est + 1.96 * se}
                )
        twfe_res = pd.DataFrame(twfe_rows).sort_values("event_time")
        ref_point = pd.DataFrame([{"event_time": ref_k, "coef": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0}])
        twfe_plot = (
            pd.concat([twfe_res, ref_point], ignore_index=True)
            .drop_duplicates(subset=["event_time"], keep="first")
            .sort_values("event_time")
        )
        plt.figure(figsize=(8, 4.5))
        plt.errorbar(
            twfe_plot["event_time"],
            twfe_plot["coef"],
            yerr=1.96 * twfe_plot["se"],
            fmt="o",
            capsize=4,
        )
        plt.axhline(0, color="black", linewidth=1)
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Event time")
        plt.ylabel(f"TWFE coef on Treated x Event-time ({outcome_col})")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
