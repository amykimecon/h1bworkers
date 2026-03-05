"""Calendar-time OPT event study centered on a single year (default: 2016).

This script uses all firms in the analysis panel and summarizes how OPT hiring
changes around a calendar-year event date. It reports:

1) Raw mean OPT outcome by year (rebased to 2015 in plots),
2) Within-firm change relative to a reference event time (default: -1).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config


def _escape(path: Path) -> str:
    return str(path).replace("'", "''")


def _ensure_derived_outcome_col(df: pd.DataFrame, col: str, x_source_col: str) -> None:
    if col in df.columns:
        return
    if col.startswith("log1p_"):
        base = col[len("log1p_") :]
        if base in df.columns:
            df[col] = np.where(df[base].notna() & (df[base] >= 0), np.log1p(df[base]), np.nan)
            return
    if x_source_col not in df.columns:
        return
    if col == "x_bin_any_nonzero":
        df[col] = (df[x_source_col].fillna(0) != 0).astype("int8")
        return
    if col == "x_bin_above_year_median":
        med = df.groupby("t")[x_source_col].transform("median")
        df[col] = pd.Series(pd.NA, index=df.index, dtype="Int8")
        mask = df[x_source_col].notna() & med.notna()
        if mask.any():
            idx = df.index[mask]
            df.loc[idx, col] = (df.loc[idx, x_source_col] > med.loc[idx]).astype("int8")
        return
    if col == "x_bin_topbot_quartile":
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


def _parse_outcome_cols(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        vals = [v.strip() for v in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        vals = [str(v).strip() for v in raw]
    else:
        vals = [str(raw).strip()]
    out: list[str] = []
    seen: set[str] = set()
    for v in vals:
        if not v or v.lower() in {"none", "null"}:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _identify_absorbing_events(
    df: pd.DataFrame,
    instrument_col_raw: str,
    event_min_t: int,
    event_max_t: int,
    local_window: int = 3,
    neighbor_shock_ratio: float = 0.5,
    use_pct_change: bool = True,
    min_shock_abs: float | None = None,
    min_shock_quantile: float | None = None,
) -> pd.DataFrame:
    if instrument_col_raw not in df.columns:
        raise ValueError(
            f"Raw instrument column '{instrument_col_raw}' not found in panel for event assignment."
        )
    if local_window < 0:
        raise ValueError("local_window must be >= 0.")
    if not (0 < neighbor_shock_ratio <= 1.0):
        raise ValueError("neighbor_shock_ratio must be in (0, 1].")
    if min_shock_quantile is not None and not (0 <= float(min_shock_quantile) <= 1):
        raise ValueError("min_shock_quantile must be in [0, 1].")

    z_df = (
        df[["c", "t", instrument_col_raw]]
        .dropna(subset=["c", "t"])
        .sort_values(["c", "t"])
        .copy()
    )
    z_df["z_lag1"] = z_df.groupby("c")[instrument_col_raw].shift(1)
    if use_pct_change:
        z_df["shock_raw"] = np.where(
            z_df["z_lag1"].notna() & (z_df["z_lag1"] != 0),
            (z_df[instrument_col_raw] - z_df["z_lag1"]) / z_df["z_lag1"],
            np.nan,
        )
    else:
        z_df["shock_raw"] = np.where(
            z_df["z_lag1"].notna(),
            z_df[instrument_col_raw] - z_df["z_lag1"],
            np.nan,
        )
    z_df = z_df.dropna(subset=["shock_raw"]).copy()
    if z_df.empty:
        return pd.DataFrame(columns=["c", "g", "event_shock", "event_abs_shock"])

    positive_in_window = z_df.loc[
        z_df["t"].between(event_min_t, event_max_t) & (z_df["shock_raw"] > 0),
        "shock_raw",
    ].to_numpy(dtype=np.float64)
    shock_floor_candidates: list[float] = [0.0]
    if min_shock_abs is not None:
        shock_floor_candidates.append(float(min_shock_abs))
    if min_shock_quantile is not None and positive_in_window.size > 0:
        shock_floor_candidates.append(
            float(np.nanquantile(positive_in_window, float(min_shock_quantile)))
        )
    min_required_positive_shock = max(shock_floor_candidates)

    picked_rows: list[dict[str, float | int]] = []
    for firm, gdf in z_df.groupby("c", sort=False):
        t_all = gdf["t"].to_numpy(dtype=np.int64)
        shock_all = gdf["shock_raw"].to_numpy(dtype=np.float64)
        abs_all = np.abs(shock_all)

        cand_mask = (
            (t_all >= event_min_t)
            & (t_all <= event_max_t)
            & (shock_all > 0)
            & (shock_all >= min_required_positive_shock)
        )
        if not cand_mask.any():
            continue
        cand_idx = np.flatnonzero(cand_mask)
        cand_idx = cand_idx[np.argsort(-shock_all[cand_idx], kind="mergesort")]

        cand_t = t_all[cand_idx][:, None]
        cand_abs = abs_all[cand_idx][:, None]
        threshold = neighbor_shock_ratio * cand_abs

        dt = np.abs(t_all[None, :] - cand_t)
        is_neighbor = (dt <= local_window) & (dt > 0)
        has_large_neighbor = np.any(is_neighbor & (abs_all[None, :] > threshold), axis=1)
        valid_pos = np.flatnonzero(~has_large_neighbor)
        if valid_pos.size == 0:
            continue

        chosen_idx = cand_idx[int(valid_pos[0])]
        chosen_shock = float(shock_all[chosen_idx])
        picked_rows.append(
            {
                "c": firm,
                "g": int(t_all[chosen_idx]),
                "event_shock": chosen_shock,
                "event_abs_shock": abs(chosen_shock),
            }
        )

    if not picked_rows:
        return pd.DataFrame(columns=["c", "g", "event_shock", "event_abs_shock"])
    return pd.DataFrame(picked_rows).drop_duplicates(subset=["c"]).copy()


def _parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calendar-time OPT event study around 2016.")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    p.add_argument("--event-year", type=int, default=2016, help="Calendar year to center event time at.")
    p.add_argument("--event-time-min", type=int, default=-5, help="Min event time (inclusive).")
    p.add_argument("--event-time-max", type=int, default=5, help="Max event time (inclusive).")
    p.add_argument(
        "--ref-event-time",
        type=int,
        default=-1,
        help="Reference event time for within-firm differences.",
    )
    p.add_argument(
        "--outcome-col",
        type=str,
        default=None,
        help=(
            "Outcome column in analysis panel (or comma-separated list) "
            "(default from config: shift_share_regressions.opt_event_outcome_cols, "
            "then opt_event_outcome_col, "
            "then alt_event_outcome_col, then alt_event_x_source_col)."
        ),
    )
    p.add_argument(
        "--x-source-col",
        type=str,
        default=None,
        help=(
            "Source column for derived x-bin outcomes "
            "(default from config: shift_share_regressions.opt_event_x_source_col, "
            "then alt_event_x_source_col)."
        ),
    )
    p.add_argument(
        "--include-bachelors-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use paths.analysis_panel_ma_ba when available.",
    )
    p.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot event-time series.",
    )
    if args is None:
        # In IPython/Jupyter, sys.argv usually includes kernel flags like "-f ...json".
        # Ignore unknown args only in that interactive context.
        argv0 = Path(sys.argv[0]).name.lower() if sys.argv else ""
        has_kernel_argv = (
            len(sys.argv) >= 3
            and sys.argv[1] == "-f"
            and str(sys.argv[2]).lower().endswith(".json")
        )
        in_ipython = (
            "IPython" in sys.modules
            or "ipykernel" in sys.modules
            or "ipykernel_launcher" in argv0
            or has_kernel_argv
        )
        if in_ipython:
            parsed, unknown = p.parse_known_args()
            if unknown:
                print(f"[info] Ignoring unknown IPython args: {unknown}")
            return parsed
    return p.parse_args(args)


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(cli_args)
    cfg = load_config(args.config)
    paths_cfg = get_cfg_section(cfg, "paths")
    build_cfg = get_cfg_section(cfg, "build_company_shift_share")
    reg_cfg = get_cfg_section(cfg, "shift_share_regressions")

    if args.event_time_min > args.event_time_max:
        raise ValueError("event_time_min must be <= event_time_max.")
    if args.ref_event_time < args.event_time_min or args.ref_event_time > args.event_time_max:
        raise ValueError("ref_event_time must lie within [event_time_min, event_time_max].")

    include_bachelors_sample = (
        args.include_bachelors_sample
        if args.include_bachelors_sample is not None
        else bool(reg_cfg.get("include_bachelors_sample", False))
    )
    panel_key = "analysis_panel"
    if include_bachelors_sample and str(paths_cfg.get("analysis_panel_ma_ba", "")).strip():
        panel_key = "analysis_panel_ma_ba"

    cli_outcomes = _parse_outcome_cols(args.outcome_col)
    cfg_outcomes = _parse_outcome_cols(reg_cfg.get("opt_event_outcome_cols", None))
    requested_outcomes = cli_outcomes or cfg_outcomes
    if len(requested_outcomes) > 1:
        for i, outcome in enumerate(requested_outcomes, start=1):
            print(
                f"[info] Running outcome {i}/{len(requested_outcomes)} "
                f"for opt event study: {outcome}"
            )
            rerun_args: list[str] = [
                "--event-year",
                str(args.event_year),
                "--event-time-min",
                str(args.event_time_min),
                "--event-time-max",
                str(args.event_time_max),
                "--ref-event-time",
                str(args.ref_event_time),
                "--outcome-col",
                outcome,
            ]
            if args.config is not None:
                rerun_args = ["--config", str(args.config), *rerun_args]
            if args.x_source_col is not None:
                rerun_args.extend(["--x-source-col", str(args.x_source_col)])
            if args.include_bachelors_sample is not None:
                rerun_args.append(
                    "--include-bachelors-sample"
                    if args.include_bachelors_sample
                    else "--no-include-bachelors-sample"
                )
            if args.plot is False:
                rerun_args.append("--no-plot")
            main(rerun_args)
        return
    if len(requested_outcomes) == 1:
        args.outcome_col = requested_outcomes[0]

    cfg_outcome = reg_cfg.get("opt_event_outcome_col", None)
    if cfg_outcome is None or str(cfg_outcome).strip().lower() in {"", "none", "null"}:
        cfg_outcome = reg_cfg.get("alt_event_outcome_col", None)
    if cfg_outcome is None or str(cfg_outcome).strip().lower() in {"", "none", "null"}:
        cfg_outcome = reg_cfg.get("alt_event_x_source_col", "masters_opt_hires_correction_aware")
    outcome_col = args.outcome_col or str(cfg_outcome)

    cfg_x_source = reg_cfg.get("opt_event_x_source_col", None)
    if cfg_x_source is None or str(cfg_x_source).strip().lower() in {"", "none", "null"}:
        cfg_x_source = reg_cfg.get("alt_event_x_source_col", "masters_opt_hires_correction_aware")
    x_source_col = args.x_source_col or str(cfg_x_source)
    instrument_col_raw = str(reg_cfg.get("instrument", "z_ct_full"))
    if instrument_col_raw.startswith("log_"):
        raise ValueError(
            f"Configured instrument '{instrument_col_raw}' appears to be logged. "
            "Use a non-log instrument column for shock assignment."
        )

    opt_shifts = bool(build_cfg.get("opt_shifts", False))
    data_min_t = int(reg_cfg.get("alt_event_data_min_t", 2008))
    data_max_t = int(reg_cfg.get("alt_event_data_max_t", 2022))
    event_min_t = int(reg_cfg.get("alt_event_event_min_t", 2012))
    event_max_t = int(reg_cfg.get("alt_event_event_max_t", 2018))
    event_local_window = int(reg_cfg.get("absorbing_event_local_window", 3))
    neighbor_shock_ratio = float(reg_cfg.get("absorbing_neighbor_shock_ratio", 0.5))

    shock_metric_cfg = str(reg_cfg.get("absorbing_event_shock_metric", "auto")).strip().lower()
    if shock_metric_cfg == "auto":
        use_pct_change_for_event_assignment = not opt_shifts
    elif shock_metric_cfg in {"pct_change", "percent_change", "pct"}:
        use_pct_change_for_event_assignment = True
    elif shock_metric_cfg in {"first_difference", "difference", "diff", "level_change"}:
        use_pct_change_for_event_assignment = False
    else:
        raise ValueError(
            "absorbing_event_shock_metric must be one of: "
            "auto, pct_change, first_difference."
        )
    min_shock_quantile_cfg = reg_cfg.get("absorbing_event_min_shock_quantile", None)
    if isinstance(min_shock_quantile_cfg, str) and min_shock_quantile_cfg.strip().lower() in {"", "none", "null"}:
        min_shock_quantile_cfg = None
    min_shock_quantile = (
        float(min_shock_quantile_cfg)
        if min_shock_quantile_cfg is not None
        else (0.90 if opt_shifts else None)
    )
    min_shock_abs_cfg = reg_cfg.get("absorbing_event_min_shock_abs", None)
    if isinstance(min_shock_abs_cfg, str) and min_shock_abs_cfg.strip().lower() in {"", "none", "null"}:
        min_shock_abs_cfg = None
    min_shock_abs = float(min_shock_abs_cfg) if min_shock_abs_cfg is not None else None

    con = ddb.connect()
    panel_path = Path(paths_cfg[panel_key])
    if not panel_path.is_absolute():
        panel_path = Path.cwd() / panel_path
    con.sql(f"CREATE OR REPLACE VIEW analysis_panel AS SELECT * FROM read_parquet('{_escape(panel_path)}')")
    print(f"[info] Using analysis panel source: paths.{panel_key} -> {panel_path}")

    available_cols = set(con.sql("DESCRIBE analysis_panel").df()["column_name"].tolist())
    needed = {"c", "t", outcome_col, x_source_col, instrument_col_raw}
    if outcome_col.startswith("log1p_"):
        needed.add(outcome_col[len("log1p_") :])
    select_cols = sorted(c for c in needed if c in available_cols)
    if "c" not in select_cols or "t" not in select_cols:
        raise ValueError("analysis_panel must contain columns 'c' and 't'.")
    if instrument_col_raw not in select_cols:
        raise ValueError(
            f"Instrument column '{instrument_col_raw}' not found in analysis panel."
        )

    year_min = args.event_year + args.event_time_min
    year_max = args.event_year + args.event_time_max
    extract_min = min(int(year_min), int(data_min_t))
    extract_max = max(int(year_max), int(data_max_t))
    cols_sql = ", ".join(select_cols)
    df = con.sql(
        f"""
        SELECT {cols_sql}
        FROM analysis_panel
        WHERE t BETWEEN {extract_min} AND {extract_max}
        """
    ).df()
    if df.empty:
        raise ValueError("No rows in selected event-time window.")

    _ensure_derived_outcome_col(df, outcome_col, x_source_col=x_source_col)
    if outcome_col not in df.columns:
        raise ValueError(
            f"Outcome column '{outcome_col}' is unavailable. "
            f"Try --outcome-col masters_opt_hires_correction_aware."
        )

    df["c"] = pd.to_numeric(df["c"], errors="coerce")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    df[instrument_col_raw] = pd.to_numeric(df[instrument_col_raw], errors="coerce")
    df = df.dropna(subset=["c", "t"]).copy()
    df["c"] = df["c"].astype("int64")
    df["t"] = df["t"].astype("int64")

    event_map = _identify_absorbing_events(
        df,
        instrument_col_raw=instrument_col_raw,
        event_min_t=event_min_t,
        event_max_t=event_max_t,
        local_window=event_local_window,
        neighbor_shock_ratio=neighbor_shock_ratio,
        use_pct_change=use_pct_change_for_event_assignment,
        min_shock_abs=min_shock_abs,
        min_shock_quantile=min_shock_quantile,
    )

    df_evt = df.loc[df["t"].between(int(year_min), int(year_max))].copy()
    df_evt = df_evt.dropna(subset=[outcome_col]).copy()
    df_evt["event_time"] = df_evt["t"] - int(args.event_year)
    df_evt = df_evt.loc[df_evt["event_time"].between(args.event_time_min, args.event_time_max)].copy()
    if df_evt.empty:
        raise ValueError("No usable rows remain after event-time filtering.")
    df_evt = df_evt.merge(event_map[["c", "g"]], on="c", how="left")
    df_evt["shock_group"] = np.select(
        [
            df_evt["g"].notna() & (df_evt["g"] < int(args.event_year)),
            df_evt["g"].notna() & (df_evt["g"] >= int(args.event_year)),
        ],
        [
            f"Shock before {int(args.event_year)}",
            f"Shock in/after {int(args.event_year)}",
        ],
        default="Never shock",
    )
    group_order = [
        f"Shock before {int(args.event_year)}",
        f"Shock in/after {int(args.event_year)}",
        "Never shock",
    ]
    group_colors = {
        group_order[0]: "tab:blue",
        group_order[1]: "tab:orange",
        group_order[2]: "tab:green",
    }

    raw = (
        df_evt.groupby("event_time", as_index=False)
        .agg(
            mean_outcome=(outcome_col, "mean"),
            sd_outcome=(outcome_col, "std"),
            n_obs=(outcome_col, "size"),
            n_firms=("c", "nunique"),
        )
        .sort_values("event_time")
    )
    raw["year"] = raw["event_time"] + int(args.event_year)
    raw["se_outcome"] = raw["sd_outcome"] / np.sqrt(raw["n_obs"].clip(lower=1))
    raw["lo"] = raw["mean_outcome"] - 1.96 * raw["se_outcome"]
    raw["hi"] = raw["mean_outcome"] + 1.96 * raw["se_outcome"]
    base_2015_rows = raw.loc[raw["year"] == 2015, "mean_outcome"]
    if base_2015_rows.empty:
        raise ValueError(
            "Year 2015 is not present in the selected plotting window; "
            "adjust --event-year / --event-time-min / --event-time-max."
        )
    base_mean_2015 = float(base_2015_rows.iloc[0])
    raw["mean_minus_2015"] = raw["mean_outcome"] - base_mean_2015
    ref_rows = raw.loc[raw["event_time"] == args.ref_event_time, "mean_outcome"]
    ref_mean = float(ref_rows.iloc[0]) if not ref_rows.empty else np.nan
    raw["mean_minus_ref"] = raw["mean_outcome"] - ref_mean

    pivot = df_evt.pivot_table(index="c", columns="event_time", values=outcome_col, aggfunc="mean")
    if args.ref_event_time not in pivot.columns:
        raise ValueError(
            f"Reference event time {args.ref_event_time} is missing from firm-level panel."
        )
    rows = []
    for r in range(args.event_time_min, args.event_time_max + 1):
        if r not in pivot.columns:
            rows.append(
                {
                    "event_time": r,
                    "diff_from_ref": np.nan,
                    "se": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                    "n_firms": 0,
                }
            )
            continue
        if r == args.ref_event_time:
            n_ref = int(pivot[args.ref_event_time].notna().sum())
            rows.append(
                {
                    "event_time": r,
                    "diff_from_ref": 0.0,
                    "se": 0.0,
                    "lo": 0.0,
                    "hi": 0.0,
                    "n_firms": n_ref,
                }
            )
            continue
        mask = pivot[args.ref_event_time].notna() & pivot[r].notna()
        diffs = pivot.loc[mask, r] - pivot.loc[mask, args.ref_event_time]
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "event_time": r,
                    "diff_from_ref": np.nan,
                    "se": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                    "n_firms": 0,
                }
            )
            continue
        mean_diff = float(diffs.mean())
        se = float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        rows.append(
            {
                "event_time": r,
                "diff_from_ref": mean_diff,
                "se": se,
                "lo": mean_diff - 1.96 * se if np.isfinite(se) else np.nan,
                "hi": mean_diff + 1.96 * se if np.isfinite(se) else np.nan,
                "n_firms": n,
            }
        )
    firm_diff = pd.DataFrame(rows).sort_values("event_time")
    firm_diff["year"] = firm_diff["event_time"] + int(args.event_year)
    raw_by_shock_group = (
        df_evt.groupby(["shock_group", "event_time"], as_index=False)
        .agg(
            mean_outcome=(outcome_col, "mean"),
            sd_outcome=(outcome_col, "std"),
            n_obs=(outcome_col, "size"),
            n_firms=("c", "nunique"),
        )
        .sort_values(["shock_group", "event_time"])
    )
    raw_by_shock_group["year"] = raw_by_shock_group["event_time"] + int(args.event_year)
    raw_by_shock_group["se_outcome"] = (
        raw_by_shock_group["sd_outcome"] / np.sqrt(raw_by_shock_group["n_obs"].clip(lower=1))
    )
    group_base_2015 = (
        raw_by_shock_group.loc[raw_by_shock_group["year"] == 2015, ["shock_group", "mean_outcome"]]
        .drop_duplicates(subset=["shock_group"])
        .rename(columns={"mean_outcome": "group_mean_2015"})
    )
    raw_by_shock_group = raw_by_shock_group.merge(group_base_2015, on="shock_group", how="left")
    raw_by_shock_group["mean_minus_2015"] = (
        raw_by_shock_group["mean_outcome"] - raw_by_shock_group["group_mean_2015"]
    )
    raw_z = (
        df_evt.groupby("event_time", as_index=False)
        .agg(
            mean_z=(instrument_col_raw, "mean"),
            sd_z=(instrument_col_raw, "std"),
            n_obs=(instrument_col_raw, "size"),
        )
        .sort_values("event_time")
    )
    raw_z["year"] = raw_z["event_time"] + int(args.event_year)
    raw_z["se_z"] = raw_z["sd_z"] / np.sqrt(raw_z["n_obs"].clip(lower=1))
    z_base_2015 = raw_z.loc[raw_z["year"] == 2015, "mean_z"]
    raw_z["mean_minus_2015_z"] = (
        raw_z["mean_z"] - float(z_base_2015.iloc[0]) if not z_base_2015.empty else np.nan
    )
    raw_by_shock_group_z = (
        df_evt.groupby(["shock_group", "event_time"], as_index=False)
        .agg(
            mean_z=(instrument_col_raw, "mean"),
            sd_z=(instrument_col_raw, "std"),
            n_obs=(instrument_col_raw, "size"),
        )
        .sort_values(["shock_group", "event_time"])
    )
    raw_by_shock_group_z["year"] = raw_by_shock_group_z["event_time"] + int(args.event_year)
    raw_by_shock_group_z["se_z"] = (
        raw_by_shock_group_z["sd_z"] / np.sqrt(raw_by_shock_group_z["n_obs"].clip(lower=1))
    )
    z_group_base_2015 = (
        raw_by_shock_group_z.loc[raw_by_shock_group_z["year"] == 2015, ["shock_group", "mean_z"]]
        .drop_duplicates(subset=["shock_group"])
        .rename(columns={"mean_z": "group_mean_z_2015"})
    )
    raw_by_shock_group_z = raw_by_shock_group_z.merge(
        z_group_base_2015,
        on="shock_group",
        how="left",
    )
    raw_by_shock_group_z["mean_minus_2015_z"] = (
        raw_by_shock_group_z["mean_z"] - raw_by_shock_group_z["group_mean_z_2015"]
    )
    shock_group_counts = (
        df_evt[["c", "shock_group"]]
        .drop_duplicates()
        .groupby("shock_group", as_index=False)
        .agg(n_firms=("c", "nunique"))
    )
    shock_group_counts["shock_group"] = pd.Categorical(
        shock_group_counts["shock_group"], categories=group_order, ordered=True
    )
    shock_group_counts = shock_group_counts.sort_values("shock_group")

    print("\n[event_study_config]")
    print(
        pd.DataFrame(
            [
                {
                    "event_year": int(args.event_year),
                    "event_time_min": int(args.event_time_min),
                    "event_time_max": int(args.event_time_max),
                    "ref_event_time": int(args.ref_event_time),
                    "outcome_col": outcome_col,
                    "instrument_col_for_shocks": instrument_col_raw,
                    "event_shock_metric": (
                        "pct_change" if use_pct_change_for_event_assignment else "first_difference"
                    ),
                    "event_min_t_for_shocks": int(event_min_t),
                    "event_max_t_for_shocks": int(event_max_t),
                    "local_window_for_shocks": int(event_local_window),
                    "neighbor_shock_ratio": float(neighbor_shock_ratio),
                    "min_shock_abs": min_shock_abs,
                    "min_shock_quantile": min_shock_quantile,
                    "n_firms_total_in_window": int(df_evt["c"].nunique()),
                    "panel_source_key": panel_key,
                }
            ]
        )
    )

    print("\n[event_study_raw_means]")
    print(raw)
    print("\n[event_study_within_firm_diff_from_ref]")
    print(firm_diff)
    print("\n[event_study_shock_group_counts]")
    print(shock_group_counts)
    print("\n[event_study_raw_means_by_shock_group]")
    print(raw_by_shock_group)
    print("\n[event_study_z_means_by_shock_group]")
    print(raw_by_shock_group_z)

    if not args.plot:
        return

    _slides_out_raw = str(reg_cfg.get("slides_out_dir", "")).strip()
    if _slides_out_raw:
        from company_shift_share.config_loader import load_config as _lc
        from pathlib import Path as _Path
        _root = str(_Path(__file__).resolve().parents[2])
        _slides_out = _Path(_slides_out_raw.replace("{root}", _root))
        _slides_out.mkdir(parents=True, exist_ok=True)
    else:
        _slides_out = None

    _oc_tag = outcome_col.replace("/", "_").replace(" ", "_")

    def _savefig(name: str) -> None:
        if _slides_out is not None:
            stem, _, ext = name.rpartition(".")
            tagged = f"{stem}_{_oc_tag}.{ext}"
            plt.savefig(_slides_out / tagged, dpi=150, bbox_inches="tight")
            print(f"[info] Saved {tagged}")

    plt.figure(figsize=(7.0, 4.5))
    plt.errorbar(
        firm_diff["year"],
        firm_diff["diff_from_ref"],
        yerr=1.96 * firm_diff["se"].fillna(0),
        fmt="o-",
        capsize=3,
        color="tab:orange",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(int(args.event_year), color="black", linestyle="--", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel(f"Within-firm diff vs event_time={args.ref_event_time}")
    plt.title("Within-firm event study (year axis)")

    plt.tight_layout()
    _savefig("opt_2016_es_main.png")
    plt.show()

    plt.figure(figsize=(9, 4.8))
    for grp in group_order:
        s = raw_by_shock_group.loc[raw_by_shock_group["shock_group"] == grp].sort_values("event_time")
        if s.empty:
            continue
        n_grp = int(
            shock_group_counts.loc[shock_group_counts["shock_group"] == grp, "n_firms"].sum()
        )
        plt.errorbar(
            s["year"],
            s["mean_minus_2015"],
            yerr=1.96 * s["se_outcome"].fillna(0),
            fmt="o-",
            capsize=3,
            color=group_colors.get(grp, None),
            label=f"{grp} (n_firms={n_grp})",
        )
    plt.axvline(int(args.event_year), color="black", linestyle="--", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel(f"Mean {outcome_col} - group mean in 2015")
    plt.title("Year means by instrument-shock timing group (relative to 2015)")
    plt.legend()
    plt.tight_layout()
    _savefig("opt_2016_es_by_shock_group.png")
    plt.show()

    plt.figure(figsize=(9, 4.8))
    for grp in group_order:
        s = raw_by_shock_group_z.loc[raw_by_shock_group_z["shock_group"] == grp].sort_values("event_time")
        if s.empty:
            continue
        n_grp = int(
            shock_group_counts.loc[shock_group_counts["shock_group"] == grp, "n_firms"].sum()
        )
        plt.errorbar(
            s["year"],
            s["mean_minus_2015_z"],
            yerr=1.96 * s["se_z"].fillna(0),
            fmt="o-",
            capsize=3,
            color=group_colors.get(grp, None),
            label=f"{grp} (n_firms={n_grp})",
        )
    plt.axvline(int(args.event_year), color="black", linestyle="--", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel(f"Mean {instrument_col_raw} - group mean in 2015")
    plt.title("Year means of z by instrument-shock timing group (relative to 2015)")
    plt.legend()
    plt.tight_layout()
    _savefig("opt_2016_es_z_by_shock_group.png")
    plt.show()


if __name__ == "__main__":
    main()
