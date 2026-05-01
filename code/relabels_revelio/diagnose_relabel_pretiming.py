"""Diagnostics for apparent pre-timing in generalized relabel event studies.

The script leaves the production relabel pipeline untouched.  It reuses the
verified generalized relabel panel and builds three memo inputs:

1. FOIA source-vs-target F-1 counts around the IPEDS relabel year.
2. IPEDS source-vs-target completions around the same relabel year.
3. Princeton Economics PhD program start-to-end length over time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb as ddb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import f1_foia.econ_relabels_opt_usage as base
import relabels_revelio.relabel_events_generalized as generalized


DEFAULT_PANEL = generalized.DEFAULT_PANEL_PARQUET
DEFAULT_OUT_DIR = generalized.DEFAULT_OUTPUT_DIR / "pretrend_timing_diagnostics"
EVENT_MIN = -5
EVENT_MAX = 4
DEGREE_TYPE = "Master"

SIDE_LABELS = {
    "source": "Source CIP group",
    "target": "Target CIP group",
}

PLOT_PALETTE = {
    "source": "#4c78a8",
    "target": "#e45756",
}


def _progress(message: str) -> None:
    print(f"[pretrend_diagnostics] {message}", flush=True)


def _sql_literal(value: object) -> str:
    return str(value).replace("'", "''")


def _event_id(row: pd.Series) -> str:
    return (
        f"{int(row['unitid'])}|{int(row['awlevel'])}|"
        f"{int(row['relabel_year'])}|{row['broad_pair_bin']}"
    )


def _load_master_events(panel_path: str | Path = DEFAULT_PANEL) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    events = panel[
        (panel["degree_type"].astype(str) == DEGREE_TYPE)
        & (panel["broad_bin_eligible"].eq(1))
        & (panel["event_flag"].eq(1))
        & (panel["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"]))
    ].copy()
    events = events.drop_duplicates(
        subset=["unitid", "awlevel", "relabel_year", "broad_pair_bin"]
    )
    events["event_id"] = events.apply(_event_id, axis=1)
    events["foia_degree_label"] = "MASTER'S"
    return events.reset_index(drop=True)


def _load_master_panel_rows(panel_path: str | Path = DEFAULT_PANEL) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    panel = panel[
        (panel["degree_type"].astype(str) == DEGREE_TYPE)
        & (panel["broad_bin_eligible"].eq(1))
        & (panel["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"]))
    ].copy()
    panel["event_id"] = panel.apply(_event_id, axis=1)
    panel["event_t"] = pd.to_numeric(panel["year"], errors="coerce") - pd.to_numeric(
        panel["relabel_year"], errors="coerce"
    )
    panel = panel[panel["event_t"].between(EVENT_MIN, EVENT_MAX)].copy()
    return panel


def _cip_map(ipeds_path: str | Path = base.IPEDS_PATH) -> dict[str, str]:
    return generalized._load_ipeds_cip_map(ipeds_path)


def _event_membership(
    events: pd.DataFrame,
    *,
    ipeds_path: str | Path = base.IPEDS_PATH,
    mode: str = "broad",
) -> pd.DataFrame:
    if mode not in {"broad", "exact"}:
        raise ValueError("mode must be 'broad' or 'exact'")

    rows: list[dict[str, object]] = []
    if mode == "broad":
        membership = generalized.build_broad_bin_membership(_cip_map(ipeds_path).keys())
        for row in events.itertuples(index=False):
            pair_bin = str(row.broad_pair_bin)
            for side, key in (("source", "source_cips"), ("target", "target_cips")):
                for cip6 in sorted(membership[pair_bin][key]):
                    rows.append(
                        {
                            "event_id": row.event_id,
                            "unitid": int(row.unitid),
                            "relabel_year": int(row.relabel_year),
                            "broad_pair_bin": pair_bin,
                            "side": side,
                            "cip6": str(cip6).zfill(6),
                        }
                    )
    else:
        for row in events.itertuples(index=False):
            for side, cip6 in (
                ("source", row.event_source_cip6),
                ("target", row.target_cip6),
            ):
                if pd.notna(cip6):
                    rows.append(
                        {
                            "event_id": row.event_id,
                            "unitid": int(row.unitid),
                            "relabel_year": int(row.relabel_year),
                            "broad_pair_bin": str(row.broad_pair_bin),
                            "side": side,
                            "cip6": str(cip6).zfill(6),
                        }
                    )
    return pd.DataFrame(rows).drop_duplicates()


def _event_time_grid(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in events.itertuples(index=False):
        for event_t in range(EVENT_MIN, EVENT_MAX + 1):
            rows.append(
                {
                    "event_id": row.event_id,
                    "unitid": int(row.unitid),
                    "relabel_year": int(row.relabel_year),
                    "broad_pair_bin": str(row.broad_pair_bin),
                    "event_t": int(event_t),
                    "calendar_year": int(row.relabel_year) + int(event_t),
                }
            )
    return pd.DataFrame(rows)


def compute_ipeds_source_target_counts(
    *,
    panel_path: str | Path = DEFAULT_PANEL,
    ipeds_path: str | Path = base.IPEDS_PATH,
    mode: str = "broad",
) -> pd.DataFrame:
    if mode == "broad":
        panel = _load_master_panel_rows(panel_path)
        source = panel[
            [
                "event_id",
                "unitid",
                "relabel_year",
                "broad_pair_bin",
                "event_t",
                "year",
                "source_total",
            ]
        ].rename(columns={"source_total": "count", "year": "calendar_year"})
        source["side"] = "source"
        target = panel[
            [
                "event_id",
                "unitid",
                "relabel_year",
                "broad_pair_bin",
                "event_t",
                "year",
                "target_total",
            ]
        ].rename(columns={"target_total": "count", "year": "calendar_year"})
        target["side"] = "target"
        out = pd.concat([source, target], ignore_index=True)
        out["dataset"] = "IPEDS completions"
        out["mode"] = mode
        out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0.0)
        return out

    events = _load_master_events(panel_path)
    membership = _event_membership(events, mode="exact")
    grid = _event_time_grid(events)
    con = ddb.connect()
    con.register("events_grid_py", grid)
    con.register("event_membership_py", membership)
    out = con.sql(
        f"""
        WITH counts AS (
            SELECT
                m.event_id,
                g.event_t,
                g.calendar_year,
                m.side,
                SUM(CAST(i.ctotalt AS DOUBLE)) AS count
            FROM events_grid_py g
            JOIN event_membership_py m
              ON g.event_id = m.event_id
            LEFT JOIN read_parquet('{_sql_literal(str(ipeds_path))}') i
              ON CAST(i.unitid AS BIGINT) = CAST(g.unitid AS BIGINT)
             AND CAST(i.year AS INTEGER) = CAST(g.calendar_year AS INTEGER)
             AND CAST(i.awlevel AS INTEGER) = 7
             AND LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = m.cip6
            GROUP BY m.event_id, g.event_t, g.calendar_year, m.side
        )
        SELECT
            g.event_id,
            g.unitid,
            g.relabel_year,
            g.broad_pair_bin,
            g.event_t,
            g.calendar_year,
            m.side,
            COALESCE(c.count, 0.0) AS count
        FROM events_grid_py g
        JOIN (SELECT DISTINCT event_id, side FROM event_membership_py) m
          ON g.event_id = m.event_id
        LEFT JOIN counts c
          ON g.event_id = c.event_id
         AND g.event_t = c.event_t
         AND m.side = c.side
        """
    ).df()
    out["dataset"] = "IPEDS completions"
    out["mode"] = mode
    return out


def compute_foia_source_target_counts(
    *,
    panel_path: str | Path = DEFAULT_PANEL,
    foia_path: str | Path = base.FOIA_PATH,
    inst_cw_path: str | Path = base.F1_INST_CW_PATH,
    ipeds_path: str | Path = base.IPEDS_PATH,
    mode: str = "broad",
) -> pd.DataFrame:
    events = _load_master_events(panel_path)
    membership = _event_membership(events, ipeds_path=ipeds_path, mode=mode)
    grid = _event_time_grid(events)
    con = ddb.connect()
    con.register("events_grid_py", grid)
    con.register("event_membership_py", membership)
    foia_path = Path(foia_path)
    inst_cw_path = Path(inst_cw_path)
    out = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                CAST(cw.UNITID AS BIGINT) AS unitid,
                LPAD(
                    CAST(
                        TRY_CAST(
                            REGEXP_REPLACE(CAST(fr.major_1_cip_code AS VARCHAR), '[^0-9]', '', 'g')
                            AS INTEGER
                        ) AS VARCHAR
                    ),
                    6,
                    '0'
                ) AS cip6,
                CAST(EXTRACT(YEAR FROM fr.program_end_date) AS INTEGER) AS grad_year,
                CAST(fr.student_key AS VARCHAR) AS student_id,
                fr.student_edu_level_desc AS degree_label,
                TRY_CAST(fr.year AS INTEGER) AS report_year
            FROM read_parquet('{_sql_literal(str(foia_path))}') fr
            LEFT JOIN read_parquet('{_sql_literal(str(inst_cw_path))}') cw
              ON fr.school_name = cw.school_name
            WHERE fr.program_end_date IS NOT NULL
              AND fr.student_edu_level_desc = 'MASTER''S'
              AND fr.student_key IS NOT NULL
              AND TRY_CAST(fr.year AS INTEGER) = CAST(EXTRACT(YEAR FROM fr.program_end_date) AS INTEGER)
        ),
        counts AS (
            SELECT
                g.event_id,
                g.unitid,
                g.relabel_year,
                g.broad_pair_bin,
                g.event_t,
                g.calendar_year,
                m.side,
                COUNT(DISTINCT f.student_id) AS count
            FROM events_grid_py g
            JOIN event_membership_py m
              ON g.event_id = m.event_id
            LEFT JOIN foia_base f
              ON f.unitid = CAST(g.unitid AS BIGINT)
             AND f.grad_year = g.calendar_year
             AND f.cip6 = m.cip6
            GROUP BY
                g.event_id,
                g.unitid,
                g.relabel_year,
                g.broad_pair_bin,
                g.event_t,
                g.calendar_year,
                m.side
        )
        SELECT * FROM counts
        ORDER BY event_id, event_t, side
        """
    ).df()
    out["dataset"] = "FOIA F-1 records"
    out["mode"] = mode
    return out


def summarize_counts(counts: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        counts.groupby(["dataset", "mode", "event_t", "side"], as_index=False)
        .agg(
            mean_count=("count", "mean"),
            sd_count=("count", "std"),
            total_count=("count", "sum"),
            n_events=("event_id", "nunique"),
            n_school_years=("event_id", "size"),
        )
        .sort_values(["dataset", "mode", "side", "event_t"])
    )
    grouped["se_count"] = grouped["sd_count"] / grouped["n_events"].pow(0.5)
    grouped["ci_low"] = grouped["mean_count"] - 1.96 * grouped["se_count"]
    grouped["ci_high"] = grouped["mean_count"] + 1.96 * grouped["se_count"]
    grouped["side_label"] = grouped["side"].map(SIDE_LABELS)
    return grouped


def summarize_switch_timing(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, mode), group in summary.groupby(["dataset", "mode"]):
        wide = group.pivot(index="event_t", columns="side", values="mean_count")
        wide["target_minus_source"] = wide.get("target") - wide.get("source")
        for start, end in [(-3, -2), (-2, -1), (-1, 0), (0, 1)]:
            if start in wide.index and end in wide.index:
                rows.append(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "interval": f"{start} to {end}",
                        "delta_target_minus_source": (
                            wide.loc[end, "target_minus_source"]
                            - wide.loc[start, "target_minus_source"]
                        ),
                        "start_target_minus_source": wide.loc[
                            start, "target_minus_source"
                        ],
                        "end_target_minus_source": wide.loc[
                            end, "target_minus_source"
                        ],
                    }
                )
    return pd.DataFrame(rows)


def summarize_switch_timing_by_bin(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, mode, broad_pair_bin), group in counts.groupby(
        ["dataset", "mode", "broad_pair_bin"]
    ):
        wide = (
            group.groupby(["event_t", "side"], as_index=False)["count"]
            .mean()
            .pivot(index="event_t", columns="side", values="count")
        )
        if "source" not in wide.columns or "target" not in wide.columns:
            continue
        wide["target_minus_source"] = wide["target"] - wide["source"]
        if not {-2, -1, 0}.issubset(set(wide.index)):
            continue
        rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "broad_pair_bin": broad_pair_bin,
                "n_events": int(group["event_id"].nunique()),
                "target_minus_source_t_minus_2": wide.loc[-2, "target_minus_source"],
                "target_minus_source_t_minus_1": wide.loc[-1, "target_minus_source"],
                "target_minus_source_t_0": wide.loc[0, "target_minus_source"],
                "change_t_minus_2_to_minus_1": (
                    wide.loc[-1, "target_minus_source"]
                    - wide.loc[-2, "target_minus_source"]
                ),
                "change_t_minus_1_to_0": (
                    wide.loc[0, "target_minus_source"]
                    - wide.loc[-1, "target_minus_source"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mode", "dataset", "n_events", "broad_pair_bin"],
        ascending=[True, True, False, True],
    )


def plot_source_target_counts(
    summary: pd.DataFrame,
    *,
    dataset: str,
    mode: str,
    out_dir: str | Path,
) -> Path:
    out_dir = Path(out_dir)
    plot_df = summary[
        (summary["dataset"].eq(dataset))
        & (summary["mode"].eq(mode))
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for side, side_df in plot_df.groupby("side"):
        side_df = side_df.sort_values("event_t")
        ax.plot(
            side_df["event_t"],
            side_df["mean_count"],
            marker="o",
            linewidth=2,
            color=PLOT_PALETTE.get(side),
            label=SIDE_LABELS.get(side, side),
        )
        ax.fill_between(
            side_df["event_t"].astype(float).to_numpy(),
            side_df["ci_low"].astype(float).to_numpy(),
            side_df["ci_high"].astype(float).to_numpy(),
            color=PLOT_PALETTE.get(side),
            alpha=0.15,
            linewidth=0,
        )
    ax.axvline(0, color="0.45", linestyle="--", linewidth=1)
    ax.axvline(-1, color="0.70", linestyle=":", linewidth=1)
    ax.set_xlabel("Years relative to IPEDS relabel year")
    ax.set_ylabel("Mean raw count per relabel event")
    ax.set_title(f"{dataset}: source vs. target counts around relabels")
    ax.legend(title=None)
    ax.grid(True, color="0.9")
    fig.tight_layout()
    out_path = out_dir / f"{dataset.lower().replace(' ', '_').replace('-', '')}_{mode}_source_target_counts.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_combined_source_target_counts(summary: pd.DataFrame, *, mode: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    datasets = ["FOIA F-1 records", "IPEDS completions"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    for ax, dataset in zip(axes, datasets, strict=True):
        plot_df = summary[
            (summary["dataset"].eq(dataset))
            & (summary["mode"].eq(mode))
        ].copy()
        for side, side_df in plot_df.groupby("side"):
            side_df = side_df.sort_values("event_t")
            ax.plot(
                side_df["event_t"],
                side_df["mean_count"],
                marker="o",
                linewidth=2,
                color=PLOT_PALETTE.get(side),
                label=SIDE_LABELS.get(side, side),
            )
            ax.fill_between(
                side_df["event_t"].astype(float).to_numpy(),
                side_df["ci_low"].astype(float).to_numpy(),
                side_df["ci_high"].astype(float).to_numpy(),
                color=PLOT_PALETTE.get(side),
                alpha=0.15,
                linewidth=0,
            )
        ax.axvline(0, color="0.45", linestyle="--", linewidth=1)
        ax.axvline(-1, color="0.70", linestyle=":", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel("Years relative to IPEDS relabel year")
        ax.grid(True, color="0.9")
    axes[0].set_ylabel("Mean raw count per relabel event")
    axes[1].legend(title=None, loc="best")
    fig.tight_layout()
    out_path = out_dir / f"foia_ipeds_{mode}_source_target_counts_combined.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def compute_princeton_econ_phd_lengths(
    *,
    foia_path: str | Path = base.FOIA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = ddb.connect()
    p = Path(foia_path)
    student_programs = con.sql(
        f"""
        WITH raw AS (
            SELECT
                CAST(student_key AS VARCHAR) AS student_id,
                TRY_CAST(year AS INTEGER) AS report_year,
                LPAD(
                    CAST(
                        TRY_CAST(
                            REGEXP_REPLACE(CAST(major_1_cip_code AS VARCHAR), '[^0-9]', '', 'g')
                            AS INTEGER
                        ) AS VARCHAR
                    ),
                    6,
                    '0'
                ) AS cip6,
                program_start_date,
                program_end_date
            FROM read_parquet('{_sql_literal(str(p))}')
            WHERE school_name = 'Princeton University'
              AND student_edu_level_desc = 'DOCTORATE'
              AND student_key IS NOT NULL
              AND program_start_date IS NOT NULL
              AND program_end_date IS NOT NULL
        ),
        econ AS (
            SELECT *
            FROM raw
            WHERE cip6 IN ('450601', '450603')
        ),
        latest AS (
            SELECT
                student_id,
                MIN(program_start_date) AS program_start_date,
                MAX(program_end_date) AS program_end_date,
                MAX(report_year) AS latest_report_year,
                COUNT(DISTINCT CAST(EXTRACT(YEAR FROM program_end_date) AS INTEGER)) AS distinct_end_years_seen
            FROM econ
            GROUP BY student_id
        )
        SELECT
            student_id,
            CAST(EXTRACT(YEAR FROM program_start_date) AS INTEGER) AS start_year,
            CAST(EXTRACT(YEAR FROM program_end_date) AS INTEGER) AS end_year,
            latest_report_year,
            distinct_end_years_seen,
            DATE_DIFF('day', program_start_date, program_end_date) / 365.25 AS program_length_years
        FROM latest
        WHERE DATE_DIFF('day', program_start_date, program_end_date) / 365.25 BETWEEN 3 AND 10
        """
    ).df()

    end_matched = con.sql(
        f"""
        WITH raw AS (
            SELECT
                CAST(student_key AS VARCHAR) AS student_id,
                TRY_CAST(year AS INTEGER) AS report_year,
                LPAD(
                    CAST(
                        TRY_CAST(
                            REGEXP_REPLACE(CAST(major_1_cip_code AS VARCHAR), '[^0-9]', '', 'g')
                            AS INTEGER
                        ) AS VARCHAR
                    ),
                    6,
                    '0'
                ) AS cip6,
                program_start_date,
                program_end_date
            FROM read_parquet('{_sql_literal(str(p))}')
            WHERE school_name = 'Princeton University'
              AND student_edu_level_desc = 'DOCTORATE'
              AND student_key IS NOT NULL
              AND program_start_date IS NOT NULL
              AND program_end_date IS NOT NULL
        ),
        econ AS (
            SELECT *
            FROM raw
            WHERE cip6 IN ('450601', '450603')
              AND report_year = CAST(EXTRACT(YEAR FROM program_end_date) AS INTEGER)
        ),
        dedup AS (
            SELECT
                student_id,
                program_start_date,
                program_end_date,
                CAST(EXTRACT(YEAR FROM program_start_date) AS INTEGER) AS start_year,
                CAST(EXTRACT(YEAR FROM program_end_date) AS INTEGER) AS end_year,
                DATE_DIFF('day', program_start_date, program_end_date) / 365.25 AS program_length_years
            FROM econ
            GROUP BY student_id, program_start_date, program_end_date
        )
        SELECT *
        FROM dedup
        WHERE program_length_years BETWEEN 3 AND 10
        """
    ).df()
    return student_programs, end_matched


def summarize_program_lengths(student_programs: pd.DataFrame, *, time_col: str) -> pd.DataFrame:
    out = (
        student_programs.dropna(subset=[time_col, "program_length_years"])
        .groupby(time_col, as_index=False)
        .agg(
            mean_years=("program_length_years", "mean"),
            sd_years=("program_length_years", "std"),
            n_students=("student_id", "nunique"),
            median_years=("program_length_years", "median"),
        )
        .sort_values(time_col)
    )
    out["se_years"] = out["sd_years"] / out["n_students"].pow(0.5)
    out["ci_low"] = out["mean_years"] - 1.96 * out["se_years"]
    out["ci_high"] = out["mean_years"] + 1.96 * out["se_years"]
    return out


def plot_princeton_lengths(summary: pd.DataFrame, *, time_col: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    plot_df = summary[summary["n_students"] >= 3].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.errorbar(
        plot_df[time_col],
        plot_df["mean_years"],
        yerr=1.96 * plot_df["se_years"],
        fmt="o-",
        color="#2f6f4e",
        capsize=4,
        linewidth=2,
        markersize=5,
    )
    ax.axhline(5, color="0.55", linestyle=":", linewidth=1, label="5 years")
    ax.axhline(6, color="0.25", linestyle="--", linewidth=1, label="6 years")
    ax.set_xlabel(time_col.replace("_", " ").title())
    ax.set_ylabel("Average program length, years")
    ax.set_title("Princeton Economics PhD: FOIA program start-to-end length")
    ax.grid(True, color="0.9")
    ax.legend(title=None)
    fig.tight_layout()
    out_path = out_dir / f"princeton_econ_phd_program_length_by_{time_col}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--foia", default=base.FOIA_PATH)
    parser.add_argument("--inst-cw", default=base.F1_INST_CW_PATH)
    parser.add_argument("--ipeds", default=base.IPEDS_PATH)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--membership-mode", choices=["broad", "exact", "both"], default="broad")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["broad", "exact"] if args.membership_mode == "both" else [args.membership_mode]
    all_counts: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    plot_paths: list[Path] = []
    for mode in modes:
        _progress(f"Computing IPEDS source/target counts ({mode})")
        ipeds_counts = compute_ipeds_source_target_counts(
            panel_path=args.panel,
            ipeds_path=args.ipeds,
            mode=mode,
        )
        _progress(f"Computing FOIA source/target counts ({mode})")
        foia_counts = compute_foia_source_target_counts(
            panel_path=args.panel,
            foia_path=args.foia,
            inst_cw_path=args.inst_cw,
            ipeds_path=args.ipeds,
            mode=mode,
        )
        counts = pd.concat([foia_counts, ipeds_counts], ignore_index=True)
        summary = summarize_counts(counts)
        counts.to_csv(out_dir / f"source_target_event_time_counts_{mode}.csv", index=False)
        summary.to_csv(out_dir / f"source_target_event_time_summary_{mode}.csv", index=False)
        all_counts.append(counts)
        all_summary.append(summary)
        for dataset in ["FOIA F-1 records", "IPEDS completions"]:
            plot_paths.append(
                plot_source_target_counts(
                    summary,
                    dataset=dataset,
                    mode=mode,
                    out_dir=out_dir,
                )
            )
        plot_paths.append(plot_combined_source_target_counts(summary, mode=mode, out_dir=out_dir))

    combined_summary = pd.concat(all_summary, ignore_index=True)
    combined_counts = pd.concat(all_counts, ignore_index=True)
    switch_summary = summarize_switch_timing(combined_summary)
    switch_by_bin = summarize_switch_timing_by_bin(combined_counts)
    combined_summary.to_csv(out_dir / "source_target_event_time_summary_all_modes.csv", index=False)
    switch_summary.to_csv(out_dir / "source_target_switch_timing_summary.csv", index=False)
    switch_by_bin.to_csv(out_dir / "source_target_switch_timing_by_bin_summary.csv", index=False)

    _progress("Computing Princeton Economics PhD program-length diagnostics")
    latest_programs, end_matched_programs = compute_princeton_econ_phd_lengths(foia_path=args.foia)
    latest_programs.to_csv(out_dir / "princeton_econ_phd_latest_student_programs.csv", index=False)
    end_matched_programs.to_csv(out_dir / "princeton_econ_phd_end_matched_records.csv", index=False)
    start_summary = summarize_program_lengths(latest_programs, time_col="start_year")
    end_summary = summarize_program_lengths(end_matched_programs, time_col="end_year")
    start_summary.to_csv(out_dir / "princeton_econ_phd_length_by_start_year.csv", index=False)
    end_summary.to_csv(out_dir / "princeton_econ_phd_length_by_end_year.csv", index=False)
    plot_paths.append(plot_princeton_lengths(start_summary, time_col="start_year", out_dir=out_dir))
    plot_paths.append(plot_princeton_lengths(end_summary, time_col="end_year", out_dir=out_dir))

    plot_index = pd.DataFrame({"plot_path": [str(path) for path in plot_paths]})
    plot_index.to_csv(out_dir / "plot_index.csv", index=False)
    _progress(f"Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
