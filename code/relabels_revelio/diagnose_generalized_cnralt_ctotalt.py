"""Diagnostics for generalized relabel raw means in IPEDS count outcomes.

The raw event-time plots in relabel_events_generalized use the DiD panel. Under
the individual-row DiD spec, IPEDS school/program counts are repeated once per
FOIA student. This script rebuilds a pair-year IPEDS panel from the matched
treated/control schools so we can separate count-path composition from
student-weighting and late-window sample attrition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relabels_revelio import relabel_events_generalized as reg


OUTCOMES = ("ctotalt", "cnralt")


def _load_school_names(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({"unitid": pd.Series(dtype="int64"), "school_name": pd.Series(dtype="object")})
    try:
        raw = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame({"unitid": pd.Series(dtype="int64"), "school_name": pd.Series(dtype="object")})

    unit_col = next((c for c in ("unitid", "UNITID", "main_unitid") if c in raw.columns), None)
    name_col = next((c for c in ("instname", "school_name", "INSTNM", "institution_name", "ipeds_name") if c in raw.columns), None)
    if unit_col is None or name_col is None:
        return pd.DataFrame({"unitid": pd.Series(dtype="int64"), "school_name": pd.Series(dtype="object")})
    out = raw[[unit_col, name_col]].rename(columns={unit_col: "unitid", name_col: "school_name"}).copy()
    out["unitid"] = pd.to_numeric(out["unitid"], errors="coerce")
    out = out.dropna(subset=["unitid"]).drop_duplicates("unitid")
    out["unitid"] = out["unitid"].astype("int64")
    return out


def _pair_ipeds_panel(
    con: duckdb.DuckDBPyConnection,
    matched_pairs: pd.DataFrame,
    *,
    ipeds_path: Path,
    event_min: int,
    event_max: int,
) -> pd.DataFrame:
    reg._ensure_ipeds_view(con, ipeds_path)
    cip_map = reg._load_ipeds_cip_map(ipeds_path)
    broad_membership = reg.build_broad_bin_membership(cip_map.keys())
    broad_any_cips = reg._broad_membership_rows(broad_membership, side="all")
    con.register("diag_pairs_py", matched_pairs)
    con.register("diag_broad_any_cips_py", broad_any_cips)
    return con.sql(
        f"""
        WITH pair_roles AS (
            SELECT
                CAST(pair_id AS BIGINT) AS pair_id,
                CAST(relabel_year AS INTEGER) AS relabel_year,
                CAST(relabel_type AS VARCHAR) AS relabel_type,
                CAST(degree_type AS VARCHAR) AS degree_type,
                CAST(awlevel AS INTEGER) AS awlevel,
                CAST(broad_pair_bin AS VARCHAR) AS broad_pair_bin,
                CAST(treated_unitid AS BIGINT) AS unitid,
                1 AS treated
            FROM diag_pairs_py
            UNION ALL
            SELECT
                CAST(pair_id AS BIGINT) AS pair_id,
                CAST(relabel_year AS INTEGER) AS relabel_year,
                CAST(relabel_type AS VARCHAR) AS relabel_type,
                CAST(degree_type AS VARCHAR) AS degree_type,
                CAST(awlevel AS INTEGER) AS awlevel,
                CAST(broad_pair_bin AS VARCHAR) AS broad_pair_bin,
                CAST(control_unitid AS BIGINT) AS unitid,
                0 AS treated
            FROM diag_pairs_py
        ),
        event_grid AS (
            SELECT
                p.*,
                CAST(t.event_t AS INTEGER) AS event_t,
                CAST(p.relabel_year + t.event_t AS INTEGER) AS calendar_year
            FROM pair_roles p
            CROSS JOIN generate_series({int(event_min)}, {int(event_max)}) AS t(event_t)
        )
        SELECT
            g.pair_id,
            g.relabel_year,
            g.relabel_type,
            g.degree_type,
            g.awlevel,
            g.broad_pair_bin,
            g.unitid,
            g.treated,
            g.event_t,
            g.calendar_year,
            COALESCE(SUM(CAST(i.ctotalt AS DOUBLE)), 0.0) AS ctotalt,
            COALESCE(SUM(CAST(i.cnralt AS DOUBLE)), 0.0) AS cnralt
        FROM event_grid g
        JOIN diag_broad_any_cips_py c
          ON c.broad_pair_bin = g.broad_pair_bin
        LEFT JOIN ipeds_raw i
          ON CAST(i.unitid AS BIGINT) = g.unitid
         AND CAST(i.awlevel AS INTEGER) = g.awlevel
         AND CAST(i.year AS INTEGER) = g.calendar_year
         AND LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = c.cip6
        GROUP BY
            g.pair_id,
            g.relabel_year,
            g.relabel_type,
            g.degree_type,
            g.awlevel,
            g.broad_pair_bin,
            g.unitid,
            g.treated,
            g.event_t,
            g.calendar_year
        """
    ).df()


def _summarize_pair_means(pair_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome in OUTCOMES:
        for (event_t, treated), g in pair_panel.groupby(["event_t", "treated"], observed=True):
            vals = pd.to_numeric(g[outcome], errors="coerce")
            rows.append(
                {
                    "outcome": outcome,
                    "event_t": int(event_t),
                    "treated": int(treated),
                    "mean_pair": vals.mean(),
                    "median_pair": vals.median(),
                    "p90_pair": vals.quantile(0.90),
                    "p99_pair": vals.quantile(0.99),
                    "n_pair_years": int(vals.notna().sum()),
                    "n_units": int(g["unitid"].nunique()),
                    "n_relabel_years": int(g["relabel_year"].nunique()),
                    "n_zero": int(vals.eq(0).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["outcome", "treated", "event_t"]).reset_index(drop=True)


def _student_weighted_summary(did_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if did_panel.empty:
        return pd.DataFrame()
    did = did_panel[did_panel["event_t"].between(reg.PLOT_EVENT_MIN, reg.PLOT_EVENT_MAX)].copy()
    for outcome in OUTCOMES:
        if outcome not in did.columns:
            continue
        for (event_t, treated), g in did.groupby(["event_t", "treated"], observed=True):
            vals = pd.to_numeric(g[outcome], errors="coerce")
            rows.append(
                {
                    "outcome": outcome,
                    "event_t": int(event_t),
                    "treated": int(treated),
                    "mean_student_row": vals.mean(),
                    "n_student_rows": int(vals.notna().sum()),
                    "n_pair_years_with_students": int(g[["pair_id", "unitid", "calendar_year"]].drop_duplicates().shape[0]),
                    "total_grads": float(pd.to_numeric(g.get("total_grads"), errors="coerce").fillna(0).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["outcome", "treated", "event_t"]).reset_index(drop=True)


def _foia_student_weights(
    con: duckdb.DuckDBPyConnection,
    matched_pairs: pd.DataFrame,
    *,
    foia_path: Path,
    inst_cw_path: Path,
    ipeds_path: Path,
) -> pd.DataFrame:
    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{reg._sql_literal(str(foia_path))}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_cw AS SELECT * FROM read_parquet('{reg._sql_literal(str(inst_cw_path))}')")
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW foia_raw_with_rownum AS
        SELECT
            CAST(ROW_NUMBER() OVER () - 1 AS BIGINT) AS original_row_num,
            *
        FROM read_parquet('{reg._sql_literal(str(foia_path))}')
        """
    )
    schema = reg.v2._resolve_foia_schema(con)
    cip_map = reg._load_ipeds_cip_map(ipeds_path)
    broad_membership = reg.build_broad_bin_membership(cip_map.keys())
    broad_any_cips = reg._broad_membership_rows(broad_membership, side="all")
    con.register("diag_pairs_weights_py", matched_pairs)
    con.register("diag_broad_any_cips_weights_py", broad_any_cips)

    foia_inst_col = str(schema["foia_inst_col"])
    foia_cip_col = str(schema["foia_cip_col"])
    foia_end_col = str(schema["foia_end_col"])
    foia_student_col = str(schema["foia_student_col"])
    foia_edu_col = str(schema["foia_edu_col"])
    cw_inst_col = str(schema["cw_inst_col"])
    cw_unitid_col = str(schema["cw_unitid_col"])
    foia_year_col = schema["foia_year_col"]
    norm_cip_expr = reg.base.normalize_cip_sql(foia_cip_col)
    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )
    grad_year_max_clause = f"AND CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) <= {int(reg.ANALYSIS_ORIGINAL_YEAR_MAX)}"
    degree_case_pairs = reg._foia_degree_case("p.degree_type")

    return con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0') AS cip6,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                {foia_edu_col} AS foia_degree_label
            FROM foia_raw_with_rownum fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE {foia_end_col} IS NOT NULL
              {year_match_clause}
              {grad_year_max_clause}
        ),
        relevant_foia AS (
            SELECT *
            FROM foia_base
            WHERE unitid IS NOT NULL
              AND cip6 IS NOT NULL
              AND grad_year IS NOT NULL
        ),
        matched_treated AS (
            SELECT
                p.pair_id,
                CAST(p.treated_unitid AS BIGINT) AS unitid,
                1 AS treated,
                p.relabel_year,
                p.relabel_type,
                p.degree_type,
                p.broad_pair_bin,
                f.grad_year AS calendar_year,
                f.student_id
            FROM relevant_foia f
            JOIN diag_pairs_weights_py p
              ON f.unitid = p.treated_unitid
             AND f.foia_degree_label = {degree_case_pairs}
            JOIN diag_broad_any_cips_weights_py m
              ON m.broad_pair_bin = p.broad_pair_bin
             AND f.cip6 = m.cip6
        ),
        matched_control AS (
            SELECT
                p.pair_id,
                CAST(p.control_unitid AS BIGINT) AS unitid,
                0 AS treated,
                p.relabel_year,
                p.relabel_type,
                p.degree_type,
                p.broad_pair_bin,
                f.grad_year AS calendar_year,
                f.student_id
            FROM relevant_foia f
            JOIN diag_pairs_weights_py p
              ON f.unitid = p.control_unitid
             AND f.foia_degree_label = {degree_case_pairs}
            JOIN diag_broad_any_cips_weights_py m
              ON m.broad_pair_bin = p.broad_pair_bin
             AND f.cip6 = m.cip6
        ),
        matched_all AS (
            SELECT * FROM matched_treated
            UNION ALL
            SELECT * FROM matched_control
        )
        SELECT
            pair_id,
            unitid,
            treated,
            relabel_year,
            relabel_type,
            degree_type,
            broad_pair_bin,
            calendar_year,
            CAST(calendar_year - relabel_year AS INTEGER) AS event_t,
            COUNT(*) AS student_rows,
            COUNT(DISTINCT student_id) AS total_grads
        FROM matched_all
        GROUP BY
            pair_id,
            unitid,
            treated,
            relabel_year,
            relabel_type,
            degree_type,
            broad_pair_bin,
            calendar_year
        """
    ).df()


def _student_weighted_summary_from_pair_panel(pair_panel: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    merged = pair_panel.merge(
        weights[["pair_id", "unitid", "treated", "event_t", "student_rows", "total_grads"]],
        on=["pair_id", "unitid", "treated", "event_t"],
        how="inner",
    )
    rows: list[dict[str, object]] = []
    for outcome in OUTCOMES:
        for (event_t, treated), g in merged.groupby(["event_t", "treated"], observed=True):
            vals = pd.to_numeric(g[outcome], errors="coerce")
            weights_num = pd.to_numeric(g["student_rows"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "outcome": outcome,
                    "event_t": int(event_t),
                    "treated": int(treated),
                    "mean_student_row": float(np.average(vals, weights=weights_num)) if weights_num.sum() else np.nan,
                    "mean_pair_with_students": vals.mean(),
                    "n_pair_years_with_students": int(len(g)),
                    "n_units_with_students": int(g["unitid"].nunique()),
                    "n_student_rows": float(weights_num.sum()),
                    "total_grads": float(pd.to_numeric(g["total_grads"], errors="coerce").fillna(0.0).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["outcome", "treated", "event_t"]).reset_index(drop=True)


def _merge_student_weights(pair_panel: pd.DataFrame, did_panel: pd.DataFrame) -> pd.DataFrame:
    if did_panel.empty or "pair_id" not in did_panel.columns:
        return pd.DataFrame()
    weights = (
        did_panel.assign(total_grads_num=pd.to_numeric(did_panel["total_grads"], errors="coerce").fillna(0.0))
        .groupby(["pair_id", "unitid", "treated", "event_t"], as_index=False, observed=True)
        .agg(student_rows=("total_grads_num", "size"), total_grads=("total_grads_num", "sum"))
    )
    merged = pair_panel.merge(weights, on=["pair_id", "unitid", "treated", "event_t"], how="left")
    merged["student_rows"] = pd.to_numeric(merged["student_rows"], errors="coerce").fillna(0.0)
    merged["total_grads"] = pd.to_numeric(merged["total_grads"], errors="coerce").fillna(0.0)
    return merged


def _influence_tables(pair_panel: pd.DataFrame, names: pd.DataFrame, *, reference_event: int) -> pd.DataFrame:
    ref = pair_panel[pair_panel["event_t"].eq(reference_event)][
        ["pair_id", "treated", "ctotalt", "cnralt"]
    ].rename(columns={"ctotalt": "ctotalt_ref", "cnralt": "cnralt_ref"})
    merged = pair_panel.merge(ref, on=["pair_id", "treated"], how="left")
    out = merged[merged["event_t"].ne(reference_event)].copy()
    for outcome in OUTCOMES:
        out[f"{outcome}_delta_from_ref"] = pd.to_numeric(out[outcome], errors="coerce") - pd.to_numeric(
            out[f"{outcome}_ref"], errors="coerce"
        )
    if "school_name" not in out.columns and not names.empty:
        out = out.merge(names, on="unitid", how="left")
    out["school_name"] = out.get("school_name", pd.Series(index=out.index, dtype="object")).fillna("")
    cols = [
        "event_t",
        "treated",
        "pair_id",
        "unitid",
        "school_name",
        "degree_type",
        "broad_pair_bin",
        "relabel_year",
        "calendar_year",
        "ctotalt",
        "ctotalt_ref",
        "ctotalt_delta_from_ref",
        "cnralt",
        "cnralt_ref",
        "cnralt_delta_from_ref",
    ]
    return out[cols].sort_values(["event_t", "treated", "pair_id"]).reset_index(drop=True)


def _top_contributors(influence: pd.DataFrame, *, event_t: int, top_n: int) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        delta_col = f"{outcome}_delta_from_ref"
        for treated in (0, 1):
            g = influence[(influence["event_t"].eq(event_t)) & (influence["treated"].eq(treated))].copy()
            if g.empty:
                continue
            total_delta = pd.to_numeric(g[delta_col], errors="coerce").sum()
            g["abs_delta"] = pd.to_numeric(g[delta_col], errors="coerce").abs()
            g["share_abs_delta"] = g["abs_delta"] / g["abs_delta"].sum() if g["abs_delta"].sum() else np.nan
            g["share_signed_delta"] = (
                pd.to_numeric(g[delta_col], errors="coerce") / total_delta if total_delta else np.nan
            )
            g["outcome"] = outcome
            rows.append(
                g.sort_values("abs_delta", ascending=False)
                .head(top_n)
                [
                    [
                        "outcome",
                        "event_t",
                        "treated",
                        "pair_id",
                        "unitid",
                        "school_name",
                        "degree_type",
                        "broad_pair_bin",
                        "relabel_year",
                        "calendar_year",
                        outcome,
                        f"{outcome}_ref",
                        delta_col,
                        "share_abs_delta",
                        "share_signed_delta",
                    ]
                ]
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _leave_top_out_summary(pair_panel: pd.DataFrame, influence: pd.DataFrame, *, event_t: int, top_n: int) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        delta_col = f"{outcome}_delta_from_ref"
        for treated in (0, 1):
            g = pair_panel[(pair_panel["event_t"].eq(event_t)) & (pair_panel["treated"].eq(treated))].copy()
            inf = influence[(influence["event_t"].eq(event_t)) & (influence["treated"].eq(treated))].copy()
            if g.empty or inf.empty:
                continue
            top_pairs = (
                inf.assign(abs_delta=pd.to_numeric(inf[delta_col], errors="coerce").abs())
                .sort_values("abs_delta", ascending=False)
                .head(top_n)["pair_id"]
                .tolist()
            )
            kept = g[~g["pair_id"].isin(top_pairs)]
            rows.append(
                {
                    "outcome": outcome,
                    "event_t": event_t,
                    "treated": treated,
                    "mean_all": pd.to_numeric(g[outcome], errors="coerce").mean(),
                    "mean_leave_top_out": pd.to_numeric(kept[outcome], errors="coerce").mean(),
                    "n_all": int(len(g)),
                    "n_kept": int(len(kept)),
                    "top_n_removed": int(top_n),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=reg.DEFAULT_PANEL_PARQUET)
    parser.add_argument("--ipeds", type=Path, default=Path(reg.base.IPEDS_PATH))
    parser.add_argument("--out-dir", type=Path, default=reg.DEFAULT_OUTPUT_DIR / "generalized_count_diagnostics")
    parser.add_argument("--event-min", type=int, default=reg.PLOT_EVENT_MIN)
    parser.add_argument("--event-max", type=int, default=reg.PLOT_EVENT_MAX)
    parser.add_argument("--reference-event", type=int, default=reg.DI_D_REFERENCE_EVENT_TIME)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--skip-did-panel", action="store_true")
    parser.add_argument("--skip-foia-weights", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    panel = pd.read_parquet(args.panel)
    panel = panel[panel["degree_type"].isin(reg.POOLED_DEGREE_TYPES)].copy()
    matches = reg.match_treated_to_never_treated(con, panel, ipeds_path=args.ipeds)
    matches = matches[matches["degree_type"].isin(reg.POOLED_DEGREE_TYPES)].copy()
    matches.to_csv(args.out_dir / "matched_pairs.csv", index=False)

    pair_panel = _pair_ipeds_panel(
        con,
        matches,
        ipeds_path=args.ipeds,
        event_min=args.event_min,
        event_max=args.event_max,
    )
    names = _load_school_names(Path(reg.DEFAULT_IPEDS_MAIN_INSTITUTIONS_PATH))
    if not names.empty:
        pair_panel = pair_panel.merge(names, on="unitid", how="left")
    pair_panel.to_csv(args.out_dir / "pair_event_ipeds_counts.csv", index=False)

    pair_summary = _summarize_pair_means(pair_panel)
    pair_summary.to_csv(args.out_dir / "pair_event_count_means.csv", index=False)

    influence = _influence_tables(pair_panel, names, reference_event=args.reference_event)
    influence.to_csv(args.out_dir / "pair_event_deltas_from_reference.csv", index=False)
    top = _top_contributors(influence, event_t=args.event_max, top_n=args.top_n)
    top.to_csv(args.out_dir / f"top_contributors_event_t{args.event_max}.csv", index=False)
    leave_top = _leave_top_out_summary(pair_panel, influence, event_t=args.event_max, top_n=args.top_n)
    leave_top.to_csv(args.out_dir / f"leave_top{args.top_n}_out_event_t{args.event_max}.csv", index=False)

    if not args.skip_foia_weights:
        weights = _foia_student_weights(
            con,
            matches,
            foia_path=Path(reg.base.FOIA_PATH),
            inst_cw_path=Path(reg.base.F1_INST_CW_PATH),
            ipeds_path=args.ipeds,
        )
        weights.to_csv(args.out_dir / "foia_student_weights_by_pair_event.csv", index=False)
        fast_student_summary = _student_weighted_summary_from_pair_panel(pair_panel, weights)
        fast_student_summary.to_csv(args.out_dir / "student_row_event_count_means_fast.csv", index=False)

    if not args.skip_did_panel:
        did_panel = reg.compute_generalized_did_panel(con, panel, degree_type=None, did_spec=reg.DEFAULT_DID_SPEC)
        student_summary = _student_weighted_summary(did_panel)
        student_summary.to_csv(args.out_dir / "student_row_event_count_means.csv", index=False)
        merged_weights = _merge_student_weights(pair_panel, did_panel)
        if not merged_weights.empty:
            merged_weights.to_csv(args.out_dir / "pair_event_ipeds_counts_with_student_weights.csv", index=False)

    print(f"Wrote diagnostics to {args.out_dir}")
    print(pair_summary[pair_summary["event_t"].isin([args.reference_event, args.event_max])].to_string(index=False))
    if not top.empty:
        print("\\nTop late-window contributors:")
        print(top.head(args.top_n * len(OUTCOMES) * 2).to_string(index=False))


if __name__ == "__main__":
    main()
