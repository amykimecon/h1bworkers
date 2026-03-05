"""
Improved relabel-event detector for economics -> econometrics transitions.

This script keeps the downstream FOIA/plot workflow from
`econ_relabels_opt_usage.py` but replaces `detect_econ_relabels` with a
program-level, persistence-aware detector.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb as ddb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    import f1_foia.econ_relabels_opt_usage as base
except ModuleNotFoundError:
    # Notebook/script fallback: find repo root containing f1_foia/.
    repo_candidates: list[Path] = []
    if "__file__" in globals():
        repo_candidates.append(Path(__file__).resolve().parents[1])
    cwd = Path.cwd().resolve()
    repo_candidates.extend([cwd, *cwd.parents])

    for candidate in repo_candidates:
        if (candidate / "f1_foia" / "econ_relabels_opt_usage.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)
            break

    import f1_foia.econ_relabels_opt_usage as base

# Keep baseline paths/config from original module.
IPEDS_PATH = base.IPEDS_PATH
RELABEL_YEAR_MIN = base.RELABEL_YEAR_MIN
RELABEL_YEAR_MAX = base.RELABEL_YEAR_MAX
RELABEL_SPECS = base.RELABEL_SPECS
PICK_FIRST_RELABEL = base.PICK_FIRST_RELABEL

# Output for the improved detector (keep original output untouched).
RELABEL_OUTPUT = Path(f"{base.INT_FOLDER}/econ_relabels_v2.parquet")

# Detection tuning.
MIN_SHARE_INTL = 0.2
MIN_SOURCE_BASELINE = base.MIN_PREV_TOTAL
MIN_SOURCE_DROP_ABS = 5.0
MIN_SOURCE_DROP_PCT = base.SINGLE_YEAR_DROP_PCT
MIN_TARGET_OFFSET_SHARE = 0.6
MAX_NET_LOSS_SHARE = 0.5
SOURCE_PERSISTENCE_DROP_PCT = 0.3
TARGET_PERSISTENCE_GAIN_SHARE = 0.3
LOOKBACK_YEARS = 3
LOOKAHEAD_YEARS = 2

# Event-time control series options: "physical_sciences", "never_treated_econ", "both"
EVENT_TIME_CONTROL_MODE = "both"


def _source_filter(spec: dict) -> tuple[str, str]:
    src_pred = base._cip_prefix_pred("cip6", spec["source_prefix"])
    exclude_clause = ""
    if spec.get("source_exclude_exact"):
        excluded = ", ".join([f"'{code}'" for code in spec["source_exclude_exact"]])
        exclude_clause = f" AND cip6 NOT IN ({excluded})"
    return src_pred, exclude_clause


def _target_filter(spec: dict) -> str:
    target_parts: list[str] = []
    for pref in spec.get("target_prefixes", []):
        target_parts.append(base._cip_prefix_pred("cip6", pref))
    for code in spec.get("target_exact", []):
        target_parts.append(base._cip_exact_pred("cip6", code))
    return " OR ".join(target_parts) if target_parts else "FALSE"


def _select_events(events: pd.DataFrame) -> pd.DataFrame:
    """Choose one source CIP per unit-year, then one event year per unit/type."""
    if events.empty:
        return events

    events = events.sort_values(
        ["unitid", "year", "relabel_score", "source_drop", "target_increase", "source_cip6"],
        ascending=[True, True, False, False, False, True],
    )
    events = events.drop_duplicates(subset=["unitid", "year", "relabel_type"], keep="first")

    if PICK_FIRST_RELABEL:
        events = events.sort_values(
            ["unitid", "relabel_type", "year", "relabel_score"],
            ascending=[True, True, True, False],
        )
        events = events.drop_duplicates(subset=["unitid", "relabel_type"], keep="first")
    else:
        events = events.sort_values(
            ["unitid", "relabel_type", "relabel_score", "source_drop", "year"],
            ascending=[True, True, False, False, True],
        )
        events = events.drop_duplicates(subset=["unitid", "relabel_type"], keep="first")

    return events.reset_index(drop=True)


def _build_event_panel(
    con: ddb.DuckDBPyConnection,
    events: pd.DataFrame,
    spec: dict,
    min_year: int,
    max_year: int,
) -> pd.DataFrame:
    src_pred, exclude_clause = _source_filter(spec)
    target_pred = _target_filter(spec)

    ipeds_ann = con.sql(
        f"""
        WITH masters AS (
            SELECT
                CAST(unitid AS BIGINT) AS unitid,
                CAST(year AS INTEGER) AS year,
                LPAD(CAST(cipcode AS VARCHAR), 6, '0') AS cip6,
                CAST(cnralt AS DOUBLE) AS cnralt,
                CAST(ctotalt AS DOUBLE) AS ctotalt
            FROM ipeds_raw
            WHERE unitid IS NOT NULL
              AND cipcode IS NOT NULL
              AND CAST(awlevel AS INTEGER) = 7
              AND CAST(share_intl AS DOUBLE) >= {MIN_SHARE_INTL}
              AND ({src_pred}{exclude_clause} OR ({target_pred}))
        )
        SELECT
            unitid,
            year,
            SUM(ctotalt) AS ctotalt,
            SUM(cnralt) AS cnralt,
            SUM(CASE WHEN {src_pred}{exclude_clause} THEN ctotalt ELSE 0 END) AS source_total,
            SUM(CASE WHEN ({target_pred}) THEN ctotalt ELSE 0 END) AS target_total
        FROM masters
        GROUP BY unitid, year
        """
    ).df()

    year_panel = pd.DataFrame({"year": list(range(min_year, max_year + 1))})
    event_cols = [
        "unitid",
        "year",
        "relabel_type",
        "source_cip6",
        "source_drop",
        "source_drop_pct",
        "target_increase",
        "target_increase_pct",
        "avg5_source_drop",
        "avg5_source_drop_pct",
        "avg5_target_increase",
        "avg5_target_increase_pct",
        "source_baseline",
        "target_baseline",
        "relabel_score",
    ]
    event_metrics = events[event_cols].copy()
    event_metrics = event_metrics.rename(columns={"year": "relabel_year", "source_cip6": "event_source_cip6"})

    panel = events[["unitid", "year", "relabel_type"]].copy()
    panel = panel.rename(columns={"year": "relabel_year"})
    panel["_tmp_key"] = 1
    year_panel["_tmp_key"] = 1
    panel = panel.merge(year_panel, on="_tmp_key", how="left").drop(columns=["_tmp_key"])

    panel = panel.merge(ipeds_ann, on=["unitid", "year"], how="left")
    for col in ["ctotalt", "cnralt", "source_total", "target_total"]:
        panel[col] = panel[col].fillna(0.0)

    panel = panel.sort_values(["unitid", "relabel_type", "relabel_year", "year"])
    grp = panel.groupby(["unitid", "relabel_type", "relabel_year"], sort=False)
    panel["source_total_prev"] = grp["source_total"].shift(1).fillna(0.0)
    panel["target_total_prev"] = grp["target_total"].shift(1).fillna(0.0)

    panel["source_total_intl"] = panel["source_total"]
    panel["source_total_intl_prev"] = panel["source_total_prev"]
    panel["target_total_intl"] = panel["target_total"]
    panel["target_total_intl_prev"] = panel["target_total_prev"]

    panel = panel.merge(event_metrics, on=["unitid", "relabel_type", "relabel_year"], how="left")
    panel["event_flag"] = (panel["year"] == panel["relabel_year"]).astype(int)
    panel["relabel_flag"] = panel["event_flag"]
    return panel


def detect_econ_relabels(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Identify relabel events using source-CIP-level drops and persistence checks.

    Relative to the original detector, this version:
    1) tracks source CIPs separately (unitid x source_cip x year),
    2) requires a minimum absolute drop,
    3) requires target replacement and post-event persistence.
    """
    if not os.path.exists(IPEDS_PATH):
        raise FileNotFoundError(f"Missing IPEDS completions parquet at {IPEDS_PATH}")

    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw AS SELECT * FROM read_parquet('{IPEDS_PATH}')")

    min_year = 2004
    max_year = 2024
    all_panels: list[pd.DataFrame] = []

    for spec in RELABEL_SPECS:
        src_pred, exclude_clause = _source_filter(spec)
        target_pred = _target_filter(spec)

        events = con.sql(
            f"""
            WITH masters AS (
                SELECT
                    CAST(unitid AS BIGINT) AS unitid,
                    CAST(year AS INTEGER) AS year,
                    LPAD(CAST(cipcode AS VARCHAR), 6, '0') AS cip6,
                    CAST(ctotalt AS DOUBLE) AS ctotalt
                FROM ipeds_raw
                WHERE unitid IS NOT NULL
                  AND cipcode IS NOT NULL
                  AND CAST(awlevel AS INTEGER) = 7
                  AND CAST(share_intl AS DOUBLE) >= {MIN_SHARE_INTL}
            ),
            source_year AS (
                SELECT
                    unitid,
                    year,
                    cip6 AS source_cip6,
                    SUM(ctotalt) AS source_total
                FROM masters
                WHERE {src_pred}{exclude_clause}
                GROUP BY unitid, year, source_cip6
            ),
            source_units AS (
                SELECT DISTINCT unitid, source_cip6 FROM source_year
            ),
            years AS (
                SELECT * FROM generate_series({min_year}, {max_year}) AS y(year)
            ),
            source_panel AS (
                SELECT
                    su.unitid,
                    su.source_cip6,
                    y.year,
                    COALESCE(sy.source_total, 0) AS source_total
                FROM source_units su
                CROSS JOIN years y
                LEFT JOIN source_year sy
                  ON sy.unitid = su.unitid
                 AND sy.source_cip6 = su.source_cip6
                 AND sy.year = y.year
            ),
            source_stats AS (
                SELECT
                    unitid,
                    source_cip6,
                    year,
                    source_total,
                    LAG(source_total) OVER (
                        PARTITION BY unitid, source_cip6
                        ORDER BY year
                    ) AS source_prev,
                    AVG(source_total) OVER (
                        PARTITION BY unitid, source_cip6
                        ORDER BY year
                        ROWS BETWEEN {LOOKBACK_YEARS} PRECEDING AND 1 PRECEDING
                    ) AS source_prev_window_avg,
                    AVG(source_total) OVER (
                        PARTITION BY unitid, source_cip6
                        ORDER BY year
                        ROWS BETWEEN 1 FOLLOWING AND {LOOKAHEAD_YEARS} FOLLOWING
                    ) AS source_post_window_avg
                FROM source_panel
            ),
            target_year AS (
                SELECT
                    unitid,
                    year,
                    SUM(ctotalt) AS target_total
                FROM masters
                WHERE ({target_pred})
                GROUP BY unitid, year
            ),
            source_unit_year AS (
                SELECT DISTINCT unitid, year FROM source_panel
            ),
            target_panel AS (
                SELECT
                    suy.unitid,
                    suy.year,
                    COALESCE(ty.target_total, 0) AS target_total
                FROM source_unit_year suy
                LEFT JOIN target_year ty
                  ON ty.unitid = suy.unitid
                 AND ty.year = suy.year
            ),
            target_stats AS (
                SELECT
                    unitid,
                    year,
                    target_total,
                    LAG(target_total) OVER (
                        PARTITION BY unitid
                        ORDER BY year
                    ) AS target_prev,
                    AVG(target_total) OVER (
                        PARTITION BY unitid
                        ORDER BY year
                        ROWS BETWEEN {LOOKBACK_YEARS} PRECEDING AND 1 PRECEDING
                    ) AS target_prev_window_avg,
                    AVG(target_total) OVER (
                        PARTITION BY unitid
                        ORDER BY year
                        ROWS BETWEEN 1 FOLLOWING AND {LOOKAHEAD_YEARS} FOLLOWING
                    ) AS target_post_window_avg
                FROM target_panel
            ),
            candidates AS (
                SELECT
                    s.unitid,
                    s.year,
                    s.source_cip6,
                    s.source_total,
                    s.source_prev,
                    t.target_total,
                    t.target_prev,
                    GREATEST(COALESCE(s.source_prev, 0) - COALESCE(s.source_total, 0), 0) AS source_drop,
                    CASE
                        WHEN COALESCE(s.source_prev, 0) > 0
                        THEN (s.source_prev - s.source_total) / s.source_prev
                        ELSE NULL
                    END AS source_drop_pct,
                    COALESCE(t.target_total, 0) - COALESCE(t.target_prev, 0) AS target_increase,
                    CASE
                        WHEN COALESCE(t.target_prev, 0) > 0
                        THEN (COALESCE(t.target_total, 0) - COALESCE(t.target_prev, 0)) / t.target_prev
                        ELSE NULL
                    END AS target_increase_pct,
                    (COALESCE(s.source_total, 0) + COALESCE(t.target_total, 0))
                        - (COALESCE(s.source_prev, 0) + COALESCE(t.target_prev, 0)) AS net_change,
                    COALESCE(s.source_prev_window_avg, s.source_prev, 0) AS source_baseline,
                    COALESCE(s.source_post_window_avg, s.source_total, 0) AS source_post,
                    COALESCE(t.target_prev_window_avg, t.target_prev, 0) AS target_baseline,
                    COALESCE(t.target_post_window_avg, t.target_total, 0) AS target_post
                FROM source_stats s
                JOIN target_stats t
                  ON t.unitid = s.unitid
                 AND t.year = s.year
                WHERE s.year BETWEEN {RELABEL_YEAR_MIN} AND {RELABEL_YEAR_MAX}
            ),
            flagged AS (
                SELECT
                    *,
                    source_post - source_baseline AS avg5_source_drop,
                    CASE
                        WHEN source_baseline > 0
                        THEN (source_post - source_baseline) / source_baseline
                        ELSE NULL
                    END AS avg5_source_drop_pct,
                    target_post - target_baseline AS avg5_target_increase,
                    CASE
                        WHEN target_baseline > 0
                        THEN (target_post - target_baseline) / target_baseline
                        ELSE NULL
                    END AS avg5_target_increase_pct,
                    (
                        COALESCE(CASE WHEN source_drop > 0 THEN target_increase / source_drop END, 0)
                        + 0.5 * COALESCE(CASE WHEN source_drop > 0 THEN (source_baseline - source_post) / source_drop END, 0)
                        + 0.5 * COALESCE(CASE WHEN source_drop > 0 THEN (target_post - target_baseline) / source_drop END, 0)
                        - 0.5 * COALESCE(CASE WHEN source_drop > 0 THEN (-LEAST(net_change, 0)) / source_drop END, 0)
                    ) AS relabel_score,
                    '{spec["name"]}' AS relabel_type
                FROM candidates
                WHERE source_baseline >= {MIN_SOURCE_BASELINE}
                  AND source_drop >= {MIN_SOURCE_DROP_ABS}
                  AND source_drop_pct >= {MIN_SOURCE_DROP_PCT}
                  AND target_increase >= {MIN_TARGET_OFFSET_SHARE} * source_drop
                  AND net_change >= -{MAX_NET_LOSS_SHARE} * source_drop
                  AND source_post <= source_baseline * (1 - {SOURCE_PERSISTENCE_DROP_PCT})
                  AND target_post >= target_baseline + {TARGET_PERSISTENCE_GAIN_SHARE} * source_drop
            )
            SELECT
                unitid,
                year,
                source_cip6,
                source_total,
                source_prev AS source_total_prev,
                source_total AS source_total_intl,
                source_prev AS source_total_intl_prev,
                target_total,
                target_prev AS target_total_prev,
                target_total AS target_total_intl,
                target_prev AS target_total_intl_prev,
                source_drop,
                source_drop_pct,
                target_increase,
                target_increase_pct,
                source_baseline AS source_prev5_avg,
                source_baseline,
                target_baseline,
                avg5_source_drop,
                avg5_source_drop_pct,
                avg5_target_increase,
                avg5_target_increase_pct,
                relabel_score,
                relabel_type
            FROM flagged
            """
        ).df()

        events = _select_events(events)
        if events.empty:
            continue

        panel = _build_event_panel(con=con, events=events, spec=spec, min_year=min_year, max_year=max_year)
        all_panels.append(panel)

    if not all_panels:
        return pd.DataFrame()

    out = pd.concat(all_panels, ignore_index=True)
    out = out.sort_values(["relabel_type", "unitid", "relabel_year", "year"]).reset_index(drop=True)
    return out


def _source_only_cip_where(col: str = "cipcode") -> str:
    """Source-side CIP predicate for relabel specs (exclude target codes)."""
    predicates: list[str] = []
    for spec in RELABEL_SPECS:
        src_pred = base._cip_prefix_pred(col, spec["source_prefix"])
        excludes = spec.get("source_exclude_exact", [])
        if excludes:
            excl = ", ".join([f"'{c}'" for c in excludes])
            predicates.append(
                f"({src_pred} AND LPAD(CAST({col} AS VARCHAR), 6, '0') NOT IN ({excl}))"
            )
        else:
            predicates.append(f"({src_pred})")
    return " OR ".join(predicates) if predicates else "FALSE"


def _match_treated_to_untreated_cohorts(
    con: ddb.DuckDBPyConnection,
    relabel_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build 1:1 treated-to-never-treated institution matches by relabel year and pre-size.

    Pre-size is source economics total in t-1.
    Matching is nearest-neighbor on absolute pre-size difference, year-by-year.
    """
    if relabel_df.empty:
        return pd.DataFrame()
    if not os.path.exists(IPEDS_PATH):
        raise FileNotFoundError(f"Missing IPEDS completions parquet at {IPEDS_PATH}")

    relabel_year_col = "relabel_year" if "relabel_year" in relabel_df.columns else "year"
    treated_events = (
        relabel_df.loc[relabel_df["event_flag"] == 1, ["unitid", relabel_year_col, "relabel_type", "source_total_prev"]]
        .dropna(subset=["unitid", relabel_year_col, "relabel_type"])
        .drop_duplicates()
        .rename(
            columns={
                relabel_year_col: "relabel_year",
                "source_total_prev": "treated_pre_size",
            }
        )
    )
    if treated_events.empty:
        return pd.DataFrame()

    treated_events["unitid"] = pd.to_numeric(treated_events["unitid"], errors="coerce").astype("Int64")
    treated_events["relabel_year"] = pd.to_numeric(treated_events["relabel_year"], errors="coerce").astype("Int64")
    treated_events["treated_pre_size"] = pd.to_numeric(
        treated_events["treated_pre_size"], errors="coerce"
    ).fillna(0.0)
    treated_events = treated_events.dropna(subset=["unitid", "relabel_year"]).copy()
    treated_events["unitid"] = treated_events["unitid"].astype("int64")
    treated_events["relabel_year"] = treated_events["relabel_year"].astype("int64")

    treated_institutions = set(treated_events["unitid"].tolist())
    spec_by_name = {spec["name"]: spec for spec in RELABEL_SPECS}

    candidate_frames: list[pd.DataFrame] = []
    relabel_types = sorted(treated_events["relabel_type"].dropna().unique().tolist())
    for relabel_type in relabel_types:
        spec = spec_by_name.get(relabel_type)
        if spec is None:
            continue
        src_pred, exclude_clause = _source_filter(spec)
        cand = con.sql(
            f"""
            WITH masters AS (
                SELECT
                    CAST(unitid AS BIGINT) AS unitid,
                    CAST(year AS INTEGER) AS year,
                    LPAD(CAST(cipcode AS VARCHAR), 6, '0') AS cip6,
                    CAST(ctotalt AS DOUBLE) AS ctotalt
                FROM read_parquet('{IPEDS_PATH}')
                WHERE unitid IS NOT NULL
                  AND cipcode IS NOT NULL
                  AND CAST(awlevel AS INTEGER) = 7
                  AND CAST(share_intl AS DOUBLE) >= {MIN_SHARE_INTL}
            ),
            source_year AS (
                SELECT
                    unitid,
                    year,
                    SUM(ctotalt) AS source_total
                FROM masters
                WHERE {src_pred}{exclude_clause}
                GROUP BY unitid, year
            )
            SELECT
                unitid,
                year + 1 AS relabel_year,
                source_total AS control_pre_size
            FROM source_year
            WHERE year + 1 BETWEEN {RELABEL_YEAR_MIN} AND {RELABEL_YEAR_MAX}
            """
        ).df()
        if cand.empty:
            continue
        cand["unitid"] = pd.to_numeric(cand["unitid"], errors="coerce").astype("Int64")
        cand["relabel_year"] = pd.to_numeric(cand["relabel_year"], errors="coerce").astype("Int64")
        cand["control_pre_size"] = pd.to_numeric(cand["control_pre_size"], errors="coerce").fillna(0.0)
        cand = cand.dropna(subset=["unitid", "relabel_year"]).copy()
        cand["unitid"] = cand["unitid"].astype("int64")
        cand["relabel_year"] = cand["relabel_year"].astype("int64")
        cand = cand[~cand["unitid"].isin(treated_institutions)].copy()
        if cand.empty:
            continue
        cand["relabel_type"] = relabel_type
        candidate_frames.append(cand)

    if not candidate_frames:
        return pd.DataFrame()
    candidates = pd.concat(candidate_frames, ignore_index=True)

    matches: list[dict[str, object]] = []
    pair_id = 0
    for (rtype, year), treated_group in treated_events.groupby(["relabel_type", "relabel_year"], sort=True):
        cand_group = candidates[
            (candidates["relabel_type"] == rtype) & (candidates["relabel_year"] == year)
        ].copy()
        if cand_group.empty:
            continue
        cand_group = cand_group.sort_values(["control_pre_size", "unitid"]).reset_index(drop=True)
        available = cand_group.copy()
        treated_group = treated_group.sort_values(["treated_pre_size", "unitid"]).reset_index(drop=True)

        for row in treated_group.itertuples(index=False):
            use_replacement = False
            if available.empty:
                use_pool = cand_group
                use_replacement = True
            else:
                use_pool = available
            if use_pool.empty:
                continue
            best_idx = (use_pool["control_pre_size"] - float(row.treated_pre_size)).abs().idxmin()
            chosen = use_pool.loc[best_idx]
            pair_id += 1
            matches.append(
                {
                    "pair_id": pair_id,
                    "relabel_type": rtype,
                    "relabel_year": int(year),
                    "treated_unitid": int(row.unitid),
                    "treated_pre_size": float(row.treated_pre_size),
                    "control_unitid": int(chosen["unitid"]),
                    "control_pre_size": float(chosen["control_pre_size"]),
                    "abs_size_diff": float(abs(chosen["control_pre_size"] - float(row.treated_pre_size))),
                    "match_with_replacement": int(use_replacement),
                }
            )
            if not use_replacement:
                available = available.drop(index=best_idx)

    if not matches:
        return pd.DataFrame()
    return pd.DataFrame(matches)


def compute_never_treated_econ_control_event_time(
    con: ddb.DuckDBPyConnection,
    relabel_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute event-time controls from never-treated institutions matched to treated cohorts.

    Matching is by relabel year and pre-treatment source-econ size (t-1), 1:1 nearest.
    """
    if relabel_df.empty:
        raise ValueError("No relabel events found; cannot compute never-treated control.")

    if not os.path.exists(base.FOIA_PATH):
        raise FileNotFoundError(f"Missing FOIA parquet at {base.FOIA_PATH}")
    if not os.path.exists(base.F1_INST_CW_PATH):
        raise FileNotFoundError(f"Missing F-1 institution crosswalk at {base.F1_INST_CW_PATH}")

    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{base.FOIA_PATH}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_cw AS SELECT * FROM read_parquet('{base.F1_INST_CW_PATH}')")

    matched_pairs = _match_treated_to_untreated_cohorts(con=con, relabel_df=relabel_df)
    if matched_pairs.empty:
        return pd.DataFrame()
    con.register("matched_pairs_py", matched_pairs)
    con.sql("CREATE OR REPLACE TEMP VIEW matched_pairs AS SELECT * FROM matched_pairs_py")

    foia_cols = [row[0] for row in con.sql("DESCRIBE foia_raw").fetchall()]
    cw_cols = [row[0] for row in con.sql("DESCRIBE f1_inst_cw").fetchall()]

    foia_inst_col = base.first_present(foia_cols, base.FOIA_INST_COLS, "FOIA institution column")
    foia_cip_col = base.first_present(foia_cols, base.FOIA_CIP_COLS, "FOIA CIP column")
    foia_end_col = base.first_present(foia_cols, base.FOIA_PROG_END_COLS, "program end date column")
    foia_student_col = base.first_present(foia_cols, base.FOIA_STUDENT_KEY_COLS, "student identifier column")
    foia_tuition_col = base.first_present(foia_cols, base.FOIA_TUITION_COLS, "tuition column")
    foia_edu_col = base.first_present(foia_cols, base.FOIA_EDU_LEVEL_COLS, "education level column")
    status_col = base.first_present(foia_cols, base.FOIA_STATUS_COLS, "requested status column")
    cw_inst_col = base.first_present(cw_cols, base.CW_INST_COLS, "crosswalk institution column")
    cw_unitid_col = base.first_present(cw_cols, base.CW_UNITID_COLS, "crosswalk unitid column")

    foia_year_col = None
    for cand in base.FOIA_YEAR_COLS:
        if cand.lower() in [c.lower() for c in foia_cols]:
            foia_year_col = next(c for c in foia_cols if c.lower() == cand.lower())
            break
    opt_end_col = None
    for cand in base.FOIA_OPT_END_COLS:
        if cand.lower() in [c.lower() for c in foia_cols]:
            opt_end_col = next(c for c in foia_cols if c.lower() == cand.lower())
            break
    if opt_end_col is None:
        raise ValueError("Could not locate an OPT authorization end column in FOIA data.")

    source_cip_where = _source_only_cip_where("cipcode")
    norm_cip_expr = base.normalize_cip_sql(foia_cip_col)
    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )

    control_calendar = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                {norm_cip_expr} AS cipcode,
                LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0') AS cip6,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                employer_name,
                employment_opt_type,
                {opt_end_col} AS opt_end_date,
                {foia_tuition_col} AS tuition,
                {foia_end_col} AS program_end_date,
                {status_col} AS requested_status
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE {foia_end_col} IS NOT NULL
              AND fr.{foia_edu_col} = 'MASTER''S'
              AND ({source_cip_where})
              {year_match_clause}
        ),
        relevant_foia AS (
            SELECT *
            FROM foia_base
            WHERE unitid IS NOT NULL
              AND cipcode IS NOT NULL
              AND grad_year IS NOT NULL
        ),
        matched_control AS (
            SELECT f.*
                 , mp.pair_id
                 , mp.relabel_type
                 , mp.relabel_year
                 , mp.treated_unitid
                 , mp.control_unitid
                 , mp.treated_pre_size
                 , mp.control_pre_size
                 , mp.abs_size_diff
                 , mp.match_with_replacement
            FROM relevant_foia f
            JOIN matched_pairs mp
              ON CAST(f.unitid AS BIGINT) = CAST(mp.control_unitid AS BIGINT)
        ),
        student_level AS (
            SELECT
                pair_id,
                unitid,
                cipcode,
                grad_year,
                MAX(CASE WHEN employer_name IS NOT NULL THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'POST-COMPLETION' THEN 1 ELSE 0 END) AS opt_ind_old,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL
                    THEN DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25
                    ELSE 0
                END AS opt_years,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                student_id,
                relabel_year,
                relabel_type
            FROM matched_control
            GROUP BY pair_id, unitid, cipcode, grad_year, student_id, relabel_year, relabel_type
        ),
        calendar_level AS (
            SELECT
                pair_id,
                grad_year AS calendar_year,
                relabel_year,
                relabel_type,
                AVG(avg_tuition) AS avg_tuition,
                COUNT(DISTINCT student_id) AS total_grads,
                COUNT(DISTINCT CASE WHEN opt_ind = 1 THEN student_id END) AS opt_users,
                COUNT(DISTINCT CASE WHEN opt_stem_ind = 1 THEN student_id END) AS opt_stem_users,
                COUNT(DISTINCT CASE WHEN status_change_ind = 1 THEN student_id END) AS status_change_users,
                SUM(opt_years) AS total_opt_years
            FROM student_level
            WHERE grad_year IS NOT NULL
            GROUP BY pair_id, calendar_year, relabel_year, relabel_type
        )
        SELECT
            calendar_year,
            relabel_year,
            relabel_type,
            avg_tuition,
            total_grads,
            opt_users,
            opt_stem_users,
            status_change_users,
            total_opt_years,
            NULL::DOUBLE AS ctotalt,
            NULL::DOUBLE AS cnralt
        FROM calendar_level
        """
    ).df()

    if control_calendar.empty:
        return pd.DataFrame()

    control_calendar["opt_share"] = control_calendar["opt_users"] / control_calendar["total_grads"]
    control_calendar["opt_stem_share"] = control_calendar["opt_stem_users"] / control_calendar["total_grads"]
    control_calendar["status_change_share"] = control_calendar["status_change_users"] / control_calendar["total_grads"]
    control_calendar["opt_years_avg"] = control_calendar["total_opt_years"] / control_calendar["total_grads"]
    control_calendar["f1_share_of_ctotalt"] = control_calendar["total_grads"] / control_calendar["ctotalt"]
    control_calendar["f1_share_of_cnralt"] = control_calendar["total_grads"] / control_calendar["cnralt"]
    control_calendar["tuition_total"] = control_calendar["avg_tuition"] * control_calendar["total_grads"]

    control_calendar["event_t"] = control_calendar["calendar_year"] - control_calendar["relabel_year"]
    control_event = (
        control_calendar.groupby(["event_t", "relabel_type"], as_index=False)
        .agg(
            total_grads=("total_grads", "sum"),
            opt_users=("opt_users", "sum"),
            opt_stem_users=("opt_stem_users", "sum"),
            status_change_users=("status_change_users", "sum"),
            total_opt_years=("total_opt_years", "sum"),
            tuition_total=("tuition_total", "sum"),
            ctotalt=("ctotalt", "sum"),
            cnralt=("cnralt", "sum"),
        )
    )
    # Institution-level never-treated controls are not matched to treated IPEDS denominators.
    control_event["ctotalt"] = pd.NA
    control_event["cnralt"] = pd.NA
    control_event["opt_share"] = control_event["opt_users"] / control_event["total_grads"]
    control_event["opt_stem_share"] = control_event["opt_stem_users"] / control_event["total_grads"]
    control_event["status_change_share"] = control_event["status_change_users"] / control_event["total_grads"]
    control_event["opt_years_avg"] = control_event["total_opt_years"] / control_event["total_grads"]
    control_event["f1_share_of_ctotalt"] = control_event["total_grads"] / control_event["ctotalt"]
    control_event["f1_share_of_cnralt"] = control_event["total_grads"] / control_event["cnralt"]
    control_event["avg_tuition"] = control_event["tuition_total"] / control_event["total_grads"]
    return control_event


def plot_opt_usage_event_time_with_control_label(
    opt_usage_event: pd.DataFrame,
    control_event: pd.DataFrame,
    control_label: str,
    yvar: str = "opt_share",
    show: bool = False,
    save: bool = False,
    file_tag: str = "control",
    make_treated_only_plot: bool = False,
) -> Path | None:
    """Plot treated-vs-control event-time series with configurable control label."""
    base.FIG_DIR.mkdir(parents=True, exist_ok=True)
    if opt_usage_event.empty:
        raise ValueError("No event-time OPT usage data to plot.")
    if control_event is None or control_event.empty:
        raise ValueError(f"No control event-time data for '{control_label}'.")

    treated = opt_usage_event.copy()
    treated["series_label"] = "Economics"
    ctrl = control_event.copy()
    ctrl["series_label"] = control_label
    plot_df = pd.concat([treated, ctrl], ignore_index=True)

    sns.set(style="whitegrid")
    sns.set_palette(base.PALETTE_SEQ)
    plt.rcParams.update({"font.size": base.BASE_FONT_SIZE})

    if make_treated_only_plot:
        fig_t, ax_t = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=treated[treated["event_t"].between(-5, 4)],
            x="event_t",
            y=yvar,
            hue="series_label",
            marker="o",
            ax=ax_t,
        )
        ax_t.set_ylabel(base.yvar_label(yvar))
        ax_t.set_xlabel("Years relative to relabel (t=0)")
        ax_t.set_title("")
        ax_t.axvline(x=0, linestyle="--", color="gray", linewidth=1)
        ax_t.legend(title=None)
        fig_t.tight_layout()
        if save:
            fig_t.savefig(base.FIG_DIR / f"{yvar}_event_time_treated.png", dpi=300)
        if show:
            plt.show()
        plt.close(fig_t)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=plot_df[plot_df["event_t"].between(-5, 4)],
        x="event_t",
        y=yvar,
        hue="series_label",
        marker="o",
        ax=ax,
    )
    ax.set_ylabel(base.yvar_label(yvar))
    ax.set_xlabel("Years relative to relabel (t=0)")
    ax.set_title("")
    ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
    ax.legend(title=None)
    fig.tight_layout()

    out_path = base.FIG_DIR / f"{yvar}_event_time_treated_control_{file_tag}.png"
    if save:
        fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_path if save else None


# Reuse original downstream functions unchanged.
compute_opt_usage = base.compute_opt_usage
compute_opt_usage_event_time = base.compute_opt_usage_event_time
compute_control_opt_usage_event_time = base.compute_control_opt_usage_event_time
plot_opt_usage = base.plot_opt_usage
plot_opt_usage_event_time = base.plot_opt_usage_event_time


def main() -> None:
    con = ddb.connect()

    relabel_df = detect_econ_relabels(con)
    if relabel_df.empty:
        print("No relabel events found with v2 detector.")
        return

    RELABEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    relabel_df.to_parquet(RELABEL_OUTPUT, index=False)
    print(f"Wrote improved relabel panel to {RELABEL_OUTPUT}")

    yvar = "opt_share"
    # YVAR_LABELS = {
    # "opt_share": "Share of F-1s using OPT",
    # "opt_stem_share": "Share of F-1s using STEM OPT",
    # "status_change_share": "Share with status change",
    # "opt_years_avg": "Average OPT years",
    # "f1_share_of_ctotalt": "FOIA share of IPEDS ctotalt",
    # "f1_share_of_cnralt": "FOIA share of IPEDS cnralt",
    # "avg_tuition": "Average tuition (USD)",
    # }

    plt.rcParams.update({"font.size": base.BASE_FONT_SIZE})
    sns.set(style="whitegrid")
    sns.set_palette(base.PALETTE_SEQ)
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=relabel_df[relabel_df["event_flag"] == 1],
        x="relabel_year",
        bins=range(RELABEL_YEAR_MIN, RELABEL_YEAR_MAX + 2),
        discrete=True,
        hue="relabel_type",
        multiple="dodge",
    )
    plt.xlabel("Relabel year")
    plt.ylabel("Count of relabel events")
    plt.title("")
    plt.tight_layout()
    plt.show()

    opt_usage = compute_opt_usage(con, relabel_df)
    fig_path = plot_opt_usage(opt_usage, yvar=yvar, show=True, save=True)
    print(f"Saved OPT usage plot to {fig_path}")

    opt_usage_event = compute_opt_usage_event_time(opt_usage)
    mode = str(EVENT_TIME_CONTROL_MODE).strip().lower()
    valid_modes = {"physical_sciences", "never_treated_econ", "both"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid EVENT_TIME_CONTROL_MODE='{EVENT_TIME_CONTROL_MODE}'. Use one of {sorted(valid_modes)}.")

    controls_to_plot: list[tuple[str, pd.DataFrame, str]] = []
    if mode in {"physical_sciences", "both"}:
        control_phys = compute_control_opt_usage_event_time(con, relabel_df)
        controls_to_plot.append(("Physical Sciences", control_phys, "physical_sciences"))
    if mode in {"never_treated_econ", "both"}:
        control_never = compute_never_treated_econ_control_event_time(con, relabel_df)
        controls_to_plot.append(("Never-treated Economics", control_never, "never_treated_econ"))

    made_treated_only = False
    for label, control_event, tag in controls_to_plot:
        out = plot_opt_usage_event_time_with_control_label(
            opt_usage_event=opt_usage_event,
            control_event=control_event,
            control_label=label,
            yvar=yvar,
            show=True,
            save=True,
            file_tag=tag,
            make_treated_only_plot=not made_treated_only,
        )
        made_treated_only = True
        if out is not None:
            print(f"Saved event-time treated-control plot ({label}) to {out}")


if __name__ == "__main__":
    main()
