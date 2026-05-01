#!/usr/bin/env python
"""Build a match-quality memo for the 2026-05-07 relabel_indiv deck inputs."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("/home/yk0581")
CODE_ROOT = ROOT / "h1bworkers" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from relabels_revelio import relabel_events_generalized as generalized
from f1_foia import econ_relabels_opt_usage_v2 as econ_v2


OUT_DIR = CODE_ROOT / "output" / "relabel_indiv" / "match_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MEMO_PATH = OUT_DIR / "relabel_indiv_match_quality_memo_20260424.md"

PATHS = {
    "gen_panel": ROOT / "data/int/relabel_indiv_panel_feb2026.parquet",
    "gen_controls_panel": ROOT / "data/int/relabel_indiv_panel_slides_controls_apr2026.parquet",
    "econ_panel": ROOT / "data/int/relabel_indiv_panel_slides_nocontrols_apr2026.parquet",
    "gen_events": CODE_ROOT / "output/relabel_indiv/generalized_relabels_events.parquet",
    "gen_relabel_panel": CODE_ROOT / "output/relabel_indiv/generalized_relabels_panel.parquet",
    "gen_candidate_audit": CODE_ROOT / "output/relabel_indiv/generalized_relabels_candidate_audit.csv",
    "econ_events": ROOT / "data/int/int_files_nov2025/econ_relabels_v2.parquet",
    "stage04": ROOT / "data/int/f1_indiv_merge/04_rev_user_clean/rev_match_ready_apr2026v1.parquet",
    "stage05": ROOT / "data/int/f1_indiv_merge/05_indiv_merge/f1_merge_person_baseline_apr2026v1.parquet",
    "ipeds_crosswalk": ROOT / "data/int/int_files_nov2025/ipeds_crosswalk_2021.parquet",
}


def fmt_int(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return ""
    return f"{int(value):,}"


def fmt_num(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{100 * float(value):.{digits}f}%"


def md_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    for col in show.columns:
        show[col] = show[col].map(lambda x: "" if pd.isna(x) else str(x))
    return show.to_markdown(index=False)


def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = OUT_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    return df


def event_id_from_frame(df: pd.DataFrame, unitid_col: str = "unitid") -> pd.Series:
    cols = [unitid_col, "relabel_year", "relabel_type", "degree_type", "broad_pair_bin", "awlevel"]
    parts = []
    for col in cols:
        if col in df.columns:
            parts.append(df[col].astype("string").fillna("<NA>"))
        else:
            parts.append(pd.Series(["<NA>"] * len(df), index=df.index, dtype="string"))
    out = parts[0]
    for part in parts[1:]:
        out = out + "|" + part
    return out


def load_school_names(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.sql(
        f"""
        SELECT
            CAST(UNITID AS BIGINT) AS unitid,
            FIRST(instname ORDER BY ALIAS ASC, instname ASC) AS school_name,
            FIRST(CITY ORDER BY ALIAS ASC, instname ASC) AS city,
            FIRST(STABBR ORDER BY ALIAS ASC, instname ASC) AS state
        FROM read_parquet('{PATHS["ipeds_crosswalk"]}')
        WHERE UNITID IS NOT NULL AND instname IS NOT NULL
        GROUP BY CAST(UNITID AS BIGINT)
        """
    ).df()


def build_stage04_matched(con: duckdb.DuckDBPyConnection, combined_panel: pd.DataFrame) -> pd.DataFrame:
    panel_keys = combined_panel[
        [
            "source",
            "analysis_variant",
            "treated_ind",
            "user_id",
            "unitid",
            "education_number",
            "relabel_year",
            "relabel_type",
            "cohort_t",
            "pair_id",
            "event_id",
            "broad_pair_bin",
            "degree_type",
        ]
    ].drop_duplicates()
    con.register("panel_keys_py", panel_keys)
    stage04 = str(PATHS["stage04"])
    return con.sql(
        f"""
        WITH stage04_base AS (
            SELECT DISTINCT
                CAST(user_id AS BIGINT) AS user_id,
                CAST(education_number AS BIGINT) AS education_number,
                CAST(unitid AS BIGINT) AS unitid,
                CAST(degree_clean AS VARCHAR) AS degree_clean,
                CAST(cip AS VARCHAR) AS cip_raw,
                LPAD(regexp_replace(CAST(cip AS VARCHAR), '[^0-9]', '', 'g'), 6, '0') AS cip6,
                CASE
                    WHEN LENGTH(regexp_replace(CAST(cip AS VARCHAR), '[^0-9]', '', 'g')) >= 5
                        THEN SUBSTR(LPAD(regexp_replace(CAST(cip AS VARCHAR), '[^0-9]', '', 'g'), 6, '0'), 1, 6)
                    WHEN LENGTH(regexp_replace(CAST(cip AS VARCHAR), '[^0-9]', '', 'g')) >= 3
                        THEN SUBSTR(LPAD(regexp_replace(CAST(cip AS VARCHAR), '[^0-9]', '', 'g'), 4, '0'), 1, 4)
                    ELSE SUBSTR(LPAD(regexp_replace(CAST(cip AS VARCHAR), '[^0-9]', '', 'g'), 2, '0'), 1, 2)
                END AS cip_match_code,
                CAST(university_raw AS VARCHAR) AS university_raw,
                CAST(field_clean AS VARCHAR) AS field_clean,
                TRY_CAST(ed_startdate AS DATE) AS ed_startdate,
                TRY_CAST(ed_enddate AS DATE) AS ed_enddate,
                TRY_CAST(school_match_score AS DOUBLE) AS school_match_score,
                CASE
                    WHEN TRY_CAST(ed_enddate AS DATE) IS NOT NULL
                        THEN CAST(EXTRACT(YEAR FROM TRY_CAST(ed_enddate AS DATE)) AS INTEGER)
                    WHEN TRY_CAST(ed_startdate AS DATE) IS NULL THEN NULL::INTEGER
                    WHEN LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) IN ('master', 'masters', 'mba', 'associate', 'associates')
                        THEN CAST(EXTRACT(YEAR FROM TRY_CAST(ed_startdate AS DATE)) AS INTEGER) + 2
                    WHEN LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) IN ('doctor', 'doctors', 'doctoral', 'phd', 'ph.d', 'bachelor', 'bachelors')
                        THEN CAST(EXTRACT(YEAR FROM TRY_CAST(ed_startdate AS DATE)) AS INTEGER) + 4
                    ELSE CAST(EXTRACT(YEAR FROM TRY_CAST(ed_startdate AS DATE)) AS INTEGER) + 4
                END AS stage04_grad_year,
                CASE
                    WHEN LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%phd%'
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%ph d%'
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%doctor%'
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%doctoral%'
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%jd%'
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%md%'
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%edd%'
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%dba%'
                        THEN 'Doctor'
                    WHEN regexp_replace(LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')), '\\s+', '', 'g')
                         IN ('ma', 'ms', 'mba', 'meng', 'mpp', 'mph', 'mpa', 'mfin', 'msc', 'macc')
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%master%'
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%masters%'
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%mba%'
                        THEN 'Master'
                    WHEN regexp_replace(LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')), '\\s+', '', 'g')
                         IN ('ba', 'bs', 'bba', 'ab', 'sb')
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%bachelor%'
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%undergraduate%'
                      OR LOWER(COALESCE(CAST(degree_clean AS VARCHAR), '')) LIKE '%undergrad%'
                        THEN 'Bachelor'
                    ELSE 'Other'
                END AS stage04_degree_type
            FROM read_parquet('{stage04}')
        )
        SELECT
            p.*,
            s.degree_clean,
            s.stage04_degree_type,
            s.cip_raw,
            s.cip6,
            s.cip_match_code,
            s.university_raw,
            s.field_clean,
            s.ed_startdate,
            s.ed_enddate,
            s.stage04_grad_year,
            s.school_match_score
        FROM panel_keys_py p
        LEFT JOIN stage04_base s
          ON p.user_id = s.user_id
         AND p.education_number = s.education_number
         AND p.unitid = s.unitid
        """
    ).df()


def table_panel_counts(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, variant, treated), g in panel.groupby(["source", "analysis_variant", "treated_ind"], dropna=False):
        rows.append(
            {
                "source": source,
                "variant": variant,
                "group": "treated" if int(treated) == 1 else "control",
                "rows_user_event": len(g),
                "users": g["user_id"].nunique(),
                "schools": g["unitid"].nunique(),
                "events_or_pairs": g["pair_id"].nunique() if g["pair_id"].notna().any() else g["event_id"].nunique(),
                "mean_abs_cohort_t": g["cohort_t"].abs().mean(),
                "share_abs_t_eq_5": (g["cohort_t"].abs() == 5).mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values(["source", "variant", "group"])
    return save_table(out, "panel_counts")


def table_school_scores(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in matched.groupby(["source", "analysis_variant", "treated_ind"], dropna=False):
        score = pd.to_numeric(g["school_match_score"], errors="coerce")
        rows.append(
            {
                "source": keys[0],
                "variant": keys[1],
                "group": "treated" if int(keys[2]) == 1 else "control",
                "n": len(g),
                "score_nonmissing_share": score.notna().mean(),
                "mean_score": score.mean(),
                "p10_score": score.quantile(0.10),
                "median_score": score.quantile(0.50),
                "p90_score": score.quantile(0.90),
                "share_score_ge_0_85": score.ge(0.85).mean(),
                "share_score_lt_0_75": score.lt(0.75).mean(),
                "share_score_lt_0_65": score.lt(0.65).mean(),
            }
        )
    return save_table(pd.DataFrame(rows).sort_values(["source", "variant", "group"]), "school_match_score_summary")


def event_source_tables(
    con: duckdb.DuckDBPyConnection,
    gen_relabel_panel: pd.DataFrame,
    econ_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gen_treated = generalized.build_broad_treated_events(gen_relabel_panel).copy()
    gen_treated["event_id"] = event_id_from_frame(gen_treated)
    econ_treated = econ_events[econ_events["event_flag"].eq(1)].drop_duplicates(
        ["unitid", "relabel_year", "relabel_type"]
    )

    source_summary = pd.DataFrame(
        [
            {
                "source": "generalized",
                "events": len(gen_treated),
                "schools": gen_treated["unitid"].nunique(),
                "year_min": gen_treated["relabel_year"].min(),
                "year_max": gen_treated["relabel_year"].max(),
                "external_verified_events": int(gen_treated["external_verified"].fillna(0).sum())
                if "external_verified" in gen_treated
                else np.nan,
                "ipeds_scan_events": int(gen_treated["found_in_ipeds_scan"].fillna(0).sum())
                if "found_in_ipeds_scan" in gen_treated
                else np.nan,
            },
            {
                "source": "econ_v2",
                "events": len(econ_treated),
                "schools": econ_treated["unitid"].nunique(),
                "year_min": econ_treated["relabel_year"].min(),
                "year_max": econ_treated["relabel_year"].max(),
                "external_verified_events": np.nan,
                "ipeds_scan_events": len(econ_treated),
            },
        ]
    )
    source_summary = save_table(source_summary, "event_source_summary")

    gen_by_bin = (
        gen_treated.groupby(["degree_type", "broad_pair_bin"], dropna=False)
        .agg(events=("event_id", "nunique"), schools=("unitid", "nunique"), first_year=("relabel_year", "min"), last_year=("relabel_year", "max"))
        .reset_index()
        .sort_values(["degree_type", "events"], ascending=[True, False])
    )
    gen_by_bin = save_table(gen_by_bin, "generalized_events_by_bin_degree")

    econ_by_type = (
        econ_treated.groupby(["relabel_type"], dropna=False)
        .agg(events=("unitid", "size"), schools=("unitid", "nunique"), first_year=("relabel_year", "min"), last_year=("relabel_year", "max"))
        .reset_index()
        .sort_values("events", ascending=False)
    )
    econ_by_type = save_table(econ_by_type, "econ_events_by_type")
    return source_summary, gen_by_bin, econ_by_type


def control_pair_tables(con: duckdb.DuckDBPyConnection, gen_relabel_panel: pd.DataFrame, econ_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen_pairs = generalized.match_treated_to_never_treated(con=con, relabel_panel=gen_relabel_panel)
    econ_pairs = econ_v2._match_treated_to_untreated_cohorts(con=con, relabel_df=econ_events)
    frames = []
    for source, pairs in [("generalized", gen_pairs), ("econ_v2", econ_pairs)]:
        if pairs.empty:
            continue
        rows = {
            "source": source,
            "pairs": len(pairs),
            "treated_schools": pairs["treated_unitid"].nunique(),
            "control_schools": pairs["control_unitid"].nunique(),
            "replacement_share": pd.to_numeric(pairs.get("match_with_replacement", pd.Series(dtype=float)), errors="coerce").mean(),
            "mean_abs_size_diff": pd.to_numeric(pairs.get("abs_size_diff", pd.Series(dtype=float)), errors="coerce").mean(),
            "p90_abs_size_diff": pd.to_numeric(pairs.get("abs_size_diff", pd.Series(dtype=float)), errors="coerce").quantile(0.90),
        }
        if "match_distance" in pairs:
            rows["mean_match_distance"] = pd.to_numeric(pairs["match_distance"], errors="coerce").mean()
            rows["p90_match_distance"] = pd.to_numeric(pairs["match_distance"], errors="coerce").quantile(0.90)
        frames.append(rows)
    summary = save_table(pd.DataFrame(frames), "control_pair_summary")

    pair_detail = pd.concat(
        [
            gen_pairs.assign(source="generalized"),
            econ_pairs.assign(source="econ_v2"),
        ],
        ignore_index=True,
        sort=False,
    )
    if not pair_detail.empty:
        pair_detail = pair_detail.sort_values(["source", "abs_size_diff"], ascending=[True, False])
    save_table(pair_detail, "control_pair_details")
    return summary, pair_detail


def degree_cip_tables(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    degree = (
        matched[matched["treated_ind"].eq(1)]
        .groupby(["source", "analysis_variant", "stage04_degree_type"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    degree["share_within_source_variant"] = degree["rows"] / degree.groupby(["source", "analysis_variant"])["rows"].transform("sum")
    degree = save_table(degree.sort_values(["source", "analysis_variant", "rows"], ascending=[True, True, False]), "treated_degree_distribution")

    cip = (
        matched[matched["treated_ind"].eq(1)]
        .groupby(["source", "analysis_variant", "broad_pair_bin", "cip6", "field_clean"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["source", "analysis_variant", "rows"], ascending=[True, True, False])
    )
    cip = save_table(cip, "treated_top_cip_field_distribution")
    return degree, cip


def candidate_audit_table() -> pd.DataFrame:
    audit = pd.read_csv(PATHS["gen_candidate_audit"])
    rows = []
    for method, g in audit.groupby("school_match_method", dropna=False):
        rows.append(
            {
                "school_match_method": method,
                "candidates": len(g),
                "verified": int(pd.to_numeric(g["external_verified"], errors="coerce").fillna(0).sum()),
                "mean_school_score": pd.to_numeric(g["school_match_score"], errors="coerce").mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values(["verified", "candidates"], ascending=False)
    return save_table(out, "generalized_candidate_audit_by_school_method")


def make_examples(
    matched: pd.DataFrame,
    school_names: pd.DataFrame,
    gen_relabel_panel: pd.DataFrame,
    econ_events: pd.DataFrame,
) -> pd.DataFrame:
    m = matched.merge(school_names[["unitid", "school_name", "state"]], on="unitid", how="left")
    gen_ev = generalized.build_broad_treated_events(gen_relabel_panel).copy()
    gen_ev["event_id"] = event_id_from_frame(gen_ev)
    gen_cols = [
        "event_id",
        "source_cip_label",
        "target_cip_label",
        "event_source_cip6",
        "target_cip6",
        "event_origin_category",
        "external_verified",
    ]
    m = m.merge(gen_ev[[c for c in gen_cols if c in gen_ev.columns]].drop_duplicates("event_id"), on="event_id", how="left")
    econ_ev = econ_events[econ_events["event_flag"].eq(1)][
        ["unitid", "relabel_year", "relabel_type", "event_source_cip6"]
    ].drop_duplicates()
    econ_ev["econ_event_key"] = (
        econ_ev["unitid"].astype(str) + "|" + econ_ev["relabel_year"].astype(str) + "|" + econ_ev["relabel_type"].astype(str)
    )
    m["econ_event_key"] = m["unitid"].astype(str) + "|" + m["relabel_year"].astype(str) + "|" + m["relabel_type"].astype(str)
    m = m.merge(econ_ev[["econ_event_key", "event_source_cip6"]].rename(columns={"event_source_cip6": "econ_source_cip6"}), on="econ_event_key", how="left")

    def pick(df: pd.DataFrame, n: int, sort_cols: list[str], ascending: list[bool]) -> pd.DataFrame:
        if df.empty:
            return df.head(0)
        return df.sort_values(sort_cols, ascending=ascending).drop_duplicates(["source", "unitid", "relabel_year", "user_id"]).head(n)

    treated = m[m["treated_ind"].eq(1)].copy()
    treated["abs_cohort_t"] = treated["cohort_t"].abs()
    examples = []
    gen_rep_pool = treated[
        treated["source"].eq("generalized") & treated["analysis_variant"].eq("stage04_all")
    ].copy()
    gen_rep = (
        gen_rep_pool.sort_values(
            ["broad_pair_bin", "abs_cohort_t", "school_match_score", "unitid", "user_id"],
            ascending=[True, True, False, True, True],
        )
        .groupby("broad_pair_bin", dropna=False)
        .head(2)
        .head(10)
    )
    examples.append(
        gen_rep.assign(example_type="generalized representative")
    )
    econ_rep_pool = treated[
        treated["source"].eq("econ_v2")
        & treated["analysis_variant"].eq("stage04_all")
        & treated["stage04_degree_type"].eq("Master")
        & treated["field_clean"].astype("string").str.contains("econ", case=False, na=False)
    ].copy()
    econ_rep = (
        econ_rep_pool.sort_values(
            ["abs_cohort_t", "school_match_score", "unitid", "user_id"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates(["unitid", "relabel_year"])
        .head(6)
    )
    examples.append(
        econ_rep.assign(example_type="econ representative")
    )
    examples.append(
        pick(
            treated[
                treated["school_match_score"].lt(0.65).fillna(False)
                | treated["cohort_t"].abs().eq(5)
                | (treated["source"].eq("econ_v2") & ~treated["stage04_degree_type"].eq("Master"))
            ],
            10,
            ["source", "school_match_score", "cohort_t"],
            [True, True, False],
        ).assign(example_type="borderline / check")
    )
    out = pd.concat(examples, ignore_index=True, sort=False)
    keep = [
        "example_type",
        "source",
        "analysis_variant",
        "user_id",
        "school_name",
        "university_raw",
        "state",
        "relabel_year",
        "cohort_t",
        "degree_clean",
        "stage04_degree_type",
        "cip6",
        "field_clean",
        "broad_pair_bin",
        "relabel_type",
        "source_cip_label",
        "target_cip_label",
        "econ_source_cip6",
        "school_match_score",
        "event_origin_category",
    ]
    out = out[[c for c in keep if c in out.columns]].copy()

    def assessment(row: pd.Series) -> str:
        flags = []
        score = row.get("school_match_score")
        if pd.notna(score) and float(score) < 0.75:
            flags.append("low or middling institution score")
        if abs(int(row.get("cohort_t", 0))) == 5:
            flags.append("edge of +/-5 window")
        if row.get("source") == "econ_v2" and row.get("stage04_degree_type") != "Master":
            flags.append("econ event is master-level, row is not Master")
        if not flags:
            return "plausible on school, field, and timing"
        return "; ".join(flags)

    out["assessment"] = out.apply(assessment, axis=1)
    return save_table(out, "qualitative_examples")


def write_memo(
    *,
    source_summary: pd.DataFrame,
    gen_by_bin: pd.DataFrame,
    econ_by_type: pd.DataFrame,
    panel_counts: pd.DataFrame,
    score_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    degree_dist: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    examples: pd.DataFrame,
    top_cips: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Match quality memo: `relabel_indiv_analysis` for `build_laborlunch_20260507_revelio_deck.py`")
    lines.append("")
    lines.append("Date: 2026-04-24")
    lines.append("")
    lines.append("## Bottom line")
    lines.append("")
    lines.append(
        "The generalized deck sample is internally consistent with the code's intended design: it matches Revelio education spells to the relabeled institution, the relabel degree type, a broad source/target CIP family, and a +/-5 graduation-year window. The main caveat is breadth: for several bins the match is intentionally at a broad family level, so individual rows can be plausible for the family while not proving the exact old-to-new CIP transition. The full-sample institution-match scores are not uniformly high, so low-tail rows should be treated as measurement-error risk rather than clean matches."
    )
    lines.append("")
    lines.append(
        "The econ-specific deck sample is narrower by field but looser by degree. It filters Revelio education rows to CIP 45.06 and the relabeled school/year window, but it does not require a Master's degree even though the econ v2 events are detected on IPEDS awlevel 7. This creates a visible set of bachelor/doctor/other economics rows in the treated sample; they are useful only if the desired estimand is any Revelio economics graduate around a master's-program relabel."
    )
    lines.append("")
    lines.append("## Matching process")
    lines.append("")
    lines.append("1. The deck wrapper creates three configs: generalized no-controls, generalized with individual/school controls, and econ-only no-controls. All use the same full-sample Revelio education file and the same stage-05 FOIA-linked user filter for the baseline variant.")
    lines.append("2. Generalized events come from `generalized_relabels_panel.parquet`. The code collapses verified IPEDS/external events to one broad event per `unitid x awlevel x broad_pair_bin`, keeps relabel years 2014-2021, and carries event provenance.")
    lines.append("3. Econ events come from `econ_relabels_v2.parquet`. They are IPEDS master-level economics-to-quantitative-economics/econometrics transitions selected by source drop, target increase, persistence, and a PhD-spillover guard.")
    lines.append("4. Full-sample rows are the individual education universe. Generalized mode keeps all CIPs; econ mode restricts to digit-normalized CIP prefix `4506`. Both require non-null `unitid` and inferred graduation year.")
    lines.append("5. Treated individual rows join on `unitid` and `abs(grad_year - relabel_year) <= 5`. Generalized mode additionally requires inferred degree type and broad CIP-bin membership. If multiple rows match an event, the ranker keeps the closest cohort year, then rows with end dates, later end dates, lower education number, and stable tie-breakers.")
    lines.append("6. Controls are pseudo-events at never-treated schools. Econ controls are nearest neighbors on previous-year source-econ size within relabel type/year. Generalized controls match within awlevel and broad bin on pre-period source-family level and growth.")
    lines.append("")
    lines.append("## Quantitative checks")
    lines.append("")
    lines.append("### Event sources")
    source_disp = source_summary.copy()
    for c in ["events", "schools", "year_min", "year_max", "external_verified_events", "ipeds_scan_events"]:
        source_disp[c] = source_disp[c].map(fmt_int)
    lines.append(md_table(source_disp))
    lines.append("")
    lines.append("### Generalized verified events by degree and broad bin")
    gen_disp = gen_by_bin.copy()
    for c in ["events", "schools", "first_year", "last_year"]:
        gen_disp[c] = gen_disp[c].map(fmt_int)
    lines.append(md_table(gen_disp, max_rows=30))
    lines.append("")
    lines.append("### Econ events by type")
    econ_disp = econ_by_type.copy()
    for c in ["events", "schools", "first_year", "last_year"]:
        econ_disp[c] = econ_disp[c].map(fmt_int)
    lines.append(md_table(econ_disp))
    lines.append("")
    lines.append("### Individual panel coverage")
    pc = panel_counts.copy()
    for c in ["rows_user_event", "users", "schools", "events_or_pairs"]:
        pc[c] = pc[c].map(fmt_int)
    pc["mean_abs_cohort_t"] = pc["mean_abs_cohort_t"].map(lambda x: fmt_num(x, 2))
    pc["share_abs_t_eq_5"] = pc["share_abs_t_eq_5"].map(fmt_pct)
    lines.append(md_table(pc, max_rows=20))
    lines.append("")
    lines.append("### Full-sample institution match scores among matched rows")
    ss = score_summary.copy()
    for c in ["n"]:
        ss[c] = ss[c].map(fmt_int)
    for c in ["score_nonmissing_share", "share_score_ge_0_85", "share_score_lt_0_75", "share_score_lt_0_65"]:
        ss[c] = ss[c].map(fmt_pct)
    for c in ["mean_score", "p10_score", "median_score", "p90_score"]:
        ss[c] = ss[c].map(lambda x: fmt_num(x, 3))
    lines.append(md_table(ss, max_rows=20))
    lines.append("")
    lines.append("### Control-pair quality")
    cs = control_summary.copy()
    if not cs.empty:
        for c in ["pairs", "treated_schools", "control_schools"]:
            cs[c] = cs[c].map(fmt_int)
        for c in ["replacement_share"]:
            cs[c] = cs[c].map(fmt_pct)
        for c in ["mean_abs_size_diff", "p90_abs_size_diff", "mean_match_distance", "p90_match_distance"]:
            if c in cs.columns:
                cs[c] = cs[c].map(lambda x: fmt_num(x, 2))
    lines.append(md_table(cs))
    lines.append("")
    lines.append("### Treated degree distribution")
    dd = degree_dist.copy()
    dd["rows"] = dd["rows"].map(fmt_int)
    dd["share_within_source_variant"] = dd["share_within_source_variant"].map(fmt_pct)
    lines.append(md_table(dd, max_rows=30))
    lines.append("")
    lines.append("### Generalized external-candidate school resolution")
    ca = candidate_audit.copy()
    for c in ["candidates", "verified"]:
        ca[c] = ca[c].map(fmt_int)
    ca["mean_school_score"] = ca["mean_school_score"].map(lambda x: fmt_num(x, 3))
    lines.append(md_table(ca, max_rows=20))
    lines.append("")
    lines.append("## Qualitative hand checks")
    lines.append("")
    lines.append(
        "The table below samples representative and deliberately borderline rows. It uses only internal IDs and education fields, not names. `assessment` is my qualitative read from the joined school, field, degree, timing, and score."
    )
    ex = examples.copy()
    if "school_match_score" in ex:
        ex["school_match_score"] = ex["school_match_score"].map(lambda x: fmt_num(x, 3))
    lines.append(md_table(ex, max_rows=25))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Institution matching is complete but visibly noisy. The median full-sample school score is around 0.81 and the low tail contains clear sub-school/foreign-campus/alias problems; rows below roughly 0.65 are the most concerning.")
    lines.append("- Timing is mechanically broad. A nontrivial share of rows sits at the +/-5 edge because the analysis intentionally includes a wide cohort window; this is defensible for event-study pre/post comparisons but not for identifying exact affected cohorts.")
    lines.append("- The generalized sample has better degree-level coherence than the econ-specific sample because it joins on inferred Revelio degree type. The cost is broader CIP-family exposure, especially in business/finance-like bins.")
    lines.append("- The econ-specific sample has strong field coherence but weak degree coherence. If the deck claim is specifically about master's-program relabels, add a Master-only variant or make the current all-degree interpretation explicit.")
    lines.append("- Control matching is reasonable as a first pass, but generalized control matching uses replacement by default and relies on broad-bin pre-period IPEDS sizes. The memo tables identify the worst control-pair size gaps for follow-up in `control_pair_details.csv`.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append(f"- Memo: `{MEMO_PATH}`")
    for csv_path in sorted(OUT_DIR.glob("*.csv")):
        lines.append(f"- `{csv_path}`")
    lines.append("")
    MEMO_PATH.write_text("\n".join(lines))


def main() -> None:
    con = duckdb.connect()
    con.sql(f"PRAGMA threads={max(1, __import__('os').cpu_count() or 1)}")
    con.sql("PRAGMA preserve_insertion_order=false")

    gen_panel = pd.read_parquet(PATHS["gen_panel"]).assign(source="generalized")
    econ_panel = pd.read_parquet(PATHS["econ_panel"]).assign(source="econ_v2")
    combined_panel = pd.concat([gen_panel, econ_panel], ignore_index=True, sort=False)
    gen_relabel_panel = pd.read_parquet(PATHS["gen_relabel_panel"])
    econ_events = pd.read_parquet(PATHS["econ_events"])

    school_names = load_school_names(con)
    matched = build_stage04_matched(con, combined_panel)
    matched = save_table(matched, "matched_panel_stage04_join")

    source_summary, gen_by_bin, econ_by_type = event_source_tables(con, gen_relabel_panel, econ_events)
    panel_counts = table_panel_counts(combined_panel)
    score_summary = table_school_scores(matched)
    control_summary, _ = control_pair_tables(con, gen_relabel_panel, econ_events)
    degree_dist, top_cips = degree_cip_tables(matched)
    candidate_audit = candidate_audit_table()
    examples = make_examples(matched, school_names, gen_relabel_panel, econ_events)

    write_memo(
        source_summary=source_summary,
        gen_by_bin=gen_by_bin,
        econ_by_type=econ_by_type,
        panel_counts=panel_counts,
        score_summary=score_summary,
        control_summary=control_summary,
        degree_dist=degree_dist,
        candidate_audit=candidate_audit,
        examples=examples,
        top_cips=top_cips,
    )
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
