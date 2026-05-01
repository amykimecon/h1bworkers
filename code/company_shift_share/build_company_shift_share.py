"""
Company-level shift-share pipeline.

Builds z_ct = Σ_k s_ckr * g_kt for companies (rcids), treatment counts of
Master's OPT hires, and outcomes based on Revelio headcounts. Inputs live
under root/data; outputs are written to root/data/out/company_shift_share.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Iterable, Optional

import duckdb as ddb
import pandas as pd
import seaborn as sns
# Ensure progress logs flush immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
    from company_shift_share.institution_mapping import load_revelio_school_map, sql_normalize_school_key
except ModuleNotFoundError:
    # Allow direct execution when repo root is not already on PYTHONPATH.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
    from company_shift_share.institution_mapping import load_revelio_school_map, sql_normalize_school_key

@dataclass(frozen=True)
class PipelinePaths:
    transitions: Path
    headcount: Path
    revelio_ipeds_foia_inst_crosswalk: Path
    revelio_inst_deterministic_map: Path | None
    revelio_ref_inst_catalog: Path | None
    f1_inst_unitid_crosswalk: Path | None
    ipeds_main_institutions: Path | None
    ipeds_ma_only: Path
    ipeds_ma_ba_only: Path | None
    foia_sevp_with_person_id: Path
    foia_sevp_with_person_id_employment_corrected: Path
    employer_crosswalk: Path
    preferred_rcids: Path
    instrument_components_out: Path
    instrument_panel_out: Path
    treatment_out: Path
    outcomes_out: Path
    analysis_panel_out: Path
    analysis_panel_out_ma_ba: Path | None
    school_shift_metric_panel_out: Path | None
    school_shift_sample_out: Path | None


def _resolve_path(paths_cfg: dict, key: str) -> Path:
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"Config paths.{key} must be set.")
    return Path(value)


def _resolve_optional_path(paths_cfg: dict, key: str) -> Path | None:
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return Path(value)


def _resolve_pipeline_paths(cfg: dict) -> PipelinePaths:
    paths_cfg = get_cfg_section(cfg, "paths")
    return PipelinePaths(
        transitions=_resolve_path(paths_cfg, "transitions_out"),
        headcount=_resolve_path(paths_cfg, "headcounts_out"),
        revelio_ipeds_foia_inst_crosswalk=_resolve_path(paths_cfg, "revelio_ipeds_foia_inst_crosswalk"),
        revelio_inst_deterministic_map=_resolve_optional_path(paths_cfg, "revelio_inst_deterministic_map"),
        revelio_ref_inst_catalog=_resolve_optional_path(paths_cfg, "revelio_ref_inst_catalog"),
        f1_inst_unitid_crosswalk=_resolve_optional_path(paths_cfg, "f1_inst_unitid_crosswalk"),
        ipeds_main_institutions=_resolve_optional_path(paths_cfg, "ipeds_main_institutions"),
        ipeds_ma_only=_resolve_path(paths_cfg, "ipeds_ma_only"),
        ipeds_ma_ba_only=_resolve_optional_path(paths_cfg, "ipeds_ma_ba_only"),
        foia_sevp_with_person_id=_resolve_path(paths_cfg, "foia_sevp_with_person_id"),
        foia_sevp_with_person_id_employment_corrected=_resolve_path(paths_cfg, "foia_sevp_with_person_id_employment_corrected"),
        employer_crosswalk=_resolve_path(paths_cfg, "employer_crosswalk"),
        preferred_rcids=_resolve_path(paths_cfg, "preferred_rcids"),
        instrument_components_out=_resolve_path(paths_cfg, "instrument_components_out"),
        instrument_panel_out=_resolve_path(paths_cfg, "instrument_panel_out"),
        treatment_out=_resolve_path(paths_cfg, "treatment_out"),
        outcomes_out=_resolve_path(paths_cfg, "outcomes_out"),
        analysis_panel_out=_resolve_path(paths_cfg, "analysis_panel_out"),
        analysis_panel_out_ma_ba=_resolve_optional_path(paths_cfg, "analysis_panel_out_ma_ba"),
        school_shift_metric_panel_out=_resolve_optional_path(paths_cfg, "school_shift_metric_panel_out"),
        school_shift_sample_out=_resolve_optional_path(paths_cfg, "school_shift_sample_out"),
    )


def _escape(path: Path) -> str:
    return str(path).replace("'", "''")


def _parse_unitids(raw: str | None) -> list[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _sql_in_list(values: list[object]) -> str:
    if not values:
        return "()"
    escaped = [str(v).replace("'", "''") for v in values]
    return "(" + ",".join(f"'{v}'" for v in escaped) + ")"

def _check_paths(paths: dict[str, Path]) -> None:
    missing = [f"{label}: {p}" for label, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def _ensure_out_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_view(con: ddb.DuckDBPyConnection, view: str, path: Path) -> None:
    _ensure_out_dir(path)
    con.sql(f"COPY (SELECT * FROM {view}) TO '{_escape(path)}' (FORMAT PARQUET)")
    print(f"Wrote {path}")


def _has_column(con: ddb.DuckDBPyConnection, view: str, column: str) -> bool:
    rows = con.execute(f"PRAGMA table_info('{view}')").fetchall()
    cols = {r[1].lower() for r in rows}
    return column.lower() in cols


def _first_present_column(
    con: ddb.DuckDBPyConnection,
    view: str,
    candidates: list[str],
) -> str | None:
    rows = con.execute(f"PRAGMA table_info('{view}')").fetchall()
    cols = {str(r[1]).lower(): str(r[1]) for r in rows}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _normalize_degree_scope(raw: str | None) -> str:
    value = str(raw or "").strip().lower().replace("-", "_")
    if value in {"", "both", "all", "ba_ma", "ma_ba", "bachelors_masters", "masters_bachelors"}:
        return "bachelors_masters"
    if value in {"bachelors", "bachelor", "ba"}:
        return "bachelors"
    if value in {"masters", "master", "ma", "ms"}:
        return "masters"
    raise ValueError(
        "Invalid degree scope. Use one of: "
        "bachelors_masters (default), bachelors, masters."
    )


def _normalize_opt_shift_normalization(raw: str | None) -> str:
    value = str(raw or "").strip().lower().replace("-", "_")
    if value in {"", "ipeds", "ipeds_graduates", "graduates", "grads"}:
        return "ipeds_graduates"
    if value in {"foia", "foia_students", "total_foia_students", "foia_total_students"}:
        return "foia_students"
    if value in {"none", "raw", "count", "counts", "unnormalized"}:
        return "none"
    raise ValueError(
        "Invalid OPT-shift normalization. Use one of: "
        "ipeds_graduates (default), foia_students, none."
    )


def _normalize_school_sample_mode(raw: str | None) -> str:
    value = str(raw or "").strip().lower().replace("-", "_")
    if value in {"", "all", "legacy"}:
        return "all"
    if value in {"matched_shift_sample", "matched_sample", "matched", "sampled"}:
        return "matched_shift_sample"
    raise ValueError(
        "Invalid school sample mode. Use one of: all (default), matched_shift_sample."
    )


def _normalize_school_shift_metric(raw: str | None) -> str:
    value = str(raw or "").strip().lower().replace("-", "_")
    if value in {"", "ihmp", "ihmp_share"}:
        return "ihmp_share"
    if value in {"international", "international_share", "intl", "intl_share"}:
        return "international_share"
    if value in {"opt_ihmp", "opt_ihmp_share"}:
        return "opt_ihmp_share"
    if value in {"opt", "opt_share"}:
        return "opt_share"
    raise ValueError(
        "Invalid school shift metric. Use one of: "
        "ihmp_share, international_share, opt_ihmp_share, opt_share."
    )


def _school_metric_uses_foia(metric: str) -> bool:
    metric_norm = str(metric or "").strip().lower().replace("-", "_")
    return metric_norm in {"opt_ihmp_share", "opt_share", "opt_share_legacy"}


def _school_metric_is_opt_family(metric: str) -> bool:
    metric_norm = str(metric or "").strip().lower().replace("-", "_")
    return metric_norm in {"opt_ihmp_share", "opt_share", "opt_share_legacy"}


def _matched_school_metric_degree_scope(include_bachelors: bool) -> str:
    return "bachelors_masters" if include_bachelors else "masters"


def _resolve_active_shift_metric(
    *,
    school_sample_mode: str,
    school_shift_metric: str | None,
    opt_shifts: bool,
) -> str:
    school_sample_mode = _normalize_school_sample_mode(school_sample_mode)
    if school_sample_mode == "matched_shift_sample":
        metric_raw = school_shift_metric if school_shift_metric is not None else (
            "opt_share" if opt_shifts else "ihmp_share"
        )
        return _normalize_school_shift_metric(metric_raw)
    return "opt_share_legacy" if opt_shifts else "ihmp_share_legacy"


def _degree_predicate_for_scope(
    con: ddb.DuckDBPyConnection,
    view: str = "foia_raw",
    degree_scope: str = "masters",
) -> Optional[str]:
    degree_scope = _normalize_degree_scope(degree_scope)
    if degree_scope == "bachelors_masters":
        degree_word_pred = (
            "LOWER(CAST({col} AS VARCHAR)) LIKE '%master%' "
            "OR LOWER(CAST({col} AS VARCHAR)) LIKE '%bachelor%'"
        )
    elif degree_scope == "bachelors":
        degree_word_pred = "LOWER(CAST({col} AS VARCHAR)) LIKE '%bachelor%'"
    else:
        degree_word_pred = "LOWER(CAST({col} AS VARCHAR)) LIKE '%master%'"

    if _has_column(con, view, "student_edu_level_desc"):
        if degree_scope == "bachelors_masters":
            return (
                "LOWER(CAST(student_edu_level_desc AS VARCHAR)) IN ('master''s', 'masters', 'bachelor''s', 'bachelors') "
                "OR (LOWER(CAST(student_edu_level_desc AS VARCHAR)) LIKE '%master%' "
                "OR LOWER(CAST(student_edu_level_desc AS VARCHAR)) LIKE '%bachelor%')"
            )
        if degree_scope == "bachelors":
            return (
                "LOWER(CAST(student_edu_level_desc AS VARCHAR)) IN ('bachelor''s', 'bachelors') "
                "OR LOWER(CAST(student_edu_level_desc AS VARCHAR)) LIKE '%bachelor%'"
            )
        return (
            "LOWER(CAST(student_edu_level_desc AS VARCHAR)) IN ('master''s', 'masters') "
            "OR LOWER(CAST(student_edu_level_desc AS VARCHAR)) LIKE '%master%'"
        )
    if _has_column(con, view, "awlevel_group"):
        if degree_scope == "bachelors_masters":
            return (
                "LOWER(CAST(awlevel_group AS VARCHAR)) IN ('master', 'masters', 'bachelor', 'bachelors') "
                "OR (LOWER(CAST(awlevel_group AS VARCHAR)) LIKE '%master%' "
                "OR LOWER(CAST(awlevel_group AS VARCHAR)) LIKE '%bachelor%')"
            )
        if degree_scope == "bachelors":
            return (
                "LOWER(CAST(awlevel_group AS VARCHAR)) IN ('bachelor', 'bachelors') "
                "OR LOWER(CAST(awlevel_group AS VARCHAR)) LIKE '%bachelor%'"
            )
        return (
            "LOWER(CAST(awlevel_group AS VARCHAR)) IN ('master', 'masters') "
            "OR LOWER(CAST(awlevel_group AS VARCHAR)) LIKE '%master%'"
        )
    if _has_column(con, view, "awlevel"):
        if degree_scope == "bachelors_masters":
            return "CAST(awlevel AS INTEGER) IN (5, 7)"
        if degree_scope == "bachelors":
            return "CAST(awlevel AS INTEGER) = 5"
        return "CAST(awlevel AS INTEGER) = 7"
    for candidate in ("education_level", "degree_level"):
        if _has_column(con, view, candidate):
            return degree_word_pred.format(col=candidate)
    return None


def _degree_predicate(
    con: ddb.DuckDBPyConnection,
    view: str = "foia_raw",
    include_bachelors: bool = False,
) -> Optional[str]:
    """
    Return a SQL predicate to filter by degree scope if columns exist.
    Prioritized checks follow common FOIA clean outputs.
    """
    scope = "bachelors_masters" if include_bachelors else "masters"
    return _degree_predicate_for_scope(con, view=view, degree_scope=scope)


def _register_inputs(
    con: ddb.DuckDBPyConnection,
    paths: PipelinePaths,
    include_bachelors: bool = False,
    opt_shifts: bool = False,
    school_sample_mode: str = "all",
    school_shift_metric: str | None = None,
    opt_shift_degree_scope: str = "bachelors_masters",
    opt_shifts_normalization: str | None = None,
    opt_shifts_normalize_by_graduates: bool = True,
) -> None:
    school_sample_mode = _normalize_school_sample_mode(school_sample_mode)
    opt_shift_degree_scope = _normalize_degree_scope(opt_shift_degree_scope)
    if opt_shifts_normalization is None:
        opt_shifts_normalization = (
            "ipeds_graduates" if opt_shifts_normalize_by_graduates else "none"
        )
    opt_shifts_normalization = _normalize_opt_shift_normalization(opt_shifts_normalization)
    active_shift_metric = _resolve_active_shift_metric(
        school_sample_mode=school_sample_mode,
        school_shift_metric=school_shift_metric,
        opt_shifts=opt_shifts,
    )
    ipeds_path = paths.ipeds_ma_ba_only if include_bachelors else paths.ipeds_ma_only
    if include_bachelors and ipeds_path is None:
        print(
            "[warn] include_bachelors=true but paths.ipeds_ma_ba_only is not set; "
            "falling back to paths.ipeds_ma_only."
        )
        ipeds_path = paths.ipeds_ma_only
    need_ipeds_ma_ba = include_bachelors or (
        opt_shifts
        and opt_shifts_normalization == "ipeds_graduates"
        and opt_shift_degree_scope in {"bachelors_masters", "bachelors"}
    )
    if need_ipeds_ma_ba and paths.ipeds_ma_ba_only is None:
        print(
            "[warn] paths.ipeds_ma_ba_only is not set; "
            "falling back to masters-only IPEDS where needed."
        )
    required_paths: dict[str, Path] = {
        "transitions": paths.transitions,
        "headcount": paths.headcount,
        "revelio_ipeds_foia_inst_crosswalk": paths.revelio_ipeds_foia_inst_crosswalk,
        "ipeds_ma_only": paths.ipeds_ma_only,
        "foia_raw_full": paths.foia_sevp_with_person_id,
        "foia_raw_corrected": paths.foia_sevp_with_person_id_employment_corrected,
        "employer_cw": paths.employer_crosswalk,
        "preferred_rcids": paths.preferred_rcids,
    }
    if need_ipeds_ma_ba and paths.ipeds_ma_ba_only is not None:
        required_paths["ipeds_ma_ba_only"] = paths.ipeds_ma_ba_only
    if opt_shifts or _school_metric_uses_foia(active_shift_metric):
        if paths.f1_inst_unitid_crosswalk is None:
            raise ValueError(
                "FOIA-based shift metrics require paths.f1_inst_unitid_crosswalk in config."
            )
        required_paths["f1_inst_unitid_crosswalk"] = paths.f1_inst_unitid_crosswalk

    _check_paths(required_paths)

    con.sql(f"CREATE OR REPLACE TEMP VIEW revelio_transitions AS SELECT * FROM read_parquet('{_escape(paths.transitions)}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW revelio_headcount AS SELECT * FROM read_parquet('{_escape(paths.headcount)}')")
    revelio_school_map, revelio_school_map_meta = load_revelio_school_map(
        legacy_crosswalk=paths.revelio_ipeds_foia_inst_crosswalk,
        deterministic_triple_map=paths.revelio_inst_deterministic_map,
        ref_inst_catalog=paths.revelio_ref_inst_catalog,
    )
    con.register("revelio_inst_cw_df", revelio_school_map[["university_raw_key", "unitid"]])
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW revelio_inst_cw AS
        SELECT
            CAST(university_raw_key AS VARCHAR) AS university_raw_norm,
            CAST(unitid AS VARCHAR) AS unitid
        FROM revelio_inst_cw_df
        WHERE university_raw_key IS NOT NULL
          AND unitid IS NOT NULL
        """
    )
    print(
        "[inputs] Revelio school map "
        f"{revelio_school_map_meta['mapping_method']} ({len(revelio_school_map):,} schools)"
    )
    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw_ma AS SELECT * FROM read_parquet('{_escape(paths.ipeds_ma_only)}')")
    has_ipeds_ma_ba = paths.ipeds_ma_ba_only is not None and paths.ipeds_ma_ba_only.exists()
    if has_ipeds_ma_ba:
        con.sql(
            f"CREATE OR REPLACE TEMP VIEW ipeds_raw_ma_ba AS "
            f"SELECT * FROM read_parquet('{_escape(paths.ipeds_ma_ba_only)}')"
        )
    else:
        con.sql("CREATE OR REPLACE TEMP VIEW ipeds_raw_ma_ba AS SELECT * FROM ipeds_raw_ma WHERE 1=0")
    ipeds_src_view = "ipeds_raw_ma_ba" if include_bachelors and has_ipeds_ma_ba else "ipeds_raw_ma"
    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw AS SELECT * FROM {ipeds_src_view}")
    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{_escape(paths.foia_sevp_with_person_id_employment_corrected)}') WHERE year_int > 2005")
    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw_full AS SELECT * FROM read_parquet('{_escape(paths.foia_sevp_with_person_id)}') WHERE year_int > 2005")
    if paths.f1_inst_unitid_crosswalk is not None and paths.f1_inst_unitid_crosswalk.exists():
        con.sql(
            "CREATE OR REPLACE TEMP VIEW f1_inst_unitid_cw AS "
            f"SELECT * FROM read_parquet('{_escape(paths.f1_inst_unitid_crosswalk)}')"
        )
    if paths.ipeds_main_institutions is not None and paths.ipeds_main_institutions.exists():
        con.sql(
            "CREATE OR REPLACE TEMP VIEW ipeds_main_institutions AS "
            f"SELECT * FROM read_parquet('{_escape(paths.ipeds_main_institutions)}')"
        )
    else:
        con.sql(
            """
            CREATE OR REPLACE TEMP VIEW ipeds_main_institutions AS
            SELECT
                CAST(NULL AS VARCHAR) AS main_unitid,
                CAST(NULL AS VARCHAR) AS ipeds_name,
                CAST(NULL AS VARCHAR) AS ipeds_instname_clean
            WHERE 1 = 0
            """
        )
    con.sql(f"CREATE OR REPLACE TEMP VIEW preferred_rcids AS SELECT DISTINCT preferred_rcid FROM read_parquet('{_escape(paths.preferred_rcids)}')")
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW employer_crosswalk AS
        SELECT ec.*
        FROM read_parquet('{_escape(paths.employer_crosswalk)}') AS ec
        JOIN preferred_rcids AS pr
          ON ec.preferred_rcid = pr.preferred_rcid
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW matched_rcids AS
        SELECT DISTINCT preferred_rcid AS rcid
        FROM employer_crosswalk
        WHERE preferred_rcid IS NOT NULL
        """
    )


def _sql_normalize_cip6(colname: str) -> str:
    digits = f"REGEXP_REPLACE(TRIM(CAST({colname} AS VARCHAR)), '[^0-9]', '', 'g')"
    return f"""
        CASE
            WHEN {digits} IS NULL OR TRIM(CAST({digits} AS VARCHAR)) = '' THEN NULL
            ELSE LPAD(SUBSTRING(TRIM(CAST({digits} AS VARCHAR)) FROM 1 FOR 6), 6, '0')
        END
    """


def _empty_school_metric_panel() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "k",
            "t",
            "metric",
            "school_size",
            "metric_level",
            "metric_share",
            "ipeds_total_students",
            "ipeds_total_intl_students",
            "foia_total_students",
            "foia_total_opt_students",
        ]
    )


def _empty_school_shift_sample() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "k",
            "school_name",
            "metric",
            "sample_role",
            "selected_for_instrument",
            "matched_school_k",
            "matched_school_name",
            "matched_pair_id",
            "treated_rank",
            "treated_score",
            "control_eligible",
            "treated_candidate",
            "has_full_window_coverage",
            "meets_min_size",
            "fails_large_yoy_drop",
            "min_required_size",
            "control_positive_cap",
            "avg_size_window",
            "log_avg_size_window",
            "max_positive_annual_change",
            "max_positive_yoy_size_change",
            "max_negative_yoy_size_change",
            "fails_large_yoy_size_jump",
        ]
    )


def _school_name_lookup(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    if not _has_column(con, "ipeds_main_institutions", "main_unitid"):
        return pd.DataFrame(columns=["k", "school_name"])
    lookup = con.sql(
        """
        SELECT
            CAST(main_unitid AS VARCHAR) AS k,
            COALESCE(
                NULLIF(TRIM(CAST(ipeds_name AS VARCHAR)), ''),
                NULLIF(TRIM(CAST(ipeds_instname_clean AS VARCHAR)), '')
            ) AS school_name
        FROM ipeds_main_institutions
        WHERE main_unitid IS NOT NULL
        """
    ).df()
    if lookup.empty:
        return pd.DataFrame(columns=["k", "school_name"])
    lookup["k"] = lookup["k"].astype(str)
    lookup["school_name"] = lookup["school_name"].fillna("").astype(str)
    lookup = lookup.drop_duplicates("k")
    return lookup


def _ipeds_program_year_ctes_sql(exclude_unitids: list[str] | None = None) -> str:
    exclude_unitids = exclude_unitids or []
    exclude_clause = ""
    if exclude_unitids:
        exclude_clause = f"AND CAST(unitid AS VARCHAR) NOT IN {_sql_in_list(exclude_unitids)}"
    return f"""
        ipeds_program_year AS (
            SELECT
                CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k,
                CAST(year AS INTEGER) AS t,
                LPAD(CAST(CAST(cipcode AS BIGINT) AS VARCHAR), 6, '0') AS cip6,
                SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0)) AS program_students,
                SUM(COALESCE(CAST(cnralt AS DOUBLE), 0)) AS program_intl_students,
                CASE
                    WHEN SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0)) > 0
                    THEN SUM(COALESCE(CAST(cnralt AS DOUBLE), 0))
                         / SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0))
                    ELSE NULL
                END AS program_share_intl
            FROM ipeds_raw
            WHERE unitid IS NOT NULL
              AND year IS NOT NULL
              AND cipcode IS NOT NULL
              {exclude_clause}
            GROUP BY 1, 2, 3
        ),
        ipeds_school_year AS (
            SELECT
                k,
                t,
                SUM(program_students) AS school_size,
                SUM(program_intl_students) AS total_intl_students
            FROM ipeds_program_year
            GROUP BY 1, 2
        )
    """


def _foia_school_program_person_ctes_sql(
    con: ddb.DuckDBPyConnection,
    *,
    degree_scope: str,
    exclude_unitids: list[str] | None = None,
) -> str:
    end_col = _first_present_column(
        con,
        "foia_raw",
        ["program_end_date", "program_completion_date", "program_end_dt", "program_complete_date"],
    )
    if end_col is None:
        raise ValueError(
            "FOIA-based school shift metrics require a program end date column "
            "(program_end_date/program_completion_date/program_end_dt/program_complete_date)."
        )
    cip_col = _first_present_column(
        con,
        "foia_raw",
        ["major_1_cip_code", "program_cip_code", "cipcode", "cip_code", "cip"],
    )
    end_expr = _date_parse_sql(end_col)
    opt_date_cols = [
        col
        for col in (
            "opt_employer_start_date",
            "opt_authorization_start_date",
            "authorization_start_date",
        )
        if _has_column(con, "foia_raw", col)
    ]
    if opt_date_cols:
        parsed_opt_dates = ", ".join(_date_parse_sql(col) for col in opt_date_cols)
        opt_activity_expr = f"CASE WHEN COALESCE({parsed_opt_dates}) IS NOT NULL THEN 1 ELSE 0 END"
    else:
        opt_activity_expr = "0"
        print(
            "[warn] Could not find OPT date columns in FOIA input. "
            "FOIA-based school shift numerators will evaluate to zero."
        )
    degree_pred = _degree_predicate_for_scope(con, view="foia_raw", degree_scope=degree_scope)
    degree_clause = f"AND ({degree_pred})" if degree_pred else ""
    if degree_pred is None:
        print(
            "[warn] Could not infer FOIA degree column for school shift metric; "
            "using all degree levels in FOIA numerator."
        )
    exclude_unitids = exclude_unitids or []
    exclude_clause = ""
    if exclude_unitids:
        exclude_clause = f"AND cw.k NOT IN {_sql_in_list(exclude_unitids)}"
    cip_expr = "CAST(NULL AS VARCHAR)"
    if cip_col is not None:
        cip_expr = _sql_normalize_cip6(cip_col)
    return f"""
        cw AS (
            SELECT
                TRIM(CAST(school_name AS VARCHAR)) AS school_name_raw,
                COALESCE(TRIM(CAST(f1_city_clean AS VARCHAR)), '') AS f1_city_clean,
                COALESCE(TRIM(CAST(f1_state_clean AS VARCHAR)), '') AS f1_state_clean,
                COALESCE(TRIM(CAST(f1_zip_clean AS VARCHAR)), '') AS f1_zip_clean,
                CAST(CAST(MIN(UNITID) AS BIGINT) AS VARCHAR) AS k
            FROM f1_inst_unitid_cw
            WHERE UNITID IS NOT NULL
              AND school_name IS NOT NULL
            GROUP BY 1, 2, 3, 4
        ),
        foia_students AS (
            SELECT
                person_id,
                CAST(EXTRACT(YEAR FROM {end_expr}) AS INTEGER) AS t,
                TRIM(CAST(school_name AS VARCHAR)) AS school_name_raw,
                COALESCE({_sql_normalize('campus_city')}, '') AS f1_city_clean,
                COALESCE({_sql_state_name_to_abbr('campus_state')}, '') AS f1_state_clean,
                COALESCE({_sql_clean_zip('campus_zip_code')}, '') AS f1_zip_clean,
                {cip_expr} AS cip6,
                {opt_activity_expr} AS has_opt
            FROM foia_raw
            WHERE person_id IS NOT NULL
              AND school_name IS NOT NULL
              AND {end_expr} IS NOT NULL
              {degree_clause}
        ),
        foia_school_program_person AS (
            SELECT
                cw.k,
                f.t,
                f.person_id,
                f.cip6,
                MAX(f.has_opt) AS ever_opt
            FROM foia_students AS f
            JOIN cw
              ON f.school_name_raw = cw.school_name_raw
             AND f.f1_city_clean = cw.f1_city_clean
             AND f.f1_state_clean = cw.f1_state_clean
             AND f.f1_zip_clean = cw.f1_zip_clean
            WHERE f.t IS NOT NULL
              {exclude_clause}
            GROUP BY 1, 2, 3, 4
        ),
        foia_school_year AS (
            SELECT
                k,
                t,
                COUNT(DISTINCT person_id) AS foia_total_students,
                COUNT(DISTINCT CASE WHEN ever_opt = 1 THEN person_id END) AS foia_total_opt_students
            FROM foia_school_program_person
            GROUP BY 1, 2
        ),
        foia_program_year AS (
            SELECT
                k,
                t,
                cip6,
                COUNT(DISTINCT person_id) AS foia_program_students,
                COUNT(DISTINCT CASE WHEN ever_opt = 1 THEN person_id END) AS foia_program_opt_students,
                CASE
                    WHEN COUNT(DISTINCT person_id) > 0
                    THEN COUNT(DISTINCT CASE WHEN ever_opt = 1 THEN person_id END)::DOUBLE
                         / COUNT(DISTINCT person_id)::DOUBLE
                    ELSE NULL
                END AS foia_program_opt_share
            FROM foia_school_program_person
            WHERE cip6 IS NOT NULL
            GROUP BY 1, 2, 3
        )
    """


def _foia_school_year_ctes_sql(
    con: ddb.DuckDBPyConnection,
    degree_scope: str,
    exclude_unitids: list[str] | None = None,
) -> str:
    """Backward-compatible helper retained for call sites that need foia_school_year only."""
    return _foia_school_program_person_ctes_sql(
        con,
        degree_scope=degree_scope,
        exclude_unitids=exclude_unitids,
    )


def _finalize_school_metric_panel(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        out = _empty_school_metric_panel()
        out["metric"] = pd.Series(dtype="string")
        return out
    out = df.copy()
    out["k"] = out["k"].astype(str)
    out["t"] = pd.to_numeric(out["t"], errors="coerce").astype("Int64")
    numeric_cols = [
        "school_size",
        "metric_level",
        "metric_share",
        "ipeds_total_students",
        "ipeds_total_intl_students",
        "foia_total_students",
        "foia_total_opt_students",
    ]
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["metric"] = metric
    return out[
        [
            "k",
            "t",
            "metric",
            "school_size",
            "metric_level",
            "metric_share",
            "ipeds_total_students",
            "ipeds_total_intl_students",
            "foia_total_students",
            "foia_total_opt_students",
        ]
    ].sort_values(["k", "t"]).reset_index(drop=True)


def _build_ipeds_school_metric_panel(
    con: ddb.DuckDBPyConnection,
    *,
    metric: str,
    exclude_unitids: list[str] | None = None,
) -> pd.DataFrame:
    metric = _normalize_school_shift_metric(metric)
    if metric not in {"ihmp_share", "international_share"}:
        raise ValueError(f"Unsupported IPEDS school metric '{metric}'.")
    if metric == "ihmp_share":
        level_expr = "COALESCE(m.metric_level, 0)"
        metric_cte = """
            school_metric AS (
                SELECT
                    k,
                    t,
                    SUM(
                        CASE
                            WHEN program_share_intl >= 0.5
                             AND program_students >= 10
                            THEN program_students
                            ELSE 0
                        END
                    ) AS metric_level
                FROM ipeds_program_year
                GROUP BY 1, 2
            )
        """
    else:
        level_expr = "COALESCE(sy.total_intl_students, 0)"
        metric_cte = """
            school_metric AS (
                SELECT
                    k,
                    t,
                    total_intl_students AS metric_level
                FROM ipeds_school_year
            )
        """
    df = con.sql(
        f"""
        WITH
        {_ipeds_program_year_ctes_sql(exclude_unitids)},
        {metric_cte}
        SELECT
            sy.k,
            sy.t,
            sy.school_size,
            {level_expr} AS metric_level,
            CASE
                WHEN sy.school_size > 0 THEN {level_expr} / sy.school_size
                ELSE NULL
            END AS metric_share,
            sy.school_size AS ipeds_total_students,
            sy.total_intl_students AS ipeds_total_intl_students,
            CAST(NULL AS DOUBLE) AS foia_total_students,
            CAST(NULL AS DOUBLE) AS foia_total_opt_students
        FROM ipeds_school_year AS sy
        LEFT JOIN school_metric AS m
          ON sy.k = m.k
         AND sy.t = m.t
        ORDER BY sy.k, sy.t
        """
    ).df()
    return _finalize_school_metric_panel(df, metric)


def _build_opt_ihmp_school_metric_panel(
    con: ddb.DuckDBPyConnection,
    *,
    degree_scope: str,
    exclude_unitids: list[str] | None = None,
    ipeds_share_intl_threshold: float = 0.30,
    foia_opt_share_threshold: float = 0.50,
    min_program_f1_count: int = 10,
) -> pd.DataFrame:
    df = con.sql(
        f"""
        WITH
        {_ipeds_program_year_ctes_sql(exclude_unitids)},
        {_foia_school_year_ctes_sql(con, degree_scope=degree_scope, exclude_unitids=exclude_unitids)},
        school_metric AS (
            SELECT
                k,
                t,
                foia_total_opt_students AS metric_level
            FROM foia_school_year
        )
        SELECT
            sy.k,
            sy.t,
            sy.school_size,
            COALESCE(m.metric_level, 0) AS metric_level,
            CASE
                WHEN sy.school_size > 0 THEN COALESCE(m.metric_level, 0) / sy.school_size
                ELSE NULL
            END AS metric_share,
            sy.school_size AS ipeds_total_students,
            sy.total_intl_students AS ipeds_total_intl_students,
            fy.foia_total_students,
            fy.foia_total_opt_students
        FROM ipeds_school_year AS sy
        LEFT JOIN school_metric AS m
          ON sy.k = m.k
         AND sy.t = m.t
        LEFT JOIN foia_school_year AS fy
          ON sy.k = fy.k
         AND sy.t = fy.t
        ORDER BY sy.k, sy.t
        """
    ).df()
    return _finalize_school_metric_panel(df, "opt_ihmp_share")


def _build_opt_share_school_metric_panel(
    con: ddb.DuckDBPyConnection,
    *,
    degree_scope: str,
    exclude_unitids: list[str] | None = None,
) -> pd.DataFrame:
    df = con.sql(
        f"""
        WITH
        {_foia_school_program_person_ctes_sql(con, degree_scope=degree_scope, exclude_unitids=exclude_unitids)}
        SELECT
            fy.k,
            fy.t,
            fy.foia_total_students AS school_size,
            fy.foia_total_opt_students AS metric_level,
            CASE
                WHEN fy.foia_total_students > 0
                THEN fy.foia_total_opt_students::DOUBLE / fy.foia_total_students::DOUBLE
                ELSE NULL
            END AS metric_share,
            CAST(NULL AS DOUBLE) AS ipeds_total_students,
            CAST(NULL AS DOUBLE) AS ipeds_total_intl_students,
            fy.foia_total_students,
            fy.foia_total_opt_students
        FROM foia_school_year AS fy
        ORDER BY fy.k, fy.t
        """
    ).df()
    return _finalize_school_metric_panel(df, "opt_share")


def _build_school_metric_panel(
    con: ddb.DuckDBPyConnection,
    *,
    metric: str,
    degree_scope: str,
    exclude_unitids: list[str] | None = None,
    opt_ihmp_ipeds_share_intl_threshold: float = 0.30,
    opt_ihmp_foia_opt_share_threshold: float = 0.50,
    opt_ihmp_min_program_f1_count: int = 10,
) -> pd.DataFrame:
    metric = _normalize_school_shift_metric(metric)
    if metric in {"ihmp_share", "international_share"}:
        return _build_ipeds_school_metric_panel(
            con,
            metric=metric,
            exclude_unitids=exclude_unitids,
        )
    if metric == "opt_ihmp_share":
        return _build_opt_ihmp_school_metric_panel(
            con,
            degree_scope=degree_scope,
            exclude_unitids=exclude_unitids,
            ipeds_share_intl_threshold=opt_ihmp_ipeds_share_intl_threshold,
            foia_opt_share_threshold=opt_ihmp_foia_opt_share_threshold,
            min_program_f1_count=opt_ihmp_min_program_f1_count,
        )
    if metric == "opt_share":
        return _build_opt_share_school_metric_panel(
            con,
            degree_scope=degree_scope,
            exclude_unitids=exclude_unitids,
        )
    raise ValueError(f"Unsupported school shift metric '{metric}'.")


def _build_school_shift_sample(
    metric_panel: pd.DataFrame,
    *,
    metric: str,
    school_name_lookup: pd.DataFrame | None = None,
    n_shifted: int = 25,
    window_start: int = 2014,
    window_end: int = 2017,
    control_positive_cap: float = 0.02,
    min_school_size: int = 100,
    opt_share_min_school_f1_count: int = 50,
    opt_share_max_yoy_drop: float = 0.50,
    restrict_treated_to_no_large_enrollment_jump: bool = False,
    max_yoy_size_jump: float = 0.50,
) -> pd.DataFrame:
    metric = _normalize_school_shift_metric(metric)
    if metric_panel.empty:
        return _empty_school_shift_sample()
    years = list(range(int(window_start), int(window_end) + 1))
    if len(years) < 2:
        raise ValueError("School sample window must span at least two years.")
    work = metric_panel.loc[
        metric_panel["t"].isin(years),
        ["k", "t", "school_size", "metric_level", "metric_share"],
    ].copy()
    if work.empty:
        return _empty_school_shift_sample()
    share_wide = work.pivot(index="k", columns="t", values="metric_share").reindex(columns=years)
    size_wide = work.pivot(index="k", columns="t", values="school_size").reindex(columns=years)
    level_wide = work.pivot(index="k", columns="t", values="metric_level").reindex(columns=years)
    summary = pd.DataFrame({"k": share_wide.index.astype(str)})
    summary["metric"] = metric
    summary = summary.set_index("k", drop=False)
    summary.index.name = None
    name_map: dict[str, str] = {}
    if school_name_lookup is not None and not school_name_lookup.empty:
        name_map = dict(
            zip(
                school_name_lookup["k"].astype(str),
                school_name_lookup["school_name"].fillna("").astype(str),
            )
        )
    summary["school_name"] = summary.index.map(name_map).fillna("")
    for year in years:
        summary[f"school_size_{year}"] = size_wide[year]
        summary[f"metric_level_{year}"] = level_wide[year]
        summary[f"metric_share_{year}"] = share_wide[year]
    share_change_cols: list[str] = []
    size_change_cols: list[str] = []
    for year_prev, year_curr in zip(years, years[1:]):
        share_col = f"metric_share_change_{year_prev}_{year_curr}"
        size_col = f"school_size_pct_change_{year_prev}_{year_curr}"
        summary[share_col] = summary[f"metric_share_{year_curr}"] - summary[f"metric_share_{year_prev}"]
        prev_size = summary[f"school_size_{year_prev}"]
        curr_size = summary[f"school_size_{year_curr}"]
        summary[size_col] = curr_size / prev_size - 1.0
        share_change_cols.append(share_col)
        size_change_cols.append(size_col)
    summary["has_full_window_coverage"] = (
        size_wide.notna().all(axis=1) & share_wide.notna().all(axis=1)
    ).astype("int8")
    min_required_size = (
        int(opt_share_min_school_f1_count)
        if metric == "opt_share"
        else int(min_school_size)
    )
    summary["min_required_size"] = min_required_size
    summary["meets_min_size"] = (
        size_wide.ge(float(min_required_size)).all(axis=1)
    ).astype("int8")
    positive_changes = summary[share_change_cols].clip(lower=0)
    summary["max_positive_annual_change"] = positive_changes.max(axis=1, skipna=True)
    summary["treated_score"] = positive_changes.max(axis=1, skipna=True)
    summary["control_positive_cap"] = float(control_positive_cap)
    summary["max_positive_yoy_size_change"] = summary[size_change_cols].max(axis=1, skipna=True)
    if metric == "opt_share":
        summary["max_negative_yoy_size_change"] = summary[size_change_cols].min(axis=1, skipna=True)
        summary["fails_large_yoy_drop"] = (
            summary["max_negative_yoy_size_change"] < -float(opt_share_max_yoy_drop)
        ).astype("int8")
    else:
        summary["max_negative_yoy_size_change"] = pd.NA
        summary["fails_large_yoy_drop"] = 0
    if restrict_treated_to_no_large_enrollment_jump:
        summary["fails_large_yoy_size_jump"] = (
            summary["max_positive_yoy_size_change"] > float(max_yoy_size_jump)
        ).astype("int8")
    else:
        summary["fails_large_yoy_size_jump"] = 0
    base_eligible = (
        (summary["has_full_window_coverage"] == 1)
        & (summary["meets_min_size"] == 1)
        & (summary["fails_large_yoy_drop"] == 0)
    )
    treated_base_eligible = (
        base_eligible
        if not restrict_treated_to_no_large_enrollment_jump
        else (base_eligible & (summary["fails_large_yoy_size_jump"] == 0))
    )
    summary["control_eligible"] = (
        base_eligible
        & positive_changes.le(float(control_positive_cap)).all(axis=1)
    ).astype("int8")
    summary["treated_candidate"] = (
        treated_base_eligible & summary["treated_score"].gt(0)
    ).astype("int8")
    size_cols = [f"school_size_{year}" for year in years]
    summary["avg_size_window"] = summary[size_cols].mean(axis=1, skipna=True)
    summary["log_avg_size_window"] = summary["avg_size_window"].apply(
        lambda value: math.log(value) if pd.notna(value) and value > 0 else float("nan")
    )
    summary["selected_for_instrument"] = 0
    summary["sample_role"] = pd.NA
    summary["matched_school_k"] = pd.NA
    summary["matched_school_name"] = pd.NA
    summary["matched_pair_id"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    treated_candidates = (
        summary.loc[summary["treated_candidate"] == 1]
        .sort_values(["treated_score", "k"], ascending=[False, True])
        .copy()
    )
    summary["treated_rank"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    if not treated_candidates.empty:
        summary.loc[treated_candidates.index, "treated_rank"] = pd.Series(
            range(1, len(treated_candidates) + 1),
            index=treated_candidates.index,
            dtype="Int64",
        )
    selected_treated = treated_candidates.head(int(n_shifted)).copy()
    control_pool = summary.loc[
        (summary["control_eligible"] == 1) & (~summary.index.isin(selected_treated.index))
    ].copy()
    assignments: list[dict[str, object]] = []
    used_controls: set[str] = set()
    pair_id = 0
    for treated_k, treated_row in selected_treated.iterrows():
        available = control_pool.loc[~control_pool.index.isin(used_controls)].copy()
        if available.empty:
            break
        available["match_gap"] = (
            available["log_avg_size_window"] - treated_row["log_avg_size_window"]
        ).abs()
        available = available.sort_values(["match_gap", "k"], ascending=[True, True])
        chosen = available.iloc[0]
        pair_id += 1
        assignments.append(
            {
                "treated_k": treated_k,
                "control_k": str(chosen["k"]),
                "matched_pair_id": pair_id,
            }
        )
        used_controls.add(str(chosen["k"]))
    if pair_id == 0 and int(n_shifted) > 0:
        raise ValueError(
            "Matched-school sampling found no treat-control pairs. "
            "Check metric thresholds and school-size requirements."
        )
    if pair_id < int(n_shifted):
        print(
            f"[warn] Requested {int(n_shifted)} shifted schools, "
            f"but only matched {pair_id} treat-control pair(s)."
        )
    for assignment in assignments:
        treated_k = str(assignment["treated_k"])
        control_k = str(assignment["control_k"])
        matched_pair_id = int(assignment["matched_pair_id"])
        summary.loc[treated_k, "selected_for_instrument"] = 1
        summary.loc[treated_k, "sample_role"] = "treated"
        summary.loc[treated_k, "matched_school_k"] = control_k
        summary.loc[treated_k, "matched_school_name"] = name_map.get(control_k, "")
        summary.loc[treated_k, "matched_pair_id"] = matched_pair_id
        summary.loc[control_k, "selected_for_instrument"] = 1
        summary.loc[control_k, "sample_role"] = "control"
        summary.loc[control_k, "matched_school_k"] = treated_k
        summary.loc[control_k, "matched_school_name"] = name_map.get(treated_k, "")
        summary.loc[control_k, "matched_pair_id"] = matched_pair_id
    out = summary.reset_index(drop=True)
    sort_cols = ["selected_for_instrument", "sample_role", "treated_rank", "k"]
    ascending = [False, True, True, True]
    out = out.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    return out


def _register_school_metric_views(
    con: ddb.DuckDBPyConnection,
    *,
    metric_panel: pd.DataFrame,
    sample_summary: pd.DataFrame,
) -> tuple[str, str]:
    panel = metric_panel.copy()
    if panel.empty:
        panel = _empty_school_metric_panel()
    if sample_summary.empty:
        sample_summary = _empty_school_shift_sample()
    sample_cols = ["k", "school_name", "selected_for_instrument", "sample_role", "matched_school_k", "matched_pair_id"]
    panel = panel.merge(sample_summary[sample_cols], on="k", how="left")
    con.register("school_shift_metric_panel_df", panel)
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW school_shift_metric_panel AS
        SELECT * FROM school_shift_metric_panel_df
        """
    )
    con.register("school_shift_sample_df", sample_summary)
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW school_shift_sample AS
        SELECT * FROM school_shift_sample_df
        """
    )
    return "school_shift_metric_panel", "school_shift_sample"


def _matched_school_sample_preview_table(
    sample_summary: pd.DataFrame,
    *,
    metric: str,
    window_start: int,
    window_end: int,
) -> pd.DataFrame:
    if sample_summary.empty:
        return pd.DataFrame()
    years = list(range(int(window_start), int(window_end) + 1))
    preview = sample_summary.loc[
        sample_summary["selected_for_instrument"] == 1
    ].copy()
    if preview.empty:
        return pd.DataFrame()
    role_order = pd.CategoricalDtype(["treated", "control"], ordered=True)
    preview["sample_role"] = preview["sample_role"].astype(role_order)
    preview["matched_pair_id_sort"] = pd.to_numeric(
        preview["matched_pair_id"], errors="coerce"
    )
    preview = preview.sort_values(
        ["matched_pair_id_sort", "sample_role", "school_name", "k"],
        na_position="last",
    )
    cols = [
        "matched_pair_id",
        "sample_role",
        "school_name",
        "k",
        "matched_school_name",
        "matched_school_k",
    ]
    rename_map = {
        "matched_pair_id": "pair_id",
        "sample_role": "role",
        "school_name": "school",
        "k": "unitid",
        "matched_school_name": "matched_school",
        "matched_school_k": "matched_unitid",
    }
    for year in years:
        share_col = f"metric_share_{year}"
        size_col = f"school_size_{year}"
        if share_col not in preview.columns:
            preview[share_col] = pd.NA
        if size_col not in preview.columns:
            preview[size_col] = pd.NA
        cols.append(share_col)
        cols.append(size_col)
        rename_map[share_col] = f"{metric}_share_{year}"
        rename_map[size_col] = size_col
    preview = preview[cols].rename(columns=rename_map).reset_index(drop=True)
    return preview


def _confirm_matched_school_sample(
    sample_summary: pd.DataFrame,
    *,
    metric: str,
    window_start: int,
    window_end: int,
) -> None:
    preview = _matched_school_sample_preview_table(
        sample_summary,
        metric=metric,
        window_start=window_start,
        window_end=window_end,
    )
    print(
        "\n[confirm] Matched-school sample preview "
        f"(metric={metric}; window={int(window_start)}-{int(window_end)}):"
    )
    if preview.empty:
        print("[confirm] No selected treated/control schools to review.")
        raise SystemExit(1)
    print(preview.to_string(index=False))
    try:
        response = input("Continue with this matched-school sample? [y/N]: ")
    except EOFError:
        print("\n[confirm] No confirmation received. Exiting.")
        raise SystemExit(1) from None
    if response.strip().lower() not in {"y", "yes"}:
        print("[confirm] User declined matched-school sample. Exiting.")
        raise SystemExit(1)


def _create_sampled_school_growth_view(con: ddb.DuckDBPyConnection) -> str:
    view_name = "ipeds_unit_growth"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        WITH selected AS (
            SELECT
                m.k,
                CAST(m.t AS INTEGER) AS t,
                CAST(COALESCE(m.metric_level, 0) AS DOUBLE) AS metric_level
            FROM school_shift_metric_panel AS m
            JOIN school_shift_sample AS s
              ON m.k = s.k
            WHERE COALESCE(s.selected_for_instrument, 0) = 1
              AND m.t IS NOT NULL
        ),
        bounds AS (
            SELECT
                k,
                MIN(t) AS min_t,
                MAX(t) AS max_t
            FROM selected
            GROUP BY k
        ),
        expanded AS (
            SELECT
                b.k,
                gs.year AS t
            FROM bounds AS b,
            LATERAL generate_series(b.min_t, b.max_t) AS gs(year)
        ),
        filled AS (
            SELECT
                e.k,
                e.t,
                COALESCE(s.metric_level, 0) AS metric_level
            FROM expanded AS e
            LEFT JOIN selected AS s
              ON e.k = s.k
             AND e.t = s.t
        )
        SELECT
            k,
            t,
            metric_level AS g_kt,
            metric_level AS g_kt_all,
            metric_level AS g_kt_intl
        FROM filled
        ORDER BY k, t
        """
    )
    return view_name


def _create_growth_view_for_pipeline(
    con: ddb.DuckDBPyConnection,
    *,
    school_sample_mode: str,
    school_shift_metric: str | None,
    include_bachelors: bool,
    use_changes: bool,
    use_log_y: bool,
    opt_shifts: bool,
    opt_shifts_degree_scope: str,
    opt_shifts_normalization: str,
    opt_shifts_normalize_by_graduates: bool,
    exclude_unitids: list[str] | None = None,
    school_sample_n_shifted: int = 25,
    school_sample_window_start: int = 2014,
    school_sample_window_end: int = 2017,
    school_sample_control_positive_cap: float = 0.02,
    school_sample_min_size: int = 100,
    opt_ihmp_ipeds_share_intl_threshold: float = 0.30,
    opt_ihmp_foia_opt_share_threshold: float = 0.50,
    opt_ihmp_min_program_f1_count: int = 10,
    opt_share_min_school_f1_count: int = 50,
    opt_share_max_yoy_drop: float = 0.50,
    restrict_treated_to_no_large_enrollment_jump: bool = False,
    school_sample_max_yoy_size_jump: float = 0.50,
    confirm_matched_school_sample: bool = False,
) -> tuple[str, str | None, str | None]:
    school_sample_mode = _normalize_school_sample_mode(school_sample_mode)
    if school_sample_mode == "all":
        if opt_shifts:
            growth_view = _create_opt_shift_growth_view(
                con,
                use_changes=use_changes,
                demean_by_school=use_log_y,
                degree_scope=opt_shifts_degree_scope,
                normalization=opt_shifts_normalization,
                normalize_by_graduates=opt_shifts_normalize_by_graduates,
                exclude_unitids=exclude_unitids,
            )
        else:
            growth_view = _create_ipeds_growth_view(
                con,
                use_changes=use_changes,
                demean_by_school=use_log_y,
                exclude_unitids=exclude_unitids,
            )
        return growth_view, None, None

    metric = _normalize_school_shift_metric(
        school_shift_metric if school_shift_metric is not None else ("opt_share" if opt_shifts else "ihmp_share")
    )
    degree_scope = _matched_school_metric_degree_scope(include_bachelors=include_bachelors)
    metric_panel = _build_school_metric_panel(
        con,
        metric=metric,
        degree_scope=degree_scope,
        exclude_unitids=exclude_unitids,
        opt_ihmp_ipeds_share_intl_threshold=opt_ihmp_ipeds_share_intl_threshold,
        opt_ihmp_foia_opt_share_threshold=opt_ihmp_foia_opt_share_threshold,
        opt_ihmp_min_program_f1_count=opt_ihmp_min_program_f1_count,
    )
    school_names = _school_name_lookup(con)
    sample_summary = _build_school_shift_sample(
        metric_panel,
        metric=metric,
        school_name_lookup=school_names,
        n_shifted=school_sample_n_shifted,
        window_start=school_sample_window_start,
        window_end=school_sample_window_end,
        control_positive_cap=school_sample_control_positive_cap,
        min_school_size=school_sample_min_size,
        opt_share_min_school_f1_count=opt_share_min_school_f1_count,
        opt_share_max_yoy_drop=opt_share_max_yoy_drop,
        restrict_treated_to_no_large_enrollment_jump=restrict_treated_to_no_large_enrollment_jump,
        max_yoy_size_jump=school_sample_max_yoy_size_jump,
    )
    _register_school_metric_views(
        con,
        metric_panel=metric_panel,
        sample_summary=sample_summary,
    )
    if use_changes:
        print(
            "[info] school_sample_mode=matched_shift_sample: school selection uses annual "
            "share changes, but the instrument is built from school-level levels."
        )
    print(
        "[info] matched-school sampling: "
        f"metric={metric}, "
        f"selected_schools={int(sample_summary.get('selected_for_instrument', pd.Series(dtype='int64')).sum())}"
    )
    if confirm_matched_school_sample:
        _confirm_matched_school_sample(
            sample_summary,
            metric=metric,
            window_start=school_sample_window_start,
            window_end=school_sample_window_end,
        )
    growth_view = _create_sampled_school_growth_view(con)
    return growth_view, "school_shift_metric_panel", "school_shift_sample"


def _create_ipeds_growth_view(
    con: ddb.DuckDBPyConnection,
    use_changes: bool = False,
    demean_by_school: bool = False,
    exclude_unitids: list[str] | None = None,
) -> str:
    """
    Recompute g_kt using the IHMA rules (international-heavy master's programs).
    """
    view_name = "ipeds_unit_growth"
    g_expr = "g_kt"
    g_all_expr = "g_kt_all"
    g_intl_expr = "g_kt_intl"
    if use_changes:
        g_expr = "ASINH(g_kt) - ASINH(g_kt_lag)"
        g_all_expr = "ASINH(g_kt_all) - ASINH(g_kt_all_lag)"
        g_intl_expr = "ASINH(g_kt_intl) - ASINH(g_kt_intl_lag)"
    demean_cte = ""
    final_view = "raw_out"
    if demean_by_school:
        demean_cte = """
        ,
        demeaned AS (
            SELECT
                k,
                t,
                g_kt - AVG(g_kt) OVER (PARTITION BY k) AS g_kt,
                g_kt_all - AVG(g_kt_all) OVER (PARTITION BY k) AS g_kt_all,
                g_kt_intl - AVG(g_kt_intl) OVER (PARTITION BY k) AS g_kt_intl
            FROM raw_out
        )
        """
        final_view = "demeaned"

    exclude_unitids = exclude_unitids or []
    exclude_clause = ""
    if exclude_unitids:
        exclude_clause = f"AND CAST(unitid AS VARCHAR) NOT IN {_sql_in_list(exclude_unitids)}"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        WITH base AS (
            SELECT
                CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k,
                CAST(cipcode AS VARCHAR) AS cipcode,
                CAST(year AS INTEGER) AS year,
                CAST(cnralt AS DOUBLE) AS cnralt,
                CAST(ctotalt AS DOUBLE) AS ctotalt,
                CAST(share_intl AS DOUBLE) AS share_intl
            FROM ipeds_raw
            WHERE unitid IS NOT NULL
              {exclude_clause}
        ),
        program_flags AS (
            SELECT
                k,
                cipcode,
                MAX(
                    CASE
                        WHEN year > 2010
                         AND share_intl >= 0.5
                         AND ctotalt >= 10
                        THEN 1
                        ELSE 0
                    END
                ) AS international_heavy
            FROM base
            WHERE cipcode IS NOT NULL
            GROUP BY k, cipcode
        ),
        joined AS (
            SELECT
                b.k,
                b.year,
                SUM(b.cnralt) AS tot_intl_students,
                SUM(COALESCE(b.ctotalt, 0) * COALESCE(p.international_heavy, 0)) AS tot_seats_ihma,
                SUM(COALESCE(b.cnralt, 0) * COALESCE(p.international_heavy, 0)) AS tot_intl_seats_ihma
            FROM base AS b
            LEFT JOIN program_flags AS p
                ON b.k = p.k
               AND b.cipcode = p.cipcode
            GROUP BY b.k, b.year
        ),
        bounds AS (
            SELECT
                k,
                MIN(year) AS min_year,
                MAX(year) AS max_year
            FROM joined
            GROUP BY k
        ),
        expanded AS (
            SELECT
                b.k,
                gs.year
            FROM bounds b,
            LATERAL generate_series(b.min_year, b.max_year) AS gs(year)
        ),
        filled AS (
            SELECT
                e.k,
                e.year,
                COALESCE(j.tot_intl_students, 0) AS tot_intl_students,
                COALESCE(j.tot_seats_ihma, 0) AS tot_seats_ihma,
                COALESCE(j.tot_intl_seats_ihma, 0) AS tot_intl_seats_ihma
            FROM expanded e
            LEFT JOIN joined j
                ON e.k = j.k
               AND e.year = j.year
        ),
        diffed AS (
            SELECT
                k,
                year,
                tot_intl_students AS g_kt_all,
                tot_seats_ihma AS g_kt, 
                tot_intl_seats_ihma AS g_kt_intl
            FROM filled
        ),
        with_lags AS (
            SELECT
                k,
                year,
                g_kt,
                g_kt_all,
                g_kt_intl,
                LAG(g_kt) OVER (PARTITION BY k ORDER BY year) AS g_kt_lag,
                LAG(g_kt_all) OVER (PARTITION BY k ORDER BY year) AS g_kt_all_lag,
                LAG(g_kt_intl) OVER (PARTITION BY k ORDER BY year) AS g_kt_intl_lag
            FROM diffed
        ),
        base_out AS (
            SELECT
                k,
                year AS t,
                {g_expr} AS g_kt,
                {g_all_expr} AS g_kt_all,
                {g_intl_expr} AS g_kt_intl
            FROM with_lags
            WHERE g_kt IS NOT NULL
        ),
        raw_out AS (
            SELECT * FROM base_out
        )
        {demean_cte}
        SELECT
            k,
            t,
            g_kt,
            g_kt_all,
            g_kt_intl
        FROM {final_view}
        """
    )
    return view_name


def _create_opt_shift_growth_view(
    con: ddb.DuckDBPyConnection,
    use_changes: bool = False,
    demean_by_school: bool = False,
    degree_scope: str = "bachelors_masters",
    normalization: str | None = None,
    normalize_by_graduates: bool = True,
    exclude_unitids: list[str] | None = None,
) -> str:
    """
    Build g_kt from FOIA OPT usage by school-year.
    Normalization modes:
      - ipeds_graduates: opt_users / IPEDS graduates
      - foia_students: opt_users / FOIA students
      - none: raw opt_users
    t is the FOIA program end year.
    """
    view_name = "ipeds_unit_growth"
    degree_scope = _normalize_degree_scope(degree_scope)
    if normalization is None:
        normalization = "ipeds_graduates" if normalize_by_graduates else "none"
    normalization = _normalize_opt_shift_normalization(normalization)
    print(
        "[info] Building OPT-based shifts with "
        f"degree_scope={degree_scope}; "
        f"normalization={normalization}."
    )

    end_col = _first_present_column(
        con,
        "foia_raw_full",
        ["program_end_date", "program_completion_date", "program_end_dt", "program_complete_date"],
    )
    if end_col is None:
        raise ValueError(
            "opt_shifts=true requires a FOIA program end date column "
            "(program_end_date/program_completion_date/program_end_dt/program_complete_date)."
        )
    end_expr = _date_parse_sql(end_col)

    opt_date_cols = [
        col
        for col in (
            "opt_employer_start_date",
            "opt_authorization_start_date",
            "authorization_start_date",
        )
        if _has_column(con, "foia_raw_full", col)
    ]
    if opt_date_cols:
        parsed_opt_dates = ", ".join(_date_parse_sql(col) for col in opt_date_cols)
        opt_activity_expr = f"CASE WHEN COALESCE({parsed_opt_dates}) IS NOT NULL THEN 1 ELSE 0 END"
    else:
        opt_activity_expr = "0"
        print(
            "[warn] Could not find OPT date columns in FOIA input. "
            "OPT-shift numerator will evaluate to zero."
        )

    degree_pred = _degree_predicate_for_scope(con, view="foia_raw_full", degree_scope=degree_scope)
    degree_clause = ""
    if degree_pred:
        degree_clause = f"AND ({degree_pred})"
    else:
        print(
            "[warn] Could not infer FOIA degree column for opt_shifts; "
            "using all degree levels in FOIA numerator."
        )

    exclude_unitids = exclude_unitids or []
    exclude_foia_clause = ""
    exclude_ipeds_clause = ""
    if exclude_unitids:
        in_list = _sql_in_list(exclude_unitids)
        exclude_foia_clause = f"AND cw.k NOT IN {in_list}"
        exclude_ipeds_clause = f"AND CAST(unitid AS VARCHAR) NOT IN {in_list}"

    post_opt_counts_ctes = """
        school_years AS (
            SELECT k, t FROM opt_counts
        ),
    """
    filled_metric_select = """
                CAST(NULL AS DOUBLE) AS total_graduates,
                COALESCE(o.opt_students, 0) AS opt_metric_kt
    """
    filled_join_clause = """
            LEFT JOIN opt_counts AS o
              ON e.k = o.k
             AND e.t = o.t
    """
    if normalization == "ipeds_graduates":
        if degree_scope == "masters":
            denom_expr = "COALESCE(ma.total_graduates_ma, 0)"
        elif degree_scope == "bachelors":
            denom_expr = (
                "GREATEST(COALESCE(maba.total_graduates_maba, 0) "
                "- COALESCE(ma.total_graduates_ma, 0), 0)"
            )
        else:
            denom_expr = "COALESCE(maba.total_graduates_maba, COALESCE(ma.total_graduates_ma, 0))"
        post_opt_counts_ctes = f"""
        ipeds_ma AS (
            SELECT
                CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k,
                CAST(year AS INTEGER) AS t,
                SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0)) AS total_graduates_ma
            FROM ipeds_raw_ma
            WHERE unitid IS NOT NULL
              AND year IS NOT NULL
              {exclude_ipeds_clause}
            GROUP BY 1, 2
        ),
        ipeds_maba AS (
            SELECT
                CAST(CAST(unitid AS BIGINT) AS VARCHAR) AS k,
                CAST(year AS INTEGER) AS t,
                SUM(COALESCE(CAST(ctotalt AS DOUBLE), 0)) AS total_graduates_maba
            FROM ipeds_raw_ma_ba
            WHERE unitid IS NOT NULL
              AND year IS NOT NULL
              {exclude_ipeds_clause}
            GROUP BY 1, 2
        ),
        denominator AS (
            SELECT
                COALESCE(ma.k, maba.k) AS k,
                COALESCE(ma.t, maba.t) AS t,
                {denom_expr} AS total_graduates
            FROM ipeds_ma AS ma
            FULL OUTER JOIN ipeds_maba AS maba
              ON ma.k = maba.k
             AND ma.t = maba.t
        ),
        school_years AS (
            SELECT k, t FROM denominator
            UNION
            SELECT k, t FROM opt_counts
        ),
        """
        filled_metric_select = """
                d.total_graduates AS total_graduates,
                CASE
                    WHEN d.total_graduates > 0
                    THEN COALESCE(o.opt_students, 0) / d.total_graduates
                    ELSE 0
                END AS opt_metric_kt
        """
        filled_join_clause = """
            LEFT JOIN opt_counts AS o
              ON e.k = o.k
             AND e.t = o.t
            JOIN denominator AS d
              ON e.k = d.k
             AND e.t = d.t
        """
    elif normalization == "foia_students":
        post_opt_counts_ctes = """
        denominator AS (
            SELECT
                k,
                t,
                COUNT(DISTINCT person_id) AS total_foia_students
            FROM foia_school_year_person
            GROUP BY 1, 2
        ),
        school_years AS (
            SELECT k, t FROM denominator
            UNION
            SELECT k, t FROM opt_counts
        ),
        """
        filled_metric_select = """
                d.total_foia_students AS total_graduates,
                CASE
                    WHEN d.total_foia_students > 0
                    THEN COALESCE(o.opt_students, 0) / d.total_foia_students
                    ELSE 0
                END AS opt_metric_kt
        """
        filled_join_clause = """
            LEFT JOIN opt_counts AS o
              ON e.k = o.k
             AND e.t = o.t
            JOIN denominator AS d
              ON e.k = d.k
             AND e.t = d.t
        """

    g_expr = "opt_metric_kt"
    g_all_expr = "opt_metric_kt"
    g_intl_expr = "opt_metric_kt"
    if use_changes:
        g_expr = "ASINH(opt_metric_kt) - ASINH(opt_metric_kt_lag)"
        g_all_expr = g_expr
        g_intl_expr = g_expr
    demean_cte = ""
    final_view = "raw_out"
    if demean_by_school:
        demean_cte = """
        ,
        demeaned AS (
            SELECT
                k,
                t,
                g_kt - AVG(g_kt) OVER (PARTITION BY k) AS g_kt,
                g_kt_all - AVG(g_kt_all) OVER (PARTITION BY k) AS g_kt_all,
                g_kt_intl - AVG(g_kt_intl) OVER (PARTITION BY k) AS g_kt_intl
            FROM raw_out
        )
        """
        final_view = "demeaned"

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        WITH cw AS (
            SELECT
                TRIM(CAST(school_name AS VARCHAR)) AS school_name_raw,
                COALESCE(TRIM(CAST(f1_city_clean AS VARCHAR)), '') AS f1_city_clean,
                COALESCE(TRIM(CAST(f1_state_clean AS VARCHAR)), '') AS f1_state_clean,
                COALESCE(TRIM(CAST(f1_zip_clean AS VARCHAR)), '') AS f1_zip_clean,
                CAST(CAST(MIN(UNITID) AS BIGINT) AS VARCHAR) AS k
            FROM f1_inst_unitid_cw
            WHERE UNITID IS NOT NULL
              AND school_name IS NOT NULL
            GROUP BY 1, 2, 3, 4
        ),
        foia_students AS (
            SELECT
                person_id,
                CAST(EXTRACT(YEAR FROM {end_expr}) AS INTEGER) AS t,
                TRIM(CAST(school_name AS VARCHAR)) AS school_name_raw,
                COALESCE({_sql_normalize('campus_city')}, '') AS f1_city_clean,
                COALESCE({_sql_state_name_to_abbr('campus_state')}, '') AS f1_state_clean,
                COALESCE({_sql_clean_zip('campus_zip_code')}, '') AS f1_zip_clean,
                {opt_activity_expr} AS has_opt
            FROM foia_raw_full
            WHERE person_id IS NOT NULL
              AND school_name IS NOT NULL
              AND {end_expr} IS NOT NULL
              {degree_clause}
        ),
        foia_school_year_person AS (
            SELECT
                cw.k,
                f.t,
                f.person_id,
                MAX(f.has_opt) AS ever_opt
            FROM foia_students AS f
            JOIN cw
              ON f.school_name_raw = cw.school_name_raw
             AND f.f1_city_clean = cw.f1_city_clean
             AND f.f1_state_clean = cw.f1_state_clean
             AND f.f1_zip_clean = cw.f1_zip_clean
            WHERE f.t IS NOT NULL
              {exclude_foia_clause}
            GROUP BY cw.k, f.t, f.person_id
        ),
        opt_counts AS (
            SELECT
                k,
                t,
                COUNT(DISTINCT CASE WHEN ever_opt = 1 THEN person_id END) AS opt_students
            FROM foia_school_year_person
            GROUP BY k, t
        ),
        {post_opt_counts_ctes}
        bounds AS (
            SELECT
                k,
                MIN(t) AS min_t,
                MAX(t) AS max_t
            FROM school_years
            GROUP BY k
        ),
        expanded AS (
            SELECT
                b.k,
                gs.year AS t
            FROM bounds b,
            LATERAL generate_series(b.min_t, b.max_t) AS gs(year)
        ),
        filled AS (
            SELECT
                e.k,
                e.t,
                COALESCE(o.opt_students, 0) AS opt_students,
                {filled_metric_select}
            FROM expanded AS e
            {filled_join_clause}
        ),
        with_lags AS (
            SELECT
                k,
                t,
                opt_metric_kt,
                LAG(opt_metric_kt) OVER (PARTITION BY k ORDER BY t) AS opt_metric_kt_lag
            FROM filled
        ),
        base_out AS (
            SELECT
                k,
                t,
                {g_expr} AS g_kt,
                {g_all_expr} AS g_kt_all,
                {g_intl_expr} AS g_kt_intl
            FROM with_lags
            WHERE opt_metric_kt IS NOT NULL
        ),
        raw_out AS (
            SELECT * FROM base_out
        )
        {demean_cte}
        SELECT
            k,
            t,
            g_kt,
            g_kt_all,
            g_kt_intl
        FROM {final_view}
        """
    )
    return view_name


def _create_transition_share_view(
    con: ddb.DuckDBPyConnection,
    share_base_year: int,
    exclude_unitids: list[str] | None = None,
) -> str:
    """
    Build base-year shares s_ck: share of new hires at company c from university k
    in the base-year window.
    Also builds share_ck_full using all available years.
    """
    window_start = int(share_base_year) - 5
    window_end = int(share_base_year)
    exclude_unitids = exclude_unitids or []
    exclude_clause = ""
    if exclude_unitids:
        exclude_clause = f"AND CAST(cw.unitid AS VARCHAR) NOT IN {_sql_in_list(exclude_unitids)}"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW transition_unit_shares AS
        WITH transitions_window AS (
            SELECT
                CAST(t.rcid AS INTEGER) AS c,
                CAST(CAST(cw.unitid AS BIGINT) AS VARCHAR) AS k,
                CAST(t.year AS INTEGER) AS year,
                t.n_transitions,
                t.total_new_hires AS total_new_hires_alt,
                SUM(t.n_transitions) OVER(PARTITION BY t.year, t.rcid) AS total_new_hires
            FROM revelio_transitions AS t
            JOIN matched_rcids mr ON t.rcid = mr.rcid
            JOIN revelio_inst_cw AS cw
              ON {sql_normalize_school_key('t.university_raw')} = cw.university_raw_norm
            WHERE t.n_transitions IS NOT NULL
              AND t.total_new_hires IS NOT NULL
              AND cw.unitid IS NOT NULL
              {exclude_clause}
              AND CAST(t.year AS INTEGER) BETWEEN {window_start} AND {window_end}
        ),
        transitions_all AS (
            SELECT
                CAST(t.rcid AS INTEGER) AS c,
                CAST(CAST(cw.unitid AS BIGINT) AS VARCHAR) AS k,
                CAST(t.year AS INTEGER) AS year,
                t.n_transitions,
                SUM(t.n_transitions) OVER (PARTITION BY t.year, t.rcid) AS total_new_hires
            FROM revelio_transitions AS t
            JOIN matched_rcids mr ON t.rcid = mr.rcid
            JOIN revelio_inst_cw AS cw
              ON {sql_normalize_school_key('t.university_raw')} = cw.university_raw_norm
            WHERE t.n_transitions IS NOT NULL
              AND cw.unitid IS NOT NULL
              AND t.year IS NOT NULL
              {exclude_clause}
        ),
        base_companies AS (
            SELECT
                c,
                SUM(total_new_hires_year) AS total_new_hires,
                SUM(total_new_hires_alt_year) AS total_new_hires_alt
            FROM (
                SELECT
                    c,
                    year,
                    MAX(total_new_hires) AS total_new_hires_year,
                    MAX(total_new_hires_alt) AS total_new_hires_alt_year
                FROM transitions_window
                GROUP BY c, year
            ) AS company_year_totals_window
            GROUP BY c
            HAVING SUM(total_new_hires_year) IS NOT NULL
        ),
        agg AS (
            SELECT
                c,
                k,
                SUM(n_transitions) AS n_transitions
            FROM transitions_window
            GROUP BY c, k
        ),
        company_year_totals_full AS (
            SELECT
                c,
                year,
                MAX(total_new_hires) AS total_new_hires_year
            FROM transitions_all
            GROUP BY c, year
        ),
        base_companies_full AS (
            SELECT
                c,
                SUM(total_new_hires_year) AS total_new_hires_full
            FROM company_year_totals_full
            GROUP BY c
        ),
        agg_full AS (
            SELECT
                c,
                k,
                SUM(n_transitions) AS n_transitions_full
            FROM transitions_all
            GROUP BY c, k
        ),
        pair_support AS (
            SELECT c, k FROM agg_full
            UNION
            SELECT c, k FROM agg
        )
        SELECT
            ps.c,
            ps.k,
            COALESCE(a.n_transitions, 0) AS n_transitions,
            COALESCE(af.n_transitions_full, 0) AS n_transitions_full,
            bc.total_new_hires,
            bc.total_new_hires_alt,
            bcf.total_new_hires_full,
            COALESCE(a.n_transitions, 0) / NULLIF(bc.total_new_hires_alt, 0) AS share_ck,
            COALESCE(a.n_transitions, 0) / NULLIF(bc.total_new_hires, 0) AS share_ck_base,
            COALESCE(af.n_transitions_full, 0) / NULLIF(bcf.total_new_hires_full, 0) AS share_ck_full
        FROM pair_support AS ps
        LEFT JOIN base_companies AS bc
          ON ps.c = bc.c
        LEFT JOIN agg AS a
          ON ps.c = a.c
         AND ps.k = a.k
        LEFT JOIN agg_full AS af
          ON ps.c = af.c
         AND ps.k = af.k
        LEFT JOIN base_companies_full AS bcf
          ON ps.c = bcf.c
        """
    )
    return "transition_unit_shares"


def _create_instrument_views(con: ddb.DuckDBPyConnection, shares_view: str, growth_view: str) -> tuple[str, str]:
    components = "company_instrument_components"
    panel = "company_instrument_panel"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {components} AS
        SELECT
            s.c,
            s.k,
            g.t,
            CASE WHEN s.share_ck > 1 THEN NULL ELSE s.share_ck END AS share_ck,
            CASE WHEN s.share_ck_base > 1 THEN NULL ELSE s.share_ck_base END AS share_ck_base,
            CASE WHEN s.share_ck_full > 1 THEN NULL ELSE s.share_ck_full END AS share_ck_full,
            s.n_transitions,
            s.n_transitions_full,
            s.total_new_hires,
            s.total_new_hires_alt,
            s.total_new_hires_full,
            g.g_kt,
            g.g_kt_all,
            g.g_kt_intl,
            CASE WHEN s.share_ck > 1 THEN NULL ELSE s.share_ck END * g.g_kt AS z_ct_component,
            CASE WHEN s.share_ck > 1 THEN NULL ELSE s.share_ck END * g.g_kt_all AS z_ct_all_component,
            CASE WHEN s.share_ck > 1 THEN NULL ELSE s.share_ck END * g.g_kt_intl AS z_ct_intl_component,
            CASE WHEN s.share_ck_full > 1 THEN NULL ELSE s.share_ck_full END * g.g_kt AS z_ct_full_component,
            CASE WHEN s.share_ck_full > 1 THEN NULL ELSE s.share_ck_full END * g.g_kt_all AS z_ct_all_full_component,
            CASE WHEN s.share_ck_full > 1 THEN NULL ELSE s.share_ck_full END * g.g_kt_intl AS z_ct_intl_full_component
        FROM {shares_view} AS s
        JOIN {growth_view} AS g
          ON s.k = g.k
        WHERE g.g_kt IS NOT NULL
        """
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW {panel} AS
        SELECT
            c,
            t,
            SUM(z_ct_component) AS z_ct,
            SUM(z_ct_all_component) AS z_ct_all,
            SUM(z_ct_intl_component) AS z_ct_intl,
            SUM(z_ct_full_component) AS z_ct_full,
            SUM(z_ct_all_full_component) AS z_ct_all_full,
            SUM(z_ct_intl_full_component) AS z_ct_intl_full,
            COUNT(
                DISTINCT CASE
                    WHEN share_ck IS NOT NULL AND share_ck <> 0
                     AND g_kt IS NOT NULL AND g_kt <> 0
                    THEN k
                    ELSE NULL
                END
            ) AS n_universities
        FROM {components}
        GROUP BY c, t
        """
    )
    return components, panel


def _sql_normalize(colname: str) -> str:
    return f"""
        TRIM(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    LOWER({colname}),
                    '[^a-z0-9 ]', ' ', 'g'
                ),
                '\\\\s+', ' ', 'g'
            )
        )
    """


def _sql_clean_company_name(companycol: str) -> str:
    suffix_regex = (
        "(?i)\\b("
        "inc|inc\\.|incorporated|llc|l\\.l\\.c|llp|l\\.l\\.p|lp|l\\.p|"
        "ltd|ltd\\.|limited|corp|corp\\.|corporation|company|co|co\\.|"
        "pllc|plc|pc|pc\\.|gmbh|ag|sa"
        ")\\b"
    )
    return _sql_normalize(f"REGEXP_REPLACE({companycol}, '{suffix_regex}', ' ', 'g')")


def _sql_state_name_to_abbr(statecol: str) -> str:
    mapping = {
        "alabama": "AL",
        "alaska": "AK",
        "arizona": "AZ",
        "arkansas": "AR",
        "california": "CA",
        "colorado": "CO",
        "connecticut": "CT",
        "delaware": "DE",
        "district of columbia": "DC",
        "washington dc": "DC",
        "dc": "DC",
        "florida": "FL",
        "georgia": "GA",
        "hawaii": "HI",
        "idaho": "ID",
        "illinois": "IL",
        "indiana": "IN",
        "iowa": "IA",
        "kansas": "KS",
        "kentucky": "KY",
        "louisiana": "LA",
        "maine": "ME",
        "maryland": "MD",
        "massachusetts": "MA",
        "michigan": "MI",
        "minnesota": "MN",
        "mississippi": "MS",
        "missouri": "MO",
        "montana": "MT",
        "nebraska": "NE",
        "nevada": "NV",
        "new hampshire": "NH",
        "new jersey": "NJ",
        "new mexico": "NM",
        "new york": "NY",
        "north carolina": "NC",
        "north dakota": "ND",
        "ohio": "OH",
        "oklahoma": "OK",
        "oregon": "OR",
        "pennsylvania": "PA",
        "rhode island": "RI",
        "south carolina": "SC",
        "south dakota": "SD",
        "tennessee": "TN",
        "texas": "TX",
        "utah": "UT",
        "vermont": "VT",
        "virginia": "VA",
        "washington": "WA",
        "west virginia": "WV",
        "wisconsin": "WI",
        "wyoming": "WY",
        "puerto rico": "PR",
        "guam": "GU",
        "american samoa": "AS",
        "northern mariana islands": "MP",
        "us virgin islands": "VI",
    }
    cases = "\n".join([f"WHEN LOWER(TRIM({statecol})) = '{name}' THEN '{abbr}'" for name, abbr in mapping.items()])
    return f"""
        CASE
            {cases}
            ELSE UPPER(TRIM({statecol}))
        END
    """


def _sql_clean_zip(zipcol: str) -> str:
    zipcolclean = f"TRIM(CAST(REGEXP_REPLACE({zipcol}, '[^0-9]', '', 'g') AS VARCHAR))"
    return f"""
        CASE
            WHEN LENGTH(TRIM(CAST({zipcolclean} AS VARCHAR))) = 4 THEN '0' || TRIM(CAST({zipcolclean} AS VARCHAR))
            WHEN LENGTH(TRIM(CAST({zipcolclean} AS VARCHAR))) >= 5 THEN SUBSTRING(TRIM(CAST({zipcolclean} AS VARCHAR)) FROM 1 FOR 5)
            ELSE TRIM(CAST({zipcolclean} AS VARCHAR))
        END
    """


def _date_parse_sql(col: str) -> str:
    return f"""
        COALESCE(
            TRY_CAST({col} AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%Y-%m-%d') AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%m/%d/%Y') AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%Y-%m-%d %H:%M:%S') AS DATE),
            TRY_CAST(try_strptime(CAST({col} AS VARCHAR), '%m/%d/%Y %H:%M:%S') AS DATE)
        )
    """


def _create_opt_counts(
    con: ddb.DuckDBPyConnection,
    outcome_lag_start: int,
    outcome_lag_end: int,
    use_changes: bool = False,
    include_non_masters: bool = False,
    include_bachelors: bool = False,
) -> str:
    # auth_start = f"COALESCE({_date_parse_sql('authorization_start_date')},{_date_parse_sql('opt_authorization_start_date')})"
    degree_clause = ""
    if include_bachelors and not include_non_masters:
        print(
            "[info] OPT counts are filtered to Master's + Bachelor's records. "
            "Output column names remain masters_opt_hires* for backward compatibility."
        )
    if not include_non_masters:
        predicate = _degree_predicate(con, view="foia_raw", include_bachelors=include_bachelors)
        if predicate:
            degree_clause = f"AND {predicate}"

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW foia_opt_authorizations_old AS
            SELECT person_id,
                {_sql_clean_company_name('employer_name')} AS f1_empname_clean,
                {_sql_normalize('employer_city')} AS f1_city_clean,
                {_sql_state_name_to_abbr('employer_state')} AS f1_state_clean,
                {_sql_clean_zip('employer_zip_code')} AS f1_zip_clean,
                MIN(EXTRACT(YEAR FROM program_end_date)) AS gradyear,
                MAX(CASE WHEN opt_employer_start_date >= program_end_date THEN 1 ELSE 0 END) AS valid_opt_hire
            FROM foia_raw_full
            WHERE employer_name IS NOT NULL {degree_clause}
            GROUP BY person_id, employer_name, employer_city, employer_state, employer_zip_code
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW person_post2014_correction_status AS
        SELECT
            f.person_id,
            MAX(
                CASE
                    WHEN f.year_int >= 2015 AND c.original_row_num IS NULL THEN 1
                    ELSE 0
                END
            ) AS has_post2014_correction
        FROM foia_raw_full AS f
        LEFT JOIN foia_raw AS c
          ON f.original_row_num = c.original_row_num
        WHERE f.person_id IS NOT NULL
        GROUP BY f.person_id
        """
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW foia_opt_authorizations_first_spell AS
        WITH base AS (
            SELECT
                person_id,
                {_sql_clean_company_name('employer_name')} AS f1_empname_clean,
                {_sql_normalize('employer_city')} AS f1_city_clean,
                {_sql_state_name_to_abbr('employer_state')} AS f1_state_clean,
                {_sql_clean_zip('employer_zip_code')} AS f1_zip_clean,
                EXTRACT(YEAR FROM program_end_date) AS gradyear,
                CASE WHEN opt_employer_start_date >= program_end_date THEN 1 ELSE 0 END AS valid_opt_hire,
                COALESCE(
                    {_date_parse_sql('opt_employer_start_date')},
                    {_date_parse_sql('opt_authorization_start_date')},
                    {_date_parse_sql('authorization_start_date')}
                ) AS spell_start_dt,
                original_row_num
            FROM foia_raw
            WHERE employer_name IS NOT NULL {degree_clause}
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY person_id, f1_empname_clean, f1_city_clean, f1_state_clean, f1_zip_clean
                    ORDER BY spell_start_dt ASC NULLS LAST, original_row_num ASC
                ) AS spell_rank
            FROM base
        )
        SELECT
            person_id,
            f1_empname_clean,
            f1_city_clean,
            f1_state_clean,
            f1_zip_clean,
            gradyear,
            valid_opt_hire
        FROM ranked
        WHERE spell_rank = 1
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW foia_opt_authorizations_correction_aware AS
        SELECT
            o.*,
            'old_fallback' AS correction_source
        FROM foia_opt_authorizations_old AS o
        LEFT JOIN person_post2014_correction_status AS pcs
          ON o.person_id = pcs.person_id
        WHERE COALESCE(pcs.has_post2014_correction, 0) = 0
        UNION ALL
        SELECT
            f.*,
            'first_spell' AS correction_source
        FROM foia_opt_authorizations_first_spell AS f
        JOIN person_post2014_correction_status AS pcs
          ON f.person_id = pcs.person_id
        WHERE pcs.has_post2014_correction = 1
        """
    )
    # con.sql(
    #     f"""
    #     CREATE OR REPLACE TEMP VIEW foia_opt_authorizations AS
    #     SELECT individual_key,
    #         {_sql_clean_company_name('employer_name')} AS f1_empname_clean,
    #         {_sql_normalize('employer_city')} AS f1_city_clean,
    #         {_sql_state_name_to_abbr('employer_state')} AS f1_state_clean,
    #         {_sql_clean_zip('employer_zip_code')} AS f1_zip_clean,
    #         {auth_start} AS auth_start
    #     FROM foia_raw
    #     WHERE employer_name IS NOT NULL AND EXTRACT(YEAR FROM {auth_start}) = year
    #       {degree_clause}
    #     """
    # )
    
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW opt_new_hires_old AS
        SELECT
            cw.preferred_rcid AS c,
            gradyear::INT AS t,
            COUNT(DISTINCT person_id) AS masters_opt_hires,
            COUNT(DISTINCT CASE WHEN valid_opt_hire = 1 THEN person_id END) AS valid_masters_opt_hires
        FROM foia_opt_authorizations_old AS f
        JOIN employer_crosswalk AS cw
          ON f.f1_empname_clean = cw.f1_empname_clean
         AND f.f1_city_clean = cw.f1_city_clean
         AND f.f1_state_clean = cw.f1_state_clean
         AND f.f1_zip_clean = cw.f1_zip_clean
        WHERE gradyear IS NOT NULL
          AND cw.preferred_rcid IS NOT NULL
        GROUP BY cw.preferred_rcid, gradyear::INT
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW opt_new_hires_correction_aware AS
        SELECT
            cw.preferred_rcid AS c,
            gradyear::INT AS t,
            COUNT(
                DISTINCT CASE
                    WHEN correction_source = 'old_fallback' AND valid_opt_hire = 1 THEN person_id
                    WHEN correction_source = 'first_spell' AND valid_opt_hire = 1 THEN person_id
                    ELSE NULL
                END
            ) AS masters_opt_hires_correction_aware
        FROM foia_opt_authorizations_correction_aware AS f
        JOIN employer_crosswalk AS cw
          ON f.f1_empname_clean = cw.f1_empname_clean
         AND f.f1_city_clean = cw.f1_city_clean
         AND f.f1_state_clean = cw.f1_state_clean
         AND f.f1_zip_clean = cw.f1_zip_clean
        WHERE gradyear IS NOT NULL
          AND cw.preferred_rcid IS NOT NULL
        GROUP BY cw.preferred_rcid, gradyear::INT
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW opt_new_hires_base AS
        SELECT
            COALESCE(o.c, n.c) AS c,
            COALESCE(o.t, n.t) AS t,
            COALESCE(o.masters_opt_hires, 0) AS masters_opt_hires,
            COALESCE(o.valid_masters_opt_hires, 0) AS valid_masters_opt_hires,
            COALESCE(n.masters_opt_hires_correction_aware, 0) AS masters_opt_hires_correction_aware
        FROM opt_new_hires_old AS o
        FULL OUTER JOIN opt_new_hires_correction_aware AS n
          ON o.c = n.c
         AND o.t = n.t
        """
    )
    lag_start = int(outcome_lag_start)
    lag_end = int(outcome_lag_end)
    x_lag_year_min = 2005
    x_lag_year_max = 2022
    x_lag_cols = ",\n            ".join(
        [
            (
                f"CASE WHEN (t + {lag}) < {x_lag_year_min} OR (t + {lag}) > {x_lag_year_max} "
                f"THEN NULL ELSE COALESCE(MAX(CASE WHEN lag = {lag} THEN x_lag END), 0) END "
                f"AS x_cst_lag{'m' + str(abs(lag)) if lag < 0 else str(lag)}"
            )
            for lag in range(lag_start, lag_end + 1)
        ]
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW opt_new_hires AS
        WITH base AS (
            SELECT
                c,
                t,
                COALESCE(masters_opt_hires, 0) AS masters_opt_hires,
                COALESCE(valid_masters_opt_hires, 0) AS valid_masters_opt_hires,
                COALESCE(masters_opt_hires_correction_aware, 0) AS masters_opt_hires_correction_aware
            FROM opt_new_hires_base
            WHERE c IS NOT NULL
              AND t IS NOT NULL
        ),
        bounds AS (
            SELECT
                c,
                MIN(t) AS min_t,
                MAX(t) AS max_t
            FROM base
            GROUP BY c
        ),
        expanded AS (
            SELECT
                b.c,
                gs.year AS t
            FROM bounds b,
            LATERAL generate_series(b.min_t, b.max_t) AS gs(year)
        ),
        filled AS (
            SELECT
                e.c,
                e.t,
                COALESCE(b.masters_opt_hires, 0) AS masters_opt_hires,
                COALESCE(b.valid_masters_opt_hires, 0) AS valid_masters_opt_hires,
                COALESCE(b.masters_opt_hires_correction_aware, 0) AS masters_opt_hires_correction_aware
            FROM expanded e
            LEFT JOIN base b
              ON e.c = b.c
             AND e.t = b.t
        ),
        long_lags AS (
            SELECT
                f.c,
                f.t,
                f.masters_opt_hires,
                f.valid_masters_opt_hires,
                f.masters_opt_hires_correction_aware,
                lag.lag AS lag,
                COALESCE(f2.masters_opt_hires_correction_aware, 0) AS x_lag
            FROM filled AS f
            CROSS JOIN LATERAL generate_series({lag_start}, {lag_end}) AS lag(lag)
            LEFT JOIN filled AS f2
              ON f.c = f2.c
             AND f2.t = f.t + lag.lag
        )
        SELECT
            c,
            t,
            MAX(masters_opt_hires) AS masters_opt_hires,
            MAX(valid_masters_opt_hires) AS valid_masters_opt_hires,
            MAX(masters_opt_hires_correction_aware) AS masters_opt_hires_correction_aware,
            {x_lag_cols}
        FROM long_lags
        GROUP BY c, t
        """
    )
    return "opt_new_hires"


def _create_outcome_views(
    con: ddb.DuckDBPyConnection,
    outcome_lag_start: int,
    outcome_lag_end: int,
    use_changes: bool = False,
) -> tuple[str, str]:
    """
    Map Revelio headcounts to hire-year t using an outcome-year lag range.
    If lag = 0, t == outcome year. If lag = 2, outcome_year = t + 2.
    Returns (outcomes_long_view, outcomes_wide_view).
    """
    lag_start = int(outcome_lag_start)
    lag_end = int(outcome_lag_end)

    def _lag_suffix(lag: int) -> str:
        return f"m{abs(lag)}" if lag < 0 else str(lag)

    y_expr = "y_cst"
    y_new_hires_expr = "y_new_hires"
    if use_changes:
        y_expr = "ASINH(y_cst) - ASINH(y_cst_lag)"
        y_new_hires_expr = "ASINH(y_new_hires) - ASINH(y_new_hires_lag)"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW outcomes_long AS
        WITH base_headcount_raw AS (
            SELECT
                CAST(rcid AS INTEGER) AS c,
                CAST(year AS INTEGER) AS outcome_year,
                CAST(total_headcount AS DOUBLE) AS y_cst
            FROM revelio_headcount
            WHERE rcid IN (SELECT rcid FROM matched_rcids)
              AND year IS NOT NULL
        ),
        base_headcount AS (
            SELECT
                c,
                outcome_year,
                COALESCE(MAX(y_cst), 0) AS y_cst
            FROM base_headcount_raw
            GROUP BY c, outcome_year
        ),
        base_new_hires_raw AS (
            SELECT
                CAST(rcid AS INTEGER) AS c,
                CAST(year AS INTEGER) AS outcome_year,
                CAST(total_new_hires AS DOUBLE) AS y_new_hires
            FROM revelio_transitions
            WHERE rcid IN (SELECT rcid FROM matched_rcids)
              AND year IS NOT NULL
              AND total_new_hires IS NOT NULL
        ),
        base_new_hires AS (
            SELECT
                c,
                outcome_year,
                COALESCE(MAX(y_new_hires), 0) AS y_new_hires
            FROM base_new_hires_raw
            GROUP BY c, outcome_year
        ),
        base_joined AS (
            SELECT
                COALESCE(h.c, n.c) AS c,
                COALESCE(h.outcome_year, n.outcome_year) AS outcome_year,
                COALESCE(h.y_cst, 0) AS y_cst,
                COALESCE(n.y_new_hires, 0) AS y_new_hires
            FROM base_headcount AS h
            FULL OUTER JOIN base_new_hires AS n
              ON h.c = n.c
             AND h.outcome_year = n.outcome_year
        ),
        bounds AS (
            SELECT
                c,
                MIN(outcome_year) AS min_year,
                MAX(outcome_year) AS max_year
            FROM base_joined
            GROUP BY c
        ),
        expanded AS (
            SELECT
                b.c,
                gs.year AS outcome_year
            FROM bounds b,
            LATERAL generate_series(b.min_year, b.max_year) AS gs(year)
        ),
        filled AS (
            SELECT
                e.c,
                e.outcome_year,
                COALESCE(b.y_cst, 0) AS y_cst,
                COALESCE(b.y_new_hires, 0) AS y_new_hires
            FROM expanded e
            LEFT JOIN base_joined b
              ON e.c = b.c
             AND e.outcome_year = b.outcome_year
        ),
        with_lags AS (
            SELECT
                c,
                outcome_year,
                y_cst,
                y_new_hires,
                LAG(y_cst) OVER (PARTITION BY c ORDER BY outcome_year) AS y_cst_lag,
                LAG(y_new_hires) OVER (PARTITION BY c ORDER BY outcome_year) AS y_new_hires_lag
            FROM filled
        )
        SELECT
            b.c,
            b.outcome_year AS s,
            b.outcome_year - lag.lag AS t,
            {y_expr} AS y_cst,
            {y_new_hires_expr} AS y_new_hires,
            lag.lag AS lag
        FROM with_lags AS b
        CROSS JOIN LATERAL generate_series({lag_start}, {lag_end}) AS lag(lag)
        """
    )

    outcome_cols_y = ",\n            ".join(
        [
            f"COALESCE(MAX(CASE WHEN lag = {lag} THEN y_cst END), 0) AS y_cst_lag{_lag_suffix(lag)}"
            for lag in range(lag_start, lag_end + 1)
        ]
    )
    outcome_cols_y_new_hires = ",\n            ".join(
        [
            f"COALESCE(MAX(CASE WHEN lag = {lag} THEN y_new_hires END), 0) AS y_new_hires_lag{_lag_suffix(lag)}"
            for lag in range(lag_start, lag_end + 1)
        ]
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW outcomes_wide AS
        SELECT
            c,
            t,
            {outcome_cols_y},
            {outcome_cols_y_new_hires}
        FROM outcomes_long
        WHERE t IS NOT NULL
        GROUP BY c, t
        """
    )
    return "outcomes_long", "outcomes_wide"


def _create_analysis_panel(
    con: ddb.DuckDBPyConnection,
    outcome_lag_start: int,
    outcome_lag_end: int,
    use_log_y: bool = False,
) -> str:
    def _lag_suffix(lag: int) -> str:
        return f"m{abs(lag)}" if lag < 0 else str(lag)

    lag_range = range(int(outcome_lag_start), int(outcome_lag_end) + 1)
    x_lag_year_min = 2005
    x_lag_year_max = 2022

    outcome_cols = []
    outcome_cols_new_hires = []
    x_lag_cols = []
    for lag in lag_range:
        suffix = _lag_suffix(lag)
        if use_log_y:
            outcome_cols.append(f"ASINH(o.y_cst_lag{suffix}) AS y_cst_lag{suffix}")
            outcome_cols_new_hires.append(f"ASINH(o.y_new_hires_lag{suffix}) AS y_new_hires_lag{suffix}")
        else:
            outcome_cols.append(f"o.y_cst_lag{suffix}")
            outcome_cols_new_hires.append(f"o.y_new_hires_lag{suffix}")
        x_lag_cols.append(
            f"CASE WHEN (o.t + {lag}) < {x_lag_year_min} OR (o.t + {lag}) > {x_lag_year_max} "
            f"THEN NULL ELSE COALESCE(x.x_cst_lag{suffix}, 0) END AS x_cst_lag{suffix}"
        )
    outcome_cols = ",\n            ".join(outcome_cols)
    outcome_cols_new_hires = ",\n            ".join(outcome_cols_new_hires)
    x_lag_cols = ",\n            ".join(x_lag_cols)

    x_expr = "COALESCE(x.masters_opt_hires, 0)"
    x_valid_expr = "COALESCE(x.valid_masters_opt_hires, 0)"
    x_corrected_expr = "COALESCE(x.masters_opt_hires_correction_aware, 0)"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW analysis_panel AS
        SELECT
            o.c,
            o.t,
            {outcome_cols},
            {outcome_cols_new_hires},
            {x_expr} AS masters_opt_hires,
            {x_valid_expr} AS valid_masters_opt_hires,
            {x_corrected_expr} AS masters_opt_hires_correction_aware,
            {x_lag_cols},
            instr.z_ct,
            instr.z_ct_all,
            instr.z_ct_intl,
            instr.z_ct_full,
            instr.z_ct_all_full,
            instr.z_ct_intl_full,
            instr.n_universities,
            st.company_state
        FROM (SELECT * FROM outcomes_wide WHERE t > 2005 AND t < 2025) AS o
        LEFT JOIN (SELECT * FROM opt_new_hires WHERE t > 2005 AND t < 2025) AS x USING (c, t)
        LEFT JOIN (SELECT * FROM company_instrument_panel WHERE t > 2005 AND t < 2025) AS instr USING (c, t)
        LEFT JOIN (
            WITH state_counts AS (
                SELECT
                    CAST(preferred_rcid AS INTEGER) AS c,
                    UPPER(TRIM(CAST(f1_state_clean AS VARCHAR))) AS company_state,
                    COUNT(*) AS n_rows
                FROM employer_crosswalk
                WHERE preferred_rcid IS NOT NULL
                  AND f1_state_clean IS NOT NULL
                  AND TRIM(CAST(f1_state_clean AS VARCHAR)) <> ''
                GROUP BY 1, 2
            ),
            ranked AS (
                SELECT
                    c,
                    company_state,
                    ROW_NUMBER() OVER (PARTITION BY c ORDER BY n_rows DESC, company_state ASC) AS rn
                FROM state_counts
            )
            SELECT c, company_state
            FROM ranked
            WHERE rn = 1
        ) AS st USING (c)
        WHERE o.t IS NOT NULL
        """
    )
    return "analysis_panel"


def _plot_f1_yearly_descriptives(
    con: ddb.DuckDBPyConnection,
    *,
    plot: bool = True,
    include_non_masters: bool = False,
    include_bachelors: bool = False,
) -> None:
    # Match the degree filter used in treatment construction.
    degree_clause = ""
    if not include_non_masters:
        degree_pred = _degree_predicate(con, view="foia_raw", include_bachelors=include_bachelors)
        if degree_pred:
            degree_clause = f"AND ({degree_pred})"

    hires_by_year = con.sql(
        """
        SELECT
            CAST(t AS INTEGER) AS year,
            SUM(masters_opt_hires_correction_aware) AS total_opt_hires,
            AVG(masters_opt_hires_correction_aware) AS avg_opt_hires_per_firm,
            COUNT(DISTINCT c) AS n_hiring_firms
        FROM opt_new_hires_correction_aware
        GROUP BY t
        ORDER BY t
        """
    ).df()

    start_col = _first_present_column(
        con,
        "foia_raw",
        ["program_start_date", "program_begin_date", "program_begin_dt", "program_start_dt"],
    )
    tuition_col = _first_present_column(
        con,
        "foia_raw",
        ["tuition__fees", "tuition_fees", "tuition", "tuition_fees_usd"],
    )

    tuition_by_start_year = None
    if start_col and tuition_col:
        start_expr = _date_parse_sql(start_col)
        tuition_by_start_year = con.sql(
            f"""
            SELECT
                CAST(EXTRACT(YEAR FROM {start_expr}) AS INTEGER) AS program_start_year,
                AVG(TRY_CAST({tuition_col} AS DOUBLE)) AS avg_tuition,
                COUNT(*) AS n_records
            FROM foia_raw
            WHERE {start_expr} IS NOT NULL
              AND TRY_CAST({tuition_col} AS DOUBLE) IS NOT NULL
              {degree_clause}
            GROUP BY 1
            ORDER BY 1
            """
        ).df()
    else:
        missing = []
        if not start_col:
            missing.append("program_start_date/program_begin_date")
        if not tuition_col:
            missing.append("tuition")
        print(
            "[warn] Skipping tuition-by-program-start-year plot; missing column(s): "
            + ", ".join(missing)
        )

    print("\n[f1_opt_hires_by_year]")
    print(hires_by_year)
    if tuition_by_start_year is not None:
        print("\n[f1_avg_tuition_by_program_start_year]")
        print(tuition_by_start_year)

    if not plot:
        return

    import matplotlib.pyplot as plt

    if not hires_by_year.empty:
        hires_plot = hires_by_year.sort_values("year")

        plt.figure(figsize=(9, 4.5))
        plt.plot(hires_plot["year"], hires_plot["total_opt_hires"], marker="o", color="tab:blue")
        plt.xlabel("Year")
        plt.ylabel("Total OPT hires")
        plt.title("F1 data: total OPT hires by year")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(9, 4.5))
        plt.plot(hires_plot["year"], hires_plot["avg_opt_hires_per_firm"], marker="o", color="tab:orange")
        plt.xlabel("Year")
        plt.ylabel("Avg OPT hires per hiring firm-year")
        plt.title("F1 data: average OPT hires per firm by year")
        plt.tight_layout()
        plt.show()
    else:
        print("[warn] No rows in opt_new_hires_correction_aware; skipping OPT-hire year plots.")

    if tuition_by_start_year is not None and not tuition_by_start_year.empty:
        tuition_plot = tuition_by_start_year.sort_values("program_start_year")
        plt.figure(figsize=(9, 4.5))
        plt.plot(tuition_plot["program_start_year"], tuition_plot["avg_tuition"], marker="o", color="tab:green")
        plt.xlabel("Program start year")
        plt.ylabel("Average tuition")
        plt.title("F1 data: average tuition by program start year")
        plt.tight_layout()
        plt.show()
    else:
        print("[warn] No tuition rows after filters; skipping tuition-by-start-year plot.")


def run_diagnostics(
    con: ddb.DuckDBPyConnection,
    plot: bool = True,
    include_non_masters: bool = False,
    include_bachelors: bool = False,
    opt_shifts: bool = False,
    school_sample_mode: str = "all",
    school_shift_metric: str | None = None,
    opt_shifts_degree_scope: str = "bachelors_masters",
    opt_shifts_normalization: str | None = None,
    opt_shifts_normalize_by_graduates: bool = True,
) -> None:
    """
    Run some basic diagnostics on the built outputs.
    """
    opt_shifts_degree_scope = _normalize_degree_scope(opt_shifts_degree_scope)
    if opt_shifts_normalization is None:
        opt_shifts_normalization = (
            "ipeds_graduates" if opt_shifts_normalize_by_graduates else "none"
        )
    opt_shifts_normalization = _normalize_opt_shift_normalization(opt_shifts_normalization)
    school_sample_mode = _normalize_school_sample_mode(school_sample_mode)
    active_metric = _resolve_active_shift_metric(
        school_sample_mode=school_sample_mode,
        school_shift_metric=school_shift_metric,
        opt_shifts=opt_shifts,
    )
    print(
        "[diagnostics] shift_source="
        f"{'matched_sample' if school_sample_mode == 'matched_shift_sample' else ('foia_opt' if opt_shifts else 'ipeds')}; "
        f"active_metric={active_metric}; "
        f"opt_shifts_degree_scope={opt_shifts_degree_scope}; "
        f"opt_shifts_normalization={opt_shifts_normalization}; "
        f"include_bachelors={include_bachelors}; "
        f"include_non_masters={include_non_masters}"
    )

    # Shifts
    print("Running diagnostics on shifts...")
    print(f"---Number of unique universities: {con.sql('SELECT COUNT(DISTINCT k) FROM ipeds_unit_growth').fetchone()[0]}")
    print(f"---Year range: {con.sql('SELECT MIN(t), MAX(t) FROM ipeds_unit_growth').fetchone()}")
    print(f"---Average shifts (g_kt, g_kt_all, g_kt_intl) across all k in 2015: {con.sql('SELECT AVG(g_kt), AVG(g_kt_all), AVG(g_kt_intl) FROM ipeds_unit_growth WHERE t = 2015').fetchone()}")
    if school_sample_mode == "matched_shift_sample":
        n_treated = con.sql(
            "SELECT COUNT(*) FROM school_shift_sample WHERE sample_role = 'treated'"
        ).fetchone()[0]
        n_control = con.sql(
            "SELECT COUNT(*) FROM school_shift_sample WHERE sample_role = 'control'"
        ).fetchone()[0]
        print(
            "---Selected treated/control schools: "
            f"{n_treated} treated, {n_control} control"
        )
    if opt_shifts or _school_metric_is_opt_family(active_metric):
        print(
            "---Share of school-years where g_kt = g_kt_all = g_kt_intl: "
            f"{con.sql('SELECT AVG(CASE WHEN g_kt = g_kt_all AND g_kt = g_kt_intl THEN 1.0 ELSE 0.0 END) FROM ipeds_unit_growth').fetchone()[0]}"
        )
    
    if plot:
        sns.histplot(con.sql('SELECT g_kt AS "g_kt in 2015" FROM ipeds_unit_growth WHERE t = 2015 AND g_kt < 1000').df()['g_kt in 2015'], bins=50)
    
    # Shares
    print("Running diagnostics on shares...")
    print(f"---Number of companies with shares: {con.sql('SELECT COUNT(DISTINCT c) FROM transition_unit_shares').fetchone()[0]}")
    print(f"---Number of universities with shares: {con.sql('SELECT COUNT(DISTINCT k) FROM transition_unit_shares').fetchone()[0]}")
    print(f"---Mean of average company share (share_ck, share_ck_full): {con.sql('SELECT AVG(share_ck), AVG(share_ck_full) FROM (SELECT AVG(share_ck) AS share_ck, AVG(share_ck_full) AS share_ck_full FROM transition_unit_shares GROUP BY c)').fetchone()}")
    
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        sns.histplot(con.sql('SELECT share_ck AS "Average share_ck by company" FROM (SELECT AVG(share_ck) AS share_ck, AVG(share_ck_full) AS share_ck_full FROM transition_unit_shares GROUP BY c) WHERE share_ck <= 1').df()['Average share_ck by company'], bins=50)
        
    # Instruments
    print("Running diagnostics on instruments...")
    print(f"---Number of universities contributing to instruments: {con.sql('SELECT COUNT(DISTINCT k) FROM company_instrument_components').fetchone()[0]}")
    print(f"---Number of companies with instruments: {con.sql('SELECT COUNT(DISTINCT c) FROM company_instrument_panel').fetchone()[0]}")
    print(f"Year range of instruments: {con.sql('SELECT MIN(t), MAX(t) FROM company_instrument_panel').fetchone()}")
    print(f"---Mean z_ct, z_ct_all, z_ct_intl, z_ct_full, z_ct_all_full, z_ct_intl_full: {con.sql('SELECT AVG(z_ct), AVG(z_ct_all), AVG(z_ct_intl), AVG(z_ct_full), AVG(z_ct_all_full), AVG(z_ct_intl_full) FROM company_instrument_panel').fetchone()}")
    
    # Independent Variable
    print("Running diagnostics on independent variable (treatment)...")
    print(f"---Number of companies with Master's OPT hires: {con.sql('SELECT COUNT(DISTINCT c) FROM opt_new_hires').fetchone()[0]}")
    print(f"---Number of company-year observations with Master's OPT hires: {con.sql('SELECT COUNT(*) FROM opt_new_hires').fetchone()[0]}")
    print(f"---Year range of Master's OPT hires: {con.sql('SELECT MIN(t), MAX(t) FROM opt_new_hires').fetchone()}")
    print(f"---Mean Master's OPT hires (hires, valid hires) per company-year: {con.sql('SELECT AVG(masters_opt_hires), AVG(valid_masters_opt_hires) FROM opt_new_hires').fetchone()}")
    
    # Dependent Variable
    print("Running diagnostics on dependent variable (outcomes)...")
    print(f"---Number of companies with outcomes: {con.sql('SELECT COUNT(DISTINCT c) FROM outcomes_long').fetchone()[0]}")
    print(f"---Number of company-year observations with outcomes: {con.sql('SELECT COUNT(*) FROM outcomes_long').fetchone()[0]}")
    print(f"---Year range of outcomes: {con.sql('SELECT MIN(s), MAX(s) FROM outcomes_long').fetchone()}")
    lag_cols = con.sql("PRAGMA table_info('outcomes_wide')").fetchall()
    lag_col_names = [r[1] for r in lag_cols if r[1].startswith("y_cst_lag0")]
    for col in lag_col_names:
        print(f"---Mean outcome for {col}: {con.sql(f'SELECT AVG({col}) FROM outcomes_wide WHERE t < 2020 AND t > 2010').fetchone()[0]}")
    
    # Analysis Panel
    print("Running diagnostics on analysis panel...")
    print(f"---Number of companies in analysis panel: {con.sql('SELECT COUNT(DISTINCT c) FROM analysis_panel WHERE z_ct IS NOT NULL AND masters_opt_hires IS NOT NULL').fetchone()[0]}")
    print(f"---Number of company-year observations in analysis panel: {con.sql('SELECT COUNT(*) FROM analysis_panel WHERE z_ct IS NOT NULL AND masters_opt_hires IS NOT NULL').fetchone()[0]}")
    print(f"---Year range of analysis panel: {con.sql('SELECT MIN(t), MAX(t) FROM analysis_panel').fetchone()}")
    
    # First Stage
    print("Running first-stage diagnostics...")
    fs = con.sql("SELECT z_ct, z_ct_all, z_ct_intl, masters_opt_hires, valid_masters_opt_hires, c, t FROM analysis_panel WHERE z_ct IS NOT NULL AND masters_opt_hires IS NOT NULL").df()
    # overlapping histograms of z_ct and masters_opt_hires
    if plot:
        plt.figure()
        sns.histplot(fs['z_ct'], bins=50)
        sns.histplot(fs['masters_opt_hires'], bins=50)
    
    # correlation matrix between z_ct, z_ct_all, z_ct_intl and masters_opt_hires, valid_masters_opt_hires
    corr_matrix = fs[['z_ct', 'z_ct_all', 'z_ct_intl', 'masters_opt_hires', 'valid_masters_opt_hires']].corr()
    print(f"---Correlation matrix:\n{corr_matrix}")
    
    # binned scatterplot
    if plot:
        plt.figure()
        sns.regplot(x='z_ct_intl', y='valid_masters_opt_hires', data=fs, lowess=True, scatter_kws={'s':10}, line_kws={'color':'red'})

    _plot_f1_yearly_descriptives(
        con,
        plot=plot,
        include_non_masters=include_non_masters,
        include_bachelors=include_bachelors,
    )

def build_pipeline(
    paths: PipelinePaths,
    *,
    outcome_lag_start: int = 0,
    outcome_lag_end: int = 5,
    share_base_year: int = 2010,
    save_outputs: bool = True,
    verbose: bool = True,
    plot_diagnostics: bool = True,
    use_changes: bool = False,
    use_log_y: bool = False,
    include_non_masters: bool = False,
    include_bachelors: bool = False,
    opt_shifts: bool = False,
    school_sample_mode: str = "all",
    school_shift_metric: str | None = None,
    school_sample_n_shifted: int = 25,
    school_sample_window_start: int = 2014,
    school_sample_window_end: int = 2017,
    school_sample_control_positive_cap: float = 0.02,
    school_sample_min_size: int = 100,
    opt_ihmp_ipeds_share_intl_threshold: float = 0.30,
    opt_ihmp_foia_opt_share_threshold: float = 0.50,
    opt_ihmp_min_program_f1_count: int = 10,
    opt_share_min_school_f1_count: int = 50,
    opt_share_max_yoy_drop: float = 0.50,
    restrict_treated_to_no_large_enrollment_jump: bool = False,
    school_sample_max_yoy_size_jump: float = 0.50,
    confirm_matched_school_sample: bool = False,
    opt_shifts_degree_scope: str = "bachelors_masters",
    opt_shifts_normalization: str | None = None,
    opt_shifts_normalize_by_graduates: bool = True,
    exclude_unitids: list[str] | None = None,
) -> None:
    con = ddb.connect()
    school_sample_mode = _normalize_school_sample_mode(school_sample_mode)
    opt_shifts_degree_scope = _normalize_degree_scope(opt_shifts_degree_scope)
    if opt_shifts_normalization is None:
        opt_shifts_normalization = (
            "ipeds_graduates" if opt_shifts_normalize_by_graduates else "none"
        )
    opt_shifts_normalization = _normalize_opt_shift_normalization(opt_shifts_normalization)
    _register_inputs(
        con,
        paths,
        include_bachelors=include_bachelors,
        opt_shifts=opt_shifts,
        school_sample_mode=school_sample_mode,
        school_shift_metric=school_shift_metric,
        opt_shift_degree_scope=opt_shifts_degree_scope,
        opt_shifts_normalization=opt_shifts_normalization,
        opt_shifts_normalize_by_graduates=opt_shifts_normalize_by_graduates,
    )

    if use_changes and use_log_y:
        raise ValueError("use_changes and use_log_y are mutually exclusive.")
    if include_non_masters and include_bachelors:
        print("[info] include_non_masters=true takes precedence; include_bachelors flag is ignored.")

    growth_view, school_metric_panel_view, school_shift_sample_view = _create_growth_view_for_pipeline(
        con,
        school_sample_mode=school_sample_mode,
        school_shift_metric=school_shift_metric,
        include_bachelors=include_bachelors,
        use_changes=use_changes,
        use_log_y=use_log_y,
        opt_shifts=opt_shifts,
        opt_shifts_degree_scope=opt_shifts_degree_scope,
        opt_shifts_normalization=opt_shifts_normalization,
        opt_shifts_normalize_by_graduates=opt_shifts_normalize_by_graduates,
        exclude_unitids=exclude_unitids,
        school_sample_n_shifted=school_sample_n_shifted,
        school_sample_window_start=school_sample_window_start,
        school_sample_window_end=school_sample_window_end,
        school_sample_control_positive_cap=school_sample_control_positive_cap,
        school_sample_min_size=school_sample_min_size,
        opt_ihmp_ipeds_share_intl_threshold=opt_ihmp_ipeds_share_intl_threshold,
        opt_ihmp_foia_opt_share_threshold=opt_ihmp_foia_opt_share_threshold,
        opt_ihmp_min_program_f1_count=opt_ihmp_min_program_f1_count,
        opt_share_min_school_f1_count=opt_share_min_school_f1_count,
        opt_share_max_yoy_drop=opt_share_max_yoy_drop,
        restrict_treated_to_no_large_enrollment_jump=restrict_treated_to_no_large_enrollment_jump,
        school_sample_max_yoy_size_jump=school_sample_max_yoy_size_jump,
        confirm_matched_school_sample=confirm_matched_school_sample,
    )
    shares_view = _create_transition_share_view(con, share_base_year, exclude_unitids=exclude_unitids)
    _create_instrument_views(con, shares_view, growth_view)
    if outcome_lag_end < outcome_lag_start:
        raise ValueError("outcome_lag_end must be >= outcome_lag_start.")
    _create_outcome_views(con, outcome_lag_start, outcome_lag_end, use_changes=use_changes)
    _create_opt_counts(
        con,
        outcome_lag_start=outcome_lag_start,
        outcome_lag_end=outcome_lag_end,
        use_changes=use_changes,
        include_non_masters=include_non_masters,
        include_bachelors=include_bachelors,
    )
    _create_analysis_panel(con, outcome_lag_start, outcome_lag_end, use_log_y=use_log_y)

    if not save_outputs:
        print("Skipping writes (save_outputs=False).")
        return

    _write_view(con, "company_instrument_components", paths.instrument_components_out)
    _write_view(con, "company_instrument_panel", paths.instrument_panel_out)
    _write_view(con, "opt_new_hires", paths.treatment_out)
    _write_view(con, "outcomes_long", paths.outcomes_out)
    # Keep default output behavior; optionally also write to a dedicated MA+BA panel path.
    _write_view(con, "analysis_panel", paths.analysis_panel_out)
    if include_bachelors and paths.analysis_panel_out_ma_ba is not None:
        _write_view(con, "analysis_panel", paths.analysis_panel_out_ma_ba)
    if school_metric_panel_view is not None and paths.school_shift_metric_panel_out is not None:
        _write_view(con, school_metric_panel_view, paths.school_shift_metric_panel_out)
    if school_shift_sample_view is not None and paths.school_shift_sample_out is not None:
        _write_view(con, school_shift_sample_view, paths.school_shift_sample_out)
    
    if verbose:
        run_diagnostics(
            con,
            plot=plot_diagnostics,
            include_non_masters=include_non_masters,
            include_bachelors=include_bachelors,
            opt_shifts=opt_shifts,
            school_sample_mode=school_sample_mode,
            school_shift_metric=school_shift_metric,
            opt_shifts_degree_scope=opt_shifts_degree_scope,
            opt_shifts_normalization=opt_shifts_normalization,
            opt_shifts_normalize_by_graduates=opt_shifts_normalize_by_graduates,
        )
        


def _parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build company-level shift-share pipeline outputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--outcome-lag",
        type=int,
        default=None,
        help="Outcome year minus hire year (overrides lag range when set).",
    )
    parser.add_argument(
        "--share-base-year",
        type=int,
        default=None,
        help="Base year to compute company-university shares (default: 2010).",
    )
    parser.add_argument(
        "--outcome-lag-start",
        type=int,
        default=None,
        help="Minimum outcome lag (inclusive).",
    )
    parser.add_argument(
        "--outcome-lag-end",
        type=int,
        default=None,
        help="Maximum outcome lag (inclusive).",
    )
    parser.add_argument(
        "--include-non-masters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use all FOIA records (skip master's-only filter).",
    )
    parser.add_argument(
        "--include-bachelors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Master's + Bachelor's records (instead of Master's-only) in IPEDS/FOIA filters.",
    )
    parser.add_argument(
        "--no-write",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip writing parquet outputs (useful for quick checks).",
    )
    parser.add_argument(
        "--use-changes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use IHS differences for g_kt and outcomes: asinh(x) - asinh(lag(x)).",
    )
    parser.add_argument(
        "--use-log-y",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Transform outcomes with inverse hyperbolic sine: y=asinh(y).",
    )
    parser.add_argument(
        "--opt-shifts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Build g_kt from FOIA OPT usage.",
    )
    parser.add_argument(
        "--opt-shifts-normalize-by-graduates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When --opt-shifts is enabled, normalize school-year OPT-user counts "
            "by IPEDS graduates."
        ),
    )
    parser.add_argument(
        "--opt-shifts-normalization",
        type=str,
        default=None,
        help=(
            "Normalization mode when --opt-shifts is enabled. "
            "Options: ipeds_graduates (default), foia_students, none."
        ),
    )
    parser.add_argument(
        "--opt-shifts-degree-scope",
        type=str,
        default=None,
        help=(
            "Degree scope for FOIA/IPEDS when --opt-shifts is enabled. "
            "Options: bachelors_masters (default), bachelors, masters."
        ),
    )
    parser.add_argument(
        "--exclude-unitids",
        default="",
        help="Comma-separated UNITIDs to exclude from both growth and transition shares.",
    )
    parser.add_argument(
        "--confirm-matched-school-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When school_sample_mode=matched_shift_sample, print the selected treated/control "
            "schools and wait for confirmation before continuing."
        ),
    )
    if args is None:
        # In IPython/Jupyter, sys.argv often contains kernel args like "-f ...".
        # Ignore unknown args only in that interactive context so main() is usable.
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
            parsed, unknown = parser.parse_known_args()
            if unknown:
                print(f"[info] Ignoring unknown IPython args: {unknown}")
            return parsed
    return parser.parse_args(args)

def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(cli_args)
    cfg = load_config(args.config)
    paths = _resolve_pipeline_paths(cfg)
    pipeline_cfg = get_cfg_section(cfg, "build_company_shift_share")

    share_base_year = args.share_base_year if args.share_base_year is not None else pipeline_cfg.get("share_base_year", 2010)
    lag_start_default = pipeline_cfg.get("outcome_lag_start", 0)
    lag_end_default = pipeline_cfg.get("outcome_lag_end", 5)
    include_non_masters = (
        args.include_non_masters
        if args.include_non_masters is not None
        else pipeline_cfg.get("include_non_masters", False)
    )
    include_bachelors = (
        args.include_bachelors
        if args.include_bachelors is not None
        else pipeline_cfg.get("include_bachelors", False)
    )
    save_outputs = (
        not args.no_write
        if args.no_write is not None
        else pipeline_cfg.get("save_outputs", True)
    )
    use_changes = args.use_changes if args.use_changes is not None else pipeline_cfg.get("use_changes", False)
    use_log_y = args.use_log_y if args.use_log_y is not None else pipeline_cfg.get("use_log_y", pipeline_cfg.get("use_log_rate", False))
    opt_shifts = args.opt_shifts if args.opt_shifts is not None else pipeline_cfg.get("opt_shifts", False)
    school_sample_mode = _normalize_school_sample_mode(
        pipeline_cfg.get("school_sample_mode", "all")
    )
    school_shift_metric_raw = pipeline_cfg.get("school_shift_metric")
    if school_shift_metric_raw is None and school_sample_mode == "matched_shift_sample":
        school_shift_metric_raw = "opt_share" if opt_shifts else "ihmp_share"
    school_shift_metric = (
        _normalize_school_shift_metric(school_shift_metric_raw)
        if school_shift_metric_raw is not None
        else None
    )
    school_sample_n_shifted = int(pipeline_cfg.get("school_sample_n_shifted", 25))
    school_sample_window_start = int(pipeline_cfg.get("school_sample_window_start", 2014))
    school_sample_window_end = int(pipeline_cfg.get("school_sample_window_end", 2017))
    school_sample_control_positive_cap = float(
        pipeline_cfg.get("school_sample_control_positive_cap", 0.02)
    )
    school_sample_min_size = int(
        pipeline_cfg.get("school_sample_min_size", 100)
    )
    opt_ihmp_ipeds_share_intl_threshold = float(
        pipeline_cfg.get("opt_ihmp_ipeds_share_intl_threshold", 0.30)
    )
    opt_ihmp_foia_opt_share_threshold = float(
        pipeline_cfg.get("opt_ihmp_foia_opt_share_threshold", 0.50)
    )
    opt_ihmp_min_program_f1_count = int(
        pipeline_cfg.get("opt_ihmp_min_program_f1_count", 10)
    )
    opt_share_min_school_f1_count = int(
        pipeline_cfg.get("opt_share_min_school_f1_count", 50)
    )
    opt_share_max_yoy_drop = float(
        pipeline_cfg.get("opt_share_max_yoy_drop", 0.50)
    )
    restrict_treated_to_no_large_enrollment_jump = bool(
        pipeline_cfg.get("restrict_treated_to_no_large_enrollment_jump", False)
    )
    school_sample_max_yoy_size_jump = float(
        pipeline_cfg.get("school_sample_max_yoy_size_jump", 0.50)
    )
    confirm_matched_school_sample = (
        args.confirm_matched_school_sample
        if args.confirm_matched_school_sample is not None
        else pipeline_cfg.get("confirm_matched_school_sample", False)
    )
    opt_shifts_degree_scope = _normalize_degree_scope(
        args.opt_shifts_degree_scope
        if args.opt_shifts_degree_scope is not None
        else pipeline_cfg.get("opt_shifts_degree_scope", "bachelors_masters")
    )
    opt_shifts_normalize_by_graduates = (
        args.opt_shifts_normalize_by_graduates
        if args.opt_shifts_normalize_by_graduates is not None
        else pipeline_cfg.get("opt_shifts_normalize_by_graduates", True)
    )
    opt_shifts_normalization_raw = (
        args.opt_shifts_normalization
        if args.opt_shifts_normalization is not None
        else pipeline_cfg.get("opt_shifts_normalization")
    )
    if opt_shifts_normalization_raw is None:
        opt_shifts_normalization = (
            "ipeds_graduates" if opt_shifts_normalize_by_graduates else "none"
        )
    else:
        opt_shifts_normalization = _normalize_opt_shift_normalization(
            opt_shifts_normalization_raw
        )
    exclude_unitids = _parse_unitids(args.exclude_unitids) if args.exclude_unitids else pipeline_cfg.get("exclude_unitids", [])
    if args.outcome_lag is not None:
        outcome_lag_start = args.outcome_lag
        outcome_lag_end = args.outcome_lag
    else:
        outcome_lag_start = args.outcome_lag_start if args.outcome_lag_start is not None else lag_start_default
        outcome_lag_end = args.outcome_lag_end if args.outcome_lag_end is not None else lag_end_default
    build_pipeline(
        paths=paths,
        outcome_lag_start=outcome_lag_start,
        outcome_lag_end=outcome_lag_end,
        share_base_year=share_base_year,
        save_outputs=save_outputs,
        verbose=pipeline_cfg.get("verbose", True),
        plot_diagnostics=pipeline_cfg.get("plot_diagnostics", True),
        use_changes=use_changes,
        use_log_y=use_log_y,
        include_non_masters=include_non_masters,
        include_bachelors=include_bachelors,
        opt_shifts=opt_shifts,
        school_sample_mode=school_sample_mode,
        school_shift_metric=school_shift_metric,
        school_sample_n_shifted=school_sample_n_shifted,
        school_sample_window_start=school_sample_window_start,
        school_sample_window_end=school_sample_window_end,
        school_sample_control_positive_cap=school_sample_control_positive_cap,
        school_sample_min_size=school_sample_min_size,
        opt_ihmp_ipeds_share_intl_threshold=opt_ihmp_ipeds_share_intl_threshold,
        opt_ihmp_foia_opt_share_threshold=opt_ihmp_foia_opt_share_threshold,
        opt_ihmp_min_program_f1_count=opt_ihmp_min_program_f1_count,
        opt_share_min_school_f1_count=opt_share_min_school_f1_count,
        opt_share_max_yoy_drop=opt_share_max_yoy_drop,
        restrict_treated_to_no_large_enrollment_jump=restrict_treated_to_no_large_enrollment_jump,
        school_sample_max_yoy_size_jump=school_sample_max_yoy_size_jump,
        confirm_matched_school_sample=confirm_matched_school_sample,
        opt_shifts_degree_scope=opt_shifts_degree_scope,
        opt_shifts_normalization=opt_shifts_normalization,
        opt_shifts_normalize_by_graduates=opt_shifts_normalize_by_graduates,
        exclude_unitids=exclude_unitids,
    )


if __name__ == "__main__":
    main()
