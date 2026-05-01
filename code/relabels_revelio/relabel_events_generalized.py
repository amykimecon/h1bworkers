"""Generalized relabel-event detector for relabels_revelio.

This standalone script expands the econ-only IPEDS/FOIA relabel workflow to:

1. scan IPEDS for same-degree CIP relabels across all CIPs,
2. ingest a messy external candidate list,
3. verify those candidates against IPEDS with relaxed thresholds,
4. merge provenance into a single event table,
5. build a v2-style verified-event panel, and
6. produce per-degree FOIA outcome plots plus a detailed text report.
"""

from __future__ import annotations

import argparse
import math
import os
import zlib
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import f1_foia.econ_relabels_opt_usage as base
import f1_foia.econ_relabels_opt_usage_v2 as v2
import laborlunch_plot_style as llstyle
from f1_indiv_merge.employer_entity_sql import (
    sql_clean_company_name_expr,
    sql_clean_zip_expr,
    sql_normalize_expr,
    sql_state_name_to_abbr_expr,
)


BASE_FONT_SIZE = base.BASE_FONT_SIZE
PLOT_EVENT_MIN = -5
PLOT_EVENT_MAX = 4
DI_D_REFERENCE_EVENT_TIME = -2
DI_D_EVENT_LINE_X = -0.5
DI_D_BROAD_BIN_MARKERS = ("o", "s", "v", "^", "D", "X", "P", "*", "h", "d", ">", "<")
DI_D_PLOT_FONT_SIZE = llstyle.AXIS_LABEL_FONT_SIZE
DI_D_PLOT_MARKER_SIZE = llstyle.MARKER_SIZE
DI_D_PLOT_MARKER_AREA = DI_D_PLOT_MARKER_SIZE * DI_D_PLOT_MARKER_SIZE
DI_D_PLOT_ERRORBAR_WIDTH = DI_D_PLOT_MARKER_SIZE
DI_D_ERRORBAR_ALPHA = llstyle.ERRORBAR_ALPHA
MULTI_SERIES_MARKER_SIZE = llstyle.MULTI_MARKER_SIZE
MULTI_SERIES_MARKER_AREA = MULTI_SERIES_MARKER_SIZE * MULTI_SERIES_MARKER_SIZE
MULTI_SERIES_ERRORBAR_WIDTH = MULTI_SERIES_MARKER_SIZE
MULTI_SERIES_MAX_STEP = 0.20
MULTI_SERIES_TARGET_TOTAL_SPAN = 0.70
ANALYSIS_ORIGINAL_YEAR_MIN = base.PLOT_YEAR_MIN
ANALYSIS_FOIA_YEAR_MAX = min(base.PLOT_YEAR_MAX, 2022)
ANALYSIS_IPEDS_YEAR_MAX = 2024
ANALYSIS_ORIGINAL_YEAR_MAX = ANALYSIS_FOIA_YEAR_MAX
MAX_RELABEL_YEAR = 2021
RELABEL_BROAD_BIN_SAMPLE_SEED = 42
RELABEL_BROAD_BIN_SAMPLE_N = 10
DEFAULT_IPEDS_COST_PANEL_PATH = Path(base.IPEDS_COST_PANEL_PATH)
DEFAULT_IPEDS_TUITION_COL = "tuition7"
DEFAULT_IPEDS_TUITION_COL_BY_DEGREE = {
    "Bachelor": "tuition3",
    "Master": "tuition7",
    "Doctor": "tuition7",
}
DEFAULT_IPEDS_FEE_COL_BY_DEGREE = {
    "Bachelor": "fee3",
    "Master": "fee7",
    "Doctor": "fee7",
}
DEFAULT_STEM_OPT_LONG_PATH = Path(base.STEM_OPT_LONG_PATH)
DEFAULT_OUTPUT_DIR = Path(base.root) / "h1bworkers" / "code" / "output" / "relabel_indiv"
DEFAULT_CANDIDATE_PATH = Path(base.root) / "data" / "llm_relabel_candidates"
DEFAULT_FOIA_PERSON_PANEL_PATH = Path(base.FOIA_PERSON_PANEL_PATH)
DEFAULT_EMPLOYER_MATCH_DIR = Path(base.root) / "data" / "company_matching_f1_apr2026"
DEFAULT_IPEDS_MAIN_INSTITUTIONS_PATH = Path(base.root) / "data" / "int" / "int_files_feb2026" / "ipeds_main_institutions.parquet"
DEFAULT_IPEDS_DIRECTORY_2024_PATH = Path(base.root) / "data" / "raw" / "ipeds" / "directory_info_hd" / "hd2024.csv"
DEFAULT_EVENTS_PARQUET = DEFAULT_OUTPUT_DIR / "generalized_relabels_events.parquet"
DEFAULT_EVENTS_CSV = DEFAULT_OUTPUT_DIR / "generalized_relabels_events.csv"
DEFAULT_PANEL_PARQUET = DEFAULT_OUTPUT_DIR / "generalized_relabels_panel.parquet"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "generalized_relabels_report.txt"
DEFAULT_CANDIDATE_AUDIT_CSV = DEFAULT_OUTPUT_DIR / "generalized_relabels_candidate_audit.csv"
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "generalized_relabels_plots"
DEFAULT_RELABEL_YEAR_MODE = "first"
DEFAULT_INCLUDE_DEGREE_SPECIFIC_PLOTS = False
DEFAULT_DID_SPEC = "individual_broad_bin_degree_fe" #collapsed_unit_fe OR individual_broad_bin_degree_fe
ESTIMATOR_DID = "did"
ESTIMATOR_STACKED_TREATED = "stacked_treated"
ESTIMATOR_BOTH = "both"
DEFAULT_ESTIMATOR = ESTIMATOR_DID
VALID_ESTIMATORS = (ESTIMATOR_DID, ESTIMATOR_STACKED_TREATED, ESTIMATOR_BOTH)
DID_SPEC_COLLAPSED_UNIT_FE = "collapsed_unit_fe"
DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE = "individual_broad_bin_degree_fe"
VALID_DID_SPECS = (
    DID_SPEC_COLLAPSED_UNIT_FE,
    DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE,
)
CONTROL_GROUP_NEVER_TREATED = "never_treated"
CONTROL_GROUP_LATE_TREATED = "late_treated"
CONTROL_GROUP_ALWAYS_STEM = "always_stem"
DEFAULT_CONTROL_GROUP = CONTROL_GROUP_NEVER_TREATED
VALID_CONTROL_GROUPS = (
    CONTROL_GROUP_NEVER_TREATED,
    CONTROL_GROUP_LATE_TREATED,
    CONTROL_GROUP_ALWAYS_STEM,
)
CONTROL_GROUP_ALIASES: dict[str, str] = {
    "never": CONTROL_GROUP_NEVER_TREATED,
    "nevertreated": CONTROL_GROUP_NEVER_TREATED,
    "never-treated": CONTROL_GROUP_NEVER_TREATED,
    "never_treated": CONTROL_GROUP_NEVER_TREATED,
    "late": CONTROL_GROUP_LATE_TREATED,
    "future": CONTROL_GROUP_LATE_TREATED,
    "future_treated": CONTROL_GROUP_LATE_TREATED,
    "late-treated": CONTROL_GROUP_LATE_TREATED,
    "late_treated": CONTROL_GROUP_LATE_TREATED,
    "always-stem": CONTROL_GROUP_ALWAYS_STEM,
    "always_stem": CONTROL_GROUP_ALWAYS_STEM,
    "stem": CONTROL_GROUP_ALWAYS_STEM,
}
LATE_TREATED_CONTROL_YEAR_MIN = 2020
LATE_TREATED_CONTROL_YEAR_MAX = 2022
ALWAYS_STEM_FIRST_YEAR_MAX = 2014
ALWAYS_STEM_COMPARABLE_CIP2 = {
    "11",  # Computer and Information Sciences
    "14",  # Engineering
    "26",  # Biological and Biomedical Sciences
    "40",  # Physical Sciences
}

STRICT_THRESHOLDS: dict[str, float] = {
    "min_share_intl": 0.20,
    "min_source_baseline": 10.0,
    "min_source_drop_abs": 5.0,
    "min_source_drop_pct": 0.50,
    "min_target_offset_share": 0.60,
    "max_net_loss_share": 0.50,
    "source_persistence_drop_share": 0.30,
    "target_persistence_gain_share": 0.30,
    "lookback_years": 3.0,
    "lookahead_years": 2.0,
}

RELAXED_THRESHOLDS: dict[str, float] = {
    "min_share_intl": 0.10,
    "min_source_baseline": 5.0,
    "min_source_drop_abs": 3.0,
    "min_source_drop_pct": 0.35,
    "min_target_offset_share": 0.40,
    "max_net_loss_share": 0.75,
    "source_persistence_drop_share": 0.15,
    "target_persistence_gain_share": 0.15,
    "lookback_years": float(v2.LOOKBACK_YEARS),
    "lookahead_years": float(v2.LOOKAHEAD_YEARS),
}

CONTROL_GUARD_THRESHOLDS: dict[str, float] = {
    "min_program_nonresident_share": 0.10,
    "min_source_pre_count": 5.0,
    "min_source_drop_count": 3.0,
    "min_source_drop_fraction": 0.35,
    "min_target_growth_share_of_source_drop_threshold": 0.60,
    "max_net_loss_share_of_source_drop": 0.75,
    "min_persistent_source_drop_fraction": 0.20,
    "min_persistent_target_gain_share_of_source_drop_threshold": 0.2,
    "pre_window_years": float(v2.LOOKBACK_YEARS),
    "post_window_years": float(v2.LOOKAHEAD_YEARS),
}

CONTROL_MATCH_SIZE_CALIPER_MIN_RATIO = 0.25
CONTROL_MATCH_SIZE_CALIPER_MAX_RATIO = 4.0
CONTROL_MATCH_SIZE_CALIPER_MIN_TREATED_LEVEL = 5.0

CONTROL_GUARD_LEGACY_KEY_MAP: dict[str, str] = {
    "min_share_intl": "min_program_nonresident_share",
    "min_source_baseline": "min_source_pre_count",
    "min_source_drop_abs": "min_source_drop_count",
    "min_source_drop_pct": "min_source_drop_fraction",
    "min_target_offset_share": "min_target_growth_share_of_source_drop_threshold",
    "max_net_loss_share": "max_net_loss_share_of_source_drop",
    "source_persistence_drop_share": "min_persistent_source_drop_fraction",
    "target_persistence_gain_share": "min_persistent_target_gain_share_of_source_drop_threshold",
    "lookback_years": "pre_window_years",
    "lookahead_years": "post_window_years",
}

MASTER_PHD_GUARD_LAG_MIN = 2
MASTER_PHD_GUARD_LAG_MAX = 4
MASTER_PHD_GUARD_SIMILARITY_SHARE = 0.5
MIN_BROAD_TREATED_RELABEL_YEAR = 2014

DIAGNOSTIC_THRESHOLDS: dict[str, float] = {
    "min_share_intl": 0.0,
    "min_source_baseline": 0.0,
    "min_source_drop_abs": 1.0,
    "min_source_drop_pct": 0.10,
    "min_target_offset_share": 0.0,
    "max_net_loss_share": 2.0,
    "source_persistence_drop_share": 0.0,
    "target_persistence_gain_share": -1.0,
    "lookback_years": float(v2.LOOKBACK_YEARS),
    "lookahead_years": float(v2.LOOKAHEAD_YEARS),
}

CANDIDATE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "candidate_id": (
        "candidateid",
        "id",
        "rowid",
        "eventid",
        "sourceid",
    ),
    "candidate_school_name": (
        "school",
        "schoolname",
        "institution",
        "institutionname",
        "university",
        "universityname",
        "college",
        "instname",
    ),
    "candidate_approx_year": (
        "year",
        "date",
        "eventdate",
        "relabeldate",
        "dateofevent",
        "dateofrelabelevent",
        "approxyear",
        "approximateyear",
        "likelyyear",
        "relabelyear",
        "eventyear",
    ),
    "candidate_program_desc": (
        "program",
        "programname",
        "programdescription",
        "programdesc",
        "description",
        "major",
        "majorname",
        "field",
        "fieldname",
    ),
    "candidate_degree_label": (
        "degree",
        "degreelabel",
        "degreetype",
        "degreelevel",
        "awardlevel",
        "level",
    ),
    "candidate_notes": (
        "notes",
        "note",
        "comment",
        "comments",
        "memo",
        "details",
    ),
}

RAW_CANDIDATE_METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "candidate_initial_cip_raw": (
        "initialcipcode",
        "initialcip",
        "originalcip",
        "priorcip",
        "oldcip",
        "sourcecip",
        "fromcip",
    ),
    "candidate_target_cip_raw": (
        "newcipcode",
        "newcip",
        "updatedcip",
        "targetcip",
        "tocip",
        "currentcip",
    ),
}

SUPPORTED_CANDIDATE_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet"}

LLM_CANDIDATE_SCHOOL_OVERRIDES: dict[str, str] = {
    "uc berkeley": "University of California-Berkeley",
    "ucla": "University of California-Los Angeles",
    "unc": "University of North Carolina at Chapel Hill",
    "usc": "University of Southern California",
    "ut austin": "University of Texas at Austin",
    "uva": "University of Virginia",
    "mit": "Massachusetts Institute of Technology",
    "northwestern": "Northwestern University",
    "texas a m": "Texas A & M University-College Station",
}

LLM_CANDIDATE_SOURCE_PRIORITY: dict[str, int] = {
    "journalism_cip_relabel_events_all_schools.csv": 0,
    "stem_opt_cip_relabel_events_batch2.csv": 1,
    "stem_opt_cip_relabel_events.csv": 2,
}

KNOWN_LLM_DUPLICATE_RULES: tuple[dict[str, object], ...] = (
    {
        "candidate_school_name": "University of California-Berkeley",
        "candidate_degree_type": "Master",
        "candidate_program_signature": "master of journalism",
        "candidate_target_cip6_hint": "090702",
    },
)

DEGREE_TYPE_TO_AWLEVELS: dict[str, tuple[int, ...]] = {
    "Bachelor": (5,),
    "Master": (7,),
    "Doctor": (9, 17),
    "Other": (),
}

DEGREE_TYPE_TO_FOIA_LABEL: dict[str, str] = {
    "Bachelor": "BACHELOR'S",
    "Master": "MASTER'S",
    "Doctor": "DOCTORATE",
}

AWLEVEL_TO_DEGREE_TYPE: dict[int, str] = {
    5: "Bachelor",
    7: "Master",
    9: "Doctor",
    17: "Doctor",
}

DEFAULT_YVARS = [
    "stem_cip_eligible_share",
    "opt_share",
    "opt_stem_share",
    "status_change_share",
    "post_grad_authorization_years_avg",
    "opt_duration_years_avg",
    "unique_employers",
    "unique_opt_cities",
    "auth_employment_tenure_years",
    "employer_opt_intensity_pctile",
    "internship_count",
    "internship_opt_years",
    "ctotalt",
    "cnralt",
    "cnralt_share_of_ctotalt",
    "f1_share_of_ctotalt",
    "f1_share_of_cnralt",
    "avg_tuition",
    "avg_tuition_ipeds",
    "avg_fees_ipeds",
    "avg_students_personal_funds",
    "avg_total_funds",
]
MASTER_APPENDIX_DID_YVARS = list(DEFAULT_YVARS)
POOLED_DEGREE_TYPES = ("Bachelor", "Master", "Doctor")
CALENDAR_YEAR_APPENDIX_TOP_N = 4
EMPLOYER_HISTORY_YVARS = (
    "unique_employers",
    "unique_opt_cities",
    "auth_employment_tenure_years",
    "employer_opt_intensity_pctile",
    "internship_count",
    "internship_opt_years",
)
PROGRAM_LEVEL_SHARE_NUMERATOR_DENOMINATOR: dict[str, tuple[str, str]] = {
    "stem_cip_eligible_share": ("stem_cip_eligible_users", "total_grads"),
    "opt_share": ("opt_users", "total_grads"),
    "opt_stem_share": ("opt_stem_users", "total_grads"),
    "status_change_share": ("status_change_users", "total_grads"),
    "post_grad_authorization_years_avg": ("total_post_grad_authorization_years", "total_grads"),
    "opt_duration_years_avg": ("total_opt_duration_years", "total_grads"),
    "opt_years_avg": ("total_post_grad_authorization_years", "total_grads"),
    "internship_count": ("total_internships", "total_grads"),
    "internship_opt_years": ("total_internship_opt_years", "total_grads"),
    "f1_share_of_ctotalt": ("total_grads", "ctotalt"),
    "f1_share_of_cnralt": ("total_grads", "cnralt"),
    "cnralt_share_of_ctotalt": ("cnralt", "ctotalt"),
    "avg_tuition": ("tuition_total", "total_grads"),
    "avg_tuition_ipeds": ("tuition_ipeds_total", "total_grads"),
    "avg_fees_ipeds": ("fees_ipeds_total", "total_grads"),
    "avg_students_personal_funds": ("students_personal_funds_total", "total_grads"),
    "avg_total_funds": ("total_funds", "total_grads"),
}
PROGRAM_LEVEL_MEAN_YVARS = {
    "unique_employers",
    "unique_opt_cities",
    "auth_employment_tenure_years",
    "employer_opt_intensity_pctile",
}
PROGRAM_LEVEL_IPEDS_COUNT_YVARS = {"ctotalt", "cnralt"}
PROGRAM_LEVEL_IPEDS_SHARE_YVARS = {"cnralt_share_of_ctotalt"}
PROGRAM_LEVEL_IPEDS_YVARS = PROGRAM_LEVEL_IPEDS_COUNT_YVARS | PROGRAM_LEVEL_IPEDS_SHARE_YVARS
PERCENTAGE_POINT_YVARS = {
    "stem_cip_eligible_share",
    "opt_share",
    "opt_stem_share",
    "status_change_share",
    "f1_share_of_ctotalt",
    "f1_share_of_cnralt",
    "cnralt_share_of_ctotalt",
}

VERIFIED_EVENT_COLUMNS = [
    "unitid",
    "awlevel",
    "degree_type",
    "relabel_year",
    "year",
    "relabel_type",
    "broad_pair_bin",
    "broad_bin_eligible",
    "event_source_cip6",
    "target_cip6",
    "source_total",
    "source_total_prev",
    "source_total_intl",
    "source_total_intl_prev",
    "target_total",
    "target_total_prev",
    "target_total_intl",
    "target_total_intl_prev",
    "ctotalt",
    "cnralt",
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
    "found_in_ipeds_scan",
    "found_in_external_candidates",
    "external_verified",
    "event_origin_category",
    "source_cip_label",
    "target_cip_label",
    "source_major",
    "target_major",
    "event_flag",
    "relabel_flag",
    "candidate_id",
    "candidate_school_name",
    "candidate_approx_year",
    "candidate_program_desc",
    "candidate_degree_label",
    "candidate_notes",
    "candidate_major",
    "candidate_source_cip_bin",
    "candidate_target_cip_bin",
    "candidate_pair_bin",
    "candidate_cip_parse_notes",
    "n_linked_candidates",
    "school_match_method",
    "school_match_score",
    "school_match_name",
    "verification_notes",
    "best_candidate_rank_score",
    "best_year_distance",
    "best_text_similarity",
    "best_nearby_year",
    "best_nearby_source_cip6",
    "best_nearby_target_cip6",
    "best_nearby_relabel_score",
    "diagnostic_best_year",
    "diagnostic_best_source_cip6",
    "diagnostic_best_target_cip6",
    "diagnostic_best_score",
]

EVENT_OUTPUT_STRING_COLUMNS = {
    "degree_type",
    "relabel_type",
    "broad_pair_bin",
    "event_source_cip6",
    "target_cip6",
    "event_origin_category",
    "source_cip_label",
    "target_cip_label",
    "source_major",
    "target_major",
    "candidate_id",
    "candidate_school_name",
    "candidate_approx_year",
    "candidate_program_desc",
    "candidate_degree_label",
    "candidate_notes",
    "candidate_major",
    "candidate_source_cip_bin",
    "candidate_target_cip_bin",
    "candidate_pair_bin",
    "candidate_cip_parse_notes",
    "school_match_method",
    "school_match_name",
    "verification_notes",
    "best_nearby_source_cip6",
    "best_nearby_target_cip6",
    "diagnostic_best_source_cip6",
    "diagnostic_best_target_cip6",
}

EVENT_OUTPUT_INT_COLUMNS = {
    "unitid",
    "awlevel",
    "relabel_year",
    "year",
    "broad_bin_eligible",
    "found_in_ipeds_scan",
    "found_in_external_candidates",
    "external_verified",
    "event_flag",
    "relabel_flag",
    "n_linked_candidates",
    "best_year_distance",
    "best_nearby_year",
    "diagnostic_best_year",
}

EVENT_OUTPUT_FLOAT_COLUMNS = {
    "source_total",
    "source_total_prev",
    "source_total_intl",
    "source_total_intl_prev",
    "target_total",
    "target_total_prev",
    "target_total_intl",
    "target_total_intl_prev",
    "ctotalt",
    "cnralt",
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
    "school_match_score",
    "best_candidate_rank_score",
    "best_text_similarity",
    "best_nearby_relabel_score",
    "diagnostic_best_score",
}


def _progress(message: str) -> None:
    print(f"[generalized_relabels] {message}")


def _slugify(value: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_") or "value"


def _tex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("$", r"\$")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    return text


def _coerce_verified_event_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for column in EVENT_OUTPUT_STRING_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype("string")
    for column in EVENT_OUTPUT_INT_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    for column in EVENT_OUTPUT_FLOAT_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Float64")
    return out


def _cip_rule(
    label: str,
    *,
    include_exact: tuple[str, ...] = (),
    include_prefixes: tuple[str, ...] = (),
    exclude_exact: tuple[str, ...] = (),
    exclude_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "label": label,
        "include_exact": tuple(str(v) for v in include_exact),
        "include_prefixes": tuple(str(v) for v in include_prefixes),
        "exclude_exact": tuple(str(v) for v in exclude_exact),
        "exclude_prefixes": tuple(str(v) for v in exclude_prefixes),
    }


BROAD_BIN_FOR_CANDIDATE_PAIR_BIN: dict[str, str] = {
    "economics_to_econometrics": "econ_to_quant_econ",
    "business_core_to_management_science_family": "business_52_to_52",
    "business_core_to_econometrics": "business_52_to_52",
    "business_core_to_statistics": "business_52_to_52",
    "business_or_accounting_to_quantitative_business": "business_52_to_52",
    "business_or_it_to_it_project_management": "business_52_to_52",
    "business_core_to_business_analytics": "business_52_to_52",
    "business_core_to_environmental_studies": "business_52_to_52",
    "finance_to_quantitative_finance_family": "finance_to_quantitative_finance",
    "journalism_to_digital_media": "communication_to_digital_media",
    "communication_media_to_digital_media": "communication_to_digital_media",
    "architecture_to_architectural_building_sciences": "architecture_design_to_built_env_stem",
    "architecture_design_to_sustainability_studies": "architecture_design_to_built_env_stem",
    "agricultural_economics_to_mathematical_economics": "agricultural_econ_to_mathematical_econ",
}

BROAD_BIN_SPECS: dict[str, dict[str, object]] = {
    "econ_to_quant_econ": {
        "source_rule": _cip_rule(
            "econ_to_quant_econ_source_group",
            include_prefixes=("4506",),
            exclude_exact=("450603",),
        ),
        "target_rule": _cip_rule("econ_to_quant_econ_target_group", include_exact=("450603",)),
    },
    "business_52_to_52": {
        "source_rule": _cip_rule(
            "business_52_to_52_source_group",
            include_prefixes=("52",),
            exclude_prefixes=("5208", "5213"),
        ),
        "target_rule": _cip_rule("business_52_to_52_target_group", include_prefixes=("5213",)),
    },
    "finance_to_quantitative_finance": {
        "source_rule": _cip_rule(
            "finance_to_quantitative_finance_source_group",
            include_prefixes=("5208",),
        ),
        "target_rule": _cip_rule(
            "finance_to_applied_math_target_group",
            include_exact=("270305", "270501"),
        ),
    },
    "communication_to_digital_media": {
        "source_rule": _cip_rule(
            "communication_to_digital_media_source_group",
            include_prefixes=("0901", "0904", "0909"),
            exclude_exact=("090702",),
        ),
        "target_rule": _cip_rule("communication_to_digital_media_target_group", include_exact=("090702",)),
    },
    "architecture_design_to_built_env_stem": {
        "source_rule": _cip_rule(
            "architecture_design_to_built_env_stem_source_group",
            include_prefixes=("0402", "0403", "0406", "5004"),
        ),
        "target_rule": _cip_rule(
            "architecture_design_to_built_env_stem_target_group",
            include_exact=("040902", "303301"),
        ),
    },
    "agricultural_econ_to_mathematical_econ": {
        "source_rule": _cip_rule(
            "agricultural_econ_to_mathematical_econ_source_group",
            include_exact=("010103",),
        ),
        "target_rule": _cip_rule(
            "agricultural_econ_to_mathematical_econ_target_group",
            include_exact=("304901",),
        ),
    },
}

BROAD_BIN_PLOT_LABELS: dict[str, str] = {
    "econ_to_quant_econ": "Econ -> Quant Econ",
    "business_52_to_52": "Business -> Management Science",
    "finance_to_quantitative_finance": "Finance -> Applied Math",
    "communication_to_digital_media": "Communication -> Digital Media",
    "architecture_design_to_built_env_stem": "Architecture/Design -> Built Env STEM",
    "agricultural_econ_to_mathematical_econ": "Ag Econ -> Mathematical Econ",
}

EXCLUDED_BROAD_PAIR_BINS = frozenset({"communication_to_digital_media"})
GENERALIZED_YVAR_LABELS = {
    "avg_fees_ipeds": "Average fees (IPEDS, USD)",
    "avg_students_personal_funds": "Average student personal funds (USD)",
    "avg_total_funds": "Average total funds (USD)",
}


def _rule_exact(cip6: str, *, label: str | None = None) -> dict[str, object]:
    cip6 = str(cip6).zfill(6)
    return _cip_rule(label or f"exact_{cip6}", include_exact=(cip6,))


def _normalize_cip6(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text in {"<NA>", "nan", "None"}:
        return None
    if re.fullmatch(r"\d{1,6}", text):
        return text.zfill(6)
    return text


def _cip_matches_rule(cip6: object, rule: dict[str, object] | None) -> bool:
    if rule is None:
        return False
    cip = _normalize_cip6(cip6)
    if not cip:
        return False
    include_exact = tuple(rule.get("include_exact", ()))
    include_prefixes = tuple(rule.get("include_prefixes", ()))
    exclude_exact = tuple(rule.get("exclude_exact", ()))
    exclude_prefixes = tuple(rule.get("exclude_prefixes", ()))
    if cip in exclude_exact:
        return False
    if any(cip.startswith(prefix) for prefix in exclude_prefixes):
        return False
    if cip in include_exact:
        return True
    if include_prefixes and any(cip.startswith(prefix) for prefix in include_prefixes):
        return True
    return not include_exact and not include_prefixes


def _broad_bin_from_candidate_pair_bin(candidate_pair_bin: object) -> object:
    if pd.isna(candidate_pair_bin):
        return pd.NA
    broad_pair_bin = BROAD_BIN_FOR_CANDIDATE_PAIR_BIN.get(str(candidate_pair_bin))
    return broad_pair_bin if broad_pair_bin is not None else pd.NA


def _broad_bin_from_exact_pair(source_cip6: object, target_cip6: object) -> object:
    matches = [
        broad_pair_bin
        for broad_pair_bin, spec in BROAD_BIN_SPECS.items()
        if _cip_matches_rule(source_cip6, spec["source_rule"]) and _cip_matches_rule(target_cip6, spec["target_rule"])
    ]
    matches = sorted(set(matches))
    if not matches:
        return pd.NA
    if len(matches) > 1:
        raise ValueError(f"Exact CIP pair {source_cip6}->{target_cip6} matched multiple broad bins: {matches}")
    return matches[0]


def _annotate_candidate_audit_broad_bins(candidate_audit: pd.DataFrame) -> pd.DataFrame:
    if candidate_audit.empty:
        return candidate_audit.copy()
    out = candidate_audit.copy()
    out["broad_pair_bin"] = out.get("candidate_pair_bin", pd.Series(pd.NA, index=out.index)).map(_broad_bin_from_candidate_pair_bin)
    return out


def _exclude_disallowed_candidate_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    had_broad_pair_bin = "broad_pair_bin" in candidates.columns
    out = _annotate_candidate_audit_broad_bins(candidates)
    keep_mask = ~out["broad_pair_bin"].isin(EXCLUDED_BROAD_PAIR_BINS)
    filtered = out.loc[keep_mask].copy()
    if not had_broad_pair_bin and "broad_pair_bin" in filtered.columns:
        filtered = filtered.drop(columns=["broad_pair_bin"])
    return filtered


def annotate_event_broad_bins(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        out = events.copy()
        out["broad_pair_bin"] = pd.Series(dtype="object")
        out["broad_bin_eligible"] = pd.Series(dtype="Int64")
        return out

    out = events.copy()
    broad_pair_bins: list[object] = []
    broad_eligible: list[int] = []
    for row in out.itertuples(index=False):
        if pd.notna(getattr(row, "event_source_cip6", pd.NA)) and pd.notna(getattr(row, "target_cip6", pd.NA)):
            broad_pair_bin = _broad_bin_from_exact_pair(row.event_source_cip6, row.target_cip6)
            broad_pair_bins.append(broad_pair_bin)
            broad_eligible.append(int(pd.notna(broad_pair_bin)))
            continue
        broad_pair_bin = _broad_bin_from_candidate_pair_bin(getattr(row, "candidate_pair_bin", pd.NA))
        broad_pair_bins.append(broad_pair_bin)
        broad_eligible.append(0)
    out["broad_pair_bin"] = pd.Series(broad_pair_bins, index=out.index, dtype="object")
    out["broad_bin_eligible"] = pd.Series(broad_eligible, index=out.index, dtype="Int64")
    return out


def _exclude_disallowed_event_rows(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return annotate_event_broad_bins(events)
    out = annotate_event_broad_bins(events)
    keep_mask = ~out["broad_pair_bin"].isin(EXCLUDED_BROAD_PAIR_BINS)
    return out.loc[keep_mask].copy()


def build_broad_bin_membership(cip_universe: list[str] | tuple[str, ...] | set[str]) -> dict[str, dict[str, tuple[str, ...]]]:
    membership: dict[str, dict[str, tuple[str, ...]]] = {}
    cip_values = sorted(str(value).zfill(6) for value in cip_universe)
    for broad_pair_bin, spec in BROAD_BIN_SPECS.items():
        source_cips = tuple(cip for cip in cip_values if _cip_matches_rule(cip, spec["source_rule"]))
        target_cips = tuple(cip for cip in cip_values if _cip_matches_rule(cip, spec["target_rule"]))
        membership[broad_pair_bin] = {
            "source_cips": source_cips,
            "target_cips": target_cips,
            "all_cips": tuple(sorted(set(source_cips) | set(target_cips))),
        }
    return membership


def _broad_membership_rows(
    membership: dict[str, dict[str, tuple[str, ...]]],
    *,
    side: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cip_key = f"{side}_cips" if side in {"source", "target"} else "all_cips"
    for broad_pair_bin, spec in membership.items():
        for cip6 in spec[cip_key]:
            rows.append({"broad_pair_bin": broad_pair_bin, "cip6": cip6})
    return pd.DataFrame(rows)


def _normalize_relabel_year_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"first", "earliest"}:
        return "first"
    if normalized in {"largest", "max"}:
        return "largest"
    raise ValueError(f"Unknown relabel_year_mode '{value}'; expected 'first' or 'largest'")


def build_broad_treated_events(
    events: pd.DataFrame,
    *,
    relabel_year_mode: str = DEFAULT_RELABEL_YEAR_MODE,
) -> pd.DataFrame:
    relabel_year_mode = _normalize_relabel_year_mode(relabel_year_mode)
    annotated = annotate_event_broad_bins(events)
    annotated["_numeric_relabel_year"] = pd.to_numeric(annotated.get("relabel_year"), errors="coerce").astype("Int64")
    eligible = annotated[
        annotated["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"])
        & annotated["_numeric_relabel_year"].notna()
        & annotated["_numeric_relabel_year"].ge(MIN_BROAD_TREATED_RELABEL_YEAR)
        & annotated["_numeric_relabel_year"].le(MAX_RELABEL_YEAR)
        & annotated["awlevel"].notna()
        & annotated["broad_bin_eligible"].eq(1)
    ].copy()
    if eligible.empty:
        return eligible

    eligible["relabel_year"] = eligible["_numeric_relabel_year"]
    eligible["awlevel"] = pd.to_numeric(eligible["awlevel"], errors="coerce").astype("Int64")
    eligible["_sort_relabel_score"] = pd.to_numeric(eligible["relabel_score"], errors="coerce").fillna(float("-inf"))
    relabel_year_ascending = relabel_year_mode == "first"
    if relabel_year_mode == "largest":
        sort_columns = [
            "unitid",
            "awlevel",
            "broad_pair_bin",
            "_sort_relabel_score",
            "relabel_year",
            "event_source_cip6",
            "target_cip6",
        ]
        sort_ascending = [True, True, True, False, False, True, True]
    else:
        sort_columns = [
            "unitid",
            "awlevel",
            "broad_pair_bin",
            "relabel_year",
            "_sort_relabel_score",
            "event_source_cip6",
            "target_cip6",
        ]
        sort_ascending = [True, True, True, relabel_year_ascending, False, True, True]
    eligible = eligible.sort_values(
        sort_columns,
        ascending=sort_ascending,
        na_position="last",
    )

    grouped_rows: list[dict[str, object]] = []
    for _, group in eligible.groupby(["unitid", "awlevel", "broad_pair_bin"], dropna=False, sort=False):
        origin_series = group.get("event_origin_category", pd.Series(pd.NA, index=group.index)).astype("string")

        def _group_max_flag(column: str, *, fallback_categories: tuple[str, ...] = ()) -> int:
            if column in group.columns:
                return int(pd.to_numeric(group[column], errors="coerce").fillna(0).max())
            if fallback_categories:
                return int(origin_series.isin(fallback_categories).any())
            return 0

        trigger = group.iloc[0].copy()
        trigger["relabel_type"] = trigger["broad_pair_bin"]
        trigger["year"] = trigger["relabel_year"]
        trigger["found_in_ipeds_scan"] = _group_max_flag("found_in_ipeds_scan", fallback_categories=("ipeds_only",))
        trigger["found_in_external_candidates"] = _group_max_flag(
            "found_in_external_candidates",
            fallback_categories=("external_ipeds_verified", "external_only"),
        )
        trigger["external_verified"] = _group_max_flag(
            "external_verified",
            fallback_categories=("external_ipeds_verified",),
        )
        trigger["event_origin_category"] = _apply_event_origin_category(trigger)
        trigger["broad_bin_eligible"] = 1
        grouped_rows.append(trigger.to_dict())

    out = pd.DataFrame(grouped_rows)
    drop_columns = [column for column in ["_sort_relabel_score", "_numeric_relabel_year"] if column in out.columns]
    if drop_columns:
        out = out.drop(columns=drop_columns)
    return out.reset_index(drop=True)


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _normalize_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[0-9]+/[0-9]+/[0-9]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_school_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(
        r"\b(at|campus|inc|inc\.)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\buniv\b", "university", text, flags=re.IGNORECASE)
    return _normalize_text(text)


def _normalize_control_guard_thresholds(thresholds: dict[str, float] | None) -> dict[str, float]:
    """Return control-school relabel guard thresholds using the clear key names."""
    out = dict(CONTROL_GUARD_THRESHOLDS)
    if thresholds is None:
        return out
    for key, value in thresholds.items():
        normalized_key = CONTROL_GUARD_LEGACY_KEY_MAP.get(key, key)
        if normalized_key not in out:
            valid_keys = sorted(set(out) | set(CONTROL_GUARD_LEGACY_KEY_MAP))
            raise KeyError(
                f"Unknown control guard threshold '{key}'. Expected one of: "
                f"{', '.join(valid_keys)}"
            )
        out[normalized_key] = float(value)
    return out


def _clean_cip_label(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"^\d+\.\d+-", "", text)
    return text.strip()


def _candidate_major(value: object) -> str:
    text = _normalize_text(value)
    return text or "unknown"


def _candidate_program_signature(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\(.*?\)", " ", text)
    return _normalize_text(text)


def _canonicalize_candidate_school_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return ""
    override_key = _normalize_school_name(text)
    return LLM_CANDIDATE_SCHOOL_OVERRIDES.get(override_key, text)


def _extract_candidate_year(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    if match:
        return int(match.group(1))
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric):
        numeric_int = int(numeric)
        if 1900 <= numeric_int <= 2099:
            return numeric_int
    return pd.NA


def _extract_cip6_hint(value: object) -> tuple[object, str]:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return pd.NA, "missing"
    text_lower = text.lower()
    if "not publicly located" in text_lower or "not located in this pass" in text_lower:
        return pd.NA, "missing"

    status = "exact"
    if "inferred" in text_lower:
        status = "inferred"
    elif "mapped" in text_lower:
        status = "mapped"
    elif "likely" in text_lower or "probably" in text_lower:
        status = "likely"

    dot_match = re.search(r"(?<!\d)(\d{1,2})\.(\d{4})(?!\d)", text)
    if dot_match:
        cip6 = f"{int(dot_match.group(1)):02d}{dot_match.group(2)}"
        if status == "exact" and not re.fullmatch(r"\s*\d{1,2}\.\d{4}\s*", text):
            status = "numeric_with_note"
        return cip6, status

    plain_match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    if plain_match:
        cip6 = plain_match.group(1)
        if status == "exact" and not re.fullmatch(r"\s*\d{6}\s*", text):
            status = "numeric_with_note"
        return cip6, status

    return pd.NA, "missing"


def _infer_candidate_confidence(notes: object, source_status: object, target_status: object) -> str:
    note_text = "" if pd.isna(notes) else str(notes).strip().lower()
    explicit = re.search(r"\[((?:medium[- ]high)|high|medium|low) confidence\]", note_text)
    if explicit:
        return explicit.group(1).replace("-", "_").replace(" ", "_")

    source_status = str(source_status or "").strip().lower()
    target_status = str(target_status or "").strip().lower()
    exact_like = {"exact", "numeric_with_note"}
    if source_status in exact_like and target_status in exact_like:
        return "high"
    if target_status in exact_like and source_status in {"inferred", "mapped", "likely"}:
        return "medium"
    if target_status in exact_like or source_status in exact_like or source_status in {"inferred", "mapped", "likely"}:
        return "medium_low"
    return "low"


def _prefer_exact_rule(cip6_hint: str | None, fallback_rule: dict[str, object]) -> dict[str, object]:
    if cip6_hint:
        return _rule_exact(cip6_hint, label=f"exact_{cip6_hint}")
    return fallback_rule


def _business_source_rule(program_norm: str, source_hint: str | None) -> dict[str, object]:
    if source_hint:
        return _rule_exact(source_hint, label=f"exact_{source_hint}")
    if "accounting" in program_norm or "macc" in program_norm:
        return _cip_rule("accounting_core", include_prefixes=("5203",))
    if "finance" in program_norm or "mfin" in program_norm:
        return _cip_rule("finance_core", include_prefixes=("5208",))
    if "information technology" in program_norm:
        return _cip_rule("business_or_it_core", include_prefixes=("52", "11"), exclude_prefixes=("5213",))
    return _cip_rule("business_core", include_prefixes=("52",), exclude_prefixes=("5213",))


def _parse_candidate_cip_constraints(row: pd.Series | dict[str, object]) -> dict[str, object]:
    row_dict = row if isinstance(row, dict) else row.to_dict()
    program_norm = _normalize_text(row_dict.get("candidate_program_desc"))
    source_hint = _normalize_cip6(row_dict.get("candidate_source_cip6_hint"))
    target_hint = _normalize_cip6(row_dict.get("candidate_target_cip6_hint"))

    source_rule: dict[str, object] | None = None
    target_rule: dict[str, object] | None = None
    pair_bin = "unparsed_candidate_pair"
    parse_notes = "no hand-parse rule matched candidate"

    if (
        source_hint == "010103"
        or target_hint == "304901"
        or "agricultural economics" in program_norm
    ):
        source_rule = _prefer_exact_rule(source_hint, _cip_rule("agricultural_economics", include_exact=("010103",)))
        target_rule = _prefer_exact_rule(target_hint, _cip_rule("mathematical_economics", include_exact=("304901",)))
        pair_bin = "agricultural_economics_to_mathematical_economics"
        parse_notes = "hand-parsed agricultural economics relabel family"
    elif target_hint == "090702" or "journal" in program_norm or "media science" in program_norm:
        if "media science" in program_norm:
            source_rule = _prefer_exact_rule(
                source_hint,
                _cip_rule(
                    "communication_media_pre_digital",
                    include_prefixes=("0901", "0904", "0909"),
                    exclude_exact=("090702",),
                ),
            )
            pair_bin = "communication_media_to_digital_media"
            parse_notes = "hand-parsed communication/media to digital-media family"
        else:
            source_rule = _prefer_exact_rule(
                source_hint,
                _cip_rule(
                    "journalism_pre_digital",
                    include_exact=("090401",),
                    include_prefixes=("0904",),
                    exclude_exact=("090702",),
                ),
            )
            pair_bin = "journalism_to_digital_media"
            parse_notes = "hand-parsed journalism to digital-media family"
        target_rule = _prefer_exact_rule(target_hint, _cip_rule("digital_media_multimedia", include_exact=("090702",)))
    elif target_hint == "450603" or "economics" in program_norm:
        if "mba major" in program_norm or "business economics public policy" in program_norm:
            source_rule = _business_source_rule(program_norm, source_hint)
            pair_bin = "business_core_to_econometrics"
            parse_notes = "hand-parsed business program moving into econometrics/quantitative economics"
        else:
            source_rule = _prefer_exact_rule(
                source_hint,
                _cip_rule(
                    "economics_pre_stem",
                    include_exact=("450601",),
                    include_prefixes=("4506",),
                    exclude_exact=("450603",),
                ),
            )
            pair_bin = "economics_to_econometrics"
            parse_notes = "hand-parsed economics general to econometrics/quantitative economics family"
        target_rule = _prefer_exact_rule(target_hint, _cip_rule("econometrics_quantitative_economics", include_exact=("450603",)))
    elif target_hint == "450102" or "public policy" in program_norm:
        source_rule = _prefer_exact_rule(
            source_hint,
            _cip_rule(
                "public_policy_social_science_core",
                include_prefixes=("4404", "4405", "4501", "4510", "4506"),
                exclude_exact=("450102",),
            ),
        )
        target_rule = _prefer_exact_rule(target_hint, _cip_rule("research_methodology_and_quant_methods", include_exact=("450102",)))
        pair_bin = "public_policy_to_quant_methods"
        parse_notes = "hand-parsed public-policy/social-science to quantitative-methods family"
    elif target_hint == "040902" or "architecture" in program_norm:
        source_rule = _prefer_exact_rule(
            source_hint,
            _cip_rule(
                "architecture_design_core",
                include_prefixes=("0402", "0403", "0406", "5004"),
                exclude_exact=("040902",),
            ),
        )
        target_rule = _prefer_exact_rule(target_hint, _cip_rule("architectural_building_sciences", include_exact=("040902",)))
        pair_bin = "architecture_to_architectural_building_sciences"
        parse_notes = "hand-parsed architecture/design relabel family"
    elif target_hint == "303301" or "urban design" in program_norm:
        source_rule = _prefer_exact_rule(
            source_hint,
            _cip_rule(
                "architecture_design_core",
                include_prefixes=("0402", "0403", "0406", "5004"),
                exclude_exact=("303301",),
            ),
        )
        target_rule = _prefer_exact_rule(target_hint, _cip_rule("sustainability_studies", include_exact=("303301",)))
        pair_bin = "architecture_design_to_sustainability_studies"
        parse_notes = "hand-parsed urban-design relabel family"
    elif target_hint == "111005" or "management of information technology" in program_norm:
        source_rule = _business_source_rule(program_norm, source_hint)
        target_rule = _prefer_exact_rule(target_hint, _cip_rule("information_technology_project_management", include_exact=("111005",)))
        pair_bin = "business_or_it_to_it_project_management"
        parse_notes = "hand-parsed business/IT relabel family"
    elif (
        "finance" in program_norm
        or "mfin" in program_norm
        or target_hint in {"270305", "270501"}
    ):
        source_rule = _prefer_exact_rule(source_hint, _cip_rule("finance_core", include_prefixes=("5208",)))
        target_rule = (
            _rule_exact(target_hint, label=f"exact_{target_hint}")
            if target_hint in {"270305", "270501"}
            else _cip_rule("finance_applied_math_target_family", include_exact=("270305", "270501"))
        )
        pair_bin = "finance_to_quantitative_finance_family"
        parse_notes = "hand-parsed finance to applied-math family"
    elif target_hint == "270305" or "accounting" in program_norm or "macc" in program_norm:
        source_rule = _business_source_rule(program_norm, source_hint)
        target_rule = _prefer_exact_rule(
            target_hint,
            _cip_rule(
                "quantitative_business_accounting_target_family",
                include_exact=("270305", "270501", "307102", "111005", "521301", "521399"),
            ),
        )
        pair_bin = "business_or_accounting_to_quantitative_business"
        parse_notes = "hand-parsed accounting/quant-business relabel family"
    elif target_hint == "307102" or "business analytics" in program_norm:
        source_rule = _business_source_rule(program_norm, source_hint)
        target_rule = _prefer_exact_rule(target_hint, _cip_rule("business_analytics", include_exact=("307102",)))
        pair_bin = "business_core_to_business_analytics"
        parse_notes = "hand-parsed business to business-analytics relabel family"
    elif target_hint == "270501" or "statistics" in program_norm:
        source_rule = _business_source_rule(program_norm, source_hint)
        target_rule = _prefer_exact_rule(target_hint, _cip_rule("statistics_general", include_exact=("270501",)))
        pair_bin = "business_core_to_statistics"
        parse_notes = "hand-parsed business to statistics relabel family"
    elif target_hint == "030103" or "bees" in program_norm or "sustainability" in program_norm:
        source_rule = _business_source_rule(program_norm, source_hint)
        target_rule = _prefer_exact_rule(target_hint, _cip_rule("environmental_studies", include_exact=("030103",)))
        pair_bin = "business_core_to_environmental_studies"
        parse_notes = "hand-parsed business to environmental-studies relabel family"
    elif target_hint in {"521301", "521399"} or "mba" in program_norm or "commerce" in program_norm or "business" in program_norm:
        source_rule = _business_source_rule(program_norm, source_hint)
        target_rule = _prefer_exact_rule(
            target_hint,
            _cip_rule("management_science_family", include_prefixes=("5213",)),
        )
        pair_bin = "business_core_to_management_science_family"
        parse_notes = "hand-parsed business/MBA/commerce to management-science family"
    return {
        "candidate_source_cip_bin": source_rule["label"] if source_rule is not None else pd.NA,
        "candidate_target_cip_bin": target_rule["label"] if target_rule is not None else pd.NA,
        "candidate_pair_bin": pair_bin,
        "candidate_cip_parse_notes": parse_notes,
        "_candidate_source_rule": source_rule,
        "_candidate_target_rule": target_rule,
    }


def _filter_pairs_for_candidate_constraints(
    pairs: pd.DataFrame,
    *,
    source_rule: dict[str, object] | None,
    target_rule: dict[str, object] | None,
) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    filtered = pairs.copy()
    if source_rule is not None:
        filtered = filtered[filtered["source_cip6"].map(lambda value: _cip_matches_rule(value, source_rule))]
    if target_rule is not None:
        filtered = filtered[filtered["target_cip6"].map(lambda value: _cip_matches_rule(value, target_rule))]
    return filtered.reset_index(drop=True)


def _rule_signature(rule: dict[str, object] | None) -> tuple[object, ...]:
    if rule is None:
        return ("none",)
    return (
        tuple(sorted(str(value) for value in rule.get("include_exact", ()))),
        tuple(sorted(str(value) for value in rule.get("include_prefixes", ()))),
        tuple(sorted(str(value) for value in rule.get("exclude_exact", ()))),
        tuple(sorted(str(value) for value in rule.get("exclude_prefixes", ()))),
    )


def derive_allowable_pair_configs(candidates: pd.DataFrame) -> list[dict[str, object]]:
    if candidates.empty:
        return []
    configs: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for row in candidates.itertuples(index=False):
        parsed = _parse_candidate_cip_constraints(row._asdict())
        source_rule = parsed["_candidate_source_rule"]
        target_rule = parsed["_candidate_target_rule"]
        if source_rule is None or target_rule is None:
            continue
        signature = (_rule_signature(source_rule), _rule_signature(target_rule))
        if signature in seen:
            continue
        seen.add(signature)
        configs.append(
            {
                "candidate_pair_bin": parsed["candidate_pair_bin"],
                "candidate_source_cip_bin": parsed["candidate_source_cip_bin"],
                "candidate_target_cip_bin": parsed["candidate_target_cip_bin"],
                "source_rule": source_rule,
                "target_rule": target_rule,
            }
        )
    return configs


def _cip_rule_sql(column: str, rule: dict[str, object] | None) -> str:
    if rule is None:
        return "FALSE"

    include_exact = tuple(str(value) for value in rule.get("include_exact", ()))
    include_prefixes = tuple(str(value) for value in rule.get("include_prefixes", ()))
    exclude_exact = tuple(str(value) for value in rule.get("exclude_exact", ()))
    exclude_prefixes = tuple(str(value) for value in rule.get("exclude_prefixes", ()))

    include_clauses: list[str] = []
    if include_exact:
        include_vals = ", ".join(f"'{_sql_literal(value)}'" for value in include_exact)
        include_clauses.append(f"{column} IN ({include_vals})")
    include_clauses.extend(f"{column} LIKE '{_sql_literal(prefix)}%'" for prefix in include_prefixes)

    clauses: list[str] = []
    if include_clauses:
        clauses.append("(" + " OR ".join(include_clauses) + ")")
    for value in exclude_exact:
        clauses.append(f"{column} <> '{_sql_literal(value)}'")
    clauses.extend(f"{column} NOT LIKE '{_sql_literal(prefix)}%'" for prefix in exclude_prefixes)
    if not clauses:
        return "TRUE"
    return "(" + " AND ".join(clauses) + ")"


def _allowed_pair_configs_sql(
    pair_configs: list[dict[str, object]] | None,
    *,
    source_column: str,
    target_column: str,
) -> str:
    if not pair_configs:
        return ""
    clauses: list[str] = []
    for config in pair_configs:
        source_sql = _cip_rule_sql(source_column, config.get("source_rule"))
        target_sql = _cip_rule_sql(target_column, config.get("target_rule"))
        clauses.append(f"({source_sql} AND {target_sql})")
    if not clauses:
        return ""
    return "(" + " OR ".join(clauses) + ")"


def _event_year_series(df: pd.DataFrame) -> pd.Series:
    if "year" in df.columns:
        return pd.to_numeric(df["year"], errors="coerce")
    if "relabel_year" in df.columns:
        return pd.to_numeric(df["relabel_year"], errors="coerce")
    return pd.Series(pd.NA, index=df.index, dtype="Float64")


def _master_awlevel_mask(df: pd.DataFrame) -> pd.Series:
    if "awlevel" in df.columns:
        return pd.to_numeric(df["awlevel"], errors="coerce").astype("Int64").isin(DEGREE_TYPE_TO_AWLEVELS["Master"])
    return df.get("degree_type", pd.Series(pd.NA, index=df.index)).astype("string").eq("Master")


def _doctor_awlevel_mask(df: pd.DataFrame) -> pd.Series:
    if "awlevel" in df.columns:
        return pd.to_numeric(df["awlevel"], errors="coerce").astype("Int64").isin(DEGREE_TYPE_TO_AWLEVELS["Doctor"])
    return df.get("degree_type", pd.Series(pd.NA, index=df.index)).astype("string").eq("Doctor")


def _load_doctorate_guard_raw(
    con: ddb.DuckDBPyConnection,
    *,
    ipeds_path: str | Path,
    unitids: list[int],
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    if not unitids or year_min > year_max:
        return pd.DataFrame(columns=["unitid", "year", "cip6", "ctotalt"])
    _ensure_ipeds_view(con, ipeds_path)
    unitid_filter = _int_clause(unitids, "unitid")
    return con.sql(
        f"""
        SELECT
            CAST(unitid AS BIGINT) AS unitid,
            CAST(year AS INTEGER) AS year,
            LPAD(CAST(cipcode AS VARCHAR), 6, '0') AS cip6,
            CAST(ctotalt AS DOUBLE) AS ctotalt
        FROM ipeds_raw
        WHERE unitid IS NOT NULL
          AND cipcode IS NOT NULL
          AND CAST(awlevel AS INTEGER) IN (9, 17)
          AND CAST(year AS INTEGER) BETWEEN {int(year_min)} AND {int(year_max)}
          {unitid_filter}
        """
    ).df()


def _doctorate_guard_totals(
    doctor_raw: pd.DataFrame,
) -> dict[tuple[int, str, int], tuple[float, float]]:
    if doctor_raw.empty:
        return {}
    raw = doctor_raw.copy()
    raw["unitid"] = pd.to_numeric(raw["unitid"], errors="coerce").astype("Int64")
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["cip6"] = raw["cip6"].astype("string")
    raw["ctotalt"] = pd.to_numeric(raw["ctotalt"], errors="coerce").fillna(0.0)
    raw = raw.dropna(subset=["unitid", "year", "cip6"])
    if raw.empty:
        return {}

    totals: dict[tuple[int, str, int], tuple[float, float]] = {}
    for broad_pair_bin, spec in BROAD_BIN_SPECS.items():
        source_mask = raw["cip6"].map(lambda value: _cip_matches_rule(value, spec["source_rule"]))
        target_mask = raw["cip6"].map(lambda value: _cip_matches_rule(value, spec["target_rule"]))
        grouped = (
            raw.assign(
                _source_total=raw["ctotalt"].where(source_mask, 0.0),
                _target_total=raw["ctotalt"].where(target_mask, 0.0),
            )
            .groupby(["unitid", "year"], as_index=False)[["_source_total", "_target_total"]]
            .sum()
        )
        for unitid, year, source_total, target_total in grouped.itertuples(index=False, name=None):
            totals[(int(unitid), broad_pair_bin, int(year))] = (
                float(source_total),
                float(target_total),
            )
    return totals


def _apply_master_phd_guard(
    con: ddb.DuckDBPyConnection,
    events: pd.DataFrame,
    *,
    ipeds_path: str | Path = base.IPEDS_PATH,
    lag_min: int = MASTER_PHD_GUARD_LAG_MIN,
    lag_max: int = MASTER_PHD_GUARD_LAG_MAX,
    similarity_share: float = MASTER_PHD_GUARD_SIMILARITY_SHARE,
    lookback_years: int = int(v2.LOOKBACK_YEARS),
) -> pd.DataFrame:
    required_cols = {"unitid", "source_drop"}
    if events.empty or not required_cols.issubset(events.columns):
        return events.copy()

    out = events.copy()
    if "broad_pair_bin" in out.columns:
        broad_pair_series = out["broad_pair_bin"].astype("string")
    elif {"event_source_cip6", "target_cip6"}.issubset(out.columns):
        broad_pair_series = annotate_event_broad_bins(out)["broad_pair_bin"].astype("string")
    elif {"source_cip6", "target_cip6"}.issubset(out.columns):
        broad_pair_series = annotate_event_broad_bins(
            out.rename(columns={"source_cip6": "event_source_cip6"})
        )["broad_pair_bin"].astype("string")
    else:
        return out

    event_year = _event_year_series(out).astype("Int64")
    master_pool = out.loc[_master_awlevel_mask(out) & event_year.notna()].copy()
    if master_pool.empty:
        return out

    master_pool = master_pool.assign(
        _master_row_id=master_pool.index,
        _master_unitid=pd.to_numeric(master_pool["unitid"], errors="coerce").astype("Int64"),
        _master_year=pd.to_numeric(event_year.loc[master_pool.index], errors="coerce").astype("Int64"),
        _master_source_drop=pd.to_numeric(master_pool["source_drop"], errors="coerce"),
        _master_target_increase=pd.to_numeric(master_pool.get("target_increase"), errors="coerce"),
        _master_broad_pair_bin=broad_pair_series.loc[master_pool.index],
    )
    master_pool = master_pool.dropna(subset=["_master_unitid", "_master_year", "_master_broad_pair_bin", "_master_source_drop"])
    if master_pool.empty:
        return out

    _ensure_ipeds_view(con, ipeds_path)
    data_min_year, data_max_year = _ipeds_year_bounds(con)
    doctor_raw = _load_doctorate_guard_raw(
        con,
        ipeds_path=ipeds_path,
        unitids=sorted(int(value) for value in master_pool["_master_unitid"].dropna().unique()),
        year_min=max(data_min_year, int(master_pool["_master_year"].min()) - int(lookback_years)),
        year_max=min(data_max_year, int(master_pool["_master_year"].max()) + int(lag_max)),
    )
    doctor_totals = _doctorate_guard_totals(doctor_raw)
    if not doctor_totals:
        return out

    guarded_row_ids: list[int] = []
    guard_columns = [
        "_master_row_id",
        "_master_unitid",
        "_master_year",
        "_master_source_drop",
        "_master_broad_pair_bin",
    ]
    for master_row_id, master_unitid, master_year, master_source_drop, master_broad_pair_bin in master_pool[guard_columns].itertuples(index=False, name=None):
        pre_years = [year for year in range(int(master_year) - int(lookback_years), int(master_year)) if data_min_year <= year <= data_max_year]
        post_years = [
            year
            for year in range(int(master_year) + int(lag_min), int(master_year) + int(lag_max) + 1)
            if data_min_year <= year <= data_max_year
        ]
        if not pre_years or not post_years:
            continue

        pre_source = sum(doctor_totals.get((int(master_unitid), str(master_broad_pair_bin), year), (0.0, 0.0))[0] for year in pre_years) / len(pre_years)
        pre_target = sum(doctor_totals.get((int(master_unitid), str(master_broad_pair_bin), year), (0.0, 0.0))[1] for year in pre_years) / len(pre_years)
        master_source_drop = float(master_source_drop)
        for post_year in post_years:
            post_source, post_target = doctor_totals.get(
                (int(master_unitid), str(master_broad_pair_bin), int(post_year)),
                (0.0, 0.0),
            )
            doctor_source_drop = float(pre_source - post_source)
            doctor_target_increase = float(post_target - pre_target)
            if (
                doctor_source_drop >= float(similarity_share) * master_source_drop
                and doctor_target_increase >= float(similarity_share) * master_source_drop
            ):
                guarded_row_ids.append(int(master_row_id))
                break

    if not guarded_row_ids:
        return out

    return out.drop(index=sorted(set(guarded_row_ids))).reset_index(drop=True)


def normalize_degree_type(label: object) -> str:
    text = _normalize_text(label)
    if not text:
        return "Other"
    compact = text.replace(" ", "")
    if any(token in text for token in ("phd", "ph d", "doctor", "doctoral", "jd", "md", "edd", "dba")):
        return "Doctor"
    if compact in {"ma", "ms", "mba", "meng", "mpp", "mph", "mpa", "mfin", "msc", "macc"}:
        return "Master"
    if any(
        token in text
        for token in (
            "master",
            "masters",
            "m a",
            "m s",
            "ms ",
            "ma ",
            "mba",
            "meng",
            "m eng",
            "mpp",
            "mph",
            "mpa",
            "mfin",
        )
    ):
        return "Master"
    if compact in {"ba", "bs", "bba", "ab", "sb"}:
        return "Bachelor"
    if any(token in text for token in ("bachelor", "bachelors", "undergraduate", "undergrad", "b a", "b s", "ba ", "bs ")):
        return "Bachelor"
    return "Other"


def awlevels_for_degree_type(degree_type: str) -> list[int]:
    return [int(v) for v in DEGREE_TYPE_TO_AWLEVELS.get(str(degree_type), ())]


def degree_type_for_awlevel(awlevel: object) -> str:
    try:
        awlevel_int = int(awlevel)
    except (TypeError, ValueError):
        return "Other"
    return AWLEVEL_TO_DEGREE_TYPE.get(awlevel_int, "Other")


def infer_candidate_columns(columns: list[str]) -> dict[str, str | None]:
    normalized = {_normalize_header(col): col for col in columns}
    inferred: dict[str, str | None] = {}
    for target_col, aliases in CANDIDATE_COLUMN_ALIASES.items():
        match = None
        for alias in aliases:
            if alias in normalized:
                match = normalized[alias]
                break
        if match is None:
            for norm_name, original in normalized.items():
                if any(alias in norm_name for alias in aliases):
                    match = original
                    break
        inferred[target_col] = match
    required = ["candidate_school_name", "candidate_approx_year", "candidate_program_desc", "candidate_degree_label"]
    missing = [col for col in required if inferred.get(col) is None]
    if missing:
        raise ValueError(f"Could not infer required candidate columns: {missing}. Available columns: {columns}")
    return inferred


def _infer_optional_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {_normalize_header(col): col for col in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for norm_name, original in normalized.items():
        if any(alias in norm_name for alias in aliases):
            return original
    return None


def _read_candidate_file(candidate_path: Path) -> pd.DataFrame:
    suffix = candidate_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(candidate_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(candidate_path)
    if suffix == ".parquet":
        return pd.read_parquet(candidate_path)
    raise ValueError(f"Unsupported candidate file type: {candidate_path.suffix}")


def _normalize_candidate_frame(df: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    inferred = infer_candidate_columns(list(df.columns))
    out = pd.DataFrame(index=df.index)
    for target_col, source_col in inferred.items():
        if source_col is None:
            out[target_col] = pd.NA
        else:
            out[target_col] = df[source_col]

    out["candidate_source_file"] = source_name
    out["candidate_school_name_raw"] = out["candidate_school_name"].fillna("").astype(str).str.strip()
    out["candidate_school_name"] = out["candidate_school_name_raw"].map(_canonicalize_candidate_school_name)
    out["candidate_date_raw"] = out["candidate_approx_year"].fillna("").astype(str).str.strip()
    out["candidate_approx_year"] = out["candidate_approx_year"].map(_extract_candidate_year).astype("Int64")
    out["candidate_program_desc"] = out["candidate_program_desc"].fillna("").astype(str).str.strip()
    out["candidate_program_signature"] = out["candidate_program_desc"].map(_candidate_program_signature)
    out["candidate_degree_label"] = out["candidate_degree_label"].fillna("").astype(str).str.strip()
    out["candidate_notes"] = out["candidate_notes"].fillna("").astype(str).str.strip()
    out["candidate_degree_type"] = out["candidate_degree_label"].map(normalize_degree_type)
    out["candidate_major"] = out["candidate_program_desc"].map(_candidate_major)

    for target_col, aliases in RAW_CANDIDATE_METADATA_ALIASES.items():
        source_col = _infer_optional_column(list(df.columns), aliases)
        out[target_col] = df[source_col] if source_col is not None else pd.NA
        out[target_col] = out[target_col].fillna("").astype(str).str.strip()

    source_hints = out["candidate_initial_cip_raw"].map(_extract_cip6_hint)
    target_hints = out["candidate_target_cip_raw"].map(_extract_cip6_hint)
    out["candidate_source_cip6_hint"] = source_hints.map(lambda pair: pair[0])
    out["candidate_source_cip_status"] = source_hints.map(lambda pair: pair[1])
    out["candidate_target_cip6_hint"] = target_hints.map(lambda pair: pair[0])
    out["candidate_target_cip_status"] = target_hints.map(lambda pair: pair[1])
    out["candidate_confidence"] = out.apply(
        lambda row: _infer_candidate_confidence(
            row["candidate_notes"],
            row["candidate_source_cip_status"],
            row["candidate_target_cip_status"],
        ),
        axis=1,
    )
    cip_constraints = out.apply(_parse_candidate_cip_constraints, axis=1, result_type="expand")
    for column in ("candidate_source_cip_bin", "candidate_target_cip_bin", "candidate_pair_bin", "candidate_cip_parse_notes"):
        out[column] = cip_constraints[column]

    generated_ids = [f"{_slugify(Path(source_name).stem)}_{idx + 1}" for idx in range(len(out))]
    candidate_ids = out["candidate_id"].fillna("").astype(str).str.strip()
    candidate_ids = candidate_ids.mask(candidate_ids.isin({"", "nan", "None"}), pd.Series(generated_ids, index=out.index))
    out["candidate_id"] = candidate_ids
    return out.reset_index(drop=True)


def _llm_source_priority(source_name: object) -> int:
    return int(LLM_CANDIDATE_SOURCE_PRIORITY.get(str(source_name), 999))


def _drop_known_llm_duplicates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    deduped = candidates.copy()
    deduped["_source_priority"] = deduped["candidate_source_file"].map(_llm_source_priority)
    deduped["_notes_len"] = deduped["candidate_notes"].astype(str).str.len()
    deduped = deduped.sort_values(
        ["_source_priority", "candidate_confidence", "_notes_len", "candidate_id"],
        ascending=[True, True, False, True],
    )
    drop_index: set[int] = set()
    for rule in KNOWN_LLM_DUPLICATE_RULES:
        mask = pd.Series(True, index=deduped.index)
        for column, value in rule.items():
            mask &= deduped[column].eq(value)
        rule_rows = deduped[mask]
        if len(rule_rows) <= 1:
            continue
        drop_index.update(rule_rows.index.tolist()[1:])
    if drop_index:
        deduped = deduped.drop(index=sorted(drop_index))
    return deduped.drop(columns=["_source_priority", "_notes_len"]).reset_index(drop=True)


def load_external_candidates(candidate_path: str | Path) -> pd.DataFrame:
    candidate_path = Path(candidate_path)
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidate_path}")

    _progress(f"Loading external candidates from {candidate_path}")
    if candidate_path.is_dir():
        frames: list[pd.DataFrame] = []
        supported_paths = [path for path in sorted(candidate_path.iterdir()) if path.suffix.lower() in SUPPORTED_CANDIDATE_SUFFIXES]
        _progress(f"Found {len(supported_paths)} supported candidate file(s) in directory input")
        for path in supported_paths:
            raw = _read_candidate_file(path)
            _progress(f"Reading {path.name} with {len(raw):,} raw row(s)")
            frames.append(_normalize_candidate_frame(raw, source_name=path.name))
        if not frames:
            raise ValueError(f"No supported candidate files found in directory: {candidate_path}")
        out = pd.concat(frames, ignore_index=True, sort=False)
        pre_dedup_rows = len(out)
        out = _drop_known_llm_duplicates(out)
        dropped = pre_dedup_rows - len(out)
        if dropped:
            _progress(f"Dropped {dropped:,} known duplicate candidate row(s) during preprocessing")
        _progress(f"Prepared {len(out):,} cleaned external candidate row(s)")
        return out.reset_index(drop=True)

    if candidate_path.suffix.lower() not in SUPPORTED_CANDIDATE_SUFFIXES:
        raise ValueError(f"Unsupported candidate file type: {candidate_path.suffix}")
    raw = _read_candidate_file(candidate_path)
    _progress(f"Reading {candidate_path.name} with {len(raw):,} raw row(s)")
    out = _normalize_candidate_frame(raw, source_name=candidate_path.name).reset_index(drop=True)
    _progress(f"Prepared {len(out):,} cleaned external candidate row(s)")
    return out


def _jaro_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    left_len = len(left)
    right_len = len(right)
    match_distance = max(left_len, right_len) // 2 - 1
    left_matches = [False] * left_len
    right_matches = [False] * right_len

    matches = 0
    transpositions = 0
    for i, left_char in enumerate(left):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, right_len)
        for j in range(start, end):
            if right_matches[j] or left_char != right[j]:
                continue
            left_matches[i] = True
            right_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    j = 0
    for i in range(left_len):
        if not left_matches[i]:
            continue
        while not right_matches[j]:
            j += 1
        if left[i] != right[j]:
            transpositions += 1
        j += 1

    return (
        matches / left_len
        + matches / right_len
        + (matches - transpositions / 2.0) / matches
    ) / 3.0


def jaro_winkler_similarity(left: object, right: object, *, prefix_scale: float = 0.1) -> float:
    left_norm = _normalize_school_name(left)
    right_norm = _normalize_school_name(right)
    jaro = _jaro_similarity(left_norm, right_norm)
    prefix = 0
    for left_char, right_char in zip(left_norm[:4], right_norm[:4]):
        if left_char != right_char:
            break
        prefix += 1
    return jaro + prefix * prefix_scale * (1.0 - jaro)


def load_school_lookup(crosswalk_path: str | Path) -> pd.DataFrame:
    crosswalk_path = Path(crosswalk_path)
    if not crosswalk_path.exists():
        raise FileNotFoundError(f"IPEDS crosswalk not found: {crosswalk_path}")
    use_cols = ["UNITID", "instname", "ALIAS"]
    raw = pd.read_parquet(crosswalk_path, columns=use_cols)
    raw = raw.rename(columns={"UNITID": "unitid", "instname": "school_name", "ALIAS": "alias_ind"})
    raw["alias_ind"] = raw["alias_ind"].fillna(False).astype(bool)
    raw["unitid"] = pd.to_numeric(raw["unitid"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["unitid", "school_name"]).copy()
    raw["unitid"] = raw["unitid"].astype("int64")
    raw["school_name"] = raw["school_name"].astype(str).str.strip()
    raw["school_name_clean"] = raw["school_name"].map(_normalize_school_name)
    raw = raw[raw["school_name_clean"].ne("")].copy()
    raw = raw.sort_values(["unitid", "alias_ind", "school_name"])
    raw = raw.drop_duplicates(subset=["unitid", "school_name_clean"], keep="first")
    return raw.reset_index(drop=True)


def resolve_school_name(
    school_name: object,
    lookup: pd.DataFrame,
    *,
    min_jw: float = 0.93,
) -> dict[str, object]:
    clean = _normalize_school_name(school_name)
    if not clean:
        return {
            "matched_unitid": pd.NA,
            "school_match_method": "missing",
            "school_match_score": float("nan"),
            "school_match_name": pd.NA,
        }

    exact = lookup[lookup["school_name_clean"] == clean].copy()
    if not exact.empty:
        exact = exact.sort_values(["alias_ind", "school_name"])
        best = exact.iloc[0]
        return {
            "matched_unitid": int(best["unitid"]),
            "school_match_method": "exact_clean",
            "school_match_score": 1.0,
            "school_match_name": str(best["school_name"]),
        }

    prefix = clean[:3]
    candidates = lookup[lookup["school_name_clean"].str.startswith(prefix, na=False)].copy()
    if candidates.empty:
        candidates = lookup.copy()
    if candidates.empty:
        return {
            "matched_unitid": pd.NA,
            "school_match_method": "unmatched",
            "school_match_score": float("nan"),
            "school_match_name": pd.NA,
        }

    candidates["jw"] = candidates["school_name_clean"].map(lambda value: jaro_winkler_similarity(clean, value))
    candidates = candidates.sort_values(["jw", "alias_ind", "school_name"], ascending=[False, True, True])
    best = candidates.iloc[0]
    if float(best["jw"]) < min_jw:
        return {
            "matched_unitid": pd.NA,
            "school_match_method": "unmatched",
            "school_match_score": float(best["jw"]),
            "school_match_name": str(best["school_name"]),
        }
    return {
        "matched_unitid": int(best["unitid"]),
        "school_match_method": "jaro_winkler",
        "school_match_score": float(best["jw"]),
        "school_match_name": str(best["school_name"]),
    }


def resolve_candidate_schools(candidates: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        resolved = resolve_school_name(row.candidate_school_name, lookup)
        out_row = dict(row._asdict())
        out_row.update(resolved)
        rows.append(out_row)
    resolved_df = pd.DataFrame(rows)
    if not resolved_df.empty:
        exact_matches = int((resolved_df["school_match_method"] == "exact_clean").sum())
        fuzzy_matches = int((resolved_df["school_match_method"] == "jaro_winkler").sum())
        unmatched = int(resolved_df["matched_unitid"].isna().sum())
        _progress(
            "Resolved candidate schools: "
            f"{len(resolved_df):,} total, {exact_matches:,} exact, {fuzzy_matches:,} fuzzy, {unmatched:,} unmatched"
        )
    return resolved_df


def _load_ipeds_cip_map(ipeds_path: str | Path) -> dict[str, str]:
    ipeds_path = Path(ipeds_path)
    if not ipeds_path.exists():
        raise FileNotFoundError(f"IPEDS completions parquet not found: {ipeds_path}")
    labels = (
        pd.read_parquet(ipeds_path, columns=["cipcode", "cipcode_lab"])
        .dropna(subset=["cipcode"])
        .drop_duplicates(subset=["cipcode"])
    )
    labels["cip6"] = labels["cipcode"].astype(str).str.zfill(6)
    labels["cip_label"] = labels["cipcode_lab"].map(_clean_cip_label)
    return dict(zip(labels["cip6"], labels["cip_label"]))


def _safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.where(denominator > 0).div(denominator)


def _filter_relabel_analysis_window(
    df: pd.DataFrame,
    *,
    year_col: str = "year",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "relabel_year" in out.columns:
        relabel_year = pd.to_numeric(out["relabel_year"], errors="coerce")
        out = out[relabel_year.isna() | relabel_year.le(MAX_RELABEL_YEAR)].copy()
    if year_col in out.columns:
        original_year = pd.to_numeric(out[year_col], errors="coerce")
        out = out[
            original_year.isna()
            | original_year.between(ANALYSIS_ORIGINAL_YEAR_MIN, ANALYSIS_ORIGINAL_YEAR_MAX)
        ].copy()
    return out


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _ensure_ipeds_cost_view(
    con: ddb.DuckDBPyConnection,
    ipeds_cost_panel_path: str | Path = DEFAULT_IPEDS_COST_PANEL_PATH,
    tuition_col: str = DEFAULT_IPEDS_TUITION_COL,
) -> str:
    ipeds_cost_panel_path = Path(ipeds_cost_panel_path)
    if ipeds_cost_panel_path.exists():
        con.sql(
            "CREATE OR REPLACE TEMP VIEW ipeds_cost_raw AS "
            f"SELECT * FROM read_parquet('{_sql_literal(str(ipeds_cost_panel_path))}')"
        )
        col_map = {
            str(col).lower(): str(col)
            for col in [row[0] for row in con.sql("DESCRIBE ipeds_cost_raw").fetchall()]
        }
        tuition_col_lower = tuition_col.lower()
        if tuition_col_lower not in col_map:
            tuition_candidates = [
                col for col in col_map.keys() if col.startswith("tuition")
            ]
            if tuition_candidates:
                tuition_col = col_map[tuition_candidates[0]]
            else:
                raise ValueError(
                    f"Could not locate an IPEDS tuition column in {ipeds_cost_panel_path}; "
                    f"expected '{tuition_col}'."
                )
        else:
            tuition_col = col_map[tuition_col_lower]
    else:
        con.sql(
            "CREATE OR REPLACE TEMP VIEW ipeds_cost_raw AS "
            "SELECT CAST(NULL AS BIGINT) AS unitid, CAST(NULL AS INTEGER) AS year, "
            "CAST(NULL AS DOUBLE) AS tuition3, CAST(NULL AS DOUBLE) AS tuition7, "
            "CAST(NULL AS DOUBLE) AS fee3, CAST(NULL AS DOUBLE) AS fee7 WHERE FALSE"
        )
    return tuition_col


def _ipeds_cost_col_map(con: ddb.DuckDBPyConnection) -> dict[str, str]:
    return {
        str(col).lower(): str(col)
        for col in [row[0] for row in con.sql("DESCRIBE ipeds_cost_raw").fetchall()]
    }


def _resolve_ipeds_cost_column(
    con: ddb.DuckDBPyConnection,
    requested_col: str,
    *,
    fallback_prefix: str,
) -> str:
    col_map = _ipeds_cost_col_map(con)
    requested_lower = str(requested_col).lower()
    aliases = [requested_lower]
    if requested_lower.startswith("fees"):
        aliases.append("fee" + requested_lower.removeprefix("fees"))
    if requested_lower in {"fee", "fees"}:
        aliases.append("fee")
    for alias in aliases:
        if alias in col_map:
            return col_map[alias]
    candidates = [col for key, col in col_map.items() if key.startswith(fallback_prefix)]
    if candidates:
        return sorted(candidates)[0]
    raise ValueError(
        f"Could not locate an IPEDS {fallback_prefix} column; expected '{requested_col}'."
    )


def _resolve_ipeds_cost_columns_by_degree(
    con: ddb.DuckDBPyConnection,
    requested_cols: dict[str, str],
    *,
    fallback_prefix: str,
) -> dict[str, str]:
    return {
        degree_type: _resolve_ipeds_cost_column(
            con,
            requested_col,
            fallback_prefix=fallback_prefix,
        )
        for degree_type, requested_col in requested_cols.items()
    }


def _ipeds_degree_cost_expr(degree_sql: str, cols_by_degree: dict[str, str]) -> str:
    parts = [
        f"WHEN {degree_sql} = '{_sql_literal(degree_type)}' THEN TRY_CAST(ic.{col} AS DOUBLE)"
        for degree_type, col in cols_by_degree.items()
    ]
    fallback_col = cols_by_degree.get("Master") or next(iter(cols_by_degree.values()))
    return "CASE " + " ".join(parts) + f" ELSE TRY_CAST(ic.{fallback_col} AS DOUBLE) END"


def _money_sql_expr(column_sql: str) -> str:
    return (
        "TRY_CAST(NULLIF(regexp_replace("
        f"CAST({column_sql} AS VARCHAR), '[^0-9.-]', '', 'g'"
        "), '') AS DOUBLE)"
    )


def _ensure_ipeds_view(con: ddb.DuckDBPyConnection, ipeds_path: str | Path) -> None:
    ipeds_path = Path(ipeds_path)
    if not ipeds_path.exists():
        raise FileNotFoundError(f"IPEDS completions parquet not found: {ipeds_path}")
    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw AS SELECT * FROM read_parquet('{_sql_literal(str(ipeds_path))}')")


def _ensure_stem_opt_cip_view(
    con: ddb.DuckDBPyConnection,
    stem_opt_long_path: str | Path = DEFAULT_STEM_OPT_LONG_PATH,
) -> None:
    stem_opt_long_path = Path(stem_opt_long_path)
    if not stem_opt_long_path.exists():
        raise FileNotFoundError(f"STEM OPT CIP list not found: {stem_opt_long_path}")
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW stem_opt_cip_first_year AS
        SELECT
            LPAD(
                REGEXP_REPLACE(CAST(cip_code AS VARCHAR), '[^0-9]', '', 'g'),
                6,
                '0'
            ) AS cip6,
            MIN(CAST(list_year AS INTEGER)) AS first_stem_year
        FROM read_csv_auto('{_sql_literal(str(stem_opt_long_path))}', header=TRUE)
        WHERE cip_code IS NOT NULL
          AND list_year IS NOT NULL
        GROUP BY 1
        """
    )


def _ipeds_year_bounds(con: ddb.DuckDBPyConnection) -> tuple[int, int]:
    bounds = con.sql(
        """
        SELECT
            CAST(MIN(CAST(year AS INTEGER)) AS INTEGER) AS min_year,
            CAST(MAX(CAST(year AS INTEGER)) AS INTEGER) AS max_year
        FROM ipeds_raw
        WHERE unitid IS NOT NULL
          AND cipcode IS NOT NULL
          AND awlevel IS NOT NULL
        """
    ).df()
    if bounds.empty or bounds["min_year"].isna().all() or bounds["max_year"].isna().all():
        raise RuntimeError("Could not determine IPEDS year bounds.")
    return int(bounds.loc[0, "min_year"]), int(bounds.loc[0, "max_year"])


def _int_clause(values: list[int] | tuple[int, ...] | None, column: str) -> str:
    if not values:
        return ""
    joined = ", ".join(str(int(v)) for v in sorted({int(v) for v in values}))
    return f" AND CAST({column} AS INTEGER) IN ({joined})"


def _clamp_relabel_year_bounds(
    year_min: int | None,
    year_max: int | None,
    *,
    clamp_to_analysis_window: bool = True,
) -> tuple[int | None, int | None]:
    year_min_value = int(year_min) if year_min is not None else None
    year_max_value = int(year_max) if year_max is not None else int(MAX_RELABEL_YEAR)
    if clamp_to_analysis_window:
        year_max_value = min(year_max_value, int(MAX_RELABEL_YEAR))
    return year_min_value, year_max_value


def scan_ipeds_pair_candidates(
    con: ddb.DuckDBPyConnection,
    *,
    ipeds_path: str | Path = base.IPEDS_PATH,
    thresholds: dict[str, float] | None = None,
    unitids: list[int] | None = None,
    awlevels: list[int] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    keep_all_sources: bool = False,
    allowed_pair_configs: list[dict[str, object]] | None = None,
    clamp_to_analysis_window: bool = True,
) -> pd.DataFrame:
    thresholds = thresholds or STRICT_THRESHOLDS
    year_min, year_max = _clamp_relabel_year_bounds(
        year_min,
        year_max,
        clamp_to_analysis_window=clamp_to_analysis_window,
    )
    _ensure_ipeds_view(con, ipeds_path)
    data_min_year, data_max_year = _ipeds_year_bounds(con)
    scan_year_min = int(year_min if year_min is not None else data_min_year + 1)
    scan_year_max = int(year_max if year_max is not None else data_max_year)
    scan_year_min = max(scan_year_min, data_min_year + 1)
    scan_year_max = min(scan_year_max, data_max_year)
    if scan_year_min > scan_year_max:
        return pd.DataFrame()

    unitid_filter = _int_clause(unitids, "unitid")
    awlevel_filter = _int_clause(awlevels, "awlevel")
    lookback_years = int(thresholds["lookback_years"])
    lookahead_years = int(thresholds["lookahead_years"])
    allowed_pair_filter = ""
    allowed_pair_sql = _allowed_pair_configs_sql(
        allowed_pair_configs,
        source_column="d.source_cip6",
        target_column="s.cip6",
    )
    if allowed_pair_sql:
        allowed_pair_filter = f" AND {allowed_pair_sql}"

    final_filter = "WHERE rn_source = 1" if keep_all_sources else "WHERE rn_source = 1 AND rn_unit_year = 1"
    df = con.sql(
        f"""
        WITH eligible AS (
            SELECT
                CAST(unitid AS BIGINT) AS unitid,
                CAST(year AS INTEGER) AS year,
                CAST(awlevel AS INTEGER) AS awlevel,
                COALESCE(CAST(awlevel_group AS VARCHAR), 'Other') AS awlevel_group,
                LPAD(CAST(cipcode AS VARCHAR), 6, '0') AS cip6,
                CAST(ctotalt AS DOUBLE) AS ctotalt,
                CAST(cnralt AS DOUBLE) AS cnralt
            FROM ipeds_raw
            WHERE unitid IS NOT NULL
              AND cipcode IS NOT NULL
              AND awlevel IS NOT NULL
              AND CAST(year AS INTEGER) BETWEEN {data_min_year} AND {data_max_year}
              AND COALESCE(CAST(share_intl AS DOUBLE), 0) >= {thresholds["min_share_intl"]}
              {unitid_filter}
              {awlevel_filter}
        ),
        unit_awlevel_cip AS (
            SELECT DISTINCT unitid, awlevel, awlevel_group, cip6
            FROM eligible
        ),
        years AS (
            SELECT * FROM generate_series({data_min_year}, {data_max_year}) AS y(year)
        ),
        panel AS (
            SELECT
                u.unitid,
                u.awlevel,
                u.awlevel_group,
                u.cip6,
                y.year,
                COALESCE(e.ctotalt, 0) AS ctotalt,
                COALESCE(e.cnralt, 0) AS cnralt
            FROM unit_awlevel_cip u
            CROSS JOIN years y
            LEFT JOIN eligible e
              ON e.unitid = u.unitid
             AND e.awlevel = u.awlevel
             AND e.cip6 = u.cip6
             AND e.year = y.year
        ),
        stats AS (
            SELECT
                unitid,
                awlevel,
                awlevel_group,
                cip6,
                year,
                ctotalt,
                cnralt,
                LAG(ctotalt) OVER (
                    PARTITION BY unitid, awlevel, cip6
                    ORDER BY year
                ) AS prev_total,
                AVG(ctotalt) OVER (
                    PARTITION BY unitid, awlevel, cip6
                    ORDER BY year
                    ROWS BETWEEN {lookback_years} PRECEDING AND 1 PRECEDING
                ) AS prev_window_avg,
                AVG(ctotalt) OVER (
                    PARTITION BY unitid, awlevel, cip6
                    ORDER BY year
                    ROWS BETWEEN 1 FOLLOWING AND {lookahead_years} FOLLOWING
                ) AS post_window_avg
            FROM panel
        ),
        drops AS (
            SELECT
                unitid,
                awlevel,
                awlevel_group,
                year,
                cip6 AS source_cip6,
                ctotalt AS source_curr,
                COALESCE(prev_total, 0) AS source_prev,
                GREATEST(COALESCE(prev_total, 0) - ctotalt, 0) AS source_drop,
                CASE
                    WHEN COALESCE(prev_total, 0) > 0
                    THEN (COALESCE(prev_total, 0) - ctotalt) / prev_total
                    ELSE NULL
                END AS source_drop_pct,
                COALESCE(prev_window_avg, prev_total, 0) AS source_baseline,
                COALESCE(post_window_avg, ctotalt, 0) AS source_post
            FROM stats
            WHERE year BETWEEN {scan_year_min} AND {scan_year_max}
              AND COALESCE(prev_window_avg, prev_total, 0) >= {thresholds["min_source_baseline"]}
              AND GREATEST(COALESCE(prev_total, 0) - ctotalt, 0) >= {thresholds["min_source_drop_abs"]}
              AND CASE
                    WHEN COALESCE(prev_total, 0) > 0
                    THEN (COALESCE(prev_total, 0) - ctotalt) / prev_total
                    ELSE 0
                  END >= {thresholds["min_source_drop_pct"]}
              AND COALESCE(post_window_avg, ctotalt, 0)
                  <= COALESCE(prev_window_avg, prev_total, 0) * (1 - {thresholds["source_persistence_drop_share"]})
        ),
        paired AS (
            SELECT
                d.unitid,
                d.awlevel,
                d.awlevel_group,
                d.year,
                d.source_cip6,
                s.cip6 AS target_cip6,
                d.source_curr AS source_total,
                d.source_prev AS source_total_prev,
                d.source_curr AS source_total_intl,
                d.source_prev AS source_total_intl_prev,
                s.ctotalt AS target_total,
                COALESCE(s.prev_total, 0) AS target_total_prev,
                s.ctotalt AS target_total_intl,
                COALESCE(s.prev_total, 0) AS target_total_intl_prev,
                d.source_drop,
                d.source_drop_pct,
                s.ctotalt - COALESCE(s.prev_total, 0) AS target_increase,
                CASE
                    WHEN COALESCE(s.prev_total, 0) > 0
                    THEN (s.ctotalt - COALESCE(s.prev_total, 0)) / s.prev_total
                    ELSE NULL
                END AS target_increase_pct,
                d.source_baseline,
                COALESCE(s.prev_window_avg, s.prev_total, 0) AS target_baseline,
                d.source_post - d.source_baseline AS avg5_source_drop,
                CASE
                    WHEN d.source_baseline > 0
                    THEN (d.source_post - d.source_baseline) / d.source_baseline
                    ELSE NULL
                END AS avg5_source_drop_pct,
                COALESCE(s.post_window_avg, s.ctotalt, 0) - COALESCE(s.prev_window_avg, s.prev_total, 0) AS avg5_target_increase,
                CASE
                    WHEN COALESCE(s.prev_window_avg, s.prev_total, 0) > 0
                    THEN (
                        COALESCE(s.post_window_avg, s.ctotalt, 0) - COALESCE(s.prev_window_avg, s.prev_total, 0)
                    ) / COALESCE(s.prev_window_avg, s.prev_total, 0)
                    ELSE NULL
                END AS avg5_target_increase_pct,
                (
                    COALESCE(
                        (s.ctotalt - COALESCE(s.prev_total, 0)) / NULLIF(d.source_drop, 0),
                        0
                    )
                    + 0.5 * COALESCE(
                        (
                            COALESCE(s.post_window_avg, s.ctotalt, 0)
                            - COALESCE(s.prev_window_avg, s.prev_total, 0)
                        ) / NULLIF(d.source_drop, 0),
                        0
                    )
                    + 0.5 * COALESCE(
                        (
                            d.source_baseline - d.source_post
                        ) / NULLIF(d.source_drop, 0),
                        0
                    )
                    - 0.5 * COALESCE(
                        (
                            -LEAST(
                                (d.source_curr + s.ctotalt) - (d.source_prev + COALESCE(s.prev_total, 0)),
                                0
                            )
                        ) / NULLIF(d.source_drop, 0),
                        0
                    )
                ) AS relabel_score,
                ROW_NUMBER() OVER (
                    PARTITION BY d.unitid, d.awlevel, d.year, d.source_cip6
                    ORDER BY
                        (
                            COALESCE(
                                (s.ctotalt - COALESCE(s.prev_total, 0)) / NULLIF(d.source_drop, 0),
                                0
                            )
                            + 0.5 * COALESCE(
                                (
                                    COALESCE(s.post_window_avg, s.ctotalt, 0)
                                    - COALESCE(s.prev_window_avg, s.prev_total, 0)
                                ) / NULLIF(d.source_drop, 0),
                                0
                            )
                            + 0.5 * COALESCE(
                                (
                                    d.source_baseline - d.source_post
                                ) / NULLIF(d.source_drop, 0),
                                0
                            )
                            - 0.5 * COALESCE(
                                (
                                    -LEAST(
                                        (d.source_curr + s.ctotalt) - (d.source_prev + COALESCE(s.prev_total, 0)),
                                        0
                                    )
                                ) / NULLIF(d.source_drop, 0),
                                0
                            )
                        ) DESC,
                        s.ctotalt - COALESCE(s.prev_total, 0) DESC,
                        s.cip6
                ) AS rn_source
            FROM drops d
            JOIN stats s
              ON d.unitid = s.unitid
             AND d.awlevel = s.awlevel
             AND d.year = s.year
             AND d.source_cip6 <> s.cip6
            WHERE s.ctotalt - COALESCE(s.prev_total, 0) >= {thresholds["min_target_offset_share"]} * d.source_drop
              AND (d.source_curr + s.ctotalt) - (d.source_prev + COALESCE(s.prev_total, 0))
                  >= -{thresholds["max_net_loss_share"]} * d.source_drop
              AND COALESCE(s.post_window_avg, s.ctotalt, 0)
                  >= COALESCE(s.prev_window_avg, s.prev_total, 0)
                     + {thresholds["target_persistence_gain_share"]} * d.source_drop
              {allowed_pair_filter}
        ),
        best_pairs AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY unitid, awlevel, year
                    ORDER BY source_drop DESC, relabel_score DESC, target_increase DESC, target_cip6
                ) AS rn_unit_year
            FROM paired
        )
        SELECT
            unitid,
            awlevel,
            awlevel_group,
            year,
            source_cip6,
            target_cip6,
            source_total,
            source_total_prev,
            source_total_intl,
            source_total_intl_prev,
            target_total,
            target_total_prev,
            target_total_intl,
            target_total_intl_prev,
            source_drop,
            source_drop_pct,
            target_increase,
            target_increase_pct,
            source_baseline,
            target_baseline,
            avg5_source_drop,
            avg5_source_drop_pct,
            avg5_target_increase,
            avg5_target_increase_pct,
            relabel_score,
            rn_source,
            rn_unit_year
        FROM best_pairs
        {final_filter}
        """
    ).df()

    if df.empty:
        return df
    df["relabel_year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["relabel_type"] = df["source_cip6"].astype(str) + "_to_" + df["target_cip6"].astype(str)
    df["degree_type"] = df["awlevel"].map(degree_type_for_awlevel)
    return df.reset_index(drop=True)


def detect_ipeds_relabels(
    con: ddb.DuckDBPyConnection,
    *,
    ipeds_path: str | Path = base.IPEDS_PATH,
    thresholds: dict[str, float] | None = None,
    allowed_pair_configs: list[dict[str, object]] | None = None,
    relabel_year_mode: str = DEFAULT_RELABEL_YEAR_MODE,
    year_min: int | None = None,
    year_max: int | None = None,
    clamp_to_analysis_window: bool = True,
) -> pd.DataFrame:
    relabel_year_mode = _normalize_relabel_year_mode(relabel_year_mode)
    events = scan_ipeds_pair_candidates(
        con,
        ipeds_path=ipeds_path,
        thresholds=thresholds or STRICT_THRESHOLDS,
        keep_all_sources=False,
        year_min=year_min,
        year_max=MAX_RELABEL_YEAR if year_max is None and clamp_to_analysis_window else year_max,
        allowed_pair_configs=allowed_pair_configs,
        clamp_to_analysis_window=clamp_to_analysis_window,
    )
    if events.empty:
        return events
    events = _apply_master_phd_guard(con, events, ipeds_path=ipeds_path)
    if events.empty:
        return events
    if relabel_year_mode == "first":
        events = events.sort_values(
            ["unitid", "awlevel", "source_cip6", "target_cip6", "year", "relabel_score"],
            ascending=[True, True, True, True, True, False],
        )
    else:
        events = events.sort_values(
            ["unitid", "awlevel", "source_cip6", "target_cip6", "relabel_score", "year"],
            ascending=[True, True, True, True, False, True],
        )
    events = events.drop_duplicates(
        subset=["unitid", "awlevel", "source_cip6", "target_cip6"],
        keep="first",
    )
    events = events.reset_index(drop=True)
    events = events.rename(columns={"source_cip6": "event_source_cip6"})
    events["found_in_ipeds_scan"] = 1
    events["found_in_external_candidates"] = 0
    events["external_verified"] = 0
    events["event_origin_category"] = "ipeds_only"
    events["year"] = events["relabel_year"]
    return events


def _text_similarity(query: object, source_label: object, target_label: object) -> float:
    query_norm = _normalize_text(query)
    if not query_norm:
        return 0.0
    candidates = [
        _normalize_text(source_label),
        _normalize_text(target_label),
        _normalize_text(f"{source_label} {target_label}"),
    ]
    best = 0.0
    for candidate in candidates:
        if not candidate:
            continue
        left_tokens = set(query_norm.split())
        right_tokens = set(candidate.split())
        overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
        seq = _jaro_similarity(query_norm, candidate)
        best = max(best, 0.5 * overlap + 0.5 * seq)
    return best


def _rank_candidate_matches(
    pairs: pd.DataFrame,
    *,
    approx_year: int | None,
    program_desc: object,
    cip_map: dict[str, str],
) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    ranked = pairs.copy()
    ranked["source_cip_label"] = ranked["source_cip6"].map(cip_map).fillna("")
    ranked["target_cip_label"] = ranked["target_cip6"].map(cip_map).fillna("")
    if approx_year is None or pd.isna(approx_year):
        ranked["best_year_distance"] = pd.NA
        ranked["year_bonus"] = 0.0
    else:
        ranked["best_year_distance"] = (pd.to_numeric(ranked["year"], errors="coerce") - int(approx_year)).abs()
        ranked["year_bonus"] = ranked["best_year_distance"].map(lambda value: max(0.0, 1.0 - float(value) / 3.0) * 0.15)
    ranked["best_text_similarity"] = ranked.apply(
        lambda row: _text_similarity(program_desc, row["source_cip_label"], row["target_cip_label"]),
        axis=1,
    )
    ranked["text_bonus"] = ranked["best_text_similarity"] * 0.25
    ranked["best_candidate_rank_score"] = pd.to_numeric(ranked["relabel_score"], errors="coerce").fillna(0.0)
    ranked["best_candidate_rank_score"] += ranked["year_bonus"] + ranked["text_bonus"]
    return ranked.sort_values(
        ["best_candidate_rank_score", "relabel_score", "source_drop", "target_increase"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _diagnostic_best_match(
    con: ddb.DuckDBPyConnection,
    *,
    unitid: int,
    awlevels: list[int],
    year_min: int,
    year_max: int,
    cip_map: dict[str, str],
    candidate_program_desc: object,
    candidate_approx_year: int | None,
    ipeds_path: str | Path,
) -> dict[str, object]:
    year_min, year_max = _clamp_relabel_year_bounds(year_min, year_max)
    diagnostic_pairs = scan_ipeds_pair_candidates(
        con,
        ipeds_path=ipeds_path,
        thresholds=DIAGNOSTIC_THRESHOLDS,
        unitids=[unitid],
        awlevels=awlevels,
        year_min=year_min,
        year_max=year_max,
        keep_all_sources=True,
    )
    if 7 in awlevels:
        diagnostic_pairs = _apply_master_phd_guard(con, diagnostic_pairs, ipeds_path=ipeds_path)
    if diagnostic_pairs.empty:
        return {
            "diagnostic_best_year": pd.NA,
            "diagnostic_best_source_cip6": pd.NA,
            "diagnostic_best_target_cip6": pd.NA,
            "diagnostic_best_score": float("nan"),
        }
    ranked = _rank_candidate_matches(
        diagnostic_pairs,
        approx_year=candidate_approx_year,
        program_desc=candidate_program_desc,
        cip_map=cip_map,
    )
    best = ranked.iloc[0]
    return {
        "diagnostic_best_year": int(best["year"]),
        "diagnostic_best_source_cip6": str(best["source_cip6"]),
        "diagnostic_best_target_cip6": str(best["target_cip6"]),
        "diagnostic_best_score": float(best["best_candidate_rank_score"]),
    }


def verify_external_candidates(
    con: ddb.DuckDBPyConnection,
    resolved_candidates: pd.DataFrame,
    *,
    ipeds_path: str | Path = base.IPEDS_PATH,
    relaxed_thresholds: dict[str, float] | None = None,
    cip_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cip_map = cip_map or _load_ipeds_cip_map(ipeds_path)
    relaxed_thresholds = relaxed_thresholds or RELAXED_THRESHOLDS
    verified_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    total_candidates = len(resolved_candidates)
    _progress(f"Starting IPEDS verification for {total_candidates:,} external candidate(s)")

    for idx, row in enumerate(resolved_candidates.itertuples(index=False), start=1):
        unitid = None if pd.isna(row.matched_unitid) else int(row.matched_unitid)
        degree_type = str(row.candidate_degree_type)
        awlevels = awlevels_for_degree_type(degree_type)
        approx_year = None if pd.isna(row.candidate_approx_year) else int(row.candidate_approx_year)
        year_min = approx_year - 3 if approx_year is not None else None
        year_max = approx_year + 3 if approx_year is not None else None
        year_min, year_max = _clamp_relabel_year_bounds(year_min, year_max)
        cip_constraints = _parse_candidate_cip_constraints(row._asdict())
        source_rule = cip_constraints["_candidate_source_rule"]
        target_rule = cip_constraints["_candidate_target_rule"]

        if total_candidates <= 10 or idx == 1 or idx == total_candidates or idx % 10 == 0:
            _progress(
                f"Verifying candidate {idx:,}/{total_candidates:,}: "
                f"{row.candidate_school_name} | {row.candidate_degree_type} | "
                f"{row.candidate_program_desc} | approx_year={approx_year}"
            )

        audit: dict[str, object] = dict(row._asdict())
        audit["candidate_source_cip_bin"] = cip_constraints["candidate_source_cip_bin"]
        audit["candidate_target_cip_bin"] = cip_constraints["candidate_target_cip_bin"]
        audit["candidate_pair_bin"] = cip_constraints["candidate_pair_bin"]
        audit["candidate_cip_parse_notes"] = cip_constraints["candidate_cip_parse_notes"]
        audit["external_verified"] = 0
        audit["verification_notes"] = ""
        audit["best_candidate_rank_score"] = float("nan")
        audit["best_year_distance"] = pd.NA
        audit["best_text_similarity"] = float("nan")
        audit["best_nearby_year"] = pd.NA
        audit["best_nearby_source_cip6"] = pd.NA
        audit["best_nearby_target_cip6"] = pd.NA
        audit["best_nearby_relabel_score"] = float("nan")
        audit.update(
            {
                "diagnostic_best_year": pd.NA,
                "diagnostic_best_source_cip6": pd.NA,
                "diagnostic_best_target_cip6": pd.NA,
                "diagnostic_best_score": float("nan"),
            }
        )

        if unitid is None:
            audit["verification_notes"] = "school_unmatched"
            audit_rows.append(audit)
            continue
        if not awlevels:
            audit["verification_notes"] = "degree_unmappable"
            audit_rows.append(audit)
            continue
        if source_rule is None or target_rule is None:
            audit["verification_notes"] = "candidate_cip_bins_unparsed"
            audit_rows.append(audit)
            continue

        nearby_pairs = scan_ipeds_pair_candidates(
            con,
            ipeds_path=ipeds_path,
            thresholds=relaxed_thresholds,
            unitids=[unitid],
            awlevels=awlevels,
            year_min=year_min,
            year_max=year_max,
            keep_all_sources=True,
        )
        if degree_type == "Master" and not nearby_pairs.empty:
            nearby_pairs = _apply_master_phd_guard(con, nearby_pairs, ipeds_path=ipeds_path)
        if nearby_pairs.empty:
            audit["verification_notes"] = "no_relaxed_ipeds_match"
            audit.update(
                _diagnostic_best_match(
                    con,
                    unitid=unitid,
                    awlevels=awlevels,
                    year_min=year_min if year_min is not None else 0,
                    year_max=year_max if year_max is not None else MAX_RELABEL_YEAR,
                    cip_map=cip_map,
                    candidate_program_desc=row.candidate_program_desc,
                    candidate_approx_year=approx_year,
                    ipeds_path=ipeds_path,
                )
            )
            audit_rows.append(audit)
            continue

        matching_pairs = _filter_pairs_for_candidate_constraints(
            nearby_pairs,
            source_rule=source_rule,
            target_rule=target_rule,
        )
        if matching_pairs.empty:
            audit["verification_notes"] = "no_relaxed_ipeds_match_for_candidate_cip_bins"
            audit.update(
                _diagnostic_best_match(
                    con,
                    unitid=unitid,
                    awlevels=awlevels,
                    year_min=year_min if year_min is not None else 0,
                    year_max=year_max if year_max is not None else MAX_RELABEL_YEAR,
                    cip_map=cip_map,
                    candidate_program_desc=row.candidate_program_desc,
                    candidate_approx_year=approx_year,
                    ipeds_path=ipeds_path,
                )
            )
            audit_rows.append(audit)
            continue

        ranked = _rank_candidate_matches(
            matching_pairs,
            approx_year=approx_year,
            program_desc=row.candidate_program_desc,
            cip_map=cip_map,
        )
        best = ranked.iloc[0]
        audit["external_verified"] = 1
        audit["verification_notes"] = "verified_relaxed_ipeds"
        audit["best_candidate_rank_score"] = float(best["best_candidate_rank_score"])
        audit["best_year_distance"] = best["best_year_distance"]
        audit["best_text_similarity"] = float(best["best_text_similarity"])
        audit["best_nearby_year"] = int(best["year"])
        audit["best_nearby_source_cip6"] = str(best["source_cip6"])
        audit["best_nearby_target_cip6"] = str(best["target_cip6"])
        audit["best_nearby_relabel_score"] = float(best["relabel_score"])
        audit.update(
            {
                "diagnostic_best_year": int(best["year"]),
                "diagnostic_best_source_cip6": str(best["source_cip6"]),
                "diagnostic_best_target_cip6": str(best["target_cip6"]),
                "diagnostic_best_score": float(best["best_candidate_rank_score"]),
            }
        )
        audit_rows.append(audit)

        verified = best.to_dict()
        verified.update(
            {
                "unitid": unitid,
                "relabel_year": int(best["year"]),
                "year": int(best["year"]),
                "event_source_cip6": str(best["source_cip6"]),
                "candidate_id": row.candidate_id,
                "candidate_school_name": row.candidate_school_name,
                "candidate_approx_year": row.candidate_approx_year,
                "candidate_program_desc": row.candidate_program_desc,
                "candidate_degree_label": row.candidate_degree_label,
                "candidate_notes": row.candidate_notes,
                "candidate_major": row.candidate_major,
                "candidate_source_cip_bin": audit["candidate_source_cip_bin"],
                "candidate_target_cip_bin": audit["candidate_target_cip_bin"],
                "candidate_pair_bin": audit["candidate_pair_bin"],
                "candidate_cip_parse_notes": audit["candidate_cip_parse_notes"],
                "school_match_method": row.school_match_method,
                "school_match_score": row.school_match_score,
                "school_match_name": row.school_match_name,
                "verification_notes": audit["verification_notes"],
                "best_candidate_rank_score": audit["best_candidate_rank_score"],
                "best_year_distance": audit["best_year_distance"],
                "best_text_similarity": audit["best_text_similarity"],
                "best_nearby_year": audit["best_nearby_year"],
                "best_nearby_source_cip6": audit["best_nearby_source_cip6"],
                "best_nearby_target_cip6": audit["best_nearby_target_cip6"],
                "best_nearby_relabel_score": audit["best_nearby_relabel_score"],
                "diagnostic_best_year": audit["diagnostic_best_year"],
                "diagnostic_best_source_cip6": audit["diagnostic_best_source_cip6"],
                "diagnostic_best_target_cip6": audit["diagnostic_best_target_cip6"],
                "diagnostic_best_score": audit["diagnostic_best_score"],
                "found_in_external_candidates": 1,
                "external_verified": 1,
            }
        )
        verified_rows.append(verified)

    verified_df = pd.DataFrame(verified_rows)
    audit_df = pd.DataFrame(audit_rows)
    verified_count = int((audit_df.get("external_verified", pd.Series(dtype=int)) == 1).sum()) if not audit_df.empty else 0
    _progress(f"Finished relaxed verification: {verified_count:,} verified, {len(audit_df) - verified_count:,} not verified")
    return verified_df, audit_df


def _event_key(row: pd.Series | dict[str, object]) -> tuple[object, ...]:
    unit_value = row["unitid"] if "unitid" in row else row.get("matched_unitid")
    source_value = row["event_source_cip6"] if "event_source_cip6" in row else row.get("source_cip6")
    return (
        int(pd.to_numeric(unit_value, errors="coerce")),
        int(pd.to_numeric(row["awlevel"], errors="coerce")),
        int(pd.to_numeric(row["relabel_year"], errors="coerce")),
        str(source_value),
        str(row["target_cip6"]),
    )


def _concat_unique(values: list[object]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return " | ".join(out)


def _optional_concat_unique(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return ""
    return _concat_unique(df[column].tolist())


def _aggregate_candidate_group(candidate_group: pd.DataFrame) -> dict[str, object]:
    return {
        "candidate_id": _concat_unique(candidate_group["candidate_id"].tolist()),
        "candidate_school_name": _concat_unique(candidate_group["candidate_school_name"].tolist()),
        "candidate_approx_year": _concat_unique(candidate_group["candidate_approx_year"].astype(str).tolist()),
        "candidate_program_desc": _concat_unique(candidate_group["candidate_program_desc"].tolist()),
        "candidate_degree_label": _concat_unique(candidate_group["candidate_degree_label"].tolist()),
        "candidate_notes": _concat_unique(candidate_group["candidate_notes"].tolist()),
        "candidate_major": _concat_unique(candidate_group["candidate_major"].tolist()),
        "candidate_source_cip_bin": _optional_concat_unique(candidate_group, "candidate_source_cip_bin"),
        "candidate_target_cip_bin": _optional_concat_unique(candidate_group, "candidate_target_cip_bin"),
        "candidate_pair_bin": _optional_concat_unique(candidate_group, "candidate_pair_bin"),
        "candidate_cip_parse_notes": _optional_concat_unique(candidate_group, "candidate_cip_parse_notes"),
        "n_linked_candidates": int(candidate_group["candidate_id"].nunique()),
        "school_match_method": _concat_unique(candidate_group["school_match_method"].tolist()),
        "school_match_score": float(pd.to_numeric(candidate_group["school_match_score"], errors="coerce").max()),
        "school_match_name": _concat_unique(candidate_group["school_match_name"].tolist()),
        "verification_notes": _concat_unique(candidate_group["verification_notes"].tolist()),
        "best_candidate_rank_score": float(pd.to_numeric(candidate_group["best_candidate_rank_score"], errors="coerce").max()),
        "best_year_distance": pd.to_numeric(candidate_group["best_year_distance"], errors="coerce").min(),
        "best_text_similarity": float(pd.to_numeric(candidate_group["best_text_similarity"], errors="coerce").max()),
        "best_nearby_year": pd.to_numeric(candidate_group["best_nearby_year"], errors="coerce").min(),
        "best_nearby_source_cip6": _concat_unique(candidate_group["best_nearby_source_cip6"].tolist()),
        "best_nearby_target_cip6": _concat_unique(candidate_group["best_nearby_target_cip6"].tolist()),
        "best_nearby_relabel_score": float(pd.to_numeric(candidate_group["best_nearby_relabel_score"], errors="coerce").max()),
        "diagnostic_best_year": pd.to_numeric(candidate_group["diagnostic_best_year"], errors="coerce").min(),
        "diagnostic_best_source_cip6": _concat_unique(candidate_group["diagnostic_best_source_cip6"].tolist()),
        "diagnostic_best_target_cip6": _concat_unique(candidate_group["diagnostic_best_target_cip6"].tolist()),
        "diagnostic_best_score": float(pd.to_numeric(candidate_group["diagnostic_best_score"], errors="coerce").max()),
    }


def _apply_event_origin_category(row: pd.Series) -> str:
    if int(row.get("found_in_external_candidates", 0) or 0) and int(row.get("external_verified", 0) or 0):
        return "external_ipeds_verified"
    if int(row.get("found_in_external_candidates", 0) or 0):
        return "external_only"
    return "ipeds_only"


def merge_event_sources(
    strict_events: pd.DataFrame,
    verified_external_events: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    *,
    cip_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    cip_map = cip_map or {}
    base_events = strict_events.copy()
    if not base_events.empty:
        base_events["found_in_ipeds_scan"] = 1
        base_events["found_in_external_candidates"] = 0
        base_events["external_verified"] = 0
        base_events["event_origin_category"] = "ipeds_only"

    verified_external = verified_external_events.copy()
    if not verified_external.empty:
        verified_external["found_in_ipeds_scan"] = 0
        verified_external["found_in_external_candidates"] = 1
        verified_external["external_verified"] = 1
        verified_external["event_origin_category"] = "external_ipeds_verified"

    merged_map: dict[tuple[object, ...], dict[str, object]] = {}
    for df in (base_events, verified_external):
        if df.empty:
            continue
        for _, row in df.iterrows():
            key = _event_key(row)
            record = merged_map.get(key, {})
            for column, value in row.items():
                if column not in record or pd.isna(record[column]) or record[column] in {"", 0}:
                    record[column] = value
            record["found_in_ipeds_scan"] = int(record.get("found_in_ipeds_scan", 0) or 0) | int(row.get("found_in_ipeds_scan", 0) or 0)
            record["found_in_external_candidates"] = int(record.get("found_in_external_candidates", 0) or 0) | int(row.get("found_in_external_candidates", 0) or 0)
            record["external_verified"] = int(record.get("external_verified", 0) or 0) | int(row.get("external_verified", 0) or 0)
            merged_map[key] = record

    verified_rows = list(merged_map.values())
    verified_df = pd.DataFrame(verified_rows)
    if not verified_df.empty and not candidate_audit.empty:
        verified_audit = candidate_audit[candidate_audit["external_verified"] == 1].copy()
        if not verified_audit.empty:
            verified_audit["awlevel"] = verified_audit["candidate_degree_type"].map(
                lambda value: awlevels_for_degree_type(str(value))[0] if awlevels_for_degree_type(str(value)) else pd.NA
            )
            verified_audit["relabel_year"] = pd.to_numeric(verified_audit["best_nearby_year"], errors="coerce").astype("Int64")
            verified_audit["event_source_cip6"] = verified_audit["best_nearby_source_cip6"]
            verified_audit["target_cip6"] = verified_audit["best_nearby_target_cip6"]
            grouped_candidates = {
                (
                    int(pd.to_numeric(group.iloc[0]["matched_unitid"], errors="coerce")),
                    int(pd.to_numeric(group.iloc[0]["awlevel"], errors="coerce")),
                    int(pd.to_numeric(group.iloc[0]["relabel_year"], errors="coerce")),
                    str(group.iloc[0]["event_source_cip6"]),
                    str(group.iloc[0]["target_cip6"]),
                ): _aggregate_candidate_group(group)
                for _, group in verified_audit.groupby(
                    ["matched_unitid", "awlevel", "relabel_year", "event_source_cip6", "target_cip6"],
                    dropna=False,
                )
            }
            for idx, row in verified_df.iterrows():
                key = _event_key(row)
                extras = grouped_candidates.get(key)
                if extras:
                    for column, value in extras.items():
                        if column in verified_df.columns and verified_df[column].dtype != "object":
                            verified_df[column] = verified_df[column].astype("object")
                        verified_df.loc[idx, column] = value

    external_only_rows: list[dict[str, object]] = []
    if not candidate_audit.empty:
        for row in candidate_audit[candidate_audit["external_verified"] != 1].itertuples(index=False):
            external_only_rows.append(
                {
                    "unitid": row.matched_unitid,
                    "awlevel": pd.NA,
                    "degree_type": row.candidate_degree_type,
                    "relabel_year": pd.NA,
                    "year": pd.NA,
                    "relabel_type": pd.NA,
                    "event_source_cip6": pd.NA,
                    "target_cip6": pd.NA,
                    "source_total": pd.NA,
                    "source_total_prev": pd.NA,
                    "source_total_intl": pd.NA,
                    "source_total_intl_prev": pd.NA,
                    "target_total": pd.NA,
                    "target_total_prev": pd.NA,
                    "target_total_intl": pd.NA,
                    "target_total_intl_prev": pd.NA,
                    "ctotalt": pd.NA,
                    "cnralt": pd.NA,
                    "source_drop": pd.NA,
                    "source_drop_pct": pd.NA,
                    "target_increase": pd.NA,
                    "target_increase_pct": pd.NA,
                    "avg5_source_drop": pd.NA,
                    "avg5_source_drop_pct": pd.NA,
                    "avg5_target_increase": pd.NA,
                    "avg5_target_increase_pct": pd.NA,
                    "source_baseline": pd.NA,
                    "target_baseline": pd.NA,
                    "relabel_score": pd.NA,
                    "found_in_ipeds_scan": 0,
                    "found_in_external_candidates": 1,
                    "external_verified": 0,
                    "event_origin_category": "external_only",
                    "source_cip_label": pd.NA,
                    "target_cip_label": pd.NA,
                    "source_major": pd.NA,
                    "target_major": pd.NA,
                    "event_flag": 0,
                    "relabel_flag": 0,
                    "candidate_id": row.candidate_id,
                    "candidate_school_name": row.candidate_school_name,
                    "candidate_approx_year": row.candidate_approx_year,
                    "candidate_program_desc": row.candidate_program_desc,
                    "candidate_degree_label": row.candidate_degree_label,
                    "candidate_notes": row.candidate_notes,
                    "candidate_major": row.candidate_major,
                    "candidate_source_cip_bin": getattr(row, "candidate_source_cip_bin", pd.NA),
                    "candidate_target_cip_bin": getattr(row, "candidate_target_cip_bin", pd.NA),
                    "candidate_pair_bin": getattr(row, "candidate_pair_bin", pd.NA),
                    "candidate_cip_parse_notes": getattr(row, "candidate_cip_parse_notes", pd.NA),
                    "n_linked_candidates": 1,
                    "school_match_method": row.school_match_method,
                    "school_match_score": row.school_match_score,
                    "school_match_name": row.school_match_name,
                    "verification_notes": row.verification_notes,
                    "best_candidate_rank_score": row.best_candidate_rank_score,
                    "best_year_distance": row.best_year_distance,
                    "best_text_similarity": row.best_text_similarity,
                    "best_nearby_year": row.best_nearby_year,
                    "best_nearby_source_cip6": row.best_nearby_source_cip6,
                    "best_nearby_target_cip6": row.best_nearby_target_cip6,
                    "best_nearby_relabel_score": row.best_nearby_relabel_score,
                    "diagnostic_best_year": row.diagnostic_best_year,
                    "diagnostic_best_source_cip6": row.diagnostic_best_source_cip6,
                    "diagnostic_best_target_cip6": row.diagnostic_best_target_cip6,
                    "diagnostic_best_score": row.diagnostic_best_score,
                }
            )

    merged = verified_df
    if external_only_rows:
        merged = pd.concat([merged, pd.DataFrame(external_only_rows)], ignore_index=True, sort=False)

    if merged.empty:
        merged = annotate_event_broad_bins(merged)
        for column in VERIFIED_EVENT_COLUMNS:
            if column not in merged.columns:
                merged[column] = pd.NA
        return merged.loc[:, VERIFIED_EVENT_COLUMNS].copy()
    merged["event_origin_category"] = merged.apply(_apply_event_origin_category, axis=1)
    existing_source_labels = merged["source_cip_label"] if "source_cip_label" in merged.columns else pd.Series(pd.NA, index=merged.index)
    existing_target_labels = merged["target_cip_label"] if "target_cip_label" in merged.columns else pd.Series(pd.NA, index=merged.index)
    merged["source_cip_label"] = merged["event_source_cip6"].map(cip_map).fillna(existing_source_labels)
    merged["target_cip_label"] = merged["target_cip6"].map(cip_map).fillna(existing_target_labels)
    merged["source_major"] = merged["source_cip_label"].map(lambda value: _clean_cip_label(value) if pd.notna(value) else value)
    merged["target_major"] = merged["target_cip_label"].map(lambda value: _clean_cip_label(value) if pd.notna(value) else value)
    merged["degree_type"] = merged.apply(
        lambda row: row["degree_type"] if pd.notna(row["degree_type"]) and str(row["degree_type"]) != "nan" else degree_type_for_awlevel(row["awlevel"]),
        axis=1,
    )
    merged["year"] = merged["year"].fillna(merged["relabel_year"])
    merged["event_flag"] = merged["event_origin_category"].ne("external_only").astype(int)
    merged["relabel_flag"] = merged["event_flag"]
    merged = annotate_event_broad_bins(merged)
    for column in VERIFIED_EVENT_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    merged = merged.loc[:, VERIFIED_EVENT_COLUMNS].copy()
    merged = merged.sort_values(
        ["degree_type", "event_origin_category", "relabel_year", "unitid", "event_source_cip6", "target_cip6"],
        na_position="last",
    ).reset_index(drop=True)
    return merged


def build_verified_event_panel(
    con: ddb.DuckDBPyConnection,
    events: pd.DataFrame,
    *,
    ipeds_path: str | Path = base.IPEDS_PATH,
    relabel_year_mode: str = DEFAULT_RELABEL_YEAR_MODE,
) -> pd.DataFrame:
    broad_events = build_broad_treated_events(events, relabel_year_mode=relabel_year_mode)
    if broad_events.empty:
        _progress("No broad-bin eligible treated events available for panel construction")
        return pd.DataFrame(columns=VERIFIED_EVENT_COLUMNS)

    _progress(f"Building verified event panel for {len(broad_events):,} broad-bin event(s)")
    _ensure_ipeds_view(con, ipeds_path)
    cip_map = _load_ipeds_cip_map(ipeds_path)
    broad_membership = build_broad_bin_membership(cip_map.keys())
    relevant_cips = sorted(
        {
            cip6
            for row in broad_events.itertuples(index=False)
            for cip6 in broad_membership[str(row.broad_pair_bin)]["all_cips"]
        }
    )
    min_year, max_year = _ipeds_year_bounds(con)
    panel_max_year = min(max_year, int(ANALYSIS_IPEDS_YEAR_MAX))
    if min_year > panel_max_year:
        return pd.DataFrame(columns=VERIFIED_EVENT_COLUMNS)
    years = pd.Index(range(int(min_year), panel_max_year + 1), name="year")
    if not relevant_cips:
        return pd.DataFrame(columns=VERIFIED_EVENT_COLUMNS)

    event_subset = broad_events[["unitid", "awlevel"]].drop_duplicates()
    con.register("verified_events_panel_keys_py", event_subset)
    cip_vals = ", ".join(f"'{_sql_literal(cip6)}'" for cip6 in relevant_cips)
    ipeds_subset = con.sql(
        f"""
        SELECT
            CAST(i.unitid AS BIGINT) AS unitid,
            CAST(i.year AS INTEGER) AS year,
            CAST(i.awlevel AS INTEGER) AS awlevel,
            LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') AS cip6,
            CAST(i.ctotalt AS DOUBLE) AS ctotalt,
            CAST(i.cnralt AS DOUBLE) AS cnralt
        FROM ipeds_raw i
        JOIN verified_events_panel_keys_py e
          ON CAST(i.unitid AS BIGINT) = e.unitid
         AND CAST(i.awlevel AS INTEGER) = e.awlevel
        WHERE LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') IN ({cip_vals})
          AND CAST(i.year AS INTEGER) BETWEEN {int(min_year)} AND {int(panel_max_year)}
        """
    ).df()

    panel_rows: list[pd.DataFrame] = []
    for event in broad_events.itertuples(index=False):
        membership = broad_membership[str(event.broad_pair_bin)]
        source_cips = set(membership["source_cips"])
        target_cips = set(membership["target_cips"])
        ev = ipeds_subset[
            (ipeds_subset["unitid"] == int(event.unitid))
            & (ipeds_subset["awlevel"] == int(event.awlevel))
            & (ipeds_subset["cip6"].isin(membership["all_cips"]))
        ].copy()

        if ev.empty:
            source_year = pd.Series(0.0, index=years)
            target_year = pd.Series(0.0, index=years)
            cnr_source = pd.Series(0.0, index=years)
            cnr_target = pd.Series(0.0, index=years)
        else:
            grouped = ev.groupby(["year", "cip6"], as_index=False).agg({"ctotalt": "sum", "cnralt": "sum"})
            source_grouped = grouped[grouped["cip6"].isin(source_cips)].groupby("year", as_index=True)[["ctotalt", "cnralt"]].sum()
            target_grouped = grouped[grouped["cip6"].isin(target_cips)].groupby("year", as_index=True)[["ctotalt", "cnralt"]].sum()
            source_year = source_grouped.get("ctotalt", pd.Series(dtype=float)).reindex(years, fill_value=0.0)
            target_year = target_grouped.get("ctotalt", pd.Series(dtype=float)).reindex(years, fill_value=0.0)
            cnr_source = source_grouped.get("cnralt", pd.Series(dtype=float)).reindex(years, fill_value=0.0)
            cnr_target = target_grouped.get("cnralt", pd.Series(dtype=float)).reindex(years, fill_value=0.0)

        panel = pd.DataFrame({"year": years})
        panel["unitid"] = int(event.unitid)
        panel["awlevel"] = int(event.awlevel)
        panel["degree_type"] = event.degree_type
        panel["relabel_year"] = int(event.relabel_year)
        panel["relabel_type"] = event.broad_pair_bin
        panel["broad_pair_bin"] = event.broad_pair_bin
        panel["broad_bin_eligible"] = 1
        panel["event_source_cip6"] = event.event_source_cip6
        panel["target_cip6"] = event.target_cip6
        panel["source_total"] = source_year.values
        panel["source_total_prev"] = source_year.shift(1, fill_value=0.0).values
        panel["source_total_intl"] = source_year.values
        panel["source_total_intl_prev"] = source_year.shift(1, fill_value=0.0).values
        panel["target_total"] = target_year.values
        panel["target_total_prev"] = target_year.shift(1, fill_value=0.0).values
        panel["target_total_intl"] = target_year.values
        panel["target_total_intl_prev"] = target_year.shift(1, fill_value=0.0).values
        panel["ctotalt"] = panel["source_total"] + panel["target_total"]
        panel["cnralt"] = cnr_source.values + cnr_target.values
        panel["event_flag"] = (panel["year"] == int(event.relabel_year)).astype(int)
        panel["relabel_flag"] = panel["event_flag"]
        metric_columns = [
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
            "found_in_ipeds_scan",
            "found_in_external_candidates",
            "external_verified",
            "event_origin_category",
            "source_cip_label",
            "target_cip_label",
            "source_major",
            "target_major",
            "candidate_id",
            "candidate_school_name",
            "candidate_approx_year",
            "candidate_program_desc",
            "candidate_degree_label",
            "candidate_notes",
            "candidate_major",
            "candidate_source_cip_bin",
            "candidate_target_cip_bin",
            "candidate_pair_bin",
            "candidate_cip_parse_notes",
            "n_linked_candidates",
            "school_match_method",
            "school_match_score",
            "school_match_name",
            "verification_notes",
            "best_candidate_rank_score",
            "best_year_distance",
            "best_text_similarity",
            "best_nearby_year",
            "best_nearby_source_cip6",
            "best_nearby_target_cip6",
            "best_nearby_relabel_score",
            "diagnostic_best_year",
            "diagnostic_best_source_cip6",
            "diagnostic_best_target_cip6",
            "diagnostic_best_score",
        ]
        for metric in metric_columns:
            panel[metric] = getattr(event, metric, pd.NA)
        for column in VERIFIED_EVENT_COLUMNS:
            if column not in panel.columns and hasattr(event, column):
                panel[column] = getattr(event, column)
            elif column not in panel.columns:
                panel[column] = pd.NA
        panel_rows.append(panel.loc[:, VERIFIED_EVENT_COLUMNS])

    out = pd.concat(panel_rows, ignore_index=True)
    out = out.sort_values(["degree_type", "broad_pair_bin", "unitid", "relabel_year", "year"]).reset_index(drop=True)
    _progress(f"Built verified event panel with {len(out):,} row(s)")
    return out


def _foia_degree_case(column_name: str = "degree_type") -> str:
    parts = []
    for degree_type, foia_label in DEGREE_TYPE_TO_FOIA_LABEL.items():
        parts.append(f"WHEN {column_name} = '{_sql_literal(degree_type)}' THEN '{_sql_literal(foia_label)}'")
    return "CASE " + " ".join(parts) + " ELSE NULL END"


def _duckdb_relation_exists(con: ddb.DuckDBPyConnection, relation_name: str) -> bool:
    try:
        con.sql(f"SELECT 1 FROM {relation_name} LIMIT 0")
        return True
    except Exception:
        return False


def _load_foia_base(
    con: ddb.DuckDBPyConnection,
    *,
    foia_path: str | Path = base.FOIA_PATH,
    inst_cw_path: str | Path = base.F1_INST_CW_PATH,
    foia_person_panel_path: str | Path = DEFAULT_FOIA_PERSON_PANEL_PATH,
    employer_match_dir: str | Path = DEFAULT_EMPLOYER_MATCH_DIR,
) -> dict[str, str | None]:
    foia_path = Path(foia_path)
    inst_cw_path = Path(inst_cw_path)
    foia_person_panel_path = Path(foia_person_panel_path)
    employer_match_dir = Path(employer_match_dir)
    if not foia_path.exists():
        raise FileNotFoundError(f"Missing FOIA parquet: {foia_path}")
    if not inst_cw_path.exists():
        raise FileNotFoundError(f"Missing F-1 institution crosswalk parquet: {inst_cw_path}")
    if not foia_person_panel_path.exists():
        raise FileNotFoundError(f"Missing FOIA person panel parquet: {foia_person_panel_path}")
    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{_sql_literal(str(foia_path))}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_cw AS SELECT * FROM read_parquet('{_sql_literal(str(inst_cw_path))}')")
    schema = v2._resolve_foia_schema(con)
    foia_cols = [row[0] for row in con.sql("DESCRIBE foia_raw").fetchall()]
    foia_col_map = {str(col).lower(): str(col) for col in foia_cols}
    schema["students_personal_funds_col"] = foia_col_map.get("students_personal_funds")
    schema["funds_from_this_school_col"] = foia_col_map.get("funds_from_this_school")
    schema["funds_from_other_sources_col"] = foia_col_map.get("funds_from_other_sources")
    _stage_foia_employer_history_views(
        con,
        schema=schema,
        foia_path=foia_path,
        foia_person_panel_path=foia_person_panel_path,
        employer_match_dir=employer_match_dir,
    )
    return schema


def _stage_foia_analysis_base(
    con: ddb.DuckDBPyConnection,
    *,
    schema: dict[str, str | None],
) -> None:
    if _duckdb_relation_exists(con, "foia_analysis_base"):
        return
    _ensure_stem_opt_cip_view(con)
    foia_inst_col = str(schema["foia_inst_col"])
    foia_cip_col = str(schema["foia_cip_col"])
    foia_end_col = str(schema["foia_end_col"])
    foia_student_col = str(schema["foia_student_col"])
    foia_tuition_col = str(schema["foia_tuition_col"])
    foia_edu_col = str(schema["foia_edu_col"])
    status_col = str(schema["status_col"])
    cw_inst_col = str(schema["cw_inst_col"])
    cw_unitid_col = str(schema["cw_unitid_col"])
    foia_year_col = schema["foia_year_col"]
    opt_end_col = str(schema["opt_end_col"])
    norm_cip_expr = base.normalize_cip_sql(foia_cip_col)
    personal_funds_expr = _money_sql_expr(f"fr.{schema['students_personal_funds_col']}") if schema.get("students_personal_funds_col") else "NULL"
    school_funds_expr = _money_sql_expr(f"fr.{schema['funds_from_this_school_col']}") if schema.get("funds_from_this_school_col") else "NULL"
    other_funds_expr = _money_sql_expr(f"fr.{schema['funds_from_other_sources_col']}") if schema.get("funds_from_other_sources_col") else "NULL"
    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )
    grad_year_max_clause = f"AND CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) <= {int(ANALYSIS_ORIGINAL_YEAR_MAX)}"
    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE foia_analysis_base AS
        SELECT *
        FROM (
            SELECT
                fr.original_row_num,
                cw.{cw_unitid_col} AS unitid,
                LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0') AS cip6,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                CASE
                    WHEN stem.first_stem_year IS NOT NULL
                     AND stem.first_stem_year <= CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)
                    THEN 1 ELSE 0
                END AS stem_cip_eligible_ind,
                employer_name,
                employment_opt_type,
                {opt_end_col} AS opt_end_date,
                {foia_tuition_col} AS tuition,
                {personal_funds_expr} AS students_personal_funds,
                {school_funds_expr} AS funds_from_this_school,
                {other_funds_expr} AS funds_from_other_sources,
                {foia_end_col} AS program_end_date,
                {status_col} AS requested_status,
                {foia_edu_col} AS foia_degree_label
            FROM foia_raw_with_rownum fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            LEFT JOIN stem_opt_cip_first_year stem
              ON stem.cip6 = LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0')
            WHERE {foia_end_col} IS NOT NULL
              {year_match_clause}
              {grad_year_max_clause}
        )
        WHERE unitid IS NOT NULL
          AND cip6 IS NOT NULL
          AND grad_year IS NOT NULL
        """
    )


def _find_artifact(base_dir: str | Path, stem: str) -> Path:
    root = Path(base_dir)
    for ext in (".parquet", ".csv"):
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing artifact '{stem}' under {root}")


def _duckdb_reader_sql(path: str | Path) -> str:
    artifact_path = Path(path)
    literal = _sql_literal(str(artifact_path))
    suffix = artifact_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return f"read_parquet('{literal}')"
    if suffix == ".csv":
        return f"read_csv_auto('{literal}', header=TRUE)"
    raise ValueError(f"Unsupported artifact format for {artifact_path}")


def _fallback_spell_key_sql(alias: str) -> str:
    spell_cols = (
        "employment_opt_type",
        "authorization_start_date",
        "authorization_end_date",
        "opt_authorization_start_date",
        "opt_authorization_end_date",
        "opt_employer_start_date",
        "opt_employer_end_date",
    )
    parts = [f"COALESCE(CAST({alias}.{col} AS VARCHAR), '')" for col in spell_cols]
    return "'fallback:' || " + " || '||' || ".join(parts)


def _student_employer_outcome_ctes(
    *,
    source_table: str,
    group_cols: list[str],
) -> str:
    group_select = ",\n                ".join(f"f.{col}" for col in group_cols)
    group_by = ", ".join(group_cols)
    join_conditions = "\n             AND ".join(f"b.{col} = c.{col}" for col in group_cols)
    tenure_join_conditions = "\n             AND ".join(f"b.{col} = t.{col}" for col in group_cols)
    pctile_join_conditions = "\n             AND ".join(f"b.{col} = p.{col}" for col in group_cols)
    return f"""
        student_employer_rows AS (
            SELECT DISTINCT
                {group_select},
                eh.foia_firm_uid,
                eh.employer_city_clean,
                eh.spell_key,
                eh.spell_start_date,
                eh.spell_end_date,
                f.program_end_date
            FROM {source_table} f
            LEFT JOIN foia_person_employer_history eh
              ON f.original_row_num = eh.original_row_num
        ),
        student_employer_counts AS (
            SELECT
                {group_by},
                COUNT(DISTINCT foia_firm_uid) AS unique_employers,
                COUNT(DISTINCT employer_city_clean) AS unique_opt_cities
            FROM student_employer_rows
            GROUP BY {group_by}
        ),
        student_spell_tenure AS (
            SELECT
                {group_by},
                AVG(spell_duration_years) AS auth_employment_tenure_years
            FROM (
                SELECT DISTINCT
                    {group_by},
                    spell_key,
                    DATE_DIFF('day', spell_start_date, spell_end_date) / 365.25 AS spell_duration_years
                FROM student_employer_rows
                WHERE spell_key IS NOT NULL
                  AND spell_start_date IS NOT NULL
                  AND spell_end_date IS NOT NULL
            )
            GROUP BY {group_by}
        ),
        student_opt_authorization_spells AS (
            SELECT
                {group_by},
                SUM(spell_duration_years) AS opt_duration_years
            FROM (
                SELECT DISTINCT
                    {group_by},
                    spell_key,
                    GREATEST(
                        DATE_DIFF(
                            'day',
                            TRY_CAST(spell_start_date AS DATE),
                            TRY_CAST(spell_end_date AS DATE)
                        ) / 365.25,
                        0.0
                    ) AS spell_duration_years
                FROM student_employer_rows
                WHERE spell_key IS NOT NULL
                  AND TRY_CAST(spell_start_date AS DATE) IS NOT NULL
                  AND TRY_CAST(spell_end_date AS DATE) IS NOT NULL
                  AND TRY_CAST(spell_end_date AS DATE) >= TRY_CAST(spell_start_date AS DATE)
            )
            GROUP BY {group_by}
        ),
        student_internship_spells AS (
            SELECT
                {group_by},
                COUNT(DISTINCT spell_key) AS internship_count,
                SUM(spell_duration_years) AS internship_opt_years
            FROM (
                SELECT DISTINCT
                    {group_by},
                    spell_key,
                    DATE_DIFF(
                        'day',
                        TRY_CAST(spell_start_date AS DATE),
                        TRY_CAST(spell_end_date AS DATE)
                    ) / 365.25 AS spell_duration_years
                FROM student_employer_rows
                WHERE spell_key IS NOT NULL
                  AND TRY_CAST(spell_start_date AS DATE) IS NOT NULL
                  AND TRY_CAST(program_end_date AS DATE) IS NOT NULL
                  AND TRY_CAST(spell_start_date AS DATE) < DATE_TRUNC('year', TRY_CAST(program_end_date AS DATE))
                  AND (
                      TRY_CAST(spell_end_date AS DATE) IS NULL
                      OR TRY_CAST(spell_end_date AS DATE) >= TRY_CAST(spell_start_date AS DATE)
                  )
            )
            GROUP BY {group_by}
        ),
        student_employer_pctile AS (
            SELECT
                {group_by},
                AVG(employer_opt_intensity_pctile) AS employer_opt_intensity_pctile
            FROM (
                SELECT DISTINCT
                    ser.{", ser.".join(group_cols)},
                    ser.foia_firm_uid,
                    pct.employer_opt_intensity_pctile
                FROM student_employer_rows ser
                LEFT JOIN foia_employer_intensity_pctiles pct
                  ON ser.foia_firm_uid = pct.foia_firm_uid
                WHERE ser.foia_firm_uid IS NOT NULL
            )
            GROUP BY {group_by}
        ),
        student_level AS (
            SELECT
                b.*,
                COALESCE(c.unique_employers, 0) AS unique_employers,
                COALESCE(c.unique_opt_cities, 0) AS unique_opt_cities,
                COALESCE(t.auth_employment_tenure_years, 0.0) AS auth_employment_tenure_years,
                COALESCE(od.opt_duration_years, 0.0) AS opt_duration_years,
                COALESCE(p.employer_opt_intensity_pctile, 0.0) AS employer_opt_intensity_pctile,
                COALESCE(i.internship_count, 0) AS internship_count,
                COALESCE(i.internship_opt_years, 0.0) AS internship_opt_years
            FROM student_level_base b
            LEFT JOIN student_employer_counts c
              ON {join_conditions}
            LEFT JOIN student_spell_tenure t
              ON {tenure_join_conditions}
            LEFT JOIN student_opt_authorization_spells od
              ON {join_conditions.replace("c.", "od.")}
            LEFT JOIN student_employer_pctile p
              ON {pctile_join_conditions}
            LEFT JOIN student_internship_spells i
              ON {join_conditions.replace("c.", "i.")}
        ),
    """


def _stage_foia_employer_history_views(
    con: ddb.DuckDBPyConnection,
    *,
    schema: dict[str, str | None],
    foia_path: str | Path,
    foia_person_panel_path: str | Path,
    employer_match_dir: str | Path,
) -> None:
    if (
        _duckdb_relation_exists(con, "foia_raw_with_rownum")
        and _duckdb_relation_exists(con, "foia_person_employer_history")
        and _duckdb_relation_exists(con, "foia_employer_intensity_pctiles")
    ):
        return
    row_entities_path = _find_artifact(employer_match_dir, "foia_row_entities")
    row_to_firm_path: Path
    try:
        row_to_firm_path = _find_artifact(employer_match_dir, "foia_row_to_firm")
    except FileNotFoundError:
        row_to_firm_path = _find_artifact(employer_match_dir, "foia_entity_to_firm")

    foia_student_col = str(schema["foia_student_col"])
    foia_person_panel_path = Path(foia_person_panel_path)
    foia_path = Path(foia_path)
    raw_with_rownum_sql = _duckdb_reader_sql(foia_path)
    person_panel_sql = _duckdb_reader_sql(foia_person_panel_path)
    row_entities_sql = _duckdb_reader_sql(row_entities_path)
    row_to_firm_sql = _duckdb_reader_sql(row_to_firm_path)
    employer_name_clean_expr = sql_clean_company_name_expr("pp.employer_name")
    employer_city_clean_expr = sql_normalize_expr("pp.employer_city")
    employer_state_clean_expr = sql_state_name_to_abbr_expr("pp.employer_state")
    employer_zip_clean_expr = sql_clean_zip_expr("pp.employer_zip_code")
    fallback_spell_key_expr = _fallback_spell_key_sql("pp")

    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE foia_raw_with_rownum AS
        SELECT
            CAST(ROW_NUMBER() OVER () - 1 AS BIGINT) AS original_row_num,
            *
        FROM {raw_with_rownum_sql}
        """
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE foia_person_panel_raw AS
        SELECT *
        FROM {person_panel_sql}
        """
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE foia_row_entities_raw AS
        SELECT *
        FROM {row_entities_sql}
        """
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE foia_row_to_firm_raw AS
        SELECT *
        FROM {row_to_firm_sql}
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP TABLE foia_row_firm_lookup AS
        WITH row_base AS (
            SELECT
                NULLIF(TRIM(CAST(re.row_name_clean AS VARCHAR)), '') AS row_name_clean,
                NULLIF(TRIM(CAST(re.row_city_clean AS VARCHAR)), '') AS row_city_clean,
                NULLIF(UPPER(TRIM(CAST(re.row_state_clean AS VARCHAR))), '') AS row_state_clean,
                NULLIF(TRIM(CAST(re.row_zip_clean AS VARCHAR)), '') AS row_zip_clean,
                NULLIF(TRIM(CAST(rtf.foia_firm_uid AS VARCHAR)), '') AS foia_firm_uid
            FROM foia_row_entities_raw re
            LEFT JOIN foia_row_to_firm_raw rtf
              ON CAST(re.foia_row_uid AS VARCHAR) = CAST(rtf.foia_row_uid AS VARCHAR)
        ),
        grouped AS (
            SELECT
                row_name_clean,
                row_city_clean,
                row_state_clean,
                row_zip_clean,
                COUNT(DISTINCT foia_firm_uid) FILTER (
                    WHERE foia_firm_uid IS NOT NULL AND foia_firm_uid <> ''
                ) AS n_firms,
                MIN(foia_firm_uid) FILTER (
                    WHERE foia_firm_uid IS NOT NULL AND foia_firm_uid <> ''
                ) AS only_foia_firm_uid
            FROM row_base
            GROUP BY 1, 2, 3, 4
        )
        SELECT
            row_name_clean,
            row_city_clean,
            row_state_clean,
            row_zip_clean,
            CASE WHEN n_firms = 1 THEN only_foia_firm_uid ELSE NULL END AS foia_firm_uid
        FROM grouped
        """
    )
    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE foia_person_employer_history AS
        WITH person_rows AS (
            SELECT
                fr.original_row_num,
                CAST(fr.{foia_student_col} AS VARCHAR) AS student_id,
                CAST(pp.person_id AS VARCHAR) AS person_id,
                COALESCE(
                    CAST(pp.person_id AS VARCHAR),
                    'student:' || CAST(fr.{foia_student_col} AS VARCHAR)
                ) AS analysis_person_id,
                COALESCE(CAST(pp.spell_key AS VARCHAR), {fallback_spell_key_expr}) AS spell_key,
                NULLIF(TRIM(CAST(pp.employer_name AS VARCHAR)), '') AS employer_name,
                NULLIF(TRIM(CAST(pp.employer_city AS VARCHAR)), '') AS employer_city,
                NULLIF(TRIM(CAST({employer_name_clean_expr} AS VARCHAR)), '') AS employer_name_clean,
                NULLIF(TRIM(CAST({employer_city_clean_expr} AS VARCHAR)), '') AS employer_city_clean,
                NULLIF(TRIM(CAST({employer_state_clean_expr} AS VARCHAR)), '') AS employer_state_clean,
                NULLIF(TRIM(CAST({employer_zip_clean_expr} AS VARCHAR)), '') AS employer_zip_clean,
                COALESCE(
                    pp.opt_employer_start_date,
                    pp.opt_authorization_start_date,
                    pp.authorization_start_date
                ) AS spell_start_date,
                COALESCE(
                    pp.opt_employer_end_date,
                    pp.opt_authorization_end_date,
                    pp.authorization_end_date
                ) AS spell_end_date
            FROM foia_raw_with_rownum fr
            LEFT JOIN foia_person_panel_raw pp
              ON fr.original_row_num = pp.original_row_num
        )
        SELECT
            pr.*,
            row_map.foia_firm_uid
        FROM person_rows pr
        LEFT JOIN foia_row_firm_lookup row_map
          ON pr.employer_name_clean IS NOT DISTINCT FROM row_map.row_name_clean
         AND pr.employer_city_clean IS NOT DISTINCT FROM row_map.row_city_clean
         AND pr.employer_state_clean IS NOT DISTINCT FROM row_map.row_state_clean
         AND pr.employer_zip_clean IS NOT DISTINCT FROM row_map.row_zip_clean
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP TABLE foia_employer_intensity_pctiles AS
        WITH employer_persons AS (
            SELECT DISTINCT
                foia_firm_uid,
                analysis_person_id
            FROM foia_person_employer_history
            WHERE foia_firm_uid IS NOT NULL
              AND analysis_person_id IS NOT NULL
        ),
        employer_counts AS (
            SELECT
                foia_firm_uid,
                COUNT(*) AS employer_opt_person_count
            FROM employer_persons
            GROUP BY 1
        )
        SELECT
            foia_firm_uid,
            employer_opt_person_count,
            100.0 * PERCENT_RANK() OVER (ORDER BY employer_opt_person_count) AS employer_opt_intensity_pctile
        FROM employer_counts
        """
    )


def compute_opt_usage_generalized(
    con: ddb.DuckDBPyConnection,
    relabel_panel: pd.DataFrame,
    *,
    degree_type: str,
    foia_path: str | Path = base.FOIA_PATH,
    inst_cw_path: str | Path = base.F1_INST_CW_PATH,
    foia_person_panel_path: str | Path = DEFAULT_FOIA_PERSON_PANEL_PATH,
    employer_match_dir: str | Path = DEFAULT_EMPLOYER_MATCH_DIR,
) -> pd.DataFrame:
    panel = relabel_panel[
        (relabel_panel["degree_type"] == degree_type)
        & relabel_panel["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"])
        & relabel_panel["broad_bin_eligible"].eq(1)
    ].copy()
    panel = _filter_relabel_analysis_window(panel)
    if panel.empty:
        return pd.DataFrame()
    schema = _load_foia_base(
        con,
        foia_path=foia_path,
        inst_cw_path=inst_cw_path,
        foia_person_panel_path=foia_person_panel_path,
        employer_match_dir=employer_match_dir,
    )
    _ensure_stem_opt_cip_view(con)
    _stage_foia_analysis_base(con, schema=schema)
    cip_map = _load_ipeds_cip_map(base.IPEDS_PATH)
    broad_membership = build_broad_bin_membership(cip_map.keys())
    broad_any_cips = _broad_membership_rows(broad_membership, side="all")

    relabel_events = panel[
        [
            "unitid",
            "awlevel",
            "year",
            "relabel_year",
            "relabel_type",
            "degree_type",
            "broad_pair_bin",
            "ctotalt",
            "cnralt",
        ]
    ].drop_duplicates()
    con.register("generalized_relabel_events_py", relabel_events)
    con.register("generalized_broad_any_cips_py", broad_any_cips)

    foia_inst_col = str(schema["foia_inst_col"])
    foia_cip_col = str(schema["foia_cip_col"])
    foia_end_col = str(schema["foia_end_col"])
    foia_student_col = str(schema["foia_student_col"])
    foia_tuition_col = str(schema["foia_tuition_col"])
    foia_edu_col = str(schema["foia_edu_col"])
    status_col = str(schema["status_col"])
    cw_inst_col = str(schema["cw_inst_col"])
    cw_unitid_col = str(schema["cw_unitid_col"])
    foia_year_col = schema["foia_year_col"]
    opt_end_col = str(schema["opt_end_col"])
    ipeds_tuition_col = _ensure_ipeds_cost_view(
        con,
        tuition_col=DEFAULT_IPEDS_TUITION_COL,
    )
    ipeds_tuition_cols = _resolve_ipeds_cost_columns_by_degree(
        con,
        DEFAULT_IPEDS_TUITION_COL_BY_DEGREE,
        fallback_prefix="tuition",
    )
    ipeds_fee_cols = _resolve_ipeds_cost_columns_by_degree(
        con,
        DEFAULT_IPEDS_FEE_COL_BY_DEGREE,
        fallback_prefix="fee",
    )
    ipeds_tuition_expr = _ipeds_degree_cost_expr("u.degree_type", ipeds_tuition_cols)
    ipeds_fee_expr = _ipeds_degree_cost_expr("u.degree_type", ipeds_fee_cols)
    personal_funds_expr = _money_sql_expr(f"fr.{schema['students_personal_funds_col']}") if schema.get("students_personal_funds_col") else "NULL"
    school_funds_expr = _money_sql_expr(f"fr.{schema['funds_from_this_school_col']}") if schema.get("funds_from_this_school_col") else "NULL"
    other_funds_expr = _money_sql_expr(f"fr.{schema['funds_from_other_sources_col']}") if schema.get("funds_from_other_sources_col") else "NULL"
    norm_cip_expr = base.normalize_cip_sql(foia_cip_col)
    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )
    grad_year_max_clause = f"AND CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) <= {int(ANALYSIS_ORIGINAL_YEAR_MAX)}"
    degree_case = _foia_degree_case("r.degree_type")
    student_outcome_ctes = _student_employer_outcome_ctes(
        source_table="flagged",
        group_cols=["unitid", "grad_year", "relabel_year", "relabel_type", "degree_type", "student_id"],
    )

    opt_usage = con.sql(
        f"""
        WITH relevant_foia AS (
            SELECT *
            FROM foia_analysis_base
        ),
        flagged AS (
            SELECT
                f.*,
                r.ctotalt,
                r.cnralt,
                r.relabel_year,
                r.relabel_type,
                r.degree_type
            FROM relevant_foia f
            JOIN generalized_relabel_events_py r
              ON f.unitid = r.unitid
             AND f.grad_year = r.year
             AND f.foia_degree_label = {degree_case}
            JOIN generalized_broad_any_cips_py m
              ON m.broad_pair_bin = r.broad_pair_bin
             AND f.cip6 = m.cip6
        ),
        student_level_base AS (
            SELECT
                unitid,
                grad_year,
                relabel_year,
                relabel_type,
                degree_type,
                ctotalt,
                cnralt,
                MAX(stem_cip_eligible_ind) AS stem_cip_eligible_ind,
                MAX(CASE WHEN employer_name IS NOT NULL THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL AND MAX(program_end_date) IS NOT NULL
                    THEN GREATEST(
                        DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25,
                        0.0
                    )
                    ELSE 0
                END AS post_grad_authorization_years,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                AVG(students_personal_funds) AS avg_students_personal_funds,
                AVG(
                    COALESCE(students_personal_funds, 0.0)
                    + COALESCE(funds_from_this_school, 0.0)
                    + COALESCE(funds_from_other_sources, 0.0)
                ) AS avg_total_funds,
                student_id
            FROM flagged
            GROUP BY unitid, grad_year, relabel_year, relabel_type, degree_type, ctotalt, cnralt, student_id
        ),
        {student_outcome_ctes}
        unit_year_counts AS (
            SELECT
                unitid,
                grad_year AS calendar_year,
                relabel_year,
                relabel_type,
                degree_type,
                COUNT(DISTINCT student_id) AS total_grads_unit
            FROM student_level
            GROUP BY unitid, grad_year, relabel_year, relabel_type, degree_type
        ),
        ipeds_tuition AS (
            SELECT
                u.unitid,
                u.calendar_year,
                u.relabel_year,
                u.relabel_type,
                u.degree_type,
                SUM(u.total_grads_unit * {ipeds_tuition_expr}) AS tuition_ipeds_total,
                SUM(u.total_grads_unit * {ipeds_fee_expr}) AS fees_ipeds_total
            FROM unit_year_counts u
            LEFT JOIN ipeds_cost_raw ic
              ON CAST(ic.unitid AS BIGINT) = CAST(u.unitid AS BIGINT)
             AND CAST(ic.year AS INTEGER) = CAST(u.calendar_year AS INTEGER)
            GROUP BY u.unitid, u.calendar_year, u.relabel_year, u.relabel_type, u.degree_type
        )
        SELECT
            student_level.grad_year AS calendar_year,
            student_level.relabel_year,
            student_level.relabel_type,
            degree_type,
            AVG(student_level.avg_tuition) AS avg_tuition,
            COUNT(DISTINCT student_id) AS total_grads,
            COUNT(DISTINCT CASE WHEN stem_cip_eligible_ind = 1 THEN student_id END) AS stem_cip_eligible_users,
            COUNT(DISTINCT CASE WHEN opt_ind = 1 THEN student_id END) AS opt_users,
            COUNT(DISTINCT CASE WHEN opt_stem_ind = 1 THEN student_id END) AS opt_stem_users,
            COUNT(DISTINCT CASE WHEN status_change_ind = 1 THEN student_id END) AS status_change_users,
            SUM(post_grad_authorization_years) AS total_post_grad_authorization_years,
            SUM(opt_duration_years) AS total_opt_duration_years,
            AVG(student_level.unique_employers) AS unique_employers,
            AVG(student_level.unique_opt_cities) AS unique_opt_cities,
            AVG(student_level.auth_employment_tenure_years) AS auth_employment_tenure_years,
            AVG(student_level.employer_opt_intensity_pctile) AS employer_opt_intensity_pctile,
            AVG(student_level.avg_students_personal_funds) AS avg_students_personal_funds,
            AVG(student_level.avg_total_funds) AS avg_total_funds,
            SUM(student_level.internship_count) AS total_internships,
            SUM(student_level.internship_opt_years) AS total_internship_opt_years,
            MAX(student_level.ctotalt) AS ctotalt,
            MAX(student_level.cnralt) AS cnralt,
            COALESCE(SUM(t.tuition_ipeds_total), 0.0) AS tuition_ipeds_total,
            COALESCE(SUM(t.fees_ipeds_total), 0.0) AS fees_ipeds_total
        FROM student_level
        LEFT JOIN ipeds_tuition t
          ON student_level.unitid = t.unitid
         AND student_level.grad_year = t.calendar_year
         AND student_level.relabel_year = t.relabel_year
         AND student_level.relabel_type = t.relabel_type
         AND student_level.degree_type = t.degree_type
        GROUP BY
            student_level.grad_year,
            student_level.relabel_year,
            student_level.relabel_type,
            student_level.degree_type
        ORDER BY
            student_level.grad_year,
            student_level.relabel_year,
            student_level.relabel_type
        """
    ).df()

    if opt_usage.empty:
        return opt_usage
    opt_usage["stem_cip_eligible_share"] = _safe_share(opt_usage["stem_cip_eligible_users"], opt_usage["total_grads"])
    opt_usage["opt_share"] = _safe_share(opt_usage["opt_users"], opt_usage["total_grads"])
    opt_usage["opt_stem_share"] = _safe_share(opt_usage["opt_stem_users"], opt_usage["total_grads"])
    opt_usage["status_change_share"] = _safe_share(opt_usage["status_change_users"], opt_usage["total_grads"])
    opt_usage["post_grad_authorization_years_avg"] = _safe_share(
        opt_usage["total_post_grad_authorization_years"],
        opt_usage["total_grads"],
    )
    opt_usage["opt_duration_years_avg"] = _safe_share(opt_usage["total_opt_duration_years"], opt_usage["total_grads"])
    opt_usage["opt_years_avg"] = opt_usage["post_grad_authorization_years_avg"]
    opt_usage["internship_count"] = _safe_share(opt_usage["total_internships"], opt_usage["total_grads"])
    opt_usage["internship_opt_years"] = _safe_share(
        opt_usage["total_internship_opt_years"],
        opt_usage["total_grads"],
    )
    opt_usage["f1_share_of_ctotalt"] = _safe_share(opt_usage["total_grads"], opt_usage["ctotalt"])
    opt_usage["f1_share_of_cnralt"] = _safe_share(opt_usage["total_grads"], opt_usage["cnralt"])
    opt_usage["tuition_total"] = opt_usage["avg_tuition"] * opt_usage["total_grads"]
    opt_usage["avg_tuition"] = _safe_share(opt_usage["tuition_total"], opt_usage["total_grads"])
    opt_usage["tuition_ipeds_total"] = pd.to_numeric(opt_usage["tuition_ipeds_total"], errors="coerce")
    opt_usage["avg_tuition_ipeds"] = _safe_share(opt_usage["tuition_ipeds_total"], opt_usage["total_grads"])
    opt_usage["fees_ipeds_total"] = pd.to_numeric(opt_usage["fees_ipeds_total"], errors="coerce")
    opt_usage["avg_fees_ipeds"] = _safe_share(opt_usage["fees_ipeds_total"], opt_usage["total_grads"])
    opt_usage["students_personal_funds_total"] = (
        pd.to_numeric(opt_usage["avg_students_personal_funds"], errors="coerce")
        * pd.to_numeric(opt_usage["total_grads"], errors="coerce")
    )
    opt_usage["total_funds"] = (
        pd.to_numeric(opt_usage["avg_total_funds"], errors="coerce")
        * pd.to_numeric(opt_usage["total_grads"], errors="coerce")
    )
    return opt_usage


def compute_opt_usage_event_time_generalized(opt_usage: pd.DataFrame) -> pd.DataFrame:
    if opt_usage.empty:
        return pd.DataFrame()
    df = _filter_relabel_analysis_window(opt_usage, year_col="calendar_year")
    if df.empty:
        return pd.DataFrame()
    optional_sum_cols = [
        "fees_ipeds_total",
        "students_personal_funds_total",
        "total_funds",
    ]
    for col in optional_sum_cols:
        if col not in df.columns:
            df[col] = 0.0
    df["event_t"] = df["calendar_year"] - df["relabel_year"]
    df = df[df["event_t"].between(PLOT_EVENT_MIN, PLOT_EVENT_MAX)].copy()
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby(["event_t", "relabel_type", "degree_type"], as_index=False)
        .agg(
            total_grads=("total_grads", "sum"),
            opt_users=("opt_users", "sum"),
            opt_stem_users=("opt_stem_users", "sum"),
            total_post_grad_authorization_years=("total_post_grad_authorization_years", "sum"),
            total_opt_duration_years=("total_opt_duration_years", "sum"),
            total_status_change_users=("status_change_users", "sum"),
            unique_employers_total=("unique_employers", "sum"),
            unique_opt_cities_total=("unique_opt_cities", "sum"),
            auth_employment_tenure_total=("auth_employment_tenure_years", "sum"),
            employer_opt_intensity_pctile_total=("employer_opt_intensity_pctile", "sum"),
            total_internships=("total_internships", "sum"),
            total_internship_opt_years=("total_internship_opt_years", "sum"),
            tuition_total=("tuition_total", "sum"),
            tuition_ipeds_total=("tuition_ipeds_total", "sum"),
            fees_ipeds_total=("fees_ipeds_total", "sum"),
            students_personal_funds_total=("students_personal_funds_total", "sum"),
            total_funds=("total_funds", "sum"),
            ctotalt=("ctotalt", "sum"),
            cnralt=("cnralt", "sum"),
        )
    )
    grouped["opt_share"] = _safe_share(grouped["opt_users"], grouped["total_grads"])
    grouped["opt_stem_share"] = _safe_share(grouped["opt_stem_users"], grouped["total_grads"])
    grouped["post_grad_authorization_years_avg"] = _safe_share(
        grouped["total_post_grad_authorization_years"],
        grouped["total_grads"],
    )
    grouped["opt_duration_years_avg"] = _safe_share(grouped["total_opt_duration_years"], grouped["total_grads"])
    grouped["opt_years_avg"] = grouped["post_grad_authorization_years_avg"]
    grouped["internship_count"] = _safe_share(grouped["total_internships"], grouped["total_grads"])
    grouped["internship_opt_years"] = _safe_share(
        grouped["total_internship_opt_years"],
        grouped["total_grads"],
    )
    grouped["status_change_share"] = _safe_share(grouped["total_status_change_users"], grouped["total_grads"])
    grouped["f1_share_of_ctotalt"] = _safe_share(grouped["total_grads"], grouped["ctotalt"])
    grouped["f1_share_of_cnralt"] = _safe_share(grouped["total_grads"], grouped["cnralt"])
    grouped["cnralt_share_of_ctotalt"] = _safe_share(grouped["cnralt"], grouped["ctotalt"])
    grouped["unique_employers"] = _safe_share(grouped["unique_employers_total"], grouped["total_grads"])
    grouped["unique_opt_cities"] = _safe_share(grouped["unique_opt_cities_total"], grouped["total_grads"])
    grouped["auth_employment_tenure_years"] = _safe_share(
        grouped["auth_employment_tenure_total"],
        grouped["total_grads"],
    )
    grouped["employer_opt_intensity_pctile"] = _safe_share(
        grouped["employer_opt_intensity_pctile_total"],
        grouped["total_grads"],
    )
    grouped["avg_tuition"] = _safe_share(grouped["tuition_total"], grouped["total_grads"])
    grouped["avg_tuition_ipeds"] = _safe_share(grouped["tuition_ipeds_total"], grouped["total_grads"])
    grouped["avg_fees_ipeds"] = _safe_share(grouped["fees_ipeds_total"], grouped["total_grads"])
    grouped["avg_students_personal_funds"] = _safe_share(
        grouped["students_personal_funds_total"],
        grouped["total_grads"],
    )
    grouped["avg_total_funds"] = _safe_share(grouped["total_funds"], grouped["total_grads"])
    return grouped


def _mean_first_difference(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    diffs = numeric.diff().dropna()
    if diffs.empty:
        return 0.0
    return float(diffs.mean())


def _match_scale(values: pd.Series) -> float:
    scale = float(pd.to_numeric(values, errors="coerce").std(ddof=0))
    if not math.isfinite(scale) or scale <= 0:
        return 1.0
    return scale


def _passes_size_ratio_caliper(
    control_value: pd.Series,
    treated_value: float,
    *,
    min_ratio: float,
    max_ratio: float,
    min_treated_level: float,
) -> pd.Series:
    control_numeric = pd.to_numeric(control_value, errors="coerce").fillna(0.0)
    if not math.isfinite(float(treated_value)) or float(treated_value) < float(min_treated_level):
        return pd.Series(True, index=control_numeric.index)
    lower = float(min_ratio) * float(treated_value)
    upper = float(max_ratio) * float(treated_value)
    return control_numeric.between(lower, upper, inclusive="both")


def _summarize_treated_preperiod_paths(
    relabel_panel: pd.DataFrame,
    treated_events: pd.DataFrame,
    *,
    lookback_years: int,
) -> pd.DataFrame:
    if treated_events.empty:
        return treated_events.copy()

    key_cols = [
        "unitid",
        "awlevel",
        "degree_type",
        "relabel_year",
        "relabel_type",
        "broad_pair_bin",
        "event_source_cip6",
        "target_cip6",
    ]
    panel = relabel_panel[key_cols + ["year", "source_total"]].copy()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel["source_total"] = pd.to_numeric(panel["source_total"], errors="coerce").fillna(0.0)
    panel = (
        panel.groupby(key_cols + ["year"], as_index=False, dropna=False)["source_total"]
        .sum()
    )
    merged = treated_events[key_cols].merge(panel, on=key_cols, how="left")
    merged = merged[
        merged["year"].between(
            pd.to_numeric(merged["relabel_year"], errors="coerce") - lookback_years,
            pd.to_numeric(merged["relabel_year"], errors="coerce") - 1,
        )
    ].copy()

    summaries: list[dict[str, object]] = []
    for key, group in merged.groupby(key_cols, dropna=False, sort=False):
        levels = (
            group.sort_values("year")["source_total"]
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .reset_index(drop=True)
        )
        row = {column: value for column, value in zip(key_cols, key)}
        row["treated_pre_avg_level"] = float(levels.mean()) if not levels.empty else 0.0
        row["treated_pre_avg_growth"] = _mean_first_difference(levels)
        summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    out = treated_events.copy()
    out["treated_source_group_pre_size"] = pd.to_numeric(out["source_total_prev"], errors="coerce").fillna(0.0)
    if summary_df.empty:
        out["treated_pre_avg_level"] = 0.0
        out["treated_pre_avg_growth"] = 0.0
        return out
    out = out.merge(summary_df, on=key_cols, how="left")
    out["treated_pre_avg_level"] = pd.to_numeric(out["treated_pre_avg_level"], errors="coerce").fillna(0.0)
    out["treated_pre_avg_growth"] = pd.to_numeric(out["treated_pre_avg_growth"], errors="coerce").fillna(0.0)
    return out


def _late_treated_control_events(
    con: ddb.DuckDBPyConnection,
    *,
    ipeds_path: str | Path,
    allowed_pair_configs: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    events = detect_ipeds_relabels(
        con,
        ipeds_path=ipeds_path,
        thresholds=STRICT_THRESHOLDS,
        allowed_pair_configs=allowed_pair_configs,
        relabel_year_mode=DEFAULT_RELABEL_YEAR_MODE,
        year_min=LATE_TREATED_CONTROL_YEAR_MIN,
        year_max=LATE_TREATED_CONTROL_YEAR_MAX,
        clamp_to_analysis_window=False,
    )
    if events.empty:
        return events
    events = annotate_event_broad_bins(events)
    relabel_year = pd.to_numeric(events["relabel_year"], errors="coerce")
    out = events[
        relabel_year.between(LATE_TREATED_CONTROL_YEAR_MIN, LATE_TREATED_CONTROL_YEAR_MAX)
        & events["broad_bin_eligible"].eq(1)
    ].copy()
    out["control_relabel_year"] = pd.to_numeric(out["relabel_year"], errors="coerce").astype("Int64")
    return out


def _always_stem_cip_rows(
    con: ddb.DuckDBPyConnection,
    *,
    first_year_max: int = ALWAYS_STEM_FIRST_YEAR_MAX,
) -> pd.DataFrame:
    _ensure_stem_opt_cip_view(con)
    rows = con.sql(
        f"""
        SELECT DISTINCT
            cip6
        FROM stem_opt_cip_first_year
        WHERE first_stem_year IS NOT NULL
          AND CAST(first_stem_year AS INTEGER) <= {int(first_year_max)}
        """
    ).df()
    if rows.empty:
        return pd.DataFrame(columns=["control_cip6"])
    rows["control_cip6"] = rows["cip6"].astype(str).str.zfill(6)
    rows = rows[rows["control_cip6"].str.slice(0, 2).isin(ALWAYS_STEM_COMPARABLE_CIP2)].copy()
    return rows[["control_cip6"]].drop_duplicates().reset_index(drop=True)


def _pair_control_cip_rows(
    matched_pairs: pd.DataFrame,
    broad_membership: dict[str, dict[str, tuple[str, ...]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if matched_pairs.empty:
        return pd.DataFrame(columns=["pair_id", "cip6"])
    for row in matched_pairs.itertuples(index=False):
        pair_id = int(getattr(row, "pair_id"))
        control_cip6 = getattr(row, "control_cip6", pd.NA)
        if pd.notna(control_cip6) and str(control_cip6).strip():
            rows.append({"pair_id": pair_id, "cip6": str(control_cip6).zfill(6)})
            continue
        broad_pair_bin = str(getattr(row, "broad_pair_bin"))
        for cip6 in broad_membership.get(broad_pair_bin, {}).get("all_cips", ()):
            rows.append({"pair_id": pair_id, "cip6": str(cip6).zfill(6)})
    if not rows:
        return pd.DataFrame(columns=["pair_id", "cip6"])
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _unit_year_carnegie_rows(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    rows = con.sql(
        """
        SELECT
            CAST(unitid AS BIGINT) AS unitid,
            CAST(year AS INTEGER) AS year,
            CAST(MAX(c21basic) AS INTEGER) AS carnegie_basic,
            MIN(CAST(c21basic_lab AS VARCHAR)) AS carnegie_basic_label
        FROM ipeds_raw
        WHERE unitid IS NOT NULL
          AND year IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    if rows.empty:
        return pd.DataFrame(columns=["unitid", "year", "carnegie_basic", "carnegie_basic_label"])
    rows["unitid"] = pd.to_numeric(rows["unitid"], errors="coerce").astype("Int64")
    rows["year"] = pd.to_numeric(rows["year"], errors="coerce").astype("Int64")
    rows["carnegie_basic"] = pd.to_numeric(rows["carnegie_basic"], errors="coerce").astype("Int64")
    return rows.dropna(subset=["unitid", "year"]).drop_duplicates(subset=["unitid", "year"]).reset_index(drop=True)


def _attach_match_carnegie(
    df: pd.DataFrame,
    carnegie_rows: pd.DataFrame,
    *,
    unitid_col: str,
    year_col: str,
    prefix: str,
) -> pd.DataFrame:
    basic_col = f"{prefix}_carnegie_basic"
    label_col = f"{prefix}_carnegie_basic_label"
    if df.empty:
        out = df.copy()
        out[basic_col] = pd.Series(dtype="Int64")
        out[label_col] = pd.Series(dtype="object")
        return out
    out = df.copy()
    out["_carnegie_match_unitid"] = pd.to_numeric(out[unitid_col], errors="coerce").astype("Int64")
    out["_carnegie_match_year"] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
    lookup = carnegie_rows.rename(
        columns={
            "unitid": "_carnegie_match_unitid",
            "year": "_carnegie_match_year",
            "carnegie_basic": basic_col,
            "carnegie_basic_label": label_col,
        }
    )
    out = out.merge(
        lookup[["_carnegie_match_unitid", "_carnegie_match_year", basic_col, label_col]],
        on=["_carnegie_match_unitid", "_carnegie_match_year"],
        how="left",
    )
    return out.drop(columns=["_carnegie_match_unitid", "_carnegie_match_year"])


def _alternate_control_candidate_pool(
    con: ddb.DuckDBPyConnection,
    *,
    control_group: str,
    min_share_intl: float,
    min_pre_year: int,
    max_pre_year: int,
) -> pd.DataFrame:
    if control_group == CONTROL_GROUP_LATE_TREATED:
        return con.sql(
            f"""
            WITH annual_source AS (
                SELECT
                    CAST(i.unitid AS BIGINT) AS unitid,
                    CAST(i.awlevel AS INTEGER) AS awlevel,
                    g.degree_type,
                    CAST(i.year AS INTEGER) AS year,
                    s.broad_pair_bin,
                    CAST(NULL AS VARCHAR) AS control_cip6,
                    MAX(l.control_relabel_year) AS control_relabel_year,
                    SUM(CAST(i.ctotalt AS DOUBLE)) AS source_total
                FROM ipeds_raw i
                JOIN generalized_broad_source_cips_py s
                  ON LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = s.cip6
                JOIN generalized_matching_groups_py g
                  ON CAST(i.awlevel AS INTEGER) = g.awlevel
                 AND s.broad_pair_bin = g.broad_pair_bin
                JOIN generalized_late_control_events_py l
                  ON CAST(i.unitid AS BIGINT) = CAST(l.unitid AS BIGINT)
                 AND CAST(i.awlevel AS INTEGER) = CAST(l.awlevel AS INTEGER)
                 AND g.degree_type = CAST(l.degree_type AS VARCHAR)
                 AND s.broad_pair_bin = CAST(l.broad_pair_bin AS VARCHAR)
                LEFT JOIN generalized_treated_group_unitids_py tgu
                  ON CAST(i.unitid AS BIGINT) = CAST(tgu.treated_unitid AS BIGINT)
                 AND g.degree_type = tgu.degree_type
                 AND s.broad_pair_bin = tgu.broad_pair_bin
                WHERE tgu.treated_unitid IS NULL
                  AND COALESCE(CAST(i.share_intl AS DOUBLE), 0) >= {float(min_share_intl)}
                  AND CAST(i.year AS INTEGER) BETWEEN {int(min_pre_year)} AND {int(max_pre_year)}
                GROUP BY
                    CAST(i.unitid AS BIGINT),
                    CAST(i.awlevel AS INTEGER),
                    g.degree_type,
                    CAST(i.year AS INTEGER),
                    s.broad_pair_bin
            ),
            candidate_units AS (
                SELECT DISTINCT unitid, awlevel, degree_type, broad_pair_bin, control_cip6, control_relabel_year
                FROM annual_source
            ),
            grid AS (
                SELECT
                    wy.relabel_year,
                    wy.awlevel,
                    wy.degree_type,
                    wy.broad_pair_bin,
                    cu.unitid,
                    cu.control_cip6,
                    cu.control_relabel_year,
                    wy.pre_year AS year
                FROM generalized_window_years_py wy
                JOIN candidate_units cu
                  ON cu.awlevel = wy.awlevel
                 AND cu.degree_type = wy.degree_type
                 AND cu.broad_pair_bin = wy.broad_pair_bin
                 AND CAST(cu.control_relabel_year AS INTEGER) > CAST(wy.relabel_year AS INTEGER)
            ),
            grid_source AS (
                SELECT
                    g.relabel_year,
                    g.awlevel,
                    g.degree_type,
                    g.broad_pair_bin,
                    g.unitid,
                    g.control_cip6,
                    g.control_relabel_year,
                    g.year,
                    COALESCE(a.source_total, 0.0) AS source_total
                FROM grid g
                LEFT JOIN annual_source a
                  ON a.unitid = g.unitid
                 AND a.awlevel = g.awlevel
                 AND a.degree_type = g.degree_type
                 AND a.broad_pair_bin = g.broad_pair_bin
                 AND a.year = g.year
            ),
            with_lag AS (
                SELECT
                    *,
                    LAG(source_total) OVER (
                        PARTITION BY relabel_year, awlevel, degree_type, broad_pair_bin, unitid
                        ORDER BY year
                    ) AS prev_source_total
                FROM grid_source
            )
            SELECT
                relabel_year,
                awlevel,
                degree_type,
                broad_pair_bin,
                unitid,
                control_cip6,
                control_relabel_year,
                AVG(source_total) AS control_pre_avg_level,
                COALESCE(
                    AVG(source_total - prev_source_total)
                    FILTER (WHERE prev_source_total IS NOT NULL),
                    0.0
                ) AS control_pre_avg_growth,
                COALESCE(
                    MAX(CASE WHEN year = relabel_year - 1 THEN source_total END),
                    0.0
                ) AS control_source_group_pre_size
            FROM with_lag
            GROUP BY relabel_year, awlevel, degree_type, broad_pair_bin, unitid, control_cip6, control_relabel_year
            """
        ).df()

    if control_group == CONTROL_GROUP_ALWAYS_STEM:
        return con.sql(
            f"""
            WITH annual_source AS (
                SELECT
                    CAST(i.unitid AS BIGINT) AS unitid,
                    CAST(i.awlevel AS INTEGER) AS awlevel,
                    g.degree_type,
                    CAST(i.year AS INTEGER) AS year,
                    g.broad_pair_bin,
                    s.control_cip6,
                    CAST(NULL AS INTEGER) AS control_relabel_year,
                    SUM(CAST(i.ctotalt AS DOUBLE)) AS source_total
                FROM ipeds_raw i
                JOIN generalized_always_stem_cips_py s
                  ON LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = s.control_cip6
                JOIN generalized_matching_groups_py g
                  ON CAST(i.awlevel AS INTEGER) = g.awlevel
                LEFT JOIN generalized_treated_group_unitids_py tgu
                  ON CAST(i.unitid AS BIGINT) = CAST(tgu.treated_unitid AS BIGINT)
                 AND g.degree_type = tgu.degree_type
                 AND g.broad_pair_bin = tgu.broad_pair_bin
                WHERE tgu.treated_unitid IS NULL
                  AND COALESCE(CAST(i.share_intl AS DOUBLE), 0) >= {float(min_share_intl)}
                  AND CAST(i.year AS INTEGER) BETWEEN {int(min_pre_year)} AND {int(max_pre_year)}
                GROUP BY
                    CAST(i.unitid AS BIGINT),
                    CAST(i.awlevel AS INTEGER),
                    g.degree_type,
                    CAST(i.year AS INTEGER),
                    g.broad_pair_bin,
                    s.control_cip6
            ),
            candidate_units AS (
                SELECT DISTINCT unitid, awlevel, degree_type, broad_pair_bin, control_cip6, control_relabel_year
                FROM annual_source
            ),
            grid AS (
                SELECT
                    wy.relabel_year,
                    wy.awlevel,
                    wy.degree_type,
                    wy.broad_pair_bin,
                    cu.unitid,
                    cu.control_cip6,
                    cu.control_relabel_year,
                    wy.pre_year AS year
                FROM generalized_window_years_py wy
                JOIN candidate_units cu
                  ON cu.awlevel = wy.awlevel
                 AND cu.degree_type = wy.degree_type
                 AND cu.broad_pair_bin = wy.broad_pair_bin
            ),
            grid_source AS (
                SELECT
                    g.relabel_year,
                    g.awlevel,
                    g.degree_type,
                    g.broad_pair_bin,
                    g.unitid,
                    g.control_cip6,
                    g.control_relabel_year,
                    g.year,
                    COALESCE(a.source_total, 0.0) AS source_total
                FROM grid g
                LEFT JOIN annual_source a
                  ON a.unitid = g.unitid
                 AND a.awlevel = g.awlevel
                 AND a.degree_type = g.degree_type
                 AND a.broad_pair_bin = g.broad_pair_bin
                 AND a.control_cip6 = g.control_cip6
                 AND a.year = g.year
            ),
            with_lag AS (
                SELECT
                    *,
                    LAG(source_total) OVER (
                        PARTITION BY relabel_year, awlevel, degree_type, broad_pair_bin, unitid, control_cip6
                        ORDER BY year
                    ) AS prev_source_total
                FROM grid_source
            )
            SELECT
                relabel_year,
                awlevel,
                degree_type,
                broad_pair_bin,
                unitid,
                control_cip6,
                control_relabel_year,
                AVG(source_total) AS control_pre_avg_level,
                COALESCE(
                    AVG(source_total - prev_source_total)
                    FILTER (WHERE prev_source_total IS NOT NULL),
                    0.0
                ) AS control_pre_avg_growth,
                COALESCE(
                    MAX(CASE WHEN year = relabel_year - 1 THEN source_total END),
                    0.0
                ) AS control_source_group_pre_size
            FROM with_lag
            GROUP BY relabel_year, awlevel, degree_type, broad_pair_bin, unitid, control_cip6, control_relabel_year
            """
        ).df()

    return pd.DataFrame()


def match_treated_to_never_treated(
    con: ddb.DuckDBPyConnection,
    relabel_panel: pd.DataFrame,
    *,
    ipeds_path: str | Path = base.IPEDS_PATH,
    control_group: str = DEFAULT_CONTROL_GROUP,
    min_share_intl: float = STRICT_THRESHOLDS["min_share_intl"],
    control_guard_thresholds: dict[str, float] | None = None,
    use_replacement: bool = True,
    lookback_years: int = int(v2.LOOKBACK_YEARS),
    size_caliper_min_ratio: float = CONTROL_MATCH_SIZE_CALIPER_MIN_RATIO,
    size_caliper_max_ratio: float = CONTROL_MATCH_SIZE_CALIPER_MAX_RATIO,
    size_caliper_min_treated_level: float = CONTROL_MATCH_SIZE_CALIPER_MIN_TREATED_LEVEL,
) -> pd.DataFrame:
    control_group = _normalize_control_group(control_group)
    events = relabel_panel[
        relabel_panel["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"])
        & relabel_panel["event_flag"].eq(1)
        & relabel_panel["broad_bin_eligible"].eq(1)
    ].copy()
    events = _filter_relabel_analysis_window(events)
    if events.empty:
        return pd.DataFrame()
    _ensure_ipeds_view(con, ipeds_path)
    cip_map = _load_ipeds_cip_map(ipeds_path)
    broad_membership = build_broad_bin_membership(cip_map.keys())
    broad_source_cips = _broad_membership_rows(broad_membership, side="source")
    broad_target_cips = _broad_membership_rows(broad_membership, side="target")
    if broad_target_cips.empty:
        broad_target_cips = pd.DataFrame(columns=["broad_pair_bin", "cip6"])
    if broad_source_cips.empty:
        return pd.DataFrame()
    late_control_events = pd.DataFrame()
    always_stem_cips = pd.DataFrame(columns=["control_cip6"])
    if control_group == CONTROL_GROUP_LATE_TREATED:
        late_control_events = _late_treated_control_events(con, ipeds_path=ipeds_path)
        if late_control_events.empty:
            return pd.DataFrame()
        late_control_events = late_control_events[
            [
                "unitid",
                "awlevel",
                "degree_type",
                "broad_pair_bin",
                "control_relabel_year",
            ]
        ].drop_duplicates()
    elif control_group == CONTROL_GROUP_ALWAYS_STEM:
        always_stem_cips = _always_stem_cip_rows(con)
        if always_stem_cips.empty:
            return pd.DataFrame()
    control_guard_thresholds = _normalize_control_guard_thresholds(control_guard_thresholds)

    treated_events = events[
        [
            "unitid",
            "awlevel",
            "degree_type",
            "relabel_year",
            "relabel_type",
            "broad_pair_bin",
            "event_source_cip6",
            "target_cip6",
            "source_total_prev",
        ]
    ].drop_duplicates()
    treated_events = _summarize_treated_preperiod_paths(
        relabel_panel,
        treated_events,
        lookback_years=lookback_years,
    )
    carnegie_rows = _unit_year_carnegie_rows(con)
    treated_events = _attach_match_carnegie(
        treated_events,
        carnegie_rows,
        unitid_col="unitid",
        year_col="relabel_year",
        prefix="treated",
    )
    treated_group_unitids = (
        treated_events[["degree_type", "broad_pair_bin", "unitid"]]
        .dropna(subset=["degree_type", "broad_pair_bin", "unitid"])
        .drop_duplicates()
        .rename(columns={"unitid": "treated_unitid"})
        .reset_index(drop=True)
    )
    if treated_group_unitids.empty:
        treated_group_unitids = pd.DataFrame(
            {
                "degree_type": pd.Series(dtype="object"),
                "broad_pair_bin": pd.Series(dtype="object"),
                "treated_unitid": pd.Series(dtype="int64"),
            }
        )
    else:
        treated_group_unitids["degree_type"] = treated_group_unitids["degree_type"].astype(str)
        treated_group_unitids["broad_pair_bin"] = treated_group_unitids["broad_pair_bin"].astype(str)
        treated_group_unitids["treated_unitid"] = pd.to_numeric(
            treated_group_unitids["treated_unitid"],
            errors="coerce",
        )
        treated_group_unitids = treated_group_unitids.dropna(subset=["treated_unitid"]).copy()
        treated_group_unitids["treated_unitid"] = treated_group_unitids["treated_unitid"].astype("int64")
    matching_groups = treated_events[["awlevel", "degree_type", "broad_pair_bin"]].drop_duplicates().reset_index(drop=True)
    window_years = pd.DataFrame(
        [
            {
                "relabel_year": int(row.relabel_year),
                "awlevel": int(row.awlevel),
                "degree_type": str(row.degree_type),
                "broad_pair_bin": str(row.broad_pair_bin),
                "pre_year": pre_year,
            }
            for row in treated_events[["relabel_year", "awlevel", "degree_type", "broad_pair_bin"]]
            .drop_duplicates()
            .itertuples(index=False)
            for pre_year in range(int(row.relabel_year) - int(lookback_years), int(row.relabel_year))
        ]
    )
    if matching_groups.empty or window_years.empty:
        return pd.DataFrame()
    min_pre_year = int(window_years["pre_year"].min())
    max_pre_year = int(window_years["pre_year"].max())
    guard_lookback_years = int(control_guard_thresholds["pre_window_years"])
    guard_lookahead_years = int(control_guard_thresholds["post_window_years"])
    data_min_year, data_max_year = _ipeds_year_bounds(con)
    min_guard_year = int(data_min_year)
    max_guard_year = int(data_max_year)
    con.register("generalized_matching_groups_py", matching_groups)
    con.register("generalized_window_years_py", window_years)
    con.register("generalized_broad_source_cips_py", broad_source_cips)
    con.register("generalized_broad_target_cips_py", broad_target_cips)
    con.register("generalized_treated_group_unitids_py", treated_group_unitids)
    if control_group == CONTROL_GROUP_LATE_TREATED:
        con.register("generalized_late_control_events_py", late_control_events)
    if control_group == CONTROL_GROUP_ALWAYS_STEM:
        con.register("generalized_always_stem_cips_py", always_stem_cips)
    candidate_pool = None
    if control_group in {CONTROL_GROUP_LATE_TREATED, CONTROL_GROUP_ALWAYS_STEM}:
        candidate_pool = _alternate_control_candidate_pool(
            con,
            control_group=control_group,
            min_share_intl=min_share_intl,
            min_pre_year=min_pre_year,
            max_pre_year=max_pre_year,
        )
    if candidate_pool is None:
        candidate_pool = con.sql(
        f"""
        WITH annual_source AS (
            SELECT
                CAST(i.unitid AS BIGINT) AS unitid,
                CAST(i.awlevel AS INTEGER) AS awlevel,
                g.degree_type,
                CAST(i.year AS INTEGER) AS year,
                s.broad_pair_bin,
                SUM(CAST(i.ctotalt AS DOUBLE)) AS source_total
            FROM ipeds_raw i
            JOIN generalized_broad_source_cips_py s
              ON LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = s.cip6
            JOIN generalized_matching_groups_py g
              ON CAST(i.awlevel AS INTEGER) = g.awlevel
             AND s.broad_pair_bin = g.broad_pair_bin
            LEFT JOIN generalized_treated_group_unitids_py tgu
              ON CAST(i.unitid AS BIGINT) = CAST(tgu.treated_unitid AS BIGINT)
             AND g.degree_type = tgu.degree_type
             AND s.broad_pair_bin = tgu.broad_pair_bin
            WHERE tgu.treated_unitid IS NULL
              AND COALESCE(CAST(i.share_intl AS DOUBLE), 0) >= {float(min_share_intl)}
              AND CAST(i.year AS INTEGER) BETWEEN {min_pre_year} AND {max_pre_year}
            GROUP BY
                CAST(i.unitid AS BIGINT),
                CAST(i.awlevel AS INTEGER),
                g.degree_type,
                CAST(i.year AS INTEGER),
                s.broad_pair_bin
        ),
        candidate_units AS (
            SELECT DISTINCT unitid, awlevel, degree_type, broad_pair_bin
            FROM annual_source
        ),
        grid AS (
            SELECT
                wy.relabel_year,
                wy.awlevel,
                wy.degree_type,
                wy.broad_pair_bin,
                cu.unitid,
                wy.pre_year AS year
            FROM generalized_window_years_py wy
            JOIN candidate_units cu
              ON cu.awlevel = wy.awlevel
             AND cu.degree_type = wy.degree_type
             AND cu.broad_pair_bin = wy.broad_pair_bin
        ),
        grid_source AS (
            SELECT
                g.relabel_year,
                g.awlevel,
                g.degree_type,
                g.broad_pair_bin,
                g.unitid,
                g.year,
                COALESCE(a.source_total, 0.0) AS source_total
            FROM grid g
            LEFT JOIN annual_source a
              ON a.unitid = g.unitid
             AND a.awlevel = g.awlevel
             AND a.degree_type = g.degree_type
             AND a.broad_pair_bin = g.broad_pair_bin
             AND a.year = g.year
        ),
        with_lag AS (
            SELECT
                *,
                LAG(source_total) OVER (
                    PARTITION BY relabel_year, awlevel, degree_type, broad_pair_bin, unitid
                    ORDER BY year
                ) AS prev_source_total
            FROM grid_source
        ),
        pre_summary AS (
        SELECT
            relabel_year,
            awlevel,
            degree_type,
            broad_pair_bin,
            unitid,
            AVG(source_total) AS control_pre_avg_level,
            COALESCE(
                AVG(source_total - prev_source_total)
                FILTER (WHERE prev_source_total IS NOT NULL),
                0.0
            ) AS control_pre_avg_growth,
            COALESCE(
                MAX(CASE WHEN year = relabel_year - 1 THEN source_total END),
                0.0
            ) AS control_source_group_pre_size
        FROM with_lag
        GROUP BY relabel_year, awlevel, degree_type, broad_pair_bin, unitid
        ),
        guard_years AS (
            SELECT * FROM generate_series({min_guard_year}, {max_guard_year}) AS y(year)
        ),
        guard_grid AS (
            SELECT
                cu.unitid,
                cu.awlevel,
                cu.degree_type,
                cu.broad_pair_bin,
                y.year
            FROM candidate_units cu
            CROSS JOIN guard_years y
        ),
        guard_source AS (
            SELECT
                CAST(i.unitid AS BIGINT) AS unitid,
                CAST(i.awlevel AS INTEGER) AS awlevel,
                s.broad_pair_bin,
                CAST(i.year AS INTEGER) AS year,
                SUM(CAST(i.ctotalt AS DOUBLE)) AS source_total
            FROM ipeds_raw i
            JOIN generalized_broad_source_cips_py s
              ON LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = s.cip6
            WHERE COALESCE(CAST(i.share_intl AS DOUBLE), 0) >= {float(control_guard_thresholds["min_program_nonresident_share"])}
              AND CAST(i.year AS INTEGER) BETWEEN {min_guard_year} AND {max_guard_year}
            GROUP BY
                CAST(i.unitid AS BIGINT),
                CAST(i.awlevel AS INTEGER),
                s.broad_pair_bin,
                CAST(i.year AS INTEGER)
        ),
        guard_target AS (
            SELECT
                CAST(i.unitid AS BIGINT) AS unitid,
                CAST(i.awlevel AS INTEGER) AS awlevel,
                t.broad_pair_bin,
                CAST(i.year AS INTEGER) AS year,
                SUM(CAST(i.ctotalt AS DOUBLE)) AS target_total
            FROM ipeds_raw i
            JOIN generalized_broad_target_cips_py t
              ON LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = t.cip6
            WHERE COALESCE(CAST(i.share_intl AS DOUBLE), 0) >= {float(control_guard_thresholds["min_program_nonresident_share"])}
              AND CAST(i.year AS INTEGER) BETWEEN {min_guard_year} AND {max_guard_year}
            GROUP BY
                CAST(i.unitid AS BIGINT),
                CAST(i.awlevel AS INTEGER),
                t.broad_pair_bin,
                CAST(i.year AS INTEGER)
        ),
        guard_panel AS (
            SELECT
                g.unitid,
                g.awlevel,
                g.degree_type,
                g.broad_pair_bin,
                g.year,
                COALESCE(s.source_total, 0.0) AS source_total,
                COALESCE(t.target_total, 0.0) AS target_total
            FROM guard_grid g
            LEFT JOIN guard_source s
              ON s.unitid = g.unitid
             AND s.awlevel = g.awlevel
             AND s.broad_pair_bin = g.broad_pair_bin
             AND s.year = g.year
            LEFT JOIN guard_target t
              ON t.unitid = g.unitid
             AND t.awlevel = g.awlevel
             AND t.broad_pair_bin = g.broad_pair_bin
             AND t.year = g.year
        ),
        guard_stats AS (
            SELECT
                *,
                LAG(source_total) OVER (
                    PARTITION BY unitid, awlevel, degree_type, broad_pair_bin
                    ORDER BY year
                ) AS prev_source_total,
                LAG(target_total) OVER (
                    PARTITION BY unitid, awlevel, degree_type, broad_pair_bin
                    ORDER BY year
                ) AS prev_target_total,
                AVG(source_total) OVER (
                    PARTITION BY unitid, awlevel, degree_type, broad_pair_bin
                    ORDER BY year
                    ROWS BETWEEN {guard_lookback_years} PRECEDING AND 1 PRECEDING
                ) AS prev_source_window_avg,
                AVG(target_total) OVER (
                    PARTITION BY unitid, awlevel, degree_type, broad_pair_bin
                    ORDER BY year
                    ROWS BETWEEN {guard_lookback_years} PRECEDING AND 1 PRECEDING
                ) AS prev_target_window_avg,
                AVG(source_total) OVER (
                    PARTITION BY unitid, awlevel, degree_type, broad_pair_bin
                    ORDER BY year
                    ROWS BETWEEN 1 FOLLOWING AND {guard_lookahead_years} FOLLOWING
                ) AS post_source_window_avg,
                AVG(target_total) OVER (
                    PARTITION BY unitid, awlevel, degree_type, broad_pair_bin
                    ORDER BY year
                    ROWS BETWEEN 1 FOLLOWING AND {guard_lookahead_years} FOLLOWING
                ) AS post_target_window_avg
            FROM guard_panel
        ),
        guard_eval AS (
            SELECT DISTINCT
                gs.awlevel,
                gs.degree_type,
                gs.broad_pair_bin,
                gs.unitid,
                GREATEST(COALESCE(gs.prev_source_total, 0) - gs.source_total, 0) AS source_drop,
                CASE
                    WHEN COALESCE(gs.prev_source_total, 0) > 0
                    THEN (COALESCE(gs.prev_source_total, 0) - gs.source_total) / gs.prev_source_total
                    ELSE 0
                END AS source_drop_pct,
                gs.target_total - COALESCE(gs.prev_target_total, 0) AS target_increase,
                (gs.source_total + gs.target_total) - (COALESCE(gs.prev_source_total, 0) + COALESCE(gs.prev_target_total, 0)) AS net_change,
                COALESCE(gs.prev_source_window_avg, gs.prev_source_total, 0) AS source_baseline,
                COALESCE(gs.prev_target_window_avg, gs.prev_target_total, 0) AS target_baseline,
                COALESCE(gs.post_source_window_avg, gs.source_total, 0) AS source_post,
                COALESCE(gs.post_target_window_avg, gs.target_total, 0) AS target_post
            FROM guard_stats gs
        ),
        bad_control_programs AS (
            SELECT DISTINCT
                awlevel,
                degree_type,
                broad_pair_bin,
                unitid
            FROM guard_eval
            WHERE (
                source_baseline >= {float(control_guard_thresholds["min_source_pre_count"])}
                AND source_drop >= {float(control_guard_thresholds["min_source_drop_count"])}
                AND source_drop_pct >= {float(control_guard_thresholds["min_source_drop_fraction"])}
            )
            AND (
                source_baseline >= {float(control_guard_thresholds["min_source_pre_count"])}
                AND source_post <= source_baseline * (1 - {float(control_guard_thresholds["min_persistent_source_drop_fraction"])})
            )
            AND target_increase >= (
                {float(control_guard_thresholds["min_target_growth_share_of_source_drop_threshold"])}
                * {float(control_guard_thresholds["min_source_drop_count"])}
            )
            AND (
                source_drop >= {float(control_guard_thresholds["min_source_drop_count"])}
                AND net_change >= -{float(control_guard_thresholds["max_net_loss_share_of_source_drop"])} * source_drop
            )
            AND target_post >= target_baseline + (
                {float(control_guard_thresholds["min_persistent_target_gain_share_of_source_drop_threshold"])}
                * {float(control_guard_thresholds["min_source_drop_count"])}
            )
        )
        SELECT p.*
        FROM pre_summary p
        LEFT JOIN bad_control_programs b
          ON b.awlevel = p.awlevel
         AND b.degree_type = p.degree_type
         AND b.broad_pair_bin = p.broad_pair_bin
         AND b.unitid = p.unitid
        WHERE b.unitid IS NULL
        """
    ).df()
    if candidate_pool.empty:
        return pd.DataFrame()
    candidate_pool = _attach_match_carnegie(
        candidate_pool,
        carnegie_rows,
        unitid_col="unitid",
        year_col="relabel_year",
        prefix="control",
    )

    matches: list[dict[str, object]] = []
    pair_id = 0
    treated_events = treated_events.sort_values(
        ["relabel_year", "awlevel", "broad_pair_bin", "unitid", "event_source_cip6", "target_cip6"]
    ).reset_index(drop=True)
    available = candidate_pool.reset_index(drop=True).copy()
    for row in treated_events.itertuples(index=False):
        pool = available[
            (available["relabel_year"] == int(row.relabel_year))
            & (available["awlevel"] == int(row.awlevel))
            & (available["degree_type"] == str(row.degree_type))
            & (available["broad_pair_bin"] == str(row.broad_pair_bin))
        ].copy()
        if pool.empty:
            continue
        treated_carnegie = getattr(row, "treated_carnegie_basic", pd.NA)
        if pd.notna(treated_carnegie):
            pool = pool[
                pd.to_numeric(pool["control_carnegie_basic"], errors="coerce").eq(float(treated_carnegie))
            ].copy()
        else:
            pool = pool[pd.to_numeric(pool["control_carnegie_basic"], errors="coerce").isna()].copy()
        if pool.empty:
            continue
        pool["abs_pre_level_diff"] = (
            pd.to_numeric(pool["control_pre_avg_level"], errors="coerce") - float(row.treated_pre_avg_level)
        ).abs()
        pool["abs_pre_growth_diff"] = (
            pd.to_numeric(pool["control_pre_avg_growth"], errors="coerce") - float(row.treated_pre_avg_growth)
        ).abs()
        pool["abs_size_diff"] = (
            pd.to_numeric(pool["control_source_group_pre_size"], errors="coerce") - float(row.source_total_prev)
        ).abs()
        level_caliper = _passes_size_ratio_caliper(
            pool["control_pre_avg_level"],
            float(row.treated_pre_avg_level),
            min_ratio=float(size_caliper_min_ratio),
            max_ratio=float(size_caliper_max_ratio),
            min_treated_level=float(size_caliper_min_treated_level),
        )
        point_caliper = _passes_size_ratio_caliper(
            pool["control_source_group_pre_size"],
            float(row.source_total_prev),
            min_ratio=float(size_caliper_min_ratio),
            max_ratio=float(size_caliper_max_ratio),
            min_treated_level=float(size_caliper_min_treated_level),
        )
        pool = pool[level_caliper & point_caliper].copy()
        if pool.empty:
            continue
        level_scale = _match_scale(pool["control_pre_avg_level"])
        growth_scale = _match_scale(pool["control_pre_avg_growth"])
        pool["match_distance"] = (
            pool["abs_pre_level_diff"] / level_scale
            + pool["abs_pre_growth_diff"] / growth_scale
        )
        pool = pool.sort_values(
            ["match_distance", "abs_pre_level_diff", "abs_pre_growth_diff", "abs_size_diff", "unitid"],
            kind="mergesort",
        )
        chosen = pool.iloc[0]
        chosen_idx = int(chosen.name)
        pair_id += 1
        matches.append(
            {
                "pair_id": pair_id,
                "relabel_year": int(row.relabel_year),
                "relabel_type": row.relabel_type,
                "degree_type": row.degree_type,
                "awlevel": int(row.awlevel),
                "broad_pair_bin": row.broad_pair_bin,
                "source_cip6": row.event_source_cip6,
                "target_cip6": row.target_cip6,
                "treated_unitid": int(row.unitid),
                "treated_source_group_pre_size": float(row.treated_source_group_pre_size),
                "treated_pre_avg_level": float(row.treated_pre_avg_level),
                "treated_pre_avg_growth": float(row.treated_pre_avg_growth),
                "treated_carnegie_basic": (
                    int(row.treated_carnegie_basic) if pd.notna(row.treated_carnegie_basic) else pd.NA
                ),
                "treated_carnegie_basic_label": row.treated_carnegie_basic_label,
                "control_unitid": int(chosen["unitid"]),
                "control_group": control_group,
                "control_cip6": (
                    str(chosen["control_cip6"]).zfill(6)
                    if "control_cip6" in chosen.index and pd.notna(chosen["control_cip6"])
                    else pd.NA
                ),
                "control_relabel_year": (
                    int(chosen["control_relabel_year"])
                    if "control_relabel_year" in chosen.index and pd.notna(chosen["control_relabel_year"])
                    else pd.NA
                ),
                "control_source_group_pre_size": float(chosen["control_source_group_pre_size"]),
                "control_pre_avg_level": float(chosen["control_pre_avg_level"]),
                "control_pre_avg_growth": float(chosen["control_pre_avg_growth"]),
                "control_carnegie_basic": (
                    int(chosen["control_carnegie_basic"])
                    if "control_carnegie_basic" in chosen.index and pd.notna(chosen["control_carnegie_basic"])
                    else pd.NA
                ),
                "control_carnegie_basic_label": (
                    chosen["control_carnegie_basic_label"]
                    if "control_carnegie_basic_label" in chosen.index
                    else pd.NA
                ),
                "abs_pre_level_diff": float(chosen["abs_pre_level_diff"]),
                "abs_pre_growth_diff": float(chosen["abs_pre_growth_diff"]),
                "abs_size_diff": float(chosen["abs_size_diff"]),
                "match_distance": float(chosen["match_distance"]),
                "match_with_replacement": int(use_replacement),
            }
        )
        if not use_replacement:
            available = available.drop(index=chosen_idx)
    return pd.DataFrame(matches)


def compute_never_treated_control_event_time_generalized(
    con: ddb.DuckDBPyConnection,
    relabel_panel: pd.DataFrame,
    *,
    degree_type: str,
    control_group: str = DEFAULT_CONTROL_GROUP,
    foia_path: str | Path = base.FOIA_PATH,
    inst_cw_path: str | Path = base.F1_INST_CW_PATH,
    ipeds_path: str | Path = base.IPEDS_PATH,
) -> pd.DataFrame:
    degree_panel = relabel_panel[
        (relabel_panel["degree_type"] == degree_type)
        & relabel_panel["broad_bin_eligible"].eq(1)
    ].copy()
    degree_panel = _filter_relabel_analysis_window(degree_panel)
    if degree_panel.empty:
        return pd.DataFrame()
    matched_pairs = match_treated_to_never_treated(
        con,
        degree_panel,
        ipeds_path=ipeds_path,
        control_group=control_group,
    )
    if matched_pairs.empty:
        return pd.DataFrame()
    schema = _load_foia_base(con, foia_path=foia_path, inst_cw_path=inst_cw_path)
    _ensure_stem_opt_cip_view(con)
    cip_map = _load_ipeds_cip_map(ipeds_path)
    broad_membership = build_broad_bin_membership(cip_map.keys())
    broad_any_cips = _broad_membership_rows(broad_membership, side="all")
    control_cips = _pair_control_cip_rows(matched_pairs, broad_membership)
    con.register("generalized_control_pairs_py", matched_pairs)
    con.register("generalized_broad_any_cips_py", broad_any_cips)
    con.register("generalized_control_cips_py", control_cips)

    foia_inst_col = str(schema["foia_inst_col"])
    foia_cip_col = str(schema["foia_cip_col"])
    foia_end_col = str(schema["foia_end_col"])
    foia_student_col = str(schema["foia_student_col"])
    foia_tuition_col = str(schema["foia_tuition_col"])
    foia_edu_col = str(schema["foia_edu_col"])
    status_col = str(schema["status_col"])
    cw_inst_col = str(schema["cw_inst_col"])
    cw_unitid_col = str(schema["cw_unitid_col"])
    foia_year_col = schema["foia_year_col"]
    opt_end_col = str(schema["opt_end_col"])
    norm_cip_expr = base.normalize_cip_sql(foia_cip_col)
    ipeds_tuition_col = _ensure_ipeds_cost_view(
        con,
        tuition_col=DEFAULT_IPEDS_TUITION_COL,
    )
    ipeds_tuition_cols = _resolve_ipeds_cost_columns_by_degree(
        con,
        DEFAULT_IPEDS_TUITION_COL_BY_DEGREE,
        fallback_prefix="tuition",
    )
    ipeds_fee_cols = _resolve_ipeds_cost_columns_by_degree(
        con,
        DEFAULT_IPEDS_FEE_COL_BY_DEGREE,
        fallback_prefix="fee",
    )
    ipeds_tuition_expr = _ipeds_degree_cost_expr("u.degree_type", ipeds_tuition_cols)
    ipeds_fee_expr = _ipeds_degree_cost_expr("u.degree_type", ipeds_fee_cols)
    personal_funds_expr = _money_sql_expr(f"fr.{schema['students_personal_funds_col']}") if schema.get("students_personal_funds_col") else "NULL"
    school_funds_expr = _money_sql_expr(f"fr.{schema['funds_from_this_school_col']}") if schema.get("funds_from_this_school_col") else "NULL"
    other_funds_expr = _money_sql_expr(f"fr.{schema['funds_from_other_sources_col']}") if schema.get("funds_from_other_sources_col") else "NULL"
    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )
    grad_year_max_clause = f"AND CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) <= {int(ANALYSIS_ORIGINAL_YEAR_MAX)}"
    degree_case = _foia_degree_case("p.degree_type")

    control_calendar = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0') AS cip6,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                CASE
                    WHEN stem.first_stem_year IS NOT NULL
                     AND stem.first_stem_year <= CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)
                    THEN 1 ELSE 0
                END AS stem_cip_eligible_ind,
                employer_name,
                employment_opt_type,
                {opt_end_col} AS opt_end_date,
                {foia_tuition_col} AS tuition,
                {personal_funds_expr} AS students_personal_funds,
                {school_funds_expr} AS funds_from_this_school,
                {other_funds_expr} AS funds_from_other_sources,
                {foia_end_col} AS program_end_date,
                {status_col} AS requested_status,
                {foia_edu_col} AS foia_degree_label
            FROM foia_raw_with_rownum fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            LEFT JOIN stem_opt_cip_first_year stem
              ON stem.cip6 = LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0')
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
        matched_control AS (
            SELECT
                f.*,
                p.pair_id,
                p.relabel_year,
                p.relabel_type,
                p.degree_type,
                p.broad_pair_bin
            FROM relevant_foia f
            JOIN generalized_control_pairs_py p
              ON f.unitid = p.control_unitid
             AND f.foia_degree_label = {degree_case}
            JOIN generalized_control_cips_py m
              ON m.pair_id = p.pair_id
             AND f.cip6 = m.cip6
        ),
        control_ipeds AS (
            SELECT
                p.pair_id,
                CAST(i.unitid AS BIGINT) AS unitid,
                CAST(i.year AS INTEGER) AS calendar_year,
                p.relabel_year,
                p.relabel_type,
                p.degree_type,
                p.broad_pair_bin,
                SUM(CAST(i.ctotalt AS DOUBLE)) AS ctotalt,
                SUM(CAST(i.cnralt AS DOUBLE)) AS cnralt
            FROM ipeds_raw i
            JOIN generalized_control_pairs_py p
              ON CAST(i.unitid AS BIGINT) = p.control_unitid
             AND CAST(i.awlevel AS INTEGER) = p.awlevel
            JOIN generalized_control_cips_py m
              ON m.pair_id = p.pair_id
             AND LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = m.cip6
            WHERE CAST(i.year AS INTEGER) <= {int(ANALYSIS_ORIGINAL_YEAR_MAX)}
            GROUP BY
                p.pair_id,
                CAST(i.unitid AS BIGINT),
                CAST(i.year AS INTEGER),
                p.relabel_year,
                p.relabel_type,
                p.degree_type,
                p.broad_pair_bin
        ),
        control_unit_year_counts AS (
            SELECT
                pair_id,
                unitid,
                grad_year AS calendar_year,
                relabel_year,
                relabel_type,
                degree_type,
                COUNT(DISTINCT student_id) AS total_grads_unit
            FROM matched_control
            GROUP BY pair_id, unitid, grad_year, relabel_year, relabel_type, degree_type
        ),
        control_ipeds_tuition AS (
            SELECT
                u.pair_id,
                u.unitid,
                u.calendar_year,
                u.relabel_year,
                u.relabel_type,
                u.degree_type,
                SUM(u.total_grads_unit * {ipeds_tuition_expr}) AS tuition_ipeds_total,
                SUM(u.total_grads_unit * {ipeds_fee_expr}) AS fees_ipeds_total
            FROM control_unit_year_counts u
            LEFT JOIN ipeds_cost_raw ic
              ON CAST(ic.unitid AS BIGINT) = CAST(u.unitid AS BIGINT)
             AND CAST(ic.year AS INTEGER) = CAST(u.calendar_year AS INTEGER)
            GROUP BY u.pair_id, u.unitid, u.calendar_year, u.relabel_year, u.relabel_type, u.degree_type
        ),
        student_level AS (
            SELECT
                pair_id,
                unitid,
                grad_year,
                relabel_year,
                relabel_type,
                degree_type,
                MAX(stem_cip_eligible_ind) AS stem_cip_eligible_ind,
                MAX(CASE WHEN employer_name IS NOT NULL THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL AND MAX(program_end_date) IS NOT NULL
                    THEN GREATEST(
                        DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25,
                        0.0
                    )
                    ELSE 0
                END AS post_grad_authorization_years,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL AND MAX(program_end_date) IS NOT NULL
                    THEN GREATEST(
                        DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25,
                        0.0
                    )
                    ELSE 0
                END AS opt_duration_years,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                AVG(students_personal_funds) AS avg_students_personal_funds,
                AVG(
                    COALESCE(students_personal_funds, 0.0)
                    + COALESCE(funds_from_this_school, 0.0)
                    + COALESCE(funds_from_other_sources, 0.0)
                ) AS avg_total_funds,
                student_id
            FROM matched_control
            GROUP BY pair_id, unitid, grad_year, relabel_year, relabel_type, degree_type, student_id
        ),
        calendar_level AS (
            SELECT
                student_level.grad_year AS calendar_year,
                student_level.relabel_year,
                student_level.relabel_type,
                student_level.degree_type,
                AVG(student_level.avg_tuition) AS avg_tuition,
                COUNT(DISTINCT student_level.student_id) AS total_grads,
                COUNT(DISTINCT CASE WHEN student_level.stem_cip_eligible_ind = 1 THEN student_level.student_id END) AS stem_cip_eligible_users,
                COUNT(DISTINCT CASE WHEN student_level.opt_ind = 1 THEN student_level.student_id END) AS opt_users,
                COUNT(DISTINCT CASE WHEN student_level.opt_stem_ind = 1 THEN student_level.student_id END) AS opt_stem_users,
                COUNT(DISTINCT CASE WHEN student_level.status_change_ind = 1 THEN student_level.student_id END) AS status_change_users,
                SUM(student_level.post_grad_authorization_years) AS total_post_grad_authorization_years,
                SUM(student_level.opt_duration_years) AS total_opt_duration_years,
                AVG(student_level.avg_students_personal_funds) AS avg_students_personal_funds,
                AVG(student_level.avg_total_funds) AS avg_total_funds,
                MAX(ci.ctotalt) AS ctotalt,
                MAX(ci.cnralt) AS cnralt,
                COALESCE(SUM(t.tuition_ipeds_total), 0.0) AS tuition_ipeds_total,
                COALESCE(SUM(t.fees_ipeds_total), 0.0) AS fees_ipeds_total
            FROM student_level
            LEFT JOIN control_ipeds ci
              ON student_level.pair_id = ci.pair_id
             AND student_level.unitid = ci.unitid
             AND student_level.grad_year = ci.calendar_year
             AND student_level.relabel_year = ci.relabel_year
             AND student_level.relabel_type = ci.relabel_type
             AND student_level.degree_type = ci.degree_type
            LEFT JOIN control_ipeds_tuition t
              ON student_level.pair_id = t.pair_id
             AND student_level.unitid = t.unitid
             AND student_level.grad_year = t.calendar_year
             AND student_level.relabel_year = t.relabel_year
             AND student_level.relabel_type = t.relabel_type
             AND student_level.degree_type = t.degree_type
            GROUP BY
                student_level.grad_year,
                student_level.relabel_year,
                student_level.relabel_type,
                student_level.degree_type
        )
        SELECT
            calendar_year,
            relabel_year,
            relabel_type,
            degree_type,
            avg_tuition,
            total_grads,
            stem_cip_eligible_users,
            opt_users,
            opt_stem_users,
            status_change_users,
            total_post_grad_authorization_years,
            total_opt_duration_years,
            ctotalt,
            cnralt,
            tuition_ipeds_total,
            fees_ipeds_total,
            avg_students_personal_funds,
            avg_total_funds
        FROM calendar_level
        """
    ).df()
    if control_calendar.empty:
        return control_calendar

    control_calendar["opt_share"] = _safe_share(control_calendar["opt_users"], control_calendar["total_grads"])
    control_calendar["stem_cip_eligible_share"] = _safe_share(control_calendar["stem_cip_eligible_users"], control_calendar["total_grads"])
    control_calendar["opt_stem_share"] = _safe_share(control_calendar["opt_stem_users"], control_calendar["total_grads"])
    control_calendar["status_change_share"] = _safe_share(
        control_calendar["status_change_users"],
        control_calendar["total_grads"],
    )
    control_calendar["post_grad_authorization_years_avg"] = _safe_share(
        control_calendar["total_post_grad_authorization_years"],
        control_calendar["total_grads"],
    )
    control_calendar["opt_duration_years_avg"] = _safe_share(
        control_calendar["total_opt_duration_years"],
        control_calendar["total_grads"],
    )
    control_calendar["opt_years_avg"] = control_calendar["post_grad_authorization_years_avg"]
    control_calendar["f1_share_of_ctotalt"] = _safe_share(control_calendar["total_grads"], control_calendar["ctotalt"])
    control_calendar["f1_share_of_cnralt"] = _safe_share(control_calendar["total_grads"], control_calendar["cnralt"])
    control_calendar["tuition_total"] = control_calendar["avg_tuition"] * control_calendar["total_grads"]
    control_calendar["avg_tuition"] = _safe_share(control_calendar["tuition_total"], control_calendar["total_grads"])
    control_calendar["avg_tuition_ipeds"] = _safe_share(control_calendar["tuition_ipeds_total"], control_calendar["total_grads"])
    control_calendar["avg_fees_ipeds"] = _safe_share(control_calendar["fees_ipeds_total"], control_calendar["total_grads"])
    control_calendar["students_personal_funds_total"] = (
        pd.to_numeric(control_calendar["avg_students_personal_funds"], errors="coerce")
        * pd.to_numeric(control_calendar["total_grads"], errors="coerce")
    )
    control_calendar["total_funds"] = (
        pd.to_numeric(control_calendar["avg_total_funds"], errors="coerce")
        * pd.to_numeric(control_calendar["total_grads"], errors="coerce")
    )
    control_calendar["event_t"] = control_calendar["calendar_year"] - control_calendar["relabel_year"]

    control_event = (
        control_calendar.groupby(["event_t", "relabel_type", "degree_type"], as_index=False)
        .agg(
            total_grads=("total_grads", "sum"),
            stem_cip_eligible_users=("stem_cip_eligible_users", "sum"),
            opt_users=("opt_users", "sum"),
            opt_stem_users=("opt_stem_users", "sum"),
            status_change_users=("status_change_users", "sum"),
            total_post_grad_authorization_years=("total_post_grad_authorization_years", "sum"),
            total_opt_duration_years=("total_opt_duration_years", "sum"),
            tuition_total=("tuition_total", "sum"),
            tuition_ipeds_total=("tuition_ipeds_total", "sum"),
            fees_ipeds_total=("fees_ipeds_total", "sum"),
            students_personal_funds_total=("students_personal_funds_total", "sum"),
            total_funds=("total_funds", "sum"),
            ctotalt=("ctotalt", "sum"),
            cnralt=("cnralt", "sum"),
        )
    )
    control_event["opt_share"] = _safe_share(control_event["opt_users"], control_event["total_grads"])
    control_event["stem_cip_eligible_share"] = _safe_share(control_event["stem_cip_eligible_users"], control_event["total_grads"])
    control_event["opt_stem_share"] = _safe_share(control_event["opt_stem_users"], control_event["total_grads"])
    control_event["status_change_share"] = _safe_share(
        control_event["status_change_users"],
        control_event["total_grads"],
    )
    control_event["post_grad_authorization_years_avg"] = _safe_share(
        control_event["total_post_grad_authorization_years"],
        control_event["total_grads"],
    )
    control_event["opt_duration_years_avg"] = _safe_share(
        control_event["total_opt_duration_years"],
        control_event["total_grads"],
    )
    control_event["opt_years_avg"] = control_event["post_grad_authorization_years_avg"]
    control_event["f1_share_of_ctotalt"] = _safe_share(control_event["total_grads"], control_event["ctotalt"])
    control_event["f1_share_of_cnralt"] = _safe_share(control_event["total_grads"], control_event["cnralt"])
    control_event["avg_tuition"] = _safe_share(control_event["tuition_total"], control_event["total_grads"])
    control_event["avg_tuition_ipeds"] = _safe_share(control_event["tuition_ipeds_total"], control_event["total_grads"])
    control_event["avg_fees_ipeds"] = _safe_share(control_event["fees_ipeds_total"], control_event["total_grads"])
    control_event["avg_students_personal_funds"] = _safe_share(
        control_event["students_personal_funds_total"],
        control_event["total_grads"],
    )
    control_event["avg_total_funds"] = _safe_share(control_event["total_funds"], control_event["total_grads"])
    return control_event


def compute_generalized_did_panel(
    con: ddb.DuckDBPyConnection,
    relabel_panel: pd.DataFrame,
    *,
    degree_type: str | None,
    did_spec: str = DEFAULT_DID_SPEC,
    control_group: str = DEFAULT_CONTROL_GROUP,
    foia_path: str | Path = base.FOIA_PATH,
    inst_cw_path: str | Path = base.F1_INST_CW_PATH,
    foia_person_panel_path: str | Path = DEFAULT_FOIA_PERSON_PANEL_PATH,
    employer_match_dir: str | Path = DEFAULT_EMPLOYER_MATCH_DIR,
    ipeds_path: str | Path = base.IPEDS_PATH,
) -> pd.DataFrame:
    normalized_did_spec = _normalize_did_spec(did_spec)
    individual_rows = normalized_did_spec == DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE
    if degree_type is None:
        degree_panel = relabel_panel[
            relabel_panel["degree_type"].isin(POOLED_DEGREE_TYPES)
            & relabel_panel["broad_bin_eligible"].eq(1)
        ].copy()
    else:
        degree_panel = relabel_panel[
            (relabel_panel["degree_type"] == degree_type)
            & relabel_panel["broad_bin_eligible"].eq(1)
        ].copy()
    degree_panel = _filter_relabel_analysis_window(degree_panel)
    if degree_panel.empty:
        return pd.DataFrame()
    matched_pairs = match_treated_to_never_treated(
        con,
        degree_panel,
        ipeds_path=ipeds_path,
        control_group=control_group,
    )
    if matched_pairs.empty:
        return pd.DataFrame()
    schema = _load_foia_base(
        con,
        foia_path=foia_path,
        inst_cw_path=inst_cw_path,
        foia_person_panel_path=foia_person_panel_path,
        employer_match_dir=employer_match_dir,
    )
    _ensure_ipeds_view(con, ipeds_path)
    _ensure_stem_opt_cip_view(con)
    _stage_foia_analysis_base(con, schema=schema)
    cip_map = _load_ipeds_cip_map(ipeds_path)
    broad_membership = build_broad_bin_membership(cip_map.keys())
    broad_any_cips = _broad_membership_rows(broad_membership, side="all")
    control_cips = _pair_control_cip_rows(matched_pairs, broad_membership)
    panel_rows = degree_panel.copy()
    relabel_events = panel_rows[
        [
            "unitid",
            "awlevel",
            "year",
            "relabel_year",
            "relabel_type",
            "broad_pair_bin",
            "degree_type",
            "ctotalt",
            "cnralt",
        ]
    ].drop_duplicates()
    con.register("generalized_did_pairs_py", matched_pairs)
    con.register("generalized_did_events_py", relabel_events)
    con.register("generalized_broad_any_cips_py", broad_any_cips)
    con.register("generalized_control_cips_py", control_cips)

    foia_inst_col = str(schema["foia_inst_col"])
    foia_cip_col = str(schema["foia_cip_col"])
    foia_end_col = str(schema["foia_end_col"])
    foia_student_col = str(schema["foia_student_col"])
    foia_tuition_col = str(schema["foia_tuition_col"])
    foia_edu_col = str(schema["foia_edu_col"])
    status_col = str(schema["status_col"])
    cw_inst_col = str(schema["cw_inst_col"])
    cw_unitid_col = str(schema["cw_unitid_col"])
    foia_year_col = schema["foia_year_col"]
    opt_end_col = str(schema["opt_end_col"])
    norm_cip_expr = base.normalize_cip_sql(foia_cip_col)
    ipeds_tuition_col = _ensure_ipeds_cost_view(
        con,
        tuition_col=DEFAULT_IPEDS_TUITION_COL,
    )
    ipeds_tuition_cols = _resolve_ipeds_cost_columns_by_degree(
        con,
        DEFAULT_IPEDS_TUITION_COL_BY_DEGREE,
        fallback_prefix="tuition",
    )
    ipeds_fee_cols = _resolve_ipeds_cost_columns_by_degree(
        con,
        DEFAULT_IPEDS_FEE_COL_BY_DEGREE,
        fallback_prefix="fee",
    )
    ipeds_tuition_expr = _ipeds_degree_cost_expr("u.degree_type", ipeds_tuition_cols)
    ipeds_fee_expr = _ipeds_degree_cost_expr("u.degree_type", ipeds_fee_cols)
    personal_funds_expr = _money_sql_expr(f"fr.{schema['students_personal_funds_col']}") if schema.get("students_personal_funds_col") else "NULL"
    school_funds_expr = _money_sql_expr(f"fr.{schema['funds_from_this_school_col']}") if schema.get("funds_from_this_school_col") else "NULL"
    other_funds_expr = _money_sql_expr(f"fr.{schema['funds_from_other_sources_col']}") if schema.get("funds_from_other_sources_col") else "NULL"
    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )
    grad_year_max_clause = f"AND CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) <= {int(ANALYSIS_ORIGINAL_YEAR_MAX)}"
    degree_case_events = _foia_degree_case("r.degree_type")
    degree_case_pairs = _foia_degree_case("p.degree_type")
    student_did_outcome_ctes = _student_employer_outcome_ctes(
        source_table="matched_treated",
        group_cols=[
            "pair_id",
            "unitid",
            "grad_year",
            "relabel_year",
            "relabel_type",
            "broad_pair_bin",
            "degree_type",
            "student_id",
        ],
    )
    treated_select = """
        SELECT
            s.pair_id,
            s.unitid,
            s.grad_year AS calendar_year,
            s.relabel_year,
            s.relabel_type,
            s.broad_pair_bin,
            s.degree_type,
            CAST(s.student_id AS VARCHAR) AS student_id,
            1.0 AS total_grads,
            CAST(s.stem_cip_eligible_ind AS DOUBLE) AS stem_cip_eligible_users,
            CAST(s.opt_ind AS DOUBLE) AS opt_users,
            CAST(s.opt_stem_ind AS DOUBLE) AS opt_stem_users,
            CAST(s.status_change_ind AS DOUBLE) AS status_change_users,
            CAST(s.post_grad_authorization_years AS DOUBLE) AS total_post_grad_authorization_years,
            CAST(s.opt_duration_years AS DOUBLE) AS total_opt_duration_years,
            CAST(s.avg_tuition AS DOUBLE) AS avg_tuition,
            CAST(s.avg_students_personal_funds AS DOUBLE) AS avg_students_personal_funds,
            CAST(s.avg_total_funds AS DOUBLE) AS avg_total_funds,
            CAST(s.unique_employers AS DOUBLE) AS unique_employers,
            CAST(s.unique_opt_cities AS DOUBLE) AS unique_opt_cities,
            CAST(s.auth_employment_tenure_years AS DOUBLE) AS auth_employment_tenure_years,
            CAST(s.employer_opt_intensity_pctile AS DOUBLE) AS employer_opt_intensity_pctile,
            CAST(s.internship_count AS DOUBLE) AS total_internships,
            CAST(s.internship_opt_years AS DOUBLE) AS total_internship_opt_years,
            COALESCE(ci.ctotalt, 0.0) AS ctotalt,
            COALESCE(ci.cnralt, 0.0) AS cnralt,
            CASE
                WHEN COALESCE(uy.total_grads_unit, 0) > 0
                THEN COALESCE(t.tuition_ipeds_total, 0.0) / CAST(uy.total_grads_unit AS DOUBLE)
                ELSE NULL
            END AS tuition_ipeds_total,
            CASE
                WHEN COALESCE(uy.total_grads_unit, 0) > 0
                THEN COALESCE(t.fees_ipeds_total, 0.0) / CAST(uy.total_grads_unit AS DOUBLE)
                ELSE NULL
            END AS fees_ipeds_total
        FROM student_level s
        LEFT JOIN treated_ipeds ci
          ON s.pair_id = ci.pair_id
         AND s.unitid = ci.unitid
         AND s.grad_year = ci.calendar_year
         AND s.relabel_year = ci.relabel_year
         AND s.relabel_type = ci.relabel_type
        LEFT JOIN treated_ipeds_tuition t
          ON s.pair_id = t.pair_id
         AND s.unitid = t.unitid
         AND s.grad_year = t.calendar_year
         AND s.relabel_year = t.relabel_year
         AND s.relabel_type = t.relabel_type
         AND s.degree_type = t.degree_type
        LEFT JOIN treated_unit_year_counts uy
          ON s.pair_id = uy.pair_id
         AND s.unitid = uy.unitid
         AND s.grad_year = uy.calendar_year
         AND s.relabel_year = uy.relabel_year
         AND s.relabel_type = uy.relabel_type
         AND s.degree_type = uy.degree_type
    """ if individual_rows else """
        SELECT
            student_level.pair_id,
            student_level.unitid,
            student_level.grad_year AS calendar_year,
            student_level.relabel_year,
            student_level.relabel_type,
            student_level.broad_pair_bin,
            AVG(student_level.avg_tuition) AS avg_tuition,
            COUNT(DISTINCT student_level.student_id) AS total_grads,
            COUNT(DISTINCT CASE WHEN student_level.stem_cip_eligible_ind = 1 THEN student_level.student_id END) AS stem_cip_eligible_users,
            COUNT(DISTINCT CASE WHEN student_level.opt_ind = 1 THEN student_level.student_id END) AS opt_users,
            COUNT(DISTINCT CASE WHEN student_level.opt_stem_ind = 1 THEN student_level.student_id END) AS opt_stem_users,
            COUNT(DISTINCT CASE WHEN student_level.status_change_ind = 1 THEN student_level.student_id END) AS status_change_users,
            SUM(student_level.post_grad_authorization_years) AS total_post_grad_authorization_years,
            SUM(student_level.opt_duration_years) AS total_opt_duration_years,
            AVG(student_level.avg_students_personal_funds) AS avg_students_personal_funds,
            AVG(student_level.avg_total_funds) AS avg_total_funds,
            AVG(student_level.unique_employers) AS unique_employers,
            AVG(student_level.unique_opt_cities) AS unique_opt_cities,
            AVG(student_level.auth_employment_tenure_years) AS auth_employment_tenure_years,
            AVG(student_level.employer_opt_intensity_pctile) AS employer_opt_intensity_pctile,
            SUM(student_level.internship_count) AS total_internships,
            SUM(student_level.internship_opt_years) AS total_internship_opt_years,
            MAX(ci.ctotalt) AS ctotalt,
            MAX(ci.cnralt) AS cnralt,
            COALESCE(MAX(t.tuition_ipeds_total), 0.0) AS tuition_ipeds_total,
            COALESCE(MAX(t.fees_ipeds_total), 0.0) AS fees_ipeds_total
        FROM student_level
        LEFT JOIN treated_ipeds ci
          ON student_level.pair_id = ci.pair_id
         AND student_level.unitid = ci.unitid
         AND student_level.grad_year = ci.calendar_year
         AND student_level.relabel_year = ci.relabel_year
         AND student_level.relabel_type = ci.relabel_type
        LEFT JOIN treated_ipeds_tuition t
          ON student_level.pair_id = t.pair_id
         AND student_level.unitid = t.unitid
         AND student_level.grad_year = t.calendar_year
         AND student_level.relabel_year = t.relabel_year
         AND student_level.relabel_type = t.relabel_type
         AND student_level.degree_type = t.degree_type
        GROUP BY
            student_level.pair_id,
            student_level.unitid,
            student_level.grad_year,
            student_level.relabel_year,
            student_level.relabel_type,
            student_level.broad_pair_bin
    """

    treated_calendar = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                fr.original_row_num,
                cw.{cw_unitid_col} AS unitid,
                LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0') AS cip6,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                CASE
                    WHEN stem.first_stem_year IS NOT NULL
                     AND stem.first_stem_year <= CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)
                    THEN 1 ELSE 0
                END AS stem_cip_eligible_ind,
                employer_name,
                employment_opt_type,
                {opt_end_col} AS opt_end_date,
                {foia_tuition_col} AS tuition,
                {personal_funds_expr} AS students_personal_funds,
                {school_funds_expr} AS funds_from_this_school,
                {other_funds_expr} AS funds_from_other_sources,
                {foia_end_col} AS program_end_date,
                {status_col} AS requested_status,
                {foia_edu_col} AS foia_degree_label
            FROM foia_raw_with_rownum fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            LEFT JOIN stem_opt_cip_first_year stem
              ON stem.cip6 = LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0')
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
        treated_flagged AS (
            SELECT
                f.*,
                r.ctotalt,
                r.cnralt,
                r.relabel_year,
                r.relabel_type,
                r.degree_type,
                r.broad_pair_bin
            FROM relevant_foia f
            JOIN generalized_did_events_py r
              ON f.unitid = r.unitid
             AND f.grad_year = r.year
             AND f.foia_degree_label = {degree_case_events}
            JOIN generalized_broad_any_cips_py m
              ON m.broad_pair_bin = r.broad_pair_bin
             AND f.cip6 = m.cip6
        ),
        matched_treated AS (
            SELECT
                tf.*,
                p.pair_id
            FROM treated_flagged tf
            JOIN generalized_did_pairs_py p
              ON tf.unitid = p.treated_unitid
             AND tf.relabel_year = p.relabel_year
             AND tf.relabel_type = p.relabel_type
             AND tf.degree_type = p.degree_type
             AND tf.broad_pair_bin = p.broad_pair_bin
        ),
        student_level_base AS (
            SELECT
                pair_id,
                unitid,
                grad_year,
                relabel_year,
                relabel_type,
                broad_pair_bin,
                degree_type,
                MAX(stem_cip_eligible_ind) AS stem_cip_eligible_ind,
                MAX(CASE WHEN employer_name IS NOT NULL THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL AND MAX(program_end_date) IS NOT NULL
                    THEN GREATEST(
                        DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25,
                        0.0
                    )
                    ELSE 0
                END AS post_grad_authorization_years,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                AVG(students_personal_funds) AS avg_students_personal_funds,
                AVG(
                    COALESCE(students_personal_funds, 0.0)
                    + COALESCE(funds_from_this_school, 0.0)
                    + COALESCE(funds_from_other_sources, 0.0)
                ) AS avg_total_funds,
                student_id
            FROM matched_treated
            GROUP BY pair_id, unitid, grad_year, relabel_year, relabel_type, broad_pair_bin, degree_type, student_id
        ),
        {student_did_outcome_ctes}
        treated_unit_year_counts AS (
            SELECT
                pair_id,
                unitid,
                grad_year AS calendar_year,
                relabel_year,
                relabel_type,
                degree_type,
                COUNT(DISTINCT student_id) AS total_grads_unit
            FROM student_level
            GROUP BY pair_id, unitid, grad_year, relabel_year, relabel_type, degree_type
        ),
        treated_ipeds AS (
            SELECT
                p.pair_id,
                CAST(i.unitid AS BIGINT) AS unitid,
                CAST(i.year AS INTEGER) AS calendar_year,
                p.relabel_year,
                p.relabel_type,
                SUM(CAST(i.ctotalt AS DOUBLE)) AS ctotalt,
                SUM(CAST(i.cnralt AS DOUBLE)) AS cnralt
            FROM ipeds_raw i
            JOIN generalized_did_pairs_py p
              ON CAST(i.unitid AS BIGINT) = p.treated_unitid
             AND CAST(i.awlevel AS INTEGER) = p.awlevel
            JOIN generalized_broad_any_cips_py m
              ON m.broad_pair_bin = p.broad_pair_bin
             AND LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = m.cip6
            WHERE CAST(i.year AS INTEGER) <= {int(ANALYSIS_ORIGINAL_YEAR_MAX)}
            GROUP BY
                p.pair_id,
                CAST(i.unitid AS BIGINT),
                CAST(i.year AS INTEGER),
                p.relabel_year,
                p.relabel_type
        ),
        treated_ipeds_tuition AS (
            SELECT
                u.pair_id,
                u.unitid,
                u.calendar_year,
                u.relabel_year,
                u.relabel_type,
                u.degree_type,
                SUM(u.total_grads_unit * {ipeds_tuition_expr}) AS tuition_ipeds_total,
                SUM(u.total_grads_unit * {ipeds_fee_expr}) AS fees_ipeds_total
            FROM treated_unit_year_counts u
            LEFT JOIN ipeds_cost_raw ic
              ON CAST(ic.unitid AS BIGINT) = CAST(u.unitid AS BIGINT)
             AND CAST(ic.year AS INTEGER) = CAST(u.calendar_year AS INTEGER)
            GROUP BY u.pair_id, u.unitid, u.calendar_year, u.relabel_year, u.relabel_type, u.degree_type
        )
        {treated_select}
        """
    ).df()

    student_control_did_outcome_ctes = _student_employer_outcome_ctes(
        source_table="matched_control",
        group_cols=[
            "pair_id",
            "unitid",
            "grad_year",
            "relabel_year",
            "relabel_type",
            "broad_pair_bin",
            "degree_type",
            "student_id",
        ],
    )
    control_select = """
        SELECT
            s.pair_id,
            s.unitid,
            s.grad_year AS calendar_year,
            s.relabel_year,
            s.relabel_type,
            s.broad_pair_bin,
            s.degree_type,
            CAST(s.student_id AS VARCHAR) AS student_id,
            1.0 AS total_grads,
            CAST(s.stem_cip_eligible_ind AS DOUBLE) AS stem_cip_eligible_users,
            CAST(s.opt_ind AS DOUBLE) AS opt_users,
            CAST(s.opt_stem_ind AS DOUBLE) AS opt_stem_users,
            CAST(s.status_change_ind AS DOUBLE) AS status_change_users,
            CAST(s.post_grad_authorization_years AS DOUBLE) AS total_post_grad_authorization_years,
            CAST(s.opt_duration_years AS DOUBLE) AS total_opt_duration_years,
            CAST(s.avg_tuition AS DOUBLE) AS avg_tuition,
            CAST(s.avg_students_personal_funds AS DOUBLE) AS avg_students_personal_funds,
            CAST(s.avg_total_funds AS DOUBLE) AS avg_total_funds,
            CAST(s.unique_employers AS DOUBLE) AS unique_employers,
            CAST(s.unique_opt_cities AS DOUBLE) AS unique_opt_cities,
            CAST(s.auth_employment_tenure_years AS DOUBLE) AS auth_employment_tenure_years,
            CAST(s.employer_opt_intensity_pctile AS DOUBLE) AS employer_opt_intensity_pctile,
            CAST(s.internship_count AS DOUBLE) AS total_internships,
            CAST(s.internship_opt_years AS DOUBLE) AS total_internship_opt_years,
            COALESCE(ci.ctotalt, 0.0) AS ctotalt,
            COALESCE(ci.cnralt, 0.0) AS cnralt,
            CASE
                WHEN COALESCE(uy.total_grads_unit, 0) > 0
                THEN COALESCE(t.tuition_ipeds_total, 0.0) / CAST(uy.total_grads_unit AS DOUBLE)
                ELSE NULL
            END AS tuition_ipeds_total,
            CASE
                WHEN COALESCE(uy.total_grads_unit, 0) > 0
                THEN COALESCE(t.fees_ipeds_total, 0.0) / CAST(uy.total_grads_unit AS DOUBLE)
                ELSE NULL
            END AS fees_ipeds_total
        FROM student_level s
        LEFT JOIN control_ipeds ci
          ON s.pair_id = ci.pair_id
         AND s.unitid = ci.unitid
         AND s.grad_year = ci.calendar_year
         AND s.relabel_year = ci.relabel_year
         AND s.relabel_type = ci.relabel_type
        LEFT JOIN control_ipeds_tuition t
          ON s.pair_id = t.pair_id
         AND s.unitid = t.unitid
         AND s.grad_year = t.calendar_year
         AND s.relabel_year = t.relabel_year
         AND s.relabel_type = t.relabel_type
         AND s.degree_type = t.degree_type
        LEFT JOIN control_unit_year_counts uy
          ON s.pair_id = uy.pair_id
         AND s.unitid = uy.unitid
         AND s.grad_year = uy.calendar_year
         AND s.relabel_year = uy.relabel_year
         AND s.relabel_type = uy.relabel_type
         AND s.degree_type = uy.degree_type
    """ if individual_rows else """
        SELECT
            s.pair_id,
            s.unitid,
            s.grad_year AS calendar_year,
            s.relabel_year,
            s.relabel_type,
            s.broad_pair_bin,
            AVG(s.avg_tuition) AS avg_tuition,
            COUNT(DISTINCT s.student_id) AS total_grads,
            COUNT(DISTINCT CASE WHEN s.stem_cip_eligible_ind = 1 THEN s.student_id END) AS stem_cip_eligible_users,
            COUNT(DISTINCT CASE WHEN s.opt_ind = 1 THEN s.student_id END) AS opt_users,
            COUNT(DISTINCT CASE WHEN s.opt_stem_ind = 1 THEN s.student_id END) AS opt_stem_users,
            COUNT(DISTINCT CASE WHEN s.status_change_ind = 1 THEN s.student_id END) AS status_change_users,
            SUM(s.post_grad_authorization_years) AS total_post_grad_authorization_years,
            SUM(s.opt_duration_years) AS total_opt_duration_years,
            AVG(s.avg_students_personal_funds) AS avg_students_personal_funds,
            AVG(s.avg_total_funds) AS avg_total_funds,
            AVG(s.unique_employers) AS unique_employers,
            AVG(s.unique_opt_cities) AS unique_opt_cities,
            AVG(s.auth_employment_tenure_years) AS auth_employment_tenure_years,
            AVG(s.employer_opt_intensity_pctile) AS employer_opt_intensity_pctile,
            SUM(s.internship_count) AS total_internships,
            SUM(s.internship_opt_years) AS total_internship_opt_years,
            MAX(ci.ctotalt) AS ctotalt,
            MAX(ci.cnralt) AS cnralt,
            COALESCE(MAX(t.tuition_ipeds_total), 0.0) AS tuition_ipeds_total,
            COALESCE(MAX(t.fees_ipeds_total), 0.0) AS fees_ipeds_total
        FROM student_level s
        LEFT JOIN control_ipeds ci
          ON s.pair_id = ci.pair_id
         AND s.unitid = ci.unitid
         AND s.grad_year = ci.calendar_year
         AND s.relabel_year = ci.relabel_year
         AND s.relabel_type = ci.relabel_type
        LEFT JOIN control_ipeds_tuition t
          ON s.pair_id = t.pair_id
         AND s.unitid = t.unitid
         AND s.grad_year = t.calendar_year
         AND s.relabel_year = t.relabel_year
         AND s.relabel_type = t.relabel_type
         AND s.degree_type = t.degree_type
        GROUP BY s.pair_id, s.unitid, s.grad_year, s.relabel_year, s.relabel_type, s.broad_pair_bin
    """

    control_calendar = con.sql(
        f"""
        WITH control_ipeds AS (
            SELECT
                p.pair_id,
                CAST(i.unitid AS BIGINT) AS unitid,
                CAST(i.year AS INTEGER) AS calendar_year,
                p.relabel_year,
                p.relabel_type,
                SUM(CAST(i.ctotalt AS DOUBLE)) AS ctotalt,
                SUM(CAST(i.cnralt AS DOUBLE)) AS cnralt
            FROM ipeds_raw i
            JOIN generalized_did_pairs_py p
              ON CAST(i.unitid AS BIGINT) = p.control_unitid
             AND CAST(i.awlevel AS INTEGER) = p.awlevel
            JOIN generalized_control_cips_py m
              ON m.pair_id = p.pair_id
             AND LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = m.cip6
            WHERE CAST(i.year AS INTEGER) <= {int(ANALYSIS_ORIGINAL_YEAR_MAX)}
            GROUP BY
                p.pair_id,
                CAST(i.unitid AS BIGINT),
                CAST(i.year AS INTEGER),
                p.relabel_year,
                p.relabel_type
        ),
        relevant_foia AS (
            SELECT *
            FROM foia_analysis_base
        ),
        matched_control AS (
            SELECT
                f.*,
                p.pair_id,
                p.relabel_year,
                p.relabel_type,
                p.degree_type,
                p.broad_pair_bin
            FROM relevant_foia f
            JOIN generalized_did_pairs_py p
              ON f.unitid = p.control_unitid
             AND f.foia_degree_label = {degree_case_pairs}
            JOIN generalized_control_cips_py m
              ON m.pair_id = p.pair_id
             AND f.cip6 = m.cip6
        ),
        student_level_base AS (
            SELECT
                pair_id,
                unitid,
                grad_year,
                relabel_year,
                relabel_type,
                broad_pair_bin,
                degree_type,
                MAX(stem_cip_eligible_ind) AS stem_cip_eligible_ind,
                MAX(CASE WHEN employer_name IS NOT NULL THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL AND MAX(program_end_date) IS NOT NULL
                    THEN GREATEST(
                        DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25,
                        0.0
                    )
                    ELSE 0
                END AS post_grad_authorization_years,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                AVG(students_personal_funds) AS avg_students_personal_funds,
                AVG(
                    COALESCE(students_personal_funds, 0.0)
                    + COALESCE(funds_from_this_school, 0.0)
                    + COALESCE(funds_from_other_sources, 0.0)
                ) AS avg_total_funds,
                student_id
            FROM matched_control
            GROUP BY pair_id, unitid, grad_year, relabel_year, relabel_type, broad_pair_bin, degree_type, student_id
        ),
        {student_control_did_outcome_ctes}
        control_unit_year_counts AS (
            SELECT
                pair_id,
                unitid,
                grad_year AS calendar_year,
                relabel_year,
                relabel_type,
                degree_type,
                COUNT(DISTINCT student_id) AS total_grads_unit
            FROM student_level
            GROUP BY pair_id, unitid, grad_year, relabel_year, relabel_type, degree_type
        ),
        control_ipeds_tuition AS (
            SELECT
                u.pair_id,
                u.unitid,
                u.calendar_year,
                u.relabel_year,
                u.relabel_type,
                u.degree_type,
                SUM(u.total_grads_unit * {ipeds_tuition_expr}) AS tuition_ipeds_total,
                SUM(u.total_grads_unit * {ipeds_fee_expr}) AS fees_ipeds_total
            FROM control_unit_year_counts u
            LEFT JOIN ipeds_cost_raw ic
              ON CAST(ic.unitid AS BIGINT) = CAST(u.unitid AS BIGINT)
             AND CAST(ic.year AS INTEGER) = CAST(u.calendar_year AS INTEGER)
            GROUP BY u.pair_id, u.unitid, u.calendar_year, u.relabel_year, u.relabel_type, u.degree_type
        )
        {control_select}
        """
    ).df()
    if treated_calendar.empty or control_calendar.empty:
        return pd.DataFrame()

    treated_calendar["treated"] = 1
    control_calendar["treated"] = 0
    did_panel = pd.concat([treated_calendar, control_calendar], ignore_index=True)
    did_panel = _apply_did_design_columns(did_panel, did_spec=normalized_did_spec)
    did_panel["event_t"] = did_panel["calendar_year"] - did_panel["relabel_year"]
    did_panel["stem_cip_eligible_share"] = _safe_share(did_panel["stem_cip_eligible_users"], did_panel["total_grads"])
    did_panel["opt_share"] = _safe_share(did_panel["opt_users"], did_panel["total_grads"])
    did_panel["opt_stem_share"] = _safe_share(did_panel["opt_stem_users"], did_panel["total_grads"])
    did_panel["status_change_share"] = _safe_share(did_panel["status_change_users"], did_panel["total_grads"])
    did_panel["post_grad_authorization_years_avg"] = _safe_share(
        did_panel["total_post_grad_authorization_years"],
        did_panel["total_grads"],
    )
    did_panel["opt_duration_years_avg"] = _safe_share(did_panel["total_opt_duration_years"], did_panel["total_grads"])
    did_panel["opt_years_avg"] = did_panel["post_grad_authorization_years_avg"]
    did_panel["internship_count"] = _safe_share(did_panel["total_internships"], did_panel["total_grads"])
    did_panel["internship_opt_years"] = _safe_share(
        did_panel["total_internship_opt_years"],
        did_panel["total_grads"],
    )
    did_panel["f1_share_of_ctotalt"] = _safe_share(did_panel["total_grads"], did_panel["ctotalt"])
    did_panel["f1_share_of_cnralt"] = _safe_share(did_panel["total_grads"], did_panel["cnralt"])
    did_panel["cnralt_share_of_ctotalt"] = _safe_share(did_panel["cnralt"], did_panel["ctotalt"])
    did_panel["tuition_total"] = did_panel["avg_tuition"] * did_panel["total_grads"]
    did_panel["avg_tuition"] = _safe_share(did_panel["tuition_total"], did_panel["total_grads"])
    did_panel["avg_tuition_ipeds"] = _safe_share(did_panel["tuition_ipeds_total"], did_panel["total_grads"])
    did_panel["avg_fees_ipeds"] = _safe_share(did_panel["fees_ipeds_total"], did_panel["total_grads"])
    did_panel["students_personal_funds_total"] = (
        pd.to_numeric(did_panel["avg_students_personal_funds"], errors="coerce")
        * pd.to_numeric(did_panel["total_grads"], errors="coerce")
    )
    did_panel["total_funds"] = (
        pd.to_numeric(did_panel["avg_total_funds"], errors="coerce")
        * pd.to_numeric(did_panel["total_grads"], errors="coerce")
    )
    did_panel["panel_level"] = "individual" if individual_rows else "collapsed"
    if not individual_rows:
        did_panel["degree_type"] = degree_type if degree_type is not None else "Pooled"
    return did_panel


def compute_generalized_ipeds_did_panel(
    con: ddb.DuckDBPyConnection,
    relabel_panel: pd.DataFrame,
    *,
    degree_type: str | None,
    did_spec: str = DEFAULT_DID_SPEC,
    control_group: str = DEFAULT_CONTROL_GROUP,
    ipeds_path: str | Path = base.IPEDS_PATH,
) -> pd.DataFrame:
    normalized_did_spec = _normalize_did_spec(did_spec)
    if degree_type is None:
        degree_panel = relabel_panel[
            relabel_panel["degree_type"].isin(POOLED_DEGREE_TYPES)
            & relabel_panel["broad_bin_eligible"].eq(1)
        ].copy()
    else:
        degree_panel = relabel_panel[
            (relabel_panel["degree_type"] == degree_type)
            & relabel_panel["broad_bin_eligible"].eq(1)
        ].copy()
    degree_panel = _filter_relabel_analysis_window(degree_panel)
    if degree_panel.empty:
        return pd.DataFrame()
    matched_pairs = match_treated_to_never_treated(
        con,
        degree_panel,
        ipeds_path=ipeds_path,
        control_group=control_group,
    )
    if matched_pairs.empty:
        return pd.DataFrame()

    _ensure_ipeds_view(con, ipeds_path)
    cip_map = _load_ipeds_cip_map(ipeds_path)
    broad_membership = build_broad_bin_membership(cip_map.keys())
    broad_any_cips = _broad_membership_rows(broad_membership, side="all")
    control_cips = _pair_control_cip_rows(matched_pairs, broad_membership)
    data_min_year, data_max_year = _ipeds_year_bounds(con)
    year_min = max(int(ANALYSIS_ORIGINAL_YEAR_MIN), int(data_min_year))
    year_max = min(int(ANALYSIS_IPEDS_YEAR_MAX), int(data_max_year))
    if year_min > year_max:
        return pd.DataFrame()

    con.register("generalized_ipeds_did_pairs_py", matched_pairs)
    con.register("generalized_ipeds_broad_any_cips_py", broad_any_cips)
    con.register("generalized_ipeds_control_cips_py", control_cips)
    did_panel = con.sql(
        f"""
        WITH pair_roles AS (
            SELECT
                CAST(pair_id AS BIGINT) AS pair_id,
                CAST(relabel_year AS INTEGER) AS relabel_year,
                CAST(relabel_type AS VARCHAR) AS relabel_type,
                CAST(broad_pair_bin AS VARCHAR) AS broad_pair_bin,
                CAST(degree_type AS VARCHAR) AS degree_type,
                CAST(awlevel AS INTEGER) AS awlevel,
                CAST(treated_unitid AS BIGINT) AS unitid,
                1 AS treated
            FROM generalized_ipeds_did_pairs_py
            UNION ALL
            SELECT
                CAST(pair_id AS BIGINT) AS pair_id,
                CAST(relabel_year AS INTEGER) AS relabel_year,
                CAST(relabel_type AS VARCHAR) AS relabel_type,
                CAST(broad_pair_bin AS VARCHAR) AS broad_pair_bin,
                CAST(degree_type AS VARCHAR) AS degree_type,
                CAST(awlevel AS INTEGER) AS awlevel,
                CAST(control_unitid AS BIGINT) AS unitid,
                0 AS treated
            FROM generalized_ipeds_did_pairs_py
        ),
        pair_role_cips AS (
            SELECT
                p.pair_id,
                1 AS treated,
                m.cip6
            FROM generalized_ipeds_did_pairs_py p
            JOIN generalized_ipeds_broad_any_cips_py m
              ON m.broad_pair_bin = p.broad_pair_bin
            UNION ALL
            SELECT
                pair_id,
                0 AS treated,
                cip6
            FROM generalized_ipeds_control_cips_py
        ),
        year_grid AS (
            SELECT
                p.*,
                CAST(y.year AS INTEGER) AS calendar_year
            FROM pair_roles p
            CROSS JOIN generate_series({int(year_min)}, {int(year_max)}) AS y(year)
        )
        SELECT
            g.pair_id,
            g.unitid,
            g.calendar_year,
            g.relabel_year,
            g.relabel_type,
            g.broad_pair_bin,
            g.degree_type,
            CAST(1.0 AS DOUBLE) AS total_grads,
            COALESCE(SUM(CAST(i.ctotalt AS DOUBLE)), 0.0) AS ctotalt,
            COALESCE(SUM(CAST(i.cnralt AS DOUBLE)), 0.0) AS cnralt,
            g.treated
        FROM year_grid g
        JOIN pair_role_cips m
          ON m.pair_id = g.pair_id
         AND m.treated = g.treated
        LEFT JOIN ipeds_raw i
          ON CAST(i.unitid AS BIGINT) = g.unitid
         AND CAST(i.awlevel AS INTEGER) = g.awlevel
         AND CAST(i.year AS INTEGER) = g.calendar_year
         AND LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = m.cip6
        GROUP BY
            g.pair_id,
            g.unitid,
            g.calendar_year,
            g.relabel_year,
            g.relabel_type,
            g.broad_pair_bin,
            g.degree_type,
            g.treated
        """
    ).df()
    if did_panel.empty:
        return did_panel
    did_panel = _apply_did_design_columns(did_panel, did_spec=normalized_did_spec)
    did_panel["event_t"] = did_panel["calendar_year"] - did_panel["relabel_year"]
    did_panel["cnralt_share_of_ctotalt"] = _safe_share(did_panel["cnralt"], did_panel["ctotalt"])
    did_panel["panel_level"] = "ipeds_program"
    return did_panel


def _save_figure(fig: plt.Figure, path: Path) -> None:
    llstyle.savefig(fig, path, dpi=300)


def _normalize_did_spec(value: str | None) -> str:
    normalized = str(value or DEFAULT_DID_SPEC).strip().lower()
    if normalized in VALID_DID_SPECS:
        return normalized
    raise ValueError(f"Unknown did_spec '{value}'; expected one of {', '.join(VALID_DID_SPECS)}")


def _normalize_estimator(value: str | None) -> str:
    normalized = str(value or DEFAULT_ESTIMATOR).strip().lower()
    if normalized in VALID_ESTIMATORS:
        return normalized
    raise ValueError(f"Unknown estimator '{value}'; expected one of {', '.join(VALID_ESTIMATORS)}")


def _normalize_control_group(value: str | None) -> str:
    normalized = str(value or DEFAULT_CONTROL_GROUP).strip().lower().replace(" ", "_")
    normalized = CONTROL_GROUP_ALIASES.get(normalized, normalized)
    if normalized in VALID_CONTROL_GROUPS:
        return normalized
    raise ValueError(f"Unknown control_group '{value}'; expected one of {', '.join(VALID_CONTROL_GROUPS)}")


def _analysis_year_max_for_yvar(yvar: str) -> int:
    if yvar in PROGRAM_LEVEL_IPEDS_YVARS:
        return int(ANALYSIS_IPEDS_YEAR_MAX)
    return int(ANALYSIS_FOIA_YEAR_MAX)


def _did_spec_from_panel(did_panel: pd.DataFrame, did_spec: str | None = None) -> str:
    if did_spec is not None:
        return _normalize_did_spec(did_spec)
    if "did_spec" in did_panel.columns:
        values = did_panel["did_spec"].dropna().astype(str).unique().tolist()
        if values:
            return _normalize_did_spec(values[0])
    return DEFAULT_DID_SPEC


def _build_did_fe_group(df: pd.DataFrame) -> pd.Series:
    unitid = pd.to_numeric(df.get("unitid"), errors="coerce").astype("Int64").astype(str)
    if not {"broad_pair_bin", "degree_type"}.issubset(df.columns):
        return unitid
    broad_pair_bin = df["broad_pair_bin"].fillna("missing").astype(str)
    degree_type = df["degree_type"].fillna("missing").astype(str)
    return unitid + "||" + broad_pair_bin + "||" + degree_type


def _apply_did_design_columns(df: pd.DataFrame, *, did_spec: str) -> pd.DataFrame:
    out = df.copy()
    normalized_spec = _normalize_did_spec(did_spec)
    out["did_spec"] = normalized_spec
    if normalized_spec == DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE:
        out["did_fe_group"] = _build_did_fe_group(out)
    return out


def _did_fe_var(reg_df: pd.DataFrame, *, did_spec: str) -> str:
    normalized_spec = _normalize_did_spec(did_spec)
    if normalized_spec == DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE and "did_fe_group" in reg_df.columns:
        if reg_df["did_fe_group"].dropna().nunique() >= 2:
            return "did_fe_group"
    return "unitid"


def _did_fe_formula_term(reg_df: pd.DataFrame, *, did_spec: str) -> str:
    return f"C({_did_fe_var(reg_df, did_spec=did_spec)})"


def _did_backend(did_spec: str) -> str:
    normalized_spec = _normalize_did_spec(did_spec)
    if normalized_spec != DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE:
        return "statsmodels"
    try:
        import pyfixest  # noqa: F401
    except Exception:
        return "statsmodels"
    return "pyfixest"


def _safe_statsmodels_bse(result: Any) -> pd.Series:
    params = result.params
    index = params.index if isinstance(params, pd.Series) else pd.Index(range(len(params)))
    try:
        cov = result.cov_params()
        diag = np.diag(np.asarray(cov, dtype=float))
        bse = np.full(len(index), np.nan, dtype=float)
        usable = diag >= 0
        bse[usable] = np.sqrt(diag[usable])
        return pd.Series(bse, index=index, dtype=float)
    except Exception:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in sqrt",
                category=RuntimeWarning,
            )
            raw_bse = result.bse
        if isinstance(raw_bse, pd.Series):
            return raw_bse.reindex(index)
        return pd.Series(raw_bse, index=index, dtype=float)


def _fit_pyfixest_ols(
    formula: str,
    *,
    data: pd.DataFrame,
    cluster_var: str,
    use_weights: bool,
) -> Any | None:
    try:
        from pyfixest.estimation import feols
    except Exception:
        return None

    kwargs: dict[str, Any] = {
        "fml": formula,
        "data": data,
        "vcov": {"CRV1": cluster_var},
        "copy_data": False,
        "store_data": False,
    }
    if use_weights:
        kwargs["weights"] = "total_grads"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"pyfixest\.estimation\.feols_",
        )
        return feols(**kwargs)


def _fit_event_study_pyfixest(
    reg_df: pd.DataFrame,
    *,
    yvar: str,
    did_spec: str,
    reference_event_time: int,
    use_weights: bool,
) -> Any | None:
    fe_var = _did_fe_var(reg_df, did_spec=did_spec)
    formula = (
        f"{yvar} ~ "
        f"i(event_t, ref={reference_event_time}) + treated + "
        f"i(event_t, treated, ref={reference_event_time}) | "
        f"{fe_var} + grad_year"
    )
    return _fit_pyfixest_ols(
        formula,
        data=reg_df,
        cluster_var="unitid",
        use_weights=use_weights,
    )


def _fit_post_summary_pyfixest(
    reg_df: pd.DataFrame,
    *,
    yvar: str,
    did_spec: str,
    use_weights: bool,
) -> Any | None:
    fe_var = _did_fe_var(reg_df, did_spec=did_spec)
    formula = f"{yvar} ~ treated * post | {fe_var} + grad_year"
    return _fit_pyfixest_ols(
        formula,
        data=reg_df,
        cluster_var="unitid",
        use_weights=use_weights,
    )


def _fit_calendar_year_pyfixest(
    cohort_df: pd.DataFrame,
    *,
    yvar: str,
    did_spec: str,
    reference_year: int,
    use_weights: bool,
) -> Any | None:
    fe_var = _did_fe_var(cohort_df, did_spec=did_spec)
    formula = f"{yvar} ~ treated + i(calendar_year, treated, ref={reference_year}) | {fe_var}"
    return _fit_pyfixest_ols(
        formula,
        data=cohort_df,
        cluster_var="unitid",
        use_weights=use_weights,
    )


def _result_params_and_bse(result: Any, *, backend: str) -> tuple[pd.Series, pd.Series]:
    if backend == "pyfixest":
        return result.coef(), result.se()
    return result.params, _safe_statsmodels_bse(result)


def _prepare_did_regression_df(
    did_panel: pd.DataFrame,
    *,
    yvar: str,
    did_spec: str | None = None,
    event_time_min: int = PLOT_EVENT_MIN,
    event_time_max: int = PLOT_EVENT_MAX,
) -> pd.DataFrame:
    if did_panel.empty or yvar not in did_panel.columns:
        return pd.DataFrame()

    normalized_did_spec = _did_spec_from_panel(did_panel, did_spec=did_spec)
    reg_df = did_panel.copy()
    reg_df = reg_df[
        reg_df["calendar_year"].between(base.PLOT_YEAR_MIN, _analysis_year_max_for_yvar(yvar))
    ].copy()
    reg_df = reg_df[reg_df["event_t"].between(event_time_min, event_time_max)].copy()
    reg_df = reg_df.dropna(subset=[yvar, "event_t", "treated", "unitid", "calendar_year", "total_grads"]).copy()
    if reg_df.empty:
        return pd.DataFrame()

    reg_df["event_t"] = pd.to_numeric(reg_df["event_t"], errors="coerce").astype(int)
    reg_df["treated"] = pd.to_numeric(reg_df["treated"], errors="coerce").astype(int)
    reg_df["unitid"] = pd.to_numeric(reg_df["unitid"], errors="coerce").astype(int)
    reg_df["grad_year"] = pd.to_numeric(reg_df["calendar_year"], errors="coerce").astype(int)
    reg_df["total_grads"] = pd.to_numeric(reg_df["total_grads"], errors="coerce").fillna(0.0)
    reg_df = reg_df[reg_df["total_grads"] > 0].copy()
    if reg_df.empty:
        return pd.DataFrame()

    treated_event_values = set(
        reg_df.loc[reg_df["treated"] == 1, "event_t"].dropna().astype(int).unique().tolist()
    )
    control_event_values = set(
        reg_df.loc[reg_df["treated"] == 0, "event_t"].dropna().astype(int).unique().tolist()
    )
    common_event_values = sorted(treated_event_values & control_event_values)
    if not common_event_values:
        return pd.DataFrame()
    reg_df = reg_df[reg_df["event_t"].isin(common_event_values)].copy()
    reg_df = _apply_did_design_columns(reg_df, did_spec=normalized_did_spec)
    return reg_df


def _find_did_interaction_param(
    params: pd.Series,
    event_t: int,
    reference_event_time: int,
) -> str | None:
    term = f"C(event_t, Treatment(reference={reference_event_time}))"
    candidates = [
        f"{term}[T.{event_t}]:treated",
        f"{term}[{event_t}]:treated",
        f"treated:{term}[T.{event_t}]",
        f"treated:{term}[{event_t}]",
    ]
    for candidate in candidates:
        if candidate in params.index:
            return candidate

    normalized_event_t = int(event_t)
    event_token_variants = {
        str(normalized_event_t),
        f"{normalized_event_t}",
        f"{float(event_t)}",
        f"{event_t:.1f}",
        f"{normalized_event_t:.0f}",
        f"{event_t:.0f}",
    }

    for token in event_token_variants:
        prefix = f"[T.{token}]"
        bare = f"[{token}]"
        for name in map(str, params.index):
            if "treated" not in str(name):
                continue
            if term in str(name) and (prefix in str(name) or bare in str(name)):
                return str(name)

    for name in map(str, params.index):
        if "treated" not in name:
            continue
        if f"{normalized_event_t}" not in name and f"{float(event_t)}" not in name and f"{event_t:.1f}" not in name:
            continue
        if f"[T.{event_t}]" in name or f"[{event_t}]" in name:
            return name
        if f"[T.{float(event_t)}]" in name or f"[{float(event_t)}]" in name:
            return name
    return None


def _find_event_time_param(
    params: pd.Series,
    event_t: int,
    reference_event_time: int,
) -> str | None:
    term = f"C(event_t, Treatment(reference={reference_event_time}))"
    candidates = [
        f"{term}[T.{event_t}]",
        f"{term}[{event_t}]",
    ]
    for candidate in candidates:
        if candidate in params.index:
            return candidate

    normalized_event_t = int(event_t)
    event_token_variants = {
        str(normalized_event_t),
        f"{float(event_t)}",
        f"{event_t:.1f}",
        f"{normalized_event_t:.0f}",
        f"{event_t:.0f}",
    }
    for name in map(str, params.index):
        if term not in name:
            continue
        if any(f"[T.{token}]" in name or f"[{token}]" in name for token in event_token_variants):
            return name
    for name in map(str, params.index):
        if "treated" in name or "event_t" not in name:
            continue
        if any(token in name for token in event_token_variants):
            return name
    return None


def _did_event_study_formula(
    yvar: str,
    *,
    reference_event_time: int,
    reg_df: pd.DataFrame,
    did_spec: str,
) -> str:
    fe_term = _did_fe_formula_term(reg_df, did_spec=did_spec)
    return (
        f"{yvar} ~ "
        f"C(event_t, Treatment(reference={reference_event_time}))*treated "
        f"+ {fe_term} + C(grad_year)"
    )


def _stacked_event_study_formula(
    yvar: str,
    *,
    reference_event_time: int,
    reg_df: pd.DataFrame,
    did_spec: str,
) -> str:
    fe_term = _did_fe_formula_term(reg_df, did_spec=did_spec)
    return (
        f"{yvar} ~ "
        f"C(event_t, Treatment(reference={reference_event_time})) "
        f"+ {fe_term} + C(grad_year)"
    )


def _prepare_stacked_regression_df(
    did_panel: pd.DataFrame,
    *,
    yvar: str,
    did_spec: str | None = None,
    event_time_min: int = PLOT_EVENT_MIN,
    event_time_max: int = PLOT_EVENT_MAX,
) -> pd.DataFrame:
    if did_panel.empty or yvar not in did_panel.columns:
        return pd.DataFrame()

    normalized_did_spec = _did_spec_from_panel(did_panel, did_spec=did_spec)
    reg_df = did_panel.copy()
    reg_df = reg_df[pd.to_numeric(reg_df["treated"], errors="coerce").eq(1)].copy()
    reg_df = reg_df[
        reg_df["calendar_year"].between(base.PLOT_YEAR_MIN, _analysis_year_max_for_yvar(yvar))
    ].copy()
    reg_df = reg_df[reg_df["event_t"].between(event_time_min, event_time_max)].copy()
    reg_df = reg_df.dropna(subset=[yvar, "event_t", "unitid", "calendar_year", "total_grads"]).copy()
    if reg_df.empty:
        return pd.DataFrame()

    reg_df["event_t"] = pd.to_numeric(reg_df["event_t"], errors="coerce").astype(int)
    reg_df["unitid"] = pd.to_numeric(reg_df["unitid"], errors="coerce").astype(int)
    reg_df["grad_year"] = pd.to_numeric(reg_df["calendar_year"], errors="coerce").astype(int)
    reg_df["total_grads"] = pd.to_numeric(reg_df["total_grads"], errors="coerce").fillna(0.0)
    reg_df = reg_df[reg_df["total_grads"] > 0].copy()
    if reg_df.empty:
        return pd.DataFrame()
    reg_df = _apply_did_design_columns(reg_df, did_spec=normalized_did_spec)
    return reg_df


def compute_did_event_study_generalized(
    did_panel: pd.DataFrame,
    *,
    yvar: str,
    did_spec: str | None = None,
    reference_event_time: int = DI_D_REFERENCE_EVENT_TIME,
    event_time_min: int = PLOT_EVENT_MIN,
    event_time_max: int = PLOT_EVENT_MAX,
    use_weights: bool = False,
) -> pd.DataFrame:
    reg_df = _prepare_did_regression_df(
        did_panel,
        yvar=yvar,
        did_spec=did_spec,
        event_time_min=event_time_min,
        event_time_max=event_time_max,
    )
    if reg_df.empty:
        return pd.DataFrame()

    event_values = sorted(pd.to_numeric(reg_df["event_t"], errors="coerce").dropna().astype(int).unique().tolist())
    n_schools = reg_df["unitid"].nunique()
    if reference_event_time not in event_values or len(event_values) < 2 or n_schools < 2:
        return pd.DataFrame()

    normalized_did_spec = _did_spec_from_panel(did_panel, did_spec=did_spec)
    backend = _did_backend(normalized_did_spec)
    result = None
    params = pd.Series(dtype=float)
    bse = pd.Series(dtype=float)
    if backend == "pyfixest":
        try:
            result = _fit_event_study_pyfixest(
                reg_df,
                yvar=yvar,
                did_spec=normalized_did_spec,
                reference_event_time=reference_event_time,
                use_weights=use_weights,
            )
            if result is not None:
                params, bse = _result_params_and_bse(result, backend=backend)
        except Exception as exc:
            _progress(f"pyfixest DiD regression failed for {yvar}; falling back to statsmodels ({exc})")
            backend = "statsmodels"

    if backend == "statsmodels":
        try:
            import statsmodels.formula.api as smf
        except Exception as exc:
            _progress(f"statsmodels unavailable; skipping DiD regression for {yvar} ({exc})")
            return pd.DataFrame()
        formula = _did_event_study_formula(
            yvar,
            reference_event_time=reference_event_time,
            reg_df=reg_df,
            did_spec=normalized_did_spec,
        )
        try:
            model = smf.wls(formula, data=reg_df, weights=reg_df["total_grads"]) if use_weights else smf.ols(formula, data=reg_df)
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": reg_df["unitid"]},
            )
        except Exception as exc:
            _progress(f"Clustered DiD regression failed for {yvar}; falling back to HC1 ({exc})")
            model = smf.wls(formula, data=reg_df, weights=reg_df["total_grads"]) if use_weights else smf.ols(formula, data=reg_df)
            result = model.fit(cov_type="HC1")
        params, bse = _result_params_and_bse(result, backend=backend)

    nobs = int(getattr(result, "nobs", len(reg_df)) or len(reg_df))

    rows: list[dict[str, object]] = []
    event_counts = (
        reg_df.groupby(["event_t", "treated"], as_index=False)
        .agg(
            n_school_years=("unitid", "size"),
            n_schools=("unitid", "nunique"),
            total_grads=("total_grads", "sum"),
        )
    )
    treated_counts = (
        event_counts[event_counts["treated"] == 1]
        .drop(columns=["treated"])
        .rename(
            columns={
                "n_school_years": "treated_n_school_years",
                "n_schools": "treated_n_schools",
                "total_grads": "treated_total_grads",
            }
        )
    )
    control_counts = (
        event_counts[event_counts["treated"] == 0]
        .drop(columns=["treated"])
        .rename(
            columns={
                "n_school_years": "control_n_school_years",
                "n_schools": "control_n_schools",
                "total_grads": "control_total_grads",
            }
        )
    )
    count_lookup = treated_counts.merge(control_counts, on="event_t", how="outer").set_index("event_t").to_dict("index")

    for event_t in event_values:
        event_counts_row = count_lookup.get(int(event_t), {})
        if event_t == reference_event_time:
            coef = 0.0
            se = 0.0
        else:
            param = _find_did_interaction_param(
                params,
                event_t=int(event_t),
                reference_event_time=reference_event_time,
            )
            coef = float(params.get(param, float("nan"))) if param is not None else float("nan")
            se = float(bse.get(param, float("nan"))) if param is not None else float("nan")
        rows.append(
            {
                "event_t": int(event_t),
                "estimator": ESTIMATOR_DID,
                "coef": coef,
                "se": se,
                "ci_low": coef - 1.96 * se,
                "ci_high": coef + 1.96 * se,
                "reference_event_t": reference_event_time,
                "nobs": nobs,
                "n_schools_total": int(n_schools),
                "treated_n_school_years": int(event_counts_row.get("treated_n_school_years", 0) or 0),
                "control_n_school_years": int(event_counts_row.get("control_n_school_years", 0) or 0),
                "treated_n_schools": int(event_counts_row.get("treated_n_schools", 0) or 0),
                "control_n_schools": int(event_counts_row.get("control_n_schools", 0) or 0),
                "treated_total_grads": float(event_counts_row.get("treated_total_grads", 0.0) or 0.0),
                "control_total_grads": float(event_counts_row.get("control_total_grads", 0.0) or 0.0),
            }
        )

    return pd.DataFrame(rows)


def compute_stacked_event_study_generalized(
    did_panel: pd.DataFrame,
    *,
    yvar: str,
    did_spec: str | None = None,
    reference_event_time: int = DI_D_REFERENCE_EVENT_TIME,
    event_time_min: int = PLOT_EVENT_MIN,
    event_time_max: int = PLOT_EVENT_MAX,
    use_weights: bool = False,
) -> pd.DataFrame:
    reg_df = _prepare_stacked_regression_df(
        did_panel,
        yvar=yvar,
        did_spec=did_spec,
        event_time_min=event_time_min,
        event_time_max=event_time_max,
    )
    if reg_df.empty:
        return pd.DataFrame()

    event_values = sorted(pd.to_numeric(reg_df["event_t"], errors="coerce").dropna().astype(int).unique().tolist())
    n_schools = reg_df["unitid"].nunique()
    if reference_event_time not in event_values or len(event_values) < 2 or n_schools < 2:
        return pd.DataFrame()

    normalized_did_spec = _did_spec_from_panel(did_panel, did_spec=did_spec)
    backend = _did_backend(normalized_did_spec)
    result = None
    params = pd.Series(dtype=float)
    bse = pd.Series(dtype=float)
    if backend == "pyfixest":
        try:
            fe_var = _did_fe_var(reg_df, did_spec=normalized_did_spec)
            result = _fit_pyfixest_ols(
                (
                    f"{yvar} ~ "
                    f"i(event_t, ref={reference_event_time}) | "
                    f"{fe_var} + grad_year"
                ),
                data=reg_df,
                cluster_var="unitid",
                use_weights=use_weights,
            )
            if result is not None:
                params, bse = _result_params_and_bse(result, backend=backend)
        except Exception as exc:
            _progress(f"pyfixest stacked event-study failed for {yvar}; falling back to statsmodels ({exc})")
            backend = "statsmodels"

    if backend == "statsmodels":
        try:
            import statsmodels.formula.api as smf
        except Exception as exc:
            _progress(f"statsmodels unavailable; skipping stacked event-study for {yvar} ({exc})")
            return pd.DataFrame()
        formula = _stacked_event_study_formula(
            yvar,
            reference_event_time=reference_event_time,
            reg_df=reg_df,
            did_spec=normalized_did_spec,
        )
        try:
            model = smf.wls(formula, data=reg_df, weights=reg_df["total_grads"]) if use_weights else smf.ols(formula, data=reg_df)
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": reg_df["unitid"]},
            )
        except Exception as exc:
            _progress(f"Clustered stacked event-study failed for {yvar}; falling back to HC1 ({exc})")
            model = smf.wls(formula, data=reg_df, weights=reg_df["total_grads"]) if use_weights else smf.ols(formula, data=reg_df)
            result = model.fit(cov_type="HC1")
        params, bse = _result_params_and_bse(result, backend=backend)

    nobs = int(getattr(result, "nobs", len(reg_df)) or len(reg_df))
    event_counts = (
        reg_df.groupby("event_t", as_index=False)
        .agg(
            n_school_years=("unitid", "size"),
            n_schools=("unitid", "nunique"),
            total_grads=("total_grads", "sum"),
        )
        .set_index("event_t")
        .to_dict("index")
    )

    rows: list[dict[str, object]] = []
    for event_t in event_values:
        event_counts_row = event_counts.get(int(event_t), {})
        if event_t == reference_event_time:
            coef = 0.0
            se = 0.0
        else:
            param = _find_event_time_param(
                params,
                event_t=int(event_t),
                reference_event_time=reference_event_time,
            )
            coef = float(params.get(param, float("nan"))) if param is not None else float("nan")
            se = float(bse.get(param, float("nan"))) if param is not None else float("nan")
        rows.append(
            {
                "event_t": int(event_t),
                "estimator": ESTIMATOR_STACKED_TREATED,
                "coef": coef,
                "se": se,
                "ci_low": coef - 1.96 * se,
                "ci_high": coef + 1.96 * se,
                "reference_event_t": reference_event_time,
                "nobs": nobs,
                "n_schools_total": int(n_schools),
                "treated_n_school_years": int(event_counts_row.get("n_school_years", 0) or 0),
                "control_n_school_years": 0,
                "treated_n_schools": int(event_counts_row.get("n_schools", 0) or 0),
                "control_n_schools": 0,
                "treated_total_grads": float(event_counts_row.get("total_grads", 0.0) or 0.0),
                "control_total_grads": 0.0,
            }
        )

    return pd.DataFrame(rows)


def _format_percentage_point_tick(value: float, _pos: object) -> str:
    scaled = 100.0 * float(value)
    if abs(scaled - round(scaled)) < 1e-9:
        return f"{scaled:.0f}"
    return f"{scaled:.1f}"


def _outcome_ylabel(yvar: str) -> str:
    label = GENERALIZED_YVAR_LABELS.get(yvar, base.yvar_label(yvar))
    if yvar in PERCENTAGE_POINT_YVARS:
        return f"{label} (pp)"
    return label


def _format_yaxis_for_outcome(ax: plt.Axes, yvar: str) -> None:
    if yvar in PERCENTAGE_POINT_YVARS:
        ax.yaxis.set_major_formatter(FuncFormatter(_format_percentage_point_tick))


def _format_outcome_value(yvar: str, value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if yvar in PERCENTAGE_POINT_YVARS:
        return f"{100.0 * value:.1f} pp"
    if yvar in {"opt_years_avg", "post_grad_authorization_years_avg", "opt_duration_years_avg"}:
        return f"{value:.2f}"
    if yvar in {"unique_employers", "unique_opt_cities"}:
        return f"{value:.2f}"
    if yvar == "internship_count":
        return f"{value:.1f}"
    if yvar in {"auth_employment_tenure_years", "internship_opt_years"}:
        return f"{value:.2f}"
    if yvar == "employer_opt_intensity_pctile":
        return f"{value:.1f}"
    if yvar in {"ctotalt", "cnralt"}:
        return f"{value:.1f}"
    if "tuition" in yvar or "fees" in yvar or "funds" in yvar:
        return f"${value:,.0f}"
    return f"{value:.3f}"


def _format_se_value(yvar: str, value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if yvar in PERCENTAGE_POINT_YVARS:
        return f"{100.0 * value:.1f} pp"
    if yvar in {"opt_years_avg", "post_grad_authorization_years_avg", "opt_duration_years_avg"}:
        return f"{value:.2f}"
    if yvar in {"unique_employers", "unique_opt_cities"}:
        return f"{value:.2f}"
    if yvar == "internship_count":
        return f"{value:.1f}"
    if yvar in {"auth_employment_tenure_years", "internship_opt_years"}:
        return f"{value:.2f}"
    if yvar == "employer_opt_intensity_pctile":
        return f"{value:.1f}"
    if yvar in {"ctotalt", "cnralt"}:
        return f"{value:.1f}"
    if "tuition" in yvar or "fees" in yvar or "funds" in yvar:
        return f"${value:,.0f}"
    return f"{value:.3f}"


def _find_post_interaction_param(params: pd.Series) -> str | None:
    for name in ("treated:post", "post:treated"):
        if name in params.index:
            return name
    return None


def _find_calendar_year_interaction_param(
    params: pd.Series,
    *,
    calendar_year: int,
    reference_year: int,
) -> str | None:
    term = f"C(calendar_year, Treatment(reference={reference_year}))"
    candidates = (
        f"{term}[T.{calendar_year}]:treated",
        f"{term}[{calendar_year}]:treated",
        f"treated:{term}[T.{calendar_year}]",
        f"treated:{term}[{calendar_year}]",
    )
    for candidate in candidates:
        if candidate in params.index:
            return candidate
    calendar_tokens = {str(int(calendar_year)), f"{float(calendar_year)}", f"{calendar_year:.1f}"}
    for name in params.index:
        if "treated" not in name or "calendar_year" not in name:
            continue
        if not any(token in name for token in calendar_tokens):
            continue
        return str(name)
    return None


def compute_post_did_summary(
    did_panel: pd.DataFrame,
    *,
    yvar: str,
    post_start_event_time: int = -1,
    did_spec: str | None = None,
    use_weights: bool = False,
) -> dict[str, float] | None:
    reg_df = _prepare_did_regression_df(did_panel, yvar=yvar, did_spec=did_spec)
    if reg_df.empty or reg_df["unitid"].nunique() < 2:
        return None

    baseline_df = reg_df[reg_df["event_t"] <= DI_D_REFERENCE_EVENT_TIME].copy()
    baseline_mean = float(baseline_df[yvar].mean()) if not baseline_df.empty else float("nan")
    reg_df["post"] = (reg_df["event_t"] >= int(post_start_event_time)).astype(int)
    if reg_df["post"].nunique() < 2 or reg_df["treated"].nunique() < 2:
        return None

    normalized_did_spec = _did_spec_from_panel(did_panel, did_spec=did_spec)
    backend = _did_backend(normalized_did_spec)
    result = None
    params = pd.Series(dtype=float)
    bse = pd.Series(dtype=float)
    if backend == "pyfixest":
        try:
            result = _fit_post_summary_pyfixest(
                reg_df,
                yvar=yvar,
                did_spec=normalized_did_spec,
                use_weights=use_weights,
            )
            if result is not None:
                params, bse = _result_params_and_bse(result, backend=backend)
        except Exception as exc:
            _progress(f"pyfixest post DiD regression failed for {yvar}; falling back to statsmodels ({exc})")
            backend = "statsmodels"

    if backend == "statsmodels":
        try:
            import statsmodels.formula.api as smf
        except Exception as exc:
            _progress(f"statsmodels unavailable; skipping pooled post summary for {yvar} ({exc})")
            return {
                "baseline_mean": baseline_mean,
                "coef": float("nan"),
                "se": float("nan"),
                "effect_size_pct": float("nan"),
            }
        fe_term = _did_fe_formula_term(reg_df, did_spec=normalized_did_spec)
        formula = f"{yvar} ~ treated*post + {fe_term} + C(grad_year)"
        try:
            model = smf.wls(formula, data=reg_df, weights=reg_df["total_grads"]) if use_weights else smf.ols(formula, data=reg_df)
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": reg_df["unitid"]},
            )
        except Exception as exc:
            _progress(f"Clustered post DiD regression failed for {yvar}; falling back to HC1 ({exc})")
            model = smf.wls(formula, data=reg_df, weights=reg_df["total_grads"]) if use_weights else smf.ols(formula, data=reg_df)
            result = model.fit(cov_type="HC1")
        params, bse = _result_params_and_bse(result, backend=backend)

    param = _find_post_interaction_param(params)
    coef = float(params.get(param, float("nan"))) if param is not None else float("nan")
    se = float(bse.get(param, float("nan"))) if param is not None else float("nan")
    effect_size_pct = float("nan")
    if pd.notna(baseline_mean) and abs(float(baseline_mean)) > 1e-12:
        effect_size_pct = 100.0 * float(coef) / float(baseline_mean)
    return {
        "baseline_mean": baseline_mean,
        "coef": coef,
        "se": se,
        "effect_size_pct": effect_size_pct,
    }


def build_did_summary_text(
    did_panel: pd.DataFrame,
    *,
    yvar: str,
    post_start_event_time: int = -1,
    did_spec: str | None = None,
    use_weights: bool = False,
) -> str | None:
    summary = compute_post_did_summary(
        did_panel,
        yvar=yvar,
        post_start_event_time=post_start_event_time,
        did_spec=did_spec,
        use_weights=use_weights,
    )
    if summary is None:
        return None
    baseline_text = _format_outcome_value(yvar, summary.get("baseline_mean"))
    coef_text = _format_outcome_value(yvar, summary.get("coef"))
    se_text = _format_se_value(yvar, summary.get("se"))
    effect_size = summary.get("effect_size_pct")
    effect_text = "NA" if effect_size is None or pd.isna(effect_size) else f"{float(effect_size):.1f}%"
    return (
        f"Baseline mean (t <= -2): {baseline_text}\n"
        f"Treat x Post (t >= -1): {coef_text} ({se_text})\n"
        f"Effect size: {effect_text}"
    )


def _add_did_summary_text(ax: plt.Axes, summary_text: str | None, *, fontsize: float | None = None) -> None:
    if summary_text:
        ax.text(
            0.02,
            0.98,
            summary_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize if fontsize is not None else max(DI_D_PLOT_FONT_SIZE - 8, 9),
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#d0d0d0",
                "alpha": 0.92,
            },
            zorder=5,
        )
    ax.figure.tight_layout()


def _multi_series_offset_step(
    n_series: int,
    *,
    max_step: float = MULTI_SERIES_MAX_STEP,
    target_total_span: float = MULTI_SERIES_TARGET_TOTAL_SPAN,
) -> float:
    return llstyle.offset_step(n_series, max_step=max_step, target_total_span=target_total_span)


def build_broad_bin_year_counts(
    events: pd.DataFrame,
    *,
    degree_type: str | None = None,
) -> pd.DataFrame:
    columns = ["relabel_year", "broad_pair_bin", "event_count", "broad_pair_label"]
    if events.empty:
        return pd.DataFrame(columns=columns)

    df = events.copy()
    if degree_type is not None:
        df = df[df["degree_type"] == degree_type]
    if "event_flag" in df.columns:
        df = df[df["event_flag"].eq(1)]
    if "event_origin_category" in df.columns:
        df = df[df["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"])]
    if "broad_bin_eligible" in df.columns:
        df = df[df["broad_bin_eligible"].eq(1)]
    df = df[df["relabel_year"].notna() & df["broad_pair_bin"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    dedupe_cols = [
        column
        for column in ["unitid", "awlevel", "degree_type", "broad_pair_bin", "relabel_year"]
        if column in df.columns
    ]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols)

    df["relabel_year"] = pd.to_numeric(df["relabel_year"], errors="coerce").astype("Int64")
    df = df[df["relabel_year"].notna() & df["relabel_year"].le(MAX_RELABEL_YEAR)].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    bin_order = list(BROAD_BIN_SPECS.keys())
    year_range = range(int(df["relabel_year"].min()), int(df["relabel_year"].max()) + 1)
    grid = pd.MultiIndex.from_product([year_range, bin_order], names=["relabel_year", "broad_pair_bin"]).to_frame(index=False)
    counts = (
        df.groupby(["relabel_year", "broad_pair_bin"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
    )
    out = grid.merge(counts, on=["relabel_year", "broad_pair_bin"], how="left")
    out["event_count"] = out["event_count"].fillna(0).astype(int)
    out["broad_pair_label"] = out["broad_pair_bin"].map(BROAD_BIN_PLOT_LABELS).fillna(out["broad_pair_bin"])
    return out.loc[:, columns]


def build_broad_bin_degree_year_counts(events: pd.DataFrame) -> pd.DataFrame:
    columns = ["relabel_year", "degree_type", "broad_pair_bin", "event_count", "broad_pair_label"]
    if events.empty:
        return pd.DataFrame(columns=columns)

    df = events.copy()
    df = df[
        (df["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"]))
        & df["event_flag"].eq(1)
        & df["broad_bin_eligible"].eq(1)
    ].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df = df[df["unitid"].notna() & df["relabel_year"].notna() & df["broad_pair_bin"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["relabel_year"] = pd.to_numeric(df["relabel_year"], errors="coerce").astype("Int64")
    df["degree_type"] = df["degree_type"].astype("string").str.strip()
    df = df[
        df["relabel_year"].notna()
        & df["relabel_year"].le(MAX_RELABEL_YEAR)
        & df["degree_type"].notna()
    ].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["broad_pair_bin"] = df["broad_pair_bin"].astype("string")
    dedupe_cols = [
        col
        for col in ["unitid", "awlevel", "degree_type", "broad_pair_bin", "relabel_year"]
        if col in df.columns
    ]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols)

    year_range = range(int(df["relabel_year"].min()), int(df["relabel_year"].max()) + 1)
    degree_order = [d for d in ["Bachelor", "Master", "Doctor"] if d in set(df["degree_type"].astype(str))]
    if not degree_order:
        degree_order = sorted(df["degree_type"].dropna().astype(str).unique())
    broad_pairs = [name for name in BROAD_BIN_SPECS.keys() if name in set(df["broad_pair_bin"].astype(str))]
    if not broad_pairs:
        broad_pairs = sorted(df["broad_pair_bin"].dropna().astype(str).unique())

    grid = pd.MultiIndex.from_product(
        [degree_order, broad_pairs, list(year_range)],
        names=["degree_type", "broad_pair_bin", "relabel_year"],
    ).to_frame(index=False)
    counts = (
        df.groupby(["degree_type", "broad_pair_bin", "relabel_year"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
    )
    out = grid.merge(counts, on=["degree_type", "broad_pair_bin", "relabel_year"], how="left")
    out["event_count"] = out["event_count"].fillna(0).astype(int)
    out["broad_pair_label"] = out["broad_pair_bin"].map(BROAD_BIN_PLOT_LABELS).fillna(out["broad_pair_bin"])
    return out.loc[:, columns]


def plot_relabel_broad_bin_degree_year_breakdown(
    events: pd.DataFrame,
    *,
    out_dir: str | Path,
) -> Path | None:
    counts = build_broad_bin_degree_year_counts(events)
    if counts.empty:
        return None

    sns.set(style="whitegrid")
    llstyle.apply_style()
    degree_order = [name for name in ["Bachelor", "Master", "Doctor"] if name in set(counts["degree_type"].astype(str))]
    if not degree_order:
        degree_order = sorted(counts["degree_type"].dropna().astype(str).unique())
    positive_bins = set(
        counts.loc[counts["event_count"].gt(0), "broad_pair_bin"].dropna().astype(str)
    )
    broad_pair_order = [name for name in BROAD_BIN_SPECS if name in positive_bins]
    if not broad_pair_order:
        broad_pair_order = sorted(counts["broad_pair_bin"].dropna().astype(str).unique())
    colors = [llstyle.color(idx) for idx in range(len(broad_pair_order))]
    fig, axes = plt.subplots(
        nrows=len(degree_order),
        ncols=1,
        figsize=(llstyle.FIGSIZE[0], max(2.0 * len(degree_order), 4.5)),
        sharex=True,
        sharey=True,
    )
    if len(degree_order) == 1:
        axes = [axes]
    width = min(0.82 / max(len(broad_pair_order), 1), 0.22)
    offsets = (np.arange(len(broad_pair_order)) - (len(broad_pair_order) - 1) / 2.0) * width
    for ax, degree_name in zip(axes, degree_order):
        degree_counts = counts[counts["degree_type"].astype(str) == degree_name]
        pivot = degree_counts.pivot_table(
            index="relabel_year",
            columns="broad_pair_bin",
            values="event_count",
            aggfunc="sum",
            fill_value=0,
        ).reindex(columns=broad_pair_order, fill_value=0).sort_index()
        x_values = np.arange(len(pivot), dtype=float)
        for idx, broad_pair_bin in enumerate(broad_pair_order):
            values = pivot[broad_pair_bin].astype(float).to_numpy()
            ax.bar(
                x_values + offsets[idx],
                values,
                width=width,
                color=colors[idx % len(colors)],
                label=BROAD_BIN_PLOT_LABELS.get(broad_pair_bin, broad_pair_bin),
                alpha=0.92,
            )
        ax.set_title("")
        ax.set_ylabel("Events")
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x_values)
        ax.set_xticklabels(pivot.index.astype(int).tolist())
    axes[-1].set_xlabel("Relabel year")
    axes[-1].tick_params(axis="x", rotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    llstyle.right_figure_legend(fig, handles, labels, title="Broad bin")

    out_path = Path(out_dir) / "relabel_broad_bin_degree_year_breakdown.png"
    _save_figure(fig, out_path)
    return out_path


def _school_display_name(unitid: int, lookup: dict[int, str] | None = None) -> str:
    if lookup is None:
        return str(int(unitid))
    name = lookup.get(int(unitid))
    if name:
        return _tex_escape(name)
    return str(int(unitid))


def _load_canonical_school_name_lookup(
    *,
    unitids: set[int] | None = None,
    crosswalk_path: str | Path = v2.CROSSWALK_PATH,
    main_institutions_path: str | Path = DEFAULT_IPEDS_MAIN_INSTITUTIONS_PATH,
    directory_path: str | Path = DEFAULT_IPEDS_DIRECTORY_2024_PATH,
) -> dict[int, str]:
    lookup: dict[int, str] = {}
    target_unitids = None if unitids is None else {int(unitid) for unitid in unitids}

    try:
        school_lookup = load_school_lookup(crosswalk_path)
        school_lookup = school_lookup.sort_values(["unitid", "alias_ind", "school_name"])
        school_lookup = school_lookup.drop_duplicates(subset=["unitid"], keep="first")
        if target_unitids is not None:
            school_lookup = school_lookup[school_lookup["unitid"].isin(target_unitids)].copy()
        lookup.update(
            {
                int(unitid): str(name).strip()
                for unitid, name in school_lookup[["unitid", "school_name"]].itertuples(index=False, name=None)
                if pd.notna(name) and str(name).strip()
            }
        )
    except Exception:
        pass

    missing_unitids = set() if target_unitids is None else target_unitids.difference(lookup)
    if target_unitids is None or missing_unitids:
        try:
            main_df = pd.read_parquet(main_institutions_path, columns=["main_unitid", "ipeds_name"])
            main_df = main_df.rename(columns={"main_unitid": "unitid", "ipeds_name": "school_name"})
            main_df["unitid"] = pd.to_numeric(main_df["unitid"], errors="coerce").astype("Int64")
            main_df = main_df.dropna(subset=["unitid", "school_name"]).copy()
            main_df["unitid"] = main_df["unitid"].astype(int)
            main_df["school_name"] = main_df["school_name"].astype(str).str.strip()
            main_df = main_df[main_df["school_name"].ne("")].copy()
            if missing_unitids:
                main_df = main_df[main_df["unitid"].isin(missing_unitids)].copy()
            lookup.update(
                {
                    int(unitid): str(name)
                    for unitid, name in main_df[["unitid", "school_name"]].drop_duplicates("unitid").itertuples(index=False, name=None)
                }
            )
        except Exception:
            pass

    missing_unitids = set() if target_unitids is None else target_unitids.difference(lookup)
    if target_unitids is None or missing_unitids:
        try:
            directory_df = pd.read_csv(directory_path, usecols=["UNITID", "INSTNM"], encoding="utf-8-sig")
            directory_df = directory_df.rename(columns={"UNITID": "unitid", "INSTNM": "school_name"})
            directory_df["unitid"] = pd.to_numeric(directory_df["unitid"], errors="coerce").astype("Int64")
            directory_df = directory_df.dropna(subset=["unitid", "school_name"]).copy()
            directory_df["unitid"] = directory_df["unitid"].astype(int)
            directory_df["school_name"] = directory_df["school_name"].astype(str).str.strip()
            directory_df = directory_df[directory_df["school_name"].ne("")].copy()
            if missing_unitids:
                directory_df = directory_df[directory_df["unitid"].isin(missing_unitids)].copy()
            lookup.update(
                {
                    int(unitid): str(name)
                    for unitid, name in directory_df[["unitid", "school_name"]].drop_duplicates("unitid").itertuples(index=False, name=None)
                }
            )
        except Exception:
            pass

    return lookup


def _sample_school_ids(values: pd.Series, sample_size: int, seed: int) -> list[int]:
    if values.empty:
        return []
    if len(values) <= sample_size:
        return list(pd.Series(values).astype(int).tolist())
    return (
        values.sample(n=sample_size, random_state=seed)
        .astype(int)
        .sort_values()
        .tolist()
    )


def write_broad_bin_treated_control_school_samples(
    con: ddb.DuckDBPyConnection,
    panel: pd.DataFrame,
    *,
    out_dir: str | Path,
    ipeds_path: str | Path = base.IPEDS_PATH,
    crosswalk_path: str | Path = v2.CROSSWALK_PATH,
    sample_size: int = RELABEL_BROAD_BIN_SAMPLE_N,
    seed: int = RELABEL_BROAD_BIN_SAMPLE_SEED,
) -> Path | None:
    matched_pairs = match_treated_to_never_treated(
        con,
        panel,
        ipeds_path=ipeds_path,
    )
    if matched_pairs.empty:
        return None

    for column in ["treated_unitid", "control_unitid"]:
        matched_pairs[column] = pd.to_numeric(matched_pairs[column], errors="coerce")
    matched_pairs = matched_pairs.dropna(subset=["treated_unitid", "control_unitid", "broad_pair_bin"]).copy()
    if matched_pairs.empty:
        return None
    matched_pairs["treated_unitid"] = matched_pairs["treated_unitid"].astype(int)
    matched_pairs["control_unitid"] = matched_pairs["control_unitid"].astype(int)
    matched_pairs["broad_pair_bin"] = matched_pairs["broad_pair_bin"].astype("string")

    broad_bin_order = [name for name in BROAD_BIN_SPECS.keys() if name in set(matched_pairs["broad_pair_bin"].astype(str))]
    if not broad_bin_order:
        broad_bin_order = sorted(matched_pairs["broad_pair_bin"].dropna().astype(str).unique())

    referenced_unitids = set(matched_pairs["treated_unitid"].tolist()) | set(matched_pairs["control_unitid"].tolist())
    lookup = _load_canonical_school_name_lookup(
        unitids=referenced_unitids,
        crosswalk_path=crosswalk_path,
    )

    table_rows: list[str] = []

    for broad_pair_bin in broad_bin_order:
        bin_mask = matched_pairs["broad_pair_bin"] == broad_pair_bin
        treated_units = pd.Series(matched_pairs.loc[bin_mask, "treated_unitid"].dropna().unique())
        control_units = pd.Series(matched_pairs.loc[bin_mask, "control_unitid"].dropna().unique())
        bin_seed = (seed + abs(zlib.adler32(str(broad_pair_bin).encode("utf-8")) % 10_000) % (2**32))
        sampled_treated = _sample_school_ids(
            values=treated_units,
            sample_size=min(sample_size, len(treated_units)),
            seed=bin_seed,
        )
        sampled_control = _sample_school_ids(
            values=control_units,
            sample_size=min(sample_size, len(control_units)),
            seed=bin_seed + 1,
        )
        treated_label = ", ".join(_school_display_name(unitid, lookup=lookup) for unitid in sampled_treated) or "None"
        control_label = ", ".join(_school_display_name(unitid, lookup=lookup) for unitid in sampled_control) or "None"
        label = BROAD_BIN_PLOT_LABELS.get(str(broad_pair_bin), str(broad_pair_bin))
        table_rows.append(f"{_tex_escape(label)} & {treated_label} & {control_label}\\\\")

    if not table_rows:
        return None

    def _build_table_lines(rows: list[str]) -> list[str]:
        lines = [
            "\\centering",
            "{\\tiny",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\renewcommand{\\arraystretch}{1.0}",
            "\\begin{tabular}{p{0.21\\linewidth}p{0.33\\linewidth}p{0.33\\linewidth}}",
            "\\textbf{Broad bin} & \\textbf{Treated schools (random sample)} & \\textbf{Matched controls (random sample)}\\\\",
            "\\hline",
        ]
        lines.extend(rows)
        lines.extend([
            "\\end{tabular}",
            "}",
            f"\\vspace{{0.15em}}\\\\\n{{\\tiny Random sample size: up to {sample_size} treated and {sample_size} matched control school IDs per broad bin; seed = {seed}.}}",
        ])
        return lines

    out_path = Path(out_dir) / "relabel_broad_bin_treated_control_school_samples.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(_build_table_lines(table_rows)) + "\n")
    chunk_size = 2
    for part_idx, start in enumerate(range(0, len(table_rows), chunk_size), start=1):
        part_rows = table_rows[start : start + chunk_size]
        part_path = out_path.with_name(
            f"{out_path.stem}_part{part_idx}{out_path.suffix}"
        )
        part_path.write_text("\n".join(_build_table_lines(part_rows)) + "\n")
    return out_path


def build_source_target_ctotalt_event_time_summary(
    con: ddb.DuckDBPyConnection,
    panel: pd.DataFrame,
    *,
    ipeds_path: str | Path = base.IPEDS_PATH,
    event_time_min: int = PLOT_EVENT_MIN,
    event_time_max: int = PLOT_EVENT_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = {
        "unitid",
        "awlevel",
        "degree_type",
        "relabel_year",
        "broad_pair_bin",
        "event_source_cip6",
        "target_cip6",
        "event_origin_category",
        "event_flag",
        "broad_bin_eligible",
    }
    if not required_cols.issubset(panel.columns):
        return pd.DataFrame(), pd.DataFrame()

    matched_pairs = match_treated_to_never_treated(
        con,
        panel,
        ipeds_path=ipeds_path,
    )
    if matched_pairs.empty:
        return pd.DataFrame(), matched_pairs
    matched_pairs = matched_pairs[matched_pairs["degree_type"].isin(POOLED_DEGREE_TYPES)].copy()
    if matched_pairs.empty:
        return pd.DataFrame(), matched_pairs

    _ensure_ipeds_view(con, ipeds_path)
    cip_map = _load_ipeds_cip_map(ipeds_path)
    broad_membership = build_broad_bin_membership(cip_map.keys())
    side_rows: list[dict[str, str]] = []
    for broad_pair_bin, membership in broad_membership.items():
        for side, label in (("source", "Source CIP"), ("target", "Target CIP")):
            for cip6 in membership[f"{side}_cips"]:
                side_rows.append(
                    {
                        "broad_pair_bin": str(broad_pair_bin),
                        "series": label,
                        "cip6": str(cip6),
                    }
                )
    side_cips = pd.DataFrame(side_rows)
    if side_cips.empty:
        return pd.DataFrame(), matched_pairs

    con.register("source_target_matches_py", matched_pairs)
    con.register("source_target_side_cips_py", side_cips)
    data_min_year, data_max_year = _ipeds_year_bounds(con)
    pair_values = con.sql(
        f"""
        WITH pair_roles AS (
            SELECT
                CAST(pair_id AS BIGINT) AS pair_id,
                CAST(relabel_year AS INTEGER) AS relabel_year,
                CAST(awlevel AS INTEGER) AS awlevel,
                CAST(degree_type AS VARCHAR) AS degree_type,
                CAST(broad_pair_bin AS VARCHAR) AS broad_pair_bin,
                CAST(treated_unitid AS BIGINT) AS unitid,
                'Treated' AS role
            FROM source_target_matches_py
            UNION ALL
            SELECT
                CAST(pair_id AS BIGINT) AS pair_id,
                CAST(relabel_year AS INTEGER) AS relabel_year,
                CAST(awlevel AS INTEGER) AS awlevel,
                CAST(degree_type AS VARCHAR) AS degree_type,
                CAST(broad_pair_bin AS VARCHAR) AS broad_pair_bin,
                CAST(control_unitid AS BIGINT) AS unitid,
                'Control' AS role
            FROM source_target_matches_py
        ),
        event_grid AS (
            SELECT
                p.*,
                CAST(t.event_t AS INTEGER) AS event_t,
                CAST(p.relabel_year + t.event_t AS INTEGER) AS year
            FROM pair_roles p
            CROSS JOIN generate_series({int(event_time_min)}, {int(event_time_max)}) AS t(event_t)
        )
        SELECT
            g.role,
            g.degree_type,
            g.event_t,
            c.series,
            g.pair_id,
            CASE
                WHEN g.year BETWEEN {int(data_min_year)} AND {int(data_max_year)}
                THEN COALESCE(SUM(CAST(i.ctotalt AS DOUBLE)), 0.0)
                ELSE NULL
            END AS ctotalt
        FROM event_grid g
        JOIN source_target_side_cips_py c
          ON c.broad_pair_bin = g.broad_pair_bin
        LEFT JOIN ipeds_raw i
          ON CAST(i.unitid AS BIGINT) = g.unitid
         AND CAST(i.awlevel AS INTEGER) = g.awlevel
         AND CAST(i.year AS INTEGER) = g.year
         AND LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = c.cip6
        GROUP BY
            g.role,
            g.degree_type,
            g.event_t,
            g.year,
            c.series,
            g.pair_id
        """
    ).df()
    if pair_values.empty:
        return pd.DataFrame(), matched_pairs

    summary = (
        pair_values.groupby(["role", "degree_type", "event_t", "series"], as_index=False)
        .agg(
            mean_ctotalt=("ctotalt", "mean"),
            sd_ctotalt=("ctotalt", "std"),
            n_pairs=("ctotalt", "count"),
        )
        .sort_values(["role", "degree_type", "event_t", "series"])
        .reset_index(drop=True)
    )
    summary["sd_ctotalt"] = pd.to_numeric(summary["sd_ctotalt"], errors="coerce").fillna(0.0)
    summary["se_ctotalt"] = summary["sd_ctotalt"] / np.sqrt(summary["n_pairs"].clip(lower=1))
    summary["ci_low"] = summary["mean_ctotalt"] - 1.96 * summary["se_ctotalt"]
    summary["ci_high"] = summary["mean_ctotalt"] + 1.96 * summary["se_ctotalt"]
    return summary, matched_pairs


def plot_source_target_ctotalt_event_time_by_degree_treated_control(
    con: ddb.DuckDBPyConnection,
    panel: pd.DataFrame,
    *,
    out_dir: str | Path,
    ipeds_path: str | Path = base.IPEDS_PATH,
) -> Path | None:
    summary, matched_pairs = build_source_target_ctotalt_event_time_summary(
        con,
        panel,
        ipeds_path=ipeds_path,
    )
    if summary.empty:
        return None

    out_path = Path(out_dir) / "source_target_ctotalt_event_time_by_degree_treated_control.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path.with_suffix(".csv"), index=False)
    matched_pairs.to_csv(
        out_path.with_name(f"{out_path.stem}_matches.csv"),
        index=False,
    )
    treated_cells = (
        matched_pairs[
            ["degree_type", "treated_unitid", "awlevel", "broad_pair_bin"]
        ]
        .rename(columns={"treated_unitid": "unitid"})
        .assign(role="Treated")
    )
    control_cells = (
        matched_pairs[
            ["degree_type", "control_unitid", "awlevel", "broad_pair_bin"]
        ]
        .rename(columns={"control_unitid": "unitid"})
        .assign(role="Control")
    )
    panel_n = pd.concat([treated_cells, control_cells], ignore_index=True)
    panel_n = (
        panel_n.dropna(subset=["unitid", "awlevel", "broad_pair_bin"])
        .drop_duplicates(["role", "degree_type", "unitid", "awlevel", "broad_pair_bin"])
        .groupby(["role", "degree_type"], as_index=False)
        .size()
        .rename(columns={"size": "unique_unitid_awlevel_broad_bin_n"})
    )
    panel_n.to_csv(
        out_path.with_name(f"{out_path.stem}_panel_n.csv"),
        index=False,
    )
    panel_n_lookup = {
        (str(row.role), str(row.degree_type)): int(row.unique_unitid_awlevel_broad_bin_n)
        for row in panel_n.itertuples(index=False)
    }

    sns.set(style="whitegrid")
    llstyle.apply_style()
    degree_order = [name for name in POOLED_DEGREE_TYPES if name in set(summary["degree_type"].astype(str))]
    if not degree_order:
        degree_order = sorted(summary["degree_type"].dropna().astype(str).unique())
    role_order = ["Treated", "Control"]
    fig, axes = plt.subplots(
        nrows=len(role_order),
        ncols=len(degree_order),
        figsize=(llstyle.FIGSIZE[0], 5.4),
        sharex=True,
        sharey="row",
    )
    if len(degree_order) == 1:
        axes = np.array(axes).reshape(len(role_order), 1)

    color_by_series = {
        "Source CIP": llstyle.color(0),
        "Target CIP": llstyle.color(1),
    }
    for row_idx, role in enumerate(role_order):
        for col_idx, degree_name in enumerate(degree_order):
            ax = axes[row_idx, col_idx]
            degree_df = summary[
                summary["role"].eq(role)
                & summary["degree_type"].astype(str).eq(degree_name)
            ].copy()
            for series in ("Source CIP", "Target CIP"):
                plot_df = degree_df[degree_df["series"].eq(series)].sort_values("event_t")
                if plot_df.empty:
                    continue
                ax.errorbar(
                    plot_df["event_t"],
                    plot_df["mean_ctotalt"],
                    yerr=1.96 * plot_df["se_ctotalt"],
                    color=color_by_series[series],
                    linestyle="-",
                    marker="o",
                    markersize=llstyle.MULTI_MARKER_SIZE,
                    linewidth=1.5,
                    elinewidth=llstyle.MULTI_MARKER_SIZE,
                    alpha=0.9,
                    capsize=0,
                    label=series,
                )
            unique_n = panel_n_lookup.get((role, degree_name), 0)
            ax.text(
                0.98,
                0.95,
                f"N={unique_n}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=BASE_FONT_SIZE - 1,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
            )
            ax.axvline(x=DI_D_EVENT_LINE_X, linestyle=":", color="gray", linewidth=1)
            if row_idx == 0:
                ax.set_title("")
            if col_idx == 0:
                ax.set_ylabel(f"{role}\nMean completions")
            else:
                ax.set_ylabel("")
            if row_idx == len(role_order) - 1:
                ax.set_xlabel("Graduation Cohort Relative to Relabel Event")
            else:
                ax.set_xlabel("")
            ax.set_xticks(list(range(PLOT_EVENT_MIN, PLOT_EVENT_MAX + 1)))
            ax.grid(axis="x", visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    llstyle.right_figure_legend(fig, handles, labels)
    _save_figure(fig, out_path)
    return out_path


def plot_relabel_year_histogram(events: pd.DataFrame, *, degree_type: str, out_dir: str | Path) -> Path | None:
    df = events[
        (events["degree_type"] == degree_type)
        & events["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"])
        & events["relabel_year"].notna()
    ].copy()
    if df.empty:
        return None
    df["relabel_year"] = pd.to_numeric(df["relabel_year"], errors="coerce")
    df = df[df["relabel_year"].le(MAX_RELABEL_YEAR)].copy()
    if df.empty:
        return None
    sns.set(style="whitegrid")
    llstyle.apply_style()
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    bins = range(int(df["relabel_year"].min()), int(df["relabel_year"].max()) + 2)
    sns.histplot(
        data=df,
        x="relabel_year",
        bins=bins,
        discrete=True,
        hue="event_origin_category",
        multiple="dodge",
        ax=ax,
    )
    ax.set_xlabel("Relabel year")
    ax.set_ylabel("Count of relabel events")
    ax.set_title("")
    llstyle.right_legend(ax)
    out_path = Path(out_dir) / f"relabel_year_histogram_{_slugify(degree_type)}.png"
    _save_figure(fig, out_path)
    return out_path


def plot_broad_bin_event_counts_by_year(
    events: pd.DataFrame,
    *,
    out_dir: str | Path,
    degree_type: str | None = None,
) -> Path | None:
    counts = build_broad_bin_year_counts(events, degree_type=degree_type)
    if counts.empty:
        return None
    sns.set(style="whitegrid")
    llstyle.apply_style()
    positive_bins = set(
        counts.loc[counts["event_count"].gt(0), "broad_pair_bin"].dropna().astype(str)
    )
    broad_pair_order = [name for name in BROAD_BIN_SPECS if name in positive_bins]
    if not broad_pair_order:
        broad_pair_order = sorted(counts["broad_pair_bin"].dropna().astype(str).unique())
    pivot = counts.pivot_table(
        index="relabel_year",
        columns="broad_pair_bin",
        values="event_count",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=broad_pair_order, fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    colors = [llstyle.color(idx) for idx in range(len(broad_pair_order))]
    width = min(0.82 / max(len(broad_pair_order), 1), 0.22)
    offsets = (np.arange(len(broad_pair_order)) - (len(broad_pair_order) - 1) / 2.0) * width
    x_values = np.arange(len(pivot), dtype=float)
    for idx, broad_pair_bin in enumerate(broad_pair_order):
        values = pivot[broad_pair_bin].astype(float).to_numpy()
        ax.bar(
            x_values + offsets[idx],
            values,
            width=width,
            color=colors[idx % len(colors)],
            label=BROAD_BIN_PLOT_LABELS.get(broad_pair_bin, broad_pair_bin),
            alpha=0.92,
        )
    ax.set_xlabel("Relabel year")
    ax.set_ylabel("Count of treated events")
    ax.set_title("")
    title_suffix = "All Degrees" if degree_type is None else degree_type
    llstyle.right_legend(ax, title=f"Broad bin ({title_suffix})")
    ax.set_xticks(x_values)
    ax.set_xticklabels(pivot.index.astype(int).tolist())
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    ax.grid(axis="x", visible=False)
    suffix = "all_degrees" if degree_type is None else _slugify(degree_type)
    out_path = Path(out_dir) / f"broad_bin_event_counts_by_year_{suffix}.png"
    _save_figure(fig, out_path)
    return out_path


def plot_opt_usage_generalized(
    opt_usage: pd.DataFrame,
    *,
    yvar: str,
    degree_type: str,
    out_dir: str | Path,
) -> Path | None:
    if yvar not in opt_usage.columns:
        return None
    data = _filter_relabel_analysis_window(opt_usage, year_col="calendar_year")
    data = data.dropna(subset=[yvar, "calendar_year", "relabel_year"]).copy()
    if data.empty:
        return None
    sns.set(style="whitegrid")
    sns.set_palette(llstyle.PALETTE)
    llstyle.apply_style()
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    sns.lineplot(
        data=data,
        x="calendar_year",
        y=yvar,
        hue="relabel_year",
        marker="o",
        markersize=DI_D_PLOT_MARKER_SIZE,
        errorbar=None,
        ax=ax,
    )
    ax.set_ylabel(_outcome_ylabel(yvar), fontsize=DI_D_PLOT_FONT_SIZE)
    _format_yaxis_for_outcome(ax, yvar)
    ax.set_xlabel("Calendar year (program end year)", fontsize=DI_D_PLOT_FONT_SIZE)
    ax.set_title("")
    llstyle.right_legend(ax, title="Relabel year")
    out_path = Path(out_dir) / f"{_slugify(degree_type)}_{yvar}_opt_usage_by_relabel_year.png"
    _save_figure(fig, out_path)
    return out_path


def _program_level_raw_mean_cells(did_panel: pd.DataFrame, *, yvar: str) -> pd.DataFrame:
    """Collapse raw-mean inputs to one program-year/event cell before plotting."""
    if did_panel.empty or yvar not in did_panel.columns:
        return pd.DataFrame()

    df = did_panel[did_panel["event_t"].between(PLOT_EVENT_MIN, PLOT_EVENT_MAX)].copy()
    required = {"event_t", "treated", "unitid", "calendar_year"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df = df.dropna(subset=[yvar, "event_t", "treated", "unitid", "calendar_year"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["event_t"] = pd.to_numeric(df["event_t"], errors="coerce").astype("Int64")
    df["treated"] = pd.to_numeric(df["treated"], errors="coerce").astype("Int64")
    df["unitid"] = pd.to_numeric(df["unitid"], errors="coerce").astype("Int64")
    df["calendar_year"] = pd.to_numeric(df["calendar_year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["event_t", "treated", "unitid", "calendar_year"]).copy()
    if df.empty:
        return pd.DataFrame()

    group_cols = ["event_t", "treated", "unitid", "calendar_year"]
    for col in ("degree_type", "broad_pair_bin", "relabel_type"):
        if col in df.columns:
            group_cols.append(col)

    if "student_id" in df.columns:
        df = df.drop_duplicates(group_cols + ["student_id"]).copy()
    else:
        df = df.drop_duplicates(group_cols).copy()

    if "tuition_total" not in df.columns and {"avg_tuition", "total_grads"}.issubset(df.columns):
        df["tuition_total"] = pd.to_numeric(df["avg_tuition"], errors="coerce") * pd.to_numeric(
            df["total_grads"], errors="coerce"
        )
    if "students_personal_funds_total" not in df.columns and {"avg_students_personal_funds", "total_grads"}.issubset(df.columns):
        df["students_personal_funds_total"] = pd.to_numeric(
            df["avg_students_personal_funds"],
            errors="coerce",
        ) * pd.to_numeric(df["total_grads"], errors="coerce")
    if "total_funds" not in df.columns and {"avg_total_funds", "total_grads"}.issubset(df.columns):
        df["total_funds"] = pd.to_numeric(df["avg_total_funds"], errors="coerce") * pd.to_numeric(
            df["total_grads"],
            errors="coerce",
        )

    numeric_cols = set(PROGRAM_LEVEL_IPEDS_COUNT_YVARS)
    numeric_cols.update(PROGRAM_LEVEL_MEAN_YVARS)
    numeric_cols.update(col for pair in PROGRAM_LEVEL_SHARE_NUMERATOR_DENOMINATOR.values() for col in pair)
    numeric_cols.update(["total_grads", yvar])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg_spec: dict[str, tuple[str, str]] = {}
    for col in PROGRAM_LEVEL_IPEDS_COUNT_YVARS:
        if col in df.columns:
            agg_spec[col] = (col, "max")
    for col in (
        "total_grads",
        "stem_cip_eligible_users",
        "opt_users",
        "opt_stem_users",
        "status_change_users",
        "total_post_grad_authorization_years",
        "total_opt_duration_years",
        "total_opt_years",
        "total_internships",
        "total_internship_opt_years",
        "tuition_total",
        "tuition_ipeds_total",
        "fees_ipeds_total",
        "students_personal_funds_total",
        "total_funds",
    ):
        if col in df.columns:
            agg_spec[col] = (col, "sum")
    for col in PROGRAM_LEVEL_MEAN_YVARS:
        if col in df.columns:
            agg_spec[col] = (col, "mean")
    if yvar not in agg_spec and yvar in df.columns:
        agg_spec[yvar] = (yvar, "mean")

    cells = df.groupby(group_cols, as_index=False, dropna=False).agg(**agg_spec)
    if yvar in PROGRAM_LEVEL_SHARE_NUMERATOR_DENOMINATOR:
        numerator, denominator = PROGRAM_LEVEL_SHARE_NUMERATOR_DENOMINATOR[yvar]
        if numerator in cells.columns and denominator in cells.columns:
            cells[yvar] = _safe_share(cells[numerator], cells[denominator])
    elif yvar in PROGRAM_LEVEL_IPEDS_YVARS:
        cells[yvar] = pd.to_numeric(cells[yvar], errors="coerce")
    elif yvar in PROGRAM_LEVEL_MEAN_YVARS:
        cells[yvar] = pd.to_numeric(cells[yvar], errors="coerce")

    return cells.dropna(subset=[yvar, "event_t", "treated", "unitid"]).copy()


def summarize_did_panel_event_time_simple_means(
    did_panel: pd.DataFrame,
    *,
    yvar: str,
) -> pd.DataFrame:
    columns = ["event_t", "treated", "mean_outcome", "n_rows", "n_units", "total_grads"]
    if did_panel.empty or yvar not in did_panel.columns:
        return pd.DataFrame(columns=columns)

    df = _program_level_raw_mean_cells(did_panel, yvar=yvar)
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["event_t"] = pd.to_numeric(df["event_t"], errors="coerce").astype(int)
    df["treated"] = pd.to_numeric(df["treated"], errors="coerce").astype(int)
    df["unitid"] = pd.to_numeric(df["unitid"], errors="coerce").astype(int)
    df["total_grads"] = pd.to_numeric(df.get("total_grads"), errors="coerce").fillna(0.0)

    grouped = (
        df.groupby(["event_t", "treated"], as_index=False)
        .agg(
            mean_outcome=(yvar, "mean"),
            n_rows=(yvar, "size"),
            n_units=("unitid", "nunique"),
            total_grads=("total_grads", "sum"),
        )
        .sort_values(["treated", "event_t"])
        .reset_index(drop=True)
    )
    return grouped.loc[:, columns]


def plot_event_time_with_control_generalized(
    did_panel: pd.DataFrame,
    *,
    yvar: str,
    degree_type: str,
    out_dir: str | Path,
) -> tuple[Path | None, Path | None]:
    summary = summarize_did_panel_event_time_simple_means(did_panel, yvar=yvar)
    treated_event = summary[summary["treated"].eq(1)].copy()
    control_event = summary[summary["treated"].eq(0)].copy()
    if treated_event.empty:
        return None, None
    sns.set(style="whitegrid")
    sns.set_palette(llstyle.PALETTE)
    llstyle.apply_style()

    treated_path: Path | None = None
    combined_path: Path | None = None

    fig_t, ax_t = plt.subplots(figsize=llstyle.FIGSIZE)
    sns.lineplot(
        data=treated_event,
        x="event_t",
        y="mean_outcome",
        marker="o",
        markersize=DI_D_PLOT_MARKER_SIZE,
        ax=ax_t,
    )
    ax_t.set_ylabel(_outcome_ylabel(yvar), fontsize=DI_D_PLOT_FONT_SIZE)
    _format_yaxis_for_outcome(ax_t, yvar)
    ax_t.set_xlabel("Graduation Cohort Relative to Relabel Event", fontsize=DI_D_PLOT_FONT_SIZE)
    ax_t.set_title("")
    ax_t.axvline(x=DI_D_EVENT_LINE_X, linestyle="--", color="gray", linewidth=1)
    treated_path = Path(out_dir) / f"{_slugify(degree_type)}_{yvar}_event_time_treated.png"
    _save_figure(fig_t, treated_path)

    if control_event is None or control_event.empty:
        return treated_path, None

    treated_plot = treated_event.copy()
    treated_plot["series_label"] = "Treated"
    control_plot = control_event.copy()
    control_plot["series_label"] = "Matched Never-Treated"
    plot_df = pd.concat([treated_plot, control_plot], ignore_index=True)

    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    sns.lineplot(
        data=plot_df,
        x="event_t",
        y="mean_outcome",
        hue="series_label",
        marker="o",
        markersize=DI_D_PLOT_MARKER_SIZE,
        ax=ax,
    )
    ax.set_ylabel(_outcome_ylabel(yvar), fontsize=DI_D_PLOT_FONT_SIZE)
    _format_yaxis_for_outcome(ax, yvar)
    ax.set_xlabel("Graduation Cohort Relative to Relabel Event", fontsize=DI_D_PLOT_FONT_SIZE)
    ax.set_title("")
    ax.axvline(x=DI_D_EVENT_LINE_X, linestyle="--", color="gray", linewidth=1)
    llstyle.right_legend(ax)
    combined_path = Path(out_dir) / f"{_slugify(degree_type)}_{yvar}_event_time_treated_control_never_treated.png"
    _save_figure(fig, combined_path)
    return treated_path, combined_path


def _did_coef_ylabel(yvar: str) -> str:
    return f"did coef: {_outcome_ylabel(yvar)}"


def plot_did_event_study_generalized(
    did_event_study: pd.DataFrame,
    *,
    yvar: str,
    degree_type: str,
    out_dir: str | Path,
    file_stem: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    reference_event_time: int = DI_D_REFERENCE_EVENT_TIME,
    summary_text: str | None = None,
) -> Path | None:
    if did_event_study.empty:
        return None
    plot_df = did_event_study.copy()
    if "reference_event_t" in plot_df.columns:
        plot_df = plot_df[plot_df["reference_event_t"] == reference_event_time].copy()
    if plot_df.empty:
        return None
    plot_df = plot_df.sort_values("event_t").copy()
    plot_df["event_t"] = pd.to_numeric(plot_df["event_t"], errors="coerce")
    plot_df = plot_df[plot_df["event_t"].between(PLOT_EVENT_MIN, PLOT_EVENT_MAX)].copy()
    plot_df = plot_df.dropna(subset=["event_t", "coef", "se"]).copy()
    if plot_df.empty:
        return None
    sns.set(style="whitegrid")
    sns.set_palette(llstyle.PALETTE)
    llstyle.apply_style()
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    ax.plot(
        plot_df["event_t"],
        plot_df["coef"],
        color=llstyle.color(0),
        linewidth=1.5,
        alpha=0.9,
        zorder=1,
    )
    ax.scatter(
        plot_df["event_t"],
        plot_df["coef"],
        color=llstyle.color(0),
        s=DI_D_PLOT_MARKER_SIZE * DI_D_PLOT_MARKER_SIZE,
        marker="o",
        linewidths=0,
    )
    ax.errorbar(
        plot_df["event_t"],
        plot_df["coef"],
        yerr=plot_df["se"],
        fmt="none",
        ecolor=llstyle.rgba(llstyle.color(0), DI_D_ERRORBAR_ALPHA),
        capsize=0,
        elinewidth=DI_D_PLOT_MARKER_SIZE,
        label="_nolegend_",
    )
    ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
    ax.axvline(x=DI_D_EVENT_LINE_X, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Graduation Cohort Relative to Relabel Event", fontsize=DI_D_PLOT_FONT_SIZE)
    ax.set_ylabel(ylabel or _did_coef_ylabel(yvar), fontsize=DI_D_PLOT_FONT_SIZE)
    _format_yaxis_for_outcome(ax, yvar)
    ax.set_title("")
    _add_did_summary_text(ax, summary_text)
    default_stem = f"{_slugify(degree_type)}_{yvar}_did_event_time_never_treated"
    out_path = Path(out_dir) / f"{file_stem or default_stem}.png"
    csv_path = out_path.with_suffix(".csv")
    plot_df.to_csv(csv_path, index=False)
    _save_figure(fig, out_path)
    return out_path


def plot_broad_bin_did_event_study_generalized(
    did_event_study: pd.DataFrame,
    *,
    yvar: str,
    degree_type: str,
    out_dir: str | Path,
    file_stem: str,
    reference_event_time: int = DI_D_REFERENCE_EVENT_TIME,
    title: str | None = None,
    summary_text: str | None = None,
) -> Path | None:
    if did_event_study.empty:
        return None
    required = {"event_t", "coef", "se", "broad_pair_bin", "reference_event_t"}
    if not required.issubset(did_event_study.columns):
        return None

    plot_df = did_event_study[
        did_event_study["reference_event_t"] == reference_event_time
    ].copy()
    if plot_df.empty:
        return None

    plot_df = plot_df.copy()
    plot_df["event_t"] = pd.to_numeric(plot_df["event_t"], errors="coerce")
    plot_df = plot_df[plot_df["event_t"].between(PLOT_EVENT_MIN, PLOT_EVENT_MAX)].copy()
    plot_df = plot_df.dropna(subset=["event_t", "coef", "se"]).copy()
    if plot_df.empty:
        return None

    plot_df["broad_pair_bin"] = plot_df["broad_pair_bin"].astype("string")
    available_bins = {
        str(value)
        for value in plot_df["broad_pair_bin"].dropna().tolist()
        if str(value)
    }
    broad_bins = [name for name in BROAD_BIN_SPECS if name in available_bins]
    if not broad_bins:
        broad_bins = sorted(available_bins)
    if not broad_bins:
        return None

    plot_df["broad_pair_label"] = plot_df["broad_pair_bin"].map(BROAD_BIN_PLOT_LABELS).fillna(
        plot_df["broad_pair_bin"]
    )
    offset_step = _multi_series_offset_step(len(broad_bins))
    center = (len(broad_bins) - 1) / 2

    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    sns.set(style="whitegrid")
    sns.set_palette(llstyle.PALETTE)
    llstyle.apply_style()

    plotted_any = False
    for idx, broad_pair_bin in enumerate(broad_bins):
        bin_df = plot_df[plot_df["broad_pair_bin"] == broad_pair_bin].sort_values("event_t").copy()
        if bin_df.empty:
            continue
        plotted_any = True
        marker = DI_D_BROAD_BIN_MARKERS[idx % len(DI_D_BROAD_BIN_MARKERS)]
        color = llstyle.color(idx)
        label = str(bin_df["broad_pair_label"].iloc[0])
        offset = (idx - center) * offset_step
        bin_df["event_t_plot"] = bin_df["event_t"] + offset
        ax.plot(
            bin_df["event_t_plot"],
            bin_df["coef"],
            color=color,
            linewidth=1.2,
            alpha=0.9,
            zorder=1,
        )
        ax.scatter(
            bin_df["event_t_plot"],
            bin_df["coef"],
            color=color,
            s=MULTI_SERIES_MARKER_AREA,
            marker=marker,
            label=label,
        )
        ax.errorbar(
            bin_df["event_t_plot"],
            bin_df["coef"],
            yerr=bin_df["se"],
            fmt="none",
            color=color,
            ecolor=llstyle.rgba(color, DI_D_ERRORBAR_ALPHA),
            capsize=0,
            elinewidth=MULTI_SERIES_ERRORBAR_WIDTH,
            label="_nolegend_",
        )

    if not plotted_any:
        return None

    ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
    ax.axvline(x=DI_D_EVENT_LINE_X, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Graduation Cohort Relative to Relabel Event", fontsize=DI_D_PLOT_FONT_SIZE)
    ax.set_ylabel(_did_coef_ylabel(yvar), fontsize=DI_D_PLOT_FONT_SIZE)
    _format_yaxis_for_outcome(ax, yvar)
    ax.set_title("")
    event_ticks = list(range(PLOT_EVENT_MIN, PLOT_EVENT_MAX + 1))
    ax.set_xticks(event_ticks)
    ax.set_xlim(PLOT_EVENT_MIN - 0.6, PLOT_EVENT_MAX + 0.6)
    llstyle.right_legend(ax, title="Broad bin")
    _add_did_summary_text(ax, None)

    out_path = Path(out_dir) / f"{file_stem}.png"
    csv_path = out_path.with_suffix(".csv")
    plot_df.to_csv(csv_path, index=False)
    _save_figure(fig, out_path)
    return out_path


def plot_degree_level_did_event_study_generalized(
    did_event_study: pd.DataFrame,
    *,
    yvar: str,
    out_dir: str | Path,
    file_stem: str,
    reference_event_time: int = DI_D_REFERENCE_EVENT_TIME,
    title: str | None = None,
    summary_text: str | None = None,
) -> Path | None:
    if did_event_study.empty:
        return None
    required = {"event_t", "coef", "se", "degree_type", "reference_event_t"}
    if not required.issubset(did_event_study.columns):
        return None

    plot_df = did_event_study[
        did_event_study["reference_event_t"] == reference_event_time
    ].copy()
    if plot_df.empty:
        return None

    plot_df["event_t"] = pd.to_numeric(plot_df["event_t"], errors="coerce")
    plot_df = plot_df[plot_df["event_t"].between(PLOT_EVENT_MIN, PLOT_EVENT_MAX)].copy()
    plot_df = plot_df.dropna(subset=["event_t", "coef", "se", "degree_type"]).copy()
    if plot_df.empty:
        return None

    degree_order = [name for name in POOLED_DEGREE_TYPES if name in set(plot_df["degree_type"].astype(str))]
    if not degree_order:
        degree_order = sorted(plot_df["degree_type"].dropna().astype(str).unique())
    if not degree_order:
        return None

    offset_step = _multi_series_offset_step(len(degree_order))
    center = (len(degree_order) - 1) / 2
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    sns.set(style="whitegrid")
    sns.set_palette(llstyle.PALETTE)
    llstyle.apply_style()

    plotted_any = False
    for idx, degree_name in enumerate(degree_order):
        degree_df = plot_df[plot_df["degree_type"].astype(str) == degree_name].sort_values("event_t").copy()
        if degree_df.empty:
            continue
        plotted_any = True
        marker = DI_D_BROAD_BIN_MARKERS[idx % len(DI_D_BROAD_BIN_MARKERS)]
        color = llstyle.color(idx)
        offset = (idx - center) * offset_step
        degree_df["event_t_plot"] = degree_df["event_t"] + offset
        ax.plot(
            degree_df["event_t_plot"],
            degree_df["coef"],
            color=color,
            linewidth=1.2,
            alpha=0.9,
            zorder=1,
        )
        ax.scatter(
            degree_df["event_t_plot"],
            degree_df["coef"],
            color=color,
            s=MULTI_SERIES_MARKER_AREA,
            marker=marker,
            label=degree_name,
        )
        ax.errorbar(
            degree_df["event_t_plot"],
            degree_df["coef"],
            yerr=degree_df["se"],
            fmt="none",
            color=color,
            ecolor=llstyle.rgba(color, DI_D_ERRORBAR_ALPHA),
            capsize=0,
            elinewidth=MULTI_SERIES_ERRORBAR_WIDTH,
            label="_nolegend_",
        )

    if not plotted_any:
        return None

    ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
    ax.axvline(x=DI_D_EVENT_LINE_X, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Graduation Cohort Relative to Relabel Event", fontsize=DI_D_PLOT_FONT_SIZE)
    ax.set_ylabel(_did_coef_ylabel(yvar), fontsize=DI_D_PLOT_FONT_SIZE)
    _format_yaxis_for_outcome(ax, yvar)
    ax.set_title("")
    ax.set_xticks(list(range(PLOT_EVENT_MIN, PLOT_EVENT_MAX + 1)))
    ax.set_xlim(PLOT_EVENT_MIN - 0.6, PLOT_EVENT_MAX + 0.6)
    llstyle.right_legend(ax, title="Degree level")
    _add_did_summary_text(ax, None)

    out_path = Path(out_dir) / f"{file_stem}.png"
    plot_df.to_csv(out_path.with_suffix(".csv"), index=False)
    _save_figure(fig, out_path)
    return out_path


def _top_relabel_years_for_calendar_appendix(
    did_panel: pd.DataFrame,
    *,
    top_n: int = CALENDAR_YEAR_APPENDIX_TOP_N,
) -> list[int]:
    if did_panel.empty or "relabel_year" not in did_panel.columns:
        return []
    treated_df = did_panel[did_panel["treated"].eq(1)].copy()
    if treated_df.empty:
        return []
    count_col = "pair_id" if "pair_id" in treated_df.columns else "unitid"
    counts = (
        treated_df.groupby("relabel_year", as_index=False)
        .agg(n_events=(count_col, "nunique"))
        .sort_values(["n_events", "relabel_year"], ascending=[False, True])
    )
    return (
        pd.to_numeric(counts["relabel_year"], errors="coerce")
        .dropna()
        .astype(int)
        .head(top_n)
        .tolist()
    )


def compute_calendar_year_did_by_relabel_year(
    did_panel: pd.DataFrame,
    *,
    yvar: str,
    relabel_years: list[int] | None = None,
    top_n: int = CALENDAR_YEAR_APPENDIX_TOP_N,
    did_spec: str | None = None,
    use_weights: bool = False,
) -> pd.DataFrame:
    if did_panel.empty or yvar not in did_panel.columns:
        return pd.DataFrame()
    relabel_years = relabel_years or _top_relabel_years_for_calendar_appendix(did_panel, top_n=top_n)
    if not relabel_years:
        return pd.DataFrame()

    normalized_did_spec = _did_spec_from_panel(did_panel, did_spec=did_spec)
    frames: list[pd.DataFrame] = []
    for relabel_year in relabel_years:
        cohort_df = did_panel[pd.to_numeric(did_panel["relabel_year"], errors="coerce").eq(int(relabel_year))].copy()
        if cohort_df.empty:
            continue
        cohort_df = cohort_df[
            cohort_df["calendar_year"].between(base.PLOT_YEAR_MIN, _analysis_year_max_for_yvar(yvar))
        ].copy()
        cohort_df = cohort_df.dropna(subset=[yvar, "calendar_year", "treated", "unitid", "total_grads"]).copy()
        if cohort_df.empty:
            continue

        cohort_df["calendar_year"] = pd.to_numeric(cohort_df["calendar_year"], errors="coerce").astype(int)
        cohort_df["treated"] = pd.to_numeric(cohort_df["treated"], errors="coerce").astype(int)
        cohort_df["unitid"] = pd.to_numeric(cohort_df["unitid"], errors="coerce").astype(int)
        cohort_df["total_grads"] = pd.to_numeric(cohort_df["total_grads"], errors="coerce").fillna(0.0)
        cohort_df = cohort_df[cohort_df["total_grads"] > 0].copy()
        if cohort_df.empty or cohort_df["unitid"].nunique() < 2:
            continue
        cohort_df = _apply_did_design_columns(cohort_df, did_spec=normalized_did_spec)

        reference_year = int(relabel_year) + DI_D_REFERENCE_EVENT_TIME
        treated_years = set(cohort_df.loc[cohort_df["treated"] == 1, "calendar_year"].dropna().astype(int).tolist())
        control_years = set(cohort_df.loc[cohort_df["treated"] == 0, "calendar_year"].dropna().astype(int).tolist())
        calendar_years = sorted(treated_years & control_years)
        cohort_df = cohort_df[cohort_df["calendar_year"].isin(calendar_years)].copy()
        if reference_year not in calendar_years or len(calendar_years) < 2:
            continue

        backend = _did_backend(normalized_did_spec)
        result = None
        params = pd.Series(dtype=float)
        bse = pd.Series(dtype=float)
        if backend == "pyfixest":
            try:
                result = _fit_calendar_year_pyfixest(
                    cohort_df,
                    yvar=yvar,
                    did_spec=normalized_did_spec,
                    reference_year=reference_year,
                    use_weights=use_weights,
                )
                if result is not None:
                    params, bse = _result_params_and_bse(result, backend=backend)
            except Exception as exc:
                _progress(
                    f"pyfixest calendar-year DiD failed for {yvar}, relabel_year={relabel_year}; "
                    f"falling back to statsmodels ({exc})"
                )
                backend = "statsmodels"

        if backend == "statsmodels":
            try:
                import statsmodels.formula.api as smf
            except Exception as exc:
                _progress(f"statsmodels unavailable; skipping calendar-year DiD appendix for {yvar} ({exc})")
                return pd.DataFrame()
            fe_term = _did_fe_formula_term(cohort_df, did_spec=normalized_did_spec)
            formula = (
                f"{yvar} ~ treated + "
                f"treated:C(calendar_year, Treatment(reference={reference_year})) + "
                f"{fe_term}"
            )
            try:
                model = smf.wls(formula, data=cohort_df, weights=cohort_df["total_grads"]) if use_weights else smf.ols(formula, data=cohort_df)
                result = model.fit(
                    cov_type="cluster",
                    cov_kwds={"groups": cohort_df["unitid"]},
                )
            except Exception as exc:
                _progress(
                    f"Clustered calendar-year DiD failed for {yvar}, relabel_year={relabel_year}; "
                    f"falling back to HC1 ({exc})"
                )
                model = smf.wls(formula, data=cohort_df, weights=cohort_df["total_grads"]) if use_weights else smf.ols(formula, data=cohort_df)
                result = model.fit(cov_type="HC1")
            params, bse = _result_params_and_bse(result, backend=backend)

        nobs = int(getattr(result, "nobs", len(cohort_df)) or len(cohort_df))

        counts = (
            cohort_df.groupby(["calendar_year", "treated"], as_index=False)
            .agg(
                n_school_years=("unitid", "size"),
                n_schools=("unitid", "nunique"),
                total_grads=("total_grads", "sum"),
            )
        )
        treated_counts = (
            counts[counts["treated"] == 1]
            .drop(columns=["treated"])
            .rename(
                columns={
                    "n_school_years": "treated_n_school_years",
                    "n_schools": "treated_n_schools",
                    "total_grads": "treated_total_grads",
                }
            )
        )
        control_counts = (
            counts[counts["treated"] == 0]
            .drop(columns=["treated"])
            .rename(
                columns={
                    "n_school_years": "control_n_school_years",
                    "n_schools": "control_n_schools",
                    "total_grads": "control_total_grads",
                }
            )
        )
        year_count_lookup = treated_counts.merge(control_counts, on="calendar_year", how="outer").set_index("calendar_year").to_dict("index")

        rows: list[dict[str, object]] = []
        for calendar_year in calendar_years:
            count_row = year_count_lookup.get(int(calendar_year), {})
            if int(calendar_year) == reference_year:
                coef = 0.0
                se = 0.0
            else:
                param = _find_calendar_year_interaction_param(
                    params,
                    calendar_year=int(calendar_year),
                    reference_year=reference_year,
                )
                coef = float(params.get(param, float("nan"))) if param is not None else float("nan")
                se = float(bse.get(param, float("nan"))) if param is not None else float("nan")
            rows.append(
                {
                    "calendar_year": int(calendar_year),
                    "coef": coef,
                    "se": se,
                    "ci_low": coef - 1.96 * se,
                    "ci_high": coef + 1.96 * se,
                    "relabel_year": int(relabel_year),
                    "reference_year": int(reference_year),
                    "nobs": nobs,
                    "n_schools_total": int(cohort_df["unitid"].nunique()),
                    "treated_n_school_years": int(count_row.get("treated_n_school_years", 0) or 0),
                    "control_n_school_years": int(count_row.get("control_n_school_years", 0) or 0),
                    "treated_n_schools": int(count_row.get("treated_n_schools", 0) or 0),
                    "control_n_schools": int(count_row.get("control_n_schools", 0) or 0),
                    "treated_total_grads": float(count_row.get("treated_total_grads", 0.0) or 0.0),
                    "control_total_grads": float(count_row.get("control_total_grads", 0.0) or 0.0),
                }
            )
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_calendar_year_did_by_relabel_year(
    calendar_year_did: pd.DataFrame,
    *,
    yvar: str,
    out_dir: str | Path,
    file_stem: str,
    title: str | None = None,
    summary_text: str | None = None,
) -> Path | None:
    if calendar_year_did.empty:
        return None
    required = {"calendar_year", "coef", "se", "relabel_year"}
    if not required.issubset(calendar_year_did.columns):
        return None

    plot_df = calendar_year_did.copy()
    plot_df["calendar_year"] = pd.to_numeric(plot_df["calendar_year"], errors="coerce")
    plot_df = plot_df.dropna(subset=["calendar_year", "coef", "se", "relabel_year"]).copy()
    if plot_df.empty:
        return None
    plot_df["relabel_year"] = pd.to_numeric(plot_df["relabel_year"], errors="coerce").astype(int)
    relabel_years = sorted(plot_df["relabel_year"].dropna().unique().tolist())
    if not relabel_years:
        return None

    sns.set(style="whitegrid")
    sns.set_palette(llstyle.PALETTE)
    llstyle.apply_style()
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    offset_step = _multi_series_offset_step(len(relabel_years))
    center = (len(relabel_years) - 1) / 2
    plotted_any = False
    for idx, relabel_year in enumerate(relabel_years):
        cohort_df = plot_df[plot_df["relabel_year"] == int(relabel_year)].sort_values("calendar_year").copy()
        if cohort_df.empty:
            continue
        plotted_any = True
        marker = DI_D_BROAD_BIN_MARKERS[idx % len(DI_D_BROAD_BIN_MARKERS)]
        color = llstyle.color(idx)
        offset = (idx - center) * offset_step
        cohort_df["calendar_year_plot"] = cohort_df["calendar_year"] + offset
        ax.plot(
            cohort_df["calendar_year_plot"],
            cohort_df["coef"],
            color=color,
            linewidth=1.2,
            alpha=0.9,
            zorder=1,
        )
        ax.scatter(
            cohort_df["calendar_year_plot"],
            cohort_df["coef"],
            color=color,
            s=MULTI_SERIES_MARKER_AREA,
            marker=marker,
            label=f"Relabel year {int(relabel_year)}",
        )
        ax.errorbar(
            cohort_df["calendar_year_plot"],
            cohort_df["coef"],
            yerr=cohort_df["se"],
            fmt="none",
            color=color,
            ecolor=llstyle.rgba(color, DI_D_ERRORBAR_ALPHA),
            capsize=0,
            elinewidth=MULTI_SERIES_ERRORBAR_WIDTH,
            label="_nolegend_",
        )

    if not plotted_any:
        return None

    ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Calendar year", fontsize=DI_D_PLOT_FONT_SIZE)
    ax.set_ylabel(_did_coef_ylabel(yvar), fontsize=DI_D_PLOT_FONT_SIZE)
    _format_yaxis_for_outcome(ax, yvar)
    ax.set_title("")
    calendar_year_ticks = sorted(plot_df["calendar_year"].dropna().astype(int).unique().tolist())
    ax.set_xticks(calendar_year_ticks)
    ax.set_xlim(min(calendar_year_ticks) - 0.8, max(calendar_year_ticks) + 0.8)
    llstyle.right_legend(ax, title="Cohort")
    _add_did_summary_text(ax, None)

    out_path = Path(out_dir) / f"{file_stem}.png"
    plot_df.to_csv(out_path.with_suffix(".csv"), index=False)
    _save_figure(fig, out_path)
    return out_path


def write_grouped_did_appendices(
    did_panel: pd.DataFrame,
    *,
    degree_level_panels: dict[str, pd.DataFrame] | None = None,
    ipeds_did_panel: pd.DataFrame | None = None,
    ipeds_degree_level_panels: dict[str, pd.DataFrame] | None = None,
    out_dir: str | Path,
    yvars: list[str] | None = None,
    did_spec: str | None = None,
    include_degree_level_plots: bool = DEFAULT_INCLUDE_DEGREE_SPECIFIC_PLOTS,
) -> list[Path]:
    if did_panel.empty and (ipeds_did_panel is None or ipeds_did_panel.empty):
        return []
    yvars = yvars or MASTER_APPENDIX_DID_YVARS
    broad_bin_dir = Path(out_dir) / "pooled_broad_bin_did_appendix"
    degree_level_dir = Path(out_dir) / "pooled_degree_level_did_appendix"
    calendar_year_dir = Path(out_dir) / "pooled_calendar_year_did_appendix"
    broad_bin_dir.mkdir(parents=True, exist_ok=True)
    degree_level_dir.mkdir(parents=True, exist_ok=True)
    calendar_year_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    broad_bin_entries: list[tuple[str, Path]] = []
    degree_entries: list[tuple[str, Path]] = []
    calendar_year_entries: list[tuple[str, Path]] = []
    panel_for_bins = did_panel if not did_panel.empty else ipeds_did_panel
    if panel_for_bins is None or panel_for_bins.empty:
        return []
    broad_pair_key = "broad_pair_bin" if "broad_pair_bin" in panel_for_bins.columns else "relabel_type"

    broad_bin_values = {
        str(value)
        for value in panel_for_bins[broad_pair_key].dropna().astype(str).unique().tolist()
        if str(value)
    }
    broad_bins = [name for name in BROAD_BIN_SPECS if name in broad_bin_values]
    if not broad_bins:
        broad_bins = sorted(broad_bin_values)
    for yvar in yvars:
        active_panel = ipeds_did_panel if yvar in PROGRAM_LEVEL_IPEDS_YVARS and ipeds_did_panel is not None else did_panel
        if active_panel.empty:
            continue
        did_study_frames: list[pd.DataFrame] = []
        for broad_pair_bin in broad_bins:
            bin_panel = active_panel[active_panel[broad_pair_key].astype(str) == broad_pair_bin].copy()
            if bin_panel.empty:
                continue
            did_event_study = compute_did_event_study_generalized(
                did_panel=bin_panel,
                yvar=yvar,
                did_spec=did_spec,
                use_weights=False,
                reference_event_time=DI_D_REFERENCE_EVENT_TIME,
            )
            if did_event_study.empty:
                continue
            did_event_study = did_event_study.assign(
                broad_pair_bin=broad_pair_bin,
            )
            did_study_frames.append(did_event_study)

        if not did_study_frames:
            continue

        did_event_study = pd.concat(did_study_frames, ignore_index=True)
        summary_text = build_did_summary_text(active_panel, yvar=yvar, did_spec=did_spec)
        out_path = plot_broad_bin_did_event_study_generalized(
            did_event_study,
            yvar=yvar,
            degree_type="Pooled",
            out_dir=broad_bin_dir,
            file_stem=f"pooled_broad_bins_{yvar}_did_event_time_never_treated",
            title="Pooled sample | Broad bins",
            reference_event_time=DI_D_REFERENCE_EVENT_TIME,
            summary_text=summary_text,
        )
        if out_path is None:
            continue
        generated_paths.append(out_path)
        generated_paths.append(out_path.with_suffix(".csv"))
        broad_bin_entries.append((yvar, out_path))

        if include_degree_level_plots and degree_level_panels:
            active_degree_level_panels = (
                ipeds_degree_level_panels
                if yvar in PROGRAM_LEVEL_IPEDS_YVARS and ipeds_degree_level_panels is not None
                else degree_level_panels
            )
            degree_frames: list[pd.DataFrame] = []
            for degree_name in POOLED_DEGREE_TYPES:
                degree_panel = active_degree_level_panels.get(degree_name) if active_degree_level_panels else None
                if degree_panel is None or degree_panel.empty:
                    continue
                degree_study = compute_did_event_study_generalized(
                    did_panel=degree_panel,
                    yvar=yvar,
                    did_spec=did_spec,
                    use_weights=False,
                    reference_event_time=DI_D_REFERENCE_EVENT_TIME,
                )
                if degree_study.empty:
                    continue
                degree_frames.append(degree_study.assign(degree_type=degree_name))
            if degree_frames:
                degree_event_study = pd.concat(degree_frames, ignore_index=True)
                degree_out_path = plot_degree_level_did_event_study_generalized(
                    degree_event_study,
                    yvar=yvar,
                    out_dir=degree_level_dir,
                    file_stem=f"pooled_degree_levels_{yvar}_did_event_time_never_treated",
                    title="Pooled sample | Degree levels",
                    reference_event_time=DI_D_REFERENCE_EVENT_TIME,
                    summary_text=summary_text,
                )
                if degree_out_path is not None:
                    generated_paths.append(degree_out_path)
                    generated_paths.append(degree_out_path.with_suffix(".csv"))
                    degree_entries.append((yvar, degree_out_path))

        calendar_year_study = compute_calendar_year_did_by_relabel_year(
            active_panel,
            yvar=yvar,
            top_n=CALENDAR_YEAR_APPENDIX_TOP_N,
            did_spec=did_spec,
            use_weights=False,
        )
        if not calendar_year_study.empty:
            calendar_year_out_path = plot_calendar_year_did_by_relabel_year(
                calendar_year_study,
                yvar=yvar,
                out_dir=calendar_year_dir,
                file_stem=f"pooled_calendar_years_{yvar}_did_by_relabel_year",
                title="Pooled sample | Calendar-year DiD by relabel cohort",
                summary_text=summary_text,
            )
            if calendar_year_out_path is not None:
                generated_paths.append(calendar_year_out_path)
                generated_paths.append(calendar_year_out_path.with_suffix(".csv"))
                calendar_year_entries.append((yvar, calendar_year_out_path))

    if broad_bin_entries:
        broad_bin_path = Path(out_dir) / "pooled_broad_bin_did_appendix.md"
        lines = [
            "# Pooled Broad-Bin DiD Appendix",
            "",
            "Single pooled-sample DiD plot per outcome with broad bins overlaid",
            "in distinct colors and marker styles.",
            "",
        ]
        for yvar, plot_path in broad_bin_entries:
            rel_plot_path = plot_path.relative_to(Path(out_dir))
            rel_csv_path = plot_path.with_suffix(".csv").relative_to(Path(out_dir))
            lines.append(
                f"- {base.yvar_label(yvar)}: [{rel_plot_path.as_posix()}] "
                f"([csv]({rel_csv_path.as_posix()}))"
            )
        broad_bin_path.write_text("\n".join(lines).rstrip() + "\n")
        generated_paths.append(broad_bin_path)

    if degree_entries:
        degree_path = Path(out_dir) / "pooled_degree_level_did_appendix.md"
        lines = [
            "# Pooled Degree-Level DiD Appendix",
            "",
            "Single pooled-sample DiD plot per outcome with bachelor's, master's,",
            "and doctoral event-study coefficients overlaid.",
            "",
        ]
        for yvar, plot_path in degree_entries:
            rel_plot_path = plot_path.relative_to(Path(out_dir))
            rel_csv_path = plot_path.with_suffix(".csv").relative_to(Path(out_dir))
            lines.append(
                f"- {base.yvar_label(yvar)}: [{rel_plot_path.as_posix()}] "
                f"([csv]({rel_csv_path.as_posix()}))"
            )
        degree_path.write_text("\n".join(lines).rstrip() + "\n")
        generated_paths.append(degree_path)

    if calendar_year_entries:
        calendar_year_path = Path(out_dir) / "pooled_calendar_year_did_appendix.md"
        lines = [
            "# Pooled Calendar-Year DiD Appendix",
            "",
            "Single pooled-sample DiD plot per outcome with separate cohort lines",
            f"for the top {CALENDAR_YEAR_APPENDIX_TOP_N} relabel years by treated-event count.",
            "",
        ]
        for yvar, plot_path in calendar_year_entries:
            rel_plot_path = plot_path.relative_to(Path(out_dir))
            rel_csv_path = plot_path.with_suffix(".csv").relative_to(Path(out_dir))
            lines.append(
                f"- {base.yvar_label(yvar)}: [{rel_plot_path.as_posix()}] "
                f"([csv]({rel_csv_path.as_posix()}))"
            )
        calendar_year_path.write_text("\n".join(lines).rstrip() + "\n")
        generated_paths.append(calendar_year_path)

    if generated_paths:
        return generated_paths

    appendix_path = Path(out_dir) / "pooled_broad_bin_did_appendix.md"
    lines = [
        "# Pooled DiD Appendix",
        "",
        "No grouped pooled-sample DiD plots were generated.",
        "",
    ]
    appendix_path.write_text("\n".join(lines).rstrip() + "\n")
    generated_paths.append(appendix_path)
    return generated_paths


def write_generalized_report(
    events: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    out_path: str | Path,
    *,
    plot_appendices: list[Path] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("Generalized relabel events identified in relabels_revelio")
    lines.append("")
    lines.append("Reference notes")
    lines.append(f"- IPEDS source: {base.IPEDS_PATH}")
    lines.append(f"- FOIA source: {base.FOIA_PATH}")
    lines.append(f"- Candidate rows audited: {len(candidate_audit)}")
    lines.append("")

    verified_candidates = int((candidate_audit.get("external_verified", pd.Series(dtype=int)) == 1).sum())
    unmatched_schools = int(candidate_audit.get("matched_unitid", pd.Series(dtype=float)).isna().sum()) if not candidate_audit.empty else 0
    unverified_candidates = int(len(candidate_audit) - verified_candidates) if not candidate_audit.empty else 0
    lines.append("Candidate verification summary")
    lines.append(f"- Total external candidates: {len(candidate_audit)}")
    lines.append(f"- Verified against IPEDS: {verified_candidates}")
    lines.append(f"- Not verified: {unverified_candidates}")
    lines.append(f"- Unmatched schools: {unmatched_schools}")
    lines.append("")

    if not events.empty:
        lines.append("Event counts by origin")
        for category, count in events["event_origin_category"].value_counts(dropna=False).sort_index().items():
            lines.append(f"- {category}: {int(count)}")
        lines.append("")

        if plot_appendices:
            lines.append("Plot appendices")
            for appendix_path in plot_appendices:
                lines.append(f"- {appendix_path}")
            lines.append("")

        lines.append("Event counts by degree x origin")
        degree_counts = (
            events.groupby(["degree_type", "event_origin_category"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["degree_type", "event_origin_category"])
        )
        for row in degree_counts.itertuples(index=False):
            lines.append(f"- {row.degree_type} | {row.event_origin_category}: {int(row.n)}")
        lines.append("")

        lines.append("Source major counts by origin")
        source_counts = (
            events[events["source_major"].notna()]
            .groupby(["source_major", "event_origin_category"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["source_major", "event_origin_category"])
        )
        for row in source_counts.itertuples(index=False):
            lines.append(f"- {row.source_major} | {row.event_origin_category}: {int(row.n)}")
        lines.append("")

        lines.append("Target major counts by origin")
        target_counts = (
            events[events["target_major"].notna()]
            .groupby(["target_major", "event_origin_category"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["target_major", "event_origin_category"])
        )
        for row in target_counts.itertuples(index=False):
            lines.append(f"- {row.target_major} | {row.event_origin_category}: {int(row.n)}")
        lines.append("")

    if not candidate_audit.empty:
        lines.append("Candidate major counts for unverified external rows")
        candidate_major_counts = (
            candidate_audit[candidate_audit["external_verified"] != 1]
            .groupby("candidate_major", dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["candidate_major"])
        )
        for row in candidate_major_counts.itertuples(index=False):
            lines.append(f"- {row.candidate_major}: {int(row.n)}")
        lines.append("")

        lines.append("Candidate audit appendix")
        audit_sorted = candidate_audit.sort_values(
            ["external_verified", "candidate_school_name", "candidate_approx_year", "candidate_id"],
            ascending=[False, True, True, True],
        )
        for row in audit_sorted.itertuples(index=False):
            lines.append(
                " | ".join(
                    [
                        str(row.candidate_id),
                        str(row.candidate_school_name),
                        str(row.candidate_approx_year),
                        str(row.candidate_degree_label),
                        row.candidate_program_desc,
                        f"pair_bin={row.candidate_pair_bin}",
                        f"source_bin={row.candidate_source_cip_bin}",
                        f"target_bin={row.candidate_target_cip_bin}",
                        f"school_match={row.school_match_method}",
                        f"matched_unitid={row.matched_unitid}",
                        f"verified={int(row.external_verified)}",
                        f"best_year={row.best_nearby_year}",
                        f"best_pair={row.best_nearby_source_cip6}->{row.best_nearby_target_cip6}",
                        f"note={row.verification_notes}",
                    ]
                )
            )
    out_path.write_text("\n".join(lines) + "\n")


def run_degree_plots(
    con: ddb.DuckDBPyConnection,
    events: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
    yvars: list[str] | None = None,
    foia_path: str | Path = base.FOIA_PATH,
    inst_cw_path: str | Path = base.F1_INST_CW_PATH,
    foia_person_panel_path: str | Path = DEFAULT_FOIA_PERSON_PANEL_PATH,
    employer_match_dir: str | Path = DEFAULT_EMPLOYER_MATCH_DIR,
    ipeds_path: str | Path = base.IPEDS_PATH,
    crosswalk_path: str | Path = v2.CROSSWALK_PATH,
    did_spec: str = DEFAULT_DID_SPEC,
    estimator: str = DEFAULT_ESTIMATOR,
    include_degree_specific_plots: bool = DEFAULT_INCLUDE_DEGREE_SPECIFIC_PLOTS,
    degree_did_panel_parquet: str | Path | None = None,
) -> list[Path]:
    estimator = _normalize_estimator(estimator)
    run_did = estimator in {ESTIMATOR_DID, ESTIMATOR_BOTH}
    run_stacked = estimator in {ESTIMATOR_STACKED_TREATED, ESTIMATOR_BOTH}
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    yvars = yvars or DEFAULT_YVARS
    events = _filter_relabel_analysis_window(events)
    panel = _filter_relabel_analysis_window(panel)
    outputs: list[Path] = []
    _progress(f"Generating plots in {plots_dir}")
    broad_bin_path = plot_broad_bin_event_counts_by_year(panel, out_dir=plots_dir)
    if broad_bin_path is not None:
        outputs.append(broad_bin_path)
    breakdown_path = plot_relabel_broad_bin_degree_year_breakdown(events, out_dir=plots_dir)
    if breakdown_path is not None:
        outputs.append(breakdown_path)
    sample_path = write_broad_bin_treated_control_school_samples(
        con,
        panel,
        out_dir=plots_dir,
        ipeds_path=ipeds_path,
        crosswalk_path=crosswalk_path,
        sample_size=RELABEL_BROAD_BIN_SAMPLE_N,
        seed=RELABEL_BROAD_BIN_SAMPLE_SEED,
    )
    if sample_path is not None:
        outputs.append(sample_path)
    source_target_path = plot_source_target_ctotalt_event_time_by_degree_treated_control(
        con,
        panel,
        out_dir=plots_dir,
        ipeds_path=ipeds_path,
    )
    if source_target_path is not None:
        outputs.append(source_target_path)
    degree_did_panels: dict[str, pd.DataFrame] = {}
    degree_ipeds_did_panels: dict[str, pd.DataFrame] = {}
    if include_degree_specific_plots:
        for degree_type in ("Bachelor", "Master", "Doctor"):
            degree_events = panel[
                (panel["degree_type"] == degree_type)
                & panel["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"])
                & panel["event_flag"].eq(1)
                & panel["broad_bin_eligible"].eq(1)
            ]
            _progress(f"Plotting degree bucket {degree_type}: {len(degree_events):,} verified event(s)")
            hist_path = plot_relabel_year_histogram(degree_events, degree_type=degree_type, out_dir=plots_dir)
            if hist_path is not None:
                outputs.append(hist_path)
            if degree_events.empty:
                _progress(f"Skipping {degree_type} outcome plots because there are no verified events")
                continue
            opt_usage = compute_opt_usage_generalized(
                con,
                panel,
                degree_type=degree_type,
                foia_path=foia_path,
                inst_cw_path=inst_cw_path,
                foia_person_panel_path=foia_person_panel_path,
                employer_match_dir=employer_match_dir,
            )
            if opt_usage.empty:
                _progress(f"Skipping {degree_type} calendar-year plots because FOIA/IPEDS join returned no rows")
            did_panel = compute_generalized_did_panel(
                con,
                panel,
                degree_type=degree_type,
                did_spec=did_spec,
                foia_path=foia_path,
                inst_cw_path=inst_cw_path,
                foia_person_panel_path=foia_person_panel_path,
                employer_match_dir=employer_match_dir,
                ipeds_path=ipeds_path,
            )
            if did_panel.empty:
                _progress(f"Skipping {degree_type} event-time plots because the matched DiD panel is empty")
            else:
                degree_did_panels[degree_type] = did_panel.copy()
            ipeds_did_panel = compute_generalized_ipeds_did_panel(
                con,
                panel,
                degree_type=degree_type,
                did_spec=did_spec,
                ipeds_path=ipeds_path,
            )
            if not ipeds_did_panel.empty:
                degree_ipeds_did_panels[degree_type] = ipeds_did_panel.copy()
            for yvar in yvars:
                if not opt_usage.empty:
                    usage_path = plot_opt_usage_generalized(opt_usage, yvar=yvar, degree_type=degree_type, out_dir=plots_dir)
                    if usage_path is not None:
                        outputs.append(usage_path)
                plot_panel = ipeds_did_panel if yvar in PROGRAM_LEVEL_IPEDS_YVARS else did_panel
                if not plot_panel.empty:
                    treated_path, control_path = plot_event_time_with_control_generalized(
                        plot_panel,
                        yvar=yvar,
                        degree_type=degree_type,
                        out_dir=plots_dir,
                    )
                    if treated_path is not None:
                        outputs.append(treated_path)
                    if control_path is not None:
                        outputs.append(control_path)
                if not plot_panel.empty and run_did:
                    did_event_study = compute_did_event_study_generalized(
                        did_panel=plot_panel,
                        yvar=yvar,
                        did_spec=did_spec,
                        reference_event_time=DI_D_REFERENCE_EVENT_TIME,
                        use_weights=False,
                    )
                    summary_text = build_did_summary_text(plot_panel, yvar=yvar, did_spec=did_spec)
                    did_path = plot_did_event_study_generalized(
                        did_event_study,
                        yvar=yvar,
                        degree_type=degree_type,
                        out_dir=plots_dir,
                        reference_event_time=DI_D_REFERENCE_EVENT_TIME,
                        summary_text=summary_text,
                    )
                    if did_path is not None:
                        outputs.append(did_path)
                if not plot_panel.empty and run_stacked:
                    stacked_event_study = compute_stacked_event_study_generalized(
                        did_panel=plot_panel,
                        yvar=yvar,
                        did_spec=did_spec,
                        reference_event_time=DI_D_REFERENCE_EVENT_TIME,
                        use_weights=False,
                    )
                    stacked_path = plot_did_event_study_generalized(
                        stacked_event_study,
                        yvar=yvar,
                        degree_type=degree_type,
                        out_dir=plots_dir,
                        file_stem=f"{_slugify(degree_type)}_{yvar}_stacked_treated_event_time",
                        title="Treated-only stacked event study",
                        ylabel="Event-study coefficient",
                        reference_event_time=DI_D_REFERENCE_EVENT_TIME,
                        summary_text=None,
                    )
                    if stacked_path is not None:
                        outputs.append(stacked_path)
    else:
        _progress("Skipping degree-specific plots, but retaining pooled degree-level DiD appendix")
        for degree_type in ("Bachelor", "Master", "Doctor"):
            _progress(f"Building {degree_type} FOIA matched DiD panel for pooled degree-level appendix")
            did_panel = compute_generalized_did_panel(
                con,
                panel,
                degree_type=degree_type,
                did_spec=did_spec,
                foia_path=foia_path,
                inst_cw_path=inst_cw_path,
                foia_person_panel_path=foia_person_panel_path,
                employer_match_dir=employer_match_dir,
                ipeds_path=ipeds_path,
            )
            if did_panel.empty:
                _progress(f"Skipping {degree_type} degree-level appendix panel because the matched DiD panel is empty")
            else:
                degree_did_panels[degree_type] = did_panel.copy()
                _progress(f"Built {degree_type} FOIA matched DiD panel with {len(did_panel):,} row(s)")
            _progress(f"Building {degree_type} IPEDS matched DiD panel for pooled degree-level appendix")
            ipeds_did_panel = compute_generalized_ipeds_did_panel(
                con,
                panel,
                degree_type=degree_type,
                did_spec=did_spec,
                ipeds_path=ipeds_path,
            )
            if not ipeds_did_panel.empty:
                degree_ipeds_did_panels[degree_type] = ipeds_did_panel.copy()
                _progress(f"Built {degree_type} IPEDS matched DiD panel with {len(ipeds_did_panel):,} row(s)")
            else:
                _progress(f"Skipping {degree_type} IPEDS degree-level appendix panel because the matched DiD panel is empty")
    if degree_did_panel_parquet is not None and degree_did_panels:
        degree_did_panel_path = Path(degree_did_panel_parquet)
        degree_did_panel_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(degree_did_panels.values(), ignore_index=True, sort=False).to_parquet(
            degree_did_panel_path,
            index=False,
        )
        _progress(f"Wrote degree-level FOIA matched DiD panel cache to {degree_did_panel_path}")
    _progress("Building pooled FOIA matched DiD panel")
    pooled_did_panel = compute_generalized_did_panel(
        con,
        panel,
        degree_type=None,
        did_spec=did_spec,
        foia_path=foia_path,
        inst_cw_path=inst_cw_path,
        foia_person_panel_path=foia_person_panel_path,
        employer_match_dir=employer_match_dir,
        ipeds_path=ipeds_path,
    )
    _progress(f"Built pooled FOIA matched DiD panel with {len(pooled_did_panel):,} row(s)")
    _progress("Building pooled IPEDS matched DiD panel")
    pooled_ipeds_did_panel = compute_generalized_ipeds_did_panel(
        con,
        panel,
        degree_type=None,
        did_spec=did_spec,
        ipeds_path=ipeds_path,
    )
    _progress(f"Built pooled IPEDS matched DiD panel with {len(pooled_ipeds_did_panel):,} row(s)")
    if pooled_did_panel.empty and pooled_ipeds_did_panel.empty:
        _progress("Skipping pooled FOIA plots because the matched DiD panel is empty")
    else:
        if run_did:
            _progress("Writing pooled grouped DiD appendices")
            appendix_paths = write_grouped_did_appendices(
                pooled_did_panel,
                degree_level_panels=degree_did_panels,
                ipeds_did_panel=pooled_ipeds_did_panel,
                ipeds_degree_level_panels=degree_ipeds_did_panels,
                out_dir=plots_dir,
                did_spec=did_spec,
                include_degree_level_plots=bool(degree_did_panels),
            )
            if appendix_paths:
                outputs.extend(appendix_paths)
                _progress(f"Generated pooled DiD appendices with {sum(path.suffix == '.png' for path in appendix_paths)} plot(s)")
        for yvar in yvars:
            plot_panel = pooled_ipeds_did_panel if yvar in PROGRAM_LEVEL_IPEDS_YVARS else pooled_did_panel
            if plot_panel.empty:
                _progress(f"Skipping pooled {yvar} raw/DID plots because the panel is empty")
                continue
            panel_label = "IPEDS" if yvar in PROGRAM_LEVEL_IPEDS_YVARS else "FOIA"
            _progress(f"Plotting pooled {panel_label} raw means for {yvar}")
            treated_path, control_path = plot_event_time_with_control_generalized(
                plot_panel,
                yvar=yvar,
                degree_type="Pooled",
                out_dir=plots_dir,
            )
            if treated_path is not None:
                outputs.append(treated_path)
            if control_path is not None:
                outputs.append(control_path)
            if run_did:
                _progress(f"Estimating pooled {panel_label} DiD event study for {yvar}")
                pooled_did_event_study = compute_did_event_study_generalized(
                    did_panel=plot_panel,
                    yvar=yvar,
                    did_spec=did_spec,
                    reference_event_time=DI_D_REFERENCE_EVENT_TIME,
                    use_weights=False,
                )
                pooled_summary_text = build_did_summary_text(plot_panel, yvar=yvar, did_spec=did_spec)
                pooled_did_path = plot_did_event_study_generalized(
                    pooled_did_event_study,
                    yvar=yvar,
                    degree_type="Pooled",
                    out_dir=plots_dir,
                    reference_event_time=DI_D_REFERENCE_EVENT_TIME,
                    summary_text=pooled_summary_text,
                )
                if pooled_did_path is not None:
                    outputs.append(pooled_did_path)
                    _progress(f"Wrote pooled {panel_label} DiD plot for {yvar} to {pooled_did_path}")
            if run_stacked:
                _progress(f"Estimating pooled {panel_label} stacked event study for {yvar}")
                pooled_stacked_event_study = compute_stacked_event_study_generalized(
                    did_panel=plot_panel,
                    yvar=yvar,
                    did_spec=did_spec,
                    reference_event_time=DI_D_REFERENCE_EVENT_TIME,
                    use_weights=False,
                )
                pooled_stacked_path = plot_did_event_study_generalized(
                    pooled_stacked_event_study,
                    yvar=yvar,
                    degree_type="Pooled",
                    out_dir=plots_dir,
                    file_stem=f"pooled_{yvar}_stacked_treated_event_time",
                    title="Pooled sample | Treated-only stacked event study",
                    ylabel="Event-study coefficient",
                    reference_event_time=DI_D_REFERENCE_EVENT_TIME,
                    summary_text=None,
                )
                if pooled_stacked_path is not None:
                    outputs.append(pooled_stacked_path)
    _progress(f"Generated {len(outputs):,} plot file(s)")
    return outputs


def _enrich_event_labels(events: pd.DataFrame, cip_map: dict[str, str]) -> pd.DataFrame:
    if events.empty:
        return events
    out = events.copy()
    out["source_cip_label"] = out["event_source_cip6"].map(cip_map)
    out["target_cip_label"] = out["target_cip6"].map(cip_map)
    out["source_major"] = out["source_cip_label"].map(_clean_cip_label)
    out["target_major"] = out["target_cip_label"].map(_clean_cip_label)
    return out


def run_pipeline(
    *,
    candidate_path: str | Path | None = DEFAULT_CANDIDATE_PATH,
    ipeds_path: str | Path = base.IPEDS_PATH,
    crosswalk_path: str | Path = v2.CROSSWALK_PATH,
    foia_path: str | Path = base.FOIA_PATH,
    inst_cw_path: str | Path = base.F1_INST_CW_PATH,
    foia_person_panel_path: str | Path = DEFAULT_FOIA_PERSON_PANEL_PATH,
    employer_match_dir: str | Path = DEFAULT_EMPLOYER_MATCH_DIR,
    events_parquet: str | Path = DEFAULT_EVENTS_PARQUET,
    events_csv: str | Path = DEFAULT_EVENTS_CSV,
    panel_parquet: str | Path = DEFAULT_PANEL_PARQUET,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    candidate_audit_csv: str | Path = DEFAULT_CANDIDATE_AUDIT_CSV,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
    relabel_year_mode: str = DEFAULT_RELABEL_YEAR_MODE,
    did_spec: str = DEFAULT_DID_SPEC,
    estimator: str = DEFAULT_ESTIMATOR,
    include_degree_specific_plots: bool = DEFAULT_INCLUDE_DEGREE_SPECIFIC_PLOTS,
    yvars: list[str] | None = None,
    degree_did_panel_parquet: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    did_spec = _normalize_did_spec(did_spec)
    estimator = _normalize_estimator(estimator)
    _progress("Starting generalized relabel pipeline")
    _progress(f"IPEDS source: {ipeds_path}")
    _progress(f"Crosswalk source: {crosswalk_path}")
    _progress(f"FOIA source: {foia_path}")
    _progress(f"FOIA person panel source: {foia_person_panel_path}")
    _progress(f"Employer match directory: {employer_match_dir}")
    _progress(f"Candidate source: {candidate_path}")
    _progress(f"DiD spec: {did_spec}")
    _progress(f"Event-study estimator: {estimator}")
    _progress(f"Include degree-specific plots: {include_degree_specific_plots}")
    con = ddb.connect()
    cip_map = _load_ipeds_cip_map(ipeds_path)
    _progress(f"Loaded {len(cip_map):,} CIP labels from IPEDS")
    candidates = pd.DataFrame()
    allowable_pair_configs: list[dict[str, object]] = []
    if candidate_path is not None:
        candidates = load_external_candidates(candidate_path)
        raw_candidate_count = len(candidates)
        candidates = _exclude_disallowed_candidate_rows(candidates)
        excluded_candidate_count = raw_candidate_count - len(candidates)
        if excluded_candidate_count > 0:
            _progress(
                "Excluded "
                f"{excluded_candidate_count:,} external candidate row(s) in disallowed broad-bin families: "
                f"{', '.join(sorted(EXCLUDED_BROAD_PAIR_BINS))}"
            )
        allowable_pair_configs = derive_allowable_pair_configs(candidates)
        _progress(
            "Derived "
            f"{len(allowable_pair_configs):,} allowable source->target CIP configuration(s) "
            f"from {len(candidates):,} external candidate row(s)"
        )
    else:
        _progress("No external candidate input provided; strict IPEDS scan will remain unrestricted")
    _progress("Running strict IPEDS relabel scan")
    strict_events = _enrich_event_labels(
        detect_ipeds_relabels(
            con,
            ipeds_path=ipeds_path,
            thresholds=STRICT_THRESHOLDS,
            allowed_pair_configs=allowable_pair_configs,
            relabel_year_mode=relabel_year_mode,
        ),
        cip_map,
    )
    raw_strict_event_count = len(strict_events)
    strict_events = _exclude_disallowed_event_rows(strict_events)
    excluded_strict_event_count = raw_strict_event_count - len(strict_events)
    if excluded_strict_event_count > 0:
        _progress(
            "Excluded "
            f"{excluded_strict_event_count:,} strict IPEDS event row(s) in disallowed broad-bin families: "
            f"{', '.join(sorted(EXCLUDED_BROAD_PAIR_BINS))}"
        )
    _progress(f"Strict IPEDS scan found {len(strict_events):,} event(s)")
    candidate_audit = pd.DataFrame()
    verified_external = pd.DataFrame()
    if candidate_path is not None:
        _progress("Loading school lookup for candidate-school resolution")
        school_lookup = load_school_lookup(crosswalk_path)
        resolved_candidates = resolve_candidate_schools(candidates, school_lookup)
        verified_external, candidate_audit = verify_external_candidates(
            con,
            resolved_candidates,
            ipeds_path=ipeds_path,
            relaxed_thresholds=STRICT_THRESHOLDS,
            cip_map=cip_map,
        )
        candidate_audit = _annotate_candidate_audit_broad_bins(candidate_audit)
        if not verified_external.empty:
            verified_external = _enrich_event_labels(verified_external, cip_map)
            raw_verified_event_count = len(verified_external)
            verified_external = _exclude_disallowed_event_rows(verified_external)
            excluded_verified_event_count = raw_verified_event_count - len(verified_external)
            if excluded_verified_event_count > 0:
                _progress(
                    "Excluded "
                    f"{excluded_verified_event_count:,} externally verified event row(s) in disallowed broad-bin families: "
                    f"{', '.join(sorted(EXCLUDED_BROAD_PAIR_BINS))}"
                )
        _progress(
            f"External candidate stage produced {len(verified_external):,} verified event(s) "
            f"from {len(candidate_audit):,} audited candidate row(s)"
        )
    else:
        _progress("No external candidate input provided; skipping candidate verification stage")

    _progress("Merging IPEDS-driven and externally driven event sources")
    merged_events = merge_event_sources(strict_events, verified_external, candidate_audit, cip_map=cip_map)
    merged_events = _exclude_disallowed_event_rows(merged_events)
    merged_events = _coerce_verified_event_output_dtypes(merged_events)
    merged_events = _filter_relabel_analysis_window(merged_events)
    _progress(f"Merged event table contains {len(merged_events):,} row(s)")
    verified_panel = build_verified_event_panel(
        con,
        merged_events,
        ipeds_path=ipeds_path,
        relabel_year_mode=relabel_year_mode,
    )
    verified_panel = _coerce_verified_event_output_dtypes(verified_panel)
    treated_event_count = int(verified_panel["event_flag"].sum()) if not verified_panel.empty else 0
    _progress(f"Broad-bin treated panel contains {treated_event_count:,} treated event row(s)")

    events_parquet = Path(events_parquet)
    events_csv = Path(events_csv)
    panel_parquet = Path(panel_parquet)
    candidate_audit_csv = Path(candidate_audit_csv)
    for path in (events_parquet, events_csv, panel_parquet, report_path, candidate_audit_csv, plots_dir):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    _progress("Writing event, panel, audit, and report outputs")
    merged_events.to_parquet(events_parquet, index=False)
    merged_events.to_csv(events_csv, index=False)
    verified_panel.to_parquet(panel_parquet, index=False)
    if not candidate_audit.empty:
        candidate_audit.to_csv(candidate_audit_csv, index=False)
        _progress(f"Wrote candidate audit CSV to {candidate_audit_csv}")
    _progress(f"Wrote merged events parquet to {events_parquet}")
    _progress(f"Wrote merged events CSV to {events_csv}")
    _progress(f"Wrote verified panel parquet to {panel_parquet}")
    plot_outputs: list[Path] = []
    try:
        plot_outputs = run_degree_plots(
            con,
            merged_events,
            verified_panel,
            plots_dir=plots_dir,
            foia_path=foia_path,
            inst_cw_path=inst_cw_path,
            foia_person_panel_path=foia_person_panel_path,
            employer_match_dir=employer_match_dir,
            ipeds_path=ipeds_path,
            crosswalk_path=crosswalk_path,
            did_spec=did_spec,
            estimator=estimator,
            include_degree_specific_plots=include_degree_specific_plots,
            yvars=yvars,
            degree_did_panel_parquet=degree_did_panel_parquet,
        )
    except Exception as exc:
        print(f"Warning: plot generation failed ({exc})")
    plot_appendices = [path for path in plot_outputs if path.name.endswith("_appendix.md")]
    _progress(f"Writing report to {report_path}")
    write_generalized_report(
        merged_events,
        candidate_audit,
        report_path,
        plot_appendices=plot_appendices,
    )
    _progress("Finished generalized relabel pipeline")
    return {
        "events": merged_events,
        "panel": verified_panel,
        "candidate_audit": candidate_audit,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generalized relabel detector for relabels_revelio.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--candidate-path",
        type=str,
        default=str(DEFAULT_CANDIDATE_PATH),
        help=f"Messy external candidate list (csv/xlsx/parquet or directory). Default: {DEFAULT_CANDIDATE_PATH}",
    )
    parser.add_argument("--ipeds-path", type=str, default=base.IPEDS_PATH, help="IPEDS completions parquet.")
    parser.add_argument("--crosswalk-path", type=str, default=v2.CROSSWALK_PATH, help="IPEDS crosswalk parquet.")
    parser.add_argument("--foia-path", type=str, default=base.FOIA_PATH, help="FOIA parquet for outcome plots.")
    parser.add_argument("--inst-cw-path", type=str, default=base.F1_INST_CW_PATH, help="FOIA institution crosswalk parquet.")
    parser.add_argument(
        "--foia-person-panel-path",
        type=str,
        default=str(DEFAULT_FOIA_PERSON_PANEL_PATH),
        help="Corrected FOIA person panel parquet used for employer-history enrichment.",
    )
    parser.add_argument(
        "--employer-match-dir",
        type=str,
        default=str(DEFAULT_EMPLOYER_MATCH_DIR),
        help="Directory containing foia_row_entities and foia_row_to_firm artifacts.",
    )
    parser.add_argument("--events-parquet", type=str, default=str(DEFAULT_EVENTS_PARQUET))
    parser.add_argument("--events-csv", type=str, default=str(DEFAULT_EVENTS_CSV))
    parser.add_argument("--panel-parquet", type=str, default=str(DEFAULT_PANEL_PARQUET))
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--candidate-audit-csv", type=str, default=str(DEFAULT_CANDIDATE_AUDIT_CSV))
    parser.add_argument("--plots-dir", type=str, default=str(DEFAULT_PLOTS_DIR))
    parser.add_argument(
        "--did-spec",
        type=str,
        choices=VALID_DID_SPECS,
        default=DEFAULT_DID_SPEC,
        help=(
            "DiD design for raw means and regressions. "
            "'collapsed_unit_fe' keeps the existing aggregated school-year panel. "
            "'individual_broad_bin_degree_fe' runs on individual rows and uses "
            "unitid x broad bin x degree type fixed effects where available."
        ),
    )
    parser.add_argument(
        "--estimator",
        type=str,
        choices=VALID_ESTIMATORS,
        default=DEFAULT_ESTIMATOR,
        help=(
            "Event-study estimator for coefficient plots: 'did' uses treated vs matched controls; "
            "'stacked_treated' excludes controls and estimates a treated-only stacked event study; "
            "'both' writes both sets of coefficient plots."
        ),
    )
    parser.add_argument(
        "--relabel-year-mode",
        type=str,
        choices=("first", "largest"),
        default=DEFAULT_RELABEL_YEAR_MODE,
        help=(
            "For repeated source->target relabels at same institution/degree over multiple years, "
            "choose either first (earliest year) or largest (highest relabel_score)."
        ),
    )
    parser.add_argument(
        "--include-degree-specific-plots",
        action="store_true",
        help=(
            "Generate the slower bachelor/master/doctor-specific plot bundles. "
            "The pooled degree-level DiD appendix is still generated by default."
        ),
    )
    if "ipykernel" in sys.modules:
        args, unknown = parser.parse_known_args()
        if unknown:
            _progress(f"Ignoring notebook/kernel CLI args: {' '.join(map(str, unknown))}")
        return args
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_pipeline(
        candidate_path=args.candidate_path,
        ipeds_path=args.ipeds_path,
        crosswalk_path=args.crosswalk_path,
        foia_path=args.foia_path,
        inst_cw_path=args.inst_cw_path,
        foia_person_panel_path=args.foia_person_panel_path,
        employer_match_dir=args.employer_match_dir,
        events_parquet=args.events_parquet,
        events_csv=args.events_csv,
        panel_parquet=args.panel_parquet,
        report_path=args.report_path,
        candidate_audit_csv=args.candidate_audit_csv,
        plots_dir=args.plots_dir,
        relabel_year_mode=args.relabel_year_mode,
        did_spec=args.did_spec,
        estimator=args.estimator,
        include_degree_specific_plots=args.include_degree_specific_plots,
    )
    print(f"Wrote {len(outputs['events']):,} merged events")
    print(f"Wrote {len(outputs['panel']):,} verified panel rows")
    if not outputs["candidate_audit"].empty:
        verified_count = int((outputs["candidate_audit"]["external_verified"] == 1).sum())
        print(f"Audited {len(outputs['candidate_audit']):,} external candidates; verified {verified_count:,}")


if __name__ == "__main__":
    main()
