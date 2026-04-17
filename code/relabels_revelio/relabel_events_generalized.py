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
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb as ddb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


BASE_FONT_SIZE = base.BASE_FONT_SIZE
PLOT_EVENT_MIN = -5
PLOT_EVENT_MAX = 4
DEFAULT_OUTPUT_DIR = Path(base.root) / "h1bworkers" / "code" / "output" / "relabel_indiv"
DEFAULT_CANDIDATE_PATH = Path(base.root) / "data" / "llm_relabel_candidates"
DEFAULT_EVENTS_PARQUET = DEFAULT_OUTPUT_DIR / "generalized_relabels_events.parquet"
DEFAULT_EVENTS_CSV = DEFAULT_OUTPUT_DIR / "generalized_relabels_events.csv"
DEFAULT_PANEL_PARQUET = DEFAULT_OUTPUT_DIR / "generalized_relabels_panel.parquet"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "generalized_relabels_report.txt"
DEFAULT_CANDIDATE_AUDIT_CSV = DEFAULT_OUTPUT_DIR / "generalized_relabels_candidate_audit.csv"
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "generalized_relabels_plots"

STRICT_THRESHOLDS: dict[str, float] = {
    "min_share_intl": 0.20,
    "min_source_baseline": float(v2.MIN_SOURCE_BASELINE),
    "min_source_drop_abs": float(v2.MIN_SOURCE_DROP_ABS),
    "min_source_drop_pct": float(v2.MIN_SOURCE_DROP_PCT),
    "min_target_offset_share": float(v2.MIN_TARGET_OFFSET_SHARE),
    "max_net_loss_share": float(v2.MAX_NET_LOSS_SHARE),
    "source_persistence_drop_share": float(v2.SOURCE_PERSISTENCE_DROP_PCT),
    "target_persistence_gain_share": float(v2.TARGET_PERSISTENCE_GAIN_SHARE),
    "lookback_years": float(v2.LOOKBACK_YEARS),
    "lookahead_years": float(v2.LOOKAHEAD_YEARS),
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
    "opt_share",
    "opt_stem_share",
    "status_change_share",
    "opt_years_avg",
    "f1_share_of_ctotalt",
    "f1_share_of_cnralt",
    "avg_tuition",
]

VERIFIED_EVENT_COLUMNS = [
    "unitid",
    "awlevel",
    "degree_type",
    "relabel_year",
    "year",
    "relabel_type",
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
    elif "finance" in program_norm or "mfin" in program_norm:
        source_rule = _business_source_rule(program_norm, source_hint)
        target_rule = _prefer_exact_rule(
            target_hint,
            _cip_rule("quantitative_finance_target_family", include_exact=("270305", "270501", "521301", "521399")),
        )
        pair_bin = "finance_to_quantitative_finance_family"
        parse_notes = "hand-parsed finance to quantitative-finance family"

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


def _ensure_ipeds_view(con: ddb.DuckDBPyConnection, ipeds_path: str | Path) -> None:
    ipeds_path = Path(ipeds_path)
    if not ipeds_path.exists():
        raise FileNotFoundError(f"IPEDS completions parquet not found: {ipeds_path}")
    con.sql(f"CREATE OR REPLACE TEMP VIEW ipeds_raw AS SELECT * FROM read_parquet('{_sql_literal(str(ipeds_path))}')")


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
) -> pd.DataFrame:
    thresholds = thresholds or STRICT_THRESHOLDS
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
) -> pd.DataFrame:
    events = scan_ipeds_pair_candidates(
        con,
        ipeds_path=ipeds_path,
        thresholds=thresholds or STRICT_THRESHOLDS,
        keep_all_sources=False,
    )
    if events.empty:
        return events
    events = events.sort_values(
        ["unitid", "awlevel", "source_cip6", "target_cip6", "year", "relabel_score"],
        ascending=[True, True, True, True, True, False],
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
    _progress(f"Starting relaxed IPEDS verification for {total_candidates:,} external candidate(s)")

    for idx, row in enumerate(resolved_candidates.itertuples(index=False), start=1):
        unitid = None if pd.isna(row.matched_unitid) else int(row.matched_unitid)
        degree_type = str(row.candidate_degree_type)
        awlevels = awlevels_for_degree_type(degree_type)
        approx_year = None if pd.isna(row.candidate_approx_year) else int(row.candidate_approx_year)
        year_min = approx_year - 3 if approx_year is not None else None
        year_max = approx_year + 3 if approx_year is not None else None
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
        if nearby_pairs.empty:
            audit["verification_notes"] = "no_relaxed_ipeds_match"
            audit.update(
                _diagnostic_best_match(
                    con,
                    unitid=unitid,
                    awlevels=awlevels,
                    year_min=year_min if year_min is not None else 0,
                    year_max=year_max if year_max is not None else 9999,
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
                    year_max=year_max if year_max is not None else 9999,
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
        return merged
    merged["event_origin_category"] = merged.apply(_apply_event_origin_category, axis=1)
    merged["source_cip_label"] = merged["event_source_cip6"].map(cip_map).fillna(merged.get("source_cip_label"))
    merged["target_cip_label"] = merged["target_cip6"].map(cip_map).fillna(merged.get("target_cip_label"))
    merged["source_major"] = merged["source_cip_label"].map(lambda value: _clean_cip_label(value) if pd.notna(value) else value)
    merged["target_major"] = merged["target_cip_label"].map(lambda value: _clean_cip_label(value) if pd.notna(value) else value)
    merged["degree_type"] = merged.apply(
        lambda row: row["degree_type"] if pd.notna(row["degree_type"]) and str(row["degree_type"]) != "nan" else degree_type_for_awlevel(row["awlevel"]),
        axis=1,
    )
    merged["year"] = merged["year"].fillna(merged["relabel_year"])
    merged["event_flag"] = merged["event_origin_category"].ne("external_only").astype(int)
    merged["relabel_flag"] = merged["event_flag"]
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
) -> pd.DataFrame:
    verified_events = events[
        events["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"])
        & events["relabel_year"].notna()
        & events["event_source_cip6"].notna()
        & events["target_cip6"].notna()
        & events["awlevel"].notna()
    ].copy()
    if verified_events.empty:
        _progress("No verified events available for panel construction")
        return pd.DataFrame(columns=VERIFIED_EVENT_COLUMNS)

    _progress(f"Building verified event panel for {len(verified_events):,} event(s)")
    _ensure_ipeds_view(con, ipeds_path)
    min_year, max_year = _ipeds_year_bounds(con)
    event_subset = verified_events[
        ["unitid", "awlevel", "event_source_cip6", "target_cip6"]
    ].drop_duplicates()
    con.register("verified_events_panel_keys_py", event_subset)
    ipeds_subset = con.sql(
        """
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
         AND (
                LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = e.event_source_cip6
             OR LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = e.target_cip6
         )
        """
    ).df()

    years = pd.Index(range(min_year, max_year + 1), name="year")
    panel_rows: list[pd.DataFrame] = []
    for event in verified_events.itertuples(index=False):
        ev = ipeds_subset[
            (ipeds_subset["unitid"] == int(event.unitid))
            & (ipeds_subset["awlevel"] == int(event.awlevel))
            & (ipeds_subset["cip6"].isin([str(event.event_source_cip6), str(event.target_cip6)]))
        ].copy()
        if ev.empty:
            source_year = pd.Series(0.0, index=years)
            target_year = pd.Series(0.0, index=years)
            cnr_source = pd.Series(0.0, index=years)
            cnr_target = pd.Series(0.0, index=years)
        else:
            grouped = ev.groupby(["year", "cip6"], as_index=False).agg({"ctotalt": "sum", "cnralt": "sum"})
            source_year = (
                grouped[grouped["cip6"] == str(event.event_source_cip6)]
                .set_index("year")["ctotalt"]
                .reindex(years, fill_value=0.0)
            )
            target_year = (
                grouped[grouped["cip6"] == str(event.target_cip6)]
                .set_index("year")["ctotalt"]
                .reindex(years, fill_value=0.0)
            )
            cnr_source = (
                grouped[grouped["cip6"] == str(event.event_source_cip6)]
                .set_index("year")["cnralt"]
                .reindex(years, fill_value=0.0)
            )
            cnr_target = (
                grouped[grouped["cip6"] == str(event.target_cip6)]
                .set_index("year")["cnralt"]
                .reindex(years, fill_value=0.0)
            )

        panel = pd.DataFrame({"year": years})
        panel["unitid"] = int(event.unitid)
        panel["awlevel"] = int(event.awlevel)
        panel["degree_type"] = event.degree_type
        panel["relabel_year"] = int(event.relabel_year)
        panel["relabel_type"] = event.relabel_type
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
        for column in VERIFIED_EVENT_COLUMNS:
            if column not in panel.columns and hasattr(event, column):
                panel[column] = getattr(event, column)
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
        panel_rows.append(panel.loc[:, VERIFIED_EVENT_COLUMNS])

    out = pd.concat(panel_rows, ignore_index=True)
    out = out.sort_values(["degree_type", "unitid", "relabel_year", "year", "event_source_cip6", "target_cip6"]).reset_index(drop=True)
    _progress(f"Built verified event panel with {len(out):,} row(s)")
    return out


def _foia_degree_case(column_name: str = "degree_type") -> str:
    parts = []
    for degree_type, foia_label in DEGREE_TYPE_TO_FOIA_LABEL.items():
        parts.append(f"WHEN {column_name} = '{_sql_literal(degree_type)}' THEN '{_sql_literal(foia_label)}'")
    return "CASE " + " ".join(parts) + " ELSE NULL END"


def _load_foia_base(
    con: ddb.DuckDBPyConnection,
    *,
    foia_path: str | Path = base.FOIA_PATH,
    inst_cw_path: str | Path = base.F1_INST_CW_PATH,
) -> dict[str, str | None]:
    foia_path = Path(foia_path)
    inst_cw_path = Path(inst_cw_path)
    if not foia_path.exists():
        raise FileNotFoundError(f"Missing FOIA parquet: {foia_path}")
    if not inst_cw_path.exists():
        raise FileNotFoundError(f"Missing F-1 institution crosswalk parquet: {inst_cw_path}")
    con.sql(f"CREATE OR REPLACE TEMP VIEW foia_raw AS SELECT * FROM read_parquet('{_sql_literal(str(foia_path))}')")
    con.sql(f"CREATE OR REPLACE TEMP VIEW f1_inst_cw AS SELECT * FROM read_parquet('{_sql_literal(str(inst_cw_path))}')")
    return v2._resolve_foia_schema(con)


def compute_opt_usage_generalized(
    con: ddb.DuckDBPyConnection,
    relabel_panel: pd.DataFrame,
    *,
    degree_type: str,
    foia_path: str | Path = base.FOIA_PATH,
    inst_cw_path: str | Path = base.F1_INST_CW_PATH,
) -> pd.DataFrame:
    panel = relabel_panel[
        (relabel_panel["degree_type"] == degree_type)
        & relabel_panel["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"])
    ].copy()
    if panel.empty:
        return pd.DataFrame()
    schema = _load_foia_base(con, foia_path=foia_path, inst_cw_path=inst_cw_path)

    relabel_events = panel[
        [
            "unitid",
            "year",
            "relabel_year",
            "relabel_type",
            "degree_type",
            "event_source_cip6",
            "target_cip6",
            "ctotalt",
            "cnralt",
        ]
    ].drop_duplicates()
    con.register("generalized_relabel_events_py", relabel_events)

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
    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )
    degree_case = _foia_degree_case("r.degree_type")

    opt_usage = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0') AS cip6,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                employer_name,
                employment_opt_type,
                {opt_end_col} AS opt_end_date,
                {foia_tuition_col} AS tuition,
                {foia_end_col} AS program_end_date,
                {status_col} AS requested_status,
                {foia_edu_col} AS foia_degree_label
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE {foia_end_col} IS NOT NULL
              {year_match_clause}
        ),
        relevant_foia AS (
            SELECT *
            FROM foia_base
            WHERE unitid IS NOT NULL
              AND cip6 IS NOT NULL
              AND grad_year IS NOT NULL
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
             AND (
                    f.cip6 = r.event_source_cip6
                 OR f.cip6 = r.target_cip6
             )
             AND f.foia_degree_label = {degree_case}
        ),
        student_level AS (
            SELECT
                unitid,
                grad_year,
                relabel_year,
                relabel_type,
                degree_type,
                ctotalt,
                cnralt,
                MAX(CASE WHEN employer_name IS NOT NULL THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL
                    THEN DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25
                    ELSE 0
                END AS opt_years,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                student_id
            FROM flagged
            GROUP BY unitid, grad_year, relabel_year, relabel_type, degree_type, ctotalt, cnralt, student_id
        )
        SELECT
            grad_year AS calendar_year,
            relabel_year,
            relabel_type,
            degree_type,
            AVG(avg_tuition) AS avg_tuition,
            COUNT(DISTINCT student_id) AS total_grads,
            COUNT(DISTINCT CASE WHEN opt_ind = 1 THEN student_id END) AS opt_users,
            COUNT(DISTINCT CASE WHEN opt_stem_ind = 1 THEN student_id END) AS opt_stem_users,
            COUNT(DISTINCT CASE WHEN status_change_ind = 1 THEN student_id END) AS status_change_users,
            SUM(opt_years) AS total_opt_years,
            MAX(ctotalt) AS ctotalt,
            MAX(cnralt) AS cnralt
        FROM student_level
        GROUP BY calendar_year, relabel_year, relabel_type, degree_type
        ORDER BY calendar_year, relabel_year, relabel_type
        """
    ).df()

    if opt_usage.empty:
        return opt_usage
    opt_usage["opt_share"] = opt_usage["opt_users"] / opt_usage["total_grads"]
    opt_usage["opt_stem_share"] = opt_usage["opt_stem_users"] / opt_usage["total_grads"]
    opt_usage["status_change_share"] = opt_usage["status_change_users"] / opt_usage["total_grads"]
    opt_usage["opt_years_avg"] = opt_usage["total_opt_years"] / opt_usage["total_grads"]
    opt_usage["f1_share_of_ctotalt"] = opt_usage["total_grads"] / opt_usage["ctotalt"]
    opt_usage["f1_share_of_cnralt"] = opt_usage["total_grads"] / opt_usage["cnralt"]
    opt_usage["tuition_total"] = opt_usage["avg_tuition"] * opt_usage["total_grads"]
    return opt_usage


def compute_opt_usage_event_time_generalized(opt_usage: pd.DataFrame) -> pd.DataFrame:
    if opt_usage.empty:
        return pd.DataFrame()
    df = opt_usage[opt_usage["calendar_year"].between(base.PLOT_YEAR_MIN, base.PLOT_YEAR_MAX)].copy()
    if df.empty:
        return pd.DataFrame()
    df["event_t"] = df["calendar_year"] - df["relabel_year"]
    grouped = (
        df.groupby(["event_t", "relabel_type", "degree_type"], as_index=False)
        .agg(
            total_grads=("total_grads", "sum"),
            opt_users=("opt_users", "sum"),
            opt_stem_users=("opt_stem_users", "sum"),
            total_opt_years=("total_opt_years", "sum"),
            total_status_change_users=("status_change_users", "sum"),
            tuition_total=("tuition_total", "sum"),
            ctotalt=("ctotalt", "sum"),
            cnralt=("cnralt", "sum"),
        )
    )
    grouped["opt_share"] = grouped["opt_users"] / grouped["total_grads"]
    grouped["opt_stem_share"] = grouped["opt_stem_users"] / grouped["total_grads"]
    grouped["opt_years_avg"] = grouped["total_opt_years"] / grouped["total_grads"]
    grouped["status_change_share"] = grouped["total_status_change_users"] / grouped["total_grads"]
    grouped["f1_share_of_ctotalt"] = grouped["total_grads"] / grouped["ctotalt"]
    grouped["f1_share_of_cnralt"] = grouped["total_grads"] / grouped["cnralt"]
    grouped["avg_tuition"] = grouped["tuition_total"] / grouped["total_grads"]
    return grouped


def match_treated_to_never_treated(
    con: ddb.DuckDBPyConnection,
    verified_events: pd.DataFrame,
    *,
    ipeds_path: str | Path = base.IPEDS_PATH,
    min_share_intl: float = STRICT_THRESHOLDS["min_share_intl"],
    use_replacement: bool = True,
) -> pd.DataFrame:
    events = verified_events[
        verified_events["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"])
    ].copy()
    if events.empty:
        return pd.DataFrame()
    _ensure_ipeds_view(con, ipeds_path)

    treated_events = events[
        [
            "unitid",
            "awlevel",
            "degree_type",
            "relabel_year",
            "relabel_type",
            "event_source_cip6",
            "target_cip6",
            "source_total_prev",
        ]
    ].drop_duplicates()
    treated_events["source_total_prev"] = pd.to_numeric(treated_events["source_total_prev"], errors="coerce").fillna(0.0)
    treated_unitids = sorted(pd.to_numeric(treated_events["unitid"], errors="coerce").dropna().astype(int).unique().tolist())
    treated_unitid_clause = ", ".join(str(v) for v in treated_unitids) if treated_unitids else "-1"
    con.register("treated_events_for_controls_py", treated_events)
    candidate_pool = con.sql(
        f"""
        SELECT
            t.relabel_year,
            t.relabel_type,
            t.degree_type,
            t.awlevel,
            t.event_source_cip6,
            t.target_cip6,
            CAST(i.unitid AS BIGINT) AS unitid,
            SUM(CAST(i.ctotalt AS DOUBLE)) AS control_pre_size
        FROM ipeds_raw i
        JOIN treated_events_for_controls_py t
          ON CAST(i.year AS INTEGER) = t.relabel_year - 1
         AND CAST(i.awlevel AS INTEGER) = t.awlevel
         AND LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = t.event_source_cip6
        WHERE CAST(i.unitid AS BIGINT) NOT IN ({treated_unitid_clause})
          AND COALESCE(CAST(i.share_intl AS DOUBLE), 0) >= {float(min_share_intl)}
        GROUP BY t.relabel_year, t.relabel_type, t.degree_type, t.awlevel, t.event_source_cip6, t.target_cip6, CAST(i.unitid AS BIGINT)
        """
    ).df()
    if candidate_pool.empty:
        return pd.DataFrame()

    matches: list[dict[str, object]] = []
    pair_id = 0
    treated_events = treated_events.sort_values(["relabel_year", "awlevel", "event_source_cip6", "unitid"]).reset_index(drop=True)
    available = candidate_pool.copy()
    for row in treated_events.itertuples(index=False):
        pool = available[
            (available["relabel_year"] == int(row.relabel_year))
            & (available["awlevel"] == int(row.awlevel))
            & (available["event_source_cip6"] == str(row.event_source_cip6))
        ].copy()
        if pool.empty:
            continue
        pool["abs_size_diff"] = (pd.to_numeric(pool["control_pre_size"], errors="coerce") - float(row.source_total_prev)).abs()
        best_idx = pool["abs_size_diff"].idxmin()
        chosen = pool.loc[best_idx]
        pair_id += 1
        matches.append(
            {
                "pair_id": pair_id,
                "relabel_year": int(row.relabel_year),
                "relabel_type": row.relabel_type,
                "degree_type": row.degree_type,
                "awlevel": int(row.awlevel),
                "source_cip6": row.event_source_cip6,
                "target_cip6": row.target_cip6,
                "treated_unitid": int(row.unitid),
                "treated_pre_size": float(row.source_total_prev),
                "control_unitid": int(chosen["unitid"]),
                "control_pre_size": float(chosen["control_pre_size"]),
                "abs_size_diff": float(chosen["abs_size_diff"]),
                "match_with_replacement": int(use_replacement),
            }
        )
        if not use_replacement:
            available = available.drop(index=best_idx)
    return pd.DataFrame(matches)


def compute_never_treated_control_event_time_generalized(
    con: ddb.DuckDBPyConnection,
    verified_events: pd.DataFrame,
    *,
    degree_type: str,
    foia_path: str | Path = base.FOIA_PATH,
    inst_cw_path: str | Path = base.F1_INST_CW_PATH,
) -> pd.DataFrame:
    degree_events = verified_events[verified_events["degree_type"] == degree_type].copy()
    matched_pairs = match_treated_to_never_treated(con, degree_events)
    if matched_pairs.empty:
        return pd.DataFrame()
    schema = _load_foia_base(con, foia_path=foia_path, inst_cw_path=inst_cw_path)
    con.register("generalized_control_pairs_py", matched_pairs)

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
    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )
    degree_case = _foia_degree_case("p.degree_type")

    control_calendar = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0') AS cip6,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                employer_name,
                employment_opt_type,
                {opt_end_col} AS opt_end_date,
                {foia_tuition_col} AS tuition,
                {foia_end_col} AS program_end_date,
                {status_col} AS requested_status,
                {foia_edu_col} AS foia_degree_label
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE {foia_end_col} IS NOT NULL
              {year_match_clause}
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
                p.degree_type
            FROM relevant_foia f
            JOIN generalized_control_pairs_py p
              ON f.unitid = p.control_unitid
             AND f.cip6 = p.source_cip6
             AND f.foia_degree_label = {degree_case}
        ),
        student_level AS (
            SELECT
                pair_id,
                grad_year,
                relabel_year,
                relabel_type,
                degree_type,
                MAX(CASE WHEN employer_name IS NOT NULL THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL
                    THEN DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25
                    ELSE 0
                END AS opt_years,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                student_id
            FROM matched_control
            GROUP BY pair_id, grad_year, relabel_year, relabel_type, degree_type, student_id
        ),
        calendar_level AS (
            SELECT
                grad_year AS calendar_year,
                relabel_year,
                relabel_type,
                degree_type,
                AVG(avg_tuition) AS avg_tuition,
                COUNT(DISTINCT student_id) AS total_grads,
                COUNT(DISTINCT CASE WHEN opt_ind = 1 THEN student_id END) AS opt_users,
                COUNT(DISTINCT CASE WHEN opt_stem_ind = 1 THEN student_id END) AS opt_stem_users,
                COUNT(DISTINCT CASE WHEN status_change_ind = 1 THEN student_id END) AS status_change_users,
                SUM(opt_years) AS total_opt_years
            FROM student_level
            GROUP BY calendar_year, relabel_year, relabel_type, degree_type
        )
        SELECT
            calendar_year,
            relabel_year,
            relabel_type,
            degree_type,
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
        return control_calendar
    control_calendar["opt_share"] = control_calendar["opt_users"] / control_calendar["total_grads"]
    control_calendar["opt_stem_share"] = control_calendar["opt_stem_users"] / control_calendar["total_grads"]
    control_calendar["status_change_share"] = control_calendar["status_change_users"] / control_calendar["total_grads"]
    control_calendar["opt_years_avg"] = control_calendar["total_opt_years"] / control_calendar["total_grads"]
    control_calendar["f1_share_of_ctotalt"] = pd.NA
    control_calendar["f1_share_of_cnralt"] = pd.NA
    control_calendar["tuition_total"] = control_calendar["avg_tuition"] * control_calendar["total_grads"]
    control_calendar["event_t"] = control_calendar["calendar_year"] - control_calendar["relabel_year"]
    control_event = (
        control_calendar.groupby(["event_t", "relabel_type", "degree_type"], as_index=False)
        .agg(
            total_grads=("total_grads", "sum"),
            opt_users=("opt_users", "sum"),
            opt_stem_users=("opt_stem_users", "sum"),
            status_change_users=("status_change_users", "sum"),
            total_opt_years=("total_opt_years", "sum"),
            tuition_total=("tuition_total", "sum"),
        )
    )
    control_event["ctotalt"] = pd.NA
    control_event["cnralt"] = pd.NA
    control_event["opt_share"] = control_event["opt_users"] / control_event["total_grads"]
    control_event["opt_stem_share"] = control_event["opt_stem_users"] / control_event["total_grads"]
    control_event["status_change_share"] = control_event["status_change_users"] / control_event["total_grads"]
    control_event["opt_years_avg"] = control_event["total_opt_years"] / control_event["total_grads"]
    control_event["f1_share_of_ctotalt"] = pd.NA
    control_event["f1_share_of_cnralt"] = pd.NA
    control_event["avg_tuition"] = control_event["tuition_total"] / control_event["total_grads"]
    return control_event


def compute_generalized_did_panel(
    con: ddb.DuckDBPyConnection,
    verified_events: pd.DataFrame,
    *,
    degree_type: str,
    foia_path: str | Path = base.FOIA_PATH,
    inst_cw_path: str | Path = base.F1_INST_CW_PATH,
    ipeds_path: str | Path = base.IPEDS_PATH,
) -> pd.DataFrame:
    degree_events = verified_events[verified_events["degree_type"] == degree_type].copy()
    if degree_events.empty:
        return pd.DataFrame()
    matched_pairs = match_treated_to_never_treated(con, degree_events, ipeds_path=ipeds_path)
    if matched_pairs.empty:
        return pd.DataFrame()
    schema = _load_foia_base(con, foia_path=foia_path, inst_cw_path=inst_cw_path)
    _ensure_ipeds_view(con, ipeds_path)
    panel_rows = build_verified_event_panel(con, degree_events, ipeds_path=ipeds_path)
    relabel_events = panel_rows[
        [
            "unitid",
            "year",
            "relabel_year",
            "relabel_type",
            "degree_type",
            "event_source_cip6",
            "target_cip6",
            "ctotalt",
            "cnralt",
        ]
    ].drop_duplicates()
    con.register("generalized_did_pairs_py", matched_pairs)
    con.register("generalized_did_events_py", relabel_events)

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
    year_match_clause = (
        f"AND CAST({foia_year_col} AS INTEGER) = CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER)"
        if foia_year_col
        else ""
    )
    degree_case_events = _foia_degree_case("r.degree_type")
    degree_case_pairs = _foia_degree_case("p.degree_type")

    treated_calendar = con.sql(
        f"""
        WITH foia_base AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0') AS cip6,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                employer_name,
                employment_opt_type,
                {opt_end_col} AS opt_end_date,
                {foia_tuition_col} AS tuition,
                {foia_end_col} AS program_end_date,
                {status_col} AS requested_status,
                {foia_edu_col} AS foia_degree_label
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE {foia_end_col} IS NOT NULL
              {year_match_clause}
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
                r.event_source_cip6,
                r.target_cip6
            FROM relevant_foia f
            JOIN generalized_did_events_py r
              ON f.unitid = r.unitid
             AND f.grad_year = r.year
             AND (
                    f.cip6 = r.event_source_cip6
                 OR f.cip6 = r.target_cip6
             )
             AND f.foia_degree_label = {degree_case_events}
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
             AND tf.event_source_cip6 = p.source_cip6
             AND tf.target_cip6 = p.target_cip6
        ),
        student_level AS (
            SELECT
                pair_id,
                unitid,
                grad_year,
                relabel_year,
                relabel_type,
                ctotalt,
                cnralt,
                MAX(CASE WHEN employer_name IS NOT NULL THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL
                    THEN DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25
                    ELSE 0
                END AS opt_years,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                student_id
            FROM matched_treated
            GROUP BY pair_id, unitid, grad_year, relabel_year, relabel_type, ctotalt, cnralt, student_id
        )
        SELECT
            pair_id,
            unitid,
            grad_year AS calendar_year,
            relabel_year,
            relabel_type,
            AVG(avg_tuition) AS avg_tuition,
            COUNT(DISTINCT student_id) AS total_grads,
            COUNT(DISTINCT CASE WHEN opt_ind = 1 THEN student_id END) AS opt_users,
            COUNT(DISTINCT CASE WHEN opt_stem_ind = 1 THEN student_id END) AS opt_stem_users,
            COUNT(DISTINCT CASE WHEN status_change_ind = 1 THEN student_id END) AS status_change_users,
            SUM(opt_years) AS total_opt_years,
            MAX(ctotalt) AS ctotalt,
            MAX(cnralt) AS cnralt
        FROM student_level
        GROUP BY pair_id, unitid, calendar_year, relabel_year, relabel_type
        """
    ).df()

    control_calendar = con.sql(
        f"""
        WITH control_ipeds AS (
            SELECT
                p.pair_id,
                CAST(i.unitid AS BIGINT) AS unitid,
                CAST(i.year AS INTEGER) AS calendar_year,
                SUM(CAST(i.ctotalt AS DOUBLE)) AS ctotalt,
                SUM(CAST(i.cnralt AS DOUBLE)) AS cnralt
            FROM ipeds_raw i
            JOIN generalized_did_pairs_py p
              ON CAST(i.unitid AS BIGINT) = p.control_unitid
             AND CAST(i.awlevel AS INTEGER) = p.awlevel
             AND LPAD(CAST(i.cipcode AS VARCHAR), 6, '0') = p.source_cip6
            GROUP BY p.pair_id, CAST(i.unitid AS BIGINT), CAST(i.year AS INTEGER)
        ),
        foia_base AS (
            SELECT
                cw.{cw_unitid_col} AS unitid,
                LPAD(CAST({norm_cip_expr} AS VARCHAR), 6, '0') AS cip6,
                CAST(EXTRACT(YEAR FROM {foia_end_col}) AS INTEGER) AS grad_year,
                CAST({foia_student_col} AS VARCHAR) AS student_id,
                employer_name,
                employment_opt_type,
                {opt_end_col} AS opt_end_date,
                {foia_tuition_col} AS tuition,
                {foia_end_col} AS program_end_date,
                {status_col} AS requested_status,
                {foia_edu_col} AS foia_degree_label
            FROM foia_raw fr
            LEFT JOIN f1_inst_cw cw
              ON fr.{foia_inst_col} = cw.{cw_inst_col}
            WHERE {foia_end_col} IS NOT NULL
              {year_match_clause}
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
                p.degree_type
            FROM relevant_foia f
            JOIN generalized_did_pairs_py p
              ON f.unitid = p.control_unitid
             AND f.cip6 = p.source_cip6
             AND f.foia_degree_label = {degree_case_pairs}
        ),
        student_level AS (
            SELECT
                pair_id,
                unitid,
                grad_year,
                relabel_year,
                relabel_type,
                MAX(CASE WHEN employer_name IS NOT NULL THEN 1 ELSE 0 END) AS opt_ind,
                MAX(CASE WHEN COALESCE(employment_opt_type, '') = 'STEM' THEN 1 ELSE 0 END) AS opt_stem_ind,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                CASE
                    WHEN MAX(opt_end_date) IS NOT NULL
                    THEN DATE_DIFF('day', MAX(program_end_date), MAX(opt_end_date)) / 365.25
                    ELSE 0
                END AS opt_years,
                AVG(TRY_CAST(tuition AS DOUBLE)) AS avg_tuition,
                student_id
            FROM matched_control
            GROUP BY pair_id, unitid, grad_year, relabel_year, relabel_type, student_id
        )
        SELECT
            s.pair_id,
            s.unitid,
            s.grad_year AS calendar_year,
            s.relabel_year,
            s.relabel_type,
            AVG(s.avg_tuition) AS avg_tuition,
            COUNT(DISTINCT s.student_id) AS total_grads,
            COUNT(DISTINCT CASE WHEN s.opt_ind = 1 THEN s.student_id END) AS opt_users,
            COUNT(DISTINCT CASE WHEN s.opt_stem_ind = 1 THEN s.student_id END) AS opt_stem_users,
            COUNT(DISTINCT CASE WHEN s.status_change_ind = 1 THEN s.student_id END) AS status_change_users,
            SUM(s.opt_years) AS total_opt_years,
            MAX(ci.ctotalt) AS ctotalt,
            MAX(ci.cnralt) AS cnralt
        FROM student_level s
        LEFT JOIN control_ipeds ci
          ON s.pair_id = ci.pair_id
         AND s.unitid = ci.unitid
         AND s.grad_year = ci.calendar_year
        GROUP BY s.pair_id, s.unitid, s.grad_year, s.relabel_year, s.relabel_type
        """
    ).df()
    if treated_calendar.empty or control_calendar.empty:
        return pd.DataFrame()

    treated_calendar["treated"] = 1
    control_calendar["treated"] = 0
    did_panel = pd.concat([treated_calendar, control_calendar], ignore_index=True)
    did_panel["event_t"] = did_panel["calendar_year"] - did_panel["relabel_year"]
    did_panel["opt_share"] = did_panel["opt_users"] / did_panel["total_grads"]
    did_panel["opt_stem_share"] = did_panel["opt_stem_users"] / did_panel["total_grads"]
    did_panel["status_change_share"] = did_panel["status_change_users"] / did_panel["total_grads"]
    did_panel["opt_years_avg"] = did_panel["total_opt_years"] / did_panel["total_grads"]
    did_panel["f1_share_of_ctotalt"] = did_panel["total_grads"] / did_panel["ctotalt"]
    did_panel["f1_share_of_cnralt"] = did_panel["total_grads"] / did_panel["cnralt"]
    did_panel["tuition_total"] = did_panel["avg_tuition"] * did_panel["total_grads"]
    did_panel["avg_tuition"] = did_panel["tuition_total"] / did_panel["total_grads"]
    did_panel["degree_type"] = degree_type
    return did_panel


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_relabel_year_histogram(events: pd.DataFrame, *, degree_type: str, out_dir: str | Path) -> Path | None:
    df = events[
        (events["degree_type"] == degree_type)
        & events["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"])
        & events["relabel_year"].notna()
    ].copy()
    if df.empty:
        return None
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": BASE_FONT_SIZE})
    fig, ax = plt.subplots(figsize=(8, 5))
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
    fig.tight_layout()
    out_path = Path(out_dir) / f"relabel_year_histogram_{_slugify(degree_type)}.png"
    _save_figure(fig, out_path)
    return out_path


def plot_opt_usage_generalized(
    opt_usage: pd.DataFrame,
    *,
    yvar: str,
    degree_type: str,
    out_dir: str | Path,
) -> Path | None:
    data = opt_usage[
        opt_usage["calendar_year"].between(base.PLOT_YEAR_MIN, base.PLOT_YEAR_MAX)
    ].copy()
    if data.empty:
        return None
    sns.set(style="whitegrid")
    sns.set_palette(base.PALETTE_SEQ)
    plt.rcParams.update({"font.size": BASE_FONT_SIZE})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x="calendar_year",
        y=yvar,
        hue="relabel_year",
        marker="o",
        ax=ax,
    )
    ax.set_ylabel(base.yvar_label(yvar))
    ax.set_xlabel("Calendar year (program end year)")
    ax.set_title("")
    fig.tight_layout()
    out_path = Path(out_dir) / f"{_slugify(degree_type)}_{yvar}_opt_usage_by_relabel_year.png"
    _save_figure(fig, out_path)
    return out_path


def plot_event_time_with_control_generalized(
    treated_event: pd.DataFrame,
    control_event: pd.DataFrame,
    *,
    yvar: str,
    degree_type: str,
    out_dir: str | Path,
) -> tuple[Path | None, Path | None]:
    if treated_event.empty:
        return None, None
    sns.set(style="whitegrid")
    sns.set_palette(base.PALETTE_SEQ)
    plt.rcParams.update({"font.size": BASE_FONT_SIZE})

    treated_path: Path | None = None
    combined_path: Path | None = None

    fig_t, ax_t = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=treated_event[treated_event["event_t"].between(PLOT_EVENT_MIN, PLOT_EVENT_MAX)],
        x="event_t",
        y=yvar,
        marker="o",
        ax=ax_t,
    )
    ax_t.set_ylabel(base.yvar_label(yvar))
    ax_t.set_xlabel("Years relative to relabel (t=0)")
    ax_t.set_title("")
    ax_t.axvline(x=0, linestyle="--", color="gray", linewidth=1)
    fig_t.tight_layout()
    treated_path = Path(out_dir) / f"{_slugify(degree_type)}_{yvar}_event_time_treated.png"
    _save_figure(fig_t, treated_path)

    if control_event is None or control_event.empty:
        return treated_path, None

    treated_plot = treated_event.copy()
    treated_plot["series_label"] = "Treated"
    control_plot = control_event.copy()
    control_plot["series_label"] = "Matched Never-Treated"
    plot_df = pd.concat([treated_plot, control_plot], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=plot_df[plot_df["event_t"].between(PLOT_EVENT_MIN, PLOT_EVENT_MAX)],
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
    combined_path = Path(out_dir) / f"{_slugify(degree_type)}_{yvar}_event_time_treated_control_never_treated.png"
    _save_figure(fig, combined_path)
    return treated_path, combined_path


def plot_did_event_study_generalized(
    did_event_study: pd.DataFrame,
    *,
    yvar: str,
    degree_type: str,
    out_dir: str | Path,
) -> Path | None:
    if did_event_study.empty:
        return None
    plot_df = did_event_study.sort_values("event_t").copy()
    sns.set(style="whitegrid")
    sns.set_palette(base.PALETTE_SEQ)
    plt.rcParams.update({"font.size": BASE_FONT_SIZE})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        plot_df["event_t"],
        plot_df["coef"],
        yerr=plot_df["se"],
        fmt="o-",
        color=base.PALETTE_SEQ[0],
        capsize=4,
        linewidth=2,
        markersize=6,
    )
    ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
    ax.axvline(x=0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Years relative to relabel (t=0)")
    ax.set_ylabel(f"DiD coef on treated x event time: {base.yvar_label(yvar)}")
    ax.set_title("")
    fig.tight_layout()
    out_path = Path(out_dir) / f"{_slugify(degree_type)}_{yvar}_did_event_time_never_treated.png"
    csv_path = out_path.with_suffix(".csv")
    plot_df.to_csv(csv_path, index=False)
    _save_figure(fig, out_path)
    return out_path


def write_generalized_report(
    events: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    out_path: str | Path,
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
    ipeds_path: str | Path = base.IPEDS_PATH,
) -> list[Path]:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    yvars = yvars or DEFAULT_YVARS
    outputs: list[Path] = []
    _progress(f"Generating plots in {plots_dir}")
    for degree_type in ("Bachelor", "Master", "Doctor"):
        degree_events = events[
            (events["degree_type"] == degree_type)
            & events["event_origin_category"].isin(["ipeds_only", "external_ipeds_verified"])
        ]
        _progress(f"Plotting degree bucket {degree_type}: {len(degree_events):,} verified event(s)")
        hist_path = plot_relabel_year_histogram(events, degree_type=degree_type, out_dir=plots_dir)
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
        )
        if opt_usage.empty:
            _progress(f"Skipping {degree_type} outcome plots because FOIA/IPEDS join returned no rows")
            continue
        opt_event = compute_opt_usage_event_time_generalized(opt_usage)
        control_event = compute_never_treated_control_event_time_generalized(
            con,
            degree_events,
            degree_type=degree_type,
            foia_path=foia_path,
            inst_cw_path=inst_cw_path,
        )
        did_panel = compute_generalized_did_panel(
            con,
            degree_events,
            degree_type=degree_type,
            foia_path=foia_path,
            inst_cw_path=inst_cw_path,
            ipeds_path=ipeds_path,
        )
        for yvar in yvars:
            usage_path = plot_opt_usage_generalized(opt_usage, yvar=yvar, degree_type=degree_type, out_dir=plots_dir)
            if usage_path is not None:
                outputs.append(usage_path)
            treated_path, control_path = plot_event_time_with_control_generalized(
                opt_event,
                control_event,
                yvar=yvar,
                degree_type=degree_type,
                out_dir=plots_dir,
            )
            if treated_path is not None:
                outputs.append(treated_path)
            if control_path is not None:
                outputs.append(control_path)
            if not did_panel.empty:
                did_event_study = v2.compute_did_event_study(did_panel=did_panel, yvar=yvar)
                did_path = plot_did_event_study_generalized(
                    did_event_study,
                    yvar=yvar,
                    degree_type=degree_type,
                    out_dir=plots_dir,
                )
                if did_path is not None:
                    outputs.append(did_path)
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
    events_parquet: str | Path = DEFAULT_EVENTS_PARQUET,
    events_csv: str | Path = DEFAULT_EVENTS_CSV,
    panel_parquet: str | Path = DEFAULT_PANEL_PARQUET,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    candidate_audit_csv: str | Path = DEFAULT_CANDIDATE_AUDIT_CSV,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
) -> dict[str, pd.DataFrame]:
    _progress("Starting generalized relabel pipeline")
    _progress(f"IPEDS source: {ipeds_path}")
    _progress(f"Crosswalk source: {crosswalk_path}")
    _progress(f"FOIA source: {foia_path}")
    _progress(f"Candidate source: {candidate_path}")
    con = ddb.connect()
    cip_map = _load_ipeds_cip_map(ipeds_path)
    _progress(f"Loaded {len(cip_map):,} CIP labels from IPEDS")
    _progress("Running strict IPEDS relabel scan")
    strict_events = _enrich_event_labels(
        detect_ipeds_relabels(con, ipeds_path=ipeds_path, thresholds=STRICT_THRESHOLDS),
        cip_map,
    )
    _progress(f"Strict IPEDS scan found {len(strict_events):,} event(s)")
    candidate_audit = pd.DataFrame()
    verified_external = pd.DataFrame()
    if candidate_path is not None:
        candidates = load_external_candidates(candidate_path)
        _progress("Loading school lookup for candidate-school resolution")
        school_lookup = load_school_lookup(crosswalk_path)
        resolved_candidates = resolve_candidate_schools(candidates, school_lookup)
        verified_external, candidate_audit = verify_external_candidates(
            con,
            resolved_candidates,
            ipeds_path=ipeds_path,
            relaxed_thresholds=RELAXED_THRESHOLDS,
            cip_map=cip_map,
        )
        if not verified_external.empty:
            verified_external = _enrich_event_labels(verified_external, cip_map)
        _progress(
            f"External candidate stage produced {len(verified_external):,} verified event(s) "
            f"from {len(candidate_audit):,} audited candidate row(s)"
        )
    else:
        _progress("No external candidate input provided; skipping candidate verification stage")

    _progress("Merging IPEDS-driven and externally driven event sources")
    merged_events = merge_event_sources(strict_events, verified_external, candidate_audit, cip_map=cip_map)
    merged_events = _coerce_verified_event_output_dtypes(merged_events)
    _progress(f"Merged event table contains {len(merged_events):,} row(s)")
    verified_panel = build_verified_event_panel(con, merged_events, ipeds_path=ipeds_path)
    verified_panel = _coerce_verified_event_output_dtypes(verified_panel)

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
    _progress(f"Writing report to {report_path}")
    write_generalized_report(merged_events, candidate_audit, report_path)
    try:
        run_degree_plots(
            con,
            merged_events,
            verified_panel,
            plots_dir=plots_dir,
            foia_path=foia_path,
            inst_cw_path=inst_cw_path,
            ipeds_path=ipeds_path,
        )
    except Exception as exc:
        print(f"Warning: plot generation failed ({exc})")
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
    parser.add_argument("--events-parquet", type=str, default=str(DEFAULT_EVENTS_PARQUET))
    parser.add_argument("--events-csv", type=str, default=str(DEFAULT_EVENTS_CSV))
    parser.add_argument("--panel-parquet", type=str, default=str(DEFAULT_PANEL_PARQUET))
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--candidate-audit-csv", type=str, default=str(DEFAULT_CANDIDATE_AUDIT_CSV))
    parser.add_argument("--plots-dir", type=str, default=str(DEFAULT_PLOTS_DIR))
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
        events_parquet=args.events_parquet,
        events_csv=args.events_csv,
        panel_parquet=args.panel_parquet,
        report_path=args.report_path,
        candidate_audit_csv=args.candidate_audit_csv,
        plots_dir=args.plots_dir,
    )
    print(f"Wrote {len(outputs['events']):,} merged events")
    print(f"Wrote {len(outputs['panel']):,} verified panel rows")
    if not outputs["candidate_audit"].empty:
        verified_count = int((outputs["candidate_audit"]["external_verified"] == 1).sum())
        print(f"Audited {len(outputs['candidate_audit']):,} external candidates; verified {verified_count:,}")


if __name__ == "__main__":
    main()
