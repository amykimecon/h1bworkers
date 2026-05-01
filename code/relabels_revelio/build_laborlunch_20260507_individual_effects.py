#!/usr/bin/env python
"""Build individual-effects assets for slides_laborlunch_20260507.

This script owns the non-firm relabel/individual assets referenced by the
labor-lunch deck.  It is intentionally an orchestrator around the existing
FOIA and Revelio analysis modules, with a small cache layer for final panels so
the expensive matching and outcome construction work is not repeated.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import duckdb as ddb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


HOME = Path.home()
CODE_ROOT = HOME / "h1bworkers" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from relabels_revelio import compare_opt_stem_did_estimators as estimator_cmp
from relabels_revelio import relabel_events_generalized as generalized
from relabels_revelio import relabel_indiv_analysis as indiv
from config import root as PROJECT_ROOT
import laborlunch_plot_style as llstyle


DEFAULT_TEX = HOME / "writing" / "slides" / "slides_laborlunch_20260507" / "slides_laborlunch_20260507.tex"
DEFAULT_CACHE_DIR = CODE_ROOT / "output" / "relabel_indiv" / "laborlunch_20260507_individual_cache"
DEFAULT_FOIA_OUTPUT_DIR = CODE_ROOT / "output" / "relabel_indiv"
DEFAULT_FOIA_PLOTS_DIR = DEFAULT_FOIA_OUTPUT_DIR / "generalized_relabels_plots"
DEFAULT_FIGURE_OUTPUT_DIR = HOME / "figures"
DEFAULT_REVELIO_MAIN_OUTPUT_DIR = HOME / "output" / "relabel_indiv"
DEFAULT_REVELIO_ECON_OUTPUT_DIR = HOME / "output" / "relabel_indiv_slides_nocontrols"
DEFAULT_REVELIO_CONTROLS_OUTPUT_DIR = HOME / "output" / "relabel_indiv_slides_controls"
DEFAULT_REVELIO_ALT_OUTPUT_DIR = HOME / "output" / "relabel_indiv_slides_alt_spec"
DEFAULT_REVELIO_ALT_TEX_MACROS = CODE_ROOT / ".tmp" / "revelio_alt_spec_macros.tex"
REVELIO_MAX_HORIZON = 4
REVELIO_POOLED_EVENT_MIN = -1
REVELIO_POOLED_EVENT_MAX = 3
REVELIO_POOLED_STATS_COLUMNS = (
    "baseline_mean",
    "effect_size",
    "pooled_post_event_min",
    "pooled_post_event_max",
)
DEFAULT_HORIZONS = list(range(REVELIO_MAX_HORIZON + 1))
DEFAULT_REVELIO_RELABEL_YEAR_MIN = 2014
DEFAULT_REVELIO_RELABEL_YEAR_MAX = 2020
DEFAULT_SAMPLE_VARIANTS = ["stage04_all", "foia_linked_person_baseline"]
DEFAULT_COHORT_YVARS = [
    "opt_share",
    "opt_stem_share",
    "post_grad_authorization_years_avg",
    "opt_duration_years_avg",
    "status_change_share",
    "f1_share_of_ctotalt",
    "f1_share_of_cnralt",
    "cnralt_share_of_ctotalt",
    "ctotalt",
    "cnralt",
    "avg_tuition",
    "avg_tuition_ipeds",
    "avg_fees_ipeds",
    "avg_students_personal_funds",
    "avg_total_funds",
    "unique_employers",
    "unique_opt_cities",
    "auth_employment_tenure_years",
    "employer_opt_intensity_pctile",
    "internship_count",
    "internship_opt_years",
]
ESTIMATION_TYPES = {"twfe", "sun_abraham", "callaway_santanna"}
REVELIO_OUTPUT_MODES = {"old_event_time", "new_pooled_post"}
REVELIO_MAIN_DID_SAMPLE_MODES = {"econ_only", "full_sample"}
REVELIO_MAIN_CONTROL_GROUPS = (
    generalized.CONTROL_GROUP_ALWAYS_STEM,
    generalized.CONTROL_GROUP_NEVER_TREATED,
)
REVELIO_SAMPLE_LABELS = {
    "econ_only": "Econ-only",
    "full_sample": "Full-sample",
}
REVELIO_ALT_SPEC_CONFIG = {
    "new_pooled_post": {
        "alternate_output_mode": "old_event_time",
        "button_suffix": "event_time",
        "button_label": "Event Time",
    },
    "old_event_time": {
        "alternate_output_mode": "new_pooled_post",
        "button_suffix": "foia_horizons",
        "button_label": "Grad Horizons",
    },
}
ESTIMATOR_AGGREGATION_VERSION = "cohort_broad_degree_full_pre5_v4"
REVELIO_OUTCOME_PANEL_VERSION = "cumulative_positions_employers_tenure_school_internship_spell_v1"
FOIA_ESTIMATOR_PANEL_VERSION = f"{ESTIMATOR_AGGREGATION_VERSION}_donor_controls_v6_degree_costs_funds"
FOIA_STATUS_CHANGE_MAX_CALENDAR_YEAR = 2020
FOIA_STATUS_CHANGE_SAMPLE_VERSION = f"cohorts_le_{FOIA_STATUS_CHANGE_MAX_CALENDAR_YEAR}"
FOIA_DEFAULT_CONTROL_GROUP = generalized.CONTROL_GROUP_NEVER_TREATED
FOIA_MAIN_COMPARISON_CONTROL_GROUPS = (
    generalized.CONTROL_GROUP_NEVER_TREATED,
    generalized.CONTROL_GROUP_ALWAYS_STEM,
)
FOIA_CONTROL_GROUP_LABELS = {
    generalized.CONTROL_GROUP_NEVER_TREATED: "Never-treated",
    generalized.CONTROL_GROUP_ALWAYS_STEM: "Always-STEM",
    generalized.CONTROL_GROUP_LATE_TREATED: "Late-treated",
}
FOIA_CONTROL_GROUP_COLORS = {
    generalized.CONTROL_GROUP_NEVER_TREATED: generalized.base.PALETTE_SEQ[0],
    generalized.CONTROL_GROUP_ALWAYS_STEM: generalized.base.PALETTE_SEQ[1],
    generalized.CONTROL_GROUP_LATE_TREATED: generalized.base.PALETTE_SEQ[2],
}
FOIA_CONTROL_GROUP_MARKERS = {
    generalized.CONTROL_GROUP_NEVER_TREATED: "o",
    generalized.CONTROL_GROUP_ALWAYS_STEM: "s",
    generalized.CONTROL_GROUP_LATE_TREATED: "^",
}
REVELIO_SAMPLE_MARKERS = {
    "stage04_all": "o",
    "foia_linked_person_baseline": "s",
    "stage04_all_foreign": "o",
    "stage04_all_non_foreign": "s",
    "foia_linked_person_baseline_foreign": "D",
    "foia_linked_person_baseline_non_foreign": "^",
}
FOIA_MAIN_LARGER_TEXT_YVARS = {"ctotalt", "cnralt_share_of_ctotalt"}
FOIA_MAIN_LARGER_TEXT_SCALE = 1.18
ESTIMATOR_LABELS = {
    "did": "TWFE",
    "twfe": "TWFE",
    "sun_abraham": "Sun-Abraham",
    "callaway_santanna": "Callaway-Sant'Anna",
}
ESTIMATOR_COMPARISON_TYPES = ("twfe", "sun_abraham", "callaway_santanna")
ESTIMATOR_ALIASES = {
    "did": "twfe",
    "simple": "twfe",
    "twfe": "twfe",
    "sun_abraham": "sun_abraham",
    "sun_abraham_iw": "sun_abraham",
    "callaway_santanna": "callaway_santanna",
}
FOIA_ESTIMATION_APPENDICES = [
    ("relabel_foia_stem_cip_eligible", "stem_cip_eligible_share", "FOIA STEM OPT CIP Eligibility"),
    ("relabel_foia_takeup", "opt_share", "FOIA OPT Take-Up"),
    ("relabel_foia_opt_stem", "opt_stem_share", "FOIA STEM OPT Take-Up"),
    ("relabel_foia_post_grad_authorization", "post_grad_authorization_years_avg", "FOIA Post-Grad Authorization"),
    ("relabel_foia_opt_duration", "opt_duration_years_avg", "FOIA OPT Duration"),
    ("relabel_foia_status_change", "status_change_share", "FOIA Status Change"),
    ("relabel_foia_f1_ctotalt", "f1_share_of_ctotalt", "FOIA F-1 Share of IPEDS Completions"),
    ("relabel_foia_f1_cnralt", "f1_share_of_cnralt", "FOIA F-1 Share of Nonresident Completions"),
    ("relabel_foia_cnralt_share", "cnralt_share_of_ctotalt", "IPEDS Nonresident Share of Completions"),
    ("relabel_foia_ctotalt", "ctotalt", "FOIA IPEDS Total Completions"),
    ("relabel_foia_cnralt", "cnralt", "FOIA IPEDS Nonresident Completions"),
    ("relabel_foia_other", "avg_tuition", "FOIA Tuition"),
    ("relabel_foia_tuition_ipeds", "avg_tuition_ipeds", "External IPEDS Tuition"),
    ("relabel_foia_fees_ipeds", "avg_fees_ipeds", "External IPEDS Fees"),
    ("relabel_foia_students_personal_funds", "avg_students_personal_funds", "FOIA Student Personal Funds"),
    ("relabel_foia_total_funds", "avg_total_funds", "FOIA Total Funds"),
    ("relabel_foia_unique_employers", "unique_employers", "FOIA Unique Employers"),
    ("relabel_foia_unique_opt_cities", "unique_opt_cities", "FOIA Unique OPT Cities"),
    ("relabel_foia_auth_employment_tenure", "auth_employment_tenure_years", "FOIA Authorization-Employment Tenure"),
    ("relabel_foia_employer_opt_intensity", "employer_opt_intensity_pctile", "FOIA Employer OPT Intensity"),
    ("relabel_foia_internship_count", "internship_count", "FOIA Average Internship Count"),
    ("relabel_foia_internship_opt_years", "internship_opt_years", "FOIA Average Internship OPT Time"),
]
FOIA_STEM_CIP_YVAR = "stem_cip_eligible_share"
REVELIO_ESTIMATION_APPENDICES = [
    ("relabel_revelio_active_us", "in_us", "active_us", "Revelio Active U.S."),
    ("relabel_revelio_linkedin", "linkedin_active_through_target_year", "linkedin_active", "Revelio LinkedIn Activity"),
    ("relabel_revelio_positions", "n_pos", "active_positions", "Revelio Active Positions"),
    ("relabel_revelio_unique_employers", "n_employers", "unique_employers", "Revelio Unique Employers"),
    ("relabel_revelio_employer_tenure", "avg_employer_tenure_years", "employer_tenure", "Revelio Employer Tenure"),
    ("relabel_revelio_in_school", "in_school", "in_school", "Revelio In School"),
    ("relabel_revelio_compensation", "salary_imputed", "compensation", "Revelio Compensation"),
    ("relabel_revelio_internship_positions", "n_internship_positions", "internship_positions", "Revelio Education-Spell Positions"),
]


def _progress(message: str) -> None:
    print(f"[laborlunch] {time.strftime('%H:%M:%S')} {message}", flush=True)


def _progress_report_every(total: int, *, target_reports: int = 8) -> int:
    if total <= 0:
        return 1
    return max(1, math.ceil(total / max(1, target_reports)))


@dataclass
class StageProgress:
    total: int
    current: int = 0

    def next(self, label: str) -> None:
        self.current += 1
        remaining = max(0, self.total - self.current)
        _progress(f"stage {self.current}/{self.total}: {label} ({remaining} stages remaining)")


@dataclass
class ProgressCounter:
    label: str
    total: int
    report_every: int | None = None
    done: int = 0

    def advance(self, latest: str | None = None) -> None:
        self.done += 1
        if self.total <= 0:
            return
        report_every = self.report_every or _progress_report_every(self.total)
        should_report = self.done == 1 or self.done == self.total or self.done % report_every == 0
        if not should_report:
            return
        remaining = max(0, self.total - self.done)
        detail = f"; latest: {latest}" if latest else ""
        _progress(f"{self.label}: {self.done}/{self.total} complete, {remaining} remaining{detail}")


@dataclass
class RunLog:
    cache_hits: list[str]
    rebuilt: list[str]
    skipped: list[str]
    output_paths: list[str]
    verbose: bool = False

    def hit(self, name: str) -> None:
        self.cache_hits.append(name)
        if self.verbose:
            _progress(f"cache hit: {name}")

    def rebuild(self, name: str) -> None:
        self.rebuilt.append(name)
        if self.verbose:
            _progress(f"rebuild: {name}")

    def skip(self, name: str) -> None:
        self.skipped.append(name)
        if self.verbose:
            _progress(f"skip: {name}")

    def output(self, path: str | Path | None) -> None:
        if path:
            self.output_paths.append(str(path))


def _elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "x"


def _revelio_event_source_mode(sample_mode: str) -> str:
    if sample_mode == "full_sample":
        return "generalized_final_sample"
    if sample_mode == "econ_only":
        return "econ_v2"
    raise ValueError(f"Unsupported Revelio sample mode: {sample_mode}")


def _revelio_primary_control_group(sample_mode: str) -> str:
    if sample_mode == "full_sample":
        return generalized.CONTROL_GROUP_ALWAYS_STEM
    return generalized.CONTROL_GROUP_NEVER_TREATED


def _filter_revelio_relabel_years(
    frame: pd.DataFrame,
    *,
    relabel_year_min: int | None,
    relabel_year_max: int | None,
) -> pd.DataFrame:
    if frame.empty or "relabel_year" not in frame.columns:
        return frame
    year = pd.to_numeric(frame["relabel_year"], errors="coerce")
    keep = year.notna()
    if relabel_year_min is not None:
        keep &= year.ge(int(relabel_year_min))
    if relabel_year_max is not None:
        keep &= year.le(int(relabel_year_max))
    return frame.loc[keep].copy()


def _revelio_control_groups(args: argparse.Namespace) -> tuple[str, ...]:
    primary = _revelio_primary_control_group(args.revelio_main_did_sample)
    if not getattr(args, "revelio_control_comparison", False):
        return (primary,)
    if args.revelio_main_did_sample != "full_sample":
        return (primary,)
    return tuple(dict.fromkeys((primary, *REVELIO_MAIN_CONTROL_GROUPS)))


def _file_ok(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size > 0


def _hash_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _lookup_config_value(data: dict[str, Any], key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _interpolate_config_string(value: str, data: dict[str, Any]) -> str:
    out = os.path.expanduser(value.replace("{root}", str(PROJECT_ROOT)))

    def repl(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        resolved = _lookup_config_value(data, key) if key else None
        return str(resolved) if resolved is not None else match.group(0)

    return re.sub(r"\$(\w+)|\$\{([^}]+)\}", repl, out)


def _walk_config(obj: Any, data: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {key: _walk_config(value, data) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_walk_config(value, data) for value in obj]
    if isinstance(obj, str):
        return _interpolate_config_string(obj, data)
    return obj


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config is not a YAML mapping: {path}")
    data = _walk_config(data, data)
    return _walk_config(data, data)


def _deep_set(data: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    cur = data
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _materialize_revelio_config(
    *,
    base_config: Path,
    out_path: Path,
    output_dir: Path,
    panel_path: Path,
    did_path: Path,
    horizons: list[int],
    event_window: int,
    event_source_mode: str,
    relabel_year_min: int | None,
    relabel_year_max: int | None,
    control_group: str,
    did_plot_mode: str,
    include_controls: bool,
) -> Path:
    data = _read_yaml(base_config)
    _deep_set(data, ("build", "event_source_mode"), event_source_mode)
    _deep_set(data, ("build", "control_group"), control_group)
    _deep_set(data, ("build", "outcome_horizons"), horizons)
    _deep_set(data, ("build", "event_window"), int(event_window))
    _deep_set(data, ("build", "pooled_post_event_min"), REVELIO_POOLED_EVENT_MIN)
    _deep_set(data, ("build", "pooled_post_event_max"), REVELIO_POOLED_EVENT_MAX)
    _deep_set(data, ("build", "relabel_year_min"), relabel_year_min)
    _deep_set(data, ("build", "relabel_year_max"), relabel_year_max)
    _deep_set(data, ("build", "did_plot_mode"), did_plot_mode)
    _deep_set(data, ("build", "did_include_individual_controls"), bool(include_controls))
    _deep_set(data, ("build", "did_include_school_char_gradyear_controls"), bool(include_controls))
    _deep_set(data, ("build", "sample_cip_prefixes"), ["4506"] if event_source_mode == "econ_v2" else [])
    _deep_set(data, ("build", "sample_variants"), DEFAULT_SAMPLE_VARIANTS)
    _deep_set(data, ("build", "cohort_plot_yvars"), DEFAULT_COHORT_YVARS + ["linkedin_match_share"])
    _deep_set(data, ("paths", "output_dir"), str(output_dir))
    _deep_set(data, ("paths", "output_panel_parquet"), str(panel_path))
    _deep_set(data, ("paths", "output_did_results_parquet"), str(did_path))
    _write_yaml(out_path, data)
    return out_path


@contextlib.contextmanager
def _patched_indiv_config(**updates: Any) -> Iterator[None]:
    old_values: dict[str, Any] = {}
    for key, value in updates.items():
        old_values[key] = getattr(indiv.cfg, key, None)
        setattr(indiv.cfg, key, value)
    old_output_dir = indiv.OUTPUT_DIR
    if "OUTPUT_DIR" in updates:
        indiv.OUTPUT_DIR = Path(updates["OUTPUT_DIR"])
        indiv.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        for key, value in old_values.items():
            setattr(indiv.cfg, key, value)
        indiv.OUTPUT_DIR = old_output_dir


def parse_slide_asset_refs(tex_path: str | Path) -> list[Path]:
    """Return non-firm .png/.tex assets referenced literally by the deck."""
    tex_path = Path(tex_path)
    text = tex_path.read_text()
    macros = {
        name: value
        for name, value in re.findall(
            r"\\(?:newcommand|renewcommand|providecommand)\{\\([A-Za-z0-9_]+)\}\{([^{}]+)\}",
            text,
        )
    }
    refs: list[Path] = []
    seen: set[str] = set()
    for raw in re.findall(r"\{([^{}]*(?:\.png|\.tex))\}", text):
        if r"\_" in raw:
            continue
        expanded = raw.strip()
        for name, value in macros.items():
            expanded = expanded.replace(f"\\{name}", value)
        if "company_shift_share" in expanded or "\\companyoutput" in raw:
            continue
        if not (expanded.endswith(".png") or expanded.endswith(".tex")):
            continue
        path = Path(expanded)
        if not path.is_absolute():
            path = (tex_path.parent / path).resolve()
        key = str(path)
        if key not in seen:
            seen.add(key)
            refs.append(path)
    return refs


def _asset_subset(paths: list[Path], *needles: str) -> list[Path]:
    out = []
    for path in paths:
        text = str(path)
        if any(needle in text for needle in needles):
            out.append(path)
    return out


def _foia_yvars_for_slide_assets(paths: list[Path]) -> list[str] | None:
    foia_assets = _asset_subset(paths, "generalized_relabels_plots")
    if not foia_assets:
        return None
    requested = [
        yvar
        for yvar in generalized.DEFAULT_YVARS
        if any(yvar in path.name for path in foia_assets)
    ]
    return requested or None


def _missing_assets(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not _file_ok(path)]


def _revelio_cached_outcome_gap(results: pd.DataFrame, panel: pd.DataFrame) -> list[str]:
    if results.empty or "outcome" not in results.columns:
        return [out for out in indiv.OUTCOMES if out in panel.columns]
    present = set(results["outcome"].dropna().astype(str))
    return [out for out in indiv.OUTCOMES if out in panel.columns and out not in present]


def _revelio_cached_variant_gap(results: pd.DataFrame, panel: pd.DataFrame) -> list[str]:
    if results.empty or "analysis_variant" not in results.columns:
        return sorted(str(v) for v in panel.get("analysis_variant", pd.Series(dtype=str)).dropna().unique())
    expected: set[str] = set()
    if "analysis_variant" in panel.columns:
        for variant, variant_panel in panel.groupby("analysis_variant", sort=False):
            variant_name = str(variant)
            expected.add(variant_name)
            if "imputed_foreign_ind" not in variant_panel.columns:
                continue
            foreign_flag = pd.to_numeric(variant_panel["imputed_foreign_ind"], errors="coerce")
            for value, suffix in ((1, "foreign"), (0, "non_foreign")):
                subgroup = variant_panel[foreign_flag.eq(value)]
                if not subgroup.empty and subgroup["treated_ind"].nunique() >= 2:
                    expected.add(f"{variant_name}_{suffix}")
    present = set(results["analysis_variant"].dropna().astype(str))
    return sorted(expected - present)


def _revelio_cached_column_gap(results: pd.DataFrame, output_mode: str) -> list[str]:
    if output_mode != "new_pooled_post":
        return []
    missing = [col for col in REVELIO_POOLED_STATS_COLUMNS if col not in results.columns]
    if not missing:
        post_min = pd.to_numeric(results["pooled_post_event_min"], errors="coerce")
        post_max = pd.to_numeric(results["pooled_post_event_max"], errors="coerce")
        if not post_min.eq(REVELIO_POOLED_EVENT_MIN).all():
            missing.append("pooled_post_event_min")
        if not post_max.eq(REVELIO_POOLED_EVENT_MAX).all():
            missing.append("pooled_post_event_max")
    return missing


def _stale_assets(paths: list[Path], sources: Sequence[Path]) -> list[Path]:
    return [path for path in paths if _any_source_newer(path, sources)]


def _any_source_newer(target: Path, sources: Sequence[Path]) -> bool:
    if not _file_ok(target):
        return True
    target_mtime = target.stat().st_mtime
    return any(_file_ok(source) and source.stat().st_mtime > target_mtime for source in sources)


def _estimation_type_path(base_dir: Path, stem: str) -> Path:
    return base_dir / "estimation_type_appendix" / f"{stem}_estimation_type_comparison.png"


def _normalize_estimator_name(value: object) -> str | None:
    text = str(value).strip().lower()
    return ESTIMATOR_ALIASES.get(text)


def _clean_estimator_name(value: object) -> str:
    text = _normalize_estimator_name(value) or str(value).strip()
    return ESTIMATOR_LABELS.get(text, text.replace("_", " ").title())


def _did_coef_axis_label(readable_name: str) -> str:
    return f"did coef: {readable_name}"


def _plot_estimation_type_comparison(
    results: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    x_col: str,
    x_label: str,
    y_label: str | None = None,
    yvar: str | None = None,
) -> Path:
    plot_df = results.dropna(subset=[x_col, "coef"]).copy()
    if plot_df.empty:
        plot_df = pd.DataFrame(
            {
                x_col: [0],
                "coef": [0.0],
                "se": [np.nan],
                "estimator": ["unavailable"],
            }
        )
    plot_df["estimator"] = plot_df.get("estimator", plot_df.get("did_estimator", "twfe")).fillna("twfe").map(_normalize_estimator_name)
    plot_df = plot_df[plot_df["estimator"].isin(ESTIMATOR_COMPARISON_TYPES)].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df["coef"] = pd.to_numeric(plot_df["coef"], errors="coerce")
    plot_df["se"] = pd.to_numeric(plot_df.get("se", np.nan), errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, "coef"]).sort_values(["estimator", x_col])
    if plot_df.empty:
        raise RuntimeError(f"No usable estimator comparison rows for {out_path}")

    llstyle.apply_style()
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    for idx, estimator in enumerate(ESTIMATOR_COMPARISON_TYPES):
        grp = plot_df[plot_df["estimator"].eq(estimator)].copy()
        if grp.empty:
            continue
        grp = grp.sort_values(x_col)
        color = llstyle.color(idx)
        offset = float(llstyle.offsets(len(ESTIMATOR_COMPARISON_TYPES))[idx])
        x_plot = grp[x_col].astype(float) + offset
        ax.plot(x_plot, grp["coef"], marker="o", markersize=llstyle.MULTI_MARKER_SIZE, linewidth=1.5, color=color, label=_clean_estimator_name(estimator))
        with_se = grp.dropna(subset=["se"])
        if not with_se.empty:
            ax.errorbar(
                with_se[x_col].astype(float) + offset,
                with_se["coef"],
                yerr=1.96 * with_se["se"],
                fmt="none",
                **llstyle.errorbar_kwargs(color, llstyle.MULTI_MARKER_SIZE),
            )
    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.9)
    if x_col == "event_t":
        ax.axvline(-0.5, color="0.55", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("")
    ax.set_xlabel(x_label)
    if y_label:
        ylabel_text = y_label
    elif yvar:
        ylabel_text = generalized._did_coef_ylabel(yvar)  # noqa: SLF001
    else:
        ylabel_text = _did_coef_axis_label(title.split(":", 1)[0])
    ax.set_ylabel(ylabel_text)
    if yvar:
        generalized._format_yaxis_for_outcome(ax, yvar)  # noqa: SLF001
    llstyle.right_legend(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    llstyle.savefig(fig, out_path)
    return out_path


def validate_estimator_dependencies(estimation_type: str) -> None:
    if estimation_type == "twfe":
        return
    if estimation_type == "sun_abraham":
        try:
            import pyfixest  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Sun-Abraham estimation requires pyfixest.") from exc
    if estimation_type == "callaway_santanna":
        try:
            from csdid.att_gt import ATTgt  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Callaway-Sant'Anna estimation requires csdid.") from exc


def _force_estimator_rebuild(args: argparse.Namespace) -> bool:
    return bool(args.force_rebuild or getattr(args, "force_estimator_rebuild", False))


def _prepare_generic_regression_df(
    panel: pd.DataFrame,
    *,
    yvar: str,
    event_col: str,
    treated_col: str,
    calendar_col: str,
    unit_col: str = "unitid",
    relabel_col: str = "relabel_year",
    weight_col: str | None = None,
) -> pd.DataFrame:
    required = [yvar, event_col, treated_col, calendar_col, unit_col, relabel_col]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        return pd.DataFrame()
    work = panel.dropna(subset=required).copy()
    if work.empty:
        return work
    work["event_t"] = pd.to_numeric(work[event_col], errors="coerce")
    work["treated"] = pd.to_numeric(work[treated_col], errors="coerce")
    work["calendar_year"] = pd.to_numeric(work[calendar_col], errors="coerce")
    work["unitid"] = pd.to_numeric(work[unit_col], errors="coerce")
    work["relabel_year"] = pd.to_numeric(work[relabel_col], errors="coerce")
    work[yvar] = pd.to_numeric(work[yvar], errors="coerce")
    work = work.dropna(subset=["event_t", "treated", "calendar_year", "unitid", "relabel_year", yvar])
    if work.empty:
        return work
    work["event_t"] = work["event_t"].astype(int)
    work["treated"] = work["treated"].astype(int)
    work["calendar_year"] = work["calendar_year"].astype(int)
    work["unitid"] = work["unitid"].astype(int)
    work["relabel_year"] = work["relabel_year"].astype(int)
    if weight_col and weight_col in work.columns:
        work["total_grads"] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0)
    elif "total_grads" in work.columns:
        work["total_grads"] = pd.to_numeric(work["total_grads"], errors="coerce").fillna(0.0)
    else:
        work["total_grads"] = 1.0
    work = work[work["total_grads"] > 0].copy()
    if "pair_id" not in work.columns:
        work["pair_id"] = work["unitid"].astype(str)
    return work


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce").astype(float)
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    ok = y.notna() & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(y[ok], weights=w[ok]))


def _count_lookup(reg_df: pd.DataFrame) -> pd.DataFrame:
    return (
        reg_df.groupby(["event_t", "treated"], as_index=False)
        .agg(
            n_school_years=("unitid", "size"),
            n_schools=("unitid", "nunique"),
            total_grads=("total_grads", "sum"),
        )
    )


def _base_estimate_row(
    reg_df: pd.DataFrame,
    event_t: int,
    estimator: str,
    coef: float,
    se: float,
    reference_event_time: int,
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    counts = _count_lookup(reg_df)
    sub_counts = counts[counts["event_t"].eq(int(event_t))]
    treated = sub_counts[sub_counts["treated"].eq(1)]
    control = sub_counts[sub_counts["treated"].eq(0)]
    donor_control = pd.DataFrame()
    if "control_panel_role" in reg_df.columns:
        donor_control = reg_df[
            reg_df["treated"].eq(0)
            & reg_df["control_panel_role"].astype(str).eq("donor")
        ]
    row = {
        "event_t": int(event_t),
        "estimator": estimator,
        "coef": float(coef) if coef is not None else float("nan"),
        "se": float(se) if se is not None else float("nan"),
        "ci_low": float(coef) - 1.96 * float(se) if pd.notna(coef) and pd.notna(se) else float("nan"),
        "ci_high": float(coef) + 1.96 * float(se) if pd.notna(coef) and pd.notna(se) else float("nan"),
        "reference_event_t": int(reference_event_time),
        "nobs": int(len(reg_df)),
        "n_schools_total": int(reg_df["unitid"].nunique()) if "unitid" in reg_df else 0,
        "treated_n_school_years": int(treated["n_school_years"].sum()) if not treated.empty else 0,
        "control_n_school_years": (
            int(len(donor_control))
            if not donor_control.empty
            else int(control["n_school_years"].sum()) if not control.empty else 0
        ),
        "treated_n_schools": int(treated["n_schools"].sum()) if not treated.empty else 0,
        "control_n_schools": (
            int(donor_control["unitid"].nunique())
            if not donor_control.empty
            else int(control["n_schools"].sum()) if not control.empty else 0
        ),
        "treated_total_grads": float(treated["total_grads"].sum()) if not treated.empty else 0.0,
        "control_total_grads": (
            float(pd.to_numeric(donor_control["total_grads"], errors="coerce").fillna(0.0).sum())
            if not donor_control.empty
            else float(control["total_grads"].sum()) if not control.empty else 0.0
        ),
    }
    if extra:
        row.update(extra)
    return row


def _generic_package_panel(
    reg_df: pd.DataFrame,
    *,
    yvar: str,
    use_weights: bool,
    include_pair_id: bool = True,
) -> pd.DataFrame:
    if reg_df.empty:
        return pd.DataFrame()
    group_cols = ["unitid", "calendar_year", "treated", "relabel_year", "event_t"]
    for col in ("pair_id", "broad_pair_bin", "degree_type"):
        if col == "pair_id" and not include_pair_id:
            continue
        if col in reg_df.columns:
            group_cols.append(col)
    grouped = (
        reg_df.groupby(group_cols, as_index=False, dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    yvar: _weighted_mean(g[yvar], g["total_grads"]) if use_weights else float(g[yvar].mean()),
                    "total_grads": float(pd.to_numeric(g["total_grads"], errors="coerce").fillna(0.0).sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    if grouped.empty:
        return grouped
    broad = grouped.get("broad_pair_bin", pd.Series("bin", index=grouped.index)).fillna("bin").astype(str)
    degree = grouped.get("degree_type", pd.Series("degree", index=grouped.index)).fillna("degree").astype(str)
    entity_parts = [
        grouped["relabel_year"].astype(str),
        broad,
        degree,
        grouped["unitid"].astype(str),
        grouped["treated"].astype(str),
    ]
    if include_pair_id:
        pair = grouped.get("pair_id", grouped["unitid"]).fillna(grouped["unitid"]).astype(str)
        entity_parts.insert(3, pair)
    grouped["pkg_entity"] = entity_parts[0]
    for part in entity_parts[1:]:
        grouped["pkg_entity"] = grouped["pkg_entity"] + "||" + part
    grouped["pkg_entity_id"] = pd.factorize(grouped["pkg_entity"], sort=False)[0].astype("int64")
    grouped["calendar_year_ref"] = grouped["calendar_year"].astype(int)
    relabel_year = pd.to_numeric(grouped["relabel_year"], errors="coerce")
    grouped["g_pyfixest"] = relabel_year.where(grouped["treated"].eq(1), 0).fillna(0).astype(int)
    grouped["g_csdid"] = grouped["g_pyfixest"].astype(int)
    grouped["treatment_stratum"] = broad + "||" + degree
    return grouped


def _combine_stratified_dynamic_estimates(
    frames: list[pd.DataFrame],
    *,
    estimator: str,
    reference_event_time: int,
) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    rows: list[dict[str, object]] = []
    count_cols = [
        "nobs",
        "n_schools_total",
        "treated_n_school_years",
        "control_n_school_years",
        "treated_n_schools",
        "control_n_schools",
        "treated_total_grads",
        "control_total_grads",
    ]
    for event_t in sorted(pd.to_numeric(combined["event_t"], errors="coerce").dropna().astype(int).unique()):
        event_rows = combined[pd.to_numeric(combined["event_t"], errors="coerce").eq(event_t)].copy()
        usable = event_rows.dropna(subset=["coef"]).copy()
        if event_t == reference_event_time:
            coef, se = 0.0, 0.0
        elif usable.empty:
            coef, se = float("nan"), float("nan")
        else:
            w = pd.to_numeric(usable.get("treated_total_grads", pd.Series(1.0, index=usable.index)), errors="coerce").fillna(0.0).astype(float)
            if w.sum() <= 0:
                w = pd.Series(1.0, index=usable.index, dtype=float)
            weights = (w / w.sum()).to_numpy(dtype=float)
            coef_vals = pd.to_numeric(usable["coef"], errors="coerce").astype(float).to_numpy()
            coef = float(np.dot(weights, coef_vals))
            se_vals = pd.to_numeric(usable.get("se", pd.Series(np.nan, index=usable.index)), errors="coerce").fillna(0.0).astype(float).to_numpy()
            se = float(np.sqrt(np.dot(np.square(weights), np.square(se_vals))))
        row: dict[str, object] = {
            "event_t": int(event_t),
            "estimator": estimator,
            "coef": coef,
            "se": se,
            "ci_low": coef - 1.96 * se if pd.notna(coef) and pd.notna(se) else float("nan"),
            "ci_high": coef + 1.96 * se if pd.notna(coef) and pd.notna(se) else float("nan"),
            "reference_event_t": int(reference_event_time),
            "aggregation_unit": "relabel_year_x_broad_bin_x_degree",
            "n_treatment_strata": int(
                pd.to_numeric(
                    event_rows.get("n_treatment_strata", pd.Series(dtype=float)),
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )
            if "n_treatment_strata" in event_rows.columns
            else int(event_rows.get("treatment_stratum", pd.Series(dtype=object)).dropna().nunique()),
        }
        for col in count_cols:
            row[col] = float(pd.to_numeric(event_rows.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        for col in [
            "nobs",
            "n_schools_total",
            "treated_n_school_years",
            "control_n_school_years",
            "treated_n_schools",
            "control_n_schools",
        ]:
            row[col] = int(row[col])
        rows.append(row)
    return pd.DataFrame(rows)


def _sun_abraham_param_event_time(param_name: object) -> int | None:
    event_match = re.search(r"rel_time.*?\[T\.(-?\d+(?:\.\d+)?)\]", str(param_name))
    if event_match is None:
        return None
    return int(float(event_match.group(1)))


def _estimate_sun_abraham(
    reg_df: pd.DataFrame,
    *,
    yvar: str,
    reference_event_time: int,
    use_weights: bool,
) -> pd.DataFrame:
    pkg = _generic_package_panel(reg_df, yvar=yvar, use_weights=use_weights, include_pair_id=False)
    if pkg.empty:
        return pd.DataFrame()
    from pyfixest.estimation import feols

    work = pkg.copy()
    work["is_treated"] = (
        work["calendar_year_ref"].ge(work["g_pyfixest"])
        & work["g_pyfixest"].gt(0)
    ).astype(int)
    first_treated = (
        work.assign(first_treated_period=work["calendar_year_ref"] * work["is_treated"])
        .groupby("pkg_entity_id")["first_treated_period"]
        .apply(lambda x: x[x > 0].min())
        .rename("first_treated_period")
    )
    work = work.merge(first_treated, on="pkg_entity_id", how="left")
    work["rel_time"] = work["calendar_year_ref"] - work["first_treated_period"]
    work["first_treated_period"] = work["first_treated_period"].replace(np.nan, 0).astype(int)
    work["rel_time"] = work["rel_time"].replace(np.nan, np.inf)

    treated_groups = (
        work.loc[work["treated"].eq(1), ["relabel_year", "treatment_stratum"]]
        .drop_duplicates()
        .sort_values(["relabel_year", "treatment_stratum"])
        .reset_index(drop=True)
    )
    if treated_groups.empty:
        return pd.DataFrame()
    dummy_meta: dict[str, tuple[int, str]] = {}
    dummy_cols: list[str] = []
    for idx, row in enumerate(treated_groups.itertuples(index=False)):
        col = f"cohort_stratum_dummy_{idx}"
        dummy_cols.append(col)
        relabel_year = int(row.relabel_year)
        treatment_stratum = str(row.treatment_stratum)
        dummy_meta[col] = (relabel_year, treatment_stratum)
        work[col] = (
            work["treated"].eq(1)
            & work["relabel_year"].eq(relabel_year)
            & work["treatment_stratum"].astype(str).eq(treatment_stratum)
        ).astype(int)

    interaction_terms = "+".join(
        f"i(rel_time, {col}, ref = {float(reference_event_time)})"
        for col in dummy_cols
    )
    formula = (
        f"{yvar} ~ {interaction_terms} | "
        "pkg_entity_id + calendar_year_ref"
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        feols_kwargs: dict[str, object] = {
            "fml": formula,
            "data": work,
            "vcov": {"CRV1": "pkg_entity_id"},
            "copy_data": False,
            "store_data": False,
        }
        if use_weights:
            feols_kwargs["weights"] = "total_grads"
        result = feols(**feols_kwargs)
    params, cov = estimator_cmp._result_params_and_cov(result)  # noqa: SLF001
    event_weight_col = "total_grads" if use_weights else "__treated_cell_weight"
    if not use_weights:
        work["__treated_cell_weight"] = 1.0
    weight_rows = (
        work[work["treated"].eq(1)]
        .groupby(["relabel_year", "treatment_stratum", "event_t"], as_index=False)
        .agg(weight=(event_weight_col, "sum"))
    )
    weight_lookup = {
        (int(row.relabel_year), str(row.treatment_stratum), int(row.event_t)): float(row.weight)
        for row in weight_rows.itertuples(index=False)
    }
    param_meta: dict[str, tuple[str, int]] = {}
    for name in params.index:
        text = str(name)
        event_t = _sun_abraham_param_event_time(text)
        if event_t is None:
            continue
        matched_dummy = next((col for col in dummy_cols if col in text), None)
        if matched_dummy is None:
            continue
        param_meta[text] = (matched_dummy, event_t)
    rows: list[dict[str, object]] = []
    for event_t in sorted(work["event_t"].dropna().astype(int).unique()):
        if event_t == reference_event_time:
            coef, se = 0.0, 0.0
        else:
            cols = [name for name, (_, e) in param_meta.items() if e == int(event_t) and name in params.index]
            if cols:
                raw_weights = np.array(
                    [
                        weight_lookup.get(
                            (
                                dummy_meta[param_meta[col][0]][0],
                                dummy_meta[param_meta[col][0]][1],
                                int(event_t),
                            ),
                            0.0,
                        )
                        for col in cols
                    ],
                    dtype=float,
                )
                if raw_weights.sum() <= 0:
                    raw_weights = np.ones(len(cols), dtype=float)
                weights = raw_weights / raw_weights.sum()
                beta = params.reindex(cols).astype(float).to_numpy()
                coef = float(np.dot(weights, beta))
                subcov = cov.reindex(index=cols, columns=cols).fillna(0.0).to_numpy(dtype=float)
                var = float(weights @ subcov @ weights)
                se = math.sqrt(var) if var >= 0 and np.isfinite(var) else float("nan")
            else:
                coef, se = float("nan"), float("nan")
        rows.append(
            _base_estimate_row(
                work,
                int(event_t),
                "sun_abraham",
                coef,
                se,
                reference_event_time,
                extra={
                    "aggregation_unit": "relabel_year_x_broad_bin_x_degree",
                    "n_treatment_strata": int(len(cols)) if event_t != reference_event_time else int(len(dummy_cols)),
                },
            )
        )
    return pd.DataFrame(rows)


def _estimate_callaway_santanna_from_pkg(
    pkg: pd.DataFrame,
    *,
    yvar: str,
    reference_event_time: int,
    bootstrap_reps: int,
    random_seed: int,
    use_weights: bool,
) -> pd.DataFrame:
    if pkg.empty:
        return pd.DataFrame()
    from csdid.att_gt import ATTgt

    def _fallback_rows(note: str) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for event_t in sorted(pkg["event_t"].dropna().astype(int).unique()):
            coef = 0.0 if event_t == reference_event_time else float("nan")
            se = 0.0 if event_t == reference_event_time else float("nan")
            rows.append(
                _base_estimate_row(
                    pkg,
                    int(event_t),
                    "callaway_santanna",
                    coef,
                    se,
                    reference_event_time,
                    extra={
                        "treatment_stratum": (
                            str(pkg["treatment_stratum"].iloc[0])
                            if "treatment_stratum" in pkg.columns
                            and pkg["treatment_stratum"].nunique(dropna=False) == 1
                            else "pooled"
                        ),
                        "aggregation_unit": "relabel_year_x_broad_bin_x_degree",
                        "n_treatment_strata": int(
                            pkg.loc[
                                pkg["treated"].eq(1) & pkg["event_t"].eq(int(event_t)),
                                ["relabel_year", "treatment_stratum"],
                            ]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "estimation_note": note,
                    },
                )
            )
        return pd.DataFrame(rows)

    cs_df = pkg[["pkg_entity_id", "calendar_year", "unitid", "g_csdid", yvar, "total_grads"]].rename(
        columns={"pkg_entity_id": "entity_id"}
    )
    biters = max(2, int(bootstrap_reps))

    def _fit_callaway(*, bstrap: bool) -> object:
        with contextlib.redirect_stdout(io.StringIO()):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                np.random.seed(int(random_seed))
                model = ATTgt(
                    yname=yvar,
                    tname="calendar_year",
                    idname="entity_id",
                    gname="g_csdid",
                    data=cs_df,
                    control_group="nevertreated",
                    panel=True,
                    allow_unbalanced_panel=True,
                    clustervar="unitid",
                    weights_name="total_grads",
                    anticipation=max(0, abs(int(reference_event_time)) - 1),
                    cband=False,
                    biters=biters,
                )
                model.fit(est_method="dr", base_period="universal", bstrap=bstrap)
                model.aggte(
                    typec="dynamic",
                    min_e=int(pkg["event_t"].min()),
                    max_e=int(pkg["event_t"].max()),
                    na_rm=True,
                    bstrap=bstrap,
                    biters=biters,
                    cband=False,
                )
                return model

    try:
        model = _fit_callaway(bstrap=True)
        estimation_note = "ok_bootstrap"
    except Exception as exc:
        try:
            model = _fit_callaway(bstrap=False)
            estimation_note = f"fallback_analytic_se_after_{type(exc).__name__}"
        except Exception as no_boot_exc:
            return _fallback_rows(
                f"callaway_santanna_failed_{type(exc).__name__}_then_{type(no_boot_exc).__name__}"
            )
    att_lookup = {
        int(event_t): (float(att), float(se))
        for event_t, att, se in zip(
            list(model.atte.get("egt", [])),
            list(model.atte.get("att_egt", [])),
            np.asarray(model.atte.get("se_egt", [[]]), dtype=float).reshape(-1).tolist(),
        )
    }
    rows: list[dict[str, object]] = []
    for event_t in sorted(pkg["event_t"].dropna().astype(int).unique()):
        if event_t == reference_event_time:
            coef, se = 0.0, 0.0
        else:
            coef, se = att_lookup.get(int(event_t), (float("nan"), float("nan")))
        rows.append(
            _base_estimate_row(
                pkg,
                int(event_t),
                "callaway_santanna",
                coef,
                se,
                reference_event_time,
                extra={
                    "treatment_stratum": (
                        str(pkg["treatment_stratum"].iloc[0])
                        if "treatment_stratum" in pkg.columns and pkg["treatment_stratum"].nunique(dropna=False) == 1
                        else "pooled"
                    ),
                    "aggregation_unit": "relabel_year_x_broad_bin_x_degree",
                    "n_treatment_strata": int(
                        pkg.loc[
                            pkg["treated"].eq(1) & pkg["event_t"].eq(int(event_t)),
                            ["relabel_year", "treatment_stratum"],
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "estimation_note": estimation_note,
                },
            )
        )
    return pd.DataFrame(rows)


def _estimate_callaway_santanna(
    reg_df: pd.DataFrame,
    *,
    yvar: str,
    reference_event_time: int,
    bootstrap_reps: int,
    random_seed: int,
    use_weights: bool,
) -> pd.DataFrame:
    pkg = _generic_package_panel(reg_df, yvar=yvar, use_weights=use_weights)
    if pkg.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for _, stratum_pkg in pkg.groupby("treatment_stratum", sort=False, dropna=False):
        if stratum_pkg["treated"].nunique() < 2:
            continue
        estimates = _estimate_callaway_santanna_from_pkg(
            stratum_pkg.copy(),
            yvar=yvar,
            reference_event_time=reference_event_time,
            bootstrap_reps=bootstrap_reps,
            random_seed=random_seed,
            use_weights=use_weights,
        )
        if not estimates.empty:
            frames.append(estimates)
    return _combine_stratified_dynamic_estimates(
        frames,
        estimator="callaway_santanna",
        reference_event_time=reference_event_time,
    )


def estimate_dynamic_effects(
    reg_df: pd.DataFrame,
    *,
    yvar: str,
    estimation_type: str,
    reference_event_time: int,
    bootstrap_reps: int,
    random_seed: int,
    use_weights: bool,
) -> pd.DataFrame:
    if reg_df.empty:
        return pd.DataFrame()
    if estimation_type == "sun_abraham":
        return _estimate_sun_abraham(
            reg_df,
            yvar=yvar,
            reference_event_time=reference_event_time,
            use_weights=use_weights,
        )
    if estimation_type == "callaway_santanna":
        return _estimate_callaway_santanna(
            reg_df,
            yvar=yvar,
            reference_event_time=reference_event_time,
            bootstrap_reps=bootstrap_reps,
            random_seed=random_seed,
            use_weights=use_weights,
        )
    raise ValueError(f"Unsupported non-TWFE estimator: {estimation_type}")


def _pooled_from_dynamic(estimates: pd.DataFrame, reg_df: pd.DataFrame) -> tuple[float, float]:
    event_t = pd.to_numeric(estimates["event_t"], errors="coerce")
    post = estimates[event_t.between(REVELIO_POOLED_EVENT_MIN, REVELIO_POOLED_EVENT_MAX)].dropna(subset=["coef"]).copy()
    if post.empty:
        return float("nan"), float("nan")
    weights = (
        reg_df[reg_df["treated"].eq(1) & reg_df["event_t"].isin(post["event_t"].astype(int))]
        .groupby("event_t")["total_grads"]
        .sum()
    )
    w = post["event_t"].astype(int).map(weights).fillna(0.0).astype(float).to_numpy()
    if w.sum() <= 0:
        w = np.ones(len(post), dtype=float)
    w = w / w.sum()
    coef = float(np.dot(w, post["coef"].astype(float).to_numpy()))
    se_vals = pd.to_numeric(post["se"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    se = float(np.sqrt(np.dot(np.square(w), np.square(se_vals))))
    return coef, se


def _weighted_panel_mean(frame: pd.DataFrame, yvar: str) -> float:
    if frame.empty or yvar not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[yvar], errors="coerce")
    weights = pd.to_numeric(
        frame["total_grads"] if "total_grads" in frame.columns else pd.Series(1.0, index=frame.index),
        errors="coerce",
    )
    keep = values.notna()
    if not keep.any():
        return float("nan")
    values = values.loc[keep]
    weights = weights.loc[keep].fillna(0.0)
    if weights.sum() <= 0:
        return float(values.mean())
    return float(np.average(values, weights=weights))


def _pooled_effect_stats(reg_df: pd.DataFrame, *, yvar: str, coef: float) -> dict[str, float]:
    event_t = pd.to_numeric(reg_df["event_t"], errors="coerce")
    treated = pd.to_numeric(reg_df["treated"], errors="coerce")
    treated_pre = reg_df[treated.eq(1) & event_t.lt(REVELIO_POOLED_EVENT_MIN)]
    treated_post = reg_df[
        treated.eq(1) & event_t.between(REVELIO_POOLED_EVENT_MIN, REVELIO_POOLED_EVENT_MAX)
    ]
    control_pre = reg_df[treated.eq(0) & event_t.lt(REVELIO_POOLED_EVENT_MIN)]
    control_post = reg_df[
        treated.eq(0) & event_t.between(REVELIO_POOLED_EVENT_MIN, REVELIO_POOLED_EVENT_MAX)
    ]
    baseline_mean = _weighted_panel_mean(treated_pre, yvar)
    effect_size = (
        float(coef / baseline_mean)
        if pd.notna(coef) and pd.notna(baseline_mean) and not np.isclose(baseline_mean, 0.0)
        else float("nan")
    )
    return {
        "baseline_mean": baseline_mean,
        "effect_size": effect_size,
        "pooled_post_event_min": float(REVELIO_POOLED_EVENT_MIN),
        "pooled_post_event_max": float(REVELIO_POOLED_EVENT_MAX),
        "treated_pre_mean": baseline_mean,
        "treated_post_mean": _weighted_panel_mean(treated_post, yvar),
        "control_pre_mean": _weighted_panel_mean(control_pre, yvar),
        "control_post_mean": _weighted_panel_mean(control_post, yvar),
    }


def _build_revelio_variant_panels(
    *,
    config_path: Path,
    output_dir: Path,
    cache_path: Path,
    force_rebuild: bool,
    horizons: list[int],
    event_window: int,
    event_source_mode: str,
    relabel_year_min: int | None,
    relabel_year_max: int | None,
    control_group: str,
    log: RunLog,
) -> pd.DataFrame:
    if _file_ok(cache_path) and not force_rebuild:
        log.hit("revelio_final_panel")
        return _filter_revelio_relabel_years(
            pd.read_parquet(cache_path),
            relabel_year_min=relabel_year_min,
            relabel_year_max=relabel_year_max,
        )

    log.rebuild("revelio_final_panel")
    if log.verbose:
        _progress("building Revelio final panel: materializing relabels, full-sample rows, controls, and outcome panels")
    cfg_data = _read_yaml(config_path)
    paths = cfg_data.get("paths", {})
    build = cfg_data.get("build", {})
    updates = {
        "BUILD_OVERWRITE": bool(force_rebuild),
        "BUILD_EVENT_SOURCE_MODE": event_source_mode,
        "BUILD_CONTROL_GROUP": control_group,
        "BUILD_EVENT_WINDOW": int(event_window),
        "BUILD_SAMPLE_GRADYEAR_WINDOW": int(build.get("sample_gradyear_window", 5)),
        "BUILD_OUTCOME_HORIZONS": horizons,
        "BUILD_COHORT_PLOT_YVARS": DEFAULT_COHORT_YVARS + ["linkedin_match_share"],
        "BUILD_SAMPLE_CIP_PREFIXES": ["4506"] if event_source_mode == "econ_v2" else [],
        "BUILD_SAMPLE_VARIANTS": DEFAULT_SAMPLE_VARIANTS,
        "BUILD_DID_PLOT_MODE": "pooled_post_by_horizon",
        "BUILD_DID_INCLUDE_INDIVIDUAL_CONTROLS": False,
        "BUILD_DID_INCLUDE_SCHOOL_CHAR_GRADYEAR_CONTROLS": False,
        "BUILD_RUN_DID": True,
        "OUTPUT_DIR": str(output_dir),
        "OUTPUT_PANEL_PARQUET": str(cache_path),
        "OUTPUT_DID_RESULTS_PARQUET": str(cache_path.with_name(cache_path.stem + "_did.parquet")),
        "RELABELS_PARQUET": paths.get("relabels_parquet", indiv.cfg.RELABELS_PARQUET),
        "GENERALIZED_RELABELS_PANEL_PARQUET": paths.get("generalized_relabels_panel_parquet", indiv.cfg.GENERALIZED_RELABELS_PANEL_PARQUET),
        "STAGE04_MERGE_READY_PARQUET": paths.get("stage04_merge_ready_parquet", indiv.cfg.STAGE04_MERGE_READY_PARQUET),
        "STAGE05_PERSON_BASELINE_PARQUET": paths.get("stage05_person_baseline_parquet", indiv.cfg.STAGE05_PERSON_BASELINE_PARQUET),
        "FOIA_PERSON_PANEL_PARQUET": paths.get("foia_person_panel_parquet", indiv.cfg.FOIA_PERSON_PANEL_PARQUET),
        "IPEDS_COST_PANEL_PARQUET": paths.get("ipeds_cost_panel_parquet", indiv.cfg.IPEDS_COST_PANEL_PARQUET),
        "IPEDS_HD_DIR": paths.get("ipeds_hd_dir", indiv.cfg.IPEDS_HD_DIR),
        "REVELIO_IPEDS_INST_CW_PARQUET": paths.get("revelio_ipeds_inst_cw_parquet", indiv.cfg.REVELIO_IPEDS_INST_CW_PARQUET),
        "IPEDS_CROSSWALK_PARQUET": paths.get("ipeds_crosswalk_parquet", indiv.cfg.IPEDS_CROSSWALK_PARQUET),
        "REV_USERS_CORE_PARQUET": paths.get("rev_users_core_parquet", indiv.cfg.REV_USERS_CORE_PARQUET),
        "REV_EDUC_CLEAN_LONG_PARQUET": paths.get("rev_educ_clean_long_parquet", indiv.cfg.REV_EDUC_CLEAN_LONG_PARQUET),
        "REV_POS_CLEAN_LONG_PARQUET": paths.get("rev_pos_clean_long_parquet", indiv.cfg.REV_POS_CLEAN_LONG_PARQUET),
        "REV_USER_NATIONALITY_PARQUET": paths.get("rev_user_nationality_parquet", indiv.cfg.REV_USER_NATIONALITY_PARQUET),
        "BUILD_EXCLUDE_US_NATIONALS": bool(build.get("exclude_us_nationals", True)),
        "BUILD_EXCLUDE_US_COUNTRY_VALUE": str(build.get("exclude_us_country_value", "United States")),
        "BUILD_INSTITUTION_MATCH_QUALITY_GATE": bool(build.get("institution_match_quality_gate", True)),
        "BUILD_INSTITUTION_MATCH_SCORE_MIN": float(build.get("institution_match_score_min", 0.85)),
        "BUILD_INSTITUTION_ALIAS_JW_MIN": float(build.get("institution_alias_jw_min", 0.92)),
        "BUILD_RSID_SUPPORT_GATE": bool(build.get("rsid_support_gate", False)),
        "BUILD_RSID_SUPPORT_MIN_SHARE": float(build.get("rsid_support_min_share", 0.05)),
        "BUILD_RSID_SUPPORT_MIN_COUNT": int(build.get("rsid_support_min_count", 10)),
        "BUILD_FOREIGN_HETEROGENEITY": bool(build.get("foreign_heterogeneity", True)),
        "BUILD_MIN_POS_DURATION_DAYS": int(build.get("min_pos_duration_days", 1)),
        "BUILD_DID_COUNTRY_TOP_N": int(build.get("did_country_top_n", 20)),
        "BUILD_CAP_TO_LATEST_AVAILABLE_YEAR": bool(build.get("cap_to_latest_available_year", True)),
        "BUILD_COHORT_EXTERNAL_TUITION_COL": str(build.get("cohort_external_tuition_col", "tuition7")),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _patched_indiv_config(**updates):
        con = ddb.connect()
        try:
            con.sql(f"PRAGMA threads={max(1, os.cpu_count() or 1)}")
            con.sql("PRAGMA preserve_insertion_order=false")
            if log.verbose:
                _progress("Revelio panel: step 1 relabel events")
            relabel_df = indiv.step1_relabels(con)
            relabel_df = _filter_revelio_relabel_years(
                relabel_df,
                relabel_year_min=relabel_year_min,
                relabel_year_max=relabel_year_max,
            )
            if relabel_df.empty:
                raise RuntimeError(
                    "No Revelio relabel events remain after applying "
                    f"relabel_year bounds {relabel_year_min}–{relabel_year_max}."
                )
            if log.verbose:
                _progress("Revelio panel: step 2 full-sample rows")
            sample_variants = indiv.step2_prepare_stage04_samples(con)
            if log.verbose:
                _progress("Revelio panel: preparing enrichment/control views")
            indiv._ensure_enrichment_views(con)  # noqa: SLF001
            control_events = indiv.build_control_events(con, relabel_df, testing_unitids=None)
            panels: list[pd.DataFrame] = []
            variant_progress = ProgressCounter(
                "Revelio panel variants",
                len(sample_variants),
                report_every=1,
            )
            for analysis_variant in sample_variants:
                sample_view = indiv._sample_view_name(analysis_variant)  # noqa: SLF001
                treated_indiv = indiv.step3_match_treated(
                    con,
                    relabel_df,
                    sample_view,
                    analysis_variant,
                    testing_unitids=None,
                    control_events=control_events,
                )
                if treated_indiv.empty:
                    variant_progress.advance(f"{analysis_variant} (no treated rows)")
                    continue
                treated_panel = indiv.step4_build_outcome_panel(
                    con,
                    treated_indiv,
                    group_label=f"treated_{indiv._variant_slug(analysis_variant)}",  # noqa: SLF001
                    analysis_variant=analysis_variant,
                )
                control_panel = indiv.step6_control_group(
                    con,
                    sample_view,
                    control_events,
                    analysis_variant,
                )
                variant_panel = pd.concat([treated_panel, control_panel], ignore_index=True) if not control_panel.empty else treated_panel
                panels.append(indiv._finalize_variant_panel(variant_panel))  # noqa: SLF001
                variant_progress.advance(str(analysis_variant))
            if not panels:
                raise RuntimeError("No Revelio variant panels were produced.")
            combined = _filter_revelio_relabel_years(
                pd.concat(panels, ignore_index=True, sort=False),
                relabel_year_min=relabel_year_min,
                relabel_year_max=relabel_year_max,
            )
            combined.to_parquet(cache_path, index=False)
            return combined
        finally:
            con.close()


def _render_revelio_twfe(
    panel: pd.DataFrame,
    *,
    output_dir: Path,
    did_results_path: Path,
    include_controls: bool,
    output_mode: str,
    event_window: int,
    log: RunLog,
    render_plots: bool = True,
    force_rebuild: bool = False,
    progress_label: str | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force_rebuild and _file_ok(did_results_path):
        did_combined = pd.read_parquet(did_results_path)
        outcome_gap = _revelio_cached_outcome_gap(did_combined, panel)
        column_gap = _revelio_cached_column_gap(did_combined, output_mode)
        if not outcome_gap and not column_gap:
            log.hit(f"revelio_twfe_{output_mode}_{'controls' if include_controls else 'no_controls'}")
            if not _file_ok(did_results_path.with_suffix(".csv")):
                did_combined.to_csv(did_results_path.with_suffix(".csv"), index=False)
            if render_plots and not did_combined.empty:
                _plot_revelio_twfe_results(
                    did_combined,
                    output_dir=output_dir,
                    did_results_path=did_results_path,
                    output_mode=output_mode,
                )
            return did_combined
        log.rebuild(
            f"revelio_twfe_{output_mode}_{'controls' if include_controls else 'no_controls'}_missing_"
            + "_".join(outcome_gap + [f"col_{col}" for col in column_gap])
        )

    did_plot_mode = "pooled_post_by_horizon" if output_mode == "new_pooled_post" else "event_study_by_cohort"
    did_results: list[pd.DataFrame] = []
    with _patched_indiv_config(
        OUTPUT_DIR=str(output_dir),
        OUTPUT_DID_RESULTS_PARQUET=str(did_results_path),
        BUILD_DID_INCLUDE_INDIVIDUAL_CONTROLS=include_controls,
        BUILD_DID_INCLUDE_SCHOOL_CHAR_GRADYEAR_CONTROLS=include_controls,
        BUILD_DID_PLOT_MODE=did_plot_mode,
        BUILD_EVENT_WINDOW=int(event_window),
        BUILD_POOLED_POST_EVENT_MIN=REVELIO_POOLED_EVENT_MIN,
        BUILD_POOLED_POST_EVENT_MAX=REVELIO_POOLED_EVENT_MAX,
        BUILD_RUN_DID=True,
    ):
        variant_groups = list(panel.groupby("analysis_variant", sort=False))
        variant_progress = ProgressCounter(
            f"Revelio TWFE [{progress_label or output_mode}] variants",
            len(variant_groups),
            report_every=1,
        )
        for variant, variant_panel in variant_groups:
            variant_panel = variant_panel.copy()
            treated_panel = variant_panel[variant_panel["treated_ind"] == 1].copy()
            control_panel = variant_panel[variant_panel["treated_ind"] == 0].copy()
            if render_plots and not include_controls:
                indiv.step5_event_study_plots(treated_panel, label="treated", analysis_variant=str(variant))
                indiv.step7_treated_vs_control_plots(treated_panel, control_panel, analysis_variant=str(variant))
            if output_mode == "new_pooled_post":
                results = indiv.step8_pooled_post_by_horizon(treated_panel, control_panel, analysis_variant=str(variant))
            else:
                results = indiv.step8_did(treated_panel, control_panel, analysis_variant=str(variant))
            if not results.empty:
                did_results.append(results)
            if render_plots and not include_controls:
                did_results.extend(
                    indiv._run_foreign_heterogeneity_did(  # noqa: SLF001
                        variant_panel,
                        analysis_variant=str(variant),
                    )
                )
            variant_progress.advance(str(variant))
        if not did_results:
            return pd.DataFrame()
        did_combined = pd.concat([x for x in did_results if x is not None and not x.empty], ignore_index=True)
        did_results_path.parent.mkdir(parents=True, exist_ok=True)
        did_combined.to_parquet(did_results_path, index=False)
        did_combined.to_csv(did_results_path.with_suffix(".csv"), index=False)
        if render_plots and not did_combined.empty:
            _plot_revelio_twfe_results(
                did_combined,
                output_dir=output_dir,
                did_results_path=did_results_path,
                output_mode=output_mode,
            )
        log.output(did_results_path)
        return did_combined


def _render_revelio_descriptive_plots(
    panel: pd.DataFrame,
    *,
    output_dir: Path,
    output_mode: str,
    event_window: int,
) -> None:
    did_plot_mode = "pooled_post_by_horizon" if output_mode == "new_pooled_post" else "event_study_by_cohort"
    with _patched_indiv_config(
        OUTPUT_DIR=str(output_dir),
        BUILD_DID_INCLUDE_INDIVIDUAL_CONTROLS=False,
        BUILD_DID_INCLUDE_SCHOOL_CHAR_GRADYEAR_CONTROLS=False,
        BUILD_DID_PLOT_MODE=did_plot_mode,
        BUILD_EVENT_WINDOW=int(event_window),
        BUILD_POOLED_POST_EVENT_MIN=REVELIO_POOLED_EVENT_MIN,
        BUILD_POOLED_POST_EVENT_MAX=REVELIO_POOLED_EVENT_MAX,
        BUILD_RUN_DID=True,
    ):
        variant_groups = list(panel.groupby("analysis_variant", sort=False))
        variant_progress = ProgressCounter(
            "Revelio descriptive plot variants",
            len(variant_groups),
            report_every=1,
        )
        for variant, variant_panel in variant_groups:
            variant_panel = variant_panel.copy()
            treated_panel = variant_panel[variant_panel["treated_ind"] == 1].copy()
            control_panel = variant_panel[variant_panel["treated_ind"] == 0].copy()
            indiv.step5_event_study_plots(treated_panel, label="treated", analysis_variant=str(variant))
            indiv.step7_treated_vs_control_plots(treated_panel, control_panel, analysis_variant=str(variant))
            variant_progress.advance(str(variant))


def _revelio_stage04_foreign_split(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "analysis_variant" not in results.columns:
        return pd.DataFrame()
    keep = {"stage04_all_foreign", "stage04_all_non_foreign"}
    return results[results["analysis_variant"].astype(str).isin(keep)].copy()


def _revelio_foia_and_stage04_foreign_split(base_results: pd.DataFrame, heterogeneity_results: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not base_results.empty and "analysis_variant" in base_results.columns:
        linked = base_results[base_results["analysis_variant"].astype(str).eq("foia_linked_person_baseline")].copy()
        if not linked.empty:
            frames.append(linked)
    stage04_split = _revelio_stage04_foreign_split(heterogeneity_results)
    if not stage04_split.empty:
        frames.append(stage04_split)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _plot_revelio_twfe_results(
    did_combined: pd.DataFrame,
    *,
    output_dir: Path,
    did_results_path: Path,
    output_mode: str,
) -> None:
    if did_combined.empty:
        return
    with _patched_indiv_config(OUTPUT_DIR=str(output_dir), OUTPUT_DID_RESULTS_PARQUET=str(did_results_path)):
        base_compare = did_combined[
            ~did_combined["analysis_variant"].map(indiv._is_foreign_heterogeneity_variant)  # noqa: SLF001
        ].copy()
        heterogeneity_compare = did_combined[
            did_combined["analysis_variant"].map(indiv._is_foreign_heterogeneity_variant)  # noqa: SLF001
        ].copy()
        full_sample_foia_split = _revelio_foia_and_stage04_foreign_split(base_compare, heterogeneity_compare)
        main_overlay_compare = pd.concat(
            [base_compare, full_sample_foia_split],
            ignore_index=True,
            sort=False,
        )
        if output_mode == "new_pooled_post":
            main_overlay_ylim = indiv._variant_comparison_ylim_map(  # noqa: SLF001
                main_overlay_compare,
                group_cols=("did_model", "outcome"),
            )
            for variant in base_compare["analysis_variant"].dropna().astype(str).unique().tolist():
                indiv._plot_pooled_horizon_profile(  # noqa: SLF001
                    base_compare,
                    analysis_variant=variant,
                    file_tag=indiv._variant_slug(variant),  # noqa: SLF001
                    title_label=indiv._analysis_variant_label(variant),  # noqa: SLF001
                )
                if variant == "foia_linked_person_baseline":
                    indiv._plot_pooled_horizon_profile(  # noqa: SLF001
                        base_compare,
                        analysis_variant=variant,
                        file_tag="foia_linked_horizons",
                        title_label="FOIA-linked sample",
                    )
            indiv._plot_pooled_variant_comparison(  # noqa: SLF001
                base_compare,
                file_tag="variant",
                title_label="match sample",
                ylim_by_key=main_overlay_ylim,
            )
            indiv._plot_pooled_variant_comparison(  # noqa: SLF001
                heterogeneity_compare,
                file_tag="foreign_status",
                title_label="imputed foreign status and match sample",
            )
            indiv._plot_pooled_variant_comparison(  # noqa: SLF001
                _revelio_stage04_foreign_split(heterogeneity_compare),
                file_tag="full_sample_foreign_split",
                title_label="full-sample foreign status",
            )
            indiv._plot_pooled_variant_comparison(  # noqa: SLF001
                full_sample_foia_split,
                file_tag="full_sample_foia_split",
                title_label="FOIA-linked and full-sample foreign status",
                ylim_by_key=main_overlay_ylim,
            )
        else:
            main_overlay_ylim = indiv._variant_comparison_ylim_map(  # noqa: SLF001
                main_overlay_compare,
                group_cols=("did_model", "horizon_years", "outcome"),
            )
            indiv._plot_did_variant_comparison(base_compare, ylim_by_key=main_overlay_ylim)  # noqa: SLF001
            indiv._plot_did_horizon_comparison(  # noqa: SLF001
                base_compare,
                analysis_variant="foia_linked_person_baseline",
                file_tag="foia_linked_horizons",
                title_label="FOIA-linked sample",
            )
            indiv._plot_did_variant_comparison(  # noqa: SLF001
                heterogeneity_compare,
                file_tag="foreign_status",
                title_label="imputed foreign status and match sample",
            )
            indiv._plot_did_variant_comparison(  # noqa: SLF001
                _revelio_stage04_foreign_split(heterogeneity_compare),
                file_tag="full_sample_foreign_split",
                title_label="full-sample foreign status",
            )
            indiv._plot_did_variant_comparison(  # noqa: SLF001
                full_sample_foia_split,
                file_tag="full_sample_foia_split",
                title_label="FOIA-linked and full-sample foreign status",
                ylim_by_key=main_overlay_ylim,
            )


def _collapse_revelio_for_generic(panel: pd.DataFrame, *, outcome: str, horizon: int, event_window: int) -> pd.DataFrame:
    base = panel.copy()
    if "target_year_observed" in base.columns:
        base = base[pd.to_numeric(base["target_year_observed"], errors="coerce").eq(1)].copy()
    base = base[pd.to_numeric(base["horizon_years"], errors="coerce").eq(int(horizon))].copy()
    base = base[pd.to_numeric(base["cohort_t"], errors="coerce").between(-event_window, event_window)].copy()
    if base.empty or outcome not in base.columns:
        return pd.DataFrame()
    group_cols = ["analysis_variant", "unitid", "relabel_year", "treated_ind", "cohort_t", "grad_year"]
    for col in ("event_id", "broad_pair_bin", "degree_type"):
        if col in base.columns:
            group_cols.append(col)
    grouped = (
        base.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            **{
                outcome: (outcome, "mean"),
                "total_grads": ("user_id", "nunique"),
            }
        )
    )
    return _prepare_generic_regression_df(
        grouped,
        yvar=outcome,
        event_col="cohort_t",
        treated_col="treated_ind",
        calendar_col="grad_year",
        weight_col="total_grads",
    )


def _iter_revelio_generic_estimation_panels(
    panel: pd.DataFrame,
    *,
    include_foreign_heterogeneity: bool,
) -> Iterator[tuple[str, pd.DataFrame]]:
    for variant, variant_panel in panel.groupby("analysis_variant", sort=False):
        variant_name = str(variant)
        variant_panel = variant_panel.copy()
        yield variant_name, variant_panel
        if not include_foreign_heterogeneity or "imputed_foreign_ind" not in variant_panel.columns:
            continue
        foreign_flag = pd.to_numeric(variant_panel["imputed_foreign_ind"], errors="coerce")
        for value, suffix in ((1, "foreign"), (0, "non_foreign")):
            subgroup = variant_panel[foreign_flag.eq(value)].copy()
            if subgroup.empty:
                continue
            if subgroup["treated_ind"].nunique() < 2:
                continue
            yield f"{variant_name}_{suffix}", subgroup


def _render_revelio_generic(
    panel: pd.DataFrame,
    *,
    output_dir: Path,
    results_path: Path,
    estimation_type: str,
    output_mode: str,
    event_window: int,
    bootstrap_reps: int,
    random_seed: int,
    log: RunLog,
    render_plots: bool = True,
    force_rebuild: bool = False,
    progress_label: str | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force_rebuild and _file_ok(results_path):
        results = pd.read_parquet(results_path)
        outcome_gap = _revelio_cached_outcome_gap(results, panel)
        variant_gap = _revelio_cached_variant_gap(results, panel) if render_plots else []
        column_gap = _revelio_cached_column_gap(results, output_mode)
        if not outcome_gap and not variant_gap and not column_gap:
            log.hit(f"revelio_{estimation_type}_main_results")
            if not _file_ok(results_path.with_suffix(".csv")):
                results.to_csv(results_path.with_suffix(".csv"), index=False)
            if render_plots and not results.empty:
                _render_revelio_descriptive_plots(
                    panel,
                    output_dir=output_dir,
                    output_mode=output_mode,
                    event_window=event_window,
                )
                _plot_revelio_generic_results(
                    results,
                    output_dir=output_dir,
                    results_path=results_path,
                    output_mode=output_mode,
                )
            return results
        log.rebuild(
            f"revelio_{estimation_type}_main_results_missing_"
            + "_".join(
                outcome_gap
                + [f"variant_{v}" for v in variant_gap]
                + [f"col_{col}" for col in column_gap]
            )
        )

    if render_plots:
        _render_revelio_descriptive_plots(
            panel,
            output_dir=output_dir,
            output_mode=output_mode,
            event_window=event_window,
        )

    rows: list[dict[str, object]] = []
    progress_prefix = f"Revelio {estimation_type}"
    if progress_label:
        progress_prefix += f" [{progress_label}]"
    estimation_tasks: list[tuple[str, pd.DataFrame, str, int]] = []
    for variant, variant_panel in _iter_revelio_generic_estimation_panels(
        panel,
        include_foreign_heterogeneity=render_plots,
    ):
        for outcome in indiv.OUTCOMES:
            if outcome not in variant_panel.columns:
                continue
            horizons = sorted(pd.to_numeric(variant_panel["horizon_years"], errors="coerce").dropna().astype(int).unique())
            estimation_tasks.extend((variant, variant_panel, outcome, int(horizon)) for horizon in horizons)
    task_progress = ProgressCounter(progress_prefix, len(estimation_tasks))
    for variant, variant_panel, outcome, horizon in estimation_tasks:
        reg_df = _collapse_revelio_for_generic(
            variant_panel,
            outcome=outcome,
            horizon=int(horizon),
            event_window=event_window,
        )
        if reg_df.empty:
            task_progress.advance(f"{variant}/{outcome}/h{horizon} (empty)")
            continue
        estimates = estimate_dynamic_effects(
            reg_df,
            yvar=outcome,
            estimation_type=estimation_type,
            reference_event_time=-2,
            bootstrap_reps=bootstrap_reps,
            random_seed=random_seed,
            use_weights=True,
        )
        if estimates.empty:
            task_progress.advance(f"{variant}/{outcome}/h{horizon} (no estimates)")
            continue
        if output_mode == "old_event_time":
            plot_df = estimates.copy()
            plot_df["analysis_variant"] = variant
            plot_df["did_model"] = estimation_type
            plot_df["did_estimator"] = estimation_type
            plot_df["outcome"] = outcome
            plot_df["horizon_years"] = int(horizon)
            plot_df["cohort_t"] = plot_df["event_t"]
            rows.extend(plot_df.to_dict("records"))
        else:
            coef, se = _pooled_from_dynamic(estimates, reg_df)
            stats = _pooled_effect_stats(reg_df, yvar=outcome, coef=coef)
            rows.append(
                {
                    "analysis_variant": variant,
                    "did_model": estimation_type,
                    "did_estimator": estimation_type,
                    "outcome": outcome,
                    "horizon_years": int(horizon),
                    "coef": coef,
                    "se": se,
                    "ci_lower": coef - 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan,
                    "ci_upper": coef + 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan,
                    **stats,
                    "n_obs": int(len(reg_df)),
                    "n_unitids": int(reg_df["unitid"].nunique()),
                }
            )
        task_progress.advance(f"{variant}/{outcome}/h{horizon}")
    results = pd.DataFrame(rows)
    if results.empty:
        return results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_path, index=False)
    results.to_csv(results_path.with_suffix(".csv"), index=False)
    if render_plots:
        _plot_revelio_generic_results(
            results,
            output_dir=output_dir,
            results_path=results_path,
            output_mode=output_mode,
        )
    log.output(results_path)
    return results


def _plot_revelio_generic_results(
    results: pd.DataFrame,
    *,
    output_dir: Path,
    results_path: Path,
    output_mode: str,
) -> None:
    with _patched_indiv_config(OUTPUT_DIR=str(output_dir), OUTPUT_DID_RESULTS_PARQUET=str(results_path)):
        base_compare = results[
            ~results["analysis_variant"].map(indiv._is_foreign_heterogeneity_variant)  # noqa: SLF001
        ].copy()
        heterogeneity_compare = results[
            results["analysis_variant"].map(indiv._is_foreign_heterogeneity_variant)  # noqa: SLF001
        ].copy()
        full_sample_foia_split = _revelio_foia_and_stage04_foreign_split(base_compare, heterogeneity_compare)
        main_overlay_compare = pd.concat(
            [base_compare, full_sample_foia_split],
            ignore_index=True,
            sort=False,
        )
        if output_mode == "new_pooled_post":
            main_overlay_ylim = indiv._variant_comparison_ylim_map(  # noqa: SLF001
                main_overlay_compare,
                group_cols=("did_model", "outcome"),
            )
            for variant in base_compare["analysis_variant"].dropna().astype(str).unique().tolist():
                indiv._plot_pooled_horizon_profile(  # noqa: SLF001
                    base_compare,
                    analysis_variant=variant,
                    file_tag=indiv._variant_slug(variant),  # noqa: SLF001
                    title_label=indiv._analysis_variant_label(variant),  # noqa: SLF001
                )
                if variant == "foia_linked_person_baseline":
                    indiv._plot_pooled_horizon_profile(  # noqa: SLF001
                        base_compare,
                        analysis_variant=variant,
                        file_tag="foia_linked_horizons",
                        title_label="FOIA-linked sample",
                    )
            indiv._plot_pooled_variant_comparison(  # noqa: SLF001
                base_compare,
                file_tag="variant",
                title_label="match sample",
                ylim_by_key=main_overlay_ylim,
            )
            indiv._plot_pooled_variant_comparison(  # noqa: SLF001
                heterogeneity_compare,
                file_tag="foreign_status",
                title_label="imputed foreign status and match sample",
            )
            indiv._plot_pooled_variant_comparison(  # noqa: SLF001
                _revelio_stage04_foreign_split(heterogeneity_compare),
                file_tag="full_sample_foreign_split",
                title_label="full-sample foreign status",
            )
            indiv._plot_pooled_variant_comparison(  # noqa: SLF001
                full_sample_foia_split,
                file_tag="full_sample_foia_split",
                title_label="FOIA-linked and full-sample foreign status",
                ylim_by_key=main_overlay_ylim,
            )
        else:
            main_overlay_ylim = indiv._variant_comparison_ylim_map(  # noqa: SLF001
                main_overlay_compare,
                group_cols=("did_model", "horizon_years", "outcome"),
            )
            indiv._plot_did_variant_comparison(base_compare, ylim_by_key=main_overlay_ylim)  # noqa: SLF001
            indiv._plot_did_horizon_comparison(  # noqa: SLF001
                base_compare,
                analysis_variant="foia_linked_person_baseline",
                file_tag="foia_linked_horizons",
                title_label="FOIA-linked sample",
            )
            indiv._plot_did_variant_comparison(  # noqa: SLF001
                heterogeneity_compare,
                file_tag="foreign_status",
                title_label="imputed foreign status and match sample",
            )
            indiv._plot_did_variant_comparison(  # noqa: SLF001
                _revelio_stage04_foreign_split(heterogeneity_compare),
                file_tag="full_sample_foreign_split",
                title_label="full-sample foreign status",
            )
            indiv._plot_did_variant_comparison(  # noqa: SLF001
                full_sample_foia_split,
                file_tag="full_sample_foia_split",
                title_label="FOIA-linked and full-sample foreign status",
                ylim_by_key=main_overlay_ylim,
            )


def _plot_revelio_control_comparison_results(
    results: pd.DataFrame,
    *,
    output_dir: Path,
    output_mode: str,
    sample_label: str,
) -> None:
    if results.empty or "control_group" not in results.columns:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    work = results.copy()
    if output_mode == "new_pooled_post":
        x_col = "horizon_years"
        x_label = "Calendar year relative to graduation"
        file_suffix = ""
        excluded = set(getattr(indiv, "HORIZON_PROFILE_EXCLUDED_OUTCOMES", set()))
        work = work[~work["outcome"].isin(excluded)].copy()
    else:
        x_col = "cohort_t" if "cohort_t" in work.columns else "event_t"
        x_label = "Graduation Cohort Relative to Relabel Event"
        file_suffix = "_t3"
        if "horizon_years" in work.columns:
            work = work[pd.to_numeric(work["horizon_years"], errors="coerce").eq(3)].copy()
    if work.empty or x_col not in work.columns:
        return

    outcomes = [out for out in indiv.OUTCOMES if out in set(work["outcome"])]
    variant_order = list(dict.fromkeys(work["analysis_variant"].dropna().astype(str).tolist()))
    control_order = [cg for cg in REVELIO_MAIN_CONTROL_GROUPS if cg in set(work["control_group"])]
    control_order += [cg for cg in work["control_group"].dropna().astype(str).unique().tolist() if cg not in control_order]
    line_styles = {
        generalized.CONTROL_GROUP_NEVER_TREATED: "-",
        generalized.CONTROL_GROUP_ALWAYS_STEM: "--",
        generalized.CONTROL_GROUP_LATE_TREATED: "-.",
    }
    offsets = np.linspace(-0.16, 0.16, num=max(1, len(control_order) * max(1, len(variant_order))))

    for outcome in outcomes:
        outcome_df = work[work["outcome"].eq(outcome)].dropna(subset=[x_col, "coef", "se"]).copy()
        if outcome_df.empty:
            continue
        llstyle.apply_style()
        fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
        plotted = False
        offset_idx = 0
        ticks = sorted(pd.to_numeric(outcome_df[x_col], errors="coerce").dropna().astype(int).unique().tolist())
        for control_group in control_order:
            for variant_idx, variant in enumerate(variant_order):
                sub = outcome_df[
                    outcome_df["control_group"].eq(control_group)
                    & outcome_df["analysis_variant"].astype(str).eq(str(variant))
                ].copy()
                if sub.empty:
                    offset_idx += 1
                    continue
                line_df = sub.sort_values(x_col)
                color = indiv._analysis_variant_color(variant)  # noqa: SLF001
                marker = REVELIO_SAMPLE_MARKERS.get(str(variant), "o")
                linestyle = line_styles.get(control_group, "-")
                x_vals = pd.to_numeric(line_df[x_col], errors="coerce").astype(float).to_numpy()
                x_vals = x_vals + float(offsets[offset_idx]) if len(offsets) else x_vals
                offset_idx += 1
                label = (
                    f"{indiv._analysis_variant_label(variant)} "  # noqa: SLF001
                    f"({FOIA_CONTROL_GROUP_LABELS.get(control_group, control_group)})"
                )
                errorbar_container = ax.errorbar(
                    x_vals,
                    pd.to_numeric(line_df["coef"], errors="coerce").astype(float).to_numpy(),
                    yerr=1.96 * pd.to_numeric(line_df["se"], errors="coerce").astype(float).to_numpy(),
                    fmt=linestyle,
                    color=color,
                    ecolor=llstyle.rgba(color, indiv.DI_D_ERRORBAR_ALPHA),
                    elinewidth=indiv.DI_D_PLOT_MARKER_SIZE,
                    capsize=0,
                    marker=marker,
                    markersize=indiv.DI_D_PLOT_MARKER_SIZE,
                    linewidth=1.5,
                    label=label,
                )
                indiv._soften_errorbar_interval(errorbar_container)  # noqa: SLF001
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        if output_mode != "new_pooled_post":
            ref = -2
            ax.scatter([ref], [0.0], facecolors="white", edgecolors="black", linewidths=1.3, s=65, zorder=4)
            ax.axvline(x=indiv.DI_D_EVENT_LINE_X, linestyle=":", color="gray", linewidth=1)
        ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
        if ticks:
            ax.set_xticks(ticks)
        ax.set_xlabel(x_label, fontsize=indiv.DI_D_PLOT_FONT_SIZE)
        ax.set_ylabel(f"did coef: {indiv.OUTCOME_LABELS.get(outcome, outcome)}", fontsize=indiv.DI_D_PLOT_FONT_SIZE)
        ax.set_title("")
        llstyle.right_legend(ax)
        out_path = output_dir / f"did_att_by_variant_{indiv.OUTCOME_FILE_LABELS.get(outcome, _slug(outcome))}{file_suffix}.png"
        llstyle.savefig(fig, out_path, dpi=150)
        outcome_df.to_csv(out_path.with_suffix(".csv"), index=False)


def _revelio_estimator_cache_path(args: argparse.Namespace, run_hash: str, estimation_type: str) -> Path:
    return args.cache_dir / f"revelio_econ_{estimation_type}_{ESTIMATOR_AGGREGATION_VERSION}_{args.revelio_output}_{run_hash}.parquet"


def _ensure_revelio_estimator_cache(
    panel: pd.DataFrame,
    *,
    args: argparse.Namespace,
    run_hash: str,
    estimation_type: str,
    log: RunLog,
) -> Path:
    cache_path = _revelio_estimator_cache_path(args, run_hash, estimation_type)
    force_estimator = _force_estimator_rebuild(args)
    rebuilt_name = f"revelio_{estimation_type}_estimator_rows"
    if force_estimator and _file_ok(cache_path) and rebuilt_name in log.rebuilt:
        log.hit(f"{rebuilt_name}_in_process")
        return cache_path
    if not force_estimator and _file_ok(cache_path) and _file_ok(cache_path.with_suffix(".csv")):
        log.hit(rebuilt_name)
        return cache_path
    log.rebuild(rebuilt_name)
    if estimation_type == "twfe":
        _render_revelio_twfe(
            panel,
            output_dir=args.cache_dir / "revelio_estimator_comparison_plot_scratch" / "twfe",
            did_results_path=cache_path,
            include_controls=False,
            output_mode=args.revelio_output,
            event_window=args.event_window,
            log=log,
            render_plots=False,
            force_rebuild=force_estimator,
            progress_label=f"estimator-cache/{args.revelio_output}",
        )
    else:
        validate_estimator_dependencies(estimation_type)
        _render_revelio_generic(
            panel,
            output_dir=args.cache_dir / "revelio_estimator_comparison_plot_scratch" / estimation_type,
            results_path=cache_path,
            estimation_type=estimation_type,
            output_mode=args.revelio_output,
            event_window=args.event_window,
            bootstrap_reps=args.bootstrap_reps,
            random_seed=args.random_seed,
            log=log,
            render_plots=False,
            force_rebuild=force_estimator,
            progress_label=f"estimator-cache/{args.revelio_output}",
        )
    if not _file_ok(cache_path.with_suffix(".csv")):
        raise RuntimeError(f"No Revelio {estimation_type} estimator cache was produced at {cache_path.with_suffix('.csv')}")
    return cache_path


def _seed_revelio_estimator_cache(
    results: pd.DataFrame,
    *,
    args: argparse.Namespace,
    run_hash: str,
    estimation_type: str,
    log: RunLog,
) -> None:
    if results.empty:
        return
    cache_path = _revelio_estimator_cache_path(args, run_hash, estimation_type)
    rebuilt_name = f"revelio_{estimation_type}_estimator_rows"
    if not _force_estimator_rebuild(args) and _file_ok(cache_path) and _file_ok(cache_path.with_suffix(".csv")):
        log.hit(f"{rebuilt_name}_from_main")
        return
    if rebuilt_name not in log.rebuilt:
        log.rebuild(rebuilt_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(cache_path, index=False)
    results.to_csv(cache_path.with_suffix(".csv"), index=False)
    log.output(cache_path)


def _load_revelio_estimation_rows(args: argparse.Namespace, run_hash: str, outcome: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for estimation_type in ESTIMATOR_COMPARISON_TYPES:
        path = _revelio_estimator_cache_path(args, run_hash, estimation_type).with_suffix(".csv")
        try:
            data = pd.read_csv(path)
        except Exception:
            continue
        if data.empty or "outcome" not in data.columns:
            continue
        data = data[data["outcome"].eq(outcome)].copy()
        if data.empty:
            continue
        if "analysis_variant" in data.columns:
            data = data[data["analysis_variant"].eq("stage04_all")].copy()
        if data.empty:
            continue
        data["estimator"] = estimation_type
        if args.revelio_output == "new_pooled_post":
            if "horizon_years" not in data.columns:
                continue
            keep = ["horizon_years", "estimator", "coef"] + (["se"] if "se" in data.columns else [])
        else:
            if "cohort_t" not in data.columns:
                continue
            data = data.rename(columns={"cohort_t": "event_t"})
            keep = ["event_t", "estimator", "coef"] + (["se"] if "se" in data.columns else [])
        frames.append(data[keep])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["estimator"] = out["estimator"].map(_normalize_estimator_name)
    out = out[out["estimator"].isin(ESTIMATOR_COMPARISON_TYPES)].copy()
    out["se"] = pd.to_numeric(out.get("se", np.nan), errors="coerce")
    return out


def _ensure_revelio_estimation_type_appendices(panel: pd.DataFrame, args: argparse.Namespace, run_hash: str, log: RunLog) -> None:
    targets = [
        _estimation_type_path(args.revelio_econ_output_dir, f"did_att_{file_label}")
        for _, _, file_label, _ in REVELIO_ESTIMATION_APPENDICES
    ]
    estimator_caches = [
        _revelio_estimator_cache_path(args, run_hash, estimator).with_suffix(".csv")
        for estimator in ESTIMATOR_COMPARISON_TYPES
    ]
    stale_targets = any(_any_source_newer(target, estimator_caches) for target in targets)
    force_estimator = _force_estimator_rebuild(args)
    if not force_estimator and not _missing_assets(targets) and not _missing_assets(estimator_caches) and not stale_targets:
        log.hit("revelio_estimation_type_appendices")
        return
    log.rebuild("revelio_estimation_type_appendices")
    estimator_progress = ProgressCounter(
        "Revelio estimator caches",
        len(ESTIMATOR_COMPARISON_TYPES),
        report_every=1,
    )
    for estimator in ESTIMATOR_COMPARISON_TYPES:
        _ensure_revelio_estimator_cache(panel, args=args, run_hash=run_hash, estimation_type=estimator, log=log)
        estimator_progress.advance(estimator)
    appendix_progress = ProgressCounter(
        "Revelio estimator appendices",
        len(REVELIO_ESTIMATION_APPENDICES),
    )
    for _, outcome, file_label, title in REVELIO_ESTIMATION_APPENDICES:
        if (
            args.revelio_output == "new_pooled_post"
            and outcome in getattr(indiv, "HORIZON_PROFILE_EXCLUDED_OUTCOMES", set())
        ):
            log.skip(f"revelio_{file_label}_pooled_estimation_type_appendix")
            appendix_progress.advance(f"{file_label} (skipped)")
            continue
        out_path = _estimation_type_path(args.revelio_econ_output_dir, f"did_att_{file_label}")
        if not force_estimator and _file_ok(out_path) and not _missing_assets(estimator_caches) and not _any_source_newer(out_path, estimator_caches):
            appendix_progress.advance(f"{file_label} (cached)")
            continue
        rows = _load_revelio_estimation_rows(args, run_hash, outcome)
        x_col = "horizon_years" if args.revelio_output == "new_pooled_post" else "event_t"
        x_label = "Years after graduation" if x_col == "horizon_years" else "Graduation cohort relative to relabel event"
        _plot_estimation_type_comparison(
            rows,
            out_path=out_path,
            title=f"{title}: by estimation type",
            x_col=x_col,
            x_label=x_label,
        )
        log.output(out_path)
        appendix_progress.advance(file_label)


def _format_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{int(value):,}"


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{100.0 * float(value):.1f}\\%"


def _format_num(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def write_revelio_matching_table(panel: pd.DataFrame, out_path: Path) -> Path:
    work = panel.copy()
    if "target_year_observed" in work.columns:
        work = work[pd.to_numeric(work["target_year_observed"], errors="coerce").eq(1)].copy()
    if "horizon_years" in work.columns and 3 in set(pd.to_numeric(work["horizon_years"], errors="coerce").dropna().astype(int)):
        work = work[pd.to_numeric(work["horizon_years"], errors="coerce").eq(3)].copy()

    def summarize(sub: pd.DataFrame, label: str) -> dict[str, object]:
        if sub.empty:
            return {
                "Sample": label,
                "Matches": 0,
                "Users": 0,
                "Schools": 0,
                "Treated": 0,
                "Control": 0,
                "Foreign": np.nan,
                "Median score": np.nan,
                "p10 score": np.nan,
                "ge85": np.nan,
                "lt75": np.nan,
                "rsid_present": np.nan,
                "rsid_support": np.nan,
            }
        score = pd.to_numeric(sub.get("school_match_score", pd.Series(index=sub.index, dtype=float)), errors="coerce")
        rsid_count = pd.to_numeric(sub.get("rsid_unitid_match_count", pd.Series(index=sub.index, dtype=float)), errors="coerce")
        rsid_req = pd.to_numeric(sub.get("rsid_unitid_required_count", pd.Series(index=sub.index, dtype=float)), errors="coerce")
        foreign = pd.to_numeric(sub.get("imputed_foreign_ind", pd.Series(index=sub.index, dtype=float)), errors="coerce")
        return {
            "Sample": label,
            "Matches": len(sub),
            "Users": sub["user_id"].nunique() if "user_id" in sub else len(sub),
            "Schools": sub["unitid"].nunique() if "unitid" in sub else 0,
            "Treated": int(pd.to_numeric(sub["treated_ind"], errors="coerce").eq(1).sum()) if "treated_ind" in sub else 0,
            "Control": int(pd.to_numeric(sub["treated_ind"], errors="coerce").eq(0).sum()) if "treated_ind" in sub else 0,
            "Foreign": float(foreign.mean()) if foreign.notna().any() else np.nan,
            "Median score": float(score.median()) if score.notna().any() else np.nan,
            "p10 score": float(score.quantile(0.10)) if score.notna().any() else np.nan,
            "ge85": float(score.ge(0.85).mean()) if score.notna().any() else np.nan,
            "lt75": float(score.lt(0.75).mean()) if score.notna().any() else np.nan,
            "rsid_present": float(sub["rsid"].notna().mean()) if "rsid" in sub else np.nan,
            "rsid_support": float(rsid_count.ge(rsid_req).mean()) if rsid_count.notna().any() and rsid_req.notna().any() else np.nan,
        }

    rows: list[dict[str, object]] = []
    labels = {
        "stage04_all": "Full-sample",
        "foia_linked_person_baseline": "FOIA-linked",
    }
    for variant, label in labels.items():
        rows.append(summarize(work[work["analysis_variant"].eq(variant)], label))
    if "imputed_foreign_ind" in work.columns:
        rows.append(summarize(work[work["analysis_variant"].eq("stage04_all") & pd.to_numeric(work["imputed_foreign_ind"], errors="coerce").eq(1)], "Full-sample: foreign"))
        rows.append(summarize(work[work["analysis_variant"].eq("stage04_all") & pd.to_numeric(work["imputed_foreign_ind"], errors="coerce").eq(0)], "Full-sample: non-foreign"))
        rows.append(summarize(work[work["analysis_variant"].eq("foia_linked_person_baseline") & pd.to_numeric(work["imputed_foreign_ind"], errors="coerce").eq(1)], "Linked: foreign"))
    summary = pd.DataFrame(rows)
    lines = [
        "\\begingroup",
        "\\setlength{\\tabcolsep}{4.2pt}",
        "\\renewcommand{\\arraystretch}{1.16}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Sample & Matches & Users & Schools & Treated & Control & Foreign \\\\",
        "\\midrule",
    ]
    for idx, row in summary.iterrows():
        if idx == 2:
            lines.append("\\midrule")
        lines.append(
            f"{row['Sample']} & {_format_int(row['Matches'])} & {_format_int(row['Users'])} & "
            f"{_format_int(row['Schools'])} & {_format_int(row['Treated'])} & {_format_int(row['Control'])} & "
            f"{_format_pct(row['Foreign'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{0.65em}",
            "",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Sample & Median score & $p_{10}$ score & $\\geq$0.85 & $<$0.75 & RSID present & RSID support \\\\",
            "\\midrule",
        ]
    )
    for idx, row in summary.iterrows():
        if idx == 2:
            lines.append("\\midrule")
        lines.append(
            f"{row['Sample']} & {_format_num(row['Median score'])} & {_format_num(row['p10 score'])} & "
            f"{_format_pct(row['ge85'])} & {_format_pct(row['lt75'])} & "
            f"{_format_pct(row['rsid_present'])} & {_format_pct(row['rsid_support'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\endgroup",
            "",
            "\\vspace{0.35em}",
            "{\\tiny Notes: Econ-only preferred sample. Matches are deduplicated user $\\times$ event rows at the three-year outcome horizon with observed target years. Scores are full-sample institution match scores when available after the institution-quality gate. RSID support reports the share that would pass the strict RSID support diagnostic. Foreign is FOIA-linked F-1 or observed non-US Revelio origin country.}",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def _run_generalized_foia(
    *,
    args: argparse.Namespace,
    slide_assets: list[Path],
    log: RunLog,
) -> None:
    foia_assets = [
        path for path in _asset_subset(slide_assets, "generalized_relabels_plots")
        if "estimation_type_appendix" not in str(path)
    ]
    panel_path = args.foia_output_dir / "generalized_relabels_panel.parquet"
    events_path = args.foia_output_dir / "generalized_relabels_events.parquet"
    audit_path = args.foia_output_dir / "generalized_relabels_candidate_audit.csv"
    report_path = args.foia_output_dir / "generalized_relabels_report.txt"
    foia_sources = [Path(generalized.__file__).resolve()]
    if (
        not args.force_rebuild
        and _file_ok(panel_path)
        and not _missing_assets(foia_assets)
        and not _stale_assets(foia_assets, foia_sources)
    ):
        log.hit("generalized_foia_assets")
        _ensure_foia_stem_cip_assets(args, log)
        return
    log.rebuild("generalized_foia_assets")
    generalized.run_pipeline(
        events_parquet=events_path,
        events_csv=events_path.with_suffix(".csv"),
        panel_parquet=panel_path,
        report_path=report_path,
        candidate_audit_csv=audit_path,
        plots_dir=args.foia_plots_dir,
        estimator=generalized.ESTIMATOR_DID,
        # Slide FOIA outcomes are program/cohort objects.  This is especially
        # important for IPEDS denominators (ctotalt/cnralt), F-1 shares, and
        # tuition, which should not be repeated across individual FOIA rows in
        # the estimating panel.
        did_spec=generalized.DID_SPEC_COLLAPSED_UNIT_FE,
        include_degree_specific_plots=False,
        yvars=_foia_yvars_for_slide_assets(slide_assets),
        degree_did_panel_parquet=(
            _foia_degree_did_panel_cache_path(args)
            if hasattr(args, "cache_dir")
            else None
        ),
    )
    log.output(panel_path)
    _ensure_foia_stem_cip_assets(args, log)


def _foia_estimator_cache_path(
    args: argparse.Namespace,
    yvar: str,
    estimation_type: str,
    control_group: str | None = None,
) -> Path:
    sample_suffix = (
        f"_{FOIA_STATUS_CHANGE_SAMPLE_VERSION}"
        if yvar == "status_change_share"
        else ""
    )
    weight_suffix = (
        "_wgrads"
        if estimation_type == "sun_abraham" and bool(getattr(args, "sun_abraham_use_weights", False))
        else ""
    )
    control_suffix = _foia_control_group_suffix(control_group)
    return (
        args.cache_dir
        / "foia_estimator_comparison"
        / (
            f"{estimation_type}{weight_suffix}_{FOIA_ESTIMATOR_PANEL_VERSION}{sample_suffix}"
            f"{control_suffix}_{yvar}_e{int(args.event_window)}_b{int(args.bootstrap_reps)}_s{int(args.random_seed)}.csv"
        )
    )


def _foia_control_group_suffix(control_group: str | None) -> str:
    normalized = generalized._normalize_control_group(control_group or FOIA_DEFAULT_CONTROL_GROUP)  # noqa: SLF001
    return "" if normalized == FOIA_DEFAULT_CONTROL_GROUP else f"_{normalized}"


def _foia_estimation_panel_for_outcome(did_panel: pd.DataFrame, yvar: str) -> pd.DataFrame:
    """Apply outcome-specific sample choices for FOIA estimator appendices."""
    if yvar != "status_change_share" or "calendar_year" not in did_panel.columns:
        return did_panel
    calendar_year = pd.to_numeric(did_panel["calendar_year"], errors="coerce")
    return did_panel[calendar_year.le(FOIA_STATUS_CHANGE_MAX_CALENDAR_YEAR)].copy()


def _foia_required_preperiod_min_year(did_panel: pd.DataFrame, event_window: int) -> int | None:
    if did_panel.empty or "relabel_year" not in did_panel.columns:
        return None
    relabel_year = pd.to_numeric(did_panel["relabel_year"], errors="coerce").dropna()
    if relabel_year.empty:
        return None
    return int(relabel_year.min()) - int(event_window)


@contextlib.contextmanager
def _patched_foia_preperiod_window(did_panel: pd.DataFrame, event_window: int) -> Iterator[None]:
    """Let labor-lunch FOIA appendices use all panel years needed for event_t=-K."""
    old_min = generalized.base.PLOT_YEAR_MIN
    required_min = _foia_required_preperiod_min_year(did_panel, event_window)
    if required_min is None:
        yield
        return
    generalized.base.PLOT_YEAR_MIN = min(int(old_min), int(required_min))
    try:
        yield
    finally:
        generalized.base.PLOT_YEAR_MIN = old_min


def _foia_pooled_did_panel_cache_path(
    args: argparse.Namespace,
    control_group: str | None = None,
) -> Path:
    return (
        args.cache_dir
        / "foia_estimator_comparison"
        / f"pooled_did_panel_{FOIA_ESTIMATOR_PANEL_VERSION}{_foia_control_group_suffix(control_group)}.parquet"
    )


def _foia_degree_did_panel_cache_path(
    args: argparse.Namespace,
    control_group: str | None = None,
) -> Path:
    return (
        args.cache_dir
        / "foia_estimator_comparison"
        / f"degree_did_panels_{FOIA_ESTIMATOR_PANEL_VERSION}{_foia_control_group_suffix(control_group)}.parquet"
    )


def _foia_estimator_panel_missing_outcomes(did_panel: pd.DataFrame) -> list[str]:
    required = [yvar for _, yvar, _ in FOIA_ESTIMATION_APPENDICES]
    return [yvar for yvar in required if yvar not in did_panel.columns]


def _foia_panel_with_nonresident_share(did_panel: pd.DataFrame) -> pd.DataFrame:
    yvar = "cnralt_share_of_ctotalt"
    if did_panel.empty or yvar in did_panel.columns:
        return did_panel
    if not {"cnralt", "ctotalt"}.issubset(did_panel.columns):
        return did_panel
    out = did_panel.copy()
    numerator = pd.to_numeric(out["cnralt"], errors="coerce")
    denominator = pd.to_numeric(out["ctotalt"], errors="coerce")
    out[yvar] = numerator / denominator.replace(0, np.nan)
    return out


def _foia_panel_with_stem_cip_eligibility(did_panel: pd.DataFrame) -> pd.DataFrame:
    if did_panel.empty or FOIA_STEM_CIP_YVAR in did_panel.columns:
        return did_panel
    required = {"unitid", "calendar_year", "broad_pair_bin", "degree_type", "total_grads"}
    if not required.issubset(did_panel.columns):
        return did_panel

    work = did_panel.copy().reset_index(drop=True)
    work["_stem_rowid"] = np.arange(len(work), dtype=np.int64)
    row_keys = work[
        ["_stem_rowid", "unitid", "calendar_year", "broad_pair_bin", "degree_type"]
    ].dropna(subset=["unitid", "calendar_year", "broad_pair_bin"])
    if row_keys.empty:
        work = work.drop(columns=["_stem_rowid"])
        return work

    con = ddb.connect()
    try:
        con.register("foia_stem_panel_rows_py", row_keys)
        generalized._ensure_ipeds_view(con, generalized.base.IPEDS_PATH)  # noqa: SLF001
        generalized._ensure_stem_opt_cip_view(con)  # noqa: SLF001
        cip_map = generalized._load_ipeds_cip_map(generalized.base.IPEDS_PATH)  # noqa: SLF001
        broad_membership = generalized.build_broad_bin_membership(cip_map.keys())
        broad_any_cips = generalized._broad_membership_rows(broad_membership, side="all")  # noqa: SLF001
        con.register("foia_stem_broad_any_cips_py", broad_any_cips)

        counts = con.sql(
            f"""
            WITH ipeds_base AS (
                SELECT
                    CAST(unitid AS BIGINT) AS unitid,
                    CAST(year AS INTEGER) AS calendar_year,
                    CAST(awlevel AS INTEGER) AS awlevel,
                    LPAD(CAST(cipcode AS VARCHAR), 6, '0') AS cip6,
                    CAST(ctotalt AS DOUBLE) AS graduates,
                    CASE
                        WHEN stem.first_stem_year IS NOT NULL
                         AND stem.first_stem_year <= CAST(year AS INTEGER)
                        THEN 1 ELSE 0
                    END AS stem_cip_eligible_ind
                FROM ipeds_raw i
                LEFT JOIN stem_opt_cip_first_year stem
                  ON stem.cip6 = LPAD(CAST(i.cipcode AS VARCHAR), 6, '0')
                WHERE unitid IS NOT NULL
                  AND year IS NOT NULL
                  AND awlevel IS NOT NULL
                  AND cipcode IS NOT NULL
                  AND ctotalt IS NOT NULL
                  AND CAST(year AS INTEGER) <= {int(generalized.ANALYSIS_ORIGINAL_YEAR_MAX)}
            ),
            matched_rows AS (
                SELECT
                    r._stem_rowid,
                    f.graduates,
                    f.stem_cip_eligible_ind
                FROM foia_stem_panel_rows_py r
                JOIN ipeds_base f
                  ON CAST(f.unitid AS BIGINT) = CAST(r.unitid AS BIGINT)
                 AND CAST(f.calendar_year AS INTEGER) = CAST(r.calendar_year AS INTEGER)
                JOIN foia_stem_broad_any_cips_py m
                  ON m.broad_pair_bin = r.broad_pair_bin
                 AND f.cip6 = m.cip6
                WHERE r.degree_type = 'Pooled'
                   OR CASE
                        WHEN f.awlevel = 5 THEN 'Bachelor'
                        WHEN f.awlevel = 7 THEN 'Master'
                        WHEN f.awlevel IN (9, 17) THEN 'Doctor'
                        ELSE 'Other'
                      END = r.degree_type
            )
            SELECT
                _stem_rowid,
                SUM(graduates) AS stem_cip_denominator,
                SUM(CASE WHEN stem_cip_eligible_ind = 1 THEN graduates ELSE 0 END) AS stem_cip_eligible_users
            FROM matched_rows
            GROUP BY 1
            """
        ).df()
    finally:
        con.close()

    if counts.empty:
        work["stem_cip_eligible_users"] = np.nan
        work[FOIA_STEM_CIP_YVAR] = np.nan
    else:
        work = work.merge(counts, on="_stem_rowid", how="left")
        denominator = pd.to_numeric(work["stem_cip_denominator"], errors="coerce")
        fallback = pd.to_numeric(work["total_grads"], errors="coerce")
        denominator = denominator.where(denominator.gt(0), fallback)
        work[FOIA_STEM_CIP_YVAR] = (
            pd.to_numeric(work["stem_cip_eligible_users"], errors="coerce") / denominator.replace(0, np.nan)
        )
    return work.drop(columns=["_stem_rowid"])


def _foia_expand_donor_controls_by_relabel_cohort(did_panel: pd.DataFrame) -> pd.DataFrame:
    if did_panel.empty or not _is_foia_donor_control_panel(did_panel):
        return did_panel.copy()
    treated = did_panel[pd.to_numeric(did_panel["treated"], errors="coerce").eq(1)].copy()
    donors = did_panel[pd.to_numeric(did_panel["treated"], errors="coerce").eq(0)].copy()
    if treated.empty or donors.empty:
        return did_panel.copy()
    cohort_keys = ["relabel_year", "relabel_type", "broad_pair_bin", "degree_type"]
    cohorts = treated[cohort_keys].dropna(subset=["relabel_year"]).drop_duplicates()
    if cohorts.empty:
        return did_panel.copy()
    expanded = (
        donors.drop(columns=["relabel_year", "relabel_type", "event_t"], errors="ignore")
        .merge(cohorts, on=["broad_pair_bin", "degree_type"], how="inner")
    )
    if expanded.empty:
        return did_panel.copy()
    expanded["event_t"] = (
        pd.to_numeric(expanded["calendar_year"], errors="coerce")
        - pd.to_numeric(expanded["relabel_year"], errors="coerce")
    )
    expanded["treated"] = 0
    expanded["control_panel_role"] = "donor_expanded"
    expanded["pair_id"] = (
        expanded["pair_id"].astype(str)
        + "||cohort_"
        + pd.to_numeric(expanded["relabel_year"], errors="coerce").astype("Int64").astype(str)
    )
    return pd.concat([treated, expanded], ignore_index=True, sort=False)


def _ensure_foia_stem_cip_assets(args: argparse.Namespace, log: RunLog) -> None:
    required_arg_paths = ("cache_dir", "foia_output_dir", "foia_plots_dir")
    if any(not hasattr(args, attr) for attr in required_arg_paths):
        log.skip("foia_stem_cip_eligible_assets")
        return
    targets = [
        args.foia_plots_dir / "pooled_stem_cip_eligible_share_did_event_time_never_treated.png",
        args.foia_plots_dir / "pooled_stem_cip_eligible_share_event_time_treated_control_never_treated.png",
        args.foia_plots_dir / "pooled_broad_bin_did_appendix" / "pooled_broad_bins_stem_cip_eligible_share_did_event_time_never_treated.png",
        args.foia_plots_dir / "pooled_degree_level_did_appendix" / "pooled_degree_levels_stem_cip_eligible_share_did_event_time_never_treated.png",
        args.foia_plots_dir / "pooled_calendar_year_did_appendix" / "pooled_calendar_years_stem_cip_eligible_share_did_by_relabel_year.png",
        _estimation_type_path(args.foia_plots_dir, "pooled_stem_cip_eligible_share"),
    ]
    if not args.force_rebuild and not _missing_assets(targets):
        log.hit("foia_stem_cip_eligible_assets")
        return

    cache_path = _foia_pooled_did_panel_cache_path(args)
    panel_path = args.foia_output_dir / "generalized_relabels_panel.parquet"
    if not cache_path.exists() and not panel_path.exists():
        return

    did_panel = _load_or_build_foia_pooled_did_panel(args, log)
    did_panel = _foia_panel_with_stem_cip_eligibility(did_panel)
    if FOIA_STEM_CIP_YVAR not in did_panel.columns or did_panel[FOIA_STEM_CIP_YVAR].dropna().empty:
        raise RuntimeError("Could not compute FOIA STEM CIP eligibility share for pooled panel.")
    did_panel.to_parquet(cache_path, index=False)
    plot_panel = _foia_expand_donor_controls_by_relabel_cohort(did_panel)

    log.rebuild("foia_stem_cip_eligible_assets")
    raw_paths = generalized.plot_event_time_with_control_generalized(
        plot_panel,
        yvar=FOIA_STEM_CIP_YVAR,
        degree_type="Pooled",
        out_dir=args.foia_plots_dir,
    )
    for path in raw_paths:
        log.output(path)

    event = _compute_foia_estimator_rows(
        did_panel,
        args=args,
        yvar=FOIA_STEM_CIP_YVAR,
        estimation_type="twfe",
    )
    if event.empty:
        raise RuntimeError("No FOIA STEM CIP eligibility event-study rows were produced.")
    main_path = generalized.plot_did_event_study_generalized(
        event,
        yvar=FOIA_STEM_CIP_YVAR,
        degree_type="Pooled",
        out_dir=args.foia_plots_dir,
        reference_event_time=generalized.DI_D_REFERENCE_EVENT_TIME,
    )
    if main_path is None:
        raise RuntimeError("No FOIA STEM CIP eligibility main plot was produced.")
    log.output(main_path)

    broad_frames = []
    for broad_pair_bin, sub in did_panel.groupby("broad_pair_bin", dropna=False):
        if sub["treated"].nunique() < 2:
            continue
        rows = _compute_foia_estimator_rows(
            sub,
            args=args,
            yvar=FOIA_STEM_CIP_YVAR,
            estimation_type="twfe",
        )
        if not rows.empty:
            rows["broad_pair_bin"] = broad_pair_bin
            broad_frames.append(rows)
    broad_event = pd.concat(broad_frames, ignore_index=True, sort=False) if broad_frames else pd.DataFrame()
    broad_path = generalized.plot_broad_bin_did_event_study_generalized(
        broad_event,
        yvar=FOIA_STEM_CIP_YVAR,
        degree_type="Pooled",
        out_dir=args.foia_plots_dir / "pooled_broad_bin_did_appendix",
        file_stem="pooled_broad_bins_stem_cip_eligible_share_did_event_time_never_treated",
        reference_event_time=generalized.DI_D_REFERENCE_EVENT_TIME,
    )
    log.output(broad_path)

    degree_frames = []
    for degree_type, sub in did_panel.groupby("degree_type", dropna=False):
        if sub["treated"].nunique() < 2:
            continue
        rows = _compute_foia_estimator_rows(
            sub,
            args=args,
            yvar=FOIA_STEM_CIP_YVAR,
            estimation_type="twfe",
        )
        if not rows.empty:
            rows["degree_type"] = degree_type
            degree_frames.append(rows)
    degree_event = pd.concat(degree_frames, ignore_index=True, sort=False) if degree_frames else pd.DataFrame()
    degree_path = generalized.plot_degree_level_did_event_study_generalized(
        degree_event,
        yvar=FOIA_STEM_CIP_YVAR,
        out_dir=args.foia_plots_dir / "pooled_degree_level_did_appendix",
        file_stem="pooled_degree_levels_stem_cip_eligible_share_did_event_time_never_treated",
        reference_event_time=generalized.DI_D_REFERENCE_EVENT_TIME,
    )
    log.output(degree_path)

    calendar_event = generalized.compute_calendar_year_did_by_relabel_year(
        plot_panel,
        yvar=FOIA_STEM_CIP_YVAR,
        did_spec=generalized.DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE,
        use_weights=False,
    )
    calendar_path = generalized.plot_calendar_year_did_by_relabel_year(
        calendar_event,
        yvar=FOIA_STEM_CIP_YVAR,
        out_dir=args.foia_plots_dir / "pooled_calendar_year_did_appendix",
        file_stem="pooled_calendar_years_stem_cip_eligible_share_did_by_relabel_year",
    )
    log.output(calendar_path)

    estimate_rows = (
        event[["event_t", "coef", "se"]].copy()
        if not event.empty
        else pd.DataFrame(columns=["event_t", "coef", "se"])
    )
    if not estimate_rows.empty:
        estimate_rows["estimator"] = "twfe"
    estimator_path = _plot_estimation_type_comparison(
        estimate_rows,
        out_path=_estimation_type_path(args.foia_plots_dir, "pooled_stem_cip_eligible_share"),
        title="FOIA STEM OPT CIP Eligibility: by estimation type",
        x_col="event_t",
        x_label="Years relative to relabel event",
        yvar=FOIA_STEM_CIP_YVAR,
    )
    log.output(estimator_path)


def _foia_make_donor_control_panel(stacked_panel: pd.DataFrame) -> pd.DataFrame:
    """Convert matched-pair controls into a unique donor-control panel for estimation."""
    if stacked_panel.empty:
        return stacked_panel.copy()
    treated = stacked_panel[pd.to_numeric(stacked_panel["treated"], errors="coerce").eq(1)].copy()
    controls = stacked_panel[pd.to_numeric(stacked_panel["treated"], errors="coerce").eq(0)].copy()
    if controls.empty:
        return treated

    donor_keys = ["unitid", "calendar_year", "broad_pair_bin", "degree_type"]
    value_cols = [
        "avg_tuition",
        "total_grads",
        "stem_cip_eligible_users",
        "opt_users",
        "opt_stem_users",
        "status_change_users",
        "total_post_grad_authorization_years",
        "total_opt_duration_years",
        "unique_employers",
        "unique_opt_cities",
        "auth_employment_tenure_years",
        "employer_opt_intensity_pctile",
        "total_internships",
        "total_internship_opt_years",
        "internship_count",
        "internship_opt_years",
        "ctotalt",
        "cnralt",
        "tuition_ipeds_total",
        "fees_ipeds_total",
        "students_personal_funds_total",
        "total_funds",
        "stem_cip_eligible_share",
        "opt_share",
        "opt_stem_share",
        "status_change_share",
        "post_grad_authorization_years_avg",
        "opt_duration_years_avg",
        "opt_years_avg",
        "f1_share_of_ctotalt",
        "f1_share_of_cnralt",
        "cnralt_share_of_ctotalt",
        "tuition_total",
        "avg_tuition_ipeds",
        "avg_fees_ipeds",
        "avg_students_personal_funds",
        "avg_total_funds",
    ]
    agg = {col: "first" for col in value_cols if col in controls.columns}
    if "did_spec" in controls.columns:
        agg["did_spec"] = "first"
    if "panel_level" in controls.columns:
        agg["panel_level"] = "first"
    donor_controls = controls.groupby(donor_keys, as_index=False, dropna=False).agg(agg)
    donor_controls["treated"] = 0
    donor_controls["pair_id"] = (
        "donor||"
        + donor_controls["unitid"].astype(str)
        + "||"
        + donor_controls["broad_pair_bin"].astype(str)
        + "||"
        + donor_controls["degree_type"].astype(str)
    )
    donor_controls["relabel_year"] = pd.NA
    donor_controls["relabel_type"] = donor_controls["broad_pair_bin"]
    donor_controls["event_t"] = pd.NA
    donor_controls["control_panel_role"] = "donor"

    treated["control_panel_role"] = "treated"
    if "pair_id" in treated.columns:
        treated["pair_id"] = treated["pair_id"].astype(str)
    out = pd.concat([treated, donor_controls], ignore_index=True, sort=False)
    out = generalized._apply_did_design_columns(  # noqa: SLF001
        out,
        did_spec=generalized.DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE,
    )
    return out


def _load_or_build_foia_stacked_did_panel(
    args: argparse.Namespace,
    log: RunLog,
    *,
    control_group: str | None = None,
) -> pd.DataFrame:
    control_group = generalized._normalize_control_group(control_group or FOIA_DEFAULT_CONTROL_GROUP)  # noqa: SLF001
    panel_path = args.foia_output_dir / "generalized_relabels_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing generalized FOIA relabel panel: {panel_path}")
    degree_cache_path = _foia_degree_did_panel_cache_path(args, control_group)
    use_degree_cache = _file_ok(degree_cache_path) and (
        "generalized_foia_assets" in log.rebuilt
        or (
            not args.force_rebuild
            and not _any_source_newer(degree_cache_path, [panel_path])
        )
    )
    if use_degree_cache:
        log.hit(f"foia_degree_did_panels_from_generalized_stage{_foia_control_group_suffix(control_group)}")
        cached = _foia_panel_with_nonresident_share(pd.read_parquet(degree_cache_path))
        if "cnralt_share_of_ctotalt" in cached.columns:
            cached.to_parquet(degree_cache_path, index=False)
        return cached

    relabel_panel = pd.read_parquet(panel_path)
    con = ddb.connect()
    try:
        con.sql(f"PRAGMA threads={max(1, os.cpu_count() or 1)}")
        con.sql("PRAGMA preserve_insertion_order=false")
        degree_panels: list[pd.DataFrame] = []
        degree_progress = ProgressCounter(
            f"FOIA donor panels [{control_group}]",
            len(generalized.POOLED_DEGREE_TYPES),
            report_every=1,
        )
        for degree in generalized.POOLED_DEGREE_TYPES:
            degree_panel = generalized.compute_generalized_did_panel(
                con,
                relabel_panel,
                degree_type=degree,
                did_spec=generalized.DID_SPEC_COLLAPSED_UNIT_FE,
                control_group=control_group,
            )
            if not degree_panel.empty:
                degree_panel["control_group"] = control_group
                degree_panels.append(degree_panel)
            degree_progress.advance(str(degree))
        stacked_panel = pd.concat(degree_panels, ignore_index=True, sort=False) if degree_panels else pd.DataFrame()
    finally:
        con.close()

    if not stacked_panel.empty:
        stacked_panel = _foia_panel_with_nonresident_share(stacked_panel)
        degree_cache_path.parent.mkdir(parents=True, exist_ok=True)
        stacked_panel.to_parquet(degree_cache_path, index=False)
        log.output(degree_cache_path)
    return stacked_panel


def _load_or_build_foia_pooled_did_panel(
    args: argparse.Namespace,
    log: RunLog,
    *,
    control_group: str | None = None,
) -> pd.DataFrame:
    control_group = generalized._normalize_control_group(control_group or FOIA_DEFAULT_CONTROL_GROUP)  # noqa: SLF001
    cache_path = _foia_pooled_did_panel_cache_path(args, control_group)
    rebuilt_name = f"foia_pooled_did_panel_for_estimator_comparison{_foia_control_group_suffix(control_group)}"
    if args.force_rebuild and _file_ok(cache_path) and rebuilt_name in log.rebuilt:
        log.hit(f"{rebuilt_name}_in_process")
        return pd.read_parquet(cache_path)
    if not args.force_rebuild and _file_ok(cache_path):
        did_panel = pd.read_parquet(cache_path)
        did_panel = _foia_panel_with_nonresident_share(did_panel)
        did_panel = _foia_panel_with_stem_cip_eligibility(did_panel)
        missing_outcomes = _foia_estimator_panel_missing_outcomes(did_panel)
        if not missing_outcomes:
            log.hit(rebuilt_name)
            if FOIA_STEM_CIP_YVAR in did_panel.columns or "cnralt_share_of_ctotalt" in did_panel.columns:
                did_panel.to_parquet(cache_path, index=False)
            return did_panel
        log.rebuild(
            rebuilt_name
            +
            f"_missing_outcomes:{','.join(missing_outcomes)}"
        )
        if cache_path.exists():
            cache_path.unlink()
    if not args.force_rebuild and control_group == FOIA_DEFAULT_CONTROL_GROUP:
        legacy_candidates = [
            args.cache_dir / "foia_estimator_comparison" / "pooled_did_panel_cohort_broad_degree_full_pre5_v2_donor_controls_v1.parquet",
            args.cache_dir / "foia_estimator_comparison" / "pooled_did_panel_collapsed_unit_fe.parquet",
        ]
        for legacy_path in legacy_candidates:
            if _file_ok(legacy_path):
                did_panel = pd.read_parquet(legacy_path)
                did_panel = _foia_panel_with_nonresident_share(did_panel)
                did_panel = _foia_panel_with_stem_cip_eligibility(did_panel)
                missing_outcomes = _foia_estimator_panel_missing_outcomes(did_panel)
                if missing_outcomes:
                    log.skip(
                        f"foia_pooled_did_panel_legacy_cache_incompatible:{legacy_path.name}"
                        f":missing={','.join(missing_outcomes)}"
                    )
                    continue
                log.hit(f"foia_pooled_did_panel_legacy_cache:{legacy_path.name}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                did_panel.to_parquet(cache_path, index=False)
                return did_panel
    stacked_panel = _load_or_build_foia_stacked_did_panel(args, log, control_group=control_group)
    _progress(f"FOIA donor panel [{control_group}]: deduplicating donor controls")
    did_panel = _foia_make_donor_control_panel(stacked_panel)
    if did_panel.empty:
        raise RuntimeError("Pooled FOIA DiD panel is empty; cannot build estimator comparison appendices.")
    did_panel = _foia_panel_with_stem_cip_eligibility(did_panel)
    did_panel = _foia_panel_with_nonresident_share(did_panel)
    missing_outcomes = _foia_estimator_panel_missing_outcomes(did_panel)
    if missing_outcomes:
        raise RuntimeError(
            "Pooled FOIA DiD panel is missing estimator outcomes: "
            + ", ".join(missing_outcomes)
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    did_panel.to_parquet(cache_path, index=False)
    log.rebuild(rebuilt_name)
    log.output(cache_path)
    return did_panel


def _is_foia_donor_control_panel(did_panel: pd.DataFrame) -> bool:
    return (
        "control_panel_role" in did_panel.columns
        and did_panel["control_panel_role"].astype(str).eq("donor").any()
    )


def _prepare_foia_donor_regression_df(
    did_panel: pd.DataFrame,
    *,
    yvar: str,
    event_window: int,
) -> pd.DataFrame:
    if did_panel.empty or yvar not in did_panel.columns:
        return pd.DataFrame()
    reg_df = did_panel.copy()
    reg_df = reg_df[reg_df["calendar_year"].between(generalized.base.PLOT_YEAR_MIN, generalized.base.PLOT_YEAR_MAX)].copy()
    reg_df["treated"] = pd.to_numeric(reg_df["treated"], errors="coerce").fillna(0).astype(int)
    reg_df["event_t"] = pd.to_numeric(reg_df["event_t"], errors="coerce")
    treated_in_window = reg_df["treated"].eq(1) & reg_df["event_t"].between(-int(event_window), int(event_window))
    donor_controls = reg_df["treated"].eq(0)
    reg_df = reg_df[treated_in_window | donor_controls].copy()
    reg_df[yvar] = pd.to_numeric(reg_df[yvar], errors="coerce")
    reg_df["calendar_year"] = pd.to_numeric(reg_df["calendar_year"], errors="coerce")
    reg_df["unitid"] = pd.to_numeric(reg_df["unitid"], errors="coerce")
    reg_df["total_grads"] = pd.to_numeric(reg_df.get("total_grads"), errors="coerce").fillna(0.0)
    reg_df = reg_df.dropna(subset=[yvar, "treated", "calendar_year", "unitid", "total_grads"])
    reg_df = reg_df[reg_df["total_grads"] > 0].copy()
    if reg_df.empty:
        return pd.DataFrame()
    reg_df["calendar_year"] = reg_df["calendar_year"].astype(int)
    reg_df["unitid"] = reg_df["unitid"].astype(int)
    reg_df.loc[reg_df["treated"].eq(1), "event_t"] = reg_df.loc[reg_df["treated"].eq(1), "event_t"].astype(int)
    treated_relabel = pd.to_numeric(reg_df.get("relabel_year"), errors="coerce")
    reg_df["relabel_year"] = treated_relabel.where(reg_df["treated"].eq(1), pd.NA)
    treated_events = reg_df.loc[reg_df["treated"].eq(1), "event_t"].dropna().astype(int)
    if treated_events.empty:
        return pd.DataFrame()
    reg_df = generalized._apply_did_design_columns(  # noqa: SLF001
        reg_df,
        did_spec=generalized.DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE,
    )
    return reg_df


def _event_dummy_name(event_t: int) -> str:
    return f"treated_event_{'m' + str(abs(int(event_t))) if int(event_t) < 0 else 'p' + str(int(event_t))}"


def _compute_foia_donor_twfe_rows(
    did_panel: pd.DataFrame,
    *,
    yvar: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    reg_df = _prepare_foia_donor_regression_df(did_panel, yvar=yvar, event_window=int(args.event_window))
    if reg_df.empty:
        return pd.DataFrame()
    event_values = sorted(reg_df.loc[reg_df["treated"].eq(1), "event_t"].dropna().astype(int).unique().tolist())
    reference_event_time = int(generalized.DI_D_REFERENCE_EVENT_TIME)
    if reference_event_time not in event_values or len(event_values) < 2:
        return pd.DataFrame()
    dummy_cols: dict[int, str] = {}
    for event_t in event_values:
        if event_t == reference_event_time:
            continue
        col = _event_dummy_name(event_t)
        reg_df[col] = (reg_df["treated"].eq(1) & reg_df["event_t"].eq(event_t)).astype(int)
        dummy_cols[event_t] = col
    if not dummy_cols:
        return pd.DataFrame()
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        print(f"statsmodels unavailable; skipping donor-control TWFE for {yvar} ({exc})")
        return pd.DataFrame()
    fe_term = generalized._did_fe_formula_term(  # noqa: SLF001
        reg_df,
        did_spec=generalized.DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE,
    )
    formula = f"{yvar} ~ {' + '.join(dummy_cols.values())} + {fe_term} + C(calendar_year)"
    try:
        model = smf.ols(formula, data=reg_df)
        result = model.fit(cov_type="cluster", cov_kwds={"groups": reg_df["unitid"]})
    except Exception as exc:
        print(f"Clustered donor-control TWFE failed for {yvar}; falling back to HC1 ({exc})")
        model = smf.ols(formula, data=reg_df)
        result = model.fit(cov_type="HC1")
    params, bse = generalized._result_params_and_bse(result, backend="statsmodels")  # noqa: SLF001
    rows: list[dict[str, object]] = []
    for event_t in event_values:
        if event_t == reference_event_time:
            coef, se = 0.0, 0.0
        else:
            col = dummy_cols[event_t]
            coef = float(params.get(col, float("nan")))
            se = float(bse.get(col, float("nan")))
        rows.append(
            _base_estimate_row(
                reg_df,
                int(event_t),
                "twfe",
                coef,
                se,
                reference_event_time,
                extra={
                    "aggregation_unit": "treated_cohort_x_deduped_donor_controls",
                    "control_design": "deduped_donor_programs",
                },
            )
        )
    return pd.DataFrame(rows)


def _compute_foia_estimator_rows(
    did_panel: pd.DataFrame,
    *,
    args: argparse.Namespace,
    yvar: str,
    estimation_type: str,
) -> pd.DataFrame:
    did_panel = _foia_estimation_panel_for_outcome(did_panel, yvar)
    with _patched_foia_preperiod_window(did_panel, int(args.event_window)):
        if estimation_type == "twfe":
            if _is_foia_donor_control_panel(did_panel):
                rows = _compute_foia_donor_twfe_rows(did_panel, yvar=yvar, args=args)
            else:
                rows = generalized.compute_did_event_study_generalized(
                    did_panel=did_panel,
                    yvar=yvar,
                    did_spec=generalized.DID_SPEC_COLLAPSED_UNIT_FE,
                    reference_event_time=generalized.DI_D_REFERENCE_EVENT_TIME,
                    event_time_min=-int(args.event_window),
                    event_time_max=int(args.event_window),
                    use_weights=False,
                )
        else:
            validate_estimator_dependencies(estimation_type)
            if _is_foia_donor_control_panel(did_panel):
                reg_df = _prepare_foia_donor_regression_df(
                    did_panel,
                    yvar=yvar,
                    event_window=int(args.event_window),
                )
            else:
                reg_df = generalized._prepare_did_regression_df(  # noqa: SLF001
                    did_panel,
                    yvar=yvar,
                    did_spec=generalized.DID_SPEC_COLLAPSED_UNIT_FE,
                    event_time_min=-int(args.event_window),
                    event_time_max=int(args.event_window),
                )
            rows = estimate_dynamic_effects(
                reg_df,
                yvar=yvar,
                estimation_type=estimation_type,
                reference_event_time=generalized.DI_D_REFERENCE_EVENT_TIME,
                bootstrap_reps=args.bootstrap_reps,
                random_seed=args.random_seed,
                use_weights=(
                    bool(getattr(args, "sun_abraham_use_weights", False))
                    if estimation_type == "sun_abraham"
                    else False
                ),
            )
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["estimator"] = estimation_type
    rows["outcome"] = yvar
    return rows


def _ensure_foia_estimator_cache(
    args: argparse.Namespace,
    *,
    yvar: str,
    estimation_type: str,
    did_panel: pd.DataFrame,
    log: RunLog,
    control_group: str | None = None,
) -> Path:
    control_group = generalized._normalize_control_group(control_group or FOIA_DEFAULT_CONTROL_GROUP)  # noqa: SLF001
    cache_path = _foia_estimator_cache_path(args, yvar, estimation_type, control_group=control_group)
    rebuilt_name = f"foia_{estimation_type}_{yvar}_estimator_rows{_foia_control_group_suffix(control_group)}"
    if _force_estimator_rebuild(args) and _file_ok(cache_path) and rebuilt_name in log.rebuilt:
        log.hit(f"{rebuilt_name}_in_process")
        return cache_path
    if not _force_estimator_rebuild(args) and _file_ok(cache_path):
        log.hit(rebuilt_name)
        return cache_path
    if log.verbose:
        _progress(f"FOIA estimator: {estimation_type} / {yvar}")
    rows = _compute_foia_estimator_rows(
        did_panel,
        args=args,
        yvar=yvar,
        estimation_type=estimation_type,
    )
    if rows.empty:
        raise RuntimeError(f"No {estimation_type} estimator rows produced for FOIA outcome {yvar}.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(cache_path, index=False)
    log.rebuild(rebuilt_name)
    log.output(cache_path)
    return cache_path


def _load_foia_estimation_rows(args: argparse.Namespace, yvar: str, did_panel: pd.DataFrame, log: RunLog) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for estimation_type in ESTIMATOR_COMPARISON_TYPES:
        cache_path = _ensure_foia_estimator_cache(
            args,
            yvar=yvar,
            estimation_type=estimation_type,
            did_panel=did_panel,
            log=log,
        )
        data = pd.read_csv(cache_path)
        data["estimator"] = estimation_type
        keep = ["event_t", "estimator", "coef"] + (["se"] if "se" in data.columns else [])
        frames.append(data[keep])

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["estimator"] = out["estimator"].map(_normalize_estimator_name)
    out = out[out["estimator"].isin(ESTIMATOR_COMPARISON_TYPES)].copy()
    out["se"] = pd.to_numeric(out.get("se", np.nan), errors="coerce")
    return out


def _ensure_foia_estimation_type_appendices(args: argparse.Namespace, log: RunLog) -> None:
    appendices = [
        entry for entry in FOIA_ESTIMATION_APPENDICES
        if entry[1] != FOIA_STEM_CIP_YVAR
    ]
    targets = [
        _estimation_type_path(args.foia_plots_dir, f"pooled_{yvar}")
        for _, yvar, _ in appendices
    ]
    estimator_caches = [
        _foia_estimator_cache_path(args, yvar, estimation_type)
        for _, yvar, _ in appendices
        for estimation_type in ESTIMATOR_COMPARISON_TYPES
    ]
    stale_targets = [
        target
        for target, (_, yvar, _) in zip(targets, appendices)
        if _any_source_newer(target, [_foia_estimator_cache_path(args, yvar, estimator) for estimator in ESTIMATOR_COMPARISON_TYPES])
    ]
    force_estimator = _force_estimator_rebuild(args)
    if not force_estimator and not _missing_assets(targets) and not _missing_assets(estimator_caches) and not stale_targets:
        log.hit("foia_estimation_type_appendices")
        return
    log.rebuild("foia_estimation_type_appendices")
    did_panel = _load_or_build_foia_pooled_did_panel(args, log)
    ipeds_did_panel: pd.DataFrame | None = None
    appendix_progress = ProgressCounter("FOIA estimator appendices", len(appendices))
    for _, yvar, title in appendices:
        out_path = _estimation_type_path(args.foia_plots_dir, f"pooled_{yvar}")
        yvar_caches = [_foia_estimator_cache_path(args, yvar, estimator) for estimator in ESTIMATOR_COMPARISON_TYPES]
        if not force_estimator and _file_ok(out_path) and not _missing_assets(yvar_caches) and not _any_source_newer(out_path, yvar_caches):
            appendix_progress.advance(f"{yvar} (cached)")
            continue
        panel = did_panel
        if yvar in generalized.PROGRAM_LEVEL_IPEDS_YVARS:
            if ipeds_did_panel is None:
                ipeds_did_panel = _load_or_build_foia_pooled_ipeds_did_panel(
                    args,
                    log,
                    control_group=FOIA_DEFAULT_CONTROL_GROUP,
                )
            panel = ipeds_did_panel
        rows = _load_foia_estimation_rows(args, yvar, panel, log)
        _plot_estimation_type_comparison(
            rows,
            out_path=out_path,
            title=f"{title}: by estimation type",
            x_col="event_t",
            x_label="Years relative to relabel event",
            yvar=yvar,
        )
        log.output(out_path)
        appendix_progress.advance(yvar)


def _foia_grouped_appendix_paths(args: argparse.Namespace, yvar: str) -> tuple[Path, Path]:
    broad_path = (
        args.foia_plots_dir
        / "pooled_broad_bin_did_appendix"
        / f"pooled_broad_bins_{yvar}_did_event_time_never_treated.png"
    )
    degree_path = (
        args.foia_plots_dir
        / "pooled_degree_level_did_appendix"
        / f"pooled_degree_levels_{yvar}_did_event_time_never_treated.png"
    )
    return broad_path, degree_path


def _foia_grouped_estimator_cache_path(args: argparse.Namespace, yvar: str, group_col: str) -> Path:
    sample_suffix = (
        f"_{FOIA_STATUS_CHANGE_SAMPLE_VERSION}"
        if yvar == "status_change_share"
        else ""
    )
    weight_suffix = (
        "_wgrads"
        if args.estimation_type == "sun_abraham" and bool(getattr(args, "sun_abraham_use_weights", False))
        else ""
    )
    return (
        args.cache_dir
        / "foia_estimator_comparison"
        / "grouped"
        / (
            f"{args.estimation_type}{weight_suffix}_{FOIA_ESTIMATOR_PANEL_VERSION}{sample_suffix}_"
            f"{group_col}_{yvar}_e{int(args.event_window)}_b{int(args.bootstrap_reps)}_s{int(args.random_seed)}.csv"
        )
    )


def _grouped_appendix_matches_estimator(path: Path, estimation_type: str) -> bool:
    if not _file_ok(path):
        return False
    csv_path = path.with_suffix(".csv")
    if not _file_ok(csv_path):
        return False
    try:
        data = pd.read_csv(csv_path, usecols=["estimator"])
    except Exception:
        return False
    if data.empty:
        return False
    estimators = data["estimator"].dropna().map(_normalize_estimator_name)
    return bool(not estimators.empty and estimators.eq(estimation_type).all())


def _compute_foia_grouped_estimator_rows(
    did_panel: pd.DataFrame,
    *,
    args: argparse.Namespace,
    yvar: str,
    group_col: str,
    log: RunLog,
) -> pd.DataFrame:
    if did_panel.empty or yvar not in did_panel.columns or group_col not in did_panel.columns:
        return pd.DataFrame()
    cache_path = _foia_grouped_estimator_cache_path(args, yvar, group_col)
    rebuilt_name = f"foia_grouped_{args.estimation_type}_{group_col}_{yvar}_rows"
    if _file_ok(cache_path) and (
        not _force_estimator_rebuild(args) or rebuilt_name in log.rebuilt
    ):
        log.hit(rebuilt_name)
        return pd.read_csv(cache_path)
    frames: list[pd.DataFrame] = []
    for group_value, sub in did_panel.groupby(group_col, dropna=False):
        if sub["treated"].nunique() < 2:
            continue
        rows = _compute_foia_estimator_rows(
            sub,
            args=args,
            yvar=yvar,
            estimation_type=args.estimation_type,
        )
        if rows.empty:
            continue
        rows[group_col] = group_value
        frames.append(rows)
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not out.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False)
        log.rebuild(rebuilt_name)
        log.output(cache_path)
    return out


def _ensure_foia_grouped_estimator_appendices(args: argparse.Namespace, log: RunLog) -> None:
    """Overwrite broad-bin and degree FOIA appendix plots with the selected estimator."""
    if args.estimation_type not in ESTIMATOR_COMPARISON_TYPES:
        log.skip(f"foia_grouped_{args.estimation_type}_appendices")
        return

    force_estimator = _force_estimator_rebuild(args)
    all_targets = [
        path
        for _, yvar, _ in FOIA_ESTIMATION_APPENDICES
        for path in _foia_grouped_appendix_paths(args, yvar)
    ]
    if (
        not force_estimator
        and all(_grouped_appendix_matches_estimator(path, args.estimation_type) for path in all_targets)
    ):
        log.hit(f"foia_grouped_{args.estimation_type}_appendices")
        return

    did_panel = _load_or_build_foia_pooled_did_panel(args, log)
    ipeds_did_panel: pd.DataFrame | None = None
    log.rebuild(f"foia_grouped_{args.estimation_type}_appendices")
    grouped_progress = ProgressCounter(
        f"FOIA grouped appendices [{args.estimation_type}]",
        len(FOIA_ESTIMATION_APPENDICES),
    )
    for _, yvar, _title in FOIA_ESTIMATION_APPENDICES:
        active_panel = did_panel
        if yvar in generalized.PROGRAM_LEVEL_IPEDS_YVARS:
            if ipeds_did_panel is None:
                ipeds_did_panel = _load_or_build_foia_pooled_ipeds_did_panel(
                    args,
                    log,
                    control_group=FOIA_DEFAULT_CONTROL_GROUP,
                )
            active_panel = ipeds_did_panel
        broad_path, degree_path = _foia_grouped_appendix_paths(args, yvar)

        if force_estimator or not _grouped_appendix_matches_estimator(broad_path, args.estimation_type):
            broad_event = _compute_foia_grouped_estimator_rows(
                active_panel,
                args=args,
                yvar=yvar,
                group_col="broad_pair_bin",
                log=log,
            )
            if not broad_event.empty:
                broad_path.parent.mkdir(parents=True, exist_ok=True)
                out_path = generalized.plot_broad_bin_did_event_study_generalized(
                    broad_event,
                    yvar=yvar,
                    degree_type="Pooled",
                    out_dir=args.foia_plots_dir / "pooled_broad_bin_did_appendix",
                    file_stem=f"pooled_broad_bins_{yvar}_did_event_time_never_treated",
                    reference_event_time=generalized.DI_D_REFERENCE_EVENT_TIME,
                )
                if out_path is not None:
                    log.output(out_path)
                    log.output(Path(out_path).with_suffix(".csv"))

        if force_estimator or not _grouped_appendix_matches_estimator(degree_path, args.estimation_type):
            degree_event = _compute_foia_grouped_estimator_rows(
                active_panel,
                args=args,
                yvar=yvar,
                group_col="degree_type",
                log=log,
            )
            if not degree_event.empty:
                degree_path.parent.mkdir(parents=True, exist_ok=True)
                out_path = generalized.plot_degree_level_did_event_study_generalized(
                    degree_event,
                    yvar=yvar,
                    out_dir=args.foia_plots_dir / "pooled_degree_level_did_appendix",
                    file_stem=f"pooled_degree_levels_{yvar}_did_event_time_never_treated",
                    reference_event_time=generalized.DI_D_REFERENCE_EVENT_TIME,
                )
                if out_path is not None:
                    log.output(out_path)
                    log.output(Path(out_path).with_suffix(".csv"))
        grouped_progress.advance(yvar)


def _foia_main_did_path(args: argparse.Namespace, yvar: str) -> Path:
    return args.foia_plots_dir / f"pooled_{yvar}_did_event_time_never_treated.png"


def _foia_main_summary_text(did_panel: pd.DataFrame, *, yvar: str) -> str | None:
    plot_panel = _foia_expand_donor_controls_by_relabel_cohort(did_panel)
    did_spec = (
        generalized.DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE
        if _is_foia_donor_control_panel(did_panel)
        else generalized.DID_SPEC_COLLAPSED_UNIT_FE
    )
    try:
        return generalized.build_did_summary_text(
            plot_panel,
            yvar=yvar,
            did_spec=did_spec,
            use_weights=False,
        )
    except Exception as exc:
        print(f"Warning: could not build FOIA summary box for {yvar}: {exc}", flush=True)
        return None


def _plot_foia_main_control_comparison_did(
    rows_by_control: dict[str, pd.DataFrame],
    *,
    args: argparse.Namespace,
    yvar: str,
    summary_text: str | None,
) -> Path:
    frames: list[pd.DataFrame] = []
    for control_group, rows in rows_by_control.items():
        if rows.empty:
            continue
        sub = rows.copy()
        sub["control_group"] = control_group
        frames.append(sub)
    if not frames:
        raise RuntimeError(f"No FOIA main control-comparison rows were produced for {yvar}.")
    plot_df = pd.concat(frames, ignore_index=True, sort=False)
    if "reference_event_t" in plot_df.columns:
        plot_df = plot_df[
            pd.to_numeric(plot_df["reference_event_t"], errors="coerce").eq(generalized.DI_D_REFERENCE_EVENT_TIME)
        ].copy()
    plot_df["event_t"] = pd.to_numeric(plot_df["event_t"], errors="coerce")
    plot_df["coef"] = pd.to_numeric(plot_df["coef"], errors="coerce")
    plot_df["se"] = pd.to_numeric(plot_df.get("se", np.nan), errors="coerce")
    plot_df = plot_df[
        plot_df["event_t"].between(generalized.PLOT_EVENT_MIN, generalized.PLOT_EVENT_MAX)
    ].dropna(subset=["event_t", "coef", "se", "control_group"]).copy()
    if plot_df.empty:
        raise RuntimeError(f"No usable FOIA main control-comparison rows were produced for {yvar}.")

    control_groups = [
        group for group in FOIA_MAIN_COMPARISON_CONTROL_GROUPS
        if group in set(plot_df["control_group"].astype(str))
    ]
    offset_step = generalized._multi_series_offset_step(len(control_groups))  # noqa: SLF001
    center = (len(control_groups) - 1) / 2
    text_scale = FOIA_MAIN_LARGER_TEXT_SCALE if yvar in FOIA_MAIN_LARGER_TEXT_YVARS else 1.0
    axis_fontsize = generalized.DI_D_PLOT_FONT_SIZE * text_scale
    tick_fontsize = llstyle.TICK_FONT_SIZE * text_scale
    legend_fontsize = llstyle.LEGEND_FONT_SIZE * text_scale
    summary_fontsize = max(generalized.DI_D_PLOT_FONT_SIZE - 8, 9) * text_scale

    llstyle.apply_style()
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    for idx, control_group in enumerate(control_groups):
        group_df = plot_df[plot_df["control_group"].eq(control_group)].sort_values("event_t").copy()
        if group_df.empty:
            continue
        offset = (idx - center) * offset_step
        group_df["event_t_plot"] = group_df["event_t"] + offset
        color = FOIA_CONTROL_GROUP_COLORS.get(control_group, generalized.base.PALETTE_SEQ[idx % len(generalized.base.PALETTE_SEQ)])
        marker = FOIA_CONTROL_GROUP_MARKERS.get(control_group, "o")
        label = FOIA_CONTROL_GROUP_LABELS.get(control_group, control_group.replace("_", " ").title())
        ax.plot(
            group_df["event_t_plot"],
            group_df["coef"],
            color=color,
            linewidth=1.5,
            alpha=0.9,
            zorder=1,
        )
        ax.scatter(
            group_df["event_t_plot"],
            group_df["coef"],
            color=color,
            s=generalized.DI_D_PLOT_MARKER_SIZE * generalized.DI_D_PLOT_MARKER_SIZE,
            marker=marker,
            linewidths=0,
            label=label,
        )
        ax.errorbar(
            group_df["event_t_plot"],
            group_df["coef"],
            yerr=group_df["se"],
            fmt="none",
            ecolor=llstyle.rgba(color, generalized.DI_D_ERRORBAR_ALPHA),
            capsize=0,
            elinewidth=generalized.DI_D_PLOT_MARKER_SIZE,
            label="_nolegend_",
        )

    ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)
    ax.axvline(x=generalized.DI_D_EVENT_LINE_X, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Graduation Cohort Relative to Relabel Event", fontsize=axis_fontsize)
    ax.set_ylabel(generalized._did_coef_ylabel(yvar), fontsize=axis_fontsize)  # noqa: SLF001
    generalized._format_yaxis_for_outcome(ax, yvar)  # noqa: SLF001
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_title("")
    event_ticks = list(range(generalized.PLOT_EVENT_MIN, generalized.PLOT_EVENT_MAX + 1))
    ax.set_xticks(event_ticks)
    ax.set_xlim(generalized.PLOT_EVENT_MIN - 0.6, generalized.PLOT_EVENT_MAX + 0.6)
    legend = llstyle.right_legend(ax)
    if legend is not None and text_scale != 1.0:
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)
        if legend.get_title() is not None:
            legend.get_title().set_fontsize(legend_fontsize)
    generalized._add_did_summary_text(ax, summary_text, fontsize=summary_fontsize)  # noqa: SLF001

    out_path = _foia_main_did_path(args, yvar)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(out_path.with_suffix(".csv"), index=False)
    llstyle.savefig(fig, out_path)
    return out_path


def _foia_main_raw_mean_path(args: argparse.Namespace, yvar: str) -> Path:
    return args.foia_plots_dir / f"pooled_{yvar}_event_time_treated_control_never_treated.png"


def _plot_foia_main_control_comparison_raw_means(
    panels_by_control: dict[str, pd.DataFrame],
    *,
    args: argparse.Namespace,
    yvar: str,
) -> Path | None:
    frames: list[pd.DataFrame] = []
    for control_group, panel in panels_by_control.items():
        summary = generalized.summarize_did_panel_event_time_simple_means(panel, yvar=yvar)
        if summary.empty:
            continue
        summary = summary.copy()
        summary["control_group"] = control_group
        frames.append(summary)
    if not frames:
        return None
    plot_df = pd.concat(frames, ignore_index=True, sort=False)
    plot_df["event_t"] = pd.to_numeric(plot_df["event_t"], errors="coerce")
    plot_df["treated"] = pd.to_numeric(plot_df["treated"], errors="coerce").astype("Int64")
    plot_df["mean_outcome"] = pd.to_numeric(plot_df["mean_outcome"], errors="coerce")
    plot_df = plot_df.dropna(subset=["event_t", "treated", "mean_outcome", "control_group"]).copy()
    if plot_df.empty:
        return None

    llstyle.apply_style()
    fig, ax = plt.subplots(figsize=llstyle.FIGSIZE)
    series_keys: list[tuple[str, int]] = []
    for control_group in FOIA_MAIN_COMPARISON_CONTROL_GROUPS:
        group_df = plot_df[plot_df["control_group"].eq(control_group)].copy()
        if group_df.empty:
            continue
        for treated_value in (1, 0):
            if not group_df[group_df["treated"].eq(treated_value)].empty:
                series_keys.append((control_group, treated_value))
    series_offsets = {
        key: float(offset)
        for key, offset in zip(series_keys, llstyle.offsets(len(series_keys), span=0.42))
    }
    for control_idx, control_group in enumerate(FOIA_MAIN_COMPARISON_CONTROL_GROUPS):
        group_df = plot_df[plot_df["control_group"].eq(control_group)].copy()
        if group_df.empty:
            continue
        color = FOIA_CONTROL_GROUP_COLORS.get(control_group, generalized.base.PALETTE_SEQ[control_idx % len(generalized.base.PALETTE_SEQ)])
        control_label = FOIA_CONTROL_GROUP_LABELS.get(control_group, control_group.replace("_", " ").title())
        for treated_value, role_label, linestyle, marker in (
            (1, "Treated", "-", FOIA_CONTROL_GROUP_MARKERS.get(control_group, "o")),
            (0, "Control", "--", "x" if control_group == generalized.CONTROL_GROUP_ALWAYS_STEM else "o"),
        ):
            sub = group_df[group_df["treated"].eq(treated_value)].sort_values("event_t")
            if sub.empty:
                continue
            x_vals = sub["event_t"].astype(float) + series_offsets.get((control_group, treated_value), 0.0)
            ax.plot(
                x_vals,
                sub["mean_outcome"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.6,
                markersize=6,
                label=f"{role_label}: {control_label}",
            )

    ax.axvline(x=generalized.DI_D_EVENT_LINE_X, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Graduation Cohort Relative to Relabel Event", fontsize=generalized.DI_D_PLOT_FONT_SIZE)
    ax.set_ylabel(generalized._outcome_ylabel(yvar), fontsize=generalized.DI_D_PLOT_FONT_SIZE)  # noqa: SLF001
    generalized._format_yaxis_for_outcome(ax, yvar)  # noqa: SLF001
    ax.set_title("")
    event_ticks = list(range(generalized.PLOT_EVENT_MIN, generalized.PLOT_EVENT_MAX + 1))
    ax.set_xticks(event_ticks)
    ax.set_xlim(generalized.PLOT_EVENT_MIN - 0.6, generalized.PLOT_EVENT_MAX + 0.6)
    llstyle.right_legend(ax)
    out_path = _foia_main_raw_mean_path(args, yvar)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(out_path.with_suffix(".csv"), index=False)
    llstyle.savefig(fig, out_path)
    return out_path


def _load_or_build_foia_pooled_ipeds_did_panel(
    args: argparse.Namespace,
    log: RunLog,
    *,
    control_group: str,
) -> pd.DataFrame:
    cache_path = (
        args.cache_dir
        / "foia_estimator_comparison"
        / f"pooled_ipeds_did_panel_{FOIA_ESTIMATOR_PANEL_VERSION}{_foia_control_group_suffix(control_group)}.parquet"
    )
    panel_path = args.foia_output_dir / "generalized_relabels_panel.parquet"
    if not args.force_rebuild and _file_ok(cache_path) and not _any_source_newer(cache_path, [panel_path]):
        log.hit(f"foia_pooled_ipeds_did_panel{_foia_control_group_suffix(control_group)}")
        cached = _foia_panel_with_nonresident_share(pd.read_parquet(cache_path))
        if "cnralt_share_of_ctotalt" in cached.columns:
            cached.to_parquet(cache_path, index=False)
        return cached
    relabel_panel = pd.read_parquet(panel_path)
    con = ddb.connect()
    try:
        con.sql(f"PRAGMA threads={max(1, os.cpu_count() or 1)}")
        con.sql("PRAGMA preserve_insertion_order=false")
        panel = generalized.compute_generalized_ipeds_did_panel(
            con,
            relabel_panel,
            degree_type=None,
            did_spec=generalized.DID_SPEC_COLLAPSED_UNIT_FE,
            control_group=control_group,
        )
    finally:
        con.close()
    if not panel.empty:
        panel = _foia_panel_with_nonresident_share(panel)
        panel["control_group"] = control_group
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache_path, index=False)
        log.output(cache_path)
    return panel


def _ensure_foia_main_control_comparison_raw_means(
    args: argparse.Namespace,
    log: RunLog,
    *,
    stacked_panels: dict[str, pd.DataFrame] | None = None,
) -> None:
    if not getattr(args, "foia_main_control_comparison", True):
        return
    stacked_panels = stacked_panels or {
        control_group: _load_or_build_foia_stacked_did_panel(args, log, control_group=control_group)
        for control_group in FOIA_MAIN_COMPARISON_CONTROL_GROUPS
    }
    ipeds_panels: dict[str, pd.DataFrame] = {}
    log.rebuild("foia_main_control_comparison_raw_means")
    raw_mean_progress = ProgressCounter(
        "FOIA main raw means",
        len(FOIA_ESTIMATION_APPENDICES),
    )
    for _, yvar, _title in FOIA_ESTIMATION_APPENDICES:
        panels_by_control = stacked_panels
        if yvar in generalized.PROGRAM_LEVEL_IPEDS_YVARS:
            if not ipeds_panels:
                ipeds_panels = {
                    control_group: _load_or_build_foia_pooled_ipeds_did_panel(
                        args,
                        log,
                        control_group=control_group,
                    )
                    for control_group in FOIA_MAIN_COMPARISON_CONTROL_GROUPS
                }
            panels_by_control = ipeds_panels
        out_path = _plot_foia_main_control_comparison_raw_means(
            panels_by_control,
            args=args,
            yvar=yvar,
        )
        if out_path is not None:
            log.output(out_path)
            log.output(out_path.with_suffix(".csv"))
        raw_mean_progress.advance(yvar)


def _ensure_foia_main_estimator_plots(args: argparse.Namespace, log: RunLog) -> None:
    """Render the FOIA/IPEDS main coefficient plots with the requested estimator.

    The generalized FOIA pipeline still creates raw means and grouped appendix
    assets, but its pooled coefficient plots are hard-coded TWFE.  Overwrite
    those main coefficient plots with the same selected-estimator rows used by
    the estimator appendix so the main deck honors --estimation-type.
    """
    if args.estimation_type not in ESTIMATOR_COMPARISON_TYPES:
        log.skip(f"foia_main_{args.estimation_type}_plots")
        return

    comparison_enabled = bool(getattr(args, "foia_main_control_comparison", True))
    control_groups = (
        FOIA_MAIN_COMPARISON_CONTROL_GROUPS
        if comparison_enabled
        else (FOIA_DEFAULT_CONTROL_GROUP,)
    )
    did_panels = {
        control_group: _load_or_build_foia_pooled_did_panel(args, log, control_group=control_group)
        for control_group in control_groups
    }
    ipeds_did_panels: dict[str, pd.DataFrame] = {}
    log.rebuild(f"foia_main_{args.estimation_type}_plots")
    main_plot_progress = ProgressCounter(
        f"FOIA main plots [{args.estimation_type}]",
        len(FOIA_ESTIMATION_APPENDICES),
    )
    for _, yvar, _title in FOIA_ESTIMATION_APPENDICES:
        panels_for_yvar = did_panels
        if yvar in generalized.PROGRAM_LEVEL_IPEDS_YVARS:
            if not ipeds_did_panels:
                ipeds_did_panels = {
                    control_group: _load_or_build_foia_pooled_ipeds_did_panel(
                        args,
                        log,
                        control_group=control_group,
                    )
                    for control_group in control_groups
                }
            panels_for_yvar = ipeds_did_panels
        summary_panel = panels_for_yvar[FOIA_DEFAULT_CONTROL_GROUP]
        rows_by_control: dict[str, pd.DataFrame] = {}
        for control_group in control_groups:
            cache_path = _ensure_foia_estimator_cache(
                args,
                yvar=yvar,
                estimation_type=args.estimation_type,
                did_panel=panels_for_yvar[control_group],
                log=log,
                control_group=control_group,
            )
            rows = pd.read_csv(cache_path)
            rows["estimator"] = args.estimation_type
            rows_by_control[control_group] = rows
        if comparison_enabled:
            out_path = _plot_foia_main_control_comparison_did(
                rows_by_control,
                args=args,
                yvar=yvar,
                summary_text=_foia_main_summary_text(summary_panel, yvar=yvar),
            )
        else:
            rows = rows_by_control[FOIA_DEFAULT_CONTROL_GROUP]
            out_path = generalized.plot_did_event_study_generalized(
                rows,
                yvar=yvar,
                degree_type="Pooled",
                out_dir=args.foia_plots_dir,
                file_stem=f"pooled_{yvar}_did_event_time_never_treated",
                reference_event_time=generalized.DI_D_REFERENCE_EVENT_TIME,
                summary_text=_foia_main_summary_text(summary_panel, yvar=yvar),
            )
        if out_path is None:
            raise RuntimeError(f"No FOIA main {args.estimation_type} plot was produced for {yvar}.")
        log.output(out_path)
        log.output(Path(out_path).with_suffix(".csv"))
        main_plot_progress.advance(yvar)
    if comparison_enabled:
        stacked_panels = {
            control_group: _load_or_build_foia_stacked_did_panel(args, log, control_group=control_group)
            for control_group in control_groups
        }
        _ensure_foia_main_control_comparison_raw_means(args, log, stacked_panels=stacked_panels)


def _ensure_revelio_linkedin_assets(
    *,
    args: argparse.Namespace,
    config_path: Path,
    slide_assets: list[Path],
    log: RunLog,
) -> None:
    linkedin_assets = [
        path for path in slide_assets
        if "linkedin_match_share" in path.name and str(path).startswith(str(args.figure_output_dir))
        and "estimation_type_appendix" not in str(path)
    ]
    if not linkedin_assets:
        return
    final_panel_rebuilt_econ = (
        getattr(args, "revelio_main_did_sample", "econ_only") == "econ_only"
        and "revelio_final_panel" in log.rebuilt
    )
    if (not args.force_rebuild or final_panel_rebuilt_econ) and not _missing_assets(linkedin_assets):
        log.hit("revelio_linkedin_match_assets")
        return
    log.rebuild("revelio_linkedin_match_assets")
    cfg_data = _read_yaml(config_path)
    paths = cfg_data.get("paths", {})
    build = cfg_data.get("build", {})
    updates = {
        "BUILD_OVERWRITE": bool(args.force_rebuild),
        "BUILD_EVENT_SOURCE_MODE": "econ_v2",
        "BUILD_COHORT_PLOT_YVARS": DEFAULT_COHORT_YVARS + ["linkedin_match_share"],
        "OUTPUT_DIR": str(args.revelio_econ_output_dir),
        "RELABELS_PARQUET": paths.get("relabels_parquet", indiv.cfg.RELABELS_PARQUET),
        "STAGE05_PERSON_BASELINE_PARQUET": paths.get("stage05_person_baseline_parquet", indiv.cfg.STAGE05_PERSON_BASELINE_PARQUET),
        "FOIA_PERSON_PANEL_PARQUET": paths.get("foia_person_panel_parquet", indiv.cfg.FOIA_PERSON_PANEL_PARQUET),
        "IPEDS_COST_PANEL_PARQUET": paths.get("ipeds_cost_panel_parquet", indiv.cfg.IPEDS_COST_PANEL_PARQUET),
        "BUILD_COHORT_EXTERNAL_TUITION_COL": str(build.get("cohort_external_tuition_col", "tuition7")),
    }
    with _patched_indiv_config(**updates):
        con = ddb.connect()
        try:
            con.sql(f"PRAGMA threads={max(1, os.cpu_count() or 1)}")
            con.sql("PRAGMA preserve_insertion_order=false")
            indiv.step1_relabels(con)
        finally:
            con.close()


def _linkedin_estimator_cache_path(args: argparse.Namespace, estimation_type: str) -> Path:
    return (
        args.cache_dir
        / "revelio_linkedin_match_estimator_comparison"
        / f"{estimation_type}_{ESTIMATOR_AGGREGATION_VERSION}_linkedin_match_share_e{int(args.event_window)}_b{int(args.bootstrap_reps)}_s{int(args.random_seed)}.csv"
    )


def _linkedin_did_panel_cache_path(args: argparse.Namespace) -> Path:
    return args.cache_dir / "revelio_linkedin_match_estimator_comparison" / "never_treated_econ_did_panel.parquet"


def _load_or_build_linkedin_did_panel(args: argparse.Namespace, log: RunLog) -> pd.DataFrame:
    cache_path = _linkedin_did_panel_cache_path(args)
    if not args.force_rebuild and _file_ok(cache_path):
        log.hit("revelio_linkedin_match_did_panel_for_estimator_comparison")
        return pd.read_parquet(cache_path)
    con = ddb.connect()
    try:
        con.sql(f"PRAGMA threads={max(1, os.cpu_count() or 1)}")
        con.sql("PRAGMA preserve_insertion_order=false")
        with _patched_indiv_config(
            OUTPUT_DIR=str(args.figure_output_dir),
            BUILD_OVERWRITE=bool(args.force_rebuild),
            BUILD_EVENT_SOURCE_MODE="econ_v2",
            BUILD_COHORT_PLOT_YVARS=DEFAULT_COHORT_YVARS + ["linkedin_match_share"],
        ):
            relabel_df = indiv.step1_relabels(con)
            did_panel = indiv.v2.compute_never_treated_econ_did_panel(
                con,
                relabel_df,
                foia_person_panel_path=indiv.cfg.FOIA_PERSON_PANEL_PARQUET,
                stage05_person_baseline_path=indiv.cfg.STAGE05_PERSON_BASELINE_PARQUET,
                ipeds_cost_panel_path=indiv.cfg.IPEDS_COST_PANEL_PARQUET,
                ipeds_tuition_col=getattr(indiv.cfg, "BUILD_COHORT_EXTERNAL_TUITION_COL", "tuition7"),
            )
    finally:
        con.close()
    if did_panel.empty:
        raise RuntimeError("LinkedIn match-share DiD panel is empty; cannot build estimator comparison appendix.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    did_panel.to_parquet(cache_path, index=False)
    log.rebuild("revelio_linkedin_match_did_panel_for_estimator_comparison")
    log.output(cache_path)
    return did_panel


def _compute_linkedin_estimator_rows(
    did_panel: pd.DataFrame,
    *,
    args: argparse.Namespace,
    estimation_type: str,
) -> pd.DataFrame:
    yvar = "linkedin_match_share"
    if estimation_type == "twfe":
        rows = indiv.v2.compute_did_event_study(
            did_panel=did_panel,
            yvar=yvar,
            reference_event_time=indiv.v2.DID_REFERENCE_EVENT_TIME,
            event_time_min=-int(args.event_window),
            event_time_max=int(args.event_window),
            use_weights=True,
        )
    else:
        validate_estimator_dependencies(estimation_type)
        reg_df = _prepare_generic_regression_df(
            did_panel,
            yvar=yvar,
            event_col="event_t",
            treated_col="treated",
            calendar_col="calendar_year",
            unit_col="unitid",
            relabel_col="relabel_year",
            weight_col="total_grads",
        )
        reg_df = reg_df[reg_df["event_t"].between(-int(args.event_window), int(args.event_window))].copy()
        rows = estimate_dynamic_effects(
            reg_df,
            yvar=yvar,
            estimation_type=estimation_type,
            reference_event_time=indiv.v2.DID_REFERENCE_EVENT_TIME,
            bootstrap_reps=args.bootstrap_reps,
            random_seed=args.random_seed,
            use_weights=True,
        )
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["estimator"] = estimation_type
    rows["outcome"] = yvar
    return rows


def _ensure_linkedin_estimator_cache(
    args: argparse.Namespace,
    *,
    estimation_type: str,
    did_panel: pd.DataFrame,
    log: RunLog,
) -> Path:
    cache_path = _linkedin_estimator_cache_path(args, estimation_type)
    if not _force_estimator_rebuild(args) and _file_ok(cache_path):
        log.hit(f"revelio_linkedin_match_{estimation_type}_estimator_rows")
        return cache_path
    rows = _compute_linkedin_estimator_rows(did_panel, args=args, estimation_type=estimation_type)
    if rows.empty:
        raise RuntimeError(f"No {estimation_type} estimator rows produced for LinkedIn match-share.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(cache_path, index=False)
    log.rebuild(f"revelio_linkedin_match_{estimation_type}_estimator_rows")
    log.output(cache_path)
    return cache_path


def _ensure_revelio_linkedin_estimation_type_appendix(args: argparse.Namespace, log: RunLog) -> None:
    out_path = _estimation_type_path(args.figure_output_dir, "linkedin_match_share")
    estimator_caches = [_linkedin_estimator_cache_path(args, estimator) for estimator in ESTIMATOR_COMPARISON_TYPES]
    force_estimator = _force_estimator_rebuild(args)
    if not force_estimator and _file_ok(out_path) and not _missing_assets(estimator_caches) and not _any_source_newer(out_path, estimator_caches):
        log.hit("revelio_linkedin_match_estimation_type_appendix")
        return
    did_panel = _load_or_build_linkedin_did_panel(args, log)
    frames = []
    for estimator in ESTIMATOR_COMPARISON_TYPES:
        cache_path = _ensure_linkedin_estimator_cache(
            args,
            estimation_type=estimator,
            did_panel=did_panel,
            log=log,
        )
        data = pd.read_csv(cache_path)
        data["estimator"] = estimator
        keep = ["event_t", "estimator", "coef"] + (["se"] if "se" in data.columns else [])
        frames.append(data[keep])
    rows = pd.concat(frames, ignore_index=True, sort=False)
    rows["estimator"] = rows["estimator"].map(_normalize_estimator_name)
    rows = rows[rows["estimator"].isin(ESTIMATOR_COMPARISON_TYPES)].copy()
    _plot_estimation_type_comparison(
        rows,
        out_path=out_path,
        title="Revelio linkage match share: by estimation type",
        x_col="event_t",
        x_label="Years relative to relabel event",
    )
    log.output(out_path)


def _run_pdflatex(tex_path: Path) -> None:
    if not tex_path.exists():
        raise FileNotFoundError(f"Slide deck not found: {tex_path}")
    _discard_corrupt_latex_auxiliary_files(tex_path)
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    subprocess.run(cmd, cwd=tex_path.parent, check=True)
    subprocess.run(cmd, cwd=tex_path.parent, check=True)


def _discard_corrupt_latex_auxiliary_files(tex_path: Path) -> list[Path]:
    aux_suffixes = (".aux", ".nav", ".snm", ".toc", ".out", ".vrb")
    discarded: list[Path] = []
    for suffix in aux_suffixes:
        path = tex_path.with_suffix(suffix)
        if not path.exists() or path.is_dir():
            continue
        if b"\x00" not in path.read_bytes():
            continue
        path.unlink()
        discarded.append(path)
    return discarded


def _revelio_alt_spec_config(output_mode: str) -> dict[str, str]:
    try:
        return REVELIO_ALT_SPEC_CONFIG[output_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported Revelio output mode for alternate spec: {output_mode}") from exc


def _write_revelio_alt_tex_macros(args: argparse.Namespace) -> None:
    spec = _revelio_alt_spec_config(args.revelio_output)
    path = args.revelio_alt_tex_macros
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "% Auto-generated by build_laborlunch_20260507_individual_effects.py.",
                rf"\renewcommand{{\relabelindivaltoutput}}{{{args.revelio_alt_output_dir}}}",
                rf"\renewcommand{{\revelioaltspecbuttonsuffix}}{{{spec['button_suffix']}}}",
                rf"\renewcommand{{\revelioaltspecbuttonlabel}}{{{spec['button_label']}}}",
                "",
            ]
        )
    )


def _copy_existing_revelio_alt_assets(args: argparse.Namespace, *, log: RunLog) -> None:
    pooled_files = [
        "did_att_by_variant_active_us.png",
        "did_att_by_variant_linkedin_active.png",
        "did_att_by_variant_active_positions.png",
        "did_att_by_variant_unique_employers.png",
        "did_att_by_variant_employer_tenure.png",
        "did_att_by_variant_in_school.png",
        "did_att_by_variant_compensation.png",
    ]
    event_time_files = [
        "did_att_by_variant_active_us_t3.png",
        "did_att_by_variant_linkedin_active_t3.png",
        "did_att_by_variant_active_positions_t3.png",
        "did_att_by_variant_unique_employers_t3.png",
        "did_att_by_variant_employer_tenure_t3.png",
        "did_att_by_variant_in_school_t3.png",
        "did_att_by_variant_compensation_t3.png",
        "did_att_by_variant_internship_positions_t3.png",
    ]
    files = pooled_files if args.revelio_output == "new_pooled_post" else event_time_files
    for name in files:
        src = args.revelio_econ_output_dir / name
        dst = args.revelio_alt_output_dir / name
        if src.exists() and src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            log.output(dst)


def _ensure_revelio_alternate_spec_appendix(
    panel: pd.DataFrame,
    *,
    args: argparse.Namespace,
    run_hash: str,
    log: RunLog,
) -> None:
    spec = _revelio_alt_spec_config(args.revelio_output)
    alt_mode = spec["alternate_output_mode"]
    if args.estimation_type == "twfe":
        _render_revelio_twfe(
            panel,
            output_dir=args.revelio_alt_output_dir,
            did_results_path=args.cache_dir / f"revelio_alt_twfe_{alt_mode}_{run_hash}.parquet",
            include_controls=False,
            output_mode=alt_mode,
            event_window=args.event_window,
            log=log,
            force_rebuild=_force_estimator_rebuild(args),
            progress_label=f"alternate/{alt_mode}",
        )
    else:
        _render_revelio_generic(
            panel,
            output_dir=args.revelio_alt_output_dir,
            results_path=args.cache_dir / (
                f"revelio_alt_{args.estimation_type}_{ESTIMATOR_AGGREGATION_VERSION}_{alt_mode}_{run_hash}.parquet"
            ),
            estimation_type=args.estimation_type,
            output_mode=alt_mode,
            event_window=args.event_window,
            bootstrap_reps=args.bootstrap_reps,
            random_seed=args.random_seed,
            log=log,
            force_rebuild=_force_estimator_rebuild(args),
        )
    _copy_existing_revelio_alt_assets(args, log=log)
    log.output(args.revelio_alt_output_dir)


def _copy_matching_table_if_needed(main_dir: Path, econ_dir: Path) -> None:
    src = main_dir / "revelio_matching_stats_summary_table.tex"
    dst = econ_dir / "revelio_matching_stats_summary_table.tex"
    if src.exists() and src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_assets(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    validate_estimator_dependencies(args.estimation_type)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.foia_output_dir.mkdir(parents=True, exist_ok=True)
    args.foia_plots_dir.mkdir(parents=True, exist_ok=True)
    args.figure_output_dir.mkdir(parents=True, exist_ok=True)
    args.revelio_main_output_dir.mkdir(parents=True, exist_ok=True)
    args.revelio_econ_output_dir.mkdir(parents=True, exist_ok=True)
    args.revelio_controls_output_dir.mkdir(parents=True, exist_ok=True)
    args.revelio_alt_output_dir.mkdir(parents=True, exist_ok=True)
    _write_revelio_alt_tex_macros(args)

    _progress(
        "starting build "
        f"(estimation_type={args.estimation_type}, force_rebuild={args.force_rebuild}, "
        f"force_estimator_rebuild={getattr(args, 'force_estimator_rebuild', False)}, "
        f"revelio_main_did_sample={args.revelio_main_did_sample})"
    )
    log = RunLog(
        cache_hits=[],
        rebuilt=[],
        skipped=[],
        output_paths=[],
        verbose=bool(getattr(args, "verbose_progress", False)),
    )
    slide_assets = parse_slide_asset_refs(args.tex)

    revelio_event_source_mode = _revelio_event_source_mode(args.revelio_main_did_sample)
    revelio_control_groups = _revelio_control_groups(args)
    revelio_primary_control_group = revelio_control_groups[0]
    total_stages = 13
    if len(revelio_control_groups) > 1:
        total_stages += 1
    if args.build_tex:
        total_stages += 1
    stages = StageProgress(total_stages)

    stages.next("generalized FOIA assets")
    _run_generalized_foia(args=args, slide_assets=slide_assets, log=log)
    stages.next("FOIA estimator appendix assets")
    _ensure_foia_estimation_type_appendices(args, log)
    stages.next(f"FOIA main plots with {args.estimation_type}")
    _ensure_foia_main_estimator_plots(args, log)
    stages.next(f"FOIA grouped appendix plots with {args.estimation_type}")
    _ensure_foia_grouped_estimator_appendices(args, log)

    if args.revelio_control_comparison and args.revelio_main_did_sample != "full_sample":
        log.skip("revelio_always_stem_comparison_requires_full_sample")

    config_payload = {
        "base_config": str(args.base_config),
        "horizons": args.horizons,
        "event_window": args.event_window,
        "revelio_output": args.revelio_output,
        "revelio_main_did_sample": args.revelio_main_did_sample,
        "revelio_event_source_mode": revelio_event_source_mode,
        "revelio_relabel_year_min": args.revelio_relabel_year_min,
        "revelio_relabel_year_max": args.revelio_relabel_year_max,
        "revelio_control_groups": list(revelio_control_groups),
        "revelio_outcome_panel_version": REVELIO_OUTCOME_PANEL_VERSION,
        "stage04": getattr(indiv.cfg, "STAGE04_MERGE_READY_PARQUET", ""),
        "stage05": getattr(indiv.cfg, "STAGE05_PERSON_BASELINE_PARQUET", ""),
    }
    run_hash = _hash_payload(config_payload)
    panel_cache = args.cache_dir / f"revelio_final_panel_{_slug(args.revelio_main_did_sample)}_{_slug(revelio_primary_control_group)}_{run_hash}.parquet"
    cfg_cache = args.cache_dir / f"relabel_indiv_laborlunch_20260507_{_slug(args.revelio_main_did_sample)}_{_slug(revelio_primary_control_group)}_{run_hash}.yaml"
    _materialize_revelio_config(
        base_config=args.base_config,
        out_path=cfg_cache,
        output_dir=args.revelio_econ_output_dir,
        panel_path=panel_cache,
        did_path=args.cache_dir / f"revelio_did_{run_hash}.parquet",
        horizons=args.horizons,
        event_window=args.event_window,
        event_source_mode=revelio_event_source_mode,
        relabel_year_min=args.revelio_relabel_year_min,
        relabel_year_max=args.revelio_relabel_year_max,
        control_group=revelio_primary_control_group,
        did_plot_mode="pooled_post_by_horizon" if args.revelio_output == "new_pooled_post" else "event_study_by_cohort",
        include_controls=False,
    )
    stages.next("Revelio final panel")
    panel = _build_revelio_variant_panels(
        config_path=cfg_cache,
        output_dir=args.revelio_econ_output_dir,
        cache_path=panel_cache,
        force_rebuild=args.force_rebuild,
        horizons=args.horizons,
        event_window=args.event_window,
        event_source_mode=revelio_event_source_mode,
        relabel_year_min=args.revelio_relabel_year_min,
        relabel_year_max=args.revelio_relabel_year_max,
        control_group=revelio_primary_control_group,
        log=log,
    )
    log.output(panel_cache)

    stages.next("Revelio LinkedIn match assets")
    _ensure_revelio_linkedin_assets(
        args=args,
        config_path=cfg_cache,
        slide_assets=slide_assets,
        log=log,
    )
    stages.next("Revelio LinkedIn estimator appendix")
    _ensure_revelio_linkedin_estimation_type_appendix(args, log)

    stages.next("Revelio matching summary table")
    table_path = write_revelio_matching_table(
        panel,
        args.revelio_main_output_dir / "revelio_matching_stats_summary_table.tex",
    )
    _copy_matching_table_if_needed(args.revelio_main_output_dir, args.revelio_econ_output_dir)
    log.output(table_path)

    revelio_panels_by_control = {revelio_primary_control_group: panel}
    if len(revelio_control_groups) > 1:
        stages.next("additional Revelio control-group panels")
        control_progress = ProgressCounter(
            "Revelio control-group panels",
            len(revelio_control_groups) - 1,
            report_every=1,
        )
        for control_group in revelio_control_groups[1:]:
            control_panel_cache = args.cache_dir / (
                f"revelio_final_panel_{_slug(args.revelio_main_did_sample)}_{_slug(control_group)}_{run_hash}.parquet"
            )
            control_cfg_cache = args.cache_dir / (
                f"relabel_indiv_laborlunch_20260507_{_slug(args.revelio_main_did_sample)}_{_slug(control_group)}_{run_hash}.yaml"
            )
            _materialize_revelio_config(
                base_config=args.base_config,
                out_path=control_cfg_cache,
                output_dir=args.revelio_econ_output_dir,
                panel_path=control_panel_cache,
                did_path=args.cache_dir / f"revelio_did_{_slug(control_group)}_{run_hash}.parquet",
                horizons=args.horizons,
                event_window=args.event_window,
                event_source_mode=revelio_event_source_mode,
                relabel_year_min=args.revelio_relabel_year_min,
                relabel_year_max=args.revelio_relabel_year_max,
                control_group=control_group,
                did_plot_mode="pooled_post_by_horizon" if args.revelio_output == "new_pooled_post" else "event_study_by_cohort",
                include_controls=False,
            )
            revelio_panels_by_control[control_group] = _build_revelio_variant_panels(
                config_path=control_cfg_cache,
                output_dir=args.revelio_econ_output_dir,
                cache_path=control_panel_cache,
                force_rebuild=args.force_rebuild,
                horizons=args.horizons,
                event_window=args.event_window,
                event_source_mode=revelio_event_source_mode,
                relabel_year_min=args.revelio_relabel_year_min,
                relabel_year_max=args.revelio_relabel_year_max,
                control_group=control_group,
                log=log,
            )
            log.output(control_panel_cache)
            control_progress.advance(control_group)

    if args.estimation_type == "twfe":
        stages.next("Revelio TWFE main plots")
        if len(revelio_panels_by_control) > 1:
            frames = []
            for control_group, control_panel in revelio_panels_by_control.items():
                result = _render_revelio_twfe(
                    control_panel,
                    output_dir=args.revelio_econ_output_dir,
                    did_results_path=args.cache_dir / (
                        f"revelio_econ_twfe_{args.revelio_output}_{_slug(control_group)}_{run_hash}.parquet"
                    ),
                    include_controls=False,
                    output_mode=args.revelio_output,
                    event_window=args.event_window,
                    log=log,
                    render_plots=control_group == revelio_primary_control_group,
                    force_rebuild=args.force_rebuild,
                )
                if not result.empty:
                    result = result.copy()
                    result["control_group"] = control_group
                    frames.append(result)
                    if control_group == revelio_primary_control_group:
                        _seed_revelio_estimator_cache(
                            result.drop(columns=["control_group"], errors="ignore"),
                            args=args,
                            run_hash=run_hash,
                            estimation_type="twfe",
                            log=log,
                        )
            if frames:
                comparison_results = pd.concat(frames, ignore_index=True, sort=False)
                _plot_revelio_control_comparison_results(
                    comparison_results,
                    output_dir=args.revelio_econ_output_dir,
                    output_mode=args.revelio_output,
                    sample_label=REVELIO_SAMPLE_LABELS.get(args.revelio_main_did_sample, args.revelio_main_did_sample),
                )
        else:
            main_twfe = _render_revelio_twfe(
                panel,
                output_dir=args.revelio_econ_output_dir,
                did_results_path=args.cache_dir / f"revelio_econ_twfe_{args.revelio_output}_{run_hash}.parquet",
                include_controls=False,
                output_mode=args.revelio_output,
                event_window=args.event_window,
                log=log,
                force_rebuild=args.force_rebuild,
            )
            _seed_revelio_estimator_cache(
                main_twfe,
                args=args,
                run_hash=run_hash,
                estimation_type="twfe",
                log=log,
            )
        stages.next("Revelio controlled TWFE plots")
        _render_revelio_twfe(
            panel,
            output_dir=args.revelio_controls_output_dir,
            did_results_path=args.cache_dir / f"revelio_controls_twfe_{args.revelio_output}_{run_hash}.parquet",
            include_controls=True,
            output_mode=args.revelio_output,
            event_window=args.event_window,
            log=log,
            force_rebuild=args.force_rebuild,
        )
    else:
        stages.next(f"Revelio {args.estimation_type} main plots")
        if len(revelio_panels_by_control) > 1:
            frames = []
            for control_group, control_panel in revelio_panels_by_control.items():
                result = _render_revelio_generic(
                    control_panel,
                    output_dir=args.revelio_econ_output_dir,
                    results_path=args.cache_dir / (
                        f"revelio_econ_{args.estimation_type}_{ESTIMATOR_AGGREGATION_VERSION}_{args.revelio_output}_{_slug(control_group)}_{run_hash}.parquet"
                    ),
                    estimation_type=args.estimation_type,
                    output_mode=args.revelio_output,
                    event_window=args.event_window,
                    bootstrap_reps=args.bootstrap_reps,
                    random_seed=args.random_seed,
                    log=log,
                    render_plots=control_group == revelio_primary_control_group,
                    force_rebuild=_force_estimator_rebuild(args),
                    progress_label=f"main/{args.revelio_output}/{control_group}",
                )
                if not result.empty:
                    result = result.copy()
                    result["control_group"] = control_group
                    frames.append(result)
                    if control_group == revelio_primary_control_group:
                        _seed_revelio_estimator_cache(
                            result.drop(columns=["control_group"], errors="ignore"),
                            args=args,
                            run_hash=run_hash,
                            estimation_type=args.estimation_type,
                            log=log,
                        )
            if frames:
                comparison_results = pd.concat(frames, ignore_index=True, sort=False)
                _plot_revelio_control_comparison_results(
                    comparison_results,
                    output_dir=args.revelio_econ_output_dir,
                    output_mode=args.revelio_output,
                    sample_label=REVELIO_SAMPLE_LABELS.get(args.revelio_main_did_sample, args.revelio_main_did_sample),
                )
        else:
            main_generic = _render_revelio_generic(
                panel,
                output_dir=args.revelio_econ_output_dir,
                results_path=args.cache_dir / f"revelio_econ_{args.estimation_type}_{ESTIMATOR_AGGREGATION_VERSION}_{args.revelio_output}_{run_hash}.parquet",
                estimation_type=args.estimation_type,
                output_mode=args.revelio_output,
                event_window=args.event_window,
                bootstrap_reps=args.bootstrap_reps,
                random_seed=args.random_seed,
                log=log,
                force_rebuild=_force_estimator_rebuild(args),
                progress_label=f"main/{args.revelio_output}/{revelio_primary_control_group}",
            )
            _seed_revelio_estimator_cache(
                main_generic,
                args=args,
                run_hash=run_hash,
                estimation_type=args.estimation_type,
                log=log,
            )
        stages.next("Revelio controlled TWFE plots")
        _render_revelio_twfe(
            panel,
            output_dir=args.revelio_controls_output_dir,
            did_results_path=args.cache_dir / f"revelio_controls_twfe_{args.revelio_output}_{run_hash}.parquet",
            include_controls=True,
            output_mode=args.revelio_output,
            event_window=args.event_window,
            log=log,
            force_rebuild=args.force_rebuild,
        )
    stages.next("Revelio alternate-spec appendix")
    _ensure_revelio_alternate_spec_appendix(panel, args=args, run_hash=run_hash, log=log)
    stages.next("Revelio estimation-type appendices")
    _ensure_revelio_estimation_type_appendices(panel, args, run_hash, log)

    stages.next("slide asset validation")
    missing = _missing_assets(slide_assets)
    if missing:
        raise RuntimeError(
            "Missing non-firm slide assets after build:\n"
            + "\n".join(f"  {path}" for path in missing)
        )

    if args.build_tex:
        stages.next("pdflatex")
        _run_pdflatex(args.tex)
        log.output(args.tex.with_suffix(".pdf"))

    manifest = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed": _elapsed(start),
        "config_hash": run_hash,
        "estimation_type": args.estimation_type,
        "revelio_output": args.revelio_output,
        "revelio_main_did_sample": args.revelio_main_did_sample,
        "revelio_event_source_mode": revelio_event_source_mode,
        "revelio_relabel_year_min": args.revelio_relabel_year_min,
        "revelio_relabel_year_max": args.revelio_relabel_year_max,
        "revelio_control_groups": list(revelio_control_groups),
        "force_rebuild": args.force_rebuild,
        "force_estimator_rebuild": getattr(args, "force_estimator_rebuild", False),
        "verbose_progress": bool(getattr(args, "verbose_progress", False)),
        "build_tex": args.build_tex,
        "input_paths": {
            "tex": str(args.tex),
            "base_config": str(args.base_config),
        },
        "output_dirs": {
            "foia_plots": str(args.foia_plots_dir),
            "figures": str(args.figure_output_dir),
            "revelio_main": str(args.revelio_main_output_dir),
            "revelio_econ": str(args.revelio_econ_output_dir),
            "revelio_controls": str(args.revelio_controls_output_dir),
            "revelio_alt": str(args.revelio_alt_output_dir),
        },
        "cache_hits": log.cache_hits,
        "rebuilt": log.rebuilt,
        "skipped": log.skipped,
        "output_paths": sorted(set(log.output_paths)),
        "slide_asset_count": len(slide_assets),
        "missing_assets": [],
    }
    manifest_path = args.cache_dir / f"laborlunch_20260507_individual_effects_manifest_{run_hash}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Completed in {_elapsed(start)}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-firm individual-effects assets for slides_laborlunch_20260507.",
        allow_abbrev=False,
    )
    parser.add_argument("--estimation-type", choices=sorted(ESTIMATION_TYPES), default="sun_abraham")
    parser.add_argument("--revelio-output", choices=sorted(REVELIO_OUTPUT_MODES), default="new_pooled_post")
    parser.add_argument("--revelio-main-did-sample", choices=sorted(REVELIO_MAIN_DID_SAMPLE_MODES), default="full_sample")
    parser.add_argument(
        "--revelio-control-comparison",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For full-sample Revelio DiD runs, overlay never-treated and always-STEM "
            "control specifications in the main Revelio coefficient plots."
        ),
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--force-estimator-rebuild", action="store_true")
    parser.add_argument("--build-tex", action="store_true")
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        help="Print detailed cache hit/rebuild/skip messages in addition to compact progress counters.",
    )
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--base-config", type=Path, default=CODE_ROOT / "configs" / "relabel_indiv.yaml")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--foia-output-dir", type=Path, default=DEFAULT_FOIA_OUTPUT_DIR)
    parser.add_argument("--foia-plots-dir", type=Path, default=DEFAULT_FOIA_PLOTS_DIR)
    parser.add_argument("--figure-output-dir", type=Path, default=DEFAULT_FIGURE_OUTPUT_DIR)
    parser.add_argument("--revelio-main-output-dir", type=Path, default=DEFAULT_REVELIO_MAIN_OUTPUT_DIR)
    parser.add_argument("--revelio-econ-output-dir", type=Path, default=DEFAULT_REVELIO_ECON_OUTPUT_DIR)
    parser.add_argument("--revelio-controls-output-dir", type=Path, default=DEFAULT_REVELIO_CONTROLS_OUTPUT_DIR)
    parser.add_argument("--revelio-alt-output-dir", type=Path, default=DEFAULT_REVELIO_ALT_OUTPUT_DIR)
    parser.add_argument("--revelio-alt-tex-macros", type=Path, default=DEFAULT_REVELIO_ALT_TEX_MACROS)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--revelio-relabel-year-min", type=int, default=DEFAULT_REVELIO_RELABEL_YEAR_MIN)
    parser.add_argument("--revelio-relabel-year-max", type=int, default=DEFAULT_REVELIO_RELABEL_YEAR_MAX)
    parser.add_argument("--event-window", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=199)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--sun-abraham-use-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use total_grads weights in the FOIA Sun-Abraham feols call and "
            "cohort-stratum aggregation. The unweighted default leaves feols "
            "unweighted and aggregates event-time estimates by treated cell counts."
        ),
    )
    parser.add_argument(
        "--foia-main-control-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overlay never-treated and always-STEM specifications in main FOIA "
            "DID/raw-means plots. Use --no-foia-main-control-comparison for "
            "single-spec main plots."
        ),
    )
    args = parser.parse_args(argv)
    args.horizons = sorted({int(h) for h in args.horizons if 0 <= int(h) <= REVELIO_MAX_HORIZON})
    if not args.horizons:
        raise ValueError(f"At least one horizon between 0 and {REVELIO_MAX_HORIZON} is required.")
    if (
        args.revelio_relabel_year_min is not None
        and args.revelio_relabel_year_max is not None
        and int(args.revelio_relabel_year_min) > int(args.revelio_relabel_year_max)
    ):
        raise ValueError("--revelio-relabel-year-min cannot exceed --revelio-relabel-year-max.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_assets(args)


if __name__ == "__main__":
    main()
