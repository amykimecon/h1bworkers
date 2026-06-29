"""Efficient cross-design comparison suite for company shift-share analyses.

This module intentionally orchestrates lower-level cached builders and prepared
panels.  It does not call the heavyweight module ``main()`` entrypoints for each
specification.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]

try:
    import laborlunch_plot_style as llstyle
except ImportError:  # pragma: no cover
    llstyle = None  # type: ignore[assignment]

try:
    import pyfixest as pf
except ImportError:  # pragma: no cover
    pf = None  # type: ignore[assignment]

try:
    from linearmodels.panel import PanelOLS
except ImportError:  # pragma: no cover
    PanelOLS = None  # type: ignore[assignment,misc]

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None  # type: ignore[assignment,misc]

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:  # pragma: no cover
    NearestNeighbors = None  # type: ignore[assignment,misc]

try:
    from company_shift_share.config_loader import get_cfg_section, load_config
    from company_shift_share import shift_share_analysis as ssa
    from company_shift_share.exposure_event_study import _get_or_build_index_result
    from company_shift_share.matched_exposure_design import (
        DEFAULT_CONFIG_PATH as MATCHED_DEFAULT_CONFIG_PATH,
        DEFAULT_SHIFT_SHARE_CONFIG_PATH,
        DEFAULT_SOURCE_CONFIG_PATH,
        _load_base_configs,
        build_matching_feature_frame_from_wrds,
        build_shift_share_components_from_configs,
        build_source_analysis_panel_from_inputs,
        build_stacked_did_panel,
        match_high_to_low_exposure,
    )
    from company_shift_share.revelio_company_features import load_or_build_company_features
    from company_shift_share.source_exposure_data import (
        load_or_build_design3_position_outcomes_cache,
        load_or_build_source_analysis_panel,
        load_or_build_wrds_company_year_workforce_cache,
        load_or_build_wrds_school_flows_cache,
    )
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import get_cfg_section, load_config
    from company_shift_share import shift_share_analysis as ssa
    from company_shift_share.exposure_event_study import _get_or_build_index_result
    from company_shift_share.matched_exposure_design import (
        DEFAULT_CONFIG_PATH as MATCHED_DEFAULT_CONFIG_PATH,
        DEFAULT_SHIFT_SHARE_CONFIG_PATH,
        DEFAULT_SOURCE_CONFIG_PATH,
        _load_base_configs,
        build_matching_feature_frame_from_wrds,
        build_shift_share_components_from_configs,
        build_source_analysis_panel_from_inputs,
        build_stacked_did_panel,
        match_high_to_low_exposure,
    )
    from company_shift_share.revelio_company_features import load_or_build_company_features
    from company_shift_share.source_exposure_data import (
        load_or_build_design3_position_outcomes_cache,
        load_or_build_source_analysis_panel,
        load_or_build_wrds_company_year_workforce_cache,
        load_or_build_wrds_school_flows_cache,
    )


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "company_shift_share_design_comparison.yaml"
)
SUITE_VERSION = "2026-05-01-design1-state-panel"
ERRORBAR_INTERVAL_ALPHA = 0.4
PLOT_MARKER_SIZE = 11
MULTI_PLOT_MARKER_SIZE = 9
PLOT_LINE_WIDTH = 1.5
PLOT_PALETTE = ("#2e8b57", "#e07a5f", "#4c78a8", "#a05195", "#ffb000", "#0072b2", "#009e73")

DEFAULT_OUTCOMES = [
    "y_cst_lag0",
    "y_new_hires_lag0",
    "y_new_hires_foreign_lag0",
    "y_new_hires_native_lag0",
    "avg_tenure_years_lag0",
]
FOREIGN_NEW_HIRE_OUTCOME = "y_new_hires_foreign_lag0"
DESIGN3_POSITION_OUTCOMES = [
    "y_new_hires_foreign_lag0",
    "y_new_hires_native_lag0",
    "y_new_hires_foreign_opt_likely_lag0",
    "y_new_hires_foreign_masters_lag0",
    "y_new_hires_native_masters_lag0",
    "y_new_hires_foreign_opt_likely_masters_lag0",
    "avg_tenure_opt_likely_jobs_lag0",
    "avg_tenure_foreign_new_hires_lag0",
    "avg_tenure_new_hires_lag0",
    "avg_tenure_foreign_new_hires_masters_lag0",
    "avg_tenure_new_hires_masters_lag0",
    "y_intern_positions_opt_likely_lag0",
    "y_intern_positions_foreign_lag0",
    "y_intern_positions_opt_likely_foreign_lag0",
]
DESIGN3_NEW_GRAD_OVERRIDE_OUTCOMES = {
    "y_new_hires_foreign_lag0",
    "y_new_hires_native_lag0",
    "y_new_hires_foreign_opt_likely_lag0",
    "y_new_hires_foreign_masters_lag0",
    "y_new_hires_native_masters_lag0",
    "y_new_hires_foreign_opt_likely_masters_lag0",
    "avg_tenure_foreign_new_hires_lag0",
    "avg_tenure_new_hires_lag0",
    "avg_tenure_foreign_new_hires_masters_lag0",
    "avg_tenure_new_hires_masters_lag0",
}
COUNT_OUTCOME_PREFIXES = ("y_",)
FIRST_STAGE_TYPES = {"ppml", "ols_continuous", "ols_binary", "ols_ihs"}
UNITS = {"firm", "local_market"}
DESIGN_NAMES = ("shift_share", "event_study", "stacked_did")
DATA_STAGE_ORDER = [
    "source_analysis_panel",
    "company_features",
    "shift_share_panel",
    "opt_probability_index",
    "workforce_panel",
    "design3_position_outcomes",
    "school_flows",
    "shift_share_components",
    "matched_design_inputs",
]


@dataclass
class StageRecord:
    name: str
    source_path: Optional[str] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    reused_disk: bool = False
    reused_memory: bool = False
    rebuilt: bool = False
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class _ConditionalPPMLFit:
    params: pd.Series
    std_errors: pd.Series
    covariance: pd.DataFrame
    nobs: int
    converged: bool
    message: str


@dataclass
class _LinearFit:
    params: pd.Series
    std_errors: pd.Series
    covariance: pd.DataFrame
    nobs: int


@dataclass
class _ConditionalPPMLDesign:
    x: np.ndarray
    y: np.ndarray
    group_ids: np.ndarray
    group_starts: np.ndarray
    group_totals: np.ndarray
    param_names: list[str]
    kept_rows: int
    dropped_zero_outcome_rows: int
    dropped_zero_outcome_units: int


@dataclass
class _ConditionalPPMLObjective:
    design: _ConditionalPPMLDesign
    sum_yx: np.ndarray

    def objective_gradient(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        x = self.design.x
        y = self.design.y
        gid = self.design.group_ids
        starts = self.design.group_starts
        totals = self.design.group_totals
        eta = x @ theta
        max_eta = np.maximum.reduceat(eta, starts)
        exp_eta = np.exp(eta - max_eta[gid])
        denom = np.add.reduceat(exp_eta, starts)
        log_denom = max_eta + np.log(denom)
        p = exp_eta / denom[gid]
        expected_x = np.add.reduceat(p[:, None] * x, starts, axis=0)
        score = self.sum_yx - (totals[:, None] * expected_x).sum(axis=0)
        neg_loglik = -float(y @ eta - totals @ log_denom)
        return neg_loglik, -score

    def hessian(self, theta: np.ndarray) -> np.ndarray:
        x = self.design.x
        gid = self.design.group_ids
        starts = self.design.group_starts
        totals = self.design.group_totals
        eta = x @ theta
        max_eta = np.maximum.reduceat(eta, starts)
        exp_eta = np.exp(eta - max_eta[gid])
        denom = np.add.reduceat(exp_eta, starts)
        p = exp_eta / denom[gid]
        expected_x = np.add.reduceat(p[:, None] * x, starts, axis=0)
        weighted_x = x * (p * totals[gid])[:, None]
        expected_xx = x.T @ weighted_x
        correction = expected_x.T @ (totals[:, None] * expected_x)
        return expected_xx - correction

    def cluster_meat(self, theta: np.ndarray) -> np.ndarray:
        x = self.design.x
        y = self.design.y
        gid = self.design.group_ids
        starts = self.design.group_starts
        totals = self.design.group_totals
        eta = x @ theta
        max_eta = np.maximum.reduceat(eta, starts)
        exp_eta = np.exp(eta - max_eta[gid])
        denom = np.add.reduceat(exp_eta, starts)
        p = exp_eta / denom[gid]
        observed = np.add.reduceat(y[:, None] * x, starts, axis=0)
        expected = totals[:, None] * np.add.reduceat(p[:, None] * x, starts, axis=0)
        group_score = observed - expected
        return group_score.T @ group_score


@dataclass
class ComparisonDataStore:
    """Lazy, in-memory cache for expensive comparison-suite inputs."""

    cfg: dict
    config_path: Optional[Path] = None
    force_rebuild_base: bool = False
    memory_cache: bool = True
    _cache: dict[str, object] = field(default_factory=dict)
    manifest: dict[str, object] = field(default_factory=lambda: {"stages": {}})

    def get(self, name: str, loader: Callable[[], object]) -> object:
        verbose = _verbose(self._comparison_cfg())
        if self.memory_cache and name in self._cache:
            record = StageRecord(name=name, reused_memory=True)
            self._update_manifest(record)
            if verbose:
                _log(f"{_stage_label(name)} reusing in-memory cache")
            return self._cache[name]
        if verbose:
            _log(f"{_stage_label(name)} starting")
            _log(f"remaining data stages after this: {_remaining_stages(name, DATA_STAGE_ORDER)}", indent=1)
        started = time.perf_counter()
        try:
            value = loader()
            record = self._record_for_value(name, value, started)
        except Exception as exc:
            record = StageRecord(name=name, elapsed_seconds=time.perf_counter() - started, error=str(exc))
            self._update_manifest(record)
            if verbose:
                _log(f"{_stage_label(name)} failed after {_fmt_seconds(record.elapsed_seconds)}: {exc}", indent=1)
            raise
        if self.memory_cache:
            self._cache[name] = value
        self._update_manifest(record)
        if verbose:
            details = []
            if record.rows is not None and record.cols is not None:
                details.append(f"{record.rows:,} rows x {record.cols:,} cols")
            if record.reused_disk:
                details.append("disk cache")
            if record.rebuilt:
                details.append("rebuilt")
            _log(
                f"{_stage_label(name)} done in {_fmt_seconds(record.elapsed_seconds)}"
                + (f" | {' | '.join(details)}" if details else "")
            )
        return value

    def _record_for_value(self, name: str, value: object, started: float) -> StageRecord:
        source_path = None
        df = value
        if isinstance(value, tuple) and value and isinstance(value[0], pd.DataFrame):
            df = value[0]
        if isinstance(value, pd.DataFrame):
            rows, cols = int(len(value)), int(len(value.columns))
        elif isinstance(df, pd.DataFrame):
            rows, cols = int(len(df)), int(len(df.columns))
        else:
            rows, cols = None, None
        source_meta = getattr(value, "_comparison_source_path", None)
        if source_meta is not None:
            source_path = str(source_meta)
        return StageRecord(
            name=name,
            source_path=source_path,
            rows=rows,
            cols=cols,
            reused_disk=bool(source_path and Path(source_path).exists() and not self.force_rebuild_base),
            rebuilt=bool(self.force_rebuild_base),
            elapsed_seconds=time.perf_counter() - started,
        )

    def _update_manifest(self, record: StageRecord) -> None:
        stages = self.manifest.setdefault("stages", {})
        if not isinstance(stages, dict):
            stages = {}
            self.manifest["stages"] = stages
        stages[record.name] = {
            "source_path": record.source_path,
            "source_mtime": _mtime(record.source_path),
            "rows": record.rows,
            "cols": record.cols,
            "reused_disk": record.reused_disk,
            "reused_memory": record.reused_memory,
            "rebuilt": record.rebuilt,
            "elapsed_seconds": round(record.elapsed_seconds, 3),
            "error": record.error,
        }

    def shift_share_panel(self) -> pd.DataFrame:
        return self.get("shift_share_panel", self._load_shift_share_panel)  # type: ignore[return-value]

    def source_analysis_panel(self) -> pd.DataFrame:
        return self.get("source_analysis_panel", self._load_source_analysis_panel)  # type: ignore[return-value]

    def company_features(self) -> pd.DataFrame:
        return self.get("company_features", self._load_company_features)  # type: ignore[return-value]

    def workforce_panel(self) -> pd.DataFrame:
        return self.get("workforce_panel", self._load_workforce_panel)  # type: ignore[return-value]

    def school_flows(self) -> pd.DataFrame:
        return self.get("school_flows", self._load_school_flows)  # type: ignore[return-value]

    def design3_position_outcomes(self) -> pd.DataFrame:
        return self.get("design3_position_outcomes", self._load_design3_position_outcomes)  # type: ignore[return-value]

    def shift_share_components(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.get("shift_share_components", self._load_shift_share_components)  # type: ignore[return-value]

    def opt_probability_index(self) -> pd.DataFrame:
        return self.get("opt_probability_index", self._load_opt_probability_index)  # type: ignore[return-value]

    def matched_design_inputs(self) -> dict[str, pd.DataFrame]:
        return self.get("matched_design_inputs", self._load_matched_design_inputs)  # type: ignore[return-value]

    def _comparison_cfg(self) -> dict:
        return get_cfg_section(self.cfg, "design_comparison")

    def _paths_cfg(self) -> dict:
        return get_cfg_section(self.cfg, "paths")

    def _load_parquet_if_available(self, path_value: object) -> Optional[pd.DataFrame]:
        if path_value is None or self.force_rebuild_base:
            return None
        path = Path(str(path_value))
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _load_required_base_cache(self, source_cfg: dict, path_key: str, stage_name: str) -> Optional[pd.DataFrame]:
        if self.force_rebuild_base or not bool(self._comparison_cfg().get("trust_existing_base_caches", True)):
            return None
        path_value = get_cfg_section(source_cfg, "paths").get(path_key)
        cached = self._load_parquet_if_available(path_value)
        if cached is not None:
            if _verbose(self._comparison_cfg()):
                _log(f"{stage_name}: reading existing parquet cache", indent=1)
                _log(f"path: {path_value}", indent=2)
            return cached
        raise FileNotFoundError(
            f"{stage_name} cache is missing at paths.{path_key}={path_value!r}. "
            "The comparison suite is configured with trust_existing_base_caches=true "
            "and force_rebuild_base=false, so it will not rebuild WRDS-derived caches. "
            "Point the config to an existing parquet or explicitly set force_rebuild_base=true."
        )

    def _load_shift_share_panel(self) -> pd.DataFrame:
        candidates: list[object] = [self._paths_cfg().get("shift_share_analysis_panel")]
        out_dir = self._paths_cfg().get("out_dir")
        if out_dir is not None:
            candidates.append(Path(str(out_dir)) / "prepared_panels" / "shift_share.parquet")
        candidates.append(get_cfg_section(load_config(DEFAULT_SHIFT_SHARE_CONFIG_PATH), "paths").get("analysis_panel"))
        checked: list[str] = []
        for path in candidates:
            if path is None:
                continue
            checked.append(str(path))
            cached = self._load_parquet_if_available(path)
            if cached is not None:
                return cached
        raise FileNotFoundError(
            "Shift-share analysis panel cache is missing. Run shift_share_analysis once, "
            "or set paths.shift_share_analysis_panel to an existing parquet. "
            f"Checked: {checked}"
        )

    def _load_source_analysis_panel(self) -> pd.DataFrame:
        cfg_path = _path_or_default(self._comparison_cfg().get("source_config_path"), DEFAULT_SOURCE_CONFIG_PATH)
        source_cfg = load_config(cfg_path)
        cached = self._load_required_base_cache(source_cfg, "opt_exposure_analysis_panel_out", "source analysis panel")
        if cached is not None:
            return cached
        exp_cfg = get_cfg_section(source_cfg, "exposure_event_study")
        panel, _ = load_or_build_source_analysis_panel(
            config_path=cfg_path,
            cfg=source_cfg,
            data_min_t=int(self._comparison_cfg().get("data_min_t", exp_cfg.get("data_min_t", 2010))),
            data_max_t=int(self._comparison_cfg().get("data_max_t", exp_cfg.get("data_max_t", 2022))),
            force_rebuild=self.force_rebuild_base,
        )
        return panel

    def _load_company_features(self) -> pd.DataFrame:
        cfg_path = _path_or_default(self._comparison_cfg().get("source_config_path"), DEFAULT_SOURCE_CONFIG_PATH)
        source_cfg = load_config(cfg_path)
        cached = self._load_required_base_cache(source_cfg, "company_features_out", "company features")
        if cached is not None:
            return cached
        cmp_cfg = self._comparison_cfg()
        features, _ = load_or_build_company_features(
            config_path=cfg_path,
            cfg=source_cfg,
            feature_year_min=int(cmp_cfg.get("baseline_start", 2010)),
            feature_year_max=int(cmp_cfg.get("baseline_end", 2013)),
            force_rebuild=self.force_rebuild_base,
        )
        return features

    def _load_workforce_panel(self) -> pd.DataFrame:
        cfg_path = _path_or_default(self._comparison_cfg().get("source_config_path"), DEFAULT_SOURCE_CONFIG_PATH)
        source_cfg = load_config(cfg_path)
        cached = self._load_required_base_cache(source_cfg, "wrds_company_year_workforce_out", "WRDS company-year workforce")
        if cached is not None:
            return cached
        cmp_cfg = self._comparison_cfg()
        workforce, _ = load_or_build_wrds_company_year_workforce_cache(
            config_path=cfg_path,
            cfg=source_cfg,
            year_min=int(cmp_cfg.get("data_min_t", 2010)),
            year_max=int(cmp_cfg.get("data_max_t", 2022)),
            force_rebuild=self.force_rebuild_base,
            cache_only=not self.force_rebuild_base,
        )
        return workforce

    def _load_school_flows(self) -> pd.DataFrame:
        cfg_path = _path_or_default(self._comparison_cfg().get("source_config_path"), DEFAULT_SOURCE_CONFIG_PATH)
        source_cfg = load_config(cfg_path)
        cached = self._load_required_base_cache(source_cfg, "wrds_school_flows_out", "WRDS school flows")
        if cached is not None:
            return cached
        cmp_cfg = self._comparison_cfg()
        school_flows, _ = load_or_build_wrds_school_flows_cache(
            config_path=cfg_path,
            cfg=source_cfg,
            year_min=int(cmp_cfg.get("data_min_t", 2010)),
            year_max=int(cmp_cfg.get("data_max_t", 2022)),
            force_rebuild=self.force_rebuild_base,
        )
        return school_flows

    def _load_design3_position_outcomes(self) -> pd.DataFrame:
        cfg_path = _path_or_default(self._comparison_cfg().get("source_config_path"), DEFAULT_SOURCE_CONFIG_PATH)
        source_cfg = load_config(cfg_path)
        cmp_cfg = self._comparison_cfg()
        try:
            source = self.source_analysis_panel()
            firm_ids = source["c"].drop_duplicates() if "c" in source.columns else None
        except Exception:
            firm_ids = None
        outcomes, _ = load_or_build_design3_position_outcomes_cache(
            config_path=cfg_path,
            cfg=source_cfg,
            year_min=int(cmp_cfg.get("data_min_t", 2010)),
            year_max=int(cmp_cfg.get("data_max_t", 2022)),
            force_rebuild=bool(cmp_cfg.get("force_rebuild_design3_position_outcomes", self.force_rebuild_base)),
            firm_ids=firm_ids,
            position_history_path=cmp_cfg.get("design3_position_history_path"),
            opt_likely_soc2=cmp_cfg.get("design3_opt_likely_soc2"),
            intern_max_days=cmp_cfg.get("design3_intern_max_days"),
        )
        return outcomes

    def _load_shift_share_components(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        matched_cfg = _matched_base_cfg(self.cfg)
        shift_cfg, source_cfg = _load_base_configs(matched_cfg)
        cmp_cfg = self._comparison_cfg()
        return build_shift_share_components_from_configs(
            shift_cfg=shift_cfg,
            source_cfg=source_cfg,
            source_flow_year_min=int(cmp_cfg.get("data_min_t", 2010)),
            source_flow_year_max=int(cmp_cfg.get("data_max_t", 2022)),
            data_min_t=int(cmp_cfg.get("data_min_t", 2010)),
            data_max_t=int(cmp_cfg.get("data_max_t", 2022)),
            force_rebuild_inputs=self.force_rebuild_base,
        )

    def _load_opt_probability_index(self) -> pd.DataFrame:
        cfg_path = _path_or_default(self._comparison_cfg().get("source_config_path"), DEFAULT_SOURCE_CONFIG_PATH)
        source_cfg = load_config(cfg_path)
        paths = get_cfg_section(source_cfg, "paths")
        cached = self._load_parquet_if_available(paths.get("opt_probability_index_out"))
        if cached is not None:
            return cached
        if bool(self._comparison_cfg().get("trust_existing_base_caches", True)) and not self.force_rebuild_base:
            raise FileNotFoundError(
                f"OPT probability index cache is missing at paths.opt_probability_index_out="
                f"{paths.get('opt_probability_index_out')!r}. The comparison suite will not rebuild "
                "WRDS-derived/model caches unless force_rebuild_base=true."
            )
        exp_cfg = copy.deepcopy(get_cfg_section(source_cfg, "exposure_event_study"))
        exp_cfg["exposure_version"] = "opt_probability_index"
        pred_df, _, _ = _get_or_build_index_result(
            self.source_analysis_panel(),
            source_cfg,
            exp_cfg,
            config_path=cfg_path,
            feature_cache={},
            index_cache={},
        )
        return pred_df

    def _load_matched_design_inputs(self) -> dict[str, pd.DataFrame]:
        matched_cfg = _matched_base_cfg(self.cfg)
        shift_cfg, source_cfg = _load_base_configs(matched_cfg)
        cmp_cfg = self._comparison_cfg()
        panel, firms, selected_meta, workforce = build_source_analysis_panel_from_inputs(
            source_cfg=source_cfg,
            data_min_t=int(cmp_cfg.get("data_min_t", 2010)),
            data_max_t=int(cmp_cfg.get("data_max_t", 2022)),
            force_rebuild_inputs=self.force_rebuild_base,
        )
        feature_min = int(cmp_cfg.get("baseline_start", 2010))
        feature_max = int(cmp_cfg.get("baseline_end", 2013))
        matching_features = build_matching_feature_frame_from_wrds(
            firms=firms,
            selected_meta=selected_meta,
            wrds_annual=workforce,
            feature_year_min=feature_min,
            feature_year_max=feature_max,
        )
        _, _, transition_shares, instrument_panel = build_shift_share_components_from_configs(
            shift_cfg=shift_cfg,
            source_cfg=source_cfg,
            source_flow_year_min=int(cmp_cfg.get("data_min_t", 2010)),
            source_flow_year_max=int(cmp_cfg.get("data_max_t", 2022)),
            data_min_t=int(cmp_cfg.get("data_min_t", 2010)),
            data_max_t=int(cmp_cfg.get("data_max_t", 2022)),
            force_rebuild_inputs=self.force_rebuild_base,
        )
        return {
            "source_panel": panel,
            "firms": firms,
            "selected_meta": selected_meta,
            "workforce": workforce,
            "matching_features": matching_features,
            "transition_shares": transition_shares,
            "instrument_panel": instrument_panel,
        }


def run_design_comparison(
    *,
    config_path: str | Path | None = None,
    cfg: Optional[dict] = None,
    force_rebuild_base: Optional[bool] = None,
    unit: Optional[str] = None,
    first_stage_type: Optional[str] = None,
    first_stage_col: Optional[str] = None,
    designs: Optional[Sequence[str] | str] = None,
    stacked_exposures: Optional[Sequence[str] | str] = None,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    cfg_full = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
    cmp_cfg = get_cfg_section(cfg_full, "design_comparison")
    paths_cfg = get_cfg_section(cfg_full, "paths")
    if force_rebuild_base is not None:
        cmp_cfg["force_rebuild_base"] = bool(force_rebuild_base)
    if unit is not None:
        cmp_cfg["unit"] = unit
    if first_stage_type is not None:
        cmp_cfg["first_stage_type"] = first_stage_type
    if first_stage_col is not None:
        cmp_cfg["first_stage_col"] = first_stage_col
    if designs is not None:
        cmp_cfg["designs_to_run"] = _parse_design_list(designs)
    if stacked_exposures is not None:
        cmp_cfg["stacked_exposures_to_run"] = _list_cfg(stacked_exposures)
    if out_dir is not None:
        paths_cfg["out_dir"] = str(out_dir)
    _validate_comparison_config(cmp_cfg)
    selected_designs = _selected_designs(cmp_cfg)

    output_dir = Path(str(paths_cfg.get("out_dir", "/tmp/company_shift_share_design_comparison")))
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    prepared_dir = output_dir / "prepared_panels"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    if _verbose(cmp_cfg):
        _log("starting efficient design comparison suite")
        _log(f"version: {SUITE_VERSION}", indent=1)
        _log(f"module: {Path(__file__).resolve()}", indent=1)
        _log(f"output directory: {output_dir}", indent=1)
        _log(
            "config: "
            f"unit={cmp_cfg.get('unit')} | first_stage={cmp_cfg.get('first_stage_type')} | "
            f"years={cmp_cfg.get('data_min_t')}-{cmp_cfg.get('data_max_t')} | "
            f"baseline={cmp_cfg.get('baseline_start')}-{cmp_cfg.get('baseline_end')} | "
            f"designs={','.join(selected_designs)}",
            indent=1,
        )
        _log(
            "cache policy: "
            f"trust_existing_base_caches={bool(cmp_cfg.get('trust_existing_base_caches', True))} | "
            f"force_rebuild_base={bool(cmp_cfg.get('force_rebuild_base', False))} | "
            f"memory_cache={bool(cmp_cfg.get('memory_cache', True))}",
            indent=1,
        )

    store = ComparisonDataStore(
        cfg=cfg_full,
        config_path=Path(config_path) if config_path is not None else None,
        force_rebuild_base=bool(cmp_cfg.get("force_rebuild_base", False)),
        memory_cache=bool(cmp_cfg.get("memory_cache", True)),
    )
    started = time.perf_counter()
    if _verbose(cmp_cfg):
        _log(f"Work plan: prepare panels -> {' -> '.join(selected_designs)} -> outputs")
    prepared = prepare_comparison_panels(store, cmp_cfg)
    if bool(cmp_cfg.get("write_prepared_panels", True)):
        if _verbose(cmp_cfg):
            _log(f"writing prepared panels to {prepared_dir}")
        for name, panel in prepared.items():
            if isinstance(panel, pd.DataFrame):
                panel_started = time.perf_counter()
                panel.to_parquet(prepared_dir / f"{name}.parquet", index=False)
                if _verbose(cmp_cfg):
                    _log(f"wrote prepared panel {name}: {_panel_summary_text(panel)} in {_fmt_seconds(time.perf_counter() - panel_started)}", indent=1)

    all_results: list[pd.DataFrame] = []
    if "shift_share" in selected_designs:
        if _verbose(cmp_cfg):
            _log("Design 1/3: shift_share starting")
        all_results.append(run_shift_share_design(prepared["shift_share"], cmp_cfg, figures_dir))
        if _verbose(cmp_cfg):
            _log("Design 1/3: shift_share finished")
    else:
        if _verbose(cmp_cfg):
            _log("Design 1/3: shift_share skipped")
    if "event_study" in selected_designs:
        if _verbose(cmp_cfg):
            _log("Design 2/3: event_study starting")
        all_results.append(run_event_study_design(prepared["event_study"], store, cmp_cfg, figures_dir))
        if _verbose(cmp_cfg):
            _log("Design 2/3: event_study finished")
    else:
        if _verbose(cmp_cfg):
            _log("Design 2/3: event_study skipped")
    if "stacked_did" in selected_designs:
        if _verbose(cmp_cfg):
            _log("Design 3/3: stacked_did starting")
        all_results.append(run_stacked_did_design(prepared["stacked_did"], cmp_cfg, figures_dir))
        if _verbose(cmp_cfg):
            _log("Design 3/3: stacked_did finished")
    else:
        if _verbose(cmp_cfg):
            _log("Design 3/3: stacked_did skipped")
    nonempty_results = [df for df in all_results if df is not None and not df.empty]
    coef_df = pd.concat(nonempty_results, ignore_index=True) if nonempty_results else pd.DataFrame()
    coef_df = add_baseline_effect_stats(coef_df, prepared, cmp_cfg)
    if _verbose(cmp_cfg):
        _log(f"writing outputs | coefficient rows={len(coef_df):,}")
    coef_path = tables_dir / "all_design_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    final_table = build_final_comparison_table(coef_df)
    final_csv = tables_dir / "design_comparison_first_stage_rf.csv"
    final_tex = tables_dir / "design_comparison_first_stage_rf.tex"
    final_table.to_csv(final_csv, index=False)
    try:
        final_tex.write_text(format_final_comparison_table_latex(final_table))
    except Exception:
        final_tex.write_text(final_table.to_string(index=False))

    summary = build_prepared_panel_summary(prepared)
    summary_path = tables_dir / "prepared_panel_summary.csv"
    summary.to_csv(summary_path, index=False)

    store.manifest["selected_config"] = {
        "unit": cmp_cfg.get("unit"),
        "first_stage_type": cmp_cfg.get("first_stage_type"),
        "designs_to_run": selected_designs,
        "stacked_exposures_to_run": _stacked_exposures_to_run(cmp_cfg),
        "outcome_cols": _list_cfg(cmp_cfg.get("outcome_cols", DEFAULT_OUTCOMES)),
        "baseline_start": int(cmp_cfg.get("baseline_start", 2010)),
        "baseline_end": int(cmp_cfg.get("baseline_end", 2013)),
        "data_min_t": int(cmp_cfg.get("data_min_t", 2010)),
        "data_max_t": int(cmp_cfg.get("data_max_t", 2022)),
        "min_pre_avg_employment": float(cmp_cfg.get("min_pre_avg_employment", 10)),
        "exclude_size_year_fe": bool(cmp_cfg.get("exclude_size_year_fe", True)),
    }
    store.manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    manifest_path = output_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(store.manifest, indent=2, sort_keys=True))
    if _verbose(cmp_cfg):
        _log(f"finished in {_fmt_seconds(time.perf_counter() - started)}")
        _log(f"coefficients: {coef_path}", indent=1)
        _log(f"final table: {final_csv}", indent=1)
        _log(f"cache manifest: {manifest_path}", indent=1)
    return {
        "out_dir": str(output_dir),
        "coefficients": coef_df,
        "final_table": final_table,
        "coef_path": str(coef_path),
        "final_csv": str(final_csv),
        "final_tex": str(final_tex),
        "manifest_path": str(manifest_path),
        "prepared_panel_summary": str(summary_path),
    }


def prepare_comparison_panels(store: ComparisonDataStore, cmp_cfg: dict) -> dict[str, pd.DataFrame]:
    if _verbose(cmp_cfg):
        _log("Preparing analysis panels")
    source = _standardize_panel_ids(store.source_analysis_panel())
    features = _standardize_feature_ids(store.company_features())
    shift = _standardize_panel_ids(store.shift_share_panel())
    if _verbose(cmp_cfg):
        _log(f"loaded source panel: {_panel_summary_text(source)}", indent=1)
        _log(f"loaded shift-share panel: {_panel_summary_text(shift)}", indent=1)
        _log(f"loaded company features: {_panel_summary_text(features)}", indent=1)
    if _needs_workforce_design_outcomes(source, cmp_cfg):
        source = attach_workforce_design_outcomes(source, store.workforce_panel(), cmp_cfg)
    if _needs_design3_position_outcomes(source, cmp_cfg):
        source = attach_design3_position_outcomes(source, store.design3_position_outcomes(), cmp_cfg)
    source = ensure_design_outcome_derivations(source, cmp_cfg)
    shift = ensure_shift_share_share_variants(shift, cmp_cfg)
    shift = attach_company_features(shift, features, cmp_cfg)
    shift = prepare_shift_share_state_panel(shift, cmp_cfg)
    event_panel = attach_company_features(source, features, cmp_cfg)
    event_panel = attach_shift_share_exposures(event_panel, shift)
    opt_index = store.opt_probability_index()
    event_panel = attach_opt_probability_index(event_panel, opt_index)
    event_panel = recompute_baseline_size_growth(event_panel, cmp_cfg)

    unit = str(cmp_cfg.get("unit", "firm"))
    shift_before = shift.copy()
    event_before = event_panel.copy()
    shift = filter_analysis_sample(shift, cmp_cfg)
    event_panel = filter_analysis_sample(event_panel, cmp_cfg)
    if _verbose(cmp_cfg):
        _log_attrition("shift_share sample filter", shift_before, shift, cmp_cfg)
        _log_attrition("event_study/stacked sample filter", event_before, event_panel, cmp_cfg)
    if unit == "local_market":
        if _verbose(cmp_cfg):
            _log(f"aggregating to local labor market using {cmp_cfg.get('local_market_col', 'company_metro_feature')}")
        shift = aggregate_to_local_market(shift, cmp_cfg)
        event_panel = aggregate_to_local_market(event_panel, cmp_cfg)
        if _verbose(cmp_cfg):
            _log(f"local-market shift_share panel: {_panel_summary_text(shift)}", indent=1)
            _log(f"local-market event_study/stacked panel: {_panel_summary_text(event_panel)}", indent=1)
    stacked_panel = event_panel.copy()
    return {
        "shift_share": shift,
        "event_study": event_panel,
        "stacked_did": stacked_panel,
    }


def run_shift_share_design(panel: pd.DataFrame, cmp_cfg: dict, figures_dir: Path) -> pd.DataFrame:
    variants = _shift_share_variants(cmp_cfg, panel)
    if _verbose(cmp_cfg):
        _log(
            f"shift_share specs={len(variants)} | outcomes={len(_available_outcomes(panel, cmp_cfg))} | "
            f"horizons={cmp_cfg.get('dynamic_horizon_start', -4)}..{cmp_cfg.get('dynamic_horizon_end', 5)}",
            indent=1,
        )
    rows: list[dict[str, object]] = []
    for idx, (spec_name, instrument_col) in enumerate(variants, start=1):
        spec_started = time.perf_counter()
        if _verbose(cmp_cfg):
            _log(f"shift_share spec {idx}/{len(variants)}: {spec_name} ({instrument_col})", indent=1)
        spec_panel = prepare_shift_share_dynamic_panel(panel, cmp_cfg, instrument_col=instrument_col)
        work = _prepare_regression_panel(spec_panel, cmp_cfg, instrument_col)
        if work.empty:
            if _verbose(cmp_cfg):
                _log("skipped: prepared regression panel is empty", indent=2)
            continue
        rows.extend(
            estimate_horizon_coefficients(
                work,
                design="shift_share",
                spec=spec_name,
                exposure_col=instrument_col,
                cmp_cfg=cmp_cfg,
                time_col="horizon",
                horizons=range(int(cmp_cfg.get("dynamic_horizon_start", -4)), int(cmp_cfg.get("dynamic_horizon_end", 5)) + 1),
                event_year=None,
            )
        )
        if _verbose(cmp_cfg):
            _log(f"finished {spec_name} in {_fmt_seconds(time.perf_counter() - spec_started)}", indent=2)
    out = pd.DataFrame(rows)
    out = add_baseline_effect_stats(out, {"shift_share": panel}, cmp_cfg)
    out.attrs["show_figures"] = bool(cmp_cfg.get("show_figures", False))
    plot_dynamic_design(out, "shift_share", "horizon", figures_dir)
    plot_raw_means_by_exposure(panel, variants, cmp_cfg, "shift_share", figures_dir, stats=out)
    return out


def prepare_shift_share_dynamic_panel(
    panel: pd.DataFrame,
    cmp_cfg: dict,
    *,
    instrument_col: Optional[str] = None,
) -> pd.DataFrame:
    """Mirror shift_share_analysis dynamic sample restrictions for Design 1."""
    if panel.empty or "t" not in panel.columns or "c" not in panel.columns:
        return panel.copy()
    work = panel.copy()
    start_t = int(cmp_cfg.get("shift_share_start_t", 2013))
    end_t = int(cmp_cfg.get("data_max_t", 2022))
    t = pd.to_numeric(work["t"], errors="coerce")
    before = work.copy()
    work = work.loc[t.between(start_t, end_t)].copy()
    if instrument_col is not None and instrument_col in work.columns:
        work = work.loc[work[instrument_col].notna()].copy()
    primary_outcome = str(_list_cfg(cmp_cfg.get("outcome_cols", DEFAULT_OUTCOMES))[0] if _list_cfg(cmp_cfg.get("outcome_cols", DEFAULT_OUTCOMES)) else "y_cst_lag0")
    if primary_outcome in work.columns:
        work = work.loc[work[primary_outcome].notna()].copy()
    if str(cmp_cfg.get("first_stage_type", "ppml")) == "ppml":
        try:
            x_col = _x_col(work, cmp_cfg)
            work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
            work = work.loc[work[x_col].notna() & work[x_col].ge(0)].copy()
        except ValueError:
            pass
    if bool(cmp_cfg.get("shift_share_enforce_balanced_panel", True)) and not work.empty:
        years = sorted(pd.to_numeric(work["t"], errors="coerce").dropna().unique())
        obs_per_unit = work.groupby("c")["t"].nunique()
        balanced_units = obs_per_unit[obs_per_unit == len(years)].index
        work = work.loc[work["c"].isin(balanced_units)].copy()
    if _verbose(cmp_cfg):
        _log("shift_share dynamic sample restriction", indent=1)
        _log(
            f"before: {len(before):,} rows | {before['c'].nunique(dropna=True):,} units",
            indent=2,
        )
        _log(
            f"after start_t={start_t}, end_t={end_t}, instrument={instrument_col}, balanced="
            f"{bool(cmp_cfg.get('shift_share_enforce_balanced_panel', True))}: "
            f"{len(work):,} rows | {work['c'].nunique(dropna=True):,} units",
            indent=2,
        )
    return work.reset_index(drop=True)


def run_event_study_design(panel: pd.DataFrame, store: ComparisonDataStore, cmp_cfg: dict, figures_dir: Path) -> pd.DataFrame:
    del store  # the panel already contains cached opt_probability_index output.
    rows: list[dict[str, object]] = []
    variants = [
        ("school_opt_share_binary", "school_opt_share_new_hire_annual_pre_level", "binary"),
        ("opt_probability_index_binary", "predicted_prob", "binary"),
    ]
    available_variants = [(a, b, c) for a, b, c in variants if b in panel.columns]
    if _verbose(cmp_cfg):
        _log(
            f"event_study specs={len(available_variants)}/{len(variants)} available | "
            f"outcomes={len(_available_outcomes(panel, cmp_cfg))} | ref_year={cmp_cfg.get('event_ref_year', 2014)}",
            indent=1,
        )
    for idx, (spec_name, exposure_col, style) in enumerate(available_variants, start=1):
        spec_started = time.perf_counter()
        if _verbose(cmp_cfg):
            _log(f"event_study spec {idx}/{len(available_variants)}: {spec_name} ({style}, {exposure_col})", indent=1)
        if exposure_col not in panel.columns:
            continue
        prep_started = time.perf_counter()
        work = _prepare_regression_panel(panel, cmp_cfg, exposure_col)
        if work.empty:
            continue
        before_exposure_drop = len(work)
        work = work.loc[pd.to_numeric(work[exposure_col], errors="coerce").notna()].copy()
        work[exposure_col] = pd.to_numeric(work[exposure_col], errors="coerce")
        if exposure_col == "predicted_prob":
            work = _filter_opt_index_analysis_sample(work, cmp_cfg)
        if _verbose(cmp_cfg):
            _log(
                f"dropped missing exposure rows: {before_exposure_drop - len(work):,}; "
                f"remaining rows={len(work):,} | units={_n_units_text(work, cmp_cfg)}",
                indent=2,
            )
        if work.empty:
            continue
        if style == "binary":
            med = work[exposure_col].median()
            work[f"{exposure_col}_above_median"] = (work[exposure_col] >= med).astype(float)
            reg_col = f"{exposure_col}_above_median"
            if _verbose(cmp_cfg):
                _log(f"binary exposure median={med:.6g}; using {reg_col}", indent=2)
        else:
            reg_col = exposure_col
        retained_cols = _event_study_needed_cols(work, cmp_cfg, reg_col)
        work = work.loc[:, retained_cols].copy()
        if _verbose(cmp_cfg):
            _log(
                f"prepared event-study regression panel: {_panel_summary_text(work)} "
                f"| retained_cols={len(retained_cols)} "
                f"in {_fmt_seconds(time.perf_counter() - prep_started)}",
                indent=2,
            )
        rows.extend(
            estimate_event_year_coefficients(
                work,
                design="event_study",
                spec=spec_name,
                exposure_col=reg_col,
                cmp_cfg=cmp_cfg,
                ref_year=int(cmp_cfg.get("event_ref_year", 2014)),
            )
        )
        if _verbose(cmp_cfg):
            _log(f"finished {spec_name} in {_fmt_seconds(time.perf_counter() - spec_started)}", indent=2)
    out = pd.DataFrame(rows)
    out = add_baseline_effect_stats(out, {"event_study": panel}, cmp_cfg)
    out.attrs["show_figures"] = bool(cmp_cfg.get("show_figures", False))
    plot_dynamic_design(out, "event_study", "year", figures_dir)
    plot_raw_means_by_exposure(panel, [(v[0], v[1]) for v in variants if v[1] in panel.columns], cmp_cfg, "event_study", figures_dir, stats=out)
    return out


def run_stacked_did_design(panel: pd.DataFrame, cmp_cfg: dict, figures_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    exposure_candidates = [
        ("ihmp_share", _first_present(panel, ["z_ct_ihmp_share", "z_ct"])),
        ("international_share", _first_present(panel, ["z_ct_international_share"])),
        ("opt_takeup", _first_present(panel, ["any_opt_hires_correction_aware", "masters_opt_hires_correction_aware"])),
    ]
    selected_exposures = set(_stacked_exposures_to_run(cmp_cfg))
    available_exposures = [(name, col) for name, col in exposure_candidates if col is not None and name in selected_exposures]
    if _verbose(cmp_cfg):
        _log(
            f"stacked_did exposure families={len(available_exposures)} | "
            f"jumps={cmp_cfg.get('stacked_jump_min_year', cmp_cfg.get('stacked_min_cohort_year', 2013))}-"
            f"{cmp_cfg.get('stacked_jump_max_year', cmp_cfg.get('stacked_max_cohort_year', 2016))} | "
            f"window=±{cmp_cfg.get('stacked_pre_years', 3)}/{cmp_cfg.get('stacked_post_years', 3)}",
            indent=1,
        )
    for exp_idx, (exposure_name, exposure_col) in enumerate(available_exposures, start=1):
        if exposure_col is None:
            continue
        exp_started = time.perf_counter()
        if _verbose(cmp_cfg):
            _log(f"stacked_did exposure {exp_idx}/{len(available_exposures)}: {exposure_name} ({exposure_col})", indent=1)
        detect_started = time.perf_counter()
        event_df = detect_largest_jump_events(
            panel,
            exposure_col=exposure_col,
            cohort_min_year=int(cmp_cfg.get("stacked_jump_min_year", cmp_cfg.get("stacked_min_cohort_year", 2013))),
            cohort_max_year=int(cmp_cfg.get("stacked_jump_max_year", cmp_cfg.get("stacked_max_cohort_year", 2016))),
            min_jump=float(cmp_cfg.get("stacked_min_event_jump", 0.0)),
            treated_jump_percentile=float(cmp_cfg.get("stacked_treated_jump_percentile", 75)),
            control_min_jump_percentile=float(cmp_cfg.get("stacked_control_min_jump_percentile", 25)),
        )
        if _verbose(cmp_cfg):
            n_treated = int(event_df.get("treated", pd.Series(dtype=int)).fillna(0).sum()) if not event_df.empty else 0
            n_controls = (
                int(event_df.get("control_eligible", pd.Series(dtype=int)).fillna(0).sum())
                if not event_df.empty and "control_eligible" in event_df.columns
                else int(event_df.get("treated", pd.Series(dtype=int)).eq(0).sum())
            )
            threshold = (
                float(pd.to_numeric(event_df.get("treated_jump_threshold"), errors="coerce").dropna().iloc[0])
                if not event_df.empty and "treated_jump_threshold" in event_df.columns and pd.to_numeric(event_df["treated_jump_threshold"], errors="coerce").notna().any()
                else float(cmp_cfg.get("stacked_min_event_jump", 0.0))
            )
            _log(
                f"detected events: treated={n_treated:,} | controls={n_controls:,} | valid={len(event_df):,} "
                f"(treated jump threshold={threshold:.6g}) in {_fmt_seconds(time.perf_counter() - detect_started)}",
                indent=2,
            )
        for matching_style in _stacked_matching_styles(cmp_cfg):
            match_started = time.perf_counter()
            if _verbose(cmp_cfg):
                _log(f"stacked_did matching style: {matching_style}", indent=2)
            stacked = build_comparison_stacked_panel(
                panel,
                event_df,
                cmp_cfg,
                matching_style=matching_style,
                exposure_col=exposure_col,
            )
            if stacked.empty:
                if _verbose(cmp_cfg):
                    _log("skipped: stacked panel is empty", indent=3)
                continue
            if _verbose(cmp_cfg):
                _log(f"stacked panel: {_panel_summary_text(stacked)} | stacks={stacked['stack_id'].nunique():,}", indent=3)
            spec = f"{matching_style}_{exposure_name}"
            rows.extend(estimate_stacked_event_coefficients(stacked, "sun_abraham", spec, cmp_cfg))
            plot_stacked_raw_means(
                stacked,
                spec,
                cmp_cfg,
                figures_dir,
                stats=add_baseline_effect_stats(pd.DataFrame(rows), {"stacked_did": panel}, cmp_cfg),
            )
            if _verbose(cmp_cfg):
                _log(f"finished {matching_style} in {_fmt_seconds(time.perf_counter() - match_started)}", indent=3)
        if _verbose(cmp_cfg):
            _log(f"finished exposure {exposure_name} in {_fmt_seconds(time.perf_counter() - exp_started)}", indent=2)
    out = pd.DataFrame(rows)
    out = add_baseline_effect_stats(out, {"stacked_did": panel}, cmp_cfg)
    out.attrs["show_figures"] = bool(cmp_cfg.get("show_figures", False))
    plot_dynamic_design(out, "stacked_did", "rel_time", figures_dir)
    return out


def estimate_horizon_coefficients(
    panel: pd.DataFrame,
    *,
    design: str,
    spec: str,
    exposure_col: str,
    cmp_cfg: dict,
    time_col: str,
    horizons: Iterable[int],
    event_year: Optional[int],
) -> list[dict[str, object]]:
    del time_col, event_year
    rows: list[dict[str, object]] = []
    unit_col = _unit_id_col(cmp_cfg)
    x_col = _x_col(panel, cmp_cfg)
    for horizon in horizons:
        work = panel.copy()
        reg_col = _normalized_col(work, exposure_col)
        x_dyn = _dynamic_value(work, x_col, int(horizon), unit_col=unit_col)
        lhs, estimator = _first_stage_lhs(work, x_dyn, cmp_cfg)
        rows.append(
            _estimate_single_term(
                work,
                lhs=lhs,
                term=reg_col,
                family="first_stage",
                design=design,
                spec=spec,
                estimator=estimator,
                time_value=int(horizon),
                time_name="horizon",
                outcome_col=x_col,
                cmp_cfg=cmp_cfg,
            )
        )
        for outcome in _available_outcomes(work, cmp_cfg):
            y_dyn = _dynamic_value(work, outcome, int(horizon), unit_col=unit_col)
            y_lhs = f"dyn_{_safe_name(outcome)}_h{_lag_suffix(horizon)}"
            work[y_lhs] = _transform_outcome(y_dyn, outcome, cmp_cfg)
            rows.append(
                _estimate_single_term(
                    work,
                    lhs=y_lhs,
                    term=reg_col,
                    family="reduced_form",
                    design=design,
                    spec=spec,
                    estimator="ols",
                    time_value=int(horizon),
                    time_name="horizon",
                    outcome_col=outcome,
                    cmp_cfg=cmp_cfg,
                )
            )
    return rows


def estimate_event_year_coefficients(
    panel: pd.DataFrame,
    *,
    design: str,
    spec: str,
    exposure_col: str,
    cmp_cfg: dict,
    ref_year: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    work = panel.copy()
    reg_col = _normalized_col(work, exposure_col)
    years = sorted(int(v) for v in pd.to_numeric(work["t"], errors="coerce").dropna().unique())
    x_col = _x_col(work, cmp_cfg)
    lhs, estimator = _first_stage_lhs(work, pd.to_numeric(work[x_col], errors="coerce"), cmp_cfg)
    interaction_cols = _add_year_interactions(work, reg_col, years, ref_year)
    term_to_time = {col: year for year, col in interaction_cols}
    term_cols = list(term_to_time.keys())
    fe = _fe_cols(work, cmp_cfg)
    outcomes = _available_outcomes(work, cmp_cfg)
    outcome_lhs: list[tuple[str, str]] = []
    for outcome in outcomes:
        y_lhs = f"reg_{_safe_name(outcome)}"
        work[y_lhs] = _transform_outcome(pd.to_numeric(work[outcome], errors="coerce"), outcome, cmp_cfg)
        outcome_lhs.append((outcome, y_lhs))

    # Build the common event-study design once.  Each outcome only needs an
    # outcome-specific missingness filter before calling pyfixest.
    base_cols = _dedupe_existing([*term_cols, *fe, lhs, *[col for _, col in outcome_lhs]], work)
    base_panel = _regression_rows(work.loc[:, base_cols], [*term_cols, *fe])
    if _verbose(cmp_cfg):
        _log(
            f"{design}/{spec}: built event-study design matrix | "
            f"rows={len(base_panel):,} | cols={len(base_panel.columns):,} | "
            f"units={_n_units_text(base_panel, cmp_cfg)} | "
            f"years={_year_range_text(base_panel)} | terms={len(term_cols)} | FE={'+'.join(fe)}",
            indent=2,
        )
    total_models = 1 + len(outcome_lhs)
    event_year = int(cmp_cfg.get("event_year", 2016))
    fs_rows = _estimate_multi_term_event_from_base(
        base_panel,
        lhs=lhs,
        term_cols=term_cols,
        family="first_stage",
        design=design,
        spec=spec,
        estimator=estimator,
        outcome_col=x_col,
        cmp_cfg=cmp_cfg,
        time_name="year",
        term_to_time=term_to_time,
        fe_cols=fe,
        model_index=1,
        model_total=total_models,
    )
    _annotate_event_time(fs_rows, event_year=event_year)
    rows.extend(fs_rows)
    rows.append(
        _reference_result(
            family="first_stage",
            design=design,
            spec=spec,
            estimator=estimator,
            time_name="year",
            time_value=ref_year,
            outcome_col=x_col,
            extra={"event_time": int(ref_year - event_year), "fe_cols": "+".join(fe)},
        )
    )
    for model_index, (outcome, y_lhs) in enumerate(outcome_lhs, start=2):
        rf_rows = _estimate_multi_term_event_from_base(
            base_panel,
            lhs=y_lhs,
            term_cols=term_cols,
            family="reduced_form",
            design=design,
            spec=spec,
            estimator="ols",
            outcome_col=outcome,
            cmp_cfg=cmp_cfg,
            time_name="year",
            term_to_time=term_to_time,
            fe_cols=fe,
            model_index=model_index,
            model_total=total_models,
        )
        _annotate_event_time(rf_rows, event_year=event_year)
        rows.extend(rf_rows)
        rows.append(
            _reference_result(
                family="reduced_form",
                design=design,
                spec=spec,
                estimator="ols",
                time_name="year",
                time_value=ref_year,
                outcome_col=outcome,
                extra={"event_time": int(ref_year - event_year), "fe_cols": "+".join(fe)},
            )
        )
    return rows


def estimate_stacked_event_coefficients(
    stacked_panel: pd.DataFrame,
    estimator_label: str,
    spec: str,
    cmp_cfg: dict,
) -> list[dict[str, object]]:
    if estimator_label == "sun_abraham":
        return _estimate_stacked_sun_abraham_event_coefficients(stacked_panel, spec, cmp_cfg)

    rows: list[dict[str, object]] = []
    event_times = sorted(int(v) for v in pd.to_numeric(stacked_panel["rel_time"], errors="coerce").dropna().unique())
    ref_event_time = int(cmp_cfg.get("stacked_ref_event_time", -1))
    non_ref = [v for v in event_times if v != ref_event_time]
    if not non_ref:
        return rows
    work = _ensure_fe_columns(stacked_panel.copy(), cmp_cfg)
    terms: list[tuple[int, str]] = []
    for rel_time in non_ref:
        col = f"stack_treated_{_lag_suffix(rel_time)}"
        work[col] = (work["treated"].eq(1) & work["rel_time"].eq(rel_time)).astype(float)
        terms.append((rel_time, col))
    x_col = _x_col(work, cmp_cfg)
    lhs, fs_estimator = _first_stage_lhs(work, pd.to_numeric(work[x_col], errors="coerce"), cmp_cfg)
    fe_cols = _stacked_fe_cols(work, cmp_cfg, estimator_label)
    rows.extend(
        _estimate_multi_term_event(
            work,
            lhs=lhs,
            terms=terms,
            family="first_stage",
            design="stacked_did",
            spec=spec,
            estimator=fs_estimator,
            outcome_col=x_col,
            cmp_cfg=cmp_cfg,
            time_name="rel_time",
            term_to_time={col: rel for rel, col in terms},
            fe_cols=fe_cols,
            cluster_col=_stacked_cluster_col(estimator_label, cmp_cfg),
        )
    )
    rows.append(
        _reference_result(
            family="first_stage",
            design="stacked_did",
            spec=spec,
            estimator=fs_estimator,
            time_name="rel_time",
            time_value=ref_event_time,
            outcome_col=x_col,
            extra={"fe_cols": "+".join(fe_cols)},
        )
    )
    for outcome in _available_outcomes(work, cmp_cfg):
        y_lhs = f"reg_{_safe_name(outcome)}"
        work[y_lhs] = _transform_outcome(pd.to_numeric(work[outcome], errors="coerce"), outcome, cmp_cfg)
        rows.extend(
            _estimate_multi_term_event(
                work,
                lhs=y_lhs,
                terms=terms,
                family="reduced_form",
                design="stacked_did",
                spec=spec,
                estimator="ols",
                outcome_col=outcome,
                cmp_cfg=cmp_cfg,
                time_name="rel_time",
                term_to_time={col: rel for rel, col in terms},
                fe_cols=fe_cols,
                cluster_col=_stacked_cluster_col(estimator_label, cmp_cfg),
            )
        )
        rows.append(
            _reference_result(
                family="reduced_form",
                design="stacked_did",
                spec=spec,
                estimator="ols",
                time_name="rel_time",
                time_value=ref_event_time,
                outcome_col=outcome,
                extra={"fe_cols": "+".join(fe_cols)},
            )
        )
    return rows


def _estimate_stacked_sun_abraham_event_coefficients(
    stacked_panel: pd.DataFrame,
    spec: str,
    cmp_cfg: dict,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    event_times = sorted(int(v) for v in pd.to_numeric(stacked_panel["rel_time"], errors="coerce").dropna().unique())
    ref_event_time = int(cmp_cfg.get("stacked_ref_event_time", -1))
    non_ref = [v for v in event_times if v != ref_event_time]
    if not non_ref:
        return rows
    work = _ensure_fe_columns(stacked_panel.copy(), cmp_cfg)
    term_to_time, term_meta = _stacked_sun_abraham_terms(work, ref_event_time)
    if not term_to_time:
        return [
            _empty_result("first_stage", "stacked_did", spec, str(cmp_cfg.get("first_stage_type", "ppml")), "rel_time", v, _x_col(work, cmp_cfg), "no cohort-specific event-time terms")
            for v in non_ref
        ]
    x_col = _x_col(work, cmp_cfg)
    lhs, fs_estimator = _first_stage_lhs(work, pd.to_numeric(work[x_col], errors="coerce"), cmp_cfg)
    fe_cols = _stacked_fe_cols(work, cmp_cfg, "sun_abraham")
    cluster_col = _stacked_cluster_col("sun_abraham", cmp_cfg)
    rows.extend(
        _estimate_stacked_sun_abraham_family(
            work,
            lhs=lhs,
            term_to_time=term_to_time,
            term_meta=term_meta,
            event_times=event_times,
            ref_event_time=ref_event_time,
            family="first_stage",
            spec=spec,
            estimator=fs_estimator,
            outcome_col=x_col,
            cmp_cfg=cmp_cfg,
            fe_cols=fe_cols,
            cluster_col=cluster_col,
        )
    )
    for outcome in _available_outcomes(work, cmp_cfg):
        y_lhs = f"reg_{_safe_name(outcome)}"
        work[y_lhs] = _transform_outcome(pd.to_numeric(work[outcome], errors="coerce"), outcome, cmp_cfg)
        rows.extend(
            _estimate_stacked_sun_abraham_family(
                work,
                lhs=y_lhs,
                term_to_time=term_to_time,
                term_meta=term_meta,
                event_times=event_times,
                ref_event_time=ref_event_time,
                family="reduced_form",
                spec=spec,
                estimator="ols",
                outcome_col=outcome,
                cmp_cfg=cmp_cfg,
                fe_cols=fe_cols,
                cluster_col=cluster_col,
            )
        )
    return rows


def _stacked_sun_abraham_terms(
    work: pd.DataFrame,
    ref_event_time: int,
) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    if "g" not in work.columns or "rel_time" not in work.columns or "treated" not in work.columns:
        return {}, {}
    g_num = pd.to_numeric(work["g"], errors="coerce")
    rel_num = pd.to_numeric(work["rel_time"], errors="coerce")
    treated = pd.to_numeric(work["treated"], errors="coerce").fillna(0).eq(1)
    cohorts = sorted(int(v) for v in g_num.loc[treated & g_num.notna()].unique())
    event_times = sorted(int(v) for v in rel_num.loc[rel_num.notna()].unique() if int(v) != int(ref_event_time))
    term_to_time: dict[str, int] = {}
    term_meta: dict[str, tuple[int, int]] = {}
    for cohort in cohorts:
        cohort_mask = treated & g_num.eq(cohort)
        if not bool(cohort_mask.any()):
            continue
        for rel_time in event_times:
            col = f"sa_g{cohort}_rt{_lag_suffix(rel_time)}"
            work[col] = (cohort_mask & rel_num.eq(rel_time)).astype(float)
            if work[col].nunique(dropna=True) <= 1:
                continue
            term_to_time[col] = int(rel_time)
            term_meta[col] = (int(cohort), int(rel_time))
    return term_to_time, term_meta


def _estimate_stacked_sun_abraham_family(
    work: pd.DataFrame,
    *,
    lhs: str,
    term_to_time: dict[str, int],
    term_meta: dict[str, tuple[int, int]],
    event_times: Sequence[int],
    ref_event_time: int,
    family: str,
    spec: str,
    estimator: str,
    outcome_col: str,
    cmp_cfg: dict,
    fe_cols: list[str],
    cluster_col: str,
) -> list[dict[str, object]]:
    term_cols = list(term_to_time.keys())
    required = _dedupe_existing([lhs, *term_cols, *fe_cols, cluster_col, "g", "rel_time", "treated"], work)
    panel = work.loc[:, required].dropna(subset=required).copy()
    if lhs in panel.columns:
        panel[lhs] = pd.to_numeric(panel[lhs], errors="coerce")
        panel = panel.loc[panel[lhs].notna()].copy()
    for col in term_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel, prefilter_note = _prefilter_ppml_first_stage_panel(
        panel,
        lhs=lhs,
        estimator=estimator,
        family=family,
        fe_cols=fe_cols,
        cmp_cfg=cmp_cfg,
        context=f"stacked_did/{spec}/{family}/{outcome_col}",
    )
    non_ref_times = [int(v) for v in event_times if int(v) != int(ref_event_time)]
    if panel.empty or lhs not in panel.columns or panel[lhs].nunique(dropna=True) <= 1:
        rows = [
            _empty_result(family, "stacked_did", spec, estimator, "rel_time", v, outcome_col, "insufficient variation")
            for v in non_ref_times
        ]
        rows.append(
            _reference_result(
                family=family,
                design="stacked_did",
                spec=spec,
                estimator=estimator,
                time_name="rel_time",
                time_value=ref_event_time,
                outcome_col=outcome_col,
                extra={"fe_cols": "+".join(fe_cols), "sa_aggregation": "cohort_weighted"},
            )
        )
        return rows
    if _verbose(cmp_cfg):
        _log(
            f"stacked_did/{spec}: {family}/{outcome_col} Sun-Abraham starting | "
            f"estimator={estimator} | rows={len(panel):,} | units={_n_units_text(panel, cmp_cfg)} | "
            f"cohort-time terms={len(term_cols)} | FE={'+'.join(fe_cols)}",
            indent=3,
        )
    started = time.perf_counter()
    try:
        if (
            estimator == "ppml"
            and family == "first_stage"
            and _can_use_event_conditional_ppml(fe_cols, cmp_cfg, cluster_col)
        ):
            fit = _fit_event_conditional_ppml(panel, lhs, term_cols, term_to_time, cmp_cfg, cluster_col, fe_cols=fe_cols)
        else:
            fit = _fit_model(panel, lhs, term_cols, estimator, fe_cols, cluster_col)
    except Exception as exc:
        if _verbose(cmp_cfg):
            _log(f"regression failed | stacked_did/{spec}/{family}/{outcome_col}: {exc}", indent=3)
        rows = [
            _empty_result(family, "stacked_did", spec, estimator, "rel_time", v, outcome_col, str(exc))
            for v in non_ref_times
        ]
        rows.append(
            _reference_result(
                family=family,
                design="stacked_did",
                spec=spec,
                estimator=estimator,
                time_name="rel_time",
                time_value=ref_event_time,
                outcome_col=outcome_col,
                extra={"fe_cols": "+".join(fe_cols), "sa_aggregation": "cohort_weighted"},
            )
        )
        return rows
    if _verbose(cmp_cfg):
        _log(
            f"stacked_did/{spec}: {family}/{outcome_col} Sun-Abraham finished in "
            f"{_fmt_seconds(time.perf_counter() - started)}",
            indent=3,
        )
    params, cov = _fit_params_cov(fit)
    weights = _stacked_sun_abraham_weights(panel, term_meta)
    fit_nobs = int(getattr(fit, "nobs", len(panel)) or len(panel))
    n_units = panel[_unit_id_col(cmp_cfg)].nunique() if _unit_id_col(cmp_cfg) in panel.columns else None
    rows: list[dict[str, object]] = []
    for rel_time in non_ref_times:
        cols = [col for col, (_, event_t) in term_meta.items() if int(event_t) == int(rel_time) and col in params.index]
        coef: Optional[float]
        se: Optional[float]
        if cols:
            raw_weights = np.array([weights.get(col, 0.0) for col in cols], dtype=float)
            if raw_weights.sum() <= 0:
                raw_weights = np.ones(len(cols), dtype=float)
            norm_weights = raw_weights / raw_weights.sum()
            beta = params.reindex(cols).astype(float).to_numpy()
            coef = float(np.dot(norm_weights, beta))
            subcov = cov.reindex(index=cols, columns=cols).fillna(0.0).to_numpy(dtype=float)
            var = float(norm_weights @ subcov @ norm_weights)
            se = math.sqrt(var) if var >= 0 and np.isfinite(var) else float("nan")
        else:
            coef, se = None, None
        row = _result_base(family, "stacked_did", spec, estimator, "rel_time", int(rel_time), outcome_col, fit_nobs, n_units)
        row.update(
            {
                "coef": coef,
                "se": se,
                "f_stat": _single_term_f(coef, se),
                "fe_cols": "+".join(fe_cols),
                "sa_aggregation": "cohort_weighted",
                "n_treatment_cohorts": len(cols),
            }
        )
        if prefilter_note:
            row["ppml_prefilter"] = prefilter_note
        rows.append(row)
    rows.append(
        _reference_result(
            family=family,
            design="stacked_did",
            spec=spec,
            estimator=estimator,
            time_name="rel_time",
            time_value=ref_event_time,
            outcome_col=outcome_col,
            extra={
                "fe_cols": "+".join(fe_cols),
                "sa_aggregation": "cohort_weighted",
                "n_treatment_cohorts": int(pd.to_numeric(panel.loc[pd.to_numeric(panel["treated"], errors="coerce").eq(1), "g"], errors="coerce").nunique()),
            },
        )
    )
    return rows


def _stacked_sun_abraham_weights(panel: pd.DataFrame, term_meta: dict[str, tuple[int, int]]) -> dict[str, float]:
    if not term_meta or not {"treated", "g", "rel_time"}.issubset(panel.columns):
        return {col: 1.0 for col in term_meta}
    treated = pd.to_numeric(panel["treated"], errors="coerce").fillna(0).eq(1)
    cells = (
        panel.loc[treated, ["g", "rel_time"]]
        .assign(g=lambda x: pd.to_numeric(x["g"], errors="coerce"), rel_time=lambda x: pd.to_numeric(x["rel_time"], errors="coerce"))
        .dropna(subset=["g", "rel_time"])
    )
    if cells.empty:
        return {col: 1.0 for col in term_meta}
    counts = cells.groupby(["g", "rel_time"]).size()
    return {
        col: float(counts.get((float(cohort), float(rel_time)), 0.0))
        for col, (cohort, rel_time) in term_meta.items()
    }


def _stacked_fe_cols(work: pd.DataFrame, cmp_cfg: dict, estimator_label: str) -> list[str]:
    if estimator_label == "sun_abraham":
        # Match the pooled-control Sun-Abraham style used in the labor-lunch
        # relabels build: entity FE plus pooled calendar-year FE, not year x
        # stack FE. Matching already handles baseline size/growth, so stacked
        # regressions intentionally ignore the global size-year FE toggle.
        return [_unit_id_col(cmp_cfg), "t"]
    cols = [_unit_id_col(cmp_cfg), "t"]
    if bool(cmp_cfg.get("stacked_twfe_include_size_year_fe", False)) and _use_size_year_fe(cmp_cfg) and "baseline_size_growth_year_fe" in work.columns:
        cols.append("baseline_size_growth_year_fe")
    return cols


def _stacked_cluster_col(estimator_label: str, cmp_cfg: dict) -> str:
    if estimator_label == "sun_abraham":
        return _unit_id_col(cmp_cfg)
    return "stack_id"


def detect_largest_jump_events(
    panel: pd.DataFrame,
    *,
    exposure_col: str,
    cohort_min_year: int,
    cohort_max_year: int,
    min_jump: float,
    jump_quantile: Optional[float] = None,
    treated_jump_percentile: Optional[float] = None,
    control_min_jump_percentile: Optional[float] = None,
) -> pd.DataFrame:
    unit_col = "c"
    work = panel[[unit_col, "t", exposure_col]].dropna(subset=[unit_col, "t"]).copy()
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work = work.dropna(subset=["t"])
    if work.empty:
        return pd.DataFrame(columns=["c", "g", "largest_jump", "min_jump", "treated", "control_eligible"])
    work["t"] = work["t"].astype(int)
    work[exposure_col] = pd.to_numeric(work[exposure_col], errors="coerce")
    work = work.sort_values([unit_col, "t"])

    panel_years = set(int(v) for v in work["t"].dropna().unique())
    needed_years = [
        year
        for year in range(int(cohort_min_year) - 1, int(cohort_max_year) + 1)
        if year in panel_years
    ]
    if not needed_years:
        return pd.DataFrame(columns=["c", "g", "largest_jump", "min_jump", "treated", "control_eligible"])
    required_n_years = len(needed_years)
    in_needed_years = work["t"].isin(needed_years)
    window_year_counts = (
        work.loc[in_needed_years & work[exposure_col].notna(), [unit_col, "t"]]
        .drop_duplicates([unit_col, "t"])
        .groupby(unit_col)["t"]
        .nunique()
    )
    valid_units = set(window_year_counts.loc[window_year_counts.eq(required_n_years)].index)
    work = work.loc[work[unit_col].isin(valid_units)].copy()
    if work.empty:
        return pd.DataFrame(columns=["c", "g", "largest_jump", "min_jump", "treated", "control_eligible"])

    work["jump"] = work[exposure_col] - work.groupby(unit_col, sort=False)[exposure_col].shift(1)

    in_window = work["t"].between(int(cohort_min_year), int(cohort_max_year))
    window_jumps = work.loc[in_window & work["jump"].notna(), [unit_col, "t", "jump"]].copy()
    if window_jumps.empty:
        return pd.DataFrame(columns=["c", "g", "largest_jump", "min_jump", "treated", "control_eligible"])
    jump_summary = (
        window_jumps.groupby(unit_col, as_index=False)
        .agg(largest_jump=("jump", "max"), min_jump=("jump", "min"))
    )

    threshold = float(min_jump)
    if treated_jump_percentile is not None:
        q = _percentile_to_quantile(treated_jump_percentile)
        q_threshold = float(jump_summary["largest_jump"].quantile(q))
        threshold = max(threshold, q_threshold)
    elif jump_quantile is not None:
        q = _percentile_to_quantile(jump_quantile)
        q_threshold = float(jump_summary["largest_jump"].quantile(q))
        threshold = max(threshold, q_threshold)
    control_threshold = np.nan
    if control_min_jump_percentile is not None:
        control_threshold = float(jump_summary["min_jump"].quantile(_percentile_to_quantile(control_min_jump_percentile)))

    candidates = window_jumps.loc[window_jumps["jump"].ge(threshold)].copy()
    events = (
        candidates.sort_values([unit_col, "jump", "t"], ascending=[True, False, True])
        .drop_duplicates(unit_col, keep="first")
        .rename(columns={"t": "g"})
    )
    out = jump_summary.merge(events[[unit_col, "g"]], on=unit_col, how="left")
    out["treated"] = out["g"].notna().astype(int)
    if control_min_jump_percentile is None or not np.isfinite(control_threshold):
        out["control_eligible"] = out["treated"].eq(0)
    else:
        out["control_eligible"] = out["treated"].eq(0) & out["min_jump"].le(control_threshold)
    out["treated_jump_threshold"] = threshold
    out["control_min_jump_threshold"] = control_threshold
    out["event_jump_threshold"] = threshold
    return out[
        [
            "c",
            "g",
            "largest_jump",
            "min_jump",
            "treated",
            "control_eligible",
            "treated_jump_threshold",
            "control_min_jump_threshold",
            "event_jump_threshold",
        ]
    ]


def build_comparison_stacked_panel(
    panel: pd.DataFrame,
    event_df: pd.DataFrame,
    cmp_cfg: dict,
    *,
    matching_style: str,
    exposure_col: Optional[str] = None,
) -> pd.DataFrame:
    started = time.perf_counter()
    pre_window = int(cmp_cfg.get("stacked_pre_years", 3))
    post_window = int(cmp_cfg.get("stacked_post_years", 3))
    treated = event_df.loc[event_df["treated"].eq(1)].dropna(subset=["g"]).copy()
    if "control_eligible" in event_df.columns:
        controls = event_df.loc[event_df["treated"].eq(0) & event_df["control_eligible"].fillna(False), ["c"]].copy()
    else:
        controls = event_df.loc[event_df["treated"].eq(0), ["c"]].copy()
    if treated.empty or controls.empty:
        return pd.DataFrame()
    if _verbose(cmp_cfg):
        _log(
            f"stacked inputs: treated={len(treated):,} | controls={len(controls):,} | "
            f"panel={_panel_summary_text(panel)}",
            indent=3,
        )
    pair_started = time.perf_counter()
    pair_df = _build_pair_frame(panel, treated, controls, matching_style=matching_style, cmp_cfg=cmp_cfg)
    if pair_df.empty:
        return pd.DataFrame()
    if _verbose(cmp_cfg):
        _log(f"built pair frame: {len(pair_df):,} pairs in {_fmt_seconds(time.perf_counter() - pair_started)}", indent=3)
    stack_started = time.perf_counter()
    stacked = _build_stacked_panel_fast(
        panel,
        pair_df,
        treated[["c", "g"]].copy(),
        cmp_cfg=cmp_cfg,
        pre_window=pre_window,
        post_window=post_window,
        exposure_col=exposure_col,
    )
    if _verbose(cmp_cfg):
        _log(
            f"materialized stacked panel fast: {_panel_summary_text(stacked)} "
            f"in {_fmt_seconds(time.perf_counter() - stack_started)} | total {_fmt_seconds(time.perf_counter() - started)}",
            indent=3,
        )
    return stacked


def filter_analysis_sample(panel: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    work = _standardize_panel_ids(panel)
    if work.empty:
        return work
    t = pd.to_numeric(work["t"], errors="coerce")
    work = work.loc[t.between(int(cmp_cfg.get("data_min_t", 2010)), int(cmp_cfg.get("data_max_t", 2022)))].copy()
    if _exclude_outside_negative_analysis(cmp_cfg):
        work = _drop_outside_negative_firms(work)
    size_col = _first_present(work, ["firm_size_annual_pre_level", "headcount_size_baseline", "total_headcount_annual_pre_level"])
    if size_col is not None:
        work[size_col] = pd.to_numeric(work[size_col], errors="coerce")
        work = work.loc[work[size_col].ge(float(cmp_cfg.get("min_pre_avg_employment", 10)))].copy()
    return work.reset_index(drop=True)


def _drop_outside_negative_firms(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or "outside_negative_candidate" not in panel.columns or "c" not in panel.columns:
        return panel
    flags = pd.to_numeric(panel["outside_negative_candidate"], errors="coerce").fillna(0)
    outside_ids = set(panel.loc[flags.eq(1), "c"].astype(str))
    if not outside_ids:
        return panel
    return panel.loc[~panel["c"].astype(str).isin(outside_ids)].copy()


def _filter_opt_index_analysis_sample(panel: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    """Mirror exposure_event_study's OPT-index analysis sample, not model-training sample."""
    work = panel.copy()
    before = len(work)
    if "event_study_sample" in work.columns:
        sample = pd.to_numeric(work["event_study_sample"], errors="coerce").fillna(0)
        if str(cmp_cfg.get("unit", "firm")) == "local_market":
            work = work.loc[sample.gt(0)].copy()
        else:
            work = work.loc[sample.eq(1)].copy()
    if _exclude_outside_negative_analysis(cmp_cfg):
        work = _drop_outside_negative_firms(work)
    if _verbose(cmp_cfg):
        _log(
            f"OPT-index analysis filter: {before - len(work):,} rows removed; "
            f"{len(work):,} rows remain",
            indent=2,
        )
    return work


def _exclude_outside_negative_analysis(cmp_cfg: dict) -> bool:
    explicit = cmp_cfg.get("exclude_outside_negative_firms", None)
    if explicit is not None:
        return bool(explicit)
    try:
        cfg_path = _path_or_default(cmp_cfg.get("source_config_path"), DEFAULT_SOURCE_CONFIG_PATH)
        source_cfg = load_config(cfg_path)
        exp_cfg = get_cfg_section(source_cfg, "exposure_event_study")
        return bool(exp_cfg.get("event_study_exclude_outside_negatives", False))
    except Exception:
        return False


def aggregate_to_local_market(panel: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    local_col = str(cmp_cfg.get("local_market_col", "company_metro_feature"))
    if local_col not in panel.columns:
        raise ValueError(f"Local-market column {local_col!r} is missing from prepared panel.")
    work = panel.copy()
    work[local_col] = work[local_col].astype("string").fillna("__MISSING__")
    work["c"] = work[local_col].astype(str)
    group_cols = ["c", "t"]
    numeric_cols = [c for c in work.columns if c not in group_cols and pd.api.types.is_numeric_dtype(work[c])]
    sum_like = [c for c in numeric_cols if _is_sum_like_col(c)]
    mean_like = [c for c in numeric_cols if c not in sum_like]
    agg = {c: "sum" for c in sum_like}
    agg.update({c: "mean" for c in mean_like})
    out = work.groupby(group_cols, dropna=False, as_index=False).agg(agg)
    out[local_col] = out["c"]
    return out


def ensure_shift_share_share_variants(panel: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    """Derive share-based shift-share instruments when old caches lack them."""
    if panel.empty or not {"c", "t"}.issubset(panel.columns):
        return panel.copy()
    requested = set(_configured_shift_share_variant_cols(cmp_cfg))
    if "international_share" in set(_stacked_exposures_to_run(cmp_cfg)):
        requested.add("z_ct_international_share")
    needed_ihmp = [col for col in ("z_ct_ihmp_share", "z_ct_ihmp_share_ar_resid") if col in requested and col not in panel.columns]
    needed_intl = "z_ct_international_share" in requested and "z_ct_international_share" not in panel.columns
    if not needed_ihmp and not needed_intl:
        return panel.copy()
    out = panel.copy()
    derived_frames: list[pd.DataFrame] = []
    try:
        shift_cfg = load_config(DEFAULT_SHIFT_SHARE_CONFIG_PATH)
        paths = get_cfg_section(shift_cfg, "paths")
        transition_path = Path(str(paths.get("transition_shares", "")))
        growth_path = Path(str(paths.get("ipeds_unit_growth", "")))
        school_metric_path = Path(str(paths.get("school_shift_metric_panel", "")))
    except Exception as exc:
        if _verbose(cmp_cfg):
            _log(f"could not resolve share-based shift-share input paths: {exc}", indent=1)
        return panel.copy()

    if needed_ihmp:
        try:
            if transition_path.exists() and growth_path.exists():
                growth_cols = pd.read_parquet(growth_path).columns
                growth = pd.read_parquet(
                    growth_path,
                    columns=[col for col in ["k", "t", "metric_share"] if col in growth_cols],
                )
                if {"k", "t", "metric_share"}.issubset(growth.columns):
                    growth = growth[["k", "t", "metric_share"]].dropna(subset=["k", "t"]).copy()
                    growth["k"] = pd.to_numeric(growth["k"], errors="coerce").astype("Int64")
                    growth["t"] = pd.to_numeric(growth["t"], errors="coerce").astype("Int64")
                    growth["g_kt_ihmp_share"] = pd.to_numeric(growth["metric_share"], errors="coerce").fillna(0.0)
                    growth = growth.sort_values(["k", "t"]).copy()
                    growth["_metric_share_lag1"] = growth.groupby("k", sort=False)["g_kt_ihmp_share"].shift(1)
                    growth["g_kt_ihmp_share_ar_resid"] = _ar_residual_school_metric(
                        growth,
                        value_col="g_kt_ihmp_share",
                        lag_col="_metric_share_lag1",
                    )
                    import duckdb as ddb

                    con = ddb.connect(database=":memory:")
                    con.register("growth", growth[["k", "t", "g_kt_ihmp_share", "g_kt_ihmp_share_ar_resid"]])
                    derived_frames.append(
                        con.sql(
                            f"""
                            SELECT
                                CAST(s.c AS BIGINT) AS c,
                                CAST(g.t AS BIGINT) AS t,
                                SUM(CASE WHEN s.share_ck > 1 THEN NULL ELSE s.share_ck END * g.g_kt_ihmp_share) AS z_ct_ihmp_share,
                                SUM(CASE WHEN s.share_ck > 1 THEN NULL ELSE s.share_ck END * g.g_kt_ihmp_share_ar_resid) AS z_ct_ihmp_share_ar_resid,
                                COUNT(DISTINCT CASE
                                    WHEN s.share_ck IS NOT NULL AND s.share_ck > 0 AND g.g_kt_ihmp_share != 0 THEN s.k
                                    END
                                ) AS n_universities_ihmp_share,
                                COUNT(DISTINCT CASE
                                    WHEN s.share_ck IS NOT NULL AND s.share_ck > 0 AND g.g_kt_ihmp_share_ar_resid != 0 THEN s.k
                                    END
                                ) AS n_universities_ihmp_share_ar_resid
                            FROM read_parquet('{_sql_path(transition_path)}') s
                            JOIN growth g
                              ON CAST(s.k AS BIGINT) = CAST(g.k AS BIGINT)
                            GROUP BY 1, 2
                            """
                        ).df()
                    )
                    con.close()
        except Exception as exc:
            if _verbose(cmp_cfg):
                _log(f"could not derive IHMP-share shift-share variants: {exc}", indent=1)

    if needed_intl:
        try:
            if transition_path.exists() and school_metric_path.exists():
                import duckdb as ddb

                con = ddb.connect(database=":memory:")
                derived_frames.append(
                    con.sql(
                        f"""
                        WITH intl_metric AS (
                            SELECT
                                CAST(k AS VARCHAR) AS k,
                                CAST(t AS BIGINT) AS t,
                                CASE
                                    WHEN TRY_CAST(ipeds_total_students AS DOUBLE) > 0 THEN
                                        TRY_CAST(ipeds_total_intl_students AS DOUBLE)
                                        / TRY_CAST(ipeds_total_students AS DOUBLE)
                                    ELSE NULL
                                END AS international_share
                            FROM read_parquet('{_sql_path(school_metric_path)}')
                        )
                        SELECT
                            CAST(s.c AS BIGINT) AS c,
                            CAST(m.t AS BIGINT) AS t,
                            SUM(
                                CASE WHEN s.share_ck > 1 THEN NULL ELSE s.share_ck END
                                * COALESCE(m.international_share, 0.0)
                            ) AS z_ct_international_share,
                            COUNT(DISTINCT CASE
                                WHEN s.share_ck IS NOT NULL AND s.share_ck > 0
                                 AND COALESCE(m.international_share, 0.0) != 0 THEN s.k
                                END
                            ) AS n_universities_international_share
                        FROM read_parquet('{_sql_path(transition_path)}') s
                        JOIN intl_metric m
                          ON CAST(s.k AS VARCHAR) = m.k
                        GROUP BY 1, 2
                        """
                    ).df()
                )
                con.close()
        except Exception as exc:
            if _verbose(cmp_cfg):
                _log(f"could not derive international-share shift-share variant: {exc}", indent=1)

    for derived in derived_frames:
        if derived.empty:
            continue
        out = out.drop(columns=[col for col in derived.columns if col not in {"c", "t"} and col in out.columns], errors="ignore")
        out = out.merge(derived, on=["c", "t"], how="left")
    if _verbose(cmp_cfg):
        present = [col for col in [*needed_ihmp, "z_ct_international_share"] if col in out.columns]
        if present:
            _log(f"derived missing share-based shift-share variants: {', '.join(present)}", indent=1)
    return out


def attach_company_features(panel: pd.DataFrame, features: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    del cmp_cfg
    if panel.empty or features.empty or "c" not in panel.columns or "c" not in features.columns:
        return panel.copy()
    feature_cols = [
        col for col in [
            "c",
            "naics2",
            "company_state_feature",
            "company_metro_feature",
            "company_hq_region",
            "firm_size_annual_pre_level",
            "firm_size_annual_pre_growth",
            "total_headcount_annual_pre_level",
            "school_opt_share_new_hire_annual_pre_level",
            "school_opt_share_tenured_annual_pre_level",
            "company_n_users_log1p",
            "preferred_rcid_source",
            "outside_negative_candidate",
            "in_analysis_universe",
        ]
        if col in features.columns
    ]
    right = features[feature_cols].drop_duplicates("c")
    overlap = [c for c in feature_cols if c != "c" and c in panel.columns]
    base = panel.drop(columns=overlap, errors="ignore")
    return base.merge(right, on="c", how="left")


def attach_opt_probability_index(panel: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or pred_df.empty or "c" not in pred_df.columns:
        return panel.copy()
    candidates = [
        "c",
        "predicted_prob",
        "predicted_index",
        "predicted_class",
        "opt_probability_index",
        "ntile",
        "ntile_label",
        "event_study_sample",
        "preferred_rcid_source",
        "outside_negative_candidate",
        "target_source",
    ]
    cols = [c for c in candidates if c in pred_df.columns]
    right = pred_df[cols].drop_duplicates("c")
    if "predicted_prob" not in right.columns and "opt_probability_index" in right.columns:
        right = right.rename(columns={"opt_probability_index": "predicted_prob"})
    return panel.drop(columns=[c for c in right.columns if c != "c" and c in panel.columns], errors="ignore").merge(right, on="c", how="left")


def attach_shift_share_exposures(panel: pd.DataFrame, shift_panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or shift_panel.empty or not {"c", "t"}.issubset(panel.columns) or not {"c", "t"}.issubset(shift_panel.columns):
        return panel.copy()
    exposure_cols = [
        col for col in shift_panel.columns
        if col.startswith("z_ct") or col.startswith("n_universities")
    ]
    if not exposure_cols:
        return panel.copy()
    right = shift_panel[["c", "t", *exposure_cols]].drop_duplicates(["c", "t"])
    overlap = [c for c in exposure_cols if c in panel.columns]
    return panel.drop(columns=overlap, errors="ignore").merge(right, on=["c", "t"], how="left")


def _needs_workforce_design_outcomes(panel: pd.DataFrame, cmp_cfg: dict) -> bool:
    requested = set(_list_cfg(cmp_cfg.get("outcome_cols", DEFAULT_OUTCOMES)))
    return "avg_tenure_years_lag0" in requested and "avg_tenure_years_lag0" not in panel.columns


def attach_workforce_design_outcomes(panel: pd.DataFrame, workforce: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    del cmp_cfg
    if panel.empty or workforce.empty or not {"c", "t"}.issubset(panel.columns) or not {"c", "t"}.issubset(workforce.columns):
        return panel.copy()
    right_cols = ["c", "t"]
    rename: dict[str, str] = {}
    if "avg_tenure_years_annual" in workforce.columns:
        right_cols.append("avg_tenure_years_annual")
        rename["avg_tenure_years_annual"] = "avg_tenure_years_lag0"
    if not rename:
        return panel.copy()
    right = workforce[right_cols].rename(columns=rename).drop_duplicates(["c", "t"])
    return panel.drop(columns=[c for c in right.columns if c not in {"c", "t"} and c in panel.columns], errors="ignore").merge(right, on=["c", "t"], how="left")


def _needs_design3_position_outcomes(panel: pd.DataFrame, cmp_cfg: dict) -> bool:
    needed = set(_design3_position_outcome_cols_for_config(cmp_cfg))
    override_requested = bool(needed.intersection(DESIGN3_NEW_GRAD_OVERRIDE_OUTCOMES))
    return bool(needed) and (override_requested or any(col not in panel.columns for col in needed))


def attach_design3_position_outcomes(panel: pd.DataFrame, outcomes: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    if panel.empty or outcomes.empty or not {"c", "t"}.issubset(panel.columns) or not {"c", "t"}.issubset(outcomes.columns):
        return panel.copy()
    value_cols = [col for col in _design3_position_outcome_cols_for_config(cmp_cfg) if col in outcomes.columns]
    if not value_cols:
        return panel.copy()
    right = outcomes[["c", "t", *value_cols]].drop_duplicates(["c", "t"])
    out = panel.drop(columns=[col for col in value_cols if col in panel.columns], errors="ignore").merge(right, on=["c", "t"], how="left")
    for col in value_cols:
        if _is_count_outcome(col):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _design3_position_outcome_cols_for_config(cmp_cfg: dict) -> list[str]:
    requested = set(_list_cfg(cmp_cfg.get("outcome_cols", DEFAULT_OUTCOMES)))
    candidates = list(DESIGN3_POSITION_OUTCOMES)
    if not bool(cmp_cfg.get("design3_replace_new_hires_with_new_grad", False)):
        candidates = [col for col in candidates if col not in DESIGN3_NEW_GRAD_OVERRIDE_OUTCOMES]
    return [col for col in candidates if col in requested]


def ensure_design_outcome_derivations(panel: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    requested = set(_list_cfg(cmp_cfg.get("outcome_cols", DEFAULT_OUTCOMES)))
    if "y_new_hires_foreign_minus_one_lag0" not in requested or "y_new_hires_foreign_lag0" not in panel.columns:
        return panel
    out = panel.copy()
    out["y_new_hires_foreign_minus_one_lag0"] = (
        pd.to_numeric(out["y_new_hires_foreign_lag0"], errors="coerce") - 1.0
    )
    return out


def recompute_baseline_size_growth(panel: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    """Overwrite baseline size/growth using year-level data in the configured pre-period."""
    if panel.empty or not {"c", "t"}.issubset(panel.columns):
        return panel.copy()
    size_col = _first_present(
        panel,
        [
            "total_headcount_wrds_annual",
            "y_cst_lag0",
            "headcount_lag0_raw",
            "total_headcount_annual",
            "firm_size_annual",
        ],
    )
    if size_col is None:
        return panel.copy()
    start = int(cmp_cfg.get("baseline_start", 2010))
    end = int(cmp_cfg.get("baseline_end", 2013))
    work = panel.copy()
    t = pd.to_numeric(work["t"], errors="coerce")
    baseline = work.loc[t.between(start, end), ["c", "t", size_col]].copy()
    baseline["t"] = pd.to_numeric(baseline["t"], errors="coerce")
    baseline[size_col] = pd.to_numeric(baseline[size_col], errors="coerce")
    baseline = baseline.dropna(subset=["c", "t", size_col])
    if baseline.empty:
        return work
    summary = (
        baseline.groupby("c", as_index=False)
        .agg(
            firm_size_annual_pre_level=(size_col, "mean"),
            firm_size_annual_pre_n_years=(size_col, "count"),
        )
    )
    growth_rows = []
    for c, group in baseline.sort_values("t").groupby("c", sort=False):
        if group["t"].nunique() < 2:
            growth = np.nan
        else:
            x = pd.to_numeric(group["t"], errors="coerce").to_numpy(dtype=float)
            y = np.arcsinh(pd.to_numeric(group[size_col], errors="coerce").to_numpy(dtype=float))
            mask = np.isfinite(x) & np.isfinite(y)
            growth = float(np.polyfit(x[mask], y[mask], 1)[0]) if mask.sum() >= 2 else np.nan
        growth_rows.append({"c": c, "firm_size_annual_pre_growth": growth})
    summary = summary.merge(pd.DataFrame(growth_rows), on="c", how="left")
    work = work.drop(
        columns=[
            "firm_size_annual_pre_level",
            "firm_size_annual_pre_n_years",
            "firm_size_annual_pre_growth",
        ],
        errors="ignore",
    )
    return work.merge(summary, on="c", how="left")


def prepare_shift_share_state_panel(panel: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    """Use shift_share_analysis' own baseline-state helper for Design 1 FE cells."""
    if panel.empty or not {"c", "t"}.issubset(panel.columns):
        return panel.copy()
    work = panel.copy()
    work["t_num"] = pd.to_numeric(work["t"], errors="coerce")
    baseline_cfg = _shift_share_baseline_cfg(cmp_cfg)
    try:
        return ssa._prepare_first_stage_state_panel(
            work,
            baseline_window_start=int(baseline_cfg.get("baseline_start", 2008)),
            baseline_window_end=int(baseline_cfg.get("baseline_end", 2013)),
            current_size_bins=int(cmp_cfg.get("baseline_size_bins", 10)),
            current_growth_bins=int(cmp_cfg.get("baseline_growth_bins", 5)),
            joint_size_growth_bins=int(cmp_cfg.get("joint_size_growth_bins", 3)),
            baseline_growth_bins=int(cmp_cfg.get("baseline_growth_bins", 5)),
            use_log_y_panel=False,
        )
    except Exception as exc:
        if _verbose(cmp_cfg):
            _log(f"shift_share state-panel helper failed; falling back to local baseline recompute: {exc}", indent=1)
        return recompute_baseline_size_growth(work, baseline_cfg)


def _shift_share_baseline_cfg(cmp_cfg: dict) -> dict:
    out = dict(cmp_cfg)
    out["baseline_start"] = int(cmp_cfg.get("shift_share_fe_baseline_start", cmp_cfg.get("baseline_start", 2010)))
    out["baseline_end"] = int(cmp_cfg.get("shift_share_fe_baseline_end", cmp_cfg.get("baseline_end", 2013)))
    return out


def build_final_comparison_table(coef_df: pd.DataFrame) -> pd.DataFrame:
    if coef_df.empty:
        return pd.DataFrame()
    first = coef_df.loc[coef_df["family"].eq("first_stage")].copy()
    first = _pick_event_row(first)
    key_cols = ["design", "spec"]
    value_cols = ["coef", "se", "f_stat", "baseline_mean", "effect_size", "n_obs", "estimator"]
    available_value_cols = [col for col in value_cols if col in first.columns]
    table = first[key_cols + available_value_cols].rename(
        columns={
            "coef": "first_stage_coef",
            "se": "first_stage_se",
            "n_obs": "first_stage_n",
            "estimator": "first_stage_estimator",
        }
    )
    return table.sort_values(key_cols).reset_index(drop=True)


def add_baseline_effect_stats(coef_df: pd.DataFrame, prepared: dict[str, pd.DataFrame], cmp_cfg: dict) -> pd.DataFrame:
    if coef_df.empty:
        return coef_df
    out = coef_df.copy()
    if "outcome_col" in out.columns:
        out["outcome_label"] = out["outcome_col"].map(_pretty_outcome_label)
    baseline_cache: dict[tuple[str, str, str], float] = {}

    def _baseline_for(row: pd.Series) -> float:
        design = str(row.get("design"))
        family = str(row.get("family"))
        outcome = str(row.get("outcome_col"))
        key = (design, family, outcome)
        if key in baseline_cache:
            return baseline_cache[key]
        panel = prepared.get(design)
        value = float("nan")
        if isinstance(panel, pd.DataFrame) and outcome in panel.columns and "t" in panel.columns:
            work = panel.copy()
            t = pd.to_numeric(work["t"], errors="coerce")
            work = work.loc[t.between(int(cmp_cfg.get("baseline_start", 2010)), int(cmp_cfg.get("baseline_end", 2013)))].copy()
            values = pd.to_numeric(work[outcome], errors="coerce")
            if family == "first_stage":
                values = _first_stage_transform(values, cmp_cfg)
            elif family == "reduced_form":
                values = _transform_outcome(values, outcome, cmp_cfg)
            value = float(values.mean()) if values.notna().any() else float("nan")
        baseline_cache[key] = value
        return value

    out["baseline_mean"] = out.apply(_baseline_for, axis=1)
    coef = pd.to_numeric(out["coef"], errors="coerce")
    baseline = pd.to_numeric(out["baseline_mean"], errors="coerce")
    out["effect_size"] = np.where(baseline.abs().gt(1.0e-12), coef / baseline, np.nan)
    return out


def format_final_comparison_table_latex(table: pd.DataFrame) -> str:
    if table.empty:
        return "\\begin{tabular}{l}\\toprule\\nNo rows.\\\\\\n\\bottomrule\\n\\end{tabular}\\n"

    def _coef_se(row: pd.Series) -> str:
        coef = _fmt_num(row.get("first_stage_coef"), digits=3)
        se = _fmt_num(row.get("first_stage_se"), digits=3)
        return rf"\shortstack{{{coef}\\({se})}}"

    lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Design & Specification & First stage & Baseline mean & Effect / mean & F-stat \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{_latex_escape(_pretty_design_label(row.get('design')))} & "
            f"{_latex_escape(_pretty_spec_label(row.get('spec')))} & "
            f"{_coef_se(row)} & "
            f"{_fmt_num(row.get('baseline_mean'), digits=3)} & "
            f"{_fmt_num(row.get('effect_size'), digits=2)} & "
            f"{_fmt_num(row.get('f_stat'), digits=1, comma=True)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)


def build_prepared_panel_summary(prepared: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in prepared.items():
        if not isinstance(df, pd.DataFrame):
            continue
        rows.append(
            {
                "panel": name,
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "n_units": int(df["c"].nunique()) if "c" in df.columns else None,
                "min_t": int(pd.to_numeric(df["t"], errors="coerce").min()) if "t" in df.columns and not df.empty else None,
                "max_t": int(pd.to_numeric(df["t"], errors="coerce").max()) if "t" in df.columns and not df.empty else None,
            }
        )
    return pd.DataFrame(rows)


def _estimate_single_term(
    work: pd.DataFrame,
    *,
    lhs: str,
    term: str,
    family: str,
    design: str,
    spec: str,
    estimator: str,
    time_value: int,
    time_name: str,
    outcome_col: str,
    cmp_cfg: dict,
    fe_cols: Optional[list[str]] = None,
    cluster_col: Optional[str] = None,
) -> dict[str, object]:
    if lhs not in work.columns:
        return _empty_result(family, design, spec, estimator, time_name, time_value, outcome_col, "missing lhs")
    requested_fe = fe_cols or _fe_cols(work, cmp_cfg)
    panel = _regression_rows(work, [lhs, term, *requested_fe])
    panel, prefilter_note = _prefilter_ppml_first_stage_panel(
        panel,
        lhs=lhs,
        estimator=estimator,
        family=family,
        fe_cols=requested_fe,
        cmp_cfg=cmp_cfg,
        context=f"{design}/{spec}/{family}/{outcome_col} {time_name}={time_value}",
    )
    if panel.empty or panel[lhs].nunique(dropna=True) <= 1 or panel[term].nunique(dropna=True) <= 1:
        return _empty_result(family, design, spec, estimator, time_name, time_value, outcome_col, "insufficient variation")
    try:
        fit = _fit_model(panel, lhs, [term], estimator, requested_fe, cluster_col or _unit_id_col(cmp_cfg))
        actual_fe = requested_fe
        fallback_reason = None
    except Exception as exc:
        fallback_fe = _fallback_fe_cols(panel, requested_fe, cmp_cfg, estimator, family)
        if fallback_fe is not None:
            if _verbose(cmp_cfg):
                _log(
                    f"regression failed; retrying fallback FE | {design}/{spec}/{family}/{outcome_col} "
                    f"{time_name}={time_value}: {exc}",
                    indent=3,
                )
            try:
                fit = _fit_model(panel, lhs, [term], estimator, fallback_fe, cluster_col or _unit_id_col(cmp_cfg))
                actual_fe = fallback_fe
                fallback_reason = str(exc)
            except Exception as fallback_exc:
                if _verbose(cmp_cfg):
                    _log(
                        f"fallback regression failed | {design}/{spec}/{family}/{outcome_col} "
                        f"{time_name}={time_value}: {fallback_exc}",
                        indent=3,
                    )
                return _empty_result(
                    family,
                    design,
                    spec,
                    estimator,
                    time_name,
                    time_value,
                    outcome_col,
                    f"primary failed: {exc}; fallback failed: {fallback_exc}",
                )
        else:
            if _verbose(cmp_cfg):
                _log(
                    f"regression failed | {design}/{spec}/{family}/{outcome_col} "
                    f"{time_name}={time_value}: {exc}",
                    indent=3,
                )
            return _empty_result(family, design, spec, estimator, time_name, time_value, outcome_col, str(exc))
    coef, se = _coef_se(fit, term)
    f_stat = _single_term_f(coef, se)
    out = _result_base(family, design, spec, estimator, time_name, time_value, outcome_col, len(panel), panel[_unit_id_col(cmp_cfg)].nunique() if _unit_id_col(cmp_cfg) in panel.columns else None)
    out.update({"coef": coef, "se": se, "f_stat": f_stat})
    out["fe_cols"] = "+".join(actual_fe)
    out["used_fallback_fe"] = bool(fallback_reason)
    out["fallback_reason"] = fallback_reason
    if prefilter_note:
        out["ppml_prefilter"] = prefilter_note
    return out


def _estimate_multi_term_event(
    work: pd.DataFrame,
    *,
    lhs: str,
    terms: Sequence[tuple[int, str]],
    family: str,
    design: str,
    spec: str,
    estimator: str,
    outcome_col: str,
    cmp_cfg: dict,
    time_name: str,
    term_to_time: dict[str, int],
    fe_cols: Optional[list[str]] = None,
    cluster_col: Optional[str] = None,
) -> list[dict[str, object]]:
    del terms
    term_cols = list(term_to_time.keys())
    fe = fe_cols or _fe_cols(work, cmp_cfg)
    panel = _regression_rows(work, [lhs, *term_cols, *fe])
    return _estimate_multi_term_event_from_base(
        panel,
        lhs=lhs,
        term_cols=term_cols,
        family=family,
        design=design,
        spec=spec,
        estimator=estimator,
        outcome_col=outcome_col,
        cmp_cfg=cmp_cfg,
        time_name=time_name,
        term_to_time=term_to_time,
        fe_cols=fe,
        cluster_col=cluster_col,
    )


def _estimate_multi_term_event_from_base(
    base_panel: pd.DataFrame,
    *,
    lhs: str,
    term_cols: list[str],
    family: str,
    design: str,
    spec: str,
    estimator: str,
    outcome_col: str,
    cmp_cfg: dict,
    time_name: str,
    term_to_time: dict[str, int],
    fe_cols: list[str],
    cluster_col: Optional[str] = None,
    model_index: Optional[int] = None,
    model_total: Optional[int] = None,
) -> list[dict[str, object]]:
    required_cols = _dedupe_existing([lhs, *term_cols, *fe_cols, cluster_col or _unit_id_col(cmp_cfg)], base_panel)
    panel = base_panel.loc[:, required_cols].dropna(subset=required_cols).copy()
    if lhs in panel.columns:
        panel[lhs] = pd.to_numeric(panel[lhs], errors="coerce")
        panel = panel.loc[panel[lhs].notna()].copy()
    panel, prefilter_note = _prefilter_ppml_first_stage_panel(
        panel,
        lhs=lhs,
        estimator=estimator,
        family=family,
        fe_cols=fe_cols,
        cmp_cfg=cmp_cfg,
        context=f"{design}/{spec}/{family}/{outcome_col}",
    )
    if panel.empty or lhs not in panel.columns or panel[lhs].nunique(dropna=True) <= 1:
        return [
            _empty_result(family, design, spec, estimator, time_name, time_value, outcome_col, "insufficient variation")
            for time_value in term_to_time.values()
        ]
    progress = f" {model_index}/{model_total}" if model_index is not None and model_total is not None else ""
    if _verbose(cmp_cfg):
        _log(
            f"{design}/{spec}: model{progress} {family}/{outcome_col} starting | "
            f"estimator={estimator} | rows={len(panel):,} | units={_n_units_text(panel, cmp_cfg)} | "
            f"terms={len(term_cols)} | FE={'+'.join(fe_cols)}",
            indent=3,
        )
    started = time.perf_counter()
    try:
        if design == "event_study" and estimator == "ols":
            fit = _fit_event_residualized_ols(panel, lhs, term_cols, fe_cols, cmp_cfg, cluster_col or _unit_id_col(cmp_cfg))
        elif (
            design == "event_study"
            and estimator == "ppml"
            and family == "first_stage"
            and _can_use_event_conditional_ppml(fe_cols, cmp_cfg, cluster_col or _unit_id_col(cmp_cfg))
        ):
            fit = _fit_event_conditional_ppml(
                panel,
                lhs,
                term_cols,
                term_to_time,
                cmp_cfg,
                cluster_col or _unit_id_col(cmp_cfg),
                fe_cols=fe_cols,
            )
        else:
            fit = _fit_model(panel, lhs, term_cols, estimator, fe_cols, cluster_col or _unit_id_col(cmp_cfg))
    except Exception as exc:
        if _verbose(cmp_cfg):
            _log(
                f"regression failed | {design}/{spec}/{family}/{outcome_col}: {exc}",
                indent=3,
            )
        return [
            _empty_result(family, design, spec, estimator, time_name, time_value, outcome_col, str(exc))
            for time_value in term_to_time.values()
        ]
    if _verbose(cmp_cfg):
        _log(
            f"{design}/{spec}: model{progress} {family}/{outcome_col} finished in "
            f"{_fmt_seconds(time.perf_counter() - started)}",
            indent=3,
        )
    rows: list[dict[str, object]] = []
    fit_nobs = int(getattr(fit, "nobs", len(panel)) or len(panel))
    for term, time_value in term_to_time.items():
        coef, se = _coef_se(fit, term)
        row = _result_base(family, design, spec, estimator, time_name, int(time_value), outcome_col, fit_nobs, panel[_unit_id_col(cmp_cfg)].nunique() if _unit_id_col(cmp_cfg) in panel.columns else None)
        row.update(
            {
                "coef": coef,
                "se": se,
                "f_stat": _single_term_f(coef, se),
                "fe_cols": "+".join(_event_ols_absorb_cols(fe_cols, panel, cmp_cfg)) if design == "event_study" and estimator == "ols" else "+".join(fe_cols),
            }
        )
        if prefilter_note:
            row["ppml_prefilter"] = prefilter_note
        rows.append(row)
    return rows


def _fit_event_residualized_ols(
    panel: pd.DataFrame,
    lhs: str,
    term_cols: list[str],
    fe_cols: list[str],
    cmp_cfg: dict,
    cluster_col: str,
) -> _LinearFit:
    absorb_cols = _event_ols_absorb_cols(fe_cols, panel, cmp_cfg)
    required = _dedupe_existing([lhs, *term_cols, *absorb_cols, cluster_col], panel)
    work = panel.loc[:, required].dropna(subset=required).copy()
    work[lhs] = pd.to_numeric(work[lhs], errors="coerce")
    for col in term_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[lhs, *term_cols, *absorb_cols, cluster_col]).copy()
    if work.empty:
        raise ValueError("Residualized event-study OLS sample is empty after filtering.")

    y = work[lhs].to_numpy(dtype=float)
    x = work[term_cols].to_numpy(dtype=float)
    values = np.column_stack([y, x])
    group_codes = [_group_codes(work[col]) for col in absorb_cols]
    residualized = _residualize_matrix_by_groups(
        values,
        group_codes,
        tol=float(cmp_cfg.get("event_ols_residualize_tol", 1e-8)),
        max_iter=int(cmp_cfg.get("event_ols_residualize_max_iter", 200)),
    )
    y_res = residualized[:, 0]
    x_res = residualized[:, 1:]
    keep = np.nanstd(x_res, axis=0) > float(cmp_cfg.get("event_ols_min_residualized_sd", 1e-12))
    kept_terms = [col for col, ok in zip(term_cols, keep) if bool(ok)]
    if not kept_terms:
        raise ValueError("All event-study OLS terms were absorbed by fixed effects.")
    x_res = x_res[:, keep]

    xtx = x_res.T @ x_res
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x_res.T @ y_res)
    resid = y_res - x_res @ beta
    cluster_codes = _group_codes(work[cluster_col]) if cluster_col in work.columns else _group_codes(work[_unit_id_col(cmp_cfg)])
    meat = _cluster_meat(x_res, resid, cluster_codes)
    cov = xtx_inv @ meat @ xtx_inv
    nobs = int(len(work))
    n_params = int(len(kept_terms))
    n_clusters = int(cluster_codes.max() + 1) if len(cluster_codes) else 0
    if n_clusters > 1 and nobs > n_params:
        cov *= (n_clusters / (n_clusters - 1)) * ((nobs - 1) / (nobs - n_params))
    diag = np.diag(cov)
    ses = np.sqrt(np.where(diag >= 0, diag, np.nan))
    return _LinearFit(
        params=pd.Series(beta, index=kept_terms),
        std_errors=pd.Series(ses, index=kept_terms),
        covariance=pd.DataFrame(cov, index=kept_terms, columns=kept_terms),
        nobs=nobs,
    )


def _event_ols_absorb_cols(fe_cols: list[str], panel: pd.DataFrame, cmp_cfg: dict) -> list[str]:
    unit_col = _unit_id_col(cmp_cfg)
    baseline_col = "baseline_size_growth_year_fe"
    out = [unit_col] if unit_col in panel.columns else []
    if "t" in fe_cols and "t" in panel.columns:
        out.append("t")
    if baseline_col in fe_cols and baseline_col in panel.columns:
        out.append(baseline_col)
    for col in fe_cols:
        if col not in out and col not in {unit_col, "t", baseline_col} and col in panel.columns:
            out.append(col)
    return out


def _group_codes(values: pd.Series) -> np.ndarray:
    return pd.Categorical(values.astype(str), ordered=False).codes.astype(np.int64, copy=False)


def _residualize_matrix_by_groups(
    values: np.ndarray,
    group_codes: list[np.ndarray],
    *,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    if not group_codes:
        return out - out.mean(axis=0, keepdims=True)
    for codes in group_codes:
        if len(codes) != len(out) or np.any(codes < 0):
            raise ValueError("Invalid fixed-effect group codes for residualization.")
    last_norm = np.inf
    for _ in range(max_iter):
        for codes in group_codes:
            n_groups = int(codes.max()) + 1
            counts = np.bincount(codes, minlength=n_groups).astype(float)
            counts[counts == 0] = 1.0
            for j in range(out.shape[1]):
                sums = np.bincount(codes, weights=out[:, j], minlength=n_groups)
                out[:, j] -= sums[codes] / counts[codes]
        norm = 0.0
        for codes in group_codes:
            n_groups = int(codes.max()) + 1
            counts = np.bincount(codes, minlength=n_groups).astype(float)
            counts[counts == 0] = 1.0
            for j in range(out.shape[1]):
                means = np.bincount(codes, weights=out[:, j], minlength=n_groups) / counts
                norm = max(norm, float(np.nanmax(np.abs(means))))
        if abs(last_norm - norm) <= tol or norm <= tol:
            break
        last_norm = norm
    return out


def _cluster_meat(x: np.ndarray, residual: np.ndarray, cluster_codes: np.ndarray) -> np.ndarray:
    n_clusters = int(cluster_codes.max()) + 1 if len(cluster_codes) else 0
    if n_clusters <= 0:
        return np.zeros((x.shape[1], x.shape[1]))
    scores = np.empty((n_clusters, x.shape[1]), dtype=float)
    for j in range(x.shape[1]):
        scores[:, j] = np.bincount(cluster_codes, weights=x[:, j] * residual, minlength=n_clusters)
    return scores.T @ scores


def _fit_event_panelols(
    panel: pd.DataFrame,
    lhs: str,
    term_cols: list[str],
    term_to_time: dict[str, int],
    cmp_cfg: dict,
    cluster_col: str,
):
    if PanelOLS is None:
        raise ImportError("linearmodels is required to run event-study PanelOLS regressions.")
    unit_col = _unit_id_col(cmp_cfg)
    baseline_col = "baseline_size_growth_year_fe"
    use_size_year_fe = _use_size_year_fe(cmp_cfg) and baseline_col in panel.columns
    required = _dedupe_existing([lhs, unit_col, "t", *term_cols, baseline_col if use_size_year_fe else None, cluster_col], panel)
    work = panel.loc[:, required].dropna(subset=required).copy()
    work[lhs] = pd.to_numeric(work[lhs], errors="coerce")
    for col in term_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[lhs, *term_cols, unit_col, "t"])
    if work.empty:
        raise ValueError("PanelOLS event-study sample is empty after filtering.")

    t_num = pd.to_numeric(work["t"], errors="coerce")
    work = work.loc[t_num.notna()].copy()
    work["_t_panelols"] = t_num.loc[work.index].astype(int)
    years = sorted(work["_t_panelols"].unique())
    treatment_ref_years = [year for year in years if int(year) not in set(int(v) for v in term_to_time.values())]
    treatment_ref_year = treatment_ref_years[0] if treatment_ref_years else None
    calendar_ref_year = _calendar_year_dummy_reference(years, treatment_ref_year)

    year_dummies = pd.get_dummies(work["_t_panelols"], prefix="yr", dtype=float)
    ref_year_col = f"yr_{calendar_ref_year}"
    if ref_year_col in year_dummies.columns:
        year_dummies = year_dummies.drop(columns=[ref_year_col])

    exog = pd.concat([work[term_cols].astype(float).reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1)
    if not exog.empty:
        zero_var = exog.columns[exog.std(axis=0) == 0].tolist()
        if zero_var:
            exog = exog.drop(columns=zero_var)
    idx = pd.MultiIndex.from_arrays(
        [work[unit_col].astype(str).to_numpy(), work["_t_panelols"].to_numpy()],
        names=[unit_col, "t"],
    )
    dep = pd.Series(work[lhs].to_numpy(dtype=float), index=idx, name=lhs)
    exog_indexed = exog.set_index(idx)
    other_effects = None
    if use_size_year_fe and baseline_col in work.columns:
        other_effects = pd.DataFrame(
            {baseline_col: work[baseline_col].astype("category").cat.codes.to_numpy()},
            index=idx,
        )
    model = PanelOLS(
        dependent=dep,
        exog=exog_indexed,
        entity_effects=True,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=False,
    )
    return model.fit(cov_type="clustered", cluster_entity=True, low_memory=True)


def _can_use_event_conditional_ppml(fe_cols: list[str], cmp_cfg: dict, cluster_col: str) -> bool:
    if minimize is None:
        return False
    unit_col = _unit_id_col(cmp_cfg)
    if cluster_col != unit_col or unit_col not in fe_cols:
        return False
    allowed = {unit_col, "t", "baseline_size_growth_year_fe"}
    return set(fe_cols).issubset(allowed)


def _fit_event_conditional_ppml(
    panel: pd.DataFrame,
    lhs: str,
    term_cols: list[str],
    term_to_time: dict[str, int],
    cmp_cfg: dict,
    cluster_col: str,
    *,
    fe_cols: Optional[list[str]] = None,
) -> _ConditionalPPMLFit:
    del cluster_col
    if minimize is None:
        raise ImportError("scipy is required to run the fast event-study conditional PPML estimator.")
    design = _build_event_conditional_ppml_design(panel, lhs, term_cols, term_to_time, cmp_cfg, fe_cols=fe_cols)
    if _verbose(cmp_cfg):
        _log(
            "fast conditional PPML first-stage sample | "
            f"kept rows={design.kept_rows:,}; "
            f"dropped all-zero outcome rows={design.dropped_zero_outcome_rows:,}; "
            f"dropped all-zero units={design.dropped_zero_outcome_units:,}; "
            f"parameters={len(design.param_names):,}",
            indent=4,
        )
    objective = _ConditionalPPMLObjective(design=design, sum_yx=design.x.T @ design.y)
    theta0 = np.zeros(len(design.param_names), dtype=float)

    def fun(theta: np.ndarray) -> float:
        value, _ = objective.objective_gradient(theta)
        return value

    def jac(theta: np.ndarray) -> np.ndarray:
        _, gradient = objective.objective_gradient(theta)
        return gradient

    result = minimize(
        fun,
        theta0,
        jac=jac,
        method="BFGS",
        options={
            "gtol": float(cmp_cfg.get("event_ppml_gradient_tol", 1e-7)),
            "maxiter": int(cmp_cfg.get("event_ppml_maxiter", 200)),
        },
    )
    if not np.isfinite(result.fun):
        raise ValueError("Fast conditional PPML failed: objective is not finite.")
    _, final_gradient = objective.objective_gradient(result.x)
    max_abs_gradient = float(np.max(np.abs(final_gradient))) if len(final_gradient) else 0.0
    success_tol = float(cmp_cfg.get("event_ppml_success_gradient_tol", 1e-4))
    converged = bool(result.success or max_abs_gradient <= success_tol)

    hessian = objective.hessian(result.x)
    meat = objective.cluster_meat(result.x)
    inv_hessian = np.linalg.pinv(hessian)
    vcov = inv_hessian @ meat @ inv_hessian
    n_groups = len(design.group_totals)
    n_params = len(design.param_names)
    if n_groups > 1 and design.kept_rows > n_params:
        vcov *= (n_groups / (n_groups - 1)) * ((design.kept_rows - 1) / (design.kept_rows - n_params))
    diag = np.diag(vcov)
    ses = np.sqrt(np.where(diag >= 0, diag, np.nan))
    return _ConditionalPPMLFit(
        params=pd.Series(result.x, index=design.param_names),
        std_errors=pd.Series(ses, index=design.param_names),
        covariance=pd.DataFrame(vcov, index=design.param_names, columns=design.param_names),
        nobs=design.kept_rows,
        converged=converged,
        message=f"{result.message}; max_abs_gradient={max_abs_gradient:.3g}",
    )


def _build_event_conditional_ppml_design(
    panel: pd.DataFrame,
    lhs: str,
    term_cols: list[str],
    term_to_time: dict[str, int],
    cmp_cfg: dict,
    *,
    fe_cols: Optional[list[str]] = None,
) -> _ConditionalPPMLDesign:
    unit_col = _unit_id_col(cmp_cfg)
    explicit_fe_cols = _conditional_ppml_explicit_fe_cols(fe_cols or [unit_col, "t"], cmp_cfg)
    required = _dedupe_existing([lhs, unit_col, "t", *term_cols, *explicit_fe_cols], panel)
    work = panel.loc[:, required].dropna(subset=required).copy()
    work[lhs] = pd.to_numeric(work[lhs], errors="coerce")
    work = work.loc[work[lhs].ge(0)].copy()
    for col in term_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work = work.dropna(subset=[lhs, unit_col, "t", *term_cols]).copy()
    if work.empty:
        raise ValueError("Fast conditional PPML sample is empty after filtering.")

    unit_totals = work.groupby(unit_col, sort=False)[lhs].transform("sum")
    positive_unit_mask = unit_totals.gt(0)
    dropped_rows = int((~positive_unit_mask).sum())
    dropped_units = int(work.loc[~positive_unit_mask, unit_col].nunique(dropna=True))
    work = work.loc[positive_unit_mask].copy()
    if work.empty:
        raise ValueError("Fast conditional PPML sample has no units with positive first-stage outcomes.")

    nuisance_cols: list[str] = []
    if explicit_fe_cols:
        dummy_frames: list[pd.DataFrame] = []
        for fe_col in explicit_fe_cols:
            dummies = pd.get_dummies(work[fe_col].astype(str), prefix=f"__fe_{_safe_name(fe_col)}", dtype=float)
            if dummies.shape[1] > 0:
                dummies = dummies.reindex(sorted(dummies.columns), axis=1)
                dummies = dummies.iloc[:, 1:]
            if not dummies.empty:
                dummy_frames.append(dummies)
            nuisance_cols.extend(str(col) for col in dummies.columns)
        if dummy_frames:
            work = pd.concat([work, *dummy_frames], axis=1, copy=False)
    else:
        years = sorted(int(v) for v in work["t"].dropna().unique())
        treatment_ref_years = [year for year in years if int(year) not in set(int(v) for v in term_to_time.values())]
        treatment_ref_year = treatment_ref_years[0] if treatment_ref_years else None
        calendar_ref_year = _calendar_year_dummy_reference(years, treatment_ref_year)
        year_frames: list[pd.Series] = []
        for year in years:
            if int(year) == int(calendar_ref_year):
                continue
            col = f"__year_{int(year)}"
            year_frames.append(work["t"].eq(int(year)).astype(float).rename(col))
            nuisance_cols.append(col)
        if year_frames:
            work = pd.concat([work, *year_frames], axis=1, copy=False)

    param_cols = [*nuisance_cols, *term_cols]
    x_frame = work.loc[:, param_cols].astype(float)
    nonzero_cols = x_frame.columns[x_frame.std(axis=0).gt(0)].tolist()
    dropped_param_cols = [col for col in param_cols if col not in nonzero_cols]
    if not nonzero_cols:
        raise ValueError("Fast conditional PPML design has no varying regressors.")
    x_frame = x_frame.loc[:, nonzero_cols]

    work["_unit_sort"] = work[unit_col].astype(str)
    work = work.assign(_row_order=np.arange(len(work))).sort_values(["_unit_sort", "t", "_row_order"])
    x = x_frame.loc[work.index].to_numpy(dtype=float)
    y = work[lhs].to_numpy(dtype=float)
    unit_codes = pd.Categorical(work["_unit_sort"], ordered=False).codes
    group_starts = np.r_[0, np.flatnonzero(np.diff(unit_codes)) + 1]
    group_ids = np.repeat(np.arange(len(group_starts)), np.diff(np.r_[group_starts, len(work)]))
    group_totals = np.add.reduceat(y, group_starts)
    if np.any(group_totals <= 0):
        raise ValueError("Fast conditional PPML retained a nonpositive-outcome unit.")
    if dropped_param_cols and _verbose(cmp_cfg):
        _log(
            "fast conditional PPML dropped non-varying parameters: "
            + ", ".join(dropped_param_cols[:8])
            + ("..." if len(dropped_param_cols) > 8 else ""),
            indent=4,
        )
    return _ConditionalPPMLDesign(
        x=x,
        y=y,
        group_ids=group_ids,
        group_starts=group_starts,
        group_totals=group_totals,
        param_names=list(x_frame.columns),
        kept_rows=int(len(work)),
        dropped_zero_outcome_rows=dropped_rows,
        dropped_zero_outcome_units=dropped_units,
    )


def _conditional_ppml_explicit_fe_cols(fe_cols: list[str], cmp_cfg: dict) -> list[str]:
    unit_col = _unit_id_col(cmp_cfg)
    return [col for col in fe_cols if col not in {unit_col, "t"}]


def _prefilter_ppml_first_stage_panel(
    panel: pd.DataFrame,
    *,
    lhs: str,
    estimator: str,
    family: str,
    fe_cols: list[str],
    cmp_cfg: dict,
    context: str,
) -> tuple[pd.DataFrame, Optional[str]]:
    if (
        estimator != "ppml"
        or family != "first_stage"
        or not bool(cmp_cfg.get("ppml_drop_all_zero_fe_groups", True))
        or lhs not in panel.columns
    ):
        return panel, None
    group_col = _ppml_first_stage_group_col(panel, fe_cols, cmp_cfg)
    if group_col is None:
        return panel, None
    work = panel.copy()
    y = pd.to_numeric(work[lhs], errors="coerce")
    work = work.loc[y.notna() & y.ge(0)].copy()
    if work.empty:
        return work, "all rows dropped after nonnegative PPML outcome filter"
    y = pd.to_numeric(work[lhs], errors="coerce")
    group_totals = y.groupby(work[group_col].astype(str), sort=False).transform("sum")
    keep = group_totals.gt(0)
    dropped_rows = int((~keep).sum())
    if dropped_rows == 0:
        return work, None
    dropped_groups = int(work.loc[~keep, group_col].astype(str).nunique(dropna=True))
    out = work.loc[keep].copy()
    note = (
        f"dropped {dropped_rows:,} all-zero {group_col} rows "
        f"({dropped_groups:,} FE groups)"
    )
    if _verbose(cmp_cfg):
        _log(f"PPML prefilter | {context}: {note}", indent=4)
    return out, note


def _ppml_first_stage_group_col(panel: pd.DataFrame, fe_cols: list[str], cmp_cfg: dict) -> Optional[str]:
    unit_col = _unit_id_col(cmp_cfg)
    if unit_col in fe_cols and unit_col in panel.columns:
        return unit_col
    for col in fe_cols:
        if col in panel.columns and ("unit" in col or col == unit_col):
            return col
    for col in fe_cols:
        if col in panel.columns and col not in {"t", "year_stack_fe", "baseline_size_growth_year_fe"}:
            return col
    return None


def _fit_model(panel: pd.DataFrame, lhs: str, terms: list[str], estimator: str, fe_cols: list[str], cluster_col: str):
    if pf is None:
        raise ImportError("pyfixest is required to run design_comparison_suite regressions.")
    rhs = " + ".join(terms)
    fe_term = " | " + " + ".join(fe_cols) if fe_cols else ""
    formula = f"{lhs} ~ {rhs}{fe_term}"
    vcov = {"CRV1": cluster_col} if cluster_col in panel.columns else None
    fit_kwargs = {
        "vcov": vcov,
        "demeaner_backend": "rust",
        "copy_data": False,
        "store_data": False,
        "lean": True,
    }
    if estimator == "ppml":
        panel = panel.loc[pd.to_numeric(panel[lhs], errors="coerce").ge(0)].copy()
        return pf.fepois(formula, data=panel, **fit_kwargs)
    return pf.feols(formula, data=panel, **fit_kwargs)


def _fallback_fe_cols(
    panel: pd.DataFrame,
    requested_fe: list[str],
    cmp_cfg: dict,
    estimator: str,
    family: str,
) -> Optional[list[str]]:
    if not bool(cmp_cfg.get("ppml_first_stage_fallback_to_firm_year_fe", True)):
        return None
    if estimator != "ppml" or family != "first_stage":
        return None
    fallback = ["c", "t"]
    if all(col in panel.columns for col in fallback) and requested_fe != fallback:
        return fallback
    return None


def plot_dynamic_design(results: pd.DataFrame, design: str, x_col: str, figures_dir: Path) -> None:
    if plt is None or results.empty or x_col not in results.columns:
        return
    for family in ("first_stage", "reduced_form"):
        sub = results.loc[results["design"].eq(design) & results["family"].eq(family)].copy()
        if family == "reduced_form":
            for outcome, group in sub.groupby("outcome_col", dropna=False):
	                _plot_coef_group(
	                    group,
	                    x_col,
	                    figures_dir / f"dynamic_{family}_{design}_{_safe_name(outcome)}.png",
	                    f"{design}: {family} {_pretty_outcome_label(outcome)}",
	                    show_figures=bool(results.attrs.get("show_figures", False)),
	                )
        else:
            _plot_coef_group(
                sub,
                x_col,
                figures_dir / f"dynamic_{family}_{design}.png",
                f"{design}: {family}",
                show_figures=bool(results.attrs.get("show_figures", False)),
            )


def plot_raw_means_by_exposure(
    panel: pd.DataFrame,
    variants: Sequence[tuple[str, str]],
    cmp_cfg: dict,
    design: str,
    figures_dir: Path,
    *,
    stats: Optional[pd.DataFrame] = None,
) -> None:
    if plt is None or panel.empty:
        return
    _apply_laborlunch_plot_style()
    x_col = _x_col(panel, cmp_cfg)
    for spec, exposure_col in variants:
        if exposure_col not in panel.columns or x_col not in panel.columns:
            continue
        work = panel[["t", exposure_col, x_col]].dropna().copy()
        if work.empty:
            continue
        plot_col = "_first_stage_plot_value"
        work[plot_col] = _first_stage_transform(pd.to_numeric(work[x_col], errors="coerce"), cmp_cfg)
        q = pd.qcut(pd.to_numeric(work[exposure_col], errors="coerce"), 4, labels=False, duplicates="drop")
        work["exposure_q"] = q
        raw = work.groupby(["t", "exposure_q"], as_index=False)[plot_col].mean()
        fig, ax = plt.subplots(figsize=_figsize())
        for idx, (qv, group) in enumerate(raw.groupby("exposure_q")):
            color = _plot_color(idx)
            ax.plot(
                group["t"],
                group[plot_col],
                marker="o",
                markersize=MULTI_PLOT_MARKER_SIZE,
                linewidth=PLOT_LINE_WIDTH,
                color=color,
                label=f"Q{int(qv) + 1}",
            )
        ax.axvline(float(cmp_cfg.get("event_year", 2016)), color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"Mean {_first_stage_axis_label(cmp_cfg)}")
        _annotate_raw_summary(ax, work, spec, stats, time_col="t", outcome_col=plot_col, cmp_cfg=cmp_cfg)
        _right_legend(ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"raw_first_stage_{design}_{_safe_name(spec)}.png", dpi=220, bbox_inches="tight")
        _display_figure_if_requested(fig, cmp_cfg)
        plt.close(fig)


def plot_stacked_raw_means(
    stacked: pd.DataFrame,
    spec: str,
    cmp_cfg: dict,
    figures_dir: Path,
    *,
    stats: Optional[pd.DataFrame] = None,
) -> None:
    if plt is None or stacked.empty:
        return
    _apply_laborlunch_plot_style()
    x_col = _x_col(stacked, cmp_cfg)
    work = stacked.copy()
    plot_col = "_first_stage_plot_value"
    work[plot_col] = _first_stage_transform(pd.to_numeric(work[x_col], errors="coerce"), cmp_cfg)
    raw = work.groupby(["rel_time", "treated"], as_index=False)[plot_col].mean()
    fig, ax = plt.subplots(figsize=_figsize())
    for idx, (treated, group) in enumerate(raw.groupby("treated")):
        color = _plot_color(idx)
        ax.plot(
            group["rel_time"],
            group[plot_col],
            marker="o",
            markersize=MULTI_PLOT_MARKER_SIZE,
            linewidth=PLOT_LINE_WIDTH,
            color=color,
            label="Treated" if treated else "Control",
        )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Event time")
    ax.set_ylabel(f"Mean {_first_stage_axis_label(cmp_cfg)}")
    _annotate_raw_summary(ax, work, spec, stats, time_col="rel_time", outcome_col=plot_col, cmp_cfg=cmp_cfg)
    _right_legend(ax)
    fig.tight_layout()
    fig.savefig(figures_dir / f"raw_first_stage_stacked_did_{_safe_name(spec)}.png", dpi=220, bbox_inches="tight")
    _display_figure_if_requested(fig, cmp_cfg)
    plt.close(fig)


def _plot_coef_group(group: pd.DataFrame, x_col: str, path: Path, title: str, *, show_figures: bool = False) -> None:
    if plt is None or group.empty:
        return
    work = group.dropna(subset=["coef", "se", x_col]).copy()
    if work.empty:
        return
    _apply_laborlunch_plot_style()
    specs = list(work["spec"].drop_duplicates())
    offsets = _plot_offsets(len(specs), span=0.42)
    fig, ax = plt.subplots(figsize=_figsize())
    for idx, spec in enumerate(specs):
        sub = work.loc[work["spec"].eq(spec)].sort_values(x_col)
        x = pd.to_numeric(sub[x_col], errors="coerce") + float(offsets[idx])
        color = _plot_color(idx)
        ax.errorbar(
            x,
            sub["coef"],
            yerr=1.96 * sub["se"],
            fmt="-",
            color=color,
            marker="o",
            markersize=PLOT_MARKER_SIZE,
            linewidth=PLOT_LINE_WIDTH,
            label=_pretty_spec_label(spec),
            **_errorbar_kwargs(color, marker_size=PLOT_MARKER_SIZE),
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    if 0 in set(pd.to_numeric(work[x_col], errors="coerce").dropna().astype(int)):
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("")
    ax.set_xlabel(_axis_label(x_col))
    ax.set_ylabel("Coefficient on normalized exposure")
    _annotate_effect_summary(ax, work, x_col)
    _right_legend(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    _display_figure_if_requested(fig, {"show_figures": show_figures})
    plt.close(fig)


def _apply_laborlunch_plot_style() -> None:
    if plt is None:
        return
    if llstyle is not None:
        llstyle.apply_style()
        return
    plt.rcParams.update(
        {
            "figure.figsize": _figsize(),
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def _figsize() -> tuple[float, float]:
    return tuple(llstyle.FIGSIZE) if llstyle is not None else (9.2, 5.1)


def _plot_color(idx: int) -> str:
    return llstyle.color(idx) if llstyle is not None else PLOT_PALETTE[idx % len(PLOT_PALETTE)]


def _plot_offsets(n_series: int, *, span: float) -> np.ndarray:
    return llstyle.offsets(n_series, span=span) if llstyle is not None else (
        np.array([0.0]) if n_series <= 1 else np.linspace(-span / 2.0, span / 2.0, num=n_series)
    )


def _errorbar_kwargs(series_color: str, marker_size: float) -> dict[str, object]:
    if llstyle is not None:
        return llstyle.errorbar_kwargs(series_color, marker_size=marker_size, alpha=ERRORBAR_INTERVAL_ALPHA)
    return {
        "ecolor": series_color,
        "elinewidth": marker_size,
        "capsize": 0,
        "alpha": ERRORBAR_INTERVAL_ALPHA,
    }


def _right_legend(ax) -> None:
    if llstyle is not None:
        llstyle.right_legend(ax)
    else:
        ax.legend(loc="best", frameon=True, framealpha=0.86, fontsize=9)


def _annotate_effect_summary(ax, work: pd.DataFrame, x_col: str) -> None:
    if work.empty:
        return
    pieces = []
    for spec, sub in work.groupby("spec", sort=False):
        row = _annotation_row(sub, x_col)
        if row is None:
            continue
        pieces.append(
            f"{_pretty_spec_label(spec)}: "
            f"base={_fmt_num(row.get('baseline_mean'), digits=2)}, "
            f"eff={_fmt_num(row.get('effect_size'), digits=2)}, "
            f"F={_fmt_num(row.get('f_stat'), digits=1)}"
        )
    if not pieces:
        return
    ax.text(
        0.015,
        0.02,
        "\n".join(pieces),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )


def _annotate_raw_summary(
    ax,
    work: pd.DataFrame,
    spec: str,
    stats: Optional[pd.DataFrame],
    *,
    time_col: str,
    outcome_col: str,
    cmp_cfg: dict,
) -> None:
    stat_row: Optional[pd.Series] = None
    if isinstance(stats, pd.DataFrame) and not stats.empty and "spec" in stats.columns:
        sub = stats.loc[stats["spec"].eq(spec) & stats["family"].eq("first_stage")].copy()
        if not sub.empty:
            stat_row = _annotation_row(sub, "rel_time" if "rel_time" in sub.columns else "horizon" if "horizon" in sub.columns else "year")
    baseline = stat_row.get("baseline_mean") if stat_row is not None else np.nan
    if not pd.notna(baseline):
        baseline = _raw_baseline_mean(work, time_col=time_col, outcome_col=outcome_col, cmp_cfg=cmp_cfg)
    effect = stat_row.get("effect_size") if stat_row is not None else np.nan
    f_stat = stat_row.get("f_stat") if stat_row is not None else np.nan
    ax.text(
        0.015,
        0.02,
        f"base={_fmt_num(baseline, digits=2)}, eff={_fmt_num(effect, digits=2)}, F={_fmt_num(f_stat, digits=1)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )


def _raw_baseline_mean(work: pd.DataFrame, *, time_col: str, outcome_col: str, cmp_cfg: dict) -> float:
    if time_col not in work.columns or outcome_col not in work.columns:
        return float("nan")
    t = pd.to_numeric(work[time_col], errors="coerce")
    if time_col == "rel_time":
        mask = t.lt(0)
    else:
        mask = t.between(int(cmp_cfg.get("baseline_start", 2010)), int(cmp_cfg.get("baseline_end", 2013)))
    values = pd.to_numeric(work.loc[mask, outcome_col], errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def _annotation_row(sub: pd.DataFrame, x_col: str) -> Optional[pd.Series]:
    if sub.empty:
        return None
    picked = _pick_event_row(sub)
    if not picked.empty:
        return picked.iloc[0]
    if x_col not in sub.columns:
        return sub.iloc[0]
    work = sub.copy()
    x = pd.to_numeric(work[x_col], errors="coerce")
    if x.notna().any():
        work = work.assign(_annotation_distance=x.abs())
        return work.sort_values("_annotation_distance").iloc[0]
    return work.iloc[0]


def _axis_label(x_col: str) -> str:
    labels = {"event_time": "Event time", "rel_time": "Event time", "horizon": "Outcome horizon", "year": "Year"}
    return labels.get(str(x_col), str(x_col))


def _pretty_spec_label(spec: object) -> str:
    labels = {
        "ihmp_levels": "IHMP levels",
        "ihmp_share_levels": "IHMP share",
        "ihmp_levels_ar_residual": "IHMP levels (AR)",
        "ihmp_share_ar_residual": "IHMP share (AR)",
        "school_opt_share_binary": "School OPT share, binary",
        "school_opt_share_continuous": "School OPT share, continuous",
        "opt_probability_index_binary": "OPT index, binary",
        "opt_probability_index_continuous": "OPT index, continuous",
        "unmatched_ihmp_share": "Unmatched IHMP share",
        "matched_ihmp_share": "Matched IHMP share",
        "matched_international_share": "Matched international share",
        "unmatched_opt_takeup": "Unmatched OPT take-up",
        "matched_opt_takeup": "Matched OPT take-up",
    }
    return labels.get(str(spec), str(spec).replace("_", " "))


def _pretty_design_label(design: object) -> str:
    labels = {
        "shift_share": "Shift-share",
        "event_study": "Common-break event study",
        "stacked_did": "Stacked DiD",
    }
    return labels.get(str(design), str(design).replace("_", " "))


def _pretty_outcome_label(outcome: object) -> str:
    labels = {
        "y_new_hires_foreign_lag0": "IHS(foreign new grad hires)",
        "y_new_hires_native_lag0": "IHS(native new grad hires)",
        "y_new_hires_foreign_opt_likely_lag0": "IHS(foreign new grad hires, OPT-likely SOC2)",
        "y_new_hires_foreign_masters_lag0": "IHS(foreign master's new grad hires)",
        "y_new_hires_native_masters_lag0": "IHS(native master's new grad hires)",
        "y_new_hires_foreign_opt_likely_masters_lag0": "IHS(foreign master's new grad hires, OPT-likely SOC2)",
        "avg_tenure_foreign_new_hires_lag0": "Avg tenure among foreign new grad hires",
        "avg_tenure_new_hires_lag0": "Avg tenure among all new grad hires",
        "avg_tenure_foreign_new_hires_masters_lag0": "Avg tenure among foreign master's new grad hires",
        "avg_tenure_new_hires_masters_lag0": "Avg tenure among master's new grad hires",
        "y_intern_positions_opt_likely_lag0": "IHS(intern users, OPT-likely SOC2)",
        "y_intern_positions_foreign_lag0": "IHS(foreign intern users)",
        "y_intern_positions_opt_likely_foreign_lag0": "IHS(foreign intern users, OPT-likely SOC2)",
    }
    return labels.get(str(outcome), str(outcome).replace("_", " "))


def _fmt_num(value: object, *, digits: int, comma: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    fmt = f",.{digits}f" if comma else f".{digits}f"
    return f"{number:{fmt}}"


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _display_figure_if_requested(fig, cmp_cfg: dict) -> None:
    if not bool(cmp_cfg.get("show_figures", False)):
        return
    try:
        from IPython.display import display
    except ImportError:  # pragma: no cover
        if plt is not None:
            plt.show()
        return
    display(fig)


def _standardize_panel_ids(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "c" in out.columns:
        out["c"] = pd.to_numeric(out["c"], errors="coerce")
        out = out.dropna(subset=["c"])
        out["c"] = out["c"].astype(int)
    if "t" in out.columns:
        out["t"] = pd.to_numeric(out["t"], errors="coerce")
        out = out.dropna(subset=["t"])
        out["t"] = out["t"].astype(int)
    return out


def _standardize_feature_ids(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if "c" in out.columns:
        out["c"] = pd.to_numeric(out["c"], errors="coerce")
        out = out.dropna(subset=["c"])
        out["c"] = out["c"].astype(int)
    return out


def _prepare_regression_panel(panel: pd.DataFrame, cmp_cfg: dict, exposure_col: str) -> pd.DataFrame:
    work = panel.copy()
    if exposure_col not in work.columns:
        return pd.DataFrame()
    work = _ensure_fe_columns(work, cmp_cfg)
    unit_col = _unit_id_col(cmp_cfg)
    work[unit_col] = work[unit_col].astype(str)
    work["t"] = pd.to_numeric(work["t"], errors="coerce").astype("Int64").astype(str)
    work[exposure_col] = pd.to_numeric(work[exposure_col], errors="coerce")
    return work


def _event_study_needed_cols(work: pd.DataFrame, cmp_cfg: dict, exposure_col: str) -> list[str]:
    cols = [
        _unit_id_col(cmp_cfg),
        "t",
        exposure_col,
        _x_col(work, cmp_cfg),
        *_available_outcomes(work, cmp_cfg),
        *_fe_cols(work, cmp_cfg),
    ]
    return _dedupe_existing(cols, work)


def _ensure_fe_columns(work: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    out = work.copy()
    if "baseline_size_growth_year_fe" in out.columns:
        return out
    size_col = _first_present(out, ["firm_size_annual_pre_level", "headcount_size_baseline", "total_headcount_annual_pre_level"])
    growth_col = _first_present(out, ["firm_size_annual_pre_growth", "headcount_growth_asinh"])
    if size_col is None:
        return out
    size = pd.to_numeric(out[size_col], errors="coerce")
    growth = pd.to_numeric(out[growth_col], errors="coerce") if growth_col else pd.Series(np.nan, index=out.index)
    out["_baseline_size_bin"] = _qbin(size, int(cmp_cfg.get("baseline_size_bins", 10)))
    out["_baseline_growth_bin"] = _qbin(growth, int(cmp_cfg.get("baseline_growth_bins", 5)))
    out["baseline_size_growth_year_fe"] = (
        out["t"].astype(str) + "__" + out["_baseline_size_bin"].astype(str) + "__" + out["_baseline_growth_bin"].astype(str)
    )
    return out


def _qbin(series: pd.Series, n_bins: int) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(vals.rank(method="first"), n_bins, labels=False, duplicates="drop").astype("Int64").astype(str)
    except Exception:
        return pd.Series(["missing"] * len(series), index=series.index)


def _first_stage_lhs(work: pd.DataFrame, values: pd.Series, cmp_cfg: dict) -> tuple[str, str]:
    first_stage_type = str(cmp_cfg.get("first_stage_type", "ppml"))
    if first_stage_type == "ppml":
        col = "first_stage_lhs_ppml"
        work[col] = pd.to_numeric(values, errors="coerce")
        return col, "ppml"
    if first_stage_type == "ols_continuous":
        col = "first_stage_lhs_ols_continuous"
        work[col] = pd.to_numeric(values, errors="coerce")
        return col, "ols"
    if first_stage_type == "ols_ihs":
        col = "first_stage_lhs_ols_ihs"
        work[col] = _first_stage_transform(values, cmp_cfg)
        return col, "ols"
    col = "first_stage_lhs_ols_binary"
    work[col] = _first_stage_transform(values, cmp_cfg)
    return col, "ols"


def _first_stage_transform(values: pd.Series, cmp_cfg: dict) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    first_stage_type = str(cmp_cfg.get("first_stage_type", "ppml"))
    if first_stage_type == "ols_binary":
        return vals.fillna(0.0).gt(0).astype(float)
    if first_stage_type == "ols_ihs":
        return np.arcsinh(vals)
    return vals


def _first_stage_axis_label(cmp_cfg: dict) -> str:
    first_stage_type = str(cmp_cfg.get("first_stage_type", "ppml"))
    x_label = "master's OPT hires" if str(cmp_cfg.get("first_stage_col", "")).startswith("masters_") else "OPT hires"
    if first_stage_type == "ols_binary":
        return "master's OPT hire indicator" if x_label.startswith("master") else "any OPT hire indicator"
    if first_stage_type == "ols_ihs":
        return f"IHS({x_label})"
    return x_label


def _dynamic_value(panel: pd.DataFrame, value_col: str, horizon: int, *, unit_col: str) -> pd.Series:
    lookup = panel[[unit_col, "t", value_col]].copy()
    lookup["t"] = pd.to_numeric(lookup["t"], errors="coerce")
    mapper = {
        (str(c), int(t)): v
        for c, t, v in lookup.dropna(subset=[unit_col, "t"])[[unit_col, "t", value_col]].itertuples(index=False, name=None)
    }
    return pd.Series(
        [
            mapper.get((str(c), int(t) + int(horizon)), np.nan)
            if pd.notna(t) else np.nan
            for c, t in panel[[unit_col, "t"]].itertuples(index=False, name=None)
        ],
        index=panel.index,
    )


def _normalized_col(work: pd.DataFrame, col: str) -> str:
    out_col = f"{col}_std"
    vals = pd.to_numeric(work[col], errors="coerce")
    sd = vals.std(skipna=True)
    if not sd or not np.isfinite(sd):
        work[out_col] = vals
    else:
        work[out_col] = (vals - vals.mean(skipna=True)) / sd
    return out_col


def _transform_outcome(values: pd.Series, outcome: str, cmp_cfg: dict) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    if str(outcome).startswith("avg_tenure") and not bool(cmp_cfg.get("use_log_tenure_outcome", False)):
        return vals
    if _is_count_outcome(str(outcome)) and str(cmp_cfg.get("count_outcome_transform", "")).lower() in {"ihs", "asinh"}:
        return np.arcsinh(vals)
    if bool(cmp_cfg.get("use_log_outcome", True)):
        return vals.apply(lambda v: math.log1p(max(float(v), 0.0)) if pd.notna(v) else np.nan)
    return vals


def _is_count_outcome(outcome: str) -> bool:
    return str(outcome).startswith(COUNT_OUTCOME_PREFIXES)


def _regression_rows(work: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in required_cols if c in work.columns]
    panel = work.dropna(subset=cols).copy()
    for col in cols:
        if col not in {"c", "t", "baseline_size_growth_year_fe", "unit_stack_fe", "year_stack_fe", "stack_id"}:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel


def _add_year_interactions(work: pd.DataFrame, reg_col: str, years: list[int], ref_year: int) -> list[tuple[int, str]]:
    terms = []
    t_num = pd.to_numeric(work["t"], errors="coerce")
    for year in years:
        if int(year) == int(ref_year):
            continue
        col = f"{reg_col}_x_year_{year}"
        work[col] = pd.to_numeric(work[reg_col], errors="coerce") * t_num.eq(year).astype(float)
        terms.append((year, col))
    return terms


def _calendar_year_dummy_reference(years: Sequence[int], treatment_ref_year: Optional[int]) -> int:
    ordered = [int(year) for year in years]
    if not ordered:
        raise ValueError("Cannot choose a calendar-year reference from an empty year list.")
    if treatment_ref_year is None:
        return ordered[0]
    for year in ordered:
        if int(year) != int(treatment_ref_year):
            return int(year)
    return ordered[0]


def _stacked_matching_styles(cmp_cfg: dict) -> list[str]:
    raw = cmp_cfg.get("stacked_matching_styles", ["unmatched", "matched"])
    styles = _list_cfg(raw) if raw is not None else ["unmatched", "matched"]
    allowed = {"unmatched", "matched"}
    invalid = [style for style in styles if style not in allowed]
    if invalid:
        raise ValueError(f"Unknown stacked matching style(s): {invalid}. Valid styles: {sorted(allowed)}")
    return styles or ["unmatched", "matched"]


def _stacked_event_jump_quantile(cmp_cfg: dict, exposure_name: str) -> Optional[float]:
    by_exposure = cmp_cfg.get("stacked_event_jump_quantiles")
    if isinstance(by_exposure, dict) and exposure_name in by_exposure:
        raw = by_exposure.get(exposure_name)
    else:
        raw = cmp_cfg.get("stacked_event_jump_quantile")
    if raw is None or str(raw).strip().lower() in {"", "none", "null"}:
        return None
    return float(raw)


def _percentile_to_quantile(raw: float) -> float:
    value = float(raw)
    if value > 1.0:
        value = value / 100.0
    return min(max(value, 0.0), 1.0)


def _build_pair_frame(
    panel: pd.DataFrame,
    treated: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    matching_style: str,
    cmp_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    control_ids = list(controls["c"].dropna().unique())
    if not control_ids:
        return pd.DataFrame()
    features = panel.drop_duplicates("c").copy()
    treated_rows = treated.dropna(subset=["c", "g"]).copy()
    if matching_style == "unmatched":
        n_pairs = min(len(treated_rows), len(control_ids))
        if n_pairs <= 0:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "pair_id": range(n_pairs),
                "treated_c": treated_rows["c"].iloc[:n_pairs].to_numpy(),
                "control_c": control_ids[:n_pairs],
            }
        )
    if matching_style == "matched":
        return _build_nearest_neighbor_pair_frame(features, treated_rows, control_ids, cmp_cfg or {})
    rows = []
    used_controls: set[object] = set()
    pair_id = 0
    for _, row in treated_rows.iterrows():
        treated_c = row["c"]
        if matching_style == "matched":
            control_c = _nearest_control(features, treated_c, control_ids, used_controls)
            if control_c is None:
                continue
            used_controls.add(control_c)
        else:
            available = [c for c in control_ids if c not in used_controls]
            if not available:
                break
            control_c = available[0]
            used_controls.add(control_c)
        rows.append({"pair_id": pair_id, "treated_c": treated_c, "control_c": control_c})
        pair_id += 1
    return pd.DataFrame(rows)


def _build_nearest_neighbor_pair_frame(
    features: pd.DataFrame,
    treated: pd.DataFrame,
    control_ids: list[object],
    cmp_cfg: dict,
) -> pd.DataFrame:
    if NearestNeighbors is None:
        return _build_nearest_control_pair_frame_slow(features, treated, control_ids, cmp_cfg)
    feature_cols = _matching_feature_cols(features)
    if not feature_cols:
        return _build_nearest_control_pair_frame_slow(features, treated, control_ids, cmp_cfg)

    firm_features = _prepare_matching_features(features.drop_duplicates("c").copy(), cmp_cfg)
    firm_features["c_key"] = firm_features["c"].astype(str)
    control_keys = {str(c) for c in control_ids}
    controls = firm_features.loc[firm_features["c_key"].isin(control_keys)].copy()
    treated_features = treated.assign(c_key=treated["c"].astype(str)).merge(
        firm_features,
        on="c_key",
        how="left",
        suffixes=("", "_feature"),
    )
    treated_features = treated_features.dropna(subset=["c_key", *feature_cols]).copy()
    if controls.empty or treated_features.empty:
        return pd.DataFrame()
    exact_cols = _stacked_exact_match_cols(controls, cmp_cfg)
    if exact_cols:
        controls = controls.dropna(subset=exact_cols).copy()
        treated_features = treated_features.dropna(subset=exact_cols).copy()
        if controls.empty or treated_features.empty:
            return pd.DataFrame()

    medians = {
        col: _finite_or_default(pd.to_numeric(controls[col], errors="coerce").median(), 0.0)
        for col in feature_cols
    }
    scales = {
        col: _finite_or_default(pd.to_numeric(controls[col], errors="coerce").std(skipna=True), 1.0)
        for col in feature_cols
    }
    control_matrix = _matching_matrix(controls, feature_cols, medians, scales)
    treated_matrix = _matching_matrix(treated_features, feature_cols, medians, scales)
    controls = controls.reset_index(drop=True)
    treated_features = treated_features.reset_index(drop=True)

    rows: list[dict[str, object]] = []
    used_controls: set[object] = set()
    if exact_cols:
        controls["_stacked_exact_key"] = controls[exact_cols].astype(str).agg("\x1f".join, axis=1)
        treated_features["_stacked_exact_key"] = treated_features[exact_cols].astype(str).agg("\x1f".join, axis=1)
        control_groups = {
            str(key): np.fromiter(idx, dtype=int)
            for key, idx in controls.groupby("_stacked_exact_key", sort=False).groups.items()
        }
        for key, pos_index in treated_features.groupby("_stacked_exact_key", sort=False).groups.items():
            positions = list(pos_index)
            control_positions = control_groups.get(str(key))
            if control_positions is None or len(control_positions) == 0:
                continue
            _assign_exact_cell_greedy(
                controls,
                control_matrix,
                treated_features,
                treated_matrix,
                positions,
                control_positions,
                used_controls,
                rows,
                allow_reuse=bool(cmp_cfg.get("stacked_match_with_replacement", False)),
            )
    else:
        global_index = _NeighborIndex(controls, control_matrix)
        _assign_neighbor_batch(
            global_index,
            treated_features,
            treated_matrix,
            list(range(len(treated_features))),
            used_controls,
            rows,
            allow_reuse=bool(cmp_cfg.get("stacked_match_with_replacement", False)),
            initial_k=int(cmp_cfg.get("stacked_match_nn_initial_k", 25)),
            max_k=int(cmp_cfg.get("stacked_match_nn_max_k", 50)),
        )
    return pd.DataFrame(rows)


@dataclass
class _NeighborIndex:
    controls: pd.DataFrame
    matrix: np.ndarray

    def __post_init__(self) -> None:
        self.nn = NearestNeighbors(algorithm="auto").fit(self.matrix)  # type: ignore[union-attr]


def _assign_neighbor_batch(
    index: _NeighborIndex,
    treated_features: pd.DataFrame,
    treated_matrix: np.ndarray,
    positions: list[int],
    used_controls: set[object],
    rows: list[dict[str, object]],
    *,
    allow_reuse: bool = False,
    initial_k: int = 25,
    max_k: int = 100,
) -> list[int]:
    if not positions or len(index.controls) == 0:
        return positions
    pending = list(positions)
    n_controls = len(index.controls)
    max_neighbors = min(max(int(max_k), int(initial_k)), n_controls)
    k = min(int(initial_k), max_neighbors)
    while pending and k <= max_neighbors:
        query = treated_matrix[pending, :]
        _, neighbor_positions = index.nn.kneighbors(query, n_neighbors=k)
        still_pending: list[int] = []
        for treated_pos, candidates in zip(pending, neighbor_positions):
            control_c = None
            for candidate_pos in candidates:
                candidate_c = index.controls.iloc[int(candidate_pos)]["c"]
                if allow_reuse or candidate_c not in used_controls:
                    control_c = candidate_c
                    break
            if control_c is None:
                still_pending.append(treated_pos)
                continue
            if not allow_reuse:
                used_controls.add(control_c)
            rows.append(
                {
                    "pair_id": len(rows),
                    "treated_c": treated_features.iloc[int(treated_pos)]["c"],
                    "control_c": control_c,
                }
            )
        if not still_pending or k == max_neighbors:
            return still_pending
        pending = still_pending
        k = min(k * 2, max_neighbors)
    return pending


def _assign_exact_cell_greedy(
    controls: pd.DataFrame,
    control_matrix: np.ndarray,
    treated_features: pd.DataFrame,
    treated_matrix: np.ndarray,
    treated_positions: list[int],
    control_positions: np.ndarray,
    used_controls: set[object],
    rows: list[dict[str, object]],
    *,
    allow_reuse: bool = False,
) -> None:
    if not treated_positions or len(control_positions) == 0:
        return
    local_controls = controls.iloc[control_positions].reset_index(drop=True)
    local_matrix = control_matrix[control_positions, :]
    available_mask = np.ones(len(local_controls), dtype=bool)
    if not allow_reuse and used_controls:
        available_mask = ~local_controls["c"].isin(used_controls).to_numpy()
    if not available_mask.any():
        return
    local_positions = np.flatnonzero(available_mask)
    local_controls = local_controls.iloc[local_positions].reset_index(drop=True)
    local_matrix = local_matrix[local_positions, :]
    query = treated_matrix[treated_positions, :]
    dist = np.square(query[:, None, :] - local_matrix[None, :, :]).sum(axis=2)
    if allow_reuse:
        best = np.argmin(dist, axis=1)
        for treated_pos, control_pos in zip(treated_positions, best):
            rows.append(
                {
                    "pair_id": len(rows),
                    "treated_c": treated_features.iloc[int(treated_pos)]["c"],
                    "control_c": local_controls.iloc[int(control_pos)]["c"],
                }
            )
        return

    assigned_treated: set[int] = set()
    assigned_controls: set[int] = set()
    order = np.argsort(dist, axis=None)
    n_controls = dist.shape[1]
    for flat_pos in order:
        treated_ix = int(flat_pos // n_controls)
        control_ix = int(flat_pos % n_controls)
        if treated_ix in assigned_treated or control_ix in assigned_controls:
            continue
        treated_pos = treated_positions[treated_ix]
        control_c = local_controls.iloc[control_ix]["c"]
        used_controls.add(control_c)
        assigned_treated.add(treated_ix)
        assigned_controls.add(control_ix)
        rows.append(
            {
                "pair_id": len(rows),
                "treated_c": treated_features.iloc[int(treated_pos)]["c"],
                "control_c": control_c,
            }
        )
        if len(assigned_treated) == len(treated_positions) or len(assigned_controls) == len(local_controls):
            break


def _matching_feature_cols(features: pd.DataFrame) -> list[str]:
    cols = []
    size_col = _first_present(features, ["firm_size_annual_pre_level", "headcount_size_baseline", "total_headcount_annual_pre_level"])
    growth_col = _first_present(features, ["firm_size_annual_pre_growth", "headcount_growth_asinh"])
    if size_col is not None:
        cols.append(size_col)
    if growth_col is not None:
        cols.append(growth_col)
    return cols


def _prepare_matching_features(features: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    out = features.copy()
    if bool(cmp_cfg.get("stacked_match_exact_size_bin", True)):
        configured_col = cmp_cfg.get("stacked_match_size_bin_col")
        if configured_col is not None and str(configured_col) in out.columns:
            out["_stacked_match_size_bin"] = out[str(configured_col)].astype(str)
        elif "_stacked_match_size_bin" not in out.columns:
            size_col = _first_present(out, ["firm_size_annual_pre_level", "headcount_size_baseline", "total_headcount_annual_pre_level"])
            if size_col is not None:
                out["_stacked_match_size_bin"] = _qbin(
                    pd.to_numeric(out[size_col], errors="coerce"),
                    int(cmp_cfg.get("baseline_size_bins", 10)),
                )
            else:
                out["_stacked_match_size_bin"] = "missing"
    return out


def _stacked_exact_match_cols(features: pd.DataFrame, cmp_cfg: dict) -> list[str]:
    cols: list[str] = []
    if bool(cmp_cfg.get("stacked_match_exact_naics", True)) and "naics2" in features.columns:
        cols.append("naics2")
    metro_col = str(cmp_cfg.get("stacked_match_metro_col", "company_metro_feature"))
    if bool(cmp_cfg.get("stacked_match_exact_metro", True)) and metro_col in features.columns:
        cols.append(metro_col)
    if bool(cmp_cfg.get("stacked_match_exact_size_bin", True)) and "_stacked_match_size_bin" in features.columns:
        cols.append("_stacked_match_size_bin")
    return cols


def _matching_matrix(df: pd.DataFrame, cols: list[str], medians: dict[str, float], scales: dict[str, float]) -> np.ndarray:
    pieces = []
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce").fillna(medians.get(col, 0.0))
        scale = scales.get(col, 1.0) or 1.0
        pieces.append(((values - medians.get(col, 0.0)) / scale).to_numpy(dtype=float))
    return np.column_stack(pieces) if pieces else np.empty((len(df), 0))


def _finite_or_default(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out) or out == 0:
        return float(default)
    return out


def _build_nearest_control_pair_frame_slow(
    features: pd.DataFrame,
    treated: pd.DataFrame,
    control_ids: list[object],
    cmp_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    rows = []
    used_controls: set[object] = set()
    prepared = _prepare_matching_features(features, cmp_cfg or {})
    for _, row in treated.iterrows():
        control_c = _nearest_control(prepared, row["c"], control_ids, used_controls, cmp_cfg or {})
        if control_c is None:
            continue
        if not bool((cmp_cfg or {}).get("stacked_match_with_replacement", False)):
            used_controls.add(control_c)
        rows.append({"pair_id": len(rows), "treated_c": row["c"], "control_c": control_c})
    return pd.DataFrame(rows)


def _build_stacked_panel_fast(
    panel: pd.DataFrame,
    pair_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    *,
    cmp_cfg: dict,
    pre_window: int,
    post_window: int,
    exposure_col: Optional[str] = None,
) -> pd.DataFrame:
    if panel.empty or pair_df.empty or cohort_df.empty:
        return pd.DataFrame()
    cohort_lookup = cohort_df.dropna(subset=["g"]).rename(columns={"c": "treated_c", "g": "treated_g"})
    pair_cohorts = pair_df.merge(cohort_lookup[["treated_c", "treated_g"]], on="treated_c", how="inner")
    if pair_cohorts.empty:
        return pd.DataFrame()

    pair_long = pd.concat(
        [
            pair_cohorts[["pair_id", "treated_c", "treated_g"]].rename(columns={"treated_c": "c"}),
            pair_cohorts[["pair_id", "control_c", "treated_g"]].rename(columns={"control_c": "c"}),
        ],
        ignore_index=True,
    )
    pair_long["treated"] = [1] * len(pair_cohorts) + [0] * len(pair_cohorts)
    pair_long = pair_long.dropna(subset=["c", "treated_g"]).copy()

    work = panel.merge(pair_long, on="c", how="inner")
    if work.empty:
        return pd.DataFrame()
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work["treated_g"] = pd.to_numeric(work["treated_g"], errors="coerce")
    work = work.loc[
        work["t"].between(work["treated_g"] - int(pre_window), work["treated_g"] + int(post_window))
    ].copy()
    if work.empty:
        return pd.DataFrame()

    window_len = int(pre_window) + int(post_window) + 1
    control = work.loc[work["treated"].eq(0)].copy()
    control_counts = control.groupby("pair_id")["t"].nunique()
    valid_pairs = set(control_counts[control_counts.eq(window_len)].index)
    pair_year_counts = work.groupby("pair_id")["t"].nunique()
    valid_pairs &= set(pair_year_counts[pair_year_counts.eq(window_len)].index)
    if exposure_col is not None and exposure_col in work.columns:
        nonmissing_exposure_counts = (
            work.loc[pd.to_numeric(work[exposure_col], errors="coerce").notna()]
            .groupby("pair_id")["t"]
            .count()
        )
        valid_pairs &= set(nonmissing_exposure_counts[nonmissing_exposure_counts.eq(2 * window_len)].index)
    if not valid_pairs:
        return pd.DataFrame()
    out = work.loc[work["pair_id"].isin(valid_pairs)].copy()
    out["stack_id"] = out["pair_id"].astype(int)
    out["g"] = out["treated_g"].astype(int)
    out["rel_time"] = out["t"].astype(int) - out["g"].astype(int)
    out["pair_stack_fe"] = out["pair_id"].astype(str) + "__" + out["stack_id"].astype(str)
    out["unit_stack_fe"] = out["c"].astype(str) + "__" + out["stack_id"].astype(str)
    out["year_stack_fe"] = out["t"].astype(int).astype(str) + "__" + out["stack_id"].astype(str)
    out["high_exposure"] = out["treated"]
    out["trajectory_name"] = "comparison"
    return out.sort_values(["stack_id", "c", "t"]).reset_index(drop=True)


def _nearest_control(
    features: pd.DataFrame,
    treated_c: object,
    control_ids: list[object],
    used_controls: set[object],
    cmp_cfg: Optional[dict] = None,
) -> Optional[object]:
    cfg = cmp_cfg or {}
    t = features.loc[features["c"].eq(treated_c)]
    if t.empty:
        return None
    trow = t.iloc[0]
    if bool(cfg.get("stacked_match_with_replacement", False)):
        available_ids = control_ids
    else:
        available_ids = [c for c in control_ids if c not in used_controls]
    controls = features.loc[features["c"].isin(available_ids)].copy()
    if controls.empty:
        return None
    exact_cols = _stacked_exact_match_cols(controls, cfg)
    for col in exact_cols:
        if pd.isna(trow.get(col)):
            return None
        controls = controls.loc[controls[col].notna()].copy()
        controls = controls.loc[controls[col].astype(str).eq(str(trow.get(col)))].copy()
        if controls.empty:
            return None
    size_col = _first_present(controls, ["firm_size_annual_pre_level", "headcount_size_baseline", "total_headcount_annual_pre_level"])
    growth_col = _first_present(controls, ["firm_size_annual_pre_growth", "headcount_growth_asinh"])
    score = pd.Series(0.0, index=controls.index)
    if size_col is not None:
        scale = pd.to_numeric(features[size_col], errors="coerce").std(skipna=True) or 1.0
        score += (pd.to_numeric(controls[size_col], errors="coerce") - float(trow.get(size_col, 0.0))).abs() / scale
    if growth_col is not None:
        scale = pd.to_numeric(features[growth_col], errors="coerce").std(skipna=True) or 1.0
        score += (pd.to_numeric(controls[growth_col], errors="coerce") - float(trow.get(growth_col, 0.0))).abs() / scale
    return controls.loc[score.sort_values().index[0], "c"]


def _materialize_matched_panel(panel: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, pair in pair_df.iterrows():
        t_panel = panel.loc[panel["c"].eq(pair["treated_c"])].copy()
        c_panel = panel.loc[panel["c"].eq(pair["control_c"])].copy()
        t_panel["pair_id"] = int(pair["pair_id"])
        c_panel["pair_id"] = int(pair["pair_id"])
        t_panel["treated"] = 1
        c_panel["treated"] = 0
        t_panel["high_exposure"] = 1
        c_panel["high_exposure"] = 0
        t_panel["trajectory_name"] = "comparison"
        c_panel["trajectory_name"] = "comparison"
        frames.extend([t_panel, c_panel])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _coef_se(fit, term: str) -> tuple[Optional[float], Optional[float]]:
    if fit is None:
        return None, None
    try:
        coefs = fit.coef()
        ses = fit.se()
        return float(coefs.get(term, np.nan)), float(ses.get(term, np.nan))
    except Exception:
        if hasattr(fit, "params") and hasattr(fit, "std_errors"):
            try:
                return float(fit.params.get(term, np.nan)), float(fit.std_errors.get(term, np.nan))
            except Exception:
                pass
        try:
            tidy = fit.tidy()
            if term in tidy.index:
                return float(tidy.loc[term, "Estimate"]), float(tidy.loc[term, "Std. Error"])
        except Exception:
            pass
    return None, None


def _fit_params_cov(fit) -> tuple[pd.Series, pd.DataFrame]:
    if hasattr(fit, "params") and hasattr(fit, "covariance"):
        params = pd.Series(getattr(fit, "params"), dtype=float)
        cov = pd.DataFrame(getattr(fit, "covariance"))
        cov = cov.reindex(index=params.index, columns=params.index)
        return params, cov
    params = pd.Series(fit.coef(), dtype=float)
    raw_cov = getattr(fit, "_vcov", None)
    if raw_cov is not None:
        cov_array = np.asarray(raw_cov, dtype=float)
        if cov_array.shape == (len(params), len(params)):
            return params, pd.DataFrame(cov_array, index=params.index, columns=params.index)
    ses = pd.Series(fit.se(), dtype=float).reindex(params.index)
    cov = pd.DataFrame(np.diag(np.square(ses.to_numpy(dtype=float))), index=params.index, columns=params.index)
    return params, cov


def _single_term_f(coef: Optional[float], se: Optional[float]) -> Optional[float]:
    if coef is None or se is None or not se or not np.isfinite(se):
        return None
    return float((coef / se) ** 2)


def _empty_result(family: str, design: str, spec: str, estimator: str, time_name: str, time_value: int, outcome_col: str, error: str) -> dict[str, object]:
    row = _result_base(family, design, spec, estimator, time_name, time_value, outcome_col, None, None)
    row["error"] = error
    return row


def _reference_result(
    *,
    family: str,
    design: str,
    spec: str,
    estimator: str,
    time_name: str,
    time_value: int,
    outcome_col: str,
    extra: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    row = _result_base(family, design, spec, estimator, time_name, time_value, outcome_col, None, None)
    row.update({"coef": 0.0, "se": 0.0, "f_stat": np.nan, "is_reference": True})
    if extra:
        row.update(extra)
    return row


def _annotate_event_time(rows: list[dict[str, object]], *, event_year: int) -> None:
    for row in rows:
        if "year" not in row:
            continue
        year = pd.to_numeric(pd.Series([row.get("year")]), errors="coerce").iloc[0]
        row["event_time"] = int(year - int(event_year)) if pd.notna(year) else np.nan
        row["is_reference"] = False


def _result_base(
    family: str,
    design: str,
    spec: str,
    estimator: str,
    time_name: str,
    time_value: int,
    outcome_col: str,
    n_obs: Optional[int],
    n_units: Optional[int],
) -> dict[str, object]:
    row: dict[str, object] = {
        "family": family,
        "design": design,
        "spec": spec,
        "estimator": estimator,
        "outcome_col": outcome_col,
        "coef": np.nan,
        "se": np.nan,
        "f_stat": np.nan,
        "n_obs": n_obs,
        "n_units": n_units,
    }
    row[time_name] = int(time_value)
    return row


def _pick_event_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    distance = pd.Series(np.nan, index=work.index, dtype=float)
    for col in ("event_time", "horizon", "rel_time"):
        if col not in work.columns:
            continue
        values = pd.to_numeric(work[col], errors="coerce")
        distance = distance.fillna(values.abs())
    if "year" in work.columns:
        years = pd.to_numeric(work["year"], errors="coerce")
        distance = distance.fillna((years - 2016).abs())
    work["_event_distance"] = distance.fillna(0)
    return work.sort_values(["design", "spec", "_event_distance"]).drop_duplicates(["design", "spec"])


def _validate_comparison_config(cmp_cfg: dict) -> None:
    unit = str(cmp_cfg.get("unit", "firm"))
    if unit not in UNITS:
        raise ValueError(f"design_comparison.unit must be one of {sorted(UNITS)}")
    first_stage_type = str(cmp_cfg.get("first_stage_type", "ppml"))
    if first_stage_type not in FIRST_STAGE_TYPES:
        raise ValueError(f"design_comparison.first_stage_type must be one of {sorted(FIRST_STAGE_TYPES)}")
    _selected_designs(cmp_cfg)
    _stacked_exposures_to_run(cmp_cfg)


def _verbose(cmp_cfg: dict) -> bool:
    return bool(cmp_cfg.get("verbose", True))


def _log(message: str, *, indent: int = 0) -> None:
    prefix = "  " * int(indent)
    print(f"[design_comparison] {prefix}{message}", flush=True)


def _fmt_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def _stage_label(name: str) -> str:
    if name in DATA_STAGE_ORDER:
        return f"Data stage {DATA_STAGE_ORDER.index(name) + 1}/{len(DATA_STAGE_ORDER)}: {name}"
    return f"Data stage: {name}"


def _remaining_stages(name: str, stages: Sequence[str]) -> str:
    if name not in stages:
        return "unknown"
    remaining = stages[stages.index(name) + 1 :]
    return ", ".join(remaining) if remaining else "none"


def _panel_summary_text(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame):
        return "not a DataFrame"
    parts = [f"{len(df):,} rows", f"{len(df.columns):,} cols"]
    if "c" in df.columns:
        parts.append(f"{df['c'].nunique(dropna=True):,} units")
    if "t" in df.columns and not df.empty:
        t = pd.to_numeric(df["t"], errors="coerce")
        if t.notna().any():
            parts.append(f"years {int(t.min())}-{int(t.max())}")
    return " | ".join(parts)


def _n_units_text(df: pd.DataFrame, cmp_cfg: dict) -> str:
    unit_col = _unit_id_col(cmp_cfg)
    if unit_col not in df.columns:
        return "unknown"
    return f"{df[unit_col].nunique(dropna=True):,}"


def _year_range_text(df: pd.DataFrame) -> str:
    if "t" not in df.columns or df.empty:
        return "unknown"
    t = pd.to_numeric(df["t"], errors="coerce")
    if not t.notna().any():
        return "unknown"
    return f"{int(t.min())}-{int(t.max())}"


def _log_attrition(label: str, before: pd.DataFrame, after: pd.DataFrame, cmp_cfg: dict) -> None:
    before_units = before["c"].nunique(dropna=True) if "c" in before.columns else np.nan
    after_units = after["c"].nunique(dropna=True) if "c" in after.columns else np.nan
    _log(label, indent=1)
    _log(f"before: {len(before):,} rows | {before_units:,} units", indent=2)
    _log(
        f"after years {cmp_cfg.get('data_min_t', 2010)}-{cmp_cfg.get('data_max_t', 2022)} "
        f"and pre avg employment >= {cmp_cfg.get('min_pre_avg_employment', 10)}: "
        f"{len(after):,} rows | {after_units:,} units",
        indent=2,
    )


def _matched_base_cfg(cfg: dict) -> dict:
    configured = get_cfg_section(cfg, "design_comparison").get("matched_config_path")
    if configured:
        return load_config(configured)
    if MATCHED_DEFAULT_CONFIG_PATH.exists():
        return load_config(MATCHED_DEFAULT_CONFIG_PATH)
    return {"matched_exposure_design": {}}


def _sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def _path_or_default(value: object, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default
    return Path(str(value))


def _mtime(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    p = Path(path)
    return p.stat().st_mtime if p.exists() else None


def _list_cfg(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [v.strip() for v in raw.split(",") if v.strip()]
    return [str(v).strip() for v in raw if str(v).strip()]  # type: ignore[union-attr]


def _parse_design_list(raw: Sequence[str] | str) -> list[str]:
    if isinstance(raw, str):
        values = [v.strip() for v in raw.split(",") if v.strip()]
    else:
        values = [str(v).strip() for v in raw if str(v).strip()]
    return values or list(DESIGN_NAMES)


def _selected_designs(cmp_cfg: dict) -> list[str]:
    raw = cmp_cfg.get("designs_to_run", DESIGN_NAMES)
    selected = _parse_design_list(raw) if raw is not None else list(DESIGN_NAMES)
    invalid = [name for name in selected if name not in DESIGN_NAMES]
    if invalid:
        raise ValueError(f"Unknown design(s) in designs_to_run: {invalid}. Valid designs: {list(DESIGN_NAMES)}")
    return [name for name in DESIGN_NAMES if name in set(selected)]


def _stacked_exposures_to_run(cmp_cfg: dict) -> list[str]:
    valid = ["ihmp_share", "international_share", "opt_takeup"]
    raw = cmp_cfg.get("stacked_exposures_to_run", valid)
    selected = _list_cfg(raw) if raw is not None else valid
    selected = selected or valid
    invalid = [name for name in selected if name not in valid]
    if invalid:
        raise ValueError(f"Unknown stacked exposure(s): {invalid}. Valid exposures: {valid}")
    return [name for name in valid if name in set(selected)]


def _configured_shift_share_variant_cols(cmp_cfg: dict) -> list[str]:
    configured = cmp_cfg.get("shift_share_instrument_variants")
    if not configured:
        return []
    raw = configured.items() if isinstance(configured, dict) else configured
    cols: list[str] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            _, col = item
        else:
            col = item
        cols.append(str(col))
    return cols


def _ar_residual_school_metric(growth: pd.DataFrame, *, value_col: str, lag_col: str) -> pd.Series:
    resid = pd.Series(0.0, index=growth.index, dtype="float64")
    mask = growth[value_col].notna() & growth[lag_col].notna()
    if mask.sum() < 3:
        return growth.groupby("k", sort=False)[value_col].diff().fillna(0.0)
    try:
        y = pd.to_numeric(growth.loc[mask, value_col], errors="coerce").astype(float)
        lag = pd.to_numeric(growth.loc[mask, lag_col], errors="coerce").astype(float)
        y_resid = ssa._residualize_fixed_effects(  # noqa: SLF001
            y,
            [growth.loc[mask, "k"].astype(str), growth.loc[mask, "t"].astype(str)],
        )
        lag_resid = ssa._residualize_fixed_effects(  # noqa: SLF001
            lag,
            [growth.loc[mask, "k"].astype(str), growth.loc[mask, "t"].astype(str)],
        )
        denom = float(np.dot(lag_resid, lag_resid))
        beta = float(np.dot(lag_resid, y_resid) / denom) if denom > 0 else 0.0
        resid.loc[mask] = y_resid - beta * lag_resid
    except Exception:
        resid = growth.groupby("k", sort=False)[value_col].diff().fillna(0.0)
    return resid


def _dedupe_existing(cols: Sequence[object], df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for col in cols:
        if col is None:
            continue
        name = str(col)
        if name in seen or name not in df.columns:
            continue
        out.append(name)
        seen.add(name)
    return out


def _available_outcomes(panel: pd.DataFrame, cmp_cfg: dict) -> list[str]:
    return [col for col in _list_cfg(cmp_cfg.get("outcome_cols", DEFAULT_OUTCOMES)) if col in panel.columns]


def _x_col(panel: pd.DataFrame, cmp_cfg: Optional[dict] = None) -> str:
    configured = None if cmp_cfg is None else cmp_cfg.get("first_stage_col")
    if configured is not None and str(configured).strip():
        configured_col = str(configured).strip()
        if configured_col not in panel.columns:
            raise ValueError(f"Configured first-stage column {configured_col!r} is missing.")
        return configured_col
    col = _first_present(panel, ["any_opt_hires_correction_aware", "masters_opt_hires_correction_aware", "masters_opt_hires", "x_bin_any_nonzero"])
    if col is None:
        raise ValueError("No first-stage treatment column found.")
    return col


def _unit_id_col(cmp_cfg: dict) -> str:
    return "c"


def _use_size_year_fe(cmp_cfg: dict) -> bool:
    return not bool(cmp_cfg.get("exclude_size_year_fe", True))


def _event_panelols_fe_label(cmp_cfg: dict) -> str:
    label = "entity+year_dummies"
    if _use_size_year_fe(cmp_cfg):
        label += "+baseline_size_growth_year_fe"
    return label


def _fe_cols(work: pd.DataFrame, cmp_cfg: dict) -> list[str]:
    cols = ["c", "t"]
    if _use_size_year_fe(cmp_cfg) and "baseline_size_growth_year_fe" in work.columns:
        cols.append("baseline_size_growth_year_fe")
    return cols


def _first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _shift_share_variants(cmp_cfg: dict, panel: pd.DataFrame) -> list[tuple[str, str]]:
    configured = cmp_cfg.get("shift_share_instrument_variants")
    if configured:
        raw = configured.items() if isinstance(configured, dict) else configured
        out = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                name, col = item
            else:
                name = col = str(item)
            if str(col) in panel.columns:
                out.append((str(name), str(col)))
        return out
    candidates = [
        ("ihmp_levels", "z_ct"),
        ("ihmp_share_levels", "z_ct_ihmp_share"),
        ("ihmp_levels_ar_residual", "z_ct_flow_ar_resid"),
        ("ihmp_share_ar_residual", "z_ct_ihmp_share_ar_resid"),
    ]
    return [(name, col) for name, col in candidates if col in panel.columns]


def _is_sum_like_col(col: str) -> bool:
    lowered = col.lower()
    return (
        lowered.startswith("y_")
        or "headcount" in lowered
        or "hires" in lowered
        or lowered.startswith("z_ct")
        or lowered.startswith("n_")
    ) and "share" not in lowered and "rate" not in lowered and "avg" not in lowered


def _safe_name(value: object) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_")
    return text or "value"


def _lag_suffix(value: int) -> str:
    return f"m{abs(int(value))}" if int(value) < 0 else str(int(value))


def _parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force-rebuild-base", action="store_true")
    parser.add_argument("--unit", choices=sorted(UNITS))
    parser.add_argument("--first-stage", choices=sorted(FIRST_STAGE_TYPES), dest="first_stage_type")
    parser.add_argument("--first-stage-col", type=str, help="Override the first-stage treatment/outcome column")
    parser.add_argument("--designs", type=str, help="Comma-separated subset: shift_share,event_study,stacked_did")
    parser.add_argument("--stacked-exposures", type=str, help="Comma-separated subset: ihmp_share,opt_takeup")
    parser.add_argument("--out-dir", type=Path)
    if args is None and _running_under_ipykernel():
        args = []
    return parser.parse_args(args)


def _running_under_ipykernel() -> bool:
    argv0 = Path(sys.argv[0]).name if sys.argv else ""
    return argv0 == "ipykernel_launcher.py" or "ipykernel" in sys.modules


def main(args: Optional[Iterable[str]] = None) -> dict[str, object]:
    parsed = _parse_args(args)
    return run_design_comparison(
        config_path=parsed.config,
        force_rebuild_base=parsed.force_rebuild_base,
        unit=parsed.unit,
        first_stage_type=parsed.first_stage_type,
        first_stage_col=parsed.first_stage_col,
        designs=parsed.designs,
        stacked_exposures=parsed.stacked_exposures,
        out_dir=parsed.out_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
