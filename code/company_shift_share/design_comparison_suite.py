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
    import pyfixest as pf
except ImportError:  # pragma: no cover
    pf = None  # type: ignore[assignment]

try:
    from linearmodels.panel import PanelOLS
except ImportError:  # pragma: no cover
    PanelOLS = None  # type: ignore[assignment,misc]

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
        load_or_build_source_analysis_panel,
        load_or_build_wrds_company_year_workforce_cache,
        load_or_build_wrds_school_flows_cache,
    )


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "company_shift_share_design_comparison.yaml"
)
SUITE_VERSION = "2026-05-01-design1-state-panel"

DEFAULT_OUTCOMES = [
    "y_cst_lag0",
    "y_new_hires_lag0",
    "y_new_hires_foreign_lag0",
    "y_new_hires_native_lag0",
    "avg_tenure_years_lag0",
]
FOREIGN_NEW_HIRE_OUTCOME = "y_new_hires_foreign_lag0"
FIRST_STAGE_TYPES = {"ppml", "ols_continuous", "ols_binary"}
UNITS = {"firm", "local_market"}
DATA_STAGE_ORDER = [
    "source_analysis_panel",
    "company_features",
    "shift_share_panel",
    "opt_probability_index",
    "workforce_panel",
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
        path = self._paths_cfg().get("shift_share_analysis_panel")
        cached = self._load_parquet_if_available(path)
        if cached is not None:
            return cached
        fallback = get_cfg_section(load_config(DEFAULT_SHIFT_SHARE_CONFIG_PATH), "paths").get("analysis_panel")
        cached = self._load_parquet_if_available(fallback)
        if cached is not None:
            return cached
        raise FileNotFoundError(
            "Shift-share analysis panel cache is missing. Run shift_share_analysis once, "
            "or set paths.shift_share_analysis_panel to an existing parquet."
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
    if out_dir is not None:
        paths_cfg["out_dir"] = str(out_dir)
    _validate_comparison_config(cmp_cfg)

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
            f"baseline={cmp_cfg.get('baseline_start')}-{cmp_cfg.get('baseline_end')}",
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
        _log("Work plan: prepare panels -> design 1 shift-share -> design 2 event-study -> design 3 stacked DiD -> outputs")
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
    if _verbose(cmp_cfg):
        _log("Design 1/3: shift_share starting")
    all_results.append(run_shift_share_design(prepared["shift_share"], cmp_cfg, figures_dir))
    if _verbose(cmp_cfg):
        _log("Design 1/3: shift_share finished")
        _log("Design 2/3: event_study starting")
    all_results.append(run_event_study_design(prepared["event_study"], store, cmp_cfg, figures_dir))
    if _verbose(cmp_cfg):
        _log("Design 2/3: event_study finished")
        _log("Design 3/3: stacked_did starting")
    all_results.append(run_stacked_did_design(prepared["stacked_did"], cmp_cfg, figures_dir))
    if _verbose(cmp_cfg):
        _log("Design 3/3: stacked_did finished")
    coef_df = pd.concat([df for df in all_results if df is not None and not df.empty], ignore_index=True)
    if _verbose(cmp_cfg):
        _log(f"writing outputs | coefficient rows={len(coef_df):,}")
    coef_path = tables_dir / "all_design_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    final_table = build_final_comparison_table(coef_df)
    final_csv = tables_dir / "design_comparison_first_stage_rf.csv"
    final_tex = tables_dir / "design_comparison_first_stage_rf.tex"
    final_table.to_csv(final_csv, index=False)
    try:
        final_tex.write_text(final_table.to_latex(index=False))
    except Exception:
        final_tex.write_text(final_table.to_string(index=False))

    summary = build_prepared_panel_summary(prepared)
    summary_path = tables_dir / "prepared_panel_summary.csv"
    summary.to_csv(summary_path, index=False)

    store.manifest["selected_config"] = {
        "unit": cmp_cfg.get("unit"),
        "first_stage_type": cmp_cfg.get("first_stage_type"),
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
    out.attrs["show_figures"] = bool(cmp_cfg.get("show_figures", False))
    plot_dynamic_design(out, "shift_share", "horizon", figures_dir)
    plot_raw_means_by_exposure(panel, variants, cmp_cfg, "shift_share", figures_dir)
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
            x_col = _x_col(work)
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
        ("school_opt_share_continuous", "school_opt_share_new_hire_annual_pre_level", "continuous"),
        ("opt_probability_index_binary", "predicted_prob", "binary"),
        ("opt_probability_index_continuous", "predicted_prob", "continuous"),
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
    out.attrs["show_figures"] = bool(cmp_cfg.get("show_figures", False))
    plot_dynamic_design(out, "event_study", "year", figures_dir)
    plot_raw_means_by_exposure(panel, [(v[0], v[1]) for v in variants if v[1] in panel.columns], cmp_cfg, "event_study", figures_dir)
    return out


def run_stacked_did_design(panel: pd.DataFrame, cmp_cfg: dict, figures_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    exposure_candidates = [
        ("ihmp_share", _first_present(panel, ["z_ct_ihmp_share", "z_ct"])),
        ("opt_takeup", _first_present(panel, ["any_opt_hires_correction_aware", "masters_opt_hires_correction_aware"])),
    ]
    available_exposures = [(name, col) for name, col in exposure_candidates if col is not None]
    if _verbose(cmp_cfg):
        _log(
            f"stacked_did exposure families={len(available_exposures)} | "
            f"cohorts={cmp_cfg.get('stacked_min_cohort_year', 2013)}-{cmp_cfg.get('stacked_max_cohort_year', 2016)} | "
            f"window=±{cmp_cfg.get('stacked_pre_years', 3)}/{cmp_cfg.get('stacked_post_years', 3)}",
            indent=1,
        )
    for exp_idx, (exposure_name, exposure_col) in enumerate(available_exposures, start=1):
        if exposure_col is None:
            continue
        exp_started = time.perf_counter()
        if _verbose(cmp_cfg):
            _log(f"stacked_did exposure {exp_idx}/{len(available_exposures)}: {exposure_name} ({exposure_col})", indent=1)
        event_df = detect_largest_jump_events(
            panel,
            exposure_col=exposure_col,
            cohort_min_year=int(cmp_cfg.get("stacked_min_cohort_year", 2013)),
            cohort_max_year=int(cmp_cfg.get("stacked_max_cohort_year", 2016)),
            min_jump=float(cmp_cfg.get("stacked_min_event_jump", 0.0)),
        )
        if _verbose(cmp_cfg):
            n_treated = int(event_df.get("treated", pd.Series(dtype=int)).fillna(0).sum()) if not event_df.empty else 0
            _log(f"detected treated events: {n_treated:,} of {len(event_df):,} units", indent=2)
        for matching_style in ("unmatched", "matched"):
            match_started = time.perf_counter()
            if _verbose(cmp_cfg):
                _log(f"stacked_did matching style: {matching_style}", indent=2)
            stacked = build_comparison_stacked_panel(
                panel,
                event_df,
                cmp_cfg,
                matching_style=matching_style,
            )
            if stacked.empty:
                if _verbose(cmp_cfg):
                    _log("skipped: stacked panel is empty", indent=3)
                continue
            if _verbose(cmp_cfg):
                _log(f"stacked panel: {_panel_summary_text(stacked)} | stacks={stacked['stack_id'].nunique():,}", indent=3)
            spec = f"{matching_style}_{exposure_name}"
            rows.extend(estimate_stacked_event_coefficients(stacked, "sun_abraham", spec, cmp_cfg))
            rows.extend(estimate_stacked_event_coefficients(stacked, "twfe", f"{spec}_twfe", cmp_cfg))
            plot_stacked_raw_means(stacked, spec, cmp_cfg, figures_dir)
            if _verbose(cmp_cfg):
                _log(f"finished {matching_style} in {_fmt_seconds(time.perf_counter() - match_started)}", indent=3)
        if _verbose(cmp_cfg):
            _log(f"finished exposure {exposure_name} in {_fmt_seconds(time.perf_counter() - exp_started)}", indent=2)
    out = pd.DataFrame(rows)
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
    x_col = _x_col(panel)
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
    x_col = _x_col(work)
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
    rows.extend(
        _estimate_multi_term_event_from_base(
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
    )
    for model_index, (outcome, y_lhs) in enumerate(outcome_lhs, start=2):
        rows.extend(
            _estimate_multi_term_event_from_base(
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
        )
    return rows


def estimate_stacked_event_coefficients(
    stacked_panel: pd.DataFrame,
    estimator_label: str,
    spec: str,
    cmp_cfg: dict,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    event_times = sorted(int(v) for v in pd.to_numeric(stacked_panel["rel_time"], errors="coerce").dropna().unique())
    ref_event_time = int(cmp_cfg.get("stacked_ref_event_time", -1))
    non_ref = [v for v in event_times if v != ref_event_time]
    if not non_ref:
        return rows
    work = stacked_panel.copy()
    terms: list[tuple[int, str]] = []
    for rel_time in non_ref:
        col = f"stack_treated_{_lag_suffix(rel_time)}"
        work[col] = (work["treated"].eq(1) & work["rel_time"].eq(rel_time)).astype(float)
        terms.append((rel_time, col))
    x_col = _x_col(work)
    lhs, fs_estimator = _first_stage_lhs(work, pd.to_numeric(work[x_col], errors="coerce"), cmp_cfg)
    fe_cols = ["unit_stack_fe", "year_stack_fe"] if estimator_label == "sun_abraham" else [_unit_id_col(cmp_cfg), "t"]
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
            cluster_col="stack_id",
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
                cluster_col="stack_id",
            )
        )
    return rows


def detect_largest_jump_events(
    panel: pd.DataFrame,
    *,
    exposure_col: str,
    cohort_min_year: int,
    cohort_max_year: int,
    min_jump: float,
) -> pd.DataFrame:
    unit_col = "c"
    work = panel[[unit_col, "t", exposure_col]].dropna(subset=[unit_col, "t"]).copy()
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work = work.dropna(subset=["t"])
    if work.empty:
        return pd.DataFrame(columns=["c", "g", "largest_jump", "treated"])
    work["t"] = work["t"].astype(int)
    work[exposure_col] = pd.to_numeric(work[exposure_col], errors="coerce").fillna(0.0)
    work = work.sort_values([unit_col, "t"])
    work["jump"] = work[exposure_col] - work.groupby(unit_col, sort=False)[exposure_col].shift(1)

    max_jump = (
        work.groupby(unit_col, as_index=False)["jump"]
        .max()
        .rename(columns={"jump": "largest_jump"})
    )
    max_jump["largest_jump"] = pd.to_numeric(max_jump["largest_jump"], errors="coerce").fillna(0.0)

    threshold = float(min_jump)
    in_window = work["t"].between(int(cohort_min_year), int(cohort_max_year))
    if threshold <= 0:
        passes_threshold = work["jump"].gt(0.0)
    else:
        passes_threshold = work["jump"].ge(threshold)
    candidates = work.loc[in_window & passes_threshold, [unit_col, "t", "jump"]].copy()
    if candidates.empty:
        out = max_jump.copy()
        out["g"] = np.nan
        out["treated"] = 0
        return out[["c", "g", "largest_jump", "treated"]]

    events = (
        candidates.sort_values([unit_col, "jump", "t"], ascending=[True, False, True])
        .drop_duplicates(unit_col, keep="first")
        .rename(columns={"t": "g"})
    )
    out = max_jump.merge(events[[unit_col, "g"]], on=unit_col, how="left")
    out["treated"] = out["g"].notna().astype(int)
    return out[["c", "g", "largest_jump", "treated"]]


def build_comparison_stacked_panel(
    panel: pd.DataFrame,
    event_df: pd.DataFrame,
    cmp_cfg: dict,
    *,
    matching_style: str,
) -> pd.DataFrame:
    pre_window = int(cmp_cfg.get("stacked_pre_years", 3))
    post_window = int(cmp_cfg.get("stacked_post_years", 3))
    treated = event_df.loc[event_df["treated"].eq(1)].dropna(subset=["g"]).copy()
    controls = event_df.loc[event_df["treated"].eq(0), ["c"]].copy()
    if treated.empty or controls.empty:
        return pd.DataFrame()
    pair_df = _build_pair_frame(panel, treated, controls, matching_style=matching_style)
    if pair_df.empty:
        return pd.DataFrame()
    matched_panel = _materialize_matched_panel(panel, pair_df)
    stacked, _ = build_stacked_did_panel(
        matched_panel,
        pair_df,
        treated[["c", "g"]].copy(),
        pre_window=pre_window,
        post_window=post_window,
    )
    return stacked


def filter_analysis_sample(panel: pd.DataFrame, cmp_cfg: dict) -> pd.DataFrame:
    work = _standardize_panel_ids(panel)
    if work.empty:
        return work
    t = pd.to_numeric(work["t"], errors="coerce")
    work = work.loc[t.between(int(cmp_cfg.get("data_min_t", 2010)), int(cmp_cfg.get("data_max_t", 2022)))].copy()
    size_col = _first_present(work, ["firm_size_annual_pre_level", "headcount_size_baseline", "total_headcount_annual_pre_level"])
    if size_col is not None:
        work[size_col] = pd.to_numeric(work[size_col], errors="coerce")
        work = work.loc[work[size_col].ge(float(cmp_cfg.get("min_pre_avg_employment", 10)))].copy()
    return work.reset_index(drop=True)


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
    candidates = ["c", "predicted_prob", "opt_probability_index", "ntile", "ntile_label"]
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
    reduced = coef_df.loc[
        coef_df["family"].eq("reduced_form") & coef_df["outcome_col"].eq(FOREIGN_NEW_HIRE_OUTCOME)
    ].copy()
    first = _pick_event_row(first)
    reduced = _pick_event_row(reduced)
    key_cols = ["design", "spec"]
    table = first[key_cols + ["coef", "se", "f_stat", "n_obs", "estimator"]].rename(
        columns={
            "coef": "first_stage_coef",
            "se": "first_stage_se",
            "n_obs": "first_stage_n",
            "estimator": "first_stage_estimator",
        }
    )
    rf = reduced[key_cols + ["coef", "se", "n_obs"]].rename(
        columns={
            "coef": "foreign_new_hire_rf_coef",
            "se": "foreign_new_hire_rf_se",
            "n_obs": "foreign_new_hire_rf_n",
        }
    )
    return table.merge(rf, on=key_cols, how="outer").sort_values(key_cols).reset_index(drop=True)


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
    panel = _regression_rows(work, [lhs, term, *(fe_cols or _fe_cols(work, cmp_cfg))])
    if panel.empty or panel[lhs].nunique(dropna=True) <= 1 or panel[term].nunique(dropna=True) <= 1:
        return _empty_result(family, design, spec, estimator, time_name, time_value, outcome_col, "insufficient variation")
    requested_fe = fe_cols or _fe_cols(panel, cmp_cfg)
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
            fit = _fit_event_panelols(panel, lhs, term_cols, term_to_time, cmp_cfg, cluster_col or _unit_id_col(cmp_cfg))
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
    for term, time_value in term_to_time.items():
        coef, se = _coef_se(fit, term)
        row = _result_base(family, design, spec, estimator, time_name, int(time_value), outcome_col, len(panel), panel[_unit_id_col(cmp_cfg)].nunique() if _unit_id_col(cmp_cfg) in panel.columns else None)
        row.update(
            {
                "coef": coef,
                "se": se,
                "f_stat": _single_term_f(coef, se),
                "fe_cols": _event_panelols_fe_label(cmp_cfg) if design == "event_study" and estimator == "ols" else "+".join(fe_cols),
            }
        )
        rows.append(row)
    return rows


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
    non_ref_years = set(int(v) for v in term_to_time.values())
    ref_candidates = [year for year in years if year not in non_ref_years]
    ref_year = ref_candidates[0] if ref_candidates else years[0]

    year_dummies = pd.get_dummies(work["_t_panelols"], prefix="yr", dtype=float)
    ref_year_col = f"yr_{ref_year}"
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
                    f"{design}: {family} {outcome}",
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
) -> None:
    if plt is None or panel.empty:
        return
    x_col = _x_col(panel)
    for spec, exposure_col in variants:
        if exposure_col not in panel.columns or x_col not in panel.columns:
            continue
        work = panel[["t", exposure_col, x_col]].dropna().copy()
        if work.empty:
            continue
        q = pd.qcut(pd.to_numeric(work[exposure_col], errors="coerce"), 4, labels=False, duplicates="drop")
        work["exposure_q"] = q
        raw = work.groupby(["t", "exposure_q"], as_index=False)[x_col].mean()
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        for qv, group in raw.groupby("exposure_q"):
            ax.plot(group["t"], group[x_col], marker="o", linewidth=1.4, label=f"Q{int(qv) + 1}")
        ax.axvline(float(cmp_cfg.get("event_year", 2016)), color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"{design} raw first stage: {spec}")
        ax.set_xlabel("Year")
        ax.set_ylabel(f"Mean {x_col}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"raw_first_stage_{design}_{_safe_name(spec)}.png", dpi=150, bbox_inches="tight")
        _display_figure_if_requested(fig, cmp_cfg)
        plt.close(fig)


def plot_stacked_raw_means(stacked: pd.DataFrame, spec: str, cmp_cfg: dict, figures_dir: Path) -> None:
    if plt is None or stacked.empty:
        return
    x_col = _x_col(stacked)
    raw = stacked.groupby(["rel_time", "treated"], as_index=False)[x_col].mean()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for treated, group in raw.groupby("treated"):
        ax.plot(group["rel_time"], group[x_col], marker="o", linewidth=1.4, label="Treated" if treated else "Control")
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_title(f"Stacked DiD raw first stage: {spec}")
    ax.set_xlabel("Event time")
    ax.set_ylabel(f"Mean {x_col}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"raw_first_stage_stacked_did_{_safe_name(spec)}.png", dpi=150, bbox_inches="tight")
    _display_figure_if_requested(fig, cmp_cfg)
    plt.close(fig)


def _plot_coef_group(group: pd.DataFrame, x_col: str, path: Path, title: str, *, show_figures: bool = False) -> None:
    if plt is None or group.empty:
        return
    work = group.dropna(subset=["coef", "se", x_col]).copy()
    if work.empty:
        return
    specs = list(work["spec"].drop_duplicates())
    offsets = np.linspace(-0.18, 0.18, max(len(specs), 1))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for idx, spec in enumerate(specs):
        sub = work.loc[work["spec"].eq(spec)].sort_values(x_col)
        x = pd.to_numeric(sub[x_col], errors="coerce") + float(offsets[idx])
        ax.errorbar(x, sub["coef"], yerr=1.96 * sub["se"], marker="o", linestyle="-", capsize=3, linewidth=1.3, label=str(spec))
    ax.axhline(0, color="black", linewidth=1)
    if 0 in set(pd.to_numeric(work[x_col], errors="coerce").dropna().astype(int)):
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Coefficient on normalized exposure")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    _display_figure_if_requested(fig, {"show_figures": show_figures})
    plt.close(fig)


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
        _x_col(work),
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
    col = "first_stage_lhs_ols_binary"
    work[col] = (pd.to_numeric(values, errors="coerce").fillna(0.0) > 0).astype(float)
    return col, "ols"


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
    if bool(cmp_cfg.get("use_log_outcome", True)):
        return vals.apply(lambda v: math.log1p(max(float(v), 0.0)) if pd.notna(v) else np.nan)
    return vals


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


def _build_pair_frame(panel: pd.DataFrame, treated: pd.DataFrame, controls: pd.DataFrame, *, matching_style: str) -> pd.DataFrame:
    control_ids = list(pd.to_numeric(controls["c"], errors="coerce").dropna().astype(int).unique())
    if not control_ids:
        return pd.DataFrame()
    features = panel.drop_duplicates("c").copy()
    rows = []
    used_controls: set[int] = set()
    pair_id = 0
    for _, row in treated.iterrows():
        treated_c = int(row["c"])
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
        rows.append({"pair_id": pair_id, "treated_c": treated_c, "control_c": int(control_c)})
        pair_id += 1
    return pd.DataFrame(rows)


def _nearest_control(features: pd.DataFrame, treated_c: int, control_ids: list[int], used_controls: set[int]) -> Optional[int]:
    t = features.loc[features["c"].eq(treated_c)]
    if t.empty:
        return None
    trow = t.iloc[0]
    controls = features.loc[features["c"].isin([c for c in control_ids if c not in used_controls])].copy()
    if controls.empty:
        return None
    if "naics2" in controls.columns and "naics2" in trow:
        exact = controls.loc[controls["naics2"].astype(str).eq(str(trow.get("naics2")))].copy()
        if not exact.empty:
            controls = exact
    size_col = _first_present(controls, ["firm_size_annual_pre_level", "headcount_size_baseline", "total_headcount_annual_pre_level"])
    growth_col = _first_present(controls, ["firm_size_annual_pre_growth", "headcount_growth_asinh"])
    score = pd.Series(0.0, index=controls.index)
    if size_col is not None:
        scale = pd.to_numeric(features[size_col], errors="coerce").std(skipna=True) or 1.0
        score += (pd.to_numeric(controls[size_col], errors="coerce") - float(trow.get(size_col, 0.0))).abs() / scale
    if growth_col is not None:
        scale = pd.to_numeric(features[growth_col], errors="coerce").std(skipna=True) or 1.0
        score += (pd.to_numeric(controls[growth_col], errors="coerce") - float(trow.get(growth_col, 0.0))).abs() / scale
    return int(controls.loc[score.sort_values().index[0], "c"])


def _materialize_matched_panel(panel: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, pair in pair_df.iterrows():
        t_panel = panel.loc[panel["c"].eq(int(pair["treated_c"]))].copy()
        c_panel = panel.loc[panel["c"].eq(int(pair["control_c"]))].copy()
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


def _single_term_f(coef: Optional[float], se: Optional[float]) -> Optional[float]:
    if coef is None or se is None or not se or not np.isfinite(se):
        return None
    return float((coef / se) ** 2)


def _empty_result(family: str, design: str, spec: str, estimator: str, time_name: str, time_value: int, outcome_col: str, error: str) -> dict[str, object]:
    row = _result_base(family, design, spec, estimator, time_name, time_value, outcome_col, None, None)
    row["error"] = error
    return row


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
    if "horizon" in work.columns:
        work["_event_distance"] = pd.to_numeric(work["horizon"], errors="coerce").abs()
    elif "rel_time" in work.columns:
        work["_event_distance"] = pd.to_numeric(work["rel_time"], errors="coerce").abs()
    elif "year" in work.columns:
        work["_event_distance"] = (pd.to_numeric(work["year"], errors="coerce") - 2016).abs()
    else:
        work["_event_distance"] = 0
    return work.sort_values(["design", "spec", "_event_distance"]).drop_duplicates(["design", "spec"])


def _validate_comparison_config(cmp_cfg: dict) -> None:
    unit = str(cmp_cfg.get("unit", "firm"))
    if unit not in UNITS:
        raise ValueError(f"design_comparison.unit must be one of {sorted(UNITS)}")
    first_stage_type = str(cmp_cfg.get("first_stage_type", "ppml"))
    if first_stage_type not in FIRST_STAGE_TYPES:
        raise ValueError(f"design_comparison.first_stage_type must be one of {sorted(FIRST_STAGE_TYPES)}")


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


def _x_col(panel: pd.DataFrame) -> str:
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
        out_dir=parsed.out_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
