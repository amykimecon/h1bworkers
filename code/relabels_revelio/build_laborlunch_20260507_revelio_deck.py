#!/usr/bin/env python
"""Populate the Revelio figures used in slides_laborlunch_20260507 and rebuild the PDF.

This wrapper runs four relabel_indiv_analysis jobs:
1. Main deck figures: econ_v2, no controls, main config output dir.
2. Always-STEM button figures: generalized full sample, always-STEM controls.
3. Appendix controlled figures: econ_v2, slide-controls output dir.
4. Appendix no-control figures: econ_v2, slide-nocontrols output dir.

It then rebuilds the slide deck with pdflatex twice.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


HOME = Path.home()
CODE_ROOT = HOME / "h1bworkers" / "code"
TMP_ROOT = HOME / ".tmp"

DEFAULT_GENERALIZED_PANEL = CODE_ROOT / "output" / "relabel_indiv" / "generalized_relabels_panel.parquet"
PREFERRED_REVELIO_HORIZONS = [0, 1, 2, 3, 4, 5]
REVELIO_POOLED_POST_EVENT_MIN = -1
REVELIO_POOLED_POST_EVENT_MAX = 3
REVELIO_MAIN_DID_SAMPLE_MODES = {"econ_only", "full_sample"}
REVELIO_CONTROL_GROUPS = {"never_treated", "always_stem", "late_treated"}
DEFAULT_MAIN_CONFIG = CODE_ROOT / "configs" / "relabel_indiv.yaml"
DEFAULT_CONTROLS_TEMPLATE = TMP_ROOT / "relabel_indiv_slides_controls.yaml"
DEFAULT_ECON_TEMPLATE = TMP_ROOT / "relabel_indiv_slides_nocontrols.yaml"
DEFAULT_TEX = HOME / "writing" / "slides" / "slides_laborlunch_20260507" / "slides_laborlunch_20260507.tex"
DEFAULT_ALWAYS_STEM_OUTPUT_DIR = HOME / "output" / "relabel_indiv_always_stem"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config template not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} is not a YAML mapping")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _set_nested(cfg: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur = cfg
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def _materialize_configs(
    *,
    generalized_panel: Path,
    main_template: Path,
    controls_template: Path,
    econ_template: Path,
    main_did_sample: str,
    control_group: str,
    always_stem_output_dir: Path,
) -> dict[str, Path]:
    main_cfg = _load_yaml(main_template)
    controls_cfg = _load_yaml(controls_template)
    econ_cfg = _load_yaml(econ_template)
    always_stem_cfg = _load_yaml(main_template)

    event_source_mode = "generalized_final_sample" if main_did_sample == "full_sample" else "econ_v2"
    sample_cip_prefixes = [] if event_source_mode == "generalized_final_sample" else ["4506"]
    for cfg in (main_cfg, controls_cfg, econ_cfg):
        _set_nested(cfg, ("build", "event_source_mode"), event_source_mode)
        _set_nested(cfg, ("build", "control_group"), control_group)
        _set_nested(cfg, ("build", "outcome_horizons"), PREFERRED_REVELIO_HORIZONS)
        _set_nested(cfg, ("build", "did_plot_mode"), "pooled_post_by_horizon")
        _set_nested(cfg, ("build", "pooled_post_event_min"), REVELIO_POOLED_POST_EVENT_MIN)
        _set_nested(cfg, ("build", "pooled_post_event_max"), REVELIO_POOLED_POST_EVENT_MAX)
        _set_nested(cfg, ("build", "sample_cip_prefixes"), sample_cip_prefixes)
        _set_nested(
            cfg,
            ("paths", "generalized_relabels_panel_parquet"),
            str(generalized_panel),
        )

    _set_nested(always_stem_cfg, ("build", "event_source_mode"), "generalized_final_sample")
    _set_nested(always_stem_cfg, ("build", "control_group"), "always_stem")
    _set_nested(always_stem_cfg, ("build", "outcome_horizons"), PREFERRED_REVELIO_HORIZONS)
    _set_nested(always_stem_cfg, ("build", "did_plot_mode"), "pooled_post_by_horizon")
    _set_nested(always_stem_cfg, ("build", "pooled_post_event_min"), REVELIO_POOLED_POST_EVENT_MIN)
    _set_nested(always_stem_cfg, ("build", "pooled_post_event_max"), REVELIO_POOLED_POST_EVENT_MAX)
    _set_nested(always_stem_cfg, ("build", "sample_cip_prefixes"), [])
    _set_nested(
        always_stem_cfg,
        ("paths", "generalized_relabels_panel_parquet"),
        str(generalized_panel),
    )
    _set_nested(always_stem_cfg, ("paths", "output_dir"), str(always_stem_output_dir))
    _set_nested(
        always_stem_cfg,
        ("paths", "output_panel_parquet"),
        str(HOME / "data" / "int" / "relabel_indiv_panel_laborlunch_20260507_always_stem.parquet"),
    )
    _set_nested(
        always_stem_cfg,
        ("paths", "output_did_results_parquet"),
        str(HOME / "data" / "int" / "relabel_indiv_did_laborlunch_20260507_always_stem.parquet"),
    )

    out_paths = {
        "main_econ": TMP_ROOT / "relabel_indiv_laborlunch_20260507_main_econ.yaml",
        "controls_econ": TMP_ROOT / "relabel_indiv_laborlunch_20260507_controls_econ.yaml",
        "econ_appendix": TMP_ROOT / "relabel_indiv_laborlunch_20260507_econ_appendix.yaml",
        "always_stem": TMP_ROOT / "relabel_indiv_laborlunch_20260507_always_stem.yaml",
    }
    _write_yaml(out_paths["main_econ"], main_cfg)
    _write_yaml(out_paths["controls_econ"], controls_cfg)
    _write_yaml(out_paths["econ_appendix"], econ_cfg)
    _write_yaml(out_paths["always_stem"], always_stem_cfg)
    return out_paths


def _run_analysis(config_path: Path) -> None:
    env = os.environ.copy()
    env["RELABEL_INDIV_CONFIG"] = str(config_path)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(CODE_ROOT) if not existing_pythonpath else f"{CODE_ROOT}:{existing_pythonpath}"
    )

    cmd = [sys.executable, "-m", "relabels_revelio.relabel_indiv_analysis"]
    print(f"\n==> Running relabel_indiv_analysis with {config_path}")
    subprocess.run(cmd, cwd=CODE_ROOT, env=env, check=True)


def _run_pdflatex(tex_path: Path) -> None:
    if not tex_path.exists():
        raise FileNotFoundError(f"Slide deck not found: {tex_path}")

    deck_dir = tex_path.parent
    tex_name = tex_path.name
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name]
    print(f"\n==> Building slide deck in {deck_dir}")
    subprocess.run(cmd, cwd=deck_dir, check=True)
    subprocess.run(cmd, cwd=deck_dir, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate Revelio figures for slides_laborlunch_20260507 and rebuild the deck."
    )
    parser.add_argument(
        "--generalized-panel",
        type=Path,
        default=DEFAULT_GENERALIZED_PANEL,
        help="Path to generalized_relabels_panel.parquet",
    )
    parser.add_argument(
        "--main-config-template",
        type=Path,
        default=DEFAULT_MAIN_CONFIG,
        help="Base config for the main generalized run",
    )
    parser.add_argument(
        "--controls-config-template",
        type=Path,
        default=DEFAULT_CONTROLS_TEMPLATE,
        help="Template config for the generalized controls appendix run",
    )
    parser.add_argument(
        "--econ-config-template",
        type=Path,
        default=DEFAULT_ECON_TEMPLATE,
        help="Template config for the econ-only appendix run",
    )
    parser.add_argument(
        "--main-did-sample",
        choices=sorted(REVELIO_MAIN_DID_SAMPLE_MODES),
        default="econ_only",
        help="Use econ-only relabel events or the finalized generalized full sample for Revelio DiD runs.",
    )
    parser.add_argument(
        "--control-group",
        choices=sorted(REVELIO_CONTROL_GROUPS),
        default="never_treated",
        help="Control group for generalized full-sample Revelio runs.",
    )
    parser.add_argument(
        "--always-stem-output-dir",
        type=Path,
        default=DEFAULT_ALWAYS_STEM_OUTPUT_DIR,
        help="Output directory for always-STEM Revelio button figures.",
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=DEFAULT_TEX,
        help="Path to slides_laborlunch_20260507.tex",
    )
    parser.add_argument(
        "--skip-tex",
        action="store_true",
        help="Populate figures only; do not rebuild the slide deck",
    )
    args = parser.parse_args()
    if args.main_did_sample != "full_sample" and args.control_group != "never_treated":
        parser.error("--control-group other than never_treated requires --main-did-sample full_sample")
    return args


def main() -> None:
    args = parse_args()
    if not args.generalized_panel.exists():
        raise FileNotFoundError(f"Generalized panel not found: {args.generalized_panel}")

    configs = _materialize_configs(
        generalized_panel=args.generalized_panel,
        main_template=args.main_config_template,
        controls_template=args.controls_config_template,
        econ_template=args.econ_config_template,
        main_did_sample=args.main_did_sample,
        control_group=args.control_group,
        always_stem_output_dir=args.always_stem_output_dir,
    )

    print("Generated run configs:")
    for key, path in configs.items():
        print(f"  {key}: {path}")

    _run_analysis(configs["main_econ"])
    _run_analysis(configs["always_stem"])
    _run_analysis(configs["controls_econ"])
    _run_analysis(configs["econ_appendix"])

    if not args.skip_tex:
        _run_pdflatex(args.tex)

    print("\nDone.")


if __name__ == "__main__":
    main()
