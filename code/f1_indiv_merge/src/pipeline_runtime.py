from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable

from src.config_loader import get_stage_config

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
STAGE_ORDER = [
    "01_f1_foia_clean",
    "02_rev_import",
    "03_rev_crosswalks",
    "04_rev_user_clean",
    "05_indiv_merge",
]


class StageDeferredError(RuntimeError):
    """Raised when a scaffolded stage is present but not implemented yet."""


def ensure_pythonpath() -> None:
    for path in (PIPELINE_ROOT, REPO_ROOT):
        as_str = str(path)
        if as_str not in sys.path:
            sys.path.insert(0, as_str)


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def sanitize_ipykernel_argv(argv: list[str] | None = None) -> list[str]:
    """Strip Jupyter/IPython kernel launcher args while preserving normal CLI strictness."""
    source = list(sys.argv[1:] if argv is None else argv)
    cleaned: list[str] = []
    skip_next = False

    for arg in source:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-f", "--f"}:
            skip_next = True
            continue
        if arg.startswith("-f=") or arg.startswith("--f="):
            continue
        cleaned.append(arg)

    return cleaned


def get_stage_module_path(pipeline_cfg: dict[str, Any], stage_name: str) -> Path:
    stage_cfg = get_stage_config(pipeline_cfg, stage_name)
    module_rel = stage_cfg.get("module_path", f"{stage_name}/stage_main.py")
    return PIPELINE_ROOT / module_rel


def load_stage_module(stage_name: str, pipeline_cfg: dict[str, Any]):
    ensure_pythonpath()
    module_path = get_stage_module_path(pipeline_cfg, stage_name)
    if not module_path.exists():
        raise FileNotFoundError(f"Stage module not found for {stage_name}: {module_path}")

    module_name = f"f1_indiv_pipeline_{stage_name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_stage_list(
    pipeline_cfg: dict[str, Any],
    requested_stages: Iterable[str] | None = None,
) -> list[str]:
    if requested_stages:
        requested = list(dict.fromkeys(requested_stages))
        invalid = [stage for stage in requested if stage not in STAGE_ORDER]
        if invalid:
            valid = ", ".join(STAGE_ORDER)
            raise ValueError(f"Unknown stage(s): {', '.join(invalid)}. Valid stages: {valid}")
        return [stage for stage in STAGE_ORDER if stage in requested]

    build_cfg = pipeline_cfg.get("build", {})
    default_stages = build_cfg.get("default_stages") or ["05_indiv_merge"]
    if not isinstance(default_stages, list):
        raise ValueError("Config key 'build.default_stages' must be a list.")
    invalid = [stage for stage in default_stages if stage not in STAGE_ORDER]
    if invalid:
        valid = ", ".join(STAGE_ORDER)
        raise ValueError(
            f"Invalid entries in build.default_stages: {', '.join(invalid)}. Valid stages: {valid}"
        )
    return [stage for stage in STAGE_ORDER if stage in default_stages]
