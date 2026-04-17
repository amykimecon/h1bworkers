from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import root  # noqa: E402

DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "pipeline.yaml"
ENV_CONFIG_PATH = os.environ.get("F1_INDIV_PIPELINE_CONFIG")
ACTIVE_CONFIG_PATH: str | None = None
_VAR_PATTERN = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


def _replace_known_vars(value: str) -> str:
    return value.replace("{root}", str(root))


def _lookup_var(data: dict[str, Any], key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _interpolate(value: str, data: dict[str, Any]) -> str:
    def _sub(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        if not key:
            return match.group(0)
        resolved = _lookup_var(data, key)
        return str(resolved) if resolved is not None else match.group(0)

    return _VAR_PATTERN.sub(_sub, value)


def _walk_and_replace(obj: Any, data: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {key: _walk_and_replace(value, data) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_replace(value, data) for value in obj]
    if isinstance(obj, str):
        expanded = os.path.expanduser(_replace_known_vars(obj))
        return _interpolate(expanded, data)
    return obj


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    global ACTIVE_CONFIG_PATH

    cfg_path = Path(path) if path else Path(ENV_CONFIG_PATH) if ENV_CONFIG_PATH else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {cfg_path}")
    ACTIVE_CONFIG_PATH = str(cfg_path.resolve())
    data = yaml.safe_load(cfg_path.read_text()) or {}
    data = _walk_and_replace(data, data)
    return _walk_and_replace(data, data)


def get_stage_config(cfg: dict[str, Any], stage_name: str) -> dict[str, Any]:
    stages = cfg.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("Config key 'stages' must be a mapping.")
    stage_cfg = stages.get(stage_name, {})
    if stage_cfg is None:
        return {}
    if not isinstance(stage_cfg, dict):
        raise ValueError(f"Stage config for '{stage_name}' must be a mapping.")
    return stage_cfg
