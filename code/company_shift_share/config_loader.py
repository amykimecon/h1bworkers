from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import root  # noqa: E402

DEFAULT_CONFIG_PATH = Path(root) / "h1bworkers" / "code" / "configs" / "company_shift_share.yaml"
REPO_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "company_shift_share.yaml"
_VAR_PATTERN = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


def _replace_root(value: str) -> str:
    return value.replace("{root}", str(root))


def _lookup_var(data: dict[str, Any], key: str) -> Any:
    if "." in key:
        current: Any = data
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current
    return data.get(key)


def _interpolate(value: str, data: dict[str, Any]) -> str:
    def _sub(match: re.Match) -> str:
        key = match.group(1) or match.group(2)
        if not key:
            return match.group(0)
        resolved = _lookup_var(data, key)
        return str(resolved) if resolved is not None else match.group(0)

    return _VAR_PATTERN.sub(_sub, value)


def _walk_and_replace(obj: Any, data: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_and_replace(v, data) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_replace(v, data) for v in obj]
    if isinstance(obj, str):
        expanded = os.path.expanduser(_replace_root(obj))
        return _interpolate(expanded, data)
    return obj


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists() and path is None and REPO_CONFIG_PATH.exists():
        cfg_path = REPO_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text()) or {}
    # Two-pass interpolation to allow ${int_dir} defined in the same file.
    data = _walk_and_replace(data, data)
    return _walk_and_replace(data, data)


def get_cfg_section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    section = cfg.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return section
