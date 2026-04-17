from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ARTIFACT_KEYS = (
    "ipeds_main_institutions",
    "revelio_ipeds_edges",
    "revelio_ipeds_best",
    "ipeds_to_revelio_families",
    "f1_ipeds_edges",
    "f1_revelio_ipeds_resolution",
    "f1_revelio_school_crosswalk",
)


def _resolve_repo_root(repo_root: str | None) -> Path:
    if repo_root:
        return Path(repo_root).expanduser().resolve()
    return Path.home() / "revelio-cleaning"


def _required_artifacts(artifact_paths: dict[str, str]) -> dict[str, Path]:
    missing_keys = [key for key in ARTIFACT_KEYS if not artifact_paths.get(key)]
    if missing_keys:
        raise ValueError(f"Missing required school-matching artifact paths: {', '.join(missing_keys)}")
    return {key: Path(artifact_paths[key]).expanduser().resolve() for key in ARTIFACT_KEYS}


def stage_external_us_school_matching_artifacts(
    *,
    artifact_paths: dict[str, str],
    revelio_cleaning_repo_root: str | None = None,
    verbose: bool = False,
) -> dict[str, Path]:
    resolved_artifacts = _required_artifacts(artifact_paths)
    missing_outputs = [str(path) for path in resolved_artifacts.values() if not path.exists()]
    if missing_outputs:
        repo_root = _resolve_repo_root(revelio_cleaning_repo_root)
        build_hint = repo_root / "scripts" / "us_school_matching" / "build_us_school_crosswalk.py"
        raise FileNotFoundError(
            "Missing external US school-matching artifacts: "
            + ", ".join(missing_outputs)
            + f". Build them from revelio-cleaning first, e.g. with {build_hint}."
        )
    if verbose:
        print(f"Staging external US school-matching artifacts from {resolved_artifacts['f1_revelio_school_crosswalk'].parent}")
    return resolved_artifacts


def run_external_us_school_matching(
    *,
    revelio_cleaning_repo_root: str | None,
    config_path: str | None,
    raw_educations_parquet: str | None,
    ipeds_crosswalk_parquet: str,
    f1_inst_unitid_crosswalk_parquet: str,
    revelio_ipeds_inst_crosswalk_parquet: str,
    artifact_paths: dict[str, str],
    python_executable: str | None = None,
    overwrite: bool = True,
    testing: bool = False,
    verbose: bool = False,
) -> dict[str, Path]:
    repo_root = _resolve_repo_root(revelio_cleaning_repo_root)
    script_path = repo_root / "scripts" / "us_school_matching" / "build_us_school_crosswalk.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing external school-matching script: {script_path}")

    resolved_artifacts = _required_artifacts(artifact_paths)
    python_bin = python_executable or sys.executable

    cmd = [python_bin, str(script_path)]
    if config_path:
        cmd.extend(["--config", str(Path(config_path).expanduser().resolve())])
    if raw_educations_parquet:
        cmd.extend(["--raw-educations-parquet", str(Path(raw_educations_parquet).expanduser().resolve())])
    cmd.extend(["--ipeds-crosswalk-parquet", str(Path(ipeds_crosswalk_parquet).expanduser().resolve())])
    cmd.extend(["--f1-inst-unitid-crosswalk-parquet", str(Path(f1_inst_unitid_crosswalk_parquet).expanduser().resolve())])
    cmd.extend(["--revelio-ipeds-inst-crosswalk-parquet", str(Path(revelio_ipeds_inst_crosswalk_parquet).expanduser().resolve())])
    cmd.extend(["--ipeds-main-institutions-parquet", str(resolved_artifacts["ipeds_main_institutions"])])
    cmd.extend(["--revelio-ipeds-edges-parquet", str(resolved_artifacts["revelio_ipeds_edges"])])
    cmd.extend(["--revelio-ipeds-best-parquet", str(resolved_artifacts["revelio_ipeds_best"])])
    cmd.extend(["--ipeds-to-revelio-families-parquet", str(resolved_artifacts["ipeds_to_revelio_families"])])
    cmd.extend(["--f1-ipeds-edges-parquet", str(resolved_artifacts["f1_ipeds_edges"])])
    cmd.extend(["--f1-revelio-ipeds-resolution-parquet", str(resolved_artifacts["f1_revelio_ipeds_resolution"])])
    cmd.extend(["--f1-revelio-school-crosswalk-parquet", str(resolved_artifacts["f1_revelio_school_crosswalk"])])
    if overwrite:
        cmd.append("--overwrite")
    else:
        cmd.append("--no-overwrite")
    if testing:
        cmd.append("--testing")
    if verbose:
        print(f"Running external school matching: {' '.join(cmd)}")

    child_env = os.environ.copy()
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child_env.setdefault("data", str(Path.home() / "data"))
    child_env.setdefault("DATA", child_env["data"])
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    repo_src = str(repo_root / "src")
    child_env["PYTHONPATH"] = f"{repo_src}:{existing_pythonpath}" if existing_pythonpath else repo_src

    subprocess.run(cmd, check=True, cwd=str(repo_root), env=child_env)

    missing_outputs = [str(path) for path in resolved_artifacts.values() if not path.exists()]
    if missing_outputs:
        raise FileNotFoundError(
            "External school-matching build completed but some artifacts are missing: "
            + ", ".join(missing_outputs)
        )
    return resolved_artifacts
