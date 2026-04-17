from __future__ import annotations

import argparse
import json
import math
import sys
from builtins import print as _print
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = STAGE_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
for _path in (STAGE_DIR, PIPELINE_ROOT, REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from common import (  # noqa: E402
    atomic_duckdb_copy_to_parquet,
    atomic_write_json,
    atomic_write_parquet,
    clear_dir,
    ensure_parent_dir,
    escape_sql_literal,
    get_duckdb_connection,
    mock_female_probability,
    mock_region_probabilities,
    resolve_user_shard_spec,
    resolve_existing_path,
    stage_clean_name_artifacts,
)
from src.config_loader import get_stage_config, load_config  # noqa: E402
from src.pipeline_runtime import coerce_bool, sanitize_ipykernel_argv  # noqa: E402

print = partial(_print, flush=True)

STAGE_NAME = "04_rev_user_clean"


def _resolve_name_source_path(cfg: dict[str, Any], stage_cfg: dict[str, Any]) -> str:
    stage02_cfg = cfg.get("stages", {}).get("02_rev_import", {})
    paths_cfg = cfg.get("paths", {})
    resolved = resolve_existing_path(
        stage_cfg.get("name_source_parquet"),
        stage_cfg.get("wrds_users_input_parquet"),
        stage02_cfg.get("wrds_users_parquet"),
        stage_cfg.get("legacy_rev_indiv_parquet"),
        paths_cfg.get("legacy_rev_indiv_parquet"),
    )
    if resolved is None:
        raise FileNotFoundError(
            f"{STAGE_NAME} could not resolve any source parquet for the local nametrace step."
        )
    return resolved


def _list_chunk_files(chunk_dir: Path, stem: str) -> list[Path]:
    return sorted(chunk_dir.glob(f"{stem}_*.parquet"))


def _count_parquet_rows(path: str | Path) -> int:
    con = get_duckdb_connection()
    return int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{escape_sql_literal(path)}')").fetchone()[0])


def _count_rows_from_paths(paths: list[Path]) -> int:
    if not paths:
        return 0
    con = get_duckdb_connection()
    return int(con.execute("SELECT COUNT(*) FROM read_parquet(?)", [[str(path) for path in paths]]).fetchone()[0])


def _load_progress_state(
    progress_path: Path,
    *,
    total_values: int,
    scoring_chunk_size: int,
) -> dict[str, Any]:
    state = {
        "status": "running",
        "total_values": int(total_values),
        "scoring_chunk_size": int(scoring_chunk_size),
        "completed_chunks": 0,
        "completed_rows": 0,
        "last_completed_chunk": None,
    }
    if not progress_path.exists():
        return state
    try:
        loaded = json.loads(progress_path.read_text())
    except json.JSONDecodeError:
        return state
    if (
        int(loaded.get("total_values", -1)) != int(total_values)
        or int(loaded.get("scoring_chunk_size", -1)) != int(scoring_chunk_size)
    ):
        return state
    state.update(loaded)
    return state


def _parse_female_prob(prediction: Any) -> float | None:
    if isinstance(prediction, dict):
        gender = prediction.get("gender")
    else:
        gender = None
    if isinstance(gender, list):
        for item in gender:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] == "female":
                try:
                    return float(item[1])
                except (TypeError, ValueError):
                    return None
    return None


def _parse_region_probs(prediction: Any) -> list[tuple[str, float]]:
    if isinstance(prediction, dict):
        region_probs = prediction.get("subregion")
    else:
        region_probs = None
    out: list[tuple[str, float]] = []
    if isinstance(region_probs, list):
        for item in region_probs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    out.append((str(item[0]), float(item[1])))
                except (TypeError, ValueError):
                    continue
    return out


def _predict_batch(
    *,
    values: list[str],
    top_k: int,
    use_mock: bool,
    tracer,
) -> list[dict[str, Any]]:
    if not values:
        return []

    if use_mock or tracer is None:
        return [
            {
                "f_prob_nt": mock_female_probability(value),
                "region_probs": mock_region_probabilities(value)[:top_k],
            }
            for value in values
        ]

    predictions = tracer.predict(values, batch_size=max(1, min(len(values), 100000)), topk=top_k)
    out: list[dict[str, Any]] = []
    for prediction in predictions:
        out.append(
            {
                "f_prob_nt": _parse_female_prob(prediction),
                "region_probs": _parse_region_probs(prediction),
            }
        )
    return out


def _merge_chunk_dir_to_parquet(chunk_files: list[Path], out_path: str | Path, columns: list[str]) -> None:
    out_file = ensure_parent_dir(out_path)
    if not chunk_files:
        atomic_write_parquet(pd.DataFrame(columns=columns), out_file, index=False)
        return
    con = get_duckdb_connection()
    atomic_duckdb_copy_to_parquet(
        con,
        """
        SELECT *
        FROM read_parquet(?)
        """,
        out_file,
        [[str(path) for path in chunk_files]],
    )


def run_nametrace(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    testing: bool | None = None,
    use_mock: bool | None = None,
    overwrite: bool | None = None,
) -> dict[str, Any]:
    cfg = pipeline_cfg or load_config(config_path)
    stage_cfg = get_stage_config(cfg, STAGE_NAME)
    wide_out_path = ensure_parent_dir(stage_cfg["nametrace_wide_parquet"])
    long_out_path = ensure_parent_dir(stage_cfg["nametrace_long_parquet"])
    effective_overwrite = coerce_bool(
        stage_cfg.get("overwrite"),
        coerce_bool(cfg.get("build", {}).get("overwrite"), True),
    ) if overwrite is None else bool(overwrite)
    effective_testing = coerce_bool(cfg.get("testing", {}).get("enabled"), False) if testing is None else bool(testing)
    effective_use_mock = coerce_bool(stage_cfg.get("nametrace_use_mock"), False) if use_mock is None else bool(use_mock)

    if wide_out_path.exists() and long_out_path.exists() and not effective_overwrite:
        return {
            "nametrace_wide_parquet": str(wide_out_path),
            "nametrace_long_parquet": str(long_out_path),
            "nametrace_source": "existing",
            "nametrace_wide_rows": _count_parquet_rows(wide_out_path),
            "nametrace_long_rows": _count_parquet_rows(long_out_path),
        }

    source_path = _resolve_name_source_path(cfg, stage_cfg)
    testing_max_users = stage_cfg.get("testing_max_users") if effective_testing else None
    shard_spec = resolve_user_shard_spec(stage_cfg)
    shard_label = str(shard_spec["user_shard_label"]) if shard_spec else None
    min_fullname_chars = int(stage_cfg.get("name_model_min_fullname_chars", 2))
    min_token_chars = int(stage_cfg.get("name_model_min_token_chars", 2))
    work_root = wide_out_path.parent / f"{wide_out_path.stem}_work"
    work_root.mkdir(parents=True, exist_ok=True)

    shard_suffix = f" ({shard_label})" if shard_label else ""
    print(f"[{STAGE_NAME}] staging distinct cleaned names from {source_path} for nametrace{shard_suffix}")
    name_artifacts = stage_clean_name_artifacts(
        source_path,
        artifact_dir=work_root / "staged_names",
        overwrite=effective_overwrite,
        testing_max_users=int(testing_max_users) if testing_max_users else None,
        shard_count=int(shard_spec["user_shard_count"]) if shard_spec else None,
        shard_id=int(shard_spec["user_shard_id"]) if shard_spec else None,
        min_fullname_chars=min_fullname_chars,
        min_token_chars=min_token_chars,
    )
    prediction_overwrite = effective_overwrite or bool(name_artifacts.get("artifacts_rebuilt"))
    print(
        f"[{STAGE_NAME}] nametrace scoring {name_artifacts['n_full_names']:,} unique full names "
        f"after short-name filtering"
    )

    top_k = int(stage_cfg.get("nametrace_top_k", 5))
    model_batch_size = int(stage_cfg.get("nametrace_batch_size", 5000))
    scoring_chunk_size = int(stage_cfg.get("nametrace_scoring_chunk_size", max(model_batch_size, 10000)))
    unique_parquet = name_artifacts["full_unique_parquet"]
    total_values = _count_parquet_rows(unique_parquet)
    wide_chunk_dir = work_root / "wide_chunks"
    long_chunk_dir = work_root / "long_chunks"

    if prediction_overwrite:
        clear_dir(wide_chunk_dir)
        clear_dir(long_chunk_dir)
        if wide_out_path.exists():
            wide_out_path.unlink()
        if long_out_path.exists():
            long_out_path.unlink()
    else:
        wide_chunk_dir.mkdir(parents=True, exist_ok=True)
        long_chunk_dir.mkdir(parents=True, exist_ok=True)

    tracer = None
    effective_mock = effective_use_mock
    if not effective_mock:
        try:
            from nametrace import NameTracer

            tracer = NameTracer()
        except Exception as exc:
            effective_mock = True
            print(f"[{STAGE_NAME}] WARNING: nametrace unavailable, using mock scores. {exc}")

    n_chunks = math.ceil(total_values / scoring_chunk_size) if total_values else 0
    existing_wide_chunks = _list_chunk_files(wide_chunk_dir, "wide")
    completed_rows = _count_rows_from_paths(existing_wide_chunks) if existing_wide_chunks and not prediction_overwrite else 0
    progress_path = work_root / "progress.json"
    progress_state = _load_progress_state(
        progress_path,
        total_values=total_values,
        scoring_chunk_size=scoring_chunk_size,
    )
    progress_state["completed_rows"] = max(int(progress_state.get("completed_rows", 0)), completed_rows)
    completed_chunks = int(progress_state.get("completed_chunks", 0))
    progress = tqdm(
        total=total_values,
        initial=completed_rows,
        desc=f"[{STAGE_NAME}] nametrace:{'mock' if effective_mock else 'model'}",
        unit="name",
        file=sys.stdout,
    )

    con = get_duckdb_connection()
    for chunk_index in range(n_chunks):
        wide_chunk_path = wide_chunk_dir / f"wide_{chunk_index:07d}.parquet"
        long_chunk_path = long_chunk_dir / f"long_{chunk_index:07d}.parquet"
        if wide_chunk_path.exists() and long_chunk_path.exists() and not prediction_overwrite:
            continue

        row_start = chunk_index * scoring_chunk_size + 1
        row_end = min(total_values, (chunk_index + 1) * scoring_chunk_size)
        chunk_df = con.execute(
            f"""
            SELECT rownum, fullname_clean
            FROM read_parquet('{escape_sql_literal(unique_parquet)}')
            WHERE rownum BETWEEN {row_start} AND {row_end}
            ORDER BY rownum
            """
        ).fetchdf()
        if chunk_df.empty:
            continue

        names = chunk_df["fullname_clean"].astype(str).tolist()
        predictions: list[dict[str, Any]] = []
        for batch_start in range(0, len(names), model_batch_size):
            batch = names[batch_start : batch_start + model_batch_size]
            try:
                predictions.extend(
                    _predict_batch(
                        values=batch,
                        top_k=top_k,
                        use_mock=effective_mock,
                        tracer=tracer,
                    )
                )
            except Exception as exc:
                if not effective_mock:
                    print(
                        f"[{STAGE_NAME}] WARNING: nametrace failed for chunk {chunk_index + 1}/{n_chunks}; "
                        f"using mock scores for the remaining batch. {exc}"
                    )
                effective_mock = True
                predictions.extend(
                    _predict_batch(
                        values=batch,
                        top_k=top_k,
                        use_mock=True,
                        tracer=None,
                    )
                )

        wide_rows: list[dict[str, Any]] = []
        long_rows: list[dict[str, Any]] = []
        for name, prediction in zip(names, predictions):
            region_probs = prediction["region_probs"]
            wide_rows.append(
                {
                    "fullname_clean": name,
                    "f_prob_nt": prediction["f_prob_nt"],
                    "region_probs_json": json.dumps(region_probs),
                }
            )
            if region_probs:
                for region, prob in region_probs:
                    long_rows.append(
                        {
                            "fullname_clean": name,
                            "f_prob_nt": prediction["f_prob_nt"],
                            "region": region,
                            "prob": prob,
                        }
                    )
            else:
                long_rows.append(
                    {
                        "fullname_clean": name,
                        "f_prob_nt": prediction["f_prob_nt"],
                        "region": None,
                        "prob": None,
                    }
                )

        atomic_write_parquet(pd.DataFrame(wide_rows), wide_chunk_path, index=False)
        atomic_write_parquet(pd.DataFrame(long_rows), long_chunk_path, index=False)
        completed_chunks += 1
        progress_state.update(
            {
                "status": "running",
                "completed_chunks": completed_chunks,
                "completed_rows": int(progress_state.get("completed_rows", 0)) + len(names),
                "last_completed_chunk": chunk_index,
            }
        )
        atomic_write_json(progress_path, progress_state)
        progress.update(len(names))
    progress.close()

    wide_chunk_files = _list_chunk_files(wide_chunk_dir, "wide")
    long_chunk_files = _list_chunk_files(long_chunk_dir, "long")
    _merge_chunk_dir_to_parquet(
        wide_chunk_files,
        wide_out_path,
        ["fullname_clean", "f_prob_nt", "region_probs_json"],
    )
    _merge_chunk_dir_to_parquet(
        long_chunk_files,
        long_out_path,
        ["fullname_clean", "f_prob_nt", "region", "prob"],
    )
    progress_state.update(
        {
            "status": "complete",
            "completed_chunks": completed_chunks,
            "completed_rows": total_values,
            "last_completed_chunk": n_chunks - 1 if n_chunks else None,
            "nametrace_wide_parquet": str(wide_out_path),
            "nametrace_long_parquet": str(long_out_path),
        }
    )
    atomic_write_json(progress_path, progress_state)

    return {
        "nametrace_wide_parquet": str(wide_out_path),
        "nametrace_long_parquet": str(long_out_path),
        "nametrace_source": source_path,
        "nametrace_wide_rows": _count_parquet_rows(wide_out_path),
        "nametrace_long_rows": _count_parquet_rows(long_out_path),
        **(
            {
                "user_shard_count": int(shard_spec["user_shard_count"]),
                "user_shard_id": int(shard_spec["user_shard_id"]),
                "user_shard_label": shard_label,
            }
            if shard_spec
            else {}
        ),
        **name_artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local stage-04 nametrace step.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    testing_group = parser.add_mutually_exclusive_group()
    testing_group.add_argument("--testing", dest="testing", action="store_true")
    testing_group.add_argument("--no-testing", dest="testing", action="store_false")
    parser.set_defaults(testing=None)
    args = parser.parse_args(sanitize_ipykernel_argv())

    stats = run_nametrace(
        config_path=args.config,
        testing=args.testing,
        use_mock=args.use_mock,
        overwrite=args.overwrite if args.overwrite else None,
    )
    print(f"[{STAGE_NAME}] nametrace_wide_parquet -> {stats['nametrace_wide_parquet']}")
    print(f"[{STAGE_NAME}] nametrace_long_parquet -> {stats['nametrace_long_parquet']}")


if __name__ == "__main__":
    main()
