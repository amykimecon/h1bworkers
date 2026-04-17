from __future__ import annotations

import argparse
import json
import math
import sys
from builtins import print as _print
from functools import lru_cache, partial
from importlib import import_module
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
    load_country_support,
    mock_country_probabilities,
    normalize_probability_dict,
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
            f"{STAGE_NAME} could not resolve any source parquet for the local name2nat step."
        )
    return resolved


def _serialize_prob_map(prob_map: dict[str, float]) -> str:
    return json.dumps(prob_map, sort_keys=True)


def _list_chunk_files(chunk_dir: Path, channel_label: str) -> list[Path]:
    return sorted(chunk_dir.glob(f"{channel_label}_*.parquet"))


def _count_parquet_rows(path: str | Path) -> int:
    con = get_duckdb_connection()
    return int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{escape_sql_literal(path)}')"
        ).fetchone()[0]
    )


def _count_rows_from_paths(paths: list[Path]) -> int:
    if not paths:
        return 0
    con = get_duckdb_connection()
    return int(
        con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [[str(path) for path in paths]],
        ).fetchone()[0]
    )


def _load_progress_state(
    progress_path: Path,
    *,
    total_values: int,
    scoring_chunk_size: int,
    channel_label: str,
) -> dict[str, Any]:
    state = {
        "channel": channel_label,
        "status": "running",
        "total_values": int(total_values),
        "scoring_chunk_size": int(scoring_chunk_size),
        "completed_chunks": 0,
        "completed_rows": 0,
        "last_completed_chunk": None,
        "names_dataset_hits": 0,
        "name2nat_hits": 0,
        "mock_hits": 0,
    }
    if not progress_path.exists():
        return state
    try:
        loaded = json.loads(progress_path.read_text())
    except json.JSONDecodeError:
        return state
    if (
        loaded.get("channel") != channel_label
        or int(loaded.get("total_values", -1)) != int(total_values)
        or int(loaded.get("scoring_chunk_size", -1)) != int(scoring_chunk_size)
    ):
        return state
    state.update(loaded)
    return state


class _NamesDatasetLookup:
    def __init__(
        self,
        *,
        country_cw: dict[str, str],
        top_n: int,
        full_first_weight: float,
        full_last_weight: float,
        cache_size: int,
    ) -> None:
        names_dataset_mod = import_module("names_dataset")
        self._dataset = names_dataset_mod.NameDataset(load_first_names=True, load_last_names=True)
        self._country_cw = country_cw
        self._top_n = max(1, int(top_n))
        self._full_first_weight = max(0.0, float(full_first_weight))
        self._full_last_weight = max(0.0, float(full_last_weight))
        self._search = lru_cache(maxsize=max(1000, int(cache_size)))(self._search_uncached)

    def _search_uncached(self, value: str) -> dict[str, Any]:
        try:
            result = self._dataset.search(value)
        except Exception:
            return {}
        return result if isinstance(result, dict) else {}

    def _normalize_country_payload(self, payload: Any) -> dict[str, float] | None:
        if not isinstance(payload, dict):
            return None
        raw_probs = payload.get("country")
        if not isinstance(raw_probs, dict) or not raw_probs:
            return None
        ranked = sorted(raw_probs.items(), key=lambda item: float(item[1]), reverse=True)[: self._top_n]
        normalized = normalize_probability_dict(dict(ranked), country_cw=self._country_cw)
        return normalized or None

    def _combine_prob_maps(
        self,
        weighted_prob_maps: list[tuple[dict[str, float] | None, float]],
    ) -> dict[str, float] | None:
        combined: dict[str, float] = {}
        active_weights = 0.0
        for prob_map, weight in weighted_prob_maps:
            if not prob_map:
                continue
            active_weights += max(0.0, float(weight))
            for country, prob in prob_map.items():
                combined[country] = combined.get(country, 0.0) + max(0.0, float(weight)) * float(prob)
        if not combined:
            return None
        if active_weights <= 0:
            active_weights = 1.0
        normalized = normalize_probability_dict(combined, country_cw=self._country_cw)
        if not normalized:
            return None
        ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)[: self._top_n]
        return normalize_probability_dict(dict(ranked), country_cw=self._country_cw) or None

    def predict(self, *, value: str, channel_label: str) -> dict[str, float] | None:
        text = str(value or "").strip()
        if not text:
            return None

        if channel_label == "first":
            return self._normalize_country_payload(self._search(text).get("first_name"))
        if channel_label == "last":
            return self._normalize_country_payload(self._search(text).get("last_name"))
        if channel_label != "full":
            return None

        parts = [part for part in text.split(" ") if part]
        if not parts:
            return None

        first_token = parts[0]
        last_token = parts[-1]
        first_probs = self._normalize_country_payload(self._search(first_token).get("first_name"))
        last_probs = self._normalize_country_payload(self._search(last_token).get("last_name"))
        if len(parts) == 1:
            return self._combine_prob_maps(
                [
                    (first_probs, self._full_first_weight),
                    (last_probs, self._full_last_weight),
                ]
            )
        return self._combine_prob_maps(
            [
                (first_probs, self._full_first_weight if last_probs else 1.0),
                (last_probs, self._full_last_weight if first_probs else 1.0),
            ]
        )


def _load_names_dataset_lookup(
    *,
    enabled: bool,
    country_cw: dict[str, str],
    top_n: int,
    full_first_weight: float,
    full_last_weight: float,
    cache_size: int,
) -> _NamesDatasetLookup | None:
    if not enabled:
        return None
    try:
        return _NamesDatasetLookup(
            country_cw=country_cw,
            top_n=top_n,
            full_first_weight=full_first_weight,
            full_last_weight=full_last_weight,
            cache_size=cache_size,
        )
    except Exception as exc:
        print(f"[{STAGE_NAME}] WARNING: names-dataset unavailable; falling back to name2nat-only scoring. {exc}")
        return None


def _predict_mock_batch(
    *,
    values: list[str],
    country_cw: dict[str, str],
    top_n: int,
) -> list[str]:
    return [
        _serialize_prob_map(
            normalize_probability_dict(
                mock_country_probabilities(value),
                country_cw=country_cw,
            )
        )
        for value in values
    ]


def _predict_name2nat_batch(
    *,
    values: list[str],
    top_n: int,
    country_cw: dict[str, str],
    model,
    mini_batch_size: int,
) -> list[str]:
    raw_predictions = model(values, top_n=int(top_n), mini_batch_size=max(1, int(mini_batch_size)))
    out: list[str] = []
    for raw_pred in raw_predictions:
        payload = raw_pred[1] if isinstance(raw_pred, (list, tuple)) and len(raw_pred) > 1 else raw_pred
        if isinstance(payload, dict):
            prob_map = payload
        else:
            try:
                prob_map = dict(payload)
            except Exception:
                prob_map = {}
        out.append(_serialize_prob_map(normalize_probability_dict(prob_map, country_cw=country_cw)))
    return out


def _predict_batch(
    *,
    values: list[str],
    channel_label: str,
    top_n: int,
    use_mock: bool,
    country_cw: dict[str, str],
    model,
    names_dataset_lookup: _NamesDatasetLookup | None,
    name2nat_fallback_batch_size: int,
    name2nat_mini_batch_size: int,
) -> dict[str, Any]:
    if not values:
        return {
            "pred_json": [],
            "model": model,
            "name2nat_failed": False,
            "names_dataset_hits": 0,
            "name2nat_hits": 0,
            "mock_hits": 0,
        }

    pred_json: list[str | None] = [None] * len(values)
    unresolved_values: list[str] = []
    unresolved_index: list[int] = []
    names_dataset_hits = 0
    name2nat_hits = 0
    mock_hits = 0
    name2nat_failed = False

    if names_dataset_lookup is not None:
        for idx, value in enumerate(values):
            lookup_probs = names_dataset_lookup.predict(value=value, channel_label=channel_label)
            if lookup_probs:
                pred_json[idx] = _serialize_prob_map(lookup_probs)
                names_dataset_hits += 1
            else:
                unresolved_values.append(value)
                unresolved_index.append(idx)
    else:
        unresolved_values = list(values)
        unresolved_index = list(range(len(values)))

    if unresolved_values:
        fallback_json: list[str]
        if use_mock:
            fallback_json = _predict_mock_batch(
                values=unresolved_values,
                country_cw=country_cw,
                top_n=top_n,
            )
            mock_hits += len(unresolved_values)
        else:
            if model is None:
                try:
                    name2nat_mod = import_module("name2nat")
                    model = name2nat_mod.Name2nat()
                except Exception as exc:
                    name2nat_failed = True
                    print(f"[{STAGE_NAME}] WARNING: name2nat unavailable for {channel_label}; using mock scores. {exc}")
            if model is not None and not name2nat_failed:
                try:
                    fallback_json = []
                    for start in range(0, len(unresolved_values), max(1, int(name2nat_fallback_batch_size))):
                        fallback_json.extend(
                            _predict_name2nat_batch(
                                values=unresolved_values[start : start + max(1, int(name2nat_fallback_batch_size))],
                                top_n=top_n,
                                country_cw=country_cw,
                                model=model,
                                mini_batch_size=name2nat_mini_batch_size,
                            )
                        )
                    name2nat_hits += len(unresolved_values)
                except Exception as exc:
                    name2nat_failed = True
                    print(
                        f"[{STAGE_NAME}] WARNING: name2nat failed for {channel_label}; "
                        f"using mock scores for the remaining unresolved names. {exc}"
                    )
            if model is None or name2nat_failed:
                fallback_json = _predict_mock_batch(
                    values=unresolved_values,
                    country_cw=country_cw,
                    top_n=top_n,
                )
                mock_hits += len(unresolved_values)

        for idx, payload in zip(unresolved_index, fallback_json):
            pred_json[idx] = payload

    return {
        "pred_json": [payload or "{}" for payload in pred_json],
        "model": model,
        "name2nat_failed": name2nat_failed,
        "names_dataset_hits": names_dataset_hits,
        "name2nat_hits": name2nat_hits,
        "mock_hits": mock_hits,
    }


def _score_channel_to_parquet(
    *,
    unique_parquet: str | Path,
    key_col: str,
    out_parquet: str | Path,
    chunk_dir: str | Path,
    channel_label: str,
    top_n: int,
    batch_size: int,
    scoring_chunk_size: int,
    use_mock: bool,
    country_cw: dict[str, str],
    overwrite: bool,
    names_dataset_lookup: _NamesDatasetLookup | None,
    name2nat_fallback_batch_size: int,
    name2nat_mini_batch_size: int,
) -> dict[str, Any]:
    out_path = ensure_parent_dir(out_parquet)
    chunk_root = Path(chunk_dir)
    if overwrite:
        clear_dir(chunk_root)
        if out_path.exists():
            out_path.unlink()
    else:
        chunk_root.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            return {
                f"{channel_label}_prediction_parquet": str(out_path),
                f"{channel_label}_prediction_rows": _count_parquet_rows(out_path),
            }

    total_values = _count_parquet_rows(unique_parquet)
    if total_values == 0:
        pd.DataFrame(columns=[key_col, "pred_json"]).to_parquet(out_path, index=False)
        return {
            f"{channel_label}_prediction_parquet": str(out_path),
            f"{channel_label}_prediction_rows": 0,
            f"{channel_label}_names_dataset_hits": 0,
            f"{channel_label}_name2nat_hits": 0,
            f"{channel_label}_mock_hits": 0,
        }

    model = None
    n_chunks = math.ceil(total_values / scoring_chunk_size)
    existing_chunks = _list_chunk_files(chunk_root, channel_label)
    completed_rows = _count_rows_from_paths(existing_chunks) if existing_chunks and not overwrite else 0
    progress_path = chunk_root / "progress.json"
    progress_state = _load_progress_state(
        progress_path,
        total_values=total_values,
        scoring_chunk_size=scoring_chunk_size,
        channel_label=channel_label,
    )
    progress_state["completed_rows"] = max(int(progress_state.get("completed_rows", 0)), completed_rows)
    backend_label = "lookup+mock" if use_mock else ("lookup+model" if names_dataset_lookup is not None else "model")
    progress = tqdm(
        total=total_values,
        initial=completed_rows,
        desc=f"[{STAGE_NAME}] name2nat:{channel_label}:{backend_label}",
        unit="name",
        file=sys.stdout,
    )
    total_names_dataset_hits = int(progress_state.get("names_dataset_hits", 0))
    total_name2nat_hits = int(progress_state.get("name2nat_hits", 0))
    total_mock_hits = int(progress_state.get("mock_hits", 0))
    completed_chunks = int(progress_state.get("completed_chunks", 0))

    con = get_duckdb_connection()
    for chunk_index in range(n_chunks):
        chunk_path = chunk_root / f"{channel_label}_{chunk_index:07d}.parquet"
        if chunk_path.exists() and not overwrite:
            continue

        row_start = chunk_index * scoring_chunk_size + 1
        row_end = min(total_values, (chunk_index + 1) * scoring_chunk_size)
        chunk_df = con.execute(
            f"""
            SELECT rownum, {key_col}
            FROM read_parquet('{escape_sql_literal(unique_parquet)}')
            WHERE rownum BETWEEN {row_start} AND {row_end}
            ORDER BY rownum
            """
        ).fetchdf()
        if chunk_df.empty:
            continue

        values = chunk_df[key_col].astype(str).tolist()
        pred_json: list[str] = []
        for batch_start in range(0, len(values), batch_size):
            batch = values[batch_start : batch_start + batch_size]
            batch_result = _predict_batch(
                values=batch,
                channel_label=channel_label,
                top_n=top_n,
                use_mock=use_mock,
                country_cw=country_cw,
                model=model,
                names_dataset_lookup=names_dataset_lookup,
                name2nat_fallback_batch_size=name2nat_fallback_batch_size,
                name2nat_mini_batch_size=name2nat_mini_batch_size,
            )
            pred_json.extend(batch_result["pred_json"])
            model = batch_result["model"]
            total_names_dataset_hits += int(batch_result["names_dataset_hits"])
            total_name2nat_hits += int(batch_result["name2nat_hits"])
            total_mock_hits += int(batch_result["mock_hits"])
            if batch_result["name2nat_failed"]:
                use_mock = True
                model = None

        chunk_out = pd.DataFrame({key_col: values, "pred_json": pred_json})
        atomic_write_parquet(chunk_out, chunk_path, index=False)
        completed_chunks += 1
        progress_state.update(
            {
                "status": "running",
                "completed_chunks": completed_chunks,
                "completed_rows": int(progress_state.get("completed_rows", 0)) + len(values),
                "last_completed_chunk": chunk_index,
                "names_dataset_hits": total_names_dataset_hits,
                "name2nat_hits": total_name2nat_hits,
                "mock_hits": total_mock_hits,
            }
        )
        atomic_write_json(progress_path, progress_state)
        progress.update(len(values))
    progress.close()

    chunk_files = _list_chunk_files(chunk_root, channel_label)
    if not chunk_files:
        atomic_write_parquet(pd.DataFrame(columns=[key_col, "pred_json"]), out_path, index=False)
    else:
        merge_con = get_duckdb_connection()
        atomic_duckdb_copy_to_parquet(
            merge_con,
            """
            SELECT *
            FROM read_parquet(?)
            """,
            out_path,
            [[str(path) for path in chunk_files]],
        )

    progress_state.update(
        {
            "status": "complete",
            "completed_chunks": completed_chunks,
            "completed_rows": total_values,
            "last_completed_chunk": n_chunks - 1 if n_chunks else None,
            "names_dataset_hits": total_names_dataset_hits,
            "name2nat_hits": total_name2nat_hits,
            "mock_hits": total_mock_hits,
            "final_prediction_parquet": str(out_path),
        }
    )
    atomic_write_json(progress_path, progress_state)
    print(
        f"[{STAGE_NAME}] name2nat:{channel_label} completed with "
        f"{total_names_dataset_hits:,} names via names-dataset, "
        f"{total_name2nat_hits:,} via name2nat, "
        f"and {total_mock_hits:,} via mock"
    )
    return {
        f"{channel_label}_prediction_parquet": str(out_path),
        f"{channel_label}_prediction_rows": _count_parquet_rows(out_path),
        f"{channel_label}_names_dataset_hits": total_names_dataset_hits,
        f"{channel_label}_name2nat_hits": total_name2nat_hits,
        f"{channel_label}_mock_hits": total_mock_hits,
    }


def _assemble_final_name2nat_output(
    *,
    base_names_parquet: str | Path,
    full_pred_parquet: str | Path,
    first_pred_parquet: str | Path,
    last_pred_parquet: str | Path,
    out_parquet: str | Path,
) -> None:
    out_path = ensure_parent_dir(out_parquet)
    con = get_duckdb_connection()
    atomic_duckdb_copy_to_parquet(
        con,
        f"""
        SELECT
            b.fullname_clean,
            b.first_name_clean,
            b.last_name_clean,
            COALESCE(CAST(fp.pred_json AS VARCHAR), '{{}}') AS pred_nats_full_json,
            CASE
                WHEN b.name_token_count > 1 THEN COALESCE(CAST(fip.pred_json AS VARCHAR), '{{}}')
                ELSE '{{}}'
            END AS pred_nats_first_json,
            CASE
                WHEN b.name_token_count > 1 THEN COALESCE(CAST(lp.pred_json AS VARCHAR), '{{}}')
                ELSE '{{}}'
            END AS pred_nats_last_json,
            COALESCE(CAST(fp.pred_json AS VARCHAR), '{{}}') AS pred_nats_name_json
        FROM read_parquet('{escape_sql_literal(base_names_parquet)}') AS b
        LEFT JOIN read_parquet('{escape_sql_literal(full_pred_parquet)}') AS fp
            ON b.fullname_clean = fp.fullname_clean
        LEFT JOIN read_parquet('{escape_sql_literal(first_pred_parquet)}') AS fip
            ON b.first_name_clean = fip.first_name_clean
        LEFT JOIN read_parquet('{escape_sql_literal(last_pred_parquet)}') AS lp
            ON b.last_name_clean = lp.last_name_clean
        ORDER BY b.full_rownum
        """,
        out_path,
    )


def run_name2nat(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    testing: bool | None = None,
    use_mock: bool | None = None,
    overwrite: bool | None = None,
) -> dict[str, Any]:
    cfg = pipeline_cfg or load_config(config_path)
    stage_cfg = get_stage_config(cfg, STAGE_NAME)
    out_path = ensure_parent_dir(stage_cfg["name2nat_parquet"])
    effective_overwrite = coerce_bool(
        stage_cfg.get("overwrite"),
        coerce_bool(cfg.get("build", {}).get("overwrite"), True),
    ) if overwrite is None else bool(overwrite)
    effective_testing = coerce_bool(cfg.get("testing", {}).get("enabled"), False) if testing is None else bool(testing)
    effective_use_mock = coerce_bool(stage_cfg.get("name2nat_use_mock"), False) if use_mock is None else bool(use_mock)

    if out_path.exists() and not effective_overwrite:
        return {
            "name2nat_parquet": str(out_path),
            "name2nat_source": "existing",
            "name2nat_rows": _count_parquet_rows(out_path),
        }

    source_path = _resolve_name_source_path(cfg, stage_cfg)
    testing_max_users = stage_cfg.get("testing_max_users") if effective_testing else None
    shard_spec = resolve_user_shard_spec(stage_cfg)
    shard_label = str(shard_spec["user_shard_label"]) if shard_spec else None
    min_fullname_chars = int(stage_cfg.get("name_model_min_fullname_chars", 2))
    min_token_chars = int(stage_cfg.get("name_model_min_token_chars", 2))
    work_root = out_path.parent / f"{out_path.stem}_work"
    work_root.mkdir(parents=True, exist_ok=True)

    shard_suffix = f" ({shard_label})" if shard_label else ""
    print(f"[{STAGE_NAME}] staging distinct cleaned names from {source_path}{shard_suffix}")
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
        f"[{STAGE_NAME}] name2nat staged {name_artifacts['n_full_names']:,} unique full names "
        f"({name_artifacts.get('n_single_token_names', 0):,} single-token names skip first/last rescoring), "
        f"{name_artifacts['n_first_names']:,} unique first names, "
        f"and {name_artifacts['n_last_names']:,} unique last names"
    )

    country_cw, _, _ = load_country_support()
    top_n = int(stage_cfg.get("name2nat_top_n", 20))
    fallback_batch_size = int(stage_cfg.get("name2nat_fallback_batch_size", stage_cfg.get("name2nat_batch_size", 2000)))
    lookup_batch_size = int(stage_cfg.get("name2nat_lookup_batch_size", max(fallback_batch_size, 10000)))
    scoring_chunk_size = int(stage_cfg.get("name2nat_scoring_chunk_size", max(lookup_batch_size, 10000)))
    name2nat_mini_batch_size = int(stage_cfg.get("name2nat_model_mini_batch_size", min(fallback_batch_size, 1024)))
    use_names_dataset_lookup = coerce_bool(stage_cfg.get("name2nat_use_names_dataset_lookup"), True) and not effective_use_mock
    names_dataset_full_first_weight = float(stage_cfg.get("name2nat_names_dataset_full_first_weight", 0.40))
    names_dataset_full_last_weight = float(stage_cfg.get("name2nat_names_dataset_full_last_weight", 0.60))
    names_dataset_cache_size = int(stage_cfg.get("name2nat_names_dataset_cache_size", 200000))
    names_dataset_lookup = _load_names_dataset_lookup(
        enabled=use_names_dataset_lookup,
        country_cw=country_cw,
        top_n=top_n,
        full_first_weight=names_dataset_full_first_weight,
        full_last_weight=names_dataset_full_last_weight,
        cache_size=names_dataset_cache_size,
    )
    if names_dataset_lookup is not None:
        print(
            f"[{STAGE_NAME}] names-dataset lookup enabled with full-name weights "
            f"first={names_dataset_full_first_weight:.2f}, last={names_dataset_full_last_weight:.2f}; "
            f"lookup_batch_size={lookup_batch_size:,}, fallback_batch_size={fallback_batch_size:,}, "
            f"scoring_chunk_size={scoring_chunk_size:,}"
        )

    full_stats = _score_channel_to_parquet(
        unique_parquet=name_artifacts["full_unique_parquet"],
        key_col="fullname_clean",
        out_parquet=work_root / "full_predictions.parquet",
        chunk_dir=work_root / "full_prediction_chunks",
        channel_label="full",
        top_n=top_n,
        batch_size=lookup_batch_size if names_dataset_lookup is not None else fallback_batch_size,
        scoring_chunk_size=scoring_chunk_size,
        use_mock=effective_use_mock,
        country_cw=country_cw,
        overwrite=prediction_overwrite,
        names_dataset_lookup=names_dataset_lookup,
        name2nat_fallback_batch_size=fallback_batch_size,
        name2nat_mini_batch_size=name2nat_mini_batch_size,
    )
    first_stats = _score_channel_to_parquet(
        unique_parquet=name_artifacts["first_unique_parquet"],
        key_col="first_name_clean",
        out_parquet=work_root / "first_predictions.parquet",
        chunk_dir=work_root / "first_prediction_chunks",
        channel_label="first",
        top_n=top_n,
        batch_size=lookup_batch_size if names_dataset_lookup is not None else fallback_batch_size,
        scoring_chunk_size=scoring_chunk_size,
        use_mock=effective_use_mock,
        country_cw=country_cw,
        overwrite=prediction_overwrite,
        names_dataset_lookup=names_dataset_lookup,
        name2nat_fallback_batch_size=fallback_batch_size,
        name2nat_mini_batch_size=name2nat_mini_batch_size,
    )
    last_stats = _score_channel_to_parquet(
        unique_parquet=name_artifacts["last_unique_parquet"],
        key_col="last_name_clean",
        out_parquet=work_root / "last_predictions.parquet",
        chunk_dir=work_root / "last_prediction_chunks",
        channel_label="last",
        top_n=top_n,
        batch_size=lookup_batch_size if names_dataset_lookup is not None else fallback_batch_size,
        scoring_chunk_size=scoring_chunk_size,
        use_mock=effective_use_mock,
        country_cw=country_cw,
        overwrite=prediction_overwrite,
        names_dataset_lookup=names_dataset_lookup,
        name2nat_fallback_batch_size=fallback_batch_size,
        name2nat_mini_batch_size=name2nat_mini_batch_size,
    )

    print(f"[{STAGE_NAME}] assembling final name2nat parquet")
    _assemble_final_name2nat_output(
        base_names_parquet=name_artifacts["base_names_parquet"],
        full_pred_parquet=full_stats["full_prediction_parquet"],
        first_pred_parquet=first_stats["first_prediction_parquet"],
        last_pred_parquet=last_stats["last_prediction_parquet"],
        out_parquet=out_path,
    )

    return {
        "name2nat_parquet": str(out_path),
        "name2nat_source": source_path,
        "name2nat_rows": _count_parquet_rows(out_path),
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
        **full_stats,
        **first_stats,
        **last_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local stage-04 name2nat step.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    testing_group = parser.add_mutually_exclusive_group()
    testing_group.add_argument("--testing", dest="testing", action="store_true")
    testing_group.add_argument("--no-testing", dest="testing", action="store_false")
    parser.set_defaults(testing=None)
    args = parser.parse_args(sanitize_ipykernel_argv())

    stats = run_name2nat(
        config_path=args.config,
        testing=args.testing,
        use_mock=args.use_mock,
        overwrite=args.overwrite if args.overwrite else None,
    )
    print(f"[{STAGE_NAME}] name2nat_parquet -> {stats['name2nat_parquet']}")
    print(f"[{STAGE_NAME}] name2nat_rows={stats['name2nat_rows']}")


if __name__ == "__main__":
    main()
