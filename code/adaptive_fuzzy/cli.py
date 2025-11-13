"""Command line interface for adaptive fuzzy university clustering."""

from __future__ import annotations

import argparse
import os
import sys
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

try:  # pragma: no cover - fallback for older sklearn
    import joblib
except ImportError:  # pragma: no cover
    from sklearn.externals import joblib  # type: ignore
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier

from . import (
    LabelledExample,
    PairCandidate,
    extract_name_features,
    build_clusters,
    clusters_to_frame,
    compute_features,
    fit_classifier,
    fit_name_classifier,
    generate_pair_candidates,
    UnionFind,
    StreamingUnionFind,
    prepare_name_groups,
    set_feature_cache,
    set_generic_name_model,
    set_non_degree_name_model,
    is_generic_name,
    normalize_name,
    FEATURE_NAMES,
    set_token_statistics_from_names,
    score_candidates,
    get_raw_name_metadata,
    mark_generic_name,
    unmark_generic_name,
    mark_non_degree_name,
    unmark_non_degree_name,
    is_non_degree_program_name,
    _is_non_degree_program_name,
    prepare_normalized_name_parquet,
    generate_ann_embeddings,
    build_faiss_index,
    AnnRetriever,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PACKAGE_ROOT / ".adaptive_fuzzy"
DEFAULT_ARCHIVE_PATH = DEFAULT_DATA_DIR / "label_history.csv"
DEFAULT_NAME_LABELS_PATH = DEFAULT_DATA_DIR / "name_labels.csv"

PENDING_GENERIC_CONFIRMATIONS: List[str] = []
CONFIRMED_GENERIC_NAMES: Set[str] = set()
PENDING_NON_DEGREE_CONFIRMATIONS: List[str] = []
CONFIRMED_NON_DEGREE_NAMES: Set[str] = set()

GENERIC_NAME_MODEL: Optional[object] = None
GENERIC_MODEL_THRESHOLD: float = 0.8
NON_DEGREE_NAME_MODEL: Optional[object] = None
NON_DEGREE_MODEL_THRESHOLD: float = 0.8

NAME_LABEL_STORE_PATH: Optional[Path] = None
_NAME_LABEL_REGISTRY: Dict[Tuple[str, str], Dict[str, Union[str, int, float]]] = {}
_NAME_LABEL_DIRTY = False

STREAMING_SCORING_CHUNK_SIZE = 200_000


def _optional_path(value: Optional[Union[str, Path]]) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        text = str(value)
    else:
        text = value
    text = text.strip()
    if not text or text.lower() == "none":
        return None
    return Path(text).expanduser()




def _load_name_label_store(path: Path) -> Dict[Tuple[str, str], Dict[str, Union[str, int, float]]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not load name labels from {path}: {exc}")
        return {}
    required = {"label_type", "canonical_name", "label_value"}
    if not required.issubset(df.columns):
        print(f"Warning: name label file {path} missing required columns {required}; ignoring.")
        return {}
    registry: Dict[Tuple[str, str], Dict[str, Union[str, int, float]]] = {}
    for row in df.itertuples(index=False):
        label_type = getattr(row, "label_type", None)
        canonical_name = getattr(row, "canonical_name", None)
        if not isinstance(label_type, str) or not isinstance(canonical_name, str):
            continue
        key = (label_type, canonical_name)
        label_value = getattr(row, "label_value", None)
        try:
            label_int = int(label_value)
        except (TypeError, ValueError):
            continue
        entry: Dict[str, Union[str, int, float]] = {
            "label_type": label_type,
            "canonical_name": canonical_name,
            "raw_name": getattr(row, "raw_name", canonical_name) if isinstance(getattr(row, "raw_name", ""), str) else canonical_name,
            "label_value": label_int,
            "source": getattr(row, "source", "loaded"),
            "updated_at": getattr(row, "updated_at", ""),
        }
        confidence = getattr(row, "confidence", None)
        if isinstance(confidence, (int, float)):
            entry["confidence"] = float(confidence)
        registry[key] = entry
    return registry


def _configure_name_label_store(path: Optional[Path]) -> None:
    global NAME_LABEL_STORE_PATH, _NAME_LABEL_REGISTRY, _NAME_LABEL_DIRTY
    NAME_LABEL_STORE_PATH = path
    _NAME_LABEL_REGISTRY = {}
    _NAME_LABEL_DIRTY = False
    if path is None:
        return
    _NAME_LABEL_REGISTRY = _load_name_label_store(path)


def _persist_name_label_store() -> None:
    global _NAME_LABEL_DIRTY
    if not _NAME_LABEL_DIRTY or NAME_LABEL_STORE_PATH is None:
        _NAME_LABEL_DIRTY = False
        return
    records = list(_NAME_LABEL_REGISTRY.values())
    df = pd.DataFrame.from_records(records)
    NAME_LABEL_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(NAME_LABEL_STORE_PATH, index=False)
    _NAME_LABEL_DIRTY = False


def _record_name_label(
    name: str,
    label_type: str,
    label_value: int,
    source: str,
    confidence: Optional[float] = None,
) -> None:
    global _NAME_LABEL_DIRTY
    canonical = normalize_name(name)
    raw = name.strip()
    if not canonical and not raw:
        return
    if not canonical:
        canonical = raw
    key = (label_type, canonical)
    entry = _NAME_LABEL_REGISTRY.get(key, {}).copy()
    entry["label_type"] = label_type
    entry["canonical_name"] = canonical
    entry["raw_name"] = raw or entry.get("raw_name", canonical)
    entry["label_value"] = int(label_value)
    entry["source"] = source
    entry["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if confidence is not None:
        entry["confidence"] = float(confidence)
    _NAME_LABEL_REGISTRY[key] = entry
    _NAME_LABEL_DIRTY = True


def _name_label_dataframe(label_type: Optional[str] = None) -> pd.DataFrame:
    if not _NAME_LABEL_REGISTRY:
        columns = ["label_type", "canonical_name", "raw_name", "label_value", "source", "updated_at", "confidence"]
        return pd.DataFrame(columns=columns)
    records = list(_NAME_LABEL_REGISTRY.values())
    df = pd.DataFrame.from_records(records)
    if label_type is not None:
        df = df[df["label_type"] == label_type].copy()
    return df


def _collect_name_training_samples(label_type: str) -> List[Tuple[str, int]]:
    df = _name_label_dataframe(label_type)
    if df.empty:
        return []
    samples: List[Tuple[str, int]] = []
    for row in df.itertuples(index=False):
        raw_name = getattr(row, "raw_name", "")
        canonical = getattr(row, "canonical_name", "")
        name = raw_name if isinstance(raw_name, str) and raw_name else canonical
        if not name:
            continue
        try:
            label_value = int(getattr(row, "label_value"))
        except (TypeError, ValueError):
            continue
        samples.append((name, label_value))
    return samples


def _bootstrap_generic_negative_labels_from_pairs(
    archive_path: Optional[Path],
    min_pair_labels: int,
    max_names: Optional[int],
    seed: Optional[int],
) -> int:
    if archive_path is None or not archive_path.exists():
        return 0
    try:
        df = pd.read_csv(archive_path)
    except Exception as exc:
        print(f"Warning: could not inspect labelled pairs from {archive_path}: {exc}")
        return 0
    required = {"name_a", "name_b", "label"}
    if not required.issubset(df.columns):
        print(
            f"Warning: labelled pair archive {archive_path} missing required columns {required}; skipping generic bootstrapping."
        )
        return 0
    filtered = df[df["label"].isin([0, 1])]
    if filtered.empty:
        return 0
    names = pd.concat([filtered["name_a"], filtered["name_b"]], ignore_index=True)
    names = names.dropna().astype(str).str.strip()
    names = names[names != ""]
    if names.empty:
        return 0
    counts = names.value_counts()
    eligible = counts[counts >= max(1, min_pair_labels)].index.tolist()
    if not eligible:
        return 0
    effective_max: Optional[int]
    if max_names is None or max_names <= 0:
        effective_max = None
    else:
        effective_max = max_names
    rng = np.random.default_rng(seed) if seed is not None else None
    if effective_max is not None and len(eligible) > effective_max:
        if rng is None:
            rng = np.random.default_rng()
        selected = list(rng.choice(eligible, size=effective_max, replace=False))
    else:
        selected = eligible
    added = 0
    for name in selected:
        canonical = normalize_name(name)
        if not canonical:
            continue
        if ("generic", canonical) in _NAME_LABEL_REGISTRY:
            continue
        if is_generic_name(canonical):
            continue
        _record_name_label(name, "generic", 0, "pair_label_bootstrap")
        added += 1
    if added:
        print(f"Bootstrapped {added} non-generic names from {archive_path}.")
    return added


def _binary_model_probability(model: Optional[object], name: str) -> Optional[float]:
    if model is None:
        return None
    try:
        features = extract_name_features(name)
    except Exception:
        return None
    proba = model.predict_proba(features.reshape(1, -1))
    classes = getattr(model, "classes_", np.array([0, 1]))
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    if proba.shape[1] == 1:
        single_class = classes[0] if len(classes) else 0
        probability = float(proba[0, 0])
        if single_class != 1:
            probability = 1.0 - probability
    else:
        if 1 in classes:
            idx = int(np.where(classes == 1)[0][0])
            probability = float(proba[0, idx])
        else:
            probability = float(proba[0].max())
    return probability


def _load_or_train_name_model(
    load_path: Optional[Path],
    train_path: Optional[Path],
    label_type: str,
    description: str,
) -> Optional[object]:
    model: Optional[object] = None
    if load_path:
        if not load_path.exists():
            print(f"Warning: {description} model path {load_path} does not exist.")
        else:
            try:
                model = joblib.load(load_path)
                print(f"Loaded {description} model from {load_path}.")
            except Exception as exc:
                print(f"Warning: failed to load {description} model from {load_path}: {exc}")
    if train_path:
        samples = _collect_name_training_samples(label_type)
        label_set = {label for _, label in samples}
        if len(label_set) < 2:
            print(
                f"Unable to train {description} model: need at least one positive and one negative name label."
            )
        else:
            model = fit_name_classifier(samples)
            train_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, train_path)
            print(f"Trained {description} model on {len(samples)} samples and saved to {train_path}.")
    return model


def _canonical_generic_name(name: str) -> str:
    return name.strip()


def queue_generic_confirmation(name: str) -> None:
    canonical = _canonical_generic_name(name)
    if not canonical:
        return
    if canonical in CONFIRMED_GENERIC_NAMES:
        return
    if canonical not in PENDING_GENERIC_CONFIRMATIONS:
        PENDING_GENERIC_CONFIRMATIONS.append(canonical)


def process_pending_generic_confirmations() -> Set[str]:
    confirmed: Set[str] = set()
    while PENDING_GENERIC_CONFIRMATIONS:
        name = PENDING_GENERIC_CONFIRMATIONS.pop(0)
        while True:
            response = input(f"Confirm marking '{name}' as a generic institution? [y/n]: ").strip().lower()
            if response in {"y", "yes"}:
                mark_generic_name(name)
                probability = _binary_model_probability(GENERIC_NAME_MODEL, name)
                _record_name_label(name, "generic", 1, "manual_confirm", probability)
                _persist_name_label_store()
                CONFIRMED_GENERIC_NAMES.add(name)
                confirmed.add(name)
                print(f"Confirmed '{name}' as generic. Withholding it from further training chunks.")
                break
            if response in {"n", "no"}:
                probability = _binary_model_probability(GENERIC_NAME_MODEL, name)
                _record_name_label(name, "generic", 0, "manual_reject", probability)
                _persist_name_label_store()
                print(f"'{name}' will remain in the candidate pool.")
                break
            print("Please respond with y or n.")
    return confirmed


def queue_nondegree_confirmation(name: str) -> None:
    canonical = normalize_name(name)
    if not canonical:
        return
    if canonical in CONFIRMED_NON_DEGREE_NAMES:
        return
    if canonical not in PENDING_NON_DEGREE_CONFIRMATIONS:
        PENDING_NON_DEGREE_CONFIRMATIONS.append(name.strip())


def process_pending_nondegree_confirmations() -> Set[str]:
    confirmed: Set[str] = set()
    while PENDING_NON_DEGREE_CONFIRMATIONS:
        name = PENDING_NON_DEGREE_CONFIRMATIONS.pop(0)
        while True:
            response = input(f"Confirm marking '{name}' as a non-degree program? [y/n]: ").strip().lower()
            if response in {"y", "yes"}:
                normalized = normalize_name(name)
                mark_non_degree_name(name)
                if normalized:
                    CONFIRMED_NON_DEGREE_NAMES.add(normalized)
                probability = _binary_model_probability(NON_DEGREE_NAME_MODEL, name)
                _record_name_label(name, "non_degree", 1, "manual_confirm", probability)
                _persist_name_label_store()
                print(f"Confirmed '{name}' as non-degree. It will be excluded from training on future runs.")
                confirmed.add(name.strip())
                break
            if response in {"n", "no"}:
                normalized = normalize_name(name)
                unmark_non_degree_name(name)
                if normalized:
                    CONFIRMED_NON_DEGREE_NAMES.discard(normalized)
                probability = _binary_model_probability(NON_DEGREE_NAME_MODEL, name)
                _record_name_label(name, "non_degree", 0, "manual_reject", probability)
                _persist_name_label_store()
                print(f"'{name}' will remain in the degree-granting set.")
                break
            print("Please respond with y or n.")
    return confirmed


def remove_generic_labels(
    labelled_store: Optional[Dict[Tuple[str, str], LabelledExample]],
    generics: Set[str],
) -> None:
    if not labelled_store or not generics:
        return
    for key in list(labelled_store.keys()):
        if key[0] in generics or key[1] in generics:
            labelled_store.pop(key, None)


def remove_nondegree_labels(
    labelled_store: Optional[Dict[Tuple[str, str], LabelledExample]],
    non_degree_names: Set[str],
) -> None:
    if not labelled_store or not non_degree_names:
        return
    for key in list(labelled_store.keys()):
        if key[0] in non_degree_names or key[1] in non_degree_names:
            labelled_store.pop(key, None)


def _model_feature_count(model: RandomForestClassifier) -> Optional[int]:
    return getattr(model, "n_features_in_", None)


def _ensure_model_compatibility(
    model: Optional[RandomForestClassifier],
    feature_length: int,
    context: str,
) -> Optional[RandomForestClassifier]:
    if model is None:
        return None
    expected = _model_feature_count(model)
    if expected is not None and expected != feature_length:
        print(
            f"Loaded model expects {expected} features but the current pipeline produces {feature_length} (context: {context})."
        )
        print("Reverting to retraining with the updated feature set.")
        return None
    return model


def summarize_random_forest(model: RandomForestClassifier, feature_names: Sequence[str], top_k: int = 10) -> str:
    lines: List[str] = []
    lines.append("RandomForestClassifier summary:")
    lines.append(f"  n_estimators: {getattr(model, 'n_estimators', 'n/a')}")
    if getattr(model, "estimators_", None):
        first_tree = getattr(model.estimators_[0], "tree_", None)
        max_depth = first_tree.max_depth if first_tree is not None else "n/a"
    else:
        max_depth = "n/a"
    lines.append(f"  max_depth (first tree): {max_depth}")
    lines.append(f"  n_features_in_: {_model_feature_count(model)}")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        paired = list(enumerate(importances))
        paired.sort(key=lambda item: item[1], reverse=True)
        lines.append("  Top feature importances:")
        for idx, value in paired[: min(top_k, len(paired))]:
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            lines.append(f"    {name}: {value:.4f}")
    else:
        lines.append("  Feature importances not available.")

    return "\n".join(lines)


@dataclass
class InteractiveConfig:
    initial_labels: int = 15
    batch_size: int = 10
    match_threshold: float = 0.6


@dataclass(frozen=True)
class PromptResult:
    action: str
    value: Optional[Union[int, str]] = None


def _filter_non_degree_names(names: Sequence[str]) -> Tuple[List[str], int]:
    filtered: List[str] = []
    removed = 0
    for name in names:
        probability = _binary_model_probability(NON_DEGREE_NAME_MODEL, name)
        heuristic_positive = _is_non_degree_program_name(name)
        model_positive = probability is not None and probability >= NON_DEGREE_MODEL_THRESHOLD
        if heuristic_positive or model_positive:
            if heuristic_positive and model_positive:
                source = "heuristic+model"
            elif heuristic_positive:
                source = "heuristic"
            else:
                source = "model"
            _record_name_label(name, "non_degree", 1, source, probability)
            removed += 1
            continue
        _record_name_label(name, "non_degree", 0, "pass", probability)
        filtered.append(name)
    return filtered, removed


def _match_probabilities(model: RandomForestClassifier, features: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(features)
    classes = getattr(model, "classes_", np.array([0, 1]))
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if proba.shape[1] == 1:
        single_class = classes[0] if len(classes) else 0
        probs = proba[:, 0]
        if single_class != 1:
            probs = 1.0 - probs
    else:
        if 1 in classes:
            idx = int(np.where(classes == 1)[0][0])
            probs = proba[:, idx]
        else:
            probs = proba.max(axis=1)
    return probs.astype(float)


def _scaled_candidate_limit(size: int, multiplier: int, minimum: int = 0, cap: Optional[int] = None) -> int:
    if size <= 0:
        return 0
    limit = max(minimum, size * multiplier)
    if cap is not None:
        limit = min(limit, cap)
    return limit



def _ensure_model_with_refit(
    model: Optional[RandomForestClassifier],
    feature_length: int,
    context: str,
    labelled_examples: Optional[Dict[Tuple[str, str], LabelledExample]],
) -> RandomForestClassifier:
    checked = _ensure_model_compatibility(model, feature_length, context)
    if checked is not None:
        return checked
    if labelled_examples:
        print("Refit model using available labels to match the updated feature set.")
        refit = fit_classifier(labelled_examples.values())
        checked = _ensure_model_compatibility(refit, feature_length, context)
        if checked is not None:
            return checked
    raise RuntimeError(
        f"No compatible model available for {context}; rerun with --resume-training to rebuild the classifier."
    )


def _read_cached_names(
    cache_path: Path,
    limit: Optional[int],
    sample_fraction: Optional[float],
    sample_seed: Optional[int],
) -> List[str]:
    pq_file = pq.ParquetFile(cache_path)
    schema_names = set(pq_file.schema.names)
    if "name" not in schema_names:
        raise ValueError(f"Names cache {cache_path} missing required 'name' column.")
    columns: List[str] = ["name"]
    has_norm = "norm" in schema_names
    if has_norm:
        columns.append("norm")
    seen_norm: Set[str] = set()
    names: List[str] = []
    for batch in pq_file.iter_batches(columns=columns, batch_size=50_000):
        data = batch.to_pydict()
        raw_values = data.get("name", [])
        norm_values = data.get("norm", []) if has_norm else []
        for idx, raw in enumerate(raw_values):
            if not isinstance(raw, str):
                continue
            if has_norm:
                norm = norm_values[idx]
                if isinstance(norm, str) and norm:
                    key = norm
                else:
                    key = normalize_name(raw)
            else:
                key = normalize_name(raw)
            if key in seen_norm:
                continue
            seen_norm.add(key)
            names.append(raw)
            if limit is not None and len(names) >= limit:
                break
        if limit is not None and len(names) >= limit:
            break
    if not names:
        raise ValueError(f"Cached names in {cache_path} were empty or invalid.")
    if sample_fraction is not None:
        if not 0 < sample_fraction <= 1:
            raise ValueError("Sample fraction must be between 0 (exclusive) and 1 (inclusive).")
        rng = np.random.default_rng(sample_seed)
        sample_count = max(1, int(len(names) * sample_fraction))
        indices = rng.choice(len(names), size=min(len(names), sample_count), replace=False)
        names = [names[idx] for idx in indices]
    return names[:limit] if limit is not None else names


def load_names(
    path: Path,
    column: Optional[str],
    limit: Optional[int],
    sample_fraction: Optional[float],
    sample_seed: Optional[int],
    cache_path: Optional[Path] = None,
) -> List[str]:
    limit_desc = "unbounded" if limit is None else f"{limit:,}"
    print(f"[load_names] Loading names (limit={limit_desc}) from {path}")
    if cache_path and cache_path.exists():
        print(f"[load_names] Found existing normalized-name cache at {cache_path}; loading directly.")
        names = _read_cached_names(cache_path, limit, sample_fraction, sample_seed)
        names, removed = _filter_non_degree_names(names)
        if removed:
            print(f"Removed {removed} non-degree program names from cached input.")
        if not names:
            raise ValueError("All cached names were filtered out as non-degree programs.")
        print(f"Loaded {len(names):,} names from cache {cache_path}.")
        return names

    if not path.exists():
        raise FileNotFoundError(f"Input file {path} does not exist")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[load_names] Normalizing names from {path} into cache {cache_path}.")
        written = prepare_normalized_name_parquet(path, column, cache_path, limit=limit)
        print(f"Cached {written:,} normalized names to {cache_path}.")
        names = _read_cached_names(cache_path, limit, sample_fraction, sample_seed)
        names, removed = _filter_non_degree_names(names)
        if removed:
            print(f"Removed {removed} non-degree program names from cached input.")
        if not names:
            raise ValueError("All cached names were filtered out as non-degree programs.")
        print(f"Loaded {len(names):,} names from cache {cache_path}.")
        return names

    print(f"[load_names] Reading raw names directly from {path} (no cache).")
    names_series: pd.Series
    if limit is not None:
        raw_values = _read_limited_raw_names(path, column, limit)
        col_name = column or "name"
        names_series = pd.Series(raw_values, name=col_name)
    else:
        if path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        elif path.suffix.lower() in {".csv", ".txt"}:
            df = pd.read_csv(path)
        else:
            raise ValueError(
                f"Unsupported file extension for {path}. Use csv, txt, parquet, or pq."
            )
        if column is None:
            if len(df.columns) != 1:
                raise ValueError(
                    "Column name must be provided when the input has multiple columns."
                )
            column = df.columns[0]
        names_series = df[column].astype(str).str.strip()

    unique_names = sorted(names_series.dropna().unique())
    print(f"[load_names] Retrieved {len(unique_names):,} unique raw names before filtering.")
    unique_names, removed = _filter_non_degree_names(unique_names)
    if removed:
        print(f"Removed {removed} non-degree program names from the input file.")
    if not unique_names:
        raise ValueError("All provided names were filtered out as non-degree programs.")

    if sample_fraction is not None:
        if not 0 < sample_fraction <= 1:
            raise ValueError("Sample fraction must be between 0 (exclusive) and 1 (inclusive).")
        rng = np.random.default_rng(sample_seed)
        sample_count = max(1, int(len(unique_names) * sample_fraction))
        unique_names = list(rng.choice(unique_names, size=min(len(unique_names), sample_count), replace=False))

    if limit is not None:
        unique_names = unique_names[:limit]
        print(f"[load_names] Trimmed to first {limit:,} names after filtering.")
    return unique_names


def _read_limited_raw_names(path: Path, column: Optional[str], limit: int) -> List[str]:
    """Return up to `limit` raw values from the specified column without reading the full file."""

    if limit <= 0:
        return []

    suffix = path.suffix.lower()
    values: List[str] = []
    remaining = limit

    if suffix in {".parquet", ".pq"}:
        pq_file = pq.ParquetFile(path)
        columns = pq_file.schema.names
        target_col = column
        if target_col is None:
            if len(columns) != 1:
                raise ValueError("Column name must be provided when the input has multiple columns.")
            target_col = columns[0]
        for batch in pq_file.iter_batches(columns=[target_col], batch_size=50_000):
            col = batch.column(0).to_pylist()
            for raw in col:
                values.append("" if raw is None else str(raw))
                remaining -= 1
                if remaining <= 0:
                    return values
        return values

    if suffix in {".csv", ".txt"}:
        target_col = column
        if target_col is None:
            preview = pd.read_csv(path, nrows=0)
            columns = preview.columns.tolist()
            if len(columns) != 1:
                raise ValueError("Column name must be provided when the input has multiple columns.")
            target_col = columns[0]
        reader = pd.read_csv(path, usecols=[target_col], chunksize=50_000)
        for chunk in reader:
            for raw in chunk[target_col].tolist():
                if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                    values.append("")
                else:
                    values.append(str(raw))
                remaining -= 1
                if remaining <= 0:
                    return values
        return values

    raise ValueError(f"Unsupported file extension for {path}. Use csv, txt, parquet, or pq.")


def prompt_label(candidate: PairCandidate) -> PromptResult:
    question = (
        f"Match these universities?\n"
        f"  A) {candidate.name_a}\n  B) {candidate.name_b}\n"
        "Enter y (match), n (not a match), ga (mark A generic), gb (mark B generic), "
        "nda (mark A non-degree), ndb (mark B non-degree), u (unsure/skip), or q (quit): "
    )
    while True:
        answer = input(question).strip().lower()
        if answer in {"y", "yes", "1"}:
            return PromptResult("label", 1)
        if answer in {"n", "no", "0"}:
            return PromptResult("label", 0)
        if answer in {"ga", "ag", "generic a", "generic_a"}:
            return PromptResult("generic", "a")
        if answer in {"gb", "bg", "generic b", "generic_b"}:
            return PromptResult("generic", "b")
        if answer in {"nda", "nondegree a", "non_degree a", "nondegree_a"}:
            return PromptResult("nondegree", "a")
        if answer in {"ndb", "nondegree b", "non_degree b", "nondegree_b"}:
            return PromptResult("nondegree", "b")
        if answer in {"u", "skip", ""}:
            return PromptResult("skip", None)
        if answer in {"q", "quit", "exit"}:
            return PromptResult("quit", None)
        print("Please respond with y, n, ga, gb, nda, ndb, u, or q.")


def labelled_to_frame(labelled: Dict[Tuple[str, str], LabelledExample]) -> pd.DataFrame:
    rows = []
    for (name_a, name_b), example in labelled.items():
        row = {
            "name_a": name_a,
            "name_b": name_b,
            "label": example.label,
        }
        for idx, value in enumerate(example.features):
            row[f"feature_{idx}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def save_labelled_examples(
    labelled: Dict[Tuple[str, str], LabelledExample],
    archive_path: Optional[Path],
) -> None:
    if archive_path is None or not labelled:
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = labelled_to_frame(labelled)
    if archive_path.exists():
        existing = pd.read_csv(archive_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["name_a", "name_b"], keep="last")
    else:
        combined = new_df
    combined.to_csv(archive_path, index=False)


def load_labelled_examples(archive_path: Optional[Path]) -> Dict[Tuple[str, str], LabelledExample]:
    if archive_path is None or not archive_path.exists():
        return {}
    df = pd.read_csv(archive_path)
    loaded: Dict[Tuple[str, str], LabelledExample] = {}
    df = df.dropna(subset=["name_a", "name_b"])
    feature_cols = sorted(
        [col for col in df.columns if col.startswith("feature_")],
        key=lambda col: int(col.split("_")[1]),
    )
    for row in df.itertuples(index=False):
        name_a = getattr(row, "name_a", None)
        name_b = getattr(row, "name_b", None)
        if not isinstance(name_a, str) or not isinstance(name_b, str):
            continue
        use_stored = bool(feature_cols) and len(feature_cols) == len(FEATURE_NAMES)
        if use_stored:
            features = np.array([getattr(row, col) for col in feature_cols], dtype=float)
        else:
            features = compute_features(name_a, name_b)
        loaded[(name_a, name_b)] = LabelledExample(features, int(row.label))
    return loaded


class CandidateCacheManager:
    """Incrementally manages candidate generation and on-disk caching."""

    def __init__(
        self,
        names: Sequence[str],
        representative_ids: Dict[str, Optional[str]],
        cache_path: Optional[Path],
        token_overlap_threshold: int,
        jaro_threshold: float,
        feature_cache: Dict[Tuple[str, str], np.ndarray],
        ann_retriever: Optional[AnnRetriever] = None,
    ) -> None:
        self.names = list(names)
        self.representative_ids = representative_ids
        self.cache_path = cache_path
        self.token_overlap_threshold = token_overlap_threshold
        self.jaro_threshold = jaro_threshold
        self.feature_cache = feature_cache
        self.pairs: Dict[Tuple[str, str], PairCandidate] = {}
        self.coverage_limit = 0
        self._name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(self.names)}
        self.ann_retriever = ann_retriever

    def coverage_size(self) -> int:
        """Return how many leading names have candidate coverage."""

        return min(self.coverage_limit, len(self.names))

    def expand_to_size(self, subset_size: int, max_candidates: Optional[int] = None) -> None:
        """Ensure the cache includes candidates for the first `subset_size` names."""

        target = min(max(0, subset_size), len(self.names))
        if target <= self.coverage_limit:
            return
        self._extend_coverage(target, max_candidates)

    def bootstrap_pairs(self, pairs_with_features: Iterable[Tuple[PairCandidate, np.ndarray]]) -> None:
        for candidate, features in pairs_with_features:
            if is_generic_name(candidate.name_a) or is_generic_name(candidate.name_b):
                continue
            key = (candidate.name_a, candidate.name_b)
            self.pairs[key] = PairCandidate(candidate.name_a, candidate.name_b, candidate.score)
            self._store_features(candidate.name_a, candidate.name_b, features)
            self._update_coverage_with_pair(candidate.name_a, candidate.name_b)

    def ensure_subset(self, subset: Sequence[str], max_candidates: Optional[int] = None) -> List[PairCandidate]:
        subset_size = len(subset)
        self.expand_to_size(subset_size, max_candidates)
        return self._collect_subset(subset)

    def add_candidates(self, candidates: Sequence[PairCandidate]) -> None:
        if not candidates:
            return
        filtered = filter_candidates_by_ground_truth(list(candidates), self.representative_ids)
        self._register_candidates(filtered, persist=True)

    def _extend_coverage(self, subset_size: int, max_candidates: Optional[int]) -> None:
        target_subset = self.names[:subset_size]
        if not target_subset:
            self.coverage_limit = subset_size
            return
        limit = max_candidates
        if limit is None:
            limit = _scaled_candidate_limit(len(target_subset), 200, minimum=10_000)
        if self.ann_retriever is not None:
            tuples = self.ann_retriever.generate_candidates(target_subset, limit)
            generated = [PairCandidate(a, b, score) for a, b, score in tuples]
        else:
            generated = generate_pair_candidates(
                target_subset,
                max_candidates=limit,
                token_overlap_threshold=self.token_overlap_threshold,
                jaro_threshold=self.jaro_threshold,
            )
        filtered = filter_candidates_by_ground_truth(generated, self.representative_ids)
        self._register_candidates(filtered, persist=True)
        self.coverage_limit = max(self.coverage_limit, subset_size)

    def _register_candidates(self, candidates: Sequence[PairCandidate], persist: bool) -> None:
        if not candidates:
            return
        new_rows: List[Dict[str, float]] = []
        for cand in candidates:
            key = (cand.name_a, cand.name_b)
            existing = self.pairs.get(key)
            if existing is not None and existing.score >= cand.score:
                continue
            if is_generic_name(cand.name_a) or is_generic_name(cand.name_b):
                continue
            self.pairs[key] = PairCandidate(cand.name_a, cand.name_b, cand.score)
            features = self._compute_features(cand.name_a, cand.name_b)
            row = {
                "name_a": cand.name_a,
                "name_b": cand.name_b,
                "score": cand.score,
            }
            for idx, value in enumerate(features):
                row[f"feature_{idx}"] = float(value)
            new_rows.append(row)
            self._update_coverage_with_pair(cand.name_a, cand.name_b)
        if persist and new_rows:
            self._persist_rows(new_rows)

    def _collect_subset(self, subset: Sequence[str]) -> List[PairCandidate]:
        subset_set = set(subset)
        relevant = [
            cand
            for (name_a, name_b), cand in self.pairs.items()
            if name_a in subset_set and name_b in subset_set
        ]
        relevant = [
            cand
            for cand in relevant
            if not (is_generic_name(cand.name_a) or is_generic_name(cand.name_b))
        ]
        relevant.sort(key=lambda cand: cand.score, reverse=True)
        return relevant

    def _compute_features(self, name_a: str, name_b: str) -> np.ndarray:
        cached = self.feature_cache.get((name_a, name_b))
        if cached is not None:
            return cached
        features = compute_features(name_a, name_b)
        self._store_features(name_a, name_b, features)
        return features

    def _store_features(self, name_a: str, name_b: str, features: np.ndarray) -> None:
        self.feature_cache[(name_a, name_b)] = features.copy()
        self.feature_cache[(name_b, name_a)] = features.copy()

    def _update_coverage_with_pair(self, name_a: str, name_b: str) -> None:
        idx_a = self._name_to_index.get(name_a, -1)
        idx_b = self._name_to_index.get(name_b, -1)
        max_idx = max(idx_a, idx_b)
        if max_idx >= 0 and max_idx + 1 > self.coverage_limit:
            self.coverage_limit = max_idx + 1

    def _persist_rows(self, rows: List[Dict[str, float]]) -> None:
        if not self.cache_path:
            return
        if not rows:
            return
        df_new = pd.DataFrame(rows)
        if self.cache_path.exists():
            conn = duckdb.connect()
            existing = conn.execute(f"SELECT * FROM read_parquet('{self.cache_path}')").df()
            conn.close()
            if not existing.empty:
                combined = pd.concat([existing, df_new], ignore_index=True)
                combined = combined.drop_duplicates(subset=["name_a", "name_b"], keep="last")
            else:
                combined = df_new
        else:
            combined = df_new
        combined = combined.reset_index(drop=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect()
        conn.register("candidate_features_df", combined)
        conn.execute(
            f"COPY candidate_features_df TO '{self.cache_path}' (FORMAT 'parquet', COMPRESSION 'ZSTD')"
        )
        conn.close()


def prepare_candidate_data(
    names: Sequence[str],
    representative_ids: Dict[str, Optional[str]],
    cache_path: Optional[Path],
    token_overlap_threshold: int,
    jaro_threshold: float,
    initial_chunk_size: Optional[int],
    ann_retriever: Optional[AnnRetriever] = None,
) -> Tuple[CandidateCacheManager, Dict[Tuple[str, str], np.ndarray]]:
    feature_cache: Dict[Tuple[str, str], np.ndarray] = {}
    manager = CandidateCacheManager(
        names,
        representative_ids,
        cache_path,
        token_overlap_threshold,
        jaro_threshold,
        feature_cache,
        ann_retriever=ann_retriever,
    )

    if cache_path and cache_path.exists():
        conn = duckdb.connect()
        df = conn.execute(f"SELECT * FROM read_parquet('{cache_path}')").df()
        conn.close()
        if df.empty:
            print(f"Candidate cache at {cache_path} was empty; regenerating.")
            try:
                cache_path.unlink()
            except FileNotFoundError:
                pass
        else:
            feature_cols = sorted(
                [col for col in df.columns if col.startswith("feature_")],
                key=lambda col: int(col.split("_")[1]),
            )
            if len(feature_cols) != len(FEATURE_NAMES):
                raise ValueError(
                    f"Cached candidate features expect {len(feature_cols)} features but pipeline has {len(FEATURE_NAMES)}. Regenerate the cache."
                )
            loaded_pairs: List[Tuple[PairCandidate, np.ndarray]] = []
            for row in df.itertuples(index=False):
                score = float(getattr(row, "score", getattr(row, feature_cols[0], 0.0)))
                features = np.array([getattr(row, col) for col in feature_cols], dtype=float)
                candidate = PairCandidate(row.name_a, row.name_b, score)
                loaded_pairs.append((candidate, features))
            filtered_keys = {
                (cand.name_a, cand.name_b)
                for cand in filter_candidates_by_ground_truth(
                    [cand for cand, _ in loaded_pairs],
                    representative_ids,
                )
            }
            usable_pairs = [
                (cand, features)
                for cand, features in loaded_pairs
                if (cand.name_a, cand.name_b) in filtered_keys
            ]
            if not usable_pairs:
                print(
                    f"Cached candidates at {cache_path} did not overlap the current dataset; regenerating."
                )
                try:
                    cache_path.unlink()
                except FileNotFoundError:
                    pass
            else:
                manager.bootstrap_pairs(usable_pairs)
                print(f"Loaded {len(usable_pairs):,} cached candidate pairs from {cache_path}.")

    initial_size = initial_chunk_size or len(names)
    if initial_size <= 0:
        initial_size = len(names)
    initial_size = min(max(2, initial_size), len(names))
    if initial_size > 1:
        initial_subset = names[:initial_size]
        initial_candidates = manager.ensure_subset(
            initial_subset,
            max_candidates=_scaled_candidate_limit(len(initial_subset), 200, minimum=10_000),
        )
        print(
            f"Prepared {len(initial_candidates):,} candidate pairs for the initial chunk of {initial_size} representative names."
        )

    return manager, feature_cache

def diagnose_candidate_gap(
    subset: Sequence[str],
    candidates: Sequence[PairCandidate],
    token_overlap_threshold: int,
    jaro_threshold: float,
) -> None:
    total = len(subset)
    print(
        "Candidate diagnostics:\n"
        f"  subset size: {total}\n"
        f"  token overlap threshold: {token_overlap_threshold}\n"
        f"  jaro threshold: {jaro_threshold}\n"
        f"  current candidates: {len(candidates)}"
    )
    cleaned = [normalize_name(name) for name in subset]
    generic_flags = [is_generic_name(clean) for clean in cleaned]
    generic_count = sum(generic_flags)
    token_sets = [set(clean.split()) for clean in cleaned]
    with_tokens = sum(1 for tokens in token_sets if tokens)
    print(
        f"  generic names: {generic_count} ({generic_count / total:.2%})\n"
        f"  names with tokens: {with_tokens} ({with_tokens / total:.2%})"
    )
    token_counter: Counter[str] = Counter()
    for tokens in token_sets:
        token_counter.update(token for token in tokens if len(token) > 1)
    common = token_counter.most_common(10)
    if common:
        print("  top shared tokens:")
        for token, freq in common:
            print(f"    {token!r}: {freq}")
    else:
        print("  no shared tokens found in subset.")


def collect_initial_labels(
    candidates: List[PairCandidate],
    to_label: int,
    archive_path: Optional[Path],
    existing_labelled: Optional[Dict[Tuple[str, str], LabelledExample]] = None,
) -> Dict[Tuple[str, str], LabelledExample]:
    existing_keys = set(existing_labelled.keys()) if existing_labelled else set()
    new_labels: Dict[Tuple[str, str], LabelledExample] = {}
    labelled_count = 0
    for candidate in candidates:
        confirmed_generics = process_pending_generic_confirmations()
        if confirmed_generics:
            remove_generic_labels(existing_labelled, confirmed_generics)
            remove_generic_labels(new_labels, confirmed_generics)
            existing_keys = set(existing_labelled.keys()) if existing_labelled else set()
        confirmed_non_degree = process_pending_nondegree_confirmations()
        if confirmed_non_degree:
            remove_nondegree_labels(existing_labelled, confirmed_non_degree)
            remove_nondegree_labels(new_labels, confirmed_non_degree)
            existing_keys = set(existing_labelled.keys()) if existing_labelled else set()
        if labelled_count >= to_label:
            break
        key = (candidate.name_a, candidate.name_b)
        if (
            is_generic_name(candidate.name_a)
            or is_generic_name(candidate.name_b)
            or is_non_degree_program_name(candidate.name_a)
            or is_non_degree_program_name(candidate.name_b)
        ):
            continue
        if key in existing_keys or key in new_labels:
            continue
        response = prompt_label(candidate)
        if response.action == "quit":
            print("Stopping labelling.")
            break
        if response.action == "skip":
            continue
        if response.action == "generic":
            chosen = candidate.name_a if response.value == "a" else candidate.name_b
            queue_generic_confirmation(chosen)
            print(f"Queued '{chosen}' for generic confirmation; will prompt on the next step.")
            continue
        if response.action == "nondegree":
            chosen = candidate.name_a if response.value == "a" else candidate.name_b
            queue_nondegree_confirmation(chosen)
            print(f"Queued '{chosen}' for non-degree confirmation; will prompt on the next step.")
            continue
        if response.action == "label":
            label_value = int(response.value) if response.value is not None else None
            if label_value is None:
                continue
            features = compute_features(candidate.name_a, candidate.name_b)
            new_labels[key] = LabelledExample(features, label_value)
            labelled_count += 1
    if not existing_keys and labelled_count == 0:
        raise RuntimeError("No labels were collected; cannot train the model.")
    remaining_generics = process_pending_generic_confirmations()
    if remaining_generics:
        remove_generic_labels(existing_labelled, remaining_generics)
        remove_generic_labels(new_labels, remaining_generics)
    remaining_non_degree = process_pending_nondegree_confirmations()
    if remaining_non_degree:
        remove_nondegree_labels(existing_labelled, remaining_non_degree)
        remove_nondegree_labels(new_labels, remaining_non_degree)
    if new_labels and archive_path:
        combined = dict(existing_labelled or {})
        combined.update(new_labels)
        save_labelled_examples(combined, archive_path)
    return new_labels


def interactive_training(
    candidates: List[PairCandidate],
    initial_labels: int,
    batch_size: int,
    archive_path: Optional[Path],
    max_iterations: int,
    convergence_threshold: float,
    initial_labelled: Optional[Dict[Tuple[str, str], LabelledExample]] = None,
    existing_model: Optional[RandomForestClassifier] = None,
) -> Tuple[RandomForestClassifier, Dict[Tuple[str, str], LabelledExample], bool, bool]:
    labelled: Dict[Tuple[str, str], LabelledExample] = dict(initial_labelled or {})

    filtered_candidates = [
        cand
        for cand in candidates
        if not (
            is_generic_name(cand.name_a)
            or is_generic_name(cand.name_b)
            or is_non_degree_program_name(cand.name_a)
            or is_non_degree_program_name(cand.name_b)
        )
    ]
    labels_added = False
    sample_feature_length: Optional[int] = None
    for cand in filtered_candidates:
        sample_feature_length = compute_features(cand.name_a, cand.name_b).shape[0]
        break
    if sample_feature_length is None:
        raise RuntimeError("No candidates available for interactive training.")

    model = _ensure_model_compatibility(existing_model, sample_feature_length, "interactive training")

    remaining_initial = max(0, initial_labels - len(labelled))
    if remaining_initial > 0:
        new_initial_labels = collect_initial_labels(
            filtered_candidates,
            remaining_initial,
            archive_path,
            existing_labelled=labelled,
        )
        labelled.update(new_initial_labels)
        if new_initial_labels:
            labels_added = True

    if not labelled:
        raise RuntimeError("No labels available to train the model.")

    if model is None or remaining_initial > 0:
        model = fit_classifier(labelled.values())

    filtered_candidates = [
        cand
        for cand in filtered_candidates
        if not (
            is_generic_name(cand.name_a)
            or is_generic_name(cand.name_b)
            or is_non_degree_program_name(cand.name_a)
            or is_non_degree_program_name(cand.name_b)
        )
    ]
    remaining_candidates = list(filtered_candidates)
    confirmed_generics = process_pending_generic_confirmations()
    if confirmed_generics:
        remove_generic_labels(labelled, confirmed_generics)
        if not labelled:
            raise RuntimeError("All labels were removed after marking names as generic.")
        model = fit_classifier(labelled.values())
        remaining_candidates = [
            cand
            for cand in remaining_candidates
            if cand.name_a not in confirmed_generics and cand.name_b not in confirmed_generics
        ]
    confirmed_non_degree = process_pending_nondegree_confirmations()
    if confirmed_non_degree:
        remove_nondegree_labels(labelled, confirmed_non_degree)
        if not labelled:
            raise RuntimeError("All labels were removed after marking names as non-degree.")
        model = fit_classifier(labelled.values())
        remaining_candidates = [
            cand
            for cand in remaining_candidates
            if cand.name_a not in confirmed_non_degree and cand.name_b not in confirmed_non_degree
        ]
    prev_scores: Dict[Tuple[str, str], float] = {
        (cand.name_a, cand.name_b): prob
        for cand, prob in score_candidates(model, remaining_candidates, labelled.keys())
    }
    converged = False

    for iteration in range(1, max_iterations + 1):
        newly_confirmed = process_pending_generic_confirmations()
        if newly_confirmed:
            remove_generic_labels(labelled, newly_confirmed)
            if not labelled:
                raise RuntimeError("All labels were removed after marking names as generic.")
            model = fit_classifier(labelled.values())
            remaining_candidates = [
                cand
                for cand in remaining_candidates
                if cand.name_a not in newly_confirmed and cand.name_b not in newly_confirmed
            ]
            if not remaining_candidates:
                print("No candidate pairs remain after filtering flagged names.")
                break
            prev_scores = {
                (cand.name_a, cand.name_b): prob
                for cand, prob in score_candidates(model, remaining_candidates, labelled.keys())
            }
        newly_confirmed_non_degree = process_pending_nondegree_confirmations()
        if newly_confirmed_non_degree:
            remove_nondegree_labels(labelled, newly_confirmed_non_degree)
            if not labelled:
                raise RuntimeError("All labels were removed after marking names as non-degree.")
            model = fit_classifier(labelled.values())
            remaining_candidates = [
                cand
                for cand in remaining_candidates
                if cand.name_a not in newly_confirmed_non_degree and cand.name_b not in newly_confirmed_non_degree
            ]
            if not remaining_candidates:
                print("No candidate pairs remain after filtering non-degree names.")
                break
            prev_scores = {
                (cand.name_a, cand.name_b): prob
                for cand, prob in score_candidates(model, remaining_candidates, labelled.keys())
            }
        if not remaining_candidates:
            print("No candidate pairs remain after filtering flagged names.")
            break
        if not prev_scores:
            print("No more unlabelled candidates to review.")
            break

        scored = sorted(
            prev_scores.items(),
            key=lambda item: abs(0.5 - item[1]),
        )
        to_inspect = scored[:batch_size]

        print(f"\nIteration {iteration}: reviewing {len(to_inspect)} candidate pairs (enter q to finish):")
        any_new_labels = False
        for (name_a, name_b), probability in to_inspect:
            if is_generic_name(name_a) or is_generic_name(name_b):
                continue
            candidate = PairCandidate(name_a, name_b, probability)
            response = prompt_label(candidate)
            if response.action == "quit":
                print("Stopping labelling loop.")
                return model, labelled, converged, labels_added
            if response.action == "skip":
                continue
            if response.action == "generic":
                chosen = candidate.name_a if response.value == "a" else candidate.name_b
                queue_generic_confirmation(chosen)
                print(f"Queued '{chosen}' for generic confirmation; will prompt on the next step.")
                continue
            if response.action == "nondegree":
                chosen = candidate.name_a if response.value == "a" else candidate.name_b
                queue_nondegree_confirmation(chosen)
                print(f"Queued '{chosen}' for non-degree confirmation; will prompt on the next step.")
                continue
            if response.action == "label":
                label_value = int(response.value) if response.value is not None else None
                if label_value is None:
                    continue
                features = compute_features(name_a, name_b)
                labelled[(name_a, name_b)] = LabelledExample(features, label_value)
                any_new_labels = True
                labels_added = True
                save_labelled_examples(labelled, archive_path)

        if not any_new_labels:
            print("No new labels collected; ending loop.")
            break

        model = fit_classifier(labelled.values())
        current_scores = {
            (cand.name_a, cand.name_b): prob
            for cand, prob in score_candidates(model, remaining_candidates, labelled.keys())
        }

        shared_keys = set(prev_scores.keys()) & set(current_scores.keys())
        if shared_keys:
            deltas = [abs(current_scores[k] - prev_scores[k]) for k in shared_keys]
            max_delta = max(deltas)
            print(f"Max probability change this iteration: {max_delta:.4f}")
            if max_delta <= convergence_threshold:
                if max_delta == 0.0:
                    print("No probability change detected after the latest labels; continuing without declaring convergence.")
                else:
                    print("Convergence threshold reached; stopping iterative labelling.")
                    prev_scores = current_scores
                    converged = True
                    break
        prev_scores = current_scores

        cont = input("Continue with another batch? [Y/n]: ").strip().lower()
        if cont in {"n", "no"}:
            break

    process_pending_generic_confirmations()
    process_pending_nondegree_confirmations()
    return model, labelled, converged, labels_added


def filter_candidates_by_ground_truth(
    candidates: List[PairCandidate],
    representative_ids: Dict[str, Optional[str]],
) -> List[PairCandidate]:
    if not representative_ids:
        return candidates
    filtered: List[PairCandidate] = []
    for candidate in candidates:
        id_a = representative_ids.get(candidate.name_a)
        id_b = representative_ids.get(candidate.name_b)
        if id_a and id_b and id_a != id_b:
            continue
        filtered.append(candidate)
    return filtered


def progressive_training(
    names: Sequence[str],
    representative_ids: Dict[str, Optional[str]],
    initial_labels: int,
    batch_size: int,
    archive_path: Optional[Path],
    max_iterations: int,
    convergence_threshold: float,
    initial_chunk_size: int,
    chunk_step_size: int,
    max_chunk_iterations: Optional[int],
    token_overlap_threshold: int,
    jaro_threshold: float,
    candidate_cache: Optional[CandidateCacheManager] = None,
    initial_labelled: Optional[Dict[Tuple[str, str], LabelledExample]] = None,
    existing_model: Optional[RandomForestClassifier] = None,
) -> Tuple[RandomForestClassifier, Dict[Tuple[str, str], LabelledExample]]:
    if not names:
        raise RuntimeError("No names provided for progressive training.")

    chunk_step = max(1, chunk_step_size)
    chunk_size = min(max(2, initial_chunk_size), len(names))
    labelled: Dict[Tuple[str, str], LabelledExample] = dict(initial_labelled or {})
    model: Optional[RandomForestClassifier] = existing_model
    chunk_index = 0

    while True:
        if max_chunk_iterations is not None and chunk_index >= max_chunk_iterations:
            print("Reached maximum number of chunk iterations; stopping progressive training.")
            break
        subset_size = min(chunk_size, len(names))
        if subset_size < 2:
            print("Chunk contains fewer than two names; expanding chunk size.")
            if subset_size >= len(names):
                break
            chunk_size = min(chunk_size + chunk_step, len(names))
            continue

        if chunk_index == 0:
            print(f"Starting progressive training with {subset_size} / {len(names)} representative names.")
        else:
            print(f"Progressive training using {subset_size} / {len(names)} representative names.")

        subset = names[:subset_size]
        if candidate_cache is not None:
            subset_candidates = candidate_cache.ensure_subset(
                subset,
                max_candidates=_scaled_candidate_limit(len(subset), 500),
            )
            if len(subset_candidates) < max(1, len(subset) // 5):
                extra_candidates = generate_pair_candidates(
                    subset,
                    max_candidates=_scaled_candidate_limit(len(subset), 200),
                    token_overlap_threshold=token_overlap_threshold,
                    jaro_threshold=jaro_threshold,
                )
                candidate_cache.add_candidates(extra_candidates)
                subset_candidates = candidate_cache.ensure_subset(subset)
            candidates = subset_candidates
        else:
            subset_candidates = generate_pair_candidates(
                subset,
                max_candidates=_scaled_candidate_limit(len(subset), 500),
                token_overlap_threshold=token_overlap_threshold,
                jaro_threshold=jaro_threshold,
            )
            candidates = filter_candidates_by_ground_truth(
                subset_candidates,
                representative_ids,
            )
        print(f"Chunk {chunk_index + 1}: generated {len(candidates):,} candidate pairs from {subset_size} names.")
        if not candidates:
            diagnose_candidate_gap(
                subset,
                [],
                token_overlap_threshold,
                jaro_threshold,
            )
            if subset_size >= len(names):
                print("No candidate pairs available even with the full dataset.")
                break
            print("No candidate pairs found in current chunk; expanding dataset.")
            chunk_size = min(chunk_size + chunk_step, len(names))
            continue

        chunk_index += 1
        print(f"\n=== Interactive chunk {chunk_index}: {subset_size} representative names ===")

        before_label_count = len(labelled)
        model, updated_labelled, converged, chunk_labels_added = interactive_training(
            candidates,
            initial_labels,
            batch_size,
            archive_path,
            max_iterations,
            convergence_threshold,
            initial_labelled=labelled,
            existing_model=model,
        )
        labelled = updated_labelled

        if subset_size >= len(names):
            break

        if converged:
            if chunk_labels_added or len(labelled) > before_label_count:
                print(
                    "Model converged on current chunk but new labels were collected; expanding to the next chunk."
                )
                chunk_size = min(chunk_size + chunk_step, len(names))
            else:
                print(
                    "Model converged on current chunk with no new labels; moving to the full dataset for final training."
                )
                chunk_size = len(names)
        else:
            chunk_size = min(chunk_size + chunk_step, len(names))

    if not labelled:
        raise RuntimeError("No labels were collected; cannot train the model.")

    if model is None:
        model = fit_classifier(labelled.values())
    return model, labelled


def cluster_from_candidate_cache_stream(
    names: Sequence[str],
    model: RandomForestClassifier,
    threshold: float,
    cache_path: Path,
    labelled_examples: Optional[Dict[Tuple[str, str], LabelledExample]],
    chunk_size: int = STREAMING_SCORING_CHUNK_SIZE,
) -> Tuple[Dict[str, List[str]], RandomForestClassifier]:
    if chunk_size <= 0:
        chunk_size = STREAMING_SCORING_CHUNK_SIZE
    if not cache_path.exists():
        raise FileNotFoundError(f"Candidate cache {cache_path} does not exist for streaming scoring.")

    conn = duckdb.connect()
    total_pairs = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{cache_path}')").fetchone()[0]
    print(f"Total candidate pairs for clustering: {total_pairs:,}")
    print(f"Streaming final candidate scoring from {cache_path} in batches of {chunk_size:,} pairs.")

    schema_cursor = conn.execute(f"SELECT * FROM read_parquet('{cache_path}') LIMIT 0")
    columns = [desc[0] for desc in schema_cursor.description]
    feature_cols = sorted(
        [col for col in columns if col.startswith('feature_')],
        key=lambda col: int(col.split('_')[1]),
    )

    use_cached_features = bool(feature_cols)
    if use_cached_features:
        try:
            model = _ensure_model_with_refit(
                model,
                len(feature_cols),
                'final clustering (cached features)',
                labelled_examples,
            )
        except RuntimeError as exc:
            print(exc)
            print('Falling back to computing features during streaming.')
            use_cached_features = False

    model_checked = use_cached_features

    select_cols = ['name_a', 'name_b']
    if use_cached_features:
        select_cols.extend(feature_cols)

    cursor = conn.execute(
        f"SELECT {', '.join(select_cols)} FROM read_parquet('{cache_path}')"
    )

    name_set = set(names)
    name_to_id = {name: idx for idx, name in enumerate(names)}
    id_to_name = list(names)
    uf = StreamingUnionFind(len(names))
    processed = 0
    chunks_processed = 0

    while True:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break

        chunk = pd.DataFrame(rows, columns=select_cols)
        if chunk.empty:
            continue

        mask = chunk["name_a"].isin(name_set) & chunk["name_b"].isin(name_set)
        if not mask.all():
            dropped = int((~mask).sum())
            if dropped:
                print(f"  skipping {dropped} cached pairs that are not in the current dataset.")
        chunk = chunk.loc[mask]
        if chunk.empty:
            processed += len(rows)
            continue
        name_a_vals = chunk['name_a'].astype(str).tolist()
        name_b_vals = chunk['name_b'].astype(str).tolist()

        if use_cached_features:
            feature_matrix = chunk[feature_cols].to_numpy(dtype=float, copy=False)
        else:
            feature_matrix = np.vstack(
                [compute_features(a, b) for a, b in zip(name_a_vals, name_b_vals)]
            )
            if feature_matrix.size == 0:
                processed += len(rows)
                continue
            if not model_checked:
                try:
                    model = _ensure_model_with_refit(
                        model,
                        feature_matrix.shape[1],
                        'final clustering',
                        labelled_examples,
                    )
                except RuntimeError:
                    conn.close()
                    raise
                model_checked = True

        probs = _match_probabilities(model, feature_matrix)
        for idx, probability in enumerate(probs):
            if probability >= threshold:
                idx_a = name_to_id.get(name_a_vals[idx])
                idx_b = name_to_id.get(name_b_vals[idx])
                if idx_a is None or idx_b is None:
                    continue
                uf.union(idx_a, idx_b)

        processed += len(rows)
        chunks_processed += 1
        if chunks_processed % 10 == 0:
            uf.compress()
        if processed >= total_pairs or len(rows) < chunk_size:
            print(f"  processed {processed:,} candidate pairs…", flush=True)
        elif processed % (10 * chunk_size) == 0:
            print(f"  processed {processed:,} candidate pairs…", flush=True)

    conn.close()
    print(f"Finished streaming {processed:,} candidate pairs.")
    uf.compress()
    return uf.groups(id_to_name), model


def cluster_from_candidate_list_streaming(
    names: Sequence[str],
    model: RandomForestClassifier,
    candidates: Sequence[PairCandidate],
    threshold: float,
    batch_size: int = STREAMING_SCORING_CHUNK_SIZE,
) -> Dict[str, List[str]]:
    if batch_size <= 0:
        batch_size = STREAMING_SCORING_CHUNK_SIZE
    name_to_id = {name: idx for idx, name in enumerate(names)}
    uf = StreamingUnionFind(len(names))
    feature_batch: List[np.ndarray] = []
    pair_batch: List[Tuple[int, int]] = []
    chunks_processed = 0

    def _flush_batch() -> None:
        nonlocal feature_batch, pair_batch, chunks_processed
        if not pair_batch:
            return
        feature_matrix = np.vstack(feature_batch)
        probs = _match_probabilities(model, feature_matrix)
        for (idx_a, idx_b), probability in zip(pair_batch, probs):
            if probability >= threshold:
                uf.union(idx_a, idx_b)
        chunks_processed += 1
        if chunks_processed % 10 == 0:
            uf.compress()
        feature_batch = []
        pair_batch = []

    for cand in candidates:
        idx_a = name_to_id.get(cand.name_a)
        idx_b = name_to_id.get(cand.name_b)
        if idx_a is None or idx_b is None:
            continue
        feature_batch.append(compute_features(cand.name_a, cand.name_b))
        pair_batch.append((idx_a, idx_b))
        if len(pair_batch) >= batch_size:
            _flush_batch()

    _flush_batch()
    uf.compress()
    return uf.groups(list(names))

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive fuzzy string matching for university clustering.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet containing university names.")
    parser.add_argument("--column", type=str, default=None, help="Column name containing university names.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for dry runs.")
    parser.add_argument("--initial-labels", type=int, default=15, help="Number of top candidates to label before training.")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of uncertain pairs to review per iteration.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of human-in-the-loop refinement iterations.",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.01,
        help="Stop early when max probability change falls below this value.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.6,
        help="Probability threshold for assigning pairs to the same cluster.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save clustered assignments as CSV.",
    )
    parser.add_argument(
        "--save-labels",
        type=Path,
        default=DEFAULT_ARCHIVE_PATH,
        help="Path to persist collected training labels (default: ~/.adaptive_fuzzy/label_history.csv).",
    )
    parser.add_argument(
        "--name-labels",
        "--name_labels",
        type=str,
        default=str(DEFAULT_NAME_LABELS_PATH),
        help="Path to persist per-name training labels for generic/non-degree models (use 'none' to disable).",
    )
    parser.add_argument(
        "--load-generic-model",
        "--load_generic_model",
        type=str,
        default=None,
        help="Path to load a trained generic-name classifier.",
    )
    parser.add_argument(
        "--load-nondegree-model",
        "--load_nondegree_model",
        type=str,
        default=None,
        help="Path to load a trained non-degree classifier.",
    )
    parser.add_argument(
        "--train-generic-model",
        "--train_generic_model",
        type=str,
        default=None,
        help="Train a generic-name classifier from stored labels and save it to this path.",
    )
    parser.add_argument(
        "--train-nondegree-model",
        "--train_nondegree_model",
        type=str,
        default=None,
        help="Train a non-degree classifier from stored labels and save it to this path.",
    )
    parser.add_argument(
        "--generic-threshold",
        "--generic_threshold",
        type=float,
        default=0.8,
        help="Probability threshold for treating a name as generic when using a classifier.",
    )
    parser.add_argument(
        "--non-degree-threshold",
        "--non_degree_threshold",
        type=float,
        default=0.8,
        help="Probability threshold for treating a name as non-degree when using a classifier.",
    )
    parser.add_argument(
        "--no-composite-split",
        action="store_true",
        help="Disable splitting composite institution entries (e.g., semicolon-delimited lists).",
    )
    parser.add_argument(
        "--composite-report-limit",
        type=int,
        default=5,
        help="Number of composite name splits to print for review (0 to suppress details).",
    )
    parser.add_argument(
        "--bootstrap-generic-negatives",
        action="store_true",
        help="Infer non-generic name labels from existing pair labels before training the generic classifier.",
    )
    parser.add_argument(
        "--bootstrap-generic-limit",
        type=int,
        default=200,
        help="Maximum number of inferred non-generic names to add when bootstrapping (default: 200; use 0 for no limit).",
    )
    parser.add_argument(
        "--bootstrap-generic-min-count",
        type=int,
        default=2,
        help="Minimum number of labelled pair occurrences required before bootstrapping a non-generic name (default: 2).",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional fraction (0-1] to randomly subsample input rows, e.g. 0.01 for 1%%.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Optional random seed to make subsampling reproducible.",
    )
    parser.add_argument(
        "--initial-chunk-size",
        type=int,
        default=1000,
        help="Number of representative names to include in the first interactive chunk.",
    )
    parser.add_argument(
        "--chunk-step-size",
        type=int,
        default=1000,
        help="How many additional representative names to add after each chunk until convergence.",
    )
    parser.add_argument(
        "--candidate-cache",
        "--candidate_cache",
        type=Path,
        default=None,
        help="Path to cache candidate pairs and features as Parquet for reuse.",
    )
    parser.add_argument(
        "--names-cache",
        "--names_cache",
        type=Path,
        default=None,
        help="Optional cache file (parquet) to store/reuse the cleaned unique names.",
    )
    parser.add_argument(
        "--ann-embeddings",
        type=Path,
        default=None,
        help="Path to persist ANN embedding memmap (Stage 2).",
    )
    parser.add_argument(
        "--build-ann-embeddings",
        action="store_true",
        help="Generate ANN embeddings for the current representative names and store them to --ann-embeddings.",
    )
    parser.add_argument(
        "--ann-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model to use for ANN embeddings.",
    )
    parser.add_argument(
        "--ann-batch-size",
        type=int,
        default=512,
        help="Batch size for ANN embedding generation.",
    )
    parser.add_argument(
        "--ann-use-gpu",
        action="store_true",
        help="Encode ANN embeddings with GPU acceleration when available.",
    )
    parser.add_argument(
        "--ann-index",
        type=Path,
        default=None,
        help="Path to persist the FAISS ANN index.",
    )
    parser.add_argument(
        "--build-ann-index",
        action="store_true",
        help="Build a FAISS IVF-PQ index over the ANN embeddings.",
    )
    parser.add_argument(
        "--ann-nlist",
        type=int,
        default=4096,
        help="Number of clusters (nlist) for the IVF-PQ index.",
    )
    parser.add_argument(
        "--ann-m",
        type=int,
        default=48,
        help="Number of subquantizers (m) for the IVF-PQ index.",
    )
    parser.add_argument(
        "--ann-nbits",
        type=int,
        default=8,
        help="Number of bits per sub-vector (nbits) for the IVF-PQ index.",
    )
    parser.add_argument(
        "--ann-training-sample",
        type=int,
        default=100_000,
        help="Number of vectors to sample when training the ANN index (defaults to 100k).",
    )
    parser.add_argument(
        "--ann-random-seed",
        type=int,
        default=42,
        help="Random seed for ANN embedding/index reproducibility.",
    )
    parser.add_argument(
        "--use-ann-retrieval",
        action="store_true",
        help="Use ANN retrieval for candidate generation (requires embeddings and index).",
    )
    parser.add_argument(
        "--ann-top-k",
        type=int,
        default=50,
        help="Number of nearest neighbours to request per name when using ANN retrieval.",
    )
    parser.add_argument(
        "--ann-nprobe",
        type=int,
        default=32,
        help="Number of inverted lists to probe during ANN search (higher improves recall at a cost).",
    )
    parser.add_argument(
        "--candidate-overlap",
        "--candidate_overlap",
        type=int,
        default=1,
        help="Minimum shared token count required to consider a pair (default: 1).",
    )
    parser.add_argument(
        "--candidate-jaro",
        "--candidate_jaro",
        type=float,
        default=0.88,
        help="Jaro similarity threshold (0-1) for candidate generation (default: 0.88).",
    )
    parser.add_argument(
        "--max-chunk-iterations",
        "--max_chunk_iterations",
        type=int,
        default=10,
        help="Maximum number of progressive chunk iterations (default: unlimited).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of representative names before progressive training.",
    )
    parser.add_argument(
        "--load-model",
        "--load_model",
        type=Path,
        default=None,
        help="Path to a previously saved classifier to reuse.",
    )
    parser.add_argument(
        "--save-model",
        "--save_model",
        type=Path,
        default=None,
        help="Path to persist the trained classifier.",
    )
    parser.add_argument(
        "--resume-training",
        "--resume_training",
        action="store_true",
        help="Continue interactive training even when a saved model is supplied.",
    )
    parser.add_argument(
        "--summarize-model",
        "--summarize_model",
        action="store_true",
        help="Print a summary of the fitted model (structure, feature importances).",
    )
    return parser.parse_args(argv)


def _main_impl(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    global GENERIC_NAME_MODEL, NON_DEGREE_NAME_MODEL, GENERIC_MODEL_THRESHOLD, NON_DEGREE_MODEL_THRESHOLD

    if args.candidate_overlap < 0:
        print("candidate-overlap must be non-negative; defaulting to 0.")
        args.candidate_overlap = 0
    args.candidate_jaro = float(min(0.99, max(0.0, args.candidate_jaro)))

    GENERIC_MODEL_THRESHOLD = float(min(max(args.generic_threshold, 0.0), 1.0))
    NON_DEGREE_MODEL_THRESHOLD = float(min(max(args.non_degree_threshold, 0.0), 1.0))

    name_label_path = _optional_path(args.name_labels)
    _configure_name_label_store(name_label_path)
    if NAME_LABEL_STORE_PATH:
        print(f"Using name label store {NAME_LABEL_STORE_PATH}")

    ann_embeddings_path = _optional_path(args.ann_embeddings)
    ann_index_path = _optional_path(args.ann_index)
    ann_retriever: Optional[AnnRetriever] = None

    if args.bootstrap_generic_negatives:
        bootstrap_max = args.bootstrap_generic_limit
        max_names = None if bootstrap_max is None or bootstrap_max <= 0 else bootstrap_max
        added = _bootstrap_generic_negative_labels_from_pairs(
            args.save_labels,
            max(1, args.bootstrap_generic_min_count),
            max_names,
            args.sample_seed,
        )
        if added:
            print("Inferred additional non-generic labels to support generic model training.")

    generic_model = _load_or_train_name_model(
        _optional_path(args.load_generic_model),
        _optional_path(args.train_generic_model),
        "generic",
        "generic-name",
    )
    try:
        set_generic_name_model(generic_model, GENERIC_MODEL_THRESHOLD)
    except ValueError as exc:
        print(f"Warning: {exc}")
        generic_model = None
        set_generic_name_model(None, GENERIC_MODEL_THRESHOLD)
    GENERIC_NAME_MODEL = generic_model

    non_degree_model = _load_or_train_name_model(
        _optional_path(args.load_nondegree_model),
        _optional_path(args.train_nondegree_model),
        "non_degree",
        "non-degree",
    )
    try:
        set_non_degree_name_model(non_degree_model, NON_DEGREE_MODEL_THRESHOLD)
    except ValueError as exc:
        print(f"Warning: {exc}")
        non_degree_model = None
        set_non_degree_name_model(None, NON_DEGREE_MODEL_THRESHOLD)
    NON_DEGREE_NAME_MODEL = non_degree_model

    raw_names = load_names(
        args.input,
        args.column,
        args.limit,
        args.sample_frac,
        args.sample_seed,
        args.names_cache,
    )
    if len(raw_names) < 2:
        print("Need at least two unique names for clustering.")
        return

    composite_split_log: List[Tuple[str, List[str]]] = []
    representative_names, representative_members, representative_ids = prepare_name_groups(
        raw_names,
        composite_log=None if args.no_composite_split else composite_split_log,
        enable_composite_split=not args.no_composite_split,
    )
    raw_name_metadata = get_raw_name_metadata()
    if composite_split_log:
        total_segments = sum(len(parts) for _, parts in composite_split_log)
        print(
            f"Identified {len(composite_split_log)} composite name entries; expanded into {total_segments} individual names."
        )
        detail_limit = max(0, args.composite_report_limit)
        if detail_limit > 0:
            for original, parts in composite_split_log[:detail_limit]:
                print(f"  Composite: {original}")
                for part in parts:
                    print(f"    -> {part}")
            if len(composite_split_log) > detail_limit:
                hidden = len(composite_split_log) - detail_limit
                print(
                    f"  ... {hidden} additional composite entries suppressed; adjust --composite-report-limit to see more."
                )
    if len(representative_names) < 2:
        print("Need at least two unique names after cleaning for clustering.")
        return

    ann_reference_names = list(representative_names)
    if args.build_ann_embeddings:
        if ann_embeddings_path is None:
            raise ValueError("Specify --ann-embeddings to store ANN embeddings when using --build-ann-embeddings.")
        device = "cuda" if args.ann_use_gpu else None
        print(
            f"Generating ANN embeddings for {len(ann_reference_names):,} representative names using {args.ann_model}…"
        )
        generate_ann_embeddings(
            ann_reference_names,
            ann_embeddings_path,
            model_name=args.ann_model,
            batch_size=max(1, args.ann_batch_size),
            device=device,
        )
        print(f"ANN embeddings written to {ann_embeddings_path}")

    if args.build_ann_index:
        if ann_embeddings_path is None or not ann_embeddings_path.exists():
            raise FileNotFoundError(
                "ANN embeddings not found; generate them with --build-ann-embeddings before building the index."
            )
        if ann_index_path is None:
            raise ValueError("Specify --ann-index to store the FAISS index when using --build-ann-index.")
        print(
            f"Building FAISS IVF-PQ index at {ann_index_path} (nlist={args.ann_nlist}, m={args.ann_m}, nbits={args.ann_nbits})…"
        )
        build_faiss_index(
            ann_embeddings_path,
            ann_index_path,
            nlist=max(1, args.ann_nlist),
            m=max(1, args.ann_m),
            nbits=max(1, args.ann_nbits),
            training_samples=max(1, args.ann_training_sample),
            random_seed=args.ann_random_seed,
        )
        print(f"ANN index saved to {ann_index_path}")

    if args.use_ann_retrieval:
        if ann_embeddings_path is None or not ann_embeddings_path.exists():
            raise FileNotFoundError(
                "ANN embeddings not found; provide --ann-embeddings pointing to a generated embedding store."
            )
        if ann_index_path is None or not ann_index_path.exists():
            raise FileNotFoundError(
                "ANN index not found; provide --ann-index pointing to a FAISS index generated for this dataset."
            )
        ann_retriever = AnnRetriever(
            ann_reference_names,
            ann_embeddings_path,
            ann_index_path,
            top_k=max(1, args.ann_top_k),
            nprobe=max(1, args.ann_nprobe),
        )
        print(
            f"Using ANN retrieval for candidate generation (top_k={ann_retriever.top_k}, nprobe={ann_retriever.nprobe})."
        )

    print(f"Loaded {len(raw_names)} unique university names.")
    if len(representative_names) != len(raw_names):
        print(
            f"Name cleaning reduced the working set to {len(representative_names)} representative names."
        )
    set_token_statistics_from_names(representative_names)
    if not args.no_shuffle:
        rng = (
            np.random.default_rng(args.sample_seed)
            if args.sample_seed is not None
            else np.random.default_rng()
        )
        rng.shuffle(representative_names)
    set_feature_cache(None)
    candidate_cache_path = args.candidate_cache
    candidate_cache, feature_cache_dict = prepare_candidate_data(
        representative_names,
        representative_ids,
        candidate_cache_path,
        args.candidate_overlap,
        args.candidate_jaro,
        args.initial_chunk_size,
        ann_retriever=ann_retriever,
    )
    existing_labelled = load_labelled_examples(args.save_labels)
    if existing_labelled:
        print(f"Loaded {len(existing_labelled)} saved labelled pairs from {args.save_labels}.")
    rep_name_set = set(representative_names)
    if existing_labelled:
        filtered_existing = {
            key: value
            for key, value in existing_labelled.items()
            if key[0] in rep_name_set and key[1] in rep_name_set
        }
        if len(filtered_existing) != len(existing_labelled):
            print(
                f"Dropped {len(existing_labelled) - len(filtered_existing)} saved labels that do not match the current dataset."
            )
        existing_labelled = filtered_existing
    for (name_a, name_b), example in existing_labelled.items():
        feature_cache_dict[(name_a, name_b)] = example.features
        feature_cache_dict[(name_b, name_a)] = example.features
    set_feature_cache(feature_cache_dict)
    model: Optional[RandomForestClassifier] = None
    loaded_model: Optional[RandomForestClassifier] = None
    if args.load_model:
        if not args.load_model.exists():
            print(f"Provided model path {args.load_model} does not exist.")
            set_feature_cache(None)
            return
        try:
            loaded_model = joblib.load(args.load_model)
            print(f"Loaded model from {args.load_model}.")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to load model from {args.load_model}: {exc}")
            set_feature_cache(None)
            return

    should_train = args.resume_training or loaded_model is None
    labelled: Dict[Tuple[str, str], LabelledExample] = existing_labelled

    if should_train:
        if len(existing_labelled) < args.initial_labels:
            print(f"Preparing to collect {args.initial_labels - len(existing_labelled)} additional initial labels.")
        else:
            print(f"Existing saved labels ({len(existing_labelled)}) meet the initial-label requirement; continuing training.")
        if args.sample_frac is not None:
            print(f"Using a random {args.sample_frac:.2%} sample of the input rows for this run.")
        try:
            model, labelled = progressive_training(
                representative_names,
                representative_ids,
                args.initial_labels,
                args.batch_size,
                args.save_labels,
                args.max_iterations,
                args.convergence_threshold,
                args.initial_chunk_size,
                args.chunk_step_size,
                args.max_chunk_iterations,
                args.candidate_overlap,
                args.candidate_jaro,
                candidate_cache,
                initial_labelled=existing_labelled,
                existing_model=loaded_model,
            )
        except RuntimeError as exc:
            print(exc)
            set_feature_cache(None)
            return
    else:
        model = loaded_model
        if model is None:
            print("No model available. Provide a saved model or enable training.")
            set_feature_cache(None)
            return
        if args.sample_frac is not None:
            print(f"Skipping training; using loaded model with a {args.sample_frac:.2%} sample.")
        else:
            print("Skipping training; using loaded model.")

    print("\nTraining complete. Building clusters...")
    if args.save_model and model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.save_model)
        print(f"Model saved to {args.save_model}")
    restored_generics = set(CONFIRMED_GENERIC_NAMES)
    if restored_generics:
        for name in restored_generics:
            unmark_generic_name(name)
        CONFIRMED_GENERIC_NAMES.clear()
        if PENDING_GENERIC_CONFIRMATIONS:
            PENDING_GENERIC_CONFIRMATIONS.clear()
        print(
            f"Restored {len(restored_generics)} names previously withheld as generic for the final clustering pass."
        )

    total_representatives = len(representative_names)
    streaming_possible = (
        candidate_cache is not None
        and not restored_generics
        and args.candidate_cache is not None
        and args.candidate_cache.exists()
    )

    if streaming_possible and candidate_cache is not None:
        coverage = candidate_cache.coverage_size()
        if coverage < total_representatives:
            print(
                f"Candidate cache covers {coverage:,} / {total_representatives:,} representative names; "
                "expanding before streaming."
            )
            candidate_cache.expand_to_size(
                total_representatives,
                max_candidates=_scaled_candidate_limit(total_representatives, 200),
            )
            coverage = candidate_cache.coverage_size()
            if coverage < total_representatives:
                print(
                    f"Warning: candidate cache still only covers {coverage:,} / {total_representatives:,} names; "
                    "streaming will continue with the available pairs."
                )

    final_candidates: Optional[List[PairCandidate]] = None

    if streaming_possible:
        try:
            clusters, model = cluster_from_candidate_cache_stream(
                representative_names,
                model,
                args.match_threshold,
                args.candidate_cache,
                labelled,
                STREAMING_SCORING_CHUNK_SIZE,
            )
        except RuntimeError as exc:
            print(exc)
            set_feature_cache(None)
            return
    else:
        if candidate_cache is not None and not restored_generics:
            final_candidates = candidate_cache.ensure_subset(
                representative_names,
                max_candidates=_scaled_candidate_limit(len(representative_names), 200),
            )
        else:
            final_candidates = filter_candidates_by_ground_truth(
                generate_pair_candidates(
                    representative_names,
                    max_candidates=_scaled_candidate_limit(len(representative_names), 10),
                    token_overlap_threshold=args.candidate_overlap,
                    jaro_threshold=args.candidate_jaro,
                ),
                representative_ids,
            )
        print(f"Total candidate pairs for clustering: {len(final_candidates):,}")
        if not final_candidates:
            print("Warning: no candidate pairs were generated for clustering.")
            diagnose_candidate_gap(
                representative_names,
                [],
                args.candidate_overlap,
                args.candidate_jaro,
            )
            set_feature_cache(None)
            return
        sample_feature_length = compute_features(
            final_candidates[0].name_a,
            final_candidates[0].name_b,
        ).shape[0]
        try:
            model = _ensure_model_with_refit(
                model,
                sample_feature_length,
                "final clustering",
                labelled,
            )
        except RuntimeError as exc:
            print(exc)
            set_feature_cache(None)
            return
        clusters = cluster_from_candidate_list_streaming(
            representative_names,
            model,
            final_candidates,
            args.match_threshold,
            STREAMING_SCORING_CHUNK_SIZE,
        )

    if args.summarize_model and model is not None:
        print("\nModel summary:")
        print(summarize_random_forest(model, FEATURE_NAMES))

    if 'clusters' not in locals() or clusters is None:
        if not final_candidates:
            print("No candidate pairs generated for final clustering.")
            set_feature_cache(None)
            return
        clusters = cluster_from_candidate_list_streaming(
            representative_names,
            model,
            final_candidates,
            args.match_threshold,
            STREAMING_SCORING_CHUNK_SIZE,
        )

    expanded_clusters: Dict[str, List[str]] = {}
    for root, members in clusters.items():
        seen: Set[str] = set()
        expanded_members: List[str] = []
        for member in members:
            for raw in representative_members.get(member, [member]):
                if raw not in seen:
                    seen.add(raw)
                    expanded_members.append(raw)
        expanded_clusters[root] = expanded_members

    cluster_frame = clusters_to_frame(expanded_clusters, name_metadata=raw_name_metadata)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cluster_frame.to_csv(args.output, index=False)
        print(f"Cluster assignments written to {args.output}")
    else:
        print(cluster_frame.to_string(index=False))

    if args.save_labels:
        save_labelled_examples(labelled, args.save_labels)
        print(f"Saved {len(labelled)} labelled examples to {args.save_labels}")
    set_feature_cache(None)
    return cluster_frame


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        _main_impl(args)
    finally:
        _persist_name_label_store()


def run_adaptive_fuzzy_pipeline(
    *,
    input: Union[str, Path],
    column: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    **overrides: Union[str, int, float, bool, Sequence[Union[str, int, float]]],
) -> pd.DataFrame:
    """Convenience helper to run the adaptive fuzzy CLI from Python code.

    Parameters
    ----------
    input : Path-like
        Path to the CSV/Parquet source of raw names (required).
    column : Optional[str]
        Column name containing the institution names (mirrors --column).
    extra_args : Optional sequence of strings
        Additional CLI-style arguments to append verbatim.
    **overrides :
        Keyword arguments matching CLI flags. For example,
        ``match_threshold=0.7`` becomes ``--match-threshold 0.7`` and
        ``no_composite_split=True`` becomes ``--no-composite-split``.

    Returns
    -------
    pandas.DataFrame
        Cluster assignments with the same schema emitted by the CLI.
    """

    argv: List[str] = ["--input", str(Path(input))]
    if column is not None:
        argv.extend(["--column", column])

    def _append_flag(flag: str, value: Any) -> None:
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, os.PathLike)):
            values = [str(item) for item in value]
            if not values:
                return
            argv.append(flag)
            argv.extend(values)
            return
        if value is None:
            return
        argv.extend([flag, str(value)])

    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        _append_flag(flag, value)

    if extra_args:
        argv.extend(str(arg) for arg in extra_args)

    args = parse_args(argv)
    result = _main_impl(args)
    _persist_name_label_store()
    if result is None:
        raise RuntimeError("Adaptive fuzzy pipeline did not produce any clusters.")
    return result


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
