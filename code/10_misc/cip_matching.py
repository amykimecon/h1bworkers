"""CIP matching utilities extracted from helpers.py"""
from __future__ import annotations

import json
import math
import re
import unicodedata
import itertools
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import jellyfish
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.sparse import csr_matrix
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *

_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-z]+")
_FIELD_STOPWORDS: Tuple[str, ...] = (
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "via",
    "vs",
    "with",
    "within",
    "without",
    "about",
    "across",
    "around",
    "between",
    "through",
    "under",
    "bachelor",
    "bachelors",
    "master",
    "masters",
    "doctor",
    "doctors",
    "doctoral",
    "doctorate",
    "phd",
    "dphil",
    "md",
    "jd",
    "edd",
    "associate",
    "associates",
    "undergraduate",
    "postgraduate",
    "graduate",
    "post",
    "degree",
    "degrees",
    "diploma",
    "diplomas",
    "certificate",
    "certificates",
    "certification",
    "program",
    "programs",
    "programme",
    "programmes",
    "curriculum",
    "curricula",
    "major",
    "majors",
    "minor",
    "minors",
    "concentration",
    "concentrations",
    "specialization",
    "specializations",
    "specialisation",
    "specialisations",
    "specialty",
    "specialties",
    "track",
    "tracks",
    "option",
    "options",
    "focus",
    "focuses",
    "field",
    "fields",
    "area",
    "areas",
    "college",
    "colleges",
    "university",
    "universities",
    "department",
    "departments",
    "dept",
    "faculty",
    "school",
    "schools",
    "academy",
    "academies",
    "institute",
    "institutes",
    "center",
    "centers",
    "centre",
    "centres",
    "bsc",
    "bs",
    "ba",
    "ab",
    "aa",
    "aas",
    "as",
    "bba",
    "bbm",
    "be",
    "bed",
    "beng",
    "btech",
    "bt",
    "llb",
    "llm",
    "ms",
    "msc",
    "ma",
    "meng",
    "mtech",
    "mfa",
    "mph",
    "mphil",
    "mres",
    "mse",
    "med",
    "m.ed",
    "mha",
    "dmd",
    "dds",
    "honours",
    "honors",
    "hon",
    "hons",
    "gpa",
    "cum laude",
)
_NEUTRAL_FIELD_TOKENS: Tuple[str, ...] = ("studies", "general", "program", "programs")
_NEUTRAL_FIELD_TOKEN_SET: Set[str] = set(_NEUTRAL_FIELD_TOKENS)
_MINOR_IN_PATTERN = re.compile(r"\b(minor|concentration)s?\s+(in|on)\s+(.*)", re.IGNORECASE)
_MINOR_PATTERN = re.compile(r"\b(minor|concentration)s?\b", re.IGNORECASE)
_APPROX_SIM_THRESHOLD = 0.84
_SCORE_TOLERANCE = 0.05
_APPROX_WEIGHT = 0.7
_FUZZ_WEIGHT = 0.5
_DEFAULT_TOKEN_WEIGHT = 1.0
_MIN_ACCEPT_SCORE = 1
_FIELD_BATCH_SIZE = 5000
_MAX_STAGE1_CANDIDATES = 100
_FALLBACK_CANDIDATES = 25
_EXTRA_TOKEN_PENALTY = 0.15
_PENALTY_TOKEN_FACTORS: Dict[str, float] = {
    "law": 5,
    "engineering": 4,
    "education": 5,
}
_HARDCODE_INVALID = "::invalid::"
_DEFAULT_MATCH_STORE_PATH = Path(root) / "data/crosswalks/cip/cip_hardcodes_temp.json"
_DEFAULT_HARDCODE_PATH = _DEFAULT_MATCH_STORE_PATH  # Backward compatibility alias
_SPLIT_TOKEN_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("with", re.compile(r"\bwith\b", re.IGNORECASE)),
    ("ampersand", re.compile(r"&")),
    ("slash", re.compile(r"/")),
)
_DIGIT_PATTERNS: Dict[int, re.Pattern[str]] = {
    2: re.compile(r'^\d{1,2}$'),
    4: re.compile(r'^\d{1,2}\.\d{2}$'),
    6: re.compile(r'^\d{1,2}\.\d{4}$'),
}

_DEFAULT_CIP_CACHE_PATH = Path(root) / "data" / "int" / "cip_match_cache.json"
_GLOBAL_CIP_CACHE_PATH: Optional[Path] = None
_GLOBAL_CIP_CACHE: Dict[str, Dict[str, Any]] = {}
_GLOBAL_CIP_CACHE_LOCK = Lock()

_LSH_BAND_SIZE = 2
_LSH_MAX_KEYS = 20
_LSH_MAX_CANDIDATES = 500

__all__ = [
    "match_fields_to_cip",
    "add_hardcoded_cip_match",
]


def _load_persistent_cip_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return {str(key): dict(value) for key, value in data.items() if isinstance(value, dict)}
    except Exception:
        pass
    return {}


def _save_persistent_cip_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _ensure_global_cache(path: Path) -> None:
    global _GLOBAL_CIP_CACHE_PATH, _GLOBAL_CIP_CACHE
    with _GLOBAL_CIP_CACHE_LOCK:
        if _GLOBAL_CIP_CACHE_PATH == path and _GLOBAL_CIP_CACHE:
            return
        cache = _load_persistent_cip_cache(path)
        _GLOBAL_CIP_CACHE_PATH = path
        _GLOBAL_CIP_CACHE = cache


def _cache_lookup(key: Optional[str]) -> Optional[Dict[str, Any]]:
    if key is None:
        return None
    with _GLOBAL_CIP_CACHE_LOCK:
        entry = _GLOBAL_CIP_CACHE.get(key)
        return dict(entry) if entry is not None else None


def _cache_update(key: str, record: Dict[str, Any]) -> None:
    with _GLOBAL_CIP_CACHE_LOCK:
        _GLOBAL_CIP_CACHE[key] = record


def _cache_flush(path: Optional[Path]) -> None:
    if path is None:
        return
    with _GLOBAL_CIP_CACHE_LOCK:
        if _GLOBAL_CIP_CACHE_PATH == path:
            _save_persistent_cip_cache(path, _GLOBAL_CIP_CACHE)


def _cip_cache_key_from_value(value: Any) -> Optional[str]:
    cleaned, _ = _normalize_for_matching(value)
    key = cleaned.strip().lower()
    return key or None


def _tf_saturation(count: int) -> float:
    """Return a saturated term-frequency weight to limit long-text dominance."""
    if count <= 0:
        return 0.0
    return math.log1p(count)

def _coerce_text(value: Any) -> str:
    """Return a stripped string representation or empty string for NA-like values."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"') and len(text) > 3:
        text = text[2:-1]
    return text


def _load_hardcoded_matches(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_hardcoded_matches(path: Path, mapping: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2, sort_keys=True)


def _extract_split_pairs(value: Any) -> List[Tuple[str, str, str]]:
    text = _coerce_text(value)
    if not text:
        return []

    pairs: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    minor_variants = _extract_minor_variants(text)
    if len(minor_variants) >= 2:
        first = minor_variants[0].strip(" ,;:-")
        second = minor_variants[1].strip(" ,;:-")
        if first and second and (first, second) not in seen:
            pairs.append(("minor", first, second))
            seen.add((first, second))

    match_in = _MINOR_IN_PATTERN.search(text)
    if match_in:
        before = text[: match_in.start()].strip(" ,;:-")
        after = match_in.group(1).strip(" ,;:-")
        if before and after and (before, after) not in seen:
            pairs.append(("minor_in", before, after))
            seen.add((before, after))

    for label, pattern in _SPLIT_TOKEN_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        before = text[: match.start()].strip(" ,;:-")
        after = text[match.end():].strip(" ,;:-")
        if before and after and (before, after) not in seen:
            pairs.append((label, before, after))
            seen.add((before, after))

    return pairs


def _extract_minor_variants(value: Any) -> List[str]:
    text = _coerce_text(value)
    if not text:
        return []

    match_in = _MINOR_IN_PATTERN.search(text)
    variants: List[str] = []
    if match_in:
        before = text[: match_in.start()].strip(" ,;:-")
        after = match_in.group(1).strip(" ,;:-")
        if before:
            variants.append(before)
        if after:
            variants.append(after)
        return [variant for variant in variants if variant]

    if _MINOR_PATTERN.search(text):
        parts = _MINOR_PATTERN.split(text)
        if parts:
            before = parts[0].strip(" ,;:-")
            after = " ".join(parts[1:]).strip(" ,;:-")
            if before:
                variants.append(before)
            if after:
                variants.append(after)
    return [variant for variant in variants if variant]


def _normalize_for_matching_with_options(
    value: Any,
    stopwords: Sequence[str] = _FIELD_STOPWORDS,
) -> Tuple[str, Tuple[str, ...], List[Set[str]]]:
    """Normalize raw text and capture slash-separated optional token groups."""
    if value is None or pd.isna(value):
        return "", tuple(), []

    text = str(value).strip()
    if not text:
        return "", tuple(), []

    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().replace("&", " and ")
    normalized = re.sub(r"\d+", " ", normalized)

    stopword_set = set(stopwords)
    neutral_set = _NEUTRAL_FIELD_TOKEN_SET

    slash_groups: List[List[str]] = []
    for segment in normalized.split():
        if "/" in segment:
            options = [opt for opt in segment.split("/") if opt]
            group_tokens: List[str] = []
            for option in options:
                option_clean = _TOKEN_SPLIT_RE.sub(" ", option)
                option_clean = re.sub(r"\s+", " ", option_clean).strip()
                for opt_token in option_clean.split():
                    if (opt_token in stopword_set and opt_token not in neutral_set) or (len(opt_token) <= 1 and opt_token not in neutral_set):
                        continue
                    group_tokens.append(opt_token)
            if group_tokens:
                slash_groups.append(group_tokens)

    normalized = _TOKEN_SPLIT_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return "", tuple(), []

    filtered_tokens: List[str] = []
    for tok in normalized.split():
        if tok in stopword_set and tok not in neutral_set:
            continue
        if len(tok) <= 1 and tok not in neutral_set:
            continue
        filtered_tokens.append(tok)

    if not filtered_tokens:
        return normalized, tuple(), []
    base_tokens = list(filtered_tokens)
    cleaned_text = " ".join(base_tokens)

    optional_groups: List[Set[str]] = []
    if slash_groups:
        token_set = set(base_tokens)
        seen: Set[frozenset[str]] = set()
        for group in slash_groups:
            group_filtered = {tok for tok in group if tok in token_set}
            if len(group_filtered) >= 2:
                frozen = frozenset(group_filtered)
                if frozen not in seen:
                    seen.add(frozen)
                    optional_groups.append(group_filtered)

    augmented_tokens = list(base_tokens)
    for group in optional_groups:
        synthetic = f"group::{ '|'.join(sorted(group)) }"
        augmented_tokens.append(synthetic)

    return cleaned_text, tuple(augmented_tokens), optional_groups


def _normalize_for_matching(value: Any, stopwords: Sequence[str] = _FIELD_STOPWORDS) -> Tuple[str, Tuple[str, ...]]:
    cleaned, tokens, _ = _normalize_for_matching_with_options(value, stopwords)
    return cleaned, tokens


def _token_overlap_components(
    query_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    token_weights: Dict[str, float],
    optional_groups: Optional[Sequence[Set[str]]] = None,
) -> Tuple[float, List[str], List[str], float, float, float]:
    query_counter = Counter(query_tokens)
    candidate_counter = Counter(candidate_tokens)
    overlap_counter = query_counter & candidate_counter

    query_total_weight = sum(
        token_weights.get(tok, _DEFAULT_TOKEN_WEIGHT) * _tf_saturation(count)
        for tok, count in query_counter.items()
    )
    candidate_total_weight = sum(
        token_weights.get(tok, _DEFAULT_TOKEN_WEIGHT) * _tf_saturation(count)
        for tok, count in candidate_counter.items()
    )

    overlap_score = sum(
        token_weights.get(tok, _DEFAULT_TOKEN_WEIGHT) * _tf_saturation(count)
        for tok, count in overlap_counter.items()
    )

    remaining_query_counter = query_counter - overlap_counter
    remaining_candidate_counter = candidate_counter - overlap_counter

    if optional_groups:
        for group in optional_groups:
            if any(overlap_counter.get(tok, 0) > 0 for tok in group):
                for tok in group:
                    if tok in remaining_candidate_counter:
                        del remaining_candidate_counter[tok]

    extra_weight = 0.0
    for tok, count in remaining_candidate_counter.items():
        if tok in _NEUTRAL_FIELD_TOKEN_SET:
            continue
        base_weight = token_weights.get(tok, _DEFAULT_TOKEN_WEIGHT) * _tf_saturation(count)
        factor = _PENALTY_TOKEN_FACTORS.get(tok, 1.0)
        extra_weight += base_weight * factor

    remaining_query = list(
        itertools.chain.from_iterable([tok] * count for tok, count in remaining_query_counter.items())
    )
    remaining_candidate = list(
        itertools.chain.from_iterable([tok] * count for tok, count in remaining_candidate_counter.items())
    )
    return (
        overlap_score,
        remaining_query,
        remaining_candidate,
        extra_weight,
        query_total_weight,
        candidate_total_weight,
    )


def _approximate_overlap(
    query_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    token_weights: Dict[str, float],
) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0

    candidate_pool = list(candidate_tokens)
    approx_score = 0.0
    for token in query_tokens:
        best_idx = -1
        best_sim = 0.0
        for idx, candidate in enumerate(candidate_pool):
            sim = jellyfish.jaro_winkler_similarity(token, candidate)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_sim >= _APPROX_SIM_THRESHOLD:
            approx_score += token_weights.get(token, _DEFAULT_TOKEN_WEIGHT) * _tf_saturation(1) * best_sim
            if best_idx >= 0:
                candidate_pool.pop(best_idx)
    return approx_score


def _fuzzy_similarity(query_text: str, candidate_text: str) -> float:
    if not query_text or not candidate_text:
        return 0.0
    return fuzz.token_set_ratio(query_text, candidate_text) / 100.0


def _partial_fuzzy_similarity(query_text: str, candidate_text: str) -> float:
    if not query_text or not candidate_text:
        return 0.0
    return fuzz.partial_ratio(query_text, candidate_text) / 100.0


def _compute_match_score_details(
    query_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    query_text: str,
    candidate_text: str,
    token_weights: Dict[str, float],
    candidate_optional: Optional[Sequence[Set[str]]] = None,
    *,
    include_approx: bool = True,
    include_fuzzy: bool = True,
    apply_penalty: bool = True,
    fuzzy_scorer: Optional[Callable[[str, str], float]] = None,
) -> Tuple[float, Dict[str, float]]:
    (
        overlap,
        remaining_query,
        remaining_candidate,
        extra_weight,
        query_weight_total,
        candidate_weight_total,
    ) = _token_overlap_components(
        query_tokens,
        candidate_tokens,
        token_weights,
        candidate_optional,
    )
    approx = _approximate_overlap(remaining_query, remaining_candidate, token_weights) if include_approx else 0.0
    if include_fuzzy:
        fuzzy_score = (fuzzy_scorer or _fuzzy_similarity)(query_text, candidate_text)
    else:
        fuzzy_score = 0.0
    penalty = _EXTRA_TOKEN_PENALTY * extra_weight if apply_penalty else 0.0
    precision = overlap / candidate_weight_total if candidate_weight_total > 0 else 0.0
    recall = overlap / query_weight_total if query_weight_total > 0 else 0.0
    if precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    overlap_component = overlap * f1
    approx_component = _APPROX_WEIGHT * approx if include_approx else 0.0
    fuzzy_component = _FUZZ_WEIGHT * fuzzy_score if include_fuzzy else 0.0
    score = max(overlap_component + approx_component + fuzzy_component - penalty, 0.0)
    parts = {
        "overlap": float(overlap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "approx": float(approx),
        "fuzzy": float(fuzzy_score),
        "penalty": float(penalty),
        "extra_weight": float(extra_weight),
        "query_weight": float(query_weight_total),
        "candidate_weight": float(candidate_weight_total),
        "overlap_component": float(overlap_component),
        "approx_component": float(approx_component),
        "fuzzy_component": float(fuzzy_component),
        "plain_score": float(overlap_component + approx_component),
    }
    return score, parts


def _compute_match_score(
    query_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    query_text: str,
    candidate_text: str,
    token_weights: Dict[str, float],
    candidate_optional: Optional[Sequence[Set[str]]] = None,
    *,
    include_approx: bool = True,
    include_fuzzy: bool = True,
    apply_penalty: bool = True,
    fuzzy_scorer: Optional[Callable[[str, str], float]] = None,
) -> float:
    score, _ = _compute_match_score_details(
        query_tokens,
        candidate_tokens,
        query_text,
        candidate_text,
        token_weights,
        candidate_optional,
        include_approx=include_approx,
        include_fuzzy=include_fuzzy,
        apply_penalty=apply_penalty,
        fuzzy_scorer=fuzzy_scorer,
    )
    return score


def _is_within_tolerance(score: float, best_score: float, tolerance: float = _SCORE_TOLERANCE) -> bool:
    return abs(score - best_score) <= tolerance


def _select_unique_match(
    scored: List[Tuple[Dict[str, Any], float]],
    tolerance: float = _SCORE_TOLERANCE,
) -> Optional[Tuple[Dict[str, Any], float]]:
    if not scored:
        return None
    best_entry, best_score = scored[0]
    if best_score <= 0:
        return None
    close = [
        (entry, score)
        for entry, score in scored
        if _is_within_tolerance(score, best_score, tolerance)
    ]
    if len(close) == 1:
        return best_entry, best_score
    return None


def _resolve_scored_entries(
    scored: List[Tuple[Dict[str, Any], float]],
    tolerance: float,
    stage1_scores: Dict[str, float],
) -> Optional[Tuple[Dict[str, Any], float]]:
    if not scored:
        return None
    best_entry, best_score = scored[0]
    close_entries = [
        entry
        for entry, score in scored
        if _is_within_tolerance(score, best_score, tolerance)
    ]
    if not close_entries:
        return None
    if len(close_entries) == 1:
        return best_entry, best_score

    def sort_key(entry: Dict[str, Any]) -> Tuple[float, str]:
        return (-stage1_scores.get(entry["code"], 0.0), entry["code"])

    chosen = sorted(close_entries, key=sort_key)[0]
    return chosen, best_score


def _score_entries(
    entries: Iterable[Dict[str, Any]],
    query_tokens: Iterable[str],
    query_text: str,
    token_weights: Dict[str, float],
    token_key: str,
    text_key: str,
    *,
    capture_details: bool = False,
    include_approx: bool = True,
    include_fuzzy: bool = True,
    apply_penalty: bool = True,
    fuzzy_scorer: Optional[Callable[[str, str], float]] = None,
) -> List[Union[Tuple[Dict[str, Any], float], Tuple[Dict[str, Any], float, Dict[str, float]]]]:
    scores: List[Union[Tuple[Dict[str, Any], float], Tuple[Dict[str, Any], float, Dict[str, float]]]] = []
    for entry in entries:
        candidate_tokens = entry.get(token_key, set())
        candidate_text = entry.get(text_key, "")
        candidate_optional = entry.get("optional_groups", [])
        if capture_details:
            score, parts = _compute_match_score_details(
                query_tokens,
                candidate_tokens,
                query_text,
                candidate_text,
                token_weights,
                candidate_optional,
                include_approx=include_approx,
                include_fuzzy=include_fuzzy,
                apply_penalty=apply_penalty,
                fuzzy_scorer=fuzzy_scorer,
            )
            scores.append((entry, score, parts))
        else:
            score = _compute_match_score(
                query_tokens,
                candidate_tokens,
                query_text,
                candidate_text,
                token_weights,
                candidate_optional,
                include_approx=include_approx,
                include_fuzzy=include_fuzzy,
                apply_penalty=apply_penalty,
                fuzzy_scorer=fuzzy_scorer,
            )
            scores.append((entry, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def _compute_token_weights(doc_counts: Counter, total_docs: int) -> Dict[str, float]:
    if total_docs <= 0:
        return {}
    neutral = set(_NEUTRAL_FIELD_TOKENS)
    weights: Dict[str, float] = {}
    for token, count in doc_counts.items():
        if token in neutral:
            weights[token] = 0.0
        else:
            weights[token] = math.log((1 + total_docs) / (1 + count)) + 1.0
    return weights


def _build_vocabulary(doc_counts: Counter) -> Dict[str, int]:
    return {token: idx for idx, token in enumerate(sorted(doc_counts))}


def _build_entry_matrix(
    catalog: Sequence[Dict[str, Any]],
    vocab: Dict[str, int],
    token_weights: Dict[str, float],
) -> csr_matrix:
    data: List[float] = []
    indices: List[int] = []
    indptr: List[int] = [0]
    for entry in catalog:
        entry_tokens = entry.get("title_tokens", set())
        for token in entry_tokens:
            vocab_idx = vocab.get(token)
            if vocab_idx is None:
                continue
            weight = token_weights.get(token, _DEFAULT_TOKEN_WEIGHT)
            if weight <= 0:
                continue
            data.append(weight)
            indices.append(vocab_idx)
        indptr.append(len(data))
    if not data:
        return csr_matrix((len(catalog), len(vocab)), dtype=float)
    return csr_matrix((data, indices, indptr), shape=(len(catalog), len(vocab)), dtype=float)


def _build_field_matrix(
    token_lists: Sequence[Tuple[str, ...]],
    vocab: Dict[str, int],
    token_weights: Dict[str, float],
) -> csr_matrix:
    data: List[float] = []
    indices: List[int] = []
    indptr: List[int] = [0]
    for tokens in token_lists:
        for token in tokens:
            vocab_idx = vocab.get(token)
            if vocab_idx is None:
                continue
            weight = token_weights.get(token, _DEFAULT_TOKEN_WEIGHT)
            if weight <= 0:
                continue
            data.append(weight)
            indices.append(vocab_idx)
        indptr.append(len(data))
    if not data:
        return csr_matrix((len(token_lists), len(vocab)), dtype=float)
    return csr_matrix((data, indices, indptr), shape=(len(token_lists), len(vocab)), dtype=float)


def _generate_lsh_keys(
    token_seq: Tuple[str, ...],
    band_size: int,
    max_keys: int,
) -> List[str]:
    if not token_seq or band_size <= 0 or max_keys <= 0:
        return []
    unique_tokens = sorted(dict.fromkeys(token_seq))
    if len(unique_tokens) < band_size:
        return []
    keys: List[str] = []
    for combo in itertools.combinations(unique_tokens, band_size):
        keys.append("|".join(combo))
        if len(keys) >= max_keys:
            break
    return keys


def _build_lsh_index(
    catalog: Sequence[Dict[str, Any]],
    band_size: int,
    max_keys: int,
) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    if band_size <= 0 or max_keys <= 0:
        return index
    for entry_idx, entry in enumerate(catalog):
        entry_tokens = tuple(sorted(entry.get("title_tokens", [])))
        keys = _generate_lsh_keys(entry_tokens, band_size, max_keys)
        for key in keys:
            bucket = index.setdefault(key, [])
            bucket.append(entry_idx)
    return index


def _lsh_stage1_candidates(
    token_seq: Tuple[str, ...],
    catalog: Sequence[Dict[str, Any]],
    token_weights: Dict[str, float],
    lsh_index: Optional[Dict[str, List[int]]],
    max_candidates: int = _MAX_STAGE1_CANDIDATES,
) -> List[Tuple[int, float]]:
    if not token_seq or not lsh_index:
        return []
    keys = _generate_lsh_keys(token_seq, _LSH_BAND_SIZE, _LSH_MAX_KEYS)
    if not keys:
        return []
    candidate_indices: Set[int] = set()
    for key in keys:
        bucket = lsh_index.get(key)
        if not bucket:
            continue
        candidate_indices.update(bucket)
        if len(candidate_indices) >= _LSH_MAX_CANDIDATES:
            break
    if not candidate_indices:
        return []
    scored: List[Tuple[int, float]] = []
    for idx in candidate_indices:
        entry = catalog[idx]
        overlap_score, _, _, extra_weight, _, _ = _token_overlap_components(
            token_seq,
            entry.get("title_tokens", set()),
            token_weights,
        )
        score = max(0.0, overlap_score - extra_weight)
        if score > 0:
            scored.append((idx, float(score)))
    if not scored:
        return []
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:max_candidates]


def _stage1_candidates(
    scores_matrix: csr_matrix,
    max_candidates: int = _MAX_STAGE1_CANDIDATES,
) -> List[List[Tuple[int, float]]]:
    results: List[List[Tuple[int, float]]] = []
    for row_idx in range(scores_matrix.shape[0]):
        row = scores_matrix.getrow(row_idx)
        if row.nnz == 0:
            results.append([])
            continue
        indices = row.indices
        data = row.data
        order = np.argsort(data)[::-1][:max_candidates]
        candidates = [(int(indices[pos]), float(data[pos])) for pos in order if data[pos] > 0]
        results.append(candidates)
    return results


def _compute_stage1_candidates_for_tokens(
    token_seq: Tuple[str, ...],
    entry_matrix: csr_matrix,
    vocab: Dict[str, int],
    token_weights: Dict[str, float],
    use_full_stage1: bool,
    lsh_index: Optional[Dict[str, List[int]]] = None,
    catalog: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Tuple[int, float]]:
    if use_full_stage1:
        return [(idx, 1.0) for idx in range(entry_matrix.shape[0])]
    if not token_seq:
        return []
    if lsh_index is not None and catalog is not None:
        lsh_candidates = _lsh_stage1_candidates(token_seq, catalog, token_weights, lsh_index)
        if lsh_candidates:
            return lsh_candidates
    field_matrix = _build_field_matrix([token_seq], vocab, token_weights)
    if field_matrix.nnz == 0:
        return []
    batch_scores = field_matrix.dot(entry_matrix.T).tocsr()
    return _stage1_candidates(batch_scores)[0]


def _stage1_ngram_precheck(
    cleaned_field: str,
    token_seq: Tuple[str, ...],
    catalog: Sequence[Dict[str, Any]],
    max_ngram: Optional[int],
    debug: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[Dict[str, Any], float]]:
    if not token_seq or len(token_seq) <= 1:
        return None

    max_len = len(token_seq)
    if max_ngram is not None:
        max_len = max(2, min(max_len, max_ngram))

    segments: List[Tuple[str, int]] = []
    normalized_field = cleaned_field.strip()
    if normalized_field and len(token_seq) == max_len:
        segments.append((normalized_field, len(token_seq)))

    seen_segments: Set[str] = set()
    for size in range(max_len, 1, -1):
        for start in range(0, len(token_seq) - size + 1):
            segment_tokens = token_seq[start : start + size]
            segment_text = " ".join(segment_tokens)
            if segment_text not in seen_segments:
                seen_segments.add(segment_text)
                segments.append((segment_text, size))

    precheck_log: List[Dict[str, Any]] = []
    for segment_text, size in segments:
        scores: List[Tuple[Dict[str, Any], float]] = []
        for entry in catalog:
            sim = jellyfish.jaro_winkler_similarity(segment_text, entry["title_raw"])
            if sim > 0:
                scores.append((entry, sim))
        if not scores:
            precheck_log.append({
                "segment": segment_text,
                "size": size,
                "best_score": 0.0,
                "candidates": [],
            })
            continue

        best_score = max(score for _, score in scores)
        candidates = [
            (entry, score)
            for entry, score in scores
            if _is_within_tolerance(score, best_score, _SCORE_TOLERANCE)
        ]

        precheck_log.append({
            "segment": segment_text,
            "size": size,
            "best_score": float(best_score),
            "candidates": [
                (entry["code"], float(score))
                for entry, score in sorted(candidates, key=lambda item: item[1], reverse=True)[:5]
            ],
        })

        if best_score >= _MIN_ACCEPT_SCORE and len(candidates) == 1:
            entry, score = candidates[0]
            if debug is not None:
                debug.setdefault("stage1_precheck", precheck_log)
                debug["precheck_segment"] = segment_text
            return entry, score

    if debug is not None and precheck_log:
        debug.setdefault("stage1_precheck", precheck_log)

    return None


def _prepare_cip_catalog(
    cip_df: pd.DataFrame,
    code_col: str,
    title_col: str,
    examples_col: Optional[str],
    definition_col: Optional[str],
) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    has_examples = bool(examples_col and examples_col in cip_df.columns)
    has_definitions = bool(definition_col and definition_col in cip_df.columns)

    for _, row in cip_df.iterrows():
        code = _coerce_text(row.get(code_col))
        if not code:
            continue

        title_raw = _coerce_text(row.get(title_col))
        if not title_raw:
            continue

        _, title_tokens, title_optional = _normalize_for_matching_with_options(title_raw)
        examples_raw = _coerce_text(row.get(examples_col)) if has_examples else ""
        definitions_raw = _coerce_text(row.get(definition_col)) if has_definitions else ""
        _, examples_tokens, examples_optional = _normalize_for_matching_with_options(examples_raw)
        _, definition_tokens, definition_optional = _normalize_for_matching_with_options(definitions_raw)

        title_tokens_seq = tuple(title_tokens)
        examples_tokens_seq = tuple(examples_tokens)
        definition_tokens_seq = tuple(definition_tokens)
        examples_text = examples_raw
        definition_text = definitions_raw
        optional_groups_set: Set[frozenset[str]] = set()
        for group in title_optional + examples_optional + definition_optional:
            frozen = frozenset(group)
            if frozen:
                optional_groups_set.add(frozen)
        optional_groups = [set(group) for group in optional_groups_set]

        entry = {
            "code": code,
            "title_raw": title_raw,
            "title_tokens": title_tokens_seq,
            "examples_tokens": examples_tokens_seq,
            "examples_text": examples_text,
            "definition_tokens": definition_tokens_seq,
            "definition_text": definition_text,
            "optional_groups": optional_groups,
        }
        catalog.append(entry)
    return catalog


def _match_field_against_catalog(
    cleaned_field: str,
    tokens: Tuple[str, ...],
    catalog: List[Dict[str, Any]],
    token_weights: Dict[str, float],
    stage1_candidates: Sequence[Tuple[int, float]],
    catalog_titles: Sequence[str],
    stage1_max_ngram: Optional[int],
    *,
    debug: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    token_seq = tuple(tokens)
    base_token_seq = tuple(tok for tok in token_seq if not tok.startswith("group::"))
    query_text = cleaned_field or " ".join(token_seq)
    stage1_details: List[Tuple[Dict[str, Any], float, Dict[str, float]]] = []

    precheck_match = _stage1_ngram_precheck(cleaned_field, base_token_seq, catalog, stage1_max_ngram, debug)
    if precheck_match is not None:
        entry, score = precheck_match
        if debug is not None:
            debug["selection_stage"] = "title_precheck"
            debug["selected_score"] = float(score)
        return {
            "code": entry["code"],
            "title_raw": entry["title_raw"],
            "score": round(float(score), 6),
            "source": "title_precheck",
        }

    if stage1_candidates:
        stage1_scores: List[Tuple[Dict[str, Any], float]] = []
        for idx, base_score in stage1_candidates:
            entry = catalog[idx]
            optional_groups = entry.get("optional_groups", [])
            score, parts = _compute_match_score_details(
                token_seq,
                entry.get("title_tokens", tuple()),
                query_text,
                entry["title_raw"],
                token_weights,
                optional_groups,
            )
            parts = parts.copy()
            parts["matrix_score"] = float(base_score)
            stage1_details.append((entry, score, parts))
            stage1_scores.append((entry, score))
        stage1_scores.sort(key=lambda item: item[1], reverse=True)
    else:
        fallback_entries: List[Dict[str, Any]] = []
        if query_text:
            extracted = process.extract(
                query_text,
                catalog_titles,
                scorer=fuzz.token_set_ratio,
                limit=_FALLBACK_CANDIDATES,
            )
            fallback_indices = [match[2] for match in extracted]
            fallback_entries = [catalog[idx] for idx in fallback_indices]
        stage1_details = _score_entries(
            fallback_entries or catalog,
            token_seq,
            query_text,
            token_weights,
            "title_tokens",
            "title_raw",
            capture_details=True,
        )
        stage1_scores = [(entry, score) for entry, score, _parts in stage1_details]

    if not stage1_scores:
        return None

    token_score_map: Dict[str, float] = {}
    for entry, _score, parts in stage1_details:
        token_score = parts["plain_score"] - parts["penalty"]
        parts["token_score"] = float(token_score)
        token_score_map[entry["code"]] = float(token_score)

    if debug is not None:
        debug["stage1_candidates"] = [
            {
                "code": entry["code"],
                "title": entry["title_raw"],
                "score": float(score),
                "components": parts,
            }
            for entry, score, parts in stage1_details
        ]

    plain_scores = [(entry, parts["plain_score"], parts) for entry, _score, parts in stage1_details]
    if plain_scores:
        best_plain = max(score for _, score, _ in plain_scores)
        plain_candidates = [
            (entry, score, parts)
            for entry, score, parts in plain_scores
            if _is_within_tolerance(score, best_plain, _SCORE_TOLERANCE)
        ]
        if len(plain_candidates) == 1 and best_plain >= _MIN_ACCEPT_SCORE:
            entry, _score, parts = plain_candidates[0]
            if debug is not None:
                debug["selection_stage"] = "title"
                debug["selected_score"] = float(parts["plain_score"])
            return {
                "code": entry["code"],
                "title_raw": entry["title_raw"],
                "score": round(float(parts["plain_score"]), 6),
                "source": "title",
            }

        if plain_candidates:
            fuzzy_ranked = sorted(
                plain_candidates,
                key=lambda item: (item[1] + item[2].get("fuzzy_component", 0.0)),
                reverse=True,
            )
            top_fuzzy_score = fuzzy_ranked[0][1] + fuzzy_ranked[0][2].get("fuzzy_component", 0.0)
            fuzzy_ties = [
                cand for cand in fuzzy_ranked
                if _is_within_tolerance(
                    cand[1] + cand[2].get("fuzzy_component", 0.0),
                    top_fuzzy_score,
                    _SCORE_TOLERANCE,
                )
            ]
            if len(fuzzy_ties) == 1 and top_fuzzy_score >= _MIN_ACCEPT_SCORE:
                entry, plain_score, parts = fuzzy_ties[0]
                final_score = plain_score + parts.get("fuzzy_component", 0.0)
                if debug is not None:
                    debug["selection_stage"] = "title_fuzzy"
                    debug["selected_score"] = float(final_score)
                return {
                    "code": entry["code"],
                    "title_raw": entry["title_raw"],
                    "score": round(float(final_score), 6),
                    "source": "title_fuzzy",
                }

    stage1_map = {entry["code"]: score for entry, score in stage1_scores}
    stage1_best_entry = stage1_scores[0][0]
    stage1_best_score = stage1_scores[0][1]

    best_token_score = max(token_score_map.values(), default=0.0)
    if best_token_score > 0:
        candidate_entries = [
            entry
            for entry, _score, parts in stage1_details
            if _is_within_tolerance(parts["token_score"], best_token_score, _SCORE_TOLERANCE)
        ]
    else:
        candidate_entries = [
            entry
            for entry, _score, parts in stage1_details
            if entry["examples_tokens"] or entry["definition_tokens"]
        ]

    if not candidate_entries:
        candidate_entries = [entry for entry, _score, _parts in stage1_details][: _MAX_STAGE1_CANDIDATES]

    stage2_details = _score_entries(
        candidate_entries,
        token_seq,
        query_text,
        token_weights,
        "examples_tokens",
        "examples_text",
        capture_details=True,
        include_approx=False,
        include_fuzzy=True,
        apply_penalty=False,
        fuzzy_scorer=_partial_fuzzy_similarity,
    )
    if debug is not None:
        debug["stage2_candidates"] = [
            {
                "code": entry["code"],
                "title": entry["title_raw"],
                "score": float(score),
                "components": parts,
            }
            for entry, score, parts in stage2_details
        ]
    stage2_scores = [(entry, score) for entry, score, _parts in stage2_details]
    resolved = _resolve_scored_entries(stage2_scores, _SCORE_TOLERANCE, stage1_map)
    if resolved:
        entry, score = resolved
        if score >= _MIN_ACCEPT_SCORE:
            if debug is not None:
                debug["selection_stage"] = "examples"
                debug["selected_score"] = float(score)
            return {
                "code": entry["code"],
                "title_raw": entry["title_raw"],
                "score": round(float(score), 6),
                "source": "examples",
            }

    stage3_details = _score_entries(
        candidate_entries,
        token_seq,
        query_text,
        token_weights,
        "definition_tokens",
        "definition_text",
        capture_details=True,
        include_approx=False,
        include_fuzzy=True,
        apply_penalty=False,
        fuzzy_scorer=_partial_fuzzy_similarity,
    )
    if debug is not None:
        debug["stage3_candidates"] = [
            {
                "code": entry["code"],
                "title": entry["title_raw"],
                "score": float(score),
                "components": parts,
            }
            for entry, score, parts in stage3_details
        ]
    stage3_scores = [(entry, score) for entry, score, _parts in stage3_details]
    resolved = _resolve_scored_entries(stage3_scores, _SCORE_TOLERANCE, stage1_map)
    if resolved:
        entry, score = resolved
        if score >= _MIN_ACCEPT_SCORE:
            if debug is not None:
                debug["selection_stage"] = "definition"
                debug["selected_score"] = float(score)
            return {
                "code": entry["code"],
                "title_raw": entry["title_raw"],
                "score": round(float(score), 6),
                "source": "definition",
            }

    if stage1_best_score >= _MIN_ACCEPT_SCORE:
        if debug is not None:
            debug["selection_stage"] = "title_best_effort"
            debug["selected_score"] = float(stage1_best_score)
        return {
            "code": stage1_best_entry["code"],
            "title_raw": stage1_best_entry["title_raw"],
            "score": round(float(stage1_best_score), 6),
            "source": "title_best_effort",
        }

    if debug is not None:
        debug["selection_stage"] = "unmatched"
        debug["selected_score"] = 0.0
    return None


def _load_cip_reference(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported CIP reference extension: {suffix}")


def _match_fields_to_cip_core(
    df: pd.DataFrame,
    cip_path: Union[str, Path] = Path(root) / "data/crosswalks/cip/CIPCode2020.csv",
    digit_length: int = 6,
    field_column: str = "field_raw",
    cip_code_column: str = "CIPCode",
    cip_title_column: str = "CIPTitle",
    cip_examples_column: str = "Examples",
    cip_definition_column: str = "CIPDefinition",
    *,
    return_debug: bool = False,
    use_full_stage1: bool = False,
    stage1_max_ngram: Optional[int] = None,
    hardcode_path: Optional[Union[str, Path]] = None,
    interactive_hardcode: bool = False,
    hardcode_sample_size: int = 10,
    match_confirm_min_score: Optional[float] = 100.0,
    **kwargs: Any,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
    """
    Match raw field names to CIP codes using weighted token overlap with fuzzy robustness.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing raw field values.
    cip_path : str or Path
        File path to the CIP reference catalog.
    digit_length : int, default 6
        Desired CIP code granularity (2, 4, or 6 digits).
    field_column : str, default "field_raw"
        Column in `df` containing the raw field names.
    cip_code_column : str, default "CIPCode"
        Column in the CIP catalog containing the CIP codes.
    cip_title_column : str, default "CIPTitle"
        Column in the CIP catalog containing official CIP titles.
    cip_examples_column : str, default "Examples"
        Column containing example program names (used for disambiguation).
    cip_definition_column : str, default "CIPDefinition"
        Column containing CIP definitions (used for disambiguation).
    return_debug : bool, default False
        When True, also return per-row diagnostic information.
    use_full_stage1 : bool, default False
        When True, bypass blocking and evaluate all catalog entries in Stage 1.
    stage1_max_ngram : int, optional
        Maximum n-gram length to consider for Stage 1 pre-checks (defaults to full token length).
    hardcode_path : str or Path, optional
        Path to the hard-coded match dictionary (defaults to data/int/cip_hardcoded_matches.json).
    interactive_hardcode : bool, default False
        When True, prompt for hard-coded matches on ambiguous cases and update the dictionary.
    hardcode_sample_size : int, default 10
        Maximum number of interactive prompts per run when interactive_hardcode is enabled.
    match_confirm_min_score : float or None, default 0.0
        Minimum Jaro-Winkler similarity required for an automatically generated match to be included
        in the final confirmation prompt. When set to None, all pending updates are saved
        automatically. Manually entered overrides are always persisted regardless of this threshold.

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with additional columns:
        - `field_clean`: normalized field text used for matching.
        - `cip_code`: matched CIP code (if any).
        - `cip_title`: matched CIP title (if any).
        - `cip_match_score`: final similarity score.
        - `cip_match_source`: which metadata field resolved the match.
        - `cip_jaro_winkler`: Jaro-Winkler similarity between the cleaned field and matched title.
        - If a split keyword is detected, additional columns with suffix `_2` capture the secondary
          match (e.g., `cip_code_2`, `cip_match_score_2`, `cip_match_source_2`, `cip_jaro_winkler_2`).
    If `return_debug` is True, also returns a list of per-row diagnostic dictionaries.
    """
    if "hardcode_confirm_min_score" in kwargs:
        alias_value = kwargs.pop("hardcode_confirm_min_score")
        if match_confirm_min_score in (0.0, None):
            match_confirm_min_score = alias_value
        elif match_confirm_min_score != alias_value:
            raise ValueError(
                "Received both 'match_confirm_min_score' and legacy "
                "'hardcode_confirm_min_score' with different values."
            )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    if field_column not in df.columns:
        raise KeyError(f"DataFrame is missing required column '{field_column}'.")

    pattern = _DIGIT_PATTERNS.get(int(digit_length))
    if pattern is None:
        raise ValueError("digit_length must be one of {2, 4, 6}.")

    path = Path(cip_path)
    if not path.exists():
        raise FileNotFoundError(f"CIP reference file not found: {path}")

    catalog_df = _load_cip_reference(path)
    if cip_code_column not in catalog_df.columns:
        raise ValueError(f"CIP reference missing '{cip_code_column}' column.")
    if cip_title_column not in catalog_df.columns:
        raise ValueError(f"CIP reference missing '{cip_title_column}' column.")

    catalog_df = catalog_df.copy()
    catalog_df[cip_code_column] = catalog_df[cip_code_column].apply(_coerce_text)
    catalog_df = catalog_df[catalog_df[cip_code_column] != ""]
    catalog_df = catalog_df[catalog_df[cip_code_column].str.match(pattern, na=False)].copy()
    if catalog_df.empty:
        raise ValueError(f"No CIP codes with {digit_length}-digit format found in {path}.")

    catalog_df[cip_title_column] = catalog_df[cip_title_column].apply(_coerce_text)
    catalog_df = catalog_df[catalog_df[cip_title_column] != ""]
    if catalog_df.empty:
        raise ValueError("CIP reference has no titles after cleaning.")

    catalog_df = catalog_df.drop_duplicates(subset=[cip_code_column])

    catalog = _prepare_cip_catalog(
        catalog_df,
        cip_code_column,
        cip_title_column,
        cip_examples_column,
        cip_definition_column,
    )
    if not catalog:
        raise ValueError("CIP reference is empty after preprocessing.")

    catalog_by_code = {entry["code"]: entry for entry in catalog}
    catalog_by_title = {entry["title_raw"].lower(): entry["code"] for entry in catalog}

    hardcode_path = Path(hardcode_path) if hardcode_path is not None else _DEFAULT_HARDCODE_PATH
    hardcoded_matches = _load_hardcoded_matches(hardcode_path)
    original_hardcoded_matches: Dict[str, str] = hardcoded_matches.copy()
    prompts_remaining = hardcode_sample_size if interactive_hardcode else 0
    collect_debug = return_debug or interactive_hardcode
    pending_match_records: Dict[str, Dict[str, Any]] = {}

    def register_pending_match(
        key: Optional[str],
        value: str,
        *,
        field_raw: Any,
        source: str,
        score: Optional[float],
        cip_title: Optional[str],
        debug_record: Optional[Dict[str, Any]],
        require_confirmation: bool = True,
        immediate: bool = False,
        jw_score: Optional[float] = None,
    ) -> None:
        if not key:
            return
        previous_value = original_hardcoded_matches.get(key)
        changed = previous_value != value
        if immediate:
            hardcoded_matches[key] = value
        entry = pending_match_records.get(key)
        if entry is None:
            entry = {
                "key": key,
                "field_clean": key,
                "field_raw": field_raw,
                "value": value,
                "source": source,
                "score": score,
                "cip_code": value if value != _HARDCODE_INVALID else None,
                "cip_title": cip_title,
                "is_invalid": value == _HARDCODE_INVALID,
                "require_confirmation": require_confirmation,
                "immediate": immediate,
                "changed": changed,
                "jw_score": jw_score,
                "debug_records": [],
            }
            pending_match_records[key] = entry
        else:
            entry.update({
                "field_raw": field_raw,
                "value": value,
                "source": source,
                "score": score,
                "cip_code": value if value != _HARDCODE_INVALID else None,
                "cip_title": cip_title,
                "is_invalid": value == _HARDCODE_INVALID,
            })
            entry["require_confirmation"] = entry.get("require_confirmation", True) and require_confirmation
            entry["immediate"] = entry.get("immediate", False) or immediate
            entry["changed"] = entry.get("changed", False) or changed
            if jw_score is not None:
                entry["jw_score"] = jw_score
        entry.setdefault("debug_records", [])
        if debug_record is not None:
            debug_record["hardcode_saved"] = False
            if not any(dr is debug_record for dr in entry["debug_records"]):
                entry["debug_records"].append(debug_record)

    def build_match_mapping(updates: Iterable[Dict[str, Any]]) -> Dict[str, str]:
        mapping: Dict[str, str] = dict(original_hardcoded_matches)
        for info in updates:
            mapping[info["key"]] = info["value"]
        return mapping

    normalized_pairs = df[field_column].apply(_normalize_for_matching)
    field_clean_list = [pair[0] for pair in normalized_pairs]
    field_tokens_list = [pair[1] for pair in normalized_pairs]

    doc_counts: Counter = Counter()
    total_docs = 0
    for tokens in field_tokens_list:
        if tokens:
            total_docs += 1
            doc_counts.update(set(tokens))
    for entry in catalog:
        for token in set(entry["title_tokens"]):
            doc_counts.setdefault(token, 0)
        for token in set(entry["examples_tokens"]):
            doc_counts.setdefault(token, 0)
        for token in set(entry["definition_tokens"]):
            doc_counts.setdefault(token, 0)
    if total_docs == 0:
        total_docs = max(len(catalog), 1)

    token_weights = _compute_token_weights(doc_counts, total_docs)
    vocab = _build_vocabulary(doc_counts)
    entry_matrix = _build_entry_matrix(catalog, vocab, token_weights)
    catalog_titles = [entry["title_raw"] for entry in catalog]

    stage1_candidates: List[Optional[List[Tuple[int, float]]]] = [None] * len(field_tokens_list)
    lsh_index: Optional[Dict[str, List[int]]] = None
    if use_full_stage1:
        all_candidates = [(idx, 1.0) for idx in range(len(catalog))]
        stage1_candidates = [all_candidates[:] for _ in field_tokens_list]
    else:
        lsh_index = _build_lsh_index(catalog, _LSH_BAND_SIZE, _LSH_MAX_KEYS)
        fallback_indices: List[int] = []
        for idx, tokens in enumerate(field_tokens_list):
            candidates = _lsh_stage1_candidates(tokens, catalog, token_weights, lsh_index)
            if candidates:
                stage1_candidates[idx] = candidates
            else:
                fallback_indices.append(idx)
        if fallback_indices:
            for start in range(0, len(fallback_indices), _FIELD_BATCH_SIZE):
                end = min(start + _FIELD_BATCH_SIZE, len(fallback_indices))
                batch_indices = fallback_indices[start:end]
                batch_tokens = [field_tokens_list[i] for i in batch_indices]
                field_matrix = _build_field_matrix(batch_tokens, vocab, token_weights)
                if field_matrix.nnz == 0:
                    for idx in batch_indices:
                        stage1_candidates[idx] = []
                    continue
                batch_scores = field_matrix.dot(entry_matrix.T).tocsr()
                batch_candidates = _stage1_candidates(batch_scores)
                for local_idx, idx in enumerate(batch_indices):
                    stage1_candidates[idx] = batch_candidates[local_idx]
    stage1_candidates = [candidates or [] for candidates in stage1_candidates]

    results: List[Dict[str, Any]] = []
    secondary_results: List[Dict[str, Any]] = []
    debug_info: List[Dict[str, Any]] = [] if collect_debug else []
    for idx, raw_value in enumerate(df[field_column]):
        cleaned_field = field_clean_list[idx]
        tokens = field_tokens_list[idx]
        base_candidates = stage1_candidates[idx] if idx < len(stage1_candidates) else []

        debug_record: Optional[Dict[str, Any]] = None
        if collect_debug:
            debug_record = {
                "field_raw": raw_value,
                "field_clean": cleaned_field,
            }

        variant_infos: List[Dict[str, Any]] = []
        seen_cleans: Set[str] = set()

        minor_variants = _extract_minor_variants(raw_value)
        for variant_idx, variant_text in enumerate(minor_variants):
            var_clean, var_tokens, _ = _normalize_for_matching_with_options(variant_text)
            if not var_clean or var_clean in seen_cleans:
                continue
            variant_infos.append({
                "label": f"minor_variant_{variant_idx}",
                "text": variant_text,
                "clean": var_clean,
                "tokens": tuple(var_tokens),
            })
            seen_cleans.add(var_clean)

        original_tuple = tuple(tokens)
        if cleaned_field and cleaned_field not in seen_cleans:
            variant_infos.append({
                "label": "original",
                "text": raw_value,
                "clean": cleaned_field,
                "tokens": original_tuple,
            })
            seen_cleans.add(cleaned_field)

        if not variant_infos:
            variant_infos.append({
                "label": "original",
                "text": raw_value,
                "clean": cleaned_field,
                "tokens": original_tuple,
            })

        matched_code = None
        matched_clean = None
        matched_tokens = None
        matched_source = "hardcode"
        original_clean = cleaned_field
        original_tokens = original_tuple

        hardcode_invalid_hit = False
        for info in variant_infos:
            key = info["clean"]
            if not key:
                continue
            code = hardcoded_matches.get(key)
            if code == _HARDCODE_INVALID:
                matched_code = None
                matched_clean = key
                matched_tokens = info["tokens"]
                matched_source = "hardcode_invalid"
                hardcode_invalid_hit = True
                if collect_debug and debug_record is not None:
                    debug_record["hardcode_hit"] = key
                    debug_record["selection_stage"] = "hardcode_invalid"
                    debug_record["selected_score"] = 1.0
                break
            entry = catalog_by_code.get(code) if code else None
            if entry is not None:
                matched_code = entry["code"]
                matched_clean = key
                matched_tokens = info["tokens"]
                matched_source = "hardcode"
                if collect_debug and debug_record is not None:
                    debug_record["hardcode_hit"] = key
                    debug_record["selection_stage"] = "hardcode"
                    debug_record["selected_score"] = 1.0
                break

        if hardcode_invalid_hit:
            selected_match = {
                "code": None,
                "title_raw": "",
                "score": 1.0,
                "source": matched_source,
            }
            cleaned_field = matched_clean or cleaned_field
            tokens = matched_tokens or tokens
        elif matched_code is not None:
            selected_match = {
                "code": matched_code,
                "title_raw": catalog_by_code[matched_code]["title_raw"],
                "score": 1.0,
                "source": matched_source,
            }
            cleaned_field = matched_clean or cleaned_field
            tokens = matched_tokens or tokens
        else:
            attempts: List[Dict[str, Any]] = []
            for info in variant_infos:
                tokens_tuple = info["tokens"]
                if info["label"] == "original":
                    variant_candidates = base_candidates
                else:
                    variant_candidates = _compute_stage1_candidates_for_tokens(
                        tokens_tuple,
                        entry_matrix,
                        vocab,
                        token_weights,
                        use_full_stage1,
                        lsh_index,
                        catalog,
                    )
                attempts.append({
                    "label": info["label"],
                    "clean": info["clean"],
                    "tokens": tokens_tuple,
                    "candidates": variant_candidates,
                })

            selected_match = None
            selected_variant_label = None
            selected_variant_clean = cleaned_field
            selected_variant_tokens = tuple(tokens)

            for attempt in attempts:
                var_clean = attempt["clean"]
                var_tokens = attempt["tokens"]
                variant_candidates = attempt["candidates"]

                if collect_debug and debug_record is not None:
                    debug_record.setdefault("variant_attempts", []).append({
                        "label": attempt["label"],
                        "variant_clean": var_clean,
                        "token_count": len(var_tokens),
                        "stage1_candidates": len(variant_candidates),
                    })

                match = _match_field_against_catalog(
                    var_clean,
                    var_tokens,
                    catalog,
                    token_weights,
                    variant_candidates,
                    catalog_titles,
                    stage1_max_ngram,
                    debug=debug_record,
                )
                if match is not None:
                    selected_match = match
                    selected_variant_label = attempt["label"]
                    selected_variant_clean = var_clean
                    selected_variant_tokens = var_tokens

            cleaned_field = selected_variant_clean
            tokens = selected_variant_tokens

            if collect_debug and debug_record is not None and selected_variant_label is not None:
                debug_record["selected_variant"] = selected_variant_label

        if interactive_hardcode and prompts_remaining > 0 and selected_match is None and debug_record is not None:
            stage2_info = debug_record.get("stage2_candidates") or []
            stage2_sorted = sorted(stage2_info, key=lambda item: item["score"], reverse=True)
            if len(stage2_sorted) >= 2:
                print("\nAmbiguous field:\n  Raw :", raw_value)
                print("  Clean:", cleaned_field)
                print("  Candidates:")
                limit = min(5, len(stage2_sorted))
                for idx_cand in range(limit):
                    cand = stage2_sorted[idx_cand]
                    print(f"    {idx_cand + 1}. {cand['code']} - {cand['title']} (score={cand['score']:.2f})")
                user_input = input("Select CIP by number/code/title, type 'invalid', or 'skip': ").strip()
                chosen_code: Optional[str] = None
                if user_input:
                    lower_input = user_input.lower()
                    if lower_input in {"skip", "s"}:
                        pass
                    elif lower_input in {"invalid", "none"}:
                        chosen_code = _HARDCODE_INVALID
                    elif user_input.isdigit():
                        idx_val = int(user_input) - 1
                        if 0 <= idx_val < limit:
                            chosen_code = stage2_sorted[idx_val]["code"]
                    else:
                        candidate = catalog_by_code.get(user_input)
                        if candidate:
                            chosen_code = user_input
                        else:
                            candidate = catalog_by_title.get(user_input.lower())
                            if candidate:
                                chosen_code = candidate
                if chosen_code == _HARDCODE_INVALID:
                    cleaned_field = cleaned_field or original_clean
                    hardcode_key = original_clean or cleaned_field
                    register_pending_match(
                        hardcode_key,
                        _HARDCODE_INVALID,
                        field_raw=raw_value,
                        source="hardcode_invalid",
                        score=None,
                        cip_title=None,
                        debug_record=debug_record,
                        require_confirmation=False,
                        immediate=True,
                        jw_score=None,
                    )
                    if selected_variant_clean and selected_variant_clean != hardcode_key:
                        register_pending_match(
                            selected_variant_clean,
                            _HARDCODE_INVALID,
                            field_raw=raw_value,
                            source="hardcode_invalid",
                            score=None,
                            cip_title=None,
                            debug_record=debug_record,
                            require_confirmation=False,
                            immediate=True,
                            jw_score=None,
                        )
                    prompts_remaining -= 1
                    selected_match = {
                        "code": None,
                        "title_raw": "",
                        "score": 1.0,
                        "source": "hardcode_invalid",
                    }
                    if debug_record is not None:
                        debug_record["selection_stage"] = "hardcode_invalid"
                        debug_record["selected_score"] = 1.0
                if chosen_code and chosen_code in catalog_by_code:
                    entry = catalog_by_code[chosen_code]
                    selected_match = {
                        "code": entry["code"],
                        "title_raw": entry["title_raw"],
                        "score": 1.0,
                        "source": "hardcode_interactive",
                    }
                    cleaned_field = cleaned_field or original_clean
                    hardcode_key = original_clean or cleaned_field
                    manual_jw: Optional[float] = None
                    try:
                        candidate_clean, _, _ = _normalize_for_matching_with_options(entry["title_raw"])
                        manual_jw = jellyfish.jaro_winkler_similarity(
                            cleaned_field or "",
                            candidate_clean or "",
                        )
                    except Exception:
                        manual_jw = None
                    register_pending_match(
                        hardcode_key,
                        entry["code"],
                        field_raw=raw_value,
                        source="hardcode_interactive",
                        score=selected_match["score"],
                        cip_title=entry["title_raw"],
                        debug_record=debug_record,
                        require_confirmation=False,
                        immediate=True,
                        jw_score=manual_jw,
                    )
                    if selected_variant_clean and selected_variant_clean != hardcode_key:
                        variant_jw: Optional[float] = None
                        try:
                            candidate_clean, _, _ = _normalize_for_matching_with_options(entry["title_raw"])
                            variant_jw = jellyfish.jaro_winkler_similarity(
                                selected_variant_clean or "",
                                candidate_clean or "",
                            )
                        except Exception:
                            variant_jw = None
                        register_pending_match(
                            selected_variant_clean,
                            entry["code"],
                            field_raw=raw_value,
                            source="hardcode_interactive",
                            score=selected_match["score"],
                            cip_title=entry["title_raw"],
                            debug_record=debug_record,
                            require_confirmation=False,
                            immediate=True,
                            jw_score=variant_jw,
                        )
                    prompts_remaining -= 1
                    if debug_record is not None:
                        debug_record["selection_stage"] = "hardcode_interactive"
                        debug_record["selected_score"] = 1.0

        if selected_match is None:
            primary_result = {
                "field_clean": cleaned_field,
                "cip_code": None,
                "cip_title": None,
                "cip_match_score": None,
                "cip_match_source": "unmatched",
                "cip_jaro_winkler": None,
            }
            if collect_debug and debug_record is not None:
                debug_record["selected"] = None
        else:
            jw_score: Optional[float] = None
            candidate_title = selected_match["title_raw"]
            if selected_match["code"] and candidate_title:
                try:
                    candidate_clean, _, _ = _normalize_for_matching_with_options(candidate_title)
                    jw_score = jellyfish.jaro_winkler_similarity(
                        cleaned_field or "",
                        candidate_clean or "",
                    )
                except Exception:
                    jw_score = None
            primary_result = {
                "field_clean": cleaned_field,
                "cip_code": selected_match["code"],
                "cip_title": selected_match["title_raw"],
                "cip_match_score": selected_match["score"],
                "cip_match_source": selected_match["source"],
                "cip_jaro_winkler": jw_score,
            }
            if collect_debug and debug_record is not None:
                debug_record["selected"] = {
                    "code": selected_match["code"],
                    "title": selected_match["title_raw"],
                    "score": selected_match["score"],
                    "source": selected_match["source"],
                    "stage": debug_record.get("selection_stage"),
                    "internal_score": debug_record.get("selected_score"),
                }
                if selected_match["source"] in {"hardcode", "hardcode_invalid"}:
                    debug_record["hardcode_saved"] = True
        results.append(primary_result)
        current_result = results[-1]

        secondary_entry: Dict[str, Any] = {
            "field_clean_2": None,
            "cip_code_2": None,
            "cip_title_2": None,
            "cip_match_score_2": None,
            "cip_match_source_2": "unmatched",
            "cip_jaro_winkler_2": None,
        }

        split_pairs = _extract_split_pairs(raw_value)
        split_debug_entries: List[Dict[str, Any]] = []
        primary_split_info: Optional[Dict[str, Any]] = None
        secondary_split_info: Optional[Dict[str, Any]] = None

        def evaluate_split_segment(segment_text: str, segment_label: str) -> Dict[str, Any]:
            clean, tokens, _ = _normalize_for_matching_with_options(segment_text)
            tokens_tuple = tuple(tokens)
            info: Dict[str, Any] = {
                "label": segment_label,
                "raw": segment_text,
                "clean": clean,
                "token_count": len(tokens_tuple),
                "match": None,
            }
            if not clean or not tokens_tuple:
                return info
            candidates = _compute_stage1_candidates_for_tokens(
                tokens_tuple,
                entry_matrix,
                vocab,
                token_weights,
                use_full_stage1,
                lsh_index,
                catalog,
            )
            info["candidate_count"] = len(candidates)
            match = _match_field_against_catalog(
                clean,
                tokens_tuple,
                catalog,
                token_weights,
                candidates,
                catalog_titles,
                stage1_max_ngram,
                debug=None,
            )
            info["match"] = match
            return info

        for label, left_text, right_text in split_pairs:
            left_info = evaluate_split_segment(left_text, f"{label}_part1")
            right_info = evaluate_split_segment(right_text, f"{label}_part2")
            split_debug_entries.extend([
                {
                    "label": left_info["label"],
                    "raw": left_info["raw"],
                    "clean": left_info["clean"],
                    "token_count": left_info.get("token_count"),
                    "match": left_info.get("match"),
                },
                {
                    "label": right_info["label"],
                    "raw": right_info["raw"],
                    "clean": right_info["clean"],
                    "token_count": right_info.get("token_count"),
                    "match": right_info.get("match"),
                },
            ])
            if left_info.get("match") or right_info.get("match"):
                primary_split_info = left_info
                secondary_split_info = right_info
                break

        if primary_split_info and primary_split_info.get("match"):
            match_info = primary_split_info["match"]
            clean_value = primary_split_info["clean"]
            jw_primary: Optional[float] = None
            if match_info["code"] and match_info["title_raw"]:
                try:
                    candidate_clean, _, _ = _normalize_for_matching_with_options(match_info["title_raw"])
                    jw_primary = jellyfish.jaro_winkler_similarity(
                        clean_value or "",
                        candidate_clean or "",
                    )
                except Exception:
                    jw_primary = None
            current_result.update({
                "field_clean": clean_value,
                "cip_code": match_info["code"],
                "cip_title": match_info["title_raw"],
                "cip_match_score": match_info["score"],
                "cip_match_source": f"split_{match_info['source']}",
                "cip_jaro_winkler": jw_primary,
            })
            if collect_debug and debug_record is not None:
                debug_record["selected"] = {
                    "code": match_info["code"],
                    "title": match_info["title_raw"],
                    "score": match_info["score"],
                    "source": f"split_{match_info['source']}",
                    "stage": debug_record.get("selection_stage"),
                    "internal_score": match_info["score"],
                }

        if secondary_split_info:
            secondary_entry["field_clean_2"] = secondary_split_info["clean"]
            match_info = secondary_split_info.get("match")
            if match_info:
                jw_secondary: Optional[float] = None
                if match_info["code"] and match_info["title_raw"]:
                    try:
                        candidate_clean, _, _ = _normalize_for_matching_with_options(match_info["title_raw"])
                        jw_secondary = jellyfish.jaro_winkler_similarity(
                            secondary_split_info["clean"] or "",
                            candidate_clean or "",
                        )
                    except Exception:
                        jw_secondary = None
                secondary_entry.update({
                    "cip_code_2": match_info["code"],
                    "cip_title_2": match_info["title_raw"],
                    "cip_match_score_2": match_info["score"],
                    "cip_match_source_2": f"split_{match_info['source']}",
                    "cip_jaro_winkler_2": jw_secondary,
                })

        secondary_results.append(secondary_entry)
        if collect_debug and debug_record is not None:
            if split_debug_entries:
                debug_record["split_attempts"] = split_debug_entries
            debug_info.append(debug_record)

    for idx, result in enumerate(results):
        field_clean = result["field_clean"]
        if not field_clean:
            continue
        if result["cip_match_source"] in {"hardcode", "hardcode_invalid", "hardcode_interactive"}:
            continue
        cip_code = result["cip_code"]
        if not cip_code:
            continue
        if field_clean in pending_match_records and not pending_match_records[field_clean].get(
            "require_confirmation", True
        ):
            continue
        raw_value = df.iloc[idx][field_column]
        score = result.get("cip_match_score")
        cip_title = result.get("cip_title")
        debug_record = debug_info[idx] if collect_debug else None
        register_pending_match(
            field_clean,
            cip_code,
            field_raw=raw_value,
            source=result.get("cip_match_source", ""),
            score=score,
            cip_title=cip_title,
            debug_record=debug_record,
            require_confirmation=True,
            immediate=False,
            jw_score=result.get("cip_jaro_winkler"),
        )

    saved_match_keys: Set[str] = set()
    updated_entries = [
        info for info in pending_match_records.values() if info.get("changed", False)
    ]
    if updated_entries:
        auto_save_entries = [
            info for info in updated_entries if not info.get("require_confirmation", True)
        ]
        confirm_entries = [
            info for info in updated_entries if info.get("require_confirmation", True)
        ]

        entries_to_save: List[Dict[str, Any]] = []
        if auto_save_entries:
            entries_to_save.extend(auto_save_entries)

        eligible_confirm_entries: List[Dict[str, Any]] = []
        if confirm_entries:
            if match_confirm_min_score is None:
                eligible_confirm_entries = confirm_entries
            else:
                eligible_confirm_entries = []
                for info in confirm_entries:
                    if info.get("is_invalid"):
                        eligible_confirm_entries.append(info)
                        continue
                    jw = info.get("jw_score")
                    if jw is None:
                        jw = 0.0
                    if jw >= match_confirm_min_score:
                        eligible_confirm_entries.append(info)

            if interactive_hardcode and match_confirm_min_score is not None:
                if eligible_confirm_entries:
                    print(
                        f"\n{len(eligible_confirm_entries)} pending match update(s) meet the minimum "
                        f"Jaro-Winkler threshold ({match_confirm_min_score})."
                    )
                    print("Pending updates:")
                    for idx, info in enumerate(eligible_confirm_entries, start=1):
                        field_display = info.get("field_clean", "")
                        raw_display = _coerce_text(info.get("field_raw", ""))
                        if info.get("is_invalid"):
                            print(f"  {idx}. {field_display} -> INVALID (raw: {raw_display})")
                        else:
                            title = info.get("cip_title") or ""
                            jw = info.get("jw_score")
                            score_str = f"{jw:.3f}" if isinstance(jw, (int, float)) else "n/a"
                            print(
                                f"  {idx}. {field_display} -> {info.get('cip_code')} "
                                f"({title}) score={score_str} (raw: {raw_display})"
                            )
                    response = input("Save these match updates? [y/N]: ").strip().lower()
                    if response in {"y", "yes"}:
                        entries_to_save.extend(eligible_confirm_entries)
                    else:
                        print("Match updates were not saved.")
                else:
                    print(
                        "\nNo pending match updates met the minimum Jaro-Winkler threshold; nothing was saved."
                    )
            else:
                entries_to_save.extend(eligible_confirm_entries)

        if entries_to_save:
            _save_hardcoded_matches(hardcode_path, build_match_mapping(entries_to_save))
            saved_match_keys = {info["key"] for info in entries_to_save}

    if pending_match_records:
        for info in pending_match_records.values():
            saved = info["key"] in saved_match_keys
            for record in info.get("debug_records", []):
                record["hardcode_saved"] = saved

    result_df = df.copy()
    result_df["field_clean"] = [item["field_clean"] for item in results]
    result_df["cip_code"] = [item["cip_code"] for item in results]
    result_df["cip_title"] = [item["cip_title"] for item in results]
    result_df["cip_match_score"] = [item["cip_match_score"] for item in results]
    result_df["cip_match_source"] = [item["cip_match_source"] for item in results]
    result_df["cip_jaro_winkler"] = [item["cip_jaro_winkler"] for item in results]
    result_df["field_clean_2"] = [item["field_clean_2"] for item in secondary_results]
    result_df["cip_code_2"] = [item["cip_code_2"] for item in secondary_results]
    result_df["cip_title_2"] = [item["cip_title_2"] for item in secondary_results]
    result_df["cip_match_score_2"] = [item["cip_match_score_2"] for item in secondary_results]
    result_df["cip_match_source_2"] = [item["cip_match_source_2"] for item in secondary_results]
    result_df["cip_jaro_winkler_2"] = [item["cip_jaro_winkler_2"] for item in secondary_results]

    if return_debug:
        return result_df, debug_info
    return result_df


def match_fields_to_cip(
    df: pd.DataFrame,
    cip_path: Union[str, Path] = Path(root) / "data/crosswalks/cip/CIPCode2020.csv",
    digit_length: int = 6,
    field_column: str = "field_raw",
    cip_code_column: str = "CIPCode",
    cip_title_column: str = "CIPTitle",
    cip_examples_column: str = "Examples",
    cip_definition_column: str = "CIPDefinition",
    *,
    use_cache: bool = True,
    cache_path: Optional[Union[str, Path]] = _DEFAULT_CIP_CACHE_PATH,
    cache_only: bool = False,
    **kwargs: Any,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
    """Version of `match_fields_to_cip` that optionally reuses persisted matches.

    Parameters
    ----------
    cache_only : bool, default False
        When True, return cached matches (if present) and skip any new CIP lookups.
        Rows missing from the cache are returned as unmatched.
    """

    return_debug = bool(kwargs.get("return_debug", False))
    effective_use_cache = use_cache and cache_path is not None and (cache_only or not return_debug)
    if cache_only and not effective_use_cache:
        raise ValueError("cache_only=True requires use_cache=True and a valid cache_path.")
    cache_file: Optional[Path] = None
    cache_initialized = False

    working = df.copy()
    working = working.reset_index(drop=False).rename(columns={"index": "__orig_index"})

    if effective_use_cache:
        cache_file = Path(cache_path).expanduser()
        _ensure_global_cache(cache_file)
        cache_initialized = True
        working["__cache_key"] = working[field_column].apply(_cip_cache_key_from_value)
    else:
        working["__cache_key"] = None

    cached_records: Dict[int, Dict[str, Any]] = {}
    needs_match_mask = [True] * len(working)

    if effective_use_cache:
        with _GLOBAL_CIP_CACHE_LOCK:
            for idx, key in enumerate(working["__cache_key"]):
                if key and key in _GLOBAL_CIP_CACHE:
                    cached_copy = dict(_GLOBAL_CIP_CACHE[key])
                    cached_copy["field_raw"] = working.loc[idx, field_column]
                    cached_records[idx] = cached_copy
                    needs_match_mask[idx] = False

    needs_match_mask = pd.Series(needs_match_mask, index=working.index)

    fresh_results: Optional[pd.DataFrame] = None
    debug_info: Optional[List[Dict[str, Any]]] = None

    if needs_match_mask.any() and not cache_only:
        subset = working.loc[needs_match_mask].drop(columns="__cache_key")
        subset_result = _match_fields_to_cip_core(
            subset,
            cip_path=cip_path,
            digit_length=digit_length,
            field_column=field_column,
            cip_code_column=cip_code_column,
            cip_title_column=cip_title_column,
            cip_examples_column=cip_examples_column,
            cip_definition_column=cip_definition_column,
            **kwargs,
        )
        if isinstance(subset_result, tuple):
            fresh_results, debug_info = subset_result
        else:
            fresh_results = subset_result
            debug_info = None
        for idx in subset.index:
            row_dict = fresh_results.loc[idx].to_dict()
            cached_records[idx] = row_dict
            key = working.loc[idx, "__cache_key"]
            if effective_use_cache and key:
                cache_entry = {k: v for k, v in row_dict.items() if k != "field_raw"}
                with _GLOBAL_CIP_CACHE_LOCK:
                    _GLOBAL_CIP_CACHE[key] = cache_entry
                cache_initialized = True
    else:
        fresh_results = pd.DataFrame(columns=working.columns)

    final_records: List[Dict[str, Any]] = []
    for idx in working.index:
        original_value = working.loc[idx, field_column]
        record = cached_records.get(idx)
        if record is None:
            base = {
                "field_raw": original_value,
                "field_clean": pd.NA,
                "cip_code": pd.NA,
                "cip_title": pd.NA,
                "cip_match_score": pd.NA,
                "cip_match_source": "unmatched",
                "cip_jaro_winkler": pd.NA,
            }
            record = base
        if field_column not in record:
            record[field_column] = original_value
        if "field_raw" not in record:
            record["field_raw"] = original_value
        record["__orig_index"] = working.loc[idx, "__orig_index"]
        final_records.append(record)

    result_frame = pd.DataFrame.from_records(final_records)
    result_frame = result_frame.sort_values("__orig_index").drop(columns="__orig_index").reset_index(drop=True)

    if effective_use_cache and cache_initialized and cache_file is not None:
        _cache_flush(cache_file)

    if return_debug:
        return result_frame, (debug_info or [])
    return result_frame


def add_hardcoded_cip_match(
    field_text: str,
    cip_identifier: str,
    cip_path: Union[str, Path] = Path(root) / "data/crosswalks/cip/CIPCode2020.csv",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Manually record a hard-coded CIP match for a field text."""

    cleaned_field, tokens, _ = _normalize_for_matching_with_options(field_text)
    if not cleaned_field:
        raise ValueError("Field text produced an empty normalized value.")

    catalog_df = _load_cip_reference(Path(cip_path))
    catalog = _prepare_cip_catalog(
        catalog_df,
        "CIPCode",
        "CIPTitle",
        "Examples",
        "CIPDefinition",
    )
    if not catalog:
        raise ValueError("CIP catalog is empty or invalid.")

    catalog_by_code = {entry["code"]: entry for entry in catalog}
    catalog_by_title = {entry["title_raw"].lower(): entry["code"] for entry in catalog}

    identifier = cip_identifier.strip()
    entry = catalog_by_code.get(identifier)
    if entry is None:
        entry_code = catalog_by_title.get(identifier.lower())
        if entry_code is not None:
            entry = catalog_by_code[entry_code]
    if entry is None:
        raise ValueError(f"CIP identifier '{cip_identifier}' not found in catalog.")

    hardcode_path = Path(output_path) if output_path is not None else _DEFAULT_HARDCODE_PATH
    mapping = _load_hardcoded_matches(hardcode_path)
    mapping[cleaned_field] = entry["code"]
    _save_hardcoded_matches(hardcode_path, mapping)
    return entry["code"]

# # STEP ONE: Read in raw fields and clean
# import duckdb as ddb

# con = ddb.connect()
# raw_fields = con.read_parquet(f"{root}/data/int/wrds_users_sep2.parquet")

# clean_fields = con.sql(f"SELECT field_raw, {help.field_clean_regex_sql('field_raw')} AS field FROM raw_fields WHERE field_raw IS NOT NULL")

# fields_for_matching = con.sql("SELECT DISTINCT field FROM clean_fields").df().sample(100, random_state=1003)

# matched, debug = match_fields_to_cip(fields_for_matching, field_column = 'field', return_debug = True)#, interactive_hardcode=True, match_confirm_min_score=0.95)

#dict = _load_hardcoded_matches(_DEFAULT_HARDCODE_PATH)
