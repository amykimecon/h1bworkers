"""Adaptive fuzzy string matching utilities for clustering university names."""

from __future__ import annotations

import itertools
import json
import math
import os
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Pattern, Sequence, Set, Tuple, Union

try:  # pragma: no cover
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover
    duckdb = None

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

try:
    from text_unidecode import unidecode
except ImportError:  # pragma: no cover - optional dependency
    unidecode = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .preprocess import prepare_normalized_name_parquet, NormalizedName
from .ann import generate_ann_embeddings, build_faiss_index, AnnRetriever


@dataclass(frozen=True)
class PairCandidate:
    """Container for a proposed pair of university names."""

    name_a: str
    name_b: str
    score: float


@dataclass
class LabelledExample:
    """Feature vector and human supplied label for a pair of names."""

    features: np.ndarray
    label: int


@dataclass(frozen=True)
class RepresentativeMetadata:
    """Resolved metadata for a cleaned institution name."""

    institution_id: Optional[str]
    city: Optional[str]
    geo_city_id: Optional[str]
    ipeds_ids: Tuple[str, ...]


GENERIC_SUFFIX_TOKENS = {
    "admin",
    "administration",
    "branch",
    "branches",
    "campus",
    "centre",
    "center",
    "dept",
    "department",
    "division",
    "foundation",
    "group",
    "headquarters",
    "hq",
    "main",
    "office",
    "offices",
    "system",
    "unit",
}
GENERIC_SUFFIX_PHRASES = {
    "main campus",
    "central campus",
}
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_WHITESPACE_RE = re.compile(r"\s+")
_PAREN_PATTERN = re.compile(r"\(([^)]*)\)")
PROGRAM_KEYWORDS = {
    "academy",
    "business",
    "business school",
    "college",
    "college of",
    "department",
    "division",
    "faculty",
    "graduate school",
    "institute",
    "school",
    "school of",
    "program",
    "centre",
    "center",
}
PROGRAM_SUFFIX_RE = re.compile(
    r"\s*(?:[-–,:]\s*)?(?:(?:college|school|faculty|department|division|program|centre|center|institute|academy|graduate school)(?:\b.*)?)$",
    re.IGNORECASE,
)

_COMPOSITE_SEPARATOR_PATTERN = re.compile(r"\s*;+\s*|\s*/\s*|\s*\|\s*|\s*\n+\s*")
_DEGREE_ABBREVIATION_PATTERN = re.compile(
    r"\b("
    r"A\.?A\.?|A\.?S\.?|B\.?A\.?|B\.?S\.?|B\.?Eng\.?|B\.?Ed\.?|M\.?A\.?|M\.?S\.?|M\.?Eng\.?|M\.?Ed\.?|M\.?Div\.?"
    r"|M\.?Phil\.?|MBA|Ph\.?D\.?|D\.?Phil\.?|LL\.?M\.?|LL\.?B\.?|J\.?D\.?|Ed\.?D\.?"
    r")\b",
    re.IGNORECASE,
)
_LOCATION_CLAUSE_PATTERN = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*(?:[A-Z]{2}|[A-Z][a-z]+)\b"
)
_COMPOSITE_ENTITY_KEYWORDS = {
    "university",
    "college",
    "academy",
    "institute",
    "school",
    "seminary",
    "polytechnic",
    "institute of technology",
}

_NON_DEGREE_REGEXES: List[Pattern[str]] = [
    re.compile(
        r"\b(certificates?|certification|credential|nanodegree|micro[-\s]?credential|micro[-\s]?degree|specialization)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(boot\s?camp|short course|summer (school|program|session)|executive (education|program)|professional (education|development)|continuing education|extension program|intensive english|english language (program|center|centre|school)|language (school|centre|center|institute)|real estate|pathway(s)? program|bridge program|study abroad|exchange program|internship program|training (center|centre|school|program)|reskilling program)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(edx|coursera|udemy|udacity|khan academy|general assembly|flatiron school|springboard|le wagon|hyperiondev|simplilearn|pluralsight|new horizons)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bonline (course|program|certificate)\b", re.IGNORECASE),
]

SCHOOL_TYPE_KEYWORDS = {
    "high_school": {
        "high school",
        "secondary school",
        "senior high",
        "lycée",
        "lycee",
        "gymnasium",
        "upper school",
        "hs",
    },
    "community_college": {
        "community college",
        "junior college",
        "jr college",
        "city college",
        "c c",
        "cc",
        "vocational school",
        "technical school",
        "technical college",
        "trade school"
    },
}

GENERIC_NAME_PATTERNS = {
    "university",
    "university of",
    "college",
    "college of",
    "school",
    "school of",
    "school of business",
    "business school",
    "faculty",
    "department",
    "institute",
    "academy",
}
GENERIC_STOP_TOKENS = {
    "university",
    "college",
    "school",
    "faculty",
    "department",
    "business",
    "institute",
    "academy",
    "of",
    "the",
    "and",
}

_TOKEN_DOC_COUNTS: Dict[str, int] = {}
_TOKEN_IDF: Dict[str, float] = {}
_TOKEN_TOTAL_DOCS: int = 0
FEATURE_CACHE: Optional[Dict[Tuple[str, str], np.ndarray]] = None
_GENERIC_NAME_OVERRIDES: Dict[str, bool] = {}
_NON_DEGREE_OVERRIDES: Dict[str, bool] = {}
_MAX_BLOCKING_BUCKET_SIZE = 1000
_MAX_COMMON_TOKEN_FREQUENCY = 10_000
_MAX_JARO_FULL_SCAN_NAMES = 50_000
_AUTO_LIMIT_MIN_NAMES = 50_000
_AUTO_LIMIT_CAP = 5_000_000
_AUTO_LIMIT_MULTIPLIER = 10
_FIRST_LETTER_UNBOUNDED_NAMES = 20_000
_MAX_FIRST_LETTER_BUCKET_PAIRS = 50_000
_MAX_FIRST_LETTER_GLOBAL_PAIRS = 5_000_000
_INSTITUTION_METADATA: Dict[str, Dict[str, Optional[str]]] = {}
_IPEDS_METADATA: Dict[str, Dict[str, Optional[str]]] = {}
_REPRESENTATIVE_METADATA: Dict[str, RepresentativeMetadata] = {}
_RAW_NAME_METADATA: Dict[str, RepresentativeMetadata] = {}
_CITY_ALIAS_DETAILS: Dict[str, Dict[str, Any]] = {}
_CITY_METADATA: Dict[str, Dict[str, Any]] = {}
_MAX_CITY_POP: float = 0.0
_CITY_STOP_TOKENS = {
    "ba",
    "to",
    "in",
    "of",
    "and",
    "university",
    "academy",
    "college",
    "mba",
    "bsc",
    "the",
    "st",
    "area",
    "center",
    "universidad",
    "central",
    "central high",
    "at",
    "valley",
    "jr",
    "sr",
}

FEATURE_NAMES: List[str] = [
    "wratio_raw",
    "token_sort_raw",
    "token_set_raw",
    "partial_ratio_raw",
    "qratio_raw",
    "wratio_ascii",
    "token_sort_ascii",
    "length_rel_diff",
    "token_count_rel_diff",
    "char_overlap",
    "city_same",
    "city_conflict",
    "country_same",
    "country_conflict",
    "idf_shared_sum",
    "idf_shared_min",
    "idf_unique_sum",
    "idf_unique_min",
    "school_type_same",
    "school_type_conflict",
]

NAME_FEATURE_NAMES: List[str] = [
    "char_len",
    "token_count",
    "avg_token_len",
    "max_token_len",
    "token_char_ratio",
    "has_program_keyword",
    "has_generic_stop_token",
    "stopword_ratio",
    "contains_numeric",
    "contains_ampersand",
    "token_diversity",
]

_GENERIC_NAME_MODEL: Optional[Any] = None
_GENERIC_NAME_MODEL_THRESHOLD: float = 0.8
_NON_DEGREE_NAME_MODEL: Optional[Any] = None
_NON_DEGREE_NAME_MODEL_THRESHOLD: float = 0.8


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    processed = unidecode(value) if unidecode else value
    processed = processed.lower()
    processed = _NON_ALNUM_RE.sub(" ", processed)
    processed = _WHITESPACE_RE.sub(" ", processed).strip()
    return processed


def extract_name_features(name: str) -> np.ndarray:
    normalized = _normalize_text(name)
    tokens = [tok for tok in normalized.split() if tok]
    char_len = len(normalized.replace(" ", ""))
    token_count = len(tokens)
    avg_token_len = (sum(len(tok) for tok in tokens) / token_count) if token_count else 0.0
    max_token_len = max((len(tok) for tok in tokens), default=0)
    token_char_ratio = (token_count / (char_len or 1)) if token_count else 0.0
    program_keyword_flag = _bool_to_float(_contains_program_keyword(name))
    generic_stop_count = sum(1 for tok in tokens if tok in GENERIC_STOP_TOKENS)
    has_generic_stop = _bool_to_float(generic_stop_count > 0)
    stopword_ratio = generic_stop_count / (token_count or 1)
    contains_numeric = _bool_to_float(any(ch.isdigit() for ch in name))
    contains_ampersand = _bool_to_float("&" in name)
    token_diversity = (len(set(tokens)) / (token_count or 1)) if token_count else 0.0
    features = [
        float(char_len),
        float(token_count),
        float(avg_token_len),
        float(max_token_len),
        float(token_char_ratio),
        program_keyword_flag,
        has_generic_stop,
        float(stopword_ratio),
        contains_numeric,
        contains_ampersand,
        float(token_diversity),
    ]
    if len(features) != len(NAME_FEATURE_NAMES):
        raise ValueError(
            f"Name feature vector length {len(features)} does not match expected {len(NAME_FEATURE_NAMES)}"
        )
    return np.array(features, dtype=float)


def _name_model_feature_count(model: Any) -> Optional[int]:
    return getattr(model, "n_features_in_", None)


def _predict_name_probability(model: Any, features: np.ndarray) -> float:
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


def fit_name_classifier(examples: Iterable[Tuple[str, int]]) -> LogisticRegression:
    rows = [(name, int(label)) for name, label in examples]
    if not rows:
        raise ValueError("No training data provided for name classifier.")
    labels = {label for _, label in rows}
    if len(labels) < 2:
        raise ValueError("Name classifier training data must contain at least two classes.")
    X = np.vstack([extract_name_features(name) for name, _ in rows])
    y = np.array([label for _, label in rows], dtype=int)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y)
    return model


def set_generic_name_model(model: Optional[Any], threshold: float = 0.8) -> None:
    global _GENERIC_NAME_MODEL, _GENERIC_NAME_MODEL_THRESHOLD
    if model is not None:
        expected = _name_model_feature_count(model)
        if expected is not None and expected != len(NAME_FEATURE_NAMES):
            raise ValueError(
                f"Generic name model expects {expected} features but pipeline produces {len(NAME_FEATURE_NAMES)}."
            )
    _GENERIC_NAME_MODEL = model
    _GENERIC_NAME_MODEL_THRESHOLD = float(min(max(threshold, 0.0), 1.0))


def set_non_degree_name_model(model: Optional[Any], threshold: float = 0.8) -> None:
    global _NON_DEGREE_NAME_MODEL, _NON_DEGREE_NAME_MODEL_THRESHOLD
    if model is not None:
        expected = _name_model_feature_count(model)
        if expected is not None and expected != len(NAME_FEATURE_NAMES):
            raise ValueError(
                f"Non-degree name model expects {expected} features but pipeline produces {len(NAME_FEATURE_NAMES)}."
            )
    _NON_DEGREE_NAME_MODEL = model
    _NON_DEGREE_NAME_MODEL_THRESHOLD = float(min(max(threshold, 0.0), 1.0))


def _contains_program_keyword(text: str) -> bool:
    lowered = text.lower()
    for keyword in PROGRAM_KEYWORDS:
        if keyword in lowered:
            return True
    return False


def _is_generic_name(clean_name: str) -> bool:
    if not clean_name:
        return True
    override = _GENERIC_NAME_OVERRIDES.get(clean_name)
    if override is not None:
        return override
    lowered = clean_name.lower().strip()
    if lowered in GENERIC_NAME_PATTERNS:
        return True
    tokens = [tok for tok in lowered.split() if tok]
    if not tokens:
        return True
    if len(tokens) <= 3 and all(tok in GENERIC_STOP_TOKENS for tok in tokens):
        return True
    # phrases like "university of" + single short token
    if len(tokens) == 3 and tokens[0] == "university" and tokens[1] == "of" and len(tokens[2]) <= 3:
        return True
    return False


def _tokenize_for_stats(name: str) -> Set[str]:
    cleaned = _normalize_text(name)
    if not cleaned:
        return set()
    return {token for token in cleaned.split() if token}


def _clean_composite_segment(segment: str) -> str:
    trimmed = segment.strip(" ,;/|-")
    trimmed = _WHITESPACE_RE.sub(" ", trimmed)
    return trimmed


def _split_composite_institution(raw: str) -> List[str]:
    if not raw:
        return []
    if ";" not in raw and "\n" not in raw and "/" not in raw and "|" not in raw:
        return []
    parts = [
        _clean_composite_segment(part)
        for part in re.split(_COMPOSITE_SEPARATOR_PATTERN, raw)
        if part and _clean_composite_segment(part)
    ]
    if len(parts) < 2:
        return []
    valid_segments: List[str] = []
    for part in parts:
        if len(part) < 6:
            continue
        token_count = len(part.split())
        hints = 0
        if _LOCATION_CLAUSE_PATTERN.search(part):
            hints += 1
        if _DEGREE_ABBREVIATION_PATTERN.search(part):
            hints += 1
        lowered = part.lower()
        if any(keyword in lowered for keyword in _COMPOSITE_ENTITY_KEYWORDS):
            hints += 1
        if token_count >= 4:
            hints += 1
        if hints >= 2 or (hints >= 1 and token_count >= 2 and len(part) >= 6):
            valid_segments.append(part)
    if len(valid_segments) < 2:
        return []
    deduped = list(dict.fromkeys(valid_segments))
    return deduped


def set_feature_cache(cache: Optional[Dict[Tuple[str, str], np.ndarray]]) -> None:
    global FEATURE_CACHE
    FEATURE_CACHE = cache


def set_generic_override(name: str, is_generic: bool) -> None:
    """Override the generic-name heuristic for a particular institution."""

    clean = _normalize_text(name)
    if not clean:
        return
    _GENERIC_NAME_OVERRIDES[clean] = bool(is_generic)


def mark_generic_name(name: str) -> None:
    """Explicitly mark a name as generic for the current session."""

    set_generic_override(name, True)


def unmark_generic_name(name: str) -> None:
    """Force a name to be treated as specific (not generic) for the current session."""

    set_generic_override(name, False)


def clear_generic_overrides() -> None:
    """Remove all manual generic-name overrides."""

    _GENERIC_NAME_OVERRIDES.clear()


def get_generic_overrides() -> Dict[str, bool]:
    """Return a copy of the current generic-name overrides."""

    return dict(_GENERIC_NAME_OVERRIDES)


def set_non_degree_override(name: str, is_non_degree: bool) -> None:
    """Override the non-degree heuristic for a particular institution."""

    clean = _normalize_text(name)
    if not clean:
        return
    _NON_DEGREE_OVERRIDES[clean] = bool(is_non_degree)


def mark_non_degree_name(name: str) -> None:
    """Explicitly mark a name as a non-degree program for the current session."""

    set_non_degree_override(name, True)


def unmark_non_degree_name(name: str) -> None:
    """Force a name to be treated as degree-granting for the current session."""

    set_non_degree_override(name, False)


def clear_non_degree_overrides() -> None:
    """Remove all manual non-degree overrides."""

    _NON_DEGREE_OVERRIDES.clear()


def get_non_degree_overrides() -> Dict[str, bool]:
    """Return a copy of the current non-degree overrides."""

    return dict(_NON_DEGREE_OVERRIDES)


def _is_non_degree_program_name(name: str) -> bool:
    """Return True when the supplied name describes a non-degree program using heuristics."""

    if not name:
        return False
    lowered = name.lower()
    for pattern in _NON_DEGREE_REGEXES:
        if pattern.search(lowered):
            return True

    normalized = _normalize_text(name)
    tokens = normalized.split()
    if not tokens:
        return False

    token_set = set(tokens)

    if ("course" in token_set or "courses" in token_set) and not any(
        keyword in lowered for keyword in ("university", "college", "institute", "school", "academy")
    ):
        return True

    if "program" in token_set:
        if any(
            keyword in lowered
            for keyword in (
                "certificate",
                "summer",
                "executive",
                "online",
                "preparatory",
                "foundation",
                "pathway",
                "language",
                "professional development",
                "continuing education",
            )
        ):
            return True

    return False


def is_non_degree_program_name(name: str) -> bool:
    """Return True when the supplied name describes a non-degree program."""

    if not name:
        return False
    normalized = _normalize_text(name)
    override = _NON_DEGREE_OVERRIDES.get(normalized)
    if override is not None:
        return override
    if _NON_DEGREE_NAME_MODEL is not None:
        features = extract_name_features(name)
        probability = _predict_name_probability(_NON_DEGREE_NAME_MODEL, features)
        if probability >= _NON_DEGREE_NAME_MODEL_THRESHOLD:
            return True
        if probability <= 1.0 - _NON_DEGREE_NAME_MODEL_THRESHOLD:
            return False
    return _is_non_degree_program_name(name)


LOCATION_STOPWORDS = {
    "academy",
    "and",
    "area",
    "arts",
    "at",
    "campus",
    "central",
    "centre",
    "center",
    "college",
    "county",
    "department",
    "district",
    "downtown",
    "east",
    "faculty",
    "for",
    "graduate",
    "high",
    "in",
    "institute",
    "institut",
    "instituto",
    "lower",
    "main",
    "metropolitan",
    "municipal",
    "north",
    "of",
    "polytechnic",
    "prefecture",
    "province",
    "region",
    "school",
    "science",
    "sciences",
    "south",
    "state",
    "technology",
    "the",
    "town",
    "university",
    "upper",
    "village",
    "west",
    "zone",
}

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _data_root() -> Path:
    try:
        from config import root as config_root  # type: ignore

        if config_root:
            return Path(config_root)
    except Exception:
        pass
    return _REPO_ROOT


@lru_cache(maxsize=1)
def _load_country_aliases() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    country_path = _data_root() / "data" / "crosswalks" / "country_dict.json"
    if country_path.exists():
        try:
            with country_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            raw = {}
    else:
        raw = {}

    for key, value in raw.items():
        alias = _normalize_text(key)
        canonical = _normalize_text(value) if isinstance(value, str) else ""
        if alias:
            aliases[alias] = canonical or alias
        if canonical:
            aliases.setdefault(canonical, canonical)
    return aliases


ADDITIONAL_COUNTRY_SYNONYMS = {
    "usa": "united states",
    "u s a": "united states",
    "u s": "united states",
    "us": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "america": "united states",
    "uk": "united kingdom",
    "u k": "united kingdom",
    "u.k.": "united kingdom",
    "great britain": "united kingdom",
    "england": "united kingdom",
    "scotland": "united kingdom",
    "wales": "united kingdom",
    "northern ireland": "united kingdom",
    "uae": "united arab emirates",
    "u a e": "united arab emirates",
    "u. a. e.": "united arab emirates",
    "drc": "democratic republic of the congo",
    "prc": "china",
    "p r china": "china",
    "peoples republic of china": "china",
    "ivory coast": "cote d'ivoire",
    "cote d ivoire": "cote d'ivoire",
    "viet nam": "vietnam",
    "brunei darussalam": "brunei",
    "swaziland": "eswatini",
    "timor leste": "timor-leste",
}


@lru_cache(maxsize=1)
def _load_geonames_aliases() -> Dict[str, str]:
    global _MAX_CITY_POP
    aliases: Dict[str, str] = {}
    data_root = _data_root()

    def register(
        name: Optional[str],
        canonical: Optional[str],
        *,
        kind: str,
        country: Optional[str],
        admin1: Optional[str],
        admin2: Optional[str],
        pop: Optional[int],
    ) -> None:
        if not name:
            return
        alias_norm = _normalize_text(name)
        if not alias_norm:
            return
        canonical_norm = _normalize_text(canonical) if canonical else alias_norm
        aliases.setdefault(alias_norm, canonical_norm)
        if canonical_norm:
            detail = _CITY_ALIAS_DETAILS.get(alias_norm)
            if detail is None or (pop and (detail.get("pop") or 0) < (pop or 0)):
                _CITY_ALIAS_DETAILS[alias_norm] = {
                    "canonical": canonical_norm,
                    "kind": kind,
                    "country": country,
                    "admin1": admin1,
                    "admin2": admin2,
                    "pop": pop,
                }
            meta = _CITY_METADATA.setdefault(
                canonical_norm,
                {
                    "country": country,
                    "admin1": admin1,
                    "admin2": admin2,
                    "pop": pop or 0,
                },
            )
            if pop and pop > (meta.get("pop") or 0):
                meta["pop"] = pop
            if country and not meta.get("country"):
                meta["country"] = country
            if admin1 and not meta.get("admin1"):
                meta["admin1"] = admin1
            if admin2 and not meta.get("admin2"):
                meta["admin2"] = admin2
            if pop and pop > 0:
                _MAX_CITY_POP = max(_MAX_CITY_POP, float(pop))

    cities_path = data_root / "data" / "crosswalks" / "geonames" / "cities500.txt"
    if cities_path.exists():
        try:
            column_names = [
                "geonameid",
                "name",
                "asciiname",
                "altnernatenames",
                "latitude",
                "longitude",
                "featureclass",
                "featurecode",
                "countrycode",
                "cc2",
                "admin1",
                "admin2",
                "admin3",
                "admin4",
                "pop",
                "elev",
                "dem",
                "timezone",
                "mod",
            ]
            for chunk in pd.read_csv(
                cities_path,
                sep="\t",
                names=column_names,
                usecols=["name", "asciiname", "altnernatenames", "countrycode", "admin1", "admin2", "pop"],
                dtype=str,
                keep_default_na=False,
                chunksize=50000,
            ):
                for row in chunk.itertuples(index=False):
                    canonical = row.asciiname or row.name
                    pop_val = _safe_int(getattr(row, "pop", None))
                    register(
                        row.name,
                        canonical,
                        kind="city",
                        country=getattr(row, "countrycode", None),
                        admin1=getattr(row, "admin1", None),
                        admin2=getattr(row, "admin2", None),
                        pop=pop_val,
                    )
                    register(
                        row.asciiname,
                        canonical,
                        kind="city",
                        country=getattr(row, "countrycode", None),
                        admin1=getattr(row, "admin1", None),
                        admin2=getattr(row, "admin2", None),
                        pop=pop_val,
                    )
                    if row.altnernatenames:
                        for alt in row.altnernatenames.split(","):
                            register(
                                alt,
                                canonical,
                                kind="city_altname",
                                country=getattr(row, "countrycode", None),
                                admin1=getattr(row, "admin1", None),
                                admin2=getattr(row, "admin2", None),
                                pop=pop_val,
                            )
        except Exception:
            pass

    admin1_path = data_root / "data" / "crosswalks" / "geonames" / "admin1CodesASCII.txt"
    if admin1_path.exists():
        try:
            admin1 = pd.read_csv(
                admin1_path,
                sep="\t",
                names=["code", "name", "asciiname", "geonameid"],
                dtype=str,
                keep_default_na=False,
            )
            for row in admin1.itertuples(index=False):
                register(
                    row.name,
                    row.name,
                    kind="admin1",
                    country=None,
                    admin1=getattr(row, "code", None),
                    admin2=None,
                    pop=None,
                )
                register(
                    row.asciiname,
                    row.name,
                    kind="admin1",
                    country=None,
                    admin1=getattr(row, "code", None),
                    admin2=None,
                    pop=None,
                )
                if row.code:
                    code_suffix = row.code.split(".")[-1]
                    register(
                        code_suffix,
                        row.name,
                        kind="admin1_abbrev",
                        country=None,
                        admin1=getattr(row, "code", None),
                        admin2=None,
                        pop=None,
                    )
        except Exception:
            pass

    admin2_path = data_root / "data" / "crosswalks" / "geonames" / "admin2Codes.txt"
    if admin2_path.exists():
        try:
            admin2 = pd.read_csv(
                admin2_path,
                sep="\t",
                names=["concatcodes", "name", "asciiname", "geonameid"],
                dtype=str,
                keep_default_na=False,
            )
            for row in admin2.itertuples(index=False):
                register(
                    row.name,
                    row.name,
                    kind="admin2",
                    country=None,
                    admin1=None,
                    admin2=getattr(row, "concatcodes", None),
                    pop=None,
                )
                register(
                    row.asciiname,
                    row.name,
                    kind="admin2",
                    country=None,
                    admin1=None,
                    admin2=getattr(row, "concatcodes", None),
                    pop=None,
                )
                if row.concatcodes:
                    suffix = row.concatcodes.split(".")[-1]
                    register(
                        suffix,
                        row.name,
                        kind="admin2_abbrev",
                        country=None,
                        admin1=None,
                        admin2=getattr(row, "concatcodes", None),
                        pop=None,
                    )
        except Exception:
            pass

    for alias, canonical in ADDITIONAL_CITY_SYNONYMS.items():
        register(
            alias,
            canonical,
            kind="alias",
            country=None,
            admin1=None,
            admin2=None,
            pop=None,
        )
        register(
            canonical,
            canonical,
            kind="alias",
            country=None,
            admin1=None,
            admin2=None,
            pop=None,
        )

    for name in ESSENTIAL_CITY_NAMES:
        register(
            name,
            name,
            kind="alias",
            country=None,
            admin1=None,
            admin2=None,
            pop=None,
        )

    return aliases


ADDITIONAL_CITY_SYNONYMS = {
    "nyc": "new york",
    "new york city": "new york",
    "la": "los angeles",
    "sf": "san francisco",
    "san fran": "san francisco",
    "dc": "washington",
    "d c": "washington",
    "washington dc": "washington",
    "st louis": "saint louis",
    "st. louis": "saint louis",
    "st petersburg": "saint petersburg",
    "st. petersburg": "saint petersburg",
    "st paul": "saint paul",
    "st. paul": "saint paul",
    "saint john": "st john",
    "st john": "st john",
    "st. john": "st john",
}

ESSENTIAL_CITY_NAMES = {
    "berkeley",
    "los angeles",
    "san francisco",
    "new york",
    "washington",
    "seoul",
    "tokyo",
    "paris",
    "london",
}


@lru_cache(maxsize=1)
def _country_aliases() -> Dict[str, str]:
    aliases = _load_country_aliases().copy()
    for alias, canonical in ADDITIONAL_COUNTRY_SYNONYMS.items():
        alias_norm = _normalize_text(alias)
        canonical_norm = _normalize_text(canonical)
        if alias_norm:
            aliases[alias_norm] = canonical_norm or alias_norm
            if canonical_norm:
                aliases.setdefault(canonical_norm, canonical_norm)
    return aliases


@lru_cache(maxsize=1)
def _city_aliases() -> Dict[str, str]:
    return _load_geonames_aliases().copy()


def _city_alias_details_map() -> Dict[str, Dict[str, Any]]:
    _load_geonames_aliases()
    return _CITY_ALIAS_DETAILS


def _tokenize_for_city_matching(value: str) -> Tuple[List[str], List[int]]:
    ascii_value = unidecode(value) if unidecode else value
    segments = ascii_value.split(",")
    tokens: List[str] = []
    segment_index: List[int] = []
    for seg_idx, segment in enumerate(segments):
        normalized = _normalize_text(segment)
        seg_tokens = [tok for tok in normalized.split() if tok]
        tokens.extend(seg_tokens)
        segment_index.extend([seg_idx] * len(seg_tokens))
    return tokens, segment_index


def _score_city_candidate(
    tokens: Sequence[str],
    detail: Dict[str, Any],
    start_idx: int,
    size: int,
    total_tokens: int,
    segment_index: Sequence[int],
) -> float:
    tokens_from_end = total_tokens - (start_idx + size)
    after_comma = segment_index[start_idx] > 0 if segment_index else False
    location_score = 1.0 if tokens_from_end == 0 else 0.85 if after_comma else max(0.4, 1.0 - 0.08 * tokens_from_end)
    char_length = sum(len(tok) for tok in tokens[start_idx : start_idx + size])
    if char_length <= 2:
        length_score = 0.2
    elif char_length <= 4:
        length_score = 0.5
    elif char_length <= 6:
        length_score = 0.8
    else:
        length_score = 1.0
    kind = detail.get("kind", "city")
    kind_score = 1.0 if kind == "city" else 0.9
    canonical = detail.get("canonical")
    pop = detail.get("pop")
    if (pop is None or pop == 0) and canonical:
        pop = _CITY_METADATA.get(canonical, {}).get("pop")
    if pop and _MAX_CITY_POP:
        pop_score = 0.7 + 0.3 * math.log(pop + 1) / math.log(_MAX_CITY_POP + 1)
    else:
        pop_score = 0.6
    return location_score * length_score * kind_score * pop_score


def _city_candidate_scores(name: str) -> Dict[str, float]:
    tokens, segment_index = _tokenize_for_city_matching(name)
    if not tokens:
        return {}
    alias_details = _city_alias_details_map()
    alias_map = _city_aliases()
    best: Dict[str, float] = {}
    total_tokens = len(tokens)
    max_ngram = min(4, total_tokens)
    for size in range(max_ngram, 0, -1):
        for start in range(total_tokens - size + 1):
            phrase_tokens = tokens[start : start + size]
            if all(tok in _CITY_STOP_TOKENS for tok in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            canonical = alias_map.get(phrase)
            if not canonical:
                continue
            detail = alias_details.get(
                phrase,
                {
                    "canonical": canonical,
                    "kind": "city",
                    "pop": _CITY_METADATA.get(canonical, {}).get("pop"),
                },
            )
            score = _score_city_candidate(tokens, detail, start, size, total_tokens, segment_index)
            if score <= 0:
                continue
            prev = best.get(canonical)
            if prev is None or score > prev:
                best[canonical] = score
    return best


def _member_city_candidates(name: str) -> Tuple[Tuple[str, ...], Dict[str, float]]:
    scored = _city_candidate_scores(name)
    if scored:
        ordered = tuple(
            city for city, _ in sorted(scored.items(), key=lambda item: (-item[1], item[0]))
        )
        return ordered, scored
    legacy = tuple(sorted(_location_signature(name)[0]))
    return legacy, {city: 0.3 for city in legacy}


def _split_institution_aliases(value: Optional[str]) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[|;]", value)
    return [part.strip() for part in parts if part and part.strip()]


def _canonicalize_institution_alias(name: str) -> str:
    cleaned = _remove_parenthetical_sections(name)
    return _normalize_text(cleaned)


def _clean_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _safe_int(value: Any) -> Optional[int]:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


def _update_institution_metadata(
    inst_id: Optional[str],
    *,
    city: Optional[str] = None,
    geo_city_id: Optional[str] = None,
    country_code: Optional[str] = None,
    inst_type: Optional[str] = None,
) -> None:
    if not inst_id:
        return
    record = _INSTITUTION_METADATA.setdefault(
        inst_id,
        {
            "city": None,
            "geo_city_id": None,
            "country_code": None,
            "type": None,
        },
    )
    cleaned_city = _clean_optional_string(city)
    cleaned_geo = _clean_optional_string(geo_city_id)
    cleaned_country = _clean_optional_string(country_code)
    cleaned_type = _clean_optional_string(inst_type)
    if cleaned_city and not record.get("city"):
        record["city"] = cleaned_city
    if cleaned_geo and not record.get("geo_city_id"):
        record["geo_city_id"] = cleaned_geo
    if cleaned_country and not record.get("country_code"):
        record["country_code"] = cleaned_country
    if cleaned_type and not record.get("type"):
        record["type"] = cleaned_type


def _resolve_representative_metadata(
    institution_id: Optional[str],
    ipeds_ids: Iterable[str],
) -> RepresentativeMetadata:
    normalized_ipeds = tuple(sorted({str(value) for value in ipeds_ids if value}))
    city = None
    geo_city_id = None
    if institution_id:
        inst_meta = _INSTITUTION_METADATA.get(institution_id)
        if inst_meta:
            city = _clean_optional_string(inst_meta.get("city")) or city
            geo_city_id = _clean_optional_string(inst_meta.get("geo_city_id")) or geo_city_id
    if not city:
        for unitid in normalized_ipeds:
            ipeds_meta = _IPEDS_METADATA.get(unitid)
            if ipeds_meta and ipeds_meta.get("city"):
                city = _clean_optional_string(ipeds_meta.get("city"))
                break
    return RepresentativeMetadata(
        institution_id,
        city,
        geo_city_id,
        normalized_ipeds,
    )


@lru_cache(maxsize=1)
def _institution_ground_truth() -> Dict[str, Set[str]]:
    aliases: defaultdict[str, Set[str]] = defaultdict(set)
    data_root = _data_root()
    sources = [
        data_root / "data" / "crosswalks" / "institutions.csv",
        data_root / "data" / "crosswalks" / "institutions_acronyms.csv",
        data_root / "data" / "crosswalks" / "institutions_altnames.csv",
    ]

    def register(inst_id: str, *values: Optional[str]) -> None:
        for value in values:
            if not value:
                continue
            for alias in _split_institution_aliases(value):
                canonical = _canonicalize_institution_alias(alias)
                if canonical:
                    aliases.setdefault(canonical, set()).add(inst_id)

    for path in sources:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception:
            continue

        id_column = "id" if "id" in df.columns else None
        if id_column is None:
            continue

        name_cols = [col for col in ("name", "alternative_names", "acronyms") if col in df.columns]
        for row in df.itertuples(index=False):
            inst_id = getattr(row, id_column, None)
            if not inst_id:
                continue
            if path.name == "institutions.csv":
                city_value = getattr(row, "city", None)
                geo_value = getattr(row, "geonames_city_id", None)
                if isinstance(geo_value, str):
                    geo_str = geo_value.strip() or None
                elif geo_value not in (None, ""):
                    geo_str = str(geo_value).strip() or None
                else:
                    geo_str = None
                _update_institution_metadata(
                    inst_id,
                    city=city_value or None,
                    geo_city_id=geo_str,
                    country_code=getattr(row, "country_code", None) or None,
                    inst_type=getattr(row, "type", None) or None,
                )
            register(inst_id, *(getattr(row, col, None) for col in name_cols))

    return aliases


def _split_ipeds_aliases(value: Optional[str]) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[;|,/]+", value)
    return [part.strip() for part in parts if part and part.strip()]


@lru_cache(maxsize=1)
def _ipeds_aliases() -> Dict[str, Set[str]]:
    aliases: defaultdict[str, Set[str]] = defaultdict(set)
    data_root = _data_root()
    csv_path = data_root / "data" / "raw" / "ipeds_cw_2021.csv"
    xls_path = data_root / "data" / "raw" / "ipeds_name_cw_2021.xlsx"
    if not csv_path.exists() or not xls_path.exists():
        missing: List[str] = []
        if not csv_path.exists():
            missing.append(str(csv_path))
        if not xls_path.exists():
            missing.append(str(xls_path))
        print(f"Warning: IPEDS crosswalk files not found ({', '.join(missing)}).")
        return aliases

    try:
        ipeds_univ = pd.read_excel(
            xls_path,
            sheet_name="Crosswalk",
            usecols=[
                "OPEID",
                "IPEDSMatch",
                "PEPSSchname",
                "PEPSLocname",
                "IPEDSInstnm",
            ],
        )
        ipeds_zip = pd.read_csv(
            csv_path,
            usecols=["UNITID", "OPEID", "INSTNM", "CITY", "STABBR", "ZIP", "ALIAS"],
        )
    except Exception as exc:
        print(f"Warning: failed to load IPEDS crosswalk ({exc}).")
        return aliases

    ipeds_univ = ipeds_univ.dropna(subset=["IPEDSMatch", "OPEID"])
    ipeds_univ["UNITID"] = (
        ipeds_univ["IPEDSMatch"].astype(str).str.replace("No match", "-1", regex=False).astype(int)
    )
    ipeds_univ = ipeds_univ[ipeds_univ["UNITID"] != -1].copy()

    merged = ipeds_univ.merge(ipeds_zip, on=["OPEID", "UNITID"], how="left", suffixes=("_cross", ""))

    for row in merged.itertuples(index=False):
        unitid_raw = getattr(row, "UNITID", "")
        unitid = _clean_optional_string(unitid_raw)
        if not unitid:
            continue
        name_candidates = [
            getattr(row, "PEPSSchname", None),
            getattr(row, "PEPSLocname", None),
            getattr(row, "IPEDSInstnm", None),
            getattr(row, "INSTNM", None),
        ]
        alias_field = _clean_optional_string(getattr(row, "ALIAS", None))
        if alias_field:
            name_candidates.extend(_split_ipeds_aliases(alias_field))
        for name in name_candidates:
            canonical = _canonicalize_institution_alias(name) if isinstance(name, str) else ""
            if canonical:
                aliases[canonical].add(unitid)
        city_value = _clean_optional_string(getattr(row, "CITY", None))
        state_value = _clean_optional_string(getattr(row, "STABBR", None))
        zip_value = _clean_optional_string(getattr(row, "ZIP", None))
        entry = _IPEDS_METADATA.setdefault(unitid, {"city": None, "state": None, "zip": None})
        if city_value and not entry.get("city"):
            entry["city"] = city_value
        if state_value and not entry.get("state"):
            entry["state"] = state_value
        if zip_value and not entry.get("zip"):
            entry["zip"] = zip_value

    return aliases


def _split_location_candidates(segment: str) -> List[str]:
    replaced = re.sub(r"\band\b", ",", segment, flags=re.IGNORECASE)
    replaced = replaced.replace("/", ",").replace("&", ",").replace(";", ",")
    return [part.strip() for part in replaced.split(",") if part.strip()]


def _extract_inline_location_segments(text: str) -> List[str]:
    lowered = text.lower()
    segments: List[str] = []
    for marker in (" at ", " in ", " - "):
        if marker in lowered:
            idx = lowered.rfind(marker)
            tail = text[idx + len(marker):].strip()
            normalized_tail = _normalize_text(tail)
            if not normalized_tail:
                continue
            token_count = len(normalized_tail.split())
            if 0 < token_count <= 4:
                segments.append(tail)
    return segments


def _classify_location_candidate(candidate: str) -> Tuple[Optional[str], Optional[str]]:
    normalized = _normalize_text(candidate)
    if not normalized:
        return None, None

    if _contains_program_keyword(candidate):
        return None, None

    country_aliases = _country_aliases()
    if normalized in country_aliases:
        return "country", country_aliases[normalized]

    tokens = normalized.split()
    if len(tokens) > 1:
        for size in range(len(tokens), 0, -1):
            phrase = " ".join(tokens[:size])
            if phrase in country_aliases:
                return "country", country_aliases[phrase]

    city_aliases = _city_aliases()
    if normalized in city_aliases:
        return "city", city_aliases[normalized]
    if len(tokens) > 1:
        for size in range(len(tokens), 0, -1):
            phrase = " ".join(tokens[:size])
            if phrase in city_aliases:
                return "city", city_aliases[phrase]

    filtered_tokens = [tok for tok in tokens if tok not in LOCATION_STOPWORDS]
    filtered = " ".join(filtered_tokens).strip()
    if filtered:
        return "city", filtered

    return None, None


@lru_cache(maxsize=8192)
def _location_signature(name: str) -> Tuple[frozenset[str], frozenset[str]]:
    ascii_name = unidecode(name) if unidecode else name
    ascii_name = ascii_name.strip()
    if not ascii_name:
        return frozenset(), frozenset()

    segments: List[str] = []
    segments.extend(_extract_parenthetical_segments(ascii_name))
    parts = ascii_name.split(",")
    if len(parts) > 1:
        segments.extend(part.strip() for part in parts[1:])
    segments.extend(_extract_inline_location_segments(ascii_name))

    city_tokens: Set[str] = set()
    country_tokens: Set[str] = set()

    for segment in segments:
        if not segment.strip():
            continue
        for candidate in _split_location_candidates(segment):
            kind, canonical = _classify_location_candidate(candidate)
            if kind == "country" and canonical:
                country_tokens.add(canonical)
            elif kind == "city" and canonical:
                city_tokens.add(canonical)

    # Additionally look for in-line city tokens anywhere in the name (not just
    # in parenthetical/comma segments) by matching against the Geonames alias map.
    normalized_tokens = [tok for tok in _normalize_text(ascii_name).split() if tok]
    if normalized_tokens:
        city_aliases = _city_aliases()
        max_window = min(3, len(normalized_tokens))
        for size in range(max_window, 0, -1):
            for idx in range(len(normalized_tokens) - size + 1):
                phrase = " ".join(normalized_tokens[idx : idx + size])
                canonical = city_aliases.get(phrase)
                if canonical:
                    city_tokens.add(canonical)

    return frozenset(city_tokens), frozenset(country_tokens)


def _bool_to_float(value: bool) -> float:
    return 1.0 if value else 0.0


def _location_feature_vector(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> List[float]:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    both_present = bool(set_a) and bool(set_b)

    same_flag = _bool_to_float(bool(intersection))
    conflict_flag = _bool_to_float(both_present and not intersection)

    return [same_flag, conflict_flag]


def _idf_feature_vector(name_a: str, name_b: str) -> List[float]:
    if _TOKEN_TOTAL_DOCS == 0:
        return [0.0, 0.0, 0.0, 0.0]

    tokens_a = _tokenize_for_stats(name_a)
    tokens_b = _tokenize_for_stats(name_b)
    shared = tokens_a & tokens_b
    unique = (tokens_a | tokens_b) - shared

    default_idf = math.log(1 + _TOKEN_TOTAL_DOCS) + 1.0

    shared_idfs = [_TOKEN_IDF.get(token, default_idf) for token in shared]
    unique_idfs = [_TOKEN_IDF.get(token, default_idf) for token in unique]

    shared_sum = float(sum(shared_idfs)) if shared_idfs else 0.0
    shared_min = float(min(shared_idfs)) if shared_idfs else 0.0
    unique_sum = float(sum(unique_idfs)) if unique_idfs else 0.0
    unique_min = float(min(unique_idfs)) if unique_idfs else 0.0

    return [shared_sum, shared_min, unique_sum, unique_min]


def _blocking_token_candidates(tokens: Sequence[str]) -> List[str]:
    if not tokens:
        return []
    filtered = [tok for tok in tokens if tok not in GENERIC_STOP_TOKENS]
    if not filtered:
        filtered = [tok for tok in tokens if tok]
    if not filtered:
        return []
    default_idf = math.log(1 + _TOKEN_TOTAL_DOCS) + 1.0 if _TOKEN_TOTAL_DOCS else 2.0
    scored = {}
    for tok in filtered:
        if len(tok) <= 1:
            continue
        scored[tok] = _TOKEN_IDF.get(tok, default_idf)
    if not scored:
        return []
    sorted_tokens = sorted(
        scored.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    selected = [tok for tok, score in sorted_tokens if score >= 1.25][:3]
    if not selected:
        selected = [tok for tok, _ in sorted_tokens[:1]]
    return selected


def _blocking_keys_for_name(name: str, clean_name: str) -> Set[str]:
    keys: Set[str] = set()
    normalized = clean_name.strip()
    if normalized:
        squashed = normalized.replace(" ", "")
        if squashed:
            keys.add(f"prefix:{squashed[:4]}")
            if len(squashed) > 4:
                keys.add(f"suffix:{squashed[-4:]}")
        tokens = [tok for tok in normalized.split() if tok]
        if tokens:
            keys.add(f"first:{tokens[0]}")
            keys.add(f"last:{tokens[-1]}")
            if len(tokens[0]) >= 4:
                keys.add(f"first4:{tokens[0][:4]}")
            if len(tokens[-1]) >= 4:
                keys.add(f"last4:{tokens[-1][:4]}")
        blocking_tokens = _blocking_token_candidates(tokens)
        for tok in blocking_tokens:
            keys.add(f"tok:{tok}")
        if len(blocking_tokens) >= 2:
            for a, b in itertools.combinations(blocking_tokens[:3], 2):
                keys.add(f"pair:{a}|{b}")
        if len(tokens) >= 2:
            keys.add(f"bi:{tokens[0]}|{tokens[1]}")
    cities, countries = _location_signature(name)
    for city in cities:
        keys.add(f"city:{city}")
    for country in countries:
        keys.add(f"country:{country}")
    if not keys:
        fallback = normalized or _normalize_text(name)
        fallback = fallback.replace(" ", "")
        if fallback:
            keys.add(f"prefix:{fallback[:4]}")
    return keys


def _shares_blocking_key(keys_a: Set[str], keys_b: Set[str]) -> bool:
    if not keys_a or not keys_b:
        return False
    return not keys_a.isdisjoint(keys_b)


def _school_type(text: str) -> str:
    lowered = text.lower()
    for keyword in SCHOOL_TYPE_KEYWORDS["high_school"]:
        if keyword in lowered:
            return "high_school"
    for keyword in SCHOOL_TYPE_KEYWORDS["community_college"]:
        if keyword in lowered:
            return "community_college"
    return "other"


def _school_type_features(name_a: str, name_b: str) -> List[float]:
    type_a = _school_type(name_a)
    type_b = _school_type(name_b)
    same_flag = _bool_to_float(type_a == type_b)
    conflict_flag = _bool_to_float(type_a != type_b and "other" not in {type_a, type_b})
    return [same_flag, conflict_flag]


def is_generic_name(name: str) -> bool:
    clean = _normalize_text(name)
    if not clean:
        return True
    override = _GENERIC_NAME_OVERRIDES.get(clean)
    if override is not None:
        return override
    if _GENERIC_NAME_MODEL is not None:
        features = extract_name_features(name)
        probability = _predict_name_probability(_GENERIC_NAME_MODEL, features)
        if probability >= _GENERIC_NAME_MODEL_THRESHOLD:
            return True
        if probability <= 1.0 - _GENERIC_NAME_MODEL_THRESHOLD:
            return False
    return _is_generic_name(clean)


def normalize_name(name: str) -> str:
    return _normalize_text(name)


def _first_letter_pairs(
    names: Sequence[str],
    max_pairs_per_bucket: Optional[int] = None,
    global_cap: Optional[int] = None,
) -> Set[Tuple[int, int]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, name in enumerate(names):
        clean = normalize_name(name)
        key = ""
        for ch in clean:
            if ch.isalnum():
                key = ch
                break
        if not key:
            continue
        buckets[key].append(idx)

    pairs: Set[Tuple[int, int]] = set()
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        bucket_limit = max_pairs_per_bucket
        if bucket_limit is None:
            # Allow the cap to grow with the bucket size so larger datasets surface more pairs.
            bucket_limit = max(2000, len(idxs) * min(len(idxs) - 1, 100))
        count = 0
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                pair = (a, b) if a < b else (b, a)
                pairs.add(pair)
                count += 1
                if global_cap is not None and len(pairs) >= global_cap:
                    return pairs
                if bucket_limit and count >= bucket_limit:
                    break
            if bucket_limit and count >= bucket_limit:
                break
        if global_cap is not None and len(pairs) >= global_cap:
            return pairs
    return pairs


def _remove_parenthetical_sections(value: str) -> str:
    return _PAREN_PATTERN.sub(" ", value)


def _extract_parenthetical_segments(value: str) -> List[str]:
    return _PAREN_PATTERN.findall(value)


def _should_use_comma_alias(segment: str) -> bool:
    if not segment:
        return False
    tokens = segment.split()
    if not tokens:
        return False
    phrase = " ".join(tokens)
    if phrase in GENERIC_SUFFIX_PHRASES:
        return True
    if len(tokens) <= 2 and all(len(token) <= 3 for token in tokens):
        return True
    return all(token in GENERIC_SUFFIX_TOKENS for token in tokens)


def _name_variants(raw_name: str) -> Tuple[str, Set[str]]:
    ascii_name = unidecode(raw_name) if unidecode else raw_name
    ascii_name = ascii_name.strip()

    normalized_full = _normalize_text(ascii_name)
    no_parenthetical = _normalize_text(_remove_parenthetical_sections(ascii_name))
    canonical = no_parenthetical or normalized_full

    variants: Set[str] = set()
    if canonical:
        variants.add(canonical)
    if normalized_full:
        variants.add(normalized_full)

    for segment in _extract_parenthetical_segments(ascii_name):
        cleaned_segment = _normalize_text(segment)
        if cleaned_segment:
            variants.add(cleaned_segment)

    segments = [segment.strip() for segment in ascii_name.split(",") if segment.strip()]
    if len(segments) > 1:
        head_clean = _normalize_text(segments[0])
        suffix_clean_segments = []
        for suffix in segments[1:]:
            cleaned_suffix = _normalize_text(suffix)
            if cleaned_suffix and _should_use_comma_alias(cleaned_suffix):
                suffix_clean_segments.append(cleaned_suffix)
                variants.add(cleaned_suffix)
        if head_clean and any(_should_use_comma_alias(suffix) for suffix in suffix_clean_segments):
            variants.add(head_clean)

    program_stripped = _strip_program_suffix(ascii_name)
    if program_stripped:
        program_canonical = _normalize_text(program_stripped)
        if program_canonical:
            variants.add(program_canonical)

    return canonical, variants


def _strip_program_suffix(value: str) -> Optional[str]:
    working = value.rstrip()
    stripped = working
    while True:
        match = PROGRAM_SUFFIX_RE.search(stripped)
        if not match:
            break
        stripped = stripped[: match.start()].rstrip(" ,-/")
    stripped = stripped.strip()
    if stripped and stripped.lower() != working.lower():
        return stripped
    return None


def set_token_statistics_from_names(names: Sequence[str]) -> None:
    global _TOKEN_TOTAL_DOCS, _TOKEN_DOC_COUNTS, _TOKEN_IDF
    counter: Counter[str] = Counter()
    total_docs = 0
    for name in names:
        tokens = _tokenize_for_stats(name)
        if not tokens:
            continue
        total_docs += 1
        counter.update(tokens)
    _TOKEN_TOTAL_DOCS = total_docs
    _TOKEN_DOC_COUNTS = dict(counter)
    if total_docs == 0:
        _TOKEN_IDF = {}
    else:
        _TOKEN_IDF = {
            token: math.log((1 + total_docs) / (1 + count)) + 1.0
            for token, count in counter.items()
        }


def prepare_name_groups(
    raw_names: Sequence[str],
    composite_log: Optional[List[Tuple[str, List[str]]]] = None,
    enable_composite_split: bool = True,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Optional[str]]]:
    """Return representative names, their associated raw variants, and known institution ids."""

    enumerated: List[Tuple[int, str]] = []
    for idx, name in enumerate(raw_names):
        if not isinstance(name, str):
            continue
        stripped = name.strip()
        if not stripped:
            continue
        segments: Sequence[str] = []
        if enable_composite_split:
            segments = _split_composite_institution(stripped)
        if segments:
            cleaned_segments = list(dict.fromkeys(segments))
            if composite_log is not None:
                composite_log.append((stripped, cleaned_segments))
            for segment in cleaned_segments:
                enumerated.append((idx, segment))
        else:
            enumerated.append((idx, stripped))
    enumerated.sort(key=lambda item: (-len(item[1]), item[0]))

    global _REPRESENTATIVE_METADATA, _RAW_NAME_METADATA

    alias_to_canonical: Dict[str, str] = {}
    canonical_to_raw: "OrderedDict[str, List[str]]" = OrderedDict()
    canonical_to_rep: Dict[str, str] = {}
    canonical_to_id: Dict[str, Optional[str]] = {}
    canonical_to_metadata: Dict[str, RepresentativeMetadata] = {}
    canonical_to_ipeds: Dict[str, Set[str]] = {}
    institution_aliases = _institution_ground_truth()
    ipeds_aliases = _ipeds_aliases()

    for _, raw in enumerated:
        canonical, variants = _name_variants(raw)
        if not canonical:
            continue

        openalex_ids: Set[str] = set()
        ipeds_ids: Set[str] = set()
        for variant in variants:
            openalex_ids.update(institution_aliases.get(variant, set()))
            ipeds_ids.update(ipeds_aliases.get(variant, set()))

        canonical_key = None
        assigned_id: Optional[str] = None

        if len(openalex_ids) == 1:
            assigned_id = next(iter(openalex_ids))
            canonical_key = f"openalex::{assigned_id}"
        else:
            for variant in variants:
                if variant in alias_to_canonical:
                    canonical_key = alias_to_canonical[variant]
                    assigned_id = canonical_to_id.get(canonical_key)
                    break
            if canonical_key is None:
                canonical_key = canonical

        if assigned_id is None and is_generic_name(raw):
            assigned_id = f"generic::{canonical_key}::{raw}"

        if canonical_key not in canonical_to_raw:
            canonical_to_raw[canonical_key] = []
            canonical_to_rep[canonical_key] = raw
            canonical_to_id[canonical_key] = assigned_id
            canonical_to_ipeds[canonical_key] = set(ipeds_ids)
            canonical_to_metadata[canonical_key] = _resolve_representative_metadata(
                assigned_id,
                canonical_to_ipeds[canonical_key],
            )
        else:
            canonical_to_ipeds.setdefault(canonical_key, set()).update(ipeds_ids)
            canonical_to_metadata[canonical_key] = _resolve_representative_metadata(
                canonical_to_id.get(canonical_key),
                canonical_to_ipeds[canonical_key],
            )

        canonical_to_raw[canonical_key].append(raw)

        for variant in variants:
            if variant and variant not in alias_to_canonical:
                alias_to_canonical[variant] = canonical_key

    representatives = [canonical_to_rep[canonical] for canonical in canonical_to_raw.keys()]
    representative_to_members = {
        canonical_to_rep[canonical]: members
        for canonical, members in canonical_to_raw.items()
    }
    representative_to_id = {
        canonical_to_rep[canonical]: canonical_to_id[canonical]
        for canonical in canonical_to_raw.keys()
    }

    _REPRESENTATIVE_METADATA = {}
    _RAW_NAME_METADATA = {}
    for canonical, rep in canonical_to_rep.items():
        metadata = canonical_to_metadata.get(canonical)
        if metadata is None:
            metadata = _resolve_representative_metadata(None, [])
        _REPRESENTATIVE_METADATA[rep] = metadata
        for raw in canonical_to_raw.get(canonical, []):
            _RAW_NAME_METADATA[raw] = metadata

    return representatives, representative_to_members, representative_to_id


class UnionFind:
    """Disjoint-set union structure used for building clusters."""

    def __init__(self, elements: Iterable[str]) -> None:
        self.parent: Dict[str, str] = {elem: elem for elem in elements}
        self.rank: Dict[str, int] = {elem: 0 for elem in elements}

    def find(self, element: str) -> str:
        root = self.parent[element]
        if root != element:
            root = self.find(root)
            self.parent[element] = root
        return root

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]
        if rank_a < rank_b:
            self.parent[root_a] = root_b
        elif rank_a > rank_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

    def groups(self) -> Dict[str, List[str]]:
        clusters: Dict[str, List[str]] = {}
        for element in self.parent:
            root = self.find(element)
            clusters.setdefault(root, []).append(element)
        return clusters


class StreamingUnionFind:
    """Array-backed disjoint-set union optimized for large streaming workloads."""

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError("Union-find size must be non-negative.")
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int16)

    def find(self, idx: int) -> int:
        parent = self.parent
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(self, a: int, b: int) -> None:
        parent = self.parent
        rank = self.rank
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = rank[root_a]
        rank_b = rank[root_b]
        if rank_a < rank_b:
            parent[root_a] = root_b
        elif rank_a > rank_b:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    def compress(self) -> None:
        parent = self.parent
        for idx in range(parent.shape[0]):
            parent[idx] = self.find(idx)

    def groups(self, labels: Sequence[str]) -> Dict[str, List[str]]:
        clusters: Dict[int, List[str]] = {}
        for idx, label in enumerate(labels):
            root = self.find(idx)
            clusters.setdefault(root, []).append(label)
        return {labels[root]: members for root, members in clusters.items()}


def compute_features(name_a: str, name_b: str) -> np.ndarray:
    """Return a feature vector combining multiple similarity metrics."""

    cache = FEATURE_CACHE
    if cache is not None:
        cached = cache.get((name_a, name_b))
        if cached is not None:
            return cached.copy()
        cached_rev = cache.get((name_b, name_a))
        if cached_rev is not None:
            return cached_rev.copy()

    name_a_lower = name_a.lower()
    name_b_lower = name_b.lower()

    ratios = [
        fuzz.WRatio(name_a, name_b),
        fuzz.token_sort_ratio(name_a, name_b),
        fuzz.token_set_ratio(name_a, name_b),
        fuzz.partial_ratio(name_a, name_b),
        fuzz.QRatio(name_a, name_b),
    ]

    if unidecode is not None:
        translit_a = unidecode(name_a)
        translit_b = unidecode(name_b)
        ratios.extend([
            fuzz.WRatio(translit_a, translit_b),
            fuzz.token_sort_ratio(translit_a, translit_b),
        ])
    else:
        ratios.extend([0.0, 0.0])

    length_a = len(name_a)
    length_b = len(name_b)
    avg_len = (length_a + length_b) / 2 or 1
    length_features = [
        abs(length_a - length_b) / avg_len,
    ]

    token_count_a = len(name_a_lower.split())
    token_count_b = len(name_b_lower.split())
    token_sum = token_count_a + token_count_b
    token_features = [
        abs(token_count_a - token_count_b) / (token_sum or 1),
    ]

    char_overlap = len(set(name_a_lower) & set(name_b_lower)) / (len(set(name_a_lower) | set(name_b_lower)) or 1)

    cities_a, countries_a = _location_signature(name_a)
    cities_b, countries_b = _location_signature(name_b)

    city_features = _location_feature_vector(cities_a, cities_b)
    country_features = _location_feature_vector(countries_a, countries_b)
    idf_features = _idf_feature_vector(name_a, name_b)

    school_features = _school_type_features(name_a, name_b)

    features = (
        ratios
        + length_features
        + token_features
        + [char_overlap]
        + city_features
        + country_features
        + idf_features
        + school_features
    )
    result = np.array(features, dtype=float)
    if result.shape[0] != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature vector length {result.shape[0]} does not match expected {len(FEATURE_NAMES)}"
        )
    if cache is not None:
        cache[(name_a, name_b)] = result
        cache[(name_b, name_a)] = result
    return result


def _effective_candidate_limit(total_names: int, requested: Optional[int]) -> Optional[int]:
    """Normalize the requested candidate limit for very large datasets."""

    if requested is not None:
        return requested
    if total_names <= _AUTO_LIMIT_MIN_NAMES:
        return None
    scaled = max(50_000, total_names * _AUTO_LIMIT_MULTIPLIER)
    return min(_AUTO_LIMIT_CAP, scaled)


def _first_letter_pair_limits(
    total_names: int,
    max_candidates: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    """Return per-bucket and global limits for the first-letter fallback."""

    if total_names <= _FIRST_LETTER_UNBOUNDED_NAMES:
        return None, None
    if max_candidates is None:
        global_cap = _MAX_FIRST_LETTER_GLOBAL_PAIRS
    else:
        global_cap = max_candidates
    bucket_cap = max(
        2_000,
        min(_MAX_FIRST_LETTER_BUCKET_PAIRS, global_cap // max(1, 26)),
    )
    return bucket_cap, global_cap


def _should_use_jaro_full_scan(total_names: int) -> bool:
    """Whether to run the quadratic DuckDB Jaro comparison step."""

    return total_names <= _MAX_JARO_FULL_SCAN_NAMES


def generate_pair_candidates(
    names: Sequence[str],
    max_candidates: Optional[int] = None,
    token_overlap_threshold: int = 1,
    jaro_threshold: float = 0.92,
) -> List[PairCandidate]:
    """Generate candidate pairs using token overlap and similarity heuristics.

    Extremely large inputs are automatically throttled to keep runtime reasonable.
    """

    if not names:
        return []

    effective_limit = _effective_candidate_limit(len(names), max_candidates)

    if duckdb is not None:
        try:
            return _generate_candidates_duckdb(
                names,
                max_candidates=effective_limit,
                token_overlap_threshold=token_overlap_threshold,
                jaro_threshold=jaro_threshold,
            )
        except Exception as exc:  # pragma: no cover - fallback safety
            print(f"DuckDB candidate generation failed ({exc}); falling back to Python implementation.")
    return _generate_candidates_fallback(
        names,
        max_candidates=effective_limit,
        token_overlap_threshold=token_overlap_threshold,
        jaro_threshold=jaro_threshold,
    )


def _generate_candidates_duckdb(
    names: Sequence[str],
    max_candidates: Optional[int],
    token_overlap_threshold: int,
    jaro_threshold: float,
) -> List[PairCandidate]:
    total_names = len(names)
    df = pd.DataFrame(
        {
            "idx": range(len(names)),
            "name": list(names),
            "clean": [_normalize_text(name) for name in names],
        }
    )
    df["is_generic"] = df["name"].apply(is_generic_name).astype(int)
    cleaned_tokens = [set(clean.split()) for clean in df["clean"].tolist()]
    cleaned = df["clean"].tolist()
    blocking_keys_cache = [
        _blocking_keys_for_name(name, clean)
        for name, clean in zip(names, cleaned)
    ]
    conn = duckdb.connect()
    conn.register("candidate_names_df", df)
    freq_cap = max(5, int(total_names * 0.05))
    freq_cap = min(_MAX_COMMON_TOKEN_FREQUENCY, freq_cap)
    limit_clause = ""
    if max_candidates is not None:
        limit_clause = f" LIMIT {int(max_candidates)}"
    use_jaro = _should_use_jaro_full_scan(total_names)
    if not use_jaro:
        print(
            f"DuckDB candidate generation: skipping Jaro all-pairs scan for {total_names:,} names "
            f"(>{_MAX_JARO_FULL_SCAN_NAMES:,})."
        )
    jaro_cte = ""
    pair_source = "token_pairs"
    if use_jaro:
        jaro_cte = """
        ,
        jaro_pairs AS (
            SELECT b1.idx AS idx_a, b2.idx AS idx_b
            FROM base b1
            JOIN base b2 ON b1.idx < b2.idx
            WHERE b1.is_generic = 0 AND b2.is_generic = 0
              AND jaro_similarity(b1.clean_norm, b2.clean_norm) >= ?
        ),
        all_pairs AS (
            SELECT idx_a, idx_b FROM token_pairs
            UNION
            SELECT idx_a, idx_b FROM jaro_pairs
        )
        """
        pair_source = "all_pairs"
    query = f"""
        WITH base AS (
            SELECT idx, name, clean, is_generic,
                   regexp_replace(clean, '\\s+', ' ', 'g') AS clean_norm,
                   regexp_split_to_array(clean, ' ') AS tokens
            FROM candidate_names_df
        ),
        exploded AS (
            SELECT idx, token
            FROM base, UNNEST(tokens) AS u(token)
            WHERE LENGTH(token) > 1
        ),
        filtered_tokens AS (
            SELECT token
            FROM exploded
            GROUP BY token
            HAVING COUNT(*) BETWEEN 1 AND ?
        ),
        token_pairs AS (
            SELECT e1.idx AS idx_a, e2.idx AS idx_b, COUNT(*) AS overlap
            FROM exploded e1
            JOIN exploded e2
              ON e1.token = e2.token AND e1.idx < e2.idx
            JOIN base b1 ON b1.idx = e1.idx
            JOIN base b2 ON b2.idx = e2.idx
            WHERE e1.token IN (SELECT token FROM filtered_tokens)
              AND b1.is_generic = 0 AND b2.is_generic = 0
            GROUP BY e1.idx, e2.idx
            HAVING COUNT(*) >= ?
        ){jaro_cte}
        SELECT idx_a, idx_b FROM {pair_source}{limit_clause}
    """
    params: List[Union[int, float]] = [freq_cap, token_overlap_threshold]
    if use_jaro:
        params.append(jaro_threshold)
    try:
        pairs = conn.execute(
            query,
            params,
        ).fetchall()
    finally:
        conn.unregister("candidate_names_df")
        conn.close()
    pair_set: Set[Tuple[int, int]] = set()
    pair_budget = max_candidates
    for idx_a, idx_b in pairs:
        a, b = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
        if not _shares_blocking_key(blocking_keys_cache[a], blocking_keys_cache[b]):
            continue
        pair_set.add((a, b))
        if pair_budget is not None and len(pair_set) >= pair_budget:
            break
    if pair_budget is None or len(pair_set) < pair_budget:
        bucket_limit, global_cap = _first_letter_pair_limits(total_names, pair_budget)
        for idx_a, idx_b in _first_letter_pairs(
            names,
            max_pairs_per_bucket=bucket_limit,
            global_cap=global_cap,
        ):
            if not _shares_blocking_key(blocking_keys_cache[idx_a], blocking_keys_cache[idx_b]):
                continue
            pair = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
            pair_set.add(pair)
            if pair_budget is not None and len(pair_set) >= pair_budget:
                break
    if not pair_set:
        raise RuntimeError(
            "duckdb candidate generation produced no pairs; consider adjusting candidate thresholds or regenerating cache"
        )

    candidates: List[PairCandidate] = []
    for idx_a, idx_b in pair_set:
        tokens_a = cleaned_tokens[idx_a]
        tokens_b = cleaned_tokens[idx_b]
        if not tokens_a or not tokens_b:
            continue
        name_a = names[idx_a]
        name_b = names[idx_b]
        score = fuzz.WRatio(name_a, name_b)
        overlap_count = len(tokens_a & tokens_b)
        meets_overlap = overlap_count >= token_overlap_threshold
        meets_jaro = score >= jaro_threshold * 100
        if not (meets_overlap or meets_jaro):
            continue
        candidates.append(PairCandidate(name_a, name_b, score))
        if max_candidates is not None and len(candidates) >= max_candidates:
            break
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _generate_candidates_fallback(
    names: Sequence[str],
    max_candidates: Optional[int],
    token_overlap_threshold: int,
    jaro_threshold: float,
) -> List[PairCandidate]:
    cleaned = [_normalize_text(name) for name in names]
    token_sets: List[Set[str]] = [set(clean.split()) for clean in cleaned]
    generic_flags = [is_generic_name(name) for name in names]
    blocking_keys_cache: List[Set[str]] = []
    blocking_buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, clean in enumerate(cleaned):
        keys = _blocking_keys_for_name(names[idx], clean)
        blocking_keys_cache.append(keys)
        for key in keys:
            blocking_buckets[key].append(idx)

    pair_scores: Dict[Tuple[int, int], float] = {}
    max_pairs = max_candidates if max_candidates is not None else float("inf")

    for key, indices in blocking_buckets.items():
        if len(indices) < 2:
            continue
        if _MAX_BLOCKING_BUCKET_SIZE and len(indices) > _MAX_BLOCKING_BUCKET_SIZE:
            continue
        unique_indices = sorted(set(indices))
        for i in range(len(unique_indices)):
            idx_a = unique_indices[i]
            if generic_flags[idx_a]:
                continue
            tokens_a = token_sets[idx_a]
            if not tokens_a:
                continue
            for j in range(i + 1, len(unique_indices)):
                idx_b = unique_indices[j]
                if generic_flags[idx_b]:
                    continue
                if not _shares_blocking_key(blocking_keys_cache[idx_a], blocking_keys_cache[idx_b]):
                    continue
                tokens_b = token_sets[idx_b]
                if not tokens_b:
                    continue
                pair = (min(idx_a, idx_b), max(idx_a, idx_b))
                if pair in pair_scores:
                    continue
                score = fuzz.WRatio(names[idx_a], names[idx_b])
                overlap_count = len(tokens_a & tokens_b)
                meets_overlap = overlap_count >= token_overlap_threshold
                meets_jaro = score >= jaro_threshold * 100
                if not (meets_overlap or meets_jaro):
                    continue
                pair_scores[pair] = score
                if len(pair_scores) >= max_pairs:
                    break
            if len(pair_scores) >= max_pairs:
                break
        if len(pair_scores) >= max_pairs:
            break

    # Include first-letter pairs in addition to token/jaro matches
    bucket_limit, global_cap = _first_letter_pair_limits(len(names), max_candidates)
    for idx_a, idx_b in _first_letter_pairs(
        names,
        max_pairs_per_bucket=bucket_limit,
        global_cap=global_cap,
    ):
        if max_candidates is not None and len(pair_scores) >= max_candidates:
            break
        pair = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
        if pair in pair_scores:
            continue
        if not _shares_blocking_key(blocking_keys_cache[idx_a], blocking_keys_cache[idx_b]):
            continue
        tokens_a = token_sets[idx_a]
        tokens_b = token_sets[idx_b]
        if not tokens_a or not tokens_b:
            continue
        score = fuzz.WRatio(names[pair[0]], names[pair[1]])
        overlap_count = len(tokens_a & tokens_b)
        meets_overlap = overlap_count >= token_overlap_threshold
        meets_jaro = score >= jaro_threshold * 100
        if not (meets_overlap or meets_jaro):
            continue
        pair_scores[pair] = score

    candidates = [
        PairCandidate(names[idx_a], names[idx_b], score)
        for (idx_a, idx_b), score in pair_scores.items()
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def fit_classifier(examples: Iterable[LabelledExample]) -> RandomForestClassifier:
    """Fit a random forest classifier on labelled examples."""

    X = np.vstack([ex.features for ex in examples])
    y = np.array([ex.label for ex in examples], dtype=int)
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X, y)
    return model


def score_candidates(
    model: RandomForestClassifier,
    candidates: Iterable[PairCandidate],
    labelled_keys: Iterable[Tuple[str, str]],
) -> List[Tuple[PairCandidate, float]]:
    """Return unlabelled candidates and their predicted match probabilities."""

    labelled_set = set(labelled_keys)
    scored: List[Tuple[PairCandidate, float]] = []
    for candidate in candidates:
        key = (candidate.name_a, candidate.name_b)
        if key in labelled_set:
            continue
        features = compute_features(candidate.name_a, candidate.name_b)
        proba = model.predict_proba(features.reshape(1, -1))
        classes = getattr(model, "classes_", np.array([0, 1]))
        if proba.shape[1] == 1:
            single_class = classes[0]
            probability = float(proba[0, 0])
            if single_class == 1:
                pass  # probability already reflects P(match)
            else:
                probability = 1.0 - probability
        else:
            if 1 in classes:
                idx = int(np.where(classes == 1)[0][0])
                probability = float(proba[0, idx])
            else:
                probability = float(proba[0].max())
        scored.append((candidate, probability))
    return scored


def build_clusters(
    names: Sequence[str],
    model: RandomForestClassifier,
    candidates: Iterable[PairCandidate],
    threshold: float,
) -> Dict[str, List[str]]:
    """Construct clusters by unioning pairs above the threshold."""

    uf = UnionFind(names)
    for candidate, probability in score_candidates(model, candidates, []):
        if probability >= threshold:
            uf.union(candidate.name_a, candidate.name_b)
    return uf.groups()


def clusters_to_frame(
    clusters: Dict[str, List[str]],
    name_metadata: Optional[Dict[str, RepresentativeMetadata]] = None,
    include_city_metadata: bool = True,
) -> pd.DataFrame:
    """Return a tidy DataFrame representation of clusters.

    Parameters
    ----------
    clusters : Mapping of cluster identifiers to their member names.
    include_city_metadata : When true, attach per-member city tags and the union
        of city tags observed within each cluster.
    """

    records: List[Dict[str, Any]] = []
    metadata_lookup = name_metadata or {}
    for cluster_id, members in clusters.items():
        member_city_map: Dict[str, Tuple[str, ...]] = {}
        member_best_city: Dict[str, Optional[str]] = {}
        cluster_cities: Tuple[str, ...] = ()
        member_token_map: Dict[str, Set[str]] = {}

        if include_city_metadata:
            city_stats: Dict[str, Dict[str, float]] = {}
            for member in members:
                member_token_map[member] = set(_normalize_text(member).split())
                meta = metadata_lookup.get(member)
                if meta and meta.city:
                    normalized_city = _normalize_text(meta.city)
                    if normalized_city:
                        member_city_map[member] = (normalized_city,)
                        member_best_city[member] = normalized_city
                        stats = city_stats.setdefault(normalized_city, {"count": 0, "best": 0.0})
                        stats["count"] += 1
                        stats["best"] = max(stats["best"], 1.5)
                        continue
                candidate_tuple, score_map = _member_city_candidates(member)
                member_city_map[member] = candidate_tuple
                best_city = candidate_tuple[0] if candidate_tuple else None
                member_best_city[member] = best_city
                for city, score in score_map.items():
                    stats = city_stats.setdefault(city, {"count": 0, "best": 0.0})
                    if best_city and city == best_city:
                        stats["count"] += 1
                    stats["best"] = max(stats["best"], score)

            for member in members:
                if member_city_map.get(member):
                    continue
                tokens = member_token_map.get(member, set())
                best_city = None
                best_tuple = (0, 0.0)
                for city, stats in city_stats.items():
                    overlap = len(tokens & set(city.split()))
                    candidate_tuple = (overlap, stats["best"])
                    if candidate_tuple > best_tuple:
                        best_tuple = candidate_tuple
                        best_city = city
                if best_city:
                    member_city_map[member] = (best_city,)
                    member_best_city[member] = best_city
                    stats = city_stats.setdefault(best_city, {"count": 0, "best": 0.0})
                    stats["count"] += 1

            cluster_cities = tuple(
                city
                for city, stats in sorted(
                    city_stats.items(),
                    key=lambda item: (-item[1]["count"], -item[1]["best"], item[0]),
                )
                if stats["count"] > 0
            )
        else:
            member_city_map = {member: tuple(sorted(_location_signature(member)[0])) for member in members}

        for member in members:
            record: Dict[str, Any] = {
                "cluster_root": cluster_id,
                "university_name": member,
            }
            if include_city_metadata:
                record["member_cities"] = member_city_map.get(member, tuple())
                record["cluster_cities"] = cluster_cities
            meta = metadata_lookup.get(member)
            record["matched_institution_id"] = meta.institution_id if meta else None
            record["matched_geo_city_id"] = meta.geo_city_id if meta else None
            record["ipeds_ids"] = meta.ipeds_ids if meta else tuple()
            matched_city = meta.city if (meta and meta.city) else None
            if not matched_city:
                matched_city = member_best_city.get(member)
            if not matched_city and cluster_cities:
                matched_city = cluster_cities[0]
            record["matched_city"] = matched_city
            records.append(record)

    if not records:
        columns = [
            "cluster_root",
            "university_name",
            "matched_city",
            "matched_geo_city_id",
            "matched_institution_id",
            "ipeds_ids",
        ]
        if include_city_metadata:
            columns.extend(["member_cities", "cluster_cities"])
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(records)
    frame = frame.sort_values(["cluster_root", "university_name"]).reset_index(drop=True)
    return frame


def get_representative_metadata() -> Dict[str, RepresentativeMetadata]:
    return dict(_REPRESENTATIVE_METADATA)


def get_raw_name_metadata() -> Dict[str, RepresentativeMetadata]:
    return dict(_RAW_NAME_METADATA)


__all__ = [
    "PairCandidate",
    "LabelledExample",
    "RepresentativeMetadata",
    "prepare_name_groups",
    "set_token_statistics_from_names",
    "set_feature_cache",
    "set_generic_override",
    "mark_generic_name",
    "unmark_generic_name",
    "clear_generic_overrides",
    "get_generic_overrides",
    "set_non_degree_override",
    "mark_non_degree_name",
    "unmark_non_degree_name",
    "clear_non_degree_overrides",
    "get_non_degree_overrides",
    "is_generic_name",
    "is_non_degree_program_name",
    "normalize_name",
    "NAME_FEATURE_NAMES",
    "extract_name_features",
    "FEATURE_NAMES",
    "compute_features",
    "generate_pair_candidates",
    "fit_name_classifier",
    "fit_classifier",
    "set_generic_name_model",
    "set_non_degree_name_model",
    "score_candidates",
    "build_clusters",
    "clusters_to_frame",
    "get_representative_metadata",
    "get_raw_name_metadata",
    "prepare_normalized_name_parquet",
    "NormalizedName",
    "generate_ann_embeddings",
    "build_faiss_index",
    "AnnRetriever",
    "StreamingUnionFind",
]
