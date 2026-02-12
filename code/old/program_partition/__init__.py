"""
File Description: Cluster LinkedIn education records into university × major programs using unsupervised learning.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from string import capwords
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Pattern, Set
import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

try:
    from text_unidecode import unidecode as text_unidecode
except ImportError:  # pragma: no cover - optional dependency
    text_unidecode = None

try:
    from googletrans import Translator as GoogleTranslator  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    GoogleTranslator = None

try:
    from deep_translator import GoogleTranslator as DeepGoogleTranslator  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    DeepGoogleTranslator = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import root  # type: ignore  # noqa: E402


@dataclass
class ProgramClusterConfig:
    """Configuration for the clustering pipeline."""

    distance_threshold: float = 0.38
    ngram_range: Tuple[int, int] = (3, 5)
    min_group_size_for_clustering: int = 5
    min_token_length: int = 2
    verbose: bool = True
    analyzer: str = "char_wb"
    min_df: int = 1
    cluster_universities: bool = True
    university_distance_threshold: float = 0.28
    university_ngram_range: Tuple[int, int] = (3, 5)
    university_min_df: int = 1
    university_analyzer: str = "char_wb"
    city_distance_threshold: float = 0.28
    city_ngram_range: Tuple[int, int] = (3, 5)
    city_min_df: int = 1
    city_analyzer: str = "char_wb"
    city_score_cutoff: int = 90
    geonames_cities_path: Optional[Path] = None
    school_type_keywords: Tuple[str, ...] = (
        "law school",
        "school of law",
        "college of law",
        "business school",
        "school of business",
        "school of management",
        "school of medicine",
        "medical school",
        "school of engineering",
        "school of graduate studies",
        "graduate school",
        "school of public policy",
        "public policy school",
        "school of education",
        "school of nursing",
        "nursing school",
    )
    university_feedback_path: Optional[Path] = None
    infer_city_from_university: bool = True
    city_inference_max_ngram: int = 4
    auto_merge_small_variants: bool = True
    small_variant_max_size: int = 10
    small_variant_min_target_size: int = 50
    small_variant_size_ratio: float = 3.0
    small_variant_similarity: int = 94
    enable_transliteration: bool = True
    university_translation_path: Optional[Path] = None
    external_university_clusters_path: Optional[Path] = None
    enable_translation: bool = True
    translation_service: Optional[str] = None
    translation_target_language: str = "en"
    translation_similarity_cutoff: int = 88
    normalization_replacement_catalog: Dict[str, Dict[str, str]] = None  # type: ignore[assignment]
    alias_resolution_min_support: int = 3
    alias_resolution_confidence: float = 0.6
    alias_similarity_cutoff: int = 90
    alias_initialism_min_length: int = 3
    enable_cip_matching: bool = False
    cip_reference_path: Optional[Path] = None
    cip_code_column: Optional[str] = None
    cip_title_column: Optional[str] = None
    cip_description_column: Optional[str] = None
    cip_level_column: Optional[str] = None
    cip_max_matches: int = 1
    cip_similarity_threshold: float = 0.25
    cip_use_llm: bool = False
    cip_llm_resolver: Optional[Callable[[Dict[str, Any], pd.DataFrame], Optional[Dict[str, Any]]]] = None
    enable_university_hints: bool = True


class ProgramClusterer:
    """Cluster LinkedIn education entries into university × major programs."""

    UNIVERSITY_ALIASES: Tuple[Tuple[str, str], ...] = (
        (r"\buniv\b", "university"),
        (r"\buniveristy\b", "university"),
        (r"\bunversity\b", "university"),
        (r"\binstit(ute|ucion)\b", "institute"),
        (r"\binst\b", "institute"),
        (r"\bpolytech\b", "polytechnic"),
        (r"\bcollege of\b", "college"),
    )

    FIELD_STOPWORDS: Tuple[str, ...] = (
        "bachelor",
        "bachelors",
        "master",
        "masters",
        "msc",
        "ms",
        "ma",
        "ba",
        "bs",
        "degree",
        "program",
        "science",
        "sciences",
        "major",
        "minor",
        "honours",
        "honors",
        "with",
    )

    CITY_INFERENCE_STOPWORDS: Tuple[str, ...] = (
        "university",
        "universities",
        "college",
        "colleges",
        "school",
        "schools",
        "law",
        "of",
        "the",
        "and",
        "dept",
        "department",
        "faculty",
        "institute",
        "institut",
        "national",
        "state",
    )

    INITIALISM_STOPWORDS: Tuple[str, ...] = (
        "of",
        "the",
        "and",
        "for",
        "in",
        "at",
        "de",
        "la",
        "el",
        "del",
        "los",
        "las",
        "le",
        "les",
        "di",
        "da",
        "do",
        "dos",
        "du",
        "l",
        "d",
    )

    HIGH_SCHOOL_PATTERNS: Tuple[str, ...] = (
        r"\bhigh\s*school\b",
        r"\bhighschool\b",
        r"\bsecondary\s+school\b",
        r"\bsecondary\s+education\b",
        r"\bprep(aratory)?\s+school\b",
        r"\bpreparatoria\b",
        r"\blycee\b",
        r"\blyceo\b",
        r"\blyceum\b",
        r"\blic(e|é)o\b",
        r"\bgymnasium\b",
    )

    def __init__(self, config: Optional[ProgramClusterConfig] = None) -> None:
        self.cfg = config or ProgramClusterConfig()
        if self.cfg.normalization_replacement_catalog is None:
            self.cfg.normalization_replacement_catalog = self._default_normalization_catalog()

        if isinstance(self.cfg.external_university_clusters_path, str):
            self.cfg.external_university_clusters_path = Path(self.cfg.external_university_clusters_path)
        if isinstance(self.cfg.cip_reference_path, str):
            self.cfg.cip_reference_path = Path(self.cfg.cip_reference_path)
        if self.cfg.cip_reference_path is not None and not self.cfg.enable_cip_matching:
            self.cfg.enable_cip_matching = True

        self._school_type_keywords = tuple(
            sorted({kw.strip().lower() for kw in self.cfg.school_type_keywords if kw.strip()})
        )
        self._manual_cluster_map, self._manual_type_map = self._load_university_feedback()
        (
            self._university_translation_map,
            self._normalized_translation_map,
        ) = self._load_university_translations()
        self._translation_normalized_keys: Tuple[str, ...] = tuple(sorted(self._normalized_translation_map.keys()))
        self._replacement_patterns = self._compile_replacement_patterns(self.cfg.normalization_replacement_catalog)
        self._country_names = self._load_country_names()
        self._translator_info = self._make_translator()
        self._translation_cache: Dict[Tuple[str, str], str] = {}
        self._city_lookup, self._city_lookup_keys = self._load_city_reference()
        self._external_university_clusters = self._load_external_university_clusters()
        if self.cfg.verbose and self._external_university_clusters is not None:
            cluster_count = self._external_university_clusters["university_cluster_id"].nunique()
            print(
                f"External university cluster reference loaded with {cluster_count:,} clusters."
            )
        (
            self._cip_catalog,
            self._cip_vectorizer,
            self._cip_feature_matrix,
        ) = self._load_cip_reference()
        if self.cfg.verbose and self._cip_catalog is not None:
            print(
                f"CIP reference loaded with {len(self._cip_catalog):,} entries."
            )
        self._ensure_alias_patterns()
        self._university_hint_keywords = self._build_university_hint_keywords() if self.cfg.enable_university_hints else {}

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build program clusters and return row-level assignments plus program summary."""

        prepared = self._prepare(df)

        if prepared.empty:
            raise ValueError("No records available after preprocessing.")

        # Global clustering on feature text
        labels = self._cluster_fields(prepared["feature_text"])
        prepared = prepared.copy()
        prepared["program_cluster"] = labels

        cluster_meta: Dict[int, Dict[str, object]] = {}
        for label, cluster in prepared.groupby("program_cluster"):
            canonical_field = self._choose_canonical(cluster["clean_field"], fallback=cluster["field_display"])
            canonical_degree = self._choose_canonical(cluster["clean_degree"], allow_empty=True)
            canonical_degree_group = self._classify_degree_level(canonical_degree, cluster)
            canonical_university = self._choose_canonical(cluster["clean_university"], fallback=cluster["university_source"])
            canonical_city = self._choose_canonical(cluster["clean_city"], allow_empty=True)
            canonical_country = self._choose_canonical(cluster["city_country_code"], allow_empty=True)
            cluster_size = len(cluster)
            max_uni_size = cluster.groupby("clean_university").size()
            max_city_size = cluster.groupby("clean_city").size()
            cluster_meta[int(label)] = {
                "canonical_field": canonical_field,
                "canonical_degree": canonical_degree,
                "canonical_degree_group": canonical_degree_group,
                "canonical_university": canonical_university,
                "canonical_city": canonical_city,
                "city_country_code": canonical_country,
                "cluster_size": cluster_size,
                "is_high_school": bool(cluster["is_high_school"].any()),
                "university_cluster_size": int(max_uni_size.max()) if not max_uni_size.empty else 0,
                "city_cluster_size": int(max_city_size.max()) if not max_city_size.empty else 0,
            }

        prepared["canonical_field"] = prepared["program_cluster"].map(lambda x: cluster_meta[int(x)]["canonical_field"])
        prepared["canonical_degree"] = prepared["program_cluster"].map(lambda x: cluster_meta[int(x)]["canonical_degree"])
        prepared["canonical_degree_group"] = prepared["program_cluster"].map(lambda x: cluster_meta[int(x)]["canonical_degree_group"])
        prepared["canonical_university"] = prepared["program_cluster"].map(lambda x: cluster_meta[int(x)]["canonical_university"])
        prepared["canonical_city"] = prepared["program_cluster"].map(lambda x: cluster_meta[int(x)]["canonical_city"])
        prepared["city_country_code"] = prepared["program_cluster"].map(lambda x: cluster_meta[int(x)]["city_country_code"])
        prepared["cluster_size"] = prepared["program_cluster"].map(lambda x: cluster_meta[int(x)]["cluster_size"])
        prepared["is_high_school"] = prepared["program_cluster"].map(lambda x: cluster_meta[int(x)]["is_high_school"])
        prepared["university_cluster_size"] = prepared["program_cluster"].map(lambda x: cluster_meta[int(x)]["university_cluster_size"])
        prepared["city_cluster_size"] = prepared["program_cluster"].map(lambda x: cluster_meta[int(x)]["city_cluster_size"])

        assignment_df = prepared.reset_index(drop=True)
        assignment_df["program_id"], _ = pd.factorize(assignment_df["program_cluster"], sort=True)
        assignment_df["program_id"] = assignment_df["program_id"] + 1

        program_info = (
            assignment_df.groupby("program_id")
            .apply(
                lambda g: (
                    g["canonical_field"].iloc[0],
                    g["canonical_degree"].iloc[0],
                    g["canonical_degree_group"].iloc[0],
                )
            )
            .to_dict()
        )

        program_name_map: Dict[int, str] = {}
        for pid, info in program_info.items():
            field, degree, degree_group = info
            degree_group_text = degree_group.strip() if isinstance(degree_group, str) else ""
            degree_text = degree.strip() if isinstance(degree, str) else ""
            if degree_group_text:
                if degree_text:
                    program_name_map[pid] = f"{field} ({degree_group_text}: {degree_text})"
                else:
                    program_name_map[pid] = f"{field} ({degree_group_text})"
            elif degree_text:
                program_name_map[pid] = f"{field} ({degree_text})"
            else:
                program_name_map[pid] = field

        assignment_df["program_name"] = assignment_df["program_id"].map(program_name_map)

        summary = (
            assignment_df.groupby("program_id", as_index=False)
            .agg(
                program_name=("program_name", "first"),
                canonical_university=("canonical_university", "first"),
                university_cluster_size=("university_cluster_size", "first"),
                canonical_city=("canonical_city", "first"),
                city_cluster_size=("city_cluster_size", "first"),
                city_country_code=("city_country_code", "first"),
                canonical_field=("canonical_field", "first"),
                canonical_degree=("canonical_degree", "first"),
                canonical_degree_group=("canonical_degree_group", "first"),
                cluster_size=("program_cluster", "count"),
                is_high_school=("is_high_school", "max"),
                example_universities=(
                    "clean_university",
                    lambda s: "; ".join([
                        x for x in pd.unique(s) if isinstance(x, str) and x
                    ][:3])
                ),
                example_fields=("field_display", lambda s: "; ".join(s.head(3))),
            )
            .sort_values("cluster_size", ascending=False)
        )

        summary["is_high_school"] = summary["is_high_school"].astype(bool)

        assignment_df, summary = self._assign_cip_codes(assignment_df, summary)

        return assignment_df, summary

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"university_name", "field"}
        available = set(df.columns)
        if required_cols.isdisjoint(available):
            raise ValueError(
                "Input data must contain at least the columns 'university_name' or 'field'."
            )

        university_cols = [c for c in ("university_raw", "university_name") if c in df.columns]
        field_cols = [c for c in ("field_raw", "field") if c in df.columns]
        degree_cols = [c for c in ("degree_raw", "degree") if c in df.columns]

        if not university_cols:
            raise ValueError("Cannot find university name columns (expected 'university_raw' or 'university_name').")
        if not field_cols:
            raise ValueError("Cannot find major/field columns (expected 'field_raw' or 'field').")

        data = df.copy()

        data["university_source"] = data[university_cols].bfill(axis=1).iloc[:, 0]
        data["field_source"] = data[field_cols].bfill(axis=1).iloc[:, 0]
        data["degree_source"] = (
            data[degree_cols].bfill(axis=1).iloc[:, 0] if degree_cols else ""
        )

        city_cols = [
            col
            for col in (
                "university_city",
                "city",
                "location_city",
                "university_location",
                "location",
            )
            if col in df.columns
        ]
        if city_cols:
            data["city_source"] = data[city_cols].bfill(axis=1).iloc[:, 0]
        else:
            data["city_source"] = ""

        # Cache-optimized normalization for universities and fields
        uniq_unis = pd.unique(data["university_source"].astype(str))
        uni_norm_map = {u: self._normalize_university(u) for u in uniq_unis}
        data["clean_university"] = data["university_source"].map(uni_norm_map)
        data = self._apply_alias_linking(data)
        data["match_university"] = data["clean_university"].map(self._strip_school_type_tokens)

        uniq_cities = pd.unique(data["city_source"].astype(str))
        city_norm_map = {c: self._normalize_city(c) for c in uniq_cities}
        data["clean_city"] = data["city_source"].map(city_norm_map)
        if self.cfg.infer_city_from_university:
            missing_city_mask = data["clean_city"].str.len() == 0
            if missing_city_mask.any():
                inferred_clean = data.loc[
                    missing_city_mask, "university_source"
                ].map(self._infer_city_from_university_name)
                inferred_clean = inferred_clean.fillna("")
                has_match = inferred_clean.str.len() > 0
                if has_match.any():
                    matched_index = inferred_clean.index[has_match]
                    data.loc[matched_index, "clean_city"] = inferred_clean.loc[has_match]
                    data.loc[matched_index, "city_source"] = inferred_clean.loc[has_match].map(
                        self._format_city_display
                    )
            peer_city_source = (
                data.loc[data["clean_city"].str.len() > 0]
                .groupby("match_university")["clean_city"]
                .agg(lambda s: self._choose_canonical(s, fallback=s, allow_empty=True))
            )
            missing_city_mask = data["clean_city"].str.len() == 0
            if missing_city_mask.any():
                inferred_from_peers = (
                    data.loc[missing_city_mask, "match_university"].map(peer_city_source).fillna("")
                )
                has_peer_match = inferred_from_peers.str.len() > 0
                if has_peer_match.any():
                    peer_indices = inferred_from_peers.index[has_peer_match]
                    data.loc[peer_indices, "clean_city"] = inferred_from_peers.loc[has_peer_match]
                    data.loc[peer_indices, "city_source"] = inferred_from_peers.loc[has_peer_match].map(
                        self._format_city_display
                    )

        canonical_city_info = data["clean_city"].map(self._canonicalize_city)
        data["canonical_city"] = canonical_city_info.map(
            lambda x: x[0] if isinstance(x, tuple) else (x if x else "")
        )
        data["city_country_code"] = canonical_city_info.map(
            lambda x: x[1] if isinstance(x, tuple) else ""
        )
        data["match_city"] = data["canonical_city"]
        data.loc[data["match_city"] == "", "match_city"] = data.loc[
            data["match_city"] == "", "clean_city"
        ]
        mask_city_country = (
            data["match_city"].str.len() > 0
            ) & (data["city_country_code"].str.len() > 0)
        data.loc[mask_city_country, "match_city"] = (
            data.loc[mask_city_country, "match_city"]
            + "__"
            + data.loc[mask_city_country, "city_country_code"]
        )

        data["is_high_school"] = data["clean_university"].map(self._is_high_school)
        data["clean_field"] = data["field_source"].map(self._normalize_field)
        data["clean_degree"] = data["degree_source"].map(self._normalize_degree)

        if self._manual_cluster_map:
            data["manual_cluster_key"] = data["clean_university"].map(self._manual_cluster_map)
        else:
            data["manual_cluster_key"] = np.nan

        if self._manual_type_map:
            manual_type = data["clean_university"].map(self._manual_type_map)
            data.loc[manual_type == "high_school", "is_high_school"] = True
            data.loc[manual_type == "university", "is_high_school"] = False

        data["is_high_school"] = data["is_high_school"].fillna(False).astype(bool)

        data["field_display"] = data["field_source"].fillna("").astype(str)
        # vectorized feature composition
        field_degree = (
            data["clean_field"].fillna("").astype(str).str.strip()
            .str.cat(
                data["clean_degree"].fillna("").astype(str).str.strip(),
                sep=" ",
            )
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        if self.cfg.enable_university_hints and self._university_hint_keywords:
            university_hints = data["match_university"].map(self._derive_university_hint)
        else:
            university_hints = pd.Series(["" for _ in range(len(data))], index=data.index)
        data["feature_text"] = (
            field_degree.str.cat(university_hints, sep=" ")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        data = data.loc[data["clean_university"].notna() & (data["feature_text"].str.len() > 0)]

        data["canonical_university"] = data["clean_university"].fillna("")
        data["canonical_city"] = data["clean_city"].fillna("")
        data["city_country_code"] = data.get("city_country_code", "").fillna("")
        data["university_cluster_size"] = data.groupby("clean_university")["clean_university"].transform("size")
        data["city_cluster_size"] = data.groupby("clean_city")["clean_city"].transform("size")

        if self.cfg.verbose:
            print(
                f"Prepared {len(data):,} records across {data['clean_university'].nunique():,} institutions (no university clustering)."
            )

        return data
    def _cluster_university_level(self, data: pd.DataFrame) -> pd.DataFrame:
        base = (
            data.groupby(["clean_university", "canonical_city"], as_index=False)
            .agg(
                university_display=("university_source", lambda s: self._choose_canonical(s, fallback=s)),
                city_display=("city_source", lambda s: self._choose_canonical(s, fallback=s, allow_empty=True)),
                clean_city=("clean_city", lambda s: self._choose_canonical(s, fallback=s, allow_empty=True)),
                total_records=("clean_university", "size"),
                match_university=("match_university", "first"),
                match_city=("match_city", "first"),
                city_country_code=("city_country_code", "first"),
                is_high_school=("is_high_school", "max"),
                manual_cluster_key=("manual_cluster_key", "first"),
            )
        )

        if base.empty:
            raise ValueError("No university records available after preprocessing.")

        base["is_high_school"] = base["is_high_school"].astype(bool)

        grouped_segments: List[pd.DataFrame] = []
        next_cluster_id = 0
        manual_cluster_ids: Dict[Tuple[str, int], int] = {}
        for (is_high_school, manual_key), subset in base.groupby(
            ["is_high_school", "manual_cluster_key"], dropna=False
        ):
            subset = subset.copy()

            city_features = subset["match_city"].fillna("")
            city_fallback = subset["clean_city"]
            city_features = city_features.where(city_features.str.len() > 0, city_fallback)

            if len(subset) > 1 and city_features.nunique() > 1:
                city_labels = self._cluster_strings(
                    city_features,
                    distance_threshold=self.cfg.city_distance_threshold,
                    ngram_range=self.cfg.city_ngram_range,
                    analyzer=self.cfg.city_analyzer,
                    min_df=self.cfg.city_min_df,
                )
            else:
                city_labels = np.zeros(len(subset), dtype=int)

            subset["city_cluster_local"] = city_labels

            for city_label, city_subset in subset.groupby("city_cluster_local", dropna=False):
                city_subset = city_subset.copy()
                features = city_subset["match_university"].fillna("")
                fallback = city_subset["clean_university"]
                features = features.where(features.str.len() > 0, fallback)

                if pd.notna(manual_key):
                    manual_key_str = str(manual_key)
                    manual_pair = (manual_key_str, int(city_label))
                    cluster_id = manual_cluster_ids.get(manual_pair)
                    if cluster_id is None:
                        cluster_id = next_cluster_id
                        manual_cluster_ids[manual_pair] = cluster_id
                        next_cluster_id += 1
                    city_subset["university_cluster_id"] = cluster_id
                else:
                    if len(city_subset) == 1 or features.nunique() <= 1:
                        local_labels = np.zeros(len(city_subset), dtype=int)
                    else:
                        local_labels = self._cluster_strings(
                            features,
                            distance_threshold=self.cfg.university_distance_threshold,
                            ngram_range=self.cfg.university_ngram_range,
                            analyzer=self.cfg.university_analyzer,
                            min_df=self.cfg.university_min_df,
                        )
                    city_subset["university_cluster_id"] = local_labels + next_cluster_id
                    next_cluster_id = city_subset["university_cluster_id"].max() + 1

                grouped_segments.append(city_subset)

        base = pd.concat(grouped_segments, ignore_index=True)

        if self.cfg.auto_merge_small_variants:
            base = self._merge_low_frequency_variants(base)

        cluster_meta: Dict[int, Dict[str, object]] = {}
        for cluster_id, cluster in base.groupby("university_cluster_id"):
            canonical = self._choose_canonical(
                cluster["university_display"],
                fallback=cluster["clean_university"],
            )
            cluster_size = int(cluster["total_records"].sum())
            canonical_city = self._choose_canonical(
                cluster["city_display"],
                fallback=cluster["canonical_city"],
                allow_empty=True,
            )
            city_cluster_size = int(cluster["total_records"].sum())
            cluster_meta[int(cluster_id)] = {
                "canonical": canonical,
                "size": cluster_size,
                "is_high_school": bool(cluster["is_high_school"].iloc[0]),
                "canonical_city": canonical_city,
                "city_records": city_cluster_size,
                "city_country_code": cluster["city_country_code"].iloc[0] if "city_country_code" in cluster.columns else "",
            }

        base["canonical_university"] = base["university_cluster_id"].map(
            lambda cid: cluster_meta[int(cid)]["canonical"]
        )
        base["university_cluster_size"] = base["university_cluster_id"].map(
            lambda cid: cluster_meta[int(cid)]["size"]
        )
        base["cluster_is_high_school"] = base["university_cluster_id"].map(
            lambda cid: cluster_meta[int(cid)]["is_high_school"]
        )
        base["canonical_city"] = base["university_cluster_id"].map(
            lambda cid: cluster_meta[int(cid)]["canonical_city"]
        )
        base["city_cluster_size"] = base["university_cluster_id"].map(
            lambda cid: cluster_meta[int(cid)]["city_records"]
        )
        base["city_country_code"] = base["university_cluster_id"].map(
            lambda cid: cluster_meta[int(cid)]["city_country_code"]
        )

        return base[
            [
                "clean_university",
                "clean_city",
                "canonical_city",
                "university_cluster_id",
                "canonical_university",
                "university_cluster_size",
                "city_cluster_size",
                "city_country_code",
                "cluster_is_high_school",
            ]
        ]

    def _cluster_university(self, university: str, group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy().reset_index(drop=True)
        if self.cfg.verbose:
            print(f"Clustering {len(group):,} records for {university}...")

        if len(group) < self.cfg.min_group_size_for_clustering:
            group["program_cluster"] = np.arange(len(group))
        else:
            labels = self._cluster_fields(group["feature_text"])
            group["program_cluster"] = labels

        cluster_meta: Dict[int, Dict[str, object]] = {}
        for label, cluster in group.groupby("program_cluster"):
            canonical_field = self._choose_canonical(cluster["clean_field"], fallback=cluster["field_display"])
            canonical_degree = self._choose_canonical(cluster["clean_degree"], allow_empty=True)
            canonical_degree_group = self._classify_degree_level(canonical_degree, cluster)
            cluster_meta[label] = {
                "canonical_field": canonical_field,
                "canonical_degree": canonical_degree,
                "canonical_degree_group": canonical_degree_group,
                "cluster_size": len(cluster),
            }

        group["canonical_field"] = group["program_cluster"].map(
            lambda x: cluster_meta[x]["canonical_field"]
        )
        group["canonical_degree"] = group["program_cluster"].map(
            lambda x: cluster_meta[x]["canonical_degree"]
        )
        group["canonical_degree_group"] = group["program_cluster"].map(
            lambda x: cluster_meta[x]["canonical_degree_group"]
        )
        group["cluster_size"] = group["program_cluster"].map(
            lambda x: cluster_meta[x]["cluster_size"]
        )

        return group

    def _cluster_fields(self, features: pd.Series) -> np.ndarray:
        return self._cluster_strings(
            features,
            distance_threshold=self.cfg.distance_threshold,
            ngram_range=self.cfg.ngram_range,
            analyzer=self.cfg.analyzer,
            min_df=self.cfg.min_df,
        )

    def _cluster_strings(
        self,
        texts: Iterable[str],
        distance_threshold: float,
        ngram_range: Tuple[int, int],
        analyzer: str,
        min_df: int,
    ) -> np.ndarray:
        series = pd.Series(list(texts), dtype=str)
        n = len(series)
        if series.nunique() <= 1:
            return np.zeros(n, dtype=int)

        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(series)

        # Identify zero rows without densifying
        nnz = matrix.getnnz(axis=1)
        zero_mask = (nnz == 0)
        if zero_mask.all():
            return np.arange(n, dtype=int)

        labels = np.full(n, -1, dtype=int)
        if zero_mask.any():
            zero_indices = np.where(zero_mask)[0]
            labels[zero_indices] = np.arange(len(zero_indices))

        nonzero_indices = np.where(~zero_mask)[0]
        X = matrix[nonzero_indices]

        # Heuristic: for small groups, fall back to Agglomerative with dense matrix;
        # for larger groups, build a sparse neighbor graph and union-find components.
        SMALL_N = 300  # tuneable cutoff
        if X.shape[0] <= SMALL_N:
            dense_nonzero = X.toarray()
            clustering = self._make_agg_clusterer(distance_threshold)
            cluster_labels = clustering.fit_predict(dense_nonzero)
        else:
            cluster_labels = self._cluster_by_neighbor_graph(X, distance_threshold)

        if zero_mask.any():
            offset = labels.max() + 1
            cluster_labels = cluster_labels + max(offset, 0)

        labels[nonzero_indices] = cluster_labels
        return labels

    def _cluster_by_neighbor_graph(self, X, distance_threshold: float) -> np.ndarray:
        # Cosine distance threshold -> similarity threshold
        sim_threshold = max(0.0, min(1.0, 1.0 - float(distance_threshold)))
        m = X.shape[0]
        if m == 0:
            return np.array([], dtype=int)

        # Limit neighbors to reduce work; use brute-force cosine but only keep k nearest
        k = int(min(max(10, m // 50), 50))  # between 10 and 50
        nn = NearestNeighbors(n_neighbors=min(k, m), metric="cosine", algorithm="brute")
        nn.fit(X)
        distances, indices = nn.kneighbors(X, return_distance=True)

        parent = list(range(m))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # distances are cosine distances; convert to similarity and apply threshold
        for i in range(m):
            for j_pos, dist in enumerate(distances[i]):
                j = indices[i, j_pos]
                if j == i:
                    continue
                sim = 1.0 - float(dist)
                if sim >= sim_threshold:
                    if i < j:
                        union(i, j)

        # compress
        roots = [find(i) for i in range(m)]
        # map roots to consecutive labels
        id_map: Dict[int, int] = {}
        next_id = 0
        labels = np.empty(m, dtype=int)
        for i, r in enumerate(roots):
            if r not in id_map:
                id_map[r] = next_id
                next_id += 1
            labels[i] = id_map[r]
        return labels
    def _make_agg_clusterer(self, distance_threshold: float) -> AgglomerativeClustering:
        kwargs = dict(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage="average",
        )
        try:
            return AgglomerativeClustering(metric="cosine", **kwargs)
        except TypeError:
            return AgglomerativeClustering(affinity="cosine", **kwargs)  # type: ignore[arg-type]

    def _compose_feature(self, clean_field: str, clean_degree: str) -> str:
        tokens = [token for token in [clean_field, clean_degree] if token]
        feature = " ".join(tokens).strip()
        return feature

    def _build_university_hint_keywords(self) -> Dict[str, str]:
        mapping = {
            "law": "law",
            "legal": "law",
            "business": "business",
            "management": "business",
            "mba": "business",
            "commerce": "business",
            "finance": "business",
            "engineering": "engineering",
            "technolog": "engineering",
            "tech": "engineering",
            "science": "science",
            "medicine": "medicine",
            "medical": "medicine",
            "health": "medicine",
            "nursing": "medicine",
            "pharmacy": "medicine",
            "public policy": "policy",
            "policy": "policy",
            "education": "education",
            "teacher": "education",
            "art": "art",
            "design": "art",
            "music": "art",
            "agricultur": "agriculture",
            "agro": "agriculture",
            "hospitality": "hospitality",
            "tourism": "hospitality",
        }
        return mapping

    def _derive_university_hint(self, text: object) -> str:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        value = str(text).lower()
        if not value:
            return ""
        hints: List[str] = []
        for keyword, hint in self._university_hint_keywords.items():
            if keyword in value:
                hints.append(hint)
        if not hints:
            return ""
        unique_hints = sorted(set(hints))
        return " ".join(unique_hints)

    def _classify_degree_level(self, canonical_degree: str, cluster: pd.DataFrame) -> str:
        # canonical_degree already normalized; cluster contains raw degree entries for reference
        text = canonical_degree.lower().strip() if canonical_degree else ""
        if not text:
            cluster_examples = cluster["clean_degree"].dropna().astype(str).tolist()
            if cluster_examples:
                text = cluster_examples[0].lower().strip()

        if not text:
            # fall back: check degree_source for hints
            raw = cluster["degree_source"].dropna().astype(str).tolist()
            text = raw[0].lower().strip() if raw else ""

        if not text:
            return "other"

        if any(keyword in text for keyword in ("phd", "ph.d", "doctor", "dphil", "dr", "doctoral")):
            return "phd"
        if any(keyword in text for keyword in ("master", "msc", "m.s", "m.a", "mba", "mpp", "mfa", "ms", "ma")):
            return "masters"
        if any(keyword in text for keyword in ("bachelor", "bsc", "ba", "bs", "b.a", "b.s", "undergrad", "btech", "beng")):
            return "bachelors"
        if any(keyword in text for keyword in ("high school", "secondary", "lycee", "lyceum", "prep", "preparatoria", "gymnasium")):
            return "high school"
        if any(keyword in text for keyword in ("associate", "aas", "community college", "a.a", "a.s")):
            return "other"
        return "other"

    def _choose_canonical(
        self,
        values: Iterable[str],
        fallback: Optional[Iterable[str]] = None,
        allow_empty: bool = False,
    ) -> str:
        series = pd.Series(list(values), dtype=str)
        series = series.loc[series.str.len() > 0]
        if series.empty and fallback is not None:
            series = pd.Series(list(fallback), dtype=str)
            series = series.loc[series.str.len() > 0]
        if series.empty:
            return "" if allow_empty else "unspecified"
        counts = series.value_counts()
        return counts.index[0]

    def _load_university_feedback(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        cluster_map: Dict[str, str] = {}
        type_map: Dict[str, str] = {}

        feedback_path = self.cfg.university_feedback_path
        if feedback_path is None:
            default_path = Path(root) / "data/config/university_feedback.csv"
            if default_path.exists():
                feedback_path = default_path

        if feedback_path and feedback_path.exists():
            try:
                feedback_df = pd.read_csv(feedback_path)
            except Exception as exc:
                print(f"Warning: failed to read university feedback at {feedback_path}: {exc}")
                return cluster_map, type_map

            if "clean_university" not in feedback_df.columns:
                print(
                    f"Warning: feedback file {feedback_path} missing 'clean_university' column; skipping manual overrides."
                )
                return cluster_map, type_map

            feedback_df = feedback_df.copy()
            feedback_df["clean_university"] = (
                feedback_df["clean_university"].astype(str).str.strip().str.lower()
            )

            if "manual_cluster_key" in feedback_df.columns:
                valid = feedback_df["manual_cluster_key"].notna()
                cluster_map = (
                    feedback_df.loc[valid, ["clean_university", "manual_cluster_key"]]
                    .assign(manual_cluster_key=lambda df: df["manual_cluster_key"].astype(str).str.strip())
                    .set_index("clean_university")["manual_cluster_key"]
                    .to_dict()
                )

            if "institution_type" in feedback_df.columns:
                valid_types = feedback_df["institution_type"].astype(str).str.lower().isin({"high_school", "university"})
                type_map = (
                    feedback_df.loc[valid_types, ["clean_university", "institution_type"]]
                    .assign(institution_type=lambda df: df["institution_type"].astype(str).str.lower().str.strip())
                    .set_index("clean_university")["institution_type"]
                    .to_dict()
                )

        return cluster_map, type_map

    def _load_university_translations(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        translation_map: Dict[str, str] = {}
        normalized_map: Dict[str, str] = {}

        path = self.cfg.university_translation_path
        default_candidates = [
            Path(root) / "data/crosswalks/institution_translations.csv",
            Path(root) / "data/crosswalks/institutions_translations.csv",
            Path(root) / "data/crosswalks/university_translations.csv",
            Path(root) / "data/crosswalks/institutions_translation.csv",
            Path(root) / "data/crosswalks/institutions.csv",
        ]

        candidate_paths: List[Path] = []
        if path is not None:
            candidate_paths.append(path)
        candidate_paths.extend([p for p in default_candidates if p not in candidate_paths])

        resolved_path: Optional[Path] = None
        for candidate in candidate_paths:
            if candidate.exists():
                resolved_path = candidate
                break

        if resolved_path is None:
            return translation_map, normalized_map

        try:
            df = pd.read_csv(resolved_path)
        except Exception as exc:  # pragma: no cover - best effort load
            print(f"Warning: failed to read university translation reference at {resolved_path}: {exc}")
            return translation_map, normalized_map

        if df.empty:
            return translation_map, normalized_map

        source_keywords = (
            "native",
            "original",
            "local",
            "source",
            "foreign",
            "non_english",
            "nonenglish",
            "language",
            "lang",
            "alias",
            "alternate",
            "altname",
            "alt_name",
            "clean_university",
            "name",
        )
        target_keywords = (
            "english",
            "translation",
            "translated",
            "romanized",
            "romanised",
            "canonical",
            "standard",
            "preferred",
            "display",
            "target",
            "en_name",
            "english_name",
            "name_en",
        )

        columns_lower = {col.lower(): col for col in df.columns}
        source_cols = [columns_lower[col] for col in columns_lower if any(key in col for key in source_keywords)]
        target_cols = [columns_lower[col] for col in columns_lower if any(key in col for key in target_keywords)]

        if not source_cols:
            source_cols = [df.columns[0]]

        if not target_cols:
            target_cols = [col for col in df.columns if col not in source_cols]
        if not target_cols:
            return translation_map, normalized_map

        def _split_values(value: object) -> List[str]:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return []
            text = str(value).strip()
            if not text:
                return []
            parts = re.split(r"[|;/,]", text)
            cleaned_parts = [part.strip() for part in parts if part and part.strip()]
            return cleaned_parts if cleaned_parts else [text]

        for _, row in df.iterrows():
            sources: List[str] = []
            for col in source_cols:
                if col not in row:
                    continue
                sources.extend(_split_values(row[col]))
            sources = [s for s in sources if s]
            if not sources:
                continue

            target_value = ""
            for col in target_cols:
                if col not in row:
                    continue
                values = _split_values(row[col])
                if values:
                    target_value = values[0]
                    break
            if not target_value:
                continue

            normalized_target = self._normalize_text(target_value)
            display_target = target_value

            for source in sources:
                translation_map[source] = display_target

                source_normalized = self._normalize_text(source)
                if source_normalized:
                    normalized_map[source_normalized] = display_target

                transliterated = self._transliterate_text(source)
                if transliterated and transliterated != source:
                    translation_map[transliterated] = display_target
                    translit_norm = self._normalize_text(transliterated)
                    if translit_norm:
                        normalized_map[translit_norm] = display_target

            if normalized_target:
                normalized_map[normalized_target] = display_target

        if not normalized_map:
            for key, value in translation_map.items():
                normalized_key = self._normalize_text(key)
                if normalized_key:
                    normalized_map[normalized_key] = value

        return translation_map, normalized_map

    def _load_external_university_clusters(self) -> Optional[pd.DataFrame]:
        path = self.cfg.external_university_clusters_path
        if path is None:
            return None
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            print(
                f"Warning: external university cluster file {path} not found; falling back to internal clustering."
            )
            return None

        try:
            if path.suffix.lower() in {".parquet", ".pq"}:
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        except Exception as exc:
            print(f"Warning: failed to read external university clusters at {path}: {exc}")
            return None

        if df.empty:
            print(f"Warning: external university cluster file {path} is empty; ignoring.")
            return None

        df = df.copy()
        columns_lower = {col.lower(): col for col in df.columns}

        def resolve_column(*names: str) -> Optional[str]:
            for name in names:
                actual = columns_lower.get(name.lower())
                if actual is not None:
                    return actual
            return None

        clean_col = resolve_column("clean_university")
        if clean_col is None:
            print(
                f"Warning: external university cluster file {path} missing 'clean_university'; ignoring."
            )
            return None
        cluster_col = resolve_column(
            "university_cluster_id",
            "cluster_id",
            "cluster",
            "cluster_key",
            "cluster_root"
        )
        if cluster_col is None:
            print(
                f"Warning: external university cluster file {path} missing 'university_cluster_id'; ignoring."
            )
            return None

        rename_map = {
            clean_col: "clean_university",
            cluster_col: "university_cluster_id",
        }
        optional_specs = {
            "clean_city": ("clean_city",),
            "canonical_university": (
                "canonical_university",
                "canonical",
                "preferred_university",
                "display_university",
            ),
            "canonical_city": ("canonical_city", "display_city"),
            "city_country_code": ("city_country_code", "country_code"),
            "university_cluster_size": ("university_cluster_size", "cluster_size"),
            "city_cluster_size": ("city_cluster_size",),
            "is_high_school": ("is_high_school", "high_school_flag"),
        }
        for target, candidates in optional_specs.items():
            actual = resolve_column(*candidates)
            if actual is not None:
                rename_map[actual] = target

        df = df.rename(columns=rename_map)

        df["clean_university"] = df["clean_university"].astype(str).str.strip().str.lower()
        df = df.loc[df["clean_university"].str.len() > 0]
        if df.empty:
            print(
                f"Warning: no usable entries found in external university cluster file {path}; ignoring."
            )
            return None

        if "clean_city" in df.columns:
            df["clean_city"] = df["clean_city"].map(self._normalize_city)

        df["university_cluster_id"] = pd.to_numeric(
            df["university_cluster_id"], errors="ignore"
        )
        if df["university_cluster_id"].dtype == object:
            df["university_cluster_id"] = (
                df["university_cluster_id"].astype(str).str.strip()
            )

        if "canonical_university" in df.columns:
            df["canonical_university"] = df["canonical_university"].astype(str).str.strip()
        if "canonical_city" in df.columns:
            df["canonical_city"] = df["canonical_city"].astype(str).str.strip()
        if "city_country_code" in df.columns:
            df["city_country_code"] = df["city_country_code"].astype(str).str.strip().str.upper()
        if "is_high_school" in df.columns:
            df["is_high_school"] = (
                df["is_high_school"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})
            )

        key_cols = ["clean_university"]
        if "clean_city" in df.columns:
            key_cols.append("clean_city")
        df = df.drop_duplicates(subset=key_cols, keep="last")
        return df

    def _merge_external_university_clusters(self, data: pd.DataFrame) -> pd.DataFrame:
        external = self._external_university_clusters
        if external is None or external.empty:
            return data

        key_cols: List[str] = ["clean_university"]
        external_has_clean_city = "clean_city" in external.columns and external["clean_city"].notna().any()
        if external_has_clean_city:
            key_cols.append("clean_city")

        ext = external.copy()
        reserved_cols = set(key_cols)
        reserved_cols.add("university_cluster_id")
        rename_map = {
            col: f"external_{col}"
            for col in ext.columns
            if col not in reserved_cols
        }
        if rename_map:
            ext = ext.rename(columns=rename_map)

        for col in list(data.columns):
            if col.startswith("external_"):
                data = data.drop(columns=[col])

        merged = data.merge(ext, how="left", on=key_cols)

        if merged["university_cluster_id"].isna().any():
            missing_mask = merged["university_cluster_id"].isna()
            fallback_labels, _ = pd.factorize(merged.loc[missing_mask, "clean_university"])
            fallback_values = [
                f"external_unmatched_{label}"
                for label in fallback_labels
            ]
            merged.loc[missing_mask, "university_cluster_id"] = fallback_values
            if self.cfg.verbose:
                unmatched = merged.loc[missing_mask, "clean_university"].nunique()
                print(
                    f"Warning: {unmatched} universities missing from external clusters; assigned fallback cluster ids."
                )

        merged["university_cluster_id"] = merged["university_cluster_id"].astype(object)

        if not external_has_clean_city:
            clusters = merged["university_cluster_id"].dropna().unique()
            inferred = []
            for cluster_id in clusters:
                cluster_mask = merged["university_cluster_id"] == cluster_id
                universities = merged.loc[cluster_mask, "clean_university"]
                city_candidates = merged.loc[cluster_mask, "clean_city"]
                canonical_city = self._choose_canonical(city_candidates, allow_empty=True)
                if not canonical_city:
                    canonical_city = self._choose_canonical(universities.map(self._infer_city_from_university_name), allow_empty=True)
                canonical_city = canonical_city or ""
                inferred.append((cluster_id, canonical_city))
            inferred_map = {cid: city for cid, city in inferred}

            merged["canonical_city"] = merged["university_cluster_id"].map(inferred_map).fillna("")
            empty_mask = merged["canonical_city"].astype(str).str.len() == 0
            if empty_mask.any():
                replacement = merged.loc[empty_mask, "clean_city"].fillna("")
                merged.loc[empty_mask, "canonical_city"] = replacement

        if "external_canonical_university" in merged.columns:
            merged["canonical_university"] = merged["external_canonical_university"].where(
                merged["external_canonical_university"].astype(str).str.len() > 0,
                merged.get("canonical_university", merged["clean_university"]),
            )
        elif "canonical_university" not in merged.columns:
            merged["canonical_university"] = merged["clean_university"]
        else:
            merged["canonical_university"] = merged["canonical_university"].fillna(merged["clean_university"])

        if "external_canonical_city" in merged.columns:
            source = merged.get("canonical_city", "")
            merged["canonical_city"] = merged["external_canonical_city"].where(
                merged["external_canonical_city"].astype(str).str.len() > 0,
                source,
            )
            merged = merged.drop(columns=["external_canonical_city"])
        elif "canonical_city" not in merged.columns or merged["canonical_city"].isna().all():
            merged["canonical_city"] = merged.get("clean_city", "")
        missing_canonical = merged["canonical_city"].astype(str).str.len() == 0
        if missing_canonical.any():
            inferred = merged.loc[missing_canonical, "clean_city"]
            merged.loc[missing_canonical, "canonical_city"] = inferred.fillna("")
        missing_canonical = merged["canonical_city"].astype(str).str.len() == 0
        if missing_canonical.any():
            inferred_city = merged.loc[missing_canonical, "clean_university"].map(
                lambda name: self._infer_city_from_university_name(name)
                if isinstance(name, str)
                else ""
            )
            inferred_city = inferred_city.fillna("")
            has_inferred = inferred_city.str.len() > 0
            if has_inferred.any():
                target_index = inferred_city.index[has_inferred]
                merged.loc[target_index, "canonical_city"] = inferred_city.loc[has_inferred]
        if "canonical_city" in merged.columns:
            merged["canonical_city"] = merged["canonical_city"].fillna("").astype(str).str.strip()

        if "external_city_country_code" in merged.columns:
            source = merged.get("city_country_code", "")
            merged["city_country_code"] = merged["external_city_country_code"].where(
                merged["external_city_country_code"].astype(str).str.len() > 0,
                source,
            )
            merged = merged.drop(columns=["external_city_country_code"])
        elif "city_country_code" not in merged.columns:
            merged["city_country_code"] = ""

        if "external_is_high_school" in merged.columns:
            mask = merged["external_is_high_school"].notna()
            merged.loc[mask, "is_high_school"] = merged.loc[mask, "external_is_high_school"].astype(bool)
            merged = merged.drop(columns=["external_is_high_school"])

        if "external_university_cluster_size" in merged.columns:
            merged["university_cluster_size"] = merged["external_university_cluster_size"].where(
                merged["external_university_cluster_size"].notna(),
                merged.groupby("university_cluster_id")["clean_university"].transform("size"),
            )
            merged = merged.drop(columns=["external_university_cluster_size"])
        else:
            if "university_cluster_size" in merged.columns:
                merged["university_cluster_size"] = merged["university_cluster_size"].fillna(
                    merged.groupby("university_cluster_id")["clean_university"].transform("size")
                )
            else:
                merged["university_cluster_size"] = merged.groupby("university_cluster_id")["clean_university"].transform("size")

        if "external_city_cluster_size" in merged.columns:
            merged["city_cluster_size"] = merged["external_city_cluster_size"].where(
                merged["external_city_cluster_size"].notna(),
                merged.groupby("university_cluster_id")["clean_university"].transform("size"),
            )
            merged = merged.drop(columns=["external_city_cluster_size"])
        else:
            if "city_cluster_size" in merged.columns:
                merged["city_cluster_size"] = merged["city_cluster_size"].fillna(
                    merged.groupby("university_cluster_id")["clean_university"].transform("size")
                )
            else:
                merged["city_cluster_size"] = merged.groupby("university_cluster_id")["clean_university"].transform("size")

        if "city_cluster_size" not in merged.columns:
            merged["city_cluster_size"] = merged.groupby("university_cluster_id")["clean_university"].transform("size")

        if "canonical_city" not in merged.columns:
            merged["canonical_city"] = merged.get("clean_city", "").fillna("")
        else:
            merged["canonical_city"] = merged["canonical_city"].fillna("").astype(str)

        if "city_country_code" not in merged.columns:
            merged["city_country_code"] = ""
        else:
            merged["city_country_code"] = merged["city_country_code"].fillna("").astype(str)

        for col in list(merged.columns):
            if col.startswith("external_"):
                merged = merged.drop(columns=[col])

        return merged

    def _load_cip_reference(
        self,
    ) -> Tuple[Optional[pd.DataFrame], Optional[TfidfVectorizer], Optional[np.ndarray]]:
        if not self.cfg.enable_cip_matching:
            return None, None, None

        path = self.cfg.cip_reference_path
        if path is None:
            print("Warning: CIP matching enabled but no reference path provided; disabling CIP matching.")
            self.cfg.enable_cip_matching = False
            return None, None, None
        if isinstance(path, str):
            path = Path(path)
            self.cfg.cip_reference_path = path
        if not path.exists():
            print(f"Warning: CIP reference file {path} not found; disabling CIP matching.")
            self.cfg.enable_cip_matching = False
            return None, None, None

        try:
            if path.suffix.lower() in {".parquet", ".pq"}:
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        except Exception as exc:
            print(f"Warning: failed to read CIP reference at {path}: {exc}")
            self.cfg.enable_cip_matching = False
            return None, None, None

        if df.empty:
            print(f"Warning: CIP reference file {path} is empty; disabling CIP matching.")
            self.cfg.enable_cip_matching = False
            return None, None, None

        df = df.copy()
        columns_lower = {col.lower(): col for col in df.columns}

        def resolve_column(explicit: Optional[str], *candidates: str) -> Optional[str]:
            if explicit:
                key = explicit.strip().lower()
                return columns_lower.get(key)
            for candidate in candidates:
                key = candidate.strip().lower()
                match = columns_lower.get(key)
                if match:
                    return match
            return None

        code_col = resolve_column(
            self.cfg.cip_code_column,
            "cipcode",
            "cip_code",
            "cip",
            "code",
        )
        title_col = resolve_column(
            self.cfg.cip_title_column,
            "title",
            "ciptitle",
            "name",
            "description",
        )
        desc_col = resolve_column(
            self.cfg.cip_description_column,
            "description",
            "cipdesc",
            "definition",
            "text",
        )
        level_col = resolve_column(
            self.cfg.cip_level_column,
            "credential",
            "award_level",
            "level",
            "degree",
        )

        if code_col is None:
            print(
                f"Warning: unable to identify CIP code column in {path}; disabling CIP matching."
            )
            self.cfg.enable_cip_matching = False
            return None, None, None

        rename_map = {code_col: "cip_code"}
        if title_col:
            rename_map[title_col] = "cip_title"
        if desc_col:
            rename_map[desc_col] = "cip_description"
        if level_col:
            rename_map[level_col] = "cip_level"
        df = df.rename(columns=rename_map)

        df["cip_code"] = df["cip_code"].astype(str).str.strip()
        df = df.loc[df["cip_code"].str.len() > 0]
        if df.empty:
            print(f"Warning: no valid CIP codes found in {path}; disabling CIP matching.")
            self.cfg.enable_cip_matching = False
            return None, None, None

        df["cip_code"] = df["cip_code"].str.replace(r"[^0-9]", "", regex=True)
        df = df.loc[df["cip_code"].str.len() == 6]
        if df.empty:
            print(f"Warning: CIP reference {path} contains no 6-digit codes; disabling CIP matching.")
            self.cfg.enable_cip_matching = False
            return None, None, None

        if "cip_title" not in df.columns:
            df["cip_title"] = ""
        if "cip_description" not in df.columns:
            df["cip_description"] = ""
        if "cip_level" not in df.columns:
            df["cip_level"] = ""

        df["cip_title"] = df["cip_title"].fillna("").astype(str).str.strip()
        df["cip_description"] = df["cip_description"].fillna("").astype(str).str.strip()
        df["cip_level"] = df["cip_level"].fillna("").astype(str).str.strip()

        df["cip_feature_text"] = (
            df["cip_title"].str.strip()
            + " "
            + df["cip_description"].str.strip()
        ).str.strip()
        df.loc[df["cip_feature_text"].str.len() == 0, "cip_feature_text"] = df["cip_title"]

        df = df.drop_duplicates(subset=["cip_code"], keep="first").reset_index(drop=True)
        df = df.loc[df["cip_code"].str.len() == 6]
        df = df.loc[df["cip_code"].str.fullmatch(r"[0-9]{6}")]
        if df.empty:
            print(f"Warning: CIP reference {path} has no unique codes; disabling CIP matching.")
            self.cfg.enable_cip_matching = False
            return None, None, None

        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
        )
        feature_matrix = vectorizer.fit_transform(df["cip_feature_text"])

        if feature_matrix.shape[0] == 0:
            print(f"Warning: failed to build features from CIP reference {path}; disabling CIP matching.")
            self.cfg.enable_cip_matching = False
            return None, None, None

        return df, vectorizer, feature_matrix

    def _compose_cip_program_feature(self, row: pd.Series) -> str:
        parts: List[str] = []
        for key in ("canonical_field", "canonical_degree", "example_fields", "program_name"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        if "canonical_university" in row and isinstance(row["canonical_university"], str):
            uni_text = row["canonical_university"].strip()
            if uni_text:
                parts.append(uni_text)
        if not parts:
            return ""
        combined = " ".join(parts)
        combined = re.sub(r"\s+", " ", combined).strip()
        return combined

    def _assign_cip_codes(
        self,
        assignments: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.cfg.enable_cip_matching or self._cip_catalog is None:
            return assignments, summary
        if summary.empty:
            return assignments, summary
        if self._cip_vectorizer is None or self._cip_feature_matrix is None:
            return assignments, summary

        program_texts = summary.apply(self._compose_cip_program_feature, axis=1)
        program_texts = program_texts.fillna("")
        empty_mask = program_texts.str.strip().str.len() == 0
        if empty_mask.any():
            fallback = summary.loc[empty_mask, "canonical_field"].fillna("").astype(str)
            program_texts.loc[empty_mask] = fallback
        text_list = program_texts.tolist()

        program_vectors = self._cip_vectorizer.transform(text_list)
        total_programs = program_vectors.shape[0]

        max_matches = max(1, int(self.cfg.cip_max_matches or 1))
        threshold = float(self.cfg.cip_similarity_threshold)
        threshold = max(0.0, min(1.0, threshold))
        batch_size = 512

        catalog = self._cip_catalog.reset_index(drop=True)

        best_codes: List[str] = ["" for _ in range(total_programs)]
        best_titles: List[str] = ["" for _ in range(total_programs)]
        best_levels: List[str] = ["" for _ in range(total_programs)]
        best_scores: List[float] = [0.0 for _ in range(total_programs)]
        serialized_matches: List[str] = ["[]" for _ in range(total_programs)]

        # compute in batches to avoid allocating a full dense P×C matrix
        for start in range(0, total_programs, batch_size):
            end = min(start + batch_size, total_programs)
            batch = program_vectors[start:end]
            if batch.shape[0] == 0:
                continue
            scores_block = cosine_similarity(batch, self._cip_feature_matrix)
            for local_idx, scores in enumerate(scores_block):
                idx = start + local_idx
                scores_array = np.asarray(scores, dtype=float)
                order = np.argsort(scores_array)[::-1]
                matches: List[Dict[str, Any]] = []

                for pos in order:
                    score = float(scores_array[pos])
                    if not math.isfinite(score) or score <= 0:
                        continue
                    if score < threshold:
                        break
                    entry = catalog.iloc[pos]
                    matches.append(
                        {
                            "cip_code": entry.get("cip_code", ""),
                            "cip_title": entry.get("cip_title", ""),
                            "cip_level": entry.get("cip_level", ""),
                            "similarity": round(score, 6),
                        }
                    )
                    if len(matches) >= max_matches:
                        break

            if not matches and self.cfg.cip_use_llm and callable(self.cfg.cip_llm_resolver):
                context = {
                    "program_feature": text_list[idx],
                    "canonical_field": summary.iloc[idx].get("canonical_field", ""),
                    "canonical_degree": summary.iloc[idx].get("canonical_degree", ""),
                    "canonical_university": summary.iloc[idx].get("canonical_university", ""),
                }
                try:
                    llm_match = self.cfg.cip_llm_resolver(context, catalog)
                except Exception as exc:
                    llm_match = None
                    if self.cfg.verbose:
                        print(f"Warning: CIP LLM resolver failed for program {summary.iloc[idx].get('program_id', idx)}: {exc}")
                if llm_match:
                    normalized_llm = {
                        "cip_code": llm_match.get("cip_code", ""),
                        "cip_title": llm_match.get("cip_title", ""),
                        "cip_level": llm_match.get("cip_level", ""),
                        "similarity": float(llm_match.get("similarity", 0.0)),
                    }
                    matches.append(normalized_llm)

                if matches:
                    best = matches[0]
                    best_codes[idx] = best.get("cip_code", "")
                    best_titles[idx] = best.get("cip_title", "")
                    best_levels[idx] = best.get("cip_level", "")
                    best_scores[idx] = float(best.get("similarity", 0.0))
                else:
                    best_codes[idx] = ""
                    best_titles[idx] = ""
                    best_levels[idx] = ""
                    best_scores[idx] = 0.0

                try:
                    serialized_matches[idx] = json.dumps(matches, ensure_ascii=False)
                except (TypeError, ValueError):
                    serialized_matches[idx] = "[]"

        summary = summary.copy()
        summary["cip_code"] = best_codes
        summary["cip_title"] = best_titles
        summary["cip_level"] = best_levels
        summary["cip_similarity"] = best_scores
        summary["cip_matches"] = serialized_matches

        if not assignments.empty:
            lookup = summary.set_index("program_id")
            for column in ("cip_code", "cip_title", "cip_level", "cip_similarity", "cip_matches"):
                assignments[column] = assignments["program_id"].map(lookup[column])

        return assignments, summary
    def _default_normalization_catalog(self) -> Dict[str, Dict[str, str]]:
        return {
            "ko": {
                r"dae\s*hag?g?yo$": " university",
                r"dae[hg]?a?k+g?gyo$": " university",
                r"dae\s*hag?won$": " graduate school",
                r"dae[hg]?a?k+won$": " graduate school",
                r"godeung?hakg?yo$": " high school",
                r"jung?hakg?yo$": " middle school",
                r"chodeung?hakg?yo$": " elementary school",
                r"hakg?yo$": " school",
            },
            "ja": {
                r"daigaku$": " university",
                r"daigakk?o$": " university",
                r"koukou$": " high school",
                r"sh[ōo]gakk?o$": " elementary school",
                r"ch[ūu]gakk?o$": " junior high school",
                r"gakk?o$": " school",
            },
            "zh": {
                r"daxue$": " university",
                r"xueyuan$": " college",
                r"zhongxue$": " middle school",
                r"gaozhong$": " high school",
                r"xiaoxue$": " elementary school",
            },
            "ru": {
                r"universitet$": " university",
                r"akademiya$": " academy",
                r"institut$": " institute",
                r"kolledzh$": " college",
                r"litsey$": " lyceum",
                r"gimnaziya$": " gymnasium",
                r"shkola$": " school",
            },
            "hi": {
                r"vishwavidyalaya$": " university",
                r"mahavidyalaya$": " college",
                r"vidyalaya$": " school",
                r"vidyapeeth$": " university",
            },
            "ar": {
                r"al\s*jami?a$": " university",
                r"jami?a$": " university",
                r"kulliyyah$": " college",
                r"madrasah$": " school",
                r"madrasat$": " school",
                r"ma[h']?ad$": " institute",
            },
            "es": {
                r"universidad$": " university",
                r"colegio$": " school",
                r"escuela$": " school",
                r"instituto$": " institute",
                r"liceo$": " high school",
            },
            "fr": {
                r"universit[ée]$": " university",
                r"[ée]cole$": " school",
                r"lyc[ée]e$": " high school",
                r"coll[ée]ge$": " college",
                r"institut$": " institute",
            },
            "de": {
                r"universit[äa]t$": " university",
                r"hochschule$": " university",
                r"fachhochschule$": " university",
                r"schule$": " school",
                r"gymnasium$": " high school",
                r"akademie$": " academy",
            },
        }

    def _compile_replacement_patterns(
        self, catalog: Optional[Dict[str, Dict[str, str]]]
    ) -> Dict[str, List[Tuple[Pattern[str], str]]]:
        compiled: Dict[str, List[Tuple[Pattern[str], str]]] = {}
        if not catalog:
            return compiled
        for language, replacements in catalog.items():
            if not replacements:
                continue
            patterns: List[Tuple[Pattern[str], str]] = []
            for pattern, replacement in replacements.items():
                try:
                    regex = re.compile(pattern, flags=re.IGNORECASE)
                except re.error:
                    continue
                patterns.append((regex, replacement))
            if patterns:
                patterns.sort(key=lambda item: len(item[0].pattern), reverse=True)
                compiled[language] = patterns
        return compiled

    def _load_country_names(self) -> Set[str]:
        names: Set[str] = set()
        potential_paths = [
            Path(root) / "data/crosswalks/country_dict.json",
            Path(root) / "data/crosswalks/countries.json",
            Path(root) / "data/crosswalks/country_codes.json",
        ]
        for path in potential_paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    items = list(data.items())
                elif isinstance(data, list):
                    items = [(entry, entry) for entry in data]
                else:
                    items = []
                for key, value in items:
                    for candidate in (key, value):
                        if not isinstance(candidate, str):
                            continue
                        normalized = self._normalize_text(candidate)
                        if normalized:
                            names.add(normalized)
                if names:
                    break
            except Exception:
                continue

        fallback_names = [
            "united states",
            "united kingdom",
            "south korea",
            "korea",
            "republic of korea",
            "north korea",
            "japan",
            "china",
            "people's republic of china",
            "germany",
            "france",
            "spain",
            "italy",
            "canada",
            "india",
            "russia",
            "brazil",
            "mexico",
            "australia",
            "new zealand",
            "singapore",
            "hong kong",
            "saudi arabia",
            "united arab emirates",
            "uae",
            "argentina",
            "chile",
            "colombia",
            "peru",
            "philippines",
            "indonesia",
            "malaysia",
            "thailand",
            "vietnam",
        ]
        for fallback in fallback_names:
            normalized = self._normalize_text(fallback)
            if normalized:
                names.add(normalized)
        names.discard("")
        return names

    def _make_translator(self) -> Optional[Dict[str, object]]:
        if not self.cfg.enable_translation:
            return None

        service = (self.cfg.translation_service or "").strip().lower()
        candidates: List[str] = []
        if service:
            candidates.append(service)
        else:
            candidates.extend(["googletrans", "deep_translator"])

        for candidate in candidates:
            if candidate == "googletrans" and GoogleTranslator is not None:
                try:
                    client = GoogleTranslator()
                    return {"service": "googletrans", "client": client}
                except Exception:
                    continue
            if candidate in {"deep_translator", "deep-translator"} and DeepGoogleTranslator is not None:
                try:
                    client = DeepGoogleTranslator(source="auto", target=self.cfg.translation_target_language)
                    return {"service": "deep_translator", "client": client}
                except Exception:
                    continue
        return None

    def _load_city_reference(self) -> Tuple[Dict[str, Tuple[str, str]], List[str]]:
        lookup: Dict[str, Tuple[str, str]] = {}

        default_path = Path(root) / "data/crosswalks/geonames/cities500.txt"
        path = self.cfg.geonames_cities_path or default_path

        if not path.exists():
            return lookup, []

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
            "population",
            "elevation",
            "dem",
            "timezone",
            "moddate",
        ]

        try:
            df = pd.read_csv(
                path,
                sep="\t",
                names=column_names,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"Warning: failed to read geonames city reference at {path}: {exc}")
            return lookup, []

        for _, row in df.iterrows():
            country = row.get("countrycode", "").strip().upper()
            canonical = self._normalize_text(row.get("name", ""))
            if not canonical:
                continue
            key = canonical
            lookup.setdefault(key, (canonical, country))

            asciiname = self._normalize_text(row.get("asciiname", ""))
            if asciiname:
                lookup.setdefault(asciiname, (canonical, country))

            alternates = row.get("altnernatenames", "")
            if alternates:
                for alt in alternates.split(","):
                    alt_norm = self._normalize_text(alt)
                    if alt_norm:
                        lookup.setdefault(alt_norm, (canonical, country))

        keys = list(lookup.keys())
        return lookup, keys

    def _is_country_name(self, text: str) -> bool:
        if not text:
            return False
        normalized = self._normalize_text(text)
        return normalized in self._country_names

    def _strip_trailing_country(self, text: str) -> str:
        if not text:
            return ""
        tokens = text.split()
        if not tokens:
            return ""
        tokens = [token for token in tokens if token]
        if not tokens:
            return ""
        changed = True
        while tokens and changed:
            changed = False
            max_window = min(3, len(tokens))
            for window in range(max_window, 0, -1):
                tail_tokens = tokens[-window:]
                tail_text = " ".join(tail_tokens)
                if self._is_country_name(tail_text):
                    tokens = tokens[:-window]
                    changed = True
                    break
                if window == 1 and self._is_country_name(tail_tokens[0]):
                    tokens = tokens[:-1]
                    changed = True
                    break
        return " ".join(tokens).strip()
    def _strip_school_type_tokens(self, text: object) -> str:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        value = str(text).lower()
        if not value:
            return ""
        stripped = value
        for phrase in self._school_type_keywords:
            if not phrase:
                continue
            pattern = r"\b" + re.escape(phrase) + r"\b"
            stripped = re.sub(pattern, " ", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        return stripped

    def _canonicalize_city(self, city: object) -> Tuple[str, str]:
        if city is None or (isinstance(city, float) and np.isnan(city)):
            return "", ""
        city = str(city).strip()
        if not city:
            return "", ""
        if not self._city_lookup:
            return city, ""
        if city in self._city_lookup:
            return self._city_lookup[city]
        if len(city) < 3 or not self._city_lookup_keys:
            return city, ""
        match = process.extractOne(city, self._city_lookup_keys, score_cutoff=self.cfg.city_score_cutoff)
        if match:
            return self._city_lookup[match[0]]
        return city, ""

    def _is_high_school(self, text: object) -> bool:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return False
        value = str(text).lower()
        if not value:
            return False
        for pattern in self.HIGH_SCHOOL_PATTERNS:
            if re.search(pattern, value):
                return True
        return False

    def _extract_primary_city(self, text: str) -> str:
        if not text:
            return ""
        lowered = text.lower()
        lowered = lowered.replace("–", "-")
        parts = re.split(r"[,\-/|;]", lowered)
        for candidate in parts:
            candidate = candidate.strip()
            if not candidate:
                continue
            candidate = re.sub(r"\b(city|municipality|metropolitan area|metro)\b", " ", candidate)
            candidate = re.sub(r"[0-9]", " ", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if len(candidate) >= 2:
                return candidate
        return lowered.strip()

    def _match_city_candidate(self, text: str) -> str:
        if not text:
            return ""
        normalized = self._normalize_text(text)
        if not normalized:
            return ""
        tokens = normalized.split()
        if not tokens:
            return ""
        max_ngram = min(len(tokens), max(1, self.cfg.city_inference_max_ngram))
        for n in range(max_ngram, 0, -1):
            for start in range(len(tokens) - n, -1, -1):
                candidate = " ".join(tokens[start : start + n])
                if self._city_lookup:
                    if candidate in self._city_lookup:
                        return candidate
                else:
                    if (
                        n == 1
                        and len(candidate) >= 3
                        and candidate not in self.CITY_INFERENCE_STOPWORDS
                    ):
                        return candidate
        if not self._city_lookup and tokens:
            for token in reversed(tokens):
                if len(token) >= 3 and token not in self.CITY_INFERENCE_STOPWORDS:
                    return token
            return tokens[0]
        return ""

    def _infer_city_from_university_name(self, text: object) -> str:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        raw = str(text).strip()
        if not raw:
            return ""
        cleaned = re.sub(r"[()\[\]]", ",", raw)
        segments = re.split(r"[|/;]", cleaned)
        for segment in reversed(segments):
            segment = segment.strip()
            if not segment:
                continue
            parts = [part.strip() for part in segment.split(",") if part.strip()]
            for part in reversed(parts):
                match = self._match_city_candidate(part)
                if match:
                    return match
        return self._match_city_candidate(raw)

    def _format_city_display(self, city_key: str) -> str:
        if not city_key:
            return ""
        info = self._city_lookup.get(city_key)
        base = info[0] if info else city_key
        formatted = capwords(base)
        return formatted

    def _translate_with_service(self, text: str) -> str:
        if not text or self._translator_info is None or not self.cfg.enable_translation:
            return ""

        cache_key = (text, self.cfg.translation_target_language)
        cached = self._translation_cache.get(cache_key)
        if cached is not None:
            return cached

        translated = ""
        service = self._translator_info.get("service") if self._translator_info else None
        client = self._translator_info.get("client") if self._translator_info else None
        try:
            if service == "googletrans" and client is not None:
                result = client.translate(text, dest=self.cfg.translation_target_language)
                translated = getattr(result, "text", "") if result else ""
            elif service == "deep_translator" and client is not None:
                translated = client.translate(text)
            elif client is not None:
                translate_callable = getattr(client, "translate", None)
                if callable(translate_callable):
                    try:
                        translated_result = translate_callable(text, dest=self.cfg.translation_target_language)  # type: ignore[arg-type]
                        translated = getattr(translated_result, "text", translated_result)
                    except TypeError:
                        translated_result = translate_callable(text)
                        translated = getattr(translated_result, "text", translated_result)
        except Exception:
            translated = ""

        translated = str(translated).strip()
        if translated:
            self._translation_cache[cache_key] = translated
        return translated

    def _transliterate_text(self, text: str) -> str:
        if not text or not self.cfg.enable_transliteration:
            return text
        if text_unidecode is None:
            return text
        transliterated = text_unidecode(text)
        transliterated = transliterated.strip()
        return transliterated or text

    def _lookup_translation(self, text: str) -> str:
        if not text:
            return ""
        if text in self._university_translation_map:
            return self._university_translation_map[text]
        normalized = self._normalize_text(text)
        if normalized in self._normalized_translation_map:
            return self._normalized_translation_map[normalized]
        return ""

    def _match_translation_map(self, text: str) -> str:
        if not text or not self._normalized_translation_map:
            return ""
        normalized = self._normalize_text(text)
        if not normalized:
            return ""
        direct = self._normalized_translation_map.get(normalized)
        if direct:
            return direct
        if not self._translation_normalized_keys:
            return ""
        match = process.extractOne(
            normalized,
            self._translation_normalized_keys,
            scorer=fuzz.WRatio,
            score_cutoff=self.cfg.translation_similarity_cutoff,
        )
        if match:
            return self._normalized_translation_map.get(match[0], "")
        return ""

    def _detect_text_languages(self, text: str) -> List[str]:
        languages: List[str] = []
        if not text:
            return languages

        has_hangul = False
        has_hiragana = False
        has_katakana = False
        has_cjk = False
        has_cyrillic = False
        has_devanagari = False
        has_arabic = False

        for ch in text:
            code = ord(ch)
            if 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
                has_hangul = True
                continue
            if 0x3040 <= code <= 0x309F:
                has_hiragana = True
                continue
            if 0x30A0 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0xFF66 <= code <= 0xFF9D:
                has_katakana = True
                continue
            if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF:
                has_cjk = True
                continue
            if 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F or 0x2DE0 <= code <= 0x2DFF:
                has_cyrillic = True
                continue
            if 0x0900 <= code <= 0x097F:
                has_devanagari = True
                continue
            if (
                0x0600 <= code <= 0x06FF
                or 0x0750 <= code <= 0x077F
                or 0x08A0 <= code <= 0x08FF
                or 0xFB50 <= code <= 0xFDFF
                or 0xFE70 <= code <= 0xFEFF
            ):
                has_arabic = True

        if has_hangul:
            languages.append("ko")
        if has_hiragana or has_katakana:
            languages.append("ja")
        if has_cjk and "ja" not in languages and "ko" not in languages:
            languages.append("zh")
        if has_cyrillic:
            languages.append("ru")
        if has_devanagari:
            languages.append("hi")
        if has_arabic:
            languages.append("ar")
        return languages

    def _apply_language_replacements(self, text: str, languages: Optional[List[str]] = None) -> str:
        if not text or not self._replacement_patterns:
            return text

        result = text
        queue: List[str] = []
        seen: Set[str] = set()

        if languages:
            for lang in languages:
                if lang in self._replacement_patterns and lang not in seen:
                    queue.append(lang)
                    seen.add(lang)

        for lang, patterns in self._replacement_patterns.items():
            if lang in seen:
                continue
            if any(pattern.search(result) for pattern, _ in patterns):
                queue.append(lang)
                seen.add(lang)

        for lang in queue:
            patterns = self._replacement_patterns.get(lang)
            if not patterns:
                continue
            for pattern, replacement in patterns:
                result, _ = pattern.subn(replacement.strip(), result)

        return result

    def _refine_romanized_text(self, text: str, languages: Optional[List[str]] = None) -> str:
        if not text:
            return ""
        refined = self._apply_language_replacements(text, languages)
        refined = re.sub(r"\s+", " ", refined).strip()
        if not refined:
            return ""
        if refined.lower() != text.lower():
            return capwords(refined)
        return refined

    def _collect_translation_candidates(self, *texts: str) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for text in texts:
            if not text:
                continue
            key = text.strip()
            if not key:
                continue
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        return ordered

    def _build_alias_map(self, data: pd.DataFrame) -> Dict[str, str]:
        alias_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        clean_names = set(data["clean_university"].astype(str))

        for _, row in data.iterrows():
            source = row.get("university_source")
            clean_name = row.get("clean_university")
            if not isinstance(source, str) or not isinstance(clean_name, str):
                continue
            for alias in self._extract_alias_candidates(source):
                alias_norm = self._normalize_alias_token(alias)
                if not alias_norm or alias_norm == clean_name:
                    continue
                alias_counts[alias_norm][clean_name] += 1

        alias_map: Dict[str, str] = {}
        min_support = max(1, int(self.cfg.alias_resolution_min_support))
        min_confidence = max(0.0, min(1.0, float(self.cfg.alias_resolution_confidence)))

        for alias_norm, counts_dict in alias_counts.items():
            total = sum(counts_dict.values())
            if total == 0:
                continue
            canonical, support = max(counts_dict.items(), key=lambda kv: kv[1])
            if support < min_support:
                continue
            confidence = support / total
            if confidence < min_confidence:
                continue

            if alias_norm == canonical:
                continue

            if alias_norm not in clean_names:
                continue
            if canonical not in clean_names:
                continue

            alias_map[alias_norm] = canonical

        # fall back to mapping strong initialisms even if not observed in source text
        for name in clean_names:
            initialism = self._make_initialism(name)
            if not initialism:
                continue
            alias_norm = self._normalize_alias_token(initialism)
            if (
                alias_norm
                and alias_norm != name
                and alias_norm in clean_names
                and alias_norm not in alias_map
            ):
                alias_map[alias_norm] = name

        return alias_map

    def _normalize_alias_token(self, alias: str) -> str:
        if not alias:
            return ""
        alias_clean = alias.strip()
        if not alias_clean:
            return ""
        alias_clean = alias_clean.replace('.', '')
        alias_clean = alias_clean.replace('&', ' and ')
        alias_clean = alias_clean.replace('/', ' ')
        alias_clean = alias_clean.replace('_', ' ')
        alias_clean = alias_clean.replace('-', ' ')
        normalized = self._normalize_text(alias_clean)
        compact = normalized.replace(' ', '')
        original_compact = alias_clean.replace(' ', '')
        if compact and (
            len(compact) <= 6
            or original_compact.isupper()
            or (compact.isalpha() and original_compact.upper() == original_compact)
        ):
            return compact
        return normalized

    def _make_initialism(self, text: str) -> str:
        if not text:
            return ""
        tokens = re.split(r"[\s\-|,\.&/]+", text)
        letters: List[str] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if token.lower() in self.INITIALISM_STOPWORDS:
                continue
            letters.append(token[0])
        initialism = ''.join(letters)
        return initialism.upper()

    def _split_alias_fragments(self, text: str) -> List[str]:
        if not text:
            return []
        fragments = re.split(r"[/\|;,]", text)
        return [fragment.strip() for fragment in fragments if fragment and fragment.strip()]

    def _ensure_alias_patterns(self) -> None:
        if getattr(self, "_alias_paren_pattern", None) is not None:
            return
        self._alias_paren_pattern = re.compile(r"\(([^()]+)\)")
        self._alias_bracket_pattern = re.compile(r"\[([^\[\]]+)\]")
        self._alias_quote_pattern = re.compile(
            r"[\"'\u201C\u201D\u2018\u2019]{1}([^\"'\u201C\u201D\u2018\u2019]{2,})[\"'\u201C\u201D\u2018\u2019]{1}"
        )
        self._alias_marker_pattern = re.compile(
            r"\b(?:aka|a\.k\.a\.|also known as|known as|formerly|trading as|doing business as)\b",
            flags=re.IGNORECASE,
        )

    def _extract_alias_candidates(self, text: str) -> List[str]:
        if not text:
            return []

        candidates: List[str] = []

        self._ensure_alias_patterns()

        def push(value: str) -> None:
            candidate = value.strip()
            if not candidate:
                return
            candidate = re.sub(r"^[\s\"'\u201C\u201D\u2018\u2019]+|[\s\"'\u201C\u201D\u2018\u2019]+$", "", candidate).strip()
            if candidate:
                candidates.append(candidate)

        for match in self._alias_paren_pattern.findall(text):
            for fragment in self._split_alias_fragments(match):
                push(fragment)

        for match in self._alias_bracket_pattern.findall(text):
            for fragment in self._split_alias_fragments(match):
                push(fragment)

        for match in self._alias_quote_pattern.findall(text):
            push(match)

        marker_split = self._alias_marker_pattern.split(text)
        if len(marker_split) > 1:
            for fragment in marker_split[1:]:
                push(fragment)

        seen: Set[str] = set()
        unique_candidates: List[str] = []
        for candidate in candidates:
            lowered = candidate.lower()
            if lowered and lowered not in seen and len(candidate) <= 80:
                seen.add(lowered)
                unique_candidates.append(candidate)

        return unique_candidates
    def _merge_low_frequency_variants(self, base: pd.DataFrame) -> pd.DataFrame:
        if base.empty:
            return base

        max_size = max(0, self.cfg.small_variant_max_size)
        similarity_cutoff = max(0, min(100, self.cfg.small_variant_similarity))
        size_ratio = max(1.0, self.cfg.small_variant_size_ratio)
        min_target_size = max(1, self.cfg.small_variant_min_target_size)

        base = base.copy()
        reassignment: Dict[int, int] = {}

        grouped = base.groupby("canonical_city", dropna=False)
        for _, subset in grouped:
            names = (
                subset["match_university"].fillna(subset["clean_university"]).astype(str).str.strip().tolist()
            )
            counts = subset["total_records"].astype(int).tolist()
            cluster_ids = subset["university_cluster_id"].astype(int).tolist()

            for pos in range(len(names)):
                name = names[pos]
                if not name:
                    continue
                count = counts[pos]
                if count > max_size:
                    continue

                min_required = max(min_target_size, math.ceil(count * size_ratio))

                candidate_strings: List[str] = []
                candidate_clusters: List[int] = []
                for cand_pos in range(len(names)):
                    if cand_pos == pos:
                        continue
                    candidate_name = names[cand_pos]
                    if not candidate_name:
                        continue
                    candidate_count = counts[cand_pos]
                    if candidate_count < min_required:
                        continue
                    candidate_strings.append(candidate_name)
                    candidate_clusters.append(cluster_ids[cand_pos])

                if not candidate_strings:
                    continue

                match = process.extractOne(
                    name,
                    candidate_strings,
                    scorer=fuzz.WRatio,
                    score_cutoff=similarity_cutoff,
                )
                if not match:
                    continue
                _, _score, match_pos = match
                if match_pos is None or match_pos < 0 or match_pos >= len(candidate_clusters):
                    continue

                source_cluster = cluster_ids[pos]
                target_cluster = candidate_clusters[match_pos]
                if source_cluster == target_cluster:
                    continue
                reassignment[source_cluster] = target_cluster

        if not reassignment:
            return base

        def resolve(cluster_id: int) -> int:
            visited: set[int] = set()
            while cluster_id in reassignment and cluster_id not in visited:
                visited.add(cluster_id)
                cluster_id = reassignment[cluster_id]
            return cluster_id

        base["university_cluster_id"] = base["university_cluster_id"].astype(int).map(resolve)
        return base

    def _select_preferred_university_name(
        self,
        name_a: str,
        name_b: str,
        counts: Dict[str, int],
    ) -> str:
        tokens_a = len(name_a.split())
        tokens_b = len(name_b.split())
        if tokens_a != tokens_b:
            return name_a if tokens_a > tokens_b else name_b
        count_a = counts.get(name_a, 0)
        count_b = counts.get(name_b, 0)
        if count_a != count_b:
            return name_a if count_a >= count_b else name_b
        if len(name_a) != len(name_b):
            return name_a if len(name_a) >= len(name_b) else name_b
        return name_a if name_a <= name_b else name_b

    def _apply_alias_linking(self, data: pd.DataFrame) -> pd.DataFrame:
        alias_map = self._build_alias_map(data)
        if not alias_map:
            return data
        data = data.copy()
        counts = data["clean_university"].value_counts().to_dict()
        existing_names = set(counts.keys())
        parent: Dict[str, str] = {}

        def find(name: str) -> str:
            parent.setdefault(name, name)
            if parent[name] != name:
                parent[name] = find(parent[name])
            return parent[name]

        def union(name_a: str, name_b: str) -> None:
            root_a = find(name_a)
            root_b = find(name_b)
            if root_a == root_b:
                return
            preferred = self._select_preferred_university_name(root_a, root_b, counts)
            other = root_b if preferred == root_a else root_a
            parent[other] = preferred

        for alias, canonical in alias_map.items():
            normalized_alias = alias
            normalized_canonical = canonical
            if normalized_alias == normalized_canonical:
                continue
            if normalized_alias not in existing_names:
                continue
            if normalized_canonical not in existing_names:
                continue
            union(normalized_alias, normalized_canonical)

        if not parent:
            return data

        canonical_lookup: Dict[str, str] = {}
        for name in existing_names:
            canonical_lookup[name] = find(name)

        data["clean_university"] = data["clean_university"].map(lambda name: canonical_lookup.get(name, name))
        return data

    def _translate_university(self, text: str) -> str:
        if not text:
            return ""

        languages = self._detect_text_languages(text)
        direct = self._lookup_translation(text)
        if direct:
            return direct

        transliterated = self._transliterate_text(text)
        transliteration_refined = self._refine_romanized_text(transliterated, languages)

        candidates = self._collect_translation_candidates(text, transliterated, transliteration_refined)
        processed: Set[str] = set()
        translations: List[str] = []

        while candidates:
            candidate = candidates.pop(0)
            if not candidate:
                continue
            if candidate in processed:
                continue
            processed.add(candidate)

            lookup = self._lookup_translation(candidate)
            if lookup:
                return lookup

            matched = self._match_translation_map(candidate)
            if matched:
                return matched

            translated_candidate = self._translate_with_service(candidate)
            if translated_candidate:
                translations.append(translated_candidate)
                lookup = self._lookup_translation(translated_candidate)
                if lookup:
                    return lookup
                matched = self._match_translation_map(translated_candidate)
                if matched:
                    return matched
                refined_translated = self._refine_romanized_text(
                    translated_candidate,
                    languages or self._detect_text_languages(translated_candidate),
                )
                if refined_translated and refined_translated not in processed:
                    candidates.append(refined_translated)

        for translated_candidate in translations:
            refined = self._refine_romanized_text(
                translated_candidate,
                languages or self._detect_text_languages(translated_candidate),
            )
            if refined:
                lookup = self._lookup_translation(refined)
                if lookup:
                    return lookup
                matched = self._match_translation_map(refined)
                if matched:
                    return matched

        if translations:
            return translations[-1]
        if transliteration_refined:
            return transliteration_refined
        if transliterated:
            return transliterated
        return text

    def _prepare_university_text(self, text: object) -> str:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        raw = str(text).strip()
        if not raw:
            return ""
        languages = self._detect_text_languages(raw)
        translated = self._translate_university(raw)
        transliterated = self._transliterate_text(translated)
        refined = self._refine_romanized_text(transliterated, languages)
        return refined or transliterated or translated

    def _normalize_city(self, text: object) -> str:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        raw = str(text).strip()
        if not raw:
            return ""
        primary = self._extract_primary_city(raw)
        normalized = self._normalize_text(primary)
        normalized = re.sub(r"\b(state|province|prefecture|county)\b", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = self._strip_trailing_country(normalized)
        return normalized

    def _strip_city_tokens(self, text: object) -> str:
        cleaned = self._normalize_city(text)
        return cleaned

    def _normalize_university(self, text: object) -> str:
        prepared = self._prepare_university_text(text)
        cleaned = self._normalize_text(prepared)
        if not cleaned:
            return ""
        for pattern, repl in self.UNIVERSITY_ALIASES:
            cleaned = re.sub(pattern, repl, cleaned)
        cleaned = re.sub(r"\b(u\.s\.a|usa)\b", "united states", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _normalize_field(self, text: object) -> str:
        cleaned = self._normalize_text(text)
        if not cleaned:
            return ""
        cleaned = re.sub(r"\b&\b", " and ", cleaned)
        cleaned = re.sub(r"[0-9]+", " ", cleaned)
        tokens = [t for t in cleaned.split() if t not in self.FIELD_STOPWORDS]
        tokens = [t for t in tokens if len(t) >= self.cfg.min_token_length]
        cleaned_tokens = " ".join(tokens)
        cleaned_tokens = re.sub(r"\s+", " ", cleaned_tokens).strip()
        return cleaned_tokens

    def _normalize_degree(self, text: object) -> str:
        cleaned = self._normalize_text(text)
        cleaned = cleaned.replace(".", " ")
        cleaned = re.sub(r"\bdegree\b", "", cleaned)
        cleaned = re.sub(r"\b(in|of|and)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _normalize_text(self, value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.lower()
        text = text.replace("/", " ")
        text = re.sub(r"[\(\)\[\]\{\},;:\-_\+\.\!]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def infer_default_input() -> Optional[Path]:
    candidates = [
        Path(root) / "data/int/ihma_educ_all_oct20.parquet",
        Path(root) / "data/int/linkedin_education_history.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_input(path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file {path} does not exist.")
    if path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension for {path}. Use parquet or csv.")

    if limit is not None:
        df = df.head(limit)
    return df


def write_output(assignments: pd.DataFrame, summary: pd.DataFrame, output: Optional[Path], summary_output: Optional[Path]) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        assignments.to_parquet(output, index=False)
    if summary_output:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(summary_output, index=False)


def parse_args() -> argparse.Namespace:
    default_input = infer_default_input()
    parser = argparse.ArgumentParser(
        description="Partition LinkedIn education history into university × major programs using unsupervised clustering.",
    )
    parser.add_argument("--input", type=Path, default=default_input, help="Input parquet/csv file containing education history.")
    parser.add_argument("--output", type=Path, default=None, help="Path to save row-level cluster assignments (parquet).")
    parser.add_argument("--summary-output", type=Path, default=None, help="Path to save program-level summary (parquet).")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for dry runs.")
    parser.add_argument("--distance-threshold", type=float, default=None, help="Override cosine distance threshold for clustering.")
    parser.add_argument("--min-group-size", type=int, default=None, help="Minimum number of records per university for clustering.")
    parser.add_argument("--university-distance-threshold", type=float, default=None, help="Override cosine distance threshold when clustering university names.")
    parser.add_argument("--no-university-clustering", action="store_true", help="Skip cross-university clustering; treat each cleaned university separately.")
    parser.add_argument("--university-feedback-path", type=Path, default=None, help="CSV file with manual clustering overrides for universities.")
    parser.add_argument("--school-type-keyword", action="append", default=None, help="Additional school-type keywords to ignore during university clustering (can be repeated).")
    parser.add_argument("--geonames-cities-path", type=Path, default=None, help="Path to geonames cities file for city canonicalization (defaults to cities500.txt).")
    parser.add_argument("--translation-path", type=Path, default=None, help="Optional CSV with university translation overrides.")
    parser.add_argument("--university-clusters", type=Path, default=None, help="Optional CSV/parquet with external university cluster assignments.")
    parser.add_argument("--disable-translation", action="store_true", help="Disable automatic translation when normalizing universities.")
    parser.add_argument("--disable-transliteration", action="store_true", help="Disable transliteration when normalizing universities.")
    parser.add_argument("--enable-cip-matching", action="store_true", help="Enable matching clustered programs to CIP codes.")
    parser.add_argument("--cip-reference", type=Path, default=None, help="CSV/parquet file containing CIP codes, titles, and descriptions.")
    parser.add_argument("--cip-code-column", type=str, default=None, help="Column name containing CIP codes (optional).")
    parser.add_argument("--cip-title-column", type=str, default=None, help="Column name containing CIP titles (optional).")
    parser.add_argument("--cip-description-column", type=str, default=None, help="Column name containing CIP descriptions/definitions (optional).")
    parser.add_argument("--cip-level-column", type=str, default=None, help="Column name containing CIP award level (optional).")
    parser.add_argument("--cip-max-matches", type=int, default=None, help="Number of top CIP matches to retain per program (default: 1).")
    parser.add_argument("--cip-similarity-threshold", type=float, default=None, help="Minimum cosine similarity (0-1) for accepting CIP matches (default: 0.25).")
    parser.add_argument("--cip-use-llm", action="store_true", help="Route low-confidence programs through a custom CIP LLM resolver (requires integration in code).")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.disable_translation:
        translation_enabled = False
    else:
        translation_enabled = True
    transliteration_enabled = not args.disable_transliteration
    cip_matching_enabled = args.enable_cip_matching or args.cip_reference is not None
    cip_max_matches = (
        args.cip_max_matches
        if args.cip_max_matches is not None
        else ProgramClusterConfig.cip_max_matches
    )
    cip_similarity_threshold = (
        args.cip_similarity_threshold
        if args.cip_similarity_threshold is not None
        else ProgramClusterConfig.cip_similarity_threshold
    )

    config = ProgramClusterConfig(
        distance_threshold=args.distance_threshold or ProgramClusterConfig.distance_threshold,
        min_group_size_for_clustering=args.min_group_size or ProgramClusterConfig.min_group_size_for_clustering,
        university_distance_threshold=args.university_distance_threshold or ProgramClusterConfig.university_distance_threshold,
        cluster_universities=not args.no_university_clustering,
        university_feedback_path=args.university_feedback_path,
        geonames_cities_path=args.geonames_cities_path,
        university_translation_path=args.translation_path,
        external_university_clusters_path=args.university_clusters,
        enable_translation=translation_enabled,
        enable_transliteration=transliteration_enabled,
        verbose=not args.quiet,
        enable_cip_matching=cip_matching_enabled,
        cip_reference_path=args.cip_reference,
        cip_code_column=args.cip_code_column,
        cip_title_column=args.cip_title_column,
        cip_description_column=args.cip_description_column,
        cip_level_column=args.cip_level_column,
        cip_max_matches=cip_max_matches,
        cip_similarity_threshold=cip_similarity_threshold,
        cip_use_llm=args.cip_use_llm,
    )

    clusterer = ProgramClusterer(config)

    input_path = args.input
    if input_path is None:
        raise ValueError("No input path provided and default input not found.")

    df = load_input(input_path, limit=args.limit)
    assignments, summary = clusterer.fit_transform(df)

    write_output(assignments, summary, args.output, args.summary_output)


if __name__ == "__main__":
    main()
