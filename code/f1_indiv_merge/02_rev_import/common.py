"""Shared helpers for stage 02_rev_import."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

DEFAULT_TOKEN_STOPWORDS = {
    "academy",
    "and",
    "at",
    "campus",
    "center",
    "centre",
    "city",
    "college",
    "community",
    "department",
    "for",
    "graduate",
    "high",
    "institute",
    "international",
    "of",
    "program",
    "school",
    "state",
    "studies",
    "system",
    "the",
    "univ",
    "universidad",
    "universite",
    "university",
}

NULLISH_STRINGS = {"", "null", "none", "<na>", "na", "n/a", "nan"}
US_COUNTRY_STRINGS = {"us", "usa", "united states", "united states of america"}


def escape_sql_literal(value: str | Path) -> str:
    return str(value).replace("'", "''")


def resolve_first_existing(candidates: Iterable[str | Path | None]) -> str | None:
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def chunk_values(values: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]


def coerce_int_list(values: Iterable[int | str] | None) -> list[int]:
    if values is None:
        return []
    out: list[int] = []
    for value in values:
        if value is None:
            continue
        out.append(int(value))
    return sorted(set(out))


def clean_institution_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"\s*(\(|\[)[^\)\]]*(\)|\])\s*", " ", normalized)
    normalized = re.sub(r"[’'.]", "", normalized)
    normalized = re.sub(r"\s*(&|\+)\s*", " and ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def build_token_boundary_regex(tokens: Iterable[str]) -> str:
    escaped_tokens = [re.escape(token) for token in tokens if token]
    if not escaped_tokens:
        return r"(?!)"
    escaped_tokens = sorted(set(escaped_tokens), key=lambda token: (-len(token), token))
    return rf"(^|\s)(?:{'|'.join(escaped_tokens)})(\s|$)"


def sql_inst_clean_expr(col: str) -> str:
    str_out = f"""
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    strip_accents(lower({col})),
                '\\s*(\\(|\\[)[^\\)\\]]*(\\)|\\])\\s*', ' ', 'g'),
            $$'|’|\\.$$, '', 'g'),
        '\\s?(&|\\+)\\s?', ' and ', 'g'),
    '[^A-z0-9\\s]', ' ', 'g')
    """
    return f"TRIM(REGEXP_REPLACE({str_out}, '\\s+', ' ', 'g'))"


def sql_degree_clean_expr() -> str:
    return """
        CASE
            WHEN lower(university_raw) ~ '.*(high\\s?school).*'
              OR lower(degree_raw) ~ '.*(high\\s?school).*'
              OR lower(field_raw) ~ '.*(high\\s?school).*'
              OR lower(degree_raw) ~ 'g\\.?e\\.?d\\.?'
            THEN 'High School'
            WHEN lower(degree_raw) ~ '.*(cert|credential|course|semester|exchange|abroad|summer|internship|edx|cdl|coursera|udemy).*'
              OR lower(university_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera|udemy).*'
              OR lower(field_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera|udemy).*'
            THEN 'Non-Degree'
            WHEN (lower(degree_raw) ~ '.*(undergrad).*')
              OR (degree_raw ~ '.*\\b(B\\.?A\\.?|B\\.?S\\.?C\\.?E\\.?|B\\.?Sc\\.?|B\\.?A\\.?E\\.?|B\\.?Eng\\.?|A\\.?B\\.?|S\\.?B\\.?|B\\.?B\\.?M\\.?|B\\.?I\\.?S\\.?)\\b.*')
              OR degree_raw ~ '^B\\.?\\s?S\\.?.*'
              OR lower(field_raw) ~ '.*bachelor.*'
              OR lower(degree_raw) ~ '.*(bachelor|baccalauréat).*'
            THEN 'Bachelor'
            WHEN lower(degree_raw) ~ '.*(master).*'
              OR degree_raw ~ '^M\\.?(Eng|Sc|A)\\.?.*'
            THEN 'Master'
            WHEN degree_raw ~ '.*(M\\.?S\\.?C\\.?E\\.?|M\\.?P\\.?A\\.?|M\\.?Eng|M\\.?Sc|M\\.?A).*'
              OR lower(field_raw) ~ '.*master.*'
              OR lower(degree_raw) ~ '.*master.*'
            THEN 'Master'
            WHEN lower(field_raw) ~ '.*(associate).*'
              OR degree_raw ~ 'A\\.?\\s?A\\.?.*'
              OR lower(degree_raw) ~ '.*associate.*'
              OR lower(field_raw) ~ '.*associate.*'
            THEN 'Associate'
            WHEN lower(degree_raw) ~ '.*(doctor|ph\\.?\\s?d\\.?|d\\.?o\\.?|dvm|jd).*'
              OR degree_raw ~ '.*(Ph\\.?D\\.?|D\\.?O\\.?|J\\.?D\\.?).*'
            THEN 'Doctor'
            WHEN university_raw ~ '.*(HS| High| HIGH| high|H\\.S\\.|S\\.?S\\.?C|H\\.?S\\.?C\\.?|I\\.?B\\.?)$'
            THEN 'High School'
            WHEN degree_raw ~ '^B\\.?\\s?[A-Z].*' THEN 'Bachelor'
            WHEN degree_raw ~ '^M\\.?\\s?[A-Z].*' THEN 'Master'
            ELSE 'Missing'
        END
    """


def sql_extract_year_expr(col: str) -> str:
    return f"""
        COALESCE(
            TRY_CAST(REGEXP_EXTRACT(CAST({col} AS VARCHAR), '([12][0-9]{{3}})', 1) AS INTEGER),
            TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST({col} AS VARCHAR), '%Y-%m-%d')) AS INTEGER),
            TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST({col} AS VARCHAR), '%m/%d/%Y')) AS INTEGER)
        )
    """


def sql_us_or_null_country_expr(col: str) -> str:
    return f"""
        CASE
            WHEN {col} IS NULL THEN TRUE
            WHEN lower(trim(CAST({col} AS VARCHAR))) IN ('', 'null', 'none', '<na>', 'na', 'n/a', 'nan') THEN TRUE
            WHEN lower(trim(CAST({col} AS VARCHAR))) IN ('us', 'usa', 'united states', 'united states of america') THEN TRUE
            ELSE FALSE
        END
    """
