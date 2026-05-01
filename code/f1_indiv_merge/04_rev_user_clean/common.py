from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import weakref
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
for _path in (PIPELINE_ROOT, REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from config import root  # noqa: E402
from helpers import fullname_clean_regex_sql  # noqa: E402
from src.duckdb_runtime import get_duckdb_memory_limit_sql_literal  # noqa: E402

ANGLO_COUNTRIES = (
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
    "New Zealand",
)

_ENTITY_WORD_RE = re.compile(
    r"\b(inc|llc|corp|ltd|co|plc|lp|llp|pllc|pc|pa|na|nv|sa|ag|bv|gmbh)\b",
    flags=re.IGNORECASE,
)
_STOPWORD_RE = re.compile(
    r"\b(the|of|and|a|an|for|in|at|by|with|de|le|la|les|el|los)\b",
    flags=re.IGNORECASE,
)
_FILLER_RE = re.compile(
    r"\b(group|services|solutions|technologies|technology|global|international|"
    r"national|american|north|south|east|west|systems|consulting|management|"
    r"partners|associates|enterprises|holdings)\b",
    flags=re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[,.;:()/&\[\]\-]+")
_WHITESPACE_RE = re.compile(r"\s+")
_LONG_TOKEN_RE_TEMPLATE = r"(^| )[^ ]{{{min_token_chars},}}( |$)"

NAME_ARTIFACT_VERSION = 2


def escape_sql_literal(value: str | Path) -> str:
    return str(value).replace("'", "''")


def resolve_existing_path(*candidates: Any) -> str | None:
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple)):
            resolved = resolve_existing_path(*candidate)
            if resolved:
                return resolved
            continue
        text = str(candidate).strip()
        if not text:
            continue
        if Path(text).exists():
            return text
    return None


def ensure_parent_dir(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def resolve_user_shard_spec(stage_cfg: dict[str, Any]) -> dict[str, int | str] | None:
    raw_count = stage_cfg.get("user_shard_count")
    raw_id = stage_cfg.get("user_shard_id")
    if raw_count in (None, "", 0) and raw_id in (None, ""):
        return None
    if raw_count in (None, "", 0) or raw_id in (None, ""):
        raise ValueError("Stage-04 sharding requires both `user_shard_count` and `user_shard_id`.")
    shard_count = int(raw_count)
    shard_id = int(raw_id)
    if shard_count < 2:
        raise ValueError("Stage-04 sharding requires `user_shard_count >= 2`.")
    if shard_id < 0 or shard_id >= shard_count:
        raise ValueError("Stage-04 sharding requires `0 <= user_shard_id < user_shard_count`.")
    return {
        "user_shard_count": shard_count,
        "user_shard_id": shard_id,
        "user_shard_label": f"shard{shard_id:04d}of{shard_count:04d}",
    }


def sql_user_shard_predicate(
    user_id_sql: str,
    *,
    shard_count: int,
    shard_id: int,
) -> str:
    return f"MOD(ABS(CAST({user_id_sql} AS BIGINT)), {int(shard_count)}) = {int(shard_id)}"


def shard_output_path(path: str | Path, *, shard_count: int, shard_id: int) -> str:
    out_path = Path(path)
    suffix = "".join(out_path.suffixes)
    stem = out_path.name[:-len(suffix)] if suffix else out_path.name
    shard_label = f"shard{int(shard_id):04d}of{int(shard_count):04d}"
    return str(out_path.with_name(f"{stem}__{shard_label}{suffix}"))


def _atomic_tmp_path(path: str | Path) -> Path:
    out_path = ensure_parent_dir(path)
    return out_path.with_name(f".{out_path.name}.tmp")


def clear_dir(path: str | Path) -> Path:
    out_path = Path(path)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    out_path = ensure_parent_dir(path)
    tmp_path = _atomic_tmp_path(out_path)
    if tmp_path.exists():
        tmp_path.unlink()
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(out_path)
    return out_path


def atomic_write_parquet(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    out_path = ensure_parent_dir(path)
    tmp_path = _atomic_tmp_path(out_path)
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_parquet(tmp_path, index=index)
    tmp_path.replace(out_path)
    return out_path


def atomic_duckdb_copy_to_parquet(
    con: duckdb.DuckDBPyConnection,
    query_sql: str,
    out_path: str | Path,
    parameters: list[Any] | None = None,
) -> Path:
    final_path = ensure_parent_dir(out_path)
    tmp_path = _atomic_tmp_path(final_path)
    if tmp_path.exists():
        tmp_path.unlink()
    copy_sql = f"COPY ({query_sql}) TO '{escape_sql_literal(tmp_path)}' (FORMAT PARQUET)"
    if parameters is None:
        con.execute(copy_sql)
    else:
        con.execute(copy_sql, parameters)
    tmp_path.replace(final_path)
    return final_path


def _cleanup_duckdb_temp_dir(path: str | Path) -> None:
    shutil.rmtree(Path(path), ignore_errors=True)


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads = {min(8, max(1, os.cpu_count() or 1))}")
    con.execute(f"SET memory_limit = '{get_duckdb_memory_limit_sql_literal()}'")
    con.execute("SET preserve_insertion_order = false")
    temp_root = Path(root) / ".tmp" / "f1_indiv_merge_duckdb"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"pid{os.getpid()}_", dir=str(temp_root)))
    weakref.finalize(con, _cleanup_duckdb_temp_dir, temp_dir)
    con.execute(f"SET temp_directory = '{escape_sql_literal(temp_dir)}'")
    con.create_function("title", lambda value: value.title() if value else value, ["VARCHAR"], "VARCHAR")
    return con


def sql_string_literal(value: str | Path) -> str:
    return "'" + escape_sql_literal(value) + "'"


def sql_normalize_alpha_expr(value_sql: str) -> str:
    base = f"COALESCE(CAST({value_sql} AS VARCHAR), '')"
    return (
        "TRIM(REGEXP_REPLACE("
        "REGEXP_REPLACE(strip_accents(lower("
        f"{base}"
        ")), '[^a-z]+', ' ', 'g'), "
        "'\\s+', ' ', 'g'))"
    )


def sql_token_list_expr(value_sql: str) -> str:
    normalized = sql_normalize_alpha_expr(value_sql)
    return f"list_distinct(list_filter(str_split({normalized}, ' '), x -> x <> ''))"


def sql_text_relation_rank_expr(source_sql: str, candidate_sql: str) -> str:
    source_norm = sql_normalize_alpha_expr(source_sql)
    candidate_norm = sql_normalize_alpha_expr(candidate_sql)
    source_tokens = sql_token_list_expr(source_sql)
    candidate_tokens = sql_token_list_expr(candidate_sql)
    return f"""
        CASE
            WHEN {source_norm} = '' OR {candidate_norm} = '' THEN 0
            WHEN {source_norm} = {candidate_norm} THEN 2
            WHEN list_has_all({source_tokens}, {candidate_tokens})
              OR list_has_all({candidate_tokens}, {source_tokens}) THEN 1
            ELSE 0
        END
    """


def sql_clean_company_name_expr(value_sql: str) -> str:
    entity_words = escape_sql_literal(_ENTITY_WORD_RE.pattern)
    stopwords = escape_sql_literal(_STOPWORD_RE.pattern)
    filler_words = escape_sql_literal(_FILLER_RE.pattern)
    punct = escape_sql_literal(_PUNCT_RE.pattern)
    text = f"lower(trim(CAST({value_sql} AS VARCHAR)))"
    cleaned = (
        f"REGEXP_REPLACE({text}, '{punct}', ' ', 'g')"
    )
    cleaned = f"REGEXP_REPLACE({cleaned}, '\\\\bincorporated\\\\b', 'inc', 'g')"
    cleaned = f"REGEXP_REPLACE({cleaned}, '\\\\bcorporation\\\\b', 'corp', 'g')"
    cleaned = f"REGEXP_REPLACE({cleaned}, '\\\\blimited\\\\b', 'ltd', 'g')"
    cleaned = f"REGEXP_REPLACE({cleaned}, '\\\\bcompany\\\\b', 'co', 'g')"
    cleaned = f"REGEXP_REPLACE({cleaned}, '{entity_words}', ' ', 'g')"
    cleaned = f"REGEXP_REPLACE({cleaned}, '{stopwords}', ' ', 'g')"
    cleaned = f"REGEXP_REPLACE({cleaned}, '{filler_words}', ' ', 'g')"
    cleaned = f"TRIM(REGEXP_REPLACE({cleaned}, '\\\\s+', ' ', 'g'))"
    return f"""
        CASE
            WHEN {value_sql} IS NULL OR TRIM(CAST({value_sql} AS VARCHAR)) = '' THEN NULL
            WHEN {cleaned} = '' THEN NULL
            ELSE {cleaned}
        END
    """


def register_country_support_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("LOAD json")
    country_path = Path(root) / "data" / "crosswalks" / "country_dict.json"
    subregion_path = Path(root) / "data" / "crosswalks" / "subregion_dict.json"
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE country_raw_map AS
        WITH src AS (
            SELECT content
            FROM read_text({sql_string_literal(country_path)})
        )
        SELECT
            key AS raw_country,
            json_extract_string(content, '$."' || key || '"') AS std_country
        FROM src,
             UNNEST(json_keys(content)) AS keys(key)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE country_label_map AS
        SELECT DISTINCT
            UPPER(TRIM(raw_country)) AS lookup_upper,
            std_country
        FROM country_raw_map
        WHERE raw_country IS NOT NULL
          AND std_country IS NOT NULL
          AND TRIM(raw_country) != ''
        UNION
        SELECT DISTINCT
            UPPER(TRIM(std_country)) AS lookup_upper,
            std_country
        FROM country_raw_map
        WHERE std_country IS NOT NULL
          AND TRIM(std_country) != ''
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE subregion_map AS
        WITH src AS (
            SELECT content
            FROM read_text({sql_string_literal(subregion_path)})
        )
        SELECT
            json_extract_string(content, '$."' || key || '"') AS subregion_candidate,
            key AS country_candidate
        FROM src,
             UNNEST(json_keys(content)) AS keys(key)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE countries_by_subregion AS
        SELECT
            subregion_candidate,
            COUNT(DISTINCT country_candidate) AS n_countries
        FROM subregion_map
        WHERE subregion_candidate IS NOT NULL
          AND country_candidate IS NOT NULL
        GROUP BY 1
        """
    )


@lru_cache(maxsize=1)
def load_country_support() -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    country_path = Path(root) / "data" / "crosswalks" / "country_dict.json"
    subregion_path = Path(root) / "data" / "crosswalks" / "subregion_dict.json"

    country_cw: dict[str, str] = {}
    subregion_by_country: dict[str, str] = {}
    if country_path.exists():
        country_cw = json.loads(country_path.read_text())
    if subregion_path.exists():
        subregion_by_country = json.loads(subregion_path.read_text())

    countries_by_subregion: dict[str, list[str]] = {}
    for country, subregion in subregion_by_country.items():
        countries_by_subregion.setdefault(subregion, []).append(country)
    for subregion in countries_by_subregion:
        countries_by_subregion[subregion] = sorted(set(countries_by_subregion[subregion]))

    return country_cw, subregion_by_country, countries_by_subregion


def standardize_country(value: Any, country_cw: dict[str, str] | None = None) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "none", "null", "<na>", "nan"}:
        return None

    country_map = country_cw if country_cw is not None else load_country_support()[0]
    if text in country_map:
        return country_map[text]

    upper = text.upper()
    for raw_country, std_country in country_map.items():
        if raw_country.upper() == upper or std_country.upper() == upper:
            return std_country

    return text.title() if text.isupper() else text


def safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def normalize_probability_dict(
    prob_map: dict[str, Any] | None,
    *,
    country_cw: dict[str, str] | None = None,
) -> dict[str, float]:
    if not prob_map:
        return {}

    normalized: dict[str, float] = {}
    for raw_label, raw_prob in prob_map.items():
        std_label = standardize_country(raw_label, country_cw=country_cw)
        if std_label is None:
            continue
        try:
            prob = float(raw_prob)
        except (TypeError, ValueError):
            continue
        if prob <= 0:
            continue
        normalized[std_label] = normalized.get(std_label, 0.0) + prob

    total = sum(normalized.values())
    if total <= 0:
        return {}
    return {label: value / total for label, value in normalized.items()}


def split_name_tokens(fullname: str) -> tuple[str, str]:
    if not isinstance(fullname, str):
        return ("", "")
    parts = [part for part in fullname.strip().split(" ") if part]
    if not parts:
        return ("", "")
    if len(parts) == 1:
        return (parts[0], parts[0])
    return (parts[0], parts[-1])


def build_clean_name_frame(
    source_path: str | Path,
    *,
    user_id_col: str = "user_id",
    fullname_col: str = "fullname",
    testing_max_users: int | None = None,
    shard_count: int | None = None,
    shard_id: int | None = None,
    min_fullname_chars: int = 2,
    min_token_chars: int = 2,
) -> pd.DataFrame:
    con = get_duckdb_connection()
    shard_where_sql = ""
    if shard_count is not None and shard_id is not None:
        shard_where_sql = f"""
          AND {sql_user_shard_predicate(
              user_id_col,
              shard_count=int(shard_count),
              shard_id=int(shard_id),
          )}
        """
    df = con.sql(
        f"""
        SELECT
            TRY_CAST({user_id_col} AS BIGINT) AS user_id,
            CAST({fullname_col} AS VARCHAR) AS fullname,
            CASE
                WHEN {fullname_col} ~ '.*[A-z].*'
                    THEN {fullname_clean_regex_sql(fullname_col)}
                ELSE ''
            END AS fullname_clean
        FROM read_parquet('{escape_sql_literal(source_path)}')
        WHERE TRY_CAST({user_id_col} AS BIGINT) IS NOT NULL
          AND {fullname_col} IS NOT NULL
          {shard_where_sql}
        """
    ).df()

    if testing_max_users is not None and testing_max_users > 0:
        keep_users = sorted(df["user_id"].dropna().astype(int).unique().tolist())[: int(testing_max_users)]
        df = df.loc[df["user_id"].isin(keep_users)].copy()

    df["fullname_clean"] = df["fullname_clean"].fillna("").astype(str).str.strip()
    df["fullname_compact"] = df["fullname_clean"].str.replace(" ", "", regex=False)
    long_token_pattern = _LONG_TOKEN_RE_TEMPLATE.format(min_token_chars=max(1, int(min_token_chars)))
    df = df.loc[df["fullname_clean"] != ""].copy()
    df = df.loc[df["fullname_compact"].str.len() >= max(1, int(min_fullname_chars))].copy()
    df = df.loc[df["fullname_clean"].str.contains(long_token_pattern, regex=True)].copy()
    df = df.drop_duplicates(subset=["user_id", "fullname_clean"]).reset_index(drop=True)
    first_last = df["fullname_clean"].map(split_name_tokens)
    df["name_token_count"] = df["fullname_clean"].str.split().map(len)
    df["first_name_clean"] = first_last.map(lambda pair: pair[0])
    df["last_name_clean"] = first_last.map(lambda pair: pair[1])
    df = df.drop(columns=["fullname_compact"])
    return df


def stage_clean_name_artifacts(
    source_path: str | Path,
    *,
    artifact_dir: str | Path,
    user_id_col: str = "user_id",
    fullname_col: str = "fullname",
    overwrite: bool = False,
    testing_max_users: int | None = None,
    shard_count: int | None = None,
    shard_id: int | None = None,
    min_fullname_chars: int = 2,
    min_token_chars: int = 2,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    base_names_path = artifact_root / "base_names.parquet"
    full_unique_path = artifact_root / "full_unique.parquet"
    first_unique_path = artifact_root / "first_unique.parquet"
    last_unique_path = artifact_root / "last_unique.parquet"
    artifact_meta_path = artifact_root / "meta.json"
    full_name_min_chars = max(1, int(min_fullname_chars))
    token_min_chars = max(1, int(min_token_chars))
    long_token_pattern = _LONG_TOKEN_RE_TEMPLATE.format(min_token_chars=token_min_chars)
    resolved_source = str(Path(source_path).resolve()) if Path(source_path).exists() else str(source_path)
    expected_meta = {
        "version": NAME_ARTIFACT_VERSION,
        "source_path": resolved_source,
        "user_id_col": user_id_col,
        "fullname_col": fullname_col,
        "testing_max_users": int(testing_max_users) if testing_max_users else None,
        "shard_count": int(shard_count) if shard_count is not None else None,
        "shard_id": int(shard_id) if shard_id is not None else None,
        "min_fullname_chars": full_name_min_chars,
        "min_token_chars": token_min_chars,
    }

    def _artifact_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
        return {
            "n_full_names": int(
                con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{escape_sql_literal(base_names_path)}')"
                ).fetchone()[0]
            ),
            "n_single_token_names": int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM read_parquet('{escape_sql_literal(base_names_path)}')
                    WHERE name_token_count = 1
                    """
                ).fetchone()[0]
            ),
            "n_first_names": int(
                con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{escape_sql_literal(first_unique_path)}')"
                ).fetchone()[0]
            ),
            "n_last_names": int(
                con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{escape_sql_literal(last_unique_path)}')"
                ).fetchone()[0]
            ),
        }

    if overwrite:
        clear_dir(artifact_root)
    else:
        artifact_root.mkdir(parents=True, exist_ok=True)

    artifacts_exist = all(path.exists() for path in [base_names_path, full_unique_path, first_unique_path, last_unique_path])
    if not overwrite and artifacts_exist:
        existing_meta = None
        if artifact_meta_path.exists():
            try:
                existing_meta = json.loads(artifact_meta_path.read_text())
            except json.JSONDecodeError:
                existing_meta = None
        if existing_meta == expected_meta:
            con = get_duckdb_connection()
            return {
                "artifact_dir": str(artifact_root),
                "base_names_parquet": str(base_names_path),
                "full_unique_parquet": str(full_unique_path),
                "first_unique_parquet": str(first_unique_path),
                "last_unique_parquet": str(last_unique_path),
                "artifacts_rebuilt": False,
                **_artifact_counts(con),
            }
        clear_dir(artifact_root)

    con = get_duckdb_connection()
    testing_scope_sql = ""
    if testing_max_users is not None and testing_max_users > 0:
        testing_scope_sql = f"""
        QUALIFY DENSE_RANK() OVER (
            ORDER BY TRY_CAST({user_id_col} AS BIGINT)
        ) <= {int(testing_max_users)}
        """
    shard_where_sql = ""
    if shard_count is not None and shard_id is not None:
        shard_where_sql = f"""
                  AND {sql_user_shard_predicate(
                      user_id_col,
                      shard_count=int(shard_count),
                      shard_id=int(shard_id),
                  )}
        """

    con.execute(
        f"""
        COPY (
            WITH scoped AS (
                SELECT
                    TRY_CAST({user_id_col} AS BIGINT) AS user_id,
                    CAST({fullname_col} AS VARCHAR) AS fullname
                FROM read_parquet('{escape_sql_literal(source_path)}')
                WHERE TRY_CAST({user_id_col} AS BIGINT) IS NOT NULL
                  AND {fullname_col} IS NOT NULL
                  {shard_where_sql}
                {testing_scope_sql}
            ),
            distinct_names AS (
                SELECT DISTINCT
                    CASE
                        WHEN fullname ~ '.*[A-z].*'
                            THEN {fullname_clean_regex_sql('fullname')}
                        ELSE ''
                    END AS fullname_clean
                FROM scoped
            ),
            filtered AS (
                SELECT
                    NULLIF(TRIM(fullname_clean), '') AS fullname_clean
                FROM distinct_names
                WHERE NULLIF(TRIM(fullname_clean), '') IS NOT NULL
                  AND LENGTH(REPLACE(TRIM(fullname_clean), ' ', '')) >= {full_name_min_chars}
                  AND regexp_matches(TRIM(fullname_clean), '{escape_sql_literal(long_token_pattern)}')
            ),
            tokenized AS (
                SELECT
                    fullname_clean,
                    CASE
                        WHEN STRPOS(fullname_clean, ' ') = 0 THEN 1
                        ELSE LENGTH(fullname_clean) - LENGTH(REPLACE(fullname_clean, ' ', '')) + 1
                    END AS name_token_count
                FROM filtered
            )
            SELECT
                ROW_NUMBER() OVER (ORDER BY fullname_clean) AS full_rownum,
                fullname_clean,
                name_token_count,
                CASE
                    WHEN STRPOS(fullname_clean, ' ') = 0 THEN fullname_clean
                    ELSE SPLIT_PART(fullname_clean, ' ', 1)
                END AS first_name_clean,
                REGEXP_EXTRACT(fullname_clean, '([^ ]+)$', 1) AS last_name_clean
            FROM tokenized
            ORDER BY full_rownum
        ) TO '{escape_sql_literal(base_names_path)}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT
                full_rownum AS rownum,
                fullname_clean
            FROM read_parquet('{escape_sql_literal(base_names_path)}')
            ORDER BY rownum
        ) TO '{escape_sql_literal(full_unique_path)}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT
                ROW_NUMBER() OVER (ORDER BY first_name_clean) AS rownum,
                first_name_clean
            FROM (
                SELECT DISTINCT first_name_clean
                FROM read_parquet('{escape_sql_literal(base_names_path)}')
                WHERE name_token_count > 1
                  AND first_name_clean IS NOT NULL
                  AND TRIM(first_name_clean) != ''
                  AND LENGTH(first_name_clean) >= {token_min_chars}
            )
            ORDER BY rownum
        ) TO '{escape_sql_literal(first_unique_path)}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT
                ROW_NUMBER() OVER (ORDER BY last_name_clean) AS rownum,
                last_name_clean
            FROM (
                SELECT DISTINCT last_name_clean
                FROM read_parquet('{escape_sql_literal(base_names_path)}')
                WHERE name_token_count > 1
                  AND last_name_clean IS NOT NULL
                  AND TRIM(last_name_clean) != ''
                  AND LENGTH(last_name_clean) >= {token_min_chars}
            )
            ORDER BY rownum
        ) TO '{escape_sql_literal(last_unique_path)}' (FORMAT PARQUET)
        """
    )
    atomic_write_json(artifact_meta_path, expected_meta)

    counts = _artifact_counts(con)
    return {
        "artifact_dir": str(artifact_root),
        "base_names_parquet": str(base_names_path),
        "full_unique_parquet": str(full_unique_path),
        "first_unique_parquet": str(first_unique_path),
        "last_unique_parquet": str(last_unique_path),
        "artifacts_rebuilt": True,
        **counts,
    }


def clean_company_name(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\bincorporated\b", "inc", text)
    text = re.sub(r"\bcorporation\b", "corp", text)
    text = re.sub(r"\blimited\b", "ltd", text)
    text = re.sub(r"\bcompany\b", "co", text)
    text = _ENTITY_WORD_RE.sub(" ", text)
    text = _STOPWORD_RE.sub(" ", text)
    text = _FILLER_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def mock_country_probabilities(name: str) -> dict[str, float]:
    lowered = (name or "").strip().lower()
    if not lowered:
        return {}
    if any(token in lowered for token in ("patel", "singh", "gupta", "kumar", "reddy", "sharma")):
        return {"India": 0.9, "United States": 0.1}
    if any(token in lowered for token in ("zhang", "wang", "liu", "chen", "li ")):
        return {"China": 0.9, "United States": 0.1}
    if any(token in lowered for token in ("garcia", "rodriguez", "hernandez", "martinez")):
        return {"Mexico": 0.6, "Spain": 0.3, "United States": 0.1}
    if any(token in lowered for token in ("ivanov", "petrov", "smirnov")):
        return {"Russia": 0.85, "Ukraine": 0.15}
    if any(token in lowered for token in ("ali", "hussain", "khan")):
        return {"Pakistan": 0.55, "India": 0.25, "Bangladesh": 0.2}
    return {"United States": 0.7, "Canada": 0.2, "United Kingdom": 0.1}


def mock_region_probabilities(name: str) -> list[tuple[str, float]]:
    lowered = (name or "").strip().lower()
    if not lowered:
        return []
    if any(token in lowered for token in ("patel", "singh", "gupta", "kumar", "reddy", "sharma")):
        return [("South Asia", 0.9), ("North America", 0.1)]
    if any(token in lowered for token in ("zhang", "wang", "liu", "chen", "li ")):
        return [("Eastern Asia", 0.9), ("North America", 0.1)]
    if any(token in lowered for token in ("garcia", "rodriguez", "hernandez", "martinez")):
        return [("Central America", 0.6), ("Southern Europe", 0.3), ("North America", 0.1)]
    if any(token in lowered for token in ("ivanov", "petrov", "smirnov")):
        return [("Eastern Europe", 0.9), ("Northern Europe", 0.1)]
    if any(token in lowered for token in ("ali", "hussain", "khan")):
        return [("South Asia", 0.7), ("Western Asia", 0.3)]
    return [("North America", 0.8), ("Northern Europe", 0.2)]


def mock_female_probability(name: str) -> float:
    lowered = (name or "").strip().lower()
    if any(token in lowered for token in ("maria", "anna", "sarah", "emily", "fatima", "sofia")):
        return 0.9
    if any(token in lowered for token in ("john", "michael", "david", "ravi", "ahmed", "wei")):
        return 0.1
    return 0.5
