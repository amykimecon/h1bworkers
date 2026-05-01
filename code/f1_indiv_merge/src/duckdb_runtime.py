"""Shared DuckDB runtime defaults for the f1_indiv_merge pipeline."""

from __future__ import annotations

import os

DEFAULT_DUCKDB_MEMORY_LIMIT = "256GB"
DUCKDB_MEMORY_LIMIT_ENV_VAR = "F1_INDIV_DUCKDB_MEMORY_LIMIT"


def get_duckdb_memory_limit(default: str = DEFAULT_DUCKDB_MEMORY_LIMIT) -> str:
    """Return the configured DuckDB memory limit for this pipeline."""
    value = os.environ.get(DUCKDB_MEMORY_LIMIT_ENV_VAR, "").strip()
    return value or default


def get_duckdb_memory_limit_sql_literal(default: str = DEFAULT_DUCKDB_MEMORY_LIMIT) -> str:
    """Return the memory limit escaped for inclusion in a SQL string literal."""
    return get_duckdb_memory_limit(default).replace("'", "''")
