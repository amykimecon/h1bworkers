"""Local FOIA raw import helpers for stage 01_f1_foia_clean."""

from __future__ import annotations

import os
import re
import tempfile
import time
from builtins import print as _print
from functools import partial
from pathlib import Path
from typing import Iterable

import duckdb as ddb
import pandas as pd
import pyarrow.parquet as pq

print = partial(_print, flush=True)

DATE_COLS = {
    "authorization_end_date",
    "authorization_start_date",
    "date_of_birth",
    "first_entry_date",
    "last_departure_date",
    "last_entry_date",
    "opt_authorization_end_date",
    "opt_authorization_start_date",
    "opt_employer_end_date",
    "opt_employer_start_date",
    "program_end_date",
    "program_start_date",
    "visa_expiration_date",
    "visa_issue_date",
}


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds / 60:.2f}m"


def _escape(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _safe_colname(name: str) -> str:
    lowered = name.lower().strip().replace(" ", "_")
    lowered = re.sub(r"[^0-9a-zA-Z_]", "", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered


def _date_parse_sql(col_sql: str) -> str:
    fmt_try = f"""
        COALESCE(
          try_strptime(CAST({col_sql} AS VARCHAR), '%Y-%m-%d %H:%M:%S'),
          try_strptime(CAST({col_sql} AS VARCHAR), '%Y-%m-%d'),
          try_strptime(CAST({col_sql} AS VARCHAR), '%m/%d/%Y %H:%M:%S'),
          try_strptime(CAST({col_sql} AS VARCHAR), '%m/%d/%Y %H:%M'),
          try_strptime(CAST({col_sql} AS VARCHAR), '%m/%d/%Y')
        )
    """

    inner = f"""
        COALESCE(
          CASE
            WHEN typeof({col_sql}) LIKE 'TIMESTAMP%' THEN CAST({col_sql} AS TIMESTAMP)
            ELSE NULL
          END,
          CASE
            WHEN typeof({col_sql}) = 'VARCHAR' THEN {fmt_try}
            ELSE NULL
          END,
          CASE
            WHEN typeof({col_sql}) IN ('DOUBLE','DECIMAL','HUGEINT','BIGINT','INTEGER') THEN
              CAST(DATE '1899-12-30' + CAST({col_sql} AS INTEGER) AS TIMESTAMP)
            ELSE NULL
          END
        )
    """

    guarded = f"""
        CASE
          WHEN ({inner}) IS NOT NULL
               AND EXTRACT(YEAR FROM ({inner})) BETWEEN 1900 AND 2100
          THEN ({inner})
          ELSE NULL
        END
    """
    return f"({guarded})"


def _varchar_sql(col_sql: str) -> str:
    return f"CAST({col_sql} AS VARCHAR)"


def _build_typed_expr(orig_col: str, alias: str) -> str:
    if alias in DATE_COLS:
        return f"{_date_parse_sql(orig_col)} AS {alias}"
    return f"{_varchar_sql(orig_col)} AS {alias}"


def _build_typed_null(alias: str) -> str:
    if alias in DATE_COLS:
        return f"CAST(NULL AS TIMESTAMP) AS {alias}"
    return f"CAST(NULL AS VARCHAR) AS {alias}"


def _read_dir_parquet(
    con: ddb.DuckDBPyConnection,
    path: str | Path,
    *,
    first_parquet: bool,
    base_cols: list[str] | None,
    verbose: bool = False,
) -> list[str]:
    path_str = str(path)
    if verbose:
        print(f"Reading yearly parquet: {path_str}")

    pq_file = pq.ParquetFile(path_str)
    orig_cols = pq_file.schema.names
    safe_cols = [_safe_colname(col) for col in orig_cols]

    if first_parquet:
        select_expr = ", ".join(
            _build_typed_expr(f'"{orig}"', safe)
            for orig, safe in zip(orig_cols, safe_cols)
        )
        con.sql(
            f"""
            CREATE OR REPLACE TABLE allyrs_raw AS
            SELECT {select_expr}
            FROM read_parquet('{_escape(path_str)}')
            """
        )
        return safe_cols

    if base_cols is None:
        raise ValueError("base_cols must be provided when appending subsequent yearly parquets.")

    select_exprs: list[str] = []
    for col in base_cols:
        if col in safe_cols:
            orig = orig_cols[safe_cols.index(col)]
            select_exprs.append(_build_typed_expr(f'"{orig}"', col))
        elif f"{col}_x" in safe_cols:
            orig = orig_cols[safe_cols.index(f"{col}_x")]
            select_exprs.append(_build_typed_expr(f'"{orig}"', col))
        else:
            select_exprs.append(_build_typed_null(col))

    con.sql(
        f"""
        INSERT INTO allyrs_raw
        SELECT {", ".join(select_exprs)}
        FROM read_parquet('{_escape(path_str)}')
        """
    )
    return base_cols


def read_foia_raw_year(
    dirname: str,
    *,
    effective_year: str,
    foia_raw_dir: str | Path,
    foia_byyear_dir: str | Path,
    drop_columns: Iterable[str] | None = None,
    save_output: bool = True,
    verbose: bool = False,
) -> pd.DataFrame | None:
    year_dir = Path(foia_raw_dir) / dirname
    if not year_dir.is_dir():
        return None

    t0 = time.perf_counter()
    frames: list[pd.DataFrame] = []
    normalized_drop_cols = {_safe_colname(col) for col in (drop_columns or [])}

    for filename in sorted(os.listdir(year_dir)):
        if not filename or not filename[0].isalpha():
            continue
        if filename.startswith("~$") or filename.lower().startswith("merged"):
            continue

        lower = filename.lower()
        if not (lower.endswith(".xlsx") or lower.endswith(".xlsm") or lower.endswith(".xls")):
            continue

        file_path = year_dir / filename
        engine = None
        if lower.endswith(".xlsx") or lower.endswith(".xlsm"):
            engine = "openpyxl"
        elif lower.endswith(".xls"):
            engine = "xlrd"

        if verbose:
            print(f"Reading raw FOIA workbook: {dirname}/{filename}")
        frames.append(pd.read_excel(file_path, engine=engine).assign(filename=filename))

    if not frames:
        raise ValueError(f"No eligible Excel workbooks found in {year_dir}")

    df_year = pd.concat(frames, ignore_index=True).assign(year=str(effective_year))
    df_year.columns = [_safe_colname(col) for col in df_year.columns]
    df_year = df_year.replace("b(6),b(7)(c)", pd.NA)

    object_cols = df_year.select_dtypes(include="object").columns.tolist()
    if object_cols:
        df_year = df_year.astype({col: "string" for col in object_cols})

    keep_cols = [col for col in df_year.columns if col not in normalized_drop_cols]
    df_out = df_year.loc[:, keep_cols].copy()

    if save_output:
        byyear_dir = Path(foia_byyear_dir)
        byyear_dir.mkdir(parents=True, exist_ok=True)
        out_path = byyear_dir / f"merged{effective_year}.parquet"
        if out_path.exists():
            out_path.unlink()
        df_out.to_parquet(out_path, index=False)

    if verbose:
        action = "Wrote" if save_output else "Prepared"
        print(f"{action} yearly FOIA parquet for {dirname} -> {effective_year} ({_fmt_elapsed(time.perf_counter() - t0)})")

    return None if save_output else df_out


def build_combined_raw_foia(
    *,
    con: ddb.DuckDBPyConnection | None = None,
    combined_parquet_path: str | Path,
    foia_raw_dir: str | Path,
    foia_byyear_dir: str | Path,
    skip_duplicate_year_dirs: Iterable[str] | None = None,
    early_year_relabels: dict[str, int | str] | None = None,
    drop_columns: Iterable[str] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> Path:
    combined_path = Path(combined_parquet_path)
    raw_dir = Path(foia_raw_dir)
    byyear_dir = Path(foia_byyear_dir)
    skip_dirs = {str(year) for year in (skip_duplicate_year_dirs or [])}
    relabel_map = {str(key): str(value) for key, value in (early_year_relabels or {}).items()}

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw FOIA directory does not exist: {raw_dir}")

    own_con = con is None
    if own_con:
        con = ddb.connect()

    assert con is not None

    if combined_path.exists() and not overwrite:
        con.sql(
            f"""
            CREATE OR REPLACE TABLE allyrs_raw AS
            SELECT *
            FROM read_parquet('{_escape(combined_path)}')
            """
        )
        if verbose:
            print(f"Reusing existing combined raw parquet: {combined_path}")
        if own_con:
            con.close()
        return combined_path

    t0 = time.perf_counter()
    first_parquet = True
    base_cols: list[str] | None = None

    for dirname in sorted(os.listdir(raw_dir), reverse=True):
        dir_path = raw_dir / dirname
        if not dir_path.is_dir() or dirname == "J-1":
            continue
        if dirname in skip_dirs:
            if verbose:
                print(f"Skipping duplicate raw directory: {dirname}")
            continue

        effective_year = relabel_map.get(dirname, dirname)
        yearly_path = byyear_dir / f"merged{effective_year}.parquet"
        if overwrite or not yearly_path.exists():
            read_foia_raw_year(
                dirname,
                effective_year=effective_year,
                foia_raw_dir=raw_dir,
                foia_byyear_dir=byyear_dir,
                drop_columns=drop_columns,
                save_output=True,
                verbose=verbose,
            )

        temp_path: Path | None = None
        path_to_read = yearly_path
        if not yearly_path.exists():
            df_year = read_foia_raw_year(
                dirname,
                effective_year=effective_year,
                foia_raw_dir=raw_dir,
                foia_byyear_dir=byyear_dir,
                drop_columns=drop_columns,
                save_output=False,
                verbose=verbose,
            )
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, dir="/tmp") as tmp:
                temp_path = Path(tmp.name)
            assert df_year is not None
            df_year.to_parquet(temp_path, index=False)
            path_to_read = temp_path

        base_cols = _read_dir_parquet(
            con,
            path_to_read,
            first_parquet=first_parquet,
            base_cols=base_cols,
            verbose=verbose,
        )
        first_parquet = False

        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    if first_parquet:
        raise ValueError(f"No eligible FOIA year directories found under {raw_dir}")

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    if combined_path.exists():
        combined_path.unlink()
    con.sql(f"COPY allyrs_raw TO '{_escape(combined_path)}' (FORMAT PARQUET)")

    if verbose:
        row_count = int(con.sql("SELECT COUNT(*) FROM allyrs_raw").fetchone()[0])
        print(
            f"Wrote combined raw FOIA parquet: {combined_path} "
            f"({row_count:,} rows; {_fmt_elapsed(time.perf_counter() - t0)})"
        )

    if own_con:
        con.close()
    return combined_path
