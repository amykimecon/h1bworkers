from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd


MATCH_SOURCE_TO_STEM = {
    "deterministic": "foia_firm_to_revelio_deterministic",
    "llm_reviewed": "foia_firm_to_revelio_llm_reviewed",
}


def _read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix in {".csv", ".gz", ".bz2", ".zip"}:
        return pd.read_csv(p, low_memory=False)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def _find_artifact(base_dir: str | Path, stem: str) -> Path:
    root = Path(base_dir)
    for ext in (".parquet", ".csv"):
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing artifact '{stem}' under {root}")


def _first_present(columns: Iterable[str], candidates: Sequence[str], label: str) -> str:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"Missing required {label}; tried: {', '.join(candidates)}")


def _coerce_rcid(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _listify(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            stripped = text[1:-1].strip()
            if not stripped:
                return []
            parts = [piece.strip().strip("'\"") for piece in stripped.split(",")]
            return [piece for piece in parts if piece]
        if ";" in text:
            return [piece.strip() for piece in text.split(";") if piece.strip()]
        return [text]
    return [value]


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _register_table(con, table_name: str, df: pd.DataFrame) -> None:
    temp_name = f"_{table_name}_src"
    con.register(temp_name, df)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {temp_name}")


def stage_external_f1_crosswalk(
    *,
    external_output_dir: str,
    match_source: str,
    out_path: str | None = None,
    con=None,
    table_name: str = "f1_employer_final_crosswalk",
    verbose: bool = False,
) -> pd.DataFrame:
    normalized_source = str(match_source).strip().lower()
    if normalized_source not in MATCH_SOURCE_TO_STEM:
        raise ValueError(
            "match_source must be one of: deterministic, llm_reviewed"
        )

    row_entities_path = _find_artifact(external_output_dir, "foia_row_entities")
    row_to_firm_path: Path
    try:
        row_to_firm_path = _find_artifact(external_output_dir, "foia_row_to_firm")
    except FileNotFoundError:
        row_to_firm_path = _find_artifact(external_output_dir, "foia_entity_to_firm")
    firm_match_path = _find_artifact(external_output_dir, MATCH_SOURCE_TO_STEM[normalized_source])

    row_entities = _read_any(row_entities_path).copy()
    row_to_firm = _read_any(row_to_firm_path).copy()
    firm_matches = _read_any(firm_match_path).copy()

    row_uid_col = _first_present(row_entities.columns, ["foia_row_uid"], "FOIA row uid column")
    employer_name_col = _first_present(
        row_entities.columns,
        ["raw_name_example", "employer_name", "foia_name", "canonical_name_clean"],
        "row-level employer name column",
    )
    name_clean_col = _first_present(
        row_entities.columns,
        ["row_name_clean", "canonical_name_clean", "foia_name_clean", "name_clean"],
        "row-level clean name column",
    )
    city_clean_col = _first_present(
        row_entities.columns,
        ["row_city_clean", "foia_city_clean", "city_clean"],
        "row-level clean city column",
    )
    state_clean_col = _first_present(
        row_entities.columns,
        ["row_state_clean", "foia_state_clean", "state_clean"],
        "row-level clean state column",
    )
    zip_clean_col = _first_present(
        row_entities.columns,
        ["row_zip_clean", "foia_zip_clean", "zip_clean"],
        "row-level clean zip column",
    )

    row_to_firm_uid_col = _first_present(row_to_firm.columns, ["foia_firm_uid"], "FOIA firm uid column")
    row_to_firm_row_uid_col = _first_present(row_to_firm.columns, ["foia_row_uid"], "FOIA row uid mapping column")

    firm_match_uid_col = _first_present(firm_matches.columns, ["foia_firm_uid"], "firm match uid column")
    rcid_col = _first_present(firm_matches.columns, ["rcid"], "rcid column")
    revelio_name_col = _first_present(
        firm_matches.columns,
        ["revelio_name", "preferred_company_name", "matched_company_name"],
        "matched company name column",
    )

    row_entities["foia_row_uid"] = row_entities[row_uid_col].astype("string")
    row_to_firm["foia_row_uid"] = row_to_firm[row_to_firm_row_uid_col].astype("string")
    row_to_firm["foia_firm_uid"] = row_to_firm[row_to_firm_uid_col].astype("string")
    firm_matches["foia_firm_uid"] = firm_matches[firm_match_uid_col].astype("string")
    firm_matches["rcid"] = firm_matches[rcid_col].map(_coerce_rcid).astype("Int64")

    validity_col = "crosswalk_validity_label" if "crosswalk_validity_label" in firm_matches.columns else None
    valid_mask = firm_matches["rcid"].notna()
    if validity_col is not None:
        valid_mask = valid_mask & firm_matches[validity_col].astype("string").eq("valid_match")
    valid_firm_matches = firm_matches.loc[valid_mask].copy()

    score_col = "score" if "score" in valid_firm_matches.columns else None
    sort_cols = ["foia_firm_uid"]
    ascending = [True]
    if score_col is not None:
        valid_firm_matches["_sort_score"] = pd.to_numeric(
            valid_firm_matches[score_col], errors="coerce"
        ).fillna(-1.0)
        sort_cols.append("_sort_score")
        ascending.append(False)
    valid_firm_matches = valid_firm_matches.sort_values(
        sort_cols + ["rcid", revelio_name_col],
        ascending=ascending + [True, True],
        kind="mergesort",
    )

    firm_rows: list[dict[str, Any]] = []
    grouped = valid_firm_matches.groupby("foia_firm_uid", sort=False) if not valid_firm_matches.empty else []
    for foia_firm_uid, group in grouped:
        rcids = _ordered_unique(_coerce_rcid(value) for value in group["rcid"].tolist())
        rcids = [value for value in rcids if value is not None]
        company_names = _ordered_unique(
            value
            for value in group[revelio_name_col].astype("string").fillna("").tolist()
            if value
        )
        preferred_rcid = rcids[0] if len(rcids) == 1 else pd.NA
        preferred_company_name = pd.NA
        preferred_match_source = pd.NA
        if len(rcids) == 1:
            preferred_rows = group[group["rcid"] == preferred_rcid]
            if not preferred_rows.empty:
                preferred_company_name = preferred_rows[revelio_name_col].astype("string").fillna("").iloc[0] or pd.NA
                if "match_source" in preferred_rows.columns:
                    preferred_match_source = preferred_rows["match_source"].astype("string").fillna("").iloc[0] or pd.NA
        if pd.notna(preferred_rcid) and (preferred_match_source is pd.NA or pd.isna(preferred_match_source)):
            preferred_match_source = normalized_source
        firm_rows.append(
            {
                "foia_firm_uid": foia_firm_uid,
                "preferred_rcid": preferred_rcid,
                "preferred_company_name": preferred_company_name,
                "preferred_match_source": preferred_match_source,
                "matched_rcids": rcids if rcids else None,
                "matched_company_names": company_names if company_names else None,
            }
        )

    firm_summary = pd.DataFrame(firm_rows)
    if firm_summary.empty:
        firm_summary = pd.DataFrame(
            columns=[
                "foia_firm_uid",
                "preferred_rcid",
                "preferred_company_name",
                "preferred_match_source",
                "matched_rcids",
                "matched_company_names",
            ]
        )

    row_base = (
        row_entities.rename(
            columns={
                employer_name_col: "employer_name",
                name_clean_col: "f1_empname_clean",
                city_clean_col: "f1_city_clean",
                state_clean_col: "f1_state_clean",
                zip_clean_col: "f1_zip_clean",
            }
        )[
            [
                "foia_row_uid",
                "employer_name",
                "f1_empname_clean",
                "f1_city_clean",
                "f1_state_clean",
                "f1_zip_clean",
            ]
        ]
        .drop_duplicates(subset=["foia_row_uid"], keep="first")
    )
    row_base = row_base.merge(
        row_to_firm[["foia_row_uid", "foia_firm_uid"]].drop_duplicates(),
        on="foia_row_uid",
        how="left",
    )
    out_df = row_base.merge(firm_summary, on="foia_firm_uid", how="left")
    out_df["preferred_rcid"] = pd.to_numeric(out_df["preferred_rcid"], errors="coerce").astype("Int64")
    out_df = out_df[
        [
            "employer_name",
            "f1_empname_clean",
            "f1_city_clean",
            "f1_state_clean",
            "f1_zip_clean",
            "foia_row_uid",
            "foia_firm_uid",
            "preferred_rcid",
            "preferred_company_name",
            "preferred_match_source",
            "matched_rcids",
            "matched_company_names",
        ]
    ].sort_values(
        ["f1_empname_clean", "f1_city_clean", "f1_state_clean", "f1_zip_clean", "employer_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_file, index=False)
        if verbose:
            print(f"Staged F1 employer crosswalk to {out_file}")

    if con is not None:
        _register_table(con, table_name, out_df)

    if verbose:
        matched_rows = int(out_df["preferred_rcid"].notna().sum())
        print(
            f"Staged {len(out_df):,} F1 employer rows from external artifacts "
            f"({matched_rows:,} rows with a unique preferred_rcid)."
        )
    return out_df


def build_preferred_rcid_activity(
    con,
    *,
    activity_parquet_path: str,
    sql_clean_company_name_expr: Callable[[str], str],
    sql_normalize_expr: Callable[[str], str],
    sql_state_name_to_abbr_expr: Callable[[str], str],
    sql_clean_zip_expr: Callable[[str], str],
    date_parse_sql: Callable[[str], str],
    individual_id_cols: Sequence[str],
    auth_start_cols: Sequence[str],
    year_col_candidates: Sequence[str] = ("year",),
    crosswalk_table: str = "f1_employer_final_crosswalk",
    auth_counts_table: str = "f1_employer_auth_counts",
    preferred_rcids_table: str = "f1_preferred_rcids_multi_year",
    auth_counts_path: str | None = None,
    rcid_list_path: str | None = None,
    save_output: bool = False,
    min_unique_individuals: int = 1,
    min_years: int = 3,
    min_max_year_gt: int | None = 2012,
    verbose: bool = False,
) -> str:
    escaped_activity_path = activity_parquet_path.replace("'", "''")
    cols = [row[0] for row in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{escaped_activity_path}')").fetchall()]
    id_col = _first_present(cols, individual_id_cols, "individual identifier column")
    auth_col = _first_present(cols, auth_start_cols, "authorization start column")
    year_col = _first_present(cols, year_col_candidates, "FOIA year column")

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW foia_with_auth AS
        SELECT
          *,
          {sql_clean_company_name_expr('employer_name')} AS f1_empname_clean,
          {sql_normalize_expr('employer_city')} AS f1_city_clean,
          {sql_state_name_to_abbr_expr('employer_state')} AS f1_state_clean,
          {sql_clean_zip_expr('employer_zip_code')} AS f1_zip_clean,
          {date_parse_sql(auth_col)} AS auth_start
        FROM read_parquet('{escaped_activity_path}')
        WHERE employer_name IS NOT NULL
        """
    )

    con.sql(
        f"""
        CREATE OR REPLACE TABLE {auth_counts_table} AS
        SELECT
          CAST(EXTRACT(YEAR FROM auth_start) AS INTEGER) AS auth_year,
          preferred_company_name,
          CAST(fcw.preferred_rcid AS BIGINT) AS preferred_rcid,
          COUNT(DISTINCT CAST(fo.{id_col} AS VARCHAR)) AS unique_individuals
        FROM foia_with_auth AS fo
        JOIN {crosswalk_table} AS fcw
          ON fo.f1_empname_clean = fcw.f1_empname_clean
         AND COALESCE(fo.f1_city_clean, '') = COALESCE(fcw.f1_city_clean, '')
         AND COALESCE(fo.f1_state_clean, '') = COALESCE(fcw.f1_state_clean, '')
         AND COALESCE(fo.f1_zip_clean, '') = COALESCE(fcw.f1_zip_clean, '')
        WHERE auth_start IS NOT NULL
          AND fcw.preferred_rcid IS NOT NULL
          AND CAST(EXTRACT(YEAR FROM auth_start) AS INTEGER) = CAST({year_col} AS INTEGER)
        GROUP BY auth_year, preferred_rcid, preferred_company_name
        """
    )

    having_parts = [f"COUNT(DISTINCT auth_year) >= {int(min_years)}"]
    if min_max_year_gt is not None:
        having_parts.append(f"MAX(auth_year) > {int(min_max_year_gt)}")
    having_sql = " AND ".join(having_parts)
    con.sql(
        f"""
        CREATE OR REPLACE TABLE {preferred_rcids_table} AS
        SELECT preferred_rcid
        FROM {auth_counts_table}
        WHERE unique_individuals >= {int(min_unique_individuals)}
        GROUP BY preferred_rcid
        HAVING {having_sql}
        """
    )

    if save_output:
        if auth_counts_path:
            con.sql(
                f"COPY (SELECT * FROM {auth_counts_table}) TO '{auth_counts_path}' (FORMAT PARQUET)"
            )
        if rcid_list_path:
            con.sql(
                f"COPY (SELECT * FROM {preferred_rcids_table}) TO '{rcid_list_path}' (FORMAT PARQUET)"
            )

    if verbose:
        auth_rows = int(con.sql(f"SELECT COUNT(*) FROM {auth_counts_table}").fetchone()[0])
        rcid_rows = int(con.sql(f"SELECT COUNT(*) FROM {preferred_rcids_table}").fetchone()[0])
        print(
            f"Built {auth_counts_table} ({auth_rows:,} rows) and "
            f"{preferred_rcids_table} ({rcid_rows:,} RCIDs)."
        )
    return preferred_rcids_table
