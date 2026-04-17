"""FOIA institution preprocessing for stage 02_rev_import."""

from __future__ import annotations

from pathlib import Path

import duckdb as ddb
import pandas as pd

from common import (
    DEFAULT_TOKEN_STOPWORDS,
    build_token_boundary_regex,
    clean_institution_text,
    escape_sql_literal,
)

ACRONYM_STOPWORDS = {"and", "at", "for", "in", "of", "the"}


def build_foia_school_token_artifacts(
    *,
    foia_parquet_path: str | Path,
    token_artifact_csv: str | Path,
    compiled_regex_txt: str | Path,
    foia_school_column: str = "school_name",
    token_min_len: int = 3,
    token_top_n: int | None = None,
    token_stopwords: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, int | str]:
    foia_path = Path(foia_parquet_path)
    token_csv_path = Path(token_artifact_csv)
    regex_txt_path = Path(compiled_regex_txt)
    if not foia_path.exists():
        raise FileNotFoundError(f"FOIA parquet not found for preprocessing: {foia_path}")

    token_csv_path.parent.mkdir(parents=True, exist_ok=True)
    regex_txt_path.parent.mkdir(parents=True, exist_ok=True)

    if token_csv_path.exists() and regex_txt_path.exists() and not overwrite:
        token_df = pd.read_csv(token_csv_path)
        regex_pattern = regex_txt_path.read_text().strip()
        return {
            "foia_input_path": str(foia_path),
            "token_artifact_csv": str(token_csv_path),
            "compiled_regex_txt": str(regex_txt_path),
            "distinct_institutions": int(token_df["n_institutions"].max()) if not token_df.empty else 0,
            "n_tokens": int(len(token_df)),
            "regex_length": len(regex_pattern),
        }

    con = ddb.connect()
    escaped_path = escape_sql_literal(foia_path)
    cols = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{escaped_path}')").fetchall()]
    if foia_school_column not in cols:
        raise ValueError(
            f"FOIA preprocessing column '{foia_school_column}' is missing from {foia_path}. "
            f"Available columns include: {cols[:20]}"
        )

    school_df = con.execute(
        f"""
        SELECT DISTINCT CAST({foia_school_column} AS VARCHAR) AS school_name
        FROM read_parquet('{escaped_path}')
        WHERE {foia_school_column} IS NOT NULL
          AND trim(CAST({foia_school_column} AS VARCHAR)) != ''
        ORDER BY school_name
        """
    ).df()

    stopwords = {token.lower() for token in (token_stopwords or [])}
    stopwords |= DEFAULT_TOKEN_STOPWORDS
    school_df["school_clean"] = school_df["school_name"].map(clean_institution_text)
    school_df = school_df[school_df["school_clean"].notna()].reset_index(drop=True)

    token_rows: list[dict[str, int | str]] = []
    for school_clean in school_df["school_clean"].tolist():
        raw_tokens = school_clean.split()
        school_tokens = {
            token
            for token in raw_tokens
            if len(token) >= token_min_len and not token.isdigit() and token not in stopwords
        }
        acronym_tokens = [
            token
            for token in raw_tokens
            if token not in ACRONYM_STOPWORDS and not token.isdigit()
        ]
        if len(acronym_tokens) >= 2:
            acronym = "".join(token[0] for token in acronym_tokens)
            if len(acronym) >= token_min_len:
                school_tokens.add(acronym)
        for token in school_tokens:
            token_rows.append({"token": token, "school_clean": school_clean})

    token_df = pd.DataFrame(token_rows)
    if token_df.empty:
        token_summary = pd.DataFrame(columns=["token", "n_institutions", "share_institutions", "token_rank"])
        regex_pattern = r"(?!)"
    else:
        token_summary = (
            token_df.groupby("token", as_index=False)["school_clean"]
            .nunique()
            .rename(columns={"school_clean": "n_institutions"})
            .sort_values(["n_institutions", "token"], ascending=[False, True])
            .reset_index(drop=True)
        )
        token_summary["share_institutions"] = (
            token_summary["n_institutions"] / max(1, school_df["school_clean"].nunique())
        )
        if token_top_n is not None:
            token_summary = token_summary.head(int(token_top_n)).copy()
        token_summary["token_rank"] = range(1, len(token_summary) + 1)
        regex_pattern = build_token_boundary_regex(token_summary["token"].tolist())

    token_summary.to_csv(token_csv_path, index=False)
    regex_txt_path.write_text(regex_pattern + "\n")

    return {
        "foia_input_path": str(foia_path),
        "token_artifact_csv": str(token_csv_path),
        "compiled_regex_txt": str(regex_txt_path),
        "distinct_institutions": int(school_df["school_clean"].nunique()),
        "n_tokens": int(len(token_summary)),
        "regex_length": len(regex_pattern),
    }
