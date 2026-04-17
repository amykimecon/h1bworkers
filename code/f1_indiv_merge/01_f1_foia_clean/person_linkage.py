"""Local person-linkage logic for stage 01_f1_foia_clean."""

from __future__ import annotations

from builtins import print as _print
from functools import partial
from pathlib import Path
from typing import Iterable

import duckdb as ddb
import pandas as pd
import pyarrow.parquet as pq

print = partial(_print, flush=True)


def _escape(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _get_columns(con: ddb.DuckDBPyConnection, view: str) -> list[str]:
    rows = con.execute(f"PRAGMA table_info('{view}')").fetchall()
    return [row[1] for row in rows]


def _group_value_expr(col: str) -> str:
    ident = _quote_ident(col)
    if col.lower().endswith("_zip_code"):
        trimmed = f"trim(CAST({ident} AS VARCHAR))"
        normalized = (
            "CASE "
            f"WHEN {ident} IS NULL THEN NULL "
            f"WHEN {trimmed} = '' THEN NULL "
            f"WHEN regexp_full_match({trimmed}, '^[0-9]+\\.0+$') THEN split_part({trimmed}, '.', 1) "
            f"ELSE {trimmed} "
            "END"
        )
        # Excel-style numeric ZIP exports (for example 10154.0) should not split
        # otherwise identical people from text ZIP values such as 10154.
        # Some exports also drop a leading zero (for example 02142 -> 2142),
        # which should be normalized back to a 5-digit ZIP.
        return (
            "CASE "
            f"WHEN {normalized} IS NULL THEN NULL "
            f"WHEN regexp_full_match({normalized}, '^[0-9]{{4}}$') THEN lpad({normalized}, 5, '0') "
            f"ELSE {normalized} "
            "END"
        )
    return ident


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, idx: int) -> int:
        parent = self.parent
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def _build_group_key_expr(group_cols: Iterable[str]) -> str:
    cols = list(group_cols)
    if not cols:
        return "md5('')"
    packed = ", ".join(f"{_quote_ident(col)} := {_group_value_expr(col)}" for col in cols)
    return f"md5(to_json(struct_pack({packed})))"


def run_person_id_linkage(
    *,
    con: ddb.DuckDBPyConnection | None = None,
    input_path: str | Path,
    crosswalk_out_path: str | Path,
    employment_corrected_out_path: str | Path,
    tmp_dir: str | Path,
    full_out_path: str | Path | None = None,
    min_year: int = 2006,
    correction_year: int = 2015,
    overwrite: bool = False,
) -> dict[str, int]:
    input_path = Path(input_path)
    crosswalk_out_path = Path(crosswalk_out_path)
    employment_corrected_out_path = Path(employment_corrected_out_path)
    full_out_path = Path(full_out_path) if full_out_path is not None else None
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Combined raw FOIA parquet not found: {input_path}")

    if (
        not overwrite
        and crosswalk_out_path.exists()
        and employment_corrected_out_path.exists()
        and (full_out_path is None or full_out_path.exists())
    ):
        return {
            "reused_outputs": 1,
        }

    own_con = con is None
    if own_con:
        con = ddb.connect()
    assert con is not None

    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE foia_raw AS
        SELECT *
        FROM read_parquet('{_escape(input_path)}')
        """
    )
    print("Loaded foia_raw.")

    cols = _get_columns(con, "foia_raw")
    col_map = {col.lower(): col for col in cols}
    required = {"filename", "student_key", "individual_key", "year"}
    missing = [col for col in required if col not in col_map]
    if missing:
        raise ValueError(f"Missing required columns in foia_raw: {missing}")

    individual_key_col = col_map["individual_key"]
    year_col = col_map["year"]
    group_cols = [col for col in cols if col.lower() not in required]
    group_key_expr = _build_group_key_expr(group_cols)

    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE foia_grouped AS
        SELECT
            *,
            rowid AS original_row_num,
            {group_key_expr} AS group_key,
            try_cast({_quote_ident(year_col)} AS INTEGER) AS year_int,
            COALESCE(CAST({_quote_ident(individual_key_col)} AS VARCHAR), '__NULL__') AS individual_key_norm
        FROM foia_raw
        WHERE try_cast({_quote_ident(year_col)} AS INTEGER) >= {int(min_year)}
        """
    )
    print("Built foia_grouped.")

    con.sql(
        """
        CREATE OR REPLACE TEMP TABLE group_year AS
        SELECT DISTINCT
            group_key,
            year_int AS year,
            individual_key_norm
        FROM foia_grouped
        WHERE year_int IS NOT NULL
        """
    )

    con.sql(
        """
        CREATE OR REPLACE TEMP TABLE group_year_counts AS
        SELECT
            group_key,
            year,
            COUNT(DISTINCT individual_key_norm) AS n_individual_keys
        FROM group_year
        GROUP BY group_key, year
        """
    )

    con.sql(
        """
        CREATE OR REPLACE TEMP TABLE group_year_clean AS
        SELECT
            gy.group_key,
            gy.year,
            gy.individual_key_norm
        FROM group_year gy
        JOIN group_year_counts gc
            ON gy.group_key = gc.group_key
           AND gy.year = gc.year
        WHERE gc.n_individual_keys = 1
        """
    )

    con.sql(
        """
        CREATE OR REPLACE TEMP TABLE group_year_sequence AS
        WITH ordered AS (
            SELECT
                group_key,
                year,
                individual_key_norm,
                LAG(year) OVER (PARTITION BY group_key ORDER BY year) AS prev_year
            FROM group_year_clean
        )
        SELECT
            group_key,
            year,
            individual_key_norm,
            SUM(
                CASE
                    WHEN prev_year IS NULL THEN 0
                    WHEN year - prev_year > 1 THEN 1
                    ELSE 0
                END
            ) OVER (
                PARTITION BY group_key
                ORDER BY year
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS seq_id
        FROM ordered
        """
    )

    con.sql(
        """
        CREATE OR REPLACE TEMP TABLE group_year_key AS
        SELECT DISTINCT
            CONCAT(group_key, '::', CAST(seq_id AS VARCHAR)) AS consecutive_group_id,
            year,
            individual_key_norm
        FROM group_year_sequence
        """
    )

    con.sql(
        """
        CREATE OR REPLACE TEMP TABLE group_id_index AS
        SELECT
            consecutive_group_id,
            ROW_NUMBER() OVER () - 1 AS gid_idx
        FROM (SELECT DISTINCT consecutive_group_id FROM group_year_key)
        """
    )
    group_count = int(con.execute("SELECT COUNT(*) FROM group_id_index").fetchone()[0])
    print(f"Indexed {group_count:,} consecutive groups.")

    edges_path = tmp_dir / "foia_person_edges.parquet"
    if edges_path.exists():
        edges_path.unlink()
    con.sql(
        f"""
        COPY (
            WITH key_year AS (
                SELECT
                    g.individual_key_norm,
                    g.year,
                    i.gid_idx
                FROM group_year_key g
                JOIN group_id_index i
                    ON g.consecutive_group_id = i.consecutive_group_id
            ),
            base AS (
                SELECT
                    individual_key_norm,
                    year,
                    MIN(gid_idx) AS base_idx,
                    COUNT(*) AS n_groups
                FROM key_year
                GROUP BY individual_key_norm, year
                HAVING COUNT(*) > 1
            )
            SELECT
                b.base_idx AS a,
                k.gid_idx AS b
            FROM key_year k
            JOIN base b
                ON k.individual_key_norm = b.individual_key_norm
               AND k.year = b.year
            WHERE k.gid_idx <> b.base_idx
        ) TO '{_escape(edges_path)}' (FORMAT PARQUET)
        """
    )

    uf = UnionFind(group_count)
    edge_rows = 0
    if group_count > 0:
        edges_file = pq.ParquetFile(edges_path)
        for batch in edges_file.iter_batches(columns=["a", "b"]):
            a_vals = batch.column(0).to_numpy(zero_copy_only=False)
            b_vals = batch.column(1).to_numpy(zero_copy_only=False)
            edge_rows += len(a_vals)
            for a_idx, b_idx in zip(a_vals, b_vals):
                uf.union(int(a_idx), int(b_idx))
    print(f"Processed {edge_rows:,} union edges total.")

    group_ids = [
        row[0]
        for row in con.execute(
            "SELECT consecutive_group_id FROM group_id_index ORDER BY gid_idx"
        ).fetchall()
    ]
    root_to_person: dict[int, int] = {}
    person_rows: list[tuple[str, int]] = []
    for idx, gid in enumerate(group_ids):
        root = uf.find(idx)
        person_id = root_to_person.setdefault(root, len(root_to_person) + 1)
        person_rows.append((gid, person_id))
    print(f"Assigned {len(root_to_person):,} person IDs.")

    person_map = pd.DataFrame(person_rows, columns=["consecutive_group_id", "person_id"])
    con.register("person_map", person_map)

    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW person_group_year AS
        SELECT
            pm.person_id,
            g.year,
            g.individual_key_norm
        FROM group_year_key g
        JOIN person_map pm
            ON g.consecutive_group_id = pm.consecutive_group_id
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TEMP VIEW person_conflicts AS
        SELECT DISTINCT person_id
        FROM (
            SELECT
                person_id,
                year,
                COUNT(DISTINCT individual_key_norm) AS n_keys
            FROM person_group_year
            GROUP BY person_id, year
        )
        WHERE n_keys > 1
        """
    )

    con.sql(
        f"""
        CREATE OR REPLACE TABLE foia_with_person AS
        SELECT
            fg.original_row_num,
            fg.* EXCLUDE (original_row_num, individual_key_norm, group_key),
            pm.person_id AS person_id,
            CASE WHEN gc.n_individual_keys > 1 THEN 1 ELSE 0 END AS flag_multi_individual_same_year,
            CASE WHEN fg.year_int IS NULL THEN 1 ELSE 0 END AS flag_bad_year,
            CASE WHEN pc.person_id IS NOT NULL THEN 1 ELSE 0 END AS flag_key_year_conflict,
            CASE WHEN pm.person_id IS NULL THEN 1 ELSE 0 END AS flag_ungrouped
        FROM foia_grouped fg
        LEFT JOIN group_year_counts gc
            ON fg.group_key = gc.group_key
           AND fg.year_int = gc.year
        LEFT JOIN group_year_sequence gs
            ON fg.group_key = gs.group_key
           AND fg.year_int = gs.year
           AND fg.individual_key_norm = gs.individual_key_norm
        LEFT JOIN person_map pm
            ON CONCAT(fg.group_key, '::', CAST(gs.seq_id AS VARCHAR)) = pm.consecutive_group_id
        LEFT JOIN person_conflicts pc
            ON pm.person_id = pc.person_id
        """
    )

    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW key_year_crosswalk AS
        SELECT DISTINCT
            {_quote_ident(individual_key_col)} AS individual_key,
            year_int AS year,
            person_id,
            flag_multi_individual_same_year,
            flag_key_year_conflict,
            flag_bad_year,
            flag_ungrouped
        FROM foia_with_person
        WHERE year_int IS NOT NULL
        """
    )

    crosswalk_out_path.parent.mkdir(parents=True, exist_ok=True)
    employment_corrected_out_path.parent.mkdir(parents=True, exist_ok=True)
    if crosswalk_out_path.exists():
        crosswalk_out_path.unlink()
    if employment_corrected_out_path.exists():
        employment_corrected_out_path.unlink()
    if full_out_path is not None:
        full_out_path.parent.mkdir(parents=True, exist_ok=True)
        if full_out_path.exists():
            full_out_path.unlink()

    con.sql(
        f"""
        COPY (SELECT * FROM key_year_crosswalk)
        TO '{_escape(crosswalk_out_path)}' (FORMAT PARQUET)
        """
    )

    if full_out_path is not None:
        con.sql(
            f"""
            COPY (
                SELECT *
                FROM foia_with_person
                ORDER BY original_row_num
            ) TO '{_escape(full_out_path)}' (FORMAT PARQUET)
            """
        )

    employer_key_cols = [
        "employer_name",
        "employer_city",
        "employer_state",
        "employer_zip_code",
        "employment_description",
    ]
    spell_key_cols = [
        "employment_opt_type",
        "authorization_start_date",
        "authorization_end_date",
        "opt_authorization_start_date",
        "opt_authorization_end_date",
        "opt_employer_start_date",
        "opt_employer_end_date",
    ]

    fw_cols = _get_columns(con, "foia_with_person")
    fw_col_map = {col.lower(): col for col in fw_cols}
    missing_correction_cols = [
        col for col in employer_key_cols + spell_key_cols if col not in fw_col_map
    ]

    if missing_correction_cols:
        print(
            "Skipping employment correction because required columns are missing: "
            + ", ".join(missing_correction_cols)
        )
        con.sql(
            f"""
            COPY (
                SELECT *
                FROM foia_with_person
                ORDER BY original_row_num
            ) TO '{_escape(employment_corrected_out_path)}' (FORMAT PARQUET)
            """
        )
    else:
        employer_expr = _build_group_key_expr(fw_col_map[col] for col in employer_key_cols)
        spell_expr = _build_group_key_expr(fw_col_map[col] for col in spell_key_cols)

        con.sql(
            f"""
            CREATE OR REPLACE TEMP TABLE foia_employment_keys AS
            SELECT
                *,
                {employer_expr} AS employer_key,
                {spell_expr} AS spell_key
            FROM foia_with_person
            """
        )
        con.sql(
            f"""
            CREATE OR REPLACE TEMP TABLE person_historical_pairs AS
            SELECT DISTINCT person_id, employer_key, spell_key
            FROM foia_employment_keys
            WHERE person_id IS NOT NULL
              AND year_int <= {int(correction_year) - 1}
            """
        )
        con.sql(
            f"""
            CREATE OR REPLACE TEMP TABLE person_historical_employers AS
            SELECT DISTINCT person_id, employer_key
            FROM foia_employment_keys
            WHERE person_id IS NOT NULL
              AND year_int <= {int(correction_year) - 1}
            """
        )
        con.sql(
            f"""
            CREATE OR REPLACE TEMP TABLE person_historical_spells AS
            SELECT DISTINCT person_id, spell_key
            FROM foia_employment_keys
            WHERE person_id IS NOT NULL
              AND year_int <= {int(correction_year) - 1}
            """
        )
        con.sql(
            f"""
            CREATE OR REPLACE TEMP TABLE foia_employment_fix_flags AS
            SELECT
                f.*,
                CASE
                    WHEN f.year_int >= {int(correction_year)}
                     AND f.person_id IS NOT NULL
                     AND he.person_id IS NOT NULL
                     AND hs.person_id IS NOT NULL
                     AND hp.person_id IS NULL
                    THEN 1 ELSE 0
                END AS flag_incorrect_employer_spell_pair,
                CASE
                    WHEN f.year_int >= {int(correction_year)}
                     AND f.person_id IS NOT NULL
                     AND hp.person_id IS NULL
                     AND (he.person_id IS NULL OR hs.person_id IS NULL)
                    THEN 1 ELSE 0
                END AS flag_unresolved_employer_spell_pair
            FROM foia_employment_keys f
            LEFT JOIN person_historical_pairs hp
                ON f.person_id = hp.person_id
               AND f.employer_key = hp.employer_key
               AND f.spell_key = hp.spell_key
            LEFT JOIN person_historical_employers he
                ON f.person_id = he.person_id
               AND f.employer_key = he.employer_key
            LEFT JOIN person_historical_spells hs
                ON f.person_id = hs.person_id
               AND f.spell_key = hs.spell_key
            """
        )
        con.sql(
            f"""
            COPY (
                SELECT *
                FROM foia_employment_fix_flags
                WHERE flag_incorrect_employer_spell_pair = 0
                ORDER BY original_row_num
            ) TO '{_escape(employment_corrected_out_path)}' (FORMAT PARQUET)
            """
        )

    crosswalk_count = int(con.execute("SELECT COUNT(*) FROM key_year_crosswalk").fetchone()[0])
    person_count = int(
        con.execute(
            "SELECT COUNT(DISTINCT person_id) FROM key_year_crosswalk WHERE person_id IS NOT NULL"
        ).fetchone()[0]
    )
    corrected_count = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{_escape(employment_corrected_out_path)}')"
        ).fetchone()[0]
    )

    print(f"Wrote {crosswalk_out_path}")
    print(f"Wrote {employment_corrected_out_path}")
    if full_out_path is not None:
        print(f"Wrote {full_out_path}")
    print(f"Crosswalk rows: {crosswalk_count:,}")
    print(f"Person IDs: {person_count:,}")
    print(f"Corrected person-panel rows: {corrected_count:,}")

    if own_con:
        con.close()

    return {
        "crosswalk_rows": crosswalk_count,
        "person_ids": person_count,
        "employment_corrected_rows": corrected_count,
    }
