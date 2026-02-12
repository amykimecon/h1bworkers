"""Link FOIA SEVP records across years into person IDs.

Creates a crosswalk of individual_key x year to person_id and
writes the original dataset with person_id + flags.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import duckdb as ddb
import pandas as pd
import pyarrow.parquet as pq

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *  # noqa: F401,F403

FOIA_INPUT_PATH = f"{root}/data/int/foia_sevp_combined_raw.parquet"
CROSSWALK_OUTPUT_PATH = f"{root}/data/int/foia_person_key_year_crosswalk.parquet"
FULL_OUTPUT_PATH = f"{root}/data/int/foia_sevp_with_person_id.parquet"
TMP_DIR = f"{root}/data/int/foia_person_linkage_tmp"

def _escape(path: Path) -> str:
    return str(path).replace("'", "''")


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _get_columns(con: ddb.DuckDBPyConnection, view: str) -> list[str]:
    rows = con.execute(f"PRAGMA table_info('{view}')").fetchall()
    return [r[1] for r in rows]


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
    packed = ", ".join(f"{_quote_ident(col)} := {_quote_ident(col)}" for col in cols)
    return f"md5(to_json(struct_pack({packed})))"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create person IDs across FOIA SEVP years.")
    parser.add_argument(
        "--input",
        default=FOIA_INPUT_PATH,
        help="Path to foia_sevp_combined_raw.parquet.",
    )
    parser.add_argument(
        "--crosswalk-out",
        default=CROSSWALK_OUTPUT_PATH,
        help="Path to output crosswalk parquet (individual_key x year to person_id).",
    )
    parser.add_argument(
        "--full-out",
        default=FULL_OUTPUT_PATH,
        help="Path to output full parquet (original data with person_id and flags).",
    )
    parser.add_argument(
        "--tmp-dir",
        default=TMP_DIR,
        help="Temporary directory for intermediate parquet files.",
    )
    return parser.parse_args()


def _prompt_path(label: str, default: str) -> str:
    prompt = f"{label} [{default}]: "
    try:
        value = input(prompt).strip()
    except EOFError:
        return default
    return value or default


def run_person_id_linkage(
    *,
    con: ddb.DuckDBPyConnection | None = None,
    input_path: str | Path = FOIA_INPUT_PATH,
    crosswalk_out_path: str | Path = CROSSWALK_OUTPUT_PATH,
    full_out_path: str | Path = FULL_OUTPUT_PATH,
    tmp_dir: str | Path = TMP_DIR,
) -> None:
    input_path = Path(input_path)
    crosswalk_out_path = Path(crosswalk_out_path)
    full_out_path = Path(full_out_path)
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    con.sql(f"CREATE OR REPLACE TEMP TABLE foia_raw AS SELECT * FROM read_parquet('{_escape(input_path)}')")
    print("Loaded foia_raw.")

    cols = _get_columns(con, "foia_raw")
    col_map = {c.lower(): c for c in cols}
    required = {"filename","student_key", "individual_key", "year"}
    missing = [c for c in required if c not in col_map]
    if missing:
        raise ValueError(f"Missing required columns in foia_raw: {missing}")

    student_key_col = col_map["student_key"]
    individual_key_col = col_map["individual_key"]
    year_col = col_map["year"]

    group_cols = [c for c in cols if c.lower() not in required]
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
        WHERE try_cast({_quote_ident(year_col)} AS INTEGER) > 2005 -- filter early so expensive grouping only runs on eligible years
        """
    )
    print(f"Built foia_grouped (raw data with group key defined based on all fields except {', '.join(required)}).")

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
    print("Built group_year (grouped on group key).")

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
    total_groups = con.execute("SELECT COUNT(DISTINCT group_key) FROM group_year").fetchone()[0]
    total_group_years = con.execute("SELECT COUNT(*) FROM group_year").fetchone()[0]
    flagged_multi_years = con.execute(
        "SELECT COUNT(*) FROM group_year_counts WHERE n_individual_keys > 1"
    ).fetchone()[0]
    clean_group_years = con.execute(
        "SELECT COUNT(*) FROM group_year_counts WHERE n_individual_keys = 1"
    ).fetchone()[0]
    print("Built group_year_counts.")
    print(f"Initial distinct groups: {total_groups:,}")
    print(f"Distinct group x year pairs: {total_group_years:,}")
    print(f"Flagged group x year (multi-individual): {flagged_multi_years:,}")
    print(f"Clean group x year (single-individual): {clean_group_years:,}")

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
    print("Built group_year_clean.")

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
    subgroup_count = con.execute(
        "SELECT COUNT(DISTINCT CONCAT(group_key, '::', CAST(seq_id AS VARCHAR))) FROM group_year_sequence"
    ).fetchone()[0]
    print("Built group_year_sequence.")
    print(f"Consecutive subgroups (group_key split by gaps): {subgroup_count:,}")

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
    group_year_key_rows = con.execute("SELECT COUNT(*) FROM group_year_key").fetchone()[0]
    print("Built group_year_key.")
    print(f"Subgroup x year x key rows: {group_year_key_rows:,}")

    con.sql(
        """
        CREATE OR REPLACE TEMP TABLE group_id_index AS
        SELECT
            consecutive_group_id,
            ROW_NUMBER() OVER () - 1 AS gid_idx
        FROM (SELECT DISTINCT consecutive_group_id FROM group_year_key)
        """
    )
    group_count = con.execute("SELECT COUNT(*) FROM group_id_index").fetchone()[0]
    print(f"Indexed {group_count:,} consecutive groups.")

    edges_path = tmp_dir / "foia_person_edges.parquet"
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
    key_year_pairs = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT individual_key_norm, year FROM group_year_key)"
    ).fetchone()[0]
    print(f"Wrote union edges to {edges_path}.")
    print(f"Distinct key x year pairs: {key_year_pairs:,}")

    uf = UnionFind(group_count)
    edges_file = pq.ParquetFile(edges_path)
    edge_rows = 0
    for batch in edges_file.iter_batches(columns=["a", "b"]):
        a_vals = batch.column(0).to_numpy(zero_copy_only=False)
        b_vals = batch.column(1).to_numpy(zero_copy_only=False)
        edge_rows += len(a_vals)
        for a_idx, b_idx in zip(a_vals, b_vals):
            uf.union(int(a_idx), int(b_idx))
        if edge_rows % 5_000_000 == 0:
            print(f"Processed {edge_rows:,} union edges...")
    print(f"Processed {edge_rows:,} union edges total.")

    group_ids = [
        r[0]
        for r in con.execute(
            "SELECT consecutive_group_id FROM group_id_index ORDER BY gid_idx"
        ).fetchall()
    ]
    root_to_person: dict[int, int] = {}
    person_rows = []
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

    con.sql(f"COPY (SELECT * FROM key_year_crosswalk) TO '{_escape(crosswalk_out_path)}' (FORMAT PARQUET)")
    con.sql(
        f"""
        COPY (
            SELECT *
            FROM foia_with_person
            ORDER BY original_row_num
        ) TO '{_escape(full_out_path)}' (FORMAT PARQUET)
        """
    )

    crosswalk_count = con.execute("SELECT COUNT(*) FROM key_year_crosswalk").fetchone()[0]
    person_count = con.execute("SELECT COUNT(DISTINCT person_id) FROM key_year_crosswalk WHERE person_id IS NOT NULL").fetchone()[0]
    flagged_multi = con.execute(
        "SELECT COUNT(*) FROM key_year_crosswalk WHERE flag_multi_individual_same_year = 1"
    ).fetchone()[0]
    flagged_conflict = con.execute(
        "SELECT COUNT(*) FROM key_year_crosswalk WHERE flag_key_year_conflict = 1"
    ).fetchone()[0]

    print(f"Wrote {crosswalk_out_path}")
    print(f"Wrote {full_out_path}")
    print(f"Crosswalk rows: {crosswalk_count:,}")
    print(f"Person IDs: {person_count:,}")
    print(f"Flagged multi-individual years: {flagged_multi:,}")
    print(f"Flagged key-year conflicts: {flagged_conflict:,}")


def main() -> None:
    args = _parse_args()
    
    con = ddb.connect()
    run_person_id_linkage(
        con=con,
        input_path=args.input,
        crosswalk_out_path=args.crosswalk_out,
        full_out_path=args.full_out,
        tmp_dir=args.tmp_dir,
    )


# if __name__ == "__main__":
#     main()
