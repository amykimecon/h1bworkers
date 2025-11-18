"""Run adaptive fuzzy clustering on a full dataset with human-in-the-loop labelling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
import duckdb

from config import root  # type: ignore
from adaptive_fuzzy import (
    build_clusters,
    clusters_to_frame,
    generate_pair_candidates,
    set_token_statistics_from_names,
)
from adaptive_fuzzy.cli import (
    DEFAULT_ARCHIVE_PATH,
    collect_initial_labels,
    interactive_training,
)


def load_names(input_path: Path, column: str, limit: Optional[int]) -> list[str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    if input_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(input_path)
    else:
        raise ValueError(
            f"Unsupported file extension for {input_path}. Use csv, txt, parquet, or pq."
        )

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {input_path}")

    names = df[column].astype(str).str.strip()
    if limit is not None:
        names = names.iloc[:limit]
    unique = names.dropna().unique().tolist()
    return unique


def iter_chunk_prefixes(
    input_path: Path,
    column: str,
    prefix_length: int,
    prefixes: Optional[Sequence[str]],
) -> Iterable[str]:
    con = duckdb.connect()
    dataset = f"read_parquet('{input_path.as_posix()}')"
    col_expr = f'"{column}"'
    if prefixes:
        for prefix in prefixes:
            yield prefix.lower()
    else:
        query = (
            f"SELECT DISTINCT lower(substr({col_expr}, 1, {prefix_length})) AS prefix "
            f"FROM {dataset} "
            f"WHERE length(trim({col_expr})) > 0"
        )
        rows = con.execute(query).fetchall()
        for (prefix,) in rows:
            yield prefix or ""
    con.close()


def load_names_for_prefix(
    input_path: Path,
    column: str,
    prefix_length: int,
    prefix: str,
) -> list[str]:
    con = duckdb.connect()
    dataset = f"read_parquet('{input_path.as_posix()}')"
    col_expr = f'"{column}"'
    if prefix:
        query = (
            f"SELECT DISTINCT {col_expr} FROM {dataset} "
            f"WHERE lower(substr({col_expr}, 1, {prefix_length})) = ?"
        )
        df = con.execute(query, [prefix]).df()
    else:
        query = (
            f"SELECT DISTINCT {col_expr} FROM {dataset} "
            f"WHERE lower(substr({col_expr}, 1, {prefix_length})) = '' "
            f"OR {col_expr} IS NULL"
        )
        df = con.execute(query).df()
    con.close()
    return df[column].astype(str).str.strip().dropna().unique().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive fuzzy university clustering over a full dataset.",
    )
    default_input = Path(root) / "data/int/wrds_users_sep2.parquet"

    parser.add_argument("--input", type=Path, default=default_input, help="Input CSV/Parquet containing university names.")
    parser.add_argument("--column", type=str, default="university_raw", help="Column containing university names.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of names to load.")
    parser.add_argument("--initial-labels", type=int, default=25, help="Number of top candidates to label before training.")
    parser.add_argument("--batch-size", type=int, default=15, help="Number of uncertain pairs to review per iteration.")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum number of interactive refinement iterations.")
    parser.add_argument("--convergence-threshold", type=float, default=0.02, help="Stop when probability change falls below this value.")
    parser.add_argument("--match-threshold", type=float, default=0.6, help="Probability threshold for clustering pairs together.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV to write the resulting clusters.")
    parser.add_argument(
        "--save-labels",
        type=Path,
        default=DEFAULT_ARCHIVE_PATH,
        help="Path to persist collected training labels (default: ~/.adaptive_fuzzy/label_history.csv).",
    )
    parser.add_argument(
        "--chunk-prefix-length",
        type=int,
        default=0,
        help="Process the dataset in chunks defined by the first N characters (0 = no chunking).",
    )
    parser.add_argument(
        "--chunk-prefixes",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of prefixes to run (only used when chunk-prefix-length > 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.chunk_prefix_length > 0:
        prefixes = list(
            iter_chunk_prefixes(args.input, args.column, args.chunk_prefix_length, args.chunk_prefixes)
        )
        if not prefixes:
            raise ValueError("No prefixes discovered; check the input column.")
        print(f"Processing dataset in {len(prefixes)} chunks (prefix length = {args.chunk_prefix_length}).")

        for idx, prefix in enumerate(prefixes, start=1):
            names = load_names_for_prefix(args.input, args.column, args.chunk_prefix_length, prefix)
            if len(names) < 2:
                print(f"[{idx}/{len(prefixes)}] Prefix '{prefix}' skipped (insufficient unique names).")
                continue

            print(f"[{idx}/{len(prefixes)}] Loaded {len(names):,} names for prefix '{prefix or '[empty]'}'.")
            run_chunk(
                names,
                args,
                chunk_label=f"prefix '{prefix or '[empty]'}'",
            )
    else:
        names = load_names(args.input, args.column, args.limit)
        if len(names) < 2:
            raise ValueError("Need at least two unique names for clustering.")

        print(f"Loaded {len(names):,} unique names from {args.input}.")
        run_chunk(names, args)


def run_chunk(names: Sequence[str], args: argparse.Namespace, chunk_label: Optional[str] = None) -> None:
    set_token_statistics_from_names(names)
    candidates = generate_pair_candidates(names)
    if not candidates:
        print(f"No candidate pairs generated for {chunk_label or 'full dataset'}; skipping.")
        return

    prefix_msg = f" for {chunk_label}" if chunk_label else ""
    print(
        f"Preparing to collect {args.initial_labels} initial labels{prefix_msg}. "
        "Respond with y/n/u/q at each prompt."
    )

    initial_labelled = collect_initial_labels(
        candidates,
        to_label=args.initial_labels,
        archive_path=args.save_labels,
    )
    if not initial_labelled:
        print(f"No labels collected{prefix_msg}; skipping chunk.")
        return

    model, labelled, _ = interactive_training(
        candidates,
        initial_labels=max(1, args.initial_labels // 5),
        batch_size=args.batch_size,
        archive_path=args.save_labels,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        initial_labelled=initial_labelled,
    )

    print(f"Collected {len(labelled)} labelled pairs{prefix_msg}. Building clusters ...")
    clusters = build_clusters(names, model, candidates, args.match_threshold)
    frame = clusters_to_frame(clusters)

    if args.output:
        if chunk_label:
            safe = chunk_label.replace(" ", "_").replace("'", "").replace("[", "").replace("]", "")
            chunk_output = args.output.with_stem(f"{args.output.stem}_{safe}")
        else:
            chunk_output = args.output
        chunk_output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(chunk_output, index=False)
        print(f"Clusters written to {chunk_output}")
    else:
        print(frame.head())
        print("... (use --output to save all clusters)")


if __name__ == "__main__":
    main()
