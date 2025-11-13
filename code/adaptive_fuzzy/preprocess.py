"""Streaming name normalization and enrichment utilities for large datasets."""

from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CORPORATE_SUFFIXES = [
    "llc",
    "l.l.c",
    "ltd",
    "limited",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "gmbh",
    "s.a",
    "s.a.",
    "sa",
    "plc",
    "ag",
    "pte",
    "pty",
    "bv",
    "sarl",
    "oy",
]
_CORPORATE_PATTERN = re.compile(
    r"(?:[,/\s]+(?:{suffix})(?:\.)?)$".format(
        suffix="|".join(re.escape(sfx) for sfx in sorted(CORPORATE_SUFFIXES, key=len, reverse=True))
    ),
    flags=re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^\w\s]+")
_ALNUM_TOKEN_RE = re.compile(r"[0-9a-z]+")


@dataclass(frozen=True)
class NormalizedName:
    """Container for enriched name features emitted by the preprocessing stage."""

    record_id: int
    raw: str
    normalized: str
    tokens: List[str]
    shingles: List[str]
    alnum: str
    length: int
    first_token: str
    last_token: str


def _strip_corporate_suffix(value: str) -> str:
    working = value
    while True:
        match = _CORPORATE_PATTERN.search(working)
        if not match:
            break
        working = working[: match.start()].rstrip(" ,/-")
    return working


def _normalize_text_lower(value: str) -> str:
    nfkd = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    lowered = stripped.lower()
    lowered = _strip_corporate_suffix(lowered)
    lowered = _NON_ALNUM_RE.sub(" ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _extract_features(raw_name: str) -> Optional[NormalizedName]:
    clean = raw_name.strip()
    if not clean:
        return None
    normalized = _normalize_text_lower(clean)
    if not normalized:
        return None
    tokens = sorted({token for token in _ALNUM_TOKEN_RE.findall(normalized)})
    token_sequence = _ALNUM_TOKEN_RE.findall(normalized)
    alnum = "".join(token_sequence)
    length = len(alnum)
    first_token = token_sequence[0] if token_sequence else ""
    last_token = token_sequence[-1] if token_sequence else ""
    if length >= 3:
        shingles = sorted({alnum[idx : idx + 3] for idx in range(length - 2)})
    else:
        shingles = []
    return NormalizedName(
        record_id=-1,
        raw=clean,
        normalized=normalized,
        tokens=tokens,
        shingles=shingles,
        alnum=alnum,
        length=length,
        first_token=first_token,
        last_token=last_token,
    )


def _iter_source_batches(
    path: Path,
    column: Optional[str],
    chunk_size: int,
) -> Iterator[pd.Series]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        reader = pd.read_csv(path, usecols=[column] if column else None, chunksize=chunk_size)
        for chunk in reader:
            if column is None:
                if len(chunk.columns) != 1:
                    raise ValueError("Input has multiple columns; specify --column.")
                yield chunk.iloc[:, 0]
            else:
                yield chunk[column]
        return
    if suffix in {".parquet", ".pq"}:
        pq_file = pq.ParquetFile(path)
        cols: Optional[List[str]] = [column] if column else None
        for batch in pq_file.iter_batches(columns=cols, batch_size=chunk_size):
            table = batch.to_pandas()
            if column is None:
                if len(table.columns) != 1:
                    raise ValueError("Input has multiple columns; specify --column.")
                yield table.iloc[:, 0]
            else:
                yield table[column]
        return
    raise ValueError(f"Unsupported input extension {path.suffix}. Use csv, txt, parquet, or pq.")


def prepare_normalized_name_parquet(
    input_path: Path,
    column: Optional[str],
    output_path: Path,
    chunk_size: int = 50_000,
    limit: Optional[int] = None,
) -> int:
    """Stream raw institution names into an enriched parquet cache."""

    print(
        f"[normalize] Building normalized-name cache at {output_path} "
        f"(limit={limit if limit is not None else 'unbounded'}, chunk_size={chunk_size:,})"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    total_written = 0
    record_id = 0
    chunk_index = 0
    start_time = time.time()
    try:
        for series in _iter_source_batches(input_path, column, chunk_size):
            chunk_index += 1
            normalized_records: List[NormalizedName] = []
            for value in series.astype(str):
                if limit is not None and record_id >= limit:
                    break
                features = _extract_features(value)
                if features is None:
                    record_id += 1
                    continue
                normalized_records.append(
                    NormalizedName(
                        record_id=record_id,
                        raw=features.raw,
                        normalized=features.normalized,
                        tokens=features.tokens,
                        shingles=features.shingles,
                        alnum=features.alnum,
                        length=features.length,
                        first_token=features.first_token,
                        last_token=features.last_token,
                    )
                )
                record_id += 1
            if not normalized_records:
                if limit is not None and record_id >= limit:
                    print(
                        f"[normalize] Reached record limit after chunk {chunk_index}; stopping.",
                        flush=True,
                    )
                    break
                continue
            table = _normalized_records_to_table(normalized_records)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            total_written += table.num_rows
            elapsed = time.time() - start_time
            print(
                f"[normalize] Chunk {chunk_index}: wrote {table.num_rows:,} rows "
                f"(cumulative {total_written:,}) in {elapsed:.1f}s",
                flush=True,
            )
            if limit is not None and record_id >= limit:
                print(f"[normalize] Hit requested limit of {limit:,} rows; stopping.", flush=True)
                break
    finally:
        if writer is not None:
            writer.close()
    total_elapsed = time.time() - start_time
    print(
        f"[normalize] Finished building cache ({total_written:,} rows) in {total_elapsed:.1f}s.",
        flush=True,
    )
    return total_written


def _normalized_records_to_table(records: Sequence[NormalizedName]) -> pa.Table:
    ids = [rec.record_id for rec in records]
    raw_names = [rec.raw for rec in records]
    normalized = [rec.normalized for rec in records]
    tokens = [rec.tokens for rec in records]
    shingles = [rec.shingles for rec in records]
    alnum = [rec.alnum for rec in records]
    lengths = [rec.length for rec in records]
    first_tokens = [rec.first_token for rec in records]
    last_tokens = [rec.last_token for rec in records]

    data = {
        "id": pa.array(ids, type=pa.int64()),
        "name": pa.array(raw_names, type=pa.string()),
        "norm": pa.array(normalized, type=pa.string()),
        "tokens": pa.array(tokens, type=pa.list_(pa.string())),
        "shingles": pa.array(shingles, type=pa.list_(pa.string())),
        "alnum": pa.array(alnum, type=pa.string()),
        "length": pa.array(lengths, type=pa.int32()),
        "first_token": pa.array(first_tokens, type=pa.string()),
        "last_token": pa.array(last_tokens, type=pa.string()),
    }
    return pa.Table.from_pydict(data)


__all__ = [
    "prepare_normalized_name_parquet",
    "NormalizedName",
]
