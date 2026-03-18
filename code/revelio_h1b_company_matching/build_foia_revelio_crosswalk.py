
#!/usr/bin/env python3
"""
Build a FOIA <-> Revelio company crosswalk (after within-source deduplication).

Inputs (from the dedupe scripts):
- foia_firms_dedup.csv            (from dedupe_foia_employers.py)
- revelio_firms_dedup.csv         (from dedupe_revelio_companies.py)

Optional inputs:
- foia_fein_to_firm.csv           (to produce FEIN->universal mapping)
- revelio_rcid_to_firm.csv        (to produce RCID->universal mapping)
- revelio_locations file          (to use a set of employee work states per company)

Outputs:
- crosswalk_candidates.csv        (top candidates per FOIA firm, for review)
- crosswalk_mutual_best.csv       (mutual-best matches above threshold)
- universal_companies.csv         (one row per universal entity)
- foia_firm_to_universal.csv
- revelio_firm_to_universal.csv
- (optional) foia_fein_to_universal.csv
- (optional) revelio_rcid_to_universal.csv
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))


import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import math
import random
import multiprocessing as mp

import pandas as pd
import yaml
import re

from company_name_cleaning import (
    name_similarity,
    naics_prefix_match_level,
    stable_hash_id,
    normalize_state,
)

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:
    import pyarrow as pa
except Exception:  # pragma: no cover
    pa = None

try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

LEGAL_BONUS = 2
LEGAL_PENALTY = 1

DEFAULTS = {
    "candidate_threshold": 85.0,
    "high_threshold": 92.0,
    "topk_per_foia": 5,
    "topn_stop": 100,
    "max_rare_tokens": 5,
    "tokens_for_block": 3,
}


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def _cfg(cfg: Dict[str, Any], *keys: str, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _expand(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.expanduser(os.path.expandvars(path))


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if p.suffix.lower() in (".csv", ".gz", ".bz2", ".zip"):
        return pd.read_csv(p, low_memory=False)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def _read_any_partial(path: str, max_rows: Optional[int]) -> pd.DataFrame:
    """Read up to max_rows rows for quick scans; falls back to full read."""
    if not max_rows or max_rows <= 0:
        return _read_any(path)
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".csv", ".gz", ".bz2", ".zip"):
        return pd.read_csv(p, low_memory=False, nrows=max_rows)
    # Parquet has no cheap row cap here; load then truncate for consistency
    df = _read_any(path)
    return df.head(max_rows)


def _as_set(x: Any, sep: str = ";") -> Set[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return set()
    s = str(x).strip()
    if not s:
        return set()
    return {t.strip() for t in s.split(sep) if t.strip()}


def _block_series(df: pd.DataFrame, cols: List[str], prefix_len: int) -> pd.Series:
    """Return lowercase prefix blocks from the first available column in cols."""
    prefix_len = max(1, int(prefix_len))
    col = next((c for c in cols if c in df.columns), None)
    if col is None:
        return pd.Series([""] * len(df), dtype="string")
    if df.empty:
        return pd.Series([], dtype="string")
    return (
        df[col]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.casefold()
        .str[:prefix_len]
    )


def _select_common_blocks(
    foia_blocks: pd.Series,
    rev_blocks: pd.Series,
    *,
    block_prefixes: Optional[list[str]],
    block_count: Optional[int],
    prefix_len: int,
    seed: int,
) -> list[str]:
    """Choose shared block prefixes between FOIA and Revelio."""
    if block_prefixes:
        return [str(b).casefold()[:max(1, prefix_len)] for b in block_prefixes if str(b).strip()]
    if not block_count or block_count <= 0:
        return []
    common = sorted(set(foia_blocks.unique()) & set(rev_blocks.unique()) - {""})
    if not common:
        return []
    rng = random.Random(seed)
    return rng.sample(common, min(block_count, len(common)))


def _state_support_score(foia_states: Set[str], rev_states: Set[str]) -> int:
    """
    Simple, interpretable scoring:
    +3 if FOIA HQ state in Rev states
    +2 if FOIA worksite state in Rev states
    -2 if both FOIA states present and neither appears in Rev states
    """
    if not rev_states:
        return 0
    hq = None
    work = None
    if foia_states:
        # first element is HQ, second is work in our usage; but treat generically:
        pass

    # We'll compute separately outside and pass in sets:
    return 0


_LEGAL_ENTITY_MAP = {
    "inc": "inc",
    "inc.": "inc",
    "incorporated": "inc",
    "corp": "corp",
    "corp.": "corp",
    "corporation": "corp",
    "llc": "llc",
    "l.l.c": "llc",
    "llc.": "llc",
    "ltd": "ltd",
    "ltd.": "ltd",
    "limited": "ltd",
    "llp": "llp",
    "l.l.p": "llp",
    "lp": "lp",
    "l.p": "lp",
}


def _legal_entity(raw_name: Any) -> Optional[str]:
    if raw_name is None:
        return None
    s = str(raw_name).strip()
    if not s:
        return None
    # Normalize to alnum tokens; check last token only.
    toks = re.findall(r"[A-Za-z0-9\.]+", s)
    if not toks:
        return None
    last = toks[-1].casefold().strip(".")
    return _LEGAL_ENTITY_MAP.get(last)


def score_pair(
    foia_row: pd.Series,
    rev_row: pd.Series,
    *,
    rev_state_set: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Compute match features and an overall 0-100 score.
    """
    foia_stub = str(foia_row.get("name_stub") or "")
    foia_base = str(foia_row.get("name_base") or "")
    rev_stub = str(rev_row.get("name_stub") or "")
    rev_base = str(rev_row.get("name_base") or "")

    name_sim = float(name_similarity(foia_stub, foia_base, rev_stub, rev_base))

    # Token stream subset check (case-insensitive)
    foia_tokens = {t for t in str(foia_row.get("token_stream") or "").casefold().split() if t}
    rev_tokens = {t for t in str(rev_row.get("token_stream") or "").casefold().split() if t}
    token_subset = False
    if foia_tokens and rev_tokens:
        token_subset = foia_tokens.issubset(rev_tokens) or rev_tokens.issubset(foia_tokens)

    # States
    foia_hq = (foia_row.get("hq_state") or None)
    foia_work = (foia_row.get("work_state") or None)

    rev_states = set()
    if rev_state_set is not None:
        rev_states |= {s for s in rev_state_set if s}
    else:
        top_state = rev_row.get("top_state") or None
        if isinstance(top_state, str) and top_state.strip():
            rev_states.add(top_state.strip())

    state_bonus = 0
    if rev_states:
        rev_states_cf = {str(s).strip().casefold() for s in rev_states if str(s).strip()}
        hq_ok = isinstance(foia_hq, str) and foia_hq.strip().casefold() in rev_states_cf
        work_ok = isinstance(foia_work, str) and foia_work.strip().casefold() in rev_states_cf
        if hq_ok:
            state_bonus += 3
        if work_ok:
            state_bonus += 2
        if (isinstance(foia_hq, str) and foia_hq.strip()) and (isinstance(foia_work, str) and foia_work.strip()) and (not hq_ok) and (not work_ok):
            state_bonus -= 2

    # NAICS
    na_match = naics_prefix_match_level(foia_row.get("naics"), rev_row.get("naics"))
    na_bonus = {6: 4, 4: 3, 3: 2, 2: 1}.get(na_match, 0)

    # Domain
    dom_a = foia_row.get("domain")
    dom_b = rev_row.get("domain")
    dom_match = bool(isinstance(dom_a, str) and isinstance(dom_b, str) and dom_a and dom_b and dom_a.casefold() == dom_b.casefold())
    dom_bonus = 8 if dom_match else 0

    # Total score
    score = max(0.0, min(100.0, name_sim + state_bonus + na_bonus + dom_bonus))

    # A simple match type label for auditability
    match_type = "name_only"
    if dom_match and name_sim >= 85:
        match_type = "domain+name"
    if state_bonus >= 3 and name_sim >= 90:
        match_type = "state+name"
    if na_match >= 4 and name_sim >= 90:
        match_type = "naics+name"
    if dom_match and state_bonus >= 3 and name_sim >= 85:
        match_type = "domain+state+name"

    return {
        "score": score,
        "name_sim": name_sim,
        "token_subset": token_subset,
        "state_bonus": state_bonus,
        "naics_match_level": na_match,
        "naics_bonus": na_bonus,
        "domain_match": dom_match,
        "domain_bonus": dom_bonus,
        "match_type": match_type,
    }


# -----------------------------------------------------------------------------
# Multiprocessing helpers for candidate scoring
# -----------------------------------------------------------------------------

_MP_FOIA = None
_MP_REV = None
_MP_INV = None
_MP_TOKEN_FREQ = None
_MP_STOP_SET = None
_MP_TOPN_STOP = None
_MP_MAX_RARE_TOKENS = None
_MP_TOKENS_FOR_BLOCK = None
_MP_MAX_BLOCK_SIZE = None
_MP_USE_PREFIX = None
_MP_CAND_THRESHOLD = None
_MP_VERBOSE = None
_MP_PREFIX_LABEL = None
_MP_REV_STATES = None


def _init_worker(
    foia_df: pd.DataFrame,
    rev_df: pd.DataFrame,
    inv: Dict[Any, List[int]],
    token_freq: Dict[str, float],
    stop_set: Set[str],
    topn_stop: int,
    max_rare_tokens: int,
    tokens_for_block: int,
    max_block_size: Optional[int],
    use_prefix: bool,
    candidate_threshold: float,
    verbose: bool,
    prefix_label: str,
    rev_states_by_uid: Dict[str, Set[str]],
) -> None:
    global _MP_FOIA, _MP_REV, _MP_INV, _MP_TOKEN_FREQ, _MP_TOPN_STOP, _MP_MAX_RARE_TOKENS
    global _MP_TOKENS_FOR_BLOCK, _MP_MAX_BLOCK_SIZE, _MP_USE_PREFIX, _MP_CAND_THRESHOLD, _MP_VERBOSE, _MP_STOP_SET
    global _MP_PREFIX_LABEL, _MP_REV_STATES
    _MP_FOIA = foia_df
    _MP_REV = rev_df
    _MP_INV = inv
    _MP_TOKEN_FREQ = token_freq
    _MP_STOP_SET = stop_set
    _MP_TOPN_STOP = topn_stop
    _MP_MAX_RARE_TOKENS = max_rare_tokens
    _MP_TOKENS_FOR_BLOCK = tokens_for_block
    _MP_MAX_BLOCK_SIZE = max_block_size
    _MP_USE_PREFIX = use_prefix
    _MP_CAND_THRESHOLD = candidate_threshold
    _MP_VERBOSE = verbose
    _MP_PREFIX_LABEL = prefix_label
    _MP_REV_STATES = rev_states_by_uid


def _passes_threshold(row_or_feats: Dict[str, Any] | pd.Series, threshold: float) -> bool:
    score = float(row_or_feats.get("score", 0.0))
    token_subset = bool(row_or_feats.get("token_subset", False))
    return score >= threshold or token_subset


def _build_stoplist(token_freq: Dict[str, float], top_n_stop: int) -> Set[str]:
    if not token_freq or not top_n_stop or top_n_stop <= 0:
        return set()
    top_tokens = sorted(token_freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n_stop]
    return {t for t, _ in top_tokens}


def _iter_batches_from_df(df: pd.DataFrame, cols: List[str], batch_size: int) -> Iterable[Tuple[int, List[Any]]]:
    if pa is None:
        # Fallback: yield row-wise values without .tolist()
        for i in range(len(df)):
            yield i, [df.at[i, c] if c in df.columns else None for c in cols]
        return
    table = pa.Table.from_pandas(df[cols], preserve_index=False)
    offset = 0
    for batch in table.to_batches(max_chunksize=max(1, int(batch_size))):
        n = batch.num_rows
        arrays = [batch.column(i) for i in range(len(cols))]
        for j in range(n):
            row_vals = [arr[j].as_py() if arr[j].is_valid else None for arr in arrays]
            yield offset + j, row_vals
        offset += n


def _compute_token_freq_stream(dfs: List[pd.DataFrame], col: str, batch_size: int) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total = 0
    for df in dfs:
        if col not in df.columns:
            continue
        for _idx, (val,) in _iter_batches_from_df(df, [col], batch_size):
            s = "" if val is None else str(val)
            if not s:
                continue
            for t in s.split():
                counts[t] = counts.get(t, 0) + 1
                total += 1
    if total == 0:
        return {}
    return {t: c / float(total) for t, c in counts.items()}


def _build_inv_stream(
    rev_df: pd.DataFrame,
    *,
    token_col: str,
    prefix_col: str,
    token_freq: Dict[str, float],
    stop_set: Set[str],
    max_tokens: int,
    tokens_for_block: int,
    batch_size: int,
) -> Dict[Tuple[str, str], List[int]]:
    inv: Dict[Tuple[str, str], List[int]] = {}
    if token_col not in rev_df.columns or prefix_col not in rev_df.columns:
        return inv
    for idx, (token_stream, prefix) in _iter_batches_from_df(rev_df, [token_col, prefix_col], batch_size):
        if prefix is None:
            continue
        prefix_str = str(prefix)
        if not prefix_str:
            continue
        s = "" if token_stream is None else str(token_stream)
        rare = _select_rare_tokens_fast(s, token_freq, stop_set=stop_set, max_tokens=max_tokens)
        for t in rare[:tokens_for_block]:
            inv.setdefault((prefix_str, t), []).append(idx)
    return inv


def _build_inv_no_prefix_stream(
    rev_df: pd.DataFrame,
    *,
    token_col: str,
    token_freq: Dict[str, float],
    stop_set: Set[str],
    max_tokens: int,
    tokens_for_block: int,
    batch_size: int,
) -> Dict[str, List[int]]:
    inv: Dict[str, List[int]] = {}
    if token_col not in rev_df.columns:
        return inv
    for idx, (token_stream,) in _iter_batches_from_df(rev_df, [token_col], batch_size):
        s = "" if token_stream is None else str(token_stream)
        rare = _select_rare_tokens_fast(s, token_freq, stop_set=stop_set, max_tokens=max_tokens)
        for t in rare[:tokens_for_block]:
            inv.setdefault(t, []).append(idx)
    return inv

def _select_rare_tokens_fast(
    token_stream: str,
    token_freq: Dict[str, float],
    *,
    stop_set: Set[str],
    max_tokens: int,
) -> List[str]:
    if not token_stream:
        return []
    toks = token_stream.split()
    if not toks:
        return []
    toks2 = [t for t in toks if t not in stop_set]
    if not toks2:
        return []
    toks2_sorted = sorted(toks2, key=lambda t: (token_freq.get(t, 0.0), t))
    out: List[str] = []
    seen = set()
    for t in toks2_sorted:
        if t in seen:
            continue
        out.append(t)
        seen.add(t)
        if len(out) >= max_tokens:
            break
    return out


def _score_chunk(idxs: List[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str], Set[str], List[Tuple[Any, Any, int]]]:
    cand_rows: List[Dict[str, Any]] = []
    near_below_rows: List[Dict[str, Any]] = []
    foia_with_any: Set[str] = set()
    rev_with_any: Set[str] = set()
    skipped: Dict[Tuple[Any, Any], int] = {}

    for i in idxs:
        row = _MP_FOIA.iloc[i]
        if _MP_USE_PREFIX:
            prefix = str(row.get("name_prefix") or "")
            if not prefix:
                continue
        else:
            prefix = ""
        s = str(row["token_stream"])
        rare = _select_rare_tokens_fast(
            s,
            _MP_TOKEN_FREQ,
            stop_set=_MP_STOP_SET or set(),
            max_tokens=_MP_MAX_RARE_TOKENS,
        )
        cand_idx: Set[int] = set()
        for t in rare[: _MP_TOKENS_FOR_BLOCK]:
            if _MP_USE_PREFIX:
                idxs2 = _MP_INV.get((prefix, t), [])
                if not idxs2:
                    continue
                if _MP_MAX_BLOCK_SIZE is not None and len(idxs2) > _MP_MAX_BLOCK_SIZE:
                    skipped.setdefault((prefix, t), len(idxs2))
                    continue
            else:
                idxs2 = _MP_INV.get(t, [])
                if not idxs2:
                    continue
                if _MP_MAX_BLOCK_SIZE is not None and len(idxs2) > _MP_MAX_BLOCK_SIZE:
                    skipped.setdefault(("", t), len(idxs2))
                    continue
            cand_idx.update(idxs2)

        if not cand_idx:
            continue
        foia_with_any.add(str(row["foia_firm_uid"]))

        for j in cand_idx:
            rev_row = _MP_REV.iloc[j]
            rev_uid = str(rev_row["revelio_firm_uid"])
            rev_with_any.add(rev_uid)
            state_set = _MP_REV_STATES.get(rev_uid, None)
            feats = score_pair(row, rev_row, rev_state_set=state_set)
            base_score = feats["score"]
            entry = {
                "foia_firm_uid": row["foia_firm_uid"],
                "revelio_firm_uid": rev_row["revelio_firm_uid"],
                "foia_name": row["display_name"],
                "revelio_name": rev_row["display_name"],
                **feats,
            }
            ent_foia = _legal_entity(row.get("raw_name"))
            ent_rev = _legal_entity(rev_row.get("raw_name"))
            legal_adj = 0
            if ent_foia and ent_rev:
                if ent_foia == ent_rev:
                    legal_adj = LEGAL_BONUS
                else:
                    legal_adj = -LEGAL_PENALTY
            entry["legal_entity_match"] = (ent_foia == ent_rev) if (ent_foia and ent_rev) else None
            entry["legal_entity_foia"] = ent_foia
            entry["legal_entity_revelio"] = ent_rev
            entry["legal_adjustment"] = legal_adj
            entry["score"] = max(0.0, min(100.0, base_score + legal_adj))
            if _passes_threshold({"score": base_score, "token_subset": entry.get("token_subset")}, _MP_CAND_THRESHOLD):
                cand_rows.append(entry)
            elif _MP_VERBOSE:
                entry["legal_entity_match"] = None
                entry["legal_entity_foia"] = None
                entry["legal_entity_revelio"] = None
                entry["legal_adjustment"] = 0
                near_below_rows.append(entry)

    skipped_list = [(k[0], k[1], v) for k, v in skipped.items()]
    return cand_rows, near_below_rows, foia_with_any, rev_with_any, skipped_list


def _chunk_indices(n: int, chunk_size: int) -> List[List[int]]:
    chunk_size = max(1, int(chunk_size))
    return [list(range(i, min(i + chunk_size, n))) for i in range(0, n, chunk_size)]


def _score_candidates(
    *,
    foia_df: pd.DataFrame,
    rev_df: pd.DataFrame,
    inv: Dict[Any, List[int]],
    token_freq: Dict[str, float],
    stop_set: Set[str],
    topn_stop: int,
    max_rare_tokens: int,
    tokens_for_block: int,
    max_block_size: Optional[int],
    use_prefix: bool,
    candidate_threshold: float,
    verbose: bool,
    rev_states_by_uid: Dict[str, Set[str]],
    num_workers: int,
    chunk_size: int,
    desc: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str], Set[str], Dict[Tuple[Any, Any], int]]:
    n = len(foia_df)
    if n == 0:
        return [], [], set(), set(), {}

    num_workers = max(1, int(num_workers))
    chunk_size = max(1, int(chunk_size))
    chunks = _chunk_indices(n, chunk_size)

    def _run_serial() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str], Set[str], Dict[Tuple[Any, Any], int]]:
        _init_worker(
            foia_df,
            rev_df,
            inv,
            token_freq,
            stop_set,
            topn_stop,
            max_rare_tokens,
            tokens_for_block,
            max_block_size,
            use_prefix,
            candidate_threshold,
            verbose,
            desc,
            rev_states_by_uid,
        )
        cand_rows_all: List[Dict[str, Any]] = []
        near_below_all: List[Dict[str, Any]] = []
        foia_any_all: Set[str] = set()
        rev_any_all: Set[str] = set()
        skipped_all: Dict[Tuple[Any, Any], int] = {}
        total_chunks = len(chunks)
        it = chunks
        if _tqdm is not None:
            it = _tqdm(it, total=total_chunks, desc=desc, unit="chunk")
        for idx, chunk in enumerate(it, start=1):
            if _tqdm is None and (idx == 1 or idx % 10 == 0 or idx == total_chunks):
                print(f"{desc}: processed {idx}/{total_chunks} chunks", flush=True)
            cand_rows, near_below, foia_any, rev_any, skipped = _score_chunk(chunk)
            cand_rows_all.extend(cand_rows)
            if verbose and near_below:
                near_below_all.extend(near_below)
            foia_any_all.update(foia_any)
            rev_any_all.update(rev_any)
            for p, t, s in skipped:
                skipped_all.setdefault((p, t), s)
        return cand_rows_all, near_below_all, foia_any_all, rev_any_all, skipped_all

    if num_workers <= 1:
        return _run_serial()

    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    pool = ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(
            foia_df,
            rev_df,
            inv,
            token_freq,
            stop_set,
            topn_stop,
            max_rare_tokens,
            tokens_for_block,
            max_block_size,
            use_prefix,
            candidate_threshold,
            verbose,
            desc,
            rev_states_by_uid,
        ),
    )
    cand_rows_all: List[Dict[str, Any]] = []
    near_below_all: List[Dict[str, Any]] = []
    foia_any_all: Set[str] = set()
    rev_any_all: Set[str] = set()
    skipped_all: Dict[Tuple[Any, Any], int] = {}

    try:
        total_chunks = len(chunks)
        it = pool.imap_unordered(_score_chunk, chunks)
        if _tqdm is not None:
            it = _tqdm(it, total=total_chunks, desc=desc, unit="chunk")
        for idx, (cand_rows, near_below, foia_any, rev_any, skipped) in enumerate(it, start=1):
            if _tqdm is None and (idx == 1 or idx % 10 == 0 or idx == total_chunks):
                print(f"{desc}: processed {idx}/{total_chunks} chunks", flush=True)
            cand_rows_all.extend(cand_rows)
            if verbose and near_below:
                near_below_all.extend(near_below)
            foia_any_all.update(foia_any)
            rev_any_all.update(rev_any)
            for p, t, s in skipped:
                skipped_all.setdefault((p, t), s)
    finally:
        pool.close()
        pool.join()

    return cand_rows_all, near_below_all, foia_any_all, rev_any_all, skipped_all



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Optional YAML config file")
    ap.add_argument("--foia-firms", help="foia_firms_dedup.csv")
    ap.add_argument("--revelio-firms", help="revelio_firms_dedup.csv")
    ap.add_argument("--outdir", help="Output directory")
    ap.add_argument("--foia-fein-to-firm", default=None, help="foia_fein_to_firm.csv (optional)")
    ap.add_argument("--revelio-rcid-to-firm", default=None, help="revelio_rcid_to_firm.csv (optional)")
    ap.add_argument("--revelio-locations", default=None, help="Optional file with employee work locations by rcid")
    ap.add_argument("--revelio-loc-rcid-col", default=None)
    ap.add_argument("--revelio-loc-state-col", default=None)
    ap.add_argument("--revelio-loc-weight-col", default=None, help="Optional weight column (counts)")

    ap.add_argument("--candidate-threshold", type=float, default=None, help="keep candidate pairs with score >= this")
    ap.add_argument("--high-threshold", type=float, default=None, help="mutual-best matches need score >= this")
    ap.add_argument("--topk-per-foia", type=int, default=None)
    ap.add_argument("--topn-stop", type=int, default=None)
    ap.add_argument("--max-rare-tokens", type=int, default=None)
    ap.add_argument("--tokens-for-block", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None, help="Randomly sample N rows from each input for testing")
    ap.add_argument("--test-seed", type=int, default=None, help="Random seed for --test-limit")
    ap.add_argument("--test-max-scan", type=int, default=None, help="Read at most this many rows from each input before sampling")
    ap.add_argument("--test-blocks", type=int, default=None, help="Sample this many shared name-prefix blocks across FOIA/Revelio (overrides --test-limit sampling)")
    ap.add_argument("--test-block-prefix-len", type=int, default=None, help="Prefix length for --test-blocks sampling (default 3)")
    ap.add_argument("--test-block-prefixes", default=None, help="Comma-separated list of block prefixes to keep (case-insensitive)")
    ap.add_argument("--verbose", action="store_true", help="Print sample candidate pairs and matches")
    ap.add_argument("--allow-many-rev-to-foia", action="store_true", help="Allow many Revelio firms to map to one FOIA firm (best per Revelio only)")
    ap.add_argument("--no-candidate-samples", type=int, default=None, help="Print N random FOIA firms with no candidate pairs")
    ap.add_argument("--second-pass-prefix-len", type=int, default=None, help="Optional second pass with smaller prefix length for unmatched firms")
    ap.add_argument("--third-pass-no-prefix", action="store_true", help="Optional third pass using rare tokens only (no prefix blocking) for unmatched firms")
    ap.add_argument("--third-pass-max-block-size", type=int, default=None, help="Max block size for third pass (token-only)")
    ap.add_argument("--num-workers", type=int, default=None, help="Number of worker processes for candidate scoring")
    ap.add_argument("--chunk-size", type=int, default=None, help="Rows per chunk for multiprocessing")
    args = ap.parse_args()

    cfg = _load_config(args.config)

    foia_firms = _expand(args.foia_firms) or _expand(_cfg(cfg, "paths", "foia_firms"))
    rev_firms = _expand(args.revelio_firms) or _expand(_cfg(cfg, "paths", "revelio_firms"))
    outdir_raw = _expand(args.outdir) or _expand(_cfg(cfg, "paths", "outdir"))
    foia_fein_to_firm = _expand(args.foia_fein_to_firm) or _expand(_cfg(cfg, "paths", "foia_fein_to_firm"))
    rev_rcid_to_firm = _expand(args.revelio_rcid_to_firm) or _expand(_cfg(cfg, "paths", "revelio_rcid_to_firm"))
    rev_locations = _expand(args.revelio_locations) or _expand(_cfg(cfg, "paths", "revelio_locations"))

    if not foia_firms or not rev_firms or not outdir_raw:
        ap.error("Provide --foia-firms/--revelio-firms/--outdir, or set paths.* in the config YAML.")

    loc_rcid_col = args.revelio_loc_rcid_col or _cfg(cfg, "columns", "revelio_loc_rcid", default="rcid")
    loc_state_col = args.revelio_loc_state_col or _cfg(cfg, "columns", "revelio_loc_state", default="state")
    loc_weight_col = args.revelio_loc_weight_col or _cfg(cfg, "columns", "revelio_loc_weight")

    candidate_threshold = args.candidate_threshold if args.candidate_threshold is not None else _cfg(cfg, "crosswalk", "candidate_threshold", default=DEFAULTS["candidate_threshold"])
    high_threshold = args.high_threshold if args.high_threshold is not None else _cfg(cfg, "crosswalk", "high_threshold", default=DEFAULTS["high_threshold"])
    topk_per_foia = args.topk_per_foia if args.topk_per_foia is not None else _cfg(cfg, "crosswalk", "topk_per_foia", default=DEFAULTS["topk_per_foia"])
    topn_stop = args.topn_stop if args.topn_stop is not None else _cfg(cfg, "crosswalk", "topn_stop", default=DEFAULTS["topn_stop"])
    max_rare_tokens = args.max_rare_tokens if args.max_rare_tokens is not None else _cfg(cfg, "crosswalk", "max_rare_tokens", default=DEFAULTS["max_rare_tokens"])
    tokens_for_block = args.tokens_for_block if args.tokens_for_block is not None else _cfg(cfg, "crosswalk", "tokens_for_block", default=DEFAULTS["tokens_for_block"])
    no_candidate_samples = args.no_candidate_samples if args.no_candidate_samples is not None else _cfg(cfg, "crosswalk", "no_candidate_samples", default=5)
    second_pass_prefix_len = args.second_pass_prefix_len if args.second_pass_prefix_len is not None else _cfg(cfg, "crosswalk", "second_pass_prefix_len", default=None)
    third_pass_no_prefix = bool(
        args.third_pass_no_prefix or _cfg(cfg, "crosswalk", "third_pass_no_prefix", default=False)
    )
    third_pass_max_block_size = args.third_pass_max_block_size if args.third_pass_max_block_size is not None else _cfg(
        cfg, "crosswalk", "third_pass_max_block_size", default=None
    )
    num_workers = args.num_workers if args.num_workers is not None else _cfg(cfg, "crosswalk", "num_workers", default=1)
    chunk_size = args.chunk_size if args.chunk_size is not None else _cfg(cfg, "crosswalk", "chunk_size", default=1000)
    test_limit = args.test_limit if args.test_limit is not None else _cfg(cfg, "testing", "test_limit")
    test_seed = args.test_seed if args.test_seed is not None else _cfg(cfg, "testing", "test_seed", default=0)
    test_max_scan = args.test_max_scan if args.test_max_scan is not None else _cfg(cfg, "testing", "test_max_scan")
    test_blocks = args.test_blocks if args.test_blocks is not None else _cfg(cfg, "testing", "test_blocks")
    test_block_prefix_len = args.test_block_prefix_len if args.test_block_prefix_len is not None else _cfg(cfg, "testing", "test_block_prefix_len", default=3)
    test_block_prefixes_raw = args.test_block_prefixes if args.test_block_prefixes is not None else _cfg(cfg, "testing", "test_block_prefixes")
    test_block_prefixes = None
    if isinstance(test_block_prefixes_raw, str):
        test_block_prefixes = [p.strip() for p in test_block_prefixes_raw.split(",") if p.strip()]
    elif isinstance(test_block_prefixes_raw, list):
        test_block_prefixes = [str(p).strip() for p in test_block_prefixes_raw if str(p).strip()]
    verbose_default = bool(_cfg(cfg, "crosswalk", "verbose", default=False))
    verbose = bool(args.verbose or verbose_default)
    verbose_samples = _cfg(cfg, "crosswalk", "verbose_samples", default=5)
    allow_many_rev_to_foia = bool(
        args.allow_many_rev_to_foia or _cfg(cfg, "crosswalk", "allow_many_rev_to_foia", default=False)
    )
    prefix_block_len = _cfg(cfg, "crosswalk", "prefix_block_len", default=_cfg(cfg, "dedupe", "block_prefix_len", default=10))
    prefix_max_block_size = _cfg(cfg, "crosswalk", "prefix_max_block_size", default=_cfg(cfg, "dedupe", "max_block_size", default=500))
    try:
        prefix_block_len = int(prefix_block_len)
    except Exception:
        prefix_block_len = 10
    try:
        prefix_max_block_size = int(prefix_max_block_size)
    except Exception:
        prefix_max_block_size = 500
    try:
        verbose_samples = max(1, int(verbose_samples))
    except Exception:
        verbose_samples = 5
    if test_limit is not None:
        try:
            test_limit = int(test_limit)
        except Exception:
            test_limit = None
    if test_seed is not None:
        try:
            test_seed = int(test_seed)
        except Exception:
            test_seed = 0
    if test_max_scan is not None:
        try:
            test_max_scan = int(test_max_scan)
        except Exception:
            test_max_scan = None
    if test_blocks is not None:
        try:
            test_blocks = int(test_blocks)
        except Exception:
            test_blocks = None
    if test_block_prefix_len is not None:
        try:
            test_block_prefix_len = int(test_block_prefix_len)
        except Exception:
            test_block_prefix_len = 3
    try:
        num_workers = int(num_workers) if num_workers is not None else 1
    except Exception:
        num_workers = 1
    try:
        chunk_size = int(chunk_size) if chunk_size is not None else 1000
    except Exception:
        chunk_size = 1000
    if third_pass_max_block_size is not None:
        try:
            third_pass_max_block_size = int(third_pass_max_block_size)
        except Exception:
            third_pass_max_block_size = None

    outdir = Path(outdir_raw)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading deduped FOIA and Revelio firms...", flush=True)
    foia_raw = _read_any_partial(str(foia_firms), test_max_scan)
    rev_raw = _read_any_partial(str(rev_firms), test_max_scan)
    print(f"Loaded FOIA deduped firms: {len(foia_raw):,}", flush=True)
    print(f"Loaded Revelio deduped firms: {len(rev_raw):,}", flush=True)
    chosen_blocks: list[str] = []
    if test_block_prefixes or (test_blocks and test_blocks > 0):
        foia_blocks = _block_series(
            foia_raw,
            ["canonical_name_stub", "name_stub", "name_base", "canonical_name_clean", "name_clean", "employer_name"],
            test_block_prefix_len or 3,
        )
        rev_blocks = _block_series(
            rev_raw,
            ["name_stub", "name_base", "name_clean", "company"],
            test_block_prefix_len or 3,
        )
        chosen_blocks = _select_common_blocks(
            foia_blocks,
            rev_blocks,
            block_prefixes=test_block_prefixes,
            block_count=test_blocks,
            prefix_len=test_block_prefix_len or 3,
            seed=test_seed or 0,
        )
        if chosen_blocks:
            foia_raw = foia_raw.loc[foia_blocks.isin(set(chosen_blocks))].copy()
            rev_raw = rev_raw.loc[rev_blocks.isin(set(chosen_blocks))].copy()
            print(
                f"Testing: keeping {len(chosen_blocks)} shared name-prefix blocks (len={test_block_prefix_len}); "
                f"rows -> FOIA {len(foia_raw):,}, Revelio {len(rev_raw):,}; blocks sample: {chosen_blocks[:10]}",
                flush=True,
            )
            if test_blocks and chosen_blocks:
                block = chosen_blocks[0]
                foia_block = foia_raw.loc[foia_blocks == block]
                rev_block = rev_raw.loc[rev_blocks == block]
                print(f"\nFull contents for block '{block}' (FOIA rows={len(foia_block):,}, Revelio rows={len(rev_block):,}):")
                with pd.option_context("display.max_rows", None, "display.max_colwidth", 120, "display.width", 200):
                    if not foia_block.empty:
                        print("\nFOIA block rows:")
                        foia_cols = [c for c in ["canonical_name_clean", "canonical_name_base", "hq_state_mode"] if c in foia_block.columns]
                        print(foia_block[foia_cols].to_string(index=False))
                    else:
                        print("\nFOIA block rows: (none)")
                    if not rev_block.empty:
                        print("\nRevelio block rows:")
                        rev_cols = [c for c in ["name_clean", "name_base", "top_state_norm"] if c in rev_block.columns]
                        print(rev_block[rev_cols].to_string(index=False))
                    else:
                        print("\nRevelio block rows: (none)")
        else:
            print("Testing: no shared blocks found with requested settings; using full sample", flush=True)
    if test_limit:
        if test_blocks:
            print(
                f"Sampling {test_limit} rows from each input after block filtering (seed={test_seed})...",
                flush=True,
            )
        else:
            print(f"Sampling {test_limit} rows from each input (seed={test_seed})...", flush=True)
        foia_raw = foia_raw.sample(n=min(test_limit, len(foia_raw)), random_state=test_seed)
        rev_raw = rev_raw.sample(n=min(test_limit, len(rev_raw)), random_state=test_seed)

    # Standardize column names expected here
    if "revelio_firm_uid" not in rev_raw.columns:
        # Fallback for non-deduped Revelio inputs (e.g., revelio_rcid_entities.csv)
        if "rcid" in rev_raw.columns:
            rev_raw = rev_raw.copy()
            rev_raw["revelio_firm_uid"] = rev_raw["rcid"]
        else:
            raise ValueError(
                "Revelio input missing 'revelio_firm_uid' and no 'rcid' column found for fallback. "
                "Provide deduped Revelio firms or a Revelio file with rcid."
            )
    foia = pd.DataFrame({
        "foia_firm_uid": foia_raw["foia_firm_uid"],
        "name_stub": foia_raw.get("canonical_name_stub", foia_raw.get("name_stub")),
        "name_base": foia_raw.get("canonical_name_base", foia_raw.get("name_base")),
        "token_stream": foia_raw.get("token_stream", ""),
        "hq_state": foia_raw.get("hq_state_mode"),
        "work_state": foia_raw.get("work_state_mode"),
        "naics": foia_raw.get("naics_mode"),
        "domain": foia_raw.get("canonical_domain", foia_raw.get("domain")),
        "raw_name": foia_raw.get("raw_name_example", foia_raw.get("canonical_name_clean", foia_raw.get("canonical_name_stub", foia_raw.get("name_stub")))),
        "display_name": foia_raw.get("canonical_name_clean", foia_raw.get("canonical_name_stub", foia_raw.get("name_stub"))),
    }).copy()

    rev = pd.DataFrame({
        "revelio_firm_uid": rev_raw["revelio_firm_uid"],
        "name_stub": rev_raw.get("name_stub"),
        "name_base": rev_raw.get("name_base"),
        "token_stream": rev_raw.get("token_stream", ""),
        "top_state": rev_raw.get("top_state_norm"),
        "naics": rev_raw.get("naics_norm"),
        "domain": rev_raw.get("domain"),
        "raw_name": rev_raw.get("company", rev_raw.get("name_clean", rev_raw.get("name_stub"))),
        "display_name": rev_raw.get("company", rev_raw.get("name_clean", rev_raw.get("name_stub"))),
    }).copy()

    foia = foia.fillna("")
    rev = rev.fillna("")
    foia["name_prefix"] = _block_series(foia, ["name_base", "name_stub"], prefix_block_len)
    rev["name_prefix"] = _block_series(rev, ["name_base", "name_stub"], prefix_block_len)
    foia_name_by_uid = dict(zip(foia["foia_firm_uid"].astype(str), foia["display_name"].astype(str)))
    rev_name_by_uid = dict(zip(rev["revelio_firm_uid"].astype(str), rev["display_name"].astype(str)))

    # Optional Revelio state sets from locations file
    rev_states_by_uid: Dict[str, Set[str]] = {}
    if rev_locations and rev_rcid_to_firm:
        print("Loading Revelio locations for state sets...", flush=True)
    if rev_locations and rev_rcid_to_firm:
        loc = _read_any_partial(rev_locations, test_max_scan)
        rcid_map = _read_any(rev_rcid_to_firm)
        if test_limit:
            loc = loc.sample(n=min(test_limit, len(loc)), random_state=test_seed)
        # rcid_map columns: rcid, revelio_firm_uid
        rcid_col = loc_rcid_col
        if rcid_col not in loc.columns:
            raise ValueError(f"--revelio-loc-rcid-col {rcid_col} not found in locations file")
        if loc_state_col not in loc.columns:
            raise ValueError(f"--revelio-loc-state-col {loc_state_col} not found in locations file")

        loc = loc.merge(rcid_map, left_on=rcid_col, right_on=rcid_map.columns[0], how="left")
        loc = loc[loc["revelio_firm_uid"].notna()].copy()

        # Normalize: one set per firm uid; optionally keep only top-N by weight
        if loc_weight_col and loc_weight_col in loc.columns:
            loc["_w"] = pd.to_numeric(loc[loc_weight_col], errors="coerce").fillna(0.0)
        else:
            loc["_w"] = 1.0

        # For each firm, take states with positive weight; keep top 10
        for uid, g in loc.groupby("revelio_firm_uid"):
            g2 = g.groupby(loc_state_col, as_index=False)["_w"].sum().sort_values("_w", ascending=False)
            states = [normalize_state(s, to="name") for s in g2[loc_state_col]]
            states = [s for s in states if isinstance(s, str) and s.strip()]
            rev_states_by_uid[str(uid)] = set(states[:10])

    # Build rare-token blocks
    print("Building rare-token index...", flush=True)
    token_freq = _compute_token_freq_stream([foia, rev], "token_stream", batch_size=chunk_size)
    stop_set = _build_stoplist(token_freq, topn_stop)

    inv = _build_inv_stream(
        rev,
        token_col="token_stream",
        prefix_col="name_prefix",
        token_freq=token_freq,
        stop_set=stop_set,
        max_tokens=max_rare_tokens,
        tokens_for_block=tokens_for_block,
        batch_size=chunk_size,
    )

    # Generate candidates and score (pass 1)
    print("Scoring candidate pairs...", flush=True)
    cand_rows, near_below_rows, foia_with_any_candidates, rev_with_any_candidates, skipped_blocks_pass1 = _score_candidates(
        foia_df=foia,
        rev_df=rev,
        inv=inv,
        token_freq=token_freq,
        stop_set=stop_set,
        topn_stop=topn_stop,
        max_rare_tokens=max_rare_tokens,
        tokens_for_block=tokens_for_block,
        max_block_size=prefix_max_block_size,
        use_prefix=True,
        candidate_threshold=candidate_threshold,
        verbose=verbose,
        rev_states_by_uid=rev_states_by_uid,
        num_workers=num_workers,
        chunk_size=chunk_size,
        desc="Pass 1",
    )
    if verbose and skipped_blocks_pass1:
        for (prefix, token), size in skipped_blocks_pass1.items():
            print(
                f"Skipping block (pass 1) prefix='{prefix}' token='{token}' size={size} > max={prefix_max_block_size}"
            )

    def _cand_df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(
            rows,
            columns=[
                "foia_firm_uid",
                "revelio_firm_uid",
                "foia_name",
                "revelio_name",
                "score",
                "name_sim",
                "token_subset",
                "state_bonus",
                "naics_match_level",
                "naics_bonus",
                "domain_match",
                "domain_bonus",
                "legal_entity_match",
                "legal_entity_foia",
                "legal_entity_revelio",
                "legal_adjustment",
                "match_type",
            ],
        )

    def _print_cand_stats(label: str, cand_df: pd.DataFrame, foia_count: int) -> None:
        print(f"{label} candidate pairs kept: {len(cand_df):,}", flush=True)
        if foia_count:
            avg_all = len(cand_df) / foia_count
        else:
            avg_all = 0.0
        if not cand_df.empty:
            avg_nonzero = cand_df.groupby("foia_firm_uid").size().mean()
        else:
            avg_nonzero = 0.0
        print(f"{label} avg candidate pairs per FOIA firm (all): {avg_all:.2f}")
        print(f"{label} avg candidate pairs per FOIA firm (with >=1): {avg_nonzero:.2f}")

    def _print_candidate_samples(label: str, cand_df: pd.DataFrame, near_below: List[Dict[str, Any]]) -> None:
        if not verbose:
            return
        sample_cols = ["foia_firm_uid", "revelio_firm_uid", "foia_name", "revelio_name", "score", "match_type"]
        above = cand_df.sort_values("score", ascending=True).head(verbose_samples)
        below_df = pd.DataFrame(near_below)
        below = below_df.sort_values("score", ascending=False).head(verbose_samples) if not below_df.empty else below_df
        print(f"\n{label} sample candidate pairs around cutoff (threshold={candidate_threshold}):")
        with pd.option_context("display.max_colwidth", 80, "display.width", 200):
            if not above.empty:
                print("\nAbove cutoff (lowest scores kept):")
                print(above[sample_cols].to_string(index=False))
            else:
                print("\nAbove cutoff (lowest scores kept): (none)")
            if not below.empty:
                print("\nBelow cutoff (highest scores dropped):")
                print(below[sample_cols].to_string(index=False))
            else:
                print("\nBelow cutoff (highest scores dropped): (none)")

    def _print_mutual_samples(label: str, mutual_df: pd.DataFrame, mutual_all_df: pd.DataFrame) -> None:
        if not verbose:
            return
        mb_cols = ["foia_firm_uid", "revelio_firm_uid", "foia_name", "revelio_name", "score", "match_type"]
        above_mb = mutual_df.sort_values("score", ascending=True).head(verbose_samples)
        below_mb = mutual_all_df[mutual_all_df["score"] < high_threshold].sort_values("score", ascending=False).head(verbose_samples) if not mutual_all_df.empty else mutual_all_df
        print(f"\n{label} sample matches around cutoff (threshold={high_threshold}):")
        if above_mb.empty:
            print("  Above cutoff (lowest scores kept): (none)")
        else:
            with pd.option_context("display.max_colwidth", 80, "display.width", 200):
                print(above_mb[mb_cols].to_string(index=False))
        if below_mb.empty:
            print("  Below cutoff (highest scores dropped): (none)")
        else:
            with pd.option_context("display.max_colwidth", 80, "display.width", 200):
                print(below_mb[mb_cols].to_string(index=False))

    def _print_new_candidate_samples(label: str, cand_df: pd.DataFrame, seed: Optional[int]) -> None:
        if not verbose:
            return
        if cand_df.empty:
            print(f"\n{label} new candidate pairs: (none)")
            return
        sample_cols = ["foia_firm_uid", "revelio_firm_uid", "foia_name", "revelio_name", "score", "match_type"]
        rng = random.Random(seed or 0)
        k = min(verbose_samples, len(cand_df))
        sample_idx = rng.sample(range(len(cand_df)), k)
        samp = cand_df.iloc[sample_idx]
        print(f"\n{label} new candidate pairs (random sample, n={k}):")
        with pd.option_context("display.max_colwidth", 80, "display.width", 200):
            print(samp[sample_cols].to_string(index=False))

    cand = _cand_df_from_rows(cand_rows)
    _print_cand_stats("Pass 1", cand, len(foia))
    _print_candidate_samples("Pass 1", cand, near_below_rows)
    cand_pass1 = cand.copy()
    near_below_rows_pass1 = list(near_below_rows)
    foia_with_any_candidates_pass1 = set(foia_with_any_candidates)
    pass2_ran = False
    pass3_ran = False
    cand_pass2 = cand.copy()
    def _compute_mutual_best(cand_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if cand_df.empty:
            return cand_df.copy(), cand_df.copy()
        best_rev_local = cand_df.sort_values("score", ascending=False).drop_duplicates("revelio_firm_uid")
        if allow_many_rev_to_foia:
            mutual_local = best_rev_local.copy()
            mutual_all_local = mutual_local.copy()
            if not mutual_local.empty:
                mutual_local = mutual_local[
                    (mutual_local["score"] >= high_threshold) | (mutual_local["token_subset"].fillna(False))
                ].copy()
            return mutual_local, mutual_all_local
        best_foia_local = cand_df.sort_values("score", ascending=False).drop_duplicates("foia_firm_uid")
        mutual_local = best_foia_local.merge(
            best_rev_local[["revelio_firm_uid", "foia_firm_uid"]].rename(
                columns={"foia_firm_uid": "foia_firm_uid_bestrev"}
            ),
            on="revelio_firm_uid",
            how="left",
        )
        mutual_local = mutual_local[mutual_local["foia_firm_uid"] == mutual_local["foia_firm_uid_bestrev"]].copy()
        mutual_all_local = mutual_local.copy()
        if not mutual_local.empty:
            mutual_local = mutual_local[
                (mutual_local["score"] >= high_threshold) | (mutual_local["token_subset"].fillna(False))
            ].copy()
        mutual_local = mutual_local.drop(columns=["foia_firm_uid_bestrev"])
        return mutual_local, mutual_all_local

    # Optional second pass with smaller prefix for unmatched firms
    if second_pass_prefix_len is not None:
        try:
            second_pass_prefix_len = int(second_pass_prefix_len)
        except Exception:
            second_pass_prefix_len = None
    if second_pass_prefix_len is not None and second_pass_prefix_len > 0 and second_pass_prefix_len < prefix_block_len:
        mutual_pass1, _mutual_all_pass1 = _compute_mutual_best(cand)
        foia_matched = set(mutual_pass1["foia_firm_uid"].astype(str)) if not mutual_pass1.empty else set()
        rev_matched = set(mutual_pass1["revelio_firm_uid"].astype(str)) if not mutual_pass1.empty else set()
        foia_unmatched = foia[~foia["foia_firm_uid"].astype(str).isin(foia_matched)].copy()
        rev_unmatched = rev[~rev["revelio_firm_uid"].astype(str).isin(rev_matched)].copy()
        if not foia_unmatched.empty and not rev_unmatched.empty:
            pass2_ran = True
            print(
                f"*****Second pass: prefix_len={second_pass_prefix_len}, FOIA={len(foia_unmatched):,}, Revelio={len(rev_unmatched):,}*****",
                flush=True,
            )
            foia_unmatched["name_prefix"] = _block_series(foia_unmatched, ["name_base", "name_stub"], second_pass_prefix_len)
            rev_unmatched["name_prefix"] = _block_series(rev_unmatched, ["name_base", "name_stub"], second_pass_prefix_len)
            print("Building rare-token index (second pass)...", flush=True)
            inv2 = _build_inv_stream(
                rev_unmatched,
                token_col="token_stream",
                prefix_col="name_prefix",
                token_freq=token_freq,
                stop_set=stop_set,
                max_tokens=max_rare_tokens,
                tokens_for_block=tokens_for_block,
                batch_size=chunk_size,
            )
            print("Scoring candidate pairs (second pass)...", flush=True)
            pass2_rows, near_below_rows_pass2, foia_any2, rev_any2, skipped_blocks_pass2 = _score_candidates(
                foia_df=foia_unmatched,
                rev_df=rev_unmatched,
                inv=inv2,
                token_freq=token_freq,
                stop_set=stop_set,
                topn_stop=topn_stop,
                max_rare_tokens=max_rare_tokens,
                tokens_for_block=tokens_for_block,
                max_block_size=prefix_max_block_size,
                use_prefix=True,
                candidate_threshold=candidate_threshold,
                verbose=verbose,
                rev_states_by_uid=rev_states_by_uid,
                num_workers=num_workers,
                chunk_size=chunk_size,
                desc="Pass 2",
            )
            if verbose and skipped_blocks_pass2:
                for (prefix, token), size in skipped_blocks_pass2.items():
                    print(
                        f"Skipping block (pass 2) prefix='{prefix}' token='{token}' size={size} > max={prefix_max_block_size}"
                    )
            cand_rows.extend(pass2_rows)
            foia_with_any_candidates.update(foia_any2)
            rev_with_any_candidates.update(rev_any2)

            cand = _cand_df_from_rows(cand_rows)
            pass2_added = len(pass2_rows)
            if pass2_added > 0:
                pass2_df = _cand_df_from_rows(pass2_rows)
            else:
                pass2_df = cand.head(0)
            _print_cand_stats("Pass 2 (added)", pass2_df, len(foia_unmatched))
            _print_new_candidate_samples("Pass 2", pass2_df, test_seed)
            cand_pass2 = cand.copy()
        else:
            cand_pass2 = cand.copy()

    # Optional third pass with no prefix (rare tokens only) for unmatched firms
    if third_pass_no_prefix:
        mutual_pass2, _mutual_all_pass2 = _compute_mutual_best(cand)
        foia_matched = set(mutual_pass2["foia_firm_uid"].astype(str)) if not mutual_pass2.empty else set()
        rev_matched = set(mutual_pass2["revelio_firm_uid"].astype(str)) if not mutual_pass2.empty else set()
        foia_unmatched = foia[~foia["foia_firm_uid"].astype(str).isin(foia_matched)].copy()
        rev_unmatched = rev[~rev["revelio_firm_uid"].astype(str).isin(rev_matched)].copy()
        if not foia_unmatched.empty and not rev_unmatched.empty:
            pass3_ran = True
            print(
                f"Third pass (no prefix): FOIA={len(foia_unmatched):,}, Revelio={len(rev_unmatched):,}",
                flush=True,
            )
            print("Building rare-token index (third pass)...", flush=True)
            inv3 = _build_inv_no_prefix_stream(
                rev_unmatched,
                token_col="token_stream",
                token_freq=token_freq,
                stop_set=stop_set,
                max_tokens=max_rare_tokens,
                tokens_for_block=tokens_for_block,
                batch_size=chunk_size,
            )

            print("Scoring candidate pairs (third pass)...", flush=True)
            pass3_rows, near_below_rows_pass3, foia_any3, rev_any3, skipped_blocks_pass3 = _score_candidates(
                foia_df=foia_unmatched,
                rev_df=rev_unmatched,
                inv=inv3,
                token_freq=token_freq,
                stop_set=stop_set,
                topn_stop=topn_stop,
                max_rare_tokens=max_rare_tokens,
                tokens_for_block=tokens_for_block,
                max_block_size=third_pass_max_block_size,
                use_prefix=False,
                candidate_threshold=candidate_threshold,
                verbose=verbose,
                rev_states_by_uid=rev_states_by_uid,
                num_workers=num_workers,
                chunk_size=chunk_size,
                desc="Pass 3",
            )
            if verbose and skipped_blocks_pass3:
                for (_prefix, token), size in skipped_blocks_pass3.items():
                    max_val = third_pass_max_block_size if third_pass_max_block_size is not None else "None"
                    print(f"Skipping block (pass 3) token='{token}' size={size} > max={max_val}")
            cand_rows.extend(pass3_rows)
            foia_with_any_candidates.update(foia_any3)
            rev_with_any_candidates.update(rev_any3)

            cand = _cand_df_from_rows(cand_rows)
            pass3_added = len(pass3_rows)
            if pass3_added > 0:
                pass3_df = _cand_df_from_rows(pass3_rows)
            else:
                pass3_df = cand.head(0)
            _print_cand_stats("Pass 3 (added)", pass3_df, len(foia_unmatched))
            _print_new_candidate_samples("Pass 3", pass3_df, test_seed)
    # (Stats already printed per pass.)
    if cand.empty:
        # Still write empty outputs for predictable downstream behavior
        (outdir / "crosswalk_candidates.csv").write_text("foia_firm_uid,revelio_firm_uid,score\n")
        (outdir / "crosswalk_mutual_best.csv").write_text("foia_firm_uid,revelio_firm_uid,score\n")
        print("No candidate matches found at the requested threshold.")
        return
    # Rank candidates per FOIA
    print("Ranking candidates and writing outputs...", flush=True)
    cand["rank_foia"] = cand.groupby("foia_firm_uid")["score"].rank(method="first", ascending=False)
    cand_top = cand[cand["rank_foia"] <= topk_per_foia].copy()
    cand_top = cand_top.sort_values(["foia_firm_uid", "rank_foia", "score"], ascending=[True, True, False])
    cand_top.to_csv(outdir / "crosswalk_candidates.csv", index=False)

    # Best matches: either mutual-best (default) or best per Revelio (many-to-one)
    mutual, mutual_all = _compute_mutual_best(cand)
    mutual.to_csv(outdir / "crosswalk_mutual_best.csv", index=False)
    match_mode_label = "best-per-revelio" if allow_many_rev_to_foia else "mutual-best"
    mutual_pass1, mutual_all_pass1 = _compute_mutual_best(cand_pass1)
    mutual_pass2, mutual_all_pass2 = _compute_mutual_best(cand_pass2)
    _print_mutual_samples(f"All matches ({match_mode_label})", mutual, mutual_all)

    # Build universal IDs via connected components of the mutual-best graph
    nodes_foia = [f"F:{x}" for x in foia["foia_firm_uid"].astype(str)]
    nodes_rev = [f"R:{x}" for x in rev["revelio_firm_uid"].astype(str)]

    if nx is None:
        # Fallback: assign universal IDs from mutual edges only (no component merging)
        # Each matched pair becomes its own universal entity; unmatched are singleton entities.
        univ_rows = []
        foia_to_univ = {}
        rev_to_univ = {}

        for _, r in mutual.iterrows():
            parts = [str(r["foia_firm_uid"]), str(r["revelio_firm_uid"])]
            uid = stable_hash_id("UNIV", parts)
            univ_rows.append({
                "universal_uid": uid,
                "foia_firm_uids": str(r["foia_firm_uid"]),
                "revelio_firm_uids": str(r["revelio_firm_uid"]),
                "representative_name": str(r["revelio_name"]) if str(r["revelio_name"]).strip() else str(r["foia_name"]),
                "min_score": float(r["score"]),
                "match_type": str(r["match_type"]),
            })
            foia_to_univ[str(r["foia_firm_uid"])] = uid
            rev_to_univ[str(r["revelio_firm_uid"])] = uid

        # Singletons
        for x in foia["foia_firm_uid"].astype(str):
            if x in foia_to_univ:
                continue
            uid = stable_hash_id("UNIV", [f"FOIA:{x}"])
            foia_to_univ[x] = uid
            univ_rows.append({
                "universal_uid": uid,
                "foia_firm_uids": x,
                "revelio_firm_uids": "",
                "representative_name": str(foia.loc[foia["foia_firm_uid"] == x, "display_name"].iloc[0]),
                "min_score": None,
                "match_type": "unmatched_foia",
            })
        for x in rev["revelio_firm_uid"].astype(str):
            if x in rev_to_univ:
                continue
            uid = stable_hash_id("UNIV", [f"REV:{x}"])
            rev_to_univ[x] = uid
            univ_rows.append({
                "universal_uid": uid,
                "foia_firm_uids": "",
                "revelio_firm_uids": x,
                "representative_name": str(rev.loc[rev["revelio_firm_uid"] == x, "display_name"].iloc[0]),
                "min_score": None,
                "match_type": "unmatched_revelio",
            })

        universal = pd.DataFrame(univ_rows)
    else:
        G = nx.Graph()
        G.add_nodes_from(nodes_foia)
        G.add_nodes_from(nodes_rev)

        for _, r in mutual.iterrows():
            G.add_edge(f"F:{r['foia_firm_uid']}", f"R:{r['revelio_firm_uid']}", score=float(r["score"]))

        components = list(nx.connected_components(G))

        # Map node -> universal uid
        node_to_univ: Dict[str, str] = {}
        univ_rows = []
        for comp in components:
            foia_ids = sorted([n[2:] for n in comp if n.startswith("F:")])
            rev_ids = sorted([n[2:] for n in comp if n.startswith("R:")])

            uid = stable_hash_id("UNIV", foia_ids + rev_ids)

            # Component score summary: minimum edge score within the component
            min_score = None
            edge_scores = []
            for u, v, d in G.subgraph(comp).edges(data=True):
                edge_scores.append(d.get("score"))
            if edge_scores:
                min_score = float(min(edge_scores))

            # Representative name: prefer Revelio if present
            rep_name = None
            if rev_ids:
                rep_name = rev_name_by_uid.get(rev_ids[0])
            elif foia_ids:
                rep_name = foia_name_by_uid.get(foia_ids[0])

            univ_rows.append({
                "universal_uid": uid,
                "foia_firm_uids": ";".join(foia_ids),
                "revelio_firm_uids": ";".join(rev_ids),
                "representative_name": rep_name,
                "min_score": min_score,
                "match_type": "matched" if (foia_ids and rev_ids) else ("unmatched_foia" if foia_ids else "unmatched_revelio"),
            })

            for n in comp:
                node_to_univ[n] = uid

        universal = pd.DataFrame(univ_rows)

        foia_to_univ = {fid: node_to_univ.get(f"F:{fid}") for fid in foia["foia_firm_uid"].astype(str)}
        rev_to_univ = {rid: node_to_univ.get(f"R:{rid}") for rid in rev["revelio_firm_uid"].astype(str)}

        # Sanity: any missing mappings are singletons not in graph (shouldn't happen since we added all nodes)
        for fid, uid in list(foia_to_univ.items()):
            if uid is None:
                uid2 = stable_hash_id("UNIV", [f"FOIA:{fid}"])
                foia_to_univ[fid] = uid2
                universal = pd.concat([universal, pd.DataFrame([{
                    "universal_uid": uid2,
                    "foia_firm_uids": fid,
                    "revelio_firm_uids": "",
                    "representative_name": str(foia.loc[foia["foia_firm_uid"] == fid, "display_name"].iloc[0]),
                    "min_score": None,
                    "match_type": "unmatched_foia",
                }])], ignore_index=True)
        for rid, uid in list(rev_to_univ.items()):
            if uid is None:
                uid2 = stable_hash_id("UNIV", [f"REV:{rid}"])
                rev_to_univ[rid] = uid2
                universal = pd.concat([universal, pd.DataFrame([{
                    "universal_uid": uid2,
                    "foia_firm_uids": "",
                    "revelio_firm_uids": rid,
                    "representative_name": str(rev.loc[rev["revelio_firm_uid"] == rid, "display_name"].iloc[0]),
                    "min_score": None,
                    "match_type": "unmatched_revelio",
                }])], ignore_index=True)

    universal.to_csv(outdir / "universal_companies.csv", index=False)

    pd.DataFrame({"foia_firm_uid": list(foia_to_univ.keys()), "universal_uid": list(foia_to_univ.values())}).to_csv(
        outdir / "foia_firm_to_universal.csv", index=False
    )
    pd.DataFrame({"revelio_firm_uid": list(rev_to_univ.keys()), "universal_uid": list(rev_to_univ.values())}).to_csv(
        outdir / "revelio_firm_to_universal.csv", index=False
    )

    # Optional original-ID to universal mappings
    if foia_fein_to_firm:
        fein_map = _read_any(foia_fein_to_firm)
        fein_map["universal_uid"] = fein_map["foia_firm_uid"].astype(str).map(foia_to_univ)
        fein_map[["fein_clean", "foia_firm_uid", "universal_uid"]].to_csv(outdir / "foia_fein_to_universal.csv", index=False)

    if rev_rcid_to_firm:
        rcid_map = _read_any(rev_rcid_to_firm)
        rcid_map["universal_uid"] = rcid_map["revelio_firm_uid"].astype(str).map(rev_to_univ)
        # rcid col name is first column in mapping file
        rcid_col = [c for c in rcid_map.columns if c not in ("revelio_firm_uid", "universal_uid", "component", "row_index")][0]
        rcid_map[[rcid_col, "revelio_firm_uid", "universal_uid"]].to_csv(outdir / "revelio_rcid_to_universal.csv", index=False)

    foia_total = len(foia)
    success_pass1 = len(set(mutual_pass1["foia_firm_uid"].astype(str))) if not mutual_pass1.empty else 0
    success_pass2 = len(set(mutual_pass2["foia_firm_uid"].astype(str))) if pass2_ran and not mutual_pass2.empty else 0
    success_pass3 = len(set(mutual["foia_firm_uid"].astype(str))) if pass3_ran and not mutual.empty else 0
    print("FOIA match breakdown:")
    print(f"  total FOIA firms: {foia_total:,}")
    foia_with_any_cand_pass1 = foia_with_any_candidates_pass1
    foia_no_cand_pass1 = foia_total - len(foia_with_any_cand_pass1)
    foia_with_cand_above_pass1 = set(cand_pass1["foia_firm_uid"].astype(str)) if not cand_pass1.empty else set()
    foia_no_cand_above_pass1 = foia_total - len(foia_with_cand_above_pass1)
    foia_with_mutual_pass1 = set(mutual_pass1["foia_firm_uid"].astype(str)) if not mutual_pass1.empty else set()
    foia_no_mutual_above_pass1 = foia_total - len(foia_with_mutual_pass1)
    print(f"  no candidate pairs (pass 1): {foia_no_cand_pass1:,}")
    print(f"  no candidates above threshold (pass 1): {foia_no_cand_above_pass1:,}")
    print(f"  no mutual-best above high threshold (pass 1): {foia_no_mutual_above_pass1:,}")
    print(f"  successful matches (pass 1): {success_pass1:,}")
    print(f"  successful matches (pass 2): {success_pass2:,}")
    print(f"  successful matches (pass 3): {success_pass3:,}")

    # Sample FOIA firms with no candidate pairs (final)
    try:
        n_sample = int(no_candidate_samples) if no_candidate_samples is not None else 0
    except Exception:
        n_sample = 0
    if n_sample > 0:
        rng = random.Random(test_seed or 0)
        foia_no_cand_final = [
            uid for uid in foia["foia_firm_uid"].astype(str)
            if uid not in foia_with_any_candidates
        ]
        if foia_no_cand_final:
            k = min(n_sample, len(foia_no_cand_final))
            sample_uids = rng.sample(foia_no_cand_final, k)
            print(f"\nSample FOIA firms with no candidate pairs (n={k}):")
            for uid in sample_uids:
                name = foia_name_by_uid.get(uid, "")
                print({"foia_firm_uid": uid, "display_name": name})


if __name__ == "__main__":
    main()
