
#!/usr/bin/env python3
"""
Deduplicate Revelio companies (RCIDs).

Expected input is a company-level table with at least:
- rcid
- company name

Optional supportive fields:
- top_state (where most employees are located)
- naics_code
- lei (strong identifier; duplicates with same LEI are collapsed)
- n_users / n (used as weight to choose canonical representative)

Outputs:
- revelio_rcid_entities.csv
- revelio_firms_dedup.csv
- revelio_rcid_to_firm.csv
- revelio_dedupe_edges.csv
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))


import argparse
import os
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, Iterable, TypeVar
import random
import itertools

import pandas as pd
import pyarrow.dataset as ds
import yaml

try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback when tqdm isn't installed
    _tqdm = None

from company_name_cleaning import (
    clean_company_name,
    load_stata_alias_rules,
    normalize_state,
    normalize_naics,
    compute_token_frequencies,
    select_rare_tokens,
)
from dedupe_utils import (
    DedupeConfig,
    _duplicate_edge_for_pair,
    assign_components,
    aggregate_components,
)

_T = TypeVar("_T")


def _progress(items: Iterable[_T], *, enabled: bool, desc: str) -> Iterable[_T]:
    if not enabled or _tqdm is None:
        return items
    return _tqdm(items, desc=desc)

DEFAULTS = {
    "rcid_col": "rcid",
    "name_col": "company",
    "top_state_col": "top_state",
    "naics_col": "naics_code",
    "lei_col": "lei",
    "weight_col": "n_users",
    "dedupe_strong_name": 98.0,
    "dedupe_name_and_state": 95.0,
    "dedupe_name_and_domain": 93.0,
    "dedupe_name_and_naics2": 96.0,
    "block_prefix_len": 10,
    "max_block_size": 500,
    "rare_topn_stop": 100,
    "rare_max_tokens": 5,
    "rare_tokens_for_block": 3,
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
    """Read up to max_rows rows if specified; falls back to full read."""
    if not max_rows or max_rows <= 0:
        return _read_any(path)
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".csv", ".gz", ".bz2", ".zip"):
        return pd.read_csv(p, low_memory=False, nrows=max_rows)
    if suf in (".parquet", ".pq"):
        table = ds.dataset(p, format="parquet").to_table(limit=max_rows)
        return table.to_pandas()
    return _read_any(path)


def _block_series(df: pd.DataFrame, col: str, prefix_len: int) -> pd.Series:
    """Compute lowercase prefix block keys from a string column."""
    return (
        df[col]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.casefold()
        .str[:max(1, prefix_len)]
    )


def _sample_blocks(
    df: pd.DataFrame,
    *,
    col: str,
    block_prefixes: Optional[list[str]],
    block_count: Optional[int],
    prefix_len: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter df to rows whose block (prefix) is in block_prefixes, or sample block_count blocks.
    Returns (filtered_df, chosen_blocks).
    """
    blocks = _block_series(df, col, prefix_len)
    available = [b for b in blocks.unique().tolist() if b]

    chosen: list[str] = []
    if block_prefixes:
        chosen = [str(b).casefold()[:max(1, prefix_len)] for b in block_prefixes if str(b).strip()]
    elif block_count and block_count > 0:
        rng = random.Random(seed)
        if available:
            chosen = rng.sample(available, min(block_count, len(available)))

    if chosen:
        mask = blocks.isin(set(chosen))
        return df.loc[mask].copy(), chosen
    return df, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Optional YAML config file")
    ap.add_argument("--revelio", help="Revelio company file (csv/parquet)")
    ap.add_argument("--outdir", help="Output directory")
    ap.add_argument("--rcid-col")
    ap.add_argument("--name-col")
    ap.add_argument("--top-state-col")
    ap.add_argument("--naics-col")
    ap.add_argument("--lei-col")
    ap.add_argument("--weight-col", help="Used to choose canonical record (fallback to n)")
    ap.add_argument("--stata-rules", default=None, help="Optional path to firm_names_preclean.do")
    ap.add_argument("--use-lei", action="store_true", help="Treat identical LEI as duplicates (recommended)")
    ap.add_argument("--dedupe-strong-name", type=float, default=None)
    ap.add_argument("--dedupe-name-and-state", type=float, default=None)
    ap.add_argument("--dedupe-name-and-domain", type=float, default=None)
    ap.add_argument("--dedupe-name-and-naics2", type=float, default=None)
    ap.add_argument("--block-prefix-len", type=int, default=None)
    ap.add_argument("--max-block-size", type=int, default=None)
    ap.add_argument("--rare-topn-stop", type=int, default=None, help="Stoplist size for rare-token blocking")
    ap.add_argument("--rare-max-tokens", type=int, default=None, help="Max rare tokens per name")
    ap.add_argument("--rare-tokens-for-block", type=int, default=None, help="Use up to this many rare tokens for blocks")
    ap.add_argument("--test-limit", type=int, default=None, help="Randomly sample N rows for quick testing")
    ap.add_argument("--test-seed", type=int, default=None, help="Random seed for --test-limit")
    ap.add_argument("--test-max-scan", type=int, default=None, help="Load at most this many rows before sampling (for testing)")
    ap.add_argument("--test-blocks", type=int, default=None, help="Sample this many name-prefix blocks (overrides --test-limit sampling)")
    ap.add_argument("--test-block-prefix-len", type=int, default=None, help="Prefix length for --test-blocks sampling (default 3)")
    ap.add_argument("--test-block-prefixes", default=None, help="Comma-separated list of block prefixes to keep (case-insensitive)")
    ap.add_argument("--verbose-clean", action="store_true", help="Print sample raw -> cleaned company name mappings")
    ap.add_argument("--verbose-clean-limit", type=int, default=None, help="Max examples to print when verbose-clean is enabled")
    ap.add_argument("--verbose", action="store_true", help="Print a sample deduped firm after clustering")
    ap.add_argument("--verbose-dup", action="store_true", help="Print an example duplicate cluster (if any)")
    ap.add_argument("--verbose-dup-limit", type=int, default=None, help="Max members to show in the duplicate example")
    args = ap.parse_args()

    cfg = _load_config(args.config)

    rev_path = _expand(args.revelio) or _expand(_cfg(cfg, "paths", "revelio"))
    outdir_raw = _expand(args.outdir) or _expand(_cfg(cfg, "paths", "outdir"))
    if not rev_path or not outdir_raw:
        ap.error("Provide --revelio and --outdir, or set paths.revelio and paths.outdir in the config YAML.")

    rcid_col = args.rcid_col or _cfg(cfg, "columns", "rcid", default=DEFAULTS["rcid_col"])
    name_col = args.name_col or _cfg(cfg, "columns", "revelio_name", default=DEFAULTS["name_col"])
    top_state_col = args.top_state_col or _cfg(cfg, "columns", "top_state", default=DEFAULTS["top_state_col"])
    naics_col = args.naics_col or _cfg(cfg, "columns", "naics", default=DEFAULTS["naics_col"])
    lei_col = args.lei_col or _cfg(cfg, "columns", "lei", default=DEFAULTS["lei_col"])
    weight_col = args.weight_col or _cfg(cfg, "columns", "weight", default=DEFAULTS["weight_col"])
    stata_rules = _expand(args.stata_rules) if args.stata_rules is not None else _expand(_cfg(cfg, "paths", "stata_rules"))
    use_lei = args.use_lei or bool(_cfg(cfg, "options", "use_lei", default=False))

    dedupe_strong_name = args.dedupe_strong_name if args.dedupe_strong_name is not None else _cfg(cfg, "dedupe", "strong_name", default=DEFAULTS["dedupe_strong_name"])
    dedupe_name_and_state = args.dedupe_name_and_state if args.dedupe_name_and_state is not None else _cfg(cfg, "dedupe", "name_and_state", default=DEFAULTS["dedupe_name_and_state"])
    dedupe_name_and_domain = args.dedupe_name_and_domain if args.dedupe_name_and_domain is not None else _cfg(cfg, "dedupe", "name_and_domain", default=DEFAULTS["dedupe_name_and_domain"])
    dedupe_name_and_naics2 = args.dedupe_name_and_naics2 if args.dedupe_name_and_naics2 is not None else _cfg(cfg, "dedupe", "name_and_naics2", default=DEFAULTS["dedupe_name_and_naics2"])
    block_prefix_len = args.block_prefix_len if args.block_prefix_len is not None else _cfg(
        cfg,
        "dedupe",
        "revelio_block_prefix_len",
        default=_cfg(cfg, "dedupe", "block_prefix_len", default=DEFAULTS["block_prefix_len"]),
    )
    max_block_size = args.max_block_size if args.max_block_size is not None else _cfg(cfg, "dedupe", "max_block_size", default=DEFAULTS["max_block_size"])
    rare_topn_stop = args.rare_topn_stop if args.rare_topn_stop is not None else _cfg(cfg, "dedupe", "rare_topn_stop", default=DEFAULTS["rare_topn_stop"])
    rare_max_tokens = args.rare_max_tokens if args.rare_max_tokens is not None else _cfg(cfg, "dedupe", "rare_max_tokens", default=DEFAULTS["rare_max_tokens"])
    rare_tokens_for_block = args.rare_tokens_for_block if args.rare_tokens_for_block is not None else _cfg(cfg, "dedupe", "rare_tokens_for_block", default=DEFAULTS["rare_tokens_for_block"])
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
    if rare_topn_stop is not None:
        try:
            rare_topn_stop = int(rare_topn_stop)
        except Exception:
            rare_topn_stop = DEFAULTS["rare_topn_stop"]
    if rare_max_tokens is not None:
        try:
            rare_max_tokens = int(rare_max_tokens)
        except Exception:
            rare_max_tokens = DEFAULTS["rare_max_tokens"]
    if rare_tokens_for_block is not None:
        try:
            rare_tokens_for_block = int(rare_tokens_for_block)
        except Exception:
            rare_tokens_for_block = DEFAULTS["rare_tokens_for_block"]
    verbose_clean = args.verbose_clean or bool(_cfg(cfg, "options", "verbose_clean", default=False))
    verbose_clean_limit = args.verbose_clean_limit if args.verbose_clean_limit is not None else _cfg(cfg, "options", "verbose_clean_limit", default=10)
    verbose = args.verbose or bool(_cfg(cfg, "options", "verbose", default=False))
    verbose_dup = args.verbose_dup or bool(_cfg(cfg, "options", "verbose_dup", default=False))
    verbose_dup_limit = args.verbose_dup_limit if args.verbose_dup_limit is not None else _cfg(cfg, "options", "verbose_dup_limit", default=5)

    outdir = Path(outdir_raw)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _read_any_partial(str(rev_path), max_rows=test_max_scan).copy()
    df = df[df[rcid_col].notna()].copy()
    chosen_blocks: list[str] = []
    if test_block_prefixes or (test_blocks and test_blocks > 0):
        df, chosen_blocks = _sample_blocks(
            df,
            col=name_col,
            block_prefixes=test_block_prefixes,
            block_count=test_blocks,
            prefix_len=test_block_prefix_len or 3,
            seed=test_seed or 0,
        )
        if chosen_blocks:
            print(f"Testing: keeping {len(chosen_blocks)} name-prefix blocks (len={test_block_prefix_len}): {chosen_blocks[:10]}", flush=True)
        else:
            print("Testing: no blocks selected; using full sample", flush=True)
    if test_limit:
        df = df.sample(n=min(test_limit, len(df)), random_state=test_seed)

    # Normalize state + NAICS
    if top_state_col in df.columns:
        df["top_state_norm"] = df[top_state_col].map(lambda x: normalize_state(x, to="name"))
    else:
        df["top_state_norm"] = None

    if naics_col in df.columns:
        df["naics_norm"] = df[naics_col].map(normalize_naics)
    else:
        df["naics_norm"] = None

    if lei_col in df.columns:
        df["lei_norm"] = df[lei_col].astype("string").str.strip().replace({"": pd.NA})
    else:
        df["lei_norm"] = pd.NA

    # Name cleaning (optionally with Stata alias rules)
    stata_rules = load_stata_alias_rules(stata_rules) if stata_rules else None
    uniq = df[name_col].astype("string").fillna("").unique().tolist()
    clean_map: Dict[str, Tuple[str, str, str, str, Optional[str]]] = {}
    for s in _progress(uniq, enabled=verbose, desc="Cleaning unique Revelio names"):
        cn = clean_company_name(s, stata_rules=stata_rules)
        clean_map[str(s)] = (cn.clean, cn.stub, cn.base, cn.token_stream, cn.domain)

    if verbose_clean:
        limit = verbose_clean_limit or 2
        print(f"Sample raw -> cleaned (showing up to {limit}):")
        for raw in list(clean_map.keys())[:limit]:
            c = clean_map[raw]
            print(f"  '{raw}' -> clean='{c[0]}', stub='{c[1]}'")

    df["name_clean"] = df[name_col].astype("string").map(lambda x: clean_map.get(str(x), ("", "", "", "", None))[0])
    df["name_stub"] = df[name_col].astype("string").map(lambda x: clean_map.get(str(x), ("", "", "", "", None))[1])
    df["name_base"] = df[name_col].astype("string").map(lambda x: clean_map.get(str(x), ("", "", "", "", None))[2])
    df["token_stream"] = df[name_col].astype("string").map(lambda x: clean_map.get(str(x), ("", "", "", "", None))[3])
    df["domain"] = df[name_col].astype("string").map(lambda x: clean_map.get(str(x), ("", "", "", "", None))[4])

    # Keep only one row per RCID (if duplicates exist)
    # Prefer the one with highest weight
    weight_col = weight_col if weight_col in df.columns else ("n" if "n" in df.columns else None)
    if weight_col:
        df["_w"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
        df = df.sort_values("_w", ascending=False).drop_duplicates(rcid_col)
    else:
        df = df.drop_duplicates(rcid_col)

    df.to_csv(outdir / "revelio_rcid_entities.csv", index=False)

    # Build duplicate edges
    cfg = DedupeConfig(
        strong_name=dedupe_strong_name,
        name_and_state=dedupe_name_and_state,
        name_and_domain=dedupe_name_and_domain,
        name_and_naics2=dedupe_name_and_naics2,
        block_prefix_len=block_prefix_len,
        max_block_size=max_block_size,
    )

    df_reset = df.reset_index(drop=True)
    token_freq = compute_token_frequencies(df_reset["token_stream"].fillna("").astype(str).tolist())
    token_freq_map = {row["token"]: float(row["freq"]) for _, row in token_freq.iterrows()}

    blocks: Dict[tuple[str, str], List[int]] = {}
    name_bases = df_reset["name_base"].fillna("").astype(str).tolist()
    token_streams = df_reset["token_stream"].fillna("").astype(str).tolist()
    for idx in _progress(range(len(name_bases)), enabled=verbose, desc="Building blocks"):
        base = name_bases[idx]
        stream = token_streams[idx]
        prefix = base[:block_prefix_len]
        if not prefix:
            continue
        rare = select_rare_tokens(
            stream,
            token_freq_map,
            top_n_stop=rare_topn_stop,
            max_tokens=rare_max_tokens,
        )
        for t in rare[:rare_tokens_for_block]:
            blocks.setdefault((prefix, t), []).append(idx)

    # Score pairs block-by-block to avoid building a giant candidate_pairs set.
    edge_map: Dict[Tuple[int, int], float] = {}
    for _, idxs in blocks.items():
        if len(idxs) <= 1:
            continue
        if len(idxs) > max_block_size:
            continue
        for i, j in itertools.combinations(idxs, 2):
            if i == j:
                continue
            edge = _duplicate_edge_for_pair(
                df_reset,
                i,
                j,
                base_col="name_base",
                stub_col="name_stub",
                state_col="top_state_norm",
                domain_col="domain",
                naics_col="naics_norm",
                cfg=cfg,
            )
            if edge is None:
                continue
            a, b, sim = edge
            if a > b:
                a, b = b, a
            prev = edge_map.get((a, b))
            if prev is None or sim > prev:
                edge_map[(a, b)] = sim

    edges = [(i, j, sim) for (i, j), sim in edge_map.items()]

    # Add strong LEI edges (optional)
    if use_lei and "lei_norm" in df.columns:
        lei_groups = df.reset_index(drop=True).groupby("lei_norm")
        for lei, g in lei_groups:
            if pd.isna(lei) or len(g) <= 1:
                continue
            idxs = g.index.tolist()
            # star edges: connect all to first
            hub = idxs[0]
            for j in idxs[1:]:
                edges.append((hub, j, 100.0))

    edges_df = pd.DataFrame(edges, columns=["i", "j", "name_similarity"])
    edges_df.to_csv(outdir / "revelio_dedupe_edges.csv", index=False)

    labels = assign_components(len(df), edges)

    canon, mapping = aggregate_components(
        df.reset_index(drop=True).assign(row_index=lambda d: d.index),
        comp_labels=labels,
        id_cols_for_uid=[rcid_col],
        uid_prefix="REVFIRM",
        weight_col=weight_col,
        keep_cols=[

            name_col,

            "name_stub",

            "name_clean",

            "name_base",

            "token_stream",

            "domain",

            "top_state_norm",

            "naics_norm",

        ],

        list_cols=[rcid_col, "lei_norm"],
    )

    canon = canon.rename(columns={"uid": "revelio_firm_uid"})
    mapping = mapping.merge(df.reset_index(drop=True)[[rcid_col]], left_on="row_index", right_index=True, how="left")
    mapping = mapping.rename(columns={"uid": "revelio_firm_uid"})

    mapping[[rcid_col, "revelio_firm_uid", "component"]].to_csv(outdir / "revelio_rcid_to_firm.csv", index=False)
    canon.to_csv(outdir / "revelio_firms_dedup.csv", index=False)

    if verbose_dup:
        comp_counts = mapping.groupby("component").size()
        dup_comps = comp_counts[comp_counts > 1]
        if dup_comps.empty:
            print("No duplicate clusters found.")
        else:
            comp_id = dup_comps.index[0]
            members = mapping[mapping["component"] == comp_id]
            show_n = verbose_dup_limit or len(members)
            print(f"Example duplicate cluster component={comp_id} (showing up to {show_n} of {len(members)} members):")
            details = df_reset.loc[df_reset.index.isin(members["row_index"])]
            for _, r in details.head(show_n).iterrows():
                print(
                    {
                        "rcid": r.get(rcid_col),
                        "raw_name": r.get(name_col),
                        "name_stub": r.get("name_stub"),
                        "top_state_norm": r.get("top_state_norm"),
                        "naics_norm": r.get("naics_norm"),
                        "lei_norm": r.get("lei_norm"),
                    }
                )

    if len(df):
        # Blocking summary (prefix + rare token)
        skipped_blocks = [v for v in blocks.values() if len(v) > max_block_size]
        skipped_entities = sum(len(v) for v in skipped_blocks)
        candidate_blocks = [v for v in blocks.values() if len(v) <= max_block_size]
        total_pairs_within_blocks = sum(len(v) * (len(v) - 1) // 2 for v in candidate_blocks)

        deg = [0] * len(df)
        for a, b, _ in edges:
            deg[a] += 1
            deg[b] += 1
        nonzero = [d for d in deg if d > 0]
        avg_cand = sum(nonzero) / len(nonzero) if nonzero else 0.0
        pct_with_cands = (len(nonzero) / len(df)) * 100
        avg_cand_all = sum(deg) / len(df)
    else:
        avg_cand = 0.0
        pct_with_cands = 0.0
        avg_cand_all = 0.0
        blocks = {}
        skipped_blocks = []
        skipped_entities = 0

    print(f"Avg candidate pairs per RCID entity (with >=1): {avg_cand:.2f}")
    print(
        f"Blocks (prefix+rare token): {len(blocks):,} total; skipped {len(skipped_blocks):,} over max_block_size "
        f"(covering {skipped_entities:,} entities)"
    )
    print(f"Total pairs within blocks (pre-threshold): {total_pairs_within_blocks:,}")


if __name__ == "__main__":
    main()
