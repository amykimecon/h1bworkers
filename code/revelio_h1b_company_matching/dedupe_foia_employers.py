
#!/usr/bin/env python3
"""
Deduplicate FOIA H-1B employers.

Input can be application-level or already-aggregated. At minimum you need:
- FEIN (or equivalent employer identifier)
- employer name

Optional supportive fields:
- HQ state (FOIA employer HQ state)
- worksite state
- NAICS
- multi-registration indicator
- selection indicator

Outputs:
- foia_fein_entities.csv       (one row per FEIN)
- foia_firms_dedup.csv         (deduped firm clusters across FEINs; conservative)
- foia_fein_to_firm.csv        (mapping FEIN -> firm uid)
- foia_dedupe_edges.csv        (debug: duplicate edges + similarity)
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
import multiprocessing as mp
import itertools

import pandas as pd
import pyarrow.dataset as ds
try:
    import pyarrow as pa
except Exception:  # pragma: no cover
    pa = None
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


def _iter_batches_from_df(df: pd.DataFrame, cols: list[str], batch_size: int) -> Iterable[Tuple[int, list[Any]]]:
    if pa is None:
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


_MP_DF = None
_MP_CFG = None
_MP_COLS = None


def _init_worker(df: pd.DataFrame, cfg: DedupeConfig, cols: dict) -> None:
    global _MP_DF, _MP_CFG, _MP_COLS
    _MP_DF = df
    _MP_CFG = cfg
    _MP_COLS = cols


def _score_block_chunk(blocks: list[list[int]]) -> list[tuple[int, int, float]]:
    edges: list[tuple[int, int, float]] = []
    for idxs in blocks:
        if len(idxs) <= 1:
            continue
        for i, j in itertools.combinations(idxs, 2):
            if i == j:
                continue
            edge = _duplicate_edge_for_pair(
                _MP_DF,
                i,
                j,
                base_col=_MP_COLS["base_col"],
                stub_col=_MP_COLS["stub_col"],
                state_col=_MP_COLS["state_col"],
                domain_col=_MP_COLS["domain_col"],
                naics_col=_MP_COLS["naics_col"],
                cfg=_MP_CFG,
            )
            if edge is not None:
                edges.append(edge)
    return edges


def _chunk_blocks(blocks: list[list[int]], chunk_size: int) -> list[list[list[int]]]:
    chunk_size = max(1, int(chunk_size))
    return [blocks[i:i + chunk_size] for i in range(0, len(blocks), chunk_size)]

# Defaults used when neither CLI args nor config provide values
DEFAULTS = {
    "fein_col": "FEIN",
    "name_col": "employer_name",
    "year_col": None,
    "hq_state_col": "state",
    "work_state_col": "WORKSITE_STATE",
    "naics_col": "NAICS_CODE",
    "multireg_col": "ben_multi_reg_ind",
    "selected_col": "status_type",
    "selected_value": "SELECTED",
    "dedupe_strong_name": 98.0,
    "dedupe_name_and_state": 95.0,
    "dedupe_name_and_domain": 93.0,
    "dedupe_name_and_naics2": 96.0,
    "block_prefix_len": 10,
    "max_block_size": 500,
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


def _clean_fein(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    # FOIA sometimes redacts FEIN with text like "(b)(3) (b)(6) (b)(7)(c)"
    if "(" in s and ")" in s:
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if digits else None


def _weighted_mode(values: pd.Series, weights: pd.Series) -> Optional[str]:
    if values.empty:
        return None
    tmp = pd.DataFrame({"v": values.astype("string"), "w": pd.to_numeric(weights, errors="coerce").fillna(0.0)})
    tmp = tmp.dropna(subset=["v"])
    if tmp.empty:
        return None
    sums = tmp.groupby("v")["w"].sum().sort_values(ascending=False)
    return str(sums.index[0]) if len(sums) else None


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
    available = [b for b in blocks.unique() if b]

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


def build_fein_table(
    df: pd.DataFrame,
    *,
    fein_col: str,
    name_col: str,
    year_col: Optional[str] = None,
    hq_state_col: Optional[str] = None,
    work_state_col: Optional[str] = None,
    naics_col: Optional[str] = None,
    multireg_col: Optional[str] = None,
    selected_col: Optional[str] = None,
    selected_value: str = "SELECTED",
    stata_rules_path: Optional[str] = None,
    verbose_clean: bool = False,
    verbose_clean_limit: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    df["fein_clean"] = df[fein_col].map(_clean_fein)
    if year_col and year_col in df.columns:
        df["fein_year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    else:
        df["fein_year"] = pd.NA
    key_cols = ["fein_clean"] + (["fein_year"] if year_col else [])
    df = df[df["fein_clean"].notna()].copy()

    # valid app indicator
    if multireg_col and multireg_col in df.columns:
        df["valid_app"] = pd.to_numeric(df[multireg_col], errors="coerce").fillna(0).astype(int).map(lambda x: 1 if x == 0 else 0)
    else:
        df["valid_app"] = 1

    # selected indicator
    if selected_col and selected_col in df.columns:
        df["_sel_raw"] = df[selected_col].astype("string")
        df["selected"] = df["_sel_raw"].str.casefold().eq(str(selected_value).casefold()).astype(int) * df["valid_app"]
    else:
        df["selected"] = 0

    # normalize states
    if hq_state_col and hq_state_col in df.columns:
        df["hq_state"] = df[hq_state_col].map(lambda x: normalize_state(x, to="name"))
    else:
        df["hq_state"] = None

    if work_state_col and work_state_col in df.columns:
        df["work_state"] = df[work_state_col].map(lambda x: normalize_state(x, to="name"))
    else:
        df["work_state"] = None

    # normalize NAICS
    if naics_col and naics_col in df.columns:
        df["naics"] = df[naics_col].map(normalize_naics)
    else:
        df["naics"] = None

    # name cleaning (optionally with Stata alias rules)
    stata_rules = load_stata_alias_rules(stata_rules_path) if stata_rules_path else None

    uniq = list(df[name_col].astype("string").fillna("").unique())
    clean_map: Dict[str, Tuple[str, str, str, str, Optional[str]]] = {}
    for s in _progress(uniq, enabled=verbose, desc="Cleaning unique FOIA names"):
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

    # Aggregations: counts
    fein_counts = df.groupby(key_cols, as_index=False).agg(
        n_apps=("valid_app", "sum"),
        n_success=("selected", "sum"),
        n_rows=("fein_clean", "size"),
    )

    # Top name variant per FEIN (weighted by valid_app)
    name_counts = (
        df.groupby(key_cols + ["name_stub"], as_index=False)["valid_app"]
        .sum()
        .rename(columns={"valid_app": "w"})
        .sort_values(key_cols + ["w"], ascending=[True] * len(key_cols) + [False])
    )
    top_name = name_counts.drop_duplicates(key_cols)[key_cols + ["name_stub"]].rename(columns={"name_stub": "canonical_name_stub"})

    # Top domain per FEIN
    dom_counts = (
        df[df["domain"].notna()]
        .groupby(key_cols + ["domain"], as_index=False)["valid_app"]
        .sum()
        .rename(columns={"valid_app": "w"})
        .sort_values(key_cols + ["w"], ascending=[True] * len(key_cols) + [False])
    )
    top_dom = dom_counts.drop_duplicates(key_cols)[key_cols + ["domain"]].rename(columns={"domain": "canonical_domain"})

    # HQ state mode per FEIN
    hq_counts = (
        df[df["hq_state"].notna()]
        .groupby(key_cols + ["hq_state"], as_index=False)["valid_app"]
        .sum()
        .rename(columns={"valid_app": "w"})
        .sort_values(key_cols + ["w"], ascending=[True] * len(key_cols) + [False])
    )
    top_hq = hq_counts.drop_duplicates(key_cols)[key_cols + ["hq_state"]].rename(columns={"hq_state": "hq_state_mode"})

    # Worksite state mode per FEIN (weighted by selected if available, else valid_app)
    w_weight = df["selected"] if df["selected"].sum() > 0 else df["valid_app"]
    work_tmp = df.copy()
    work_tmp["_w_work"] = w_weight
    work_counts = (
        work_tmp[work_tmp["work_state"].notna()]
        .groupby(key_cols + ["work_state"], as_index=False)["_w_work"]
        .sum()
        .rename(columns={"_w_work": "w"})
        .sort_values(key_cols + ["w"], ascending=[True] * len(key_cols) + [False])
    )
    top_work = work_counts.drop_duplicates(key_cols)[key_cols + ["work_state"]].rename(columns={"work_state": "work_state_mode"})

    # NAICS mode per FEIN (weighted by selected if available, else valid_app)
    na_tmp = df.copy()
    na_tmp["_w_na"] = w_weight
    na_counts = (
        na_tmp[na_tmp["naics"].notna()]
        .groupby(key_cols + ["naics"], as_index=False)["_w_na"]
        .sum()
        .rename(columns={"_w_na": "w"})
        .sort_values(key_cols + ["w"], ascending=[True] * len(key_cols) + [False])
    )
    top_na = na_counts.drop_duplicates(key_cols)[key_cols + ["naics"]].rename(columns={"naics": "naics_mode"})

    # Representative base/clean/token_stream for the chosen canonical_name_stub
    # (We pick the first row within the FEIN where name_stub == canonical_name_stub.)
    rep = (
        df[key_cols + [name_col, "name_stub", "name_clean", "name_base", "token_stream"]]
        .merge(top_name, on=key_cols, how="left")
    )
    rep = rep[rep["name_stub"] == rep["canonical_name_stub"]].drop_duplicates(key_cols)
    rep = rep[key_cols + [name_col, "name_clean", "name_base", "token_stream"]].rename(
        columns={
            name_col: "raw_name_example",
            "name_clean": "canonical_name_clean",
            "name_base": "canonical_name_base",
        }
    )

    fein_tbl = (
        fein_counts
        .merge(top_name, on=key_cols, how="left")
        .merge(rep, on=key_cols, how="left")
        .merge(top_dom, on=key_cols, how="left")
        .merge(top_hq, on=key_cols, how="left")
        .merge(top_work, on=key_cols, how="left")
        .merge(top_na, on=key_cols, how="left")
    )

    # Add a convenient "name_*" alias expected by downstream utilities
    fein_tbl["name_stub"] = fein_tbl["canonical_name_stub"]
    fein_tbl["name_base"] = fein_tbl["canonical_name_base"]
    fein_tbl["token_stream"] = fein_tbl["token_stream"].fillna("")

    fein_tbl = fein_tbl.reset_index(drop=True)
    return fein_tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Optional YAML config file")
    ap.add_argument("--foia", help="FOIA raw/aggregated file (csv/parquet)")
    ap.add_argument("--outdir", help="Output directory")
    ap.add_argument("--fein-col")
    ap.add_argument("--name-col")
    ap.add_argument("--hq-state-col")
    ap.add_argument("--year-col")
    ap.add_argument("--work-state-col")
    ap.add_argument("--naics-col")
    ap.add_argument("--multireg-col")
    ap.add_argument("--selected-col")
    ap.add_argument("--selected-value")
    ap.add_argument("--stata-rules", default=None, help="Optional path to firm_names_preclean.do")
    ap.add_argument("--dedupe-strong-name", type=float, default=None)
    ap.add_argument("--dedupe-name-and-state", type=float, default=None)
    ap.add_argument("--dedupe-name-and-domain", type=float, default=None)
    ap.add_argument("--dedupe-name-and-naics2", type=float, default=None)
    ap.add_argument("--block-prefix-len", type=int, default=None)
    ap.add_argument("--max-block-size", type=int, default=None)
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
    ap.add_argument("--num-workers", type=int, default=None, help="Number of worker processes for scoring pairs")
    ap.add_argument("--block-chunk-size", type=int, default=None, help="Blocks per chunk for multiprocessing")
    args = ap.parse_args()

    cfg = _load_config(args.config)

    foia_path = _expand(args.foia) or _expand(_cfg(cfg, "paths", "foia"))
    outdir_raw = _expand(args.outdir) or _expand(_cfg(cfg, "paths", "outdir"))
    if not foia_path or not outdir_raw:
        ap.error("Provide --foia and --outdir, or set paths.foia and paths.outdir in the config YAML.")

    fein_col = args.fein_col or _cfg(cfg, "columns", "fein", default=DEFAULTS["fein_col"])
    name_col = args.name_col or _cfg(cfg, "columns", "foia_name", default=DEFAULTS["name_col"])
    year_col = args.year_col or _cfg(cfg, "columns", "year", default=DEFAULTS["year_col"])
    hq_state_col = args.hq_state_col or _cfg(cfg, "columns", "hq_state", default=DEFAULTS["hq_state_col"])
    work_state_col = args.work_state_col or _cfg(cfg, "columns", "work_state", default=DEFAULTS["work_state_col"])
    naics_col = args.naics_col or _cfg(cfg, "columns", "naics", default=DEFAULTS["naics_col"])
    multireg_col = args.multireg_col or _cfg(cfg, "columns", "multireg", default=DEFAULTS["multireg_col"])
    selected_col = args.selected_col or _cfg(cfg, "columns", "selected", default=DEFAULTS["selected_col"])
    selected_value = args.selected_value or _cfg(cfg, "columns", "selected_value", default=DEFAULTS["selected_value"])
    stata_rules = _expand(args.stata_rules) if args.stata_rules is not None else _expand(_cfg(cfg, "paths", "stata_rules"))

    dedupe_strong_name = args.dedupe_strong_name if args.dedupe_strong_name is not None else _cfg(cfg, "dedupe", "strong_name", default=DEFAULTS["dedupe_strong_name"])
    dedupe_name_and_state = args.dedupe_name_and_state if args.dedupe_name_and_state is not None else _cfg(cfg, "dedupe", "name_and_state", default=DEFAULTS["dedupe_name_and_state"])
    dedupe_name_and_domain = args.dedupe_name_and_domain if args.dedupe_name_and_domain is not None else _cfg(cfg, "dedupe", "name_and_domain", default=DEFAULTS["dedupe_name_and_domain"])
    dedupe_name_and_naics2 = args.dedupe_name_and_naics2 if args.dedupe_name_and_naics2 is not None else _cfg(cfg, "dedupe", "name_and_naics2", default=DEFAULTS["dedupe_name_and_naics2"])
    block_prefix_len = args.block_prefix_len if args.block_prefix_len is not None else _cfg(
        cfg,
        "dedupe",
        "foia_block_prefix_len",
        default=_cfg(cfg, "dedupe", "block_prefix_len", default=DEFAULTS["block_prefix_len"]),
    )
    max_block_size = args.max_block_size if args.max_block_size is not None else _cfg(cfg, "dedupe", "max_block_size", default=DEFAULTS["max_block_size"])
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
    verbose_clean = args.verbose_clean or bool(_cfg(cfg, "options", "verbose_clean", default=False))
    verbose_clean_limit = args.verbose_clean_limit if args.verbose_clean_limit is not None else _cfg(cfg, "options", "verbose_clean_limit", default=10)
    verbose = args.verbose or bool(_cfg(cfg, "options", "verbose", default=False))
    verbose_dup = args.verbose_dup or bool(_cfg(cfg, "options", "verbose_dup", default=False))
    verbose_dup_limit = args.verbose_dup_limit if args.verbose_dup_limit is not None else _cfg(cfg, "options", "verbose_dup_limit", default=5)
    num_workers = args.num_workers if args.num_workers is not None else _cfg(cfg, "dedupe", "num_workers", default=1)
    block_chunk_size = args.block_chunk_size if args.block_chunk_size is not None else _cfg(cfg, "dedupe", "block_chunk_size", default=200)
    try:
        num_workers = int(num_workers) if num_workers is not None else 1
    except Exception:
        num_workers = 1
    try:
        block_chunk_size = int(block_chunk_size) if block_chunk_size is not None else 200
    except Exception:
        block_chunk_size = 200

    outdir = Path(outdir_raw)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _read_any_partial(str(foia_path), max_rows=test_max_scan)
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
    fein_tbl = build_fein_table(
        df,
        fein_col=fein_col,
        name_col=name_col,
        year_col=year_col,
        hq_state_col=hq_state_col,
        work_state_col=work_state_col,
        naics_col=naics_col,
        multireg_col=multireg_col,
        selected_col=selected_col,
        selected_value=selected_value,
        stata_rules_path=stata_rules,
        verbose_clean=verbose_clean,
        verbose_clean_limit=verbose_clean_limit,
        verbose=verbose,
    )

    fein_tbl.to_csv(outdir / "foia_fein_entities.csv", index=False)

    # Deduplicate across FEINs (firm-level clusters)
    cfg = DedupeConfig(
        strong_name=dedupe_strong_name,
        name_and_state=dedupe_name_and_state,
        name_and_domain=dedupe_name_and_domain,
        name_and_naics2=dedupe_name_and_naics2,
        block_prefix_len=block_prefix_len,
        max_block_size=max_block_size,
    )

    # Blocking summary to help tune prefix/size
    blocks: Dict[str, List[int]] = {}
    for idx, (b,) in _iter_batches_from_df(fein_tbl, ["name_base"], batch_size=block_chunk_size):
        s = "" if b is None else str(b)
        blocks.setdefault(s[:block_prefix_len], []).append(idx)
    skipped_blocks = [v for v in blocks.values() if len(v) > max_block_size]
    skipped_entities = sum(len(v) for v in skipped_blocks)
    candidate_blocks = [v for v in blocks.values() if len(v) <= max_block_size]
    total_pairs_within_blocks = sum(len(v) * (len(v) - 1) // 2 for v in candidate_blocks)
    print(
        f"Blocks: {len(blocks):,} total; skipped {len(skipped_blocks):,} over max_block_size "
        f"(covering {skipped_entities:,} entities)"
    )
    print(f"Total pairs within blocks (pre-threshold): {total_pairs_within_blocks:,}")

    blocks_to_score = [idxs for idxs in blocks.values() if 1 < len(idxs) <= max_block_size]
    cols = {
        "base_col": "name_base",
        "stub_col": "name_stub",
        "state_col": "hq_state_mode",
        "domain_col": "canonical_domain",
        "naics_col": "naics_mode",
    }
    edges: List[Tuple[int, int, float]] = []
    if blocks_to_score:
        chunks = _chunk_blocks(blocks_to_score, block_chunk_size)
        if num_workers <= 1:
            _init_worker(fein_tbl, cfg, cols)
            it = chunks
            if _tqdm is not None:
                it = _tqdm(it, total=len(chunks), desc="Scoring blocks", unit="chunk")
            for chunk in it:
                edges.extend(_score_block_chunk(chunk))
        else:
            ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
            pool = ctx.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(fein_tbl, cfg, cols),
            )
            try:
                it = pool.imap_unordered(_score_block_chunk, chunks)
                if _tqdm is not None:
                    it = _tqdm(it, total=len(chunks), desc="Scoring blocks", unit="chunk")
                for part in it:
                    edges.extend(part)
            finally:
                pool.close()
                pool.join()
    edges_df = pd.DataFrame(edges, columns=["i", "j", "name_similarity"])
    edges_df.to_csv(outdir / "foia_dedupe_edges.csv", index=False)

    if len(fein_tbl):
        deg = [0] * len(fein_tbl)
        for a, b, _ in edges:
            deg[a] += 1
            deg[b] += 1
        nonzero = [d for d in deg if d > 0]
        avg_cand_nonzero = sum(nonzero) / len(nonzero) if nonzero else 0.0
        pct_with_cands = (len(nonzero) / len(fein_tbl)) * 100
        avg_cand_all = sum(deg) / len(fein_tbl)
    else:
        avg_cand_nonzero = 0.0
        pct_with_cands = 0.0
        avg_cand_all = 0.0
    print(f"Avg candidate pairs per FEIN entity (with >=1): {avg_cand_nonzero:.2f}")

    labels = assign_components(len(fein_tbl), edges)
    canon, mapping = aggregate_components(
        fein_tbl.assign(row_index=fein_tbl.index),
        comp_labels=labels,
        id_cols_for_uid=["fein_clean"],
        uid_prefix="FOIAFIRM",
        weight_col="n_success",
        keep_cols=[

            "canonical_name_stub",

            "canonical_name_clean",

            "canonical_name_base",

            "token_stream",

            "hq_state_mode",

            "work_state_mode",

            "naics_mode",

            "canonical_domain",

            "raw_name_example",

        ],

        list_cols=["fein_clean"],
    )

    canon = canon.rename(columns={"uid": "foia_firm_uid"})
    mapping_cols = ["fein_clean"]
    if "fein_year" in fein_tbl.columns:
        mapping_cols.append("fein_year")
    mapping = mapping.merge(fein_tbl[mapping_cols], left_on="row_index", right_index=True, how="left")
    mapping = mapping.rename(columns={"uid": "foia_firm_uid"})
    mapping[mapping_cols + ["foia_firm_uid", "component"]].to_csv(outdir / "foia_fein_to_firm.csv", index=False)
    canon.to_csv(outdir / "foia_firms_dedup.csv", index=False)

    # Keep output focused on blocking stats + optional duplicate example.

    if verbose_dup:
        comp_counts = mapping.groupby("component").size()
        comp_feins = mapping.groupby("component")["fein_clean"].nunique()
        dup_comps = comp_counts[comp_counts > 1]
        multi_fein_comps = comp_feins[comp_feins > 1]
        if dup_comps.empty:
            print("No duplicate clusters found.")
        else:
            # Prefer a component with multiple FEINs; fallback to any duplicate component
            if not multi_fein_comps.empty:
                comp_id = multi_fein_comps.index[0]
            else:
                comp_id = dup_comps.index[0]
            members = mapping[mapping["component"] == comp_id]
            show_n = verbose_dup_limit or len(members)
            print(f"Example duplicate cluster component={comp_id} (showing up to {show_n} of {len(members)} members):")
            # join to fein_tbl to show canonical names/states
            details = fein_tbl.loc[fein_tbl.index.isin(members["row_index"])]
            for _, r in details.head(show_n).iterrows():
                print(
                    {
                        "fein_clean": r.get("fein_clean"),
                        "raw_name": r.get("raw_name_example"),
                        "raw_stub": r.get("canonical_name_stub"),
                        "hq_state_mode": r.get("hq_state_mode"),
                        "work_state_mode": r.get("work_state_mode"),
                        "naics_mode": r.get("naics_mode"),
                    }
                )


if __name__ == "__main__":
    main()
