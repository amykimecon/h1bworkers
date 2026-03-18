
"""
Deduplication utilities for company/entity tables.

The core idea:
- Create normalized name forms (stub/base/token_stream)
- Use blocking keys to generate candidate duplicate pairs
- Score pairs using name similarity + supportive fields (state/NAICS/domain)
- Build connected components (union-find) from "duplicate edges"
- Aggregate each component to a canonical record + mapping table

This module is intentionally conservative to avoid over-collapsing subsidiaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable, Sequence, Set
import itertools
import math

import pandas as pd

from company_name_cleaning import name_similarity, naics_prefix_match_level, stable_hash_id


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def components(self) -> Dict[int, List[int]]:
        comps: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            comps.setdefault(r, []).append(i)
        return comps


@dataclass
class DedupeConfig:
    """
    Thresholds for deduplication (within one dataset).

    - strong_name: if name similarity >= strong_name, merge (almost always)
    - name_and_state: if name similarity >= name_and_state and state matches, merge
    - name_and_domain: if name similarity >= name_and_domain and domain matches, merge
    - name_and_naics2: if name similarity >= name_and_naics2 and NAICS2 matches, merge

    `block_prefix_len`: name_base prefix used for blocking comparisons
    `max_block_size`: skip blocks above this size to avoid O(n^2) blowups
    """
    strong_name: float = 98.0
    name_and_state: float = 95.0
    name_and_domain: float = 93.0
    name_and_naics2: float = 96.0

    block_prefix_len: int = 10
    max_block_size: int = 500


def _state_match(a: Any, b: Any) -> Optional[bool]:
    if a is None or (isinstance(a, float) and pd.isna(a)) or str(a).strip() == "":
        return None
    if b is None or (isinstance(b, float) and pd.isna(b)) or str(b).strip() == "":
        return None
    return str(a).strip().casefold() == str(b).strip().casefold()


def _duplicate_edge_for_pair(
    df: pd.DataFrame,
    i: int,
    j: int,
    *,
    base_col: str,
    stub_col: str,
    state_col: Optional[str] = None,
    domain_col: Optional[str] = None,
    naics_col: Optional[str] = None,
    cfg: DedupeConfig = DedupeConfig(),
) -> Optional[Tuple[int, int, float]]:
    a_base = str(df.at[i, base_col] or "")
    b_base = str(df.at[j, base_col] or "")
    if not a_base or not b_base:
        return None

    a_stub = str(df.at[i, stub_col] or "")
    b_stub = str(df.at[j, stub_col] or "")
    sim = name_similarity(a_stub, a_base, b_stub, b_base)

    if sim >= cfg.strong_name:
        return (i, j, sim)

    # supportive fields
    st_ok = None
    if state_col:
        st_ok = _state_match(df.at[i, state_col], df.at[j, state_col])

    dom_ok = None
    if domain_col:
        da = df.at[i, domain_col]
        db = df.at[j, domain_col]
        if isinstance(da, str) and isinstance(db, str) and da and db:
            dom_ok = da.casefold() == db.casefold()

    na_ok = None
    if naics_col:
        na_ok = naics_prefix_match_level(df.at[i, naics_col], df.at[j, naics_col]) >= 2

    # Conservative merge rules
    if st_ok is True and sim >= cfg.name_and_state:
        return (i, j, sim)
    if dom_ok is True and sim >= cfg.name_and_domain:
        return (i, j, sim)
    if na_ok is True and sim >= cfg.name_and_naics2:
        return (i, j, sim)

    return None


def find_duplicate_edges(
    df: pd.DataFrame,
    *,
    base_col: str,
    stub_col: str,
    state_col: Optional[str] = None,
    domain_col: Optional[str] = None,
    naics_col: Optional[str] = None,
    cfg: DedupeConfig = DedupeConfig(),
) -> List[Tuple[int, int, float]]:
    """
    Returns list of (i, j, sim_score) for record indices in df considered duplicates.
    """
    # Build blocks by prefix of base string (fast, deterministic)
    base_series = df[base_col].fillna("").astype(str)
    blocks: Dict[str, List[int]] = {}
    for idx, b in enumerate(base_series.tolist()):
        key = b[: cfg.block_prefix_len]
        blocks.setdefault(key, []).append(idx)

    edges: List[Tuple[int, int, float]] = []

    for _, idxs in blocks.items():
        if len(idxs) <= 1:
            continue
        if len(idxs) > cfg.max_block_size:
            # Block is too big; skip to avoid blowups (user can adjust config)
            continue

        for i, j in itertools.combinations(idxs, 2):
            edge = _duplicate_edge_for_pair(
                df,
                i,
                j,
                base_col=base_col,
                stub_col=stub_col,
                state_col=state_col,
                domain_col=domain_col,
                naics_col=naics_col,
                cfg=cfg,
            )
            if edge is not None:
                edges.append(edge)

    return edges


def find_duplicate_edges_from_pairs(
    df: pd.DataFrame,
    pairs: Iterable[Tuple[int, int]],
    *,
    base_col: str,
    stub_col: str,
    state_col: Optional[str] = None,
    domain_col: Optional[str] = None,
    naics_col: Optional[str] = None,
    cfg: DedupeConfig = DedupeConfig(),
) -> List[Tuple[int, int, float]]:
    """
    Score candidate pairs and return only those that satisfy duplicate rules.
    """
    edges: List[Tuple[int, int, float]] = []
    for i, j in pairs:
        edge = _duplicate_edge_for_pair(
            df,
            i,
            j,
            base_col=base_col,
            stub_col=stub_col,
            state_col=state_col,
            domain_col=domain_col,
            naics_col=naics_col,
            cfg=cfg,
        )
        if edge is not None:
            edges.append(edge)
    return edges


def assign_components(
    n: int,
    edges: Iterable[Tuple[int, int, float]],
) -> List[int]:
    """
    Given n nodes and edges, return component label per node (0..k-1).
    """
    uf = UnionFind(n)
    for i, j, _ in edges:
        uf.union(i, j)
    comps = uf.components()
    # map root -> compact label
    root_to_label = {root: k for k, root in enumerate(sorted(comps.keys()))}
    labels = [root_to_label[uf.find(i)] for i in range(n)]
    return labels


def aggregate_components(
    df: pd.DataFrame,
    *,
    comp_labels: Sequence[int],
    id_cols_for_uid: List[str],
    uid_prefix: str,
    weight_col: Optional[str] = None,
    keep_cols: Optional[List[str]] = None,
    list_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate df rows by component labels into canonical rows.

    Returns:
      canon_df: one row per component with uid_prefix-based stable hash ID
      map_df: mapping from original row index to canonical uid

    Parameters
    ----------
    id_cols_for_uid:
      Columns used to build a stable hash for the canonical UID. We recommend
      using a stable identifier (e.g., FEIN, RCID) and taking the *sorted list*
      of IDs in the component.

    weight_col:
      Column used to pick the representative row (max weight).

    keep_cols:
      Columns copied from the representative row.

    list_cols:
      Columns that will be aggregated into a ';'-joined list of unique values.
    """
    df = df.copy()
    df["_comp"] = list(comp_labels)

    if weight_col and weight_col in df.columns:
        df["_weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    else:
        df["_weight"] = 0.0

    keep_cols = keep_cols or []
    list_cols = list_cols or []

    canon_rows = []
    map_rows = []

    for comp, g in df.groupby("_comp", sort=True):
        g = g.copy()

        # representative row: highest weight; tie-break by longest base (more specific)
        g["_base_len"] = g.get("name_base", "").astype(str).str.len()
        rep_idx = g.sort_values(["_weight", "_base_len"], ascending=[False, False]).index[0]
        rep = g.loc[rep_idx]

        # stable uid from sorted ID list across the component
        key_parts = []
        for c in id_cols_for_uid:
            vals = sorted({str(v) for v in g[c].dropna().astype(str).tolist() if str(v).strip() != ""})
            key_parts.append(",".join(vals))
        uid = stable_hash_id(uid_prefix, key_parts)

        out = {"uid": uid, "component": int(comp), "n_members": int(len(g))}
        for c in keep_cols:
            if c in df.columns:
                out[c] = rep[c]
        for c in list_cols:
            if c in df.columns:
                vals = sorted({str(v) for v in g[c].dropna().astype(str).tolist() if str(v).strip() != ""})
                out[c] = ";".join(vals) if vals else None

        canon_rows.append(out)

        for idx in g.index.tolist():
            map_rows.append({"row_index": int(idx), "uid": uid, "component": int(comp)})

    canon_df = pd.DataFrame(canon_rows)
    map_df = pd.DataFrame(map_rows)
    return canon_df, map_df
