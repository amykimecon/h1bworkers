#!/usr/bin/env python3
"""Query Blackfish LLM to validate FOIA->Revelio company matches."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml

try:
    import openai  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Install openai to run the LLM stage: pip install openai") from e


DEFAULTS = {
    "score_threshold": 95.0,
    "batch_size": 5,
    "model": "gpt-4o-mini",
    "sleep_seconds": 0.0,
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


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _expand(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.expanduser(os.path.expandvars(path))


def _make_client(base_url: Optional[str], api_key: Optional[str]):
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("BLACKFISH_API_KEY")
    if not key:
        raise ValueError("Provide API key via --api-key or env OPENAI_API_KEY/BLACKFISH_API_KEY")
    kwargs: Dict[str, Any] = {"api_key": key}
    if base_url or os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")
    return openai.OpenAI(**kwargs)


def _eligible_foia(df: pd.DataFrame, score_threshold: float) -> List[str]:
    g = df.groupby("foia_firm_uid", sort=False)
    stats = g["score"].agg(["count", "min"]).reset_index()
    keep = stats[(stats["count"] > 1) | (stats["min"] < score_threshold)]
    return keep["foia_firm_uid"].astype(str).tolist()


def _load_master(path: Path) -> tuple[Optional[List[str]], set[str], bool]:
    """Return (eligible_list_or_None, processed_set, has_sections)."""
    if not path.exists():
        return None, set(), False
    eligible: List[str] = []
    processed: set[str] = set()
    has_sections = False
    mode = "processed"
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if s.upper().startswith("# ELIGIBLE"):
                    mode = "eligible"
                    has_sections = True
                elif s.upper().startswith("# PROCESSED"):
                    mode = "processed"
                    has_sections = True
                continue
            if mode == "eligible":
                eligible.append(s)
            else:
                processed.add(s)
    return (eligible if has_sections else None), processed, has_sections


def _write_master(path: Path, eligible: List[str], processed: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# ELIGIBLE\n")
        for s in eligible:
            f.write(f"{s}\n")
        f.write("# PROCESSED\n")
        for s in processed:
            f.write(f"{s}\n")


def _append_master(path: Path, foia_uid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"{foia_uid}\n")


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)[:120]


def _build_system_prompt() -> str:
    return (
        "You are validating candidate company matches between one 'FOIA' firm (usually clean, possible typos, mostly deduplicated) from administrative data and multiple candidate 'Revelio' firms (messy, typos and duplicates likely) from LinkedIn data. Many LinkedIn firms may be matched to the same FOIA firm. The state is not very important and should be used only to adjust the confidence by small amounts. The number of employees is even less important (if all else equal, lower number of employees relative to other valid match(es) is a positive signal of a valid match). \n"
        "Return JSON ONLY as an array of objects, one per candidate, with keys:\n"
        "- candidate_index (int, 1-based within this prompt)\n"
        "- is_valid (boolean)\n"
        "- confidence (0..1)\n"
        "- reason (<=25 words)\n"
        "If unsure, set is_valid=false with low confidence.\n"
    )


def _build_user_prompt(foia_row: dict, candidates: List[dict], batch_index: int, total_batches: int) -> str:
    hq = foia_row.get("hq_state") or "NA"
    work = foia_row.get("work_state") or "NA"
    header = (
        f"FOIA firm: {foia_row.get('foia_name')} (uid={foia_row.get('foia_firm_uid')})\n"
        f"FOIA State(s): {hq}, {work}\n"
        f"Batch: {batch_index}/{total_batches}\n"
        "Candidates:\n"
    )
    lines = []
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"{i}. LinkedIn Firm Name: {c.get('revelio_name')}; "
            f"LinkedIn State: {c.get('revelio_state') or 'NA'}; "
            f"Number of Employees on LinkedIn: {c.get('revelio_employee_n') or 'NA'}; "
            # f"score={c.get('score')}; "
            # f"match_type={c.get('match_type')}"
        )
    return header + "\n".join(lines) + "\n"


def _attach_foia_states(df: pd.DataFrame, foia_firms_path: Path, hq_col: str, work_col: str) -> pd.DataFrame:
    if not foia_firms_path.exists():
        return df
    foia = pd.read_csv(
        foia_firms_path,
        usecols=["foia_firm_uid", hq_col, work_col],
        dtype={"foia_firm_uid": "string"},
        low_memory=False,
    )
    foia = foia.rename(columns={hq_col: "hq_state", work_col: "work_state"})
    return df.merge(foia, on="foia_firm_uid", how="left")


def _norm_id(val: object) -> str:
    s = "" if val is None else str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _attach_revelio_fields(
    df: pd.DataFrame,
    entities_path: Path,
    rcid_col: str,
    emp_col: str,
    state_col: str,
) -> pd.DataFrame:
    if not entities_path.exists() or df.empty:
        return df
    needed = {_norm_id(v) for v in df["revelio_firm_uid"].dropna().tolist() if _norm_id(v)}
    if not needed:
        return df

    rcid_to_n: dict[str, float] = {}
    rcid_to_state: dict[str, str] = {}
    for chunk in pd.read_csv(
        entities_path,
        usecols=[rcid_col, emp_col, state_col],
        dtype={rcid_col: "string", state_col: "string"},
        chunksize=500_000,
        low_memory=False,
    ):
        chunk["rcid_norm"] = chunk[rcid_col].map(_norm_id)
        chunk = chunk[chunk["rcid_norm"].isin(needed)]
        if chunk.empty:
            continue
        for rcid, n, st in zip(chunk["rcid_norm"], chunk[emp_col], chunk[state_col]):
            if pd.isna(rcid):
                continue
            try:
                n_val = float(n)
            except Exception:
                n_val = None
            if n_val is not None:
                prev = rcid_to_n.get(rcid)
                if prev is None or n_val > prev:
                    rcid_to_n[rcid] = n_val
            if rcid not in rcid_to_state and isinstance(st, str) and st.strip():
                rcid_to_state[rcid] = st.strip()

    df = df.copy()
    df["revelio_employee_n"] = df["revelio_firm_uid"].map(lambda v: rcid_to_n.get(_norm_id(v)))
    df["revelio_state"] = df["revelio_firm_uid"].map(lambda v: rcid_to_state.get(_norm_id(v)))
    return df


def _iter_batches(rows: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def _record_shard_event(
    log_dir: Path,
    shard: int,
    num_shards: int,
    status: str,
    detail: Optional[str] = None,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts}\tstatus={status}\tshard={shard}\tnum_shards={num_shards}"
    if detail:
        line += f"\t{detail}"
    with (log_dir / "shard_runs.log").open("a") as f:
        f.write(line + "\n")
    marker = log_dir / f"shard_{shard}_of_{num_shards}.{status}"
    with marker.open("a") as f:
        f.write(line + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="YAML config")
    p.add_argument("--candidates", help="crosswalk_candidates.csv (override config)")
    p.add_argument("--foia-firms", help="foia_firms_dedup.csv (override config)")
    p.add_argument("--revelio-entities", help="revelio_rcid_entities.csv (override config)")
    p.add_argument("--output-dir", help="Output directory for raw JSON responses")
    p.add_argument("--master-list", help="Master list file (processed foia_firm_uid)")
    p.add_argument("--model", help="LLM model name")
    p.add_argument("--batch-size", type=int, default=None, help="Companies per request")
    p.add_argument("--score-threshold", type=float, default=None, help="Threshold for low-score rule")
    p.add_argument("--base-url", default=None, help="OPENAI_BASE_URL / Blackfish endpoint")
    p.add_argument("--api-key", default=None, help="OPENAI_API_KEY / Blackfish key")
    p.add_argument("--limit-foia", type=int, default=None, help="Limit number of FOIA firms processed")
    p.add_argument("--shard", type=int, default=None, help="Shard index (0-based)")
    p.add_argument("--num-shards", type=int, default=None, help="Total number of shards")
    p.add_argument("--sleep-seconds", type=float, default=None, help="Sleep between requests")
    p.add_argument("--dry-run", action="store_true", help="Show counts and exit")
    args = p.parse_args()

    cfg = _load_config(args.config)
    print(f"Config loaded: {args.config}", flush=True)

    candidates = _expand(args.candidates) or _expand(_cfg(cfg, "paths", "crosswalk_candidates"))
    foia_firms = _expand(args.foia_firms) or _expand(_cfg(cfg, "paths", "foia_firms"))
    revelio_entities = _expand(args.revelio_entities) or _expand(_cfg(cfg, "paths", "revelio_entities"))
    out_dir = Path(_expand(args.output_dir) or _expand(_cfg(cfg, "paths", "output_dir", default="llm_outputs")))
    master_list = Path(
        _expand(args.master_list)
        or _expand(_cfg(cfg, "paths", "master_list", default=out_dir / "llm_master.txt"))
    )
    print(f"Candidates path: {candidates}", flush=True)
    print(f"FOIA firms path: {foia_firms}", flush=True)
    print(f"Revelio entities path: {revelio_entities}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"Master list: {master_list}", flush=True)

    score_threshold = args.score_threshold if args.score_threshold is not None else float(
        _cfg(cfg, "options", "score_threshold", default=DEFAULTS["score_threshold"])
    )
    batch_size = args.batch_size if args.batch_size is not None else int(
        _cfg(cfg, "options", "batch_size", default=DEFAULTS["batch_size"])
    )
    sleep_seconds = args.sleep_seconds if args.sleep_seconds is not None else float(
        _cfg(cfg, "options", "sleep_seconds", default=DEFAULTS["sleep_seconds"])
    )
    base_url = args.base_url if args.base_url is not None else _cfg(cfg, "options", "base_url", default=None)
    api_key = args.api_key if args.api_key is not None else _cfg(cfg, "options", "api_key", default=None)
    verbose = _as_bool(_cfg(cfg, "options", "verbose", default=False))
    print(
        f"Options: batch_size={batch_size} score_threshold={score_threshold} "
        f"sleep_seconds={sleep_seconds} base_url={base_url} verbose={verbose}",
        flush=True,
    )

    if not candidates:
        p.error("Provide --candidates or set paths.crosswalk_candidates in config.")
    cand_path = Path(candidates)
    if not cand_path.exists():
        p.error(f"Candidates file not found: {cand_path}")

    df = pd.read_csv(
        cand_path,
        dtype={"foia_firm_uid": "string", "revelio_firm_uid": "string"},
        low_memory=False,
    )
    print(f"Loaded candidates: {len(df):,} rows", flush=True)
    if df.empty:
        print("No candidate rows.")
        return

    foia_hq_col = _cfg(cfg, "columns", "foia_hq", default="hq_state_mode")
    foia_work_col = _cfg(cfg, "columns", "foia_work", default="work_state_mode")
    if foia_firms:
        print("Attaching FOIA state fields...", flush=True)
        df = _attach_foia_states(df, Path(foia_firms), foia_hq_col, foia_work_col)

    rev_rcid_col = _cfg(cfg, "columns", "revelio_rcid", default="rcid")
    rev_n_col = _cfg(cfg, "columns", "revelio_employee_n", default="n")
    rev_state_col = _cfg(cfg, "columns", "revelio_state", default="top_state")
    if revelio_entities:
        print("Attaching Revelio fields...", flush=True)
        df = _attach_revelio_fields(df, Path(revelio_entities), rev_rcid_col, rev_n_col, rev_state_col)

    eligible_cached, done, has_sections = _load_master(master_list)
    if eligible_cached is None:
        print("Building master list of eligible FOIA firms...", flush=True)
        foia_uids = _eligible_foia(df, score_threshold)
        _write_master(master_list, sorted(foia_uids), sorted(done))
        eligible_cached, done, has_sections = _load_master(master_list)
    foia_uids = eligible_cached or []
    if not foia_uids:
        print("No FOIA firms meet the queue criteria.")
        return

    foia_uids = sorted(foia_uids)
    print(f"Eligible FOIA firms (from master list): {len(foia_uids):,}", flush=True)
    global_remaining = sum(1 for u in foia_uids if u not in done)
    global_total = len(done) + global_remaining
    shard_log_dir: Optional[Path] = None
    if (args.shard is None) != (args.num_shards is None):
        p.error("Provide both --shard and --num-shards, or neither.")
    if args.num_shards is not None:
        if args.num_shards <= 0:
            p.error("--num-shards must be > 0")
        if args.shard < 0 or args.shard >= args.num_shards:
            p.error("--shard must be in [0, num-shards)")
        total_before = len(foia_uids)
        foia_uids = [u for i, u in enumerate(foia_uids) if (i % args.num_shards) == args.shard]
        print(
            f"Shard {args.shard}/{args.num_shards}: {len(foia_uids):,} of {total_before:,} eligible FOIA firms",
            flush=True,
        )
        shard_log_dir = out_dir / "shards"

    # After sharding, filter out already-processed firms to keep shard membership deterministic.
    foia_uids = [u for u in foia_uids if u not in done]
    print(f"Eligible FOIA firms (after master filter): {len(foia_uids):,}", flush=True)
    if args.limit_foia:
        foia_uids = foia_uids[: int(args.limit_foia)]
        print(f"Limit applied: processing {len(foia_uids):,} FOIA firms", flush=True)

    if args.dry_run:
        if shard_log_dir is not None:
            _record_shard_event(shard_log_dir, args.shard, args.num_shards, "dry_run")
        print(f"Eligible foia firms: {len(foia_uids)} (after master filter)")
        return

    client = _make_client(base_url=base_url, api_key=api_key)
    system_prompt = _build_system_prompt()
    print("LLM client initialized.", flush=True)
    
    models = client.models.list()
    print(models)
    
    model = args.model or _cfg(cfg, "options", "model", default=models.data[0].id)
    print(f"Using model: {model}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {out_dir}", flush=True)

    if shard_log_dir is not None:
        _record_shard_event(
            shard_log_dir,
            args.shard,
            args.num_shards,
            "start",
            detail=f"model={model} config={args.config or ''}",
        )

    total_firms = global_total
    processed_count = len(done)
    shard_total = len(foia_uids)
    shard_done = 0
    progress_every = 10
    for foia_uid in foia_uids:
        print(f"Processing FOIA firm: {foia_uid}", flush=True)
        sub = df[df["foia_firm_uid"].astype(str) == foia_uid].copy()
        if sub.empty:
            print("No rows for FOIA firm; skipping.", flush=True)
            continue
        sub = sub.sort_values("score", ascending=False)
        foia_name = sub["foia_name"].iloc[0] if "foia_name" in sub.columns else foia_uid
        hq_state = sub["hq_state"].iloc[0] if "hq_state" in sub.columns else None
        work_state = sub["work_state"].iloc[0] if "work_state" in sub.columns else None

        rows = sub.to_dict(orient="records")
        batches = list(_iter_batches(rows, batch_size))
        total_batches = len(batches)
        print(f"Total batches for {foia_uid}: {total_batches}", flush=True)

        for b_idx, batch in enumerate(batches, start=1):
            batch_started = time.time()
            foia_row = {
                "foia_firm_uid": foia_uid,
                "foia_name": foia_name,
                "hq_state": hq_state,
                "work_state": work_state,
            }
            user_prompt = _build_user_prompt(foia_row, batch, b_idx, total_batches)
            if verbose:
                print(
                    f"=== LLM request foia_uid={foia_uid} batch={b_idx}/{total_batches} ===",
                    flush=True,
                )
                print(system_prompt, flush=True)
                print(user_prompt, flush=True)
                print("=== end request ===", flush=True)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=120,
            )
            content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else None
            if verbose:
                print(
                    f"=== LLM response foia_uid={foia_uid} batch={b_idx}/{total_batches} ===",
                    flush=True,
                )
                print(content or "", flush=True)
                print("=== end response ===", flush=True)
            out = {
                "foia_firm_uid": foia_uid,
                "foia_name": foia_name,
                "batch_index": b_idx,
                "total_batches": total_batches,
                "model": model,
                "prompt": {"system": system_prompt, "user": user_prompt},
                "response": content,
            }
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"{_safe_name(foia_uid)}__b{b_idx:03d}of{total_batches:03d}__{ts}.json"
            with (out_dir / fname).open("w") as f:
                json.dump(out, f, ensure_ascii=True, indent=2)
            elapsed = time.time() - batch_started
            print(
                f"Wrote {fname} (batch {b_idx}/{total_batches}) in {elapsed:.1f}s",
                flush=True,
            )
            if sleep_seconds and sleep_seconds > 0:
                time.sleep(float(sleep_seconds))

        _append_master(master_list, foia_uid)
        processed_count += 1
        shard_done += 1
        print(
            f"Progress (shard): {shard_done}/{shard_total} | "
            f"Progress (global): {processed_count}/{total_firms}",
            flush=True,
        )
        if processed_count % progress_every == 0 or processed_count == total_firms:
            print(f"Checkpoint (global): {processed_count}/{total_firms} firms processed", flush=True)

    if shard_log_dir is not None:
        _record_shard_event(
            shard_log_dir,
            args.shard,
            args.num_shards,
            "done",
            detail=f"model={model}",
        )


if __name__ == "__main__":
    main()
