#!/usr/bin/env python3
"""Compile LLM review JSON outputs into cleaned match tables."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".csv", ".gz", ".bz2", ".zip"):
        return pd.read_csv(p, low_memory=False)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def _clean_fein(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    if "(" in s and ")" in s:
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if digits else None


def _canonical_fein_key(x: Any) -> Optional[str]:
    """Normalize FEIN for joins across artifacts with inconsistent leading zeros."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = "".join(ch for ch in str(x).strip() if ch.isdigit())
    if not s:
        return None
    s = s.lstrip("0")
    return s if s else "0"


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_response(text: Optional[str]) -> Tuple[List[Dict[str, Any]], str]:
    if not text:
        return [], "empty"
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data, "ok"
        return [], "not_list"
    except Exception:
        pass

    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data, "ok_extracted"
            return [], "extracted_not_list"
        except Exception:
            return [], "extract_parse_error"
    return [], "parse_error"


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


def _count_candidates_from_prompt(prompt_user: Optional[str]) -> Optional[int]:
    if not prompt_user:
        return None
    count = 0
    for line in prompt_user.splitlines():
        if re.match(r"\s*\d+\.\s+", line):
            count += 1
    return count or None


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if v == 1:
            return True
        if v == 0:
            return False
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _iter_llm_files(out_dir: Path) -> Iterable[Path]:
    if not out_dir.exists():
        return []
    return sorted(p for p in out_dir.glob("*.json") if p.is_file())


def _name_base(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(x).casefold())


def _state_mismatch_exception_mask(df: pd.DataFrame, min_confidence: Optional[float]) -> pd.Series:
    """Exception rule: keep top candidate even below threshold for explicit state-mismatch reasons."""
    if min_confidence is None:
        return pd.Series(False, index=df.index)
    reason = df.get("reason", pd.Series("", index=df.index)).astype("string").fillna("").str.casefold()
    has_phrase = (
        reason.str.contains("different state", regex=False)
        | reason.str.contains("state mismatch", regex=False)
        | reason.str.contains("wrong state", regex=False)
    )
    top_choice = pd.to_numeric(df.get("candidate_index"), errors="coerce").eq(1)
    below_threshold = pd.to_numeric(df.get("confidence"), errors="coerce").fillna(0.0) < float(min_confidence)
    return top_choice & below_threshold & has_phrase


def _mark_kept_matches(
    judgments: pd.DataFrame,
    *,
    min_confidence: Optional[float],
    require_valid: bool,
) -> pd.DataFrame:
    out = judgments.copy()
    if "valid_by_exception" not in out.columns:
        out["valid_by_exception"] = _state_mismatch_exception_mask(out, min_confidence=min_confidence)
    keep = pd.Series(True, index=out.index)
    if require_valid:
        keep = (out["is_valid"] == True) | out["valid_by_exception"]  # noqa: E712
    if min_confidence is not None:
        keep = keep & (
            (pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0) >= float(min_confidence))
            | out["valid_by_exception"]
        )
    out["is_kept_match"] = keep
    return out


def _classify_firm_statuses(
    *,
    foia_uids: pd.Series,
    candidates: pd.DataFrame,
    judgments_eval: pd.DataFrame,
    score_threshold: float,
) -> pd.DataFrame:
    cand_stats = (
        candidates.groupby("foia_firm_uid", sort=False)["score"]
        .agg(candidate_count="count", min_candidate_score="min")
        .reset_index()
    )
    cand_stats["llm_eligible"] = (
        (cand_stats["candidate_count"] > 1)
        | (pd.to_numeric(cand_stats["min_candidate_score"], errors="coerce") < float(score_threshold))
    )

    reviewed = (
        judgments_eval.groupby("foia_firm_uid", sort=False)
        .agg(
            llm_reviewed=("foia_firm_uid", "size"),
            llm_kept_matches=("is_kept_match", lambda s: int(pd.Series(s).fillna(False).sum())),
        )
        .reset_index()
    )
    reviewed["llm_reviewed"] = reviewed["llm_reviewed"] > 0
    reviewed["llm_has_match"] = reviewed["llm_kept_matches"] > 0

    universe = pd.DataFrame({"foia_firm_uid": foia_uids.astype("string").dropna().drop_duplicates()})
    status = universe.merge(cand_stats, on="foia_firm_uid", how="left").merge(reviewed, on="foia_firm_uid", how="left")
    status["candidate_count"] = pd.to_numeric(status["candidate_count"], errors="coerce").fillna(0).astype(int)
    status["llm_eligible"] = status["llm_eligible"].fillna(False)
    status["llm_reviewed"] = status["llm_reviewed"].fillna(False)
    status["llm_has_match"] = status["llm_has_match"].fillna(False)

    status["firm_status"] = "llm not reviewed"
    status.loc[status["candidate_count"] == 0, "firm_status"] = "no candidates"
    status.loc[(status["candidate_count"] > 0) & (~status["llm_eligible"]), "firm_status"] = "exact match"
    status.loc[status["llm_eligible"] & status["llm_reviewed"] & status["llm_has_match"], "firm_status"] = "llm match"
    status.loc[status["llm_eligible"] & status["llm_reviewed"] & (~status["llm_has_match"]), "firm_status"] = "llm no match"
    status.loc[status["llm_eligible"] & (~status["llm_reviewed"]), "firm_status"] = "llm not reviewed"
    return status


def _map_foia_row_to_unique_firm(
    *,
    foia_raw: pd.DataFrame,
    fein_map: pd.DataFrame,
    foia_firms: pd.DataFrame,
) -> pd.DataFrame:
    foia_firms = foia_firms.copy()
    foia_firms["foia_firm_uid"] = foia_firms["foia_firm_uid"].astype("string")
    for c in ["canonical_name_clean", "canonical_name_base", "canonical_name_stub", "raw_name_example"]:
        if c not in foia_firms.columns:
            foia_firms[c] = ""
        foia_firms[c] = foia_firms[c].astype("string").fillna("")

    firm_ref = foia_firms[
        ["foia_firm_uid", "canonical_name_clean", "canonical_name_base", "canonical_name_stub", "raw_name_example"]
    ].drop_duplicates(subset=["foia_firm_uid"])
    candidates = fein_map.merge(firm_ref, on="foia_firm_uid", how="left")
    candidates["fein_key"] = candidates["fein_clean"].map(_canonical_fein_key).astype("string")

    raw_unique = foia_raw[["fein_clean", "fein_year", "original_name"]].drop_duplicates().reset_index(drop=True)
    raw_unique["fein_key"] = raw_unique["fein_clean"].map(_canonical_fein_key).astype("string")
    raw_unique["raw_key"] = raw_unique.index.astype("int64")
    raw_unique["original_name_base"] = raw_unique["original_name"].map(_name_base)
    raw_unique["original_name_clean"] = raw_unique["original_name"].astype("string").fillna("").str.casefold()

    merged = raw_unique.merge(candidates, on=["fein_key", "fein_year"], how="inner", suffixes=("_raw", "_map"))
    if merged.empty:
        return merged[["fein_clean_raw", "fein_year", "original_name", "foia_firm_uid"]].rename(
            columns={"fein_clean_raw": "fein_clean"}
        )

    merged["cand_base"] = merged["canonical_name_base"].astype("string").fillna("").map(_name_base)
    merged["cand_clean"] = merged["canonical_name_clean"].astype("string").fillna("").str.casefold()
    merged["cand_stub"] = merged["canonical_name_stub"].astype("string").fillna("").str.casefold()
    merged["raw_ex"] = merged["raw_name_example"].astype("string").fillna("").str.casefold()

    merged["exact_base"] = (merged["original_name_base"] == merged["cand_base"]).astype(int)
    merged["exact_clean"] = (merged["original_name_clean"] == merged["cand_clean"]).astype(int)
    merged["sim_base"] = merged.apply(
        lambda r: SequenceMatcher(None, r["original_name_base"], r["cand_base"]).ratio()
        if r["original_name_base"] and r["cand_base"]
        else 0.0,
        axis=1,
    )
    merged["sim_clean"] = merged.apply(
        lambda r: max(
            SequenceMatcher(None, r["original_name_clean"], r["cand_clean"]).ratio()
            if r["original_name_clean"] and r["cand_clean"]
            else 0.0,
            SequenceMatcher(None, r["original_name_clean"], r["cand_stub"]).ratio()
            if r["original_name_clean"] and r["cand_stub"]
            else 0.0,
            SequenceMatcher(None, r["original_name_clean"], r["raw_ex"]).ratio()
            if r["original_name_clean"] and r["raw_ex"]
            else 0.0,
        ),
        axis=1,
    )
    merged["name_score"] = (
        merged["exact_base"] * 1000.0
        + merged["exact_clean"] * 500.0
        + merged["sim_base"] * 100.0
        + merged["sim_clean"] * 100.0
    )

    best = (
        merged.sort_values(["raw_key", "name_score", "foia_firm_uid"], ascending=[True, False, True], kind="mergesort")
        .drop_duplicates(subset=["raw_key"], keep="first")
        .copy()
    )
    return best[["fein_clean_raw", "fein_year", "original_name", "foia_firm_uid"]].rename(
        columns={"fein_clean_raw": "fein_clean"}
    )


def _build_foia_to_rcid_crosswalk(
    *,
    judgments: pd.DataFrame,
    foia_raw_path: Path,
    foia_fein_to_firm_path: Path,
    foia_firms_path: Path,
    fein_col: str,
    year_col: str,
    name_col: str,
) -> pd.DataFrame:
    keep_cols = [
        "foia_firm_uid",
        "revelio_firm_uid",
        "revelio_name",
        "confidence",
        "is_valid",
        "score",
        "match_type",
        "batch_index",
        "candidate_index",
        "reason",
        "llm_file",
        "parse_status",
        "valid_by_exception",
    ]
    existing = [c for c in keep_cols if c in judgments.columns]
    keep = judgments[existing].copy()
    keep = keep.dropna(subset=["foia_firm_uid", "revelio_firm_uid"])
    keep["foia_firm_uid"] = keep["foia_firm_uid"].astype("string")
    keep["revelio_firm_uid"] = keep["revelio_firm_uid"].astype("string")

    fein_map = _read_any(str(foia_fein_to_firm_path))
    if "foia_firm_uid" not in fein_map.columns or "fein_clean" not in fein_map.columns:
        raise ValueError("foia_fein_to_firm must include at least: foia_firm_uid, fein_clean")
    fein_map = fein_map.copy()
    fein_map["foia_firm_uid"] = fein_map["foia_firm_uid"].astype("string")
    fein_map["fein_clean"] = fein_map["fein_clean"].astype("string")
    if "fein_year" in fein_map.columns:
        fein_map["fein_year"] = pd.to_numeric(fein_map["fein_year"], errors="coerce").astype("Int64")
    else:
        fein_map["fein_year"] = pd.NA
    fein_map = fein_map[["foia_firm_uid", "fein_clean", "fein_year"]].drop_duplicates()

    foia_firms = _read_any(str(foia_firms_path))
    if "foia_firm_uid" not in foia_firms.columns:
        raise ValueError("foia_firms_dedup must include foia_firm_uid")

    foia_raw = _read_any(str(foia_raw_path))
    if fein_col not in foia_raw.columns:
        raise ValueError(f"FOIA raw file missing FEIN column: {fein_col}")
    if year_col not in foia_raw.columns:
        raise ValueError(f"FOIA raw file missing year column: {year_col}")
    if name_col not in foia_raw.columns:
        raise ValueError(f"FOIA raw file missing employer name column: {name_col}")
    foia_raw = foia_raw.copy()
    foia_raw["fein_clean"] = foia_raw[fein_col].map(_clean_fein).astype("string")
    foia_raw["fein_year"] = pd.to_numeric(foia_raw[year_col], errors="coerce").astype("Int64")
    foia_raw["original_name"] = foia_raw[name_col].astype("string")
    foia_raw = foia_raw[["fein_clean", "fein_year", "original_name"]]
    foia_raw = foia_raw.dropna(subset=["fein_clean", "fein_year", "original_name"])
    foia_raw = foia_raw.drop_duplicates()

    foia_with_firm = _map_foia_row_to_unique_firm(
        foia_raw=foia_raw,
        fein_map=fein_map,
        foia_firms=foia_firms,
    )
    crosswalk = foia_with_firm.merge(keep, on="foia_firm_uid", how="inner")
    crosswalk["rcid"] = crosswalk["revelio_firm_uid"]
    out_cols = ["fein_clean", "fein_year", "original_name", "foia_firm_uid", "revelio_firm_uid", "revelio_name", "rcid"]
    out_cols += [
        c
        for c in [
            "is_valid",
            "valid_by_exception",
            "confidence",
            "score",
            "match_type",
            "batch_index",
            "candidate_index",
            "reason",
            "parse_status",
            "llm_file",
        ]
        if c in crosswalk.columns
    ]
    return crosswalk[out_cols].drop_duplicates()


def _build_full_firm_status_crosswalk(
    *,
    foia_with_firm: pd.DataFrame,
    foia_firm_uids: pd.Series,
    candidates: pd.DataFrame,
    judgments_eval: pd.DataFrame,
    score_threshold: float,
) -> pd.DataFrame:
    base_universe = pd.DataFrame({"foia_firm_uid": foia_firm_uids.astype("string").dropna().drop_duplicates()})
    foia_rows = base_universe.merge(foia_with_firm, on="foia_firm_uid", how="left")
    for c in ["fein_clean", "fein_year", "original_name"]:
        if c not in foia_rows.columns:
            foia_rows[c] = pd.NA

    status = _classify_firm_statuses(
        foia_uids=foia_rows["foia_firm_uid"],
        candidates=candidates,
        judgments_eval=judgments_eval,
        score_threshold=score_threshold,
    )
    base = foia_rows.merge(status[["foia_firm_uid", "firm_status"]], on="foia_firm_uid", how="left")

    exact = base[base["firm_status"] == "exact match"].merge(
        candidates.sort_values(["foia_firm_uid", "score"], ascending=[True, False], kind="mergesort")
        .drop_duplicates(subset=["foia_firm_uid"], keep="first")[
            ["foia_firm_uid", "revelio_firm_uid", "revelio_name", "score", "match_type"]
        ],
        on="foia_firm_uid",
        how="left",
    )
    exact["rcid"] = exact["revelio_firm_uid"]
    exact["is_kept_match"] = True

    llm_rows = base[base["firm_status"].isin(["llm match", "llm no match"])].merge(
        judgments_eval.copy(),
        on="foia_firm_uid",
        how="left",
    )
    llm_rows["rcid"] = llm_rows["revelio_firm_uid"]

    unresolved = base[base["firm_status"].isin(["llm not reviewed", "no candidates"])].copy()
    for c in [
        "revelio_firm_uid",
        "revelio_name",
        "rcid",
        "is_valid",
        "valid_by_exception",
        "confidence",
        "score",
        "match_type",
        "batch_index",
        "candidate_index",
        "reason",
        "parse_status",
        "llm_file",
        "is_kept_match",
    ]:
        unresolved[c] = pd.NA

    out = pd.concat([exact, llm_rows, unresolved], ignore_index=True, sort=False)
    out["is_kept_match"] = out["is_kept_match"].fillna(False)
    out["firm_validity_label"] = "firm_no_match"
    out.loc[out["firm_status"].isin(["exact match", "llm match"]), "firm_validity_label"] = "firm_has_match"
    out.loc[out["firm_status"] == "llm not reviewed", "firm_validity_label"] = "firm_unreviewed"
    out["crosswalk_validity_label"] = "unmatched_firm"
    out.loc[out["firm_status"].isin(["llm match", "llm no match"]) & (~out["is_kept_match"]), "crosswalk_validity_label"] = "llm_rejected"
    out.loc[out["is_kept_match"], "crosswalk_validity_label"] = "valid_match"

    out_cols = [
        "fein_clean",
        "fein_year",
        "original_name",
        "foia_firm_uid",
        "firm_status",
        "firm_validity_label",
        "crosswalk_validity_label",
        "is_kept_match",
        "revelio_firm_uid",
        "revelio_name",
        "rcid",
        "is_valid",
        "valid_by_exception",
        "confidence",
        "score",
        "match_type",
        "batch_index",
        "candidate_index",
        "reason",
        "parse_status",
        "llm_file",
    ]
    for c in out_cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[out_cols].drop_duplicates()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config (same schema as run_crosswalk_llm_review.py)")
    ap.add_argument("--candidates", help="crosswalk_candidates.csv (override config)")
    ap.add_argument("--llm-output-dir", help="Directory with LLM JSON outputs (override config)")
    ap.add_argument("--out-all", default=None, help="Output CSV (all LLM judgments)")
    ap.add_argument("--out-matches", default=None, help="Output CSV (filtered matches)")
    ap.add_argument("--out-crosswalk", default=None, help="Output CSV (FEIN x YEAR x Original Name -> valid RCID matches)")
    ap.add_argument("--out-crosswalk-all", default=None, help="Output CSV (FEIN x YEAR x Original Name -> RCID using all parsed matches)")
    ap.add_argument("--foia", default=None, help="Raw FOIA file used to expand to original FEIN/year/name rows")
    ap.add_argument("--foia-fein-to-firm", default=None, help="foia_fein_to_firm.csv for FEIN/year -> foia_firm_uid mapping")
    ap.add_argument("--foia-firms", default=None, help="foia_firms_dedup.csv used to disambiguate FEIN/year/name -> foia_firm_uid")
    ap.add_argument("--fein-col", default=None, help="FEIN column in raw FOIA file")
    ap.add_argument("--year-col", default=None, help="Year column in raw FOIA file")
    ap.add_argument("--foia-name-col", default=None, help="Employer/original name column in raw FOIA file")
    ap.add_argument("--score-threshold", type=float, default=None, help="LLM eligibility threshold (same meaning as run_crosswalk_llm_review.py)")
    ap.add_argument("--min-confidence", type=float, default=None, help="Min confidence for matches")
    ap.add_argument("--require-valid", action="store_true", help="Require is_valid == True for matches")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    cand_path = Path(
        _expand(args.candidates)
        or _expand(_cfg(cfg, "paths", "crosswalk_candidates"))
        or ""
    )
    if not cand_path.exists():
        ap.error(f"Candidates file not found: {cand_path}")

    out_dir = Path(
        _expand(args.llm_output_dir)
        or _expand(_cfg(cfg, "paths", "output_dir", default="llm_outputs"))
    )
    out_all = Path(
        _expand(args.out_all)
        or _expand(_cfg(cfg, "paths", "out_all"))
        or str(out_dir / "llm_review_all.csv")
    )
    out_matches = Path(
        _expand(args.out_matches)
        or _expand(_cfg(cfg, "paths", "out_matches"))
        or str(out_dir / "llm_review_matches.csv")
    )
    out_crosswalk = Path(
        _expand(args.out_crosswalk)
        or _expand(_cfg(cfg, "paths", "out_crosswalk"))
        or str(out_dir / "llm_review_valid_foia_to_rcid_crosswalk.csv")
    )
    out_crosswalk_all = Path(
        _expand(args.out_crosswalk_all)
        or _expand(_cfg(cfg, "paths", "out_crosswalk_all"))
        or str(out_dir / "llm_review_all_foia_to_rcid_crosswalk.csv")
    )
    foia_raw_path = _expand(args.foia) or _expand(_cfg(cfg, "paths", "foia"))
    foia_fein_to_firm_path = (
        _expand(args.foia_fein_to_firm)
        or _expand(_cfg(cfg, "paths", "foia_fein_to_firm"))
        or str(cand_path.parent / "foia_fein_to_firm.csv")
    )
    foia_firms_path = (
        _expand(args.foia_firms)
        or _expand(_cfg(cfg, "paths", "foia_firms"))
        or str(cand_path.parent / "foia_firms_dedup.csv")
    )
    fein_col = args.fein_col or _cfg(cfg, "columns", "fein_col", default="FEIN")
    year_col = args.year_col or _cfg(cfg, "columns", "year", default="lottery_year")
    foia_name_col = args.foia_name_col or _cfg(cfg, "columns", "foia_name", default="employer_name")
    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else float(_cfg(cfg, "options", "score_threshold", default=95.0))
    )
    min_confidence = (
        args.min_confidence
        if args.min_confidence is not None
        else _cfg(cfg, "options", "min_confidence", default=0.5)
    )
    require_valid = (
        True
        if args.require_valid
        else bool(_cfg(cfg, "options", "require_valid", default=False))
    )

    cand = pd.read_csv(cand_path, dtype={"foia_firm_uid": "string", "revelio_firm_uid": "string"}, low_memory=False)
    if cand.empty:
        raise SystemExit("Candidates file is empty.")

    if "score" in cand.columns:
        cand = cand.sort_values(["foia_firm_uid", "score"], ascending=[True, False], kind="mergesort")
    else:
        cand = cand.sort_values(["foia_firm_uid"], kind="mergesort")

    cand_groups: Dict[str, pd.DataFrame] = {
        str(k): v.reset_index(drop=True) for k, v in cand.groupby("foia_firm_uid", sort=False)
    }

    llm_files = list(_iter_llm_files(out_dir))
    if not llm_files:
        raise SystemExit(f"No JSON files found in {out_dir}")

    batches_by_foia: Dict[str, List[Dict[str, Any]]] = {}
    for path in llm_files:
        payload = _safe_read_json(path)
        if not payload:
            continue
        foia_uid = _stringify(payload.get("foia_firm_uid"))
        if not foia_uid:
            continue
        batch_index = payload.get("batch_index")
        prompt_user = None
        if isinstance(payload.get("prompt"), dict):
            prompt_user = payload["prompt"].get("user")
        batch_len = _count_candidates_from_prompt(prompt_user)
        resp_text = payload.get("response")
        parsed, parse_status = _parse_response(resp_text)
        batches_by_foia.setdefault(foia_uid, []).append(
            {
                "path": path,
                "batch_index": int(batch_index) if batch_index is not None else None,
                "total_batches": payload.get("total_batches"),
                "model": payload.get("model"),
                "prompt_user": prompt_user,
                "batch_len": batch_len,
                "responses": parsed,
                "parse_status": parse_status,
            }
        )

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for foia_uid, batches in batches_by_foia.items():
        cand_rows = cand_groups.get(str(foia_uid))
        if cand_rows is None or cand_rows.empty:
            skipped += 1
            continue

        batches_sorted = sorted(batches, key=lambda b: (b["batch_index"] is None, b["batch_index"]))
        cursor = 0
        for b in batches_sorted:
            responses = b["responses"]
            if b["batch_len"] is None:
                if responses:
                    b_len = max(_as_float(r.get("candidate_index")) or 0 for r in responses)
                    b_len = int(b_len) if b_len else len(responses)
                else:
                    b_len = 0
            else:
                b_len = int(b["batch_len"])

            batch_slice = cand_rows.iloc[cursor: cursor + b_len] if b_len else cand_rows.iloc[0:0]
            cursor += b_len

            for item in responses:
                cidx = _as_float(item.get("candidate_index"))
                cidx = int(cidx) if cidx is not None else None
                if cidx is None or cidx <= 0 or cidx > len(batch_slice):
                    continue
                cand_row = batch_slice.iloc[cidx - 1]
                rows.append(
                    {
                        "foia_firm_uid": _stringify(cand_row.get("foia_firm_uid")),
                        "revelio_firm_uid": _stringify(cand_row.get("revelio_firm_uid")),
                        "foia_name": _stringify(cand_row.get("foia_name")),
                        "revelio_name": _stringify(cand_row.get("revelio_name")),
                        "score": cand_row.get("score"),
                        "match_type": cand_row.get("match_type"),
                        "batch_index": b.get("batch_index"),
                        "candidate_index": cidx,
                        "is_valid": _as_bool(item.get("is_valid")),
                        "confidence": _as_float(item.get("confidence")),
                        "reason": _stringify(item.get("reason")),
                        "model": b.get("model"),
                        "llm_file": str(b.get("path")),
                        "parse_status": b.get("parse_status"),
                    }
                )

    if not rows:
        raise SystemExit("No LLM judgments parsed into rows.")

    df = pd.DataFrame(rows)
    df.to_csv(out_all, index=False)

    judgments_eval = _mark_kept_matches(
        df,
        min_confidence=min_confidence,
        require_valid=require_valid,
    )
    matches = judgments_eval[judgments_eval["is_kept_match"]].copy()
    matches.to_csv(out_matches, index=False)

    wrote_crosswalk_valid = False
    wrote_crosswalk_all = False
    if (
        foia_raw_path
        and Path(foia_raw_path).exists()
        and Path(foia_fein_to_firm_path).exists()
        and Path(foia_firms_path).exists()
    ):
        fein_map = _read_any(str(Path(foia_fein_to_firm_path)))
        fein_map["foia_firm_uid"] = fein_map["foia_firm_uid"].astype("string")
        fein_map["fein_clean"] = fein_map["fein_clean"].astype("string")
        if "fein_year" in fein_map.columns:
            fein_map["fein_year"] = pd.to_numeric(fein_map["fein_year"], errors="coerce").astype("Int64")
        else:
            fein_map["fein_year"] = pd.NA
        fein_map = fein_map[["foia_firm_uid", "fein_clean", "fein_year"]].drop_duplicates()

        foia_firms_tbl = _read_any(str(Path(foia_firms_path)))
        foia_raw_tbl = _read_any(str(Path(foia_raw_path)))
        foia_raw_tbl["fein_clean"] = foia_raw_tbl[fein_col].map(_clean_fein).astype("string")
        foia_raw_tbl["fein_year"] = pd.to_numeric(foia_raw_tbl[year_col], errors="coerce").astype("Int64")
        foia_raw_tbl["original_name"] = foia_raw_tbl[foia_name_col].astype("string")
        foia_raw_tbl = foia_raw_tbl[["fein_clean", "fein_year", "original_name"]]
        foia_raw_tbl = foia_raw_tbl.dropna(subset=["fein_clean", "fein_year", "original_name"]).drop_duplicates()
        foia_with_firm = _map_foia_row_to_unique_firm(
            foia_raw=foia_raw_tbl,
            fein_map=fein_map,
            foia_firms=foia_firms_tbl,
        )

        crosswalk_all = _build_full_firm_status_crosswalk(
            foia_with_firm=foia_with_firm,
            foia_firm_uids=foia_firms_tbl["foia_firm_uid"],
            candidates=cand,
            judgments_eval=judgments_eval,
            score_threshold=score_threshold,
        )
        crosswalk_all.to_csv(out_crosswalk_all, index=False)
        wrote_crosswalk_all = True

        crosswalk_valid = crosswalk_all[crosswalk_all["crosswalk_validity_label"] == "valid_match"].copy()
        crosswalk_valid.to_csv(out_crosswalk, index=False)
        wrote_crosswalk_valid = True

    print(f"Wrote all judgments: {out_all} ({len(df):,} rows)")
    print(f"Wrote matches: {out_matches} ({len(matches):,} rows)")
    n_exc = int(matches["valid_by_exception"].fillna(False).sum()) if "valid_by_exception" in matches.columns else 0
    if n_exc:
        print(f"Included {n_exc:,} matches via state-mismatch exception rule.")
    if wrote_crosswalk_valid:
        print(f"Wrote FOIA->RCID valid crosswalk: {out_crosswalk} ({len(crosswalk_valid):,} rows)")
    if wrote_crosswalk_all:
        print(f"Wrote FOIA->RCID all-match crosswalk: {out_crosswalk_all} ({len(crosswalk_all):,} rows)")
    if not wrote_crosswalk_valid or not wrote_crosswalk_all:
        print(
            "Skipped one or both FOIA->RCID crosswalk outputs (need readable --foia, --foia-fein-to-firm, --foia-firms; "
            f"checked foia={foia_raw_path}, foia_fein_to_firm={foia_fein_to_firm_path}, foia_firms={foia_firms_path})"
        )
    if skipped:
        print(f"Skipped {skipped} FOIA firms not found in candidates.")


if __name__ == "__main__":
    main()
