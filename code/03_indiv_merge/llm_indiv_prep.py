#!/usr/bin/env python3
"""
llm_indiv_prep.py — Format baseline H-1B×Revelio merge output into structured
LLM prompts for probabilistic individual matching.

One JSONL record per (firm × lottery_year) group. Each record contains a system
prompt and a user prompt that can be submitted directly to an OpenAI-compatible
chat API. Groups exceeding size thresholds are skipped and recorded in the
metadata parquet.

Reads:
  - merge_filt_baseline_{run_tag}.parquet  (candidate pairs, post-scoring)
  - foia_indiv_{run_tag}.parquet           (for employer names per firm_uid)
  - rev_educ_long_{run_tag}.parquet        (raw education history per user)
  - merged_pos_clean_{run_tag}.parquet     (raw position history per user)
  - wrds_users_{run_tag}.parquet           (optional: degree_raw/field_raw)

Writes:
  - llm_indiv_prompts_{run_tag}.jsonl          (one record per eligible group)
  - llm_indiv_prompts_{run_tag}_meta.parquet   (all groups, including skipped)
"""

import sys
sys.argv = sys.argv[:1]  # iPython-safe: prevent argparse from consuming notebook args

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
import yaml


# ─── Config ──────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "llm_indiv_prep.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _cfg(cfg: Dict[str, Any], *keys: str, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _resolve_path(template: str, run_tag: str, root: str) -> Path:
    """Substitute {root} and ${run_tag} placeholders in a path template."""
    return Path(template.replace("{root}", root).replace("${run_tag}", run_tag))


# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at linking de-identified H-1B visa records to LinkedIn user profiles.

You will be given:
1. A set of H-1B FOIA applications for a single employer and fiscal year. These are de-identified — no applicant names are provided. Each application includes year of birth, gender, and country of birth.
2. A set of candidate LinkedIn (Revelio) users who have been pre-screened as plausible matches for at least one application at this employer. Each profile includes the user's full name, raw education history, and raw position history.

Your task: assign a probabilistic mapping from each FOIA application to the most likely matching Revelio user.

Important constraints and context:
- Approximately 50% of H-1B applications have no true match in the Revelio data (the person may not have a LinkedIn profile or may not appear in the dataset). Err strongly on the side of returning null — only match when there is clear, specific evidence from multiple signals.
- Each Revelio user can be assigned to at most one FOIA application.
- Country of birth is the strongest matching signal; year of birth is second; gender is third. A good match requires consistency across multiple signals.
- Position history should confirm employment at this employer starting between 0 and 3 years (in some cases as much as 4 years) before the lottery, and not ending before the lottery (in March of the fiscal year).

Return your response as a JSON array with exactly one object per FOIA application, in the same order they were listed:
[
  {
    "foia_indiv_id": "<id>",
    "matched_user_id": "<user_id or null>",
    "confidence": <float 0.0–1.0>,
    "reasoning": "<brief explanation, max 30 words>"
  }
]
Return JSON only. No other text."""


# ─── Prompt formatting helpers ────────────────────────────────────────────────

def _fmt_gender(female_ind) -> Optional[str]:
    """Convert female_ind (0/1/null) to a readable string, or None if missing."""
    if female_ind is None:
        return None
    try:
        if pd.isna(female_ind):
            return None
    except (TypeError, ValueError):
        pass
    return "Female" if int(female_ind) == 1 else "Male"


def _fmt_year(date_val) -> Optional[str]:
    """Extract 4-digit year from a date string, date object, or year int. Returns None if missing."""
    if date_val is None:
        return None
    try:
        if pd.isna(date_val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(date_val).strip()
    # handles 'YYYY-MM-DD', 'YYYY', integers like 2015, etc.
    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return None


def _fmt_text(val: Any) -> str:
    """Convert nullable values to clean printable text."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def _fmt_app(idx: int, row: dict) -> str:
    """Format one FOIA application as a text block for the user prompt."""
    lines = [f"[APP-{idx}] foia_indiv_id: {row['foia_indiv_id']}"]
    yob = row.get("yob")
    if yob is not None:
        try:
            if not pd.isna(yob):
                lines.append(f"  Year of Birth: {int(yob)}")
        except (TypeError, ValueError):
            pass
    gender = _fmt_gender(row.get("female_ind"))
    if gender:
        lines.append(f"  Gender: {gender}")
    country = row.get("foia_country")
    if country and not (isinstance(country, float) and pd.isna(country)):
        lines.append(f"  Country of Birth: {country}")
    return "\n".join(lines)


def _fmt_edu_record(rec: dict) -> Optional[str]:
    """Format one education record as a single list item. Returns None if empty."""
    univ = _fmt_text(rec.get("university_raw"))
    degree_raw = _fmt_text(rec.get("degree_raw"))
    field_raw = _fmt_text(rec.get("field_raw"))
    degree_clean = _fmt_text(rec.get("degree_clean"))

    if degree_raw and field_raw:
        deg = f"{degree_raw} | {field_raw}"
    elif degree_raw:
        deg = degree_raw
    elif field_raw:
        deg = field_raw
    else:
        # Fallback if raw fields are unavailable.
        deg = degree_clean

    start = _fmt_year(rec.get("ed_startdate"))
    end = _fmt_year(rec.get("ed_enddate"))
    period = f"{start or '?'}–{end or 'present'}" if (start or end) else None
    parts = [p for p in [univ, deg, period] if p]
    return "    - " + " | ".join(parts) if parts else None


def _fmt_pos_record(rec: dict, highlight: bool = False) -> Optional[str]:
    """Format one position record as a single list item. Returns None if empty."""
    company = _fmt_text(rec.get("company_raw"))
    title   = _fmt_text(rec.get("title_raw"))
    country = _fmt_text(rec.get("country"))
    start   = _fmt_year(rec.get("startdate"))
    end     = _fmt_year(rec.get("enddate"))
    period  = f"{start or '?'}–{end or 'present'}" if (start or end) else None
    parts   = [p for p in [company, title, country, period] if p]
    if not parts:
        return None
    # Positions matched to this employer's Revelio company ID are marked with >>
    prefix = "  >> " if highlight else "    - "
    suffix = "  [<-- matched employer]" if highlight else ""
    return prefix + " | ".join(parts) + suffix


def _fmt_user(
    idx: int,
    user_id: str,
    fullname: Optional[str],
    edu_records: List[dict],
    pos_records: List[dict],
    highlight_rcids: Optional[set] = None,
    linkedin_url: Optional[str] = None,
) -> str:
    """Format one Revelio user profile as a text block for the user prompt.

    highlight_rcids: set of rcids (Revelio company IDs) used in the baseline match
    for this user. Positions whose rcid is in this set are marked with >>.
    """
    lines = [f"[USER-{idx}] user_id: {user_id}"]
    if fullname and fullname.strip():
        name_str = fullname.strip()
        if linkedin_url and linkedin_url.strip():
            name_str += f" [{linkedin_url.strip()}]"
        lines.append(f"  Name: {name_str}")
    # Education history (chronological)
    edu_lines = [_fmt_edu_record(r) for r in edu_records]
    edu_lines = [l for l in edu_lines if l]
    if edu_lines:
        lines.append(f"  Education ({len(edu_lines)}):")
        lines.extend(edu_lines)
    # Position history (most recent first); matched-employer positions are highlighted
    pos_lines = [
        _fmt_pos_record(
            r,
            highlight=(
                highlight_rcids is not None
                and r.get("rcid") is not None
                and str(r["rcid"]) in highlight_rcids
            ),
        )
        for r in pos_records
    ]
    pos_lines = [l for l in pos_lines if l]
    if pos_lines:
        # Intentionally include full position history (no sampling/truncation).
        lines.append(f"  Positions ({len(pos_lines)}):")
        lines.extend(pos_lines)
    return "\n".join(lines)


def _format_user_prompt(
    apps: List[dict],
    users: List[dict],
    edu_by_user: Dict[str, List[dict]],
    pos_by_user: Dict[str, List[dict]],
    employer_name: str,
    lottery_year: int,
    rcids_by_user: Optional[Dict[str, set]] = None,
    linkedin_url_by_user: Optional[Dict[str, str]] = None,
) -> str:
    """Build the full user-side prompt for one firm × lottery_year group.

    rcids_by_user: mapping user_id → set of rcids that appear in the baseline for
    that user. Used to highlight matched-employer positions with >>.
    linkedin_url_by_user: mapping user_id → LinkedIn profile URL string.
    """
    lines = [
        f"## Employer: {employer_name} | Lottery Year: {lottery_year}",
        f"## {len(apps)} FOIA Application(s) | {len(users)} Candidate Revelio User(s)",
        "",
        "### FOIA Applications (de-identified)",
    ]
    for i, app in enumerate(apps, 1):
        lines.append(_fmt_app(i, app))
        lines.append("")

    lines += ["### Candidate Revelio Users"]
    for i, user in enumerate(users, 1):
        uid = str(user["user_id"])
        edu = edu_by_user.get(uid, [])
        pos = pos_by_user.get(uid, [])
        highlight_rcids = rcids_by_user.get(uid) if rcids_by_user else None
        linkedin_url = linkedin_url_by_user.get(uid) if linkedin_url_by_user else None
        lines.append(_fmt_user(i, uid, user.get("fullname"), edu, pos,
                               highlight_rcids=highlight_rcids,
                               linkedin_url=linkedin_url))
        lines.append("")

    return "\n".join(lines).rstrip()


# ─── Main pipeline ────────────────────────────────────────────────────────────

def build_llm_prompts():
    t0 = time.time()

    # ── Load config ────────────────────────────────────────────────────────
    cfg = _load_config(CONFIG_PATH)
    run_tag = cfg.get("run_tag", "feb2026")

    # Resolve root from config.py (handles user-specific paths)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import root  # noqa: E402

    def rp(key: str) -> Path:
        tmpl = _cfg(cfg, "paths", key)
        if not tmpl:
            raise ValueError(f"Missing required config key: paths.{key}")
        return _resolve_path(tmpl, run_tag, root)

    baseline_path  = rp("baseline_parquet")
    foia_indiv_path = rp("foia_indiv_parquet")
    educ_path      = rp("rev_educ_long_parquet")
    pos_path       = rp("merged_pos_clean_parquet")
    wrds_users_tmpl = _cfg(
        cfg,
        "paths",
        "wrds_users_parquet",
        default="{root}/data/int/wrds_users_${run_tag}.parquet",
    )
    wrds_users_path = _resolve_path(wrds_users_tmpl, run_tag, root)
    out_dir        = _resolve_path(_cfg(cfg, "paths", "output_dir", default="{root}/data/int/"), run_tag, root)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    max_cands    = _cfg(cfg, "options", "max_candidates_per_group", default=30)
    max_apps_cfg = _cfg(cfg, "options", "max_apps_per_group", default=50)
    testing      = bool(_cfg(cfg, "testing", "enabled", default=False))
    n_test       = int(_cfg(cfg, "testing", "n_groups", default=5))
    random_seed  = int(_cfg(cfg, "testing", "random_seed", default=0))

    print(f"[llm_indiv_prep] run_tag={run_tag}  testing={testing}")
    print(f"  baseline:  {baseline_path}")
    print(f"  foia_indiv:{foia_indiv_path}")
    print(f"  educ:      {educ_path}")
    print(f"  pos:       {pos_path}")
    print(f"  wrds_users:{wrds_users_path}")

    # ── Step 1: Load parquets into DuckDB ─────────────────────────────────
    print("\n[1/4] Loading parquets into DuckDB views...")
    t1 = time.time()
    con = duckdb.connect()
    con.execute(f"CREATE VIEW baseline   AS SELECT * FROM read_parquet('{baseline_path}')")
    con.execute(f"CREATE VIEW foia_indiv AS SELECT * FROM read_parquet('{foia_indiv_path}')")
    con.execute(f"CREATE VIEW rev_educ   AS SELECT * FROM read_parquet('{educ_path}')")
    con.execute(f"CREATE VIEW merged_pos AS SELECT * FROM read_parquet('{pos_path}')")
    print(f"  Views registered in {time.time()-t1:.1f}s")

    # Quick size check
    n_baseline = con.execute("SELECT COUNT(*) FROM baseline").fetchone()[0]
    print(f"  Baseline rows: {n_baseline:,}")

    # ── Step 2: Build group-level summary, apply size filters ─────────────
    print("\n[2/4] Computing group sizes and applying size filters...")
    t2 = time.time()
    groups_df = con.execute("""
        SELECT
            foia_firm_uid,
            lottery_year,
            COUNT(DISTINCT foia_indiv_id) AS n_apps,
            COUNT(DISTINCT user_id)       AS n_candidates
        FROM baseline
        GROUP BY foia_firm_uid, lottery_year
        ORDER BY n_candidates ASC
    """).df()

    print(f"  Total firm×year groups:  {len(groups_df):,}")
    print(f"  Total FOIA apps:         {groups_df['n_apps'].sum():,}")

    # Tag skipped groups
    cand_skip = groups_df["n_candidates"] > max_cands
    app_skip  = groups_df["n_apps"] > max_apps_cfg
    groups_df["skipped"] = cand_skip | app_skip
    groups_df["skip_reason"] = ""
    groups_df.loc[cand_skip, "skip_reason"] = f"n_candidates>{max_cands}"
    groups_df.loc[app_skip & ~cand_skip, "skip_reason"] = f"n_apps>{max_apps_cfg}"

    eligible = groups_df[~groups_df["skipped"]].copy().reset_index(drop=True)
    print(f"  Eligible groups:         {len(eligible):,}  "
          f"(skipped {groups_df['skipped'].sum():,} over-size groups)")
    print(f"  Eligible FOIA apps:      {eligible['n_apps'].sum():,}")
    print(f"  Group size stats  — candidates: "
          f"median={eligible['n_candidates'].median():.0f}, "
          f"max={eligible['n_candidates'].max()}")

    # Testing: sample a small random subset
    if testing:
        eligible = eligible.sample(n=min(n_test, len(eligible)),
                                   random_state=random_seed).reset_index(drop=True)
        print(f"  [TESTING] Sampled {len(eligible)} groups (seed={random_seed})")

    eligible["group_id"] = (eligible["foia_firm_uid"].astype(str)
                            + "__" + eligible["lottery_year"].astype(str))
    print(f"  Group size computed in {time.time()-t2:.1f}s")

    if len(eligible) == 0:
        print("  No eligible groups — exiting.")
        return

    # ── Step 3: Preload supporting data for eligible groups ───────────────
    print("\n[3/4] Preloading employer names, education, and position records...")
    t3 = time.time()

    # Employer names (one per firm_uid, from foia_indiv)
    emp_df = con.execute("""
        SELECT foia_firm_uid, MAX(employer_name) AS employer_name
        FROM foia_indiv
        GROUP BY foia_firm_uid
    """).df()
    emp_map: Dict[str, str] = dict(
        zip(emp_df["foia_firm_uid"].astype(str), emp_df["employer_name"])
    )
    print(f"  Employer names loaded: {len(emp_map):,}")

    # Create a temp table of eligible firm×year pairs for filtering
    eligible_fw = eligible[["foia_firm_uid", "lottery_year"]].copy()
    con.execute("""
        CREATE TEMP TABLE _eligible_fw AS
        SELECT foia_firm_uid, lottery_year FROM eligible_fw
    """)

    # Get all user_ids appearing in eligible groups
    candidate_users_df = con.execute("""
        SELECT DISTINCT b.user_id
        FROM baseline b
        JOIN _eligible_fw g USING (foia_firm_uid, lottery_year)
        WHERE b.user_id IS NOT NULL
    """).df()
    print(f"  Unique Revelio users in eligible groups: {len(candidate_users_df):,}")

    # Education records (chronological order)
    wrds_available = wrds_users_path.exists()
    if wrds_available:
        print("  wrds_users found; enriching education with degree_raw/field_raw.")
        edu_df = con.execute(f"""
            WITH edu_ranked AS (
                SELECT
                    e.user_id,
                    e.education_number,
                    e.university_raw,
                    e.degree_clean,
                    e.ed_startdate,
                    e.ed_enddate,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.user_id, e.education_number
                        ORDER BY e.matchscore DESC NULLS LAST, e.matchtype DESC NULLS LAST
                    ) AS edu_rank
                FROM rev_educ e
                JOIN candidate_users_df c ON e.user_id = c.user_id
            ),
            wrds_edu AS (
                SELECT
                    w.user_id,
                    w.education_number,
                    w.degree_raw,
                    w.field_raw,
                    ROW_NUMBER() OVER (
                        PARTITION BY w.user_id, w.education_number
                        ORDER BY w.updated_dt DESC NULLS LAST
                    ) AS wrds_rank
                FROM read_parquet('{wrds_users_path}') w
                JOIN candidate_users_df c ON w.user_id = c.user_id
            )
            SELECT
                e.user_id,
                e.education_number,
                e.university_raw,
                e.degree_clean,
                w.degree_raw,
                w.field_raw,
                e.ed_startdate,
                e.ed_enddate
            FROM edu_ranked e
            LEFT JOIN wrds_edu w
                ON e.user_id = w.user_id
               AND e.education_number = w.education_number
               AND w.wrds_rank = 1
            WHERE e.edu_rank = 1
            ORDER BY e.user_id, e.education_number
        """).df()
    else:
        print("  wrds_users not found; falling back to degree_clean only.")
        edu_df = con.execute("""
            WITH edu_ranked AS (
                SELECT
                    e.user_id,
                    e.education_number,
                    e.university_raw,
                    e.degree_clean,
                    e.ed_startdate,
                    e.ed_enddate,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.user_id, e.education_number
                        ORDER BY e.matchscore DESC NULLS LAST, e.matchtype DESC NULLS LAST
                    ) AS edu_rank
                FROM rev_educ e
                JOIN candidate_users_df c ON e.user_id = c.user_id
            )
            SELECT
                user_id,
                education_number,
                university_raw,
                degree_clean,
                NULL::VARCHAR AS degree_raw,
                NULL::VARCHAR AS field_raw,
                ed_startdate,
                ed_enddate
            FROM edu_ranked
            WHERE edu_rank = 1
            ORDER BY user_id, education_number
        """).df()
    # Group into dict: user_id → list of record dicts
    edu_by_user: Dict[str, List[dict]] = {}
    for row in edu_df.to_dict("records"):
        edu_by_user.setdefault(str(row["user_id"]), []).append(row)
    print(f"  Education records loaded: {len(edu_df):,}")

    # LinkedIn profile URLs (from wrds_users, if available)
    linkedin_url_by_user: Dict[str, str] = {}
    if wrds_available:
        url_df = con.execute(f"""
            SELECT DISTINCT ON (user_id) user_id, profile_linkedin_url
            FROM read_parquet('{wrds_users_path}')
            WHERE profile_linkedin_url IS NOT NULL AND profile_linkedin_url != ''
        """).df()
        for row in url_df.to_dict("records"):
            linkedin_url_by_user[str(row["user_id"])] = row["profile_linkedin_url"]
        print(f"  LinkedIn URLs loaded: {len(linkedin_url_by_user):,}")

    # Position records (most recent first); include rcid so matched-employer
    # positions can be highlighted in the prompt.
    pos_df = con.execute("""
        SELECT p.user_id, p.rcid, p.company_raw, p.title_raw, p.country, p.startdate, p.enddate
        FROM merged_pos p
        JOIN candidate_users_df c ON p.user_id = c.user_id
        ORDER BY p.user_id, p.startdate DESC NULLS LAST
    """).df()
    pos_by_user: Dict[str, List[dict]] = {}
    for row in pos_df.to_dict("records"):
        pos_by_user.setdefault(str(row["user_id"]), []).append(row)
    print(f"  Position records loaded:  {len(pos_df):,}")

    # Build group×user → set of matched rcids from baseline (used to highlight positions).
    # Keyed by (foia_firm_uid, lottery_year, user_id) so each prompt only highlights
    # positions matched to *that* employer, not any employer the user appears with.
    rcids_raw = con.execute("""
        SELECT DISTINCT foia_firm_uid, lottery_year, user_id, rcid
        FROM baseline
        WHERE user_id IS NOT NULL AND rcid IS NOT NULL
    """).df()
    rcids_by_group_user: Dict[tuple, set] = {}
    for row in rcids_raw.to_dict("records"):
        key = (str(row["foia_firm_uid"]), int(row["lottery_year"]), str(row["user_id"]))
        rcids_by_group_user.setdefault(key, set()).add(str(row["rcid"]))
    print(f"  Matched rcids loaded for {len(rcids_by_group_user):,} group×user pairs")
    print(f"  Supporting data loaded in {time.time()-t3:.1f}s")

    # ── Step 4: Format prompts and write JSONL ─────────────────────────────
    print(f"\n[4/4] Formatting {len(eligible):,} prompts...")
    t4 = time.time()

    out_jsonl = Path(out_dir) / f"llm_indiv_prompts_{run_tag}.jsonl"
    n_written = 0

    with out_jsonl.open("w") as f_out:
        for idx, grp in enumerate(eligible.itertuples(index=False), 1):
            firm_uid  = grp.foia_firm_uid
            year      = grp.lottery_year
            group_id  = grp.group_id

            # Fetch per-group FOIA apps and Revelio users from baseline
            apps_df = con.execute(f"""
                SELECT DISTINCT foia_indiv_id, foia_country, female_ind, yob
                FROM baseline
                WHERE foia_firm_uid = '{firm_uid}' AND lottery_year = {year}
                ORDER BY foia_indiv_id
            """).df()

            users_df = con.execute(f"""
                SELECT DISTINCT user_id, fullname
                FROM baseline
                WHERE foia_firm_uid = '{firm_uid}' AND lottery_year = {year}
                ORDER BY user_id
            """).df()

            apps  = apps_df.to_dict("records")
            users = users_df.to_dict("records")
            employer_name = emp_map.get(str(firm_uid), str(firm_uid))

            # Build per-group rcid lookup for highlighting matched-employer positions
            group_rcids_by_user: Dict[str, set] = {}
            for uid in [str(u["user_id"]) for u in users]:
                s = rcids_by_group_user.get((str(firm_uid), int(year), uid))
                if s:
                    group_rcids_by_user[uid] = s

            user_prompt = _format_user_prompt(
                apps, users, edu_by_user, pos_by_user, employer_name, int(year),
                rcids_by_user=group_rcids_by_user,
                linkedin_url_by_user=linkedin_url_by_user,
            )

            record = {
                "group_id":       group_id,
                "foia_firm_uid":  str(firm_uid),
                "lottery_year":   int(year),
                "employer_name":  employer_name,
                "n_apps":         len(apps),
                "n_candidates":   len(users),
                "system_prompt":  SYSTEM_PROMPT,
                "user_prompt":    user_prompt,
            }
            f_out.write(json.dumps(record) + "\n")
            n_written += 1
            
            if testing: 
                print(f"\n--- Prompt for group {idx} / {len(eligible)} ---")
                print(user_prompt)

            if idx % 1000 == 0 or idx == len(eligible):
                elapsed = time.time() - t0
                rate = idx / (time.time() - t4)
                print(f"  [{idx:>6}/{len(eligible)}]  {n_written:,} written  "
                      f"({elapsed:.0f}s elapsed, {rate:.0f} groups/s)")

    print(f"\n  JSONL written: {out_jsonl}  ({n_written:,} records)")

    # ── Write metadata parquet (all groups, incl. skipped) ─────────────────
    out_meta = Path(out_dir) / f"llm_indiv_prompts_{run_tag}_meta.parquet"
    groups_df["group_id"] = (groups_df["foia_firm_uid"].astype(str)
                             + "__" + groups_df["lottery_year"].astype(str))
    groups_df.to_parquet(out_meta, index=False)
    print(f"  Metadata parquet: {out_meta}")
    print(f"  Total groups: {len(groups_df):,}  |  "
          f"Eligible (written): {(~groups_df['skipped']).sum():,}  |  "
          f"Skipped: {groups_df['skipped'].sum():,}")

    print(f"\n[llm_indiv_prep] Finished in {time.time()-t0:.1f}s")


# Run when executed as a script or sourced in iPython
build_llm_prompts()
