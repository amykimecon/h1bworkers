#!/usr/bin/env python3
"""Compare matched FOIA firms (FEIN x year) across old and new merge outputs."""

import argparse
import csv
import os
import random
from decimal import Decimal, InvalidOperation
import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 


DEFAULT_OLD_PATH = f"{root}/data/int/good_match_ids_mar20.csv"
DEFAULT_NEW_PATH = (
    f"{root}/data/int/company_matching_jan28/llm_review_all_foia_to_rcid_crosswalk.csv"
)
DEFAULT_FOIA_FULL_PATH = f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv"
DEFAULT_RCID_ENTITIES_PATH = f"{root}/data/int/company_matching_jan28/revelio_rcid_entities.csv"


# revised code
import pandas as pd
import duckdb as ddb 
con = ddb.connect()

old = con.read_csv(DEFAULT_OLD_PATH)
new =con.read_csv(DEFAULT_NEW_PATH,         
                  strict_mode = False,
        ignore_errors = True,
        all_varchar = True)
full_foia = con.read_csv(DEFAULT_FOIA_FULL_PATH)
rcids = con.read_csv(DEFAULT_RCID_ENTITIES_PATH)

old_with_names = con.sql("SELECT foia_id, FEIN, lottery_year, old.rcid AS old_rcid, company AS old_company, n AS n, matchtype AS old_matchtype FROM old LEFT JOIN (SELECT rcid, n, company FROM rcids) AS rcids ON old.rcid = rcids.rcid")

new_with_names = con.sql("SELECT foia_firm_uid, fein_clean, fein_year, new.rcid, company, n, crosswalk_validity_label, firm_validity_label, revelio_firm_uid, candidate_index, reason FROM new LEFT JOIN (SELECT rcid, n, company FROM rcids) AS rcids ON new.rcid = rcids.rcid")

full_foia_no_dups = con.sql("SELECT FEIN, lottery_year, employer_name FROM full_foia WHERE NOT FEIN = '(b)(3) (b)(6) (b)(7)(c)' GROUP BY FEIN, lottery_year, employer_name")

all_merge = con.sql("""
                    SELECT * FROM
                    full_foia_no_dups AS full_foia
                    LEFT JOIN
                    old_with_names AS old ON full_foia.FEIN = old.FEIN AND full_foia.lottery_year = old.lottery_year
                    LEFT JOIN
                    new_with_names AS new ON full_foia.FEIN = new.fein_clean AND full_foia.lottery_year = new.fein_year
                    """)



def _require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for notebook helpers in compare_company_matching.py"
        ) from exc
    return pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare matched FOIA firms where firm unit is FEIN x year.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--root",
        default=os.path.expanduser("~"),
        help="Root folder that contains the data directory (default: home directory).",
    )
    parser.add_argument(
        "--old-path",
        default=DEFAULT_OLD_PATH,
        help=f"Old matching CSV path relative to --root (default: {DEFAULT_OLD_PATH}).",
    )
    parser.add_argument(
        "--new-path",
        default=DEFAULT_NEW_PATH,
        help=f"New matching CSV path relative to --root (default: {DEFAULT_NEW_PATH}).",
    )
    parser.add_argument(
        "--foia-full-path",
        default=DEFAULT_FOIA_FULL_PATH,
        help=(
            "Full FOIA source CSV path relative to --root for context in samples "
            f"(default: {DEFAULT_FOIA_FULL_PATH})."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of sample firms to print from each exclusive set (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--require-valid-new",
        action="store_true",
        help=(
            "If set, keeps only rows where is_valid is truthy in the new file "
            "(ignored if column is missing)."
        ),
    )
    # In notebook environments, ipykernel injects extra CLI args (e.g., -f ...json).
    # parse_known_args keeps this script usable in both CLI and notebooks.
    args, _unknown = parser.parse_known_args(argv)
    return args


def resolve_path(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def normalize_numlike(value):
    text = str(value).strip()
    if text == "":
        return ""
    try:
        num = Decimal(text)
    except InvalidOperation:
        return text
    if num == num.to_integral_value():
        return str(int(num))
    return format(num.normalize(), "f").rstrip("0").rstrip(".")


def normalize_year(value):
    text = normalize_numlike(value)
    if text == "":
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return ""


def is_truthy(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def require_columns(header, required, path):
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def make_foia_key(fein_value, year_value):
    fein = normalize_numlike(fein_value)
    year = normalize_year(year_value)
    if fein == "" or year == "":
        return None
    return f"{fein}|{year}"


def add_foia_firm_key(df, fein_col, year_col, key_col="foia_firm_key"):
    """Add normalized FEIN-year key column to a pandas DataFrame."""
    pd = _require_pandas()
    fein = df[fein_col].fillna("").map(normalize_numlike)
    year = df[year_col].fillna("").map(normalize_year)
    out = df.copy()
    out[key_col] = fein + "|" + year
    out.loc[(fein == "") | (year == ""), key_col] = pd.NA
    return out


def _to_bool_series(series):
    return series.fillna("").map(is_truthy)


def _prefix_nonkey_columns(df, prefix, key_col="foia_firm_key"):
    rename_map = {c: f"{prefix}{c}" for c in df.columns if c != key_col}
    return df.rename(columns=rename_map)


def _firm_level_summary(df, match_type_col, prefix, key_col="foia_firm_key"):
    pd = _require_pandas()
    work = df[df[key_col].notna()].copy()
    if match_type_col not in work.columns:
        work[match_type_col] = ""

    def uniq_pipe(values):
        vals = sorted({str(v) for v in values if str(v) != "" and str(v).lower() != "nan"})
        return "|".join(vals)

    summary = (
        work.groupby(key_col, dropna=True)
        .agg(
            **{
                f"{prefix}n_rows": (key_col, "size"),
                f"{prefix}match_types": (match_type_col, uniq_pipe),
            }
        )
        .reset_index()
    )
    summary[f"{prefix}matched"] = True
    return summary


def _normalize_rcid_series(series):
    pd = _require_pandas()
    out = series.fillna("").map(normalize_numlike)
    out = out.replace("", pd.NA)
    return out


def _firm_level_match_info(df, prefix, match_type_col, key_col="foia_firm_key"):
    pd = _require_pandas()
    work = df[df[key_col].notna()].copy()
    if "rcid_norm" not in work.columns:
        work["rcid_norm"] = pd.NA
    if match_type_col not in work.columns:
        work[match_type_col] = ""

    def uniq_pipe(values):
        vals = sorted({str(v) for v in values if str(v) != "" and str(v).lower() != "nan"})
        return "|".join(vals)

    agg = {
        f"{prefix}n_rows": (key_col, "size"),
        f"{prefix}n_unique_rcid": ("rcid_norm", lambda s: s.dropna().nunique()),
        f"{prefix}match_types": (match_type_col, uniq_pipe),
        f"{prefix}rcids": ("rcid_norm", uniq_pipe),
    }

    entity_cols = {
        "company": "companies",
        "ultimate_parent_company_name": "ultimate_parents",
        "ticker": "tickers",
        "factset_entity_id": "factset_entity_ids",
        "gvkey": "gvkeys",
    }
    for col, out_name in entity_cols.items():
        if col in work.columns:
            agg[f"{prefix}{out_name}"] = (col, uniq_pipe)

    summary = work.groupby(key_col, dropna=True).agg(**agg).reset_index()
    summary[f"{prefix}matched"] = True
    return summary


def _first_nonempty(series):
    for v in series:
        if v is None:
            continue
        s = str(v).strip()
        if s != "" and s.lower() != "nan":
            return v
    return ""


def _build_combined_match_detail(old_df, new_df):
    pd = _require_pandas()

    old_keep = [c for c in [
        "foia_firm_key",
        "rcid_norm",
        "old_match_type",
        "matchtype",
        "company",
        "ultimate_parent_company_name",
        "ticker",
        "factset_entity_id",
        "gvkey",
        "rcid",
    ] if c in old_df.columns]
    new_keep = [c for c in [
        "foia_firm_key",
        "rcid_norm",
        "new_match_type",
        "match_type",
        "company",
        "ultimate_parent_company_name",
        "ticker",
        "factset_entity_id",
        "gvkey",
        "rcid",
        "new_is_valid_bool",
        "is_valid",
    ] if c in new_df.columns]

    old_detail = old_df[old_keep].copy()
    old_detail["match_source"] = "old"
    old_detail = old_detail.rename(
        columns={
            "old_match_type": "match_type_standardized",
            "matchtype": "match_type_raw",
            "new_is_valid_bool": "is_valid_bool",
        }
    )

    new_detail = new_df[new_keep].copy()
    new_detail["match_source"] = "new"
    new_detail = new_detail.rename(
        columns={
            "new_match_type": "match_type_standardized",
            "match_type": "match_type_raw",
            "new_is_valid_bool": "is_valid_bool",
        }
    )

    for col in [
        "foia_firm_key",
        "rcid_norm",
        "match_type_standardized",
        "match_type_raw",
        "company",
        "ultimate_parent_company_name",
        "ticker",
        "factset_entity_id",
        "gvkey",
        "rcid",
        "is_valid_bool",
        "is_valid",
        "match_source",
    ]:
        if col not in old_detail.columns:
            old_detail[col] = pd.NA
        if col not in new_detail.columns:
            new_detail[col] = pd.NA

    combined = pd.concat([old_detail, new_detail], ignore_index=True)
    combined = combined[[
        "foia_firm_key",
        "match_source",
        "rcid_norm",
        "rcid",
        "match_type_standardized",
        "match_type_raw",
        "is_valid_bool",
        "is_valid",
        "company",
        "ultimate_parent_company_name",
        "ticker",
        "factset_entity_id",
        "gvkey",
    ]]
    return combined


def load_notebook_joins(
    root=os.path.expanduser("~"),
    old_path=DEFAULT_OLD_PATH,
    new_path=DEFAULT_NEW_PATH,
    foia_full_path=DEFAULT_FOIA_FULL_PATH,
    rcid_entities_path=DEFAULT_RCID_ENTITIES_PATH,
    require_valid_new=True,
):
    """
    Load raw FOIA data and left-join match outputs for notebook exploration.

    Returns dict with:
      - raw
      - old_matches
      - new_matches
      - raw_left_old
      - raw_left_new
      - raw_firm_level (raw + old/new match summaries at FEIN-year level)
      - raw_firm_level_match_info (raw + old/new match and entity info at FEIN-year level)
      - firm_match_detail (single detail table with old+new matches + firm summary)
      - old_firm_summary
      - new_firm_summary
      - old_firm_match_info
      - new_firm_match_info
    """
    pd = _require_pandas()

    old_abs = resolve_path(root, old_path)
    new_abs = resolve_path(root, new_path)
    raw_abs = resolve_path(root, foia_full_path)
    entity_abs = resolve_path(root, rcid_entities_path)

    for p in [old_abs, new_abs, raw_abs]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    raw = pd.read_csv(raw_abs, dtype=str, low_memory=False)
    old = pd.read_csv(old_abs, dtype=str, low_memory=False)
    new = pd.read_csv(new_abs, dtype=str, low_memory=False)
    entities = None
    if os.path.exists(entity_abs):
        entities = pd.read_csv(entity_abs, dtype=str, low_memory=False)

    required_raw_cols = ["FEIN", "lottery_year"]
    required_old_cols = ["FEIN", "lottery_year"]
    required_new_cols = ["fein_clean", "fein_year"]

    for cols, tab_name, df in [
        (required_raw_cols, "raw", raw),
        (required_old_cols, "old", old),
        (required_new_cols, "new", new),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {tab_name} dataset: {missing}")

    raw = add_foia_firm_key(raw, "FEIN", "lottery_year")
    old = add_foia_firm_key(old, "FEIN", "lottery_year")
    new = add_foia_firm_key(new, "fein_clean", "fein_year")

    if "matchtype" in old.columns:
        old["old_match_type"] = old["matchtype"]
    else:
        old["old_match_type"] = ""
    if "match_type" in new.columns:
        new["new_match_type"] = new["match_type"]
    else:
        new["new_match_type"] = ""

    if "is_valid" in new.columns:
        new["new_is_valid_bool"] = _to_bool_series(new["is_valid"])
        if require_valid_new:
            new = new[new["new_is_valid_bool"]].copy()
    else:
        new["new_is_valid_bool"] = pd.NA

    if "rcid" in old.columns:
        old["rcid_norm"] = _normalize_rcid_series(old["rcid"])
    else:
        old["rcid_norm"] = pd.NA
    if "rcid" in new.columns:
        new["rcid_norm"] = _normalize_rcid_series(new["rcid"])
    else:
        new["rcid_norm"] = pd.NA

    if entities is not None:
        if "rcid" not in entities.columns:
            raise ValueError(f"Missing columns in entity dataset: ['rcid'] ({entity_abs})")
        entities["rcid_norm"] = _normalize_rcid_series(entities["rcid"])
        keep_entity_cols = [
            "rcid_norm",
            "company",
            "ultimate_parent_company_name",
            "factset_entity_id",
            "gvkey",
            "ticker",
        ]
        keep_entity_cols = [c for c in keep_entity_cols if c in entities.columns]
        entity_lookup = (
            entities.loc[entities["rcid_norm"].notna(), keep_entity_cols]
            .drop_duplicates(subset=["rcid_norm"])
            .copy()
        )
        old = old.merge(entity_lookup, on="rcid_norm", how="left")
        new = new.merge(entity_lookup, on="rcid_norm", how="left")

    raw_left_old = raw.merge(
        _prefix_nonkey_columns(old, "old_"),
        on="foia_firm_key",
        how="left",
    )
    raw_left_new = raw.merge(
        _prefix_nonkey_columns(new, "new_"),
        on="foia_firm_key",
        how="left",
    )

    old_firm_summary = _firm_level_summary(old, "old_match_type", prefix="old_")
    new_firm_summary = _firm_level_summary(new, "new_match_type", prefix="new_")
    raw_firm_level = (
        raw.merge(old_firm_summary, on="foia_firm_key", how="left")
        .merge(new_firm_summary, on="foia_firm_key", how="left")
    )
    raw_firm_level["old_matched"] = raw_firm_level["old_matched"].fillna(False)
    raw_firm_level["new_matched"] = raw_firm_level["new_matched"].fillna(False)

    old_firm_match_info = _firm_level_match_info(old, prefix="old_", match_type_col="old_match_type")
    new_firm_match_info = _firm_level_match_info(new, prefix="new_", match_type_col="new_match_type")
    raw_firm_level_match_info = (
        raw.merge(old_firm_match_info, on="foia_firm_key", how="left")
        .merge(new_firm_match_info, on="foia_firm_key", how="left")
    )
    raw_firm_level_match_info["old_matched"] = raw_firm_level_match_info["old_matched"].fillna(False)
    raw_firm_level_match_info["new_matched"] = raw_firm_level_match_info["new_matched"].fillna(False)

    firm_base_cols = [c for c in ["foia_firm_key", "FEIN", "lottery_year", "employer_name", "state"] if c in raw.columns]
    firm_level_base = (
        raw[firm_base_cols]
        .groupby("foia_firm_key", dropna=True)
        .agg({c: _first_nonempty for c in firm_base_cols if c != "foia_firm_key"})
        .reset_index()
    )

    combined_match_detail = _build_combined_match_detail(old, new)
    firm_match_detail = (
        combined_match_detail
        .merge(firm_level_base, on="foia_firm_key", how="left")
        .merge(
            raw_firm_level_match_info.drop_duplicates(subset=["foia_firm_key"]),
            on="foia_firm_key",
            how="left",
            suffixes=("", "_firm_level"),
        )
    )

    return {
        "raw": raw,
        "old_matches": old,
        "new_matches": new,
        "rcid_entities": entities,
        "raw_left_old": raw_left_old,
        "raw_left_new": raw_left_new,
        "raw_firm_level": raw_firm_level,
        "raw_firm_level_match_info": raw_firm_level_match_info,
        "firm_match_detail": firm_match_detail,
        "old_firm_summary": old_firm_summary,
        "new_firm_summary": new_firm_summary,
        "old_firm_match_info": old_firm_match_info,
        "new_firm_match_info": new_firm_match_info,
    }


def load_foia_firms(path, fein_col, year_col, require_valid=False):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        require_columns(header, [fein_col, year_col], path)

        firms = set()
        examples = {}
        row_count = 0
        skipped_invalid = 0
        missing_key_count = 0

        context_cols = [
            fein_col,
            year_col,
            "matchtype",
            "original_name",
            "match_type",
            "is_valid",
            "foia_firm_uid",
            "rcid",
        ]

        for row in reader:
            row_count += 1
            if require_valid and "is_valid" in row and not is_truthy(row["is_valid"]):
                skipped_invalid += 1
                continue

            key = make_foia_key(row.get(fein_col, ""), row.get(year_col, ""))
            if key is None:
                missing_key_count += 1
                continue

            firms.add(key)
            if key not in examples:
                examples[key] = {
                    k: row.get(k, "") for k in context_cols if k in row and row.get(k, "")
                }

    return {
        "row_count": row_count,
        "skipped_invalid": skipped_invalid,
        "missing_key_count": missing_key_count,
        "firms": firms,
        "examples": examples,
    }


def load_foia_context(path):
    if not os.path.exists(path):
        return {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        require_columns(header, ["FEIN", "lottery_year"], path)

        out = {}
        for row in reader:
            key = make_foia_key(row.get("FEIN", ""), row.get("lottery_year", ""))
            if key is None or key in out:
                continue
            out[key] = {
                "employer_name": row.get("employer_name", ""),
                "state": row.get("state", ""),
                "status_type": row.get("status_type", ""),
            }
    return out


def merge_examples(primary_examples, context_examples):
    out = {}
    for k, v in primary_examples.items():
        merged = {}
        if k in context_examples:
            merged.update({x: y for x, y in context_examples[k].items() if y != ""})
        merged.update(v)
        out[k] = merged
    return out


def print_sample(title, keys, examples, sample_size, rng):
    print(f"\n{title}: {len(keys)}")
    if not keys:
        print("  (none)")
        return

    for key in rng.sample(sorted(keys), min(sample_size, len(keys))):
        fein, year = key.split("|", 1)
        context = examples.get(key, {})
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            print(f"  FEIN={fein}, year={year} | {context_str}")
        else:
            print(f"  FEIN={fein}, year={year}")


def main(argv=None):
    args = parse_args(argv)
    rng = random.Random(args.seed)

    old_path = resolve_path(args.root, args.old_path)
    new_path = resolve_path(args.root, args.new_path)
    full_path = resolve_path(args.root, args.foia_full_path)

    if not os.path.exists(old_path):
        raise FileNotFoundError(f"Old file not found: {old_path}")
    if not os.path.exists(new_path):
        raise FileNotFoundError(f"New file not found: {new_path}")

    old_data = load_foia_firms(old_path, fein_col="FEIN", year_col="lottery_year")
    new_data = load_foia_firms(
        new_path,
        fein_col="fein_clean",
        year_col="fein_year",
        require_valid=args.require_valid_new,
    )

    foia_context = load_foia_context(full_path)
    old_examples = merge_examples(old_data["examples"], foia_context)
    new_examples = merge_examples(new_data["examples"], foia_context)

    old_firms = old_data["firms"]
    new_firms = new_data["firms"]
    overlap = old_firms & new_firms
    old_only = old_firms - new_firms
    new_only = new_firms - old_firms
    union = old_firms | new_firms

    print("=== FOIA Firm Matching Comparison (FEIN x year) ===")
    print(f"Old file: {old_path}")
    print(f"  rows={old_data['row_count']}")
    print(f"  rows_missing_fein_or_year={old_data['missing_key_count']}")
    print(f"  unique_foia_firms_matched={len(old_firms)}")

    print(f"New file: {new_path}")
    print(f"  rows={new_data['row_count']}")
    if args.require_valid_new:
        print(f"  rows_skipped_invalid={new_data['skipped_invalid']}")
    print(f"  rows_missing_fein_or_year={new_data['missing_key_count']}")
    print(f"  unique_foia_firms_matched={len(new_firms)}")

    if os.path.exists(full_path):
        print(f"Context file loaded: {full_path}")
    else:
        print(f"Context file not found (samples will be limited): {full_path}")

    print("\n=== Overlap ===")
    print(f"Shared FEIN-year firms: {len(overlap)}")
    print(f"Old only FEIN-year firms: {len(old_only)}")
    print(f"New only FEIN-year firms: {len(new_only)}")
    print(
        f"Jaccard overlap (shared / union): "
        f"{(len(overlap) / len(union)) if union else 0:.4f}"
    )

    print_sample(
        "Sample FEIN-year firms in old but not new",
        old_only,
        old_examples,
        args.sample_size,
        rng,
    )
    print_sample(
        "Sample FEIN-year firms in new but not old",
        new_only,
        new_examples,
        args.sample_size,
        rng,
    )


if __name__ == "__main__":
    main()
