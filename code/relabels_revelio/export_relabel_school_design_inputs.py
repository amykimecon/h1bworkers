"""Export relabel treated/control school inputs for external individual panels.

The output is intentionally design-facing rather than outcome-facing: one file
keeps matched treated-control pairs, and one long file has a row per role
(treated/control) within each pair. FICE is exported as the six-digit OPEID stem
from IPEDS HD files, which is the closest local crosswalk available here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import duckdb as ddb
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import f1_foia.econ_relabels_opt_usage as base
from relabels_revelio import relabel_events_generalized as generalized


DEFAULT_OUTPUT_DIR = (
    Path(base.root)
    / "h1bworkers"
    / "code"
    / "output"
    / "relabel_indiv"
    / "texas_school_design_inputs"
)


def _cip_dotted(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return pd.NA
    digits = digits.zfill(6)
    return f"{digits[:2]}.{digits[2:]}"


def _read_hd_file(path: Path) -> pd.DataFrame:
    usecols = ["UNITID", "INSTNM", "OPEID", "STABBR"]
    for encoding in ("utf-8", "latin1"):
        try:
            return pd.read_csv(path, dtype=str, usecols=lambda c: c in usecols, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, usecols=lambda c: c in usecols, encoding="latin1")


def _load_institution_lookup(hd_dir: Path) -> pd.DataFrame:
    """Return one row per UNITID with school name, OPEID, and OPEID6/FICE."""
    hd_files = sorted(hd_dir.glob("hd*.csv"), reverse=True)
    frames: list[pd.DataFrame] = []
    for path in hd_files:
        year_digits = "".join(ch for ch in path.stem if ch.isdigit())
        year = int(year_digits[-4:]) if len(year_digits) >= 4 else pd.NA
        frame = _read_hd_file(path)
        frame["hd_year"] = year
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["unitid", "school_name", "state", "opeid8", "fice", "hd_year"])

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"UNITID": "unitid", "INSTNM": "school_name", "OPEID": "opeid8", "STABBR": "state"})
    out["unitid"] = pd.to_numeric(out["unitid"], errors="coerce").astype("Int64")
    out["opeid8"] = out["opeid8"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    out["fice"] = out["opeid8"].str.slice(0, 6)
    out["hd_year"] = pd.to_numeric(out["hd_year"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["unitid"]).copy()
    out = out.sort_values(["unitid", "hd_year"], ascending=[True, False], kind="mergesort")
    return out.drop_duplicates("unitid", keep="first").reset_index(drop=True)


def _attach_role_institution_info(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
    *,
    id_col: str,
    prefix: str,
) -> pd.DataFrame:
    out = df.copy()
    unit_col = f"{prefix}_unitid"
    out[unit_col] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
    rename = {
        "unitid": unit_col,
        "school_name": f"{prefix}_school_name",
        "state": f"{prefix}_state",
        "opeid8": f"{prefix}_opeid8",
        "fice": f"{prefix}_fice",
    }
    cols = ["unitid", "school_name", "state", "opeid8", "fice"]
    return out.merge(lookup[cols].rename(columns=rename), on=unit_col, how="left")


def _prepare_pair_export(matches: pd.DataFrame, lookup: pd.DataFrame, control_group: str) -> pd.DataFrame:
    out = matches.copy()
    out["control_group"] = control_group
    out = _attach_role_institution_info(out, lookup, id_col="treated_unitid", prefix="treated")
    out = _attach_role_institution_info(out, lookup, id_col="control_unitid", prefix="control")

    for col in ("source_cip6", "target_cip6", "control_cip6"):
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = out[col].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)
        out[f"{col}_dotted"] = out[col].map(_cip_dotted)
    if "control_cip2" not in out.columns:
        out["control_cip2"] = out["control_cip6"].str.slice(0, 2)
    out["control_cip2"] = out["control_cip2"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(2)

    ordered = [
        "control_group",
        "pair_id",
        "relabel_year",
        "relabel_type",
        "degree_type",
        "awlevel",
        "broad_pair_bin",
        "source_cip6",
        "source_cip6_dotted",
        "target_cip6",
        "target_cip6_dotted",
        "treated_unitid",
        "treated_fice",
        "treated_opeid8",
        "treated_school_name",
        "treated_state",
        "control_unitid",
        "control_fice",
        "control_opeid8",
        "control_school_name",
        "control_state",
        "control_cip6",
        "control_cip6_dotted",
        "control_cip2",
        "control_relabel_year",
        "treated_source_group_pre_size",
        "control_source_group_pre_size",
        "treated_pre_avg_level",
        "control_pre_avg_level",
        "treated_pre_avg_growth",
        "control_pre_avg_growth",
        "match_distance",
        "match_with_replacement",
        "treated_carnegie_basic",
        "treated_carnegie_basic_label",
        "control_carnegie_basic",
        "control_carnegie_basic_label",
    ]
    present = [col for col in ordered if col in out.columns]
    rest = [col for col in out.columns if col not in present]
    return out[present + rest].sort_values(
        ["control_group", "relabel_year", "degree_type", "broad_pair_bin", "treated_unitid", "control_unitid"],
        kind="mergesort",
    )


def _make_long_export(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in pair_df.itertuples(index=False):
        base_row = {
            "control_group": row.control_group,
            "pair_id": row.pair_id,
            "assigned_relabel_year": row.relabel_year,
            "event_relabel_year": row.relabel_year,
            "relabel_type": row.relabel_type,
            "degree_type": row.degree_type,
            "awlevel": row.awlevel,
            "broad_pair_bin": row.broad_pair_bin,
            "event_source_cip6": row.source_cip6,
            "event_source_cip6_dotted": row.source_cip6_dotted,
            "event_target_cip6": row.target_cip6,
            "event_target_cip6_dotted": row.target_cip6_dotted,
        }
        rows.append(
            {
                **base_row,
                "role": "treated",
                "unitid": row.treated_unitid,
                "fice": row.treated_fice,
                "opeid8": row.treated_opeid8,
                "school_name": row.treated_school_name,
                "state": row.treated_state,
                "role_cip6": row.source_cip6,
                "role_cip6_dotted": row.source_cip6_dotted,
                "role_cip2": str(row.source_cip6)[:2] if pd.notna(row.source_cip6) else pd.NA,
                "actual_control_relabel_year": pd.NA,
            }
        )
        rows.append(
            {
                **base_row,
                "role": "control",
                "unitid": row.control_unitid,
                "fice": row.control_fice,
                "opeid8": row.control_opeid8,
                "school_name": row.control_school_name,
                "state": row.control_state,
                "role_cip6": row.control_cip6,
                "role_cip6_dotted": row.control_cip6_dotted,
                "role_cip2": row.control_cip2,
                "actual_control_relabel_year": row.control_relabel_year,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["control_group", "assigned_relabel_year", "degree_type", "broad_pair_bin", "pair_id", "role"],
        kind="mergesort",
    )


def _write_summary(pair_df: pd.DataFrame, long_df: pd.DataFrame, output_dir: Path) -> None:
    lines = []
    lines.append("control_group,pairs,treated_schools,control_schools,treated_fice,control_fice")
    for control_group, g in pair_df.groupby("control_group", dropna=False):
        long_g = long_df[long_df["control_group"].eq(control_group)]
        treated = long_g[long_g["role"].eq("treated")]
        control = long_g[long_g["role"].eq("control")]
        lines.append(
            ",".join(
                [
                    str(control_group),
                    str(g["pair_id"].nunique()),
                    str(treated["unitid"].nunique()),
                    str(control["unitid"].nunique()),
                    str(treated["fice"].nunique(dropna=True)),
                    str(control["fice"].nunique(dropna=True)),
                ]
            )
        )
    (output_dir / "summary.csv").write_text("\n".join(lines) + "\n")


def export_design_inputs(
    *,
    relabel_panel_path: Path,
    ipeds_path: Path,
    hd_dir: Path,
    output_dir: Path,
    control_groups: Iterable[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    relabel_panel = pd.read_parquet(relabel_panel_path)
    lookup = _load_institution_lookup(hd_dir)

    con = ddb.connect()
    pair_frames: list[pd.DataFrame] = []
    long_frames: list[pd.DataFrame] = []
    try:
        for control_group_raw in control_groups:
            control_group = generalized._normalize_control_group(control_group_raw)
            matches = generalized.match_treated_to_never_treated(
                con,
                relabel_panel,
                ipeds_path=ipeds_path,
                control_group=control_group,
            )
            pair_df = _prepare_pair_export(matches, lookup, control_group)
            long_df = _make_long_export(pair_df)
            pair_df.to_csv(output_dir / f"relabel_school_pairs_{control_group}.csv", index=False)
            long_df.to_csv(output_dir / f"relabel_school_list_{control_group}.csv", index=False)
            pair_frames.append(pair_df)
            long_frames.append(long_df)
    finally:
        con.close()

    all_pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    all_long = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    all_pairs.to_csv(output_dir / "relabel_school_pairs_all_designs.csv", index=False)
    all_long.to_csv(output_dir / "relabel_school_list_all_designs.csv", index=False)
    _write_summary(all_pairs, all_long, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relabel-panel", type=Path, default=generalized.DEFAULT_PANEL_PARQUET)
    parser.add_argument("--ipeds", type=Path, default=Path(base.IPEDS_PATH))
    parser.add_argument("--hd-dir", type=Path, default=Path(base.root) / "data" / "raw" / "ipeds" / "directory_info_hd")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--control-groups",
        nargs="+",
        default=[generalized.CONTROL_GROUP_NEVER_TREATED, generalized.CONTROL_GROUP_ALWAYS_STEM],
        help="Control designs to export. Defaults to never_treated and always_stem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_design_inputs(
        relabel_panel_path=args.relabel_panel,
        ipeds_path=args.ipeds,
        hd_dir=args.hd_dir,
        output_dir=args.output_dir,
        control_groups=args.control_groups,
    )
    print(f"Wrote relabel school design inputs to {args.output_dir}")


if __name__ == "__main__":
    main()
