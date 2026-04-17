"""
Audit helper for school x grad-class master's-econ samples.

Given a school name and graduation year, this script:
  1. Pulls matched education rows directly from the `relabel_indiv_model`
     treated/control outputs for that school/year, restricted to master's-econ
     grads.
  2. Pulls the full relabel-model position histories for the matched users.
  3. Computes a first qualifying post-grad US job as supplemental context.
  4. Computes relabel-model-style 1/3/5/10-year `still_in_us` indicators from
     the matched position history.
  5. Writes a readable relabel txt audit report and a separate FOIA txt report.

The first qualifying post-grad US job is still shown for context and mirrors the
older transition logic:
  - US position
  - non-null rcid
  - start date in [grad_date, grad_date + 1 year]
  - duration at least `min_position_days`

The printed 1/3/5/10-year stay indicators now align to the relabel analysis
logic instead:
  - evaluation year = grad_year + horizon
  - optionally capped to the latest observed position year in the relabel
    position outputs
  - indicator = 1 iff the user has any US position overlapping that eval year

Interactive use:
    import company_shift_share.audit_school_grad_class as audit
    result = audit.audit_school_grad_class("Yale University", 2015)
    result["path"]
    print(result["preview"])
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import duckdb as ddb
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

try:
    from company_shift_share.config_loader import (
        DEFAULT_CONFIG_PATH as COMPANY_CFG_DEFAULT,
        get_cfg_section,
        load_config as load_company_config,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import (  # type: ignore
        DEFAULT_CONFIG_PATH as COMPANY_CFG_DEFAULT,
        get_cfg_section,
        load_config as load_company_config,
    )

try:
    from relabels_revelio.relabel_indiv_model_config import (
        DEFAULT_CONFIG_PATH as RELABEL_CFG_DEFAULT,
        load_config as load_relabel_config,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from relabels_revelio.relabel_indiv_model_config import (  # type: ignore
        DEFAULT_CONFIG_PATH as RELABEL_CFG_DEFAULT,
        load_config as load_relabel_config,
    )


HORIZONS_YEARS = (1, 3, 5, 10)
FILTER_LABEL = "master's econ grads"
IMMEDIATE_PHD_WINDOW_DAYS = 365


@dataclass(frozen=True)
class RelabelPaths:
    treated_education: Path
    treated_positions: Path
    control_education: Path | None
    control_positions: Path | None
    cap_to_latest_available_year: bool
    run_tag: str


@dataclass(frozen=True)
class FoiaPaths:
    foia_indiv: Path


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _quote_sql_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _slugify(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "school"


def _norm_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _fmt_date(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return pd.Timestamp(value).date().isoformat()


def _fmt_scalar(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(value)


def _fmt_rcid(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(int(value))


def _parquet_has_column(path: Path, column_name: str) -> bool:
    con = ddb.connect()
    try:
        cols = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_escape_sql_literal(str(path))}')"
        ).df()["column_name"].astype(str)
        return column_name in set(cols.tolist())
    finally:
        con.close()


def _econ_text_match_sql(col: str) -> str:
    return f"lower(coalesce({col}, '')) LIKE '%econom%'"


def _master_degree_match_sql(col: str) -> str:
    return (
        f"lower(coalesce({col}, '')) LIKE '%master%'"
        f" OR lower(coalesce({col}, '')) LIKE '%mphil%'"
        f" OR lower(coalesce({col}, '')) LIKE '%m.a.%'"
        f" OR lower(coalesce({col}, '')) LIKE '%m.s.%'"
        f" OR lower(coalesce({col}, '')) LIKE '%m.a %'"
        f" OR lower(coalesce({col}, '')) LIKE '%m.s %'"
    )


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _foia_school_like_pattern(school_name: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", _norm_text(school_name))
    stopwords = {"the", "of", "at", "and", "in", "for", "to"}
    significant_tokens = [tok for tok in tokens if tok not in stopwords]
    if not significant_tokens:
        significant_tokens = tokens
    if not significant_tokens:
        return "%"
    return "%" + "%".join(significant_tokens) + "%"


def _running_in_ipython() -> bool:
    try:
        get_ipython()  # type: ignore[name-defined]
        return True
    except NameError:
        return False


def _has_explicit_cli_args(argv: Iterable[str]) -> bool:
    args = list(argv)
    for arg in args:
        if arg in {"--school", "--grad-year"}:
            return True
        if arg.startswith("--school=") or arg.startswith("--grad-year="):
            return True
    return False


def default_output_path(
    school_name: str,
    grad_year: int,
    output_dir: str | Path | None = None,
) -> Path:
    base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "output" / "audits"
    )
    return base_dir / f"school_grad_class_{_slugify(school_name)}_{int(grad_year)}.txt"


def default_foia_output_path(
    school_name: str,
    grad_year: int,
    output_dir: str | Path | None = None,
) -> Path:
    base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "output" / "audits"
    )
    return base_dir / f"school_grad_class_{_slugify(school_name)}_{int(grad_year)}_foia.txt"


def _resolve_relabel_paths(
    relabel_config_path: str | Path | None,
) -> RelabelPaths:
    relabel_cfg = load_relabel_config(relabel_config_path or RELABEL_CFG_DEFAULT)
    relabel_paths = relabel_cfg.get("paths", {})
    analysis_cfg = relabel_cfg.get("analysis", {})

    treated_education = Path(relabel_paths["matched_education_parquet"])
    treated_positions = Path(relabel_paths["matched_positions_parquet"])

    control_education_raw = str(relabel_paths.get("never_treated_education_parquet", "") or "").strip()
    control_positions_raw = str(relabel_paths.get("never_treated_positions_parquet", "") or "").strip()
    control_education = Path(control_education_raw) if control_education_raw else None
    control_positions = Path(control_positions_raw) if control_positions_raw else None

    missing = [str(p) for p in (treated_education, treated_positions) if not p.exists()]
    if control_education is not None and not control_education.exists():
        missing.append(str(control_education))
    if control_positions is not None and not control_positions.exists():
        missing.append(str(control_positions))
    if missing:
        raise FileNotFoundError("Missing required relabel inputs:\n" + "\n".join(missing))

    if (control_education is None) != (control_positions is None):
        raise ValueError(
            "Control relabel outputs are partially configured. Expected both education "
            "and positions paths or neither."
        )

    return RelabelPaths(
        treated_education=treated_education,
        treated_positions=treated_positions,
        control_education=control_education,
        control_positions=control_positions,
        cap_to_latest_available_year=_as_bool(
            analysis_cfg.get("cap_to_latest_available_year"),
            True,
        ),
        run_tag=str(relabel_cfg.get("run_tag", "apr2026")),
    )


def _resolve_foia_paths(company_config_path: str | Path | None) -> FoiaPaths:
    company_cfg = load_company_config(company_config_path or COMPANY_CFG_DEFAULT)
    company_paths = get_cfg_section(company_cfg, "paths")
    legacy_foia_indiv = Path(company_paths["foia_sevp_with_person_id_employment_corrected"])
    int_root = (
        legacy_foia_indiv.parent.parent
        if legacy_foia_indiv.parent.name.startswith("int_files_")
        else legacy_foia_indiv.parent
    )
    apr2026v1_foia_indiv = (
        int_root
        / "f1_indiv_merge"
        / "01_f1_foia_clean"
        / "foia_person_panel_apr2026v1.parquet"
    )
    foia_indiv = apr2026v1_foia_indiv if apr2026v1_foia_indiv.exists() else legacy_foia_indiv
    if not foia_indiv.exists():
        raise FileNotFoundError(
            "Missing required FOIA input. Checked:\n"
            f"{apr2026v1_foia_indiv}\n"
            f"{legacy_foia_indiv}"
        )
    return FoiaPaths(foia_indiv=foia_indiv)


def _source_path_pairs(
    paths: RelabelPaths,
    kind: str,
) -> list[tuple[str, Path]]:
    if kind == "education":
        pairs = [("treated", paths.treated_education)]
        if paths.control_education is not None:
            pairs.append(("control", paths.control_education))
        return pairs
    if kind == "positions":
        pairs = [("treated", paths.treated_positions)]
        if paths.control_positions is not None:
            pairs.append(("control", paths.control_positions))
        return pairs
    raise ValueError(f"Unsupported relabel source kind: {kind}")


def _relabel_union_sql(
    paths: RelabelPaths,
    kind: str,
    source_labels: Sequence[str] | None = None,
) -> str:
    pairs = _source_path_pairs(paths, kind)
    wanted = None if source_labels is None else {label for label in source_labels}
    selects = []
    for label, path in pairs:
        if wanted is not None and label not in wanted:
            continue
        if kind == "education" and not _parquet_has_column(path, "exclude_immediate_same_inst_phd_after_master_ind"):
            selects.append(
                "SELECT "
                f"'{label}' AS source_dataset, "
                "src.*, "
                "0::INTEGER AS exclude_immediate_same_inst_phd_after_master_ind "
                f"FROM read_parquet('{_escape_sql_literal(str(path))}') AS src"
            )
        else:
            selects.append(
                f"SELECT '{label}' AS source_dataset, * FROM read_parquet('{_escape_sql_literal(str(path))}')"
            )
    if not selects:
        raise ValueError(f"No relabel {kind} sources selected.")
    return "\nUNION ALL\n".join(selects)


def _load_matching_spells(
    con: ddb.DuckDBPyConnection,
    paths: RelabelPaths,
    school_name: str,
    grad_year: int,
) -> pd.DataFrame:
    education_sql = _relabel_union_sql(paths, "education")
    query = f"""
        WITH relabel_education AS (
            {education_sql}
        ),
        ranked AS (
            SELECT
                source_dataset,
                CAST(unitid AS BIGINT) AS unitid,
                school_name,
                school_name_clean,
                CAST(relabel_year AS INTEGER) AS relabel_year,
                relabel_type,
                CAST(event_rsid AS BIGINT) AS event_rsid,
                event_rsid_university_name,
                CAST(rsid_candidate_count AS BIGINT) AS rsid_candidate_count,
                CAST(rsid_name_match_score AS BIGINT) AS rsid_name_match_score,
                CAST(user_id AS BIGINT) AS user_id,
                fullname,
                CAST(rsid AS BIGINT) AS rsid,
                university_name,
                CAST(education_number AS BIGINT) AS education_number,
                TRY_CAST(ed_startdate AS DATE) AS ed_startdate,
                TRY_CAST(ed_enddate AS DATE) AS ed_enddate,
                CAST(ed_end_year AS INTEGER) AS grad_year,
                degree AS degree_clean,
                field AS field_clean,
                university_country,
                university_location,
                university_raw,
                degree_raw,
                field_raw,
                description,
                COALESCE(CAST(exclude_immediate_same_inst_phd_after_master_ind AS INTEGER), 0)
                    AS exclude_immediate_same_inst_phd_after_master_ind,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        source_dataset,
                        CAST(user_id AS BIGINT),
                        COALESCE(CAST(education_number AS BIGINT), -1)
                    ORDER BY
                        CASE WHEN TRY_CAST(ed_enddate AS DATE) IS NULL THEN 1 ELSE 0 END,
                        TRY_CAST(ed_enddate AS DATE) DESC,
                        CAST(relabel_year AS INTEGER),
                CAST(unitid AS BIGINT)
                ) AS rn
            FROM relabel_education
            WHERE school_name IS NOT NULL
              AND lower(trim(school_name)) = lower(trim(?))
              AND user_id IS NOT NULL
              AND ed_end_year IS NOT NULL
              AND CAST(ed_end_year AS INTEGER) = ?
              AND (
                    lower(coalesce(degree, '')) = 'master'
                    OR {_master_degree_match_sql('degree_raw')}
              )
              AND (
                    {_econ_text_match_sql('field')}
                    OR {_econ_text_match_sql('field_raw')}
              )
        )
        SELECT
            * EXCLUDE(rn)
        FROM ranked
        WHERE rn = 1
        ORDER BY source_dataset, school_name, fullname, user_id, education_number
    """
    return con.execute(query, [school_name, int(grad_year)]).df()


def _load_positions(
    con: ddb.DuckDBPyConnection,
    paths: RelabelPaths,
    user_ids: Iterable[int],
    source_labels: Sequence[str],
) -> pd.DataFrame:
    user_ids = sorted({int(x) for x in user_ids})
    if not user_ids:
        return pd.DataFrame(
            columns=[
                "user_id",
                "position_id",
                "position_number",
                "rcid",
                "country",
                "country_lc",
                "startdate",
                "enddate",
                "company_raw",
                "title_raw",
                "company_norm",
                "source_dataset",
            ]
        )

    user_df = pd.DataFrame({"user_id": user_ids})
    con.register("matched_users_param", user_df)
    positions_sql = _relabel_union_sql(paths, "positions", source_labels=source_labels)
    query = f"""
        WITH relabel_positions AS (
            {positions_sql}
        ),
        filtered AS (
            SELECT
                CAST(p.user_id AS BIGINT) AS user_id,
                CAST(p.position_id AS BIGINT) AS position_id,
                CAST(p.position_number AS BIGINT) AS position_number,
                CAST(p.rcid AS BIGINT) AS rcid,
                p.country,
                lower(coalesce(p.country, '')) AS country_lc,
                TRY_CAST(p.startdate AS DATE) AS startdate,
                TRY_CAST(p.enddate AS DATE) AS enddate,
                p.company_raw,
                p.title_raw,
                p.source_dataset
            FROM relabel_positions AS p
            JOIN matched_users_param AS u
              ON CAST(p.user_id AS BIGINT) = u.user_id
            WHERE p.user_id IS NOT NULL
              AND TRY_CAST(p.startdate AS DATE) IS NOT NULL
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        user_id,
                        COALESCE(position_id, -1),
                        COALESCE(position_number, -1),
                        startdate,
                        enddate,
                        COALESCE(company_raw, ''),
                        COALESCE(title_raw, '')
                    ORDER BY CASE source_dataset WHEN 'treated' THEN 0 ELSE 1 END
                ) AS rn
            FROM filtered
        )
        SELECT
            user_id,
            position_id,
            position_number,
            rcid,
            country,
            country_lc,
            startdate,
            enddate,
            company_raw,
            title_raw,
            source_dataset,
            lower(trim(company_raw)) AS company_norm
        FROM ranked
        WHERE rn = 1
        ORDER BY user_id, startdate, position_number, position_id
    """
    positions = con.execute(query).df()
    con.unregister("matched_users_param")
    return positions


def _latest_available_year(
    con: ddb.DuckDBPyConnection,
    paths: RelabelPaths,
    source_labels: Sequence[str],
) -> int:
    positions_sql = _relabel_union_sql(paths, "positions", source_labels=source_labels)
    query = f"""
        WITH relabel_positions AS (
            {positions_sql}
        ),
        years AS (
            SELECT EXTRACT(YEAR FROM TRY_CAST(startdate AS DATE))::INTEGER AS obs_year
            FROM relabel_positions
            WHERE TRY_CAST(startdate AS DATE) IS NOT NULL
            UNION ALL
            SELECT EXTRACT(YEAR FROM TRY_CAST(enddate AS DATE))::INTEGER AS obs_year
            FROM relabel_positions
            WHERE TRY_CAST(enddate AS DATE) IS NOT NULL
        )
        SELECT MAX(obs_year) AS latest_available_year
        FROM years
        WHERE obs_year IS NOT NULL
    """
    latest = con.execute(query).fetchone()[0]
    if latest is None:
        raise ValueError("Could not determine latest available year from relabel positions.")
    return int(latest)


def _position_duration_days(row: pd.Series, as_of_date: pd.Timestamp) -> int | None:
    if pd.isna(row["startdate"]):
        return None
    enddate = row["enddate"] if pd.notna(row["enddate"]) else as_of_date
    return int((enddate - row["startdate"]).days)


def _summarize_employers(active_positions: pd.DataFrame) -> str:
    if active_positions.empty:
        return ""
    deduped = (
        active_positions.sort_values(["startdate", "position_number", "position_id"])
        .drop_duplicates(["rcid", "company_norm"])
    )
    values = [
        f"{_fmt_scalar(row['company_raw'])} [rcid={_fmt_rcid(row['rcid'])}]"
        for _, row in deduped.iterrows()
    ]
    if len(values) <= 5:
        return "; ".join(values)
    return "; ".join(values[:5]) + f"; ... (+{len(values) - 5} more)"


def _compute_spell_audit(
    spell: pd.Series,
    user_positions: pd.DataFrame,
    as_of_date: pd.Timestamp,
    min_position_days: int,
    entry_window_years: int,
    latest_available_year: int,
    cap_to_latest_available_year: bool,
) -> dict[str, object]:
    grad_date = spell["ed_enddate"]
    grad_year = spell["grad_year"]
    out: dict[str, object] = {
        "first_job_company": pd.NA,
        "first_job_rcid": pd.NA,
        "first_job_startdate": pd.NaT,
        "first_job_enddate": pd.NaT,
        "first_job_title": pd.NA,
        "first_job_position_id": None,
        "first_job_note": "no qualifying first post-grad US position found",
    }

    positions = user_positions.copy()
    if not positions.empty:
        positions["effective_enddate"] = positions["enddate"].where(
            positions["enddate"].notna(),
            as_of_date,
        )
        positions["duration_days"] = positions.apply(
            _position_duration_days,
            axis=1,
            as_of_date=as_of_date,
        )

    if not positions.empty and pd.notna(grad_date):
        window_end = grad_date + pd.DateOffset(years=entry_window_years)
        qualifying = positions.loc[
            positions["country_lc"].eq("united states")
            & positions["rcid"].notna()
            & (positions["startdate"] >= grad_date)
            & (positions["startdate"] <= window_end)
            & (positions["duration_days"] >= int(min_position_days))
        ].sort_values(["startdate", "position_number", "position_id"])

        if not qualifying.empty:
            first_job = qualifying.iloc[0]
            out["first_job_company"] = first_job["company_raw"]
            out["first_job_rcid"] = first_job["rcid"]
            out["first_job_startdate"] = first_job["startdate"]
            out["first_job_enddate"] = first_job["enddate"]
            out["first_job_title"] = first_job["title_raw"]
            out["first_job_position_id"] = first_job["position_id"]
            out["first_job_note"] = "qualifying first post-grad US position"

    for horizon in HORIZONS_YEARS:
        key = f"stay_{horizon}y"
        note_key = f"stay_{horizon}y_note"

        if pd.isna(grad_year):
            out[key] = pd.NA
            out[note_key] = "missing grad_year"
            continue

        target_year = int(grad_year) + int(horizon)
        eval_year = target_year
        used_latest_avail = 0
        if cap_to_latest_available_year and target_year > latest_available_year:
            eval_year = latest_available_year
            used_latest_avail = 1
        target_year_observed = int(target_year <= latest_available_year)

        eval_start = pd.Timestamp(year=eval_year, month=1, day=1)
        eval_end = pd.Timestamp(year=eval_year, month=12, day=31)
        if positions.empty:
            active_us = pd.DataFrame()
        else:
            active_us = positions.loc[
                positions["country_lc"].eq("united states")
                & (positions["startdate"] <= eval_end)
                & (positions["enddate"].isna() | (positions["enddate"] >= eval_start))
            ].copy()
        active_txt = _summarize_employers(active_us)

        out[key] = int(not active_us.empty)
        if active_us.empty:
            out[note_key] = (
                f"no US position active in eval_year={eval_year}"
                f" | target_year={target_year}"
                f" | target_year_observed={target_year_observed}"
                f" | used_latest_avail={used_latest_avail}"
            )
        else:
            out[note_key] = (
                f"US position active in eval_year={eval_year}"
                f" | target_year={target_year}"
                f" | target_year_observed={target_year_observed}"
                f" | used_latest_avail={used_latest_avail}"
                + (f" | active_us_employers: {active_txt}" if active_txt else "")
            )

    return out


def _build_report(
    school_name: str,
    grad_year: int,
    relabel_paths: RelabelPaths,
    spell_rows: list[dict[str, object]],
    positions_by_user: dict[int, pd.DataFrame],
    as_of_date: pd.Timestamp,
    min_position_days: int,
    entry_window_years: int,
    latest_available_year: int,
    source_labels: Sequence[str],
    n_candidate_spells_before_immediate_phd_exclusion: int,
    n_excluded_immediate_same_inst_phd_after_master: int,
) -> str:
    lines: list[str] = []
    lines.append("School audit report")
    lines.append(f"School query: {school_name}")
    lines.append(f"Grad year: {grad_year}")
    lines.append(f"Filter: {FILTER_LABEL}")
    lines.append("Data source: relabel_indiv_model outputs")
    lines.append(f"Relabel config default: {RELABEL_CFG_DEFAULT}")
    lines.append(f"Relabel run_tag: {relabel_paths.run_tag}")
    lines.append(f"Matched relabel source datasets: {', '.join(source_labels)}")
    lines.append(f"As-of date: {_fmt_date(as_of_date)}")
    lines.append(
        "Stay indicator logic: relabel_indiv_model-style still_in_us at grad_year + horizon "
        f"(latest available position year={latest_available_year}, "
        f"cap_to_latest_available_year={int(relabel_paths.cap_to_latest_available_year)})"
    )
    lines.append(
        "Doctoral-continuation exclusion: drop master's rows where the same user has a Doctor spell "
        f"at the same institution overlapping the master's or starting within {IMMEDIATE_PHD_WINDOW_DAYS} days after the master's end"
    )
    lines.append(f"Qualifying first-job window: [grad_date, grad_date + {entry_window_years} year(s)]")
    lines.append(f"Qualifying first-job minimum duration: {min_position_days} days")

    matched_users = {int(r["user_id"]) for r in spell_rows}
    users_with_positions = {uid for uid in matched_users if uid in positions_by_user}
    lines.append(f"Candidate education spells before doctoral-continuation exclusion: {n_candidate_spells_before_immediate_phd_exclusion:,}")
    lines.append(f"Excluded same-institution doctoral continuers: {n_excluded_immediate_same_inst_phd_after_master:,}")
    lines.append(f"Matched education spells after exclusion: {len(spell_rows):,}")
    lines.append(f"Matched unique users: {len(matched_users):,}")
    lines.append(f"Matched users with any positions in relabel outputs: {len(users_with_positions):,}")
    lines.append(f"Matched users with no positions in relabel outputs: {len(matched_users - users_with_positions):,}")

    if spell_rows:
        events = (
            pd.DataFrame(spell_rows)[
                [
                    "source_dataset",
                    "unitid",
                    "school_name",
                    "school_name_clean",
                    "relabel_year",
                    "relabel_type",
                    "event_rsid",
                    "event_rsid_university_name",
                ]
            ]
            .drop_duplicates()
            .sort_values(["source_dataset", "relabel_year", "unitid", "event_rsid"])
        )
        lines.append(f"Matched school-event rows: {len(events):,}")
        for _, event in events.iterrows():
            lines.append(
                "  - "
                + " | ".join(
                    [
                        f"source={_fmt_scalar(event['source_dataset'])}",
                        f"unitid={_fmt_scalar(event['unitid'])}",
                        f"school_name={_fmt_scalar(event['school_name'])}",
                        f"school_name_clean={_fmt_scalar(event['school_name_clean'])}",
                        f"relabel_year={_fmt_scalar(event['relabel_year'])}",
                        f"relabel_type={_fmt_scalar(event['relabel_type'])}",
                        f"event_rsid={_fmt_rcid(event['event_rsid'])}",
                        f"event_school={_fmt_scalar(event['event_rsid_university_name'])}",
                    ]
                )
            )
    lines.append(
        "Coverage note: the main report now comes directly from relabel_indiv_model "
        "education/positions outputs after the immediate-PhD exclusion; '(no positions found)' "
        "means that user_id is absent from the matched relabel position parquet(s) for the selected "
        "source dataset(s)."
    )
    lines.append("")

    if not spell_rows:
        if n_candidate_spells_before_immediate_phd_exclusion > 0 and n_excluded_immediate_same_inst_phd_after_master == n_candidate_spells_before_immediate_phd_exclusion:
            lines.append(
                "All candidate Revelio master spells were excluded because the user appears to continue "
                "into a same-institution Doctor program during or immediately after the master's."
            )
        else:
            lines.append("No matching completed education spells found in relabel_indiv_model outputs.")
        return "\n".join(lines) + "\n"

    for idx, row in enumerate(spell_rows, start=1):
        user_id = int(row["user_id"])
        user_positions = positions_by_user.get(user_id, pd.DataFrame())

        lines.append("=" * 88)
        lines.append(
            f"Spell {idx} of {len(spell_rows)}"
            f" | source_dataset={_fmt_scalar(row['source_dataset'])}"
            f" | user_id={user_id}"
            f" | education_number={_fmt_scalar(row['education_number'])}"
        )
        lines.append(f"name: {_fmt_scalar(row['fullname'])}")
        lines.append(
            "school_event: "
            f"unitid={_fmt_scalar(row['unitid'])}"
            f" | school_name={_fmt_scalar(row['school_name'])}"
            f" | school_name_clean={_fmt_scalar(row['school_name_clean'])}"
            f" | relabel_year={_fmt_scalar(row['relabel_year'])}"
            f" | relabel_type={_fmt_scalar(row['relabel_type'])}"
            f" | event_rsid={_fmt_rcid(row['event_rsid'])}"
            f" | event_school={_fmt_scalar(row['event_rsid_university_name'])}"
        )
        lines.append(
            "matched_education: "
            f"rsid={_fmt_rcid(row['rsid'])}"
            f" | university_raw={_fmt_scalar(row['university_raw'])}"
            f" | university_name={_fmt_scalar(row['university_name'])}"
            f" | university_country={_fmt_scalar(row['university_country'])}"
            f" | university_location={_fmt_scalar(row['university_location'])}"
            f" | rsid_candidate_count={_fmt_scalar(row['rsid_candidate_count'])}"
            f" | rsid_name_match_score={_fmt_scalar(row['rsid_name_match_score'])}"
        )
        lines.append(
            "education_dates: "
            f"{_fmt_date(row['ed_startdate'])} to {_fmt_date(row['ed_enddate'])}"
        )
        lines.append(f"degree_raw: {_fmt_scalar(row['degree_raw'])}")
        lines.append(f"field_raw: {_fmt_scalar(row['field_raw'])}")
        lines.append(f"degree_clean: {_fmt_scalar(row['degree_clean'])}")
        lines.append(f"field_clean: {_fmt_scalar(row['field_clean'])}")
        lines.append(f"description: {_fmt_scalar(row['description'])}")
        lines.append(
            "first_qualifying_postgrad_us_job: "
            f"{_fmt_scalar(row['first_job_company'])}"
            f" | rcid={_fmt_rcid(row['first_job_rcid'])}"
            f" | start={_fmt_date(row['first_job_startdate'])}"
            f" | end={_fmt_date(row['first_job_enddate'])}"
            f" | title={_fmt_scalar(row['first_job_title'])}"
            f" | note={_fmt_scalar(row['first_job_note'])}"
        )
        lines.append("assigned_stay_indicators:")
        for horizon in HORIZONS_YEARS:
            lines.append(
                f"  {horizon}y: {_fmt_scalar(row[f'stay_{horizon}y'])}"
                f" | {row[f'stay_{horizon}y_note']}"
            )

        lines.append("postgrad_position_history:")
        if user_positions.empty:
            lines.append("  (no positions found)")
            continue

        grad_date = row["ed_enddate"]
        postgrad_positions = user_positions.copy()
        postgrad_positions["effective_enddate"] = postgrad_positions["enddate"].where(
            postgrad_positions["enddate"].notna(),
            as_of_date,
        )
        if pd.notna(grad_date):
            postgrad_positions = postgrad_positions.loc[
                postgrad_positions["effective_enddate"] >= grad_date
            ].copy()

        if postgrad_positions.empty:
            lines.append("  (no post-grad positions found)")
            continue

        first_job_position_id = row.get("first_job_position_id")
        for _, pos in postgrad_positions.sort_values(["startdate", "position_number", "position_id"]).iterrows():
            marker = "* " if first_job_position_id is not None and pos["position_id"] == first_job_position_id else "- "
            lines.append(
                marker
                + f"{_fmt_date(pos['startdate'])} to {_fmt_date(pos['enddate'])}"
                + f" | employer={_fmt_scalar(pos['company_raw'])}"
                + f" | title={_fmt_scalar(pos['title_raw'])}"
                + f" | country={_fmt_scalar(pos['country'])}"
                + f" | rcid={_fmt_rcid(pos['rcid'])}"
                + f" | source={_fmt_scalar(pos['source_dataset'])}"
            )

    return "\n".join(lines) + "\n"


def _load_foia_people(
    con: ddb.DuckDBPyConnection,
    foia_path: Path,
    school_name: str,
    grad_year: int,
) -> pd.DataFrame:
    school_like_pattern = _foia_school_like_pattern(school_name)
    major_2_cip_col = (
        "major_2_cip_code_"
        if _parquet_has_column(foia_path, "major_2_cip_code_")
        else "major_2_cip_code"
    )
    query = f"""
        WITH base AS (
            SELECT
                CAST(person_id AS BIGINT) AS person_id,
                year_int,
                school_name,
                TRY_CAST(program_end_date AS DATE) AS program_end_date,
                TRY_CAST(authorization_start_date AS DATE) AS authorization_start_date,
                TRY_CAST(authorization_end_date AS DATE) AS authorization_end_date,
                NULLIF(trim(employer_name), '') AS employer_name,
                NULLIF(trim(employment_opt_type), '') AS employment_opt_type,
                NULLIF(trim(requested_status), '') AS requested_status,
                NULLIF(trim(status_code), '') AS status_code,
                NULLIF(trim(class_of_admission), '') AS class_of_admission,
                NULLIF(trim(country_of_birth), '') AS country_of_birth,
                NULLIF(trim(country_of_citizenship), '') AS country_of_citizenship,
                NULLIF(trim(student_edu_level_desc), '') AS student_edu_level_desc,
                NULLIF(trim(major_1_description), '') AS major_1_description,
                NULLIF(trim(major_2_description), '') AS major_2_description,
                NULLIF(trim(minor_description), '') AS minor_description,
                NULLIF(trim(major_1_cip_code), '') AS major_1_cip_code,
                NULLIF(trim({_quote_sql_ident(major_2_cip_col)}), '') AS major_2_cip_code,
                NULLIF(trim(minor_cip_code), '') AS minor_cip_code
            FROM read_parquet('{_escape_sql_literal(str(foia_path))}')
            WHERE person_id IS NOT NULL
              AND school_name IS NOT NULL
              AND (
                    lower(trim(school_name)) = lower(trim(?))
                    OR lower(trim(school_name)) LIKE ?
              )
              AND program_end_date IS NOT NULL
              AND EXTRACT(YEAR FROM TRY_CAST(program_end_date AS DATE)) = ?
              AND upper(coalesce(student_edu_level_desc, '')) = 'MASTER''S'
              AND (
                    lower(coalesce(major_1_description, '')) LIKE '%econom%'
                    OR lower(coalesce(major_2_description, '')) LIKE '%econom%'
                    OR lower(coalesce(minor_description, '')) LIKE '%econom%'
                    OR coalesce(major_1_cip_code, '') LIKE '45.06%'
                    OR coalesce(major_1_cip_code, '') LIKE '52.06%'
                    OR coalesce({_quote_sql_ident(major_2_cip_col)}, '') LIKE '45.06%'
                    OR coalesce({_quote_sql_ident(major_2_cip_col)}, '') LIKE '52.06%'
                    OR coalesce(minor_cip_code, '') LIKE '45.06%'
                    OR coalesce(minor_cip_code, '') LIKE '52.06%'
              )
        ),
        person_agg AS (
            SELECT
                person_id,
                MIN(program_end_date) AS min_program_end_date,
                MAX(program_end_date) AS max_program_end_date,
                MIN(year_int) AS min_reported_year,
                MAX(year_int) AS max_reported_year,
                COUNT(*) AS n_rows,
                COUNT(
                    DISTINCT CONCAT(
                        COALESCE(employer_name, 'NA'),
                        ' | ',
                        COALESCE(CAST(authorization_start_date AS VARCHAR), 'NA'),
                        ' | ',
                        COALESCE(CAST(authorization_end_date AS VARCHAR), 'NA'),
                        ' | ',
                        COALESCE(employment_opt_type, 'NA')
                    )
                ) AS n_employment_spells,
                MAX(CASE WHEN requested_status IS NOT NULL THEN 1 ELSE 0 END) AS status_change_ind,
                list_sort(
                    list(DISTINCT COALESCE(country_of_citizenship, country_of_birth))
                    FILTER (WHERE COALESCE(country_of_citizenship, country_of_birth) IS NOT NULL)
                ) AS country_of_origin_values,
                list_sort(list(DISTINCT country_of_citizenship) FILTER (WHERE country_of_citizenship IS NOT NULL)) AS country_of_citizenship_values,
                list_sort(list(DISTINCT country_of_birth) FILTER (WHERE country_of_birth IS NOT NULL)) AS country_of_birth_values,
                list_sort(list(DISTINCT employment_opt_type) FILTER (WHERE employment_opt_type IS NOT NULL)) AS employment_opt_type_values,
                list_sort(
                    list(
                        DISTINCT CONCAT(
                            'employer=',
                            COALESCE(employer_name, 'NA'),
                            ' | authorization=',
                            COALESCE(CAST(authorization_start_date AS VARCHAR), 'NA'),
                            ' to ',
                            COALESCE(CAST(authorization_end_date AS VARCHAR), 'NA'),
                            ' | employment_opt_type=',
                            COALESCE(employment_opt_type, 'NA')
                        )
                    )
                ) AS employment_spells,
                list_sort(list(DISTINCT requested_status) FILTER (WHERE requested_status IS NOT NULL)) AS requested_status_values,
                list_sort(list(DISTINCT status_code) FILTER (WHERE status_code IS NOT NULL)) AS status_code_values,
                list_sort(list(DISTINCT class_of_admission) FILTER (WHERE class_of_admission IS NOT NULL)) AS class_of_admission_values,
                list_sort(list(DISTINCT major_1_description) FILTER (WHERE major_1_description IS NOT NULL)) AS major_1_description_values
            FROM base
            GROUP BY person_id
        )
        SELECT *
        FROM person_agg
        ORDER BY person_id
    """
    return con.execute(query, [school_name, school_like_pattern, int(grad_year)]).df()


def _list_to_text(value: object) -> str:
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return "[" + "; ".join(str(v) for v in value) + "]"
    if value is None:
        return "[]"
    try:
        if pd.isna(value):
            return "[]"
    except (TypeError, ValueError):
        pass
    return str(value)


def _build_foia_report(
    school_name: str,
    grad_year: int,
    foia_people: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("FOIA school/program-end audit report")
    lines.append(f"School query: {school_name}")
    lines.append(f"Program end year: {grad_year}")
    lines.append(f"Filter: {FILTER_LABEL}")
    lines.append("Status-change convention: status_change_ind = 1 iff requested_status is non-null on any matching FOIA row")
    lines.append(f"Matched unique person_ids: {len(foia_people):,}")
    lines.append("")

    if foia_people.empty:
        lines.append("No matching FOIA person_ids found.")
        return "\n".join(lines) + "\n"

    for idx, row in enumerate(foia_people.to_dict("records"), start=1):
        lines.append("=" * 88)
        lines.append(f"Person {idx} of {len(foia_people)} | person_id={int(row['person_id'])}")
        lines.append(
            "program_end_date_range: "
            f"{_fmt_date(row['min_program_end_date'])} to {_fmt_date(row['max_program_end_date'])}"
        )
        lines.append(
            "reported_year_range: "
            f"{_fmt_scalar(row['min_reported_year'])} to {_fmt_scalar(row['max_reported_year'])}"
        )
        lines.append(f"n_rows: {int(row['n_rows'])}")
        lines.append(f"status_change_ind: {int(row['status_change_ind'])}")
        lines.append(f"country_of_origin_values: {_list_to_text(row['country_of_origin_values'])}")
        lines.append(f"country_of_citizenship_values: {_list_to_text(row['country_of_citizenship_values'])}")
        lines.append(f"country_of_birth_values: {_list_to_text(row['country_of_birth_values'])}")
        lines.append(f"employment_opt_type_values: {_list_to_text(row['employment_opt_type_values'])}")
        lines.append(f"requested_status_values: {_list_to_text(row['requested_status_values'])}")
        lines.append(f"status_code_values: {_list_to_text(row['status_code_values'])}")
        lines.append(f"class_of_admission_values: {_list_to_text(row['class_of_admission_values'])}")
        lines.append(f"major_1_description_values: {_list_to_text(row['major_1_description_values'])}")
        lines.append(f"employment_spells ({int(row['n_employment_spells'])}): {_list_to_text(row['employment_spells'])}")

    return "\n".join(lines) + "\n"


def build_audit_report(
    school_name: str,
    grad_year: int,
    output_path: str | Path | None = None,
    company_config_path: str | Path | None = None,
    relabel_config_path: str | Path | None = None,
    as_of_date: date | str | None = None,
    min_position_days: int = 365,
    entry_window_years: int = 1,
) -> Path:
    _ = company_config_path
    if output_path is None:
        output_path = default_output_path(school_name=school_name, grad_year=int(grad_year))
    output_path = Path(output_path)

    relabel_paths = _resolve_relabel_paths(relabel_config_path)
    as_of_ts = pd.Timestamp(as_of_date or date.today()).normalize()
    all_source_labels = [label for label, _ in _source_path_pairs(relabel_paths, "education")]

    con = ddb.connect()
    spells_all = _load_matching_spells(
        con=con,
        paths=relabel_paths,
        school_name=school_name,
        grad_year=int(grad_year),
    )
    n_candidate_spells_before_immediate_phd_exclusion = len(spells_all)
    n_excluded_immediate_same_inst_phd_after_master = int(
        spells_all["exclude_immediate_same_inst_phd_after_master_ind"].fillna(False).astype(bool).sum()
    ) if not spells_all.empty else 0
    spells = (
        spells_all.loc[
            ~spells_all["exclude_immediate_same_inst_phd_after_master_ind"].fillna(False).astype(bool)
        ]
        .drop(columns=["exclude_immediate_same_inst_phd_after_master_ind"], errors="ignore")
        .reset_index(drop=True)
    )

    source_labels = (
        sorted(spells["source_dataset"].dropna().astype(str).unique().tolist())
        if not spells.empty
        else all_source_labels
    )
    latest_available_year = _latest_available_year(
        con=con,
        paths=relabel_paths,
        source_labels=source_labels,
    )
    positions = _load_positions(
        con=con,
        paths=relabel_paths,
        user_ids=spells["user_id"].tolist(),
        source_labels=source_labels,
    )

    positions_by_user: dict[int, pd.DataFrame] = {
        int(user_id): grp.copy()
        for user_id, grp in positions.groupby("user_id", sort=False)
    }

    spell_rows: list[dict[str, object]] = []
    for _, spell in spells.iterrows():
        user_id = int(spell["user_id"])
        user_positions = positions_by_user.get(user_id, pd.DataFrame())
        spell_audit = _compute_spell_audit(
            spell=spell,
            user_positions=user_positions,
            as_of_date=as_of_ts,
            min_position_days=int(min_position_days),
            entry_window_years=int(entry_window_years),
            latest_available_year=latest_available_year,
            cap_to_latest_available_year=relabel_paths.cap_to_latest_available_year,
        )
        row = spell.to_dict()
        row.update(spell_audit)
        spell_rows.append(row)

    report = _build_report(
        school_name=school_name,
        grad_year=int(grad_year),
        relabel_paths=relabel_paths,
        spell_rows=spell_rows,
        positions_by_user=positions_by_user,
        as_of_date=as_of_ts,
        min_position_days=int(min_position_days),
        entry_window_years=int(entry_window_years),
        latest_available_year=latest_available_year,
        source_labels=source_labels,
        n_candidate_spells_before_immediate_phd_exclusion=n_candidate_spells_before_immediate_phd_exclusion,
        n_excluded_immediate_same_inst_phd_after_master=n_excluded_immediate_same_inst_phd_after_master,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return output_path


def build_foia_audit_report(
    school_name: str,
    grad_year: int,
    output_path: str | Path | None = None,
    company_config_path: str | Path | None = None,
) -> Path:
    if output_path is None:
        output_path = default_foia_output_path(school_name=school_name, grad_year=int(grad_year))
    output_path = Path(output_path)

    foia_paths = _resolve_foia_paths(company_config_path)
    con = ddb.connect()
    foia_people = _load_foia_people(
        con=con,
        foia_path=foia_paths.foia_indiv,
        school_name=school_name,
        grad_year=int(grad_year),
    )
    report = _build_foia_report(
        school_name=school_name,
        grad_year=int(grad_year),
        foia_people=foia_people,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return output_path


def audit_school_grad_class(
    school_name: str,
    grad_year: int,
    output_path: str | Path | None = None,
    foia_output_path: str | Path | None = None,
    company_config_path: str | Path | None = None,
    relabel_config_path: str | Path | None = None,
    as_of_date: date | str | None = None,
    min_position_days: int = 365,
    entry_window_years: int = 1,
    preview_lines: int = 30,
    print_preview: bool = False,
) -> dict[str, object]:
    """
    IPython-friendly wrapper around the relabel + FOIA audit builders.

    Returns a small dict so interactive sessions can inspect both outputs
    without rereading the files.
    """
    written = build_audit_report(
        school_name=school_name,
        grad_year=int(grad_year),
        output_path=output_path,
        company_config_path=company_config_path,
        relabel_config_path=relabel_config_path,
        as_of_date=as_of_date,
        min_position_days=int(min_position_days),
        entry_window_years=int(entry_window_years),
    )
    foia_written = build_foia_audit_report(
        school_name=school_name,
        grad_year=int(grad_year),
        output_path=foia_output_path,
        company_config_path=company_config_path,
    )
    text = written.read_text()
    preview = "\n".join(text.splitlines()[: max(0, int(preview_lines))])
    foia_text = foia_written.read_text()
    foia_preview = "\n".join(foia_text.splitlines()[: max(0, int(preview_lines))])
    if print_preview:
        if preview:
            print(preview)
        if foia_preview:
            print("\n--- FOIA ---")
            print(foia_preview)
    return {
        "path": written,
        "text": text,
        "preview": preview,
        "foia_path": foia_written,
        "foia_text": foia_text,
        "foia_preview": foia_preview,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write txt audit reports for a relabel-indiv-model school x grad-class sample."
    )
    parser.add_argument("--school", required=True, help="School name as it appears in relabel_indiv_model outputs.")
    parser.add_argument("--grad-year", required=True, type=int, help="Graduation year (education end year).")
    parser.add_argument("--output", type=Path, default=None, help="Output txt path. Defaults to output/audits/school_grad_class_<school>_<year>.txt")
    parser.add_argument("--foia-output", type=Path, default=None, help="FOIA output txt path. Defaults to output/audits/school_grad_class_<school>_<year>_foia.txt")
    parser.add_argument(
        "--company-config",
        type=Path,
        default=None,
        help=f"Optional company_shift_share config path for the FOIA parquet (default: {COMPANY_CFG_DEFAULT}).",
    )
    parser.add_argument(
        "--relabel-config",
        type=Path,
        default=None,
        help=f"Optional relabel_indiv_model config path (default: {RELABEL_CFG_DEFAULT}).",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Optional date used only for open-ended-position duration checks in the supplemental first-job context.",
    )
    parser.add_argument(
        "--min-position-days",
        type=int,
        default=365,
        help="Minimum duration for the supplemental first post-grad qualifying job.",
    )
    parser.add_argument(
        "--entry-window-years",
        type=int,
        default=1,
        help="Years after graduation within which the supplemental first qualifying job must start.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output = args.output or default_output_path(args.school, int(args.grad_year))
    foia_output = args.foia_output or default_foia_output_path(args.school, int(args.grad_year))

    print(
        f"Building school audit for school='{args.school}', grad_year={int(args.grad_year)} "
        f"from relabel_indiv_model outputs"
    )
    written = build_audit_report(
        school_name=args.school,
        grad_year=int(args.grad_year),
        output_path=output,
        company_config_path=args.company_config,
        relabel_config_path=args.relabel_config,
        as_of_date=args.as_of_date,
        min_position_days=int(args.min_position_days),
        entry_window_years=int(args.entry_window_years),
    )
    foia_written = build_foia_audit_report(
        school_name=args.school,
        grad_year=int(args.grad_year),
        output_path=foia_output,
        company_config_path=args.company_config,
    )
    print(f"Wrote audit report: {written}")
    print(f"Wrote FOIA audit report: {foia_written}")


if __name__ == "__main__":
    if _running_in_ipython() and not _has_explicit_cli_args(sys.argv[1:]):
        print(
            "Interactive use:\n"
            "  import company_shift_share.audit_school_grad_class as audit\n"
            "  result = audit.audit_school_grad_class('Yale University', 2015)\n"
            "  result['path']\n"
            "  result['foia_path']\n"
            "  print(result['preview'])\n"
            "  print(result['foia_preview'])\n\n"
            "CLI use:\n"
            "  %run audit_school_grad_class.py --school \"Yale University\" --grad-year 2015"
        )
    else:
        main()
