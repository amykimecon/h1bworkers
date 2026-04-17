"""Run the local 05_indiv_merge stage."""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import sys
import time
from builtins import print as _print
from functools import partial
from pathlib import Path
from typing import Any

import duckdb

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = STAGE_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
for _path in (STAGE_DIR, PIPELINE_ROOT, REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import src.config_loader as cfg_loader
from src.pipeline_runtime import coerce_bool, sanitize_ipykernel_argv
from src.progress_tracker import mark_stage_complete

print = partial(_print, flush=True)

STAGE_NAME = "05_indiv_merge"
SPELL_REQUIRED_COLS = {
    "spell_id",
    "person_id",
    "user_id",
    "match_rank",
    "weight_norm",
    "total_score",
    "country_score",
    "n_match_filt",
}
PERSON_REQUIRED_COLS = {
    "person_id",
    "user_id",
    "person_match_rank",
    "person_weight_norm",
    "person_score_sum",
}


def _escape(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _load_local_merge_modules(config_path: str | Path | None):
    if config_path:
        os.environ["F1_INDIV_PIPELINE_CONFIG"] = str(config_path)
    import merge_config
    import merge_logic

    merge_config = importlib.reload(merge_config)
    merge_logic = importlib.reload(merge_logic)
    return merge_config, merge_logic


def _person_agg_output_path(stage_cfg: dict[str, Any]) -> str:
    configured = stage_cfg.get("person_agg_parquet")
    if configured:
        return str(configured)
    person_baseline_path = stage_cfg.get("person_baseline_parquet")
    if not person_baseline_path:
        raise KeyError("Stage-05 config must define `person_baseline_parquet`.")
    base_path = Path(str(person_baseline_path))
    suffix = "".join(base_path.suffixes)
    stem = base_path.name[:-len(suffix)] if suffix else base_path.name
    if "person_baseline" in stem:
        stem = stem.replace("person_baseline", "person_agg")
    else:
        stem = f"{stem}_agg"
    return str(base_path.with_name(f"{stem}{suffix}"))


def _resolve_person_shard_spec(stage_cfg: dict[str, Any]) -> dict[str, int | str] | None:
    raw_count = stage_cfg.get("person_shard_count")
    raw_id = stage_cfg.get("person_shard_id")
    if raw_count in (None, "", 0) and raw_id in (None, ""):
        return None
    if raw_count in (None, "", 0) or raw_id in (None, ""):
        raise ValueError("Stage-05 sharding requires both `person_shard_count` and `person_shard_id`.")
    shard_count = int(raw_count)
    shard_id = int(raw_id)
    if shard_count < 2:
        raise ValueError("Stage-05 sharding requires `person_shard_count >= 2`.")
    if shard_id < 0 or shard_id >= shard_count:
        raise ValueError("Stage-05 sharding requires `0 <= person_shard_id < person_shard_count`.")
    return {
        "person_shard_count": shard_count,
        "person_shard_id": shard_id,
        "person_shard_label": f"shard{shard_id:04d}of{shard_count:04d}",
    }


def _shard_output_path(path: str | Path, *, shard_count: int, shard_id: int) -> str:
    out_path = Path(path)
    suffix = "".join(out_path.suffixes)
    stem = out_path.name[:-len(suffix)] if suffix else out_path.name
    shard_label = f"shard{int(shard_id):04d}of{int(shard_count):04d}"
    return str(out_path.with_name(f"{stem}__{shard_label}{suffix}"))


def _configure_person_shard(
    cfg: dict[str, Any],
    *,
    shard_count: int | None,
    shard_id: int | None,
) -> dict[str, int | str] | None:
    stage_cfg = cfg_loader.get_stage_config(cfg, STAGE_NAME)
    if shard_count is not None or shard_id is not None:
        if shard_count is None or shard_id is None:
            raise ValueError("Stage-05 sharding requires both `shard_count` and `shard_id`.")
        stage_cfg["person_shard_count"] = int(shard_count)
        stage_cfg["person_shard_id"] = int(shard_id)
    return _resolve_person_shard_spec(stage_cfg)


def _build_shard_output_paths(
    stage_cfg: dict[str, Any],
    shard_spec: dict[str, int | str],
) -> dict[str, str]:
    shard_count = int(shard_spec["person_shard_count"])
    shard_id = int(shard_spec["person_shard_id"])
    return {
        "baseline_parquet": _shard_output_path(
            str(stage_cfg["baseline_parquet"]),
            shard_count=shard_count,
            shard_id=shard_id,
        ),
        "person_agg_parquet": _shard_output_path(
            _person_agg_output_path(stage_cfg),
            shard_count=shard_count,
            shard_id=shard_id,
        ),
    }


def _read_parquet_list_sql(paths: list[str | Path]) -> str:
    escaped = ", ".join(f"'{_escape(path)}'" for path in paths)
    return f"read_parquet([{escaped}])"


def _running_in_ipykernel() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


def _publish_interactive_namespace(namespace: dict[str, Any]) -> None:
    globals().update(namespace)
    try:
        from IPython import get_ipython
    except ImportError:
        return
    shell = get_ipython()
    user_ns = getattr(shell, "user_ns", None)
    if isinstance(user_ns, dict):
        user_ns.update(namespace)


def _build_ipython_namespace(merge_module, *, testing: bool) -> dict[str, Any]:
    tables = merge_module.get_runtime_table_names(testing=testing)

    def audit_person(person_id: int, **kwargs):
        return merge_module.audit_person_candidates(
            person_id,
            testing=testing,
            con=merge_module.con_f1,
            **kwargs,
        )

    def check_person(person_id: int, **kwargs):
        return merge_module.check_person(
            person_id,
            testing=testing,
            con=merge_module.con_f1,
            **kwargs,
        )

    return {
        "merge": merge_module,
        "con": merge_module.con_f1,
        "tables": tables,
        "audit_person": audit_person,
        "check_person": check_person,
    }


def _apply_interactive_testing_overrides(
    merge_module,
    *,
    testing: bool,
    sample_n_persons: int | None,
    individual_keys: list[str] | None,
    person_ids: list[int] | None,
) -> None:
    if not testing:
        return
    merge_module.cfg.TESTING_ENABLED = True
    if individual_keys:
        normalized_individual_keys = [str(individual_key).strip() for individual_key in individual_keys if str(individual_key).strip()]
        merge_module.cfg.TESTING_INDIVIDUAL_KEYS = normalized_individual_keys or None
        merge_module.cfg.TESTING_PERSON_IDS = None
        merge_module.cfg.TESTING_SAMPLE_N_PERSONS = len(normalized_individual_keys)
        print(
            f"[{STAGE_NAME}] Interactive testing override: "
            f"individual_keys={normalized_individual_keys}, "
            f"materialize_intermediate_tables={merge_module.cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES}"
        )
        return
    if person_ids:
        normalized_person_ids = [int(person_id) for person_id in person_ids]
        merge_module.cfg.TESTING_PERSON_IDS = normalized_person_ids
        merge_module.cfg.TESTING_INDIVIDUAL_KEYS = None
        merge_module.cfg.TESTING_SAMPLE_N_PERSONS = len(normalized_person_ids)
        print(
            f"[{STAGE_NAME}] Interactive testing override: "
            f"person_ids={normalized_person_ids}, "
            f"materialize_intermediate_tables={merge_module.cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES}"
        )
        return

    target_sample_n = 10 if sample_n_persons is None else max(1, int(sample_n_persons))
    current_sample_n = int(getattr(merge_module.cfg, "TESTING_SAMPLE_N_PERSONS", target_sample_n))
    effective_sample_n = min(current_sample_n, target_sample_n)
    merge_module.cfg.TESTING_INDIVIDUAL_KEYS = None
    merge_module.cfg.TESTING_PERSON_IDS = None
    merge_module.cfg.TESTING_SAMPLE_N_PERSONS = effective_sample_n
    print(
        f"[{STAGE_NAME}] Interactive testing override: "
        f"sample_n_persons={effective_sample_n}, "
        f"materialize_intermediate_tables={merge_module.cfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES}"
    )


def _require_parquet(
    con: duckdb.DuckDBPyConnection,
    path: str | Path,
    *,
    required_cols: set[str],
    label: str,
) -> int:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{label} not written: {path_obj}")
    cols = {
        row[0]
        for row in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{_escape(path_obj)}')"
        ).fetchall()
    }
    missing = sorted(required_cols - cols)
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    n_rows = int(
        con.sql(
            f"SELECT COUNT(*) FROM read_parquet('{_escape(path_obj)}')"
        ).fetchone()[0]
    )
    if n_rows <= 0:
        raise ValueError(f"{label} is empty: {path_obj}")
    return n_rows


def _run_acceptance_checks(
    stage_cfg: dict[str, Any],
    merge_cfg,
    merge_module,
) -> None:
    con = merge_module.get_duckdb_connection()
    try:
        spell_outputs = {
            "baseline": stage_cfg["baseline_parquet"],
            "mult2": stage_cfg["mult2_parquet"],
            "mult4": stage_cfg["mult4_parquet"],
            "mult6": stage_cfg["mult6_parquet"],
            "strict": stage_cfg["strict_parquet"],
        }
        person_outputs = {
            "person_baseline": stage_cfg["person_baseline_parquet"],
            "person_strict": stage_cfg["person_strict_parquet"],
        }

        for label, path in spell_outputs.items():
            _require_parquet(con, path, required_cols=SPELL_REQUIRED_COLS, label=f"{label} spell output")
        for label, path in person_outputs.items():
            _require_parquet(con, path, required_cols=PERSON_REQUIRED_COLS, label=f"{label} output")

        strict_bad = int(
            con.sql(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{_escape(stage_cfg["strict_parquet"])}')
                WHERE match_rank != 1
                   OR weight_norm < {merge_cfg.STRICT_MIN_WEIGHT_NORM}
                   OR total_score < {merge_cfg.STRICT_MIN_TOTAL_SCORE}
                   OR country_score < {merge_cfg.STRICT_MIN_COUNTRY_SCORE}
                """
            ).fetchone()[0]
        )
        if strict_bad:
            raise ValueError(f"Strict spell output violated threshold invariants on {strict_bad} rows.")

        person_rank_bad = int(
            con.sql(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{_escape(stage_cfg["person_baseline_parquet"])}')
                WHERE person_match_rank != 1
                """
            ).fetchone()[0]
        )
        if person_rank_bad:
            raise ValueError(f"Person baseline output contains {person_rank_bad} non-rank-1 rows.")

        duplicate_persons = int(
            con.sql(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT person_id
                    FROM read_parquet('{_escape(stage_cfg["person_baseline_parquet"])}')
                    GROUP BY person_id
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        if duplicate_persons:
            raise ValueError(f"Person baseline output contains {duplicate_persons} duplicated person_ids.")

        strict_person_bad = int(
            con.sql(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{_escape(stage_cfg["person_strict_parquet"])}')
                WHERE person_match_rank != 1
                   OR person_weight_norm < {merge_cfg.STRICT_PERSON_MIN_WEIGHT_NORM}
                """
            ).fetchone()[0]
        )
        if strict_person_bad:
            raise ValueError(f"Person strict output violated strict-person invariants on {strict_person_bad} rows.")

        print(f"[{STAGE_NAME}] Acceptance checks passed")
    finally:
        con.close()


def _compare_reference_outputs(stage_cfg: dict[str, Any], merge_cfg, merge_module) -> None:
    if not merge_cfg.COMPARE_TO_REFERENCE_OUTPUTS:
        return

    comparisons = [
        ("spell_baseline", stage_cfg["baseline_parquet"], merge_cfg.REFERENCE_SPELL_BASELINE_PARQUET, ["spell_id", "person_id", "user_id"]),
        ("spell_strict", stage_cfg["strict_parquet"], merge_cfg.REFERENCE_SPELL_STRICT_PARQUET, ["spell_id", "person_id", "user_id"]),
        ("person_baseline", stage_cfg["person_baseline_parquet"], merge_cfg.REFERENCE_PERSON_BASELINE_PARQUET, ["person_id", "user_id"]),
    ]
    con = merge_module.get_duckdb_connection()
    try:
        for label, built_path, ref_path, key_cols in comparisons:
            if not ref_path:
                print(f"[{STAGE_NAME}] Reference comparison skipped for {label}: no reference path configured")
                continue
            ref_file = Path(ref_path)
            if not ref_file.exists():
                print(f"[{STAGE_NAME}] Reference comparison skipped for {label}: missing {ref_file}")
                continue

            key_sql = ", ".join(key_cols)
            built_stats = con.sql(
                f"""
                SELECT COUNT(*) AS n_rows, COUNT(DISTINCT ({key_sql})) AS n_keys
                FROM read_parquet('{_escape(built_path)}')
                """
            ).df().iloc[0]
            ref_stats = con.sql(
                f"""
                SELECT COUNT(*) AS n_rows, COUNT(DISTINCT ({key_sql})) AS n_keys
                FROM read_parquet('{_escape(ref_file)}')
                """
            ).df().iloc[0]
            if int(built_stats["n_rows"]) != int(ref_stats["n_rows"]) or int(built_stats["n_keys"]) != int(ref_stats["n_keys"]):
                raise ValueError(
                    f"Reference comparison failed for {label}: "
                    f"built rows/keys={int(built_stats['n_rows'])}/{int(built_stats['n_keys'])}, "
                    f"reference rows/keys={int(ref_stats['n_rows'])}/{int(ref_stats['n_keys'])}"
                )
            print(
                f"[{STAGE_NAME}] Reference comparison passed for {label}: "
                f"{int(built_stats['n_rows']):,} rows, {int(built_stats['n_keys']):,} distinct keys"
            )
    finally:
        con.close()


def _run_post_build_checks(
    *,
    stage_cfg: dict[str, Any],
    merge_cfg,
    merge_module,
    testing_enabled: bool,
    run_acceptance_checks: bool,
    compare_reference_outputs: bool | None,
) -> None:
    if testing_enabled:
        return
    if run_acceptance_checks:
        _run_acceptance_checks(stage_cfg, merge_cfg, merge_module)
    should_compare = (
        merge_cfg.COMPARE_TO_REFERENCE_OUTPUTS
        if compare_reference_outputs is None
        else bool(compare_reference_outputs)
    )
    if should_compare:
        merge_cfg.COMPARE_TO_REFERENCE_OUTPUTS = True
        _compare_reference_outputs(stage_cfg, merge_cfg, merge_module)


def _write_final_outputs_from_relations(
    *,
    con: duckdb.DuckDBPyConnection,
    stage_cfg: dict[str, Any],
    merge_cfg,
    merge_module,
    baseline_relation: str,
    person_agg_relation: str,
    overwrite: bool,
    renumber_spell_ids: bool = False,
) -> dict[str, Any]:
    assignment_label = (
        "global 1:1 assignment"
        if merge_cfg.BUILD_ENFORCE_INDIVIDUAL_ONE_TO_ONE
        else "per-person rank-1 only"
    )
    print(f"  Resolving person-user matches via {assignment_label}...", end=" ", flush=True)
    t_assign = time.perf_counter()
    assigned_people_df = merge_module._solve_individual_assignment(
        con,
        person_agg_relation,
        enforce_one_to_one=merge_cfg.BUILD_ENFORCE_INDIVIDUAL_ONE_TO_ONE,
    )
    con.execute("DROP VIEW IF EXISTS _person_assigned_df_view")
    con.register("_person_assigned_df_view", assigned_people_df)
    assigned_people_tab = "_f1_person_assigned"
    merge_module.materialize_table(assigned_people_tab, "SELECT * FROM _person_assigned_df_view", con=con)
    con.execute("DROP VIEW IF EXISTS _person_assigned_df_view")
    print(f"{len(assigned_people_df):,} assigned pairs ({merge_module._fmt_elapsed(time.perf_counter() - t_assign)})")

    person_rank1 = con.sql(
        f"""
        SELECT
            COUNT(DISTINCT person_id)   AS n_persons,
            COUNT(DISTINCT user_id)     AS n_users,
            SUM(has_employer_match_ind) AS n_with_employer,
            ROUND(AVG(person_weight_norm), 3) AS avg_weight_norm,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY person_score_sum), 3) AS median_score
        FROM {assigned_people_tab}
        """
    ).df().iloc[0]
    pct_emp = 100 * float(person_rank1["n_with_employer"]) / max(1, int(person_rank1["n_persons"]))
    print(
        f"  person matches: {int(person_rank1['n_persons']):,} persons, "
        f"{int(person_rank1['n_users']):,} distinct user_ids"
    )
    print(f"  With employer confirmation: {int(person_rank1['n_with_employer']):,} ({pct_emp:.1f}%)")
    print(
        f"  avg person_weight_norm={float(person_rank1['avg_weight_norm']):.3f}, "
        f"median person_score_sum={float(person_rank1['median_score']):.3f}"
    )

    assigned_join = (
        f"JOIN {assigned_people_tab} AS ap "
        "ON b.person_id = ap.person_id AND b.user_id = ap.user_id"
    )
    print("\n[Stage 6: Writing output parquets]")
    print("  Writing spell-level outputs...")
    assigned_baseline_tab = None
    baseline_output_q = (
        f"SELECT b.* FROM {baseline_relation} AS b {assigned_join} WHERE b.match_rank = 1"
    )
    if renumber_spell_ids:
        assigned_baseline_tab = "_f1_baseline_assigned_renumbered"
        merge_module.materialize_table(
            assigned_baseline_tab,
            f"""
            WITH assigned AS (
                {baseline_output_q}
            )
            SELECT
                * EXCLUDE (spell_id),
                ROW_NUMBER() OVER (
                    ORDER BY
                        person_id,
                        f1_school_name,
                        f1_country_std,
                        f1_degree_level,
                        f1_prog_start_year,
                        user_id
                ) AS spell_id
            FROM assigned
            """,
            con=con,
        )
        baseline_output_q = f"SELECT * FROM {assigned_baseline_tab}"
    merge_module.write_query_to_parquet(
        query=baseline_output_q,
        out_path=str(stage_cfg["baseline_parquet"]),
        overwrite=overwrite,
        con=con,
    )

    for cutoff, out_path in [
        (2, stage_cfg["mult2_parquet"]),
        (4, stage_cfg["mult4_parquet"]),
        (6, stage_cfg["mult6_parquet"]),
    ]:
        if assigned_baseline_tab is not None:
            mult_q = f"SELECT * FROM {assigned_baseline_tab} WHERE n_match_filt <= {cutoff}"
        else:
            mult_q = (
                f"SELECT b.* FROM {baseline_relation} AS b {assigned_join} "
                f"WHERE b.match_rank = 1 AND b.n_match_filt <= {cutoff}"
            )
        mult_counts = merge_module._f1_merge_stage_counts(mult_q, con=con)
        merge_module._print_merge_stage(f"mult{cutoff}", mult_counts)
        merge_module.write_query_to_parquet(
            query=mult_q,
            out_path=str(out_path),
            overwrite=overwrite,
            con=con,
        )

    if assigned_baseline_tab is not None:
        strict_q = f"SELECT * FROM ({merge_module._build_f1_stage_strict_query(assigned_baseline_tab)}) AS s"
    else:
        strict_q = (
            f"SELECT s.* FROM ({merge_module._build_f1_stage_strict_query(baseline_relation)}) AS s "
            f"JOIN {assigned_people_tab} AS ap "
            "ON s.person_id = ap.person_id AND s.user_id = ap.user_id"
        )
    strict_counts = merge_module._f1_merge_stage_counts(strict_q, con=con)
    merge_module._print_merge_stage("strict", strict_counts)
    merge_module.write_query_to_parquet(
        query=strict_q,
        out_path=str(stage_cfg["strict_parquet"]),
        overwrite=overwrite,
        con=con,
    )

    base_counts = merge_module._f1_merge_stage_counts(baseline_output_q, con=con)

    print("  Writing person-level outputs...")
    merge_module.write_query_to_parquet(
        query=f"SELECT * FROM {assigned_people_tab}",
        out_path=str(stage_cfg["person_baseline_parquet"]),
        overwrite=overwrite,
        con=con,
    )

    strict_person_min_wn = merge_cfg.STRICT_PERSON_MIN_WEIGHT_NORM
    strict_person_q = (
        f"SELECT * FROM {assigned_people_tab} "
        f"WHERE person_weight_norm >= {strict_person_min_wn}"
    )
    strict_person_counts = con.sql(
        f"""
        SELECT COUNT(DISTINCT person_id) AS n_persons, COUNT(DISTINCT user_id) AS n_users
        FROM ({strict_person_q})
        """
    ).df().iloc[0]
    print(
        f"  person_strict: {int(strict_person_counts['n_persons']):,} persons, "
        f"{int(strict_person_counts['n_users']):,} users "
        f"(person_weight_norm >= {strict_person_min_wn})"
    )
    merge_module.write_query_to_parquet(
        query=strict_person_q,
        out_path=str(stage_cfg["person_strict_parquet"]),
        overwrite=overwrite,
        con=con,
    )

    print("\n" + "=" * 70)
    print("F1 MERGE COMPLETE — Summary")
    print(f"  run_tag:  {merge_cfg.RUN_TAG}")
    print(
        f"  spell/baseline: {base_counts['n_persons']:,} persons, "
        f"{base_counts['n_spells']:,} spells, "
        f"{base_counts['mult']:.2f}x mult"
    )
    print(
        f"  spell/strict:   {strict_counts['n_persons']:,} persons, "
        f"{strict_counts['n_spells']:,} spells, "
        f"{strict_counts['mult']:.2f}x mult"
    )
    print(
        f"  person/baseline:{int(person_rank1['n_persons']):,} persons, "
        f"{int(person_rank1['n_users']):,} users "
        f"({pct_emp:.1f}% with employer confirmation)"
    )
    print(
        f"  person/strict:  {int(strict_person_counts['n_persons']):,} persons, "
        f"{int(strict_person_counts['n_users']):,} users "
        f"(weight_norm >= {strict_person_min_wn})"
    )
    print("=" * 70)
    return {
        "assigned_people_tab": assigned_people_tab,
        "baseline_output_q": baseline_output_q,
        "person_baseline_n": int(person_rank1["n_persons"]),
    }


def _merge_stage05_sharded_outputs(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    *,
    shard_count: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(pipeline_cfg or cfg_loader.load_config(config_path))
    stage_cfg = cfg_loader.get_stage_config(cfg, STAGE_NAME)
    effective_shard_count = shard_count
    if effective_shard_count is None:
        raw_value = stage_cfg.get("person_shard_count")
        if raw_value in (None, "", 0):
            raise ValueError("Shard merge requires `shard_count` (or `person_shard_count` in stage config).")
        effective_shard_count = int(raw_value)
    if int(effective_shard_count) < 2:
        raise ValueError("Shard merge requires `shard_count >= 2`.")

    effective_config_path = config_path or cfg_loader.ACTIVE_CONFIG_PATH
    merge_cfg, merge_module = _load_local_merge_modules(effective_config_path)
    overwrite = bool(merge_cfg.BUILD_OVERWRITE)
    baseline_path = str(stage_cfg["baseline_parquet"])
    person_agg_path = _person_agg_output_path(stage_cfg)
    baseline_shard_paths = [
        _shard_output_path(baseline_path, shard_count=int(effective_shard_count), shard_id=shard_id)
        for shard_id in range(int(effective_shard_count))
    ]
    person_agg_shard_paths = [
        _shard_output_path(person_agg_path, shard_count=int(effective_shard_count), shard_id=shard_id)
        for shard_id in range(int(effective_shard_count))
    ]
    for label, paths in {
        "baseline_parquet": baseline_shard_paths,
        "person_agg_parquet": person_agg_shard_paths,
    }.items():
        existing = [Path(path).exists() for path in paths]
        if not all(existing):
            missing = [path for path, exists in zip(paths, existing) if not exists]
            raise FileNotFoundError(
                f"Cannot merge `{label}` because some shard outputs are missing: {missing}"
            )

    con = merge_module.get_duckdb_connection()
    try:
        print(
            f"[{STAGE_NAME}] Merging shard person aggregates from "
            f"{int(effective_shard_count)} shard files"
        )
        merge_module.materialize_table(
            "_f1_person_agg_merged",
            f"SELECT * FROM {_read_parquet_list_sql(person_agg_shard_paths)}",
            con=con,
        )
        print(
            f"[{STAGE_NAME}] Merging shard baseline spell outputs from "
            f"{int(effective_shard_count)} shard files"
        )
        merge_module.materialize_table(
            "_f1_baseline_rank1_merged",
            f"SELECT * FROM {_read_parquet_list_sql(baseline_shard_paths)}",
            con=con,
        )
        _write_final_outputs_from_relations(
            con=con,
            stage_cfg=stage_cfg,
            merge_cfg=merge_cfg,
            merge_module=merge_module,
            baseline_relation="_f1_baseline_rank1_merged",
            person_agg_relation="_f1_person_agg_merged",
            overwrite=overwrite,
            renumber_spell_ids=True,
        )
    finally:
        con.close()

    return {
        "person_shard_count": int(effective_shard_count),
        "merged_outputs": [
            "baseline_parquet",
            "mult2_parquet",
            "mult4_parquet",
            "mult6_parquet",
            "strict_parquet",
            "person_baseline_parquet",
            "person_strict_parquet",
        ],
    }


def run(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    testing: bool | None = None,
    run_merge: bool = True,
    run_acceptance_checks: bool = True,
    compare_reference_outputs: bool | None = None,
    shard_count: int | None = None,
    shard_id: int | None = None,
) -> None:
    cfg = pipeline_cfg or cfg_loader.load_config(config_path)
    stage_cfg = cfg_loader.get_stage_config(cfg, STAGE_NAME)
    effective_config_path = config_path or cfg_loader.ACTIVE_CONFIG_PATH
    testing_enabled = coerce_bool(cfg.get("testing", {}).get("enabled"), False) if testing is None else bool(testing)
    shard_spec = _configure_person_shard(cfg, shard_count=shard_count, shard_id=shard_id)
    shard_output_paths = _build_shard_output_paths(stage_cfg, shard_spec) if shard_spec else None

    print(f"Pipeline config: {cfg_loader.ACTIVE_CONFIG_PATH}")
    merge_cfg, merge_module = _load_local_merge_modules(effective_config_path)

    if run_merge:
        print(f"[{STAGE_NAME}] Building merge outputs (testing={testing_enabled})")
        merge_module.build_f1_merge_inputs(
            testing=testing_enabled,
            person_shard_count=(int(shard_spec["person_shard_count"]) if shard_spec else None),
            person_shard_id=(int(shard_spec["person_shard_id"]) if shard_spec else None),
            shard_output_paths=shard_output_paths,
        )

    if shard_spec is None:
        _run_post_build_checks(
            stage_cfg=stage_cfg,
            merge_cfg=merge_cfg,
            merge_module=merge_module,
            testing_enabled=testing_enabled,
            run_acceptance_checks=run_acceptance_checks,
            compare_reference_outputs=compare_reference_outputs,
        )


def launch_ipython_session(
    config_path: str | Path | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    testing: bool = True,
    sample_n_persons: int | None = None,
    individual_keys: list[str] | None = None,
    person_ids: list[int] | None = None,
) -> dict[str, Any]:
    cfg = pipeline_cfg or cfg_loader.load_config(config_path)
    effective_config_path = config_path or cfg_loader.ACTIVE_CONFIG_PATH
    _, merge_module = _load_local_merge_modules(effective_config_path)
    _apply_interactive_testing_overrides(
        merge_module,
        testing=testing,
        sample_n_persons=sample_n_persons,
        individual_keys=individual_keys,
        person_ids=person_ids,
    )
    print(f"[{STAGE_NAME}] Building interactive tables (testing={testing})")
    merge_module.build_f1_merge_inputs(testing=testing, con=merge_module.con_f1)
    namespace = _build_ipython_namespace(merge_module, testing=testing)

    if _running_in_ipykernel():
        _publish_interactive_namespace(namespace)
        print(
            f"[{STAGE_NAME}] Detected ipykernel; published merge, con, tables, "
            f"audit_person, and check_person to the interactive namespace."
        )
        return namespace

    try:
        from IPython import start_ipython
    except ImportError as exc:
        raise RuntimeError("IPython is not installed in this environment.") from exc

    print(
        f"[{STAGE_NAME}] Launching IPython with merge, con, tables, "
        f"audit_person, and check_person ({len(namespace['tables'])} table handles)"
    )
    start_ipython(argv=[], user_ns=namespace)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local 05_indiv_merge stage.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-acceptance-checks", action="store_true")
    parser.add_argument("--compare-reference-outputs", action="store_true")
    parser.add_argument("--ipython", action="store_true")
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--merge-shards", action="store_true")
    testing_group = parser.add_mutually_exclusive_group()
    testing_group.add_argument("--testing", dest="testing", action="store_true")
    testing_group.add_argument("--no-testing", dest="testing", action="store_false")
    parser.set_defaults(testing=None)
    clean_argv = sanitize_ipykernel_argv()
    args = parser.parse_args(clean_argv)
    auto_ipython = _running_in_ipykernel() and not clean_argv

    cfg = cfg_loader.load_config(args.config)
    effective_testing = coerce_bool(cfg.get("testing", {}).get("enabled"), False) if args.testing is None else bool(args.testing)
    sharding_requested = bool(args.shard_count is not None or args.shard_id is not None)
    if not sharding_requested:
        stage_shard_spec = _resolve_person_shard_spec(cfg_loader.get_stage_config(cfg, STAGE_NAME))
        sharding_requested = stage_shard_spec is not None
    t0 = time.perf_counter()
    if args.merge_shards:
        if args.shard_id is not None:
            raise ValueError("`--merge-shards` does not accept `--shard-id`.")
        merge_outputs = _merge_stage05_sharded_outputs(
            config_path=args.config,
            pipeline_cfg=cfg,
            shard_count=args.shard_count,
        )
        print(f"[{STAGE_NAME}] merged shard outputs: {merge_outputs.get('merged_outputs', [])}")
        if not effective_testing:
            merge_cfg, merge_module = _load_local_merge_modules(args.config or cfg_loader.ACTIVE_CONFIG_PATH)
            _run_post_build_checks(
                stage_cfg=cfg_loader.get_stage_config(cfg, STAGE_NAME),
                merge_cfg=merge_cfg,
                merge_module=merge_module,
                testing_enabled=effective_testing,
                run_acceptance_checks=not args.skip_acceptance_checks,
                compare_reference_outputs=(True if args.compare_reference_outputs else None),
            )
            mark_stage_complete(STAGE_NAME, time.perf_counter() - t0)
        return
    if args.ipython or auto_ipython:
        if auto_ipython and not args.ipython:
            print(f"[{STAGE_NAME}] Detected ipykernel execution; auto-launching interactive session setup.")
        else:
            print(f"[{STAGE_NAME}] Launching IPython session instead of full stage execution.")
        launch_ipython_session(
            config_path=args.config,
            pipeline_cfg=cfg,
            testing=True,
        )
        return
    run(
        config_path=args.config,
        pipeline_cfg=cfg,
        testing=args.testing,
        run_merge=not args.skip_merge,
        run_acceptance_checks=not args.skip_acceptance_checks,
        compare_reference_outputs=args.compare_reference_outputs,
        shard_count=args.shard_count,
        shard_id=args.shard_id,
    )
    if not effective_testing and not sharding_requested:
        mark_stage_complete(STAGE_NAME, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
