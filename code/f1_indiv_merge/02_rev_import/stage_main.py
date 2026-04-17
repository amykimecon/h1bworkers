"""Stage runner for 02_rev_import."""

from __future__ import annotations

import argparse
import sys
import time
from builtins import print as _print
from functools import partial
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
for _path in (STAGE_DIR, PIPELINE_ROOT, REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from src.config_loader import get_stage_config, load_config
from src.progress_tracker import mark_stage_complete
from src.pipeline_runtime import sanitize_ipykernel_argv

from common import coerce_int_list, resolve_first_existing
from compile_matches import compile_matched_user_artifacts
from education_scan import scan_wrds_education_shards
from final_imports import (
    consolidate_wrds_user_and_position_artifacts,
    import_wrds_user_and_position_artifacts,
)
from preprocess import build_foia_school_token_artifacts
from shard_planner import build_user_id_shard_manifest
from wrds_query import fetch_user_id_bounds, get_wrds_connection

print = partial(_print, flush=True)

STAGE_NAME = "02_rev_import"


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _resolve_scan_overrides(stage_cfg: dict, testing: bool) -> dict[str, int | list[int] | None]:
    if testing:
        selected_shard_ids = coerce_int_list(stage_cfg.get("testing_selected_shard_ids"))
        max_shards = stage_cfg.get("testing_max_shards")
        row_limit_per_shard = stage_cfg.get("testing_row_limit_per_shard")
        final_import_max_users = stage_cfg.get("testing_final_import_max_users")
        final_import_chunk_size = stage_cfg.get("testing_final_import_chunk_size")
    else:
        selected_shard_ids = coerce_int_list(stage_cfg.get("selected_shard_ids"))
        max_shards = stage_cfg.get("max_shards")
        row_limit_per_shard = stage_cfg.get("row_limit_per_shard")
        final_import_max_users = stage_cfg.get("final_import_max_users")
        final_import_chunk_size = stage_cfg.get("final_import_chunk_size")

    return {
        "selected_shard_ids": selected_shard_ids or None,
        "max_shards": int(max_shards) if max_shards is not None else None,
        "row_limit_per_shard": int(row_limit_per_shard) if row_limit_per_shard is not None else None,
        "final_import_max_users": int(final_import_max_users) if final_import_max_users is not None else None,
        "final_import_chunk_size": int(final_import_chunk_size) if final_import_chunk_size is not None else None,
    }


def _resolve_persist_match_outputs(stage_cfg: dict, testing: bool) -> bool:
    key = "testing_persist_match_outputs" if testing else "persist_match_outputs"
    return bool(stage_cfg.get(key, stage_cfg.get("persist_match_outputs", False)))


def _resolve_selected_shard_ids(
    *,
    config_selected_shard_ids: list[int] | None,
    selected_shard_ids_override: list[int] | None,
    shard_id_start: int | None,
    shard_id_end: int | None,
    n_shards: int,
) -> list[int] | None:
    if selected_shard_ids_override is None and shard_id_start is None and shard_id_end is None:
        return config_selected_shard_ids

    selected: set[int] = set(selected_shard_ids_override or [])
    if shard_id_start is not None or shard_id_end is not None:
        range_start = 0 if shard_id_start is None else int(shard_id_start)
        range_end = (int(n_shards) - 1) if shard_id_end is None else int(shard_id_end)
        if range_start < 0 or range_end < 0:
            raise ValueError("Shard id overrides must be non-negative.")
        if range_end < range_start:
            raise ValueError(
                f"shard_id_end must be >= shard_id_start, got {range_start=} {range_end=}"
            )
        selected.update(range(range_start, range_end + 1))

    out = sorted(selected)
    return out or None


def import_revelio_users(
    config_path: str | Path | None = None,
    pipeline_cfg: dict | None = None,
    testing: bool | None = None,
    selected_shard_ids: list[int] | None = None,
    shard_id_start: int | None = None,
    shard_id_end: int | None = None,
    max_shards: int | None = None,
    row_limit_per_shard: int | None = None,
    final_import_max_users: int | None = None,
    final_import_chunk_size: int | None = None,
    persist_match_outputs: bool | None = None,
    scan_only: bool = False,
    skip_scan: bool = False,
    db=None,
) -> dict[str, int | str | bool]:
    cfg = pipeline_cfg or load_config(config_path)
    stage_cfg = get_stage_config(cfg, STAGE_NAME)
    build_cfg = cfg.get("build", {})
    testing_cfg = cfg.get("testing", {})

    overwrite = bool(build_cfg.get("overwrite", False))
    effective_testing = bool(testing_cfg.get("enabled", False) if testing is None else testing)

    foia_input_path = resolve_first_existing(
        [
            stage_cfg.get("cleaned_foia_input_parquet"),
            stage_cfg.get("cleaned_foia_fallback_parquet"),
        ]
    )
    if foia_input_path is None:
        raise FileNotFoundError(
            f"{STAGE_NAME} could not find either the stage-01 cleaned FOIA output "
            "or the configured legacy FOIA fallback parquet."
        )

    n_shards = int(stage_cfg.get("n_shards", 3000))
    chunk_size = int(stage_cfg.get("final_import_chunk_size", 10000))
    scan_start_year_threshold = int(stage_cfg.get("shard_scan_start_year_threshold", 2000))
    scan_end_year_threshold = int(stage_cfg.get("shard_scan_end_year_threshold", 2004))
    filtered_out_sample_n = int(stage_cfg.get("testing_filtered_out_sample_n", 5)) if effective_testing else 0
    overrides = _resolve_scan_overrides(stage_cfg, effective_testing)
    overrides["selected_shard_ids"] = _resolve_selected_shard_ids(
        config_selected_shard_ids=overrides["selected_shard_ids"],
        selected_shard_ids_override=coerce_int_list(selected_shard_ids) if selected_shard_ids else None,
        shard_id_start=shard_id_start,
        shard_id_end=shard_id_end,
        n_shards=n_shards,
    )
    if max_shards is not None:
        overrides["max_shards"] = int(max_shards)
    if row_limit_per_shard is not None:
        overrides["row_limit_per_shard"] = int(row_limit_per_shard)
    if final_import_max_users is not None:
        overrides["final_import_max_users"] = int(final_import_max_users)
    if final_import_chunk_size is not None:
        overrides["final_import_chunk_size"] = int(final_import_chunk_size)
    if scan_only and skip_scan:
        raise ValueError("scan_only and skip_scan cannot both be true.")
    effective_chunk_size = overrides["final_import_chunk_size"] or chunk_size
    if final_import_max_users is None:
        overrides["final_import_max_users"] = None
    use_direct_shard_imports = True
    if final_import_max_users is not None:
        use_direct_shard_imports = False
    resolved_persist_match_outputs = _resolve_persist_match_outputs(stage_cfg, effective_testing)
    if persist_match_outputs is not None:
        resolved_persist_match_outputs = bool(persist_match_outputs)
    write_match_outputs = resolved_persist_match_outputs or not use_direct_shard_imports
    own_db = db is None
    if own_db:
        db = get_wrds_connection(wrds_username=stage_cfg.get("wrds_username"))

    results: dict[str, int | str | bool] = {
        "testing": effective_testing,
        "n_shards": n_shards,
        "scan_only": scan_only,
        "skip_scan": skip_scan,
        "use_direct_shard_imports": use_direct_shard_imports,
        "write_match_outputs": write_match_outputs,
        "final_import_max_users": overrides["final_import_max_users"],
    }

    print(f"[{STAGE_NAME}] Building FOIA token and regex artifacts")
    token_stats = build_foia_school_token_artifacts(
        foia_parquet_path=foia_input_path,
        token_artifact_csv=stage_cfg["token_artifact_csv"],
        compiled_regex_txt=stage_cfg["compiled_regex_txt"],
        foia_school_column=str(stage_cfg.get("foia_school_column", "school_name")),
        token_min_len=int(stage_cfg.get("token_min_len", 3)),
        token_top_n=stage_cfg.get("token_top_n"),
        token_stopwords=_as_list(stage_cfg.get("token_stopwords")),
        overwrite=overwrite,
    )
    results.update(token_stats)

    shard_manifest_path = Path(stage_cfg["shard_manifest_parquet"])
    user_id_min_override = stage_cfg.get("user_id_min_override")
    user_id_max_override = stage_cfg.get("user_id_max_override")
    bounds_stats: dict[str, int | str] = {}
    if shard_manifest_path.exists() and not overwrite:
        print(f"[{STAGE_NAME}] Reusing existing shard manifest")
    else:
        if (user_id_min_override is None) ^ (user_id_max_override is None):
            raise ValueError("user_id_min_override and user_id_max_override must be set together.")
        if user_id_min_override is not None and user_id_max_override is not None:
            bounds_stats = {
                "user_id_bounds_source_relation": "config_override",
                "resolved_user_id_min": int(user_id_min_override),
                "resolved_user_id_max": int(user_id_max_override),
            }
        else:
            bounds_source_relation = str(
                stage_cfg.get("user_id_bounds_source_relation", "revelio.individual_user_education")
            )
            print(f"[{STAGE_NAME}] Scanning WRDS for global user_id bounds from {bounds_source_relation}")
            bounds_stats = fetch_user_id_bounds(
                db,
                source_relation=bounds_source_relation,
            )
        results.update(bounds_stats)

    print(f"[{STAGE_NAME}] Building range shard manifest")
    shard_stats = build_user_id_shard_manifest(
        shard_manifest_parquet=stage_cfg["shard_manifest_parquet"],
        n_shards=n_shards,
        user_id_min=int(bounds_stats["resolved_user_id_min"]) if bounds_stats else None,
        user_id_max=int(bounds_stats["resolved_user_id_max"]) if bounds_stats else None,
        overwrite=overwrite,
    )
    results.update(shard_stats)

    try:
        if not skip_scan:
            print(
                f"[{STAGE_NAME}] Scanning WRDS education shards "
                f"(testing={effective_testing}, selected_shard_ids={overrides['selected_shard_ids']}, "
                f"max_shards={overrides['max_shards']}, row_limit_per_shard={overrides['row_limit_per_shard']}, "
                f"direct_shard_imports={use_direct_shard_imports}, final_import_chunk_size={effective_chunk_size})"
            )
            scan_stats = scan_wrds_education_shards(
                compiled_regex_txt=stage_cfg["compiled_regex_txt"],
                shard_manifest_parquet=stage_cfg["shard_manifest_parquet"],
                matched_education_shard_dir=stage_cfg.get("matched_education_shard_dir"),
                matched_user_shard_dir=stage_cfg.get("matched_user_shard_dir"),
                wrds_users_chunk_dir=stage_cfg.get("wrds_users_chunk_dir") if use_direct_shard_imports else None,
                wrds_positions_chunk_dir=stage_cfg.get("wrds_positions_chunk_dir") if use_direct_shard_imports else None,
                wrds_username=stage_cfg.get("wrds_username"),
                selected_shard_ids=overrides["selected_shard_ids"],
                max_shards=overrides["max_shards"],
                row_limit_per_shard=overrides["row_limit_per_shard"],
                start_year_threshold=scan_start_year_threshold,
                end_year_threshold=scan_end_year_threshold,
                filtered_out_sample_n=filtered_out_sample_n,
                final_import_chunk_size=effective_chunk_size,
                write_match_outputs=write_match_outputs,
                overwrite=overwrite,
                db=db,
            )
            results.update(scan_stats)

        if scan_only:
            return results

        direct_marker_dir = Path(stage_cfg["wrds_users_chunk_dir"]) / "_shard_markers"
        have_completed_direct_shards = direct_marker_dir.exists() and any(direct_marker_dir.glob("*.json"))
        if use_direct_shard_imports and have_completed_direct_shards:
            print(f"[{STAGE_NAME}] Consolidating shard-level WRDS user and position outputs")
            import_stats = consolidate_wrds_user_and_position_artifacts(
                wrds_users_parquet=stage_cfg["wrds_users_parquet"],
                wrds_positions_parquet=stage_cfg["wrds_positions_parquet"],
                wrds_users_chunk_dir=stage_cfg["wrds_users_chunk_dir"],
                wrds_positions_chunk_dir=stage_cfg["wrds_positions_chunk_dir"],
                overwrite=overwrite,
            )
            results.update(import_stats)
        else:
            print(f"[{STAGE_NAME}] Compiling shard-level matches")
            match_stats = compile_matched_user_artifacts(
                matched_education_shard_dir=stage_cfg["matched_education_shard_dir"],
                matched_user_shard_dir=stage_cfg["matched_user_shard_dir"],
                matched_education_parquet=stage_cfg["matched_education_parquet"],
                matched_user_list_parquet=stage_cfg["matched_user_list_parquet"],
                matched_user_list_csv=stage_cfg.get("matched_user_list_csv"),
                overwrite=overwrite,
            )
            results.update(match_stats)

            print(
                f"[{STAGE_NAME}] Importing final WRDS user and position artifacts "
                f"(chunk_size={effective_chunk_size}, max_users={overrides['final_import_max_users']})"
            )
            import_stats = import_wrds_user_and_position_artifacts(
                matched_user_list_parquet=stage_cfg["matched_user_list_parquet"],
                wrds_users_parquet=stage_cfg["wrds_users_parquet"],
                wrds_positions_parquet=stage_cfg["wrds_positions_parquet"],
                wrds_users_chunk_dir=stage_cfg["wrds_users_chunk_dir"],
                wrds_positions_chunk_dir=stage_cfg["wrds_positions_chunk_dir"],
                wrds_username=stage_cfg.get("wrds_username"),
                chunk_size=effective_chunk_size,
                max_users=overrides["final_import_max_users"],
                overwrite=overwrite,
                db=db,
            )
            results.update(import_stats)
    finally:
        if own_db and db is not None:
            try:
                db.close()
            except Exception:
                pass
    return results


def run(
    config_path: str | Path | None = None,
    pipeline_cfg: dict | None = None,
    testing: bool | None = None,
    selected_shard_ids: list[int] | None = None,
    shard_id_start: int | None = None,
    shard_id_end: int | None = None,
    max_shards: int | None = None,
    row_limit_per_shard: int | None = None,
    final_import_max_users: int | None = None,
    final_import_chunk_size: int | None = None,
    persist_match_outputs: bool | None = None,
    scan_only: bool = False,
    skip_scan: bool = False,
    db=None,
) -> dict[str, int | str | bool]:
    return import_revelio_users(
        config_path=config_path,
        pipeline_cfg=pipeline_cfg,
        testing=testing,
        selected_shard_ids=selected_shard_ids,
        shard_id_start=shard_id_start,
        shard_id_end=shard_id_end,
        max_shards=max_shards,
        row_limit_per_shard=row_limit_per_shard,
        final_import_max_users=final_import_max_users,
        final_import_chunk_size=final_import_chunk_size,
        persist_match_outputs=persist_match_outputs,
        scan_only=scan_only,
        skip_scan=skip_scan,
        db=db,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 02_rev_import.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--selected-shard-ids",
        nargs="+",
        type=int,
        default=None,
        help="Explicit shard ids to scan. Overrides config shard selection.",
    )
    parser.add_argument(
        "--shard-id-start",
        type=int,
        default=None,
        help="Inclusive shard-id range start override.",
    )
    parser.add_argument(
        "--shard-id-end",
        type=int,
        default=None,
        help="Inclusive shard-id range end override.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Cap the number of selected shards processed in this run.",
    )
    parser.add_argument(
        "--row-limit-per-shard",
        type=int,
        default=None,
        help="Temporary WRDS LIMIT applied per shard query for debugging.",
    )
    parser.add_argument(
        "--final-import-max-users",
        type=int,
        default=None,
        help="Cap the number of matched user_ids imported in the legacy final-import path. Explicitly setting this re-enables matched-user intermediates.",
    )
    parser.add_argument(
        "--final-import-chunk-size",
        type=int,
        default=None,
        help="Override final WRDS import chunk size.",
    )
    persist_group = parser.add_mutually_exclusive_group()
    persist_group.add_argument(
        "--persist-match-outputs",
        dest="persist_match_outputs",
        action="store_true",
        help="Also write matched_education/matched_user shard outputs during the scan phase.",
    )
    persist_group.add_argument(
        "--no-persist-match-outputs",
        dest="persist_match_outputs",
        action="store_false",
        help="Do not write matched_education/matched_user shard outputs unless required for legacy final import mode.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Run token build, shard manifest, and shard-level scan/import work only; skip final consolidation.",
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="Skip shard scanning and only consolidate/import from existing shard outputs.",
    )
    testing_group = parser.add_mutually_exclusive_group()
    testing_group.add_argument("--testing", dest="testing", action="store_true")
    testing_group.add_argument("--no-testing", dest="testing", action="store_false")
    parser.set_defaults(testing=None, persist_match_outputs=None)
    args = parser.parse_args(sanitize_ipykernel_argv())
    cfg = load_config(args.config)
    effective_testing = bool(cfg.get("testing", {}).get("enabled", False) if args.testing is None else args.testing)
    t0 = time.perf_counter()
    out = run(
        config_path=args.config,
        pipeline_cfg=cfg,
        testing=args.testing,
        selected_shard_ids=args.selected_shard_ids,
        shard_id_start=args.shard_id_start,
        shard_id_end=args.shard_id_end,
        max_shards=args.max_shards,
        row_limit_per_shard=args.row_limit_per_shard,
        final_import_max_users=args.final_import_max_users,
        final_import_chunk_size=args.final_import_chunk_size,
        persist_match_outputs=args.persist_match_outputs,
        scan_only=args.scan_only,
        skip_scan=args.skip_scan,
    )
    is_partial_run = any(
        value is not None
        for value in [
            args.selected_shard_ids,
            args.shard_id_start,
            args.shard_id_end,
            args.max_shards,
            args.row_limit_per_shard,
            args.final_import_max_users,
        ]
    ) or args.scan_only
    if not effective_testing and not is_partial_run:
        mark_stage_complete(STAGE_NAME, time.perf_counter() - t0)
    print(f"[{STAGE_NAME}] Done.")
    for key, value in out.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
