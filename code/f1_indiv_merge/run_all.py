"""Run selected stages from the f1_indiv_merge scaffold."""

from __future__ import annotations

import argparse
import time
from builtins import print as _print
from functools import partial
from pathlib import Path

import src.config_loader as cfg_loader
from src.progress_tracker import mark_stage_complete
from src.pipeline_runtime import (
    STAGE_ORDER,
    StageDeferredError,
    coerce_bool,
    load_stage_module,
    resolve_stage_list,
    sanitize_ipykernel_argv,
)

print = partial(_print, flush=True)


def _print_stage_table(pipeline_cfg: dict) -> None:
    print("Available stages:")
    for stage_name in STAGE_ORDER:
        stage_cfg = cfg_loader.get_stage_config(pipeline_cfg, stage_name)
        enabled = coerce_bool(stage_cfg.get("enabled"), False)
        status = stage_cfg.get("status", "configured")
        print(f"  - {stage_name}: enabled={enabled} status={status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scaffolded f1_indiv_merge pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the root pipeline YAML. Defaults to f1_indiv_merge/pipeline.yaml.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Stage names to run. Defaults to build.default_stages from the pipeline config.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print configured stages and exit.",
    )
    parser.add_argument(
        "--allow-deferred",
        action="store_true",
        help="Skip scaffolded deferred stages instead of failing the run.",
    )
    testing_group = parser.add_mutually_exclusive_group()
    testing_group.add_argument("--testing", dest="testing", action="store_true")
    testing_group.add_argument("--no-testing", dest="testing", action="store_false")
    parser.set_defaults(testing=None)
    args = parser.parse_args(sanitize_ipykernel_argv())

    pipeline_cfg = cfg_loader.load_config(args.config)
    if args.list_stages:
        print(f"Using config: {cfg_loader.ACTIVE_CONFIG_PATH}")
        _print_stage_table(pipeline_cfg)
        return

    selected_stages = resolve_stage_list(pipeline_cfg, args.stages)
    default_testing = coerce_bool(pipeline_cfg.get("testing", {}).get("enabled"), False)
    testing = default_testing if args.testing is None else args.testing
    stop_on_deferred = coerce_bool(
        pipeline_cfg.get("build", {}).get("stop_on_deferred_stage"),
        True,
    )

    print(f"Using config: {cfg_loader.ACTIVE_CONFIG_PATH}")
    print(f"Stages: {', '.join(selected_stages)}")
    print(f"Testing: {testing}")

    for stage_name in selected_stages:
        module = load_stage_module(stage_name, pipeline_cfg)
        print("")
        print(f"[run_all] Starting {stage_name}")
        stage_t0 = time.perf_counter()
        try:
            module.run(
                config_path=args.config,
                pipeline_cfg=pipeline_cfg,
                testing=testing,
            )
            if not testing:
                mark_stage_complete(stage_name, time.perf_counter() - stage_t0)
        except StageDeferredError as exc:
            print(f"[run_all] {stage_name}: deferred")
            print(f"          {exc}")
            if stop_on_deferred and not args.allow_deferred:
                raise SystemExit(2) from exc
        print(f"[run_all] Finished {stage_name}")


if __name__ == "__main__":
    main()
