#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly STAGE_MAIN="${SCRIPT_DIR}/stage_main.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${PIPELINE_ROOT}/pipeline.yaml"
SHARD_COUNT=16
MAX_PARALLEL=1
RUN_MERGE=1
MERGE_ONLY=0
LOG_DIR="${PIPELINE_ROOT}/logs/stage05_shards_$(date +%Y%m%d_%H%M%S)"

declare -a EXTRA_STAGE_ARGS=()
declare -a ACTIVE_PIDS=()
declare -A PID_TO_SHARD=()

usage() {
  cat <<'EOF'
Usage: run_stage05_shards.sh [wrapper-options] [-- stage_main_options]

Runs every stage-05 shard, then optionally merges them back into the standard
stage-05 output paths. The wrapper defaults to sequential execution.

Wrapper options:
  --config PATH           Pipeline config path. Default: f1_indiv_merge/pipeline.yaml
  --python BIN            Python executable to use. Default: $PYTHON_BIN or python
  --shard-count N         Number of shards to run. Default: 16
  --max-parallel N        Concurrent shard workers. Default: 1
  --log-dir PATH          Directory for per-shard logs
  --no-merge              Run shard jobs only; do not merge at the end
  --merge-only            Skip shard jobs and only run the merge step
  -h, --help              Show this help text

Any arguments after '--' are forwarded to stage_main.py for both shard runs and
the merge step. Example forwarded flags: --no-testing, --testing,
--skip-acceptance-checks, --compare-reference-outputs.

Examples:
  run_stage05_shards.sh --shard-count 16 -- --no-testing
  run_stage05_shards.sh --shard-count 16 --max-parallel 4 -- --no-testing
  run_stage05_shards.sh --shard-count 16 --merge-only -- --no-testing
EOF
}

die() {
  echo "[stage05-shards] $*" >&2
  exit 2
}

is_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

log_path_for_shard() {
  local shard_id="$1"
  printf '%s/shard_%04d.log' "$LOG_DIR" "$shard_id"
}

print_cmd() {
  printf '%q' "$1"
  shift || true
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

print_log_tail() {
  local log_file="$1"
  if [[ -f "$log_file" ]]; then
    echo "[stage05-shards] tail of ${log_file}:" >&2
    tail -n 40 "$log_file" >&2 || true
  fi
}

build_shard_cmd() {
  local shard_id="$1"
  SHARD_CMD=(
    "$PYTHON_BIN"
    "$STAGE_MAIN"
    --config "$CONFIG_PATH"
    --shard-count "$SHARD_COUNT"
    --shard-id "$shard_id"
  )
  SHARD_CMD+=("${EXTRA_STAGE_ARGS[@]}")
}

build_merge_cmd() {
  MERGE_CMD=(
    "$PYTHON_BIN"
    "$STAGE_MAIN"
    --config "$CONFIG_PATH"
    --shard-count "$SHARD_COUNT"
    --merge-shards
  )
  MERGE_CMD+=("${EXTRA_STAGE_ARGS[@]}")
}

run_shard_sync() {
  local shard_id="$1"
  local log_file
  log_file="$(log_path_for_shard "$shard_id")"
  build_shard_cmd "$shard_id"
  echo "[stage05-shards] shard ${shard_id}/${SHARD_COUNT} starting; log=${log_file}"
  if {
    printf '[%s] COMMAND: ' "$(date '+%F %T')"
    print_cmd "${SHARD_CMD[@]}"
    "${SHARD_CMD[@]}"
  } >"$log_file" 2>&1; then
    echo "[stage05-shards] shard ${shard_id}/${SHARD_COUNT} completed"
    return 0
  fi
  echo "[stage05-shards] shard ${shard_id}/${SHARD_COUNT} failed; log=${log_file}" >&2
  print_log_tail "$log_file"
  return 1
}

launch_shard_async() {
  local shard_id="$1"
  local log_file
  local pid
  log_file="$(log_path_for_shard "$shard_id")"
  build_shard_cmd "$shard_id"
  echo "[stage05-shards] shard ${shard_id}/${SHARD_COUNT} starting in background; log=${log_file}"
  (
    printf '[%s] COMMAND: ' "$(date '+%F %T')"
    print_cmd "${SHARD_CMD[@]}"
    "${SHARD_CMD[@]}"
  ) >"$log_file" 2>&1 &
  pid=$!
  ACTIVE_PIDS+=("$pid")
  PID_TO_SHARD["$pid"]="$shard_id"
}

remove_active_pid() {
  local finished_pid="$1"
  local -a remaining=()
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if [[ "$pid" != "$finished_pid" ]]; then
      remaining+=("$pid")
    fi
  done
  ACTIVE_PIDS=("${remaining[@]}")
}

cleanup_background_jobs() {
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

wait_for_one_background_job() {
  local finished_pid=""
  local status=0
  local shard_id
  local log_file
  if wait -n -p finished_pid; then
    status=0
  else
    status=$?
  fi
  shard_id="${PID_TO_SHARD[$finished_pid]}"
  unset 'PID_TO_SHARD[$finished_pid]'
  remove_active_pid "$finished_pid"
  log_file="$(log_path_for_shard "$shard_id")"
  if (( status == 0 )); then
    echo "[stage05-shards] shard ${shard_id}/${SHARD_COUNT} completed"
    return 0
  fi
  echo "[stage05-shards] shard ${shard_id}/${SHARD_COUNT} failed; log=${log_file}" >&2
  print_log_tail "$log_file"
  cleanup_background_jobs
  return "$status"
}

run_all_shards() {
  local shard_id
  if (( MAX_PARALLEL <= 1 )); then
    for (( shard_id = 0; shard_id < SHARD_COUNT; shard_id += 1 )); do
      run_shard_sync "$shard_id"
    done
    return 0
  fi

  trap cleanup_background_jobs INT TERM
  for (( shard_id = 0; shard_id < SHARD_COUNT; shard_id += 1 )); do
    while (( ${#ACTIVE_PIDS[@]} >= MAX_PARALLEL )); do
      wait_for_one_background_job
    done
    launch_shard_async "$shard_id"
  done
  while (( ${#ACTIVE_PIDS[@]} > 0 )); do
    wait_for_one_background_job
  done
}

run_merge() {
  local log_file="${LOG_DIR}/merge.log"
  build_merge_cmd
  echo "[stage05-shards] merge starting; log=${log_file}"
  if {
    printf '[%s] COMMAND: ' "$(date '+%F %T')"
    print_cmd "${MERGE_CMD[@]}"
    "${MERGE_CMD[@]}"
  } >"$log_file" 2>&1; then
    echo "[stage05-shards] merge completed"
    return 0
  fi
  echo "[stage05-shards] merge failed; log=${log_file}" >&2
  print_log_tail "$log_file"
  return 1
}

while (($# > 0)); do
  case "$1" in
    --config)
      (($# >= 2)) || die "--config requires a path"
      CONFIG_PATH="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --shard-count)
      (($# >= 2)) || die "--shard-count requires an integer"
      SHARD_COUNT="$2"
      shift 2
      ;;
    --max-parallel)
      (($# >= 2)) || die "--max-parallel requires an integer"
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --log-dir)
      (($# >= 2)) || die "--log-dir requires a path"
      LOG_DIR="$2"
      shift 2
      ;;
    --no-merge)
      RUN_MERGE=0
      shift
      ;;
    --merge-only)
      MERGE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_STAGE_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_STAGE_ARGS+=("$1")
      shift
      ;;
  esac
done

is_integer "$SHARD_COUNT" || die "--shard-count must be a non-negative integer"
(( SHARD_COUNT >= 2 )) || die "--shard-count must be at least 2"
is_integer "$MAX_PARALLEL" || die "--max-parallel must be a non-negative integer"
(( MAX_PARALLEL >= 1 )) || die "--max-parallel must be at least 1"
if (( MAX_PARALLEL > SHARD_COUNT )); then
  MAX_PARALLEL="$SHARD_COUNT"
fi
if (( MERGE_ONLY && !RUN_MERGE )); then
  die "--merge-only cannot be combined with --no-merge"
fi

for arg in "${EXTRA_STAGE_ARGS[@]}"; do
  case "$arg" in
    --config|--shard-count|--shard-id|--merge-shards)
      die "wrapper-managed flag forwarded via stage_main args: ${arg}"
      ;;
  esac
done

mkdir -p "$LOG_DIR"

echo "[stage05-shards] config=${CONFIG_PATH}"
echo "[stage05-shards] shard_count=${SHARD_COUNT} max_parallel=${MAX_PARALLEL}"
echo "[stage05-shards] logs=${LOG_DIR}"

if (( MERGE_ONLY )); then
  run_merge
  exit 0
fi

run_all_shards

if (( RUN_MERGE )); then
  run_merge
fi
