from __future__ import annotations

import argparse
import getpass
import re
import sys
from builtins import print as _print
from datetime import datetime
from functools import partial
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROGRESS_FILE = PIPELINE_ROOT / "progress.md"

print = partial(_print, flush=True)

_VALID_STATUSES = {"☐", "~", "✓", "↩"}


def _parse_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r"-+", cell) for cell in cells if cell)


def _reformat_table(data_rows: list[list[str]]) -> list[str]:
    if not data_rows:
        return []
    ncols = max(len(row) for row in data_rows)
    padded = [row + [""] * (ncols - len(row)) for row in data_rows]
    widths = [max(len(padded[i][j]) for i in range(len(padded))) for j in range(ncols)]
    widths = [max(width, 3) for width in widths]

    lines: list[str] = []
    for idx, row in enumerate(padded):
        cells = " | ".join(cell.ljust(widths[j]) for j, cell in enumerate(row))
        lines.append(f"| {cells} |")
        if idx == 0:
            sep = "-|-".join("-" * widths[j] for j in range(ncols))
            lines.append(f"| {sep} |")
    return lines


def _format_duration(duration_seconds: float) -> str:
    duration_seconds = max(0.0, float(duration_seconds))
    total_seconds = int(round(duration_seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _update_file(
    *,
    progress_file: Path,
    stage_name: str,
    status: str,
    timestamp: str = "",
    username: str = "",
    duration: str = "",
) -> bool:
    if status not in _VALID_STATUSES:
        raise ValueError(f"Invalid progress status: {status}")

    text = progress_file.read_text()
    lines = text.splitlines(keepends=True)
    target = f"`{stage_name}`"
    result: list[str] = []
    updated = False
    i = 0

    while i < len(lines):
        row = _parse_row(lines[i].rstrip("\n"))
        if row is None:
            result.append(lines[i])
            i += 1
            continue

        raw_block: list[str] = []
        while i < len(lines):
            parsed = _parse_row(lines[i].rstrip("\n"))
            if parsed is None:
                break
            raw_block.append(lines[i].rstrip("\n"))
            i += 1

        data_rows: list[list[str]] = []
        for raw_line in raw_block:
            cells = _parse_row(raw_line)
            if cells is not None and not _is_separator(cells):
                data_rows.append(list(cells))

        if data_rows:
            header = data_rows[0]
            for row_cells in data_rows[1:]:
                if row_cells and row_cells[0] == target:
                    header_index = {name: idx for idx, name in enumerate(header)}
                    if "Status" in header_index:
                        row_cells[header_index["Status"]] = status
                    if "Completed at" in header_index:
                        row_cells[header_index["Completed at"]] = timestamp
                    if "By" in header_index:
                        row_cells[header_index["By"]] = username
                    if "Duration" in header_index:
                        row_cells[header_index["Duration"]] = duration
                    updated = True

        for formatted_line in _reformat_table(data_rows):
            result.append(formatted_line + "\n")

    if updated:
        progress_file.write_text("".join(result))
    return updated


def mark_stage_complete(stage_name: str, duration_seconds: float) -> bool:
    try:
        if not PROGRESS_FILE.exists():
            return False
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        try:
            username = getpass.getuser()
        except Exception:
            username = "unknown"
        return _update_file(
            progress_file=PROGRESS_FILE,
            stage_name=stage_name,
            status="✓",
            timestamp=timestamp,
            username=username,
            duration=_format_duration(duration_seconds),
        )
    except Exception:
        return False


def mark_stage_reset(stage_name: str, status: str = "↩") -> bool:
    try:
        if not PROGRESS_FILE.exists():
            return False
        return _update_file(
            progress_file=PROGRESS_FILE,
            stage_name=stage_name,
            status=status,
            timestamp="",
            username="",
            duration="",
        )
    except Exception:
        return False


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Update f1_indiv_merge progress.md")
    sub = parser.add_subparsers(dest="cmd", required=True)

    complete = sub.add_parser("complete", help="Mark a stage complete")
    complete.add_argument("--stage", required=True)
    complete.add_argument("--duration-seconds", required=True, type=float)

    reset = sub.add_parser("reset", help="Reset a stage row")
    reset.add_argument("--stage", required=True)
    reset.add_argument("--status", default="↩")

    args = parser.parse_args()
    if args.cmd == "complete":
        ok = mark_stage_complete(args.stage, args.duration_seconds)
        if not ok:
            print(f"[progress_tracker] No matching row found for {args.stage}", file=sys.stderr)
            raise SystemExit(1)
    elif args.cmd == "reset":
        ok = mark_stage_reset(args.stage, status=args.status)
        if not ok:
            print(f"[progress_tracker] No matching row found for {args.stage}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    _cli()
