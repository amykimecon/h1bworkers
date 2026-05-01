from __future__ import annotations

import argparse
import json
import os
import re
import smtplib
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Optional


SELECTED_POS_LARGE_TOTAL_RE = re.compile(
    r"\[wrds_workforce_extract\] Proactive large-firm slicing for selected-us-positions: .* "
    r"scheduled as (?P<n>\d[\d,]*) singleton task\(s\)"
)
SELECTED_POS_BATCH_DONE_RE = re.compile(
    r"\[wrds_workforce_extract\] DONE  workforce-selected-us-positions batch "
    r"(?P<done>\d[\d,]*)/(?P<total>\d[\d,]*):"
)
SELECTED_POS_LARGE_DONE_RE = re.compile(
    r"\[wrds_workforce_extract\] DONE  workforce-selected-us-positions large-firm "
)
SELECTED_POS_SCAN_DONE_RE = re.compile(
    r"\[wrds_workforce_extract\] DONE selected-us-positions scan:"
)
USER_HISTORY_START_RE = re.compile(
    r"\[wrds_workforce_extract\] START user-history extract: "
    r"(?P<user_ids>\d[\d,]*) user ids \| (?P<chunks>\d[\d,]*) chunks \| workers=(?P<workers>\d[\d,]*)"
)
USER_HISTORY_IMPORT_RE = re.compile(
    r"\[wrds_workforce_extract\] worker (?P<worker>\d[\d,]*)/(?P<workers>\d[\d,]*) import chunk "
    r"(?P<done>\d[\d,]*)/(?P<total>\d[\d,]*):"
)
USER_HISTORY_DONE_RE = re.compile(
    r"\[wrds_workforce_extract\] DONE user-history extract: (?P<chunks>\d[\d,]*) chunks"
)
LOCAL_PROFILE_BUILD_RE = re.compile(r"\[local_user_profile_cache\] BUILD:")
LOCAL_PROFILE_DONE_RE = re.compile(r"\[local_user_profile_cache\] DONE:")
WORKFORCE_BUILD_RE = re.compile(r"\[wrds_workforce_cache\] BUILD:")
WORKFORCE_DONE_RE = re.compile(r"\[wrds_workforce_cache\] DONE:")
SCHOOL_FLOWS_BUILD_RE = re.compile(r"\[wrds_school_flows_cache\] BUILD:")
SCHOOL_FLOWS_DONE_RE = re.compile(r"\[wrds_school_flows_cache\] DONE:")
FEATURES_DONE_RE = re.compile(r"\[company_features\] Wrote ")


@dataclass
class ProgressStatus:
    stage_key: str
    stage_label: str
    last_line: str
    detail_lines: list[str]
    progress_done: Optional[int] = None
    progress_total: Optional[int] = None
    progress_percent: Optional[float] = None
    stage_eta_seconds: Optional[float] = None
    status_timestamp_utc: Optional[str] = None


def _parse_int(text: str | None) -> Optional[int]:
    if text is None:
        return None
    cleaned = str(text).replace(",", "").strip()
    if not cleaned:
        return None
    return int(cleaned)


def _human_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _format_percent(done: Optional[int], total: Optional[int]) -> Optional[float]:
    if done is None or total is None or total <= 0:
        return None
    return float(done) / float(total) * 100.0


def parse_rebuild_log(log_text: str) -> ProgressStatus:
    selected_regular_done: Optional[int] = None
    selected_regular_total: Optional[int] = None
    selected_large_total = 0
    selected_large_done = 0
    selected_scan_done = False

    user_history_started = False
    user_history_done = False
    user_history_chunks_total: Optional[int] = None
    user_history_chunks_done: Optional[int] = None
    user_history_workers: Optional[int] = None
    user_history_user_ids: Optional[int] = None

    local_profile_build = False
    local_profile_done = False
    workforce_build = False
    workforce_done = False
    school_flows_build = False
    school_flows_done = False
    features_done = False
    last_line = ""

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        last_line = line

        match = SELECTED_POS_LARGE_TOTAL_RE.search(line)
        if match:
            selected_large_total = _parse_int(match.group("n")) or 0
            continue

        match = SELECTED_POS_BATCH_DONE_RE.search(line)
        if match:
            selected_regular_done = _parse_int(match.group("done"))
            selected_regular_total = _parse_int(match.group("total"))
            continue

        if SELECTED_POS_LARGE_DONE_RE.search(line):
            selected_large_done += 1
            continue

        if SELECTED_POS_SCAN_DONE_RE.search(line):
            selected_scan_done = True
            continue

        match = USER_HISTORY_START_RE.search(line)
        if match:
            user_history_started = True
            user_history_user_ids = _parse_int(match.group("user_ids"))
            user_history_chunks_total = _parse_int(match.group("chunks"))
            user_history_workers = _parse_int(match.group("workers"))
            continue

        match = USER_HISTORY_IMPORT_RE.search(line)
        if match:
            user_history_started = True
            user_history_chunks_done = _parse_int(match.group("done"))
            user_history_chunks_total = _parse_int(match.group("total"))
            user_history_workers = _parse_int(match.group("workers"))
            continue

        if USER_HISTORY_DONE_RE.search(line):
            user_history_done = True
            continue

        if LOCAL_PROFILE_BUILD_RE.search(line):
            local_profile_build = True
            continue
        if LOCAL_PROFILE_DONE_RE.search(line):
            local_profile_done = True
            continue
        if WORKFORCE_BUILD_RE.search(line):
            workforce_build = True
            continue
        if WORKFORCE_DONE_RE.search(line):
            workforce_done = True
            continue
        if SCHOOL_FLOWS_BUILD_RE.search(line):
            school_flows_build = True
            continue
        if SCHOOL_FLOWS_DONE_RE.search(line):
            school_flows_done = True
            continue
        if FEATURES_DONE_RE.search(line):
            features_done = True
            continue

    if features_done:
        return ProgressStatus(
            stage_key="complete",
            stage_label="Complete",
            last_line=last_line,
            detail_lines=["The company-features output has been written."],
            progress_done=1,
            progress_total=1,
            progress_percent=100.0,
        )

    if school_flows_build and not school_flows_done:
        return ProgressStatus(
            stage_key="school_flows",
            stage_label="Local School-Flows Build",
            last_line=last_line,
            detail_lines=["The run is in the school-flows cache build stage."],
        )

    if local_profile_done and workforce_build and not workforce_done:
        return ProgressStatus(
            stage_key="workforce_panel",
            stage_label="Local Workforce Panel Build",
            last_line=last_line,
            detail_lines=["The user-profile cache is done and the run is aggregating the local workforce panel."],
        )

    if local_profile_build and not local_profile_done:
        return ProgressStatus(
            stage_key="local_user_profile_cache",
            stage_label="Local User-Profile Cache",
            last_line=last_line,
            detail_lines=["The run is building the local user-profile/origin cache."],
        )

    if user_history_started and not user_history_done:
        detail_lines = []
        if user_history_user_ids is not None:
            detail_lines.append(f"User ids: {user_history_user_ids:,}")
        if user_history_workers is not None:
            detail_lines.append(f"Workers: {user_history_workers:,}")
        return ProgressStatus(
            stage_key="user_history_extract",
            stage_label="WRDS User/History Extract",
            last_line=last_line,
            detail_lines=detail_lines,
            progress_done=user_history_chunks_done,
            progress_total=user_history_chunks_total,
            progress_percent=_format_percent(user_history_chunks_done, user_history_chunks_total),
        )

    if workforce_build and not selected_scan_done:
        total = None
        done = None
        detail_lines = []
        if selected_regular_total is not None:
            total = selected_regular_total + selected_large_total
        if selected_regular_done is not None:
            done = selected_regular_done + selected_large_done
        if selected_regular_total is not None:
            detail_lines.append(
                f"Regular batches: {selected_regular_done or 0:,}/{selected_regular_total:,}"
            )
        if selected_large_total:
            detail_lines.append(
                f"Large-firm tasks: {selected_large_done:,}/{selected_large_total:,}"
            )
        return ProgressStatus(
            stage_key="selected_positions_extract",
            stage_label="WRDS Selected-US-Positions Extract",
            last_line=last_line,
            detail_lines=detail_lines,
            progress_done=done,
            progress_total=total,
            progress_percent=_format_percent(done, total),
        )

    if workforce_build and selected_scan_done and not user_history_started:
        return ProgressStatus(
            stage_key="selected_positions_merge",
            stage_label="Merging Selected-US-Positions Extract",
            last_line=last_line,
            detail_lines=["The selected-US-positions scan is done and the run is transitioning to user/history extraction."],
        )

    if workforce_build:
        return ProgressStatus(
            stage_key="workforce_cache",
            stage_label="WRDS Workforce Cache Build",
            last_line=last_line,
            detail_lines=["The run is inside the workforce-cache build."],
        )

    return ProgressStatus(
        stage_key="startup",
        stage_label="Startup",
        last_line=last_line,
        detail_lines=["The tracker has not yet observed a quantitative rebuild stage."],
    )


def _state_key(status: ProgressStatus) -> tuple[str, Optional[int], Optional[int]]:
    return status.stage_key, status.progress_done, status.progress_total


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"observations": [], "last_snapshot": None}
    return json.loads(path.read_text())


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def update_state(state: dict, status: ProgressStatus, now_utc: datetime) -> dict:
    observations = list(state.get("observations", []))
    snapshot = {
        "ts": now_utc.isoformat(),
        "stage_key": status.stage_key,
        "progress_done": status.progress_done,
        "progress_total": status.progress_total,
    }
    if not observations or any(observations[-1].get(k) != v for k, v in snapshot.items()):
        observations.append(snapshot)
    state["observations"] = observations[-200:]
    state["last_snapshot"] = snapshot
    return state


def estimate_stage_eta_seconds(state: dict, status: ProgressStatus) -> Optional[float]:
    if status.progress_done is None or status.progress_total is None:
        return None
    if status.progress_done >= status.progress_total:
        return 0.0
    observations = [
        obs
        for obs in state.get("observations", [])
        if obs.get("stage_key") == status.stage_key
        and obs.get("progress_total") == status.progress_total
        and obs.get("progress_done") is not None
    ]
    if len(observations) < 2:
        return None
    first = observations[0]
    last = observations[-1]
    try:
        t0 = datetime.fromisoformat(first["ts"])
        t1 = datetime.fromisoformat(last["ts"])
    except Exception:
        return None
    delta_seconds = (t1 - t0).total_seconds()
    delta_done = int(last["progress_done"]) - int(first["progress_done"])
    if delta_seconds <= 0 or delta_done <= 0:
        return None
    rate = float(delta_done) / float(delta_seconds)
    remaining = int(status.progress_total) - int(status.progress_done)
    if remaining <= 0:
        return 0.0
    return float(remaining) / rate


def render_status_text(
    *,
    log_path: Path,
    status: ProgressStatus,
) -> str:
    lines = [
        f"Rebuild progress update for {log_path}",
        f"Timestamp (UTC): {status.status_timestamp_utc}",
        f"Stage: {status.stage_label}",
    ]
    if status.progress_done is not None and status.progress_total is not None:
        progress = f"{status.progress_done:,}/{status.progress_total:,}"
        if status.progress_percent is not None:
            progress += f" ({status.progress_percent:.2f}%)"
        lines.append(f"Progress: {progress}")
    if status.stage_eta_seconds is not None:
        lines.append(f"Stage ETA: {_human_duration(status.stage_eta_seconds)}")
    else:
        lines.append("Stage ETA: unknown")
    if status.detail_lines:
        lines.extend(status.detail_lines)
    if status.last_line:
        lines.append(f"Last log line: {status.last_line}")
    return "\n".join(lines)


def maybe_send_email(
    *,
    subject_prefix: str,
    status: ProgressStatus,
    body: str,
) -> bool:
    email_to = os.environ.get("PROGRESS_EMAIL_TO", "").strip()
    smtp_host = os.environ.get("PROGRESS_SMTP_HOST", "").strip()
    smtp_from = os.environ.get("PROGRESS_SMTP_FROM", "").strip()
    if not email_to or not smtp_host or not smtp_from:
        return False

    smtp_port = int(os.environ.get("PROGRESS_SMTP_PORT", "587"))
    smtp_username = os.environ.get("PROGRESS_SMTP_USERNAME", "").strip()
    smtp_password = os.environ.get("PROGRESS_SMTP_PASSWORD", "")
    use_starttls = os.environ.get("PROGRESS_SMTP_STARTTLS", "1").strip().lower() not in {"0", "false", "no"}

    msg = EmailMessage()
    progress_suffix = ""
    if status.progress_done is not None and status.progress_total is not None:
        progress_suffix = f" {status.progress_done:,}/{status.progress_total:,}"
    msg["Subject"] = f"{subject_prefix} {status.stage_label}{progress_suffix}".strip()
    msg["From"] = smtp_from
    msg["To"] = email_to
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
        if use_starttls:
            smtp.starttls()
        if smtp_username:
            smtp.login(smtp_username, smtp_password)
        smtp.send_message(msg)
    return True


def track_once(
    *,
    log_path: Path,
    state_path: Path,
    status_path: Optional[Path],
    subject_prefix: str,
) -> ProgressStatus:
    now_utc = datetime.now(timezone.utc)
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    status = parse_rebuild_log(log_text)
    status.status_timestamp_utc = now_utc.isoformat()

    state = load_state(state_path)
    state = update_state(state, status, now_utc)
    status.stage_eta_seconds = estimate_stage_eta_seconds(state, status)
    save_state(state_path, state)

    body = render_status_text(log_path=log_path, status=status)
    if status_path is not None:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(body + "\n")
        status_path.with_suffix(status_path.suffix + ".json").write_text(
            json.dumps(asdict(status), indent=2, sort_keys=True)
        )

    maybe_send_email(
        subject_prefix=subject_prefix,
        status=status,
        body=body,
    )
    return status


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track rebuild progress from a log file.")
    parser.add_argument("--log-path", required=True, help="Path to the rebuild log.")
    parser.add_argument("--state-path", required=True, help="Path to the tracker state JSON.")
    parser.add_argument("--status-path", default=None, help="Optional plain-text status snapshot path.")
    parser.add_argument("--interval-hours", type=float, default=3.0, help="Polling interval for daemon mode.")
    parser.add_argument("--subject-prefix", default="[company_shift_share rebuild]", help="Email subject prefix.")
    parser.add_argument("--once", action="store_true", help="Run one snapshot and exit.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    log_path = Path(args.log_path)
    state_path = Path(args.state_path)
    status_path = Path(args.status_path) if args.status_path else None

    if args.once:
        track_once(
            log_path=log_path,
            state_path=state_path,
            status_path=status_path,
            subject_prefix=args.subject_prefix,
        )
        return 0

    interval_seconds = max(60.0, float(args.interval_hours) * 3600.0)
    while True:
        track_once(
            log_path=log_path,
            state_path=state_path,
            status_path=status_path,
            subject_prefix=args.subject_prefix,
        )
        time.sleep(interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
