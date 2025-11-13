"""Utilities for building institution crosswalk CSVs from the OpenAlex API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

import requests
from requests import Response, Session

try:  # optional progress bar
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - fallback when tqdm missing
    tqdm = None


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PACKAGE_ROOT / ".adaptive_fuzzy"
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OPENALEX_CACHE_PATH = DEFAULT_DATA_DIR / "openalex_institutions.jsonl"

OPENALEX_API_URL = "https://api.openalex.org/institutions"
DEFAULT_SELECT_FIELDS = ",".join(
    [
        "id",
        "display_name",
        "display_name_alternatives",
        "display_name_acronyms",
        "country_code",
        "geo",
        "type",
        "international",
    ]
)


def _default_crosswalk_dir() -> Path:
    try:
        from config import root as config_root  # type: ignore

        if config_root:
            return Path(config_root) / "data" / "crosswalks"
    except Exception:
        pass
    return PACKAGE_ROOT.parents[1] / "data" / "crosswalks"


def _clean_str(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def _unique(values: Iterable[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for value in values:
        cleaned = _clean_str(value)
        if cleaned is None:
            continue
        seen.setdefault(cleaned, None)
    return list(seen.keys())


@dataclass
class OpenAlexInstitution:
    """Minimal snapshot of an OpenAlex institution record."""

    openalex_id: str
    display_name: str
    country_code: Optional[str]
    institution_type: Optional[str]
    alternative_names: List[str]
    acronyms: List[str]
    city: Optional[str]
    geonames_city_id: Optional[str]

    @property
    def aliases(self) -> List[str]:
        values = [self.display_name, *self.alternative_names, *self.acronyms]
        return sorted({val for val in _unique(values) if val})

    @classmethod
    def from_api(cls, payload: Dict[str, object]) -> Optional["OpenAlexInstitution"]:
        raw_id = payload.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            return None
        short_id = raw_id.rsplit("/", 1)[-1]
        if short_id.startswith("I"):
            short_id = short_id[1:]

        name = _clean_str(payload.get("display_name", "")) or ""
        alternative_names: List[str] = []
        alt_field = payload.get("display_name_alternatives")
        if isinstance(alt_field, list):
            alternative_names.extend(_unique(str(item) for item in alt_field))
        international = payload.get("international")
        if isinstance(international, dict):
            intl_display = international.get("display_name")
            if isinstance(intl_display, dict):
                for value in intl_display.values():
                    cleaned = _clean_str(value)
                    if cleaned:
                        alternative_names.append(cleaned)
            else:
                intl_name = _clean_str(intl_display)
                if intl_name:
                    alternative_names.append(intl_name)
        alternative_names = _unique(alternative_names)

        acronyms: List[str] = []
        acronym_field = payload.get("display_name_acronyms")
        if isinstance(acronym_field, list):
            acronyms.extend(_unique(str(item) for item in acronym_field))
        acronyms = _unique(acronyms)

        ids = payload.get("ids")
        if isinstance(ids, dict):
            ror_value = _clean_str(ids.get("ror"))
        else:
            ror_value = None

        geo = payload.get("geo")
        if isinstance(geo, dict):
            city = _clean_str(geo.get("city"))
            geonames_city_id = _clean_str(geo.get("geonames_city_id"))
        else:
            city = None
            geonames_city_id = None

        return cls(
            openalex_id=short_id,
            display_name=name,
            country_code=_clean_str(payload.get("country_code")),
            institution_type=_clean_str(payload.get("type")),
            alternative_names=alternative_names,
            acronyms=acronyms,
            city=city,
            geonames_city_id=geonames_city_id,
        )


def _request_with_retry(
    session: Session,
    params: Dict[str, object],
    *,
    timeout: float,
    max_retries: int,
) -> Response:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = session.get(OPENALEX_API_URL, params=params, timeout=timeout)
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", "1"))
                time.sleep(max(wait, 1.0))
                continue
            if 500 <= response.status_code < 600:
                time.sleep(2**attempt)
                last_error = RuntimeError(
                    f"OpenAlex returned {response.status_code}: {response.text[:200]}"
                )
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2**attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to contact OpenAlex API after retries")


def iter_openalex_institutions(
    *,
    per_page: int = 200,
    max_records: Optional[int] = None,
    email: Optional[str] = None,
    cursor: str = "*",
    sleep_seconds: float = 0.2,
    timeout: float = 30.0,
    max_retries: int = 5,
    filters: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    initial_count: int = 0,
    page_offset: int = 0,
    record_callback: Optional[Callable[[str, int, int], None]] = None,
    page_callback: Optional[Callable[[Optional[str], int], None]] = None,
) -> Iterator[OpenAlexInstitution]:
    """Yield OpenAlex institutions by streaming through the paginated API."""

    session = requests.Session()
    consumed = max(0, initial_count)
    if max_records is not None and consumed >= max_records:
        return
    next_cursor = cursor
    pending_offset = max(0, page_offset)
    params: Dict[str, object] = {
        "per_page": max(1, min(200, per_page)),
        "select": DEFAULT_SELECT_FIELDS,
    }
    if filters:
        params.update(filters)
    if email:
        params["mailto"] = email

    while next_cursor:
        current_cursor = next_cursor
        params["cursor"] = current_cursor
        response = _request_with_retry(
            session,
            params,
            timeout=timeout,
            max_retries=max_retries,
        )
        payload = response.json()
        results = payload.get("results") or []
        index_in_page = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            index_in_page += 1
            if pending_offset and index_in_page <= pending_offset:
                continue
            record = OpenAlexInstitution.from_api(item)
            if record is None:
                continue
            yield record
            consumed += 1
            if record_callback is not None:
                record_callback(current_cursor, index_in_page, consumed)
            if progress_callback is not None:
                progress_callback(consumed)
            if max_records is not None and consumed >= max_records:
                return
        pending_offset = 0

        meta = payload.get("meta") or {}
        next_cursor = meta.get("next_cursor")
        if page_callback is not None:
            page_callback(next_cursor, consumed)
        if not next_cursor:
            break
        time.sleep(max(sleep_seconds, 0.0))


def _read_cache(path: Path) -> List[OpenAlexInstitution]:
    records: List[OpenAlexInstitution] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            data = json.loads(text)
            records.append(OpenAlexInstitution(**data))
    return records


def load_openalex_crosswalk(
    *,
    cache_path: Optional[Path] = None,
    refresh: bool = False,
    per_page: int = 200,
    max_records: Optional[int] = None,
    email: Optional[str] = None,
    filters: Optional[Dict[str, str]] = None,
    sleep_seconds: float = 0.2,
    timeout: float = 30.0,
    max_retries: int = 5,
) -> List[OpenAlexInstitution]:
    """Load (or build) the OpenAlex institution cache."""

    path = cache_path or Path(
        os.environ.get("OPENALEX_CROSSWALK_CACHE", DEFAULT_OPENALEX_CACHE_PATH)
    ).expanduser()

    if path.exists() and not refresh:
        return _read_cache(path)

    return _build_cache_with_checkpoint(
        path,
        per_page=per_page,
        max_records=max_records,
        email=email,
        filters=filters,
        sleep_seconds=sleep_seconds,
        timeout=timeout,
        max_retries=max_retries,
    )


def _checkpoint_paths(cache_path: Path) -> Tuple[Path, Path]:
    partial = cache_path.with_suffix(cache_path.suffix + ".partial")
    state = cache_path.with_suffix(cache_path.suffix + ".state.json")
    return partial, state


def _read_checkpoint(state_path: Path) -> Optional[Dict[str, object]]:
    if not state_path.exists():
        return None
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _write_checkpoint(
    state_path: Path,
    cursor: Optional[str],
    downloaded: int,
    offset: int,
) -> None:
    payload = {
        "cursor": cursor,
        "downloaded": int(downloaded),
        "offset": max(0, int(offset)),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload), encoding="utf-8")


def _build_cache_with_checkpoint(
    cache_path: Path,
    *,
    per_page: int,
    max_records: Optional[int],
    email: Optional[str],
    filters: Optional[Dict[str, str]],
    sleep_seconds: float,
    timeout: float,
    max_retries: int,
) -> List[OpenAlexInstitution]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path, state_path = _checkpoint_paths(cache_path)

    resume_state = _read_checkpoint(state_path) if partial_path.exists() else None
    if partial_path.exists() and resume_state is None:
        # Corrupt or missing state; start fresh
        partial_path.unlink()
        if state_path.exists():
            state_path.unlink()

    if cache_path.exists():
        cache_path.unlink()

    if not partial_path.exists():
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_file = partial_path.open("w", encoding="utf-8")
        resume_cursor: Optional[str] = "*"
        downloaded_existing = 0
        resume_offset = 0
    else:
        downloaded_existing = int(resume_state.get("downloaded", 0)) if resume_state else 0
        resume_cursor = resume_state.get("cursor") if resume_state else "*"
        resume_offset = int(resume_state.get("offset", 0)) if resume_state else 0
        if resume_state and resume_cursor is None:
            # Previous run finished downloading; finalize
            partial_file = partial_path.open("a", encoding="utf-8")
            partial_file.close()
            os.replace(partial_path, cache_path)
            try:
                state_path.unlink()
            except FileNotFoundError:
                pass
            return _read_cache(cache_path)
        partial_file = partial_path.open("a", encoding="utf-8")

    if max_records is not None and downloaded_existing >= max_records:
        partial_file.close()
        os.replace(partial_path, cache_path)
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass
        return _read_cache(cache_path)

    progress_bar = None
    printed = {"last": downloaded_existing}

    def _progress(count: int) -> None:
        if progress_bar is not None:
            progress_bar.update(max(0, count - _progress.last))
        else:
            if count - printed["last"] >= 500:
                print(f"Fetched {count:,} institutions...")
                printed["last"] = count
        _progress.last = count

    _progress.last = downloaded_existing

    if tqdm is not None:
        if max_records is not None:
            progress_bar = tqdm(total=max_records, unit="inst", desc="OpenAlex", leave=False)
            progress_bar.update(downloaded_existing)
        else:
            progress_bar = tqdm(unit="inst", desc="OpenAlex", leave=False)
            progress_bar.update(downloaded_existing)

    def _record_checkpoint(current_cursor: str, offset: int, consumed: int) -> None:
        _write_checkpoint(state_path, current_cursor, consumed, offset)

    def _page_callback(next_cursor: Optional[str], consumed: int) -> None:
        _write_checkpoint(state_path, next_cursor, consumed, 0)

    cursor_value = resume_cursor or "*"
    _write_checkpoint(state_path, cursor_value, downloaded_existing, resume_offset)
    try:
        for record in iter_openalex_institutions(
            per_page=per_page,
            max_records=max_records,
            email=email,
            cursor=cursor_value,
            sleep_seconds=sleep_seconds,
            timeout=timeout,
            max_retries=max_retries,
            filters=filters,
            progress_callback=_progress,
            initial_count=downloaded_existing,
            page_offset=resume_offset,
            record_callback=_record_checkpoint,
            page_callback=_page_callback,
        ):
            partial_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            partial_file.flush()
    finally:
        partial_file.close()
        if progress_bar is not None:
            progress_bar.close()

    if not partial_path.exists():
        raise RuntimeError("Failed to create OpenAlex cache file")

    os.replace(partial_path, cache_path)
    try:
        state_path.unlink()
    except FileNotFoundError:
        pass
    return _read_cache(cache_path)


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_institution_crosswalks(
    *,
    output_dir: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    refresh_cache: bool = False,
    per_page: int = 200,
    max_records: Optional[int] = None,
    email: Optional[str] = None,
    filters: Optional[Dict[str, str]] = None,
    sleep_seconds: float = 0.2,
    timeout: float = 30.0,
    max_retries: int = 5,
) -> Dict[str, Path]:
    """Generate the institution CSVs expected by the clustering pipeline."""

    print("Loading OpenAlex institution data...")
    records = load_openalex_crosswalk(
        cache_path=cache_path,
        refresh=refresh_cache,
        per_page=per_page,
        max_records=max_records,
        email=email,
        filters=filters,
        sleep_seconds=sleep_seconds,
        timeout=timeout,
        max_retries=max_retries,
    )
    print(f"Fetched {len(records):,} total institutions.")

    target_dir = output_dir or Path(
        os.environ.get("OPENALEX_CROSSWALK_DIR", _default_crosswalk_dir())
    ).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    base_rows: List[Dict[str, str]] = []
    alt_rows: List[Dict[str, str]] = []
    acronym_rows: List[Dict[str, str]] = []

    for record in records:
        alt_joined = "|".join(record.alternative_names)
        acr_joined = "|".join(record.acronyms)
        base_rows.append(
            {
                "id": record.openalex_id,
                "name": record.display_name,
                "country_code": record.country_code or "",
                "city": record.city or "",
                "geonames_city_id": record.geonames_city_id or "",
                "type": record.institution_type or "",
                "acronyms": acr_joined,
                "alternative_names": alt_joined,
            }
        )
        for alt_name in record.alternative_names:
            alt_rows.append(
                {
                    "id": record.openalex_id,
                    "name": record.display_name,
                    "country_code": record.country_code or "",
                    "city": record.city or "",
                    "geonames_city_id": record.geonames_city_id or "",
                    "alternative_names": alt_name,
                    "type": record.institution_type or "",
                }
            )
        for acronym in record.acronyms:
            acronym_rows.append(
                {
                    "id": record.openalex_id,
                    "name": record.display_name,
                    "country_code": record.country_code or "",
                    "city": record.city or "",
                    "geonames_city_id": record.geonames_city_id or "",
                    "acronyms": acronym,
                    "type": record.institution_type or "",
                }
            )

    files = {
        "institutions": target_dir / "institutions.csv",
        "acronyms": target_dir / "institutions_acronyms.csv",
        "altnames": target_dir / "institutions_altnames.csv",
    }

    print(f"Writing CSVs to {target_dir} ...")

    _write_csv(
        files["institutions"],
        base_rows,
        [
            "id",
            "name",
            "country_code",
            "city",
            "geonames_city_id",
            "type",
            "acronyms",
            "alternative_names",
        ],
    )
    if alt_rows:
        _write_csv(
            files["altnames"],
            alt_rows,
            ["id", "name", "country_code", "city", "geonames_city_id", "alternative_names", "type"],
        )
    else:
        files["altnames"].write_text(
            "id,name,country_code,city,geonames_city_id,alternative_names,type\n",
            encoding="utf-8",
        )
    if acronym_rows:
        _write_csv(
            files["acronyms"],
            acronym_rows,
            ["id", "name", "country_code", "city", "geonames_city_id", "acronyms", "type"],
        )
    else:
        files["acronyms"].write_text(
            "id,name,country_code,city,geonames_city_id,acronyms,type\n",
            encoding="utf-8",
        )

    return files


def _parse_filter_args(pairs: Optional[List[str]]) -> Dict[str, str]:
    if not pairs:
        return {}
    parsed: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Filter '{item}' must use key=value syntax")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the institution crosswalk CSVs from OpenAlex data.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination directory (defaults to config root)")
    parser.add_argument("--cache-path", type=Path, default=None, help="Optional path for the JSONL cache.")
    parser.add_argument("--refresh-cache", action="store_true", help="Force a fresh OpenAlex download even if cache exists.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional limit on total OpenAlex institutions to fetch.")
    parser.add_argument("--per-page", type=int, default=200, help="OpenAlex page size (max 200).")
    parser.add_argument("--email", type=str, default=None, help="Contact email passed via OpenAlex mailto param.")
    parser.add_argument("--filters", nargs="*", type=str, help="Additional OpenAlex filters as key=value pairs.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Delay between OpenAlex pages.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout per request (seconds).")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum HTTP retries per request.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    filters = _parse_filter_args(args.filters)
    paths = export_institution_crosswalks(
        output_dir=args.output_dir,
        cache_path=args.cache_path,
        refresh_cache=args.refresh_cache,
        per_page=max(1, min(200, args.per_page)),
        max_records=args.max_records,
        email=args.email,
        filters=filters,
        sleep_seconds=max(0.0, args.sleep_seconds),
        timeout=max(1.0, args.timeout),
        max_retries=max(1, args.max_retries),
    )
    print("Generated institution crosswalk files:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "OpenAlexInstitution",
    "iter_openalex_institutions",
    "load_openalex_crosswalk",
    "export_institution_crosswalks",
    "DEFAULT_OPENALEX_CACHE_PATH",
]
