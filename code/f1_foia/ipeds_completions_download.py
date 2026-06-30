"""Download and assemble the IPEDS Completions panel.

This mirrors ``ipeds_download.py`` for IPEDS cost/enrollment data, but the
completions workflow needs extra harmonization:

* Completions A files exist for 2000-2024.
* Completions distance-education program files exist for 2013-2024.
* Pre-2020 CIP codes are mapped to 2020 CIP codes before the panel is stacked.

The default output is the raw Stata file expected elsewhere in this repo:
``{root}/data/raw/ipeds_completions_all.dta``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
import requests


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


IPEDS_BASE_URL = "https://nces.ed.gov/ipeds/datacenter/data"
CIP_BASE_URL = "https://nces.ed.gov/ipeds/cipcode/Files"

DEFAULT_START_YEAR = 2000
DEFAULT_DEP_START_YEAR = 2013
DEFAULT_END_YEAR = 2024

COMPLETIONS_A_COLUMNS = [
    "unitid",
    "cipcode",
    "majornum",
    "awlevel",
]
COMPLETIONS_A_PREFIXES = (
    "ctotal",
    "caian",
    "casia",
    "cbkaa",
    "chisp",
    "cnhpi",
    "cwhit",
    "c2mor",
    "cunkn",
    "cnral",
)
COMPLETIONS_DEP_COLUMNS = ["unitid", "cipcode"]
COMPLETIONS_DEP_PREFIXES = ("ptotal", "pbachl", "pmastr", "pdocrs")

OLD_RACE_RENAMES = {
    "crace01": "cnralm",
    "crace02": "cnralw",
    "crace03": "cbkaam",
    "crace04": "cbkaaw",
    "crace05": "caianm",
    "crace06": "caianw",
    "crace07": "casiam",
    "crace08": "casiaw",
    "crace09": "chispm",
    "crace10": "chispw",
    "crace11": "cwhitm",
    "crace12": "cwhitw",
    "crace13": "cunknm",
    "crace14": "cunknw",
    "crace15": "ctotalm",
    "crace16": "ctotalw",
}
OLD_RACE_TOTAL_RENAMES = {
    "crace17": "cnralt",
    "crace18": "cbkaat",
    "crace19": "caiant",
    "crace20": "casiat",
    "crace21": "chispt",
    "crace22": "cwhitt",
    "crace23": "cunknt",
    "crace24": "ctotalt",
}

AWLEVEL_LABELS = {
    1: "Certificates of less than 1 year",
    2: "Certificates of at least 1 but less than 2 years",
    3: "Associate's degree",
    4: "Certificates of at least 2 but less than 4 years",
    5: "Bachelor's degree",
    6: "Postbaccalaureate certificate",
    7: "Master's degree",
    8: "Post-master's certificate",
    9: "Doctors degrees (old)",
    10: "First-professional degrees (old)",
    11: "First-professional certificates (old)",
    12: "Degrees total",
    13: "Certificates below the baccalaureate total",
    14: "Certificates above the baccalaureate total",
    15: "Degrees/certificates total",
    17: "Doctor's degree - research/scholarship",
    18: "Doctor's degree - professional practice",
    19: "Doctor's degree - other",
    20: "Certificates of less than 12 weeks",
    21: "Certificates of at least 12 weeks but less than 1 year",
}

CIP2_LABELS = {
    1: "Agriculture",
    3: "Conservation",
    4: "Architecture",
    5: "Cultural/Gender Studies",
    9: "Communication",
    10: "Communications Tech",
    11: "Computer Science",
    12: "Culinary Services",
    13: "Education",
    14: "Engineering",
    15: "Engineering Tech",
    16: "Languages",
    19: "Consumer Sciences",
    22: "Law",
    23: "English",
    24: "General Studies",
    25: "Library Science",
    26: "Biology",
    27: "Math",
    29: "Military Tech",
    30: "Multi/Interdisciplinary Studies",
    31: "Parks/Rec",
    38: "Philosophy",
    39: "Theology",
    40: "Physical Sciences",
    41: "Science Tech",
    42: "Psychology",
    43: "Protective Services",
    44: "Public Administration",
    45: "Social Sciences",
    46: "Construction Trades",
    47: "Mechanic and Repair Tech",
    48: "Precision Production",
    49: "Transportation",
    50: "Arts",
    51: "Healthcare",
    52: "Business",
    54: "History",
    99: "Grand total",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_root_from_config() -> Path | None:
    """Return config.root if it can be imported without failing on local secrets."""
    code_dir = Path(__file__).resolve().parents[1]
    if str(code_dir) not in sys.path:
        sys.path.append(str(code_dir))
    try:
        import config as project_config  # type: ignore
    except Exception:
        return None
    root_value = getattr(project_config, "root", None)
    return Path(root_value).expanduser() if root_value else None


def default_project_root() -> Path:
    env_root = os.environ.get("H1BWORKERS_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return _project_root_from_config() or _repo_root()


def _download_file(url: str, local_path: Path, label: str, force: bool = False) -> bool:
    if local_path.exists() and not force:
        print(f"  [{label}] cached")
        return True

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [{label}] downloading ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        print(f"SKIP ({exc})")
        return False
    except requests.RequestException as exc:
        print(f"ERROR ({exc})")
        return False

    local_path.write_bytes(resp.content)
    print("done")
    return True


def _candidate_zip_names(kind: str, year: int, stata: bool = False) -> list[str]:
    if kind == "a":
        stem = f"C{year}_A"
    elif kind == "dep":
        stem = f"C{year}DEP"
    else:
        raise ValueError(f"Unknown completions kind: {kind}")
    return [f"{stem}_Data_Stata.zip"] if stata else [f"{stem}.zip", f"{stem}_Data_Stata.zip"]


def _download_first_available(
    filenames: list[str],
    raw_dir: Path,
    label: str,
    force: bool = False,
) -> Path | None:
    for filename in filenames:
        local_path = raw_dir / filename
        if _download_file(f"{IPEDS_BASE_URL}/{filename}", local_path, f"{label}/{filename}", force=force):
            return local_path
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"^ï»¿", "", regex=True)
    )
    auto_renamed = [c for c in df.columns if re.search(r"\.\d+$", c) and c.rsplit(".", 1)[0] in df.columns]
    if auto_renamed:
        df = df.drop(columns=auto_renamed)
    return df.loc[:, ~df.columns.duplicated()]


def _find_file_in_zip(zf: zipfile.ZipFile, stems: list[str], ext: str) -> str | None:
    names = zf.namelist()
    lower_to_name = {name.lower(): name for name in names}
    candidates: list[str] = []
    for stem in stems:
        stem = stem.lower()
        candidates.extend(
            [
                f"{stem}{ext}",
                f"{stem}_data_stata{ext}",
                f"{stem}_rv{ext}",
                f"{stem}_data_stata_rv{ext}",
            ]
        )
    for candidate in candidates:
        if candidate in lower_to_name:
            return lower_to_name[candidate]

    matches = [
        name
        for name in names
        if name.lower().endswith(ext)
        and "_dict" not in name.lower()
        and "_label" not in name.lower()
    ]
    return matches[0] if matches else None


def _find_file_in_dir(directory: Path, stems: list[str], ext: str) -> Path | None:
    files = [path for path in directory.rglob(f"*{ext}") if path.is_file()]
    lower_to_path = {path.name.lower(): path for path in files}
    candidates: list[str] = []
    for stem in stems:
        stem = stem.lower()
        candidates.extend(
            [
                f"{stem}{ext}",
                f"{stem}_data_stata{ext}",
                f"{stem}_rv{ext}",
                f"{stem}_data_stata_rv{ext}",
            ]
        )
    for candidate in candidates:
        if candidate in lower_to_path:
            return lower_to_path[candidate]

    matches = [
        path
        for path in files
        if "_dict" not in path.name.lower()
        and "_label" not in path.name.lower()
    ]
    return matches[0] if matches else None


def _read_csv_from_zip(zip_path: Path, stems: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = _find_file_in_zip(zf, stems, ".csv")
        if csv_name is None:
            raise FileNotFoundError(f"No CSV found in {zip_path}")
        with zf.open(csv_name) as handle:
            df = pd.read_csv(handle, encoding="latin-1", low_memory=False, index_col=False)
    return _normalize_columns(df)


def _read_csv_from_source(source_path: Path, stems: list[str]) -> pd.DataFrame:
    if source_path.is_dir():
        csv_path = _find_file_in_dir(source_path, stems, ".csv")
        if csv_path is None:
            raise FileNotFoundError(f"No CSV found in {source_path}")
        return _normalize_columns(pd.read_csv(csv_path, encoding="latin-1", low_memory=False, index_col=False))
    return _read_csv_from_zip(source_path, stems)


def _read_do_from_zip(zip_path: Path, stems: list[str]) -> str | None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            do_name = _find_file_in_zip(zf, stems, ".do")
            if do_name is None:
                return None
            return zf.read(do_name).decode("latin-1")
    except zipfile.BadZipFile:
        return None


def _read_do_from_source(source_path: Path, stems: list[str]) -> str | None:
    if source_path.is_dir():
        do_path = _find_file_in_dir(source_path, stems, ".do")
        if do_path is None:
            return None
        return do_path.read_text(encoding="latin-1")
    return _read_do_from_zip(source_path, stems)


def _candidate_source_dirs(kind: str, year: int) -> list[str]:
    if kind == "a":
        return [f"C{year}_A_Data_Stata", f"C{year}_A"]
    if kind == "dep":
        return [f"C{year}DEP_Data_Stata", f"C{year}DEP"]
    raise ValueError(f"Unknown completions kind: {kind}")


def _find_existing_source(raw_dir: Path, kind: str, year: int, stata: bool = False) -> Path | None:
    candidate_names = _candidate_source_dirs(kind, year)
    if not stata:
        candidate_names.extend(_candidate_zip_names(kind, year, stata=False))
    else:
        candidate_names.extend(_candidate_zip_names(kind, year, stata=True))
    for name in candidate_names:
        path = raw_dir / name
        if path.exists():
            return path
    return None


def parse_stata_labels(do_text: str) -> dict[str, dict]:
    """Parse variable labels, value label definitions, and assignments from Stata .do text."""
    text = re.sub(r"\s*///\s*\n\s*", " ", do_text)

    var_labels: dict[str, str] = {}
    for match in re.finditer(r'label\s+variable\s+(\w+)\s+"([^"]*)"', text, re.IGNORECASE):
        var_labels[match.group(1).lower()] = match.group(2)

    val_label_defs: dict[str, dict[int, str]] = {}
    for match in re.finditer(
        r'label\s+define\s+(\w+)\s+((?:(?:-?\d+)\s+"[^"]*"\s*)+)',
        text,
        re.IGNORECASE,
    ):
        label_name = match.group(1).lower()
        pairs = val_label_defs.get(label_name, {})
        for pair_match in re.finditer(r'(-?\d+)\s+"([^"]*)"', match.group(2)):
            pairs[int(pair_match.group(1))] = pair_match.group(2)
        if pairs:
            val_label_defs[label_name] = pairs

    val_label_assigns: dict[str, str] = {}
    for match in re.finditer(r"label\s+values\s+(\w+)\s+(\w+)", text, re.IGNORECASE):
        val_label_assigns[match.group(1).lower()] = match.group(2).lower()

    return {
        "var_labels": var_labels,
        "val_label_defs": val_label_defs,
        "val_label_assigns": val_label_assigns,
    }


def _value_label(labels: dict[str, dict] | None, column: str) -> dict[int, str]:
    if not labels:
        return {}
    label_name = labels.get("val_label_assigns", {}).get(column)
    if not label_name:
        return {}
    return dict(labels.get("val_label_defs", {}).get(label_name, {}))


def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        cleaned = df[col].replace(".", np.nan)
        converted = pd.to_numeric(cleaned, errors="coerce")
        if converted.notna().sum() >= cleaned.notna().sum():
            df[col] = converted
        else:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    return df


def _clean_cip_code_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace('="', "", regex=False)
        .str.replace('"', "", regex=False)
        .str.replace(".", "", regex=False)
    )
    cleaned = cleaned.where(cleaned.str.fullmatch(r"\d+").fillna(False))
    return pd.to_numeric(cleaned, errors="coerce")


def _read_cip_crosswalk(raw_dir: Path, from_year: int, to_year: int, force: bool = False) -> pd.DataFrame:
    filename = f"Crosswalk{from_year}to{to_year}.csv"
    local_path = raw_dir / filename
    if not local_path.exists() or force:
        ok = _download_file(
            f"{CIP_BASE_URL}/{filename}",
            local_path,
            f"CIP/{filename}",
            force=force,
        )
        if not ok:
            raise FileNotFoundError(
                f"Missing {filename}; download it from NCES or place it at {local_path}"
            )

    df = pd.read_csv(local_path, dtype=str, encoding="latin-1")
    df.columns = df.columns.str.lower().str.strip()
    from_col = f"cipcode{from_year}"
    to_col = f"cipcode{to_year}"
    to_title_col = f"ciptitle{to_year}"

    df[f"cip_cleaned{from_year}"] = _clean_cip_code_series(df[from_col])
    df[f"cip_cleaned{to_year}"] = _clean_cip_code_series(df[to_col])

    deleted = df.get("action", pd.Series("", index=df.index)).astype("string").str.lower().eq("deleted")
    extracted = (
        df.get(to_title_col, pd.Series("", index=df.index))
        .astype("string")
        .str.extract(r"([0-9]{1,2}\.[0-9]+)", expand=False)
        .str.replace(".", "", regex=False)
    )
    df.loc[deleted & df[f"cip_cleaned{to_year}"].isna(), f"cip_cleaned{to_year}"] = pd.to_numeric(
        extracted,
        errors="coerce",
    )
    return df.dropna(subset=[f"cip_cleaned{from_year}"])


def build_cip_maps(raw_dir: Path, force: bool = False) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    cw_2000_2010 = _read_cip_crosswalk(raw_dir, 2000, 2010, force=force)
    cw_2010_2020 = _read_cip_crosswalk(raw_dir, 2010, 2020, force=force)

    map_2010_2020_df = cw_2010_2020[["cip_cleaned2010", "cip_cleaned2020"]].dropna().drop_duplicates(
        "cip_cleaned2010",
        keep="first",
    )
    map_2010_2020 = {
        int(row.cip_cleaned2010): int(row.cip_cleaned2020)
        for row in map_2010_2020_df.itertuples(index=False)
    }

    merged = cw_2000_2010.merge(
        map_2010_2020_df,
        on="cip_cleaned2010",
        how="left",
        suffixes=("", "_from2020"),
    )
    merged["cip_2020_final"] = merged["cip_cleaned2020"]
    map_2000_2020_df = merged[["cip_cleaned2000", "cip_2020_final"]].dropna().drop_duplicates(
        "cip_cleaned2000",
        keep="first",
    )
    map_2000_2020 = {
        int(row.cip_cleaned2000): int(row.cip_2020_final)
        for row in map_2000_2020_df.itertuples(index=False)
    }

    title_map: dict[int, str] = {}
    code_col = "cip_cleaned2020"
    title_col = "ciptitle2020"
    for row in cw_2010_2020[[code_col, title_col]].dropna(subset=[code_col]).itertuples(index=False):
        code = int(getattr(row, code_col))
        title = str(getattr(row, title_col)).strip()
        title_map.setdefault(code, title)

    return map_2000_2020, map_2010_2020, title_map


def _apply_cip_map(df: pd.DataFrame, year: int, map_2000_2020: dict[int, int], map_2010_2020: dict[int, int]) -> pd.DataFrame:
    df = df.copy()
    if "cipcode" not in df.columns:
        return df
    cip = pd.to_numeric(df["cipcode"], errors="coerce")
    mapping = map_2000_2020 if year <= 2009 else map_2010_2020 if year <= 2019 else None
    if mapping:
        mapped = cip.map(mapping)
        cip = cip.astype("float64")
        cip.loc[mapped.notna()] = mapped.loc[mapped.notna()].astype("float64")
    df["cipcode"] = cip
    return df


def _select_columns(df: pd.DataFrame, base_cols: list[str], prefixes: tuple[str, ...]) -> pd.DataFrame:
    cols = [col for col in base_cols if col in df.columns]
    cols.extend([col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)])
    cols = list(dict.fromkeys(cols))
    return df[cols].copy()


def _completion_stems(kind: str, year: int) -> list[str]:
    if kind == "a":
        return [f"c{year}_a", f"c{year}_a_data_stata"]
    return [f"c{year}dep", f"c{year}dep_data_stata"]


def _download_and_read_component(
    raw_dir: Path,
    kind: str,
    year: int,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict] | None]:
    print(f"\n  -- {year} {kind.upper()} --")
    stems = _completion_stems(kind, year)
    data_source = None if force else _find_existing_source(raw_dir, kind, year, stata=False)
    if data_source is None:
        data_source = _download_first_available(
            _candidate_zip_names(kind, year, stata=False),
            raw_dir,
            f"{kind.upper()} data/{year}",
            force=force,
        )
    if data_source is None:
        raise FileNotFoundError(f"Could not download completions {kind.upper()} data for {year}")

    df = _read_csv_from_source(data_source, stems)
    print(f"  [data] {len(df):,} rows from {data_source.name}")

    labels = None
    stata_source = None if force else _find_existing_source(raw_dir, kind, year, stata=True)
    if stata_source is None:
        stata_source = _download_first_available(
            _candidate_zip_names(kind, year, stata=True),
            raw_dir,
            f"{kind.upper()} Stata/{year}",
            force=force,
        )
    if stata_source is not None:
        do_text = _read_do_from_source(stata_source, stems)
        if do_text:
            labels = parse_stata_labels(do_text)
            print(
                f"  [Stata] {len(labels['var_labels'])} var labels, "
                f"{len(labels['val_label_assigns'])} coded vars"
            )
    return df, labels


def process_completions_a_year(
    raw_df: pd.DataFrame,
    year: int,
    map_2000_2020: dict[int, int],
    map_2010_2020: dict[int, int],
) -> pd.DataFrame:
    df = _apply_cip_map(raw_df, year, map_2000_2020, map_2010_2020)
    if year <= 2007:
        rename_map = {old: new for old, new in OLD_RACE_RENAMES.items() if old in df.columns}
        if year >= 2002:
            rename_map.update({old: new for old, new in OLD_RACE_TOTAL_RENAMES.items() if old in df.columns})
        df = df.rename(columns=rename_map)
        if "c2mort" not in df.columns:
            df["c2mort"] = np.nan
        if "cnhpit" not in df.columns:
            df["cnhpit"] = np.nan
    if year <= 2000 and "majornum" not in df.columns:
        df["majornum"] = np.nan

    df = _select_columns(df, COMPLETIONS_A_COLUMNS, COMPLETIONS_A_PREFIXES)
    df["year"] = year
    return _normalize_dtypes(df)


def process_completions_dep_year(
    raw_df: pd.DataFrame,
    year: int,
    map_2010_2020: dict[int, int],
) -> pd.DataFrame:
    df = _apply_cip_map(raw_df, year, {}, map_2010_2020)
    df = _select_columns(df, COMPLETIONS_DEP_COLUMNS, COMPLETIONS_DEP_PREFIXES)
    df["year"] = year
    return _normalize_dtypes(df)


def _stable_group_id(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    keys = df[cols].astype("Int64").astype("string").fillna("<NA>").agg("|".join, axis=1)
    unique_keys = {key: idx + 1 for idx, key in enumerate(sorted(keys.unique()))}
    return keys.map(unique_keys).astype("int64")


def finalize_completions_panel(completions_a: pd.DataFrame, completions_dep: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["unitid", "cipcode", "year"]
    panel = completions_a.merge(completions_dep, on=merge_cols, how="left", validate="m:1")

    for col in ["cnralt", "ctotalt", "unitid", "cipcode", "awlevel", "majornum", "year"]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel["share_intl"] = np.where(
        pd.to_numeric(panel.get("ctotalt"), errors="coerce") > 0,
        pd.to_numeric(panel.get("cnralt"), errors="coerce") / pd.to_numeric(panel.get("ctotalt"), errors="coerce"),
        np.nan,
    )
    panel["cip2dig"] = np.floor(pd.to_numeric(panel["cipcode"], errors="coerce") / 10000)
    panel["cip4dig"] = np.floor(pd.to_numeric(panel["cipcode"], errors="coerce") / 100)
    panel.loc[panel["cip2dig"].eq(0), "cip2dig"] = 99
    panel["finance"] = panel["cip4dig"].eq(5208)
    panel["orfe"] = panel["cip4dig"].eq(1437)
    panel["allfin"] = panel["finance"] | panel["orfe"] | panel["cip2dig"].isin([45, 27])
    panel["master"] = panel["awlevel"].eq(7)
    panel["phd"] = panel["awlevel"].isin([9, 17])
    panel = panel[panel["year"].ne(2000)].copy()
    panel["progid"] = _stable_group_id(panel, ["unitid", "cipcode", "awlevel", "majornum"])

    return _normalize_dtypes(panel)


def _latest_label(label_by_year: dict[int, dict[str, dict] | None], column: str) -> dict[int, str]:
    for year in sorted(label_by_year, reverse=True):
        labels = label_by_year[year]
        value_label = _value_label(labels, column)
        if value_label:
            return value_label
    return {}


def add_label_columns(
    panel: pd.DataFrame,
    cip_title_map: dict[int, str],
    a_labels_by_year: dict[int, dict[str, dict] | None],
) -> pd.DataFrame:
    panel = panel.copy()
    cip_labels = _latest_label(a_labels_by_year, "cipcode")
    if not cip_labels and cip_title_map:
        cip_labels = {code: f"{code:06d}-{title}" for code, title in cip_title_map.items()}
    awlevel_labels = _latest_label(a_labels_by_year, "awlevel") or AWLEVEL_LABELS

    cip_numeric = pd.to_numeric(panel.get("cipcode"), errors="coerce")
    awlevel_numeric = pd.to_numeric(panel.get("awlevel"), errors="coerce")
    cip2_numeric = pd.to_numeric(panel.get("cip2dig"), errors="coerce")
    panel["cipcode_lab"] = cip_numeric.map(cip_labels)
    panel["awlevel_lab"] = awlevel_numeric.map(awlevel_labels)
    panel["cip2dig_lab"] = cip2_numeric.map(CIP2_LABELS)
    return panel


def _stata_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bool_cols = out.select_dtypes(include="bool").columns
    for col in bool_cols:
        out[col] = out[col].astype("int8")
    for col in out.columns:
        if pd.api.types.is_integer_dtype(out[col]) and out[col].isna().any():
            out[col] = out[col].astype("float64")
    return out


def write_outputs(
    panel: pd.DataFrame,
    output_dta: Path | None,
    historical_dta: Path | None,
    panel_parquet: Path | None,
    labels_json: Path | None,
    a_labels_by_year: dict[int, dict[str, dict] | None],
    cip_title_map: dict[int, str],
) -> None:
    labeled_panel = add_label_columns(panel, cip_title_map, a_labels_by_year)

    if panel_parquet is not None:
        panel_parquet.parent.mkdir(parents=True, exist_ok=True)
        labeled_panel.to_parquet(panel_parquet, index=False)
        print(f"\n  Saved parquet panel -> {panel_parquet}")

    value_labels = {
        "awlevel": _latest_label(a_labels_by_year, "awlevel") or AWLEVEL_LABELS,
        "cip2dig": CIP2_LABELS,
    }
    value_labels = {col: labels for col, labels in value_labels.items() if col in panel.columns and labels}

    written_dta: Path | None = None
    if output_dta is not None:
        output_dta.parent.mkdir(parents=True, exist_ok=True)
        _stata_ready(labeled_panel).to_stata(output_dta, write_index=False, version=118, value_labels=value_labels)
        print(f"  Saved raw Stata panel -> {output_dta}")
        written_dta = output_dta

    if historical_dta is not None:
        historical_dta.parent.mkdir(parents=True, exist_ok=True)
        if written_dta is not None and historical_dta.resolve() != written_dta.resolve():
            shutil.copyfile(written_dta, historical_dta)
        else:
            _stata_ready(labeled_panel).to_stata(historical_dta, write_index=False, version=118, value_labels=value_labels)
        print(f"  Saved historical Stata panel -> {historical_dta}")

    if labels_json is not None:
        labels_json.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "a_years": {str(year): labels for year, labels in a_labels_by_year.items() if labels is not None},
            "cip2_labels": {str(key): value for key, value in CIP2_LABELS.items()},
        }
        labels_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"  Saved label metadata -> {labels_json}")


def build_completions_panel(
    raw_dir: Path,
    start_year: int,
    dep_start_year: int,
    end_year: int,
    force: bool = False,
    write_intermediate_dta: bool = False,
) -> tuple[pd.DataFrame, dict[int, dict[str, dict] | None], dict[int, str]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    map_2000_2020, map_2010_2020, cip_title_map = build_cip_maps(raw_dir, force=force)

    a_frames: list[pd.DataFrame] = []
    dep_frames: list[pd.DataFrame] = []
    a_labels_by_year: dict[int, dict[str, dict] | None] = {}

    for year in range(start_year, end_year + 1):
        raw_df, labels = _download_and_read_component(raw_dir, "a", year, force=force)
        a_labels_by_year[year] = labels
        processed = process_completions_a_year(raw_df, year, map_2000_2020, map_2010_2020)
        a_frames.append(processed)
        if write_intermediate_dta:
            path = raw_dir / f"completions{year}a.dta"
            _stata_ready(processed).to_stata(path, write_index=False, version=118)
            print(f"  Saved intermediate -> {path}")

    for year in range(dep_start_year, end_year + 1):
        raw_df, _labels = _download_and_read_component(raw_dir, "dep", year, force=force)
        processed = process_completions_dep_year(raw_df, year, map_2010_2020)
        dep_frames.append(processed)
        if write_intermediate_dta:
            path = raw_dir / f"completions{year}dep.dta"
            _stata_ready(processed).to_stata(path, write_index=False, version=118)
            print(f"  Saved intermediate -> {path}")

    completions_a = pd.concat(a_frames, ignore_index=True, sort=False)
    completions_dep = (
        pd.concat(dep_frames, ignore_index=True, sort=False)
        if dep_frames
        else pd.DataFrame(columns=["unitid", "cipcode", "year"])
    )

    print(
        f"\n  Completions A panel: {len(completions_a):,} rows | "
        f"{completions_a.shape[1]} cols | years {start_year}-{end_year}"
    )
    if dep_frames:
        dep_years = f"{dep_start_year}-{end_year}"
    else:
        dep_years = "not requested"
    print(
        f"  Completions DEP panel: {len(completions_dep):,} rows | "
        f"{completions_dep.shape[1]} cols | years {dep_years}"
    )

    panel = finalize_completions_panel(completions_a, completions_dep)
    print(
        f"  Final completions panel: {len(panel):,} rows | "
        f"{panel.shape[1]} cols | years {int(panel['year'].min())}-{int(panel['year'].max())}"
    )
    return panel, a_labels_by_year, cip_title_map


def parse_args() -> argparse.Namespace:
    default_root = default_project_root()
    parser = argparse.ArgumentParser(description="Download and assemble IPEDS completions data.")
    parser.add_argument("--root", type=Path, default=default_root, help="Project data root.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory for raw IPEDS downloads.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--dep-start-year", type=int, default=DEFAULT_DEP_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--force", action="store_true", help="Re-download existing source files.")
    parser.add_argument(
        "--write-intermediate-dta",
        action="store_true",
        help="Also save per-year completions{year}a/dep.dta files.",
    )
    parser.add_argument(
        "--skip-dta",
        action="store_true",
        help="Do not write the raw Stata panel.",
    )
    parser.add_argument(
        "--skip-historical-dta",
        action="store_true",
        help="Do not write data/raw/ipeds/completions_all.dta.",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Do not write data/raw/ipeds/ipeds_completions_panel.parquet.",
    )
    parser.add_argument("--output-dta", type=Path, default=None)
    parser.add_argument("--historical-output-dta", type=Path, default=None)
    parser.add_argument("--panel-parquet", type=Path, default=None)
    parser.add_argument("--labels-json", type=Path, default=None)
    args = parser.parse_args()

    raw_dir = args.raw_dir or args.root / "data/raw/ipeds"
    args.raw_dir = raw_dir
    args.output_dta = args.output_dta or args.root / "data/raw/ipeds_completions_all.dta"
    args.historical_output_dta = args.historical_output_dta or raw_dir / "completions_all.dta"
    args.panel_parquet = args.panel_parquet or raw_dir / "ipeds_completions_panel.parquet"
    args.labels_json = args.labels_json or raw_dir / "ipeds_completions_labels.json"
    return args


def main() -> None:
    args = parse_args()
    if args.dep_start_year < args.start_year:
        raise ValueError("--dep-start-year must be greater than or equal to --start-year")
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")

    panel, a_labels_by_year, cip_title_map = build_completions_panel(
        raw_dir=args.raw_dir,
        start_year=args.start_year,
        dep_start_year=args.dep_start_year,
        end_year=args.end_year,
        force=args.force,
        write_intermediate_dta=args.write_intermediate_dta,
    )
    write_outputs(
        panel=panel,
        output_dta=None if args.skip_dta else args.output_dta,
        historical_dta=None if args.skip_historical_dta else args.historical_output_dta,
        panel_parquet=None if args.skip_parquet else args.panel_parquet,
        labels_json=args.labels_json,
        a_labels_by_year=a_labels_by_year,
        cip_title_map=cip_title_map,
    )


if __name__ == "__main__":
    main()
